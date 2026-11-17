"""verify/lipschitz/measure.py on a tiny random Llama (CPU, seconds): the Jacobian norms against the exact
Jacobian, the closed-form head constant against autograd, Lean's errBound, and the whole pipeline (the measured-
constant bound sits above the error the softcap / mask / sink / lever hosts actually make)."""
import math

import pytest
import torch

from act.act_swap_agreement import greedy
from verify.lipschitz import measure as M


@pytest.fixture(scope="module")
def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_num_threads(min(4, torch.get_num_threads()))      # tiny tensors: more threads only contend
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                      rms_norm_eps=1e-5, tie_word_embeddings=False, attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
            else:
                p.mul_(3.0)                                  # O(1) logits, non-trivial attention
    for p in model.parameters():
        p.requires_grad_(False)
    M.S.fam, M.S.cfg, M.S.idx, M.S.idx_moe = "llama", cfg, {}, {}
    M.install_attention(model, "llama")
    return model


def _layer_input(model, hooks, seq):
    _, _, caps = M.capture(model, hooks, seq, keep_inputs=True)
    return caps


def test_jacobian_norms_match_exact(tiny):
    hooks = M.Hooks(tiny, "llama")
    try:
        seq = torch.randint(0, 97, (1, 9))
        caps = _layer_input(tiny, hooks, seq)
        layer, cap = hooks.layers[1], caps[1]
        X = cap[0][0] if cap[0] else cap[1]["hidden_states"]
        T, D = X.shape[1], X.shape[2]
        for zero in (None, ("mlp", 1)):                        # block, attention residual map
            f = M._layer_fn(layer, cap, zero)
            J = torch.autograd.functional.jacobian(lambda x: f(x[None])[0], X[0]).reshape(T, D, T * D)
            pos = [0, 3, T - 1]
            spec, mix, _ = M.per_token_norms(f, X, pos, n_l2=60, n_mix=30)
            for k, t in enumerate(pos):
                exact = torch.linalg.matrix_norm(J[t], ord=2)
                assert abs(spec[k] - exact) <= 1e-3 * exact, (zero, t, spec[k], exact)
                # the mixed (inf,2) norm: >= the l2 one, <= sum over positions of the blocks' norms
                upper = sum(torch.linalg.matrix_norm(J[t][:, s * D:(s + 1) * D], ord=2) for s in range(T))
                assert spec[k] - 1e-5 <= mix[k] <= upper + 1e-4, (zero, t, mix[k], upper)
        f = M._layer_fn(layer, cap, ("attn", 1))               # the FFN residual map: per position
        s, _ = M.per_position_norms(f, X, n_it=60)
        J = torch.autograd.functional.jacobian(lambda x: f(x[None])[0], X[0])
        for t in range(T):
            exact = torch.linalg.matrix_norm(J[t, :, t, :], ord=2)
            assert abs(s[t] - exact) <= 1e-3 * exact
            assert J[t, :, (t + 1) % T, :].abs().max() == 0           # no cross-position terms
    finally:
        hooks.remove()


@pytest.mark.parametrize("kind", ["rms", "ln"])
def test_head_constant_closed_form(kind):
    torch.manual_seed(1)
    d, V = 16, 50
    W = torch.randn(V, d)
    if kind == "rms":
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        nm = LlamaRMSNorm(d, eps=1e-5)
    else:
        nm = torch.nn.LayerNorm(d, eps=1e-5)
        with torch.no_grad():
            nm.bias.uniform_(-1, 1)
    with torch.no_grad():
        nm.weight.uniform_(0.5, 1.5)
    X = torch.randn(5, d) * 3
    got = M.head_constants(W, kind, nm.weight.detach(), 1e-5, X)
    for i in range(5):
        ref = M.head_constant_autograd(nm, W, X[i])
        assert abs(got[i] - ref) <= 1e-4 * ref, (i, got[i], ref)


def test_err_bound_forms():
    L = [1.5, 3.0, 0.8, 2.0]
    k = [0.1, 2.0, 0.5, 1.0]
    eb = M.err_bound(L, k)
    assert math.isclose(math.exp(M.log_err_bound(L, [math.log(x) for x in k])), eb, rel_tol=1e-12)
    prod = math.prod(max(l, 1) for l in L) * sum(k)
    assert eb <= prod
    th = M.log_threshold(0.7, L, [math.log(x) for x in k], 0.3)
    assert math.isclose(2 * 0.7 * prod * math.exp(th["logg_prod"]), 0.3, rel_tol=1e-9)
    assert th["logg_sharp"] >= th["logg_prod"]


def test_pipeline_bound_above_observed(tiny, tmp_path):
    torch.manual_seed(2)
    refs = []
    for _ in range(3):
        ids = torch.randint(0, 97, (1, 5))
        refs.append((torch.cat([ids, greedy(tiny, ids, n_new=12)], 1), 5))
    res = M.measure(tiny, "llama", refs, "tiny", device="cpu", jac_prompts=2, stride=2, chunk=8, n_l2=15, n_mix=8,
                    out_dir=str(tmp_path), log=lambda *a: None)
    assert (tmp_path / "tiny.json").exists()
    assert res["Lh"]["max"] > 0 and res["margins"]["nonpositive"] == 0
    for l in res["layers"]:
        assert l["jac"]["block_mix"]["max"] >= l["jac"]["block_spec"]["max"] - 1e-6
        assert l["rec"]["zmax"] > 0 and l["rec"]["M"] > 0
    rules = res["rules"]
    assert {"softcap_host", "mask", "sink", "lever_qk", "lever_post"} <= set(rules)
    # thresholds move the right way with the margin
    for name in ("softcap_host", "mask", "lever_qk", "lever_post"):
        assert rules[name]["param_star_prod_median_prompt"] <= rules[name]["param_star_prod"] + 1e-9
    assert rules["sink"]["param_star_prod_median_prompt"] >= rules["sink"]["param_star_prod"] - 1e-9
    # the bound with measured local constants is above the error each host makes (it is a bound)
    n = 0
    for name, rows in res["observed"].items():
        for row in rows:
            if row["max_err"] > 0:
                assert row["log10_bound_sharp"] >= math.log10(row["max_err"]), (name, row)
                assert row["log10_bound_prod"] >= row["log10_bound_sharp"] - 1e-9
                n += 1
            if row["param"] in (rules[name]["setting"],) and math.isfinite(row["param"]):
                assert row["top1"] == 1.0
    assert n > 10
    txt = M.report(str(tmp_path))
    assert "softcap_host" in txt and "tiny" in txt
