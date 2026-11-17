"""Reproductions of the confirmed findings of the correctness review (verify/REVIEW.md, branch code-review e620874),
as passing tests asserting the correct behaviour (each was an xfail(strict=True) there). Findings 1, 2, 6, 7, 8 were
fixed in round 2, pass 5; 3, 4, 5, 7 (the transformers version), 9 and 10 on branch tooling-fixes, with the
Lean-bridge MISMATCHES 1 and 2 of verify/BRIDGES.md and two of the review's unconfirmed items, at the end. The
numbering follows verify/REVIEW.md."""
import functools
import os

import pytest

from graph import model_graph as G

HUB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph", "hub_10k.json")
needs_hub = pytest.mark.skipif(not os.path.exists(HUB), reason="no zoo snapshot")


@functools.lru_cache(maxsize=None)
def _hub():
    return dict(G.iter_json_configs(HUB))


def _pt(cfg, name="x"):
    return {"slots": G.classify(cfg), "dims": G.dims_of(cfg), "members": [name]}


def _mamba(hidden, layers, state):
    """A stock MambaForCausalLM config (no per-layer list, as every Mamba-1 checkpoint ships)."""
    return dict(model_type="mamba", architectures=["MambaForCausalLM"], hidden_size=hidden, num_hidden_layers=layers,
                intermediate_size=2 * hidden, state_size=state, conv_kernel=4, time_step_rank="auto",
                layer_norm_epsilon=1e-5, vocab_size=50280, tie_word_embeddings=True)


# --- 2. pure recurrent stacks counted as full-attention layers ------------------------------------------------

def test_pure_ssm_layers_are_recurrent_not_full_attention():
    d = G.dims_of(_mamba(768, 24, 16))
    assert d["kR_eff"] == 24 and d["kR_param"] == 24
    assert d["kFU_eff"] == 0 and d["attn_eff"] == 0 and d["full_eff"] == 0


@needs_hub
def test_pure_ssm_snapshot_models_have_no_attention_layers():
    for name in ("state-spaces/mamba-130m-hf", "tiiuae/falcon-mamba-7b", "RWKV/rwkv-4-169m-pile",
                 "fla-hub/gla-1.3B-100B", "fla-hub/rwkv7-0.1B-g1"):
        d = G.dims_of(_hub()[name])
        assert d["attn_eff"] == 0 and d["kFU_eff"] == 0 and d["kR_eff"] == d["eff_layers"], name


# --- 8. mode="mixgap" skipped the recurrent capacity check for attention-free hosts -----------------------------

def test_mixgap_pure_ssm_host_still_needs_the_recurrent_dims():
    t, h = _pt(_mamba(768, 24, 32), "t"), _pt(_mamba(1024, 48, 16), "h")
    assert not G._rec_ok([t, h])[0, 1]                     # the host's state is too small ...
    assert not G.emulates(h, t, mode="payable")            # ... which the headline sees
    assert not G.emulates(h, t, mode="mixgap")             # ... and mixgap does too


# --- 6. ALiBi <- learned-abs: the 'width' currency was never charged; round 2, pass 5 walls the edge ---------------
# (the ramp channel is read through every pre-norm: refuted as exact, proofs/PROOFS.md "Pre-norm rescue
# attempts"; no norm-free host is in the zoo)

def test_alibi_on_learned_abs_host_is_a_wall():
    assert not any(p == "learned_abs" for p, _, _ in G.POS_ORDER["alibi"])
    common = dict(n_layer=24, n_head=16, hidden_size=1024, n_embd=1024, layer_norm_epsilon=1e-5, vocab_size=50257,
                  tie_word_embeddings=True)
    bloom = _pt(dict(common, model_type="bloom", architectures=["BloomForCausalLM"]), "bloom")
    gpt2 = _pt(dict(common, model_type="gpt2", architectures=["GPT2LMHeadModel"], n_positions=2048,
                    activation_function="gelu_new", n_inner=8192), "gpt2")
    assert bloom["slots"]["pos"] == "alibi" and gpt2["slots"]["pos"] == "learned_abs"
    assert not G.emulates(gpt2, bloom)
    big = _pt(dict(common, model_type="gpt2", architectures=["GPT2LMHeadModel"], n_positions=2048, n_embd=4096,
                   hidden_size=4096, n_layer=48, n_head=32, activation_function="gelu_new"), "gpt2-big")
    assert not G.emulates(big, bloom)                      # a wall, not a width charge


@needs_hub
def test_alibi_on_learned_abs_host_snapshot_pair():
    bloom, sour = _pt(_hub()["bigscience/bloom-560m"]), _pt(_hub()["SourAsslips/Sour"])
    assert sour["dims"]["hidden"] == bloom["dims"]["hidden"]
    assert not G.emulates(sour, bloom)


# --- 1. post_numbers counted block-causal frontier models as autoregressive ---------------------------------------

@needs_hub
def test_post_numbers_ar_split_matches_frontier_md():
    from graph.frontier import frontier
    from graph.post_numbers import GRAPH_DEPS, cached, graph_section
    g = cached("graph", GRAPH_DEPS, graph_section)
    rows, *_ = frontier(mode="payable")
    assert g["frontier_len"] - g["frontier_diffusion"] == sum(r[2]["gen"] == "ar" for r in rows)


# --- 7. post_numbers' elephantine cache key missed a file it reads ------------------------------------------------

def test_eleph_cache_key_covers_the_stock_loader_tests():
    import glob
    import inspect

    import elephantine.loader_coverage as lc
    from graph.post_numbers import ELEPH_DEPS, ROOT
    assert "tests.test_stock_loaders" in inspect.getsource(lc._stock_tests)
    files = {os.path.relpath(f, ROOT) for p in ELEPH_DEPS for f in glob.glob(os.path.join(ROOT, p))}
    assert "tests/test_stock_loaders.py" in files


def test_eleph_cache_key_covers_the_transformers_version(monkeypatch):
    import transformers

    from graph.post_numbers import ELEPH_DEPS, _fingerprint, elephantine_section
    a = _fingerprint(ELEPH_DEPS, elephantine_section)
    monkeypatch.setattr(transformers, "__version__", transformers.__version__ + ".post-test")
    assert _fingerprint(ELEPH_DEPS, elephantine_section) != a


# --- the dims the relation compares are all in the dedup key (REVIEW "whole-relation checks") --------------------

def test_mla_branch_dims_are_in_the_dedup_key():
    kd = G._point_key_dims()
    assert "mla_qswitch" in kd and "mla_lat_free" in kd and "kv_shapes" in kd


# --- REVIEW suspicious 3: per-layer KV-head shapes --------------------------------------------------------------

@needs_hub
def test_per_layer_kv_shapes_are_read():
    d = G.dims_of(_hub()["XiaomiMiMo/MiMo-V2.5"]) if "XiaomiMiMo/MiMo-V2.5" in _hub() else None
    if d is None:
        pytest.skip("MiMo-V2.5 not in the snapshot")
    assert len(d["kv_shapes"]) >= 2                       # the sliding layers' 8 x 8 beside the full layers'


# --- 3. check_post: a round number in the draft passes for a wrong value ----------------------------------------

def test_check_post_approx_rejects_wrong_round_numbers():
    from graph.check_post import compare
    assert not compare({"kind": "approx", "value": 89.9}, "80")[0]
    assert not compare({"kind": "approx", "value": 199}, "100")[0]
    assert not compare({"kind": "approx", "value": 2999}, "2,000")[0]
    assert not compare({"kind": "approx", "value": 96.55}, "96")[0]        # an integer must round: 97


def test_check_post_approx_still_accepts_rounding_and_decimal_truncation():
    from graph.check_post import compare
    assert compare({"kind": "approx", "value": 89.9}, "90")[0]
    assert compare({"kind": "approx", "value": 2546}, "2,550")[0]
    assert compare({"kind": "approx", "value": 2546}, "2,500")[0]               # rounded at its trailing zeros
    assert not compare({"kind": "approx", "value": 2556}, "2,500")[0]           # ... but rounded, not truncated
    assert compare({"kind": "approx", "value": 2960}, "3,000")[0]
    assert compare({"kind": "approx", "value": 1.48}, "1.4")[0] and compare({"kind": "approx", "value": 1.48}, "1.5")[0]
    assert not compare({"kind": "approx", "value": 1.48}, "1.3")[0]


# --- 4. check_post: a model name passes as any token subset of an alias -----------------------------------------

def test_check_post_name_rejects_a_sibling_model():
    from graph.check_post import compare
    assert not compare({"kind": "name", "value": "MiMo-V2.5-Pro", "aliases": ["MiMo-V2.5-Pro"]}, "MiMo-V2.5")[0]
    assert not compare({"kind": "name", "value": "Kimi-K3.1-Pro", "aliases": ["Kimi-K3.1-Pro"]}, "Pro")[0]


def test_check_post_name_accepts_a_whole_alias():
    from graph.check_post import compare
    e = {"kind": "name", "value": "Qwen-72B", "aliases": ["Qwen/Qwen-72B", "Qwen-72B-Chat"]}
    assert compare(e, "**Qwen-72B**")[0] and compare(e, "qwen 72b")[0] and compare(e, "Qwen/Qwen-72B")[0]
    assert compare(e, "the chat model (Qwen-72B-Chat)")[0]
    assert not compare(e, "Qwen-72B-Instruct")[0] and not compare(e, "Qwen")[0]


# --- 9. universal.from_gemma caps Gemma-3's attention, which transformers never does ---------------------------

def test_from_gemma3_does_not_cap_attention_the_target_does_not_cap():
    import torch
    import transformers as tr

    import elephantine.universal as U
    from tests.test_universal_elephantine import _ids, _lift_fp32, _rand_
    cfg = tr.Gemma3TextConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3,
                              num_attention_heads=6, num_key_value_heads=2, head_dim=16, sliding_window=4,
                              attn_logit_softcapping=1.0, attn_implementation="eager")
    torch.manual_seed(0)
    with torch.no_grad():
        t = _rand_(tr.Gemma3ForCausalLM(cfg).to(torch.float64).eval())
        x = _ids(97, T=12)
        out = U.from_gemma(t)(x)
        out = out.logits if hasattr(out, "logits") else out
        with _lift_fp32():
            ref = t(x).logits
    assert (ref - out).abs().max() < 1e-12 * ref.abs().max()


# --- 10. edge_plan: the TransMLA note is missing for MHA / no-KV hosts -------------------------------------------

def test_edge_plan_transmla_note_for_every_non_mla_host():
    from graph.edge_plan import axis_rows
    slots = {"kv": "MLA", "ffn": "glu", "block": "sequential", "norm": "RMSNorm", "mixer": "softmax",
             "crosslayer_kv": "no", "window": "full"}
    dims = dict(kv_cap=576, q_cap=4096, qheads=32, ffn_total=1000, ffn_active=1000, eff_layers=10,
                param_layers=10, hidden=1024, head_dim=128)
    A = {"slots": slots, "dims": dims}
    B = {"slots": dict(slots, kv="MHA"), "dims": dict(dims, kv_cap=4096)}
    row = next(r for r in axis_rows(A, B) if r["axis"] == "kv_cap")
    assert row["need"] == 2 * 576
    assert "TransMLA" in row["note"]


# --- 5. elephantine.emulate: a Qwen-MoE host's own sliding window is ignored ------------------------------------

@pytest.mark.parametrize("hmt,extra", [("qwen3_moe", {}),
                                       ("qwen2_moe", dict(head_dim=16, shared_expert_intermediate_size=16))])
def test_qwen_moe_host_with_sliding_window_is_built_exactly(hmt, extra):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from elephantine import emulate as E
    from tests.test_universal_elephantine import _lift_fp32

    def raw(mt, **kw):
        c = AutoConfig.for_model(mt, vocab_size=512, max_position_embeddings=256, **kw)
        d = c.to_dict()
        d["model_type"] = mt
        d["architectures"] = [AutoModelForCausalLM._model_mapping[type(c)].__name__]
        return d
    tmt = "qwen3" if hmt == "qwen3_moe" else "llama"
    traw = raw(tmt, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16)
    hraw = raw(hmt, **{**dict(hidden_size=96, intermediate_size=160, num_hidden_layers=4, num_attention_heads=6,
                              num_key_value_heads=3, head_dim=32, moe_intermediate_size=112, num_experts=4,
                              num_experts_per_tok=2, use_sliding_window=True, sliding_window=4,
                              max_window_layers=2), **extra})
    plan, con = E.gate(traw, hraw)                       # the relation and v1 both accept the pair
    tc = E.resolved(traw)
    torch.manual_seed(0)
    tm = AutoModelForCausalLM.from_config(tc, attn_implementation="eager", torch_dtype=torch.float64).eval()
    with torch.no_grad():
        for p in tm.parameters():
            p.add_(torch.randn_like(p) * 0.08)
    hd = E.host_config_dict(hraw, con)
    sd = E.build_host(E.read_target(tm, tc, con.tfam), tc, hd, con, log=lambda *a: None)
    hm = AutoModelForCausalLM.from_config(E.resolved(hd), attn_implementation="eager", torch_dtype=torch.float64).eval()
    hm.load_state_dict({k: v.to(torch.float64) for k, v in sd.items()})
    x = torch.randint(0, 512, (1, 24))
    with torch.no_grad(), _lift_fp32():
        a, b = tm(input_ids=x).logits, hm(input_ids=x).logits
    assert ((a - b).abs().max() / a.abs().max()).item() < 1e-12


def test_qwen_moe_host_window_is_switched_off_or_refused():
    """REVIEW 5, the plan side: a Qwen-MoE host whose config turns its window on gets use_sliding_window False
    for a full-window target; a sliding-window target is the relation's mask edge (emulate v3: the script sets no
    window on these families, so the host's own goes off and the window is the caller's mask)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from elephantine import emulate as E

    def raw(mt, **kw):
        c = AutoConfig.for_model(mt, vocab_size=512, max_position_embeddings=256, **kw)
        d = c.to_dict()
        d["model_type"] = mt
        d["architectures"] = [AutoModelForCausalLM._model_mapping[type(c)].__name__]
        return d
    hraw = raw("qwen3_moe", hidden_size=96, intermediate_size=160, num_hidden_layers=4, num_attention_heads=6,
               num_key_value_heads=3, head_dim=32, moe_intermediate_size=112, num_experts=4, num_experts_per_tok=2,
               use_sliding_window=True, sliding_window=4)
    assert E._windows(E.resolved(hraw), E.FAMILIES["qwen3_moe"]) == [4] * 4     # the stock class slides every layer
    full = raw("qwen3", hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16)
    con = E.plan_construction(full, hraw)
    assert con.config_changes["use_sliding_window"] is False
    assert E.resolved(E.host_config_dict(hraw, con)).sliding_window is None
    off = dict(hraw, use_sliding_window=False)
    assert "use_sliding_window" not in E.plan_construction(full, off).config_changes
    slide = raw("qwen3", hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, sliding_window=4, use_sliding_window=True,
                layer_types=["sliding_attention"] * 2)
    con = E.plan_construction(slide, hraw)
    assert con.extras["mask_window"] == 4 and con.config_changes["use_sliding_window"] is False
    mixed = raw("qwen3", hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, sliding_window=4, use_sliding_window=True,
                layer_types=["sliding_attention", "full_attention"])
    with pytest.raises(E.Refused, match="window"):
        E.plan_construction(mixed, hraw)


# --- unconfirmed items checked and fixed --------------------------------------------------------------------------

def test_universal_defines_ln_from_once():
    """REVIEW suspicious 9: _ln_from was defined twice; the later one (no bias-free LayerNorm) silently won."""
    import ast
    import inspect

    import torch

    import elephantine.universal as U
    tree = ast.parse(inspect.getsource(U))
    assert sum(isinstance(n, ast.FunctionDef) and n.name == "_ln_from" for n in tree.body) == 1
    ln, dst = torch.nn.LayerNorm(8, bias=False), U.UNorm(8)
    with torch.no_grad():
        U._ln_from(dst, ln)
    assert float(dst.centre) == 1.0 and not dst.bias.any()


@needs_hub
def test_normalize_is_idempotent_for_lfm2():
    """REVIEW suspicious 6: LFM2 configs without block_ff_dim had the 2/3 auto-adjust applied again on a second
    _normalize pass. dims_of(raw) is unchanged by the fix."""
    n = 0
    for name, c in _hub().items():
        if isinstance(c, dict) and str(c.get("model_type", "")).startswith("lfm2"):
            once = G._lm_config(c)
            assert G._normalize(once) == once, name
            assert G.dims_of(once) == G.dims_of(c), name
            n += 1
    assert n


# --- verify/BRIDGES.md MISMATCHES 1 and 2 (branch lean-bridges): the fixed constructions are exact ---------------

def _neox_random_ln(seed, layers=2, d=32, parallel=True):
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=61, hidden_size=d, num_hidden_layers=layers, num_attention_heads=4,
                        intermediate_size=2 * d, max_position_embeddings=64, rotary_pct=0.5,
                        use_parallel_residual=parallel, attn_implementation="eager")
    m = GPTNeoXForCausalLM(cfg).to(torch.float64).eval()
    with torch.no_grad():                         # a real gamma and beta on every LayerNorm
        for mod in m.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.weight.copy_(1 + 0.3 * torch.randn_like(mod.weight))
                mod.bias.copy_(0.3 * torch.randn_like(mod.bias))
    return m


@pytest.mark.parametrize("seed,parallel", [(0, True), (1, True), (2, False), (3, False)])
def test_bridge_layerNorm_sub_rmsNorm_convert_pythia(seed, parallel):
    """EP.layerNorm_sub_rmsNorm on convert/layernorm_to_rmsnorm.convert_pythia (BRIDGES MISMATCH 1): after the
    conversion no LayerNorm is left, the residual stream is P times the target's at every layer, and the
    logits agree to float64 rounding (the old converter dropped the centring: a gap > 1e-4)."""
    import torch

    from convert.layernorm_to_rmsnorm import convert_pythia
    m0, m1 = _neox_random_ln(seed, parallel=parallel), _neox_random_ln(seed, parallel=parallel)
    info = convert_pythia(m1)
    assert info["writers"] > 0 and info["embeddings"] >= 1
    assert not any(isinstance(x, torch.nn.LayerNorm) for x in m1.modules())
    ids = torch.randint(0, 61, (2, 12), generator=torch.Generator().manual_seed(seed + 1))
    with torch.no_grad():
        o0, o1 = m0(ids, output_hidden_states=True), m1(ids, output_hidden_states=True)
    for h0, h1 in zip(o0.hidden_states[:-1], o1.hidden_states[:-1]):
        assert (h1 - (h0 - h0.mean(-1, keepdim=True))).abs().max() <= 1e-10 * (1 + h0.abs().max().item())
    assert (o0.logits - o1.logits).abs().max() <= 1e-10 * (1 + o0.logits.abs().max().item())


@pytest.mark.parametrize("n,m", [(1, 2), (2, 4), (1, 4), (2, 8), (3, 6), (3, 4), (2, 3), (3, 5), (5, 7), (4, 4)])
@pytest.mark.parametrize("cols", [1, 5, 12, 30])
def test_bridge_dense_sub_nm(n, m, cols):
    """EP.dense_sub_nm on convert/nm_sparsity.dense_to_wide_nm (BRIDGES MISMATCH 2), N | M or not, and any
    column count: S is N:M sparse (at most N nonzeros in every group of M), ceil(cols / N) groups wide, holds
    every dense weight once, and S @ expand(x) == W @ x to float64 rounding."""
    import torch

    from convert.nm_sparsity import dense_to_wide_nm, max_nonzeros_per_group
    g = torch.Generator().manual_seed(1000 * n + 10 * m + cols)
    W = torch.randn(7, cols, dtype=torch.float64, generator=g)
    x = torch.randn(3, cols, dtype=torch.float64, generator=g)
    S, expand = dense_to_wide_nm(W, n, m)
    assert S.shape[1] == -(-cols // n) * m
    assert max_nonzeros_per_group(S, m) <= n
    assert int((S != 0).sum()) == int((W != 0).sum())
    bound = 64 * cols * 2.0 ** -53 * (W.abs().sum(1).max() * x.abs().max())
    assert (expand(x) @ S.T - x @ W.T).abs().max() <= bound
