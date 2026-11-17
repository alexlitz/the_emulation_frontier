"""AliceAI (KDA + gated attention, sigmoid top-k router with bias-corrected selection, Block AttnRes) on the
universal host, against a labelled REFERENCE transcription of the remote modelling code (tests/refs/alice_ai.py:
read as text, never run; the remote file imports no model classes from transformers, so it is transcribed
whole). The remote-code version (tests/test_universal_elephantine.py
test_same_host_carries_aliceai_attnres_sigmoid_router) stays remote-gated. fp64 throughout; each family branch
switched off on the host agrees with the reference switched off the same way, and moves the output.

Run:  pytest tests/test_notremote_alice.py -q
"""
import copy

import torch

from tests.refs import alice_ai as R

F64 = torch.float64


def _rand_(model, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def _ids(vocab, T=10, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (2, T))


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1.0)


def _alice():
    """The old test's config, init (post_init's _init_weights, then rand_) and selection-bias randomisation."""
    cfg = R.AliceAIConfig(vocab_size=97, hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, num_experts=6, num_experts_per_tok=2,
        moe_intermediate_size=24, shared_expert_intermediate_size=32, block_attn_res_block_size=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention",
                     "linear_attention", "full_attention"])
    t = _rand_(R.init_weights_(R.AliceAIForCausalLM(cfg)).to(F64).eval())
    with torch.no_grad():                                   # make the selection bias matter
        for l in t.model.layers:
            l.mlp.gate.e_score_correction_bias.copy_(torch.randn(6, dtype=F64) * 0.3)
    return t


@torch.no_grad()
def test_same_host_carries_aliceai_attnres_sigmoid_router_notremote():
    """Gap bound 1e-12 relative: the reference is fp64-clean (its fp32 casts lifted)."""
    from elephantine.universal import from_alice
    t = _alice()
    kinds = [l.layer_type for l in t.model.layers]
    assert {"linear_attention", "full_attention"} <= set(kinds), kinds
    x = _ids(97)
    ref = t(x, use_cache=False).logits
    u = from_alice(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))

    def check(name, t_off, host_off):
        off = t_off(x).logits
        assert _rel(ref, off) > 1e-3, (name, _rel(ref, off))
        assert _rel(off, host_off(x)) < 1e-12, (name, _rel(off, host_off(x)))

    # router selection bias off: reference bias zeroed  vs  host sel_bias zeroed
    t2 = copy.deepcopy(t)
    for l in t2.model.layers:
        l.mlp.gate.e_score_correction_bias.zero_()
    h = copy.deepcopy(u)
    for ul in h.layers:
        ul.ffn.sel_bias.zero_()
    check("selection bias", t2, h)

    # AttnRes depth-softmax queries off (a uniform mean over block sums): reference res_proj zeroed  vs  host ar_q*
    t2 = copy.deepcopy(t)
    for l in t2.model.layers:
        if hasattr(l, "attn_res_proj"):
            l.attn_res_proj.weight.zero_()
        l.mlp_res_proj.weight.zero_()
    t2.model.attnres_final.res_proj.weight.zero_()
    h = copy.deepcopy(u)
    for ul in h.layers:
        ul.ar_qa.zero_(); ul.ar_qf.zero_()
    h.ar_q_out.zero_()
    check("attnres queries", t2, h)

    # AttnRes blocks: one block spanning all 6 layers instead of blocks of 2 (both sides read the config)
    t2 = copy.deepcopy(t)
    t2.config.block_attn_res_block_size = 6
    check("attnres block size", t2, from_alice(t2))

    # gated attention's output gate held at sigmoid(0) = 1/2: reference gate rows of q_proj zeroed  vs  host g_proj
    t2 = copy.deepcopy(t)
    hd, H = 16, 4
    for l in t2.model.layers:
        if l.layer_type == "full_attention":
            l.self_attn.q_proj.weight.view(H, 2 * hd, -1)[:, hd:].zero_()
    h = copy.deepcopy(u)
    for ul, l in zip(h.layers, t.model.layers):
        if l.layer_type == "full_attention":
            ul.attn.g_proj.weight.zero_()
    check("attention output gate", t2, h)

    # KDA beta range: allow_negative_eigenvalues (beta * 2) switched on, both sides read the config
    t2 = copy.deepcopy(t)
    t2.config.kda_allow_negative_eigenvalues = True
    check("kda negative eigenvalues", t2, from_alice(t2))
