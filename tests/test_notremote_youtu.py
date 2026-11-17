"""Youtu-LLM (tencent/Youtu-LLM-A7B-Base, youtu_vl_moe) on the universal host WITHOUT running its remote code:
the text model is the labelled REFERENCE transcription in tests/refs/youtu.py (read from the snapshot as text),
its chunk_kda the fla-core 0.5.2 torch references, all at fp64.

transformers 5.18's stock `youtu` (dense MLA, no output gate, no KDA, no MoE) and `kimi_linear` (separate
q / k / v convs, low-rank output gate, no lower-bound gate) are not this architecture, so neither is used.

Run:  pytest tests/test_notremote_youtu.py -q
"""
import copy

import pytest
import torch

from tests.test_universal_elephantine import _fla052_kda_reference, _ids

F64 = torch.float64


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1.0)


def _youtu_ref():
    from tests.refs import youtu as Y
    cfg = Y.YoutuVLMoEConfig(vocab_size=97, hidden_size=64, intermediate_size=96,
        moe_intermediate_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, head_dim=16,
        q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        linear_attention_pattern=[1, 1, 1, 0], n_routed_experts=8, num_experts_per_tok=2,
        moe_shared_experts_intermediate_size=32, mrope_section=[2, 1, 1],
        rope_parameters={"rope_type": "default", "rope_theta": 1600000.0, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        kda_use_fla_short_conv=False, kda_use_fused_rmsnorm_gated=False, mla_attn_implementation="eager")
    torch.manual_seed(0)
    t = Y.YoutuVLForConditionalGeneration(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in t.named_parameters():                        # some parameters start as torch.empty
            p.copy_(torch.randn_like(p) * 0.1 + (1.0 if n.endswith(("norm.weight", "layernorm.weight")) else 0.0))
        for mod in t.modules():
            if isinstance(mod, Y.YoutuVLTopkRouter) and mod.qb_beta is not None:
                mod.qb_beta.normal_(0, 0.3)
    return Y, t


@torch.no_grad()
def test_same_host_carries_youtu_kda_mla_notremote():
    """Youtu-LLM text model: KDA (fused qkv, one short conv, low-rank forget gate with the lower-bound
    sigmoid decay, full-rank output gate) + MLA (latent norms, Megatron-interleaved RoPE, head-wise
    gate) + a quantile-balancing sigmoid MoE (selection on logits - bias, modality-aware text row) with a
    shared expert. Reference: tests/refs/youtu.py (the remote modelling file transcribed, its fp32 casts,
    the hard-fp32 router included, lifted to fp64; chunk_kda = fla 0.5.2's torch references at fp64), so
    the bound is 1e-12 relative. Switched off on the host AND the reference the same way: QB selection
    (sel_logit = 0 <-> the reference's sigmoid + bias selection path), the MLA head-wise gate, the KDA
    lower-bound decay, the KDA output gate."""
    from elephantine.universal import from_youtu
    Y, t = _youtu_ref()
    assert [l.is_linear_attn for l in t.model.language_model.layers] == [True, True, True, False]
    assert [isinstance(l.mlp, Y.YoutuVLMoE) for l in t.model.language_model.layers] == [False, True, True, True]
    x = _ids(97)
    ref = t(input_ids=x, use_cache=False).logits
    host = from_youtu(t)
    d = _rel(ref, host(x))
    print("youtu gap", d)
    assert d < 1e-12, d

    def off(host_fn, ref_fn):
        h = from_youtu(t)
        host_fn(h)
        out = h(x)
        moved = _rel(ref, out)
        assert moved > 1e-3, moved
        t2 = copy.deepcopy(t)
        ref_fn(t2)
        r2 = t2(input_ids=x, use_cache=False).logits
        agree = _rel(r2, out)
        assert agree < 1e-12, agree
        return moved, agree

    def ref_sel_off(m):                                          # select on sigmoid(logits) + bias instead
        for mod in m.modules():
            if isinstance(mod, Y.YoutuVLTopkRouter):
                mod.use_quantile_balancing = False
                mod.e_score_correction_bias = -mod.qb_beta
                mod.qb_beta = None

    def host_sel_off(h):
        for l in h.layers:
            l.ffn.sel_logit.zero_()

    def ref_mla_gate_off(m):
        for l in m.model.language_model.layers:
            if not l.is_linear_attn:
                l.self_attn.g_proj = None

    def host_mla_gate_off(h):
        for l, tl in zip(h.layers, t.model.language_model.layers):
            if not tl.is_linear_attn:
                l.attn.gate_coef.copy_(torch.tensor([1.0, 0.0]))

    def ref_decay_off(m):                                        # lower_bound 0: log decay 0 (safe_gate off)
        for l in m.model.language_model.layers:
            if l.is_linear_attn:
                l.self_attn.kda_gate_lower_bound, l.self_attn.safe_gate = 0.0, False

    def host_decay_off(h):
        for l, tl in zip(h.layers, t.model.language_model.layers):
            if tl.is_linear_attn:
                l.ssm.A2.zero_()

    def ref_ogate_flat(m):                                       # output gate input zeroed: sigmoid(0) = 1/2
        for l in m.model.language_model.layers:
            if l.is_linear_attn:
                l.self_attn.g_proj.weight.zero_()

    def host_ogate_flat(h):
        for l, tl in zip(h.layers, t.model.language_model.layers):
            if tl.is_linear_attn:
                l.ssm.pz.weight.zero_()

    for name, hf, rf in (("sel_logit", host_sel_off, ref_sel_off), ("mla_gate", host_mla_gate_off, ref_mla_gate_off),
                         ("kda_decay", host_decay_off, ref_decay_off), ("kda_out_gate", host_ogate_flat, ref_ogate_flat)):
        moved, agree = off(hf, rf)
        print(name, "moved", moved, "ref agrees", agree)

    # the transcribed chunk_kda against fla 0.5.2's own torch references (fp32 inside): whole-model fp32 level
    try:
        lib = _fla052_kda_reference()
    except pytest.skip.Exception:
        return
    saved = Y.chunk_kda
    try:
        Y.chunk_kda = lib
        ref_lib = t(input_ids=x, use_cache=False).logits.double()
    finally:
        Y.chunk_kda = saved
    d_lib = _rel(ref, ref_lib)
    print("fla-library chunk_kda gap", d_lib)
    assert d_lib < 1e-5, d_lib
