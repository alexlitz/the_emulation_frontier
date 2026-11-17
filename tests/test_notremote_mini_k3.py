"""Mini-K3 (Mamba-2 variants) on the universal host, checked against a REFERENCE transcription of the
remote modelling code (tests/refs/mini_k3.py) instead of executing the Hub files."""
import copy

import pytest
import torch
import torch.nn as nn

from tests.test_universal_elephantine import _ids, _rand_

F64 = torch.float64


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1)


@torch.no_grad()
def test_same_host_carries_mini_k3_notremote():
    """Mini-K3 (Mamba-2 variants): Mamba-2 with grouped gated norm, NoPE MLA with latent RMSNorms,
    SiTU-GLU dense + latent MoE (sigmoid router with selection bias, post-routing norm), Block AttnRes.
    The reference lifts the remote code's fp32 casts to fp64, so the bound is 1e-12 relative.
    Each family branch switched off on the host moves the logits by > 1e-3 relative (SiTU-GLU: > 5e-4,
    its caps at 4 / 25 bind only on the larger pre-activations; measured 7.7e-4), and the reference
    with the same branch switched off agrees with the switched-off host (AttnRes: host margin only, the
    reference has no plain-residual mode)."""
    from tests.refs.mini_k3 import MiniK3Config, MiniK3ForCausalLM
    from elephantine.universal import from_mini_k3
    cfg = MiniK3Config.tiny_for_test()
    torch.manual_seed(0)
    t = MiniK3ForCausalLM(cfg).eval()
    _rand_(t)
    for l in t.model.layers:
        if l.is_moe:
            l.feed_forward.router.e_score_correction_bias.copy_(
                torch.randn_like(l.feed_forward.router.e_score_correction_bias) * 0.3)
    t = t.to(F64)
    assert any(l.is_mamba for l in t.model.layers) and any(l.is_moe for l in t.model.layers)
    assert [l.is_mamba for l in t.model.layers] == [True, True, True, False, False]
    assert [l.is_moe for l in t.model.layers] == [False, True, True, True, True]
    x = _ids(128, T=10)
    ref = t(x).logits
    u = from_mini_k3(t)
    gap = _rel(ref, u(x))
    assert gap <= 1e-12, gap

    Dl = cfg.routed_expert_hidden_size

    def host_off(fn):
        h = copy.deepcopy(u)
        for i, (ul, tl) in enumerate(zip(h.layers, t.model.layers)):
            fn(ul, tl)
        return h

    def ref_off(fn):
        r = copy.deepcopy(t)
        for tl in r.model.layers:
            fn(tl)
        return r(x).logits

    def sel_bias_off(ul, tl):
        if tl.is_moe:
            ul.ffn.sel_bias.zero_()

    def ref_sel_bias_off(tl):
        if tl.is_moe:
            tl.feed_forward.router.e_score_correction_bias.zero_()

    def group_norm_off(ul, tl):                     # per-group gated norm -> one group over d_inner
        if tl.is_mamba:
            ul.ssm.avg.fill_(1 / tl.self_attn.d_inner)

    def ref_group_norm_off(tl):
        if tl.is_mamba:
            tl.self_attn.norm.groups = 1

    def latent_norms_off(ul, tl):                   # MLA q / kv latent RMSNorms -> identity
        if not tl.is_mamba:
            ul.attn.n_avg.zero_(); ul.attn.n_eps.fill_(1.0); ul.attn.n_w.fill_(1.0)

    def ref_latent_norms_off(tl):
        if not tl.is_mamba:
            tl.self_attn.q_a_layernorm = nn.Identity(); tl.self_attn.kv_a_layernorm = nn.Identity()

    def moe_norm_off(ul, tl):                       # latent MoE post-routing RMSNorm -> identity
        if tl.is_moe:
            ul.ffn.r_avg.zero_(); ul.ffn.r_eps.fill_(1.0); ul.ffn.r_w[:Dl] = 1.0

    def ref_moe_norm_off(tl):
        if tl.is_moe:
            tl.feed_forward.routed_expert_norm = nn.Identity()

    def situ_off(ul, tl):                           # SiTU-GLU -> SwiGLU
        ul.ffn.c_situ.zero_()

    def ref_situ_off(tl):                           # caps at 1e9: beta tanh(z / beta) = z to ~1e-18
        for mod in tl.feed_forward.modules():
            if hasattr(mod, "gate_beta"):
                mod.gate_beta = mod.up_beta = 1e9

    def out_gate_off(ul, tl):                       # MLA sigmoid output gate -> constant 1/2
        if not tl.is_mamba:
            ul.attn.gate_coef.copy_(torch.tensor([0.5, 0.0]))

    def ref_out_gate_off(tl):
        if not tl.is_mamba:
            tl.self_attn.g_proj.weight.zero_()

    checks = {
        "router selection bias": (sel_bias_off, ref_sel_bias_off),
        "grouped gated norm": (group_norm_off, ref_group_norm_off),
        "MLA latent norms": (latent_norms_off, ref_latent_norms_off),
        "latent MoE norm": (moe_norm_off, ref_moe_norm_off),
        "SiTU-GLU": (situ_off, ref_situ_off),
        "MLA output gate": (out_gate_off, ref_out_gate_off),
    }
    for name, (hf, rf) in checks.items():
        off = host_off(hf)(x)
        margin = _rel(ref, off)
        assert margin > (5e-4 if name == "SiTU-GLU" else 1e-3), (name, margin)
        agree = _rel(ref_off(rf), off)
        assert agree <= 1e-12, (name, agree)

    h = copy.deepcopy(u)
    h.cfg = copy.copy(h.cfg); h.cfg.attnres_block = 0       # Block AttnRes -> the plain residual stream
    margin = _rel(ref, h(x))
    assert margin > 1e-3, ("Block AttnRes", margin)
