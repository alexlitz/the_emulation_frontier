"""Motif-3 against a labelled reference transcription of its remote code (tests/refs/motif3.py: the source repo,
revision, file sha256 and line ranges are in its header). Nothing downloaded is executed: the modelling file
was read as text and written out as plain torch. fp64 throughout (the remote code's fp32 casts lifted), so the
host must agree to 1e-12 relative; each Motif-specific branch, switched off, must move the output.

Run:  pytest tests/test_notremote_motif3.py -q
"""
import pytest
import torch

from tests.refs import motif3 as mm

F64 = torch.float64


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


def _x(d, T=10, seed=3):
    torch.manual_seed(seed)
    return torch.randn(2, T, d, dtype=F64)


@pytest.mark.parametrize("hc,score", [(1e6, "sigmoid"), (0.7, "softmax")], ids=["motif3", "clamped-softmax"])
@pytest.mark.parametrize("kind", ["dense", "moe"])
@torch.no_grad()
def test_ffn_block_carries_motif3_polynorm_notremote(kind, hc, score):
    """Motif-3 FFNs (reference transcription) -> UFFN's PolyNorm-GLU branch via motif_ffn_block: the dense
    MotifMLP (PolyNormTorch, sigmoid weights, hidden clamp, output scale) and the MoE (sigmoid / softmax router,
    top-k on scores + expert_bias, route_norm, route_scale 2; per-expert GroupedPolyNorm with the bias clamp
    and result clamp; a PolyNormTorch shared expert). Motif-3's own config: hidden_clamp 1e6, scale 0.5, bias
    clamp 0.5; the second case clamps hard (0.7) so the clamps are exercised."""
    from elephantine.universal import motif_ffn_block
    cfg = mm.MotifConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
                         num_experts=6, experts_top_k=2, num_shared_experts=1, score_func=score, route_norm=True,
                         route_scale=2.0, hidden_act="poly_norm", polynorm_output_scale=0.5,
                         polynorm_bias_clamp=0.5, hidden_clamp=hc, load_balance_coeff=1e-4)
    torch.manual_seed(0)
    m = (mm.MotifMLP(cfg) if kind == "dense" else mm.MoE(cfg)).to(F64).eval()
    for p in m.parameters():
        p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 1.0))
    if kind == "moe":
        m.expert_bias.copy_(torch.randn_like(m.expert_bias) * 0.1)
    x = _x(32)
    run = lambda: (lambda o: o[0] if isinstance(o, tuple) else o)(m(x))
    ref, f = run(), motif_ffn_block(m, cfg)
    assert ref.dtype == F64 and _rel(ref, f(x)) < 1e-12, _rel(ref, f(x))
    if hc < 1:                                                   # the hidden clamps really bind in the clamped case
        f2 = motif_ffn_block(m, cfg)
        f2.poly_cfg[1] = f2.poly_cfg[4] = 1e6
        assert _rel(ref, f2(x)) > 1e-3
    f.c_poly.zero_()                                             # the PolyNorm-GLU branch really matters
    assert _rel(ref, f(x)) > 1e-1


def _motif_lm():
    cfg = mm.MotifConfig(vocab_size=97, hidden_size=32, intermediate_size=40, num_hidden_layers=4,
        num_attention_heads=6, num_key_value_heads=2, head_dim=12, num_noise_heads=2, q_lora_rank=16, kv_lora_rank=12,
        qk_rope_head_dim=4, v_head_dim=8, attention_cls="gdla", diff_v2=True, elementwise_attn_output_gate=True,
        max_position_embeddings=64, rope_theta=10000.0, swa_rope_theta=500.0, original_seq_len=16, rope_factor=4.0,
        mscale=1.0, rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16,
                                  "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "rope_theta": 10000.0},
        use_sliding_window=True, sliding_window=4, sliding_window_pattern="interleave", sliding_window_period=2,
        num_experts=6, experts_top_k=2, num_shared_experts=1, moe_intermediate_size=16, interleave_moe_layer_step=1,
        n_dense_first_layers=1, score_func="sigmoid", route_norm=True, route_scale=2.0, load_balance_coeff=1e-4,
        hidden_act="poly_norm", polynorm_output_scale=0.5, polynorm_bias_clamp=0.5, hidden_clamp=1e6,
        mhc_enabled=True, mhc_expansion_rate=3, mhc_sinkhorn_iters=20, rms_norm_eps=1e-5)
    torch.manual_seed(0)
    t = mm.MotifForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in t.named_parameters():
            p.add_(torch.randn_like(p) * (0.3 if ("alpha" in n or "bias" in n or "act_fn" in n) else 0.05))
        for l in t.model.layers:
            if l.moe_enabled:
                l.moe.expert_bias.copy_(torch.randn_like(l.moe.expert_bias) * 0.05)
    return t


@torch.no_grad()
def test_same_host_carries_motif3_whole_model_notremote():
    """Motif-3 (reference transcription of MotifForCausalLM), tiny: GDLA attention (q / kv LoRA latents with
    RMSNorms, NoPE + shared-RoPE key split, GQA, v_head_dim < head_dim, groups of 2 signal + 1 noise heads:
    o_i - sigmoid(lambda_proj x)_i o_noise, elementwise sigmoid output gate), YaRN rotary + mscale^2 on full
    layers, interleaved sliding-window layers on their own plain RoPE (swa_rope_theta), a dense PolyNorm-GLU
    layer then sigmoid-routed PolyNorm MoE layers with a shared expert, mHC (3 streams, 20 Sinkhorn steps),
    vs from_motif at fp64. Switching off the differential (noise) term, the output gate, the window, the
    PolyNorm-GLU branch or the hyper-connection mixing each moves the logits."""
    from elephantine.universal import from_motif
    t = _motif_lm()
    L = t.model.layers
    assert [l.self_attn.sliding_window for l in L] == [None, 5, None, 5] and [l.moe_enabled for l in L] == [False, True, True, True]
    assert L[0].self_attn.scaling > L[1].self_attn.scaling            # mscale^2 on the full (YaRN) layers
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    ref = t(x)
    assert ref.dtype == F64
    d = _rel(ref, from_motif(t)(x))
    assert d < 1e-12, d
    offs = {"diff": (lambda l: l.attn.dl_c.zero_(), 1e-2),
            "gate": (lambda l: l.attn.gate_coef.copy_(torch.tensor([1.0, 0.0])), 1e-2),
            "window": (lambda l: l.attn.window.fill_(float("inf")), 1e-2),
            "polynorm": (lambda l: l.ffn.c_poly.zero_(), 1e-2),
            # the streams start as copies of one embedding, so the Sinkhorn mixing moves them less (~5e-4)
            "mhc-sinkhorn": (lambda l: (l.hc_a.sk.zero_(), l.hc_f.sk.zero_()), 1e-4)}
    for name, (off, floor) in offs.items():
        h = from_motif(t)
        for l in h.layers:
            off(l)
        assert _rel(ref, h(x)) > floor, name                           # each branch really matters
