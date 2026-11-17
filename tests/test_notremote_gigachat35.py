"""GigaChat-3.5 against a labelled reference transcription of its remote code (tests/refs/gigachat35.py: the source
repo, revision, file sha256 and line ranges are in its header). Nothing downloaded is executed: the modelling
file was read as text and written out as plain torch (the rotary table is the stock transformers yarn init).
fp64 throughout (the remote code's fp32 casts lifted), so the host must agree to 1e-12 relative; each
GigaChat-specific branch, switched off, must move the output.

Run:  pytest tests/test_notremote_gigachat35.py -q
"""
import torch

from tests.refs import gigachat35 as gm

F64 = torch.float64


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


def _gigachat35():
    cfg = gm.GigaChat35Config(vocab_size=97, hidden_size=64, intermediate_size=48, moe_intermediate_size=16,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=6,
        routed_scaling_factor=2.5, kv_lora_rank=24, q_lora_rank=32, qk_rope_head_dim=8, v_head_dim=12,
        qk_nope_head_dim=16, n_group=1, topk_group=1, num_experts_per_tok=2, first_k_dense_replace=1,
        norm_topk_prob=True, rms_norm_eps=1e-6, rope_theta=10000.0, rope_interleave=True, max_position_embeddings=512,
        rope_scaling={"beta_fast": 32, "beta_slow": 1, "factor": 8.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                      "original_max_position_embeddings": 64, "type": "yarn"},
        norm_type="ZeroCenteredGatedNorm", layernorm_type="pre_post", layernorm_gating_weight=2.0,
        gated_attention=True, use_mla_scaling_factor=True, linear_attention_type="GigaChat35GatedDeltaNet",
        full_attention_layers=[1, 3], linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_gating_type="gated_rmsnorm_sigmoid_zero_centered",
        linear_sigmoid_gate_scale=2.0, linear_attn_o_norm_eps=1e-6, swiglu_limit=0.08)
    torch.manual_seed(0)
    t = gm.GigaChat35ForCausalLM(cfg).to(F64).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in t.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for l in t.model.layers:                                 # make the selection bias matter
            if hasattr(l.mlp, "gate"):
                l.mlp.gate.e_score_correction_bias.copy_(torch.randn(6, dtype=F64) * 0.3)
    return t


@torch.no_grad()
def test_same_host_carries_gigachat35_notremote():
    """GigaChat-3.5 (reference transcription): Qwen3-Next-layout gated-DeltaNet with a zero-centered sigmoid gated
    norm (x2), gated MLA (latent norms, interleaved RoPE + YaRN, alpha scaling), sandwich zero-centered GATED norms
    (rank-16 sigmoid self-gate), sigmoid router + selection bias, hard SwiGLU clipping, vs from_gigachat35 at fp64.
    Switching off the SwiGLU clip, the norm self-gates, the attention output gate, the selection bias or the
    post (sandwich) norms each moves the logits."""
    from elephantine.universal import from_gigachat35
    t = _gigachat35()
    assert [l.layer_type for l in t.model.layers] == ["linear_attention", "full_attention"] * 2
    x = torch.randint(0, 97, (2, 10), generator=torch.Generator().manual_seed(0))
    ref = t(x)
    assert ref.dtype == F64
    d = _rel(ref, from_gigachat35(t)(x))
    assert d < 1e-12, d
    offs = {"swiglu-clip": lambda l: l.ffn.c_clip.zero_(),
            "norm-self-gate": lambda l: [n.zero_() for n in (l.norm_a.gate_c, l.norm_s.gate_c, l.norm_f.gate_c, l.attn.n_gate_c)],
            "attn-gate": lambda l: l.attn.gate_coef.copy_(torch.tensor([1.0, 0.0])),
            "selection-bias": lambda l: l.ffn.sel_bias.zero_(),
            "post-norms": lambda l: l.post_coef.copy_(torch.tensor([1.0, 0.0]).expand_as(l.post_coef))}
    for name, off in offs.items():
        h = from_gigachat35(t)
        for l in h.layers:
            off(l)
        assert _rel(ref, h(x)) > 1e-3, name                     # each branch really matters
