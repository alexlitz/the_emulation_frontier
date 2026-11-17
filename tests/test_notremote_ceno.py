"""CENO on the universal host, against a labelled transcription of its remote code (tests/refs/ceno.py).

The remote-code version (tests/test_universal_elephantine.py::test_same_host_carries_ceno) executes the
snapshot's modeling_ceno.py and is skipped unless remote code is allowed; this one runs no downloaded code.

Run:  pytest tests/test_notremote_ceno.py -q
"""
import torch

from tests.refs import ceno as R

F64 = torch.float64


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


@torch.no_grad()
def test_same_host_carries_ceno_notremote():
    """CENO: Nemotron-H-style Mamba-2 / MoE / attention / MLP hybrid whose non-Mamba blocks replace the residual
    stream by a norm of it (stream-norm branches), double norms in series (the norm pre stage), ungated squared-ReLU
    experts with post-top-k softmax routing, NoPE attention, a per-group gated RMSNorm in the Mamba-2 blocks.
    The old test's config (the extra kwargs n_groups / conv_kernel / chunk_size override the mamba_* ones, as
    PretrainedConfig does), fp64 with the reference's fp32 casts lifted: agreement to 1e-12 relative. Each
    family-specific branch switched off in the host moves the logits."""
    from elephantine.universal import from_ceno
    cfg = R.CENOConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=7,
                       hybrid_override_pattern="ME*E-M*", num_attention_heads=4, num_key_value_heads=2,
                       attention_head_dim=8, mamba_num_heads=4, mamba_head_dim=8, ssm_state_size=8, n_groups=2,
                       conv_kernel=4, chunk_size=4, num_experts=4, moe_top_k=2, mlp_hidden_act="relu2",
                       mlp_use_swiglu=False, qkv_layernorm=True, mamba_in_proj_layernorm=True, use_mamba_kernels=False,
                       use_conv_bias=True, layer_norm_epsilon=1e-5, pad_token_id=1, bos_token_id=0, eos_token_id=0)
    cfg.ceno_mamba_gate_in_ssm = True
    torch.manual_seed(0)
    t = R.CENOForCausalLM(cfg).to(F64).eval()
    torch.manual_seed(0)
    for p in t.parameters():
        p.add_(torch.randn_like(p) * 0.05)
    assert [b.block_type for b in t.backbone.layers] == ["mamba", "moe", "attention", "moe", "mlp", "mamba", "attention"]
    assert t.backbone.layers[0].mixer.n_groups == 2 and t.backbone.layers[0].mixer.chunk_size == 4
    x = torch.randint(0, 97, (2, 10), generator=torch.Generator().manual_seed(0))   # T=10: two full chunks + a padded one
    ref = t(x)
    assert ref.dtype == F64
    host = from_ceno(t)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))

    def off(fn):
        h = from_ceno(t)
        for ul, b in zip(h.layers, t.backbone.layers):
            fn(ul, b.block_type)
        return _rel(ref, h(x))

    def no_groups(ul, bt):                       # one RMSNorm over the full Mamba width, not per group
        if bt == "mamba":
            ul.ssm.avg.fill_(1.0 / ul.ssm.avg.shape[0])

    def no_pre(ul, bt):                          # drop the first of the two norms in series
        ul.norm_a.pre_c.zero_(); ul.norm_f.pre_c.zero_()

    moved = {
        "stream norms": off(lambda ul, bt: ul.sn_c.zero_()),
        "double norm": off(no_pre),
        "post-top-k softmax": off(lambda ul, bt: ul.ffn.renorm.zero_()),
        "relu^2 experts": off(lambda ul, bt: ul.ffn.c_relu2.zero_() if bt in ("moe", "mlp") else None),
        "grouped gated norm": off(no_groups),
    }
    for name, m in moved.items():
        assert m > 1e-4, (name, m)
