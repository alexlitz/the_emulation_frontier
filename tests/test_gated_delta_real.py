"""The runnable gated-delta layer reproduces the REAL Qwen3-Next gated-DeltaNet module.

Instantiates transformers' actual `Qwen3NextGatedDeltaNet` (the shipped gated-DeltaNet mixer
that Qwen3-Next / the zoo's non-Mamba recurrences use), forces it onto its CPU torch path
(the `fla` fast path is GPU-only), runs its FULL forward -- conv, fused in-proj, per-head
decay/beta, the recurrence, the gated RMSNorm, out-proj -- and captures the module's OWN
internal recurrence inputs (post-conv, post-activation q/k/v/g/beta). It then shows
`attn.gated_delta_layer.gd_recurrence` reproduces the exact recurrence the real module ran,
token-for-token. So the elephantine's gated-delta primitive is the real library operator on
the real module's own activations -- not a stand-in. This is what turns the 420 non-Mamba
linear recurrences from constructed to EXECUTED.

Run:  pytest tests/test_gated_delta_real.py -q
"""
import types

import pytest

torch = pytest.importorskip("torch")
q3 = pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")

from attn.gated_delta_layer import gd_recurrence

F64 = torch.float64


def _real_layer_on_cpu():
    cfg = q3.Qwen3NextConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, linear_num_key_heads=2, linear_num_value_heads=4,
                             linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                             vocab_size=64, intermediate_size=128, hidden_act="silu")
    cfg.dtype = F64
    try:
        m = q3.Qwen3NextGatedDeltaNet(cfg, 0).to(F64).eval()
    except Exception as e:                                            # pragma: no cover
        pytest.skip(f"Qwen3NextGatedDeltaNet unbuildable here: {type(e).__name__}: {e}")
    # force the CPU torch reference for the recurrence AND the gated RMSNorm (fla kernels are GPU-only)
    old = m.norm
    eps = float(getattr(old, "eps", getattr(old, "variance_epsilon", 1e-6)))
    norm = q3.Qwen3NextRMSNormGated(old.weight.shape[0], eps=eps).to(F64)
    with torch.no_grad():
        norm.weight.copy_(old.weight.double())
    m.norm = norm
    return m


def test_gd_recurrence_reproduces_the_real_qwen3next_module_recurrence():
    m = _real_layer_on_cpu()
    cap = {}

    def capture(query, key, value, g, beta, **kw):                   # intercept the real module's call
        cap.update(q=query, k=key, v=value, g=g, beta=beta, kw=kw)
        return q3.torch_chunk_gated_delta_rule(query, key, value, g, beta, **kw)
    m.chunk_gated_delta_rule = capture

    torch.manual_seed(0)
    x = torch.randn(1, 9, 64, dtype=F64)
    with torch.no_grad():
        m(x)                                                         # runs the FULL real module forward
    assert cap, "the real module did not reach the recurrence"
    assert cap["kw"].get("use_qk_l2norm_in_kernel") is True          # the shipped module l2-norms q,k

    # replay the real module's own captured (q,k,v,g,beta) through OUR general rule
    q, k, v, g, beta = cap["q"], cap["k"], cap["v"], cap["g"], cap["beta"]
    B, T, H, dk = q.shape
    qn = q3.l2norm(q, dim=-1, eps=1e-6); kn = q3.l2norm(k, dim=-1, eps=1e-6)
    scale = 1 / dk ** 0.5
    a = g.exp()[..., None].expand(B, T, H, dk).permute(0, 2, 1, 3).contiguous()   # scalar decay per (t,head)
    mine = gd_recurrence((qn * scale).transpose(1, 2), kn.transpose(1, 2), v.transpose(1, 2),
                         a, beta.transpose(1, 2), beta.transpose(1, 2)).transpose(1, 2)
    ref, _ = q3.torch_chunk_gated_delta_rule(q, k, v, g, beta, **cap["kw"])
    # the transformers reference forces fp32 internally; require our rule within that fp32 gap
    assert (ref - mine).abs().max().item() < 1e-5, "our rule must reproduce the real module's recurrence"


def test_real_qwen3next_module_full_forward_runs_on_cpu():
    # sanity: the whole real mixer (not just the recurrence) executes end-to-end on CPU via the
    # torch fallbacks, so the capture above is of a genuine full-module forward.
    m = _real_layer_on_cpu()
    m.chunk_gated_delta_rule = q3.torch_chunk_gated_delta_rule
    torch.manual_seed(1)
    x = torch.randn(1, 7, 64, dtype=F64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 7, 64) and torch.isfinite(y).all()
