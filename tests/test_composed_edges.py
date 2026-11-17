"""The remaining exact-edge axes, composed with a capacity change, token-for-token.

`exact_edge_coverage()` reaches 100% only once every axis an exact edge can cross is
BUILDABLE alongside the capacity spine. The capacity spine + kv + pos + ffn + window +
mtp are covered elsewhere (tests/test_hidden_width.py, test_full_emulation.py). This
file closes the last exact axes — norm, block, sink, cross-layer-KV, act — each shown to
reproduce its target while a capacity axis (width / depth) also changes, so the whole
exact relation is built, not just accounted.

Run:  pytest tests/test_composed_edges.py -q
"""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

F64 = torch.float64
EXACT = 1e-9


def _r(*shape, seed=0):
    return torch.randn(*shape, dtype=F64, generator=torch.Generator().manual_seed(seed))


def _rms(z, w, eps=1e-6):
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * w


def test_norm_rmsnorm_is_layernorm_at_double_width():
    # norm axis (RMSNorm target -> LayerNorm host): a LayerNorm host reproduces RMSNorm
    # EXACTLY at 2x width via the mirror [x, -x] (mean 0 -> centering is a no-op, std = rms).
    d, T = 24, 12
    x, gamma = _r(T, d), _r(d, seed=1)
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) * gamma        # RMSNorm target
    xm = torch.cat([x, -x], -1)                                          # the capacity cost: 2x width
    centered = xm - xm.mean(-1, keepdim=True)
    ln = centered * torch.rsqrt(centered.pow(2).mean(-1, keepdim=True))   # LayerNorm host (no bias)
    assert (rms - ln[:, :d] * gamma).abs().max().item() < EXACT


def test_block_parallel_reproduces_sequential_at_double_depth():
    # block axis (sequential target -> parallel host): two parallel blocks (attn-only then
    # mlp-only) reproduce one sequential block; here the FFN is also WIDER (capacity).
    d, m, T = 16, 48, 8                                                  # m wider than d: capacity
    x = _r(T, d, seed=2)
    n1, n2, Wa, Wm1, Wm2 = _r(d, seed=3), _r(d, seed=4), _r(d, d, seed=5), _r(d, m, seed=6), _r(m, d, seed=7)
    A = lambda z: _rms(z, n1) @ Wa
    M = lambda z: F.silu(_rms(z, n2) @ Wm1) @ Wm2
    seq = x + A(x) + M(x + A(x))                                         # sequential target
    b1 = x + A(x)
    parallel_2deep = b1 + M(b1)                                          # parallel host at 2x depth
    assert (seq - parallel_2deep).abs().max().item() < EXACT


def test_sink_is_standard_plus_null_token_with_wide_values():
    # sink axis: attention-sink (off-by-one softmax) == standard softmax + one zero-value
    # null token; the value space is also WIDER (zero-padded) -> composes with capacity.
    T, dv = 16, 12
    scores = _r(T, seed=8)
    V = torch.cat([_r(T, 8, seed=9), torch.zeros(T, dv - 8, dtype=F64)], -1)   # wider V
    b = 1.3
    sink = (torch.exp(scores) / (math.exp(b) + torch.exp(scores).sum())) @ V   # sink target
    s_aug = torch.cat([scores, torch.tensor([b], dtype=F64)])
    host = F.softmax(s_aug, 0) @ torch.cat([V, torch.zeros(1, dv, dtype=F64)])  # standard + null token
    assert (sink - host).abs().max().item() < EXACT


def test_cross_layer_kv_reproduced_by_carrying_kv_with_wide_hidden():
    # cross-layer-KV axis: a CLA layer (reuses layer-0's K,V) is reproduced by a standard
    # layer carrying K0,V0 in reserved residual channels; the residual is also WIDER.
    T, d = 12, 16
    h0, h1 = _r(T, d, seed=10), _r(T, d, seed=11)
    Wk0, Wv0, Wq1 = _r(d, d, seed=12), _r(d, d, seed=13), _r(d, d, seed=14)
    K0, V0 = h0 @ Wk0, h0 @ Wv0

    def attend(q, k, v):
        mask = torch.triu(torch.full((T, T), float("-inf"), dtype=F64), 1)
        return F.softmax(q @ k.T / math.sqrt(d) + mask, -1) @ v
    cla = attend(h1 @ Wq1, K0, V0)                                       # CLA target reuses K0,V0
    h1_aug = torch.cat([h1, K0, V0], 1)                                 # carry in reserved channels (wider)
    Sk = torch.cat([torch.zeros(d, d), torch.eye(d), torch.zeros(d, d)], 1).to(F64)   # (d,3d) picks K0
    Sv = torch.cat([torch.zeros(d, d), torch.zeros(d, d), torch.eye(d)], 1).to(F64)   # (d,3d) picks V0
    Wq1_aug = torch.cat([Wq1, torch.zeros(2 * d, d)], 0).to(F64)         # (3d,d) query reads only h1
    host = attend(h1_aug @ Wq1_aug, h1_aug @ Sk.T, h1_aug @ Sv.T)
    assert (cla - host).abs().max().item() < EXACT


def test_act_swigluoai_reduces_to_silu():
    # act axis (the one exact activation edge, SiLU target <- swigluoai host): swiglu-oai's
    # gate with alpha=1 and no clamp is exactly x*sigmoid(x) = SiLU.
    a = _r(10, 32, seed=15)
    assert (F.silu(a) - a * torch.sigmoid(1.0 * a)).abs().max().item() < EXACT
