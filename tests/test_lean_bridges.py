"""Layer 2 of the verification plan: each Lean theorem (namespace EP, proofs/lean)
checked against the repo function that BUILDS that construction.

Every test is named after its theorem and checks the equation (exact), the convergence (limit) or the
inequality (bound) the theorem states, on the repo's own function, over Hypothesis-drawn inputs in
float64. verify/BRIDGES.md maps theorem -> function -> test -> status. Tests named `test_condition_*`
take one of the conditions the proofs exposed (PROOFS.md, "Where the formal statement needs a
condition") and show the equation FAILS when it is violated.

Tolerances. U = 2^-53 is the float64 unit roundoff. A result assembled from n floating-point
operations on terms of magnitude <= S is within about n U S of the exact value, so an exact identity is
checked at `fp(S, n)` = n U (1 + S) with n a generous count for that computation (stated at each use).
Bounds get the same slack on the right-hand side. Where `transformers` itself rounds to float32 (the
MoE routing weights, RoPE cos / sin tables, eager softmax) the slack is float32's, 2^-24, and says so.
"""
from __future__ import annotations

import contextlib
import io
import math
import re
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

F64 = torch.float64
U = 2.0 ** -53                      # float64 unit roundoff
U32 = 2.0 ** -24                    # float32 unit roundoff (where transformers casts to float32)


def fp(scale, n=1e3):
    """n float64 roundings of terms of size <= scale."""
    return n * U * (1.0 + float(scale))


SEED = st.integers(0, 2 ** 31 - 1)
FAST = settings(max_examples=5000, deadline=None)                 # pure numeric
MID = settings(max_examples=500, deadline=None)                   # small torch computations
SLOW = settings(max_examples=80, deadline=None,                  # stock transformers modules / models
                suppress_health_check=[HealthCheck.too_slow])
FLOAT = lambda lo, hi: st.floats(lo, hi, allow_nan=False, allow_infinity=False)


def rng(seed):
    return np.random.default_rng(seed)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def printed_errs(fn, *a, **kw):
    """Run a repo `check_*` routine that prints 'max err = X' and return every X it printed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **kw)
    errs = [float(x) for x in re.findall(r"max err = ([-+0-9.eE]+)", buf.getvalue())]
    assert errs, buf.getvalue()
    return errs


# =====================================================================================================
# Position.lean
# =====================================================================================================

from attn import positional_cross as pc          # noqa: E402


@FAST
@given(SEED, FLOAT(-3, 3), st.integers(-300, 300), st.integers(-300, 300))
def test_pairScore_eq(seed, w, i, j):
    """EP.pairScore_eq: <R(i w) q, R(j w) k> = (q.k) cos((i-j) w) + (q1 k2 - q2 k1) sin((i-j) w)."""
    q, k = rng(seed).standard_normal((2, 2))
    lhs = pc.rope_score([w], [q], [k], i, j)
    rhs = (q @ k) * math.cos((i - j) * w) + (q[0] * k[1] - q[1] * k[0]) * math.sin((i - j) * w)
    # two 2x2 rotations (angles up to 900 rad: cos / sin argument reduction ~ angle U) and a dot
    assert abs(lhs - rhs) <= fp(np.abs(q).sum() * np.abs(k).sum(), 50 * (1 + abs(i * w) + abs(j * w)))


@FAST
@given(SEED, st.integers(-200, 200), st.integers(-200, 200))
def test_pairScore_zero_freq(seed, i, j):
    """EP.pairScore_zero_freq: a pair at frequency 0 contributes q.k at every pair of positions."""
    q, k = rng(seed).standard_normal((2, 2))
    assert abs(pc.rope_score([0.0], [q], [k], i, j) - q @ k) <= fp(np.abs(q) @ np.abs(k), 10)


@MID
@given(SEED, st.integers(2, 40))
def test_nope_sub_rope(seed, seq):
    """EP.nope_sub_rope: inv_freq = 0 is the identity rotation, so the RoPE logit is q.k
    (attn/positional_cross.nope_from_rope_error: the RoPE score field at inv_freq 0 vs the content score)."""
    assert pc.nope_from_rope_error(seq=seq, seed=seed) <= fp(64, 100)   # |q.k| <= ~64 for d = 8 normals


@SLOW
@given(SEED, st.integers(4, 16), st.integers(2, 4))
def test_nope_sub_rope_on_llama(seed, seq, stretch):
    """EP.nope_sub_rope on the stock LlamaModel (hf/nope_rope.build): with inv_freq zeroed, changing the
    relative distances changes nothing (cos = 1, sin = 0 are exact even in float32); with RoPE on it does."""
    from hf import nope_rope
    rope_change, nope_change = nope_rope.build(seq=seq, stretch=stretch, seed=seed)
    assert nope_change == 0.0 and rope_change > 1e-6


@SLOW
@given(FLOAT(2.0, 1e7), st.sampled_from([8, 16, 32]))
def test_rope_theta_cannot_reach_nope(theta, d):
    """EP.rope_theta_cannot_reach_nope: theta^(-0/P) = 1 -- whatever rope_theta, the first pair of a
    default-RoPE head rotates at frequency 1, so the config alone never gives NoPE (the repo's NoPE
    route writes the inv_freq buffer: hf/nope_rope.py, hf/partial_rope_host.set_inv_freq)."""
    from hf import partial_rope_host as prh
    _, _, rot = prh._llama_target(d=d, theta=theta)
    assert float(rot.inv_freq[0]) == 1.0


def _alibi_case(seed, P):
    r = rng(seed)
    return r.standard_normal((P, 2)), r.standard_normal((P, 2))


@FAST
@given(SEED, st.integers(1, 6), FLOAT(-2, 2), FLOAT(1e-3, 2.0), st.integers(0, 300), st.integers(0, 300))
def test_alibi_host_score(seed, P, m, theta, i, j):
    """EP.alibi_host_score: content pairs at frequency 0 plus the frozen pair (-m/theta, 0), (0, 1) at
    frequency theta give exactly  q.k - (m/theta) sin(theta (i - j))  (attn/positional_cross.alibi_rope_host)."""
    q, k = _alibi_case(seed, P)
    s = pc.rope_score(*pc.alibi_rope_host(q, k, m, theta), i, j)
    want = (q * k).sum() - m / theta * math.sin((i - j) * theta)
    # the frozen pair carries |m/theta| (up to 2e3) through two rotations of angle <= 600
    assert abs(s - want) <= fp(abs(m / theta) + np.abs(q).sum() * np.abs(k).sum(), 50 * (1 + 300 * theta))


@FAST
@given(SEED, st.integers(1, 4), FLOAT(-2, 2), FLOAT(1e-4, 0.5), st.integers(0, 200), st.integers(0, 200))
def test_alibi_sub_rope_bound(seed, P, m, theta, i, j):
    """EP.alibi_sub_rope_bound: |host logit - (q.k - m (i - j))| <= |m| |i - j|^3 theta^2 / 6."""
    q, k = _alibi_case(seed, P)
    s = pc.rope_score(*pc.alibi_rope_host(q, k, m, theta), i, j)
    gap = abs(s - ((q * k).sum() - m * (i - j)))
    bound = abs(m) * abs(i - j) ** 3 * theta ** 2 / 6
    assert gap <= bound + fp(abs(m / theta) + abs(m * (i - j)) + np.abs(q).sum() * np.abs(k).sum(), 50 * (1 + 200 * theta))


@FAST
@given(FLOAT(-4, 4), FLOAT(1e-4, 0.5), st.integers(1, 128))
def test_alibi_sub_rope_bound_on_seed_measure(slope, theta, D):
    """EP.alibi_sub_rope_bound on the repo's own measurement (attn/positional_cross.alibi_smallangle_error:
    max over 0 <= d <= D of |-(m/theta) sin(theta d) + m d|) <= |m| D^3 theta^2 / 6."""
    err = pc.alibi_smallangle_error(slope, theta, D=D)
    assert err <= abs(slope) * D ** 3 * theta ** 2 / 6 + fp(abs(slope) * D, 20)


@MID
@given(SEED, st.integers(1, 4), FLOAT(-2, 2), st.integers(0, 64), st.integers(0, 64))
def test_alibi_sub_rope_limit(seed, P, m, i, j):
    """EP.alibi_sub_rope_limit: as theta -> 0 the host logit tends to the ALiBi logit q.k - m (i - j)
    (the gap falls with theta^2 down to the float64 floor |m/theta| U)."""
    q, k = _alibi_case(seed, P)
    target = (q * k).sum() - m * (i - j)
    thetas = 10.0 ** -np.arange(1, 8)
    gaps = [abs(pc.rope_score(*pc.alibi_rope_host(q, k, m, th), i, j) - target) for th in thetas]
    for th, g in zip(thetas, gaps):
        assert g <= abs(m) * abs(i - j) ** 3 * th ** 2 / 6 + fp(abs(m / th) + abs(m * (i - j)), 100)
    assert gaps[-1] <= 1e-8 * (1 + abs(target) + abs(m * (i - j)))


@MID
@given(SEED, st.integers(2, 12), FLOAT(-1, 1), st.integers(0, 40))
def test_alibi_sub_rope_attn(seed, n_keys, m, i):
    """EP.alibi_sub_rope_attn: the attention output over the keys converges with the logits; each weight
    is within exp(2 delta) of ALiBi's (delta = the logit bound at theta), so the output is too."""
    r = rng(seed)
    q, ks, v = r.standard_normal((2, 2)), r.standard_normal((n_keys, 2, 2)), r.standard_normal((n_keys, 3))
    pos = r.integers(0, i + 1, n_keys)                                   # causal: keys at j <= i
    alibi = np.array([(q * k).sum() - m * (i - j) for k, j in zip(ks, pos)])
    w_t = np.exp(alibi - alibi.max()); w_t /= w_t.sum()
    for th in (1e-1, 1e-2, 1e-3):
        s = np.array([pc.rope_score(*pc.alibi_rope_host(q, k, m, th), i, j) for k, j in zip(ks, pos)])
        w_h = np.exp(s - s.max()); w_h /= w_h.sum()
        delta = abs(m) * i ** 3 * th ** 2 / 6 + fp(abs(m / th), 100)
        assert np.all(w_h <= np.exp(2 * delta) * w_t + fp(1, 100)) and np.all(w_t <= np.exp(2 * delta) * w_h + fp(1, 100))
        assert np.abs(w_h @ v - w_t @ v).max() <= (np.exp(2 * delta) - 1) * np.abs(v).sum(0).max() + fp(np.abs(v).max(), 100)


@FAST
@given(FLOAT(-3, 3), st.integers(-100, 100), st.integers(-100, 100), FLOAT(-5, 5))
def test_const_channel_rotates(w, i, p, s):
    """EP.const_channel_rotates: a constant query component (1, 0) against a fixed key (s, 0) in a pair of
    frequency w gives s cos((i - p) w) -- position-dependent unless w = 0."""
    got = pc.rope_score([w], [[1.0, 0.0]], [[s, 0.0]], i, p)
    assert abs(got - s * math.cos((i - p) * w)) <= fp(abs(s), 50 * (1 + abs(i * w) + abs(p * w)))


@FAST
@given(FLOAT(-50, 50), FLOAT(1e-3, 1.0), st.integers(0, 200), st.integers(0, 200))
def test_alibi_pair_odd(N, theta, i, j):
    """EP.alibi_pair_odd: the frozen pair's logit is odd in i - j (attn/positional_cross.rope_pair_score)."""
    a, b = pc.rope_pair_score(N, theta, j, i), pc.rope_pair_score(N, theta, i, j)
    assert abs(a + b) <= fp(abs(N), 50 * (1 + 200 * theta))


@MID
@given(SEED, FLOAT(0.1, 2), FLOAT(1e-4, 1e-2))
def test_condition_alibi_causal_only(seed, m, theta):
    """Condition 3: ALiBi <= RoPE holds for CAUSAL attention only. The frozen pair gives -m (i - j), odd
    in i - j; bidirectional ALiBi needs -m |i - j|. For a key after the query (j > i) the host is off by
    2 m (j - i) however small theta is."""
    q, k = _alibi_case(seed, 2)
    i, j = 3, 20
    s = pc.rope_score(*pc.alibi_rope_host(q, k, m, theta), i, j)
    bidir = (q * k).sum() - m * abs(i - j)
    assert abs(s - bidir) >= 2 * m * (j - i) - abs(m) * (j - i) ** 3 * theta ** 2 / 6 - 1e-9


@MID
@given(SEED, FLOAT(0.1, 2))
def test_condition_alibi_content_pairs_need_zero_freq(seed, m):
    """Condition 3 (second half): the content pairs must sit at frequency 0 (a buffer write, as for NoPE).
    Left at a stock RoPE schedule they keep rotating and the logit is not q.k - m (i - j) at any theta.

    The threshold is a lower bound, not a heuristic. A content pair p at frequency w_p contributes
    u_p(i) = <R(i w_p) q_p, k_p> = A_p cos(i w_p + phi_p), A_p = |q_p| |k_p| (the rotation the theorem
    states), and the frozen pair -(m/theta) sin(theta i). The gap g(i) = logit(i) - (q.k - m i) has second
    difference  D2 g(i) = -sum_p 4 sin^2(w_p/2) u_p(i) + 4 sin^2(theta/2) (m/theta) sin(theta i)  (the ALiBi
    target is linear in i), and max|g| >= max|D2 g| / 4. The w = 1 pair gives |D2| >= 4 sin^2(1/2) A_0 c at
    some i in 2..28, where c = cos(G/2) and G is the largest gap the points i mod pi leave on the circle
    (whatever phi_0 is, some i + phi_0 is within G/2 of a multiple of pi). The other pairs and the frozen pair
    take at most their amplitudes off. So
        max_i |g(i)| >= sin^2(1/2) A_0 c - sum_{p>=1} sin^2(w_p/2) A_p - sin^2(theta/2) m/theta.
    (The old threshold 1e-2 sum|q| sum|k| was a guess; it failed at Hypothesis seed 39, gap 0.26 vs 0.35.)"""
    q, k = _alibi_case(seed, 4)
    theta = 1e-4
    freqs, qh, kh = pc.alibi_rope_host(q, k, m, theta)
    stock = freqs.copy()
    stock[:-1] = 10000.0 ** (-np.arange(4) / 4)                    # the content pairs rotate (default RoPE)
    gaps = [abs(pc.rope_score(stock, qh, kh, i, 0) - ((q * k).sum() - m * i)) for i in range(1, 30)]
    pts = np.sort(np.mod(np.arange(2, 29), np.pi))
    G = max(np.diff(pts).max(), np.pi - pts[-1] + pts[0])
    A = np.linalg.norm(q, axis=1) * np.linalg.norm(k, axis=1)
    s2 = np.sin(stock[:-1] / 2) ** 2
    bound = s2[0] * A[0] * math.cos(G / 2) - (s2[1:] * A[1:]).sum() - math.sin(theta / 2) ** 2 * m / theta
    assume(bound > 1e-6 * (1 + A.sum()))                          # A_0 not degenerate (q_0 or k_0 ~ 0)
    assert max(gaps) >= bound - 1e-9 * (1 + A.sum() + m * 30)


@MID
@given(FLOAT(0.05, 3), FLOAT(-5, 5).filter(lambda s: abs(s) > 0.1))
def test_condition_sink_constant_channel_needs_zero_freq(w, s):
    """Condition 4: the sink key (0, s) read by the constant query channel 1 must sit in an UNROTATED pair.
    In a pair of frequency w != 0 the sink logit s cos((i - p) w) depends on the query position i."""
    logits = [pc.rope_score([w], [[1.0, 0.0]], [[s, 0.0]], i, 0) for i in range(64)]
    assert max(logits) - min(logits) > 0.1 * abs(s)
    still = [pc.rope_score([0.0], [[1.0, 0.0]], [[s, 0.0]], i, 0) for i in range(64)]
    assert max(still) - min(still) == 0.0


@MID
@given(SEED, st.integers(8, 64))
def test_condition_constant_channel_needs_a_bias(seed, d):
    """Conditions 3 / 4 (a corollary the table leaves implicit): the constant channel (the frozen pair's
    k = (0, 1), the sink's query 1) needs a q / k BIAS. A residual channel held at 1 is not constant after
    the host's pre-RMSNorm -- it is divided by each token's rms (norm/layernorm_rmsnorm.py says the same of
    LayerNorm's beta), so a bias-free projection of it varies per token."""
    from norm.layernorm_rmsnorm import _rms
    x = torch.randn(16, d, dtype=F64, generator=gen(seed))
    x[:, 0] = 1.0                                                 # the reserved constant channel
    read = (x * _rms(x, 1e-6))[:, 0]                              # what a bias-free k_proj row sees
    assert (read.max() - read.min()) / read.abs().max() > 1e-3


# --- partial RoPE host -------------------------------------------------------------------------------

from hf import partial_rope_host as prh          # noqa: E402


def _stub_target(H, KV, d, rot, seed, q_norm=False):
    g = gen(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    hid = 8
    return prh.Target(H=H, KV=KV, d=d, rot=rot, dv=d, wq=r(H * d, hid), wk=r(KV * d, hid), wv=r(KV * d, hid),
                      wo=r(hid, H * d), inv_freq=torch.rand(rot // 2, dtype=F64, generator=g), scale=d ** -0.5,
                      q_norm=r(d).abs() + 0.5 if q_norm else None, k_norm=r(d).abs() + 0.5 if q_norm else None,
                      eps=1e-6 if q_norm else 0.0)


@MID
@given(SEED, st.sampled_from([4, 8, 16]), st.integers(0, 3), st.booleans(), st.integers(1, 3))
def test_partial_rope_host_frequencies(seed, d, extra_pairs, interleaved, mult):
    """EP.partial_rope_host, hypothesis hw: column_map puts every target pair on a host pair and the host's
    inv_freq on that pair equals the target's frequency (the other host pairs at 0)."""
    t = _stub_target(1, 1, d, d, seed)
    hd = d * mult + 2 * extra_pairs
    rot_h = min(hd, d + 2 * extra_pairs)
    h = prh.Host(H=1, KV=1, hd=hd, rot=rot_h, interleaved=interleaved, scale=hd ** -0.5)
    m, inv = prh.column_map(t, h)
    hp = prh.pairs(rot_h, interleaved)
    for jj, (a, b) in enumerate(prh.pairs(d, False)):
        assert (m[a], m[b]) == hp[jj] and inv[jj] == t.inv_freq[jj]
    assert torch.all(inv[d // 2:] == 0)


@SLOW
@given(SEED, st.sampled_from([("glm", dict(hd=32, factor=0.5)), ("glm", dict(hd=64, factor=0.5)),
                              ("phi", dict(hd=32, factor=0.5, KV=4)), ("neox", dict(hd=64, factor=0.25, KV=4)),
                              ("glm", dict(hd=32, factor=0.5, host_H=8, host_KV=4))]))
def test_partial_rope_host(seed, case):
    """EP.partial_rope_host: zero-padded q / k on a host whose first P_t frequencies are the target's give
    the target logit -- here the whole attention block (hf/partial_rope_host.module_check: a stock
    Llama / Qwen2 target in stock GLM / Phi / GPT-NeoX partial-RoPE hosts). Both sides build cos / sin
    from the same float32 inv_freq values, so they agree to float64 rounding."""
    family, kw = case
    assert prh.module_check(family, seed=seed, **kw) <= 1e-12


@FAST
@given(FLOAT(2.0, 1e7), st.integers(1, 64), st.integers(0, 64))
def test_defaultFreq_match(theta_t, Pt, e):
    """EP.defaultFreq_match / EP.partial_rope_freq_match: base theta_t^(rot_h / d_t) gives host pair j the
    target's frequency theta_t^(-2j / d_t), for every j < P_t (the repo's config-only route,
    hf/partial_rope_host.theta_only_check: rope_theta_h = theta_t ** (rot_h / d_t))."""
    assume(e <= 7 * Pt)                            # theta_t^((Pt+e)/Pt) stays inside the float64 range
    d_t, rot_h = 2 * Pt, 2 * (Pt + e)
    theta_h = theta_t ** (rot_h / d_t)
    j = np.arange(Pt)
    host = theta_h ** (-2 * j / rot_h)
    tgt = theta_t ** (-2 * j / d_t)
    # theta_h carries one rounding (U relative); the second pow scales it by |log host| <~ 32, plus its own
    assert np.all(np.abs(host - tgt) <= 200 * U * tgt)


@SLOW
@given(SEED, st.sampled_from([(16, 64, 0.5), (16, 32, 0.5), (8, 64, 0.25)]), FLOAT(1e3, 1e6))
def test_partial_rope_host_config(seed, dims, theta):
    """EP.partial_rope_host_config: the config alone (no buffer write) -- a default-RoPE GLM host with
    rope_theta_h = theta_t ** (rot_h / d_t) reproduces a default-RoPE Llama head. transformers computes
    both inv_freq tables in float32 (x 2^-24 relative per pow) and the cos / sin from them."""
    d, hd, factor = dims
    freq_err, out_err = prh.theta_only_check(d=d, hd=hd, factor=factor, theta=theta, seed=seed)
    assert freq_err <= 4 * U32 and out_err <= 1e-6


@MID
@given(SEED, st.sampled_from([4, 8, 16]), st.integers(1, 48), FLOAT(1e-8, 1e-2))
def test_qknorm_pad(seed, d, e, eps):
    """EP.qknorm_pad: an RMSNorm over the zero-padded head with eps_h = eps d / (d + e) and weight
    sqrt(d / (d + e)) w returns the padded target norm (hf/partial_rope_host.place, its qk-norm branch)."""
    t = _stub_target(1, 1, d, d, seed, q_norm=True)
    t.eps = eps
    hd = d + e
    h = prh.Host(H=1, KV=1, hd=hd, rot=d, interleaved=False, scale=t.scale, qk_norm="plain")
    P = prh.place(t, h)
    m, _ = prh.column_map(t, h)
    x = torch.randn(5, d, dtype=F64, generator=gen(seed + 1))
    xp = torch.zeros(5, hd, dtype=F64)
    xp[:, m] = x
    host = xp * torch.rsqrt(xp.pow(2).mean(-1, keepdim=True) + P["eps"]) * P["k_norm"]
    want = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * t.k_norm
    assert abs(P["eps"] - eps * d / hd) <= 4 * U * eps
    assert (host[:, m] - want).abs().max() <= fp(want.abs().max(), 4 * hd)
    assert host[:, [c for c in range(hd) if c not in m]].abs().max() == 0


@SLOW
@given(SEED)
def test_qknorm_pad_on_qwen3_next(seed):
    """EP.qknorm_pad inside a stock gated, qk-normed Qwen3-Next host (hf/partial_rope_host.module_check):
    its qk RMSNorms upcast to float32, so the agreement is float32's."""
    assert prh.module_check("qwen3_next", hd=64, factor=0.25, qwen3=True, seed=seed) <= 1e-6


# --- retention ----------------------------------------------------------------------------------------

from attn import output_rotation as orot         # noqa: E402


@MID
@given(SEED, st.integers(1, 24), st.sampled_from([2, 4, 8]), FLOAT(-1.5, 1.5), FLOAT(0.0, 1.0))
def test_retention_parallel_eq_recurrent(seed, T, d, theta, gamma):
    """EP.retention_parallel_eq_recurrent: q~_n^T S_n with S_n = gamma S_{n-1} + k~_n v_n^T equals
    sum_{m<=n} gamma^(n-m) (q~_n . k~_m) v_m (attn/output_rotation)."""
    g = gen(seed)
    Q, K, V = (torch.randn(T, d, dtype=F64, generator=g) for _ in range(3))
    par = orot.parallel_retention(Q, K, V, theta, gamma)
    rec = orot.recurrent_retention(Q, K, V, theta, gamma)
    assert (par - rec).abs().max() <= fp((Q.abs().sum(1).max() * K.abs().sum(1).max() * V.abs().sum(0).max()).item(), 8 * T * d)


@MID
@given(SEED, st.integers(1, 12), st.sampled_from([2, 4]), FLOAT(-1.5, 1.5), FLOAT(0.0, 1.0))
def test_retState_eq(seed, T, d, theta, gamma):
    """EP.retState_eq: the recurrent state S_n = sum_{m<=n} gamma^(n-m) k~_m v_m^T. Read row by row out of
    attn/output_rotation.recurrent_retention (query e_i at every step reads R_n e_i . S_n)."""
    g = gen(seed)
    K, V = torch.randn(T, d, dtype=F64, generator=g), torch.randn(T, d, dtype=F64, generator=g)
    Rs = orot.rot_stack(T, d, theta)
    rows = torch.stack([orot.recurrent_retention(torch.eye(d, dtype=F64)[i].expand(T, d), K, V, theta, gamma)
                        for i in range(d)], 1)                     # (T, d, dv): rows[n] = R_n^T S_n
    Kr = orot.rotate(K, Rs)
    for n in range(T):
        S = sum(gamma ** (n - m) * torch.outer(Kr[m], V[m]) for m in range(n + 1))
        assert (Rs[n] @ rows[n] - S).abs().max() <= fp((K.abs().max() * V.abs().max() * T * d).item(), 16 * T * d)


@FAST
@given(SEED, st.integers(0, 40), st.integers(0, 40), FLOAT(-2, 2))
def test_rot_rides(seed, n, m, theta):
    """EP.rot_rides: (R_n q) . (R_m k) = q . R_{m-n} k -- the rotation rides in the state
    (attn/output_rotation.rot_stack)."""
    q, k = torch.randn(2, 4, dtype=F64, generator=gen(seed))
    Rs = orot.rot_stack(max(n, m) + 1, 4, theta)
    lhs = (Rs[n] @ q) @ (Rs[m] @ k)
    R_rel = Rs[m - n] if m >= n else Rs[n - m].T
    assert abs(lhs - q @ (R_rel @ k)) <= fp((q.abs().sum() * k.abs().sum()).item(), 100 * (1 + 40 * abs(theta)))


# =====================================================================================================
# Norm.lean
# =====================================================================================================

from elephantine import universal as uni         # noqa: E402


def _unorm_ln(w, b, eps):
    n = uni.UNorm(w.numel())
    with torch.no_grad():
        n.weight.copy_(w); n.bias.copy_(b); n.centre.fill_(1.0); n.eps.fill_(eps)
    return n


@MID
@given(SEED, st.integers(2, 64), FLOAT(1e-8, 1e-2), FLOAT(0.01, 100))
def test_layerNorm_eq_rms_center(seed, n, eps, scale):
    """EP.layerNorm_eq_rms_center: LN(x) = gamma (.) RMS(P x) + beta, same eps. The repo's LayerNorm IS
    this right-hand side (elephantine/universal.UNorm with centre = 1, which _ln_from loads every
    nn.LayerNorm into); checked against torch's layer_norm."""
    g = gen(seed)
    x, w, b = scale * torch.randn(7, n, dtype=F64, generator=g), torch.randn(n, dtype=F64, generator=g), torch.randn(n, dtype=F64, generator=g)
    with torch.no_grad():
        got = _unorm_ln(w, b, eps)(x)
    want = F.layer_norm(x, (n,), w, b, eps)
    assert (got - want).abs().max() <= fp(want.abs().max().item(), 20 * n)


@MID
@given(SEED, st.integers(2, 64))
def test_layerNorm_eq_rms_center_patched_variant(seed, n):
    """EP.layerNorm_eq_rms_center on norm/layernorm_rmsnorm.make_variant('exact') (centre -> RMSNorm ->
    affine, patched onto an nn.LayerNorm). It computes in float32 (x.float()), so the slack is float32's."""
    from norm.layernorm_rmsnorm import make_variant
    g = gen(seed)
    ln = torch.nn.LayerNorm(n, eps=1e-5)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(n, generator=g)); ln.bias.copy_(torch.randn(n, generator=g))
    x = torch.randn(5, n, generator=g)
    got = make_variant("exact")(ln)(ln, x)
    want = F.layer_norm(x.double(), (n,), ln.weight.double(), ln.bias.double(), 1e-5)
    # float32 rounding of the centred input is amplified by 1 / sqrt(var + eps) (Hypothesis: seed 3703, n = 2, a row
    # of near-equal entries): the slack scales with |x| / sqrt(var + eps) times the gain, not with the output
    xd = x.double()
    amp = (xd.abs().amax(-1) / torch.sqrt(xd.var(-1, unbiased=False) + 1e-5)).max().item()
    assert (got.double() - want).abs().max() <= 50 * n * U32 * (1 + want.abs().max().item()
                                                                 + amp * ln.weight.abs().max().item())


@FAST
@given(SEED, st.integers(1, 64))
def test_centred_rms_is_variance(seed, n):
    """EP.centred_rms_is_variance: mean((x - mean x)^2) is the biased variance, so normalising the centred
    vector by its RMS is LayerNorm's normalisation (norm/layernorm_rmsnorm._rms on P x)."""
    from norm.layernorm_rmsnorm import _rms
    x = torch.randn(n, dtype=F64, generator=gen(seed)) * 3 + 1
    c = x - x.mean()
    eps = 1e-6
    assert abs(_rms(c, eps).item() - (x.var(unbiased=False) + eps).rsqrt().item()) <= fp(1 / math.sqrt(eps), 10 * n)


@MID
@given(SEED, st.integers(1, 32), st.integers(1, 32), st.booleans())
def test_affine_fold(seed, n, m, has_bias):
    """EP.affine_fold: W (gamma (.) z + beta) + b = (W diag gamma) z + (W beta + b)
    (convert/layernorm_to_rmsnorm._fold_beta_into adds W beta to the consumer's bias, creating one if
    absent; gamma stays as the RMSNorm gain)."""
    from convert.layernorm_to_rmsnorm import _fold_beta_into
    g = gen(seed)
    lin = torch.nn.Linear(n, m, bias=has_bias).double()
    with torch.no_grad():
        lin.weight.copy_(torch.randn(m, n, dtype=F64, generator=g))
        if has_bias:
            lin.bias.copy_(torch.randn(m, dtype=F64, generator=g))
    gam, beta, z = (torch.randn(n, dtype=F64, generator=g) for _ in range(3))
    with torch.no_grad():
        W, b0 = lin.weight.clone(), (lin.bias.clone() if has_bias else torch.zeros(m, dtype=F64))
        want = lin(gam * z + beta)
        _fold_beta_into(lin, beta)
        got = lin(gam * z)
    # Floating-point bound (Higham, gamma_k = k U / (1 - k U)): a sum of k products is within gamma_k of
    # the sum of their absolute values. Per output row i, with T_i = sum_j |W_ij| (|gamma_j z_j| + |beta_j|) + |b_i|:
    #   want: u = fl(fl(gamma z) + beta) (2 roundings), then W u + b (n + 1 terms)       -> gamma_{n+3} T_i
    #   got:  b' = fl(b + fl(W beta)) (n + 1 terms), then W fl(gamma z) + b' (n + 2)      -> gamma_{n+4} T_i
    # so |got_i - want_i| <= 2 gamma_{n+4} T_i (to first order in U; the slack covers the second order).
    gk = lambda k_: k_ * U / (1 - k_ * U)
    T = W.abs() @ ((gam * z).abs() + beta.abs()) + b0.abs()
    assert ((got - want).abs() <= 2 * gk(n + 4) * T * (1 + 1e-12)).all()


def _neox(seed, layers=2, d=32):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=61, hidden_size=d, num_hidden_layers=layers, num_attention_heads=4,
                        intermediate_size=2 * d, max_position_embeddings=64, rotary_pct=0.5,
                        attn_implementation="eager")
    m = GPTNeoXForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():                         # give every LayerNorm a real gamma and beta
        for mod in m.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.weight.copy_(1 + 0.3 * torch.randn_like(mod.weight))
                mod.bias.copy_(0.3 * torch.randn_like(mod.bias))
    return m


def _opt(seed, layers=2, d=32):
    from transformers import OPTConfig, OPTForCausalLM
    torch.manual_seed(seed)
    cfg = OPTConfig(vocab_size=61, hidden_size=d, num_hidden_layers=layers, num_attention_heads=4, ffn_dim=2 * d,
                    max_position_embeddings=64, word_embed_proj_dim=d, do_layer_norm_before=True,
                    attn_implementation="eager")
    m = OPTForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.weight.copy_(1 + 0.3 * torch.randn_like(mod.weight))
                mod.bias.copy_(0.3 * torch.randn_like(mod.bias))
    return m


def _fold_pair(make, seed, **kw):
    from norm.layernorm_rmsnorm import fold_layernorm_model
    m0, m1 = make(seed), make(seed)
    fold_layernorm_model(m1, **kw)
    ids = torch.randint(0, 61, (2, 12), generator=gen(seed + 1))
    with torch.no_grad():
        o0, o1 = m0(ids, output_hidden_states=True), m1(ids, output_hidden_states=True)
    return m0, m1, ids, o0, o1


@SLOW
@given(SEED, st.sampled_from(["neox", "opt"]))
def test_layerNorm_sub_rmsNorm(seed, fam):
    """EP.layerNorm_sub_rmsNorm (norm/layernorm_rmsnorm.fold_layernorm_model on a stock GPT-NeoX / OPT,
    fp64, every LayerNorm with a random gamma and beta): (1) the host residual is P times the target
    residual at every layer; (2) the logits are equal. Eager attention rounds its softmax to float32 in
    both models, but the host's attention inputs are the same up to float64 rounding, so the softmax
    sees the same values: the slack is float64's over a 2-layer forward."""
    m0, m1, ids, o0, o1 = _fold_pair(_neox if fam == "neox" else _opt, seed)
    for h0, h1 in zip(o0.hidden_states[:-1], o1.hidden_states[:-1]):   # the last entry is after the final norm
        P_h0 = h0 - h0.mean(-1, keepdim=True)
        assert (h1 - P_h0).abs().max() <= 1e-10 * (1 + h0.abs().max().item())
    assert (o0.logits - o1.logits).abs().max() <= 1e-10 * (1 + o0.logits.abs().max().item())


@SLOW
@given(SEED)
def test_condition_layernorm_fold_needs_lm_head_bias(seed):
    """Condition 8: with beta != 0 on the final norm, the host's lm_head needs a bias (W_U beta_f). Folding
    without it (fold_layernorm_model(lm_head_bias=False)) shifts every logit by a per-vocab constant."""
    _, _, _, o0, o1 = _fold_pair(_neox, seed, lm_head_bias=False)
    assert (o0.logits - o1.logits).abs().max() > 1e-4


@SLOW
@given(SEED)
def test_condition_layernorm_fold_needs_untied_head(seed):
    """Condition 8: the embedding is post-composed with P (E -> E P), so the fold unties a tied head. Keeping
    them tied -- the head reading the CENTRED E -- changes the logits (OPT ties its head)."""
    from norm.layernorm_rmsnorm import fold_layernorm_model
    m0, m1 = _opt(seed), _opt(seed)
    info = fold_layernorm_model(m1)
    assert info["untied"]
    with torch.no_grad():
        m1.lm_head.weight.copy_(m1.model.decoder.embed_tokens.weight)        # re-tie to the centred table
    ids = torch.randint(0, 61, (2, 12), generator=gen(seed + 1))
    with torch.no_grad():
        assert (m0(ids).logits - m1(ids).logits).abs().max() > 1e-4


@SLOW
@given(SEED, st.sampled_from(["writer", "embedding", "eps"]))
def test_condition_layernorm_fold_needs_every_writer_centred(seed, what):
    """Condition 8: EVERY residual writer is post-composed with P (embedding included) and the eps stays the
    same. Undo the centring of one writer, or of the embedding, or move one norm's eps, and the logits move."""
    from norm.layernorm_rmsnorm import fold_layernorm_model
    m0, m1 = _neox(seed), _neox(seed)
    fold_layernorm_model(m1)
    with torch.no_grad():
        if what == "writer":
            w = m1.gpt_neox.layers[0].attention.dense.weight
            w.add_(0.5 * torch.randn(1, w.shape[1], dtype=F64, generator=gen(seed)))   # a non-zero column mean
        elif what == "embedding":
            m1.gpt_neox.embed_in.weight.add_(0.5)
        else:
            m1.gpt_neox.layers[0].input_layernorm.eps = 1e-1
    ids = torch.randint(0, 61, (2, 12), generator=gen(seed + 1))
    with torch.no_grad():
        assert (m0(ids).logits - m1(ids).logits).abs().max() > 1e-6


@SLOW
@given(SEED)
def test_convert_pythia_folds_the_centring(seed):
    """(Was MISMATCH 1, fixed in round 2, pass 5.) convert/layernorm_to_rmsnorm.convert_pythia now runs the exact
    fold, EP.layerNorm_sub_rmsNorm: beta into the consumers and an lm_head bias, the centring into every writer.
    The old beta-only conversion (convert_pythia_beta_only) still computes a different function."""
    from convert.layernorm_to_rmsnorm import convert_pythia, convert_pythia_beta_only
    m0, m1, m2 = _neox(seed), _neox(seed), _neox(seed)
    convert_pythia(m1)
    convert_pythia_beta_only(m2)
    ids = torch.randint(0, 61, (2, 12), generator=gen(seed + 1))
    with torch.no_grad():
        ref = m0(ids).logits
        assert (ref - m1(ids).logits).abs().max() <= 1e-9 * max(1.0, ref.abs().max().item())
        assert (ref - m2(ids).logits).abs().max() > 1e-4


@MID
@given(SEED, st.integers(1, 32), FLOAT(1e-8, 1e-2))
def test_layerNorm_mirror(seed, n, eps):
    """EP.layerNorm_mirror (NO CODE: the repo builds no mirror host; graph/model_graph only charges its 2x
    width). The norm-level identity on the repo's LayerNorm (universal.UNorm, centre = 1): weight
    [gamma; gamma], bias 0, the same eps, on [x; -x] returns [RMS(x); -RMS(x)]."""
    g = gen(seed)
    x, w = torch.randn(4, n, dtype=F64, generator=g), torch.randn(n, dtype=F64, generator=g)
    with torch.no_grad():
        got = _unorm_ln(torch.cat([w, w]), torch.zeros(2 * n, dtype=F64), eps)(torch.cat([x, -x], -1))
    r = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    assert (got - torch.cat([r, -r], -1)).abs().max() <= fp(r.abs().max().item(), 40 * n)


# =====================================================================================================
# Attention.lean
# =====================================================================================================

from hf import gqa_mha                           # noqa: E402


@SLOW
@given(SEED, st.sampled_from([(8, 1), (8, 2), (8, 4), (4, 2), (4, 1)]), st.integers(2, 16))
def test_gqa_sub_mha(seed, heads, seq):
    """EP.gqa_sub_mha: an MHA host with KV heads tiled (W_k,h := W_k,g(h)) reproduces GQA / MQA
    (hf/gqa_mha.build: a stock LlamaModel GQA target and its unmodified MHA host, fp64)."""
    n_heads, n_kv = heads
    assert gqa_mha.build(n_heads=n_heads, n_kv=n_kv, seq=seq, seed=seed) <= 1e-12


@FAST
@given(SEED, st.sampled_from([1, 2, 4]), st.sampled_from([1, 2, 4]), st.sampled_from([2, 4, 8]))
def test_groupOf(seed, n_kv, rep, hd):
    """EP.groupOf / EP.gqa_tiled_scores: the repeat_interleave tiling sends query head h to KV head h / r,
    in both repo tilers (hf/gqa_mha.expand_kv, elephantine/universal._tile_kv), so every tiled per-head
    score equals the GQA score."""
    H = n_kv * rep
    w = torch.randn(n_kv * hd, 5, dtype=F64, generator=gen(seed))
    a, b = gqa_mha.expand_kv(w, n_kv, H, hd), uni._tile_kv(w, n_kv, H, hd)
    assert torch.equal(a, b)
    for h in range(H):
        assert torch.equal(a[h * hd:(h + 1) * hd], w[(h // rep) * hd:(h // rep + 1) * hd])


@FAST
@given(st.integers(1, 4), st.integers(1, 4), st.integers(1, 4), st.integers(1, 8))
def test_gqa_tiled_scores(KVt, gt, gh, KVh_extra):
    """EP.gqa_tiled_scores on elephantine/emulate.head_maps: host query head qmap[j] reads a host KV group
    whose kvmap is the target's KV head j // g_t, for every target query head j."""
    from elephantine.emulate import Refused, head_maps
    Ht, n = KVt * gt, -(-gt // gh)
    KVh = KVt * n + KVh_extra - 1
    Hh = KVh * gh
    try:
        qmap, kvmap = head_maps(Ht, KVt, Hh, KVh)
    except Refused:
        pytest.fail("head_maps refused a host with enough KV groups")
    assert len({h for h, _ in qmap.values()}) == Ht
    for j in range(Ht):                       # (emulate v3: (host head, block) / [(target KV head, block), ...])
        h, blk = qmap[j]
        assert (j // gt, blk) in kvmap[h // gh]


@SLOW
@given(SEED, st.sampled_from([2, 4]), st.integers(2, 12))
def test_mqa_sub_gqa(seed, G, seq):
    """EP.mqa_sub_gqa: a GQA host whose G KV heads are all copies of the single MQA head reproduces MQA
    (hf/gqa_mha.expand_kv from 1 KV head to G, on stock LlamaModels)."""
    from transformers.models.llama.modeling_llama import LlamaModel
    torch.manual_seed(seed)
    H, hd = 8, 8
    mqa = LlamaModel(gqa_mha._cfg(H, 1)).double().eval()
    gqa = LlamaModel(gqa_mha._cfg(H, G)).double().eval()
    sd = {k: (gqa_mha.expand_kv(v, 1, G, hd) if k.endswith(("k_proj.weight", "v_proj.weight")) else v)
          for k, v in mqa.state_dict().items()}
    gqa.load_state_dict(sd, strict=True)
    x = torch.randn(1, seq, 64, dtype=F64, generator=gen(seed))
    with torch.no_grad():
        assert (mqa(inputs_embeds=x).last_hidden_state - gqa(inputs_embeds=x).last_hidden_state).abs().max() <= 1e-12


@SLOW
@given(SEED, st.sampled_from([(32, 4), (48, 6), (32, 8)]), st.integers(2, 24))
def test_mha_sub_wide_mqa(seed, dims, T):
    """EP.mha_sub_wide_mqa: one shared KV head of width H d_h, query h zero outside slice h, o_proj reading
    slice h (attn/mha_mqa_gqa.verify_mqa_to_mha_exact on a stock GPT-2 layer, random biases, fp64)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    from attn.mha_mqa_gqa import verify_mqa_to_mha_exact
    d, H = dims
    torch.manual_seed(seed)
    m = GPT2LMHeadModel(GPT2Config(n_embd=d, n_head=H, n_layer=1, vocab_size=11, n_positions=32))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    with contextlib.redirect_stdout(io.StringIO()):
        err = verify_mqa_to_mha_exact(m, T=T, seed=seed)
    assert err <= 1e-12


@MID
@given(SEED, st.sampled_from([(32, 2, 8, 8, 4), (64, 4, 16, 16, 24), (48, 3, 8, 12, 6)]), st.integers(2, 20))
def test_mla_sub_mha(seed, dims, seq):
    """EP.mla_sub_mha: MHA with W_K = W_UK W_DKV, W_V = W_UV W_DKV equals the norm-free MLA head
    (attn/mla_rank.mla_absorption_error: explicit materialised MHA vs absorbed latent attention)."""
    from attn.mla_rank import mla_absorption_error
    hidden, H, nope, v, r = dims
    s_err, o_err = mla_absorption_error(hidden=hidden, n_head=H, qk_nope=nope, v_head=v, rank=r, seq=seq, seed=seed)
    # entries of x W_DKV W_UK are sums of hidden*rank normal products, |score| <~ 1e3: 1e4 roundings of it
    assert s_err <= fp(1e3, 1e4) * 10 and o_err <= fp(1e3, 1e4) * 10


def _masked_vs_subset(scores, allow, v):
    """softmax over all keys with the {0, -inf} mask  vs  softmax over the allowed keys only."""
    full = torch.softmax(scores.masked_fill(~allow, float("-inf")), -1) @ v
    sub = torch.stack([torch.softmax(scores[i, allow[i]], -1) @ v[allow[i]] for i in range(scores.shape[0])])
    return (full - sub).abs().max().item()


@MID
@given(SEED, st.integers(4, 32), st.sampled_from(["causal", "local", "strided", "block", "global", "sink",
                                                   "sliding", "dilated", "block_local"]))
def test_pattern_sub_full(seed, N, kind):
    """EP.pattern_sub_full: full attention with any 0/1 pattern as its mask IS the sparse target, for the
    repo's patterns (attn/sparse_patterns, block/mask_reach)."""
    from attn import sparse_patterns as sp
    from block import mask_reach as mr
    make = {"causal": lambda: sp.causal(N), "local": lambda: sp.local(N, 3), "strided": lambda: sp.strided(N, 3),
            "block": lambda: sp.block_sparse(N, 4), "global": lambda: sp.global_tokens(N, 2, 2),
            "sink": lambda: sp.sink(N, 2, 3), "sliding": lambda: mr.sliding(N, 4), "dilated": lambda: mr.dilated(N, 2, 3),
            "block_local": lambda: mr.block_local(N, 3)}[kind]
    allow = torch.from_numpy(make())
    g = gen(seed)
    scores, v = torch.randn(N, N, dtype=F64, generator=g) * 3, torch.randn(N, 5, dtype=F64, generator=g)
    assert _masked_vs_subset(scores, allow, v) <= fp(v.abs().max().item(), 10 * N)


@SLOW
@given(SEED, st.integers(2, 8), st.integers(4, 16))
def test_local_sub_full(seed, W, seq):
    """EP.local_sub_full: the full-attention Qwen2Model given the band as its attention_mask equals the
    sliding-window Qwen2Model with the same weights (hf/local_full.build)."""
    from hf import local_full
    eq, diff = local_full.build(W=W, seq=seq, seed=seed)
    assert eq <= 1e-12 and (diff > 1e-9 or seq <= W)


@MID
@given(SEED, st.integers(2, 8), st.integers(4, 20), FLOAT(0.5, 60))
def test_finite_mask_limit(seed, W, seq, B):
    """EP.finite_mask_limit: the {0, -inf} mask is the B -> inf limit of the additive mask -B (hf/local_full's
    band). The masked keys' total weight is at most n_masked exp(max masked s - B - logsumexp(allowed s)),
    and the output moves by at most twice that times max |v|. hf/local_full.band_mask itself uses
    B = finfo.max, where exp underflows to exactly 0."""
    from hf import local_full
    allow = local_full.band_mask(seq, W)[0, 0] == 0
    g = gen(seed)
    s, v = torch.randn(seq, seq, dtype=F64, generator=g), torch.randn(seq, 3, dtype=F64, generator=g)
    hard = torch.softmax(s.masked_fill(~allow, float("-inf")), -1) @ v
    soft = torch.softmax(s - B * (~allow).to(F64), -1) @ v
    lse = torch.logsumexp(s.masked_fill(~allow, float("-inf")), -1)
    mx = s.masked_fill(allow, float("-inf")).max(-1).values
    leak = ((~allow).sum(-1) * torch.exp(mx - B - lse)).nan_to_num(0.0)
    assert torch.all((soft - hard).abs().max(-1).values <= 2 * leak * v.abs().max() + fp(v.abs().max().item(), 10 * seq))
    banded = torch.softmax(s + local_full.band_mask(seq, W)[0, 0], -1) @ v
    assert (banded - hard).abs().max() <= fp(v.abs().max().item(), 10 * seq)


@MID
@given(SEED, st.integers(2, 6), FLOAT(1, 40))
def test_attn_bias_limit(seed, E, B):
    """EP.attn_bias_limit: attention with logits a_j + B c_j (all c_j <= 0, some = 0) tends to attention
    over {c_j = 0} as B -> inf; the leaked weight is <= sum_{c_j<0} exp(a_j + B c_j - logsumexp_{c=0} a).
    Checked on attn/qk_norm.attend (the repo's causal softmax), each query row a separate instance."""
    from attn.qk_norm import attend
    g = gen(seed)
    a = torch.randn(E, E, dtype=F64, generator=g)
    c = -torch.rand(E, E, dtype=F64, generator=g) * (torch.rand(E, E, generator=g) > 0.4)
    c.fill_diagonal_(0.0)                                           # every row keeps an unpenalised key
    A, A0 = attend(a + B * c), attend(a.masked_fill(c < 0, float("-inf")))
    causal = torch.tril(torch.ones(E, E, dtype=torch.bool))
    lse0 = torch.logsumexp(a.masked_fill((c < 0) | ~causal, float("-inf")), -1, keepdim=True)
    leak = torch.where((c < 0) & causal, torch.exp(a + B * c - lse0), torch.zeros_like(a)).sum(-1)
    assert torch.all((A - A0).abs().sum(-1) <= 2 * leak + fp(1, 10 * E))


@MID
@given(SEED, st.integers(2, 16), st.integers(1, 3))
def test_causal_stack(seed, T, L):
    """EP.causal_attn_layer, EP.Causal.comp, EP.causal_stack: bidirectional layers given the causal mask
    (attn/ar_diffusion.bidirectional_with_causal_mask) make a causal stack -- changing tokens after t never
    changes position t's output, bit for bit."""
    from attn.ar_diffusion import bidirectional_with_causal_mask as layer
    g = gen(seed)
    d = 6
    Ws = [torch.randn(3, d, d, dtype=F64, generator=g) / d ** 0.5 for _ in range(L)]

    def stack(x):
        for W in Ws:
            x = x + layer(x @ W[0], x @ W[1], torch.tanh(x @ W[2]))
        return x
    x = torch.randn(T, d, dtype=F64, generator=g)
    t = int(torch.randint(0, T, (1,), generator=g))
    y = x.clone()
    y[t + 1:] = torch.randn(T - t - 1, d, dtype=F64, generator=g)
    assert torch.equal(stack(x)[: t + 1], stack(y)[: t + 1])


@MID
@given(SEED, st.integers(1, 32), st.integers(1, 16), FLOAT(-8, 8))
def test_sink_sub_plain(seed, N, dq, b):
    """EP.sink_sub_plain: one extra key with logit b and value 0 equals off-by-one attention. The logits are
    realised as in the theorem -- query (q, 1), real keys (k_j, 0), sink key (0, b) -- and handed to
    attn/attention_sink.emulate_with_null_token, which appends that sink logit; compared to sink_attention."""
    from attn.attention_sink import emulate_with_null_token, sink_attention
    g = gen(seed)
    q, k, V = torch.randn(dq, dtype=F64, generator=g), torch.randn(N, dq, dtype=F64, generator=g), torch.randn(N, 4, dtype=F64, generator=g)
    q1 = torch.cat([q, torch.ones(1, dtype=F64)])
    keys = torch.cat([torch.cat([k, torch.zeros(N, 1, dtype=F64)], 1), torch.cat([torch.zeros(dq, dtype=F64), torch.tensor([b], dtype=F64)])[None]])
    logits = keys @ q1
    assert logits[-1].item() == b
    scores = logits[:-1] / dq ** 0.5
    got = emulate_with_null_token(scores, V, b)
    want = sink_attention(scores, V, b)
    assert (got - want).abs().max() <= fp(V.abs().max().item(), 10 * N)


@MID
@given(SEED, st.integers(1, 32), FLOAT(-60, 5))
def test_sink_to_plain_limit(seed, N, b):
    """EP.sink_to_plain_limit: as the sink logit b -> -inf, sink attention tends to plain attention; the gap
    is at most  (e^b / (e^b + sum e^s)) 2 max|v|  (attn/attention_sink.sink_attention)."""
    from attn.attention_sink import sink_attention
    g = gen(seed)
    s, V = torch.randn(N, dtype=F64, generator=g), torch.randn(N, 4, dtype=F64, generator=g)
    gap = (sink_attention(s, V, b) - torch.softmax(s, 0) @ V).abs().max().item()
    mass = math.exp(b) / (math.exp(b) + torch.exp(s).sum().item())
    assert gap <= 2 * mass * V.abs().max().item() + fp(V.abs().max().item(), 10 * N)


@MID
@given(SEED, st.sampled_from([4, 8]), st.integers(2, 10), st.sampled_from([8, 12]))
def test_k_eq_v_sub_separate(seed, hd, T, d):
    """EP.k_eq_v_sub_separate: W_V := W_K with V taken from the RAW k_proj output (before the k-norm and the
    rotary) and its own v-norm -- what elephantine/universal.from_gemma4 loads for attention_k_eq_v layers
    (wv = k_proj; UAttention.v_c). Checked on UAttention against a direct computation."""
    g = gen(seed)
    c = uni.UConfig(vocab=4, d=d, L=1, heads=1, head_dim=hd)
    a = uni.UAttention(c)
    wq, wk, wo = torch.randn(hd, d, dtype=F64, generator=g), torch.randn(hd, d, dtype=F64, generator=g), torch.randn(d, hd, dtype=F64, generator=g)
    kw = torch.rand(hd, dtype=F64, generator=g) + 0.5
    inv = torch.rand(hd // 2, dtype=F64, generator=g)
    with torch.no_grad():
        uni._attn_any(a, wq, wk, wk, wo, n_kv=1, scale=1.0, inv_freq=inv,
                      k_norm=types.SimpleNamespace(weight=kw, eps=1e-6))
        a.v_c.fill_(1.0)
        x = torch.randn(1, T, d, dtype=F64, generator=g)
        got = a(x, causal=True)[0]
    q, kraw = x[0] @ wq.T, x[0] @ wk.T
    rms = lambda t: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
    k = rms(kraw) * kw
    ang = torch.arange(T, dtype=F64)[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    rope = lambda t: t * cos + torch.cat([-t[:, hd // 2:], t[:, : hd // 2]], -1) * sin
    s = (rope(q) @ rope(k).T).masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    want = (torch.softmax(s, -1) @ rms(kraw)) @ wo.T
    assert (got - want).abs().max() <= fp((wo.abs().sum(1).max() * 3).item(), 100 * hd * T)


@FAST
@given(SEED, st.integers(1, 64), FLOAT(1e-8, 1.0))
def test_rmsNorm_bounded(seed, n, eps):
    """EP.rmsNorm_bounded: every coordinate of an RMSNorm output is at most |w_i| sqrt(n) (eps >= 0)
    (attn/qk_norm.rmsnorm, weight 1)."""
    from attn.qk_norm import rmsnorm
    x = torch.randn(n, dtype=F64, generator=gen(seed)) * 10 ** float(rng(seed).uniform(-3, 3))
    assert rmsnorm(x, eps).abs().max().item() <= math.sqrt(n) * (1 + 8 * U)


@MID
@given(SEED, st.integers(2, 12), FLOAT(0.0, 3.0))
def test_softmax_ratio_bound(seed, T, delta):
    """EP.softmax_ratio_bound: if |s_j - t_j| <= delta for every key, each softmax weight is within a factor
    exp(2 delta) (attn/qk_norm.attend, causal rows)."""
    from attn.qk_norm import attend
    g = gen(seed)
    t = torch.randn(T, T, dtype=F64, generator=g) * 3
    s = t + delta * (2 * torch.rand(T, T, dtype=F64, generator=g) - 1)
    A, B = attend(s), attend(t)
    assert torch.all(A <= math.exp(2 * delta) * B * (1 + 16 * U) + 1e-300)


# =====================================================================================================
# MoE.lean
# =====================================================================================================

from moe import dense_emulates_moe as dem        # noqa: E402
from moe import zero_router as zr                # noqa: E402


def _dense_ffn(seed, m, d):
    g = gen(seed)
    return (torch.randn(m, d, dtype=F64, generator=g) / d ** 0.5, torch.randn(m, d, dtype=F64, generator=g) / d ** 0.5,
            torch.randn(d, m, dtype=F64, generator=g) / m ** 0.5)


@SLOW
@given(SEED, st.sampled_from([("mixtral", {}), ("qwen2_moe", {}), ("qwen2_moe", {"norm_topk_prob": True})]))
def test_dense_sub_moe(seed, case):
    """EP.dense_sub_moe (with EP.zero_router_constant_gate): zero router weights give every token the same
    gates, nonzero on the k experts the implementation's tie-break picks; dealing the k e_w dense units into
    those experts (+ shared) and dividing each W_2 by its gate reproduces the dense FFN
    (moe/zero_router.deal_dense_into_moe on stock Mixtral / Qwen2-MoE blocks). transformers computes the
    routing weights in float32, so each gate is exact to 2^-24 relative."""
    fam, kw = case
    block, cfg = zr.make_block(fam, F64, **kw)
    d, k = cfg.hidden_size, cfg.num_experts_per_tok
    ew = zr._proj(block.experts[0])[0].weight.shape[0]
    sh = zr._shared(block)
    m = k * ew + (zr._proj(sh)[0].weight.shape[0] if sh is not None else 0)
    Wg, Wu, Wd = _dense_ffn(seed, m, d)
    picked, weights = zr.deal_dense_into_moe(block, Wg, Wu, Wd)
    x = torch.randn(2, 9, d, dtype=F64, generator=gen(seed + 1))
    with torch.no_grad():
        logits = block.gate(x.view(-1, d))
        assert torch.equal(logits, logits[:1].expand_as(logits))              # zero_router_constant_gate
        out = block(x)
        out = out[0] if isinstance(out, tuple) else out
    dense = F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)
    assert len(picked) == k
    assert (out - dense).abs().max() <= 8 * k * U32 * (1 + dense.abs().max().item())


@SLOW
@given(st.sampled_from([4, 8, 16]), st.sampled_from([1, 2]))
def test_softmax_const(E, k):
    """EP.softmax_const: softmax over constant logits is 1/E -- the gate moe/zero_router.probe_router
    measures on a Qwen2-MoE block with a zeroed router and norm_topk_prob = False (float32 routing)."""
    from transformers import Qwen2MoeConfig
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
    torch.manual_seed(0)
    cfg = Qwen2MoeConfig(hidden_size=16, moe_intermediate_size=8, shared_expert_intermediate_size=8,
                         num_experts=E, num_experts_per_tok=k, norm_topk_prob=False)
    block = Qwen2MoeSparseMoeBlock(cfg).double().eval()
    with torch.no_grad():
        block.gate.weight.zero_()
        block.shared_expert_gate.weight.zero_()
    picked, w = zr.probe_router(block, 16, F64)
    assert len(picked) == k and all(abs(w[j] - 1 / E) <= 4 * U32 for j in picked)


@SLOW
@given(SEED)
def test_condition_dense_sub_moe_needs_the_real_tie_break(seed):
    """EP.dense_sub_moe deals into the experts sel(b) the IMPLEMENTATION picks. Dealing into experts 0..k-1
    (the naive reading) drops the FFN on a stock Mixtral block, which picks others at a tie."""
    block, cfg = zr.make_block("mixtral", F64)
    d, k = cfg.hidden_size, cfg.num_experts_per_tok
    ew = zr._proj(block.experts[0])[0].weight.shape[0]
    Wg, Wu, Wd = _dense_ffn(seed, k * ew, d)
    probed, _ = zr.deal_dense_into_moe(block, Wg, Wu, Wd, picked=[0, 1])
    assert sorted(probed) != [0, 1]
    x = torch.randn(1, 9, d, dtype=F64, generator=gen(seed + 1))
    with torch.no_grad():
        out = block(x)
        out = out[0] if isinstance(out, tuple) else out
    dense = F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)
    assert (out - dense).abs().max() > 1e-3 * dense.abs().max()


@SLOW
@given(SEED, st.sampled_from([(8, 1), (8, 2), (4, 4)]))
def test_tied_moe_is_dense(seed, Ek):
    """EP.tied_moe_is_dense: identical experts and gates summing to 1 (Mixtral renormalises its top-k) give
    the expert FFN for ANY router (hf/moe_dense.build: random router, stock MixtralSparseMoeBlock). The
    float32 routing makes sum gates = 1 to ~k 2^-24."""
    from hf import moe_dense
    E, k = Ek
    assert moe_dense.build(E=E, k=k, seed=seed) <= 1e-5


@SLOW
@given(SEED, st.sampled_from([(8, 2), (4, 1), (8, 4)]))
def test_condition_tied_moe_needs_gates_summing_to_one(seed, Ek):
    """Condition 2: without renormalisation (Qwen2-MoE norm_topk_prob = False) a tied MoE with a live router
    outputs (sum of its top-k probabilities)(x) f(x), varying per token -- not f(x). The repo's MoE hosts
    avoid this by zeroing the router first (moe/zero_router.deal_dense_into_moe mode='replicate' divides by
    the probed gate sum; elephantine/emulate.build_host scales down_proj by E / top_k): exact again.

    No heuristic threshold: the test checks the statement itself. With S_t the sum of token t's top-k
    softmax probabilities (computed as the block does: float32 softmax, cast back), out_t = S_t f_t to
    float64 rounding, so |out_t - f_t| = (1 - S_t) |f_t| exactly up to that rounding. For k < E, 1 - S_t is
    the excluded experts' mass, > 0 in exact arithmetic (in float32 a token's can round to 0, or S_t to just
    above 1, so it is not asserted per token). The test asserts the gap is at least
    max_t (1 - S_t)|f_t| - tol, and that this lower bound is positive (Hypothesis skips a draw where no
    token's excluded mass survives float32). tol is a Higham bound: the expert MLP (16 + 12 + a few terms per dot product) evaluated on
    a token subset vs the dense reference, then k weighted adds, each within gamma_64 of the absolute-value
    forward M_t = |Wd| ((|Wg| |x|) * (|Wu| |x|))."""
    from transformers import Qwen2MoeConfig
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
    E, k = Ek
    torch.manual_seed(seed % 1000)
    cfg = Qwen2MoeConfig(hidden_size=16, moe_intermediate_size=12, shared_expert_intermediate_size=8,
                         num_experts=E, num_experts_per_tok=k, norm_topk_prob=False)
    block = Qwen2MoeSparseMoeBlock(cfg).double().eval()
    Wg, Wu, Wd = _dense_ffn(seed, 12, 16)
    with torch.no_grad():
        block.gate.weight.copy_(torch.randn(E, 16, dtype=F64, generator=gen(seed)))
        for p in zr._proj(block.shared_expert):
            p.weight.zero_()
        for e in block.experts:
            for p, w in zip(zr._proj(e), (Wg, Wu, Wd)):
                p.weight.copy_(w)
        x = torch.randn(1, 9, 16, dtype=F64, generator=gen(seed + 1))
        dense = F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)
        out = block(x)[0] if isinstance(block(x), tuple) else block(x)
        probs = F.softmax(block.gate(x.view(-1, 16)), dim=1, dtype=torch.float)    # as the block computes them
        S = torch.topk(probs, k, dim=-1).values.to(F64).sum(-1).view(1, 9, 1)
        M = F.linear(F.linear(x.abs(), Wg.abs()) * F.linear(x.abs(), Wu.abs()), Wd.abs())
        gk = lambda k_: k_ * U / (1 - k_ * U)
        tol = 2 * (k + 1) * gk(64) * M
        assert ((out - S * dense).abs() <= tol).all()                  # the statement: out = S(x) f(x)
        if k < E:
            lower = ((1 - S) * dense.abs() - tol).max().item()          # |out - f| >= (1 - S)|f| - tol
            assume(lower > 0)
            assert (out - dense).abs().max().item() >= lower
        zr.deal_dense_into_moe(block, Wg, Wu, Wd, mode="replicate")
        out = block(x)[0] if isinstance(block(x), tuple) else block(x)
    assert (out - dense).abs().max() <= 8 * k * U32 * (1 + dense.abs().max().item())


def _moe_case(seed, T=24, d=8, ew=6, E=4):
    g = gen(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    return r(T, d), r(E, d, ew), r(E, ew, d), r(d, E)


@MID
@given(SEED, st.integers(2, 8))
def test_dense_hard_select(seed, E):
    """EP.dense_hard_select: the one-hot select of the argmax picks Y_{e*} exactly, in width (hold every
    expert + a select layer) and in depth (E masked residual FFN sublayers)
    (moe/dense_emulates_moe.dense_hardselect / dense_masked_depth vs moe_top1)."""
    x, W1, W2, Wg = _moe_case(seed, E=E)
    ref = dem.moe_top1(x, W1, W2, Wg)
    scale = dem.experts_all(x, W1, W2).abs().max().item()
    assert (dem.dense_hardselect(x, W1, W2, Wg) - ref).abs().max() <= fp(scale, 10 * E)
    assert (dem.dense_masked_depth(x, W1, W2, Wg) - ref).abs().max() <= fp(scale, 100 * E)


@MID
@given(SEED, st.integers(2, 8), FLOAT(0.1, 2000))
def test_dense_soft_router_limit(seed, E, beta):
    """EP.dense_soft_router_limit: with a UNIQUE argmax, sum_e softmax(beta z)_e Y_e -> Y_{e*}; the gap is at
    most (E - 1) exp(-beta gap_z) 2 max|Y| (moe/dense_emulates_moe.dense_softrouter vs moe_top1)."""
    x, W1, W2, Wg = _moe_case(seed, E=E)
    z = x @ Wg
    top2 = z.topk(2, -1).values
    margin = (top2[:, 0] - top2[:, 1])                                      # > 0 a.s.: unique argmax
    Y = dem.experts_all(x, W1, W2)
    err = (dem.dense_softrouter(x, W1, W2, Wg, beta) - dem.moe_top1(x, W1, W2, Wg)).abs().max(-1).values
    bound = (E - 1) * torch.exp(-beta * margin) * 2 * Y.abs().amax((1, 2))
    assert torch.all(err <= bound + fp(Y.abs().max().item(), 10 * E))


@MID
@given(SEED, FLOAT(0.0, 1e4))
def test_soft_router_tie(seed, beta):
    """EP.soft_router_tie (condition 1): with two experts tied for the max, softmax(beta z) stays (1/2, 1/2)
    for EVERY beta, so the smooth router returns the average and never the tie-broken pick
    (moe/dense_emulates_moe.dense_softrouter with two equal router columns)."""
    x, W1, W2, Wg = _moe_case(seed, E=2)
    Wg = torch.cat([Wg[:, :1], Wg[:, :1]], 1)
    Y = dem.experts_all(x, W1, W2)
    soft = dem.dense_softrouter(x, W1, W2, Wg, beta)
    assert (soft - (Y[:, 0] + Y[:, 1]) / 2).abs().max() <= fp(Y.abs().max().item(), 10)
    gap = (soft - dem.moe_top1(x, W1, W2, Wg)).abs().max()
    assert gap >= 0.99 * (Y[:, 0] - Y[:, 1]).abs().max() / 2


@MID
@given(SEED, st.integers(2, 8))
def test_condition_top1_unit_gate(seed, E):
    """Condition 1 (second half): the select construction reproduces a top-1 MoE with a UNIT gate
    (renormalised top-1; what moe_top1 defines). A Switch-style top-1 that multiplies by the router
    probability p_{e*}(x) is a different function; the hard select misses it by (1 - p_{e*}) |Y_{e*}|."""
    x, W1, W2, Wg = _moe_case(seed, E=E)
    p = torch.softmax(x @ Wg, -1)
    switch = dem.moe_top1(x, W1, W2, Wg) * p.max(-1, keepdim=True).values
    gap = (dem.dense_hardselect(x, W1, W2, Wg) - switch).abs().max(-1).values
    want = (1 - p.max(-1).values) * dem.moe_top1(x, W1, W2, Wg).abs().max(-1).values
    assert torch.allclose(gap, want, rtol=1e-12, atol=fp(1.0, 100))
    assert gap.max() > 1e-3


# =====================================================================================================
# Activation.lean
# =====================================================================================================

@MID
@given(SEED)
def test_polarization(seed):
    """EP.polarization: p b = ((p + b)^2 - (p - b)^2) / 4 -- a bilinear GLU as a squared FFN of 2x width
    (act/relu_swiglu.check_bilinear_as_square). |a b| ~ 64, summed over m = 128 units into |y| ~ 1e3."""
    from act.relu_swiglu import check_bilinear_as_square
    (err,) = printed_errs(check_bilinear_as_square, seed=seed)
    assert err <= fp(1e4, 1e3)


@MID
@given(SEED)
def test_swiglu_eq_sq_ffn(seed):
    """EP.swiglu_eq_sq_ffn: sum_u p_u b_u w_u equals the squared-activation FFN with units (p +- b)^2 and
    down columns +-w / 4 (act/relu_swiglu.check_swiglu_needs_depth; same magnitudes as test_polarization)."""
    from act.relu_swiglu import check_swiglu_needs_depth
    (err,) = printed_errs(check_swiglu_needs_depth, seed=seed)
    assert err <= fp(1e4, 1e3)


@FAST
@given(SEED, FLOAT(0.5, 8))
def test_yarotsky_product_via_square(seed, scale):
    """EP.polarization on act/relu_net.gated_gate's product: 0.25 (square(p + b) - square(p - b)) with
    exact squares is p b (the depth-L squares only add the Yarotsky error, tested below)."""
    from act import relu_net as rn
    p, b = rng(seed).uniform(-scale, scale, 2)
    assert abs(0.25 * ((p + b) ** 2 - (p - b) ** 2) - p * b) <= fp(scale ** 2, 8)


@MID
@given(SEED)
def test_glu_ungated(seed):
    """EP.glu_ungated: with the up branch held at 1 (zero weights, bias 1) a gated unit is the plain
    activation (act/relu_swiglu.check_glu_superset_vanilla: multiplying by 1.0 is exact)."""
    from act.relu_swiglu import check_glu_superset_vanilla
    assert printed_errs(check_glu_superset_vanilla, seed=seed) == [0.0]


@MID
@given(SEED)
def test_sigmoid_scale_fold(seed):
    """EP.sigmoid_scale_fold: g sigmoid(a g) = SiLU(a g) / a at a = 1.702 (act/exotic_activations.
    swigluoai_folds_into_silu, |g| <~ 8)."""
    from act.exotic_activations import swigluoai_folds_into_silu
    assert swigluoai_folds_into_silu(seed) <= fp(8, 20)


ALPHA, LIM = 1.702, 7.0


def _swigluoai(g, u):
    """gpt-oss's unit as transformers' GptOssExperts computes it (and EP.swigluoai states it)."""
    g = g.clamp(max=LIM)
    return (u.clamp(-LIM, LIM) + 1) * g * torch.sigmoid(ALPHA * g)


def _uffn_oai():
    f = uni.UFFN(uni.UConfig(vocab=4, d=4, L=1))
    with torch.no_grad():                         # what elephantine/universal.from_gpt_oss sets
        f.c_clip.fill_(1.0)
        f.clip_lim.copy_(torch.tensor([ALPHA * LIM, LIM], dtype=F64))
        f.up_shift.fill_(1.0)
    return f


@FAST
@given(FLOAT(-30, LIM), FLOAT(-LIM, LIM))
def test_swigluoai_sub_silu(g, u):
    """EP.swigluoai_sub_silu: for gate <= 7 and |up| <= 7, swigluoai = (up + 1) SiLU(1.702 g) / 1.702 --
    the universal host's SwiGLU unit with 1.702 folded into the gate rows, +1 into up_shift and 1/1.702
    into w_down (elephantine/universal.from_gpt_oss, UFFN._act)."""
    f = _uffn_oai()
    G, Uu = torch.tensor([g], dtype=F64), torch.tensor([u], dtype=F64)
    with torch.no_grad():
        host = f._act(ALPHA * G, Uu) / ALPHA
    rhs = (Uu + 1) * F.silu(ALPHA * G) / ALPHA
    scale = (abs(u) + 1) * (abs(g) + 1)
    assert abs((host - rhs).item()) <= fp(scale, 20) and abs((_swigluoai(G, Uu) - rhs).item()) <= fp(scale, 20)


@SLOW
@given(SEED, FLOAT(1, 60))
def test_swigluoai_sub_silu_on_gpt_oss(seed, gate_scale):
    """EP.swigluoai_sub_silu through the loader: elephantine/universal.from_gpt_oss on a stock
    GptOssForCausalLM (fp64; gate / up scaled so that some units sit outside the clamp regime, where the
    host's own clip takes over), layer by layer: host FFN == the stock MoE MLP. The stock router runs its
    softmax in float32."""
    from transformers import GptOssConfig, GptOssForCausalLM
    torch.manual_seed(seed % 10000)
    cfg = GptOssConfig(vocab_size=31, hidden_size=16, intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=8, num_local_experts=4, num_experts_per_tok=2,
                       sliding_window=4, layer_types=["full_attention"], attn_implementation="eager")
    t = GptOssForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():
        for p in t.parameters():
            p.add_(0.3 * torch.randn_like(p))
        t.model.layers[0].mlp.experts.gate_up_proj.mul_(gate_scale)
    h = uni.from_gpt_oss(t)
    x = torch.randn(1, 7, 16, dtype=F64, generator=gen(seed))
    with torch.no_grad():
        ref = t.model.layers[0].mlp(x)
        ref = ref[0] if isinstance(ref, tuple) else ref
        got = h.layers[0].ffn(x)
    assert (got - ref).abs().max() <= 16 * U32 * (1 + ref.abs().max().item())


@FAST
@given(FLOAT(-LIM, LIM), FLOAT(-LIM, LIM))
def test_condition_swigluoai_needs_up_bias(g, u):
    """Condition 5: the fold needs an up-projection bias (or a constant channel) for the +1. A bias-free
    SiLU host (up_shift = 0) is off by exactly SiLU(1.702 g) / 1.702."""
    f = _uffn_oai()
    with torch.no_grad():
        f.up_shift.fill_(0.0)
        G, Uu = torch.tensor([g], dtype=F64), torch.tensor([u], dtype=F64)
        gap = (f._act(ALPHA * G, Uu) / ALPHA - _swigluoai(G, Uu)).item()
    assert abs(gap + (F.silu(ALPHA * G) / ALPHA).item()) <= fp((abs(u) + 1) * (abs(g) + 1), 20)


@FAST
@given(FLOAT(-30, 30), FLOAT(-30, 30))
def test_condition_swigluoai_clamp_regime(g, u):
    """Condition 5 (regime): (up + 1) SiLU(1.702 g) / 1.702 equals swigluoai only while gate <= 7 and
    |up| <= 7. Outside it the target's clamps bite and a plain SiLU host (no clip) differs."""
    G, Uu = torch.tensor([g], dtype=F64), torch.tensor([u], dtype=F64)
    plain = (Uu + 1) * F.silu(ALPHA * G) / ALPHA
    gap = abs((plain - _swigluoai(G, Uu)).item())
    if g <= LIM and abs(u) <= LIM:
        assert gap <= fp((abs(u) + 1) * (abs(g) + 1), 20)
    elif (g > LIM + 0.1 and abs(u + 1) > 0.1) or (abs(u) > LIM + 0.1 and g > 0.1):
        assert gap > 1e-3


@FAST
@given(FLOAT(-50, 50), FLOAT(0.01, 1e4))
def test_relu_silu_bound(x, beta):
    """EP.relu_silu_bound: |SiLU(beta x) / beta - ReLU(x)| <= |x| e^{-beta |x|} <= 1 / beta
    (act/activations.swish_b(x, beta) = x sigmoid(beta x) = SiLU(beta x) / beta; act/relu_swiglu.swish_beta)."""
    from act.activations import swish_b
    from act.relu_swiglu import swish_beta
    X = torch.tensor([x], dtype=F64)
    for fn in (swish_b, swish_beta):
        gap = abs((fn(X, beta) - F.relu(X)).item())
        assert gap <= abs(x) * math.exp(-beta * abs(x)) + fp(abs(x), 8)
        assert gap <= 1 / beta + fp(abs(x), 8)


@MID
@given(st.floats(1.0, 1e6))
def test_relu_sub_silu(beta):
    """EP.relu_sub_silu: SiLU(beta x) / beta -> ReLU(x) uniformly in x (max over a grid <= 1 / beta -> 0)."""
    from act.activations import swish_b
    x = torch.linspace(-50, 50, 20001, dtype=F64)
    assert (swish_b(x, beta) - F.relu(x)).abs().max().item() <= 1 / beta + fp(50, 8)


@FAST
@given(FLOAT(-50, 50), FLOAT(0.01, 1e4))
def test_relu_sub_gelu(x, beta):
    """EP.relu_sub_gelu: GeLU(beta x) / beta = x Phi(beta x) -> ReLU(x). The gap is |x| Phi(-beta |x|) <=
    phi(0) / beta by Mills' ratio, so the limit is uniform (act/activations.gelu_b)."""
    from act.activations import gelu_b
    X = torch.tensor([x], dtype=F64)
    gap = abs((gelu_b(X, beta) - F.relu(X)).item())
    assert gap <= 1 / (beta * math.sqrt(2 * math.pi)) + fp(abs(x), 8)


@FAST
@given(FLOAT(-1e3, 1e3), FLOAT(0.1, 100))
def test_softcap_bound(z, c):
    """EP.softcap_bound: |c tanh(z / c) - z| <= |z|^3 / (3 c^2) (attn/softcap.softcap; the universal host's
    UAttention.cap computes the same expression)."""
    from attn.softcap import softcap
    gap = abs(softcap(torch.tensor([z], dtype=F64), c).item() - z)
    assert gap <= abs(z) ** 3 / (3 * c ** 2) + fp(abs(z), 8)


@MID
@given(SEED, st.integers(2, 24), FLOAT(0.5, 50), FLOAT(0.01, 2.0))
def test_softcap_attn_regime(seed, T, c, ratio):
    """EP.softcap_attn_regime: if every logit has |z| <= R, the plain host's attention weights are within a
    factor exp(2 R^3 / (3 c^2)) of the capped model's, both ways (attn/softcap.softcap, attn/qk_norm.attend)."""
    from attn.qk_norm import attend
    from attn.softcap import softcap
    z = torch.randn(T, T, dtype=F64, generator=gen(seed))
    z = z / z.abs().max() * ratio * c
    R = z.abs().max().item()
    A, B = attend(z), attend(softcap(z, c))
    f = math.exp(2 * R ** 3 / (3 * c ** 2)) * (1 + 32 * U)
    assert torch.all(A <= f * B + 1e-300) and torch.all(B <= f * A + 1e-300)


@FAST
@given(SEED, st.integers(1, 64), FLOAT(0.05, 20), FLOAT(0.05, 20))
def test_qknorm_logit(seed, n, rq, rk):
    """EP.qknorm_logit: the qk-norm logit is (rq rk) / (rms q rms k) times the plain logit of q / rq, k / rk,
    for any folded constants rq, rk (attn/qk_norm.rmsnorm, eps 1e-6, weight 1)."""
    from attn.qk_norm import rmsnorm
    q, k = torch.randn(2, n, dtype=F64, generator=gen(seed)) * 3
    rms = lambda t: (t.pow(2).mean() + 1e-6).sqrt()
    lhs = rmsnorm(q) @ rmsnorm(k)
    rhs = rq * rk / (rms(q) * rms(k)) * ((q / rq) @ (k / rk))
    assert abs((lhs - rhs).item()) <= fp(n, 4 * n)


@FAST
@given(SEED, st.integers(1, 64))
def test_qknorm_sub_plain_regime(seed, n):
    """EP.qknorm_sub_plain_regime: on tokens whose rms equal the folded constants, the plain logit with
    q / rq, k / rk IS the qk-norm logit (attn/qk_norm.rmsnorm; the eps belongs inside rms). Note
    attn/qk_norm.part2_fold_vs_variance folds the mean of the eps-FREE rms, so its 'exact at CV = 0' is
    exact only to a relative eps / ms ~ 1e-6."""
    from attn.qk_norm import rmsnorm
    q, k = torch.randn(2, n, dtype=F64, generator=gen(seed)) * 3
    rq, rk = (q.pow(2).mean() + 1e-6).sqrt(), (k.pow(2).mean() + 1e-6).sqrt()
    assert abs((rmsnorm(q) @ rmsnorm(k) - (q / rq) @ (k / rk)).item()) <= fp(n, 4 * n)


# =====================================================================================================
# Approx.lean
# =====================================================================================================

from act import relu_net as rn                   # noqa: E402


def _smooth(seed):
    """f = A sin(w x + p) + B x^2, |f''| <= A w^2 + 2|B| =: M."""
    A, w, p, B = rng(seed).uniform([0.1, 0.1, -3, -0.5], [3, 4, 3, 0.5])
    return (lambda x: A * np.sin(w * x + p) + B * x * x), A * w * w + 2 * abs(B)


@MID
@given(SEED, st.integers(1, 64), FLOAT(-8, 0), FLOAT(0.5, 8))
def test_reluNet_on_piece(seed, N, lo, width):
    """EP.reluNet_on_piece: one hidden layer of ReLUs with second-difference output weights is exactly the
    piecewise-linear interpolant (act/relu_net.relu_units / relu_net vs pw_linear's np.interp)."""
    f, _ = _smooth(seed)
    hi = lo + width
    x = np.linspace(lo, hi, 2001)
    net, interp = rn.relu_net(rn.relu_units(f, N, lo, hi), x), rn.pw_linear(f, N, lo, hi)(x)
    scale = np.abs(interp).max() + np.abs(np.diff(f(np.linspace(lo, hi, N + 1)))).sum() * 4
    assert np.abs(net - interp).max() <= fp(scale, 20 * N)


@MID
@given(SEED, st.integers(1, 200), FLOAT(-8, 0), FLOAT(0.5, 8))
def test_relu_net_approx(seed, N, lo, width):
    """EP.relu_net_approx / EP.interp_error_h2: if |f''| <= M, the N-unit ReLU net (act/relu_net.pw_linear) is
    within M (b - a)^2 / (8 N^2) of f on [a, b]."""
    f, M = _smooth(seed)
    hi = lo + width
    x = np.linspace(lo, hi, 4001)
    err = np.abs(rn.pw_linear(f, N, lo, hi)(x) - f(x)).max()
    assert err <= M * width ** 2 / (8 * N ** 2) + fp(np.abs(f(x)).max(), 50)


@MID
@given(SEED, st.integers(1, 64))
def test_relu_net_approx_ungated_error(seed, N):
    """EP.relu_net_approx on act/relu_net.ungated_error (the repo's measured width law, [-8, 8], 20000 points)."""
    f, M = _smooth(seed)
    assert rn.ungated_error(f, N) <= M * 16 ** 2 / (8 * N ** 2) + fp(60, 50)


@MID
@given(SEED, st.integers(1, 64), FLOAT(-8, 0), FLOAT(0.5, 8))
def test_interp_error(seed, N, lo, width):
    """EP.interp_error: on each piece [u, v] the chord misses f by at most M (x - u)(v - x) / 2."""
    f, M = _smooth(seed)
    hi = lo + width
    t = np.linspace(lo, hi, N + 1)
    x = np.linspace(lo, hi, 4001)
    piece = np.clip(np.searchsorted(t, x, side="right") - 1, 0, N - 1)
    u, v = t[piece], t[piece + 1]
    err = np.abs(rn.pw_linear(f, N, lo, hi)(x) - f(x))
    assert np.all(err <= M * (x - u) * (v - x) / 2 + fp(np.abs(f(x)).max(), 50))


@FAST
@given(FLOAT(0, 1), st.integers(0, 20))
def test_yarotsky_sq_bound(x, m):
    """EP.yarotsky_sq_bound: 0 <= f_m(x) - x^2 <= 4^-m / 4 on [0, 1] (act/relu_net.yarotsky_sq01)."""
    gap = float(rn.yarotsky_sq01(np.array([x]), m)[0]) - x * x
    assert -fp(1, 8 * (m + 1)) <= gap <= 4.0 ** -m / 4 + fp(1, 8 * (m + 1))


@FAST
@given(FLOAT(0, 1), st.integers(0, 20))
def test_yarotsky_exact(x, m):
    """EP.yarotsky_exact: f_m(x) - x^2 = g^m(x) (1 - g^m(x)) / 4^m, g the tooth 2 ReLU(x) - 4 ReLU(x - 1/2)
    (act/relu_net.yarotsky_sq01 and act/relu_net._tent)."""
    gx = np.array([x])
    for _ in range(m):
        gx = rn._tent(gx)
    want = float(gx[0] * (1 - gx[0])) / 4.0 ** m
    assert abs(float(rn.yarotsky_sq01(np.array([x]), m)[0]) - x * x - want) <= fp(1, 8 * (m + 1)) + 2 ** m * 4 * U / 4.0 ** m


@FAST
@given(FLOAT(-1, 1), st.integers(0, 20))
def test_yarotsky_sym_bound(x, m):
    """EP.yarotsky_sym_bound: x^2 on [-1, 1] through |x| = ReLU(x) + ReLU(-x) (act/relu_net.square, M = 1):
    0 <= square - x^2 <= 4^-m / 4."""
    gap = float(rn.square(np.array([x]), m, 1.0)[0]) - x * x
    assert -fp(1, 8 * (m + 1)) <= gap <= 4.0 ** -m / 4 + fp(1, 8 * (m + 1))


@FAST
@given(FLOAT(-1, 1), FLOAT(-1, 1), st.integers(0, 20))
def test_yarotsky_product_bound(p, b, m):
    """EP.yarotsky_product_bound: for |p|, |b| <= 1, p b = ((p+b)/2)^2 - ((p-b)/2)^2 from two depth-m squares is
    within 1 / (2 4^m) -- act/relu_net.gated_gate's product step 0.25 (square(p+b) - square(p-b)) at M = 2."""
    prod = 0.25 * (float(rn.square(np.array([p + b]), m, 2.0)[0]) - float(rn.square(np.array([p - b]), m, 2.0)[0]))
    assert abs(prod - p * b) <= 1 / (2 * 4.0 ** m) + fp(1, 16 * (m + 1))


# =====================================================================================================
# Block.lean
# =====================================================================================================

@SLOW
@given(SEED)
def test_seq_sub_parallel(seed):
    """EP.seq_sub_parallel / EP.zeroed_sublayer: an attention-only parallel block (MLP output zeroed) and then
    an MLP-only one (attention output zeroed) equal one sequential block, pre-norms included
    (block/prenorm_carry.parallel_hosts_seq_2x_depth: stock GPT-NeoX layers, real LayerNorms, fp64; the
    eager softmax rounds to float32 identically on both sides)."""
    from block.prenorm_carry import parallel_hosts_seq_2x_depth
    err, scale = parallel_hosts_seq_2x_depth(F64, seed=seed)
    assert err <= fp(scale, 1e4)


@MID
@given(SEED)
def test_seq_sub_parallel_toy(seed):
    """EP.seq_sub_parallel and EP.parallel_sub_seq_wide (L = 1) in block/parallel_residual.
    verify_both_constructions (norm-free toy, fp64; |y| ~ 1e2)."""
    from block.parallel_residual import verify_both_constructions
    errs = printed_errs(verify_both_constructions, seed=seed)
    assert len(errs) == 2 and max(errs) <= fp(1e3, 1e3)


def _sublayers(seed, L, d=8):
    g = gen(seed)
    mk = lambda: (torch.randn(d, d, dtype=F64, generator=g) / d ** 0.5, torch.rand(d, dtype=F64, generator=g) + 0.5)
    As, Ms = [], []
    for _ in range(L):
        (Wa, na), (Wm, nm) = mk(), mk()
        As.append(lambda z, Wa=Wa, na=na: torch.tanh(z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * na @ Wa))
        Ms.append(lambda z, Wm=Wm, nm=nm: F.silu(z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * nm @ Wm))
    return As, Ms


@MID
@given(SEED, st.integers(1, 6))
def test_parallel_sub_seq_wide(seed, L):
    """EP.parallel_sub_seq_wide: L parallel blocks in L sequential blocks at (L + 1) d width, norm-free reads
    (block/parallel_residual.parallel_stack_in_seq_wide vs parallel_stack; sublayers normalise their own
    linear readout)."""
    from block.parallel_residual import parallel_stack, parallel_stack_in_seq_wide
    As, Ms = _sublayers(seed, L)
    x = torch.randn(5, 8, dtype=F64, generator=gen(seed + 1))
    want = parallel_stack(x, As, Ms)
    assert (parallel_stack_in_seq_wide(x, As, Ms) - want).abs().max() <= fp(want.abs().max().item(), 100 * L)


@MID
@given(SEED, st.integers(2, 5))
def test_condition_parallel_sub_seq_width(seed, L):
    """Condition 7: one reserved channel (2d width) carries ONE block. Reused across L >= 2 blocks, block l's
    MLP readout 'every channel but the reserved one' no longer equals block l's input (the earlier
    attention writes are still in it), and the stack is off."""
    from block.parallel_residual import parallel_stack
    As, Ms = _sublayers(seed, L)
    x = torch.randn(5, 8, dtype=F64, generator=gen(seed + 1))
    main, res = x, torch.zeros_like(x)
    for A, M in zip(As, Ms):                         # 2d: attention always writes the same reserved channel
        res = res + A(main + res)
        main = main + M(main)
    assert (main + res - parallel_stack(x, As, Ms)).abs().max() > 1e-6


@SLOW
@given(SEED)
def test_condition_parallel_sub_seq_needs_norm_free(seed):
    """Condition 7 (pre-norm): on a stock LlamaDecoderLayer of width 2d with a single-rms pre-norm the
    construction is not exact (block/prenorm_carry.seq_hosts_parallel); with the per-group RMSNorm reader it
    is (the norm-free form EP.parallel_sub_seq_wide proves), up to the eager softmax's float32 rounding
    (host and target logits differ in their last float64 bits, which the float32 cast can amplify to 2^-24)."""
    from block.prenorm_carry import seq_hosts_parallel
    e_stock, s = seq_hosts_parallel(F64, group_norm=False, seed=seed)
    e_group, s2 = seq_hosts_parallel(F64, group_norm=True, seed=seed)
    assert e_stock > 1e-4 * s and e_group <= 16 * U32 * (1 + s2)


@SLOW
@given(SEED, st.sampled_from([(1, 3), (2, 2), (3, 2), (1, 5)]), st.integers(2, 12))
def test_looped_sub_untied(seed, bK, seq):
    """EP.looped_sub_untied: K loops of a b-layer block equal a Kb-layer untied stack with theta_{t+b} =
    theta_t (hf/looped_untied.build: a stock LlamaModel, tied periodically, vs the block's own layers looped)."""
    from hf.looped_untied import build
    b, K = bK
    assert build(b=b, K=K, seq=seq, seed=seed) <= 1e-12


def test_mtp_sub_plain():
    """EP.mtp_sub_plain: the main logits do not depend on modules that never write back into the trunk
    (hf/real_reductions.check_mimo_mtp: MiMo-7B's MTP tensors are disjoint from the base path and it has no
    head of its own; structural, from the cached weight index)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from hf.real_reductions import check_mimo_mtp
    (row,) = check_mimo_mtp()
    if "[SKIP" in row["model"]:                    # hf.real_reductions._err: the index is not cached
        pytest.skip(f"MiMo-7B-Base index not available: {row['note']}")
    assert row["ok"] and row["worst"] == 0.0


@FAST
@given(SEED, st.sampled_from([(1, 2), (2, 4), (1, 4), (4, 8), (2, 8), (3, 6)]), st.integers(1, 6), st.integers(1, 6))
def test_dense_sub_nm(seed, nm, rows, groups):
    """EP.dense_sub_nm: the dense columns placed on N of every M slots, input spread the same way, is N:M
    sparse and computes W x exactly (convert/nm_sparsity.dense_to_wide_nm, for N | M)."""
    from convert.nm_sparsity import dense_to_wide_nm, max_nonzeros_per_group
    n, m = nm
    W = torch.randn(rows, groups * n, dtype=F64, generator=gen(seed))
    x = torch.randn(3, groups * n, dtype=F64, generator=gen(seed + 1))
    S, expand = dense_to_wide_nm(W, n, m)
    assert S.shape[1] == groups * m and max_nonzeros_per_group(S, m) <= n
    assert (expand(x) @ S.T - x @ W.T).abs().max() <= fp((W.abs().sum(1).max() * x.abs().max()).item(), 4 * groups * n)


@MID
@given(SEED, st.sampled_from([(3, 4), (2, 3), (3, 5)]))
def test_dense_sub_nm_when_N_does_not_divide_M(seed, nm):
    """(Was MISMATCH 2, fixed in round 2, pass 5.) convert/nm_sparsity.dense_to_wide_nm places each group's N dense
    columns on N of the M slots (width M/N, groups of N columns) for any N <= M, as EP.dense_sub_nm: N:M sparse and
    exact also when N does not divide M (3:4, 2:3, 3:5)."""
    from convert.nm_sparsity import dense_to_wide_nm, max_nonzeros_per_group
    n, m = nm
    W = torch.randn(4, 4 * m, dtype=F64, generator=gen(seed))
    x = torch.randn(3, 4 * m, dtype=F64, generator=gen(seed + 1))
    S, expand = dense_to_wide_nm(W, n, m)
    assert max_nonzeros_per_group(S, m) <= n
    assert (expand(x) @ S.T - x @ W.T).abs().max() <= fp((W.abs().sum(1).max() * x.abs().max()).item(), 16 * m)


@FAST
@given(SEED, st.sampled_from([(1, 2), (2, 4), (1, 4), (3, 4), (2, 8)]))
def test_nm_sub_dense(seed, nm):
    """EP.nm_sub_dense: an N:M matrix is a dense matrix (convert/nm_sparsity.prune_nm's output is N:M and
    multiplies as an ordinary dense matrix)."""
    from convert.nm_sparsity import max_nonzeros_per_group, prune_nm
    n, m = nm
    W = torch.randn(5, 4 * m, dtype=F64, generator=gen(seed))
    S = prune_nm(W, n, m)
    assert max_nonzeros_per_group(S, m) <= n
    x = torch.randn(4 * m, dtype=F64, generator=gen(seed + 1))
    assert torch.equal(S @ x, torch.stack([(S[r] * x).sum() for r in range(5)])) or \
        (S @ x - torch.stack([(S[r] * x).sum() for r in range(5)])).abs().max() <= fp(S.abs().sum(1).max().item() * x.abs().max().item(), 8 * m)


# =====================================================================================================
# Walls.lean (the positive theorems) and Constructions.lean
# =====================================================================================================

@MID
@given(st.integers(2, 40), st.integers(2, 10), st.integers(1, 8))
def test_winReach_iff(N, w, L):
    """EP.winReach_iff: L stacked windows of width w reach exactly j <= i with i - j <= L (w - 1)
    (block/mask_reach.reach on mask_reach.sliding)."""
    from block import mask_reach as mr
    R = mr.reach(mr.sliding(N, w), L)
    i, j = np.indices((N, N))
    assert np.array_equal(R, (j <= i) & (i - j <= L * (w - 1)))


@MID
@given(st.integers(2, 80), st.integers(2, 12))
def test_winLocal_stack(T, w):
    """EP.winLocal_stack: a context of T tokens needs L >= ceil((T - 1) / (w - 1)) windowed layers
    (block/mask_reach.diameter on the sliding mask == mask_reach.depth_to_global_formula)."""
    from block import mask_reach as mr
    assert mr.diameter(mr.sliding(T, w), T) == max(1, mr.depth_to_global_formula(T, w))


@FAST
@given(SEED, st.integers(0, 100), st.integers(0, 100), st.integers(-50, 50))
def test_rope_shift_invariant(seed, i, j, c):
    """EP.rope_shift_invariant: RoPE logits are invariant under a joint shift of the query and key positions
    (attn/rope_window.rope_apply)."""
    from attn.rope_window import rope_apply
    r = rng(seed)
    q, k = r.standard_normal(8), r.standard_normal(8)
    th = 10000.0 ** -(np.repeat(np.arange(4), 2) / 4)
    a = rope_apply(q, i + c, th) @ rope_apply(k, j + c, th)
    b = rope_apply(q, i, th) @ rope_apply(k, j, th)
    assert abs(a - b) <= fp(np.abs(q).sum() * np.abs(k).sum(), 50 * (1 + 150))
