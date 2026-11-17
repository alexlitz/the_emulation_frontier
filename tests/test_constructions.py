"""Validate the numerical emulation CONSTRUCTIONS (the exact identities).

Each test re-derives a construction with random tensors (no model download) and
asserts the emulation error is at fp noise (exact) or behaves as claimed
(approximate / limit). Run:  pytest tests/test_constructions.py
"""
import math

import torch
import torch.nn.functional as F

from attn.softmax_linear import favor_features, performer_attention, softmax_attention

torch.manual_seed(0)
F64 = torch.float64


def randn(*s):
    return torch.randn(*s, dtype=F64)


# --- positional: ALiBi = a fixed RoPE pair (seed result) ---------------------

def rot(phi):
    return torch.tensor([[math.cos(phi), -math.sin(phi)],
                         [math.sin(phi), math.cos(phi)]], dtype=F64)


def test_alibi_is_fixed_rope_pair_exact():
    # q_fixed=(-N,0), k_fixed=(0,1); dot(R_i q, R_j k) == -N sin(theta*(i-j))
    N, theta = 7.3, 0.01
    qf = torch.tensor([-N, 0.0], dtype=F64)
    kf = torch.tensor([0.0, 1.0], dtype=F64)
    for i in range(0, 40, 3):
        for j in range(0, i + 1, 2):
            lhs = (rot(i * theta) @ qf) @ (rot(j * theta) @ kf)
            rhs = -N * math.sin(theta * (i - j))
            assert abs(lhs.item() - rhs) < 1e-10


def test_alibi_small_angle_is_linear_bias():
    # -N sin(theta d) ~= -(N theta) d = -m d  for small theta d
    N, theta = 1e4, 1e-4
    m = N * theta                       # ALiBi slope
    for d in range(0, 50):
        approx = -N * math.sin(theta * d)
        assert abs(approx - (-m * d)) < 1e-3 * max(1.0, d)


# --- attention: MQA (wide shared KV) reproduces MHA exactly ------------------

def test_mqa_wide_emulates_mha_exact():
    T, H, hd = 16, 4, 12
    d = H * hd
    X, Wq, Wk, Wv, Wo = randn(T, d), randn(d, d), randn(d, d), randn(d, d), randn(d, d)
    scale = 1.0 / math.sqrt(hd)
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=F64), 1)

    q = (X @ Wq).view(T, H, hd); k = (X @ Wk).view(T, H, hd); v = (X @ Wv).view(T, H, hd)
    out = torch.empty(T, H, hd, dtype=F64)
    for h in range(H):
        a = F.softmax((q[:, h] @ k[:, h].T) * scale + mask, dim=-1)
        out[:, h] = a @ v[:, h]
    mha = out.reshape(T, d) @ Wo

    Kb, Vb = X @ Wk, X @ Wv                       # ONE shared wide KV head
    sel = []
    for h in range(H):
        qb = torch.zeros(T, d, dtype=F64)
        qb[:, h * hd:(h + 1) * hd] = X @ Wq[:, h * hd:(h + 1) * hd]
        a = F.softmax((qb @ Kb.T) * scale + mask, dim=-1)
        sel.append((a @ Vb)[:, h * hd:(h + 1) * hd])
    mqa = torch.cat(sel, -1) @ Wo
    assert (mha - mqa).abs().max().item() < 1e-10


# --- attention: MLA absorption (low-rank latent = MHA) ----------------------

def test_mla_absorption_identity_exact():
    T, dc, nope, vh = 20, 16, 8, 8
    c, W_UK, W_UV, qC, A = randn(T, dc), randn(dc, nope), randn(dc, vh), randn(T, nope), randn(T, T)
    # score: qC . (W_UK^T applied) -- explicit per-head key vs absorbed into query
    kC = c @ W_UK
    assert (qC @ kC.T - (qC @ W_UK.T) @ c.T).abs().max().item() < 1e-9
    # output: A (c W_UV) == (A c) W_UV
    assert (A @ (c @ W_UV) - (A @ c) @ W_UV).abs().max().item() < 1e-9


# --- normalization: LayerNorm = center + RMSNorm + bias ---------------------

def test_qknorm_folds_to_constant_scale_at_stable_magnitude():
    # QK-norm == plain attention with a fixed per-head scale WHEN ||q|| is constant
    from attn.qk_norm import attend, make_qk, rmsnorm
    q, k = make_qk(48, 64, cv=0.0, seed=1)                # cv=0 -> constant magnitudes
    a_qk = attend(rmsnorm(q) @ rmsnorm(k).T)
    cq, ck = q.pow(2).mean(-1).sqrt().mean(), k.pow(2).mean(-1).sqrt().mean()
    a_fold = attend((q / cq) @ (k / ck).T)               # plain attn, folded scale
    assert (a_qk - a_fold).abs().max().item() < 1e-4     # ~exact (eps-limited) vs 2e-2 at CV=0.8


def test_softcap_folds_to_identity_when_unsaturated():
    # softcap(x)=c*tanh(x/c) ~= x for |x| << c, so a plain model absorbs it
    from attn.qk_norm import attend
    from attn.softcap import softcap
    g = torch.Generator().manual_seed(3)
    s = torch.randn(48, 48, dtype=F64, generator=g)
    small = s / s.abs().max() * 0.1                              # max|logit| = 0.1 << cap=1
    assert (attend(small) - attend(softcap(small, 1.0))).abs().max().item() < 1e-4
    big = s / s.abs().max() * 6.0                                # max|logit| = 6 -> saturating
    assert (attend(big) - attend(softcap(big, 1.0))).abs().max().item() > 1e-2


def test_attention_sink_is_standard_plus_null_token():
    # gpt-oss's learned softmax sink == standard attention + a zero-value null key
    from attn.attention_sink import emulate_with_null_token, sink_attention
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(32, generator=g, dtype=F64)
    V = torch.randn(32, 16, generator=g, dtype=F64)
    for b in (-4.0, 0.0, 2.0, 5.0):
        assert (sink_attention(scores, V, b) - emulate_with_null_token(scores, V, b)).abs().max().item() < 1e-12
    # b -> -inf recovers plain softmax (free reverse)
    assert (sink_attention(scores, V, -50.0) - F.softmax(scores, 0) @ V).abs().max().item() < 1e-12


def test_sparse_patterns_recover_full_at_diameter():
    # every sparse pattern <= full (mask); full recovered at DEPTH = graph diameter
    from attn.sparse_patterns import causal, diameter, local, strided
    N = 64
    assert diameter(causal(N), N) == 1                        # full is full
    assert diameter(local(N, 8) | strided(N, 8), N) == 2      # factorized -> 2 hops
    assert diameter(strided(N, 8), N) is None                 # strided alone never reaches full
    assert diameter(local(N, 8), N) > 4                       # naive local -> O(N/W)


def test_local_and_ssm_are_dual_restrictions():
    # local = bounded REACH (fails beyond window); SSM/linear = bounded RESOLUTION
    # (degrades with load); full softmax = both. All <= full.
    from attn.local_vs_ssm import full, linear_ssm, local, make, relerr
    N, d, W = 192, 128, 32
    Kin, Vin, q, tgt = make(N, d, N - 1 - 8, 0)              # target inside window
    assert relerr(local(Kin, Vin, q, W, N - 1), tgt) < 1e-6
    Kout, Vout, q2, tgt2 = make(N, d, N - 1 - 80, 0)         # target beyond window
    assert relerr(local(Kout, Vout, q2, W, N - 1), tgt2) > 0.5   # local: bounded reach
    assert relerr(full(Kout, Vout, q2), tgt2) < 1e-6            # full: exact anywhere
    # SSM/linear degrades with load; full stays exact
    Ks, Vs, qs, ts = make(32, d, 27, 1)
    Kb, Vb, qb, tb = make(512, d, 507, 1)
    assert relerr(linear_ssm(Kb, Vb, qb), tb) > relerr(linear_ssm(Ks, Vs, qs), ts) > 0.1
    assert relerr(full(Kb, Vb, qb), tb) < 1e-6


def test_dense_embeds_in_wider_2of4_exactly():
    # N:M sparsity, the "dense fits in bigger sparse" direction (2:4 at 2x width)
    from convert.nm_sparsity import dense_to_wide_nm, max_nonzeros_per_group
    W, x = randn(16, 32), randn(8, 32)
    S, expand = dense_to_wide_nm(W, 2, 4)
    assert (expand(x) @ S.T - x @ W.T).abs().max().item() < 1e-10   # exact matmul
    assert max_nonzeros_per_group(S) <= 2                           # valid 2:4 pattern
    assert int((S != 0).sum().item()) == W.numel()                 # conserved: nonzeros == dense params


def test_sequential_block_is_two_parallel_blocks():
    # block structure: one sequential block == two parallel blocks (attn-only, mlp-only)
    T, d, m = 8, 64, 128
    x, n1, n2, Wa, Wm1, Wm2 = randn(T, d), randn(d), randn(d), randn(d, d), randn(d, m), randn(m, d)
    rms = lambda z, w: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    A = lambda z: rms(z, n1) @ Wa
    M = lambda z: F.silu(rms(z, n2) @ Wm1) @ Wm2
    seq = x + A(x) + M(x + A(x))                          # sequential block
    b1 = x + A(x)                                         # parallel block: attn only
    b2 = b1 + M(b1)                                       # parallel block: mlp only
    assert (seq - b2).abs().max().item() < 1e-10          # sequential <= parallel at 2x depth


def test_layernorm_equals_center_rmsnorm_bias():
    T, d, eps = 32, 48, 1e-5
    x, w, b = randn(T, d), randn(d), randn(d)
    ref = F.layer_norm(x, (d,), w, b, eps)
    c = x - x.mean(-1, keepdim=True)
    exact = c * torch.rsqrt((c ** 2).mean(-1, keepdim=True) + eps) * w + b
    assert (ref - exact).abs().max().item() < 1e-10


def test_dropping_centering_changes_output():
    T, d, eps = 32, 48, 1e-5
    x, w, b = randn(T, d) + 3.0, randn(d), randn(d)      # nonzero mean
    ref = F.layer_norm(x, (d,), w, b, eps)
    drop = x * torch.rsqrt((x ** 2).mean(-1, keepdim=True) + eps) * w + b
    assert (ref - drop).abs().max().item() > 1e-2         # centering matters


# --- activations -------------------------------------------------------------

def silu_beta(x, b):
    return x * torch.sigmoid(b * x)


def test_relu_is_beta_limit_of_swish():
    x = torch.linspace(-6, 6, 4001, dtype=F64)
    relu = F.relu(x)
    errs = [(silu_beta(x, b) - relu).abs().max().item() for b in [1, 4, 16, 64, 256]]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))   # monotone down
    assert errs[-1] < 2e-3


def test_tanh_is_affine_sigmoid():
    x = torch.linspace(-8, 8, 4001, dtype=F64)
    assert (torch.tanh(x) - (2 * torch.sigmoid(2 * x) - 1)).abs().max().item() < 1e-12


def test_abs_from_two_relus():
    x = torch.linspace(-8, 8, 4001, dtype=F64)
    assert (x.abs() - (F.relu(x) + F.relu(-x))).abs().max().item() < 1e-12


def test_bilinear_glu_equals_squared_ffn_2x_width():
    T, d, m = 16, 32, 24
    x, Wg, Wu, Wd = randn(T, d), randn(d, m), randn(d, m), randn(m, d)
    a, b = x @ Wg, x @ Wu
    bilinear = (a * b) @ Wd
    h = torch.cat([a + b, a - b], -1)
    square_ffn = (0.25 * h[:, :m] ** 2 - 0.25 * h[:, m:] ** 2) @ Wd
    assert (bilinear - square_ffn).abs().max().item() < 1e-9


def test_swiglu_equals_two_layer_squared_ffn():
    T, d, m = 16, 32, 24
    x, Wg, Wu, Wd = randn(T, d), randn(d, m), randn(d, m), randn(m, d)
    a, b = x @ Wg, x @ Wu
    swiglu = (F.silu(a) * b) @ Wd
    p = F.silu(a)                                         # layer 1 (nonlinear)
    h = torch.cat([p + b, p - b], -1)                    # layer 2: product via square
    two_layer = (0.25 * h[:, :m] ** 2 - 0.25 * h[:, m:] ** 2) @ Wd
    assert (swiglu - two_layer).abs().max().item() < 1e-9


def test_gelu_silu_has_a_nonzero_floor():
    # GeLU<=SiLU is the one approx edge with a FLOOR: min_beta max|GeLU-Swish_b| is
    # ~0.0139 and does NOT decrease with grid resolution (no currency reaches exact).
    xf = torch.linspace(-12, 12, 60001, dtype=F64)
    g = xf * 0.5 * (1 + torch.erf(xf / math.sqrt(2)))    # precompute once
    err = lambda b: (g - silu_beta(xf, b)).abs().max().item()
    coarse = min(err(b) for b in torch.linspace(1.6, 1.95, 60, dtype=F64))
    finer = min(err(b) for b in torch.linspace(1.74, 1.81, 120, dtype=F64))
    assert 0.0138 < finer < 0.0141                       # the floor, ~1.4% at the bump
    assert abs(finer - coarse) < 1e-3                    # it is a FLOOR, not shrinking to 0


# --- softmax vs linear attention --------------------------------------------

def test_performer_error_decreases_with_features():
    T, d = 64, 32
    Q = F.normalize(randn(T, d), dim=-1)
    K = F.normalize(randn(T, d), dim=-1)
    V = randn(T, d)
    exact = softmax_attention(Q, K, V, 1.0)

    def err(m, reps=6):
        return sum(((performer_attention(Q, K, V, randn(m, d), 1.0) - exact).norm()
                    / exact.norm()).item() for _ in range(reps)) / reps
    assert err(64) > err(1024) > err(8192)               # ~1/sqrt(m)


# --- output-path rotation: retentive/linear attention carries rotation in the state ---

def test_retention_parallel_equals_recurrent():
    # the recurrent state S_n = sum gamma^(n-m) (R_m k_m) v_m^T holds ROTATED keys, so
    # the position rotation lives on the output path -- and the two forms agree exactly.
    from attn.output_rotation import parallel_retention, recurrent_retention
    g = torch.Generator().manual_seed(1)
    T, d, dv = 18, 8, 8
    Q, K, V = (torch.randn(T, d, dtype=F64, generator=g) for _ in range(3))
    par = parallel_retention(Q, K, V, theta=0.3, gamma=0.95)
    rec = recurrent_retention(Q, K, V, theta=0.3, gamma=0.95)
    assert (par - rec).abs().max().item() < 1e-12


def test_output_rotation_is_relative():
    # like RoPE, the output rotation is RELATIVE: starting the rotations at any offset
    # leaves the retention output unchanged (q_n^T R_(m-n) k_m depends only on n-m).
    from attn.output_rotation import parallel_retention
    g = torch.Generator().manual_seed(2)
    T, d = 16, 8
    Q, K, V = (torch.randn(T, d, dtype=F64, generator=g) for _ in range(3))
    base = parallel_retention(Q, K, V, theta=0.4, gamma=0.9)
    shifted = parallel_retention(Q, K, V, theta=0.4, gamma=0.9, offset=5)
    assert (base - shifted).abs().max().item() < 1e-12


def test_retention_collapses_to_plain_linear_attention():
    # theta=0 (no rotation), gamma=1 (no decay) -> plain causal linear attention;
    # the rotation+decay are exactly the added OUTPUT-path structure.
    from attn.output_rotation import parallel_retention
    g = torch.Generator().manual_seed(3)
    T, d = 14, 8
    Q, K, V = (torch.randn(T, d, dtype=F64, generator=g) for _ in range(3))
    plain = parallel_retention(Q, K, V, theta=0.0, gamma=1.0)
    lin = torch.tril(Q @ K.T) @ V
    assert (plain - lin).abs().max().item() < 1e-12


# --- MoE <-> dense, both directions, with honest currencies ------------------

def test_moe_reproduces_wide_dense_via_k_experts():
    # MoE emulates a width-(k*ew) dense FFN: k experts = the k width-ew blocks, a
    # constant uniform (1/k) router, W2 scaled by k. Exact -> the active-width bound.
    g = torch.Generator().manual_seed(4)
    T, d, ew, k = 16, 8, 6, 3
    m = k * ew
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    x, W1, W2 = r(T, d), r(d, m), r(m, d)
    dense = F.silu(x @ W1) @ W2
    out = torch.zeros(T, d, dtype=F64)
    for i in range(k):
        W1i, W2i = W1[:, i * ew:(i + 1) * ew], W2[i * ew:(i + 1) * ew, :]
        out += (1.0 / k) * (F.silu(x @ W1i) @ (k * W2i))
    assert (dense - out).abs().max().item() < 1e-10


def test_dense_reproduces_top1_moe_two_ways():
    # the REVERSE, two realizations of the SAME reduction, both bit-exact:
    #  - width + select: hold all E experts (E x width) + a hard argmax select layer;
    #  - 2x layers + masking: a mask layer + E masked FFN sublayers (E x depth).
    from moe.dense_emulates_moe import build
    exact, masked, _ = build()
    assert exact < 1e-12          # width + select
    assert masked < 1e-12         # depth + masking (the graph's 2x-depth currency)


def test_dense_router_reaches_moe_only_in_the_limit():
    # the routing argmax is discontinuous: a SMOOTH dense router reaches the MoE only
    # as sharpness beta->inf (a limit, like ReLU<=SiLU) -- NOT free/exact.
    from moe.dense_emulates_moe import build
    *_, errs = build()
    e = [v for _, v in errs]
    assert e[0] > e[-1]            # sharper router -> closer
    assert e[-1] < 1e-9           # exact only in the limit


def test_mtp_heads_do_not_change_base_forward():
    # multi-token prediction: auxiliary predict heads share the trunk; dropping them
    # leaves the MAIN next-token logits bit-identical -> MTP folds away for free.
    g = torch.Generator().manual_seed(5)
    d, V, T = 16, 32, 8
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    Wt, Wmain, Waux = r(d, d), r(d, V), [r(d, V) for _ in range(3)]
    x = r(T, d)
    trunk = F.silu(x @ Wt)
    main = trunk @ Wmain
    _ = [trunk @ Wa for Wa in Waux]          # aux heads exist but never feed the main head
    assert (main - (F.silu(x @ Wt) @ Wmain)).abs().max().item() < 1e-12


def test_cross_layer_kv_reproduced_by_carrying_kv():
    # CLA subset standard, EXACT, currency = WIDTH: a standard layer carrying the
    # shared K,V in reserved channels reproduces a cross-layer-KV layer.
    from attn.cross_layer_kv import build
    assert build() < 1e-10


def test_softmax_hard_selects_linear_floors():
    T, d = 64, 64
    q = torch.zeros(d, dtype=F64); q[0] = 1.0
    K = F.normalize(randn(T, d), dim=-1); K[0] = q
    scores = K @ q
    assert F.softmax(64 * scores, 0)[0].item() > 0.99    # softmax -> one-hot
    phi = lambda z: F.elu(z) + 1
    w = phi(K) @ phi(q); w = w / w.sum()
    assert w[0].item() < 0.2                              # linear cannot concentrate
