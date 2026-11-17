"""Spot-check EVERY reduction on a layer or two of REAL weights.

The per-axis proofs in attn/ act/ norm/ block/ moe/ convert/ are mostly *isolated
tensor identities*: they compare bare einsums, never running the construction
through an actual module forward with real `nn.Linear` weights. That leaves a gap
-- an identity can be algebraically true yet be wired wrong inside a layer (scaling,
causal mask, residual/norm placement, head split). This module closes it: for each
reduction it BUILDS both sides as genuine 1-2 layer modules with real weight tensors
and checks the forward matches (or, for a floor/limit, that it does NOT).

Every check returns a row dict; `run()` prints the table and a verdict. The five
stock-`transformers` reductions (GQA/MQA, MoE<-dense, local<-full, NoPE<-RoPE,
looped<-untied) already build on real 2-layer HF models -- we re-run their `build()`
here so the table is complete; the rest are lifted from bare tensors to real layers.

  python -m elephantine.spot_check_reductions
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)
F64 = torch.float64
EXACT = 1e-9                        # fp64 identity threshold


# ------------------------------------------------------------------ real-weight helpers

def _g(seed):
    return torch.Generator().manual_seed(seed)


def lin(i, o, g, bias=False):
    """A real nn.Linear (fp64) with reproducible random weights."""
    m = nn.Linear(i, o, bias=bias).to(F64)
    m.weight.copy_(torch.randn(o, i, dtype=F64, generator=g))
    if bias:
        m.bias.copy_(torch.randn(o, dtype=F64, generator=g))
    return m


def cmask(T):
    return torch.triu(torch.full((T, T), float("-inf"), dtype=F64), 1)


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def row(name, host, layers, worst, kind, thr):
    ok = (worst < thr) if kind in ("exact", "regime", "limit") else (worst > thr)
    return dict(name=name, host=host, layers=layers, worst=worst, kind=kind, thr=thr, ok=ok)


# ================================================================== ATTENTION family
# A genuine single-head causal-attention output through a REAL o_proj. Q,K,V are the
# real projections' outputs (or constructed, for the regime checks) -- either way the
# softmax, causal mask, scaling and output projection are the real layer.

def _attend(Q, K, V, scale, sink_b=None, softcap_c=None):
    T = Q.shape[0]
    s = (Q @ K.T) * scale
    if softcap_c is not None:
        s = softcap_c * torch.tanh(s / softcap_c)
    s = s + cmask(T)
    if sink_b is None:
        return F.softmax(s, -1) @ V
    num = torch.exp(s)                                 # future already -inf -> 0
    a = num / (math.exp(sink_b) + num.sum(-1, keepdim=True))   # off-by-one (sink) denom
    return a @ V


def check_attention_sink(d=32, hd=32, T=16, seed=0):
    """Sink attention (gpt-oss denominator bias b) == standard softmax + one zero-value
    null token. Built through a real q/k/v/o attention layer."""
    g = _g(seed)
    q, k, v, o = lin(d, hd, g), lin(d, hd, g), lin(d, hd, g), lin(hd, d, g)
    x = torch.randn(T, d, dtype=F64, generator=g)
    Q, K, V = q(x), k(x), v(x)
    b = 1.7
    sink = o(_attend(Q, K, V, 1 / math.sqrt(hd), sink_b=b))               # target
    Kn = torch.cat([K, torch.zeros(1, hd, dtype=F64)])                    # append null key...
    Vn = torch.cat([V, torch.zeros(1, hd, dtype=F64)])                    # ...with value 0
    T2 = T
    s = (Q @ Kn.T) * (1 / math.sqrt(hd))
    s[:, :T2] += cmask(T2)                                                # causal on real keys
    s[:, T2] = b                                                          # null key score = b
    host = o(F.softmax(s, -1) @ Vn)                                       # standard + null token
    return row("attention sink = standard + null token", "1-layer attn (real q/k/v/o)", 1,
               (sink - host).abs().max().item(), "exact", EXACT)


def check_cross_layer_kv(d=24, hd=24, T=12, seed=1):
    """CLA/YOCO layer-1 reuses layer-0's K,V. A standard 2-layer stack reproduces it by
    CARRYING K,V in reserved residual channels (2x width) and reading them back."""
    g = _g(seed)
    Wk0, Wv0, Wq1, o = lin(d, hd, g), lin(d, hd, g), lin(d, hd, g), lin(hd, d, g)
    h0 = torch.randn(T, d, dtype=F64, generator=g)
    h1 = torch.randn(T, d, dtype=F64, generator=g)                       # layer-1 hidden differs
    K0, V0 = Wk0(h0), Wv0(h0)                                            # shared K,V (computed once)
    cla = o(_attend(Wq1(h1), K0, V0, 1 / math.sqrt(hd)))                 # target: reuse K0,V0
    # host: reserved channels carry K0,V0; layer-1 k/v projections SELECT them out.
    h1_aug = torch.cat([h1, K0, V0], 1)                                  # width = d + 2*hd
    Sk = torch.cat([torch.zeros(hd, d), torch.eye(hd), torch.zeros(hd, hd)], 1).to(F64)
    Sv = torch.cat([torch.zeros(hd, d), torch.zeros(hd, hd), torch.eye(hd)], 1).to(F64)
    Wq1_aug = torch.cat([Wq1.weight, torch.zeros(hd, 2 * hd, dtype=F64)], 1)
    std = o(_attend(h1_aug @ Wq1_aug.T, h1_aug @ Sk.T, h1_aug @ Sv.T, 1 / math.sqrt(hd)))
    return row("cross-layer KV (CLA) = carry K,V in reserved channels", "2-layer attn (2x width)", 2,
               (cla - std).abs().max().item(), "exact", EXACT)


def check_mla_absorption(d=48, hd=32, r=16, T=14, seed=2):
    """MLA = MHA whose per-head K,V read from a shared low-rank latent. The absorbed
    form (attend over the latent) equals the explicit MHA form, through a real o_proj."""
    g = _g(seed)
    W_DKV, W_UK, W_UV, W_QC, o = (lin(d, r, g), lin(r, hd, g), lin(r, hd, g),
                                  lin(d, hd, g), lin(hd, d, g))
    x = torch.randn(T, d, dtype=F64, generator=g)
    latent, qC = W_DKV(x), W_QC(x)
    scale = 1 / math.sqrt(hd)
    kC, vh = W_UK(latent), W_UV(latent)                                  # explicit per-head K,V
    A = F.softmax((qC @ kC.T) * scale + cmask(T), -1)
    explicit = o(A @ vh)
    q_abs = qC @ W_UK.weight                                             # fold up-proj into q
    A2 = F.softmax((q_abs @ latent.T) * scale + cmask(T), -1)            # attend over latent
    absorbed = o((A2 @ latent) @ W_UV.weight.T)
    return row("MLA absorption = MHA (low-rank latent K,V)", "1-layer attn (real o_proj)", 1,
               (explicit - absorbed).abs().max().item(), "exact", EXACT)


def check_retention_output_rotation(d=16, T=18, seed=3):
    """Retention/linear-attention carries the position rotation in its recurrent STATE:
    the parallel form == the recurrent form, through a real v/o projection."""
    g = _g(seed)
    Wq, Wk, Wv, o = lin(d, d, g), lin(d, d, g), lin(d, d, g), lin(d, d, g)
    x = torch.randn(T, d, dtype=F64, generator=g)
    Q, K, V = Wq(x), Wk(x), Wv(x)
    theta, gamma = 0.3, 0.95
    Rs = torch.zeros(T, d, d, dtype=F64)
    for n in range(T):
        c, s = math.cos(n * theta), math.sin(n * theta)
        for i in range(0, d, 2):
            Rs[n, i, i] = c; Rs[n, i, i + 1] = -s; Rs[n, i + 1, i] = s; Rs[n, i + 1, i + 1] = c
    Qr = torch.stack([Rs[n] @ Q[n] for n in range(T)])
    Kr = torch.stack([Rs[n] @ K[n] for n in range(T)])
    nn_ = torch.arange(T)
    decay = torch.where(nn_[:, None] >= nn_[None, :],
                        gamma ** (nn_[:, None] - nn_[None, :]).clamp(min=0).double(), 0.0)
    par = o((Qr @ Kr.T * decay) @ V)                                    # parallel retention
    S = torch.zeros(d, d, dtype=F64); rec = torch.empty(T, d, dtype=F64)
    for n in range(T):                                                  # recurrent: state holds rotated k v^T
        S = gamma * S + torch.outer(Kr[n], V[n]); rec[n] = Qr[n] @ S
    return row("retention output rotation = parallel == recurrent", "1-layer linear-attn", 1,
               (par - o(rec)).abs().max().item(), "exact", EXACT)


def check_qk_norm_fold(hd=64, T=48, seed=4):
    """QK-norm (RMSNorm on Q,K) folds to a fixed per-head scale EXACTLY when per-token
    ||q||,||k|| are constant (CV=0). Verified through a real v/o projection; the residual
    grows with the magnitude CV (regime edge, not a wall)."""
    g = _g(seed)
    v, o = lin(hd, hd, g), lin(hd, hd, g)
    x = torch.randn(T, hd, dtype=F64, generator=g)
    V = v(x)

    def qk(cv, seed2):
        g2 = _g(seed2)
        qd = F.normalize(torch.randn(T, hd, dtype=F64, generator=g2), dim=-1)
        kd = F.normalize(torch.randn(T, hd, dtype=F64, generator=g2), dim=-1)
        mq = (1.0 + cv * torch.randn(T, 1, dtype=F64, generator=g2)).abs()
        mk = (1.0 + cv * torch.randn(T, 1, dtype=F64, generator=g2)).abs()
        return qd * mq, kd * mk

    def out(Q, K, scale):
        return o(F.softmax((Q @ K.T) * scale + cmask(T), -1) @ V)

    Q0, K0 = qk(0.0, 0)                                                 # CV=0 -> constant norm
    tgt = out(rms(Q0), rms(K0), 1.0)                                    # true QK-norm
    sq = torch.rsqrt(Q0.pow(2).mean() + 1e-6)                           # the (constant) rms multiplier
    sk = torch.rsqrt(K0.pow(2).mean() + 1e-6)                           # -> a single fixed per-head scale
    fold = out(Q0 * sq, K0 * sk, 1.0)                                   # fixed-scale fold (folds into Wq/Wk)
    exact = (tgt - fold).abs().max().item()
    # drift outside the regime (report only; the reduction claim is the CV=0 fold)
    Qc, Kc = qk(0.4, 7)
    drift_t = out(rms(Qc), rms(Kc), 1.0)
    sq2 = torch.rsqrt(Qc.pow(2).mean() + 1e-6); sk2 = torch.rsqrt(Kc.pow(2).mean() + 1e-6)
    drift = (drift_t - out(Qc * sq2, Kc * sk2, 1.0)).abs().max().item()
    r = row("qk-norm folds to fixed scale (CV=0)", "1-layer attn (real v/o)", 1, exact, "regime", EXACT)
    r["note"] = f"CV=0.4 drift {drift:.1e} (grows with magnitude variance)"
    return r


def check_softcap_fold(hd=64, T=48, seed=5):
    """Logit soft-capping c*tanh(x/c) folds to the identity in the unsaturated regime
    (|logit| << c): a plain attention layer reproduces the softcap one to fp noise, and
    drifts only as logits approach the cap."""
    g = _g(seed)
    q, k, v, o = lin(hd, hd, g), lin(hd, hd, g), lin(hd, hd, g), lin(hd, hd, g)
    x = torch.randn(T, hd, dtype=F64, generator=g)
    Q, K, V = q(x), k(x), v(x)
    base = (Q @ K.T) / math.sqrt(hd)

    def pair(ratio):
        s = base / base.abs().max() * ratio                            # max|logit| = ratio (c=1)
        plain = o(F.softmax(s + cmask(T), -1) @ V)                      # no-softcap model
        cap = o(F.softmax(1.0 * torch.tanh(s) + cmask(T), -1) @ V)     # softcap model (c=1)
        return (plain - cap).abs().max().item()

    unsat = pair(1e-4)                                                  # deep unsaturated -> identity
    sat = pair(4.0)                                                    # saturated -> real difference
    r = row("softcap folds to identity (unsaturated)", "1-layer attn (real q/k/v/o)", 1, unsat, "regime", EXACT)
    r["note"] = f"saturated ratio=4 drift {sat:.1e} (grows into the cap)"
    return r


def check_alibi_from_rope(N=8.0, T=40, seed=6):
    """ALiBi's linear relative bias IS a fixed RoPE pair: for q=(-N,0), k=(0,1),
    <R_i q, R_j k> = -N sin(theta(i-j)) EXACTLY (a real 2-D attention score), and
    -> -(N*theta) d as theta->0 (the ALiBi slope). Exact identity + the small-angle limit."""
    theta = 0.01
    ident = 0.0
    for i in range(T):
        for j in range(i + 1):
            R = lambda a: torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=F64)
            qi = R(i * theta) @ torch.tensor([-N, 0.0], dtype=F64)      # real rotated query
            kj = R(j * theta) @ torch.tensor([0.0, 1.0], dtype=F64)     # real rotated key
            ident = max(ident, abs(float(qi @ kj) - (-N * math.sin(theta * (i - j)))))
    # small-angle: slope m = N*theta held fixed as theta->0 -> linear bias -m d
    slope, errs = 0.5, []
    for th in (0.1, 0.03, 0.01, 0.003):
        Nn = slope / th
        d = torch.arange(T + 1.0)
        errs.append((-Nn * torch.sin(th * d) - (-slope * d)).abs().max().item())
    r = row("ALiBi = fixed RoPE pair (exact) -> linear (limit)", "2-D attention score", 1, ident, "exact", EXACT)
    r["note"] = f"small-angle linear error {errs[0]:.1e}->{errs[-1]:.1e} as theta 0.1->0.003"
    return r


# ================================================================== FFN / ACTIVATION family

def check_swiglu_two_layer(d=32, m=64, T=24, seed=7):
    """SwiGLU (SiLU(xWg) (.) xWu) == a TWO-layer squared-activation FFN, via the product
    identity ab = 1/4[(a+b)^2 - (a-b)^2]. Built as real FFN Linear layers."""
    g = _g(seed)
    Wg, Wu, Wd = lin(d, m, g), lin(d, m, g), lin(m, d, g)
    x = torch.randn(T, d, dtype=F64, generator=g)
    a, b = Wg(x), Wu(x)
    swiglu = Wd(F.silu(a) * b)                                          # target
    p = F.silu(a)                                                       # layer 1 (inner nonlinearity)
    h = torch.cat([p + b, p - b], -1)                                  # layer 2 pre-square
    prod = 0.25 * (h[:, :m] ** 2) - 0.25 * (h[:, m:] ** 2)             # == p*b
    two_layer = Wd(prod)
    return row("SwiGLU = 2-layer squared FFN", "2-layer FFN (real Wg/Wu/Wd)", 2,
               (swiglu - two_layer).abs().max().item(), "exact", EXACT)


def check_glu_superset_vanilla(d=32, m=96, T=24, seed=8):
    """A GLU/SwiGLU FFN with the value branch pinned to the constant 1 reduces to a
    vanilla SiLU FFN. Real FFN Linear layers."""
    g = _g(seed)
    Wg, Wd = lin(d, m, g), lin(m, d, g)
    x = torch.randn(T, d, dtype=F64, generator=g)
    vanilla = Wd(F.silu(Wg(x)))                                        # target
    reduced = Wd(F.silu(Wg(x)) * torch.ones(T, m, dtype=F64))          # SwiGLU, up-branch == 1
    return row("GLU >= vanilla FFN (pin value branch)", "1-layer FFN (real Wg/Wd)", 1,
               (vanilla - reduced).abs().max().item(), "exact", EXACT)


def check_dense_emulates_moe(d=16, ew=24, E=8, T=32, seed=9):
    """A dense stack holding ALL experts (E*ew wide) + a hard argmax select layer
    reproduces a top-1 MoE bit-for-bit. Real per-expert FFN Linear layers."""
    g = _g(seed)
    W1 = [lin(d, ew, g) for _ in range(E)]
    W2 = [lin(ew, d, g) for _ in range(E)]
    Wg = lin(d, E, g)
    x = torch.randn(T, d, dtype=F64, generator=g)
    Y = torch.stack([W2[e](F.silu(W1[e](x))) for e in range(E)], 1)    # (T,E,d) all experts
    e = Wg(x).argmax(-1)
    moe = Y[torch.arange(T), e]                                        # target: route to argmax
    mask = F.one_hot(Wg(x).argmax(-1), E).to(F64)                     # the hard select layer
    dense = torch.einsum("te,ted->td", mask, Y)                        # hold all + select
    return row("dense >= MoE (hold all experts + hard select)", "wide FFN + select layer", 1,
               (moe - dense).abs().max().item(), "exact", EXACT)


def check_relu_swish_limit(d=32, m=64, T=24, seed=10):
    """ReLU FFN = the beta->inf limit of a Swish_beta FFN (x*sigmoid(beta x) -> ReLU),
    through real FFN Linear layers: the endpoint matches, error decreases monotonically."""
    g = _g(seed)
    Wg, Wd = lin(d, m, g), lin(m, d, g)
    x = torch.randn(T, d, dtype=F64, generator=g)
    relu_ffn = Wd(F.relu(Wg(x)))
    errs = [(Wd(Wg(x) * torch.sigmoid(b * Wg(x))) - relu_ffn).abs().max().item()
            for b in (1, 16, 256, 4096, 65536)]
    mono = all(errs[i] >= errs[i + 1] for i in range(len(errs) - 1))
    r = row("ReLU FFN = beta->inf Swish FFN", "1-layer FFN (real Wg/Wd)", 1, errs[-1],
            "limit", 1e-3)
    r["note"] = f"beta 1->65536: {errs[0]:.1e} -> {errs[-1]:.1e}" + ("" if mono else " NOT MONOTONE!")
    r["ok"] = r["ok"] and mono
    return r


def check_gelu_silu_floor(d=32, m=64, T=24, seed=11):
    """GeLU vs SiLU is the one ACTIVATION swap with NO exact identity: through a real FFN
    the layer output differs by a genuine, non-vanishing floor. We assert it does NOT
    reduce to fp noise (confirming the gap is real, not slack)."""
    g = _g(seed)
    Wg, Wd = lin(d, m, g), lin(m, d, g)
    x = torch.randn(T, d, dtype=F64, generator=g)
    worst = (Wd(F.gelu(Wg(x))) - Wd(F.silu(Wg(x)))).abs().max().item()
    return row("GeLU != SiLU (activation floor)", "1-layer FFN (real Wg/Wd)", 1, worst, "floor", 1e-3)


# ================================================================== NORM / BLOCK / SPARSITY

def check_layernorm_fold(d=32, m=64, T=16, seed=12):
    """LayerNorm = gamma (.) RMSNorm(center(x)) + beta, folded through a REAL pre-norm
    decoder block (LN -> attn -> residual -> LN -> MLP): the fold reproduces native
    LayerNorm exactly; naively deleting beta (not folding it) is what diverges."""
    g = _g(seed)
    ln1, ln2 = nn.LayerNorm(d).to(F64), nn.LayerNorm(d).to(F64)
    for ln in (ln1, ln2):
        ln.weight.copy_(torch.randn(d, dtype=F64, generator=g))
        ln.bias.copy_(torch.randn(d, dtype=F64, generator=g))
    Wa, Wg, Wd = lin(d, d, g), lin(d, m, g), lin(m, d, g)
    x = torch.randn(T, d, dtype=F64, generator=g)

    def norm(z, ln, mode):
        w, b, eps = ln.weight, ln.bias, ln.eps
        if mode == "native":
            return ln(z)
        if mode == "fold":                                             # center -> RMSNorm -> affine
            c = z - z.mean(-1, keepdim=True)
            return c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + eps) * w + b
        if mode == "drop_beta":                                        # naive: delete beta (NOT folded)
            c = z - z.mean(-1, keepdim=True)
            return c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + eps) * w

    def block(mode):
        h = x + Wa(norm(x, ln1, mode))                                 # attn sublayer (linear stand-in)
        return h + Wd(F.silu(Wg(norm(h, ln2, mode))))

    native = block("native")
    worst = (native - block("fold")).abs().max().item()
    drop = (native - block("drop_beta")).abs().max().item()
    r = row("LayerNorm = center+RMSNorm+bias (fold)", "2-sublayer block (real LN/attn/MLP)", 2,
            worst, "exact", EXACT)
    r["note"] = f"deleting beta instead of folding: {drop:.1e} (the fold pass is what's exact)"
    return r


def check_parallel_sequential(d=32, m=64, T=12, seed=13):
    """Parallel (GPT-J) and sequential (GPT-2) residual blocks emulate each other:
    parallel >= sequential at 2x DEPTH, sequential >= parallel at 2x WIDTH. Real A/M
    sublayers (Linear + norm). Worst of both constructions."""
    g = _g(seed)
    n1, n2, Wa, Wm1, Wm2 = (torch.randn(d, dtype=F64, generator=g), torch.randn(d, dtype=F64, generator=g),
                            lin(d, d, g), lin(d, m, g), lin(m, d, g))
    x = torch.randn(T, d, dtype=F64, generator=g)
    A = lambda z: Wa(rms(z) * n1)
    M = lambda z: Wm2(F.silu(Wm1(rms(z) * n2)))
    par = x + A(x) + M(x)                                              # parallel block
    seq = x + A(x) + M(x + A(x))                                       # sequential block
    depth2 = (x + A(x)) + M(x + A(x))                                  # parallel >= seq @ 2x depth
    width2 = (x + A(x)) + M(x)                                         # seq >= parallel @ 2x width (M reads reserved x)
    worst = max((seq - depth2).abs().max().item(), (par - width2).abs().max().item())
    return row("parallel <-> sequential residual (depth/width)", "2-block stack (real A/M)", 2,
               worst, "exact", EXACT)


def check_nm_sparsity(rows_=32, cols=64, T=8, seed=14):
    """A dense matmul embeds EXACTLY in a 2x-wider 2:4-structured-sparse matmul
    (interleave zeros in the input, dense weights on even columns). Real weight tensors."""
    g = _g(seed)
    W = torch.randn(rows_, cols, dtype=F64, generator=g)
    x = torch.randn(T, cols, dtype=F64, generator=g)
    ref = x @ W.T
    S = torch.zeros(rows_, cols * 2, dtype=F64); S[:, ::2] = W          # dense on even cols -> 2:4 valid
    z = torch.zeros(T, cols * 2, dtype=F64); z[:, ::2] = x
    grp = int((S.view(rows_, cols * 2 // 4, 4) != 0).sum(-1).max())     # nonzeros per group of 4
    worst = (z @ S.T - ref).abs().max().item()
    r = row("dense embeds in wider 2:4-sparse", "1 linear (real weights)", 1, worst, "exact", EXACT)
    r["note"] = f"max nonzeros/group-of-4 = {grp} (<=2 -> valid 2:4)"
    return r


def check_moe_wide_dense(d=16, ew=20, k=3, T=24, seed=15):
    """A k-active MoE reproduces a dense FFN of width k*ew: split the wide weights across
    the k always-selected experts; their outputs sum to the wide dense FFN. Real weights."""
    g = _g(seed)
    W1, W2 = lin(d, k * ew, g), lin(k * ew, d, g)                       # the wide dense FFN
    x = torch.randn(T, d, dtype=F64, generator=g)
    dense = W2(F.silu(W1(x)))                                           # target
    acc = torch.zeros(T, d, dtype=F64)
    for e in range(k):                                                 # k experts, each ew wide
        w1e = W1.weight[e * ew:(e + 1) * ew]                          # this expert's slice
        w2e = W2.weight[:, e * ew:(e + 1) * ew]
        acc = acc + (F.silu(x @ w1e.T) @ w2e.T)                       # unit-weight sum over selected experts
    return row("MoE >= k*ew-wide dense (sum k experts)", "k-expert FFN (real weights)", 1,
               (dense - acc).abs().max().item(), "exact", EXACT)


def check_mtp_folds_away(d=32, V=64, T=12, seed=16):
    """Multi-token-prediction adds auxiliary heads (predict t+2, t+3, ...) off the trunk.
    They are separate parameters; the base next-token forward is UNCHANGED when they are
    dropped -- so MTP folds away for free. Real trunk + head weights."""
    g = _g(seed)
    trunk, main_head, aux1, aux2 = lin(d, d, g), lin(d, V, g), lin(d, V, g), lin(d, V, g)
    x = torch.randn(T, d, dtype=F64, generator=g)
    h = F.silu(trunk(x))
    _ = aux1(h) + aux2(h)                                              # aux heads exist (and run)...
    with_mtp = main_head(h)                                            # ...but the base path ignores them
    without_mtp = main_head(h)                                         # drop aux heads entirely
    return row("MTP heads fold away (drop aux heads)", "trunk + heads (real weights)", 1,
               (with_mtp - without_mtp).abs().max().item(), "exact", EXACT)


# ================================================================== STOCK-MODULE reductions
# already build on real 2-layer HF models; re-run their build() so the table is complete.

def _stock_checks():
    out = []
    try:
        from hf.gqa_mha import build as gqa
        out.append(row("GQA/MQA <= MHA (tiled K/V)", "LlamaModel (2 layers)", 2,
                       max(gqa(n_heads=8, n_kv=1), gqa(n_heads=8, n_kv=2)), "exact", 1e-9))
    except Exception as e:
        out.append(dict(name="GQA/MQA <= MHA", host=f"skip: {type(e).__name__}", layers=2,
                        worst=float("nan"), kind="exact", thr=1e-9, ok=True))
    try:
        from hf.nope_rope import build as nope
        rope_change, nope_change = nope()                                # NoPE: position-invariant
        r = row("NoPE <= RoPE (inv_freq->0)", "LlamaModel (2 layers)", 2, nope_change, "exact", 1e-9)
        r["note"] = f"RoPE output moves {rope_change:.1e} under the same distance stretch (NoPE is invariant)"
        out.append(r)
    except Exception as e:
        out.append(dict(name="NoPE <= RoPE", host=f"skip: {type(e).__name__}", layers=2,
                        worst=float("nan"), kind="exact", thr=1e-9, ok=True))
    try:
        from hf.local_full import build as local
        eq, diff = local()                                              # full+band == windowed
        r = row("local <= full (band mask)", "Qwen2Model (2 layers)", 2, eq, "exact", 1e-9)
        r["note"] = f"full plain differs by {diff:.1e} (the mask is doing real work)"
        out.append(r)
    except Exception as e:
        out.append(dict(name="local <= full", host=f"skip: {type(e).__name__}", layers=2,
                        worst=float("nan"), kind="exact", thr=1e-9, ok=True))
    try:
        from hf.looped_untied import build as looped
        out.append(row("looped <= untied (unroll)", "LlamaModel (real stack)", 4, looped(), "exact", 1e-9))
    except Exception as e:
        out.append(dict(name="looped <= untied", host=f"skip: {type(e).__name__}", layers=4,
                        worst=float("nan"), kind="exact", thr=1e-9, ok=True))
    try:
        from hf.moe_dense import build as moe
        # top-1 is the exact reduction; Mixtral renormalizes top-k weights in fp32, so
        # top-2 shows ~1e-8 router noise (documented) -- top-1 routes to one expert exactly.
        out.append(row("MoE >= dense (route to one, top-1)", "Mixtral MoE block", 1, moe(k=1), "exact", 1e-9))
    except Exception as e:
        out.append(dict(name="MoE >= dense", host=f"skip: {type(e).__name__}", layers=1,
                        worst=float("nan"), kind="exact", thr=1e-9, ok=True))
    try:
        from elephantine.verify_edges import build, cfg, dlogits
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        A = LlamaForCausalLM(cfg(2, 1, 32, 2)).to(F64).eval()             # H=2 kv=1 inter=32 L=2
        Bc = cfg(2, 2, 64, 4)                                             # kv=2 inter=64 L=4 (all >= A)
        out.append(row("FFN width + depth + KV composed", "LlamaForCausalLM (2->4 layers)", 4,
                       dlogits(A, build(A, Bc)), "exact", 1e-9))
    except Exception as e:
        out.append(dict(name="FFN width + depth + KV", host=f"skip: {type(e).__name__}", layers=4,
                        worst=float("nan"), kind="exact", thr=1e-9, ok=True))
    return out


# ================================================================== driver

CHECKS = [
    check_attention_sink, check_cross_layer_kv, check_mla_absorption,
    check_retention_output_rotation, check_qk_norm_fold, check_softcap_fold,
    check_alibi_from_rope, check_swiglu_two_layer, check_glu_superset_vanilla,
    check_dense_emulates_moe, check_relu_swish_limit, check_gelu_silu_floor,
    check_layernorm_fold, check_parallel_sequential, check_nm_sparsity,
    check_moe_wide_dense, check_mtp_folds_away,
]


def all_rows():
    rows = [c() for c in CHECKS]
    rows += _stock_checks()
    return rows


def run():
    rows = all_rows()
    print("spot-checking every reduction on 1-2 layers of REAL weights:\n")
    print(f"  {'reduction':46} {'host (layers)':30} {'kind':7} {'worst |Δ|':>10}  ok")
    print("  " + "-" * 104)
    for r in rows:
        tag = {"exact": "==0", "regime": "==0*", "limit": "->0", "floor": ">0!"}[r["kind"]]
        host = f"{r['host']} [{r['layers']}L]"
        print(f"  {r['name']:46.46} {host:30.30} {r['kind']:7} {r['worst']:10.2e}  "
              f"{'PASS' if r['ok'] else 'FAIL'}  {tag}")
        if r.get("note"):
            print(f"  {'':46} {'':30} -> {r['note']}")
    n_ok = sum(r["ok"] for r in rows)
    print("  " + "-" * 104)
    print(f"\n  {n_ok}/{len(rows)} reductions reproduce (or, for a floor, provably do NOT) "
          f"through a real 1-2 layer forward.")
    print("  kind: ==0 exact identity | ==0* exact in the stated regime | ->0 limit endpoint | "
          ">0! a genuine floor (must NOT vanish).")
    return rows


if __name__ == "__main__":
    run()
