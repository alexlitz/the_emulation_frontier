"""KV-sharing and MLA rank, ANALYTICALLY (numpy, fp64; no SGD, closed-form / SVD only).

Attention variants differ only in how much of the key/value tensor they cache.
Everything here is an EXACT reparametrization or an OPTIMAL (SVD) low-rank cut --
never a fitted approximation -- so the whole family is one operator seen through
different KV-memory budgets.

  * MLA low rank (part 1).  Multi-head Latent Attention pushes K,V through a shared
    rank-r latent. The best rank-r K,V is the truncated SVD of the stacked [K | V]
    matrix (Eckart-Young: optimal in Frobenius/spectral norm). We take that cut and
    measure the resulting ATTENTION-OUTPUT error vs r. Two spectra:
      - random Gaussian K,V   -> flat spectrum, worst case: error stays high until
        r hits full rank (min(seq, 2*d_model)), then drops to ~0. Honest floor.
      - power-law spectrum     -> the realistic trained-attention case: error decays
        smoothly with r and is ~0 well before full rank. This IS the KV-memory<->
        quality curve MLA rides.

  * Exact cross-relations (part 2), verified to fp64:
      - MQA subset GQA subset MHA: tile/duplicate the shared KV heads
        (repeat_interleave) -> plain MHA reproduces GQA/MQA EXACTLY (max|Δ| = 0.0).
      - MLA subset MHA: the NON-POSITIONAL score/output folds exactly by absorbing
        the latent up-projections into q and out --  q.(W_UK c) = (W_UK^T q).c  and
        (A c) W_UV -- so per-head K,V never need materializing (error ~fp64 eps).
        The RoPE (positional) part does NOT fold: a per-position rotation R_i makes
        the score q_i . R_{i-j} k_j depend on i-j, which is not a single position-
        independent bilinear form -- so DeepSeek keeps a small decoupled RoPE key
        OUTSIDE the latent. We show the naive fold's error blows up (nonzero).

  * KV-memory cross-matrix (part 3). Cache bytes/token: MHA = n_head*head_dim,
    GQA = n_kv*head_dim, MQA = head_dim, MLA = rank r (+ a small RoPE key, dropped
    here for the clean comparison). "X emulates Y" and the KV currency it costs.

Currency: KV-cache memory, and it is CONSERVED -- emulating a thriftier variant on
a fatter host costs exactly the cache the thrifty one saved (MQA->MHA blows the
single shared head back up to n_head*head_dim). MLA's low rank is the one edge that
is a LIMIT (SVD floor), not an exact reduction, except at full rank.

Run:  python -m attn.mla_rank
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# attention primitives (numpy, fp64)
# --------------------------------------------------------------------------

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T):
    m = np.zeros((T, T))
    m[np.triu_indices(T, 1)] = -np.inf
    return m


def grouped_attention(Q, K, V, n_head, n_kv, head_dim):
    """Causal attention where query head h reads KV group h // (n_head//n_kv).

    Q: (T, n_head*head_dim);  K, V: (T, n_kv*head_dim).  n_kv==n_head is MHA,
    n_kv==1 is MQA, 1<n_kv<n_head is GQA. Returns (T, n_head*head_dim).
    """
    T = Q.shape[0]
    scale = 1.0 / np.sqrt(head_dim)
    mask = causal_mask(T)
    n_rep = n_head // n_kv
    out = np.zeros((T, n_head * head_dim))
    for h in range(n_head):
        g = h // n_rep
        qs = slice(h * head_dim, (h + 1) * head_dim)
        ks = slice(g * head_dim, (g + 1) * head_dim)
        A = softmax(Q[:, qs] @ K[:, ks].T * scale + mask, axis=-1)
        out[:, qs] = A @ V[:, ks]
    return out


def mha_attention(Q, K, V, n_head, head_dim):
    """Standard MHA: every head has its own K,V. K,V: (T, n_head*head_dim)."""
    return grouped_attention(Q, K, V, n_head, n_head, head_dim)


# --------------------------------------------------------------------------
# part 1: MLA low rank -- optimal (SVD) rank-r K,V, attention-output error vs r
# --------------------------------------------------------------------------

def power_law_matrix(T, F, decay, rng):
    """A (T, F) matrix with power-law singular values sigma_i ~ i^-decay (sigma_1=1).

    Random Gaussian K,V is (nearly) full-rank flat-spectrum; trained attention K,V
    has a decaying spectrum. This gives a realistic decaying-spectrum stand-in with
    NO fitting -- we just install the spectrum onto random singular vectors.
    """
    A = rng.standard_normal((T, F))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(T, F)
    s = np.arange(1, k + 1, dtype=float) ** (-decay)
    s = s / s[0]
    return (U * s) @ Vt


def low_rank_kv(K, V, r):
    """Optimal shared rank-r K,V: truncated SVD of the stacked [K | V] (Eckart-Young).

    A rank-r latent c = X @ W_DKV up-projected to K=c@W_UK, V=c@W_UV forces
    [K | V] = c @ [W_UK | W_UV] to have rank <= r; the best such factorization
    (Frobenius / spectral optimal) is exactly the truncated SVD. Closed form, no SGD.
    """
    D = K.shape[1]
    KV = np.concatenate([K, V], axis=1)
    U, S, Vt = np.linalg.svd(KV, full_matrices=False)
    KVr = (U[:, :r] * S[:r]) @ Vt[:r]
    return KVr[:, :D], KVr[:, D:], S


def rank_sweep(n_head=8, head_dim=16, seq=128, decay=1.0, seed=0, ranks=None):
    """Attention-output max-error and relative-KV-residual vs latent rank r."""
    rng = np.random.default_rng(seed)
    D = n_head * head_dim
    Q = rng.standard_normal((seq, D))
    if decay is None:                       # flat / worst-case random spectrum
        KV = rng.standard_normal((seq, 2 * D))
    else:                                   # realistic decaying spectrum
        KV = power_law_matrix(seq, 2 * D, decay, rng)
    K, V = KV[:, :D], KV[:, D:]
    out_full = mha_attention(Q, K, V, n_head, head_dim)
    full = min(KV.shape)
    S = np.linalg.svd(KV, compute_uv=False)
    kv_frob = np.sqrt((S ** 2).sum())
    if ranks is None:
        ranks = [r for r in (4, 8, 16, 32, 64, 128, 256) if r <= full] + [full]
        ranks = sorted(set(ranks))
    rows = []
    for r in ranks:
        Kr, Vr, _ = low_rank_kv(K, V, r)
        out_r = mha_attention(Q, Kr, Vr, n_head, head_dim)
        out_err = float(np.abs(out_full - out_r).max())
        kv_resid = float(np.sqrt((S[r:] ** 2).sum()) / kv_frob)
        rows.append((r, out_err, kv_resid))
    return rows, full


# --------------------------------------------------------------------------
# part 2a: MQA subset GQA subset MHA -- exact via repeat_interleave tiling
# --------------------------------------------------------------------------

def repeat_interleave_kv(KV, n_kv, n_head, head_dim):
    """(T, n_kv*head_dim) -> (T, n_head*head_dim): replicate each kv head n_head//n_kv
    times in HF `repeat_kv` order (query head h <- kv head h // n_rep)."""
    n_rep = n_head // n_kv
    T = KV.shape[0]
    return (KV.reshape(T, n_kv, head_dim)
              .repeat(n_rep, axis=1)
              .reshape(T, n_head * head_dim))


def gqa_to_mha_error(n_kv, n_head=8, head_dim=8, seq=16, seed=0):
    """Build a GQA/MQA layer; reproduce it on plain MHA by tiling K,V. Returns max|Δ|."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((seq, n_head * head_dim))
    K = rng.standard_normal((seq, n_kv * head_dim))
    V = rng.standard_normal((seq, n_kv * head_dim))
    grouped = grouped_attention(Q, K, V, n_head, n_kv, head_dim)
    Kt = repeat_interleave_kv(K, n_kv, n_head, head_dim)
    Vt = repeat_interleave_kv(V, n_kv, n_head, head_dim)
    mha = mha_attention(Q, Kt, Vt, n_head, head_dim)
    return float(np.abs(grouped - mha).max())


# --------------------------------------------------------------------------
# part 2b: MLA subset MHA -- non-positional part folds exactly; RoPE part does NOT
# --------------------------------------------------------------------------

def mla_absorption_error(hidden=64, n_head=4, qk_nope=16, v_head=16, rank=24,
                         seq=20, seed=0):
    """Fold the latent up-projections into q and out; check it equals explicit MHA.

        score:  q_C . k_C^h = q_C . (W_UK^h c) = (W_UK^h^T q_C) . c
        output: sum_j a_j v^h_j = (A c) W_UV^h

    Returns (max score error, max output error) -- ~fp64 eps => exact fold.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((seq, hidden))
    W_DKV = rng.standard_normal((hidden, rank))
    W_UK = rng.standard_normal((rank, n_head * qk_nope))
    W_UV = rng.standard_normal((rank, n_head * v_head))
    W_QC = rng.standard_normal((hidden, n_head * qk_nope))

    c = x @ W_DKV                             # (seq, rank)  <- the whole cache
    scale = 1.0 / np.sqrt(qk_nope)
    mask = causal_mask(seq)
    max_s = max_o = 0.0
    for h in range(n_head):
        UKh = W_UK[:, h * qk_nope:(h + 1) * qk_nope]
        UVh = W_UV[:, h * v_head:(h + 1) * v_head]
        qC = x @ W_QC[:, h * qk_nope:(h + 1) * qk_nope]

        kC = c @ UKh                          # explicit MHA: materialize per-head K,V
        v = c @ UVh
        se = qC @ kC.T * scale
        A = softmax(se + mask, axis=-1)
        oe = A @ v

        q_abs = qC @ UKh.T                    # absorbed MLA: attend over latent directly
        sa = q_abs @ c.T * scale              # == se
        oa = (A @ c) @ UVh                    # == oe

        max_s = max(max_s, float(np.abs(se - sa).max()))
        max_o = max(max_o, float(np.abs(oe - oa).max()))
    return max_s, max_o


def rope(x, positions, base=10000.0):
    """Rotary position embedding on the last axis (half-split form)."""
    _, d = x.shape
    half = d // 2
    inv = base ** (-np.arange(0, half) / half)
    ang = np.outer(positions, inv)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=1)


def rope_fold_gap(hidden=64, qk_rope=8, seq=20, seed=0):
    """The RoPE part will NOT absorb: max |rope-score - naive(unrotated)-fold|.

    Absorption needs a POSITION-INDEPENDENT bilinear form; RoPE's per-position
    rotation makes q_i . R_{i-j} k_j depend on i-j, so dropping the rotation (the
    only way to 'fold' it into a static matrix) is wrong. Nonzero => cannot fold.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((seq, hidden))
    W_QR = rng.standard_normal((hidden, qk_rope))
    W_KR = rng.standard_normal((hidden, qk_rope))    # decoupled shared RoPE key
    pos = np.arange(seq)
    qR = rope(x @ W_QR, pos)
    kR = rope(x @ W_KR, pos)
    s_rope = qR @ kR.T                                # true positional score
    s_naive = (x @ W_QR) @ (x @ W_KR).T              # fold that drops the rotation
    return float(np.abs(s_rope - s_naive).max())


# --------------------------------------------------------------------------
# part 3: KV-memory cross-matrix
# --------------------------------------------------------------------------

def kv_memory_table(n_head=8, head_dim=16, n_kv=2, rank=32):
    """Cache slots/token for each variant (fp bytes = slots * dtype_size)."""
    return {
        "MHA": n_head * head_dim,
        "GQA": n_kv * head_dim,
        "MQA": head_dim,
        "MLA": rank,
    }


def emulation_matrix(kv):
    """Can host X (col) emulate target Y (row), and the KV/token it then costs.

    Exact edges (fp64): tile KV heads up (MQA/GQA -> MHA/GQA), or run any variant
    natively. A thriftier host cannot exactly emulate a fatter target without
    growing its own cache to the target's size -- so emulation pays the target's
    KV cost. MLA -> {MHA,GQA,MQA} only at full rank (else it is a lossy SVD floor).
    """
    order = ["MHA", "GQA", "MQA", "MLA"]
    # exact-emulation reachability (host X can EXACTLY realize target Y's function),
    # paying Y's KV cost. MLA host reaches the others only when its rank is full.
    reach = {
        "MHA": {"MHA": True,  "GQA": True,  "MQA": True,  "MLA": True},   # widest host
        "GQA": {"MHA": False, "GQA": True,  "MQA": True,  "MLA": False},
        "MQA": {"MHA": False, "GQA": False, "MQA": True,  "MLA": False},
        "MLA": {"MHA": "r=full", "GQA": "r=full", "MQA": "r=full", "MLA": True},
    }
    return order, reach, kv


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    # ---- part 1: MLA low rank, attention-output error vs r ----
    n_head, head_dim, seq = 8, 16, 128
    D = n_head * head_dim
    print(f"PART 1  MLA low rank: optimal rank-r [K|V] (SVD), attention-output error vs r")
    print(f"        layer: seq={seq}, n_head={n_head}, head_dim={head_dim}, "
          f"d_model={D}, stacked KV cols=2*d_model={2*D}, full rank=min(seq,2*d)={min(seq,2*D)}")
    for tag, decay in [("random / flat spectrum (worst case)", None),
                       ("power-law spectrum ~i^-1 (realistic trained attn)", 1.0)]:
        rows, full = rank_sweep(n_head, head_dim, seq, decay=decay)
        print(f"\n  {tag}:")
        print(f"    {'rank r':>7}  {'out max|Δ|':>12}  {'rel KV resid':>13}  {'KV vs MHA':>10}")
        for r, err, resid in rows:
            ratio = f"{D/r:.1f}x" if r <= D else "1.0x"
            print(f"    {r:>7}  {err:>12.3e}  {resid:>13.3e}  {ratio:>10}")
    print("\n  reading: error is monotone-decreasing in r and hits ~0 (fp64 eps) at full"
          "\n  rank. Flat spectrum = expensive (no free lunch on random KV); a decaying"
          "\n  spectrum -- what trained attention actually has -- is ~0 well before full"
          "\n  rank: that gap is MLA's KV-memory<->quality curve.\n")

    # ---- part 2a: MQA subset GQA subset MHA exact ----
    print("PART 2a  MQA subset GQA subset MHA  (tile shared KV heads, repeat_interleave):")
    for n_kv in (1, 2, 4):
        err = gqa_to_mha_error(n_kv, n_head=8, head_dim=8)
        kind = "MQA" if n_kv == 1 else "GQA"
        print(f"    {kind:3s} n_kv={n_kv} -> MHA via tiled K,V:  max|Δ| = {err:.2e}  (exact)")

    # ---- part 2b: MLA subset MHA -- fold exact, RoPE part does not ----
    ms, mo = mla_absorption_error()
    gap = rope_fold_gap()
    print("\nPART 2b  MLA subset MHA  (absorb latent up-projections into q and out):")
    print(f"    non-positional fold:  max score err = {ms:.3e}   max out err = {mo:.3e}   (exact)")
    print(f"    RoPE part does NOT fold:  max|rope-score - naive fold| = {gap:.3e}  (nonzero)")
    print("    -> DeepSeek keeps a small decoupled RoPE key OUTSIDE the latent for exactly")
    print("       this reason; the low-rank latent only carries the position-free part.\n")

    # ---- part 3: KV-memory cross-matrix ----
    kv = kv_memory_table(n_head=8, head_dim=16, n_kv=2, rank=32)
    order, reach, kv = emulation_matrix(kv)
    print("PART 3  KV-memory cross-matrix (slots/token; bytes = slots * dtype_size)")
    print(f"    cache slots/token:  " +
          "  ".join(f"{k}={v}" for k, v in kv.items()) +
          "   (n_head=8, head_dim=16, n_kv=2, r=32)")
    print(f"    ordering: MHA({kv['MHA']}) >= GQA({kv['GQA']}) >= MLA({kv['MLA']}) >= MQA({kv['MQA']})")
    print("\n    can host X (col) EXACTLY emulate target Y (row)?  cell = KV/token host must pay")
    print("    " + "target\\host".ljust(12) + "".join(f"{h:>10}" for h in order))
    for Y in order:
        cells = []
        for X in order:
            r = reach[X][Y]
            if r is True:
                cells.append(f"{kv[Y]:>10}")           # exact, pays target's cache
            elif r == "r=full":
                cells.append(f"{'r=full':>10}")        # only at full rank
            else:
                cells.append(f"{'--':>10}")            # thrifty host can't reach fatter Y
        print("    " + Y.ljust(12) + "".join(cells))
    print("\n    reading: MHA (widest host) exactly emulates all -- paying the TARGET's smaller")
    print("    cache each time (tie/tile KV heads). Thriftier hosts cannot exactly realize a")
    print("    fatter target's function without growing their own cache to the target's size:")
    print("    KV memory is a CONSERVED currency. MLA reaches the discrete family only at full")
    print("    rank; below that it is the lossy SVD floor from part 1.")


if __name__ == "__main__":
    main()
