"""Positional-encoding reductions, ANALYTICALLY (numpy only, fp64, no SGD).

Every scheme adds a field g(i,j) to the (i,j) attention LOGIT. We characterise each
by the SPAN of fields its construction can add, then read reductions off the spans:
host X emulates target Y iff Y's field lies in X's span (floor = min-max residual of
the closed-form least-squares fit). Four results:

  1. NoPE  = RoPE at inv_freq -> 0. The rotation is the identity, the score depends on
     content only. EXACT 0.0 (a free reverse; any host emulates NoPE by zeroing its field).

  2. ALiBi = a fixed RoPE pair (the project's seed result). q=(-N,0), k=(0,1) give
     <R_i q, R_j k> = -N sin(theta (i-j))  EXACTLY, and -N sin(theta d) -> -(N theta) d
     as theta -> 0: the linear relative bias of ALiBi. Slope m = N*theta is held fixed
     while theta -> 0 (so N -> inf: MAGNITUDE currency). Error is the sin remainder,
     ~ m*theta^2*D^3/6 over distances |d|<=D -- QUADRATIC in the angle (exact -> limit).

  3. RoPE -> hard local window. N extra RoPE dims add a relative bias
     f(d) = sum_k A_k cos(w_k d); the partial Fourier sum of a rectangular window (1 in
     band |d|<=w, 0 out) is the natural construction. A hard mask is DISCONTINUOUS, so
     in sup-norm this FLOORS (Gibbs: the partial sum hits the mid-value 0.5 at the jump,
     max err ~0.5, and the overshoot settles at ~0.09 -- neither vanishes with N). Away
     from the edge it converges ~1/N; in L2 ~1/sqrt(N); and if the window edge is SMOOTH
     (a raised-cosine ramp, no discontinuity) N frequencies get you there SPECTRALLY (fast).
     So the blog's "very closely approximate a hard local mask" is true only up to a Gibbs
     floor at the boundary; the interior/soft-edge cost is small.

  4. Cross-emulation matrix over {NoPE, RoPE, ALiBi, learned-abs, T5-relbias, sinusoidal}
     -- which span contains which, analogous to act/activation_cross.py.

Currency: NoPE free; ALiBi<=RoPE magnitude (tiny angle); RoPE->window width N (Gibbs floor).
No scipy, no fitting-by-SGD -- all closed form (lstsq / Fourier coefficients / sin remainder).

Run:  python -m attn.positional_cross
"""
import numpy as np

# ------------------------------------------------------------------ RoPE primitives

def rot(phi):
    """2x2 rotation by angle phi."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def rope_pair_score(N, theta, i, j):
    """<R_i q, R_j k> for q=(-N,0), k=(0,1): the fixed RoPE pair. == -N sin(theta(i-j))."""
    q = rot(i * theta) @ np.array([-float(N), 0.0])
    k = rot(j * theta) @ np.array([0.0, 1.0])
    return float(q @ k)


# ------------------------------------------------------ 1. NoPE = RoPE at inv_freq->0

def nope_from_rope_error(seq=16, seed=0):
    """inv_freq -> 0 is the identity rotation: the RoPE score becomes content-only (NoPE).
    We compare the RoPE score field at inv_freq=0 against the position-free content score.
    EXACT 0.0."""
    rng = np.random.default_rng(seed)
    d = 8
    q = rng.standard_normal((seq, d))
    k = rng.standard_normal((seq, d))
    # RoPE with inv_freq -> 0: every rotation angle is 0, rotations are identity.
    inv_freq = np.zeros(d // 2)
    pos = np.arange(seq)
    ang = pos[:, None] * inv_freq[None, :]            # all zero
    cos, sin = np.cos(ang), np.sin(ang)               # cos=1, sin=0 -> identity
    qe, ke = q.reshape(seq, d // 2, 2), k.reshape(seq, d // 2, 2)
    qr = np.stack([qe[..., 0] * cos - qe[..., 1] * sin,
                   qe[..., 0] * sin + qe[..., 1] * cos], -1).reshape(seq, d)
    kr = np.stack([ke[..., 0] * cos - ke[..., 1] * sin,
                   ke[..., 0] * sin + ke[..., 1] * cos], -1).reshape(seq, d)
    rope_scores = qr @ kr.T
    content_scores = q @ k.T                          # pure NoPE (no position)
    return float(np.abs(rope_scores - content_scores).max())


# ---------------------------------------------- 2. ALiBi = a fixed RoPE pair (seed)

def alibi_identity_error(N=7.3, theta=0.01, D=40):
    """<R_i q, R_j k> == -N sin(theta(i-j)) EXACTLY (fp noise) for the fixed pair."""
    e = 0.0
    for i in range(D):
        for j in range(i + 1):
            lhs = rope_pair_score(N, theta, i, j)
            rhs = -N * np.sin(theta * (i - j))
            e = max(e, abs(lhs - rhs))
    return e


def alibi_smallangle_error(slope, theta, D=64):
    """Fix ALiBi slope m=slope; the RoPE pair with angle theta has N=m/theta so
    -N sin(theta d) approximates -m d. Return max|score - (-m d)| over 0<=d<=D.
    The sin remainder makes this ~ m*theta^2*D^3/6  (QUADRATIC in theta)."""
    N = slope / theta
    d = np.arange(D + 1.0)
    rope = -N * np.sin(theta * d)          # what RoPE actually produces
    alibi = -slope * d                     # the ALiBi linear bias
    return float(np.abs(rope - alibi).max())


def rope_score(freqs, q, k, i, j):
    """A RoPE head's logit as a sum over its pairs: sum_p <R(i w_p) q_p, R(j w_p) k_p>.
    freqs (P,), q / k (P, 2) -- one row per rotating pair."""
    freqs, q, k = np.asarray(freqs, float), np.asarray(q, float), np.asarray(k, float)
    out = 0.0
    for w, qp, kp in zip(freqs, q, k):
        out += float((rot(i * w) @ qp) @ (rot(j * w) @ kp))
    return out


def alibi_rope_host(q, k, m, theta):
    """The whole RoPE host head for an ALiBi head of slope m: the content pairs q / k (P, 2) at
    frequency 0 (unrotated, a buffer write as for NoPE), plus ONE frozen pair at frequency theta with
    the constant components q = (-m/theta, 0), k = (0, 1). Returns (freqs, q_host, k_host) with the
    frozen pair last. Its logit is q.k - (m/theta) sin(theta (i - j))  ->  q.k - m (i - j)."""
    q, k = np.asarray(q, float), np.asarray(k, float)
    freqs = np.r_[np.zeros(len(q)), theta]
    return freqs, np.vstack([q, [-m / theta, 0.0]]), np.vstack([k, [0.0, 1.0]])


# ------------------------------------------ 3. RoPE -> hard local window (Fourier)

def _circ_dist(P):
    d = np.arange(P)
    return np.minimum(d, P - d)


def rope_window_partial_sum(P, w, N, smooth=0):
    """N-term truncated cosine (RoPE relative-bias) approx of a length-P periodic window.
    Hard target (smooth=0): 1 for circular-dist<=w else 0 (a rectangular band).
    Smooth target (smooth=s>0): a raised-cosine edge of ramp width s (no discontinuity).
    Returns (dist, target, approx). The cosines ARE the fields sum_k A_k cos(w_k d) that
    RoPE on N extra 2-D frequency dims contributes to the relative logit."""
    d = np.arange(P)
    dd = _circ_dist(P)
    if smooth <= 0:
        target = (dd <= w).astype(float)
    else:
        t = np.clip((dd - (w - smooth / 2)) / smooth, 0.0, 1.0)
        target = 0.5 * (1 + np.cos(np.pi * t))
    approx = np.zeros(P)
    for k in range(N + 1):
        ck = (2.0 if k > 0 else 1.0) / P * np.sum(target * np.cos(2 * np.pi * k * d / P))
        approx += ck * np.cos(2 * np.pi * k * d / P)   # a genuine RoPE relative frequency
    return dd, target, approx


def window_errors(P, w, N, smooth=0, guard=4):
    """Three honest metrics for the N-frequency window approximation."""
    dd, target, approx = rope_window_partial_sum(P, w, N, smooth)
    err = np.abs(approx - target)
    linf = float(err.max())                                  # includes the jump: Gibbs floor
    interior = float(err[np.abs(dd - w) > guard].max())      # away from the edge: converges
    rms = float(np.sqrt(np.mean(err ** 2)))                  # L2: ~1/sqrt(N)
    overshoot = float(approx.max() - 1.0)                    # Gibbs overshoot -> ~0.09 floor
    return linf, interior, rms, overshoot


# ---------------------------------------------- 4. cross-emulation matrix (spans)

def _grid(L):
    ii = np.arange(L)
    I, J = np.meshgrid(ii, ii, indexing="ij")
    return ii, (I - J).astype(float)


def _vaswani_P(L, dim=16, base=1e4):
    pos = np.arange(L)[:, None]
    k = np.arange(dim // 2)[None, :]
    ang = pos / base ** (2 * k / dim)
    return np.concatenate([np.sin(ang), np.cos(ang)], 1)     # L x dim FIXED embedding


def _spans(L):
    """The reachable additive-logit-field span (list of LxL basis fields) of each scheme."""
    ii, D = _grid(L)
    one = np.ones((L, L))
    P = _vaswani_P(L)

    nope = [one]
    alibi = [D, one]                                          # linear relative ramp
    rope = [one]                                              # band-limited RELATIVE (geometric w_k)
    for w in (1e4 ** (-np.arange(20) / 20)) * np.pi:
        rope += [np.cos(w * D), np.sin(w * D)]
    t5 = [(D == d).astype(float) for d in np.unique(D)]       # ARBITRARY relative g(d)
    labs = ([np.repeat((np.eye(L)[i])[:, None], L, 1) for i in range(L)] +
            [np.repeat((np.eye(L)[j])[None, :], L, 0) for j in range(L)])  # arbitrary a(i)+b(j)
    sinus = []                                                # FIXED abs embedding + PE.PE relative
    for c in range(P.shape[1]):
        sinus.append(np.repeat(P[:, c][:, None], L, 1))       # a(i) from fixed P
        sinus.append(np.repeat(P[:, c][None, :], L, 0))       # b(j) from fixed P
    sinus.append(P @ P.T)                                     # relative even PE.PE term
    return {"nope": nope, "alibi": alibi, "rope": rope,
            "t5_relbias": t5, "learned_abs": labs, "sinusoidal": sinus}


def _targets(L):
    """The canonical field each scheme WANTS to produce, normalised to max|.|=1."""
    ii, D = _grid(L)
    P = _vaswani_P(L)
    norm = lambda a: a if np.abs(a).max() == 0 else a / np.abs(a).max()
    return {
        "nope": np.zeros((L, L)),
        "alibi": norm(-D),
        "rope": norm(np.cos(0.7 * D) + 0.5 * np.sin(1.9 * D)),
        "t5_relbias": norm((np.abs(D) <= 3).astype(float) - 0.15 * np.abs(D)),
        "learned_abs": norm(np.sin(1.3 * ii)[:, None] + np.cos(0.7 * ii)[None, :]),
        "sinusoidal": norm(P @ P.T),
    }


def _fit(cols, target):
    A = np.stack([c.ravel() for c in cols], 1)
    t = target.ravel()
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return float(np.abs(A @ coef - t).max())


def cross_matrix(L=24):
    """floor[Y][X] = min-max residual of best host-X fit to target-Y's field (0 = exact)."""
    spans, targets = _spans(L), _targets(L)
    names = list(spans)
    M = {Y: {X: (0.0 if X == Y else _fit(spans[X], targets[Y])) for X in names} for Y in names}
    # RoPE emulating the ALiBi ramp is exact-in-LIMIT, not exact-at-finite-angle. Overwrite that
    # one cell with the single-physical-pair finite-angle floor (the honest seed number).
    M["alibi"]["rope"] = alibi_smallangle_error(1.0, 0.03, D=L - 1) / (1.0 * (L - 1))
    return names, M


# ------------------------------------------------------------------------ main

def main():
    print("1. NoPE = RoPE at inv_freq -> 0 (identity rotation), max|score diff|:")
    print(f"   {nope_from_rope_error():.1e}   (EXACT 0.0 -- free reverse)\n")

    print("2. ALiBi = a fixed RoPE pair  q=(-N,0), k=(0,1):")
    print(f"   identity  <R_i q,R_j k> == -N sin(theta(i-j)):  max|diff| = {alibi_identity_error():.1e}  (exact)")
    print("   small-angle -N sin(theta d) -> -(N theta) d, slope m=1 fixed, over |d|<=64")
    print("   (theta*D<1 so we are IN the small-angle regime; ratio col = err(theta)/err(theta/2)):")
    prev = None
    for th in (1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4):
        e = alibi_smallangle_error(1.0, th)
        ratio = "" if prev is None else f"  ratio~{prev / e:.2f} (=4 => theta^2)"
        print(f"     theta={th:.3e}  max|Δ|={e:.2e}   (~ m theta^2 D^3/6){ratio}")
        prev = e
    print("     -> QUADRATIC in the angle: halve theta => quarter the error (exact -> limit).\n")

    print("3. RoPE (N extra freq dims) -> hard local window w, P=256 w=16, max|Δ| of the bias:")
    print("   N    Linf(w/ jump)  interior(no edge)   RMS(L2)   Gibbs-overshoot")
    for N in (4, 8, 16, 32, 64):
        linf, inter, rms, ov = window_errors(256, 16, N)
        print(f"   {N:3d}    {linf:8.4f}       {inter:8.4f}       {rms:8.4f}      {ov:8.4f}")
    print("   -> HARD edge FLOORS in sup-norm (~0.5 at the jump, overshoot -> ~0.09; a Gibbs FLOOR).")
    print("      interior ~1/N, L2 ~1/sqrt(N). With a SMOOTH (raised-cosine) edge instead:")
    print("   N    Linf(smooth edge s=8)")
    for N in (4, 8, 16, 32, 64):
        linf, *_ = window_errors(256, 16, N, smooth=8)
        print(f"   {N:3d}    {linf:.2e}")
    print("   -> no discontinuity => SPECTRAL (fast) decay. So the hard-mask floor is the edge only.\n")

    names, M = cross_matrix()
    print("4. positional cross-emulation FLOOR  host X (col) -> target Y (row), normalised max|Δ|:")
    print("   (0=exact; ~1=incomparable; small=cheap floor)")
    print("   target\\host  " + "  ".join(f"{n:>11}" for n in names))
    for Y in names:
        cells = []
        for X in names:
            v = M[Y][X]
            cells.append("      exact" if v == 0 else f"{v:11.4f}")
        print(f"   {Y:>11}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
