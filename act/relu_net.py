"""Every activation is emulable by a ReLU network — ANALYTICALLY, explicit weights,
no training (training would only confuse representation with optimization).

  * An UNGATED activation phi(z) (SiLU, GeLU, mish, tanh) is a 1-D curve, so a SINGLE
    hidden layer of N ReLUs — the piecewise-linear interpolant — approximates it with
    error ~ 1/N^2:  phi(z) ~ phi(k0) + sum_i c_i * ReLU(z - k_i), the c_i the second
    differences of phi at the knots. One layer genuinely suffices.

  * A GATED activation (SwiGLU = SiLU(a)*b, GeGLU = GeLU(a)*b) needs the PRODUCT.
    Write  p*b = 1/4((p+b)^2 - (p-b)^2)  and build x^2 from ReLUs by Yarotsky's
    sawtooth:  x^2 ~ x - sum_{s=1..L} g_s(x)/4^s,  g the tent map, g_s its s-fold
    composition (depth L), error ~ 4^-L. A single layer CAN do the product too
    (universal approximation), but needs ~O(1/eps) width; depth is exponentially
    cheaper — that is the only sense in which the gate "needs" depth.

  * FREE knots (free_knot_minimax): K ReLUs + an affine skip with the knots placed by a
    true minimax search. On [-8,8], K=1/2/3 gives SiLU 0.140/0.050/0.050, GeLU
    0.086/0.058/0.058, mish 0.110/0.057/0.042, tanh 0.605/0.071/0.071. 2-3 free units match
    ~16 uniform ones. The error falls in steps (symmetric curves gain from knots in pairs),
    and K^2*err only levels off (~0.5-0.9) for K >= 5, so ~1/K^2 holds only asymptotically.

Currency: width for the curve (~1/N^2), depth for the gate (~4^-L). Bounded error,
never exact — a floor/limit edge, NOT one of the exact frontier reductions.

Run:  python -m act.relu_net
"""
from math import erf

import numpy as np

relu = lambda x: np.maximum(0.0, x)
silu = lambda z: z / (1 + np.exp(-np.clip(z, -30, 30)))
_erf = np.vectorize(erf)
gelu = lambda z: 0.5 * z * (1 + _erf(z / np.sqrt(2)))
mish = lambda z: z * np.tanh(relu(z) + np.log1p(np.exp(-np.abs(z))))   # z*tanh(softplus z), stable
tanh = np.tanh


def pw_linear(phi, N, lo=-8.0, hi=8.0):
    """A single hidden layer of N ReLUs == the piecewise-linear interpolant of phi."""
    k = np.linspace(lo, hi, N + 1)
    return lambda z: np.interp(z, k, phi(k))


def relu_units(phi, N, lo=-8.0, hi=8.0):
    """The N-unit ReLU layer pw_linear stands for, written out: knots t_i = lo + i h, slopes
    s_i = (phi(t_{i+1}) - phi(t_i)) / h. Returns (t, b, w): bias b = phi(t_0), a linear first-knot term
    s_0 (x - t_0) (w[0]) and units ReLU(x - t_i), i = 1..N-1, with output weights s_i - s_{i-1}
    (the second differences)."""
    t = np.linspace(lo, hi, N + 1)
    y = phi(t)
    s = np.diff(y) / np.diff(t)
    return t, float(y[0]), np.r_[s[0], np.diff(s)]


def relu_net(units, x):
    """Evaluate relu_units: b + w_0 (x - t_0) + sum_{i>=1} w_i ReLU(x - t_i)."""
    t, b, w = units
    x = np.asarray(x, float)
    return b + w[0] * (x - t[0]) + sum(w[i] * relu(x - t[i]) for i in range(1, len(w)))


def ungated_error(phi, N, lo=-8.0, hi=8.0):
    x = np.linspace(lo, hi, 20000)
    return float(np.abs(phi(x) - pw_linear(phi, N, lo, hi)(x)).max())


# --- optimal small-K: a true minimax FREE-KNOT fit --------------------------------------
# phi(x) ~ a x + b + sum_{i<=K} c_i ReLU(x - t_i) on [lo, hi]  (K ReLU units + the affine
# skip). For FIXED knots t the best (a, b, c) in sup-norm is a linear program
#   min s  s.t.  |A(t) w - phi(x_j)| <= s  on a dense grid x_j,
# solved exactly (HiGHS). The knots are then optimised globally: differential evolution
# over t in [lo, hi]^K (seeded, deterministic) + a Nelder-Mead polish, and the final error
# is re-measured by the LP on a 6001-point grid. Checked against a brute-force knot grid
# (step 0.5) for K=3: DE is never worse. A knot may land outside the interval / duplicate
# another -- that is a legal (wasted) unit, so the fit is non-increasing in K.
# Default interval [-8, 8], the same one ungated_error / pw_linear use above (and the one
# that reproduces the post's K=1 tanh 0.605); the activation cross-matrix uses [-6, 6].

def _knot_lp(phi, t, x):
    from scipy.optimize import linprog
    A = np.column_stack([x, np.ones_like(x)] + [relu(x - ti) for ti in t])
    y = phi(x)
    n, p = A.shape
    one = np.ones((n, 1))
    r = linprog(np.r_[np.zeros(p), 1.0], A_ub=np.block([[A, -one], [-A, -one]]), b_ub=np.r_[y, -y],
                bounds=[(None, None)] * p + [(0, None)], method="highs")
    return float(r.fun), r.x[:p]


def free_knot_minimax(phi, K, lo=-8.0, hi=8.0, seeds=(0, 1, 2), n_search=401, n_final=6001, x0=None):
    """Best sup-norm fit of phi on [lo, hi] by a x + b + sum_i c_i ReLU(x - t_i), K free knots.
    Global search: differential evolution from each seed (and from x0, e.g. the K-1 optimum
    plus a spare knot), each polished by Nelder-Mead; the best is re-measured on n_final points.
    Returns (max|err|, knots, (a, b, c_1..c_K))."""
    from scipy.optimize import differential_evolution, minimize
    xs = np.linspace(lo, hi, n_search)
    f = lambda t: _knot_lp(phi, np.sort(np.clip(t, lo, hi)), xs)[0]
    starts = [(s, None) for s in seeds] + ([] if x0 is None else [(seeds[0], np.clip(x0, lo, hi))])
    best = (np.inf, None)
    for s, init in starts:
        r = differential_evolution(f, [(lo, hi)] * K, seed=s, popsize=12, maxiter=60, tol=1e-8,
                                   polish=False, x0=init)
        p = minimize(f, r.x, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-9))
        for cand in (r.x, p.x):
            v = f(cand)
            if v < best[0]:
                best = (v, np.sort(np.clip(cand, lo, hi)))
    t = best[1]
    err, w = _knot_lp(phi, t, np.linspace(lo, hi, n_final))
    return err, t, w


def free_knot_table(phi, Kmax=3, lo=-8.0, hi=8.0, seeds=(0, 1, 2)):
    """[(K, err, knots)] for K = 1..Kmax. Each K also searches from the K-1 optimum plus a spare
    knot at the interval edge (a no-op unit), so the reported error is non-increasing in K."""
    out, prev = [], None
    for K in range(1, Kmax + 1):
        x0 = None if prev is None else np.r_[prev, hi]
        err, t, _ = free_knot_minimax(phi, K, lo, hi, seeds=seeds, x0=x0)
        if out and err > out[-1][1]:            # the warm start itself is a valid K-knot fit
            err, t = out[-1][1], np.r_[prev, hi]
        out.append((K, err, t))
        prev = t
    return out


def _tent(x):
    return 2 * relu(x) - 4 * relu(x - 0.5)          # tent map on [0,1], one ReLU layer


def yarotsky_sq01(x, L):
    """x^2 on [0,1] from an L-deep ReLU sawtooth (explicit; no fitting)."""
    f = np.array(x, float)
    g = f.copy()
    for s in range(1, L + 1):
        g = _tent(g)
        f = f - g / (4.0 ** s)
    return f


def square(x, L, M):
    ax = (relu(x) + relu(-x)) / M                    # |x|/M via one ReLU layer
    return (M * M) * yarotsky_sq01(ax, L)


def gated_gate(phi, a, b, N, L, M=12.0):
    """phi(a)*b: piecewise-linear phi (width N) then the product by squaring (depth L)."""
    p = pw_linear(phi, N)(a)
    return 0.25 * (square(p + b, L, M) - square(p - b, L, M))


def square_error(L):
    t = np.linspace(0, 1, 20000)
    return float(np.abs(t ** 2 - yarotsky_sq01(t, L)).max())


def gate_error(phi, N, L, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(-4, 4, 40000), rng.uniform(-4, 4, 40000)
    return float(np.abs(gated_gate(phi, a, b, N, L) - phi(a) * b).max())


def exotic_table(Ns=(2, 4, 8, 16, 32, 64), lo=-6.0, hi=6.0):
    """Uniform N-ReLU interpolation error of situ and the real (HF) xIELU on [lo, hi]."""
    from act.activation_cross import ACTS
    return {name: [(N, ungated_error(ACTS[name], N, lo, hi)) for N in Ns] for name in ("situ", "xielu")}


def main(free_knots=True):
    print("UNGATED  phi <- ONE hidden layer of N ReLUs (piecewise-linear, uniform knots on [-8,8]), max|Δ|:")
    for name, phi in [("SiLU", silu), ("GeLU", gelu), ("mish", mish), ("tanh", tanh)]:
        errs = [ungated_error(phi, N) for N in (16, 64, 256)]
        print(f"  {name:5s} N=16/64/256 -> " + " / ".join(f"{e:.2e}" for e in errs) + "   (~1/N^2)")
    print("EXOTIC  uniform N-ReLU on [-6,6], max|Δ|  (and N^2 * err):")
    for name, rows in exotic_table().items():
        print(f"  {name:5s} " + "  ".join(f"N={N}:{e:.3g}" for N, e in rows)
              + "   N^2*err = " + " ".join(f"{e * N * N:.2f}" for N, e in rows))
    if free_knots:
        print("FREE-KNOT minimax  a x + b + sum_{i<=K} c_i ReLU(x - t_i) on [-8,8], max|Δ|:")
        for name, phi in [("SiLU", silu), ("GeLU", gelu), ("mish", mish), ("tanh", tanh)]:
            tab = free_knot_table(phi, 3)
            print(f"  {name:5s} " + "  ".join(f"K={K}:{e:.4f}" for K, e, _ in tab)
                  + "   knots " + " | ".join(",".join(f"{v:.2f}" for v in t) for _, _, t in tab))
    print("SQUARING x^2 <- Yarotsky sawtooth, DEPTH L (explicit), max|Δ|:")
    for L in (2, 4, 6):
        print(f"  L={L} -> {square_error(L):.1e}   (~4^-L)")
    print("GATED  SwiGLU/GeGLU gate phi(a)*b <- width 256 + product depth L, max|Δ|:")
    for name, phi in [("SwiGLU", silu), ("GeGLU", gelu)]:
        print(f"  {name:6s} L=2/4/6 -> " + " / ".join(f"{gate_error(phi, 256, L):.1e}" for L in (2, 4, 6)))


if __name__ == "__main__":
    main()
