"""1D compensators: approximate g(t) = 1/t on [t_lo, t_hi] (t = the pad's normed coordinate,
proportional to 1/r) in sup RELATIVE error |f(t) t - 1|. Approximating r over [r_min, r_max] is
the same problem (t ∝ 1/r), so only D = t_hi / t_lo = r_max / r_min matters.

Families (k = number of hidden units):
  (a) bias-free GLU:   f(t) = t * sum_k c_k silu(a_k t)       (unit k: silu(a_k t) * (b_k t))
  (b) ReLU with biases: f(t) = c_0 + sum_k c_k relu(t - b_k)  (k = 1 is affine)
For fixed nonlinear parameters (a_k or knots b_k) the minimax over c is a linear program on a
dense grid (solved exactly); the nonlinear parameters are searched (geometric grids, Nelder-Mead).
The result is an UPPER bound on the best achievable error of the family.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import expit

GRID = 4000      # final evaluation grid
SEARCH = 300     # grid used while searching the nonlinear parameters


def silu(x):
    return x * expit(x)


def minimax_lp(Phi, t):
    """min_c max_i |t_i * (Phi c)_i - 1| ; returns (err, c). Columns are rescaled to unit max for
    conditioning; falls back across HiGHS methods; (inf, None) if every method fails."""
    A = Phi * t[:, None]
    sc = np.abs(A).max(0); sc[sc == 0] = 1
    A = A / sc
    n, k = A.shape
    obj = np.zeros(k + 1); obj[-1] = 1
    Aub = np.vstack([np.hstack([A, -np.ones((n, 1))]), np.hstack([-A, -np.ones((n, 1))])])
    bub = np.concatenate([np.ones(n), -np.ones(n)])
    for meth in ("highs", "highs-ipm", "highs-ds"):
        res = linprog(obj, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * k + [(0, None)], method=meth,
                      options={"time_limit": 20.0})
        if res.status == 0:
            return res.x[-1], res.x[:k] / sc
    return np.inf, None


def grid(D, n=GRID):
    return np.geomspace(1.0 / D, 1.0, n)            # t scaled so t_hi = 1


def glu_basis(t, a):
    return t[:, None] * silu(np.outer(t, a))


def relu_basis(t, knots):
    return np.hstack([np.ones((len(t), 1)), np.maximum(t[:, None] - knots[None], 0)])


def fit_glu(D, k):
    t = grid(D, SEARCH)

    def err(p):
        lo, hi = p
        a = np.sign(hi) * np.geomspace(np.exp(lo), np.exp(lo) * np.exp(abs(hi)), k) if k > 1 \
            else np.array([np.exp(lo)])
        return minimax_lp(glu_basis(t, a), t)[0], a

    best = (np.inf, None, None)
    for lo0, hi0 in ((-1.0, 3.0), (1.0, 3.0), (1.0, 6.0), (2.5, -3.0)):
        r = minimize(lambda p: err(p)[0], [lo0, hi0], method="Nelder-Mead",
                     options=dict(maxiter=80, xatol=1e-3, fatol=1e-14))
        e, a = err(r.x)
        if e < best[0]:
            best = (e, a)
    tf = grid(D)
    e, c = minimax_lp(glu_basis(tf, best[1]), tf)
    if c is None:                                   # fine-grid LP failed: keep the search fit
        e, c = minimax_lp(glu_basis(t, best[1]), t)
    return e, best[1], c


def fit_relu(D, k):
    t = grid(D, SEARCH)
    if k == 1:
        knots = np.array([t[0] - 1])
        e, c = minimax_lp(relu_basis(t, knots), t)
        return e, knots, c

    def knots_of(p):                 # affine knot below the range + k-1 interior knots
        z = np.cumsum(np.exp(p)) / np.exp(p).sum()
        return np.concatenate([[t[0] - 1], t[0] * (1 / t[0]) ** z[:-1]])

    def err(p):
        kn = knots_of(p)[:k]
        return minimax_lp(relu_basis(t, kn), t)[0]

    p0 = np.zeros(k)                                     # geometric spacing
    best_p, best_e = p0, err(p0)
    if k <= 8:
        r = minimize(err, p0, method="Nelder-Mead", options=dict(maxiter=60 * k, fatol=1e-12))
        if r.fun < best_e:
            best_p, best_e = r.x, r.fun
    kn = knots_of(best_p)[:k]
    tf = grid(D)
    kn[0] = tf[0] - 1
    e, c = minimax_lp(relu_basis(tf, kn), tf)
    return e, kn, c


def const_err(D):
    """Best fixed scale (k = 0): (D - 1) / (D + 1)."""
    return (D - 1) / (D + 1)


class Comp:
    """A fitted compensator on [t_lo, t_hi] in original units: f(t) ≈ 1/t."""

    def __init__(self, fam, k, t_lo, t_hi):
        self.fam, self.t_hi = fam, t_hi
        D = t_hi / t_lo
        if fam == "glu":
            self.err, self.a, self.c = fit_glu(D, k)
        else:
            self.err, self.kn, self.c = fit_relu(D, k)

    def __call__(self, t):                              # t: np array, original units
        shape = np.shape(t)
        s = np.asarray(t, dtype=np.float64).ravel() / self.t_hi
        if self.fam == "glu":
            f = glu_basis(s, self.a) @ self.c
        else:
            f = relu_basis(s, self.kn) @ self.c
        return (f / self.t_hi).reshape(shape)            # 1/t scales as 1/t_hi


def verify(comp, t_lo, t_hi, dtype=np.float64, n=20001):
    """Independent check of a fitted compensator: sup |f(t) t - 1| on a finer grid (not the LP
    grid), evaluated in `dtype` (float32 exposes cancellation between large coefficients)."""
    t = np.geomspace(t_lo, t_hi, n).astype(dtype)
    s = (t / dtype(comp.t_hi)).astype(dtype)
    c = comp.c.astype(dtype)
    if comp.fam == "glu":
        a = comp.a.astype(dtype)
        z = np.outer(s, a)
        B = s[:, None] * (z / (dtype(1) + np.exp(-z)))
    else:
        kn = comp.kn.astype(dtype)
        B = np.hstack([np.ones((len(s), 1), dtype), np.maximum(s[:, None] - kn[None], dtype(0))])
    f = (B @ c) / dtype(comp.t_hi)
    return float(np.max(np.abs(f.astype(np.float64) * t.astype(np.float64) - 1)))
