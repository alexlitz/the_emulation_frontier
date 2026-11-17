"""Cross-emulation between activations, analytically (numpy only, no fitting-by-SGD).

To emulate a target Y with a host X you may fold scale/shift into the neighbouring
linear layers — i.e. use  a*X(c*x + d) + e*x + f. The min-max error of that best
single-op fit is the DIRECT floor for X -> Y. Two estimators, both on [-6, 6]:
  floor()          (c, d) on a coarse grid (c in [0.2, 25], |d| <= 2), (a, e, f) by least
                   squares; the reported number is that fit's max error. An UPPER bound,
                   and a loose one in many cells (it misses c = 1.702, so even the exact
                   swigluoai = SiLU(1.702x)/1.702 fold shows 0.006).
  floor_minimax()  the same grid as starts, then (c, d) polished by Nelder-Mead on the
                   sup-norm-optimal (LP) (a, e, f). Much tighter. Cells that read 0.000
                   for ReLU targets are the beta->inf LIMIT (c grows without bound), not
                   exact folds.
The structure that falls out:

  * ReLU is the bottom: every smooth swish emulates it near-exactly (the beta->inf
    limit), but ReLU is a poor single-op host for a curve (floor ~0.1) — it needs a
    NETWORK (act/relu_net.py, ~1/N^2).
  * {SiLU, GeLU, mish, swigluoai} are a mutually-interchangeable cluster (minimax <= 0.02).
  * relu2 / xIELU: xIELU hosts relu2 near-exactly. relu2 hosts xIELU only to 0.14-0.17,
    because the real xIELU's negative branch is nonlinear. A swish host also gets within
    ~0.14 of xIELU. The reverse (xIELU / relu2 hosting a swish curve) stays at 0.3-0.4.

With a NETWORK of ANY non-polynomial activation every pair reduces to every other
(universal approximation), so these single-op floors are the cheap DIRECT edges; the
network route completes the graph at ~1/N^2.

Run:  python -m act.activation_cross
"""
from math import erf

import numpy as np

relu = lambda z: np.maximum(0.0, z)
_erf = np.vectorize(erf)
_sig = lambda z: 1.0 / (1 + np.exp(-np.clip(z, -30, 30)))
_sp = lambda z: np.log1p(np.exp(-np.abs(z))) + relu(z)        # softplus, numerically stable
# xIELU exactly as transformers.activations.XIELUActivation._xielu_python computes it, at the
# module's DEFAULT initialisation (alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
#   alpha_p = softplus(log(expm1(0.8)))              = 0.8
#   alpha_n = beta + softplus(log(expm1(0.8 - 0.5))) = 0.8
#   xIELU(x) = alpha_p x^2 + beta x                          (x > 0)
#            = alpha_n (expm1(min(x, eps)) - x) + beta x     (x <= 0)
# HF stores the raw parameters in bf16 by default, which rounds alpha_p / alpha_n to
# 0.8004 / 0.8009; we use the exact values (HF's module built with dtype=float64 gives them).
# A trained Apertus checkpoint carries its own per-layer alpha_p / alpha_n -- these are the
# init values, not any checkpoint's. The negative branch is NOT linear: it is
# beta x - alpha_n (x + 1 - e^x), which tends to -0.3x - 0.8 (grows like +0.3|x|) as x -> -inf.
# (An earlier version used softplus(0.8)*relu(z)^2 + 0.5z: softplus applied to the init
# VALUE, and a linear negative branch. That made xIELU look like an exact relu^2 fold.)
XIELU_ALPHA_P, XIELU_ALPHA_N, XIELU_BETA, XIELU_EPS = 0.8, 0.8, 0.5, -1e-6


def xielu(z, alpha_p=XIELU_ALPHA_P, alpha_n=XIELU_ALPHA_N, beta=XIELU_BETA, eps=XIELU_EPS):
    z = np.asarray(z, dtype=float)
    neg = alpha_n * (np.expm1(np.minimum(z, eps)) - z) + beta * z
    return np.where(z > 0, alpha_p * z * z + beta * z, neg)


# every activation that actually appears in the top-10K zoo (the act slot). tanh is dropped
# (no model uses it); the exotic three are added -- swigluoai / situ / xIELU -- plus squared ReLU.
ACTS = {
    "ReLU": relu,
    "relu2": lambda z: relu(z) ** 2,                          # squared ReLU (Primer etc.)
    "SiLU": lambda z: z * _sig(z),
    "GeLU": lambda z: 0.5 * z * (1 + _erf(z / np.sqrt(2))),
    # the tanh approximation (gelu_new / gelu_pytorch_tanh: GPT-2, Gemma, Bloom, ...), a separate act value
    "GeLU_tanh": lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))),
    "mish": lambda z: z * np.tanh(_sp(z)),
    "swigluoai": lambda z: z * _sig(1.702 * z),               # gpt-oss / MiniMax-M3 == SiLU(1.702z)/1.702
    "situ": lambda z: 4.0 * np.tanh(z / 4.0) * _sig(z),       # Kimi-K3, trained beta=4 (SiLU is beta->inf)
    "xielu": lambda z: xielu(z),                              # Apertus, the real HF definition (above)
}


def floor(X, Y, lo=-6.0, hi=6.0, n=1500):
    """min-max error of the best single-op  a*X(c*x+d)+e*x+f  ~  Y(x)."""
    x = np.linspace(lo, hi, n)
    Yx = Y(x)
    best = 9.9
    for c in np.r_[np.linspace(0.2, 3, 15), np.linspace(3.5, 25, 12)]:
        for d in np.linspace(-2, 2, 9):
            A = np.stack([X(c * x + d), x, np.ones_like(x)], 1)   # columns weight a, e, f
            coef, *_ = np.linalg.lstsq(A, Yx, rcond=None)
            best = min(best, float(np.abs(A @ coef - Yx).max()))
    return best


def _minimax_aef(Xc, x, Yx, coef=False):
    """sup-norm-optimal (a, e, f) for a*Xc + e*x + f ~ Yx: a 4-variable LP (HiGHS).
    Returns the error, or (error, (a, e, f)) with coef=True."""
    from scipy.optimize import linprog
    if not np.all(np.isfinite(Xc)) or np.abs(Xc).max() > 1e8:
        return (np.inf, None) if coef else np.inf
    A = np.stack([Xc, x, np.ones_like(x)], 1)
    one = np.ones((len(x), 1))
    r = linprog([0, 0, 0, 1.0], A_ub=np.block([[A, -one], [-A, -one]]), b_ub=np.r_[Yx, -Yx],
                bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    err = float(r.fun) if r.status == 0 else np.inf
    return (err, (tuple(float(v) for v in r.x[:3]) if r.status == 0 else None)) if coef else err


def fit_minimax(X, Y, lo=-6.0, hi=6.0, n=1500, starts=6):
    """floor_minimax() together with its minimiser: (floor, (a, c, d, e, f)) such that
    a*X(c*x + d) + e*x + f is the best single-op fit of Y on [lo, hi] (max-error `floor`).
    This is the function a host with activation X computes in place of Y (act/act_swap_agreement.py
    runs it inside real models, through fit_host). X is the bare activation: for swigluoai that is the
    UNCLAMPED gate branch, so a ReLU fit here (c ~ 1767, a ~ 1/c) is the scaled-gate route that the real
    clamped unit refutes; fit_host routes that pair to fit_relu_on_swigluoai instead."""
    from scipy.optimize import minimize
    x = np.linspace(lo, hi, n)
    Yx = Y(x)
    lsq = []
    for c in np.r_[np.linspace(0.2, 3, 15), np.linspace(3.5, 25, 12)]:
        for d in np.linspace(-2, 2, 9):
            A = np.stack([X(c * x + d), x, np.ones_like(x)], 1)
            coef, *_ = np.linalg.lstsq(A, Yx, rcond=None)
            lsq.append((float(np.abs(A @ coef - Yx).max()), c, d))
    lsq.sort()
    f = lambda p: _minimax_aef(X(p[0] * x + p[1]), x, Yx)
    best, arg = np.inf, None
    for _, c, d in lsq[:starts]:
        r = minimize(f, [c, d], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=400))
        for cd in (r.x, (c, d)):
            v = f(cd)
            if v < best:
                best, arg = v, (float(cd[0]), float(cd[1]))
    err, (a, e, f0) = _minimax_aef(X(arg[0] * x + arg[1]), x, Yx, coef=True)
    return err, (a, arg[0], arg[1], e, f0)


# --- ReLU on a swigluoai host: the up-bias route (Lean EP.relu_sub_swigluoai) ---------------------
# The real swigluoai unit (gpt-oss; AlexKoff88/gpt-oss-tinystories has no swiglu_limit in its config, so
# GptOssConfig's default 7.0 applies, and gate_up_proj carries a bias) is
#     unit(gate, up) = (clip(up, -L, L) + 1) * g' sigmoid(1.702 g'),   g' = min(gate, L),  L = 7.
# ACTS["swigluoai"] is the UNCLAMPED gate branch z sigmoid(1.702 z). On it, fit_minimax's ReLU fit is the
# scaled-gate route a X(c x + d) with c ~ 1767, a ~ 1/c (the relu_sub_silu route). On the real unit that
# route is refuted (EP.swigluoai_scaled_gate_vanishes): the gate clamp caps X(c x) at K = L sigmoid(1.702 L),
# so a X(c x) = K / c ~ 0.004 for every x > 0. The valid route uses the up input and an up bias:
#     gate row c z, up row a z with up bias -1, down 1 / (a K);   |a z| <= L - 1 (the up clamp is inactive)
# which is relu(z) EXACTLY for z >= L / c, and a LIMIT (c -> inf) on z < L / c. It needs a host with an
# up bias (gpt-oss has one) and bounded inputs (a is chosen from the input bound R: a = (L - 1) / R).
SWIGLUOAI_LIMIT, SWIGLUOAI_ALPHA = 7.0, 1.702
SWIGLUOAI_K = SWIGLUOAI_LIMIT * float(_sig(SWIGLUOAI_ALPHA * SWIGLUOAI_LIMIT))
RELU_SWIGLUOAI_C = 1e5            # the limit's gate scale: floor = M / c (M ~ 1.75), so ~1.8e-5


def swigluoai_unit(gate, up, limit=SWIGLUOAI_LIMIT, alpha=SWIGLUOAI_ALPHA):
    """The clamped gpt-oss unit (clip(up, -L, L) + 1) g' sigmoid(alpha g'), g' = min(gate, L) (numpy)."""
    g = np.minimum(gate, limit)
    return (np.clip(up, -limit, limit) + 1.0) * g * _sig(alpha * g)


def relu_swigluoai_params(bound=6.0, c=RELU_SWIGLUOAI_C, limit=SWIGLUOAI_LIMIT, alpha=SWIGLUOAI_ALPHA):
    """The relu_sub_swigluoai weights for inputs |z| <= bound: gate c z, up a z + up_bias, down."""
    K = limit * float(_sig(alpha * limit))
    a = (limit - 1.0) / float(bound)
    return dict(route="relu_sub_swigluoai", a=a, c=float(c), up_bias=-1.0, down=1.0 / (a * K),
                limit=float(limit), alpha=float(alpha), bound=float(bound))


def relu_on_swigluoai(z, p):
    """What the swigluoai host unit computes for a ReLU target along relu_sub_swigluoai (numpy)."""
    z = np.asarray(z, dtype=float)
    return swigluoai_unit(p["c"] * z, p["a"] * z + p["up_bias"], p["limit"], p["alpha"]) * p["down"]


def fit_relu_on_swigluoai(lo=-6.0, hi=6.0, c=RELU_SWIGLUOAI_C, limit=SWIGLUOAI_LIMIT, alpha=SWIGLUOAI_ALPHA):
    """(floor, params) of relu_sub_swigluoai on [lo, hi]. The floor is the sup error, taken in t = c z (the
    error lives within |z| <~ 30 / c of the kink, which any fixed z grid misses): for 0 < t < L it is
    (t / c)(1 - t sigmoid(alpha t) / K), for t < 0 it is t^2 sigmoid(alpha t) / (c K), and 0 for t >= L
    (exact there, as long as |a z| <= L - 1, which a = (L - 1) / max(|lo|, |hi|) guarantees)."""
    p = relu_swigluoai_params(max(abs(lo), abs(hi)), c, limit, alpha)
    t = np.r_[np.linspace(max(c * lo, -60.0), 0.0, 200001), np.linspace(0.0, min(c * hi, limit), 200001)]
    z = t / c
    floor = float(np.abs(relu_on_swigluoai(z, p) - relu(z)).max())
    return floor, p


def _is_relu(Y, lo=-6.0, hi=6.0):
    x = np.linspace(lo, hi, 1501)
    return bool(np.allclose(Y(x), relu(x), rtol=0, atol=1e-12))


def fit_host(host, Y, lo=-6.0, hi=6.0, **kw):
    """The fit a host with activation `host` (a name in ACTS) uses for the target Y: (floor, params).
    The single-op fit_minimax, except a ReLU target on a swigluoai host, which takes relu_sub_swigluoai
    (params a dict; the single-op scaled-gate route is refuted on the real, clamped unit)."""
    if host == "swigluoai" and _is_relu(Y, lo, hi):
        return fit_relu_on_swigluoai(lo, hi)
    return fit_minimax(ACTS[host], Y, lo, hi, **kw)


def floor_minimax(X, Y, lo=-6.0, hi=6.0, n=1500, starts=6):
    """The TRUE best single-op fit:  min over (a,c,d,e,f) of max|a X(cx+d) + ex + f - Y|.
    floor()'s (c, d) grid is coarse (c steps ~2 above 3.5, |d| <= 2) and its (a, e, f) are least
    squares, so it over-states many cells. Here: rank the same grid by floor()'s criterion, then
    from the best `starts` points polish (c, d) by Nelder-Mead on the sup-norm-optimal (LP)
    (a, e, f) error. c, d are unconstrained (any value folds into the neighbouring linears)."""
    return fit_minimax(X, Y, lo, hi, n, starts)[0]


def matrix(names=None, method="grid"):
    names = names or list(ACTS)
    fl = floor if method == "grid" else floor_minimax
    return {Y: {X: (0.0 if X == Y else fl(ACTS[X], ACTS[Y])) for X in names} for Y in names}


def main():
    names = list(ACTS)
    for method, title in (("grid", "grid (c,d) + least-squares (a,e,f)  [floor()]"),
                          ("minimax", "true minimax: polished (c,d) + LP (a,e,f)  [floor_minimax()]")):
        M = matrix(names, method)
        print(f"single-op cross-emulation FLOOR, {title}\n  host X (col) -> target Y (row), max|Δ| on [-6,6]:")
        print("target\\host  " + "  ".join(f"{n:>9}" for n in names))
        for Y in names:
            print(f"{Y:>10}  " + "  ".join(("        —" if X == Y else f"{M[Y][X]:9.3f}") for X in names))
        print()


if __name__ == "__main__":
    main()
