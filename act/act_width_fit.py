"""k-unit activation fits: one target FFN unit rebuilt from k host units.

A target unit computes  Y(a) * b  (GLU; ungated: Y(a)). A host whose FFN is k times as wide can
spend k units on it:  sum_i alpha_i X(beta_i a + gamma_i) * b  -- gate rows i scaled by beta_i
(and shifted by gamma_i), up rows copied, down-projection columns scaled by alpha_i. That is exact
weight-setting; the only error is the k-term fit of Y by X, measured here as a minimax error on
[-R, R] with R the pre-activation range real models reach (act/data/act_swap_agreement.jsonl
"pre.max_abs": up to 72 for Qwen2.5-1.5B, 109 for Pythia, 151 for nanochat).

Two families:
  shift    gamma_i free -- needs a gate bias in the host (mlp_bias / a biased c_fc). A constant
           residual channel would do too, but under RMSNorm none exists
           (prenorm_readout_obstruction, proofs/PROOFS.md).
  noshift  gamma_i = 0:  sum_i alpha_i X(beta_i a). Any host can do this.
There is NO free linear term e*a and no constant f (unlike activation_cross.fit_minimax, which
folds a*X(cx+d)+ex+f): a linear term costs host units like any other (for a swish host it is 2
units exactly, X(a) - X(-a) = a), a constant needs a shift (beta_i = 0).

The fit (fit_k): candidate starts -- the (k-1)-term optimum plus each of the best dictionary atoms
(greedy), the noshift optimum (for shift), random starts -- each polished by SLSQP on the epigraph
form (min t s.t. |fit - Y| <= t on a grid), alpha re-solved by the minimax LP on the full grid of
[-R, R] (fine uniform core [-12, 12], geometric tails) and re-checked on a 4x denser grid. It is a
local search, so every number is an UPPER bound on the true k-term minimax error (k_min is an upper
bound too); the curves are forced monotone in k.

The table run stops a curve early once it is hopeless (_hopeless: > 0.2 at k >= 2, > 0.02 at k >= 4);
the cells near the swish cluster were then recomputed to k = 8 without the stop (--no-give-up) and
the borderline k re-searched with 40 - 150 random starts (--refine), as stored.

Run:  python -m act.act_width_fit [--R 6 160] [--kmax 8] [--jobs 16]   (writes act/data/act_width_fits.json)
      python -m act.act_width_fit --no-give-up 'SiLU|mish|160|noshift' --refine 'GeLU|SiLU|160|noshift:5'
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

SQ2 = math.sqrt(2.0)


def _xielu(z, alpha_p=0.8, alpha_n=0.8, beta=0.5, eps=-1e-6):
    return torch.where(z > 0, alpha_p * z * z + beta * z,
                       alpha_n * (torch.expm1(torch.clamp(z, max=eps)) - z) + beta * z)


# torch versions of act/activation_cross.ACTS (same definitions; xIELU at its default init), plus the
# tanh GeLU (GPT-2's gelu_new, Gemma's gelu_pytorch_tanh) as its own target row
TA = {
    "ReLU": torch.relu,
    "relu2": lambda z: torch.relu(z) ** 2,
    "SiLU": torch.nn.functional.silu,
    "GeLU": lambda z: 0.5 * z * (1 + torch.erf(z / SQ2)),
    "gelu_tanh": lambda z: torch.nn.functional.gelu(z, approximate="tanh"),
    "mish": torch.nn.functional.mish,
    "swigluoai": lambda z: z * torch.sigmoid(1.702 * z),
    "situ": lambda z: 4.0 * torch.tanh(z / 4.0) * torch.sigmoid(z),
    "xielu": _xielu,
}
HOSTS = ("ReLU", "relu2", "SiLU", "GeLU", "mish", "swigluoai", "situ", "xielu")
TARGETS = ("ReLU", "relu2", "SiLU", "GeLU", "gelu_tanh", "mish", "swigluoai", "situ", "xielu")


def grid(R, n_core=3001, core=12.0, n_tail=600, dense=1):
    """Fine uniform core [-c, c] (c = min(core, R)), geometric tails out to R (float64)."""
    c = min(core, R)
    x = np.linspace(-c, c, (n_core - 1) * dense + 1)
    if R > c:
        t = np.geomspace(c, R, n_tail * dense)[1:]
        x = np.r_[-t[::-1], x, t]
    return torch.tensor(x, dtype=torch.float64)


def units(X, x, beta, gamma):
    """[len(x), k]: the k host units X(beta_i x + gamma_i)."""
    b, g = torch.as_tensor(beta, dtype=torch.float64), torch.as_tensor(gamma, dtype=torch.float64)
    return X(x[:, None] * b[None, :] + g[None, :])


def lp_alpha(F, y):
    """minimax alpha for F @ alpha ~ y (columns of F = the k host units): (err, alpha)."""
    from scipy.optimize import linprog
    F, y = np.asarray(F), np.asarray(y)
    if not np.all(np.isfinite(F)) or np.abs(F).max() > 1e12:
        return np.inf, None
    s = np.abs(F).max(0)
    s[s == 0] = 1
    k, one = F.shape[1], np.ones((len(y), 1))
    r = linprog(np.r_[np.zeros(k), 1.0], A_ub=np.block([[F / s, -one], [-F / s, -one]]), b_ub=np.r_[y, -y],
                bounds=[(None, None)] * k + [(0, None)], method="highs")
    if r.status != 0:
        return np.inf, None
    return float(r.fun), r.x[:k] / s


def _vals_grads(X, Z):
    Z = Z.clone().requires_grad_(True)
    v = X(Z)
    g, = torch.autograd.grad(v.sum(), Z)
    return v.detach().numpy(), g.numpy()


def polish(X, x, y, alpha, beta, gamma, shift, iters=300):
    """Local minimax polish: SLSQP on  min t  s.t.  |sum_i alpha_i X(beta_i x + gamma_i) - y| <= t."""
    from scipy.optimize import minimize
    k, xn, yn = len(beta), x.numpy(), y.numpy()

    def unpack(v):
        return v[:k], v[k:2 * k], (v[2 * k:3 * k] if shift else np.zeros(k))

    def resid(v):
        a, b, g = unpack(v)
        V, D = _vals_grads(X, torch.tensor(np.outer(xn, b) + g))
        J = [V, D * xn[:, None] * a] + ([D * a] if shift else [])
        return V @ a - yn, np.concatenate(J, 1)

    v0 = np.r_[alpha, beta, gamma if shift else []]
    r0, _ = resid(v0)
    n, one = len(v0), np.ones((len(xn), 1))

    def cons(w):
        return np.r_[w[-1] - resid(w[:-1])[0], w[-1] + resid(w[:-1])[0]]

    def cjac(w):
        J = resid(w[:-1])[1]
        return np.block([[-J, one], [J, one]])
    with np.errstate(all="ignore"):
        res = minimize(lambda w: w[-1], np.r_[v0, np.abs(r0).max()], jac=lambda w: np.r_[np.zeros(n), 1.0],
                       constraints=[dict(type="ineq", fun=cons, jac=cjac)], method="SLSQP",
                       options=dict(maxiter=iters, ftol=1e-15))
    _, b, g = unpack(res.x[:-1])
    return np.asarray(b, float), np.asarray(g, float)


def _dictionary(shift):
    bs = np.geomspace(0.05, 30, 40)
    bs = np.r_[-bs[::-1], bs]
    if not shift:
        return [(b, 0.0) for b in bs]
    return [(b, g) for b in bs for g in np.linspace(-6, 6, 13)] + [(0.0, 1.0)]


def evaluate(X, Y, R, p, dense=4):
    """Max |fit - Y| of params p on a `dense`x grid of [-R, R]."""
    x = grid(R, dense=dense)
    return float((units(X, x, p["beta"], p["gamma"]) @ torch.tensor(p["alpha"], dtype=torch.float64) - Y(x)).abs().max())


def fit_k(X, Y, k, R, shift, warm=None, extra=(), n_random=10, n_greedy=6, seed=0):
    """Best k-term fit sum alpha_i X(beta_i a + gamma_i) of Y on [-R, R] found by the local search
    above: (err, dict(alpha, beta, gamma)). warm: the (k-1)-term params; extra: more k-term starts."""
    xs = grid(R, n_core=601, n_tail=120)
    ys = Y(xs)
    xf, yf = grid(R), Y(grid(R))
    rng = np.random.default_rng(seed + 1000 * k + (7 if shift else 0))
    starts = [(np.asarray(p["beta"], float), np.asarray(p["gamma"], float)) for p in extra]
    if warm is not None:                      # greedy: the (k-1) optimum + the best new atom
        B0, G0 = np.asarray(warm["beta"], float), np.asarray(warm["gamma"], float)
        F0 = units(X, xs, B0, G0)
        D = _dictionary(shift)
        sc = []
        for b, g in D:
            F = torch.cat([F0, units(X, xs, [b], [g])], 1)
            if not torch.isfinite(F).all():
                sc.append(np.inf)
                continue
            a = torch.linalg.lstsq(F, ys[:, None]).solution[:, 0]
            sc.append(float((F @ a - ys).abs().max()))
        for j in np.argsort(sc)[:n_greedy]:
            starts.append((np.r_[B0, D[j][0]], np.r_[G0, D[j][1]]))
    for _ in range(n_random):
        starts.append((rng.choice([-1, 1], k) * np.exp(rng.uniform(-1.5, 2.0, k)),
                       rng.uniform(-3, 3, k) if shift else np.zeros(k)))
    best = (np.inf, None)
    for b0, g0 in starts:
        e0, a0 = lp_alpha(units(X, xs, b0, g0), ys)
        if a0 is None:
            continue
        try:
            b, g = polish(X, xs, ys, a0, b0, g0, shift)
        except Exception:
            b, g = b0, g0
        for bb, gg in ((b, g), (b0, g0)):
            if not shift:
                gg = np.zeros(k)
            err, a = lp_alpha(units(X, xf, bb, gg), yf)
            if a is not None and err < best[0]:
                best = (err, dict(alpha=a.tolist(), beta=list(map(float, bb)), gamma=list(map(float, gg))))
    if best[1] is None:
        return np.inf, None
    return max(best[0], evaluate(X, Y, R, best[1])), best[1]


def k_curve(host, target, R, shift, kmax=8, Y=None, extra=None, give_up=None):
    """[(k, err, params)] for k = 1..kmax, monotone (each k warm-starts from k-1; a k that does worse
    keeps the (k-1) fit plus a zero unit). extra: {k: [params]} more starts (the noshift optima).
    give_up(k, err) -> True stops early (the remaining k are not computed)."""
    X, Y = TA[host], (Y or TA[target])
    out, warm = [], None
    for k in range(1, kmax + 1):
        err, p = fit_k(X, Y, k, R, shift, warm=warm, extra=(extra or {}).get(k, ()))
        if out and out[-1][2] is not None and not err <= out[-1][1]:
            pe = out[-1][2]
            p = dict(alpha=pe["alpha"] + [0.0], beta=pe["beta"] + [1.0], gamma=pe["gamma"] + [0.0])
            err = out[-1][1]
        out.append((k, err, p))
        warm = p
        if give_up and give_up(k, err):
            break
    # backward pass: the (k+1)-term optimum with one unit dropped, polished, as a k-term start (fixes a
    # forward k that got stuck in a worse basin than the one k+1 found)
    for i in range(len(out) - 2, -1, -1):
        k, e, p = out[i]
        q = out[i + 1][2]
        if q is None:
            continue
        drops = [dict(alpha=[a for j, a in enumerate(q["alpha"]) if j != d],
                      beta=[b for j, b in enumerate(q["beta"]) if j != d],
                      gamma=[g for j, g in enumerate(q["gamma"]) if j != d]) for d in range(k + 1)]
        e2, p2 = fit_k(X, Y, k, R, shift, extra=drops, n_random=0, n_greedy=0)
        if e2 < e:
            out[i] = (k, e2, p2)
    for i in range(1, len(out)):                # monotone again
        if out[i][1] > out[i - 1][1]:
            k, e, p = out[i - 1]
            out[i] = (i + 1, e, dict(alpha=p["alpha"] + [0.0], beta=p["beta"] + [1.0], gamma=p["gamma"] + [0.0]))
    return out


def k_min(curve, thr):
    """Smallest k with error <= thr in a k_curve / stored [[k, err, p]] list, else None."""
    for k, e, _ in curve:
        if e <= thr:
            return k
    return None


ACT_FLOOR_MAX = 0.0005


def _hopeless(k, err):
    # the curves fall by ~2-3x per added unit at best; past these errors 8 units cannot reach 5e-4
    return (k >= 2 and err > 0.2) or (k >= 4 and err > 0.02)


def _job(args):
    t, h, R, kmax = args
    torch.set_num_threads(1)
    res = {}
    no = k_curve(h, t, R, False, kmax, give_up=_hopeless)
    res[f"{t}|{h}|{R:g}|noshift"] = [[k, e, p] for k, e, p in no]
    sh = k_curve(h, t, R, True, kmax, extra={k: [p] for k, _, p in no if p}, give_up=_hopeless)
    res[f"{t}|{h}|{R:g}|shift"] = [[k, e, p] for k, e, p in sh]
    return res


def refine(curve, host, target, R, shift, k, n_random=40, seed=1):
    """More starts for one k of a stored curve ([[k, err, p], ...]): the stored k fit, the stored (k+1)
    fit with each unit dropped, the (k-1) fit greedily extended, n_random random starts. Returns the
    updated curve (monotone)."""
    X, Y = TA[host], TA[target]
    extra = [curve[k - 1][2]] if curve[k - 1][2] else []
    if len(curve) > k and curve[k][2]:
        q = curve[k][2]
        extra += [dict(alpha=[a for j, a in enumerate(q["alpha"]) if j != d],
                       beta=[b for j, b in enumerate(q["beta"]) if j != d],
                       gamma=[g for j, g in enumerate(q["gamma"]) if j != d]) for d in range(k + 1)]
    warm = curve[k - 2][2] if k >= 2 else None
    e, p = fit_k(X, Y, k, R, shift, warm=warm, extra=extra, n_random=n_random, seed=seed)
    curve = [list(c) for c in curve]
    if e < curve[k - 1][1]:
        curve[k - 1] = [k, e, p]
        for i in range(k, len(curve)):                 # later k never worse than this one
            if curve[i][1] > e:
                curve[i] = [i + 1, e, dict(alpha=p["alpha"] + [0.0] * (i + 1 - k), beta=p["beta"] + [1.0] * (i + 1 - k),
                                           gamma=p["gamma"] + [0.0] * (i + 1 - k))]
    return curve


def _monotone(curve):
    """[[k, err, p], ...] with every k at least as good as k-1 (pad the (k-1) fit with a zero unit)."""
    out = [list(c) for c in curve]
    for i in range(1, len(out)):
        if out[i - 1][2] is not None and not out[i][1] <= out[i - 1][1]:
            p = out[i - 1][2]
            out[i] = [i + 1, out[i - 1][1], dict(alpha=p["alpha"] + [0.0], beta=p["beta"] + [1.0],
                                                 gamma=p["gamma"] + [0.0])]
    return out


def _refine_job(args):
    key, curve, k, n_random = args
    torch.set_num_threads(1)
    t, h, R, fam = key.split("|")
    return key, k, refine(curve, h, t, float(R), fam == "shift", k, n_random)


def _full_job(key):
    torch.set_num_threads(1)
    t, h, R, fam = key.split("|")
    return key, [[k, e, p] for k, e, p in k_curve(h, t, float(R), fam == "shift", 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, nargs="+", default=[6.0, 160.0])
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--hosts", nargs="+", default=list(HOSTS))
    ap.add_argument("--out", default="act/data/act_width_fits.json")
    ap.add_argument("--refine", nargs="+", help="KEY:K cells to re-search with more starts, e.g. 'GeLU|SiLU|160|noshift:5'")
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--no-give-up", nargs="+", default=[], help="KEYs to recompute without the early stop")
    a = ap.parse_args()
    if a.refine or a.no_give_up:
        from multiprocessing import get_context
        res = json.load(open(a.out))
        jobs = [(c.rsplit(":", 1)[0], res[c.rsplit(":", 1)[0]], int(c.rsplit(":", 1)[1]), a.n_random) for c in a.refine or ()]
        with get_context("spawn").Pool(a.jobs) as pool:
            outs = list(pool.imap_unordered(_full_job, a.no_give_up)) if a.no_give_up else []
            outs2 = list(pool.imap_unordered(_refine_job, jobs)) if jobs else []
        res = json.load(open(a.out))
        for key, c in outs:
            old = res.get(key, [])
            res[key] = _monotone([min(x, y, key=lambda v: v[1]) for x, y in zip(old, c)] + c[len(old):])
            print(f"{key:36s} " + " ".join(f"{e:.2g}" for _, e, _ in c), file=sys.stderr, flush=True)
        for key, k, c in sorted(outs2, key=lambda o: o[1]):       # apply in k order (each only lowers)
            old = res[key]
            res[key] = _monotone([min(x, y, key=lambda v: v[1]) for x, y in zip(old, c)] + old[len(c):])
            print(f"{key}:{k}  {old[k - 1][1]:.3g} -> {res[key][k - 1][1]:.3g}", file=sys.stderr, flush=True)
        json.dump(res, open(a.out, "w"))
        return
    try:
        res = json.load(open(a.out))
    except Exception:
        res = {}
    todo = [(t, h, R, a.kmax) for R in a.R for t in a.targets for h in a.hosts
            if h != t and f"{t}|{h}|{R:g}|shift" not in res]
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.jobs) as pool:
        for r in pool.imap_unordered(_job, todo):
            res.update(r)
            for key, c in r.items():
                print(f"{key:36s} " + " ".join(f"{e:.2g}" for _, e, _ in c), file=sys.stderr, flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
