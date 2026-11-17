"""Synthetic float64 check of the exact pre-norm top-1 MoE construction (constant-norm select).

Target: x -> x + sum_j relu(p[e*, j] . N_t(x)) v[e*, j], N_t = RMSNorm(eps_t, w), e* = argmax of
z = W_r N_t(x) (unit gate, ReLU experts).
Host residual [x; pad; g] (g: E(E-1) ordered-pair channels), ONE RMSNorm over all m channels,
weight [sqrt(d/m) w; 1; 1..], eps_h = (d eps_t - S) / m, S = E(E-1)/2 * dc^2, pad = sqrt(S).
  select FFN (ReLU, biases): g[e,e'] = relu(L(z_e' - z_e)) - relu(L(z_e' - z_e) - dc), pad -= sqrt(S)
  expert FFN: relu(p[e,j] . u_x - lam * sum_e' u_g[e,e']) units, plus the same clamps (recomputed
  from its own, exact, read) writing -g, and pad += sqrt(S).
On a STRONG margin (all pairwise router gaps >= dr, L dr >= dc) |g|^2 = S exactly at the expert's
read, so both reads see the target's norm. Also: the naive carry (s_e = sum relu(z_e' - z_e), no
pad, eps_h = eps_t d / m) and the construction on a top-only margin.
Run: PYENV_VERSION=3.12.6 python verify/prenorm_rescue/moe_exact_check.py
"""
import json
import os

import numpy as np

rng = np.random.default_rng(0)
d, E, ew, eps_t = 16, 4, 8, 1e-5
B = 3.0                                    # |x| <= B
dr = 0.05                                  # router margin
w = rng.uniform(0.5, 1.5, d)
Wr = rng.normal(size=(E, d))
p = rng.normal(size=(E, ew, d)) / np.sqrt(d)
v = rng.normal(size=(E, ew, d))
relu = lambda t: np.maximum(t, 0.0)
pairs = [(e, f) for e in range(E) for f in range(E) if e != f]


def rms(h, eps, W):
    return W * h / np.sqrt(np.mean(h ** 2) + eps)


def target(x):
    u = rms(x, eps_t, w)
    es = int(np.argmax(Wr @ u))
    return x + relu(p[es] @ u) @ v[es], es


def sample(strong, n):
    xs = []
    while len(xs) < n:
        x = rng.normal(size=d)
        x *= rng.uniform(0.02, B) / np.linalg.norm(x)
        z = Wr @ rms(x, eps_t, w)
        zs = np.sort(z)
        gaps = np.diff(zs)
        if (strong and gaps.min() >= dr) or (not strong and gaps[-1] >= dr):
            xs.append(x)
    return xs


def exact_host(x, dc=3e-3):
    S = len(pairs) / 2 * dc ** 2
    m = d + 1 + len(pairs)
    eps_h = (d * eps_t - S) / m
    assert eps_h >= 0
    W = np.concatenate([np.sqrt(d / m) * w, np.ones(1 + len(pairs))])
    L = dc / dr
    X, P, G = slice(0, d), d, slice(d + 1, m)
    h = np.concatenate([x, [np.sqrt(S)], np.zeros(len(pairs))])
    clamps = lambda z: np.array([relu(L * (z[f] - z[e])) - relu(L * (z[f] - z[e]) - dc) for e, f in pairs])
    # select FFN
    u = rms(h, eps_h, W)
    h = h.copy(); h[G] += clamps(Wr @ u[X]); h[P] -= np.sqrt(S)
    # expert FFN
    u = rms(h, eps_h, W)
    R = np.sqrt((B ** 2 + S) / m + eps_h)                     # largest host rms on |x| <= B
    pmax = np.abs(p).sum(-1).max() * np.abs(u[X]).max() + 1   # bound on |p . u_x| (loose)
    lam = pmax * R / dc
    mask = np.zeros(E)
    for k, (e, f) in enumerate(pairs):
        mask[e] += u[G][k]
    y = sum(relu(p[e] @ u[X] - lam * mask[e]) @ v[e] for e in range(E))
    out = h.copy(); out[X] += y; out[G] -= clamps(Wr @ u[X]); out[P] += np.sqrt(S)
    return out, m


def naive_host(x):
    m = d + E
    eps_h = eps_t * d / m
    W = np.concatenate([np.sqrt(d / m) * w, np.ones(E)])
    h = np.concatenate([x, np.zeros(E)])
    u = rms(h, eps_h, W)
    z = Wr @ u[:d]
    s = np.array([relu(z - z[e]).sum() for e in range(E)])
    h = h.copy(); h[d:] += s
    u = rms(h, eps_h, W)
    R = np.sqrt((B ** 2 + np.sum(s ** 2)) / m + eps_h)
    lam = (np.abs(p).sum(-1).max() * np.abs(u[:d]).max() + 1) * R / dr
    y = sum(relu(p[e] @ u[:d] - lam * u[d + e]) @ v[e] for e in range(E))
    out = h.copy(); out[:d] += y; out[d:] -= s
    return out


def run(xs, host):
    err = rel = extra = 0.0
    for x in xs:
        t, _ = target(x)
        o = host(x)
        o, m = o if isinstance(o, tuple) else (o, None)
        err = max(err, np.abs(o[:d] - t).max())
        rel = max(rel, np.abs(o[:d] - t).max() / np.abs(t - x).max())
        if m is not None:
            ref = np.zeros(m - d); ref[0] = np.sqrt(len(pairs) / 2 * 3e-3 ** 2)
            extra = max(extra, np.abs(o[d:] - ref).max())
    return dict(max_abs_err=err, max_rel_err_of_update=rel, max_extra_channel_err=extra)


if __name__ == "__main__":
    strong, top = sample(True, 10000), sample(False, 3000)
    res = {"setup": dict(d=d, E=E, ew=ew, eps_t=eps_t, B=B, dr=dr, dc=3e-3, n_strong=len(strong),
                         n_top_only=len(top), dtype="float64"),
           "exact_strong_margin": run(strong, exact_host),
           "naive_carry_strong_margin": run(strong, naive_host),
           "exact_top_only_margin": run(top, exact_host)}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                     "moe_exact_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
