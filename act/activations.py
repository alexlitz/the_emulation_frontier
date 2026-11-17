"""Activation-function subset graph.

Same three edge types as everywhere else:

  MAGNITUDE-LIMIT (like ALiBi<=RoPE):  ReLU is the beta->inf limit of MANY smooth
      families -- Swish_b, Softplus_b, GeLU_b. The hard op is what every smooth
      gate converges to; the currency is magnitude beta.

  EXACT REPARAM (free / fold):  affine identities between activations.
      tanh(x) = 2*sigmoid(2x) - 1
      LeakyReLU_a(x) = ReLU(x) - a*ReLU(-x)      (2 ReLU units)
      |x| = ReLU(x) + ReLU(-x) ;  ReLU(x) = (x + |x|)/2
      ReLU = LeakyReLU_0 = ELU_0 = PReLU_0        (parameter restriction)

  UNIVERSAL PWL SUPERSET:  Maxout_k(x) = max_i (w_i x + b_i). Any max of k affine
      pieces -- ReLU, LeakyReLU, |x|, identity -- is an EXACT Maxout_2. Currency:
      pieces k (= width). A ReLU network and a Maxout network are the same class
      (both continuous piecewise-linear).

  CROSS-FAMILY APPROX:  GeLU ~= Swish_b at b ~= 1.7 (classic value 1.702; a
      max-abs fit gives ~1.77) -- one activation almost a special case of
      another's parameterized family.

  WIDTH APPROX:  any smooth activation <= a ReLU network; error decays with the
      number of knots (a PWL interpolant IS a ReLU net of that width).

Part 6 swaps gpt2's GeLU at runtime for these and measures the drift.

Run:
  python -m act.activations              # the constructions (synthetic)
  python -m act.activations --gpt2       # + real gpt2 activation swap
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

import harness as H

SQRT2 = math.sqrt(2.0)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / SQRT2))


def swish_b(x, b):
    return x * torch.sigmoid(b * x)


def softplus_b(x, b):
    return F.softplus(b * x) / b


def gelu_b(x, b):
    return x * 0.5 * (1.0 + torch.erf(b * x / SQRT2))


# --- 1. ReLU as the beta->inf limit of several smooth families ----------------

def limit_families():
    x = torch.linspace(-6, 6, 6001, dtype=torch.float64)
    relu = F.relu(x)
    fams = {"Swish_b": swish_b, "Softplus_b": softplus_b, "GeLU_b": gelu_b}
    print("1. ReLU = lim_{beta->inf} of smooth families   max|family_b - ReLU|:")
    print(f"     {'beta':>6}  " + "  ".join(f"{k:>12}" for k in fams))
    for b in [1, 4, 16, 64, 256, 1024]:
        errs = [f"{(fn(x, b) - relu).abs().max().item():12.3e}" for fn in fams.values()]
        print(f"     {b:>6}  " + "  ".join(errs))
    print("   -> the hard op is the common magnitude-limit of every smooth gate\n")


# --- 2. exact affine-reparam identities --------------------------------------

def exact_reparam():
    x = torch.linspace(-8, 8, 8001, dtype=torch.float64)
    checks = {
        "tanh = 2*sigmoid(2x)-1": (torch.tanh(x), 2 * torch.sigmoid(2 * x) - 1),
        "|x| = ReLU(x)+ReLU(-x)": (x.abs(), F.relu(x) + F.relu(-x)),
        "ReLU = (x+|x|)/2": (F.relu(x), (x + x.abs()) / 2),
        "LeakyReLU_.1 = ReLU(x)-.1 ReLU(-x)": (F.leaky_relu(x, 0.1), F.relu(x) - 0.1 * F.relu(-x)),
        "ReLU = LeakyReLU_0": (F.relu(x), F.leaky_relu(x, 0.0)),
        "ReLU = ELU_0": (F.relu(x), F.elu(x, 0.0)),
    }
    print("2. exact affine-reparam edges (free / fold):")
    for name, (a, b) in checks.items():
        print(f"     {name:<38} max err {(a - b).abs().max().item():.3e}")
    print()


# --- 3. Maxout as the universal piecewise-linear superset --------------------

def maxout(x, pieces):  # pieces: list of (w, b)
    stack = torch.stack([w * x + b for (w, b) in pieces], dim=0)
    return stack.max(dim=0).values


def maxout_universal():
    x = torch.linspace(-8, 8, 8001, dtype=torch.float64)
    targets = {
        "ReLU": (F.relu(x), [(0.0, 0.0), (1.0, 0.0)]),
        "LeakyReLU_.2": (F.leaky_relu(x, 0.2), [(0.2, 0.0), (1.0, 0.0)]),
        "|x|": (x.abs(), [(-1.0, 0.0), (1.0, 0.0)]),
        "identity": (x, [(1.0, 0.0)]),
        "hard 3-piece": (
            torch.maximum(torch.maximum(-x - 1, 0.5 * x), 2 * x - 1),
            [(-1.0, -1.0), (0.5, 0.0), (2.0, -1.0)],
        ),
    }
    print("3. Maxout_k is the exact superset of any max-of-k affine pieces:")
    for name, (tgt, pieces) in targets.items():
        err = (tgt - maxout(x, pieces)).abs().max().item()
        print(f"     {name:<16} as Maxout_{len(pieces)}   max err {err:.3e}")
    print("   -> ReLU/LeakyReLU/|x|/identity all subset Maxout; currency is pieces\n")


# --- 4. GeLU ~= Swish_b cross-family fit -------------------------------------

def _golden_min(f, lo, hi, it=200):
    gr = (math.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(it):
        if f(c) < f(d):
            hi = d
        else:
            lo = c
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    return (lo + hi) / 2


def gelu_silu_floor(L=15.0, n=600001):
    """Precise fp64 floor of the SiLU->GeLU emulation. A SiLU model reaches the
    family Swish_b(x)=SiLU(b x)/b by folding b into adjacent weights, so the best
    it can do is min_b max_x |GeLU(x) - Swish_b(x)|. This does NOT go to zero -- it
    is the one 'approx-with-a-floor' edge (no currency drives it to exact)."""
    x = torch.linspace(-L, L, n, dtype=torch.float64)
    g = gelu(x)
    maxerr = lambda b: (g - swish_b(x, b)).abs().max().item()
    b = _golden_min(maxerr, 1.2, 2.4)
    floor = maxerr(b)
    resid = (g - swish_b(x, b)).abs()
    xm = x[resid.argmax()].item()
    rms = (resid.pow(2).mean().sqrt()).item()
    peak = g[resid.argmax()].abs().item()
    print("4. GeLU <= SiLU (cross-family), precise fp64 floor:")
    print(f"     best beta = {b:.6f}")
    print(f"     max|GeLU - Swish_beta| = {floor:.6e}   at x = {xm:.4f}  "
          f"({100 * floor / peak:.2f}% of the value there)")
    print(f"     RMS error = {rms:.6e}")
    print(f"     it is a FLOOR:  beta-0.1 -> {maxerr(b - 0.1):.4e},  "
          f"beta+0.1 -> {maxerr(b + 0.1):.4e}   (no resource drives it to 0)\n")
    return b, floor


# --- 5. smooth activation <= ReLU network (width) ----------------------------

def width_approx(L=8.0):
    xf = torch.linspace(-L, L, 20001, dtype=torch.float64)
    g = gelu(xf)
    print("5. GeLU <= ReLU-network via PWL interpolation (a width-N ReLU net):")
    for n in [4, 8, 16, 32, 64, 128]:
        knots = torch.linspace(-L, L, n, dtype=torch.float64)
        kv = gelu(knots)
        # piecewise-linear interpolation == a ReLU net with n-1 breakpoints
        idx = torch.searchsorted(knots, xf).clamp(1, n - 1)
        x0, x1 = knots[idx - 1], knots[idx]
        y0, y1 = kv[idx - 1], kv[idx]
        interp = y0 + (y1 - y0) * (xf - x0) / (x1 - x0)
        err = (interp - g).abs().max().item()
        print(f"     N={n:>4} knots   max|interp - GeLU| = {err:.3e}")
    print("   -> smooth acts are exact only in the wide limit; currency is width\n")


# --- 6. real model: swap gpt2's GeLU ----------------------------------------

def _act_fn(kind, beta=1.702, n_knots=16, L=8.0):
    if kind.startswith("swish"):
        return lambda x: swish_b(x, beta)
    if kind == "relu":
        return F.relu
    if kind == "gelu":
        return gelu
    if kind == "pwl":
        knots = torch.linspace(-L, L, n_knots)
        kv = gelu(knots)

        def f(x):
            k = knots.to(x.device, x.dtype)
            v = kv.to(x.device, x.dtype)
            idx = torch.searchsorted(k, x).clamp(1, n_knots - 1)
            x0, x1 = k[idx - 1], k[idx]
            return v[idx - 1] + (v[idx] - v[idx - 1]) * (x - x0) / (x1 - x0)

        return f
    raise ValueError(kind)


def run_gpt2(best_beta, max_length=128, num_samples=32):
    model, tok, device = H.load("gpt2", dtype=torch.float32)
    acts = [blk.mlp.act for blk in model.transformer.h]
    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    print(f"6. real gpt2 activation swap (native = GeLU), {len(acts)} MLP blocks:")
    H.print_row("native GeLU", H.compare(base, base))
    variants = [
        (f"Swish_{best_beta:.3f} (fit)", _act_fn("swish", beta=best_beta)),
        ("Swish_1.0", _act_fn("swish", beta=1.0)),
        ("ReLU", _act_fn("relu")),
        ("PWL-8", _act_fn("pwl", n_knots=8)),
        ("PWL-32", _act_fn("pwl", n_knots=32)),
    ]
    for label, fn in variants:
        with H.patched_forwards(acts, lambda m, fn=fn: (lambda self, x: fn(x))):
            logits = H.logits_for(model, batches)
        H.print_row(label, H.compare(base, logits))
    print("reading: Swish at the fitted beta tracks GeLU far better than Swish_1; "
          "ReLU (the beta->inf limit) drifts; PWL approaches native as knots grow.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt2", action="store_true")
    args = ap.parse_args()
    limit_families()
    exact_reparam()
    maxout_universal()
    best_b, _ = gelu_silu_floor()
    width_approx()
    if args.gpt2:
        run_gpt2(best_b)
