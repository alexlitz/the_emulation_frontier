"""Round-3 relation checks on the snapshot shapes: reflexive, transitive (0 violations) in exact / payable / mixgap /
capacity, exact <= payable <= mixgap <= capacity, and the frontier size per mode (the headline is payable).

Run:  python -m verify.r3.relcheck            # prints the table, exits 1 on any failure
      python -m verify.r3.relcheck --loo a,b  # leave-one-out: the payable headline with each P5_OFF port switched off
"""
from __future__ import annotations

import sys

import numpy as np

from graph import model_graph as mg
from graph.currency_curve import n_frontier, points

MODES = ("exact", "payable", "mixgap", "capacity")


def split(P, ok):
    """(AR, non-AR) frontier classes."""
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)
    front = np.where(~strict.any(1))[0]
    reps, used = [], set()
    for i in front:
        if i in used:
            continue
        used.update(j for j in front if ok[i, j] and ok[j, i])
        reps.append(i)
    ar = sum(P[i]["slots"]["gen"] == "ar" for i in reps)
    return ar, len(reps) - ar


def check(P=None, verbose=True):
    P = P or points()
    M, bad = {}, []
    for m in MODES:
        ok = mg.emul_matrix(P, mode=m)
        M[m] = ok
        S = ok.astype(np.int32)
        viol = int((((S @ S) > 0) & ~ok).sum())
        refl = bool(np.diag(ok).all())
        ar, nar = split(P, ok)
        if verbose:
            print(f"{m:9s} edges {int(ok.sum()):>7}  frontier {ar + nar:>4} ({ar} AR + {nar})  reflexive {refl}  "
                  f"transitivity violations {viol}")
        if not refl or viol:
            bad.append(f"{m}: reflexive {refl}, violations {viol}")
    for a, b in zip(MODES, MODES[1:]):
        extra = int((M[a] & ~M[b]).sum())
        if verbose:
            print(f"{a} <= {b}: {extra} pairs outside")
        if extra:
            bad.append(f"{a} not <= {b}: {extra}")
    if verbose:
        print(f"shapes {len(P)}")
    return M, bad


def loo(specs, P=None):
    """The payable headline with each round-3 spec (R3_OFF, classification included: the points are rebuilt) or
    pass-5 port (P5_OFF, a name in model_graph.P5_PORTS) switched off on its own."""
    base = n_frontier(mg.emul_matrix(points(), mode="payable"))
    print(f"payable headline {base}; leave-one-out (change off -> headline, delta):")
    rows = []
    for s in specs:
        off = mg.P5_OFF if s in mg.P5_PORTS else mg.R3_OFF
        off.add(s)
        try:
            h = n_frontier(mg.emul_matrix(points(), mode="payable"))
        finally:
            off.discard(s)
        rows.append((s, h, h - base))
        print(f"  {s:40s} {h:>4}  {h - base:+d}", flush=True)
    return base, rows


if __name__ == "__main__":
    if "--loo" in sys.argv:
        loo(sys.argv[sys.argv.index("--loo") + 1].split(","))
    else:
        _, bad = check()
        if bad:
            print("FAIL:", bad)
            sys.exit(1)
        print("OK")
