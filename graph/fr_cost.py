"""fr-cost: which construction CHARGES keep frontier classes on the payable frontier, and what a cheaper construction
would free.

For each charge name in model_graph.COST_RELAX_NAMES the charge is set to its floor (multiplier 1, extra 0) with every
type wall kept (model_graph.COST_RELAX: a counterfactual hook, empty in the relation). The pairs that pass only then are
COST-BLOCKED by that charge; a frontier class that acquires a strict dominator is FREED by it. Per freed class the binding
pair is reported with the base relation's failing checks and the needed / available numbers on the charged axis.

Run:  python -m graph.fr_cost                 # writes dist/fr_cost/rank.json and prints the ranked table
      python -m graph.fr_cost --combos        # + every pair of charges: the classes only the pair frees (combos.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from graph import model_graph as mg
from graph.currency_curve import n_frontier, points

OUT = "dist/fr_cost"

COST_CHECKS = ("hidden", "kv_cap", "q_cap", "qheads", "head_dim", "ffn width", "ffn per-layer widths", "depth (eff)",
               "depth (param)", "kv heads", "layer kinds", "kv MLA latents", "recurrent dims", "ffn MoE routing", "K=V")


def reps_of(ok):
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)
    front = np.where(~strict.any(1))[0]
    reps, used = [], set()
    for i in front:
        if i in used:
            continue
        eq = [j for j in front if ok[i, j] and ok[j, i]]
        used.update(eq)
        reps.append(int(i))
    return reps


def relation(P, relax=(), mode="payable", explain=None):
    old = set(mg.COST_RELAX)
    mg.COST_RELAX.clear()
    mg.COST_RELAX.update(relax)
    try:
        return mg.emul_matrix(P, mode=mode, explain=explain)
    finally:
        mg.COST_RELAX.clear()
        mg.COST_RELAX.update(old)


def name(P, i):
    return P[i]["members"][0]


def axis_numbers(P, i, j, checks):
    """The charged axes' target / host numbers for a refused pair (the base relation)."""
    dt, dh = P[i]["dims"], P[j]["dims"]
    out = {}
    for c, keys in (("hidden", ("hidden",)), ("ffn width", ("ffn_total", "ffn_active")),
                    ("depth (eff)", ("eff_layers",)), ("depth (param)", ("param_layers",)),
                    ("kv_cap", ("kv_cap",)), ("q_cap", ("q_cap",)), ("kv heads", ("kv_heads", "q_per_kv", "head_dim")),
                    ("head_dim", ("head_dim",)), ("qheads", ("qheads",))):
        if c in checks:
            out[c] = {k: (int(dt.get(k, 0)), int(dh.get(k, 0))) for k in keys}
    return out


def analyse(P=None, names=None, log=print):
    P = P or points()
    names = names or mg.COST_RELAX_NAMES
    ex = {}
    base = relation(P, explain=ex)
    reps = reps_of(base)
    rs = set(reps)
    nb = len(reps)
    log(f"payable base: {len(P)} shapes, frontier {nb}")
    rows = []
    for k in names:
        ok = relation(P, (k,))
        new = ok & ~base
        strict = ok & ~ok.T
        np.fill_diagonal(strict, False)
        freed = [i for i in reps if strict[i].any()]
        info = []
        for i in freed:
            hosts = [int(j) for j in np.nonzero(strict[i])[0]]
            # the binding pair: a host whose base refusal is cost-only, widest first
            best = max(hosts, key=lambda j: int(base[:, j].sum()))
            checks = sorted(c for c, m in ex.items() if m[i, best])
            info.append({"target": name(P, i), "host": name(P, best), "n_hosts": len(hosts),
                         "base_checks": checks, "axes": axis_numbers(P, i, best, checks)})
        nf = n_frontier(ok)
        rows.append({"charge": k, "frontier": nf, "delta": nf - nb, "new_pairs": int(new.sum()),
                     "new_pairs_frontier_targets": int(new[reps].sum()), "freed": info})
        log(f"  {k:14s} frontier {nf:4d} ({nf - nb:+d})  new pairs {int(new.sum()):6d}  "
            f"freed {', '.join(x['target'].split('/')[-1] for x in info)}", flush=True)
    allk = relation(P, tuple(names))
    na = n_frontier(allk)
    log(f"  {'ALL':14s} frontier {na:4d} ({na - nb:+d})  new pairs {int((allk & ~base).sum()):6d}")
    # cost-only blocked pairs (base explain): every refusing check is a cost / capacity check
    cost_only = []
    for i in reps:
        for j in range(len(P)):
            if base[i, j] or i == j:
                continue
            checks = [c for c, m in ex.items() if m[i, j]]
            if checks and all(c in COST_CHECKS for c in checks):
                cost_only.append((i, j, tuple(sorted(checks))))
    by_class = {}
    for i, j, c in cost_only:
        by_class.setdefault(i, []).append((j, c))
    return {"n_shapes": len(P), "frontier": nb, "rows": rows, "all_relaxed": na,
            "cost_only_pairs": len(cost_only), "cost_only_classes": len(by_class),
            "cost_only": {name(P, i): [(name(P, j), list(c)) for j, c in v[:8]] for i, v in by_class.items()}}


def combos(P=None, names=None, log=print):
    """{"a+b": [classes freed by relaxing a and b together, not by either alone]} over every pair of charges."""
    import itertools
    P = P or points()
    names = names or mg.COST_RELAX_NAMES
    reps = reps_of(relation(P))

    def freed(relax):
        ok = relation(P, relax)
        s = ok & ~ok.T
        np.fill_diagonal(s, False)
        return {i for i in reps if s[i].any()}
    single = {k: freed((k,)) for k in names}
    out = {}
    for a, b in itertools.combinations(names, 2):
        fr = freed((a, b)) - single[a] - single[b]
        if fr:
            out[f"{a}+{b}"] = sorted(name(P, i) for i in fr)
            log(f"  {a} + {b}: {out[f'{a}+{b}']}", flush=True)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--combos" in sys.argv:
        with open(os.path.join(OUT, "combos.json"), "w") as f:
            json.dump(combos(), f, indent=1)
    r = analyse()
    with open(os.path.join(OUT, "rank.json"), "w") as f:
        json.dump(r, f, indent=1)
    print(f"cost-only refused (target frontier class, host) pairs: {r['cost_only_pairs']} over "
          f"{r['cost_only_classes']} classes")


if __name__ == "__main__":
    sys.exit(main())
