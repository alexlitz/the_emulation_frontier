"""The levers stacked, and the walls that remain once the rules as stated are applied (R0).

  1. a chain of choices: baseline -> R0 (window + eps, the rules as stated) -> drafts excluded -> (c) numeric
     scalars -> (a) every shape-preserving config value -> DSA ignored;
  2. leave-one-out over the components ON TOP OF R0 (the wall leverage that remains);
  3. a greedy order: the component whose removal frees the most frontier models, removed one after another
     (on R0), 12 steps.

Run:  python -m verify.levers.summary     # writes verify/levers/out/summary.json
"""
from __future__ import annotations

import json
import os

import numpy as np

from verify.levers.extras import DRAFT
from verify.levers.family_code import load_caps
from verify.levers.relax import R0, SETS, apply
from verify.levers.rel import AND, components, frontier_set, left, points, short

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
# pure capacity comparisons (size, not type): not walls a relaxation could remove
CAPACITY = {"kv_cap", "mla_penalty", "q_cap", "qheads", "head_dim", "ffn_total", "ffn_active", "hidden", "depth_eff",
            "depth_param", "kinds_eff", "kinds_param", "norm_mirror"}


def run():
    P, cfg = points()
    fam = load_caps()["families"]
    keep = [i for i, p in enumerate(P) if any(not DRAFT.search(n) for n in p["members"])]
    ix = np.ix_(keep, keep)
    Pk = [P[i] for i in keep]
    out = {"chain": []}

    def f(ok, sub=False):
        return len(frontier_set(ok[ix], Pk) if sub else frontier_set(ok, P))

    C = components(P)
    CR = components(P, apply(P, cfg, fam, R0))
    CC = components(P, apply(P, cfg, fam, SETS["(c) numeric scalars + R0"]))
    CA = components(P, apply(P, cfg, fam, SETS["(a) all shape-preserving + R0"]))
    dsa = ("feat:dsa:target", "feat:dsa:host")
    chain = [("headline", AND(C), False), ("R0: window + eps (rules as stated)", AND(CR), False),
             ("R0 + drafts / assistants excluded", AND(CR), True),
             ("+ (c) numeric scalars", AND(CC), True), ("+ (a) every shape-preserving config value", AND(CA), True),
             ("+ DSA ignored (context <= index_topk)", AND(CA, drop=dsa), True)]
    for name, ok, sub in chain:
        n = f(ok, sub)
        out["chain"].append(dict(step=name, frontier=n))
        print(f"{name:55s} {n}")
    # leave-one-out on R0
    okR = AND(CR)
    repsR = frontier_set(okR, P)
    loo = []
    for k in CR:
        ok = AND(CR, drop=(k,))
        g = left(repsR, ok)
        loo.append(dict(rule=k, frontier=len(frontier_set(ok, P)), left=[short(P, i) for i in g][:8]))
    loo.sort(key=lambda r: r["frontier"])
    out["loo_on_R0"] = loo
    print("\nleave-one-out on R0 (top):")
    for r in loo[:15]:
        print(f"  {r['rule']:28s} {r['frontier']}  {r['left'][:5]}")
    # greedy on R0, over the type walls only
    dropped, greedy = [], []
    cur = len(repsR)
    for _ in range(12):
        best = None
        for k in CR:
            if k in dropped or k in CAPACITY:
                continue
            n = len(frontier_set(AND(CR, drop=tuple(dropped + [k])), P))
            if best is None or n < best[1]:
                best = (k, n)
        dropped.append(best[0])
        greedy.append(dict(drop=best[0], frontier=best[1], freed=cur - best[1]))
        cur = best[1]
    out["greedy_on_R0"] = greedy
    print("\ngreedy on R0:")
    for g in greedy:
        print(f"  - {g['drop']:28s} -> {g['frontier']} (-{g['freed']})")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    run()
