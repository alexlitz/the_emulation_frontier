"""Frontier study, part 2: where round 3's growth (pass 5's 248 -> today's headline) comes from.

Every round-3 change is keyed (graph/model_graph.py R3_OFF). This computes the payable frontier (by member name) with
every key off (the pass-5 relation on today's snapshot reading), with everything on, and with each key off on its own
(leave-one-out) and on its own (leave-one-in from all-off), and records which frontier models each key adds / removes.

Run:  python -m graph.frontier_study_growth [--jobs 4]   # writes dist/frontier_study/growth.json
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

OUT = "dist/frontier_study"


def r3_keys():
    from graph import model_graph as mg
    ks = set()
    for t in mg._R3_TABLES:
        ks |= set(t.r3_add) | set(t.r3_drop)
    src = open(mg.__file__).read()
    ks |= set(re.findall(r'_r3\(\s*"([A-Za-z0-9_]+)"', src))
    return sorted(ks)


def frontier_names(off):
    """Frontier class representatives' member lists with the keys in `off` switched off."""
    from graph import model_graph as mg
    from graph.currency_curve import points
    from graph.frontier_study import reps_strict
    saved = set(mg.R3_OFF)
    mg.R3_OFF.clear()
    mg.R3_OFF.update(off)
    try:
        P = points()
        ok = mg.emul_matrix(P, mode="payable")
        reps = reps_strict(ok)
        return [P[i]["members"] for i in reps], len(P)
    finally:
        mg.R3_OFF.clear()
        mg.R3_OFF.update(saved)


def _job(args):
    tag, off = args
    fr, npts = frontier_names(off)
    return tag, fr, npts


def main():
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    keys = r3_keys()
    tasks = [("all_on", []), ("all_off", keys)]
    tasks += [(f"loo:{k}", [k]) for k in keys]
    tasks += [(f"loi:{k}", [x for x in keys if x != k]) for k in keys]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/growth_raw.json"
    done = json.load(open(path)) if os.path.exists(path) else {}
    todo = [t for t in tasks if t[0] not in done]
    print(f"{len(keys)} keys, {len(todo)} runs to do", flush=True)
    from multiprocessing import Pool
    with Pool(jobs) as pool:
        for tag, fr, npts in pool.imap_unordered(_job, todo):
            done[tag] = dict(frontier=fr, shapes=npts)
            print(tag, len(fr), flush=True)
            json.dump(done, open(path, "w"))


if __name__ == "__main__":
    main()
