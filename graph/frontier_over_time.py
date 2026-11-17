"""How big is the emulation frontier (the maximal / Pareto set) over time?

Each architecture class is dated by the *earliest* Hub upload among its members
(graph/model_meta.json). At each month t we take the classes that exist by then
and count how many are maximal — not strictly dominated by any other *present*
class. A new model can (a) be dominated → nothing changes, (b) be incomparable →
the frontier grows, or (c) strictly dominate a sitting frontier model → that one
falls off. So the maximal set need not grow monotonically.

  python -m graph.frontier_over_time
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph.model_graph import (_dedup_points, _junky, classify, dims_of,
                               emul_matrix, iter_json_configs)

JSON = "graph/hub_10k.json"
META = "graph/model_meta.json"


def load():
    meta = json.load(open(META))
    date = {k: v["created_at"][:7] for k, v in meta.items() if v.get("created_at")}
    models = []
    for name, cfg in iter_json_configs(JSON):
        if _junky(name):
            continue
        c = classify(cfg)
        if c is None:
            continue
        d = dims_of(cfg)
        if d["hidden"] and d["layers"]:
            models.append({"name": name, "slots": c, "dims": d, "date": date.get(name)})
    return models


def months(lo, hi):
    y, m = int(lo[:4]), int(lo[5:7])
    out = []
    while (y, m) <= (int(hi[:4]), int(hi[5:7])):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def analyze():
    models = load()
    P = _dedup_points(models)
    # date each class by its EARLIEST member upload
    pdate = []
    for p in P:
        ds = [m for m in (models_by_name.get(n) for n in p["members"]) if m]
        ds = [d for d in ds if d]
        pdate.append(min(ds) if ds else None)
    pdate = np.array([d or "9999-99" for d in pdate])
    counts = np.array([len(p["members"]) for p in P])

    ok = emul_matrix(P, mode="payable")              # HEADLINE: capacity + payable type walls
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)                 # strict[i,j]: i strictly dominated by j

    lo = min(d for d in pdate if d != "9999-99")
    series = []
    hi = max(d for d in pdate if d != "9999-99")
    for t in months(lo, hi):
        present = pdate <= t
        if not present.any():
            continue
        dominated = strict[:, present].any(1)       # some present class strictly dominates i
        front = present & ~dominated
        fi = np.where(front)[0]; reps = []; used = set()   # collapse mutual ties -> classes
        for i in fi:
            if i in used:
                continue
            eq = [j for j in fi if ok[i, j] and ok[j, i]]
            used.update(eq)
            reps.append(i)
        series.append((t, len(reps), int(present.sum()),
                       int(counts[present].sum()), front))
    return P, pdate, counts, series


def main():
    global models_by_name
    ms = load()
    models_by_name = {m["name"]: m["date"] for m in ms}
    P, pdate, counts, series = analyze()

    # milestone table: quarterly
    print(f"{'month':8s} {'frontier':>8} {'classes':>7} {'models':>6}   new frontier entrants")
    prev = set()
    for t, fsz, ncls, nmod, front in series:
        if t.endswith(("-01", "-04", "-07", "-10")) or t == series[-1][0]:
            cur = {i for i in range(len(P)) if front[i]}
            new = cur - prev
            names = [P[i]["members"][0].split("/")[-1] for i in sorted(new, key=lambda i: -counts[i])][:3]
            print(f"{t:8s} {fsz:>8} {ncls:>7} {nmod:>6}   {', '.join(names)}")
            prev = cur

    ts = [s[0] for s in series]
    fsz = [s[1] for s in series]
    ncls = [s[2] for s in series]
    x = np.arange(len(ts))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(x, fsz, color="#c0392b", lw=2.2, label="emulation frontier (maximal set)")
    ax.fill_between(x, fsz, color="#c0392b", alpha=0.12)
    ax.set_ylabel("frontier size (maximal classes)", color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax.twinx()
    ax2.plot(x, ncls, color="#5f86ad", lw=1.6, ls="--", label="distinct configs present")
    ax2.set_ylabel("distinct configs (arch × size) present", color="#5f86ad")
    ax2.tick_params(axis="y", labelcolor="#5f86ad")
    tick = [i for i, t in enumerate(ts) if t.endswith("-01") or t.endswith("-07")]
    ax.set_xticks(tick)
    ax.set_xticklabels([ts[i] for i in tick], rotation=45, ha="right", fontsize=8)
    ax.set_title("Maximal set (emulation frontier) vs. distinct configs, by earliest upload")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig("graph/fig_frontier_time.png", dpi=130)
    print("\nwrote graph/fig_frontier_time.png")
    print(f"final: frontier={fsz[-1]}  classes={ncls[-1]}  ({ts[-1]})")


if __name__ == "__main__":
    main()
