"""Historical analyses of architecture evolution (uses repo creation dates).

  1. KV-scheme share + median kv_cap over time  -> the compression trend.
  2. Architecture diversity (new / cumulative classes) -> converging or diverging?
  3. Most-universal checkpoint among all models released so far -> frontier evolution.
"""
import json
import numpy as np
from graph.model_graph import (classify, dims_of, _dedup_points, emul_matrix,
                               short, _junky, SLOTS)


def period(ca):
    return f"{ca.year}-H{1 if ca.month <= 6 else 2}"


def main():
    from huggingface_hub import list_models
    dates = {}
    for m in list_models(filter="text-generation", sort="downloads", direction=-1,
                         limit=5000, expand=["createdAt"]):
        if getattr(m, "created_at", None):
            dates[m.id] = m.created_at

    rows = json.load(open("graph/hub_10k.json"))
    seen, ms = set(), []
    for r in rows:
        if _junky(r["id"]) or r["id"] in seen:
            continue
        c = classify(r["config"]); d = dims_of(r["config"]); ca = dates.get(r["id"])
        if c and d["hidden"] and d["layers"] and ca:
            seen.add(r["id"]); ms.append({"name": r["id"], "slots": c, "dims": d, "date": ca})
    periods = sorted({period(m["date"]) for m in ms})
    out = [f"{len(ms)} dated models.\n"]

    out.append("1. KV-scheme share + median kv_cap among models created each period:")
    out.append(f"   {'period':8} {'n':>4}   {'MHA':>4} {'GQA':>4} {'MQA':>4} {'MLA':>4}   {'median kv_cap':>13}")
    for p in periods:
        s = [m for m in ms if period(m["date"]) == p]
        if len(s) < 4:
            continue
        kv = [m["slots"]["kv"] for m in s]
        med = sorted(m["dims"]["kv_cap"] for m in s)[len(s) // 2]
        n = len(s)
        out.append(f"   {p:8} {n:>4}   " + " ".join(f"{100*kv.count(x)/n:>3.0f}%"
                   for x in ("MHA", "GQA", "MQA", "MLA")) + f"   {med:>13}")

    out.append("\n2. architecture diversity (distinct 9-slot classes):")
    out.append(f"   {'period':8} {'n':>4}   {'new classes':>11} {'cumulative':>11}")
    known = set()
    for p in periods:
        s = [m for m in ms if period(m["date"]) == p]
        if len(s) < 4:
            continue
        cls = {tuple(m["slots"][k] for k, _ in SLOTS) for m in s}
        new = cls - known; known |= cls
        out.append(f"   {p:8} {len(s):>4}   {len(new):>11} {len(known):>11}")

    out.append("\n3. most-universal checkpoint among ALL models released so far (cumulative):")
    out.append(f"   {'by end of':10} {'models':>6}   {'top emulator':32} {'covers':>7}")
    for p in periods:
        s = [m for m in ms if period(m["date"]) <= p]
        if len(s) < 10:
            continue
        P = _dedup_points(s)
        ok = emul_matrix(P)
        cov = (ok * np.array([len(x["members"]) for x in P])[:, None]).sum(0)
        j = int(cov.argmax())
        out.append(f"   {p:10} {len(s):>6}   {short(P[j]['members'][0]):32} {100*cov[j]/len(s):>6.0f}%")

    open("graph/historical.txt", "w").write("\n".join(out))
    print("done")


if __name__ == "__main__":
    main()
