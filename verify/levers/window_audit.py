"""Window audit: every family with sliding-window layers in the snapshot, whether its in-library code reads the
window from the config (so a host may set it >= the context: verify/levers/relax.py window lever), and the models
whose window cannot bind as published (window >= their own max_position_embeddings).

Run:  python -m verify.levers.window_audit     # writes verify/levers/out/window_audit.md
"""
from __future__ import annotations

import collections
import os

from verify.levers.family_code import load_caps
from verify.levers.relax import COMPRESSED
from verify.levers.rel import lm, points

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def run():
    P, cfg = points()
    caps = load_caps()
    fam = caps["families"]
    by = collections.defaultdict(lambda: dict(points=0, local_layers=0, examples=[], beyond=[]))
    for p in P:
        d = p["dims"]
        lu = d.get("kLU_param", 0) + d.get("kLT_param", 0)
        if not lu and p["slots"]["window"] != "local":
            continue
        n = p["members"][0]
        c = lm(cfg[n])
        mt = str(c.get("model_type"))
        b = by[mt]
        b["points"] += 1
        b["local_layers"] += lu
        if len(b["examples"]) < 2:
            b["examples"].append(n.split("/")[-1])
        sw, mp = c.get("sliding_window"), c.get("max_position_embeddings")
        if isinstance(sw, (int, float)) and isinstance(mp, (int, float)) and sw >= mp:
            b["beyond"].append(f"{n.split('/')[-1]} ({sw} >= {mp})")
    lines = [f"transformers main {caps['transformers']}", "",
             "| model_type | points | sliding layers | code reads the window | verdict | examples |",
             "|---|---:|---:|---|---|---|"]
    tot = collections.Counter()
    for mt, b in sorted(by.items(), key=lambda kv: -kv[1]["points"]):
        f = fam.get(mt) or {}
        ev = (f.get("caps") or {}).get("window_cfg")
        if mt in COMPRESSED:
            verdict = "compressed layers stay local (only compress-ratio-0 layers convert)"
        elif ev:
            verdict = "window >= context = full attention"
        elif f.get("code"):
            verdict = "in library, no window read (the window never binds)"
        else:
            verdict = "remote code: not checked"
        tot[verdict] += b["points"]
        lines.append(f"| {mt} | {b['points']} | {b['local_layers']} | {('`' + ev + '`') if ev else '-'} | {verdict} | "
                     f"{', '.join(b['examples'])} |")
    lines += ["", "Points per verdict: " + "; ".join(f"{v}: {k}" for v, k in tot.most_common()), "",
              "Windows that cannot bind as published (window >= own max_position_embeddings): "
              + "; ".join(x for b in by.values() for x in b["beyond"])]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "window_audit.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    run()
