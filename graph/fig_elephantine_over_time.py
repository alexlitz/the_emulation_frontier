"""Blog figure: how big the elephantine (the per-axis join of every model on the Hub so far) would
have been at each month, against the largest single model that existed then, on each axis.

Models are dated by their earliest Hub upload (graph/model_meta.json), as in frontier_over_time.
At month t the join is graph.minimal_emulator.join over the models present by t, and "the largest
model" is the present model with the most parameters on the same size proxy (minimal_emulator.size).
On a single axis the join equals that axis's record holder (MLA caches count 2x), so the gap in each
panel is how far the biggest model sits from the record on that axis; the last panel shows the gap
compounding into total parameters.

  python -m graph.fig_elephantine_over_time      # -> graph/fig_elephantine_over_time.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from graph.minimal_emulator import join, load, size

META = "graph/model_meta.json"
OUT = "graph/fig_elephantine_over_time.png"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
ELE, BIG = "#2a78d6", "#eb6834"
PANELS = [("hidden", "hidden width"), ("eff_layers", "depth (layers)"), ("kv_cap", "KV cache width / token"),
          ("qheads", "query heads"), ("ffn_total", "FFN units per layer"), ("params", "total parameters")]


def _fmt(v, _):
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g}{suf}"
    return f"{v:g}"


def months(lo, hi):
    y, m = map(int, lo.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= hi:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def series():
    meta = json.load(open(META))
    date = {k: v["created_at"][:7] for k, v in meta.items() if v.get("created_at")}
    models = [m for m in load() if m["name"] in date]
    for m in models:
        m["date"] = date[m["name"]]
    ts = months(min(m["date"] for m in models), max(m["date"] for m in models))
    rows = []
    for t in ts:
        present = [m for m in models if m["date"] <= t]
        j = join(present)
        big = max(present, key=lambda m: size(m["dims"])[0])
        row = {"t": t, "big": big["name"].split("/")[-1], "n": len(present)}
        for k, _ in PANELS:
            if k == "params":
                row["ele_" + k], row["big_" + k] = size(j)[0], size(big["dims"])[0]
            else:
                row["ele_" + k], row["big_" + k] = j[k], big["dims"][k]
        rows.append(row)
    return rows, len(models)


def main():
    rows, n = series()
    x = list(range(len(rows)))
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.4), dpi=200, facecolor=SURF)
    for ax, (k, title) in zip(axs.flat, PANELS):
        ax.set_facecolor(SURF)
        e = [r["ele_" + k] for r in rows]
        b = [r["big_" + k] for r in rows]
        ax.step(x, e, where="post", color=ELE, lw=2, label="elephantine (join of every model so far)")
        ax.step(x, b, where="post", color=BIG, lw=2, label="largest single model (by parameters)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ticks = [i for i, r in enumerate(rows) if r["t"].endswith("-01")]
        ax.set_xticks(ticks, [rows[i]["t"][:4] for i in ticks])
        if k == "params":
            ratio = e[-1] / b[-1]
            ax.annotate(f"{e[-1] / 1e12:.0f}T", (x[-1], e[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, color=INK, va="center")
            ax.annotate(f"{b[-1] / 1e12:.1f}T", (x[-1], b[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, color=INK, va="center")
            ax.text(0.03, 0.95, f"now: {ratio:.0f}x the largest model\n({rows[-1]['big']})", transform=ax.transAxes,
                    fontsize=8.5, color=INK2, va="top")
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.006, 0.915), frameon=False, fontsize=9.5, ncol=2,
               labelcolor=INK2)
    fig.text(0.012, 0.975, "How big the elephantine would have been, month by month", fontsize=14,
             color=INK, weight="bold", va="top")
    fig.text(0.012, 0.935, f"{n:,} dated decoder LMs, each counted from its earliest Hub upload; log scales. "
             "On each axis the join is that axis's record; the largest model holds few records at once.",
             fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 0.98, 0.86))
    fig.savefig(OUT, facecolor=SURF)
    last = rows[-1]
    print(f"wrote {OUT}; {last['t']}: elephantine {last['ele_params'] / 1e12:.1f}T vs largest "
          f"{last['big']} {last['big_params'] / 1e12:.2f}T")
    for r in rows:
        if r["t"].endswith("-01") or r is last:
            print(r["t"], r["n"], f"ele {r['ele_params'] / 1e12:.3f}T", f"big {r['big_params'] / 1e12:.3f}T", r["big"])


if __name__ == "__main__":
    main()
