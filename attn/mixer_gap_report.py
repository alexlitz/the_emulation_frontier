"""Tabulate attn/data/mixer_gap_agreement.jsonl (attn/mixer_gap_agreement.py, attn/mixer_gap_distill.py):
is the softmax <- SSM / linear / hybrid near-gap payable?

Definitions (the act/act_swap_report.py standard):
  strict   top-1 >= 0.99 and >= 90% of 32-token greedy continuations identical   (THE standard)
  bf16     no worse than the ORIGINAL model run in bf16: top-1 >= top-1(bf16), ex32 >= ex32(bf16)
  loose    top-1 >= 0.98 and ex32 >= 0.50

Per family the table lists every construction / budget variant with its price per target mixer
layer (host heads, host layers, whether state must be carried under a non-stock pre-norm). A family's
edge would be payable only if some construction passes AT A PRICE A ZOO HOST CAN PAY; the verdicts
recorded in graph/model_graph.py MIXER_GAP are checked against these rows (tests/test_mixer_gap.py).

  python -m attn.mixer_gap_report          # table + verdicts, writes attn/fig_mixer_gap.png
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict

DATA = "attn/data/mixer_gap_agreement.jsonl"
FIG = "attn/fig_mixer_gap.png"
FAMILY_NAME = {"mamba": "Mamba-1 (mamba-130m)", "rwkv4": "RWKV-4 (rwkv-4-169m)",
               "granite": "Mamba-2 hybrid (granite-4.0-h-350m)", "lfm2": "short-conv hybrid (LFM2-350M)",
               "qwen35": "gated-DeltaNet hybrid (Qwen3.5-0.8B)"}
DEFS = {
    "strict": lambda r, c: r["top1"] >= 0.99 and r["ex32"] >= 0.90,
    "bf16": lambda r, c: c is not None and r["top1"] >= c["top1"] and r["ex32"] >= c["ex32"],
    "loose": lambda r, c: r["top1"] >= 0.98 and r["ex32"] >= 0.50,
}


# COMPLETE constructions: every piece of the mixer built from softmax heads / SiLU-GLU units (no oracle
# prefix sum, conv, shift or scalar nonlinearity). The other variants isolate one piece each.
COMPLETE = re.compile(r"^(full-construction.*|conv\d+|onelayer-.*)$")


def load(path=DATA):
    rows = OrderedDict()
    for line in open(path):
        r = json.loads(line)
        key = (r["family"], r["variant"], tuple(r["layers"]) if r.get("layers") else None)
        rows[key] = r                                   # the last measurement wins
    return list(rows.values())


def whole(rows):
    return [r for r in rows if not r.get("layers")]


def controls(rows):
    return {r["family"]: r for r in whole(rows) if r["variant"] == "bf16"}


def verdicts(r, c):
    return {n: bool(ok(r, c)) for n, ok in DEFS.items()}


def _price(cost):
    if not cost or "heads" not in cost:
        return "-"
    d = cost.get("depth", "?")
    return f"{cost['heads']:,} heads x {d} layer{'s' * (d != 1)}" + (" + group norm" if cost.get("groupnorm") else "")


def table(rows):
    ctrl = controls(rows)
    lines = ["| family | variant | price per mixer layer | top-1 | ident. 32 | ident. 64 | mean KL | strict | bf16 | loose |",
             "|---|---|---|---:|---:|---:|---:|:-:|:-:|:-:|"]
    for r in whole(rows):
        c = ctrl.get(r["family"])
        v = verdicts(r, c)
        mark = lambda b: "pass" if b else "fail"
        lines.append(f"| {FAMILY_NAME.get(r['family'], r['family'])} | {r['variant']} | {_price(r.get('cost'))} | "
                     f"{r['top1']:.4f} | {r.get('ex32', 0):.0%} | {r.get('ex64', 0):.0%} | {r['kl']:.2e} | "
                     f"{mark(v['strict'])} | {mark(v['bf16'])} | {mark(v['loose'])} |")
    return "\n".join(lines)


def per_layer(rows):
    out = OrderedDict()
    for r in rows:
        if r.get("layers") and len(r["layers"]) == 1:
            out.setdefault((r["family"], r["variant"]), []).append((r["layers"][0], r))
    return out


# --- scaling panels: (family, variant regex with one numeric group, x label, log x) --------------
PANELS = [
    ("mamba", r"^sink-L([\d.]+)$", "Mamba-1: sink margin L", False),
    ("mamba", r"^heads(\d+)$", "Mamba-1: decay rates H (exact: 24,576)", True),
    ("rwkv4", r"^softmax-m(\d+)$", "RWKV-4: DFT frequencies m (bonus u)", False),
    ("rwkv4", r"^softmax-m128-shift([\d.]+)$", "RWKV-4: token-shift beta", True),
    ("granite", r"^smallT-eps([\d.e-]+)$", "Mamba-2: temperature eps (fp32)", True),
    ("lfm2", r"^conv([\d.]+)$", "LFM2 short conv: shift beta", True),
    ("qwen35", r"^neumann-p(\d+)$", "gated DeltaNet: Jacobi hops p", False),
    ("mamba", r"^sink-L30-softplus(\d+)$", "Mamba-1: SiLU units K (softplus)", True),
]


def panel_points(rows, fam, rx):
    pts = []
    for r in whole(rows):
        m = re.match(rx, r["variant"]) if r["family"] == fam else None
        if m:
            pts.append((float(m.group(1)), r))
        elif fam == "mamba" and rx.startswith("^heads") and r["family"] == fam and r["variant"] == "sink-L30":
            pts.append((24576.0, r))                     # one head per (channel, state): the full construction
    return sorted(pts, key=lambda p: p[0])


def figure(rows, out=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [p for p in PANELS if panel_points(rows, p[0], p[1])]
    nc = 4
    nr = (len(panels) + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(13, 3.1 * nr), sharey=True, squeeze=False)
    axs = axs.ravel()
    ink, muted, grid = "#0b0b0b", "#8a8984", "#e6e5e0"
    blue, orange = "#2a78d6", "#eb6834"
    ctrl = controls(rows)
    for ax, (fam, rx, title, logx) in zip(axs, panels):
        pts = panel_points(rows, fam, rx)
        xs = [x if x > 0 else (0.5 if logx else 0) for x, _ in pts]
        ax.plot(xs, [r["top1"] for _, r in pts], color=blue, lw=2, marker="o", ms=6,
                mec="white", mew=1.2, label="top-1 agreement", zorder=3)
        ax.plot(xs, [r["ex32"] for _, r in pts], color=orange, lw=2, marker="D", ms=6,
                mec="white", mew=1.2, label="identical 32-token gen.", zorder=3)
        ax.axhline(0.99, color=ink, lw=0.8, ls="-", zorder=1)
        ax.axhline(0.90, color=ink, lw=0.8, ls=":", zorder=1)
        if fam in ctrl:
            ax.axhline(ctrl[fam]["ex32"], color=muted, lw=1, ls="--", zorder=1)
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.minorticks_off()
        ax.set_title(title, fontsize=8.5, color=ink)
        ax.tick_params(labelsize=7, colors=ink)
        ax.set_ylim(-0.03, 1.03)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(axis="y", color=grid, lw=0.6, zorder=0)
    for ax in axs[len(panels):]:
        ax.set_visible(False)
    for k in range(0, len(panels), nc):
        axs[k].set_ylabel("agreement with the original", fontsize=8, color=ink)
    axs[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Softmax constructions of real recurrent mixers (every mixer layer swapped), scored on the original's "
                 "own greedy continuations.\nSolid: top-1 99%; dotted: 90% identical (the payable standard); dashed: "
                 "identical-32 of the original model run in bf16. Each passing point costs heads / depth the edge never charged.",
                 fontsize=8.5, color=ink)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


def main():
    rows = load()
    print(table(rows))
    ctrl = controls(rows)
    print()
    for fam in FAMILY_NAME:
        fr = [r for r in whole(rows) if r["family"] == fam and r["variant"] not in ("exact", "bf16")]
        if not fr:
            continue
        ok = [r for r in fr if DEFS["strict"](r, ctrl.get(fam))]
        comp = [r for r in ok if COMPLETE.match(r["variant"])]
        print(f"{FAMILY_NAME[fam]}: {len(ok)}/{len(fr)} variants pass strict; complete constructions passing: "
              + (", ".join(f"{r['variant']} ({_price(r['cost'])})" for r in comp) or "none"))
    for (fam, var), lst in per_layer(rows).items():
        lst = sorted(lst)
        print(f"per-layer {fam} {var} (top-1" + ("/ident.32" if "ex32" in lst[0][1] else ", teacher-forced") + "): "
              + "  ".join(f"L{i} {r['top1']:.3f}" + (f"/{r['ex32']:.2f}" if "ex32" in r else "") for i, r in lst))
    print(figure(rows))


if __name__ == "__main__":
    main()
