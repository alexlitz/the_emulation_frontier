"""Tabulate act/data/act_swap_agreement.jsonl (act/act_swap_agreement.py) and calibrate ACT_FLOOR_MAX.

For each definition of "high agreement" the calibrated threshold is the largest floor F such that
EVERY measured (model, host) swap with floor <= F meets the definition, on every model (worst case
over models and depths). Definitions:

  strict    top1 >= 0.99 and >= 90% of 32-token greedy continuations identical
  bf16      no worse than serving the ORIGINAL model in bf16 (the default deployment dtype):
            top1 >= top1(bf16) and ex32 >= ex32(bf16), per model
  loose     top1 >= 0.98 and ex32 >= 0.50
  top1-95   top1 >= 0.95 (teacher-forced only)

  python -m act.act_swap_report            # prints the table + thresholds, writes act/fig_act_swap.png
"""
from __future__ import annotations

import json
from collections import OrderedDict

DATA = "act/data/act_swap_agreement.jsonl"
TARGET_ACT = {"gpt2": "GeLU", "gpt2-medium": "GeLU", "EleutherAI/pythia-160m": "GeLU",
              "EleutherAI/pythia-410m": "GeLU", "EleutherAI/pythia-1.4b": "GeLU", "google/gemma-2-2b": "GeLU",
              "facebook/opt-125m": "ReLU", "facebook/opt-350m": "ReLU", "facebook/opt-1.3b": "ReLU",
              "HuggingFaceTB/SmolLM2-135M": "SiLU", "Qwen/Qwen2.5-0.5B": "SiLU", "Qwen/Qwen2.5-1.5B": "SiLU",
              "meta-llama/Llama-3.2-1B": "SiLU", "swiss-ai/Apertus-v1.1-0.5B": "xielu",
              "nanochat-students/nanochat-d20": "relu2", "AlexKoff88/gpt-oss-tinystories": "swigluoai"}
CONTROLS = ("self", "bf16", "fp16")


def load(path=DATA):
    rows = OrderedDict()
    for line in open(path):
        r = json.loads(line)
        rows[(r["model"], r["host"])] = r                 # the last measurement wins
    return list(rows.values())


DEFS = {
    "strict": lambda r, c: r["top1"] >= 0.99 and r["ex32"] >= 0.90,
    "bf16": lambda r, c: r["top1"] >= c["top1"] and r["ex32"] >= c["ex32"],
    "loose": lambda r, c: r["top1"] >= 0.98 and r["ex32"] >= 0.50,
    "top1-95": lambda r, c: r["top1"] >= 0.95,
}


def swaps(rows):
    """(row, bf16 control row) for every host swap that is not the target's own activation."""
    ctrl = {r["model"]: r for r in rows if r["host"] == "bf16"}
    out = []
    for r in rows:
        if r["host"] in CONTROLS or r["model"] not in ctrl:
            continue
        if TARGET_ACT.get(r["model"]) == r["host"]:
            continue                                      # the target's own activation, refit
        out.append((r, ctrl[r["model"]]))
    return out


def threshold(sw, ok):
    """largest F with every swap at floor <= F passing; also the first failing floor."""
    fails = sorted(r["floor"] for r, c in sw if not ok(r, c))
    first_fail = fails[0] if fails else float("inf")
    passing = [r["floor"] for r, c in sw if r["floor"] < first_fail]
    return (max(passing) if passing else None), first_fail


def table(rows):
    lines = ["| model | layers | target act | host act | floor | top-1 | ident. 32 | ident. 64 | mean KL | "
             "in-dist max err |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (TARGET_ACT.get(r["model"], "?"), r["layers"], r["model"],
                                         r["host"] in CONTROLS, r["floor"], r["host"])):
        tgt = TARGET_ACT.get(r["model"], "?")
        host = {"self": "(own module)", "bf16": "(original, bf16)", "fp16": "(original, fp16)"}.get(r["host"], r["host"])
        lines.append(f"| {r['model'].split('/')[-1]} | {r['layers']} | {tgt} | {host} | {r['floor']:.3f} | "
                     f"{r['top1']:.4f} | {r['ex32']:.0%} | {r['ex64']:.0%} | {r['kl']:.2e} | "
                     f"{r.get('err_indist_max', 0):.3g} |")
    return "\n".join(lines)


def figure(rows, out="act/fig_act_swap.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sw = swaps(rows)
    fams = [f for f in ("GeLU", "SiLU", "ReLU", "relu2", "xielu", "swigluoai")
            if any(TARGET_ACT.get(r["model"]) == f for r, _ in sw)]
    nc = 3
    nr = (len(fams) + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(10.5, 3.3 * nr), sharey=True, squeeze=False)
    axs = axs.ravel()
    ink, muted = "#0b0b0b", "#8a8984"
    from graph.model_graph import ACT_FLOOR_MAX
    for ax, fam in zip(axs, fams):
        pts = [(r, c) for r, c in sw if TARGET_ACT.get(r["model"]) == fam]
        for dose, fill in ((False, True), (True, False)):
            sub = [(r, c) for r, c in pts if ("*" in r["host"]) == dose]
            if not sub:
                continue
            x = [max(r["floor"], 1e-4) for r, _ in sub]
            kw = dict(s=34, linewidths=1.2, zorder=3)
            ax.scatter(x, [r["top1"] for r, _ in sub], marker="o",
                       c="#2a78d6" if fill else "white", edgecolors="white" if fill else "#2a78d6",
                       label=None if dose else "top-1 agreement", **kw)
            ax.scatter(x, [r["ex32"] for r, _ in sub], marker="D",
                       c="#eb6834" if fill else "white", edgecolors="white" if fill else "#eb6834",
                       label=None if dose else "identical 32-token gen.", **kw)
        b = [c["top1"] for _, c in pts]
        if b:
            ax.axhline(min(b), color=muted, lw=1, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(7e-5, 2.5)
        ax.axvline(0.15, color=muted, lw=1, ls=":", zorder=1)
        ax.axvline(ACT_FLOOR_MAX, color=ink, lw=1, ls="-", zorder=1)
        nm = len({r["model"] for r, _ in pts})
        ax.set_title(f"target {fam}  ({nm} model{'s' * (nm > 1)})", fontsize=9, color=ink)
        ax.set_xlabel("single-op floor on [-6, 6]  (0 plotted at 1e-4)", fontsize=8, color=ink)
        ax.tick_params(labelsize=7, colors=ink)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
    for ax in axs[len(fams):]:
        ax.set_visible(False)
    for k in range(0, len(fams), nc):
        axs[k].set_ylabel("agreement with the original model", fontsize=8, color=ink)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Activation swap vs single-op floor (filled: real host fits; hollow: scaled fits Y + t(fit - Y)).\n"
                 f"Solid line: calibrated ACT_FLOOR_MAX {ACT_FLOOR_MAX}; dotted: old 0.15; dashed: worst "
                 "top-1 of the original model run in bf16", fontsize=9, color=ink)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


def main():
    rows = load()
    print(table(rows))
    sw = swaps(rows)
    print()
    for name, ok in DEFS.items():
        th, ff = threshold(sw, ok)
        npass = sum(ok(r, c) for r, c in sw)
        print(f"{name:8}: every swap with floor <= {th} passes; first failure at floor {ff:.3f}; "
              f"{npass}/{len(sw)} swaps pass")
    # per-cell (target act, host act) pass rates under each definition
    print()
    cells = OrderedDict()
    for r, c in sorted(sw, key=lambda rc: rc[0]["floor"]):
        cells.setdefault((TARGET_ACT.get(r["model"]), r["host"]), []).append((r, c))
    for (t, h), lst in cells.items():
        print(f"{t:>9} <- {h:<9} floor {lst[0][0]['floor']:.3f}: top1 "
              f"{min(r['top1'] for r, _ in lst):.4f}-{max(r['top1'] for r, _ in lst):.4f}  ex32 "
              f"{min(r['ex32'] for r, _ in lst):.2f}-{max(r['ex32'] for r, _ in lst):.2f}  "
              + "  ".join(f"{n}:{sum(ok(r, c) for r, c in lst)}/{len(lst)}" for n, ok in DEFS.items()))
    print(figure(rows))


if __name__ == "__main__":
    main()
