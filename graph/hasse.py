"""Render a Hasse diagram (poset covering diagram) for each per-slot order.

Each `*_ORDER` dict already stores immediate parents = covering relations, so it
IS a Hasse diagram; we just draw it. Higher = more general (emulates more); edges
labelled by currency, dashed = approximate/limit edge."""
import subprocess

from graph.model_graph import (ACT_ORDER, BLOCK_ORDER, FFN_ORDER, KV_ORDER, NORM_ORDER,
                               POS_ORDER, QKNORM_ORDER, SOFTCAP_ORDER, WINDOW_ORDER)

ORDERS = {
    "positional": POS_ORDER, "kv_sharing": KV_ORDER, "ffn_moe": FFN_ORDER,
    "activation": ACT_ORDER, "normalization": NORM_ORDER, "block_structure": BLOCK_ORDER,
    "attn_window": WINDOW_ORDER, "qk_norm": QKNORM_ORDER, "logit_softcap": SOFTCAP_ORDER,
}
COLOR = {"free": "#4a4", "fold": "#48c", "mag": "#c62", "kv": "#a4a", "width": "#a80",
         "depth": "#a80", "cap": "#a80", "approx": "#c62", "mask": "#4a4", "rot": "#a80"}


def render(name, order, path):
    nodes = set(order) | {p for v in order.values() for p, _, _ in v}
    lines = [f"digraph {name} {{",
             '  bgcolor="transparent"; rankdir=LR; margin=0;',   # transparent, landscape (right = more general)
             '  node [shape=box, style="rounded,filled", fillcolor="#f4f4f4", '
             'fontname="Helvetica", fontsize=11]; edge [fontname="Helvetica", fontsize=9];']
    for n in sorted(nodes):
        abstract = n.endswith("_top") or n == "AffineNorm" or n == "additive_bias"
        fill = "#ffe08a" if abstract else "#eef3fb"
        lines.append(f'  "{n}" [fillcolor="{fill}"];')
    for a, parents in order.items():
        for (b, cur, exact) in parents:
            style = "solid" if exact else "dashed"
            col = COLOR.get(cur, "#888")
            lines.append(f'  "{a}" -> "{b}" [label=" {cur}", color="{col}", '
                         f'fontcolor="{col}", style={style}];')
    lines.append("}")
    open(path + ".dot", "w").write("\n".join(lines))
    try:
        subprocess.run(["dot", "-Tsvg", path + ".dot", "-o", path + ".svg"],
                       check=True, capture_output=True)
        return True
    except Exception:
        return False


def main():
    done = []
    for name, order in ORDERS.items():
        ok = render(name, order, f"graph/hasse_{name}")
        done.append((name, len(set(order) | {p for v in order.values() for p, _, _ in v}), ok))
    with open("graph/hasse_index.txt", "w") as f:
        f.write("Hasse diagrams (one per architectural slot):\n\n")
        for name, n, ok in done:
            f.write(f"  graph/hasse_{name}.svg   {n} nodes   {'rendered' if ok else 'DOT only'}\n")
    print("done")


if __name__ == "__main__":
    main()
