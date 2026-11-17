"""Figures for the diffusion side of the emulation graph.

  fig_diffusion_paths.{svg,png}  every diffusion LM in the snapshot, deduplicated by shape, with the
      emulation paths among them (Hasse diagram of the HEADLINE emul_matrix(mode="payable")
      restricted to gen=diffusion; an arrow A -> B means B emulates A, and green marks edges that
      are exact in the strict order). Frontier tops (maximal diffusion shapes) are highlighted. The
      grey line names the AR model each diffusion shape shares its tensors with: its declared
      base_model if the card gives one, else the most recognisable AR checkpoint of identical shape.
      The wall is ONE-WAY: a diffusion host emulates its AR twin by applying the causal mask
      (AR <= diffusion, attn/ar_diffusion.py, exact), but no AR model emulates a diffusion one. The
      universal host (elephantine/universal.py) carries both sides by switching the mask.
  fig_diffusion_mask.png  why the wall exists, and how one host spans it: a causal mask decoding
      left to right, one token per step, vs a full (bidirectional) mask unmasking tokens in parallel
      over denoising steps.

  python -m graph.fig_diffusion
"""
from __future__ import annotations

import subprocess
from collections import defaultdict

import numpy as np

from graph.model_graph import _junky, classify, dims_of, emul_matrix, iter_json_configs

SNAP = "graph/hub_10k.json"
SHAPE = ("hidden", "layers", "kv_cap", "q_cap", "ffn_total", "ffn_active", "head_dim")


def points():
    out = []
    for r, c in iter_json_configs(SNAP):
        if _junky(r):
            continue
        try:
            s = classify(c)
        except Exception:
            continue
        if s:
            out.append((r, {"slots": s, "dims": dims_of(c), "mt": c.get("model_type")}))
    return out


def _short(r):
    return r.split("/")[-1]


CANON = ("Qwen/", "meta-llama/", "mistralai/", "google/", "nvidia/", "microsoft/", "deepseek-ai/",
         "ibm-granite/", "allenai/", "inclusionAI/")


def _canonical(repos):
    """The most recognisable AR checkpoint of a same-shape family: an official org's repo, else one
    whose name is a well-known base family, else the shortest name."""
    import re
    fam = re.compile(r"^(Qwen\d|Llama-\d|Meta-Llama|Ministral|Mistral|gemma|Gemma|Phi-|granite)", re.I)
    rank = lambda r: (not r.startswith(CANON), not fam.match(r.split("/")[-1]), len(r.split("/")[-1]), r)
    name = min(repos, key=rank).split("/")[-1]
    m = re.match(r"^(.*?\d+(?:\.\d+)?B(?:-A\d+(?:\.\d+)?B)?)", name)    # cut "-Instruct-2512-..." etc.
    return m.group(1) if m else name


def _lineage(members, twin, diffusion_ids=()):
    """Grey third line: the declared AR base (model card base_model) if any, else the same-shape AR
    family. Declared lineage is fetched once from the Hub (cached in graph/diffusion_lineage.json).
    A declared base that is itself a diffusion LM (LLaDA2.1-mini-ORL -> LLaDA2.1-mini) is not an AR
    base and is skipped."""
    import json, os
    path = "graph/diffusion_lineage.json"
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for m in members:
        if m not in cache:
            try:
                from huggingface_hub import model_info
                cd = model_info(m).card_data or {}
                bm = cd.get("base_model") if hasattr(cd, "get") else None
                cache[m] = bm[0] if isinstance(bm, list) and bm else bm
            except Exception:
                cache[m] = None
    json.dump(cache, open(path, "w"), indent=1, sort_keys=True)
    is_diff = lambda b: b in diffusion_ids or any(t in b.lower() for t in ("diffusion", "llada"))
    declared = [cache[m] for m in members if cache.get(m) and not is_diff(cache[m])]
    if declared:
        return f"from {declared[0].split('/')[-1]} (AR)"
    return f"same tensors as {_canonical(twin)} (AR)" if twin else ""


def _label(members, p, line3=""):
    d = p["dims"]
    names = sorted({_short(m) for m in members}, key=len)
    head = names[0] + (f"  (+{len(names) - 1})" if len(names) > 1 else "")
    ffn = "MoE" if p["slots"]["ffn"].startswith("moe") else "dense"
    mixer = p["slots"].get("mixer", "softmax")
    g = f'<BR/><FONT COLOR="#777777" POINT-SIZE="8">{line3}</FONT>' if line3 else ""
    return f"<<B>{head}</B><BR/>h{d['hidden']} · L{d['layers']} · {ffn} · {mixer}{g}>"


def fig_paths(items, out="graph/fig_diffusion_paths"):
    diff = [(r, p) for r, p in items if p["slots"]["gen"] == "diffusion"]
    ar = [(r, p) for r, p in items if p["slots"]["gen"] == "ar"]
    key = lambda p: tuple(p["dims"][k] for k in SHAPE) + (p["slots"]["ffn"],)
    groups = defaultdict(list)
    for r, p in diff:
        groups[key(p)].append((r, p))
    nodes = [(k, [r for r, _ in v], v[0][1]) for k, v in groups.items()]
    P = [p for _, _, p in nodes]
    cap = emul_matrix(P, mode="payable")                # cap[i, j]: node j emulates node i (headline)
    exact = emul_matrix(P, mode="exact")
    n = len(P)
    up = cap & ~np.eye(n, dtype=bool)
    same = up & up.T                                    # equal capacity (shouldn't happen across shapes)
    strict = up & ~same
    # Hasse: keep i -> j unless some k sits strictly between
    hasse = strict & ~((strict.astype(int) @ strict.astype(int)) > 0)
    tops = [i for i in range(n) if not strict[i].any()]
    twins = defaultdict(list)
    for r, p in ar:
        twins[key(p)].append(r)

    size = lambda p: p["dims"]["hidden"] * p["dims"]["layers"]
    order = sorted(range(n), key=lambda i: size(P[i]))
    L = ['digraph G {', 'rankdir=BT; nodesep=0.35; ranksep=0.55; splines=true; newrank=true;',
         'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#555555"];',
         'edge [color="#888888", arrowsize=0.7];',
         'labelloc="t"; fontname="Helvetica"; fontsize=15;',
         'label="Diffusion LMs: who emulates whom (arrow A -> B: B emulates A)\\n'
         'green = exact edge; gold = frontier top; grey line = the AR model it shares tensors with: the diffusion shape emulates it (causal mask), never the reverse";',
         'subgraph cluster_diff { label="diffusion paradigm (bidirectional denoising)"; style="rounded,dashed";'
         ' color="#4a7bd0"; fontcolor="#4a7bd0"; fontsize=12;']
    for i in order:
        k, mem, p = nodes[i]
        fill = "#f6d365" if i in tops else "#dbe7fb"
        line3 = _lineage(mem, twins.get(k), {r for r, _ in diff})
        L.append(f'  d{i} [label={_label(mem, p, line3)}, fillcolor="{fill}"'
                 f'{", penwidth=2" if i in tops else ""}];')
    L.append('}')
    for i in range(n):
        for j in range(n):
            if hasse[i, j]:
                col = "#2e9e5b" if exact[i, j] else "#888888"
                L.append(f'  d{i} -> d{j} [color="{col}", penwidth={1.6 if exact[i, j] else 1.0}];')
    L.append('}')
    open(out + ".dot", "w").write("\n".join(L))
    subprocess.run(["dot", "-Tsvg", out + ".dot", "-o", out + ".svg"], check=True)
    # this graphviz build has no png renderer: rasterize the svg instead
    subprocess.run(["rsvg-convert", "-z", "1.6", "-b", "white", out + ".svg", "-o", out + ".png"], check=True)
    return dict(models=len(diff), shapes=n, tops=[nodes[i][1] for i in tops],
                edges=int(hasse.sum()), exact_edges=int((hasse & exact).sum()),
                twins=sum(1 for k, _, _ in nodes if twins.get(k)))


def fig_mask(out="graph/fig_diffusion_mask.png", T=8, steps=4, seed=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2), gridspec_kw={"width_ratios": [1, 1.35]})
    causal, full = np.tril(np.ones((T, T))), np.ones((T, T))
    for a, m, t in ((ax[0, 0], causal, "causal mask (autoregressive)"), (ax[1, 0], full, "full mask (diffusion)")):
        a.imshow(m, cmap="Blues", vmin=0, vmax=1.6)
        a.set_title(t, fontsize=11)
        a.set_xlabel("attends to position")
        a.set_ylabel("query position")
        a.set_xticks(range(T))
        a.set_yticks(range(T))
    # decoding schedules: which positions are revealed at each step
    ar = np.full((T, T), np.nan)
    for s in range(T):
        ar[s, : s + 1] = np.arange(s + 1)
    order = rng.permutation(T)
    per = np.array_split(order, steps)
    df = np.full((steps, T), np.nan)
    revealed = []
    for s, chunk in enumerate(per):
        revealed += list(chunk)
        for pos in revealed:
            df[s, pos] = next(i for i, c in enumerate(per) if pos in c)
    for a, m, t, ys in ((ax[0, 1], ar, f"AR decoding: 1 token / step ({T} steps)", T),
                        (ax[1, 1], df, f"diffusion decoding: parallel unmasking ({steps} steps)", steps)):
        a.imshow(np.isnan(m), cmap="Greys", vmin=0, vmax=3, aspect="auto")
        for s in range(ys):
            for pos in range(T):
                if np.isnan(m[s, pos]):
                    a.text(pos, s, "[M]", ha="center", va="center", fontsize=8, color="#888888")
                else:
                    new = m[s, pos] == (s if ys == steps else s)
                    a.text(pos, s, "tok", ha="center", va="center", fontsize=8,
                           color="#c0392b" if new else "#2c3e50", fontweight="bold" if new else "normal")
        a.set_title(t, fontsize=11)
        a.set_xlabel("sequence position")
        a.set_ylabel("step")
        a.set_xticks(range(T))
        a.set_yticks(range(ys))
    fig.suptitle("Same weights, two decode modes: causal is the full mask restricted (AR <= diffusion),\n"
                 "the reverse is the one-way wall (the universal host runs either by switching the mask; "
                 "red = written this step)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    info = fig_paths(points())
    fig_mask()
    print(f"diffusion: {info['models']} models -> {info['shapes']} shapes, {info['edges']} Hasse edges "
          f"({info['exact_edges']} exact), {info['twins']} with AR twins; frontier tops: "
          + "; ".join(", ".join(_short(m) for m in t) for t in info["tops"]))


if __name__ == "__main__":
    main()
