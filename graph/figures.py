"""Matplotlib figures for the writeup:

  fig_coverage.png  -- every model as a point (params proxy vs # models it emulates),
                       frontier highlighted, Qwen-72B annotated: the top emulator
                       is not the biggest model.
  fig_minimal.png   -- the minimal universal emulator (the join) vs the top real models
                       on each capacity axis: it out-maxes all of them at once.

Run:  python -m graph.figures
"""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph.minimal_emulator import join, params
from graph.currency_curve import n_frontier
from graph.model_graph import (_dedup_points, _junky, classify, dims_of, emul_matrix,
                               iter_json_configs)

JSON = "graph/hub_10k.json"
META = "graph/model_meta.json"


def load():
    models = []
    for name, cfg in iter_json_configs(JSON):
        if _junky(name):
            continue
        c = classify(cfg)
        if c is None:
            continue
        d = dims_of(cfg)
        if d["hidden"] and d["layers"]:
            models.append({"name": name, "slots": c, "dims": d})
    return models


def prm(d):
    return d["layers"] * 4 * d["hidden"] ** 2 + 3 * d["hidden"] * d["ffn_units"]


_META = None
Q4 = ("nvfp4", "mxfp4", "-fp4", "w4a", "int4", "4bit", "dq4", "-fp4-")


def _meta():
    global _META
    if _META is None:
        try:
            _META = json.load(open(META))
        except FileNotFoundError:
            _META = {}
    return _META


def _nominal(n):                                  # total-B from the name, ignoring the A##B active tag
    n = re.sub(r"[Aa]\d+(?:\.\d+)?[Bb]", "", n.split("/")[-1])
    m = re.search(r"(?<![.\dxX])(\d{2,4})[Bb](?![a-z])", n)
    return float(m.group(1)) if m else None


_VL = None
_RANK = None
# packed-weight checkpoints (MLX / GPTQ / AWQ / bnb / IQ quants): their safetensors total counts packed
# words, not parameters, so it under-counts (LFM2.5-1.2B-Instruct-MLX-6bit: 256M)
_PACKED = re.compile(r"\d+-?bits?\b|\biq\d|gptq|awq|bnb|exl\d|mlx")
_VL_KEYS = ("vision_config", "vision_tower", "visual", "vision_encoder", "mm_vision_tower", "audio_config",
            "speech_config")
_VL_ARCH = ("ConditionalGeneration", "Vision", "VL", "Llava", "Idefics", "ImageText", "Multimodal")


def _snapshot():
    """(names of the snapshot's multimodal repos, {name: download rank}). A multimodal checkpoint's
    safetensors total counts its vision / audio tower; the shape class (the text LM) does not."""
    global _VL, _RANK
    if _VL is None:
        _VL, _RANK = set(), {}
        for k, (name, cfg) in enumerate(iter_json_configs(JSON)):     # snapshot order = downloads
            _RANK.setdefault(name, k)
            arch = " ".join(cfg.get("architectures") or [])
            if any(x in cfg for x in _VL_KEYS) or any(a in arch for a in _VL_ARCH):
                _VL.add(name)
    return _VL, _RANK


def real_prm(p):
    """Real total params for a class: the exact HF-safetensors count of its most-downloaded
    unquantised TEXT-LM member, if it has one. Not the largest member's: a class can hold a
    multimodal checkpoint whose total counts its vision tower (granite-docling-258M in
    SmolLM2-135M's class: 258M for a 135M LM) or a fine-tune with a grown vocab; and packed
    MLX / GPTQ / AWQ / IQ repos under-count. Otherwise the earlier rule (_real_prm_rule) over
    the text members (all members if every one is multimodal)."""
    meta, names = _meta(), p["members"]
    hf = lambda n: meta.get(n, {}).get("hf_total")
    vl, rank = _snapshot()
    text = sorted((n for n in names if n not in vl), key=lambda n: rank.get(n, len(rank)))
    clean = [hf(n) for n in text if hf(n) and not any(q in n.lower() for q in Q4) and not _PACKED.search(n.lower())]
    if clean:
        return clean[0]
    return _real_prm_rule(p, text or names)


def _real_prm_rule(p, names):
    """exact HF-safetensors count from a non-quant member if we have one (the largest); else the
    name-nominal (which un-packs NVFP4); else the validated proxy for softmax; else a 4-bit-unpack
    estimate. Fixes the hybrid over-count (Nemotron 7T→550B) without disturbing the softmax cluster."""
    meta = _meta()
    hf = lambda n: meta.get(n, {}).get("hf_total")
    clean = max([hf(n) for n in names if hf(n) and not any(q in n.lower() for q in Q4)], default=None)
    if clean:
        return clean
    nom = next((_nominal(n) for n in names if _nominal(n)), None)
    if nom:
        return nom * 1e9
    if p["slots"]["mixer"] == "softmax":
        return prm(p["dims"])
    anyhf = max([hf(n) for n in names if hf(n)], default=None)
    return anyhf * 1.65 if anyhf else prm(p["dims"])


def fig_coverage(models):
    P = _dedup_points(models)
    ok = emul_matrix(P, mode="payable")                      # HEADLINE: capacity + payable type walls
    counts = np.array([len(p["members"]) for p in P])
    emulated = (ok * counts[:, None]).sum(0)                 # models each point emulates
    pr = np.array([real_prm(p) for p in P], float)
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)
    front = ~strict.any(1)
    nfront = n_frontier(ok)                                  # frontier classes (collapse ties)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(pr[~front] / 1e9, emulated[~front], s=9, c="#c9d3df", label="model", zorder=1)
    ax.scatter(pr[front] / 1e9, emulated[front], s=34, c="#c0392b",
               edgecolor="#7b1a10", linewidth=0.4, label=f"emulator frontier ({nfront})", zorder=3)
    for i, p in enumerate(P):
        nm = p["members"][0].split("/")[-1]
        if nm.startswith("Qwen-72B") or nm.startswith("Llama-3.1-405B"):
            ax.annotate(nm, (pr[i] / 1e9, emulated[i]), fontsize=8, xytext=(6, -2),
                        textcoords="offset points", color="#111")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (rough proxy, B)")
    ax.set_ylabel("# models it can emulate")
    ax.set_title("The top emulator is not the biggest model")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig("graph/fig_coverage.png", dpi=130)
    print("wrote graph/fig_coverage.png")


def fig_minimal(models):
    j = join(models)                    # per-axis max over EVERY model (all mixers, AR + diffusion)
    axes = [("hidden", "hidden"), ("layers", "layers"), ("kv_cap", "kv/token"),
            ("qheads", "q-heads"), ("ffn_total", "FFN / layer")]
    picks = {}
    for m in models:
        picks[m["name"].split("/")[-1]] = m["dims"]
    ref = ["Qwen-72B", "Llama-3.1-405B-FP8", "Kimi-K2.5", "DeepSeek-V2"]
    ref = [r for r in ref if any(r in k for k in picks)]
    refdims = [next(picks[k] for k in picks if r in k) for r in ref]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(axes))
    w = 0.15
    series = [("minimal emulator", j, "#c0392b")] + \
             [(r, d, c) for r, d, c in zip(ref, refdims, ["#3b6", "#38c", "#a80", "#96c"])]
    for i, (label, d, col) in enumerate(series):
        vals = [d[k] for k, _ in axes]
        ax.bar(x + (i - len(series) / 2) * w + w / 2, vals, w, label=label, color=col)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in axes])
    ax.set_ylabel("value (log)")
    ax.set_title("The minimal universal emulator out-maxes every real model on every axis")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("graph/fig_minimal.png", dpi=130)
    print("wrote graph/fig_minimal.png")


def fig_funnel(models):
    P = _dedup_points(models)
    ok = emul_matrix(P, mode="payable")                      # HEADLINE: capacity + payable type walls
    nfront = n_frontier(ok)                                  # frontier classes (collapse ties)
    stages = [("configs", len(json.load(open(JSON)))), ("decoder LMs", len(models)),
              ("arch+size shapes", len(P)), ("frontier", nfront)]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    cols = ["#c9d3df", "#9db4cc", "#5f86ad", "#c0392b"]
    for i, (label, v) in enumerate(stages):
        half = 0.5 * (0.15 + 0.85 * (i == 0))                # visual taper (log-ish)
        w = 0.9 * (np.log10(v) / np.log10(stages[0][1]))
        ax.barh(-i, w, height=0.62, color=cols[i], zorder=2)
        ax.text(w + 0.02, -i, f"{v:,}  {label}", va="center", fontsize=11, color="#111")
    ax.set_xlim(0, 1.4); ax.set_ylim(-3.7, 0.7)
    ax.axis("off")
    ax.set_title("From the Hub to the frontier", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig("graph/fig_funnel.png", dpi=130)
    print("wrote graph/fig_funnel.png")


def fig_updown(models):
    P = _dedup_points(models)
    ok = emul_matrix(P, mode="payable")                      # HEADLINE: capacity + payable type walls
    counts = np.array([len(p["members"]) for p in P])
    out = (ok * counts[:, None]).sum(0)                      # # each spec emulates (out-degree)
    inn = (ok * counts[None, :]).sum(1)                      # # that emulate each spec (in-degree)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(out + 1, inn + 1, s=8, c="#5f86ad", alpha=0.5, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("# models it emulates (out)")
    ax.set_ylabel("# models that emulate it (in)")
    ax.set_title("Every model: reach out vs reachability in (they anticorrelate)")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig("graph/fig_updown.png", dpi=130)
    print("wrote graph/fig_updown.png")


if __name__ == "__main__":
    models = load()
    print(f"{len(models)} models loaded")
    fig_funnel(models)
    fig_coverage(models)
    fig_updown(models)
    fig_minimal(models)
