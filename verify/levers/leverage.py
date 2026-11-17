"""Wall leverage: for every blocking rule of the headline relation, the frontier with THAT rule alone
relaxed (every other rule kept). Rules are the named components of verify/levers/rel.py (whose AND is
asserted equal to emul_matrix(mode="payable")), the per-pair cells of each categorical slot, each
FEATURE side, and the coupled rules that live in several components (the window ZERO_EDGES charge and
its per-layer-kind twin, the MoE 2x depth, K = V, the near-gap, the CLA carry, the act floor threshold).

Run:  python -m verify.levers.leverage        # writes verify/levers/out/leverage.{json,md}
"""
from __future__ import annotations

import json
import os

import numpy as np

from verify.levers.rel import AND, components, check, frontier_set, left, mg, short

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

ATT = ("FU", "FT", "LU", "LT")


def reach_plus(extra):
    r = {k: tuple(v) for k, v in mg.KIND_REACH.items()}
    for h, ts in extra.items():
        r[h] = tuple(dict.fromkeys(r[h] + tuple(ts)))
    return r


REACH_WINDOW = {"LU": ("FU", "FT"), "LT": ("FT",)}            # a sliding layer plays a full one
REACH_KEQV = {"FT": ("FU",), "LT": ("LU",)}                     # a K = V layer plays an own-V one
REACH_GAP = {h: ("R",) for h in ATT}                            # attention plays a recurrent layer
REACH_CLA = {h: ("S",) for h in ATT}                            # own-KV attention plays a KV-sharing layer
REACH_REC_ATT = {"R": ATT}                                      # a recurrent layer plays attention (nonsense; leverage only)


def coupled(P, C):
    """{name: (ok, note)} for the rules that live in several components."""
    out = {}
    W = components(P, kind_reach=reach_plus(REACH_WINDOW), window_all_layers=True)
    out["window: sliding-window host layers count as full (ZERO_EDGES window + kinds LU->FU)"] = AND(W)
    K = components(P, kind_reach=reach_plus(REACH_KEQV))
    out["K=V: a K=V host layer plays an own-V layer (keqv caps + kinds)"] = AND(K, drop=("keqv_caps",))
    G = components(P, kind_reach=reach_plus(REACH_GAP))
    out["near-gap: softmax plays recurrent layers (nonplain:mixer + rf target + kinds)"] = AND(
        G, drop=("nonplain:mixer", "feat:rf:target"))
    S = components(P, kind_reach=reach_plus(REACH_CLA))
    out["CLA carry: own-KV attention plays KV-sharing layers (nonplain:cla + kinds)"] = AND(S, drop=("nonplain:cla",))
    R = components(P, kind_reach=reach_plus(REACH_REC_ATT))
    out["[implausible] recurrent host layers play attention (kinds R->attn; mixer depth)"] = AND(R)
    M1 = components(P, moe_factor=1)
    out["MoE<-dense: no 2x depth (moe factor 1)"] = AND(M1)
    out["MoE<-dense: any router (top-1 unit-gate rule dropped)"] = AND(C, drop=("moe_dense_top1",))
    out["MoE<-dense: both (no 2x depth, any router)"] = AND(M1, drop=("moe_dense_top1",))
    for fm in (0.01, 0.015, 0.02, 0.05, 0.15, 1e9):
        A = components(P, act_floor_max=fm)
        out[f"act floor threshold ACT_FLOOR_MAX -> {fm:g}"] = AND(A)
    out["all MLA rules (mla_wall + latents + penalty)"] = AND(C, drop=("mla_wall", "mla_latents", "mla_penalty"))
    out["kv_cap incl. MLA penalty"] = AND(C, drop=("kv_cap", "mla_penalty"))
    out["hidden incl. norm mirror"] = AND(C, drop=("hidden", "norm_mirror"))
    out["depth (eff + param + kinds)"] = AND(C, drop=("depth_eff", "depth_param", "kinds_eff", "kinds_param"))
    out["qknorm both ways + post_norm both sides"] = AND(C, drop=("cat:qknorm", "feat:post_norm:target",
                                                                 "feat:post_norm:host"))
    out["all FEATURES"] = AND(C, drop=tuple(k for k in C if k.startswith("feat:")))
    out["all bias features (bias_qkv + bias_out)"] = AND(C, drop=("feat:bias_qkv:target", "feat:bias_out:target"))
    return out


def pair_relax(P, C):
    """Per (slot, target value, host value) cell of each categorical slot, and each feature side."""
    out = {}
    for slot in mg.CAT_SLOTS:
        m = C[f"cat:{slot}"]
        a = np.array([p["slots"][slot] for p in P])
        for ta in sorted(set(a.tolist())):
            for hb in sorted(set(a.tolist())):
                cell = (a == ta)[:, None] & (a == hb)[None, :]
                if (cell & ~m).any():
                    out[f"cat:{slot} {ta}<-{hb}"] = AND(C, replace={f"cat:{slot}": m | cell})
    return out


def run():
    P, C = check()
    base = AND(C)
    reps = frontier_set(base, P)
    cnt = np.array([len(p["members"]) for p in P])
    rows = []

    def add(kind, name, ok):
        new = frontier_set(ok, P)
        gone = left(reps, ok)
        rows.append(dict(kind=kind, rule=name, frontier=len(new), freed=len(reps) - len(new),
                         left=[short(P, i) for i in sorted(gone, key=lambda i: -cnt[ok[:, i]].sum())]))

    for k in C:
        add("component", k, AND(C, drop=(k,)))
    for k, ok in coupled(P, C).items():
        add("coupled", k, ok)
    for k, ok in pair_relax(P, C).items():
        add("pair", k, ok)
    rows.sort(key=lambda r: (-r["freed"], r["rule"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "leverage.json"), "w") as f:
        json.dump(dict(base=len(reps), points=len(P), models=int(cnt.sum()), rows=rows), f, indent=1)
    lines = [f"Baseline headline frontier: {len(reps)} ({len(P)} shapes, {int(cnt.sum())} models).", "",
             "| rule relaxed alone | kind | frontier | freed | examples (left the frontier) |",
             "|---|---|---:|---:|---|"]
    for r in rows:
        if r["freed"] <= 0 and not r["left"]:
            continue
        ex = ", ".join(f"`{x}`" for x in r["left"][:6]) + (f" +{len(r['left']) - 6}" if len(r["left"]) > 6 else "")
        lines.append(f"| {r['rule']} | {r['kind']} | {r['frontier']} | {r['freed']} | {ex} |")
    zero = [r["rule"] for r in rows if r["freed"] <= 0 and not r["left"]]
    lines += ["", "No effect alone: " + ", ".join(f"`{z}`" for z in zero) + "."]
    with open(os.path.join(OUT, "leverage.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:70]))


if __name__ == "__main__":
    run()
