"""Frontier effects of the construction candidates, the ignorable-by-argument items and token splitting.

Each item is a modification of the headline relation (verify/levers/rel.py components), measured alone and on
top of R0 (the rules as stated: window sizes and norm epsilons as free host config, verify/levers/relax.py).
Items that a CORRECT reading of the code makes stricter (qk-norm variants) are measured too: they move the
frontier the other way.

Run:  python -m verify.levers.extras      # writes verify/levers/out/extras.json
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from verify.levers.family_code import CACHE, load_caps
from verify.levers.relax import R0, apply
from verify.levers.rel import AND, components, frontier_set, left, lm, mg, points, short

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DRAFT = re.compile(r"dflash|dspark|eagle-?3|assistant|-spec-|medusa|draft|mtp-head", re.I)   # not RWKV-5 "Eagle"


def qknorm_variant(mt):
    """'rms_head' / 'rms_whole' / 'ln_head' read from the family's q-norm constructor (None: no in-library code)."""
    d = {"mimo_v2": "mimo_v2_flash"}.get(mt, mt)
    for cand in (d, d[:-5] if d.endswith("_text") else d):
        path = os.path.join(CACHE, f"modeling_{cand}.py")
        if os.path.exists(path):
            src = open(path).read()
            m = re.search(r"self\.(q_norm|q_layernorm|query_layernorm) = (.*)", src)
            if not m:
                return None
            line = m.group(2) + src[m.end():m.end() + 200]
            if "LayerNorm(" in m.group(2) and "RMS" not in m.group(2):
                return "ln_head"
            if "num_attention_heads * self.head_dim" in line or "hidden_size" in line.split(")")[0]:
                return "rms_whole"
            return "rms_head"
    return None


def run():
    P, cfg = points()
    fam = load_caps()["families"]
    C = components(P)
    base = AND(C)
    reps = frontier_set(base, P)
    PH0 = apply(P, cfg, fam, R0)
    C0 = components(P, PH0)
    cnt = np.array([len(p["members"]) for p in P])
    res = {}

    def rep(name, ok_b, ok_r, P_=None, reps_=None):
        P_ = P_ or P
        out = {}
        for tag, ok, rf in (("alone", ok_b, reps), ("on_R0", ok_r, frontier_set(AND(C0), P))):
            if ok is None:
                continue
            new = frontier_set(ok, P_)
            gone = left(rf if P_ is P else frontier_set(ok_b, P_), ok) if P_ is P else []
            out[tag] = dict(frontier=len(new), left=[short(P_, i) for i in gone][:20])
        res[name] = out
        print(f"{name:70s} " + "  ".join(f"{k} {v['frontier']}" for k, v in out.items()))

    rep("baseline / R0", base, AND(C0))
    # --- qk-norm variants (a correction: the slot lumps per-head RMS, whole-projection RMS and per-head LayerNorm)
    var = []
    for p in P:
        vs = {qknorm_variant(str(lm(cfg[n]).get("model_type"))) for n in p["members"]} - {None}
        var.append(next(iter(vs)) if len(vs) == 1 else None)
    qT = np.array([p["slots"]["qknorm"] == "yes" for p in P])
    vv = np.array([v or "" for v in var])
    known = vv != ""
    mismatch = qT[:, None] & qT[None, :] & known[:, None] & known[None, :] & (vv[:, None] != vv[None, :])
    res["qknorm_variants_count"] = {v: int((vv == v).sum()) for v in set(vv.tolist())}
    rep("correction: q-norm variant must match (rms_head / rms_whole / ln_head)", base & ~mismatch,
        AND(C0) & ~mismatch)
    # --- post-only (post-norm, no pre-norm) target on a sandwich host: the final norm needs the residual large,
    # the linearised pre-norms need it small (one residual scale) -> upper bound only
    post_only = {"olmo2", "olmo3", "exaone4", "exaone4_5", "flex_olmo", "olmo_hybrid"}
    po = np.array([bool({str(lm(cfg[n]).get("model_type")) for n in p["members"]} & post_only) for p in P])
    rep("[upper bound] post-only targets free on any post-norm host", AND(C, replace={
        "feat:post_norm:target": C["feat:post_norm:target"] | po[:, None]}), AND(C0, replace={
        "feat:post_norm:target": C0["feat:post_norm:target"] | po[:, None]}))
    rep("[upper bound] post-norm target wall dropped", AND(C, drop=("feat:post_norm:target",)),
        AND(C0, drop=("feat:post_norm:target",)))
    # --- KV-head packing: one host KV head holds floor(hd_h / hd_t) target KV heads (q zero on the other blocks)
    kht, qgt = np.array([p["dims"].get("kv_heads", 0) for p in P]), np.array([p["dims"].get("q_per_kv", 0) for p in P])
    hd = np.array([p["dims"].get("head_dim", 0) for p in P])
    pack = np.maximum(hd[None, :] // np.maximum(hd[:, None], 1), 1)
    eff_h = kht[None, :] * pack
    have = (kht > 0) & (qgt > 0)
    need = kht[:, None] * -(-qgt[:, None] // np.maximum(qgt[None, :] * pack, 1))
    packed = ~(have[:, None] & have[None, :]) | (eff_h >= kht[:, None])
    rep("KV-head packing (head_dim blocks) replaces the KV-head-count rule",
        AND(C, replace={"kv_heads": packed}), AND(C0, replace={"kv_heads": packed}))
    # --- K = V host layer plays an own-V layer with [k; v] in a 2x-wide head (keqv caps -> 2x head_dim)
    rep("[upper bound] K=V host layers play own-V layers (keqv rule dropped)", AND(C, drop=("keqv_caps",)),
        AND(C0, drop=("keqv_caps",)))
    # --- hybrid <- hybrid across recurrent families (universal gated-delta form): upper bound
    rep("[upper bound] recurrent family wall dropped (rf target side)", AND(C, drop=("feat:rf:target",)),
        AND(C0, drop=("feat:rf:target",)))
    # --- learned sinks via the start token: upper bound
    rep("[upper bound] sink target on a plain host", AND(C, drop=("cat:sink",)), AND(C0, drop=("cat:sink",)))
    # --- output-side bias via a start-token 'bias head': upper bound (o / v bias only is exact; q, MLP-input,
    # LayerNorm-beta biases are not)
    rep("[upper bound] all bias walls dropped", AND(C, drop=("feat:bias_out:target", "feat:bias_qkv:target")),
        AND(C0, drop=("feat:bias_out:target", "feat:bias_qkv:target")))
    # --- ignorable: DSA (identical while the context <= index_topk)
    rep("ignore DSA (context <= index_topk)", AND(C, drop=("feat:dsa:target", "feat:dsa:host")),
        AND(C0, drop=("feat:dsa:target", "feat:dsa:host")))
    # --- ignorable: final-logit softcap / MTP: already payable / free -> nothing to drop
    fs = sum(1 for p in P if p["slots"]["softcap"] == "yes")
    res["softcap_points"] = fs
    rep("ignore softcap (already a payable regime)", AND(C, drop=("cat:softcap",)), AND(C0, drop=("cat:softcap",)))
    rep("ignore MTP (already free both ways)", AND(C, drop=("cat:mtp",)), AND(C0, drop=("cat:mtp",)))
    # --- exclude draft / assistant models: recompute on the points that have a non-draft member
    keep = [i for i, p in enumerate(P) if any(not DRAFT.search(n) for n in p["members"])]
    Pk = [P[i] for i in keep]
    drafts = sorted({short(P, i) for i in reps if i not in keep})
    ixk = np.ix_(keep, keep)
    nb, nr = len(frontier_set(base[ixk], Pk)), len(frontier_set(AND(C0)[ixk], Pk))
    res["exclude drafts"] = dict(points_removed=len(P) - len(keep), frontier_drafts=drafts,
                                 alone=dict(frontier=nb), on_R0=dict(frontier=nr))
    print(f"{'exclude draft / assistant models (' + str(len(P) - len(keep)) + ' points)':70s} alone {nb}  on_R0 {nr}")
    # --- token splitting (optimistic): target kv_cap, q_cap, qheads divided by k
    for k in (2, 4, 8):
        for with_heads in (False, True):
            R = {}
            for nm, Cx in (("b", C), ("r", C0)):
                dimT = lambda key: np.array([p["dims"].get(key, 0) for p in P])
                dimH = lambda key: np.array([p["dims"].get(key, 0) for p in (P if nm == "b" else PH0)])
                mlaT = np.array(["MLA" in mg.branches(p["slots"]["kv"]) for p in P])
                mlaH = np.array(["MLA" in mg.branches(p["slots"]["kv"]) for p in (P if nm == "b" else PH0)])
                pen = np.where(mlaT[:, None] & ~mlaH[None, :], mg.MLA_PENALTY, 1)
                kvt = -(-dimT("kv_cap") // k)
                rp = {"kv_cap": dimH("kv_cap")[None, :] >= kvt[:, None],
                      "mla_penalty": (pen == 1) | (dimH("kv_cap")[None, :] >= kvt[:, None] * pen),
                      "q_cap": dimH("q_cap")[None, :] >= -(-dimT("q_cap") // k)[:, None],
                      "qheads": dimH("qheads")[None, :] >= -(-dimT("qheads") // k)[:, None]}
                if with_heads:
                    kt = -(-kht // k)
                    nd = kt[:, None] * -(-qgt[:, None] // np.maximum(np.array([p["dims"].get("q_per_kv", 0) for p in (P if nm == "b" else PH0)])[None, :], 1))
                    kvh_h = np.array([p["dims"].get("kv_heads", 0) for p in (P if nm == "b" else PH0)])
                    rp["kv_heads"] = ~(have[:, None] & have[None, :]) | (kvh_h[None, :] >= nd)
                R[nm] = AND(Cx, replace=rp)
            rep(f"token splitting k={k} (kv_cap, q_cap, qheads / k{' + kv_heads / k' if with_heads else ''})",
                R["b"], R["r"])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "extras.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
