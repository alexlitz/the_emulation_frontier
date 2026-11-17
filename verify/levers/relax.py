"""Config-value levers: the frontier when a HOST may set config values its family's code reads.

Host-side transforms of the points (rel.components(P, PH)); targets are never changed (a target is the
function it computes). Each lever is grounded in the family's code (verify/levers/family_code.py: the config
read, transformers main) unless marked otherwise.

  window      sliding_window >= context: every sliding layer of a host becomes a full layer (the RoPE table,
              shapes, sinks of the layer are unchanged). Code-checked for every in-library family: the window
              is only the mask's parameter (and, in Cohere-2 / EXAONE-4 / Kolibri, the RoPE-on switch, which an
              int window keeps on). DeepSeek-V4-style compressed layers are NOT converted: with a full window
              they still attend to the compressor's extra entries; only their compress-ratio-0 layers are.
  eps         per-norm effective epsilon (all norm eps are allowed config values): a shared eps E plus a
              power-of-two scale S on a norm's input gives that norm eps_eff = E / S^2 (bit-exact: power-of-two
              scaling commutes with fp rounding). Pre-norms keep eps_eff = the target's eps (residual scaled by
              sqrt(E / eps_t)); a norm the target lacks gets eps_eff >> its input (linear regime: fl(ms + E) = E),
              its gain or the next linear map absorbs the constant. Switches OFF, on the host: q / k norms (with a
              gain), the post-norms of a SANDWICH host (pre- and post-norm; a post-only host would lose its only
              norm), the weight-free Gemma-4 v_norm (o_proj absorbs), MLA latent norms (-> plain low-rank attention).
              verify/levers/eps_lever.py checks it numerically.
  dsa_off     index_topk >= context: the DSA indexer keeps every key (dense attention).
  group_off   topk_group = n_group: group-limited routing off.     group_on  n_group / topk_group set on a family
              whose router code has groups.
  nope_off    no_rope_layers all RoPE.                                  nope_on  set per-layer NoPE.
  topk        num_experts_per_tok raised (a MoE host's active FFN up to its total).
  partial     partial_rotary_factor raised to 1 (rot / pass split code; not MLA, whose rotary width is a shape).
  act         hidden_act set to any parameter-free ACT2FN activation (SiLU, GeLU, ReLU, relu2, mish).
  parallel    use_parallel_residual / parallel_attn switched.
  scoring     scoring_func switched (softmax <-> sigmoid router).

Run:  python -m verify.levers.relax      # writes verify/levers/out/relax.{json,md}
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from verify.levers.family_code import load_caps
from verify.levers.rel import AND, components, frontier_set, left, lm, mg, points, short

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SANDWICH = {"gemma2", "gemma3", "gemma3_text", "gemma3n_text", "gemma4_text", "gemma4_unified_text", "diffusion_gemma",
            "diffusion_gemma_text", "glm4", "afmoe", "muse_glimmer", "muse_glimmer_text", "muse_spark", "muse_spark_text",
            "cost_wise_gemma"}   # pre- AND post-sublayer norms (code: input / pre_* and post_* norms both built)
COMPRESSED = {"deepseek_v4", "deepseek_v41_text", "ernie5_5_moe"}      # compressed-KV layers beside the window
PARAM_FREE_ACTS = ("SiLU", "GeLU", "ReLU", "relu2", "mish")
WEIGHTLESS_QKNORM = {"nanochat_gpt", "nanochat"}


def _families(p, cfg):
    return [str(lm(cfg[n]).get("model_type")) for n in p["members"]]


def _caps(p, cfg, fam):
    """Union of the code capabilities of the point's members' families (any member is a host of that shape)."""
    out = set()
    for mt in _families(p, cfg):
        out |= set((fam.get(mt) or {}).get("caps", {}))
    return out


def _in_lib(p, cfg, fam):
    return any((fam.get(mt) or {}).get("code") for mt in _families(p, cfg))


def _convertible_compressed(c, n_param):
    """Layers of a compressed-attention (DeepSeek-V4-style) config that are pure sliding-window layers."""
    lt = c.get("layer_types")
    src = c.get("kv_source_layer_ids")
    own = (lambda i: i in src) if isinstance(src, list) and src else (lambda i: True)
    if isinstance(lt, list):
        return sum(1 for i, x in enumerate(lt) if x == "sliding_attention" and own(i))
    cr = c.get("compress_ratios")
    if isinstance(cr, list):
        return sum(1 for i, x in enumerate(cr[:n_param]) if x == 0 and own(i))
    return 0


def window_full(p, cfg, fam, verified_only=True):
    """Host view of p with its sliding window set to the full context (None if nothing changes)."""
    d = p["dims"]
    lu, lt = d.get("kLU_param", 0), d.get("kLT_param", 0)
    if not (lu or lt or p["slots"]["window"] == "local"):
        return None
    mts = _families(p, cfg)
    if verified_only and "window_cfg" not in _caps(p, cfg, fam):
        return None
    q = copy.deepcopy(p)
    dd = q["dims"]
    c = lm(cfg[p["members"][0]])
    if set(mts) & COMPRESSED:
        mv = min(lu, _convertible_compressed(c, d.get("param_layers", 0)))
        if not mv:
            return None
        r = d.get("kLU_eff", 0) / max(lu, 1)
        for w, k in (("param", mv), ("eff", int(round(mv * r)))):
            dd[f"kLU_{w}"] -= k
            dd[f"kFU_{w}"] = dd.get(f"kFU_{w}", 0) + k
            dd[f"full_{w}"] = dd.get(f"full_{w}", 0) + k
        if dd.get("kLU_param", 0) == 0 and dd.get("kLT_param", 0) == 0:
            q["slots"]["window"] = "full"
        return q
    for w in ("eff", "param"):
        dd[f"kFU_{w}"] = dd.get(f"kFU_{w}", 0) + dd.get(f"kLU_{w}", 0)
        dd[f"kFT_{w}"] = dd.get(f"kFT_{w}", 0) + dd.get(f"kLT_{w}", 0)
        dd[f"kLU_{w}"] = dd[f"kLT_{w}"] = 0
        dd[f"full_{w}"] = dd.get(f"attn_{w}", dd.get(f"full_{w}", 0))
    q["slots"]["window"] = "full"
    return q


def apply(P, cfg, fam, levers, window_verified=True):
    """Host views PH of P with the given levers applied."""
    PH = []
    for p in P:
        q = p
        if "window" in levers or "window_all" in levers:
            w = window_full(p, cfg, fam, verified_only=("window_all" not in levers) and window_verified)
            q = w if w is not None else p
        q = copy.deepcopy(q)
        caps, mts, s = _caps(p, cfg, fam), set(_families(p, cfg)), q["slots"]
        feats = set(filter(None, s.get("feats", "").split("+")))
        q["sw_on"], q["sw_off"] = set(), set()
        lib = _in_lib(p, cfg, fam)
        if "eps" in levers and lib and "norm_eps_cfg" in caps:
            if s["qknorm"] == "yes" and not (mts & WEIGHTLESS_QKNORM):
                s["qknorm"] = "yes|no"
            if "post_norm" in feats and mts & SANDWICH:
                q["sw_off"].add("post_norm")
            if "vnorm" in feats:
                q["sw_off"].add("vnorm")
            if "MLA" in mg.branches(s["kv"]):
                q["mla_eps"] = True
        if "qknorm_switch" in levers and "qknorm_switch" in caps and s["qknorm"] == "yes":
            s["qknorm"] = "yes|no"
        if "dsa_off" in levers and "index_topk" in caps:
            q["sw_off"].add("dsa")
        if "group_off" in levers and "router_group_cfg" in caps:
            q["sw_off"].add("router_group")
        if "group_on" in levers and "router_group_cfg" in caps:
            q["sw_on"].add("router_group")
        if "nope_off" in levers and "nope_cfg" in caps:
            q["sw_off"].add("nope_layers")
        if "nope_on" in levers and "nope_cfg" in caps:
            q["sw_on"].add("nope_layers")
        if ("topk" in levers and "moe_topk" in caps and mg.FFN_RANK.get(s["ffn"], 0) >= mg.FFN_RANK["moe"]
                and ("router_group" not in feats or "group_off" in levers)):   # a group limit caps the top-k
            q["topk_free"] = True
        if "partial" in levers and "partial_rope" in caps and "MLA" not in mg.branches(s["kv"]):
            hd = q["dims"].get("head_dim", 0)
            cap = q["dims"].get("rope_cap", q["dims"].get("rope_sched", ()))
            q["dims"]["rope_cap"] = tuple(max(c, hd) if c > 0 else c for c in cap)
        if "act" in levers and "act_config" in caps and s["act"] in PARAM_FREE_ACTS:
            s["act"] = "|".join(PARAM_FREE_ACTS)
        if "parallel" in levers and "parallel_switch" in caps:
            s["block"] = "parallel|sequential"
        if "scoring" in levers and "scoring_switch" in caps:
            q["sw_on"].add("router_sigmoid")
        PH.append(q)
    return PH


R0 = ("window", "eps")
SETS = {
    "R0 (rules as stated: window + eps)": R0,
    "R0 with window for every family (remote code too)": ("window_all", "eps"),
    "(b) OFF switches + R0": R0 + ("qknorm_switch", "dsa_off", "group_off", "nope_off"),
    "(c) numeric scalars + R0": R0 + ("topk", "partial", "group_off", "group_on", "dsa_off"),
    "(a) all shape-preserving + R0": R0 + ("qknorm_switch", "dsa_off", "group_off", "group_on", "nope_off", "nope_on",
                                            "topk", "partial", "act", "parallel", "scoring"),
    "(b) OFF switches alone": ("qknorm_switch", "dsa_off", "group_off", "nope_off"),
    "(c) numeric scalars alone": ("topk", "partial", "group_off", "group_on", "dsa_off"),
    "(a) shape-preserving, window/eps excluded": ("qknorm_switch", "dsa_off", "group_off", "group_on", "nope_off",
                                                  "nope_on", "topk", "partial", "act", "parallel", "scoring"),
}
SINGLE = ("window", "window_all", "eps", "qknorm_switch", "dsa_off", "group_off", "group_on", "nope_off", "nope_on",
          "topk", "partial", "act", "parallel", "scoring")


def corrected_points(P, cfg):
    """The targets AND hosts whose sliding window cannot bind as published: window >= the model's own
    max_position_embeddings, or a family whose code never reads the window (Llama: eagle-3 drafts) -> full."""
    out, fixed = [], []
    for p in P:
        c = lm(cfg[p["members"][0]])
        sw, mp = c.get("sliding_window"), c.get("max_position_embeddings")
        ign = str(c.get("model_type")) == "llama" and p["slots"]["window"] == "local"
        if (isinstance(sw, (int, float)) and isinstance(mp, (int, float)) and sw >= mp and
                p["slots"]["window"] == "local") or ign:
            q = window_full(p, cfg, {}, verified_only=False)
            if q is not None:
                out.append(q)
                fixed.append(short(P, len(out) - 1))
                continue
        out.append(p)
    return out, fixed


def run():
    P, cfg = points()
    fam = load_caps()["families"]
    C0 = components(P)
    base = AND(C0)
    reps = frontier_set(base, P)
    cnt = np.array([len(p["members"]) for p in P])
    res = {"base": len(reps)}

    def measure(PH, PT=None):
        T = PT if PT is not None else P
        ok = AND(components(T, PH))
        new = frontier_set(ok, T)
        gone = left(reps, ok)
        return dict(frontier=len(new), freed=len(reps) - len(new),
                    left=[short(P, i) for i in sorted(gone, key=lambda i: -cnt[ok[:, i]].sum())]), ok

    rows = []
    for name, lev in SETS.items():
        r, _ = measure(apply(P, cfg, fam, lev))
        rows.append(dict(set=name, levers=list(lev), **r))
        print(f"{name:55s} {r['frontier']:4d}  (-{r['freed']})")
    singles = []
    PH0 = apply(P, cfg, fam, R0)
    for lv in SINGLE:
        r, _ = measure(apply(P, cfg, fam, (lv,)))
        r2, _ = measure(apply(P, cfg, fam, tuple(dict.fromkeys(R0 + (lv,)))))
        singles.append(dict(lever=lv, alone=r, on_R0=r2))
        print(f"  {lv:14s} alone {r['frontier']:4d} (-{r['freed']:2d})   on R0 {r2['frontier']:4d}")
    # eps sub-levers
    sub = {}
    for nm, keep in (("qknorm", "qknorm"), ("post_norm", "post_norm"), ("vnorm", "vnorm"), ("mla", "mla")):
        PH = apply(P, cfg, fam, ("eps",))
        for p0, q in zip(P, PH):
            if keep != "qknorm":
                q["slots"]["qknorm"] = p0["slots"]["qknorm"]
            if keep != "post_norm":
                q["sw_off"].discard("post_norm")
            if keep != "vnorm":
                q["sw_off"].discard("vnorm")
            if keep != "mla":
                q.pop("mla_eps", None)
        r, _ = measure(PH)
        sub[nm] = r
        print(f"  eps:{nm:10s} {r['frontier']:4d} (-{r['freed']})  {r['left'][:8]}")
    # corrections: windows that cannot bind as published (target AND host side)
    PC, fixed = corrected_points(P, cfg)
    rc, _ = measure(PC, PC)
    print(f"correction (window >= own context, or ignored by code) on {len(fixed)} points: frontier {rc['frontier']}")
    res.update(sets=rows, singles=singles, eps_sub=sub, correction=dict(points=fixed, **rc))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "relax.json"), "w") as f:
        json.dump(res, f, indent=1)
    # which hosts each freed model goes to under R0 (for the report)
    ok0 = AND(components(P, PH0))
    det = {}
    for i in left(reps, ok0):
        hosts = [j for j in range(len(P)) if ok0[i, j] and not ok0[j, i] and not base[i, j]]
        det[short(P, i)] = sorted({short(P, j) for j in hosts})[:4]
    with open(os.path.join(OUT, "relax_R0_hosts.json"), "w") as f:
        json.dump(det, f, indent=1)
    return res


if __name__ == "__main__":
    run()
