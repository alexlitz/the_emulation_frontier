"""Compute the emulator frontier and write FRONTIER.md: the maximal models and the
binding reason each is on the frontier (a capacity RECORD, a type WALL with no currency,
a PARETO trade-off, or the one-way PARADIGM wall).

The HEADLINE relation is emul_matrix(mode="payable"): capacity + every type wall that has a
currency paid; a wall with no currency blocks. The secondary "if every type wall were free"
frontier (mode="capacity") is computed alongside.

Run:  python -m graph.frontier            # writes FRONTIER.md
      python -m graph.frontier --top 1000 # print only: the N most-downloaded decoder LMs
"""
from __future__ import annotations

import numpy as np

from graph.model_graph import (CAT_SLOTS, FEATURES, FFN_RANK, feats_ok, MOE_DEPTH_FACTOR, NORM_WIDTH_FACTOR, ROT_HOSTS,
                               ROT_TARGETS, ZERO_EDGES, _HALL, _dedup_points, _junky, blocked_walls, classify,
                               dims_of, emul_matrix, iter_json_configs, payable_reaches, rope_fits, short,
                               slot_reaches, kv_heads_fit, alibi_nests, moe_why, _rec_ok, EPS_LEVER_FEATS,
                               host_rope_cap)

_alibi_sig = lambda d: (int(d.get("alibi_heads", 0)), int(d.get("alibi_max", 8) or 8), bool(d.get("alibi_bf16", 0)),
                        float(d.get("alibi_scale", 1.0) or 1.0))     # (round 3: Falcon's 1 / sqrt(head_dim) slope scale)

AX = ["kv_cap", "q_cap", "qheads", "head_dim", "ffn_total", "ffn_active", "hidden", "eff_layers"]
JSON = "graph/hub_10k.json"


def load(top=None):
    """top: only the N most-downloaded decoder LMs (the snapshot is in download order)."""
    M = []
    for n, c in iter_json_configs(JSON):
        if _junky(n):
            continue
        cl = classify(c)
        d = dims_of(c) if cl else None
        if cl and d["hidden"] and d["layers"]:
            if top and len(M) >= top:
                break
            M.append({"name": n, "slots": cl, "dims": d})
    return M


def reps_of(ok, P):
    """Frontier equivalence classes (mutual ties collapsed) -> representative point indices."""
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)
    front = np.where(~strict.any(1))[0]
    reps, used = [], set()
    for i in front:
        if i in used:
            continue
        eq = [j for j in front if ok[i, j] and ok[j, i]]
        used.update(eq)
        reps.append(min(eq, key=lambda j: ("test" in P[j]["members"][0].lower(),
                                            len(P[j]["members"][0]))))
    return reps


def blockers(A, B, mode="payable"):
    """Why host B does not emulate target A although B dominates A on raw capacity: every check of
    emul_matrix that refuses the pair (round 2, pass 5: read from emul_matrix(explain=...), so the list is exactly
    the relation's -- no failing pair without a reason, no passing pair with one), each with its detail."""
    sa, sb, da, db = A["slots"], B["slots"], A["dims"], B["dims"]
    why = {}
    emul_matrix([A, B], mode=mode, explain=why)
    failing = [nm for nm, m in why.items() if m[0, 1]]
    out = []
    plain = lambda s: s["mixer"] == "softmax" and s["crosslayer_kv"] == "no"
    fac = MOE_DEPTH_FACTOR if (FFN_RANK.get(sb["ffn"], 3) < FFN_RANK.get(sa["ffn"], 3)
                               or (sb["block"] == "parallel" and sa["block"] == "sequential")) else 1
    for nm in failing:
        if nm.startswith("slot:"):
            s = nm[5:]
            note = {"pos": " (no start-token / zero-frequency route on this host)" if sa["pos"] != sb["pos"] else "",
                    "act": " (no one-unit floor, and no k-unit width fit the host's FFN pays)",
                    "sink": " (a sink-free host needs a q bias and a spare zero-frequency slot)",
                    "norm": " (the LayerNorm fold's beta needs an lm_head bias, or a measured compensator)"}.get(s, "")
            out.append(f"{s} {sa[s]}<-{sb[s]}" + note)
        elif nm == "feature":
            T, H = set(filter(None, sa.get("feats", "").split("+"))), set(filter(None, sb.get("feats", "").split("+")))
            rule = lambda x: FEATURES.get(x) or (("wall", "free") if x.startswith("rf_") else ("free", "free"))
            off = (EPS_LEVER_FEATS & H) if db.get("eps_lever") and mode != "exact" else set()
            out.append("feature " + ", ".join([f"{x}<-none" for x in sorted(T - H) if rule(x)[0] == "wall"]
                                             + [f"none<-{x}" for x in sorted(H - T) if rule(x)[1] == "wall"
                                                and x not in off]))
        elif nm == "mixer/cla":
            for s in ("mixer", "crosslayer_kv"):
                if sa[s] != sb[s]:
                    out.append(f"{s} {sa[s]}<-{sb[s]}")
        elif nm == "pos rot gate":
            rot = lambda t: "+".join(str(x) if x >= 0 else f"{-x} NoPE" for x in t) or "0"   # -r: NoPE layers
            fixed = "" if db.get("rope_pairs_free") else (
                "; a fixed-rotary host (rope_theta only) plays only fixed-rotary, default-or-own-type targets"
                if not da.get("rope_pairs_free") else "; a fixed-rotary host cannot play a pairs-free (longrope) target")
            out.append(f"pos {sa['pos']}<-{sb['pos']} (rotated dims per head: host {rot(host_rope_cap(B))} "
                       f"< target {rot(da.get('rope_sched', ()))}{fixed})")
        elif nm == "pos fixed-rotary packing":
            out.append(f"pos: a fixed-rotary host plays a longrope-capable target only as its MHA expansion "
                       f"(kv heads {db.get('kv_heads')} < {da.get('qheads')} q heads)")
        elif nm == "kv heads":
            out.append(f"kv heads: host {db['kv_heads']} x {db['q_per_kv']} q (head_dim {db['head_dim']}) cannot hold "
                       f"target {da['kv_heads']} x {da['q_per_kv']} q (head_dim {da['head_dim']})")
        elif nm == "K=V":
            out.append("K=V: host's independent-V layers too narrow")
        elif nm == "kv MLA<->plain":
            out.append(f"kv {sa['kv']}<-{sb['kv']} (MLA latent norm; a plain target needs the eps lever and an "
                       f"MQA-RoPE or NoPE shape the shared rotary key carries)")
        elif nm == "kv MLA latents":
            out.append("kv MLA latents / nope / v head width")
        elif nm in ("kv_cap", "q_cap", "qheads", "head_dim", "hidden"):
            out.append(f"{nm}: host {db.get(nm)} < target {da.get(nm)} (+ the charges of the walls it crosses)")
        elif nm == "loop":
            out.append("loop untied<-looped: the host's later iterations cannot be silenced (only a plain looped host, "
                       "Nanbeige4.2, by the counter construction)")
        elif nm == "ffn width":
            out.append(f"ffn width: host {db['ffn_total']} / {db['ffn_active']} < target {da['ffn_total']} / "
                       f"{da['ffn_active']} (x k for an activation fit, + charged units)")
        elif nm == "ffn per-layer widths":
            out.append("ffn per-layer widths")
        elif nm in ("depth (eff)", "depth (param)"):
            if plain(sa) and (not plain(sb) or (sa["window"] == "full" and sb["window"] == "local")):
                key = "full_eff" if sa["window"] == "full" else "attn_eff"
                kinds = [k for k in (("mixer", sa["mixer"], sb["mixer"]), ("window", sa["window"], sb["window"]),
                                     ("crosslayer_kv", sa["crosslayer_kv"], sb["crosslayer_kv"])) if k in ZERO_EDGES]
                what = "/".join(k[0] for k in kinds) or "mixer"
                out.append(f"{what} depth (host's usable layers {db.get(key)} < {fac * da['eff_layers']})")
            else:
                out.append(f"{nm}: host {db['eff_layers'] if 'eff' in nm else db['param_layers']} < target "
                           f"{fac * da['eff_layers'] if 'eff' in nm else da['param_layers']} (+ charged layers)")
        elif nm == "layer kinds":
            for w, f in (("eff", fac), ("param", 1)):
                for X, N in _HALL:
                    dem = sum(da.get(f"k{t}_{w}", 0) for t in X) * f
                    sup = sum(db.get(f"k{h}_{w}", 0) for h in N)
                    if sup < dem:
                        out.append(f"layers {'+'.join(X)} ({w}): host {sup} < {dem}")
                        break
            if not any(x.startswith("layers ") for x in out):
                out.append("layers: the host's layer kinds (window-free hosts as full; a window-free target as full "
                           "on a fixed-window host) or the charged extra layers")
        elif nm == "ffn MoE<-dense":
            out.append("ffn MoE<-dense: not a top-1 unit-gate router")
        elif nm == "ffn MoE routing":
            w = moe_why((da["moe_sig"], da["hidden"], da["ffn_total"]), (db["moe_sig"], db["hidden"], db["ffn_total"])) \
                if da.get("moe_sig") and db.get("moe_sig") else None
            out.append(f"ffn {w}" if w else "ffn MoE routing (at the activation fit's width)")
        elif nm == "block parallel<-sequential":
            out.append("block parallel<-sequential: pre-norm carry")
        elif nm == "pos alibi<-rope":
            out.append("pos alibi<-rope: needs a causal target and a spare pair")
        elif nm == "pos alibi<-alibi":
            out.append("pos alibi<-alibi: the host's slopes do not include the target's")
        elif nm == "pos nope<-alibi":
            out.append("pos nope<-alibi: the host's ALiBi slopes are fixed in code")
        elif nm == "recurrent dims":
            rc = lambda d: "/".join(str(d.get(k, 0)) for k in ("rc_h", "rc_k", "rc_v", "rc_kh", "rc_ng", "rc_conv",
                                                                "rc_rank"))
            out.append(f"recurrent dims h/k/v/kh/ng/conv/rank: host {rc(db)} cannot hold target {rc(da)}"
                       + (" (Zamba shared blocks / adapters)" if da.get("zamba_mem") else ""))
        elif nm == "compensator: hosts an unmeasured target":
            out.append("compensator: the target hosts an unmeasured-range target (transitivity)")
        else:
            out.append(nm)
    if "hidden" in failing and sa["norm"] == "RMSNorm" and sb["norm"] == "LayerNorm" and \
            db["hidden"] < NORM_WIDTH_FACTOR * da["hidden"]:
        out.append("norm RMSNorm<-LayerNorm needs 2x hidden")
    return list(dict.fromkeys(out))                    # (eff and param depth say the same)


def bindings(P, ok, cap, reps):
    """Why each frontier representative is on the frontier: {point index: binding}. A RECORD (holds
    a global capacity maximum), a WALL (dominated on capacity, kept by a type wall with no currency),
    a PARETO trade-off, or the one-way PARADIGM wall. ok is the relation the frontier is taken under,
    cap the capacity-only one."""
    cnt = np.array([len(p["members"]) for p in P])
    vals = {a: np.array([p["dims"][a] for p in P]) for a in AX}
    mx = {a: vals[a].max() for a in AX}
    cap_strict = cap & ~cap.T
    np.fill_diagonal(cap_strict, False)
    out = {}
    for i in reps:
        d, s = P[i]["dims"], P[i]["slots"]
        rec = [a for a in AX if d[a] == mx[a]]
        doms = [j for j in np.where(cap_strict[i])[0]]            # capacity-dominators the walls stop
        if s["gen"] != "ar":                         # the one-way paradigm wall (diffusion; block-causal SDAR is
            why = f"PARADIGM gen={s['gen']}"             # one class with it under the mask currency: GEN_ORDER)
        elif rec:                                      # holds a global capacity maximum
            why = "RECORD " + ", ".join(rec)
        elif doms:                                     # dominated by size, kept only by a type wall
            j = max(doms, key=lambda j: int(cnt[ok[:, j]].sum()))
            why = f"WALL vs {short(P[j]['members'][0])}: " + "; ".join(blockers(P[i], P[j]))
        else:                                          # maximal by a capacity combination
            defs = sorted((sum(int(vals[a][j] < d[a]) for a in AX), j) for j in range(len(P)) if j != i)
            _, jmin = defs[0]
            lead = sorted({a for a in AX if vals[a][jmin] < d[a]})
            why = "PARETO " + (", ".join(lead) if lead else "combination (penalty / param-layer protected)")
        out[i] = why
    return out


def frontier(top=None, mode="payable"):
    M = load(top)
    P = _dedup_points(M)
    ok = emul_matrix(P, mode=mode)
    cap = emul_matrix(P, mode="capacity") if mode != "capacity" else ok
    cnt = np.array([len(p["members"]) for p in P])
    reps = reps_of(ok, P)
    cover = {i: int(cnt[ok[:, i]].sum()) for i in reps}
    reps.sort(key=lambda i: -cover[i])
    why = bindings(P, ok, cap, reps)
    rows = [(cover[i], short(P[i]["members"][0]), P[i]["slots"], P[i]["dims"], why[i]) for i in reps]
    return rows, len(P), int(cnt.sum()), P


def _core(s):
    return f"{s['norm'][:3]}·{s['pos']}·{s['kv']}·{s['ffn']}·{s['act']}"


def main():
    import sys
    if sys.argv[1:2] == ["--top"]:                     # python -m graph.frontier --top 1000: print only
        n = int(sys.argv[2])
        for mode in ("payable", "capacity"):
            rows, npts, n_models, _ = frontier(n, mode)
            nd = sum(r[2]["gen"] != "ar" for r in rows)
            print(f"[{mode}] top {n_models} decoder LMs -> {npts} shapes -> frontier {len(rows)} "
                  f"({len(rows) - nd} AR + {nd} diffusion)")
            if mode == "payable":
                for r in rows:                         # (cover, name, slots, dims, binding)
                    print(f"  {r[0]:5d}  {r[1]:42s} {r[-1]}")
        return
    rows, npts, n_models, P = frontier()
    crow, _, _, _ = frontier(mode="capacity")
    nd = sum(r[2]["gen"] != "ar" for r in rows)
    cnd = sum(r[2]["gen"] != "ar" for r in crow)
    walls = blocked_walls(P)
    lines = [
        "# The emulator frontier",
        "",
        f"The **{len(rows)} maximal models** ({len(rows) - nd} autoregressive + {nd} diffusion: "
        f"{sum(r[2]['gen'] == 'diffusion' for r in rows)} bidirectional, {sum(r[2]['gen'] == 'block' for r in rows)} "
        f"block-causal"
        # (round 3: the prefix-LM gen value, counted when present)
        + (f", {sum(r[2]['gen'] == 'prefix' for r in rows)} prefix-LM" if any(r[2]["gen"] == "prefix" for r in rows)
           else "")
        + ") that no "
        f"other model can emulate — computed over the {npts} distinct checkpoint shapes ({n_models} models) "
        "in `graph/hub_10k.json` under the **headline relation**: capacity dominance, plus every type "
        "wall that *has a currency* paid (exact edges, floors / regimes, limits — "
        "`emul_matrix(mode=\"payable\")`). A slot difference with no currency blocks emulation; that "
        "includes a softmax host for an SSM / linear / hybrid target (the near-gap is measured not "
        "payable: `attn/mixer_gap_agreement.py`) and a non-sharing host for a cross-layer-KV-sharing "
        "target (the carry has a ~6e-2 pre-norm floor: `block/prenorm_carry.py`). "
        "Regenerate with `python -m graph.frontier`.",
        "",
        f"Secondary — *if every type wall were free* (`mode=\"capacity\"`: capacity + the one-way "
        f"AR ⊆ diffusion wall only): **{len(crow)}** ({len(crow) - cnd} AR + {cnd} diffusion).",
        "",
        f"`cover` = how many of the {n_models} models it can emulate. **Binding** = why it is on the "
        "frontier: **RECORD** (holds the global maximum on that capacity axis), **WALL** (a bigger "
        "model dominates it on capacity but a type wall with no currency stops it — the named blocker "
        "is the widest-covering such model), **PARETO** (maximal by a capacity trade-off), or "
        "**PARADIGM** (a diffusion model: diffusion hosts emulate AR targets through the causal mask, "
        "but no AR host emulates a diffusion target).",
        "",
        "| cover | model | norm·pos·kv·ffn·act | h | L | kv | q-heads | ffn_total | binding |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cover, name, s, d, why in rows:
        lines.append(f"| {cover} | `{name}` | {_core(s)} | {d['hidden']} | {d['eff_layers']} | "
                     f"{d['kv_cap']} | {d['qheads']} | {d['ffn_total']} | {why} |")
    kinds = {}
    for _, _, _, _, why in rows:
        kinds[why.split()[0]] = kinds.get(why.split()[0], 0) + 1
    lines += [
        "",
        "## The read",
        "",
        "Bindings: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + ".",
        "",
        "1. **Capacity records** — "
        + ", ".join(f"`{name}` ({why[len('RECORD '):]})" for _, name, _, _, why in rows
                    if why.startswith("RECORD"))
        + ". Nothing exceeds them on their axis.",
        "2. **Type walls** — models a bigger model would host if the wall were free: absolute "
        "positions (learned-abs: no relative scheme reproduces the per-position table; the start-token rule "
        "pays it only on a host that can zero its frequencies, at +2 layers and T + 1 FFN units), activations "
        "with no exact fold or limit to the host's (GeLU, mish, xIELU, squared-ReLU, situ, poly_norm: "
        "every single-op floor is above the calibrated ACT_FLOOR_MAX, i.e. not high-agreement "
        "generation on real models -- act/act_swap_agreement.py; the k-unit width fits pay GeLU <-> SiLU "
        "at FFN x 3 - 6), q / k norms, post-norms and biases a host lacks (the eps lever only switches a "
        "host's norm OFF), fixed-rotary hosts (rope_theta only) for longrope-capable targets, recurrent mixers on a softmax host "
        "(SSM / linear / hybrid targets: the softmax constructions that reach high-agreement generation "
        "need per-channel heads, extra depth and a non-stock pre-norm, the delta rule depth ~ context -- "
        "attn/mixer_gap_agreement.py), cross-layer KV sharing (CLA) targets on a non-sharing host "
        "(the carried copy is rescaled by the shared pre-norm, ~6e-2 -- block/prenorm_carry.py), "
        "and hybrid / sliding-window hosts "
        "whose attention layers are too few to host a deep softmax stack (the zeroing edge charges "
        "depth). See the wall list below.",
        "3. **Pareto trade-offs** — maximal by a combination of capacities.",
        "4. **Paradigm** — the maximal diffusion models. The wall is one-way: a diffusion model "
        "emulates its AR twin by applying the causal mask, never the reverse.",
        "",
        "## Walls with no currency (target ← host)",
        "",
        "Every slot-value pair present in the snapshot that the headline relation cannot cross "
        "(`graph/model_graph.py` `blocked_walls()`):",
        "",
        "| slot | target | host | why |",
        "|---|---|---|---|",
    ]
    for slot, a, b, why in walls:
        lines.append(f"| {slot} | {a} | {b} | {why} |")
    lines += [
        "",
        "## Secondary: if every type wall were free",
        "",
        f"{len(crow)} maximal models by capacity alone: "
        + ", ".join(f"`{name}` ({cover})" for cover, name, _, _, _ in crow) + ".",
    ]
    with open("FRONTIER.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote FRONTIER.md: {len(rows)} frontier models (payable); {len(crow)} if every wall were free")


if __name__ == "__main__":
    main()
