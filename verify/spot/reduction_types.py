"""Every type of reduction the headline relation can use, generated from the relation's code (graph/model_graph.py).

  slot      one-hop transitions of each categorical slot's order (SLOTS: value -> [(parent, currency, exact)]),
            kept when the headline pays them -- payable_reaches, or a pass-5 port (P5_PORTS) that crosses the wall
            -- plus every transition the headline relation actually crosses between two zoo shapes
            (emul_matrix(mode="payable")); the abstract tops (pos_top, act_top, ...) are not real values
  zero      the ZERO_EDGES (a host with some layers of the target's kind; depth is the currency)
  feature   every FEATURES rule with a free / limit side: "target" = a target feature the host lacks (the host
            reproduces it by a weight setting), "host" = a host feature the target lacks (switched off); round 3:
            "nest" = a FEATURES nesting (FEAT_NEST: a target feature carried by a host parent feature)
  port      every pass-5 port (P5_PORTS: config levers, the eps lever per norm kind, the start-token rules, the
            activation width, the rescues)
  capacity  the capacity axes edge_plan pads (AXIS_PAD), KV-head packing (position-free and RoPE / longrope), the
            per-layer-kind depth matching (KIND_REACH, host kind -> target kind), the width / depth factors
            (NORM_WIDTH_FACTOR, MOE_DEPTH_FACTOR), the rot gate (rope_fits), MLA latents (an integer multiple)
  moe       expert duplication (m = 1 / m > 1) per renorm kind, group routing per group score, the router-score
            matches, the dense / MoE layer counts, a MoE host for a dense target, a dense host for a top-1 target
  recurrent the per-family recurrent-capacity rules (RC_* of dims_of: GE / M2 / GDN / KDA / EQ) and Zamba2's shared
            blocks
  paradigm  the generation masks (GEN_ORDER) -- also slot types -- and self-conditioning switched off

  python -m verify.spot.reduction_types            # writes verify/spot/reduction_types.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "reduction_types.json"
ABSTRACT = {"pos_top", "act_top", "AffineNorm"}


def _slot_types(G, observed):
    out = {}
    for slot, order in G.SLOTS:
        for a, parents in order.items():
            for b, cur, exact in parents:
                if a in ABSTRACT or b in ABSTRACT:
                    continue
                tid = f"slot:{slot}:{a}->{b}"
                # used by the headline: crossed between two zoo shapes, or payable and not walled by a global flag
                # (PAR_CARRY_PAYABLE walls a sequential host for a parallel target in emul_matrix)
                walled = slot == "block" and (a, b) == ("parallel", "sequential") and not G.PAR_CARRY_PAYABLE
                pay = bool(G.payable_reaches(slot, a, b)) and not walled
                out[tid] = {"id": tid, "kind": "slot", "slot": slot, "target": a, "host": b, "currency": cur,
                            "exact": bool(exact), "headline": pay or tid in observed, "one_hop": True,
                            "observed": observed.get(tid, 0)}
    for tid, n in observed.items():
        if tid not in out:
            _, slot, ab = tid.split(":", 2)
            a, b = ab.split("->")
            from graph.reduce import SIZE_SLOTS
            cost = G.edge_cost(dict(G.SLOTS)[slot], a, b)
            zero = (slot, a, b) in G.ZERO_EDGES
            out[tid] = {"id": tid, "kind": "slot", "slot": slot, "target": a, "host": b,
                        "currency": (",".join(dict.fromkeys(cost[0])) if cost else
                                     "depth (ZERO_EDGES)" if zero else
                                     "size (capacity rows)" if slot in SIZE_SLOTS else "port"),
                        "exact": bool(cost[1]) if cost else True, "headline": True, "one_hop": False, "observed": n}
    return out


def _observed(G):
    """slot transitions the headline crosses between zoo shapes (count of edges)."""
    import numpy as np
    from graph.currency_curve import points
    P = points()
    ok = G.emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, False)
    names = [s for s, _ in G.SLOTS]
    cnt = {}
    for i, j in np.argwhere(ok):
        sa, sb = P[i]["slots"], P[j]["slots"]
        for s in names:
            if sa[s] != sb[s]:
                k = f"slot:{s}:{sa[s]}->{sb[s]}"
                cnt[k] = cnt.get(k, 0) + 1
    return cnt


def types(observed=True):
    import graph.model_graph as G
    from graph.edge_plan import AXIS_PAD
    obs = _observed(G) if observed else {}
    T = _slot_types(G, obs)

    def add(tid, kind, note, **kw):
        T[tid] = {"id": tid, "kind": kind, "note": note, "headline": True, **kw}
    for (slot, a, b), cur in G.ZERO_EDGES.items():
        add(f"zero:{slot}:{a}<-{b}", "zero", cur, slot=slot, target=a, host=b)
    for f, (t_rule, h_rule, note) in G.FEATURES.items():
        for side, rule in (("target", t_rule), ("host", h_rule)):
            if rule in ("free", "limit"):
                add(f"feature:{f}:{side}", "feature", note[:200], feature=f, side=side, rule=rule)
    for f, parents in getattr(G, "FEAT_NEST", {}).items():     # round 3: a target feature carried by a host parent
        for p, kind in parents:
            add(f"feature:{f}:nest:{p}", "feature", f"FEAT_NEST: {f} on a host with {p} ({kind})", feature=f,
                side="nest", parent=p, rule=kind)
    for f in getattr(G, "R3_FEAT_NEST", {}):                   # round 3 / fr-mixer: the recurrent lever nests
        for p, kind in G._r3_nest(f):
            add(f"feature:{f}:nest:{p}", "feature", f"R3_FEAT_NEST: {f} on a host with {p} ({kind}, eps lever)",
                feature=f, side="nest", parent=p, rule=kind)
    for p in getattr(G, "P5_PORTS", ()):
        add(f"port:{p}", "port", "pass-5 port (graph.model_graph P5_PORTS)", port=p)
        if not getattr(G, "p5_port_live", lambda _: True)(p):
            # round 3, batch 3b: a port whose credit a round-3 key removed (P5_REMOVED_BY) credits no edge
            T[f"port:{p}"].update(headline=False, note=f"removed by {G.P5_REMOVED_BY[p]} (graph.model_graph)")
    if G._r3("nope_rope_limit"):
        # round 3: the linear-scaling host config lever (a limit; graph.model_graph nope_rope_limit_slot / _rot)
        add("lever:nope_rope_limit", "lever", "host config lever: linear rope scaling factor -> inf on a host whose "
            "rotary reads it (rope_lin), a NoPE target or a mixed target's NoPE layers (Lean rope_scale_limit_identity)",
            lever="nope_rope_limit", rule="limit")
    for ax, note in AXIS_PAD.items():
        add(f"capacity:{ax}", "capacity", note, axis=ax)
    add("capacity:kv_pack:nope", "capacity", "KV-head packing, position-free (kv_heads_fit)")
    add("capacity:kv_pack:rope", "capacity", "KV-head packing on a rope_pairs_free (longrope) host: the target's "
        "frequencies once per packed block (kv_heads_fit, rope_fits)")
    add("capacity:rot_gate", "capacity", "partial-RoPE host for a RoPE target: the target's rotating pairs in the "
        "host's inv_freq tables (rope_fits, hf/partial_rope_host.py)")
    for h, ts in G.KIND_REACH.items():
        for t in ts:
            add(f"capacity:layer_kind:{t}<-{h}", "capacity", "per-layer-kind depth matching (KIND_REACH)",
                host_kind=h, target_kind=t)
    add("capacity:norm_width", "capacity", f"x{G.NORM_WIDTH_FACTOR} hidden: the [h; -h] mirror (RMSNorm target, "
        f"LayerNorm host)")
    add("capacity:moe_depth", "capacity", f"x{G.MOE_DEPTH_FACTOR} depth: a less-rich FFN host / a parallel host for "
        f"a sequential target")
    add("capacity:mla_latent", "capacity", "MLA <- MLA: latents an integer multiple of the target's (or any wider "
        "under the eps lever)")
    renorms = sorted({v for v in G.RENORM_FIXED.values()} | {"R", "P"})
    for r in renorms:
        for m in ("1", "m"):
            add(f"moe:dup:{r}:{m}", "moe", f"MoE <- MoE expert duplication, renorm {r}, m = {m} copies (moe_why)",
                renorm=r, copies=m)
    for g in sorted(set(G.GROUP_SCORE_TYPES.values())):
        add(f"moe:group:{g}", "moe", f"group-limited routing, group score {g} (moe_why)", group=g)
    add("moe:score", "moe", "router score kinds a host's code computes (score_cap, moe_why)")
    add("moe:dense_layers", "moe", "dense-first target layers on host dense / spare MoE layers (moe_why)")
    add("moe:shared", "moe", "shared expert, host >= target (moe_why)")
    add("moe:latent", "moe", "expert-input latent rank, host >= target (moe_why)")
    add("moe:extra_experts", "moe", "extra host experts scored below the copies (a selection bias / a free hidden dim)")
    # fr-moe: the constant-channel constructions (moe/router_frontier.py), used while their spec is on
    for spec, tid, note in (("moe_null_experts", "moe:null_experts", "a top-k remainder k_h = m k_t + r filled by r "
                             "always-selected zero-weight null experts (selection-only bias + a free hidden dim)"),
                            ("moe_plain_extras_sink", "moe:plain_extras_sink", "extra experts of an unrenormalised "
                             "softmax host sunk by -M on a free hidden dim's constant channel")):
        add(tid, "moe", note)
        T[tid]["headline"] = G._r3(spec) and hasattr(G, "_moe_copy_plans")
    add("moe:host_for_dense", "moe", "dense target on a MoE host: zero router, units over the selected experts")
    add("moe:dense_host_top1", "moe", "dense host for a top-1 unit-gate MoE target (moe/dense_host_prenorm.py)")
    for name in ("RC_GE", "RC_M2", "RC_GDN", "RC_KDA", "RC_EQ"):
        if hasattr(G, name):
            add(f"recurrent:{name[3:].lower()}", "recurrent", f"recurrent capacity rule {name} (_rc_pair_ok)",
                rule=getattr(G, name))
    add("recurrent:zamba_shared", "recurrent", "Zamba2 shared blocks a multiple of the target's (_rec_ok)")
    return dict(sorted(T.items()))


def main():
    T = types()
    OUT.write_text(json.dumps(list(T.values()), indent=1) + "\n")
    hl = [t for t in T.values() if t["headline"]]
    print(f"wrote {OUT}: {len(T)} types, {len(hl)} used by the headline")


if __name__ == "__main__":
    main()
