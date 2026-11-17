"""solve(A, B): search the reductions to fit a target model A into a host B.

Given a target A and a host B, find the plan that fits A's forward pass into B --
the per-slot reduction and how each currency is paid -- plus the capacity cost. If B
can't host A as-is, report the obstructions (incomparable / must-match slots) and the
MINIMAL upgrade to B that would make it fit. Where a reduction has a width<->depth
dual (holding MoE experts), it tries both and reports which B can afford.

This is the search layer over the rules in model_graph: the per-slot orders are the
moves, edge_cost is the move cost, and emulates()'s capacity checks are feasibility.

  python -m graph.solve Mistral-7B-v0.1 Qwen1.5-72B
  python -m graph.solve xLAM-8x22b-r Qwen1.5-72B --verify
"""
from __future__ import annotations

import argparse
import math

from graph.model_graph import FFN_RANK, SLOT_DEFAULT, SLOTS, classify, dims_of, edge_cost
from graph.reduce import (DEMOS, DTYPES, SIZE_SLOTS, axis_gaps, describe, load_index,
                          resolve)

# map each capacity axis to the target-A dim it derives from (for the "overhead" cost)
AXIS_BASE = {"kv_cap": "kv_cap", "q_cap": "q_cap", "qheads": "qheads",
             "ffn_total": "ffn_total", "ffn_active": "ffn_active",
             "eff_layers": "eff_layers", "param_layers": "param_layers", "hidden": "hidden"}


FFN_AXES = {"ffn_total", "ffn_active", "eff_layers"}


def ffn_dual(A, B):
    """Holding A's E experts, the width<->depth dual:
      - width: one wide FFN of ffn_total, + 2x depth for the routing circuit;
      - depth: E masked FFN sublayers, each one expert wide (ew = ffn_total/E).
    Returns each option's requirement and whether B affords it."""
    sa, sb, da, db = A["slots"], B["slots"], A["dims"], B["dims"]
    if FFN_RANK[sb["ffn"]] >= FFN_RANK[sa["ffn"]]:
        return None                                   # B already >= A's FFN richness
    E = max(da["experts"], 1)
    ew = da["ffn_total"] / E                           # one expert's width
    width_ok = (db["ffn_total"] >= da["ffn_total"] and db["ffn_active"] >= da["ffn_active"]
                and db["eff_layers"] >= 2 * da["eff_layers"])
    depth_layers = da["eff_layers"] + E               # E masked sublayers, one expert each
    depth_ok = db["ffn_active"] >= ew and db["eff_layers"] >= depth_layers
    return {"width": {"ffn_total": da["ffn_total"], "depth2x": 2 * da["eff_layers"], "ok": width_ok},
            "depth": {"per_layer": int(ew), "layers": depth_layers, "ok": depth_ok}}


def solve(A, B):
    sa, sb, da = A["slots"], B["slots"], A["dims"]
    plan, obstructions = [], []
    for slot, order in SLOTS:
        a, b = sa.get(slot, SLOT_DEFAULT.get(slot)), sb.get(slot, SLOT_DEFAULT.get(slot))   # (round 3: head / input)
        if a == b:
            continue
        if slot in SIZE_SLOTS:
            plan.append((slot, f"{a}->{b}", "capacity"))
            continue
        c = edge_cost(order, a, b)                     # A_slot -> B_slot (A subset B)
        if c is None:
            obstructions.append((slot, a, b))          # incomparable / must-match
        else:
            cur, exact = c
            plan.append((slot, f"{a}->{b}",
                         ",".join(dict.fromkeys(cur)) + ("" if exact else " (approx)")))
    gaps = axis_gaps(A, B)                              # (axis, need, have, ok) with penalties
    cost = {ax: need - da[AXIS_BASE[ax]] for ax, need, _, _ in gaps
            if need > da[AXIS_BASE[ax]]}                # reduction overhead beyond A's own
    upgrade = {ax: need - have for ax, need, have, ok in gaps if not ok}
    dual = ffn_dual(A, B)
    feasible = not obstructions and not upgrade         # fits by the standard (width) model
    # the search: if the only misses are FFN capacity and the DEPTH dual clears them, it fits
    via_depth = (not feasible and not obstructions and dual and dual["depth"]["ok"]
                 and set(upgrade) <= FFN_AXES)
    return {"feasible": feasible, "via_depth": via_depth, "plan": plan,
            "obstructions": obstructions, "cost": cost, "upgrade": upgrade, "ffn_dual": dual}


def _pretty(r, an, bn, A, verify, dt):
    print(f"\n=== fit  {an}  into  {bn}  ===")
    verdict = ("FITS" if r["feasible"] else
               "FITS (only via the depth dual — hold experts across layers)" if r["via_depth"] else
               "DOES NOT FIT (as-is)")
    print(f"  {verdict}\n")
    print("  plan (per-slot reduction B applies to host A):")
    for slot, mv, how in r["plan"]:
        print(f"    {slot:13} {mv:22} {how}")
    if not r["plan"]:
        print("    (identical architectures)")
    if r["cost"]:
        print("\n  capacity the plan needs (overhead beyond A's own):")
        for ax, v in r["cost"].items():
            print(f"    {ax:13} +{v}")
    if r["ffn_dual"]:
        w, d = r["ffn_dual"]["width"], r["ffn_dual"]["depth"]
        pick = "width" if w["ok"] else ("depth" if d["ok"] else "neither -> upgrade")
        print(f"\n  FFN experts (width<->depth dual):"
              f"\n    width: one FFN of ffn_total {w['ffn_total']} + {w['depth2x']} layers  [{'ok' if w['ok'] else 'NO'}]"
              f"\n    depth: {d['layers']} layers, {d['per_layer']} wide each          [{'ok' if d['ok'] else 'NO'}]"
              f"\n    -> pay in {pick}")
    if r["obstructions"]:
        print("\n  obstructions (no reduction either way):")
        for slot, a, b in r["obstructions"]:
            print(f"    {slot:13} {a} vs {b}")
    if r["upgrade"] and not r["via_depth"]:
        print("\n  minimal upgrade to B to make it fit (width model):")
        for ax, v in r["upgrade"].items():
            print(f"    {ax:13} +{v}")
    if verify:
        slots = {slot for slot, _, _ in r["plan"]}
        for s in [x for x in DEMOS if x in slots]:
            print(f"\n  verify[{s}] identity in {dt}: max|Δ| = {DEMOS[s](dt):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="model to fit")
    ap.add_argument("host", help="model to fit it into")
    ap.add_argument("--from-json", default="graph/hub_10k.json")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    index = load_index(args.from_json)
    an, ca = resolve(args.target, index)
    bn, cb = resolve(args.host, index)
    A = {"slots": classify(ca), "dims": dims_of(ca)}
    B = {"slots": classify(cb), "dims": dims_of(cb)}
    if A["slots"] is None or B["slots"] is None:
        raise SystemExit("one of the models did not classify as a decoder LM")
    print(describe("target A = " + an, A))
    print(describe("host   B = " + bn, B))
    _pretty(solve(A, B), an, bn, A, args.verify, DTYPES[args.dtype])


if __name__ == "__main__":
    main()
