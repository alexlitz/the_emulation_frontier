"""construct(A, B): the exhaustive constructive search. If target A's forward pass can be
embedded into host B by ANY combination of reductions in this framework, find it -- mixing
currencies across axes and trading width<->depth per layer -- and report the construction.
Only call it IMPOSSIBLE when there is a genuine wall.

solve.py finds THE cheapest one-way plan (special -> general). This adds the *reverse /
near-gap / network* routes that make a categorical slot payable even when B is the more
special one:

  * activation:  any pointwise curve folds into a wider FFN (width ~1/N^2, a floor);
  * window:      a local host reaches full span by stacking depth ceil((T-1)/(w-1));
  * mixer:       an SSM/linear host near-gap-approximates softmax (and vice-versa);
  * position:    a learned-abs host reproduces any relative bias as a per-position table
                 over a bounded context T.

So a verdict of IMPOSSIBLE means a real wall -- the AR<->diffusion decode-paradigm split, or
a capacity B cannot afford even after the width<->depth trade -- not merely "no one-way edge".

  python -m graph.construct Apertus-8B Qwen1.5-72B
  python -m graph.construct LLaDA-8B Qwen1.5-72B      # -> IMPOSSIBLE (gen wall)
"""
from __future__ import annotations

import argparse

from graph.model_graph import SLOT_DEFAULT, SLOTS, classify, dims_of, edge_cost
from graph.reduce import (SIZE_SLOTS, axis_gaps, describe, load_index, resolve)
from graph.solve import ffn_dual

REL_POS = {"rope", "alibi", "partial_rope", "t5_relbias", "additive_bias"}


def reverse_route(slot, a, b):
    """How host slot-value b pays to emulate the more-general / incomparable target value a.
    Returns (currency, exact, note) or None when there is no reverse route (a genuine wall)."""
    if slot == "act":
        return ("width ~1/N^2 net", False, "any pointwise curve via a wider FFN")
    if slot == "window" and a == "full" and b == "local":
        return ("depth ceil((T-1)/(w-1))", True, "stack local windows to span the full context")
    if slot == "mixer" and a == "softmax" and b in ("ssm", "linear", "hybrid"):
        return ("features/state (near-gap)", False, "random-feature / SSM approx of softmax")
    if slot == "pos":
        if a in REL_POS and b == "learned_abs":
            return ("context (bounded T)", False, "learned per-position table reproduces the bias over T")
        if a == "rope" and b == "alibi":
            return ("frequency budget", False, "reproduce RoPE's field on a frequency subset")
        # rope <- partial_rope is a forward edge now (currency "rot": exact when the host rotates enough
        # dims per head, hf/partial_rope_host.py); below that no construction
    return None


def construct(A, B):
    """Every per-slot route (forward + reverse), the per-layer capacity trade, and a verdict."""
    sa, sb = A["slots"], B["slots"]
    routes, walls = [], []
    for slot, order in SLOTS:
        a, b = sa.get(slot, SLOT_DEFAULT.get(slot)), sb.get(slot, SLOT_DEFAULT.get(slot))   # (round 3: head / input)
        if a == b or slot in SIZE_SLOTS:
            continue
        opts = []
        fwd = edge_cost(order, a, b)                 # a <= b: B general enough (direct, free/cheap)
        if fwd is not None:
            cur, exact = fwd
            opts.append((",".join(dict.fromkeys(cur)) or "free", exact, "direct (B ≥ A)"))
        rev = reverse_route(slot, a, b)              # B more special / incomparable: pay to emulate A
        if rev is not None:
            opts.append(rev)
        (routes if opts else walls).append((slot, a, b, opts) if opts else (slot, a, b))

    gaps = axis_gaps(A, B)                            # size axes, with TransMLA / MoE penalties
    upgrade = {ax: need - have for ax, need, have, ok in gaps if not ok}
    dual = ffn_dual(A, B)
    cap_ok = (not upgrade) or (dual and dual["depth"]["ok"] and set(upgrade) <= {"ffn_total", "ffn_active", "eff_layers"})
    constructible = not walls and cap_ok
    approx = any(not any(o[1] for o in opts) for _, _, _, opts in routes)   # a slot with only floor routes
    return {"constructible": constructible, "approx": approx, "routes": routes,
            "walls": walls, "upgrade": upgrade, "ffn_dual": dual, "cap_ok": cap_ok}


def pretty(r, an, bn):
    print(f"\n=== construct  {an}  into host  {bn}  ===")
    if r["constructible"]:
        print("  CONSTRUCTIBLE" + (" (approx — some slot has only a floor route)\n" if r["approx"]
                                   else " (exact)\n"))
    else:
        why = "genuine wall" if r["walls"] else "host too small (see capacity)"
        print(f"  IMPOSSIBLE as-is — {why}\n")
    if r["routes"]:
        print("  per-slot routes (host B emulates target A; pick any — currencies mix freely):")
        for slot, a, b, opts in r["routes"]:
            print(f"    {slot:9} {a} -> {b}")
            for cur, exact, note in opts:
                print(f"        via {cur:26} [{'exact' if exact else 'approx'}]  {note}")
    if r["ffn_dual"]:
        w, d = r["ffn_dual"]["width"], r["ffn_dual"]["depth"]
        print("\n  FFN experts — per-layer width<->depth trade (a partial, mixed construction):")
        print(f"        width: one FFN of {w['ffn_total']} + {w['depth2x']} layers    [{'ok' if w['ok'] else 'NO'}]")
        print(f"        depth: {d['layers']} layers, {d['per_layer']} wide each        [{'ok' if d['ok'] else 'NO'}]")
    if r["walls"]:
        print("\n  GENUINE WALL (no route either way in-framework):")
        for slot, a, b in r["walls"]:
            extra = "   (AR<->diffusion: a decode-paradigm change, not a weight setting)" if slot == "gen" else ""
            print(f"        {slot:9} {a} vs {b}{extra}")
    if r["upgrade"]:
        print("\n  capacity B must add (or trade into depth):")
        for ax, v in r["upgrade"].items():
            print(f"        {ax:12} +{v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("host")
    ap.add_argument("--from-json", default="graph/hub_10k.json")
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
    pretty(construct(A, B), an, bn)


if __name__ == "__main__":
    main()
