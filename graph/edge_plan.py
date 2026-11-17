"""edge_plan(A, B): the construction behind one edge "host B emulates target A", as data.

graph.reduce prints the same thing for a pair of repos; this returns it structured, so that the
emulation script (elephantine/emulate.py) and the zoo page's edge view (graph.model_graph.zoo_data)
show one and the same plan. Everything is read from graph.reduce / graph.model_graph -- the slot
orders, edge_cost, ZERO_EDGES, payable_reaches, axis_gaps and emulates() -- nothing is re-derived:

  slots  every categorical slot where A and B differ: its kind (a size slot decided by the capacity
         rows, a ZERO_EDGES depth rule, a route with a currency, or a wall), the currency, and whether
         the step is exact or a limit / regime / floor;
  axes   every capacity axis: what A needs (with the TransMLA / MoE-depth / block / norm-mirror
         penalties) and what B has (only the host's layers of the target's kind under ZERO_EDGES);
  ok     the headline relation (emulates(B, A)); exact: the exact slice (mode="exact"); cap: the
         "if every type wall were free" relation.

  python -m graph.edge_plan TARGET HOST          # the plan, as text
"""
from __future__ import annotations

import argparse

from graph.model_graph import (FFN_RANK, MLA_PENALTY, MOE_DEPTH_FACTOR, NORM_WIDTH_FACTOR, WALL_NOTES, ZERO_EDGES, emulates, payable_reaches)
from graph.reduce import SIZE_SLOTS, axis_gaps, cat_diffs

# the repo code that builds / measures each step (paths relative to the repo root)
SLOT_CODE = {
    "norm": ["convert/layernorm_to_rmsnorm.py", "norm/layernorm_rmsnorm.py"],
    "pos": ["attn/positional_cross.py", "hf/nope_rope.py"],
    "kv": ["hf/gqa_mha.py", "attn/mha_mqa_gqa.py", "attn/mla.py"],
    "ffn": ["hf/moe_dense.py", "moe/dense_emulates_moe.py"],
    "act": ["act/activation_cross.py", "act/act_swap_agreement.py"],
    "block": ["block/parallel_residual.py"],
    "window": ["hf/local_full.py", "attn/rope_window.py"],
    "qknorm": ["attn/qk_norm.py"],
    "softcap": ["attn/softcap.py"],
    "mixer": ["attn/mixer_gap_agreement.py", "attn/ssm_cross.py"],
    "sink": ["attn/attention_sink.py"],
    "crosslayer_kv": ["attn/cross_layer_kv.py", "block/prenorm_carry.py"],
    "mtp": ["hf/real_reductions.py"],
    "gen": ["attn/ar_diffusion.py"],
    "loop": ["hf/looped_untied.py"],
}
AXIS_CODE = {
    "hidden": ["elephantine/hidden_width.py"],
    "kv_cap": ["hf/gqa_mha.py"], "q_cap": ["hf/gqa_mha.py"], "qheads": ["hf/gqa_mha.py"],
    "head_dim": ["hf/gqa_mha.py"],
    "ffn_total": ["hf/full_emulation.py", "hf/moe_dense.py"], "ffn_active": ["hf/full_emulation.py", "hf/moe_dense.py"],
    "eff_layers": ["hf/full_emulation.py"], "param_layers": ["hf/full_emulation.py"],
}
AXIS_LABEL = {"hidden": "hidden width", "kv_cap": "KV width", "q_cap": "query width", "qheads": "query heads",
              "head_dim": "head dim", "ffn_total": "FFN width (total)", "ffn_active": "FFN width (active)",
              "eff_layers": "depth (unrolled)", "param_layers": "depth (distinct layers)"}
# what each axis is padded with when the host is bigger (the capacity step of the construction)
AXIS_PAD = {"hidden": "zero-padded; eps and norm gains rescaled",
            "kv_cap": "KV heads tiled over the host's groups, extras zeroed",
            "q_cap": "extra query heads zeroed",
            "qheads": "extra query heads zeroed",
            "head_dim": "zero-padded; RoPE pairs on matching frequencies",
            "ffn_total": "extra units zeroed",
            "ffn_active": "extra units zeroed, or experts tied to the FFN",
            "eff_layers": "extra layers are the identity",
            "param_layers": "extra layers are the identity"}


def _plain(s):
    return s["mixer"] == "softmax" and s["crosslayer_kv"] == "no"


def _wall_note(slot, a, b):
    notes = WALL_NOTES.get(slot, {})
    return notes.get((a, b)) or notes.get((a, "*")) or notes.get(("*", b)) or notes.get(("*", "*")) or ""


def slot_row(slot, a, b, cost, plain):
    """One differing slot, classified exactly as graph.reduce.direction prints it, plus whether the
    headline relation pays for it (payable_reaches: the near-gap / CLA carry are routes but unpaid)."""
    if slot in SIZE_SLOTS:
        return {"slot": slot, "target": a, "host": b, "kind": "size", "currency": "capacity",
                "exact": True, "payable": True, "note": "decided by the capacity rows"}
    if (slot, a, b) in ZERO_EDGES and plain:
        return {"slot": slot, "target": a, "host": b, "kind": "zero", "currency": ZERO_EDGES[(slot, a, b)],
                "exact": True, "payable": True, "note": "exact; only the host's layers of the target's kind count"}
    if cost is None:
        return {"slot": slot, "target": a, "host": b, "kind": "wall", "currency": "",
                "exact": False, "payable": False, "note": _wall_note(slot, a, b)}
    cur, exact = cost
    payable = payable_reaches(slot, a, b)
    return {"slot": slot, "target": a, "host": b, "kind": "route", "currency": ",".join(dict.fromkeys(cur)),
            "exact": bool(exact), "payable": bool(payable),
            "note": "" if payable else (_wall_note(slot, a, b) or "a route exists, but the headline does not pay it")}


def axis_notes(A, B):
    """Why a need differs from the target's own value (the penalties axis_gaps applies)."""
    sa, sb = A["slots"], B["slots"]
    notes = {}
    if sa["kv"] == "MLA" and "MLA" not in str(sb["kv"]).split("|"):       # as reduce.axis_gaps charges it
        notes["kv_cap"] = f"x{MLA_PENALTY}: TransMLA penalty (MLA target, non-MLA host)"
    deep = []
    if FFN_RANK[sb["ffn"]] < FFN_RANK[sa["ffn"]]:
        deep.append(f"x{MOE_DEPTH_FACTOR}: a less-rich FFN host routes in depth")
    if sa["block"] != sb["block"] and sb["block"] != "sequential":
        deep.append(f"x{MOE_DEPTH_FACTOR}: a parallel-block host runs a sequential block in two layers")
    if deep:
        notes["eff_layers"] = "; ".join(deep)
    wide = []
    if sa["block"] != sb["block"] and sb["block"] == "sequential":
        wide.append("x2: a sequential host carries a parallel block's input")
    if sa["norm"] == "RMSNorm" and sb["norm"] == "LayerNorm":
        wide.append(f"x{NORM_WIDTH_FACTOR}: the [h; -h] mirror (RMSNorm target, LayerNorm host)")
    if wide:
        notes["hidden"] = "; ".join(wide)
    return notes


def axis_rows(A, B):
    """axis_gaps (graph.reduce) plus the head_dim check emul_matrix makes, with notes."""
    da, db = A["dims"], B["dims"]
    notes = axis_notes(A, B)
    rows = []
    for n, need, have, ok in axis_gaps(A, B):
        note = notes.get(n, "")
        own = db.get(n)
        if have != own and n in ("eff_layers", "param_layers"):
            note = (note + "; " if note else "") + "only the host's layers of the target's kind count (ZERO_EDGES)"
        rows.append({"axis": n, "target": da.get(n), "need": need, "have": have, "host": own,
                     "ok": bool(ok), "note": note})
    rows.append({"axis": "head_dim", "target": da.get("head_dim", 0), "need": da.get("head_dim", 0),
                 "have": db.get("head_dim", 0), "host": db.get("head_dim", 0),
                 "ok": db.get("head_dim", 0) >= da.get("head_dim", 0), "note": ""})
    return rows


def edge_plan(A, B, ok=None, exact=None, cap=None):
    """The structured plan for "B emulates A" (A, B: {"slots": classify(cfg), "dims": dims_of(cfg)}).
    ok / exact / cap: the relation's verdicts when already known (a row of emul_matrix)."""
    plain = _plain(A["slots"])
    slots = [slot_row(s, a, b, c, plain) for s, a, b, c in cat_diffs(A, B)]
    ok = emulates(B, A) if ok is None else ok
    exact = ok and (emulates(B, A, mode="exact") if exact is None else exact)
    cap = emulates(B, A, mode="capacity") if cap is None else cap
    return {"ok": bool(ok), "exact": bool(exact), "cap": bool(cap), "slots": slots, "axes": axis_rows(A, B)}


def verdict(plan):
    if not plan["ok"]:
        return "NOT AN EDGE"
    return "EXACT EDGE" if plan["exact"] else "EDGE (uses a limit / regime / floor)"


def format_plan(plan, tname, hname):
    """The plan as text (the script's printout and README)."""
    out = [f"{hname}  emulates  {tname} ?   {verdict(plan)}"
           + ("" if plan["ok"] else f"   [if every type wall were free: {'YES' if plan['cap'] else 'NO'}]")]
    out.append("  slots (target -> host):")
    if not plan["slots"]:
        out.append("    (no categorical slot differs)")
    for r in plan["slots"]:
        tag = {"size": "size-based (capacity rows)", "zero": f"{r['currency']} [exact]",
               "wall": "WALL (no currency)"}.get(r["kind"])
        if tag is None:
            tag = f"{r['currency']} [{'exact' if r['exact'] else 'limit / regime / floor'}]"
            if not r["payable"]:
                tag += " NOT PAID by the headline"
        out.append(f"    {r['slot']:13} {r['target']} -> {r['host']}   {tag}" + (f"   ({r['note']})" if r["note"] and r["kind"] in ("wall", "route") else ""))
    out.append("  capacity (need <= have):")
    for r in plan["axes"]:
        flag = "ok" if r["ok"] else "FAIL"
        extra = f"   ({r['note']})" if r["note"] else ""
        out.append(f"    {r['axis']:13} target {r['target']:>9}  need {r['need']:>9}  host {r['have']:>9}  {flag}{extra}")
    return "\n".join(out)


# --- the zoo page's edge view: compact tables for every edge the page lets you open --------------------

ZOO_AXES = ("hidden", "kv_cap", "q_cap", "qheads", "head_dim", "ffn_total", "ffn_active", "eff_layers", "param_layers")
HAVE_KEYS = ZOO_AXES + ("full_eff", "attn_eff", "full_param", "attn_param")


def approx_reason(plan):
    bad = [f"{r['slot']} {r['target']} -> {r['host']} ({r['currency']})" for r in plan["slots"]
           if r["kind"] == "route" and not r["exact"]]
    return ("the edge needs a limit / regime / floor (" + "; ".join(bad) +
            "): the script writes exact constructions only")


def cmd_repo(members, configs):
    """The repo the page's command names: the most-downloaded member the script can read (the stock class
    of a supported family, not a quantized checkpoint; one whose own settings it builds first), else the
    most-downloaded member."""
    from elephantine.emulate import Refused, family_of, side_support
    readable = []
    for m in members:
        c = configs.get(m)
        if c is None or c.get("quantization_config"):
            continue
        try:
            family_of(c, "host")
        except Refused:
            continue
        if side_support(c, "host") is None:
            return m
        readable.append(m)
    return readable[0] if readable else members[0]


def zoo_edges(P, exact, cap, pairs, repos, configs):
    """Compact plan tables for the (target i, host j) pairs, all edges of the headline relation:
      rows     unique slot rows [slot, target, host, kind, currency, exact, payable, note]
      sets     unique lists of row ids (the differing slots of an edge)
      axsets   unique lists of capacity deviations: [axis, factor, have-key, note] means
               need = factor x target[axis], have = host[have-key]; [axis, need, have, note, 1] is literal;
               an axis not listed has need = target[axis], have = host[axis] and no note
      notes    axis note strings
      support  the script's status strings ("" = builds it: exact, or a limit edge it builds as a labelled limit
               construction, elephantine.emulate.LIMIT_BUILT); tr / hr: per shape, the reason it refuses its command
               repo as a target / as a host whatever the other side (0 = none)
      eb       per pair 5 bytes, base64: slot set (uint16 LE), axis set | exact << 7 (uint8), pair status
               (uint16 LE: the approximate edge, or the pair-level reason when neither side is refused)
    repos[i]: the repo the page's command uses for shape i; configs: repo -> raw config."""
    import base64
    import struct

    from elephantine.emulate import limit_ok, side_support, support
    rows, row_id, sets, set_id, axsets, ax_id = [], {}, [], {}, [], {}
    notes, note_id, sup, sup_id = [""], {"": 0}, [""], {"": 0}

    def intern(table, index, key):
        if key not in index:
            index[key] = len(table)
            table.append(key)
        return index[key]

    def side(i, role):
        c = configs.get(repos[i])
        rs = ["config not in the snapshot"] if c is None else side_support(c, role)
        return intern(sup, sup_id, "\n".join(rs)) if rs else 0

    n = len(P)
    tr = [side(i, "target") for i in range(n)]
    hr = [side(i, "host") for i in range(n)]
    pair_cache, buf = {}, bytearray()
    for i, j in pairs:
        A, B = P[i], P[j]
        plan = edge_plan(A, B, ok=True, exact=bool(exact[i, j]), cap=bool(cap[i, j]))
        ids = tuple(intern(rows, row_id, (r["slot"], r["target"], r["host"], r["kind"], r["currency"],
                                          int(r["exact"]), int(r["payable"]), r["note"])) for r in plan["slots"])
        s = intern(sets, set_id, ids)
        devs = []
        for r in plan["axes"]:
            ax, t, need, have = ZOO_AXES.index(r["axis"]), A["dims"].get(r["axis"], 0), r["need"], r["have"]
            nt = intern(notes, note_id, r["note"])
            if need == t and have == B["dims"].get(r["axis"], 0) and nt == 0:
                continue
            hk = next((k for k in (r["axis"],) + HAVE_KEYS if B["dims"].get(k, 0) == have), None)
            if t and need % t == 0 and hk is not None:
                devs.append((ax, need // t, HAVE_KEYS.index(hk), nt))
            else:
                devs.append((ax, need, have, nt, 1))
        a = intern(axsets, ax_id, tuple(devs))
        v = 0
        lim = not plan["exact"] and limit_ok(plan)
        if not plan["exact"] and not lim:
            v = intern(sup, sup_id, approx_reason(plan))
        elif not tr[i] and not hr[j]:
            key = (repos[i], repos[j], lim)
            if key not in pair_cache:
                rs = support(configs[repos[i]], configs[repos[j]], limit=lim)
                pair_cache[key] = intern(sup, sup_id, "\n".join(rs)) if rs else 0
            v = pair_cache[key]
        assert s < 65536 and a < 128 and v < 65536, (s, a, v)
        buf += struct.pack("<HBH", s, a | (int(plan["exact"]) << 7), v)
    return {"n": len(pairs), "axes": list(ZOO_AXES), "have_keys": list(HAVE_KEYS),
            "rows": [list(r) for r in rows], "sets": [list(x) for x in sets],
            "axsets": [[list(d) for d in x] for x in axsets], "notes": notes, "support": sup, "tr": tr, "hr": hr,
            "eb": base64.b64encode(bytes(buf)).decode(),
            "slot_code": SLOT_CODE, "axis_code": AXIS_CODE, "axis_label": AXIS_LABEL, "axis_pad": AXIS_PAD}


def main():
    from graph.model_graph import classify, dims_of
    from graph.reduce import load_index, resolve
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("host")
    ap.add_argument("--from-json", default="graph/hub_10k.json")
    args = ap.parse_args()
    index = load_index(args.from_json)
    tn, tc = resolve(args.target, index)
    hn, hc = resolve(args.host, index)
    A = {"slots": classify(tc), "dims": dims_of(tc)}
    B = {"slots": classify(hc), "dims": dims_of(hc)}
    if A["slots"] is None or B["slots"] is None:
        raise SystemExit("one of the models did not classify as a decoder LM")
    print(format_plan(edge_plan(A, B), tn, hn))


if __name__ == "__main__":
    main()
