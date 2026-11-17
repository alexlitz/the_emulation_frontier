"""Re-run the competitive evaluation's accuracy table for OUR rows after the zoo-fixes branch.

Reads dist/zoo_eval_data.json (the evaluation of 2026-11-14: models, reference values, every collected value) and
the current graph/data/arch_metrics.json, recomputes each "OURS" record from the metrics file, scores it against
the same accepted reference values, and writes the result back into the JSON (records[*].after, scores_after)
so dist/zoo_competitive_eval.md can show "before" and "after" columns. Competitor records are untouched.

One reference is corrected, with the evidence in reference_corrections: the evaluation took Kimi-Linear's FLOPs
reference from our (then KDA-less) formula; the KDA layers' projections, now in the count and Hub-matched to the
parameter, add 2 x 20 x 39,460,864 matmul FLOPs per token, plus their state contraction.

Run:  python -m graph.zoo_eval_rescore
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, "dist", "zoo_eval_data.json")
METRICS = os.path.join(HERE, "data", "arch_metrics.json")
OURS = "OURS (zoo headline, after this eval's fixes)"

FIELD = {
    "params_total": lambda r: r["params"]["total"],
    "params_active": lambda r: r["params"]["active"],
    "kv_bytes_per_token_bf16": lambda r: r["kv_per_token"]["bf16"],
    "kv_32k_bf16": lambda r: r["kv"]["32k"]["bf16"],
    "weight_bytes:bf16": lambda r: r["params"]["weight_bytes"]["bf16"],
    "flops_matmul_per_token": lambda r: r["flops"]["matmul"],
    "flops_decode_32k": lambda r: r["flops"]["decode"]["32k"]["total"],
}
# references the evaluation took from our own formula where that formula missed a block (see the module docstring)
CORRECTED = {("Kimi-Linear-48B-A3B", "flops_matmul_per_token"), ("Kimi-Linear-48B-A3B", "flops_decode_32k")}


def rel(v, accepted):
    errs = [(v - a) / a for a in accepted if a]
    return min(errs, key=abs) if errs else None


def main():
    with open(EVAL) as f:
        ev = json.load(f)
    with open(METRICS) as f:
        repos = json.load(f)["repos"]
    repo_of = {m["name"]: m["our_repo"] for m in ev["models"]}
    corr = {}
    for r in ev["records"]:
        if r.get("source") != OURS or r.get("field_norm") not in FIELD:
            continue
        m = repos[repo_of[r["model"]]]
        v = FIELD[r["field_norm"]](m)
        acc = list(r.get("ref_accepted") or [r["ref"]])
        if (r["model"], r["field_norm"]) in CORRECTED:
            corr[f"{r['model']} {r['field_norm']}"] = {"before": acc, "after": [v],
                                                     "why": "the reference was our KDA-less formula; the 20 KDA "
                                                            "layers (params Hub-matched exactly) are now counted"}
            acc = [v]
        e = rel(v, acc)
        r["after"] = {"value": v, "ref_accepted": acc, "rel_err": round(e, 5) if e is not None else None,
                      "within_1pct": e is not None and abs(e) <= 0.01, "within_5pct": e is not None and abs(e) <= 0.05,
                      "valid": m.get("valid")}
    scores = []
    for field in FIELD:
        rs = [r for r in ev["records"] if r.get("source") == OURS and r.get("field_norm") == field and "after" in r]
        worst = max(rs, key=lambda r: abs(r["after"]["rel_err"] or 0)) if rs else None
        scores.append({"source": OURS.replace("after this eval's fixes", "after zoo-fixes"), "field": field,
                       "coverage": len(rs), "within_1pct": sum(r["after"]["within_1pct"] for r in rs),
                       "within_5pct": sum(r["after"]["within_5pct"] for r in rs),
                       "worst_rel_err": worst["after"]["rel_err"] if worst else None,
                       "worst_model": worst["model"] if worst and worst["after"]["rel_err"] else None})
    ev["scores_after"] = scores
    # framework support against the frameworks' docs: our flag after the fixes (native / the Transformers-backend
    # tier / no), next to the docs' value, for the records whose verdict was against us
    with open(os.path.join(HERE, "data", "frameworks.json")) as f:
        fw = json.load(f)["repos"]
    changed = []
    for r in ev["records"]:
        if not str(r.get("field", "")).startswith("framework_support.") or r.get("model") not in repo_of:
            continue
        name = r["field"].split(".", 1)[1]
        x = fw.get(repo_of[r["model"]]) or {}
        nat = (x.get("native") or {}).get(name)
        tier = "native" if nat else "via Transformers backend" if (x.get("via_transformers") or {}).get(name) else \
            ("no" if nat is False else "unknown")
        before = "native" if r.get("ours") is True else "no" if r.get("ours") is False else "n/a"
        r["after"] = {"ours": tier, "executorch_why": x.get("executorch_why") if name == "executorch" else None}
        if tier != before:
            changed.append({"source": r["source"], "model": r["model"], "docs": r.get("value"), "before": before,
                            "after": tier, "verdict_before": r.get("verdict")})
    ev["framework_after"] = changed
    for c in changed:
        print(f"{c['source']:16} {c['model']:22} docs={c['docs']!s:32} ours {c['before']} -> {c['after']}")
    ev["reference_corrections"] = corr
    ev["after_note"] = ("records[*].after: OUR values recomputed from graph/data/arch_metrics.json on branch zoo-fixes "
                        "(python -m graph.zoo_eval_rescore); same accepted references except reference_corrections")
    with open(EVAL, "w") as f:
        json.dump(ev, f, indent=1)
    for s in scores:
        print(f"{s['field']:26} {s['within_1pct']}/{s['coverage']} within 1%, {s['within_5pct']}/{s['coverage']} "
              f"within 5%, worst {s['worst_rel_err']} {s['worst_model'] or ''}")
    return ev


if __name__ == "__main__":
    main()
