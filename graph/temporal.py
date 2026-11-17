"""Temporal: architecture-feature adoption over time.

Fetches each model's repo creation date, bins by half-year, and reports the
fraction of models created in that period that use each architectural feature.
The population created at time T reflects the architectures popular at T, so the
curves show the field converging (RoPE -> GQA -> SwiGLU -> qk-norm -> MLA)."""
import json
from graph.model_graph import classify, _junky

FEATS = ["rope", "MHA", "GQA/MQA", "MLA", "MoE", "SwiGLU", "qk-norm", "parallel", "sliding"]


def main():
    from huggingface_hub import list_models
    dates = {}
    for m in list_models(filter="text-generation", sort="downloads", direction=-1,
                         limit=5000, expand=["createdAt"]):
        ca = getattr(m, "created_at", None)
        if ca is not None:
            dates[m.id] = ca

    rows = json.load(open("graph/hub_10k.json"))
    bins, matched = {}, 0
    for r in rows:
        if _junky(r["id"]):
            continue
        c = classify(r["config"])
        ca = dates.get(r["id"])
        if not c or ca is None:
            continue
        matched += 1
        key = f"{ca.year}-H{1 if ca.month <= 6 else 2}"
        b = bins.setdefault(key, {f: 0 for f in FEATS} | {"n": 0})
        b["n"] += 1
        b["rope"] += c["pos"] in ("rope", "partial_rope")
        b["MHA"] += c["kv"] == "MHA"
        b["GQA/MQA"] += c["kv"] in ("GQA", "MQA")
        b["MLA"] += c["kv"] == "MLA"
        b["MoE"] += c["ffn"] in ("moe", "moe_shared")
        b["SwiGLU"] += c["ffn"] in ("glu", "moe", "moe_shared")
        b["qk-norm"] += c["qknorm"] != "no"
        b["parallel"] += c["block"] == "parallel"
        b["sliding"] += c["window"] == "local"

    with open("graph/temporal.txt", "w") as f:
        f.write(f"{len(dates)} dates fetched, {matched} models matched. "
                "% of models created each half-year using each feature:\n\n")
        f.write(f"{'period':8} {'n':>4}   " + " ".join(f"{x:>8}" for x in FEATS) + "\n")
        for k in sorted(bins):
            b = bins[k]
            if b["n"] < 4:
                continue
            f.write(f"{k:8} {b['n']:>4}   "
                    + " ".join(f"{100 * b[x] / b['n']:>7.0f}%" for x in FEATS) + "\n")
    print("done")


if __name__ == "__main__":
    main()
