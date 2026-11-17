"""Per-finding carrier table for the round-3 specs (verify/r3/carriers.json).

Inputs (gitignored sweep outputs, read from a worktree that holds them; pass --raw DIR):
  hunks_raw.jsonl / inventory_raw.json              the code-diff sweep (findings.json's 118 keys)
  hunks_raw.jsonl.<ext> / inventory_raw.json.<ext>  the no-code hunt (reviews/nc_*.jsonl, nocode_dsv41.jsonl)
and the reviews (verify/sweep/reviews here plus the no-code hunt's, pass --nc-reviews DIR).
For every UNMAPPED finding: its hunks (id, file, class.member, reason), the carrier models (snapshot ids whose code holds
a hunk), the placed ones (classify() is not None) and their model types. classify() is called read-only.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--nc-reviews", required=True)
    a = ap.parse_args()
    from graph import model_graph as MG
    aliases = json.load(open(os.path.join(ROOT, "verify/sweep/finding_aliases.json")))
    H, var_models = {}, collections.defaultdict(set)
    for suf in ["", ".ext", ".ext2", ".needle", ".tfmain", ".dsv41"]:
        for line in open(os.path.join(a.raw, "hunks_raw.jsonl" + suf)):
            h = json.loads(line)
            H.setdefault(h["id"], h)
        for v, ms in json.load(open(os.path.join(a.raw, "inventory_raw.json" + suf)))["var_models"].items():
            var_models[v] |= set(ms)
    R = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "verify/sweep/reviews/*.jsonl"))) + sorted(
            glob.glob(os.path.join(a.nc_reviews, "n*.jsonl"))):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                r["_src"] = os.path.basename(p)
                R[r["id"]] = r
    snap = json.load(open(os.path.join(ROOT, "graph/hub_10k.json")))
    cfg_of = {m["id"]: m["config"] or {} for m in snap}
    placed, mt_of = {}, {}
    for m, cfg in cfg_of.items():
        try:
            placed[m] = MG.classify(cfg) is not None
        except Exception:
            placed[m] = False
        lm = MG._lm_unwrap(cfg)
        mt_of[m] = str((lm if isinstance(lm, dict) else {}).get("model_type") or cfg.get("model_type") or "<none>")
    F = collections.defaultdict(lambda: dict(hunks=[], models=set(), types=set(), dirs=collections.Counter(), src=set()))
    for hid, r in R.items():
        if r.get("label") != "UNMAPPED" or hid not in H:
            continue
        h = H[hid]
        k = aliases.get(r.get("finding"), r.get("finding")) or "unnamed"
        f = F[k]
        ms = set()
        for o in h["occ"]:
            ms |= var_models.get(o["variant"], set())
        f["models"] |= ms & set(cfg_of)
        f["types"] |= set(h["types"])
        f["dirs"][r.get("direction")] += 1
        f["src"].add(r["_src"])
        o = h["occ"][0]
        f["hunks"].append(dict(id=hid, file=o["file"], member=f"{o['cls']}.{o['member']}", kind=h["kind"],
                               reason=r.get("reason"), suggest=r.get("suggest")))
    out = {}
    for k, f in F.items():
        pm = sorted(m for m in f["models"] if placed.get(m))
        out[k] = dict(types=sorted(f["types"]), direction=f["dirs"].most_common(1)[0][0], sources=sorted(f["src"]),
                      carriers=sorted(f["models"]), placed=pm,
                      placed_types=dict(collections.Counter(mt_of[m] for m in pm)), hunks=f["hunks"])
    json.dump(out, open(os.path.join(HERE, "carriers.json"), "w"), indent=1, sort_keys=True)
    print(len(out), "findings")


if __name__ == "__main__":
    main()
