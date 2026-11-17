"""Show sweep hunks with what the classifier makes of the models that carry them.

    python verify/sweep/show.py <hunk id> [...]          # the hunk text, its occurrences, and classify() of 4 models
    python verify/sweep/show.py --type <model_type>      # every hunk of a type, one line each
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)


def load():
    p = os.path.join(HERE, "hunks_raw.jsonl")      # (not committed: python verify/sweep/sweep.py regenerates it)
    if not os.path.exists(p):
        p = os.path.join(HERE, "hunks.jsonl")
    out = {h["id"]: h for h in map(json.loads, open(p))}
    for q in sorted(glob.glob(os.path.join(HERE, "hunks_raw.jsonl.*"))):    # partial runs (SWEEP_TAG)
        if not q.endswith(".tmp"):
            out.update({h["id"]: h for h in map(json.loads, open(q))})
    return out


def main():
    from graph import model_graph as MG
    H = load()
    p = os.path.join(HERE, "inventory_raw.json")
    inv = json.load(open(p if os.path.exists(p) else os.path.join(HERE, "inventory.json")))
    for q in sorted(glob.glob(os.path.join(HERE, "inventory_raw.json.*"))):   # partial runs (SWEEP_TAG)
        if not q.endswith(".tmp"):
            for v, ms in json.load(open(q))["var_models"].items():
                inv["var_models"].setdefault(v, ms)
    models = {m["id"]: m for m in json.load(open(os.path.join(ROOT, "graph/hub_10k.json")))}
    args = sys.argv[1:]
    if args[:1] == ["--type"]:
        for h in H.values():
            if args[1] in h["types"]:
                o = h["occ"][0]
                print(h["id"], h["kind"], h["role"], (o["cls"] if isinstance(o, dict) else o.split(" ")[2]), h["member"], h["models"])
        return
    for i in args:
        h = H[i]
        print("=" * 100)
        print(i, h["kind"], h["role"], h["member"], "models", h["models"], "placed", h["placed_models"])
        print("types:", ", ".join(h["types"][:30]))
        for o in h["occ"][:6]:
            print("  occ:", (o["variant"], o["file"], o["cls"], "<-", o["parent"]) if isinstance(o, dict) else o)
        if len(h["occ"]) > 6:
            print("  ... +", len(h["occ"]) - 6, "occurrences")
        print(h["text"][:6000])
        vs = {o["variant"] if isinstance(o, dict) else o.split(" ")[0] for o in h["occ"]}
        mids = sorted({m for v in vs for m in inv["var_models"].get(v, [])})[:4]
        for m in mids:
            c = models[m]["config"]
            try:
                r = MG.classify(c)
            except Exception as e:
                r = f"error {e}"
            print("  classify", m, "->", r)


if __name__ == "__main__":
    main()
