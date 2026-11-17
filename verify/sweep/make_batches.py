"""Cut the hunks no rule decides into review batches, grouped by family (the LM type most of a hunk's models carry).

    python verify/sweep/make_batches.py [max_lines]     # writes verify/sweep/batches/bNN.jsonl + batches/index.json

A hunk that already has a review line (reviews/*.jsonl) is not batched again.
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rules import rule  # noqa: E402


def reviewed():
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "reviews", "*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main():
    max_lines = int(sys.argv[1]) if len(sys.argv) > 1 else 3500
    H = [json.loads(l) for l in open(os.path.join(HERE, "hunks_raw.jsonl"))]
    inv = json.load(open(os.path.join(HERE, "inventory_raw.json")))
    done = reviewed()
    var_types = {}
    models = {m["id"]: m for m in json.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                                               "graph/hub_10k.json")))}
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from graph import model_graph as MG
    lm_type = {}
    for mid, m in models.items():
        lm = MG._lm_unwrap(m["config"] or {})
        lm_type[mid] = str((lm or {}).get("model_type") or (m["config"] or {}).get("model_type") or "<none>")
    for v, mids in inv["var_models"].items():
        var_types[v] = collections.Counter(lm_type[m] for m in mids)
    fam = collections.defaultdict(list)
    for h in H:
        if rule(h) or h["id"] in done:
            continue
        c = collections.Counter()
        for o in h["occ"]:
            c.update(var_types.get(o["variant"], {}))
        key = c.most_common(1)[0][0] if c else "<none>"
        fam[key].append(h)
    os.makedirs(os.path.join(HERE, "batches"), exist_ok=True)
    for p in glob.glob(os.path.join(HERE, "batches", "*.jsonl")):
        os.remove(p)
    batches, cur, cur_lines, cur_fams = [], [], 0, []
    size = lambda h: len(h["text"].splitlines()) + 3
    for f in sorted(fam, key=lambda f: -sum(map(size, fam[f]))):
        hs = sorted(fam[f], key=lambda h: (h["role"], h["occ"][0]["cls"], h["member"]))
        for h in hs:
            if cur and cur_lines + size(h) > max_lines and (f not in cur_fams or cur_lines > max_lines):
                batches.append((cur, cur_fams))
                cur, cur_lines, cur_fams = [], 0, []
            cur.append(h)
            cur_lines += size(h)
            if f not in cur_fams:
                cur_fams.append(f)
    if cur:
        batches.append((cur, cur_fams))
    index = []
    for i, (hs, fams) in enumerate(batches):
        name = f"b{i:02d}"
        with open(os.path.join(HERE, "batches", name + ".jsonl"), "w") as fh:
            for h in hs:
                fh.write(json.dumps({k: h[k] for k in ("id", "kind", "role", "member", "occ", "text", "types",
                                                        "models", "placed_models")}) + "\n")
        index.append(dict(batch=name, hunks=len(hs), lines=sum(map(size, hs)), families=fams))
    json.dump(index, open(os.path.join(HERE, "batches", "index.json"), "w"), indent=1)
    for x in index:
        print(x["batch"], x["hunks"], x["lines"], ", ".join(x["families"][:8]), "..." if len(x["families"]) > 8 else "")


if __name__ == "__main__":
    main()
