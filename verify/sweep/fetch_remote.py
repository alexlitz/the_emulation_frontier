"""Fetch the Hub remote code (auto_map modules and their relative imports) of every snapshot config, as TEXT.

Nothing fetched here is imported or executed: the files are read by build_inventory.py / extract_hunks.py as source.
Output: ~/.cache/code_sweep/remote_index.json  {repo: {file: local_path | null}}.
"""
import ast
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.expanduser("~/.cache/code_sweep/remote_index.json")


def auto_maps(c):
    out = []
    if isinstance(c, dict):
        if isinstance(c.get("auto_map"), dict):
            out.append(c["auto_map"])
        for k, v in c.items():
            if k != "auto_map" and isinstance(v, dict):
                out += auto_maps(v)
    return out


def wanted(models):
    """(repo, file) for every module an auto_map names."""
    pairs = set()
    for m in models:
        for a in auto_maps(m.get("config") or {}):
            for v in a.values():
                for x in (v if isinstance(v, list) else [v]):
                    if not isinstance(x, str):
                        continue
                    repo = m["id"]
                    if "--" in x:
                        repo, x = x.split("--", 1)
                    pairs.add((repo, x.rsplit(".", 1)[0].replace(".", "/") + ".py"))
    return pairs


def rel_imports(path):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except Exception:
        return set(re.findall(r"from \.([A-Za-z0-9_]+) import", open(path, errors="replace").read()))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module:
            mods.add(n.module)
    return mods


def fetch(pair):
    repo, f = pair
    try:
        return pair, hf_hub_download(repo, f)
    except Exception as e:  # gated / deleted / missing file
        return pair, None


def main():
    models = json.load(open(os.path.join(ROOT, "graph/hub_10k.json")))
    idx = json.load(open(OUT)) if os.path.exists(OUT) and "--fresh" not in sys.argv else {}
    todo = {p for p in wanted(models) if p[1] not in idx.get(p[0], {})}
    with ThreadPoolExecutor(16) as ex:
        while todo:
            nxt = set()
            for (repo, f), path in ex.map(fetch, sorted(todo)):
                idx.setdefault(repo, {})[f] = path
                if path:
                    for mod in rel_imports(path):
                        g = mod.replace(".", "/") + ".py"
                        if g not in idx[repo]:
                            nxt.add((repo, g))
            todo = nxt - {(r, f) for r in idx for f in idx[r]}
            print(len(todo), "relative imports to fetch", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(idx, open(OUT, "w"), indent=1, sort_keys=True)
    n = sum(len(v) for v in idx.values())
    print(n, "files,", sum(1 for v in idx.values() for p in v.values() if p is None), "missing")


if __name__ == "__main__":
    main()
