"""Fetch per-repo release date (createdAt) and real parameter count
(the safetensors parameter counts) for every config in the snapshot, cached to
graph/model_meta.json. Serves both the frontier-over-time analysis and the
corrected coverage figure. Idempotent: re-running only fetches missing ids.

The parameter count is the Hub API's safetensors.total, unless that is broken: the API's total can disagree
with its own per-dtype safetensors.parameters by orders of magnitude (Thox-ai/ThoxOS-Kernel-27B: total 3,054,832
while the per-dtype counts sum to the LM plus its Qwen3-VL vision tower), and then the per-dtype sum is used. Where the two
disagree, the figure not used is kept (hf_total_api / hf_total_dtype_sum, see param_total).

  python -m graph.fetch_meta                  # fetch the ids not cached yet
  python -m graph.fetch_meta --recheck        # re-read the parameter counts of the cached ids too
  python -m graph.fetch_meta --recheck REPO   # ... of the named ids only
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import model_info

SNAP = "graph/hub_10k.json"
OUT = "graph/model_meta.json"
OTHER_KEYS = ("hf_total_api", "hf_total_dtype_sum")


def param_total(sf):
    """(total, other) of a SafeTensorsInfo-like object.

    The API's safetensors.total is the parameter count (for a quantised checkpoint it counts the unpacked
    weights, where the per-dtype counts hold the packed elements and the scales), so it is kept, unless it is
    broken: below half the sum of the per-dtype counts, which no packing produces (ThoxOS-Kernel-27B 3,054,832
    against 27,356,728,560; Hermes-4-14B 424,960 against 14,768,307,200), and then the per-dtype sum is the
    count. `other` is the figure not used, when the two differ (stored as hf_total_api or hf_total_dtype_sum)."""
    if not sf:
        return None, None
    api = getattr(sf, "total", None) or None
    per = getattr(sf, "parameters", None) or {}
    s = sum(int(v) for v in per.values()) if per else None
    if s and (api is None or api < s / 2):
        return s, ({"hf_total_api": api} if api is not None else None)
    if api is not None and s and s != api:
        return api, {"hf_total_dtype_sum": s}
    return api or s, None


def one(repo):
    try:
        mi = model_info(repo, expand=["createdAt", "safetensors"])
        ca = getattr(mi, "created_at", None)
        total, other = param_total(getattr(mi, "safetensors", None))
        rec = {"created_at": ca.isoformat() if ca else None, "hf_total": total}
        rec.update(other or {})
        return repo, rec
    except Exception as e:                       # gated / 404 / renamed
        return repo, {"created_at": None, "hf_total": None, "err": type(e).__name__}


def recheck(cache, ids, errors=None):
    """Re-read the parameter counts of ids (keeping the cached date); returns the changed ids. Ids the API
    refuses now (gone, gated, rate-limited) keep their cached record and are listed in errors."""
    changed = []
    errors = [] if errors is None else errors
    with ThreadPoolExecutor(max_workers=4) as ex:
        for repo, meta in ex.map(one, ids):
            old = cache.get(repo) or {}
            if meta.get("err"):
                errors.append((repo, meta["err"]))
                continue
            new = {k: v for k, v in old.items() if k not in OTHER_KEYS}
            new.update({k: meta[k] for k in ("hf_total",) + OTHER_KEYS if k in meta})
            if not new.get("created_at"):
                new["created_at"] = meta["created_at"]
            if new != old:
                changed.append((repo, old.get("hf_total"), new.get("hf_total")))
            cache[repo] = new
    return changed


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ids = [x["id"] for x in json.load(open(SNAP))]
    try:
        cache = json.load(open(OUT))
    except FileNotFoundError:
        cache = {}
    if argv and argv[0] == "--recheck":
        want = argv[1:] or [i for i in ids if i in cache]
        errors = []
        changed = recheck(cache, want, errors)
        json.dump(cache, open(OUT, "w"))
        for repo, a, b in changed:
            print(f"  {repo}: {a} -> {b}")
        print(f"rechecked {len(want)} ids, {len(changed)} changed, {len(errors)} refused (kept) -> {OUT}")
        return
    todo = [i for i in ids if i not in cache]
    print(f"{len(ids)} ids, {len(cache)} cached, {len(todo)} to fetch")
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for repo, meta in ex.map(one, todo):
            cache[repo] = meta
            done += 1
            if done % 100 == 0:
                json.dump(cache, open(OUT, "w"))
                print(f"  {done}/{len(todo)}")
    json.dump(cache, open(OUT, "w"))
    ok = sum(1 for v in cache.values() if v.get("created_at"))
    tot = sum(1 for v in cache.values() if v.get("hf_total"))
    print(f"done: {len(cache)} cached, {ok} with dates, {tot} with HF totals -> {OUT}")


if __name__ == "__main__":
    main()
