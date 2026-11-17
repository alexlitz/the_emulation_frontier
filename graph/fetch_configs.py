"""Fetch the top-N most-downloaded text-generation configs from the HF Hub.

Downloads only each repo's config.json (a few KB) -- never weights -- and writes
graph/hub_configs.json = [{id, downloads, config}, ...] for graph.model_graph.

Run:
  python -m graph.fetch_configs --limit 500
  python -m graph.model_graph --from-json graph/hub_configs.json --abstract
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download, list_models

# quantization / format / precision markers -> the SAME model, different bytes.
QUANT = re.compile(
    r"(?i)(fp8|fp16|bf16|fp4|nvfp4|int8|int4|8bit|4bit|w8a8|w4a16|awq|gptq|gptqmodel|"
    r"gguf|mlx|bnb|nf4|hqq|quantized|q[0-9]_[0-9a-z_]+|-4bit|-8bit)")
JUNK = ("tiny-random", "internal-testing", "-random-", "/tiny-random")


def unique_key(mid):
    """Normalize a repo id to its distinct-model identity (drop org, quant/format)."""
    name = mid.split("/")[-1].lower()
    name = QUANT.sub("", name)
    return re.sub(r"[-_.\s]+", "-", name).strip("-")


# genuine misses (no config.json / gated / gone); anything else is retried
GENUINE = ("EntryNotFoundError", "GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError")


def fetch_one(mid, tries=6):
    """-> (id, config | None, reason). Transient failures (rate limits, timeouts, the misleading
    LocalEntryNotFoundError a dropped connection raises) are retried with backoff; silently
    dropping them under-sampled the earlier snapshots by ~40%."""
    for i in range(tries):
        try:
            with open(hf_hub_download(mid, "config.json")) as f:
                return mid, json.load(f), "ok"
        except Exception as e:
            name = type(e).__name__
            if name in GENUINE:
                return mid, None, name
            if name == "JSONDecodeError":
                return mid, None, name
            time.sleep(min(60, 2 ** i))
    return mid, None, "transient:" + name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="number of UNIQUE models to keep")
    ap.add_argument("--pool", type=int, default=1500, help="raw listing size to dedup from")
    ap.add_argument("--filter", default="text-generation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="graph/hub_configs.json")
    args = ap.parse_args()

    pool = list(list_models(filter=args.filter, sort="downloads", direction=-1, limit=args.pool))
    # dedup: keep the most-downloaded repo per normalized identity
    best = {}
    for m in pool:
        if any(j in m.id.lower() for j in JUNK):
            continue
        dl = getattr(m, "downloads", 0) or 0
        k = unique_key(m.id)
        if k not in best or dl > best[k][1]:
            best[k] = (m.id, dl)
    uniq = sorted(best.values(), key=lambda x: -x[1])[:args.limit]
    print(f"listed {len(pool)} -> {len(best)} unique -> fetching config.json for top {len(uniq)} ...")

    rows, why = [], Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, mid): (mid, dl) for mid, dl in uniq}
        for i, fut in enumerate(as_completed(futs), 1):
            mid, cfg, reason = fut.result()
            why[reason.split(":")[0]] += 1
            if cfg:
                rows.append({"id": mid, "downloads": dict(uniq).get(mid, 0), "config": cfg})
            if i % 500 == 0:
                print(f"  {i}/{len(uniq)}  {dict(why)}")
    ok, miss = why["ok"], sum(v for k, v in why.items() if k != "ok")

    rows.sort(key=lambda r: -r["downloads"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f)
    print(f"wrote {args.out}: {ok} unique configs; misses by reason: "
          f"{ {k: v for k, v in why.items() if k != 'ok'} }")
    if why["transient"]:
        print(f"WARNING: {why['transient']} configs failed transiently after retries; rerun to fill them")


if __name__ == "__main__":
    main()
