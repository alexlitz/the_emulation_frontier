"""Fetch the Hub pipeline tag (and library name) of every zoo repo: metadata only, from the HF model API.

The snapshot (graph/hub_10k.json) keeps only id, downloads and config.json. The backdoor host ranking
(graph.frameworks) needs the task a repo is published for, to exclude embedding / reward / classifier models
(e.g. Alibaba-NLP/gte-Qwen2-7B-instruct, a Qwen2ForCausalLM config tagged sentence-similarity). No code and no
weights are downloaded. Writes graph/data/pipeline_tags.json = {repo: {"pipeline_tag", "library_name"} | null}.

Run:  python -m graph.fetch_pipeline_tags
"""
from __future__ import annotations

import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "pipeline_tags.json")


def _one(api, repo, tries=5):
    for i in range(tries):
        try:
            info = api.model_info(repo)
            return repo, {"pipeline_tag": getattr(info, "pipeline_tag", None),
                          "library_name": getattr(info, "library_name", None)}
        except Exception as e:                        # gated / gone repos still answer model_info; retry the rest
            name = type(e).__name__
            if name in ("RepositoryNotFoundError", "GatedRepoError", "RevisionNotFoundError"):
                return repo, None
            time.sleep(min(30, 2 ** i))
    return repo, None


def main(workers=8):
    from huggingface_hub import HfApi

    from graph.frontier import load
    repos = [m["name"] for m in load()]
    old = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            old = json.load(f).get("repos", {})
    todo = [r for r in repos if r not in old]
    api = HfApi()
    out = dict(old)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, api, r) for r in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            r, v = fu.result()
            out[r] = v
            if k % 500 == 0:
                print(f"  {k}/{len(todo)}")
    data = {"fetched": datetime.date.today().isoformat(), "source": "huggingface_hub HfApi.model_info (metadata)",
            "repos": {r: out.get(r) for r in repos}}
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"), sort_keys=True)
    miss = sum(1 for r in repos if not out.get(r))
    print(f"wrote {OUT}: {len(repos)} repos, {miss} without metadata")


if __name__ == "__main__":
    main()
