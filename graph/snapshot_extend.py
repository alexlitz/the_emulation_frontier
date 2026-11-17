"""A CANDIDATE extension of the zoo snapshot: the text-capable decoder repos graph/hub_10k.json's listing missed.

graph/fetch_configs.py builds the snapshot from list_models(filter="text-generation"), so a repo tagged
image-text-to-text / any-to-any, or untagged, is never listed even when its config.json holds a causal-LM text model
(google/gemma-4-*, meta-llama/Llama-4-*, google/gemma-3-27b-it, mistralai/Mixtral-*: dist/zoo_missing_flagships.md).
This lists those repos with the snapshot's own criteria and computes what the zoo shows for them. It writes
graph/data/hub_extension.json and changes neither the snapshot nor the relation (the frontier impact is computed at
the final merge).

Criteria, the snapshot's (graph/fetch_configs.py main):
  pool       list_models(pipeline_tag=t, sort="downloads") for t in TAGS, plus the most-downloaded repos of the
             SEED_PUBLISHERS that carry no pipeline tag
  cutoff     downloads >= the snapshot's smallest (its listing stopped there); the snapshot has no likes rule, so
             likes are recorded, not filtered
  junk       the JUNK markers (tiny-random, internal-testing)
  dedup      unique_key (org and quantisation / format markers dropped): the most-downloaded repo per key over the
             snapshot AND the pool; a pool repo is kept only where it wins its key. Where a snapshot derivative held
             the key (nvidia/Gemma-4-31B-IT-NVFP4 for google/gemma-4-31B-it), the original publisher's repo is kept and
             the holder is recorded as `shadows`. One departure: when the key's winner has no config.json (a GGUF
             port), the next most-downloaded repo of the key is tried (up to MAX_FALLBACK), recorded as
             key_fallback_from -- the snapshot's listing loses such identities. And where a derivative holds the
             key (the snapshot's repo or the pool's winner), the original publisher's repo (a SEED_PUBLISHERS org,
             no quantisation marker) is kept too, as original_beside_derivative with shadows = the derivative:
             google/gemma-3-27b-it beside gaunernst/gemma-3-27b-it-int4-awq
  model      config.json read (configs and metadata only, never weights; remote code never run): the zoo's own
             model filter (graph.frontier.load: classify() accepts it -- drafters / encoders / speech are refused --,
             the text dims are non-zero, the name is not junk) and a generative class (graph.frameworks.generative)
  gated      the config is read with the token in use; a gated repo the token cannot read is listed in
             gated_without_config (the Hub API exposes no public config mirror: model_info(expand=["config"]) gives
             only architectures / model_type, not the dims)

For each kept repo: its shape (graph.model_graph._dedup_points over the zoo + the extension: it joins an existing zoo
shape, named by that shape's representative, or is a new shape), arch_metrics (metrics, hub_reason against the Hub
safetensors count, read with graph.fetch_meta.param_total) and framework support (graph.frameworks' native_support /
transformers_tier / the ExecuTorch per-model rule / generative; the backdoor routes of a joined shape are the shape's
own in graph/data/frameworks.json, a new shape's need the relation).

Run:  python -m graph.snapshot_extend           # network: listing, config.json and model_info; cached in CACHE
      python -m graph.snapshot_extend --offline # recompute from the cache only
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "hub_extension.json")
REPORT = os.path.join(os.path.dirname(HERE), "dist", "snapshot_extension.md")
SNAP = os.path.join(HERE, "hub_10k.json")
CACHE = os.environ.get("SNAPSHOT_EXTEND_CACHE", os.path.expanduser("~/.cache/snapshot_extend"))
TAGS = ("image-text-to-text", "any-to-any")
SEED_PUBLISHERS = ("google", "meta-llama", "mistralai", "Qwen", "deepseek-ai", "microsoft", "nvidia", "openai",
                   "moonshotai", "zai-org", "ibm-granite", "allenai", "THUDM", "baidu", "tencent", "LGAI-EXAONE",
                   "stepfun-ai", "MiniMaxAI", "XiaomiMiMo", "inclusionAI", "CohereLabs", "HuggingFaceTB",
                   "tiiuae", "apple", "amazon", "ServiceNow-AI", "internlm", "OpenGVLab", "rednote-hilab")
SEED_LIMIT = 400                      # most-downloaded repos listed per seed publisher
WORKERS = 4
MAX_FALLBACK = 5                      # repos of the same key tried after one without config.json
NO_CONFIG = ("EntryNotFoundError", "JSONDecodeError")
BIG_KEY = 20_000                      # config values larger than this (bytes of JSON) are stored as a marker
FLAGSHIPS = ("google/gemma-3-27b-it", "google/gemma-4-", "mistralai/Mixtral-", "meta-llama/Llama-4-")


# ------------------------------------------------------------------------------------------------ selection (pure)

def snapshot_cutoff(snapshot):
    return min(int(x.get("downloads") or 0) for x in snapshot)


def ranked(pool, snapshot, cutoff=None):
    """{key: (pool rows most downloaded first, shadows)} of the keys a pool repo wins under the snapshot's criteria:
    not in the snapshot, not junk, at or above the cutoff, and more downloaded than the snapshot repo holding its
    unique_key (shadows = that holder, or None). The rows after the first are the fall-backs when the first has no
    config.json (a GGUF / original-format repo: the snapshot's listing kept it and lost the identity, as
    google/gemma-3-27b-it is lost to its GGUF ports)."""
    from graph.fetch_configs import JUNK, unique_key
    cutoff = snapshot_cutoff(snapshot) if cutoff is None else cutoff
    snap_ids = {x["id"] for x in snapshot}
    holder = {}
    for x in snapshot:
        k = unique_key(x["id"])
        if k not in holder or (x.get("downloads") or 0) > (holder[k].get("downloads") or 0):
            holder[k] = x
    by = {}
    seen = set()
    for r in pool:
        if r["id"] in snap_ids or r["id"] in seen or any(j in r["id"].lower() for j in JUNK):
            continue
        seen.add(r["id"])
        if (r.get("downloads") or 0) < cutoff:
            continue
        by.setdefault(unique_key(r["id"]), []).append(r)
    out = {}
    for k, rows in by.items():
        h = holder.get(k)
        rows = sorted(rows, key=lambda r: (-(r.get("downloads") or 0), r["id"]))
        if h is not None:
            rows = [r for r in rows if (r.get("downloads") or 0) > (h.get("downloads") or 0)]
        if rows:
            out[k] = (rows, h["id"] if h is not None else None)
    return out


def is_original(repo):
    """A publisher's own repo: a SEED_PUBLISHERS org and no quantisation / format marker in the name."""
    from graph.fetch_configs import QUANT
    org, _, name = repo.partition("/")
    return org in SEED_PUBLISHERS and not QUANT.search(name)


def originals(pool, snapshot, cutoff=None):
    """[(row, key, shadows)]: where a derivative holds a key (the snapshot's repo, or the pool's winner), the
    original publisher's repo of that key too (is_original, the most downloaded one, at or above the cutoff):
    google/gemma-3-27b-it beside gaunernst/gemma-3-27b-it-int4-awq. shadows = the derivative."""
    from graph.fetch_configs import JUNK, unique_key
    cutoff = snapshot_cutoff(snapshot) if cutoff is None else cutoff
    snap_ids = {x["id"] for x in snapshot}
    win = {}
    for x in list(snapshot) + list(pool):
        if any(j in x["id"].lower() for j in JUNK):
            continue
        k = unique_key(x["id"])
        if k not in win or (x.get("downloads") or 0) > (win[k].get("downloads") or 0):
            win[k] = x
    out, seen = [], set()
    for r in sorted(pool, key=lambda r: (-(r.get("downloads") or 0), r["id"])):
        k = unique_key(r["id"])
        if (r["id"] in snap_ids or r["id"] in seen or k in seen or (r.get("downloads") or 0) < cutoff
                or not is_original(r["id"]) or win[k]["id"] == r["id"] or is_original(win[k]["id"])
                or any(x["id"] in snap_ids and is_original(x["id"]) for x in snapshot if unique_key(x["id"]) == k)):
            continue
        seen.update((r["id"], k))
        out.append((r, k, win[k]["id"]))
    return out


def select(pool, snapshot, cutoff=None):
    """The pool repos the snapshot's criteria keep, one per key: [(row, key, shadows)], most downloaded first."""
    out = [(rows[0], k, sh) for k, (rows, sh) in ranked(pool, snapshot, cutoff).items()]
    out.sort(key=lambda t: (-(t[0].get("downloads") or 0), t[0]["id"]))
    return out


def text_lm(cfg, repo="", tag=None):
    """(bool, why): the zoo's model filter on a raw config (graph.frontier.load) plus a generative class."""
    from graph.frameworks import generative
    from graph.frontier import _junky
    from graph.model_graph import classify, dims_of
    if not isinstance(cfg, dict):
        return False, "no config"
    if _junky(repo):
        return False, "junk name"
    try:
        cl = classify(cfg)
    except Exception as e:                                # a config the classifier cannot read
        return False, f"classify: {type(e).__name__}"
    if not cl:
        return False, "classify: not a decoder LM (drafter / encoder / speech / skipped type)"
    d = dims_of(cfg)
    if not (d["hidden"] and d["layers"]):
        return False, "no text dims"
    ok, why = generative(cfg, tag)
    return (True, None) if ok else (False, why)


# ------------------------------------------------------------------------------------------------ network (cached)

def _cache(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name.replace("/", "__"))


def _retry(fn, tries=6):
    err = None
    for i in range(tries):
        try:
            return fn(), None
        except Exception as e:                            # rate limits / timeouts are retried, genuine misses not
            n = type(e).__name__
            if n in ("EntryNotFoundError", "GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError",
                     "JSONDecodeError"):
                return None, n
            err = n
            time.sleep(min(60, 2 ** i))
    return None, "transient:" + str(err)


def list_pool(offline=False):
    from huggingface_hub import list_models
    rows = []
    exp = ["downloads", "likes", "gated", "pipeline_tag", "library_name"]

    def row(m):
        return {"id": m.id, "downloads": m.downloads or 0, "likes": m.likes or 0, "gated": m.gated,
                "pipeline_tag": m.pipeline_tag, "library": getattr(m, "library_name", None)}
    for tag in TAGS:
        path = _cache(f"list_{tag}.json")
        if not os.path.exists(path) and not offline:
            got = []
            for m in list_models(pipeline_tag=tag, sort="downloads", direction=-1, limit=20000, expand=exp):
                got.append(row(m))
                if (m.downloads or 0) < 1000:
                    break
            json.dump(got, open(path, "w"))
        rows += json.load(open(path)) if os.path.exists(path) else []
    for pub in SEED_PUBLISHERS:
        path = _cache(f"list_author_{pub}.json")
        if not os.path.exists(path) and not offline:
            got, _ = _retry(lambda: [row(m) for m in list_models(author=pub, sort="downloads", direction=-1,
                                                                   limit=SEED_LIMIT, expand=exp)])
            json.dump(got or [], open(path, "w"))
        if os.path.exists(path):
            rows += [r for r in json.load(open(path)) if not r.get("pipeline_tag")]
    return rows


def fetch(repo, offline=False):
    """{config, config_status, hub_total, hub_other} for one repo, cached (config.json and model_info only)."""
    path = _cache(f"repo_{repo}.json")
    if os.path.exists(path):
        rec = json.load(open(path))
        if not str(rec.get("config_status", "")).startswith("transient") or offline:
            return rec
    if offline:
        return {"config": None, "config_status": "not cached", "hub_total": None, "hub_other": None}
    from huggingface_hub import hf_hub_download, model_info
    from graph.fetch_meta import param_total

    def cfg():
        with open(hf_hub_download(repo, "config.json")) as f:
            return json.load(f)
    c, why = _retry(cfg)
    rec = {"config": c, "config_status": "ok" if c is not None else why, "hub_total": None, "hub_other": None}
    if c is not None:
        mi, _ = _retry(lambda: model_info(repo, expand=["safetensors"]))
        if mi is not None:
            rec["hub_total"], rec["hub_other"] = param_total(getattr(mi, "safetensors", None))
    json.dump(rec, open(path, "w"))
    return rec


# ------------------------------------------------------------------------------------------------ the extension

def trim_config(raw):
    """The raw config with any value over BIG_KEY bytes of JSON (per-layer quantisation maps, vocabulary maps)
    replaced by a marker; the metrics are computed from the whole config."""
    out = {}
    for k, v in raw.items():
        n = len(json.dumps(v, default=str))
        out[k] = f"<{n:,d} bytes not stored>" if n > BIG_KEY else v
    return out


def _compact_metrics(m):
    keep = ("model_type", "arch", "max_context", "max_context_source", "max_context_unbounded", "layers", "notes",
            "params", "kv_per_token", "state", "kv_minimum")
    out = {k: m.get(k) for k in keep}
    out["kv"] = {L: {"bf16": v["bf16"], "beyond_max": v["beyond_max"]} for L, v in (m.get("kv") or {}).items()}
    out["flops"] = {"matmul": m["flops"]["matmul"],
                    "decode": {L: v["total"] for L, v in m["flops"]["decode"].items()},
                    "prefill": {L: v["total"] for L, v in m["flops"]["prefill"].items()}}
    out["memory"] = {L: v["batch1"] for L, v in (m.get("memory") or {}).items()}
    return out


# extension repos whose upload is not the model its config describes (the snapshot's params_triage bucket B; the
# evidence: their safetensors headers, read 2026-11-14): params formula-only, the anomaly shown
EXT_ANOMALIES = {
    "openbmb/MiniCPM-OCR-Jev": "a classifier upload: score.weight [2, h] replaces the LM head; Hub = formula - lm_head "
                               "(508,559,360) + score (4,096) + the Qwen3-VL vision tower (331,416,576) exactly",
    "RBDash-Team/Xing-VL2.0-Decision": "the config is untied (tie_word_embeddings false) but the upload stores no lm_head: "
                                       "Hub = formula - lm_head (1,271,398,400) + the Qwen3-VL vision tower "
                                       "(460,730,096) exactly",
    "fastino/GLiDE-27B": "as Xing-VL2.0-Decision: an untied config, no lm_head in the upload; Hub = formula - lm_head + "
                         "vision tower exactly",
    "kiel2/Kiel-2-VQA": "the upload is not the config's model: a Qwen2.5-VL-7B config over a LLaVA-1.5-7B checkpoint (a "
                        "Llama-7B language_model with a 32,064-row vocab, a CLIP-L vision_tower, a multi_modal_projector)",
    "mobilint/Qwen3-VL-8B-Instruct-Batch16": "NPU-compiled .mxq binaries (text-W4V8.mxq, vision-W4V8.mxq); "
                                             "model.safetensors holds only the embedding and a few tensors",
}


def _valid_params(name, raw, p, prm, hub):
    """The zoo's params status for a non-traced repo (graph.arch_metrics.build): hub-matched, failed (an
    unexplained Hub difference above 0.5% of an unquantised checkpoint) or formula-only."""
    from graph import arch_metrics as AM
    st, why = AM.hub_reason(p, raw, prm["total"], hub, name)
    if name in EXT_ANOMALIES:
        return "formula-only", why
    if st == "hub-matched":
        return "hub-matched", why
    if st == "unexplained" and not AM.is_quantised(name, raw) and hub and abs(hub - prm["total"]) > 0.005 * prm["total"]:
        return "failed", why
    return "formula-only", why


def frameworks_of(name, cfg, tag, shape_members, configs, fws):
    """graph.frameworks' per-repo support for one extension repo (native, the Transformers tier, ExecuTorch per
    model, generative)."""
    from graph.frameworks import _family, export_dims, generative, native_support, transformers_tier
    nat = native_support(cfg, fws)
    why = None
    if nat.get("executorch"):
        et = fws["executorch"]
        et_dims = {(_family(c["model_type"]), (c["hidden"], c["layers"], c["heads"], c["kv_heads"])): c["file"]
                   for c in et.get("export_configs") or []}
        et_ids = {x["id"]: x["source"] for x in et.get("model_ids") or []}
        named = next((m for m in shape_members if m in et_ids), None)
        dm = export_dims(cfg)
        hit = et_dims.get((_family(cfg.get("model_type")), dm)) if dm else None
        if named:
            why = f"same shape as {named} ({et_ids[named]})"
        elif hit:
            why = f"export recipe {hit}"
        else:
            nat["executorch"] = False
            why = "model type registered, but no export recipe for this model (size / variant)"
    out = {"native": nat}
    tf = transformers_tier(cfg, fws, nat)
    if tf:
        out["via_transformers"] = tf
    if why:
        out["executorch_why"] = why
    g, gw = generative(cfg, tag)
    if not g:
        out["not_generative"] = gw
    return out


def build(offline=False, log=print):
    from graph import arch_metrics as AM
    from graph.framework_registries import load as load_registries
    from graph.frontier import load
    from graph.model_graph import _dedup_points, classify, dims_of, iter_json_configs
    t0 = time.time()
    snapshot = json.load(open(SNAP))
    cutoff = snapshot_cutoff(snapshot)
    pool = list_pool(offline)
    rk = ranked(pool, snapshot, cutoff)
    cand = select(pool, snapshot, cutoff)
    log(f"pool {len(pool)} rows -> {len(cand)} candidate keys (cutoff {cutoff:,d} downloads)")

    def first_with_config(t):
        r, k, sh = t
        tried = []
        for alt in rk[k][0][:MAX_FALLBACK + 1]:
            rec = fetch(alt["id"], offline)
            if rec["config"] is not None or rec["config_status"] not in NO_CONFIG:
                return (alt, k, sh), dict(rec, no_config_first=tried)
            tried.append(alt["id"])
        return t, dict(fetch(r["id"], offline), no_config_first=tried[1:])
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(first_with_config, cand))
    orig = originals(pool, snapshot, cutoff)
    kept_ids = {t[0]["id"] for t, _ in res}
    orig = [t for t in orig if t[0]["id"] not in kept_ids]
    log(f"+ {len(orig)} original-publisher repos of keys a derivative holds")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res += [((t[0], t[1], t[2]), dict(rec, original=True)) for t, rec in
                zip(orig, ex.map(lambda t: fetch(t[0]["id"], offline), orig))]
    cand = [t for t, _ in res]
    got = [rec for _, rec in res]
    zoo = load()
    zoo_names = {m["name"] for m in zoo}
    snap_ids = {x["id"] for x in snapshot}
    configs = dict(iter_json_configs(SNAP))
    kept, dropped, gated = [], {}, []
    for (r, k, sh), rec in zip(cand, got):
        if rec["config"] is None:
            if rec["config_status"] == "GatedRepoError" or (r.get("gated") and rec["config_status"] != "EntryNotFoundError"):
                gated.append({"id": r["id"], "downloads": r["downloads"], "gated": r.get("gated"),
                              "pipeline_tag": r.get("pipeline_tag"), "status": rec["config_status"]})
            else:
                dropped[r["id"]] = rec["config_status"]
            continue
        ok, why = text_lm(rec["config"], r["id"], r.get("pipeline_tag"))
        if not ok:
            dropped[r["id"]] = why
            continue
        kept.append((r, k, sh, rec))
    new_models = [{"name": r["id"], "slots": classify(rec["config"]), "dims": dims_of(rec["config"])}
                  for r, k, sh, rec in kept]
    P0 = _dedup_points(zoo)
    P = _dedup_points(zoo + new_models)
    am = json.load(open(AM.OUT))
    shapes = am["shapes"]
    assert len(shapes) == len(P0)
    shape_of = {nm: i for i, p in enumerate(P) for nm in p["members"]}
    fws = load_registries()["frameworks"]
    fwj = json.load(open(os.path.join(HERE, "data", "frameworks.json")))
    allcfg = dict(configs)
    allcfg.update({r["id"]: rec["config"] for r, k, sh, rec in kept})
    repos = {}
    for r, k, sh, rec in kept:
        nm, raw = r["id"], rec["config"]
        i = shape_of[nm]
        joins = shapes[i]["rep"] if i < len(P0) else None
        new_peers = sorted(m for m in P[i]["members"] if m not in zoo_names)
        ent = {"downloads": r["downloads"], "likes": r.get("likes"), "pipeline_tag": r.get("pipeline_tag"),
               "gated": r.get("gated"), "key": k, "shadows": sh, "joins_shape": joins, "new_shape": joins is None,
               "shape_peers_new": [m for m in new_peers if m != nm], "hub_total": rec["hub_total"],
               "key_fallback_from": rec.get("no_config_first") or [], "original_beside_derivative":
               bool(rec.get("original")), "config": trim_config(raw)}
        try:
            p = AM.plan(raw)
            m = AM.metrics(raw, rec["hub_total"], p=p)
            v, why = _valid_params(nm, raw, p, m["params"], rec["hub_total"])
            m["params"]["hub_delta_reason"] = why
            m["params"]["checkpoint_anomaly"] = EXT_ANOMALIES.get(nm)
            ent.update(metrics=_compact_metrics(m), hub_reason=why, valid_params=v,
                       checkpoint_anomaly=EXT_ANOMALIES.get(nm))
        except Exception as e:
            ent.update(metrics=None, hub_reason=None, valid_params=None, metrics_error=f"{type(e).__name__}: {e}"[:200])
        ent["frameworks"] = frameworks_of(nm, raw, r.get("pipeline_tag"), P[i]["members"], allcfg, fws)
        if joins is not None:
            ent["frameworks"]["backdoor_shape"] = i            # graph/data/frameworks.json shapes[i].backdoor
            assert fwj["shapes"][i]["rep"] == P0[i]["members"][0]
        repos[nm] = ent
    new_shape_keys = {shape_of[nm] for nm, e in repos.items() if e["new_shape"]}
    joined = {}
    for nm, e in repos.items():
        if e["joins_shape"]:
            joined.setdefault(e["joins_shape"], []).append(nm)
    vp = {}
    for e in repos.values():
        vp[e["valid_params"]] = vp.get(e["valid_params"], 0) + 1
    flag = {f: sorted(nm for nm in repos if nm.startswith(f)) for f in FLAGSHIPS}
    counts = {"pool": len(pool), "candidate_keys": len(cand) - len(orig), "candidates": len(cand), "kept": len(repos), "new_shapes": len(new_shape_keys),
              "repos_in_new_shapes": sum(e["new_shape"] for e in repos.values()),
              "repos_joining_zoo_shapes": sum(not e["new_shape"] for e in repos.values()),
              "zoo_shapes_joined": len(joined), "shadowing_a_snapshot_key": sum(e["shadows"] in snap_ids for e in repos.values()),
              "originals_beside_a_derivative": sum(e["original_beside_derivative"] for e in repos.values()),
              "kept_after_a_config_less_repo": sum(bool(e["key_fallback_from"]) for e in repos.values()),
              "gated_without_config": len(gated), "dropped": len(dropped), "valid_params": vp,
              "seconds": round(time.time() - t0, 1)}
    why_dropped = {}
    for w in dropped.values():
        w = str(w).split(":")[0] if str(w).startswith("classify") or str(w).startswith("transient") else str(w)
        why_dropped[w] = why_dropped.get(w, 0) + 1
    counts["dropped_by_reason"] = why_dropped
    return {"generated": datetime.date.today().isoformat(), "snapshot": "graph/hub_10k.json",
            "criteria": {"tags": list(TAGS), "untagged_seed_publishers": list(SEED_PUBLISHERS),
                         "seed_limit": SEED_LIMIT, "downloads_cutoff": cutoff, "likes": "recorded, not filtered "
                         "(the snapshot has no likes rule)", "junk": "graph.fetch_configs.JUNK",
                         "dedup": "graph.fetch_configs.unique_key, most downloaded over snapshot + pool",
                         "model": "graph.frontier.load's filter (classify, text dims, junk) + graph.frameworks.generative"},
            "counts": counts, "flagships": flag, "joined": joined, "repos": repos, "gated_without_config": gated,
            "dropped": dropped}


def report(d):
    """dist/snapshot_extension.md from the written JSON."""
    from collections import Counter
    R = d["repos"]
    c = d["counts"]
    L = []
    w = L.append
    w = L.append
    w("# Candidate snapshot extension: the text LMs the snapshot's listing missed (zoo-errors, %s)" % d["generated"])
    w("")
    w("A **candidate**: `graph/hub_10k.json` and the relation are unchanged. `python -m graph.snapshot_extend` lists the repos")
    w("tagged image-text-to-text / any-to-any, plus the untagged repos of %d large publishers, keeps those whose config.json" % len(SEED_PUBLISHERS))
    w("holds a causal-LM text model, and computes what the zoo shows for them (`graph/data/hub_extension.json`). Configs and")
    w("metadata only (config.json and model_info), never weights; no remote code run. The frontier / relation impact is")
    w("left to the final merge (no emul_matrix here). A rerun from the cache (`--offline`) takes about %d s; from scratch, about" % round(c["seconds"]))
    w("4 minutes (2,500 config.json fetches and the safetensors counts, 4 workers).")
    w("")
    w("## Criteria (the snapshot's, `graph/fetch_configs.py`)")
    w("")
    w(f"- **Pool:** {c['pool']:,d} listing rows: `list_models(pipeline_tag=...)` for image-text-to-text and any-to-any, sorted by")
    w(f"  downloads, plus the top {SEED_LIMIT} repos of each seed publisher that carry no pipeline tag (Mixtral is untagged).")
    w(f"- **Cutoff:** downloads >= {d['criteria']['downloads_cutoff']:,d}, the snapshot's smallest. The snapshot has no likes rule, so likes are")
    w("  recorded, not filtered.")
    w("- **Junk and dedup:** `JUNK`, and `unique_key` over the snapshot and the pool together: a pool repo is kept only where it")
    w("  is the most downloaded of its key. Two departures, both recorded per repo:")
    w(f"  - when the key's winner has no config.json (a GGUF port), the next repo of the key is tried ({c['kept_after_a_config_less_repo']} repos, `key_fallback_from`);")
    w(f"  - where a derivative holds the key, the original publisher's repo is kept beside it ({c['originals_beside_a_derivative']} repos, `original_beside_derivative`;")
    w("    google/gemma-3-27b-it beside gaunernst/gemma-3-27b-it-int4-awq, Llama-4-Maverick-17B-128E-Instruct beside its FP8 repo).")
    w("- **Model:** the zoo's own filter (`graph.frontier.load`: `classify` accepts it, the text dims are non-zero) and a generative")
    w("  class (`graph.frameworks.generative`).")
    w("- **Gated:** read with the owner's token. A gated repo it cannot read is listed below; the Hub API has no public config")
    w("  mirror (`model_info(expand=[\"config\"])` gives only architectures / model_type, not the dims).")
    w("")
    w("## Counts")
    w("")
    w("| | |")
    w("|---|---:|")
    for k, lab in (("candidate_keys", "candidate keys (after cutoff, junk, dedup)"), ("candidates", "repos fetched (+ originals beside a derivative)"),
                   ("kept", "**kept: text-LM repos**"), ("repos_joining_zoo_shapes", "join an existing zoo shape"),
                   ("zoo_shapes_joined", "zoo shapes joined"), ("repos_in_new_shapes", "in new shapes"), ("new_shapes", "**new shapes**"),
                   ("shadowing_a_snapshot_key", "shadow a snapshot repo holding their key"), ("gated_without_config", "gated, no config readable"),
                   ("dropped", "dropped")):
        w(f"| {lab} | {c[k]:,d} |")
    w("")
    w("Dropped: " + ", ".join(f"{v:,d} {k}" for k, v in sorted(c["dropped_by_reason"].items(), key=lambda x: -x[1])) +
      " (EntryNotFoundError = no config.json: GGUF / original-format repos; classify = not a decoder LM: drafters, encoders,")
    w("speech, OCR / diffusion pipelines the classifier skips).")
    w("")
    w("## Flagships now covered")
    w("")
    w("| repo | downloads | gated | shape | params (formula) | params check |")
    w("|---|---:|---|---|---:|---|")
    flag = []
    for f in FLAGSHIPS:
        flag += d["flagships"][f]
    big = sorted((nm for nm, e in R.items() if nm.split("/")[0] in SEED_PUBLISHERS and is_original(nm) and nm not in flag),
                 key=lambda nm: -R[nm]["downloads"])[:40]
    def row(nm):
        e = R[nm]
        sh = f"joins `{e['joins_shape']}`" if e["joins_shape"] else "**new shape**"
        t = e["metrics"]["params"]["total"]
        w(f"| `{nm}` | {e['downloads']:,d} | {e['gated'] or 'no'} | {sh} | {t / 1e9:.2f}B | {e['valid_params']} ({e['hub_reason']}) |")
    for nm in sorted(flag, key=lambda nm: -R[nm]["downloads"]):
        row(nm)
    w("")
    w("Other publishers' own repos in the extension (the 40 most downloaded):")
    w("")
    w("| repo | downloads | gated | shape | params (formula) | params check |")
    w("|---|---:|---|---|---:|---|")
    for nm in big:
        row(nm)
    w("")
    w("## Existing zoo shapes the extension joins (the 25 largest)")
    w("")
    w("| zoo shape (representative) | extension repos | e.g. |")
    w("|---|---:|---|")
    for rep, ms in sorted(d["joined"].items(), key=lambda kv: -len(kv[1]))[:25]:
        ms = sorted(ms, key=lambda m: -R[m]["downloads"])
        w(f"| `{rep}` | {len(ms)} | {', '.join('`' + m + '`' for m in ms[:3])} |")
    w("")
    w("## New shapes (the 30 most downloaded)")
    w("")
    seen, rows = set(), []
    for nm in sorted((nm for nm, e in R.items() if e["new_shape"]), key=lambda nm: -R[nm]["downloads"]):
        e = R[nm]
        key = tuple(sorted([nm] + e["shape_peers_new"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append((nm, e))
    w("| repo | downloads | model type | other extension repos of the shape |")
    w("|---|---:|---|---:|")
    for nm, e in rows[:30]:
        w(f"| `{nm}` | {e['downloads']:,d} | {e['metrics']['model_type']} | {len(e['shape_peers_new'])} |")
    w("")
    w("## Params check on the extension")
    w("")
    vp = c["valid_params"]
    w(f"hub-matched {vp.get('hub-matched', 0):,d}, formula-only {vp.get('formula-only', 0):,d} (quantised uploads, towers, no Hub count),")
    w(f"**failed {vp.get('failed', 0)}**. The 17 checks that failed at the first run were triaged: 9 by builders (the")
    w("snapshot-extension families of graph/arch_metrics.py's BESPOKE: Inkling, Ling-3 BailingMoeV3, LLaDA-MoE, fuyu,")
    w("Dynin-Omni, Intern-S2-Mobius, stock Mamba-2 keys for Mamba-Codestral, Nemotron-H Puzzle per-block experts), 3 as")
    w("quantised uploads the markers missed (MLX `quantization` key, oQ / bpw names), and these as checkpoint anomalies")
    w("(EXT_ANOMALIES, the snapshot's bucket B: formula-only, the figures the architecture's):")
    w("")
    w("| repo | anomaly | Hub difference |")
    w("|---|---|---|")
    for nm in sorted(EXT_ANOMALIES):
        if nm in R:
            w(f"| `{nm}` | {EXT_ANOMALIES[nm]} | {R[nm]['hub_reason']} |")
    if vp.get("failed"):
        w("")
        w("Still failed:")
        w("")
        w("| repo | model type | formula | Hub difference |")
        w("|---|---|---:|---|")
        for nm, e in sorted(R.items()):
            if e["valid_params"] == "failed":
                w(f"| `{nm}` | {e['metrics']['model_type']} | {e['metrics']['params']['total'] / 1e9:.2f}B | {e['hub_reason']} |")
    w("")
    w("## Framework support")
    w("")
    nat = Counter()
    for e in R.values():
        for fw, v in e["frameworks"]["native"].items():
            nat[fw] += bool(v)
    w("Native support per framework over the %d kept repos (graph.frameworks' per-repo rules, ExecuTorch per model): " % len(R) +
      ", ".join(f"{fw} {n:,d}" for fw, n in nat.most_common()) + ".")
    w("A repo that joins a zoo shape has that shape's backdoor routes (`frameworks.backdoor_shape` indexes")
    w("graph/data/frameworks.json `shapes`); a new shape's routes need the relation (final merge).")
    w("")
    w("## What is left")
    w("")
    w(f"- **Gated, no config readable with this token ({c['gated_without_config']}):** " +
      ", ".join(f"`{g['id']}`" for g in sorted(d["gated_without_config"], key=lambda g: -g["downloads"])) + ".")
    w("- **No config.json:** GGUF / original-format repos (the `*-qat-q4_0-gguf` Gemma repos, Llama-4 `*-Original`); nothing to classify.")
    w("- The relation / frontier impact of the new shapes (computed at the final merge).")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report", default=REPORT)
    a = ap.parse_args(argv)
    d = build(a.offline)
    with open(a.out, "w") as f:
        json.dump(d, f, separators=(",", ":"))
    print(json.dumps(d["counts"], indent=1))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        with open(a.report, "w") as f:
            f.write(report(d))
        print(f"wrote {a.report}")


if __name__ == "__main__":
    sys.exit(main())
