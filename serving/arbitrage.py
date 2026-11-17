"""Price spreads between OpenRouter (the sell side) and Featherless (the buy side).

  book 1: OpenRouter model T whose own repo is on Featherless for less than T's cheapest
          OpenRouter provider (plain resale).
  book 2: T uploads into a *different*, cheaper Featherless class that emulates it exactly
          (emul_matrix(exact=True)), paying that class's long-tail rate.
  book 3: T's checkpoint is EMBEDDED into a bigger Llama-family class (serving.embed: wider
          hidden, more heads / regrouped KV, multiple head_dim, wider FFN, extra layers, padded
          vocab), checked against the host's own serving code, and priced per WORKLOAD: OpenRouter
          endpoints bill cached input at their input_cache_read rate (~20% of input on 565/793
          endpoints), and Featherless has no cache discount.

Prices are blended $/Mtok at 3:1 input:output. Featherless charges uploaded / long-tail repos 4x a
class's official rate (true of 175/179 classes), so an upload is priced at 4 x the class minimum.
Book 2 hits still need a hand check that the host class's serving code realizes the reduction
(e.g. phi4 has no qk-norm, and embedding-only classes don't serve chat).

  python -m serving.arbitrage fetch     # refresh serving/data/{or_models,or_endpoints,fl_models}.json
  python -m serving.arbitrage           # -> serving/data/arbitrage.json
  python -m serving.arbitrage classes [moe]   # uploads that embed exactly into a cheaper class
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

DATA = "serving/data"
UPLOAD_MULT = 4
# (input tokens, output tokens, fraction of input served from cache) per request mix
WORKLOADS = {"chat 3:1": (3, 1, 0.0), "rag 20:1": (20, 1, 0.0), "agent 20:1, 80% cached": (20, 1, 0.8),
             "generation 1:3": (1, 3, 0.0)}


def read_json(name, data=None):
    """serving/data-style JSON from `data` (default DATA), falling back to a gzipped copy (snapshots)."""
    path = os.path.join(data or DATA, name)
    if os.path.exists(path):
        return json.load(open(path))
    with gzip.open(path + ".gz", "rt") as f:
        return json.load(f)


def blend(i, o):
    return (3 * i + o) / 4


def fetch(data=None):
    import requests
    d = data or DATA
    get = lambda u: requests.get(u, timeout=60).json()
    orm = get("https://openrouter.ai/api/v1/models")
    json.dump(orm, open(f"{d}/or_models.json", "w"))
    open_w = [m for m in orm["data"] if m.get("hugging_face_id")]
    with ThreadPoolExecutor(8) as ex:
        eps = dict(ex.map(lambda m: (m["id"], get(f"https://openrouter.ai/api/v1/models/{m['id']}/endpoints")["data"]),
                          open_w))
    json.dump(eps, open(f"{d}/or_endpoints.json", "w"))
    json.dump(get("https://api.featherless.ai/v1/models"), open(f"{d}/fl_models.json", "w"))
    print(f"openrouter {len(orm['data'])} models ({len(eps)} open-weight), featherless refreshed")


def load(data=None):
    orm = {m["id"]: m for m in read_json("or_models.json", data)["data"]}
    eps = read_json("or_endpoints.json", data)
    fl = read_json("fl_models.json", data)["data"]
    sell = {}                                           # OR id -> (hf id, min price, providers)
    for mid, d in eps.items():
        ps = [(blend(float(e["pricing"]["prompt"]) * 1e6, float(e["pricing"]["completion"]) * 1e6), e["provider_name"])
              for e in d.get("endpoints", []) if float(e["pricing"]["prompt"]) > 0]
        if ps:
            sell[mid] = (orm[mid]["hugging_face_id"], min(ps)[0], sorted({p for _, p in ps}))
    classes = defaultdict(list)
    for m in fl:
        classes[m["model_class"]].append((blend(m["pricing"]["input"], m["pricing"]["output"]), m["id"]))
    return sell, fl, classes


def book1(sell, fl):
    by_id = {m["id"].lower(): m for m in fl}
    rows = []
    for mid, (hf, p, provs) in sell.items():
        m = by_id.get(hf.lower())
        if m:
            fp = blend(m["pricing"]["input"], m["pricing"]["output"])
            if fp < p:
                rows.append(dict(target=mid, or_min=round(p, 4), or_providers=provs, fl_price=round(fp, 4),
                                 fl_class=m["model_class"], fl_ctx=m["context_length"], spread=round(1 - fp / p, 3)))
    return sorted(rows, key=lambda r: -r["spread"])


def book2(sell, classes):
    sys.path.insert(0, ".")
    from huggingface_hub import hf_hub_download
    from graph.model_graph import classify, dims_of, emul_matrix, iter_json_configs
    hub = dict(iter_json_configs("graph/hub_10k.json"))

    def point(repos):
        for r in repos:
            try:
                c = hub.get(r) or json.load(open(hf_hub_download(r, "config.json")))
                s = classify(c)
                if s:
                    return r, {"slots": s, "dims": dims_of(c)}
            except Exception:
                pass
        return None, None

    reps = {c: [i for _, i in sorted(v)[:3]] for c, v in classes.items()}       # official repos first
    with ThreadPoolExecutor(8) as ex:
        host = dict(zip(reps, ex.map(point, reps.values())))
        tgt = dict(zip(sell, ex.map(lambda m: point([sell[m][0]]), sell)))
    H = [(c, p) for c, (_, p) in host.items() if p]
    T = [(m, p) for m, (_, p) in tgt.items() if p]
    ok = emul_matrix([p for _, p in T] + [p for _, p in H], exact=True)
    rows = []
    for i, (m, _) in enumerate(T):
        upload = lambda c: round(UPLOAD_MULT * min(p for p, _ in classes[c]), 4)
        best = min(((upload(c), c) for j, (c, _) in enumerate(H) if ok[i, len(T) + j]), default=None)
        if best and best[0] < sell[m][1]:
            rows.append(dict(target=m, or_min=round(sell[m][1], 4), host_class=best[1], host_rep=host[best[1]][0],
                             upload_price=best[0], spread=round(1 - best[0] / sell[m][1], 3)))
    return sorted(rows, key=lambda r: -r["spread"])


def or_cost(pricing, w):
    """$/Mtok of the mix for one OpenRouter endpoint."""
    i, o, cached = WORKLOADS[w]
    pin, pout = float(pricing["prompt"]) * 1e6, float(pricing["completion"]) * 1e6
    pc = float(pricing.get("input_cache_read") or 0) * 1e6 or pin
    return (i * ((1 - cached) * pin + cached * pc) + o * pout) / (i + o)


def fl_cost(pin, pout, w):
    i, o, _ = WORKLOADS[w]                              # no cache discount
    return (i * pin + o * pout) / (i + o)


def book3(classes_raw, data=None):
    """Per OpenRouter target: cheapest Featherless route (own listing, or an upload embedded into a
    handled class per serving.embed.plan) vs the cheapest OpenRouter endpoint, per workload."""
    from huggingface_hub import hf_hub_download
    from serving.embed import FAMILY, plan
    hub = {}
    try:
        from graph.model_graph import iter_json_configs
        hub = dict(iter_json_configs("graph/hub_10k.json"))
    except Exception:
        pass

    def cfg(repo):
        try:
            c = hub.get(repo) or json.load(open(hf_hub_download(repo, "config.json")))
            return c.get("text_config", c) | {"model_type": c.get("model_type")} if "text_config" in c else c
        except Exception:
            return None

    orm = {m["id"]: m for m in read_json("or_models.json", data)["data"]}
    eps = read_json("or_endpoints.json", data)
    fl = read_json("fl_models.json", data)["data"]
    own = {m["id"].lower(): m for m in fl}
    hosts = {}                                          # class -> (rep repo, cfg, upload in, upload out)
    reps = {c: sorted(v, key=lambda m: m["pricing"]["input"])[:3] for c, v in classes_raw.items()}
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(reps, ex.map(lambda ms: next(((m["id"], c) for m in ms for c in [cfg(m["id"])]
                                                     if c and FAMILY.get(c.get("model_type"))), (None, None)),
                                    reps.values())))
        tcfgs = dict(zip(eps, ex.map(lambda mid: cfg(orm[mid]["hugging_face_id"]), eps)))
    for c, (rep, hc) in got.items():
        if hc:
            pmin = min(classes_raw[c], key=lambda m: m["pricing"]["input"])["pricing"]
            hosts[c] = (rep, hc, UPLOAD_MULT * pmin["input"], UPLOAD_MULT * pmin["output"])
    rows = []
    for mid, d in eps.items():
        es = [e for e in d.get("endpoints", []) if float(e["pricing"]["prompt"]) > 0]
        if not es:
            continue
        routes = []
        m = own.get(orm[mid]["hugging_face_id"].lower())
        if m:
            routes.append(("listed " + m["model_class"], m["pricing"]["input"], m["pricing"]["output"], "listed"))
        tc = tcfgs.get(mid)
        if tc and tc.get("model_type") in ("granite", "phi3"):         # exact conversion first
            from serving.convert import _base_cfg, why_not
            if why_not(tc) is None:
                sw = tc.get("model_type") == "phi3" and tc.get("sliding_window")
                tc = {**_base_cfg(tc, "mistral" if sw else "llama"), "sliding_window": sw or None,
                      "tie_word_embeddings": tc.get("model_type") == "phi3" and bool(tc.get("tie_word_embeddings"))}
        if tc and FAMILY.get(tc.get("model_type")):
            for c, (rep, hc, pin, pout) in hosts.items():
                pl = plan(tc, hc)
                if pl["ok"]:
                    routes.append((f"embed->{c}", pin, pout, pl["exact"]))
        if not routes:
            continue
        per = {}
        for w in WORKLOADS:
            o = min(or_cost(e["pricing"], w) for e in es)
            r = min(routes, key=lambda r: fl_cost(r[1], r[2], w))
            f = fl_cost(r[1], r[2], w)
            per[w] = dict(or_min=round(o, 4), fl=round(f, 4), route=r[0], exactness=r[3], spread=round(1 - f / o, 3))
        if any(v["spread"] > 0 for v in per.values()):
            rows.append(dict(target=mid, workloads=per))
    return sorted(rows, key=lambda r: -max(v["spread"] for v in r["workloads"].values()))


def class_embeds(classes_raw, moe_only=False):
    """Featherless classes whose uploads embed exactly (serving.embed.plan) into a CHEAPER class, both
    priced at the long-tail upload rate (4x the class minimum, blended 3:1). Configs: config.json only.
    One row per (upload class, cheapest such host); moe_only keeps rows whose target or host is MoE."""
    from huggingface_hub import hf_hub_download
    from serving.embed import FAMILY, MOE, plan

    def cfg(repo):
        try:
            c = json.load(open(hf_hub_download(repo, "config.json")))
            return c.get("text_config", c) | {"model_type": c.get("model_type")} if "text_config" in c else c
        except Exception:
            return None

    reps = {c: sorted(v, key=lambda m: m["pricing"]["input"])[:3] for c, v in classes_raw.items()}
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(reps, ex.map(lambda ms: next(((m["id"], c) for m in ms for c in [cfg(m["id"])]
                                                     if c and FAMILY.get(c.get("model_type"))), (None, None)),
                                    reps.values())))
    up = {c: UPLOAD_MULT * min(blend(m["pricing"]["input"], m["pricing"]["output"]) for m in classes_raw[c])
          for c in got}
    ok = {c: hc for c, (_, hc) in got.items() if hc}
    rows = []
    for t, tc in ok.items():
        hits = []
        for h, hc in ok.items():
            if h == t or up[h] >= up[t]:
                continue
            if moe_only and tc.get("model_type") not in MOE and hc.get("model_type") not in MOE:
                continue
            pl = plan(tc, hc)
            if pl["ok"]:
                hits.append((up[h], h, pl["exact"], sorted({r[0] for _, r in pl["layers"]})))
        if hits:
            p, h, ex_, routes = min(hits)
            rows.append(dict(target=t, target_rep=got[t][0], upload=round(up[t], 4), host=h, host_rep=got[h][0],
                             host_upload=round(p, 4), saving=round(1 - p / up[t], 3), exact=ex_, routes=routes,
                             members=len(classes_raw[t])))
    return sorted(rows, key=lambda r: -r["saving"]), {c: (got[c][0], (hc or {}).get("model_type")) for c, hc in
                                                      ((c, got[c][1]) for c in got)}


def books(data=None):
    """All three books (plus the Featherless price-structure tally) computed from the pull in `data`."""
    sell, fl, classes = load(data)
    ratio = Counter("official+4x" if {round(p / min(q for q, _ in v), 2) for p, _ in v} <= {1.0, 4.0} else "other"
                    for v in classes.values())
    raw = defaultdict(list)
    for m in fl:
        raw[m["model_class"]].append(m)
    return {"book1": book1(sell, fl), "book2": book2(sell, classes), "book3": book3(raw, data),
            "fl_price_structure": dict(ratio)}


def main():
    if sys.argv[1:] == ["fetch"]:
        return fetch()
    if sys.argv[1:2] == ["classes"]:                    # python -m serving.arbitrage classes [moe]
        raw = defaultdict(list)
        for m in read_json("fl_models.json")["data"]:
            raw[m["model_class"]].append(m)
        rows, _ = class_embeds(raw, moe_only=sys.argv[2:] == ["moe"])
        for r in rows:
            print(f"{r['target']} ({r['upload']:.2f}) -> {r['host']} ({r['host_upload']:.2f}; {r['exact']}; "
                  f"{'/'.join(r['routes'])})  {r['saving']:.0%}  {r['members']} listed")
        return
    out = books()
    json.dump(out, open(f"{DATA}/arbitrage.json", "w"), indent=1)
    print(f"book3: {len(out['book3'])} targets with a positive spread on some workload")
    for r in out["book3"]:
        print(f"  {r['target']}")
        for w, v in r["workloads"].items():
            print(f"     {w:24s} OR {v['or_min']:.3f}  FL {v['fl']:.3f} via {v['route']} ({v['exactness']})  {v['spread']:+.0%}")
    for k in ("book1", "book2"):
        print(f"{k}: {len(out[k])}")
        for r in out[k]:
            print(f"  {r['spread']:.0%}  {r['target']:45s} OR {r['or_min']:.3f} -> "
                  f"{r.get('fl_class') or r['host_class']} {r.get('fl_price') or r['upload_price']:.3f}")


if __name__ == "__main__":
    main()
