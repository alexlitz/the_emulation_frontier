"""graph/data/recipes.json: per zoo model and framework, the recipe route, its cost and its validation status
(recipes/RECIPES_SCHEMA.md). Written by `python -m recipes.make_recipe --index`."""
from __future__ import annotations

import datetime
import json
import os
import subprocess
from collections import Counter

from recipes.make_recipe import (FRAMEWORKS, OUT_JSON, ROOT, VALIDATION, exact_host_config, exact_support,
                                 loader_of, native_ok, params_of, slug)

SCHEMA_VERSION = 1
CONFIGS = os.path.join(ROOT, "graph", "hub_10k.json")
FRAMEWORKS_JSON = os.path.join(ROOT, "graph", "data", "frameworks.json")
CLI_FRAMEWORKS = ("vllm", "llama.cpp", "mlx-lm")
MAX_EXACT_TRIES = 8
ROUTES = {
    "native": "the framework's Llama-style reference runs the target's own config and weights",
    "exact-host": "an exact edge into a Llama-style host: elephantine/emulate.py v1 rebuilds the target's weights in "
                  "the host's architecture (exactly), and the framework's Llama-style reference runs that checkpoint",
    "universal": "the universal host (elephantine/universal.py) set to the target by its from_* loader, exported as "
                 "tef-ucore and run by the framework's universal-host reference",
    "limit-host": "reserved: a payable (limit / floor / regime) edge, approximate; v1 builds none, so no model "
                  "carries this route",
}


def frameworks_json():
    """frameworks.json from the working tree, else from the zoo-metrics branch, else None."""
    if os.path.isfile(FRAMEWORKS_JSON):
        return json.load(open(FRAMEWORKS_JSON))
    try:
        out = subprocess.run(["git", "show", "zoo-metrics:graph/data/frameworks.json"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except Exception:
        return None


def features(cfg):
    """The Llama-style feature set of a natively runnable config (what a validation must have covered)."""
    from recipes.common.llama_spec import llama_spec
    s = llama_spec(cfg)
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    typ = (rs.get("rope_type", rs.get("type", "default")) if isinstance(rs, dict) else "default") or "default"
    f = {"mt:" + s["model_type"], "rope:" + typ, "tied" if s["tie"] else "untied"}
    if s["rot"] < s["hd"]:
        f.add("partial_rope")
    if any(w is not None for w in s["windows"]):
        f.add("window")
    for k in ("qkv_bias", "o_bias", "mlp_bias", "qk_norm"):
        if s[k]:
            f.add(k)
    if s["KV"] < s["H"]:
        f.add("gqa")
    return f


def _validation():
    if not os.path.isfile(VALIDATION):
        return {"entries": {}, "frameworks": {}}
    return json.load(open(VALIDATION))


def _ok(e):
    return bool(e.get("pass") or e.get("pass_floor"))


def _shapes_and_exact(fwj):
    """(P, exact) over the zoo's shapes in frameworks.json order: exact[i, j] = j emulates i exactly."""
    from graph.frontier import load
    from graph.model_graph import _dedup_points, emul_matrix
    P = _dedup_points(load())
    if fwj is not None and len(P) != len(fwj["shapes"]):
        raise RuntimeError(f"{len(P)} shapes here, {len(fwj['shapes'])} in frameworks.json: regenerate one of them")
    return P, emul_matrix(P, mode="exact")


def _cli(fw, repo, n, fwj, cfgs, support_cache):
    nat = fwj["repos"].get(repo, {}).get("native", {}).get(fw)
    tgt = f"./{slug(repo)}"
    run = {"vllm": "vllm serve {m}",
           "llama.cpp": "python convert_hf_to_gguf.py {m} --outfile {s}.gguf && llama-cli -m {s}.gguf "
                        "-p 'The capital of France is' -n 32 --temp 0",
           "mlx-lm": "mlx_lm.generate --model {m} --prompt 'The capital of France is' --max-tokens 32 --temp 0"}[fw]
    if nat:
        m = repo if fw != "llama.cpp" else tgt
        pre = f"huggingface-cli download {repo} --local-dir {tgt} && " if fw == "llama.cpp" else ""
        return {"route": "native", "cmd": pre + run.format(m=m, s=slug(repo))}
    shape = fwj["repos"].get(repo, {}).get("shape")
    bd = fwj["shapes"][shape]["backdoor"].get(fw, {}) if shape is not None else {}
    hosts = bd.get("hosts") or []
    if not hosts:
        return {"route": None, "why": "not native, and no natively supported shape emulates it"}
    from elephantine.emulate import support as emulate_support
    bad = None
    for h in hosts:                                   # the listed hosts, best first: the first v1 builds
        key = (repo, h["host"])
        if key not in support_cache:
            hraw = cfgs.get(h["host"])
            support_cache[key] = emulate_support(cfgs.get(repo, {}), hraw) if (hraw and h["exact"]) else (
                ["the edge is payable (approximate): v1 builds exact edges only"] if hraw else ["host config not in snapshot"])
        bad = support_cache[key]
        if bad is None:
            break
    if bad is not None:
        h = hosts[0]
        bad = support_cache[(repo, h["host"])]
    d = f"./{slug(repo)}-in-{slug(h['host'])}"
    out = {"route": "backdoor", "host": h["host"], "edge": h["edge"], "params_x": h.get("params_x"),
           "buildable": bad is None}
    if bad is None:
        out["cmd"] = f"python -m elephantine.emulate {repo} {h['host']} --out {d} && " + run.format(m=d, s=slug(repo))
    else:
        out["why_not_buildable"] = "; ".join(bad)[:300]
    return out


def build_index():
    from graph.model_graph import ZOO_SNAPSHOT, iter_json_configs
    from recipes.make_recipe import SCRIPTS
    from recipes.validate import script_path
    fwj = frameworks_json()
    cfgs = dict(iter_json_configs(CONFIGS))
    order = {n: k for k, n in enumerate(cfgs)}                    # snapshot order = downloads
    if fwj is not None:
        repos = list(fwj["repos"])
    else:
        from graph.frontier import load
        repos = [m["name"] for m in load()]
    V = _validation()
    ents = [e for e in V["entries"].values() if _ok(e)]
    # what each framework has validated
    e2e = {fw: {} for fw in FRAMEWORKS}
    loaders_ok = {fw: {} for fw in FRAMEWORKS}
    native_feats = {fw: set() for fw in FRAMEWORKS}
    exact_ok = {fw: None for fw in FRAMEWORKS}
    caveats = {fw: {} for fw in FRAMEWORKS}                    # failing checkpoint runs, by code path
    for e in V["entries"].values():
        if e.get("kind") == "checkpoint" and not _ok(e) and e.get("framework") in caveats:
            key = e.get("loader") if e.get("route") == "universal" else next((f for f in e.get("features") or [] if f.startswith("mt:")), None)
            if "max_abs_diff" in e:
                msg = (f"{e['id']}: greedy {'identical' if e.get('greedy_identical') else 'different'}, "
                       f"max |d logit| {e['max_abs_diff']:.1e} vs transformers fp32 ({e.get('fp64_max_abs_diff', 0):.1e} "
                       f"vs fp64; transformers' own fp32 error {e.get('hf32_fp64_floor', 0):.1e})")
            else:
                msg = f"{e['id']}: failed: " + e.get("error", "")[-200:]
            caveats[e["framework"]].setdefault(key, []).append(msg)
    for e in ents:
        fw = e["framework"]
        if e.get("kind") == "checkpoint":
            e2e[fw][(e["target"], e["route"])] = e["id"]
        if e["route"] == "universal" and e.get("loader"):
            loaders_ok[fw].setdefault(e["loader"], e["id"])
        if e["route"] in ("native", "exact-host") and e.get("features"):
            native_feats[fw] |= set(e["features"])
        if e["route"] == "exact-host":
            exact_ok[fw] = exact_ok[fw] or e["id"]
    # routes (framework-independent: every framework here has both references)
    native, loader, why = {}, {}, {}
    for r in repos:
        c = cfgs.get(r, {})
        native[r] = native_ok(c)
        loader[r], why[r] = loader_of(c)
    # exact-host candidates for the targets the Llama-style reference cannot run
    from elephantine.emulate import TARGET_FAMILIES
    exact_route = {}
    need = [r for r in repos if native[r] is not None and cfgs.get(r, {}).get("model_type") in TARGET_FAMILIES]
    if need and fwj is not None:
        P, ex = _shapes_and_exact(fwj)
        host_of = {}
        for j, p in enumerate(P):
            ok = [m for m in p["members"] if native.get(m, native_ok(cfgs.get(m, {}))) is None]
            if ok:
                host_of[j] = min(ok, key=lambda m: order.get(m, 1 << 30))
        hp = {j: params_of(cfgs[h]) for j, h in host_of.items()}
        for r in need:
            i = fwj["repos"][r]["shape"]
            tp = params_of(cfgs[r])
            cands = sorted((j for j in host_of if ex[i, j]), key=lambda j: (hp[j] or 1e30, host_of[j]))
            for j in cands[:MAX_EXACT_TRIES]:
                if exact_support(cfgs[r], cfgs[host_of[j]]) is None:
                    exact_route[r] = {"host": host_of[j], "params_x": round(hp[j] / tp, 4) if hp[j] and tp else None,
                                      "width_x": round(P[j]["dims"]["hidden"] / max(P[i]["dims"]["hidden"], 1), 4),
                                      "depth_x": round(P[j]["dims"]["eff_layers"] / max(P[i]["dims"]["eff_layers"], 1), 4)}
                    break
    models, counts = {}, {fw: Counter() for fw in FRAMEWORKS}
    support_cache = {}
    for r in repos:
        c = cfgs.get(r, {})
        m = {"model_type": c.get("model_type"), "shape": fwj["repos"][r]["shape"] if fwj else None,
             "loader": loader[r], "frameworks": {}}
        if native[r] is None:
            route, host, cost = "native", None, {"params_x": 1.0, "width_x": 1.0, "depth_x": 1.0}
        elif r in exact_route:
            x = exact_route[r]
            route, host, cost = "exact-host", x["host"], {k: x[k] for k in ("params_x", "width_x", "depth_x")}
        elif loader[r] is not None:
            route, host, cost = "universal", "universal", {"params_x": None, "width_x": 1.0, "depth_x": 1.0}
        else:
            route, host, cost = None, None, None
        feats = features(c) if route == "native" else (
            features(exact_host_config(c, cfgs[host])) if route == "exact-host" else None)
        for fw in FRAMEWORKS:
            ent = {"route": route, "host": host, "cost": dict(cost) if cost else None, "script": None,
                   "make": f"python -m recipes.make_recipe {r} --framework {fw}" if route else None,
                   "status": None, "validated_by": None, "why": None}
            if route is None:
                ent["why"] = f"native: {native[r]}; universal: {why[r]}"
                counts[fw]["none"] += 1
            else:
                counts[fw][route] += 1
                vid = e2e[fw].get((r, route))
                if vid:
                    ent["status"], ent["validated_by"] = "e2e", vid
                    ve = V["entries"][vid]
                    if route == "universal" and ve.get("run_params") and params_of(c):
                        ent["cost"]["params_x"] = round(ve["run_params"] / params_of(c), 4)
                elif route == "universal" and loader[r] in loaders_ok[fw]:
                    ent["status"], ent["validated_by"] = "family", loaders_ok[fw][loader[r]]
                elif route in ("native", "exact-host") and feats <= native_feats[fw] and (
                        route == "native" or exact_ok[fw]):
                    ent["status"] = "family"
                    ent["validated_by"] = exact_ok[fw] if route == "exact-host" else None
                else:
                    ent["status"] = "untested"
                    ref = [f"{k}: {x['refused']}" for k, x in V.get("export_refusals", {}).items()
                           if x.get("loader") == loader[r]]
                    ent["why"] = (f"loader {loader[r]} not validated in {fw}" + (
                        f" (tef-ucore refuses its validation case {ref[0]})" if ref else "") if route == "universal" else
                                  f"features {sorted(feats - native_feats[fw])} not validated in {fw}"
                                  if feats - native_feats[fw] else f"no exact-host validation in {fw}")
                counts[fw][ent["status"]] += 1
                ck = loader[r] if route == "universal" else next((f for f in feats if f.startswith("mt:")), None)
                if ent["status"] == "family" and caveats[fw].get(ck):
                    ent["caveat"] = ("a checkpoint run on this code path missed the pass criteria: "
                                     + "; ".join(caveats[fw][ck]))
                    counts[fw]["family_with_caveat"] += 1
                sp = script_path(fw, r, None, None)
                if os.path.isfile(sp):
                    ent["script"] = os.path.relpath(sp, ROOT)
            m["frameworks"][fw] = ent
        if fwj is not None:
            m["native_cli"] = {fw: _cli(fw, r, None, fwj, cfgs, support_cache) for fw in CLI_FRAMEWORKS}
        models[r] = m
    for fw in FRAMEWORKS:
        counts[fw]["models"] = len(repos)
        counts[fw]["tested_route"] = counts[fw]["e2e"] + counts[fw]["family"]
    fwinfo = {}
    for fw in FRAMEWORKS:
        info = dict(V.get("frameworks", {}).get(fw) or {"package": fw, "version": None, "python": None, "venv": None})
        info["references"] = sorted(os.path.relpath(os.path.join(ROOT, "recipes", "frameworks", fw, f), ROOT)
                                    for f in ("llama.py", "universal.py")
                                    if os.path.isfile(os.path.join(ROOT, "recipes", "frameworks", fw, f)))
        fwinfo[fw] = info
    val = sorted(V["entries"].values(), key=lambda e: e["id"])
    return {"schema_version": SCHEMA_VERSION, "generated": datetime.date.today().isoformat(), "snapshot": ZOO_SNAPSHOT,
            "frameworks": fwinfo, "routes": ROUTES, "validation": val, "export_refusals": V.get("export_refusals", {}),
            "models": models,
            "counts": {fw: dict(c) for fw, c in counts.items()},
            "native_cli_source": "graph/data/frameworks.json (zoo-metrics)" if fwj is not None else None}


def write_index():
    data = build_index()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    for fw, c in data["counts"].items():
        print(f"{fw:10s} models {c['models']}  native {c.get('native', 0)}  exact-host {c.get('exact-host', 0)}  "
              f"universal {c.get('universal', 0)}  none {c.get('none', 0)}  |  e2e {c.get('e2e', 0)}  "
              f"family {c.get('family', 0)}  untested {c.get('untested', 0)}  -> tested route {c['tested_route']}")
    return data
