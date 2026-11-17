"""End-to-end validation of the recipes: generate each script, build its run directory, run it in the
framework's own venv, compare with transformers (recipes/compare.py), and record the result in
recipes/validation.json (which graph/data/recipes.json is built from).

    python -m recipes.validate [--only ID ...] [--frameworks pytorch jax ...] [--force]
    python -m recipes.validate --ingest-cases DIR      # record recipes/run_cases results (random-weight cases)

The venvs are found under $TEF_RECIPES_VENVS (default $TEF_SCRATCH/recipes-venvs/<framework>; TEF_SCRATCH
defaults to ~/.cache/the_emulation_frontier); a framework
whose venv is missing is skipped. The build stage runs in this (the repository's) environment.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recipes.make_recipe import FRAMEWORKS, VALIDATION, plan, emit, slug  # noqa: E402

SCRATCH = os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier"))
VENVS = os.environ.get("TEF_RECIPES_VENVS", os.path.join(SCRATCH, "recipes-venvs"))
WORK = os.environ.get("TEF_RECIPES_WORK", os.path.join(SCRATCH, "recipes-runs"))
SCRIPTS = os.path.join(ROOT, "recipes", "scripts")
NEW_TOKENS = 32

# id -> (target, route, host); route None = the route make_recipe picks
TARGETS = {
    "native/HuggingFaceTB/SmolLM2-135M": ("HuggingFaceTB/SmolLM2-135M", None, None),
    "universal/openai-community/gpt2": ("openai-community/gpt2", None, None),
    "universal/EleutherAI/pythia-14m": ("EleutherAI/pythia-14m", None, None),
    "exact-host/HuggingFaceTB/SmolLM2-135M@HuggingFaceTB/SmolLM2-360M":
        ("HuggingFaceTB/SmolLM2-135M", "exact-host", "HuggingFaceTB/SmolLM2-360M"),
    "native/Qwen/Qwen2.5-0.5B": ("Qwen/Qwen2.5-0.5B", None, None),
    "universal/facebook/opt-125m": ("facebook/opt-125m", None, None),
    "universal/bigcode/tiny_starcoder_py": ("bigcode/tiny_starcoder_py", None, None),
    "universal/google/gemma-3-270m": ("google/gemma-3-270m", None, None),
}


def venv_python(fw):
    p = os.path.join(VENVS, fw, "bin", "python")
    return p if os.path.isfile(p) else None


def _which(name):
    import shutil
    return shutil.which(name)


def fw_env(fw):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "2"), TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1",
               TF_CPP_MIN_LOG_LEVEL="2", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2",
               JAX_PLATFORMS="cpu", DEV="CPU", PARALLEL="2", NUM_CPU_THREADS="2")   # tinygrad: 2 compile workers
    if fw == "tinygrad" and "CC" not in os.environ and not _which("clang"):
        env["CC"] = os.path.join(ROOT, "recipes", "frameworks", "tinygrad", "zigcc.py")   # zig cc (ziglang wheel)
        env["TEF_ZIG_PYTHON"] = venv_python("tinygrad")
    if fw == "tinygrad":
        env.setdefault("CACHEDB", os.path.join(VENVS, "tinygrad", "tinygrad-cache.db"))
    env.pop("PYTHONPATH", None)
    return env


def _venv_label(venv):
    """The venv as recorded in validation.json: relative to $TEF_RECIPES_VENVS (no machine paths in the data)."""
    rel = os.path.relpath(venv, VENVS)
    return "$TEF_RECIPES_VENVS/" + rel if not rel.startswith("..") else os.path.basename(venv)


def fw_version(fw):
    py = venv_python(fw)
    if py is None:
        return None
    pkg = {"pytorch": "torch", "jax": "jax", "tinygrad": "tinygrad", "tensorflow": "tensorflow", "dynet": "dynet"}[fw]
    code = ("import sys, importlib.metadata as m\nfor n in %r:\n    try:\n        print(m.version(n)); break\n"
            "    except Exception: pass\nprint(sys.version.split()[0])" % ([pkg, pkg + "-cpu", "dyNET"],))
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, env=fw_env(fw)).stdout.split()
    return {"package": pkg, "version": out[0] if out else None, "python": out[-1] if out else None,
            "venv": _venv_label(os.path.dirname(os.path.dirname(py)))}


def script_path(fw, target, route, host):
    name = slug(target)
    if route == "exact-host":
        name += "--in--" + slug(host)
    elif route is not None:
        name += "--" + route
    return os.path.join(SCRIPTS, fw, name + ".py")


def write_script(fw, target, route, host):
    p = plan(target, fw, host=host, route=route)
    path = script_path(fw, target, route, host)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(emit(target, fw, p))
    os.chmod(path, 0o755)
    return path, p


def reference(target):
    out = os.path.join(WORK, "refs", slug(target) + ".npz")
    if not os.path.isfile(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run([sys.executable, "-m", "recipes.reference", target, out], check=True, cwd=ROOT,
                       env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2")))
    return out


def build(script, target, route, host, force=False):
    d = os.path.join(WORK, "build", os.path.basename(script)[:-3])
    if os.path.isfile(os.path.join(d, "config.json")) and not force:
        return d
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "recipes", "tef_recipes"),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"), USE_TF="0")
    subprocess.run([sys.executable, script, "build", "--out", d], check=True, env=env, cwd=WORK)
    return d


def load_validation():
    if os.path.isfile(VALIDATION):
        return json.load(open(VALIDATION))
    return {"entries": {}, "frameworks": {}}


def save_validation(v, ids=None):
    """Write v merged into what is on disk (several validate runs may share the file): the entries `ids`
    (default: all of v's) replace the disk's, the others on disk are kept."""
    disk = load_validation()
    disk["entries"].update(v["entries"] if ids is None else {k: v["entries"][k] for k in ids})
    disk["frameworks"].update(v.get("frameworks", {}))
    for k in v:
        if k not in ("entries", "frameworks"):
            disk[k] = v[k]
    v.update(disk)
    tmp = VALIDATION + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(disk, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, VALIDATION)


def _script_changed(fw, target, route, host, rel):
    """The checked-in script differs from what make_recipe writes today (code changed since it ran)."""
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        return True
    return open(path).read() != emit(target, fw, plan(target, fw, host=host, route=route))


def run_one(fw, vid, target, route, host, v, force=False, timeout=7200, save=True, stale=False):
    import numpy as np
    from recipes.compare import compare
    eid = f"{fw}/{vid}"
    if eid in v["entries"] and not force:
        old = v["entries"][eid]
        if not stale or not old.get("script") or not _script_changed(fw, target, route, host, old["script"]):
            return old
    py = venv_python(fw)
    if py is None:
        return None
    ref = reference(target)
    script, p = write_script(fw, target, route, host)
    d = build(script, target, route, host)
    ids = ",".join(map(str, np.load(ref)["ids"].tolist()))
    out = os.path.join(WORK, "out", f"{fw}--{slug(vid)}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    r = subprocess.run([py, script, "run", "--dir", d, "--ids", ids, "--new-tokens", str(NEW_TOKENS), "--npz", out],
                       capture_output=True, text=True, env=fw_env(fw), cwd=WORK, timeout=timeout)
    secs = round(time.time() - t0, 1)
    e = {"id": eid, "framework": fw, "route": p["route"], "target": target, "host": p.get("host"),
         "loader": p.get("loader"), "kind": "checkpoint", "script": os.path.relpath(script, ROOT),
         "seconds": secs, "date": datetime.date.today().isoformat()}
    if r.returncode != 0:
        e.update({"pass": False, "pass_floor": False, "error": (r.stderr or r.stdout)[-1500:]})
    else:
        e.update(compare(ref, out))
        cfg = json.load(open(os.path.join(d, "config.json")))
        if cfg.get("format") == "tef-ucore":
            e["run_params"] = cfg.get("n_params")
        else:
            from recipes.index import features
            e["features"] = sorted(features(cfg))
    v["entries"][eid] = e
    if save:
        save_validation(v, [eid])
    return e


def ingest_cases(cases_dir, v):
    """Record recipes/run_cases.py results: tiny random-weight configs (kind "random")."""
    idx = json.load(open(os.path.join(cases_dir, "cases.json")))
    v["export_refusals"] = {k.split("/", 1)[1]: {"loader": r.get("loader"), "model_type": r.get("model_type"),
                                                 "refused": r.get("refused") or r.get("error")}
                            for k, r in idx.items() if k.startswith("universal/") and (r.get("refused") or r.get("error"))}
    n = 0
    for fw in FRAMEWORKS:
        for kind in ("universal", "llama"):
            p = os.path.join(cases_dir, f"res-{fw}-{kind}.json")
            if not os.path.isfile(p):
                continue
            for case, r in json.load(open(p)).items():
                meta = idx.get(f"{kind}/{case}", {})
                route = "universal" if kind == "universal" else "native"
                e = {"id": f"{fw}/{route}-random/{case}", "framework": fw, "route": route, "target": f"random:{case}",
                     "host": "universal" if kind == "universal" else None, "loader": meta.get("loader"),
                     "model_type": meta.get("model_type"), "kind": "random", "date": datetime.date.today().isoformat()}
                e.update({k: r.get(k) for k in ("max_abs_diff", "decode_max_abs_diff", "greedy_identical",
                                                "greedy_tokens", "logit_scale", "pass", "error", "seconds")
                          if k in r})
                e["pass_floor"] = e.get("pass", False)
                if kind == "llama":
                    from recipes.index import features
                    e["features"] = sorted(features(json.load(open(os.path.join(cases_dir, "llama", case, "config.json")))))
                v["entries"][e["id"]] = e
                n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--frameworks", nargs="*", default=list(FRAMEWORKS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stale", action="store_true", help="rerun entries whose script changed since they ran")
    ap.add_argument("--ingest-cases")
    ap.add_argument("--recompare", action="store_true", help="recompute the comparisons from the saved runs")
    a = ap.parse_args(argv)
    v = load_validation()
    if a.recompare:
        from recipes.compare import compare
        for e in v["entries"].values():
            if e.get("kind") != "checkpoint" or e.get("error"):
                continue
            vid = e["id"].split("/", 1)[1]
            out = os.path.join(WORK, "out", f"{e['framework']}--{slug(vid)}.npz")
            if os.path.isfile(out):
                e.update(compare(reference(e["target"]), out))
        save_validation(v)
        return 0
    for fw in FRAMEWORKS:
        info = fw_version(fw)
        if info:
            v["frameworks"][fw] = info
    if a.ingest_cases:
        print("ingested", ingest_cases(a.ingest_cases, v), "case results")
        save_validation(v)
        return 0
    for vid, (target, route, host) in TARGETS.items():
        if a.only and vid not in a.only:
            continue
        for fw in a.frameworks:
            try:
                e = run_one(fw, vid, target, route, host, v, force=a.force, stale=a.stale)
            except Exception as ex:                       # a build or reference failure: record it
                e = {"id": f"{fw}/{vid}", "framework": fw, "target": target, "pass": False, "pass_floor": False,
                     "error": f"{type(ex).__name__}: {ex}"[-1500:], "date": datetime.date.today().isoformat()}
                v["entries"][e["id"]] = e
                save_validation(v, [e["id"]])
            if e is None:
                print(f"skip {fw:10s} {vid} (no venv)")
                continue
            print(f"{'ok  ' if e.get('pass') else ('ok~ ' if e.get('pass_floor') else 'FAIL')} {fw:10s} {vid}: "
                  + (e.get("error", "")[-300:] if e.get("error") else
                     f"max|d| {e['max_abs_diff']:.2e} (pos>=1 {e['max_abs_diff_pos1']:.2e}, fp64 "
                     f"{e.get('fp64_max_abs_diff', float('nan')):.2e} vs hf32 floor {e.get('hf32_fp64_floor', float('nan')):.2e})"
                     f" greedy {e['greedy_identical']} {e['seconds']}s"), flush=True)
    save_validation(v, [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
