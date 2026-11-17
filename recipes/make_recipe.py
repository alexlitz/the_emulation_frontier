"""Write a downloadable recipe: one self-contained script that runs a zoo model in a framework.

    python -m recipes.make_recipe TARGET --framework {pytorch,jax,tinygrad,tensorflow,dynet} [--host H]
                                         [--route native|exact-host|universal] [--out FILE]
    python -m recipes.make_recipe --index            # graph/data/recipes.json (+ the checked-in tested scripts)

The route (RECIPES_SCHEMA.md): `native` when the framework's Llama-style reference runs the target's own
config, else the cheapest `exact-host` (an exact edge into a Llama-style host that elephantine/emulate.py v1
builds), else `universal` (the universal host's from_* loader, exported as tef-ucore). --host forces an
exact-host route through that host.

The script has two stages:
    python recipe.py build [--out DIR]     # download (+ build the host); needs torch + transformers +
                                           # tef_recipes for exact-host / universal, nothing for native
    python recipe.py run [--dir DIR] [--prompt TEXT | --ids 1,2,3] [--new-tokens 32]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FRAMEWORKS = ("pytorch", "jax", "tinygrad", "tensorflow", "dynet")
RUN_REQS = {
    "pytorch": "torch numpy  (optional: tokenizers)",
    "jax": "jax[cpu] numpy  (optional: tokenizers)",
    "tinygrad": "tinygrad numpy  (optional: tokenizers; CPU needs clang, or CC=<zig cc wrapper>)",
    "tensorflow": "tensorflow-cpu numpy  (optional: tokenizers)",
    "dynet": "dyNET==2.1.2 numpy<1.25  on CPython 3.8  (optional: tokenizers<0.21)",
}
BUILD_REQS = {
    "native": "none (downloads over HTTPS with the standard library)",
    "exact-host": "torch transformers safetensors huggingface_hub tef-recipes  (pip install ./recipes/tef_recipes)",
    "universal": "torch transformers safetensors huggingface_hub tef-recipes  (pip install ./recipes/tef_recipes)",
}
GITHUB = "https://github.com/alexlitz/the_emulation_frontier"
SCRIPTS = os.path.join(ROOT, "recipes", "scripts")
OUT_JSON = os.path.join(ROOT, "graph", "data", "recipes.json")
VALIDATION = os.path.join(ROOT, "recipes", "validation.json")


def slug(repo):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", repo.strip("/").replace("/", "__"))


# --- configs ------------------------------------------------------------------------------------------

def raw_config(repo):
    """config.json of a Hub repo or a local directory (the HF cache first)."""
    from recipes.common.runtime import hub_file
    return json.load(open(hub_file(repo, "config.json", os.path.expanduser("~/.cache/tef-recipes"))))


def native_ok(cfg):
    """None if the Llama-style reference runs this config as is, else the reason."""
    from recipes.common.llama_spec import Unsupported, llama_spec
    try:
        llama_spec(cfg)
    except Unsupported as e:
        return str(e)
    except Exception as e:                              # a malformed config
        return f"config not understood: {type(e).__name__}: {e}"
    return None


def loader_of(cfg):
    """(universal-host loader, None) or (None, the reason)."""
    from elephantine.loader_coverage import loader_for
    from graph.model_graph import _lm_config
    try:
        return loader_for(_lm_config(cfg))
    except Exception as e:
        return None, f"config not understood: {type(e).__name__}: {e}"


def exact_host_config(traw, hraw):
    """The config.json emulate.py v1 writes for target-in-host (the host's own, with the changes it may make)."""
    from elephantine import emulate as E
    return E.host_config_dict(hraw, E.plan_construction(traw, hraw))


def exact_support(traw, hraw):
    """None if emulate.py v1 builds target-in-host exactly (config level) AND the Llama-style reference runs the
    checkpoint it writes, else the reasons."""
    from elephantine import emulate as E
    bad = E.support(traw, hraw)
    if bad:
        return bad
    try:
        r = native_ok(exact_host_config(traw, hraw))
    except Exception as e:
        return [f"{type(e).__name__}: {e}"]
    return None if r is None else [f"the host checkpoint it writes is not one the Llama-style reference runs: {r}"]


def params_of(raw):
    from elephantine import emulate as E
    try:
        c = E.resolved(raw)
        return E._params(c, E.FAMILIES[c.model_type])
    except Exception:
        return None


def plan(target, framework, host=None, route=None, cfg=None):
    """The route a recipe takes: {"route", "host", "loader", "cost", "why"}."""
    cfg = cfg if cfg is not None else raw_config(target)
    why = []
    if host is not None or route == "exact-host":
        if host is None:
            raise SystemExit("--route exact-host needs --host")
        hraw = raw_config(host)
        r = native_ok(hraw)
        if r:
            raise SystemExit(f"host {host}: the Llama-style reference cannot run it ({r})")
        bad = exact_support(cfg, hraw)
        if bad:
            raise SystemExit(f"{target} in {host}: emulate.py v1 does not build this pair: " + "; ".join(bad))
        tp, hp = params_of(cfg), params_of(hraw)
        return {"route": "exact-host", "host": host, "loader": None,
                "cost": {"params_x": round(hp / tp, 4) if tp and hp else None}, "why": None}
    if route in (None, "native"):
        r = native_ok(cfg)
        if r is None:
            return {"route": "native", "host": None, "loader": None, "cost": {"params_x": 1.0}, "why": None}
        if route == "native":
            raise SystemExit(f"{target}: no native route ({r})")
        why.append(f"native: {r}")
    if route in (None, "universal"):
        ld, r = loader_of(cfg)
        if ld is not None:
            return {"route": "universal", "host": "universal", "loader": ld, "cost": {"params_x": None}, "why": None}
        why.append(f"universal: {r}")
    raise SystemExit(f"{target}: no route for {framework} ({'; '.join(why)})")


# --- script emission ------------------------------------------------------------------------------------

def _src(rel):
    lines = open(os.path.join(ROOT, rel)).read().splitlines()
    return "\n".join(l for l in lines if "# tef:inline" not in l and not l.startswith("from __future__"))


def _indent(text, n=4):
    pad = " " * n
    return "\n".join(pad + l if l.strip() else l for l in text.splitlines())


BUILD_FN = '''
def build(args):
    """Download the target (and build its host) into args.out."""
    route = DEFAULTS["route"]
    target = args.target or DEFAULTS["target"]
    out = args.out
    os.makedirs(out, exist_ok=True)
    if route == "native":
        cfg, shards = hub_model_files(target, args.cache)
        for p in [cfg] + shards + [hub_file(target, f, args.cache, optional=True)
                                   for f in ("model.safetensors.index.json", "tokenizer.json")]:
            if p:
                dst = os.path.join(out, os.path.basename(p))
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.realpath(p), dst)
    else:
        try:
            from tef_recipes import build as tb
        except ImportError:
            raise SystemExit("the build stage needs tef_recipes: pip install "
                             "'git+%s#subdirectory=recipes/tef_recipes' (with torch, transformers)" % GITHUB)
        if route == "exact-host":
            tb.build_exact(target, args.host or DEFAULTS["host"], out)
        else:
            tb.build_universal(target, DEFAULTS["loader"], out)
    print("built", out, "(%s route); now: python %s run --dir %s" % (route, sys.argv[0], out))
'''


def emit(target, framework, p, out_dir_default=None):
    """The recipe script text."""
    kind = "llama" if p["route"] in ("native", "exact-host") else "universal"
    parts = ["recipes/common/runtime.py"]
    run_parts = []
    if kind == "llama":
        run_parts.append("recipes/common/llama_spec.py")
    elif framework in ("jax", "tensorflow", "tinygrad"):
        run_parts.append("recipes/common/ucore_forward.py")
    run_parts.append(f"recipes/frameworks/{framework}/{kind}.py")
    defaults = {"target": target, "framework": framework, "route": p["route"], "host": p.get("host"),
                "loader": p.get("loader"),
                "out": out_dir_default or f"./{slug(target)}-{framework}"}
    route_line = {"native": "native: the target's own Llama-style weights",
                  "exact-host": f"exact-host: {target} rebuilt exactly as a {p['host']}-shaped Llama-style checkpoint",
                  "universal": f"universal host via elephantine.universal.{p.get('loader')}"}[p["route"]]
    head = f'''#!/usr/bin/env python3
"""Emulation-frontier recipe: run {target} in {framework}.

Route: {route_line}.
Written by `python -m recipes.make_recipe {target} --framework {framework}` ({GITHUB}).

    python {{this}} build [--out DIR]      # stage 1: download{'' if p['route'] == 'native' else ' + build the host weights'}
    python {{this}} run [--dir DIR] [--prompt TEXT | --ids 1,2,3] [--new-tokens 32]
"""
# requirements-build: {BUILD_REQS[p['route']]}
# requirements-run: {RUN_REQS[framework]}
'''.replace("{this}", f"{slug(target)}-{framework}.py")
    body = [head, "import os", "import sys", "", f"GITHUB = {GITHUB!r}",
            "DEFAULTS = " + repr(defaults), ""]
    body += ["# ---- " + parts[0] + " " + "-" * 40, _src(parts[0]), BUILD_FN]
    body += ["_RUN = __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'run'",
             "if _RUN:  # the framework code is only imported for the run stage (the build stage may run elsewhere)"]
    for rp in run_parts:
        body += [f"    # ---- {rp} " + "-" * 40, _indent(_src(rp))]
    body += ["", "", "if __name__ == '__main__':",
             f"    sys.exit(run_cli(lambda d: load_model(d), {framework!r}, build=build, defaults=DEFAULTS))", ""]
    return "\n".join(body)


CLI_FRAMEWORKS = ("vllm", "llama.cpp", "mlx-lm")


def cli_recipe(target, framework):
    """A shell recipe for a serving framework (not run-tested here): its native registry, or the backdoor
    host graph/data/frameworks.json lists, built by elephantine/emulate.py when v1 builds that edge."""
    from recipes.index import _cli, frameworks_json
    from graph.model_graph import iter_json_configs
    from recipes.index import CONFIGS
    fwj = frameworks_json()
    if fwj is None or target not in fwj["repos"]:
        raise SystemExit(f"{target}: not in graph/data/frameworks.json (python -m graph.frameworks on zoo-metrics)")
    cfgs = dict(iter_json_configs(CONFIGS))
    c = _cli(framework, target, None, fwj, cfgs, {})
    v = fwj["frameworks"][framework]
    head = (f"#!/bin/sh\n# Emulation-frontier recipe: {target} on {framework} {v['version']} "
            f"(registry read {v['fetched']}; not run-tested here).\n# Route: {c['route']}"
            + (f" via {c['host']} ({c['edge']} edge)" if c.get("host") else "") + "\n")
    if c.get("cmd"):
        return head + "set -e\n" + c["cmd"].replace(" && ", "\n") + "\n"
    return head + "# No recipe: " + (c.get("why") or c.get("why_not_buildable") or "?") + "\nexit 1\n"


def write_recipe(target, framework, host=None, route=None, out=None, cfg=None):
    p = plan(target, framework, host=host, route=route, cfg=cfg)
    text = emit(target, framework, p)
    out = out or f"{slug(target)}-{framework}.py"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    os.chmod(out, 0o755)
    return out, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", nargs="?")
    ap.add_argument("--framework", choices=FRAMEWORKS + CLI_FRAMEWORKS)
    ap.add_argument("--host")
    ap.add_argument("--route", choices=["native", "exact-host", "universal"])
    ap.add_argument("--out")
    ap.add_argument("--index", action="store_true", help="write graph/data/recipes.json")
    a = ap.parse_args(argv)
    if a.index:
        from recipes.index import write_index
        write_index()
        return 0
    if not a.target or not a.framework:
        ap.error("TARGET and --framework are required")
    if a.host and a.route in ("native", "universal"):
        ap.error(f"--host only applies to --route exact-host (not {a.route})")
    if a.framework in CLI_FRAMEWORKS:
        text = cli_recipe(a.target, a.framework)
        out = a.out or f"{slug(a.target)}-{a.framework}.sh"
        with open(out, "w") as f:
            f.write(text)
        os.chmod(out, 0o755)
        print(f"wrote {out}")
        return 0
    out, p = write_recipe(a.target, a.framework, a.host, a.route, a.out)
    print(f"wrote {out}: {p['route']}" + (f" via {p['host']}" if p.get("host") not in (None, "universal") else "")
          + (f" ({p['loader']})" if p.get("loader") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
