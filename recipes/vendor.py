"""Copy the construction code the recipes need into the pip-installable tef_recipes package.

    python -m recipes.vendor          # rewrite the vendored files
    python -m recipes.vendor --check  # exit 1 if they drifted from the repo (tests/test_recipes.py runs this)

The universal host (elephantine/universal.py, with its branch modules elephantine/branches_attn.py and
branches_stream.py), its gated-delta recurrence (attn/gated_delta_layer.py) and the exact-host constructions
(elephantine/emulate.py) are copied verbatim except for their repo-internal imports, so a recipe's build stage
needs `pip install ./recipes/tef_recipes` and not this repository.

emulate.py's few `from graph.* import` lines become imports from tef_recipes/graph_parts.py, written here from the
source of the graph definitions the construction reads (GRAPH_PARTS). Any other graph name -- the relation gate
(edge_plan, classify, ...), which a recipe does not run: its pair was checked when the recipe was generated --
raises an ImportError that says so. --check also fails when a vendored file imports a repository package.
"""
from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "recipes", "tef_recipes", "tef_recipes")
HEADER = "# Vendored from the_emulation_frontier/{src} by `python -m recipes.vendor`. Do not edit: edit {src}.\n"

# (source, destination, [(old, new) import rewrites])
FILES = [
    ("elephantine/universal.py", "universal.py",
     [("from attn.gated_delta_layer import gd_recurrence", "from tef_recipes.gated_delta import gd_recurrence"),
      ("from elephantine import branches_stream as _bs", "from tef_recipes import branches_stream as _bs"),
      ("from elephantine.branches_attn import *", "from tef_recipes.branches_attn import *"),
      ("from elephantine.branches_stream import (", "from tef_recipes.branches_stream import (")]),
    # the release's branch modules (eleph-gaps-attn / eleph-gaps-stream), imported by universal.py
    ("elephantine/branches_attn.py", "branches_attn.py",
     [("from elephantine.universal import (", "from tef_recipes.universal import (")]),
    ("elephantine/branches_stream.py", "branches_stream.py",
     [("    from elephantine import universal\n", "    from tef_recipes import universal\n")]),
    ("attn/gated_delta_layer.py", "gated_delta.py", []),
    ("elephantine/emulate.py", "emulate.py",
     [("if str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n", ""),
      (re.compile(r"^(\s*)from graph\.\w+ import "), r"\1from tef_recipes.graph_parts import ")]),
]

# the graph definitions the vendored construction code imports: (source file, [top-level names])
GRAPH_PARTS = [("graph/model_graph.py", ["ROPE_PAIRS_FREE_TYPES", "kv_hosts_needed"])]
GRAPH_PARTS_DST = "graph_parts.py"
REPO_PACKAGES = re.compile(r"^\s*(?:from|import)\s+(?:graph|hf|attn|elephantine|recipes|block|moe|norm|act|approx|"
                           r"verify|convert|serving|proofs|harness|tests)\b.*$", re.M)


def render(src, rewrites):
    text = open(os.path.join(ROOT, src)).read()
    for old, new in rewrites:
        if isinstance(old, re.Pattern):
            text, n = re.subn(old.pattern, new, text, flags=re.M)
        else:
            n = text.count(old)
            text = text.replace(old, new)
        if not n:
            raise SystemExit(f"{src}: expected import line not found: {getattr(old, 'pattern', old)}")
    return HEADER.format(src=src) + text


def render_graph_parts():
    """tef_recipes/graph_parts.py: the GRAPH_PARTS definitions' source, verbatim, and a module __getattr__ that
    turns any other graph name into an ImportError naming it."""
    out = ['"""Definitions from the_emulation_frontier\'s graph package that the vendored construction code reads.\n\n'
           'Any other name is part of the emulation relation, which tef_recipes does not ship (a recipe\'s pair was '
           'checked\nagainst it when the recipe was generated): importing one raises ImportError.\n"""\n']
    for src, names in GRAPH_PARTS:
        text = open(os.path.join(ROOT, src)).read()
        found = {}
        for node in ast.parse(text).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                found[node.name] = ast.get_source_segment(text, node)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in names:
                        found[t.id] = ast.get_source_segment(text, node)
        missing = [n for n in names if n not in found]
        if missing:
            raise SystemExit(f"{src}: definitions not found: {', '.join(missing)}")
        for n in names:
            out.append(f"\n# {src}: {n}\n{found[n]}\n")
    out.append('''

def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    raise ImportError(f"{name} belongs to the emulation relation (the_emulation_frontier's graph package), which "
                      "tef_recipes does not ship: run the relation gate (python -m elephantine.emulate) from a "
                      "checkout of the repository")
''')
    srcs = ", ".join(src for src, _ in GRAPH_PARTS)
    return HEADER.format(src=srcs) + "".join(out)


def repo_imports(text):
    """Lines of a vendored file that import a package of this repository (a standalone install has none)."""
    return REPO_PACKAGES.findall(text)


def main(argv=None):
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    bad = []
    files = [(dst, render(src, rw)) for src, dst, rw in FILES] + [(GRAPH_PARTS_DST, render_graph_parts())]
    for dst, want in files:
        path = os.path.join(PKG, dst)
        have = open(path).read() if os.path.isfile(path) else None
        if have != want:
            if check:
                bad.append(dst)
            else:
                with open(path, "w") as f:
                    f.write(want)
                print("wrote", os.path.relpath(path, ROOT))
    if bad:
        print("vendored files out of date:", ", ".join(bad), "(python -m recipes.vendor)")
        return 1
    leaks = [(f, line.strip()) for f in sorted(os.listdir(PKG)) if f.endswith(".py")
             for line in repo_imports(open(os.path.join(PKG, f)).read())]
    if leaks:
        print("vendored files import this repository (a pip install has no such package):")
        for f, line in leaks:
            print(f"  {f}: {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
