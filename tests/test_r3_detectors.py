"""Round-3 detectors (verify/r3/detectors.py) against their fixtures on the real snapshot configs.

Each round-3 spec (verify/r3/specs/<name>.md) names a standalone detect_<name>(cfg, model_type) and the snapshot repos it
must flag (with the expected value) and must not flag (verify/r3/fixtures/*.json). The detectors do not import
graph/model_graph, so this test pins the classifier change before it is applied. Specs d_* are relation rules (Z3
transitivity) and have no config detector.
"""
import importlib.util
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "verify", "r3")
sys.path.insert(0, R3)
import r3util  # noqa: E402

spec = importlib.util.spec_from_file_location("r3_detectors", os.path.join(R3, "detectors.py"))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

SNAP = {m["id"]: m["config"] or {} for m in json.load(open(os.path.join(ROOT, "graph", "hub_10k.json")))}
FIXTURES = {}
for fn in sorted(os.listdir(os.path.join(R3, "fixtures"))):
    if fn.endswith(".json"):
        for k, v in json.load(open(os.path.join(R3, "fixtures", fn))).items():
            assert k not in FIXTURES, f"fixture {k} defined twice"
            FIXTURES[k] = v
SPECS = sorted(f[:-3] for f in os.listdir(os.path.join(R3, "specs")) if f.endswith(".md"))
CARRIERS = json.load(open(os.path.join(R3, "carriers.json")))


def _run(fx, repo):
    cfg = SNAP[repo]
    if fx.get("repo_keyed"):        # the code fact is not in the config (identical configs): keyed by repo id
        cfg = {**cfg, "_repo_id": repo}
    return getattr(D, fx["detector"])(cfg, r3util.model_type_of(cfg))


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_flags(name):
    fx = FIXTURES[name]
    assert fx["flag"] or fx.get("no_positive"), f"{name}: no positive fixture (and no no_positive reason)"
    for repo, want in fx["flag"].items():
        got = _run(fx, repo)
        assert got, f"{name}: {repo} not flagged"
        if want is not True:
            assert got == want, f"{name}: {repo} gives {got!r}, want {want!r}"


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_not_flags(name):
    fx = FIXTURES[name]
    for repo in fx["not_flag"]:
        assert not _run(fx, repo), f"{name}: {repo} must not be flagged"


def test_detectors_never_raise():
    names = [n for n in dir(D) if n.startswith("detect_")]
    for cfg in SNAP.values():
        mt = r3util.model_type_of(cfg)
        for n in names:
            getattr(D, n)(cfg, mt)


def test_every_live_finding_has_a_spec():
    live = {k for k, f in CARRIERS.items() if f["live"]}
    assert not live - set(SPECS), sorted(live - set(SPECS))


def test_every_spec_has_a_detector_and_fixtures():
    for s in SPECS:
        if s.startswith("d_"):
            continue
        assert hasattr(D, "detect_" + s), s
        assert s in FIXTURES, s


def test_detectors_do_not_import_model_graph():
    src = open(os.path.join(R3, "detectors.py")).read() + open(os.path.join(R3, "r3util.py")).read()
    assert not re.search(r"^\s*(from|import)\s+graph\b", src, re.M)
