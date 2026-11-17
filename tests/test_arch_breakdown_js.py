"""The browser port of graph/arch_breakdown.py (graph/templates/arch_breakdown.js) against the Python, on every snapshot
config, both ways the page uses it:

  lookup  the JS plans the raw config itself (graph/templates/arch_metrics.js plan + plan_ordered) and breaks it down
  card    the JS breaks down the plan the page ships (arch_breakdown.page_plan of the Python plan)

Every field must be equal: integers, booleans, strings and nulls exactly, floats to 1e-12 relative
(tests/test_arch_metrics_js._compare). A config the Python cannot read must fail in the JS too.

Run:  pytest tests/test_arch_breakdown_js.py -n 4
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile

import pytest

from tests.test_arch_metrics_js import _compare, _snapshot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(HERE, "tests", "arch_breakdown_js_runner.js")
NODE = shutil.which("node")
CHUNKS = 4
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def snapshot():
    return _snapshot()


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    from graph import arch_breakdown as BD
    p = tmp_path_factory.mktemp("bd") / "tables.json"
    p.write_text(json.dumps(BD.js_tables()))
    return str(p)


def _run(tables, cases):
    with tempfile.TemporaryDirectory() as tmp:
        cp, op = os.path.join(tmp, "cases.json"), os.path.join(tmp, "out.json")
        with open(cp, "w") as f:
            json.dump(cases, f)
        r = subprocess.run([NODE, RUNNER, tables, cp, op], capture_output=True, text=True, timeout=3000)
        assert r.returncode == 0, r.stderr[-3000:]
        with open(op) as f:
            return json.load(f)


def _python(part):
    from graph import arch_breakdown as BD
    logging.getLogger("transformers").setLevel(logging.ERROR)
    want, plans = {}, {}
    for name, raw, _ in part:
        try:
            p = BD.plan_ordered(raw)
            want[name] = ("ok", json.loads(json.dumps(BD.breakdown(p, raw))))
            plans[name] = BD.page_plan(p)
        except Exception as e:
            want[name] = ("error", f"{type(e).__name__}: {e}")
    return want, plans


def _diff(want, got):
    bad = []
    for name, (kind, w) in want.items():
        g = got[name]
        if kind == "error":
            if "error" not in g:
                bad.append((name, "Python raised, JS did not", w))
            continue
        if "error" in g:
            bad.append((name, "JS raised", g["error"], g.get("stack")))
            continue
        out = []
        _compare(w, g["ok"], "", out)
        if out:
            bad.append((name, out[:5]))
    return bad


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_breakdown_js_equals_python(snapshot, tables, chunk):
    part = snapshot[chunk::CHUNKS]
    want, plans = _python(part)
    got = _run(tables, [{"name": n, "raw": raw} for n, raw, _ in part])
    bad = _diff(want, got)
    assert not bad, f"lookup path: {len(bad)} of {len(part)} configs differ: {bad[:6]}"
    got = _run(tables, [{"name": n, "plan": plans[n]} for n in plans])
    bad = _diff({n: want[n] for n in plans}, got)
    assert not bad, f"card path: {len(bad)} of {len(plans)} configs differ: {bad[:6]}"


def test_head_stored_gguf(snapshot, tables):
    """A tied checkpoint that stores its head (arch_metrics hub reason 'tied head stored') gets an output.weight in
    the GGUF: the same in both."""
    from graph import arch_breakdown as BD
    cases, want = [], {}
    for name, raw, _ in snapshot:
        p = BD.plan_ordered(raw)
        if p["tied"] and len(cases) < 40:
            p["head_stored"] = True
            want[name] = ("ok", json.loads(json.dumps(BD.breakdown(p, raw))))
            cases.append({"name": name, "plan": BD.page_plan(p)})
    got = _run(tables, cases)
    assert not _diff(want, got)
