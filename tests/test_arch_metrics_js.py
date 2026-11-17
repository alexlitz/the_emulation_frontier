"""The browser port of graph/arch_metrics.py (graph/templates/arch_metrics.js) against the Python, on every snapshot
config.

Every zoo model (graph.frontier.load(), configs from graph/hub_10k.json, Hub totals from graph/model_meta.json) goes
through Python `metrics(raw, hub_total)` and through the JS under node, with no Python input to the JS (it resolves
the stock transformers config classes itself, from graph/data/arch_metrics_js_tables.json). The two records must be
equal field by field: integers, booleans, strings and nulls exactly (the notes included), floats to 1e-12 relative.
A config the Python rejects must be rejected by the JS too. test_js_port_on_class_defaults does the same for every LM
model type of the installed transformers, as its config class's defaults (the lookup of an arbitrary Hub id).

The tables are generated (`python -m graph.arch_metrics_js`); test_tables_current fails when graph/model_graph.py or
graph/arch_metrics.py changed a table the port reads, or transformers was upgraded, without regenerating them.

Run:  pytest tests/test_arch_metrics_js.py -n 4
"""
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(HERE, "graph", "templates", "arch_metrics.js")
TABLES = os.path.join(HERE, "graph", "data", "arch_metrics_js_tables.json")
RUNNER = os.path.join(HERE, "tests", "arch_metrics_js_runner.js")
CHUNKS = 4
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _snapshot():
    """(name, raw config, hub total) of every zoo model, in zoo order."""
    from graph.frontier import load
    from graph.model_graph import iter_json_configs
    configs = dict(iter_json_configs(os.path.join(HERE, "graph", "hub_10k.json")))
    with open(os.path.join(HERE, "graph", "model_meta.json")) as f:
        meta = json.load(f)
    return [(m["name"], configs[m["name"]], (meta.get(m["name"]) or {}).get("hf_total")) for m in load()]


def _compare(a, b, path, out):
    """Python value a against JS value b; mismatches appended to out as (path, a, b)."""
    if isinstance(a, bool) or isinstance(b, bool):
        if not (isinstance(a, bool) and isinstance(b, bool) and a == b):
            out.append((path, a, b))
    elif a is None or b is None:
        if not (a is None and b is None):
            out.append((path, a, b))
    elif isinstance(a, str) or isinstance(b, str):
        if a != b:
            out.append((path, a, b))
    elif isinstance(a, int):
        if not (isinstance(b, (int, float)) and float(b).is_integer() and int(b) == a):
            out.append((path, a, b))
    elif isinstance(a, float):
        if not (isinstance(b, (int, float)) and (a == b or (math.isfinite(a) and abs(a - b) <= 1e-12 * max(abs(a), abs(b))))):
            out.append((path, a, b))
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            out.append((path, sorted(a) if isinstance(a, dict) else a, sorted(b) if isinstance(b, dict) else b))
        else:
            for k in a:
                _compare(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, list) or len(a) != len(b):
            out.append((path, a, b))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _compare(x, y, f"{path}[{i}]", out)
    else:
        out.append((path, f"<{type(a).__name__}>", b))


def _run_js(cases):
    with tempfile.TemporaryDirectory() as tmp:
        cp, op = os.path.join(tmp, "cases.json"), os.path.join(tmp, "out.json")
        with open(cp, "w") as f:
            json.dump(cases, f)                            # Python's JSON (bare Infinity / NaN, as config.json has)
        r = subprocess.run([NODE, RUNNER, TABLES, cp, op], capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, r.stderr[-3000:]
        with open(op) as f:
            return json.load(f)


@pytest.fixture(scope="module")
def snapshot():
    return _snapshot()


def test_snapshot_size(snapshot):
    assert len(snapshot) == 3906                       # (round 3: 3901 -> 3906 placed models)


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_js_port_equals_python(snapshot, chunk):
    from graph import arch_metrics as AM
    logging.getLogger("transformers").setLevel(logging.ERROR)
    part = snapshot[chunk::CHUNKS]
    want = {}
    for name, raw, hub in part:
        try:
            want[name] = ("ok", AM.metrics(raw, hub))
        except Exception as e:                             # the Python rejects the config: the JS must too
            want[name] = ("error", f"{type(e).__name__}: {e}")
    got = _run_js([{"name": n, "raw": raw, "hub": hub} for n, raw, hub in part])
    bad = []
    for name, _, _ in part:
        kind, w = want[name]
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
    assert not bad, f"{len(bad)} of {len(part)} configs differ: {bad[:10]}"


def test_js_port_on_class_defaults():
    """Beyond the snapshot: every LM model type of the installed transformers (causal-LM / image-text-to-text /
    vision-to-text heads), as its config class's defaults with its head class as the architecture -- the configs a
    lookup of an arbitrary Hub id resolves through the replayed config classes."""
    from transformers import CONFIG_MAPPING
    from transformers.models.auto import modeling_auto as ma

    from graph import arch_metrics as AM
    logging.getLogger("transformers").setLevel(logging.ERROR)
    heads = {**ma.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES, **ma.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
             **ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES}
    cases, want = [], {}
    for mt, arch in sorted(heads.items()):
        try:
            raw = json.loads(json.dumps(CONFIG_MAPPING[mt]().to_dict(), default=str))
        except Exception:                                  # a class that needs arguments: no default config
            continue
        raw["architectures"] = [arch[0] if isinstance(arch, (list, tuple)) else arch]
        name = "defaults:" + mt
        cases.append({"name": name, "raw": raw, "hub": None})
        try:
            want[name] = ("ok", json.loads(json.dumps(AM.metrics(raw), default=str)))
        except Exception as e:
            want[name] = ("error", f"{type(e).__name__}: {e}")
    assert len(cases) >= 150
    got = _run_js(cases)
    bad = []
    for c in cases:
        kind, w = want[c["name"]]
        g = got[c["name"]]
        if kind == "error" or "error" in g:
            if (kind == "error") != ("error" in g):
                bad.append((c["name"], w if kind == "error" else "Python ok", g.get("error")))
            continue
        out = []
        _compare(w, g["ok"], "", out)
        if out:
            bad.append((c["name"], out[:5]))
    assert not bad, f"{len(bad)} of {len(cases)} differ: {bad[:10]}"


def test_tables_current():
    """The generated tables hold the current model_graph / arch_metrics tables and the installed transformers."""
    import transformers

    from graph import arch_metrics as AM
    from graph import arch_metrics_js as G
    from graph import model_graph as mg
    with open(TABLES) as f:
        t = json.load(f)
    assert t["tf"]["version"] == transformers.__version__, "transformers changed: python -m graph.arch_metrics_js"
    for n in G.MG_NAMES:
        assert t["mg"][n] == json.loads(json.dumps(G._plain(getattr(mg, n)))), f"model_graph.{n} changed: regenerate"
    for n in G.AM_NAMES:
        assert t["am"][n] == json.loads(json.dumps(G._plain(getattr(AM, n)))), f"arch_metrics.{n} changed: regenerate"


def test_parse_config_python_json():
    """Hub config.json files are written by Python's json: bare Infinity / NaN must parse (Mamba time_step_limit)."""
    js = (f"const AM = require({json.dumps(JS)});"
          "const c = AM.parseConfig('{\"a\": [0.0, Infinity], \"b\": -Infinity, \"c\": NaN, \"s\": \"Infinity\", \"n\": 3}');"
          "console.log(JSON.stringify([c.a[1] === Infinity, c.b === -Infinity, Number.isNaN(c.c), c.s, c.n]));")
    r = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=60)
    assert json.loads(r.stdout) == [True, True, True, "Infinity", 3], r.stderr


def test_python_number_semantics():
    """round() to even, round(x, 4) on the exact binary value, exactly rounded big int / int, // and % flooring."""
    js = (f"const I = require({json.dumps(JS)})._internal;"
          "console.log(JSON.stringify([I.pyRound(2.5), I.pyRound(3.5), I.pyRound(-2.5), I.pyRoundN(1.03125, 4), "
          "I.pyRoundN(0.00015, 4), I.pyRoundN(2.675, 2), I.divExact(10n ** 30n + 1n, 3n), I.floordiv(-7, 2), "
          "I.pymod(-7, 2), I.floordiv(2 ** 60, 3)]));")
    r = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=60)
    got = json.loads(r.stdout)
    assert got[:6] == [round(2.5), round(3.5), round(-2.5), round(1.03125, 4), round(0.00015, 4), round(2.675, 2)]
    assert got[6] == (10 ** 30 + 1) / 3
    assert got[7:9] == [-7 // 2, -7 % 2]
    assert float(got[9]) == float(2 ** 60 // 3)
