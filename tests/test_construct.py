"""construct(A, B) finds a construction if ANY exists in-framework (mixing currencies /
per-layer trades), and reports a genuine wall only for a real obstruction.

Run:  python -m pytest tests/test_construct.py -q
"""
from graph.construct import construct
from tests.test_graph_rules import mk


def test_identical_is_trivially_constructible():
    r = construct(mk(), mk())
    assert r["constructible"] and not r["routes"] and not r["walls"]


def test_xielu_target_folds_into_silu_host_via_width():
    # host SiLU, target xIELU: no single-op fold, but the width network route makes it
    # CONSTRUCTIBLE (approx) -- exactly the "find it if possible" behaviour.
    r = construct(mk(act="xielu"), mk(act="SiLU"))
    assert r["constructible"] and r["approx"]
    act = [o for s, a, b, o in r["routes"] if s == "act"][0]
    assert any("width" in cur for cur, exact, note in act)


def test_diffusion_vs_ar_is_the_one_genuine_wall():
    r = construct(mk(gen="diffusion"), mk(gen="ar"))
    assert not r["constructible"]
    assert any(s == "gen" for s, *_ in r["walls"])


def test_local_target_reaches_full_host_but_full_target_needs_depth():
    # window is one-way in the base order; the reverse (full target, local host) is payable
    # in DEPTH, so construct must still find it (exact) rather than call it a wall.
    r = construct(mk(window="full"), mk(window="local"))
    win = [o for s, a, b, o in r["routes"] if s == "window"]
    assert win and any("depth" in cur for cur, exact, note in win[0])
