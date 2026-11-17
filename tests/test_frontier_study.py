"""Frontier study itemiser (graph/frontier_study.py, from branch frontier-study): the item refinement of emul_matrix's
refusals is exact, and moe_reasons agrees with moe_why (the release's rules); the branch's constructions
(block/frontier_constructions.py, post-publication tooling, no relation change) are exact on the reference decoder
(fp64) with live negative controls."""
import numpy as np
import pytest

from block import frontier_constructions as fc
from graph import model_graph as mg
from graph import frontier_study as fs
from graph.currency_curve import points


@pytest.fixture(scope="module")
def sub():
    P = points()
    return P[:120] + P[-60:]


def test_items_refine_emul_matrix(sub):
    P, ok, cap, items = fs.build(sub)                  # build() asserts ok == (no item) itself
    cnt = sum(m.astype(np.int16) for m in items.values())
    assert np.array_equal(ok, cnt == 0)
    assert all(fs.kind_of(k) for k in items)


def test_moe_reasons_first_is_moe_why():
    P = points()
    keys = [(p["dims"].get("moe_sig"), p["dims"]["hidden"], p["dims"]["ffn_total"]) for p in P
            if p["dims"].get("moe_sig")]
    keys = keys[::7][:60]
    for a in keys:
        for b in keys:
            w, rs = mg.moe_why(a, b), fs.moe_reasons(a, b)
            assert bool(w) == bool(rs) and (not w or w == rs[0])


@pytest.mark.parametrize("name", sorted(fc.REF_CHECKS))
def test_construction_exact(name):
    r = fc.REF_CHECKS[name]()
    assert r.rel < 1e-12 and r.top1 == 1.0 and r.identical == r.n


@pytest.mark.parametrize("name", sorted(fc.REF_NEGATIVE))
def test_negative_control_differs(name):
    r = fc.REF_NEGATIVE[name]()
    assert r.rel > 1e-2 and r.identical < r.n
