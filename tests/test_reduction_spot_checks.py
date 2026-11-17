"""Every reduction, spot-checked on 1-2 layers of REAL weights.

The per-axis proofs in attn/ act/ norm/ block/ moe/ convert/ are mostly isolated
tensor identities. This suite runs each reduction as a genuine module forward with
real nn.Linear weights (see elephantine/spot_check_reductions.py) and requires:

  * exact / regime / limit-endpoint edges reproduce to fp64 noise (< 1e-9), and
  * the one activation FLOOR (GeLU vs SiLU) provably does NOT vanish (> 1e-3),

so an edge cannot merely be algebraically true while wired wrong inside a layer.

Run:  pytest tests/test_reduction_spot_checks.py -q
"""
import pytest

torch = pytest.importorskip("torch")

from elephantine.spot_check_reductions import CHECKS, _stock_checks


@pytest.mark.parametrize("check", CHECKS, ids=[c.__name__ for c in CHECKS])
def test_reduction_reproduces_on_real_layers(check):
    r = check()
    if r["kind"] in ("exact", "regime", "limit"):
        assert r["worst"] < r["thr"], f"{r['name']}: worst |Δ| = {r['worst']:.2e} (host {r['host']})"
    else:  # floor: must stay a genuine gap, not collapse to fp noise
        assert r["worst"] > r["thr"], f"{r['name']}: floor collapsed to {r['worst']:.2e} (should be a real gap)"
    assert r["ok"]


def test_all_lifted_reductions_present():
    # guard against a check silently dropping out of the table
    assert len(CHECKS) == 17


def test_stock_module_builds_reproduce():
    pytest.importorskip("transformers")
    rows = _stock_checks()
    assert rows, "no stock-module checks ran"
    for r in rows:
        if r["host"].startswith("skip:"):
            continue                                   # environment lacks that module -- fine
        assert r["ok"], f"{r['name']}: worst |Δ| = {r['worst']:.2e} (host {r['host']})"


def test_floor_edge_is_the_only_non_vanishing_one():
    # exactly one reduction is a genuine (non-vanishing) floor: GeLU vs SiLU
    floors = [c() for c in CHECKS if c().get("kind") == "floor"]
    assert len(floors) == 1 and floors[0]["worst"] > 1e-3
