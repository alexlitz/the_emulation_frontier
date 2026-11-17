"""The frontier is a function of the currency budget: it shrinks monotonically from the
strict-exact order through the payable HEADLINE to the capacity-only secondary and then to 1
(the elephantine).

Run:  python -m pytest tests/test_currency_curve.py -q
"""
from graph.currency_curve import curve, n_frontier, points
from graph.model_graph import REACH, emul_matrix


def test_frontier_shrinks_monotonically_with_budget():
    c = curve(points())
    seq = [c["exact"], c["payable"], c["mixgap"], c["capacity"], c["elephantine"]]
    assert seq == sorted(seq, reverse=True), seq            # monotone non-increasing
    # each budget's point is the frontier of that relation (the numbers and their history are pinned in
    # test_edge_soundness.test_frontier_definitions_headline_and_slices: exact / HEADLINE payable /
    # mixgap / capacity = 233 / 218 / 198 / 51 after the edge-case audit and the partial-RoPE host)
    P = points()
    for mode in ("exact", "payable", "mixgap", "capacity"):
        assert c[mode] == n_frontier(emul_matrix(P, mode=mode)), mode
    assert c["payable"] > c["capacity"]                     # the payable walls bind
    assert c["elephantine"] == 1


def test_dissolving_walls_restores_global_reach():
    # curve() must leave the slot orders untouched
    _ = curve()
    assert "rope" not in REACH["pos"].get("learned_abs", set())   # abs still can't reach rope
    assert "softmax" in REACH["mixer"].get("ssm", set())          # the TYPE order intact (edge unpaid)
    assert "diffusion" in REACH["gen"]["ar"] and "ar" not in REACH["gen"].get("diffusion", {"diffusion"})
