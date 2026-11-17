"""SSM / linear-recurrence families form a one-way subsumption order, NOT mutual emulation.

Verifies (attn/ssm_cross.py): the general gated-delta rule reproduces every family exactly
by a knob-restriction; distinct families never mutually emulate; gated-DeltaNet is the top.

Run:  pytest tests/test_ssm_cross.py -q
"""
import pytest

torch = pytest.importorskip("torch")

from attn.ssm_cross import FAMILIES, gated_delta_emulate, general, knobs, subsumes


def _outputs(seed=0):
    gen = torch.Generator().manual_seed(seed)
    T, dk, dv = 12, 8, 8
    r = lambda *s: torch.randn(*s, dtype=torch.float64, generator=gen)
    q, k, v = r(T, dk), r(T, dk), r(T, dv)
    kgen = torch.Generator().manual_seed(1)
    return {f: general(q, k, v, *knobs(f, T, dk, torch.Generator().manual_seed(1))) for f in FAMILIES}


def test_general_rule_subsumes_every_family_and_gated_delta_is_top():
    # gated-DeltaNet subsumes all; linear is the bottom
    for f in FAMILIES:
        assert subsumes("gated_deltanet", f)
        assert subsumes(f, "linear")
    # the two mid branches are genuinely incomparable (channel-decay vs delta)
    assert not subsumes("gla", "deltanet") and not subsumes("deltanet", "gla")


def test_no_mutual_emulation_between_distinct_families():
    fams = list(FAMILIES)
    mutual = [(x, y) for x in fams for y in fams if x < y and subsumes(x, y) and subsumes(y, x)]
    assert mutual == [], f"expected a strict one-way order, found mutual pairs: {mutual}"


def test_gated_delta_host_carries_every_family_with_state_inflation():
    # the recurrence-level elephantine: ONE gated-delta layer at a WIDER state reproduces
    # every SSM family token-for-token -> the primitive that closes the last 8% of AR.
    for fam in FAMILIES:
        assert gated_delta_emulate(fam, state_pad=8) < 1e-12, fam


def test_cross_family_gap_is_real():
    # a family without the delta term cannot reproduce a delta-rule output, and a scalar
    # decay cannot reproduce a per-channel gate -> a large, irreducible gap.
    o = _outputs()
    assert (o["gated_deltanet"] - o["gla"]).abs().max().item() > 1.0     # missing the delta correction
    assert (o["deltanet"] - o["gla"]).abs().max().item() > 1.0           # incomparable branches
    # and the subsumed direction is exact: linear == gated-delta run with linear's knobs
    T, dk = 12, 8
    gen = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, dtype=torch.float64, generator=gen)
    q, k, v = r(T, dk), r(T, dk), r(T, 8)
    S = torch.zeros(8, dk, dtype=torch.float64)
    ref = torch.zeros(T, 8, dtype=torch.float64)
    for t in range(T):                                                  # native linear attention
        S = S + torch.outer(v[t], k[t]); ref[t] = S @ q[t]
    lin = general(q, k, v, torch.ones(T, dk, dtype=torch.float64),
                  torch.zeros(T, dtype=torch.float64), torch.ones(T, dtype=torch.float64))
    assert (ref - lin).abs().max().item() < 1e-12
