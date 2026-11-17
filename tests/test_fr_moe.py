"""Branch fr-moe: the router / MoE rules (graph/model_graph.py moe_why, _moe_copy_plans, _group_cap), their constructions
(moe/router_frontier.py; catalog entries fm_null_experts*, fm_plain_extras_sink_*) and their real-checkpoint
measurements (moe/data/router_frontier.jsonl).

  moe_null_experts       a top-k remainder k_h = m k_t + r: r always-selected zero-weight null experts, on a host with
                         a SELECTION-ONLY bias and a free hidden dim (the constant channel); a limit (payable, not exact)
  moe_plain_extras_sink  an unrenormalised softmax target with extra experts on its own-P host, the extras sunk by a
                         free hidden dim; a limit (payable, not exact)
  group_code_by_arch     Edge0 (typeless, BailingMoeV3ForCausalLM) and Ming-3.0-ASR read as the top-2-sum group code
  (refuted: moe_softmax_exp_regime -- a softmax target on a sigmoid host through the exp regime; granite-3.x MoE fails
  the standard because of its massive-activation first token: test_exp_regime_refuted)
"""
import contextlib
import json
import os

import pytest

from graph import model_graph as mg


@contextlib.contextmanager
def on(*specs):
    was = [s for s in specs if s in mg.R3_OFF]
    mg.R3_OFF.difference_update(was)
    try:
        yield
    finally:
        mg.R3_OFF.update(was)


@contextlib.contextmanager
def off(*specs):
    added = [s for s in specs if s not in mg.R3_OFF]
    mg.R3_OFF.update(added)
    try:
        yield
    finally:
        mg.R3_OFF.difference_update(added)


def _msig(E=8, k=2, ew=16, shared=0, renorm="R", renorm_cap=("R",), score="softmax", score_cap=("softmax",), ng=0,
          tg=0, gcap="", n_moe=4, n_dense=0, dense_w=0, latent=0, bias=0, **r3):
    f = dict(zip(mg.MOE_SIG_R3, mg.MOE_SIG_R3_DENSE))
    f.update(r3)
    return (E, k, ew, shared, renorm, tuple(renorm_cap), score, tuple(score_cap), ng, tg, gcap, n_moe, n_dense, dense_w,
            latent, bias) + tuple(f[x] for x in mg.MOE_SIG_R3)


def _why(t, h, ht=64, hh=64, strict=False):
    return mg.moe_why((t, ht, 128), (h, hh, 1 << 20), strict)


SIG = dict(score="sigmoid", score_cap=("sigmoid",), renorm_cap=("P", "R"))


def test_null_experts_default_off():
    """The release ships moe_null_experts off (measured on untrained checkpoints only)."""
    assert "moe_null_experts" in mg.R3_OFF
    t, h = _msig(E=128, k=6, bias=1, **SIG), _msig(E=160, k=8, bias=1, **SIG)
    assert _why(t, h, hh=96)


@pytest.fixture
def nulls_on():
    with on("moe_null_experts"):
        yield


def test_null_experts(nulls_on):
    """sarvam-30b's shape (top-6 of 128) on GLM-4.5's (top-8 of 160): m = 1 copy + 2 nulls; NemotronH-0.3B's (top-8 of
    32) on Nemotron-3-Ultra's (top-22 of 512): m = 2 + 6 nulls. Needs the free hidden dim, a selection-only bias (1, not
    gpt-oss's logit bias 2) and room for the nulls; a limit, so not in the exact slice."""
    t, h = _msig(E=128, k=6, bias=1, **SIG), _msig(E=160, k=8, bias=1, **SIG)
    assert _why(t, h, hh=96) is None
    assert "top-k" in _why(t, h)                                    # no free hidden dim
    assert "top-k" in _why(t, _msig(E=160, k=8, bias=0, **SIG), hh=96)
    assert "top-k" in _why(t, _msig(E=160, k=8, bias=2, **SIG), hh=96)
    assert "top-k" in _why(t, h, hh=96, strict=True)
    assert _why(_msig(E=32, k=8, bias=1, **SIG), _msig(E=512, k=22, bias=1, **SIG), hh=96) is None
    assert _why(_msig(E=256, k=8, bias=1, **SIG), _msig(E=256, k=10, bias=1, **SIG), hh=96)    # 256 < 256 + 2 nulls
    assert _why(_msig(E=8, k=2, bias=1, ng=4, tg=2, gcap="top2sum", **SIG),
                _msig(E=16, k=3, bias=1, ng=4, tg=2, gcap="top2sum", **SIG), hh=96)          # (not with groups)
    with off("moe_null_experts"):
        assert _why(t, h, hh=96)


def test_plain_extras_sink():
    """An unrenormalised softmax target (Qwen2.5-0.5B-MoE: 32 experts, top-8) on its own-P host with extras (Qwen2-57B:
    64): payable with a free hidden dim; still walled on an equal-hidden host, a renormalising host (the transitivity
    guard) and in the exact slice."""
    P = dict(renorm="P", renorm_cap=("P", "R"))
    t, h = _msig(E=32, k=8, **P), _msig(E=64, k=8, **P)
    assert _why(t, h, hh=96) is None
    assert "unrenormalised" in _why(t, h)
    assert "unrenormalised" in _why(t, _msig(E=64, k=8, renorm="R", renorm_cap=("P", "R")), hh=96)
    assert "unrenormalised" in _why(t, h, hh=96, strict=True)
    assert _why(t, _msig(E=32, k=8, **P)) is None                   # exactly m E_t: exact, as before
    with off("moe_plain_extras_sink"):
        assert _why(t, h, hh=96)


def test_exp_regime_refuted():
    """A renormalised softmax target on a renormalised sigmoid host stays a router-score wall: the exp-regime
    construction (moe/router_frontier.py exp:*) is below the standard on real checkpoints."""
    t, h = _msig(E=48, k=4), _msig(E=128, k=8, bias=1, **SIG)
    assert "router score" in _why(t, h, hh=4096)
    path = "moe/data/router_frontier.jsonl"
    if os.path.exists(path):
        rows = [json.loads(x) for x in open(path)]
        exp = [r for r in rows if r["pair"].startswith("exp:")]
        assert exp and not any(r["payable"] for r in exp)


def _zoo(name):
    for n, c in mg.iter_json_configs(mg.ZOO_CONFIGS):
        if n == name:
            return c
    pytest.skip(f"{name} not in the snapshot")


@pytest.mark.parametrize("name", ["Edge0/Edge0-8B-A1B-preview", "inclusionAI/Ming-3.0-asr-tiny"])
def test_group_code_by_arch(name):
    """The BailingMoeV3 gate's group_limited_topk (top-2 sum per group, -inf fill) read for a typeless Edge0 config and
    for Ming-3.0-ASR: their group cap is top2sum, the code Ling-3's points carry (an unread '?' before)."""
    c = _zoo(name)
    dims = mg.dims_of(c)
    assert mg._moe_fields(dims["moe_sig"])["group_cap"] == "top2sum"
    with off("group_code_by_arch"):
        assert mg._moe_fields(mg.dims_of(c)["moe_sig"])["group_cap"].startswith("?")


@pytest.mark.slow
def test_fr_moe_headline():
    """The payable headline with the fr-moe rules, and each rule's leave-one-out delta (exact slice: only the group
    code, a classification fix, moves it)."""
    from graph.currency_curve import n_frontier, points
    specs = ("moe_plain_extras_sink", "group_code_by_arch")
    base = n_frontier(mg.emul_matrix(points(), mode="payable"))
    deltas = {}
    for s in specs:
        with off(s):
            deltas[s] = n_frontier(mg.emul_matrix(points(), mode="payable")) - base
    # (release-rel: 293 with the other first-wave rules merged; 285 on the branch base e621748)
    assert base == 293 and deltas == {"moe_plain_extras_sink": 2, "group_code_by_arch": 2}
    with on("moe_null_experts"):
        assert n_frontier(mg.emul_matrix(points(), mode="payable")) == 291   # (release-rel: 291; 283 on the branch base) -- nulls on: -2
