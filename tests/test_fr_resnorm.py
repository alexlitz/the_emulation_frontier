"""fr-resnorm: the relation's three new credits (graph/model_graph.py ports hc_off, sc_norm_off, ln_beta_readout;
dist/fr_resnorm.md), on the snapshot's own shapes. The constructions themselves are catalog entries
(verify/spot/catalog/fr.py, tests/test_reduction_catalog.py) and real-checkpoint measurements (hf/data/*.jsonl).

  python -m pytest tests/test_fr_resnorm.py -q
"""
import contextlib
import functools
import json
import os

import numpy as np
import pytest

from graph import model_graph as mg
from graph.currency_curve import points

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = ("fr_hc_host_off", "fr_ln_beta_readout", "fr_selfcond_norm_host")


@contextlib.contextmanager
def off(*specs):
    added = [s for s in specs if s not in mg.R3_OFF]
    for s in added:
        mg.R3_OFF.add(s)
    try:
        yield
    finally:
        for s in added:
            mg.R3_OFF.discard(s)


@functools.lru_cache(maxsize=None)
def _P():
    P = points()
    return P, {m: i for i, p in enumerate(P) for m in p["members"]}


@functools.lru_cache(maxsize=None)
def _ok(specs_off=(), mode="payable"):
    P, _ = _P()
    with off(*specs_off):
        return mg.emul_matrix(P, mode=mode)


def _edge(t, h, specs_off=(), mode="payable"):
    P, idx = _P()
    return bool(_ok(specs_off, mode)[idx[t], idx[h]])


def test_hc_host_off_pays_glm5_on_hy4():
    """GLM-5 (no hyper-connections) on Hy4 (hc_hy4): the only wall was the host's streams; exact, so in every mode."""
    assert _edge("zai-org/GLM-5", "tencent/Hy4-preview-FP8")
    assert not _edge("zai-org/GLM-5", "tencent/Hy4-preview-FP8", ("fr_hc_host_off",))


def test_hc_host_off_unchecked_variants_stay_walls():
    """A dsv41 / xing / needle / motif / dsv4_nano host keeps its streams (remote, unread code)."""
    for v in ("dsv41", "xing", "needle", "motif", "dsv4_nano", "san"):
        assert mg._fr_host_off(f"hc_{v}+hyper_conn", 0, True) == frozenset()
    for v in mg.HC_HOST_OFF_VARIANTS:
        assert mg._fr_host_off(f"hc_{v}+hyper_conn", 0, True) == frozenset({"hyper_conn", f"hc_{v}"})
    assert mg._fr_host_off("hyper_conn", 0, True) == frozenset()         # no variant key: no switch-off


def test_hc_target_side_stays_a_wall():
    """A plain host never plays a hyper-connection target."""
    assert not mg.feats_ok("hc_hy4+hyper_conn", "", host_off=frozenset())
    assert mg.feats_ok("", "hc_hy4+hyper_conn", host_off=mg._fr_host_off("hc_hy4+hyper_conn", 0, True))


def test_sc_norm_off_pays_wald_on_diffusiongemma_only_payable():
    t, h = "org2ai/Wald-26B-A4B", "yifanyu/DiffusionGemma-26B-A4B-it-ORL"
    assert _edge(t, h)
    assert not _edge(t, h, ("fr_selfcond_norm_host",))
    assert not _edge(t, h, mode="exact")                                   # a limit (tied head): not exact


def test_sc_norm_off_not_for_unread_code():
    """QwenDLM-ORL's selfcond norm (remote, unpublished modeling_sdar.py) stays a wall."""
    P, idx = _P()
    assert not P[idx["yifanyu/QwenDLM-4B-ORL"]]["dims"].get("sc_norm_off")
    assert P[idx["yifanyu/DiffusionGemma-26B-A4B-it-ORL"]]["dims"].get("sc_norm_off")


def test_ln_beta_readout_needs_ffn_bias_and_untied_head():
    """The four LayerNorm-beta frontier targets stay walled: their best hosts lack an FFN-input bias (TW3-JRGL-v2)
    or tie the head (granite-8b-code)."""
    for t, h in (("stabilityai/stable-code-3b", "paloalma/TW3-JRGL-v2"),
                 ("microsoft/Phi-3-small-128k-instruct", "ibm-granite/granite-8b-code-base-4k")):
        assert not _edge(t, h)
    ok, ok0 = _ok(), _ok(("fr_ln_beta_readout",))
    new = np.argwhere(ok & ~ok0)
    P, _ = _P()
    assert len(new)
    for i, j in new:
        a, b = P[i], P[j]
        assert a["slots"]["norm"] == "LayerNorm" and a["dims"].get("ln_beta")
        assert b["dims"].get("ffn_bias") and "tied_emb" not in b["slots"].get("feats", "").split("+")
        assert b["dims"]["hidden"] >= a["dims"]["hidden"] + 1


def test_ports_switch_off():
    for port, spec in (("hc_off", "fr_hc_host_off"), ("sc_norm_off", "fr_selfcond_norm_host"),
                       ("ln_beta_readout", "fr_ln_beta_readout")):
        assert port in mg.P5_PORTS
        mg.P5_OFF.add(port)
        try:
            P, _ = _P()
            assert np.array_equal(mg.emul_matrix(P, mode="payable"), _ok((spec,)))
        finally:
            mg.P5_OFF.discard(port)


def test_no_new_transitivity_violations():
    """The headline's transitivity violations (9 on round3, before this branch) are unchanged."""
    P, _ = _P()

    def viol(ok):
        S = ok.astype(np.int32)
        return {(i, j) for i, j in np.argwhere(((S @ S) > 0) & ~ok)}
    assert viol(_ok()) <= viol(_ok(SPECS))
    assert not viol(_ok(mode="exact"))


def test_measurements_payable():
    """Every recorded real-checkpoint run of the three rules meets the standard (top-1 >= 99%, >= 90% identical
    32-token continuations); at least two target checkpoints each."""
    for fn, key, cond in (("hc_host_agreement.jsonl", "target", lambda r: True),
                          ("ln_beta_readout.jsonl", "target", lambda r: r["variant"] == "readout"),
                          ("selfcond_norm_agreement.jsonl", "target", lambda r: True)):
        path = os.path.join(ROOT, "hf", "data", fn)
        if not os.path.exists(path):
            pytest.skip(f"no {fn}")
        rows = [r for r in map(json.loads, open(path)) if cond(r)]
        assert rows and all(r["payable"] for r in rows), fn
        assert len({r[key] for r in rows}) >= 2, fn
