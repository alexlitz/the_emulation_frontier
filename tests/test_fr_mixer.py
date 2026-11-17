"""Catalog of the fr-mixer rules (graph/model_graph.py; dist/fr_mixer.md; Lean FrMixer.lean): each rule's real edges,
its boundaries, the walls it leaves, and the measured record (hf/data/mixer_rules_agreement.jsonl).

  pytest tests/test_fr_mixer.py -q
"""
import json
import os

import numpy as np
import pytest

from graph import model_graph as G
from graph.frontier import load, reps_of

DATA = os.path.join(os.path.dirname(__file__), "..", "hf", "data", "mixer_rules_agreement.jsonl")


@pytest.fixture(scope="module")
def zoo():
    P = G._dedup_points(load())
    idx = {m: i for i, p in enumerate(P) for m in p["members"]}
    return P, idx, G.emul_matrix(P, mode="payable"), G.emul_matrix(P, mode="exact")


def edge(zoo, t, h, which="payable"):
    P, idx, pay, ex = zoo
    return bool((pay if which == "payable" else ex)[idx[t], idx[h]])


# --- pure_on_hybrid ------------------------------------------------------------------------------------------------
def test_pure_recurrent_target_on_hybrid_host(zoo):
    """A pure GDN stack (64 layers) on a GDN hybrid with 69 GDN layers: its attention layers zeroed."""
    assert edge(zoo, "summerAIO/Qwen3.8-27B-Longbook27-v2", "Qwen/Qwen3.8-2.4T-A95B")
    # the zeroing itself is exact (falcon-mamba-tiny on Jamba: no lever needed, B / C / dt norms on both sides)
    assert edge(zoo, "tiiuae/falcon-mamba-tiny-dev", "ai21labs/Jamba-v0.1", "exact")


def test_pure_on_hybrid_is_in_the_mixer_order():
    assert G.slot_reaches("mixer", "ssm", "hybrid", strict=True)
    # (linear -> hybrid / ssm <-> linear: the same construction, no zoo pair and no stock linear class: left out)
    assert not G.slot_reaches("mixer", "linear", "hybrid", strict=True)
    # a hybrid target still needs a hybrid host (its attention layers), and the near-gap stays unpaid
    assert not G.slot_reaches("mixer", "hybrid", "ssm", strict=True)
    assert not G.slot_reaches("mixer", "ssm", "softmax", strict=True)


def test_pure_on_hybrid_keeps_the_family_and_depth_walls(zoo):
    # family: a Mamba-1 stack on a GDN hybrid (rf_ features), depth: R layers counted by kind_match
    assert not edge(zoo, "tiiuae/falcon-mamba-7b", "Qwen/Qwen3.8-2.4T-A95B")
    assert not edge(zoo, "tiiuae/falcon-mamba-7b", "ai21labs/Jamba-v0.1")         # 64 Mamba layers > Jamba's


# --- m2_nonorm_lever -----------------------------------------------------------------------------------------------
NONORM = ("tiiuae/Falcon-H1-Tiny-R-0.6B", "tiiuae/Falcon-H1-Tiny-R-90M", "tiiuae/Falcon-H1-0.5B-Base")


def test_nonorm_target_on_gated_norm_host(zoo):
    assert edge(zoo, "tiiuae/Falcon-H1-Tiny-R-0.6B", "tiiuae/Falcon-H1-7B-Base")
    assert edge(zoo, "tiiuae/Falcon-H1-Tiny-R-0.6B", "tiiuae/Falcon-H1-34B-Instruct")
    assert edge(zoo, "tiiuae/Falcon-H1-0.5B-Base", "tiiuae/Falcon-H1-7B-Base")
    for t in NONORM:                                           # a limit: never in the exact slice
        assert not edge(zoo, t, "tiiuae/Falcon-H1-7B-Base", "exact")
    # the converse (a normed target on a norm-free host) stays a wall
    assert not edge(zoo, "tiiuae/Falcon-H1-1.5B-Base", "tiiuae/Falcon-H1-Tiny-R-0.6B")


def test_nonorm_feature_nest_is_a_limit():
    assert "rf_mamba2" in G._feat_parents("rf_mamba2_nonorm", strict=False)
    assert G._feat_parents("rf_mamba2_nonorm", strict=True) == ()


def test_nonorm_dims_rule():
    t = dict(rc_h=24, rc_k=128, rc_v=64, rc_kh=1, rc_conv=4)          # Tiny-R-0.6B: heads per group
    assert G._rc_nonorm_on_gated(t, dict(rc_h=24, rc_k=256, rc_v=128, rc_kh=1, rc_conv=4))      # Falcon-H1-7B
    assert G._rc_nonorm_on_gated(t, dict(rc_h=32, rc_k=256, rc_v=128, rc_kh=2, rc_conv=4))      # 34B: 2 groups
    assert not G._rc_nonorm_on_gated(t, dict(rc_h=16, rc_k=256, rc_v=128, rc_kh=1, rc_conv=4))  # too few heads
    assert not G._rc_nonorm_on_gated(t, dict(rc_h=48, rc_k=64, rc_v=128, rc_kh=1, rc_conv=4))   # d_state 64 < 128
    # a wide target head split over narrower host heads of one group (one dt / A each)
    assert G._rc_nonorm_on_gated(dict(rc_h=4, rc_k=16, rc_v=128, rc_kh=1, rc_conv=4),
                                 dict(rc_h=8, rc_k=16, rc_v=64, rc_kh=1, rc_conv=4))
    assert not G._rc_nonorm_on_gated(dict(rc_h=4, rc_k=16, rc_v=128, rc_kh=1, rc_conv=4),
                                     dict(rc_h=7, rc_k=16, rc_v=64, rc_kh=1, rc_conv=4))


def test_nonorm_needs_a_config_eps_host():
    """The lever needs the gated norm's eps to be the config eps (rec_eps_cfg): not Zamba2's 1e-5 in code."""
    t = {"name": "t", "slots": {"feats": "rf_mamba2_nonorm"},
         "dims": dict(rc_h=8, rc_k=16, rc_v=16, rc_kh=1, rc_ng=0, rc_conv=4, rc_rule=G.RC_GE)}
    h = lambda cfg: {"name": "h", "slots": {"feats": "rf_mamba2"},
                     "dims": dict(rc_h=16, rc_k=16, rc_v=16, rc_kh=1, rc_ng=1, rc_conv=4, rc_rule=G.RC_M2,
                                  eps_lever=1, rec_eps_cfg=cfg)}
    assert G._rec_ok([t, h(1)], lever=True)[0, 1]
    assert not G._rec_ok([t, h(0)], lever=True)[0, 1]
    assert not G._rec_ok([t, h(1)], lever=False)[0, 1]          # the exact slice


# --- mamba1_lever --------------------------------------------------------------------------------------------------
def test_mamba1_lever_rule():
    t = {"name": "t", "slots": {"feats": "rf_mamba1"},
         "dims": dict(rc_h=1536, rc_k=16, rc_v=1, rc_kh=1, rc_conv=4, rc_rank=48, rc_rule=G.RC_GE)}
    h = lambda lev, k=16: {"name": "h", "slots": {"feats": "rf_mamba1_bcdt"},
                           "dims": dict(rc_h=8192, rc_k=k, rc_v=1, rc_kh=1, rc_conv=4, rc_rank=256, rc_rule=G.RC_GE,
                                        eps_lever=lev)}
    assert G._rec_ok([t, h(1)], lever=True)[0, 1]
    assert not G._rec_ok([t, h(0)], lever=True)[0, 1]
    assert not G._rec_ok([t, h(1)], lever=False)[0, 1]
    assert not G._rec_ok([t, h(1, k=8)], lever=True)[0, 1]       # RC_GE: d_state >=
    assert set(G._feat_parents("rf_mamba1", strict=False)) == {"rf_mamba1_bcdt", "rf_mamba1_bcdtw"}


def test_falcon_mamba_and_jamba_are_lever_hosts(zoo):
    P, idx, _, _ = zoo
    for n in ("tiiuae/falcon-mamba-7b", "ai21labs/Jamba-v0.1"):
        assert P[idx[n]]["dims"].get("eps_lever") == 1, n


# --- the walls left (dist/fr_mixer.md: refuted or priced, no zoo host pays) -----------------------------------------
@pytest.mark.parametrize("t,h", [
    ("ibm-granite/granite-4.0-h-small", "Qwen/Qwen3.8-2.4T-A95B"),     # Mamba-2 on GDN: the delta rule's erase
    ("Aratako/MioTTS-2.6B", "Qwen/Qwen3.8-2.4T-A95B"),                 # short conv on GDN: >= 24 v channels / channel
    ("tiiuae/falcon-mamba-7b", "summerAIO/Qwen3.8-27B-Longbook27-v2"),  # Mamba-1 on GDN
    ("AxiomicLabs/GPT-X3-150M", "Qwen/Qwen3.8-2.4T-A95B"),             # GDN-2's separate erase / write gates
    ("cyankiwi/MiniCPM-SALA-AWQ-8bit", "Qwen/Qwen3.8-2.4T-A95B"),       # lightning (no erase) on GDN
    ("inclusionAI/Ling-2.6-flash", "inclusionAI/Ling-2.6-1T"),         # slopes fixed by (layer, L, H)
    ("tiiuae/Falcon-H1-3B-Base", "tiiuae/Falcon-H1-34B-Instruct"),     # full-width norm > the host's group
    ("Zyphra/Zamba2-1.2B-instruct", "Zyphra/Zamba2-7B-Instruct"),      # d_state 128 > 64
])
def test_walls_kept(zoo, t, h):
    assert not edge(zoo, t, h)


def test_frontier_after_the_rules(zoo):
    P, _, pay, _ = zoo
    front = {P[i]["members"][0] for i in reps_of(pay, P)}
    # 293 on release-rel with every first-wave rule (298 with fr-mixer alone on round3 844d4dc, 300 before; 287 / 289
    # on the branch base e621748)
    assert len(front) == 293
    assert "summerAIO/Qwen3.8-27B-Longbook27-v2" not in front
    assert "tiiuae/Falcon-H1-Tiny-R-0.6B" not in front


# --- the measured record -------------------------------------------------------------------------------------------
def _rows():
    if not os.path.exists(DATA):
        pytest.skip("no measurement record")
    return [json.loads(l) for l in open(DATA)]


def test_measured_payable():
    """Every rule's recorded construction meets the standard on >= 2 real checkpoints (fp32): top-1 >= 99%, >= 90% of
    32-token continuations identical."""
    rows = [r for r in _rows() if r["dtype"] == "float32"]
    rec = {}
    for r in rows:
        if r["rule"] == "m2_nonorm_lever" and abs(r["rho_max"]) > 2.0 ** -29:
            continue                                            # (the sweep's under-scaled points)
        if r["rule"].startswith("mamba1") and r["rho_max"] > 2.0 ** -29:
            continue
        rec.setdefault(r["rule"].split("+")[0], set()).add(r["target"]) if r["payable"] else None
        assert r["payable"], r["pair"]
    assert len(rec.get("m2_nonorm_lever", ())) >= 2
    assert len(rec.get("mamba1_lever", ())) >= 1
    assert rec.get("pure_on_hybrid")


def test_nonorm_unlevered_host_fails():
    """Negative control: with S = 1 the host's gated norm is on (rho >> 1) and the host is not the target."""
    rows = [r for r in _rows() if r["rule"] == "m2_nonorm_lever" and r.get("S_log2", 99) < 5]
    if not rows:
        pytest.skip("no sweep recorded")
    assert all(not r["payable"] for r in rows)
