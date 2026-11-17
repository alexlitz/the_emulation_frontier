"""Branch fr-cost: the cost-charge counterfactual hook and the cheaper constructions it found.

  pytest tests/test_fr_cost.py -q
"""
import math

import numpy as np
import pytest

from graph import model_graph as G
from tests.test_graph_rules import mk


def test_cost_relax_is_empty_in_the_relation():
    """COST_RELAX is a counterfactual hook only: empty by default, every name a known charge."""
    assert not G.COST_RELAX
    assert len(set(G.COST_RELAX_NAMES)) == len(G.COST_RELAX_NAMES)


def test_cost_relax_restores_itself():
    from graph.fr_cost import relation
    P = [mk(act="GeLU", ffn_total=4096, ffn_active=4096), mk(act="SiLU", ffn_total=8192, ffn_active=8192)]
    base = relation(P)
    relaxed = relation(P, ("act_k",))
    assert not G.COST_RELAX
    assert (relation(P) == base).all()
    assert (relaxed | ~base).all()                      # relaxing a charge only adds pairs


# --- cost_norm_subspace: RMSNorm target on a LayerNorm host at hidden + 1 ---------------------------------------

def test_subspace_basis_is_a_mean_zero_isometry():
    import torch
    from hf.norm_subspace import subspace_basis
    for d, dh in ((7, 8), (24, 25), (24, 40)):
        U = subspace_basis(d, dh)
        assert (U.T @ U - torch.eye(d, dtype=U.dtype)).abs().max() < 1e-13
        assert U.sum(0).abs().max() < 1e-13


def test_layernorm_of_the_isometry_is_rmsnorm_fp64():
    pytest.importorskip("torch")
    from hf.norm_subspace import check_functional
    r = check_functional()
    assert r["err"] < 1e-12 and r["orth"] < 1e-12 and r["mean0"] < 1e-12


def test_random_llama_on_a_stablelm_host_fp64():
    pytest.importorskip("transformers")
    from hf.norm_subspace import check_random_llama
    r = check_random_llama()
    assert r["max_abs"] < 1e-12 and r["host_hidden"] < 2 * r["target_hidden"]


def _rms_on_ln(ht, hh, t_feats="", h_feats="", **kw):
    t = mk(norm="RMSNorm", hidden=ht, **kw)
    h = mk(norm="LayerNorm", hidden=hh, **kw)
    t["slots"]["feats"], h["slots"]["feats"] = t_feats, h_feats
    return G.emulates(h, t)


def test_subspace_charge_is_hidden_plus_one():
    assert _rms_on_ln(1024, 1025)
    assert not _rms_on_ln(1024, 1024)


def test_subspace_charge_keeps_the_mirror_where_it_must():
    # a residual-written gained norm on the target (sandwich post-norms) or a tied host: 2x
    assert not _rms_on_ln(1024, 1025, t_feats="post_norm", h_feats="post_norm")
    assert _rms_on_ln(1024, 2048, t_feats="post_norm", h_feats="post_norm")
    assert not _rms_on_ln(1024, 1025, t_feats="tied_emb", h_feats="tied_emb")
    assert _rms_on_ln(1024, 2048, t_feats="tied_emb", h_feats="tied_emb")


def test_subspace_key_off_is_the_mirror():
    G.R3_OFF.add("cost_norm_subspace")
    try:
        assert not _rms_on_ln(1024, 1025)
        assert _rms_on_ln(1024, 2048)
    finally:
        G.R3_OFF.discard("cost_norm_subspace")


def test_recorded_real_pairs_pass_the_standard():
    """hf/data/norm_subspace_agreement.jsonl: >= 2 real Llama-family targets on stock StableLM hosts at d + head_dim,
    fp32, top-1 >= 99% and >= 90% identical 32-token continuations."""
    import json
    import os
    from hf.norm_subspace import OUT
    rows = [json.loads(x) for x in open(OUT)] if os.path.exists(OUT) else []
    good = {r["model"] for r in rows if r["payable"] and r["host_hidden"] < r["mirror_hidden"]}
    assert len(good) >= 2, rows


# --- cost_act_k_measured: GeLU <- SiLU (no shift) at the measured k = 5 ---------------------------------------------

def test_measured_act_k():
    assert G.act_width_k("GeLU", "SiLU", False) == 5
    assert G.act_width_k("GeLU_tanh", "SiLU", False) == 5
    assert G.act_width_k("GeLU", "SiLU", True) == 5             # the shift table's own k
    assert G.act_width_k("SiLU", "GeLU", False) == 3            # not measured cheaper: unchanged
    G.R3_OFF.add("cost_act_k_measured")
    try:
        assert G.act_width_k("GeLU", "SiLU", False) == 6
    finally:
        G.R3_OFF.discard("cost_act_k_measured")


def test_measured_act_k_rests_on_the_recorded_agreement():
    """Every GeLU model measured at k = 5 (no shift) on a SiLU host passes the strict standard; k = 4 does not."""
    from graph.act_width import AGREEMENT, payable
    k5 = [v for (m, t, h, k, s), v in AGREEMENT.items() if (t, h, k, s) == ("GeLU", "SiLU", 5, False)]
    k4 = [v for (m, t, h, k, s), v in AGREEMENT.items() if (t, h, k, s) == ("GeLU", "SiLU", 4, False)]
    assert len(k5) >= 6 and all(payable(a, b) for a, b, _ in k5)
    assert not all(payable(a, b) for a, b, _ in k4)
