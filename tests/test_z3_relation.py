"""The Z3 model of the emulation relation (verify/z3_relation.py) agrees with graph.model_graph.emul_matrix, and the
solver checks of its properties; the edge table (verify/edge_table.py) has no unjustified headline rule.

  pytest tests/test_z3_relation.py                    # differential sample, proofs, fast transitivity conjuncts
  Z3_TRANSITIVITY=1 pytest tests/test_z3_relation.py  # + every transitivity conjunct in every mode (~6 h)

The full run (every ordered pair of the real shapes, every mode, both flag settings, 150 Hypothesis lists per flag
set) is `python -m verify.z3_relation`; it also rewrites verify/EDGE_JUSTIFICATION.md.
"""
import os

import numpy as np
import pytest

pytest.importorskip("z3")
pytest.importorskip("hypothesis")

from graph import model_graph as G
from verify import edge_table as E
from verify import z3_relation as Z

TIMEOUT_MS = 30_000


@pytest.fixture(scope="module")
def zoo():
    M, P = Z.real_points()
    J = Z.join_host(M)
    return M, P, J, Z.Space(P, extra=[J])


@pytest.fixture(scope="module")
def model(zoo):
    return Z.Model(zoo[3])


def test_real_shapes_are_well_formed(zoo):
    """Every built and observed wf clause holds on every real shape (the proofs' domain contains the zoo)."""
    assert Z.check_wf_real(zoo[3], zoo[1], log=lambda *_: None) == 0


def test_closed_forms_match_the_code():
    """need_heads is kv_hosts_needed (its greedy loop) on a box; the lemma need_antitone_q holds on its box."""
    assert Z.check_kv_closed_form()[0] == 0
    assert Z.check_lemmas() == 0


@pytest.mark.parametrize("ctx,vocab", [(False, False), (True, True)])
def test_differential_real_sample(zoo, ctx, vocab):
    """The model agrees with emul_matrix on every ordered pair of a sample of real shapes, every mode, and with the
    joined host as a host column (python -m verify.z3_relation runs all the pairs)."""
    _, P, J, space = zoo
    rng = np.random.default_rng(0)
    S = [P[i] for i in sorted(rng.choice(len(P), 150, replace=False))]
    assert Z.differential_real(space, S, flags=((ctx, vocab),), z3_sample=30, log=lambda *_: None, J=J) == 0


def test_differential_hypothesis(zoo):
    """...and on synthetic point lists over the whole value space (tops, branch sets, sentinels, NoPE slots, MoE
    signatures, recurrent dims), most points mutants of a few (near-equal pairs: edges and their boundaries)."""
    assert Z.differential_hypothesis(zoo[3], examples=25, log=lambda *_: None) == 0


@pytest.fixture(scope="module")
def payable_parts(zoo):
    _, P, _, space = zoo
    ev = Z.Evaluator(Z.Model(space, False, False, Z.ROPE_K_DIFF, Z.FW_K_DIFF))
    return ev.parts_matrix(P, "payable"), ev.matrix(P, "payable"), G.emul_matrix(P, mode="payable")


@pytest.mark.parametrize("drop", ["rot_gate", "kv_heads", "moe", "kinds", "rec", "feats", "mla_latent", "pos", "act"])
def test_differential_has_teeth(payable_parts, drop):
    """Negative control: the model with one conjunct dropped disagrees with emul_matrix on real pairs."""
    parts, full, real = payable_parts
    off = ~np.eye(len(real), dtype=bool)
    assert (full == real).all()
    weaker = np.logical_and.reduce([m for k, m in parts.items() if k not in (drop, "compensator")])
    assert (weaker & ~real)[off].any(), drop


@pytest.mark.parametrize("mode", Z.MODES)
def test_reflexive_on_realistic_shapes(model, mode):
    """Reflexive (the code's diagonal) on every realistic shape. On wf shapes the rot gate refutes it: a fraction-
    free point whose rotary dims exceed its head_dim does not host itself (fixed_rot_fits: d_t <= hd_h), confirmed on
    emul_matrix; no real shape has rotary dims > head_dim."""
    st, out = Z.prove_reflexive(model, mode, TIMEOUT_MS)
    bad = {k: v for k, v in out.items() if v[0] != "PROVED"}
    assert set(bad) <= {"rot_gate"}, bad
    for k, (s, cex) in bad.items():
        assert s == "REFUTED" and cex["on_realistic"] == "PROVED" and cex["code"] is False


@pytest.mark.parametrize("lo,hi", list(zip(Z.MODES, Z.MODES[1:])))
def test_mode_monotone(model, lo, hi):
    assert Z.prove_monotone(model, lo, hi, TIMEOUT_MS)[0] == "PROVED"


def test_nomix_is_payable(model):
    assert Z.prove_alias(model, TIMEOUT_MS) == "PROVED"


def test_real_shapes_transitive_and_monotone(zoo):
    """On the real shapes the relation is transitive and monotone in every mode (no counterexample is a real triple)."""
    _, P, _, space = zoo
    assert all(n == 0 for n, _ in Z.real_transitivity(space, P, log=lambda *_: None).values())
    assert all(n == 0 for n in Z.real_monotone(P, log=lambda *_: None).values())


# (round 3, batch 4a (d_transitivity_depth): the zero-edge counts are built as the own-KV kind sums (a "built" wf
# clause), and depth was PROVED transitive in exact and payable; in mixgap the solver times out on it with the CLA-carry
# routing (d_transitivity_mixer_cla; PROVED with that key off), and mixer_cla keeps a wf counterexample through an
# SSM-labelled host with attention layers (none on the zoo: the observed clause). The per-conjunct expectation table
# EXPECTED_NOT_TRANSITIVE / MAY_TIME_OUT gave way to the opt-in test_transitivity_counterexamples_are_real below and the
# per-mode properties of verify/EDGE_JUSTIFICATION.md. depth is not in FAST_PROVED: on round3 f88b44a and on the merge
# alike its exact / payable queries time out at 30 s and 90 s under load (a 7.5 s payable proof once), not a fast check)


# conjuncts proved transitive in seconds (alone, or with the relation on both hops): a fast regression check
FAST_PROVED = {"payable": ["mtp", "gen", "softcap", "mixer_cla", "qheads", "head_dim", "alibi_alibi", "nope_alibi",
                           "zamba"],
               "capacity": ["gen", "kv_cap", "q_cap", "qheads", "head_dim", "ffn", "depth", "hidden"]}


@pytest.mark.parametrize("mode", sorted(FAST_PROVED))
def test_transitive_conjuncts(model, mode):
    st, out = Z.transitivity(model, mode, TIMEOUT_MS, parts=FAST_PROVED[mode])
    assert {k: v["status"] for k, v in out.items()} == {k: "PROVED" for k in FAST_PROVED[mode]}


@pytest.mark.skipif(not os.environ.get("Z3_TRANSITIVITY"), reason="every conjunct in every mode takes ~6 h: "
                    "Z3_TRANSITIVITY=1 (python -m verify.z3_relation runs it)")
@pytest.mark.parametrize("mode", Z.MODES)
def test_transitivity_counterexamples_are_real(model, mode):
    """Every counterexample the solver finds is a violation of the real emul_matrix."""
    st, out = Z.transitivity(model, mode, 90_000, min_budget_s=60)
    assert all(v["cex"]["confirmed_on_code"] for v in out.values() if v.get("cex"))


@pytest.mark.xfail(strict=True, reason="emul_matrix is not transitive for all inputs (verify/EDGE_JUSTIFICATION.md, "
                   "Properties: per-conjunct counterexamples confirmed on the code); capacity is UNKNOWN (kv_heads)")
def test_transitive_payable(model):
    st, out = Z.transitivity(model, "payable", TIMEOUT_MS, parts=["norm"])
    assert st == "PROVED"


def test_box_transitive_payable(zoo):
    """Bounded exhaustive check of the real code: the KV-head / rotary families and the recurrent rules' family are
    transitive in payable (pass 5 closed the cap > schedule packing violation of pass 4)."""
    space = zoo[3]
    for fam in ("heads", "heads_eq", "rec"):
        assert Z.box_transitivity(space, fam, ("payable",), 4, log=lambda *_: None)["payable"][0] == 0, fam


@pytest.mark.parametrize("mode", Z.MODES)
def test_join_dominates(zoo, model, mode):
    """The joined host emulates every realistic model except the blocks it does not carry: activations outside its
    tested branches, the qk-norm variants outside HOST_QKNORM, the FEATURES outside HOST_FEATS."""
    st, out = Z.join_dominance(model, mode, zoo[0], TIMEOUT_MS, 10_000)
    blocked = {k for k, (s, _) in out.items() if s != "PROVED"}
    assert blocked <= JOIN_MAY_BLOCK[mode], blocked
    if "feats" in blocked:
        from graph.minimal_emulator import HOST_FEATS
        fb = set(out["feats"][1]["blocked_features"])
        assert fb == JOIN_BLOCKED_FEATS and not fb & set(HOST_FEATS), fb


# the joined host carries only the TESTED activation branches, the qk-norm variants of HOST_QKNORM and HOST_FEATS
# (round 3, batch 1: and only its UNorm embedding-norm branches -- no gated embedding norm -- and no grouped RMSNorm:
# emb_norm / norm_groups were {act, qknorm, feats}; in the exact slice also an attention-only post-norm target, whose
# eps lever is a limit: pna_lever; round 3, batch 2d: and no normed query latent -- Step-3's share_q_dim is hosted only
# by an equal one: q_lat_norm was not in the sets; round 3, batch 3b: and no HadamardMLP -- needle's FFN is hosted only by
# its own code at equal n: hadamard was not in the sets; round 3, batch 4b: and only its Full / Block AttnRes partitions
# -- ARLoop's windowed block AttnRes (attn_res_sig "18/3") has no UniversalElephantine branch: attn_res_sig was not in
# the sets)
# (release-all: the join now carries the zoo's RMSNorm group counts as a branch set, the gated embedding norm, the
# post_norm_attn feature, ARLoop's 18/3 and 24/3 partitions and the value-embedding kinds ve|ve_layer|ve_shared:
# norm_groups and pna_lever left the sets. emb_norm stays for an unlisted embedding-norm kind only (the "<emb?>"
# sentinel); attn_res_sig for "2/0" and unlisted signatures; ve_kind joins them -- v_residual (and unlisted kinds) have
# no UniversalElephantine branch. The feats conjunct blocks exactly JOIN_BLOCKED_FEATS, the features outside
# HOST_FEATS a realistic target may carry, the same in every mode)
JOIN_MAY_BLOCK = {"exact": {"act", "qknorm", "feats", "emb_norm", "q_lat_norm", "hadamard", "attn_res_sig", "ve_kind"},
                  "payable": {"act", "qknorm", "feats", "emb_norm", "q_lat_norm", "hadamard", "attn_res_sig", "ve_kind"},
                  "mixgap": {"act", "qknorm", "feats", "emb_norm", "q_lat_norm", "hadamard", "attn_res_sig", "ve_kind"},
                  "capacity": set()}
# (release-fix: the six activated / biased output-gate keys out_gate_[head_][bias_]{silu,softplus} left it -- the host's
# gate_silu branch and g_proj's bias, test_activated_output_gate)
JOIN_BLOCKED_FEATS = {"byte_patch", "cca", "cross_attn", "doc_mask", "enc_input", "engram_klm", "engram_needle", "fope",
                      "hc_needle", "hc_san", "loop_kv_mix", "mem_attn", "mla_q_raw", "mla_qmix", "mova",
                      "out_gate_qlat", "rope_warp"}


@pytest.fixture(scope="module")
def edge_rows(zoo):
    return E.edge_rows(zoo[3], zoo[1])


def test_edge_table_names_real_theorems(edge_rows):
    """edge_rows raises on a theorem name the Lean library / PROOFS.md lacks; every allowed rule has a row."""
    assert edge_rows


def _gaps(rows):
    return sorted(f"{r['just'].verdict}: {r['rule']}" for r in E.headline_gaps(rows)[0])


# (round 3, batch 3b: "glu <- vanilla at 2x depth" left the list -- a wall now, spot_glu_vanilla; the z3-final reason
# named it as the 1 gap. Round 3 then had 12 gaps -- the prefix-LM generation transitions and seven feature switches --
# closed by round 3's drafted edge-table rows (dist/r3docs/edge_table_rows.diff), and the release's one new gap, fr-mixer's
# mixer ssm -> hybrid, by its FrMixer row: 0 now, so the strict xfail is gone)
def test_no_unjustified_headline_rules(edge_rows):
    assert _gaps(edge_rows) == []
