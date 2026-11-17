"""The k-unit activation swap (graph/act_width.py): activation walls paid in FFN width.

  * the K tables are the first k whose stored k-term fit (act/data/act_width_fits.json) is within
    ACT_FLOOR_MAX on [-160, 160], and the stored fits re-evaluate to their recorded errors;
  * the k = 1 cells agree with the one-unit rule of graph/model_graph.py (exact folds and limits);
  * the shift-free obstruction: a swish host (X(z) - X(-z) = z) without gate shifts has a LINEAR odd
    part, so it cannot fit mish / situ / relu2 / xIELU below their odd-part distance from a line;
  * the recorded real-model agreement: the minimal-k fit passes the harness's strict standard;
  * act_width_ok charges ffn_total and ffn_active x k; width_edges adds exactly those edges.
"""
import json
import os

import numpy as np
import pytest
import torch

import graph.act_width as aw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FITS = os.path.join(ROOT, "act", "data", "act_width_fits.json")
AGREE = os.path.join(ROOT, "act", "data", "act_width_agreement.jsonl")
SWISH = ("SiLU", "GeLU", "swigluoai")


def _fits():
    return json.load(open(FITS))


def test_threshold_is_the_headline_one():
    from graph.model_graph import ACT_FLOOR_MAX
    assert aw.ACT_FLOOR_MAX == ACT_FLOOR_MAX


def test_k_tables_match_the_stored_fits():
    res = _fits()
    for table, fam in ((aw.K_NOSHIFT, "noshift"), (aw.K_SHIFT, "shift")):
        for (t, h), k in table.items():
            if k == 1 and ((t, h) in aw.ONE_UNIT or t == h):
                continue
            names = aw.FIT_ROWS.get(t, (t,))
            for name in names:                       # zoo "GeLU" = the erf and the tanh GeLU
                c = res[f"{name}|{h}|{aw.R_TABLE:g}|{fam}"]
                errs = [e for _, e, _ in c]
                assert errs[k - 1] <= aw.ACT_FLOOR_MAX, (t, h, fam, name, errs)
            assert any(res[f"{n}|{h}|{aw.R_TABLE:g}|{fam}"][k - 2][1] > aw.ACT_FLOOR_MAX for n in names) \
                or k == 1, (t, h, fam)
    # every cell with a stored fit within the threshold at some k <= K_MAX is in the table
    for key, c in res.items():
        t, h, R, fam = key.split("|")
        if float(R) != aw.R_TABLE or t == "gelu_tanh":
            continue
        kmin = next((k for k, e, _ in c if e <= aw.ACT_FLOOR_MAX), None)
        if kmin is None:
            continue
        table = aw.K_NOSHIFT if fam == "noshift" else aw.K_SHIFT
        if h == "gelu_tanh" or (t, h) == ("mish", "GeLU"):
            # round 2, pass 5: the tanh-GeLU host fits; one k for both GeLU hosts (HOST_GELU_TANH), mish <- GeLU
            # dropped because the tanh host has no fit within the threshold
            assert aw.HOST_GELU_TANH.get(t, 1) is None or table.get((t, "GeLU"), 1) >= kmin, (key, kmin)
            continue
        assert (t, h) in table and table[(t, h)] <= kmin or (t, h) in aw.ONE_UNIT, (key, kmin)


@pytest.mark.parametrize("cell", [("GeLU", "SiLU", "noshift"), ("GeLU", "SiLU", "shift"),
                                  ("SiLU", "GeLU", "noshift")])
def test_stored_fit_reevaluates(cell):
    from act.act_width_fit import TA, evaluate
    t, h, fam = cell
    table = aw.K_NOSHIFT if fam == "noshift" else aw.K_SHIFT
    k = table[(t, h)]
    k_, err, p = _fits()[f"{t}|{h}|{aw.R_TABLE:g}|{fam}"][k - 1]
    assert k_ == k and len(p["alpha"]) == k
    if fam == "noshift":
        assert all(g == 0 for g in p["gamma"])
    e = evaluate(TA[h], TA[t], aw.R_TABLE, p, dense=2)
    assert e <= aw.ACT_FLOOR_MAX and abs(e - err) < 1e-6, (e, err)


def test_one_unit_cells_agree_with_the_headline_rule():
    from graph.model_graph import ACT_FLOOR_MAX, ACT_FLOORS
    for t, row in ACT_FLOORS.items():
        for h, fl in row.items():
            if fl <= ACT_FLOOR_MAX:
                assert aw.act_k(t, h) == 1, (t, h)
            else:
                assert aw.act_k(t, h) != 1 and aw.act_k(t, h, gate_bias=True) != 1, (t, h)
    assert aw.act_k("GeLU", "GeLU") == 1


def test_walls_stay_walls():
    for t in ("relu2", "xielu", "situ"):
        for h in SWISH + ("mish",):
            assert aw.act_k(t, h, gate_bias=True) is None, (t, h)
    assert aw.act_k("poly_norm", "SiLU", True) is None


def test_bias_free_swish_host_has_linear_odd_part():
    """sum_i alpha_i X(beta_i a) with X(z) - X(-z) = z: its odd part is (sum alpha_i beta_i) a / 2, so the
    fit error is at least the target's odd-part distance from a line -- mish's is 0.10 on [-160, 160]."""
    from act.act_width_fit import TA, grid, lp_alpha
    rng = np.random.default_rng(0)
    x = torch.linspace(-50, 50, 2001, dtype=torch.float64)
    for h in SWISH:
        a, b = rng.normal(size=5), rng.normal(size=5) * 3
        F = lambda z: (TA[h](z[:, None] * torch.tensor(b)) @ torch.tensor(a))
        assert torch.allclose(F(x) - F(-x), float((a * b).sum()) * x, atol=1e-9)
    xp = grid(aw.R_TABLE)
    xp = xp[xp >= 0]
    for t, lo in (("mish", 0.09), ("situ", 1.0)):
        odd = (TA[t](xp) - TA[t](-xp)) / 2
        bound, _ = lp_alpha(xp[:, None].numpy(), odd.numpy())
        assert bound > lo
        for h in SWISH:
            assert (t, h) not in aw.K_NOSHIFT
    for t in ("SiLU", "GeLU", "gelu_tanh", "swigluoai"):             # odd part exactly a / 2
        assert torch.allclose((TA[t](xp) - TA[t](-xp)) / 2, xp / 2, atol=1e-12)


def test_act_width_ok_charges_ffn():
    k = aw.act_k("GeLU", "SiLU")
    assert k is not None and k > 1
    assert aw.act_width_ok("GeLU", "SiLU", target_ffn=(100, 100), host_ffn=(100 * k, 100 * k))
    assert not aw.act_width_ok("GeLU", "SiLU", target_ffn=(100, 100), host_ffn=(100 * k - 1, 100 * k))
    assert not aw.act_width_ok("GeLU", "SiLU", target_ffn=(100, 100), host_ffn=(100 * k, 100 * k - 1))
    assert aw.act_k("GeLU", "SiLU", gate_bias=True) <= k
    # branch sets: a universal host takes its cheapest branch; a branch-set target needs every branch
    assert aw.act_k("GeLU", "SiLU|GeLU") == 1
    assert aw.act_k("GeLU|SiLU", "SiLU") == k
    assert aw.act_k("relu2|SiLU", "SiLU") is None


def test_host_gate_bias():
    assert aw.host_gate_bias({"model_type": "gpt2"})
    assert aw.host_gate_bias({"model_type": "gpt_neox"})
    assert not aw.host_gate_bias({"model_type": "llama", "mlp_bias": False})
    assert aw.host_gate_bias({"model_type": "llama", "mlp_bias": True})
    assert not aw.host_gate_bias({"model_type": "qwen", "no_bias": True})
    assert not aw.host_gate_bias({"model_type": "qwen2"})
    assert not aw.host_gate_bias({"model_type": "falcon", "bias": False})
    assert aw.host_gate_bias({"model_type": "gpt_oss"})


def test_width_edges():
    def pt(act, f):
        return {"slots": {"act": act}, "dims": {"ffn_total": f, "ffn_active": f}, "members": [act]}
    k = aw.act_k("GeLU", "SiLU")
    P = [pt("GeLU", 10), pt("SiLU", 10 * k), pt("SiLU", 10 * k - 1), pt("relu2", 1)]
    n = len(P)
    ok = np.eye(n, dtype=bool)
    free = np.ones((n, n), bool)
    ex = aw.width_edges(P, ok, free, [False] * n)
    assert ex[0, 1] and not ex[0, 2] and not ex[3].any()
    assert not (ex & ok).any()


def test_recorded_agreement_minimal_k_passes():
    """Every model run (act/act_width_agreement.py) passes the strict standard at the table's k."""
    assert aw.AGREEMENT
    for (model, t, h, k, shift), (top1, ex32, ex64) in aw.AGREEMENT.items():
        if k == aw.act_k(t, h, gate_bias=shift):
            assert aw.payable(top1, ex32), (model, t, h, k, shift, top1, ex32)


def test_agreement_matches_the_data_file():
    rows = [json.loads(l) for l in open(AGREE)]
    got = {}
    for r in rows:
        got[(r["model"], r["host"], r["k"], r["shift"])] = (r["top1"], r["ex32"], r["ex64"])
    for (model, t, h, k, shift), v in aw.AGREEMENT.items():
        assert np.allclose(got[(model, h, k, shift)], v, atol=1e-4), (model, h, k, shift)
