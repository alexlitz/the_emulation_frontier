"""The frontier is a FUNCTION of the currency budget, not a fixed set.

Most type walls have a payable reverse edge (untie KV -> memory, dense -> MoE -> params, a hybrid
host zeroing its SSM layers -> depth, an activation fold or limit, ...), so "incomparable" usually
means "expensive or approximate one way". What stays is (a) walls with NO currency in the slot
orders (e.g. a RoPE host for a learned-abs target, a SiLU host for a GeLU / mish / xIELU / situ /
relu2 / poly_norm target -- activation floors above the calibrated ACT_FLOOR_MAX are walls; the list
is graph/model_graph.py blocked_walls()), (b) capacity, and (c) the one-way AR <= diffusion paradigm.

So the frontier shrinks as you allow more currency:

  exact             every crossed edge exact (folds, masks, zeroing, the mirror norm, AR <= diffusion)
  payable           + floors / limits / regimes: every edge that HAS a currency is paid -- THE
                    HEADLINE frontier (graph/frontier.py, FRONTIER.md). The softmax <- SSM / linear /
                    hybrid near-gap is NOT among them: measured, its softmax constructions pass only at
                    heads / depth / a norm change the edge never charged (attn/mixer_gap_agreement.py).
                    Nor is the cross-layer KV (CLA) carry: its ~6e-2 pre-norm floor (block/prenorm_carry.py)
                    is two orders above ACT_FLOOR_MAX (CLA_CARRY_PAYABLE = False; 101 -> 106). A RoPE
                    target on a partial-RoPE host is paid where the host rotates enough dims per head
                    (the rot gate, hf/partial_rope_host.py; 106 -> 104 with the rotary-factor fixes).
                    The edge-case audit's walls (per-layer kinds, K = V, MLA, the measured norm / bias /
                    sink walls, the forward-pass FEATURES, KV heads by count) with the rot gate on
                    full-RoPE hosts too: 104 -> 218
  mixgap            + that near-gap and the CLA carry granted free (the pre-measurement headline)
  capacity          the secondary "if every type wall were free": only capacity and the one-way
                    gen wall remain -- maximal by SIZE alone
  elephantine       a bigger model (the join, graph/minimal_emulator.py) dominates them all

Run:  python -m graph.currency_curve
"""
from __future__ import annotations

import numpy as np

from graph.model_graph import (_dedup_points, _junky, classify, dims_of,
                               emul_matrix, iter_json_configs)


def points(json_path="graph/hub_10k.json"):
    M = []
    for n, c in iter_json_configs(json_path):
        if _junky(n):
            continue
        cl = classify(c)
        d = dims_of(c) if cl else None
        if cl and d["hidden"] and d["layers"]:
            M.append({"name": n, "slots": cl, "dims": d})
    return _dedup_points(M)


def n_frontier(ok):
    """Number of frontier equivalence classes under the emulation matrix ok."""
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)
    front = np.where(~strict.any(1))[0]
    reps, used = [], set()
    for i in front:
        if i in used:
            continue
        eq = [j for j in front if ok[i, j] and ok[j, i]]
        used.update(eq)
        reps.append(i)
    return len(reps)


def curve(P=None):
    """The frontier-vs-budget curve as a dict of {budget level: frontier size}. `payable` is the
    HEADLINE; `capacity` the secondary 'if every type wall were free'; exact is stricter, mixgap
    grants the unpriced softmax <- SSM near-gap and the CLA carry."""
    P = P or points()
    out = {m: n_frontier(emul_matrix(P, mode=m)) for m in ("exact", "payable", "mixgap", "capacity")}
    out["elephantine"] = 1        # the join dominates every point (asserted in tests/test_graph_rules.py)
    return out


def main():
    c = curve()
    labels = [
        ("exact", "exact only (no floor / limit / regime / gap)"),
        ("payable", "+ floors, limits, regimes: every wall with a currency paid  [HEADLINE]"),
        ("mixgap", "+ the softmax <- SSM / linear / hybrid near-gap and the CLA carry granted free (measured not payable)"),
        ("capacity", "if every type wall were free: capacity + the one-way gen wall  [secondary]"),
        ("elephantine", "+ the elephantine (bigger than all)"),
    ]
    print("frontier as a function of the currency budget:")
    for k, desc in labels:
        print(f"  {c[k]:>3}   {desc}")


if __name__ == "__main__":
    main()
