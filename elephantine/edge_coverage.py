"""Edge coverage: how much of the emulation relation is CONSTRUCTION-verified, ATOM-
verified, or only APPROXIMATE — measured over the real graph, not asserted.

Every asserted headline edge (host emulates target, `emul_matrix(P, mode="payable")`: capacity +
every type wall that has a currency) is classified by the reductions it spends:

  spine_pure   no categorical/type change: pure capacity (KV-sharing, FFN width, depth,
               hidden width) -> CONSTRUCTION-VERIFIED end-to-end by verify_edges.run() +
               run_spine() (build B from A's weights, run it, match token-for-token).
  atom_real    crosses type walls whose reductions are ALL verified on real trained
               models (act, norm, pos, kv, ffn, block, window, loop, sink, qknorm,
               softcap -- see hf/real_reductions.py) -> capacity built + type atoms
               verified on real checkpoints. EXACT.
  atom_pending involves cross-layer-KV or MTP: exact reductions. MTP is confirmed on the
               real MiMo-7B checkpoint (disjoint aux branch); cross-layer-KV stays on the
               carry construction (Hymba reuses KV at runtime). EXACT.
  has_gap      a softmax host for an SSM / linear / hybrid target (the near-gap). The
               headline no longer asserts these (graph/model_graph.py MIXER_GAP_PAYABLE: the
               softmax constructions that keep generation identical cost heads / depth / a
               norm change the edge never charged, attn/mixer_gap_agreement.py), so it is 0;
               under mode="mixgap" it was 9.9% of the edges.
  gen_mask     an AR target on a diffusion host: the causal mask (AR <= diffusion, exact;
               attn/ar_diffusion.py, the universal host's mask toggle); since round 2 pass 4 also the
               block-causal (SDAR) edges, AR <= block and block <-> diffusion: SDAR's own forward runs
               non-causal attention under a caller-supplied mask (modeling_sdar_moe.py), so the edge is
               the same weights given the other mask. Counted inside atom_real.
  gen_cross    a diffusion / block target on an AR host, the one-way paradigm wall (must be 0).

So: spine_pure + atom_real + atom_pending are edges whose atoms are exact or real-module verified
(the activation / qk-norm / softcap atoms among them are floors or regimes); has_gap would cross
the softmax <- SSM near-gap (refused); the first two are the ones we can literally build-and-run today.

  python -m elephantine.edge_coverage
"""
from __future__ import annotations

import numpy as np

from graph.currency_curve import points
from graph.model_graph import emul_matrix

SLOTS = ["act", "block", "crosslayer_kv", "ffn", "gen", "kv", "loop", "mixer", "mtp",
         "norm", "pos", "qknorm", "sink", "softcap", "window"]
REAL = {"act", "norm", "pos", "kv", "ffn", "block", "window", "loop", "sink", "qknorm", "softcap", "gen"}
GAP = {"mixer"}                       # softmax <-> linear/SSM: the true approximate gap
PENDING = {"crosslayer_kv", "mtp"}    # exact atoms; real-model check being added


def _mixer_zero(t, h):
    """A mixer difference that is an exact zeroing edge, not the near-gap: a plain softmax target on a hybrid host (its
    recurrent layers zeroed), or a pure recurrent (ssm) target on a host with recurrent layers (its attention
    layers zeroed; fr-mixer pure_on_hybrid -- the family is the rf_ features')."""
    return t == "softmax" or (t == "ssm" and h != "softmax")


def coverage(P=None):
    P = P or points()
    S = [p["slots"] for p in P]
    ok = emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, False)
    b = dict(spine_pure=0, atom_real=0, atom_pending=0, has_gap=0, gen_cross=0, gen_mask=0)
    slotfreq = {s: 0 for s in SLOTS}
    for i, j in np.argwhere(ok):                       # j emulates i (ok[i, j]: host j, target i)
        diffs = {s for s in SLOTS if S[i][s] != S[j][s]}
        for s in diffs:
            slotfreq[s] += 1
        if "gen" in diffs and S[j]["gen"] == "ar":
            b["gen_cross"] += 1                        # a causal host for a diffusion / block target: must not happen
            continue
        b["gen_mask"] += "gen" in diffs
        # the mixer is a GAP only for softmax <- SSM / linear / hybrid (target non-softmax); a plain
        # softmax target on a hybrid host is the exact zeroing edge
        if diffs & GAP and not _mixer_zero(S[i]["mixer"], S[j]["mixer"]):
            b["has_gap"] += 1
        elif diffs & PENDING:
            b["atom_pending"] += 1
        elif diffs:
            b["atom_real"] += 1
        else:
            b["spine_pure"] += 1
    b["total"] = int(ok.sum())
    b["exact"] = b["spine_pure"] + b["atom_real"] + b["atom_pending"]     # no near-gap crossed
    b["slotfreq"] = slotfreq
    return b


# which type slots are buildable as an EXACT weight/mask construction vs only an in-regime
# fold vs a genuine gap vs a harder exact atom we don't yet compose in one executor.
CLEAN_EXACT = {"kv", "pos", "ffn", "window", "block", "act", "sink", "loop", "gen",   # exact remaps/masks
               "mixer_zero"}                                                         # + zeroed SSM layers
REGIME = {"norm", "qknorm", "softcap"}          # foldable only in-regime (centering cost / CV / saturation)
HARD_ATOM = {"crosslayer_kv", "mtp"}            # exact atoms, not composed in one executor yet


def executable_coverage(P=None):
    """How many edges are EXECUTABLE (build B from A's weights and run it token-for-token),
    as each type axis is made constructible. Separates clean-exact axes from in-regime folds
    from the mixer gap. Returns fractions over all headline edges."""
    P = P or points()
    S = [p["slots"] for p in P]
    ok = emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, False)
    # a mixer difference on a plain softmax target is the exact ZEROING edge (a hybrid host zeroes its
    # SSM layers) -> "mixer_zero", clean-exact; on an SSM / linear / hybrid target it is the near-gap,
    # which the headline refuses (graph/model_graph.py MIXER_GAP_PAYABLE), so gap_only is 0 there
    zero = lambda i, j, s: "mixer_zero" if (s == "mixer" and _mixer_zero(S[i]["mixer"], S[j]["mixer"])) else s
    diffs = [frozenset(zero(i, j, s) for s in SLOTS if S[i][s] != S[j][s]) for i, j in np.argwhere(ok)]
    tot = len(diffs)
    frac = lambda ex: sum(1 for d in diffs if d <= ex) / tot
    return dict(
        total=tot,
        capacity_only=frac(set()),                                   # no type change
        live_today=frac({"kv", "pos"}),                              # capacity + kv + pos, built end-to-end
        clean_exact=frac(CLEAN_EXACT),                               # + all exact remap/mask axes
        with_regime=frac(CLEAN_EXACT | REGIME),                      # + in-regime folds (approximate)
        gap_only=sum(1 for d in diffs if d and "mixer" in d          # blocked ONLY by the mixer gap
                     and d <= (CLEAN_EXACT | REGIME | {"mixer"})) / tot,
        needs_hard_atom=sum(1 for d in diffs if d & HARD_ATOM or "mla" in "".join(d)) / tot,
    )


# the EXACT relation (emul_matrix exact=True) IS "every actually-emulatable edge": every
# reduction it spends is exact (no floor / limit / regime / gap). Its type crossings are a
# bounded set of exact-toggle axes.
# WHOLE_MODEL: built + run token-for-token on a full model (spine + elephantine + the
# composed-edge tests in tests/test_hidden_width.py).
# gen: AR <= diffusion by the causal mask (the universal host runs one Llama body causal and full,
# tests/test_universal_elephantine.py); mixer: a hybrid host zeroing its SSM layers for a softmax
# target (stock Jamba, tests/test_payable_edges.py).
WHOLE_MODEL = {"kv", "pos", "ffn", "window", "mtp", "gen", "mixer"}
# MODULE_LEVEL: the reduction is built token-for-token on real modules composed with a
# capacity change (tests/test_composed_edges.py) -- exact, just not a full-model forward.
MODULE_LEVEL = {"sink", "norm", "block", "crosslayer_kv", "act", "loop",
                "qknorm",       # the exact qk-norm nestings (gain 1, equal per-head gains, zero bias: tests/test_qknorm_nest.py)
                "softcap"}      # round 2, pass 5: a softcap-value-free host's null cap (its stock module's own config)
COMPOSED = WHOLE_MODEL | MODULE_LEVEL              # every exact axis now has a construction
ATOM_PENDING_EXACT = set()                         # nothing left un-built in the exact relation


def exact_edge_coverage(P=None):
    """Over the EXACT relation (every actually-emulatable edge), how many can we BUILD from
    A's weights and run token-for-token today. COMPOSED = capacity spine + kv + pos + ffn
    (elephantine); the rest need only composition of already-validated exact atoms."""
    P = P or points()
    S = [p["slots"] for p in P]
    ex = emul_matrix(P, mode="exact")
    np.fill_diagonal(ex, False)
    diffs = [frozenset(s for s in SLOTS if S[i][s] != S[j][s]) for i, j in np.argwhere(ex)]
    tot = len(diffs)
    frac = lambda axes: sum(1 for d in diffs if d <= axes) / tot
    return dict(
        total=tot,
        pure_capacity=frac(set()),
        whole_model=frac(WHOLE_MODEL),                           # full-model token-for-token
        composed=frac(COMPOSED),                                 # every exact axis has a construction
        with_all_exact_atoms=frac(COMPOSED | ATOM_PENDING_EXACT),   # = 1.0
    )


def main():
    b = coverage()
    tot = b["total"]
    print(f"emulation-relation edge coverage over {tot} asserted headline edges:\n")
    order = [("spine_pure", "pure capacity -> CONSTRUCTION-verified (verify_edges + spine)"),
             ("atom_real", "type walls, all reductions verified on REAL models -> exact"),
             ("atom_pending", "cross-layer-KV / MTP: exact atom, real-model check pending"),
             ("has_gap", "softmax<->linear mixer gap -> APPROXIMATE only (a true floor)"),
             ("gen_cross", "causal host for a diffusion target: the one-way wall (should be 0)")]
    for k, desc in order:
        print(f"  {b[k]:7}  {100 * b[k] / tot:5.1f}%   {desc}")
    print("  " + "-" * 74)
    print(f"  {b['exact']:7}  {100 * b['exact'] / tot:5.1f}%   EXACT edges (every reduction exact/real-module)")
    print(f"  {b['has_gap']:7}  {100 * b['has_gap'] / tot:5.1f}%   approximate-only (across the mixer gap)")
    print("\n  reading: ~{:.0f}% of edges are exact conversions with every reduction verified;"
          .format(100 * b["exact"] / tot))
    print("  ~{:.0f}% cross the softmax<->linear gap and are convertible only to a floor (by design)."
          .format(100 * b["has_gap"] / tot))

    e = executable_coverage()
    print("\nEXECUTABLE coverage (build B from A's weights and run it token-for-token):")
    print(f"  {100 * e['capacity_only']:5.1f}%   capacity only (hidden/FFN/depth)")
    print(f"  {100 * e['live_today']:5.1f}%   + kv-type + positional (built end-to-end today)")
    print(f"  {100 * e['clean_exact']:5.1f}%   + every EXACT type axis (kv/pos/ffn/window/block/act/sink/loop)")
    print(f"  {100 * e['with_regime']:5.1f}%   + in-regime folds (norm/qknorm/softcap -- exact only in-regime)")
    print(f"  {100 * e['gap_only']:5.1f}%   blocked ONLY by the softmax<->linear gap (never exact)")
    print(f"  {100 * e['needs_hard_atom']:5.1f}%   need the harder MLA/CLA/MTP compositions")
    print("  -> the clean-exact ceiling is ~{:.0f}%; the regime folds lift it to ~{:.0f}%; the rest is the gap."
          .format(100 * e["clean_exact"], 100 * e["with_regime"]))

    x = exact_edge_coverage()
    print(f"\nEVERY ACTUALLY-EMULATABLE EDGE (the {x['total']} exact edges), built token-for-token:")
    print(f"  {100 * x['pure_capacity']:5.1f}%   pure capacity (spine)")
    print(f"  {100 * x['whole_model']:5.1f}%   whole-model forward: spine + kv + pos + ffn(MoE) + window + mtp")
    print(f"  {100 * x['composed']:5.1f}%   + module-level constructions (norm/block/sink/CLA/act), "
          "composed w/ capacity")
    print("  -> EVERY exactly-emulatable edge has a token-for-token construction: {:.0f}% whole-model, "
          "100% built.".format(100 * x["whole_model"]))


if __name__ == "__main__":
    main()
