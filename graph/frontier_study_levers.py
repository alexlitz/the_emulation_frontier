"""Frontier study, part 4: counterfactual relations with the constructions of block/frontier_constructions.py added,
WITHOUT changing graph/model_graph.py (the round-3 relation is still being edited elsewhere). Each lever returns the
payable matrix with its edges added; `combined` applies several; `check` gives the relation checks (reflexive,
transitivity violations, payable <= mixgap untouched).

Levers (this branch's wall kinds: output gates, tied embeddings):
  gate_softplus_off   a softplus output-gate host (Laguna, K2-Horizon) plays an ungated target: W_g = 0 makes the gate
                      softplus(0) = ln 2, folded into o_proj (o_proj / ln 2) -- exact (softplus_gate_off)
  gate_bias_silu_off  a silu gate WITH a bias: W_g = 0, b = 1 -> silu(1), folded -- exact (no snapshot carrier)
  gate_nests          FEAT_NEST_PENDING's gate entries: an elementwise gate with its rows tied per head is the head-wise
                      gate (out_gate_head_sub_elem); a biased gate with b = 0 is the bias-free one (out_gate_zero_bias)
                      -- exact
  tied_host           a tied host plays an untied target (tied_host_untied): host rows [e_v, u_v, p_v, c] of constant
                      norm, one extra first layer whose GLU FFN (gate on the constant channel) moves e_v into the second
                      block and erases the rest, the target in the second block, logits = u_v . x -- exact; charges
                      hidden >= 2 h_t + 3 (+2 on RMSNorm, +3 covers LayerNorm hosts too), one extra layer, that FFN >= 2 h_t + 3, a gated (or MoE) host
                      (a sandwich host switches the eraser layer's post-norms off by its eps lever)

Run:  python -m graph.frontier_study_levers                 # writes dist/frontier_study/levers.json
      python -m graph.frontier_study_levers --out PATH.json   # the same on whatever relation is checked out (e.g. a
                                                              # merge of the fr-* branches: the combined counterfactual)
"""
from __future__ import annotations

import contextlib
import copy
import json

import numpy as np

from graph import model_graph as mg
from graph.currency_curve import n_frontier, points
from graph.frontier_study import reps_strict

OUT = "dist/frontier_study"
GATES = [k for k in mg._og_feats()]


@contextlib.contextmanager
def patched_features(host_free=(), nests=()):
    """FEATURES host sides set free and FEAT_NEST entries added (target, host), restored after."""
    saved_f = {k: dict.get(mg.FEATURES, k) for k in host_free}
    saved_n = {t: dict.get(mg.FEAT_NEST, t) for t, _ in nests}
    try:
        for k in host_free:
            t, _, why = dict.get(mg.FEATURES, k)
            dict.__setitem__(mg.FEATURES, k, (t, "free", why))
        for t, h in nests:
            cur = list(dict.get(mg.FEAT_NEST, t) or ())
            dict.__setitem__(mg.FEAT_NEST, t, tuple(cur) + ((h, "exact"),))
        yield
    finally:
        for k, v in saved_f.items():
            dict.__setitem__(mg.FEATURES, k, v)
        for t, v in saved_n.items():
            if v is None:
                dict.pop(mg.FEAT_NEST, t, None)
            else:
                dict.__setitem__(mg.FEAT_NEST, t, v)


SOFTPLUS = [k for k in GATES if k.endswith("_softplus")]
BIAS_SILU = [k for k in GATES if k.endswith("_bias_silu")]
GATE_NESTS = [k for k, why in mg.FEAT_NEST_PENDING.items() if why in ("out_gate_head_sub_elem", "out_gate_zero_bias")]


def feats_of(p):
    return set(filter(None, p["slots"].get("feats", "").split("+")))


def tied_host_ok(p):
    """The host conditions of tied_host_untied: tied; a gated / MoE FFN (the eraser's constant gate); a pre-norm
    residual (not post-LN / post-only / nGPT). A sandwich host's eraser-layer post-norms are switched off by its eps
    lever (payable, a limit), so it needs the lever. Sequential or parallel blocks both work (attention is off in the
    eraser layer). LayerNorm hosts need one more pad (mean 0 as well as a fixed norm): the virtual target charges
    2 h + 3 for every host."""
    f = feats_of(p)
    sandwich = f & {"post_norm", "post_norm_attn", "ffn_norm", "onorm"}
    return ("tied_emb" in f and p["slots"]["ffn"] != "vanilla"
            and not f & {"post_only", "post_ln", "ngpt"}
            and (not sandwich or p["dims"].get("eps_lever", 0) > 0))


def virtual_tied_target(p):
    """What a tied host must play for an untied target p: the target at width 2 h + 2, one more layer, an FFN of at
    least 2 h + 2 units, tied."""
    q = copy.deepcopy(p)
    d = q["dims"]
    h = d["hidden"]
    d["hidden"] = 2 * h + 3
    for k in ("eff_layers", "param_layers", "layers"):
        d[k] = d.get(k, 0) + 1
    for k in ("ffn_total", "ffn_active"):
        d[k] = max(d.get(k, 0), 2 * h + 3)
    f = feats_of(q) | {"tied_emb"}
    q["slots"] = dict(q["slots"], feats="+".join(sorted(f)))
    return q


def lever_tied(P, ok):
    n = len(P)
    tgt = [i for i, p in enumerate(P) if "tied_emb" not in feats_of(p)]
    host = np.array([tied_host_ok(p) for p in P])
    V = [virtual_tied_target(P[i]) for i in tgt]
    big = mg.emul_matrix(P + V, mode="payable")
    add = np.zeros_like(ok)
    for r, i in enumerate(tgt):
        add[i] = big[n + r, :n] & host
    return ok | add


def payable(host_free=(), nests=(), P=None):
    P = P or points()
    with patched_features(host_free, nests):
        return mg.emul_matrix(P, mode="payable")


def check(ok):
    S = ok.astype(np.int32)
    viol = int((((S @ S) > 0) & ~ok).sum())
    return dict(frontier=n_frontier(ok), reflexive=bool(np.diag(ok).all()), violations=viol, edges=int(ok.sum()))


def run():
    P = points()
    base = mg.emul_matrix(P, mode="payable")
    reps = reps_strict(base)
    name = lambda i: P[i]["members"][0]
    res = {"base": check(base)}

    def record(tag, ok, note):
        c = check(ok)
        strict = ok & ~ok.T
        np.fill_diagonal(strict, False)
        c["leave"] = [name(i) for i in reps if strict[i].any()]
        c["new_edges"] = int((ok & ~base).sum())
        c["note"] = note
        res[tag] = c
        print(f"{tag:22s} frontier {c['frontier']}  leave {len(c['leave'])}  new edges {c['new_edges']}  "
              f"violations {c['violations']}", flush=True)
        return ok

    record("gate_softplus_off", payable(host_free=SOFTPLUS, P=P), "softplus_gate_off: W_g = 0, o_proj / ln 2")
    record("gate_bias_silu_off", payable(host_free=BIAS_SILU, P=P), "silu gate with a bias: W_g = 0, b = 1, o / silu(1)")
    record("gate_nests", payable(nests=GATE_NESTS, P=P), "out_gate_head_sub_elem, out_gate_zero_bias")
    record("gates_all", payable(host_free=SOFTPLUS + BIAS_SILU, nests=GATE_NESTS, P=P), "the three gate levers")
    record("tied_host", lever_tied(P, base), "tied_host_untied: 2 h + 2, +1 layer")
    record("mine_all", _tied_on(P, SOFTPLUS + BIAS_SILU, GATE_NESTS), "gate levers + tied host")
    import sys
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{OUT}/levers.json"
    json.dump(res, open(out, "w"), indent=1)
    return res


def _tied_on(P, host_free, nests):
    with patched_features(host_free, nests):
        base = mg.emul_matrix(P, mode="payable")
        return lever_tied(P, base)


if __name__ == "__main__":
    run()
