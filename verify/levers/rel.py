"""The headline relation emul_matrix(mode="payable") split into NAMED components, so that each rule
can be relaxed on its own (frontier levers). The AND of the components is asserted equal to
graph.model_graph.emul_matrix(P, mode="payable") (check()).

components(P, PH=None, kind_reach=None, ...) returns {name: bool[target i, host j]}. PH (same order
as P) is a HOST-side view of the points: a relaxation that changes what a host can be configured to do
(a sliding-window host with its window set to the full context) edits PH and leaves the targets alone.

Run:  python -m verify.levers.rel      # checks the decomposition against emul_matrix
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph import model_graph as mg  # noqa: E402
from graph.frontier import JSON, load, reps_of  # noqa: E402

CACHE = os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "frontier_levers_points.pkl")


def points():
    """(P, cfg): the deduplicated headline points and {model name: raw config}."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    M = load()
    P = mg._dedup_points(M)
    want = {n for p in P for n in p["members"]}
    cfg = {n: c for n, c in mg.iter_json_configs(JSON) if n in want}
    with open(CACHE, "wb") as f:
        pickle.dump((P, cfg), f)
    return P, cfg


def lm(c):
    return mg._lm_config(c)


def _feat_rule(x):
    return mg.FEATURES.get(x) or (("wall", "free") if x.startswith("rf_") else ("free", "free"))


def components(P, PH=None, kind_reach=None, window_all_layers=False, act_floor_max=None, moe_factor=None):
    """Named components of the payable relation, [i, j] = host j emulates target i.
    kind_reach: override of KIND_REACH for kind_match; window_all_layers: a sliding-window host's local
    layers count as full ones for a plain full target (the ZERO_EDGES window charge removed)."""
    PH = PH if PH is not None else P
    n = len(P)
    C = {}
    MF = mg.MOE_DEPTH_FACTOR if moe_factor is None else moe_factor

    def dimT(k, fb=None):
        return np.array([p["dims"].get(k, p["dims"].get(fb, 0) if fb else 0) for p in P], dtype=np.int64)

    def dimH(k, fb=None):
        return np.array([p["dims"].get(k, p["dims"].get(fb, 0) if fb else 0) for p in PH], dtype=np.int64)

    saT = lambda s: np.array([p["slots"][s] for p in P])
    saH = lambda s: np.array([p["slots"][s] for p in PH])

    def cat(slot, strict, reach_fn=None):
        a, b = saT(slot), saH(slot)
        ua, ub = sorted(set(a.tolist())), sorted(set(b.tolist()))
        f = reach_fn or (lambda x, y: mg.slot_reaches(slot, x, y, strict))
        cm = np.array([[f(x, y) for y in ub] for x in ua])
        return cm[np.ix_([ua.index(v) for v in a], [ub.index(v) for v in b])]

    # --- categorical slots
    for slot in mg.CAT_SLOTS:
        if slot == "act" and act_floor_max is not None:
            order = mg._act_order(act_floor_max)
            reach = mg.one_hop(order)
            C["cat:act"] = cat("act", False, lambda x, y: any(bb in reach.get(x, {x}) for bb in mg.branches(y)))
        else:
            C[f"cat:{slot}"] = cat(slot, mg._regime_strict(slot))
    # --- FEATURES, one component per feature and side
    fT = [set(filter(None, p["slots"].get("feats", "").split("+"))) for p in P]
    fH = [set(filter(None, p["slots"].get("feats", "").split("+"))) for p in PH]
    assert not any(p["slots"].get("feats", "").startswith("*") for p in PH)
    allf = sorted(set().union(*fT, *fH))
    for x in allf:
        t, h = np.array([x in s for s in fT]), np.array([x in s for s in fH])
        rt, rh = _feat_rule(x)[:2]
        name = "rf" if x.startswith("rf_") else x
        # host-side switches (verify/levers/relax.py): a host that can turn x ON (sw_on) or OFF (sw_off)
        on = np.array([x in p.get("sw_on", ()) for p in PH])
        off = np.array([x in p.get("sw_off", ()) for p in PH])
        if rt not in ("free", "limit"):
            m = ~(t[:, None] & ~(h | on)[None, :])
            C[f"feat:{name}:target"] = C.get(f"feat:{name}:target", True) & m
        if rh not in ("free", "limit"):
            m = ~((h & ~off)[None, :] & ~t[:, None])
            C[f"feat:{name}:host"] = C.get(f"feat:{name}:host", True) & m
    # --- mixer / cla categorical for non-plain targets
    mixT, claT = saT("mixer"), saT("crosslayer_kv")
    mixH, claH, winH = saH("mixer"), saH("crosslayer_kv"), saH("window")
    plainT = (mixT == "softmax") & (claT == "no")
    plainH = (mixH == "softmax") & (claH == "no")
    no_gap = not mg.MIXER_GAP_PAYABLE
    cla_strict = not mg.CLA_CARRY_PAYABLE
    c_mix = cat("mixer", no_gap)
    c_cla = cat("crosslayer_kv", cla_strict)
    C["nonplain:mixer"] = np.where(plainT[:, None], True, c_mix)
    C["nonplain:cla"] = np.where(plainT[:, None], True, c_cla)
    # --- the rot gate
    posT, posH = saT("pos"), saH("pos")
    gated = np.isin(posT, list(mg.ROT_TARGETS))[:, None] & np.isin(posH, list(mg.ROT_HOSTS))[None, :]
    need = [tuple(p["dims"].get("rope_sched", ())) for p in P]
    capH = [tuple(p["dims"].get("rope_cap", p["dims"].get("rope_sched", ()))) for p in PH]
    un, uc = sorted(set(need)), sorted(set(capH))
    fm = np.array([[mg.rope_fits(a, b) for b in uc] for a in un])
    C["rot_gate"] = ~gated | fm[np.ix_([un.index(u) for u in need], [uc.index(u) for u in capH])]
    # --- K = V
    C["keqv_caps"] = np.ones((n, n), bool)
    for k in ("kv_cap", "q_cap", "qheads", "head_dim"):
        C["keqv_caps"] &= dimH(k + "_u", k)[None, :] >= dimT(k + "_u", k)[:, None]
    # --- MLA
    kvT, kvH = saT("kv"), saH("kv")
    mlaT = np.array(["MLA" in mg.branches(v) for v in kvT])
    mlaH = np.array(["MLA" in mg.branches(v) for v in kvH])
    plain_kvH = np.array([any(b != "MLA" for b in mg.branches(v)) for v in kvH])
    C["mla_wall"] = np.where(mlaT[:, None], mlaH[None, :], plain_kvH[None, :])
    # the eps lever on an MLA host (relax.py, mla_eps): its latent norms run in the linear regime, so it is plain
    # low-rank attention with ONE shared rotary key -- a plain target whose keys need no per-KV-head rotation
    # (NoPE, or one KV head) fits if the latent holds the target's K and V (rank <= 2 kv_cap)
    epsH = np.array([bool(p.get("mla_eps")) for p in PH])
    if epsH.any():
        posT0 = saT("pos")
        one_rot = (posT0 == "nope") | (dimT("kv_heads") == 1)
        okp = (mlaH & epsH)[None, :] & ~mlaT[:, None] & one_rot[:, None] & (
            dimH("kv_latent")[None, :] >= np.minimum(dimT("hidden"), 2 * dimT("kv_cap"))[:, None])
        C["mla_wall"] = C["mla_wall"] | okp
    qlT, klT, qlH, klH, qsH = dimT("q_latent"), dimT("kv_latent"), dimH("q_latent"), dimH("kv_latent"), dimH("mla_qswitch")
    C["mla_latents"] = ~(mlaT[:, None] & mlaH[None, :]) | (
        (((qlH[None, :] > 0) == (qlT[:, None] > 0)) | (qsH[None, :] > 0))
        & (qlH[None, :] >= qlT[:, None]) & (klH[None, :] >= klT[:, None]))
    if epsH.any():      # host q-latent norm linearised: a target without a q latent fits if the latent holds its q rank
        okq = (mlaT[:, None] & mlaH[None, :] & epsH[None, :] & (qlT[:, None] == 0) & (qlH[None, :] > 0)
               & (qlH[None, :] >= np.minimum(dimT("hidden"), dimT("q_cap"))[:, None]) & (klH[None, :] >= klT[:, None]))
        C["mla_latents"] = C["mla_latents"] | okq
    # --- capacities
    kvt, kvh = dimT("kv_cap"), dimH("kv_cap")
    pen = np.where(mlaT[:, None] & ~mlaH[None, :], mg.MLA_PENALTY, 1)
    C["kv_cap"] = kvh[None, :] >= kvt[:, None]
    C["mla_penalty"] = (pen == 1) | (kvh[None, :] >= kvt[:, None] * pen)
    C["q_cap"] = dimH("q_cap")[None, :] >= dimT("q_cap")[:, None]
    kht, qgt, khh, qgh = dimT("kv_heads"), dimT("q_per_kv"), dimH("kv_heads"), dimH("q_per_kv")
    have_t, have_h = (kht > 0) & (qgt > 0), (khh > 0) & (qgh > 0)
    needh = kht[:, None] * -(-qgt[:, None] // np.maximum(qgh[None, :], 1))
    C["kv_heads"] = ~(have_t[:, None] & have_h[None, :]) | (khh[None, :] >= needh)
    C["qheads"] = dimH("qheads")[None, :] >= dimT("qheads")[:, None]
    C["head_dim"] = dimH("head_dim")[None, :] >= dimT("head_dim")[:, None]
    C["ffn_total"] = dimH("ffn_total")[None, :] >= dimT("ffn_total")[:, None]
    # a MoE host whose top-k is a free config value (relax.py, topk_free) raises k to its expert count
    actH = np.where([bool(p.get("topk_free")) for p in PH], dimH("ffn_total"), dimH("ffn_active"))
    C["ffn_active"] = actH[None, :] >= dimT("ffn_active")[:, None]
    # --- depth (with the ZERO_EDGES usable layers)
    elT, plT = dimT("eff_layers"), dimT("param_layers")
    elH, plH = dimH("eff_layers"), dimH("param_layers")
    winT = saT("window")
    need_full = (winT == "full")[:, None]
    h_local = (winH == "local")[None, :]
    fe, fp = dimH("full_eff", "eff_layers"), dimH("full_param", "param_layers")
    ae, ap = dimH("attn_eff", "eff_layers"), dimH("attn_param", "param_layers")
    if window_all_layers:
        all_layers = np.broadcast_to(plainH[None, :], (n, n))
        zero_e = np.where(all_layers, elH[None, :], np.where(need_full & ~h_local, fe[None, :], ae[None, :]))
        zero_p = np.where(all_layers, plH[None, :], np.where(need_full & ~h_local, fp[None, :], ap[None, :]))
    else:
        all_layers = plainH[None, :] & ~(need_full & h_local)
        zero_e = np.where(all_layers, elH[None, :], np.where(need_full, fe[None, :], ae[None, :]))
        zero_p = np.where(all_layers, plH[None, :], np.where(need_full, fp[None, :], ap[None, :]))
    usable_e = np.where(plainT[:, None], zero_e, elH[None, :])
    usable_p = np.where(plainT[:, None], zero_p, plH[None, :])
    ffnT, ffnH = saT("ffn"), saH("ffn")
    frT = np.array([mg.FFN_RANK.get(f, max(mg.FFN_RANK.values())) for f in ffnT])
    frH = np.array([mg.FFN_RANK.get(f, max(mg.FFN_RANK.values())) for f in ffnH])
    blT, blH = saT("block"), saH("block")
    h_seq = np.array(["sequential" in mg.branches(v) for v in blH])      # a branch set: a config switch
    h_par = np.array(["parallel" in mg.branches(v) for v in blH])
    par_emu_seq = ~h_seq[None, :] & (blT[:, None] == "sequential")
    seq_emu_par = ~h_par[None, :] & (blT[:, None] == "parallel")
    depth2 = (frH[None, :] < frT[:, None]) | par_emu_seq
    C["depth_eff"] = usable_e >= np.where(depth2, MF * elT[:, None], elT[:, None])
    C["depth_param"] = usable_p >= plT[:, None]
    C["par_carry"] = ~seq_emu_par
    dense_h = (frH <= mg.FFN_RANK["glu"])[None, :] & (frT >= mg.FFN_RANK["moe"])[:, None]
    C["moe_dense_top1"] = ~dense_h | (dimT("moe_unit_top1") > 0)[:, None]
    hdT, hdH = dimT("head_dim"), dimH("head_dim")
    al = (posT == "alibi")[:, None] & np.isin(posH, ("rope", "partial_rope"))[None, :]
    C["alibi_rope"] = ~al | ((saT("gen") == "ar")[:, None] & (hdH[None, :] >= hdT[:, None] + 2))
    # --- per-layer kinds (Hall)
    C["kinds_eff"] = kind_match2(P, PH, "eff", np.where(depth2, MF, 1), kind_reach)
    C["kinds_param"] = kind_match2(P, PH, "param", 1, kind_reach)
    # --- width
    width = np.where(seq_emu_par, 2, 1)
    normT, normH = saT("norm"), saH("norm")
    C["hidden"] = dimH("hidden")[None, :] >= width * dimT("hidden")[:, None]
    mirror = (normT[:, None] == "RMSNorm") & (normH[None, :] == "LayerNorm")
    C["norm_mirror"] = ~mirror | (dimH("hidden")[None, :] >= width * dimT("hidden")[:, None] * mg.NORM_WIDTH_FACTOR)
    return C


def kind_match2(P, PH, which, factor, reach=None):
    """graph.model_graph.kind_match with separate target / host point views."""
    reach = reach or mg.KIND_REACH
    hall = mg._HALL if reach is mg.KIND_REACH else mg._subsets_hall(reach)
    cT = {k: np.array([mg._kind_count(p, k, which) for p in P], dtype=np.int64) for k in mg.LAYER_KINDS}
    cH = {k: np.array([mg._kind_count(p, k, which) for p in PH], dtype=np.int64) for k in mg.LAYER_KINDS}
    ok = np.ones((len(P), len(P)), bool)
    for X, N in hall:
        demand = sum(cT[t] for t in X)
        supply = sum(cH[h] for h in N)
        ok &= supply[None, :] >= demand[:, None] * factor
    return ok


def AND(C, drop=(), replace=None):
    replace = replace or {}
    ok = None
    for k, v in C.items():
        if k in drop:
            continue
        v = replace.get(k, v)
        ok = v.copy() if ok is None else ok & v
    return ok


def frontier_set(ok, P):
    """The frontier representatives (indices) of relation ok."""
    return reps_of(ok, P)


def frontier_count(ok, P):
    return len(frontier_set(ok, P))


def nmodels(P):
    return np.array([len(p["members"]) for p in P])


def left(base_reps, ok):
    """Baseline frontier reps that are no longer maximal under ok (strictly dominated by some point, or
    merged into a mutual class whose kept representative is another point)."""
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)
    return [i for i in base_reps if strict[i].any()]


def short(P, i):
    return mg.short(P[i]["members"][0])


def check():
    P, _ = points()
    C = components(P)
    ok = AND(C)
    ref = mg.emul_matrix(P, mode="payable")
    assert (ok == ref).all(), f"decomposition differs on {(ok != ref).sum()} cells"
    print(f"decomposition == emul_matrix(payable) on {len(P)} points; {len(C)} components; "
          f"frontier {frontier_count(ok, P)}")
    return P, C


if __name__ == "__main__":
    check()
