"""Frontier study: what keeps each payable-frontier model on the frontier (the walls that stop every would-be
dominating host), counterfactual leave-one-in on each wall, and presentation numbers (families, downloads, base
shapes). Read-only on the relation: everything here is computed from emul_matrix(mode="payable", explain=...).

Every refusing check of a pair (emul_matrix's explain masks) is refined into ITEMS: a categorical slot check into its
value pair ("slot:qknorm head<-no"), the feature check into each offending feature ("feat:bias_qkv<-none" = the target
has it and the host neither has nor nests it; "feat:none<-post_norm" = the host has it and cannot switch it off), the
mixer / cla check into its slot pairs. The refinement is exact: a pair is refused iff it has at least one item
(asserted in build()). Each item has a KIND (bias, norm placement, qk-norm, out gate, router / MoE, activation,
positions, mixer, capacity, ...).

Counterfactual "wall W removed": a pair whose items are all in W passes. The frontier under that relation is compared
to the headline (n_frontier, and which frontier classes acquire a strict dominator).

Run:  python -m graph.frontier_study            # writes dist/frontier_study/*.json and the tables

(Release: taken from branch frontier-study, e1e7611, as the itemiser graph/post_numbers.py's floor / relaxed numbers
use -- moe_reasons follows the release's moe_why (fr-moe's sunk extras; fr-cost's counterfactual hooks are empty),
and the feature items read the host switch-offs emul_matrix reads: the eps lever, fr-resnorm's _fr_host_off and
fr-positions' NoPE-layer host. None of the branch's levers or constructions are part of the relation.)
"""
from __future__ import annotations

import collections
import json
import os
import re

import numpy as np

from graph import model_graph as mg
from graph.currency_curve import points

OUT = "dist/frontier_study"

CAPACITY = {"hidden", "kv_cap", "q_cap", "qheads", "head_dim", "ffn width", "depth (eff)", "depth (param)",
            "ffn per-layer widths"}

# item -> kind (first matching rule wins)
KIND_RULES = [
    (r"^(hidden|kv_cap|q_cap|qheads|head_dim|ffn width|depth \(eff\)|depth \(param\)|ffn per-layer widths)$",
     "capacity"),
    (r"^(kv heads|K=V|o_proj groups|layer kinds)$", "attention layout (heads / K=V / layer kinds)"),
    (r"^(kv MLA|q latent norm|feat:(none<-)?mla_|feat:mla_)", "MLA"),
    (r"^(feat:(none<-)?bias_|feat:bias_|ffn vanilla<-gated up bias|bias shift host)", "bias"),
    (r"^slot:qknorm|^feat:(none<-)?vnorm|^feat:vnorm", "qk-norm / v-norm"),
    (r"^feat:(none<-)?(post_norm|post_only|post_ln|ffn_norm|onorm|loop_bnorm|moe_branch_norm|expert_norm)"
     r"|^feat:(post_norm|post_only|post_ln|ffn_norm|onorm|loop_bnorm|moe_branch_norm|expert_norm)"
     r"|^post_norm_attn lever|^slot:norm|^embedding_norm|^grouped_rmsnorm|^gated_norm", "norm kind / placement"),
    (r"^feat:(none<-)?(out_gate|shared_gate)|^feat:(out_gate|shared_gate)", "output / shared-expert gate"),
    (r"^moe:(experts|expert width|shared expert|expert latent|layers|residual MLP|MoE<-dense: (hidden|ffn))",
     "MoE capacity (experts / widths / layers)"),
    (r"^moe:top-k", "MoE top-k multiple"),
    (r"^feat:(none<-)?(router|hash_route|scmoe|moe_|block_routing|par_moe_res)"
     r"|^feat:(router|hash_route|scmoe|moe_|block_routing|par_moe_res)|^ffn MoE|^moe:", "router / MoE"),
    (r"^slot:act|^feat:(none<-)?act_clip|^feat:act_clip|^swigluoai", "activation"),
    (r"^slot:pos|^pos |^feat:(none<-)?(nope_layers|q_temp|fope|rope_warp)|^feat:(nope_layers|q_temp|fope|rope_warp)",
     "positions"),
    (r"^mixer|^cla |^recurrent dims|^feat:(none<-)?rf_|^feat:rf_|^feat:(none<-)?zamba|^feat:zamba", "mixer / recurrence"),
    (r"^feat:(none<-)?tied_emb|^feat:tied_emb|^factorized embedding", "tied / factorized embeddings"),
    (r"^ffn gated<-vanilla|^ffn hadamard", "FFN form (GLU / vanilla)"),
    (r"^block parallel", "parallel block"),
    (r"^slot:gen", "paradigm (gen)"),
    (r"^slot:sink|^slot:softcap", "sink / softcap"),
    (r"^loop|^feat:(none<-)?(loop_|hrm_|early_exit|rd_controller)|^feat:(loop_|hrm_|early_exit|rd_controller)",
     "loops"),
    (r"^feat:(none<-)?(hyper_conn|hc_|engram|value_emb|attn_res|ple|backout|res_|ngpt|byte_patch|selfcond|canon)"
     r"|^feat:(hyper_conn|hc_|engram|value_emb|attn_res|ple|backout|res_|ngpt|byte_patch|selfcond|canon)"
     r"|^(attn_res partition|engram geometry|value_emb kind|canon cover)", "residual streams / extra inputs"),
    (r"^feat:", "other attention / forward features"),
]


def kind_of(item):
    for pat, k in KIND_RULES:
        if re.search(pat, item):
            return k
    return "other"


def _feat_items(t, h, off, strict=False):
    """The offending features of target feature string t on host string h (the feats_ok logic, item by item)."""
    ok_rule = {"free"} if strict else {"free", "limit"}
    rule = lambda x: mg.FEATURES.get(x) or (("wall", "free") if x.startswith("rf_") or x.startswith("engram_")
                                            else ("free", "free"))
    T = set(filter(None, (t or "").split("+")))
    H = set(filter(None, (h or "").split("+")))
    lack = T - H
    nest = {x: [p for p in mg._feat_parents(x, strict) if p in H] for x in lack}
    spent = {p for ps in nest.values() for p in ps}
    out = [f"feat:{x}<-none" for x in sorted(lack) if rule(x)[0] not in ok_rule and not nest[x]]
    out += [f"feat:none<-{x}" for x in sorted(H - T) if rule(x)[1] not in ok_rule and x not in off and x not in spent]
    return out


def _one(n, i, j):
    m = np.zeros((n, n), bool)
    m[i, j] = True
    return m


def _moe_cat(w):
    """A moe_why reason with its numbers dropped (the category)."""
    w = re.sub(r"-?\d+(\.\d+)?", "N", w)
    w = re.sub(r"(N[ x/+]*)+", "N ", w).strip()
    return w


def moe_reasons(t, h, strict=False):
    """Every failing MoE routing condition of moe_why (graph/model_graph.py), not only the first: moe_why's checks in
    order, each recorded instead of returned. Where top-k is not a multiple, the later checks use m = floor(k_h / k_t)
    (the copies part of a remainder construction). The first reason equals moe_why's (asserted in build())."""
    (st, ht, ft), (sh, hh, fh) = t, h
    if not st or not sh:
        return []
    assert not mg._r3("moe_null_experts"), "moe_reasons: the null-expert plans are not itemised"
    T, H = mg._moe_fields(st), mg._moe_fields(sh)
    Et, Eh = T["E"], H["E"]
    out = []
    if not Et and not Eh:
        return out
    if Et and not Eh:
        if not (T["k"] == 1 and T["renorm"] == "R"):
            out.append(f"MoE<-dense: top-{T['k']} renorm {T['renorm']} is not a top-1 unit gate")
        form = mg._r3_moe_forms_dense_host(T)
        if form:
            out.append(f"MoE<-dense: {form}")
        if hh < ht + Et:
            out.append(f"MoE<-dense: hidden {hh} < {ht} + {Et} select channels")
        if fh < ft + 2 * Et * (Et - 1):
            out.append(f"MoE<-dense: ffn {fh} < {ft} + {2 * Et * (Et - 1)} erase units")
        return out
    if T["branch"] != H["branch"]:
        out.append("branch post-norms: the target's branch-normed sum needs a branch-normed host" if T["branch"] else
                   "branch post-norms: a host branch is double-normed (needs theorem double_rmsnorm_fold)")
    if H["par_res"] and Et and not T["par_res"]:
        out.append("a parallel MoE + residual-MLP host plays a sequential MoE layer only at 2x depth (not plumbed)")
    rank_ok = (not H["latent"] or H["latent"] >= ht) and H["latent_norm"] in ("", "sum?")
    capd = (H["k"] * H["ew"] if rank_ok else 0) + H["shared"]
    if H["branch"]:
        capd = max(H["k"] * H["ew"] if rank_ok else 0, H["shared"])
    if H["par_res"]:
        capd = H["par_res"]
    dense_slots = lambda spare: ((H["n_dense"] if H["dense_w"] >= T["dense_w"] else 0)
                                 + (spare if capd >= T["dense_w"] else 0))
    if not Et:
        if T["n_dense"] > dense_slots(H["n_moe"]):
            out.append(f"layers: {T['n_dense']} dense target layers, host dense / MoE-as-dense layers too few or narrow")
        return out
    if H["k"] % T["k"]:
        out.append(f"top-k {H['k']} not a multiple of {T['k']}")
    m = max(1, H["k"] // T["k"])
    if Eh < m * Et:
        out.append(f"experts {Eh} < {m} x {Et}")
    if m * H["ew"] < T["ew"]:
        out.append(f"expert width {m} x {H['ew']} < {T['ew']}")
    if H["shared"] < T["shared"]:
        out.append(f"shared expert {H['shared']} < {T['shared']}")
    if (H["latent"] or hh) < (T["latent"] or ht):
        out.append(f"expert latent {H['latent'] or hh} < {T['latent'] or ht}")
    if H["latent_norm"] not in mg.LATENT_NORM_PLAYS.get(T["latent_norm"], (T["latent_norm"],)):
        out.append(f"latent norm {T['latent_norm'] or 'none'}<-{H['latent_norm'] or 'none'}")
    if H["par_res"] < T["par_res"]:
        out.append(f"residual MLP {H['par_res']} < {T['par_res']}")
    if H["n_moe"] < T["n_moe"]:
        out.append(f"layers: host MoE layers {H['n_moe']} < {T['n_moe']}")
    elif T["n_dense"] > dense_slots(H["n_moe"] - T["n_moe"]):
        out.append(f"layers: {T['n_dense']} dense target layers, host dense / spare MoE layers too few or narrow")
    if T["score"] not in H["score_cap"]:
        out.append(f"router score {T['score']}<-{'/'.join(H['score_cap'])}")
    if T["renorm"] not in H["renorm_cap"]:
        out.append(f"renorm {T['renorm']}<-{'/'.join(H['renorm_cap'])}")
    if T["score"] == "sparsemixer" and m != 1 and mg._r3("spot_phimoe_dup"):
        out.append(f"SparseMixer with {m} copies (its second pick masks one index)")
    if (T["ident"] or H["ident"]) and H["ident"] != m * T["ident"]:
        out.append(f"identity slots {H['ident']} != {m} x {T['ident']}")
    if T["blk"] and not (H["blk"] == T["blk"] and H["blk_cap"] == T["blk_cap"] and m == 1 and Eh == Et):
        out.append(f"block routing {T['blk']}/{T['blk_cap']}<-{H['blk'] or 'none'}/{H['blk_cap']} (m {m}, E {Eh}/{Et})")
    if T["renorm"] == "P" and T["score"] == "softmax":
        if not (H["renorm"] == "P" and H["score"] == "softmax"
                and (Eh == m * Et or (mg._r3("moe_plain_extras_sink") and not strict and hh > ht))):
            out.append("unrenormalised softmax: needs a host whose own router is too, with exactly m x the experts")
    elif Eh > m * Et and not (H["bias"] or hh > ht):
        out.append("extra experts need a selection bias or a free hidden dim to score below the copies")
    if T["n_group"] and T["group_cap"] != H["group_cap"]:
        out.append(f"group routing {T['group_cap']}<-{H['group_cap'] or 'none'}")
    if T["n_group"] and str(T["group_cap"]).startswith("top2sum") and m != 1:
        out.append(f"top-2-sum group score with {m} copies")
    if T["n_group"] and T["group_cap"] in ("top2sum_zero", "per_group") and Eh != Et:
        out.append(f"{T['group_cap']} groups: host experts {Eh} != {Et}")
    if T["n_group"] and Eh % (m * T["n_group"]):
        out.append(f"experts {Eh} not in {m} x {T['n_group']} equal groups")
    return out


def build(P=None, mode="payable"):
    """P, ok, cap, and the item masks {item: [i, j] refused}."""
    P = P or points()
    why = {}
    ok = mg.emul_matrix(P, mode=mode, explain=why)
    cap = mg.emul_matrix(P, mode="capacity")
    n = len(P)
    sa = lambda s: np.array([p["slots"][s] for p in P])
    items = {}

    def put(name, m):
        if m.any():
            items[name] = items.get(name, np.zeros((n, n), bool)) | m

    for nm, m in why.items():
        if not m.any():
            continue
        if nm.startswith("slot:"):
            s = nm[5:]
            v = sa(s)
            for a in sorted(set(v[np.where(m.any(1))[0]])):
                for b in sorted(set(v[np.where(m[v == a].any(0))[0]])):
                    put(f"slot:{s} {a}<-{b}", m & (v == a)[:, None] & (v == b)[None, :])
        elif nm == "mixer/cla":
            for s, lab in (("mixer", "mixer"), ("crosslayer_kv", "cla")):
                v = sa(s)
                diff = v[:, None] != v[None, :]
                for a in sorted(set(v)):
                    for b in sorted(set(v)):
                        if a != b:
                            put(f"{lab} {a}<-{b}", m & diff & (v == a)[:, None] & (v == b)[None, :])
            # a mixer/cla refusal with equal mixer and cla values cannot happen; checked below
        elif nm == "feature":
            fts = np.array([p["slots"].get("feats", "") for p in P])
            eps_h = np.array([p["dims"].get("eps_lever", 0) > 0 for p in P]) & (mode not in ("capacity", "exact"))
            offs = [frozenset(mg.EPS_LEVER_FEATS & set(f.split("+"))) if e and mg._p5("eps_feats") else frozenset()
                    for f, e in zip(fts, eps_h)]
            # (release: + fr-resnorm's host switch-offs, as emul_matrix)
            sc = [bool(p["dims"].get("sc_norm_off", 0)) for p in P]
            offs = [o | mg._fr_host_off(f, c, mode != "exact") for o, f, c in zip(offs, fts, sc)]
            hk = list(zip(fts.tolist(), offs))
            uf = sorted(set(fts.tolist()))
            uh = sorted(set(hk), key=lambda x: (x[0], sorted(x[1])))
            fi = np.array([uf.index(x) for x in fts])
            hi = np.array([uh.index(x) for x in hk])
            for a_i, a in enumerate(uf):
                for b_i, (b, off) in enumerate(uh):
                    # (release: fr-positions' NoPE-layer host drops its NoPE-layer features for a target without them)
                    off2 = off | (mg.NOPE_HOST_OFF if (mg._r3("fp_nope_layer_host") and "nope_layers" not in a.split("+"))
                                  else frozenset())
                    its = _feat_items(a, b, off2, mode == "exact")
                    if not its:
                        continue
                    pm = m & (fi == a_i)[:, None] & (hi == b_i)[None, :]
                    for it in its:
                        put(it, pm)
        elif nm in ("ffn MoE routing", "ffn MoE<-dense"):
            key = [(p["dims"].get("moe_sig"), p["dims"]["hidden"], p["dims"]["ffn_total"]) for p in P]
            memo, acc = {}, collections.defaultdict(list)
            for i, j in zip(*np.where(m)):
                kk = (key[i], key[j])
                if kk not in memo:
                    w = mg.moe_why(*kk, strict=mode == "exact") if key[i][0] and key[j][0] else None
                    rs = moe_reasons(*kk, strict=mode == "exact") if key[i][0] and key[j][0] else []
                    assert bool(w) == bool(rs) and (not w or w == rs[0]), (kk, w, rs)
                    memo[kk] = [_moe_cat(r) for r in rs] or (["MoE<-dense (other)"] if nm == "ffn MoE<-dense"
                                                             else ["at the activation-fit width"])
                for c in memo[kk]:
                    acc[c].append((i, j))
            for c, ij in acc.items():
                mm = np.zeros((n, n), bool)
                a = np.array(ij)
                mm[a[:, 0], a[:, 1]] = True
                put(f"moe:{c}", mm)
        else:
            put(nm, m)
    # emul_matrix evaluates the MoE routing check (_moe_ok) only on pairs every earlier check passed, so its explain
    # mask is silent on pairs already refused. Fill it in: every refused pair with a MoE side and no MoE item gets its
    # moe_reasons (at the target's own width: the activation-width multiple k is not known for a refused pair)
    key = [(p["dims"].get("moe_sig"), p["dims"]["hidden"], p["dims"]["ffn_total"]) for p in P]
    moe_side = np.array([bool(k[0]) and bool(mg._moe_fields(k[0])["E"]) for k in key])
    have = np.zeros((n, n), bool)
    for k_, m_ in items.items():
        if k_.startswith("moe:"):
            have |= m_
    memo, acc = {}, collections.defaultdict(list)
    for i, j in zip(*np.where(~ok & ~have & (moe_side[:, None] | moe_side[None, :]))):
        if not (key[i][0] and key[j][0]):
            continue
        kk = (key[i], key[j])
        if kk not in memo:
            memo[kk] = [_moe_cat(r) for r in moe_reasons(*kk, strict=mode == "exact")]
        for c in memo[kk]:
            acc[c].append((i, j))
    for c, ij in acc.items():
        mm = np.zeros((n, n), bool)
        a = np.array(ij)
        mm[a[:, 0], a[:, 1]] = True
        put(f"moe:{c}", mm)
    cnt = sum(m.astype(np.int16) for m in items.values())
    assert np.array_equal(ok, cnt == 0), "item refinement differs from emul_matrix"
    return P, ok, cap, items


def reps_strict(ok):
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)
    front = np.where(~strict.any(1))[0]
    reps, used = [], set()
    for i in front:
        if i in used:
            continue
        eq = [j for j in front if ok[i, j] and ok[j, i]]
        used.update(eq)
        reps.append(int(i))
    return reps


def leaves(ok2, reps):
    """Base frontier reps that acquire a strict dominator under ok2."""
    strict = ok2 & ~ok2.T
    np.fill_diagonal(strict, False)
    return [i for i in reps if strict[i].any()]


def relax(items, cnt, names):
    """ok with every item in names granted."""
    c = cnt.copy()
    for nm in names:
        c -= items[nm].astype(np.int16)
    return c == 0


def name_of(p):
    return p["members"][0]


def family_of(name):
    """A coarse family / lab key: the org and the leading name stem."""
    org, _, rest = name.partition("/")
    stem = re.split(r"[-_.]?\d", rest, maxsplit=1)[0].lower().rstrip("-_.") or rest.lower()
    return org.lower(), stem


def per_model(P, ok, cap, items, reps):
    """For each frontier rep i: the would-be dominators (capacity-strict dominators of i), each one's failing items;
    the minimal failing sets; the single items whose removal alone takes i off the frontier."""
    n = len(P)
    names = sorted(items)
    stack = np.stack([items[k][reps] for k in names])          # [item, rep, host]
    cap_strict = cap & ~cap.T
    np.fill_diagonal(cap_strict, False)
    out = {}
    for r_i, i in enumerate(reps):
        hosts = [j for j in range(n) if j != i and not (ok[i, j] and ok[j, i])]
        F = {j: frozenset(names[k] for k in np.where(stack[:, r_i, j])[0]) for j in hosts}
        # j must not be <= i once the walls fall: it must not emulate-back (approximated by: base ok[j, i] false and
        # not capacity-equal); capacity dominators are the natural set
        cand = [j for j in hosts if cap_strict[i, j]]
        sets = sorted({F[j] for j in cand if F[j]}, key=len)
        minimal = [s for s in sets if not any(o < s for o in sets)]
        mlen = min((len(s) for s in minimal), default=0)
        # hosts realising the minimal-cardinality sets
        best = [j for j in cand if F[j] and len(F[j]) == mlen]
        singles = sorted({next(iter(F[j])) for j in cand if len(F[j]) == 1})
        tw = [s for s in minimal if not (s & CAPACITY)]                     # type-wall-only minimal sets
        out[i] = dict(name=name_of(P[i]), members=len(P[i]["members"]), n_capdom=len(cand), minimal=minimal,
                      mlen=mlen, best=best, singles=singles, typewall_sets=tw)
    return out


CACHE = os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "frontier_study_items.npz")


def cached_build(refresh=False):
    """build() with the item masks cached (packed bits) between runs of this study."""
    import pickle
    if not refresh and os.path.exists(CACHE) and os.path.exists(CACHE + ".pkl"):
        z = np.load(CACHE)
        P, names = pickle.load(open(CACHE + ".pkl", "rb"))
        n = len(P)
        unp = lambda a: np.unpackbits(a)[: n * n].reshape(n, n).astype(bool)
        items = {nm: unp(z[f"i{k}"]) for k, nm in enumerate(names)}
        return P, unp(z["ok"]), unp(z["cap"]), items
    P, ok, cap, items = build()
    names = sorted(items)
    np.savez_compressed(CACHE, ok=np.packbits(ok), cap=np.packbits(cap),
                        **{f"i{k}": np.packbits(items[nm]) for k, nm in enumerate(names)})
    pickle.dump((P, names), open(CACHE + ".pkl", "wb"))
    return P, ok, cap, items


def kindsets(P, ok, cap, items, reps, cover):
    """Per frontier rep: the minimal KIND sets whose removal lets some capacity dominator strictly dominate it, and
    the binding one (minimum cardinality; ties -> the set realised by the widest-covering host)."""
    n = len(P)
    names = sorted(items)
    knames = [kind_of(x) for x in names]
    stack = np.stack([items[k][reps] for k in names])
    cap_strict = cap & ~cap.T
    np.fill_diagonal(cap_strict, False)
    out = {}
    for r_i, i in enumerate(reps):
        cand = np.where(cap_strict[i])[0]
        K = {}
        for j in cand:
            ks = frozenset(knames[k] for k in np.where(stack[:, r_i, j])[0])
            if ks and (ks not in K or cover[j] > cover[K[ks]]):
                K[ks] = j
        sets = sorted(K, key=len)
        minimal = [s for s in sets if not any(o < s for o in sets)]
        if minimal:
            m = min(len(s) for s in minimal)
            bind = max((s for s in minimal if len(s) == m), key=lambda s: cover[K[s]])
            out[i] = dict(minimal=[sorted(s) for s in minimal], binding=sorted(bind), via=name_of(P[K[bind]]))
        else:
            out[i] = dict(minimal=[], binding=["(capacity-maximal: no capacity dominator)"], via="")
    return out


def single_pairs(item, P=None, ok=None, cap=None, items=None):
    """(frontier target, host) pairs whose ONLY refusing item is `item` (host capacity-strictly above the target)."""
    if P is None:
        P, ok, cap, items = cached_build()
    reps = reps_strict(ok)
    cnt = sum(m.astype(np.int16) for m in items.values())
    one = (cnt == 1) & items[item]
    cs = cap & ~cap.T
    return [(i, j) for i in reps for j in np.where(one[i] & cs[i])[0]], P


def main():
    os.makedirs(OUT, exist_ok=True)
    P, ok, cap, items = cached_build("--refresh" in __import__("sys").argv)
    reps = reps_strict(ok)
    cnt = sum(m.astype(np.int16) for m in items.values())
    nmem = np.array([len(p["members"]) for p in P])
    cover = {j: int(nmem[ok[:, j]].sum()) for j in range(len(P))}
    print(f"shapes {len(P)}  frontier {len(reps)}  items {len(items)}")
    pm = per_model(P, ok, cap, items, reps)
    ks = kindsets(P, ok, cap, items, reps, cover)
    by_item = collections.defaultdict(lambda: dict(in_min=set(), single=set()))
    for i, r in pm.items():
        for s in r["minimal"]:
            for it in s:
                by_item[it]["in_min"].add(i)
        for it in r["singles"]:
            by_item[it]["single"].add(i)
    kinds = sorted({kind_of(k) for k in items})
    kind_items = {k: [x for x in items if kind_of(x) == k] for k in kinds}
    # leave-one-out on a kind (that kind granted) and leave-one-in (only that kind walled; capacity and gen kept)
    cf_kind = {}
    keep = ("capacity", "paradigm (gen)")
    for k in kinds:
        ok2 = relax(items, cnt, kind_items[k])
        lv = leaves(ok2, reps)
        others = [x for x in items if kind_of(x) not in keep and kind_of(x) != k]
        only = relax(items, cnt, others)
        cf_kind[k] = dict(n_frontier=len(reps_strict(ok2)), leave=[name_of(P[i]) for i in lv],
                          only_this_wall=len(reps_strict(only)) if k not in keep else None,
                          binds=sum(k in r["binding"] for r in ks.values()),
                          in_minimal=sum(any(k in s for s in r["minimal"]) for r in ks.values()))
        print(f"kind {k:45s} frontier {cf_kind[k]['n_frontier']:4d}  leave {len(lv):3d}  "
              f"only-this {cf_kind[k]['only_this_wall']}  binds {cf_kind[k]['binds']}", flush=True)
    # greedy: grant the kind that lowers the frontier most, repeatedly (capacity and gen never granted)
    greedy, granted = [], []
    pool = [k for k in kinds if k not in keep]
    while pool:
        best = min(pool, key=lambda k: len(reps_strict(relax(items, cnt, sum((kind_items[x] for x in granted + [k]),
                                                                              [])))))
        granted.append(best)
        pool.remove(best)
        greedy.append((best, len(reps_strict(relax(items, cnt, sum((kind_items[x] for x in granted), []))))))
        print("greedy", greedy[-1], flush=True)
    cf_item = {}
    for it in sorted(by_item, key=lambda x: -len(by_item[x]["in_min"])):
        if it in CAPACITY:
            continue
        ok2 = relax(items, cnt, [it])
        lv = leaves(ok2, reps)
        cf_item[it] = dict(kind=kind_of(it), n_frontier=len(reps_strict(ok2)), leave=[name_of(P[i]) for i in lv],
                           in_min=len(by_item[it]["in_min"]), single=len(by_item[it]["single"]))
    tw = [x for x in items if kind_of(x) not in keep]
    ok_tw = relax(items, cnt, tw)
    mine = ["bias", "output / shared-expert gate", "qk-norm / v-norm", "attention layout (heads / K=V / layer kinds)",
            "MLA", "activation", "tied / factorized embeddings", "other attention / forward features", "sink / softcap",
            "parallel block", "FFN form (GLU / vanilla)"]
    fcount = collections.Counter(x for p in P for x in set(filter(None, p["slots"].get("feats", "").split("+"))))
    rare = {k: sum(1 for i in reps if any(fcount[x] <= k for x in filter(None, P[i]["slots"].get("feats", "").split("+"))))
            for k in (1, 2, 3, 5)}
    mine_all = len(reps_strict(relax(items, cnt, [x for x in items if kind_of(x) in mine])))
    json.dump(dict(
        n_shapes=len(P), n_frontier=len(reps), n_frontier_no_typewalls=len(reps_strict(ok_tw)), mine_all_granted=mine_all, rare_feature_classes=rare,
        models={name_of(P[i]): dict(r, minimal=[sorted(s) for s in r["minimal"]],
                                    typewall_sets=[sorted(s) for s in r["typewall_sets"]],
                                    best=[name_of(P[j]) for j in r["best"][:5]],
                                    kind_minimal=ks[i]["minimal"], binding=ks[i]["binding"], via=ks[i]["via"],
                                    cover=cover[i], slots=P[i]["slots"], gen=P[i]["slots"]["gen"],
                                    dims={a: P[i]["dims"].get(a) for a in ("hidden", "eff_layers", "kv_cap", "qheads",
                                                                             "ffn_total", "ffn_active")},
                                    all_members=P[i]["members"])
                for i, r in pm.items()},
        cf_kind=cf_kind, greedy=greedy, cf_item=cf_item), open(f"{OUT}/attribution.json", "w"), indent=1,
        default=list)
    print("wrote", f"{OUT}/attribution.json")


# --- release: the post's floor / relaxed numbers (graph/post_numbers.py, section "study") ------------------------
KEEP = ("capacity", "paradigm (gen)")
RELAX_BIAS_NORM = ("bias", "norm kind / placement", "qk-norm / v-norm", "output / shared-expert gate", "sink / softcap")


def floor_numbers(P=None):
    """The itemised counterfactuals the post quotes, on the payable (headline) relation:
      floor_types_free           every type wall granted (each item whose kind is not capacity or the paradigm):
                                 only the construction costs (capacity) and the AR <= diffusion split remain
      floor_types_paradigm_free  the paradigm granted too (capacity only)
      max_single_kind_freed      the most frontier classes one wall kind frees (classes that acquire a strict
                                 dominator with that kind's items granted), and which kind
      unique_feature_frontier    frontier classes carrying a feature no other snapshot shape has
      headline_relaxed_bias_norm the headline with the bias, norm, qk- / v-norm, output-gate and sink / softcap kinds
                                 granted"""
    P, ok, cap, items = build(P)
    reps = reps_strict(ok)
    cnt = sum(m.astype(np.int16) for m in items.values())
    kinds = sorted({kind_of(k) for k in items})
    kind_items = {k: [x for x in items if kind_of(x) == k] for k in kinds}
    tw = [x for x in items if kind_of(x) not in KEEP]
    tp = [x for x in items if kind_of(x) != "capacity"]
    freed = {k: len(leaves(relax(items, cnt, kind_items[k]), reps)) for k in kinds if k not in KEEP}
    best = max(freed, key=freed.get)
    fcount = collections.Counter(x for p in P for x in set(filter(None, p["slots"].get("feats", "").split("+"))))
    uniq = sum(1 for i in reps if any(fcount[x] <= 1 for x in filter(None, P[i]["slots"].get("feats", "").split("+"))))
    rbn = [x for x in items if kind_of(x) in RELAX_BIAS_NORM]
    return {"headline": len(reps), "n_shapes": len(P),
            "floor_types_free": len(reps_strict(relax(items, cnt, tw))),
            "floor_types_paradigm_free": len(reps_strict(relax(items, cnt, tp))),
            "max_single_kind_freed": freed[best], "max_single_kind": best,
            "kind_freed": dict(sorted(freed.items(), key=lambda kv: -kv[1])),
            "unique_feature_frontier": uniq,
            "headline_relaxed_bias_norm": len(reps_strict(relax(items, cnt, rbn)))}


if __name__ == "__main__":
    if "--floor" in __import__("sys").argv:
        print(json.dumps(floor_numbers(), indent=1))
    else:
        main()
