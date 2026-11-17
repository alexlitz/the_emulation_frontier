"""Rescue rules for two edges the relation grants for free, with their measured verdicts and costs.

PORTED: the dense-host rule in round 2, pass 3; the looped-host rule in pass 5 (graph/model_graph.emul_matrix:
LOOP_PLAIN_TYPES, dims loop_plain / loop_norm / loop_vbias; every other looped host a wall for an untied target).

1. dense host  >=  MoE target (ffn moe / moe_shared on a vanilla / glu host, charged 2x depth by
   MOE_DEPTH_FACTOR). No continuous host equals a top-1 MoE at any depth (`continuous_host_ne_top1_moe`),
   so the edge is at best approximate. The pre-norm construction (moe/dense_host_prenorm.py) spends 2 host
   layers per MoE layer -- a select FFN writing gamma * m_e, m_e = sum_a t^2 sigma(beta t), t = z_a - z_e,
   into E carried channels at gamma = 1e-8 (|carry|^2 / |h|^2 ~ 1e-12, so the RMSNorms see the target's
   stream), then an expert FFN whose units for expert e have their gate shifted by -(Lambda / gamma) c_e / r
   (a scale-robust threshold: only the sign of c_e / r matters), plus 2E(E-1) erase units recomputing m.
   MEASURED (moe/data/dense_host_prenorm.jsonl, fp32, 24 prompts, the target's own greedy continuations):
     - the one real top-1 unit-gate MoE in the snapshot that runs stock (Huihui-MoE-0.8B-2E, 28 layers,
       E = 2) and gpt-oss-tinystories forced to top-1: top-1 100%, every 64-token continuation identical,
       KL 5e-9 / 9e-13, at Lambda = 1e12, beta = 1e7 (soft zone: router margins |t| <~ 1e-5);
     - at Lambda = 1e6, beta = 2e3 (soft zone ~1e-2) Huihui diverges (top-1 97.8%, ex32 38%): 1.5% of its
       token-layers are within 1e-2 of a routing tie, 0.12% within 1e-3;
     - gamma = 1 (the norm-free construction's scale) and dropping the erase both fail outright (top-1
       1-45%): the stale carry masks the next layer's winner;
     - a TOP-K target (k >= 2) is NOT reproduced by masks: on gpt-oss-tinystories' native top-2 the top-1
       construction gets top-1 76.6%, exact top-k membership with uniform gates 77.3%; only the ORACLE
       (membership by the k-th logit and the target's own gate weights) passes. Membership needs
       relu(min(t_a, t_b)) (two nonlinear stages) and the gate is a third multiplicative factor a SwiGLU unit
       lacks; neither can be carried (a carried value is read divided by the rms). A gate STAIRCASE (L
       masked copies of every expert, gate rounded to 1/L) needs L = 64 to pass (L = 16: ex32 71%);
     - OLMoE-1B-7B (top-8 of 64, no renormalisation; bf16-generated continuations scored teacher-forced in
       fp32): the oracle 99.93% top-1, the top-1 masks 0.65%; forced to top-1 the unit-gate host gets 0.72%,
       because without renormalisation the top-1 gate is p_max(x), not 1 -- the third factor again;
     - router margins on real text: Huihui top-1 margin < 1e-2 on 1.5% of token-layers (< 1e-3: 0.12%);
       gpt-oss-tinystories 0.6% (0.06%); OLMoE top-1 margin 2.9% (0.30%) and its top-8 boundary
       z_(8) - z_(9) 11.7% (1.3%);
     - stories15M_MOE (ggml test model) has four IDENTICAL experts: routing is moot, every variant but the
       broken carries passes at the sharp constants -- a control, not evidence for top-k.
   VERDICT: payable for top-1 unit-gate targets (k = 1 with renormalised / softmax-after-top-k gates, or a
   constant gate) without a gated shared expert, at 2x depth, +E residual channels and +2E(E-1) FFN units
   per host layer (on top of the E * e_w + shared units the relation already charges). E = 1 targets are
   dense FFNs: exact at 1x. Every other MoE target on a dense host is a wall (top-k would need >= 3 layers
   per MoE layer for membership and ~64x the expert units for the gate; not built end to end).

2. looped host  >=  untied target (a looped host of b distinct layers, K loops, for an untied target of
   L <= b layers at eff_layers >= L). Iteration 1 must be the target, iterations 2..K the identity. The
   pre-norm construction (hf/looped_host_prenorm.py): a counter channel c (the block's last FFN writes
   gamma |x|^2 / d at every token, exactly, from 2d + 2 SwiGLU units, and a huge count at the start token
   via a flag channel f), FFN units gate-shifted by -B c / r, and an attention marker in the lowest-
   frequency RoPE pair so every query attends to the start token from iteration 2 on, whose value has been
   swamped by its own counter. Iteration 1 reads c = 0 and is the target exactly. A value bias b_v (Qwen2)
   survives the silenced attention; 4d more units per layer cancel W_o b_v from iteration 2 on (unmasked
   minus gate-shifted |x|^2 / d: on only when c > 0). A host that RMS-normalises between iterations
   (Nanbeige) needs 2 refresh units per layer, c += G (c / r)^2, to keep the start token swamped.
   MEASURED (hf/data/looped_host_prenorm.jsonl): SmolLM2-135M and Qwen2.5-0.5B looped K = 2, 4: top-1 100%,
   every continuation identical, KL ~1e-11 off the start-token id (a recurring start-token id -- SmolLM2's
   BOS = EOS generated mid-text -- carries the flag and gives up its own distribution, argmax kept);
   the norm-between-loops host and an untied prelude / looped middle / untied coda host pass (the coda's
   K / V projections get a column on the counter channel carrying the start token's constant target key /
   value; erasing the counter instead leaves an fp32 cancellation residue that fails on Qwen2.5); naive
   looping, FFN-only masking and the norm-between-loops host without refresh fail; one huge counter at
   every token flattens the logits.
   VERDICT: payable at +2 residual channels and 2d + 2 FFN units in one layer (+ 2 per layer with a norm
   between loops, + 4d per layer with a value bias), under the start-token rule (the input begins with one
   fixed token), for a host whose every iteration re-runs pre-norm causal self-attention over the current
   iteration's states without QK-norm (LOOP_PLAIN_TYPES). Of the zoo's eight looped hosts only Nanbeige4.2
   qualifies; IQuest-Loop (loop-1 KV + a 64-token window), ARLoop / kotodama (attention residuals), Dory
   (nGPT, sliding windows), LFM3-RD (gated convolutions), PreNormLoop / qwen3loop (QK-norm bounds the
   marker logit) are walls.

FRONTIER EFFECT (frontier_effect(): graph/hub_10k.json, payable headline, 842 shapes / 3,933 models):
current frontier 104. MoE rule alone: 106; the dense-host >= MoE slice goes 2,918 -> 98 edges (65 exact
E = 1 targets, now at 1x depth, 31 payable top-1: Huihui-MoE-0.8B-2E); Qwen/Qwen2-57B-A14B and
cloudyu/Yi-34Bx2-MoE-60B join the frontier. Loop rule alone: 104; looped >= untied 797 -> 113 edges, all on
Nanbeige4.2-3B. Both: 106 (96 MoE + 104 loop edges; the slices share 73 MoE-target / looped-dense-host
pairs, which must pass both). Walling both slices outright also gives 106: the rescue changes covers
(Qwen-72B 2,083 -> 2,044 models), not the frontier count. The rest of the 3,268 "2x depth" pairs -- 330
moe_shared targets on routed-MoE hosts and 20 GLU targets on vanilla hosts -- are outside these rules.

Pure functions only; graph/model_graph.py is not edited. `apply(ok, P, cfgs)` returns the relation with both
rules applied; `frontier_effect()` measures what they do to the current headline.
"""
from __future__ import annotations

# --- the agreement standard (act/act_swap_agreement.py, attn/mixer_gap_agreement.py) ----------------------
PAYABLE_TOP1 = 0.99         # teacher-forced top-1 agreement on the target's own continuations
PAYABLE_EX32 = 0.90         # fraction of prompts whose 32-token greedy continuation is identical


def passes(row):
    """A measured row meets the payable standard (a teacher-forced-only row, ex32 None, never does)."""
    return row["top1"] >= PAYABLE_TOP1 and row.get("ex32") is not None and row["ex32"] >= PAYABLE_EX32


# --- 1. dense host >= MoE target ----------------------------------------------------------------------
DENSE_MOE_TOP1_PAYABLE = True      # top-1 unit-gate targets: the pre-norm select / mask construction
DENSE_MOE_TOPK_PAYABLE = False     # k >= 2: membership needs 2 nonlinear stages, the gate a third factor
MOE_DENSE_DEPTH = 2                # host layers per target MoE layer (select FFN, then expert FFN)
MOE_SELECT_SCALE = 1e-8            # gamma: the carried select signal, relative to the residual
MOE_MASK_STRENGTH = 1e12           # Lambda: the gate shift per unit of m / r
MOE_SELECT_BETA = 1e7              # beta: select-unit sharpness, t^2 sigma(beta t)
MOE_SOFT_ZONE = 1e-5               # router margins below this blend two experts at the measured constants

# MoE families whose shared expert is multiplied by a sigmoid of the input (a third factor)
GATED_SHARED_TYPES = frozenset({"qwen2_moe", "qwen3_next", "qwen3_5_moe", "qwen3_5", "qwen3_6_moe",
                                "hunyuan_v1_moe"})
# families whose top-k weights are always renormalised / softmaxed over the selected experts
RENORM_TYPES = frozenset({"mixtral", "gpt_oss", "minimax"})
_TOPK_KEYS = ("num_experts_per_tok", "moe_topk", "moe_top_k", "moe_k", "num_experts_per_token", "top_k_experts",
              "experts_top_k", "num_selects")
_E_KEYS = ("num_experts", "n_routed_experts", "num_local_experts", "num_routed_experts", "moe_num_experts")


def _num(v):
    if isinstance(v, list):
        v = max((x for x in v if isinstance(x, (int, float))), default=None)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def moe_routing(cfg):
    """The routing facts a rule needs, from a raw (language-model) config: E, k, whether the k selected
    weights are renormalised, whether a shared expert is present / sigmoid-gated."""
    E = next((_num(cfg.get(k)) for k in _E_KEYS if _num(cfg.get(k))), None) \
        or _num((cfg.get("ffn_config") or {}).get("moe_num_experts"))
    k = next((_num(cfg.get(kk)) for kk in _TOPK_KEYS if _num(cfg.get(kk))), None) \
        or _num((cfg.get("ffn_config") or {}).get("moe_top_k"))
    mt = cfg.get("model_type")
    renorm = cfg.get("norm_topk_prob")
    if renorm is None and mt in RENORM_TYPES:
        renorm = True
    shared = bool(cfg.get("shared_expert_intermediate_size") or _num(cfg.get("n_shared_experts"))
                  or _num(cfg.get("num_shared_experts")) or cfg.get("moe_shared_expert_intermediate_size")
                  or cfg.get("shared_intermediate_size"))
    return dict(E=int(E) if E else None, k=int(k) if k else None, renorm=renorm, model_type=mt,
                shared=shared, shared_gated=shared and mt in GATED_SHARED_TYPES)


def dense_moe_verdict(cfg):
    """('exact' | 'payable' | 'wall', reason) for a MoE target on a dense host."""
    r = moe_routing(cfg)
    if r["E"] is not None and r["E"] <= 1:
        if r["shared_gated"]:
            return "wall", "one routed expert but a sigmoid-gated shared expert (a third factor)"
        return "exact", "one expert: the router's gate is a constant, the block is a dense FFN"
    if r["k"] is None or r["E"] is None:
        return "wall", "expert count / top-k not read from the config"
    if r["k"] >= 2:
        return "wall", (f"top-{r['k']}: membership is not one-layer separable bias-free and the gate weight "
                        "is a third multiplicative factor (measured: masks alone ~77% top-1)")
    if r["renorm"] is not True:
        return "wall", "top-1 without renormalisation: the output is p_max(x) * Y (a third factor)"
    if r["shared_gated"]:
        return "wall", "sigmoid-gated shared expert: a third multiplicative factor"
    return "payable", "top-1 unit gate: select FFN + masked expert FFN under the pre-norm (measured exact)"


def dense_moe_cost(cfg, dims=None):
    """Charged cost of the payable construction: depth factor, extra residual channels, extra FFN units
    per host layer (beyond the E * e_w + shared the relation already compares)."""
    v, _ = dense_moe_verdict(cfg)
    E = moe_routing(cfg)["E"] or (dims or {}).get("experts") or 0
    if v == "exact":
        return dict(depth=1, channels=0, units=0)
    if v == "payable":
        return dict(depth=MOE_DENSE_DEPTH, channels=E, units=2 * E * (E - 1))
    return None


# --- 2. looped host >= untied target -------------------------------------------------------------------
LOOP_EXTRA_CHANNELS = 2            # the counter c and the start-token flag f
LOOP_COUNTER_SCALE = 1e-6          # gamma: per-iteration counter at ordinary tokens
LOOP_START_COUNTER = 1e9           # the start token's counter, relative to its |h|
LOOP_NEEDS_START_TOKEN = True      # the start-token rule: the input begins with one fixed token


def loop_units(d):
    """FFN units the counter needs in the block's last layer: |x|^2 by 2d SwiGLU units, f^2 by 2."""
    return 2 * d + 2


# Looped families by how their later iterations run (each repo's modeling file, read 2026-11-13). Covered:
# every iteration re-runs pre-norm decoder layers with fresh causal self-attention over the current
# iteration's states, no QK-norm, nothing between iterations but (measured) an RMSNorm.
LOOP_PLAIN_TYPES = frozenset({
    "nanbeige",          # Nanbeige4.2: 22 layers x 2, its final RMSNorm also runs between loops (measured:
                         # hf/looped_host_prenorm.py host-internorm, with 2 refresh units per layer)
})
LOOP_NOT_COVERED = {
    "iquestloopcoder": "loop 2 attends to loop-1 KV (global) and a 64-token window of loop-2 KV, mixed by a "
                       "query gate: the start token leaves the window and loop-1 values cannot be silenced",
    "lfm3_rd": "gated short convolutions (B x * conv(C x)) in the recurrent motif: a product of linear maps "
               "has no gate to shift",
    "arloop": "block attention residuals: each sublayer reads a softmax mix of RMS-normed block outputs, "
              "not the residual sum",
    "kotodama": "block attention residuals (attn_res): no plain residual stream to carry the counter",
    "dory": "nGPT residuals (unit-norm lerp), prelude re-injection every loop, 1024-token sliding layers",
    "arloop_baseline": "QK-norm after RoPE bounds the start-token marker logit (~sqrt(head_dim) w_q w_k): "
                       "later iterations cannot be fully silenced",
    "qwen3loop": "Qwen3 layers with QK-norm: the start-token marker logit is bounded, later iterations "
                 "leak (the untied prelude / coda split itself is measured fine: host-split)",
}


def looped_untied_verdict(host_cfg, plain_types=None, not_covered=None):
    """('payable' | 'wall', reason) for an untied target on a looped host of this config."""
    mt = host_cfg.get("model_type")
    nc = LOOP_NOT_COVERED if not_covered is None else not_covered
    if mt in nc:
        return "wall", nc[mt]
    if mt in (LOOP_PLAIN_TYPES if plain_types is None else plain_types):
        return "payable", "counter + start-token attention marker under the pre-norm (measured exact)"
    return "wall", "later iterations not known to re-run plain causal self-attention over the current states"


# --- applying both rules to a relation -------------------------------------------------------------------
def dense_moe_slice(P):
    """[i, j] = target i is a MoE (moe / moe_shared) and host j a dense (vanilla / glu) FFN."""
    import numpy as np
    ffn = np.array([p["slots"]["ffn"] for p in P])
    return np.isin(ffn, ["moe", "moe_shared"])[:, None] & np.isin(ffn, ["vanilla", "glu"])[None, :]


def loop_slice(P):
    """[i, j] = target i untied, host j looped."""
    import numpy as np
    loop = np.array([p["slots"]["loop"] for p in P])
    return (loop == "untied")[:, None] & (loop == "looped")[None, :]


def _point_verdict(p, cfgs, fn, best=False):
    """A point's verdict over its members. As a TARGET every member must be reproduced (the strictest:
    wall < payable < exact); as a HOST any member that can host will do (best=True)."""
    order = {"wall": 0, "payable": 1, "exact": 2}
    vs = [fn(cfgs[n])[0] for n in p["members"] if n in cfgs]
    if not vs:
        return "wall"
    return (max if best else min)(vs, key=order.get)


def apply(ok, P, cfgs, mode="payable", moe=True, loop=True):
    """The relation ok (emul_matrix(P, mode)) with the rescue rules applied to the two slices.
    cfgs: member name -> language-model config. Outside the slices ok is returned unchanged."""
    import copy

    from graph.model_graph import emul_matrix
    out = ok.copy()
    if moe:
        sl = dense_moe_slice(P)
        verdict = [_point_verdict(p, cfgs, dense_moe_verdict) if p["slots"]["ffn"].startswith("moe") else None
                   for p in P]
        # exact (E = 1): the target is a dense FFN -> the relation with the target's ffn slot set to glu
        Pe = copy.deepcopy(P)
        for p, v in zip(Pe, verdict):
            if v == "exact":
                p["slots"]["ffn"] = "glu"
        ok_exact = emul_matrix(Pe, mode=mode)
        # payable (top-1 unit gate): +E channels and +2E(E-1) units per layer, at the 2x depth
        Pp = copy.deepcopy(P)
        for p, v, q in zip(Pp, verdict, P):
            if v == "payable":
                E = q["dims"].get("experts") or 0
                p["dims"] = dict(p["dims"], hidden=p["dims"]["hidden"] + E,
                                 ffn_total=p["dims"]["ffn_total"] + 2 * E * (E - 1))
        ok_pay = emul_matrix(Pp, mode=mode)
        for i, v in enumerate(verdict):
            if v is None:
                continue
            row = sl[i]
            out[i, row] = {"exact": ok_exact[i, row], "payable": ok_pay[i, row]}.get(v, False)
    if loop:
        sl = loop_slice(P)
        hv = [_point_verdict(p, cfgs, looped_untied_verdict, best=True) if p["slots"]["loop"] == "looped" else None
              for p in P]
        Pl = copy.deepcopy(P)
        for p in Pl:
            if p["slots"]["loop"] == "untied":
                p["dims"] = dict(p["dims"], hidden=p["dims"]["hidden"] + LOOP_EXTRA_CHANNELS)
        ok_loop = emul_matrix(Pl, mode=mode)
        for j, v in enumerate(hv):
            if v is None:
                continue
            col = sl[:, j]
            # AND with what the MoE rule left: a MoE target on a looped dense host must pass both
            out[col, j] = out[col, j] & ok_loop[col, j] if v == "payable" else False
    return out


def frontier_effect(json_path="graph/hub_10k.json"):
    """Frontier size and slice edge counts on the snapshot: current, each rule alone, both, and both walled."""
    import numpy as np

    from graph.currency_curve import n_frontier
    from graph.frontier import load
    from graph.model_graph import _dedup_points, _lm_config, emul_matrix, iter_json_configs
    M = load()
    P = _dedup_points(M)
    cfgs = {n: _lm_config(c) for n, c in iter_json_configs(json_path)}
    ok = emul_matrix(P, mode="payable")
    off = ~np.eye(len(P), dtype=bool)
    sm, sl = dense_moe_slice(P) & off, loop_slice(P) & off
    cnt = np.array([len(p["members"]) for p in P])

    def stats(o):
        return dict(frontier=n_frontier(o), moe_edges=int((o & sm).sum()), loop_edges=int((o & sl).sum()),
                    covered_models=int(((o & off) * cnt[:, None]).sum()))
    walled = ok.copy()
    walled[sm] = False
    walled[sl] = False
    res = {"current": stats(ok),
           "moe_rule": stats(apply(ok, P, cfgs, loop=False)),
           "loop_rule": stats(apply(ok, P, cfgs, moe=False)),
           "both_rules": stats(apply(ok, P, cfgs)),
           "both_walled": stats(walled)}
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(frontier_effect(), indent=1))
