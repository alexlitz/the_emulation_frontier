"""Mismatch analysis -> conversion plan.

Starts from the repo's own reduction test (graph.reduce: categorical slot differences + capacity
gaps), then lists every structural difference between the two configs/modeling codes and says,
for each, whether the conversion step is EXACT (a fold or padding: function preserved, up to
float rounding), EPS (exact only as eps -> 0) or APPROXIMATE (and which approximator handles it).

  python -m approx.plan google/gemma-4-26B-A4B openai/gpt-oss-20b
"""
from __future__ import annotations

import argparse
import json
import os

from approx.arch import SCfg, TCfg


def _cfg(repo):
    import glob
    hits = sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/config.json")))
    if hits:
        return json.load(open(hits[-1]))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(repo, "config.json")))


def reduce_report(tc_raw, sc_raw):
    """graph.reduce's view: slot diffs and capacity gaps for 'student emulates teacher'."""
    from graph.model_graph import classify, dims_of
    from graph.reduce import axis_gaps, cat_diffs
    tcd = tc_raw.get("text_config", tc_raw) | {"architectures": tc_raw.get("architectures"), "model_type": tc_raw.get("model_type")}
    A = {"slots": classify(tc_raw), "dims": dims_of(tc_raw)}
    B = {"slots": classify(sc_raw), "dims": dims_of(sc_raw)}
    return A, B, cat_diffs(A, B), axis_gaps(A, B)


def plan(tc: TCfg, sc: SCfg):
    """[(component, teacher, student, EXACT|EPS|APPROX, step)]"""
    ng = sum(t == "full_attention" for t in tc.layer_types)
    P = []
    add = lambda *r: P.append(r)
    pad = sc.h > tc.h
    add("hidden", tc.h, sc.h, "EXACT", f"zero-pad {sc.h - tc.h} residual dims; every RMSNorm weight x sqrt({tc.h}/{sc.h}) "
        f"so the student's mean over the padded width equals the teacher's, AND the eps scaled with it: the student's "
        f"rms_norm_eps is eps {tc.h}/{sc.h} (Lean hidden_pad_needs_eps; keeping the teacher's eps misses by eps/ms)")
    add("vocab / tokenizer", tc.vocab, "201088 (own tokenizer)", "EXACT", f"keep the teacher tokenizer, student vocab_size={tc.vocab} (config value)")
    add("embedding scale", f"x{tc.embed_scale:g} (sqrt h, in weight dtype)", "none", "EXACT", "fold into embed_tokens (computed in the teacher's dtype)")
    add("tied lm_head", "tied to embed", "untied", "EXACT", "lm_head = unscaled teacher embedding")
    add("final norm", "x/rms * w", "x/rms * w", "EXACT", "copy (with the pad fold)")
    add("final logit softcap", tc.softcap, "none", "APPROX", "dropped; one lm_head scale fitted by KL on calibration (tanh is ~linear below the cap)")
    add("layer_scalar", "per-layer scalar on the whole residual", "none", "EXACT" if pad else "EPS",
        "student residual = teacher residual / C_l (C_l = prod of the scalars so far): every update is divided by its "
        "cumulative scale. RMSNorm with eps > 0 is NOT scale invariant (Lean rmsNorm_smul_eps: RMSNorm_eps(h/C) = "
        "RMSNorm_{eps C^2}(h)), so layer l needs eps/C_l^2, a different eps per layer, which one rms_norm_eps cannot give. "
        + ("Exact via the eps channel (approx.convert.eps_plan): the first padded dim carries a constant v_l with "
           "v_l^2 = h_t eps/C_l^2 - h_s eps_s (norm weights 0 there; set by the embedding, moved by each layer's "
           "down_proj_bias), with student eps_s = eps h_t/(h_s max C^2)" if pad else
           "No padded dim for an eps channel: the student keeps eps and each norm misses by "
           "sqrt((ms + eps)/(ms + eps C_l^2)), relative ~ eps |C_l^2 - 1| / (2 ms)"))
    add("depth", tc.L, sc.L, "APPROX", f"drop {tc.L - sc.L} layers chosen by ablation dKL (domain calibration) under the type "
        "constraint below; the next kept layer's least-squares refit targets the teacher stream AFTER the dropped ones (absorbs them)")
    add("layer types", f"{tc.L - ng} sliding (w={tc.window}) + {ng} global", "alternating sliding/full",
        "EXACT/APPROX", f"teacher global layers only at student full positions (DP); sliding_window={tc.window} (config value), "
        f"so a teacher sliding layer at a student sliding slot is exact; at a full slot exact for context <= {tc.window}")
    add("attention scale", "1.0 (q,k RMS-normed)", f"1/sqrt({sc.hd})", "EXACT", "folded into q_proj (with yarn attention_scaling^-2 if used)")
    add("sinks", "none", "per-head sink logit", "EXACT", "sink = -1e4 (contributes exp(-1e4) = 0)")
    add("q/k RMSNorm (with weight), v RMSNorm (no weight)", "per head", "none (q/k/v are affine + bias)", "APPROX",
        "per-token normalization replaced by the LS-best affine map of the normalized input x (fits how rms(Wx) varies with x's direction); biases used as intercepts")
    add("global k == v", "v = v_norm(k_proj x)", "separate v_proj", "EXACT", "v_proj built from k_proj")
    add("q width", f"{tc.nq}x{tc.hd} sliding / {tc.nq}x{tc.hd_g} global", f"{sc.nq}x{sc.hd}", "APPROX",
        f"{sc.nq // tc.nq} student q-heads per teacher head; sliding q width {tc.nq * tc.hd} = {sc.nq * sc.hd} but every student head "
        f"only sees its kv group's {sc.hd} key dims, global q is {tc.nq * tc.hd_g // (sc.nq * sc.hd)}x wider")
    add("kv width (GQA)", f"{tc.nkv}x{tc.hd} sliding / {tc.nkv_g}x{tc.hd_g} global", f"{sc.nkv}x{sc.hd}", "APPROX",
        f"keys: pick {sc.hd // 2} of the teacher kv head's rope pairs per student kv group (importance = attention-weighted score "
        "variance, Hungarian assignment to student frequencies); values: output-weighted PCA (top eigvecs of C^1/2 W_o W_o^T C^1/2); "
        "o_proj refit by LS on the student's per-head outputs")
    add("rope", "sliding theta 1e4 full; global theta 1e6, 25% rotary (proportional), 75% NoPE", f"one theta for all, full rotary",
        "APPROX", "student theta = teacher sliding theta (every 4th sliding frequency exact); global/NoPE pairs go to the nearest "
        "student frequency (lowest is ~1e-4 rad/token) -- the assignment cost includes the frequency mismatch")
    add("sandwich norms", "post-attn, pre/post-FFN x2, post-FFN RMSNorms", "pre-norm only", "APPROX",
        "pre-norms: EXACT folds into the following linear maps (gate/up/router columns); post-norms: per-channel LS gain "
        "(f ~ a*m1 + b*m2) folded into the down projections, then o_proj / FFN gain+bias refit against the teacher stream")
    add("dense MLP alongside MoE", f"dense {tc.Id} GeLU MLP + MoE", "MoE only", "APPROX",
        f"a unit replicated in all {sc.E} experts is dense (top-k weights sum to 1): D dense units (by importance) are "
        f"replicated, member-expert units pruned by importance to make room; D chosen to maximize kept importance")
    add("experts", f"{tc.E} x {tc.Ie}, top-{tc.topk}", f"{sc.E} x {sc.I}, top-{sc.topk}", "APPROX",
        f"group {tc.E // sc.E} teacher experts per student expert ({tc.E // sc.E}x{tc.Ie}={tc.E // sc.E * tc.Ie} <= {sc.I}) by "
        "co-activation of the calibration routing (optionally dropping rarely-used experts first)")
    add("router", f"softmax({tc.E}) -> top-{tc.topk} renorm x per_expert_scale; input rms(x)*scale/sqrt(h)",
        f"linear+bias -> top-{sc.topk} -> softmax", "APPROX",
        "input scaling EXACT fold; per_expert_scale EXACT fold into down; group logit = weighted LS onto log-sum-exp of the members' logits")
    add("activation", "gelu_tanh(g) * u", "(clamp(u,+-7)+1) * clamp(g,<=7)*sigmoid(1.702 g)", "APPROX",
        "up bias -1 cancels the +1 (EXACT); up scaled into the clamp range, inverse folded into down (EXACT inside range); "
        "per-unit gate scale c: g*sigmoid(1.702 c g) ~ gelu(g) fitted with the clamp, 1/c folded into down")
    add("biases", "none", "q/k/v/o, router, expert gate_up/down", "EXACT", "zero, or used as LS intercepts")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher", nargs="?", default="google/gemma-4-26B-A4B")
    ap.add_argument("student", nargs="?", default="openai/gpt-oss-20b")
    a = ap.parse_args()
    traw, sraw = _cfg(a.teacher), _cfg(a.student)
    try:
        A, B, diffs, gaps = reduce_report(traw, sraw)
        print(f"graph.reduce: does {a.student} emulate {a.teacher}?")
        for s, x, y, c in diffs:
            print(f"  slot {s}: {x} -> {y}  currency {c}")
        for n, need, have, ok in gaps:
            if need != have:
                print(f"  {n:13} need {need:>9} have {have:>9} {'ok' if ok else 'FAIL'}")
    except Exception as e:   # pragma: no cover
        print("graph.reduce unavailable:", e)
    tc = TCfg.from_hf(traw)
    sc = SCfg.from_hf(sraw)
    sc.vocab = tc.vocab
    print("\nconversion plan:")
    for comp, t, s, kind, step in plan(tc, sc):
        print(f"  [{kind:6}] {comp}: {t}  ->  {s}\n           {step}")


if __name__ == "__main__":
    main()
