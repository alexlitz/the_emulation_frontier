# router_score_unread
live 16 / placed 22 (k2_horizon 9, grug_moe 7, beam 2, muse_spark_text 2, kolibri1 1, sarvam_mla 1); direction both; batch a9_router

## 1. What the code computes
- Sigmoid scoring FIXED in code, no key p5 reads (p5 _router_flags -> 'softmax'):
  - Beam: $NOCODE_CACHE/beam/transformers-main/modeling_beam.py:339-375 (BeamTopKRouter = DeepseekV3TopkRouter):
    :354 `scores = sigmoid(logits)`, :355 `+ e_score_correction_bias` (selection only), group top-2-sum with
    `-inf` fill :369, :372-373 renorm iff norm_topk_prob (config True).
  - Sarvam-105B: sarvamai/sarvam-105b/modeling_sarvam_moe.py:274-275 (`scoring_func = "sigmoid"`, `topk_method =
    "noaux_tc"` hard-coded), :281-284 (groups refused), :289 `norm_topk_prob = True`, :311-327 sigmoid, top-k on
    score + bias, renorm (k > 1), x routed_scaling_factor.
  - Grug-MoE: ext:grug_moe/gh__marin-community__marin/skyrl_train_models_grug_moe.py:509-510 (weight + persistent
    fp32 `bias` buffer), :544-553 top-k on `logits + bias` (LOGIT domain), weights `sigmoid(selected logits) * 2.5 /
    (sum + eps)` (always R; 2.5 folds).
  - Muse-Spark: tfmain:models/muse_spark/modular_muse_spark.py:316-337: `ACT2FN["sigmoid"]` (:320), logits x
    output_multiplier (:329, folds into W), top-k on score + e_score_correction_bias (:331-332), `/ (sum + 1e-15)`
    always (:335), x routed_scaling_factor.
  - Kolibri-1: tfmain:models/kolibri1/modular_kolibri1.py:118-139: top-k on `logits + e_score_correction_bias`
    (LOGIT domain, :136), weights `sigmoid(selected logits)` (:137), renorm iff norm_topk_prob (:138-139; snapshot False).
  - Mark-SLM-V3 (model_type deepseek2, arch MarkV3ForCausalLM): miceworks/Mark-SLM-V3-150M/modeling_mark_v3.py:186,
    195-200 `p = sigmoid(l)`, top-k on `p + exp_probs_b`, `/ sum.clamp(1e-6)` always.
- Score READ from a config key the code honours, missing from SCORE_CONFIG_TYPES (host score_cap = its own score only):
  - K2-Horizon: IFM/K2-Horizon-7B/modeling_k2_horizon.py:538, 562-566 softmax or sigmoid from `router_score_func`;
    :569-572 selection bias = gate.bias (moe_gate_bias, read by p5); :575-577 renorm by norm_topk_prob.
  - Motif: Motif-Technologies/Motif-3/modeling_motif.py:846-851 `score_func` sigmoid / softmax.
  - Kohaku: KBlueLeaf/TIPO-v2.1-1B-A200M/modeling_kohaku.py:147-152 softmax / sqrtsoftplus / sigmoid from scoring_func.
- K2-Horizon-3.7B / 7B / 32B, K2-Type-0.9B, Jais-3-0.9B / 32B are dense (num_experts 0): not live, correctly unflagged.

## 2. Classifier change
- reads: model_type, architectures, router_score_func / score_func / scoring_func, the expert count
- new: SIGMOID_ROUTER_TYPES += beam, sarvam_mla, grug_moe, muse_spark, muse_spark_text, kolibri1, deepseek2 (only
  Mark-SLM-V3 carries it in the snapshot); SCORE_CONFIG_TYPES += k2_horizon ("sigmoid", "softmax"), motif ("sigmoid",
  "softmax"), kohaku ("sigmoid", "softmax", "sqrtsoftplus"); companions RENORM_FIXED sarvam_mla / grug_moe /
  muse_spark / muse_spark_text / deepseek2 = "R" (k > 1), GROUP_SCORE_TYPES beam = "top2sum"
- detection:
```python
def detect_router_score_unread(cfg, model_type):
    lm = lm_of(cfg); mt = model_type
    if not _n_experts(lm):
        return None
    if mt in ("beam", "sarvam_mla", "grug_moe", "muse_spark", "muse_spark_text", "kolibri1"):
        return "sigmoid/sigmoid"
    if mt == "deepseek2" and "MarkV3ForCausalLM" in archs(cfg):
        return "sigmoid/sigmoid"
    if mt == "k2_horizon":
        return (_s(lm.get("router_score_func")) or "softmax") + "/sigmoid|softmax"
    if mt == "motif":
        return (_s(lm.get("score_func")) or "softmax") + "/sigmoid|softmax"
    if mt == "kohaku":
        return (_s(lm.get("scoring_func")) or "sigmoid") + "/sigmoid|softmax|sqrtsoftplus"
    return None
```
  (value = "<score>/<score_cap joined by |>")
- apply note: p5:model_graph.py SIGMOID_ROUTER_TYPES (~1924; membership also gives router_bias, see
  router_bias_unread for the logit-domain ones grug_moe / kolibri1), SCORE_CONFIG_TYPES (~875), RENORM_FIXED (~855),
  GROUP_SCORE_TYPES (~879). No code change to _moe_sig / moe_why.

## 3. Relation rule
- target side: a sigmoid target needs "sigmoid" in the host's score_cap (unchanged moe_why rule)
- host side: a K2-Horizon / Motif / Kohaku host plays softmax targets too (its config switches the score: free);
  the code-fixed sigmoid hosts lose softmax targets
- kind: exact (same score); wall (sigmoid target <- softmax-only host; softmax target <- fixed-sigmoid host, the
  sigmoid -> softmax limit is unmeasured)
- cost: none
- free by the config rules: the router scoring type is free where the code reads it (K2 / Motif / Kohaku)
- theorem: `softmax_router_ne_sigmoid` (Walls row "router_sigmoid target <- softmax host"); `sigmoid_router_limit`
  (limit, unmeasured: kept as a wall); same-score copies `moeR_dup_renorm`

## 4. Test fixtures
- flag: Beam-1.0 (both), sarvam-105b, Grug-67B-Datakit-09.21, snowball-67b-base, Muse-Spark GGUF / mlx, Kolibri-1,
  Mark-SLM-V3 -> "sigmoid/sigmoid"; K2-Horizon-375B, Jais-3-375B, Jais-3-MoVA -> "sigmoid/sigmoid|softmax";
  Motif-3 -> "sigmoid/sigmoid|softmax"; TIPO-v2.1 -> "sigmoid/sigmoid|softmax|sqrtsoftplus"
- not flag: IFM/K2-Horizon-7B, Jais-3-32B (dense), DeepSeek-V2-Lite, DeepSeek-V3 (already tabled), Qwen3-30B-A3B,
  Ling-1T (score_function read)

## Verdict
real. Both: 14 MoE models (Beam 2, Grug 7, Muse 2, Kolibri, Sarvam, Mark) become sigmoid targets and lose the
softmax hosts credited today (fewer edges), and as hosts swap softmax targets for sigmoid ones; K2-Horizon MoE x3,
Motif, Kohaku gain softmax targets as hosts (more edges, though Motif / AliceAI are walled elsewhere).
