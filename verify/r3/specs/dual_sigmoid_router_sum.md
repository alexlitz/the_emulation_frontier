# dual_sigmoid_router_sum
live 1 / placed 1 (ernie5_5_moe 1); direction both; batch a9_router

## 1. What the code computes
- ext:ernie5_5_moe/gh__baidu__vLLM-ERNIE/moe.py:120-121: requires `moe_split_feature_routing` (snapshot true);
  :124 refuses n_group / topk_group != 1; :126-127 sigmoid scoring only.
- :151-157 gate = Linear(hidden, 2 * E) (two routers in one GEMM); :159-161 a per-expert routed scale parameter (used :204);
  :162-164 e_score_correction_bias.
- :217 `scores = sigmoid(l[:, :E]) + sigmoid(l[:, E:])` (in (0, 2)).
- :66-88 ernie5_5_routing: top-k on `scores + bias` (score-domain selection bias), weights = unbiased scores,
  renormalised iff norm_topk_prob (config true), then x `scale[ids]` (a per-expert fold into that expert's down-proj).
- p5 _router_flags reads scoring_func "sigmoid" -> score "sigmoid", as for one sigmoid router.
- sigma(a.x) + sigma(b.x) is no sigma(c.x) (nor a constant multiple of one) for linear a, b, c unless a = b: the
  selection boundaries and the renormalised weights differ.

## 2. Classifier change
- reads: moe_split_feature_routing, model_type ernie5_5_moe
- new: moe_sig score "sigmoid2", score_cap ("sigmoid", "sigmoid2"), in _moe_sig (+ SCORE_CONFIG_TYPES entry)
- detection:
```python
def detect_dual_sigmoid_router_sum(cfg, model_type):
    lm = lm_of(cfg)
    if not _n_experts(lm):
        return None
    if model_type == "ernie5_5_moe" and lm.get("moe_split_feature_routing"):
        return "sigmoid2"
    return None
```
- apply note: p5:model_graph.py `_moe_sig` (~925): `score = "sigmoid2" if d.get("moe_split_feature_routing") else ...`;
  SCORE_CONFIG_TYPES (~875) += "ernie5_5_moe": ("sigmoid", "sigmoid2") (the tied-row construction below, not a
  config switch); `_feats` keeps router_sigmoid (the feature is free; moe_why prices the score).

## 3. Relation rule
- target side: a sigmoid2 target needs a sigmoid2 host (same code); single-sigmoid / softmax hosts: wall
- host side: an ERNIE-5.5 host plays a single-sigmoid target with the same renorm: tie the two router rows (b = a):
  score 2 sigma(a); selection topk(2 sigma(a) + 2 b_t) = topk(sigma(a) + b_t) (host bias 2 b_t); renormalised weights
  cancel the 2, unrenormalised ones fold 1/2 into the per-expert scale. Softmax targets: wall (sigmoid only, :126);
  grouped targets: wall (:124, already so: no group_cap)
- kind: exact (sigmoid2 <- sigmoid2; sigmoid <- sigmoid2 host by tying); wall (sigmoid2 <- any other)
- cost: none
- free by the config rules: none (the split routing is a shape-changing switch: 2E router rows)
- theorem: `moeR_scaling_fold` (the factor 2 / the scale fold), `router_bias_off` | needs theorem: sigmoid2_tied_rows
  (tied rows = 2 sigma, selection with doubled bias) and sigmoid2_ne_sigmoid (no single sigmoid router equals the sum)

## 4. Test fixtures
- flag: baidu/ERNIE-5.5-Flash-Base -> "sigmoid2"
- not flag: baidu/ERNIE-4.5-21B-A3B-PT, baidu/ERNIE-4.5-300B-A47B-PT (ernie4_5_moe: one softmax router),
  deepseek-ai/DeepSeek-V3

## Verdict
real, small. Fewer edges for ERNIE-5.5-Flash as a target (no single-sigmoid host plays it); as a host it keeps the
sigmoid targets (by tying), so no host-side loss.
