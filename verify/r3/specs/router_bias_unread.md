# router_bias_unread
live 5 / placed 5 (Motif 1, alice_ai 1, deepseek2 1, kohaku 1, youtu_vl_moe 1); direction over-credit; batch a9_router

## 1. What the code computes
- Youtu-A7B: tencent/Youtu-LLM-A7B-Base/configuration_youtu_vl_moe.py:78-80 defaults `topk_method =
  "quantile_balancing"`, `moe_router_enable_expert_bias = True`, `moe_router_expert_bias_update_method = "quantile"`
  (the snapshot omits the bias key); modeling_youtu_vl_moe.py:406-409 `_quantile_balancing_enabled`; :450-474 builds
  `qb_beta` (QB) else `e_score_correction_bias`; :518-524 QB selection `topk(logits - qb_beta, k + 1)[:, :k]` in the
  RAW-LOGIT domain; :527-549 otherwise `sigmoid + bias` with -inf group fill; :554-561 weights = sigmoid(logits)
  renormalised (norm_topk_prob) x routed_scaling_factor.
- Motif-3: Motif-Technologies/Motif-3/modeling_motif.py:978-983 `expert_bias` exists iff `load_balance_coeff is not
  None` (snapshot 1e-4); :854-856 top-k on `scores + expert_bias`, weights the unbiased scores.
- AliceAI: AMAImedia/AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF/modeling_alice_ai.py:535-540 e_score_correction_bias
  iff `router_bias_correction` (snapshot true); :547-553 selection on score + bias, weights unbiased.
- Mark-SLM-V3: miceworks/Mark-SLM-V3-150M/modeling_mark_v3.py:186 `exp_probs_b` always; :196-199 top-k on
  `sigmoid + exp_probs_b`, weights unbiased.
- Kohaku: KBlueLeaf/TIPO-v2.1-1B-A200M/modeling_kohaku.py:141 persistent `expert_bias` always; :159-160 top-k on
  `scores + expert_bias`, weights unbiased.
- Same unread fact in the router_score_unread types (p5 gives them no bias either): Beam modeling_beam.py:355, Muse-
  Spark modular_muse_spark.py:331 (score domain); Grug skyrl_train_models_grug_moe.py:546-551 and Kolibri-1
  modular_kolibri1.py:136-137 (logit domain: top-k on `l + b`, weights sigmoid(l)).
- Domain matters: for a sigmoid router, top-k on sigma(l) + b and top-k on l + b' order experts differently unless
  b, b' are constant (the boundary sigma(l1) - sigma(l2) = c is not a line l1 - l2 = c').

## 2. Classifier change
- reads: load_balance_coeff (Motif), router_bias_correction (AliceAI), moe_router_enable_expert_bias with the code
  default True + topk_method / moe_router_expert_bias_update_method (youtu_vl_moe), model_type / architectures
- new: FEATURES key `router_bias_logit` (a logit-domain selection bias, emitted INSTEAD of router_bias) in _feats;
  ROUTER_BIAS_TYPES += kohaku, beam, muse_spark, muse_spark_text; a new ROUTER_BIAS_LOGIT_TYPES = {grug_moe, kolibri1};
  _CLASS_DEFAULT_FALLBACK["moe_router_enable_expert_bias"] = {"youtu_vl_moe": True}
- detection:
```python
def detect_router_bias_unread(cfg, model_type):
    lm = lm_of(cfg); mt = model_type
    if not _n_experts(lm):
        return None
    if mt == "youtu_vl_moe":
        if not lm.get("moe_router_enable_expert_bias", True):
            return None
        qb = (_s(lm.get("moe_router_expert_bias_update_method")) == "quantile"
              or _s(lm.get("topk_method")) in ("quantile_balancing", "quantile"))
        return "logit" if qb else "score"
    if mt == "motif":
        return "score" if lm.get("load_balance_coeff") is not None else None
    if mt == "alice_ai":
        return "score" if lm.get("router_bias_correction") else None
    if mt == "deepseek2" and "MarkV3ForCausalLM" in archs(cfg):
        return "score"
    if mt in ("kohaku", "beam", "muse_spark", "muse_spark_text"):
        return "score"
    if mt in ("grug_moe", "kolibri1"):
        return "logit"
    return None
```
- apply note: p5:model_graph.py `_router_flags` (~893): add `d.get("router_bias_correction")`, `d.get(
  "load_balance_coeff") is not None and mt == "motif"`, `_cfg_value(d, "moe_router_enable_expert_bias")` (so the
  youtu default applies) to `bias`, and return a 4th value `logit` (mt in ROUTER_BIAS_LOGIT_TYPES or youtu QB).
  `_feats` (~1784): `f.add("router_bias_logit" if logit else "router_bias")`. `_moe_sig` keeps `bias` truthy for both
  (the extra-experts rule). Mark-SLM gets its bias via SIGMOID_ROUTER_TYPES += deepseek2 (router_score_unread).

## 3. Relation rule
- target side: a score-domain biased target needs a score-domain biased host (router_bias, unchanged); a logit-domain
  target needs a logit-domain host (router_bias_logit)
- host side: either biased host plays a bias-free target (b = 0); neither plays the other domain
- kind: exact (same domain, or b = 0); wall (cross domain; biased target <- bias-free host)
- cost: none
- free by the config rules: the bias values are weights (b = 0 free); the domain is code, not config
- theorem: `router_bias_off` (host b = 0), `router_bias_not_bias_free` (Walls); needs theorem:
  router_bias_domain_ne (sigmoid router: no b' with topk(l + b') = topk(sigma(l) + b) for all l unless b is constant)

## 4. Test fixtures
- flag: Youtu-LLM-A7B-Base -> "logit"; Motif-3, AliceAI, Mark-SLM-V3, TIPO-v2.1, Beam-1.0, Muse-Spark GGUF -> "score";
  Grug-67B-10.05, Kolibri-1 -> "logit"
- not flag: DeepSeek-V3 (bias read from topk_method), sarvam-105b (moe_router_enable_expert_bias read), K2-Horizon-375B
  (moe_gate_bias read), DeepSeek-V2-Lite, Qwen3-30B-A3B

## Verdict
real. Fewer edges: the 5 carriers (plus Beam 2, Muse 2, Kohaku; Grug 7, Kolibri, Youtu as logit-domain) stop being
played by bias-free hosts; Motif-3 and AliceAI are walled elsewhere today (poly_norm / attn_res), so the visible change
is ~15 models, mostly Beam / Muse / Grug targets losing V3-style or bias-free hosts.
