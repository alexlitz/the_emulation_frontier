# hash_route_by_code
live 1 / placed 1 (deep 1); direction over-credit; batch a9_router

## 1. What the code computes
- AETHORIA-AI/TR-HASH-MOE-500M-HF/configuration_deep.py:29, 33, 55, 67-68: `mlp_type` must be "token_routed" and
  `routing_strategy` must be "token_id_balanced_hash" (both the defaults; others raise); :32, :40 route weight
  top_k_primary_weight 0.5 and routed_output_scale 2.0; :107 `expert_width = intermediate_size // num_experts`.
- modeling_deep.py:213-229 DeepTokenRoutedMLP.forward: `routes = topk_token_to_expert[:, token_ids]` (:218, a
  persisted token-id table: NO learned router), `weights = route_weights` constants (0.5, 0.5) (:219), per-expert
  SwiGLU on the routed tokens, output `shared_output_scale * shared + routed_output_scale * routed` (:229).
- p5 sets hash_route only from num_hash_layers (`_feats` ~1787): deep gets a content-routed softmax MoE, renorm
  '?deep', and ew = intermediate_size (4x the code's expert width 512 / 4 = 128).
- Pacific-i64/TR-MOE-306: same model_type deep, no mlp_type key (DeepConfig default token_routed), no auto_map and no
  code in the cache: flagged by config, not live.

## 2. Classifier change
- reads: model_type deep, mlp_type (default token_routed), the expert count
- new: FEATURES key hash_route (existing) in _feats; dims ew = intermediate_size // num_experts in dims_of
- detection:
```python
def detect_hash_route_by_code(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "deep" or not _n_experts(lm):
        return None
    if _s(lm.get("mlp_type", "token_routed")) != "token_routed":
        return None
    return True
```
- apply note: p5:model_graph.py `_feats` (~1787): `if _num(d.get("num_hash_layers")) > 0 or (mt == "deep" and
  str(d.get("mlp_type", "token_routed")) == "token_routed"): f.add("hash_route")`. dims_of (~3114, expert width):
  for mt == "deep" use intermediate_size // num_experts.

## 3. Relation rule
- target side: a hash-routed target needs a hash-routed host (FEATURES hash_route ("wall", "wall"), unchanged)
- host side: a deep host has no content router (routing_strategy is forced): wall for every learned-router target
- kind: wall both ways (as the existing hash_route rule)
- cost: none
- free by the config rules: none (DeepConfig refuses any other routing_strategy)
- theorem: `hash_not_content` (host side, PROOFS Round-2 "hash_route: learned target <- hash host"); the target side
  is the open "hash target <- learned host" row (`topk_code_select`, `code_margin`, `renorm_softmax_shift`): here the
  route weights are constants, which a renormalised softmax host gives exactly with zero content router rows plus
  the k-hot code, so this target side could later become payable; kept a wall (the rule is wall/wall today)

## 4. Test fixtures
- flag: AETHORIA-AI/TR-HASH-MOE-500M-HF -> True, Pacific-i64/TR-MOE-306 -> True (same type, config only)
- not flag: deepseek-ai/DeepSeek-V2-Lite, Qwen/Qwen3-30B-A3B, allenai/OLMoE-1B-7B-0924

## Verdict
real, small. Fewer edges: TR-HASH-MOE-500M stops being credited to / from every learned-router MoE (today a softmax
'?deep' target only its own family plays, so the visible loss is its host side: dense / MoE targets it was
credited with as a 4-expert shared MoE host, ~tens of small models).
