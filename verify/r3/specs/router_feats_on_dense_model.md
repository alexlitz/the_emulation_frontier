# router_feats_on_dense_model
live 1 / placed 1 (cyrah 1); direction over-wall; batch a3_placement

## 1. What the code computes
- Lilgarca/my-model-v01/modeling_cyrah.py:532-533: `dense = index < first_k_dense_replace or num_experts <= 1`;
  snapshot first_k_dense_replace 14 = num_hidden_layers and num_experts 1: every layer is CyrahMLP, CyrahMoE
  (:186-235) is never built. (Even when built, CyrahMoE selects top-k on the scores with no selection bias,
  :217-218: `topk_method noaux_tc` is not read there.)
- agurung/cobalt-seeded-*-s3 / -s2 (nemotron_h, 5 in the snapshot): layers_block_type lists linear_attention /
  full_attention / mlp and no "moe" (NemotronH builds an MoE block only for "moe" entries, tf5
  models/nemotron_h/modeling_nemotron_h.py:905 block-type mapping "moe": NemotronHMoE); their n_routed_experts 8 / norm_topk_prob are template
  keys. (Their mtp_layers_block_type "moe" is an aux MTP head: irrelevant to the main logits.)
- p5 `_feats` (p5:model_graph.py :1777-1791) adds router_sigmoid / router_sqrtsoftplus / router_bias / router_group /
  shared_gate whenever `_moe_like(d)` (:509: any num_experts / moe_num_experts / n_routed_experts) or
  num_local_experts, even though classify already decides ffn dense (`_moe_disabled` or n_exp <= 1, :2263-2264).
  router_bias is a target-side wall. p5 output: cyrah feats router_bias + router_sqrtsoftplus; the 5 cobalt
  configs router_bias + router_sigmoid, with ffn glu / vanilla.

## 2. Classifier change
- reads: expert-count keys, first_k_dense_replace, layers_block_type / hybrid_override_pattern, is_moe / use_moe
- new: no router features when no routed layer is built, in _feats
- detection:
```python
def detect_router_feats_on_dense_model(cfg, model_type):
    lm = lm_of(cfg)
    present = [k for k in _EXPERT_KEYS if lm.get(k) is not None and not isinstance(lm.get(k), bool)]
    if not present or not any(num(lm.get(k)) for k in present):
        return None
    n_exp = max(num(lm.get(k)) for k in present)
    L = _n_layers(lm)
    if n_exp <= 1:
        return "dense"
    fk = num(lm.get("first_k_dense_replace"))
    if L and fk >= L:
        return "dense"
    lbt = lm.get("layers_block_type")
    if isinstance(lbt, list) and lbt:
        s = {str(x).lower() for x in lbt}
        if ("mlp" in s or "ffn" in s) and "moe" not in s:
            return "dense"
    pat = lm.get("hybrid_override_pattern")
    if isinstance(pat, str) and pat and "-" in pat and "E" not in pat:
        return "dense"
    return None
```
  (_EXPERT_KEYS = num_experts, moe_num_experts, n_routed_experts, num_local_experts; _n_layers = the layer count.)
- apply note: p5:model_graph.py `_feats` (:1777): guard the router block with a helper `_moe_built(d)` = not
  `_moe_disabled(d)` and max expert count > 1 and `_dense_first(d) < _layer_count(d)`; i.e. the same decision
  classify uses for ffn (:2272), plus the first_k_dense_replace >= L case (also make classify's ffn test use it).

## 3. Relation rule
- target side: a dense target carries no router features (router_bias wall gone)
- host side: unchanged (router features are free on the host side)
- kind: exact (removes a spurious wall)
- cost: none
- free by the config rules: n/a
- theorem: none needed (the feature is absent from the forward); `dense_sub_moe` unaffected

## 4. Test fixtures
- flag: Lilgarca/my-model-v01 (num_experts 1, first_k 14 = L), agurung/cobalt-...-vs16-s3 and -base-s3 (no moe
  entry), nvidia/Hymba-1.5B-Base (num_experts 1), ai21labs/AI21-Jamba-Reasoning-3B (num_experts 1),
  dphn/dolphin-2.9.1-mixtral-1x22b (one local expert) -> dense. (The last three get no router feature in p5 today:
  same code fact, no current effect.) Also flagged: the other 3 cobalt configs, Jamba2-3B, Hymba-Instruct,
  nvidia/Kimi-K2.6-Eagle3 (a drafter, skipped).
- not flag: deepseek-ai/DeepSeek-V2-Lite, nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (MoE nemotron_h), ai21labs/Jamba-v0.1
  (16 experts), Qwen/Qwen1.5-MoE-A2.7B, RedHatAI/Mixtral-8x7B-Instruct-v0.1, nvidia/Nemotron-H-8B-Base-8K

## Verdict
real: 6 points (Cyrah + 5 cobalt) lose a spurious router_bias target wall: edges INTO them from bias-free hosts
appear (more edges, 6 targets). Region: MoE.
