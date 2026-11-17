# moa_routed_attention_heads
live 1 / placed 1 (jetmoe 1); direction over-credit; batch a7_attn_exotic

## 1. What the code computes
- tf5:models/jetmoe/modular_jetmoe.py:212-231 JetMoeMoA: E = num_local_experts (q_proj, o_proj) expert pairs,
  input_linear `hidden -> kv_channels * num_kv_heads` and output_linear back, a learned output bias (:228).
- :106-149 JetMoeTopKGating: `logits = W x`, top-k, `softmax(top_k_logits)` (renormalised over the k), no bias.
- :288-373 JetMoeAttention: `query = experts.map(x)` (:337, k expert queries per token, :239-260), ONE shared kv_proj
  (:324, :338), RoPE, k / v `repeat(1, top_k, 1, 1)` (:356-357), softmax attention, then `experts.reduce` (:371):
  `sum_j gate_j * o_proj_{e_j}(attn_j) + bias` (:262-282).
- configuration_jetmoe.py:58-59, 73: num_attention_heads := num_key_value_heads * top_k (jetmoe-8b: 16 kv heads,
  k = 2 -> 32; E = 8). The snapshot carries the legacy keys moe_num_experts 8 / moe_top_k 2 (tf5 defaults agree).
- So the token's 32 query heads are 2 routed blocks of 16 from 8 expert q-projections, and the output is a
  router-weighted sum: a top-k MoE on the attention's q / o projections. p5 classify gives jetmoe a plain GQA softmax
  attention (32 q heads, 16 kv heads) with ffn=moe; the MoA bias (:281) is not in its feats (bias_out).

## 2. Classifier change
- reads: model_type jetmoe; num_local_experts / moe_num_experts, num_experts_per_tok / moe_top_k
- new: FEATURES key `moa` ("wall", "free") plus dims field `moa` = "E<e>k<k>", in _feats + FEATURES + dims_of;
  also bias_out for jetmoe (the MoA output bias)
- detection:
```python
def detect_moa_routed_attention_heads(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "jetmoe":
        return None
    e = num(lm.get("num_local_experts", lm.get("moe_num_experts", 8)))
    k = num(lm.get("num_experts_per_tok", lm.get("moe_top_k", 2)))
    return f"E{e}k{k}" if e > 1 and k >= 1 else None
```
- apply note: p5:model_graph.py `_feats`: `if mt == "jetmoe": f.add("moa"); f.add("bias_out")`; FEATURES: add "moa";
  dims_of: `moa` string, and for a moa host the q-head capacity used by kv_heads_fit is num_kv_heads (one tied
  expert block), not num_attention_heads.

## 3. Relation rule
- target side: a MoA target needs a MoA host with the same top-k and score (softmax after top-k), E_h >= E_t: the
  MoE <- MoE routing rules applied to the attention router (expert duplication, extra experts kept out of the
  top-k); a plain-attention host cannot select heads per token (discontinuous at the routing boundary): wall.
- host side: a MoA host ties all E experts (same q / o weights): with renormalised gates summing to 1 the output is
  the one block's attention, any router: exact; capacity num_kv_heads q heads on num_kv_heads kv heads (MHA-like
  per block), output bias set to 0.
- kind: wall (target <- plain host); exact (MoA host -> plain target, tied experts; MoA <- MoA per the routing rules)
- cost: host q-head capacity = num_kv_heads (16 for jetmoe-8b, not 32)
- free by the config rules: top-k is NOT free (num_experts_per_tok); router renorm / scaling are free values
- theorem: `tied_moe_is_dense` (host side, gates sum to 1), `continuous_host_ne_top1_moe` (target-side wall),
  MoE <- MoE routing table rows (moeR_*) for MoA <- MoA

## 4. Test fixtures
- flag: jetmoe/jetmoe-8b -> "E8k2"
- not flag: RedHatAI/Mixtral-8x7B-Instruct-v0.1, allenai/OLMoE-1B-7B-0924 (FFN MoE, plain attention), Qwen/Qwen3-8B, deepseek-ai/DeepSeek-V3

## Verdict
real. Fewer edges: jetmoe-8b stops being hosted by every plain GQA-MoE host big enough (likely dozens to hundreds of
host classes) and its host capacity drops from 32 to 16 q heads (some of its hosted targets go).
