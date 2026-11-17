# block_routing
live 1 / placed 1 (llada2_moe 1); direction over-credit; batch a9_router

## 1. What the code computes
- inclusionAI/LLaDA2.2-mini/modeling_llada2_moe.py:228-229, 236 LLaDA2MoeGate.__init__ reads block_size (32),
  expert_capacity (48 of 256), registers expert_bias; no n_group / topk_group.
- :244-275 block_routing: tokens in blocks of block_size (assert divisible, :248); block score per expert = max over
  the block's tokens of sigmoid + bias (:255); keep the top expert_capacity experts per block (:258-266); per-token
  top-k among the allowed experts (-inf for the rest, :272-273).
- :284-296 forward: scores = sigmoid(logits), selection on scores + expert_bias via block_routing, weights = gathered
  unbiased scores, renormalised when top_k > 1, x routed_scaling_factor.
- A token's experts depend on the other tokens of its block. The config's n_group 8 / topk_group 4 are not read by
  this gate, yet p5 gives router_group and a top2sum group limit (GROUP_SCORE_TYPES llada2_moe).
- LLaDA2.0 / 2.1 (no expert_capacity / block_size keys) use the group-limited gate (yifanyu/LLaDA2.1-mini-ORL
  modeling_llada2_moe.py:344, 359): not flagged.

## 2. Classifier change
- reads: model_type llada2_moe, expert_capacity, block_size, the expert count
- new: FEATURES key `block_routing` ("wall", "free") in _feats; group fields zeroed in _moe_sig for this gate
- detection:
```python
def detect_block_routing(cfg, model_type):
    lm = lm_of(cfg)
    E = _n_experts(lm)
    if model_type != "llada2_moe" or not E:
        return None
    cap, bs = num(lm.get("expert_capacity")), num(lm.get("block_size"))
    if cap and bs and cap < E:
        return True
    return None
```
- apply note: p5:model_graph.py `_feats` router block (~1777-1793): add block_routing under the detector's condition
  and skip router_group then; `_moe_sig` (~939): G = g = 0, gcap "" when the llada2_moe config has expert_capacity and
  block_size (the gate never reads n_group). FEATURES: `"block_routing": ("wall", "free", "...")`.

## 3. Relation rule
- target side: a block-routed target needs a block-routed host with the same block_size; with m copies the host
  keeps m x expert_capacity (copies tie at block level as whole classes, strict otherwise) and k_h = m k_t
- host side: a block-routed host plays a per-token target with expert_capacity = num_experts (all allowed:
  :258-273 reduce to plain top-k on score + bias), or block_size 1 with expert_capacity >= k; the token count must be
  divisible by block_size (:248)
- kind: exact (block <- block, same block_size; plain <- block host via capacity = E); wall (block <- plain host)
- cost: none
- free by the config rules: expert_capacity / block_size are routing numbers (shape-preserving, like a window >=
  context): free on the host; the block-routing code is the switch
- theorem: `moeR_topk_unique`, `moeR_copies_topk` (per block, both levels strict) | needs theorem:
  block_routing_copies (two-level copy selection) and block_routing_not_tokenwise (a block-routed layer is not a
  per-token router: wall evidence for the target side)

## 4. Test fixtures
- flag: inclusionAI/LLaDA2.2-mini -> True
- not flag: inclusionAI/LLaDA2.1-mini, inclusionAI/LLaDA2.0-mini, yifanyu/LLaDA2.1-mini-ORL (group-limited gate),
  inclusionAI/Ling-mini-2.0

## Verdict
real, small. Fewer edges: LLaDA2.2-mini stops being played by the per-token diffusion MoE hosts (LLaDA2.0 / 2.1, ~5)
and loses the wrong top2sum group compare; as a host it keeps per-token targets (capacity = E) and drops the group-
limited targets it was credited with via the unread n_group (now no group_cap).
