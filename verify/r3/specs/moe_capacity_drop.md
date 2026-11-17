# moe_capacity_drop
live 1 / placed 1 (hunyuan_v1_moe 1); direction over-credit; batch a9_router

## 1. What the code computes
- tencent/Hunyuan-A13B-Instruct config auto_map -> hunyuan.HunYuanMoEV1ForCausalLM; hunyuan.py:43 imports
  HunYuanDecoderLayer from modeling_hunyuan.py, so the remote MoE below runs.
- modeling_hunyuan.py:74-140 topkgating: softmax (:76); capacity `max(topk, topk * S // E)` per forward, S = all tokens
  of the batch (:78); top-k (:82); weights renormalised over the selected (:91-94); priority = cumsum over the
  flattened (choice rank, token) order (:98-113: every token's 1st choice before any 2nd choice); assignments with
  priority >= capacity are dropped (:127-131; combine weight 0, no re-renormalisation, :136).
- :456-467 HunYuanTopKGate.forward calls topkgating(logits, moe_topk[0]); moe_drop_tokens (snapshot false) is not
  read by this path.
- Consequences: a token's k-th expert can be dropped by OTHER tokens' (also later tokens') earlier choices, so the
  output is not a per-token function and not causal; prefill (S >= 64: capacity S / 8 at k = 8, E = 64) drops,
  single-token decode (capacity 8 >= 1) never does.
- Near misses: the transformers-native hunyuan_v1_moe router has no capacity (4.57 modeling_hunyuan_v1_moe.py: no
  "capacity"); tencent/HunyuanImage-3.0/hunyuan.py:262-267 honours drop_tokens (false -> capacity = max load, dropless).

## 2. Classifier change
- reads: model_type hunyuan_v1_moe, auto_map (remote hunyuan.*)
- new: FEATURES key `moe_capacity` ("wall", "wall") in _feats + FEATURES
- detection:
```python
def detect_moe_capacity_drop(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "hunyuan_v1_moe" or not _n_experts(lm):
        return None
    return True if "hunyuan.HunYuan" in auto_map(cfg) else None
```
- apply note: p5:model_graph.py `_feats` router block (~1777-1793): `if mt == "hunyuan_v1_moe" and "hunyuan.HunYuan" in
  " ".join(map(str, (d.get("auto_map") or {}).values())): f.add("moe_capacity")`; FEATURES: `"moe_capacity":
  ("wall", "wall", "per-forward expert capacity with dropped assignments (Hunyuan-A13B remote topkgating): a token's
  experts depend on the other tokens of the forward; no per-token router forms it, and the capacity is not a config
  value")`. RENORM_FIXED keeps hunyuan_v1_moe R.

## 3. Relation rule
- target side: wall (no per-token MoE host reproduces batch-dependent drops; even another capacity host matches only
  with the same E / k ratio, the same token order and the same batch)
- host side: wall (the capacity formula is code; moe_drop_tokens is not read, so the host cannot go dropless)
- kind: wall both ways
- cost: none
- free by the config rules: none (moe_drop_tokens would be a switch, and this code ignores it)
- theorem: needs theorem: capacity_drop_not_tokenwise (an MoE layer with per-forward capacity is not a function of
  the token alone: two contexts with the same token and different co-routed tokens give different outputs)

## 4. Test fixtures
- flag: tencent/Hunyuan-A13B-Instruct -> True
- not flag: tencent/HunyuanImage-3.0 (honours moe_drop_tokens false), Qwen/Qwen3-30B-A3B, allenai/OLMoE-1B-7B-0924

## Verdict
real, small. Fewer edges: Hunyuan-A13B (softmax R top-8 of 64 + shared, today a normal MoE point) loses all MoE
edges both ways (as a host: the dense / MoE targets it is credited with, ~hundreds below its 80B scale; as a target,
the larger softmax-R hosts).
