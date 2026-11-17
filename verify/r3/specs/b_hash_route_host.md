# b_hash_route_host
live 20 / placed 17 (deepseek_v4 19 = 17 num_hash_layers 3 + 2 mlp_layer_types, memt_deepseek_v4 1); direction over-wall (claimed); batch bd

## 1. What the code computes
- tf5 DeepseekV4Config: `default_num_hash_layers = 3` (tf5:models/deepseek_v4/configuration_deepseek_v4.py:165); the legacy `num_hash_layers` is popped (:253) and folded into `mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * ...` (:278-281). A config with neither key gets 3 hash layers.
- The MoE block picks its router class from the layer type: `self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"`; `self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)` (tf5:models/deepseek_v4/modeling_deepseek_v4.py:1126-1127).
- HashRouter: `indices = self.tid2eid[input_ids.reshape(-1)]` -- selection a function of the token id only; weights `score_fn(logits).gather(indices)` renormalised (:1115-1121); buffer `tid2eid = nn.Buffer(torch.zeros(vocab_size, top_k), persistent=True)` (:1109). TopKRouter instead has `e_score_correction_bias` (:1080) and `torch.topk(scores + bias)` (:1086). Both have the same `weight` (E x h). Reference V4 code: `self.hash = layer_id < args.n_hash_layers`, `tid2eid = nn.Parameter(...)` (ext:deepseek_v41/hub__deepseek-ai__DeepSeek-V4-Flash-0731/inference/model.py:561-564).
- V4.1 has no hash routing (no `tid2eid` / `n_hash_layers` in ext:deepseek_v41/hub__deepseek-ai__DeepSeek-V4.1-Flash/inference/model.py). MemT subclasses the tf5 V4 model (MemTensor/MemT-1.0-Flash/modeling_memt.py:34, :249) and keeps `tid2eid` (:264).
- The hash layer still holds a content-free path: the routed experts plus `shared_experts(residual)` (tf5 modeling_deepseek_v4.py:1129, :1140): zero routed experts (down = 0) leave the shared expert, a plain dense SwiGLU of width `intermediate_size`.
- p5 relation: `FEATURES["hash_route"] = ("wall", "wall", ...)` (p5:model_graph.py:1710-1713), set only from `num_hash_layers > 0` (:1787-1788): the tf5-format configs (mlp_layer_types "hash_moe": MemT-1.0-Flash, Fx-Bio, DeepSeek-V4-Pro-0.5B) get no hash_route feature -- a target-side over-credit (side finding, 3 repos).

## 2. Classifier change
- reads: `mlp_layer_types` ("hash_moe" entries), `num_hash_layers`, else the deepseek_v4 class default 3.
- new: no FEATURES change for the switch (keep the wall); side fix: `_feats` sets `hash_route` from `mlp_layer_types` too (and from the class default on a deepseek_v4 config with neither key). Optional exact in-shape rule: dims `n_hash` = the hash-layer count, read by moe_ok as dense-only layers (see 3).
- detection:
```python
def detect_b_hash_route_host(cfg, model_type):
    lm = lm_of(cfg)
    mlt = lm.get("mlp_layer_types")
    if isinstance(mlt, list) and any(str(x) == "hash_moe" for x in mlt):
        return "mlp_layer_types"
    if num(lm.get("num_hash_layers")) > 0:
        return "num_hash_layers"
    if model_type in _V4_TYPES and lm.get("num_hash_layers") is None and mlt is None:
        return "num_hash_layers"                   # tf5 default_num_hash_layers = 3
    return None
```
- apply note: p5:model_graph.py `_feats` :1787-1788 (add the mlp_layer_types read). For the in-shape rule: `_moe_sig` / dims_of give `n_hash`, and `moe_why` / `_moe_ok` count a hash host's hash layers among its dense slots (H n_moe -= n_hash, n_dense += n_hash at dense_w = shared width), with FEATURES["hash_route"] host side then "free" (the hash layers never route a target layer). Keep FEATURES as is unless that rule lands.

## 3. Relation rule
- target side: unchanged here (see b_hash_route_target_learned_host).
- host side: (a) the config switch num_hash_layers = 0 / mlp_layer_types all "moe" makes every layer a learned top-k router; (b) in shape, a hash layer plays a DENSE target layer on its shared expert (routed experts zeroed) and never a routed one.
- kind: (a) exact if allowed; (b) exact (`dense_sub_moe_shared`'s zero-router step is not even needed: the routed sum is 0 with down = 0).
- cost: (a) none; (b) the host's MoE-layer count drops by n_hash (3 for V4): target n_moe <= host n_moe - n_hash, target dense layers <= host dense + n_hash (width <= shared intermediate).
- free by the config rules: (a) NOT free. num_hash_layers / mlp_layer_types swap the router module (HashRouter vs TopKRouter) and the persistent buffer set (`tid2eid` dropped, `e_score_correction_bias` added): a feature switch of the selection mechanism, and the state dict changes shape. The "router scoring type is free" rule covers the score function, not hash vs learned selection. Recommend: keep the wall; (b) is the honest exact alternative.
- theorem: `hash_not_content` (PROOFS.md "hash_route: learned target <- hash host": impossible AT a hash layer; "host side a config switch"); (b) needs no new theorem (zero down-projection adds 0; the shared expert is a dense FFN: `dense_sub_moe_shared` row).

## 4. Test fixtures
- flag: deepseek-ai/DeepSeek-V4-Flash-0731, deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4-Pro, deepseek-ai/DeepSeek-V4-Pro-0813 -> "num_hash_layers" (3); MemTensor/MemT-1.0-Flash, endless-frontier/Fx-Bio, inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B -> "mlp_layer_types".
- not flag: HuggingFaceTB/nanowhale-100m-base (V4, num_hash_layers 0); deepseek-ai/DeepSeek-V4.1-Flash (no hash routing in V4.1 code); baidu/ERNIE-5.5-Flash-Base; zai-org/GLM-5.3, poolside/Laguna-XS.2 (mlp_layer_types of dense / sparse, no hash); deepseek-ai/DeepSeek-V3.

## Verdict
real but keep the wall for the config switch (a feature switch that changes the buffer set: not free by the
user's rules). The exact in-shape rule (hash layers = dense-only layers) is sound; alone it opens 0 real pairs, and
together with kv_compress / act_clip host sides 10 shape pairs (V4 hosts -> nanowhale-100m / MemT / Fx-Bio; every
hash host carries hyper_conn, so only hyper_conn targets are in reach). Side finding: 3 tf5-format V4 repos miss
hash_route (target-side over-credit, fewer edges once fixed).
