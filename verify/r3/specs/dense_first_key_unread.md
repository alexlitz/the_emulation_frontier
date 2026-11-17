# dense_first_key_unread
live 1 / placed 1 (kohaku 1); direction both; batch a10_moe_ffn

## 1. What the code computes
- `KohakuDecoderLayer.__init__`: `self.mlp = KohakuMLP(hidden, intermediate_size) if layer_idx < config.first_k_dense else KohakuMoE(config)` (KBlueLeaf/TIPO-v2.1-1B-A200M/modeling_kohaku.py:178-186). The class default is `first_k_dense = 1` (configuration_kohaku.py:36, 61).
- `KohakuMoE`: n_routed_experts x moe_intermediate_size SwiGLU experts plus `shared_expert = KohakuMLP(d, moe_intermediate_size * n_shared_experts)`. Sigmoid scores, top-k chosen on `scores + expert_bias` (selection only), renormalised when norm_topk_prob and k > 1, times routed_scaling_factor (modeling_kohaku.py:122-175).
- Snapshot: TIPO-v2.1-1B-A200M has first_k_dense 1, intermediate_size 2048, 64 experts x 384, top-8, 1 shared (384), 16 layers.
- p5 `_dense_first` (p5:model_graph.py:785-788) reads first_k_dense_replace / num_dense_layers / n_dense_first_layers, not first_k_dense. So `_moe_layer_mask` is None, `_moe_sig` gives n_moe 16, n_dense 0, and ffn_units counts 16 MoE layers.

## 2. Classifier change
- reads: `first_k_dense` (kohaku; class default 1).
- new: `_dense_first` key (feeds the moe_sig n_moe / n_dense / dense_w and dims ffn_units).
- detection:
```python
def detect_dense_first_key_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "kohaku":
        k = lm.get("first_k_dense", 1)
        return num(k) if num(k) > 0 else None
    return None
```
- apply note: `_dense_first` (~785): `... or _num(d.get("n_dense_first_layers")) or (_num(d.get("first_k_dense", 1)) if str(d.get("model_type") or "").lower() == "kohaku" else 0)`. `_moe_layer_mask` (~845) and the ffn_units mask path (~3243) then pick it up unchanged (freq 1). Side note, not this finding: Kohaku's `expert_bias` is a selection-only bias that `_router_flags` does not see. Add `kohaku` to ROUTER_BIAS_TYPES (~1926) in the same apply.

## 3. Relation rule
- target side: layer 0 is a dense SwiGLU of width 2048. Its host needs a dense layer >= 2048, or a MoE layer used as dense with k*ew + shared >= 2048 (8 x 384 + 384 = 3456 on a Kohaku-like host). Today the target asks for 16 MoE layers and no dense one.
- host side: the host has 15 MoE layers plus 1 dense layer (2048), not 16 MoE layers. A MoE target with 16 MoE layers no longer fits a 16-layer Kohaku host.
- kind: exact (layer-kind accounting, moe_why `layers`).
- cost: depth by layer kind.
- free by the config rules: first_k_dense changes the module per layer (a feature switch / shape), so it is not free.
- theorem: `dense_sub_moe_shared` (dense target layer on a host MoE layer, PROVED exact); the layer-count rule is capacity.

## 4. Test fixtures
- flag: KBlueLeaf/TIPO-v2.1-1B-A200M -> 1 (the only kohaku point).
- not flag: Qwen/Qwen3-30B-A3B (all-MoE), deepseek-ai/DeepSeek-V4-Flash, moonshotai/Kimi-Linear-48B-A3B-Instruct (dense-first read through other keys or none).
- scan: exactly the TIPO repo.

## Verdict
real, small. Both directions for 1 point. As a target it needs a host dense / MoE-as-dense slot for layer 0, which most MoE hosts with k*ew + shared >= 2048 provide, so it loses almost nothing. As a host it loses MoE targets needing 16 MoE layers (and its ffn_units drop). Expect a few edges either way.
