# hierarchical_moe_unread
live 1 / placed 1 (meditor2 1); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
- Config: `total_sub_experts = num_macro_domains * num_sub_experts_per_domain` (Yo-Purple/Meditor-2-3.4B-A1.8B-HMoE/configuration_meditor2.py:28-31, 59-63). Snapshot: 3 x 5 = 15 sub-experts of width sub_intermediate_size 768, shared_intermediate_size 8192, hidden 2048, 24 layers.
- `Meditor2DecoderLayer`: `self.mlp = Meditor2HierarchicalMoEMLP(config)` on every layer (modeling_meditor2.py:308).
- `Meditor2HierarchicalMoEMLP` (modeling_meditor2.py:194-230): shared SwiGLU (8192), `macro_router = Linear(hidden, 3)`, 3 `micro_routers` Linear(hidden, 5), stacked routed SwiGLU weights (15 x 768), and `routed_scale = Parameter(0.5)`.
- forward (modeling_meditor2.py:232-297): `d = argmax softmax(macro_router x)`, then `e = argmax softmax(micro_router_d x)`, and `out = shared(x) + routed_scale * expert_{d,e}(x)`. It is a two-level hard top-1, and the expert's weight is the constant routed_scale (the probabilities never multiply the output).
- p5 reads none of these keys. classify gives ffn = glu with experts 0 and inter = shared_intermediate_size 8192 (a dense model).

## 2. Classifier change
- reads: `num_macro_domains`, `num_sub_experts_per_domain`, `sub_intermediate_size`, `shared_intermediate_size` (meditor2; class defaults 3 / 5 / 768 / 8192).
- new: classify ffn = moe_shared (E = D*S, k = 1). dims_of: ew = sub_intermediate_size, shared = shared_intermediate_size. MOE_SIG `score` = "hier", `score_cap` = ("hier", "softmax"), renorm "R" (constant weight = a top-1 renormalised weight 1, scale folded into down_proj).
- detection:
```python
def detect_hierarchical_moe_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "meditor2":
        return None
    d = num(lm.get("num_macro_domains", 3))
    s = num(lm.get("num_sub_experts_per_domain", 5))
    w = num(lm.get("sub_intermediate_size", 768))
    return [d, s, w] if d * s > 1 else None
```
- apply note: classify (~2264, `n_exp` chain): `or (num_macro_domains * num_sub_experts_per_domain if mt == "meditor2" else None)`. `_TOPK_ALT_KEYS` / dims_of k: top-1 for meditor2. dims_of (~3143): ew `sub_intermediate_size`. `shared_intermediate_size` is already in the shared chain and in `_has_shared_expert`. `_moe_sig` (~930): `score = "hier"`, `score_cap = ("hier", "softmax")`, `renorm = renorm_cap = "R"` for meditor2. In `moe_why` (~995): `if H["score"] == "hier" and T["score"] == "softmax" and Et > S_h: wall` (S_h = sub-experts per domain; needs dims `hier_sub`), and `if T["score"] == "hier" and H["score"] != "hier": wall`. Put that check before the `Et and not Eh` (MoE <- dense) branch, so a dense host does not get the flat top-1 select rule for a hierarchical target.

## 3. Relation rule
- target side: a two-level argmax partitions the input space into regions (macro cone) intersected with (micro cone). A flat top-1 argmax of 15 linear scores generally cannot reproduce that partition (a flat argmax of linear scores is one polyhedral fan; the hierarchical one has independent fans per domain). So it is a wall unless the host is hierarchical with the same (D, S) shape, or larger D_h >= D_t, S_h >= S_t (extra slots kept out by zero rows... argmax has no selection bias, so it is exact only at equal D and S).
- host side: a Meditor host plays a flat softmax top-1 renormalised target (weight 1) with E_t <= S_h. Zero macro_router rows make all macro logits equal, so torch.argmax picks domain 0 deterministically. micro_router_0 is then a flat top-1 over S_h = 5 experts, and the weight is the constant routed_scale, which folds into down_proj. Dense targets use the shared 8192 (`moe_sub_moe_shared`).
- kind: target wall; host exact (E_t <= 5, top-1 R targets; dense via shared).
- cost: none.
- free by the config rules: none (domain / sub-expert counts are module shapes).
- theorem: host needs theorem `argmax_tie_first_select` (equal logits give index 0, a code fact about torch.argmax) with `moeR_topk_unique` (flat top-1 duplication) and `moe_sub_moe_shared`. Target wall needs theorem `hier_argmax_not_flat`: there is a 2-level argmax partition (D = 2, S = 2) that no flat argmax of 4 linear scores induces.

## 4. Test fixtures
- flag: Yo-Purple/Meditor-2-3.4B-A1.8B-HMoE -> [3, 5, 768].
- not flag: Qwen/Qwen3-30B-A3B (flat MoE), Qwen/Qwen1.5-MoE-A2.7B (flat MoE + shared), meta-llama/Llama-3.2-3B (dense).
- scan: exactly the Meditor-2 repo.

## Verdict
real. Fewer edges: Meditor-2 is classified as a dense GLU-8192 model. As a target it currently has every dense host with ffn >= 8192 (hundreds of edges), and all of them disappear. As a host it loses dense targets wider than 8192 (it keeps <= 8192 via the shared expert, plus flat top-1 MoE targets with <= 5 experts). Roughly 1 point, with on the order of hundreds of target-side edges dropped.
