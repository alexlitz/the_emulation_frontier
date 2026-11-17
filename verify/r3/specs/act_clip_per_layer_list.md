# act_clip_per_layer_list
live 1 / placed 2 (bailing_hybrid 1, ming_v3_asr 1); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
- `BailingMoeV3SparseMoeBlock._get_swiglu_limit(limit_list, layer_idx)`: `limit_list[layer_idx] or None` (0 means no clamp) (inclusionAI/Ling-3.1-flash/modeling_bailing_moe_v3.py:420-424). The block reads `expert_swiglu_limit_list` for the routed experts and `share_expert_swiglu_limit_list` for the shared expert (:429-449).
- `BailingMoeV3MLP.forward`: `gate = act(gate_proj x)`; if a limit is set, `gate.clamp(max=L)` and `up.clamp(-L, L)` (:342-348). This is the post-activation order (see act_clip_order: it converts exactly to the pre order).
- Ming-3.0-asr-tiny (inclusionAI/Ming-3.0-asr-tiny/modeling_ming_v3_asr.py BailingMoeV3MLP / _setup_experts) uses the same code. Its lists are null, so it does not clamp: placed but correctly not live.
- Snapshot: Ling-3.1-flash and Baekpica/Ling-3.1-flash-1019-VQ-GGUF clamp at 10 on the last 7 routed / 8 shared layers. TensorFold/Ling-3.0-flash-oQ4 / -oQ5 clamp at 4 (routed) and 5 / 7 (shared) on the last 7-8 layers. The carrier list names only Ling-3.1-flash, but the other three take the same path by config.
- p5 `_feats` (p5:model_graph.py:1826) reads only swiglu_limit / swiglu_limits / swiglu_limits_shared, so none of the four gets `act_clip`.

## 2. Classifier change
- reads: `expert_swiglu_limit_list`, `share_expert_swiglu_limit_list` (bailing_hybrid V3 code).
- new: the `act_clip` FEATURES key (existing), in `_feats`.
- detection:
```python
def detect_act_clip_per_layer_list(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "bailing_hybrid":
        return None
    vals = []
    for k in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"):
        v = lm.get(k)
        if isinstance(v, (list, tuple)):
            vals += [fnum(x) for x in v if fnum(x)]
    return num(max(vals)) if vals else None
```
- apply note: `_feats` (~1826): `lim = d.get("swiglu_limit") or d.get("swiglu_limits") or d.get("swiglu_limits_shared") or d.get("expert_swiglu_limit_list") or d.get("share_expert_swiglu_limit_list")`. `_num` of a list is its max, so an all-zero list stays off. No type gate is needed: only BailingMoeV3 configs carry the keys.

## 3. Relation rule
- target side: a clamped SwiGLU on some layers. An unclamped (analytic) host cannot form a clamp on its up line, so a clamped host is needed (FEATURES act_clip, wall).
- host side: the existing act_clip host wall (not tight: a limit is a config value and can be raised above every activation of a bounded context). The Ling-3 lists are per-layer config.
- kind: wall (target); wall, not tight (host) -- unchanged rule, newly applied.
- cost: none.
- free by the config rules: the limits are caps (free values); the clamp code path is a feature.
- theorem: target `analytic_ne_clamp`, `act_clip_unclamped_fails` (PROVED impossible for analytic hosts); host `act_clip_regime`, `act_clip_not_exact`.

## 4. Test fixtures
- flag: inclusionAI/Ling-3.1-flash -> 10, Baekpica/Ling-3.1-flash-1019-VQ-GGUF -> 10, TensorFold/Ling-3.0-flash-oQ5 -> 7, TensorFold/Ling-3.0-flash-oQ4 -> 7.
- not flag: rapid-mlx/Ling-3.0-tiny-MLX-4bit (bailing_hybrid, no lists), Edge0/Edge0-8B-A1B-preview (list keys null), stepfun-ai/Step-3.5-Flash (clamped through swiglu_limits, already read), deepseek-ai/DeepSeek-V4-Flash.
- scan: exactly the 4 Ling-3.0 / 3.1-flash repos.

## Verdict
real. Fewer edges: the 4 Ling-3 flash points gain act_clip. As targets they lose every unclamped host (they keep clamped hosts: DeepSeek-V4 / GLM-5-Next / Step-3.5 / Hy4, subject to their other walls). As hosts they lose every unclamped target. That is 4 points; the per-layer pattern caveat of act_clip_order applies to the clamped hosts they keep.
