# k_eq_v_by_code
live 2 / placed 2 (diffusion_gemma_text 2); direction both; batch a7_attn_exotic

## 1. What the code computes
- tf5:models/diffusion_gemma/modular_diffusion_gemma.py:101 the config class DELETES `attention_k_eq_v`
  (`= AttributeError()`): K = V is not a config value here.
- :258-262 (decoder) and :360-364 (encoder): `v_proj = Linear(...) if self.is_sliding else None` -- every
  non-sliding layer has no v_proj; :293 / :395 `value_states = v_proj(x) if v_proj is not None else key_states`
  (the raw k_proj output, before k_norm / RoPE at :295-296), then `v_norm` (weight-free, :270, :299). This is
  Gemma-4's attention_k_eq_v path fixed in code.
- Snapshot: layer_types mixes sliding and full layers; global_head_dim 512, num_global_key_value_heads 2.
- p5:model_graph.py:1063-1066 `_k_eq_v(d) = bool(d.get("attention_k_eq_v"))` -> False for both repos, so the
  layer-kind count (:1440-1455) marks the full layers FU (own V) instead of FT, and `_attn_variants` (:1088-1095)
  tags the global shape tied=False (it already reads num_global_key_value_heads, because the key is absent).

## 2. Classifier change
- reads: model_type diffusion_gemma / diffusion_gemma_text, layer_types
- new: table entry K_EQ_V_BY_CODE = {"diffusion_gemma", "diffusion_gemma_text"} read by `_k_eq_v`
- detection:
```python
def detect_k_eq_v_by_code(cfg, model_type):
    lm = lm_of(cfg)
    if not model_type.startswith("diffusion_gemma"):
        return None
    lt = layer_types(lm)
    return True if (not lt or any(x != "sliding_attention" for x in lt)) else None
```
- apply note: p5:model_graph.py `_k_eq_v` (1063): `return bool(d.get("attention_k_eq_v")) or
  str(d.get("model_type") or "").lower() in K_EQ_V_BY_CODE`; no other change (the kind counter and _attn_variants
  read `_k_eq_v`).

## 3. Relation rule
- target side: its full layers become FT: any FU or FT host layer plays them (KIND_REACH: own-V reaches K = V by
  v_proj := k_proj) -- unchanged hosts still qualify, so no target-side loss; FT also lets K = V hosts (Gemma-4
  attention_k_eq_v, DeepSeek-V4) play its full layers, which today they cannot (FU target layer on an FT host).
- host side: its full layers can no longer play FU target layers (a K = V layer cannot give a target its own V).
- kind: exact (both directions, existing KIND_REACH rule)
- cost: none
- free by the config rules: not a config value here (code-fixed); for Gemma-4 it is a feature switch
- theorem: existing layer-kind matching (PROOFS "Every allowed rule", own-V plays K = V: v_proj := k_proj, exact)

## 4. Test fixtures
- flag: nvidia/diffusiongemma-26B-A4B-it-NVFP4, yifanyu/DiffusionGemma-26B-A4B-it-ORL -> True
- not flag: nvidia/Gemma-4-26B-A4B-NVFP4 (gemma4_text, K = V via the config key: already read),
  google/gemma-4-E2B-it-assistant (gemma4_text, attention_k_eq_v False), Lilgarca/my-model-v01 (cyrah, key False)

## Verdict
real, small, both ways: the 2 DiffusionGemma repos gain K = V hosts for their full layers (Gemma-4 26B / 31B K = V
hosts, if diffusion gen and every other slot fit: likely 0-few edges since gen=diffusion walls AR hosts) and lose
hosting FU-layer targets (few: diffusion targets with own-V full layers, e.g. LLaDA / Dream classes).
