# layer_types_all_full_window_ignored
live 1 / placed 1 (vaultgemma 1); direction both; batch a6_masks

The scan also flags udgamlabs/Vemana-Base (gemma4_text) and EssentialAI/rnj-1.5-instruct (gemma3_text). Both save an
all-'full_attention' layer_types with a binding sliding_window (512 < 4096 and similar), and their code gates the window
the same way.

## 1. What the code computes
- VaultGemmaConfig defaults layer_types to alternating sliding / full only when layer_types is None (tf5
  models/vaultgemma/configuration_vaultgemma.py:92-96). vaultgemma-1b saves 26 x 'full_attention', sliding_window 512,
  max_position_embeddings 1024.
- The attention applies the window only on sliding layers: `self.sliding_window = config.sliding_window if
  self.layer_type == "sliding_attention" else None` (tf5 models/vaultgemma/modeling_vaultgemma.py:189). The mask
  mapping is keyed by layer_types (:431, :441). Every layer is full causal attention, and sliding_window is dead config.
- The same gate is in gemma2 (tf5 gemma2/modeling_gemma2.py:246), gemma3 (gemma3/modeling_gemma3.py:335) and gemma4
  (gemma4/modeling_gemma4.py:1170-1171).

## 2. Classifier change
- reads: layer_types (list), sliding_window, max_position_embeddings, model_type in LAYER_TYPED_WINDOW_TYPES
- new: window = "full" in classify (and eff_window = max_context in dims_of) when layer_types names no sliding layer;
  new table LAYER_TYPED_WINDOW_TYPES = {gemma2, gemma3, gemma3_text, gemma3n_text, gemma4, gemma4_text,
  gemma4_unified_text, vaultgemma, diffusion_gemma_text}
- detection:
```python
def detect_layer_types_all_full_window_ignored(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in LAYER_TYPED_WINDOW_TYPES:
        return None
    lt = layer_types(lm)
    sw, mpe = num(lm.get("sliding_window")), num(lm.get("max_position_embeddings"))
    if not lt or not sw or (mpe and sw >= mpe) or any("sliding" in x for x in lt):
        return None
    return "full"
```
- apply note: in `_normalize` (p5:model_graph.py ~L2673, the window tail), add: if mt in LAYER_TYPED_WINDOW_TYPES and
  layer_types is a list with no 'sliding' entry, then `u.update(sliding_window=None, use_sliding_window=False)`. Do not
  make it generic. DeepSeek-V4's 'compressed_sparse_attention' layers and cyrah's 'conv' + 'full_attention' list also
  have no 'sliding' entry, but their code keeps a window (inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B,
  Lilgarca/my-model-v01 in the scan).

## 3. Relation rule
- target side: an all-full target (26 FU)
- host side: an all-full host
- kind: exact (label fix only)
- cost: none
- free by the config rules: the window is free for vaultgemma / gemma3_text / gemma4_text (WINDOW_FREE_TYPES)
- theorem: none needed (the code never applies the window); `window_ge_full` covers the lever reading

## 4. Test fixtures
- flag: google/vaultgemma-1b -> "full"; udgamlabs/Vemana-Base -> "full"; EssentialAI/rnj-1.5-instruct -> "full"
- not flag: google/gemma-2-2b, google/gemma-3-1b-it (sliding layers listed or derived);
  inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B (compressed layers keep a window); Lilgarca/my-model-v01 (cyrah,
  unlisted code); Qwen/Qwen3-0.6B (no binding window)

## Verdict
Already handled in effect by round2-p5. `_attn_counts` reads layer_types first (~L1351), so the kinds are already
26 FU. classify's window=local only switches on win_free (all three types are window-free), and win_free on an all-FU
point changes nothing (kind_match host_full ~L3646 moves LU only). So the headline is unchanged. The fix corrects the
window label and dims eff_window (512 -> 1024, context axis only) for 3 models.
