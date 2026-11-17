# swa_window_plus_one
live 1 / placed 1 (sn38-nanochrono 1); direction over-credit; batch a6_masks

## 1. What the code computes
- Per-layer windows: `config.sliding_window if t == "sliding_attention" else config.max_position_embeddings`
  (nocode sn38-nanochrono/github__chronollm__sn38/modeling_nanochrono.py:135-136). The snapshot has 21 sliding layers
  and 7 full (layer_types), sliding_window 512, max_position_embeddings 2048.
- `_attend`: the mask keeps key j iff `row - col <= window` (:78-79). In decode it keeps the last `window + 1` keys
  (:71-73). So a sliding layer sees window + 1 = 513 keys; HF semantics (and dims_of eff_window) give 512.
- The same inclusive convention (nanochat's left window) is in gaggle (modeling_gaggle.py:72-73) and Motif
  (`sliding_window + 1`, modeling_motif.py:571). Those two are folded into sliding_window_keys_unread (W + 1 there).

## 2. Classifier change
- reads: model_type sn38-nanochrono, sliding_window, layer_types
- new: dims_of eff_window = sliding_window + 1 (via `_normalize` writing `sliding_window = W + 1`)
- detection:
```python
def detect_swa_window_plus_one(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "sn38-nanochrono":
        return None
    sw, mpe = num(lm.get("sliding_window")), num(lm.get("max_position_embeddings"))
    if not sw or not any("sliding" in x for x in layer_types(lm)) or (mpe and sw + 1 >= mpe):
        return None
    return sw + 1
```
- apply note: a table INCLUSIVE_WINDOW_TYPES = {"sn38-nanochrono"} (plus gaggle / motif, handled in their
  `_normalize` blocks), and in `_normalize` (p5:model_graph.py, before the ~L2673 tail): `u["sliding_window"] = sw + 1`.
  Nothing else reads the number in the headline.

## 3. Relation rule
- target side: a sliding layer of reach 513: on the context axis it needs a host window >= 513 (or depth, winReach)
- host side: reach 513
- kind: exact (a number fix)
- cost: none in the headline; the context axis (USE_CONTEXT) compares eff_window
- free by the config rules: the window size is free in principle, but sn38-nanochrono is remote code (no lever)
- theorem: `winReach_iff` / `winLocal_stack` (context axis only)

## 4. Test fixtures
- flag: kingbear329/sn38-r16-c12-step687-alpha05 -> 513
- not flag: castorini/gaggle-... (handled in sliding_window_keys_unread), google/gemma-3-1b-it (HF exclusive band),
  mistralai/Mistral-7B-v0.1

## Verdict
Real but headline-neutral. The window slot (local) and the 3:1 layer kinds are already right. Only dims eff_window moves
by one (512 -> 513), and only USE_CONTEXT reads it, so no headline edge changes. Region: attention (context axis).
