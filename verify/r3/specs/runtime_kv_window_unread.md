# runtime_kv_window_unread
live 2 / placed 2 (needle 2); direction over-credit; batch a6_masks

Really live: 1 (needle3, kv_window 256). needle2 has no kv_window key; its sliding_window is already 256.

## 1. What the code computes
- `TransformerConfig.kv_window: int = 0` (nocode needle/gh__cactus-compute__cactus/config.py:45).
- The reference forward never reads it. The JAX model masks with `cfg.sliding_window` and `cfg.global_layers` only
  (architecture.py:469-475; the MLX port is the same, architecture_mlx.py:243-259).
- `effective_kv_window(config) = min(kv_budget_window(config), config.kv_window)` (architecture.py:821-823). build.py:56
  and export.py:473-496 write it into the .cact header for the C++ runtime, which is not in the fetched code. Its
  docstring calls it "the sliding-window width the model was trained with ... must be RUN at it" (export.py:33-37).
- So the deployed engine likely caps every layer's reach at 256 keys, the global layers too. The Python reference
  computes sliding_window 1024 on 16 layers and full attention on the 4 global ones. The C++ semantics are
  unverifiable (not fetched).

## 2. Classifier change
- reads: model_type needle, kv_window, sliding_window
- new: none in the headline. At most dims_of eff_window = min(sliding_window, kv_window) (context axis), recorded as a
  run condition
- detection:
```python
def detect_runtime_kv_window_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "needle":
        return None
    kw, sw = num(lm.get("kv_window")), num(lm.get("sliding_window"))
    if not kw:
        return None
    return min(sw, kw) if sw else kw
```
- apply note: no change to classify / _attn_counts. If recorded, add it in dims_of next to eff_window (p5:model_graph.py
  ~L3236) as a separate run-condition field (e.g. `run_kv_window`). Do not let it move eff_window: the relation models
  the reference forward.

## 3. Relation rule
- target side: none (the reference function has no kv_window)
- host side: none
- kind: n/a
- cost: none
- free by the config rules: kv_window is a window size (free), but it is a deployment cap, not a model switch
- theorem: n/a (`winReach_iff` would price it on the context axis)

## 4. Test fixtures
- flag: Cactus-Compute/needle3 -> 256
- not flag: Cactus-Compute/needle2 (no kv_window), mistralai/Mistral-7B-v0.1

## Verdict
Non-issue for the relation. The reference forward (architecture.py:469-475) never reads kv_window. It is a C++-runtime
KV budget, a run condition like the eps lever's dtype, and that code is not available to verify. It could only touch
dims eff_window (context axis). No headline edge moves.
