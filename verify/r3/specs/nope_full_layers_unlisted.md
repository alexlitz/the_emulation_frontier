# nope_full_layers_unlisted
live 7 / placed 7 (grug_moe 7); direction over-credit; batch a5_positions

## 1. What the code computes
- Grug-MoE `ext:grug_moe/gh__marin-community__marin/skyrl_train_models_grug_moe.py`:
  - `:50`: `GRUG_DEFAULT_GLOBAL_EVERY = 4`.
  - `:91-98`: `grug_long_layer_flags(n)` is long for `(idx+1) % 4 == 0 or idx == n-1`.
  - `:979`: it is called with `len(self.layers)` and the default global_every.
- Config `:193,286-287,355`: `disable_long_rope` defaults to True, any other value raises, and it is then forced True.
- Attention `:730-731`: `if not is_long: q, k = _apply_half_rope(...)`. Long (full-attention) layers get no RoPE; sliding layers get half RoPE (64 of 128 dims, `:680-700`; see partial_rope_keys_unread).
- With 26 layers, layers 3, 7, 11, 15, 19, 23 and 25 are NoPE: 7 of 26.
- Snapshot: all 7 grug_moe configs (4 omit disable_long_rope, 3 set it true). None carries no_rope_layers or layer_types.
- round2-p5: `_nope_layer_mix` (p5:model_graph.py:1290-1298) sees nothing. Checked: Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 -> pos rope, feats '', rope_sched (128,). Not fixed in p5.

## 2. Classifier change
- reads: model_type grug_moe, num_hidden_layers (the layer rule is code-fixed).
- new: `no_rope_layers` (1 = RoPE on the sliding layers, 0 on the long ones) written in `_normalize`. This yields FEATURES `nope_layers` and a negative rope_sched entry (-64, matching the 64-dim rotating table once partial_rope_keys_unread lands).
- detection:
```python
def detect_nope_full_layers_unlisted(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "grug_moe":
        return None
    n = num(lm.get("num_hidden_layers"))
    if not n:
        return None
    return [0 if ((i + 1) % 4 == 0 or i == n - 1) else 1 for i in range(n)]
```
- apply note:
  - In `_normalize`, add `if mt == "grug_moe" and not g("no_rope_layers"): u["no_rope_layers"] = [...]` (formula above), beside the llama4 block.
  - Grug's window is the per-layer sliding window (sliding_window 2048 on non-long layers). If layer_types is needed for window=local / kinds, also set `u["layer_types"]` = full_attention on long layers, sliding_attention elsewhere.

## 3. Relation rule
- target side: nope_layers wall on hosts without a RoPE / NoPE mix. Mixed hosts need their zero-frequency room for 64 or 128 dims (rope_fits) beside the 64-dim rotating table.
- host side: nope_layers wall (a mixed host's NoPE layers cannot play rotating layers).
- kind: wall (both ways).
- cost: none.
- free by the config rules: disable_long_rope is forced True in code (not even a switch). Nothing is free.
- theorem: needs theorem (PROOFS.md "nope_layers (both)" OPEN).

## 4. Test fixtures
- flag: every live repo -> the 26-entry list [1,1,1,0, ..., 1,0] (the first one pinned exactly, the rest true)
  - open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21
  - open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom
  - open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05
  - open-athena/Grug-67B-A2B-GLM53-RLVR-SFT-2026.09.23
  - open-athena/Snowball-67B-A2B-Identity-PathB-A13-2026.10.30
  - open-athena/Snowball-67B-A2B-Identity-PathB-A3-2026.10.30
  - open-athena/snowball-67b-a2b-base-262k-qk175
- not flag:
  - HuggingFaceTB/SmolLM3-3B (NoPE layers already read from no_rope_layers)
  - google/gemma-3-1b-it

## Verdict
Real (over-credit).
- 7 Grug points gain nope_layers and lose all plain-RoPE partners: fewer edges, roughly their current edge set (MoE 67B, so mostly as targets on large MoE hosts).
- Apply together with partial_rope_keys_unread (the same 7 points, rope_sched (64, -64)).
