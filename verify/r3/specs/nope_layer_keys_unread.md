# nope_layer_keys_unread
live 5 / placed 5 (domynedge 1, kolibri1 1, muse_glimmer 1, muse_glimmer_text 2); direction both; batch a5_positions

## 1. What the code computes
- Muse-Glimmer:
  - `tf5:models/muse_glimmer/modeling_muse_glimmer.py:521-522`: `position_embeddings if self.config.layer_rope_theta[i] else None`. NoPE where layer_rope_theta[i] == 0.
  - `configuration_muse_glimmer.py:147,161-165`: when the key is absent, `layer_rope_theta = [0 if (n-1-i) % 4 == 0 else rope_theta]`. That is NoPE every 4th layer counted from the last, by CLASS DEFAULT.
- Muse-Spark: same code in `tfmain:models/muse_spark/configuration_muse_spark.py:141,175-178` and `modeling_muse_spark.py:666-667`.
- Domyn-Edge `domyn/Domyn-Edge-Base/modeling_domynedge.py:208,242-244`: `use_rope = config.position_layers[layer_idx] == "rope"`, and RoPE is applied only then. The config default (`configuration_domynedge.py:34`) is ["nope","nope"] + ["rope","rope","rope","nope"]·8 + ["nope","nope"]. The snapshot gives 12 NoPE full layers out of 36.
- Kolibri-1 `tfmain:models/kolibri1/modeling_kolibri1.py:150-152,184`: Exaone4-style attention, `if self.sliding_window is None or self.is_sliding:` apply RoPE. With sliding_window 513 (class default `modular_kolibri1.py:77`, positive-validated `:100-101`), the full-attention layers (every 5th) are NoPE.
- round2-p5:
  - `_nope_layer_mix` (p5:model_graph.py:1290-1298) reads no_rope_layers / no_rope_layer_interval / no_rope_layer_indices / per-type None rope entries.
  - `_normalize` (`:2649-2652`) synthesises no_rope_layers for afmoe / cohere2 / exaone4 / exaone4_5 / exaone_moe, but not kolibri1.
  - Checked with p5 classify: wayfinder-glimmer, Muse-Spark-Open, Domyn-Edge-Base and Kolibri-1 get feats without nope_layers and rope_sched (hd,). The MLX Muse-Glimmer repo carries no_rope_layers and is already flagged.
  - Not fixed in p5.

## 2. Classifier change
- reads: `layer_rope_theta` (with the class default when absent) for muse_glimmer(_text) / muse_spark(_text); `position_layers` (domynedge); `layer_types` + `sliding_window` (kolibri1).
- new: `no_rope_layers` written in `_normalize`. FEATURES `nope_layers` and the negative rope_sched entry then follow from the existing `_nope_layer_mix` / `_rope_schedules`.
- detection:
```python
MUSE_TYPES = ("muse_glimmer", "muse_glimmer_text", "muse_spark", "muse_spark_text")

def detect_nope_layer_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    n = num(lm.get("num_hidden_layers"))
    if model_type in MUSE_TYPES:
        lrt = lm.get("layer_rope_theta")
        if isinstance(lrt, list) and lrt:
            out = [int(bool(x)) for x in lrt]
        elif isinstance(lm.get("no_rope_layers"), list) and lm["no_rope_layers"]:
            out = [int(bool(x)) for x in lm["no_rope_layers"]]
        elif n:
            out = [0 if (n - 1 - i) % 4 == 0 else 1 for i in range(n)]
        else:
            return None
        return out if 0 in out and 1 in out else None
    if model_type == "domynedge":
        pl = lm.get("position_layers")
        if isinstance(pl, list) and pl:
            out = [int(str(x) == "rope") for x in pl]
            return out if 0 in out and 1 in out else None
        return None
    if model_type == "kolibri1":
        lt = layer_types(lm)
        sw = lm.get("sliding_window", 513)
        if sw and any("sliding" in x for x in lt) and any("full" in x for x in lt):
            return [int("sliding" in x) for x in lt]
    return None
```
- apply note:
  - In `_normalize`, after the llama4 block: for MUSE_TYPES without no_rope_layers, set `u["no_rope_layers"]` from layer_rope_theta, or from the class-default formula.
  - For domynedge, set it from position_layers.
  - Add "kolibri1" to the afmoe / cohere2 / exaone4 tuple at `:2649`, with its sliding_window default 513 (like afmoe's, it may be omitted).

## 3. Relation rule
- target side: unchanged rules for nope_layers (wall on hosts without it), plus the rot gate's zero-frequency slots.
- host side: these hosts already carry the nope_layers wall once flagged. They cannot play plain RoPE targets (layer alignment), and they play mixed targets through rope_fits.
- kind: wall (both ways, as FEATURES nope_layers).
- cost: none.
- free by the config rules: Kolibri's sliding_window is free (WINDOW_FREE_TYPES), but an int window keeps the RoPE-off on full layers (p5's window audit note), so it does not remove the mix. layer_rope_theta / position_layers are per-layer switches (not free). Muse's theta value is free only where nonzero.
- theorem: needs theorem (PROOFS.md "nope_layers (both), multi_rope target" is OPEN; truncation measured not payable).

## 4. Test fixtures
- flag: each -> the no_rope_layers list (true in the fixture)
  - RadixArk/Muse-Glimmer-q4-MLX (already carries no_rope_layers; the code fact still holds)
  - hweller1/wayfinder-glimmer-30b, unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit (layer_rope_theta zeros)
  - staccs/Muse-Spark-Open-816B-A42B-lecore-mlx, Interchained/Muse-Spark-Open-816B-A42B-GGUF (muse_spark_text: live, not in placed)
  - domyn/Domyn-Edge-Base (position_layers)
  - eins78/Kolibri-1-mlx-mixed-4-8-bit (sliding_window 513 + full layers)
- not flag:
  - LGAI-EXAONE/EXAONE-4.0-32B (same Exaone4 code path, already normalised by p5 under exaone4)
  - HuggingFaceTB/SmolLM3-3B (no_rope_layers, read)
  - google/gemma-3-1b-it

## Verdict
Real, both directions.
- 7 points gain the nope_layers feature: 5 placed plus 2 Muse-Spark.
- They lose every edge to plain-RoPE hosts and targets, and keep only mixed (nope_layers) partners whose tables fit.
- Fewer edges: roughly the current edge set of these 7 points (Muse is wide and has post_norm, so its partner set is already narrow).
