# dsv4_layer_types_schema
live 2 / placed 21 (deepseek_v4 21); direction both; batch a8_mla_dsv4

## 1. What the code computes
- tf5:models/deepseek_v4/configuration_deepseek_v4.py:157-165 fields `layer_types`, `compress_rates` (default {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}, :159), `mlp_layer_types`, `default_num_hash_layers = 3` (:165).
- __post_init__ :250-253 pops the legacy `compress_ratios`, `compress_rate_csa / _hca`, `num_hash_layers`; :266-276 layer_types = explicit > legacy compress_ratios mapped by `_COMPRESS_RATIO_TO_LAYER_TYPE` {0: sliding, 4: CSA, 128: HCA} (:28-32) > default `["heavily_compressed_attention"] * min(n, 2) + CSA / HCA interleave` (every layer compressed); :278-282 mlp_layer_types = explicit > `["hash_moe"] * n_hash + ["moe"] * rest` with n_hash = legacy num_hash_layers, else default_num_hash_layers 3.
- modeling_deepseek_v4.py:777 the attention picks its compressor by `config.layer_types[layer_idx]`; :1126 `self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"`.
- MemT (MemTensor/MemT-1.0-Flash/configuration_memt.py:6,20) subclasses DeepseekV4Config, so the same schema applies (its snapshot config is in the new schema too).
- p5 today: `_feats` sets kv_compress only from `compress_ratios` (p5:model_graph.py:1822-1823) and hash_route only from `num_hash_layers` (:1787-1788). endless-frontier/Fx-Bio and inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B (and MemT-1.0-Flash) carry only layer_types / mlp_layer_types (CSA / HCA layers + 3 hash_moe layers): they get neither feature. (_moe_layer_mask already reads mlp_layer_types for the dense / MoE mask, :810-812, but not 'hash_moe' as a feature.) The other 19 V4 configs carry the legacy keys and are read correctly.

## 2. Classifier change
- reads: layer_types (any "*compressed*" entry), mlp_layer_types ("hash_moe"), legacy compress_ratios / num_hash_layers, the DeepseekV4Config defaults when neither form is present; model types deepseek_v4, memt_deepseek_v4
- new: no new value: kv_compress and hash_route (existing FEATURES) read from the tf5 schema, in `_feats`
- detection:
```python
def detect_dsv4_layer_types_schema(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in _DSV4_SCHEMA_TYPES:          # ("deepseek_v4", "memt_deepseek_v4")
        return None
    feats = []
    lt, mlt = lm.get("layer_types"), lm.get("mlp_layer_types")
    if lm.get("compress_ratios") is None or (isinstance(lt, list) and lt):
        n = num(lm.get("num_hidden_layers"))
        comp = any("compressed" in str(x) for x in lt[:n or None]) if isinstance(lt, list) and lt else n > 0
        if comp:
            feats.append("kv_compress")
    if lm.get("num_hash_layers") is None or (isinstance(mlt, list) and mlt):
        n = num(lm.get("num_hidden_layers"))
        hashed = any(str(x) == "hash_moe" for x in mlt[:n or None]) if isinstance(mlt, list) and mlt else n > 0
        if hashed:
            feats.append("hash_route")
    return "+".join(feats) or None
```
(flags only the configs the legacy reader misses; returns the features the tf5 schedule implies)
- apply note: p5:model_graph.py `_feats`: replace the kv_compress test (:1822) by a helper `_dsv4_layer_types(d)` = layer_types if a non-empty list, else the compress_ratios mapping, else (mt in deepseek_v4 / memt_deepseek_v4) the default all-compressed list; `kv_compress` iff any entry contains "compressed" (or a legacy ratio > 0 for the non-tf5 types deepseek_v41_text / ernie5_5_moe, unchanged). Replace the hash_route test (:1787) by `_num(d.get("num_hash_layers")) > 0 or "hash_moe" in (mlp_layer_types or []) or (mt in ("deepseek_v4", "memt_deepseek_v4") and "num_hash_layers" not in d and not mlp_layer_types)` (default 3). Also `no_compressor` in the dsa test (:1811) should use the same helper (all sliding = no compressor).

## 3. Relation rule
- target side / host side: unchanged rules (kv_compress wall / wall, hash_route wall / wall; FEATURES :1692, :1710); only the reading changes
- kind: wall (both features, both ways)
- cost: none
- free by the config rules: layer_types / mlp_layer_types are feature switches (compressor kind, hash routing), not free
- theorem: the existing rows: `pool_one` (kv_compress, wall not tight on the host side) and `hash_not_content` / `topk_code_select` (hash_route) in PROOFS.md "Round-2 rules"

## 4. Test fixtures
- flag: endless-frontier/Fx-Bio -> "kv_compress+hash_route"; inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B -> "kv_compress+hash_route" (the 2 live); MemTensor/MemT-1.0-Flash -> "kv_compress+hash_route" (memt_deepseek_v4, same tf5 config class, also new schema: a third carrier outside the carriers' type list)
- not flag: deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4-Pro, deepseek-ai/DeepSeek-V4-Flash-0731 (legacy keys, already read); HuggingFaceTB/nanowhale-100m(-base) (legacy keys, all-zero ratios, 0 hash layers); deepseek-ai/DeepSeek-V4.1-Flash (V4.1 code, not this config class); baidu/ERNIE-5.5-Flash-Base

## Verdict
real (both): Fx-Bio, V4-Pro-0.5B and MemT-1.0-Flash gain kv_compress + hash_route. Fewer edges: their edges to / from non-compressing, non-hash models disappear (over-credit). More edges: today the legacy-keyed V4 configs (kv_compress, hash_route, both wall-wall) cannot play them nor be played by them; after the fix the 19 V4 configs and these 3 pair up where dims / MoE allow (over-wall; up to 2 x 19 x 3 pairs, realistically a few tens: Fx-Bio has the V4-Flash dims, the 0.5B toy is played by any wider V4). Net: both, tens of pairs.
