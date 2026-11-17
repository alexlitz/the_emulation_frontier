# kv_compress_variant
live 24 / placed 24 (deepseek_v41_text 24); direction over-credit; batch a8_mla_dsv4

## 1. What the code computes
- V4 (tf5:models/deepseek_v4/modeling_deepseek_v4.py): HCA compressor :365-393, one pool per `compress_rates["heavily_compressed_attention"]` (128) non-overlapping tokens with a learned `position_bias` (compress_rate x head_dim, :393); CSA compressor :592-621, kv_proj / gate_proj of 2 x head_dim and position_bias (compress_rate x 2 head_dim) giving overlapping windows of 2 x 4 tokens at stride 4 (:253-269 "effective width 2 * compress_rate_csa, stride compress_rate_csa"). configuration_deepseek_v4.py:28-32 maps legacy compress_ratios {0, 4, 128} only.
- V4.1 (ext:deepseek_v41/hub__deepseek-ai__DeepSeek-V4.1-Flash/inference/model.py): `Compressor` :429-485: ratio 1 -> `return self.norm(self.wkv(x))` (:461-462: a per-token second key, no pooling, no gate); ratio > 1 -> wkv / wgate of head_dim (no 2x overlap halves), `kv = (kv * softmax(score)).sum` over non-overlapping groups of `ratio` tokens (:473-475), NO position bias.
- V4.1 `Attention.__init__` :619-660: "compress_ratio > 0 does not mean the layer compresses its own KV: only kv_source_layers do, the rest read that same cache"; `_compress_kv` :740-764 reads `shared_attn.compress_kv` written by the source layer. Snapshot: compress_ratios = 2 on layers 2-19, 1 on 20-39, kv_source_layer_ids [2, 8, 14, 20].
- ERNIE-5.5 (ext:ernie5_5_moe/.../attention.py:155) `Ernie5_5Compressor(DeepseekCompressor)`: vLLM's DeepSeek-V4 compressor at ratio 128 (C128), i.e. the V4 code.
- MemT-1.0 (configuration_memt.py:6,20 subclasses tf5 DeepseekV4Config; its layers are tf5 V4's).
- p5 today: `kv_compress` is one boolean from `compress_ratios` (p5:model_graph.py:1822-1823), wall both ways (FEATURES :1692). V4 and V4.1 are therefore credited to each other whenever the other features allow it (V4.1 host -> V4 target: engram is free on the host).

## 2. Classifier change
- reads: model_type (deepseek_v4 / memt_deepseek_v4 / ernie5_5_moe -> V4 code; deepseek_v41 / deepseek_v41_text -> V4.1 code), compress_ratios / layer_types (any compressed layer)
- new: FEATURES key `"kv_compress_v41"` beside `"kv_compress"` (V4.1's compressor), in `_feats` (equivalently a variant value of the feature: the two are walled from each other)
- detection:
```python
def detect_kv_compress_variant(cfg, model_type):
    lm = lm_of(cfg)
    cr = lm.get("compress_ratios")
    if model_type in ("deepseek_v41", "deepseek_v41_text"):
        return "dsv41" if isinstance(cr, list) and any(num(x) > 0 for x in cr) else None
    if model_type in _DSV4_SCHEMA_TYPES:                 # ("deepseek_v4", "memt_deepseek_v4")
        return "dsv4" if _dsv4_has_compressed(lm) else None
    if model_type == "ernie5_5_moe":
        return "dsv4" if isinstance(cr, list) and any(num(x) > 1 for x in cr) else None
    return None
```
- apply note: p5:model_graph.py `_feats` (:1822-1823): replace `f.add("kv_compress")` by `f.add("kv_compress_v41" if mt in ("deepseek_v41", "deepseek_v41_text") else "kv_compress")`; add to FEATURES (after :1694) `"kv_compress_v41": ("wall", "wall", "DeepSeek-V4.1 compressed KV: non-overlapping ratio-2 softmax pools without a position bias, ratio-1 per-token norm(wkv x) keys, pools built on kv_source_layers and read by later layers -- not V4's CSA / HCA pools")`. Keep "kv_compress" for V4 / MemT / ERNIE (ERNIE uses V4's compressor; its ratio-128-only schedule is a schedule, not a code difference). Pair with dsv4_layer_types_schema (V4 configs in the layer_types schema).

## 3. Relation rule
- target side: a V4.1 target needs a V4.1 host (kv_compress_v41); a V4 target needs a V4-code host (kv_compress)
- host side: a compressing host plays no non-compressing target and no target of the other variant
- kind: wall (both ways, both variants)
- cost: none
- free by the config rules: compress ratios per layer are code-path selectors (V4: layer_types; not free); a V4 host setting ratio 0 everywhere is the existing "not tight" note (`pool_one`)
- theorem: needs theorem: "a non-overlapping, bias-free softmax pool of r tokens (V4.1) equals no overlapping-window pool with a learned position bias of ratio 4 or 128 (V4) on all inputs, and vice versa"; the wall to plain attention is the existing kv_compress row (PROOFS.md "kv_compress: compress host -> plain target", `pool_one`, wall not tight on the host side)

## 4. Test fixtures
- flag (dsv41): deepseek-ai/DeepSeek-V4.1-Flash, deepseek-ai/DeepSeek-V4.1-Flash-1017, amd/DeepSeek-V4.1-Flash-Quark-MXFP4, OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated, rapid-mlx/DeepSeek-V4.1-Flash-DSpark-4d2e-MLX, apetersson/DeepSeek-V4.1-Flash-MixedQ2-GGUF, pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit (scan: all 24 deepseek_v41_text)
- flag (dsv4): deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4-Pro, endless-frontier/Fx-Bio (layer_types schema), MemTensor/MemT-1.0-Flash, baidu/ERNIE-5.5-Flash-Base
- not flag: HuggingFaceTB/nanowhale-100m-base, HuggingFaceTB/nanowhale-100m (compress_ratios all 0: no compressor); deepseek-ai/DeepSeek-V3.2 (DSA, no pools); zai-org/GLM-5.3; dots-studio/dots3-note-prev

## Verdict
real (over-credit): the V4.1 host -> V4 target edges (and any V4 host -> V4.1 target edges) disappear. Upper bound 24 x 21 V4.1-host / V4-target pairs (fewer in practice: hyper_conn variant, dims and MoE walls already cut most); fewer edges, likely tens of pairs. No new edges.
