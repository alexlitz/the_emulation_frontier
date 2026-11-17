# b_kv_compress_host
live 45 / placed 42 (live: deepseek_v41_text 24, deepseek_v4 19 = 17 compress_ratios + 2 layer_types, ernie5_5_moe 1, memt_deepseek_v4 1; placed = the p5 feature: 24 + 17 + 1); direction over-wall (claimed); batch bd

## 1. What the code computes
- tf5 DeepSeek-V4 legacy key map `_COMPRESS_RATIO_TO_LAYER_TYPE = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}` (tf5:models/deepseek_v4/configuration_deepseek_v4.py:27-32), applied to `compress_ratios` (:250, :268-269). There is NO ratio-1 entry: a legacy config with ratio 1 raises KeyError in tf5.
- The layer builds its compressor from the layer type: `COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None` (tf5:models/deepseek_v4/modeling_deepseek_v4.py:800-802). HCA / CSA compressors own parameters whose shapes depend on the rate: `position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))` (:393), `(compress_rate, 2 * head_dim)` (:504, :621), plus the CSA indexer.
- Reference V4 code: `self.compress_ratio = args.compress_ratios[layer_id]`, `if self.compress_ratio: self.compressor = Compressor(...)`, `if self.compress_ratio == 4: self.indexer = Indexer(...)` (ext:deepseek_v41/hub__deepseek-ai__DeepSeek-V4-Flash/inference/model.py:453, :466-469); `ape = nn.Parameter(torch.empty(compress_ratio, ...))` (:294).
- V4.1 reference: ratio 1 IS a compressor ("ratio 1 is a plain projection": `self.wkv`, `self.norm`; `wgate` only `if compress_ratio > 1`) whose entries are concatenated to the window KV (ext:deepseek_v41/hub__deepseek-ai__DeepSeek-V4.1-Flash/inference/model.py:443-459, :616-619, :681). So `pool_one`'s "ratio 1 builds no compressor" (vLLM) is not what tf5 or the V4.1 reference do; the no-compressor value is ratio 0.
- ERNIE-5.5: `is_dsa_layer`: `compress_ratios[layer_id] <= 0` is SWA / DSA-only (ext:ernie5_5_moe/gh__baidu__vLLM-ERNIE/config.py:155-158).
- In-shape neutralisation fails: a compressed entry's key / value go through `kv_norm` / the compressor norm; zero weights give a zero key (logit 0, weight exp(0) > 0), and the number of compressed entries grows with the position, so no weight setting of a ratio > 0 layer is plain attention (`kv_compress_not_plain` covers the converse direction; this direction is a counting argument, not in PROOFS.md).
- p5 relation: `FEATURES["kv_compress"] = ("wall", "wall", ...)` (p5:model_graph.py:1692-1694), set when `compress_ratios` has a nonzero entry (:1822-1823). p5 does NOT read the tf5 `layer_types` form (MemT-1.0-Flash, Fx-Bio, DeepSeek-V4-Pro-0.5B: 3 repos with compressed layer types and no `compress_ratios` -> no kv_compress feature: a target-side over-credit, side finding).

## 2. Classifier change
- reads: `compress_ratios` (any entry > 0), else tf5 `layer_types` with a compressed type; model types deepseek_v4 / deepseek_v41* / memt_deepseek_v4 / ernie5_5_moe.
- new: none for the host switch (recommendation: keep the wall). Side fix: `_feats` also sets `kv_compress` from `layer_types` (compressed types), and ERNIE's negative entries are not compressed layers (`> 0`, not `!= 0`).
- detection:
```python
def detect_b_kv_compress_host(cfg, model_type):
    lm = lm_of(cfg)
    if not _v4_like(model_type):
        return None
    cr = lm.get("compress_ratios")
    if isinstance(cr, list) and any(num(x) > 0 for x in cr):
        return "compress_ratios"
    if any(x in _V4_COMPRESSED for x in layer_types(lm)):
        return "layer_types"
    return None
```
- apply note: p5:model_graph.py `_feats` :1822-1823 (`any(_num(x) > 0 ...)` or a compressed `layer_types` entry). FEATURES["kv_compress"] host side stays "wall". If the switch is ruled free: add "kv_compress" to a new `CONFIG_OFF_FEATS` merged into `host_off` at emul_matrix :4229 (all modes), never to FEATURES (it is not a weight setting).

## 3. Relation rule
- target side: unchanged wall (`kv_compress_not_plain`).
- host side: a compress host plays a plain target only by rewriting the per-layer ratio to 0 (the layer becomes `sliding_attention`, its window a free value >= the context).
- kind: exact IF the switch is allowed (the rewritten layer is the plain sliding layer, `window_ge_full`); wall otherwise.
- cost: none for the switch. In-shape alternative (no config change, exact): the host's ratio-0 layers are plain attention layers, the others are zeroed (o_b_proj = 0, experts 0: residual identity) -- target depth <= the host's ratio-0 layer count (ratio-0 entries: V4-Flash-0731 5 of 46 incl. MTP slots, V4-Pro 1 of 62, V4.1-Flash 5 of 43), the ZERO_EDGES attn_eff currency.
- free by the config rules: NOT free. `compress_ratios` changes the module set and parameter shapes (compressor wkv / wgate / ape / position_bias / indexer are created or not; position_bias is `compress_rate` rows): it is a shape-changing field, like adding qk-norm, and a per-layer feature switch like `use_sliding_window`. Recommend: keep the wall.
- theorem: `pool_one` (PROOFS.md Round-2 rules, "kv_compress: compress host -> plain target") is about a ratio-1 pool; the tf5 / reference switch is ratio 0, which needs no theorem (no compressor is built). The in-shape depth alternative uses the ZERO_EDGES layer-zeroing argument.

## 4. Test fixtures
- flag: deepseek-ai/DeepSeek-V4-Flash-0731 -> "compress_ratios"; deepseek-ai/DeepSeek-V4-Pro -> "compress_ratios"; deepseek-ai/DeepSeek-V4.1-Flash -> "compress_ratios"; deepseek-ai/DeepSeek-V4.1-Flash-1017 -> "compress_ratios"; baidu/ERNIE-5.5-Flash-Base -> "compress_ratios" (128 entries); MemTensor/MemT-1.0-Flash, endless-frontier/Fx-Bio, inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B -> "layer_types" (tf5 format).
- not flag: HuggingFaceTB/nanowhale-100m-base (V4, every ratio 0: no compressor); zai-org/GLM-5.3, deepseek-ai/DeepSeek-V3 (DSA / MLA, no pools); tencent/Hy4-preview-FP8, shashanth-27xA9/GLM-5.3-SuperExpert-v2-9.2GB (V4 relatives without pools).

## Verdict
real but keep the wall: the host switch is a shape-changing schedule field, not a free value. Frontier effect of
opening it anyway (measured on the 931 real shapes, FEATURES host side flipped, z3-final code): kv_compress alone
0 new pairs; with hash_route and act_clip opened too, 10 shape pairs -- V4 hosts (amd/DeepSeek-V4-Flash-MXFP4,
DeepSeek-V4-Pro, deepseek-v4-flash-0731-spark) playing nanowhale-100m and the tf5-format V4 relatives (MemT,
Fx-Bio); every kv_compress host also carries hyper_conn (wall both ways), so only hyper_conn targets are reachable.
Side finding (target side, over-credit, 3 repos): kv_compress missing on tf5-format V4 configs (layer_types).
