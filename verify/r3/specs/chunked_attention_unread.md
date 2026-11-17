# chunked_attention_unread
live 1 / placed 1 (llama4_text 1); direction over-credit; batch a6_masks

## 1. What the code computes
- Llama4TextConfig.__post_init__ (tf5 models/llama4/configuration_llama4.py:176-202):
  - `no_rope_layers` defaults to `int((i + 1) % no_rope_layer_interval != 0)` (1 = RoPE) when empty (:180-183). The
    Scout snapshot saves `no_rope_layers: []`.
  - `layer_types = ["chunked_attention" if no_rope else "full_attention" ...]` when None (:198-201). The snapshot has
    no layer_types.
  - Result: 36 chunked RoPE layers and 12 full NoPE layers (every 4th) of 48.
- Model forward: masks keyed by layer_types, 'chunked_attention' -> `create_chunked_causal_mask`
  (tf5 models/llama4/modeling_llama4.py:548-561).
- The chunked mask is causal AND `(kv_idx - pad) // chunk_size == (q_idx - pad) // chunk_size`
  (tf5 masking_utils.py:106-113, :163-167, chunk_size = config.attention_chunk_size, :1402-1404). It is
  block-diagonal causal with ABSOLUTE chunk boundaries: query i sees keys in [C*floor(i/C), i]. That is not a sliding
  band. attention_chunk_size 8192, max_position_embeddings 10485760.
- p5: `_normalize` derives the NoPE default (~L2655) but not layer_types. classify gives window=full and
  `_attn_counts` 48 FU (no sliding_window, ~L1416).

## 2. Classifier change
- reads: model_type llama4 / llama4_text, attention_chunk_size, no_rope_layers / no_rope_layer_interval, layer_types
- new: dims eff_window = attention_chunk_size for the chunked layers (context axis). The kinds stay FU (see apply note).
  Optionally layer_types derived in `_normalize` with a new `_kind` value.
- detection (returns [n_chunked, chunk_size]):
```python
def detect_chunked_attention_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in ("llama4", "llama4_text"):
        return None
    n = layer_count(lm)
    c, mpe = num(lm.get("attention_chunk_size")), num(lm.get("max_position_embeddings"))
    if not n or not c or (mpe and c >= mpe):
        return None
    lt = layer_types(lm)
    if lt:
        k = sum(1 for x in lt if "chunk" in x)
    else:
        nrl = lm.get("no_rope_layers"); iv = num(lm.get("no_rope_layer_interval")) or 4
        nrl = nrl if (isinstance(nrl, list) and nrl) else [int((i + 1) % iv != 0) for i in range(n)]
        k = sum(1 for x in nrl if x)
    return [k, c] if k else None
```
- apply note (decided). KEEP the chunked layers counted as FU in `_attn_counts`; do NOT derive layer_types with
  'chunked_attention' entries. `_kind` (~L1305) maps anything containing "chunk" to 'local' (_LOCAL_MARKS ~L1301),
  which would make a band-local host play a block-diagonal layer (wrong). If layer_types is derived, `_kind` must map
  'chunked_attention' to 'full'. Record the reach as dims eff_window = attention_chunk_size (dims_of ~L3236) for
  USE_CONTEXT. Add llama4 / llama4_text to the window-lever doc as "chunk size is config"
  (create_chunked_causal_mask reads config.attention_chunk_size).

## 3. Relation rule
- target side: a chunked layer is played by a full host layer under the chunk mask (pattern_sub_full: any fixed 0/1
  mask on full attention). A band-local host cannot form absolute chunk boundaries (a wall, but FU accounting already
  excludes it).
- host side: a Llama-4 host sets attention_chunk_size >= T (a window size: free, read by transformers' code), so its
  chunked layers are full causal layers and supply FU
- kind: exact both ways (target: mask currency; host: the window lever, `window_ge_full` analog for chunks)
- cost: none
- free by the config rules: attention_chunk_size is a window size (free); the chunked layer schedule is a feature switch
  (fixed)
- theorem: `pattern_sub_full` (target side); host side, chunk >= T is causal: needs theorem "chunked mask with
  C >= T equals the causal mask" (trivial; `window_ge_full`'s proof shape)

## 4. Test fixtures
- flag: lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-8bit -> [36, 8192]
- not flag: meta-llama/Llama-3.1-8B-Instruct (llama, no chunking); tudor-iustin22/Bluebird-1.5-Instruct (fabric's
  attention_chunk_size is a query-chunking compute detail, fabric_runtime.py:36, :177); google/gemma-3-1b-it

## Verdict
Real code fact, headline-neutral. With the chunk-size lever on the host side and the mask currency on the target side,
the chunked layers sit exactly where p5's FU count puts them. The only move is eff_window (10485760 -> 8192 on 36 of
48 layers, context axis). The risk is an implementation that derives layer_types and lets `_kind` call 'chunked'
local. That would over-credit band-local hosts for this 1 model. Region: attention.
