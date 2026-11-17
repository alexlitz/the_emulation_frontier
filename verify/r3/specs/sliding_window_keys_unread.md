# sliding_window_keys_unread
live 24 / placed 24 (grug_moe 7, plamo3 3, mpt 2, needle 2, meditor 1, meditor2 1, Motif 1, cost_wise_gemma 1, edgerock 1, ethoric 1, fabric 1, gaggle 1, iquest_q1 1, jet_nemotron 1); direction over-credit (in fact both, per family); batch a6_masks

Really live: 21. The 2 PhoGPT (mpt) snapshots do not take the path: `attn_config.sliding_window_size` is absent (default -1)
and `attn_impl` is "torch"; MPTConfig refuses a window off the flash path (vinai/PhoGPT-4B-Chat/configuration_mpt.py:155).
needle2 lists no global layers, so all 27 layers are local and p5 already reads that (sliding_window 256).

## 1. What the code computes
Per family: which softmax layers are full, which are sliding, and how many keys a sliding layer sees.
- cost_wise_gemma (BAAI bge-reranker-v2.5-gemma2-lightweight): layers are transformers-4.42 `Gemma2DecoderLayer`
  (gemma_model.py:56, :258); `is_sliding = not bool(layer_idx % 2)` (tf_old/4.42.4/.../gemma2/modeling_gemma2.py:624);
  mask `tril(ones, diagonal=-sliding_window)` -> W keys (:640-645). 21 sliding (even), 21 full, W 4096 < 8192.
- needle3: band `pos_q - pos_k < sliding_window` except on `global_layers` (nocode needle/gh__cactus-compute__cactus/
  architecture.py:469-475; default (4, 9, 14, 19), config.py:36). The snapshot keeps them in `extras.global_attention_layers`.
  4 full, 16 sliding, W 1024.
- iquest_q1: layer_types = first + block * num_hybrid_layers_block + last (nocode iquest_q1/gh__IQuestLab__vllm-iquest-q1/
  configs.py:95-99; defaults :72-91); per-layer window in model.py:70-90. [full] + [full, swa, swa, swa] x 21 + [full] x 3:
  25 full, 63 sliding, W 4096.
- Motif-3: `effective_window = sliding_window + 1`; under pattern 'interleave', SWA iff `layer_idx % period != 0`
  (Motif-3/modeling_motif.py:568-577). Flash only (:1297-1307), transformers flash window (w-1, w-1) (tf5
  modeling_flash_attention_utils.py:675). So W = 129 keys. 14 full (i % 4 == 0 over 53), 39 sliding.
- meditor / meditor2: `is_global_layer = (layer_idx + 1) % global_attn_every_n_layers == 0`, others sliding_window_size
  (Meditor-1 modeling_meditor.py:122-123; Meditor-2 modeling_meditor2.py:130-131). 4 full, 20 sliding, W 2048. (The
  prefill of these layers is separately broken: swa_prefill_key_truncation.)
- gaggle: `window_sizes`: S -> ceil(sequence_len/4/128)*128 = 512, L -> sequence_len, last layer forced long
  (configuration_gaggle.py:78-85); mask `row - col <= window` = window + 1 keys (modeling_gaggle.py:72-73). 'SSSL'
  over 34 layers: 9 full (i % 4 == 3, plus layer 33), 25 sliding, W 513. (The review said 26 sliding; it is 25.)
- jet_nemotron: 'swa' layers get `efficient_attention_config["swa"]["window_size"]` (modeling_jet_nemotron.py:348-353),
  mask `j > i - W` (:207-210). W 1152. The layer kinds are already read ('swa' -> local); the window size is not.
- grug_moe: long iff `(idx + 1) % global_every == 0 or idx == L - 1` (nocode grug_moe/gh__marin-community__marin/
  skyrl_train_models_grug_moe.py:91-98; default 4, :50). Sliding band `key >= query - (W - 1)` (:778). 7 full
  (3, 7, ..., 23, 25), 19 sliding, W 2048.
- plamo3: `is_full_attn = not (layer_idx + 1) % sliding_window_pattern` (plamo-3-translate-2b/modeling_plamo.py:115-116);
  `swa_mask` `q - W < k < q + W` (:426-431), with the causal mask W keys. 16 layers: 2 full; 24 layers: 3 full. W 2048.
- edgerock: `fixed_memory_swa_layers` (3, 7, ..., 31: the layer_types 'full_attention' layers) become
  `Qwen3_5FixedWindowAttention` (modeling_edgerock.py:1036-1041), `swa_window = fixed_memory_swa_window` (:827),
  band `key_pos >= query_pos - window + 1` (:417-418). 0 full, 8 sliding, W 1024 (24 linear layers unchanged).
- ethoric: `attn_window` on `attn_window_layers` only (Ethoric-10M/attention.py:180-185), `q_pos - k_pos < window`
  (:97-98). 11 full, 1 sliding, W 16.
- fabric (Bluebird-1.5): every layer's GQAAttention gets `local_attention_window` (fabric_runtime.py:269, :282, :310-314),
  mask `q - W < k <= q` (:121-126). 0 full, 24 sliding, W 2048. (The FabricMemoryBlock memory branch on layers i % 3 == 2,
  :278-296, is a separate unmapped mixer.)

## 2. Classifier change
- reads: per family as above (sliding_window_size, global_attn_every_n_layers, window_pattern / sequence_len,
  efficient_attention_config.swa.window_size, global_every, sliding_window_pattern / window_size, fixed_memory_swa_*,
  attn_window / attn_window_layers, local_attention_window / architecture, extras.global_attention_layers,
  first_layers_types / hybrid_layers_types_block / num_hybrid_layers_block / last_layers_types,
  sliding_window_pattern 'interleave' / sliding_window_period), model types listed above
- new: the standard keys `layer_types` ('full_attention' / 'sliding_attention', other entries kept) and `sliding_window`
  (visible keys), in `_normalize`
- detection (returns [n_full, n_local, W]; full code in verify/r3/parts/a6_masks.py):
```python
def detect_sliding_window_keys_unread(cfg, model_type):
    lm = lm_of(cfg); n = layer_count(lm)
    ...  # one branch per family: build kinds = ['full' | 'local' | 'rec'] per layer and the window W
    return _layout(kinds, W, mpe)   # [n_full, n_local, W]; None when no sliding layer binds (W >= context)
```
- apply note: in `_normalize` (p5:model_graph.py ~L2543), one block per family, BEFORE the "window >= max_position_embeddings"
  tail (~L2673) so a non-binding window still drops: set `u["layer_types"]` from the kinds and `u["sliding_window"] = W`.
  For jet_nemotron set only `sliding_window` (its 'swa' entries already map to local in `_kind`); for edgerock rewrite
  the swa-layer entries of the existing layer_types to 'sliding_attention'. Motif and gaggle use W + 1 (inclusive band).
  Then classify's window slot (~L2319) reads `sliding_window` -> 'local', `_attn_counts` (~L1351) reads layer_types,
  dims_of eff_window (~L3235) reads sliding_window. Optional, decided yes: add cost_wise_gemma to WINDOW_FREE_TYPES.
  Its window is read by transformers' own Gemma2DecoderLayer.forward (4.42 L637-645), not by remote code.

## 3. Relation rule
- target side: its sliding layers (LU) need a host FU or LU layer, its full layers an FU one (KIND_REACH ~L1481)
- host side: its sliding layers play only LU targets (no window lever: remote code; cost_wise_gemma excepted)
- kind: exact (full -> local by the band mask; host sliding layers zeroed when unused)
- cost: none new (the per-kind depth count of kind_match / ZERO_EDGES)
- free by the config rules: window sizes are free, but only where the host's code reads them (WINDOW_FREE_TYPES); these
  are remote-code families, so no lever. The layer schedule (which layers slide) is a feature switch, not free.
- theorem: `local_sub_full`, `pattern_sub_full` (full host plays a sliding layer), `zeroed_sublayer` (ZERO_EDGES);
  context axis only: `winReach_iff` / `winLocal_stack`

## 4. Test fixtures
- flag: BAAI/bge-reranker-v2.5-gemma2-lightweight -> [21, 21, 4096]; Cactus-Compute/needle3 -> [4, 16, 1024];
  IQuestLab/IQuest-Q1 -> [25, 63, 4096]; Motif-Technologies/Motif-3 -> [14, 39, 129]; Yo-Purple/Meditor-1 /
  Meditor-2 -> [4, 20, 2048]; castorini/gaggle-... -> [9, 25, 513]; jet-ai/Jet-Nemotron-2B -> [2, 2, 1152];
  3 of 7 open-athena grug_moe -> [7, 19, 2048] (the scan flags all 7); pfnet/plamo-3-translate-2b -> [3, 21, 2048];
  pfnet/plamo-3-0.8b-fin-instruct -> [2, 14, 2048]; rockmatrix-ai/EdgeRock -> [0, 8, 1024]; techdotus/Ethoric-10M ->
  [11, 1, 16]; tudor-iustin22/Bluebird-1.5-Instruct -> [0, 24, 2048]
- not flag: vinai/PhoGPT-4B, vinai/PhoGPT-4B-Chat (no sliding_window_size, torch attn_impl); Cactus-Compute/needle2
  (no global layers: all local, already read); google/gemma-2-2b, google/gemma-3-1b-it (gemma2 / gemma3 read by p5);
  kingbear329/sn38-... (layer_types read); mistralai/Mistral-7B-v0.1, Qwen/Qwen3-0.6B

## Verdict
Real, in both directions. Each type is its own config class, so the edges move only on these 21 points' rows and columns.
- Fewer host edges for meditor / meditor2 / gaggle / plamo3 / edgerock / ethoric / fabric: they were counted all-full
  and lose their sliding layers as full-target hosts. 9 models; fabric and edgerock end up with 0 full layers.
- More target edges for the same models: sliding layers can now go on band-local hosts.
- More host edges for cost_wise_gemma / iquest_q1 / Motif / grug_moe / needle3: they were counted all-local and gain
  full layers (11 models). Fewer target edges for them, because those full layers need FU.
- jet_nemotron: eff_window only (context axis); no headline edge moves.
Rough total: on the order of 10^2 to 10^3 pair flips spread over ~20 points. Some of these pairs will also be
walled by other round-3 features: ethoric doc_mask, fabric / edgerock memory branches, and needle's extras.mhc_lanes /
engram, which p5's _feats does not read today.
