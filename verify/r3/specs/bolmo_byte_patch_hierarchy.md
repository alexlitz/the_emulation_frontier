# bolmo_byte_patch_hierarchy
live 2 / placed 2 (bolmo 2); direction over-credit; batch a12_extra_inputs

## 1. What the code computes
(allenai/Bolmo-1B-Stage1/modeling_bolmo.py; allenai/Bolmo-1B ships the same classes)
- Input is bytes (vocab 520): the local encoder embeds `byte_embedding(tokens) + subword_embedding(expanded_input_ids)` (subword vocab 100278, `add_expanded_embeddings`) (:599-616, :656-659), runs `num_local_encoder_layers` (1) BolmoLocalLayers (:618-619, :719-735).
- BolmoLocalLayer: `x + mLSTM(norm x)`, then `x + SwiGLU(norm x)` of `local_intermediate_size` (2816) (:562-590); the mixer is xlstm's mLSTMLayer (exp. input gate, soft-capped gates, multihead norm, sigmoid output gate) with `num_local_heads` 16 (:406-560).
- BolmoBoundaryPredictor: cosine of q_proj(h_t) and k_proj(h_{t+lookahead}), lookahead 1 (:358-403, :387-388); `compute_boundary_mask(logprobs, "sample:0")` = `logprob > log 0.5` (utils_bolmo.py:7-13): content-dependent patch boundaries.
- `_pool`: one patch state per boundary (gather at boundary positions) (:663-695); the global Olmo layers (16, `block_type reordered_norm`) run over the patch sequence with patch positions (:1050-1131).
- BolmoLocalDecoder: `initial_norm` on the patch outputs, `in_projection` (hidden x hidden, bias) of the byte states, `_depool` = each byte gathers its patch's global output by `cumsum(boundary_mask) - 1` and adds it (:775-870, :856), then `num_local_decoder_layers` (4) mLSTM layers; final norm and lm_head at byte resolution (:1133-1141, :1158).
- p5 sees only the global stack: mixer softmax, ffn glu, no recurrent family, feats `''` (dims from hidden / layers of the global model; vocab 520).

## 2. Classifier change
- reads: model_type bolmo (code-fixed)
- new: FEATURES key `byte_patch` = ("wall", "wall"), in `FEATURES` + `_feats`
- detection:
```python
def detect_bolmo_byte_patch_hierarchy(cfg, model_type):
    return "byte_patch" if model_type == "bolmo" else None
```
- apply note: add to `FEATURES` (p5:model_graph.py:1597) `"byte_patch": ("wall", "wall", "byte-latent hierarchy (Bolmo): byte encoder / decoder with mLSTM local layers around a global transformer on content-pooled patches; no token-level stack forms the pool / depool, and the boundary predictor and local mLSTM layers have no off switch")`; `_feats`: `if mt == "bolmo": f.add("byte_patch")`. Alternative (equally valid): add "bolmo" to `SKIP_TYPES` (:646) as not a plain causal decoder, like reformer / xlnet; the feature keeps the 2 models placed but isolated. Do NOT add `rf_mlstm`: the mLSTM layers live inside the hierarchy, covered by the wall.

## 3. Relation rule
- target side: no host without the hierarchy can play it (the global layers see a data-dependent number of positions; no token-level layer pools / unpools).
- host side: a Bolmo host cannot play a plain byte-level target: all-boundary patches would need the predictor forced to 1 (a threshold change is not a config value here: `boundary_threshold` "sample:0" is a decoding rule, and the lookahead cosine is learned), and the decoder's `initial_norm` of the patch stream plus the local mLSTM layers stay on.
- kind: wall both ways; Bolmo <-> Bolmo exact only at identical structure (same code and dims).
- cost: none.
- free by the config rules: none.
- theorem: needs theorem (both sides OPEN): "a fixed-depth token-level transformer cannot reproduce content-dependent pool / depool" -- not stated in PROOFS.md.

## 4. Test fixtures
- flag: allenai/Bolmo-1B, allenai/Bolmo-1B-Stage1 -> "byte_patch".
- not flag: allenai/OLMo-2-0425-1B, allenai/Olmo-3-1025-7B (the global stack's parent families).
- scan: exactly the 2 bolmo repos.

## Verdict
real. Fewer edges both ways: Bolmo-1B / Bolmo-1B-Stage1 drop out of every Olmo-like edge (2 models; each pair edge between them stays).
