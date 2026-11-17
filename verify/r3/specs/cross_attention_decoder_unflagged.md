# cross_attention_decoder_unflagged
live 4 / placed 4 (parler_tts_decoder 4); direction over-credit; batch a3_placement

## 1. What the code computes
- ext:parler_tts/gh__huggingface__parler-tts/modeling_parler_tts.py:940-978 ParlerTTSDecoderLayer: besides self-attention and the
  FFN every layer builds `encoder_attn` (:967) and `encoder_attn_layer_norm` (:978); k / v heads
  num_cross_attention_key_value_heads (configuration_parler_tts.py:146-148, default num_key_value_heads).
- modeling_parler_tts.py:1036-1052: `if encoder_hidden_states is not None:` residual + encoder_attn(LN(h),
  key_value_states = encoder states) between self-attention and the FFN.
- modeling_parler_tts.py:2388-2392 and the top-level forward: the T5 description encoder's states
  (-> enc_to_dec_proj when widths differ, x attention_mask) are ALWAYS passed as the cross-attention memory by
  ParlerTTSForConditionalGeneration, the model the repos ship (architectures ParlerTTSForConditionalGeneration;
  outer is_encoder_decoder true, decoder add_cross_attention true).
- p5 classifies parler_tts_decoder as a plain decoder-only LM: no cross-attention feature (p5:model_graph.py FEATURES
  :1597).

## 2. Classifier change
- reads: LM add_cross_attention, outer is_encoder_decoder
- new: FEATURES "cross_attn" ("wall", "free"), in _feats
- detection:
```python
def detect_cross_attention_decoder_unflagged(cfg, model_type):
    lm = lm_of(cfg)
    if truthy_flag(lm.get("add_cross_attention")) and truthy_flag((cfg or {}).get("is_encoder_decoder")):
        return True
    return None
```
- apply note: p5:model_graph.py `_feats(d, outer)` (:1741): `if d.get("add_cross_attention") and
  outer.get("is_encoder_decoder"): f.add("cross_attn")`; FEATURES entry "cross_attn": ("wall", "free", "per-layer
  cross-attention to an encoder memory (Parler-TTS decoder over the T5 description encoder): a decoder-only host has
  no encoder input; a cross-attention host zeroes its encoder_attn out_proj").

## 3. Relation rule
- target side: wall: each layer reads an encoder memory (an extra input a decoder-only host has no place for)
- host side: free: a Parler host zeroes every encoder_attn out_proj (the cross-attention sublayer adds exactly 0) or
  runs without encoder states (:1038), and plays plain decoder-only targets
- kind: wall (target) / exact (host)
- cost: none
- free by the config rules: add_cross_attention is a feature switch (not free); zeroing weights is free
- theorem: host side `zeroed_sublayer` / `extra_input_off` (PROVED exact); target side: OPEN, the same class as the
  extra-input walls ("switch_lora / engram / ple / value_emb target <- plain host", PROOFS.md walls table)

## 4. Test fixtures
- flag: the 4 live parler-tts decoders and parler-tts/parler_tts_mini_v0.1 (same code; not placed today) -> True
- not flag: BreezeBlue/Breeze-TTS-2 (encoder feeds the input stream, not cross-attention), kmhf/hf-moshika,
  openai-community/gpt2, kuleshov-group/bd3lm-owt-block_size4 (cross_attn key of a DiT, already skipped)

## Verdict
real: every edge INTO the 4 Parler points disappears (fewer edges, 4 targets); their host edges stay. Region:
attention.
