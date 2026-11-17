# text_encoder_input_embeds
live 2 / placed 2 (breeze 2); direction over-credit; batch a3_placement

## 1. What the code computes
- ext:breeze/gh__breezeblue-ai__breeze-tts/models/breeze.py:947-1004: with `text_encoder_config` set (every snapshot) the
  model builds `self.text_encoder = AutoModel.from_config(text_encoder_config)` (a T5Gemma2 encoder: 26
  bidirectional layers, hidden 1152) and `text_encoder_proj = Linear(1152 -> hidden 2048)` (text_encoder_proj_type
  "linear", :1000-1004).
- breeze.py:1352-1460 `convert_input_ids_to_embeds`: text segments run through the encoder (positions restart per
  segment, padding masked); :1453-1458 `inputs_embeds = zeros(...)`, `inputs_embeds[text_ids_mask] = text_embeds`
  (projected encoder output, :1479-1491). The embed_text_tokens lookup (:928, :1578) is only the no-encoder fallback.
- So the LM's text inputs depend on the WHOLE text segment (bidirectional), not on the token at that position: no
  causal host with a per-token embedding forms them.
- Audio positions (breeze.py:1585-1600) are embedded by `embed_audio_tokens` (:783-802): the sum over the 16 codebook
  rows of one frame.
- p5 has no feature for an encoder-fed input stream (p5:model_graph.py FEATURES :1597).

## 2. Classifier change
- reads: model_type breeze, text_encoder_config (dict)
- new: FEATURES "enc_input" ("wall", "free"), in _feats
- detection:
```python
def detect_text_encoder_input_embeds(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "breeze" and isinstance(lm.get("text_encoder_config"), dict):
        return True
    return None
```
- apply note: p5:model_graph.py `_feats` (:1741): `if isinstance(get("text_encoder_config"), dict): f.add("enc_input")`
  (`get` already falls back to the outer config, needed once breeze_backbone_config_unread unwraps the backbone);
  FEATURES entry "enc_input": ("wall", "free", "the LM's text inputs are a bidirectional encoder's projected output
  (Breeze T5Gemma2 text encoder): no per-token embedding of a causal host forms them; a host feeds plain tokens
  through its own embedding path").

## 3. Relation rule
- target side: wall: the input at a text position is a function of the whole text segment (a bidirectional
  encoder), which a causal host's per-position embedding cannot reproduce
- host side: free: a Breeze host plays a plain target through its audio-frame embedding (embed_audio_tokens with the
  15 other codebook rows zero = one table, breeze.py:799-802; vocab size ignored), the text encoder unused
- kind: wall (target) / exact (host)
- cost: none
- free by the config rules: text_encoder_config is a feature switch (not free); the host-side route needs no config
- theorem: host side `extra_input_off` / `zeroed_sublayer` (PROVED exact); target side needs theorem: "a causal stack
  whose position-t input depends only on token t cannot match inputs that depend on later tokens of the segment"
  (cf. `causal_attn_layer` for the gen=diffusion wall)

## 4. Test fixtures
- flag: BreezeBlue/Breeze-TTS-2 -> True; drbaph/Breeze-TTS-2-comfyui -> True
- not flag: Qwen/Qwen3-0.6B (the backbone family), kmhf/hf-moshika (audio LM with token inputs),
  parler-tts/parler-tts-mini-v1 (encoder states enter by cross-attention: cross_attention_decoder_unflagged),
  XinyueWangg/TimeBraid-2.5B

## Verdict
real: every edge INTO the 2 Breeze points disappears (fewer edges; they become target-walled except from a Breeze
host); host edges unchanged. Region: classification.
