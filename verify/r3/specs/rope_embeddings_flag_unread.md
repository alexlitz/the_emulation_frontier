# rope_embeddings_flag_unread
live 4 / placed 4 (parler_tts_decoder 4); direction over-credit; batch a5_positions

## 1. What the code computes
- `ext:parler_tts/gh__huggingface__parler-tts/configuration_parler_tts.py:92-94,130-131,158-159`: `rope_embeddings=False` and `rope_theta=10_000.0` defaults.
- `ext:parler_tts/.../modeling_parler_tts.py`:
  - `:1358-1368` (ParlerTTSDecoder.__init__): `if not config.rope_embeddings: self.embed_positions = ParlerTTSSinusoidalPositionalEmbedding(...)`; the rotary module is built only otherwise.
  - `:1506-1510` (forward): `positions = self.embed_positions(inputs_embeds, ...)` added to the input embeddings.
  - `:516,540,643,665,858,880` (attention): RoPE on q / k only `if self.rope_embeddings`.
  - With the flag false (every snapshot), positions are the fixed Musicgen sinusoid table (`:326-327`) added to inputs_embeds, with no rotary.
- Snapshot: parler-tts-large-v1, -mini-v1, -mini-v1.1 and -mini-multilingual-v1.1 set `rope_embeddings: false` and carry `rope_theta`. parler_tts_mini_v0.1 omits both (class default False).
- round2-p5: classify's `rope_like` (`p5:model_graph.py:2183-2186`) fires on rope_theta, giving pos=rope for the four v1 configs. v0.1 has no rope key and returns None (unplaced). Checked with p5 classify. Not fixed in p5.

## 2. Classifier change
- reads: model_type parler_tts_decoder and `rope_embeddings` (default False).
- new: pos slot `sinusoidal`, in classify (SINUSOIDAL_TYPES, gated by the flag).
- detection:
```python
def detect_rope_embeddings_flag_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "parler_tts_decoder" and not truthy_flag(lm.get("rope_embeddings", False)):
        return "sinusoidal"
    return None
```
- apply note: in classify's positional block, before `elif mt in SINUSOIDAL_TYPES`, add `elif mt == "parler_tts_decoder" and not d.get("rope_embeddings"): pos = "sinusoidal"`. Equivalently, put parler_tts_decoder in SINUSOIDAL_TYPES and route `rope_embeddings` true to the rope branch. v0.1 then places (norm / act come from its config; check it has the keys).

## 3. Relation rule
- target side: sinusoidal. POS_ORDER has only `("pos_top", "free", True)`: a wall on every real host today. A learned_abs host could set its table to the sinusoid exactly (the table is a free weight). That edge is left to a later pass in p5 (POS_ORDER comment) and is not added here.
- host side: a sinusoidal host has a fixed table. It plays only a NoPE target and only if the table can be cancelled (it cannot: it is code, not weights), so it is a wall for everything except sinusoidal targets with the same table.
- kind: wall both ways (current POS_ORDER), exact to pos_top.
- cost: none.
- free by the config rules: rope_embeddings is a feature switch (not free).
- theorem: needs theorem for the deferred edge: `sinusoid_sub_learned_abs`, i.e. a learned table set to the sinusoid is the same input (trivial; same shape as `nope_sub_learned_abs`).

## 4. Test fixtures
- flag: each -> sinusoidal
  - parler-tts/parler-tts-large-v1
  - parler-tts/parler-tts-mini-multilingual-v1.1
  - parler-tts/parler-tts-mini-v1
  - parler-tts/parler-tts-mini-v1.1
  - parler-tts/parler_tts_mini_v0.1 (flag omitted = default False; unplaced today)
- not flag:
  - meta-llama/Llama-3.1-8B
  - Qwen/Qwen2.5-0.5B

## Verdict
Real (over-credit).
- 4 Parler decoders move from pos=rope to sinusoidal: they lose every RoPE host and target. Fewer edges: their whole current edge set, until a learned_abs <- sinusoidal edge is added.
- One more model (v0.1) becomes placeable.
