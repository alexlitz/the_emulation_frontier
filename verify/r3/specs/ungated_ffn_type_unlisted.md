# ungated_ffn_type_unlisted
live 4 / placed 4 (parler_tts_decoder 4); direction over-credit; batch a2_dims

## 1. What the code computes
- `ext:parler_tts/gh__huggingface__parler-tts/modeling_parler_tts.py:940` ParlerTTSDecoderLayer: `:958`
  `activation_fn = ACT2FN[config.activation_function]` (gelu, erf); `:979-980` `fc1: embed -> ffn_dim`, `fc2: ffn_dim
  -> embed`, bias=False; `:1060-1062` `fc2(activation_fn(fc1(x)))`: an UNGATED two-matrix MLP. (The layer also builds
  `encoder_attn` cross-attention, `:978`: the separate cross_attention_decoder_unflagged finding.)
- Snapshot: the four parler-tts v1 decoders (model_type parler_tts_decoder under an outer parler_tts) carry
  `activation_function: gelu`, `ffn_dim`, `rope_embeddings`, `rope_theta`; parler_tts_mini_v0.1 has no rope keys.
- MusicGen / MusicGen-Melody decoders (same fc1 -> act -> fc2 decoder layer in transformers) are absent from the
  snapshot; listed for completeness only.
- p5 (`p5:model_graph.py` classify FFN): parler_tts_decoder is in neither VANILLA, NEMOTRON_H_TYPES nor
  UNGATED_BY_CODE and has no `mlp_use_swiglu` key -> `ffn = "glu"` (the "modern default"). act GeLU is right.
  Not fixed in p5.

## 2. Classifier change
- reads: model_type
- new: slot ffn = "vanilla", via UNGATED_BY_CODE entries
- detection:
```python
UNGATED_NEW = frozenset({"parler_tts_decoder", "musicgen", "musicgen_decoder", "musicgen_melody",
                         "musicgen_melody_decoder"})

def detect_ungated_ffn_type_unlisted(cfg, model_type):
    return "vanilla" if model_type in UNGATED_NEW else None
```
- apply note: add the five types to `UNGATED_BY_CODE` (comment: ParlerTTSDecoderLayer / MusicgenDecoderLayer fc1 ->
  act -> fc2, bias-free).

## 3. Relation rule
- target side: a vanilla target is played by vanilla hosts and by GLU hosts (vanilla -> glu, `glu_ungated`, needs an
  up bias or a constant channel; FFN_ORDER "free"): more in-edges (vanilla hosts now qualify).
- host side: a vanilla host cannot play GLU targets (glu -> vanilla only at 2x depth with squared units / a bound):
  fewer out-edges -- it was credited as a GLU host.
- kind: exact (target side); host side as FFN_ORDER prices glu <- vanilla (not exact for a GeLU host: bound)
- cost: none on the target side
- free by the config rules: not free (gated vs ungated is a feature of the code)
- theorem: `glu_ungated` (ffn: vanilla -> GLU); `swiglu_eq_sq_ffn`, `yarotsky_product_bound` (glu -> vanilla)

## 4. Test fixtures
- flag: parler-tts/parler-tts-large-v1, parler-tts-mini-multilingual-v1.1, parler-tts-mini-v1, parler-tts-mini-v1.1
  (live) and parler-tts/parler_tts_mini_v0.1 (same code; unplaced: no positional keys) -> "vanilla"
- not flag: thuml/sundial-base-128m (SwiGLU MLP, `modeling_sundial.py:163-177`), openai-community/gpt2 (already
  VANILLA)
- scan: the 5 parler_tts_decoder repos.

## Verdict
real, 4 placed models: their host out-edges to GLU targets disappear (fewer edges, likely the larger part) and
vanilla hosts gain them as targets (more). Tens of edges. (Whether an audio-codebook decoder with cross-attention
belongs in the zoo is the separate cross_attention_decoder_unflagged question.)
