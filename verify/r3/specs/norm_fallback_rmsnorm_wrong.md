# norm_fallback_rmsnorm_wrong
live 5 / placed 5 (parler_tts_decoder 4, sundial 1); direction over-credit; batch a2_dims

## 1. What the code computes
- ParlerTTS decoder: `ext:parler_tts/gh__huggingface__parler-tts/modeling_parler_tts.py:961,978,981`
  `self_attn_layer_norm`, `encoder_attn_layer_norm`, `final_layer_norm = nn.LayerNorm(embed_dim)` (mean-centred,
  affine weight AND bias, eps 1e-5); `:1373` the decoder's final `nn.LayerNorm(hidden_size)`. Linear layers bias-free
  (`:979-980`), so the only output-side constants are the LayerNorm betas.
- Sundial: `thuml/sundial-base-128m/modeling_sundial.py:190-191` `norm1 / norm2 = torch.nn.LayerNorm(hidden)`,
  `:262` final `torch.nn.LayerNorm(hidden)` (affine, bias, eps 1e-5). It is a time-series model: input
  `SundialPatchEmbedding` (`:28-51`, unfolded value patches), FlowLoss head (`:11`), not a token LM.
- Neither config has an eps key (`rms_norm_eps` / `layer_norm_eps` / ...); both carry `rope_theta`.
- p5 (`p5:model_graph.py` classify norm): not in NORM_BY_CODE, no eps key, not in LN_FAMILIES -> the last branch
  `norm = "RMSNorm" if (mt in GATED or modern)` with modern = rope_like -> RMSNorm. bias_out (the LayerNorm beta) is
  missing too (not in BIAS_OUT_TYPES). Not fixed in p5; SKIP_TYPES has no sundial.

## 2. Classifier change
- reads: model_type
- new: slot norm = "LayerNorm" via NORM_BY_CODE; FEATURES bias_out via BIAS_OUT_TYPES (the affine beta); for
  sundial, preferably SKIP_TYPES
- detection:
```python
NORM_NEW = {"parler_tts_decoder": "LayerNorm", "musicgen": "LayerNorm", "musicgen_decoder": "LayerNorm",
            "musicgen_melody": "LayerNorm", "musicgen_melody_decoder": "LayerNorm", "sundial": "LayerNorm"}

def detect_norm_fallback_rmsnorm_wrong(cfg, model_type):
    return NORM_NEW.get(model_type)
```
- apply note: add the entries to `NORM_BY_CODE` (checked before any key) and parler_tts_decoder / musicgen* /
  sundial to `BIAS_OUT_TYPES` (the LayerNorm beta; `_pass5_dims` then sets ln_beta, so the LayerNorm -> RMSNorm
  fold needs an lm_head bias or the compensator). Open design decision: sundial reads value patches and emits a
  flow-matching sample, so `SKIP_TYPES.add("sundial")` (non-token model) is the faithful fix; the NORM_BY_CODE entry
  is for the case it stays.

## 3. Relation rule
- target side: a LayerNorm target with beta on an RMSNorm host needs the fold with an lm_head bias (exact) or the
  pre-norm compensator (payable, measured families only): fewer in-edges than as an RMSNorm target.
- host side: a LayerNorm host plays RMSNorm targets only at 2x width (`rmsNorm_sub_layerNorm_mirror`), LayerNorm
  targets directly: fewer out-edges to RMSNorm targets at equal width.
- kind: exact (LN <- RMS fold with lm_head bias; RMS <- LN mirror at 2x width); limit / measured (compensator)
- cost: 2x hidden for RMSNorm targets on these hosts (NORM_WIDTH_FACTOR)
- free by the config rules: not free (norm class is code)
- theorem: `layerNorm_sub_rmsNorm`, `layerNorm_eq_rms_center`, `affine_fold` (LayerNorm -> RMSNorm);
  `layerNorm_mirror`, `rmsNorm_sub_layerNorm_mirror` (RMSNorm -> LayerNorm at 2x)

## 4. Test fixtures
- flag: parler-tts/parler-tts-large-v1, -mini-multilingual-v1.1, -mini-v1, -mini-v1.1 -> "LayerNorm" (live);
  thuml/sundial-base-128m -> "LayerNorm" (live); scan also flags parler-tts/parler_tts_mini_v0.1 (same code,
  unplaced)
- not flag: openai-community/gpt2 (LN family, already LayerNorm), Qwen/Qwen3-0.6B (RMSNorm by key)
- scan: 6 repos (5 parler + sundial).

## Verdict
real, 5 placed models: they lose RMSNorm-host / RMSNorm-target edges at equal width (fewer edges; the RMSNorm
mirror at 2x width and the LN fold keep some). Tens of edges; sundial should rather leave the zoo (SKIP_TYPES).
