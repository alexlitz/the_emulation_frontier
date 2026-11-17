# audio_key_read_as_lm
live 1 / placed 1 (arkasr 1); direction both; batch a3_placement

## 1. What the code computes
- Edge0/Audio8-ASR-0.1B/modeling_arkasr.py:8, :92: the LM is `Qwen2ForCausalLM(config)` (transformers' stock class);
  no `use_rope` read anywhere in modeling_arkasr.py. `use_rope` (configuration_arkasr.py:21, :60, default False;
  snapshot false) belongs to the audio side.
- Edge0/ARK-ASR-3B/modeling_arkasr.py:7, :75: `ArkasrForConditionalGeneration(Qwen2ForCausalLM)`; `use_rope` is passed
  only to the audio encoder (:23). (Snapshot use_rope true, so its pos is already rope.)
- tf4:models/qwen2/modeling_qwen2.py:134-135: q_proj / k_proj (and v_proj) `bias=True` always; :158
  apply_rotary_pos_emb always; :321 Qwen2RotaryEmbedding(theta = rope_theta 1e6).
- p5 `_nope_attention` (p5:model_graph.py :2413-2420) reads `use_rope is False` and its docstring names ArkASR, so
  Audio8-ASR-0.1B gets pos=nope; BIAS_QKV_TYPES (:1947) / QBIAS_TYPES (:3816) lack arkasr, so BOTH arkasr points
  miss bias_qkv (p5 classify: Audio8 pos nope feats tied_emb; ARK-ASR-3B pos rope feats tied_emb).

## 2. Classifier change
- reads: model_type arkasr
- new: arkasr read as the stock qwen2 LM: pos rope (use_rope ignored), feats += bias_qkv; in _nope_attention,
  BIAS_QKV_TYPES, QBIAS_TYPES
- detection:
```python
def detect_audio_key_read_as_lm(cfg, model_type):
    if model_type == "arkasr":
        return "qwen2"
    return None
```
- apply note: p5:model_graph.py `_nope_attention` (:2413): `(d.get("use_rope") is False and mt != "arkasr")` and drop
  ArkASR from the docstring; add "arkasr" to BIAS_QKV_TYPES (:1947) and QBIAS_TYPES (:3816). (Solar-Open2 keeps the
  use_rope rule: its own code honours it.)

## 3. Relation rule
- target side: a rope + bias_qkv target needs a RoPE host (rot-gated) with q/k/v biases
- host side: as before it plays nope targets (rope host, `nope_sub_rope`) and now biased / unbiased ones
- kind: exact (existing rules), the change moves the point
- cost: none
- free by the config rules: rope_theta free; bias_qkv is shape-changing (never free)
- theorem: `nope_sub_rope`, `partial_rope_host` (rot gate); bias wall `biased_proj_not_bias_free`; host `bias_off`

## 4. Test fixtures
- flag: Edge0/Audio8-ASR-0.1B -> qwen2 (use_rope false read as nope today); Edge0/ARK-ASR-3B -> qwen2 (same LM
  class; misses bias_qkv today)
- not flag: Qwen/Qwen2-0.5B (the stock family itself), upstage/Solar-Open2-250B and
  nota-ai/Solar-Open2-250B-Nota-NVFP4 (use_rope false honoured by their own code)

## Verdict
real: Audio8-ASR-0.1B goes nope -> rope and both arkasr points gain bias_qkv: edges INTO them from NoPE / bias-free
hosts disappear (fewer edges, 2 targets); as hosts they gain bias-capable host edges (more edges, small). Region:
attention.
