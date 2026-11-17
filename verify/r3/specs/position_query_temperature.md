# position_query_temperature
live 24 / placed 31 (beam 2, llama4_text 1, ministral3 10, mistral4 1, mistral_large4_text 2, nemotron_labs_diffusion 4, qwen 11); direction over-credit; batch a5_positions

## 1. What the code computes
Five code paths multiply the query by a factor that depends on the position. They are not in round2-p5: no FEATURES key, no dims field, and none of `use_logn_attn`, `llama_4_scaling_beta`, `attn_temperature_tuning` or `softmax_scaling` is read anywhere in p5:model_graph.py.
- Qwen-1 `Qwen/Qwen-7B-Chat/modeling_qwen.py:289-294`: `logn_list[i] = log_{seq_length}(i) if i > seq_length else 1`. At `:482-490` (eval only, `key_size > seq_length and use_logn_attn`): `query *= logn_tensor[key_len-q_len : key_len]`. The factor is 1 up to seq_length and grows after. Qwen-VL and Qwen-Audio use the same code (hunks dcc48e5547f6 and 19cd762964d0).
- Ministral-3 `tf5:models/ministral3/modeling_ministral3.py:105-107,146-150`: after RoPE, `q *= 1 + beta*log(1 + floor(pos / original_max_position_embeddings))`, with beta = `rope_parameters.llama_4_scaling_beta`. Mistral-4 MLA (`tf5:models/mistral4/modeling_mistral4.py:370-372,489-493`) does the same. Its config default carries beta 0.1 (`configuration_mistral4.py:123`). Nemotron-Labs-Diffusion has the same formula in its remote `modeling_ministral.py` (`_get_llama_4_attn_scale`, hunk f7569ff95346). The factor is 1 while pos < original_max.
- Llama-4 `tf5:models/llama4/modeling_llama4.py:321-323,367-374`: on NoPE layers only (`not self.use_rope`) with `attn_temperature_tuning` (class default True, `configuration_llama4.py:171-173`; floor_scale 8192, attn_scale 0.1), `q *= 1 + attn_scale*log1p(floor((pos+1)/floor_scale))`. The factor is 1 while pos+1 < 8192.
- Mistral-Large-4 `tfmain:models/mistral_large4/modular_mistral_large4.py:296-309`: `MistralLarge4TextSoftmaxScaling` has LEARNED per-head `scale_h`, `bias_h` (checkpoint parameters), and every query of every layer is multiplied by `scale_h*log(pos+1) + bias_h`. Nothing gates it: it acts from position 1 on.
- Beam `tfmain:models/beam/modular_beam.py:172-174,207-208,420-428`: on layer types whose rope entry is None (the NoPE full layers), `q *= exp(0.1*log1p((pos+1)/2048)) = (1+(pos+1)/2048)^0.1`. The constants are fixed in code and nothing gates it (1.0005 at pos 0).
- Configs that are placed but do not take the path: Qwen-72B / -72B-Chat (`use_logn_attn` false) and Mistral-Medium-3.5-128B (`llama_4_scaling_beta` 0). Configs that take the path but never fire inside max_position_embeddings: Qwen-1_8B (seq 8192 = mpe), Qwen-Audio / -Chat (2048 = mpe), Nemotron-Labs-Diffusion-8B-Base (mpe 4096 < 16384).

## 2. Classifier change
- reads: `use_logn_attn` + `seq_length` (qwen); `rope_parameters.llama_4_scaling_beta` + `original_max_position_embeddings` (ministral3, mistral4, nemotron_labs_diffusion); `attn_temperature_tuning` / `attn_scale` / `floor_scale` with class defaults (llama4 / llama4_text); model type mistral_large4(_text); beam's None rope entries.
- new: two FEATURES keys and one dims field.
  - FEATURES `q_temp` ("wall", "free"): the ungated learned form, mistral_large4.
  - FEATURES `q_temp_nope` ("wall", "wall"): Beam's code-fixed NoPE-layer temperature.
  - dims `pos_gate` (int, 0 = none): the gated forms, logn / llama4_beta / nope_temp, give the position where the factor first differs from 1. It is read only under USE_CONTEXT.
- detection:
```python
def detect_position_query_temperature(cfg, model_type):
    lm = lm_of(cfg)
    mt = model_type
    if mt == "qwen" and lm.get("use_logn_attn") is True:
        return f"logn@{num(lm.get('seq_length')) or 8192}"
    if mt in ("ministral3", "mistral4", "nemotron_labs_diffusion"):
        rp = rope_params(lm)
        beta = fnum(rp.get("llama_4_scaling_beta"))
        if beta is None and mt == "mistral4" and not (lm.get("rope_parameters") or lm.get("rope_scaling")):
            beta = 0.1                                     # Mistral4Config default rope_parameters
        if beta:
            return f"llama4_beta@{num(rp.get('original_max_position_embeddings')) or 8192}"
        return None
    if mt in ("llama4", "llama4_text"):
        if lm.get("attn_temperature_tuning", True) is False or fnum(lm.get("attn_scale"), 0.1) == 0:
            return None
        nrl = lm.get("no_rope_layers")
        has_nope = not (isinstance(nrl, list) and nrl and all(x for x in nrl))  # class default: NoPE every 4th
        return f"nope_temp@{num(lm.get('floor_scale')) or 8192}" if has_nope else None
    if mt in ("mistral_large4_text", "mistral_large4"):
        return "learned_logpos"
    if mt == "beam":
        if any(v is None for v in _per_type(lm.get("rope_parameters")).values()):
            return "ssmax"
    return None
```
- apply note: in `_feats`, after the `nope_layers` line: "learned_logpos" -> `f.add("q_temp")`, "ssmax" -> `f.add("q_temp_nope")`. Add both keys to FEATURES. In `dims_of`, add `pos_gate` (the number after "@", 0 otherwise) beside `max_context`. Add it to `_point_key_dims` only `if USE_CONTEXT`, as max_context is. `_per_type` is model_graph's `_rope_layer_types`.

## 3. Relation rule
- target side:
  - gated forms: exact on any host while the context is below pos_gate (the factor is identically 1). The same decision as DSA treated as dense: the headline (USE_CONTEXT off) is unchanged, and under USE_CONTEXT the credited context is capped at min(max_context, pos_gate).
  - `q_temp`: wall. log(pos+1) cannot fold into W_q, and the factor is not 1 at any position past 0 unless scale_h = 0.
  - `q_temp_nope`: wall.
- host side:
  - gated: free. llama_4_scaling_beta, attn_scale and floor_scale are scales; Qwen's seq_length is the gate size and set ≥ the context it is never reached; beta = 0 / attn_scale = 0 give exactly 1.
  - `q_temp`: free (scale_h = 0, bias_h = 1 are weights).
  - `q_temp_nope`: wall (fixed in code on the host's NoPE layers).
- kind: gated exact (context-capped); q_temp wall / exact; q_temp_nope wall / wall.
- cost: none.
- free by the config rules: beta, attn_scale, floor_scale and seq_length are scales / sizes (free). `use_logn_attn` and `attn_temperature_tuning` are feature switches (not free), but they are not needed, since the scale lever already reaches 1. Beam's constants are code (not free).
- theorem: gated = `scalar_fold` (factor 1) on positions < pos_gate, the same as `topk_all` for DSA. Needs theorem: `learned_logpos_not_plain`, i.e. q·k·(s log(t+1)+b) for s ≠ 0 is matched by no position-free bilinear logit (one head, two positions).

## 4. Test fixtures
- flag:
  - Qwen/Qwen-7B -> logn@8192
  - Qwen/Qwen-14B, Qwen/Qwen-VL -> logn@2048 (fires inside mpe 8192)
  - Qwen/Qwen-Audio -> logn@2048 (= mpe, never fires)
  - ArmGPT/ArmenianGPT-1.0-3B, RedHatAI/Ministral-3-14B-Instruct-2512-FP8-dynamic, nvidia/Nemotron-Labs-Diffusion-8B -> llama4_beta@16384
  - mlx-community/Devstral-Small-2-24B-Instruct-2512-OptiQ-4bit, mlx-community/Mistral-Small-4-119B-2603-OptiQ-2bit -> llama4_beta@8192
  - lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-8bit -> nope_temp@8192 (keys omitted, so the class defaults apply)
  - mlx-community/Mistral-Large-4-1T-A52B-Text-4bit -> learned_logpos
  - reflection/Beam-1.0-501B-A23B -> ssmax
- not flag:
  - Qwen/Qwen-72B and -72B-Chat (use_logn_attn false)
  - nvidia/Mistral-Medium-3.5-128B-NVFP4 (beta 0)
  - saidutta69/Ministral-8B-Instruct-heretic (ministral, the parent family: no temperature)
  - meta-llama/Llama-3.1-8B
  - Qwen/Qwen2.5-0.5B

## Verdict
Real, but the over-credit is mostly context-gated.
- Headline (USE_CONTEXT off):
  - only q_temp (2 Mistral-Large-4 points: they lose every non-q_temp host)
  - q_temp_nope (2 Beam points: as targets they lose every host, as hosts they lose every target without it; their NoPE layers are already walled by nope_layers, so the change is small)
- The 9 Qwen-1 logn, 15 llama4_beta and 1 nope_temp configs keep their edges. Under USE_CONTEXT their credited context drops to pos_gate (2048 / 8192 / 16384).
