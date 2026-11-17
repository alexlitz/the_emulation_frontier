# dynamic_ntk_in_context
live 11 / placed 13 (meditor 1, meditor2 1, qwen 11); direction over-credit; batch a5_positions

## 1. What the code computes
- Qwen-1 `Qwen/Qwen-7B-Chat/modeling_qwen.py` (Qwen-VL and Qwen-Audio are the same: hunks 7032df424ae7, ddc1a6b06fe6, 61c4ae5833be):
  - `:735-739`: `get_ntk_alpha(L) = max(2^ceil(log2(L/seq_length) + 1) - 1, 1)`. This is 1 for L ≤ seq_length, 3 up to 2·seq_length, then 7, ...
  - `:833-850`: in eval with `use_dynamic_ntk`, alpha comes from the (per-sample, unpadded) prefill length L, and decode reuses the cached alpha.
  - `:1287-1298`: `base = rotary_emb_base * alpha^(dim/(dim-2))` whenever alpha changes.
  - Every position's rotation therefore depends on the total prompt length once L > seq_length. The snapshots set seq_length 2048 (14B, VL; mpe 8192) or 8192 (7B mpe 32768; 1.8B mpe 8192), and Audio 2048 (mpe 2048).
- Meditor-2 `Yo-Purple/Meditor-2-3.4B-A1.8B-HMoE/modeling_meditor2.py:71-93,130,147`: on global layers (every global_attn_every_n_layers-th) with L = max(position_ids)+1 > 8192, `base' = base·(L/8192)^(dim/(dim-2))`. The threshold is fixed in code (`original_max_position_embeddings=8192`).
- Meditor-1 `Yo-Purple/Meditor-1-2.2B-A500M-MoE/modeling_meditor.py:60-85,136-141`: the same with threshold 32768. The finding's 8192 is only Meditor-2.
- Inside the threshold every one of these is plain RoPE with the configured base.
- round2-p5 reads none of this: `use_dynamic_ntk` is not referenced, and meditor / meditor2 give plain pos=rope. Not fixed in p5.
- This is the same pattern as position_query_temperature: an identity below a length and a length-dependent change above it.

## 2. Classifier change
- reads: model_type qwen + `use_dynamic_ntk` + `seq_length`; model types meditor / meditor2 with their remote auto_map (code-fixed 32768 / 8192).
- new: dims `pos_gate` (the same field as position_query_temperature: min over a model's gates), in `dims_of`. It is read only under USE_CONTEXT. There is no FEATURES key.
- detection:
```python
def detect_dynamic_ntk_in_context(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "qwen" and lm.get("use_dynamic_ntk") is True:
        return f"dyn_ntk@{num(lm.get('seq_length')) or 8192}"
    if model_type == "meditor" and "modeling_meditor" in auto_map(cfg):
        return "dyn_ntk@32768"
    if model_type == "meditor2" and "modeling_meditor2" in auto_map(cfg):
        return "dyn_ntk@8192"
    return None
```
- apply note: in `dims_of`, set `pos_gate = min(nonzero gates)` from this detector and from position_query_temperature (Qwen-1 has both, both at seq_length). Under USE_CONTEXT, cap the target's `max_context` demand at pos_gate (emul_matrix's USE_CONTEXT block, p5:model_graph.py:4511). A host with a gate caps what it supplies the same way unless the gate is config (Qwen's seq_length: free). Nothing changes in the headline.

## 3. Relation rule
- target side: exact on any RoPE host for contexts ≤ pos_gate (the table is the plain one). Beyond it, a length-dependent table that no fixed-table host forms. This is the DSA precedent: credited, with the context capped at pos_gate under USE_CONTEXT.
- host side:
  - Qwen-1: free. seq_length is a size; set it ≥ the context and alpha = 1 always. use_dynamic_ntk is a switch, but it is not needed.
  - Meditor: the gate is code-fixed. Its hosts supply plain RoPE only up to 8192 / 32768, so their own max_context is capped at the gate.
- kind: exact (context-capped).
- cost: none.
- free by the config rules: rope scaling and seq_length are free. use_dynamic_ntk (a switch) and Meditor's constant are not.
- theorem: inside the gate the table is the configured one: `defaultFreq_match` (identity, alpha = 1). Needs theorem (only if a wall is wanted beyond the gate): `dyn_ntk_not_fixed_table`, i.e. no fixed inv_freq reproduces rotations whose base depends on L for two lengths L1 ≠ L2 > seq_length.

## 4. Test fixtures
- flag (all live repos):
  - Qwen/Qwen-14B, Qwen-14B-Chat, Qwen-VL, Qwen-VL-Chat, Qwen-Audio, Qwen-Audio-Chat -> dyn_ntk@2048
  - Qwen/Qwen-7B, Qwen-7B-Chat, Qwen-1_8B -> dyn_ntk@8192
  - Yo-Purple/Meditor-1-2.2B-A500M-MoE -> dyn_ntk@32768
  - Yo-Purple/Meditor-2-3.4B-A1.8B-HMoE -> dyn_ntk@8192
- not flag:
  - Qwen/Qwen-72B, Qwen-72B-Chat (use_dynamic_ntk false: the placed non-live carriers)
  - Qwen/Qwen2.5-0.5B (qwen2: no dynamic NTK)
  - meta-llama/Llama-3.1-8B

## Verdict
Real, context-only.
- Headline relation (USE_CONTEXT off): no edge change, under the DSA-as-dense decision.
- Under USE_CONTEXT, 11 targets have their credited context capped at 2048 / 8192 / 32768. Five of them are already capped by max_position_embeddings = seq_length, so the real change is the 6 Qwen-1 with seq_length < mpe, plus Meditor (mpe 524288 -> 8192 / 32768).
- If a hard wall is preferred beyond the gate instead, the 11 targets lose every host without the same dynamic scheme: fewer edges.
