# ernie_mla_layer_dims_unread
live 1 / placed 1 (ernie5_5_moe 1); direction both; batch a8_mla_dsv4

## 1. What the code computes
(ext:ernie5_5_moe = $NOCODE_CACHE/ernie5_5_moe/gh__baidu__vLLM-ERNIE/, the vLLM-ERNIE plugin)
- config.py:77-81 `compress_ratios` default `[128] * (num_hidden_layers + num_nextn_predict_layers)`; :155-158 `is_dsa_layer(i) = i < len(compress_ratios) and compress_ratios[i] <= 0`. Snapshot: 7 layers at -2 (indices 7, 14, 21, 28, 35, 42 and the MTP slot), the rest 128 (HCA).
- config.py:29-34, :82-87 the DSA / MLA layer dims are separate keys: hybrid_mla_num_attention_heads 64, q_lora_rank 1024, kv_lora_rank 512, qk_nope_head_dim 192, qk_rope_head_dim 64, v_head_dim 256; :162-183 the `mla_*` properties return them ("DSA layer dims, kept apart from the HCA fields of the same names").
- model.py:120-121 `if config.is_dsa_layer(layer_id): self.attn = Ernie5_5DSAAttention(...)`, else the HCA attention.
- attention.py:780-829 `Ernie5_5DSAAttention.__init__`: heads = config.mla_num_heads, q_a -> q_a_layernorm (q_lora_rank) -> q_b_proj (heads x (192 + 64)), kv_a_layernorm (512), kv_b_proj (heads x (192 + 256)), o_gate Linear(q_lora_rank -> heads x v) (out gate on the normed q latent), VHAPostMix, attn_sink, DSA indexer, scale qk_head_dim^-0.5, rotary over qk_rope_head_dim at mla_rope_parameters (plain RoPE at rope_theta, config.py:94-95 comment).
- The top-level keys (qk_nope_head_dim 448, v_head_dim 512, head_dim 512, num_key_value_heads 1) are the HCA layers' (DeepSeek-V4-style K = V head over compressed pools).
- p5 today: dims_of reads the top-level keys for MLA: mla_nope 448, mla_v 512, q_cap 64 x 512 = 32768, kv_cap 576 (p5:model_graph.py:3089-3092, :3315-3320); _attn_variants (:1075-1112) has no hybrid_mla_* entry. The real MLA layers have mla_nope 192, mla_v 256, q heads 64 x 256.

## 2. Classifier change
- reads: compress_ratios (<= 0 entries), hybrid_mla_num_attention_heads, hybrid_mla_q_lora_rank, hybrid_mla_kv_lora_rank, hybrid_mla_qk_nope_head_dim, hybrid_mla_qk_rope_head_dim, hybrid_mla_v_head_dim (Ernie5_5Config defaults 64 / 1024 / 512 / 192 / 64 / 256)
- new: dims fields kv_latent / q_latent / mla_rope / mla_nope / mla_v from hybrid_mla_*, and an _attn_variants entry (heads, 192 + 64, 512 + 64) for the MLA layers, in dims_of / _attn_variants
- detection:
```python
def detect_ernie_mla_layer_dims_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "ernie5_5_moe":
        return None
    cr = lm.get("compress_ratios")
    if not (isinstance(cr, list) and any(isinstance(x, (int, float)) and x <= 0 for x in cr)):
        return None
    g = lambda k, dflt: num(lm.get(k)) if lm.get(k) is not None else dflt     # Ernie5_5Config defaults
    return {"heads": g("hybrid_mla_num_attention_heads", 64), "q_latent": g("hybrid_mla_q_lora_rank", 1024),
            "kv_latent": g("hybrid_mla_kv_lora_rank", 512), "mla_nope": g("hybrid_mla_qk_nope_head_dim", 192),
            "mla_rope": g("hybrid_mla_qk_rope_head_dim", 64), "mla_v": g("hybrid_mla_v_head_dim", 256)}
```
- apply note: p5:model_graph.py `dims_of` (:3315-3320): for mt == "ernie5_5_moe" with a compress_ratios entry <= 0, take kv_latent / q_latent / mla_rope / mla_nope / mla_v from the hybrid_mla_* keys instead of the top-level ones (they are the only MLA layers); `_attn_variants` (:1105 block): append `(hybrid_mla_num_attention_heads, nope + rope, kv_lora_rank + rope, False)` so q_cap / kv_cap still hold the wider HCA shape (64 x 512) through the max. (The HCA layers are not MLA at all; their kv shape stays the top-level one; whether the kv slot should be "MLA|MQA" is a separate finding.)

## 3. Relation rule
- target side: an ERNIE-5.5 target needs host mla_nope >= 192, mla_v >= 256 (was 448 / 512): less demanding
- host side: an ERNIE-5.5 host offers mla_nope 192, mla_v 256 (was 448 / 512): it stops being credited with MLA targets whose per-head widths exceed 192 / 256
- kind: exact (the same MLA width rule: zero pad, host >= target); no new kind
- cost: none
- free by the config rules: none (shapes)
- theorem: the existing MLA width rule: `mla_sub_mha` / `qknorm_pad`, `rmsNorm_eps_inj` (PROOFS.md "kv: MLA -> MLA with a wider latent")

## 4. Test fixtures
- flag: baidu/ERNIE-5.5-Flash-Base -> {heads 64, q_latent 1024, kv_latent 512, mla_nope 192, mla_rope 64, mla_v 256}
- not flag: deepseek-ai/DeepSeek-V4-Flash (compress_ratios with 0 = sliding, not MLA; no hybrid_mla_*); deepseek-ai/DeepSeek-V4.1-Flash; dots-studio/dots3-note-prev (swa_* MLA, read by _attn_variants already); deepseek-ai/DeepSeek-V3.2

## Verdict
real (both), frontier-neutral today: the dims fix is correct, but ERNIE-5.5 is walled from every other model by kv_compress and hyper_conn (wall both ways) and its situ activation, so no edge appears or disappears (0 pairs). Apply for correctness.
