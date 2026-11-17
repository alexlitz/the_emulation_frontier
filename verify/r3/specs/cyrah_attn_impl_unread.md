# cyrah_attn_impl_unread
live 1 / placed 1 (cyrah 1); direction both; batch a3_placement

## 1. What the code computes
- Lilgarca/my-model-v01/modeling_cyrah.py:376-395: `impl = config.attn_impl` (snapshot "gqa"): plain GQA,
  q_proj h x (nope + rope) = 16 x 64, k / v_proj num_key_value_heads (8) x 64, `q_norm` / `k_norm` =
  CyrahRMSNorm(head_dim) shared across heads (per-head weighted RMSNorm), `gate = Linear(d, h * vd, bias=True)`
  when use_output_gate; then `return` -- no q / kv LoRA latent, no indexer (index_q / index_k are built only after the
  return, :413-415).
- modeling_cyrah.py:432-454 forward (gqa): q, k RMS-normed then RoPE (`apply_rope`, rotate_half since
  rope_interleave false); repeat_interleave for GQA; softmax(q k^T x scale + mask); `+ _window_mask` only
  `if self.sliding`; out o * sigmoid(gate(x)); no DSA (`_sparse_keep` only in the MLA branch, :478-480), no XSA.
- modeling_cyrah.py:529: `sliding = (layer_type == "sliding_attention")`; snapshot layer_types are conv /
  full_attention only, so no layer is windowed although sliding_window = 256.
- RoPE: rotary_dim 64 (CyrahModel :632) = qk_nope_head_dim 0 + qk_rope_head_dim 64 = head_dim: full RoPE.
- p5 classify (p5:model_graph.py :2131): kv MLA (kv_lora_rank set), pos partial_rope (rotary_dim key), window local
  (sliding_window key), qknorm no, feats dsa (index_topk) -- all wrong for attn_impl gqa.

## 2. Classifier change
- reads: model_type cyrah, attn_impl, num_key_value_heads, qk_*_head_dim, rotary_dim, layer_types
- new: slots kv=GQA, pos=rope, window=full, qknorm=head; feats drop dsa; dims drop the MLA latent (kv from
  num_key_value_heads); in classify / _feats / dims_of
- detection:
```python
def detect_cyrah_attn_impl_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "cyrah" or str(lm.get("attn_impl") or "mla").lower() != "gqa":
        return None
    nh = num(lm.get("num_attention_heads"))
    nkv = max(1, num(lm.get("num_key_value_heads")) or 1)
    kv = "MHA" if nkv == nh else ("MQA" if nkv == 1 else "GQA")
    hd = num(lm.get("qk_nope_head_dim")) + num(lm.get("qk_rope_head_dim"))
    rot = num(lm.get("rotary_dim")) or num(lm.get("qk_rope_head_dim"))
    pos = "rope" if hd and rot >= hd else "partial_rope"
    window = "local" if any(t == "sliding_attention" for t in layer_types(lm)) else "full"
    return f"kv={kv},pos={pos},window={window},qknorm=head"
```
- apply note: p5:model_graph.py `_normalize` (:2543): for mt == "cyrah" and attn_impl == "gqa" drop kv_lora_rank /
  q_lora_rank / index_topk / index_n_heads (set None and filter), set head_dim = nope + rope, and if rotary_dim >=
  head_dim drop rotary_dim; if no layer_types entry is "sliding_attention" set sliding_window None (the
  layer_types branch of classify's window test then sees no sliding layer). Add "cyrah" to QKNORM_TYPES with
  QKNORM_VARIANT_BY_TYPE (:1879) "head". This fixes classify, _feats (dsa) and dims_of together.

## 3. Relation rule
- target side: a GQA, full-RoPE, qk-norm head, out-gated hybrid (conv + softmax) target: needs a qk-normed, gated
  host of the shortconv family (see cyrah_shortconv_is_lfm2)
- host side: a GQA host (tiling plays MQA / GQA), qk-norm head host (eps lever for plain targets), out_gate free
- kind: exact (existing rules)
- cost: none (KV capacity from 8 KV heads x 64 instead of the 32-wide latent)
- free by the config rules: rope theta, window size free; attn_impl is a feature switch
- theorem: `mqa_sub_gqa`, `gqa_sub_mha`; `qknorm_logit` / `knorm_not_plain`; `out_gate_off`; `local_sub_full`

## 4. Test fixtures
- flag: Lilgarca/my-model-v01 -> kv=GQA,pos=rope,window=full,qknorm=head
- not flag: LiquidAI/LFM2-350M (the conv + GQA family it resembles), deepseek-ai/DeepSeek-V2-Lite (a real MLA),
  Qwen/Qwen3-0.6B (qk-norm GQA, different type)

## Verdict
real: one point moves (MLA / local / dsa / no-qknorm -> GQA / full / qknorm head): as a target it now needs a qk-norm
host (fewer edges), as a host it drops the MLA-host limit rules and the local window (more edges where GQA hosts are
needed). 1 model, both directions. Region: attention.
