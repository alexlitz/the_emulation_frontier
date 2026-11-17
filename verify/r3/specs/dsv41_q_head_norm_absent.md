# dsv41_q_head_norm_absent
live 24 / placed 24 (deepseek_v41_text 24: 23 under outer deepseek_v41, 1 flat); direction over-credit; batch a1_typetables

## 1. What the code computes
- V4 (reference): ext deepseek_v41/hub__deepseek-ai__DeepSeek-V4-Flash/inference/model.py:458 `q_norm = RMSNorm(q_lora_rank)` (latent), :461 `kv_norm = RMSNorm(head_dim)`; forward :496-498 `q = wq_b(q_norm(wq_a x))` then `q *= rsqrt(mean(q^2) + eps)` -- a weight-free per-head q norm; :503 `kv = kv_norm(kv)` (the one K = V head, weighted). This is p5's "dsv4" variant (p5:model_graph.py:215: weight-free per-head q norm + weighted K = V norm).
- V4.1: ext deepseek_v41/hub__deepseek-ai__DeepSeek-V4.1-Flash/inference/model.py:642 latent q_norm, :643 wq_b, :645 kv_norm; forward :771-773 `qr = q_norm(wq_a x)`, `q = wq_b(qr)`, `apply_rotary_emb(q[..., -rd:])` -- NO per-head rsqrt on q; :706 `kv = kv_norm(wkv x)`. The MTP block likewise (:1059-1062). So V4.1 has only the weighted K = V norm.
- p5 maps deepseek_v41 / deepseek_v41_text to "dsv4" (QKNORM_VARIANT_BY_TYPE :1887) and keeps vnorm (VNORM_TYPES :1941: right, K = V).

## 2. Classifier change
- reads: model_type
- new: QKNORM_VARIANTS value "dsv4_kv" (weighted norm of the single K = V head, no q norm), in QKNORM_VARIANT_BY_TYPE
- detection:
```python
def detect_dsv41_q_head_norm_absent(cfg, model_type):
    return "dsv4_kv" if model_type in ("deepseek_v41", "deepseek_v41_text") else None
```
- apply note: p5:model_graph.py add "dsv4_kv" to QKNORM_VARIANTS (:224) and set `deepseek_v41`, `deepseek_v41_text` -> "dsv4_kv" in QKNORM_VARIANT_BY_TYPE (:1887, replacing their "dsv4"); keep them in QKNORM_ALWAYS and VNORM_TYPES. Add "dsv4_kv" to EPS_LEVER_QKNORM (:3786): its only norm is weighted, so a V4.1 host levers it off (the vnorm part is already lever-capable).

## 3. Relation rule
- target side: a dsv4_kv target needs a dsv4_kv host (no other variant has a k-only norm on a K = V head); a dsv4 host cannot drop its weight-free q norm (not lever-capable), so dsv4 -> dsv4_kv is a wall
- host side: a dsv4_kv host cannot play a dsv4 target (it has no q norm to add); it plays plain-norm targets only by the eps lever (limit)
- QKNORM_NEST: none (dsv4_kv nests in nothing and contains nothing)
- kind: wall (dsv4 <-> dsv4_kv both ways); limit (dsv4_kv host -> plain, eps lever)
- cost: none
- free by the config rules: no
- theorem: `knorm_not_plain` (PROOFS.md "qk-norm target <- plain host": a k-norm cannot be dropped exactly); `epsLever_bound` / `leverAttn_model_limit` (the lever); needs theorem: `q_headnorm_not_plain` -- a weight-free per-head q norm is not removable / addable by weights (the same per-token rescaling argument as `qknorm_out_of_regime`, on q alone)

## 4. Test fixtures
- flag: Infatoshi/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored, LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw, OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated (flat deepseek_v41_text), TensorFold/DeepSeek-V4.1-Flash-MLX-2bit-MTP, amd/DeepSeek-V4.1-Flash-Quark-MXFP4, deepseek-ai/DeepSeek-V4.1-Flash, deepseek-ai/DeepSeek-V4.1-Flash-1017 -> "dsv4_kv"
- not flag: deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4-Flash-0731 (V4: dsv4, keeps the q norm); deepseek-ai/DeepSeek-V3.2 (plain MLA)
- scan: exactly the 24 live repos.

## Verdict
real, but no frontier change today: the V4 / V4.1 family is isolated by hyper_conn (wall both ways), kv_compress and hash_route; the change removes the V4 <-> V4.1 qk-norm equivalence (which those walls also block for V4.1 vs V4 differences in engram). Encode for the record; if hyper_conn is ever priced, V4.1 <-> V4 edges must stay walled on this slot. memt_deepseek_v4 should be checked the same way (not in this batch).
