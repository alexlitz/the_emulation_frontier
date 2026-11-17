# glm5_next_mla_no_head_qknorm
live 7 / placed 7 (glm5_next_text 7: 6 under outer glm5_next, 1 under glm_superexpert); direction both; batch a1_typetables

## 1. What the code computes
- tf5:models/glm5_next/modeling_glm5_next.py:1102 Glm5NextTextAttention (the softmax layers): :1128-1130 `q_a_proj`, `q_a_layernorm = RMSNorm(q_lora_rank)`, `q_b_proj`; :1132-1137 `kv_a_proj_with_mqa`, `kv_a_layernorm = RMSNorm(kv_lora_rank)`; :1188-1193 `query = q_b_proj(q_a_layernorm(q_a(x)))`, `k_pass = kv_a_layernorm(kv_pass)`. No per-head q_norm / k_norm: DeepSeek-V3.2-style MLA whose only norms are the latent ones, which kv=MLA already carries.
- The other q / k norms in the file are not on softmax layers of the text model: :801 / :836 `k_norm = LayerNorm(head_dim)` is the DSA indexer's key (DSA is dense by decision); :1613-1614 q_norm / k_norm are Glm5NextVisionAttention (vision tower); the KDA linear layers l2-normalise q / k inside the recurrence (:481-483, :539-541, `use_qk_l2norm_in_kernel=True` :747, :759) -- part of rf_kda, not the qknorm slot.
- configuration_glm5_next.py has no qk-norm key.
- p5 lists glm5_next / glm5_next_text in QKNORM_ALWAYS (p5:model_graph.py:1856) and QKNORM_HEAD_CHECKED (:1898): classify gives qknorm=head.

## 2. Classifier change
- reads: model_type
- new: qknorm slot "no" (drop the two types from QKNORM_ALWAYS and QKNORM_HEAD_CHECKED)
- detection:
```python
def detect_glm5_next_mla_no_head_qknorm(cfg, model_type):
    return "no" if model_type in ("glm5_next", "glm5_next_text") else None
```
- apply note: p5:model_graph.py delete "glm5_next", "glm5_next_text" from QKNORM_ALWAYS (:1856) and QKNORM_HEAD_CHECKED (:1898). Nothing else reads them for qk-norm. (The outer glm_superexpert wrapper resolves to glm5_next_text through _lm_unwrap already.)

## 3. Relation rule
- target side: these 7 targets no longer need a qk-norm host: MLA hosts without per-head norms (DeepSeek-V3 / V3.2, GLM-5 glm_moe_dsa, Kimi) become admissible on this slot (they stay walled by hyper_conn, act_clip and rf_kda, see verdict)
- host side: they no longer claim head-qknorm targets (over-credit removed); a plain target is played without the eps lever
- kind: exact (MLA <- MLA on this slot); the removed head edges had no construction
- cost: none
- free by the config rules: n/a (no key)
- theorem: `knorm_not_plain` (PROOFS.md "qk-norm target <- plain host", why the old host side was wrong); the MLA latent norms are carried by kv=MLA (`mla_latent_norm_not_affine` row)

## 4. Test fixtures
- flag (every live repo): 12fn/GLM-5.3-Flash-NVFP4-W4A16-2x-RTX-PRO-6000, INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound, avlp12/GLM-5.3-Flash-Alis-MLX-4bit, dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16, defnoteatin/PseudoBrain-GLM-5.3-Flash-Chronos-v7.5, local-inference-lab/GLM-5.3-Flash-NVFP4-Spark, shashanth-27xA9/GLM-5.3-SuperExpert-v2-9.2GB -> "no"
- not flag: zai-org/GLM-5.3 (glm_moe_dsa: plain MLA, already no); zai-org/GLM-4.5-Air (glm4_moe: use_qk_norm switch); deepseek-ai/DeepSeek-V3.2

## Verdict
real, both directions, but no frontier change today: GLM-5-Next is isolated by hyper_conn (wall both ways), act_clip and its rf_kda recurrent layers. Correct the slot so the record (and any later hyper_conn pricing) is right.
