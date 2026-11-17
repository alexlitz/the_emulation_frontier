# mla_k_rope_norm
live 1 / placed 1 (dots3_note 1); direction both; batch a8_mla_dsv4

## 1. What the code computes
- tfmain:models/dots3_note/modular_dots3_note.py:489 `self.k_rope_only_layernorm = Dots3NoteRMSNorm(self.qk_rope_head_dim, eps=config.rms_norm_eps)`, built unconditionally in `Dots3NoteTextAttention.__init__` (no config switch); the same class serves the full and the sliding layers (per_layer_config, :479-485).
- Dots3NoteRMSNorm is a WEIGHTED RMSNorm (modeling_dots3_note.py:60 `self.weight = nn.Parameter(torch.ones(hidden_size))`, :67-68 `x * rsqrt(mean(x^2) + eps) * weight`).
- forward :516-523: `kv_pass, k_rot = split(kv_a_proj_with_mqa(x), [kv_lora_rank, qk_rope_head_dim])`; `k_rot = k_rope_only_layernorm(k_rot)` BEFORE interleaved RoPE (:525-527). So the one shared rotary key head (64 dims) is unit-RMS per token (times a learned gain); no other MLA port normalises k_rot (DeepSeek-V2/V3/V3.2, Kimi, GLM-4.7-Flash, A.X-K1/K2 pass k_rot raw).
- q side unchanged: q_rot comes from q_b_proj(q_a_layernorm(q_a x)) with no extra norm (:510-514).
- Also in this forward (not this finding): latent scales `q_lora_scale = (hidden / q_lora_rank)^0.5`, `kv_lora_scale` (:490-491, :512, :519) are constants that fold into q_b / kv_b (exact); per-head sigmoid gate g_proj (:576-577) is the out_gate feature (mapped elsewhere). `apply_mla_qkv_lora_rescale` in the config is not read by the tfmain config (grep: only the modular file's constants apply).
- p5 today: dots3_note has no use_qk_norm key and is not in QKNORM_ALWAYS, so classify gives `qknorm = "no"` (p5:model_graph.py:2332-2342).

## 2. Classifier change
- reads: model_type dots3_note (code fact; the norm has no key), kv_lora_rank (MLA path)
- new: qknorm slot value `"mla_krope"` (QKNORM_VARIANTS), in classify via QKNORM_VARIANT_BY_TYPE + a code-fixed qknorm entry
- detection:
```python
def detect_mla_k_rope_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "dots3_note" and lm.get("kv_lora_rank"):
        return "mla_krope"
    return None
```
- apply note: p5:model_graph.py: add `"mla_krope"` to `QKNORM_VARIANTS` (:224); add `"dots3_note"` to `QKNORM_ALWAYS` (:1852, so qknorm becomes "yes") and `"dots3_note": "mla_krope"` to `QKNORM_VARIANT_BY_TYPE` (:1879). `QKNORM_ORDER` is generated from QKNORM_VARIANTS (no nest entry: no other variant contains it). Add `"mla_krope"` to `EPS_LEVER_QKNORM` (:3786): it is a gained RMSNorm with eps = rms_norm_eps > 0.

## 3. Relation rule
- target side: a dots3_note target needs a host whose shared rotary key is RMS-normed per token with a gain (only dots3_note): every other host is walled (qknorm "no" -> "mla_krope" is the regime edge, approx, not payable, as for every qknorm variant).
- host side: a dots3_note host plays a plain-MLA target by switching k_rope_only_layernorm off with the eps lever (kv_a_proj's rope rows / S, norm weight S sqrt(eps)): a limit, payable under the lever, as for the other gained qk-norms. Its other MLA facts (latents, q_lora_scale folds) are unchanged.
- kind: wall (target, exact slice); limit (host, eps lever)
- cost: none
- free by the config rules: the eps is free (the lever); the norm itself is code (no switch), so switching it is the lever, not a config flag
- theorem: `knorm_not_plain` (PROOFS.md "qk-norm target <- plain host", PROVED IMPOSSIBLE) for the target wall; `epsLever_bound`, `leverAttn_layer_bound` ("qk-norm / v-norm / MLA latent norm on the host, plain target") for the host lever (k = the rope key; RoPE after the norm is inside B(q, k))

## 4. Test fixtures
- flag: dots-studio/dots3-note-prev -> "mla_krope" (the only dots3_note config)
- not flag: rednote-hilab/dots.llm1.base (dots1 parent family, GQA, no MLA); deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-V3.2 (DeepseekV3Attention, the base class, raw k_rot); skt/A.X-K1, skt/A.X-K2-NVFP4 (MLA, no rope-key norm); axteam/GLM-4.7-Flash-NVFP4A16 (glm4_moe_lite MLA)

## Verdict
real (both): as a target dots3-note-prev loses the plain-MLA hosts it is credited from today (MLA + DSA + out_gate + sink hosts: a handful at most, e.g. Kimi / A.X MLA hosts with the same dims); as a host it keeps plain-MLA targets only through the eps lever (payable limit, so the headline is unchanged where the lever is on). Net: fewer edges, roughly 0-5 pairs.
