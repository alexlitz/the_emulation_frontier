# act_clip_order
live 2 / placed 2 (step3p5 2); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
- post (clamp AFTER SiLU): Step-3.5 `Step3p5MLP.forward` `gate = silu(g).clamp(max=L)`, `up.clamp(-L, L)` (stepfun-ai/Step-3.5-Flash/modeling_step3p5.py:224-231), and the same in `Step3p5MoEMLP.get_expert_output` (:324-332). The per-layer limits come from swiglu_limits / swiglu_limits_shared, with 0 / None meaning none (:556-577). Also tf5 Step3p7 `Step3p7MLP.forward` / `Step3p7Experts._apply_gate` (tf5:models/step3p7/modular_step3p7.py:789-792, 804-808) and Ling-3 `BailingMoeV3MLP.forward` (inclusionAI/Ling-3.1-flash/modeling_bailing_moe_v3.py:342-348).
- pre (clamp BEFORE SiLU): DeepSeek-V4 `gate_proj(x).clamp(max=L)` then silu (tf5:models/deepseek_v4/modeling_deepseek_v4.py:1022-1027, 1066-1067); GLM-5-Next (tf5:models/glm5_next/modeling_glm5_next.py:98-104, 140-141); Hy4 `_apply_gate` (tf5:models/hy_v4/modeling_hy_v4.py:582-586); MemT-1.0-Flash inherits the DeepSeek-V4 module (MemTensor/MemT-1.0-Flash/modeling_memt.py:34).
- p5 `_feats` (p5:model_graph.py:1826-1828) sets one `act_clip` for both orders. FEATURES act_clip is (wall, wall), so clamped <-> clamped edges are credited regardless of order.

## 2. Classifier change
- reads: the clamp keys + model_type.
- new: none proposed (the detector records the order, as a dims field `act_clip_order` if ever needed).
- detection:
```python
_CLIP_PRE = {"deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4", "glm5_next", "glm5_next_text",
             "hy_v4"}
_CLIP_POST = {"step3p5", "step3p7", "bailing_hybrid"}

def detect_act_clip_order(cfg, model_type):
    lm = lm_of(cfg)
    if not _clip_limits(lm):          # any nonzero swiglu_limit(s)(_shared) / *_swiglu_limit_list entry
        return None
    if model_type in _CLIP_PRE:
        return "pre"
    if model_type in _CLIP_POST:
        return "post"
    return None
```
- apply note: no change to p5:model_graph.py for the order. See the Verdict for the real residual.

## 3. Relation rule
- The orders convert exactly, given that the limit is a free cap and the up weights are free.
  - pre host <- post target (L_t): set the host limit L' = silu^-1(L_t) (> L_t), multiply the up rows by a = L'/L_t and the down columns by 1/a. Gate: silu(min(g, L')) = min(silu(g), L_t) for every g (silu is increasing on [-1.28, inf) and negative below, while L_t > 0). Up: clamp(a u, -L', L') / a = clamp(u, -L_t, L_t). Exact.
  - post host <- pre target (L_t): host limit L' = silu(L_t), a = L'/L_t. min(silu(g), silu(L_t)) = silu(min(g, L_t)), and the up clamp rescales the same way. Exact.
- kind: exact both ways (order alone).
- cost: none.
- free by the config rules: swiglu_limit(s) is a cap, so it is free (PROOFS round-2 act_clip row: "c is config"). Up / down scaling is weights.
- theorem: needs theorem `act_clip_order_swap`: for L > 0, silu(min(g, silu^-1(L))) = min(silu(g), L) and min(silu(g), silu(L)) = silu(min(g, L)), plus `ffn_smul` for the up / down rescale.

## 4. Test fixtures
- flag: stepfun-ai/Step-3.5-Flash, stepfun-ai/Step-3.5-Flash-Base-Midtrain, inclusionAI/Ling-3.1-flash, TensorFold/Ling-3.0-flash-oQ4 -> "post"; deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4.1-Flash, local-inference-lab/GLM-5.3-Flash-NVFP4-Spark, tencent/Hy4-preview-FP8, MemTensor/MemT-1.0-Flash -> "pre".
- not flag: openai/gpt-oss-20b, nvidia/MiniMax-M3-NVFP4 (swigluoai clamps: not act_clip); LGAI-EXAONE/K-EXAONE-2.0-750B-A37B (swiglu_limits key, but the tf5 exaone_moe code has no clamp: order unknown); rapid-mlx/Ling-3.0-tiny-MLX-4bit (lists null); Qwen/Qwen3-30B-A3B.

## Verdict
non-issue: the clamp order converts exactly through the free limit and an up / down weight rescale, so pre <-> post edges are correct. The real residual is the per-layer / per-branch limit PATTERN, which the act_clip feature does not compare. DeepSeek-V4 / GLM-5-Next / Hy4 have ONE swiglu_limit for every layer, routed and shared, and V4's code requires a finite limit (`clamp(max=None)` with `min=None` raises). Step-3.5 clamps only its last layers, at 7 (routed) and 16 (shared), and Ling-3 at 10 / 4-7. A uniform-limit host cannot leave the target's unclamped layers unclamped (only in the regime of `act_clip_regime`). It cannot give routed and shared different gate caps either: the gate is not rescalable, though the up cap is. So post-pattern targets on V4 / GLM / Hy4 hosts are over-credited. Suggest (a follow-up, not this finding) a dims field `act_clip_sched` = sorted set of (layer, branch) clamped, and a host rule "host per-layer / per-branch limits are config (lists) or uniform-and-target-uniform". K-EXAONE-2.0 gets act_clip from swiglu_limits, but no public code clamps; flag it for the a-tables batch.
