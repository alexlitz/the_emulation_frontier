# identity_experts
live 3 / placed 3 (<none> 3); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
- Router: `classifier = Linear(hidden, n_routed_experts + zero_expert_num)`, softmax over ALL slots, top-k chosen on `scores + e_score_correction_bias` (selection only), weights = the unrenormalised softmax scores times routed_scaling_factor (tf5:models/longcat_flash/modeling_longcat_flash.py:143-168; remote meituan-longcat/LongCat-Flash-Thinking/modeling_longcat_flash.py:102-103).
- Experts: slots `>= n_routed_experts` return `identity_expert(x)` = x. The output is `sum_{k in topk} w_k * (E_k(x) or x)` (tf5 modeling_longcat_flash.py:171-218; remote modeling_longcat_flash.py:146-170). The LongCat-2601 remote also allows `zero_expert_type == 'drop'` (zero output, modeling_longcat.py:194-200); every snapshot config uses `identity`.
- Snapshot: Flash-Chat / Thinking / Thinking-2601 have 512 SwiGLU experts + 256 identity slots, top-12, scale 6. LongCat-2.0 (768 + 128) and Flash-Lite-Sparse (256 + 128) carry the same keys but are not placed: no model_type and no remote code in the snapshot (flagged by config; the carrier list has only the 3).
- p5: `_moe_sig` E counts 512 (the identity slots are invisible), and no feature marks them. The configs have NO model_type, so `RENORM_FIXED["longcat_flash"] = "P"` (p5:model_graph.py:865) and the selection bias are never applied. renorm is "?" (only the same family) and bias is 0.

## 2. Classifier change
- reads: `zero_expert_num`, `zero_expert_type` (any type: the LongCat configs are type-less).
- new: MOE_SIG field `ident` (identity-slot count) in `_moe_sig`, and FEATURES key `identity_experts` (record) in `_feats`.
- detection:
```python
def detect_identity_experts(cfg, model_type):
    lm = lm_of(cfg)
    z = num(lm.get("zero_expert_num"))
    if z > 0 and str(lm.get("zero_expert_type") or "identity").lower() == "identity":
        return z
    return None
```
- apply note: `_moe_sig` (~908): append `ident = detector value or 0` to the tuple and `"ident"` to `MOE_SIG` (~950). In `moe_why` (~954), after `m` is computed: `if (T["ident"] or H["ident"]) and H["ident"] != m * T["ident"]: return "identity experts ..."`. With identity slots the softmax normalises over E + Z slots, so in the `P`/softmax branch (~1000) compare `Eh + H["ident"] == m * (Et + T["ident"])`. Separately (a classification fix the same apply can carry): key LongCat by `architectures` (LongcatFlashForCausalLM / LongcatCausalLM) as `longcat_flash`, so that RENORM_FIXED P and `e_score_correction_bias` (ROUTER_BIAS_TYPES) apply.

## 3. Relation rule
- target side: an identity slot outputs w_k x. A SwiGLU expert silu(g) * u has no constant channel under a pre-norm, so it cannot equal the identity on an input ball (bilinear in x near 0, not linear). The slot also takes softmax mass in the unrenormalised denominator. So a host needs identity slots: H.ident = m T.ident, the m copies of each target slot carrying the target's router row. The softmax then gives each copy p/m, and the m copies sum to p (`moeR_copies_topk`, the `P` duplication rule).
- host side: a LongCat host playing a target without identity slots must keep its Z slots out of the top-k (the selection bias does this). But under an unrenormalised softmax their e^0 terms still sit in the denominator, which varies per token (refuted dummy experts, PROOFS "MoE <- MoE routing" correction (1): a limit, not exact). So it is a wall.
- kind: wall both ways, except exact when H.ident = m T.ident (plus the existing MoE rules).
- cost: none.
- free by the config rules: routed_scaling_factor is free; zero_expert_num is a feature switch (a module count), so it is not free.
- theorem: host-side wall `moeR_dup_plain` / the dummy-expert refutation (MoERouting.lean). Target wall needs theorem `swiglu_not_identity`: no SwiGLU unit bank without a bias equals x on a neighbourhood of 0 (its Taylor series has no linear term). Exact identity duplication: `moeR_copies_topk` with identity Y_e.

## 4. Test fixtures
- flag: meituan-longcat/LongCat-Flash-Chat, LongCat-Flash-Thinking, LongCat-Flash-Thinking-2601 -> 256; meituan-longcat/LongCat-2.0, LongCat-Flash-Lite-Sparse -> 128 (same keys, unplaced).
- not flag: deepseek-ai/DeepSeek-V4-Flash (no identity slots), Qwen/Qwen3-30B-A3B, openai/gpt-oss-20b.
- scan: exactly the 5 LongCat repos.

## Verdict
real. Fewer edges: the 3 placed LongCat points lose every non-LongCat MoE host and target. Today they match only each other (renorm "?" for type-less configs), so the visible change is small (about 3 points). What the fix mainly settles is that the LongCat <-> LongCat edges require equal identity counts.
