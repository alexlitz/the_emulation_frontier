# no_model_type_longcat
live 3 / placed 3 (<none> 3); direction both; batch a3_placement

## 1. What the code computes
- The five LongCat configs in the snapshot carry no `model_type` (architectures LongcatFlashForCausalLM: Flash-Chat,
  Flash-Thinking; LongcatCausalLM: Flash-Thinking-2601, LongCat-2.0, Flash-Lite-Sparse). The config classes say
  `model_type = "longcat"` (meituan-longcat/LongCat-Flash-Thinking-2601/configuration_longcat.py:95); transformers
  5.18 has the family as longcat_flash (tf5:models/longcat_flash/modeling_longcat_flash.py).
- Router, meituan-longcat/LongCat-Flash-Thinking-2601/modeling_longcat.py:122-153: E = n_routed_experts +
  zero_expert_num (:127); scores = softmax(logits) (:146); selection on scores + e_score_correction_bias (:133, :137,
  a selection-only bias); weights NOT renormalised unless norm_topk_prob (config default False,
  configuration_longcat.py:134; key absent in every snapshot) then x routed_scaling_factor (:152).
- Zero-computation experts, modeling_longcat.py:194-199: experts with index >= n_routed_experts return the input
  (zero_expert_type "identity": `expert_output = expert_input`) or 0 ("drop"). Snapshot: identity, 128 / 256 of them.
- Shortcut-connected MoE block, modeling_longcat.py:514-579 (same in modeling_longcat_flash.py:400-461 and
  tf5 modeling_longcat_flash.py:460-534): per layer two (attention + dense MLP) sublayers;
  x1 = x + attn0(n0 x); m = moe(p0 x1); x2 = x1 + mlp0(p0 x1); x3 = x2 + attn1(n1 x2); out = x3 + mlp1(p1 x3) + m.
  The MoE output joins after the SECOND sublayer, so attn1 / mlp1 do not see it.
- p5 keys all LongCat handling on model_type "longcat_flash" (RENORM_FIXED :865, O_BIAS_FROM_ATTN_BIAS :1936); with
  mt "" the router renorm is "?" and no router_bias, and no rule models the ScMoE block or zero experts.

## 2. Classifier change
- reads: missing model_type + architectures in {LongcatFlashForCausalLM, LongcatCausalLM, LongcatForCausalLM};
  zero_expert_num / zero_expert_type
- new: model_type "longcat_flash" (ARCH_TYPE_FALLBACK table); FEATURES "scmoe" and "zero_expert"; ROUTER_BIAS_TYPES
  += longcat_flash
- detection:
```python
LONGCAT_ARCHS = {"LongcatFlashForCausalLM", "LongcatCausalLM", "LongcatForCausalLM"}
def detect_no_model_type_longcat(cfg, model_type):
    lm = lm_of(cfg)
    if lm.get("model_type") or (cfg or {}).get("model_type"):
        return None
    if set(archs(cfg)) & LONGCAT_ARCHS:
        return "longcat_flash"
    return None
```
- apply note: p5:model_graph.py `_lm_config` (:2684): after `_lm_unwrap`, if the result has no model_type and its
  architectures hit `ARCH_TYPE_FALLBACK = {a: "longcat_flash" for a in LONGCAT_ARCHS}`, set model_type. Add
  "longcat_flash" to ROUTER_BIAS_TYPES (:1930). In `_feats` (:1741): `if mt == "longcat_flash": f.add("scmoe")`;
  `if _num(d.get("zero_expert_num")) > 0 and d.get("zero_expert_type", "identity") == "identity": f.add("zero_expert")`
  ("drop" zero experts are a renorm-free mass sink: price them in moe_sig as extras). FEATURES entries:
  "scmoe": ("wall", "free"), "zero_expert": ("wall", "free").

## 3. Relation rule
- target side: a LongCat target needs a LongCat host (scmoe): a sequential host feeds m into attn1's input. Identity
  zero experts need a host whose router can pick an identity expert (zero_expert).
- host side: a LongCat host plays a plain sequential MoE target exactly (one target layer per host layer: attn1 = 0,
  mlp1 = 0, mlp0 = 0, the MoE reads p0 x1 and is added to the residual), and two dense target layers per host layer
  (MoE output 0). Identity zero experts on the host: e_score_correction_bias is a weight, so -inf on them means they
  are never selected; they stay in the softmax denominator (router rows 0 -> logit 0), i.e. they are extras: exact
  for a renormalising target (norm_topk_prob is free: `moeR_dup_renorm` "whatever the extras"), a limit for a P
  target (`moeR_dup_plain_extras_ne` / `_bound`). Price them in moe_sig as extras (E = n_routed + zero), so the
  FEATURES host side is free.
- kind: scmoe wall (target) / exact (host); zero_expert wall (target) / exact or limit via moe_why (host)
- cost: host depth 2 x num_layers for dense targets (dims layers reads num_layers = 28: conservative)
- free by the config rules: norm_topk_prob / routed_scaling_factor free (routing numbers); zero_expert_num is a
  feature switch (changes E), not free
- theorem: host side `zeroed_sublayer` (PROVED exact); `router_bias_off`; needs theorem: "a sequential pre-norm stack
  cannot add a branch output after the next attention sublayer" (scmoe target wall) and "no SwiGLU expert equals the
  identity on the normed input" (zero_expert target wall; cf. `analytic_ne_clamp`-style argument)

## 4. Test fixtures
- flag: meituan-longcat/LongCat-Flash-Chat, LongCat-Flash-Thinking, LongCat-Flash-Thinking-2601 (live) ->
  longcat_flash; meituan-longcat/LongCat-2.0, LongCat-Flash-Lite-Sparse -> longcat_flash (same arch, no code in
  the snapshot cache, not in carriers)
- not flag: openbmb/MiniCPM4-0.5B (also no model_type, MiniCPM arch), openbmb/MiniCPM-2B-sft-bf16 (no model_type),
  deepseek-ai/DeepSeek-V2-Lite (the MLA + MoE look-alike)

## Verdict
real: the 3 live (5 total) LongCat points get renorm P, router_bias, scmoe and zero_expert: every edge INTO them
from non-LongCat hosts disappears (fewer edges, ~3-5 targets); as hosts they keep their sequential-MoE edges (now
justified by the scmoe host rule) and the renorm "?" -> P / extras pricing decides which MoE targets they play (both
directions, a handful of edges). Region: MoE + classification.
