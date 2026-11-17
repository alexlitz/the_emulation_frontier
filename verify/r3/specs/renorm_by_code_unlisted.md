# renorm_by_code_unlisted
live 48 / placed 48 (ceno 12, bailing_hybrid 9, bailing_moe 7, llada2_moe 5, mistral_large4_text 2, <none> (Edge0) 1, Motif 1, alice_ai 1, arctic 1, bananamind2_moe 1, gemma3_moe 1, iquest_q1 1, llama_moe 1, meditor 1, ming_v3_asr 1, param2moe 1, step3_text 1, zaqeris_axiom 1); direction over-wall (and over-credit for key=True hosts); batch a9_router

## 1. What the code computes
- Bailing V2 family, renorm fixed by `top_k > 1`, norm_topk_prob never read (no `norm_topk_prob` in any of these files):
  inclusionAI/Ling-1T/modeling_bailing_moe_v2.py:345, inclusionAI/Ling-2.6-1T/modeling_bailing_moe_v2_5.py:362,
  inclusionAI/Ling-3.1-flash/modeling_bailing_moe_v3.py:409, Edge0/Edge0-8B-A1B-preview/modeling_bailing_moe_v3.py:472,
  bharatgenai/Param2-17B-A2.4B-Thinking/modeling_param2moe.py:337, yifanyu/LLaDA2.1-mini-ORL/modeling_llada2_moe.py:363-367,
  inclusionAI/LLaDA2.2-mini/modeling_llada2_moe.py:291-293, inclusionAI/Ming-3.0-asr-tiny/modeling_ming_v3_asr.py:487-488:
  `topk_weight = scores / (scores.sum(-1) + 1e-20) if top_k > 1 else scores` (sigmoid scores). So R at k > 1, P at k = 1.
- bailing_moe v1 DOES read it: inclusionAI/Ling-lite/modeling_bailing_moe.py:430, 451-453 (`if top_k > 1 and
  norm_topk_prob`). Hence by ARCHITECTURE (BailingMoeForCausalLM v1 vs V2 / V2_5 / V3), not model_type.
- tfmain bailing_hybrid (transformers main, modular_bailing_hybrid.py:116, 137, 397) is a DeepseekV3 router that reads
  norm_topk_prob; every snapshot bailing_hybrid repo ships auto_map remote code (above), which is taken as the code.
- CENO: CladeTeam/CENO-1.3B-A450M-1m/modeling_ceno.py:1184-1192: pre-softmax `p / (sum p + 1e-20)` or post-top-k
  softmax over the k: R either way; norm_topk_prob not read.
- AliceAI: AMAImedia/AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF/modeling_alice_ai.py:552-554 `weights / weights.sum()`.
- ZAQERIS: ZAQERIS/ZAQERIS_AXIOM/modeling_zaqeris_axiom.py:351-352 softmax, top-k, `/ sum` always.
- Arctic: Snowflake/snowflake-arctic-instruct/modeling_arctic.py:973-976 softmax, top-k, `/= sum` iff top_k > 1.
- Gemma3-MoE: tardellirs/gemma3moe-16x4-270m-starter/modeling_gemma3_moe.py:58-60 softmax, top-k, `/ sum`.
- IQuest-Q1: ext:iquest_q1/gh__IQuestLab__vllm-iquest-q1/model.py:125, 157 softmax top-k FusedMoE `renormalize=True`.
- Mistral-Large-4: tfmain:models/mistral_large4/modular_mistral_large4.py:528 (`del self.norm_topk_prob`), 549-552
  (`topk_weights /= sum + 1e-20` always, then x routed_scaling_factor).
- LLaMA-MoE: llama-moe/LLaMA-MoE-v1-3_0B-2_16/modeling_llama_moe_hf.py:494-497 top-k of logits, softmax over the k
  (= R) when `gate_use_softmax` (default True, :786; snapshot True).
- Meditor: Yo-Purple/Meditor-1-2.2B-A500M-MoE/modeling_meditor.py:231-234 softmax, top-k, `p / (p.detach() + 1e-6)`:
  forward value p/(p+1e-6) ~ 1 (rel. error <= 1e-6 / p_top1 <= 2.2e-5 at 22 experts): R at k = 1 (snapshot), a
  constant-1 weight ('unit', neither R nor P) at k > 1.
- Motif: Motif-Technologies/Motif-3/modeling_motif.py:806, 867-870: renormalises iff `route_norm` (a key p5 does not
  list in RENORM_KEYS), then x route_scale.
- P fixed in code: BananaMind/BananaMind-2-MoE/modeling_bananamind2moe.py:185-198 (top-1, gate = max softmax prob);
  stepfun-ai/step3/modeling_step3.py:365-368 (softmax, top-k, no renorm; norm_expert_weight not read, grep empty);
  allenai/MolmoE-1B-0924/modeling_molmoe.py:829-833 (softmax, top-k, no renorm).

## 2. Classifier change
- reads: architectures, model_type, top-k, route_norm, gate_use_softmax
- new: table entries RENORM_FIXED (+ a new RENORM_BY_ARCH), RENORM_KEYS += "route_norm", in _moe_sig
- detection:
```python
def detect_renorm_by_code_unlisted(cfg, model_type):
    lm = lm_of(cfg); mt = model_type
    if not _n_experts(lm):
        return None
    k = _top_k(lm)
    if any(a in _BAILING_V2_ARCHS for a in archs(cfg)):
        return "R" if k > 1 else "P"
    if mt in _RENORM_R_TYPES:          # ceno alice_ai zaqeris_axiom gemma3_moe iquest_q1 mistral_large4(_text)
        return "R"
    if mt in _RENORM_P_TYPES:          # bananamind2_moe step3_text step3_vl molmo
        return "P"
    if mt == "arctic":
        return "R" if k > 1 else "P"
    if mt == "meditor":
        return "R" if k <= 1 else "unit"
    if mt == "llama_moe":
        return "R" if lm.get("gate_use_softmax", True) else "raw"
    if mt == "motif" and lm.get("route_norm") is not None:
        return "key:R" if lm.get("route_norm") else "key:P"
    return None
```
- apply note: p5:model_graph.py `_moe_sig` (~908): before `if mt in RENORM_FIXED`, check
  `RENORM_BY_ARCH = {"BailingMoeV2ForCausalLM", "BailingMoeV2_5ForCausalLM", "BailingMoeV3ForCausalLM",
  "LLaDA2MoeModelLM", "Param2MoEForCausalLM", "MingV3ASRForConditionalGeneration"}` against d["architectures"]
  (Edge0 has no model_type); renorm = "R" if k > 1 else "P", renorm_cap = (renorm,). RENORM_FIXED (~855) add R: ceno,
  alice_ai, zaqeris_axiom, gemma3_moe, iquest_q1, mistral_large4, mistral_large4_text, llama_moe, meditor, arctic
  (arctic / meditor: k-dependent like the arch rule: arctic k = 1 -> P, meditor k > 1 -> "unit"); P: bananamind2_moe,
  step3_text, step3_vl, molmo. RENORM_KEYS (~872) += "route_norm".

## 3. Relation rule
- target side: unchanged moe_why rule (renorm must be in the host's renorm_cap); the change only corrects values:
  '?type' (family-only) becomes R / P, so these targets join the R / P pools.
- host side: a code-fixed router gets renorm_cap (R,) or (P,): hosts that read norm_topk_prob=True from the config
  today get (P, R) and lose the P targets (Bailing V2 / V3, LLaDA2, Param2, CENO, Edge0); Step-3 loses R targets.
- kind: exact (R <- R, P <- P with the existing extras rules); wall (R vs P across)
- cost: none
- free by the config rules: norm_topk_prob is free only where the code reads it (Motif's route_norm: free); fixed in
  code here, so not free
- theorem: `moeR_dup_renorm`, `moeR_dup_plain` (exact rows); `moeR_renorm_host_needs_const_mass`,
  `moeR_plain_host_needs_const_mass` (R vs P wall), MoE <- MoE routing table. Meditor's p/(p+1e-6) is a 1e-6 limit
  (well inside fp32 agreement; no measurement needed beyond the standard).

## 4. Test fixtures
- flag: Ling-1T, Ring-2.5-1T (key absent), Ling-3.1-flash, Edge0 (no model_type), LLaDA2.1-mini-ORL, Param2, Ming-3.0,
  CENO-1.3B-1m, AliceAI, LLaMA-MoE, ZAQERIS, Arctic, Meditor, gemma3moe, IQuest-Q1, Mistral-Large-4 4bit -> "R";
  Motif-3 -> "key:R"; BananaMind-2-MoE, stepfun-ai/step3, MolmoE-1B -> "P"
- not flag: inclusionAI/Ling-lite, Ling-plus (bailing_moe v1 reads the key), allenai/Molmo-7B-D-0924 (dense molmo),
  Step-3.5-Flash (step3p5), Qwen3-30B-A3B, DeepSeek-V3, Mistral-Small-4 (mistral4 reads the key)
- scan: also flags Baekpica/Ling-3.1-flash-1019-VQ-GGUF and inclusionAI/LLaDA2.2-mini (same code; not in the carrier
  list) and MolmoE-1B (carrier, not placed in the relation: harmless entry).

## Verdict
real. Both directions: ~20 '?type' targets (AliceAI, Arctic, BananaMind, gemma3moe, IQuest, LLaMA-MoE, Meditor,
Mistral-L4 x2, ZAQERIS, Motif, Ring/Ling-2.5/2.6 key-absent x4, ...) gain every host of their renorm kind (more
edges); ~30 key=True hosts (CENO 12, Bailing V2/V3 ~16, LLaDA2 5, Param2, Edge0) stop hosting P targets, Step-3
stops hosting R targets (fewer edges).
