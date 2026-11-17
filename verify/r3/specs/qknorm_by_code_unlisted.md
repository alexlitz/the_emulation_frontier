# qknorm_by_code_unlisted
live 65 / placed 65 (lfm3 20, grug_moe 7, dory 4, plamo3 3, lfm3_rd 2, beam 2, muse_glimmer_text 2, needle 2, and 1 each: alice_ai, bananamind2_moe, bitsy_parallel, cyrah, domynedge, edgerock, ethoric, gemma3_moe, gpts3_hybrid, gptx3, hulumed_qwen2, hulumed_qwen3, iquest_q1, isaac, klm, kolibri1, llada2_moe, memt_qwen3_5_moe, monkeyocrv2, muse_glimmer, nexus, qwen3loop, viuai); direction both; batch a1_typetables

## 1. What the code computes
p5 classify (p5:model_graph.py:2332-2342) sets qknorm only from keys or `QKNORM_TYPES` (= QKNORM_ALWAYS, :1852) / `QKNORM_BY_CODE` (:1875, gaggle only); every type here builds its q / k norms with no key, so p5 gives qknorm=no.
- **head** (per-head RMSNorm with one head_dim gain, before RoPE; a (1 + w) gain is a reparametrisation):
  - lfm3 junkdogjoe/LFM3-Lobotomized-2B/modeling_lfm3.py:229-230, :245-246; lfm3_rd cybertruck32489/LFM3-1.7B-RD/modeling_lfm3_rd.py:443-444, :452-453
  - plamo3 pfnet/plamo-3-0.8b-fin-instruct/modeling_plamo.py:483-484 (`Plamo3RMSNorm(qk_dim, offset=1.0)`, offsets :14-15), :512-513
  - llada2_moe preview inclusionAI/LLaDA2.0-mini-preview/modeling_llada2_moe.py:396-397 unconditional (its config default use_qk_norm=False, configuration_llada2_moe.py:19); the other LLaDA2 repos gate on `use_qk_norm` (modeling :448 / :463) with class default True (configuration :19) and 3 of them omit the key (LLaDA2.0-mini, 2.1-mini, 2.1-flash: p5 gives them no too)
  - alice_ai modeling_alice_ai.py:175-180 (zero_centered), :242-243; gpts3_hybrid modeling_gpts3.py:261-262, :269-271; gptx3 modeling_gptx3.py:79-80, :86-87
  - edgerock rockmatrix-ai/EdgeRock/modeling_edgerock.py:344, :899 (transformers Qwen3_5Attention, tf5:models/qwen3_5/modeling_qwen3_5.py:772-773)
  - cyrah Lilgarca/my-model-v01/modeling_cyrah.py:386-387 (gqa) / :405-406 (mla), :436-437 / :467-468
  - nexus ifx-pse-sys-ml/flame-27m-grounded/modeling_nexus.py:106-107, :124; bananamind2_moe modeling_bananamind2moe.py:61-62, :77-78; bitsy_parallel zarko1321/Bitsy500M/modeling_bitsy.py:47-48, :53-54; klm MinimaLabs/KLM-1.5/modeling_klm.py:148-149, :161-162; memt_qwen3_5_moe MemTensor/MemT-1.0-Nano/modeling_memt.py:212 (Qwen3_5MoeAttention), :237-238
  - Qwen3 bodies (tf5:models/qwen3/modeling_qwen3.py:237-238): isaac PerceptronAI/Isaac-0.1/modular_isaac.py:1572 (AutoModel of a Qwen3 text_config), hulumed_qwen3 ZJU-AI4H/Hulu-Med-4B/modeling_hulumed_qwen3.py:163 (HulumedMetaModel + Qwen3Model), monkeyocrv2 zenosai/MonkeyOCRv2-B-Parsing/modeling_monkeyocrv2.py:19 (Qwen3ForCausalLM subclass), qwen3loop ext qwen3loop/.../modeling_qwen3loop.py:78 (Qwen3Model, stock Qwen3DecoderLayer)
  - gemma3_moe tardellirs/gemma3moe-16x4-270m-starter/modeling_gemma3_moe.py:72-75 (Gemma3DecoderLayer with the MLP swapped; tf5:models/gemma3/modeling_gemma3.py:338-339)
  - kolibri1 tfmain:models/kolibri1/modular_kolibri1.py:104 (Exaone4Attention; tf5:models/exaone4/modeling_exaone4.py:217-218); beam tfmain:models/beam/modeling_beam.py:228-229, :249-250
  - iquest_q1 ext iquest_q1/gh__IQuestLab__vllm-iquest-q1/model.py:270, :350-352, :397-409; viuai ext viuai/hub__ViuAI__ViuAI-500M/code/model.py:57-58, :67; needle ext needle/gh__cactus-compute__cactus/architecture.py:209-210 (ZCRMSNorm, before RoPE :212-215)
  - ethoric techdotus/Ethoric-10M/attention.py:162-166: only when `str(config.qk_norm) == "per_head"` (the snapshot: "per_head"; p5 tests `qk_norm is True`)
- **head_qfree** (new): domynedge domyn/Domyn-Edge-Base/modeling_domynedge.py:223 q_norm `RMSNorm(elementwise_affine=False)`, :224 k_norm weighted, :238-239 before RoPE
- **head_free_gain** (new, shared with qknorm_fixed_gain): muse_glimmer(_text) tf5:models/muse_glimmer/modeling_muse_glimmer.py:322 weightless per-head RMSNorm, :340-341 `q = rms(q) * qk_scale_factor` (3.87 default, configuration_muse_glimmer.py:144); grug_moe skyrl_train_models_grug_moe.py:728-733 weight-free q / k norm, `q *= qk_mult` (1.75 in the snapshots)
- **ngpt** (new): dory modeling_dory.py:350-351 weight-free L2 norms (:50-58), :383-384 `q *= (sqk * sqk_scale)^2` per channel AFTER RoPE
- hulumed_qwen2 (in the carrier list) is a Qwen2 body: no q / k norm -- not flagged (its finding is bias_qkv).

## 2. Classifier change
- reads: model_type; Ethoric `qk_norm` string; LLaDA2 `use_qk_norm` with class default True
- new: qknorm slot values: "head" for the types above; new QKNORM_VARIANTS "head_qfree", "head_free_gain", "ngpt"; table entries in QKNORM_ALWAYS / QKNORM_HEAD_CHECKED / QKNORM_VARIANT_BY_TYPE
- detection:
```python
QKNORM_HEAD_BY_CODE = {"alice_ai", "bananamind2_moe", "beam", "bitsy_parallel", "cyrah", "edgerock", "gemma3_moe",
    "gpts3_hybrid", "gptx3", "hulumed_qwen3", "iquest_q1", "isaac", "klm", "kolibri1", "lfm3", "lfm3_rd",
    "memt_qwen3_5_moe", "monkeyocrv2", "needle", "nexus", "plamo3", "qwen3loop", "viuai"}
QKNORM_OTHER_BY_CODE = {"domynedge": "head_qfree", "muse_glimmer": "head_free_gain",
    "muse_glimmer_text": "head_free_gain", "grug_moe": "head_free_gain", "dory": "ngpt"}

def detect_qknorm_by_code_unlisted(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in QKNORM_HEAD_BY_CODE:
        return "head"
    if model_type in QKNORM_OTHER_BY_CODE:
        return QKNORM_OTHER_BY_CODE[model_type]
    if model_type == "ethoric":
        return "head" if str(lm.get("qk_norm") or "").lower() == "per_head" else None
    if model_type == "llada2_moe":
        return "head" if lm.get("use_qk_norm", True) is not False else None
    return None
```
- apply note: in p5:model_graph.py add QKNORM_HEAD_BY_CODE to `QKNORM_ALWAYS` and `QKNORM_HEAD_CHECKED` (:1852, :1893); the four others to `QKNORM_ALWAYS` and `QKNORM_VARIANT_BY_TYPE` (:1879); add "head_qfree", "head_free_gain", "ngpt" to `QKNORM_VARIANTS` (:224). In classify (:2332) accept `str(d.get("qk_norm")).lower() == "per_head"` as yes; for llada2_moe add `_CLASS_DEFAULT_FALLBACK["use_qk_norm"]["llada2_moe"] = True` and read `_cfg_value(d, "use_qk_norm")` (llada2_moe is already in QKNORM_HEAD_CHECKED; the preview's unconditional norm is covered by the same default since its config omits the key).

## 3. Relation rule
- target side: a qk-normed target needs a host of the same variant or one in QKNORM_NEST; plain hosts only by the regime fold (not payable).
- host side: these hosts now carry qknorm: a "head" host plays plain targets only by the eps lever (limit, EPS_LEVER_QKNORM); the weight-free ones (head_free_gain, ngpt, head_qfree's q) cannot be levered off.
- QKNORM_NEST additions (host variants that contain the target's by a weight / config setting):
  `head_free: + ("head_qfree", "head_free_gain", "ngpt")`; `head_qfree: ("head", "head_distinct")` (q gain 1); `head_free_gain: ("head", "head_post", "head_distinct", "head_free_gain", "ngpt")` (gain set to the constant). "ngpt" is nested only in itself.
- kind: exact (nestings); limit (eps lever host -> plain); wall otherwise
- cost: none
- free by the config rules: the gain constant (a scale) is free; adding a qk-norm never is
- theorem: `qkn_gain_one`, `rmsNorm_one_rot_comm`, `weightFree_comm` (PROOFS.md "qk-norm: head_free <- head / head_post / head_distinct"), `qkn_equal_gains`; `epsLever_bound` / `leverAttn_model_limit` (eps lever); needs theorem: `qkn_const_gain` (a weight-free per-head norm times a constant c equals the gained norm with w = c 1, and commutes with RoPE) for the head_free_gain / head_qfree nestings

## 4. Test fixtures
- flag: one per type (29 repos, fixtures/a1_typetables.json): e.g. pfnet/plamo-3-0.8b-fin-instruct -> head; inclusionAI/LLaDA2.0-mini-preview -> head; inclusionAI/LLaDA2.0-mini -> head (default True); LiquidAI/LFM3-1.7B-Base -> head; techdotus/Ethoric-10M -> head; domyn/Domyn-Edge-Base -> head_qfree; unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit -> head_free_gain; open-athena/Grug-67B-...-2026.09.21 -> head_free_gain; nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8 -> ngpt
- not flag: ZJU-AI4H/Hulu-Med-7B (Qwen2 body); Qwen/Qwen3-0.6B, google/gemma-3-1b-it, LGAI-EXAONE/EXAONE-4.0-32B, LiquidAI/LFM2.5-230M-Base, pfnet/plamo-2-1b, arcee-ai/Trinity-Mini (parents already listed); SmallDoge/Doge-20M (doge already listed)
- scan: 73 repos = 64 live (hulumed_qwen2 excluded) + 4 unplaced lfm3 copies + 5 more llada2_moe (3 key-absent over-walls, 2 with use_qk_norm true already head) + nothing unrelated.

## Verdict
real, both directions. Fewer edges for ~65 targets (plain hosts no longer play them; they now need a qk-norm host) and new exact edges between them and the existing head hosts (Qwen3 / Gemma-3 / LFM2 ...: e.g. hulumed_qwen3 / isaac / monkeyocrv2 / qwen3loop <- qwen3 and back). As hosts, ~45 "head" models gain plain targets only by the eps lever (limit). Net: a few hundred edges move; the largest block is lfm3 (20).
