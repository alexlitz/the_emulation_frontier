# group_mask_zero_fill
live 5 / placed 15 (deepseek_v3 9, kimi_k2 4, axk2 2); direction both; batch a9_router

## 1. What the code computes
- Zero fill of the excluded groups' BIASED scores, then top-k: tf5:models/axk2/modular_axk2.py:236-237 ("fill with 0.0
  intentionally to match reference implementation"), :242-245 sigmoid + e_score_correction_bias;
  moonshotai/Moonlight-16B-A3B/modeling_deepseek.py:439-462 (`tmp_scores = scores_for_choice.masked_fill(~score_mask,
  0.0)`, :459); microsoft/MAI-DS-R1/modeling_deepseek.py:458; meituan/DeepSeek-R1-Block-INT8 and
  nvidia/DeepSeek-R1-NVFP4-v2 modeling_deepseek.py:458; moonshotai/Kimi-K2-Instruct/modeling_deepseek.py:458;
  nvidia/Kimi-K2.5-NVFP4/modeling_deepseek.py:472-473; jdopensource/JoyAI-LLM-Flash/modeling_deepseek.py:147.
- -inf fill (a hard group mask): deepseek-ai/DeepSeek-V3, DeepSeek-R1, R1-0528, V3.1 modeling_deepseek.py:457;
  ByteDance-Seed/academic-ds-9B/modeling_deepseek.py:457; tf5:models/deepseek_v3/modeling_deepseek_v3.py:162; Beam
  modeling_beam.py:369; Bailing V2 / V2.5 / V3 / LLaDA2 / Ming gates (e.g. Ling-1T modeling_bailing_moe_v2.py:328).
  transformers 4.57 (installed) modeling_deepseek_v3.py:139 is 0.0: the native port changed between 4.57 and 5.18;
  5.18 is taken as the native reference (the 4 native deepseek_v3 snapshots have n_group 1 anyway).
- Effect: an excluded expert scores 0; a kept expert with sigma(l) + b < 0 (b < 0) ranks below it, so when fewer than
  top_k kept experts have a positive biased score, experts from dropped groups are selected. Softmax / bias-free
  zero-fill routers (DeepSeek-V2 group_limited_greedy, Mistral-Large-4 modular_mistral_large4.py:546) never hit it
  (scores > 0): not flagged.
- Per-repo code, not per type: the first-release modeling_deepseek.py (0.0) still reads config.seq_aux
  (MAI-DS-R1 modeling_deepseek.py:401; Kimi-K2 :401; K2.5 :421); the -inf revision does not (DeepSeek-R1 file: no
  seq_aux). Config proxy: remote modeling_deepseek + "seq_aux" key + noaux_tc. Known mismatch: ByteDance
  academic-ds-9B reads seq_aux (:400) but fills -inf (flagged by the proxy: the one false positive). unsloth
  DeepSeek-R1 / V3-0324 GGUF-UD configs are first-release copies (code not cached): flagged, unverified.
- The bias sign is a checkpoint buffer: the config cannot tell whether b < 0 occurs, so every 0-fill path is the risk.

## 2. Classifier change
- reads: model_type, auto_map, seq_aux presence, topk_method
- new: group_cap value "top2sum_zero" (GROUP_SCORE_TYPES axk2 / kimi_k2 / joyai_llm_flash; per config for deepseek_v3)
  in _moe_sig; a GROUP_PLAYS table in moe_why
- detection:
```python
def detect_group_mask_zero_fill(cfg, model_type):
    lm = lm_of(cfg); mt = model_type
    if not _n_experts(lm):
        return None
    if mt == "axk2":
        return "top2sum_zero"
    am = auto_map(cfg)
    remote_ds = "modeling_deepseek" in am or "modeling_kimi_k25" in am
    if mt == "joyai_llm_flash" and remote_ds:
        return "top2sum_zero"
    if (mt in ("deepseek_v3", "kimi_k2") and remote_ds and "seq_aux" in lm
            and _s(lm.get("topk_method")) == "noaux_tc"):
        return "top2sum_zero"
    return None
```
- apply note: p5:model_graph.py GROUP_SCORE_TYPES (~879): axk2, kimi_k2, joyai_llm_flash -> "top2sum_zero"; `_moe_sig`
  (~942): `gcap = "top2sum_zero"` for deepseek_v3 when the proxy holds (it receives the LM dict d; auto_map / seq_aux
  are on it for flat configs; the kimi_k25 wrapper's auto_map is on the outer config: pass it in or read
  d.get("auto_map") of the wrapper in dims_of). moe_why (~1008): replace `T["group_cap"] != H["group_cap"]` by
  `H["group_cap"] not in GROUP_PLAYS.get(T["group_cap"], {T["group_cap"]})` with GROUP_PLAYS = {"top2sum": {"top2sum",
  "top2sum_zero"}} (a zero-fill host for a -inf target needs H["bias"]), and the m != 1 rule on
  `T["group_cap"].startswith("top2sum")`.

## 3. Relation rule
- target side: a zero-fill grouped target needs a zero-fill host of the same group shape with m = 1 (same code: the
  copy construction with the same fill); a -inf host is a wall (equal only while >= k kept experts score > 0: a
  regime the config cannot certify)
- host side: a zero-fill host plays a -inf (top2sum) target exactly: add one constant c > -min_e b_e to every
  selection bias; group top-2 sums shift by 2c and in-group order is unchanged, every kept expert then scores > 0 =
  the filled 0, so 0-fill selects as -inf does. Host-side relevant even at n_group 1 (n_group / topk_group are free)
- kind: exact (zero <- zero, m = 1; top2sum <- zero via the shift); wall (zero <- top2sum)
- cost: none
- free by the config rules: n_group / topk_group free; the fill is code; the bias shift is a weight change
- theorem: `moeR_group_copies`, `moeR_group_top2sum_dup_flips` (m = 1). Recommendation on the round3_inputs G item:
  change the Lean definition (a fill parameter in {-inf, 0}) rather than add "host bias >= 0" to moeR_group_copies;
  the condition that matters is the TARGET's bias sign under a 0-fill target, and for -inf targets the host's sign is
  free by the shift. Needs theorem: group_zero_fill_shift (0-fill with all biased scores > 0 = -inf fill; a uniform
  bias shift preserves the group and expert selections) and moeR_group_copies for fill 0 at m = 1.

## 4. Test fixtures
- flag: skt/A.X-K2-NVFP4, Baekpica/A.X-K2-VQ-GGUF, microsoft/MAI-DS-R1, meituan/DeepSeek-R1-Block-INT8,
  nvidia/DeepSeek-R1-NVFP4-v2, moonshotai/Kimi-K2-Instruct, Moonlight-16B-A3B, nvidia/Kimi-K2.5-NVFP4,
  mlx-community/Kimi-K2.6-mlx-DQ3_K_M-q8, jdopensource/JoyAI-LLM-Flash -> "top2sum_zero"
- not flag: deepseek-ai/DeepSeek-V3, DeepSeek-R1, nvidia/DeepSeek-R1-0528-NVFP4-v2 (-inf revision), DeepSeek-V3.2
  (tf5), skt/A.X-K1, DeepSeek-V2-Chat (softmax max-group, 0.0 harmless), Ling-1T, Beam-1.0, Mistral-Large-4

## Verdict
real. Both: the 5 live grouped zero-fill targets (A.X-K2 2, MAI-DS-R1, R1-Block-INT8, R1-NVFP4-v2) lose the -inf
DeepSeek-V3 / R1 / Ling / GLM hosts credited today (fewer edges, ~5 targets x ~20 hosts); Kimi-K2 / Moonlight / JoyAI
/ A.X-K2 hosts keep the -inf targets via the bias shift and gain the zero-fill ones (more edges for ~12 hosts).
