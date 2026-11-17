# out_gate_by_code_unlisted
live 71 / placed 71 (alice_ai 1, beam 2, dory 4, dots3_note 1, ethoric 1, gpts3_hybrid 1, gptx3 1, grug_moe 7, hrm_text 3, hy_v4 1, lfm3 20 (one via lfm3_omni), lfm3_rd 2, muse_glimmer 1, muse_glimmer_text 2, qwen4_exp 1, qwen4_exp_text 23); direction over-credit; batch a1_typetables

## 1. What the code computes
Every type below builds an attention output gate `o <- o * sigmoid(W_g x [+ b])` before o_proj, with no config switch (Ethoric is the exception: `attn_gate` picks it). p5 (p5:model_graph.py:1977) has `OUT_GATE_TYPES = {qwen3_next, afmoe, laguna, minicpm_sala, step3p5}` and none of these types. They split by **granularity** and **gate bias**:
- head-wise (W_g: hidden -> num_heads, one scalar per head):
  - lfm3: junkdogjoe/LFM3-Lobotomized-2B/modeling_lfm3.py:231 `gate_proj = Linear(hidden, num_attention_heads, bias=False)`, :270-271 `attn_output * sigmoid(gate_proj(x)).unsqueeze(-1)`
  - lfm3_rd: cybertruck32489/LFM3-1.7B-RD/modeling_lfm3_rd.py:445, :469-470 (same)
  - grug_moe: ext grug_moe/.../skyrl_train_models_grug_moe.py:714 `attn_gate = Linear(hidden, num_heads)`, :746 `2.0 * sigmoid(attn_gate(x))` (GRUG_ATTN_GATE_SCALE = 2.0, :57; the 2 folds into o_proj)
  - dots3_note: tfmain:models/dots3_note/modeling_dots3_note.py:501 `g_proj = Linear(hidden, num_heads)`, :604-605
- element-wise (one gate per channel, heads x head_dim):
  - qwen4_exp(_text): tf5:models/qwen4_exp/modeling_qwen4_exp.py:830-831 q_proj is `heads*head_dim*2`, :866 chunk into (query, gate), :897 `attn_output * sigmoid(gate)`; 24 of 25 configs have no attn_output_gate key
  - alice_ai: AMAImedia/AliceAI-.../modeling_alice_ai.py:163-164 (q_proj 2x), :228-235, :271
  - gpts3_hybrid: AxiomicLabs/GPT-S3-3M/modeling_gpts3.py:257, :266, :297; gptx3: AxiomicLabs/GPT-X3-150M/modeling_gptx3.py:75, :85, :104
  - hy_v4: tf5:models/hy_v4/modeling_hy_v4.py:384 `gate_proj: hidden -> num_heads * v_head_dim`, :420, :490
  - muse_glimmer(_text): tf5:models/muse_glimmer/modeling_muse_glimmer.py:321, :368
  - hrm_text: tf5:models/hrm_text/modeling_hrm_text.py:187-191 (bias = config.attention_bias, false/absent in the 3 snapshots), :232
  - dory: ext dory/transformers-pr50099/modeling_dory.py:352, :413 `sigmoid(gate * (s_gate * s_gate_scale)^2)` (a per-channel logit scale = a row scaling of W_g: folds). Its whole-output L2 norm before o_proj (:414 attn_sub_norm) is a separate wall, not this finding.
  - klm: MinimaLabs/KLM-1.5/modeling_klm.py:146, :167 (not in the carrier list, same code fact; p5 gives KLM no out_gate)
  - needle: ext needle/gh__cactus-compute__cactus/architecture.py:257-259 (same; not in the carrier list)
- with a gate bias (a constant inside the sigmoid):
  - beam: tfmain:models/beam/modeling_beam.py:232 elementwise gate_proj, :234 `gate_bias = Parameter(zeros(num_heads))`, :278-279
  - ethoric: techdotus/Ethoric-10M/attention.py:168-173 `headwise -> Linear(d, n_head, bias=True)`, `elementwise -> Linear(d, n_head*hd, bias=True)`, else none; :257-263. The snapshot has `attn_gate: "headwise"`.

## 2. Classifier change
- reads: model_type; Ethoric `attn_gate`; hrm_text `attention_bias`
- new: types into `OUT_GATE_TYPES` (feature out_gate), plus a dims field `out_gate_kind` in {head, elem, head_bias, elem_bias}, in dims_of (default for every out_gate config: "elem", except the head-wise keys `headwise_attn_output_gate` / `use_head_wise_attn_gate` / `gated_attention_proj_granularity_type` -> "head")
- detection:
```python
OUT_GATE_BY_CODE = {"lfm3": "head", "lfm3_rd": "head", "grug_moe": "head", "dots3_note": "head",
                    "dots3_note_text": "head", "qwen4_exp": "elem", "qwen4_exp_text": "elem", "alice_ai": "elem",
                    "gpts3_hybrid": "elem", "gptx3": "elem", "hy_v4": "elem", "muse_glimmer": "elem",
                    "muse_glimmer_text": "elem", "dory": "elem", "klm": "elem", "needle": "elem", "beam": "elem_bias"}

def detect_out_gate_by_code_unlisted(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "ethoric":
        g = str(lm.get("attn_gate") or "").lower()
        return {"headwise": "head_bias", "elementwise": "elem_bias"}.get(g)
    if model_type == "hrm_text":
        return "elem_bias" if lm.get("attention_bias") is True else "elem"
    return OUT_GATE_BY_CODE.get(model_type)
```
- apply note: in p5:model_graph.py replace the `OUT_GATE_TYPES` set (line 1977) by the dict `OUT_GATE_BY_CODE` above merged with the existing five (`qwen3_next/afmoe/laguna/minicpm_sala: "elem"`, `step3p5: "head"` -- check step3p5's head-wise gate when applying); `_feats` keeps `mt in OUT_GATE_TYPES` (dict membership) and adds `or (mt == "ethoric" and str(d.get("attn_gate")).lower() in ("headwise", "elementwise"))`; dims_of gets `out_gate_kind` from the same function.

## 3. Relation rule
- target side: a gated target needs a gated host (FEATURES out_gate stays wall on the target side). Add, in emul_matrix, when both have out_gate: target `head*` <- host `head*` or `elem*` (an elementwise host copies each head's gate row across its head_dim channels); target `elem*` <- host `elem*` only; target `*_bias` <- host `*_bias` only.
- host side: a gated host plays an ungated target (W_g = 0, o_proj doubled); a biased gate sets b = 0; an elem host plays a head target.
- kind: exact (all allowed directions); wall (elem target on a head host; bias target on a bias-free gate; any gated target on an ungated host)
- cost: none
- free by the config rules: nothing; Ethoric's `attn_gate` is a feature switch (shape-changing), not free
- theorem: `out_gate_off` (PROOFS.md "out_gate host -> ungated target"); ungated host for a gated target is OPEN ("out_gate / shared_gate target <- ungated host"); the bias wall is `biased_proj_not_bias_free` (the gate logit is a projection of the pre-norm output); needs theorem: `out_gate_head_sub_elem` (an elementwise gate with rows tied within each head equals the head-wise gate) -- a one-line instance of weight tying

## 4. Test fixtures
- flag: AJ-Gonzalez/coder-lfm3-1.7b -> head; junkdogjoe/LFM3-Lobotomized-2B -> head (lfm3_omni wrapper); cybertruck32489/LFM3-1.7B-RD -> head; PeetPedro/qwen4_exp-8b-... -> elem; Jab1718/qwen3.8-flash-coder-85gb-bf16 -> elem; AMAImedia/AliceAI-... -> elem; techdotus/Ethoric-10M -> head_bias; AxiomicLabs/GPT-S3-3M, AxiomicLabs/GPT-X3-150M -> elem; tencent/Hy4-preview-FP8 -> elem; RadixArk/Muse-Glimmer-q4-MLX, hweller1/wayfinder-glimmer-30b -> elem; sapientinc/HRM-Text-1B, danish-foundation-models/DFM-Mimir -> elem; ProCreations/Dory-Beam-4B -> elem; open-athena/Grug-67B-...-2026.09.21 -> head; dots-studio/dots3-note-prev -> head; reflection/Beam-1.0-501B-A23B -> elem_bias; MinimaLabs/KLM-1.5, Cactus-Compute/needle2 -> elem
- not flag: LiquidAI/LFM2.5-230M-Base (lfm2 parent, no gate); Qwen/Qwen3-0.6B, Qwen/Qwen3-30B-A3B (Qwen3 / Beam parent); tencent/Hy3 (Hy4 parent); rednote-hilab/dots.llm1.base (dots parent); Qwen/Qwen3-Next-80B-A3B-Instruct and arcee-ai/Trinity-Mini (already in OUT_GATE_TYPES); jkminder/d16_optimized_537m_seed5 (nanochat ve_gate is a value gate, not an output gate); Lilgarca/my-model-v01 (cyrah: already out_gate via use_output_gate)
- scan: 79 repos -- the 71 live + 4 unplaced lfm3 copies (ONNX / GGUF / CoreML / phonon-refiner, same code) + whittleOrg/Whittle-Next-27B-A3B (qwen4_exp_text with attn_output_gate true: already out_gate, same code fact) + klm 1 + needle 2 (not in the carrier list; p5 also gives them no out_gate).

## Verdict
real. Fewer edges: ~74 gated targets lose every ungated host (most are masked today by hyper_conn / ple / attn_res / post_norm on qwen4_exp, alice_ai, muse; the unmasked losers are lfm3 / lfm3_rd (22), hrm_text (3), grug_moe (7), dory (4), gptx3 / gpts3 / ethoric / klm / needle). As hosts they gain nothing new beyond ungated targets they already played. The granularity / bias split removes elem-target <- head-host edges (e.g. qwen3_next <- lfm3) that the bare feature would otherwise create.
