# out_gate_headwise
live 17 / placed 17 (spark2_5 9, laguna 6, ming_v3_asr 1, youtu_vl_moe 1); direction over-credit; batch a7_attn_exotic

## 1. What the code computes
- tencent/Youtu-LLM-A7B-Base/modeling_youtu_vl_moe.py:884-891: `mla_output_gate_type` (default "head_wise"); with
  `attention_output_gate` builds `g_proj: hidden -> num_heads` (head_wise) or `hidden -> num_heads*v_head_dim` (channel_wise).
  :985-990: `o = o * sigmoid(g_proj(x)).unsqueeze(-1)` (head_wise), before o_proj.
- tf5:models/laguna/modular_laguna.py:289-291: `gate_per_head = gating is True or gating == "per-head"`; g_proj out =
  num_heads (per-head) else num_heads*head_dim. :335-339 `o * softplus(g_proj(x)).unsqueeze(-1)` per head.
  poolside/Laguna-XS-2.1/modeling_laguna.py:392-397 (`gate_per_head = gating == "per-head"`; its snapshot says
  "per-head"); poolside/Laguna-XS.2/modeling_laguna.py:368 always `Linear(hidden, num_heads)` (config gating True).
  poolside/Laguna-M.1 config gating "per-element": ELEMENTWISE (placed by type only; not head-wise).
- inclusionAI/Ming-3.0-asr-tiny/modeling_ming_v3_asr.py:763-770: `gated_attention_proj_granularity_type` None / head_wise
  (`Linear(hidden, num_heads)`) / element_wise; :889-893 `o * sigmoid(g)[..., None]`. Same code in
  Edge0/Edge0-8B-A1B-preview/modeling_bailing_moe_v3.py:1334-1338, 1419-1425 (Ling-3 bailing_hybrid sets the same key;
  its code is not local: taken as the same BailingMoeV3 attention).
- JuneoAI/Laplace-1-Vector/modeling_spark.py:154 `g_proj = Linear(hidden, num_heads)` iff `headwise_attn_output_gate`;
  :199-206 `o * sigmoid(g)` (gate_attn_act_mode sigmoid; silu allowed, none in the zoo), per head.
- Not in the carrier list but the same code fact: stepfun-ai/Step-3.5-Flash/modeling_step3p5.py:458-462, 527-531
  (`use_head_wise_attn_gate`: `Linear(hidden, num_heads)`, `o * g.unsqueeze(-1).sigmoid()`); tf5 step3p7
  modeling_step3p7.py:708-710, 751 (no snapshot); techdotus/Ethoric-10M/attention.py:168-171, 257-263 (`attn_gate`
  headwise, gate_proj WITH bias); athenista/slm-tavenel-10m/hf_model.py:63-66, 97-100 (`attn_gate` headwise).
- A head-wise gate is the elementwise gate o_hd * sigmoid(w_{h,d} . x) with w_{h,d} = w_h for every d: an elementwise
  gate with tied rows. The converse is false: per-channel data-dependent gates have rank > num_heads.

## 2. Classifier change
- reads: model_type; gating (laguna / step3p7), attention_output_gate + mla_output_gate_type (youtu_vl_moe),
  gated_attention_proj_granularity_type, headwise_attn_output_gate (spark2_5 / motif), use_head_wise_attn_gate
  (step3p5), attn_gate (ethoric / slm_tavenel)
- new: FEATURES key `out_gate_head` (replaces `out_gate` in the feature string when head-wise), in _feats + FEATURES;
  a nesting table FEAT_NEST = {"out_gate_head": ("out_gate",)} read by feats_ok. Composes with out_gate_activation:
  the key is `out_gate` + (`_head` if head-wise) + (`_softplus` / `_silu` if not sigmoid), e.g. Laguna-S-2.1
  `out_gate_head_softplus`, Laguna-M.1 `out_gate_softplus`; nesting only drops `_head`, never changes the activation.
- detection:
```python
def detect_out_gate_headwise(cfg, model_type):
    lm = lm_of(cfg)
    mt = model_type
    g = lm.get("gating", True)
    if mt in ("laguna", "step3p7"):
        return "head" if (g is True or _s(g) == "per-head") else None
    if mt == "youtu_vl_moe":
        if truthy_flag(lm.get("attention_output_gate")) and _s(lm.get("mla_output_gate_type", "head_wise")) == "head_wise":
            return "head"
        return None
    if _s(lm.get("gated_attention_proj_granularity_type")) == "head_wise":
        return "head"
    if mt in ("spark2_5", "motif") and lm.get("headwise_attn_output_gate") is True:
        return "head"
    if mt == "step3p5" and lm.get("use_head_wise_attn_gate") is True:
        return "head"
    if mt in ("ethoric", "slm_tavenel") and _s(lm.get("attn_gate", "headwise")) == "headwise":
        return "head"
    return None
```
- apply note: p5:model_graph.py `_feats` (the `if (d.get("attn_output_gate") ...): f.add("out_gate")` block, ~1747-1755):
  compute the key as above instead of the bare "out_gate"; also add `attn_gate in ("headwise", "elementwise")`
  (ethoric / slm_tavenel, today no out_gate at all) and `elementwise_attn_output_gate` (Motif-3) to the trigger keys;
  drop step3p5 / laguna from relying on OUT_GATE_TYPES alone (laguna gating False = no gate). FEATURES: add
  `out_gate_head` ("wall", "free"). feats_ok: a target feature x is carried if x in H or any FEAT_NEST[x] in H.

## 3. Relation rule
- target side: a head-wise target needs a gated host of the same activation, head-wise or elementwise (tie the
  elementwise rows within each head: exact); an ungated host is a wall (as out_gate).
- host side: a head-wise host plays an ungated target (W_g = 0: sigmoid 1/2 or softplus ln 2, rescale o_proj) and a
  head-wise target; it cannot play an elementwise target (wall: one scalar per head cannot form d_h distinct gates).
- kind: exact (head target <- elem host, ungated target <- head host); wall (elem target <- head host)
- cost: none
- free by the config rules: none; granularity is a shape-changing switch (g_proj width), not a free value
- theorem: `out_gate_off` (row "out_gate host -> ungated target") | needs theorem: out_gate_head_sub_elem
  (elementwise gate with W_g rows tied per head = the head-wise gate) and out_gate_elem_not_head (a head-wise gate
  cannot equal a generic elementwise one: rank argument)

## 4. Test fixtures
- flag: tencent/Youtu-LLM-A7B-Base, poolside/Laguna-XS.2 (gating True), poolside/Laguna-XS-2.1, poolside/Laguna-S-2.1,
  inclusionAI/Ming-3.0-asr-tiny, JuneoAI/Laplace-1-Vector, XHToken/Spark-X2.6-4B-Base, stepfun-ai/Step-3.5-Flash,
  inclusionAI/Ling-3.1-flash, Edge0/Edge0-8B-A1B-preview, techdotus/Ethoric-10M, athenista/slm-tavenel-10m -> "head"
- not flag: poolside/Laguna-M.1 (per-element), Qwen/Qwen3-Next-80B-A3B-Instruct, arcee-ai/Trinity-Mini (elementwise
  sigmoid), Motif-Technologies/Motif-3 (headwise false, elementwise true), inception42/Jais-3-MoVA-36B-A4B
  (elementwise softplus), IFM/K2-Horizon-7B (no gate)

## Verdict
real. Fewer edges: the ~31 head-wise models (Spark 9, Laguna 5, Ling-3 5, tavenel 5, Step-3.5 2, Youtu, Ming, Edge0,
Ethoric) stop hosting elementwise-gated targets (~310 Qwen3.5 / 3.6 / Next / AFMoE / MiniCPM-SALA / Kimi MLA ...);
ethoric / slm_tavenel gain the out_gate feature they lack today (fewer edges for them as targets on ungated hosts).
