# out_gate_key_unread
live 1 / placed 1 (ernie5_5_moe 1); direction over-credit; batch a1_typetables

## 1. What the code computes
Source: ext ernie5_5_moe/gh__baidu__vLLM-ERNIE/attention.py (the vLLM port; the Hub config is $NOCODE_CACHE/ernie5_5_moe/hub__baidu__ERNIE-5.5-Flash-Base/config.json).
- DSA (MLA) layers, Ernie5_5DSAAttention (:763): :823-830 `o_gate = ColumnParallelLinear(q_lora_rank, num_heads * v_head_dim)`; :887 `gate = o_gate(q_c)` with q_c = q_a_layernorm(q_a(x)) (the normed q latent); :904 `wo_b(o.flatten(1) * sigmoid(gate))` -- an elementwise sigmoid output gate whose input is the normed q latent, not x.
- HCA (compressed) layers, Ernie5_5HCAAttentionMixin (:168): :187-194 `o_gate = ColumnParallelLinear(q_lora_rank, n_groups * o_lora_rank)`; :215 `gate = o_gate(qr)`; :235 `wo_b(z * sigmoid(gate))` with z = the grouped low-rank wo_a output: the gate sits between wo_a and wo_b, per o_lora channel.
- Both are built unconditionally in the vLLM code (no `use_gated_attn` test anywhere in attention.py); the config sets `use_gated_attn: true`, `gated_attn_use_q_lora: true`.
- p5 `_feats` (p5:model_graph.py:1748-1755) reads none of these keys and ernie5_5_moe is not in OUT_GATE_TYPES: classify gives `dsa+hyper_conn+kv_compress+router_bias+router_sigmoid`, no out_gate.

## 2. Classifier change
- reads: model_type ernie5_5_moe, `use_gated_attn` (absent -> on, the code is unconditional)
- new: FEATURES key `out_gate` plus the dims field `out_gate_kind` = "elem_qlat" (see out_gate_by_code_unlisted), in _feats / dims_of
- detection:
```python
def detect_out_gate_key_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "ernie5_5_moe" and lm.get("use_gated_attn", True) is not False:
        return "elem_qlat"
    return None
```
- apply note: in `_feats` (p5:model_graph.py:1748) add `or (mt == "ernie5_5_moe" and d.get("use_gated_attn", True) is not False)` to the out_gate condition; in dims_of set `out_gate_kind = "elem_qlat"` for it.

## 3. Relation rule
- target side: needs a host whose gate reads the normed q latent ("elem_qlat"): an x-input gate (sigmoid(W x)) cannot form sigmoid(W RMSNorm(A x)) -- the per-token rescaling is the latent-norm wall (`mla_latent_norm_not_affine`-type). Only elem_qlat hosts (none other in the snapshot).
- host side: an elem_qlat host plays ungated targets (zero o_gate: sigmoid 1/2, double wo_b) and x-gated targets cannot be played by it (a latent-input gate is not an x-input gate)
- kind: wall (target side; and elem_qlat host -> x-gated target); exact (host -> ungated target)
- cost: none
- free by the config rules: no (`use_gated_attn` is a feature switch)
- theorem: `out_gate_off` (host side, PROOFS.md "out_gate host -> ungated target"); target side OPEN ("out_gate / shared_gate target <- ungated host"); needs theorem: a gate of a normed latent is not a gate of x (`gate_qlat_not_x`: sigmoid(w . RMS(Ax)) vs sigmoid(v . x), counterexample along a ray x -> t x)

## 4. Test fixtures
- flag: baidu/ERNIE-5.5-Flash-Base -> "elem_qlat"
- not flag: baidu/ERNIE-4.5-21B-A3B-PT (ernie4_5_moe: no gate); deepseek-ai/DeepSeek-V4-Flash (the HCA / CSA parent without the gate)

## Verdict
real, but no frontier change today: ERNIE-5.5 is already isolated by hyper_conn (wall both ways) and kv_compress. Encode it so the record is right and the model stays walled if those walls are ever priced. Its VHAPostMix cross-head mix (:110, :195) is a further unrepresented op (not this batch).
