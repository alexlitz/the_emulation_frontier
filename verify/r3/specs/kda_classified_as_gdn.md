# kda_classified_as_gdn
live 1 / placed 1 (alice_ai 1); direction both; batch a11_recurrent_loops

## 1. What the code computes
(AMAImedia/AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF/modeling_alice_ai.py; the two code snapshots are identical)
- `linear_attention` layers build `AliceAIKDA` (:586-587).
- `AliceAIKDA.__init__` (:275-315): separate depthwise causal convs on q, k (key_dim) and v (value_dim), kernel
  linear_conv_kernel_dim (:292-301); low-rank decay `f_b(f_a(x))`, hidden -> head_v_dim -> key_dim (:305-306); beta
  `b_proj` per key head (:307); low-rank output gate `g_b(g_a(x))` rank head_v_dim (:308-309); per-head `o_norm`
  RMSNorm(head_v_dim) (:310-312); `a_log_bias` per head, `dt_bias` per key CHANNEL (:314-315).
- `_torch_kda` (:338-385): q, k l2-normalised, q scaled head_k_dim^-0.5 (:347-348); gate =
  -exp(A_log[h]) * softplus(alpha + dt_bias) per key channel (:349-354); beta = sigmoid(b), x2 only if
  kda_allow_negative_eigenvalues (:355-357, false in the snapshot); state *= exp(gate) per channel (:379).
- `_kda` dispatches to fla chunk_kda / fused_recurrent_kda with `use_qk_l2norm_in_kernel=True` (:386-430).
- `forward`: o = o_proj(o_norm(core) * sigmoid(g_b g_a x)) (:444-470) -- the Kimi-Linear KDA block.
- Config: linear_num_key_heads = linear_num_value_heads = 32, head dims 128, conv 4, no lower bound key.
- p5: alice_ai is not in RECURRENT_FAMILY; `_recurrent_family`'s fallback sees `linear_num_value_heads` and returns
  "gdn" (p5:model_graph.py:2053-2055): classified rf_gdn (per-head scalar decay), rc_rule RC_GDN.

## 2. Classifier change
- reads: model_type alice_ai; layer_types; kda_allow_negative_eigenvalues.
- new: `RECURRENT_FAMILY["alice_ai"] = "kda"` (then `_kda_variant` -> "kda": no lower-bound key).
- detection:
```python
def detect_kda_classified_as_gdn(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "alice_ai":
        return None
    lt = layer_types(lm)
    if lt and "linear_attention" not in lt:
        return None
    return "kda_negeig" if lm.get("kda_allow_negative_eigenvalues") is True else "kda"
```
- apply note: RECURRENT_FAMILY (p5:model_graph.py:2020-2036), next to "youtu_vl_moe": add `"alice_ai": "kda"`.
  `_rec_dims` KDA branch (:2830-2840) already reads the right dims through its top-level fallback: H =
  linear_num_value_heads 32, hd = linear_key_head_dim 128, K = linear_conv_kernel_dim 4, rank = hd 128 (= the code's
  f_a rank head_v_dim; equal here). A kda_allow_negative_eigenvalues=true config (none in the snapshot) would need a
  "kda_negeig" variant like gdn_negeig; not added now.

## 3. Relation rule
- target side: needs a host whose recurrent layers carry rf_kda (Kimi-Linear, GLM-5-Next without the bound, Youtu, the
  universal host); RC_KDA: equal head_dim, heads / conv / rank >=.
- host side: free (rf_ default; zeroed for targets without KDA layers).
- kind: exact on the existing RC_KDA rule; wall to/from rf_gdn (as for every family pair).
- cost: none.
- free by the config rules: n/a (the recurrence is code).
- theorem: `deltaRow_inert`, `deltaRow_pad`, `delta_out_pad` (PROOFS.md "recurrent: gated delta rule (GDN, KDA) zero
  rows / cols") for the RC_KDA padding.

## 4. Test fixtures
- flag: AMAImedia/AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF -> "kda" (the only alice_ai config).
- not flag: gdubicki/Qwen3-Coder-Next-NVFP4-GB10 (real GDN, same linear_* keys); moonshotai/Kimi-Linear-48B-A3B-Instruct
  (KDA already by type); upstage/Solar-Open2-250B (kda_* keys, other type); avlp12/GLM-5.3-Flash-Alis-MLX-4bit
  (glm5_next KDA by type); Advisio/Radha-9B (qwen3_5_text GDN).

## Verdict
real. Both directions, 0 edges today: AliceAI carries attn_res (wall both ways). When attn_res gets a rule it moves
from the GDN hosts to the KDA ones (~tens of pairs either way). Region: recurrent-other (classification).
