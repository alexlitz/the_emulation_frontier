# doge_dynamic_mask
live 2 / placed 2 (doge 2); direction over-credit; batch a6_masks

## 1. What the code computes
- DogeAttention builds `self.A = zeros(num_key_value_heads)` and `dt_proj` (values -> kv heads), and stores
  `keep_window_size` (tf5 models/doge/modeling_doge.py:261, :273-274).
- Forward (:303-326): `dt_states = exp(A * softplus(dt_proj(values)))` per kv head and key (:305-308).
  `attn_mask = prepare_dynamic_mask(...)` (:309-314), repeated to the query heads (:315), is passed as the ADDITIVE
  attention mask (:326).
- prepare_dynamic_mask (:336-371): `attn_mask[b, h, i, j] = dt_states[b, h, j]` (:356-358), causal / padding positions
  set to min (:359-365). If the key count > keep_window_size, only the top keep_window_size keys by bias stay (:366-370).
- So the logits are q.k / sqrt(d) + exp(A softplus(w . v_j)): a content-dependent per-key bias, always on. The top-k
  never fires in the snapshots (keep_window_size 2048 = max_position_embeddings).
- Doge-320M's remote modeling_doge.py has the same math per query head (A per attention head, no repeat_kv;
  Doge-320M/modeling_doge.py:252-317, :327-362). Doge-20M runs the transformers code (no auto_map).
- p5: doge reads as plain softmax attention (no feature). It is in WINDOW_FREE_TYPES (~L3739) for its sliding_window,
  which is null in both snapshots.

## 2. Classifier change
- reads: model_type doge
- new: FEATURES key `dyn_mask` = ("wall", "free"), in `_feats` + FEATURES
- detection:
```python
def detect_doge_dynamic_mask(cfg, model_type):
    return "dyn_mask" if model_type == "doge" else None
```
- apply note: FEATURES (p5:model_graph.py ~L1597): add
  `"dyn_mask": ("wall", "free", "Doge dynamic mask: a per-key logit bias exp(A softplus(dt_proj(v))) (+ top-k keys
  beyond keep_window_size): a content-dependent bias no plain layer forms; a Doge host sets A = 0 (bias 1,
  softmax-invariant) and keep_window_size >= T")`. In `_feats` (~L1741): `if mt == "doge": f.add("dyn_mask")`.

## 3. Relation rule
- target side: wall. A plain host would need the bias b_j = exp(A softplus(w . v_j)), a nonlinear function of the key
  token's state, as an extra q.k channel against a constant query channel. Under a pre-norm the constant readout needs
  a channel proportional to the RMS (`const_readout_channel`), the same obstruction as bias_qkv.
- host side: free. A = 0 (a weight) gives exp(0) = 1 on every unmasked key, a constant shift that softmax ignores.
  keep_window_size >= T (a window size: free config, read by transformers' DogeAttention) keeps the top-k off. Exact.
- kind: wall (target); exact (host)
- cost: none
- free by the config rules: keep_window_size is a window size (free); A is a weight (the emulator sets weights)
- theorem: host side, softmax shift invariance (Common.lean softmax; as for `sink_sub_plain`'s logit-0 case).
  Target side: needs theorem "a value-dependent per-key logit bias is not formed by plain pre-norm attention"
  (`const_readout_channel` gives the pre-norm half).

## 4. Test fixtures
- flag: SmallDoge/Doge-20M -> "dyn_mask"; SmallDoge/Doge-320M -> "dyn_mask"
- not flag: Qwen/Qwen3-0.6B, meta-llama/Llama-3.2-1B-Instruct (plain attention)

## Verdict
Real, fewer edges for 2 models as targets: every non-Doge host loses the two Doge targets. As hosts nothing changes
(free). The two Doge points still play each other. WINDOW_FREE_TYPES' doge entry stays right: the window is a mask
parameter, and the dynamic bias is orthogonal to it. Region: attention.
