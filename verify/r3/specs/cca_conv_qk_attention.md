# cca_conv_qk_attention
live 1 / placed 1 (zaya1_vl 1); direction both; batch a7_attn_exotic

## 1. What the code computes
- ext:zaya1_vl/gh__Zyphra__transformers/modeling_zaya1_vl.py:943-970: ZayaAttention ALWAYS builds `self.qkv = CCA(...)`
  (no branch on config.cca; configuration_zaya1_vl.py:61, 121 `cca=True`, :102-103, 174-175 cca_time0 = cca_time1 = 2).
- CCA (:676-759): linear_q / linear_k into a compressed latent, `conv_qk` = depthwise Conv1d(k=cca_time0) then grouped
  Conv1d(k=cca_time1, groups = q + kv heads) over time (:739-751), causal left pad (:846-848); `temp =
  zeros(num_kv_heads)` (:758-759); val_proj1 / val_proj2 (:716-717), each latent_k_dim / 2.
- forward: query = conv_out_q + qk_mean_q, key = conv_out_k + qk_mean_k (:851-859: the (q + k) / 2 mean residual);
  value = concat(val_proj1(x_t), val_proj2(x_{t-1})) (:862-881: half of every value head reads the previous token);
  `key = key * sqrt(d) / |key| * temp_kv`, `query = query * sqrt(d) / |query|` (:883-888: weight-free per-head L2 norm,
  no eps, with a learned per-kv-head temperature). Then partial RoPE + GQA softmax with scale 1/sqrt(d) (:1030).
- tf5:models/zaya/modeling_zaya.py:154-170 ZayaCCAProjection is the same block for the text-only ZAYA (no snapshot).
- p5 classify gives ZAYA1-VL a plain GQA softmax, qknorm=no, feats tied_emb only.

## 2. Classifier change
- reads: model_type zaya1_vl / zaya, cca (default True)
- new: FEATURES key `cca` ("wall", "wall"), in _feats + FEATURES (optionally qknorm "head_free" via
  QKNORM_VARIANT_BY_TYPE for the record; the cca wall decides the edges)
- detection:
```python
def detect_cca_conv_qk_attention(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in ("zaya1_vl", "zaya") and lm.get("cca", True) is not False:
        return True
    return None
```
- apply note: p5:model_graph.py `_feats` (after the out_gate block): `if mt in ("zaya1_vl", "zaya") and
  d.get("cca", True) is not False: f.add("cca")`; FEATURES: add "cca".

## 3. Relation rule
- target side: q / k built from a causal conv over the last tokens and half of each value from x_{t-1}: token mixing
  inside the projections that no plain attention layer forms (a softmax head reaches "previous token" only as a
  sharpness limit): wall.
- host side: zeroing the convs leaves the mean residual (fine) and val_proj2 = 0 halves the value width, but the
  eps-free L2 norm on q and k bounds every logit by d * |temp| and fixes |q| = |k| = sqrt(d): a plain target's
  unnormalised logits are not reachable (the measured qk-norm fold, not payable): wall.
- kind: wall (both directions)
- cost: none
- free by the config rules: cca_time0 / cca_time1 are kernel shapes; the CCA itself is a feature switch
- theorem: needs theorem: cca_ne_plain (token-shift inside q / k / v is not a softmax head) ; host side follows the
  qknorm "approx, not payable" row (QKNORM_ORDER [("no", "approx", False)])

## 4. Test fixtures
- flag: Zyphra/ZAYA1-VL-8B -> True
- not flag: Zyphra/Zamba2-2.7B (zamba2, the same lab, Mamba hybrid), Zyphra/ZR1-1.5B (qwen2), Qwen/Qwen3-8B

## Verdict
real. Fewer edges: ZAYA1-VL-8B (an 8B MoE, today a plain GQA-MoE class) loses all hosts and all its hosted plain
targets (likely dozens to hundreds of edges); no CCA host exists to pair with.
