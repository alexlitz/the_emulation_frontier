# xsa_self_value_projection
live 9 / placed 9 (grug_moe 7, gpts2 1, gptx2 1); direction over-credit; batch a7_attn_exotic

## 1. What the code computes
- AxiomicLabs/GPT-X2.5-135M/modeling_gptx2.py:62 reads `xsa_projection`; :106-111 after SDPA, per query head in kv
  group g: `y <- y - (y . v_i / max(|v_i|^2, 1e-6)) v_i`, v_i = the CURRENT token's value of group g; then o_proj.
- AxiomicLabs/GPT-S-5M/modeling_gpts2.py:55, 91-97: `v_n = normalize(v_i)`, `y <- y - (y . v_n) v_n` (same map, eps form
  differs).
- ext:grug_moe/gh__marin-community__marin/skyrl_train_models_grug_moe.py:743-745 (no config switch):
  `o <- o - (o . v / (|v|^2 + 1e-6)) v`, then `o *= 2 sigmoid(attn_gate(x))` per head (:746-747);
  ext:grug_moe/gh__marin-community__vllm/grugmoe.py:1046-1061 the same.
- Same code fact, not in the carrier list: paradigma-inc/limite-1b-violetto/modeling_limite.py:187 (`has_xsa = xsa and
  layer_idx in xsa_layers`), :207-208 per-head `xsa_alpha`, :367-395 `y <- y - tanh(alpha_h) (y . v_hat) v_hat`
  (normalize eps = xsa_normalize_eps): a learned strength in (-1, 1), 0 = off. Lilgarca/my-model-v01
  modeling_cyrah.py:484-486 has unit XSA under use_xsa (False in its snapshot).
- y . v_hat is a per-token data-dependent projection (rank-1 removal along the token's own value): no softmax head,
  linear map or gate forms it; on a unit-XSA host it cannot be switched off (no weight makes v_i orthogonal to every
  y while keeping the values; a constant extra value channel would need a constant input, absent under pre-norm).

## 2. Classifier change
- reads: model_type gpts2 / gptx2 (xsa_projection), grug_moe (always), cyrah (use_xsa), limite (xsa, xsa_layers)
- new: FEATURES keys `xsa` ("wall", "wall") and `xsa_scaled` ("wall", "free"), in _feats + FEATURES; an xsa target on
  an xsa_scaled host is a limit (tanh(alpha) -> 1), via FEAT_NEST-with-kind {"xsa": (("xsa_scaled", "limit"),)}
- detection:
```python
def detect_xsa_self_value_projection(cfg, model_type):
    lm = lm_of(cfg)
    mt = model_type
    if mt in ("gpts2", "gptx2") and lm.get("xsa_projection") is True:
        return "unit"
    if mt == "grug_moe":
        return "unit"
    if mt == "cyrah" and lm.get("use_xsa", True) is True:
        return "unit"
    if mt == "limite" and lm.get("xsa", True) is True and lm.get("xsa_layers"):
        return "scaled"
    return None
```
- apply note: p5:model_graph.py `_feats` (after the out_gate block): add "xsa" for "unit", "xsa_scaled" for "scaled"
  (the detector's tests); FEATURES: add both. feats_ok: the FEAT_NEST lookup of out_gate_headwise, with a "limit"
  kind honoured only outside strict (exact slice).

## 3. Relation rule
- target side: an XSA target needs an XSA host (unit: exact on unit hosts with the eps difference a free config
  value; a limit on an xsa_scaled host as alpha -> inf); plain hosts are a wall.
- host side: a unit-XSA host plays only unit-XSA targets (wall to plain targets); an xsa_scaled host plays plain
  targets exactly (xsa_alpha = 0) and scaled targets of any alpha exactly.
- kind: wall (plain <-> unit); exact (scaled host -> plain / scaled target); limit (unit target <- scaled host)
- cost: none (Limite's xsa_layers must cover the target's XSA layers: layer alignment, not priced)
- free by the config rules: the eps (1e-6 clamp / normalize eps / xsa_normalize_eps) is a norm eps (free); the XSA
  switch itself is a feature switch
- theorem: needs theorem: xsa_ne_plain (y - (y.v_hat) v_hat is not a fixed linear map of the head output), xsa_off
  (alpha = 0), xsa_alpha_limit (tanh(alpha) -> 1 gives the unit map, bounded |y|)

## 4. Test fixtures
- flag: AxiomicLabs/GPT-S-5M, AxiomicLabs/GPT-X2.5-135M, open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21,
  open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05,
  open-athena/Grug-67B-A2B-GLM53-RLVR-SFT-2026.09.23, open-athena/Snowball-67B-A2B-Identity-PathB-A13-2026.10.30,
  open-athena/Snowball-67B-A2B-Identity-PathB-A3-2026.10.30, open-athena/snowball-67b-a2b-base-262k-qk175 -> "unit";
  paradigma-inc/limite-1b-violetto -> "scaled"
- not flag: AxiomicLabs/GPT-X3-150M (gptx3, no xsa key), AxiomicLabs/GPT-S3-3M (gpts3_hybrid), Lilgarca/my-model-v01
  (use_xsa False)

## Verdict
real. Fewer edges: the 9 unit-XSA models lose every host and stop hosting plain targets (the 7 Grug 67B-A2B MoEs are
large hosts: likely hundreds of target classes); Limite keeps hosting plain targets and becomes the only (limit) host
for unit-XSA targets that fit it.
