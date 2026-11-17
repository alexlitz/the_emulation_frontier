# post_norm_variant
live 2 / placed 2 (muse_spark_text 2); direction over-credit; batch a4_norms

## 1. What the code computes
- MuseSparkTextDecoderLayer replaces Glimmer's post norms: `post_attention_layernorm = MuseSparkRMSNorm(hidden, eps=post_norm_eps, with_scale=False)` (weight-free) and `post_feedforward_layernorm = MuseSparkTextInputScaledRMSNorm(hidden, eps=post_norm_eps)` (tfmain:models/muse_spark/modular_muse_spark.py:469-473).
- InputScaledRMSNorm: `output = x * (1.0 + self.weight)` THEN `self._norm(output)` -- gain before the normalisation, unit-RMS output (modular_muse_spark.py:302-305).
- Both outputs go straight into the residual gate: `hidden_states = self.post_attention_residual_gate(residual, hidden_states)` (:496-497, :502-503), MuseSparkResidualGate = `alpha * residual + beta * hidden_states`, beta = sigmoid(gate / T) a per-channel vector (:451-462).
- Consequences (algebra on the code): (a) beta ⊙ N(y) is a weighted post-norm with gain beta, so the weight-free post-norm carries a gain after all, in the gate; (b) N((1 + w) ⊙ y) with y = post_expert_proj(...) / down_proj(...): (1 + w) folds into that projection's rows exactly, leaving a weight-free norm, then (a). Eps (post_norm_eps 1e-8) is a free config value.
- So the "variant" is a reparametrisation of the Gemma-style weighted sandwich post-norm; what differs from every other post_norm family is the residual coefficient alpha (≠ 1, coupled to beta: alpha^2 = sigmoid(-w)(1 + sigmoid(w)) = 1 - beta^2 above the 1e-3 clamp) -- the residual_gain_vector finding.

## 2. Classifier change
- reads: model_type muse_spark / muse_spark_text.
- new: none for the norm itself (keep FEATURES post_norm, already set: muse_spark / muse_spark_text are in POST_NORM_TYPES, p5:model_graph.py ~1968-1971). The residual-gate slot is specified in residual_gain_vector ("res_gain" = "gate").
- detection (records the code fact; the relation does not need a new slot):
```python
def detect_post_norm_variant(cfg, model_type):
    if model_type in ("muse_spark", "muse_spark_text"):
        return "weightfree_inputscaled"
    return None
```
- apply note: nothing to apply in classify / _feats; make sure the residual_gain_vector change lands (that is where Muse-Spark's over-credit is).

## 3. Relation rule
- target side: a Muse-Spark target on a Gemma-style sandwich host: post-norm gain := beta, the (1 + w) input scale folded into down_proj / post_expert_proj -- the post-norm part is exact; the alpha residual is not (residual_gain_vector).
- host side: a Muse-Spark host playing a sandwich target: post-norm gain must come from beta, and beta fixes alpha (alpha^2 + beta^2 = 1): not separable from the gate -> handled by residual_gain_vector ("gate": wall both ways).
- kind: exact for the norm variant itself; the wall lives in the residual gate.
- cost: none.
- free by the config rules: post_norm_eps free (norm eps); the norm variant is code.
- theorem: `norm_weight_fold` / `scalar_fold` (gains into the adjacent linear map) and `zero_centred_norm` ((1 + w) reparametrisation) -- PROOFS "Every allowed rule" rows.

## 4. Test fixtures
- flag: Interchained/Muse-Spark-Open-816B-A42B-GGUF, staccs/Muse-Spark-Open-816B-A42B-lecore-mlx -> "weightfree_inputscaled".
- not flag: hweller1/wayfinder-glimmer-30b, RadixArk/Muse-Glimmer-q4-MLX (the parent Muse-Glimmer: weighted Gemma-2 post norms, plain residual).

## Verdict
non-issue as its own slot: the weight-free / input-scaled post norms are a reparametrisation of a weighted post-norm (gain in the gate's beta, input scale folded into the projection); the real over-credit is the coupled residual gate, covered by residual_gain_vector. No edge change from this spec.
