# expert_output_norm
live 2 / placed 2 (muse_spark_text 2); direction both; batch a10_moe_ffn

## 1. What the code computes
- `MuseSparkExperts.__init__`: experts at `moe_hidden_size` (4096; see kimi_latent_moe_norm), `post_expert_norm = MuseSparkTextInputScaledRMSNorm(4096, eps=post_norm_eps)` (tfmain:models/muse_spark/modular_muse_spark.py:340-345).
- `MuseSparkTextInputScaledRMSNorm.forward`: `N(x) = z / sqrt(ms(z) + eps)` with `z = (1 + w) * x`. The gain is INSIDE the norm and the output has no gain, so its RMS is ~1 (modular_muse_spark.py:302-307).
- `MuseSparkExperts.forward`: `y_e = N(down(act(gate) * up))`, then `* top_k_weights` and `index_add` (:362-371). Every selected expert contributes a unit-RMS vector times its routing weight.
- Router (:325-337): sigmoid with a selection bias, renormalised (+1e-15), times routed_scaling_factor.
- Snapshot: both repos have `post_norm_eps` 1e-8. muse_spark / muse_spark_text are the only types with the code.
- p5 has nothing for a per-expert output norm; classify sees a plain sigmoid MoE (the post_norm feature covers only the sublayer sandwich norm).

## 2. Classifier change
- reads: model_type muse_spark / muse_spark_text (code-fixed, no switch).
- new: FEATURES key `expert_norm` in `_feats`.
- detection:
```python
def detect_expert_output_norm(cfg, model_type):
    if model_type in ("muse_spark", "muse_spark_text"):
        return "input_scaled_rms"
    return None
```
- apply note: p5:model_graph.py FEATURES (~1597): `"expert_norm": ("wall", "wall", "an RMSNorm on every routed expert's output before the routing weight (Muse-Spark post_expert_norm)")`. `_feats` (~1777, inside the router block): `if mt in ("muse_spark", "muse_spark_text"): f.add("expert_norm")`. Under the eps lever, make the host side payable: add `expert_norm` to the lever-switchable set next to the post-norm entries (emul_matrix ~4125, `eps_h`).

## 3. Relation rule
- target side: sum_k w_k N(E_k(x)) has each expert output normalised per token. A plain expert bank's output scales with its input, and no linear readout of a plain MoE gives unit-RMS expert outputs. Only an `expert_norm` host plays it (same norm form, expert width >= target's, MoE rules as today).
- host side: a Muse host playing a plain MoE target. With (1 + w) = 1, write the expert output / S. Then N(y/S) = (y/S) / sqrt(ms(y)/S^2 + eps) -> y / (S sqrt(eps)) as S -> inf, and post_expert_proj x S sqrt(eps) undoes the scale. That is the eps lever (`epsLever_eq` with the gain moved to the following linear map), and eps = post_norm_eps is free (a host may raise it to 1). It is a limit with no measurement, so it stays a wall in the exact slice and is payable only under the lever.
- kind: target wall; host limit (eps lever).
- cost: none.
- free by the config rules: post_norm_eps is free (norm eps). The norm itself has no off switch.
- theorem: host `epsLever_eq` / `epsLever_bound` (gain on the next linear map); `postNorm_layer_bound` is the sublayer analogue. Target wall needs theorem `expert_norm_not_plain`: the degree-0 ray argument of `mla_value_ray`, with a single selected expert (k = 1) on a ray, gives output w N(E(rho x)), constant in rho, while a SwiGLU expert is not degree 0.

## 4. Test fixtures
- flag: Interchained/Muse-Spark-Open-816B-A42B-GGUF, staccs/Muse-Spark-Open-816B-A42B-lecore-mlx -> "input_scaled_rms" (every live repo).
- not flag: RadixArk/Muse-Glimmer-q4-MLX, hweller1/wayfinder-glimmer-30b (the Muse-Glimmer parent, no expert norm); nvidia/Kimi-K3-NVFP4 (a latent norm on the sum, not per expert); Qwen/Qwen3-30B-A3B.
- scan: exactly the 2 Muse-Spark repos.

## Verdict
real. Fewer edges: the 2 Muse-Spark points lose every non-Muse MoE host and, outside the lever slice, every plain-MoE target. In the lever slice they regain hosting. Muse-Spark is already post_norm / sandwich-walled, so roughly the 2 points and their few within-sandwich MoE edges change.
