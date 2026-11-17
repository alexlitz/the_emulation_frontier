# grouped_rmsnorm
live 6 / placed 10 (k2_horizon 10, incl. the Jais-ASR wrapper); direction over-credit; batch a4_norms

## 1. What the code computes
- `K2HorizonRMSNorm(hidden_size, n_groups, eps)`: `x.reshape(..., n_groups, -1)`, `variance = x.pow(2).mean(-1)` PER GROUP, `x * rsqrt(variance + eps)`, reshape back, `* weight` (IFM/K2-Horizon-7B/modeling_k2_horizon.py:613-634; same class in IFM/Jais-ASR/modeling_jais_asr.py K2HorizonRMSNorm).
- Built with `n_groups=config.layernorm_num_groups` for input_layernorm / post_attention_layernorm (modeling_k2_horizon.py:660-667) and the final norm (:835-838); `assert hidden_size % layernorm_num_groups == 0`.
- Snapshot values: 4 (K2-Horizon-7B, -32B, Jais-3-32B), 2 (K2-Horizon-3.7B, Jais-ASR, Jais-3-MoVA-36B-A4B), 1 = an ordinary RMSNorm (K2-Horizon-375B-A23B, K2-Type-0.9B, Jais-3-0.9B, Jais-3-375B-A23B: placed, correctly not live).
- With g groups of size s = d / g the statistic is a per-group RMS: N(x) = concat_i w_i * x_i / rms(x_i). For g > 1 this is not a function of the full-width RMS; classify reports norm=RMSNorm.

## 2. Classifier change
- reads: `layernorm_num_groups` (k2_horizon / jais_asr code).
- new: dims field `norm_groups` (int, 1 = full width; 0 for non-RMS norms is fine), in `_pass5_dims` (or `dims_of`).
- detection:
```python
def detect_grouped_rmsnorm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in ("k2_horizon", "jais_asr"):
        g = num(lm.get("layernorm_num_groups"))
        return g if g > 1 else None
    return None
```
- apply note: p5:model_graph.py `_pass5_dims` (~3026): out["norm_groups"] = detector value or 1; add to `_point_key_dims` (~4605). In `emul_matrix` (next to the feats_ok mask, ~4229) refuse (t, h) unless `g_h >= g_t and hidden_h // g_h >= hidden_t // g_t` (both RMSNorm); a grouped target on a LayerNorm host, or g_t > 1 on a g_h = 1 host, is a wall. Classify's norm slot stays RMSNorm.

## 3. Relation rule
- target side: a target with g_t groups of size s_t needs a host with g_h >= g_t groups of size s_h >= s_t: each target group sits in its own host group (zero-padded to s_h, padded rows scaled so the mean square is unchanged, or eps' = eps s_t / s_h -- eps is free), unused host groups stay 0 (0 / sqrt(eps) = 0). Two target groups can never share one host group (one joint statistic).
- host side: a grouped host (g_h > 1) plays a full-width target (g_t = 1) only when hidden_t <= hidden_h / g_h (the whole target in one group); otherwise wall. Most K2 hosts (7B: 4 groups) play only targets <= hidden / 4.
- kind: exact when the rule holds; wall otherwise (both directions).
- cost: width: hidden_h >= g_h * s_t (the host's group size sets the usable width).
- free by the config rules: eps is free; layernorm_num_groups is shape-preserving in weights but changes the computation -- a feature switch (a K2 host may NOT set it to 1: not listed as free). If the round decides it is a norm config value like eps, a K2 host with layernorm_num_groups := g_t would play any g_t target of equal width: an open design question.
- theorem: `rmsNorm_pad_scaled` (zero-padded rows scaled by c, c^2 s_t = s_h, same eps) + `rmsNorm_rep` (whole copies) per group, `zeroed_sublayer` (unused groups); refutation `grouped_norm_zero_pad_fails` (zero padding at the same eps, unscaled). Wall: needs theorem `grouped_rms_not_full`: for g > 1 the per-group normaliser is not a function of the full-width RMS (two inputs with equal total RMS, unequal group RMS).

## 4. Test fixtures
- flag: IFM/Jais-ASR -> 2; IFM/K2-Horizon-3.7B -> 2; IFM/K2-Horizon-32B -> 4; IFM/K2-Horizon-7B -> 4; inception42/Jais-3-32B -> 4; inception42/Jais-3-MoVA-36B-A4B -> 2 (every live repo).
- not flag: IFM/K2-Horizon-375B-A23B, IFM/K2-Type-0.9B, inception42/Jais-3-0.9B, inception42/Jais-3-375B-A23B (same code, layernorm_num_groups 1 = full width).
- scan: exactly the 6 live repos.

## Verdict
real. Fewer edges: the 6 grouped K2 / Jais points lose every non-grouped host (as targets) and every target wider than hidden / g (as hosts); K2 points with equal (g, s) or g_h >= g_t, s_h >= s_t keep their mutual edges (e.g. 7B / 32B g = 4 hosts play the g = 2 3.7B only if hidden_h / 4 >= hidden_t / 2).
