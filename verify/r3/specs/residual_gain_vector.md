# residual_gain_vector
live 5 / placed 5 (doge 2, muse_spark_text 2, zaya1_vl 1); direction over-credit; batch a4_norms

## 1. What the code computes
- Doge: `input_residual = nn.Parameter(torch.ones(hidden_size))`, `post_attention_residual = ...ones(hidden_size)` (tf5:models/doge/modular_doge.py:389, :393); forward `hidden_states = self.input_residual * residual + hidden_states` and `self.post_attention_residual * residual + hidden_states` (:418, :427): h = g1 ⊙ r + attn(N r), then g2 ⊙ h + mlp(N h). Always built (no config switch).
- ZAYA1-VL (under `scale_residual_merge`, true in the snapshot): `ResidualScaling`: `hidden = (hidden + b_h) * s_h`, and (not on layer 0) `residual = (residual + b_r) * s_r`, per-channel vectors init 1 / 0 (ext:zaya1_vl/gh__Zyphra__transformers/modeling_zaya1_vl.py:1512-1531); applied before every residual merge of the attention / MLP layers (:1390-1391, :1434-1435; :1981-1982, :2021-2022) and before the final merge (:2295-2296, :2423-2424).
- Muse-Spark: `MuseSparkResidualGate`: w = gate / T, beta = sigmoid(w), alpha = sqrt(clamp(sigmoid(-w) (1 + beta), min 1e-3)), out = alpha ⊙ residual + beta ⊙ h (tfmain:models/muse_spark/modular_muse_spark.py:451-462), at both residual adds (:496-497, :502-503). Above the clamp alpha^2 = 1 - beta^2: the two coefficients are coupled per channel.
- A per-channel (non-uniform) gain on the residual inside a pre-norm stack is not a fold: the next RMSNorm sees D ⊙ x, and RMS(D x) is not a function of RMS(x); a uniform gain is the stream-scalar case (exact only with eps / lambda^2 per layer). p5 has no slot for any of these (block sequential, feats from other keys).

## 2. Classifier change
- reads: model_type; zaya `scale_residual_merge`.
- new: FEATURES keys `res_gain` ("wall", "free") for Doge and ZAYA, `res_bias` ("wall", "free") for ZAYA only (value "gain_bias" = res_gain + res_bias), `res_gate` ("wall", "wall") for Muse-Spark ("gate"), in `_feats`.
- detection:
```python
def detect_residual_gain_vector(cfg, model_type):
    if model_type == "doge":
        return "gain"
    if model_type in ("zaya1_vl", "zaya", "zaya1") and truthy_flag(get(cfg, "scale_residual_merge")):
        return "gain_bias"
    if model_type in ("muse_spark", "muse_spark_text"):
        return "gate"
    return None
```
- apply note: p5:model_graph.py FEATURES (after "value_emb", ~1620) two entries; `_feats` (~1760): `if mt == "doge" or (mt in ZAYA_TYPES and d.get("scale_residual_merge")): f.add("res_gain")` (+ `f.add("res_bias")` for ZAYA); `if mt in ("muse_spark", "muse_spark_text"): f.add("res_gate")`. Nothing replaced.

## 3. Relation rule
- target side: a res_gain / res_gate target needs a host with the same per-channel residual gains (Doge / ZAYA hosts play each other's gains; ZAYA's bias too -- a Doge host has no residual bias, so a ZAYA target needs a ZAYA host: res_bias).
- host side: Doge sets g = 1, ZAYA s = 1, b = 0: the plain residual, exact. A Muse-Spark host cannot: alpha = 1 forces beta -> 0 (sigmoid(w) -> 0), killing the sublayer; with a uniform stream rescale c_l the per-channel alpha^2 + beta^2 = 1 still forces beta uniform -> wall.
- kind: res_gain wall (target side) / exact (host side); res_gate wall / wall.
- cost: none.
- free by the config rules: the gains are weights (ones = off), not config; scale_residual_merge is a feature switch (it adds parameters) -- not free.
- theorem: host side exact: identity setting (`value_emb_off`-style, needs theorem `res_gain_one`: g = 1 is the plain residual -- trivial). Target side: needs theorem `res_gain_not_fold`: for a non-uniform diagonal D there is no pre-norm host with D-free residual equal on all inputs (RMSNorm(D x) / RMSNorm(x) not constant); cf. `rmsNorm_smul_eps`, `stream_scalar_needs_eps` (uniform case: REFUTED as free at fixed eps).

## 4. Test fixtures
- flag: SmallDoge/Doge-20M, SmallDoge/Doge-320M -> "gain"; Zyphra/ZAYA1-VL-8B -> "gain_bias"; Interchained/Muse-Spark-Open-816B-A42B-GGUF, staccs/Muse-Spark-Open-816B-A42B-lecore-mlx -> "gate" (every live repo).
- not flag: hweller1/wayfinder-glimmer-30b (Muse-Glimmer parent: plain residual); castorini/gaggle-nanochat-pretrained-climbmix-20260924 (x0 / resid lambdas are scalars per layer under value_emb, not a per-channel gain).

## Verdict
real. Fewer edges: Doge (2) and ZAYA1-VL (1) targets lose every host without residual gains; Muse-Spark (2) loses everything outside itself both ways. Doge / ZAYA as hosts are unchanged (free). Small count (5 points). Region: recurrent-other (residual stream).
