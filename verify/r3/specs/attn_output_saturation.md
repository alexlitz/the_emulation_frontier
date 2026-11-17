# attn_output_saturation
live 1 / placed 1 (qwen3 1); direction over-credit; batch a6_masks

## 1. What the code computes
- Qwen3-0.6B-D-ROS loads the remote modeling_qwen3_eq.py (auto_map AutoModelForCausalLM "modeling_qwen3_eq.Qwen3EQForCausalLM").
  Qwen3DROSAttention (:22) subclasses Qwen3Attention and post-processes its output (after o_proj):
  `conditioned = self._apply_dros(attn_output)` (:178), then a per-layer gain.
- `_apply_dros` (:133-166): off if `d_ros_config.enabled` is false (:136-137). Active only on layers
  int(0.6 L) <= i <= int(0.9 L) (layer_ratio, :140-144: layers 16..25 of 28). It splits the channel axis into
  avg-pool bands (kernels 9 / 3, :113-120; linear, and the bands sum back to x). Each band is saturated by
  `_saturate_band` (:85-106): above a threshold (22 / 28 / 32) the excess becomes knee * tanh(excess * alpha / knee)
  (:75-78). The result is mixed `(1 - dry_wet) x + dry_wet shaped` with dry_wet 0.85 (:146, :162-166).
- `_compute_layer_gain` (:35-54): layer_gains_db {12: 1.5, 13: 1.5, 22: -2, 23: -2.5} -> a scalar 10^(dB/20) per layer.
  A scalar on the attention output folds into o_proj (exact).
- Below every threshold the shaped bands sum to x and D-ROS is the identity. Above, it is a clamp-like nonlinearity on
  the residual write: a regime feature like act_clip / clip_qkv.
- p5: plain qwen3 (qknorm head, no feature).

## 2. Classifier change
- reads: d_ros_config (enabled, dry_wet), auto_map module modeling_qwen3_eq
- new: FEATURES key `attn_out_sat` = ("wall", "wall"), in `_feats` + FEATURES
- detection:
```python
def detect_attn_output_saturation(cfg, model_type):
    lm = lm_of(cfg)
    dc = lm.get("d_ros_config", (cfg or {}).get("d_ros_config"))
    if not isinstance(dc, dict) or dc.get("enabled", True) is False or "modeling_qwen3_eq" not in auto_map(cfg):
        return None
    dw = fnum(dc.get("dry_wet"), 0.70)
    return "attn_out_sat" if dw > 0 else None
```
- apply note: FEATURES (p5:model_graph.py ~L1597, next to act_clip / clip_qkv): `"attn_out_sat": ("wall", "wall",
  "attention-output saturation (Qwen3-D-ROS d_ros_config): per-band threshold / tanh-knee compression of the attention
  output on the middle layers, dry / wet mixed: exact only below the thresholds (a regime, not measured)")`. In `_feats`
  (~L1741) the auto_map is on `outer`: `if isinstance(get("d_ros_config"), dict) and get("d_ros_config").get("enabled",
  True) is not False and "modeling_qwen3_eq" in str(outer.get("auto_map") or d.get("auto_map") or ""):
  f.add("attn_out_sat")`. The layer gains need no rule (folded into o_proj).

## 3. Relation rule
- target side: wall. A plain host computes the identity where D-ROS saturates. Exact only while every band stays below
  its threshold: a regime, no measured agreement.
- host side: wall under p5's convention (remote code gets no config lever). The thresholds are cap values read from the
  config (:87), so by the "scales and caps" design rule a D-ROS host could set them huge and be the identity. That is
  free if remote-code levers are ever granted.
- kind: wall both ways (regime, as act_clip)
- cost: none
- free by the config rules: thresholds / knee = cap values (free in principle, remote); d_ros_config.enabled = a feature
  switch (not free)
- theorem: needs theorem "a thresholded saturation of the attention output equals the identity only on the region
  below the threshold" (the regime statement act_clip / clip_qkv share; no Lean row)

## 4. Test fixtures
- flag: aifeifei798/Qwen3-0.6B-D-ROS -> "attn_out_sat"
- not flag: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B (same qwen3 config class, transformers code)

## Verdict
Real, fewer edges for 1 model. Qwen3-0.6B-D-ROS leaves the large qwen3 class both as target and as host: every edge
to and from it crosses the wall. That is hundreds of pairs, since it was a full member of the qwen3 / Qwen3-0.6B
config class. Region: attention.
