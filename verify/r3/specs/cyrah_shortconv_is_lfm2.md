# cyrah_shortconv_is_lfm2
live 1 / placed 1 (cyrah 1); direction over-wall; batch a3_placement

## 1. What the code computes
- Lilgarca/my-model-v01/modeling_cyrah.py:250-265 CyrahShortConv, conv_dual_gate true (snapshot):
  `in_proj = Linear(d, 3d, bias=False)`, depthwise `Conv1d(d, d, L = conv_L_cache = 3, groups=d, padding=L-1,
  bias=conv_bias=False)`, `pre, post, chan = in_proj(x).chunk(3)`; `y = conv(pre * chan)[..., :t]`;
  `return out_proj(y * post)`.
- tf5:models/lfm2/modeling_lfm2.py:344 `in_proj = Linear(h, 3h, bias=conv_bias)`, :360-362 `B, C, x = in_proj(..).chunk(3)`,
  `B * x`, causal depthwise conv (L_cache), :386 `y = C * conv_out`, then out_proj: out_proj(C * conv(B * x)). The same
  map with the same chunk order (B = pre, C = post, x = chan); modeling_cyrah.py:1524-1531 states the LFM2 in_proj
  transfers verbatim.
- The single-gate layout (conv_dual_gate false, :266-268) is out_proj(gate * conv(chan)) -- not LFM2 (no B gate).
- p5 `_recurrent_family` (p5:model_graph.py :2040) has no cyrah entry -> "other_cyrah" (own-family only); RECURRENT_FAMILY
  (:2020) maps lfm2 / lfm2_moe / lfm3 / lfm3_rd to "shortconv" (RC_GE, rc_conv = conv_L_cache, :2842-2843).

## 2. Classifier change
- reads: model_type cyrah, conv_dual_gate, layer_types containing "conv"
- new: recurrent family "shortconv" for cyrah with conv_dual_gate, in _recurrent_family
- detection:
```python
def detect_cyrah_shortconv_is_lfm2(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "cyrah" and truthy_flag(lm.get("conv_dual_gate")) and "conv" in layer_types(lm):
        return "shortconv"
    return None
```
- apply note: p5:model_graph.py `_recurrent_family` (:2040), before the RECURRENT_FAMILY lookup:
  `if mt == "cyrah": return "shortconv" if d.get("conv_dual_gate") else "other_cyrah"` (GDN2 layers, layer_types
  "linear_attention", would need their own family if a Cyrah config ever lists them; none does). _rec_dims picks
  rc_conv from conv_L_cache (:2843). Note conv_bias: Cyrah's in_proj has no bias whatever conv_bias says; LFM2
  biases in_proj too when conv_bias -- both false in the snapshot; for conv_bias true keep "other_cyrah".

## 3. Relation rule
- target side: a shortconv-family host with rc_conv >= 3 plays Cyrah's conv layers (RC_GE)
- host side: Cyrah plays LFM2 short-conv layers (same map; conv width >=)
- kind: exact
- cost: none
- free by the config rules: n/a (same code)
- theorem: `cconv_pad` (zero extra taps), `zeroed_sublayer`; same-map identity

## 4. Test fixtures
- flag: Lilgarca/my-model-v01 -> shortconv
- not flag: LiquidAI/LFM2-350M, LiquidAI/LFM2.5-230M-Base (already shortconv by type; the detector is Cyrah-only)

## Verdict
real: the Cyrah point joins the shortconv family: rf edges with LFM2-family points appear both ways (more edges,
1 model; still gated by its other features -- out_gate, qknorm head after cyrah_attn_impl_unread). Region:
recurrent-other.
