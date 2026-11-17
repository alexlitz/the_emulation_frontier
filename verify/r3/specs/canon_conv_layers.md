# canon_conv_layers
live 4 / placed 4 (dhara_ar 1, ethoric 1, needle 2); direction over-credit; batch a7_attn_exotic

## 1. What the code computes
- codelion/dhara-250m/modeling_dhara_ar.py:46-100 CanonLayer: depthwise `Conv1d(kernel canon_kernel, groups=channels,
  bias=canon_bias)` with causal left pad (:87-100), optional SiLU (canon_activation), residual x + conv(x)
  (canon_residual). Snapshot: canon_set "ABCD", kernel 4, residual True, no activation, no bias.
  Positions: A before attention (:449-457, :491-492), B on q / k / v after projection, before RoPE (:276-300,
  :334-338), C before the MLP (:463-471), D on the SwiGLU hidden before down_proj (:417-434).
- techdotus/Ethoric-10M/modeling_ethoric.py:38-74 Canon: causal depthwise conv1d, no bias, residual h + conv(h),
  doc-masked at eos boundaries; :126-132 built from config.canon {enabled, positions, kernel}; :146-159 applied at A
  (normed attention input) and C (normed MLP input). Snapshot: enabled, ["A", "C"], kernel 4.
- ext:needle/gh__cactus-compute__cactus/architecture.py:184-203: with qkv_conv_taps = n, q / k / v each become
  `sum_j taps_j * shift_right(z, j)` (identity init, :186-191; same class as a residual conv) before q / k norm and
  RoPE (position B); config.py:85 defaults qkv_conv_taps to 0 when absent. needle3 extras.qkv_conv_taps = 3;
  Cactus-Compute/needle2 has none (its extras.engram.conv_taps is the engram value conv, a separate feature):
  needle2 is placed by type only, NOT live.
- A short causal conv mixes the last k tokens' channels with fixed per-channel weights. A softmax head reaches
  "attend exactly to t - j" only as a sharpness limit (and loses the per-channel weights), so no plain block
  forms it; a Canon host zeroes the conv (residual: w = 0; needle: taps = (1, 0, ..)) and is the plain block.

## 2. Classifier change
- reads: dhara_ar canon_set / canon_kernel; ethoric canon dict; needle extras.qkv_conv_taps
- new: FEATURES key `canon` ("wall", "free") plus dims field `canon` = "<positions>/k<kernel>", in _feats +
  FEATURES + dims_of
- detection:
```python
def detect_canon_conv_layers(cfg, model_type):
    lm = lm_of(cfg)
    mt = model_type
    if mt == "dhara_ar":
        cs = lm.get("canon_set")
        pos = "".join(sorted(set(str(cs).upper()) & set("ABCD"))) if isinstance(cs, str) else ""
        k = num(lm.get("canon_kernel", 4))
        return f"{pos}/k{k}" if pos and k > 1 else None
    if mt == "ethoric":
        c = lm.get("canon") if isinstance(lm.get("canon"), dict) else (cfg.get("canon") if isinstance(cfg.get("canon"), dict) else {})
        if not c.get("enabled", False):
            return None
        pos = "".join(sorted(set(str(p).upper() for p in (c.get("positions") or []) if isinstance(p, str)) & set("ABCD")))
        k = num(c.get("kernel", 4))
        return f"{pos}/k{k}" if pos and k > 1 else None
    if mt == "needle":
        ex = cfg.get("extras") if isinstance(cfg.get("extras"), dict) else {}
        k = num(ex.get("qkv_conv_taps", lm.get("qkv_conv_taps", 0)))
        return f"B/k{k}" if k > 1 else None
    return None
```
- apply note: p5:model_graph.py `_feats` (new line after the value_emb block, ~1767): `f.add("canon")` when the
  detector fires (needle's extras are on the OUTER config: pass `outer`); dims_of: `canon` string; emul_matrix
  feature check: a canon target on a canon host additionally needs host positions >= target positions and host
  kernel >= target kernel (extra taps zero). FEATURES: add "canon".

## 3. Relation rule
- target side: needs a Canon host with a superset of positions and kernel >= the target's (exact: zero the
  unused convs and taps); plain hosts: wall.
- host side: a Canon host zeroes every conv (residual branch 0 / identity taps) and plays any plain target: free.
- kind: exact (canon target <- covering canon host; canon host -> plain target); wall (canon target <- plain host)
- cost: none
- free by the config rules: kernel / positions are shape switches, not free values; Ethoric doc-masking is inert
  on one document
- theorem: needs theorem: canon_off (zero conv weights give the identity, exact) and canon_pad (a kernel-k' >= k
  conv with zero extra taps equals the kernel-k conv); target side OPEN (same class as the token-mixing walls)

## 4. Test fixtures
- flag: codelion/dhara-250m -> "ABCD/k4", techdotus/Ethoric-10M -> "AC/k4", Cactus-Compute/needle3 -> "B/k3"
- not flag: Cactus-Compute/needle2 (no qkv_conv_taps: off), AxiomicLabs/GPT-X2.5-135M (plain small GQA),
  Qwen/Qwen3-8B

## Verdict
real. Fewer edges: the 3 Canon models are small (10M-250M) plain-looking GQA targets today, hosted by most larger
softmax hosts (each likely hundreds of host classes); with the wall they lose every host (no other Canon host
covers them: dhara ABCD/k4 covers Ethoric AC/k4 only if every other slot fits). As hosts they keep their plain
targets.
