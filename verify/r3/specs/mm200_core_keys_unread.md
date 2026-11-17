# mm200_core_keys_unread
live 1 / placed 1 (mm200 1); direction both; batch a2_dims

## 1. What the code computes
- `Karasumikura/Ying-K1.5-Lite-200M/mm200_hf.py:37-54` MM200HFConfig keeps every MM200Config field and `to_mm200()`
  builds the native config; `:57-66` MM200ForCausalLM builds `MM200(config.to_mm200())`, `_tied_weights_keys = []`.
- Partial RoPE: `modeling_mm200.py:65` default `partial_rope_frac = 0.25`; `:125-126` `rot_dim = max(2, int(head_dim *
  frac) // 2 * 2)` and inv_freq over rot_dim; `:141-142` only the first rot_dim dims of q / k rotate. Snapshot:
  head_dim 64, frac 0.25 -> 16 rotated dims.
- Output gate AFTER o_proj: `modeling_mm200.py:163-164` `gate_proj = Linear(hidden, nh * hd)` (768 = hidden here);
  `:229-232` `out = self.o_proj(out); if use_gated_attention: out = out * sigmoid(gate_proj(x))`. Qwen3-Next's
  out_gate is before o_proj (`o_proj(attn * sigmoid(W_g x))`); here the hidden-wide gate multiplies o_proj's output.
- Also (no change needed): q / k RMSNorm per head before RoPE `:165-167,181-183` (qknorm head, mapped:
  QKNORM_HEAD_CHECKED), zero-centred norms `:90,254` (foldable), local layers `:170` every i % 4 != 3 with mask
  `0 <= q - kv <= W` (`:224`, W + 1 keys: a window value), `tie_embeddings: false` (`:275`), yarn_mscale 1.0.
- p5 (`p5:model_graph.py`): `_rotary_factor` reads rotary_pct / partial_rotary_factor only -> pos=rope,
  rope_sched (64,); `_feats` out_gate keys do not include use_gated_attention and mm200 is not in OUT_GATE_TYPES ->
  feats ''. Not fixed in p5.

## 2. Classifier change
- reads: partial_rope_frac (mm200), use_gated_attention (mm200, default True)
- new: pos = partial_rope with rope_sched (16,) (`_rotary_factor`); FEATURES key `out_gate_post` (new), in `_feats`
- detection:
```python
def detect_mm200_core_keys_unread(cfg, model_type):
    if model_type != "mm200":
        return None
    lm = lm_of(cfg)
    h, nh = _hidden(lm), _heads(lm)
    hd = num(lm.get("head_dim")) or (h // nh if nh else 0)
    frac = fnum(lm.get("partial_rope_frac"), 0.25)
    rot = max(2, int(hd * frac) // 2 * 2) if hd else 0
    gate = "out_gate_post" if lm.get("use_gated_attention", True) else ""
    if rot >= hd and not gate:
        return None
    return [rot, gate]
```
- apply note: `_rotary_factor`: for mt == "mm200" return `float(d.get("partial_rope_frac", 0.25))` before the
  class-default loop (then classify gives partial_rope and `_rope_schedules` need = round(0.25 * 64) = 16, which
  equals the code's rot_dim here; the code's `// 2 * 2` floor matters only for odd products). FEATURES:
  `"out_gate_post": ("wall", "free", "attention output gate AFTER o_proj, o_proj(attn) * sigmoid(W_g x) (MM200):
  a data-dependent product on the hidden-wide output; a gated host sets W_g = 0 (gate 1/2) and doubles o_proj")`;
  `_feats`: `if mt == "mm200" and d.get("use_gated_attention", True): f.add("out_gate_post")`. Keep it distinct from
  out_gate: neither variant plays the other (pre- vs post-o_proj gating differ unless o_proj is diagonal).

## 3. Relation rule
- target side: as partial_rope (16 of 64) it is rot-gated (hosts' tables must hold 16 rotating dims + 48 pass), and
  out_gate_post walls it from every host without that feature (only MM200 itself): fewer in-edges (to ~0).
- host side: out_gate_post is free on the host side (gate off: W_g = 0, 2 W_o), so its host edges stay; its rot
  cap drops from 64 to 16 rotating dims: it no longer plays full-RoPE targets unless its fraction is free
  (partial_rope_frac is a config value of this code: add mm200 to ROT_FRAC_FREE_TYPES).
- kind: exact (gate off; rot gate), wall (out_gate_post target side)
- cost: none
- free by the config rules: partial_rope_frac is free (partial_rotary_factor); use_gated_attention is a feature
  switch the host can turn off only by weights (here W_g = 0 is exact anyway)
- theorem: `out_gate_off` (out_gate host -> ungated target; the post-o_proj form is the same identity applied after
  o_proj); `partial_rope_host`, `partial_rotary_raise` (rot gate / raising the fraction); the target-side wall is
  OPEN (Walls: out_gate target <- ungated host).

## 4. Test fixtures
- flag: Karasumikura/Ying-K1.5-Lite-200M -> [16, "out_gate_post"]
- not flag: Qwen/Qwen3-Next-80B-A3B-Instruct (pre-o_proj out_gate), poolside/Laguna-S-2.1 (out_gate type, not mm200)
- scan: only Ying-K1.5-Lite-200M.

## Verdict
real, one model: as a target it loses every host but MM200 (fewer edges, its in-edges); as a host it keeps
ungated targets, loses full-RoPE targets unless mm200 joins ROT_FRAC_FREE_TYPES. A handful to tens of edges.
