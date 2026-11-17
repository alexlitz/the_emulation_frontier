# gla_rc_dims_unread
live 1 / placed 1 (gla 1); direction over-credit; batch a11_recurrent_loops

## 1. What the code computes
(ext:gla/fla/gla.py `GatedLinearAttention.__init__`, :76-175; built by ext:gla/fla/modeling_gla.py:45-61)
- `num_kv_heads = num_kv_heads or num_heads` (gla.py:105); key / value widths `hidden * expand_k`, `hidden * expand_v`
  (:115-116); `head_k_dim = key_dim // num_heads`, `head_v_dim = value_dim // num_heads` (:128-129). Snapshot hidden 2048,
  expand_k 0.5, expand_v 1, num_heads 4 -> head_k 256, head_v 512, kv heads 4.
- Short convs only `if use_short_conv` (:137-156); the snapshot sets use_short_conv false (conv_size 4 unused).
- Decay gate `gk_proj = Linear(hidden, gate_low_rank_dim) -> Linear(., key_dim_per_group, bias)` (:158-159) with
  `gate_low_rank_dim = 16` as the __init__ default (:93); modeling_gla.py does not pass it (:45-61), so it is 16 always.
- Output: swish-gated per-head RMSNorm(head_v_dim) (:162-167).
- The same reading applies to fla DeltaNet (ext:delta_net/fla/delta_net.py:116-121, :137-156): head dims
  hidden * expand / num_heads, use_short_conv default true; no gate rank.
- p5 `_rec_dims` RC_EQ fallback (p5:model_graph.py:2845-2853): rc_h = num_heads 4, rc_k = head_dim (absent -> 0),
  rc_v 0, rc_conv = conv_size 4 (although no conv is built), rc_rank = gate_low_rank_dim (absent -> 0).

## 2. Classifier change
- reads: expand_k, expand_v, num_heads, num_kv_heads, use_short_conv, conv_size, hidden_size (fla GLA / DeltaNet).
- new: dims rc_h / rc_k / rc_v / rc_kh / rc_conv / rc_rank for fam "gla" (and "deltanet"), in `_rec_dims`.
- detection (returns [rc_h, rc_k, rc_v, rc_kh, rc_conv, rc_rank]):
```python
def detect_gla_rc_dims_unread(cfg, model_type):
    lm = lm_of(cfg)
    a = archs(cfg)
    if model_type == "gla" and "GLAForCausalLM" in a:
        ek, ev, short, rank = fnum(lm.get("expand_k"), 0.5), fnum(lm.get("expand_v"), 1.0), False, 16
    elif model_type == "delta_net" and "DeltaNetForCausalLM" in a:
        ek, ev, short, rank = fnum(lm.get("expand_k"), 1.0), fnum(lm.get("expand_v"), 1.0), True, 0
    else:
        return None
    hidden, H = num(lm.get("hidden_size")), num(lm.get("num_heads")) or 4
    if not hidden:
        return None
    kh = (num(lm.get("num_kv_heads")) or H) if model_type == "gla" else H
    short = lm.get("use_short_conv", short) is True
    conv = (num(lm.get("conv_size")) or 4) if short else 0
    return [H, round(hidden * ek) // H, round(hidden * ev) // H, kh, conv, rank]
```
- apply note: `_rec_dims` (p5:model_graph.py:2760), a branch before the RC_EQ fallback (:2844):
  `if fam in ("gla", "deltanet") and mt in ("gla", "delta_net"):` set the six dims as above, `rc_rule=RC_EQ`, rc_x =
  num_hidden_layers (as the fallback), return. Keep the fallback for retnet / rwkv / jet.

## 3. Relation rule
- target side: an rf_gla target needs a host with rf_gla layers at EQUAL rc dims (RC_EQ) or the universal host.
- host side: free (zeroed for targets without GLA layers).
- kind: exact (RC_EQ unchanged; only the dims become the code's).
- cost: none.
- free by the config rules: n/a (shapes).
- theorem: none new (RC_EQ is equality); a later GE rule for GLA would use `dscan_inert` / `cconv_pad` (zero-padded
  keys / taps are inert in a diagonal-gated scan; the swish-gated per-head RMSNorm forbids padding head_v).

## 4. Test fixtures
- flag: fla-hub/gla-1.3B-100B -> [4, 256, 512, 4, 0, 16]; fla-hub/delta_net-1.3B-100B -> [16, 128, 128, 16, 4, 0]
  (the DeltaNet dims the fla_num_heads_unplaced placement needs).
- not flag: fla-hub/transformer-1.3B-100B (fla, softmax); fla-hub/rwkv7-0.1B-g1 (fla RWKV-7, own dims);
  RWKV/v5-Eagle-7B-HF (rwkv5, RC_EQ fallback); ridger/MMfreeLM-370M (hgrn_bit).

## Verdict
real, small. The only GLA point changes dims (rc_k 0 -> 256, rc_v 0 -> 512, rc_conv 4 -> 0, rc_rank 0 -> 16); no other
rf_gla point exists, so the frontier moves by ~0 edges (the universal host plays it at the join either way). It fixes
the stored shape for later rules. Region: recurrent-other.
