# v_head_dim_narrow
live 11 / placed 13 (mimo_v2 9, mimo_v2_flash 1, naive_n05_flash 1, needle 2); direction both; batch a2_dims

## 1. What the code computes
- MiMo-V2-Flash: `XiaomiMiMo/MiMo-V2-Flash/modeling_mimo_v2_flash.py:307` sliding layers `v_head_dim =
  config.swa_v_head_dim`, `:312` full layers `config.v_head_dim`; `:325-326` `v_hidden_size = n_kv * v_head_dim`,
  `o_hidden_size = n_heads * v_head_dim`; `:353` V viewed `(-1, v_head_dim)` while Q / K use head_dim 192. Snapshot:
  head_dim 192, v_head_dim = swa_v_head_dim = 128. MiMo-V2 / V2.5 / V2.6 (`dendriteholdings/teutonic-II-110B-genesis/
  modeling_mimo_v2.py`, MiMoV2Attention.forward) and Naive-N0.5-Flash (`NaiveAI/Naive-N0.5-Flash/
  modeling_naive_n05_flash.py`, attention forward: V viewed with v_dim 128) are the same layout.
- Hymba (same code fact, not a carrier): `nvidia/Hymba-1.5B-Instruct/modeling_hymba.py:503` `v_head_dim =
  config.v_head_dim if > 0 else head_dim`, `:524,530` v_proj `n_kv * v_head_dim`, o_proj `heads * v_head_dim`: V 128
  wide vs q / k 64 (hidden 1600 / 25 heads).
- Needle (not live): `ext:needle/gh__cactus-compute__cactus/architecture.py:174-175` `qk_hd = qk_head_dim or d_model //
  heads`, `v_hd = v_head_dim or ...`; `config.py:17-18` defaults 48 / 64. needle3's config.json saves `qk_head_dim 48`,
  `value_head_dim 64` (the dataclass field is `v_head_dim`; both give 64). Ambiguous: the cactus runtime's field names
  are not the HF ones (d_model / num_heads), so how needle2 (head_dim 64, no qk key) loads is not determined; needle is
  a tool-calling engine format and the carriers' predicate leaves both out.
- p5 (`p5:model_graph.py` dims_of): non-MLA `kv_cap = n_kv * head_dim`, `_attn_variants` swa `kv_cap = swa_n_kv *
  swa_head_dim`; v_head_dim / swa_v_head_dim are read only for MLA (`mla_v`). MiMo-V2.5: kv_cap 1536 = 8 x 192 (the
  sliding shape) treats V as 192 wide; the code's widest V is 8 x 128 = 1024. `_kv_fit_shape` packs by head_dim only. Not fixed in p5.

## 2. Classifier change
- reads: v_head_dim, swa_v_head_dim (mimo_v2 / mimo_v2_flash / naive_n05_flash / hymba), qk_head_dim /
  value_head_dim (needle)
- new: dims `v_head` (V width per KV head, widest layer) and `v_shapes` beside kv_shapes, in dims_of / `_kv_shapes`;
  a V condition in `_kv_fit_shape`
- detection:
```python
V_WIDTH_TYPES = frozenset({"mimo_v2", "mimo_v2_flash", "naive_n05_flash", "needle", "hymba"})

def detect_v_head_dim_narrow(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in V_WIDTH_TYPES or lm.get("kv_lora_rank"):
        return None
    h, nh = _hidden(lm), _heads(lm)
    if not nh:
        return None
    qk = num(lm.get("qk_head_dim")) or num(lm.get("head_dim")) or h // nh
    v = num(lm.get("value_head_dim")) or num(lm.get("v_head_dim")) or qk
    return [qk, v] if v != qk else None
```
- apply note: in dims_of add `"v_head": v` (v_head_dim / swa_v_head_dim max, else head_dim) and per variant extend
  `_kv_shapes`' tuples to (kv heads, q per kv, qk head dim, v head dim) (`_attn_variants` gains the swa_v_head_dim);
  `kv_cap = n_kv * (qk + v) / 2` is NOT the fix -- keep kv_cap for K and compare V separately. In `_kv_fit_shape`
  the packing factor becomes `q = min(hd_h // hd_t, v_h // v_t, qpg_h)` and a pair needs `v_h >= v_t` (zero-padded V
  columns; o_proj reads zeros). For needle, head_dim = qk_head_dim (rope_sched over 48).

## 3. Relation rule
- target side: a target's V heads need host V heads at least as wide: Hymba / needle3 (V wider than q / k) need
  wider hosts (fewer in-edges); MiMo / Naive (V 128 < 192) admit hosts whose V is 128-192 only if their K is 192
  (no change in practice).
- host side: a MiMo / Naive host offers V 128 per head, not 192: it no longer plays targets with 129-192-wide V
  (e.g. a 192-head-dim plain target): fewer out-edges; packing factor bounded by V width too.
- kind: exact
- cost: none
- free by the config rules: not free (V width is shape)
- theorem: `qknorm_pad` / `zeroed_sublayer` (wider heads, zero o_proj columns); `kvPack_nope` (value slices tau);
  needs theorem: a V-only zero padding (host V wider than target V, Q / K equal) is exact -- a one-line instance of
  `kvPack_score_slices` with the identity pair slice.

## 4. Test fixtures
- flag: XiaomiMiMo/MiMo-V2.5, MiMo-V2-Flash, MiMo-V2.5-Pro, MiMo-V2.6-Flash-RL, NaiveAI/Naive-N0.5-Flash,
  dendriteholdings/teutonic-II-110B-genesis, dealignai/MiMo-V2.6-Flash-RL-UNCENSORED -> [192, 128];
  Cactus-Compute/needle3 -> [48, 64]; nvidia/Hymba-1.5B-Base -> [64, 128]
- not flag: Cactus-Compute/needle2 (no qk / v keys: one width), deepseek-ai/DeepSeek-V2-Lite (MLA: v_head_dim read
  as mla_v), XiaomiMiMo/MiMo-7B-Base (mimo, uniform heads)
- scan: 16 repos (10 mimo_v2 + 2 mimo_v2_flash + Naive = the 11 live plus 2 siblings; needle3; Hymba x2).

## Verdict
real. Mostly fewer edges: the 11 MiMo / Naive hosts lose targets needing V wider than 128 under their 192 heads and
some packings; Hymba x2 (rf_hymba-walled, only Hymba hosts) and needle3 tighten as targets. Few edges (MiMo hosts
are huge; their targets mostly fit 128-wide V).
