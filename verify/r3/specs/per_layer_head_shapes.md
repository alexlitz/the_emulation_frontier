# per_layer_head_shapes
live 19 / placed 19 (mimo_v2 9, laguna 6, deci 1, domynedge 1, mimo_v2_flash 1, naive_n05_flash 1); direction both;
batch a2_dims

## 1. What the code computes
- Laguna: `tf5:models/laguna/modeling_laguna.py:427` `LagunaAttention(config, layer_idx,
  config.num_attention_heads_per_layer[layer_idx])`; `tf5:models/laguna/configuration_laguna.py:135-136` default
  `[num_attention_heads] * n` when None. Remote `poolside/Laguna-XS-2.1/modeling_laguna.py:366,382,385` `num_heads =
  num_heads or num_attention_heads`, q_proj / o_proj `num_heads * head_dim`, 8 KV heads on every layer. Snapshot:
  S-2.1 48 heads x 12 full / 72 x 36 sliding; XS-2.1 / XS.2 48 x 10 / 64 x 30. Laguna-M.1 has no list (uniform 64:
  carrier, not really live).
- DeciLM-7B: `Deci/DeciLM-7B-instruct/modeling_decilm.py:35,38` `head_dim = hidden // heads`, `num_key_value_heads =
  config.num_key_value_heads_per_layer[layer_idx]` (4 on 7 layers, 2 on 14, 1 on 11).
- Domyn-Edge: `domyn/Domyn-Edge-Base/configuration_domynedge.py:39,103-110,122-124` head_sizes per layer -> 
  `per_layer_config[i].head_dim` (built in __init__, not saved), rope_dim = the rope layers' head size;
  `modeling_domynedge.py:203,212-221` q / k / v / o at `heads * per_layer head_dim` (256 on the 12 full NoPE layers, 64
  on the 24 sliding RoPE layers), `:87` inv_freq over `config.rope_dim` (64).
- MiMo-V2 / V2-Flash / Naive-N0.5: sliding layers take swa_num_key_value_heads 8 / swa_head_dim (e.g.
  `XiaomiMiMo/MiMo-V2-Flash/modeling_mimo_v2_flash.py:307-312,325`), full layers num_key_value_heads 4.
- p5 (`p5:model_graph.py`): `_kv_shapes` collects the top-level shape plus `_attn_variants` (swa_*, per_layer_config,
  global_head_dim, attention_other_setting) into dims kv_shapes, and `kv_heads_fit` asks every target shape to fit
  SOME host shape. So the MiMo-V2 / Flash / Naive TARGET side is already fixed in p5. Not read:
  num_attention_heads_per_layer (Laguna: heads / q_cap / qheads 48, q_per_kv 6), num_key_value_heads_per_layer
  beyond its max (`_num` of the list; DeciLM kv_heads 4 everywhere), head_sizes (Domyn: head_dim 80 = 2560 / 32,
  matching no layer; rope_sched (80,)). HOST side for every multi-shape stack (MiMo included): a host is credited its
  best shape for each target shape, i.e. DeciLM's 4-KV-head shape on all 32 layers although 25 layers have fewer.

## 2. Classifier change
- reads: num_attention_heads_per_layer, num_key_value_heads_per_layer, head_sizes (+ rope_dim / position_layers for
  Domyn's rope schedule), swa_* (existing)
- new: dims kv_shapes from the per-layer lists, a per-shape layer count `kv_shape_counts`, q_cap / qheads / head_dim
  as the widest layer, in dims_of / `_kv_shapes`; a counted fit in `kv_heads_fit`
- detection:
```python
def detect_per_layer_head_shapes(cfg, model_type):
    lm = lm_of(cfg)
    h, nh = _hidden(lm), _heads(lm)
    if not nh or lm.get("kv_lora_rank"):
        return None
    nkv = num(lm.get("num_key_value_heads")) or nh
    hd = num(lm.get("head_dim")) or h // nh
    shapes = set()
    def add(q, k, d):
        if q and k and d:
            shapes.add((int(k), max(1, int(q) // int(k)), int(d)))
    qpl, kpl, hs = lm.get("num_attention_heads_per_layer"), lm.get("num_key_value_heads_per_layer"), lm.get("head_sizes")
    if isinstance(qpl, list) and qpl:
        for q in qpl: add(num(q), nkv, hd)
    elif isinstance(kpl, list) and kpl:
        for k in kpl: add(nh, num(k), hd)
    elif isinstance(hs, list) and hs:
        for d in hs: add(nh, nkv, num(d))
    elif lm.get("swa_head_dim") or lm.get("swa_num_key_value_heads"):
        add(nh, nkv, hd)
        add(num(lm.get("swa_num_attention_heads")) or nh, num(lm.get("swa_num_key_value_heads")) or nkv,
            num(lm.get("swa_head_dim")) or hd)
    return [list(x) for x in sorted(shapes)] if len(shapes) > 1 else None
```
- apply note: (1) `_kv_shapes`: add the per-layer list shapes (as tuples). (2) dims_of: `q_cap` / `qheads` = max over
  num_attention_heads_per_layer x head_dim; `head_dim` / `q_cap` / `kv_cap` = max over head_sizes (Domyn), and
  `_rope_schedules` need = rope_dim for domynedge (position_layers -> no_rope_layers in `_normalize`, which also gives
  nope_layers). (3) `kv_heads_fit`: replace "each target shape fits SOME host shape" by a counted fit: new dims
  `kv_shape_counts = ((kv, qpk, hd, n_layers), ...)`; a target with counts c_a fits a host with counts c_b iff for
  every up-set U of target shapes (Hall), sum of c_a over U <= number of host layers whose shape fits some shape in
  U (the `_subsets_hall` pattern kind_match uses; fit = `_kv_fit_shape`). Single-shape stacks reduce to today's rule.

## 3. Relation rule
- target side: every target layer needs a host layer whose KV / query / head-dim shape fits it (Laguna-S's 36 layers
  of 72 query heads, Domyn's 256-wide full layers): fewer hosts.
- host side: a host offers each shape only on as many layers as it has (DeciLM: 7 layers of 4 KV heads; MiMo: its
  full layers 4 x 16): fewer targets.
- kind: exact
- cost: none (capacity per layer)
- free by the config rules: not free (head counts / widths are shape)
- theorem: `zeroed_sublayer`, `qknorm_pad` (more / wider heads), `mqa_sub_gqa` / `gqa_sub_mha`, `kvPack_*` per layer;
  needs theorem: per-layer composition (layer i of the host plays layer pi(i) of the target for an order-preserving
  injection pi when depth is padded with zeroed layers) -- `zeroed_sublayer` covers the padding, the per-layer
  matching under order is not stated (the counted Hall check ignores order, as kind_match does).

## 4. Test fixtures
- flag: Deci/DeciLM-7B-instruct -> [[1,32,128],[2,16,128],[4,8,128]]; domyn/Domyn-Edge-Base -> [[8,4,64],[8,4,256]];
  poolside/Laguna-XS-2.1, Laguna-XS.2 -> [[8,6,128],[8,8,128]]; poolside/Laguna-S-2.1, TensorFold/Laguna-S-2.1-MLX-6bit,
  mlx-community/Laguna-S-2.1-oQ4e -> [[8,6,128],[8,9,128]]; XiaomiMiMo/MiMo-V2.5, MiMo-V2-Flash, NaiveAI/Naive-N0.5-Flash
  -> [[4,16,192],[8,8,192]]; dendriteholdings/teutonic-II-110B-genesis -> [[4,12,192],[8,6,192]]
- not flag: poolside/Laguna-M.1 (no per-layer list: uniform), XiaomiMiMo/MiMo-V2.5-Pro (swa shape = full shape,
  8 x 16 x 192: one shape), google/gemma-3-1b-it, Qwen/Qwen3-30B-A3B
- scan: 15 repos. Of the 19 live, Laguna-M.1 and the four MiMo-V2.5 / V2.6 Pro configs have one attention shape
  (they are live for v_head_dim_narrow only); TensorFold/MiMo-V2.6-Flash-RL-MLX-4bit-MTP is an extra Flash sibling.

## Verdict
real for Laguna (5), DeciLM, Domyn and for the host side of every multi-shape stack; the MiMo-V2 / Naive target side
is already fixed in round2-p5 (`_kv_shapes` / `kv_heads_fit`). Mostly fewer edges (~15 models, as hosts and as
targets); Domyn's wider full layers add some host capacity.
