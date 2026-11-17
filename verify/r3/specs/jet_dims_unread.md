# jet_dims_unread
live 1 / placed 1 (jet_nemotron 1); direction over-credit; batch a2_dims

## 1. What the code computes
- `jet-ai/Jet-Nemotron-2B/jet_block.py:84-94` from `efficient_attention_config.jet`: `key_dim = num_heads * head_dim`
  (12 x 96 = 1152), `value_dim = int(key_dim * expand_v)` (3072), `head_v_dim = int(head_dim * expand_v)` (256).
- `jet_block.py:112-116` q / k at key_dim, v at value_dim, b / a (beta, decay) per head; `:146-148` g_proj at
  value_dim, `o_norm = FusedRMSNormGated(head_v_dim)`, o_proj from value_dim.
- `jet_block.py:137-144` `DynamicShortConvolution(hidden_size=value_dim, kernel_size=conv_size,
  generator_input_size=hidden_size, generator_reduction=dconv_generator_reduction)`;
  `jet-ai/Jet-Nemotron-2B/dynamic_conv.py:55,69,71` `generator_hidden_size = value_dim // reduction` (3072 // 8 = 384),
  `w1: hidden (1536) -> 384`, `w2: 384 -> value_dim * kernel_size` (+ bias): per-token conv kernels from a rank-384
  generator. (The reviewer's hidden // reduction = 192 is not what the code builds: the reduction divides value_dim.)
- p5 (`p5:model_graph.py` `_rec_dims`, RC_EQ branch): `rc_h = jet.num_heads` (12), `rc_k = jet.head_dim` (96),
  `rc_conv = jet.conv_size` (4), `rc_v = 0`, `rc_rank = g("decay_low_rank_dim", ...)` = 0 (no such key);
  `_rc_pair_ok` RC_EQ compares every RC_DIMS equal, so expand_v and the generator rank never enter. Not fixed in p5.

## 2. Classifier change
- reads: efficient_attention_config.jet.{head_dim, num_heads, expand_v, dconv_generator_reduction}
- new: dims rc_v = head_dim * expand_v, rc_rank = value_dim // dconv_generator_reduction, in `_rec_dims`
- detection:
```python
def detect_jet_dims_unread(cfg, model_type):
    lm = lm_of(cfg)
    jet = (lm.get("efficient_attention_config") or {}).get("jet") if model_type == "jet_nemotron" else None
    if not isinstance(jet, dict):
        return None
    hd, H = num(jet.get("head_dim")) or 256, num(jet.get("num_heads")) or 6
    ev = fnum(jet.get("expand_v"), 2.0)
    red = num(jet.get("dconv_generator_reduction"))
    value_dim = int(H * hd * ev)
    return [int(hd * ev), value_dim // red if red else value_dim]
```
- apply note: in `_rec_dims`' RC_EQ branch, when `fam == "jet"` add `rc_v=int(head_dim * expand_v)` and
  `rc_rank=value_dim // dconv_generator_reduction` (JetBlockConfig defaults `jet_block.py:44-49`: expand_v 2.0,
  num_heads 6, head_dim 256, reduction 8) to the `out.update(...)`.

## 3. Relation rule
- target side: a Jet target needs a Jet host with the same rc_v and rc_rank (RC_EQ), or the universal host (rc_free)
  at the join's dims, which now must carry V width 256 per head and the 384-rank kernel generator.
- host side: a Jet host plays only Jet targets with equal dims.
- kind: exact (RC_EQ is equality; no new lever)
- cost: the universal host's recurrent join grows (rc_v 256, rc_rank 384)
- free by the config rules: not free (expand_v and the generator width are shape)
- theorem: RC_EQ equality needs none; a GE rule would need `lowrank_pad` (rank) and `deltaRow_pad` (value rows) --
  not claimed here.

## 4. Test fixtures
- flag: jet-ai/Jet-Nemotron-2B -> [256, 384]
- not flag: nvidia/Nemotron-H-8B-Base-8K (Mamba-2 hybrid, no jet block), Qwen/Qwen3-Next-80B-A3B-Instruct (GDN, no
  dynamic conv)
- scan: only Jet-Nemotron-2B.

## Verdict
real but latent: one jet_nemotron model, so no edge between real models changes (RC_EQ already pairs it only with
itself); only the universal host's recurrent join and any future Jet point (a second expand_v) are affected.
