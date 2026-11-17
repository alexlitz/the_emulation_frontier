# head_dim_key_unread
live 5 / placed 60 (nemotron_h 45, ceno 12, jetmoe 1, plamo2vl 1, zamba 1); direction both; batch a2_dims

## 1. What the code computes
- Nemotron-H remote (Nemotron-H-4B-Base-8K / -Instruct-128K):
  `nvidia/Nemotron-H-4B-Base-8K/modeling_nemotron_h.py:845-848` `head_dim = config.attention_head_dim if not None
  else hidden_size // num_attention_heads`; `configuration_nemotron_h.py:143,186` default `attention_head_dim=128`.
  Both configs: hidden 3072, 32 heads, attention_head_dim 128, no head_dim -> code 128, dims_of 96.
- tf5 native path: `tf5:models/nemotron_h/configuration_nemotron_h.py:98` `head_dim: int = 128` (no
  attention_head_dim field); `tf5:models/nemotron_h/modeling_nemotron_h.py:855` `getattr(config, "head_dim", ...)`.
  Same 128 on both code paths for these two configs; a config with neither key also gets 128 (none in the snapshot).
- CENO: `CladeTeam/CENO-80M-A30M-base/modeling_ceno.py:1261-1264` same rule; every CENO config sets
  `attention_head_dim: null` and head_dim = hidden // heads, so CENO coincides (carrier, not live).
- Zamba-1: `tf5:models/zamba/configuration_zamba.py:60` `attribute_map head_dim -> attention_head_dim`;
  `:99-100` `attention_hidden_size = 2*hidden`, `attention_head_dim = 2*hidden // heads`;
  `tf5:models/zamba/modeling_zamba.py:133-143` q/k/v read the 2h-wide concat, `head_dim = attention_head_dim`,
  `scaling = (head_dim/2)^-0.5` (folds into W_q). Zamba-7B-v1: 2*3712 // 16 = 464 (dims_of: 232).
- Zamba-2 (same code fact, not in the carriers): `tf5:models/zamba2/configuration_zamba2.py:71,114-115`,
  `tf5:models/zamba2/modeling_zamba2.py:246-256`: head_dim = attention_head_dim = 2*hidden // heads (snapshot
  configs save it: 128 / 160 / 224 vs dims_of 64 / 80 / 112).
- JetMoE: `tf5:models/jetmoe/configuration_jetmoe.py:48` `attribute_map head_dim -> kv_channels`, `:54` default 128;
  `tf5:models/jetmoe/modeling_jetmoe.py:453-460` `kv_projection_size = kv_channels * num_kv_heads`,
  `head_dim = kv_channels`; `:100` rotary dim = `config.head_dim` (= kv_channels). jetmoe-8b: 128 (dims_of 64).
- PLaMo-2: `pfnet/plamo-2.1-2b-vl/modeling_plamo2.py:1042-1053` `head_dim = config.hidden_size_per_head`,
  q / k / v / o sized by it; `:115` default 128. plamo-2.1-2b-vl: hidden 2048 / 32 heads = 64, code 128.
- Effect in p5 (`p5:model_graph.py` dims_of): head_dim, q_cap, kv_cap, kv_shapes and `_rope_schedules`'
  rope_sched / rope_cap all use `head_dim key or hidden // heads`. Nemotron-H-4B: q_cap 3072 / kv_cap 768 (code
  4096 / 1024); jetmoe-8b q_cap 2048 / kv_cap 1024 / rope (64,) (code 4096 / 2048 / (128,)); plamo-2.1-2b-vl q_cap
  2048 / kv_cap 256 / rope (64,) (code 4096 / 512 / (128,)); Zamba-7B q_cap = kv_cap 3712 (code 7424).

## 2. Classifier change
- reads: attention_head_dim (nemotron_h / nemotron_h_puzzle / ceno / zamba / zamba2), kv_channels (jetmoe),
  hidden_size_per_head (plamo2 / plamo2vl); the zamba 2*hidden // heads default; the nemotron_h 128 default
- new: dims field values head_dim / q_cap / kv_cap / q_cap_u / kv_cap_u / kv_shapes / rope_sched / rope_cap,
  through one helper `_attn_head_dim(d)` in dims_of and `_rope_schedules`
- detection:
```python
def detect_head_dim_key_unread(cfg, model_type):
    lm = lm_of(cfg)
    h, nh = _hidden(lm), _heads(lm)
    if not nh:
        return None
    plain = num(lm.get("head_dim")) or h // nh
    hd = 0
    if model_type in ("nemotron_h", "nemotron_h_puzzle", "ceno"):
        if lm.get("head_dim") is None:
            if lm.get("attention_head_dim") is not None:
                hd = num(lm["attention_head_dim"])
            elif model_type != "ceno" and "attention_head_dim" not in lm:
                hd = 128
    elif model_type in ("zamba", "zamba2"):
        hd = num(lm.get("attention_head_dim")) or (2 * h // nh)
    elif model_type == "jetmoe":
        hd = num(lm.get("kv_channels")) or 128
    elif model_type in ("plamo2", "plamo2vl"):
        hd = num(lm.get("hidden_size_per_head")) or 128
    return hd if hd and hd != plain else None
```
- apply note: add `_attn_head_dim(d)` beside `_kv_shapes` returning the detector's value or the current
  `head_dim key or hidden // n_heads`; use it for `head_dim` at the top of `dims_of` (replaces the
  `(g("head_dim") ...) or (hidden // n_heads)` expression) and for `hd` in `_rope_schedules`. (Equivalent: in
  `_normalize`, set `u["head_dim"]` for these types, as the chatglm branch does for kv_channels; that also feeds
  `_kv_shapes` and `classify`'s `qr < hd` test.) Zamba's `rope_sched` matters only under zamba_shared (Zamba-1 is
  NoPE by NOPE_BY_CODE).

## 3. Relation rule
- target side: hosts must hold the true widths (q_cap, kv_cap, head_dim per KV head, rope_sched): fewer hosts.
- host side: the host offers the true, larger widths: more targets.
- kind: exact (both directions; capacity only)
- cost: none (a dims correction)
- free by the config rules: not a config lever; head widths are shape-changing
- theorem: `qknorm_pad`, `zeroed_sublayer` (Every allowed rule: q_cap / qheads / head_dim, more or wider heads);
  KV packing `kvPack_nope` / `kvPack_rope_iff`; rot gate `partial_rope_host`.

## 4. Test fixtures
- flag: Zyphra/Zamba-7B-v1 -> 464; jetmoe/jetmoe-8b -> 128; nvidia/Nemotron-H-4B-Base-8K -> 128;
  nvidia/Nemotron-H-4B-Instruct-128K -> 128; pfnet/plamo-2.1-2b-vl -> 128 (the 5 live);
  Zyphra/Zamba2-2.7B -> 160, Zyphra/Zamba2-7B-Instruct -> 224 (same code fact, Zamba-2)
- not flag: nvidia/Nemotron-H-8B-Base-8K (attention_head_dim 128 = 4096/32), nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
  (head_dim 128 saved), nvidia/Nemotron-H-56B-Base-8K, CladeTeam/CENO-80M-A30M-base and -1.3B (attention_head_dim
  null), zai-org/glm-4-9b (kv_channels already read by the chatglm branch)
- scan: 11 repos (the 5 live + 6 Zamba-2); nothing else.

## Verdict
real. Both directions, small: Nemotron-H-4B x2 and jetmoe-8b / plamo-2.1-2b-vl gain host capacity (more out-edges)
and lose under-sized hosts (fewer in-edges); Zamba / Zamba-2 change only Zamba-to-Zamba edges (zamba_shared walls).
Roughly 5-11 models, tens of edges.
