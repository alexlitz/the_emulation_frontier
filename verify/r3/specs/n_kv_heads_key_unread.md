# n_kv_heads_key_unread
live 1 / placed 1 (nae1 1); direction over-credit; batch a2_dims

## 1. What the code computes
- `ext:nae1/hub__Subject-Emu-5259__NeuralAI-Nae1/model/nae1_config.py:48-49,63-66,86` Nae1Config(n_head=12,
  n_kv_heads=4), checks `n_head % n_kv_heads == 0`, stores `self.n_kv_heads`.
- `ext:nae1/hub__Subject-Emu-5259__NeuralAI-Nae1/model/nae1_model.py:112,118-120` Nae1GQA: q_proj
  `n_embd -> n_head * head_dim`, k_proj / v_proj `n_embd -> n_kv_heads * head_dim` (4 x 64 = 256);
  `:130-133` k, v viewed as n_kv_heads heads; `:145-146` `repeat_interleave(n_query_groups)`: GQA 12 / 4.
- Snapshot config: `n_head 12, n_kv_heads 4, n_query_groups 3, head_dim 64, n_embd 768`, no num_key_value_heads.
- p5 (`p5:model_graph.py`): classify's nkv reads `num_key_value_heads / num_kv_heads / n_head_kv /
  num_attention_groups`, dims_of's n_kv `num_key_value_heads, num_kv_heads, n_head_kv,
  num_key_value_heads_per_layer, num_attention_groups`; neither reads `n_kv_heads`, so kv=MHA, kv_heads 12,
  kv_cap 768, q_per_kv 1 (code: GQA, 4, 256, 3). Not fixed in p5.
- Other snapshot configs with an n_kv_heads / n_kv_head key and no standard key: nanochat / gaggle (n_kv_head =
  n_head: MHA either way), and types whose heads are under n_heads (llada: `_normalize` reads it; falcon_ocr,
  slm_tavenel, tamelm, minspark, ac_swiglu, lumina_next, molmo: classify has no head count, so unplaced).

## 2. Classifier change
- reads: n_kv_heads, n_kv_head (alias of num_key_value_heads)
- new: slot kv = GQA / MQA and dims kv_heads / kv_cap / q_per_kv, in classify and dims_of
- detection:
```python
_KV_STD_KEYS = ("num_key_value_heads", "num_kv_heads", "n_head_kv", "num_attention_groups",
                "num_key_value_heads_per_layer", "multi_query")

def detect_n_kv_heads_key_unread(cfg, model_type):
    lm = lm_of(cfg)
    nh = _heads(lm)
    if not nh or any(lm.get(k) not in (None, False) for k in _KV_STD_KEYS):
        return None
    kv = num(lm.get("n_kv_heads") or lm.get("n_kv_head"))
    return kv if kv and kv != nh else None
```
- apply note: append "n_kv_heads", "n_kv_head" to the key chain of classify's `nkv = _num(...)` and of dims_of's
  `n_kv = g(...)`, after the existing keys; `_attn_variants`' `nkv` too. (Alternatively a `_normalize` rename for
  any type without num_key_value_heads.)

## 3. Relation rule
- target side: a GQA 12/4 target needs >= 4 host KV heads with query groups (kv_hosts_needed): many more hosts
  (any GQA host with >= 4 KV heads, KV_ORDER GQA -> MHA tiling).
- host side: as a GQA host with 4 KV heads it plays fewer targets than as the 12-KV-head MHA host it is credited as.
- kind: exact
- cost: none
- free by the config rules: not free (KV head count is shape)
- theorem: `mqa_sub_gqa`, `gqa_sub_mha`, `groupOf` (Every allowed rule: kv tiling); `kvPack_*` (KV-head packing)

## 4. Test fixtures
- flag: Subject-Emu-5259/NeuralAI-Nae1 -> 4
- not flag: castorini/gaggle-nanochat-pretrained-climbmix-20260924 and l0ulan/chinese-babylm-nanochat-d22-step10000
  (n_kv_head = n_head), techdotus/Ethoric-10M (n_kv_head beside num_key_value_heads), GSAI-ML/LLaDA-8B-Instruct
  (n_heads type, normalised), tiiuae/Falcon-OCR (n_heads, no placed head count)
- scan: only Nae1.

## Verdict
real, one model: Nae1 moves from MHA (kv_cap 768) to GQA (256): its host out-edges shrink (fewer), its in-edges
grow (more hosts). A handful of edges.
