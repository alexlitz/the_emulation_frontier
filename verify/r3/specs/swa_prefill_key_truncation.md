# swa_prefill_key_truncation
live 1 / placed 1 (meditor2 1); direction over-credit; batch a6_masks

The detector also flags Yo-Purple/Meditor-1-2.2B-A500M-MoE (meditor): its code is the same (modeling_meditor.py:164-185).

## 1. What the code computes
- Sliding layers (`is_global_layer` false for (i + 1) % global_attn_every_n_layers != 0, 20 of 24) truncate K / V to
  the LAST sliding_window_size keys before attention: `if k.size(2) > self.sliding_window: k = k[:, :, -W:]`
  (Meditor-2-3.4B-A1.8B-HMoE/modeling_meditor2.py:171-175; Meditor-1 modeling_meditor.py:164-168).
- The explicit band-mask branch requires `kv_len > self.sliding_window` (meditor2 :181-185), which cannot hold after the
  truncation. It is dead code.
- So a cache-less prefill of length L > W runs `F.scaled_dot_product_attention(q, k, v, is_causal=True)` with q_len L
  and kv_len W (:186-188). SDPA's is_causal mask is top-left aligned (tril of an L x W ones matrix). Query i attends
  keys at positions L-W .. L-W+min(i, W-1). Every query i < L-1 sees future tokens; earlier queries also lose their real
  past.
- L <= W: plain causal attention (no truncation). Decode with a cache (q_len 1): the last W keys, a correct window. But
  that cache holds K / V computed from the leaked prefill states.
- So for contexts <= 2048 the sliding layers are full causal attention. Beyond that, the teacher-forced function is
  non-causal and no causal sliding-window target computes it.

## 2. Classifier change
- reads: model_type meditor / meditor2, global_attn_every_n_layers (default 6), sliding_window_size (default 2048)
- new: FEATURES key `swa_trunc` = ("wall", "wall"), in `_feats` + FEATURES
- detection:
```python
def detect_swa_prefill_key_truncation(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in ("meditor", "meditor2"):
        return None
    g = num(lm.get("global_attn_every_n_layers")) or 6
    n = layer_count(lm)
    if not n or all((i + 1) % g == 0 for i in range(n)):
        return None
    return "swa_trunc"
```
- apply note: FEATURES dict (p5:model_graph.py ~L1597) gets the entry; in `_feats` (~L1741) add
  `if mt in ("meditor", "meditor2"): f.add("swa_trunc")`. Keep the sliding-layer kinds from sliding_window_keys_unread
  (they still matter for the context-axis record).

## 3. Relation rule
- target side: wall. Its sliding layers leak future tokens on prefills longer than W. A causal host (`causal_stack`:
  position t depends only on tokens <= t) cannot compute that, and a bidirectional host gets the top-left-aligned key
  subset only by an arbitrary caller mask (not a relation currency).
- host side: wall. Beyond W tokens its sliding layers can neither play a full layer nor a band-local one. It plays a
  plain target only through its 4 global layers (zeroed_sublayer), which kind_match already prices via the LU kinds,
  except when sliding targets are matched onto its leaky layers. The feature keeps those out.
- kind: wall (both); exact only for contexts <= W (a context caveat like DSA's, not taken by default)
- cost: none
- free by the config rules: sliding_window_size is a window size, but remote code gets no lever, and with W >= T the
  layers are plain causal (that would be the lever's reading if one were granted)
- theorem: `causal_stack` / `causal_attn_layer` (a causal host cannot see the future). needs theorem: "no causal stack
  equals top-left causal attention over the last W keys at L > W", a `causal_ne_bidirectional`-style two-point witness.

## 4. Test fixtures
- flag: Yo-Purple/Meditor-2-3.4B-A1.8B-HMoE -> "swa_trunc"; Yo-Purple/Meditor-1-2.2B-A500M-MoE -> "swa_trunc" (same code)
- not flag: mistralai/Mistral-7B-v0.1 (correct band mask), google/gemma-2-2b, open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21

## Verdict
Real, fewer edges for 2 models (both remote-code MoE families with no other member). Meditor-1 and Meditor-2 lose
every edge as target and as host, except to each other (both carry the feature). That is at most a few dozen pairs.
Region: attention.
