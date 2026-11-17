# sliding_window_backend_dependent
live 3 / placed 3 (phimoe 3); direction both; batch a6_masks

Really live: 2 (Phi-tiny-MoE, Phi-mini-MoE). Phi-3.5-MoE has sliding_window 131072 = max_position_embeddings. The
window never binds, and p5's `_normalize` tail already drops it (p5:model_graph.py ~L2673-2679).

## 1. What the code computes
- Phi-tiny / Phi-mini-MoE run the remote `modeling_slimmoe.py` (auto_map AutoModelForCausalLM
  "modeling_slimmoe.PhiMoEForCausalLM"); sliding_window 2047 < max_position_embeddings 4096.
- Default backend sdpa (`_supports_sdpa = True`, Phi-tiny-MoE-instruct/modeling_slimmoe.py:1173): the model forward
  builds `_prepare_4d_causal_attention_mask_for_sdpa(attention_mask, (b, T), x, past)` with NO sliding_window
  (:1366-1374). Plain causal attention over the whole prefix.
- eager: `_prepare_4d_causal_attention_mask(..., sliding_window=self.config.sliding_window)` (:1375-1383), W = 2047 keys.
- flash_attention_2: `window_size=(self.config.sliding_window, 0)` (:655, used when kv_seq_len > sliding_window,
  :490-494), W + 1 = 2048 keys.
- So the computed function depends on the backend. The relation models the default load, sdpa (no attn_implementation in
  the configs): full causal attention on every layer.
- Phi-3.5-MoE's remote modeling_phimoe.py has the same three paths (:1369, :1382), but its window equals the context.

## 2. Classifier change
- reads: model_type phimoe, auto_map module modeling_slimmoe, sliding_window, max_position_embeddings
- new: window = "full" (and no sliding_window) for the slimmoe code, in `_normalize`
- detection:
```python
def detect_sliding_window_backend_dependent(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "phimoe" or "modeling_slimmoe" not in auto_map(cfg):
        return None
    sw, mpe = num(lm.get("sliding_window")), num(lm.get("max_position_embeddings"))
    return "full" if sw and (not mpe or sw < mpe) else None
```
- apply note: `_normalize` (p5:model_graph.py ~L2543) sees only the LM dict, so key on the snapshot's auto_map through
  `_lm_unwrap`'s outer dict, or on the `architectures` + a `head_dim` key (slimmoe configs carry head_dim, Phi-3.5-MoE
  does not). Set `u.update(sliding_window=None, use_sliding_window=False)` as the existing tail does. Do not drop phimoe
  from WINDOW_FREE_TYPES: transformers' own phimoe code still reads the window.

## 3. Relation rule
- target side: a full-window target (FU on all 32 layers)
- host side: a full-window host
- kind: exact (no new rule; the label changes)
- cost: none
- free by the config rules: the window size is free for phimoe (WINDOW_FREE_TYPES ~L3738). That is why p5 already
  treats these points as full on both sides.
- theorem: `window_ge_full` / `band_ge_causal` (PROOFS.md "window lever: window >= T is full")

## 4. Test fixtures
- flag: microsoft/Phi-tiny-MoE-instruct -> "full"; microsoft/Phi-mini-MoE-instruct -> "full" (slimmoe sdpa, 2047 < 4096)
- not flag: microsoft/Phi-3.5-MoE-instruct (window = context); inference-optimization/Phi-3.5-MoE-0.8B-A0.2B
  (transformers code, no auto_map); microsoft/Phi-3.5-mini-instruct (phi3); mistralai/Mistral-7B-v0.1

## Verdict
Already handled in effect by round2-p5, so the headline does not change. phimoe is in WINDOW_FREE_TYPES:
- `_pass5_dims` win_free (~L3036) makes a slimmoe host's sliding layers supply as full (kind_match host_full, ~L3646).
- On a fixed-window host the transitivity rule tgt_full (~L4262) / conv (~L3654) demands its sliding layers as full ones.
So the window=local label is equivalent to full on both sides. What is left is the record: the window label and dims
eff_window (2047, only read under USE_CONTEXT). Apply the label fix for correctness; the frontier does not change.
