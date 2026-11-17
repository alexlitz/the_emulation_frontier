# bidirectional_flag_true
live 7 / placed 9 (qwen2 4, mistral 2, mixtral 2, qwen3 1); direction both; batch a3_placement

## 1. What the code computes
- MongoDB/voyage-4-nano/modeling_qwen3_bidirectional.py:30-45: Qwen3BidirectionalModel sets every
  `layer.self_attn.is_causal = False`; :62-78 `or_mask_function = bidirectional_mask_function(attention_mask)` (every
  token attends every non-padded token); :38, :90 output = Linear(hidden, num_labels = 2048) on the last state (an
  embedder; the auto_map maps only AutoModel). Snapshot `use_bidirectional_attention: true`.
- NovaSearch/stella_en_1.5B_v5/modeling_qwen.py:970 `Qwen2Model.forward(..., is_causal=False)`, :1165
  `Qwen2ForCausalLM.forward(..., is_causal=False)`, :1032 / :1045 a causal 4-D mask only `if is_causal`: bidirectional
  unless the caller passes True; the config has no is_causal key. Same file in sdadas/stella-pl-retrieval-8k
  (modeling_qwen.py:995, :1210). gte-Qwen2 ships the same code with config is_causal false.
- GritLM/GritLM-7B/modeling_gritlm7b.py:952 `is_causal: Optional[bool] = True` (8x7B the same): causal by default --
  the GritLM "live" entries are not live (gen ar is right).
- tf5:models/gemma/modeling_gemma.py:239, gemma2/modeling_gemma2.py:231, gemma3/modeling_gemma3.py:320 / :547:
  `is_causal = not use_bidirectional_attention` (a bool flag; Gemma-4 uses the string "vision" / "all").
- p5 `_gen` (p5:model_graph.py :2373-2384) recognises `is_causal is False` and `use_bidirectional_attention == "all"`
  only: voyage-4-nano and both stella configs are gen ar.

## 2. Classifier change
- reads: is_causal, use_bidirectional_attention (bool True), auto_map (*Bidirectional*, the stella / gte
  modeling_qwen.Qwen2* remote code)
- new: gen "diffusion" for those, in _gen
- detection:
```python
def detect_bidirectional_flag_true(cfg, model_type):
    lm = lm_of(cfg)
    am = auto_map(cfg)
    if any(k in lm or k in (cfg or {}) for k in _DRAFTER_KEYS):
        return None                           # drafters (DFlash is_causal False) are skipped before gen is read
    if lm.get("is_causal") is False:
        return "diffusion"
    if "modeling_qwen.Qwen2" in am and "is_causal" not in lm and "is_causal" not in (cfg or {}):
        return "diffusion"                    # modeling_qwen.py Qwen2Model / Qwen2ForCausalLM.forward is_causal=False
    if "Bidirectional" in am:
        return "diffusion"
    if lm.get("use_bidirectional_attention") is True and model_type in _BIDIR_ATTN_TYPES:
        return "diffusion"
    return None
```
  (_BIDIR_ATTN_TYPES = gemma, gemma2, gemma3, gemma3_text, qwen3; _DRAFTER_KEYS = p5 DRAFTER_KEYS.)
- apply note: p5:model_graph.py `_gen` (:2373): `_gen(d, mt, outer)` with outer's auto_map; add
  `or d.get("use_bidirectional_attention") is True` (Gemma-1/2/3 code and voyage), `or "Bidirectional" in
  auto_map`, `or ("modeling_qwen.Qwen2" in auto_map and "is_causal" not in d)` (stella / gte remote default).

## 3. Relation rule
- target side: a bidirectional target needs a bidirectional host (an AR host cannot see the future)
- host side: a bidirectional host plays AR targets under a causal mask, block targets under the block mask
- kind: exact (GEN_ORDER: ar -> diffusion / block)
- cost: none
- free by the config rules: is_causal / use_bidirectional_attention are feature switches (not free)
- theorem: `causal_attn_layer`, `causal_stack` (ar <= diffusion); `block_sub_diffusion`, `diffusion_sub_block`

## 4. Test fixtures
- flag: MongoDB/voyage-4-nano, NovaSearch/stella_en_1.5B_v5, sdadas/stella-pl-retrieval-8k (missed today),
  Alibaba-NLP/gte-Qwen2-1.5B-instruct, gte-Qwen2-7B-instruct (already diffusion via is_causal false) -> diffusion
- not flag: GritLM/GritLM-7B, -7B-KTO, -8x7B, -8x7B-KTO (code default is_causal True), google/gemma-3-1b-it
  (use_bidirectional_attention false), Qwen/Qwen2-0.5B, incoai/GLM-5.3-DFlash2 (a drafter with is_causal false,
  skipped)

## Verdict
real: 3 points (voyage-4-nano, 2 stella) become gen diffusion: edges INTO them from AR hosts disappear (fewer edges,
3 targets); as hosts they gain AR targets of their capacity (more edges). Region: attention.
