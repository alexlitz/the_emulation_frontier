# prefix_lm_mask
live 1 / placed 6 (mpt 5, gemma 1); direction over-credit; batch a6_masks

Really live: 1 (JayRay5/DIVE-Doc-FRD). All 5 MPT snapshots save attn_config.prefix_lm false or absent, so MPT's
`_apply_prefix_mask` never runs (anas-awadalla/mpt-7b/modeling_mpt.py, MPTModel._apply_prefix_mask).

## 1. What the code computes
- DIVE-Doc wraps a Gemma text model (text_config model_type gemma, 18 layers) behind a PaliGemma-style
  `_update_causal_mask` (JayRay5/DIVE-Doc-FRD/modeling_divedoc.py:221-300).
- Outside training (`is_training = token_type_ids is not None and labels is not None`, :386) with sequence_length != 1:
  `causal_mask[:, :sequence_length] = 0.0` (:273-277). Then `causal_mask *= arange(target) > cache_position` (:279)
  masks only cache slots beyond the prefill. Every prefill token attends to every other prefill token, later ones
  included.
- Decode steps (sequence_length == 1) are causal over the cache. Net effect: a prefix-LM whose prefix is the whole
  prompt, causal for the generated tokens. A teacher-forced forward over a sequence is fully bidirectional.
- A caller-supplied 4D mask is returned as is (:261-263), so the model takes any mask (the GEN_ORDER mask currency).
- The mask is passed to the Gemma LM (:438-442). classify reads only text_config, so it gives gen=ar (`_gen` ~L2373).

## 2. Classifier change
- reads: outer model_type DIVEdoc (or paligemma) / architectures DIVEdoc / PaliGemmaForConditionalGeneration; MPT
  attn_config.prefix_lm
- new: gen value "prefix", in classify (`_gen`, given the outer config) + GEN_ORDER entries
- detection:
```python
def detect_prefix_lm_mask(cfg, model_type):
    lm = lm_of(cfg)
    if outer_type(cfg) in ("divedoc", "paligemma") or any(a in ("DIVEdoc", "PaliGemmaForConditionalGeneration")
                                                          for a in archs(cfg)):
        return "prefix"
    if model_type == "mpt" and truthy_flag((lm.get("attn_config") or {}).get("prefix_lm")):
        return "prefix"
    return None
```
- apply note:
  - `_gen(d, mt)` (p5:model_graph.py ~L2373) needs the outer config: call it from classify as `_gen(d, mt, outer)` and
    return "prefix" for the wrappers above. For mpt, `_normalize` copies attn_config.prefix_lm to a flat `prefix_lm`.
  - GEN_ORDER (~L379), all edges in the mask currency: "ar": + ("prefix", "mask", True); "prefix": [("diffusion",
    "mask", True), ("block", "mask", True)]; "diffusion" and "block": + ("prefix", "mask", True).
  - TOP (~L406) is unchanged. `reach()` (~L1484) is a fixpoint, so the prefix <-> diffusion / block cycle is fine; the
    existing block <-> diffusion cycle already works that way.

## 3. Relation rule
- target side: needs a host that attends bidirectionally over the prompt: a diffusion / block host, or a prefix host,
  given the mask. An AR host cannot: position 0's output reads position 1 (`causal_ne_bidirectional`).
- host side: a prefix host takes a caller 4D mask (:261-263), so it plays causal (ar), block and diffusion targets under
  the mask currency
- kind: wall (AR host); exact (mask currency) for prefix <-> diffusion / block and ar -> prefix
- cost: none (the mask currency, as GEN_ORDER's existing edges)
- free by the config rules: n/a (the mask is code + caller input)
- theorem: `causal_ne_bidirectional` (wall); `pattern_sub_full`, `block_sub_diffusion`, `block_ge_full`,
  `causal_stack` (mask edges)

## 4. Test fixtures
- flag: JayRay5/DIVE-Doc-FRD -> "prefix"
- not flag: anas-awadalla/mpt-7b, explosion-testing/mpt-test, replit/replit-code-v1-3b, vinai/PhoGPT-4B,
  vinai/PhoGPT-4B-Chat (prefix_lm false); google/gemma-2b (plain gemma, causal); google/gemma-3-1b-it

## Verdict
Real, 1 model.
- As a target, DIVE-Doc-FRD loses every AR host (most of the gemma-compatible column: likely tens to a few hundred
  edges) and keeps only diffusion / block-class hosts.
- As a host it gains the diffusion / block targets its other slots allow (few).
Net: fewer edges. Region: classification (gen slot).
