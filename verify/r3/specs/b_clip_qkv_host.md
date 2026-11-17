# b_clip_qkv_host
live 3 / placed 3 (olmo 3: OLMo-1B-0724-hf, OLMo-7B-0724-hf, OLMo-7B-0424-hf, clip_qkv 8.0); direction over-wall; batch bd

## 1. What the code computes
- tf5 OLMo-1: `if self.config.clip_qkv is not None:` `query_states.clamp_(min=-clip_qkv, max=clip_qkv)`, same for key / value (tf5:models/olmo/modeling_olmo.py:245-248). `None` = no clamp.
- tf5 OLMoE: the same `if self.config.clip_qkv is not None` clamp (tf5:models/olmoe/modeling_olmoe.py:253-256).
- tf5 MPT: `self.clip_qkv = config.attn_config.clip_qkv`; `if self.clip_qkv: mixed_qkv = mixed_qkv.clamp(...)` (tf5:models/mpt/modeling_mpt.py:81, :97-98). tf5 DBRX: `min_val = -self.clip_qkv if self.clip_qkv is not None else None`; `qkv_states.clamp(min=min_val, max=self.clip_qkv)` (tf5:models/dbrx/modeling_dbrx.py:191, :215-216): clamp(None, None) is refused by torch, but DBRX never builds it with None in the snapshot (no dbrx config carries a clip).
- The clamp has no parameters: switching it off changes no shape.
- p5 relation: `FEATURES["clip_qkv"] = ("wall", "wall", ...)` (p5:model_graph.py:1701-1703), set by `d.get("clip_qkv")` (:1824-1825); MPT's `attn_config.clip_qkv` is lifted by `_normalize` (:2584-2585).
- Snapshot: only the three OLMo-1 0424 / 0724 base checkpoints set it (8.0); OLMo-7B-0724-Instruct-hf, OLMo-1B-hf, every OLMoE / MPT / Molmo / FlexOlmo config has None.

## 2. Classifier change
- reads: `clip_qkv` (LM config) or `attn_config.clip_qkv`, > 0.
- new: FEATURES["clip_qkv"] host side "wall" -> "free".
- detection:
```python
def detect_b_clip_qkv_host(cfg, model_type):
    lm = lm_of(cfg)
    if (fnum(lm.get("clip_qkv"), 0.0) or 0.0) > 0:
        return "clip_qkv"
    ac = lm.get("attn_config")
    if isinstance(ac, dict) and (fnum(ac.get("clip_qkv"), 0.0) or 0.0) > 0:
        return "attn_config.clip_qkv"
    return None
```
- apply note: p5:model_graph.py FEATURES :1701 `("wall", "wall", ...)` -> `("wall", "free", "... a clamped host sets clip_qkv None (off)")`. Nothing else: the Z3 model's frule reads FEATURES.

## 3. Relation rule
- target side: unchanged wall (`clip_qkv_plain_fails`: a plain host's logit difference is affine in the query input, the target's a scaled clamp).
- host side: a clamped host sets `clip_qkv = None` and is the unclamped family.
- kind: exact (the code skips the clamp: identity on every input; `clampSym_id` for any c above the activations).
- cost: none.
- free by the config rules: free -- a cap value, shape-preserving, not a feature switch (no module or parameter depends on it).
- theorem: `clampSym_id`, `clip_not_exact` (PROOFS.md Round-2 rules, "clip_qkv: clamped host -> unclamped target", "wall NOT TIGHT"); with None the clamp is not run, so no regime argument is needed.

## 4. Test fixtures
- flag: allenai/OLMo-1B-0724-hf, allenai/OLMo-7B-0724-hf, allenai/OLMo-7B-0424-hf -> "clip_qkv" (8.0).
- not flag: allenai/OLMo-1B-hf, allenai/OLMo-7B-0724-Instruct-hf (same type, clip None); allenai/OLMoE-1B-7B-0924 (OLMoE code reads it, None); replit/replit-code-v1-3b, anas-awadalla/mpt-7b (MPT attn_config.clip_qkv None); allenai/Molmo-7B-D-0924, allenai/Flex-reddit-2x7B-1T (key present, None).

## Verdict
real (over-wall), exact, host side free: the 3 clamped OLMo-1 checkpoints become hosts for the unclamped targets
their other slots fit (OLMo-1 / non-parametric-LayerNorm families). Measured (931 real shapes, host side flipped): exact +203 shape
pairs, payable +206 (314 target repos): OLMo-7B-0424 / 0724-hf host 314 target repos, OLMo-1B-0724-hf 83.
