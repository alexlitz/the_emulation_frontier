# ngpt_sphere_residual
live 4 / placed 4 (dory 4); direction over-credit; batch a4_norms

## 1. What the code computes
(ext:dory/transformers-pr50099/modeling_dory.py)
- `DoryL2Norm`: `x * rsqrt(x.pow(2).sum(-1) + eps)`, eps 1e-6, no weight: unit L2 norm, not unit RMS (:50-61).
- DoryDecoderLayer: all four norms are DoryL2Norm (:451-454); per-channel rates `lambda_1`, `lambda_2` (init initializer_range) with `lambda_scale = ngpt_alpha_init / initializer_range` (:456-458).
- forward: `h = input_layernorm(h)` -- the residual itself is normalised; `a = post_attention_layernorm(attn(h))`; `h = h + |lambda_1 s| * (a - h)`; `h = pre_feedforward_layernorm(h)`; `m = post_feedforward_layernorm(mlp(h))`; `h = h + |lambda_2 s| * (m - h)` (:468-483). The stream is an interpolation between unit vectors, renormalised before each sublayer: h <- L2(h) + r ⊙ (L2(f) - L2(h)).
- Model: the loop section restarts `h = self.norm(h) + input_hidden_states` each loop (:617-620), final `self.norm` L2 (:636). The lm_head has a per-vocab learned logit scale (:488-497).
- p5: block sequential, norm RMSNorm, feats from other keys (the post-norms of the Gemma-2-like layer are not flagged either); the looped-host note already lists "Dory nGPT" as a wall for an untied target (p5:model_graph.py:3829-3833), but only on the loop path.

## 2. Classifier change
- reads: model_type dory (code fact; ngpt_alpha_init / initializer_range are scales).
- new: FEATURES key `ngpt` ("wall", "wall"), set in `_feats` for NGPT_TYPES = {"dory"}.
- detection:
```python
def detect_ngpt_sphere_residual(cfg, model_type):
    if model_type == "dory":
        return "ngpt"
    return None
```
- apply note: p5:model_graph.py FEATURES (next to "post_ln", ~1690); `_feats` beside the post_ln test (~1824): `if mt in NGPT_TYPES: f.add("ngpt")`. Do not also add post_norm / post_ln (ngpt subsumes them; a separate key keeps Dory from matching sandwich or post-LN hosts).

## 3. Relation rule
- target side: an nGPT target needs an nGPT host: no additive pre-norm (or sandwich, or post-LN) stack keeps its stream on the unit sphere with per-channel interpolation rates (r ⊙ (a - h) with r per channel, then renormalised).
- host side: a Dory host cannot leave the sphere: every sublayer input and the residual are L2-normalised (weight-free), so it plays no additive target. The rate lambda -> 0 freezes the stream, it does not give h + f.
- kind: wall both ways.
- cost: none.
- free by the config rules: ngpt_alpha_init / initializer_range only scale the rates (free); the sphere residual is code, not free.
- theorem: needs theorem `ngpt_not_additive`: for any additive pre-norm layer x -> x + f(N x) there are inputs on which the nGPT layer's output norm (pinned near 1) and the additive output's differ (and conversely). Related: `selfcond_norm_not_exact` (always-on weight-free norm, the same "cannot switch off" shape).

## 4. Test fixtures
- flag: ProCreations/Dory-Beam-4B, netprojects/nemotron-dory1.0-4b-mlx-8bit, nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8, vladimir94/dory-1.0-4b-ane-coreml -> "ngpt" (every live repo).
- not flag: BAAI/bge-reranker-v2.5-gemma2-lightweight (Gemma-2 sandwich: the layer Dory's docstring cites, additive); arcee-ai/Trinity-Large-Base (afmoe: sandwich, additive).

## Verdict
real. Fewer edges: the 4 Dory points lose every non-Dory partner both ways (today credited as plain RMSNorm sequential points; the loop wall already removes untied targets on Dory hosts, so the new loss is mainly Dory as a TARGET on plain / looped hosts and Dory hosting tied targets). Region: recurrent-other (residual stream).
