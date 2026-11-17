# remote_code_not_config_family
live 2 / placed 2 (glm5_next_text 1, quillan_ronin_oni 1); direction both; batch a3_placement

## 1. What the code computes
- shashanth-27xA9/GLM-5.3-SuperExpert-v2-9.2GB: outer model_type glm_superexpert
  (configuration_glm_superexpert.py:9) whose `get_text_config` returns self (:50-51): the code reads the TOP-level
  class defaults, while classify unwraps text_config (a GLM-5.3 glm5_next_text: hybrid KDA / MLA / mHC, 288 experts).
  modeling_glm_superexpert.py: GLMSuperExpertAttention (:159-181) plain softmax(q k^T / sqrt(d) + mask) v, MHA, no
  RoPE, no causal mask built anywhere (the caller's mask is added raw); GLMSuperExpertMoE (:114-134)
  `F.softmax(routing_logits)` over ALL experts, every expert evaluated (dense soft MoE) + shared; GLMDenseMLP /
  GLMSuperExpertSingleMLP (:102, :147) unclamped SiLU-GLU (no swiglu_limit). Packed / dequantised weights
  (:23, :61) that do not match a GLM-5.3 checkpoint layout.
- CrashOverrideX/Quillan-Ronin: modeling_quillan_oni.py:11-27 wraps `NativeQuillanRoninOni(config.to_core_config())`
  from a native module (quillan_v5_4_oni): affine LayerNorm blocks, biased fused c_attn, dense SwiGLU PLUS a
  tanh-gated soft mixture of all 34 experts, recirculation / ingest gates, BitLinear layers. Config
  `weights_status: checkpoint_pending` (no released weights); router_mode dense_pull.
- p5 classifies both from the config keys (glm5_next_text slots; a plain RMSNorm top-k MoE for Quillan); no
  SKIP entry for either type (p5:model_graph.py SKIP_TYPES :646).

## 2. Classifier change
- reads: outer / LM model_type in {glm_superexpert, quillan_ronin_oni}
- new: SKIP_TYPES += {"glm_superexpert", "quillan_ronin_oni"} (skipped() reason non_decoder; or a new reason
  "remote_bespoke"), in SKIP_TYPES
- detection:
```python
REMOTE_REPLACES_FAMILY = {"quillan_ronin_oni", "glm_superexpert"}
def detect_remote_code_not_config_family(cfg, model_type):
    if outer_type(cfg) in REMOTE_REPLACES_FAMILY or model_type in REMOTE_REPLACES_FAMILY:
        return "skip"
    return None
```
- apply note: p5:model_graph.py SKIP_TYPES (:646) add both types; skipped() (:668) already tests the outer and the
  LM type, so the GLM wrapper (outer glm_superexpert, LM glm5_next_text) is caught by its outer type.

## 3. Relation rule
- target side: none (dropped)
- host side: none (dropped)
- kind: wall (removal). Classifying them from their own code instead (GLM-SuperExpert: nope, non-causal gen,
  dense-softmax MoE with no top-k; Quillan: LayerNorm + soft all-expert mixture + BitLinear) would give points no
  real checkpoint backs (unloadable / pending weights), so skip.
- cost: none
- free by the config rules: n/a
- theorem: none needed (classification)

## 4. Test fixtures
- flag: CrashOverrideX/Quillan-Ronin -> skip; shashanth-27xA9/GLM-5.3-SuperExpert-v2-9.2GB -> skip
- not flag: dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16, local-inference-lab/GLM-5.3-Flash-NVFP4-Spark,
  defnoteatin/PseudoBrain-GLM-5.3-Flash-Chronos-v7.5 (glm5_next_text with the stock family's code)

## Verdict
real: removes 2 placed points (fewer edges; the GLM one is a large glm5_next point, likely only a duplicate of other
GLM-5.3 points, so the frontier barely moves). A general "auto_map replaces the family" rule is not decidable from
the config; keep it a per-type list. Region: classification.
