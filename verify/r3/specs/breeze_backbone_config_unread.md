# breeze_backbone_config_unread
live 2 / placed 2 (breeze 2); direction both; batch a3_placement

## 1. What the code computes
- ext:breeze/gh__breezeblue-ai__breeze-tts/models/breeze_backbone_factory.py:38-60 `create_backbone`: backbone_model_type
  "breeze" -> the native Llama-style BreezeBackboneModel, anything else -> the adapter; :105-135 `create_from_config`:
  `llm_config = AutoConfig.for_model(**config.backbone_config)` (:129).
- breeze_backbone_factory.py:164-179 `_create_qwen3_layers`: Qwen3DecoderLayer (per-head q_norm / k_norm RMSNorm before
  RoPE), Qwen3RMSNorm(eps = backbone_config.rms_norm_eps = 1e-6), Qwen3RotaryEmbedding(backbone_config: theta 1e6, no
  scaling). Every snapshot config: backbone_model_type "qwen3" with a bundled backbone_config (Qwen3ForCausalLM,
  hidden 2048, 28 layers, 16 / 8 heads, head_dim 128).
- The top-level keys classify reads are Llama-style (rope_theta 5e5 with llama3 scaling factor 32, rms_norm_eps 1e-5,
  no qk-norm key) and are unused by the backbone layers. The lm_head is the wrapper's own
  `Linear(hidden, vocab_size + 1)` (ext:.../models/breeze.py:923), so the output side stays untied (top-level
  tie_word_embeddings false), whatever backbone_config.tie_word_embeddings says.
- p5 `_lm_unwrap` (p5:model_graph.py :2689) returns the flat top level (it has dims), so qknorm=no, eps 1e-5.

## 2. Classifier change
- reads: model_type breeze, backbone_model_type, backbone_config
- new: _lm_unwrap descends into backbone_config (model_type qwen3) keeping the wrapper's tie_word_embeddings
- detection:
```python
def detect_breeze_backbone_config_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "breeze":
        return None
    bt = str(lm.get("backbone_model_type") or "breeze").lower()
    bb = lm.get("backbone_config")
    if bt != "breeze" and isinstance(bb, dict):
        return str(bb.get("model_type") or bt).lower()
    return None
```
- apply note: p5:model_graph.py `_lm_unwrap` (:2689), first branch: if `d.get("backbone_model_type") not in (None,
  "breeze")` and `isinstance(d.get("backbone_config"), dict)` with LM dims, return
  `{**d["backbone_config"], "tie_word_embeddings": d.get("tie_word_embeddings", False)}` (model_type from the
  backbone: qwen3 -> QKNORM_ALWAYS-style qk-norm head via the qwen3 tables). Combine with text_encoder_input_embeds
  (the "breeze_text_in" feature must still be added from the OUTER config: pass outer to _feats as today).

## 3. Relation rule
- target side: a Qwen3 target (qknorm head, eps 1e-6) needs a qk-normed host (QKNORM_ORDER) -- moot while
  text_encoder_input_embeds walls Breeze as a target
- host side: a qk-norm head host plays head / plain targets by the existing qknorm rules (eps lever)
- kind: exact (existing rules)
- cost: none
- free by the config rules: rope_theta / scaling and eps are free; adding qk-norm is shape-changing (not free)
- theorem: `qknorm_logit`, `knorm_not_plain` (wall), `epsLever_bound` / `leverAttn_model_bound` (host side limit)

## 4. Test fixtures
- flag: BreezeBlue/Breeze-TTS-2 -> qwen3; drbaph/Breeze-TTS-2-comfyui -> qwen3
- not flag: Qwen/Qwen3-0.6B (the backbone family itself), kmhf/hf-moshika (another audio LM), parler-tts/parler-tts-mini-v1

## Verdict
real: the 2 Breeze points become qknorm=head: as hosts they lose plain-target exact edges (qk-norm host -> plain is
the eps-lever limit), as targets they need qk-norm hosts (but they are walled as targets by
text_encoder_input_embeds). Few edges either way (2 points). Region: classification + attention.
