# not_a_causal_lm
live 3 / placed 3 (minicpm 1, cost_wise_gemma 1, sundial 1); direction over-credit; batch a3_placement

## 1. What the code computes
- BAAI/bge-reranker-v2-minicpm-layerwise/modeling_minicpm_reranker.py:1173-1185: for head_type not in
  {raw, complex} (config head_type "simple"; class default "simple", configuration_minicpm_reranker.py:144) the
  `lm_head` is `ModuleList([LayerWiseHead(hidden, 1)] for start_layer..num_hidden_layers)`; LayerWiseHead (:1134-1142)
  is `Linear(hidden, 1, bias=False)`. Forward (:1323) emits one relevance score per token per exit layer, not vocab
  logits.
- BAAI/bge-reranker-v2.5-gemma2-lightweight/gemma_model.py:504-511: `layer_wise` true (the snapshot) builds
  `ModuleList([CostWiseHead(hidden, 1)] for start_layer..L step layer_sep)` (CostWiseHead :485-493 =
  Linear(hidden, 1)); :589-600 picks cutoff layers. Output: a scalar score per exit layer.
- thuml/sundial-base-128m/modeling_sundial.py:28-45 SundialPatchEmbedding: no token table, input is a float series cut
  into 16-value patches (+ mask) through a SiLU MLP; :403 the head is `FlowLoss(output_token_lens[-1]=720, hidden, ...)`;
  flow_loss.py:40-49 `sample`: 50 Euler steps from Gaussian noise. The "logits" are stochastic continuous samples.
- (kuleshov-group/bd3lm-owt-block_size4, the 4th carrier, is already in SKIP_TYPES "bd3lm": p5 line 657.)
- p5 does not fix it: no score-head / forecaster check (p5:model_graph.py skipped() :668, SKIP_TYPES :646).

## 2. Classifier change
- reads: model_type minicpm + head_type (not raw / complex) or arch LayerWiseMiniCPMForCausalLM; model_type
  cost_wise_gemma + layer_wise; model_type sundial / arch SundialForPrediction
- new: skipped() reason "not_lm" (values score_head / continuous_io), in skipped
- detection:
```python
def detect_not_a_causal_lm(cfg, model_type):
    lm = lm_of(cfg)
    a = set(archs(cfg))
    if model_type == "minicpm" and ("head_type" in lm or "LayerWiseMiniCPMForCausalLM" in a):
        if str(lm.get("head_type", "simple")).lower() not in ("raw", "complex"):
            return "score_head"
    if model_type == "cost_wise_gemma" and truthy_flag(lm.get("layer_wise")):
        return "score_head"
    if model_type == "sundial" or "SundialForPrediction" in a:
        return "continuous_io"
    return None
```
- apply note: in p5:model_graph.py `skipped(c)` (line 668), after the SKIP_TYPES test add
  `if _not_lm(c, d): return "not_lm"` with the three tests above (model_type from d, architectures from c and d).
  Add "sundial" to SKIP_TYPES instead if a type-only rule is preferred (the reranker tests need the config keys).

## 3. Relation rule
- target side: none (dropped from the zoo: no next-token logits to reproduce)
- host side: none (dropped; a score head cannot emit a vocabulary distribution)
- kind: wall (removal)
- cost: none
- free by the config rules: head_type / layer_wise select a different head module (a feature switch, not free)
- theorem: none needed (classification: the model is not a causal LM)

## 4. Test fixtures
- flag: BAAI/bge-reranker-v2-minicpm-layerwise -> score_head (head_type simple);
  BAAI/bge-reranker-v2.5-gemma2-lightweight -> score_head (layer_wise true);
  thuml/sundial-base-128m -> continuous_io
- not flag: BAAI/bge-reranker-v2-gemma (a gemma reranker that reads a real vocab logit), openbmb/MiniCPM4-8B
  (minicpm, no head_type), google/gemma-2-2b (parent family of cost_wise_gemma),
  kuleshov-group/bd3lm-owt-block_size4 (already skipped by type), XinyueWangg/TimeBraid-2.5B (time-series wrapper
  around a real Qwen3 LM)

## Verdict
real: removes 3 placed points; edges into / out of them disappear (the minicpm reranker and the 3584-wide
cost_wise_gemma were capacity hosts for small dense targets, so a few frontier members may change). Region:
classification.
