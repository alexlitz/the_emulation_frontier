# bias_out_by_code_unlisted
live 4 / placed 4 (recurrent_gemma 2, jetmoe 1, nae1 1); direction over-credit; batch a1_typetables

## 1. What the code computes
- recurrent_gemma: tf5:models/recurrent_gemma/modeling_recurrent_gemma.py:226-228 q / k / v `bias=config.attention_bias` (false), :229 `o_proj = Linear(..., bias=True)` unconditionally; MLP :489-491 gate / up / down `bias=True`.
- jetmoe: tf5:models/jetmoe/modeling_jetmoe.py:235 `self.bias = Parameter(empty(input_size))` and :274 `layer_output = layer_output + self.bias` (MoE output); the MoA has its own, :294 / :347.
- nae1: ext nae1/hub__Subject-Emu-5259__NeuralAI-Nae1/model/nae1_model.py:203-204 `nn.LayerNorm(n_embd, eps)` (affine weight AND bias), :236 final_layernorm likewise; projections bias-free (:92-94, :118-121). p5 gives norm=LayerNorm but no bias_out, so the beta is dropped.
- p5 `_feats` (p5:model_graph.py:1797-1805) sets bias_out from BIAS_OUT_TYPES (:1955) or keys; none of the three types is listed or has a key (classify: recurrentgemma `rf_rglru+tied_emb`, jetmoe `tied_emb`, Nae1 `tied_emb`).

## 2. Classifier change
- reads: model_type
- new: table entries in BIAS_OUT_TYPES (feature bias_out)
- detection:
```python
BIAS_OUT_BY_CODE = {"recurrent_gemma", "jetmoe", "nae1"}

def detect_bias_out_by_code_unlisted(cfg, model_type):
    return True if model_type in BIAS_OUT_BY_CODE else None
```
- apply note: add the three to `BIAS_OUT_TYPES` (p5:model_graph.py:1955). For nae1 the ln_beta fold rule at :3061 (`cl["norm"] == "LayerNorm" and "bias_out" in feats ...`) then fires, as for any affine LayerNorm. Add recurrent_gemma to `_GATE_BIAS_TYPES` (:2985, the FFN gate / up bias that `_ffn_bias` reads): its gate / up projections are bias=True (:489-490).

## 3. Relation rule
- target side: a bias_out target needs a host with output-side biases (FEATURES bias_out: wall on the target side; OPEN in PROOFS.md, no construction)
- host side: a bias_out host plays bias-free targets by zeroing (free)
- kind: wall (target <- bias-free host, no currency); exact (host -> bias-free target)
- cost: none
- free by the config rules: no (mlp_bias / attention_bias are shape-changing; here there is no key)
- theorem: `bias_off` (PROOFS.md "bias_qkv / bias_out host -> unbiased target"); target side OPEN ("bias_out target <- bias-free host"); the LayerNorm beta fold: `layerNorm_eq_rms_center`, `affine_fold` (row "LayerNorm <= RMSNorm": beta needs consumer biases)

## 4. Test fixtures
- flag: google/recurrentgemma-2b, google/recurrentgemma-2b-it, jetmoe/jetmoe-8b, Subject-Emu-5259/NeuralAI-Nae1 -> true
- not flag: google/gemma-2-2b-it (Gemma parent, bias-free o_proj / MLP); ggml-org/stories15M_MOE (mixtral MoE, no output bias); Qwen/Qwen3-0.6B
- scan: exactly the 4 live repos.

## Verdict
real, small. Fewer edges: 4 targets lose every bias-free host (RecurrentGemma's rf_rglru already limits its hosts to RG-LRU models, so few edges move there; jetmoe-8b and Nae1 lose most of theirs -- Nae1 as a LayerNorm target was being played by RMSNorm hosts through the beta-less fold).
