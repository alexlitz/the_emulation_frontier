# bias_qkv_table_wrong
live 1 / placed 1 (gpt_neox_japanese 1); direction both; batch a1_typetables

## 1. What the code computes
- tf5:models/gpt_neox_japanese/modeling_gpt_neox_japanese.py:169 `query_key_value = Linear(hidden, 3 * hidden, bias=False)`; :170 `dense = Linear(hidden, hidden, bias=False)`.
- :172-173 `dense_bias = Parameter(zeros(hidden)) if use_bias else None`; :332 `use_bias = (layer_number == num_hidden_layers - 1)`: only the LAST layer's attention has an output bias, added after the residual dropout (:363, :223 returns it). An output-side bias: bias_out stays.
- MLP: :312, :314 dense_h_to_4h / dense_4h_to_h `bias=False`; the norms are nn.LayerNorm (beta): bias_out stays.
- p5 lists gpt_neox_japanese in BIAS_QKV_TYPES (p5:model_graph.py:1948) and BIAS_OUT_TYPES (:1955), so abeja/gpt-neox-japanese-2.7b gets bias_qkv + bias_out (classify: `bias_out+bias_qkv+tied_emb`). The bias_qkv part is wrong.

## 2. Classifier change
- reads: model_type
- new: remove gpt_neox_japanese from BIAS_QKV_TYPES (feature bias_qkv off); keep it in BIAS_OUT_TYPES
- detection:
```python
def detect_bias_qkv_table_wrong(cfg, model_type):
    """True = the type's code has NO q / k / v bias although BIAS_QKV_TYPES lists it."""
    return True if model_type == "gpt_neox_japanese" else None
```
- apply note: p5:model_graph.py:1948, delete "gpt_neox_japanese" from the BIAS_QKV_TYPES literal. Also delete it from `_GATE_BIAS_TYPES` (:2985, the FFN gate / up bias used by the act-width shifts): dense_h_to_4h is bias=False (:312). It is not in QBIAS_TYPES (:3816), correctly.

## 3. Relation rule
- target side: as a target it no longer needs a q / k / v-biased host: bias-free (still bias_out-carrying, LayerNorm) hosts become admissible
- host side: as a host it can no longer play bias_qkv targets (it has no q / k / v bias to set)
- kind: exact (biased host -> it, via `bias_off`); the removed host edges were wrong (no construction)
- cost: none
- free by the config rules: n/a (code-fixed, no key)
- theorem: `bias_off` (biased hosts still play it); `biased_proj_not_bias_free` (why it cannot host bias_qkv targets)

## 4. Test fixtures
- flag: abeja/gpt-neox-japanese-2.7b -> true (qkv bias=False in code)
- not flag: EleutherAI/pythia-160m (gpt_neox: query_key_value bias=True, correctly listed); rinna/japanese-gpt-neox-small (a gpt_neox model, not gpt_neox_japanese)

## Verdict
real, small. One model: it loses its bias_qkv targets as a host (GPT-2 / OPT / Bloom / Pythia-sized biased models it was credited with) and gains bias-free LayerNorm hosts with output biases as a target; on the order of tens of edges each way.
