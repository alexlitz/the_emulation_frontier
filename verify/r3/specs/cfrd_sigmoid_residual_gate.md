# cfrd_sigmoid_residual_gate
live 3 / placed 3 (cfrd 3); direction over-credit; batch a12_extra_inputs

## 1. What the code computes
- FoldedCell: `attn_gate = Linear(d_model, 1, bias=True)`, `ffn_gate = Linear(d_model, 1, bias=True)` (alice-noa-chan/haru_1.1/model.py:508-514; init w = 0, b = residual_gate_init = -1, trained); `x = x + residual_scale * sigmoid(attn_gate(u)) * (local + sigmoid(g_d) memory)` (:546-547); `x = x + residual_scale * sigmoid(ffn_gate(v)) * ffn(v)` (:557-558); `residual_scale = 1/sqrt(recurrences)` (:619). Same in haru/model.py (:464, :475, :536) and haru_2/model.py (:566, :577, :645).
- BindingBlock (haru_1.1 / haru_2, `use_binding_block: true`): `x + (1/sqrt 2) sigmoid(w_a . u + b_a) attn(u)`, then the same with the SwiGLU (haru_1.1/model.py:470-491; haru_2/model.py:482-501).
- So each residual update is scaled by a data-dependent per-token scalar `c * sigmoid(w . u + b)` of the normed input: a residual gate on BOTH sublayers. Not out_gate (an attention-only elementwise / head-wise gate without a bias, p5:model_graph.py FEATURES "out_gate").
- p5: no feature for it (cfrd feats `''` / `tied_emb`).

## 2. Classifier change
- reads: model_type cfrd (code-fixed)
- new: FEATURES key `res_gate` = ("wall", "free"), in `FEATURES` + `_feats`
- detection:
```python
def detect_cfrd_sigmoid_residual_gate(cfg, model_type):
    return "res_gate" if model_type == "cfrd" else None
```
- apply note: add to `FEATURES` (p5:model_graph.py:1597, beside out_gate) `"res_gate": ("wall", "free", "a data-dependent scalar gate on each residual update, x + c sigmoid(w . u + b) f(u) (CFRD FoldedCell / BindingBlock): no ungated layer forms the product; a gated host sets w = 0, b = 0 (gate 1/2) and doubles the sublayer's output projection")`; in `_feats` add `RES_GATE_TYPES = {"cfrd"}` -> `f.add("res_gate")`.

## 3. Relation rule
- target side: needs a res_gate host. An out_gate host does not suffice: its gate covers attention only (not the FFN) and has no gate bias b.
- host side: free -- w = 0, b = 0 gives sigmoid = 1/2; c and the 2 fold into o_proj / down_proj (and the memory read_o), exact.
- kind: target wall; host exact.
- cost: none.
- free by the config rules: none (code-fixed; the gate init `residual_gate_init` is a trained-parameter init, not a forward switch).
- theorem: host side needs theorem `res_gate_off` (the `out_gate_off` construction, PROOFS.md "out_gate host -> ungated target": W_g = 0, sigma = 1/2, 2 W_o, applied per sublayer; the b = 0 setting is a weight). Target side OPEN (as "out_gate / shared_gate target <- ungated host").

## 4. Test fixtures
- flag: alice-noa-chan/haru, alice-noa-chan/haru_1.1, alice-noa-chan/haru_2 -> "res_gate".
- not flag: Qwen/Qwen3-Next-80B-A3B-Instruct (out_gate, not a residual gate), Cactus-Compute/needle3 (needle: `skip + sigmoid(attn_gate) * x` with a constant parameter gate, architecture.py:344-345, :364 -- data-independent, folds into out_proj), meta-llama/Llama-3.1-8B-Instruct.
- scan: exactly the 3 cfrd repos.

## Verdict
real. Fewer edges: the 3 cfrd targets lose every host (no other family carries res_gate); host side unchanged.
