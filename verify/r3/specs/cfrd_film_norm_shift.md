# cfrd_film_norm_shift
live 3 / placed 3 (cfrd 3); direction over-credit; batch a12_extra_inputs

## 1. What the code computes
- `phase_rms_norm(x, norm, scale, shift) = norm(x) * (1 + 0.1 tanh(scale)) + 0.1 shift` (alice-noa-chan/haru_1.1/model.py:185-198; haru/model.py:183-196; haru_2/model.py:195-208). `norm` = weighted RMSNorm (:137-147).
- Parameters per recurrence depth d: `attn_phase_scale / attn_phase_shift / ffn_phase_scale / ffn_phase_shift = zeros(recurrences, d_model)` (haru_1.1/model.py:605-608), trained (zero init only).
- Used on both sublayer inputs of every FoldedCell step: `u = phase_rms_norm(x, attn_norm, scale[d], shift[d])` -> local attention (q / k / v), `local_context = u + local_update` -> summary memory, `attn_gate(u)` (:530-547); `v = phase_rms_norm(x, ffn_norm, ...)` -> SwiGLU w1 / w3 and `ffn_gate(v)` (:550-558). HF entry `modeling_cfrd.py:13` imports `CFRDLanguageModel` from model.py.
- So the shift is a per-depth beta on the norm: a constant into every projection reading it (q / k / v biases, MLP-in biases, memory projections); the scale is a per-depth norm gain.
- BindingBlock (haru_1.1 / haru_2, `use_binding_block`) uses plain RMSNorm (no shift) (:466-491).
- p5 gives cfrd feats `''` / `tied_emb`: no bias_qkv / bias_out.

## 2. Classifier change
- reads: model_type cfrd (code-fixed)
- new: FEATURES `bias_qkv` + `bias_out` via BIAS_QKV_TYPES / BIAS_OUT_TYPES, in `_feats`
- detection:
```python
def detect_cfrd_film_norm_shift(cfg, model_type):
    return "bias_qkv+bias_out" if model_type == "cfrd" else None
```
- apply note: add "cfrd" to `BIAS_QKV_TYPES` (p5:model_graph.py:1947) and `BIAS_OUT_TYPES` (:1955). The `_feats` bias_qkv test reads `use_qkv_bias` / `qkv_bias` defaults True -- cfrd configs carry neither, so it fires. dims `q_bias` (`_q_bias`, :2964) should count cfrd as biased if the start-token / bias levers read it (QBIAS_TYPES :3816): add "cfrd" there too.

## 3. Relation rule
- target side: a host with q / k / v and MLP-input biases (a beta-carrying norm folds into them); a bias-free host is a wall (q / k / v: proved; the MLP side: open).
- host side: free (zero the biases).
- kind: target wall; host exact.
- cost: none.
- free by the config rules: none (a parameter, no config switch). The per-depth scale is a norm gain: free for an untied host, but per-iteration on a shared cell (haru 2 cells / 6 recurrences, haru_1.1 3 / 6; haru_2 6 / 6 is unshared) -- a looped host with one norm per cell cannot carry it; that belongs to cfrd's loop classification (p5 gives cfrd `untied`), not here.
- theorem: target `biased_proj_not_bias_free` (PROOFS.md "bias_qkv target <- bias-free host", PROVED impossible), bias_out target OPEN; host `bias_off`.

## 4. Test fixtures
- flag: alice-noa-chan/haru, alice-noa-chan/haru_1.1, alice-noa-chan/haru_2 -> "bias_qkv+bias_out".
- not flag: osunlp/ARLoop-0.6B (another looped cell model, no shift), Qwen/Qwen3-0.6B, meta-llama/Llama-3.1-8B-Instruct.
- scan: exactly the 3 cfrd repos.

## Verdict
real (assuming trained non-zero shifts; weights not inspected). Fewer edges: the 3 cfrd targets lose every bias-free host (most of the graph); host side unchanged.
