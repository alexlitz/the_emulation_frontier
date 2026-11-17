# diff_attn
live 1 / placed 1 (Motif 1); direction both; batch a7_attn_exotic

## 1. What the code computes
- Motif-Technologies/Motif-3/modeling_motif.py:1053-1062: the decoder layer picks `attention_cls` ("gdla" in the
  snapshot) -> MotifGDLAttention. :555-557 `grouped_ratio = (H - num_noise_heads) // num_noise_heads` (80, 16 -> 4),
  `n_signal_heads = grouped_ratio * num_noise_heads` (64).
- :619 `lambda_proj = Linear(hidden, n_signal_heads, bias=False)` (diff_v2 True); :687 `lambda_full = lambda_proj(x)`.
- :754-766: heads rearranged `(g gs)` with gs = grouped_ratio + 1; signal heads a_s, the group's noise head a_n
  broadcast by repeat_kv (:504, repeat_interleave on dim 2, :763); `out_s = a_s - sigmoid(lambda_s(x)) * a_n`.
- :600-607, 655-656, 768-769: `wq_b_gate(q_latent)` elementwise gate (elementwise_attn_output_gate true in the snapshot)
  `out * sigmoid(gate)`; :773 `wo` over the 64 signal heads only.
- 80 heads on 16 kv heads = 5 q heads per kv head = one (4 signal + 1 noise) group per kv head: the noise head reads the
  same K / V as its signals, so its value cannot be zeroed separately; sigmoid(lambda) > 0 for every input (lambda
  has no bias and reads the RMS-normed x: no constant to drive it to -inf).
- p5 _feats (p5:model_graph.py:1747-1755) does not read elementwise_attn_output_gate: Motif-3 has no out_gate today
  (folded into out_gate_headwise's apply note).

## 2. Classifier change
- reads: model_type motif (lower-cased "Motif"), attention_cls, num_noise_heads
- new: FEATURES key `diff_attn` ("wall", "wall"), in _feats + FEATURES
- detection:
```python
def detect_diff_attn(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "motif" and _s(lm.get("attention_cls", "basic")) == "gdla" and num(lm.get("num_noise_heads")) > 0:
        return True
    return None
```
- apply note: p5:model_graph.py `_feats` (after the out_gate block): `if mt == "motif" and d.get("attention_cls") ==
  "gdla" and _num(d.get("num_noise_heads")) > 0: f.add("diff_attn")`; FEATURES: add "diff_attn".

## 3. Relation rule
- target side: a differential target subtracts a data-weighted second head (a product sigmoid(lambda(x)) * a_n, like an
  output gate) -- no plain head forms it: wall.
- host side: a differential host cannot drop the noise term (sigmoid never reaches 0; the noise head shares the signal
  heads' K / V; setting the noise query equal to a signal query subtracts half of that head from all four signals):
  wall. sigmoid(lambda) -> 0 is reachable only as a limit with a constant channel the host does not have.
- kind: wall (both directions)
- cost: none
- free by the config rules: none (attention_cls / num_noise_heads are feature switches)
- theorem: needs theorem: diff_attn_ne_plain (a_s - sigma(w.x) a_n is not a softmax head for generic w; analogue of
  the OPEN out_gate target wall)

## 4. Test fixtures
- flag: Motif-Technologies/Motif-3 -> True
- not flag: deepseek-ai/DeepSeek-V3 (MLA without the differential split), Qwen/Qwen3-8B (plain)

## Verdict
real but no frontier change today: Motif-3 is already walled both ways by act=poly_norm and hyper_conn
(mhc_enabled). Recorded so Motif-3 stays isolated if those walls move.
