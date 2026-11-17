# out_gate_activation
live 7 / placed 16 (k2_horizon 10, laguna 6); direction over-credit; batch a7_attn_exotic

## 1. What the code computes
- tf5:models/laguna/modular_laguna.py:335 `gate = softplus(g_proj(x))`, `o * gate` (per head :336-338, else per
  element) before o_proj; poolside/Laguna-XS-2.1/modeling_laguna.py:458-459 (same, skipped when gating False);
  poolside/Laguna-XS.2/modeling_laguna.py:412 (same, always per head). AFMoE (the parent) and the FEATURES out_gate rule
  (p5:model_graph.py:1602-1605) assume `o * sigmoid(W_g x)`.
- inception42/Jais-3-MoVA-36B-A4B/modeling_k2_horizon.py:225-230: gate_proj `hidden -> heads*head_dim` iff
  `attention_gate_func` is not None; :312-321 `silu(g)` or `softplus(g, beta=ln 2)`, elementwise. Same in
  IFM/Jais-ASR/modeling_jais_asr.py:719-721, 832-837. Only Jais-3-MoVA sets "softplus"; the other 9 k2_horizon
  configs have attention_gate_func None (no gate: not live).
- softplus_beta(z) = softplus(beta z) / beta: K2's gate is Laguna's with W_g scaled by ln 2 and o_proj by 1/ln 2 -- one
  class, "softplus". softplus(0) = ln 2 (K2: 1), sigmoid(0) = 1/2, silu(0) = 0.
- JuneoAI/Laplace-1-Vector/modeling_spark.py:199-204 allows gate_attn_act_mode "silu" (every snapshot: sigmoid).
- sigmoid is bounded in (0, 1), softplus unbounded in (0, inf), silu signed: no fixed weight setting turns one
  data-dependent gate into another (each is a nonlinear function of W_g x; a ratio of two is not linear in x).

## 2. Classifier change
- reads: model_type laguna (gating), k2_horizon (attention_gate_func), spark2_5 (gate_attn_act_mode)
- new: an activation suffix on the out_gate FEATURES key: `out_gate_softplus`, `out_gate_silu` (+ `_head` from
  out_gate_headwise: Laguna XS / S `out_gate_head_softplus`, Laguna-M.1 `out_gate_softplus`), in _feats + FEATURES
- detection:
```python
def detect_out_gate_activation(cfg, model_type):
    lm = lm_of(cfg)
    mt = model_type
    if mt == "laguna":
        return "softplus" if lm.get("gating", True) is not False else None
    if mt == "k2_horizon":
        f = _s(lm.get("attention_gate_func"))
        return f if f in ("softplus", "silu") else None
    if mt == "spark2_5" and lm.get("headwise_attn_output_gate") is True:
        return "silu" if _s(lm.get("gate_attn_act_mode")) == "silu" else None
    return None
```
- apply note: p5:model_graph.py `_feats` out_gate block (~1747-1755): the key is
  "out_gate" + ("_head" if head-wise) + ("_" + act if act != sigmoid); remove "laguna" from OUT_GATE_TYPES (1977)
  so gating False gives no gate. FEATURES (after "out_gate", 1602): `out_gate_softplus` ("wall", "free"),
  `out_gate_head_softplus` ("wall", "free"), `out_gate_silu` ("wall", "wall"); FEAT_NEST (see out_gate_headwise):
  "out_gate_head_softplus": ("out_gate_softplus",). No nesting across activations.

## 3. Relation rule
- target side: a softplus- (silu-) gated target needs a host gate of the same activation (granularity per
  out_gate_headwise); sigmoid / ungated hosts are a wall.
- host side: a softplus host plays an ungated target (W_g = 0: gate ln 2, o_proj / ln 2) -- free; a silu host
  cannot (silu(0) = 0, no bias in gate_proj): wall; neither plays a sigmoid-gated target (wall), nor conversely.
- kind: exact (same activation; softplus host -> ungated target); wall (across activations; silu host -> ungated)
- cost: none
- free by the config rules: none; the gate activation is an activation-function choice, which the rules say is not
  free (softplus beta is a scale and folds: K2 = Laguna class)
- theorem: `out_gate_off` (the sigmoid row; the softplus case is the same fold with constant ln 2) | needs theorem:
  softplus_gate_off (W_g = 0 gives ln 2, rescale o_proj) and gate_act_ne (sigmoid / softplus gates differ for some x)

## 4. Test fixtures
- flag: poolside/Laguna-XS.2, poolside/Laguna-XS-2.1, poolside/Laguna-S-2.1, poolside/Laguna-M.1,
  TensorFold/Laguna-S-2.1-MLX-6bit, mlx-community/Laguna-S-2.1-oQ4e, inception42/Jais-3-MoVA-36B-A4B -> "softplus"
- not flag: IFM/K2-Horizon-7B, inception42/Jais-3-32B, IFM/Jais-ASR (attention_gate_func None: no gate),
  arcee-ai/Trinity-Mini (AFMoE, the sigmoid parent), Qwen/Qwen3-Next-80B-A3B-Instruct, JuneoAI/Laplace-1-Vector
  (sigmoid), stepfun-ai/Step-3.5-Flash (sigmoid head-wise)

## Verdict
real. Fewer edges: the 7 softplus models (6 Laguna, Jais-3-MoVA) lose every sigmoid-gated host (~330 Qwen3.5 / Next /
AFMoE / Step-3.5 ...) and stop hosting sigmoid-gated targets; softplus hosts keep ungated targets. Composes with
out_gate_headwise: Laguna XS / S are head-wise softplus, Laguna-M.1 and Jais-MoVA elementwise softplus (M.1 hosts XS / S).
