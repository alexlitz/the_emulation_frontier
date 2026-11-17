# mova_routed_values
live 1 / placed 9 (k2_horizon 9); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
(paths: inception42/Jais-3-MoVA-36B-A4B/modeling_k2_horizon.py, snapshot 44dd2b3d; the IFM/K2-Horizon-7B copy is identical at these lines)
- `K2HorizonDecoderLayer.__init__`: `is_sparse_layer = layer not in mlp_only_layers and num_experts > 0 and (i+1) % decoder_sparse_step == 0`. On sparse layers with `config.mova_num_experts > 0` the attention is `K2HorizonMoVAAttention` (:647-651).
- `K2HorizonMoVAAttention.__init__`: q_proj, k_proj, o_proj, but no v_proj. `v_router = Linear(hidden, mova_num_experts, bias=moe_gate_bias)` and `v_experts = mova_num_experts x Linear(hidden, n_kv * head_dim, bias=False)` (:351-392). The output gate is `gate_proj` with `attention_gate_func` (softplus here, :506).
- forward (:428-447): `calc_router_weights(logits, bias, router_score_func, top_k = mova_num_experts_per_tok, router_scaling_factor)`, then `combine_routed_experts(..., activation=F.silu)`. So V = s * sum_{k in topk} w_k silu(W_k x).
- `calc_router_weights` (:136-166): sigmoid or softmax scores, the bias for selection only, renormalised whenever top_k > 1 (no norm_topk_prob), times scaling_factor. `combine_routed_experts` (:169-~190) applies silu to each selected expert's output before the weighted sum.
- Snapshot: Jais-3-MoVA-36B-A4B has mova_num_experts 64, top 4, sigmoid, moe_gate_bias true, router_scaling_factor 2.5, mlp_only_layers [0, 1, 2]: 64 value experts on every sparse layer. The other 8 k2_horizon repos (and the Jais-ASR wrapper) have mova_num_experts 0 and use the plain attention.
- p5 does not read `mova_*`. classify sees out_gate plus the MoE FFN, and the value path counts as a plain v_proj.

## 2. Classifier change
- reads: `mova_num_experts`, `mova_num_experts_per_tok`, `num_experts` (k2_horizon / jais_asr code).
- new: FEATURES key `mova` in `_feats`.
- detection:
```python
def detect_mova_routed_values(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in ("k2_horizon", "jais_asr"):
        return None
    e, k = num(lm.get("mova_num_experts")), num(lm.get("mova_num_experts_per_tok"))
    if e > 0 and num(lm.get("num_experts")) > 0:
        return [e, k]
    return None
```
- apply note: p5:model_graph.py FEATURES (~1597): add `"mova": ("wall", "wall", "routed value experts V = sum_k w_k silu(W_k x) (K2-Horizon / Jais-3 MoVA)")`. In `_feats` (~1741, near out_gate): `if mt in ("k2_horizon", "jais_asr") and _num(d.get("mova_num_experts")) > 0 and _num(d.get("num_experts")) > 0: f.add("mova")`. If a MoVA <- MoVA rule is wanted later, the dims are [E_v, k_v].

## 3. Relation rule
- target side: V is a data-dependent routed sum of silu'd projections. No plain attention (V linear in x) forms it, because silu(W x) is not linear. A MoVA target needs a MoVA host with k_h a multiple of k_t and E_h >= m E_t (the MoE duplication rules, applied to the value router).
- host side: a MoVA host playing a linear-V target needs V = W x. Tied value experts plus renormalised weights give a constant total weight s (`tied_moe_is_dense`), but silu(W x) stays nonlinear. Only the small-input limit silu(z/S) * 2S -> z gives a linear V. That is a limit with no measured agreement, so it is a wall by the high-agreement standard.
- kind: wall (target); wall (host; limit unmeasured).
- cost: none.
- free by the config rules: router_scaling_factor and the scoring type are free routing numbers. mova_num_experts is a feature switch (it changes the module), so it is not free.
- theorem: target wall needs theorem `mova_not_linear_v`: for a fixed key set, a plain head's output is linear in the value input, and silu(W x) is not (it is analytic but not linear). Host limit needs a theorem in the `epsLever_*` style for silu (`silu_small_scale_limit`), plus a measurement.

## 4. Test fixtures
- flag: inception42/Jais-3-MoVA-36B-A4B -> [64, 4] (the only live repo).
- not flag: IFM/K2-Horizon-375B-A23B, inception42/Jais-3-375B-A23B (k2_horizon MoE, mova_num_experts 0); IFM/K2-Horizon-7B, IFM/K2-Horizon-32B (dense k2_horizon); IFM/Jais-ASR (no MoVA).
- scan: exactly Jais-3-MoVA-36B-A4B.

## Verdict
real. Fewer edges: Jais-3-MoVA-36B-A4B loses every host (no other MoVA point) and, as a host, every target. Its current edges are with out_gate-capable / grouped-norm-compatible MoE hosts and targets (grouped_rmsnorm, a4_norms, already narrows them). Expect about 1 point with a handful of edges to drop.
