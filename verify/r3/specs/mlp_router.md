# mlp_router
live 1 / placed 1 (llama_moe 1); direction over-credit; batch a9_router

## 1. What the code computes
- llama-moe/LLaMA-MoE-v1-3_0B-2_16/modeling_llama_moe_hf.py:777-791 `_create_gate`: gate_type TopKBalancedNoisyGate
  (default) with `gate_network=kwargs.get("gate_network", "mlp")`, `use_softmax=gate_use_softmax` (default True).
- :440-455 get_gate_network: "linear" = Linear(H, E) (zero-init); "mlp" = Linear(H, E, no bias) -> Tanh ->
  Linear(E, E, no bias). Snapshot: gate_network "mlp", gates "mlp".
- :484-497 forward: logits = W2 tanh(W1 x) (noise only when training, :486), top-(k+1) then the first k, softmax over
  the k selected logits (renorm R, see renorm_by_code_unlisted).
- A linear router's logits are W x; W2 tanh(W1 x) is bounded and nonlinear in x, so no linear router equals it, and
  the reverse holds only as the limit W1 = eps W, W2 = I / eps (tanh(eps z) / eps -> z), unmeasured.

## 2. Classifier change
- reads: model_type llama_moe, gate_type, gate_network
- new: FEATURES key `router_mlp` ("wall", "wall") in _feats + FEATURES
- detection:
```python
def detect_mlp_router(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "llama_moe" or not _n_experts(lm):
        return None
    if _s(lm.get("gate_type", "TopKBalancedNoisyGate")) != "topkbalancednoisygate":
        return None
    return "tanh_mlp" if _s(lm.get("gate_network", "mlp")) == "mlp" else None
```
- apply note: p5:model_graph.py `_feats` router block (~1777-1793): `if mt == "llama_moe" and str(d.get("gate_network",
  "mlp")).lower() == "mlp": f.add("router_mlp")`; FEATURES (~1654 group): `"router_mlp": ("wall", "wall", "a tanh-MLP
  router W2 tanh(W1 x) (LLaMA-MoE gate_network mlp): no linear router forms it; the linear limit eps -> 0 is
  unmeasured")`.

## 3. Relation rule
- target side: an MLP-router target needs an MLP-router host (same E: W1 H x E, W2 E x E)
- host side: a tanh-MLP host plays a linear-router target only as the limit tanh(eps W x) / eps -> W x (W2 = I / eps);
  not measured, so a wall; gate_network "linear" would be a different (shape-changing) code path
- kind: exact (router_mlp <- router_mlp, same shape); wall (both cross directions; host-side limit unmeasured)
- cost: none
- free by the config rules: none (gate_network is a feature switch that changes the parameter shapes)
- theorem: needs theorem: tanh_router_limit (selection and weights converge off routing ties as eps -> 0) and a
  measured high-agreement run before the host side may become a limit edge; tanh_router_ne_linear (target side)

## 4. Test fixtures
- flag: llama-moe/LLaMA-MoE-v1-3_0B-2_16 -> "tanh_mlp"
- not flag: Qwen/Qwen3-30B-A3B, allenai/OLMoE-1B-7B-0924, ggml-org/stories15M_MOE (linear routers)

## Verdict
real, small. Fewer edges: LLaMA-MoE-v1-3B (today renorm '?llama_moe', so few edges as a target) loses its host-side
credit for linear-router MoE targets (a handful of small softmax-R models) and every edge it would gain from the
renorm fix.
