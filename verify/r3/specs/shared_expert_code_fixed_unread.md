# shared_expert_code_fixed_unread
live 1 / placed 1 (llama4_text 1); direction both; batch a10_moe_ffn

## 1. What the code computes
- `Llama4TextMoe.__init__` always builds `shared_expert = Llama4TextMLP(config)`, width `config.intermediate_size` (Llama4TextMLP default, tf5:models/llama4/modeling_llama4.py:88-96, 163-164). There is no config switch.
- `Llama4TextMoe.forward`: `routed_in = x * router_scores` (input-side sigmoid gating), `out = shared_expert(x) + sum(routed_out)` (modeling_llama4.py:166-174).
- `Llama4TextDecoderLayer`: MoE on `layer_idx in config.moe_layers`, otherwise a dense `Llama4TextMLP(intermediate_size=config.intermediate_size_mlp)` (modeling_llama4.py:407-411).
- Snapshot: Llama-4-Scout-17B-16E-MLX-text-8bit has intermediate_size 8192, intermediate_size_mlp 16384, 16 experts, top-1, interleave_moe_layer_step 1 (every layer MoE). It has no shared-expert key.
- p5 `_has_shared_expert` (p5:model_graph.py:2423) reads only shared keys, so Scout gets ffn = moe, moe_sig shared = 0, and ffn_active = 1 x 8192 instead of 16384.

## 2. Classifier change
- reads: model_type llama4 / llama4_text, `intermediate_size`.
- new: a type entry in `_has_shared_expert` plus the shared width in `dims_of`.
- detection:
```python
def detect_shared_expert_code_fixed_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in ("llama4", "llama4_text") and num(lm.get("num_local_experts")) > 1:
        return num(lm.get("intermediate_size")) or 8192
    return None
```
- apply note: `_has_shared_expert` (~2423): `or str(d.get("model_type") or "").lower() in ("llama4", "llama4_text")`. In `dims_of` (~3147, the `shared = (...)` chain): `or (g("intermediate_size") if d.get("model_type") in ("llama4", "llama4_text") else 0)`. classify's ffn becomes moe_shared, and `_moe_sig` gets shared = 8192. Optional (no live effect, Scout has no dense layers): the dense width of llama4 dense layers is intermediate_size_mlp; read it as dense_inter for llama4 when moe_layers / step leave dense layers (Maverick).

## 3. Relation rule
- target side: an always-on shared expert of width 8192 needs a host with shared >= 8192 (moe_why `shared expert`), or dense capacity (`moe_sub_moe_shared` from the host side).
- host side: the host gains 8192 always-on units for dense / MoE-as-dense plays (`dense_sub_moe_shared`).
- kind: exact (capacity accounting).
- cost: width (shared 8192).
- free by the config rules: n/a (code-fixed).
- theorem: `moe_sub_moe_shared`, `dense_sub_moe_shared` (PROVED exact).

## 4. Test fixtures
- flag: lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-8bit -> 8192 (the only llama4 point).
- not flag: meta-llama/Llama-3.1-8B, meta-llama/Llama-3.1-70B (llama, dense), HIT-SCIR/Chinese-Mixtral-8x7B (MoE, no shared expert).
- scan: exactly the one Scout repo.

## Verdict
real, classification-only today. Scout's ffn slot changes moe -> moe_shared and its ffn_active / ffn_total grow by 8192. Its edges do not change, because router_in_gate (wall / wall) already confines llama4 to llama4 targets and hosts, and the snapshot has one llama4 point. The change matters once router_in_gate gets a host-side rule.
