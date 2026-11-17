# hrm_hierarchical_loop
live 3 / placed 3 (hrm_text 3); direction both; batch a11_recurrent_loops

## 1. What the code computes
(tf5:models/hrm_text/modular_hrm_text.py)
- `HrmTextConfig.__post_init__`: if num_layers_per_stack is None, it takes the saved num_hidden_layers and
  num_hidden_layers is REWRITTEN to `num_layers_per_stack * H_cycles * (L_cycles + 1)` (:114-120); defaults H_cycles 2,
  L_cycles 3 (:98-99), embedding_scale 1 / initializer_range (:111-112).
- `HrmTextModel.__init__`: no single stack / final norm; two stacks `L_module`, `H_module` (each num_layers_per_stack
  layers + a weight-free final RMSNorm, :222-230) and a fixed zero `z_L_init` (:282-294).
- forward (:348-400): z_H = embeddings * embedding_scale, z_L = z_L_init (zeros); for each of H_cycles: L_cycles times
  `z_L = L_stack(z_L + z_H)`, then `z_H = H_stack(z_H + z_L)`; output z_H. 8 stack passes = 128 layer applications over
  32 distinct layers, two interacting states. L_bp_cycles / no_grad only affect training.
- All three snapshots save num_hidden_layers 16, no num_layers_per_stack, H 2, L 3 -> P 16, param 32, eff 128.
- p5: classify reads 16 untied layers (loop=untied, eff = param = 16).

## 2. Classifier change
- reads: model_type hrm_text; num_layers_per_stack / num_hidden_layers, H_cycles, L_cycles.
- new: dims (param_layers, eff_layers) = (2P, P H (L + 1)) in `_loop_depth` / `dims_of`; loop = "looped"; FEATURES key
  `hrm_two_state` ("wall", "wall") in `_feats`.
- detection (returns [param_layers, eff_layers]):
```python
def detect_hrm_hierarchical_loop(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "hrm_text":
        return None
    H = num(lm.get("H_cycles")) if "H_cycles" in lm else 2
    L = num(lm.get("L_cycles")) if "L_cycles" in lm else 3
    P = num(lm.get("num_layers_per_stack")) or num(lm.get("num_hidden_layers")) or 16
    return [2 * P, P * H * (L + 1)] if H and P else None
```
- apply note: p5:model_graph.py `_looped` (:682): `if d.get("model_type") == "hrm_text": return True`; `_loop_depth`
  (:700) first branch for hrm_text (as above; when num_layers_per_stack is saved, num_hidden_layers is already the
  unrolled count -- P comes from num_layers_per_stack then). FEATURES (:1597) `"hrm_two_state": ("wall", "wall", "two
  residual states z_L, z_H summed into each other's stack input every pass (HRM)")`; `_feats` (:1741) for hrm_text.
  The per-pass stack norm is loop_boundary_norm ("stack_norm").

## 3. Relation rule
- target side: an untied host unrolling HRM must carry z_H while running the L passes and add it at every L-pass start
  (and z_L at the H pass): two d-wide states in one residual stream, both seen by every pre-norm, plus the boundary
  norms. No looped host has the two-state schedule either. Wall.
- host side: an HRM host cannot drop the second state: z_L enters the H stack (z_H + z_L) and z_H every L pass; with
  L-stack weights zero, z_L = final_norm(z_H) (weight-free norm of a nonzero input), not zero. Wall.
- kind: wall both ways (depth dims recorded for later rules).
- cost: none.
- free by the config rules: H_cycles / L_cycles are loop counts (not free); embedding_scale is a scale (free).
- theorem: needs theorem `two_state_loop_not_single_stream` (the cross-state injection is a stored-state add, as
  loop_input_injection).

## 4. Test fixtures
- flag: sapientinc/HRM-Text-1B, danish-foundation-models/DFM-Mimir, DFM-Mimir-v1.5 -> [32, 128] (every hrm_text config).
- not flag: Nanbeige/Nanbeige4.2-3B (single-state loop); ByteDance/Ouro-1.4B; 01-ai/Yi-1.5-34B-32K (Llama, HRM's base
  decoder layer).

## Verdict
real. Fewer edges: the 3 HRM points, today plain 16-layer untied Llama-like points (prefix_lm only with token_type_ids),
are hosted by and host many plain points (~hundreds to 1000+ pairs each way); after the change they pair only with
each other. Region: recurrent-other (loop).
