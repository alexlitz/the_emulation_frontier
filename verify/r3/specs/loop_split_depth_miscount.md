# loop_split_depth_miscount
live 1 / placed 1 (qwen3loop 1); direction both; batch a11_recurrent_loops

## 1. What the code computes
(ext:qwen3loop/hub__Lordnyx__qwen3loop-0.6b-sft-deep-supervision-v1/engine/qwen3loop_python/modeling_qwen3loop.py)
- `get_double_loop_split_layer_order(n, m)`: m defaults to n // 2; `first_unlooped = (n - m) // 2`, the middle
  [first_unlooped, first_unlooped + m) repeated `middle_repeats = (n + m) // m` times, prefix and suffix once
  (:37-75).
- `get_num_loops()` returns 1 under enable_double_loop_split (else len(loop_loss_weights) + 1, else num_loops) (:89-95);
  `get_layer_execution_order()` selects the split order (:97-103).
- forward: each execution has its own KV slot (`virtual_layer_idx = execution_idx`, :212-216): plain fresh attention.
- Snapshot: n 28, loop_middle_layers 14, num_loops 2, enable_double_loop_split true: 7 + 14 x 3 + 7 = 56 executions of
  28 weight layers. Nanbeige's own code has the same function (Nanbeige4.2-3B/modeling_nanbeige.py:403-430; its configs
  leave enable_double_loop_split null).
- p5 `_loop_depth` (p5:model_graph.py:708-712) pairs (loop_middle_layers, num_loops): eff = 28 + 14 x (2 - 1) = 42.

## 2. Classifier change
- reads: model_type nanbeige / qwen3loop; enable_double_loop_split, num_hidden_layers, loop_middle_layers.
- new: dims (param_layers, eff_layers) in `_loop_depth`.
- detection (returns [param_layers, eff_layers]):
```python
def detect_loop_split_depth_miscount(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in ("nanbeige", "qwen3loop") or lm.get("enable_double_loop_split") is not True:
        return None
    n = num(lm.get("num_hidden_layers"))
    m = num(lm.get("loop_middle_layers")) or n // 2
    if not n or not m or n % m:
        return None
    return [n, n + m * ((n + m) // m - 1)]
```
- apply note: `_loop_depth` (p5:model_graph.py:700), first branch: `if d.get("enable_double_loop_split") is True and
  n_layers:` m = loop_middle_layers or n // 2; return (n, n + m * ((n + m) // m - 1)). Then the (loop_middle_layers,
  num_loops) pair (:709) only applies without the split (and should use get_num_loops: len(loop_loss_weights) + 1 when
  weights exist).

## 3. Relation rule
- target side: looped target on an untied host: `looped_sub_untied` at depth eff_layers = 56 (not 42).
- host side: unchanged (p5: qwen3loop is not a plain looped host -- qk-norm; a wall for untied targets).
- kind: exact (depth accounting).
- cost: depth: an untied host needs 56 layers, not 42.
- free by the config rules: loop_middle_layers / the split are structure, not free.
- theorem: `looped_sub_untied` (PROOFS.md "looped ⊆ untied"; the split order is a fixed layer sequence with ties).

## 4. Test fixtures
- flag: Lordnyx/qwen3loop-0.6b-sft-deep-supervision-v1 -> [28, 56].
- not flag: Nanbeige/Nanbeige4.2-3B, -3B-Base (split off: num_loops 2 over the whole stack); ProCreations/Dory-Beam-4B
  (span x loops); 0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther (plain qwen3, the parent family).

## Verdict
real. Fewer edges for 1 point as a target (untied hosts with 42 <= eff < 56 at hidden >= 1024 drop: a few hundred
small Qwen3 / Llama pairs at most); as a host it is walled already. Region: recurrent-other (depth).
