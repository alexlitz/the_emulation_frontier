# loop_depth_unread
live 9 / placed 9 (ouro 4, cfrd 3, lfm3_rd 2); direction both; batch a11_recurrent_loops

## 1. What the code computes
- Ouro (ByteDance/Ouro-1.4B-Thinking/modeling_ouro.py): `self.total_ut_steps = getattr(config, "total_ut_steps", 4)`
  (:560); forward runs ALL num_hidden_layers layers `for current_ut in range(self.total_ut_steps)` (:640-652), weights
  tied, a separate KV slot per (step, layer) (`max_cache_size = total_layers * total_ut_steps`, :591-594). Snapshot:
  total_ut_steps 4, L = 24 / 48 -> param L, eff 4L = 96 / 192.
- CFRD (alice-noa-chan/haru*/model.py): `cells = ModuleList(FoldedCell x physical_cells)` (haru :519), loop
  `for recurrence_index in range(run_recurrences): cell = self.cells[recurrence_index % len(self.cells)]` (:605-606);
  the HF wrapper runs `config.inference_recurrences` (modeling_cfrd.py:70), default = recurrences
  (configuration_cfrd.py:72). Per-depth FiLM vectors (attn/ffn_phase_scale/shift per recurrence, :522-525) and a
  1/sqrt(recurrences) residual scale (:536). haru_1.1 / haru_2 add a `BindingBlock` when use_binding_block
  (haru_2/model.py:622, applied before the logits :748-749). Snapshot: haru 2 cells x 6 (no binding), haru_1.1 3 x 6 +
  binding, haru_2 6 x 6 + binding (untied).
- LFM3-RD (cybertruck32489/LFM3-1.7B-RD/modeling_lfm3_rd.py): prelude num_prelude_layers, a core of num_motifs phases of
  motif_size layers, coda (:904-909); the recurrent loop runs ONE phase per step, `m_idx = s % n_motifs` for
  `s < T = recurrent_loops` (:1102-1113); steps < num_motifs finish the remaining phases FFN-only (`_complete_ffn_tail`,
  :937-947). Snapshot: 2 + 5 x 4 + 2 = 24, T = 5 = num_motifs: every layer runs once -> param 24, eff 24.
- p5: `_LOOP_COUNT_KEYS` (p5:model_graph.py:696) lacks total_ut_steps -> Ouro loop=untied, eff = L. No layer-count key
  for CFRD -> dims layers 0 (`dims_of` :3186), loop=untied. `_loop_depth`'s LFM3-RD branch (:713-714) gives
  n + num_motifs * motif_size * (T - 1) = 104 and `_looped` (:682-690) says looped (recurrent_loops 5 > 1).

## 2. Classifier change
- reads: total_ut_steps (ouro); physical_cells, recurrences, inference_recurrences, use_binding_block (cfrd);
  num_motifs, motif_size, recurrent_loops, num_hidden_layers (lfm3_rd).
- new: dims (param_layers, eff_layers) in `_loop_depth` / `dims_of`; slot loop = "looped" iff eff_layers > param_layers
  (classify / `_looped`).
- detection (returns [param_layers, eff_layers]):
```python
def detect_loop_depth_unread(cfg, model_type):
    lm = lm_of(cfg)
    n = num(lm.get("num_hidden_layers"))
    if model_type == "ouro":
        steps = num(lm.get("total_ut_steps")) or 4
        return [n, n * steps] if n else None
    if model_type == "cfrd":
        cells = num(lm.get("physical_cells")) or 2
        rec = num(lm.get("inference_recurrences")) or num(lm.get("recurrences")) or 6
        bind = 1 if lm.get("use_binding_block") is True else 0
        return [min(cells, rec) + bind, rec + bind]
    if model_type == "lfm3_rd":
        m, s, t = num(lm.get("num_motifs")), num(lm.get("motif_size")), num(lm.get("recurrent_loops"))
        if not (n and m and s and t):
            return None
        return [n, n + s * max(0, t - m)]
    return None
```
- apply note: p5:model_graph.py `_LOOP_COUNT_KEYS` (:696) += "total_ut_steps" (the whole-stack fallback at :715-716 then
  gives L x 4; keep the getattr default 4 for an ouro config without it). `_loop_depth` (:700): add before the LFM3-RD
  branch a CFRD branch `if g("physical_cells") and g("recurrences"):` (as above) and REPLACE the LFM3-RD formula with
  `n + motif_size * max(0, T - num_motifs)`. `_looped` (:682): add physical_cells / total_ut_steps, and make the loop
  slot (classify :2359) "looped" only when `_loop_depth` gives eff > param (LFM3-RD at T = 5 and haru_2 are untied).
  `dims_of` (:3186): for cfrd set n_layers from the same branch (there is no layer-count key).

## 3. Relation rule
- target side: a looped target on an untied host: `looped_sub_untied` (exact, depth eff_layers); Ouro additionally hits
  loop_boundary_norm / per_token_early_exit walls (separate specs).
- host side: a looped host for an untied target: p5's loop rule (only LOOP_PLAIN_TYPES are payable; Ouro / CFRD are not
  listed, so a wall).
- kind: exact (depth accounting only; no new rule).
- cost: depth: an untied host needs eff_layers (Ouro 96 / 192, haru 6, haru_1.1 7); LFM3-RD 104 -> 24.
- free by the config rules: total_ut_steps / recurrences are depth counts, not free values.
- theorem: `looped_sub_untied` (PROOFS.md "looped ⊆ untied"); `untied_sub_looped_periodic` for the host side.

## 4. Test fixtures
- flag: ByteDance/Ouro-1.4B, -1.4B-Thinking -> [24, 96]; Ouro-2.6B, -2.6B-Thinking -> [48, 192]; alice-noa-chan/haru ->
  [2, 6], haru_1.1 -> [4, 7], haru_2 -> [7, 7] (untied); cybertruck32489/LFM3-1.7B-RD, -RD-v2 -> [24, 24] (untied).
- not flag: Nanbeige/Nanbeige4.2-3B (num_loops: read right), ProCreations/Dory-Beam-4B (span keys: right),
  IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct (loop_num), osunlp/ARLoop-0.6B (physical_layers),
  LuxiaSL/kotodama-tsumugi-1b (loop_len), Lordnyx/qwen3loop-... (its own spec), LiquidAI/LFM3-1.7B-Base (plain lfm3).

## Verdict
real. Both: Ouro 4 points become looped with 4x depth (fewer edges: as targets they need >= 96 / 192-layer hosts, as
hosts they stop playing untied targets, ~hundreds of host pairs today); LFM3-RD 2 points drop from eff 104 / looped
to 24 / untied (more edges as targets, ~tens; still walled as hosts by lfm3rd_recurrent_controller); CFRD 3 points get
real depth instead of 0 (today hosted by everything at depth 0: fewer edges as targets, more as hosts). Region:
recurrent-other (depth).
