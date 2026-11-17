# lfm3rd_recurrent_controller
live 2 / placed 2 (lfm3_rd 2); direction over-credit; batch a11_recurrent_loops

## 1. What the code computes
(cybertruck32489/LFM3-1.7B-RD/modeling_lfm3_rd.py; -v2's file differs in layout, same forward: v2 :1114-1115,
:1194-1195)
- `Lfm3RDController` (:195-): `step_scales = Parameter(ones(32))` (:211), injection (inj_norm, rank-64 inj_A / inj_B,
  tanh gates), per-motif refinement adapters, token gates, a halting head.
- `Lfm3RDModel.forward`, manual mode (the snapshots: reasoning_mode "manual", reasoning_effort null): prelude ->
  `prelude_out` (:1017-1018); T = recurrent_loops phases, one motif per step (:1102-1113); after the core
  `hidden_states = prelude_out + step_scales[T-1] * (h_core - prelude_out)` (:1179-1180), then the coda.
  At T = 5 = num_motifs every layer runs once, but the coda reads p + s (sum of the 20 core layers' updates) with a
  learned scalar s; the core layers' pre-norms read the unscaled stream.
- T > num_motifs adds `inject_feedback` (h = prelude_out + tanh(g_c) (0.08 RMSNorm(h_latent - prelude_out) + B A
  RMSNorm(.)), :261-265) and `compute_deep_refinement_step` (h_latent += sigmoid(w . RMSNorm(h) + g_s) (1 - 0.85 p_ready)
  (...)  :269-290); T < num_motifs runs the remaining phases FFN-only; auto mode halts per data (per_token_early_exit).
- p5: classify gives the lfm3 features (hybrid, rf_shortconv, post_norm, tied) + loop=looped with eff 104; nothing
  for the controller.

## 2. Classifier change
- reads: model_type lfm3_rd (num_motifs, recurrent_loops).
- new: FEATURES key `rd_controller` ("wall", "free"), in `_feats`.
- detection:
```python
def detect_lfm3rd_recurrent_controller(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "lfm3_rd" and num(lm.get("num_motifs")) and num(lm.get("recurrent_loops")):
        return True
    return None
```
- apply note: p5:model_graph.py FEATURES (:1597): `"rd_controller": ("wall", "free", "LFM3-RD: the core's summed update
  rescaled by a learned step_scale before the coda (plus feedback injection / gated refinement for loops > motifs)")`;
  `_feats` (:1741): `if mt == "lfm3_rd": f.add("rd_controller")`.

## 3. Relation rule
- target side: an untied host computes p + sum(updates); the target's coda needs p + s * sum(updates). Folding s into
  the core's output projections changes what the later core layers' pre-norms read (the stream would be p + s * partial
  sums mid-core), so no exact fold unless s = 1. Wall (exact iff step_scales[T-1] == 1, a weight value: the trained
  checkpoint was not checked; a later pass may read it and lift the wall per checkpoint).
- host side: free: an LFM3-RD host sets step_scales[T-1] = 1 (a weight) and runs at its config T = num_motifs in manual
  mode, where the feedback / refinement paths are inactive: a plain 24-layer LFM3 stack.
- kind: wall target side; free host side.
- cost: none.
- free by the config rules: step_scales are weights; reasoning_mode / recurrent_loops are feature switches / depth, not
  free (the host keeps its own manual T = 5).
- theorem: needs theorem `scaled_core_sum_not_prenorm` (a post-hoc scalar on a block of residual updates is not an
  additive pre-norm stack unless s = 1; cf. the residual-multiplier rules, which scale each update as it is added).

## 4. Test fixtures
- flag: cybertruck32489/LFM3-1.7B-RD, cybertruck32489/LFM3-1.7B-RD-v2 -> true (all live).
- not flag: LiquidAI/LFM3-1.7B-Base (the parent lfm3, no controller); Nanbeige/Nanbeige4.2-3B; ProCreations/Dory-Beam-4B;
  ByteDance/Ouro-1.4B.

## Verdict
real. Fewer edges for the 2 points as targets: with loop_depth_unread they become untied 24-layer LFM3 points, which
would otherwise get the LFM3 cluster's exact edges (~tens of hosts); as hosts they keep playing LFM3 targets (free).
Region: recurrent-other.
