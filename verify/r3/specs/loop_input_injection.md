# loop_input_injection
live 4 / placed 4 (dory 4); direction over-credit; batch a11_recurrent_loops

## 1. What the code computes
- Dory (ext:dory/transformers-pr50099/modeling_dory.py `DoryModel.forward`): prelude `layers[:n_input_layers]` once
  (:612-616); `input_hidden_states = self.norm(hidden_states)` (:618); `for loop_idx in range(n_recurrent_loops):
  hidden_states = self.norm(hidden_states) + input_hidden_states` then the recurrent span (:619-628); coda once
  (:631-634); final norm (:636). norm = DoryL2Norm (unit L2, weight-free, :50-61). Snapshot: 16 + 40 x 2 + 16.
  The prelude output e is a second stream carried across the whole recurrent block and re-added at each loop start.
- Huginn / Raven (tomg-group-umd/huginn-0125/raven_modeling_minimal.py): `core_block_forward` starts every step with
  `x = adapter(cat([x, input_embeds]))` (:774), adapter = Linear(2 n_embd, n_embd) (:554). Not placed today
  (classify None), flagged for when it is.
- LFM3-RD: `inject_feedback(prelude_out, h_latent, cycle)` only when recurrent_loops > num_motifs
  (cybertruck32489/LFM3-1.7B-RD/modeling_lfm3_rd.py:261-265, called :1139-1141); the snapshots run T = 5 = num_motifs:
  not active (covered by lfm3rd_recurrent_controller).
- p5: Dory classified loop=looped, eff 56 / param 36 (after the "mlp" sub-layer fraction); the only loop rule is depth
  (LOOP_ORDER looped <- untied, p5:model_graph.py:404).

## 2. Classifier change
- reads: model_type dory (n_recurrent_loops, n_recurrent_layers), huginn_raven (mean_recurrence), lfm3_rd
  (recurrent_loops > num_motifs).
- new: FEATURES key `loop_inject` ("wall", "wall"), in `_feats`.
- detection:
```python
def detect_loop_input_injection(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "dory" and num(lm.get("n_recurrent_loops")) >= 1 and num(lm.get("n_recurrent_layers")):
        return "add"
    if model_type in ("huginn_raven", "huginn", "raven") and num(lm.get("mean_recurrence")) >= 1:
        return "concat_adapter"
    if model_type == "lfm3_rd" and num(lm.get("recurrent_loops")) > num(lm.get("num_motifs")) > 0:
        return "feedback"
    return None
```
- apply note: p5:model_graph.py FEATURES (:1597): `"loop_inject": ("wall", "wall", "the loop input re-derived from the
  prelude output / embeddings at every iteration (Dory h = L2(h) + e, Huginn adapter(cat[x, e])): a second stream an
  unrolled pre-norm stack does not carry")`; `_feats` (:1741): add it when the detector fires.

## 3. Relation rule
- target side: unrolled on an untied host the loop-start op is h <- N(h) + e with e computed layers earlier: e would
  ride in the residual (seen by every pre-norm between) and N(h) is a mid-stack renorm. Needs a host with the same
  injection (Dory <-> Dory; Huginn <-> Huginn). Wall.
- host side: Dory's injection is hard-coded (the norm + add at every loop start); with n_recurrent_loops = 1 it still
  computes 2 L2(h). Wall.
- kind: wall both ways.
- cost: none.
- free by the config rules: n_recurrent_loops is a depth count, not free; no switch exists.
- theorem: needs theorem `loop_inject_not_prenorm`: an untied pre-norm stack has no layer computing N(h) + e(h_0) for a
  stored earlier state; a carry of e (`vanishing_carry_limit`) is at best a limit, and here e is a magnitude consumer
  (added to the stream), so not covered.

## 4. Test fixtures
- flag: ProCreations/Dory-Beam-4B, netprojects/nemotron-dory1.0-4b-mlx-8bit, nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8,
  vladimir94/dory-1.0-4b-ane-coreml -> "add" (all live); tomg-group-umd/huginn-0125 -> "concat_adapter" (not placed).
- not flag: Nanbeige/Nanbeige4.2-3B (loop restarts from the stream); ByteDance/Ouro-1.4B; cybertruck32489/LFM3-1.7B-RD
  (T = num_motifs: no feedback); IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct; sapientinc/HRM-Text-1B (its two-state
  injection is hrm_hierarchical_loop).

## Verdict
real, ~0 new edges lost: Dory already carries the ngpt wall (a4_norms ngpt_sphere_residual) both ways; this records
the loop-specific reason and covers Huginn once placed. Region: recurrent-other (loop).
