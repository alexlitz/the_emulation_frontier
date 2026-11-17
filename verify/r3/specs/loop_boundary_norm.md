# loop_boundary_norm
live 5 / placed 5 (hrm_text 3, nanbeige 2); direction both; batch a11_recurrent_loops

## 1. What the code computes
- Nanbeige4.2 (Nanbeige/Nanbeige4.2-3B/modeling_nanbeige.py): after each of num_loops passes
  `if not getattr(config, "skip_loop_final_norm", False): hidden_states = self.norm(hidden_states)` (:2401-2402); with
  the flag true the norm runs once at the end (:2411-2412). Snapshot: num_loops 2, skip_loop_final_norm false -> loop 2
  starts from RMSNorm(h) * w_final (the shared final-norm weight).
- qwen3loop (ext:qwen3loop/.../modeling_qwen3loop.py): the same per-pass `self.norm` (:262-263), but
  `get_num_loops()` returns 1 under enable_double_loop_split (:89-95): the Lordnyx snapshot norms once (not flagged).
- Ouro (ByteDance/Ouro-1.4B-Thinking/modeling_ouro.py): `hidden_states = self.norm(hidden_states)` after every one of
  the total_ut_steps passes, no switch (:654). (Not in the carrier list; the same code fact, found under
  loop_depth_unread.)
- HRM-Text (tf5:models/hrm_text/modular_hrm_text.py): `HrmTextStack` = num_layers_per_stack layers + a weight-free
  `final_norm = HrmTextRMSNorm(eps)` (:225-230, NanoChatRMSNorm: x rsqrt(mean x^2 + eps), no weight) applied at the end
  of EVERY stack pass (:250), 8 passes per forward (H_cycles 2 x (L_cycles 3 + 1)).
- Dory (ext:dory/transformers-pr50099/modeling_dory.py): `hidden_states = self.norm(hidden_states) + input_hidden_states`
  at every loop start (:619-620), norm = DoryL2Norm (unit L2, weight-free, :50-61).
- p5: Nanbeige is LOOP_PLAIN / LOOP_NORM_BETWEEN (p5:model_graph.py:3834-3835): the norm is priced only for an untied
  target on a Nanbeige HOST (+2 units / layer, :4414). As a TARGET, LOOP_ORDER's looped <- untied edge ("depth", exact,
  :404) claims the unrolled stack equals an untied pre-norm stack, which has no stream renormalisation mid-stack.

## 2. Classifier change
- reads: model_type nanbeige / qwen3loop (num_loops, loop_loss_weights, skip_loop_final_norm, enable_double_loop_split),
  ouro (total_ut_steps), hrm_text, dory.
- new: FEATURES key `loop_bnorm` ("wall", "wall"), set in `_feats`; host side switched off (host_off) for the
  config-flag kind (LOOP_BNORM_FLAG_TYPES = {"nanbeige", "qwen3loop"}: skip_loop_final_norm = true is a norm on/off flag).
- detection (returns the kind):
```python
def detect_loop_boundary_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in ("nanbeige", "qwen3loop"):
        if lm.get("skip_loop_final_norm") is True or lm.get("enable_double_loop_split") is True:
            return None
        w = lm.get("loop_loss_weights")
        loops = len(w) + 1 if (model_type == "qwen3loop" and isinstance(w, list) and w) else num(lm.get("num_loops"))
        return "final_norm_cfg" if loops > 1 else None
    if model_type == "ouro":
        return "final_norm" if (num(lm.get("total_ut_steps")) or 4) > 1 else None
    if model_type == "hrm_text":
        return "stack_norm"
    if model_type == "dory":
        return "l2" if num(lm.get("n_recurrent_loops")) >= 1 and num(lm.get("n_recurrent_layers")) else None
    return None
```
- apply note: FEATURES (p5:model_graph.py:1597, beside "post_ln"): `"loop_bnorm": ("wall", "wall", "the residual stream
  renormalised at every loop / stack-pass boundary ...")`; `_feats` (:1741): `if <detector> : f.add("loop_bnorm")`.
  emul_matrix's host_off (built before :4234): add "loop_bnorm" for hosts in LOOP_BNORM_FLAG_TYPES. Optional: with the
  flag free, p5's +2 units / layer loop_norm charge on the Nanbeige host (:4414) can drop to 0 (set the flag).

## 3. Relation rule
- target side: a looped target with a boundary norm needs a host that renormalises the stream at the same boundaries:
  another loop_bnorm host with the same loop structure (Nanbeige <-> Ouro: whole stack, final-norm weight between
  loops). On an untied pre-norm host the unrolled stack needs RMSNorm(h) between layer b and b+1: no pre-norm layer
  sets its output stream to a normalised copy of its input (as post_ln). Wall.
- host side: Nanbeige / qwen3loop: free (skip_loop_final_norm = true); Ouro / HRM / Dory: hard-coded, wall (a
  scale-invariant renorm cannot be switched off by weights).
- kind: wall (target side, all kinds); host side free for final_norm_cfg, wall otherwise.
- cost: none (the p5 untied-on-Nanbeige rescue keeps its own pricing).
- free by the config rules: skip_loop_final_norm (norm on/off flag) is free; the Ouro / HRM / Dory norms are code.
- theorem: needs theorem `looped_bnorm_not_unrolled`: `looped_sub_untied`'s premise (K loops of a block = K b layers
  with tied weights) fails when a norm N(h) sits between blocks: an untied pre-norm stack x -> x + f(N x) has no
  layer whose output is N(x) (cf. the post_ln walls).

## 4. Test fixtures
- flag: Nanbeige/Nanbeige4.2-3B, -3B-Base -> "final_norm_cfg"; danish-foundation-models/DFM-Mimir, -v1.5,
  sapientinc/HRM-Text-1B -> "stack_norm"; ByteDance/Ouro-1.4B, Ouro-2.6B-Thinking -> "final_norm" (all 4 Ouro flag);
  ProCreations/Dory-Beam-4B -> "l2" (all 4 Dory flag).
- not flag: Lordnyx/qwen3loop-... (double split: one norm at the end); IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct (norm
  only after the last loop, modeling_iquestloopcoder.py:975); TheBloke/Nanbeige-16B-Base-32K-GGUF (old Nanbeige, no
  loop keys); cybertruck32489/LFM3-1.7B-RD; osunlp/ARLoop-0.6B; alice-noa-chan/haru.

## Verdict
real. Fewer edges: the 2 Nanbeige targets lose their exact looped <- untied edges (~hundreds of untied hosts with
eff_layers >= 44 today); Ouro / HRM / Dory are mostly walled already (post-norms / ngpt / untied depth) but get the
right reason. Nanbeige as a host keeps p5's untied-target rescue (flag free). Region: recurrent-other (loop).
