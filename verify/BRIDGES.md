# Bridges: Lean theorems → the repo code that builds them

Layer 2 of the verification plan. The Lean library (`proofs/lean`, namespace `EP`, 118 theorems and lemmas when this file was started, proved over exact reals) says what each construction computes. This file links every theorem to the repo function that builds it. `tests/test_lean_bridges.py` checks the theorem's own equation, limit or bound on that function. It uses Hypothesis-drawn inputs in float64, with tolerances derived from float64 rounding (see the test module's docstring). Where `transformers` itself rounds to float32, the slack is float32's, and the test says so.

The library has since grown to 567. The 460 theorems added after the first 118 are in "Batch 2" at the end of this file, with tests in `tests/test_lean_bridges_2.py`. The 29 of those that batch 2 left as NO CODE now have code and tests (`tests/test_lean_bridges_3.py`, "Batch 3" at the end).

Statuses:

- **BRIDGED**: a repo function builds the construction, and the test checks the theorem's statement on it. "(helper)" marks the three places where this branch added a small pure function that writes out a construction the repo had only inline or in part. Those helpers are listed at the end.
- **NO CODE**: no repo function builds it. The row says what is missing.
- **MISMATCH**: a repo function builds something different from the theorem. Each one is explained below the table.
- Helper lemmas (calculus, centring, sums) and impossibility theorems are listed for completeness. They are not counted, because there is no construction to bridge.

`pairScore_zero_freq` is tagged `@[simp] theorem`, so `check.sh`'s grep (and the "118") skips it. It is bridged anyway.

## Counts

| status | rows |
|---|---|
| BRIDGED | 78 (2 of them the fixed MISMATCH rows 1 and 2) |
| NO CODE | 5 (`layerNorm_mirror`, `rmsNorm_sub_layerNorm_mirror`, `seq_stack_sub_parallel`, `silu_shift_mask`, `causal_uniform_head_position`) |
| MISMATCH | 1 (the graph's unconditional swigluoai ⊆ SiLU label for `swigluoai_sub_silu`; rows 1 and 2, `convert_pythia` and `dense_to_wide_nm` with N ∤ M, are fixed and BRIDGED) |
| not counted | 26 helper lemmas, 10 impossibility theorems, 2 negative controls |

75 of the 118 counted theorems are BRIDGED. The 76th BRIDGED row is `pairScore_zero_freq`, which sits outside the 118. The arithmetic: 75 + 5 + 26 + 10 + 2 = 118.

The remaining MISMATCH row (and the two fixed ones) sit next to BRIDGED rows for the same theorems. Each of those theorems also has a repo function that does build it correctly.

## The table

### Common.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `attn_bias_limit` | `attn/qk_norm.attend` (penalised logits a + B c) | `test_attn_bias_limit` | BRIDGED |
| `softmax_ratio_bound` | `attn/qk_norm.attend` | `test_softmax_ratio_bound` | BRIDGED |
| `sum_exp_pos`, `attn_univ_eq_softmax`, `attn_add_const`, `tendsto_exp_neg_rate`, `tendsto_exp_rate`, `tendsto_attn`, `sigmoid_le_exp` | (helper lemmas) | n/a | not counted |

### Position.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `pairScore_eq` | `attn/positional_cross.rope_score` (helper) / `rot` | `test_pairScore_eq` | BRIDGED |
| `pairScore_zero_freq` | `attn/positional_cross.rope_score` | `test_pairScore_zero_freq` | BRIDGED |
| `nope_sub_rope` | `attn/positional_cross.nope_from_rope_error`; `hf/nope_rope.build` (stock Llama, inv_freq ← 0) | `test_nope_sub_rope`, `test_nope_sub_rope_on_llama` | BRIDGED |
| `rope_theta_cannot_reach_nope` | `hf/partial_rope_host._llama_target` (stock rotary: inv_freq[0] = 1 for any θ) | `test_rope_theta_cannot_reach_nope` | BRIDGED |
| `alibi_host_score` | `attn/positional_cross.alibi_rope_host` + `rope_score` (helper); the frozen pair alone is `rope_pair_score` | `test_alibi_host_score` | BRIDGED (helper) |
| `alibi_sub_rope_limit` | same | `test_alibi_sub_rope_limit` | BRIDGED (helper) |
| `alibi_sub_rope_bound` | same; `attn/positional_cross.alibi_smallangle_error` | `test_alibi_sub_rope_bound`, `test_alibi_sub_rope_bound_on_seed_measure` | BRIDGED |
| `alibi_sub_rope_attn` | same, softmax over the host logits | `test_alibi_sub_rope_attn` | BRIDGED (helper) |
| `const_channel_rotates` | `attn/positional_cross.rope_score` | `test_const_channel_rotates` (+ conditions 3/4 below) | BRIDGED |
| `alibi_pair_odd` | `attn/positional_cross.rope_pair_score` | `test_alibi_pair_odd` | BRIDGED |
| `partial_rope_host` | `hf/partial_rope_host.column_map` / `place` / `module_check` (stock GLM / Phi / NeoX hosts) | `test_partial_rope_host_frequencies`, `test_partial_rope_host` | BRIDGED |
| `defaultFreq_match` | `hf/partial_rope_host.theta_only_check`'s route θ_h = θ_t^(rot_h/d_t) | `test_defaultFreq_match` | BRIDGED |
| `partial_rope_host_config` | `hf/partial_rope_host.theta_only_check` | `test_partial_rope_host_config` | BRIDGED |
| `qknorm_pad` | `hf/partial_rope_host.place` (qk-norm branch: ε·d/hd, √(d/hd)·w); `module_check("qwen3_next")` | `test_qknorm_pad`, `test_qknorm_pad_on_qwen3_next` | BRIDGED |
| `retState_eq` | `attn/output_rotation.recurrent_retention` (state read row by row) | `test_retState_eq` | BRIDGED |
| `retention_parallel_eq_recurrent` | `attn/output_rotation.parallel_retention` / `recurrent_retention` | `test_retention_parallel_eq_recurrent` | BRIDGED |
| `rot_rides` | `attn/output_rotation.rot_stack` | `test_rot_rides` | BRIDGED |

### Norm.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `layerNorm_eq_rms_center` | `elephantine/universal.UNorm` (centre = 1, the target of `_ln_from`); `norm/layernorm_rmsnorm.make_variant("exact")` | `test_layerNorm_eq_rms_center`, `test_layerNorm_eq_rms_center_patched_variant` | BRIDGED |
| `affine_fold` | `convert/layernorm_to_rmsnorm._fold_beta_into` (the same `W β` fold `fold_layernorm_model` does) | `test_affine_fold` | BRIDGED |
| `layerNorm_sub_rmsNorm` | `norm/layernorm_rmsnorm.fold_layernorm_model` (stock GPT-NeoX and OPT, random γ, β): residual = P·target residual at every layer, equal logits | `test_layerNorm_sub_rmsNorm` (+ condition 8 tests) | BRIDGED |
| `layerNorm_sub_rmsNorm` | `convert/layernorm_to_rmsnorm.convert_pythia` (= `fold_layernorm_model`) | `test_convert_pythia_folds_the_centring`, `test_bridge_layerNorm_sub_rmsNorm_convert_pythia` | BRIDGED (was MISMATCH 1, fixed) |
| `layerNorm_mirror` | none; norm-level identity checked on `universal.UNorm` | `test_layerNorm_mirror` | NO CODE: no function writes the `[h; −h]` mirror host. `graph/model_graph` / `graph/reduce` only charge its ×2 width (`NORM_WIDTH_FACTOR`). |
| `rmsNorm_sub_layerNorm_mirror` | none | n/a | NO CODE: the whole-stack mirror (every writer writing `[w; −w]`, head reading the first half) is not built anywhere. |
| `mean_center`, `center_center`, `center_add`, `mean_mirror`, `sumsq_mirror` | (helper lemmas) | n/a | not counted |

### Attention.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `gqa_sub_mha` | `hf/gqa_mha.expand_kv` / `build` (stock LlamaModel) | `test_gqa_sub_mha` | BRIDGED |
| `mqa_sub_gqa` | `hf/gqa_mha.expand_kv` (1 → G KV heads, stock LlamaModel) | `test_mqa_sub_gqa` | BRIDGED |
| `mha_sub_wide_mqa` | `attn/mha_mqa_gqa.verify_mqa_to_mha_exact` (stock GPT-2 layer) | `test_mha_sub_wide_mqa` | BRIDGED |
| `mla_sub_mha` | `attn/mla_rank.mla_absorption_error` (also `attn/mla.verify_absorption`, print-only) | `test_mla_sub_mha` | BRIDGED |
| `rmsNorm_bounded` | `attn/qk_norm.rmsnorm` | `test_rmsNorm_bounded` | BRIDGED |
| `pattern_sub_full` | `attn/sparse_patterns.*`, `block/mask_reach.*` masks | `test_pattern_sub_full` | BRIDGED |
| `local_sub_full` | `hf/local_full.build` / `band_mask` (stock Qwen2Model) | `test_local_sub_full` | BRIDGED |
| `finite_mask_limit` | `hf/local_full.band_mask` (finite −B) | `test_finite_mask_limit` | BRIDGED |
| `causal_attn_layer`, `Causal.comp`, `causal_stack` | `attn/ar_diffusion.bidirectional_with_causal_mask` | `test_causal_stack` | BRIDGED (3 rows) |
| `sink_sub_plain` | `attn/attention_sink.emulate_with_null_token` vs `sink_attention`, logits built as (q,1)·(k,0), (q,1)·(0,b) | `test_sink_sub_plain` | BRIDGED |
| `sink_to_plain_limit` | `attn/attention_sink.sink_attention` | `test_sink_to_plain_limit` | BRIDGED |
| `k_eq_v_sub_separate` | `elephantine/universal._attn_any` with wv = wk + `UAttention.v_c` (what `from_gemma4` loads for K = V layers) | `test_k_eq_v_sub_separate` | BRIDGED |
| `logit_diff_of_weights` | (helper lemma) | n/a | not counted |
| `mla_latent_norm_not_affine`, `mla_latent_norm_not_mha`, `affine_eq_bounded_slope_zero`, `no_finite_feature_map`, `no_finite_feature_map_vec`, `linear_attn_ne_softmax` | (impossibility) | n/a | not counted |

### MoE.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `zero_router_constant_gate` | `moe/zero_router.deal_dense_into_moe` / `probe_router` (router logits equal for every token) | `test_dense_sub_moe` | BRIDGED |
| `softmax_const` | `moe/zero_router.probe_router` (Qwen2-MoE, norm_topk_prob = False: gate 1/E) | `test_softmax_const` | BRIDGED |
| `dense_sub_moe` | `moe/zero_router.deal_dense_into_moe` (stock Mixtral, Qwen2-MoE ±norm_topk_prob) | `test_dense_sub_moe`, `test_condition_dense_sub_moe_needs_the_real_tie_break` | BRIDGED |
| `tied_moe_is_dense` | `hf/moe_dense.build` (stock Mixtral, live router) | `test_tied_moe_is_dense` (+ condition 2) | BRIDGED |
| `dense_hard_select` | `moe/dense_emulates_moe.dense_hardselect`, `dense_masked_depth` | `test_dense_hard_select` | BRIDGED |
| `dense_soft_router_limit` | `moe/dense_emulates_moe.dense_softrouter` | `test_dense_soft_router_limit` | BRIDGED |
| `soft_router_tie` | `moe/dense_emulates_moe.dense_softrouter` | `test_soft_router_tie` | BRIDGED |
| `silu_shift_mask` | none | n/a | NO CODE: `dense_masked_depth` masks by multiplying with a hard one-hot. Nothing builds the SiLU host's mask by shifting the gate pre-activations by −B. |

### Activation.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `polarization` | `act/relu_swiglu.check_bilinear_as_square`; `act/relu_net.gated_gate`'s product | `test_polarization`, `test_yarotsky_product_via_square` | BRIDGED |
| `swiglu_eq_sq_ffn` | `act/relu_swiglu.check_swiglu_needs_depth` | `test_swiglu_eq_sq_ffn` | BRIDGED |
| `sigmoid_scale_fold` | `act/exotic_activations.swigluoai_folds_into_silu` | `test_sigmoid_scale_fold` | BRIDGED |
| `swigluoai_sub_silu` | `elephantine/universal.from_gpt_oss` + `UFFN._act` (gate ×1.702, `up_shift` = 1, w_down /1.702, clip) | `test_swigluoai_sub_silu`, `test_swigluoai_sub_silu_on_gpt_oss` (+ condition 5) | BRIDGED |
| `swigluoai_sub_silu` | `graph/model_graph.ACT_EXACT` ∋ (swigluoai, SiLU) | `test_condition_swigluoai_needs_up_bias`, `test_condition_swigluoai_clamp_regime` | **MISMATCH** (3) |
| `relu_silu_bound` | `act/activations.swish_b`, `act/relu_swiglu.swish_beta` | `test_relu_silu_bound` | BRIDGED |
| `relu_sub_silu` | `act/activations.swish_b` | `test_relu_sub_silu` | BRIDGED |
| `relu_sub_gelu` | `act/activations.gelu_b` | `test_relu_sub_gelu` | BRIDGED |
| `glu_ungated` | `act/relu_swiglu.check_glu_superset_vanilla` | `test_glu_ungated` | BRIDGED |
| `softcap_bound` | `attn/softcap.softcap` (= `UAttention.cap`'s expression) | `test_softcap_bound` | BRIDGED |
| `softcap_attn_regime` | `attn/softcap.softcap`, `attn/qk_norm.attend` | `test_softcap_attn_regime` | BRIDGED |
| `qknorm_logit` | `attn/qk_norm.rmsnorm` | `test_qknorm_logit` | BRIDGED |
| `qknorm_sub_plain_regime` | `attn/qk_norm.rmsnorm` | `test_qknorm_sub_plain_regime` | BRIDGED (note a) |
| `hasDerivAt_tanh`, `tanh_le_self`, `tanh_nonneg`, `self_sub_cube_le_tanh` | (helper lemmas) | n/a | not counted |

### Approx.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `reluNet_on_piece` | `act/relu_net.relu_units` + `relu_net` (helper) vs `pw_linear` | `test_reluNet_on_piece` | BRIDGED (helper) |
| `interp_error` | `act/relu_net.pw_linear` | `test_interp_error` | BRIDGED |
| `interp_error_h2` | `act/relu_net.pw_linear` | `test_relu_net_approx` | BRIDGED |
| `relu_net_approx` | `act/relu_net.pw_linear`, `ungated_error` | `test_relu_net_approx`, `test_relu_net_approx_ungated_error` | BRIDGED |
| `yarotsky_exact` | `act/relu_net.yarotsky_sq01`, `_tent` | `test_yarotsky_exact` | BRIDGED |
| `yarotsky_sq_bound` | `act/relu_net.yarotsky_sq01` | `test_yarotsky_sq_bound` | BRIDGED |
| `yarotsky_sym_bound` | `act/relu_net.square` (M = 1) | `test_yarotsky_sym_bound` | BRIDGED |
| `yarotsky_product_bound` | `act/relu_net.square` (M = 2), the product step of `gated_gate` | `test_yarotsky_product_bound` | BRIDGED |
| `deriv2_add_quad`, `tooth_mem`, `iterate_tooth_mem`, `q_tooth`, `relu_add_relu_neg` | (helper lemmas) | n/a | not counted |

### Block.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `zeroed_sublayer` | `block/prenorm_carry.parallel_hosts_seq_2x_depth` (zeroed `dense_4h_to_h` / `dense`) | `test_seq_sub_parallel` | BRIDGED |
| `seq_sub_parallel` | `block/prenorm_carry.parallel_hosts_seq_2x_depth` (stock GPT-NeoX); `block/parallel_residual.verify_both_constructions` | `test_seq_sub_parallel`, `test_seq_sub_parallel_toy` | BRIDGED |
| `seq_stack_sub_parallel` | none | n/a | NO CODE: only one block is built. No function turns an L-block sequential stack into 2L parallel blocks. |
| `parallel_sub_seq_wide` | `block/parallel_residual.parallel_stack_in_seq_wide` (helper, any L); L = 1: `verify_both_constructions`, `prenorm_carry.seq_hosts_parallel(group_norm=True)` | `test_parallel_sub_seq_wide`, `test_seq_sub_parallel_toy` (+ condition 7) | BRIDGED (helper for L ≥ 2) |
| `looped_sub_untied` | `hf/looped_untied.build` (stock LlamaModel) | `test_looped_sub_untied` | BRIDGED |
| `mtp_sub_plain` | `hf/real_reductions.check_mimo_mtp` (MiMo-7B index: the MTP tensors are disjoint from the base path; structural, cached) | `test_mtp_sub_plain` | BRIDGED |
| `dense_sub_nm` | `convert/nm_sparsity.dense_to_wide_nm` (N ∣ M) | `test_dense_sub_nm` | BRIDGED |
| `dense_sub_nm` | `convert/nm_sparsity.dense_to_wide_nm` (N ∤ M) | `test_dense_sub_nm_when_N_does_not_divide_M`, `test_bridge_dense_sub_nm` | BRIDGED (was MISMATCH 2, fixed) |
| `nm_sub_dense` | `convert/nm_sparsity.prune_nm` | `test_nm_sub_dense` | BRIDGED |
| `untied_add`, `untied_congr` | (helper lemmas) | n/a | not counted |

### Walls.lean (positive theorems) and Constructions.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `rope_shift_invariant` | `attn/rope_window.rope_apply` | `test_rope_shift_invariant` | BRIDGED |
| `winReach_iff` | `block/mask_reach.reach` on `sliding` | `test_winReach_iff` | BRIDGED |
| `winLocal_stack` | `block/mask_reach.diameter`, `depth_to_global_formula` | `test_winLocal_stack` | BRIDGED |
| `causal_uniform_head_position` | none | n/a | NO CODE: no function builds the causal head with zero logits and a position-0 marker value that outputs 1/(t+1). That is the depth route to absolute positions (PROOFS finding 6). |
| `partial_rope_freq_match` | `hf/partial_rope_host.theta_only_check`'s route | `test_defaultFreq_match` | BRIDGED |
| `gqa_tiled_scores` | `elephantine/emulate.head_maps`; `hf/gqa_mha.expand_kv` = `universal._tile_kv` | `test_gqa_tiled_scores`, `test_groupOf` | BRIDGED |
| `centred_rms_is_variance` | `norm/layernorm_rmsnorm._rms` on P x | `test_centred_rms_is_variance` | BRIDGED |
| `analytic_ne_kink`, `silu_family_analytic` | (helper lemmas) | n/a | not counted |
| `relu_not_exact_silu`, `relu_sq_not_exact_silu`, `rope_cannot_learned_abs`, `causal_ne_bidirectional` | (impossibility) | n/a | not counted |
| `polarization_half_false`, `partial_rope_wrong_base_false` (NegativeControl) | (negative controls) | n/a | not counted |

## The MISMATCH rows

**Round 2, pass 5: all three are fixed.** (1) `convert_pythia` now runs `norm/layernorm_rmsnorm.fold_layernorm_model`
(the old beta-only conversion is `convert_pythia_beta_only`); `test_convert_pythia_folds_the_centring` checks the
exact fold and the old function's gap. (2) `dense_to_wide_nm` places each group of N columns on N of the M slots for
any N <= M; `test_dense_sub_nm_when_N_does_not_divide_M` checks 3:4, 2:3 and 3:5. (3) `graph/model_graph` prices a
swigluoai target on a host without swigluoai as a bias-gated edge: the host needs an FFN up bias (dims `ffn_bias`), and
the clamp regime stays a limit (`ACT_LIMIT`). The conditions table below (3a / 3b ALiBi causal + a spare frequency-0
pair, 4 the sink's unrotated constant channel, 3/4' a q bias) is enforced by `emul_matrix` too: the ALiBi pair and the
sink's spare slot only where the host's config sets a frequency to 0 (pairs-free or fraction-free rotary, or pass dims),
the sink only on a host with a q bias (the start-token rule). The text below is the pre-fix record.

1. **`convert/layernorm_to_rmsnorm.convert_pythia` vs `layerNorm_sub_rmsNorm`.** The converter folds β into the consumers' biases and swaps every LayerNorm for an RMSNorm. It does not centre the residual writers or the embedding. Its docstring calls the dropped centring "the ONE irreducible penalty". The theorem shows the centring is foldable: post-compose every writer and the embedding with P, and the fold is exact. So the converter computes a different, lossy function. `test_mismatch_convert_pythia_drops_centring` shows a logit gap of more than 1e−4 on a stock GPT-NeoX with random γ, β. `norm/layernorm_rmsnorm.fold_layernorm_model` is the correct construction (BRIDGED). The converter's docstring and its "only irreducible cost" reading are out of date.
2. **`convert/nm_sparsity.dense_to_wide_nm` with N ∤ M vs `dense_sub_nm`.** The function widens by `f = M // N` and writes the dense columns on every f-th slot. That is right when N divides M (1:2, 2:4, 1:4, 2:8). For 3:4, 2:3 or 3:5, `f = 1`: the "wide" matrix is W itself, dense, with up to M nonzeros per group of M. The theorem places each group of N dense columns on N of the M slots, at width M/N with groups of N columns. `test_mismatch_dense_sub_nm_when_N_does_not_divide_M` asserts that more than N nonzeros per group appear. The repo only exercises 2:4, so no reported number is affected.
3. **`graph/model_graph.ACT_EXACT ∋ ("swigluoai", "SiLU")` vs `swigluoai_sub_silu`.** The graph labels a swigluoai target on a SiLU host an exact fold, unconditionally. The theorem gives exactness only while gate ≤ 7 and |up| ≤ 7, and only if the host can add the +1 to the up branch (an up-projection bias or a constant channel). A Llama-style bias-free SiLU MLP can do neither: it misses by exactly SiLU(1.702g)/1.702 without the +1 (`test_condition_swigluoai_needs_up_bias`), and outside the clamp regime the unclamped fold differs (`test_condition_swigluoai_clamp_regime`). The universal host is fine: it has `up_shift` and its own clip (`test_swigluoai_sub_silu_on_gpt_oss`). The graph cell should be a regime / bias-gated edge. It has a sibling: ACT_LIMIT already marks the reverse (SiLU, swigluoai) as a limit because of the clamp. These tests don't assert anything about the graph itself, because other branches are editing `graph/model_graph.py`.

Note a. `attn/qk_norm.part2_fold_vs_variance` folds the mean of the ε-free rms (`q.pow(2).mean(-1).sqrt()`), while `rmsnorm` puts ε inside the root. So its "exact at CV = 0" holds only to a relative ε/ms ≈ 1e−6. `qknorm_sub_plain_regime` is exact when the folded constant is the rms with ε, as the test uses. This is a numeric nit, not a MISMATCH.

## The conditions the proofs exposed

Each condition, whether the repo's construction enforces it, and the test that shows the equation fails without it.

| # | condition | enforced in the repo? | failure test |
|---|---|---|---|
| 1a | dense ⊇ top-1 MoE, β → ∞, needs a **unique argmax** | No. `dense_softrouter` takes any logits. Its docstring says "exact only away from routing boundaries", but nothing checks for ties. | `test_soft_router_tie`: two tied experts give (Y₀+Y₁)/2 at every β. The gap to the tie-broken pick stays at \|Y₀−Y₁\|/2. |
| 1b | the select construction assumes a **unit gate** (renormalised top-1) | Only by definition: `moe_top1`, the reference, has a unit gate. The graph pays the MoE-depth edge (`MOE_DEPTH_FACTOR` = 2, marked "heuristic") for every less-rich-FFN host, whatever the target's top-k or gate (Switch / sigmoid-score top-1, top-k > 1). The proved constructions are top-1 with a unit gate, at E× width plus a select, or E masked sublayers. | `test_condition_top1_unit_gate`: against a Switch-style p·Y target, the hard select misses by exactly (1 − p_{e*})\|Y_{e*}\|. |
| 2 | MoE → dense by **tying experts** needs Σ gates = 1 | Yes, where it is used. `hf/moe_dense.build` uses Mixtral, which renormalises. The MoE hosts in `elephantine/emulate.build_host` and `moe/zero_router` zero the router first, so the gates are constant: `emulate` scales down_proj by E/top_k when `norm_topk_prob` is False, and `deal_dense_into_moe` divides by the probed gate sum. | `test_condition_tied_moe_needs_gates_summing_to_one`: a Qwen2-MoE with `norm_topk_prob=False` and a live router misses f(x) by more than 1e−3·\|f\|. The repo's zero-router route is exact. |
| 3a | ALiBi ⊆ RoPE holds for **causal** attention only | No check. The graph's `POS_ORDER["alibi"]` edges to rope / partial_rope (a "mag" limit) do not look at causality. Every ALiBi model in the zoo (BLOOM, MPT, …) is causal, so no reported number is affected. | `test_condition_alibi_causal_only`: for j > i the host is off by about 2m(j − i) however small θ is. |
| 3b | ALiBi ⊆ RoPE needs a **spare pair at inv_freq = 0** for the content (the frozen pair at θ) | Not in any built construction, because none exists outside `attn/positional_cross`. The graph edge assumes it silently. It is a buffer write, like NoPE. | `test_condition_alibi_content_pairs_need_zero_freq`: content pairs left on the default schedule never give q·k − m(i−j). |
| 4 | the **sink's constant channel** must be unrotated (zero-frequency pair) | No. The repo realises the sink only at score level: `attn/attention_sink.emulate_with_null_token` and `hf/real_reductions.check_gptoss_sink` append the logit b directly, and `UAttention.null_logit` is native to the universal host. `SINK_ORDER` in the graph pays standard ⊇ sink as a free "fold" without asking for a zero-frequency pair or the +1 KV slot. | `test_condition_sink_constant_channel_needs_zero_freq`: in a rotating pair the sink logit s·cos((i−p)ω) moves with the query position. |
| 3/4′ | (implicit in both) the constant components (k = (0, 1), the query's 1) need a **q / k bias**: a reserved residual channel is not constant after a pre-RMSNorm | No. The graph edges don't check for q/k biases. A bias-free Llama-style host (no q/k bias) cannot write the constant. | `test_condition_constant_channel_needs_a_bias`: a residual channel held at 1 varies per token after RMSNorm. |
| 5 | swigluoai ⊆ SiLU needs the **+1 up bias** and the **clamp regime** | In the universal host, yes (`up_shift` = 1, its own clip). In the graph, no: see MISMATCH 3. | `test_condition_swigluoai_needs_up_bias`, `test_condition_swigluoai_clamp_regime` |
| 7 | parallel ⊆ seq needs **(L+1)·d width**, and the norm-free (per-group norm) reads | Width: the repo builds one block at 2d only (`verify_both_constructions`, `seq_hosts_parallel`). The graph charges ×2 width (`wf = 2` in `graph/reduce.axis_gaps`, which mirrors `emulates`) for the whole model, not (L+1)×. Norm: `block/prenorm_carry` documents and measures the stock pre-norm failure, and the graph prices the CLA analogue as a floor. | `test_condition_parallel_sub_seq_width`: one reserved channel reused over L ≥ 2 blocks is off. `test_condition_parallel_sub_seq_needs_norm_free`: the stock single-rms pre-norm is off by about 1e−2; `GroupRMSNorm` is exact. |
| 8 | LayerNorm fold: **same ε**; **every writer** (embedding included) post-composed with P, so a tied head is **untied**; **biases** in the first linears and the **lm_head** when β ≠ 0 | Yes, in `fold_layernorm_model`: it centres every writer and the embeddings, unties a tied head, adds `W β` to the consumers' biases and an lm_head bias (unless `lm_head_bias=False`), and keeps ε. It asserts the consumers have biases only implicitly (`c.bias.add_`, which raises on a bias-free consumer). It supports only GPT-NeoX / OPT. `convert_pythia` violates the writer condition (MISMATCH 1). | `test_condition_layernorm_fold_needs_lm_head_bias`, `test_condition_layernorm_fold_needs_untied_head`, `test_condition_layernorm_fold_needs_every_writer_centred` (a writer, the embedding, or one norm's ε) |

(Condition 6 in PROOFS.md, learned-absolute positions, is about a wall, not a construction. Its positive half, `causal_uniform_head_position`, is NO CODE above.)

## Helpers added in this branch

All three are pure functions, and nothing else in the repo calls them:

- `attn/positional_cross.rope_score(freqs, q, k, i, j)` and `alibi_rope_host(q, k, m, theta)`: the whole ALiBi host head, with content pairs at frequency 0 plus the frozen pair. Before this branch only the frozen pair existed (`rope_pair_score`).
- `act/relu_net.relu_units(phi, N, lo, hi)` and `relu_net(units, x)`: the N-unit ReLU layer with second-difference output weights. `pw_linear` returned `np.interp`, which is the interpolant itself, not a ReLU network.
- `block/parallel_residual.parallel_stack(x, As, Ms)` and `parallel_stack_in_seq_wide(x, As, Ms)`: the (L+1)-channel host for L parallel blocks. Before this branch only L = 1 (2d) was built.

Run: `PYENV_VERSION=3.12.6 python -m pytest tests/test_lean_bridges.py -q`

---

# Batch 2: the theorems added since the first 118

The Lean library now has 567 theorems and lemmas (`check.sh`'s count). This batch covers the 460 that the first batch does not list. They are in 25 files: the new files (`ActFits`, `ActLimits`, `ActWidth`, `AuditRules`, `Capacity`, `CapacityDepth`, `Converses`, `EpsLever`, `KVPack`, `LearnedAbs`, `LimitInstances`, `LimitInstances2`, `LimitKit`, `MLAHost`, `MLARope`, `MoERouting`, `ModelLimit`, `PosBias`, `PreNorm`, `PreNormRescue`, `RandomFeatures`, `RopeRank`) and a few lemmas added to `Norm`, `Position` and `Common`. The tests are in `tests/test_lean_bridges_2.py`, with the same conventions as the first batch: float64, Hypothesis, and tolerances justified by rounding. A limit or bound theorem is checked by evaluating the stated bound numerically on the repo's code.

Two status labels are new in this batch:

- **"(setting)"**: the repo's universal host carries the feature as a module, and the test applies the theorem's weight setting to that module. No loader writes the setting, because the universal host switches the feature off with its own coefficient.
- **"(kit)"**: generic composition theorems (`model_limit`, `model_error_bound`, the `LimitKit` lemmas). No repo function computes their bound by itself. They are exercised by every whole-model instance test and are not counted.

Whole-model theorems take Lipschitz constants, value bounds and regions as hypotheses, and no real transformer layer is globally Lipschitz. So their bounds are checked on small residual stacks whose constants can be computed, built from the repo's own swap module, attention host and greedy routine. Per-layer statements are checked on stock HF layers. On the stock eps-lever hosts, the whole-model bound's 1/S² rate is checked directly.

## Batch 2 counts

| status | theorems |
|---|---|
| BRIDGED | 155 (126 in batch 2, 29 in batch 3) |
| NO CODE | 0 |
| MISMATCH | 6 rows (each on a theorem that also has a BRIDGED row, or on a refutation of a repo claim) |
| not counted | 305 (helper lemmas, impossibility / converse theorems, the interval checker's 71 lemmas in `ActFits`, kit lemmas) |

155 + 0 + 305 = 460. The extraction also matched four words in docstrings as names (`gives`, `does`, `over`, `on`); they are not theorems.

Batch 2 left 29 theorems as NO CODE. Batch 3 (see "Batch 3" at the end of this file) wrote the code for all 29 and bridged them in `tests/test_lean_bridges_3.py`. Their rows below say "(batch 3)".

MISMATCH: W1-1 (the "(L+1)·d" width claim in `block/parallel_residual`), W2-1 (`approx` keeps ε under a stream scalar and under hidden padding), W2-2 (the sink fold called exact), W3-1 (the ReLU fit on a swigluoai host takes the refuted scaled-gate route), W4-1 (`graph/eps_lever.EXACT`) and W5-1 (`serving/embed` duplication has no strict-top-k condition). They are explained after the tables.

## Corrections to the first batch

- **Condition 7, width half.** The first batch says "parallel ⊆ seq needs (L+1)·d width" and that the graph's ×2 is too little. `parallel_sub_seq_2x` proves that norm-free, 2d is enough at any L. The trick is 2d identity units per block in the MLP, which move the attention write out of the spare channel. So the graph's `wf = 2` is right norm-free, but it does not charge the 2d extra MLP units. `test_condition_parallel_sub_seq_width` is still true: one channel reused without the move fails. The norm half of condition 7 stands (`parallel_sub_seq_prenorm_fails`). See MISMATCH W1-1.
- **`causal_uniform_head_position`.** The first batch marks it NO CODE. `hf/bos_constructions` (branch bos-token, copied here) now builds that head: `marker_weights(T, 0)` gives 1/(t+1). It is exercised inside `test_learned_abs_two_block` at L = 0.
- **A first-batch test that fails on this branch.** `test_condition_alibi_content_pairs_need_zero_freq` fails at Hypothesis seed 39: the gap is 0.26 against a threshold of 0.35. The threshold `1e-2·Σ|q|·Σ|k|` is a heuristic and not a bound, so the failure is in the test, not the theorem. The test is left as it is here, because this branch only adds tests. (r3-nonrelation replaces the threshold with a lower bound derived from the per-pair rotation: max|gap| ≥ sin²(1/2)·A_0·cos(G/2) − Σ_{p≥1} sin²(w_p/2)·A_p − sin²(θ/2)·m/θ, by the second difference.)

## Batch 2 tables

### Capacity.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `hidden_pad_stack` | `elephantine/emulate.build_host` (hidden d -> D: eps' = eps d/D, every norm gain x sqrt(d/D), readers / writers zero-padded) | `test_hidden_pad_stack` | BRIDGED |
| `hidden_pad_blocks` | `elephantine/hidden_width.widen` (stock Llama, residual checked after every attention and FFN sublayer) | `test_hidden_pad_blocks` | BRIDGED |
| `hidden_pad_needs_eps` | `elephantine/hidden_width.widen_config` (sets eps d/D); keeping the target's eps fails | `test_condition_hidden_pad_needs_eps` | not counted (converse; condition test) |
| `layerNorm_pad_center` | `norm/layernorm_pad.pad_ln_params` / `host_layer_norm` (batch 3) | `test_layerNorm_pad_center` (batch 3, through `F.layer_norm`), `test_condition_layerNorm_pad_needs_centring` | BRIDGED (batch 3) |
| `hidden_pad_stack_layerNorm` | `norm/layernorm_pad.pad_layernorm_model` (batch 3): a stock GPT-NeoX / OPT host of width d + k·head_dim (centred writers and embeddings, padded LayerNorms with eps d/D, zero extra heads, untied head). `elephantine/push_hub_demo.inflate` still refuses LayerNorm targets | `test_hidden_pad_stack_layerNorm` (tiny random stock models, every layer's residual and the logits), `test_hidden_pad_stack_layerNorm_real_checkpoint` (pythia-14m 128 → 160, OPT-125m 768 → 832, float64 logits within ~1e-13), `test_condition_hidden_pad_stack_layerNorm_needs_centred_writers`; batch 2's `test_hidden_pad_stack_layerNorm_is_refused_by_inflate` | BRIDGED (batch 3) |
| `ffn_extra_units` | `emulate.build_host` (`ffn_in` / `ffn_out`: extra units' down columns 0); also `hidden_width.build_caps`, `serving/embed.mlp` | `test_ffn_extra_units` | BRIDGED |
| `ffn_extra_zero_input` | `emulate.build_host` (extra units' gate / up rows 0; down columns randomised in the test) | `test_ffn_extra_zero_input` | BRIDGED |
| `glu_zero_up_unit` | `emulate.build_host` (extra SwiGLU units, up row 0; gate rows randomised in the test) | `test_glu_zero_up_unit` | BRIDGED |
| `glu_zero_unit` | `emulate.build_host` as built (both rows 0) | `test_glu_zero_unit` | BRIDGED |
| `plain_zero_unit` | `elephantine/universal.UFFN._act` ungated branches (ReLU, relu², GeLU: phi(0) = 0; xIELU: phi(0) != 0) | `test_plain_zero_unit` | BRIDGED |
| `moe_wider_experts` | `serving/embed.embed_state` (Qwen3-MoE -> wider experts, live router) ; `emulate.build_host` tied mode pads the expert width too | `test_moe_wider_experts` | BRIDGED |
| `moe_extra_experts` | `serving/embed.embed_state` (dummy experts: zero weights, router rows -M on the free dim, norm_topk_prob = True) | `test_moe_extra_experts` | BRIDGED |
| `moe_sub_moe_shared` | `emulate.build_host` 'tied' mode on a Qwen2-MoE host (shared down-projection 0; shared gate / up / gate weights randomised in the test); `serving/embed` zero shared expert on unused / steered layers | `test_moe_sub_moe_shared` | BRIDGED |
| `moe_sub_moe_shared_gated` | `emulate.build_host` 'tied' mode as built (`shared_expert_gate` 0, shared down 0) | `test_moe_sub_moe_shared_gated` | BRIDGED |
| `shared_gate_half` | `emulate.build_host` 'shared' mode (`shared_expert_gate` 0 -> sigma = 1/2, shared down x 2) | `test_shared_gate_half` | BRIDGED |
| `unpad_padVec`, `padVec_add'`, `mean_padVec_center` | (helper lemmas) | n/a | not counted |

### CapacityDepth.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `moe_two_layer_margin` | `moe/dense_emulates_moe.dense_two_layer_margin` (helper) | `test_moe_two_layer_margin`, `test_condition_moe_two_layer_margin_needs_margin` | BRIDGED (helper) |
| `untied_sub_looped_periodic` | `hf/looped_untied.build`'s periodic tied stack (b' = b) + `elephantine/hidden_width.build_caps` identity layers (b' > b) | `test_untied_sub_looped_periodic` | BRIDGED |
| `looped_width2_neg` | `block/looped_counter.width2_block` (batch 3), with `neg_target` (`EP.negTarget`) | `test_looped_width2_neg` (batch 3) | BRIDGED (batch 3) |
| `looped_counter_two` | `block/looped_counter.counter_block` (batch 3): one residual ReLU-MLP block on [x; c; o], both layers' units masked by −B c and −B (o − c), a counter unit relu(o); `embed_counter`, `loop`. (`hf/looped_host_prenorm`, branch moe-loop-rescue, is a different pre-norm LIMIT construction) | `test_looped_counter_two`, `test_condition_looped_counter_two_needs_bounded_inputs` (batch 3) | BRIDGED (batch 3) |
| `relu_identity_pair` | `block/parallel_residual.parallel_stack_in_seq_2x`'s identity units (`F.relu`) (helper) | `test_relu_identity_pair` | BRIDGED (helper) |
| `silu_identity_pair` | same, `F.silu` (helper) | `test_silu_identity_pair` | BRIDGED (helper) |
| `parallel_sub_seq_2x` | `block/parallel_residual.parallel_stack_in_seq_2x` (helper) | `test_parallel_sub_seq_2x` | BRIDGED (helper) |
| `parallel_sub_seq_2x` | `block/parallel_residual` / `parallel_stack_in_seq_wide` and `block/prenorm_carry` docstrings: "(L+1) d" needed | `test_parallel_residual_width_claim` | **MISMATCH** (W1-1); docstrings fixed on r3-nonrelation (2d width plus 2d MLP units per block, norm-free); the graph's unit charge is in verify/r3_notes.md |
| `rmsNorm_continuous`, `toyTop1_pos`, `toyTop1_neg`, `relu_nonneg'`, `relu_of_nonpos`, `relu_of_nonneg`, `append_smul_split'`, `untied_identity`, `negTarget_eq` | (helper lemmas) | n/a | not counted |
| `top1_moe_not_continuous`, `continuous_host_ne_top1_moe`, `select_carry_prenorm_fails`, `looped_same_width_fails` | (impossibility) | n/a | not counted |

### AuditRules.lean

The emulate rows run `elephantine/emulate.py`'s own `plan_construction` / `read_target` / `build_host` on tiny random stock models in memory, load the state dict into the host's stock class and compare logits in float64 (fp32 casts lifted). "(setting)" marks rows where the repo's universal host carries the switch as a module and the test applies the theorem's weight setting to it; no loader writes that setting, because the universal host switches the feature off with its own coefficient.

| theorem | repo function | test | status |
|---|---|---|---|
| `bias_off` | `elephantine/emulate.build_host` (bias-free target on a Llama host with `attention_bias` and `mlp_bias`: every host bias left at 0) | `test_bias_off` | BRIDGED |
| `out_gate_off` | `elephantine/universal.UAttention` output gate (`gate_coef = [0, 1]`, `g_proj = 0`, `o_proj` ×2) | `test_out_gate_off` | BRIDGED (setting) |
| `moe_shared_gate_off` | `elephantine/emulate.build_host`, `ffn_mode = "shared"` (Qwen2-MoE host: `shared_expert_gate = 0`, shared `down_proj` ×2) | `test_moe_shared_gate_off` | BRIDGED |
| `extra_input_off` | `elephantine/universal.ULayer.ple_*` (per-layer embeddings) and `UEngram` with zero read-out | `test_extra_input_off` | BRIDGED (setting) |
| `value_emb_off` | `attn/value_emb.add_value_mix` / `set_mix` / `value_emb_off` (batch 3): v_j + λ u_j installed on a stock attention's v_proj, and the λ = 0 setting. The universal host still has no value embeddings (`graph/minimal_emulator.HOST_FEATS`) | `test_value_emb_off` (batch 3, stock Llama: logits bit-identical to the plain model) | BRIDGED (batch 3) |
| `switch_lora_off` | `elephantine/universal` switched LoRA (`lora_adapters`, `la` / `lb`, switch head `sw_score` / `sw_val`) | `test_switch_lora_off` | BRIDGED (setting) |
| `router_bias_off` | `elephantine/universal.UFFN.sel_bias` (selection-only bias, loaded from `e_score_correction_bias` by `_router_from`) at 0 | `test_router_bias_off` | BRIDGED (setting) |
| `multi_rope_tables_equal` | `elephantine/universal.from_llama` (writes the one-table target's `inv_freq` into every layer's `UAttention.inv_freq`) | `test_multi_rope_tables_equal` | BRIDGED |
| `scalar_fold` | `elephantine/emulate.build_host` (Granite target on a Llama host: embedding, residual and logit multipliers folded into `embed_tokens`, `o_proj` / `down_proj`, `lm_head`); `elephantine/universal.from_granite` (`_head_from`, `_fold_residual`) | `test_scalar_fold` | BRIDGED |
| `attn_scale_fold` | `elephantine/emulate.build_host` (`alpha = s_t / s_h` into the host's q rows) | `test_attn_scale_fold` | BRIDGED |
| `norm_weight_fold` | `elephantine/emulate.build_host` (tied host: the logit / embedding scales go into the final norm weight, `kappa·gam·c_h·L_h/a`) | `test_norm_weight_fold` | BRIDGED |
| `zero_centred_norm` | `elephantine/emulate.build_host` / `read_target` (Gemma target: effective `1 + w` stored in a standard host norm) | `test_zero_centred_norm` | BRIDGED |
| `zero_centred_host` | `elephantine/emulate.build_host` `norm_vec` (Gemma host: stored = effective − 1) | `test_zero_centred_host` | BRIDGED |
| `rmsNorm_smul_eps` | `elephantine/universal.UNorm`; `elephantine/emulate.plan_construction` (host stream = `a` × target stream, `rms_norm_eps = a² ε d/D`) | `test_rmsNorm_smul_eps`, `test_rmsNorm_smul_eps_in_emulate` | BRIDGED |
| `rmsNorm_smul_eps` / `stream_scalar_needs_eps` | `approx/convert.Converter.run` + `approx/plan.py` ("layer_scalar ... EXACT", student eps = teacher eps) | `test_mismatch_approx_layer_scalar_keeps_eps` | **MISMATCH** (W2-1); fixed on r3-nonrelation (`approx.convert.eps_plan`: student eps = eps h_t/(h_s max C²) plus a constant eps channel giving each layer eps/C_l²; tests/test_approx.py) |
| `sigmoid_router_limit` | `elephantine/universal.UFFN` sigmoid branch (`score_c = 1`, renormalised) with the router bias at −s | `test_sigmoid_router_limit` | BRIDGED (setting) |
| `tied_sub_untied` | `elephantine/emulate.build_host` (tied target, untied host: `lm_head := E`) | `test_tied_sub_untied` | BRIDGED |
| `sink_host_factor` | `attn/attention_sink.sink_attention`, `emulate_with_null_token` | `test_sink_host_factor` | BRIDGED |
| `sink_host_factor` / `sink_host_ne_plain` | `graph/model_graph.SINK_ORDER["no"] = ("yes", "fold", True)`; `approx/plan.py` ("sinks ... EXACT, sink = −1e4"); `attn/attention_sink` docstring ("b → −inf (free)") | `test_mismatch_sink_fold_is_a_limit` | **MISMATCH** (W2-2) |
| `bias_free_ffn_zero` | `block/marker_stream.is_bias_free` and `marker_host` (batch 3): the hidden-padded Llama host (`elephantine/hidden_width.widen`) with a start token embedded on a pad channel. `hf/bos_constructions.sink_host` (biased gpt-oss experts) still cancels the marker layer by layer instead | `test_bias_free_ffn_zero` (stock LlamaMLP, six activations), `test_bias_free_ffn_zero_marker_stream` (the marker's residual is exactly κ e_d at every layer), `test_condition_bias_free_ffn_zero_needs_bias_free` (batch 3) | BRIDGED (batch 3) |
| `two_mul_sigmoid_zero`, `ffn_smul`, `exp_mul_sigmoid_shift`, `rmsNorm_neg`, `rmsNorm_one_dim` | (helper lemmas; `ffn_smul` is exercised inside `test_moe_shared_gate_off`) | n/a | not counted |
| `stream_scalar_needs_eps`, `softmax_router_ne_sigmoid`, `router_bias_not_bias_free`, `sink_host_ne_plain`, `biased_proj_not_bias_free`, `knorm_not_plain`, `vnorm_not_plain` | (converse / impossibility) | `stream_scalar_needs_eps`: `test_mismatch_approx_layer_scalar_keeps_eps`; `sink_host_ne_plain`: `test_mismatch_sink_fold_is_a_limit` | not counted |

### PosBias.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `nope_sub_alibi` | `elephantine/universal.UAttention.alibi` at slope 0 | `test_nope_sub_alibi` | BRIDGED |
| `nope_sub_learned_abs` | `elephantine/universal.UniversalElephantine.pos` (learned-abs table) at 0 | `test_nope_sub_learned_abs` | BRIDGED |
| `alibi_sub_additive_bias` | `elephantine/universal.UAttention.alibi` (the ramp table, `s += m_h (j − i)`; loaded by `from_bloom`) | `test_alibi_sub_additive_bias` | BRIDGED |
| `nope_sub_additive_bias` | `elephantine/universal.UAttention.alibi` at 0 (the only additive-bias branch the repo has) | `test_nope_sub_additive_bias` | BRIDGED |
| `t5_sub_additive_bias` | `attn/posbias_hosts.t5_bucket` / `t5_bias_table` / `bias_attention` (batch 3): the additive-bias head with T5's bucketed table, weights from a stock `T5Attention` | `test_t5_sub_additive_bias` (batch 3: against `T5Attention`'s forward, encoder and decoder buckets) | BRIDGED (batch 3) |
| `alibi_sub_learned_abs_causal` | `attn/posbias_hosts.alibi_learned_abs_host` + `learned_abs_head` (batch 3): table [0; 1; t], q reads the 1, k reads m t / scale. `universal.from_bloom` still uses native ALiBi slopes | `test_alibi_sub_learned_abs_causal` (batch 3, BLOOM's `build_alibi_tensor` slopes, causal and full key sets) | BRIDGED (batch 3) |
| `bias_sub_learned_abs` | `attn/posbias_hosts.onehot_learned_abs_host` + `learned_abs_head` (batch 3): one-hot position channels, k reads Bᵀ e_j | `test_bias_sub_learned_abs` (batch 3: random tables, bidirectional ALiBi, a stock T5 table) | BRIDGED (batch 3) |

### Converses.lean

All of these are converses (a condition the construction needs). They are not counted. Most already have a condition test in `tests/test_lean_bridges.py`.

| theorem | condition test | status |
|---|---|---|
| `hard_select_needs_unit_gate` | `test_condition_top1_unit_gate` (batch 1) | not counted |
| `tied_moe_not_dense` | `test_condition_tied_moe_needs_gates_summing_to_one` (batch 1) | not counted |
| `alibi_host_not_bidirectional`, `alibi_limit_not_bidirectional` | `test_condition_alibi_causal_only` (batch 1) | not counted |
| `content_pair_rotates` | `test_condition_alibi_content_pairs_need_zero_freq` (batch 1) | not counted |
| `sink_needs_zero_freq` | `test_condition_sink_constant_channel_needs_zero_freq` (batch 1) | not counted |
| `swigluoai_gate_out_of_regime`, `swigluoai_up_out_of_regime` | `test_condition_swigluoai_clamp_regime` (batch 1) | not counted |
| `swigluoai_needs_up_bias` | `test_condition_swigluoai_needs_up_bias` (batch 1) | not counted |
| `rmsNorm_eps_inj`, `layerNorm_ne_rms_other_eps` | `test_condition_layernorm_fold_needs_every_writer_centred` (its ε case, batch 1); also `test_mismatch_approx_hidden_pad_keeps_eps` here | not counted |
| `qknorm_out_of_regime` | `test_condition_qknorm_out_of_regime` (new) | not counted |
| `gate_pos` | (helper lemma) | not counted |

### Leftovers in Norm.lean, Position.lean, Common.lean

| theorem | status |
|---|---|
| `mirror_left`, `mirror_right` (Norm), `padVec_left`, `padVec_right` (Position) | helper lemmas, not counted |
| `gives` (Common) | not a theorem. The extraction regex matched the phrase "The same lemma gives" in `Common.lean`'s module docstring. |

### ActLimits.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `relu2_xielu_bound` | `act/exotic_activations.xielu`, `act/activation_cross.xielu` with the act-swap fold a = 1/(α_p c²) | `test_relu2_xielu_bound` | BRIDGED |
| `relu2_sub_xielu` | same (c → ∞) | `test_relu2_sub_xielu` | BRIDGED |
| `relu_mish_bound` | `act/act_swap_agreement.TACTS["mish"]` (F.mish), `act/activation_cross.ACTS["mish"]` | `test_relu_mish_bound` | BRIDGED |
| `relu_sub_mish` | same (β → ∞) | `test_relu_sub_mish` | BRIDGED |
| `relu_sub_swigluoai` | `act/exotic_activations.relu_swigluoai_rows` (helper: gate c z, up a z − 1, down 1/(aK)) on the universal host's gpt-oss unit `elephantine/universal.UFFN._act` (clip_lim / up_shift / 1.702 fold as `from_gpt_oss` sets them) | `test_relu_sub_swigluoai`, `test_condition_relu_sub_swigluoai_needs_bounded_inputs`, `test_swigluoai_host_unit_is_the_lean_unit` | BRIDGED (helper) |
| `relu_sub_swigluoai` | `act/activation_cross.fit_minimax` / `act/act_swap_agreement.FitAct` for (ReLU, swigluoai), cached in `act/data/act_fits_cache.json` | `test_relu_on_swigluoai_fit_takes_the_up_bias_route`, `test_swigluoai_scaled_gate_vanishes` | **MISMATCH** (W3-1); fixed on r3-nonrelation (`activation_cross.fit_host` takes `relu_sub_swigluoai`; the graph's ACT_LIMIT condition is in verify/r3_notes.md) |
| `swigluoai_scaled_gate_vanishes` | (refutation of the naive route) checked on `UFFN._act` | `test_swigluoai_scaled_gate_vanishes` | not counted (refutation; it is what MISMATCH W3-1 runs into) |
| `xielu_ne_relu2` | (impossibility) | n/a | not counted |
| `softplus_ge`, `softplus_pos`, `softplus_le_exp`, `tanh_lt_one'`, `one_sub_tanh_le`, `mul_exp_neg_le`, `swK_pos` | (helper lemmas) | n/a | not counted |

### ActWidth.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `shiftfree_odd_part` | `act/act_width_fit.units` on every stored shift-free fit (`act/data/act_width_fits.json`, hosts SiLU / GeLU / swigluoai) | `test_shiftfree_odd_part` | BRIDGED |
| `silu_odd_pair` | `act/act_width_fit.TA["SiLU"]` (also GeLU, swigluoai) | `test_silu_odd_pair` | BRIDGED |
| `odd_obstruction` | the stored shift-free fits (any swish host) | `test_odd_obstruction` | BRIDGED |
| `shiftfree_silu_obstruction` | the stored `T\|SiLU\|R\|noshift` fits; as a floor on the stored mish errors (`graph/act_width`'s "0.10 on [−160, 160]") | `test_shiftfree_silu_obstruction`, `test_shiftfree_silu_obstruction_floors_the_stored_mish_fits` | BRIDGED |

### ActFits.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `actFit_mish_silu6` | `act/act_width_fit.units` / `evaluate` with the stored fit `"mish\|SiLU\|6\|shift"`, k = 6 | `test_actFit_mish_silu6` | BRIDGED |
| `actFit_mish_silu6_explicit` | same; the stored α, β round to the Lean integers at 2⁻²⁴ and γ at 2⁻⁴⁰, every one of the 18 | `test_actFit_mish_silu6_explicit` | BRIDGED |
| `actFit_expPos`, `actFit_expEnc`, `actFit_mish_eq`, `actFit_chunk`, `actFit_group0`–`actFit_group7`, `actFit_chunks`, `actFit_stepE`, `actFit_stepAll`, `actFit_cell`, `actFit_run`, `actFit_init`, `actFit_allBelow`, `actFit_hasDerivAt_*`, `actFit_sig*`, `actFit_g*`, `actFit_h*`, `actFit_tay*`, `actFit_sq*`, `actFit_z*`, `actFit_in_*`, `actFit_cdiv_*`, `actFit_div_le`, `actFit_S`, `actFit_S_pos`, `actFit_ble`, `actFit_X`, `actFit_unit_*`, `actFit_derE`, `actFit_mish_val`, `actFit_mish_der`, `actFit_sum`, `actFit_next_*`, `actFit_sigmoid_eq` (71) | (the interval checker's lemmas) | n/a | not counted |

### LimitInstances.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `relu_gelu_bound` | `act/activations.gelu_b`, `act/act_swap_agreement.TACTS["GeLU"]` | `test_relu_gelu_bound` | BRIDGED |
| `actLayer_dist` | `act/act_swap_agreement.act_slots` / `install` / `FitAct` (the GeLU ← SiLU single-op fit of `act_fits_cache.json`) in a stock OPT decoder layer | `test_actLayer_dist`, `test_silu_layer_bound` | BRIDGED |
| `act_swap_model_bound` | `act/activations._act_fn("swish", 1.702)` vs `activations.gelu` (the swap `run_gpt2` measures), toy stack | `test_act_swap_model_bound` | BRIDGED |
| `act_swap_model_bound_on` | `FitAct("SiLU", act_fits_cache 'GELUActivation[]\|SiLU')`, toy stack, region checked | `test_act_swap_model_bound_on` | BRIDGED |
| `act_swap_greedy` | through its instances (`relu_*_greedy` below) | `test_relu_silu_greedy` | BRIDGED |
| `act_swap_greedy_on` | `FitAct` GeLU ← SiLU fit, `act_swap_agreement.greedy` on a toy LM | `test_act_swap_greedy_on` | BRIDGED |
| `act_swap_model_limit`, `act_swap_model_limit_on` | through the instances (`FitAct(host, (1/β, β, 0, 0, 0))`, `FitAct("xielu", …)`) | `test_relu_silu_model_limit`, `test_relu2_xielu_model_limit` | BRIDGED |
| `relu_mish_model_bound`, `relu_gelu_model_bound` | `FitAct("mish" / "GeLU", (1/β, β, 0, 0, 0))` in a toy ReLU-MLP stack | `test_relu_mish_model_bound`, `test_relu_gelu_model_bound` | BRIDGED (2 rows) |
| `relu_mish_greedy`, `relu_gelu_greedy` | same, `act_swap_agreement.greedy`, β = 1.01 β* | `test_relu_silu_greedy` (parametrised over SiLU / mish / GeLU) | BRIDGED (2 rows) |
| `relu_mish_model_limit`, `relu_gelu_model_limit` | same, β = 10^k | `test_relu_silu_model_limit` | BRIDGED (2 rows) |
| `relu2_xielu_model_bound` | `FitAct("xielu", (1/(α_p c²), c, 0, 0, 0))` (TACTS xIELU, default init), toy relu² stack, region checked | `test_relu2_xielu_model_bound` | BRIDGED |
| `relu2_xielu_greedy` | same, c = 1.01 max(c*, region c) | `test_relu2_xielu_greedy` | BRIDGED |
| `relu2_xielu_model_limit` | same, c = 10^k | `test_relu2_xielu_model_limit` | BRIDGED |
| `final_softcap_greedy` | `attn/softcap.softcap` (the expression of `UniversalElephantine.final_cap`), `act_swap_agreement.greedy` | `test_final_softcap_greedy` | BRIDGED (note W3-a) |
| `gauss_tail_abs`, `gauss_cdf_neg_le`, `gauss_one_sub_cdf_le`, `mul_tail_le`, `reluLayer_eq_actLayer`, `sum_eta_eq`, `wsum_nonneg`, `tendstoLocallyUniformly_congr_eventually`, `inv_mul_lt_of_lt`, `model_const_nonneg`, `xielu_host_close`, `isPick_comp_strictMono`, `tanh_strictMono`, `softcap_strictMono`, `greedy_comp` | (helper lemmas) | n/a | not counted |

### ModelLimit.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `silu_layer_bound` | `act_swap_agreement.install` + `FitAct("SiLU", (1/β, β, 0, 0, 0))` in a stock OPT decoder layer (ReLU) | `test_silu_layer_bound` | BRIDGED |
| `relu_silu_model_bound` | `FitAct("SiLU", …)` in a toy ReLU-MLP stack (l2 states, sup logits, L_i = ‖A‖ + ‖W‖‖P‖) | `test_relu_silu_model_bound` | BRIDGED |
| `relu_silu_model_limit` | same, β = 10^k | `test_relu_silu_model_limit` | BRIDGED |
| `relu_silu_greedy` | same, `act_swap_agreement.greedy`, β = 1.01 β* | `test_relu_silu_greedy` | BRIDGED |
| `pick_eq_of_margin`, `greedy_preserved`, `greedy_of_bound` | `act/act_swap_agreement.greedy` (argmax pick, KV-cache fallback path) | `test_greedy_preserved` | BRIDGED (3 rows) |
| `attn_perturb` | `attn/qk_norm.attend` | `test_attn_perturb` | BRIDGED |
| `rope_alibi_layer_bound` | `attn/positional_cross.alibi_rope_host` + `rope_score`, softmax `attn/qk_norm.attend` | `test_rope_alibi_layer_bound` | BRIDGED |
| `alibi_rope_model_bound` | same, toy stack (constant q, k; values W tanh x) | `test_alibi_rope_model_bound` | BRIDGED |
| `alibi_rope_model_limit` | same, θ = 10^−k | `test_alibi_rope_model_limit` | BRIDGED |
| `alibi_rope_greedy` | same, `act_swap_agreement.greedy` on a toy LM over the last T tokens | `test_alibi_rope_greedy` | BRIDGED |
| `model_limit`, `model_limit_pointwise`, `model_limit_compact`, `model_limit_head`, `model_error_bound`, `errBound_le_prod_sum`, `model_error_bound_head` | (generic composition kit; no repo function computes errBound) — every instance test above goes through them | n/a | not counted (kit) |
| `tie_flips` | (counterexample: no guarantee at margin 0) | n/a | not counted |
| `continuous_run`, `tendsto_layer_comp`, `continuous_relu`, `attn_weight_le`, `attn_eq_weights`, `causalSet_nonempty`, `exp_two_mul_sub_one_le` | (helper lemmas) | n/a | not counted |

### LimitKit.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `model_bound_prod`, `model_error_bound_on`, `model_bound_on_prod`, `greedy_of_layer_bounds`, `greedy_of_layer_bounds_on`, `model_limit_on`, `model_limit_of_rates` | (generic kit; exercised by `test_act_swap_model_bound_on`, `test_act_swap_greedy_on`, `test_relu2_xielu_*`) | n/a | not counted (kit) |
| `one_le_prod_range`, `one_le_prod_finset`, `errBound_nonneg`, `errBound_le_prod_sum_of_le`, `tendsto_prod_sum_rates` | (helper lemmas) | n/a | not counted |

The toy stacks: the whole-model theorems take arbitrary layers with Lipschitz constants, value bounds and regions as hypotheses. No real transformer layer is globally Lipschitz (softmax attention, unbounded states), so the whole-model statements are checked on small residual MLP / ALiBi stacks whose constants are computable, with the repo's own swap module (`FitAct`, as `act_swap_agreement` installs it), the repo's frozen-pair host and the repo's greedy routine. The per-layer statements (`silu_layer_bound`, `actLayer_dist`) are checked on stock OPT decoder layers through `act_slots` / `install`.

### EpsLever.lean

The construction is `hf/eps_lever_agreement.host_weights` (branch measure-batch2, copied here): a host norm the target lacks reads its input ×2^-a (the writing projection's rows) and gets gain 2^a √E. That is `EP.epsLeverNorm` with S = 2^a, ε = E (the host's own ε > 0), w = 1. `verify/levers/eps_lever.py` (branch frontier-levers, not copied) builds the same lever with the residual-scaled split (residual ×S_res, E = ε_t S_res²). `graph/eps_lever.py` (measure-batch2, copied) picks a.

| theorem | repo function | test | status |
|---|---|---|---|
| `epsLever_eq` | `hf/eps_lever_agreement.host_weights` (q_proj, k_proj ×2^-a; q_norm, k_norm = 2^a √E), stock `Qwen3RMSNorm` | `test_epsLever_eq`, `test_epsLever_eq_host_weights` | BRIDGED |
| `epsLever_eq` | `graph/eps_lever.EXACT = True`, `RHO_EXACT`, `PROPOSED` ("EXACT, free") | `test_mismatch_eps_lever_exact_flag` | **MISMATCH** (W4-1) |
| `epsLever_coord_bound` | the levered norm, S √ε · `attn/qk_norm.rmsnorm`(x/S) · w | `test_epsLever_coord_bound` | BRIDGED |
| `epsLever_bound` | same | `test_epsLever_bound` | BRIDGED |
| `epsLever_tendsto`, `epsLever_uniform` | `graph/eps_lever.a_needed` (S chosen per ball ms ≤ ms_max) | `test_epsLever_uniform` | BRIDGED (2 rows) |
| `leverAttn_layer_bound` | `hf/eps_lever_agreement.host_weights` / `host_config` / `write_host` (Llama target, stock Qwen3 host, qk-norm off), layer 0 attention | `test_leverAttn_layer_bound` | BRIDGED |
| `postNorm_layer_bound` | `hf/eps_lever_agreement.host_weights` (Gemma target, Gemma-2 host: o_proj ×2^-a, post-norm gain 2^a √E), stock `Gemma2RMSNorm` | `test_postNorm_layer_bound` | BRIDGED |
| `leverAttn_model_bound`, `leverAttn_model_limit`, `postNorm_model_bound`, `postNorm_model_limit`, `lever_model_bound`, `lever_model_limit` | the whole stock hosts from `hf/eps_lever_agreement` (Qwen3 ← Llama, Gemma-2 ← Gemma) | `test_leverAttn_model_bound_and_postNorm_model_bound` | BRIDGED (6 rows). The Lipschitz constants are hypotheses, so the test checks the bound's form: gap × S² constant from a = 6 to a = 8 (1/S² rate, within 10 %), and the gap at the float32 floor at a = 22. |
| `leverAttn_greedy`, `postNorm_greedy`, `lever_greedy` | same, with `act/act_swap_agreement.greedy` | `test_leverAttn_greedy_and_postNorm_greedy` | BRIDGED (3 rows) |
| `leverAttnStack_layer` | (kit bookkeeping: per-layer error P_i · const / S², P_i a hypothesis) | n/a | not counted |
| `leverU_nonneg`, `leverFactor_pos`, `leverFactor_le_one`, `one_sub_leverFactor_le`, `leverU_le`, `tendsto_const_div_sq`, `optLever_eq_smul`, `optFactor_pos`, `optFactor_le_one`, `one_sub_optFactor_le`, `epsLever_attn_value`, `lever_sum_div`, `leverAttnConst_nonneg`, `postNorm_layer_on` | (helper lemmas) | n/a | not counted |

### LimitInstances2.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `attn_extra_keys_bound` | `attn/qk_norm.attend` (S = causal) vs the band of `hf/local_full.band_mask` (S′) | `test_attn_extra_keys_bound` | BRIDGED |
| `mask_layer_bound` | `hf/local_full.band_mask`, clamped to −B (band_mask itself is the B = finfo.max end point) | `test_mask_layer_bound` | BRIDGED |
| `sink_layer_bound` | `attn/attention_sink.sink_attention` | `test_sink_layer_bound` | BRIDGED |
| `cap_layer_bound` (+ `cap_const_le`'s form) | `attn/softcap.softcap`, `attn/qk_norm.attend` | `test_cap_layer_bound` | BRIDGED |
| `soft_router_layer_bound` | `moe/dense_emulates_moe.dense_softrouter` vs `moe_top1` | `test_soft_router_layer_bound` | BRIDGED |
| `mask_model_bound`, `mask_model_limit` | `hf/local_full` (stock 2-layer Qwen2Model, full config + band clamped to −B, vs the sliding-window config) | `test_mask_model_bound` | BRIDGED (2 rows). e^{−B} form: gap × e^B does not grow from B = 10 to 14; float32 floor at B = 60. |
| `mask_greedy` | `block/limit_stacks.AttnStack.mask_host_layer` (batch 3: `hf/local_full.band_mask` clamped at −B over all T keys, every layer of an n-layer one-head stack) with `SeqEmbed`, `greedy`, `lipschitz` | `test_mask_greedy` (batch 3: at B = B* + 0.5 with measured Lh, L_i, R, δ the host's greedy tokens are the target's) | BRIDGED (batch 3) |
| `sink_model_bound`, `sink_greedy`, `sink_model_limit` | `block/limit_stacks.AttnStack.sink_host_layer` (batch 3: `attn/attention_sink.sink_attention` in every layer) | `test_sink_model_bound`, `test_sink_greedy`, `test_sink_model_limit` (batch 3) | BRIDGED (batch 3, 3 rows) |
| `cap_model_bound`, `cap_greedy`, `cap_model_limit` | `block/limit_stacks.AttnStack.cap_host_layer` (batch 3: `attn/softcap.softcap` in every layer) | `test_cap_model_bound` (both the (4/3) R³ M / c² form and the exp form), `test_cap_greedy`, `test_cap_model_limit` (1/c² rate) (batch 3) | BRIDGED (batch 3, 3 rows) |
| `soft_router_model_bound`, `soft_router_greedy`, `soft_router_model_limit` | `block/limit_stacks.RouterStack` (batch 3: `moe/dense_emulates_moe.dense_softrouter` vs `moe_top1` in every layer of a per-token stack) | `test_soft_router_model_bound`, `test_soft_router_greedy`, `test_soft_router_model_limit` (batch 3) | BRIDGED (batch 3, 3 rows) |
| `vanishing_carry_rate` | `moe/dense_host_prenorm.HostNorm` + `Side` (branch moe-loop-rescue, copied): the stream read and the carried read `c / side.r` ×1/δ | `test_vanishing_carry_rate` | BRIDGED |
| `scaled_read_layer_bound` | `HostNorm`'s stream read (λ = rms_t / rms_host) into the compensator-form consumer x + A y | `test_scaled_read_layer_bound` | BRIDGED |
| `scaled_read_model_bound`, `scaled_read_greedy`, `scaled_read_model_limit` | `block/limit_stacks.ReadStack` (batch 3): every layer's read through the padded host norm over [x; δ s_i] (`F.rms_norm`, ε d/(d+e), the algebra of `moe/dense_host_prenorm.HostNorm`), consumer x + A_i y; ρ = δ²‖s‖²/(2dε) from `vanishing_carry_rate` | `test_scaled_read_model_bound`, `test_scaled_read_greedy`, `test_scaled_read_model_limit` (δ² rate) (batch 3) | BRIDGED (batch 3, 3 rows) |
| `compensator_model_bound` | `block/limit_stacks.CompensatorStack` (batch 3): reads u_i(x) t/κ through a host norm over [x; κ], times a least-squares polynomial g(t) ≈ κ/t on the pad range (`fit`, ρ its sup relative error). The 1-D compensator fits of real models (`fitlib.py` etc.) are in `verify/prenorm_rescue/` | `test_compensator_model_bound` (batch 3, fit degrees 1 to 8) | BRIDGED (batch 3) |
| `mix_bound`, `attn_norm_le`, `resid_dist_le`, `attn_singleton`, `sum_const_range`, `cap_const_le`, `one_sub_ratio_sqrt` | (helper lemmas) | n/a | not counted |

### PreNorm.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `prenorm_readout_obstruction`, `prenorm_readout_eps_pos`, `channel_carry_prenorm_fails`, `parallel_sub_seq_prenorm_fails`, `cla_carry_prenorm_fails` | (impossibility; the failing carries are measured in `block/prenorm_carry.py` and checked by the first batch's `test_condition_parallel_sub_seq_needs_norm_free`) | n/a | not counted |
| `sum_sq_pos_of_ne_zero`, `append_smul_split` | (helper lemmas) | n/a | not counted |

### PreNormRescue.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `vanishing_carry_limit` | `moe/dense_host_prenorm.HostNorm` (δ → 0, down to the construction's GAMMA = 1e-8) | `test_vanishing_carry_limit` | BRIDGED |
| `const_norm_carry` | `norm/const_norm.const_norm_host` / `const_norm_eps` / `carried_factor` (batch 3): weight [√(d/(d+e)) w; wz], ε′ = (dε − S)/(d+e), on torch's `F.rms_norm` | `test_const_norm_carry`, `test_condition_const_norm_carry_needs_fixed_norm` (batch 3) | BRIDGED (batch 3) |
| `moe_select_prenorm_exact` | `moe/exact_select_prenorm.build_exact_select_host` (batch 3): residual [x; g; κ], one RMSNorm with ε′ = (dε − S)/(d+E²+1), a clamped-pair select ReLU FFN and a masked-expert / erase ReLU FFN, all with biases; `target_moe` is the ReLU top-1 target. (`moe/dense_host_prenorm` remains the LIMIT variant) | `test_moe_select_prenorm_exact` (all three conclusions), `test_condition_moe_select_needs_strong_margin` (batch 3) | BRIDGED (batch 3) |
| `pairSel_sq_sum` | `moe/exact_select_prenorm.pair_sel` / `clamp_u` / `strong_margin` (batch 3) | `test_pairSel_sq_sum` (batch 3) | BRIDGED (batch 3) |
| `rmsNorm_zero_sq_sum` | `norm/const_norm.glu_const_writer` (batch 3): 2m bias-free SiLU GLU units in a stock LlamaMLP writing Σ u_i² = m after an ε = 0 norm | `test_rmsNorm_zero_sq_sum` (batch 3) | BRIDGED (batch 3) |
| `eps_zero_no_equaliser`, `no_bounded_analytic_equaliser`, `prenorm_write_analytic_bounded`, `no_prenorm_rms_equaliser`, `const_readout_channel`, `no_prenorm_homogeneous_channel` | (impossibility / algebra behind the impossibilities) | n/a | not counted |
| `rmsNorm_zero_smul`, `analyticAt_coord`, `analyticAt_affine_coord`, `AnalyticAt.sqrt_of_pos`, `analytic_const_of_open`, `rmsNorm_abs_le`, `clampU_nonneg`, `clampU_of_nonpos`, `clampU_of_ge`, `pairSel_pair_sq`, `pairSel_winner`, `pairSel_loser`, `masked_experts`, `pairFlat_sq_sum` | (helper lemmas) | n/a | not counted |

### RandomFeatures.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `rf_unbiased` | `attn/linear_softmax.favor_features` (Gauss–Hermite expectation over the feature rows) | `test_rf_unbiased` | BRIDGED |
| `rf_mse`, `rf_variance` | same, m-row W, product quadrature over the m rows | `test_rf_mse_and_variance` | BRIDGED (2 rows) |
| `rf_chebyshev` | same, Monte Carlo with Hoeffding slack | `test_rf_chebyshev` | BRIDGED |
| `gauss_exp_integrable`, `gauss_exp_integral`, `sum_sel`, `rf_feature`, `exp_mul_exp_sum`, `rfCoef_sq`, `rfCoef_add_sq`, `sq_sum_add`, `rfEst_eq`, `rfEst_measurable`, `rfEst_memLp` | (helper lemmas; `gauss_exp_integral` is exercised by `test_rf_unbiased`) | n/a | not counted |

### KVPack.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `kvPack_rope_score` | `verify/levers/constructions.kv_pack` (copied from branch frontier-levers), checked through the head output | `test_kvPack_rope_attn` | BRIDGED |
| `kvPack_rope_attn` | `verify/levers/constructions.kv_pack`: two RoPE KV heads in one host KV head of width 2 d_h, frequencies repeated per slice, 1/√(2d_h) folded into q | `test_kvPack_rope_attn` | BRIDGED |
| `kvPack_nope` | `kv_pack(nope=True)` (helper keyword) | `test_kvPack_nope` | BRIDGED (helper) |
| `kvPack_pair_iff` | `verify/levers/constructions.rope` (the half-split rotary `kv_pack` uses), ω vs ω + 2πn | `test_kvPack_pair_iff` | BRIDGED |
| `kvPack_rope_iff` | `kv_pack` with the matching table (exact) and with one slice pair's frequency moved (`inv_host=`, helper keyword): not exact | `test_kvPack_rope_attn`, `test_kvPack_rope_iff` | BRIDGED (helper) |
| `kvPack_default_fails`, `kvPack_default_not_exact` | (impossibility) `kv_pack` with the host's own default schedule over 2 d_h is off | `test_kvPack_default_not_exact` | not counted |
| `kvPack_pairScore_zero`, `kvPack_score_slices`, `kvPack_attn_apply`, `kvPack_cos_sin_nat`, `kvPack_cos_sin_int`, `kvPack_defaultFreq_mem`, `kvPack_defaultFreq_inj` | (helper lemmas) | n/a | not counted |

Note. The model builders do not pack: `elephantine/emulate.head_maps` (v1) tiles KV heads and raises `Refused` when packing would be needed (`test_condition_kvPack_emulate_refuses_packing`), while `graph/model_graph.kv_heads_fit` pays packed edges (a longrope per-block frequency table). The only code that builds the packed head is the frontier-levers check. No whole-model packed host is built anywhere.

### MLARope.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `mla_rope_sub_mha_partial` | `elephantine/universal._mla_from` (via `from_deepseek`, stock DeepseekV2ForCausalLM): pass pairs at inv_freq 0, the target's frequencies on the rope pairs, every head's rope key rows = the shared `kv_a_proj` rope rows reading x, content key and value rows reading the latent; logits equal the stock model's | `test_mla_rope_sub_mha_partial` | BRIDGED (note b) |
| `stackPairs_inl`, `stackPairs_inr` | (helper lemmas) | n/a | not counted |

Note b. The theorem is norm-free. `_mla_from` also carries the latent RMSNorm (`kv_a_layernorm`, and the q-latent norm), on the universal host's normed-latent input branch (`pn`, `n_avg`, `n_w`), so the host head is "MHA over [x; normed latent]", not a plain MHA head. The per-head layout is the theorem's. That a plain MHA cannot carry the norm is `mla_latent_norm_not_mha` (batch 1).

### MLAHost.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `mla_host_sphere_full_rank` | `attn/mla_host.write_mla_value_host` / `mla_value_matrix` / `single_token_matrix` (batch 3): a stock DeepseekV3Attention written with an isometric latent down-projection (W^T W = α I, rank ≥ width), latent norm weight w and M = (D/α) A Wᵀ diag(1/w) through kv_b_proj and o_proj. The zoo still has no ε = 0 target, so no loader uses it | `test_mla_host_sphere_full_rank` (stock Llama GQA target vs the stock MLA host on the sphere), `test_condition_mla_host_needs_the_sphere` (batch 3); batch 2's `test_mla_host_sphere_full_rank_no_code` still holds (it pins `elephantine.universal` only) | BRIDGED (batch 3) |
| `mla_value_ray`, `mla_host_not_prenorm_target`, `mla_host_sphere_rank` | (impossibility) | n/a | not counted |
| `rmsNorm_smul` | (helper lemma) | n/a | not counted |

### MoERouting.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `moeR_copies_topk` | `serving/embed.embed_state` (r copies of each router row at index c·E_A + e, dummies reading −M on the free dim), selection by `torch.topk` | `test_moeR_copies_topk` | BRIDGED |
| `moeR_dup_renorm` | `serving/embed.embed_state` (whole zero-padded copies: each copy's output is Y_e), norm_topk_prob = True; also on the stock `Qwen3MoeSparseMoeBlock` | `test_moeR_dup_renorm`, `test_moeR_dup_on_stock_block` | BRIDGED |
| `moeR_dup_renorm`, `moeR_copies_topk` | `serving/embed` docstring / `plan()`: "k_B = r k_A is exact", no strictness condition | `test_moe_dup_refuses_tied_router_rows`, `test_moe_dup_accepts_distinct_router_rows` | **MISMATCH** (W5-1); fixed on r3-nonrelation (docstring states the strict top-k; `embed_state` refuses tied router rows, `ties="warn"` embeds with a warning) |
| `moeR_dup_plain_factor` | `serving/embed.embed_state`, norm_topk_prob = False, with dummies (free-dim value ρ drawn so that Z_D ranges from O(1) to underflow) | `test_moeR_dup_plain_factor` | BRIDGED |
| `moeR_dup_plain_extras_bound` | same | `test_moeR_dup_plain_factor` | BRIDGED |
| `moeR_dup_plain` | `serving/embed.embed_state` with E_B = r E_A (no dummies); stock block | `test_moeR_dup_plain`, `test_moeR_dup_on_stock_block` | BRIDGED |
| `moeR_dup_plain_extras_ne` | `serving/embed` dummies: exact only through exp underflow (−1e8 on the free dim; real-valued it is the limit Z_D → 0), as the theorem says. The docstring says "exactly 0 in fp32", which is a floating-point statement and true. | `test_moeR_dup_plain_factor` (the factor < 1 at small ρ) | not counted (refutation of exactness over ℝ) |
| `moeR_scaling_fold`, `moeR_scaling_fold_plain` | `moe/zero_router.deal_dense_into_moe` on a stock DeepseekV3MoE: the probed gate (s_h/k renormalised, s_h/2 not) is divided out of the down-projections, so any routed_scaling_factor gives the dense output | `test_moeR_scaling_fold` | BRIDGED (2 rows) |
| `moeR_group_copies` | `moe/group_dup.group_dup_moe` / `dup_index` (batch 3): a stock DeepseekV3MoE host with m copies per expert, copy r of group g its own host group (n_group G m, topk_group tg m, top-k k m). `serving/embed` still handles only Qwen2/3-MoE | `test_moeR_group_copies` (batch 3: probed selections are exactly the copies; outputs agree, renormalised and plain gates) | BRIDGED (batch 3) |
| `moeR_dup_needs_divisibility`, `moeR_dup_needs_experts`, `moeR_renorm_host_needs_const_mass`, `moeR_plain_host_needs_const_mass`, `moeR_mass_varies` | (necessity / impossibility) all enforced by `serving/embed.plan` (k_B multiple of k_A, E_B ≥ (k_B/k_A) E_A, norm_topk_prob equal) | `test_moeR_dup_needs_divisibility_and_experts_enforced` | not counted |
| `moeR_tie_split`, `moeR_group_top2sum_dup_flips` | (counterexamples) | `test_mismatch_moe_dup_splits_copies_at_a_tie` | not counted |
| `moeR_topk_unique`, `moeR_card_copies`, `moeR_mem_copies_inl`, `moeR_not_mem_copies_inr`, `moeR_sum_copies`, `moeR_renormOut_smul`, `moeR_plainOut_smul`, `moeR_plain_eq_mass`, `moeR_prod_top` | (helper lemmas) | n/a | not counted |

### LearnedAbs.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `learned_abs_two_block` | `hf/bos_constructions.write_bos_prefix` / `prefix_check` (copied from branch bos-token): stock bias-free LlamaForCausalLM, NoPE, width d + 5; L = 0 (the Lean's zero-logit head) and the default L = log T; ReLU, SiLU and GELU units | `test_learned_abs_two_block` | BRIDGED (note c) |
| `padded_prenorm_sublayer` | `hf/bos_constructions.learned_abs_host` (target layers at width d + 5, weight √(d/(d+5))·w, ε·d/(d+5)), layer by layer | `test_padded_prenorm_sublayer` | BRIDGED |
| `padded_prenorm_stack` | same, every layer | `test_padded_prenorm_sublayer` | BRIDGED |
| `learned_abs_model` | `hf/bos_constructions.learned_abs_host` on a random GPT-2 (logits) | `test_learned_abs_model` | BRIDGED |
| `sink_uniform_position` | `attn/attention_sink.sink_attention` (zero logits, sink logit 0, constant value) | `test_sink_uniform_position` | BRIDGED (head output only; the sink-based learned-abs construction is not built, as PROOFS says) |
| `PresConst.pointwise`, `PresConst.comp`, `PresConst.residual`, `PresConst.attention`, `PresConst.stack`, `no_learned_abs_without_marker` | (impossibility) stock LlamaModel with RoPE keeps a constant sequence constant | `test_no_learned_abs_without_marker` | not counted |
| `hostVec_left`, `hostVec_right`, `hostVec_ext`, `hostVec_add`, `hostVec_smul`, `hostVec_sum`, `dot_hostVec`, `sumsq_hostVec`, `padVec_eq_hostVec`, `rmsNorm_one`, `relu_mul_of_nonneg`, `relu_sub_relu_neg`, `reluMLP_smul`, `reluMLP_sum_elim`, `reluMLP_prod`, `posKnot_strictMono`, `posKnot_nonneg`, `posKnot_lt_one`, `pwl_apply`, `reluNet_posKnot`, `bosFlag_nonneg`, `bosFlag_sq_le`, `padS_sq`, `padC_pos`, `hostEmb_sumsq`, `stream_sumsq`, `rms_stream`, `rms_arg_pos`, `sum_smul_eCh`, `sum_smul_content`, `contentOf_padVec`, `attn_const_value` | (helper lemmas) | n/a | not counted |

Note c. The code generalises the Lean construction in three documented ways, none of which changes the statement: the marker logit L (u_t = e^L/(e^L + t); L = 0 is the Lean's 1/(t+1)); kink units shared across the d content coordinates (T + 1 + 5 units in block 2 instead of 5 + d(T + 3)); and any ramp-like activation (the knot system is solved exactly, not as a limit). With L = 0 the same code also builds batch 1's `causal_uniform_head_position` head (marked NO CODE there; `hf/bos_constructions.marker_weights(T, 0) = 1/(t+1)`).

### RopeRank.lean

| theorem | repo function | test | status |
|---|---|---|---|
| `partial_rope_too_few`, `partial_rope_too_few_default` | (impossibility) enforced: `hf/partial_rope_host.column_map` raises for rot_h < rot_t | `test_partial_rope_too_few_enforced` | not counted |
| `recOps_sum`, `recOps_rec`, `sinusoid_rec`, `pairScore_key0` | (helper lemmas) | n/a | not counted |

## Batch 2: the MISMATCH rows

- **W1-1. `block/parallel_residual` and `block/prenorm_carry` vs `parallel_sub_seq_2x`.** The module docstring of `block/parallel_residual` ("an L-block parallel model needs (L+1) d"), the docstring of `parallel_stack_in_seq_wide` ("Width 2d is enough for L = 1 only") and the whole-model caveat in `block/prenorm_carry` ("an L-block parallel model needs (L+1) d ... each block's attention write must land outside the span of all earlier writes") say a norm-free sequential host needs (L+1)·d width for L parallel blocks. So does the comment on the seq-host-for-a-parallel-target rule in `graph/model_graph.py` (around line 3792: "costs (L+1) d there, not 2d: Lean parallel_sub_seq_wide"; that rule's decision rests on the pre-norm failure, which stands, so only the comment is wrong; not edited here). The theorem builds it at 2d for every L: the MLP sublayer, which may read and write the whole residual, moves the attention's write out of the second channel with 2d identity units φ(v) − φ(−v) = v (ReLU exactly, SiLU / GeLU too) and erases it, so the second channel is 0 again at every block boundary and can be reused. The argument in the docstrings assumes the attention write can only be removed by later linear readouts, never moved by a sublayer. `parallel_stack_in_seq_wide` is still correct, just (L+1)/2 times wider than needed; `test_mismatch_parallel_residual_width_claim` shows the 2d host exact at L = 2…5. Consequence for the first batch: BRIDGES condition 7's width half ("parallel ⊆ seq needs (L+1)·d width"; "the graph charges ×2 width ... not (L+1)×") is wrong norm-free — the graph's ×2 (`wf = 2`) is right norm-free, at +2d MLP units per block, which the graph does not charge. `test_condition_parallel_sub_seq_width` (one reserved channel reused WITHOUT the move fails) is still true. Under a pre-norm the 2× layout fails (`parallel_sub_seq_prenorm_fails`), so the norm half of condition 7 stands.

- **W2-1. `approx/convert.Converter.run` / `approx/plan.py` vs `rmsNorm_smul_eps` and `stream_scalar_needs_eps`.** `approx/plan.py` lists Gemma-4's per-layer `layer_scalar` as an EXACT fold: "student residual = teacher residual / prod(scalars so far) ... (all sub-blocks start with a scale-invariant RMSNorm)". `Converter.run` builds exactly that, dividing by the cumulative scalar `C_l`. The student keeps the teacher's ε (`approx/real.py`: `s.eps = tc.eps`). With ε > 0, RMSNorm is not scale invariant: `RMSNorm_ε(h/C) = RMSNorm_{εC²}(h)`. So the student's norm misses the teacher's by the factor `√((ms + ε)/(ms + εC²))` at every coordinate. It is exact only if layer l's norms use `ε/C_l²`, a different ε per layer, which one `rms_norm_eps` cannot give. `test_mismatch_approx_layer_scalar_keeps_eps` measures the gap at exactly that factor, and shows that ε/C² closes it. The same converter's "hidden: zero-pad ... every RMSNorm weight × sqrt(h_t/h_s) ... EXACT" row has the sibling defect (Capacity.lean, `hidden_pad_needs_eps`). It keeps ε where ε·d/D is needed (`test_mismatch_approx_hidden_pad_keeps_eps`). Both gaps are relative ε/ms. That is small on real activations but nonzero, so "EXACT" is wrong. `elephantine/emulate.py` does it right: `rms_norm_eps = a² ε d/D` (`test_rmsNorm_smul_eps_in_emulate`).
- **W2-2. The "exact" sink fold vs `sink_host_factor` / `sink_host_ne_plain`.** Three places treat a plain target on a sink host as exact:
  - `graph/model_graph.SINK_ORDER["no"] = [("yes", "fold", True)]` (exact = True);
  - `approx/plan.py` ("sinks ... EXACT", sink = −1e4);
  - `attn/attention_sink`'s docstring ("standard ≥ sink: set b → −inf (free)").

  At every finite sink logit b, the sink host is the plain head times `Z/(e^b + Z) < 1`. It misses a nonzero plain output by exactly `e^b/(e^b + Z)·|o|`, which `test_mismatch_sink_fold_is_a_limit` checks to 1e−9 relative. It matches only in the limit b → −∞. In floating point it matches once `e^b` underflows against Z (b = −1e4, or b = −inf as a stored value). So the edge is a limit that floating point happens to reach exactly, not an exact weight setting. The test does not assert anything about `graph/model_graph.py`, which other branches are editing.

W3-1. **ReLU on a swigluoai host: the repo's fit is the refuted scaled-gate route.** `act/activation_cross.fit_minimax(ACTS["swigluoai"], relu)` (cached as `act/data/act_fits_cache.json` `'ReLU[]|swigluoai'`, and run inside real models by `act/act_swap_agreement.FitAct`) returns `a·X(c x + d) + e x + f` with c ≈ 1767, a ≈ 1/c, d ≈ 0: the gate scaled toward a step, exactly the `relu_sub_silu` route. `ACTS["swigluoai"]` / `TACTS["swigluoai"]` is the UNCLAMPED `z σ(1.702 z)`, on which the fit is within 1e−6 of ReLU. The real gpt-oss unit clamps the gate at 7 (`elephantine/universal.from_gpt_oss`: clip_lim [1.702·7, 7]); there the same fit gives `a·K ≈ 0.004` for every x > 0 (K = 7σ(11.914)), off by about x (`swigluoai_scaled_gate_vanishes`; `test_mismatch_relu_on_swigluoai_fit_scales_the_gate` checks the gap ≥ 0.49 on [0.5, 6]). The construction that works on the clamped unit is `relu_sub_swigluoai`: gate c z, up a z − 1 (an up bias, which gpt-oss has), down 1/(aK), inputs |a z| ≤ 6; it is exact for z ≥ 7/c and a limit for z < 0 (`test_relu_sub_swigluoai`; the helper writes those rows). The graph's `ACT_LIMIT ∋ ("ReLU", "swigluoai")` label (a magnitude limit) matches the theorem only with that up bias and bounded inputs, neither of which the relation checks; the act-swap agreement number for a ReLU target on a swigluoai host measures an unclamped hypothetical host, not gpt-oss's unit. No ReLU zoo model is measured against a real clamped swigluoai host, so no headline number moves.

Note W3-a. `final_softcap_greedy` holds over the reals at every c > 0. In float64, `tanh` rounds to exactly ±1 once |z|/c > ~19, so two large logits can tie after the cap and the argmax can change; the test keeps |z| < 15c. Gemma-2's final cap is 30 with logits rarely above ~30·19, so this does not bite in practice, but the "every c" reading does not survive floating point.

W4-1. **`graph/eps_lever.py` (`EXACT = True`, `RHO_EXACT = 2^-25`, `PROPOSED`: "plain target on a qk-norm host: EXACT, free") vs `epsLever_eq`.** The lever is a limit over the reals. S√ε·N_ε(1, x/S) = x/√(1+u) with u = ‖x‖²/(dS²ε) > 0 for every x ≠ 0, so it is exact at no finite S (`epsLever_eq`, `epsLever_tendsto`). It is "exact" only because float32 rounds fl(ms + ε) to ε once u < 2^-25. This is the same pattern PROOFS.md flags as REFUTED for the −M dummy rows of `serving/embed.py`. The flag is also wrong in float64: `host_norm_off(.., "float64")` pays it as free, but at the module's own `a_needed(.., RHO_EXACT)` a float64 host misses by u/2 ≈ 1.5e-8 relative, about 10^8 float64 roundings. `test_mismatch_eps_lever_exact_flag` shows this. The rule should be a LIMIT edge with the threshold `S² > 2 Lh(ΠL)(ΣC_i)/δ` (`leverAttn_greedy`), paid by measurement as `hf/eps_lever_agreement` already does. The construction itself (`host_weights`) is correct, and it is BRIDGED above.

Doc note (not counted): on integration, `attn/mla_norm_agreement.py`'s docstring says an MLA host's latent norm "cannot be switched off (only its gain is free)". `leverAttn_layer_bound` (with kc = vv = the latent) says it can, as a limit. Branch measure-batch2 already rewrites that paragraph, but it calls the switch "exact", which is W4-1 again.

W5-1. **`serving/embed` expert duplication vs `moeR_copies_topk` / `moeR_dup_renorm` (`moeR_tie_split`).** The module docstring and `plan()` call "k_B = r k_A" exact with no condition ("the r copies are tied in probability, so the host's top-r k_A set is exactly the copies of the target's top-k_A"). The theorem needs the target's top-k to be strict: every selected expert must beat every unselected one. At a tie across the target's boundary, the host's top-(r k) can take copies of more than k distinct experts. Its output (renormalised or not) is then a mixture that no target tie-break gives (`moeR_tie_split`). This is not only a measure-zero worry: a router with tied rows is exactly what router steering writes (zero router rows, `serving/embed`'s own "steer" route and `moe/zero_router`). `test_mismatch_moe_dup_splits_copies_at_a_tie` takes a Qwen3-MoE target with its layer-0 router zeroed and embeds it with r = 2. `plan()` returns ok, and the stock host block selects 4 distinct experts where the target selects 2, so the outputs differ by O(|Y|). Fix: state the strictness condition, or refuse (or re-steer) targets whose router can tie at the top-k boundary. In float32 routing, near-ties closer than float32 resolution can split the copies too. `test_moeR_dup_on_stock_block` drops such tokens.

## Batch 2: helpers and modules copied from other branches

Helpers. These are small pure functions, and nothing else in the repo calls them:

- `block/parallel_residual.parallel_stack_in_seq_2x(x, As, Ms, phi=F.relu)`: the 2d-width sequential host for L parallel blocks, with identity units (`parallel_sub_seq_2x`). Before this, only the (L+1)d `parallel_stack_in_seq_wide` and the L = 1 case existed.
- `moe/dense_emulates_moe.dense_two_layer_margin(z, P, V, lam)`: the norm-free dense host with two FFN layers (`moe_two_layer_margin`). The select FFN computes s_e = Σ relu(z_a − z_e). The masked units compute relu(p − λ s_e), and erase units follow.
- `act/exotic_activations.relu_swigluoai_rows(a, c)` and the constants `SWIGLUOAI_LIMIT`, `SWIGLUOAI_K`: the gate, up, up-bias and down scales of `relu_sub_swigluoai`.
- `verify/levers/constructions.kv_pack` gained the keywords `nope=False` and `inv_host=None`. With the defaults, the function is the frontier-levers one unchanged.

Modules copied unchanged from other branches, for the tests:

- `act/act_width_fit.py` and `act/data/act_width_fits.json`, from act-width-fits.
- `hf/eps_lever_agreement.py` and `graph/eps_lever.py`, from measure-batch2.
- `moe/dense_host_prenorm.py`, from moe-loop-rescue.
- `hf/bos_constructions.py`, from bos-token.
- `verify/__init__.py`, `verify/levers/__init__.py` and `verify/levers/constructions.py`, from frontier-levers.

The in-memory emulate harness (`_emulate`) is local to the test file.

Run: `PYENV_VERSION=3.12.6 OMP_NUM_THREADS=4 python -m pytest tests/test_lean_bridges_2.py -q`


# Batch 3: code for the 29 NO CODE theorems of batch 2

Batch 2 left 29 theorems without a repo function that builds their construction. This batch writes that code and checks each theorem on it in `tests/test_lean_bridges_3.py` (float64, Hypothesis, the same tolerance rules; float32 slack only where `transformers` forces float32, and the test says so). The rows are updated in place in the batch 2 tables above.

## Batch 3 counts

| status | theorems |
|---|---|
| BRIDGED | 29 |
| NO CODE | 0 |
| MISMATCH | 0 |

So batch 2 now stands at 155 BRIDGED, 0 NO CODE.

## Batch 3: the code

| theorems | module | what it builds |
|---|---|---|
| `layerNorm_pad_center`, `hidden_pad_stack_layerNorm` | `norm/layernorm_pad.py` | `pad_ln_params` / `host_layer_norm`: the host LayerNorm (weight [√(d/D) γ; 0], bias [β; 0], ε d/D). `pad_layernorm_model(model, e)`: a stock GPT-NeoX or OPT host of the same class at width d + e. Writers and embeddings are centred, every LayerNorm is padded, e / head_dim extra heads have zero q/k/v rows and zero output columns, and the head is untied. Exact (float64 logits within ~1e-13) on pythia-14m (128 → 160) and OPT-125m (768 → 832). |
| `looped_width2_neg`, `looped_counter_two` | `block/looped_counter.py` | The width-2 quarter-rotation block, and the counter block. The counter block is one ReLU MLP on [x; c; o] holding both layers' units, masked by −B c and −B (o − c), plus a counter unit relu(o). |
| `t5_sub_additive_bias`, `alibi_sub_learned_abs_causal`, `bias_sub_learned_abs` | `attn/posbias_hosts.py` | The additive-bias head with T5's bucketed table (checked against stock `T5Attention`). Learned-absolute hosts as weight matrices on [x; position channels]: [1; t] for ALiBi, one-hot for any table. |
| `value_emb_off` | `attn/value_emb.py` | v + λ u installed on a stock attention's v_proj, and the λ = 0 switch. |
| `bias_free_ffn_zero` | `block/marker_stream.py` | `is_bias_free`. `marker_host`: the hidden-padded Llama host with a start token embedded on a pad channel. Its residual is exactly κ e_d at every layer. |
| `const_norm_carry`, `rmsNorm_zero_sq_sum` | `norm/const_norm.py` | The ε-budget host norm (ε′ = (dε − S)/(d+e)). `glu_const_writer`: 2m bias-free SiLU GLU units in a stock LlamaMLP that write the constant m. |
| `moe_select_prenorm_exact`, `pairSel_sq_sum` | `moe/exact_select_prenorm.py` | The exact two-layer select host for a ReLU top-1 MoE target: residual [x; g; κ], one RMSNorm with ε′ = (dε − S)/(d+E²+1), a clamped-pair select FFN, and a masked-expert / erase FFN. All layers are written as weight matrices and biases. |
| `mla_host_sphere_full_rank` | `attn/mla_host.py` | Writes a stock DeepseekV3Attention whose single-token output is a stock Llama GQA attention's A x. The latent down-projection is an isometry, and M = (D/α) A Wᵀ diag(1/w) goes through kv_b_proj and o_proj. |
| `moeR_group_copies` | `moe/group_dup.py` | A stock DeepseekV3MoE duplication host: copy r of group g becomes host group (g, r), with n_group G m, topk_group tg m, top-k k m. |
| `mask_greedy`, `sink_*` (3), `cap_*` (3), `soft_router_*` (3), `scaled_read_*` (3), `compensator_model_bound` | `block/limit_stacks.py` | Multi-layer hosts and targets on tiny random stacks, built from the repo's per-layer pieces: `hf/local_full.band_mask` clamped at −B, `attn/attention_sink.sink_attention`, `attn/softcap.softcap`, `moe/dense_emulates_moe.dense_softrouter` / `moe_top1`, the padded host norm over [x; δ s], and a polynomial reciprocal compensator. Also included: `lipschitz` (the measured constants), `SeqEmbed` / `BagEmbed` and `greedy`. |

## Batch 3: how the whole-model limit families are checked

The theorems take per-layer Lipschitz constants, a head constant, value / read bounds and a region as hypotheses. The tests do four things:

1. Build the n-layer host and target.
2. Measure the constants. L_i is the largest ∞→∞ norm of the target layer's Jacobian over both trajectories and random points near them, clamped at 1. Lh is ‖W_h‖∞. M, R, δ_r and ρ come from the weights and the trajectories. δ is the target's smallest top-1 − top-2 gap over the generated steps.
3. Evaluate the theorem's formula.
4. Check three things against it:
   - the measured logit error is below the bound;
   - past the stated threshold (B*, b*, c*, β*, ρ*) the host's greedy tokens equal the target's;
   - along each limit the error goes to float64 noise, at the stated rate where the theorem gives one (1/c², δ²).

The measured L_i is a sample maximum, not a certified constant. So these tests check the formula on real numbers, not the hypotheses. The bounds are very loose. At B = 10, b = −5 and c = 100, the bound is 10^3 to 10^14 times the measured error (product of the L_i, times n, times worst-case per-layer constants). This agrees with the Lipschitz branch.

## Batch 3: conditions the code exposed (none contradicts a theorem)

- **`hidden_pad_stack_layerNorm` on stock classes.** GPT-NeoX and OPT tie the head dim to hidden / heads. So a stock host keeps the head dim only if e is a multiple of it, and the extra heads must be silent. A tied OPT head must be untied: the input embedding is centred, but the head keeps the original rows. OPT's LayerNorm ε is not a config field, so the host sets it on the modules.
- **`moeR_group_copies` vs HF's group routing.** `DeepseekV3TopkRouter.get_topk_indices` drops a group by setting its experts' scores to 0, not to −∞. With a negative `e_score_correction_bias`, an expert of a dropped group can then beat a kept one, which `IsGroupTopK` does not model. The test keeps the bias ≥ 0. The stock `DeepseekV3MoE` only runs in float32: its accumulator takes the router's float32 dtype, and a float64 model raises.
- **`moe_select_prenorm_exact`.** The budget S ≤ dε forces δ ≤ √(2dε/(E(E−1))). The strong margin then needs Λ ≥ δ / (the smallest router gap), and λ ≥ R · max(loser pre-activation) / δ. The construction is exact, but its weights grow as the router gaps shrink.
- **`mla_host_sphere_full_rank`.** It applies only to a target whose attention input lies on a sphere (pre-norm ε = 0). No zoo model has one, so nothing loads it. Batch 2's `test_mla_host_sphere_full_rank_no_code` still passes: it pins `elephantine.universal` only.
- **T5.** `T5Attention` takes its softmax in float32 even in a float64 model. The slack in `test_t5_sub_additive_bias` is float32's.

Run: `PYENV_VERSION=3.12.6 OMP_NUM_THREADS=4 python -m pytest tests/test_lean_bridges_3.py -q` (39 tests, about 4 minutes on a loaded machine).
