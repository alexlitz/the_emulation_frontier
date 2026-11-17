# Validation ledger

Every reduction the graph uses, mapped to how it is validated. Status legend:

- **exact** — a numerical identity re-derived with random weights, matches to fp
  noise (`< 1e-9`).
- **real-module** — the *stock, unmodified* `transformers` class, parameterized into
  the special case, matching forward-for-forward.
- **limit** — reached only as a parameter → ∞ (validated: error decreases monotonically to fp noise).
- **floor / gap** — an approximation validated *as* an approximation, i.e. proven to
  **not** reach exact (a residual that does not shrink).
- **regime** — exact under a stated magnitude condition, proven to drift outside it.
- **edge** — exercised through `emulates()` (the categorical order); definitional.

**The headline relation** (`emul_matrix(mode="payable")`, `emulates()`, the frontier, `FRONTIER.md`,
the figures, `zoo.html`, `reduce` verdicts): capacity dominance plus every type wall that **has a
currency** — exact, floor / regime, or limit, each a row below. A slot difference with
no currency blocks emulation (`graph/model_graph.py::blocked_walls()`, listed in `FRONTIER.md`);
the softmax ← SSM / linear / hybrid near-gap is one of them, measured not payable (row 48), and so
is the cross-layer KV (CLA) carry (row 28: a ~6e-2 pre-norm floor, `CLA_CARRY_PAYABLE = False`).
Frontier **293** (279 AR + 14 non-AR: 10 bidirectional, 3 block-causal, 1 prefix-LM) over 942 shapes (3,906 models). The
secondary *"if every type wall were free"* (`mode="capacity"`: capacity + the one-way AR ⊆ diffusion wall) is **49**
(42 + 7); the stricter slice is 320 (`exact`), and granting the near-gap and the CLA carry free (`mixgap`, the old
headline) gives 265. (The release, 2026-11-15: 300 → 293 with the first-wave constructions -- pure recurrent stacks on
hybrids, the norm-free Mamba-2 lever, hyper-connection / DiffusionGemma host switch-offs, the LayerNorm-β readout, the
RoPE + NoPE-layer host, sinusoid tables, the same-table / table-length fixes, RMSNorm at hidden + 1 on LayerNorm, GeLU ←
SiLU at ×5, the sunk MoE extras, the BailingMoeV3 group code; the edge table's 8 catalog-only switch-offs are CATALOG,
a verified instance with the theorem pending. Round 3, 2026-11-15: 248 → 300: classification from the code sweep, the attention / MoE /
recurrent / stream rules, the spot checks' soundness walls, the router score function a feature switch, the readout
rule for score / embedding heads, NoPE on RoPE as a limit; 0 transitivity violations on the real shapes in every
mode, `python -m verify.r3.relcheck`. Round 2: 218 → 219 classification → 221 attention / norms → 258 MoE / MLA → 271 recurrent /
diffusion → 248 the rules, rescues and refutations, rows 50 - 56; 106 / 126 / 85 before the partial-RoPE host edge and the rotary-factor
classification fix, row 49; 104 / 122 / 83 before the edge-case audit: per-layer kinds, K = V, MLA's
latent norm, the measured qk-norm / post-norm / bias / sink walls, the forward-pass features, KV heads by
count, the Lean-checked conditions, and the rot gate on full-RoPE hosts. The relation is reflexive and
transitive in every mode, exact ⊆ payable ⊆ mixgap ⊆ capacity.)

2,969 tests in all (`pytest --collect-only`; 122 of them `tests/test_stock_loaders.py`). The relation's own: 109 `test_graph_rules` + 29 `test_constructions` + 20 `test_reduction_spot_checks`
+ 13 `test_hf_modules` + 7 `test_full_emulation` (a whole LlamaForCausalLM emulated to logits)
+ 5 `test_exotic_activations` + 15 `test_fp32` + 4 `test_hidden_width` (the full 4-axis capacity
spine built token-for-token) + 4 `test_edge_coverage` (the measured split of the headline edges: none crosses the near-gap) + 24 `test_payable_edges` (the headline's new currency edges and the activation threshold's calibration) + 13 `test_mixer_gap` (the softmax constructions of recurrent mixers and their recorded real-model swaps).
(fp64 stays the default because it certifies exactness — at fp32 an exact identity and a 1e-6
approximation are indistinguishable.) Test the reduction between any two models:
`python -m graph.reduce MODEL_A MODEL_B --verify`.

**Validation ladder (three rungs).** Each reduction below is checked at up to three
strictly-increasing strengths:

1. **fp64 tensor identity** (the tables below) — the algebra on random tensors.
2. **synthetic real-layer forward** — `elephantine/spot_check_reductions.py` rebuilds
   every reduction as a genuine 1-2 layer module forward (real `nn.Linear` weights,
   real softmax / causal mask / residual) and requires fp64 reproduction, closing the
   gap between "algebraically true" and "wired correctly inside a layer" (23/23).
   Run `python -m elephantine.spot_check_reductions`.
3. **real trained checkpoint** — `hf/real_reductions.py` loads twelve real checkpoints
   (gpt2, OPT-125m, Pythia-70m, Qwen2.5-0.5B, BLOOM-560m, tiny_starcoder, OLMo-2-1B,
   Mamba-130m, Gemma-2-2b, DeepSeek-V2-Lite, gpt-oss-20b, MiMo-7B), patches or remaps ONE real
   component, and re-verifies the reduction on the trained model's own forward (18/18; MTP is confirmed
   structurally on MiMo-7B's real weight index).
   Highlights: GQA/MQA→MHA `0.0`, NoPE zeroing real `inv_freq` → position-invariant,
   SwiGLU=2-layer-square `1.8e-15`, qk-norm CV on OLMo-2, Mamba recurrent≡parallel,
   **MLA absorption on DeepSeek-V2's real weights** (`8.5e-14`, ~9× cache), and the
   **attention sink identity on gpt-oss's real learned sink logits** (`5.3e-15`).
   DeepSeek-V2 and gpt-oss use partial (layer-0) weight loads. Only cross-layer-KV
   (CLA) still lacks a clean real-checkpoint forward check (Hymba reuses KV at runtime,
   not by weight omission) and stays on rungs 1-2. Run
   `python -m hf.real_reductions`.

| # | reduction (currency) | status | where validated |
|---|---|---|---|
| 1 | ALiBi = fixed RoPE pair (magnitude) | exact→limit | `test_alibi_is_fixed_rope_pair_exact`, `..._small_angle_is_linear_bias` |
| 2 | ALiBi ⊆ RoPE (magnitude) | exact + edge | `test_alibi_below_rope` |
| 3 | partial-RoPE ⊆ RoPE (free); RoPE ⊆ partial-RoPE rot-gated (row 49) | edge | `test_payable_edges.py`, `test_partial_rope_host.py::test_rot_gate_in_emulates` |
| 4 | **NoPE ⊆ RoPE** (free; inv_freq→0) | **real-module** | `hf/nope_rope.py`, `test_nope_is_rope_with_zero_frequency` |
| 5 | learned-abs vs RoPE incomparable — a **wall** in the headline (both ways) | edge | `test_learned_abs_and_rope_incomparable_in_order`; field matrix `test_cross_matrix_structure` |
| 6 | LayerNorm = center + RMSNorm + bias (fold) | exact | `test_layernorm_equals_center_rmsnorm_bias`, `test_dropping_centering_changes_output` |
| 7 | **MQA/GQA ⊆ MHA** (KV memory) | exact + **real-module** | `test_mqa_wide_emulates_mha_exact`; `hf/gqa_mha.py` (**0.0**) |
| 8 | MLA absorption = MHA (KV memory) | exact | `test_mla_absorption_identity_exact`; **caveat:** real MLA (DeepSeek, Mini-K3) RMS-normalizes the q and kv latents before up-projecting, which a plain MHA host cannot reproduce — exact only with the host's normalized-latent attention branch (`UAttention.pn`, tested on Mini-K3) |
| 9 | MLA rope part NOT absorbable | structural | documented in `attn/mla.py`; `q_cap`/`kv_cap` split in `dims_of` |
| 10 | **MoE ⊇ dense** (route to one) | exact + **real-module** | `moe/moe_dense.py` (OLMoE); `hf/moe_dense.py` (Mixtral, `2.8e-16` top-1) |
| 11 | MoE ⊇ wider dense, width k·ew (capacity) | exact | `test_moe_reproduces_wide_dense_via_k_experts` |
| 12 | **Dense ⊇ MoE**, hold all experts + hard select (capacity + depth) | exact | `moe/dense_emulates_moe.py`, `test_dense_hardselect_reproduces_top1_moe_exactly` |
| 13 | Dense ⊇ MoE, smooth router (limit) | limit | `test_dense_router_reaches_moe_only_in_the_limit` (`27 → 0.0` as β→∞) |
| 14 | SwiGLU = 2-layer squared FFN (free) | exact | `test_swiglu_equals_two_layer_squared_ffn`, `test_bilinear_glu...` |
| 15 | vanilla ⊆ GLU; GLU needs depth for reverse | exact + graph | `test_vanilla_glu_not_mutually_emulable` |
| 16 | ReLU = β→∞ limit of Swish/GeLU (magnitude) | limit | `test_relu_is_beta_limit_of_swish` |
| 17 | GeLU ≈ SiLU (approx) | **floor** (0.0139 Swish-β; single-op minimax 0.012) | `test_gelu_silu_has_a_nonzero_floor`; the thresholded floor table is row 44 |
| 18 | |x| from 2 ReLUs; tanh = affine sigmoid | exact | `test_abs_from_two_relus`, `test_tanh_is_affine_sigmoid` |
| 19 | sequential = 2 parallel blocks (width/depth) | exact | `test_sequential_block_is_two_parallel_blocks` |
| 20 | **local ⊆ full** (mask) | exact + **real-module** | `attn/local_vs_ssm.py`; `hf/local_full.py` (**0.0**) |
| 21 | every sparse pattern ⊆ full at depth = diameter | exact | `test_sparse_patterns_recover_full_at_diameter` |
| 22 | local ⟂ SSM (dual restrictions of full) | exact | `test_local_and_ssm_are_dual_restrictions` |
| 23 | softmax ⊋ linear/SSM (near-gap) | **gap** (∞); not payable (row 48) | `test_softmax_hard_selects_linear_floors`, `test_performer_error_decreases_with_features` |
| 24 | output-path rotation (retention state) | exact | `test_retention_parallel_equals_recurrent`, `test_output_rotation_is_relative`, `test_retention_collapses...` |
| 25 | qk-norm foldable (approx), **both directions** (same residual) | regime | `test_qknorm_folds_to_constant_scale_at_stable_magnitude`; `test_payable_edges.py::test_qknorm_regime_is_the_same_residual_both_ways` |
| 26 | softcap foldable (approx), **both directions** (same residual) | regime | `test_softcap_folds_to_identity_when_unsaturated`; `test_payable_edges.py::test_softcap_regime_is_the_same_residual_both_ways` |
| 27 | attention sink = standard + null token (fold) | exact | `test_attention_sink_is_standard_plus_null_token` (`1e-12`) |
| 28 | **cross-layer KV ⊆ standard** (width; carry K,V) — priced as a **floor** in the type order (not exact), **not payable** under the headline (`CLA_CARRY_PAYABLE = False`: the ~6e-2 pre-norm floor is two orders above `ACT_FLOOR_MAX`; `mode="mixgap"` grants it) | exact (norm-free only) | `attn/cross_layer_kv.py`, `test_cross_layer_kv_reproduced_by_carrying_kv`; under a stock pre-norm NOT exact at any width (`block/prenorm_carry.py`, `test_cla_carry_is_not_exact_under_stock_prenorm`) |
| 29 | **MTP folds away** (free; drop aux heads) | exact | `test_mtp_heads_do_not_change_base_forward` |
| 30 | **AR ⊆ diffusion** (mask), one-way: a diffusion host applies the causal mask; no AR host emulates a diffusion target | exact + edge | `attn/ar_diffusion.py` (max\|Δ\| = 0), `test_payable_edges.py::test_ar_is_a_masked_diffusion_forward`, `test_gen_paradigm_is_one_way`; the universal host runs one body causal and full (`test_same_host_carries_a_diffusion_forward_with_the_full_mask`) |
| 31 | **looped ⊆ untied** (depth) | **real-module** | `hf/looped_untied.py` (**0.0**); `test_untied_unrolls_a_looped_stack` |
| 32 | N:M dense ↪ wider 2:4 sparse (FLOPs/width) | exact | `test_dense_embeds_in_wider_2of4_exactly` |
| 33 | query-head-count richness gate | edge | `test_qheads_richness_gate` |
| 34 | every categorical slot edge (12 axes) | edge | `test_categorical_edge_general_emulates_special` (parametrized) |
| 35 | size rules: kv_cap, q_cap, ffn total/active, MLA penalty, depth, width, eff/param layers | graph | `test_bigger_kv_cap...`, `test_tied_kv_needs_extra...`, `test_dense_needs_double_depth...`, `test_depth_not_substitutable_by_width`, `test_untied_unrolls...` |
| 36 | SSM/linear-recurrence families are a ONE-WAY order (gated-delta top), **not** mutual | exact + gap | `attn/ssm_cross.py`; `test_no_mutual_emulation_between_distinct_families`, `test_cross_family_gap_is_real` |
| 37 | **hybrid (Jamba) target emulated**, SSM recurrence CARRIED (not identity) | exact | `elephantine/make_elephantine.py::emulate_hybrid`; `test_elephantine_carries_a_hybrid_target` |
| 38 | **Mamba-1 ⊆ gated-delta** (one head per channel, dk = state, dv = 1, no delta term) | **real-module** | `tests/test_mamba1_gated_delta.py` vs the real `MambaMixer` (**2.8e-17** with its fp32 casts mirrored) |
| 39 | **one universal host** carries every family as a weight setting (`elephantine/universal.py`) — 3,900 of 3,906 mixer families tested, 3,900 carried with their activation (open: Cyrah's gated code and five unclassified recurrences; the 583 GeLU and 4 mish models on exact UFFN branches — erf / tanh-approx GeLU, ungated or GeGLU, and a mish GLU, selected from `hidden_act` — vs real GPT-2, Pythia-70m, Gemma-3-270m and a stock mish Llama, since their floors to SiLU are above the calibrated `ACT_FLOOR_MAX`; xIELU vs real Apertus; Motif-3 whole model — GDLA differential heads, PolyNorm MoE, mHC — vs its own remote code — `ACT_STATUS`) | **real-module** | `tests/test_universal_elephantine.py`: every mixer family against its real code (transformers modules, HF remote code, upstream mamba_ssm / causal_conv1d / fla kernels, transformers 5.18 in a subprocess; PseudoBrain's pb27-vllm plugin under vLLM 0.31 in an isolated venv, fp32 ~1e-6). Gaps are the references' own fp32 level (fp64 references: RWKV-6/7, GigaChat-3.5, GPT-S3 ~1e-14; EdgeRock memory / Engram blocks ~1e-16); each branch is ablation-checked. Per-family evidence in `graph/minimal_emulator.py::FAMILY_STATUS`. **Loaded and tested** (`elephantine/loader_coverage.py`, printed by `graph.minimal_emulator`): 3,531 of the 3,900 carried models are accepted by a whole-model `from_*` loader that a test runs against their model_type's own code (151 types; 2,093 of them with a released checkpoint of the type checked; no test that runs remote code is credited, the families it covers count through tests against the labelled transcriptions in `tests/refs/`); the other 369 are carried by priced slot constructions only |
| 39b | **stock-transformers loaders**: the one host vs ~80 stock architectures (Llama / Mistral / Qwen / Gemma-1..4 / Phi / Bloom / OPT / Falcon / GPT-J / CodeGen / BigCode / OLMo / Granite / DeepSeek-V2/V3/V3.2 / GLM-5 / GPT-OSS / MoEs / Mamba hybrids / Nemotron-H / Kimi-Linear / ...), each family's ops a host branch (softcaps, ALiBi, sinks, qk-norm variants, v-norm, clip_qkv, query temperature, split FFN, PLE, KV slots, DSA indexer RoPE / sharing, ungated ReLU, swigluoai, group-limited / SparseMixer routing, dt floor, residual coefficients, ...) | **real-module** | `tests/test_stock_loaders.py`: synthetic stock configs in fp64, fp32 casts lifted ~1e-15 rel (documented exceptions: fp32 tables built at init — Bloom ALiBi / GeLU constant ~5e-11, Ministral-3 / Mistral-4 temperature ~3e-9, MiniMax decay ~4e-9, Phi-MoE angles ~3e-10), each branch ablated (switching it off moves the logits); 15 released checkpoints (gemma-2-2b 1.6e-14 rel, opt-350m post-LN 4.4e-15, OLMo-2-1B 7.9e-15, Qwen3-0.6B 6.8e-15, falcon-rw-1b 7.5e-5: Falcon rounds its ALiBi bias to bf16) |

| 40 | **ZERO_EDGES — hybrid host ⊇ softmax target** (depth: only the host's attention layers count; SSM layers zeroed) | **real-module** | `test_payable_edges.py::test_hybrid_host_zeroes_its_ssm_layers_for_a_softmax_target` (stock Jamba hosts a NoPE Llama, fp32-norm level); the universal host runs Llama with its SSM block zeroed (`test_one_host_carries_a_softmax_ar_model`) |
| 41 | **ZERO_EDGES — sliding host ⊇ full target** (depth: only its full-window layers count) | **real-module** | `test_payable_edges.py::test_sliding_host_zeroes_its_sliding_layers_for_a_full_target` (stock Qwen2 `layer_types`, `< 1e-9`; the same target on the sliding layers fails) |
| 42 | **ZERO_EDGES — KV-sharing host ⊇ standard target** (depth: only own-KV layers count) | exact (same identity-layer construction as 40–41) | a zeroed sharing layer contributes 0 whatever K,V it reads; counts in `dims_of` (`attn_eff` / `full_eff`) |
| 43 | **LayerNorm host ⊇ RMSNorm target** (width: the mirror [h; −h], 2× hidden) | exact | `test_composed_edges.py::test_norm_rmsnorm_is_layernorm_at_double_width`; charged as 2× hidden in `emul_matrix` |
| 44 | **activation floors**: host X pays for target Y iff the single-op minimax floor ≤ `ACT_FLOOR_MAX` = 0.0005 (one hop) — **calibrated on real models**: every FFN activation replaced by the host's best fit, scored on the original's own greedy 64-token continuations; payable = top-1 ≥ 99% and ≥ 90% of 32-token continuations identical. Folds / limits (0.000: swigluoai ↔ SiLU, ReLU ← swish, relu² ← xIELU) pass at 100% / 100%; every floor ≥ 0.009 fails on every model (GeLU ← SiLU 0.012: 96.5–98.2% top-1, 21–54% identical; SiLU ← ReLU 0.137: 14–74%; SiLU ← GeLU 0.014: 26–98%); scaled fits bracket the boundary at 0.0004 – 0.001. So every floor cell is a wall (the old 0.15 band admitted SiLU ← ReLU, xIELU ← SiLU, GeLU / SiLU / mish ↔ each other) | measured agreement | `act/act_swap_agreement.py` (data `act/data/act_swap_agreement.jsonl`, table / thresholds / figure `act/act_swap_report.py`); `act/activation_cross.py::fit_minimax`; `test_payable_edges.py::test_threshold_is_calibrated_on_the_recorded_swaps`, `test_live_swap_gpt2_and_opt`, `test_activation_floor_cells_match_the_recorded_table`, `test_activation_relation_is_the_one_hop_table` |
| 45 | **ALiBi ⊆ learned-abs** (a per-position key table on a reserved channel; bounded T) — round 2, pass 5: a WALL in the relation (the ramp channel is read through every pre-norm, j / R_j: refuted as exact, proofs/PROOFS.md "Pre-norm rescue attempts"; its width was never charged, verify/REVIEW.md 6) | exact (field level, norm-free) | `attn/positional_cross.py` cross matrix (alibi ← learned_abs 0.0000), `test_cross_matrix_structure` |
| 46 | **ALiBi ⊆ partial-RoPE** (the fixed-pair construction on one rotary pair) | limit | same construction as rows 1–2 |
| 47 | **parallel-block branch** in the universal host (no 2× width for parallel targets in the join) | **real-module** | `test_universal_elephantine.py::test_same_host_carries_a_parallel_block_cohere` (stock Cohere, fp32-norm level; the sequential path differs) |
| 48 | **softmax ← SSM / linear / hybrid is NOT payable** — every recurrent mixer of a real checkpoint replaced by an explicit softmax construction, scored like row 44 (top-1 ≥ 99%, ≥ 90% of 32-token continuations identical). Complete constructions pass (sink-normalised decay logits, small-temperature q·k, shift heads, GLU products): Mamba-1 (mamba-130m) 99.9% / 96% at 26,116 heads × 4 layers + 256 SiLU units per channel for softplus; RWKV-4-169m 100% / 100% at 769 heads × 2 layers; Mamba-2 (granite-4.0-h-350m) 100% / 100% at 101 heads × 3 layers; LFM2-350M short conv 99.9% / 100% in ONE stock layer at 12,288 heads (or 3 heads × 3 layers); gated DeltaNet (Qwen3.5-0.8B) only at ≥ T − 1 Jacobi hops (depth ~ context; 64 hops 98.9% / 100%, 32 hops 92% / 54%). Every multi-layer construction carries state across a stock pre-norm (needs a group norm, `block/prenorm_carry.py`) and every one-layer construction needs > 128 heads, so no zoo host pays. At the edge's own price it fails: one layer with a fixed decay (mamba-130m 85% / 4%; granite 85% / 12%; one mamba layer at a time 96–99.7% top-1), 16–4,096 shared decay rates (57–95% / ≤ 25%), RWKV-4 without its diagonal bonus (92% / 4%), a distilled softmax layer of 1× / 4× the target's width (2–16% top-1) | measured agreement | `attn/mixer_gap_agreement.py` (data `attn/data/mixer_gap_agreement.jsonl`, table / figure `attn/mixer_gap_report.py`), probe `attn/mixer_gap_distill.py`; `test_mixer_gap.py` (each construction converges in its limit on fp64 tensors; the recorded rows; `MIXER_GAP` prices vs the zoo; the headline refuses the edge) |
| 49 | **RoPE ⊆ partial-RoPE when the host rotates enough dims per head** — target pair j onto host pair j (a q / k column permutation: half-split NeoX / Phi / Qwen3-Next or interleaved GLM), its frequencies in the host's `inv_freq` buffer (as NoPE ⊆ RoPE), the target's pass dims onto host pass dims or zero-frequency pairs, the softmax scale and the YaRN mscale folded into `W_q` (into the qk-norm weight on a Qwen3-Next host, eps scaled by d_t / hd_h), a gated host's gate zeroed and `o_proj` doubled. Exact: fp64 max\|Δ\| ≤ 4.4e-16 on stock modules (GLM, Phi, GPT-NeoX hosts; Llama / Qwen2 targets; spare heads; the config-only route rope_theta_h = theta_t^(rot_h / d_t)), ≤ 3e-8 where HF's float32 rotary tables or qk RMSNorms differ (a YaRN target on a default host, Qwen3 in Qwen3-Next); a whole LlamaForCausalLM in a GlmForCausalLM 0.0, greedy identical; the real SmolLM2-135M in a GLM host 100% top-1, 100% identical over 24 prompts (float32). The graph gate: the target's RoPE schedules fit the host's rotated dims per head (`rope_fits`; one inv_freq table per schedule), on full-RoPE hosts too (several tables, Gemma-3's local / global theta, need room in one table or a table each), with a RoPE + NoPE target's NoPE layers counted as zero-frequency slots (SmolLM3, Beam). Below it: SmolLM2 with half its pairs (the low frequencies dropped) 99.3% top-1 but 83% identical, 41% top-1 at 2k context — not payable | exact (real-module) + measured | `hf/partial_rope_host.py`; `tests/test_partial_rope_host.py` |
| 50 | **the eps lever** — a host switches OFF a norm the target lacks (q / k norm, sandwich post-norms, Gemma-4's v-norm, the MLA latent norm): the norm's input scaled by 2^-a into its linear regime, its gain 2^a √ε; the same effective-eps reading relaxes the MLA latent / Mamba-2 group / GDN / KDA head multiples to ≥ | limit (exact to rounding), measured | `hf/eps_lever_agreement.py` (Qwen3 / Gemma-2 / -3 / -4 / DeepSeek-V2 hosts of SmolLM2-135M, Llama-3.2-1B, gemma-2b: 100% top-1, 100% identical, fp32 and fp64; valid under the run-dtype reading: the fp16 exclusion is a run-time caveat (run the emulator in bf16 / fp32), not a property of the stored checkpoint), `graph/eps_lever.py`, `tests/test_measure_batch2.py`, `tests/test_pass5_rules.py` |
| 51 | **the start-token rules** — a learned-abs target on a position-free / RoPE host (+2 full layers, +5 / +10 hidden, T + 1 + erasers units) and a sink target on a sink-free host with a q bias (+2 / +1 head dims at frequency 0, +1 hidden); the marker is the emulator's only | exact (Lean `learned_abs_two_block`, `sink_sub_plain`), real-module | `hf/bos_constructions.py` (stock GPT-2 / Llama / Qwen2 / gpt-oss-tinystories, fp64 ~1e-14), `tests/test_bos_constructions.py`, `tests/test_bos_rules.py`, `tests/test_pass5_rules.py` |
| 52 | **activation width** — a swap no one-unit floor pays, at FFN × k (k host units per target unit, shifts only with a host gate bias); GeLU erf ↔ tanh one class | measured (approximate) | `act/act_width_agreement.py` (Pythia / GPT-2 / Gemma-2 / SmolLM2 / Qwen2.5 / Llama-3.2 / gpt-oss-tinystories at the table's k: every one passes), `act/gelu_swap_agreement.py` (six models, both ways), `act/data/act_width_fits.json` (+ the tanh-GeLU hosts), `tests/test_act_width.py` |
| 53 | **looped host ⊇ untied target** — only a host whose every iteration re-runs fresh causal pre-norm attention without qk-norm (Nanbeige4.2): counter + start-token marker, +2 channels, 2d + 2 units (+2 / layer between-loop norm, +4d / layer v bias); every other looped host a wall | measured (top-1 100%, identical) | `hf/looped_host_prenorm.py`, `graph/rescue_rules.py`, `tests/test_rescue.py` |
| 54 | **pre-norm compensators** — the LayerNorm fold without an lm_head bias and parallel ⊆ seq by a pad + 1-D reciprocal compensator (GLU k = 8, ReLU k = 16) for families with measured RMS range D ≤ 31 (gpt_neox); transitivity removes every credit on the snapshot (each credited target hosts an unmeasured one) | approximate, measured on Pythia-160m | `verify/prenorm_rescue/` (`ln_fold_e2e.json`, `parallel_e2e.json`), `tests/test_pass5_rules.py` |
| 55 | **config levers** — a host sets the values its family's code reads: the sliding window (≥ the context: every sliding layer full), the rotary fraction, softcap values, q / k norm flags; RoPE frequencies only on a longrope-capable rotary (a fixed one: θ only, Lean `defaultFreq_match`, `rope_theta_cannot_reach_nope`) | exact (config), code-checked | `verify/levers/` (family_code.json, window_audit.md), `tests/test_pass5_rules.py` |
| 56 | **refutations / walls** — a swigluoai target needs a host up bias (Lean `swigluoai_sub_silu`); a LayerNorm-beta target on an RMSNorm host an lm_head bias (Lean condition 8); a plain target on a sink host is a limit (Lean `sink_host_factor`), not exact | edge | `verify/BRIDGES.md`, `tests/test_lean_bridges.py`, `tests/test_pass5_rules.py` |

## The genuine boundaries (validated as *not* reachable)

- **softmax → linear** is a true ∞ gap: linear attention approximates softmax with
  `1/√m` error and never reaches it (row 23).
- **GeLU ≈ SiLU** has a **0.0139 floor** — the one approximate edge with no exact
  limit (row 17).
- **dense → MoE** is exact only with a *hard* router; the smooth version is a limit,
  because routing is a discontinuous argmax (rows 12–13).
- **Walls with no currency** (block the headline; `FRONTIER.md` lists every pair present in the
  snapshot): learned-abs targets on relative hosts, RoPE / partial-RoPE targets on non-rotary
  hosts, a RoPE target on a partial-RoPE host that rotates fewer dims per head than the target's
  RoPE uses (row 49: no construction; truncating the low frequencies fails the standard), every
  activation pair without an exact fold or limit (all single-op floors are above the calibrated
  `ACT_FLOOR_MAX`: GeLU / mish / xIELU / situ / relu² / poly_norm on a SiLU host, a ReLU host for
  anything smooth), diffusion targets on causal hosts, SSM / linear / hybrid targets on softmax
  hosts (row 48), SSM / linear targets on hybrid hosts (the mixer slot does not carry the SSM
  family), and plain softmax targets
  deeper than a hybrid / sliding host's attention / full layers.

## Known non-firm items (honest)

- **Dormant positional free-edges** (`nope→learned_abs/alibi`, `alibi→additive_bias`,
  `t5_relbias→additive_bias`): declared as definitional restrictions, **not**
  individually tested. They do not fire in the real zoo (`t5` is filtered; no model
  classifies to `additive_bias`), so they never affect the frontier. This is the only
  remaining "asserted, untested" corner, and it is inert.

Everything load-bearing — every edge that touches a real model in the 3,906-model
frontier — is exact, a real-module proof, or a validated limit/approximation.
