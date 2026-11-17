"""Every rule the emulation relation allows -> the Lean theorem that justifies it (proofs/PROOFS.md),
the recorded measurement for an approximate edge, or UNJUSTIFIED. Writes verify/EDGE_JUSTIFICATION.md.

  python -m verify.z3_relation --edges            # (re)write the table

The rules are ENUMERATED from the code, as the Z3 model's tables are (verify/z3_relation.py Space): every slot
transition slot_reaches / regime_reaches allows in exact / payable / mixgap (decomposed into the order's hops), the
ZERO_EDGES, every feature a host may lack or carry (feats_ok on one-feature sets), and the capacity rules -- one or
more per numeric conjunct of emul_matrix (Model.parts). This file holds only the JUSTIFICATIONS, keyed by rule; a
rule the code allows with no entry here is UNJUSTIFIED by construction (so a new rule shows up as a gap). Every
theorem name is checked against the Lean sources (or PROOFS.md where the sources are absent).

Verdicts: PROVED (a theorem covers the rule as the relation uses it; kind exact / limit / bound / regime), CODE
(the host's code path itself, no construction: a null config value, an identical layer), CATALOG (a verified
end-to-end instance in the reduction catalog, the theorem pending),
MEASURED (an approximate edge whose payability is a recorded measurement, no theorem), REFUTED (a theorem proves the
rule false as the relation uses it), UNJUSTIFIED (nothing). A HEADLINE rule is one the payable relation allows; the
gaps are the headline rules that are REFUTED or UNJUSTIFIED. A rule in the EXACT slice whose justification is only
a limit / bound / measurement is flagged too.
"""
from __future__ import annotations

import os
import re
from collections import deque, namedtuple

import numpy as np

from graph import model_graph as G

PROOFS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proofs")
PROOFS_MD = os.path.join(PROOFS_DIR, "PROOFS.md")
LEAN_DIR = os.path.join(PROOFS_DIR, "lean", "EmulationProofs")

J = namedtuple("J", "verdict kind thms note evidence")


def j(verdict, kind="", thms=(), note="", evidence=""):
    return J(verdict, kind, tuple(thms), note, evidence)


def lean_names():
    """Theorem / lemma / def names of the Lean library (else the backticked names of PROOFS.md); None if neither."""
    names = set()
    if os.path.isdir(LEAN_DIR):
        pat = re.compile(r"\b(?:theorem|lemma|def)\s+([A-Za-z_][A-Za-z0-9_.']*)")
        for f in os.listdir(LEAN_DIR):
            if f.endswith(".lean"):
                names |= set(pat.findall(open(os.path.join(LEAN_DIR, f)).read()))
    if os.path.exists(PROOFS_MD):
        names |= set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", open(PROOFS_MD).read()))
    return names or None


# ---- one ORDER hop (slot, target value, host value) ---------------------------------------------------------------
HOP = {
    ("norm", "LayerNorm", "RMSNorm"): j("PROVED", "exact", ["layerNorm_eq_rms_center", "affine_fold", "layerNorm_sub_rmsNorm"],
                                        "same eps; writers centred; the beta folds into the consumers' biases and an "
                                        "lm_head bias (a beta target needs bias_out on the host: FEATURES)"),
    ("norm", "RMSNorm", "LayerNorm"): j("PROVED", "exact", ["layerNorm_mirror", "rmsNorm_sub_layerNorm_mirror"],
                                        "the [h; -h] mirror at 2x hidden (NORM_WIDTH_FACTOR charged)"),
    ("pos", "nope", "partial_rope"): j("PROVED", "exact", ["nope_sub_rope", "pairScore_zero_freq"],
                                       "inv_freq = 0 on the rotary pairs (a buffer write; rope_theta alone cannot: "
                                       "rope_theta_cannot_reach_nope)"),
    ("pos", "nope", "learned_abs"): j("PROVED", "exact", ["nope_sub_learned_abs"], "a zero position table"),
    ("pos", "nope", "alibi"): j("PROVED", "exact", ["nope_sub_alibi"], "slope 0 (emul_matrix: only a switchable host)"),
    ("pos", "alibi", "rope"): j("PROVED", "limit", ["alibi_sub_rope_limit", "alibi_sub_rope_bound", "alibi_rope_model_bound",
                                                    "alibi_rope_greedy"],
                                "causal targets only (alibi_host_not_bidirectional) and a spare zero-frequency pair: "
                                "emul_matrix checks gen = ar and head_dim_h >= head_dim_t + 2"),
    ("pos", "alibi", "partial_rope"): j("PROVED", "limit", ["alibi_sub_rope_limit", "alibi_sub_rope_bound",
                                                            "alibi_rope_model_bound", "alibi_rope_greedy"],
                                        "the rotating pair on the host's rotary sub-space; causal; a spare pair"),
    ("pos", "alibi", "additive_bias"): j("PROVED", "exact", ["alibi_sub_additive_bias"], "the bias table is the ramp"),
    ("pos", "alibi", "learned_abs"): j("REFUTED", "exact", ["alibi_sub_learned_abs_causal", "const_readout_channel",
                                                            "no_prenorm_homogeneous_channel"],
                                       "norm-free only; under a pre-norm the key-side m j reaches the logit as j / R_j "
                                       "(PROOFS.md pre-norm rescue: REFUTED exact; the compensator approximation not run)",
                                       "attn/positional_cross.py residual 0.0000 is the norm-free toy"),
    ("pos", "t5_relbias", "additive_bias"): j("PROVED", "exact", ["t5_sub_additive_bias"], "the bucket table"),
    ("pos", "partial_rope", "rope"): j("PROVED", "exact", ["nope_sub_rope", "pairScore_zero_freq"],
                                       "the full-RoPE host sets inv_freq = 0 on the target's pass pairs"),
    ("pos", "rope", "partial_rope"): j("PROVED", "exact", ["partial_rope_host", "defaultFreq_match", "partial_rope_host_config"],
                                       "rot-gated (rope_fits; the converse partial_rope_too_few)"),
    ("act", "ReLU", "SiLU"): j("PROVED", "limit", ["relu_sub_silu", "relu_silu_bound", "relu_silu_model_bound", "relu_silu_greedy"],
                               "never exact (relu_not_exact_silu)"),
    ("act", "ReLU", "GeLU"): j("PROVED", "limit", ["relu_sub_gelu", "relu_gelu_bound", "relu_gelu_greedy"]),
    ("act", "ReLU", "GeLU_tanh"): j("MEASURED", "limit", [],
                                    "the tanh-approximate GeLU: GeLU_tanh(beta x) / beta -> relu x holds as for erf GeLU, "
                                    "but the Lean theorems state the Gaussian-CDF form only",
                                    "ACT_FLOORS[ReLU][GeLU_tanh] = 0.000 (act/activation_cross.py; ACT_LIMIT)"),
    ("act", "GeLU", "GeLU_tanh"): j("MEASURED", "floor", [], "a one-unit fit (floor 0.0003 <= ACT_FLOOR_MAX)",
                                    "act/data/gelu_swap_agreement.jsonl (six models, raw swap: payable)"),
    ("act", "GeLU_tanh", "GeLU"): j("MEASURED", "floor", [], "a one-unit fit (floor 0.0003 <= ACT_FLOOR_MAX)",
                                    "act/data/gelu_swap_agreement.jsonl (six models, raw swap: payable)"),
    ("act", "ReLU", "mish"): j("PROVED", "limit", ["relu_sub_mish", "relu_mish_bound", "relu_mish_greedy"]),
    ("act", "ReLU", "swigluoai"): j("PROVED", "limit", ["relu_sub_swigluoai", "swigluoai_scaled_gate_vanishes"],
                                    "the named route (scale the gate) is REFUTED by the +-7 clamp; proved by another "
                                    "route with the up bias -1 gpt-oss has, bounded inputs (round 3, batch 3b: the "
                                    "host's up bias is checked, dims bias_up -- w3_relu_oai_bias)"),
    ("act", "SiLU", "swigluoai"): j("PROVED", "regime", ["sigmoid_scale_fold"],
                                    "g sigma(1.702 g) = SiLU(1.702 g) / 1.702 read backwards; the host's (up + 1) undone by "
                                    "its up bias; the +-7 clamp keeps it a regime (inputs scaled inside it)"),
    ("act", "swigluoai", "SiLU"): j("PROVED", "regime", ["sigmoid_scale_fold", "swigluoai_sub_silu", "swigluoai_needs_up_bias"],
                                    "exact in the clamp regime with an up-projection bias on the host (emul_matrix checks ffn_bias since pass 5)"),
    ("act", "relu2", "xielu"): j("PROVED", "limit", ["relu2_sub_xielu", "relu2_xielu_bound", "relu2_xielu_greedy"],
                                 "never exact (xielu_ne_relu2); bounded pre-activations"),
    ("window", "local", "full"): j("PROVED", "exact", ["local_sub_full", "pattern_sub_full"], "the band mask"),
    ("qknorm", "head_free", "head"): j("PROVED", "exact", ['qkn_gain_one'], "a weight setting of the host's norm"),
    ("qknorm", "head_free", "head_post"): j("PROVED", "exact", ['qkn_gain_one', 'rmsNorm_one_rot_comm', 'weightFree_comm', 'rot_sumsq'], "a weight setting of the host's norm"),
    ("qknorm", "head_free", "head_distinct"): j("PROVED", "exact", ['qkn_gain_one', 'qkn_equal_gains'], "a weight setting of the host's norm"),
    ("qknorm", "head", "head_distinct"): j("PROVED", "exact", ['qkn_equal_gains'], "a weight setting of the host's norm"),
    ("qknorm", "head_ln", "head_ln_bias"): j("PROVED", "exact", ['qkn_ln_zero_bias'], "a weight setting of the host's norm"),
    ("qknorm", "ln_shared_bias", "head_ln_bias"): j("PROVED", "exact", ['qkn_ln_equal_gains'], "a weight setting of the host's norm"),
    # round 3, batch 2a (qknorm_by_code_unlisted / qknorm_fixed_gain)
    ("qknorm", "head_free", "head_qfree"): j("PROVED", "exact", ['qkn_gain_one', 'weightFree_comm'],
                                             "the host's k gain 1; a weight-free norm commutes with RoPE"),
    ("qknorm", "head_free", "head_free_gain"): j("CODE", "exact", ['qkn_gain_one'],
                                                 "the host's gain key set to 1 (a config scale, free): the weight-free norm"),
    ("qknorm", "head_qfree", "head"): j("PROVED", "exact", ['qkn_gain_one'], "q gain 1, k gain the target's"),
    ("qknorm", "head_qfree", "head_distinct"): j("PROVED", "exact", ['qkn_gain_one', 'qkn_equal_gains'],
                                                 "q gains 1, k gains the target's on every head"),
    ("softcap", "final", "no"): j("PROVED", "exact", ["final_softcap_greedy", "softcap_strictMono"],
                                  "dropping a final-logit cap never moves the argmax (greedy-exact)",
                                  "attn/regime_agreement.py: Gemma-2 100% / 100%"),
    ("softcap", "no", "final"): j("PROVED", "exact", ["final_softcap_greedy", "softcap_strictMono"],
                                  "a final-capped host: the cap never moves the argmax (greedy-exact)",
                                  "attn/regime_agreement.py: final cap 15 / 30, 100% / 100%"),
    ("softcap", "no", "attn"): j("PROVED", "limit", ["cap_layer_bound", "cap_model_bound", "cap_greedy", "cap_model_limit"],
                                 "the host's attention cap c -> inf (a config value)",
                                 "attn/regime_agreement.py: six plain checkpoints pass from c = 50 - 5,000"),
    ("softcap", "final", "attn"): j("PROVED", "limit", ["cap_greedy", "final_softcap_greedy"],
                                    "attention cap set large, the final cap kept / dropped"),
    ("sink", "no", "yes"): j("PROVED", "limit", ["sink_to_plain_limit", "sink_model_bound", "sink_greedy", "sink_host_factor"],
                             "b -> -inf; not exact at any b (sink_host_ne_plain): a regime edge since pass 5, out of the exact slice"),
    ("sink", "yes", "no"): j("MEASURED", "regime", [], "drop the sinks: measured NOT payable; the start-token rule is the capacity rule below", "block/fold_agreement.py: gpt-oss-tinystories 97.5% / 33%"),
    ("mixer", "ssm", "softmax"): j("UNJUSTIFIED", "measured", [],
                                   "the near-gap, granted only by mode mixgap; measured NOT payable at any fixed price "
                                   "(MIXER_GAP); the reverse is impossible (no_finite_feature_map)",
                                   "attn/mixer_gap_agreement.py (MIXER_GAP)"),
    ("mixer", "linear", "softmax"): j("UNJUSTIFIED", "measured", [], "the near-gap (mixgap only), measured not payable",
                                      "attn/mixer_gap_agreement.py (MIXER_GAP)"),
    ("mixer", "hybrid", "softmax"): j("UNJUSTIFIED", "measured", [], "the near-gap (mixgap only), measured not payable",
                                      "attn/mixer_gap_agreement.py (MIXER_GAP)"),
    # release (branch fr-mixer, pure_on_hybrid): a pure recurrent stack on a hybrid host, the attention layers zeroed and
    # target layer i on the host's i-th recurrent layer (the family by the rf_ features, the depth by kind_match)
    ("mixer", "ssm", "hybrid"): j("PROVED", "exact", ["padded_run", "assign_padded", "assign_run", "assign_off",
                                                      "zeroed_sublayer"],
                                  "zero the attention layers (o_proj = 0, down_proj = 0): an order-preserving embedding "
                                  "of the target's layers into the host's recurrent ones",
                                  "verify/spot/catalog/frm.py frm_mamba2_on_bamba; hf/data/mixer_rules_agreement.jsonl "
                                  "pure:mamba2-130m@bamba 100% / 100%"),
    ("crosslayer_kv", "yes", "no"): j("REFUTED", "measured", ["cla_carry_prenorm_fails"],
                                      "the CLA carry (mixgap only); the pre-norm carry is refuted", "measured gap 6e-2"),
    ("mtp", "yes", "no"): j("PROVED", "exact", ["mtp_sub_plain"]),
    ("mtp", "no", "yes"): j("PROVED", "exact", ["mtp_sub_plain"], "the main logits do not read the MTP modules"),
    ("gen", "ar", "diffusion"): j("PROVED", "exact", ["causal_attn_layer", "causal_stack"], "the causal mask"),
    ("gen", "ar", "block"): j("PROVED", "exact", ["block_one_causal", "causal_stack"], "block size 1 is the causal mask"),
    ("gen", "block", "diffusion"): j("PROVED", "exact", ["block_sub_diffusion", "pattern_sub_full"], "the block mask on a full host"),
    ("gen", "diffusion", "block"): j("PROVED", "exact", ["diffusion_sub_block", "block_ge_full"], "an all-ones caller mask"),
    # round 3, batch 1 (prefix_lm_mask): the prefix-LM generation value. A full host under the caller's prefix mask is
    # pattern_sub_full; a prefix-LM host (or a block host) run under the caller's mask has no PROOFS.md lemma: the
    # catalog instances (fp64, the reference decoder) are the evidence, as for the other catalog-backed hops
    ("gen", "prefix", "diffusion"): j("PROVED", "exact", ["pattern_sub_full"], "the prefix mask on a full host",
                                     "verify/spot/catalog/r3.py r3_gen_prefix_diffusion"),
    ("gen", "ar", "prefix"): j("CODE", "exact", [], "a prefix-LM host run under the caller's causal mask (prefix 0)",
                               "verify/spot/catalog/r3.py r3_gen_ar_prefix"),
    ("gen", "block", "prefix"): j("CODE", "exact", [], "a prefix-LM host run under the caller's block mask",
                                  "verify/spot/catalog/r3.py r3_gen_block_prefix"),
    ("gen", "diffusion", "prefix"): j("CODE", "exact", [], "a prefix-LM host run under an all-ones caller mask",
                                      "verify/spot/catalog/r3.py r3_gen_diffusion_prefix"),
    ("gen", "prefix", "block"): j("CODE", "exact", [], "a block-causal host run under the caller's prefix mask",
                                  "verify/spot/catalog/r3.py r3_gen_prefix_block"),
    # round 3, nonlm_heads (a design decision on readouts: the output read as fixed rows at fixed positions): no PROOFS.md
    # theorem (needs theorem readout_row_subset; the lm_head := any matrix setting of tied_sub_untied, read at fixed
    # rows); the catalog instances are the evidence
    ("head", "score", "lm"): j("CODE", "exact", [], "an UNTIED host's lm_head row := the score head (a head bias: the "
                               "host's bias_out), read at that row at the scored token",
                               "verify/spot/catalog/r3.py r3_head_score_lm"),
    ("head", "embedding", "lm"): j("CODE", "exact", [], "a last- / first-token embedding without L2 only: lm_head rows := "
                                   "the Dense (or the identity), read at that position (no snapshot model: every "
                                   "embedding pools by the mean or normalises)",
                                   "verify/spot/catalog/r3.py r3_head_embedding_lm"),
}
ZERO = {
    ("mixer", "softmax", "hybrid"): j("PROVED", "exact", ["zeroed_sublayer"], "zero the SSM / linear sublayers; depth "
                                      "charged on the host's own-KV attention layers"),
    ("window", "full", "local"): j("PROVED", "exact", ["zeroed_sublayer"], "zero the sliding layers; depth on full layers"),
    ("crosslayer_kv", "no", "yes"): j("PROVED", "exact", ["zeroed_sublayer"], "zero the KV-sharing layers"),
}
# ---- features: (feature, side) -- side "target": the target has it, the host lacks it; "host": the reverse ----
FEAT = {
    ("tied_emb", "target"): j("PROVED", "exact", ["tied_sub_untied"], "lm_head := E^T"),
    ("out_gate", "host"): j("PROVED", "exact", ["out_gate_off"], "W_g = 0 (gate 1/2), 2 W_o"),
    # round 3, batch 1: the classification batch's host switch-offs (the free sides in graph/model_graph.FEATURES name
    # the Lean lemma; the catalog instance builds each from the target's weights) and its two target sides
    ("cross_attn", "host"): j("PROVED", "exact", ["zeroed_sublayer"], "the encoder_attn out_proj 0: the cross-attention "
                              "sublayer adds 0", "verify/spot/catalog/r3.py r3_cross_attn_off"),
    ("enc_input", "host"): j("CODE", "exact", [], "the target's tokens fed through the host's own per-token embedding "
                             "path, the encoder unused", "verify/spot/catalog/r3.py r3_enc_input_off"),
    ("gated_norm", "host"): j("PROVED", "exact", ["out_gate_off", "shared_gate_half"], "fc2 = 0 (gate 1/2) and the norm "
                              "weight doubled", "verify/spot/catalog/r3.py r3_gated_norm_off"),
    ("out_gate_post", "host"): j("PROVED", "exact", ["out_gate_off"], "W_g = 0 (gate 1/2) after o_proj, o_proj doubled",
                                 "verify/spot/catalog/r3.py r3_out_gate_post_off"),
    ("scmoe", "host"): j("PROVED", "exact", ["zeroed_sublayer"], "attn1 / mlp0 / mlp1 zeroed: one sequential MoE layer",
                         "verify/spot/catalog/r3.py r3_scmoe_off"),
    ("fake_quant", "target"): j("CODE", "exact", [], "the target's forward is the fixed linear map q(W): a plain host "
                                "stores the quantised matrices", "verify/spot/catalog/r3.py r3_fake_quant_target"),
    ("post_norm_attn", "target"): j("PROVED", "limit", ["postNorm_layer_bound", "postNorm_model_bound", "postNorm_greedy"],
                                    "a full-sandwich host's MLP post-norm switched off by the eps lever (eps_lever hosts "
                                    "only, emul_matrix)", "verify/spot/catalog/r3.py r3_post_norm_attn_target"),
    # round 3, batch 2a: the head-wise / biased sigmoid gates and ERNIE-5.5's q-latent gate (the gate projection 0)
    **{(f, "host"): j("PROVED", "exact", ["out_gate_off"], "the gate projection (and its bias) 0: gate 1/2, 2 W_o")
       for f in ("out_gate_head", "out_gate_bias", "out_gate_head_bias", "out_gate_qlat")},
    # round 3, batch 2b: the learned query temperature and the warped RoPE phase set to the identity by weights
    ("q_temp", "host"): j("CODE", "exact", ["scalar_fold"], "s_h = 0, b_h = 1 (weights): the factor is 1 at every position",
                          "verify/spot/catalog/r3.py r3_q_temp_off"),
    ("rope_warp", "host"): j("CODE", "exact", ["scalar_fold"], "the metric projections 0 (weights): mass = spin = 0, the "
                             "plain table", "verify/spot/catalog/r3.py r3_rope_warp_off"),
    # round 3, batch 2c: Doge's dynamic mask with A = 0 (a common logit shift) and an eos document mask never triggered
    ("dyn_mask", "host"): j("PROVED", "exact", ["renorm_softmax_shift"], "A = 0 (a weight): bias exp(0) = 1 on every "
                            "unmasked key, a common shift; keep_window_size >= T (config) keeps the top-k off",
                            "verify/spot/catalog/r3.py r3_dyn_mask_off"),
    ("doc_mask", "host"): j("CATALOG", "exact", [], "the host's eos id is mapped to no target token (vocab ignored): one "
                            "document, the causal mask (restriction)", "verify/spot/catalog/r3.py r3_doc_mask_off"),
    # round 3, batch 2d: head-side branches switched off by a weight (no PROOFS.md lemma for vha / xsa / canon / the
    # memory: the catalog instance is the measurement), JetMoE's tied experts, and clip_qkv's host config lever
    ("vha", "host"): j("CATALOG", "exact", [], "U = V = 0 (weights): the head mix m = I + V U^T is the identity",
                       "verify/spot/catalog/r3.py r3_vha_off"),
    ("xsa_scaled", "host"): j("CATALOG", "exact", [], "xsa_alpha = 0 (a weight): tanh(0) = 0 removes nothing",
                              "verify/spot/catalog/r3.py r3_xsa_scaled_off"),
    ("canon", "host"): j("CATALOG", "exact", [], "the conv taps 0 (weights): the residual Canon layer is the identity",
                         "verify/spot/catalog/r3.py r3_canon_off"),
    ("moa", "host"): j("PROVED", "exact", ["tied_moe_is_dense"], "every expert's q / o tied to the target's (one KV-head "
                       "block), the output bias 0: the renormalised top-k gates sum to 1",
                       "verify/spot/catalog/r3.py r3_moa_off"),
    ("mem_attn", "host"): j("CATALOG", "exact", [], "read_o = 0 (a weight): the memory path adds 0",
                            "verify/spot/catalog/r3.py r3_mem_attn_off"),
    ("mla_q_raw", "host"): j("CATALOG", "exact", [], "the raw-latent columns of q_gate_proj 0 (weights): the plain normed-"
                             "latent query", "verify/spot/catalog/r3.py r3_mla_q_raw_off"),
    # (host config lever: clip_qkv None, shape-preserving -- the code skips the clamp)
    ("clip_qkv", "host"): j("PROVED", "exact", ["clampSym_id"], "host config lever clip_qkv None: the clamp is not run "
                            "(the identity on every input)", "verify/spot/catalog/r3.py r3_clip_qkv_off"),
    # round 3, batch 3b (host config lever: swiglu_limit = inf / 0 / None -- the clamp is the identity or not built)
    ("act_clip", "host"): j("PROVED", "exact", ["act_clip_regime"], "host config lever swiglu_limit = inf (tf5 V4 / "
                            "GLM-5-Next / Hy4: clamp(x, -inf, inf) = x) or 0 / None (V4.1, Step-3.5 / 3.7, Ling-3: no "
                            "clamp built): exact on every input", "verify/spot/catalog/r3.py r3_act_clip_off"),
    # round 3, batch 4a: the loop boundaries' host sides -- Nanbeige4.2's host config lever skip_loop_final_norm (a norm
    # on/off flag, shape-preserving) and LFM3-RD's step scale set to 1 (a weight); every other loop boundary
    # (loop_bnorm, loop_inject, loop_kv_mix, hrm_two_state, early_exit) is a wall both ways
    ("loop_bnorm_cfg", "host"): j("CODE", "exact", [], "host config lever skip_loop_final_norm = true: the stream is "
                                  "normed once at the end (modeling_nanbeige.py:2401-2412), the plain looped stack",
                                  "verify/spot/catalog/r3.py r3_loop_bnorm_off"),
    ("rd_controller", "host"): j("PROVED", "exact", ["scalar_fold"], "step_scales[T-1] = 1 (a weight) at the config's "
                                 "manual T = num_motifs (no feedback / refinement): p + 1 (h - p) = h",
                                 "verify/spot/catalog/r3.py r3_rd_controller_off"),
    # round 3, batch 4b: the residual-stream / extra-input host sides (weights only; every target side is a wall, and
    # hc_* / res_gate_ms / ngpt / byte_patch / engram_klm are walls both ways)
    ("res_gain", "host"): j("CATALOG", "exact", [], "the residual gains g = 1 (Doge) / s = 1 (ZAYA), weights: 1 (.) r = r, "
                            "the plain residual", "verify/spot/catalog/r3.py r3_res_gain_off"),
    ("res_bias", "host"): j("PROVED", "exact", ["bias_off"], "ZAYA's residual bias b = 0 (with s = 1)",
                            "verify/spot/catalog/r3.py r3_res_bias_off"),
    ("res_gate", "host"): j("CATALOG", "exact", ["out_gate_off", "bias_off", "scalar_fold"], "CFRD's gate weight and bias "
                            "0 (sigma = 1/2 on every token), the output projections x 2 / residual_scale",
                            "verify/spot/catalog/r3.py r3_res_gate_off"),
    ("backout", "host"): j("PROVED", "exact", ["extra_input_off"], "backout_lambda = 0: x_L - 0 x_m = x_L",
                           "verify/spot/catalog/r3.py r3_backout_off"),
    ("engram_qwen4", "host"): j("PROVED", "exact", ["extra_input_off"], "value_proj 0: the gate times 0 and the conv of "
                                "norm(0) = 0 add 0", "verify/spot/catalog/r3.py r3_engram_gate_off"),
    ("engram_allgdn", "host"): j("PROVED", "exact", ["extra_input_off"], "val 0: v = 0 and its causal conv 0",
                                 "verify/spot/catalog/r3.py r3_engram_gate_off"),
    ("engram_needle", "host"): j("PROVED", "exact", ["extra_input_off"], "value projection 0: the gated taps add 0",
                                 "verify/spot/catalog/r3.py r3_engram_gate_off"),
    ("engram_crowd", "host"): j("PROVED", "exact", ["extra_input_off"], "the bucket table 0",
                                "verify/spot/catalog/r3.py r3_engram_crowd_off"),
    ("switch_lora", "host"): j("PROVED", "exact", ["switch_lora_off"]),
    ("engram", "host"): j("PROVED", "exact", ["extra_input_off"]),
    ("ple", "host"): j("PROVED", "exact", ["extra_input_off"]),
    ("value_emb", "host"): j("PROVED", "exact", ["value_emb_off"]),
    ("bias_qkv", "host"): j("PROVED", "exact", ["bias_off"]),
    ("bias_out", "host"): j("PROVED", "exact", ["bias_off"]),
    ("router_bias", "host"): j("PROVED", "exact", ["router_bias_off"], "b = 0"),
    # round 3, batch 3a: the router / block host sides (block_routing: a host config lever)
    ("router_bias_logit", "host"): j("PROVED", "exact", ["router_bias_off"], "b = 0 (the logit-domain bias)",
                                     "verify/spot/catalog/r3.py r3_router_bias_logit_off"),
    ("block_routing", "host"): j("PROVED", "exact", ["topk_all"], "host config lever expert_capacity = num_experts: "
                                 "every expert allowed in every block, the plain per-token top-k",
                                 "verify/spot/catalog/r3.py r3_block_routing_off"),
    ("par_moe_res", "host"): j("PROVED", "exact", ["zeroed_sublayer"], "the experts' down-projections 0: the residual "
                               "MLP on the post-attention stream is the sequential block's FFN (dense targets only; "
                               "moe_why keeps MoE targets off)", "verify/spot/catalog/r3.py r3_par_moe_res_off"),
    ("shared_gate", "host"): j("PROVED", "exact", ["moe_shared_gate_off", "ffn_smul"], "gate input 0, down-proj doubled"),
    # round 3, nonlm_heads: a host's exit heads are outputs nobody reads (no snapshot model has the feature)
    ("exit_heads", "host"): j("CODE", "exact", [], "read only the exit at L (the final logits): the other exit heads' "
                              "outputs are unread", "verify/spot/catalog/r3.py r3_exit_heads_off"),
    ("selfcond", "host"): j("PROVED", "exact", ["selfcond_zero_path"], "down_proj = 0 adds exactly 0"),
    ("router_sigmoid", "target"): j("PROVED", "exact", ["moeR_dup_renorm"], "free here; priced by the MoE rules (moe_why: "
                                    "the host's code / config computes the target's score)"),
    ("router_sigmoid", "host"): j("PROVED", "exact", ["moeR_dup_renorm"], "free here; priced by the MoE rules"),
    ("router_sqrtsoftplus", "target"): j("PROVED", "exact", ["moeR_dup_renorm"], "free here; priced by the MoE rules"),
    ("router_sqrtsoftplus", "host"): j("PROVED", "exact", ["moeR_dup_renorm"], "free here; priced by the MoE rules"),
    ("router_group", "target"): j("PROVED", "exact", ["moeR_group_copies"], "free here; priced by the MoE rules (group shape)"),
    ("router_group", "host"): j("PROVED", "exact", ["moeR_group_copies"], "free here; priced by the MoE rules"),
    ("dsa", "target"): j("PROVED", "exact", ["topk_all", "dsa_dense", "dsa_dense_causal"], "exact while the context <= index_topk (the context caveat)"),
    ("dsa", "host"): j("PROVED", "exact", ["topk_all", "dsa_dense"], "the host sets index_topk >= the context"),
    ("post_norm", "host"): j("PROVED", "limit", ["postNorm_layer_bound", "postNorm_model_bound", "postNorm_greedy"],
                             "the eps lever (pass 5: a lever-capable host switches its sandwich post-norms off)"),
    ("vnorm", "host"): j("PROVED", "limit", ["leverAttn_layer_bound", "leverAttn_model_bound", "leverAttn_greedy"],
                         "the eps lever on Gemma-4's v-norm (pass 5)"),
    ("rf_", "host"): j("PROVED", "exact", ["zeroed_sublayer"], "a host's extra recurrent layers zeroed"),
    ("rf_", "target"): j("UNJUSTIFIED", "measured", [], "the near-gap (mixgap only): any recurrent family on softmax "
                         "layers; measured not payable", "attn/mixer_gap_agreement.py"),
}
# ---- capacity rules: (name, conjunct, modes, justification, predicate over (ts, hs, td, hd) real-pair arrays) ----
CapRule = namedtuple("CapRule", "name conjunct modes just pred")
_ALL3 = ("exact", "payable", "mixgap")


def _isin(x, vs):
    return np.isin(x, list(vs))


CAPACITY = [
    CapRule("depth: a deeper host (identity layers)", "depth", _ALL3,
            j("PROVED", "exact", ["zeroed_sublayer"]), lambda ts, hs, td, hd: hd["eff_layers"] > td["eff_layers"]),
    # round 3 (kind_order): the layers of a mixed-kind target go IN ORDER onto host layers of a reaching kind, the rest
    # zeroed (FrMixer padded_run: a stack with identity layers inserted computes the target)
    CapRule("layer kinds: the target's layer sequence embedded in order (unused host layers zeroed)", "kinds", _ALL3,
            j("PROVED", "exact", ["padded_run", "assign_padded", "zeroed_sublayer"]),
            lambda ts, hs, td, hd: (td["kseq"] > 0) & (hd["kseq"] > 0)),
    # round 3 (mamba_ffn_none): a block with no FFN on any FFN host, the host MLP's output projection 0
    CapRule("ffn: no FFN (pure Mamba blocks) on an FFN host (down_proj 0)", "ffn", _ALL3,
            j("PROVED", "exact", ["zeroed_sublayer"], "every expert's / the shared expert's down_proj 0 on a MoE host"),
            lambda ts, hs, td, hd: (ts["ffn"] == "none") & (hs["ffn"] != "none")),
    CapRule("hidden: a wider residual (zero-padded; host eps = eps d / D)", "hidden", _ALL3,
            j("PROVED", "exact", ["hidden_pad_stack", "hidden_pad_blocks", "hidden_pad_stack_layerNorm", "layerNorm_pad_center"],
              "exact with the host norm eps set to eps d / D (a config value); with the target's own eps it is not "
              "(hidden_pad_needs_eps)"),
            lambda ts, hs, td, hd: hd["hidden"] > td["hidden"]),
    CapRule("q_cap / qheads / head_dim: more or wider query heads", "qheads", _ALL3,
            j("PROVED", "exact", ["qknorm_pad", "zeroed_sublayer", "attn_scale_fold"]),
            lambda ts, hs, td, hd: (hd["qheads"] > td["qheads"]) | (hd["head_dim"] > td["head_dim"])),
    CapRule("ffn_total / ffn_active: a wider FFN", "ffn", _ALL3,
            j("PROVED", "exact", ["ffn_extra_units", "ffn_extra_zero_input", "glu_zero_up_unit", "moe_wider_experts"]),
            lambda ts, hs, td, hd: hd["ffn_total"] > td["ffn_total"]),
    CapRule("per-layer FFN widths: each target layer on a host layer at least as wide", "ffn_widths", _ALL3,
            j("PROVED", "exact", ["ffn_extra_units"]), lambda ts, hs, td, hd: (td["fw_listed"] > 0) | (hd["fw_listed"] > 0)),
    CapRule("kv: MQA / GQA target on a host with more KV heads (tile)", "kv_heads", _ALL3,
            j("PROVED", "exact", ["mqa_sub_gqa", "gqa_sub_mha", "groupOf"]),
            lambda ts, hs, td, hd: (hd["kv_heads"] > td["kv_heads"]) & _isin(ts["kv"], ("MQA", "GQA"))),
    CapRule("kv: a target KV group split over host KV heads (K, V replicated)", "kv_heads", _ALL3,
            j("PROVED", "exact", ["gqa_tiled_scores", "groupOf"]),
            lambda ts, hs, td, hd: (td["q_per_kv"] > hd["q_per_kv"]) & (hd["kv_heads"] > 0)),
    CapRule("kv: KV-head packing (several target KV heads in one wider host head)", "kv_heads", _ALL3,
            j("PROVED", "exact", ["kvPack_nope", "kvPack_rope_iff", "kvPack_rope_attn", "mha_sub_wide_mqa"],
              "RoPE targets only on a pairs-free host (kvPack_default_not_exact)"),
            lambda ts, hs, td, hd: (hd["kv_heads"] > 0) & (td["kv_heads"] > hd["kv_heads"])),
    CapRule("rot gate on a pairs-free host: the target's schedules (or its config class) in the host's tables", "rot_gate",
            _ALL3, j("PROVED", "exact", ["partial_rope_host", "multi_rope_tables_equal", "nope_sub_rope"],
                     "the converse partial_rope_too_few"),
            lambda ts, hs, td, hd: _isin(ts["pos"], G.ROT_TARGETS) & _isin(hs["pos"], G.ROT_HOSTS) & (hd["rope_pairs_free"] > 0)),
    CapRule("rot gate on a fixed-rotary host: rope_theta (and the rotary fraction) only, one default-type schedule",
            "rot_gate", _ALL3,
            j("PROVED", "exact", ["defaultFreq_match", "partial_rope_host_config", "partial_rotary_raise",
                                  "partial_rotary_raise_buffer"], "rope_theta_cannot_reach_nope: no zero slot by theta"),
            lambda ts, hs, td, hd: _isin(ts["pos"], G.ROT_TARGETS) & _isin(hs["pos"], G.ROT_HOSTS) & (hd["rope_pairs_free"] == 0)),
    CapRule("zero-frequency room: a NoPE / ALiBi target's dims at inv_freq 0 on a RoPE host (config only)", "pos", _ALL3,
            j("PROVED", "exact", ["nope_sub_rope", "pairScore_zero_freq", "rope_theta_cannot_reach_nope"]),
            lambda ts, hs, td, hd: _isin(ts["pos"], ("nope", "alibi")) & _isin(hs["pos"], G.ROT_HOSTS)),
    # round 3 (nope_rope_limit): the host config lever linear rope scaling, factor -> inf (a limit, payable / mixgap)
    CapRule("nope_rope_limit: a NoPE target on a RoPE host whose rotary reads linear scaling (factor -> inf)", "pos",
            ("payable", "mixgap"),
            j("PROVED", "limit", ["rope_scale_limit_identity", "rope_scale_limit_bound", "rope_scale_limit_attn"],
              "a host config lever; never exact at a finite factor (rope_theta_cannot_reach_nope)",
              "attn/data/nope_rope_limit_agreement.jsonl (granite-4.0-h-350m-base); verify/spot/catalog/r3.py "
              "r3_nope_rope_limit"),
            lambda ts, hs, td, hd: _isin(ts["pos"], ("nope",)) & _isin(hs["pos"], G.ROT_HOSTS) & (hd["rope_lin"] > 0)),
    CapRule("nope_rope_limit: a mixed target's NoPE layers on a host table of their own scaled to infinity (one linear "
            "factor per table), its rotating schedules on the host's other tables", "rot_gate", ("payable", "mixgap"),
            j("PROVED", "limit", ["rope_scale_limit_identity", "rope_scale_limit_attn", "defaultFreq_match",
                                  "partial_rope_host_config"], "a host config lever (rope_lin 2)"),
            lambda ts, hs, td, hd: _isin(ts["pos"], G.ROT_TARGETS) & _isin(hs["pos"], G.ROT_HOSTS) & (hd["rope_lin"] >= 2)),
    CapRule("a pairs-free RoPE target on a fixed-rotary host as its MHA expansion", "rot_pairs_free", _ALL3,
            j("PROVED", "exact", ["gqa_sub_mha", "groupOf"], "the wall's reason is kvPack_default_not_exact"),
            lambda ts, hs, td, hd: _isin(ts["pos"], G.ROT_TARGETS) & _isin(hs["pos"], G.ROT_HOSTS)
            & (td["rope_pairs_free"] > 0) & (hd["rope_pairs_free"] == 0)),
    CapRule("window lever: a window-free host's sliding layers are full (window >= the context)", "kinds", _ALL3,
            j("PROVED", "exact", ["band_ge_causal", "window_ge_full"], "exact up to the context (the context caveat)"),
            lambda ts, hs, td, hd: hd["win_free"] > 0),
    CapRule("layer kinds: a target's K = V layers on own-V host layers", "kinds", _ALL3,
            j("PROVED", "exact", ["k_eq_v_sub_separate"]),
            lambda ts, hs, td, hd: (td["kFT_eff"] + td["kLT_eff"] > hd["kFT_eff"] + hd["kLT_eff"])),
    CapRule("layer kinds: full host layers play sliding target layers (masked)", "kinds", _ALL3,
            j("PROVED", "exact", ["local_sub_full", "pattern_sub_full"]),
            lambda ts, hs, td, hd: (td["kLU_eff"] + td["kLT_eff"] > hd["kLU_eff"] + hd["kLT_eff"])),
    CapRule("layer kinds (mixgap): attention host layers play recurrent / KV-sharing target layers", "kinds", ("mixgap",),
            j("UNJUSTIFIED", "measured", [], "the near-gap and the CLA carry (measured not payable)"),
            lambda ts, hs, td, hd: (td["kR_eff"] > hd["kR_eff"]) | (td["kS_eff"] > hd["kS_eff"])),
    CapRule("config: softcap values (a null cap is no cap)", "softcap", _ALL3,
            j("CODE", "exact", [], "the code skips a null cap: no construction"),
            lambda ts, hs, td, hd: hd["softcap_free"] > 0),
    CapRule("config: a q / k norm switched off by its flag", "qknorm", _ALL3,
            j("CODE", "exact", [], "the code builds no norm when the flag is off"),
            lambda ts, hs, td, hd: (hd["qk_switch"] > 0) & (ts["qknorm"] == "no")),
    CapRule("eps lever: a gained qk-norm switched off (plain target)", "qknorm", ("payable", "mixgap"),
            j("PROVED", "limit", ["epsLever_bound", "leverAttn_layer_bound", "leverAttn_model_bound", "leverAttn_greedy"],
              "", "hf/data/eps_lever_agreement.jsonl: 100% / 100%"),
            lambda ts, hs, td, hd: (hd["eps_lever"] > 0) & (ts["qknorm"] == "no") & (hs["qknorm"] != "no")),
    CapRule("eps lever: a plain target on an MLA host (latent norms off)", "mla_wall", ("payable", "mixgap"),
            j("PROVED", "limit", ["leverAttn_model_bound", "leverAttn_greedy", "mla_sub_mha"],
              "a limit, never exact (mla_value_ray); the target's K, V through the latent"),
            lambda ts, hs, td, hd: (ts["kv"] != "MLA") & (hs["kv"] == "MLA")),
    CapRule("MLA <- MLA: latents an integer multiple (m copies), q-latent norm present in both, per-head nope / v >=",
            "mla_latent", _ALL3, j("PROVED", "exact", ["rmsNorm_rep", "attn_scale_fold"],
                                  "m copies keep the rms (the hard-coded eps kept); zero padding needs the eps lever"),
            lambda ts, hs, td, hd: (ts["kv"] == "MLA") & (hs["kv"] == "MLA")),
    CapRule("eps lever: a wider MLA latent (zero-padded, eps_eff = the target's)", "mla_latent", ("payable", "mixgap"),
            j("PROVED", "limit", ["epsLever_bound", "leverAttn_model_bound", "rmsNorm_eps_inj"]),
            lambda ts, hs, td, hd: (ts["kv"] == "MLA") & (hs["kv"] == "MLA") & (hd["eps_lever"] > 0)),
    CapRule("start token: a learned-abs target on a position-free / RoPE host (+2 layers, +5 / +10 hidden, T + 1 units)",
            "pos", _ALL3, j("PROVED", "exact", ["learned_abs_two_block", "learned_abs_model", "padded_prenorm_stack"],
                            "bounded context, a start marker", "hf/data/bos_constructions.jsonl"),
            lambda ts, hs, td, hd: (ts["pos"] == "learned_abs") & _isin(hs["pos"], ("nope", "rope", "partial_rope"))),
    CapRule("start token: a sink target on a sink-free host with a q bias and a spare zero-frequency slot", "sink", _ALL3,
            j("PROVED", "exact", ["sink_sub_plain", "sink_needs_zero_freq", "bias_free_ffn_zero"],
              "that the start token's stream stays on its channel is argued in prose (PROOFS.md)",
              "hf/data/bos_constructions.jsonl"),
            lambda ts, hs, td, hd: (ts["sink"] == "yes") & (hs["sink"] == "no") & (hd["q_bias"] > 0)),
    CapRule("activation width: a swap at FFN width x k (act_k)", "act", ("payable", "mixgap"),
            j("MEASURED", "bound", ["actFit_mish_silu6", "shiftfree_silu_obstruction"],
              "a certified fit only for mish <- 6 SiLU; the other k are numeric fits (round 3, batch 3b: none on a "
              "swigluoai host branch -- the fits ignore gpt-oss's gate clamp, catalog p5_act_width_gelu_oai: "
              "spot_act_width_oai)",
              "act/data/act_width_agreement.jsonl, act/data/act_width_fits.json"),
            lambda ts, hs, td, hd: ts["act"] != hs["act"]),
    CapRule("dense FFN target on a MoE host (zero router; units dealt over the k experts + shared)", "moe", _ALL3,
            j("PROVED", "exact", ["dense_sub_moe", "dense_sub_moe_shared", "zero_router_constant_gate"]),
            lambda ts, hs, td, hd: (td["moe_E"] == 0) & (hd["moe_E"] > 0)),
    CapRule("MoE <- MoE by expert duplication (k_h = m k_t, E_h >= m E_t, m f_h >= f_t, renorm / score / group shape)",
            "moe", _ALL3,
            j("PROVED", "exact", ["moeR_copies_topk", "moeR_dup_renorm", "moeR_dup_plain", "moeR_scaling_fold",
                                  "moeR_group_copies", "moe_sub_moe_shared", "const_norm_carry"],
              "exact off routing ties (moeR_tie_split)"),
            lambda ts, hs, td, hd: (td["moe_E"] > 0) & (hd["moe_E"] > 0)),
    CapRule("MoE target on a dense host: top-1 unit gate, 2x depth, +E channels, +2E(E-1) units (not exact)",
            "dense_host", ("payable", "mixgap"),
            j("PROVED", "limit", ["vanishing_carry_limit", "moe_two_layer_margin", "moe_select_prenorm_exact"],
              "REFUTED as exact (continuous_host_ne_top1_moe), out of the exact slice",
              "moe/data/dense_host_prenorm.jsonl: top-1 100%"),
            lambda ts, hs, td, hd: (td["moe_E"] > 0) & (hd["moe_E"] == 0)),
    # (round 3, batch 3b, spot_glu_vanilla: walled in every mode but capacity -- needs theorem glu_sub_vanilla_prenorm)
    CapRule("glu target on a vanilla host at 2x depth (FFN_RANK)", "depth",
            ("capacity",) if G._r3("spot_glu_vanilla") else _ALL3,
            j("UNJUSTIFIED", "bound", ["swiglu_eq_sq_ffn", "yarotsky_product_bound"],
              "exact only with squared units; a ReLU host gets a bound with many units -- neither at the relation's "
              "price (2x depth, same width), and both need act(g) carried across a pre-norm (round 3: a wall, "
              "spot_glu_vanilla)"),
            lambda ts, hs, td, hd: (ts["ffn"] == "glu") & (hs["ffn"] == "vanilla")),
    CapRule("sequential target on a parallel host at 2x depth", "depth", _ALL3,
            j("PROVED", "exact", ["seq_sub_parallel", "seq_stack_sub_parallel"]),
            lambda ts, hs, td, hd: (ts["block"] == "sequential") & (hs["block"] == "parallel")),
    # (round 3, batch 3b, spot_comp_removed: the compensators credit nothing -- no host construction builds them; verify/prenorm_rescue/ measures them)
    CapRule("parallel target on a sequential host: the RMS compensator (measured families, k d units, +1 hidden)",
            "par_carry", ("payable",) if G.p5_port_live("comp_par") else (),
            j("PROVED", "bound", ["compensator_model_bound", "vanishing_carry_limit", "scaled_read_greedy"],
              "given the measured fit (rho)", "verify/prenorm_rescue: Pythia-160m GLU k=8 100%"),
            lambda ts, hs, td, hd: (ts["block"] == "parallel") & (hs["block"] == "sequential") & (td["rms_ok"] > 0)),
    CapRule("parallel target on a sequential host at 2x width (mixgap only)", "par_carry", ("mixgap",),
            j("REFUTED", "exact", ["parallel_sub_seq_prenorm_fails", "parallel_sub_seq_2x"],
              "norm-free 2x width + 2 d MLP units per block is proved (the units charged: w1_par_units); under a "
              "pre-norm the carry is refuted"),
            lambda ts, hs, td, hd: (ts["block"] == "parallel") & (hs["block"] == "sequential")),
    CapRule("LayerNorm-beta target on an RMSNorm host with an lm_head bias", "norm", _ALL3,
            j("PROVED", "exact", ["layerNorm_sub_rmsNorm"], "Lean condition 8: the beta term into the lm_head bias"),
            lambda ts, hs, td, hd: (ts["norm"] == "LayerNorm") & (td["ln_beta"] > 0) & (hs["norm"] == "RMSNorm")
            & (hd["lm_head_bias"] > 0)),
    CapRule("LayerNorm-beta target on an RMSNorm host without one: the RMS compensator (measured families)", "norm",
            ("payable", "mixgap") if G.p5_port_live("comp_ln") else (),
            j("PROVED", "bound", ["compensator_model_bound", "const_readout_channel", "scaled_read_greedy"],
              "given the measured fit (rho)", "verify/prenorm_rescue: Pythia-160m GLU k=8 100%"),
            lambda ts, hs, td, hd: (ts["norm"] == "LayerNorm") & (td["ln_beta"] > 0) & (hs["norm"] == "RMSNorm")
            & (hd["lm_head_bias"] == 0)),
    # round 3, batch 3b: the vanilla <- gated step on the host's up bias (spot_vanilla_glu_bias; a bias-free host's only
    # constant is a residual channel the pre-norm rescales: catalog p5_vanilla_on_glu_nobias), needle's HadamardMLP on its
    # own code (hadamard_mlp_unlisted)
    CapRule("vanilla target on a gated / MoE host: the up branch held at 1 by the host's up bias (bias_up)", "up_bias",
            _ALL3, j("PROVED", "exact", ["glu_ungated", "dense_sub_moe"], "up weights 0, up bias 1",
                     "verify/spot/catalog/ffn_moe.py (vanilla -> GeGLU with an up bias), p5.py p5_vanilla_on_moe_bias"),
            lambda ts, hs, td, hd: (ts["ffn"] == "vanilla") & (hs["ffn"] != "vanilla") & (hd["bias_up"] > 0)),
    CapRule("HadamardMLP target on the same HadamardMLP (equal n)", "hadamard", _ALL3,
            j("CODE", "exact", [], "the identical layer (the same Kronecker blocks and permutations)"),
            lambda ts, hs, td, hd: (td["hada_n"] > 0) & (td["hada_n"] == hd["hada_n"])),
    CapRule("looped target on an untied host", "depth", _ALL3, j("PROVED", "exact", ["looped_sub_untied"]),
            lambda ts, hs, td, hd: (ts["loop"] == "looped") & (hs["loop"] == "untied")),
    CapRule("untied target on a PLAIN looped host (counter + start token, +2 channels, 2d + 2 units)", "loop",
            ("payable", "mixgap"),
            j("MEASURED", "limit", ["looped_counter_two", "untied_sub_looped_periodic"],
              "proved norm-free with a counter; the pre-norm host measured", "hf/data/looped_host_prenorm.jsonl"),
            lambda ts, hs, td, hd: (ts["loop"] == "untied") & (hs["loop"] == "looped")),
    CapRule("ALiBi <- ALiBi: the host's code-computed slopes contain the target's (extra heads zeroed)", "alibi_alibi",
            _ALL3, j("PROVED", "exact", ["alibiSlopes_mono", "alibiExps_mono", "heads_inject"],
                     "Falcon's bf16 set nests in no real set (alibi_falcon_slope_not_mem): the closed form keeps them apart"),
            lambda ts, hs, td, hd: (ts["pos"] == "alibi") & (hs["pos"] == "alibi")),
    CapRule("NoPE target on a switchable ALiBi host (alibi_off)", "nope_alibi", _ALL3,
            j("PROVED", "exact", ["nope_sub_alibi", "nope_sub_rope", "mpt_bias_max_limit"],
              "Falcon's alibi flag (exact: RoPE then); the remote MPT's bias max is a limit (fp32 underflow)"),
            lambda ts, hs, td, hd: (ts["pos"] == "nope") & (hs["pos"] == "alibi")),
    CapRule("recurrent RC_GE: zero-padded heads / states / conv / ranks", "rec", ("exact", "payable"),
            j("PROVED", "exact", ["dscan_inert", "scan_pad", "mamba_state_pad", "mamba_channel_inert", "rmsNorm_pad_scaled",
                                  "cconv_pad", "lowrank_pad"]),
            lambda ts, hs, td, hd: (td["rc_rule"] == G.RC_GE) & (hd["rc_rule"] > 0)),
    CapRule("recurrent RC_M2: Mamba-2 whole-group copies (w_h = m w_t)", "rec", ("exact", "payable"),
            j("PROVED", "exact", ["rmsNorm_rep", "dscan_embed", "cconv_pad"], "zero padding at the same eps REFUTED "
              "(grouped_norm_zero_pad_fails): the eps lever rule below"),
            lambda ts, hs, td, hd: (td["rc_rule"] == G.RC_M2) & (hd["rc_rule"] > 0)),
    CapRule("recurrent RC_GDN: equal key dims, value multiple, key heads replicated", "rec", ("exact", "payable"),
            j("PROVED", "exact", ["deltaRow_inert", "deltaRow_pad", "delta_out_pad", "rmsNorm_rep", "gdn_dk_needed", "cconv_pad"]),
            lambda ts, hs, td, hd: (td["rc_rule"] == G.RC_GDN) & (hd["rc_rule"] > 0)),
    CapRule("recurrent RC_KDA: equal head dims; heads, conv, gate rank >=", "rec", ("exact", "payable"),
            j("PROVED", "exact", ["deltaRow_inert", "deltaRow_pad", "fullrank_plays_lowrank", "lowrank_pad", "cconv_pad"]),
            lambda ts, hs, td, hd: (td["rc_rule"] == G.RC_KDA) & (hd["rc_rule"] > 0)),
    CapRule("recurrent RC_EQ: identical recurrent dims", "rec", ("exact", "payable"),
            j("CODE", "exact", [], "the identical layer (no construction)"),
            lambda ts, hs, td, hd: (td["rc_rule"] == G.RC_EQ) & (hd["rc_rule"] > 0)),
    CapRule("eps lever on the recurrent output norm: zero-padded groups / heads (M2, GDN, KDA)", "rec", ("payable",),
            j("PROVED", "limit", ["epsLever_bound", "lever_model_bound", "rmsNorm_pad_scaled"],
              "eps_h = eps_t r where the code reads the eps (rec_eps_cfg)", "hf/data/eps_lever_agreement.jsonl"),
            lambda ts, hs, td, hd: (td["rc_rule"] > 0) & (hd["eps_lever"] > 0) & (hd["rec_eps_cfg"] > 0)),
    CapRule("Zamba2 shared blocks: a multiple of the target's, adapter ranks >=", "zamba", ("exact", "payable"),
            j("PROVED", "exact", ["zamba_shared_index", "zamba_shared_needs_multiple", "lowrank_pad"]),
            lambda ts, hs, td, hd: (td["zamba_mem"] > 0) & (hd["zamba_mem"] > 0)),
    CapRule("TransMLA penalty: an MLA target on a non-MLA host at 2x kv_cap (capacity mode only)", "kv_cap", ("capacity",),
            j("REFUTED", "exact", ["mla_latent_norm_not_mha"], "every other mode walls it"),
            lambda ts, hs, td, hd: (ts["kv"] == "MLA") & (hs["kv"] != "MLA")),
    CapRule("context (opt-in USE_CONTEXT): a local host for a wider window at ceil((w_t - 1) / (w_h - 1)) x depth",
            "window_depth", (), j("UNJUSTIFIED", "exact", ["winReach_iff", "winLocal_stack"],
                                  "reachability only (the receptive field), not a construction of the target's attention"),
            None),
]


def _path(order, a, b):
    """The node path edge_cost would use (shortest all-exact path, else shortest)."""
    for exact_only in (True, False):
        q, seen = deque([(a, [a])]), {a}
        while q:
            cur, path = q.popleft()
            if cur == b:
                return path
            for (p, _, ex) in order.get(cur, []):
                if (ex or not exact_only) and p not in seen:
                    seen.add(p)
                    q.append((p, path + [p]))
    return None


# (release: CATALOG = a verified end-to-end instance in the reduction catalog, the theorem pending -- the simple host
# switch-offs vha, xsa, canon, mem_attn, mla_q_raw, doc_mask, res_gate, res_gain; justified, not proved)
_RANK = {"PROVED": 0, "CODE": 0, "CATALOG": 1, "MEASURED": 1, "UNJUSTIFIED": 2, "REFUTED": 3}


def _combine(js):
    """A path's justification: the worst verdict of its hops, every theorem, the weakest kind."""
    worst = max(js, key=lambda x: _RANK[x.verdict])
    kinds = [x.kind for x in js]
    kind = next((k for k in ("measured", "regime", "bound", "limit") if k in kinds), "exact")
    return J(worst.verdict, kind, tuple(dict.fromkeys(t for x in js for t in x.thms)),
             " | ".join(dict.fromkeys(x.note for x in js if x.note)), " | ".join(dict.fromkeys(x.evidence for x in js if x.evidence)))


def _ok_name(names, t):
    return names is None or t in names


def edge_rows(space, P, R=None):
    """Every allowed rule with its justification and real-edge counts. R: {mode: emul_matrix(P, mode)}."""
    from verify.z3_relation import ALL_SLOTS, SENTINELS, Model, Evaluator, point_values, BASE_DIMS, KIND_DIMS
    S = space
    names = lean_names()
    R = R or {m: G.emul_matrix(P, mode=m) for m in ("exact", "payable", "mixgap", "capacity")}
    orders = dict(G.SLOTS)
    rows = []
    bad_names = set()
    sl = {s: np.array([p["slots"][s] for p in P], dtype=object) for s in ALL_SLOTS}
    off = ~np.eye(len(P), dtype=bool)

    def used(sel):
        return {m: int((R[m] & sel & off).sum()) for m in ("exact", "payable", "mixgap")}

    def check(js):
        for t in js.thms:
            if not _ok_name(names, t):
                bad_names.add(t)
    # ---- slot transitions
    keys = {}
    for mode in ("exact", "payable", "mixgap"):
        for key, tab in S.tables.items():
            if key[0] == "cat" and key[2] == mode:
                s = key[1]
            elif key[0] in ("mixer", "cla") and key[1] == mode:
                s = "mixer" if key[0] == "mixer" else "crosslayer_kv"
            else:
                continue
            for a, b in tab:
                if a != b:
                    keys.setdefault((s, a, b), []).append(mode)
    for (s, a, b), modes in sorted(keys.items()):
        if "|" in a or "|" in b or a in SENTINELS or b in SENTINELS or a in S.abstract[s] or b in S.abstract[s]:
            continue
        if (s, a, b) in HOP:
            just, path = HOP[(s, a, b)], [a, b]
        else:
            path = _path(orders[s], a, b) or [a, b]
            hops = [HOP.get((s, x, y), j("UNJUSTIFIED", "", [], f"no entry for the hop {x} -> {y}"))
                    for x, y in zip(path, path[1:])]
            just = _combine(hops)
        check(just)
        sel = (sl[s][:, None] == a) & (sl[s][None, :] == b)
        rows.append(dict(group="slot", rule=f"{s}: {a} -> {b}", path=" -> ".join(path), modes=tuple(dict.fromkeys(modes)),
                         just=just, used=used(sel)))
    for (s, a, b), just in ZERO.items():
        check(just)
        sel = (sl[s][:, None] == a) & (sl[s][None, :] == b)
        rows.append(dict(group="zero", rule=f"{s}: {a} -> {b} (ZERO_EDGES, plain targets)", path=f"{a} -> {b}",
                         modes=("exact", "payable", "mixgap"), just=just, used=used(sel)))
    # ---- features
    fsets = [set(filter(None, p["slots"].get("feats", "").split("+"))) for p in P]
    for f in S.feats:
        if f in ("rf_*",):
            continue
        for side, ix in (("target", 0), ("host", 1)):
            modes = tuple(m for m in ("exact", "payable", "mixgap")
                          if S.frule[(m == "exact", m == "mixgap")][f][ix])
            if not modes:
                continue
            key = (f, side) if (f, side) in FEAT else (("rf_", side) if f.startswith("rf_") else None)
            just = FEAT.get(key, j("UNJUSTIFIED", "", [], "no entry")) if key else j("UNJUSTIFIED", "", [], "no entry")
            check(just)
            has = np.array([f in x for x in fsets])
            sel = (has[:, None] & ~has[None, :]) if side == "target" else (~has[:, None] & has[None, :])
            rows.append(dict(group="feature", rule=f"{f}: {'target has it, host lacks it' if side == 'target' else 'host has it, target lacks it'}",
                             path="", modes=modes, just=just, used=used(sel)))
    # ---- capacity rules
    m = Model(S, False, False, 3, 27)
    vals = [point_values(m, p) for p in P]
    dn = lambda k: np.array([v[k] for v in vals], dtype=np.int64)
    numk = BASE_DIMS + KIND_DIMS + ("fw_listed", "moe_E", "kseq")
    td = {k: dn(k)[:, None] for k in numk}
    hd = {k: dn(k)[None, :] for k in numk}
    ts = {k: sl[k][:, None] for k in ALL_SLOTS}
    hs = {k: sl[k][None, :] for k in ALL_SLOTS}
    for r in CAPACITY:
        check(r.just)
        sel = (np.broadcast_to(r.pred(ts, hs, td, hd), off.shape) if r.pred else np.zeros_like(off))
        rows.append(dict(group="capacity", rule=r.name, path=r.conjunct, modes=r.modes, just=r.just, used=used(sel)))
    if bad_names:
        raise AssertionError(f"theorem names not in the Lean library / PROOFS.md: {sorted(bad_names)}")
    return rows


def headline_gaps(rows):
    """Headline (payable) rules that are REFUTED or UNJUSTIFIED, and exact-slice rules justified only approximately."""
    gaps = [r for r in rows if "payable" in r["modes"] and r["just"].verdict in ("REFUTED", "UNJUSTIFIED")]
    exact_slice = [r for r in rows if "exact" in r["modes"] and r["just"].verdict in ("PROVED", "MEASURED", "CODE",
                                                                                      "CATALOG")
                   and r["just"].kind not in ("exact", "")]
    return gaps, exact_slice


def write(rows, path, n_points, props=None):
    gaps, exs = headline_gaps(rows)
    cnt = lambda g: [r for r in rows if r["group"] == g]
    with open(path, "w") as f:
        w = lambda x="": f.write(x + "\n")
        u = lambda r: f"{r['used']['exact']:,} / {r['used']['payable']:,} / {r['used']['mixgap']:,}"
        th = lambda r: ", ".join(f"`{t}`" for t in r["just"].thms) or "—"
        w("# Edge justification: every rule the relation allows, and the theorem behind it")
        w()
        w("Generated by `python -m verify.z3_relation --edges` (verify/edge_table.py) from `graph.model_graph`: the slot "
          "transitions are enumerated from `slot_reaches` / `regime_reaches` with the strictness `emul_matrix` derives per "
          "mode (as the Z3 model's tables are), decomposed into the order's hops; the ZERO_EDGES; every feature a host may "
          "lack or carry (`feats_ok` on one-feature sets); and the capacity rules, one or more per numeric conjunct of "
          "`emul_matrix` (`verify/z3_relation.py` Model.parts). Theorem names: `proofs/PROOFS.md` (Lean 4 + "
          "Mathlib, namespace EP); every name is checked against the Lean sources.")
        w()
        w("Verdicts: **PROVED** (a theorem covers the rule as the relation uses it; kind exact / limit / bound / regime), "
          "**CODE** (the host's own code path, no construction: a null config value, an identical layer), "
          "**CATALOG** (verified instance (catalog), theorem pending: an end-to-end instance in the reduction catalog "
          "builds and checks the rule, its one-line theorem is not written yet -- not counted as proved), "
          "**MEASURED** (an approximate edge whose payability is a recorded measurement), **REFUTED** (a theorem proves "
          "the rule false as used), **UNJUSTIFIED** (nothing). A *headline* rule is one the payable relation allows. "
          f"`real edges` counts ordered pairs of the {n_points} real shapes in the exact / payable / mixgap matrix whose "
          "values are that transition (a capacity rule: whose pair uses it). `capacity` mode ignores every slot but gen "
          "and is not listed per row. Transitions into the abstract tops (AffineNorm, pos_top, act_top) and branch sets "
          "are omitted.")
        w()
        for g in ("slot", "zero", "feature", "capacity"):
            rs = cnt(g)
            hd = [r for r in rs if "payable" in r["modes"]]
            w(f"- {g} rules: {len(rs)} ({len(hd)} headline): "
              + ", ".join(f"{v} {sum(1 for r in hd if r['just'].verdict == v)}" for v in _RANK))
        w()
        w(f"**{len(gaps)} headline rules are UNJUSTIFIED or REFUTED** "
          f"({sum(1 for r in gaps if r['used']['payable'])} carry payable real edges); "
          f"{len(exs)} exact-slice rules are justified only as a limit / bound / regime / measurement.")
        w()
        w("## Headline gaps (UNJUSTIFIED / REFUTED rules the payable relation allows)")
        w()
        w("| rule | verdict | theorem(s) | why | evidence | real edges (exact / payable / mixgap) |")
        w("|---|---|---|---|---|---|")
        for r in sorted(gaps, key=lambda r: (-r["used"]["payable"], r["rule"])):
            w(f"| {r['rule']} | {r['just'].verdict} | {th(r)} | {r['just'].note} | {r['just'].evidence or '—'} | {u(r)} |")
        w()
        w("## Exact-slice rules justified only approximately")
        w()
        w("| rule | kind | theorem(s) | note | real edges (exact / payable / mixgap) |")
        w("|---|---|---|---|---|")
        for r in exs:
            w(f"| {r['rule']} | {r['just'].kind} | {th(r)} | {r['just'].note} | {u(r)} |")
        for g, title in (("slot", "Every allowed slot transition"), ("zero", "ZERO_EDGES"),
                         ("feature", "Every allowed feature transition"),
                         ("capacity", "Every capacity rule (one or more per numeric conjunct of emul_matrix)")):
            w()
            w(f"## {title}")
            w()
            w("| rule | " + ("path | " if g == "slot" else ("conjunct | " if g == "capacity" else ""))
              + "modes | verdict | kind | theorem(s) | note | evidence | real edges (exact / payable / mixgap) |")
            w("|---|" + ("---|" if g in ("slot", "capacity") else "") + "---|---|---|---|---|---|---|")
            for r in cnt(g):
                extra = f"{r['path']} | " if g in ("slot", "capacity") else ""
                w(f"| {r['rule']} | {extra}{', '.join(r['modes']) or 'opt-in'} | {r['just'].verdict} | {r['just'].kind or '—'} | "
                  f"{th(r)} | {r['just'].note or '—'} | {r['just'].evidence or '—'} | {u(r)} |")
        if props:
            w()
            w("## Properties of the relation (verify/z3_relation.py)")
            w()
            for line in props:
                w(line)
    return gaps, exs
