"""Coverage matrix: every value the classifier can emit, against the host, the proofs and the builder.

    PYENV_VERSION=3.12.6 python verify/sweep/coverage.py      # -> verify/sweep/coverage.json, coverage.md

Rows: every SLOT value (graph.model_graph.SLOTS orders + the qk-norm variants + every value classify emits),
every FEATURES key, every rf_* recurrent family (RECURRENT_FAMILY + the variants _recurrent_family returns),
plus the sweep's new findings (verify/sweep/findings.json, when present).

Columns
  models             configs in graph/hub_10k.json whose classify() carries the value (computed)
  classifier         where graph/model_graph.py reads it (static citation)
  universal_host     the elephantine/universal.py branch that carries it and the test that runs it (static,
                     from the code and graph/minimal_emulator.py's host_point / HOST_FEATS / FAMILY_STATUS),
                     plus, computed, how many of the value's models a TESTED whole-model loader accepts
                     (elephantine/loader_coverage.classify_model)
  rule_target        the rule when the value is on the TARGET: the ORDER edges out of it (computed) and the proof /
                     measurement (static, proofs/PROOFS.md names)
  rule_host          the same, the value on the HOST
  emulate            can elephantine/emulate.py (v1) build it (static, from plan_construction / FAMILIES), plus,
                     computed, how many of the value's models side_support accepts as target / as host
Nothing here runs remote code: configs are read as JSON; the loaders are only asked whether they ACCEPT a config.
"""
import collections
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from graph import model_graph as MG  # noqa: E402

PROOFS = "proofs/PROOFS.md"
TU = "tests/test_universal_elephantine.py"
TS = "tests/test_stock_loaders.py"

# --- static judgements, each with its citation ------------------------------------------------------------------
# key -> (universal host branch + test, rule as target, rule as host, emulate.py v1)
# "rule" strings name the theorem (PROOFS.md) or the measurement file; the ORDER edges are added by the code.
S = {}


def row(key, uh, rt, rh, em):
    S[key] = dict(universal_host=uh, rule_target=rt, rule_host=rh, emulate=em)


# norm
row("norm=RMSNorm", f"yes: UNorm (weight, eps); NORM_STATUS tested vs real Llama ({TU})",
    "on a LayerNorm host at 2x width: PROVED exact (rmsNorm_sub_layerNorm_mirror, PROOFS.md)",
    "plays LayerNorm targets: PROVED exact (layerNorm_sub_rmsNorm, affine_fold; LN beta needs an lm_head bias, "
    "REFUTED exact without one: const_readout_channel)",
    "yes: every v1 family is RMSNorm (FAMILIES; eps' = eps d/D in plan_construction)")
row("norm=LayerNorm", f"yes: UNorm centre = 1 + bias; NORM_STATUS tested vs real Phi-4-flash ({TU}), stock GPT-2 / OPT / "
    f"Bloom / Falcon ({TS})",
    "on an RMSNorm host: PROVED exact (layerNorm_sub_rmsNorm; needs host biases / lm_head bias for beta)",
    "plays RMSNorm targets at 2x width: PROVED exact (mirror, rmsNorm_sub_layerNorm_mirror)",
    "no: no v1 family is LayerNorm (family_of refuses)")
row("norm=AffineNorm", "abstract top (NORM_ORDER), not a model value", "-", "-", "-")
# pos
row("pos=rope", f"yes: UAttention.inv_freq settable (loaders _rope_of / _attn_any); tested in every stock RoPE case ({TS})",
    "on a partial_rope host: PROVED exact rot-gated (partial_rope_host; partial_rope_too_few impossible below rot_t)",
    "plays nope / partial_rope (nope_sub_rope, PROVED exact) and alibi (alibi_sub_rope_limit: PROVED limit, causal)",
    "yes: RoPE in every v1 family; theta rescaled on a head_dim change (dim_map), pointwise scalings only")
row("pos=partial_rope", f"yes: inv_freq = 0 on the unrotated pairs + _partial_rope_perm; tested phi3-partial-window, "
    f"phi, nemotron, persimmon ({TS}), Pythia ({TU})",
    "on a rope host: PROVED exact (nope_sub_rope on the frozen pairs); on a partial host: rot-gated (partial_rope_host)",
    "plays rope targets rot-gated (partial_rope_host), nope (exact), alibi (limit)",
    "partial: only an equal partial_rotary_factor at equal head_dim (plan_construction rope checks; phi3)")
row("pos=nope", f"yes: inv_freq = 0 (UAttention); tested granitemoehybrid-dense-nope / smollm3-nope ({TS})",
    "on rope / partial_rope / learned_abs / alibi hosts: PROVED exact (nope_sub_rope, nope_sub_learned_abs, nope_sub_alibi)",
    "plays only NoPE targets (rope_theta_cannot_reach_nope: a config-only host cannot freeze; the relation's hosts "
    "write inv_freq)", "no: the v1 families all rotate (no inv_freq = 0 setting in plan_construction)")
row("pos=alibi", f"yes: UAttention.alibi slopes; tested bloom-alibi, falcon-alibi ({TS})",
    "on rope / partial_rope: PROVED limit (alibi_sub_rope_limit, alibi_rope_model_bound; 'mag' currency, not exact); "
    "on learned_abs: PROVED norm-free only, pre-norm REFUTED exact (ALiBi on a learned-abs host, Pre-norm rescue)",
    "plays nope targets (nope_sub_alibi, exact)", "no: no v1 ALiBi family")
row("pos=learned_abs", f"yes: UniversalElephantine.pos table (max_pos); tested GPT-2 real checkpoints ({TU}), OPT ({TS})",
    "only the abstract top: a RoPE layer cannot form it (rope_cannot_learned_abs, PROVED impossible one layer; "
    "learned_abs_two_block PROVED exact with a BOS marker, not used by the relation)",
    "plays nope (nope_sub_learned_abs exact) and alibi at +width (alibi_sub_learned_abs_causal, norm-free; pre-norm open)",
    "no: no v1 learned-abs family")
row("pos=sinusoidal", "partial: the learned-abs table can hold a fixed sinusoid (host_point pos_top), but no XGLM / CTRL "
    "loader exists (loader_coverage.LOADERS)", "only the abstract top (POS_ORDER sinusoidal -> pos_top); no theorem",
    "plays nothing (no edge into it)", "no")
for v in ("pos_top", "additive_bias", "t5_relbias"):
    row(f"pos={v}", "abstract value (POS_ORDER), 0 models", "-", "-", "-")
# kv
row("kv=MHA", f"yes: UAttention is MHA (host_point kv MHA|MLA); every test", "the top of KV_ORDER",
    "plays GQA / MQA by tiling: PROVED exact (gqa_sub_mha, mqa_sub_gqa)", "yes (head_maps)")
row("kv=GQA", f"yes: _tile_kv tiles KV heads; every GQA stock case ({TS})",
    "on GQA / MHA hosts: PROVED exact (gqa_sub_mha, groupOf); fewer host KV heads: kvPack_rope_iff (written frequencies)",
    "plays MQA (mqa_sub_gqa) and GQA with fewer KV heads", "yes (head_maps repeat_interleave)")
row("kv=MQA", f"yes: tiling; tested gpt_bigcode-mqa, falcon-mqa-parallel ({TS})",
    "on GQA / MHA hosts: PROVED exact (mqa_sub_gqa)", "plays MHA / GQA only via one wide KV head (mha_sub_wide_mqa; "
    "default RoPE schedule PROVED impossible: kvPack_default_fails)", "yes (head_maps)")
row("kv=MLA", f"yes: _mla_from (latent-norm features attn_norm_feats); tested deepseek_v2 / v3, glm4_moe_lite, minicpm3, "
    f"youtu, mistral4 ({TS})", "on MHA / GQA: REFUTED for the zoo (mla_latent_norm_not_mha: the latent RMSNorm); "
    "on an MLA host with a wider latent: exact only if eps_h = eps_t r_t / r_h (rmsNorm_eps_inj; relation does not check eps)",
    "plays plain targets: REFUTED exact (mla_host_not_prenorm_target); PROVED limit via the eps lever (leverAttn_*)",
    "no: no v1 MLA family")
row("kv=none", f"n/a: SSM-only sentinel (no attention KV); the host zeroes attention", "-", "-", "no")
# ffn
row("ffn=vanilla", f"yes: UFFN ungated c_gelu / c_relu / c_relu2 on the shared path; GPT-2 / OPT / NeoX ({TU}, {TS})",
    "on a GLU host: PROVED exact (glu_ungated; needs an up bias / constant channel)",
    "plays GLU only at 2x depth with squared units / BOUND for ReLU (swiglu_eq_sq_ffn, yarotsky_product_bound)", "no")
row("ffn=glu", f"yes: UFFN shared SwiGLU path; every Llama-style test", "on a MoE host: PROVED exact (dense_sub_moe, "
    "zero_router_constant_gate)", "plays vanilla targets (glu_ungated); MoE targets only at 2x depth: REFUTED exact "
    "(continuous_host_ne_top1_moe)", "yes (dense targets and hosts)")
row("ffn=moe", f"yes: UFFN routed top-k (renorm, eps, selection bias, groups); mixtral / qwen2_moe / olmoe / dots1 / "
    f"glm4_moe / deepseek ({TS})", "on moe_shared: PROVED exact (moe_sub_moe_shared); MoE <- MoE: moeR_* PROVED under "
    "conditions (duplication, renorm equal); on dense at 2x depth REFUTED exact",
    "plays dense targets: PROVED exact (dense_sub_moe)", "host only: mixtral / qwen2_moe / qwen3_moe (TARGET_FAMILIES excludes MoE)")
row("ffn=moe_shared", f"yes: UFFN s_* shared path + s_router sigmoid gate (s_coef); qwen2_moe, deepseek ({TS})",
    "the top of FFN_ORDER; MoE <- MoE capacity (moeR_*)", "plays moe (moe_sub_moe_shared_gated, shared_gate_half) and dense",
    "host only: qwen2_moe / qwen3_moe with shared experts ('shared' mode)")
# act (floors are measured: ACT_FLOORS, act/activation_cross.py; payable threshold act/act_swap_agreement.py)
_AF = "ACT_FLOORS (act/activation_cross.py; payable <= ACT_FLOOR_MAX 0.0005 calibrated by act/act_swap_agreement.py)"
row("act=SiLU", f"yes: UFFN silu(g) * u (ACT_STATUS silu tested vs real Llama, {TU})",
    f"one-hop measured {_AF}; SiLU <-> swigluoai PROVED exact in the clamp regime (swigluoai_sub_silu)",
    "plays ReLU (relu_sub_silu, PROVED limit), swigluoai (fold); GeLU NOT PROVABLE (floor 0.0139)", "yes (ACT_MATH silu)")
row("act=GeLU", f"yes: UFFN c_gelu / c_geglu erf (gelu_tanh = 0); real Pythia ({TU})",
    f"measured {_AF}: no swish host within threshold (GeLU ~ SiLU NOT PROVABLE, 0.0139)",
    "plays ReLU (relu_sub_gelu, PROVED limit)", "partial: copied as is when both sides share it (ACT_MATH gelu_erf)")
row("act=GeLU_tanh", f"yes: c_gelu / c_geglu with gelu_tanh = 1; real GPT-2, Gemma-3-270m ({TU})",
    f"measured {_AF}", "plays ReLU (limit)", "yes for Gemma families (gelu_pytorch_tanh both sides)")
row("act=ReLU", f"yes: UFFN c_relu (ACT_STATUS relu tested vs OPT, {TS})",
    "on SiLU / GeLU / mish / swigluoai: PROVED limit (relu_sub_silu, relu_sub_gelu, relu_sub_mish, relu_sub_swigluoai)",
    "plays nothing else (no host reaches a smooth act exactly)", "partial: only same-act pairs (ACT_MATH relu); no v1 ReLU family")
row("act=relu2", f"yes: UFFN c_relu2 (ACT_STATUS relu2 tested vs real CENO, {TU}); nemotron_h ({TS})",
    "on xIELU: PROVED limit, never exact (relu2_sub_xielu); swish hosts PROVED never exact (relu_sq_not_exact_silu)",
    "plays nothing else", "no")
row("act=mish", f"yes: UFFN c_mish (stock Llama hidden_act='mish', {TU})", f"measured {_AF} (mish ~ SiLU NOT PROVABLE, 0.10)",
    "plays ReLU (relu_sub_mish, PROVED limit)", "partial: same-act pairs only")
row("act=swigluoai", f"yes: UFFN alpha fold + clip_lim + up_shift (ACT_STATUS swigluoai tested vs stock gpt-oss, {TS})",
    "on SiLU: PROVED exact in the clamp regime (swigluoai_sub_silu; needs up bias)", "plays SiLU (fold), ReLU (limit)", "no")
row("act=xielu", f"yes: UFFN c_xielu (ACT_STATUS xielu tested vs real Apertus, {TU})", f"measured {_AF}",
    "plays relu2 (relu2_sub_xielu, PROVED limit)", "no")
row("act=situ", f"yes: UFFN c_situ (ACT_STATUS situ tested vs real Mini-K3, {TU})", f"measured {_AF} (ACT_FLOORS row situ)",
    "per ACT_FLOORS", "no")
row("act=poly_norm", f"yes: UFFN c_poly (ACT_STATUS poly_norm tested vs Motif-3, {TU})",
    "none: not in ACT_FLOORS (incomparable, own value)", "none", "no")
row("act=act_top", "abstract top", "-", "-", "-")
# block
row("block=sequential", f"yes: ULayer seq = 1 (default)", "the top of BLOCK_ORDER; on a parallel host at 2x depth "
    "PROVED exact (seq_sub_parallel; not an edge of BLOCK_ORDER)", "plays parallel at 2x width: PROVED norm-free, REFUTED "
    "under a pre-norm (parallel_sub_seq_prenorm_fails; PAR_CARRY_PAYABLE False)", "yes")
row("block=parallel", f"yes: ULayer.par; tested cohere parallel ({TU}), stablelm-parallel / falcon-mqa-parallel / gptj ({TS})",
    "on sequential: REFUTED exact under a pre-norm (parallel_sub_seq_prenorm_fails); measured carry not payable",
    "plays sequential at 2x depth (seq_sub_parallel)", "no (no v1 parallel family)")
# window
row("window=full", "yes: UAttention.window = inf", "plays local by masking (local_sub_full); ZERO_EDGES full <- local by depth",
    "plays local targets: PROVED exact (local_sub_full)", "yes")
row("window=local", f"yes: UAttention.window (+ n_sink); mistral-window / qwen2-window / gemma3 ({TS})",
    "on full: PROVED exact (local_sub_full, pattern_sub_full)", "plays full targets only by zeroing its sliding layers "
    "(ZERO_EDGES, zeroed_sublayer)", "partial: one window size; 'global' hosts need every target layer windowed (_windows)")
# qk-norm
_QN = ("qk-norm <-> none: REGIME, measured NOT payable (block/fold_agreement.py; knorm_not_plain PROVED IMPOSSIBLE for "
       "a plain host; leverAttn_* PROVED limit for switching a host's norm off)")
row("qknorm=no", "yes", _QN, _QN, "yes (both sides without qk-norm)")
_QH = {"head": "per-head (H, hd) gains q_nw / k_nw; qwen3 / gemma3 / apertus",
       "head_distinct": "per-head-distinct gains are the (H, hd) weight; PLaMo-2 ({TU})",
       "head_free": "gains = 1", "head_post": "qk_post = 1; hunyuan-qk-post ({TS})",
       "proj": "qk_full = 1; olmo2 / olmo3 / flex_olmo ({TS})", "proj_ln": "qk_full + qk_centre; olmo-clip ({TS})",
       "head_ln": "qk_centre (LayerNorm per head); cohere-qk-norm, stablelm-qk-ln ({TS})",
       "head_ln_bias": "qk_centre + q_nb / k_nb biases (Chameleon: no loader)",
       "ln_shared_bias": "qk_centre + shared bias; persimmon ({TS})"}
for v in MG.QKNORM_VARIANTS:
    nests = MG.QKNORM_NEST.get(v, ())
    into = [k for k, hs in MG.QKNORM_NEST.items() if v in hs]
    uh = (f"yes: UAttention._qk_norm, {_QH[v].format(TU=TU, TS=TS)} (minimal_emulator HOST_QKNORM)" if v in _QH else
          "no: not in HOST_QKNORM (minimal_emulator: 'not DeepSeek-V4's (K = V norm) nor CENO's input renormalisation'; "
          "CENO carried by from_ceno's two-stage norm _norm_chain, tested_loader count below)")
    row(f"qknorm={v}", uh,
        (f"on {', '.join(nests)} hosts exact (QKNORM_NEST weight settings); " if nests else "same variant only; ") + _QN,
        (f"plays {', '.join(into)} targets exact (QKNORM_NEST); " if into else "") + _QN,
        "yes for 'head' (qwen3 / gemma3_text, same both sides); no otherwise" if v == "head" else "no")
# softcap
_RP = "REGIME_PAYABLE measured (attn/regime_agreement.py, data attn/data/regime_agreement.jsonl)"
row("softcap=no", "yes: cap = inf", f"on attn / final hosts with the cap large: payable ({_RP}; cap_model_limit PROVED limit)",
    f"plays final-capped targets (final_softcap_greedy PROVED exact for greedy); attention caps NOT payable ({_RP})",
    "yes (equal softcaps both sides)")
row("softcap=attn", f"yes: UAttention.cap; gemma2-softcaps ({TS})", f"on no / final hosts: NOT payable ({_RP}; Gemma-2-9B 75%)",
    f"plays no / final targets with the cap large: payable ({_RP})", "partial: only equal caps both sides (gemma2)")
row("softcap=final", f"yes: UniversalElephantine.final_cap; gemma2-softcaps ({TS}), Gemma-4 ({TS})",
    "on no hosts: PROVED exact for greedy (final_softcap_greedy), payable", f"plays no targets (cap large): payable ({_RP})",
    "partial: only equal caps")
# mixer
_MG = "near-gap NOT payable, measured (attn/mixer_gap_agreement.py, MIXER_GAP); softmax not linear PROVED (no_finite_feature_map)"
row("mixer=softmax", f"yes: UAttention in every layer", "plays nothing recurrent (softmax not in any linear family)",
    f"plays ssm / linear / hybrid targets: {_MG}", "yes (v1 is softmax-only)")
for v in ("ssm", "linear", "hybrid"):
    row(f"mixer={v}", "yes: USSM gated-delta block (host_point mixer softmax|hybrid|ssm|linear); per-family status in the rf_* rows",
        f"on a softmax host: {_MG}; on a same-family host: the rf_* / rc_* capacity rules",
        "hybrid hosts play softmax targets by zeroing the recurrent layers (ZERO_EDGES, zeroed_sublayer, PROVED exact)"
        if v == "hybrid" else "plays nothing softmax", "no")
# sink / cla / mtp / gen / loop
row("sink=no", "yes", "on a sink host: REFUTED as exact (sink_host_ne_plain); PROVED limit b -> -inf (sink_model_limit)",
    "plays sink targets: drop-sinks regime, measured not payable (SINK_ADD_PAYABLE False, block/fold_agreement.py "
    "gpt-oss 97.5% / 33%)", "yes")
row("sink=yes", f"yes: UAttention.null_logit (learned per-head sinks: gpt_oss, granite_swa, mimo_v2_flash; {TS})",
    "on a plain host: measured NOT payable (drop-sinks; sink_sub_plain PROVED exact only with a q bias + zero-frequency pair)",
    "plays plain targets (fold): REFUTED as exact, PROVED limit (sink_host_factor, sink_to_plain_limit)", "no")
row("crosslayer_kv=no", "yes", "on a CLA host by depth (ZERO_EDGES: zero the KV-sharing layers)",
    "plays CLA targets: width carry REFUTED exact under a pre-norm (cla_carry_prenorm_fails; CLA_CARRY_PAYABLE False)", "yes")
row("crosslayer_kv=yes", f"yes: UAttention.kv_read / kv_write / kv_slot (KV register); Hymba ({TU}), Gemma-4 kv-share ({TS})",
    "on a plain host: REFUTED exact (cla_carry_prenorm_fails), measured ~6e-2 floor (block/prenorm_carry.py)",
    "plays plain targets by zeroing the sharing layers (zeroed_sublayer)", "no")
row("mtp=no", "yes (auxiliary heads ignored)", "free both ways: PROVED exact (mtp_sub_plain)", "free (mtp_sub_plain)", "yes")
row("mtp=yes", "yes (the MTP heads are not part of the main logits; not built)", "free: PROVED exact (mtp_sub_plain)",
    "free (mtp_sub_plain)", "n/a (no v1 family carries MTP layers)")
row("gen=ar", "yes: causal mask", "on diffusion / block hosts: PROVED exact (causal_attn_layer, causal_stack)",
    "plays no diffusion target (causal_ne_bidirectional, PROVED impossible)", "yes")
row("gen=diffusion", f"yes: full mask (FAMILY_STATUS diffusion tested, {TU}; test_diffusion_real)",
    "only on diffusion / block hosts (GEN_ORDER mask)", "plays ar (causal mask) and block targets", "no")
row("gen=block", "no: block-causal / caller-supplied mask is an OPEN host branch (minimal_emulator FAMILY_STATUS block-causal)",
    "on a diffusion host (GEN_ORDER mask; code reading of modeling_sdar_moe.py, no theorem)", "plays ar / diffusion (mask)", "no")
row("loop=untied", "yes (untied stack)", "the top of LOOP_ORDER", "plays looped targets by unrolling: PROVED exact (looped_sub_untied)", "yes")
row("loop=looped", f"yes by unrolling (host is untied); LFM3-RD unrolled ({TU})",
    "on an untied host at the unrolled depth: PROVED exact (looped_sub_untied)",
    "plays untied targets: REFUTED at the target's width (looped_same_width_fails); periodic targets only", "no")

# FEATURES
_F = {
    "tied_emb": (f"yes: lm_head is its own matrix (set := E^T); every tied test", "free on an untied host: PROVED exact (tied_sub_untied)",
                 "wall: OPEN (PROOFS.md walls: tied host -> untied target)", "yes: front end (tie_word_embeddings set false when needed)"),
    "out_gate": (f"yes: UAttention.gate_coef / g_proj (gate_sp softplus, hg_* head gate); qwen3_next ({TU}), laguna / afmoe ({TS})",
                 "wall: OPEN (out_gate target <- ungated host)", "free: PROVED exact (out_gate_off)", "no"),
    "switch_lora": (f"yes: lora_adapters + switch head; granite_switch ({TU})", "wall: OPEN", "free: PROVED exact (switch_lora_off)", "no"),
    "hyper_conn": (f"yes: UHC (hc_mult Sinkhorn streams); glm5_next / Motif-3 ({TU})", "wall: OPEN (measured / structural)",
                   "wall: OPEN", "no"),
    "engram": (f"yes: UEngram; allgdn ({TU})", "wall: OPEN", "free: PROVED exact (extra_input_off)", "no"),
    "ple": (f"yes: ple_E / ple_P / ple_proj; gemma4-ple-kvshare-keqv ({TS})", "wall: OPEN", "free: PROVED exact (extra_input_off)", "no"),
    "value_emb": ("no: not carried (minimal_emulator HOST_FEATS: 'Not carried: nanochat's value embeddings / smear')",
                  "wall: OPEN", "free: PROVED exact (value_emb_off)", "no"),
    "attn_res": (f"yes: attnres_block / attnres_full; alice_ai, gpts3, mini_k3 ({TU})", "wall: OPEN", "wall: OPEN", "no"),
    "vnorm": (f"yes: UAttention.v_c / v_nw; gemma4 ({TS})", "wall: PROVED IMPOSSIBLE (vnorm_not_plain)",
              "wall: not exact; PROVED limit via the eps lever (leverAttn_*), not in the relation", "no"),
    "bias_qkv": (f"yes: _lin(bias=True) q/k/v; qwen2 / llama-biases ({TS})", "wall: PROVED IMPOSSIBLE (biased_proj_not_bias_free)",
                 "free: PROVED exact (bias_off)", "yes: qwen2 'qkv' and llama attention_bias families; refused onto a bias-free host"),
    "bias_out": (f"yes: o_proj / FFN / lm_head biases, UNorm bias; GPT-2 / OPT / NeoX ({TU}, {TS})", "wall: OPEN (no construction known)",
                 "free: PROVED exact (bias_off)", "partial: llama o_proj (attention_bias) and mlp_bias only"),
    "post_norm": (f"yes: post_a / post_f with post_coef; gemma2/3 ({TS}, {TU}), glm4-sandwich ({TS})",
                  "wall: OPEN (measured 14% top-1, block/fold_agreement.py OLMo-2)", "wall: OPEN (eps lever limit postNorm_model_limit not in the relation)",
                  "partial: gemma2 / gemma3_text only, sandwich on both sides (plan_construction)"),
    "post_only": (f"yes: UNorm.ident pre-norm + post norms; olmo2 / olmo3 / exaone4 ({TS})", "wall: OPEN (post-norm target, Pre-norm rescue)",
                  "wall", "no"),
    "router_sigmoid": (f"yes: UFFN.score_c; deepseek_v3 / glm4_moe / dots1 ({TS})",
                       "priced by moe_why: a softmax host PROVED IMPOSSIBLE (softmax_router_ne_sigmoid)",
                       "plays softmax routers: PROVED limit (sigmoid_router_limit), not exact", "no (MoE targets refused)"),
    "router_bias": (f"yes: UFFN.sel_bias; deepseek_v3, lfm2_moe ({TS}, {TU})", "wall: PROVED IMPOSSIBLE (router_bias_not_bias_free)",
                    "free: PROVED exact (router_bias_off)", "no"),
    "shared_gate": (f"yes: UFFN.s_router / s_coef; qwen2_moe ({TS}), qwen3_next ({TU})", "wall: OPEN",
                    "free: PROVED exact (moe_shared_gate_off, shared_gate_half)", "no"),
    "router_group": (f"yes: UFFN.grp buffer; deepseek_v2-group-limited, axk1 ({TS})",
                     "priced by moe_why: PROVED sufficient with copies in own groups (moeR_group_copies); REFUTED same-shape m>1",
                     "priced by moe_why (OPEN against an ungrouped target)", "no"),
    "router_in_gate": ("no: no Llama-4 loader (loader_coverage.LOADERS has no llama4)", "wall (FEATURES; f(s x) != s f(x)); no theorem",
                       "wall; no theorem", "no"),
    "nope_layers": (f"yes: per-layer inv_freq (0 on NoPE layers); smollm3-nope ({TS})", "wall: OPEN (truncation measured not payable)",
                    "wall: OPEN", "no"),
    "dsa": (f"yes: UAttention DSA indexer (dsa_*); glm_moe_dsa / deepseek_v32 ({TS})",
            "free (treated as dense; exact while context <= index_topk): OPEN beyond", "free: OPEN beyond index_topk", "no"),
    "kv_compress": ("no: no DeepSeek-V4 loader (compressed KV pools not a UAttention branch)", "wall; no theorem", "wall; no theorem", "no"),
    "post_ln": (f"yes: stream_norms (sn_in / sn_out); opt-post-ln ({TS})", "wall (FEATURES); Pre-norm rescue: OPEN / REFUTED natural constructions",
                "wall", "no"),
    "clip_qkv": (f"yes: UAttention.qkv_clip; olmo-clip / olmoe-clip ({TS})", "wall (regime, not measured); no theorem",
                 "wall; no theorem", "no"),
    "act_clip": (f"yes: UFFN.c_clip / clip_lim; gigachat35 ({TU})", "wall (regime, not measured); swigluoai clamp-regime theorems analogous",
                 "wall", "no"),
    "router_sqrtsoftplus": ("no: UFFN.score_c is softmax / sigmoid only (no sqrt(softplus)); no DeepSeek-V4 loader",
                            "priced by moe_why (score must match); no theorem", "priced by moe_why", "no"),
    "hash_route": ("no: no token-id routing branch in UFFN", "wall; no theorem", "wall; no theorem", "no"),
    "selfcond": ("no: no self-conditioning input in UniversalElephantine", "wall; no theorem", "free (zero the signal); no theorem", "no"),
    "selfcond_norm": ("no", "wall; no theorem", "wall; no theorem", "no"),
    "zamba_shared": ("no: no Zamba loader (FEATURES text)", "wall; no theorem", "wall; no theorem", "no"),
}
for k, (uh, rt, rh, em) in _F.items():
    row(f"feat={k}", uh, rt, rh, em)

# recurrent families: FAMILY_STATUS key in graph/minimal_emulator.py that carries each
_RF = {"mamba1": "mamba1", "mamba1_bcdt": "mamba1 (falcon_mamba: from_falcon_mamba)", "mamba1_bcdtw": "mamba1 (jamba: from_jamba)",
       "mamba1_mh": None, "hymba": "hymba", "sambay": "phi4flash", "mamba2": "mamba2", "mamba2_nonorm": "mamba2 (falcon_h1-no-norm)",
       "mamba2_nbg": "mamba2 (falcon_h1-gate-norm)", "plamo2": "plamo2", "gdn": "gated-deltanet", "gdn_negeig": "gated-deltanet "
       "(from_olmo_hybrid, tf5 case olmo_hybrid)", "gdn2": "gpts3", "gka": "gka", "kda": "kda", "kda_lb": "glm5_next / youtu_kda",
       "shortconv": "lfm2 / lfm3 / lfm2_moe", "rglru": "rg-lru", "lightning": "lightning", "rwkv4": "rwkv4", "rwkv5": "rwkv5",
       "rwkv6": "rwkv6", "rwkv7": "rwkv7", "gla": "gla", "retnet": "retnet", "jet": "jet",
       "bailing_lightning": None, "sala_lightning": None, "other_cyrah": None}
_RC = {"mamba1": "RC_GE", "mamba1_bcdt": "RC_GE", "mamba1_bcdtw": "RC_GE", "mamba2": "RC_M2 (RC_GE / RC_EQ for some shapes)",
       "mamba2_nonorm": "RC_GE", "plamo2": "RC_GE", "gdn": "RC_GDN", "gdn_negeig": "RC_GDN", "kda": "RC_KDA", "kda_lb": "RC_KDA",
       "shortconv": "RC_GE"}
for f, fs in _RF.items():
    uh = (f"yes: USSM, FAMILY_STATUS {fs} ({TU})" if fs else
          "no: " + {"mamba1_mh": "Zamba-1 multi-head Mamba: no loader (zamba_shared)",
                    "bailing_lightning": "Ling-2.5 lightning: no loader", "sala_lightning": "MiniCPM-SALA: no loader",
                    "other_cyrah": "FAMILY_STATUS cyrah open (gated code)"}[f])
    rule = _RC.get(f, "RC_EQ")
    row(f"feat=rf_{f}", uh,
        f"same family host only, capacity rule {rule} (model_graph _rc_pair_ok; RC_* comment block, code-argued, no Lean "
        f"theorem); softmax host: {_MG}",
        "plays its own family (rc rule) and, in a hybrid, softmax targets by zeroing (zeroed_sublayer)", "no")


# --- computed columns ---------------------------------------------------------------------------------------------
def order_edges():
    out = collections.defaultdict(lambda: {"as_target": [], "as_host": []})
    for slot, order in MG.SLOTS:
        for t, edges in order.items():
            for h, cur, ex in edges:
                pay = ex or MG.REGIME_PAYABLE.get((slot, t, h), False)
                s = f"{slot}={h} ({cur}{', exact' if ex else ''}{'' if pay else ', not payable'})"
                out[f"{slot}={t}"]["as_target"].append(s)
                out[f"{slot}={h}"]["as_host"].append(f"{slot}={t} ({cur}{', exact' if ex else ''}{'' if pay else ', not payable'})")
    for (slot, t, h), cur in MG.ZERO_EDGES.items():
        out[f"{slot}={t}"]["as_target"].append(f"{slot}={h} (zero edge: {cur})")
        out[f"{slot}={h}"]["as_host"].append(f"{slot}={t} (zero edge: {cur})")
    for k, (tr, hr, _) in MG.FEATURES.items():
        out[f"feat={k}"]["as_target"].append(f"FEATURES target rule: {tr}")
        out[f"feat={k}"]["as_host"].append(f"FEATURES host rule: {hr}")
    return out


def computed():
    from elephantine import loader_coverage as LC
    from elephantine import emulate as EM
    models = json.load(open("graph/hub_10k.json"))
    cnt, tested, em_t, em_h = (collections.Counter() for _ in range(4))
    for m in models:
        cfg = m.get("config") or {}
        try:
            c = MG.classify(cfg)
        except Exception:
            c = None
        if not c:
            continue
        keys = [f"{s}={c[s]}" for s, _ in MG.SLOTS] + [f"feat={f}" for f in c["feats"].split("+") if f]
        try:
            ld, ok, _, _ = LC.classify_model(cfg)
        except Exception:
            ok = False
        try:
            st, sh = EM.side_support(cfg, "target") is None, EM.side_support(cfg, "host") is None
        except Exception:
            st = sh = False
        for k in keys:
            cnt[k] += 1
            tested[k] += bool(ok)
            em_t[k] += st
            em_h[k] += sh
    return cnt, tested, em_t, em_h


WHERE = {"norm": "classify: norm block (NORM_BY_CODE, rms_norm_eps / layer_norm_eps keys)",
         "pos": "classify: positional block (_is_alibi, SINUSOIDAL_TYPES, LEARNED_ABS, _rotary_factor, _nope_attention)",
         "kv": "classify: KV block (kv_lora_rank, multi_query, num_key_value_heads)",
         "ffn": "classify: FFN block (n_exp, _has_shared_expert, VANILLA / UNGATED_BY_CODE)",
         "act": "classify: raw_act (ACT_KEY_BY_CODE, GELU_FORM_BY_CODE, RAW_ACT_BY_CODE)",
         "block": "classify: use_parallel_residual / parallel_attn / PARALLEL_BY_CODE",
         "window": "classify: sliding_window / layer_types / _block_experts",
         "qknorm": "classify: qk-norm keys + QKNORM_ALWAYS / _qknorm_variant",
         "softcap": "classify: attn_logit_softcapping / final_logit_softcapping",
         "mixer": "classify: SSM_TYPES / LINEAR_TYPES / HYBRID_TYPES / structural fallback",
         "sink": "classify: _SINK_KEYS / SINK_TYPES", "crosslayer_kv": "classify: CLA_TYPES / _kv_source_shared",
         "mtp": "classify: num_nextn_predict_layers / num_mtp_layers", "gen": "_gen (DIFFUSION_TYPES, BLOCK_DIFFUSION_TYPES)",
         "loop": "_looped (LOOPED_TYPES, _LOOP_COUNT_KEYS)"}


def main():
    edges = order_edges()
    cnt, tested, em_t, em_h = computed()
    keys = list(S)
    for k in cnt:                               # a value classify emits that the static table lacks
        if k not in S:
            keys.append(k)
    rows = []
    for k in keys:
        slot, val = k.split("=", 1)
        st = S.get(k, dict(universal_host="unclear: no static entry (new value)", rule_target="unclear",
                           rule_host="unclear", emulate="unclear"))
        where = (f"_feats (FEATURES['{val}'])" if slot == "feat" and val in MG.FEATURES else
                 "_rf_feats / _recurrent_family (RECURRENT_FAMILY, RF_NESTS)" if slot == "feat" and val.startswith("rf_") else
                 WHERE.get(slot, "?"))
        e = edges.get(k, {"as_target": [], "as_host": []})
        n = cnt.get(k, 0)
        rows.append(dict(key=k, models=n, read_by_classifier=True, classifier_where=where,
                         universal_host=st["universal_host"], tested_loader_models=tested.get(k, 0),
                         rule_target=st["rule_target"], rule_host=st["rule_host"],
                         order_edges_as_target=e["as_target"], order_edges_as_host=e["as_host"],
                         emulate=st["emulate"], emulate_target_models=em_t.get(k, 0), emulate_host_models=em_h.get(k, 0)))
    fp = os.path.join(HERE, "findings.json")
    if os.path.exists(fp):
        for f in json.load(open(fp)):
            rows.append(dict(key=f["key"], models=f.get("live_models", f.get("models", 0)), read_by_classifier=False,
                             classifier_where="no (sweep finding, " + f.get("direction", "?") + "; "
                             + f.get("note", "")[:140] + ")",
                             universal_host="not checked (a value the classifier does not emit)",
                             tested_loader_models=0, rule_target="none (unmapped)", rule_host="none (unmapped)",
                             order_edges_as_target=[], order_edges_as_host=[],
                             emulate="no (emulate.py builds classifier values only)", emulate_target_models=0,
                             emulate_host_models=0, finding=True))
    json.dump(rows, open(os.path.join(HERE, "coverage.json"), "w"), indent=1)
    md = ["| feature value | models | classifier | universal host (tested-loader models) | rule as target | rule as host "
          "| emulate.py (v1 target / host models) |", "|---|---|---|---|---|---|---|"]
    esc = lambda s: str(s).replace("|", "/").replace("\n", " ")
    for r in rows:
        md.append(f"| {r['key']} | {r['models']} | {'yes' if r['read_by_classifier'] else 'NO'}: {esc(r['classifier_where'])} "
                  f"| {esc(r['universal_host'])} ({r['tested_loader_models']}) | {esc(r['rule_target'])} | {esc(r['rule_host'])} "
                  f"| {esc(r['emulate'])} ({r['emulate_target_models']} / {r['emulate_host_models']}) |")
    open(os.path.join(HERE, "coverage.md"), "w").write(
        "# Coverage matrix\n\nGenerated by `verify/sweep/coverage.py` (do not edit). Proof names refer to "
        f"`{PROOFS}`; models = configs in graph/hub_10k.json whose classify() carries the value; tested-loader models = "
        "of those, how many a tested whole-model universal-host loader accepts (elephantine/loader_coverage.py).\n\n"
        + "\n".join(md) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
