"""Which carriers of an UNMAPPED finding actually take the code path (the "live" models), per finding.

A finding's carriers are every snapshot model whose code contains one of its hunks. Many findings need a config path
(Gemma-4 enable_moe_block, a key set, a type the table misses): LIVE[key](cfg, lm, cls, dims) says whether a placed
carrier takes it. Findings without a predicate count every placed carrier (the families are small and code-fixed).
build_report.py ranks the UNMAPPED list by live models.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from graph import model_graph as MG  # noqa: E402


def _n(x):
    return MG._num(x)


def _rp(lm):
    r = lm.get("rope_parameters") or lm.get("rope_scaling") or {}
    return r if isinstance(r, dict) else {}


def _feats(c):
    return set((c or {}).get("feats", "").split("+")) - {""}


def _code_head_dim(lm):
    mt = str(lm.get("model_type") or "").lower()
    hidden, heads = _n(lm.get("hidden_size")), _n(lm.get("num_attention_heads"))
    for k in ("attention_head_dim", "kv_channels", "hidden_size_per_head"):
        if _n(lm.get(k)):
            return _n(lm[k])
    if mt == "zamba" and hidden and heads:
        return 2 * hidden // heads
    return 0


def _blocks(lm):
    return [b for b in (lm.get("block_configs") or []) if isinstance(b, dict)]


LIVE = {
    "gemma4_branch_post_norms": lambda cfg, lm, c, d: bool(lm.get("enable_moe_block")),
    "head_dim_key_unread": lambda cfg, lm, c, d: bool(_code_head_dim(lm)) and _code_head_dim(lm) != d.get("head_dim"),
    "out_gate_by_code_unlisted": lambda cfg, lm, c, d: "out_gate" not in _feats(c),
    "qknorm_by_code_unlisted": lambda cfg, lm, c, d: c.get("qknorm") == "no",
    "renorm_by_code_unlisted": lambda cfg, lm, c, d: c.get("ffn") in ("moe", "moe_shared"),
    "embedding_norm": lambda cfg, lm, c, d: str(lm.get("model_type")).lower() in (
        "bloom", "gaggle", "muse_glimmer", "muse_glimmer_text", "nanochat_gpt"),
    "falcon_alibi_slope_scale": lambda cfg, lm, c, d: str(lm.get("model_type")) == "falcon" and c.get("pos") == "alibi",
    "hyper_conn_variant": lambda cfg, lm, c, d: "hyper_conn" in _feats(c),
    "position_query_temperature": lambda cfg, lm, c, d: bool(
        lm.get("use_logn_attn") or _rp(lm).get("llama_4_scaling_beta") or lm.get("llama_4_scaling_beta")
        or (str(lm.get("model_type")) in ("llama4", "llama4_text") and lm.get("attn_temperature_tuning", True))),
    "dsv4_layer_types_schema": lambda cfg, lm, c, d: bool(
        (lm.get("layer_types") and "compress_ratios" not in lm) or (lm.get("mlp_layer_types") and "num_hash_layers" not in lm)),
    "grouped_lowrank_oproj": lambda cfg, lm, c, d: bool(_n(lm.get("o_groups")) or _n(lm.get("o_lora_rank"))),
    "norm_placement_keys_unread": lambda cfg, lm, c, d: not (
        str(lm.get("model_type")) == "molmo" and not lm.get("norm_after")),
    "out_gate_activation": lambda cfg, lm, c, d: str(lm.get("model_type")) == "laguna" or bool(lm.get("attention_gate_func")),
    "tie_default_unread": lambda cfg, lm, c, d: "tied_emb" not in _feats(c),
    "factorized_embedding_proj": lambda cfg, lm, c, d: bool(
        _n(lm.get("word_embed_proj_dim")) and _n(lm.get("word_embed_proj_dim")) != _n(lm.get("hidden_size"))),
    "group_mask_zero_fill": lambda cfg, lm, c, d: 1 < _n(lm.get("n_group")) and 0 < _n(lm.get("topk_group")) < _n(lm.get("n_group")),
    "dynamic_ntk_in_context": lambda cfg, lm, c, d: str(lm.get("model_type")) != "qwen" or bool(lm.get("use_dynamic_ntk")),
    "partial_rope_keys_unread": lambda cfg, lm, c, d: bool(
        (_n(lm.get("rope_head_dim")) and _n(lm.get("rope_head_dim")) < (_n(lm.get("head_dim")) or 1e9))
        or (lm.get("rope_pct") is not None and _n(lm.get("rope_pct")) < 1) or isinstance(lm.get("partial_rotary_factors"), list)),
    "bias_qkv_by_code_unlisted": lambda cfg, lm, c, d: "bias_qkv" not in _feats(c),
    "bias_out_by_code_unlisted": lambda cfg, lm, c, d: "bias_out" not in _feats(c),
    "v_head_dim_narrow": lambda cfg, lm, c, d: bool(_n(lm.get("v_head_dim")) and _n(lm.get("v_head_dim")) != _n(lm.get("head_dim"))),
    "grouped_rmsnorm": lambda cfg, lm, c, d: _n(lm.get("layernorm_num_groups")) > 1,
    "router_score_unread": lambda cfg, lm, c, d: c.get("ffn") in ("moe", "moe_shared"),
    "bidirectional_flag_true": lambda cfg, lm, c, d: c.get("gen") == "ar",
    "hybrid_layer_schedule_unread": lambda cfg, lm, c, d: c.get("mixer") == "softmax",
    "mova_routed_values": lambda cfg, lm, c, d: bool(_n(lm.get("mova_num_experts"))),
    "rotary_dim_ignored": lambda cfg, lm, c, d: bool(lm.get("rotary_dim")) and lm.get("partial_rotary_factor") is None,
    "decilm_linear_subblock": lambda cfg, lm, c, d: any(
        (b.get("attention") or {}).get("replace_with_linear") or (b.get("ffn") or {}).get("replace_with_linear")
        for b in _blocks(lm)),
    "decilm_window_sink_tokens": lambda cfg, lm, c, d: any(
        (b.get("attention") or {}).get("window_length") or (b.get("attention") or {}).get("num_sink_tokens")
        for b in _blocks(lm)),
    "prefix_lm_mask": lambda cfg, lm, c, d: str(cfg.get("model_type")) == "DIVEdoc" or bool(
        (lm.get("attn_config") or {}).get("prefix_lm")),
    "use_mem_rope_unread": lambda cfg, lm, c, d: lm.get("use_mem_rope") is not True,
    "mpt_remote_alibi_default_false": lambda cfg, lm, c, d: "alibi" not in (lm.get("attn_config") or {}),
    "mpt_norm_type_unread": lambda cfg, lm, c, d: "layernorm" not in str(lm.get("norm_type", "low_precision_layernorm")),
    "mpt_qk_norm_variant": lambda cfg, lm, c, d: bool((lm.get("attn_config") or {}).get("qk_ln")
                                                      or (lm.get("attn_config") or {}).get("qk_gn")),
    "mpt_ffn_config_unread": lambda cfg, lm, c, d: (lm.get("ffn_config") or {}).get("ffn_type", "mptmlp") != "mptmlp",
    "mpt_attn_config_rope_unread": lambda cfg, lm, c, d: bool((lm.get("attn_config") or {}).get("rope")),
    "router_bias_unread": lambda cfg, lm, c, d: "router_bias" not in _feats(c),
    "sliding_window_keys_unread": lambda cfg, lm, c, d: True,
    "act_clip_per_layer_list": lambda cfg, lm, c, d: any(
        isinstance(lm.get(k), list) and any(_n(x) for x in lm[k])
        for k in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list")),
    "loop_depth_unread": lambda cfg, lm, c, d: c.get("loop") != "looped" or True,
    "input_vocab_extra_rows": lambda cfg, lm, c, d: False,      # vocab is not priced (USE_VOCAB = False)
    "decilm": None,
}
LIVE = {k: v for k, v in LIVE.items() if v}
NOTES = {
    "input_vocab_extra_rows": "vocab is not priced (USE_VOCAB = False): live only under the opt-in vocab axis",
    "decilm_linear_subblock": "no snapshot Nemotron-NAS block sets replace_with_linear",
    "decilm_window_sink_tokens": "no snapshot Nemotron-NAS block sets window_length / num_sink_tokens",
}

# one-line statements for merged / high-ranked findings (the hunk-level reasons stay in findings.json "reasons")
TEXT = {
    "out_gate_by_code_unlisted": (
        "the attention output is always multiplied by sigmoid(W_g x) in code (Qwen4-exp, LFM3 / LFM3-RD, Hy4, HRM-Text, "
        "GPT-S3 / X3, Ethoric, AliceAI, Muse-Glimmer, ...), with no config key _feats reads, so out_gate is missing",
        "add the types to OUT_GATE_TYPES (and read gated_mla / elementwise_attn_output_gate / attn_gate keys)"),
    "qknorm_by_code_unlisted": (
        "q / k norms built unconditionally in code (LFM3, PLaMo-3, LLaDA2-preview, Bitsy, BananaMind, Cyrah, "
        "Hulu-Med-4B, MonkeyOCRv2, Gemma3-MoE, Ethoric (string flag), ...) but the type is not in QKNORM_ALWAYS: qknorm=no",
        "add the types (with their variant) to QKNORM_ALWAYS / QKNORM_VARIANT_BY_TYPE; treat qk_norm strings as true"),
    "renorm_by_code_unlisted": (
        "the router renormalises (or never renormalises) in code whatever norm_topk_prob says (Bailing V2 / V2.5 / V3, "
        "LLaDA2, CENO, Step-3, Arctic, LLaMA-MoE, Gemma3-MoE, AliceAI, ZAQERIS, Meditor, Motif route_norm): moe_sig gives "
        "'?type' (only its own family plays it) or the wrong renorm cap",
        "RENORM_FIXED entries per type / architecture (Bailing v1 does read the key); add route_norm to RENORM_KEYS"),
    "embedding_norm": (
        "a norm on the embedding output (BLOOM word_embeddings_layernorm; weight-free RMSNorm in Muse-Glimmer, gaggle, "
        "nanochat): folds into an untied target's table, but a host with it can only produce normed embeddings",
        "an emb_norm feature: free as target (untied), wall as host"),
    "hyper_conn_variant": (
        "one hyper_conn bit covers different stream mixers: DeepSeek-V4 Sinkhorn comb (comb^T), nanowhale / Xing-4 "
        "comb (no transpose, clamped logits, mean collapse), Hy4 identity streams with 2 sigmoid posts, Qwen4-exp "
        "low-rank gated residual, DeepSeek-V4.1 pre-mix lagged one sub-layer (one-hot start, Sinkhorn-pre head)",
        "a hyper_conn variant value (sinkhorn / sinkhorn_T / sinkhorn_lagged / identity / lowrank_gated) compared for "
        "equality"),
    "gdn_out_gate_act": (
        "Qwen4-exp's GatedDeltaNet output norm is norm(y) * sigmoid(z) (output_gate_type), not Qwen3-Next's norm(y) * "
        "silu(z), under the same rf_gdn family", "an rf_gdn_sigmoid variant read from output_gate_type"),
    "qwen4_ple_reads_ngrams": (
        "Qwen4-exp's 'PLE' is a hashed n-gram embedding over previous tokens (query gate, dilated short conv): it reads "
        "earlier tokens, unlike Gemma's per-token PLE, but is labelled ple", "label it engram (or its own feature)"),
    "position_query_temperature": (
        "q scaled by a position-dependent factor inside the context: Qwen-1 logn (past seq_length), Llama-4 NoPE "
        "attn_temperature_tuning, Ministral-3 / Mistral-4 / Nemotron-Labs-Diffusion llama_4_scaling_beta "
        "(1 + beta log(1 + floor(pos / L0)))", "a qtemp feature (the universal host's qtemp), context-gated"),
    "local_rope_theta_unread": (
        "a second RoPE table on sliding layers (OLMo-3: YaRN on full layers only; PLaMo-3 rope_local_theta) that "
        "_rope_schedules does not see", "read the per-layer-type tables into rope_sched"),
    "grouped_lowrank_oproj": (
        "DeepSeek-V4's o_proj is grouped low-rank (o_groups x o_lora_rank): a rank bound no dims axis reads",
        "an o_rank dims axis (host o_rank >= target's)"),
    "rope_cap_zero_tail_table": (
        "LFM3's partial RoPE pads the table with zero frequencies to head_dim, so the host table covers the head; "
        "_rope_schedules caps it at the rotated width", "cap = head_dim for zero-tail tables"),
    "per_layer_head_shapes": (
        "head counts / KV heads / head_dim per layer (Laguna num_attention_heads_per_layer, MiMo-V2 swa_* heads, DeciLM "
        "per-layer KV heads, Domyn head_sizes) collapsed to one value", "max over layers for capacity, min for q_per_kv"),
    "gemma4_branch_post_norms": (
        "Gemma-4 enable_moe_block: dense MLP and MoE outputs each post-normed, then the sum normed again -- not a "
        "plain shared expert", "a moe sub-value for per-branch post-normed dense + MoE FFNs (masked today by vnorm)"),
    "out_gate_headwise": (
        "head-wise output gates (one sigmoid per head: Laguna, Spark-2.5, Youtu-A7B, Ming) share out_gate with "
        "elementwise ones, so a head-wise host is credited with elementwise targets", "split out_gate by granularity"),
    "norm_placement_keys_unread": (
        "norm placement read from keys / code the classifier ignores: sandwich (PLaMo-3, Ouro, Domyn, Ethoric peri_ln, "
        "cost_wise_gemma), post-only (Molmo norm_after, Bolmo reordered_norm), MiniMax-Text-01 post-LN with final norm",
        "type / key entries in POST_NORM_TYPES / POST_ONLY_TYPES; MiniMax postnorm -> post_ln"),
    "tie_default_unread": (
        "lm_head tied by a class default or in code (Cyrah, Bitsy, Gemma3-MoE, Kanana-2, TinyMixtral, Nexus, Inferno, "
        "Phi-3-small, Mima, CFRD, untyped MiniCPM) with no key the classifier sees", "_CLASS_DEFAULT_FALLBACK entries"),
    "sliding_window_keys_unread": (
        "sliding windows set by keys the window slot does not read (PLaMo-3 window_size + pattern, Motif / gaggle / "
        "Meditor patterns, EdgeRock / Fabric / Ethoric window keys, Jet swa.window_size, MPT attn_config)",
        "read the keys into window / layer kinds"),
    "ceno_megatron_stream_renorm": (
        "CENO blocks renormalise the stream (attention: pre_mlp_norm on the output, no second residual; MoE: the residual "
        "is the normed input)", "a post_ln-style wall by code for ceno"),
    "dynamic_ntk_in_context": (
        "dynamic NTK that changes RoPE inside the context (Qwen-1 past seq_length; Meditor global layers past 8192)",
        "a context-gated rope feature"),
    "bias_qkv_by_code_unlisted": (
        "q / k / v biases hard-coded (Qwen2 bodies under wrapper types: Step-Audio-2, FastVLM, nanoLLaVA, Hulu-Med, "
        "VideoLLaMA3, ArkASR; Sundial; Meditor)", "add the types to BIAS_QKV_TYPES"),
    "config_default_unread": (
        "a remote / tf5 class default the classifier does not know (LLaDA2 use_qk_norm True, Muse-Glimmer "
        "final_logit_softcapping 20, EXAONE-MoE sliding_window 4096)", "_CLASS_DEFAULT_FALLBACK entries"),
    "ffn_width_keys_unread": (
        "FFN widths from keys dims_of does not read (expert_intermediate_size, dense_intermediate_size, ffn_mult, "
        "ffn_hidden, RWKV-5 computed width, RecurrentGemma // 2)", "read the keys / add FUSED_GLU_HALF entries"),
    "v_head_dim_narrow": (
        "V heads narrower than Q / K (MiMo-V2 v_head_dim 128 vs 192) outside MLA; kv_cap uses head_dim",
        "a V-width term in kv_cap"),
    "loop_depth_unread": (
        "loops the classifier misses or miscounts (Ouro total_ut_steps 4, LFM3-RD motif depth, CFRD folded cells)",
        "loop count keys / per-family depth"),
    "head_dim_key_unread": (
        "head_dim from attention_head_dim / kv_channels / hidden_size_per_head (Nemotron-H-4B, Zamba, JetMoE, PLaMo-2-VL); "
        "dims_of derives hidden / heads", "read the aliases (and attribute_map) in dims_of"),
}
