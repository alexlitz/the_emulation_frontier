"""The minimal universal emulator: the smallest single model that emulates them all.

No real model emulates the whole zoo -- the headline frontier has 218 maximal elements (51 even if
every type wall were free) because capacity trades off (wide dense cache vs huge MoE FFN vs deep
stack) and some type walls have no currency. But there is a
*least upper bound*: the coordinate-wise supremum over every model, with the top
categorical value on each slot and each capacity axis raised (with its currency
penalty) until it dominates everything. That join is a real spec -- just larger than
anything anyone has trained.

The host is ONE architecture, `elephantine/universal.py` (`UniversalElephantine`): every
layer is softmax attention (RoPE; a causal mask for autoregressive targets, a full mask for
diffusion targets, since causal attention is the full host masked down), then a general
gated-delta SSM block, then a routed + shared MoE. Every op a family may or may not use is a
branch with a coefficient (qk-norm, output gates, the decay / delta / write formulas, an RMS
norm whose grouping is a weight, the RWKV-4 ratio, the RG-LRU reset), so each target is a
weight setting of the same model:

  - softmax AR and diffusion: the attention sub-block, mask chosen per target;
  - every recurrence: the SSM sub-block. Mamba-1 is a gated-delta recurrence with one head per
    channel (tests/test_mamba1_gated_delta.py, exact against the real MambaMixer), so one SSM
    primitive covers Mamba-1/2, gated-DeltaNet, lightning, RG-LRU, RWKV-4 and LFM2's gated
    short conv.

tests/test_universal_elephantine.py checks the one class against every family's real code:
whole models for Llama, GPT-2, Pythia, Gemma-3 (exact GeLU branches), a mish Llama, a diffusion forward, Mamba, the Qwen3-Next hybrid, PLaMo-2 and Hymba;
blocks for Mamba-2, Mamba-3, LFM2, MiniMax lightning, RWKV-4, RWKV-5, RG-LRU, and (on GPU, fla's
Triton kernels) GLA, RetNet and Kimi KDA; whole models for EdgeRock (score-selected memory),
AllGDN-Engram (hashed n-gram memory), GigaChat-3.5 (gated norms, MLA), LFM2 / LFM2-MoE / LFM3 /
LFM3-RD, CENO, GLM-5.3-Flash (hyper-connections, DSA indexer), Youtu-LLM, Jet-Nemotron, GPT-S3
and MemT. `FAMILY_STATUS` below records the evidence per family. The FFN activation and the norm
are applied inside UFFN / UNorm, so a model is counted as carried only if its activation has a host
branch or a currency edge to one (`ACT_STATUS`: tested / priced / open) and its norm has a branch
(`NORM_STATUS`). Other non-mixer slots (position, softcap, ...) are priced by the currency graph.
Separately, `loaded_split` counts the carried models that are LOADED AND TESTED: a whole-model `from_*`
loader accepts their model_type / config and a test runs it against that type's own code
(elephantine/loader_coverage.py; the stock transformers families in tests/test_stock_loaders.py, incl.
released checkpoints); the rest are carried only by priced slot constructions.

The spec is the per-axis maximum over EVERY model (`join`): the host emulates each of them under
emul_matrix(mode="capacity"), and under the HEADLINE emul_matrix(mode="payable") every model whose
activation / norm the host carries (host_point's branch sets; tests/test_graph_rules.py checks both).

Run:  python -m graph.minimal_emulator --from-json graph/hub_10k.json
"""
from __future__ import annotations

import argparse

from graph.model_graph import (MLA_PENALTY, RC_DIMS, classify, dims_of, iter_json_configs,
                               iter_local_configs, raw_act, _junky, _lm_config)

_MAMBA_TYPES = ("jamba", "nemotron_h", "zamba", "mamba", "falcon_h", "granitemoehybrid",
                "plamo", "hunyuan", "bamba")


def is_mamba_family(cfg):
    """True if the model's recurrence is a Mamba-SSM the hybrid host reproduces exactly."""
    d = _lm_config(cfg)
    mt = str(d.get("model_type", "")).lower()
    if any(k in mt for k in _MAMBA_TYPES):
        return True
    if d.get("mamba_num_heads") or d.get("mamba_d_state") or d.get("ssm_state_size") or d.get("mamba_expand"):
        return True
    lbt = d.get("layers_block_type") or d.get("layer_types") or []
    return isinstance(lbt, list) and any("mamba" in str(x).lower() for x in lbt)


# Recurrent families, keyed by model_type, and how the universal host carries each.
_FAMILY_TYPES = {
    "mamba1": ("mamba", "falcon_mamba", "jamba", "zamba"),
    "mamba2": ("nemotron_h", "nemotron_h_puzzle", "granitemoehybrid", "falcon_h1", "zamba2", "bamba"),
    "gated-deltanet": ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_next", "qwen4_exp_text", "olmo_hybrid",
                       "qwen3_5", "qwen3_5_moe", "qwen4_exp",      # same config schema, top-level type
                       "diba", "agnes_text", "diffuqwen35"),       # Qwen3.5 bodies (remote code)
    "lfm2": ("lfm2",), "lightning": ("minimax", "minimax_text_01", "minimax_m1"), "rwkv4": ("rwkv",),
    "rwkv5": ("rwkv5",), "rwkv7": ("rwkv7",), "alice_ai": ("alice_ai",), "mini_k3": ("mini_k3",),
    "nemotron_flash": ("nemotron_flash",), "phi4flash": ("phi4flash",), "rwkv6": ("rwkv6",),
    "granite_switch": ("granite_switch",), "rg-lru": ("recurrent_gemma",), "kda": ("kimi_linear",), "gla": ("gla",),
    "retnet": ("retnet",), "mamba3": ("qwen3_mamba3",), "plamo2": ("plamo2", "plamo2vl"), "hymba": ("hymba",),
    # new with the 2026-10-19 snapshot (one model each)
    "edgerock": ("edgerock",), "allgdn": ("allgdn",), "gigachat3_5": ("gigachat3_5",),
    # new with the 2026-11-10 snapshot (or newly recognised as recurrent by the classifier fixes)
    "lfm3": ("lfm3", "lfm3_rd"), "lfm2_moe": ("lfm2_moe",), "ceno": ("ceno",),
    "glm5_next": ("glm5_next_text",), "memt": ("memt_qwen3_5_moe",), "cyrah": ("cyrah",),
    "youtu_kda": ("youtu_vl_moe",), "gka": ("hybrid_qwen3",), "jet": ("jet_nemotron",),
    "gdn24": ("qwen3_5_gdn24",), "gpts3": ("gpts3_hybrid", "gptx3"),
    # M11 (host gaps): the Ling hybrids and MiniCPM-SALA
    "bailing_hybrid": ("bailing_hybrid",), "minicpm_sala": ("minicpm_sala",),
    # the release: fla DeltaNet (round 3's fla_num_heads_unplaced placed fla-hub/delta_net-*)
    "deltanet": ("delta_net",),
}
_T = "tests/test_universal_elephantine.py"
FAMILY_STATUS = {      # (status, evidence)
    "softmax-ar":     ("tested", f"whole model vs real Llama ({_T}) and the stock transformers families' own code "
                                 f"(tests/test_stock_loaders.py: Llama / Mistral / Qwen / Gemma-1..4 / Phi / Bloom / OPT / "
                                 f"Falcon / GPT-J / OLMo / Granite / DeepSeek / GPT-OSS / MoEs ..., 15 released "
                                 f"checkpoints); remaining slots priced by the currency graph"),
    "gdla":           ("tested", f"whole model vs Motif-3's own remote code (tf 5.18; its flash_attention_2 entry -> a torch kernel with flash's causal / GQA / window semantics): GDLA = latent-norm MLA q / kv, shared-RoPE key, grouped differential heads o_i - sigmoid(lambda x)_i o_noise (dl branch), elementwise output gate, YaRN + mscale^2 full layers, plain-RoPE SWA layers; PolyNorm MoE; mHC (Sinkhorn variant); fp32 ~4e-7, fp32 casts lifted ~1e-15 ({_T})"),
    "diffusion":      ("tested", f"same host, full mask ({_T}); real Nemotron-Diffusion in test_diffusion_real"),
    # round 2, pass 4: the relation puts block-causal targets (SDAR) in one class with diffusion (GEN_ORDER: the same
    # weights under a caller-supplied mask); M11: the host's attention takes a block size or a caller-supplied mask
    "block-causal":   ("tested", "block-causal / caller-supplied mask (UniversalElephantine.block_size, forward(mask=)): "
                                 "from_sdar vs stock Qwen3 / Qwen3-MoE under SDAR's own block mask (SDAR's attention is "
                                 "Qwen3's run non-causally under the caller's mask; remote code read, not run) "
                                 "(tests/test_host_gaps.py)"),
    # round 3 (nonlm_kept / nonlm_heads, design decisions): the score-head rerankers and sundial stay placed, and every
    # model whose readout is not the lm_head (the sentence embeddings, GRM's value head) has its head slot; the
    # relation's join carries them (host_point head / input branch sets) and the universal host's branches
    # (UniversalElephantine.readout, UPatchEmbed, UFlowHead) with their loaders carry them: HEAD_STATUS / INPUT_STATUS
    # below (tests/test_nonlm_host.py), checked in is_carried beside the mixer family
    "mamba1":         ("tested", f"whole model vs real Mamba ({_T}; test_mamba1_gated_delta)"),
    "gated-deltanet": ("tested", f"whole model vs real Qwen3-Next, Qwen3.5/Diba, Agnes, dQwen3.5 ({_T})"),
    "mamba2":         ("tested", f"block vs real Mamba2Mixer ({_T})"),
    "lfm2":           ("tested", f"block vs real Lfm2ShortConv ({_T})"),
    "lightning":      ("tested", f"block vs real MiniMaxLightningAttention ({_T})"),
    "rwkv4":          ("tested", f"block vs real RwkvSelfAttention ({_T}); whole model (from_rwkv: UFFN's token-shift "
                                 f"channel-mix branch) vs stock RwkvForCausalLM (tests/test_host_gaps.py)"),
    "rg-lru":         ("tested", f"block vs real RecurrentGemmaRecurrentBlock ({_T})"),
    "mamba3":         ("tested", f"block vs real Qwen3Mamba3Mixer, MIMO + trapezoid + rotation ({_T})"),
    "plamo2":         ("tested", f"whole model vs real PLaMo-2, upstream mamba_ssm/causal_conv1d refs ({_T}); M11: from_plamo2 "
                                 f"now sets the sliding window (attention_window_size) on the non-full layers, checked "
                                 f"with the window binding (T = 12, window 3) against a transcription of PLaMo-2's "
                                 f"Attention / swa_mask (tests/test_host_gaps.py)"),
    "hymba":          ("tested", f"whole model vs real Hymba (parallel, KV sharing, meta tokens, flex window) ({_T})"),
    "kda":            ("tested", f"block vs real fla KimiDeltaAttention, Triton on GPU ({_T})"),
    "gla":            ("tested", f"block vs real fla GatedLinearAttention, Triton on GPU ({_T})"),
    "retnet":         ("tested", f"block vs real fla MultiScaleRetention (RoPE + fixed decay), GPU ({_T})"),
    "rwkv5":          ("tested", f"block vs real RWKV-5 remote code, CPU path ({_T})"),
    "rwkv7":          ("tested", f"blocks vs real RWKV-7 remote code incl. v_first, fp64 exact ({_T})"),
    "nemotron_flash": ("tested", f"blocks vs its real DeltaNet (fla) and Mamba-2 (mamba_ssm Triton) on GPU; layout uses tested branches ({_T})"),
    "rwkv6":          ("tested", f"block vs real RWKV-6 remote code (ddlerp, exp decay, pre-read), fp64 exact ({_T})"),
    "granite_switch": ("tested", f"whole model vs real Granite-Switch @24aa420: per-token switched aLoRA ({_T})"),
    "phi4flash":      ("tested", f"whole model vs real Phi-4-flash: diff attn, GMU, YOCO KV, LayerNorm ({_T})"),
    "alice_ai":       ("tested", f"whole model vs real AliceAI: KDA, sigmoid router, Block AttnRes ({_T})"),
    "mini_k3":        ("tested", f"whole model vs real Mini-K3: Mamba-2, latent-norm MLA, SiTU latent MoE, AttnRes ({_T})"),
    "edgerock":       ("tested", f"whole model vs real EdgeRock: SWA + parallel score-selected memory read (writer MLP, top-capacity admission, null key, head gate) ({_T})"),
    "allgdn":         ("tested", f"whole model vs real AllGDN-Engram: all-GDN Qwen3.5 + hashed n-gram memory (UEngram) ({_T})"),
    "gigachat3_5":    ("tested", f"whole model vs real GigaChat-3.5: zero-centered gated norms, sigmoid-gated GDN, YaRN MLA, SwiGLU clip, fp64 ~1e-14 ({_T})"),
    # open, 2026-11-10: recurrent bodies whose real code has not been run against the host
    "lfm3":           ("tested", f"whole model vs real LFM3 (width-7 gated conv, per-head gate, full + SWA RoPE, sandwich norm) and LFM3-RD unrolled at its manual loop count (merged shared-core LoRA, p + c(x - p) rescale) ({_T})"),
    "lfm2_moe":       ("tested", f"whole model vs real Lfm2MoeForCausalLM (tf 5.18): gated conv + sigmoid MoE, expert-bias selection, (sum + 1e-6) renorm ({_T})"),
    "ceno":           ("tested", f"whole model vs real CENO: Mamba-2, stream-replacing norms, two-stage norms, relu^2 experts, post-top-k softmax ({_T})"),
    "glm5_next":      ("tested", f"whole model vs real Glm5Next text model (tf 5.18): mHC streams, KDA, NoPE MLA under the DSA k-pool indexer, sigmoid MoE; SuperExpert vs its own (dense, bidirectional) code; Chronos ships only a vLLM weight decoder for the same architecture ({_T})"),
    "memt":           ("tested", f"whole model vs real MemT (tf 5.18): its LM forward is Qwen3.5-MoE (3-D experts); the session memory (stable-GDN2 state written by commit events, read from frozen snapshots) is inactive without a session and not carried ({_T})"),
    "cyrah":          ("open", "Cyrah any-to-any: modeling code is in a gated repo (403), only config.json is readable; cannot be run or mapped exactly"),
    "youtu_kda":      ("tested", f"whole model vs real Youtu-LLM text model: KDA (lower-bound gate, full-rank gate) via fla 0.5.2 torch refs, interleaved-RoPE MLA + head gate, logit-domain QB routing; image tokens not carried ({_T})"),
    "gka":            ("tested", f"whole model vs real HybridQwen3 '*-GKA-GKA-*' (awslabs hybrid-model-factory @b9979c5, Triton on cuda:0, IEEE fp32 dots) ~6e-6 rel, and the GatedKalmaNet layer's chunked prefill and per-token decode kernels ~1e-5 rel: ridge readout = a key-covariance state H_t + the fixed 30-step Chebyshev solve (lambda = 0.02 ||H_t||_F) feeding the GLA query ({_T})"),
    "jet":            ("tested", f"whole model vs real Jet-Nemotron (naive dynamic conv; fla kernels -> torch refs): JetBlock dynamic conv on v (ssm_dconv), SWA, biased Qwen2 attention ({_T})"),
    "gdn24":          ("tested", "Qwen3.8-27B with all 16 full-attention layers converted: every layer is the stock Qwen3_5GatedDeltaNet + offset RMSNorm + SwiGLU (remote code read, not executed; the same composition runs through transformers' Qwen3.5 with all-linear layer_types -> from_qwen35, test_same_host_carries_qwen35_gdn24_all_linear)"),
    "gpts3":          ("tested", f"whole model vs real GPT-S3: GDN-2 (erase gate on the read key, w*v), gated GQA, Full AttnRes; fp64-lifted ~1e-14 ({_T}); GPT-X3 (the same GDN-2 with low-rank gate projections, a plain residual stream) vs a REFERENCE of modeling_gptx3.py (from_gptx3, tests/test_host_gaps.py)"),
    # M11 (host gaps)
    "deltanet":       ("tested", "block vs a REFERENCE of fla 0.4.0's DeltaNet (layers/delta_net.py, use_gate False; the "
                                 "ungated delta rule of ops/delta_rule/naive.py, silu short convs, l2 q / k, per-head "
                                 "o_norm; allow_neg_eigval too): _deltanet_from, tests/test_host_gaps.py test_deltanet_block"),
    "bailing_hybrid": ("tested", "whole model (from_bailing_hybrid): Ling-3.x vs the stock BailingHybridForCausalLM "
                                 "(transformers 5.22.2: KDA + head-gated MLA + DeepSeek-V3 MoE) and vs a REFERENCE "
                                 "transcription of BailingMoeV3 (separate-conv KDA, q-LoRA MLA, per-layer SwiGLU limits: "
                                 "UFFN.clip_post); Ling-2.5 / 2.6 vs a REFERENCE transcription of BailingMoeV2_5 "
                                 "(Lightning / simple-GLA with qk-norm + partial RoPE + GroupRMSNorm gate, MLA); remote "
                                 "code read, not run; fp64 ~1e-15 (tests/test_host_gaps.py)"),
    "minicpm_sala":   ("tested", "whole model (from_minicpm_sala) vs a REFERENCE transcription of modeling_minicpm_sala.py "
                                 "(remote code read, not run): minicpm4 softmax GQA (+ output gate) and lightning (simple GLA, "
                                 "qk-norm, RoPE, output norm + gate) layers, muP scales; the InfLLM-v2 sparse attention "
                                 "(CUDA, kv length >= dense_len 8,192) is not carried (tests/test_minicpm_sala_host.py)"),
    # M12: bidirectional embedding encoders built from decoder bodies (gte-Qwen2, stella, voyage-4-nano)
    "bidirectional":  ("tested", "full mask over the real keys (the host's causal=False + forward(mask=)): stock Qwen2Model / "
                                 "Qwen3Model under the bidirectional mask their remote code builds (gte / stella "
                                 "modeling_qwen.py is_causal=False, voyage modeling_qwen3_bidirectional.py), + the "
                                 "sentence-transformers pooling (tests/test_nonlm_host.py)"),
    "pseudobrain":    ("tested", f"whole model vs the repos' own pb27-vllm plugin (PB27Qwen3 / PB27Qwen3Moe) run under vLLM 0.31 in an isolated venv: Qwen3 (-MoE) with a sliding window on every attention layer + a parallel Qwen3.5 gated-DeltaNet on pb_hybrid_layers summed with attention (seq = 0); fp32 ~1e-6, bf16 path at its own bf16-vs-fp32 noise; sinks / Quest memory tier not served or not engaged ({_T})"),
}


def ssm_family(cfg):
    if _lm_config(cfg).get("pb_hybrid_layers") and str(_lm_config(cfg).get("model_type")) in ("qwen3", "qwen3_moe"):
        return "pseudobrain"
    mt = str(_lm_config(cfg).get("model_type", "")).lower()
    return next((f for f, types in _FAMILY_TYPES.items() if mt in types), "other:" + mt)


def ssm_state(cfg):
    """Recurrent state elements per layer (heads * dk * dv) where the config makes it explicit."""
    d = _lm_config(cfg)
    g = lambda *ks: next((d[k] for k in ks if isinstance(d.get(k), (int, float)) and d.get(k)), 0)
    f = ssm_family(cfg)
    if f == "gated-deltanet":
        return g("linear_num_value_heads") * g("linear_key_head_dim") * g("linear_value_head_dim")
    if f == "mamba1":
        return int(g("expand", "mamba_expand") * g("hidden_size")) * g("state_size", "mamba_d_state", "ssm_state_size")
    if f == "mamba2":
        heads = g("mamba_num_heads", "mamba_n_heads", "n_mamba_heads")
        hd = g("mamba_head_dim", "mamba_d_head", "mamba_headdim")
        return heads * hd * g("ssm_state_size", "mamba_d_state", "mamba_state_dim", "state_size")
    return 0


def params(h, L, ffn_total):
    """rough transformer param count: attn 4h^2 + FFN 3h*ffn_total, per layer."""
    return L * (4 * h * h + 3 * h * ffn_total)


# The capacity axes emul_matrix compares (+ the opt-in context / vocab axes).
# The join takes the per-axis maximum over EVERY model, so each axis value is a real model's value
# (times the join's currency penalty where one applies), and the joined host is >= every model on
# every compared axis at once.
JOIN_AXES = ("hidden", "eff_layers", "param_layers", "kv_cap", "q_cap", "qheads", "head_dim",
             "ffn_total", "ffn_active", "max_context", "eff_window", "vocab", "kv_latent", "q_latent", "mla_rope",
             "mla_nope", "mla_v") + RC_DIMS     # + the recurrent block's dims (round 2, pass 4: model_graph RC_DIMS)
# round 3 (nonlm_heads): the readout widths (a head of width max readout_w, the value head's hidden layer readout_ff,
# exit heads at every exit layer count: dist/r3_nonlm.md section 5)
JOIN_AXES = JOIN_AXES + ("readout_w", "readout_ff", "exits")


def join(models):
    """Per-axis supremum over `models` (every model, any mixer) -> the minimal emulator dims.

    Each axis is the max of one real model's value (record holders in d["who"]); MLA targets count
    at 2x cache (TransMLA, conservative: emul_matrix only charges it to a GQA / MQA host).
    Parallel-block targets are NOT charged 2x width: the host has a parallel-block branch (ULayer.par,
    tested against a stock Cohere model in tests/test_universal_elephantine.py), so it runs them at
    their own width; the 2x rule stays for real model-vs-model edges. The host is
    a UNIFORM stack: `layers` untied layers, each with the widest FFN bank (`ffn_total` units, the
    per-layer axis emul_matrix compares), so its whole-model FFN is layers * ffn_total. The real
    models' whole-model FFN budget (`ffn_budget`, dims_of's ffn_units: the FFN present in each
    layer, summed) is reported as the width<->depth lower bound; emul_matrix does not trade FFN
    width for depth, so that bound alone does not dominate the zoo."""
    d = {k: 0 for k in JOIN_AXES}
    d["ffn_budget"], who, unknown = 0, {}, {}
    for m in models:
        s, dm = m["slots"], m["dims"]
        pen = {"kv_cap": MLA_PENALTY if s["kv"] == "MLA" else 1}   # no 2x width: the host has a parallel branch
        for k in JOIN_AXES + ("ffn_budget",):
            v = dm["ffn_units" if k == "ffn_budget" else k]
            if v is None and k in ("max_context", "eff_window"):
                # round 3 (model_graph max_context_unknown): an unknown context is supplied only by an unbounded host
                # (model_graph._ctx_ge), so the join, a constructed host, takes CTX_UNBOUNDED there (after the loop: a
                # real unbounded model, when there is one, stays the record holder)
                unknown.setdefault(k, m["name"])
                continue
            v = v * pen.get(k, 1)
            if v > d[k]:
                d[k], who[k] = v, m["name"]
    for k, name in unknown.items():
        if d[k] < _mg.CTX_UNBOUNDED:
            d[k], who[k] = _mg.CTX_UNBOUNDED, name
    L = max(d["eff_layers"], d["param_layers"])               # untied: >= every unrolled depth
    d.update(layers=L, eff_layers=L, param_layers=L, ffn_units=L * d["ffn_total"], who=who)
    # the RMSNorm group counts the zoo binds (grouped_rmsnorm): the host's UNorm takes any of them per target
    # (UNorm.groups, from_k2_horizon), a branch set the relation's grouped_rmsnorm rule reads for this point only
    gs = sorted({1} | {int(m["dims"].get("norm_groups", 1) or 1) for m in models})
    d["norm_groups_set"] = "|".join(map(str, gs)) if len(gs) > 1 else 1
    return d


# model_graph.FEATURES the universal host carries as switchable branches (elephantine/universal.py): the
# output gate (gate_coef), switched LoRA (lora_adapters), hyper-connections (UHC), Engram (UEngram), PLE,
# AttnRes, the v-norm (v_c), q / k / v / o / FFN / router biases and the norm centre + bias, post-sublayer
# norms (post_coef; with the pre-norm the identity, UNorm.ident, the post-only OLMo-2 layout: from_olmo), the DSA
# indexer. Not carried: nanochat's value embeddings / smear (value_emb).
HOST_FEATS = {"out_gate", "switch_lora", "hyper_conn", "engram", "ple", "attn_res", "vnorm", "bias_qkv",
              "bias_out", "post_norm", "post_only", "dsa", "router_sigmoid", "router_bias", "router_group", "shared_gate", "rf_*", "nope_layers",
              "clip_qkv",      # UAttention.qkv_clip (OLMo-1 clip_qkv; from_olmo)
              "post_ln"}       # stream_norms (post-LayerNorm OPT-350m: from_opt, tests/test_stock_loaders.py)
# round 3, batch 2a (the output-gate keys, model_graph._out_gate_key): UAttention's gate is o * (c0 + c1 sigmoid(g_proj x)
# + gate_sp softplus(g_proj x)) with an elementwise g_proj, and its loaders tie the rows per head for a head-wise gate
# (from_laguna: g.repeat_interleave(hd) with gate_sp 1, tests/test_stock_loaders.py "laguna" gating per-head; the Youtu
# MLA loader for mla_output_gate_type head_wise) -- so it carries the head-wise and softplus forms by a tested loader. Not
# the biased gates (no loader writes g_proj's bias), the silu gate, nor ERNIE-5.5's q-latent gate. (release-fix: + the
# silu gate, UAttention.gate_silu, and the activated gates with g_proj's bias, per channel or per head --
# tests/test_host_gaps.py test_activated_output_gate, fp64 vs a REFERENCE of K2-Horizon's gated attention with the
# head-wise / biased W_g of Ethoric / Beam; no zoo shape binds these six keys, so no count moves.)
from graph import model_graph as _mg  # noqa: E402

HOST_FEATS = _mg._R3Set(HOST_FEATS, add={
    "out_gate_headwise": {"out_gate_head"}, "out_gate_by_code_unlisted": {"out_gate_head"},
    "out_gate_activation": {"out_gate_softplus", "out_gate_head_softplus", "out_gate_silu", "out_gate_head_silu",
                            "out_gate_bias_silu", "out_gate_head_bias_silu", "out_gate_bias_softplus",
                            "out_gate_head_bias_softplus"},
    # round 3, batch 4b: UHC is DeepSeek-V4's HyperConnection (comb^T, Sinkhorn) with a MEAN collapse at the end
    # (UniversalElephantine.forward: x = X.mean(2)) -- GLM-5-Next's variant (from_glm5_next) -- and its sk branch is
    # Motif-3's MHCLayer (from_motif), both run against the real code in tests/test_universal_elephantine.py; not V4's
    # learned HyperHead, V4.1's lagged pre, comb-orientation variants, Hy4 / Qwen4-exp / needle / SAN
    "hyper_conn_variant": {"hc_glm5_next", "hc_motif"},
    # UEngram is AllGDN's rolling hash with the gated conv (test_same_host_carries_allgdn_engram), not DeepSeek-V4.1's
    # prime-bucket XOR Engram (ngram_memory_unread)
    "ngram_memory_unread": {"engram_allgdn"},
    # round 3 (nonlm_kept, a design decision): the two score-head rerankers (score_head: a Linear(hidden, 1) head per
    # exit layer) and sundial (ts_patch: continuous value-patch input, flow-matching head) stay placed and the join
    # carries them. The universal host's branches for these (non-vocab output heads with pooling, continuous patch
    # inputs) and their loaders are tested in tests/test_nonlm_host.py (minimal_emulator HEAD_STATUS / INPUT_STATUS)
    "nonlm_kept": {"score_head", "ts_patch"},
    # round 3 (nonlm_heads): the exit heads (cfg.exits / exit_heads: exit j reads norm(x after exits[j] layers) through
    # its own head; UniversalElephantine.set_exits), carried as the feature exit_heads (no snapshot model binds it)
    "nonlm_heads": {"exit_heads"}},
    drop={"ngram_memory_unread": {"engram"}})
# the release: features whose code the universal host now runs, each against a reference transcription or the stock
# class of a model binding it (the join holds them; the code-defined variant, as model_graph.FEATURES words it):
#   DeepSeek-V4 (from_deepseek_v4 vs a fp64 transcription of the official inference code, tests/test_deepseek_v4_host.py):
#     hc_dsv4 (comb^T Sinkhorn + the learned HyperHead collapse), kv_compress (the CSA / HCA pools), hash_route
#     (tid2eid), router_sqrtsoftplus, act_clip (UFFN clip_lim: gate <= L, |up| <= L, no up shift)
#   DeepSeek-V4.1 (from_deepseek_v41, tests/test_deepseek_v41_host.py): hc_dsv41 (the lagged pre), kv_compress_v41,
#     engram (the prime-bucket XOR Engram on the hc stream)
#   moe_branch_norm: Gemma-4's two normed MoE branches (from_gemma4 vs the stock Gemma4ForCausalLM, test_stock_loaders
#     "gemma4-moe"); diff_attn: Motif-3's differential heads (from_motif, test_universal_elephantine); rd_controller:
#     LFM3-RD's step_scale at its manual loop count (from_lfm, test_same_host_carries_lfm3_rd_unrolled);
#     router_bias_logit: a logit-domain selection bias (UMoE.sel_logit, from_youtu, test_same_host_carries_youtu_kda_mla)
#   post_norm_attn: a post-norm on the attention output only (ULayer.post_coef per sub-block; test_post_norm_attention_only
#     in tests/test_host_gaps.py, a stock Llama with its attention outputs RMS-normed)
#   q_temp: Mistral-Large-4's learned per-head query temperature q (s_h log(t + 1) + b_h) (UAttention.qt_s / qt_b;
#     test_query_temperature_and_constant_qk_gain, a reference of MistralLarge4TextSoftmaxScaling)
#   out_gate_bias / out_gate_head_bias: the biased sigmoid output gate per channel / per head (UAttention.g_proj's bias),
#     q_temp_nope: Beam's code-fixed (1 + (t + 1) / 2048)^0.1 query temperature (UAttention.qt_pow);
#     test_biased_output_gate_and_power_temperature
# (keyed by the round-3 spec that reads the feature where there is one, so R3_OFF leaves them out with it)
HOST_FEATS = _mg._R3Set(HOST_FEATS, add={
    "release_host": {"kv_compress", "hash_route", "router_sqrtsoftplus", "act_clip", "engram", "moe_branch_norm"},
    "norm_placement_keys_unread": {"post_norm_attn"},
    "position_query_temperature": {"q_temp", "q_temp_nope"},
    "out_gate_by_code_unlisted": {"out_gate_bias", "out_gate_head_bias"},
    "hyper_conn_variant": {"hc_dsv4", "hc_dsv41"}, "kv_compress_variant": {"kv_compress_v41"},
    "diff_attn": {"diff_attn"}, "router_bias_unread": {"router_bias_logit"}, "lfm3rd_recurrent_controller": {"rd_controller"}})
# eleph-gaps-attn: the round-3 attention features whose code the universal host now runs (UAttention.add_branches, each
# off by default; loaders / setters in elephantine/branches_attn.py; tests/test_attn_gaps.py, fp64, each branch shown
# to bind):
#   xsa / xsa_scaled: exclusive self attention (xsa_c, xsa_eps; from_gptx2 vs REFERENCES of GPT-X2.5 / GPT-S2 whole
#     models, set_xsa vs Limite's tanh(alpha) and Grug's + eps forms); out_gate_post: MM200's gate after o_proj
#     (pg_proj; from_mm200 vs a REFERENCE whole model); dyn_mask: Doge's value-dependent key bias + top-keep keys
#     (dm_proj; from_doge vs the stock DogeForCausalLM and Doge-320M's remote per-head attention) and res_gain, its
#     per-channel residual gains (ULayer.rgain, eleph-gaps-stream's URGain); swa_trunc: Meditor's top-left truncated K / V (trunc; vs a REFERENCE
#     of Meditor2Attention); attn_out_sat: D-ROS (sat, _dros; from_qwen3_dros vs the stock Qwen3 + a REFERENCE of
#     Qwen3DROSAttention); moa: JetMoE's routed attention experts (hr_proj; from_jetmoe vs the stock JetMoe); onorm:
#     Muse-Spark's weight-free per-head output norm (o_coef / o_avg; set_onorm vs a REFERENCE of MuseSparkTextAttention);
#     vha: ERNIE-5.5's VHAPostMix (o_hmix; set_vha vs the spec's transcription)
# Not added: doc_mask (the host masks attention to the document -- set_doc_mask, test_doc_mask_attention -- but the
# feature also resets Ethoric's canon convs and KLM's recurrence / n-gram memory at the eos, which the host does not).
HOST_FEATS = _mg._R3Set(HOST_FEATS, add={
    "xsa_self_value_projection": {"xsa", "xsa_scaled"}, "mm200_core_keys_unread": {"out_gate_post"},
    "doge_dynamic_mask": {"dyn_mask"}, "residual_gain_vector": {"res_gain"}, "swa_prefill_key_truncation": {"swa_trunc"},
    "attn_output_saturation": {"attn_out_sat"}, "moa_routed_attention_heads": {"moa"},
    "attn_output_head_norm": {"onorm"}, "vha_head_mix": {"vha"}})
# eleph-gaps-stream: the residual-stream / input / FFN / norm / loop features the host now runs (elephantine/
# branches_stream.py), each against the stock class or a labelled reference of a model binding it
# (tests/test_eleph_gaps_stream.py):
#   hc_qwen4_exp: Qwen4-Exp's GatedResidual streams (hc_kind "gated"); engram_qwen4: its EOS-reset n-gram PLE on the
#     streams (UNgramPLE) -- from_qwen4_exp vs the stock transformers 5.18 Qwen4ExpForCausalLM
#   zamba_shared: the shared transformer block over concat(x, embedding) feeding the SSM input (UCatBlock) -- from_zamba2
#     vs the stock Zamba2ForCausalLM (per-invocation LoRA adapters folded into the untied copies), from_zamba vs the
#     stock ZambaForCausalLM (its multi-head Mamba-1 mixer: one host head per channel)
#   value_emb: gated per-token value-embedding tables (UValueEmb; kinds ve / ve_shared / ve_layer, not KLM's
#     v_residual), the smear (USmear) and the x0 / resid lambdas (UX0Mix) -- from_nanochat vs REFERENCES of
#     nanochat_gpt / gaggle / sn38-nanochrono, from_arloop vs a REFERENCE of ARLoop / PreNormLoop (+ ARLoop's
#     windowed Block AttnRes); backout: x - lambda x_mid before the final norm (bo_write / bo_c) -- from_nanochat
STREAM_FEATS = {}       # spec -> the FEATURES keys it carries (several entries may share a spec)


def _stream(spec, *keys):
    STREAM_FEATS.setdefault(spec, set()).update(keys)


_stream("hyper_conn_variant", "hc_qwen4_exp")
_stream("qwen4_ple_reads_ngrams", "engram_qwen4")
_stream("stream_host", "zamba_shared", "value_emb")
_stream("backout_residual", "backout")
# scmoe: LongCat's shortcut-connected MoE (ULayer.moe_defer) + identity experts (UFFN.exp_ident) -- from_longcat_flash
#   vs the stock LongcatFlashForCausalLM
_stream("no_model_type_longcat", "scmoe")
# loop_bnorm: the loop-boundary norm (a stream norm at each unrolled pass end, ULayer.sn_out); early_exit: the
#   per-token exit among the pass outputs (UEarlyExit: the gate's exit pdf, a threshold or a fixed step) -- from_ouro
#   vs a REFERENCE of Ouro's modeling_ouro.py; loop_bnorm_cfg: Nanbeige's pass norm (skip_loop_final_norm false) --
#   from_nanbeige vs a REFERENCE of modeling_nanbeige.py
_stream("loop_boundary_norm", "loop_bnorm", "loop_bnorm_cfg")
_stream("per_token_early_exit", "early_exit")
# hrm_two_state: HRM's z_H / z_L states in the register (UTwoState) -- from_hrm_text vs the stock transformers 5.18
#   HrmTextForCausalLM
_stream("hrm_hierarchical_loop", "hrm_two_state")
# hc_dsv4_nano: DeepSeek-V4 hyper-connections read comb X (UHCVariant "comb") -- from_nanowhale vs a REFERENCE of
#   nanowhale's modeling_deepseek_v4.py
_stream("hyper_conn_variant", "hc_dsv4_nano")
# hc_xing: Xing4.0's clamped, max-shifted Sinkhorn read comb X (UHCVariant "xing") -- from_xing vs a REFERENCE of
#   modeling_xing4_0.py
_stream("hyper_conn_variant", "hc_xing")
# hc_hy4: HY-V4's pre / post-only streams (UHCVariant "nocomb") -- from_hy_v4 vs the stock transformers 5.18
#   HYV4ForCausalLM
_stream("hyper_conn_variant", "hc_hy4")
# selfcond / selfcond_norm: the self-conditioning input path (forward(selfcond=), USelfCond, its always-on post-norm)
#   -- from_sdar_selfcond vs a REFERENCE of QwenDLM-ORL's modeling_sdar.py
_stream("stream_host", "selfcond", "selfcond_norm")
# ngpt / ffn_norm / loop_inject: Dory's unit-sphere streams (stream norms + URGain's nGPT update), the FFN hidden norm
#   (UFFN.s_hnorm), the loop re-injection (ULoopInject) -- from_dory vs a REFERENCE of modeling_dory.py (PR 50099);
#   its unit q / k (qk-norm "ngpt") is the per-head norm after the rotary with sqk^2 as the weight (HOST_QKNORM below)
_stream("ngpt_sphere_residual", "ngpt")
_stream("ffn_inner_norm", "ffn_norm")
_stream("loop_input_injection", "loop_inject")
# gated_norm: the low-rank sigmoid-gated pre-norm (UNorm's gate branch at any rank; the gated embedding-norm site:
#   init_model) -- block tests vs the stock transformers 5.18 AXK2GatedRMSNorm and a REFERENCE of Grug-MoE's
#   GrugMoeGatedNorm (neither whole model is carried: their raw-latent MLA query / XSA are attention features)
_stream("gated_prenorm_lowrank", "gated_norm")
# res_gain / res_bias: per-channel residual gains and a bias per sub-block (URGain) -- from_doge vs the stock
#   DogeForCausalLM (its dynamic mask neutral, A = 0), ZAYA1's ResidualScaling merges vs a REFERENCE
#   (zaya_residual_scaling_into)
_stream("residual_gain_vector", "res_gain", "res_bias")
# res_gate: a data-dependent scalar gate on each residual update (URGain gw / gc) -- CFRD's BindingBlock vs a REFERENCE
#   (cfrd_binding_into); res_gate_ms: Muse-Spark's coupled per-channel alpha / beta residual gate (URGain gx / go,
#   muse_residual_gate_into) vs a REFERENCE of MuseSparkResidualGate. Block level: CFRD's summary memory and
#   Muse-Spark's attention output norm / latent MoE are not carried
_stream("cfrd_sigmoid_residual_gate", "res_gate")
_stream("residual_gain_vector", "res_gate_ms")
# canon: Canon layers (UCanon: short causal depthwise convs + residual) at A (attention input), B (q / k / v), C (FFN
#   input), D (FFN hidden) -- from_dhara vs a REFERENCE of modeling_dhara_ar.py (Ethoric's doc-masked Canon resets at
#   EOS: not this branch; needle's taps are its JAX code)
_stream("canon_conv_layers", "canon")
# engram_crowd: Crowd-v1's hashed bigram input embedding (UBigramHash) -- from_crowd vs a REFERENCE of modeling_crowd.py
_stream("ngram_memory_unread", "engram_crowd")
# router_mlp: an MLP router W2 tanh(W1 x) (URouterMLP) -- from_llama_moe vs a REFERENCE of modeling_llama_moe_hf.py
_stream("mlp_router", "router_mlp")
# par_moe_res: Arctic's dense residual MLP on the post-attention stream beside the routed experts on the layer input
#   (ffn_split + ULayer.moe_par) -- from_arctic vs a REFERENCE of modeling_arctic.py
_stream("arctic_dense_residual_parallel_moe", "par_moe_res")
# moe_capacity: per-forward expert-capacity drops (UFFN.cap_drop, capacity_keep) -- from_hunyuan_capacity vs the stock
#   HunYuan-MoE body under a REFERENCE of HunYuan-A13B's topkgating
_stream("moe_capacity_drop", "moe_capacity")
# block_routing: per-block expert sets (UFFN.blk_route, block_route_mask) -- block test vs a REFERENCE of LLaDA2.2's
#   LLaDA2MoeSparseMoeBlock (no host loader for the LLaDA2 body)
_stream("block_routing", "block_routing")
# router_in_gate: experts reading the routing weight times the input (UFFN.in_gate) -- from_llama4 vs the stock
#   Llama4ForCausalLM (its NoPE query temperature needs UAttention.qtemp_off, floor((t + 1) / floor_scale))
_stream("stream_host", "router_in_gate")
# multimax: ERNIE-5.5's elementwise logit map (UMultimax) -- vs a REFERENCE of vLLM-ERNIE's multimax. (Its summed
#   dual-sigmoid router is UFFN.router2, block-tested too; the relation compares it as a moe_sig score, which the
#   universal host's point does not carry)
_stream("multimax_logits", "multimax")
# expert_norm: Muse-Spark's per-expert output norm on a normed latent (UFFN e_pre / e_post / e_out, UConfig latent_moe)
#   -- muse_spark_moe_into vs a REFERENCE of MuseSparkMoE (block level; its attention output norm is not carried)
_stream("expert_output_norm", "expert_norm")
# norm_groups (grouped_rmsnorm, a dims field): UNorm.groups, the RMS per group (from_k2_horizon vs a REFERENCE of
#   K2-Horizon's K2HorizonRMSNorm on the Llama body). The join's dims carry the zoo's group counts as a branch set
#   (join: norm_groups_set, host_point dims norm_groups "1|2|4"), which model_graph._r3_masks' grouped_rmsnorm rule
#   reads for this point only (g_h >= g_t and hidden_h / g_h >= hidden_t / g_t for some listed g_h)
STREAM_VE_KINDS = "ve|ve_layer|ve_shared"           # host_point ve_kind (value_emb_keys_unread's kinds it carries)
# host_point attn_res_sig: ARLoop's block partitions "<attnres_block_size>/<sw_blocks>" (from_arloop: a source every
#   attnres_block_size / 2 layers + the embedding + the last sw_blocks, ULayer.ar_sw; any size / window, tested at 4/2
#   vs a REFERENCE of modeling_arloop.py) -- the snapshot's two, listed since the relation reads a finite branch set
STREAM_AR_SIGS = "18/3|24/3"
HOST_FEATS = _mg._R3Set(HOST_FEATS, add=STREAM_FEATS)


# the qk-norm variants UAttention._qk_norm carries (model_graph.QKNORM_VARIANTS): per head or over the whole
# projection (qk_full), RMS or LayerNorm with a bias (qk_centre, q_nb / k_nb), before or after the rotary (qk_post),
# with (H, hd) gains (per-head-distinct; a shared or unit gain is a setting of them). The release: DeepSeek-V4's (a
# weight-free q norm, the weighted norm of the one K = V head; from_deepseek_v4, tests/test_deepseek_v4_host.py) and
# V4.1's (dsv4_kv: the K = V norm only; tests/test_deepseek_v41_host.py), and CENO's input renormalisation (qkv_input:
# qkv_norm(norm(x)), two norms in series; from_ceno vs the real CENO, test_universal_elephantine), and the weight-free
# per-head norm times a constant logit gain c (head_free_gain: q gain c, k gain 1; tests/test_host_gaps.py
# test_query_temperature_and_constant_qk_gain).
HOST_QKNORM = "|".join(("no", "head", "head_distinct", "head_free", "head_post", "proj", "proj_ln", "head_ln",
                        "head_ln_bias", "ln_shared_bias", "dsv4", "dsv4_kv", "qkv_input", "head_free_gain"))
# eleph-gaps-stream: nGPT's unit q / k with the per-channel sqk^2 q scale after the rotary (Dory: from_dory) -- the
# rotary is orthogonal, so it is the per-head norm after the rotary (qk_post) with per-head-distinct weights
HOST_QKNORM = HOST_QKNORM + "|ngpt"


# round 3 (nonlm_heads): the readout kinds and inputs the universal host carries as branches (dist/r3_nonlm.md:
# every head kind, the token table beside UPatchEmbed; HEAD_STATUS / INPUT_STATUS, tests/test_nonlm_host.py)
HOST_HEAD = "lm|score|embedding|value|flow"
HOST_INPUT = "tokens|patches"


def host_point(j, gen="ar"):
    """The joined host as an emul_matrix point, the join's dims. Where the host carries several
    variants natively (elephantine/universal.py), its slot is a BRANCH SET 'a|b' (model_graph.branches):
    norm RMSNorm|LayerNorm (UNorm centre), block sequential|parallel (ULayer.par), qk-norm HOST_QKNORM,
    softcap no|final|attn (UAttention.cap, UniversalElephantine.final_cap),
    cross-layer KV no|yes (the KV register), mixer softmax|hybrid|ssm|linear (attention + the gated-delta
    block; every layer has attention, so a plain target uses all L), act = the TESTED UFFN branches
    (ACT_STATUS), position pos_top (settable RoPE inv_freq + a learned-abs table). gen is the one
    slot it sets per target (causal or full mask)."""
    dims = {k: j[k] for k in JOIN_AXES}
    dims["norm_groups"] = j.get("norm_groups_set", 1)        # (a branch set: model_graph grouped_rmsnorm)
    L = j["layers"]
    dims.update(layers=L, ffn_units=j["ffn_units"], heads=j["qheads"], inter=j["ffn_total"],
                experts=0, attn_eff=L, attn_param=L, full_eff=L, full_param=L,
                # every layer: own-KV full attention (masks to local; copies k_proj for K = V), the KV register
                # (a KV-sharing layer) and the gated-delta block (a recurrent layer)
                kFU_eff=L, kFU_param=L, kS_eff=L, kS_param=L, kR_eff=L, kR_param=L,
                kv_cap_u=j["kv_cap"], q_cap_u=j["q_cap"], qheads_u=j["qheads"], head_dim_u=j["head_dim"],
                mla_qswitch=1,                 # the q latent norm is a branch (UAttention attn_norm_feats)
                mla_lat_free=1,                # ... a block-averaged RMS over the target's latent, its eps (n_avg, n_eps)
                kv_heads=j["qheads"], q_per_kv=1,   # MHA: one KV head per query head
                # the gated-delta block at the join's recurrent dims: its RMS grouping is a weight (USSM.avg), its
                # norm eps a buffer and its q / k scale a coefficient, so it plays every family's shape that fits
                # (model_graph._rec_ok skips the per-family rules for it)
                rc_free=1, rc_rule=0,
                # round 2, pass 5: q / FFN biases are branches (UAttention / UFFN), its norms switch off by weights
                q_bias=1, ffn_bias=1, eps_lever=1,
                # round 3, batch 3b: the FFN up-projection bias is that UFFN branch too (bias_up: a vanilla target on
                # a gated / MoE host, ReLU <- swigluoai)
                bias_up=1,
                # round 3, batch 1: the embedding norm is a switchable UNorm branch (emb_norm_c; off, LayerNorm or
                # weighted RMS: elephantine/universal.py from_bloom), and the loader folds OPT's project_in /
                # project_out into its own untied tables (from_opt; real opt-350m in tests/test_stock_loaders.py);
                # the release: + the low-rank gated embedding norm (Grug-MoE's embed_gated_norm: emb_norm built gated
                # when norm_gate_rank > 0, tests/test_eleph_gaps_stream.py test_gated_norm_blocks_grug_reference)
                emb_norm="|ln|rms|rms_gated", emb_fold=1,
                # round 3, batch 4b: Full AttnRes over every sublayer (attnres_full: from_gpts3) and Block AttnRes on the
                # key-path families (attnres_block: from_alice / from_mini_k3 / from_kimi_linear); UEngram's orders /
                # heads / slots / dim / layers are set from the target's config ("*"); no value embeddings, no norm shift
                attn_res_sig="1/0|-1/-1|" + STREAM_AR_SIGS, engram_sig="*", ve_kind=STREAM_VE_KINDS, bias_shift=0,
                # round 3 (nonlm_heads; dist/r3_nonlm.md section 2): UniversalElephantine.readout
                # out = l2?(cap(head(act?(head_in(pool(norm(h))))))) -- any pooling (none / last / first / mean) and L2
                # ("*"), a biased head of any width, exits; UFlowHead / UPatchEmbed set from the target's config ("*")
                readout_sig="*", readout_bias=1, flow_sig="*", patch_sig="*")
    acts = "|".join(sorted(ACT_SLOT[a] for a, (st, _) in ACT_STATUS.items() if st == "tested"))
    # sink: the scalar per-head null logit (UAttention.null_logit) and the release's learned sink KEY (IQuest-Q1:
    # UAttention.null_key, _iquest_attn, tests/test_host_gaps.py test_iquest_q1_sink_key)
    slots = {"norm": "RMSNorm|LayerNorm", "pos": "pos_top", "kv": "MHA|MLA", "ffn": "moe_shared", "act": acts,
             "block": "sequential|parallel", "window": "full", "qknorm": HOST_QKNORM, "softcap": "no|final|attn",
             "mixer": "softmax|hybrid|ssm|linear", "sink": "no|yes|key", "crosslayer_kv": "no|yes", "mtp": "no", "gen": gen,
             "loop": "untied", "head": HOST_HEAD, "input": HOST_INPUT, "feats": "*" + "+".join(sorted(HOST_FEATS))}
    return {"slots": slots, "dims": dims, "members": [f"elephantine-join ({gen})"]}


def size(dm, ffn_units=None):
    """Total / active parameter proxy, the same accounting for the host and for real models:
    attention h * (2 q_cap + 2 kv_cap) per layer (Q, O and K, V; 4h^2 for plain MHA) over the
    unrolled depth, FFN 3h per unit over the units actually present (whole-model `ffn_units`; active
    = ffn_active per layer). Embeddings and SSM weights are left out (small next to the FFN)."""
    h, L = dm["hidden"], dm["eff_layers"]
    attn = L * h * (2 * dm["q_cap"] + 2 * dm["kv_cap"])
    units = dm["ffn_units"] if ffn_units is None else ffn_units
    return attn + 3 * h * units, attn + 3 * h * L * dm["ffn_active"]


_TU = "tests/test_universal_elephantine.py"
# Activations the host must implement NATIVELY (the FFN activation is applied inside UFFN, a weight
# setting cannot add a nonlinearity the module does not have). Keyed by act_name(); status:
#   tested -- a UFFN branch, run against real code;
#   priced -- no branch, but a currency edge in ACT_ORDER to a tested branch (approx / limit / fold);
#   open   -- no branch and no reduction (ACT_ORDER: incomparable).  Flip to tested when a branch lands.
ACT_STATUS = {
    "silu":      ("tested", f"UFFN silu(g) * u: whole model vs real Llama ({_TU})"),
    "situ":      ("tested", f"UFFN c_situ SiTU-GLU: whole model vs real Mini-K3 ({_TU})"),
    "relu2":     ("tested", f"UFFN c_relu2 ungated relu(u)^2: whole model vs real CENO ({_TU})"),
    "gelu":      ("tested", f"UFFN c_gelu ungated gelu(u) / c_geglu gated gelu(g) * u, erf or tanh-approx "
                            f"(gelu_tanh; the loader reads hidden_act / activation_function): whole model vs "
                            f"real GPT-2 (gelu_new, NoPE + learned-abs), Pythia-70m (erf, partial RoPE, parallel "
                            f"block) and Gemma-3-270m (GeGLU-tanh, sliding + full layers, sandwich norms), "
                            f"fp32 casts lifted ~1e-14 rel; small stock Gemma (1) / GPT-NeoX sequential too ({_TU})"),
    "gelu_tanh": ("tested", f"the same UFFN c_gelu / c_geglu branch in its tanh-approximate form (gelu_tanh): real "
                            f"GPT-2 (gelu_new) and Gemma-3-270m (GeGLU-tanh) above ({_TU})"),
    "relu":      ("tested", "UFFN c_relu ungated relu(u) (the zoo's ReLU models are OPT's ungated fc1 / fc2): whole model "
                            "vs stock OPTForCausalLM pre- and post-LayerNorm and real facebook/opt-125m / opt-350m, "
                            "fp32 casts lifted ~1e-15 (tests/test_stock_loaders.py)"),
    "mish":      ("tested", f"UFFN c_mish gated mish(g) * u: whole model vs a stock LlamaForCausalLM with "
                            f"hidden_act='mish' (the DanTagGen architecture), fp32 casts lifted ~1e-15 ({_TU})"),
    "swigluoai": ("tested", "UFFN composition: alpha = 1.702 folded into the gate rows (silu(alpha g) / alpha), the "
                            "hard clip with separate gate / up limits (clip_lim), the (u + 1) up_shift after the clip, "
                            "1 / alpha into w_down; whole model vs stock GptOssForCausalLM with the clip biting, fp32 "
                            "casts lifted ~5e-16 (tests/test_stock_loaders.py)"),
    "xielu":     ("tested", f"UFFN c_xielu ungated xIELU(u) (per-layer learned alpha_p / alpha_n, beta, eps): "
                            f"whole model vs real Apertus (default / llama3 / linear RoPE, qk-norm) ~2e-7, "
                            f"fp32 casts lifted ~1e-15 ({_TU})"),
    "poly_norm": ("tested", f"UFFN c_poly PolyNorm-GLU (per-expert weights, bias / hidden / result clamps, "
                            f"output scale): block vs Motif-3's own MotifMLP and MoE (sigmoid router + "
                            f"shared expert) ~1e-7 rel, fp32 casts lifted ~1e-15, and inside the whole Motif-3 "
                            f"model (FAMILY_STATUS gdla) ({_TU})"),
}
# act_name() -> the classify act-slot value (for the host's act branch set in host_point)
ACT_SLOT = {"silu": "SiLU", "gelu": "GeLU", "gelu_tanh": "GeLU_tanh", "relu": "ReLU", "relu2": "relu2", "situ": "situ",
            "xielu": "xielu",
            "poly_norm": "poly_norm", "mish": "mish", "swigluoai": "swigluoai"}


def act_status(a):
    """ACT_STATUS[a], with a 'priced' row checked against the CURRENT activation relation: it holds
    only if some tested host branch reaches the activation in REACH["act"] (one hop, the
    ACT_FLOOR_MAX threshold); otherwise the activation is open."""
    from graph import model_graph as mg
    st, how = ACT_STATUS.get(a, ("open", "no host branch, not in ACT_ORDER"))
    if st != "priced":
        return st, how
    tested = {ACT_SLOT[b] for b, (s, _) in ACT_STATUS.items() if s == "tested"}
    if tested & set(mg.REACH["act"].get(ACT_SLOT.get(a, a), ())):
        return st, how
    return "open", (f"no host branch and no tested branch within ACT_FLOOR_MAX = {mg.ACT_FLOOR_MAX} "
                    f"(was: {how})")


NORM_STATUS = {
    "RMSNorm":   ("tested", f"UNorm (weight, eps): whole model vs real Llama ({_TU})"),
    "LayerNorm": ("tested", f"UNorm centre = 1 + bias: whole model vs real Phi-4-flash ({_TU})"),
}
_RELU2 = ("relu2", "squared_relu", "sqrelu", "relu_squared")


def act_name(cfg, slots):
    """The FFN activation the host must run: classify's act, with squared ReLU split from ReLU."""
    a = slots["act"]
    if a == "ReLU":
        d = _lm_config(cfg)
        return "relu2" if raw_act(d) in _RELU2 else "relu"
    return a.lower()


def mixer_family(m):
    if m["slots"]["mixer"] != "softmax":
        return ssm_family(m["cfg"])
    ro = readout_kind(m)[1]
    if ro is not None and ro.get("causal") is False:                # an embedding encoder run bidirectionally
        return "bidirectional"
    if _lm_config(m["cfg"]).get("attention_cls") == "gdla":      # Motif-3: differential latent heads
        return "gdla"
    gen = m["slots"]["gen"]
    # block-causal denoising (SDAR): the host's block_size / caller-supplied mask
    return {"diffusion": "diffusion", "block": "block-causal"}.get(gen, "softmax-ar")


# M12: the readout (what the model outputs) and the input (what it reads) -- an LM is ("lm", tokens); the
# non-next-token models of the snapshot (elephantine.loader_coverage.READOUTS / readout_of) have host branches
_TN = "tests/test_nonlm_host.py"
HEAD_STATUS = {
    "lm":        ("tested", f"lm_head over the vocabulary (+ bias, final softcap) ({_TU})"),
    "score":     ("tested", f"readout: a narrow linear head (head_out) per early exit (set_exits: norm(x after e layers)): "
                            f"BGE layer-wise MiniCPM (REFERENCE of modeling_minicpm_reranker.py, fp64 ~1e-16) and cost-wise "
                            f"Gemma-2 (stock body + gemma_model.py's exits / CostWiseHead / softcap) ({_TN})"),
    "value":     ("tested", f"readout: pooled last real token -> hidden layer (head_in, relu) -> Linear(n, 1): GRM ValueHead "
                            f"on the stock Gemma body ({_TN})"),
    "embedding": ("tested", f"readout: pooling (last real token / mean / CLS) -> Dense or identity -> L2 normalise: "
                            f"sentence-transformers Pooling / Dense / Normalize on stock Qwen2 / Qwen3 bodies, causal and "
                            f"bidirectional; voyage-4-nano's per-token linear ({_TN})"),
    "flow":      ("tested", f"readout: the last patch's hidden state (identity head) -> UFlowHead, Sundial's flow-matching "
                            f"velocity network (REFERENCE of flow_loss.py, fp64 ~1e-16); the Euler sampler loop is runtime "
                            f"(flow_sample runs it for a given noise) ({_TN})"),
}
INPUT_STATUS = {
    "tokens":    ("tested", f"the token table ({_TU})"),
    "patches":   ("tested", f"UPatchEmbed: left-padded patches + mask channel -> MLP + linear residual, RevIN: Sundial "
                            f"(REFERENCE of modeling_sundial.py, fp64 ~1e-16) ({_TN})"),
}


def readout_kind(m):
    """(head kind, the readout entry or None, input kind) of a snapshot model."""
    from elephantine.loader_coverage import readout_of
    ro = readout_of(_lm_config(m["cfg"]), m.get("name")) or readout_of(m["cfg"], m.get("name"))
    if ro is None:
        return "lm", None, "tokens"
    return ro["kind"], ro, ro.get("input", "tokens")


def is_carried(m):
    """Carried iff its mixer family is tested, its activation is not open, its norm has a host branch, and
    its readout (lm / score / value / embedding / flow) and input (tokens / patches) have tested host branches
    (the headline accounting: the other slots are priced by the currency graph)."""
    st = lambda tab, k: tab.get(k, ("open",))[0]
    head, _, inp = readout_kind(m)
    return (st(FAMILY_STATUS, mixer_family(m)) == "tested" and act_status(act_name(m["cfg"], m["slots"]))[0] != "open"
            and st(NORM_STATUS, m["slots"]["norm"]) == "tested"
            and st(HEAD_STATUS, head) == "tested" and st(INPUT_STATUS, inp) == "tested")


def coverage(models):
    """Per model: carried (is_carried). Returns (carried, native) where native also requires the
    activation to be a tested branch (not only priced)."""
    carried = native = 0
    for m in models:
        ok = is_carried(m)
        carried += ok
        native += ok and act_status(act_name(m["cfg"], m["slots"]))[0] == "tested"
    return carried, native


def loaded_split(models, include_remote=False):
    """Of the carried models, those that are LOADED AND TESTED: a whole-model from_* loader accepts the
    model's model_type and config and a test runs that loader against that type's own code
    (elephantine/loader_coverage.py); the rest are carried by priced slot constructions. Returns
    (carried, loaded_and_tested). By default a test that runs remote code (tests/conftest.py
    REMOTE_CODE_TESTS) does not count; include_remote=True credits those too. Does not change the
    headline relation."""
    from elephantine.loader_coverage import classify_model
    carried = loaded = 0
    for m in models:
        if is_carried(m):
            carried += 1
            loaded += classify_model(m["cfg"], m.get("name"), include_remote)[1]
    return carried, loaded


def load(path="graph/hub_10k.json", cache=None, top=None):
    items = list(iter_json_configs(path))
    if cache:
        items += list(iter_local_configs(cache))
    models = []
    for name, cfg in items:
        if _junky(name):
            continue
        c = classify(cfg)
        if c is None:
            continue
        m = {"name": name, "slots": c, "dims": dims_of(cfg), "mamba": is_mamba_family(cfg), "cfg": cfg}
        if m["dims"]["hidden"] == 0 or m["dims"]["layers"] == 0:
            continue
        if top and len(models) >= top:
            break
        models.append(m)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", default="graph/hub_10k.json")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--top", type=int, default=None,
                    help="only the N most-downloaded decoder LMs (the snapshot is in download order)")
    ap.add_argument("--include-remote", action="store_true",
                    help="also credit tests that run remote code (tests/conftest.py REMOTE_CODE_TESTS)")
    args = ap.parse_args()
    models = load(args.from_json, args.cache, args.top)
    gens, mixers = {}, {}
    for m in models:
        c = m["slots"]
        gens[c["gen"]] = gens.get(c["gen"], 0) + 1
        mixers[c["mixer"]] = mixers.get(c["mixer"], 0) + 1

    n = len(models)
    print(f"{n} models: gen={gens}  mixer={mixers}")
    # ONE host (elephantine/universal.py): full attention (causal mask for AR, full mask for
    # diffusion) + the general gated-delta SSM block + MoE. Every model is a weight setting --
    # if the host also has its activation (UFFN) and norm (UNorm) branch.
    fam, acts = {}, {}
    for m in models:
        f = mixer_family(m)
        fam[f] = fam.get(f, 0) + 1
        if FAMILY_STATUS.get(f, ("open",))[0] == "tested":
            a = act_name(m["cfg"], m["slots"])
            acts[a] = acts.get(a, 0) + 1
    fam_ok = sum(v for f, v in fam.items() if FAMILY_STATUS.get(f, ("open",))[0] == "tested")
    carried, native = coverage(models)
    print(f"mixer family tested: {fam_ok} of {n}")
    print(f"carried by the one universal host: {carried} of {n} ({100 * carried / n:.1f}%)  "
          f"[mixer tested + activation tested or priced + norm branch]")
    print(f"  natively (activation is a tested host branch, not only priced): {native} of {n}")
    _, loaded = loaded_split(models, include_remote=args.include_remote)
    print(f"  loaded and tested (a from_* loader runs the model's own code for its model_type): {loaded} of "
          f"{carried}; the other {carried - loaded} carried by priced slot constructions\n")
    print("  family            models  status")
    for f, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        st, how = FAMILY_STATUS.get(f, ("open", "unclassified recurrence"))
        print(f"  {f:16s} {v:7d}  {st:12s} {how}")
    heads = {}
    for m in models:
        k = readout_kind(m)
        heads[(k[0], k[2])] = heads.get((k[0], k[2]), 0) + 1
    print("\n  readout / input     models  status")
    for (h, i), v in sorted(heads.items(), key=lambda kv: -kv[1]):
        print(f"  {h + ' / ' + i:18s} {v:7d}  {HEAD_STATUS.get(h, ('open',))[0]} / {INPUT_STATUS.get(i, ('open',))[0]}")
    print("\n  activation (models in tested mixer families)")
    for a, v in sorted(acts.items(), key=lambda kv: -kv[1]):
        st, how = act_status(a)
        print(f"  {a:16s} {v:7d}  {st:12s} {how}")

    # the join over EVERY model it must dominate (all mixers, AR and diffusion): per-axis maxima
    j = join(models)
    pos_vals = {m["slots"]["pos"] for m in models}
    abs_pe = {"learned_abs"} & pos_vals
    rel_pe = {"rope", "partial_rope", "alibi"} & pos_vals
    pos = "learned_abs + rope (both)" if (abs_pe and rel_pe) else next(iter(pos_vals))
    w = lambda k: j["who"][k].split("/")[-1]

    print(f"\n=== the minimal universal emulator (one host for all {n}) ===")
    hp = host_point(j)["slots"]
    print(f"  categorical: {hp['norm']} | {pos} | MHA | moe_shared | {hp['act']} | {hp['block']} | "
          f"full (causal or full mask) | softmax + gated-delta SSM (hybrid) | untied")
    print(f"  hidden      h = {j['hidden']:>10}   ({w('hidden')})")
    print(f"  layers      L = {j['layers']:>10}   (untied; >= every effective depth: {w('eff_layers')})")
    print(f"  kv/token   kv = {j['kv_cap']:>10}   (MHA; 2x for MLA targets: {w('kv_cap')})")
    print(f"  q/token     q = {j['q_cap']:>10}   ({w('q_cap')})")
    print(f"  query heads   = {j['qheads']:>10}   ({w('qheads')})")
    print(f"  head dim      = {j['head_dim']:>10}   ({w('head_dim')})")
    print(f"  ffn/layer     = {j['ffn_total']:>10}   (widest FFN bank, every layer: {w('ffn_total')})")
    print(f"  ffn units     = {j['ffn_units']:>10}   (L x ffn/layer; largest real budget "
          f"{j['ffn_budget']}: {w('ffn_budget')})")
    print(f"  ffn active    = {j['ffn_active']:>10}   ({w('ffn_active')})")
    cx = lambda v: "unbounded" if v == _mg.CTX_UNBOUNDED else ("unknown" if v is None else v)   # (round 3)
    print(f"  context       = {cx(j['max_context']):>10}   (window {cx(j['eff_window'])}, vocab {j['vocab']}; "
          f"opt-in axes)")
    sized = [(ssm_state(m["cfg"]), m["name"]) for m in models if m["slots"]["mixer"] != "softmax"]
    sized = [x for x in sized if x[0]]
    if sized:
        st, who = max(sized)
        print(f"  ssm state     = {st:>10}   per layer (widest recurrent state: {who.split('/')[-1]}; "
              f"{len(sized)} recurrent models sized)")
    P, A = size(j)
    Pb, _ = size(j, ffn_units=j["ffn_budget"])
    print(f"\n  => ~{P/1e12:.1f}T params total, ~{A/1e12:.2f}T active "
          f"(width<->depth lower bound, FFN = the largest real budget: ~{Pb/1e12:.1f}T)")
    big = max(models, key=lambda m: size(m["dims"])[0])
    bp = size(big["dims"])[0]
    print(f"  vs biggest real model: {big['name'].split('/')[-1]} ~{bp/1e12:.1f}T "
          f"-> the join is ~{P/bp:.1f}x larger than anything that exists")


if __name__ == "__main__":
    main()
