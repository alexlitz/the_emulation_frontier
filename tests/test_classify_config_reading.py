"""How configs are READ (round 2, pass 1: classification fixes from the literature reviews), on the real
snapshot configs (graph/hub_10k.json): config-class defaults, renamed types, drafters, the dropped legacy
families, mislabels, NoPE detection, layer parsing and the generation labels.

Run:  python -m pytest tests/test_classify_config_reading.py -q
"""
import functools

import pytest

from graph.model_graph import (ZOO_CONFIGS, _ffn_widths_fit, classify, dims_of, emulates, iter_json_configs,
                               skipped)


@functools.lru_cache(maxsize=None)
def _cfgs():
    return dict(iter_json_configs(ZOO_CONFIGS))


def C(name):
    c = _cfgs().get(name)
    if c is None:
        pytest.skip(f"{name} not in the snapshot")
    return c


def S(name):
    return classify(C(name))


def F(name):
    return set(filter(None, S(name)["feats"].split("+")))


def test_class_defaults_not_hand_lists():
    # tie_word_embeddings absent = the config class default (MambaConfig / CohereConfig / FalconConfig: True)
    assert "tied_emb" in F("state-spaces/mamba-130m-hf")
    assert "tied_emb" in F("tiiuae/falcon-7b")
    # GPTNeoXConfig use_parallel_residual default True (gpt-neox-20b and the pythia v0 configs omit it)
    assert S("EleutherAI/gpt-neox-20b")["block"] == "parallel"
    # PersimmonConfig / GlmConfig partial_rotary_factor 0.5 when the key is absent; ChatGLM 0.5 by code
    assert S("zai-org/glm-4-9b")["pos"] == "partial_rope"
    for name in ("adept/persimmon-8b-base", "adept/persimmon-8b-chat"):
        if name in _cfgs():
            assert S(name)["pos"] == "partial_rope"


def test_drafters_and_assistants_are_not_standalone_lms():
    for name in ("z-lab/Qwen3-8B-DFlash-b16", "google/gemma-4-E2B-it-assistant", "nvidia/Kimi-K2.6-Eagle3",
                 "openbmb/MiniCPM5-2B-DSpark"):
        assert skipped(C(name)) == "drafter" and classify(C(name)) is None, name
    # a full model with an ATTACHED draft head is still a model
    assert classify(C("deepseek-ai/DeepSeek-V4-Flash-0731")) is not None
    # SKIP_TYPES tests the OUTER model type too (gemma3n wraps a gemma3n_text)
    assert skipped(C("mlx-community/gemma-3n-E2B-it-lm-4bit")) == "non_decoder"


def test_dropped_legacy_families_are_read():
    neo = S("EleutherAI/gpt-neo-125m")                    # num_heads; global / local layers, window 256
    assert (neo["pos"], neo["window"], neo["act"]) == ("learned_abs", "local", "GeLU_tanh")
    d = dims_of(C("EleutherAI/gpt-neo-125m"))
    assert (d["kFU_eff"], d["kLU_eff"]) == (6, 6)
    f7 = S("tiiuae/falcon-7b")                             # RoPE by code, MQA, parallel
    assert (f7["pos"], f7["kv"], f7["block"]) == ("rope", "MQA", "parallel")
    assert S("anas-awadalla/mpt-7b")["pos"] == "alibi"    # n_heads, attn_config.alibi
    assert S("facebook/xglm-564M")["pos"] == "sinusoidal"  # a fixed table, not learned_abs
    assert S("microsoft/biogpt")["pos"] == "learned_abs"
    assert S("GSAI-ML/LLaDA-8B-Base")["gen"] == "diffusion"   # OLMo-style keys
    assert S("internlm/internlm-7b")["pos"] == "rope" and "bias_out" in F("internlm/internlm-7b")
    elm = dims_of(C("apple/OpenELM-270M"))                  # per-layer heads / FFN widths
    assert elm["q_cap"] == 20 * 64 and len(elm["ffn_widths"]) == 16
    assert "post_ln" in F("openai-community/openai-gpt")


def test_mislabels():
    for name in ("baichuan-inc/Baichuan2-13B-Chat", "baichuan-inc/Baichuan-13B-Chat"):
        assert S(name)["pos"] == "alibi", name                  # 13B Baichuans run ALiBi
    assert S("baichuan-inc/Baichuan2-7B-Chat")["pos"] == "rope"
    assert S("tiiuae/falcon-rw-1b")["block"] == "sequential"  # parallel_attn false
    assert S("tiiuae/falcon-11B")["kv"] == "GQA"             # new_decoder_architecture: num_kv_heads 8
    assert S("bigcode/santacoder")["kv"] == "MQA"            # attention_head_type multiquery
    assert S("swiss-ai/Apertus-8B-2509")["ffn"] == "vanilla"
    assert S("microsoft/Phi-3.5-MoE-instruct")["norm"] == "LayerNorm"
    assert S("microsoft/Phi-3.5-mini-instruct")["window"] == "full"   # window 262144 >= max_position 131072
    assert S("openai-community/gpt2")["act"] == "GeLU_tanh"           # gelu_new
    assert S("EleutherAI/pythia-160m")["act"] == "GeLU"               # gelu (erf)
    assert S("google/gemma-7b")["act"] == "GeLU"                      # GemmaMLP reads hidden_act
    assert "bias_out" in F("paloalma/TW3-JRGL-v2")                    # attention_bias also biases o_proj
    assert S("CladeTeam/CENO-1.3B-A450M-base")["ffn"] == "moe"         # n_shared_experts 0
    assert dims_of(C("LiquidAI/LFM2-350M"))["ffn_total"] == 4608       # block_auto_adjust_ff_dim
    assert dims_of(C("huihui-ai/Huihui-MoE-0.8B-2E"))["moe_unit_top1"] == 1
    assert {"router_sigmoid", "router_bias"} <= F("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    assert S("XiaomiMiMo/MiMo-V2-Flash")["sink"] == "yes"


def test_deepseek_v4():
    name = "deepseek-ai/DeepSeek-V4-Flash"
    s, d = S(name), dims_of(C(name))
    assert (s["pos"], s["qknorm"], s["sink"]) == ("partial_rope", "dsv4", "yes")   # (round 2 pass 2: the qk-norm variant)
    assert {"vnorm", "hash_route", "act_clip", "router_sqrtsoftplus"} <= F(name)
    assert d["kv_cap_u"] == 0 and d["kFU_eff"] + d["kLU_eff"] == 0     # K = V on every layer


def test_nope_and_layer_parsing():
    for name in ("nvidia/Nemotron-H-8B-Base-8K", "ibm-granite/granite-4.0-h-tiny", "allenai/Olmo-Hybrid-7B",
                 "moonshotai/Kimi-Linear-48B-A3B-Instruct"):
        assert S(name)["pos"] == "nope", name
    assert S("ibm-ai-platform/Bamba-9B-v1")["pos"] == "partial_rope"
    assert "nope_layers" in F("arcee-ai/Trinity-Mini")                 # RoPE only on the sliding layers
    for name in ("inclusionAI/Ling-2.6-flash", "cyankiwi/MiniCPM-SALA-AWQ-8bit"):
        s, d = S(name), dims_of(C(name))
        assert s["mixer"] == "hybrid" and d["kR_eff"] > d["kFU_eff"] > 0, name


def test_generation_labels():
    assert S("Dream-org/Dream-v0-Base-7B")["gen"] == "diffusion"       # model_type "Dream"
    assert "bias_qkv" in F("Dream-org/Dream-v0-Base-7B")
    assert S("Asilarkness/DiffuRefill-1B")["gen"] == "diffusion"       # attention_mask "bidirectional"
    assert S("JetLM/SDAR-8B-Chat")["gen"] == "block" and S("JetLM/SDAR-8B-Chat")["qknorm"] == "head"


def test_new_values_are_walls_for_now():
    from tests.test_graph_rules import mk
    # round 2, pass 5: erf and tanh GeLU are measured payable both ways (graph/gelu_variants.py): approximate floors
    assert emulates(mk(act="GeLU"), mk(act="GeLU_tanh")) and emulates(mk(act="GeLU_tanh"), mk(act="GeLU"))
    assert not emulates(mk(act="GeLU"), mk(act="GeLU_tanh"), mode="exact")
    assert emulates(mk(act="GeLU_tanh"), mk(act="ReLU"))               # the beta -> inf limit, as for erf GeLU
    # gen "block" was a wall both ways in pass 1; pass 4 (2026-11-13): SDAR's caller-mask forward puts it in one class
    # with diffusion (GEN_ORDER), AR below both; an AR host still plays neither (emulates(host, target))
    assert emulates(mk(gen="block"), mk(gen="ar")) and emulates(mk(gen="diffusion"), mk(gen="block"))
    assert emulates(mk(gen="block"), mk(gen="diffusion")) and not emulates(mk(gen="ar"), mk(gen="block"))
    # fr-positions (fp_sinusoid_table): a learned-abs host holds a fixed sinusoid table; a sinusoidal host loads no other
    assert emulates(mk(pos="learned_abs"), mk(pos="sinusoidal"))
    assert not emulates(mk(pos="sinusoidal"), mk(pos="learned_abs"))


def test_ffn_widths_fit_is_a_sorted_domination():
    assert _ffn_widths_fit((8, 4), (8, 4)) and _ffn_widths_fit((8, 4), (9, 4, 1))
    assert not _ffn_widths_fit((8, 8), (9, 4))                         # the widest layer is not every layer
    assert _ffn_widths_fit([4, 2], (5, 4, 1)) and not _ffn_widths_fit([4, 3], (5, 4, 1))   # uniform target
    assert _ffn_widths_fit((8, 4), [8, 2]) and not _ffn_widths_fit((8, 4, 4), [8, 2])      # uniform host


def test_moe_routing_facts():
    """Round 2, pass 3: the routing facts the MoE rules read (dims moe_sig, graph.model_graph.MOE_SIG), and the
    dense / MoE layer kinds from the keys the code reads (ERNIE, Step, MiMo-V2, MiniMax-M3, Flint were unread)."""
    from graph.model_graph import MOE_SIG
    sig = lambda name: dict(zip(MOE_SIG, dims_of(C(name))["moe_sig"]))
    for name, (n_moe, n_dense) in (("baidu/ERNIE-4.5-21B-A3B-PT", (27, 1)),          # moe_layer_start_index 1
                                   ("stepfun-ai/step3", (56, 5)),                    # moe_layers_enum 4..59
                                   ("XiaomiMiMo/MiMo-V2.5", (47, 1)),                # moe_layer_freq list [0, 1, ...]
                                   ("nvidia/MiniMax-M3-NVFP4", (57, 3)),             # moe_layer_freq list [0, 0, 0, 1, ...]
                                   ("mlx-community/flint-mixed-3-4bit", (47, 2)),    # mlp_only_layers [0, 1]
                                   ("ai21labs/Jamba-v0.1", (16, 16)),                # expert_layer_period 2, offset 1
                                   ("deepseek-ai/DeepSeek-R1", (58, 3))):            # first_k_dense_replace 3
        s = sig(name)
        assert (s["n_moe"], s["n_dense"]) == (n_moe, n_dense), name
    v2, r1 = sig("deepseek-ai/DeepSeek-V2-Lite"), sig("deepseek-ai/DeepSeek-R1")
    assert (v2["renorm"], v2["renorm_cap"], v2["score"]) == ("P", ("P",), "softmax")   # the HF port ignores the key
    assert (r1["renorm"], r1["score"], r1["n_group"], r1["topk_group"], r1["group_cap"]) == ("R", "sigmoid", 8, 4, "top2sum")
    assert sig("nvidia/Qwen3.6-35B-A3B-NVFP4")["renorm_cap"] == ("R",)            # Qwen3.5-MoE renormalises in code
    assert sig("Qwen/Qwen3-Coder-30B-A3B-Instruct")["renorm_cap"] == ("P", "R")    # norm_topk_prob read
    # ACT2FN[scoring_func] (round 3, score_not_free: the router score is a feature switch, so a host plays only its
    # own score; the settable set the code reads comes back with the key off)
    assert sig("deepseek-ai/DeepSeek-V4-Flash")["score_cap"] == ("sqrtsoftplus",)
    mg_ = __import__("graph.model_graph", fromlist=["R3_OFF"])
    mg_.R3_OFF.add("score_not_free")
    try:
        assert sig("deepseek-ai/DeepSeek-V4-Flash")["score_cap"] == ("sigmoid", "sqrtsoftplus")
    finally:
        mg_.R3_OFF.discard("score_not_free")
    assert sig("nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4")["latent"] == 1024        # moe_latent_size
    assert S("openai/gpt-oss-20b")["act"] == S("nvidia/gpt-oss-puzzle-88B")["act"] == "swigluoai"
    assert "router_in_gate" in F("lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-8bit")
    assert "shared_gate" in F("Schiltmans/Qwen3.8-Flash-Next-Coder-cut240")         # Qwen4-exp


def test_moe_routing_rules():
    """moe_why on the Lean duplication results (MoERouting.lean) and the measured dense-host rule."""
    from graph.model_graph import MOE_SIG, MOE_SIG_R3, MOE_SIG_R3_DENSE, moe_why

    def s(**kw):
        d = dict(E=8, k=2, ew=1024, shared=0, renorm="R", renorm_cap=("P", "R"), score="softmax",
                 score_cap=("softmax",), n_group=0, topk_group=0, group_cap="", n_moe=24, n_dense=0, dense_w=0,
                 latent=0, bias=0)
        d.update(zip(MOE_SIG_R3, MOE_SIG_R3_DENSE))       # (round 3, batch 3a: the MoE-block forms, none here)
        d.update(kw)
        return tuple(d[k] for k in MOE_SIG)
    ok = lambda t, h, ht=1024, hh=1024, ft=8192, fh=8192: moe_why((t, ht, ft), (h, hh, fh)) is None
    t = s()
    assert ok(t, s(k=4, E=16, ew=512))                       # m = 2 copies, half-width slices
    assert not ok(t, s(k=3, E=16))                           # top-k is fixed: 3 is no multiple of 2
    assert not ok(t, s(k=4, E=12))                           # E_h < m E_t
    assert not ok(t, s(E=16)) and ok(t, s(E=16, bias=1)) and ok(t, s(E=16), hh=1025)   # extras need a sink
    plain = s(renorm="P")
    assert ok(plain, s(renorm="P", k=4, E=16)) and not ok(plain, s(renorm="P", k=4, E=24, bias=1))  # exactly m E_t
    assert not ok(plain, s(renorm="R", E=8))                 # the host's own router unrenormalised (transitivity)
    assert not ok(t, s(renorm_cap=("P",), renorm="P"))       # a code-fixed unrenormalised host
    assert not ok(t, s(score="sigmoid", score_cap=("sigmoid",)))   # no sigmoid -> softmax limit
    grp = s(E=64, k=8, n_group=8, topk_group=4, group_cap="top2sum", score="sigmoid", score_cap=("sigmoid",))
    assert ok(grp, s(E=64, k=8, group_cap="top2sum", score="sigmoid", score_cap=("sigmoid",)))   # groups are config
    assert not ok(grp, s(E=128, k=16, group_cap="top2sum", score="sigmoid", score_cap=("sigmoid",), bias=1))  # m = 2
    assert not ok(grp, s(E=64, k=8, group_cap="max", score="sigmoid", score_cap=("sigmoid",)))
    assert not ok(s(shared=2048), s(k=4, E=16, ew=2048, shared=1024))   # the shared width on its own
    assert not ok(s(n_moe=22, n_dense=2, dense_w=4096), s(n_moe=24, n_dense=0))   # no spare MoE layer for the dense ones
    assert ok(s(n_moe=22, n_dense=2, dense_w=2048), s(n_moe=24, n_dense=0))       # k e_w = 2048: MoE-as-dense
    dense = s(E=0, k=0, ew=0, n_moe=0, n_dense=24, dense_w=4096)
    top1 = s(k=1, E=4)
    assert ok(top1, dense, hh=1028, fh=8192 + 24) and not ok(top1, dense, hh=1027, fh=8192 + 24)   # +E channels
    assert not ok(top1, dense, hh=1028, fh=8192 + 23)                                              # +2E(E-1) units
    assert not ok(s(k=2), dense, hh=4096, fh=1 << 20)                                              # top-k >= 2: wall


# --- round 2, pass 4 (2026-11-13): recurrent capacity, the recurrent variants, block-causal / self-conditioning ---

def D(name):
    return dims_of(C(name))


def _rc(name):
    from graph.model_graph import RC_DIMS
    d = D(name)
    return {k: d[k] for k in RC_DIMS + ("rc_rule",)}


def test_recurrent_variants_split():
    # Mamba-1: plain / weight-free dt-B-C norms (Falcon-Mamba) / weighted (Jamba, which nests the weight-free one)
    assert {x for x in F("state-spaces/mamba-130m-hf") if x.startswith("rf_")} == {"rf_mamba1"}
    assert {x for x in F("tiiuae/falcon-mamba-7b") if x.startswith("rf_")} == {"rf_mamba1_bcdt"}
    assert {x for x in F("ai21labs/Jamba-v0.1") if x.startswith("rf_")} == {"rf_mamba1_bcdt", "rf_mamba1_bcdtw"}
    assert "rf_hymba" in F("nvidia/Hymba-1.5B-Base") and "rf_sambay" in F("microsoft/Phi-4-mini-flash-reasoning")
    # Mamba-2: Falcon-H1 without mamba_rms_norm, PLaMo-2's B / C / dt norms
    assert "rf_mamba2_nonorm" in F("tiiuae/Falcon-H1-0.5B-Base") and "rf_mamba2" in F("tiiuae/Falcon-H1-7B-Instruct")
    assert "rf_plamo2" in F("pfnet/plamo-2-1b")
    # GDN: OLMo-Hybrid beta in (0, 2), GPT-S3 GDN-2, Amazon GKA
    assert "rf_gdn_negeig" in F("allenai/Olmo-Hybrid-7B")
    assert "rf_gdn2" in F("AxiomicLabs/GPT-S3-3M") and "rf_gka" in F("amazon/GKA-primed-HQwen3-8B-Instruct")
    # KDA: the lower bound is config in GLM-5-Next / Kimi-K3 / Youtu / Ling-3 (they also run the softplus form);
    # Kimi-Linear-48B's code has only the softplus form
    assert {x for x in F("moonshotai/Kimi-Linear-48B-A3B-Instruct") if x.startswith("rf_")} == {"rf_kda"}
    for m in ("moonshotai/Kimi-K3.1-Pro", "tencent/Youtu-LLM-A7B-Base", "inclusionAI/Ling-3.1-flash"):
        assert {x for x in F(m) if x.startswith("rf_")} == {"rf_kda", "rf_kda_lb"}, m


def test_recurrent_capacity_dims():
    from graph.model_graph import RC_GDN, RC_GE, RC_KDA, RC_M2
    n = _rc("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    assert (n["rc_h"], n["rc_v"], n["rc_k"], n["rc_kh"], n["rc_ng"], n["rc_rule"]) == (96, 80, 128, 8, 8, RC_M2)
    g = _rc("ibm-granite/granite-4.0-h-tiny")                 # full-width gated norm, one B / C group
    assert (g["rc_h"], g["rc_v"], g["rc_kh"], g["rc_ng"]) == (48, 64, 1, 1)
    f = _rc("tiiuae/Falcon-H1-0.5B-Base")                     # no norm: GE on heads per group
    assert (f["rc_ng"], f["rc_rule"]) == (0, RC_GE)
    m = _rc("tiiuae/falcon-mamba-7b")                         # d_inner from the checkpoint's intermediate_size
    assert (m["rc_h"], m["rc_k"], m["rc_rank"], m["rc_conv"]) == (8192, 16, 256, 4)
    q = _rc("Qwen/Qwen3-Coder-Next")
    assert (q["rc_h"], q["rc_kh"], q["rc_k"], q["rc_v"], q["rc_rule"]) == (32, 16, 128, 128, RC_GDN)
    k = _rc("moonshotai/Kimi-K3.1-Pro")                       # use_full_rank_gate: the gate rank is the width
    assert (k["rc_h"], k["rc_k"], k["rc_rule"]) == (96, 128, RC_KDA) and k["rc_rank"] == D("moonshotai/Kimi-K3.1-Pro")["hidden"]
    assert _rc("LiquidAI/LFM3-1.7B-Base")["rc_conv"] == 7 and _rc("LiquidAI/LFM2-1.2B")["rc_conv"] == 3


def test_recurrent_capacity_rules():
    from graph.model_graph import RC_DIMS, RC_GDN, RC_GE, RC_KDA, RC_M2, _rc_pair_ok
    z = dict.fromkeys(RC_DIMS, 0)
    mk = lambda rule, **kw: {**z, **kw, "rc_rule": rule}
    # Mamba-1 (and every variant): zero-padded channels / states / ranks / taps are inert
    t = mk(RC_GE, rc_h=1536, rc_k=16, rc_v=1, rc_kh=1, rc_conv=4, rc_rank=48)
    assert _rc_pair_ok(t, mk(RC_GE, rc_h=2048, rc_k=16, rc_v=1, rc_kh=1, rc_conv=4, rc_rank=64))
    assert not _rc_pair_ok(t, mk(RC_GE, rc_h=2048, rc_k=8, rc_v=1, rc_kh=1, rc_conv=4, rc_rank=64))
    assert not _rc_pair_ok(t, mk(RC_GE, rc_h=2048, rc_k=16, rc_v=1, rc_kh=1, rc_conv=3, rc_rank=64))   # narrower conv
    # Mamba-2's grouped gated norm: host groups of m whole copies of a target group
    m2 = lambda nh, P, G, N=128: mk(RC_M2, rc_h=nh, rc_v=P, rc_kh=G, rc_ng=G, rc_k=N, rc_conv=4)
    assert _rc_pair_ok(m2(128, 64, 8), m2(256, 64, 8, 256))            # Nemotron-H-8B <- 56B: w 1024 -> 2048, m = 2
    assert not _rc_pair_ok(m2(96, 80, 8), m2(256, 64, 8, 256))         # lumi-4b <- 56B: w 960 does not divide 2048
    assert not _rc_pair_ok(m2(32, 128, 1, 256), m2(24, 128, 1, 256))   # Falcon-H1-3B <- 7B: 24 < 32 heads
    assert not _rc_pair_ok(m2(64, 64, 1), m2(112, 64, 2, 64))          # Zamba2-1.2B <- 7B: d_state 64 < 128
    assert not _rc_pair_ok(m2(64, 64, 2), m2(128, 64, 1))              # one host group cannot hold two target groups
    assert _rc_pair_ok(m2(64, 64, 1), m2(64, 128, 1)) and not _rc_pair_ok(m2(64, 80, 1), m2(64, 128, 1))
    # GDN: the key head dim fixes q's 1 / sqrt(dk) scale (equal); value heads replicate; key heads by count
    gd = lambda hv, hk, dk, dv: mk(RC_GDN, rc_h=hv, rc_kh=hk, rc_k=dk, rc_v=dv, rc_conv=4)
    assert _rc_pair_ok(gd(16, 16, 128, 128), gd(32, 16, 128, 128))
    assert not _rc_pair_ok(gd(16, 4, 64, 64), gd(32, 16, 128, 128))    # dk 64 vs 128
    assert _rc_pair_ok(gd(16, 16, 128, 64), gd(16, 16, 128, 128)) and not _rc_pair_ok(gd(16, 16, 128, 96), gd(16, 16, 128, 128))
    assert not _rc_pair_ok(gd(32, 16, 128, 128), gd(32, 8, 128, 128))  # 16 key heads need 16 host key heads
    # KDA: head_dim equal, heads / conv / gate rank >=
    kd = lambda H, hd, r: mk(RC_KDA, rc_h=H, rc_k=hd, rc_v=hd, rc_kh=H, rc_conv=4, rc_rank=r)
    assert _rc_pair_ok(kd(32, 128, 128), kd(96, 128, 7168)) and not _rc_pair_ok(kd(32, 128, 7168), kd(96, 128, 128))


def test_recurrent_capacity_edges_from_the_review():
    # credited payable edges the literature review measured as broken (recurrent capacity / variants), now walled
    pts = lambda *ns: [{"slots": S(n), "dims": D(n), "members": [n]} for n in ns]
    from graph.model_graph import emul_matrix
    for t, h in (("tiiuae/Falcon-H1-3B-Instruct", "tiiuae/Falcon-H1-7B-Instruct"),
                 ("ceia-br/lumi-4b", "nvidia/Nemotron-H-56B-Base-8K"),
                 ("Zyphra/Zamba2-1.2B-instruct", "Zyphra/Zamba2-7B-Instruct"),
                 ("ibm-granite/granite-4.0-h-350m", "tiiuae/Falcon-H1-34B-Instruct"),
                 ("tiiuae/Falcon-H1-0.5B-Base", "tiiuae/Falcon-H1-7B-Instruct"),
                 ("state-spaces/mamba-130m-hf", "tiiuae/falcon-mamba-7b"),
                 ("tiiuae/falcon-mamba-tiny-dev", "state-spaces/mamba-130m-hf"),
                 ("AxiomicLabs/GPT-S3-3M", "MemTensor/MemT-1.0-Nano")):
        P = pts(t, h)
        # (round 2, pass 5: the eps lever relaxes the Mamba-2 head rule -- a target head split over ceil(P_t / P_h)
        # host heads, the zero-padded group's eps set in the config -- so lumi-4b <- Nemotron-H-56B is payable; the
        # exact slice keeps the review's wall)
        # (round 3, mamba1_lever with mamba_ffn_none: mamba-130m <- falcon-mamba-7b is payable too -- the dt / B / C
        # norms switched off by the eps lever, measured 100% / 95.8% on Falcon-Mamba, hf/data/mixer_rules_agreement.jsonl;
        # the review measured the unlevered construction, so the exact slice keeps its wall; was a payable wall)
        # (fr-mixer, m2_nonorm_lever: the norm-free Falcon-H1-0.5B on a gated-norm host is the residual-scaled eps lever,
        # measured 100% / 100% on the 0.5B itself (hf/data/mixer_rules_agreement.jsonl) -- payable, the exact slice keeps
        # the review's wall, which measured the unlevered construction)
        lever = (t, h) in {("ceia-br/lumi-4b", "nvidia/Nemotron-H-56B-Base-8K"),
                           ("state-spaces/mamba-130m-hf", "tiiuae/falcon-mamba-7b"),
                           ("tiiuae/Falcon-H1-0.5B-Base", "tiiuae/Falcon-H1-7B-Instruct")}
        assert not emul_matrix(P, mode="exact" if lever else "payable")[0, 1], (t, h)
        if lever:
            assert emul_matrix(P, mode="payable")[0, 1]
    # ... and a within-family edge that stays: Falcon-H1-Tiny-90M (no norm) <- Falcon-H1-0.5B (no norm)
    P = pts("tiiuae/Falcon-H1-Tiny-90M-Instruct", "tiiuae/Falcon-H1-0.5B-Base")
    assert emul_matrix(P, mode="payable")[0, 1]


def test_block_causal_and_selfcond():
    from graph.model_graph import payable_reaches
    # SDAR runs non-causal attention under a caller mask: AR <= block, block == diffusion (mask); AR plays neither
    assert payable_reaches("gen", "ar", "block") and payable_reaches("gen", "block", "diffusion")
    assert payable_reaches("gen", "diffusion", "block") and not payable_reaches("gen", "block", "ar")
    assert S("JetLM/SDAR-1.7B-Chat")["gen"] == "block"
    # self-conditioning: an extra input; DiffusionGemma's path ends in an always-on weight-free embedding norm
    # (selfcond_norm: a wall both ways), LLaDA2.1-ORL's has no norm (a host zeroes it)
    assert {"selfcond", "selfcond_norm"} <= F("nvidia/diffusiongemma-26B-A4B-it-NVFP4")
    assert "selfcond" in F("yifanyu/LLaDA2.1-mini-ORL") and "selfcond_norm" not in F("yifanyu/LLaDA2.1-mini-ORL")
    # Zamba's shared block: a wall both ways; Zamba2 -> Zamba2 by shared blocks (a multiple) and adapter rank
    assert "zamba_shared" in F("Zyphra/Zamba2-7B-Instruct") and "zamba_shared" in F("Zyphra/Zamba-7B-v1")
    assert D("Zyphra/Zamba2-7B-Instruct")["zamba_mem"] == 2 and D("Zyphra/Zamba2-1.2B-instruct")["zamba_mem"] == 1
