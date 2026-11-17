"""loader_coverage.py -- which zoo models the universal host is checked against IN THEIR OWN CODE.

graph/minimal_emulator.py counts a model as carried when its mixer family is tested and its
activation / norm have host branches; the other slots are priced by tested constructions. This
inventory asks the stricter, per-model question: is there a whole-model `from_*` loader in
elephantine/universal.py that ACCEPTS this model's model_type and config (the config fields the
loader asserts on or reads), and does a test run that loader against the model_type's real code
(the stock `transformers` class, or the model's own remote code / plugin) and compare logits?

For every classified decoder LM in the snapshot (graph.minimal_emulator.load: iter_json_configs,
_junky, classify, nonzero dims -- the same 3,906) this reports, by model_type:
    models        in the snapshot
    loader        accepted by some from_* loader (LOADERS[...].check returns None)
    tested        ... and a real-code test exercises that loader on that model_type (TESTS)
    checkpoint    ... and one of those tests runs a real released checkpoint of that type
A model whose output is not a next-token distribution (a score / value head, an embedding, Sundial's flow
head: READOUTS / readout_of, M12) is dispatched to its readout loader instead, tested against that readout.

Run:  python -m elephantine.loader_coverage [--all] [--why]
"""
from __future__ import annotations

import argparse
import ast
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T = "tests/test_universal_elephantine.py"


def _rope_type(c):
    rs = c.get("rope_scaling") or c.get("rope_parameters") or {}
    if not isinstance(rs, dict):
        return None
    t = rs.get("rope_type", rs.get("type"))
    return None if t in (None, "default") else t


def _arch(c):
    a = c.get("architectures") or []
    return a[0] if isinstance(a, list) and a else None


@dataclass
class Loader:
    name: str                       # the from_* in elephantine/universal.py
    types: tuple                    # model_types it is written for
    archs: tuple = ()               # accepted `architectures[0]` (None / absent is accepted too)
    check: Callable = None          # cfg -> None (accepted) or the reason it is not
    note: str = ""

    def accepts(self, cfg):
        if cfg.get("model_type") not in self.types:
            return f"model_type {cfg.get('model_type')!r}"
        a = _arch(cfg)
        if self.archs and a is not None and a not in self.archs and not _stock_class(cfg.get("model_type"), a):
            return f"architecture {a} (not the stock class: own / remote code)"
        return self.check(cfg) if self.check else None


def _stock_class(mt, arch):
    """arch is a stock transformers class of this model_type (e.g. the headless BloomModel, or
    LlamaForSequenceClassification: the same decoder weights)."""
    import transformers
    try:
        cls = getattr(transformers, arch)
    except Exception:
        return False
    return getattr(getattr(cls, "config_class", None), "model_type", None) == mt


_GELUS = ("gelu", "gelu_python", "gelu_new", "gelu_pytorch_tanh", "gelu_fast", "gelu_accurate", "gelu_tanh")


def _act_ok(key, gated, default):
    """The FFN activation has a UFFN branch: gated silu / mish / GeLU (GeGLU), ungated GeLU / relu / relu^2."""
    ok = (("silu", "swish", "mish") + _GELUS) if gated else (_GELUS + ("relu", "relu2", "relu_squared"))

    def chk(c):
        a = str(c.get(key) or default).lower()
        return None if a in ok else f"{key}={a} ({'gated' if gated else 'ungated'}: no UFFN branch)"
    return chk


# RoPE scalings the host carries as a fixed inv_freq (+ attention_scaling folded into the scale). dynamic NTK
# and LongRoPE switch inv_freq with the sequence length: the host holds the short-context setting (exact
# for T <= original_max_position_embeddings, which is where the tests run them).
_ROPES = (None, "linear", "llama3", "yarn", "longrope", "su", "dynamic")


def _rope_ok(c):
    t = _rope_type(c)
    return None if t in _ROPES else f"rope {t}"


def _all(*checks):
    def chk(c):
        for f in checks:
            r = f(c)
            if r:
                return r
        return None
    return chk


def _not(key, why=None):
    return lambda c: (why or key) if c.get(key) else None


def _pseudobrain_check(c):
    return None if c.get("pb_hybrid_layers") else "no pb_hybrid_layers (a plain Qwen3: not the plugin)"


_LLAMA_TYPES = {
    "llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM", "ministral": "MinistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM", "helium": "HeliumForCausalLM",
    "seed_oss": "SeedOssForCausalLM", "ernie4_5": "Ernie4_5ForCausalLM", "smollm3": "SmolLM3ForCausalLM",
    "arcee": "ArceeForCausalLM", "mixtral": "MixtralForCausalLM", "qwen2_moe": "Qwen2MoeForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM", "olmoe": "OlmoeForCausalLM", "ernie4_5_moe": "Ernie4_5_MoeForCausalLM",
    "glm4_moe": "Glm4MoeForCausalLM", "dots1": "Dots1ForCausalLM",
    # transformers 5
    "ministral3": "Ministral3ForCausalLM", "hy_v3": "HYV3ForCausalLM", "exaone_moe": "ExaoneMoeForCausalLM",
    "solar_open": "SolarOpenForCausalLM", "mellum": "MellumForCausalLM", "minimax_m2": "MiniMaxM2ForCausalLM",
}


def _llama_check(c):
    if c.get("pb_hybrid_layers"):
        return "pb_hybrid_layers (PseudoBrain: from_pseudobrain)"
    gated = c.get("model_type") != "arcee"
    return _all(_rope_ok, _act_ok("hidden_act", gated, "silu" if gated else "relu2"))(c)


def _qwen4_check(c):
    """transformers 5.18's Qwen4-Exp accepts the config: its layer kinds (full_attention is read as indexed_attention),
    a sigmoid / silu output gate, an EOS id when PLE layers are on."""
    extra = set(c.get("layer_types") or ()) - {"linear_attention", "indexed_attention", "full_attention"}
    if extra:
        return f"layer types {sorted(extra)} (not transformers 5.18's)"
    if (c.get("output_gate_type") or c.get("hidden_act") or "silu") not in ("sigmoid", "silu"):
        return "output gate"
    if c.get("ple_layer_ids") and c.get("eos_token_id") in (None, []):
        return "PLE layers without an eos_token_id"
    return None


def _k2_check(c):
    """K2-Horizon dense with the Llama attention (no output gate, no qk-norm, full-head RoPE, no MoE / MoVA)."""
    hd = c.get("head_dim") or (c.get("hidden_size") or 0) // max(c.get("num_attention_heads") or 1, 1)
    if c.get("attention_gate_func") or c.get("query_key_norm"):
        return "attention_gate_func / query_key_norm"
    if (c.get("rope_head_dim") or hd) != hd:
        return "partial (interleaved) RoPE"
    if c.get("num_experts") or c.get("mova_num_experts"):
        return "MoE / MoVA layers"
    return None


LOADERS = [
    Loader("from_pseudobrain", ("qwen3", "qwen3_moe"), (), _pseudobrain_check),
    Loader("from_llama", tuple(_LLAMA_TYPES), tuple(_LLAMA_TYPES.values()), _llama_check),
    Loader("from_gpt2", ("gpt2",), ("GPT2LMHeadModel",),
           _all(_not("add_cross_attention"), _act_ok("activation_function", False, "gelu_new"))),
    Loader("from_gpt_neox", ("gpt_neox",), ("GPTNeoXForCausalLM",),
           _all(lambda c: f"rope {_rope_type(c)}" if _rope_type(c) else None, _act_ok("hidden_act", False, "gelu"))),
    Loader("from_gemma", ("gemma", "gemma2", "vaultgemma", "gemma3_text"),
           ("GemmaForCausalLM", "Gemma2ForCausalLM", "VaultGemmaForCausalLM", "Gemma3ForCausalLM"), _rope_ok),
    Loader("from_granite", ("granite", "granitemoe", "granitemoeshared", "granite_swa", "granitemoe_swa"),
           ("GraniteForCausalLM", "GraniteMoeForCausalLM", "GraniteMoeSharedForCausalLM", "GraniteSWAForCausalLM",
            "GraniteMoeSWAForCausalLM"),
           _all(_rope_ok, _act_ok("hidden_act", True, "silu"))),
    Loader("from_phi3", ("phi3",), ("Phi3ForCausalLM",), _all(_rope_ok, _act_ok("hidden_act", True, "silu"))),
    Loader("from_glm", ("glm", "glm4"), ("GlmForCausalLM", "Glm4ForCausalLM"), _rope_ok),
    Loader("from_olmo", ("olmo", "olmo2", "olmo3", "flex_olmo", "exaone4"),
           ("OlmoForCausalLM", "Olmo2ForCausalLM", "Olmo3ForCausalLM", "FlexOlmoForCausalLM", "Exaone4ForCausalLM"),
           _rope_ok),
    Loader("from_cohere", ("cohere",), ("CohereForCausalLM",)),
    Loader("from_cohere2", ("cohere2",), ("Cohere2ForCausalLM",), _rope_ok),
    # HunYuan-A13B's remote MoE router drops over-capacity assignments (modeling_hunyuan.py topkgating): before the
    # stock loader, which would otherwise accept its config
    Loader("from_hunyuan_capacity", ("hunyuan_v1_moe",), (),
           lambda c: None if "hunyuan." in str((c.get("auto_map") or {}).get("AutoModelForCausalLM", ""))
           else "the stock HunYuan-MoE class (from_hunyuan)"),
    Loader("from_hunyuan", ("hunyuan_v1_dense", "hunyuan_v1_moe"),
           ("HunYuanDenseV1ForCausalLM", "HunYuanMoEV1ForCausalLM"), _rope_ok),
    Loader("from_starcoder2", ("starcoder2",), ("Starcoder2ForCausalLM",),
           _all(_rope_ok, _act_ok("hidden_act", False, "gelu_pytorch_tanh"))),
    Loader("from_nemotron", ("nemotron",), ("NemotronForCausalLM",),
           _all(_rope_ok, _act_ok("hidden_act", False, "relu2"))),
    Loader("from_stablelm", ("stablelm",), ("StableLmForCausalLM",), _rope_ok),
    Loader("from_persimmon", ("persimmon",), ("PersimmonForCausalLM",),
           _all(_rope_ok, _act_ok("hidden_act", False, "relu2"))),
    Loader("from_phi", ("phi",), ("PhiForCausalLM",), _all(_rope_ok, _act_ok("hidden_act", False, "gelu_new"))),
    Loader("from_gptj", ("gptj", "codegen"), ("GPTJForCausalLM", "CodeGenForCausalLM"),
           _act_ok("activation_function", False, "gelu_new")),
    Loader("from_gpt_bigcode", ("gpt_bigcode",), ("GPTBigCodeForCausalLM",),
           _act_ok("activation_function", False, "gelu_pytorch_tanh")),
    Loader("from_bloom", ("bloom",), ("BloomForCausalLM",), _not("apply_residual_connection_post_layernorm")),
    Loader("from_opt", ("opt",), ("OPTForCausalLM",),
           _act_ok("activation_function", False, "relu")),
    Loader("from_falcon", ("falcon",), ("FalconForCausalLM",), _all(_rope_ok, _act_ok("activation", False, "gelu"))),
    Loader("from_deepseek", ("deepseek_v2", "deepseek_v3", "glm4_moe_lite", "youtu", "minicpm3", "axk1", "mistral4"),
           ("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "Glm4MoeLiteForCausalLM", "YoutuForCausalLM",
            "MiniCPM3ForCausalLM", "AXK1ForCausalLM", "Mistral4ForCausalLM"),
           _all(_rope_ok, _not("attention_bias", "MLA latent projection biases"))),
    Loader("from_bamba", ("bamba", "granitemoehybrid"), ("BambaForCausalLM", "GraniteMoeHybridForCausalLM"), _rope_ok),
    Loader("from_falcon_h1", ("falcon_h1",), ("FalconH1ForCausalLM",), _rope_ok),
    Loader("from_jamba", ("jamba",), ("JambaForCausalLM",)),
    Loader("from_falcon_mamba", ("falcon_mamba",), ("FalconMambaForCausalLM",)),
    Loader("from_nemotron_h", ("nemotron_h",), ("NemotronHForCausalLM",),
           _act_ok("mlp_hidden_act", False, "relu2")),
    Loader("from_afmoe", ("afmoe",), ("AfmoeForCausalLM",), _rope_ok),
    Loader("from_recurrent_gemma", ("recurrent_gemma",), ("RecurrentGemmaForCausalLM",)),
    Loader("from_kimi_linear", ("kimi_linear",), ("KimiLinearForCausalLM",)),
    Loader("from_phimoe", ("phimoe",), ("PhimoeForCausalLM",), _rope_ok),
    Loader("from_mimo_v2_flash", ("mimo_v2_flash",), ("MiMoV2FlashForCausalLM",)),
    Loader("from_qwen_vl", ("qwen2_5_vl", "qwen2_5_vl_text", "qwen3_vl", "qwen3_vl_text", "qwen3_vl_moe", "qwen3_vl_moe_text"),
           ("Qwen2_5_VLForConditionalGeneration", "Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration")),
    Loader("from_hyperclovax", ("hyperclovax",), ("HyperCLOVAXForCausalLM",), _rope_ok),
    Loader("from_olmo_hybrid", ("olmo_hybrid",), ("OlmoHybridForCausalLM",), _rope_ok),
    Loader("from_minimax", ("minimax",), ("MiniMaxForCausalLM",), _rope_ok),
    Loader("from_gemma4", ("gemma4_text", "gemma4_unified_text", "gemma4"),
           ("Gemma4ForCausalLM", "Gemma4UnifiedForCausalLM", "Gemma4ForConditionalGeneration"),
           _act_ok("hidden_activation", True, "gelu_pytorch_tanh")),
    Loader("from_dsa", ("glm_moe_dsa", "deepseek_v32"), ("GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM"), _rope_ok),
    Loader("from_laguna", ("laguna",), ("LagunaForCausalLM",), _rope_ok),
    Loader("from_gpt_oss", ("gpt_oss",), ("GptOssForCausalLM",), _rope_ok),
    Loader("from_apertus", ("apertus",), ("ApertusForCausalLM",)),
    Loader("from_mamba", ("mamba",), ("MambaForCausalLM",)),
    Loader("from_qwen3next", ("qwen3_next",), ("Qwen3NextForCausalLM",)),
    Loader("from_qwen35", ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_5", "qwen3_5_moe", "memt_qwen3_5_moe",
                           "qwen3_5_gdn24", "allgdn", "edgerock")),
    Loader("from_plamo2", ("plamo2", "plamo2vl")),
    Loader("from_hymba", ("hymba",)),
    Loader("from_alice", ("alice_ai",)),
    Loader("from_mini_k3", ("mini_k3",)),
    Loader("from_phi4flash", ("phi4flash",)),
    Loader("from_granite_switch", ("granite_switch",)),
    Loader("from_gigachat35", ("gigachat3_5",)),
    Loader("from_lfm", ("lfm2", "lfm2_moe", "lfm3", "lfm3_rd")),
    Loader("from_ceno", ("ceno",)),
    Loader("from_glm5_next", ("glm5_next_text",)),
    Loader("from_youtu", ("youtu_vl_moe",)),
    Loader("from_jet_nemotron", ("jet_nemotron",)),
    Loader("from_gpts3", ("gpts3_hybrid",)),
    Loader("from_gptx3", ("gptx3",)),
    Loader("from_hybrid_qwen3", ("hybrid_qwen3",)),
    Loader("from_motif", ("Motif",)),
    Loader("from_bailing_hybrid", ("bailing_hybrid",),
           ("BailingMoeV2_5ForCausalLM", "BailingMoeV3ForCausalLM", "BailingHybridForCausalLM"),
           _all(_rope_ok, *[_not(k) for k in ("use_mla_nope", "value_norm", "up_proj_norm", "use_nGPT",
                                               "scale_router_input", "use_kda_lora")],
                lambda c: "low-rank KDA gates (no_kda_lora false): untested" if c.get("no_kda_lora") is False else None)),
    Loader("from_minicpm_sala", ("minicpm_sala",), ("MiniCPMSALAForCausalLM",),
           lambda c: "rope_scaling" if c.get("rope_scaling") else None),
    Loader("from_rwkv", ("rwkv",), ("RwkvForCausalLM",),
           lambda c: None if (c.get("attention_hidden_size") or c.get("hidden_size")) == c.get("hidden_size")
           else "attention_hidden_size != hidden_size"),
    # nanowhale's own remote modeling_deepseek_v4.py (not the stock class: comb X hyper-connections): before the stock
    # DeepSeek-V4 loader, which would otherwise accept its config
    Loader("from_nanowhale", ("deepseek_v4",), (),
           lambda c: None if "modeling_deepseek_v4" in str((c.get("auto_map") or {}).get("AutoModelForCausalLM", ""))
           else "the stock DeepSeek-V4 class (from_deepseek_v4)"),
    Loader("from_deepseek_v4", ("deepseek_v4", "deepseek_v41", "deepseek_v41_text"),
           ("DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"),
           lambda c: None if (c.get("n_shared_experts") or 1) == 1 and not c.get("attention_bias") and not c.get("mlp_bias")
           else "n_shared_experts != 1 / biases"),
    # eleph-gaps-stream (elephantine/branches_stream.py, tests/test_eleph_gaps_stream.py)
    Loader("from_qwen4_exp", ("qwen4_exp_text", "qwen4_exp"),
           ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"), _qwen4_check),
    Loader("from_zamba2", ("zamba2",), ("Zamba2ForCausalLM",)),
    Loader("from_zamba", ("zamba",), ("ZambaForCausalLM",)),
    Loader("from_nanochat", ("nanochat_gpt", "gaggle", "sn38-nanochrono"),
           ("NanochatGPTForCausalLM", "GaggleForCausalLM", "NanochronoForCausalLM")),
    Loader("from_arloop", ("arloop", "arloop_baseline"), ("ARLoopForCausalLM", "ARLoopBaselineForCausalLM")),
    Loader("from_k2_horizon", ("k2_horizon",), ("K2HorizonForCausalLM",), _k2_check),
    # the LongCat configs carry no model_type (model_graph._lm_config reads it from the architecture); only the stock
    # class's architecture: LongcatCausalLM (Thinking-2601, LongCat-2.0, Flash-Lite-Sparse) is its own remote code
    Loader("from_ouro", ("ouro",), ("OuroForCausalLM",)),
    Loader("from_hrm_text", ("hrm_text",), ("HrmTextForCausalLM",)),
    Loader("from_xing", ("xing4_0",), ("Xing4_0ForCausalLM",),
           lambda c: "group-limited routing" if (c.get("n_group") or 1) > 1 and c.get("topk_group") != c.get("n_group")
           else None),
    Loader("from_hy_v4", ("hy_v4",), ("HYV4ForCausalLM",)),
    Loader("from_crowd", ("crowd",), ("CrowdForCausalLM",)),
    Loader("from_llama4", ("llama4_text", "llama4"), ("Llama4ForCausalLM", "Llama4ForConditionalGeneration"),
           lambda c: None if (c.get("rope_scaling") or {}).get("rope_type", "llama3") == "llama3" else "rope_scaling"),
    Loader("from_arctic", ("arctic",), ("ArcticForCausalLM",),
           lambda c: "rope_scaling" if c.get("rope_scaling") else None),
    Loader("from_llama_moe", ("llama_moe",), ("LlamaMoEForCausalLM",),
           lambda c: next((f"{k}={c.get(k)!r}" for k, ok in (
               ("gate_type", lambda v: (v or "TopKBalancedNoisyGate") == "TopKBalancedNoisyGate"),
               ("calculator_type", lambda v: (v or "UniversalCalculator") == "UniversalCalculator"),
               ("gate_use_softmax", lambda v: v is not False), ("multiply_gate_scores", lambda v: v is not False),
               ("add_weight_norm", lambda v: not v), ("rope_scaling", lambda v: not v)) if not ok(c.get(k))), None)),
    Loader("from_dhara", ("dhara_ar",), ("DharaARForCausalLM",),
           lambda c: "YaRN rope_scaling" if (c.get("rope_scaling") or {}).get("type") == "yarn" else None),
    Loader("from_dory", ("dory",), ("DoryForCausalLM",),
           lambda c: None if not c.get("attention_output_gate_logit_scale_init") else "a logit-scaled output gate"),
    Loader("from_nanbeige", ("nanbeige",), ("NanbeigeForCausalLM",),
           lambda c: next((k for k in ("enable_hyper_connection", "enable_double_loop_split", "loop_share_kv",
                                       "enable_depth_attention", "qk_layernorm", "emb_neighbor_num", "rope_scaling",
                                       "insert_ngram_layer_idx", "ngram_insert_all_layers") if c.get(k)), None)),
    Loader("from_longcat_flash", ("longcat_flash",), ("LongcatFlashForCausalLM",),
           lambda c: None if (c.get("architectures") or [None])[0] == "LongcatFlashForCausalLM"
           else "architecture LongcatCausalLM (own remote code, not the stock class)"),
    # QwenDLM-ORL: SDAR + its self-conditioning input path (the stock from_sdar rejects self_conditioning_prob)
    Loader("from_sdar_selfcond", ("sdar",), ("SDARForCausalLM",),
           lambda c: None if c.get("self_conditioning_prob") else "no self-conditioning (from_sdar)"),
    Loader("from_sdar", ("sdar", "sdar_moe"), ("SDARForCausalLM", "SDARMoeForCausalLM"),
           _all(_rope_ok, _not("self_conditioning_prob", "self-conditioning input path (QwenDLM-ORL): not carried"))),
]

# eleph-gaps-attn (elephantine/branches_attn.py): the round-3 attention features' whole-model loaders
_JETMOE_KEYS = (("moe_num_experts", "num_local_experts", 8), ("moe_top_k", "num_experts_per_tok", 2),
                ("ffn_hidden_size", "intermediate_size", 5632))


def _jetmoe_check(c):
    """jetmoe-8b's config is the jetmoe package's schema (moe_num_experts / moe_top_k / ffn_hidden_size); the stock
    JetMoeForCausalLM (its transformers port) reads num_local_experts / num_experts_per_tok / intermediate_size and
    their defaults: accepted when the package keys equal what the stock class reads."""
    for old, new, dflt in _JETMOE_KEYS:
        if old in c and c[old] != c.get(new, dflt):
            return f"{old}={c[old]} but the stock class reads {new}={c.get(new, dflt)}"
    return _act_ok("activation_function", True, "silu")(c)


ATTN_LOADERS = [
    Loader("from_qwen3_dros", ("qwen3",), (), lambda c: None if c.get("d_ros_config") else "no d_ros_config"),
    Loader("from_gptx2", ("gptx2", "gpts2"), (),
           _not("use_symplectic_rope", "symplectic RoPE (GPT-X2 CIx: rope_warp, a content-warped phase): not carried")),
    Loader("from_mm200", ("mm200",), (),
           lambda c: "use_weight_norm (untested)" if c.get("use_weight_norm") else None),
    Loader("from_doge", ("doge",), ("DogeForCausalLM",),
           _all(_not("is_moe", "Doge CDMoE: not carried"), _act_ok("hidden_act", True, "silu"))),
    Loader("from_jetmoe", ("jetmoe",), (), _jetmoe_check),
]
LOADERS = ATTN_LOADERS[:1] + LOADERS + ATTN_LOADERS[1:]   # D-ROS before from_llama (a qwen3 with d_ros_config)

# model_type -> [(test function, kind, loader)]; kind: "stock" (the transformers class, synthetic
# weights), "remote" (the model's own code), "plugin" (its vLLM plugin), "checkpoint" (real weights),
# "reference" (a labelled transcription of the remote code, read as text, not run).
TESTS = {
    "llama":            [("test_one_host_carries_a_softmax_ar_model", "stock", "from_llama"),
                         ("test_same_host_carries_llama_mish", "stock", "from_llama")],
    "gpt2":             [("test_same_host_carries_gpt2_gelu", "stock", "from_gpt2"),
                         ("test_same_host_carries_real_gelu_checkpoints", "checkpoint", "from_gpt2")],
    "gpt_neox":         [("test_same_host_carries_gpt_neox_gelu", "stock", "from_gpt_neox"),
                         ("test_same_host_carries_real_gelu_checkpoints", "checkpoint", "from_gpt_neox")],
    "gemma":            [("test_same_host_carries_gemma1_geglu", "stock", "from_gemma")],
    "gemma3_text":      [("test_same_host_carries_gemma3_geglu_tanh", "stock", "from_gemma"),
                         ("test_same_host_carries_real_gelu_checkpoints", "checkpoint", "from_gemma")],
    "apertus":          [("test_same_host_carries_apertus_xielu", "stock", "from_apertus")],
    "cohere":           [("test_same_host_carries_a_parallel_block_cohere", "stock", "from_cohere")],
    "mamba":            [("test_same_host_carries_a_mamba1_model", "stock", "from_mamba")],
    "qwen3_next":       [("test_same_host_carries_a_qwen3next_hybrid", "stock", "from_qwen3next")],
    "qwen3_5_text":     [("test_same_host_carries_qwen35_and_diba", "stock", "from_qwen35"),
                         ("test_same_host_carries_qwen35_gdn24_all_linear", "stock", "from_qwen35")],
    "qwen3_5_moe_text": [("test_same_host_carries_qwen35_moe_and_memt", "stock", "from_qwen35"),
                         ("test_same_host_carries_qwen35_moe_and_memt_notremote", "stock", "from_qwen35")],
    "memt_qwen3_5_moe": [("test_same_host_carries_qwen35_moe_and_memt", "remote", "from_qwen35"),
                         ("test_same_host_carries_qwen35_moe_and_memt_notremote", "reference", "from_qwen35")],
    "qwen3_5_gdn24":    [("test_same_host_carries_qwen35_gdn24_all_linear", "stock", "from_qwen35")],
    "allgdn":           [("test_same_host_carries_allgdn_engram", "remote", "from_qwen35")],
    "edgerock":         [("test_same_host_carries_edgerock_fixed_memory", "remote", "from_qwen35")],
    "plamo2":           [("test_same_host_carries_plamo2", "remote", "from_plamo2"),
                         ("test_same_host_carries_plamo2_notremote", "reference", "from_plamo2")],
    "hymba":            [("test_same_host_carries_hymba_parallel_kv_sharing_meta_tokens", "remote", "from_hymba"),
                         ("test_same_host_carries_hymba_parallel_kv_sharing_meta_tokens_notremote", "reference", "from_hymba"),
                         ("test_same_host_carries_hymba_window_with_meta_token_sinks_notremote", "reference", "from_hymba")],
    "alice_ai":         [("test_same_host_carries_aliceai_attnres_sigmoid_router", "remote", "from_alice")],
    "mini_k3":          [("test_same_host_carries_mini_k3", "remote", "from_mini_k3")],
    "phi4flash":        [("test_same_host_carries_phi4flash_sambay", "remote", "from_phi4flash")],
    "granite_switch":   [("test_same_host_carries_granite_switch_alora", "remote", "from_granite_switch"),
                         ("test_same_host_carries_granite_switch_alora_notremote", "reference", "from_granite_switch")],
    "gigachat3_5":      [("test_same_host_carries_gigachat35", "remote", "from_gigachat35"),
                         ("test_same_host_carries_gigachat35_notremote", "reference", "from_gigachat35")],
    "lfm2":             [("test_same_host_carries_lfm2_whole_model", "stock", "from_lfm")],
    "lfm2_moe":         [("test_same_host_carries_lfm2_moe_and_lfm3", "stock", "from_lfm"),
                         ("test_same_host_carries_lfm2_moe_notremote", "stock", "from_lfm")],
    "lfm3":             [("test_same_host_carries_lfm2_moe_and_lfm3", "remote", "from_lfm"),
                         ("test_same_host_carries_lfm3_notremote", "reference", "from_lfm")],
    "lfm3_rd":          [("test_same_host_carries_lfm3_rd_unrolled", "remote", "from_lfm"),
                         ("test_same_host_carries_lfm3_rd_unrolled_notremote", "reference", "from_lfm")],
    "ceno":             [("test_same_host_carries_ceno", "remote", "from_ceno"),
                         ("test_same_host_carries_ceno_notremote", "reference", "from_ceno")],
    "glm5_next_text":   [("test_same_host_carries_glm5_next", "stock", "from_glm5_next")],
    "youtu_vl_moe":     [("test_same_host_carries_youtu_kda_mla", "remote", "from_youtu")],
    "jet_nemotron":     [("test_same_host_carries_jet_nemotron", "remote", "from_jet_nemotron")],
    "gpts3_hybrid":     [("test_same_host_carries_gpts3_gdn2_full_attnres", "remote", "from_gpts3")],
    "gptx3":            [("test_same_host_carries_gptx3", "reference", "from_gptx3")],
    "hybrid_qwen3":     [("test_same_host_carries_gka_hybrid_qwen3", "remote", "from_hybrid_qwen3")],
    "Motif":            [("test_same_host_carries_motif3_whole_model", "remote", "from_motif"),
                         ("test_same_host_carries_motif3_whole_model_notremote", "reference", "from_motif")],
    # SDAR's own code is remote (read, not run): its body is Qwen3 (-MoE) and its attention takes the caller's mask,
    # so the reference is the stock Qwen3 (-MoE) class under SDAR's block-causal mask (tests/test_host_gaps.py)
    "sdar":             [("test_same_host_carries_sdar_block_causal", "stock", "from_sdar")],
    # M11: kind "reference" = a test against a labelled transcription of the family's remote code (read as text,
    # never run), where no stock class exists
    "bailing_hybrid":   [("test_same_host_carries_ling3_stock_bailing_hybrid", "stock", "from_bailing_hybrid"),
                         ("test_same_host_carries_ling3_kda_gated_mla_swiglu_limit", "reference", "from_bailing_hybrid"),
                         ("test_same_host_carries_ling25_lightning_mla", "reference", "from_bailing_hybrid")],
    "minicpm_sala":     [("test_same_host_carries_minicpm_sala", "reference", "from_minicpm_sala")],
    "rwkv":             [("test_same_host_carries_rwkv4_whole_model", "stock", "from_rwkv")],
    "deepseek_v4":      [("test_same_host_carries_deepseek_v4", "stock", "from_deepseek_v4")],
    # V4.1: no transformers class; vs a labelled transcription of the official inference/model.py (read as text)
    "deepseek_v41":     [("test_same_host_carries_deepseek_v41", "reference", "from_deepseek_v4")],
    "deepseek_v41_text": [("test_same_host_carries_deepseek_v41", "reference", "from_deepseek_v4")],
    "sdar_moe":         [("test_same_host_carries_sdar_block_causal", "stock", "from_sdar")],
    # eleph-gaps-stream (tests/test_eleph_gaps_stream.py)
    "qwen4_exp_text":   [("test_same_host_carries_qwen4_exp_gated_residual_ngram_ple", "stock", "from_qwen4_exp")],
    "qwen4_exp":        [("test_same_host_carries_qwen4_exp_gated_residual_ngram_ple", "stock", "from_qwen4_exp")],
    "zamba2":           [("test_same_host_carries_zamba2_shared_block", "stock", "from_zamba2")],
    "zamba":            [("test_same_host_carries_zamba_v1_shared_block", "stock", "from_zamba")],
    "nanochat_gpt": [("test_same_host_carries_nanochat_value_embeddings_backout", "reference", "from_nanochat")],
    "gaggle": [("test_same_host_carries_nanochat_value_embeddings_backout", "reference", "from_nanochat")],
    "sn38-nanochrono": [("test_same_host_carries_nanochat_value_embeddings_backout", "reference", "from_nanochat")],
    "arloop": [("test_same_host_carries_arloop_shared_value_embeddings", "reference", "from_arloop")],
    "arloop_baseline": [("test_same_host_carries_arloop_shared_value_embeddings", "reference", "from_arloop")],
    "k2_horizon": [("test_same_host_carries_k2_horizon_grouped_rmsnorm", "reference", "from_k2_horizon")],
    "longcat_flash": [("test_same_host_carries_longcat_flash_scmoe", "stock", "from_longcat_flash")],
    "ouro": [("test_same_host_carries_ouro_loop_norm_early_exit", "reference", "from_ouro")],
    "hrm_text": [("test_same_host_carries_hrm_text_two_state_loops", "stock", "from_hrm_text")],
    "nanbeige": [("test_same_host_carries_nanbeige_loop_norm", "reference", "from_nanbeige")],
    "xing4_0": [("test_same_host_carries_xing_hyper_connections", "reference", "from_xing")],
    "hy_v4": [("test_same_host_carries_hy_v4_nocomb_hyper_connections", "stock", "from_hy_v4")],
    "dory": [("test_same_host_carries_dory_ngpt_loops", "reference", "from_dory")],
    "dhara_ar": [("test_same_host_carries_dhara_canon_layers", "reference", "from_dhara")],
    "crowd": [("test_same_host_carries_crowd_bigram_engram", "reference", "from_crowd")],
    "llama_moe": [("test_same_host_carries_llama_moe_mlp_router", "reference", "from_llama_moe")],
    "arctic": [("test_same_host_carries_arctic_parallel_residual_moe", "reference", "from_arctic")],
    "llama4_text": [("test_same_host_carries_llama4_input_gated_experts", "stock", "from_llama4")],
    "llama4": [("test_same_host_carries_llama4_input_gated_experts", "stock", "from_llama4")],
    # M12: non-next-token readouts / inputs (tests/test_nonlm_host.py). "reference" = a labelled transcription
    # of the remote code; the embedding / value / exit readouts on stock bodies are "stock" + a transcribed head
    "minicpm":          [("test_same_host_carries_minicpm_layerwise_reranker", "reference", "from_minicpm_layerwise")],
    "cost_wise_gemma":  [("test_same_host_carries_cost_wise_gemma", "stock", "from_cost_wise_gemma")],
    "sundial":          [("test_same_host_carries_sundial", "reference", "from_sundial")],
    # composite (multimodal-wrapper) configs whose text_config IS the tested decoder type (the snapshot's
    # text_config omits its model_type, so the wrapper's type is the one classified)
    "qwen3_5":          [("test_same_host_carries_qwen35_and_diba", "stock", "from_qwen35")],
    "qwen3_5_moe":      [("test_same_host_carries_qwen35_moe_and_memt", "stock", "from_qwen35"),
                         ("test_same_host_carries_qwen35_moe_and_memt_notremote", "stock", "from_qwen35")],
    "gemma4":           [("test_same_host_carries_tf5_family", "stock", "from_gemma4")],
    "qwen3":            [("test_same_host_carries_pseudobrain_vllm_plugin", "plugin", "from_pseudobrain")],
    "qwen3_moe":        [("test_same_host_carries_pseudobrain_vllm_plugin", "plugin", "from_pseudobrain")],
}
# eleph-gaps-attn: tests/test_attn_gaps.py ("reference": a labelled transcription of the remote code, read as text)
TESTS["qwen3"] = TESTS["qwen3"] + [("test_same_host_carries_qwen3_dros", "reference", "from_qwen3_dros")]
TESTS.update({
    "gptx2":       [("test_same_host_carries_gptx2_xsa", "reference", "from_gptx2")],
    "gpts2":       [("test_same_host_carries_gptx2_xsa", "reference", "from_gptx2")],
    "mm200":       [("test_same_host_carries_mm200_post_gate", "reference", "from_mm200")],
    "doge":        [("test_same_host_carries_doge_dynamic_mask", "stock", "from_doge")],
    "jetmoe":      [("test_same_host_carries_jetmoe_moa", "stock", "from_jetmoe")],
})
# (one from_llama4, eleph-gaps-stream's with eleph-gaps-attn's chunk mask: both tests exercise it)
TESTS["llama4_text"] = TESTS["llama4_text"] + [("test_same_host_carries_llama4_chunked_dense", "stock", "from_llama4")]

# eleph-gaps-stream: model_types that already have rows
TESTS["deepseek_v4"] = TESTS["deepseek_v4"] + [
    ("test_same_host_carries_nanowhale_comb_hyper_connections", "reference", "from_nanowhale")]
TESTS["sdar"] = TESTS["sdar"] + [("test_same_host_carries_qwendlm_self_conditioning", "reference", "from_sdar_selfcond")]
TESTS["hunyuan_v1_moe"] = TESTS.get("hunyuan_v1_moe", []) + [
    ("test_same_host_carries_hunyuan_a13b_capacity_drops", "reference", "from_hunyuan_capacity")]

# Remote-code tests rebuilt without running remote code (tests/test_notremote_*.py): "reference" = a labelled
# transcription under tests/refs/ (read as text), "stock" = a transformers / library class
NOTREMOTE_B_TESTS = [
    ("gpts3_hybrid", "test_same_host_carries_gpts3_gdn2_full_attnres_notremote", "reference", "from_gpts3"),
    ("jet_nemotron", "test_same_host_carries_jet_nemotron_notremote", "reference", "from_jet_nemotron"),
    ("mini_k3", "test_same_host_carries_mini_k3_notremote", "reference", "from_mini_k3"),
    ("hybrid_qwen3", "test_same_host_carries_gka_hybrid_qwen3_notremote", "reference", "from_hybrid_qwen3"),
    ("youtu_vl_moe", "test_same_host_carries_youtu_kda_mla_notremote", "reference", "from_youtu"),
    ("phi4flash", "test_same_host_carries_phi4flash_sambay_notremote", "reference", "from_phi4flash"),
    ("alice_ai", "test_same_host_carries_aliceai_attnres_sigmoid_router_notremote", "reference", "from_alice"),
    ("allgdn", "test_same_host_carries_allgdn_engram_notremote", "reference", "from_qwen35"),
    ("edgerock", "test_same_host_carries_edgerock_fixed_memory_notremote", "reference", "from_qwen35"),
    ("qwen3", "test_same_host_carries_pseudobrain_vllm_plugin_notremote", "reference", "from_pseudobrain"),
    ("qwen3_moe", "test_same_host_carries_pseudobrain_vllm_plugin_notremote", "reference", "from_pseudobrain"),
]
for _mt, _test, _kind, _loader in NOTREMOTE_B_TESTS:
    TESTS[_mt] = TESTS.get(_mt, []) + [(_test, _kind, _loader)]


def _stock_tests():
    """TESTS rows read off tests/test_stock_loaders.py: every CASES entry (a stock transformers config
    class -> its model_type) and every REAL checkpoint (its model_type)."""
    import importlib
    import transformers
    mod = importlib.import_module("tests.test_stock_loaders")
    rows = defaultdict(list)
    for case, (cfg_name, _, loader, _, opt) in mod.CASES.items():
        cls = getattr(transformers, cfg_name)
        mts = {cls.model_type} | ({cls().text_config.model_type} if opt.get("vl") else set())   # VL: + its decoder
        for mt in mts:
            rows[mt].append(("test_same_host_carries_stock_family", "stock", loader))
    for repo, (loader, mt, _) in mod.REAL.items():
        rows[mt].append(("test_real_checkpoints", "checkpoint", loader))
    for case, (_, _, loader, _, _) in mod.TF5_CASES.items():
        rows[mod.TF5_TYPES[case]].append(("test_same_host_carries_tf5_family", "stock", loader))
    return rows


# M12 readouts on model_types that already have TESTS rows (the body's): the readout's own test
READOUT_TESTS = {
    "qwen3": [("test_same_host_carries_sentence_embeddings", "stock", "from_embedding")],
    "qwen2": [("test_same_host_carries_sentence_embeddings", "stock", "from_embedding")],
    "gemma": [("test_same_host_carries_grm_value_head", "stock", "from_grm")],
}


def remote_code_tests():
    """The test functions that execute code downloaded from the Hub / GitHub (tests/conftest.py's
    REMOTE_CODE_TESTS, skipped unless TEF_ALLOW_REMOTE_CODE=1): their rows do not count by default."""
    from tests.conftest import REMOTE_CODE_TESTS
    return set().union(*REMOTE_CODE_TESTS.values())


def all_tests(include_remote=False):
    """model_type -> its test rows. By default without the rows of remote-code tests (remote_code_tests):
    a model_type counts as tested only through tests that run no downloaded code."""
    rows = defaultdict(list, {k: list(v) for k, v in TESTS.items()})
    for mt, r in READOUT_TESTS.items():
        rows[mt] += r
    for mt, r in _stock_tests().items():
        rows[mt] += r
    if not include_remote:
        remote = remote_code_tests()
        rows = defaultdict(list, {mt: [t for t in r if t[0] not in remote] for mt, r in rows.items()})
    return rows


# --- M12: models whose output is not a next-token distribution, or whose input is not tokens --------------
# kind: "score" (a narrow head, early exits: rerankers), "value" (a reward MLP head), "embedding" (the pooled
# hidden state: sentence-transformers Pooling -> [Dense] -> [Normalize]), "flow" (continuous patch input, a
# flow-matching head network). Read off each repo's config / modules.json / 1_Pooling / 2_Dense configs and
# its remote code (as text). Models that SCORE THROUGH THEIR LM HEAD (Qwen3-Reranker, mxbai-rerank-v2,
# ctxl-rerank, bge-reranker-v2-gemma: sentence-transformers LogitScore of yes / no vocabulary logits;
# Nemotron-70B-Reward, math-shepherd PRM, judges: a vocabulary logit) are LMs: their logits are the output.
_ST = "sentence-transformers modules.json + 1_Pooling/config.json"
_EMB = dict(kind="embedding", loader="from_embedding")
READOUTS = {
    # last-token pooling + Normalize, causal stock body
    "Qwen/Qwen3-Embedding-0.6B":  dict(_EMB, pool="last", l2=True, causal=True, src=_ST),
    "Qwen/Qwen3-Embedding-4B":    dict(_EMB, pool="last", l2=True, causal=True, src=_ST),
    "Qwen/Qwen3-Embedding-8B":    dict(_EMB, pool="last", l2=True, causal=True, src=_ST),
    "boboliu/Qwen3-Embedding-4B-W4A16-G128":       dict(_EMB, pool="last", l2=True, causal=True, src="base_model Qwen3-Embedding-4B"),
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ": dict(_EMB, pool="last", l2=True, causal=True, src="base_model Qwen3-Embedding-0.6B"),
    "mlx-community/Qwen3-Embedding-4B-4bit-DWQ":   dict(_EMB, pool="last", l2=True, causal=True, src="base_model Qwen3-Embedding-4B"),
    "mlx-community/Qwen3-Embedding-8B-4bit-DWQ":   dict(_EMB, pool="last", l2=True, causal=True, src="base_model Qwen3-Embedding-8B"),
    "mlx-community/Qwen3-Embedding-8B-mxfp8":      dict(_EMB, pool="last", l2=True, causal=True, src=_ST),
    "jinaai/jina-code-embeddings-0.5b": dict(_EMB, pool="last", l2=True, causal=True, src=_ST),
    "jinaai/jina-code-embeddings-1.5b": dict(_EMB, pool="last", l2=True, causal=True, src=_ST),
    # bidirectional bodies (remote modeling_qwen.py Qwen2Model, forward(is_causal=False) by default)
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": dict(_EMB, pool="last", l2=True, causal=False, src=_ST + " + config is_causal false"),
    "Alibaba-NLP/gte-Qwen2-7B-instruct":   dict(_EMB, pool="last", l2=True, causal=False, src=_ST + " + config is_causal false"),
    "NovaSearch/stella_en_1.5B_v5":  dict(_EMB, pool="mean", l2=False, causal=False, dense=(1536, 1024, True),
                                          src=_ST + " + 2_Dense_1024 + modeling_qwen.py"),
    "sdadas/stella-pl-retrieval-8k": dict(_EMB, pool="mean", l2=False, causal=False, dense=(1536, 1024, True),
                                          src=_ST + " + 2_Dense + modeling_qwen.py"),
    # Qwen3BidirectionalModel: a per-token Linear(hidden, num_labels) (no bias), mean pooling, Normalize
    "MongoDB/voyage-4-nano": dict(_EMB, pool="mean", l2=True, causal=False, dense=(1024, 2048, False),
                                  src=_ST + " + modeling_qwen3_bidirectional.py"),
    # VLM2Vec (TIGER-Lab, its own GitHub code): the last real token's hidden state, normalised
    "TIGER-Lab/VLM2Vec-Full": dict(_EMB, pool="last", l2=True, causal=True, src="VLM2Vec model card / code"),
}


def readout_of(cfg, name=None):
    """The non-next-token readout of a snapshot model (a READOUTS / rule entry), or None for an LM."""
    if name in READOUTS:
        return READOUTS[name]
    c = cfg if isinstance(cfg, dict) else {}
    mt, a = c.get("model_type"), _arch(c)
    if mt == "minicpm" and ("head_type" in c or a == "LayerWiseMiniCPMForCausalLM") and \
            str(c.get("head_type", "simple")).lower() not in ("raw", "complex"):
        return dict(kind="score", loader="from_minicpm_layerwise", pool="none", exits=True, src="modeling_minicpm_reranker.py")
    if mt == "cost_wise_gemma" and c.get("layer_wise"):
        return dict(kind="score", loader="from_cost_wise_gemma", pool="none", exits=True, src="gemma_model.py")
    if c.get("vhead_layer_type") or (c.get("auto_map") or {}).get("AutoModel") == "model.AutoModelForCausalLMWithValueHead":
        return dict(kind="value", loader="from_grm", pool="last", src="GRM model.py ValueHead")
    if mt == "sundial" or a == "SundialForPrediction":
        return dict(kind="flow", loader="from_sundial", pool="last", input="patches",
                    src="modeling_sundial.py + flow_loss.py")
    return None


def _readout_check(c):
    return None if c.get("head_type") is not None or _arch(c) == "LayerWiseMiniCPMForCausalLM" else \
        "a MiniCPM without head_type: the LM (modeling_minicpm.py) has no loader"


# the readout loaders (dispatched by readout_of, before LOADERS); the body loader must accept the config too
READOUT_LOADERS = {
    "from_minicpm_layerwise": Loader("from_minicpm_layerwise", ("minicpm",), ("LayerWiseMiniCPMForCausalLM",),
                                     _all(lambda c: "rope_scaling" if c.get("rope_scaling") else None, _readout_check)),
    "from_cost_wise_gemma": Loader("from_cost_wise_gemma", ("cost_wise_gemma",), ("CostWiseGemmaForCausalLM",)),
    "from_sundial": Loader("from_sundial", ("sundial",), ("SundialForPrediction",)),
}
_BODY = {"from_embedding": "from_llama", "from_grm": None}   # None: from_gemma or from_llama by model_type


def readout_loader_for(cfg, ro):
    """(loader, None) when the readout's loader (and its body loader) accepts cfg, else (None, why)."""
    ld = READOUT_LOADERS.get(ro["loader"])
    if ld is not None:
        r = ld.accepts(cfg)
        return (ld.name, None) if r is None else (None, f"{ld.name}: {r}")
    body, why = loader_for(cfg)
    want = _BODY.get(ro["loader"])
    ok = body == want if want else body in ("from_gemma", "from_llama")
    return (ro["loader"], None) if ok else (None, f"{ro['loader']}: body {body or why}")


def loader_for(cfg):
    """(loader name, None) if some loader accepts cfg, else (None, the reasons)."""
    why = []
    for ld in LOADERS:
        r = ld.accepts(cfg)
        if r is None:
            return ld.name, None
        if cfg.get("model_type") in ld.types:
            why.append(f"{ld.name}: {r}")
    return None, "; ".join(why) or "no loader for this model_type"


_CACHE = {}


def _ALL_TESTS(include_remote=False):
    key = ("t", include_remote)
    if key not in _CACHE:
        _CACHE[key] = all_tests(include_remote)
    return _CACHE[key]


def classify_model(cfg, model=None, include_remote=False):
    """-> (loader or None, tested: bool, checkpoint: bool, reason). `model` (the repo id) picks the readout of
    a non-next-token model (readout_of): its loader is the readout's, tested against that readout's code."""
    from graph.model_graph import _lm_config
    c = _lm_config(cfg)
    ro = readout_of(c, model) or readout_of(cfg, model)
    name, why = readout_loader_for(c, ro) if ro else loader_for(c)
    if name is None:
        return None, False, False, why
    rows = [t for t in _ALL_TESTS(include_remote).get(c.get("model_type"), []) if t[2] == name]
    return name, bool(rows), any(k == "checkpoint" for _, k, _ in rows), None


def test_functions():
    """Every test function name defined under tests/ -> the source of its body."""
    out = {}
    tdir = os.path.join(ROOT, "tests")
    for fn in sorted(os.listdir(tdir)):
        if fn.startswith("test_") and fn.endswith(".py"):
            src = open(os.path.join(tdir, fn)).read()
            lines = src.split("\n")
            for n in ast.parse(src).body:
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
                    out[n.name] = "\n".join(lines[n.lineno - 1:n.end_lineno])
    return out


def check_tests():
    """Each TESTS row names an existing test that mentions its loader (or a dispatch table that
    maps to it). Returns the list of problems (empty = consistent)."""
    import elephantine.universal as U
    fns, bad = test_functions(), []
    tdir = os.path.join(ROOT, "tests")
    src_all = "\n".join(open(os.path.join(tdir, f)).read() for f in os.listdir(tdir) if f.endswith(".py"))
    for mt, rows in all_tests(include_remote=True).items():
        for test, kind, loader in rows:
            if not hasattr(U, loader):
                bad.append(f"{mt}: {loader} is not in elephantine.universal")
            if test not in fns:
                bad.append(f"{mt}: {test} not found under tests/")
            elif not re.search(rf"\b{loader}\b", fns[test]) and not re.search(rf"[\"']{loader}[\"']", src_all):
                bad.append(f"{mt}: {test} does not exercise {loader}")
    return bad


def inventory(models=None, include_remote=False):
    from graph.minimal_emulator import load
    from graph.model_graph import _lm_config
    models = load() if models is None else models
    rows = defaultdict(Counter)
    why = defaultdict(Counter)
    for m in models:
        mt = str(_lm_config(m["cfg"]).get("model_type"))
        ld, tested, ckpt, reason = classify_model(m["cfg"], m.get("name"), include_remote)
        r = rows[mt]
        r["models"] += 1
        r["loader"] += ld is not None
        r["tested"] += tested
        r["checkpoint"] += ckpt
        if ld is None:
            why[mt][reason] += 1
        elif not tested:
            why[mt][f"{ld}: no real-code test of this model_type"] += 1
    return rows, why


TF5_MODELS = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0/transformers/models")


def uncovered_by_category(rows, why):
    """The models without a tested loader, by reason: remote code (not in transformers 4.57 / 5.18:
    it is not run unless TEF_ALLOW_REMOTE_CODE=1), a stock model_type whose `architectures` names its own
    class (speculative-decoding drafts, Eagle-3 heads, custom heads: remote code too), or a stock
    transformers family with no host loader yet."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C4
    out = Counter()
    for mt, r in rows.items():
        left = r["models"] - r["tested"]
        if not left:
            continue
        for reason, k in why[mt].items():
            if "architecture" in reason:
                out["stock model_type, own (remote-code) architecture class"] += k
            elif mt in C4 or os.path.isdir(os.path.join(TF5_MODELS, mt)) or mt in ("gemma3n_text", "qwen4_exp_text"):
                out["transformers-native family without a host loader (see --why)"] += k
            elif mt == "None":
                out["no model_type in its config (remote code)"] += k
            else:
                out["remote code only (not in transformers 4.57 / 5.18)"] += k
    return out


def totals(rows):
    t = Counter()
    for r in rows.values():
        t.update(r)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="list every model_type (default: top 60 + covered)")
    ap.add_argument("--why", action="store_true", help="print why uncovered models are uncovered")
    ap.add_argument("--include-remote", action="store_true",
                    help="also credit tests that run remote code (tests/conftest.py REMOTE_CODE_TESTS)")
    args = ap.parse_args()
    bad = check_tests()
    for b in bad:
        print("TESTS table problem:", b)
    rows, why = inventory(include_remote=args.include_remote)
    t = totals(rows)
    n = t["models"]
    print(f"{n} models, {len(rows)} model_types")
    print(f"  accepted by a from_* loader:            {t['loader']:5d}  ({100 * t['loader'] / n:.1f}%)")
    print(f"  ... with a real-code test of its type:  {t['tested']:5d}  ({100 * t['tested'] / n:.1f}%)")
    print(f"  ... incl. a real released checkpoint:   {t['checkpoint']:5d}  ({100 * t['checkpoint'] / n:.1f}%)")
    print(f"  model_types with a tested loader: {sum(1 for r in rows.values() if r['tested'])}")
    print("  not loaded-and-tested, by why:")
    for cat, k in uncovered_by_category(rows, why).most_common():
        print(f"    {k:5d}  {cat}")
    print(f"\n  {'model_type':28s} models loader tested ckpt")
    order = sorted(rows.items(), key=lambda kv: (-kv[1]["models"], kv[0]))
    for i, (mt, r) in enumerate(order):
        if args.all or i < 60 or r["loader"]:
            print(f"  {mt:28s} {r['models']:6d} {r['loader']:6d} {r['tested']:6d} {r['checkpoint']:4d}")
            if args.why:
                for reason, k in why[mt].most_common(3):
                    print(f"      {k:4d}  {reason}")


if __name__ == "__main__":
    main()
