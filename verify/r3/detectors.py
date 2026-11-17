"""Round-3 detectors: one standalone detect_<finding>(cfg, model_type) per spec in verify/r3/specs/.

Written in batches, one section per spec batch below. cfg is the raw snapshot
config, model_type = r3util.model_type_of(cfg); a falsy return means "not flagged", anything else is the new slot value /
FEATURES key / dims value the spec names. The detectors do not import the relation's module; tests/test_r3_detectors.py
checks each against its fixtures (verify/r3/fixtures/*.json) on graph/hub_10k.json.
"""
import json  # noqa: F401
import math  # noqa: F401
import re  # noqa: F401

from r3util import (  # noqa: F401
    archs, auto_map, fnum, get, has_lm_dims, layer_count, layer_types, lm_of, model_type_of, num, outer_type,
    rope_params, truthy_flag,
)


# ==================== batch a1_typetables ====================

# --- out_gate_by_code_unlisted ------------------------------------------------------------------------------------
# attention output gate o * sigmoid(W_g x) built unconditionally by the type's code; value = granularity
# ("head": one gate per head, W_g: hidden -> num_heads; "elem": one per channel, heads * head_dim) + "_bias" when
# the gate projection carries a bias
OUT_GATE_BY_CODE = {
    "lfm3": "head", "lfm3_rd": "head",                  # gate_proj: hidden -> num_attention_heads, bias=False
    "grug_moe": "head",                                 # attn_gate: hidden -> num_heads, 2 * sigmoid
    "dots3_note": "head", "dots3_note_text": "head",    # g_proj: hidden -> num_heads
    "qwen4_exp": "elem", "qwen4_exp_text": "elem",      # q_proj 2x wide -> (query, gate)
    "alice_ai": "elem", "gpts3_hybrid": "elem", "gptx3": "elem",   # q(_gate)_proj 2x wide
    "hy_v4": "elem", "muse_glimmer": "elem", "muse_glimmer_text": "elem", "dory": "elem",
    "klm": "elem", "needle": "elem",
    "beam": "elem_bias",                                # gate_proj + per-head gate_bias
}


def detect_out_gate_by_code_unlisted(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "ethoric":                         # EthoricAttention: attn_gate selects the gate, bias=True
        g = str(lm.get("attn_gate") or "").lower()
        return {"headwise": "head_bias", "elementwise": "elem_bias"}.get(g)
    if model_type == "hrm_text":                        # gate_proj bias = config.attention_bias
        return "elem_bias" if lm.get("attention_bias") is True else "elem"
    return OUT_GATE_BY_CODE.get(model_type)


# --- qknorm_by_code_unlisted --------------------------------------------------------------------------------------
# per-head q / k norms built with no config switch; value = the QKNORM_VARIANTS value of the code
QKNORM_HEAD_BY_CODE = {
    "alice_ai", "bananamind2_moe", "beam", "bitsy_parallel", "cyrah", "edgerock", "gemma3_moe", "gpts3_hybrid",
    "gptx3", "hulumed_qwen3", "iquest_q1", "isaac", "klm", "kolibri1", "lfm3", "lfm3_rd", "memt_qwen3_5_moe",
    "monkeyocrv2", "needle", "nexus", "plamo3", "qwen3loop", "viuai"}
QKNORM_OTHER_BY_CODE = {
    "domynedge": "head_qfree",                          # q weight-free, k weighted (per head)
    "muse_glimmer": "head_free_gain", "muse_glimmer_text": "head_free_gain",   # weight-free, q * qk_scale_factor
    "grug_moe": "head_free_gain",                       # weight-free, q * qk_mult
    "dory": "ngpt",                                     # weight-free L2, per-channel q gain after RoPE
}


def detect_qknorm_by_code_unlisted(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in QKNORM_HEAD_BY_CODE:
        return "head"
    if model_type in QKNORM_OTHER_BY_CODE:
        return QKNORM_OTHER_BY_CODE[model_type]
    if model_type == "ethoric":                         # q_norm / k_norm only when qk_norm == "per_head"
        return "head" if str(lm.get("qk_norm") or "").lower() == "per_head" else None
    if model_type == "llada2_moe":                      # preview: unconditional; others: use_qk_norm (class default True)
        return "head" if lm.get("use_qk_norm", True) is not False else None
    return None


# --- qknorm_always_remote_lacks -----------------------------------------------------------------------------------
def detect_qknorm_always_remote_lacks(cfg, model_type):
    """Doge configs that run the remote modeling_doge.py (no q_norm / k_norm) instead of transformers' Doge."""
    if model_type == "doge" and "modeling_doge" in auto_map(cfg):
        return "no"
    return None


# --- qknorm_fixed_gain --------------------------------------------------------------------------------------------
# weight-free per-head q / k norm followed by a constant multiplier from the config: (key, applied to q and k?)
QK_GAIN_KEYS = {"nanochat_gpt": ("qk_sharpen_scale", None, True),
                "gaggle": ("qk_scale", None, True),
                "sn38-nanochrono": ("attention_multiplier", None, True),
                "grug_moe": ("qk_mult", 1.5703274004183786, False),
                "muse_glimmer": ("qk_scale_factor", 3.87, False), "muse_glimmer_text": ("qk_scale_factor", 3.87, False)}


def qk_gain(cfg, model_type):
    """The logit multiplier (over 1 / sqrt(head_dim)) of a weight-free qk-norm with a constant gain, else None."""
    if model_type not in QK_GAIN_KEYS:
        return None
    key, default, both = QK_GAIN_KEYS[model_type]
    s = fnum(lm_of(cfg).get(key), default)
    if s is None:
        return None
    g = s * s if both else s
    return round(g, 4) if abs(g - 1.0) > 1e-9 else None


def detect_qknorm_fixed_gain(cfg, model_type):
    return "head_free_gain" if qk_gain(cfg, model_type) else None


# --- bias_qkv_by_code_unlisted ------------------------------------------------------------------------------------
# renamed types whose LM is transformers' Qwen2Model (q / k / v bias=True hard-coded) or own code with bias=True
BIAS_QKV_BY_CODE = {"arkasr", "cambrian_qwen", "hulumed_qwen2", "llava-qwen2", "llava_qwen2", "meditor",
                    "spatiallm_qwen", "step_audio_2", "sundial", "videollama3_qwen2"}


def detect_bias_qkv_by_code_unlisted(cfg, model_type):
    return True if model_type in BIAS_QKV_BY_CODE else None


# --- bias_qkv_table_wrong -----------------------------------------------------------------------------------------
def detect_bias_qkv_table_wrong(cfg, model_type):
    """True = the type's code has NO q / k / v bias although BIAS_QKV_TYPES lists it (query_key_value bias=False)."""
    return True if model_type == "gpt_neox_japanese" else None


# --- bias_out_by_code_unlisted ------------------------------------------------------------------------------------
BIAS_OUT_BY_CODE = {"recurrent_gemma",   # o_proj bias=True, MLP gate / up / down bias=True
                    "jetmoe",            # MoE / MoA output bias parameter
                    "nae1"}              # nn.LayerNorm (affine beta) stream norms


def detect_bias_out_by_code_unlisted(cfg, model_type):
    return True if model_type in BIAS_OUT_BY_CODE else None


# --- out_gate_key_unread ------------------------------------------------------------------------------------------
def detect_out_gate_key_unread(cfg, model_type):
    """ERNIE-5.5: wo_b(o * sigmoid(o_gate(q latent))): an elementwise gate read from the normed q latent."""
    lm = lm_of(cfg)
    if model_type == "ernie5_5_moe" and lm.get("use_gated_attn", True) is not False:
        return "elem_qlat"
    return None


# --- parallel_by_code_unlisted ------------------------------------------------------------------------------------
def detect_parallel_by_code_unlisted(cfg, model_type):
    return "parallel" if model_type == "bitsy_parallel" else None


# --- nanochat_port_unlisted ---------------------------------------------------------------------------------------
NANOCHAT_PORTS = {"gaggle", "pit", "sn38-nanochrono"}


def detect_nanochat_port_unlisted(cfg, model_type):
    """nanochat ports: up -> relu^2 -> down (no gate), weight-free per-head q / k RMSNorm."""
    return "ffn=vanilla,act=relu2,qknorm=head_free" if model_type in NANOCHAT_PORTS else None


# --- glm5_next_mla_no_head_qknorm ---------------------------------------------------------------------------------
def detect_glm5_next_mla_no_head_qknorm(cfg, model_type):
    """GLM-5-Next softmax layers are MLA with only the q_a / kv_a latent norms: qknorm no."""
    return "no" if model_type in ("glm5_next", "glm5_next_text") else None


# --- dsv41_q_head_norm_absent -------------------------------------------------------------------------------------
def detect_dsv41_q_head_norm_absent(cfg, model_type):
    """DeepSeek-V4.1 drops V4's weight-free per-head q norm: only the K = V kv_norm remains."""
    return "dsv4_kv" if model_type in ("deepseek_v41", "deepseek_v41_text") else None


# ==================== batch a2_dims ====================

def _hidden(lm):
    return num(lm.get("hidden_size") or lm.get("n_embd") or lm.get("d_model") or lm.get("hidden_dim"))


def _heads(lm):
    return num(lm.get("num_attention_heads") or lm.get("n_head"))


# --- head_dim_key_unread --------------------------------------------------------------------------------------------
NEMOTRON_H_LIKE = ("nemotron_h", "nemotron_h_puzzle", "ceno")


def detect_head_dim_key_unread(cfg, model_type):
    """The attention head width the family's code builds from a key other than head_dim (or a derived default), when
    it differs from what a plain read gives (head_dim key, else hidden // heads). Returns that head width."""
    lm = lm_of(cfg)
    h, nh = _hidden(lm), _heads(lm)
    if not nh:
        return None
    plain = num(lm.get("head_dim")) or h // nh
    hd = 0
    if model_type in NEMOTRON_H_LIKE:
        # remote NemotronHAttention / CENO: attention_head_dim if not None else hidden // heads; tf5: head_dim
        # (class default 128). A config with neither key gets 128 from both code paths (nemotron_h only).
        if lm.get("head_dim") is None:
            if lm.get("attention_head_dim") is not None:
                hd = num(lm["attention_head_dim"])
            elif model_type != "ceno" and "attention_head_dim" not in lm:
                hd = 128
    elif model_type in ("zamba", "zamba2"):
        hd = num(lm.get("attention_head_dim")) or (2 * h // nh)      # attribute_map head_dim -> attention_head_dim
    elif model_type == "jetmoe":
        hd = num(lm.get("kv_channels")) or 128                       # attribute_map head_dim -> kv_channels
    elif model_type in ("plamo2", "plamo2vl"):
        hd = num(lm.get("hidden_size_per_head")) or 128              # Attention: head_dim = hidden_size_per_head
    return hd if hd and hd != plain else None


# --- tie_default_unread ---------------------------------------------------------------------------------------------
# remote config classes whose tie_word_embeddings default is True (a saved config omitting the key is tied)
TIE_DEFAULT_TRUE = frozenset({"minicpm", "inferno", "cyrah", "nexus", "kanana2_tiny", "phi3small", "tinymixtral",
                              "bitsy_parallel", "gemma3_moe"})
# lm_head = the embedding table in code, whatever the config says
TIED_BY_CODE = frozenset({"cfrd", "mima", "inferno"})


def _tie_type(cfg, model_type):
    if model_type:
        return model_type
    if any(a == "MiniCPMForCausalLM" for a in archs(cfg)) and "configuration_minicpm" in auto_map(cfg):
        return "minicpm"                                             # MiniCPM-1B / 2B / 4-0.5B, BitCPM: no model_type
    return ""


def detect_tie_default_unread(cfg, model_type):
    lm = lm_of(cfg)
    mt = _tie_type(cfg, model_type)
    if mt in TIED_BY_CODE:
        return True
    if mt in TIE_DEFAULT_TRUE and not any(k in lm or k in (cfg or {})
                                          for k in ("tie_word_embeddings", "tie_embedding", "weight_tying")):
        if mt == "minicpm" and not any(a == "MiniCPMForCausalLM" for a in archs(cfg)):
            return None                                              # the layerwise reranker is not this class
        return True
    return None


# --- config_default_unread ------------------------------------------------------------------------------------------
def detect_config_default_unread(cfg, model_type):
    """(key, class default) pairs the code runs with because the saved config omits the key, where the default
    changes the classification."""
    lm = lm_of(cfg)
    out = []
    if model_type == "llada2_moe" and "use_qk_norm" not in lm:
        out.append(["use_qk_norm", True])              # LLaDA2MoeConfig default True (preview: q / k norm in code)
    if model_type in ("muse_glimmer", "muse_glimmer_text") and "final_logit_softcapping" not in lm:
        out.append(["final_logit_softcapping", 20.0])  # MuseGlimmerTextConfig default, applied unconditionally
    if (model_type == "exaone_moe" and "sliding_window" not in lm and isinstance(lm.get("layer_types"), list)
            and any("sliding" in str(x) for x in lm["layer_types"])):
        out.append(["sliding_window", 4096])           # ExaoneMoeConfig default: full layers NoPE, window 4096
    return out or None


# --- ffn_width_keys_unread ------------------------------------------------------------------------------------------
FLA_MLP_TYPES = frozenset({"gla", "delta_net", "hgrn_bit"})        # fla GatedMLP(hidden_ratio, intermediate_size=None)


def detect_ffn_width_keys_unread(cfg, model_type):
    """(dims field, width) the code builds from a key / formula dims_of does not read: 'inter' = the dense GLU / MLP
    width, 'ew' = the routed expert width, 'dense_inter' = the dense layers' width beside MoE layers."""
    lm = lm_of(cfg)
    h = _hidden(lm)
    inter = lm.get("intermediate_size")
    if model_type in ("minimax_m3_vl", "minimax_m3_vl_text", "iquest_q1") and num(lm.get("dense_intermediate_size")):
        d = num(lm["dense_intermediate_size"])
        return ["dense_inter", d] if d != num(inter) else None
    if model_type in ("recurrent_gemma", "recurrentgemma") and num(inter):
        return ["inter", num(inter) // 2]                            # RecurrentGemmaMlp: intermediate_size // 2
    if model_type == "rwkv5" and inter is None and h:
        return ["inter", int((h * 3.5) // 32 * 32)]                  # RwkvFeedForward default width
    if model_type == "mima" and num(lm.get("ffn_hidden")):
        return ["inter", num(lm["ffn_hidden"])]
    if model_type in ("tinymixtral", "bananamind2_moe") and num(lm.get("expert_intermediate_size")):
        return ["ew", num(lm["expert_intermediate_size"])]
    if model_type == "ivme" and fnum(lm.get("ffn_mult")) and h:
        w = int(h * fnum(lm["ffn_mult"]) * 2 / 3)
        return ["inter", (w + 7) // 8 * 8]                           # IvmeSwiGLU round_up_8
    if model_type == "zaya1_vl" and lm.get("gated_linear_unit") and num(lm.get("ffn_hidden_size")):
        return ["ew", num(lm["ffn_hidden_size"]) // 2]               # MLP: fc1 -> ffn_hidden_size, GLU halves
    if model_type in FLA_MLP_TYPES and inter is None and h:
        w = int(h * (fnum(lm.get("hidden_ratio")) or 4) * 2 / 3)
        return ["inter", 256 * ((w + 255) // 256)]
    return None


# --- n_kv_heads_key_unread ------------------------------------------------------------------------------------------
_KV_STD_KEYS = ("num_key_value_heads", "num_kv_heads", "n_head_kv", "num_attention_groups",
                "num_key_value_heads_per_layer", "multi_query")


def detect_n_kv_heads_key_unread(cfg, model_type):
    """The KV-head count under an n_kv_heads / n_kv_head key (no standard key), when it differs from the query
    heads (a placed config: heads under num_attention_heads / n_head)."""
    lm = lm_of(cfg)
    nh = _heads(lm)
    if not nh or any(lm.get(k) not in (None, False) for k in _KV_STD_KEYS):
        return None
    kv = num(lm.get("n_kv_heads") or lm.get("n_kv_head"))
    return kv if kv and kv != nh else None


# --- per_layer_head_shapes ------------------------------------------------------------------------------------------
def detect_per_layer_head_shapes(cfg, model_type):
    """The distinct per-layer attention shapes (kv heads, q heads per kv head, head dim), sorted, when the layers
    differ: per-layer q-head lists (Laguna), per-layer KV-head lists (DeciLM), per-layer head sizes (Domyn-Edge),
    the sliding layers' own swa_* shape (MiMo-V2 / Naive-N0.5)."""
    lm = lm_of(cfg)
    h, nh = _hidden(lm), _heads(lm)
    if not nh or lm.get("kv_lora_rank"):
        return None
    nkv = num(lm.get("num_key_value_heads")) or nh
    hd = num(lm.get("head_dim")) or h // nh
    shapes = set()

    def add(q, k, d):
        if q and k and d:
            shapes.add((int(k), max(1, int(q) // int(k)), int(d)))
    qpl, kpl, hs = lm.get("num_attention_heads_per_layer"), lm.get("num_key_value_heads_per_layer"), lm.get("head_sizes")
    if isinstance(qpl, list) and qpl:
        for q in qpl:
            add(num(q), nkv, hd)
    elif isinstance(kpl, list) and kpl:
        for k in kpl:
            add(nh, num(k), hd)
    elif isinstance(hs, list) and hs:
        for d in hs:
            add(nh, nkv, num(d))
    elif lm.get("swa_head_dim") or lm.get("swa_num_key_value_heads"):
        add(nh, nkv, hd)
        add(num(lm.get("swa_num_attention_heads")) or nh, num(lm.get("swa_num_key_value_heads")) or nkv,
            num(lm.get("swa_head_dim")) or hd)
    return [list(x) for x in sorted(shapes)] if len(shapes) > 1 else None


# --- v_head_dim_narrow ----------------------------------------------------------------------------------------------
V_WIDTH_TYPES = frozenset({"mimo_v2", "mimo_v2_flash", "naive_n05_flash", "needle", "hymba"})


def detect_v_head_dim_narrow(cfg, model_type):
    """(q / k head dim, v head dim) of a non-MLA attention whose V heads are not as wide as its Q / K heads."""
    lm = lm_of(cfg)
    if model_type not in V_WIDTH_TYPES or lm.get("kv_lora_rank"):
        return None
    h, nh = _hidden(lm), _heads(lm)
    if not nh:
        return None
    qk = num(lm.get("qk_head_dim")) or num(lm.get("head_dim")) or h // nh
    v = num(lm.get("value_head_dim")) or num(lm.get("v_head_dim")) or qk
    return [qk, v] if v != qk else None


# --- jet_dims_unread ------------------------------------------------------------------------------------------------
def detect_jet_dims_unread(cfg, model_type):
    """(rc_v, rc_rank) of a JetBlock: the value head width head_dim * expand_v and the dynamic-conv generator rank
    value_dim // dconv_generator_reduction."""
    lm = lm_of(cfg)
    jet = (lm.get("efficient_attention_config") or {}).get("jet") if model_type == "jet_nemotron" else None
    if not isinstance(jet, dict):
        return None
    hd, H = num(jet.get("head_dim")) or 256, num(jet.get("num_heads")) or 6
    ev = fnum(jet.get("expand_v"), 2.0)
    red = num(jet.get("dconv_generator_reduction"))
    value_dim = int(H * hd * ev)
    return [int(hd * ev), value_dim // red if red else value_dim]


# --- mm200_core_keys_unread -----------------------------------------------------------------------------------------
def detect_mm200_core_keys_unread(cfg, model_type):
    """(rotated dims per head, gate variant) of the native MM200 core: rot_dim = max(2, int(head_dim *
    partial_rope_frac) // 2 * 2), and 'out_gate_post' when use_gated_attention multiplies the o_proj OUTPUT."""
    if model_type != "mm200":
        return None
    lm = lm_of(cfg)
    h, nh = _hidden(lm), _heads(lm)
    hd = num(lm.get("head_dim")) or (h // nh if nh else 0)
    frac = fnum(lm.get("partial_rope_frac"), 0.25)                   # MM200Config default 0.25
    rot = max(2, int(hd * frac) // 2 * 2) if hd else 0
    gate = "out_gate_post" if lm.get("use_gated_attention", True) else ""
    if rot >= hd and not gate:
        return None
    return [rot, gate]


# --- switch_cache_layers_counted ------------------------------------------------------------------------------------
SWITCH_CACHE_LAYERS = 2


def detect_switch_cache_layers_counted(cfg, model_type):
    """The decoder layers Granite-Switch builds: num_hidden_layers - 2 cache slots when adapters are present."""
    lm = lm_of(cfg)
    if model_type != "granite_switch" or num(lm.get("num_adapters")) <= 0:
        return None
    n = num(lm.get("num_hidden_layers"))
    return n - SWITCH_CACHE_LAYERS if n > SWITCH_CACHE_LAYERS else None


# --- ungated_ffn_type_unlisted --------------------------------------------------------------------------------------
UNGATED_NEW = frozenset({"parler_tts_decoder", "musicgen", "musicgen_decoder", "musicgen_melody",
                         "musicgen_melody_decoder"})


def detect_ungated_ffn_type_unlisted(cfg, model_type):
    """'vanilla': the decoder FFN is fc2(act(fc1 x)) (ParlerTTS / MusicGen decoder layers)."""
    return "vanilla" if model_type in UNGATED_NEW else None


# --- norm_fallback_rmsnorm_wrong ------------------------------------------------------------------------------------
NORM_NEW = {"parler_tts_decoder": "LayerNorm", "musicgen": "LayerNorm", "musicgen_decoder": "LayerNorm",
            "musicgen_melody": "LayerNorm", "musicgen_melody_decoder": "LayerNorm", "sundial": "LayerNorm"}


def detect_norm_fallback_rmsnorm_wrong(cfg, model_type):
    """The norm class the code builds (nn.LayerNorm, affine with bias) for a type whose config has no eps key."""
    return NORM_NEW.get(model_type)


# ==================== batch a3_placement ====================

# --- not_a_causal_lm ------------------------------------------------------------------------------------------------
def detect_not_a_causal_lm(cfg, model_type):
    """The forward emits no next-token logits: a reranker score head (Linear(hidden, 1) per exit layer) or a
    continuous-output forecaster (patch-MLP input, flow-matching sampler output)."""
    lm = lm_of(cfg)
    a = set(archs(cfg))
    # bge-reranker-v2-minicpm-layerwise: head_type other than raw / complex -> LayerWiseHead(hidden, 1)
    if model_type == "minicpm" and ("head_type" in lm or "LayerWiseMiniCPMForCausalLM" in a):
        if str(lm.get("head_type", "simple")).lower() not in ("raw", "complex"):
            return "score_head"
    # bge-reranker-v2.5-gemma2-lightweight: layer_wise -> CostWiseHead(hidden, 1) per cutoff layer
    if model_type == "cost_wise_gemma" and truthy_flag(lm.get("layer_wise")):
        return "score_head"
    # Sundial: continuous patches in, flow-matching samples out
    if model_type == "sundial" or "SundialForPrediction" in a:
        return "continuous_io"
    return None


# --- no_model_type_longcat ------------------------------------------------------------------------------------------
LONGCAT_ARCHS = {"LongcatFlashForCausalLM", "LongcatCausalLM", "LongcatForCausalLM"}


def detect_no_model_type_longcat(cfg, model_type):
    """A LongCat-Flash config saved without model_type: the family is read from architectures."""
    lm = lm_of(cfg)
    if lm.get("model_type") or (cfg or {}).get("model_type"):
        return None
    if set(archs(cfg)) & LONGCAT_ARCHS:
        return "longcat_flash"
    return None


# --- remote_code_not_config_family ----------------------------------------------------------------------------------
REMOTE_REPLACES_FAMILY = {"quillan_ronin_oni", "glm_superexpert"}


def detect_remote_code_not_config_family(cfg, model_type):
    """The auto_map modeling file runs a bespoke stack unrelated to the keys classify reads."""
    if outer_type(cfg) in REMOTE_REPLACES_FAMILY or model_type in REMOTE_REPLACES_FAMILY:
        return "skip"
    return None


# --- audio_key_read_as_lm -------------------------------------------------------------------------------------------
def detect_audio_key_read_as_lm(cfg, model_type):
    """ArkASR's LM is transformers' Qwen2ForCausalLM(config): RoPE always (use_rope is audio-side), q/k/v bias."""
    if model_type == "arkasr":
        return "qwen2"
    return None


# --- breeze ---------------------------------------------------------------------------------------------------------
def detect_breeze_backbone_config_unread(cfg, model_type):
    """Breeze builds its backbone layers from backbone_config (backbone_model_type != 'breeze')."""
    lm = lm_of(cfg)
    if model_type != "breeze":
        return None
    bt = str(lm.get("backbone_model_type") or "breeze").lower()
    bb = lm.get("backbone_config")
    if bt != "breeze" and isinstance(bb, dict):
        return str(bb.get("model_type") or bt).lower()
    return None


def detect_text_encoder_input_embeds(cfg, model_type):
    """Text positions are filled by a (bidirectional) text encoder's projected output, not an embedding lookup."""
    lm = lm_of(cfg)
    if model_type == "breeze" and isinstance(lm.get("text_encoder_config"), dict):
        return True
    return None


def detect_lm_head_vocab_plus_one(cfg, model_type):
    """Breeze's backbone lm_head has vocab_size + 1 rows (an extra EOS class)."""
    lm = lm_of(cfg)
    if model_type == "breeze" and num(lm.get("vocab_size")):
        return num(lm.get("vocab_size")) + 1
    return None


# --- parler ---------------------------------------------------------------------------------------------------------
def detect_multi_codebook_lm_heads(cfg, model_type):
    """Parler-TTS decoder: num_codebooks heads (or one fused head) and a codebook-sum input."""
    lm = lm_of(cfg)
    if model_type == "parler_tts_decoder" and num(lm.get("num_codebooks")) > 1:
        return num(lm.get("num_codebooks"))
    return None


def detect_prompt_prefix_embed_table(cfg, model_type):
    """Parler-TTS with prompt_cross_attention off: transcript tokens embedded by embed_prompts and prepended."""
    if outer_type(cfg) == "parler_tts" and not truthy_flag((cfg or {}).get("prompt_cross_attention")):
        return True
    return None


def detect_cross_attention_decoder_unflagged(cfg, model_type):
    """A decoder of an encoder-decoder model whose every layer cross-attends to the encoder states."""
    lm = lm_of(cfg)
    if truthy_flag(lm.get("add_cross_attention")) and truthy_flag((cfg or {}).get("is_encoder_decoder")):
        return True
    return None


# --- fake_quant_forward ---------------------------------------------------------------------------------------------
# The BitCPM-CANN *-unquantized configs are byte-identical to the stock MiniCPM4 / BitCPM-CANN configs (the
# LinearQuantizer is in the repos' modeling files only), so the code fact is not visible in a config: the change is a
# per-repo table applied where the repo id is known. The detector can only return the table membership of a repo id
# carried in the config, which the snapshot configs never carry.
FAKE_QUANT_REPOS = {"openbmb/BitCPM-CANN-0.5B-unquantized", "openbmb/BitCPM-CANN-1B-unquantized",
                    "openbmb/BitCPM-CANN-3B-unquantized", "openbmb/BitCPM-CANN-8B-unquantized"}


def detect_fake_quant_forward(cfg, model_type):
    rid = (cfg or {}).get("_repo_id")
    return True if rid in FAKE_QUANT_REPOS else None


# --- cyrah ----------------------------------------------------------------------------------------------------------
def detect_cyrah_attn_impl_unread(cfg, model_type):
    """Cyrah attn_impl 'gqa': plain GQA with per-head q/k RMSNorm, all of head_dim rotated, no indexer, windowed only
    on 'sliding_attention' layers. Returns the slots the code runs as "kv=..,pos=..,window=..,qknorm=.."."""
    lm = lm_of(cfg)
    if model_type != "cyrah" or str(lm.get("attn_impl") or "mla").lower() != "gqa":
        return None
    nh = num(lm.get("num_attention_heads"))
    nkv = max(1, num(lm.get("num_key_value_heads")) or 1)
    kv = "MHA" if nkv == nh else ("MQA" if nkv == 1 else "GQA")
    hd = num(lm.get("qk_nope_head_dim")) + num(lm.get("qk_rope_head_dim"))
    rot = num(lm.get("rotary_dim")) or num(lm.get("qk_rope_head_dim"))
    pos = "rope" if hd and rot >= hd else "partial_rope"
    window = "local" if any(t == "sliding_attention" for t in layer_types(lm)) else "full"
    return f"kv={kv},pos={pos},window={window},qknorm=head"


def detect_cyrah_shortconv_is_lfm2(cfg, model_type):
    """Cyrah conv layers with conv_dual_gate: out_proj(C * conv(B * x)) = the LFM2 short conv."""
    lm = lm_of(cfg)
    if model_type == "cyrah" and truthy_flag(lm.get("conv_dual_gate")) and "conv" in layer_types(lm):
        return "shortconv"
    return None


# --- router_feats_on_dense_model ------------------------------------------------------------------------------------
_EXPERT_KEYS = ("num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts")


def _n_layers(lm):
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        if lm.get(k):
            return num(lm[k])
    return len(layer_types(lm))


def detect_router_feats_on_dense_model(cfg, model_type):
    """The config carries router keys but the code builds no routed layer: one expert, every layer dense by
    first_k_dense_replace, or a per-layer block list naming dense MLPs and no MoE."""
    lm = lm_of(cfg)
    present = [k for k in _EXPERT_KEYS if lm.get(k) is not None and not isinstance(lm.get(k), bool)]
    if not present or not any(num(lm.get(k)) for k in present):
        return None
    n_exp = max(num(lm.get(k)) for k in present)
    L = _n_layers(lm)
    if n_exp <= 1:
        return "dense"
    fk = num(lm.get("first_k_dense_replace"))
    if L and fk >= L:
        return "dense"
    lbt = lm.get("layers_block_type")
    if isinstance(lbt, list) and lbt:
        s = {str(x).lower() for x in lbt}
        if ("mlp" in s or "ffn" in s) and "moe" not in s:
            return "dense"
    pat = lm.get("hybrid_override_pattern")
    if isinstance(pat, str) and pat and "-" in pat and "E" not in pat:
        return "dense"
    return None


# --- attn_feats_without_attention -----------------------------------------------------------------------------------
_REC_MARKS = ("linear", "mamba", "ssm", "gdn", "delta", "kda", "rwkv", "conv", "recurrent", "rglru", "lightning")
_ATTN_ONLY_KEYS = ("attn_output_gate", "attention_output_gate", "use_output_gate", "partial_rotary_factor",
                   "rope_parameters", "rope_theta", "rope_scaling", "use_qk_norm", "qk_norm", "qk_layernorm")


def detect_attn_feats_without_attention(cfg, model_type):
    """Every layer of the schedule is recurrent (no softmax attention layer), yet the config carries
    attention-only traits (output gate, RoPE, q/k norm) that the code never applies."""
    lm = lm_of(cfg)
    lt = [str(x).lower() for x in (lm.get("layer_types") or lm.get("layers_block_type") or [])
          if not isinstance(x, (dict, list))]
    if not lt:
        return None
    rec = [x for x in lt if any(m in x for m in _REC_MARKS)]
    rest = [x for x in lt if x not in rec and x not in ("mlp", "moe", "ffn", "-", "e")]
    if not rec or rest:
        return None
    if any(truthy_flag(lm.get(k)) for k in _ATTN_ONLY_KEYS):
        return "no_attention"
    return None


# --- factorized_embedding_proj --------------------------------------------------------------------------------------
def detect_factorized_embedding_proj(cfg, model_type):
    """OPT word_embed_proj_dim != hidden_size: embedding / tied head at width wpd, project_in / project_out."""
    lm = lm_of(cfg)
    if model_type != "opt":
        return None
    wpd, h = num(lm.get("word_embed_proj_dim")), num(lm.get("hidden_size"))
    if wpd and h and wpd != h:
        return wpd
    return None


# --- bidirectional_flag_true ----------------------------------------------------------------------------------------
_DRAFTER_KEYS = ("dflash_config", "num_target_layers", "draft_vocab_size", "eagle_config", "backbone_hidden_size")
_BIDIR_ATTN_TYPES = ("gemma", "gemma2", "gemma3", "gemma3_text", "qwen3")


def detect_bidirectional_flag_true(cfg, model_type):
    """The code runs a non-causal mask: is_causal False (gte / stella modeling_qwen.py), the stella / gte remote
    Qwen2 code with no is_causal key (forward default False), use_bidirectional_attention True (Gemma-1 / 2 / 3
    is_causal = not flag; voyage Qwen3BidirectionalModel), or a *Bidirectional* remote class."""
    lm = lm_of(cfg)
    am = auto_map(cfg)
    if any(k in lm or k in (cfg or {}) for k in _DRAFTER_KEYS):
        return None                           # drafters (DFlash is_causal False) are skipped before gen is read
    if lm.get("is_causal") is False:
        return "diffusion"
    if "modeling_qwen.Qwen2" in am and "is_causal" not in lm and "is_causal" not in (cfg or {}):
        return "diffusion"                    # modeling_qwen.py Qwen2Model / Qwen2ForCausalLM.forward is_causal=False
    if "Bidirectional" in am:
        return "diffusion"
    if lm.get("use_bidirectional_attention") is True and model_type in _BIDIR_ATTN_TYPES:
        return "diffusion"
    return None


# ==================== batch a4_norms ====================

# ---- embedding_norm ------------------------------------------------------------------------------------------------
# a norm applied to the token embeddings before layer 0, fixed in code. Kind:
#   ln         affine LayerNorm (BLOOM word_embeddings_layernorm, RWKV-4/5/7 pre_ln / pre_norm on layer 0)
#   rms_free   weight-free RMS norm (nanochat F.rms_norm, Gaggle, sn38-nanochrono, Muse-Glimmer embed_norm, NeoMME)
#   rms        weighted RMSNorm (Inkling embed_norm)
#   rms_gated  weighted RMSNorm then a low-rank sigmoid gate (Grug-MoE embed_norm + embed_gated_norm)
EMB_NORM_BY_CODE = {"bloom": "ln", "rwkv": "ln", "rwkv5": "ln", "rwkv6": "ln",
                    "nanochat_gpt": "rms_free", "nanochat": "rms_free", "gaggle": "rms_free",
                    "sn38-nanochrono": "rms_free", "muse_glimmer": "rms_free", "muse_glimmer_text": "rms_free",
                    "neomme": "rms_free", "inkling": "rms", "grug_moe": "rms_gated"}


def detect_embedding_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "rwkv7":                       # fla RWKV7Block: pre_norm on layer 0 iff norm_first (default True)
        return "ln" if lm.get("norm_first", True) is not False else None
    return EMB_NORM_BY_CODE.get(model_type)


# ---- norm_placement_keys_unread ------------------------------------------------------------------------------------
# the FEATURES the norm placement fixed in code (or a placement key) gives, '+'-joined
_G1_POST_NORM_BY_CODE = {"plamo3", "ethoric", "domynedge", "ouro", "cost_wise_gemma", "beam", "kolibri1"}
_G1_POST_LN_REMOTE = {"minimax_text_01", "minimax_m1"}      # remote code: residual = normed stream iff postnorm


def detect_norm_placement_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in _G1_POST_NORM_BY_CODE:
        return "post_norm"
    if model_type == "molmo" and lm.get("norm_after") is True:
        return "post_only"
    if model_type == "bolmo" and (lm.get("block_type") or "reordered_norm") == "reordered_norm":
        return "post_only"
    if model_type == "minimax" and not any("MiniMaxM2" in a for a in archs(cfg)):
        return "post_ln"                                    # tf5 MiniMaxDecoderLayer: always the normed residual
    if model_type in _G1_POST_LN_REMOTE and truthy_flag(lm.get("postnorm")):
        return "post_ln"
    if model_type == "iquest_q1":
        return "post_ln+post_norm+post_norm_attn"           # normed residual, attention-only post-norm (layers > 0)
    if model_type == "needle":
        return "post_norm+post_norm_attn"                   # post_attn_norm only; no MLP post-norm
    return None


# ---- grouped_rmsnorm -----------------------------------------------------------------------------------------------
def detect_grouped_rmsnorm(cfg, model_type):
    """K2HorizonRMSNorm(n_groups = layernorm_num_groups) on every decoder / final norm: the group count (> 1)."""
    lm = lm_of(cfg)
    if model_type in ("k2_horizon", "jais_asr"):
        g = num(lm.get("layernorm_num_groups"))
        return g if g > 1 else None
    return None


# ---- post_norm_variant ---------------------------------------------------------------------------------------------
def detect_post_norm_variant(cfg, model_type):
    """Muse-Spark: weight-free post-attention norm, input-scaled post-FFN norm (rms((1 + w) x)), no output gain."""
    if model_type in ("muse_spark", "muse_spark_text"):
        return "weightfree_inputscaled"
    return None


# ---- attn_output_head_norm -----------------------------------------------------------------------------------------
def detect_attn_output_head_norm(cfg, model_type):
    """A norm on the attention output before o_proj: Muse-Spark o_norm (weight-free, per head over head_dim);
    BitNet attn_sub_norm (weighted, full width; none in the snapshot)."""
    if model_type in ("muse_spark", "muse_spark_text"):
        return "head_free"
    if model_type == "bitnet":
        return "full_w"
    return None


# ---- ceno_megatron_stream_renorm -----------------------------------------------------------------------------------
def detect_ceno_megatron_stream_renorm(cfg, model_type):
    """CENOBlock: attention blocks return pre_mlp_norm(x + attn(norm x)); MoE / MLP blocks take norm(x) as the
    residual. Any attention / MoE / MLP block renormalises the stream -> FEATURES post_ln."""
    if model_type != "ceno":
        return None
    lm = lm_of(cfg)
    pat = lm.get("hybrid_override_pattern")
    lbt = lm.get("layers_block_type")
    if isinstance(pat, str) and any(c in pat for c in "*E-"):
        return "post_ln"
    if isinstance(lbt, list) and any(str(x) in ("attention", "moe", "mlp") for x in lbt):
        return "post_ln"
    return None


# ---- residual_gain_vector ------------------------------------------------------------------------------------------
def detect_residual_gain_vector(cfg, model_type):
    """Learned per-channel gains on the residual branch: Doge g * r + f (gain), ZAYA (r + b) * s (gain_bias, under
    scale_residual_merge), Muse-Spark alpha(w) r + beta(w) f with alpha^2 + beta^2 = 1 (gate)."""
    if model_type == "doge":
        return "gain"
    if model_type in ("zaya1_vl", "zaya", "zaya1") and truthy_flag(get(cfg, "scale_residual_merge")):
        return "gain_bias"
    if model_type in ("muse_spark", "muse_spark_text"):
        return "gate"
    return None


# ---- backout_residual ----------------------------------------------------------------------------------------------
def detect_backout_residual(cfg, model_type):
    """Final hidden x_L - backout_lambda * x_m before the final norm: Gaggle (m = n_layer // 2, always built),
    nanochat_gpt (m = config backout_layer, when not None)."""
    if model_type == "gaggle":
        return True
    if model_type == "nanochat_gpt" and get(cfg, "backout_layer") is not None:
        return True
    return None


# ---- ngpt_sphere_residual ------------------------------------------------------------------------------------------
def detect_ngpt_sphere_residual(cfg, model_type):
    """Dory (nGPT): weight-free L2 norms, h <- L2(h) + |lambda s| (L2(f) - L2(h)) instead of h + f."""
    if model_type == "dory":
        return "ngpt"
    return None


# ---- ffn_inner_norm ------------------------------------------------------------------------------------------------
def detect_ffn_inner_norm(cfg, model_type):
    """A norm on the FFN intermediate act(gate) * up before down_proj: Dory ffn_sub_norm (weight-free L2, eps 1e-6),
    BitNet ffn_sub_norm (weighted RMSNorm; none in the snapshot)."""
    if model_type == "dory":
        return "l2_free"
    if model_type == "bitnet":
        return "rms_w"
    return None


# ---- gated_prenorm_lowrank -----------------------------------------------------------------------------------------
def detect_gated_prenorm_lowrank(cfg, model_type):
    """y = RMSNorm(x); y * sigmoid(W2 silu(W1 y)) on the sublayer inputs: (sites, rank).
    A.X-K2: input norm every layer + post-attention norm on sparse layers, rank gated_norm_rank (16);
    Grug-MoE: both layer norms, the embedding norm and the final norm, rank GRUG_GATED_NORM_RANK = 128."""
    lm = lm_of(cfg)
    if model_type == "axk2":
        return ("layers", num(lm.get("gated_norm_rank")) or 16)
    if model_type == "grug_moe":
        return ("layers+emb+final", 128)
    return None


# ---- multimax_logits -----------------------------------------------------------------------------------------------
def detect_multimax_logits(cfg, model_type):
    """ERNIE-5.5 lm_head multimax: l + t0 relu(r0 - l) + t1 relu(l - r1) + t2 relu(r2 - l)^2 + t3 relu(l - r3)^2."""
    v = get(cfg, "multimax")
    if model_type.startswith("ernie5_5") and v:
        return str(v)
    return None


# ==================== batch a5_positions ====================

def _head_dim(lm):
    nh = num(lm.get("num_attention_heads") or lm.get("n_head"))
    hidden = num(lm.get("hidden_size") or lm.get("d_model") or lm.get("n_embd"))
    return num(lm.get("head_dim")) or (hidden // nh if nh else 0)


def _per_type(rp):
    """rope_parameters keyed by layer type -> {type: dict or None}."""
    if not isinstance(rp, dict):
        return {}
    return {k: v for k, v in rp.items() if (isinstance(v, dict) or v is None) and "attention" in k}


# ---------------------------------------------------------------------------------------------------------------
def detect_position_query_temperature(cfg, model_type):
    """A position-dependent query multiplier. Returns the code path and, for the gated ones, the first position at
    which it differs from 1: 'logn@<seq_length>' (Qwen-1 use_logn_attn), 'llama4_beta@<orig>' (Ministral-3 /
    Mistral-4 / Nemotron-Labs-Diffusion rope_parameters.llama_4_scaling_beta), 'nope_temp@<floor_scale>' (Llama-4
    attn_temperature_tuning on its NoPE layers), 'learned_logpos' (Mistral-Large-4 learned per-head
    scale_h log(pos+1) + bias_h), 'ssmax' (Beam (1 + (pos+1)/2048)^0.1 on its NoPE layers)."""
    lm = lm_of(cfg)
    mt = model_type
    if mt == "qwen" and lm.get("use_logn_attn") is True:
        return f"logn@{num(lm.get('seq_length')) or 8192}"
    if mt in ("ministral3", "mistral4", "nemotron_labs_diffusion"):
        rp = rope_params(lm)
        beta = fnum(rp.get("llama_4_scaling_beta"))
        if beta is None and mt == "mistral4" and not (lm.get("rope_parameters") or lm.get("rope_scaling")):
            beta = 0.1                                     # Mistral4Config default rope_parameters
        if beta:
            return f"llama4_beta@{num(rp.get('original_max_position_embeddings')) or 8192}"
        return None
    if mt in ("llama4", "llama4_text"):
        if lm.get("attn_temperature_tuning", True) is False or fnum(lm.get("attn_scale"), 0.1) == 0:
            return None
        nrl = lm.get("no_rope_layers")
        has_nope = not (isinstance(nrl, list) and nrl and all(x for x in nrl))  # class default: NoPE every 4th
        return f"nope_temp@{num(lm.get('floor_scale')) or 8192}" if has_nope else None
    if mt in ("mistral_large4_text", "mistral_large4"):
        return "learned_logpos"
    if mt == "beam":
        if any(v is None for v in _per_type(lm.get("rope_parameters")).values()):
            return "ssmax"
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_local_rope_theta_unread(cfg, model_type):
    """Two inv_freq tables by layer type that _rope_schedules sees as one: OLMo-3 (a legacy rope_scaling goes to
    full_attention only; sliding layers keep the default table) and PLaMo-3 (rope_local_theta on the sliding layers,
    rope_theta / YaRN on the full ones). Returns the corrected rope_sched: one head_dim entry per table."""
    lm = lm_of(cfg)
    hd = _head_dim(lm)
    lt = layer_types(lm)
    if model_type == "olmo3":
        rs = lm.get("rope_scaling")
        per = _per_type(lm.get("rope_parameters"))
        if per:
            kinds = {repr(sorted((v or {}).items())) for v in per.values()}
            return [hd, hd] if len(kinds) > 1 else None
        if (isinstance(rs, dict) and str(rs.get("rope_type") or rs.get("type") or "default") != "default"
                and any("sliding" in x for x in lt) and any("full" in x for x in lt)):
            return [hd, hd]
        return None
    if model_type == "plamo3":
        has_sliding = bool(lm.get("window_size") or any("sliding" in x for x in lt))
        loc, glob = lm.get("rope_local_theta", 10000), lm.get("rope_theta", 10000)
        if has_sliding and (loc != glob or fnum(lm.get("rope_scaling_factor"), 1.0) != 1.0
                            or len(_per_type(lm.get("rope_parameters"))) > 1):
            return [hd, hd]
    return None


# ---------------------------------------------------------------------------------------------------------------
LFM3_TYPES = ("lfm3", "lfm3_rd", "lfm3_omni")


def detect_rope_cap_zero_tail_table(cfg, model_type):
    """LFM3's remote rotary (modeling_lfm3.py compute_default_rope_parameters / compute_rope_parameters) builds every
    table over the WHOLE head with zero frequencies past partial_rotary_factor: the buffer drives head_dim rotated
    dims. Returns the corrected rope_cap: head_dim per rotary table (one per attention layer type present)."""
    lm = lm_of(cfg)
    if model_type not in LFM3_TYPES or "modeling_lfm3" not in auto_map(cfg):
        return None
    hd = _head_dim(lm)
    lt = layer_types(lm)
    per = _per_type(lm.get("rope_parameters"))
    n = len([k for k in per if k in lt]) or 1
    return [hd] * n


# ---------------------------------------------------------------------------------------------------------------
def detect_partial_rope_keys_unread(cfg, model_type):
    """Rotated dims set by a key _rotary_factor / _rope_schedules do not read. Returns the corrected rope_sched
    (rotating entries only; NoPE layers are nope_full_layers_unlisted's): Step-3.5 partial_rotary_factors per layer
    (one table per distinct (factor, theta)), K2-Horizon / Jais-3 rope_head_dim < head_dim, StableLM-Epoch rope_pct,
    ZAYA1-VL partial_rotary_factor class default 0.5, Grug-MoE hard-coded half RoPE."""
    lm = lm_of(cfg)
    hd = _head_dim(lm)
    if not hd:
        return None
    if model_type == "step3p5" and isinstance(lm.get("partial_rotary_factors"), list):
        prf = lm["partial_rotary_factors"]
        th = lm.get("rope_theta")
        th = th if isinstance(th, list) and len(th) == len(prf) else [th] * len(prf)
        tables = {}
        for f, t in zip(prf, th):
            tables[(f, t)] = int(round(float(f) * hd))
        sched = sorted(tables.values(), reverse=True)
        return sched if any(r < hd for r in sched) else None
    if model_type == "k2_horizon":
        r = num(lm.get("rope_head_dim"))
        return [r] if 0 < r < hd else None
    if model_type == "stablelm_epoch" and fnum(lm.get("rope_pct")) is not None and "rotary_pct" not in lm:
        r = int(hd * float(lm["rope_pct"]))
        return [r] if r < hd else None
    if model_type == "zaya1_vl" and lm.get("partial_rotary_factor") is None:
        rp = rope_params(lm)
        f = fnum(rp.get("partial_rotary_factor"), 0.5)
        r = int(round(f * hd))
        return [r] if r < hd else None
    if model_type == "grug_moe":
        return [hd // 2]
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_rotary_dim_ignored(cfg, model_type):
    """BailingMoeV2 (Ling-1T / Ring-1T remote modeling_bailing_moe_v2.py) rotates head_dim * partial_rotary_factor
    (absent -> 1.0) through transformers' default rope init; a config rotary_dim is never read. Returns the corrected
    rope_sched (head_dim,) where a config gives rotary_dim < head_dim and no partial_rotary_factor."""
    lm = lm_of(cfg)
    if model_type != "bailing_moe" or "bailing_moe_v2" not in auto_map(cfg):
        return None
    hd = _head_dim(lm)
    rd = num(lm.get("rotary_dim"))
    if rd and rd < hd and lm.get("partial_rotary_factor") is None and rope_params(lm).get("partial_rotary_factor") is None:
        return [hd]
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_dynamic_ntk_in_context(cfg, model_type):
    """A RoPE base that depends on the current sequence length L (dynamic NTK) once L passes a threshold. Returns
    'dyn_ntk@<threshold>': Qwen-1 use_dynamic_ntk (threshold seq_length), Meditor-1 / -2 global layers (code-fixed
    32768 / 8192)."""
    lm = lm_of(cfg)
    if model_type == "qwen" and lm.get("use_dynamic_ntk") is True:
        return f"dyn_ntk@{num(lm.get('seq_length')) or 8192}"
    if model_type == "meditor" and "modeling_meditor" in auto_map(cfg):
        return "dyn_ntk@32768"
    if model_type == "meditor2" and "modeling_meditor2" in auto_map(cfg):
        return "dyn_ntk@8192"
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_falcon_alibi_slope_scale(cfg, model_type):
    """Falcon's ALiBi path computes softmax((q k + alibi) / sqrt(head_dim)) (SDPA: the mask holds alibi / sqrt(hd)):
    its effective slopes are bf16(S(n)) / sqrt(head_dim). Returns that factor (alibi_scale), 1 / sqrt(head_dim)."""
    lm = lm_of(cfg)
    if model_type != "falcon" or not lm.get("alibi"):
        return None
    nh = num(lm.get("num_attention_heads") or lm.get("n_head"))
    hd = num(lm.get("hidden_size")) // nh if nh else 0
    return 1.0 / hd ** 0.5 if hd else None


# ---------------------------------------------------------------------------------------------------------------
def detect_mpt_remote_alibi_default_false(cfg, model_type):
    """Remote llm-foundry MPTConfig fills attn_config from attn_config_defaults (alibi False) and builds wpe when
    alibi is off (learned_pos_emb True default): an auto_map MPT config whose attn_config omits alibi runs learned
    absolute positions. Returns the corrected pos value."""
    lm = lm_of(cfg)
    if model_type != "mpt" or "modeling_mpt" not in auto_map(cfg):
        return None
    ac = lm.get("attn_config")
    ac = ac if isinstance(ac, dict) else {}
    if "alibi" in ac:
        return None
    return "learned_abs" if lm.get("learned_pos_emb", True) is not False else "nope"


# ---------------------------------------------------------------------------------------------------------------
def detect_use_mem_rope_unread(cfg, model_type):
    """Zamba2's shared attention applies RoPE only when config.use_mem_rope (class default False). Returns the
    corrected pos value 'nope' where use_mem_rope is not True."""
    lm = lm_of(cfg)
    if model_type == "zamba2" and lm.get("use_mem_rope") is not True:
        return "nope"
    return None


# ---------------------------------------------------------------------------------------------------------------
MUSE_TYPES = ("muse_glimmer", "muse_glimmer_text", "muse_spark", "muse_spark_text")


def detect_nope_layer_keys_unread(cfg, model_type):
    """Per-layer RoPE / NoPE switches the classifier does not read. Returns the no_rope_layers list (1 = RoPE,
    0 = NoPE) the code runs: Muse-Glimmer / Muse-Spark layer_rope_theta (0 = NoPE; absent = the class default, NoPE
    every 4th layer counted from the last), Domyn-Edge position_layers ('rope' / 'nope'), Kolibri-1 (Exaone4Attention:
    RoPE only on sliding layers while sliding_window is set)."""
    lm = lm_of(cfg)
    n = num(lm.get("num_hidden_layers"))
    if model_type in MUSE_TYPES:
        lrt = lm.get("layer_rope_theta")
        if isinstance(lrt, list) and lrt:
            out = [int(bool(x)) for x in lrt]
        elif isinstance(lm.get("no_rope_layers"), list) and lm["no_rope_layers"]:
            out = [int(bool(x)) for x in lm["no_rope_layers"]]
        elif n:
            out = [0 if (n - 1 - i) % 4 == 0 else 1 for i in range(n)]
        else:
            return None
        return out if 0 in out and 1 in out else None
    if model_type == "domynedge":
        pl = lm.get("position_layers")
        if isinstance(pl, list) and pl:
            out = [int(str(x) == "rope") for x in pl]
            return out if 0 in out and 1 in out else None
        return None
    if model_type == "kolibri1":
        lt = layer_types(lm)
        sw = lm.get("sliding_window", 513)
        if sw and any("sliding" in x for x in lt) and any("full" in x for x in lt):
            return [int("sliding" in x) for x in lt]
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_nope_full_layers_unlisted(cfg, model_type):
    """Grug-MoE: disable_long_rope (default and forced True) leaves its 'long' full-attention layers (every 4th and
    the last, grug_long_layer_flags) without RoPE; the sliding layers rotate. Returns no_rope_layers (1 = RoPE)."""
    lm = lm_of(cfg)
    if model_type != "grug_moe":
        return None
    n = num(lm.get("num_hidden_layers"))
    if not n:
        return None
    return [0 if ((i + 1) % 4 == 0 or i == n - 1) else 1 for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------
def detect_rope_embeddings_flag_unread(cfg, model_type):
    """Parler-TTS decoder: rotary only when config.rope_embeddings (default False); otherwise the fixed sinusoidal
    table added to inputs_embeds. Returns the corrected pos value 'sinusoidal'."""
    lm = lm_of(cfg)
    if model_type == "parler_tts_decoder" and not truthy_flag(lm.get("rope_embeddings", False)):
        return "sinusoidal"
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_fope_positions(cfg, model_type):
    """Intern-S1-Pro FoPE: any of rope_scaling fope_init_factor / fope_sep_heads / num_inv_freq set turns the rotary
    into InternS1ProMoeTextFourierEmbedding (learned per-KV-head cos / sin mixing matrices, frequencies below
    2 pi / max_position_embeddings dropped). Returns the FEATURES key 'fope'."""
    lm = lm_of(cfg)
    rs = lm.get("rope_scaling") if isinstance(lm.get("rope_scaling"), dict) else {}
    rp = lm.get("rope_parameters") if isinstance(lm.get("rope_parameters"), dict) else {}
    if any((d.get(k) is not None) for d in (rs, rp) for k in ("fope_init_factor", "fope_sep_heads", "num_inv_freq")):
        return "fope"
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_content_warped_rope(cfg, model_type):
    """GPT-X2 (CIx) use_symplectic_rope: each layer's RoPE phase is warped by a content-driven, non-causal,
    stateful clock (compute_metric_warped_freqs_cis). Returns the FEATURES key 'rope_warp'."""
    lm = lm_of(cfg)
    if model_type == "gptx2" and lm.get("use_symplectic_rope") is True and "modeling_gptx2" in auto_map(cfg):
        return "rope_warp"
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_use_mla_false_rope_dims(cfg, model_type):
    """DeepSeek-V2 remote code with use_mla false (kv_lora_rank null) builds LlamaAttention (MHA over
    hidden / heads with full RoPE); qk_rope_head_dim (0 in the config) is unread. Returns the corrected rope_sched
    (head_dim,)."""
    lm = lm_of(cfg)
    if model_type != "deepseek_v2" or lm.get("use_mla") is not False or lm.get("kv_lora_rank"):
        return None
    nh = num(lm.get("num_attention_heads"))
    hd = num(lm.get("hidden_size")) // nh if nh else 0
    return [hd] if hd else None


# ==================== batch a6_masks ====================

def _ilist(v):
    """A list of ints (a JSON string of a list is read too), else []."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return []
    return [int(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)] if isinstance(v, list) else []


def _slist(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return []
    return [str(x) for x in v] if isinstance(v, list) else []


def _layout(kinds, window, mpe):
    """(n_full, n_local, window) of a per-layer 'full' / 'local' list; falsy when no sliding layer binds."""
    nl = sum(1 for k in kinds if k == "local")
    nf = sum(1 for k in kinds if k == "full")
    if not nl or not window or (mpe and window >= mpe):
        return None
    return [nf, nl, int(window)]


# ---------------------------------------------------------------------------------------------------------------
def detect_sliding_window_keys_unread(cfg, model_type):
    """The code's softmax-layer layout (n_full, n_local, visible keys per sliding layer) for the families whose
    window / schedule keys p5 does not read (each from the family's own code; see the spec)."""
    lm = lm_of(cfg)
    mt = model_type
    n = layer_count(lm)
    mpe = num(lm.get("max_position_embeddings") or lm.get("max_seq_len") or lm.get("sequence_len"))
    if not n:
        return None
    if mt == "cost_wise_gemma":                       # transformers-4.42 Gemma2DecoderLayer: is_sliding = not idx % 2
        return _layout(["local" if i % 2 == 0 else "full" for i in range(n)], num(lm.get("sliding_window")), mpe)
    if mt == "needle":                                # cactus architecture.py: band on all but global_layers
        gl = set(_ilist((lm.get("extras") or {}).get("global_attention_layers")))
        if not gl:
            return None                               # (no global layers listed: every layer local, already read)
        return _layout(["full" if i in gl else "local" for i in range(n)], num(lm.get("sliding_window")), mpe)
    if mt == "iquest_q1":                             # IQuestQ1Config: first + block * n_blocks + last
        if lm.get("use_hybrid_layers", True) is False or lm.get("use_sliding_window", True) is False:
            return None
        first = _slist(lm.get("first_layers_types")) or ["full_attention"]
        block = _slist(lm.get("hybrid_layers_types_block")) or ["full_attention"] + ["sliding_attention"] * 3
        last = _slist(lm.get("last_layers_types")) or ["full_attention"] * 3
        lt = first + block * (num(lm.get("num_hybrid_layers_block")) or 21) + last
        if len(lt) != n:
            return None
        return _layout(["local" if "sliding" in x else "full" for x in lt], num(lm.get("sliding_window")), mpe)
    if mt == "motif":                                 # MotifGDLAttention: window sliding_window + 1 on SWA layers
        sw = num(lm.get("sliding_window"))
        if not lm.get("use_sliding_window") or not sw:
            return None
        pat = str(lm.get("sliding_window_pattern") or "interleave")
        per = num(lm.get("sliding_window_period")) or 2
        if pat == "all":
            kinds = ["local"] * n
        elif pat == "interleave":
            kinds = ["local" if i % per != 0 else "full" for i in range(n)]
        else:
            return None
        return _layout(kinds, sw + 1, mpe)
    if mt in ("meditor", "meditor2"):                 # is_global_layer = (i + 1) % global_attn_every_n_layers == 0
        g = num(lm.get("global_attn_every_n_layers")) or 6
        sw = num(lm.get("sliding_window_size")) or 2048
        return _layout(["full" if (i + 1) % g == 0 else "local" for i in range(n)], sw, mpe)
    if mt == "gaggle":                                # GaggleConfig.window_sizes; mask row - col <= window
        pat = str(lm.get("window_pattern") or "SSSL").upper()
        if not pat or any(c not in "SL" for c in pat):
            return None
        long_w = num(lm.get("sequence_len")) or 2048
        short_w = -(-long_w // 4 // 128) * 128
        kinds = ["full" if pat[i % len(pat)] == "L" else "local" for i in range(n)]
        kinds[-1] = "full"
        return _layout(kinds, short_w + 1, long_w + 1)
    if mt == "jet_nemotron":                          # 'swa' layers: efficient_attention_config.swa.window_size
        lt = layer_types(lm)
        swa = ((lm.get("efficient_attention_config") or {}).get("swa") or {})
        w = num(swa.get("window_size")) if isinstance(swa, dict) else 0
        if "swa" not in lt or not w:
            return None
        return _layout(["local" if x == "swa" else ("full" if x == "attn" else "rec") for x in lt], w, mpe)
    if mt == "grug_moe":                              # grug_long_layer_flags: (i + 1) % global_every == 0 or last
        g = num(lm.get("global_every")) or 4
        sw = num(lm.get("sliding_window")) or 2048
        return _layout(["full" if ((i + 1) % g == 0 or i == n - 1) else "local" for i in range(n)], sw, mpe)
    if mt == "plamo3":                                # is_full_attn: (i + 1) % sliding_window_pattern == 0
        p = num(lm.get("sliding_window_pattern")) or 8
        w = num(lm.get("window_size")) or 2048
        return _layout(["full" if (i + 1) % p == 0 else "local" for i in range(n)], w, mpe)
    if mt == "edgerock":                              # fixed_memory_swa_layers -> Qwen3_5FixedWindowAttention
        swa = set(_ilist(lm.get("fixed_memory_swa_layers")))
        lt = _slist(lm.get("layer_types"))
        if not swa or len(lt) != n:
            return None
        kinds = ["rec" if x == "linear_attention" else ("local" if i in swa else "full") for i, x in enumerate(lt)]
        return _layout(kinds, num(lm.get("fixed_memory_swa_window")) or 1024, mpe)
    if mt == "ethoric":                               # attn_window on attn_window_layers (q - k < W)
        wl = set(_ilist(lm.get("attn_window_layers")))
        w = num(lm.get("attn_window"))
        if not wl or not w:
            return None
        return _layout(["local" if i in wl else "full" for i in range(n)], w, mpe)
    if mt == "fabric":                                # every layer's GQAAttention: window local_attention_window
        if str(lm.get("architecture") or "fabric") == "full":
            return None
        return _layout(["local"] * n, num(lm.get("local_attention_window")) or 2048,
                       num(lm.get("sequence_length")) or mpe)
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_sliding_window_backend_dependent(cfg, model_type):
    """Phi-tiny / Phi-mini-MoE: the remote modeling_slimmoe.py SDPA path (the default backend) builds a plain causal
    mask, so a binding sliding_window is NOT applied: the window slot the code computes is 'full'."""
    lm = lm_of(cfg)
    if model_type != "phimoe" or "modeling_slimmoe" not in auto_map(cfg):
        return None
    sw, mpe = num(lm.get("sliding_window")), num(lm.get("max_position_embeddings"))
    return "full" if sw and (not mpe or sw < mpe) else None


# ---------------------------------------------------------------------------------------------------------------
# families whose attention applies config.sliding_window only on 'sliding_attention' layer_types entries
LAYER_TYPED_WINDOW_TYPES = frozenset({"gemma2", "gemma3", "gemma3_text", "gemma3n_text", "gemma4", "gemma4_text",
                                      "gemma4_unified_text", "vaultgemma", "diffusion_gemma_text"})


def detect_layer_types_all_full_window_ignored(cfg, model_type):
    """A binding sliding_window that the code never applies: layer_types lists no sliding layer -> window 'full'."""
    lm = lm_of(cfg)
    if model_type not in LAYER_TYPED_WINDOW_TYPES:
        return None
    lt = layer_types(lm)
    sw, mpe = num(lm.get("sliding_window")), num(lm.get("max_position_embeddings"))
    if not lt or not sw or (mpe and sw >= mpe) or any("sliding" in x for x in lt):
        return None
    return "full"


# ---------------------------------------------------------------------------------------------------------------
def detect_swa_prefill_key_truncation(cfg, model_type):
    """Meditor / Meditor-2: sliding layers truncate K / V to the last sliding_window_size keys BEFORE the causal
    SDPA, so a cache-less prefill longer than the window is top-left causal over the wrong keys (future tokens).
    Returns the FEATURES key."""
    lm = lm_of(cfg)
    if model_type not in ("meditor", "meditor2"):
        return None
    g = num(lm.get("global_attn_every_n_layers")) or 6
    n = layer_count(lm)
    if not n or all((i + 1) % g == 0 for i in range(n)):
        return None
    return "swa_trunc"


# ---------------------------------------------------------------------------------------------------------------
def detect_swa_window_plus_one(cfg, model_type):
    """sn38-nanochrono: _attend keeps key j iff row - col <= window, i.e. sliding_window + 1 visible keys."""
    lm = lm_of(cfg)
    if model_type != "sn38-nanochrono":
        return None
    sw, mpe = num(lm.get("sliding_window")), num(lm.get("max_position_embeddings"))
    if not sw or not any("sliding" in x for x in layer_types(lm)) or (mpe and sw + 1 >= mpe):
        return None
    return sw + 1


# ---------------------------------------------------------------------------------------------------------------
def detect_runtime_kv_window_unread(cfg, model_type):
    """needle kv_window: the C++ runtime's KV window (min(sliding_window, kv_window)); the reference forward never
    reads it. Returns the runtime window (a deployment fact, not the model function)."""
    lm = lm_of(cfg)
    if model_type != "needle":
        return None
    kw, sw = num(lm.get("kv_window")), num(lm.get("sliding_window"))
    if not kw:
        return None
    return min(sw, kw) if sw else kw


# ---------------------------------------------------------------------------------------------------------------
def detect_chunked_attention_unread(cfg, model_type):
    """Llama-4: chunked_attention (block-diagonal causal, attention_chunk_size) on every RoPE layer
    (Llama4TextConfig.__post_init__ layer_types). Returns (n_chunked, chunk_size)."""
    lm = lm_of(cfg)
    if model_type not in ("llama4", "llama4_text"):
        return None
    n = layer_count(lm)
    c, mpe = num(lm.get("attention_chunk_size")), num(lm.get("max_position_embeddings"))
    if not n or not c or (mpe and c >= mpe):
        return None
    lt = layer_types(lm)
    if lt:
        k = sum(1 for x in lt if "chunk" in x)
    else:
        nrl = lm.get("no_rope_layers")
        iv = num(lm.get("no_rope_layer_interval")) or 4
        nrl = nrl if (isinstance(nrl, list) and nrl) else [int((i + 1) % iv != 0) for i in range(n)]
        k = sum(1 for x in nrl if x)
    return [k, c] if k else None


# ---------------------------------------------------------------------------------------------------------------
def detect_blocksparse_mask(cfg, model_type):
    """Phi-3-small: block-sparse attention (local blocks + vertical stride, per-head shifted) on every layer with
    (i + 1) % dense_attention_every_n_layers != 0. Returns (n_blocksparse, dense-equivalent context)."""
    lm = lm_of(cfg)
    if model_type != "phi3small":
        return None
    n = layer_count(lm)
    d = num(lm.get("dense_attention_every_n_layers"))
    bs, lb = num(lm.get("blocksparse_block_size")), num(lm.get("blocksparse_num_local_blocks"))
    if not n or not bs or not lb:
        return None
    k = sum(1 for i in range(n) if not (d and (i + 1) % d == 0))
    mpe = num(lm.get("max_position_embeddings"))
    if not k or (mpe and bs * lb >= mpe):
        return None
    return [k, bs * lb]


# ---------------------------------------------------------------------------------------------------------------
def detect_doge_dynamic_mask(cfg, model_type):
    """Doge: the additive per-key logit bias exp(A * softplus(dt_proj(v))) (+ top-keep_window_size keys) on every
    attention layer. Returns the FEATURES key."""
    return "dyn_mask" if model_type == "doge" else None


# ---------------------------------------------------------------------------------------------------------------
def detect_eos_document_mask(cfg, model_type):
    """In-sequence documents split at eos: same-document attention (+ conv / recurrent / n-gram resets).
    Ethoric: doc_masking (default True); KLM: always (segments()). Returns the FEATURES key."""
    lm = lm_of(cfg)
    if model_type == "ethoric" and lm.get("doc_masking", True) is not False:
        return "doc_mask"
    if model_type == "klm":
        return "doc_mask"
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_prefix_lm_mask(cfg, model_type):
    """A prefix-LM mask: bidirectional over the prompt (prefill), causal after. DIVE-Doc's PaliGemma-style
    _update_causal_mask (eval: causal_mask[:, :sequence_length] = 0); MPT attn_config.prefix_lm. Returns gen."""
    lm = lm_of(cfg)
    if outer_type(cfg) in ("divedoc", "paligemma") or any(a in ("DIVEdoc", "PaliGemmaForConditionalGeneration")
                                                          for a in archs(cfg)):
        return "prefix"
    if model_type == "mpt" and truthy_flag((lm.get("attn_config") or {}).get("prefix_lm")):
        return "prefix"
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_code_ignores_window_and_sink(cfg, model_type):
    """nanowhale (remote modeling_deepseek_v4.py, DeepseekV4Attention / DeepseekV4Model): a full triu causal mask
    and no attention sink in the SDPA path, whatever sliding_window says. Returns (window, sink)."""
    if model_type != "deepseek_v4" or "modeling_deepseek_v4.DeepseekV4ForCausalLM" not in auto_map(cfg):
        return None
    lm = lm_of(cfg)
    cr = lm.get("compress_ratios")
    if isinstance(cr, list) and any(num(x) for x in cr):
        return None                                   # (a compressed V4: not the nanowhale code)
    return ["full", "no"]


# ---------------------------------------------------------------------------------------------------------------
def detect_hybrid_layer_schedule_unread(cfg, model_type):
    """Hybrid stacks whose recurrent-layer schedule p5 does not read (model_type / key unknown to it).
    Returns (n_softmax, n_recurrent, recurrent family)."""
    lm = lm_of(cfg)
    n = layer_count(lm)
    if not n:
        return None
    a = archs(cfg)
    if model_type == "gptx3" and num(lm.get("gdn_layers_per_gqa")):      # (i + 1) % (k + 1) == 0 softmax
        k = num(lm["gdn_layers_per_gqa"])
        s = sum(1 for i in range(n) if (i + 1) % (k + 1) == 0)
        return [s, n - s, "gdn2"]
    if model_type == "klm":                                               # i % every == every - 1 softmax
        e = num(lm.get("attention_every")) or 4
        s = sum(1 for i in range(n) if i % e == e - 1)
        return [s, n - s, "gdn2"]
    if model_type == "solar_open2":                                       # gqa_layers, else gqa_interval
        gl = lm.get("gqa_layers")
        iv = num(lm.get("gqa_interval"))
        if isinstance(gl, list):
            s = sum(1 for i in range(n) if i in set(_ilist(gl)))
        elif iv:
            s = sum(1 for i in range(n) if (i + 1) % iv == 0)
        else:
            return None
        neg = lm.get("kda_allow_neg_eigval", False) is True
        fam = "kda_lb_negeig" if neg else "kda_lb"
        return [s, n - s, fam] if s < n else None
    if model_type != "bailing_hybrid" and ("BailingMoeV3ForCausalLM" in a or model_type == "ming_v3_asr"):
        G = num(lm.get("layer_group_size")) or (5 if model_type == "ming_v3_asr" else 0)
        if not G:
            return None
        s = sum(1 for i in range(n) if (i + 1) % G == 0 or i >= n // G * G)
        lb = lm.get("kda_lower_bound") if lm.get("kda_safe_gate") else None
        return [s, n - s, "kda_lb" if lb is not None else "kda"]
    return None


# ---------------------------------------------------------------------------------------------------------------
def detect_attn_output_saturation(cfg, model_type):
    """Qwen3-0.6B-D-ROS (remote modeling_qwen3_eq.py): every attention output passes _apply_dros (threshold /
    tanh-knee saturation on avg-pool bands in layers int(0.6 L)..int(0.9 L), dry / wet mix). FEATURES key."""
    lm = lm_of(cfg)
    dc = lm.get("d_ros_config", (cfg or {}).get("d_ros_config"))
    if not isinstance(dc, dict) or dc.get("enabled", True) is False or "modeling_qwen3_eq" not in auto_map(cfg):
        return None
    dw = fnum(dc.get("dry_wet"), 0.70)
    return "attn_out_sat" if dw > 0 else None


# ==================== batch a7_attn_exotic ====================

def _s(v):
    return str(v).strip().lower() if v is not None else ""


# --- out_gate_headwise -------------------------------------------------------------------------------------------
_BAILING_V3_ARCHS = ("BailingMoeV3ForCausalLM", "BailingHybridForCausalLM")


def detect_out_gate_headwise(cfg, model_type):
    """'head' when the attention output gate is ONE sigmoid / softplus scalar per head broadcast over head_dim."""
    lm = lm_of(cfg)
    mt = model_type
    g = lm.get("gating", True)
    if mt in ("laguna", "step3p7"):                       # tf5 modular_laguna.py:289, modeling_step3p7.py:708
        return "head" if (g is True or _s(g) == "per-head") else None
    if mt == "youtu_vl_moe":                              # modeling_youtu_vl_moe.py:884-891 (default head_wise)
        if truthy_flag(lm.get("attention_output_gate")) and _s(lm.get("mla_output_gate_type", "head_wise")) == "head_wise":
            return "head"
        return None
    if _s(lm.get("gated_attention_proj_granularity_type")) == "head_wise":   # BailingMoeV3 MLA (Ming / Edge0 / Ling-3)
        return "head"
    if mt in ("spark2_5", "motif") and lm.get("headwise_attn_output_gate") is True:   # modeling_spark.py:154
        return "head"
    if mt == "step3p5" and lm.get("use_head_wise_attn_gate") is True:      # modeling_step3p5.py:458-462
        return "head"
    if mt in ("ethoric", "slm_tavenel") and _s(lm.get("attn_gate", "headwise")) == "headwise":  # attention.py:168
        return "head"
    return None


# --- out_gate_activation -----------------------------------------------------------------------------------------
def detect_out_gate_activation(cfg, model_type):
    """The output gate's activation when it is NOT sigmoid: 'softplus' (Laguna softplus(Wx); K2 softplus(Wx, beta=ln 2)
    = softplus(ln2 Wx) / ln2, the same function class) or 'silu'. Sigmoid gates and ungated configs return None."""
    lm = lm_of(cfg)
    mt = model_type
    if mt == "laguna":                                    # tf5 modular_laguna.py:335; Laguna-XS-2.1 :459 (gating False: none)
        return "softplus" if lm.get("gating", True) is not False else None
    if mt == "k2_horizon":                                # modeling_k2_horizon.py:225-230, 312-319
        f = _s(lm.get("attention_gate_func"))
        return f if f in ("softplus", "silu") else None
    if mt == "spark2_5" and lm.get("headwise_attn_output_gate") is True:   # modeling_spark.py:199-202
        return "silu" if _s(lm.get("gate_attn_act_mode")) == "silu" else None
    return None


# --- sink_key_query_dependent ------------------------------------------------------------------------------------
def detect_sink_key_query_dependent(cfg, model_type):
    """'key' when the learned sink is a zero-value KEY (logit q . s * scale, q normed and rotated, s unrotated) instead
    of a per-head scalar logit: IQuest-Q1 (vllm-iquest-q1 model.py:354-365, attention.py:66-89)."""
    lm = lm_of(cfg)
    if model_type == "iquest_q1" and lm.get("enable_sink_attention", False) is True:
        return "key"
    return None


# --- vha_head_mix ------------------------------------------------------------------------------------------------
def detect_vha_head_mix(cfg, model_type):
    """True for ERNIE-5.5: every attention layer mixes its head outputs o <- (I + V U^T) o (vLLM-ERNIE attention.py:110-146,
    built unconditionally at :195 / :831; use_vha_attention in the config)."""
    lm = lm_of(cfg)
    if model_type == "ernie5_5_moe" and lm.get("use_vha_attention", True) is not False:
        return True
    return None


# --- diff_attn ---------------------------------------------------------------------------------------------------
def detect_diff_attn(cfg, model_type):
    """True for Motif GDLA differential attention: out_s = a_s - sigmoid(lambda_proj(x))_s * a_noise(group(s))
    (Motif-3 modeling_motif.py:555-557, 687, 754-766)."""
    lm = lm_of(cfg)
    if model_type == "motif" and _s(lm.get("attention_cls", "basic")) == "gdla" and num(lm.get("num_noise_heads")) > 0:
        return True
    return None


# --- xsa_self_value_projection -----------------------------------------------------------------------------------
def detect_xsa_self_value_projection(cfg, model_type):
    """Exclusive self-attention: each head output loses its component along the token's own value vector.
    'unit' = y - (y . v_hat) v_hat always (GPT-S / GPT-X2.5 xsa_projection, Grug-MoE unconditionally);
    'scaled' = y - tanh(alpha_h) (y . v_hat) v_hat with a learned per-head alpha (Limite xsa + xsa_layers)."""
    lm = lm_of(cfg)
    mt = model_type
    if mt in ("gpts2", "gptx2") and lm.get("xsa_projection") is True:   # modeling_gpts2.py:91-97, modeling_gptx2.py:106-111
        return "unit"
    if mt == "grug_moe":                                                # skyrl_train_models_grug_moe.py:743-745 (no switch)
        return "unit"
    if mt == "cyrah" and lm.get("use_xsa", True) is True:               # modeling_cyrah.py:369, 484
        return "unit"
    if mt == "limite" and lm.get("xsa", True) is True and lm.get("xsa_layers"):   # modeling_limite.py:187, 367-395
        return "scaled"
    return None


# --- canon_conv_layers -------------------------------------------------------------------------------------------
def detect_canon_conv_layers(cfg, model_type):
    """Canon short causal depthwise convs (Allen-Zhu, Physics of LMs 4.1) inside the block: '<positions>/k<kernel>'
    with positions from A (attention input), B (q / k / v after projection), C (MLP input), D (MLP hidden).
    dhara_ar canon_set / canon_kernel (modeling_dhara_ar.py:46-100, 276-338, 417-434, 449-492); Ethoric
    canon.{enabled, positions, kernel} (modeling_ethoric.py:38-74, 126-159); needle extras.qkv_conv_taps (position B,
    architecture.py:184-203; absent = 0 = off, config.py:85)."""
    lm = lm_of(cfg)
    mt = model_type
    if mt == "dhara_ar":
        cs = lm.get("canon_set")
        pos = "".join(sorted(set(str(cs).upper()) & set("ABCD"))) if isinstance(cs, str) else ""
        k = num(lm.get("canon_kernel", 4))
        return f"{pos}/k{k}" if pos and k > 1 else None
    if mt == "ethoric":
        c = lm.get("canon") if isinstance(lm.get("canon"), dict) else (cfg.get("canon") if isinstance(cfg.get("canon"), dict) else {})
        if not c.get("enabled", False):
            return None
        pos = "".join(sorted(set(str(p).upper() for p in (c.get("positions") or []) if isinstance(p, str)) & set("ABCD")))
        k = num(c.get("kernel", 4))
        return f"{pos}/k{k}" if pos and k > 1 else None
    if mt == "needle":
        ex = cfg.get("extras") if isinstance(cfg.get("extras"), dict) else {}
        k = num(ex.get("qkv_conv_taps", lm.get("qkv_conv_taps", 0)))
        return f"B/k{k}" if k > 1 else None
    return None


# --- cca_conv_qk_attention ---------------------------------------------------------------------------------------
def detect_cca_conv_qk_attention(cfg, model_type):
    """True for ZAYA1 compressed convolutional attention (CCA): q / k pass a 2-stage causal conv over time plus a
    (q + k) / 2 mean residual, half of v reads the PREVIOUS token, q / k are L2-normed to sqrt(head_dim) per head and
    k is scaled by a learned per-kv-head temperature (ZAYA1-VL modeling_zaya1_vl.py:676-759, 842-888; built for every
    layer at :961; tf5 models/zaya/modeling_zaya.py:154-170)."""
    lm = lm_of(cfg)
    if model_type in ("zaya1_vl", "zaya") and lm.get("cca", True) is not False:
        return True
    return None


# --- moa_routed_attention_heads ----------------------------------------------------------------------------------
def detect_moa_routed_attention_heads(cfg, model_type):
    """'E<experts>k<top_k>' for JetMoE mixture-of-attention: a top-k softmax router picks k of E (q_proj, o_proj) expert
    pairs per token over SHARED k / v heads (tf5 modular_jetmoe.py:212-282, 288-373; num_attention_heads =
    num_key_value_heads * top_k, configuration_jetmoe.py:73)."""
    lm = lm_of(cfg)
    if model_type != "jetmoe":
        return None
    e = num(lm.get("num_local_experts", lm.get("moe_num_experts", 8)))
    k = num(lm.get("num_experts_per_tok", lm.get("moe_top_k", 2)))
    return f"E{e}k{k}" if e > 1 and k >= 1 else None


# --- memory_attention --------------------------------------------------------------------------------------------
def detect_memory_attention(cfg, model_type):
    """A second attention path that reads COMPRESSED / SELECTED earlier-chunk memory: 'summary' = learned-query softmax
    pools of each earlier chunk (haru cfrd CausalSummaryMemory with chunk-local attention, model.py:201-445; Bluebird
    fabric ChunkSummarizer + MemoryAttention on every 3rd layer, fabric_runtime.py:209-296, 310-311); 'topk_kv' =
    admission-scored top-capacity keys of all earlier chunks (EdgeRock IndependentGQAMemory, modeling_edgerock.py:51-80,
    122-322)."""
    lm = lm_of(cfg)
    mt = model_type
    if mt == "cfrd":                                   # haru_2 model.py:515-521: cell_attention "full" drops the memory
        return "summary" if _s(lm.get("cell_attention", "local")) != "full" else None
    if mt == "fabric":
        return "summary" if _s(lm.get("architecture", "fabric")) == "fabric" else None
    if mt == "edgerock":
        if lm.get("fixed_memory_swa_layers") and (isinstance(lm.get("independent_memory"), dict)
                                                  or "independent" in _s(lm.get("fixed_memory_revision"))):
            return "topk_kv"
    return None


# --- k_eq_v_by_code ----------------------------------------------------------------------------------------------
def detect_k_eq_v_by_code(cfg, model_type):
    """True for DiffusionGemma: v_proj = None on every non-sliding layer BY CODE (V = v_norm(k_proj x)); the config
    deletes attention_k_eq_v (tf5 modular_diffusion_gemma.py:101, 258-262, 293, 360-364, 395)."""
    lm = lm_of(cfg)
    if not model_type.startswith("diffusion_gemma"):
        return None
    lt = layer_types(lm)
    return True if (not lt or any(x != "sliding_attention" for x in lt)) else None


# --- grouped_lowrank_oproj ---------------------------------------------------------------------------------------
_GROUPED_O_TYPES = ("deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4", "ernie5_5_moe")


def detect_grouped_lowrank_oproj(cfg, model_type):
    """'g<o_groups>r<o_lora_rank>' for the grouped low-rank output projection o_b(blockdiag_g(o_a)(o)): heads split into
    o_groups groups, each projected to o_lora_rank, then mixed to hidden (tf5 modular_deepseek_v4.py:278-307, 697-700,
    775-777; V4.1 inference/model.py:71-76, 634; ERNIE-5.5 vLLM attention.py:176-185, 230-243)."""
    lm = lm_of(cfg)
    if model_type not in _GROUPED_O_TYPES:
        return None
    g, r = num(lm.get("o_groups")), num(lm.get("o_lora_rank"))
    return f"g{g}r{r}" if g >= 1 and r > 0 else None


# ==================== batch a8_mla_dsv4 ====================

# ---- mla_k_rope_norm -----------------------------------------------------------------------------------------------
# dots3_note: Dots3NoteTextAttention builds k_rope_only_layernorm (a weighted RMSNorm over the shared qk_rope_head_dim
# key) unconditionally, on every layer (full and sliding share the class). New qknorm variant "mla_krope".
def detect_mla_k_rope_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "dots3_note" and lm.get("kv_lora_rank"):
        return "mla_krope"
    return None


# ---- mla_query_raw_latent_skip -------------------------------------------------------------------------------------
# A.X-K2: q_gate_proj reads cat(q_a_layernorm(q_a x), q_a x): the query also sees the UN-normed q latent.
def detect_mla_query_raw_latent_skip(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "axk2" and lm.get("kv_lora_rank") and num(lm.get("q_lora_rank")) > 0:
        return True
    return None


# ---- step3_mfa_query_latent_norm -----------------------------------------------------------------------------------
# Step-3 MFA: q = wq(inter_norm(q_proj x)), q_proj hidden -> share_q_dim (Step3TextConfig default 2048), weighted
# RMSNorm on it. The new dims value: the normed query-latent rank.
def detect_step3_mfa_query_latent_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in ("step3_text", "step3_vl"):
        return num(lm.get("share_q_dim")) or 2048
    return None


# ---- hybrid_mla_layers_unread --------------------------------------------------------------------------------------
# sabi-noema-mla: text_config.shared_mla (enabled) turns the full-attention layers it lists into a shared pre-RoPE MLA
# (no latent norm, per-head q_norm before learned rope / residual query mixes, mean_v = a V bias). Returns the MLA dims.
_SABI_DEFAULT = {"kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 384, "v_head_dim": 256}


def detect_hybrid_mla_layers_unread(cfg, model_type):
    lm = lm_of(cfg)
    sm = lm.get("shared_mla")
    if not isinstance(sm, dict) or sm.get("enabled", True) is False:
        return None
    blk = {**_SABI_DEFAULT, **sm}
    layers = [i for i in (blk.get("layers") or []) if isinstance(i, int)]
    lt = lm.get("layer_types") if isinstance(lm.get("layer_types"), list) else []
    full = [i for i, t in enumerate(lt) if t == "full_attention"]
    if not layers:
        return None
    return {"kv": "MLA" if set(full) <= set(layers) else "GQA|MLA",
            "kv_latent": num(blk["kv_lora_rank"]), "q_latent": 0, "mla_rope": num(blk["qk_rope_head_dim"]),
            "mla_nope": num(blk["qk_nope_head_dim"]), "mla_v": num(blk["v_head_dim"]),
            "mla_layers": len(layers)}


# ---- ernie_mla_layer_dims_unread -----------------------------------------------------------------------------------
# ERNIE-5.5: compress_ratios[i] <= 0 marks the DSA / MLA layers (Ernie5_5Config.is_dsa_layer), whose dims are the
# hybrid_mla_* keys (Ernie5_5Config.mla_* properties); the top-level qk_nope / v / head_dim are the HCA layers'.
def detect_ernie_mla_layer_dims_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "ernie5_5_moe":
        return None
    cr = lm.get("compress_ratios")
    if not (isinstance(cr, list) and any(isinstance(x, (int, float)) and x <= 0 for x in cr)):
        return None
    g = lambda k, dflt: num(lm.get(k)) if lm.get(k) is not None else dflt     # Ernie5_5Config defaults
    return {"heads": g("hybrid_mla_num_attention_heads", 64), "q_latent": g("hybrid_mla_q_lora_rank", 1024),
            "kv_latent": g("hybrid_mla_kv_lora_rank", 512), "mla_nope": g("hybrid_mla_qk_nope_head_dim", 192),
            "mla_rope": g("hybrid_mla_qk_rope_head_dim", 64), "mla_v": g("hybrid_mla_v_head_dim", 256)}


# ---- kv_compress_variant -------------------------------------------------------------------------------------------
# which compressor code builds the pools: "dsv4" (tf5 DeepseekV4 CSA ratio-4 overlapping windows + position_bias,
# HCA ratio-128 + position_bias; ERNIE-5.5 runs vLLM's DeepSeek-V4 compressor at ratio 128) vs "dsv41" (DeepSeek-V4.1
# inference/model.py: non-overlapping ratio-2 pools with no position bias, ratio-1 per-token norm(wkv x) keys, pools
# built only on kv_source_layers and read by the later layers).
_DSV4_SCHEMA_TYPES = ("deepseek_v4", "memt_deepseek_v4")


def _dsv4_has_compressed(lm):
    cr = lm.get("compress_ratios")
    lt = lm.get("layer_types")
    if isinstance(lt, list) and lt:
        return any("compressed" in str(x) for x in lt)
    if isinstance(cr, list):
        return any(num(x) > 0 for x in cr)
    return True                         # DeepseekV4Config default: all-compressed attention


def detect_kv_compress_variant(cfg, model_type):
    lm = lm_of(cfg)
    cr = lm.get("compress_ratios")
    if model_type in ("deepseek_v41", "deepseek_v41_text"):
        return "dsv41" if isinstance(cr, list) and any(num(x) > 0 for x in cr) else None
    if model_type in _DSV4_SCHEMA_TYPES:
        return "dsv4" if _dsv4_has_compressed(lm) else None
    if model_type == "ernie5_5_moe":
        return "dsv4" if isinstance(cr, list) and any(num(x) > 1 for x in cr) else None
    return None


# ---- dsv4_layer_types_schema ---------------------------------------------------------------------------------------
# tf5 DeepseekV4Config (and MemT's subclass): layer_types / mlp_layer_types are the schedule; the legacy
# compress_ratios / num_hash_layers only seed them when absent; neither -> defaults (all-compressed, 3 hash layers).
# Flags the configs whose schedule is NOT readable from the legacy keys; returns the features it implies.
def detect_dsv4_layer_types_schema(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in _DSV4_SCHEMA_TYPES:
        return None
    feats = []
    lt, mlt = lm.get("layer_types"), lm.get("mlp_layer_types")
    if lm.get("compress_ratios") is None or (isinstance(lt, list) and lt):
        n = num(lm.get("num_hidden_layers"))
        if isinstance(lt, list) and lt:
            comp = any("compressed" in str(x) for x in lt[:n or None])
        else:
            comp = n > 0                                  # default schedule: every layer compressed
        if comp:
            feats.append("kv_compress")
    if lm.get("num_hash_layers") is None or (isinstance(mlt, list) and mlt):
        n = num(lm.get("num_hidden_layers"))
        if isinstance(mlt, list) and mlt:
            hashed = any(str(x) == "hash_moe" for x in mlt[:n or None])
        else:
            hashed = n > 0                                # default_num_hash_layers = 3
        if hashed:
            feats.append("hash_route")
    return "+".join(feats) or None


# ---- hyper_conn_variant --------------------------------------------------------------------------------------------
# the hyper-connection code a config runs (one value per code, compared like the qknorm variants)
HC_VARIANT_BY_TYPE = {"deepseek_v4": "dsv4", "memt_deepseek_v4": "dsv4", "ernie5_5_moe": "dsv4",
                      "glm5_next": "glm5_next", "glm5_next_text": "glm5_next",
                      "deepseek_v41": "dsv41", "deepseek_v41_text": "dsv41",
                      "xing4_0": "xing", "hy_v4": "hy4", "qwen4_exp": "qwen4_exp", "qwen4_exp_text": "qwen4_exp",
                      "motif": "motif", "needle": "needle", "san": "san"}


def _hc_on(lm, cfg):
    ex = lm.get("extras") if isinstance(lm.get("extras"), dict) else {}
    return (num(lm.get("hc_mult")) > 1 or num(lm.get("hc_count")) > 1 or lm.get("mhc") or lm.get("mhc_enabled")
            or num(lm.get("mhc_expansion_rate")) > 1 or num(lm.get("hc_rate")) > 1 or lm.get("hc_magnitude")
            or num(lm.get("mhc_lanes")) > 1 or num(ex.get("mhc_lanes")) > 1)


def detect_hyper_conn_variant(cfg, model_type):
    lm = lm_of(cfg)
    if not _hc_on(lm, cfg) or model_type not in HC_VARIANT_BY_TYPE:
        return None
    if model_type == "deepseek_v4" and "modeling_deepseek_v4" in auto_map(cfg):
        return "dsv4_nano"              # nanowhale remote code: comb (not comb^T), its own Sinkhorn
    if model_type == "ernie5_5_moe" and lm.get("enable_hyper_connections") is False:
        return None
    return HC_VARIANT_BY_TYPE[model_type]


# ---- hyper_conn_key_unread -----------------------------------------------------------------------------------------
# needle (extras.mhc_lanes; the cactus TransformerConfig default is 4) and its PyTorch port SAN (top-level mhc_lanes,
# default 4): an mhc_lanes-lane manifold-constrained hyper-connection. Returns the lane count.
def detect_hyper_conn_key_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "needle":
        ex = lm.get("extras") if isinstance(lm.get("extras"), dict) else {}
        n = num(ex.get("mhc_lanes")) if "mhc_lanes" in ex else 4
        return n if n > 1 else None
    if model_type == "san":
        n = num(lm.get("mhc_lanes")) if "mhc_lanes" in lm else 4
        return n if n > 1 else None
    return None


# ==================== batch a9_router ====================

def _a9_s(v):
    return str(v or "").strip().lower()


def _n_experts(lm):
    """Routed experts of a MoE LM config (0 for a dense one)."""
    for k in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts", "num_routed_experts"):
        v = num(lm.get(k))
        if v:
            return v
    return 0


def _top_k(lm):
    for k in ("num_experts_per_tok", "moe_top_k", "moe_topk", "moe_k", "top_k", "num_experts_per_token",
              "experts_top_k", "num_selects"):
        v = num(lm.get(k))
        if v:
            return v
    return 0


# BailingMoeV2 / V2.5 / V3 gates and their forks (LLaDA2-MoE, Param2-MoE, Ming-3): scores / sum when top_k > 1,
# norm_topk_prob never read. bailing_moe v1 (BailingMoeForCausalLM) honours norm_topk_prob, so this is by architecture.
_BAILING_V2_ARCHS = ("BailingMoeV2ForCausalLM", "BailingMoeV2_5ForCausalLM", "BailingMoeV3ForCausalLM",
                     "LLaDA2MoeModelLM", "Param2MoEForCausalLM", "MingV3ASRForConditionalGeneration")
# softmax (or sigmoid) top-k, always renormalised, no key read
_RENORM_R_TYPES = ("ceno", "alice_ai", "zaqeris_axiom", "gemma3_moe", "iquest_q1", "mistral_large4",
                   "mistral_large4_text")
# softmax over all experts, top-k, never renormalised, no key read
_RENORM_P_TYPES = ("bananamind2_moe", "step3_text", "step3_vl", "molmo")


def detect_renorm_by_code_unlisted(cfg, model_type):
    """The top-k weight renormalisation the CODE applies ('R' / 'P'; 'key:R' / 'key:P' when the code reads a key
    p5 does not list; 'unit' for Meditor's straight-through weight at k > 1, 'raw' for LLaMA-MoE without softmax)."""
    lm = lm_of(cfg)
    mt = model_type
    if not _n_experts(lm):
        return None
    k = _top_k(lm)
    if any(a in _BAILING_V2_ARCHS for a in archs(cfg)):
        return "R" if k > 1 else "P"
    if mt in _RENORM_R_TYPES:
        return "R"
    if mt in _RENORM_P_TYPES:
        return "P"
    if mt == "arctic":
        return "R" if k > 1 else "P"
    if mt == "meditor":
        return "R" if k <= 1 else "unit"
    if mt == "llama_moe":
        return "R" if lm.get("gate_use_softmax", True) else "raw"
    if mt == "motif" and lm.get("route_norm") is not None:
        return "key:R" if lm.get("route_norm") else "key:P"
    return None


def detect_router_score_unread(cfg, model_type):
    """'<score>/<score_cap>' of a MoE router whose code fixes sigmoid scoring with no key p5 reads, or whose code
    takes the score from a config key p5 does not list in SCORE_CONFIG_TYPES."""
    lm = lm_of(cfg)
    mt = model_type
    if not _n_experts(lm):
        return None
    if mt in ("beam", "sarvam_mla", "grug_moe", "muse_spark", "muse_spark_text", "kolibri1"):
        return "sigmoid/sigmoid"
    if mt == "deepseek2" and "MarkV3ForCausalLM" in archs(cfg):
        return "sigmoid/sigmoid"
    if mt == "k2_horizon":
        sc = _a9_s(lm.get("router_score_func")) or "softmax"
        return sc + "/sigmoid|softmax"
    if mt == "motif":
        sc = _a9_s(lm.get("score_func")) or "softmax"
        return sc + "/sigmoid|softmax"
    if mt == "kohaku":
        sc = _a9_s(lm.get("scoring_func")) or "sigmoid"
        return sc + "/sigmoid|softmax|sqrtsoftplus"
    return None


def detect_router_bias_unread(cfg, model_type):
    """The domain of a selection-only expert bias p5 does not see: 'score' (top-k on score + b, DeepSeek-V3 style)
    or 'logit' (top-k on logit + b, weights from the unbiased score)."""
    lm = lm_of(cfg)
    mt = model_type
    if not _n_experts(lm):
        return None
    if mt == "youtu_vl_moe":
        if not lm.get("moe_router_enable_expert_bias", True):
            return None
        qb = (_a9_s(lm.get("moe_router_expert_bias_update_method")) == "quantile"
              or _a9_s(lm.get("topk_method")) in ("quantile_balancing", "quantile"))
        return "logit" if qb else "score"
    if mt == "motif":
        return "score" if lm.get("load_balance_coeff") is not None else None
    if mt == "alice_ai":
        return "score" if lm.get("router_bias_correction") else None
    if mt == "deepseek2" and "MarkV3ForCausalLM" in archs(cfg):
        return "score"
    if mt in ("kohaku", "beam", "muse_spark", "muse_spark_text"):
        return "score"
    if mt in ("grug_moe", "kolibri1"):
        return "logit"
    return None


def detect_group_mask_zero_fill(cfg, model_type):
    """'top2sum_zero': a DeepSeek-V3-style top-2-sum group limit whose code fills the excluded experts' BIASED scores
    with 0.0 (not -inf). Per repo code: tf5 A.X-K2, Kimi-K2 / JoyAI remote code, and the first-release
    modeling_deepseek.py (its config still carries seq_aux, which that MoEGate reads)."""
    lm = lm_of(cfg)
    mt = model_type
    if not _n_experts(lm):
        return None
    if mt == "axk2":
        return "top2sum_zero"
    am = auto_map(cfg)
    remote_ds = "modeling_deepseek" in am or "modeling_kimi_k25" in am
    if mt == "joyai_llm_flash" and remote_ds:
        return "top2sum_zero"
    if (mt in ("deepseek_v3", "kimi_k2") and remote_ds and "seq_aux" in lm
            and _a9_s(lm.get("topk_method")) == "noaux_tc"):
        return "top2sum_zero"
    return None


def detect_router_group_balanced(cfg, model_type):
    """'per_group': Intern-S1-Pro's grouped router (router_n_groups > 1): top-(k / G) inside EVERY group."""
    lm = lm_of(cfg)
    if not _n_experts(lm):
        return None
    if num(lm.get("router_n_groups")) > 1 and model_type.startswith("interns1_pro"):
        return "per_group"
    return None


def detect_dual_sigmoid_router_sum(cfg, model_type):
    """'sigmoid2': ERNIE-5.5 split-feature routing, score = sigmoid(l0) + sigmoid(l1)."""
    lm = lm_of(cfg)
    if not _n_experts(lm):
        return None
    if model_type == "ernie5_5_moe" and lm.get("moe_split_feature_routing"):
        return "sigmoid2"
    return None


def detect_hash_route_by_code(cfg, model_type):
    """True: the 'deep' family's token-routed MLP (experts from a token-id table, constant route weights)."""
    lm = lm_of(cfg)
    if model_type != "deep" or not _n_experts(lm):
        return None
    if _a9_s(lm.get("mlp_type", "token_routed")) != "token_routed":
        return None
    return True


def detect_mlp_router(cfg, model_type):
    """'tanh_mlp': LLaMA-MoE's TopKBalancedNoisyGate with gate_network 'mlp' (logits = W2 tanh(W1 x))."""
    lm = lm_of(cfg)
    if model_type != "llama_moe" or not _n_experts(lm):
        return None
    if _a9_s(lm.get("gate_type", "TopKBalancedNoisyGate")) != "topkbalancednoisygate":
        return None
    return "tanh_mlp" if _a9_s(lm.get("gate_network", "mlp")) == "mlp" else None


def detect_moe_capacity_drop(cfg, model_type):
    """True: Hunyuan-A13B's remote topkgating (expert capacity max(k, k S / E) per forward, over-capacity
    assignments dropped; moe_drop_tokens not read). The native hunyuan_v1_moe code has no capacity."""
    lm = lm_of(cfg)
    if model_type != "hunyuan_v1_moe" or not _n_experts(lm):
        return None
    am = auto_map(cfg)
    return True if "hunyuan.HunYuan" in am else None


def detect_block_routing(cfg, model_type):
    """True: LLaDA2.2's block routing (per block of block_size tokens only expert_capacity experts allowed)."""
    lm = lm_of(cfg)
    E = _n_experts(lm)
    if model_type != "llada2_moe" or not E:
        return None
    cap, bs = num(lm.get("expert_capacity")), num(lm.get("block_size"))
    if cap and bs and cap < E:
        return True
    return None


# ==================== batch a10_moe_ffn ====================

# ---- gemma4_branch_post_norms --------------------------------------------------------------------------------------
def detect_gemma4_branch_post_norms(cfg, model_type):
    """Gemma-4 enable_moe_block / DiffusionGemma: post_ffn_norm(post_ffn_norm_1(mlp(.)) + post_ffn_norm_2(experts(.))),
    each FFN branch post-normed on its own, the sum normed again."""
    lm = lm_of(cfg)
    if model_type.startswith("gemma4") and lm.get("enable_moe_block") is True and num(lm.get("num_experts")) > 1:
        return "branch_norm"
    if model_type.startswith("diffusion_gemma") and num(lm.get("num_experts")) > 1:
        return "branch_norm"                  # DiffusionGemma*TextLayer: the same branch norms, unconditionally
    return None


# ---- mova_routed_values --------------------------------------------------------------------------------------------
def detect_mova_routed_values(cfg, model_type):
    """K2-Horizon MoVA attention on the sparse layers: V = sum_k w_k silu(W_k x) over mova_num_experts routed value
    experts. Returns [value experts, top-k]."""
    lm = lm_of(cfg)
    if model_type not in ("k2_horizon", "jais_asr"):
        return None
    e, k = num(lm.get("mova_num_experts")), num(lm.get("mova_num_experts_per_tok"))
    if e > 0 and num(lm.get("num_experts")) > 0:          # MoVA is built only on sparse (MoE) layers
        return [e, k]
    return None


# ---- kimi_latent_moe_norm ------------------------------------------------------------------------------------------
def detect_kimi_latent_moe_norm(cfg, model_type):
    """A latent MoE with an RMSNorm in the latent: ["sum", width] = the norm on the weighted routed sum before the up
    projection (Kimi-K3 routed_expert_norm, ERNIE-5.5 latent_norm); ["in", width] = the norm on the latent expert input
    (Muse-Spark pre_expert_norm)."""
    lm = lm_of(cfg)
    if model_type == "kimi_linear":
        w = num(lm.get("routed_expert_hidden_size"))
        if w and lm.get("latent_moe_use_norm") is True:
            return ["sum", w]
        return None
    if model_type == "ernie5_5_moe":
        w = num(lm.get("moe_latent_size"))
        if w and lm.get("latent_moe_use_norm") is not False:   # vLLM-ERNIE builds latent_norm unconditionally
            return ["sum", w]
        return None
    if model_type in ("muse_spark", "muse_spark_text"):
        return ["in", num(lm.get("moe_hidden_size")) or 4096]  # MuseSparkMoE: pre_expert_norm always built
    return None


# ---- expert_output_norm --------------------------------------------------------------------------------------------
def detect_expert_output_norm(cfg, model_type):
    """Muse-Spark: post_expert_norm = rms((1 + w) y) on every routed expert's output before the routing weight."""
    if model_type in ("muse_spark", "muse_spark_text"):
        return "input_scaled_rms"
    return None


# ---- identity_experts ----------------------------------------------------------------------------------------------
def detect_identity_experts(cfg, model_type):
    """LongCat zero-computation experts: zero_expert_num router slots whose expert is the identity (summed with the
    SwiGLU experts under one softmax top-k). Returns the identity-slot count."""
    lm = lm_of(cfg)
    z = num(lm.get("zero_expert_num"))
    if z > 0 and str(lm.get("zero_expert_type") or "identity").lower() == "identity":
        return z
    return None


# ---- shared_gate_undetected ----------------------------------------------------------------------------------------
SHARED_GATE_CODE = {"qwen2_moe", "qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_omni_moe",
                    "qwen4_exp", "qwen4_exp_text",
                    "alice_ai",            # AliceAISparseMoEBlock.shared_expert_gate
                    "memt_qwen3_5_moe"}    # MemTDecoderLayer.mlp = Qwen3_5MoeSparseMoeBlock


def detect_shared_gate_undetected(cfg, model_type):
    """A sigmoid-gated shared expert in code: shared(x) * sigmoid(w . x)."""
    lm = lm_of(cfg)
    if model_type in SHARED_GATE_CODE and (num(lm.get("shared_expert_intermediate_size")) > 0
                                           or model_type == "qwen3_next"):
        return True
    return None


# ---- shared_expert_code_fixed_unread -------------------------------------------------------------------------------
def detect_shared_expert_code_fixed_unread(cfg, model_type):
    """Llama4TextMoe always builds shared_expert = Llama4TextMLP(config): width intermediate_size."""
    lm = lm_of(cfg)
    if model_type in ("llama4", "llama4_text") and num(lm.get("num_local_experts")) > 1:
        return num(lm.get("intermediate_size")) or 8192
    return None


# ---- dense_first_key_unread ----------------------------------------------------------------------------------------
def detect_dense_first_key_unread(cfg, model_type):
    """Kohaku: a dense MLP (intermediate_size) on layers < first_k_dense (class default 1), MoE after."""
    lm = lm_of(cfg)
    if model_type == "kohaku":
        k = lm.get("first_k_dense", 1)
        return num(k) if num(k) > 0 else None
    return None


# ---- hierarchical_moe_unread ---------------------------------------------------------------------------------------
def detect_hierarchical_moe_unread(cfg, model_type):
    """Meditor-2 HMoE: shared SwiGLU + two-level hard top-1 (macro domain argmax, then sub-expert argmax) with the
    constant weight routed_scale. Returns [domains, sub-experts per domain, sub-expert width]."""
    lm = lm_of(cfg)
    if model_type != "meditor2":
        return None
    d = num(lm.get("num_macro_domains", 3))
    s = num(lm.get("num_sub_experts_per_domain", 5))
    w = num(lm.get("sub_intermediate_size", 768))
    return [d, s, w] if d * s > 1 else None


# ---- arctic_dense_residual_parallel_moe ----------------------------------------------------------------------------
def detect_arctic_dense_residual_parallel_moe(cfg, model_type):
    """Arctic parallel_attn_mlp_res: h = x + attn(n1 x); out = h + residual_mlp(n_r h) + moe(n2 x) on MoE layers.
    Returns the residual MLP width (hidden_size)."""
    lm = lm_of(cfg)
    if model_type == "arctic" and lm.get("parallel_attn_mlp_res") is True and num(lm.get("num_local_experts")) > 1:
        return num(lm.get("hidden_size"))
    return None


# ---- act_clip_order ------------------------------------------------------------------------------------------------
_CLIP_PRE = {"deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4", "glm5_next", "glm5_next_text",
             "hy_v4"}                      # silu(min(g, L)) * clamp(u, -L, L)
_CLIP_POST = {"step3p5", "step3p7", "bailing_hybrid"}   # min(silu(g), L) * clamp(u, -L, L)


def _clip_limits(lm):
    vals = []
    for k in ("swiglu_limit", "swiglu_limits", "swiglu_limits_shared", "expert_swiglu_limit_list",
              "share_expert_swiglu_limit_list"):
        v = lm.get(k)
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            f = fnum(x)
            if f:
                vals.append(f)
    return vals


def detect_act_clip_order(cfg, model_type):
    """Where a clamped SwiGLU clamps the gate: 'pre' (before SiLU) or 'post' (after SiLU)."""
    lm = lm_of(cfg)
    if not _clip_limits(lm):
        return None
    if model_type in _CLIP_PRE:
        return "pre"
    if model_type in _CLIP_POST:
        return "post"
    return None


# ---- act_clip_per_layer_list ---------------------------------------------------------------------------------------
def detect_act_clip_per_layer_list(cfg, model_type):
    """Ling-3 (BailingMoeV3): per-layer clamps from expert_swiglu_limit_list / share_expert_swiglu_limit_list
    (0 = off). Returns the largest limit (a nonzero entry exists)."""
    lm = lm_of(cfg)
    if model_type != "bailing_hybrid":
        return None
    vals = []
    for k in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"):
        v = lm.get(k)
        if isinstance(v, (list, tuple)):
            vals += [fnum(x) for x in v if fnum(x)]
    return num(max(vals)) if vals else None


# ==================== batch a11_recurrent_loops ====================

# ---- gdn_out_gate_act ----------------------------------------------------------------------------------------------
# Qwen4-exp GatedDeltaNet: o = RMSNorm(core) * ACT2FN[output_gate_type or hidden_act](z) (Qwen3-Next / 3.5: silu
# fixed). Returns the gate activation when it is not silu (the new GDN variant "gdn_sig").
def detect_gdn_out_gate_act(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in ("qwen4_exp", "qwen4_exp_text"):
        return None
    lt = layer_types(lm)
    if lt and "linear_attention" not in lt:
        return None                                         # no GatedDeltaNet layer
    if not lt and not lm.get("linear_num_value_heads"):
        return None                                         # no text-LM recurrent keys (a bare / foreign config)
    act = str(lm.get("output_gate_type") or lm.get("hidden_act") or "silu").lower()
    return act if act != "silu" else None


# ---- kda_classified_as_gdn -----------------------------------------------------------------------------------------
# AliceAI (remote modeling_alice_ai.py): the linear_attention layers are AliceAIKDA (per-channel decay, low-rank gate,
# fla chunk_kda). Returns the recurrent family "kda" ("kda_negeig" if kda_allow_negative_eigenvalues: beta x 2).
def detect_kda_classified_as_gdn(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "alice_ai":
        return None
    lt = layer_types(lm)
    if lt and "linear_attention" not in lt:
        return None
    return "kda_negeig" if lm.get("kda_allow_negative_eigenvalues") is True else "kda"


# ---- fla_num_heads_unplaced ----------------------------------------------------------------------------------------
# flash-linear-attention configs (fla-hub): heads are num_heads / num_kv_heads, never num_attention_heads, so the
# softmax transformer and DeltaNet are dropped. Returns the head count classify should read.
FLA_ARCHS = ("TransformerForCausalLM", "DeltaNetForCausalLM", "GLAForCausalLM")


def detect_fla_num_heads_unplaced(cfg, model_type):
    lm = lm_of(cfg)
    if not any(a in FLA_ARCHS for a in archs(cfg)):
        return None
    if lm.get("num_attention_heads") or not num(lm.get("num_heads")):
        return None
    return num(lm.get("num_heads"))


# ---- gla_rc_dims_unread --------------------------------------------------------------------------------------------
# fla GatedLinearAttention / DeltaNet: key / value widths hidden * expand_k / expand_v split over num_heads, GQA by
# num_kv_heads (GLA), short conv only with use_short_conv, GLA gate rank fixed 16 (gla.py default; modeling_gla.py
# does not pass it). Returns [rc_h, rc_k, rc_v, rc_kh, rc_conv, rc_rank].
def detect_gla_rc_dims_unread(cfg, model_type):
    lm = lm_of(cfg)
    a = archs(cfg)
    if model_type == "gla" and "GLAForCausalLM" in a:
        ek, ev, short, rank = fnum(lm.get("expand_k"), 0.5), fnum(lm.get("expand_v"), 1.0), False, 16
    elif model_type == "delta_net" and "DeltaNetForCausalLM" in a:
        ek, ev, short, rank = fnum(lm.get("expand_k"), 1.0), fnum(lm.get("expand_v"), 1.0), True, 0
    else:
        return None
    hidden, H = num(lm.get("hidden_size")), num(lm.get("num_heads")) or 4
    if not hidden:
        return None
    kh = (num(lm.get("num_kv_heads")) or H) if model_type == "gla" else H
    short = lm.get("use_short_conv", short) is True
    conv = (num(lm.get("conv_size")) or 4) if short else 0
    return [H, round(hidden * ek) // H, round(hidden * ev) // H, kh, conv, rank]


# ---- loop_depth_unread ---------------------------------------------------------------------------------------------
# The (param_layers, eff_layers) the code runs, for the loop schemas _loop_depth misreads: Ouro (whole stack x
# total_ut_steps), CFRD (physical_cells cells over inference_recurrences, + a binding block), LFM3-RD (one motif per
# loop step: prelude + coda + motif_size x recurrent_loops, the remaining motifs run once when loops < motifs).
def detect_loop_depth_unread(cfg, model_type):
    lm = lm_of(cfg)
    n = num(lm.get("num_hidden_layers"))
    if model_type == "ouro":
        steps = num(lm.get("total_ut_steps")) or 4            # OuroModel: getattr(config, "total_ut_steps", 4)
        return [n, n * steps] if n else None
    if model_type == "cfrd":
        cells = num(lm.get("physical_cells")) or 2
        rec = num(lm.get("inference_recurrences")) or num(lm.get("recurrences")) or 6
        bind = 1 if lm.get("use_binding_block") is True else 0
        return [min(cells, rec) + bind, rec + bind]
    if model_type == "lfm3_rd":
        m, s, t = num(lm.get("num_motifs")), num(lm.get("motif_size")), num(lm.get("recurrent_loops"))
        if not (n and m and s and t):
            return None
        return [n, n + s * max(0, t - m)]
    return None


# ---- loop_boundary_norm --------------------------------------------------------------------------------------------
# A norm of the residual stream at every loop / stack-pass boundary:
#   final_norm_cfg  Nanbeige / qwen3loop: h := final_norm(h) after each loop unless skip_loop_final_norm (a config flag)
#   final_norm      Ouro: h := final_norm(h) after each of total_ut_steps loops (no switch)
#   stack_norm      HRM-Text: each stack pass ends in a weight-free RMSNorm
#   l2              Dory: h := L2(h) + e at every loop start (weight-free unit-L2 norm)
def detect_loop_boundary_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type in ("nanbeige", "qwen3loop"):
        if lm.get("skip_loop_final_norm") is True or lm.get("enable_double_loop_split") is True:
            return None                                       # norm once at the end / get_num_loops() == 1
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


# ---- loop_input_injection ------------------------------------------------------------------------------------------
# The loop input re-derived from the prelude output / embeddings every iteration:
#   add             Dory: h = L2(h) + L2(prelude_out) at each loop start
#   concat_adapter  Huginn / Raven: x = adapter(cat[x, e]) at each core-block step
#   feedback        LFM3-RD with recurrent_loops > num_motifs: h = prelude_out + tanh(g) (0.08 RMSNorm(.) + B A ...)
def detect_loop_input_injection(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "dory" and num(lm.get("n_recurrent_loops")) >= 1 and num(lm.get("n_recurrent_layers")):
        return "add"
    if model_type in ("huginn_raven", "huginn", "raven") and num(lm.get("mean_recurrence")) >= 1:
        return "concat_adapter"
    if model_type == "lfm3_rd" and num(lm.get("recurrent_loops")) > num(lm.get("num_motifs")) > 0:
        return "feedback"
    return None


# ---- loop_kv_share_gated_mix ---------------------------------------------------------------------------------------
# IQuest-Loop: loops >= 2 attend (1 - g) * local(window loop_window_size over the current loop's K/V) + g * global(loop
# 1's K/V of the same layer), g = sigmoid(q . w_h + b_h) per head. Returns the local window.
def detect_loop_kv_share_gated_mix(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "iquestloopcoder" and (num(lm.get("loop_num")) or 2) > 1:
        return num(lm.get("loop_window_size")) or 64
    return None


# ---- loop_split_depth_miscount -------------------------------------------------------------------------------------
# Nanbeige / qwen3loop LoopSplit (enable_double_loop_split): prefix (n - m) / 2 once, the middle m layers
# (n + m) // m times, suffix once; num_loops ignored. Returns [param_layers, eff_layers].
def detect_loop_split_depth_miscount(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in ("nanbeige", "qwen3loop") or lm.get("enable_double_loop_split") is not True:
        return None
    n = num(lm.get("num_hidden_layers"))
    m = num(lm.get("loop_middle_layers")) or n // 2
    if not n or not m or n % m:
        return None
    return [n, n + m * ((n + m) // m - 1)]


# ---- hrm_hierarchical_loop -----------------------------------------------------------------------------------------
# HRM-Text: two stacks (L, H) of P = num_layers_per_stack layers (default: the saved num_hidden_layers), run as
# H_cycles x (L_cycles L-passes on z_L + z_H, then one H-pass on z_H + z_L). Returns [param_layers, eff_layers].
def detect_hrm_hierarchical_loop(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "hrm_text":
        return None
    H = num(lm.get("H_cycles")) if "H_cycles" in lm else 2
    L = num(lm.get("L_cycles")) if "L_cycles" in lm else 3
    P = num(lm.get("num_layers_per_stack")) or num(lm.get("num_hidden_layers")) or 16
    return [2 * P, P * H * (L + 1)] if H and P else None


# ---- per_token_early_exit ------------------------------------------------------------------------------------------
# A data-dependent per-token exit depth at inference: Ouro's early_exit_threshold (first loop step whose cumulative exit
# pdf >= threshold) or early_exit_step; LFM3-RD's auto halting (reasoning_mode / reasoning_effort "auto").
def detect_per_token_early_exit(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "ouro" and (num(lm.get("total_ut_steps")) or 4) > 1:
        # (early_exit_step, checked first by the code, is a FIXED exit depth: no data-dependent exit)
        if lm.get("early_exit_step") is None and lm.get("early_exit_threshold") is not None:
            return "threshold"
        return None
    if model_type == "lfm3_rd" and "auto" in (str(lm.get("reasoning_mode")), str(lm.get("reasoning_effort"))):
        return "act"
    return None


# ---- lfm3rd_recurrent_controller -----------------------------------------------------------------------------------
# LFM3-RD: h = prelude_out + step_scales[T-1] * (h_core - prelude_out) after the recurrent core (every T), plus the
# controller's feedback injection / gated refinement when T > num_motifs. A bare feature.
def detect_lfm3rd_recurrent_controller(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "lfm3_rd" and num(lm.get("num_motifs")) and num(lm.get("recurrent_loops")):
        return True
    return None


# ==================== batch a12_extra_inputs ====================

# ---- qwen4_ple_reads_ngrams ----------------------------------------------------------------------------------------
# Qwen4-exp's "PLE" (Qwen4ExpTextPLELayer) is a hashed n-gram memory over the current and ngram_size - 1 previous
# tokens (XOR of token * seeded multiplier, mod per-head primes), gated per hyper-connection stream by a signed-sqrt
# sigmoid of <norm key, norm query>, plus a dilated depthwise short conv: an Engram, not Gemma's per-token PLE.
# Built only on the layers in ple_layer_ids (one-indexed); [] = no PLE layer at all.
QWEN4_TYPES = ("qwen4_exp", "qwen4_exp_text")


def detect_qwen4_ple_reads_ngrams(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in QWEN4_TYPES:
        return None
    ids = lm.get("ple_layer_ids")
    return "engram_qwen4" if isinstance(ids, list) and ids else None


# ---- ngram_memory_unread -------------------------------------------------------------------------------------------
# n-gram memories the _feats keys (engram_layer_ids / engram_vocab_size) miss, one FEATURES name per code family
# (different hashes / addressing / injection points: no family's memory plays another's).
def detect_ngram_memory_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "klm":                     # NgramMemory: exact bigram / trigram key banks, an AttnRes source
        return "engram_klm" if num(lm.get("n_bigrams")) + num(lm.get("n_trigrams")) > 0 else None
    if model_type == "crowd":                   # CrowdEngram: hash(prev * 1000003 + tok) % buckets, added to wte
        return "engram_crowd" if num(lm.get("engram_buckets")) > 0 else None
    if model_type == "allgdn":                  # Engram hooked before config.engram["layers"]
        ec = lm.get("engram")
        return "engram_allgdn" if isinstance(ec, dict) and ec.get("layers") else None
    if model_type == "needle":                  # Engram at config extras.engram.sites
        ex = lm.get("extras") if isinstance(lm.get("extras"), dict) else {}
        ec = ex.get("engram") if isinstance(ex.get("engram"), dict) else {}
        return "engram_needle" if ec.get("sites") else None
    return None


# ---- value_emb_keys_unread -----------------------------------------------------------------------------------------
# value embeddings / value residuals fixed in code (no config key _feats reads), by kind:
#   ve_shared   one shared token -> V table added on every other layer, gate 2 sigmoid(W x[:32]) (ARLoop, PreNormLoop)
#   ve_layer    per-layer token -> V tables on aux_layers, gate 3 sigmoid(W x[:aux_gate_dim]) (sn38-nanochrono)
#   v_residual  V mixed with the first attention layer's V by learned scalars (KLM value_mix)
VALUE_EMB_BY_CODE = {"arloop": "ve_shared", "arloop_baseline": "ve_shared", "klm": "v_residual"}


def detect_value_emb_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "sn38-nanochrono":
        aux = lm.get("aux_layers")              # None -> every other layer (configuration_nanochrono.py:46-47)
        return "ve_layer" if aux is None or (isinstance(aux, list) and aux) else None
    return VALUE_EMB_BY_CODE.get(model_type)


# ---- attn_res_keys_unread ------------------------------------------------------------------------------------------
# AttnRes depth-softmax residuals whose keys _feats does not read: [sublayers per source block, window in blocks
# (0 = every earlier block)]. GPT-S3: every sublayer output its own source; KLM: one source per layer (out + ffn) with
# the mixer output joining as the FFN's partial; ARLoop: blocks of attnres_block_size sublayers, the embedding plus
# the last sw_blocks blocks.
def detect_attn_res_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "gpts3_hybrid":
        return [1, 0]
    if model_type == "klm":
        return [2, 0]
    if model_type == "arloop":
        bs = num(lm.get("attnres_block_size"))
        return [bs, num(lm.get("sw_blocks"))] if bs > 0 else None
    return None


# ---- cfrd_film_norm_shift ------------------------------------------------------------------------------------------
# phase_rms_norm: RMSNorm(x) * (1 + 0.1 tanh(scale_d)) + 0.1 shift_d on both FoldedCell norms: a per-depth beta, i.e.
# a constant into every projection reading the norm (q / k / v, summary memory, gates, SwiGLU w1 / w3).
def detect_cfrd_film_norm_shift(cfg, model_type):
    return "bias_qkv+bias_out" if model_type == "cfrd" else None


# ---- cfrd_sigmoid_residual_gate ------------------------------------------------------------------------------------
# FoldedCell / BindingBlock: x + c sigmoid(w . u + b) f(u) on both sublayers (a data-dependent scalar residual gate).
def detect_cfrd_sigmoid_residual_gate(cfg, model_type):
    return "res_gate" if model_type == "cfrd" else None


# ---- bolmo_byte_patch_hierarchy ------------------------------------------------------------------------------------
# Bolmo: byte-level local encoder (mLSTM) + boundary predictor -> pooled patches -> global Olmo layers -> depool +
# local mLSTM decoder. Code-fixed for the type.
def detect_bolmo_byte_patch_hierarchy(cfg, model_type):
    return "byte_patch" if model_type == "bolmo" else None


# ---- hadamard_mlp_unlisted -----------------------------------------------------------------------------------------
# needle: every block's FFN is HadamardMLP (Kronecker-factored transforms at width next_pow2(d), fixed seeded
# permutations, a rank-8 softmax conditioner); no gate / up / down matrices.
def detect_hadamard_mlp_unlisted(cfg, model_type):
    return "hadamard" if model_type == "needle" else None


# ==================== batch bd ====================

_V4_COMPRESSED = ("compressed_sparse_attention", "heavily_compressed_attention")
_V4_TYPES = ("deepseek_v4", "memt_deepseek_v4")


def _v4_like(model_type):
    return model_type in _V4_TYPES or model_type.startswith("deepseek_v41") or model_type == "ernie5_5_moe"


# ---- b_kv_compress_host --------------------------------------------------------------------------------------------
# a host with compressed-KV layers (DeepSeek-V4 CSA / HCA pools, V4.1 ratio-1 / 2 compressors, ERNIE-5.5 HCA) whose
# per-layer compress ratio is config: ratio 0 builds no compressor (tf5 deepseek_v4 _COMPRESS_RATIO_TO_LAYER_TYPE
# 0 -> sliding_attention, compressor None; reference inference/model.py `if self.compress_ratio:`; ERNIE-5.5 ratio
# <= 0 = SWA / DSA-only). Returns the config key the host would rewrite.
def detect_b_kv_compress_host(cfg, model_type):
    lm = lm_of(cfg)
    if not _v4_like(model_type):
        return None
    cr = lm.get("compress_ratios")
    if isinstance(cr, list) and any(num(x) > 0 for x in cr):
        return "compress_ratios"
    if any(x in _V4_COMPRESSED for x in layer_types(lm)):
        return "layer_types"                       # tf5-format V4 config (MemT, Fx-Bio): the layer type list
    return None


# ---- b_hash_route_host ---------------------------------------------------------------------------------------------
# a host with hash-routed MoE layers (tid2eid[input_ids]) whose count is config: num_hash_layers (legacy key, folded
# into mlp_layer_types by tf5 DeepseekV4Config; default 3 when neither is given) or mlp_layer_types "hash_moe".
def detect_b_hash_route_host(cfg, model_type):
    lm = lm_of(cfg)
    mlt = lm.get("mlp_layer_types")
    if isinstance(mlt, list) and any(str(x) == "hash_moe" for x in mlt):
        return "mlp_layer_types"
    if num(lm.get("num_hash_layers")) > 0:
        return "num_hash_layers"
    if model_type in _V4_TYPES and lm.get("num_hash_layers") is None and mlt is None:
        return "num_hash_layers"                   # tf5 default_num_hash_layers = 3
    return None


# ---- b_act_clip_host -----------------------------------------------------------------------------------------------
# a clamped-SwiGLU host (gate <= L, |up| <= L, no swigluoai (up + 1)) whose limit L is read from the config by its
# code: L = +inf (tf5 clamp) or 0 / None (Step-3.5 / 3.7, V4.1 reference code) is no clamp. Families checked in code:
_ACT_CLIP_CFG_TYPES = {"deepseek_v4", "memt_deepseek_v4", "deepseek_v41", "deepseek_v41_text", "glm5_next",
                       "glm5_next_text", "hy_v4", "step3p5", "step3p7"}


def detect_b_act_clip_host(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in _ACT_CLIP_CFG_TYPES:
        return None                                # (exaone_moe: tf5 ExaoneMoe never reads swiglu_limits)
    for k in ("swiglu_limit", "swiglu_limits", "swiglu_limits_shared"):
        v = lm.get(k)
        if isinstance(v, list):
            if any((fnum(x, 0.0) or 0.0) > 0 for x in v):
                return k
        elif (fnum(v, 0.0) or 0.0) > 0:
            return k
    return None


# ---- b_clip_qkv_host -----------------------------------------------------------------------------------------------
# a host whose q / k / v are clamped to [-c, c] by a config value (tf5 OLMo / OLMoE `if clip_qkv is not None`, MPT /
# DBRX attn_config.clip_qkv): None switches the clamp off.
def detect_b_clip_qkv_host(cfg, model_type):
    lm = lm_of(cfg)
    if (fnum(lm.get("clip_qkv"), 0.0) or 0.0) > 0:
        return "clip_qkv"
    ac = lm.get("attn_config")
    if isinstance(ac, dict) and (fnum(ac.get("clip_qkv"), 0.0) or 0.0) > 0:
        return "attn_config.clip_qkv"
    return None


# ---- b_selfcond_norm_escape ----------------------------------------------------------------------------------------
# a self-conditioning host whose path ends in an always-on weight-free RMSNorm of the embeddings (DiffusionGemma
# post_norm, QwenDLM-ORL self_conditioning_post_norm) AND an untied lm_head: only then is the zeroed path's per-token
# rescale foldable into the input table (selfcond_norm_escape). Returns "untied"; a tied selfcond_norm host -> None.
def detect_b_selfcond_norm_escape(cfg, model_type):
    lm = lm_of(cfg)
    norm = model_type.startswith("diffusion_gemma") or lm.get("self_conditioning_post_norm") is True
    if not norm:
        return None
    tie = lm.get("tie_word_embeddings", (cfg or {}).get("tie_word_embeddings"))
    if tie is None:
        tie = True                                 # DiffusionGemma / SDAR config class default: tied
    return "untied" if tie is False else None


# ---- b_hash_route_target_learned_host ------------------------------------------------------------------------------
# a hash-routed TARGET: the gate score its hashed experts are weighted by. A learned-router host reproduces the hashed
# selection exactly (k-hot code channels, topk_code_select / code_margin); the weights are exact only for a
# renormalised softmax score (renorm_softmax_shift). Returns the score name ("softmax" = exact; V4's "sqrtsoftplus" =
# OPEN). The selection fact is the same as the host one.
def detect_b_hash_route_target_learned_host(cfg, model_type):
    if not detect_b_hash_route_host(cfg, model_type):
        return None
    lm = lm_of(cfg)
    sf = str(lm.get("scoring_func") or ("sqrtsoftplus" if model_type in _V4_TYPES else "softmax")).lower()
    return sf
