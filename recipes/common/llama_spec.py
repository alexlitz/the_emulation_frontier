"""The Llama-style decoder as a plain spec read off a Hub config.json (no transformers): the shape, per-layer
sliding windows, biases, qk-norm and the RoPE table, with HF transformers' own conventions (fp32 inv_freq,
`kv > q - W` windows, Qwen's max_window_layers). Every framework's llama.py runs this spec. Inlined into recipes.

Supported (anything else raises Unsupported with the reason):
  model_type  llama, mistral, ministral, qwen2, qwen3
  rope        default, linear, llama3, yarn; partial_rotary_factor
  FFN         dense SwiGLU (hidden_act silu); optional mlp_bias (llama)
"""
import math

import numpy as np

LLAMA_TYPES = ("llama", "mistral", "ministral", "qwen2", "qwen3")
ROPE_TYPES = ("default", "linear", "llama3", "yarn")


class Unsupported(Exception):
    pass


def _rope_params(cfg):
    rp = cfg.get("rope_parameters")
    rs = cfg.get("rope_scaling")
    if isinstance(rp, dict) and "rope_theta" in rp:          # transformers 5 layout
        if any(isinstance(v, dict) for v in rp.values()):
            raise Unsupported("per-layer-type rope_parameters")
        return float(rp["rope_theta"]), dict(rp)
    theta = float(cfg.get("rope_theta", 10000.0) or 10000.0)
    return theta, dict(rs or {})


def rope_table(cfg, hd):
    """(inv_freq fp32 (rot/2,), attention factor, rot_dim)."""
    theta, rs = _rope_params(cfg)
    typ = rs.get("rope_type", rs.get("type", "default")) or "default"
    if typ not in ROPE_TYPES:
        raise Unsupported("rope type %r" % typ)
    prf = float(cfg.get("partial_rotary_factor", rs.get("partial_rotary_factor", 1.0)) or 1.0)
    dim = int(hd * prf)
    f32 = np.float32
    ar = np.arange(0, dim, 2).astype(f32) / f32(dim)
    pos_freqs = (f32(theta) ** ar).astype(f32)
    inv = (f32(1.0) / pos_freqs).astype(f32)
    att = 1.0
    if typ == "linear":
        inv = (inv / f32(rs["factor"])).astype(f32)
    elif typ == "llama3":
        factor = rs["factor"]
        lo, hi = rs["low_freq_factor"], rs["high_freq_factor"]
        old = rs["original_max_position_embeddings"]
        lo_w, hi_w = old / lo, old / hi
        wl = (2 * math.pi / inv).astype(f32)
        il = np.where(wl > lo_w, inv / f32(factor), inv).astype(f32)
        sm = ((old / wl - lo) / (hi - lo)).astype(f32)
        smoothed = ((1 - sm) * il / f32(factor) + sm * il).astype(f32)
        med = ~(wl < hi_w) & ~(wl > lo_w)
        inv = np.where(med, smoothed, il).astype(f32)
    elif typ == "yarn":
        factor = rs["factor"]
        att = rs.get("attention_factor")
        ms, msa = rs.get("mscale"), rs.get("mscale_all_dim")
        omax = rs.get("original_max_position_embeddings") or cfg.get("max_position_embeddings")

        def gm(s, m=1):
            return 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0
        if att is None:
            att = float(gm(factor, ms) / gm(factor, msa)) if (ms and msa) else gm(factor)
        bf, bs = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1

        def corr(nr):
            return (dim * math.log(omax / (nr * 2 * math.pi))) / (2 * math.log(theta))
        lo, hi = corr(bf), corr(bs)
        if rs.get("truncate", True):
            lo, hi = math.floor(lo), math.ceil(hi)
        lo, hi = max(lo, 0), min(hi, dim - 1)
        if lo == hi:
            hi += 0.001
        ramp = np.clip((np.arange(dim // 2, dtype=f32) - f32(lo)) / f32(hi - lo), 0, 1).astype(f32)
        ext = (f32(1) - ramp).astype(f32)
        interp = (f32(1.0) / (f32(factor) * pos_freqs)).astype(f32)
        inv = (interp * (1 - ext) + inv * ext).astype(f32)
    return inv, float(att), dim


def llama_spec(cfg):
    """dict spec of a Llama-style config.json; Unsupported if it is not one this reference runs."""
    mt = cfg.get("model_type")
    if mt not in LLAMA_TYPES:
        raise Unsupported("model_type %r is not a Llama-style decoder" % mt)
    if cfg.get("quantization_config"):
        raise Unsupported("quantized checkpoint")
    for k in ("num_local_experts", "num_experts", "moe_intermediate_size"):
        if cfg.get(k):
            raise Unsupported("MoE (%s)" % k)
    act = cfg.get("hidden_act", "silu")
    if act not in ("silu", "swish"):
        raise Unsupported("hidden_act %r" % act)
    d, L, H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
    KV = cfg.get("num_key_value_heads") or H
    hd = cfg.get("head_dim") or d // H
    inv, att, rot = rope_table(cfg, hd)
    # windows (None = full attention), HF semantics: key s visible to query t iff t - W < s <= t
    win = [None] * L
    lt = cfg.get("layer_types")
    if mt in ("qwen2", "qwen3"):
        sw = cfg.get("sliding_window") if cfg.get("use_sliding_window", False) else None
        if lt is None:
            mwl = cfg.get("max_window_layers", 28)
            lt = ["sliding_attention" if sw is not None and i >= mwl else "full_attention" for i in range(L)]
        win = [sw if t == "sliding_attention" else None for t in lt]
    elif mt == "mistral":
        win = [cfg.get("sliding_window")] * L
    elif mt == "ministral" or lt is not None:
        sw = cfg.get("sliding_window")
        lt = lt or ["sliding_attention" if sw else "full_attention"] * L
        win = [sw if t == "sliding_attention" else None for t in lt]
    if mt == "qwen2":
        qkv_b, o_b = True, False
    elif mt in ("llama", "qwen3"):
        qkv_b = o_b = bool(cfg.get("attention_bias", False))
    else:
        qkv_b = o_b = False
    return {
        "model_type": mt, "vocab": cfg["vocab_size"], "d": d, "L": L, "H": H, "KV": KV, "hd": hd,
        "ff": cfg["intermediate_size"], "eps": float(cfg.get("rms_norm_eps", 1e-6)),
        "tie": bool(cfg.get("tie_word_embeddings", False)),      # the llama / mistral / qwen config default
        "inv_freq": inv, "rope_att": att, "rot": rot, "windows": win,
        "qkv_bias": qkv_b, "o_bias": o_b, "mlp_bias": bool(cfg.get("mlp_bias", False)) and mt == "llama",
        "qk_norm": mt == "qwen3", "scale": hd ** -0.5,
        "eos": cfg.get("eos_token_id"),
    }


def llama_weights(spec, sd):
    """The tensors of a Llama-style checkpoint under short per-layer names (numpy):
    {"embed", "norm", "head", "layers": [{"q","k","v","o","bq","bk","bv","bo","qn","kn","ln1","ln2","g","u","dn", ...}]}"""
    pre = "model."
    if pre + "embed_tokens.weight" not in sd:
        pre = ""
    W = {"embed": sd[pre + "embed_tokens.weight"], "norm": sd[pre + "norm.weight"]}
    W["head"] = W["embed"] if spec["tie"] else sd["lm_head.weight"]
    layers = []
    for i in range(spec["L"]):
        p = "%slayers.%d." % (pre, i)
        a, m = p + "self_attn.", p + "mlp."
        l = {"q": sd[a + "q_proj.weight"], "k": sd[a + "k_proj.weight"], "v": sd[a + "v_proj.weight"],
             "o": sd[a + "o_proj.weight"], "ln1": sd[p + "input_layernorm.weight"],
             "ln2": sd[p + "post_attention_layernorm.weight"],
             "g": sd[m + "gate_proj.weight"], "u": sd[m + "up_proj.weight"], "dn": sd[m + "down_proj.weight"]}
        for n, k in (("bq", "q_proj"), ("bk", "k_proj"), ("bv", "v_proj"), ("bo", "o_proj")):
            if a + k + ".bias" in sd:
                l[n] = sd[a + k + ".bias"]
        for n, k in (("bg", "gate_proj"), ("bu", "up_proj"), ("bd", "down_proj")):
            if m + k + ".bias" in sd:
                l[n] = sd[m + k + ".bias"]
        if spec["qk_norm"]:
            l["qn"], l["kn"] = sd[a + "q_norm.weight"], sd[a + "k_norm.weight"]
        layers.append(l)
    W["layers"] = layers
    return W
