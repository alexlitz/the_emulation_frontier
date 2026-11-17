"""Architecture metrics for every model in the zoo, computed from its config plus code facts: parameters (total,
active, embedding), KV cache per layer kind, recurrent state, FLOPs per token (prefill and decode), memory per pass
and memory traffic per decode step. The schema is in graph/arch_metrics_schema.md, and the trace checks against
the real transformers code are in graph/arch_trace.py.

The model is reduced to a PLAN: the embedding / head / final norm, and the decoder layers grouped by identical
shape. Each group has a count (unrolled and weight-distinct), a mixer (softmax attention: global / sliding / MLA /
K = V / KV-shared; or a recurrent family), an FFN (dense, GLU, MoE with shared experts) and its norm parameters.
Every metric is a sum over the groups, so the plan is the one place where the architecture is read:

  params      embedding + head + final norm + sum_g np_g (attn + ffn + rec + norm)
  active      params - sum_g np_g (routed experts not selected: (E - k) / E of the routed weights)
  matmul      2 x (linear weights applied per token, MoE active only) + 2 x the head
  attention   per query, 2 x heads x span x (d_qk + d_v), span = min(L, W)  (causal) or L (dense)
  KV          per token per layer: K and V heads (window layers cap at W - 1 tokens, the HF cache convention)

Run:  python -m graph.arch_metrics          # writes graph/data/arch_metrics.json
"""
from __future__ import annotations

import copy
import datetime
import json
import logging
import math
import os

from graph.model_graph import (BIAS_OUT_TYPES, BIAS_QKV_TYPES, FUSED_GLU_HALF, O_BIAS_FROM_ATTN_BIAS,
                               POST_NORM_TYPES, POST_ONLY_TYPES, SHARED_GATE_TYPES, _attn_counts, _cfg_value,
                               _dense_mlp_beside_moe, _has_shared_expert, _k_eq_v, _k_eq_v_all, _lm_config,
                               _moe_layer_mask, _num, _recurrent_family, _rec_dims, _tied, classify, dims_of,
                               MOE_SIG, VNORM_TYPES)

# round 3 (bias_out_by_code_unlisted, norm_fallback_rmsnorm_wrong): model types whose "bias_out" feature is the affine
# LayerNorm's beta (Nae1's nn.LayerNorm, nae1_model.py:203-204), not an o_proj / MLP bias
BIAS_OUT_NORM_BETA = {"nae1"}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "arch_metrics.json")
SCHEMA_VERSION = 1
CONTEXTS = {"1k": 1024, "4k": 4096, "8k": 8192, "32k": 32768, "128k": 131072}
DTYPE_BYTES = {"bf16": 2, "fp8": 1, "int4": 0.5}
BATCH_B = 16
CONVENTIONS = {
    "flops": "2 x multiply-accumulates of matrix products only (linear layers, QK^T, AV, SSM state contractions, "
             "convolutions); norms, activations, softmax, RoPE, routing top-k not counted "
             "(torch.utils.flop_counter.FlopCounterMode)",
    "attention_causal": "a query at position i attends min(i, W) keys; prefill amortised = sum_i min(i, W) / L",
    "attention_dense": "every query of an L-token prefill scores all L keys (HF eager / SDPA; what FlopCounterMode "
                       "measures); decode identical to causal",
    "decode_at_L": "the cache holds L - 1 tokens and the step produces token L; a window layer attends min(L, W) keys",
    "lm_head": "prefill counts the LM head at every position, decode once",
    "kv_window": "a sliding-window layer caches its W - 1 most recent tokens (transformers DynamicSlidingWindowLayer): "
                 "min(L, W - 1) tokens after L",
    "mla": "headline KV = the latent (kv_lora_rank + qk_rope_head_dim per layer, as vLLM / SGLang / FlashMLA cache it); "
           "kv_materialized = per-head K and V (transformers DynamicCache); FLOPs follow the transformers code "
           "(kv_b_proj expanded per token, attention on per-head K / V)",
    "k_eq_v": "a K = V layer (no v_proj) caches K and V as two tensors, as transformers (Gemma-4: V = v_norm(k_proj x), "
              "K = RoPE(k_norm(k_proj x)), both passed to cache.update), vLLM and llama.cpp allocate them; DeepSeek-V4's "
              "cache aliases V to K and holds one; kv_minimum counts one tensor for every K = V layer",
    "weights": "params x dtype bytes; quantisation scales and zero points ignored",
    "memory": "weights (bf16) + KV(L) (bf16) + recurrent state (bf16), batch 1 and batch B (weights once)",
    "bandwidth": "one decode step: active non-embedding weights + LM head + one embedding row, the KV read over the "
                 "attended span of every layer, the recurrent state read and written (bf16 throughout)",
}

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ config access

def _stock_config_class(mt):
    try:
        from transformers import CONFIG_MAPPING
    except Exception:          # pragma: no cover
        return None
    return CONFIG_MAPPING[mt] if mt and mt in CONFIG_MAPPING else None


def stock_arch(raw):
    """True when the config's architecture is the stock transformers code for its model_type: the type has a stock
    config class and architectures[0] (if any) is a stock class. A config whose model_type is stock but whose class
    is not (lmstudio MiniMax-M2 tagged 'minimax', LLaVA-OneVision's LlavaQwenForCausalLM) runs remote code."""
    mt = raw.get("model_type")
    if _stock_config_class(mt) is None:
        return False
    arch = (raw.get("architectures") or [None])[0]
    if not isinstance(arch, str):
        return True
    if mt == "mpt" and str((raw.get("attn_config") or {}).get("attn_type", "")).startswith("grouped"):
        return False                                       # MosaicML GQA: the stock MptAttention is MHA only
    for sub in ("text_config",):                           # a wrapper whose text model type is not stock
        st = raw.get(sub)
        if isinstance(st, dict) and st.get("model_type") and _stock_config_class(st["model_type"]) is None:
            return False
    name = _stock_class_names().get(arch.lower())          # JetMoEForCausalLM / PhiMoE / MPT: the stock port's
    if name is None:                                       # class differs only in case
        return False
    try:
        import transformers
        cls = getattr(transformers, name)
        cfg_cls = getattr(cls, "config_class", None)
        # the class must belong to this model type (BitnetForCausalLM tagged 'llama' is BitNet code, not Llama)
        return cfg_cls is None or cfg_cls is _stock_config_class(mt) or mt in str(getattr(cls, "__module__", ""))
    except Exception:          # pragma: no cover
        return False


_STOCK_NAMES = None


def _stock_class_names():
    global _STOCK_NAMES
    if _STOCK_NAMES is None:
        try:
            import transformers
            _STOCK_NAMES = {n.lower(): n for n in dir(transformers)}
        except Exception:      # pragma: no cover
            _STOCK_NAMES = {}
    return _STOCK_NAMES


def resolve(raw):
    """The LM config as the model runs it: (d, stock). For a model type with a stock transformers config class the
    raw dict goes through that class (stock code: its defaults and post-init, e.g. layer_types), and its text
    sub-config is taken; otherwise the raw LM sub-config (model_graph._lm_config). Remote code is never run."""
    mt = raw.get("model_type")
    cls = _stock_config_class(mt) if stock_arch(raw) else None
    if cls is not None:
        lg = logging.getLogger("transformers")
        lvl = lg.level
        lg.setLevel(logging.ERROR)
        try:
            hc = cls.from_dict(copy.deepcopy(raw))
            text = hc.get_text_config(decoder=True) if hasattr(hc, "get_text_config") else hc
            d = text.to_dict()
            d.setdefault("model_type", getattr(text, "model_type", mt))
            for k in ("tie_word_embeddings",):           # the top level decides the tie for wrappers
                if hasattr(hc, k) and text is not hc:
                    d["_outer_" + k] = getattr(hc, k)
            return d, True
        except Exception as e:                            # a config the stock class rejects: read it raw
            log.debug("stock config failed for %s: %s", mt, e)
        finally:
            lg.setLevel(lvl)
    return _lm_config(raw), False


# model types whose attention code sets head_dim = hidden_size // num_attention_heads whatever the config holds
HEAD_DIM_FROM_HIDDEN = frozenset({"stablelm", "phi", "gpt_neox", "persimmon", "gptj", "codegen", "bloom", "falcon",
                                  "gpt2", "gpt_bigcode", "opt", "gpt_neo", "mpt", "xglm", "ctrl", "biogpt",
                                  "openai-gpt", "starcoder2", "gpt_neox_japanese", "phimoe", "jamba",
                                  "granitemoehybrid"})


def _g(d, *keys, default=0):
    for k in keys:
        v = d.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return v
    return default


# ------------------------------------------------------------------------------------------------ the plan

def _norm_params(norm, width, bias=None):
    """Parameters of one norm over `width`: RMSNorm = weight; LayerNorm = weight + bias; non-affine = 0."""
    if norm == "none":
        return 0
    if bias is None:
        bias = norm == "LayerNorm"
    return width * (2 if bias else 1)


def _layer_kinds(d, raw, n):
    """Per-layer mixer kind list ('full' / 'sliding' / 'rec' / 'shared' / 'none'), length n, plus the window."""
    lt = d.get("layer_types")
    mt = str(d.get("model_type") or "").lower()
    W = _g(d, "sliding_window", "attention_chunk_size", "attention_window_size") or None
    if mt == "gpt_neo":
        W = _g(d, "window_size", default=256)
    kinds = None
    if isinstance(lt, list) and len(lt) == n and all(isinstance(x, str) for x in lt):
        kinds = []
        for x in lt:
            s = x.lower()
            if s in ("full_attention", "attention", "global", "full", "global_attention"):
                kinds.append("full")
            elif "sliding" in s or "chunk" in s or s == "local":
                kinds.append("sliding")
            elif s in ("linear_attention", "mamba", "conv", "recurrent", "ssm") or "linear" in s or "mamba" in s:
                kinds.append("rec")
            else:
                kinds.append("full")
    if kinds is None:
        _, _, base, _, kc, _ = _attn_counts(raw if raw.get("model_type") == mt else d, n)
        full = kc["FU"] + kc["FT"]
        loc = kc["LU"] + kc["LT"]
        kinds = ["full"] * full + ["sliding"] * loc + ["rec"] * kc["R"] + ["shared"] * kc["S"]
        kinds += ["full"] * max(0, n - len(kinds))
        kinds = kinds[:n]
    if "sliding" in kinds and not W:
        kinds = ["full" if k == "sliding" else k for k in kinds]
    if d.get("use_sliding_window") is False and mt in ("qwen2", "qwen2_moe", "qwen3", "qwen3_moe"):
        kinds = ["full" if k == "sliding" else k for k in kinds]
    return kinds, (int(W) if W and "sliding" in kinds else None)


def plan(raw):
    """The architecture PLAN of one raw config (see the module docstring)."""
    d, stock = resolve(raw)
    cl = classify(raw) or {}
    dims = dims_of(raw)
    mt = str(d.get("model_type") or raw.get("model_type") or "").lower()
    feats = set(filter(None, cl.get("feats", "").split("+")))
    notes = []
    hidden = int(_g(d, "hidden_size", "n_embd", "d_model", "dim", default=dims["hidden"]))
    vocab = int(_g(d, "vocab_size", "n_vocab", "padded_vocab_size", default=dims["vocab"]))
    nh = int(_g(d, "num_attention_heads", "n_head", "num_heads", default=dims["heads"]))
    hd = int(_g(d, "head_dim", "attention_head_dim") or (hidden // nh if nh else 0))
    if mt in HEAD_DIM_FROM_HIDDEN:
        hd = hidden // nh if nh else 0                      # head_dim = hidden // heads in code, whatever the config says
    if mt in ("zamba", "zamba2"):
        hd = int(_g(d, "attention_head_dim") or hd)
    nkv = int(_g(d, "num_key_value_heads", "num_kv_heads", "n_head_kv") or (1 if d.get("multi_query") else nh))
    n = int(_g(d, "num_hidden_layers", "n_layer", "num_layers", default=dims["layers"]))
    norm = cl.get("norm", "RMSNorm")
    if mt == "olmo":
        norm = "none"                                       # OlmoLayerNorm: no affine parameters
    if stock:      # transformers ties per the TEXT config (get_text_config(decoder=True).tie_word_embeddings)
        tied = bool(d.get("tie_word_embeddings", True))
    else:
        tied = _tied(d, raw)
    p = {"mt": mt, "arch": (raw.get("architectures") or [None])[0], "stock": stock, "hidden": hidden,
         "vocab": vocab, "tied": tied, "notes": notes, "max_context": dims["max_context"]}
    # embeddings and head
    emb = vocab * hidden
    pos_rows = 0
    if cl.get("pos") == "learned_abs":
        pos_rows = int(_g(d, "max_position_embeddings", "n_positions", "n_ctx"))
        if mt in ("opt", "biogpt"):
            pos_rows += 2                                   # OPT / BioGPT LearnedPositionalEmbedding: offset 2
    p["embedding"] = emb + pos_rows * hidden
    p["lm_head_W"] = vocab * hidden
    p["lm_head"] = 0 if tied else vocab * hidden
    final_norm = _norm_params(norm, hidden, bias=True if norm == "LayerNorm" else None)
    p["final_norm"] = final_norm
    p["extra"] = 0                                          # other params outside the layers (biases, norms)
    p["extra_W"] = 0                                        # linear weights outside the layers, applied per token
    # attention shape
    qkv_bias = "bias_qkv" in feats
    out_bias = "bias_out" in feats
    if mt in BIAS_OUT_NORM_BETA:
        out_bias = False                                    # (their bias_out is the LayerNorm beta, counted with the norm)
    o_bias = out_bias
    if mt in O_BIAS_FROM_ATTN_BIAS:
        o_bias = bool(d.get("attention_bias"))              # LlamaAttention: o_proj bias = attention_bias only
    if mt in ("qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl", "qwen2_vl_text", "qwen2_5_vl_text", "seed_oss",
              "glm", "glm4"):
        o_bias = bool(d.get("attention_out_bias", False)) if mt == "seed_oss" else False
    mlp_bias = bool(d.get("mlp_bias")) or (out_bias and mt not in O_BIAS_FROM_ATTN_BIAS)
    qkn = cl.get("qknorm", "no")
    qk_norm = 0
    if qkn in ("head", "head_post", "head_distinct"):
        qk_norm = 2 * hd
    elif qkn in ("proj",):
        qk_norm = nh * hd + nkv * hd
    elif qkn in ("head_ln",):
        qk_norm = 2 * (nh * hd + nkv * hd)                  # Cohere / StableLM: per-head LayerNorm weight (+bias)
    elif qkn == "head_free":
        qk_norm = 0
    # (round 3, model_graph._out_gate_key: the output gate is a family of keys out_gate[_head][_softplus|_silu][_bias] /
    # out_gate_post / out_gate_qlat -- every one is a gate projection; "_head" ones give one scalar per head)
    og = [f for f in feats if f.startswith("out_gate")]
    attn = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": qkv_bias, "o_bias": o_bias, "qk_norm": qk_norm,
            "out_gate": bool(og),
            "gate_headwise": bool(d.get("headwise_attn_output_gate") or d.get("use_head_wise_attn_gate")
                                  or any(f.startswith("out_gate_head") for f in og))}
    mla = None
    if d.get("kv_lora_rank"):
        rope = int(_g(d, "qk_rope_head_dim"))
        nope = int(_g(d, "qk_nope_head_dim"))
        vdim = int(_g(d, "v_head_dim") or nope)
        mla = {"q_lora": int(_g(d, "q_lora_rank") or 0), "kv_lora": int(d["kv_lora_rank"]), "rope": rope,
               "nope": nope, "v": vdim}
        attn.update(dqk=nope + rope, dv=vdim, nkv=nh)
    # FFN
    sig = dict(zip(MOE_SIG, dims["moe_sig"]))
    ffn_kind = cl.get("ffn", "glu")
    dense_inter = int(dims["inter"] or 0)
    if stock and isinstance(d.get("intermediate_size"), int) and mt not in ("mamba", "falcon_mamba", "moshi"):
        dense_inter = int(d["intermediate_size"])           # the stock config class's resolved width
    if mt == "lfm2" and d.get("block_auto_adjust_ff_dim"):
        dense_inter = int(2 * dense_inter / 3)              # Lfm2MLP: auto-adjusted SwiGLU width
        if d.get("block_ffn_dim_multiplier") is not None:
            dense_inter = int(d["block_ffn_dim_multiplier"] * dense_inter)
            m_ = int(d.get("block_multiple_of") or 256)
            dense_inter = m_ * ((dense_inter + m_ - 1) // m_)
    glu = ffn_kind in ("glu", "moe", "moe_shared")
    if mt in ("arcee", "nemotron", "apertus", "starcoder2", "phi", "gpt2", "gpt_bigcode", "gpt_neox", "gptj",
              "codegen", "bloom", "falcon", "opt", "gpt_neo", "mpt", "xglm", "ctrl", "biogpt", "openai-gpt",
              "persimmon", "gpt_neox_japanese", "stablelm") and ffn_kind == "vanilla":
        glu = False
    if ffn_kind == "vanilla":
        glu = False
    dense_ffn = {"glu": glu, "inter": dense_inter, "bias": mlp_bias}
    moe = None
    if sig["E"]:
        moe = {"E": sig["E"], "k": sig["k"], "ew": sig["ew"], "shared": sig["shared"],
               "shared_gate": "shared_gate" in feats or (mt in SHARED_GATE_TYPES and sig["shared"] > 0),
               "router_bias": mt in ("ernie4_5_moe", "gpt_oss"), "expert_bias": mt in ("gpt_oss",), "glu": True,
               "out_bias": mt == "jetmoe", "dense_compute": mt in DENSE_EXPERT_CODE and stock}
        if stock:
            # the stock expert width: moe_intermediate_size where the class defines it as an int (Qwen-MoE, DeepSeek,
            # GLM-4-MoE, dots, ERNIE), else intermediate_size (Mixtral, OLMoE, Granite-MoE, JetMoE, Jamba, Llama-4,
            # Hunyuan-MoE -- whose moe_intermediate_size list the code does not read)
            ew = d.get("moe_intermediate_size")
            if not isinstance(ew, int) or isinstance(ew, bool):
                ew = d.get("intermediate_size")
            if isinstance(ew, int) and mt not in ("doge",):
                moe["ew"] = ew
    if moe is None and stock and mt in ("mixtral", "phimoe", "minimax") and d.get("num_local_experts") == 1:
        # one expert: dense for the zoo's relation (dims_of), but the stock code still builds the router (h x 1)
        moe = {"E": 1, "k": 1, "ew": int(d["intermediate_size"]), "shared": 0, "shared_gate": False,
               "router_bias": False, "expert_bias": False, "glu": True, "out_bias": False, "dense_compute": False}
    # layer kinds and counts
    kinds, W = _layer_kinds(d, raw, n)
    lt_ = d.get("layer_types")
    # the groups' `at` is the real layer order when the kinds come from a per-layer list (else from counts: kind order)
    p["order_exact"] = len(set(kinds)) <= 1 or (isinstance(lt_, list) and len(lt_) == n and all(
        isinstance(x, str) for x in lt_))
    moe_mask = _moe_layer_mask(d, n) if moe else None
    if moe and moe_mask is None:
        moe_mask = [True] * n
    keqv = _k_eq_v(d) or _k_eq_v_all(d)
    groups, at = {}, {}
    for i, k in enumerate(kinds):
        is_moe = bool(moe and moe_mask and moe_mask[i])
        key = (k, is_moe)
        groups.setdefault(key, 0)
        groups[key] += 1
        at.setdefault(key, []).append(i)
    G = []
    for (k, is_moe), cnt in groups.items():
        g = {"n": cnt, "np": cnt, "at": at[(k, is_moe)], "kind": k, "window": W if k == "sliding" else None, "attn": None, "mla": None,
             "rec": None, "ffn": None, "moe": None, "norm": 0, "k_eq_v": bool(keqv and (k == "full" or _k_eq_v_all(d)))}
        if k in ("full", "sliding", "shared"):
            g["attn"] = dict(attn)
            g["mla"] = mla
        if k == "rec":
            g["rec"] = {"family": _recurrent_family(d, cl.get("mixer", "hybrid"))}
        if is_moe:
            g["moe"] = moe
        else:
            g["ffn"] = dense_ffn
        nn_ = 2
        if "post_norm" in feats:
            nn_ = 4
        if "post_only" in feats:
            nn_ = 2
        if cl.get("block") == "parallel":
            nn_ = 1
            if mt == "falcon" and d.get("new_decoder_architecture") and _g(d, "num_ln_in_parallel_attn", default=2) == 2:
                nn_ = 2
            if mt in ("gpt_neox", "bitsy_parallel"):         # (Bitsy: attn_norm, mlp_norm, modeling_bitsy.py:81-95)
                nn_ = 2
        g["norm"] = nn_ * _norm_params(norm, hidden)
        G.append(g)
    p["groups"] = G
    p["layers"] = n
    if "dsa" in feats and _g(d, "index_head_dim") and _g(d, "index_n_heads"):
        # the DSA lightning indexer of every attention layer (modeling_glm_moe_dsa.py GlmMoeDsaIndexer, read as text):
        # wq_b (q_lora_rank -> heads x dim), wk (hidden -> dim), k_norm (LayerNorm), weights_proj (hidden -> heads)
        ih, idim = int(_g(d, "index_n_heads")), int(_g(d, "index_head_dim"))
        ql = int(_g(d, "q_lora_rank") or hidden)
        p["dsa"] = {"heads": ih, "dim": idim, "topk": int(_g(d, "index_topk", default=2048))}
        p["dsa_dim"] = idim
        add = lambda a: dict(a, extra_W=a.get("extra_W", 0) + ql * ih * idim + hidden * idim + hidden * ih,
                             extra_b=a.get("extra_b", 0) + 2 * idim)
        it = d.get("indexer_types")
        if isinstance(it, list) and len(it) == n and mt == "glm_moe_dsa":
            # GLM-5.x indexer_types (modeling_glm_moe_dsa.py, transformers main, read as text: GlmMoeDsaAttention
            # builds an indexer only when indexer_types[i] == "full"; a "shared" layer reuses the previous full
            # layer's top-k selection): only those layers hold the indexer weights and cache its keys
            layers = []
            for i, k in enumerate(kinds):
                is_moe = bool(moe and moe_mask and moe_mask[i])
                t = next(g for g in G if g["kind"] == k and (g["moe"] is not None) == is_moe)
                g = {kk: v for kk, v in t.items() if kk not in ("n", "np")}
                if g["attn"] is not None:
                    g["dsa_index"] = it[i] == "full"
                    if g["dsa_index"]:
                        g["attn"] = add(g["attn"])
                layers.append(g)
            G = _per_layer_groups(layers)
            p["groups"] = G
        else:
            for g in G:
                if g["attn"] is not None:
                    g["attn"] = add(g["attn"])
    _type_fixups(p, d, raw, cl, dims)
    if mt.startswith("gemma4"):
        _gemma4(p, d)
    if p["arch"] in LONGCAT_ARCHS:
        _longcat(p, d)
    if mt == "nemotron-nas" and isinstance(d.get("block_configs"), list):
        _nemotron_nas(p, d)
    if mt == "openelm" and isinstance(d.get("num_query_heads"), list):
        _openelm(p, d)
    kvl = d.get("num_key_value_heads_per_layer")
    if isinstance(kvl, list) and len(kvl) == p["layers"] and len(p["groups"]) == 1 and p["groups"][0]["attn"]:
        # DeciLM-7B: a KV-head count per layer
        tmpl = p["groups"][0]
        layers = []
        for x in kvl:
            g = {k: v for k, v in tmpl.items() if k not in ("n", "np")}
            g["attn"] = dict(tmpl["attn"], nkv=int(x))
            layers.append(g)
        p["groups"] = _per_layer_groups(layers)
    if dims["param_layers"] and dims["eff_layers"] and dims["param_layers"] < dims["eff_layers"] and not stock:
        # a looped / recurrent-depth model: the unrolled depth runs param_layers distinct weight sets. The plan's
        # groups hold the config's layer count, which is either the unrolled depth (ARLoop: 72 over 24) or already
        # the distinct count (Nanbeige-4.2 num_loops, IQuest-Loop loop_num, qwen3loop: every layer stored); scale
        # only by distinct / planned, and only when the plan holds more layers than there are weight sets
        L = sum(g["n"] for g in p["groups"]) or p["layers"]
        if L > dims["param_layers"]:
            f = dims["param_layers"] / L
            for g in p["groups"]:
                g["np"] = max(1, round(g["n"] * f)) if g["n"] else 0
        p["notes"].append(f"looped: {dims['eff_layers']} unrolled layers over {dims['param_layers']} distinct ones")
    _recurrent(p, d, raw, cl)
    _bespoke(p, d, raw, cl)
    p["max_context"], p["max_context_source"], p["max_context_unbounded"] = _max_context(p, d, raw, cl)
    return p


# ------------------------------------------------------------------------------------------------ max context

CTX_KEYS = ("max_position_embeddings", "max_seq_len", "n_positions", "seq_length")       # dims_of's keys, in order
# rope types whose original x factor is the extended window (llama3's factor shapes the frequencies: Llama-3.2's
# 8,192 x 32 is not its 131,072 limit)
ROPE_EXTENDING = ("yarn", "deepseek_yarn", "dynamic", "linear", "longrope")
CTX_KEYS_ALT = ("model_max_length", "max_sequence_length", "sequence_len", "context_length", "max_seq_length",
                "block_size", "n_ctx")


def _ctx_num(v):
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 0


def _max_context(p, d, raw, cl):
    """(tokens, source, unbounded): the model's context limit read by arch_metrics itself, not the classifier's
    4096 default (graph/model_graph.py dims_of), with its provenance:
      config      the positions key dims_of reads (max_position_embeddings, max_seq_len, n_positions, seq_length) in
                  the LM config, clamped to 2^20 as dims_of does
      rope        no positions key: rope_scaling (or rope_parameters) original_max_position_embeddings x factor, for
                  the window-extending types (ROPE_EXTENDING); with a positions key, a larger extended window is
                  only named in the source (Qwen3 YaRN: 32,768 x 4 = 131,072 declared beside its 40,960)
      other key   no positions key: model_max_length (Kimi-Linear: 1,048,576), max_sequence_length (LLaMA-1),
                  sequence_len, context_length, block_size, n_ctx -- the LM config, then the top level
      class       the stock config class's default (FalconConfig, LlamaConfig: max_position_embeddings 2,048)
      unbounded   no limit anywhere and no full-attention layer: a recurrent or window-only stack (Mamba, RWKV,
                  RecurrentGemma) has no positional limit; sized at the largest standard context (128k)
      default     none of these: 4,096, marked as a default"""
    lm = _lm_config(raw)
    lm = lm if isinstance(lm, dict) else {}
    for k in CTX_KEYS:
        v = _ctx_num(lm.get(k))
        if v:
            val, src = min(v, 1 << 20), f"config: {k}"
            break
    else:
        val = src = None
    rs = None
    for c in (lm, d, raw):
        for k in ("rope_scaling", "rope_parameters"):
            if isinstance(c.get(k), dict) and c[k].get("factor") is not None:
                rs = c[k]
                break
        if rs:
            break
    ext = None
    if rs and str(rs.get("rope_type") or rs.get("type") or "") in ROPE_EXTENDING:
        orig = _ctx_num(rs.get("original_max_position_embeddings")) or _ctx_num(
            lm.get("original_max_position_embeddings"))
        fac = rs.get("factor")
        if orig and isinstance(fac, (int, float)) and not isinstance(fac, bool) and fac > 1:
            ext = (min(int(orig * fac), 1 << 20),
                   f"rope_scaling {rs.get('rope_type') or rs.get('type')}: original_max_position_embeddings "
                   f"{orig:,d} x factor {fac:g}")
    if val is not None:
        if ext and ext[0] > val:                       # stated, not applied: the positions key is the vendor's limit
            src += f" ({ext[1]} = {ext[0]:,d} declared)"
        return val, src, False
    if ext:
        return ext[0], ext[1], False
    alt = next(((min(_ctx_num(c.get(k)), 1 << 20), f"{where}: {k}") for c, where in
                ((lm, "config"), (raw, "config (top level)")) for k in CTX_KEYS_ALT if _ctx_num(c.get(k))), None)
    full = any(g["attn"] is not None and g["window"] is None and g["kind"] != "none" for g in p["groups"])
    if not full and cl.get("pos") != "learned_abs":
        return CONTEXTS["128k"], ("unbounded: a recurrent / window-only stack has no positional limit (sized at 128k)"
                                  + (f"; trained context {alt[0]:,d} ({alt[1]})" if alt else "")), True
    if alt:
        return alt[0], alt[1], False
    if p["stock"]:
        v = _ctx_num(d.get("max_position_embeddings"))
        if v:
            return min(v, 1 << 20), f"class default: max_position_embeddings of the stock {p['mt']} config", False
    return 4096, "default: no context limit in the config or the code's defaults (4,096 assumed)", False


# ------------------------------------------------------------------------------------------------ recurrent mixers
# Each builder returns the mixer's linear weights W (applied per token), other params b, the state it carries per
# sequence (conv_state, ssm_state elements, as the transformers cache holds them) and core_flops per token (the
# state contractions and the causal conv, by the FLOPs convention). Code-read from transformers 4.57.

def _mamba1(d, h, mt):
    if mt in ("mamba", "falcon_mamba"):
        di, N, K = int(d["intermediate_size"]), int(d["state_size"]), int(d["conv_kernel"])
        R = d.get("time_step_rank")
        R = math.ceil(h / 16) if R in (None, "auto") else int(R)
        bias, cbias = bool(d.get("use_bias")), bool(d.get("use_conv_bias", True))
        extra = 0
    else:                                                   # jamba / zamba (mamba_* keys)
        di = int(d.get("mamba_expand", 2) * h)
        N, K = int(d.get("mamba_d_state", 16)), int(d.get("mamba_d_conv", 4))
        R = d.get("mamba_dt_rank")
        R = math.ceil(h / 16) if R in (None, "auto") else int(R)
        bias, cbias = bool(d.get("mamba_proj_bias")), bool(d.get("mamba_conv_bias", True))
        extra = (R + 2 * N) if mt == "jamba" else 0         # Jamba dt / b / c RMSNorms
    W = h * 2 * di + di * (R + 2 * N) + R * di + di * h
    b = (2 * di + h if bias else 0) + di * K + (di if cbias else 0) + di + di * N + di + extra
    # Zamba's multi-head mixer applies x_proj / dt_proj as raw parameters (x_proj_weight, dt_proj_weight)
    W_raw = di * (R + 2 * N) + R * di if mt == "zamba" else 0
    return {"family": "mamba1", "W": W, "b": b, "conv_state": di * K, "ssm_state": di * N,
            "core_flops": 2 * di * K + 2 * di * N, "conv_ch": di, "conv_k": K, "W_raw": W_raw,
            "readout": 2 * di * N}


def _mamba2(d, h, mt):
    if mt == "zamba2":
        H, P, G = int(d["n_mamba_heads"]), int(d["mamba_headdim"]), int(d.get("mamba_ngroups", 1))
        I = int(d.get("mamba_expand", 2) * h)
        cbias, pbias = bool(d.get("use_conv_bias", True)), bool(d.get("add_bias_linear"))
        obias, norm = pbias, True
    elif mt == "mamba2":
        # stock Mamba2Mixer (transformers models/mamba2/modeling_mamba2.py lines 234-284, read as text): num_heads,
        # head_dim, n_groups, expand, use_bias / use_conv_bias; the gated RMSNorm (line 281) always
        H, P, G = int(d["num_heads"]), int(d["head_dim"]), int(d.get("n_groups", 1))
        I = int(d.get("expand", 2) * h)
        cbias, pbias = bool(d.get("use_conv_bias", True)), bool(d.get("use_bias"))
        obias, norm = pbias, True
    else:
        H, P, G = int(d["mamba_n_heads"]), int(d["mamba_d_head"]), int(d.get("mamba_n_groups", 1))
        I = int(d.get("mamba_d_ssm") or d.get("mamba_expand", 2) * h)
        cbias, pbias = bool(d.get("mamba_conv_bias", True)), bool(d.get("mamba_proj_bias"))
        obias = bool(d.get("projectors_bias")) if mt == "falcon_h1" else pbias
        norm = bool(d.get("mamba_rms_norm", True)) if mt == "falcon_h1" else True
    N = int(d.get("state_size", 128) if mt == "mamba2" else d.get("mamba_d_state", 128))
    K = int(d.get("conv_kernel", 4) if mt == "mamba2" else d.get("mamba_d_conv", 4))
    conv = I + 2 * G * N
    proj = 2 * I + 2 * G * N + H
    W = h * proj + I * h
    b = (proj if pbias else 0) + conv * K + (conv if cbias else 0) + 3 * H + (I if norm else 0) + (h if obias else 0)
    return {"family": "mamba2", "W": W, "b": b, "conv_state": conv * K, "ssm_state": H * P * N,
            "core_flops": 2 * conv * K + 2 * H * P * N, "conv_ch": conv, "conv_k": K}


def _gdn(d, h, mt):
    Hk, Hv = int(d["linear_num_key_heads"]), int(d["linear_num_value_heads"])
    dk, dv, K = int(d["linear_key_head_dim"]), int(d["linear_value_head_dim"]), int(d.get("linear_conv_kernel_dim", 4))
    kd, vd = Hk * dk, Hv * dv
    conv = 2 * kd + vd
    W = h * (2 * kd + 2 * vd) + h * 2 * Hv + vd * h
    b = conv * K + 2 * Hv + dv                               # conv (no bias), dt_bias, A_log, gated RMSNorm(dv)
    return {"family": "gdn", "W": W, "b": b, "conv_state": conv * K, "ssm_state": Hv * dk * dv,
            "core_flops": 2 * conv * K + 6 * Hv * dk * dv, "conv_ch": conv, "conv_k": K}


def _kda(d, h, mt):
    """Kimi Delta Attention (Kimi-Linear), transcribed from the code read as text, not run:
    moonshotai/Kimi-Linear-48B-A3B-Instruct modeling_kimi.py (Hub snapshot e1df551) KimiDeltaAttention.__init__,
    lines 444-503, and transformers 5.18 models/kimi_linear/modeling_kimi_linear.py KimiLinearDeltaAttention,
    lines 569-604, with KimiLinearForgetGate, lines 316-328 (the same tensors, the three convs fused into one):
      q_proj, k_proj, v_proj   h -> H x dk (lines 465-469 / 585-587)
      q / k / v ShortConvolution  depthwise, kernel K, no bias (lines 471-485 / 590-597): 3 x H x dk x K
      A_log (1, 1, H, 1), dt_bias (H x dk)   (lines 487-494 / 327-328)
      f_a_proj h -> dk, f_b_proj dk -> H x dk  (lines 490-491 / 325-326)
      b_proj h -> H            (line 496 / 599)
      g_a_proj h -> dk, g_b_proj dk -> H x dk  (lines 498-499 / 601-602)
      o_norm FusedRMSNormGated(dk): dk       (line 501 / 603)
      o_proj H x dk -> h       (line 503 / 604)
    The cache holds the three conv states (H x dk x K each, modeling_kimi.py lines 593-594) and the recurrent
    state H x dk x dv (fused_recurrent_kda / chunk_kda initial_state, lines 568-592). The recurrence is gated
    DeltaNet's with a per-channel decay, so its state contraction is the same 6 x H x dk x dv per token."""
    lac = d.get("linear_attn_config") if isinstance(d.get("linear_attn_config"), dict) else {}
    dk = int(lac.get("head_dim") or d.get("linear_head_dim") or 128)
    H = int(lac.get("num_heads") or d.get("linear_num_heads") or 32)
    K = int(lac.get("short_conv_kernel_size") or d.get("linear_conv_kernel_dim") or 4)
    q = H * dk
    conv = 3 * q
    W = 3 * h * q + (h * dk + dk * q) + h * H + (h * dk + dk * q) + q * h
    b = conv * K + H + q + dk
    return {"family": "kda", "W": W, "b": b, "conv_state": conv * K, "ssm_state": H * dk * dk,
            "core_flops": 2 * conv * K + 6 * H * dk * dk, "conv_ch": conv, "conv_k": K}


def _shortconv(d, h, mt):
    K = int(d.get("conv_L_cache", 3))
    bias = bool(d.get("conv_bias"))
    W = h * 3 * h + h * h
    b = h * K + ((h + 3 * h + h) if bias else 0)
    return {"family": "shortconv", "W": W, "b": b, "conv_state": h * K, "ssm_state": 0, "core_flops": 2 * h * K,
            "conv_ch": h, "conv_k": K, "W_raw": 0, "readout": 0}


def _rglru(d, h, mt):
    lru, nh, K = int(d["lru_width"]), int(d["num_attention_heads"]), int(d.get("conv1d_width", 4))
    bw = lru // nh
    W = 2 * h * lru + lru * h + 2 * nh * bw * bw            # linear_y, linear_x, linear_out, the two gate matmuls
    b = 2 * lru + h + lru * K + lru + lru + 2 * lru         # biases, conv (+ bias), recurrent_param, gate biases
    return {"family": "rglru", "W": W, "b": b, "conv_state": lru * (K - 1), "ssm_state": lru,
            "core_flops": 2 * lru * K, "conv_ch": lru, "conv_k": K, "W_raw": 2 * nh * bw * bw, "readout": 0}


def _rwkv4(d, h, mt):
    ah = int(d.get("attention_hidden_size") or h)
    W = 3 * h * ah + ah * h
    b = 3 * h + 2 * ah
    return {"family": "rwkv4", "W": W, "b": b, "conv_state": 0, "ssm_state": 2 * h + 3 * ah, "core_flops": 0}


def _lightning(d, h, mt):
    nh = int(d["num_attention_heads"])
    hd = int(d.get("head_dim") or h // nh)
    W = h * 3 * nh * hd + nh * hd * h + h * nh * hd
    b = nh * hd
    return {"family": "lightning", "W": W, "b": b, "conv_state": 0, "ssm_state": nh * hd * hd,
            "core_flops": 6 * nh * hd * hd}


def _bailing_lightning(d, h, mt):
    """Ling / Ring 2.5 - 2.6 (bailing_hybrid, BailingMoeV2_5) linear-attention layers: the same fused
    query_key_value (h -> 3 x heads x hd), g_proj (h -> heads x hd), dense (heads x hd -> h) and g_norm (heads x hd)
    as the lightning builder, plus query_layernorm / key_layernorm (hd each) under use_qk_norm (Ling-2.6-flash
    tensors model.layers.N.attention.{query_key_value, g_proj, dense, g_norm, query_layernorm, key_layernorm}; the
    head count is num_kv_heads_for_linear_attn when set)."""
    dd = dict(d)
    if d.get("num_kv_heads_for_linear_attn"):
        dd["num_attention_heads"] = d["num_kv_heads_for_linear_attn"]
    rec = _lightning(dd, h, mt)
    nh = int(dd["num_attention_heads"])
    hd = int(d.get("head_dim") or h // nh)
    if d.get("use_qk_norm"):
        rec["b"] += 2 * hd
    rec["family"] = "bailing_lightning"
    return rec


def _gla(d, h, mt):
    """Gated Linear Attention, fla GatedLinearAttention (fla-org flash-linear-attention main, fla/layers/gla.py
    lines 100-180, read as text): key_dim = h x expand_k, value_dim = h x expand_v, kv groups = num_heads /
    num_kv_heads; q_proj h -> key_dim, k_proj h -> key_dim / groups, v_proj h -> value_dim / groups, g_proj
    h -> value_dim (use_output_gate), gk_proj h -> gate_low_rank_dim -> key_dim / groups (+ bias), o_proj
    value_dim -> h, optional q / k / v ShortConvolution (kernel conv_size), g_norm over head_v_dim. The state is
    heads x head_k x head_v."""
    nh = int(d.get("num_heads") or d.get("num_attention_heads"))
    nkv = int(d.get("num_kv_heads") or nh)
    kd, vd = round(h * float(d.get("expand_k", 0.5))), round(h * float(d.get("expand_v", 1)))
    grp = nh // nkv
    kg, vg = kd // grp, vd // grp
    r = int(d.get("gate_low_rank_dim") or 16)
    K = int(d.get("conv_size") or 4)
    W = h * kd + h * kg + h * vg + (h * vd if d.get("use_output_gate", True) else 0) + vd * h + h * r + r * kg
    conv = (kd + kg + vg) if d.get("use_short_conv") else 0
    b = kg + (vd // nh if d.get("elementwise_affine", True) else 0) + conv * K + (conv if d.get("conv_bias") else 0)
    hk, hv = kd // nh, vd // nh
    return {"family": "gla", "W": W, "b": b, "conv_state": conv * K, "ssm_state": nh * hk * hv,
            "core_flops": 2 * conv * K + 4 * nh * hk * hv, "conv_ch": conv, "conv_k": K}


def _deltanet(d, h, mt):
    """fla DeltaNet (fla-org flash-linear-attention 0.4.0, fla/layers/delta_net.py lines 114-164, read as text):
    key_dim = h x expand_k, value_dim = h x expand_v; q_proj / k_proj h -> key_dim, v_proj h -> value_dim, b_proj
    h -> heads (use_beta), q / k / v ShortConvolution (kernel conv_size, use_short_conv), g_proj h -> value_dim
    (use_gate), o_norm over head_v_dim (built in both branches), o_proj value_dim -> h. The state is heads x head_k x
    head_v, the delta-rule update 6 x heads x dk x dv per token like gated DeltaNet's."""
    nh = int(d.get("num_heads") or d.get("num_attention_heads"))
    kd, vd = int(h * float(d.get("expand_k", 1))), int(h * float(d.get("expand_v", 1)))
    K = int(d.get("conv_size") or 4)
    W = 2 * h * kd + h * vd + (h * nh if d.get("use_beta", True) else 0) + (h * vd if d.get("use_gate") else 0) + vd * h
    conv = (2 * kd + vd) if d.get("use_short_conv", True) else 0
    b = conv * K + (conv if d.get("conv_bias") else 0) + vd // nh
    hk, hv = kd // nh, vd // nh
    return {"family": "deltanet", "W": W, "b": b, "conv_state": conv * K, "ssm_state": nh * hk * hv,
            "core_flops": 2 * conv * K + 6 * nh * hk * hv, "conv_ch": conv, "conv_k": K}


def _rwkv7(d, h, mt):
    """RWKV-7 time mix, fla RWKV7Attention (fla-org flash-linear-attention main, fla/layers/rwkv7.py lines 52-145,
    read as text; LoRA fla/layers/rwkv6.py lines 222-246: two Linears, the second with bias unless bias=False):
    key_dim = h, value_dim (default h), heads = h / head_dim; x_r / x_w / x_k / x_v / x_a / x_g (h each), k_k, k_a
    (h), r_k (heads x head_dim); r / k / v / o projections; w_lora (decay_low_rank_dim, bias), a_lora
    (a_low_rank_dim, bias), g_lora (gate_low_rank_dim, no bias); g_norm GroupNorm (weight + bias). v_lora
    (v_low_rank_dim, bias) exists on layers 1.. only (added by _rwkv7_layers). The state is heads x hd x hd plus
    the token-shift row."""
    hd = int(d.get("head_dim") or 64)
    nh = h // hd
    v = int(d.get("value_dim") or h) if not isinstance(d.get("value_dim"), list) else int(d["value_dim"][0])
    Dw, Da, Dg = int(d.get("decay_low_rank_dim") or 64), int(d.get("a_low_rank_dim") or 64), \
        int(d.get("gate_low_rank_dim") or 128)
    W = 2 * h * h + h * v + v * h \
        + (h * Dw + Dw * h) + (h * Da + Da * h) + (h * Dg + Dg * v)
    b = 6 * h + 2 * h + nh * hd + h + h + 2 * v
    return {"family": "rwkv7", "W": W, "b": b, "conv_state": 0, "ssm_state": nh * hd * (v // nh) + h,
            "core_flops": 6 * nh * hd * (v // nh)}


def _rwkv7_layers(p, d, g):
    """RWKV-7 blocks (fla fla/models/rwkv7/modeling_rwkv7.py lines 37-64 RWKV7FeedForward: x_k (h), key h -> I,
    value I -> h; lines 113-140 RWKV7Block: attn_norm, ffn_norm LayerNorms (bias = norm_bias), pre_norm on layer 0
    under norm_first); layer 0 has no v_lora, so it is its own group."""
    h = p["hidden"]
    I = int(d.get("intermediate_size") or int(h * float(d.get("hidden_ratio") or 4)))
    unit = h * (2 if d.get("norm_bias", True) else 1)
    v = int(d.get("value_dim") or h) if not isinstance(d.get("value_dim"), list) else int(d["value_dim"][0])
    Dv = int(d.get("v_low_rank_dim") or 16)
    n = g["n"]
    g.update(n=1, np=1, at=[0], ffn={"glu": False, "inter": I, "bias": False}, norm=2 * unit + h)
    rest = dict(g, n=n - 1, np=n - 1, at=list(range(1, n)), rec=dict(g["rec"], W=g["rec"]["W"] + h * Dv + Dv * v,
                                                  b=g["rec"]["b"] + v))
    p["groups"] = [g] + ([rest] if n > 1 else [])
    p["order_exact"] = True
    if d.get("norm_first", True):
        p["extra"] += unit                                  # layer 0 pre_norm
    p["final_norm"] = unit


def _zamba(p, d, rec):
    """Zamba / Zamba2: a Mamba block in every layer; the 'hybrid' layers also run a SHARED transformer block (one, or
    num_mem_blocks of them cycled) on concat(hidden, input embeddings) -- attention from 2h, o_proj to h, a GLU MLP --
    plus a per-layer linear (h -> h) and, in Zamba2, per-invocation LoRA adapters on q / k / v and the MLP gate_up."""
    mt, h = p["mt"], p["hidden"]
    n = p["layers"]
    lbt = d.get("layers_block_type") or []
    n_h = sum(1 for x in lbt if x == "hybrid")
    M = int(d.get("num_mem_blocks") or 1) if mt == "zamba2" else 1
    nh, nkv = int(d["num_attention_heads"]), int(d.get("num_key_value_heads") or d["num_attention_heads"])
    hd = int(d.get("attention_head_dim") or (2 * h) // nh)
    inter = int(d["intermediate_size"])
    a = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": False, "o_bias": False, "qk_norm": 0,
         "out_gate": False, "in_dim": int(d.get("attention_hidden_size") or 2 * h)}
    hyb = [i for i, x in enumerate(lbt) if x == "hybrid"]
    rec_g = {"n": n, "np": n, "at": list(range(n)), "kind": "rec", "window": None, "attn": None, "mla": None, "rec": dict(rec),
             "ffn": None, "moe": None, "norm": h, "k_eq_v": False}
    shared = {"n": n_h, "np": M, "at": hyb, "kind": "full", "window": None, "attn": a, "mla": None, "rec": None,
              "ffn": {"glu": True, "inter": inter, "bias": False}, "moe": None, "norm": 2 * h + h, "k_eq_v": False}
    glue_W = h * h                                          # the hybrid layer's own linear
    if mt == "zamba2":
        r = int(d.get("adapter_rank") or 128)
        if d.get("use_shared_attention_adapter"):
            glue_W += 3 * (a["in_dim"] * r + r * nh * hd)
        if d.get("use_shared_mlp_adapter", True):
            glue_W += h * r + r * 2 * inter
    glue = {"n": n_h, "np": n_h, "at": hyb, "kind": "none", "window": None, "attn": None, "mla": None,
            "rec": {"family": "glue", "W": glue_W, "b": 0, "conv_state": 0, "ssm_state": 0, "core_flops": 0},
            "ffn": None, "moe": None, "norm": 0, "k_eq_v": False}
    p["groups"] = [rec_g, shared, glue]
    p["order_exact"] = True


def _nemotron_h(p, d, raw):
    """Nemotron-H (and CENO, the Nemotron-H-Puzzle): one block per pattern entry -- M a Mamba-2 mixer, * attention,
    - a non-gated MLP, E a MoE (non-gated experts, optionally in a moe_latent_size latent, plus one non-gated shared
    expert) -- each with one pre-norm and no FFN of its own (transformers v5.23 modeling_nemotron_h.py, read as text)."""
    h = p["hidden"]
    pat = raw.get("hybrid_override_pattern") or d.get("hybrid_override_pattern")
    bc = d.get("block_configs") if isinstance(d.get("block_configs"), list) else None
    if isinstance(pat, str) and pat:
        blocks = [{"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}.get(ch, "mlp") for ch in pat]
    else:
        lbt = d.get("layers_block_type") or d.get("layer_types") or []
        mp = {"linear_attention": "mamba", "full_attention": "attention"}
        blocks = [mp.get(str(x), str(x)) for x in lbt]
    per_block = [dict(b) if isinstance(b, dict) else {} for b in (bc or [{}] * len(blocks))]
    dd = dict(d)
    dd.setdefault("mamba_n_heads", d.get("mamba_num_heads"))
    dd.setdefault("mamba_d_head", d.get("mamba_head_dim"))
    dd.setdefault("mamba_n_groups", d.get("n_groups", d.get("mamba_n_groups", 1)))
    dd.setdefault("mamba_d_state", d.get("ssm_state_size", d.get("mamba_d_state", 128)))
    dd.setdefault("mamba_d_conv", d.get("conv_kernel", d.get("mamba_d_conv", 4)))
    dd["mamba_d_ssm"] = int(dd["mamba_n_heads"]) * int(dd["mamba_d_head"])
    dd["mamba_conv_bias"] = d.get("use_conv_bias", True)
    dd["mamba_proj_bias"] = d.get("use_bias", False)
    rec = _mamba2(dd, h, "nemotron_h")
    # head_dim, else the remote config's attention_head_dim (Nemotron-H-4B remote modeling_nemotron_h.py:845-848,
    # read as text: config.attention_head_dim when set, else hidden // heads)
    nh = int(d["num_attention_heads"])
    hd = int(d.get("head_dim") or d.get("attention_head_dim") or h // nh)
    nkv = int(d.get("num_key_value_heads") or nh)
    attn = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": bool(d.get("attention_bias")), "o_bias": False,
            "qk_norm": 0, "out_gate": False}
    glu = d.get("mlp_use_swiglu") is True
    mlp_b = bool(d.get("mlp_bias"))
    E = int(d.get("n_routed_experts") or d.get("num_experts") or 0)
    shared_w = int(d.get("moe_shared_expert_intermediate_size") or 0)
    if d.get("n_shared_experts") == 0:
        shared_w = 0
    G, at = {}, {}
    for i, b in enumerate(blocks):
        cfg = per_block[i] if i < len(per_block) else {}
        if b == "mamba":
            key = ("rec",)
        elif b == "attention":
            key = ("attn",)
        elif b == "moe":
            key = ("moe", int(cfg.get("moe_intermediate_size") or d.get("moe_intermediate_size")),
                   int(cfg.get("num_experts_per_tok") or d.get("num_experts_per_tok") or d.get("moe_top_k") or 1),
                   int(cfg.get("n_routed_experts") or cfg.get("num_experts") or E))
        else:
            key = ("mlp", int(cfg.get("intermediate_size") or d.get("intermediate_size")))
        G[key] = G.get(key, 0) + 1
        at.setdefault(key, []).append(i)
    groups = []
    for key, cnt in G.items():
        g = {"n": cnt, "np": cnt, "at": at[key], "kind": "none", "window": None, "attn": None, "mla": None, "rec": None,
             "ffn": None, "moe": None, "norm": h, "k_eq_v": False}
        if key[0] == "rec":
            g.update(kind="rec", rec=dict(rec))
        elif key[0] == "attn":
            g.update(kind="full", attn=dict(attn))
        elif key[0] == "moe":
            g["moe"] = {"E": key[3], "k": key[2], "ew": key[1], "shared": shared_w, "shared_gate": False,
                        "router_bias": False, "expert_bias": False, "glu": glu, "shared_glu": glu,
                        "latent": int(d.get("moe_latent_size") or 0) or None, "out_bias": False}
        else:
            g["ffn"] = {"glu": glu, "inter": key[1], "bias": mlp_b}
        groups.append(g)
    p["groups"] = groups
    p["layers"] = len(blocks)
    p["order_exact"] = True


REC_BUILDERS = {"mamba1": _mamba1, "mamba1_bcdt": _mamba1, "mamba1_bcdtw": _mamba1, "mamba1_mh": _mamba1,
                "mamba2": _mamba2, "mamba2_nonorm": _mamba2, "mamba2_nbg": _mamba2, "gdn": _gdn, "gdn_negeig": _gdn, "gdn_sig": _gdn, "kda": _kda, "kda_lb_negeig": _kda,
                "shortconv": _shortconv, "rglru": _rglru, "rwkv4": _rwkv4, "lightning": _lightning,
                "bailing_lightning": _bailing_lightning, "gla": _gla, "rwkv7": _rwkv7, "deltanet": _deltanet}


def _recurrent(p, d, raw, cl):
    """Fill the recurrent groups (and the families whose every layer is recurrent) from the family builders."""
    mt, h = p["mt"], p["hidden"]
    fam = _recurrent_family(d, cl.get("mixer", "softmax")) if cl.get("mixer", "softmax") != "softmax" else None
    if fam is None:
        return
    if mt in ("nemotron_h", "nemotron_h_puzzle", "ceno"):
        # its own builder maps the Nemotron-H keys (mamba_num_heads, mamba_head_dim, ...) before _mamba2; the
        # generic builder would fail on them and mark the recurrent state unknown
        _nemotron_h(p, d, raw)
        return
    build = REC_BUILDERS.get(fam)
    rec = None
    if build is not None:
        try:
            rec = build(d, h, mt)
        except (KeyError, TypeError, ValueError) as e:
            p["notes"].append(f"recurrent family {fam}: config keys missing ({e}); recurrent params unknown")
    if rec is None:
        rec = {"family": fam, "W": 0, "b": 0, "conv_state": 0, "ssm_state": 0, "core_flops": 0}
        p.setdefault("unknown", set()).add("rec")
    if mt in ("zamba", "zamba2"):
        _zamba(p, d, rec)
        return
    pure = cl.get("mixer") in ("ssm", "linear")
    if pure:                                                # every layer one recurrent block
        n = p["layers"]
        norm = h * (2 if cl.get("norm") == "LayerNorm" else 1)
        g = {"n": n, "np": n, "at": list(range(n)), "kind": "rec", "window": None, "attn": None, "mla": None,
             "rec": dict(rec), "ffn": None, "moe": None, "norm": norm, "k_eq_v": False}
        p["groups"] = [g]
        p["order_exact"] = True
        if mt == "rwkv":
            I = int(d.get("intermediate_size") or 4 * h)
            g["ffn"] = {"glu": False, "inter": I, "bias": False, "rwkv": True}
            g["norm"] = 2 * 2 * h                           # ln1, ln2 (LayerNorm)
            p["extra"] += 2 * h                             # layer 0 pre_ln
            p["final_norm"] = 2 * h
        elif fam in ("gla", "gdn", "deltanet"):
            # a recurrent mixer in every layer, each with its own SwiGLU MLP and two RMSNorms: fla GLABlock (fla-org
            # flash-linear-attention main, fla/models/gla/modeling_gla.py lines 40-67: attn_norm, attn, mlp_norm,
            # GLAMLP; width fla/modules/mlp.py lines 39-45: 256 x ceil(int(h x hidden_ratio x 2/3) / 256) when
            # intermediate_size is null), AllGDN (Qwen3_5ForCausalLM with every layer linear_attention, remote
            # modeling_allgdn.py lines 19-24, 132-145) and Qwen3.5-GDN24 (remote modeling_qwen35_gdn24.py lines
            # 70-79: linear_attn, SuperSwiGLUMLP, input / post_attention RMSNorm)
            I = d.get("intermediate_size")
            if not isinstance(I, int) or isinstance(I, bool) or not I:
                hr = d.get("hidden_ratio") or 4
                I = int(h * hr * 2 / 3)
                I = 256 * ((I + 255) // 256)
            g["ffn"] = {"glu": True, "inter": int(I), "bias": False}
            g["norm"] = 2 * norm
        elif fam == "rwkv7":
            _rwkv7_layers(p, d, g)
        return
    for g in p["groups"]:
        if g["kind"] == "rec" or mt == "falcon_h1":
            g["rec"] = dict(rec)


# Per family, what the stock transformers 4.57 modeling code builds where the generic plan (feats / classify) is
# not enough: q/k/v bias, o_proj bias, MLP bias, norm bias, LM-head bias, norms per layer. A value that is a string
# is a config key (read with the class default). Code-read: modeling_<type>.py __init__ of the attention / MLP /
# decoder layer / model.
LAYOUT = {
    "gpt2": dict(qkv=True, o=True, mlp=True, nb=True),
    "gpt_bigcode": dict(qkv=True, o=True, mlp=True, nb=True),
    "openai-gpt": dict(qkv=True, o=True, mlp=True, nb=True, final_norm=False),
    "opt": dict(qkv="enable_bias", o="enable_bias", mlp="enable_bias", nb="layer_norm_elementwise_affine"),
    "bloom": dict(qkv=True, o=True, mlp=True, nb=True, emb_norm=True),
    "gpt_neox": dict(qkv="attention_bias", o="attention_bias", mlp=True, nb=True, norms=2),
    "gpt_neox_japanese": dict(qkv=False, o=False, mlp=False, nb=True, norms=2, last_dense_bias=True),
    "gptj": dict(qkv=False, o=False, mlp=True, nb=True, head_bias=True, norms=1),
    "codegen": dict(qkv=False, o=False, mlp=True, nb=True, head_bias=True, norms=1),
    "gpt_neo": dict(qkv=False, o=True, mlp=True, nb=True, norms=2),
    "falcon": dict(qkv="bias", o="bias", mlp="bias", nb=True),
    "mpt": dict(qkv=False, o=False, mlp=False, nb=False, norms=2),
    "xglm": dict(qkv=True, o=True, mlp=True, nb=True),
    "ctrl": dict(qkv=True, o=True, mlp=True, nb=True, head_bias=True),
    "biogpt": dict(qkv=True, o=True, mlp=True, nb=True),
    "phi": dict(qkv=True, o=True, mlp=True, nb=True, head_bias=True, norms=1),
    "persimmon": dict(qkv=True, o=True, mlp=True, nb=True),
    "stablelm": dict(qkv="use_qkv_bias", o=False, mlp=False, nb=True),
    "starcoder2": dict(qkv="use_bias", o="use_bias", mlp="use_bias", nb=True),
    "nemotron": dict(qkv="attention_bias", o="attention_bias", mlp="mlp_bias", nb=True),
    "cohere": dict(qkv="attention_bias", o="attention_bias", mlp=False, nb=False, norms=1),
    "cohere2": dict(qkv="attention_bias", o="attention_bias", mlp=False, nb=False, norms=1),
    "phimoe": dict(head_bias="lm_head_bias", nb=True),
    "recurrent_gemma": dict(qkv="attention_bias", o=True, mlp=True, nb=False),
    "minimax": dict(norms=2),
    "minimax_m1": dict(norms=2),
    "minimax_text_01": dict(norms=2),
}


def _gemma4(p, d):
    """Gemma-4 text stack (transformers v5.23 modeling_gemma4.py / configuration_gemma4.py, read as text): full layers
    use global_head_dim (default 512) and, when attention_k_eq_v, num_global_key_value_heads with no v_proj (V = K);
    the last num_kv_shared_layers layers reuse another layer's K / V (no k / v projections or norms; double-wide MLP
    under use_double_wide_mlp); per-layer embeddings (hidden_size_per_layer_input, default 256: a
    vocab_size_per_layer_input x (layers x hp) table, a model projection, a gate / projection / norm per layer);
    the MoE block (enable_moe_block) beside the dense MLP: router proj + scale + per-expert scale, experts, 3 norms."""
    h = p["hidden"]
    n = int(d["num_hidden_layers"])
    lt = d.get("layer_types") or ["sliding_attention" if (i + 1) % 6 else "full_attention" for i in range(n)]
    lt = list(lt)[:n]
    if lt and lt[-1] != "full_attention":
        lt[-1] = "full_attention"
    nh = int(d["num_attention_heads"])
    hd = int(d.get("head_dim") or 256)
    ghd = int(d.get("global_head_dim") or 512)
    keqv = bool(d.get("attention_k_eq_v"))
    nkv = int(d.get("num_key_value_heads") or nh)
    gkv = d.get("num_global_key_value_heads") if keqv else None
    gkv = int(gkv) if gkv else nkv
    nsh = int(d.get("num_kv_shared_layers") or 0)
    inter = int(d["intermediate_size"])
    wide = bool(d.get("use_double_wide_mlp"))
    hp = d.get("hidden_size_per_layer_input", 256)
    hp = int(hp) if hp else 0
    bias = bool(d.get("attention_bias"))
    W = int(d.get("sliding_window") or 512)
    moe = None
    if d.get("enable_moe_block"):
        E, k, ew = int(d["num_experts"]), int(d.get("top_k_experts") or 1), int(d["moe_intermediate_size"])
        moe = {"E": E, "k": k, "ew": ew, "shared": inter, "shared_gate": False, "router_bias": False,
               "expert_bias": False, "glu": True, "out_bias": False, "gemma4": True}
    # per_layer_config (some gemma4_text exports: {"05": {"head_dim": 512, "num_key_value_heads": 4}, ...}) sets a
    # layer's head_dim / KV heads directly (jgeuter/Jeff-1.0-Large, udgamlabs/Lilavati: the safetensors shapes of
    # those layers' q / k / o projections and norms are these)
    plc = d.get("per_layer_config") if isinstance(d.get("per_layer_config"), dict) else {}
    G, at = {}, {}
    for i, t in enumerate(lt):
        full = t == "full_attention"
        shared = i >= n - nsh
        o = plc.get(f"{i:02d}") or plc.get(str(i)) or {}
        dh = int(o.get("head_dim") or (ghd if full else hd))
        kv = int(o.get("num_key_value_heads") or (gkv if full else nkv))
        key = (full, shared, dh, kv)
        G[key] = G.get(key, 0) + 1
        at.setdefault(key, []).append(i)
    groups = []
    for (full, shared, dh, kv), cnt in G.items():
        tied = keqv and full
        a = {"nh": nh, "nkv": kv, "dqk": dh, "dv": dh, "qkv_bias": bias, "o_bias": bias, "out_gate": False,
             "qk_norm": dh + (0 if shared else dh)}        # q_norm, k_norm (v_norm has no scale)
        if shared:
            a["no_kv_proj"] = True
        g = {"n": cnt, "np": cnt, "at": at[(full, shared, dh, kv)],
             "kind": "shared" if shared else ("full" if full else "sliding"),
             "window": None if full else W, "attn": a, "mla": None, "rec": None,
             "ffn": None if moe else {"glu": True, "inter": inter * (2 if (wide and shared) else 1), "bias": False},
             "moe": dict(moe) if moe else None, "norm": 4 * h, "k_eq_v": tied}
        if moe:
            g["norm"] += 3 * h + h + moe["E"]               # 3 extra norms, router scale, per-expert scale
        if hp:
            g["norm"] += h                                 # post_per_layer_input_norm
            a["extra_W"] = 2 * h * hp                      # per_layer_input_gate, per_layer_projection
        groups.append(g)
    p["groups"] = groups
    p["layers"] = n
    p["order_exact"] = True
    if hp:
        vple = int(d.get("vocab_size_per_layer_input") or p["vocab"])
        p["embedding"] += vple * n * hp                    # embed_tokens_per_layer (an embedding table)
        p["extra_W"] += h * n * hp                         # per_layer_model_projection
        p["extra"] += hp                                   # per_layer_projection_norm


LONGCAT_ARCHS = ("LongcatFlashForCausalLM", "LongcatCausalLM")


def _longcat(p, d):
    """LongCat-Flash / LongCat-2.0 / Flash-Lite (configs without a model_type). transformers 4.57
    models/longcat_flash/modeling_longcat_flash.py, read as text: lines 120-130 the router over
    n_routed_experts + zero_expert_num (zero experts are identities, no weights), lines 413-436 the decoder layer:
    TWO MLA attentions (self_attn[0], self_attn[1]), TWO dense SwiGLU mlps of ffn_hidden_size, one shortcut MoE of
    expert_ffn_hidden_size experts (top moe_topk of E + Z), and four RMSNorms. LongCat-2.0 / Flash-Lite (remote
    modeling_longcat_ngram.py of LongCat-Flash-Lite, lines 71-104, read as text) add a DSA indexer on self_attn[0]
    only (cli_factor 2: the second attention reuses its selection; k_norm an RMSNorm) and the n-gram
    over-tokenizer: oe_split_num x (oe_neighbor_num - 1) embedding tables of int(oe_vocab_size_ratio x V + 2i + 1)
    rows and h / N columns, each projected back to h. Groups: A = attention 0 + dense MLP 0, B = attention 1 + the
    MoE, C = dense MLP 1 (no attention); num_layers of each."""
    h, V = p["hidden"], p["vocab"]
    n = int(d.get("num_layers") or d.get("num_hidden_layers"))
    nh = int(d["num_attention_heads"])
    rope, nope = int(d["qk_rope_head_dim"]), int(d["qk_nope_head_dim"])
    vd = int(d.get("v_head_dim") or nope)
    mla = {"q_lora": int(d.get("q_lora_rank") or 0), "kv_lora": int(d["kv_lora_rank"]), "rope": rope, "nope": nope,
           "v": vd}
    attn = {"nh": nh, "nkv": nh, "dqk": nope + rope, "dv": vd, "qkv_bias": False, "o_bias": False, "qk_norm": 0,
            "out_gate": False}
    fi, E, Z = int(d["ffn_hidden_size"]), int(d["n_routed_experts"]), int(d.get("zero_expert_num") or 0)
    moe = {"E": E, "k": int(d.get("moe_topk") or d.get("num_experts_per_tok") or 1),
           "ew": int(d["expert_ffn_hidden_size"]), "shared": 0, "shared_gate": False, "router_bias": False,
           "expert_bias": False, "glu": True, "out_bias": False, "zero": Z}
    dense = {"glu": True, "inter": fi, "bias": False}
    a0 = dict(attn)
    if d.get("index_n_heads"):
        ih, idim = int(d["index_n_heads"]), int(d["index_head_dim"])
        ql = mla["q_lora"] or h
        a0.update(extra_W=ql * ih * idim + h * idim + h * ih, extra_b=idim)
        p["dsa"] = {"heads": ih, "dim": idim, "topk": int(d.get("index_topk") or 2048)}
        p["dsa_dim"] = idim
    blank = {"window": None, "rec": None, "k_eq_v": False}
    blank["at"] = list(range(n))
    A = dict(blank, n=n, np=n, kind="full", attn=a0, mla=mla, ffn=dense, moe=None, norm=2 * h)
    B = dict(blank, n=n, np=n, kind="full", attn=dict(attn), mla=mla, ffn=None, moe=moe, norm=h)
    C = dict(blank, n=n, np=n, kind="none", attn=None, mla=None, ffn=dict(dense), moe=None, norm=h)
    if d.get("index_n_heads"):
        A["dsa_index"], B["dsa_index"] = True, False
    p["groups"] = [A, B, C]
    p["order_exact"] = True
    p["layers"] = n
    p["final_norm"] = h
    if d.get("oe_split_num"):
        N = int(d["oe_split_num"]) * (int(d.get("oe_neighbor_num") or 2) - 1)
        dim = h // N
        m = float(d.get("oe_vocab_size_ratio") or 0) * V
        p["embedding"] += sum(int(m + 2 * i + 1) * dim for i in range(N))
        p["extra_W"] += N * dim * h
        p["longcat_oe"] = sum(int(m + 2 * i + 1) * dim + dim * h for i in range(N))
    p["notes"].append(f"LongCat: two MLA attentions, two dense MLPs and one MoE per layer ({n} layers); active "
                      f"params count all {moe['k']} selected experts as real (zero experts, {Z} of {E + Z}, are "
                      "identities, so this is the upper bound)")


def _per_layer_groups(layers):
    """Collapse a list of per-layer group dicts into counted groups (identical layers share one group). `at` lists
    each group's layer indices (the layer order, read by graph/arch_breakdown.py; not part of the identity)."""
    out = {}
    for i, g in enumerate(layers):
        g = {k: v for k, v in g.items() if k != "at"}
        key = json.dumps(g, sort_keys=True, default=str)
        if key in out:
            out[key]["n"] += 1
            out[key]["np"] += 1
            out[key]["at"].append(i)
        else:
            out[key] = dict(g, n=1, np=1, at=[i])
    return list(out.values())


def _blank(kind="full"):
    return {"kind": kind, "window": None, "attn": None, "mla": None, "rec": None, "ffn": None, "moe": None,
            "norm": 0, "k_eq_v": False}


def _nemotron_nas(p, d):
    """DeciLM / Nemotron-NAS (remote modeling_decilm.py, read as text): per block an attention that is a no-op, a
    linear map (replace_with_linear) or GQA with n_heads_in_group, and an FFN that is a no-op, a linear map or a
    SwiGLU of width find_multiple(int(2 x ffn_mult x hidden / 3), 256); each present sublayer has its own RMSNorm."""
    h = p["hidden"]
    nh = int(d["num_attention_heads"])
    hd = int(d.get("head_dim") or h // nh)
    layers = []
    for b in d.get("block_configs") or []:
        a, f = b.get("attention") or {}, b.get("ffn") or {}
        g = _blank()
        if a.get("no_op"):
            g["kind"] = "none"
        elif a.get("replace_with_linear"):
            g["kind"] = "none"
            g["rec"] = {"family": "glue", "W": h * h, "b": 0, "conv_state": 0, "ssm_state": 0, "core_flops": 0}
            g["norm"] += h
        else:
            g["attn"] = {"nh": nh, "nkv": nh // int(a.get("n_heads_in_group") or 1), "dqk": hd, "dv": hd,
                         "qkv_bias": bool(d.get("attention_bias")), "o_bias": bool(d.get("attention_bias")),
                         "qk_norm": 0, "out_gate": False}
            if a.get("window_length"):
                g.update(kind="sliding", window=int(a["window_length"]))
            g["norm"] += h
        if f.get("no_op"):
            pass
        elif f.get("replace_with_linear"):
            g["ffn"] = {"glu": False, "inter": 0, "bias": False, "linear": True}
            g["norm"] += h
        else:
            inter = int(2 * float(f.get("ffn_mult") or 4) * h / 3)
            inter = -(-inter // 256) * 256
            g["ffn"] = {"glu": True, "inter": inter, "bias": bool(d.get("mlp_bias"))}
            g["norm"] += h
        layers.append(g)
    if layers:
        p["groups"] = _per_layer_groups(layers)
        p["layers"] = len(layers)
        p["order_exact"] = True


def _openelm(p, d):
    """OpenELM (remote modeling_openelm.py, read as text): layer-wise query / KV heads and FFN multipliers; a fused
    qkv_proj, q / k RMSNorms over head_dim, a GLU FFN of width make_divisible(mult x model_dim, ffn_dim_divisor)."""
    from graph.model_graph import _make_divisible
    h = p["hidden"]
    hd = int(d["head_dim"])
    nq, nkv, fm = d["num_query_heads"], d["num_kv_heads"], d["ffn_multipliers"]
    div = int(d.get("ffn_dim_divisor") or 256)
    glu = d.get("ffn_with_glu", True)
    layers = []
    for i in range(int(d["num_transformer_layers"])):
        g = _blank()
        g["attn"] = {"nh": int(nq[i]), "nkv": int(nkv[i]), "dqk": hd, "dv": hd, "qkv_bias": False, "o_bias": False,
                     "qk_norm": 2 * hd if d.get("normalize_qk_projections") else 0, "out_gate": False}
        g["ffn"] = {"glu": bool(glu), "inter": _make_divisible(float(fm[i]) * h, div), "bias": False}
        g["norm"] = 2 * h
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)
    p["layers"] = len(layers)
    p["order_exact"] = True


def _layout_value(v, d):
    if isinstance(v, str):
        return bool(d.get(v))
    return v


def _type_fixups(p, d, raw, cl, dims):
    """Per-family structure read from the stock transformers modeling code (4.57), where the generic plan differs."""
    mt = p["mt"]
    h = p["hidden"]
    lay = LAYOUT.get(mt)
    norm = cl.get("norm", "RMSNorm")
    if lay:
        nb = _layout_value(lay.get("nb", norm == "LayerNorm"), d)
        unit = h * (2 if nb else 1)
        if lay.get("final_norm") is False:
            p["final_norm"] = 0
        else:
            p["final_norm"] = unit
        if _layout_value(lay.get("head_bias", False), d):
            p["extra"] += p["vocab"]                       # lm_head bias
        if lay.get("emb_norm"):
            p["extra"] += 2 * h                            # Bloom word_embeddings_layernorm
        if mt == "opt":
            wp = int(d.get("word_embed_proj_dim") or h)
            if wp != h:                                    # project_in / project_out around the stack
                p["embedding"] = p["vocab"] * wp + (p["embedding"] - p["vocab"] * h)
                p["lm_head_W"] = p["vocab"] * wp
                p["lm_head"] = 0 if p["tied"] else p["vocab"] * wp
                p["extra_W"] += 2 * wp * h
            if not d.get("do_layer_norm_before", True) or d.get("_remove_final_layer_norm"):
                p["final_norm"] = 0
        if lay.get("last_dense_bias"):
            p["extra"] += h                                # GPT-NeoX-Japanese: dense_bias on the last layer only
    if mt == "moshi":
        # the temporal transformer: text embedding (vocab + 1 rows, padding), one audio-codebook embedding per
        # codebook of each of the two streams (audio_vocab_size + 1 rows each); the depth decoder is a separate tower (not counted)
        nq = int(d.get("num_codebooks") or 8)
        av = int(d.get("audio_vocab_size") or 2048)
        p["embedding"] = (p["vocab"] + 1) * h + 2 * nq * (av + 1) * h     # user + model streams
        p["notes"].append("depth decoder (per-step codebook generation) not counted")
    for g in p["groups"]:
        a, f = g["attn"], g["ffn"]
        if mt == "moshi" and f is not None:
            g["ffn"] = f = dict(f, inter=int(d.get("ffn_dim", 4 * h)) // 2)   # MoshiGatingMLP fc1: h -> ffn_dim
        if mt == "recurrent_gemma" and f is not None:
            g["ffn"] = f = dict(f, inter=int(d["intermediate_size"]) // 2)    # RecurrentGemmaMlp
        if lay and a is not None:
            a["qkv_bias"] = _layout_value(lay.get("qkv", a["qkv_bias"]), d)
            a["o_bias"] = _layout_value(lay.get("o", a["o_bias"]), d)
        if lay and f is not None:
            f["bias"] = _layout_value(lay.get("mlp", f["bias"]), d)
        if lay:
            nl = lay.get("norms", 2)
            if mt == "stablelm":
                nl = 1 if d.get("use_parallel_residual") else 2
            if mt == "falcon":
                if not d.get("parallel_attn", True):
                    nl = 2
                else:
                    nl = 2 if (d.get("num_ln_in_parallel_attn") or (2 if d.get("new_decoder_architecture") else 1)) == 2 \
                        else 1
            g["norm"] = nl * unit
        if a is None:
            continue
        hd = a["dqk"]
        if mt == "falcon":
            if d.get("new_decoder_architecture"):
                p["falcon_new_arch"] = True
                a["nkv"] = int(d.get("num_kv_heads") or a["nh"])
            elif d.get("multi_query"):
                a["nkv"] = 1
            else:
                a["nkv"] = a["nh"]
            if f is not None:
                f["inter"] = int(d.get("ffn_hidden_size") or 4 * h)
        if mt == "mpt":
            a["nkv"] = a["nh"]                              # stock MptAttention: Wqkv = 3 x hidden
            if f is not None:
                f["inter"] = 4 * h                          # MptMLP: 4 x hidden whatever expansion_ratio says
        if mt == "ctrl" and f is not None:
            f["inter"] = int(d.get("dff") or 4 * h)
        if mt == "gpt_neox_japanese" and f is not None:
            f["inter"] = int(h * d.get("intermediate_multiple_size", 4))
        if mt == "persimmon" and d.get("qk_layernorm", True):
            a["qk_norm"] = 4 * hd                           # q_layernorm / k_layernorm: LayerNorm(head_dim), w + b
        if mt == "stablelm":
            a["qk_norm"] = (a["nh"] + a["nkv"]) * hd if d.get("qk_layernorm") else 0
        if mt == "cohere":
            a["qk_norm"] = (a["nh"] + a["nkv"]) * hd if d.get("use_qk_norm") else 0
        if mt == "cohere2":
            a["qk_norm"] = 0
        if mt == "apertus":
            g["norm"] += 2                                  # XIELU activation: alpha_p, alpha_n per MLP
        if mt == "gpt_oss":
            a["sinks"] = a["nh"]                            # GptOssAttention.sinks: one learned logit per head
        if mt == "doge":
            # Doge dynamic mask attention: dt_proj (KV heads x head_dim -> KV heads), A (KV heads); and the
            # decoder layer's input_residual / post_attention_residual scales (hidden each)
            a["extra_W"] = a["nkv"] * hd * a["nkv"]
            a["extra_b"] = a["nkv"]
            g["norm"] += 2 * h
        if mt in ("llama4", "llama4_text") and g["moe"] is not None:
            g["moe"] = dict(g["moe"], shared=int(d.get("intermediate_size")))   # Llama4TextMoe.shared_expert, always
        if mt in ("llama4", "llama4_text") and f is not None:
            g["ffn"] = dict(f, inter=int(d.get("intermediate_size_mlp") or f["inter"]))   # dense layers' MLP
        if mt == "jetmoe":
            # JetMoE mixture-of-attention: E attention experts (input_linear h -> D, output_linear D -> h, D = KV heads x
            # kv_channels), top-k routed, a shared kv_proj (h -> 2 D), a router (h -> E), an output bias (h)
            E, k = int(d["num_local_experts"]), int(d["num_experts_per_tok"])
            D = a["nkv"] * int(d.get("kv_channels") or hd)
            a.update(moa=dict(E=E, k=k, D=D), dqk=int(d.get("kv_channels") or hd), dv=int(d.get("kv_channels") or hd),
                     nh=k * a["nkv"])
    _remote_fixups(p, d, raw, cl, dims)


NANOCHAT_TYPES = ("nanochat_gpt", "gaggle", "sn38-nanochrono")
DSV4_TYPES = ("deepseek_v4", "memt_deepseek_v4", "deepseek_v41_text", "deepseek_v41")


def _dsv4_layers(d, n):
    """Per layer of a DeepSeek-V4 config: (compress ratio, hash-routed MoE)."""
    lt = d.get("layer_types")
    if isinstance(lt, list) and isinstance(d.get("compress_rates"), dict):
        cr = d["compress_rates"]
        ratios = [0 if t == "sliding_attention" else int(cr.get(t, 0)) for t in lt[:n]]
    else:
        ratios = [int(x) for x in list(d.get("compress_ratios") or [0] * n)[:n]]
    ratios += [0] * (n - len(ratios))
    mlt = d.get("mlp_layer_types")
    if isinstance(mlt, list):
        hashl = [t == "hash_moe" for t in mlt[:n]]
    else:
        hashl = [i < int(d.get("num_hash_layers") or 0) for i in range(n)]
    hashl += [False] * (n - len(hashl))
    return ratios, hashl


def _deepseek_v4(p, d, raw):
    """DeepSeek-V4 (deepseek_v4, MemT-1.0-Flash) and V4.1 (deepseek_v41_text) parameters, read as text from the
    local transformers-5 port of DeepSeek-V4 (modeling_deepseek_v4.py lines 156-171 the compressor, 284-288 the
    indexer, 381-393 the attention, 510-512 the MoE gate) and the V4.1 configs' layer-id lists. The generic plan
    keeps the KV / FLOPs shape (one K = V head of head_dim, windows); this overrides the parameter blocks:
      attention   wq_a h -> q_lora, q_norm, wq_b q_lora -> heads x hd, wkv h -> hd (the single K = V head), kv_norm,
                  grouped output LoRA wo_a [o_groups x o_lora, heads x hd / o_groups], wo_b [h, o_groups x o_lora],
                  attn_sink (heads)
      compressor  (V4: compress ratio r > 0) ape [r, cd], wkv / wgate h -> cd, norm (hd); cd = 2 hd when r == 4
      indexer     (V4: r == 4) wq_b q_lora -> heads_i x dim_i, weights_proj h -> heads_i, its own compressor
      hyper-conn. per layer attention and FFN mixers: fn [(2 + hc) hc, hc h], base [(2 + hc) hc], scale [3]
      MoE         router weight E x h, gate.bias E except on hash-routed layers, E routed and n_shared SwiGLU experts
      globals     embedding, head, final norm, hc_head (fn [hc, hc h], base [hc], scale [1]; not in V4.1)
    V4.1: the compressor sits on kv_source_layer_ids (kv_proj, kv_norm, gate_proj when the ratio > 1, k_proj,
    k_norm), the indexer's q / weights projections on index_source_layer_ids, Engram on engram_layer_ids
    (q / k weights [hc, h], wkv [(hc + 1) h, (ngram - 1) x heads x dim]) with global engram_tables. MemT adds its
    memory module on memt.layers (MemTensor modeling_memt.py lines 58-69)."""
    mt = p["mt"]
    h, V = p["hidden"], p["vocab"]
    n = p["layers"]
    nh, hd, ql = int(d["num_attention_heads"]), int(d["head_dim"]), int(d["q_lora_rank"])
    og, ol = int(d["o_groups"]), int(d["o_lora_rank"])
    hc = int(d.get("hc_mult") or 4)
    mix = (2 + hc) * hc
    ih, idim = int(d.get("index_n_heads") or 0), int(d.get("index_head_dim") or 0)
    v41 = mt.startswith("deepseek_v41")
    ratios, hashl = _dsv4_layers(d, n)
    kvsrc = set(d.get("kv_source_layer_ids") or [])
    idxsrc = set(d.get("index_source_layer_ids") or [])
    eng = set(d.get("engram_layer_ids") or [])
    memt = d.get("memt") if isinstance(d.get("memt"), dict) else None
    mlay = set((memt or {}).get("layers") or [])
    W0 = h * ql + ql * nh * hd + h * hd + nh * hd * ol + og * ol * h
    b0 = ql + hd + nh

    def comp(r, dd):
        cd = 2 * dd if r == 4 else dd
        return 2 * cd * h, r * cd + dd

    def per(i, g):
        if g["attn"] is None:
            return g
        W, b = W0 + 2 * mix * hc * h, b0 + 2 * (mix + 3)
        r = ratios[i]
        has_idx = False
        if not v41:
            if r:
                w, bb = comp(r, hd)
                W, b = W + w, b + bb
            if r == 4 and ih:
                w, bb = comp(4, idim)
                W, b = W + ql * ih * idim + h * ih + w, b + bb
                has_idx = True
        else:
            if i in kvsrc:
                W += hd * h + (hd * h if r > 1 else 0) + idim * hd
                b += hd + idim
            if i in idxsrc:
                W += ql * ih * idim + h * ih
                has_idx = True
            if i in eng:
                W += 2 * hc * h + (hc + 1) * h * (int(d["engram_max_ngram_size"]) - 1) * int(d["engram_n_heads"]) \
                    * int(d["engram_head_dim"])
        if memt and i in mlay:
            Hm, dm = int(memt["num_memory_heads"]), int(memt["head_dim"])
            qr, kr, G, gr = int(memt["q_rank"]), int(memt["kv_rank"]), int(memt["kv_groups"]), int(memt["gate_rank"])
            nkvm = int(memt.get("num_kv_heads") or Hm)
            W += h * (qr + 2 * G * kr + 4 * gr) + Hm * dm * qr + 2 * nkvm * dm * kr + 4 * Hm * dm * gr
            b += 2 * G * kr + 2 * Hm * dm + Hm + dm + qr
        g = dict(g, attn=dict(g["attn"], W_override=W, b_override=b))
        if p.get("dsa_dim"):
            g["dsa_index"] = has_idx
        if g["moe"] is not None:
            g["moe"] = dict(g["moe"], router_bias=not hashl[i] and not v41)
        return g
    _regroup_layers(p, d, raw, per)
    if not v41:
        p["extra_W"] += hc * hc * h                        # hc_head_fn
        p["extra"] += hc + 1                               # hc_head_base, hc_head_scale
    if v41 and isinstance(d.get("engram_num_embeddings"), list):
        p["embedding"] += sum(int(x) for x in d["engram_num_embeddings"]) * int(d["engram_head_dim"])
    p["dsv4_hash_layers"] = sum(hashl)


def _regroup_layers(p, d, raw, fn):
    """Split a generic plan (groups keyed by layer kind and MoE, as plan() builds them) into per-layer copies,
    apply fn(i, group) -> group to each, and merge identical layers back into counted groups."""
    n = p["layers"]
    kinds, _ = _layer_kinds(d, raw, n)
    has_moe = any(g["moe"] is not None for g in p["groups"])
    mask = (_moe_layer_mask(d, n) or [True] * n) if has_moe else [False] * n
    layers = []
    for i, k in enumerate(kinds):
        t = next((g for g in p["groups"] if g["kind"] == k and (g["moe"] is not None) == bool(mask[i])), None)
        if t is None:
            return False
        layers.append(fn(i, {kk: v for kk, v in t.items() if kk not in ("n", "np")}))
    p["groups"] = _per_layer_groups(layers)
    return True


def _remote_fixups(p, d, raw, cl, dims):
    """Per-family structure of remote (Hub) modeling code, read as text and never run, where the generic plan
    differs; each block cites the file and lines read. Applied at the end of _type_fixups."""
    mt = p["mt"]
    h = p["hidden"]
    V = p["vocab"]
    n = p["layers"]
    eg = d.get("engram") if isinstance(d.get("engram"), dict) else None
    if mt == "allgdn" and eg:
        # AllGDN Engram modules (remote modeling_allgdn.py lines 76-87, 132-145): per layer in engram.layers a
        # hashed n-gram table (len(orders) x heads x slots rows of dim), key / val (len(orders) x heads x dim -> h),
        # two RMS scales nh / nk (h each) and a depthwise conv (h x 4, no bias)
        O, Hh, S, D = len(eg.get("orders") or (2, 3)), int(eg.get("heads") or 4), int(eg.get("slots") or 65536), \
            int(eg.get("dim") or 128)
        L = len(eg.get("layers") or [])
        p["embedding"] += L * O * Hh * S * D
        p["extra_W"] += L * 2 * O * Hh * D * h
        p["extra"] += L * (2 * h + 4 * h)
    if mt in NANOCHAT_TYPES:
        # nanochat (karpathy/nanochat gpt.py) and its HF ports, read as text: the norms are F.rms_norm with no weight
        # (no norm parameters at all); value embeddings, one V x h table per value-embedding layer, mixed into V by
        # a per-head gate ve_gate (ve_gate_channels -> heads); per-layer resid_lambdas and x0_lambdas, a smear gate
        # (smear channels -> 1, plus smear_lambda) and backout_lambda. Ports: jkminder nanochat_gpt
        # (modeling_nanochat_gpt.py: value_embeds per value_embedding_layers, ve_gate, resid_lambdas, x0_lambdas,
        # smear_gate [1, smear_gate_channels], smear_lambda, backout_lambda); gaggle (modeling_gaggle.py: value
        # embeddings on the layers i with i % 2 == (n - 1) % 2, a relu^2 MLP c_fc / c_proj); sn38-nanochrono
        # (aux_embeds per aux_layers, g_proj [heads, aux_gate_dim], mix_gate [1, mix_gate_dim], mix_scale,
        # tap_scale, residual_scales, skip_scales; relu^2 MLP)
        if mt == "gaggle":
            ve = [i for i in range(n) if i % 2 == (n - 1) % 2]
        else:
            ve = d.get("aux_layers") if mt == "sn38-nanochrono" else d.get("value_embedding_layers")
            ve = ve if isinstance(ve, list) else []
        nh = next((g["attn"]["nh"] for g in p["groups"] if g["attn"] is not None), 0)
        G = int(d.get("ve_gate_channels") or d.get("aux_gate_dim") or 12)
        C = int(d.get("smear_gate_channels") or d.get("mix_gate_dim") or d.get("smear_channels") or 24)
        p["final_norm"] = 0
        for g in p["groups"]:
            g["norm"] = 0
            if g["attn"] is not None:
                g["attn"] = dict(g["attn"], qk_norm=0)
            if mt in ("gaggle", "sn38-nanochrono") and g["ffn"] is not None:
                g["ffn"] = dict(g["ffn"], glu=False)
        p["embedding"] += len(ve) * V * h
        p["extra"] += len(ve) * nh * G + 2 * n + C + 2
    hpl = d.get("num_attention_heads_per_layer")
    if mt == "laguna" and isinstance(hpl, list) and len(hpl) == n:
        # Laguna (poolside remote modeling_laguna.py, read as text: LagunaAttention takes
        # num_attention_heads_per_layer[layer_idx] query heads; with gating the output gate g_proj is head-wise,
        # hidden -> heads, a sigmoid per head; Laguna-XS.2 tensors model.layers.N.self_attn.{q_proj, g_proj, o_proj})
        def per(i, g):
            if g["attn"] is not None:
                g["attn"] = dict(g["attn"], nh=int(hpl[i]), out_gate=bool(d.get("gating", True)),
                                 gate_headwise=bool(d.get("gating", True)))
            return g
        _regroup_layers(p, d, raw, per)
    pb = d.get("pb_hybrid_layers")
    if isinstance(pb, list) and pb and d.get("linear_num_key_heads"):
        # PseudoBrain (PB27Qwen3ForCausalLM; defnoteatin/PseudoBrain-* safetensors model.layers.N.gdn.* on the
        # pb_hybrid_layers): a gated DeltaNet block beside the attention of those layers, with the linear_* keys
        # of the Qwen3-Next / Qwen3.5 GatedDeltaNet (the _gdn builder)
        rec = _gdn(d, h, mt)
        pbs = set(int(i) for i in pb if isinstance(i, int))
        _regroup_layers(p, d, raw, lambda i, g: dict(g, rec=dict(rec)) if i in pbs else g)
    if mt in DSV4_TYPES and d.get("o_groups") and d.get("q_lora_rank"):
        _deepseek_v4(p, d, raw)
    if mt == "arctic" and d.get("use_residual"):
        # Snowflake Arctic (remote modeling_arctic.py, read as text: lines 874-898 ArcticMLP, whose ffn_dim is
        # hidden_size when is_residual_mlp; lines 1047-1049 the decoder layer's residual_mlp and its
        # residual_layernorm under use_residual): an always-on SwiGLU of width h and one more RMSNorm per layer
        for g in p["groups"]:
            if g["moe"] is not None:
                g["moe"] = dict(g["moe"], shared=g["moe"]["shared"] + h)
                g["norm"] += h
    if mt == "hrm_text":
        # HRM-Text (transformers main models/hrm_text/modeling_hrm_text.py, read as text: lines 49-62 the
        # weightless RMSNorm, 164-185 the attention with a fused gate / q / k / v projection [4 x heads x hd, h] and
        # a sigmoid output gate, 279-287 the two stacks H_module and L_module of num_layers_per_stack layers each,
        # 404-414 z_L_init [h]): two weight sets of every layer; per token the L stack runs H_cycles x L_cycles times
        # and the H stack H_cycles times, so the unrolled depth is per_stack x H_cycles x (L_cycles + 1)
        per = int(d.get("num_layers_per_stack") or d.get("num_hidden_layers"))
        Hc, Lc = int(d.get("H_cycles") or 1), int(d.get("L_cycles") or 1)
        p["final_norm"] = 0
        for g in p["groups"]:
            g["norm"] = 0
            g["np"] = 2 * per * g["n"] // max(1, sum(x["n"] for x in p["groups"]))
            g["n"] = g["np"] // 2 * Hc * (Lc + 1)
            if g["attn"] is not None:
                g["attn"] = dict(g["attn"], out_gate=True, gate_headwise=False, qk_norm=0)
        p["layers"] = per * Hc * (Lc + 1)
        p["extra"] += h
        p["notes"].append(f"HRM: two stacks of {per} layers; per token the unrolled depth is {p['layers']} "
                          f"({Hc} H cycles x ({Lc} L cycles + 1))")
    if mt == "iquestloopcoder":
        # LoopGateProjection per layer (remote modeling_iquestloopcoder.py lines 427-434, 878-881): a
        # (heads x head_dim) weight and a (heads) bias, applied on loops 2+
        for g in p["groups"]:
            if g["attn"] is not None:
                a = g["attn"]
                g["attn"] = dict(a, extra_b=a.get("extra_b", 0) + a["nh"] * a["dqk"] + a["nh"])


# ------------------------------------------------------------------------------------------------ bespoke families
# Remote (Hub) families whose structure the generic plan cannot express: each is_<family>(p, d, raw) detects the
# family, and each _<family>(p, d, raw, cl) rebuilds or patches the plan, transcribed from the modelling code read
# as text (never run) and cited by file and line in its docstring. _bespoke runs the first that matches, at the end
# of plan(), after _recurrent. Blocks a checkpoint stores beyond the parameters (buffers, towers, adapters) go in
# p["stored_blocks"] = {name: count}, which hub_components names.

# ---------------------------------------------------------------------------------------------- deep (token routed)

def is_token_routed(p, d, raw):
    return p["mt"] == "deep" and d.get("mlp_type") == "token_routed"


def _token_routed(p, d, raw, cl):
    """Complexity-ML 'deep' token-routed MoE (DeepForCausalLM), transcribed from AETHORIA-AI/TR-HASH-MOE-500M-HF
    modeling_deep.py and configuration_deep.py, read as text, not run:
      expert width   intermediate_size // num_experts (configuration_deep.py line 107); top-k = num_experts_per_tok,
                     or the legacy `top_k` key (lines 52-54); head_dim = hidden // heads (line 105)
      routed experts gate_proj_w / up_proj_w [E, h, I/E], down_proj_w [E, I/E, h] (modeling_deep.py lines 200-202);
                     each token uses its top_k experts (route_weights, lines 191-192, 222-231)
      no router      the route is a fixed token-id table: buffer topk_token_to_expert [top_k, V] (line 195), plus
                     token_to_expert [V] under persist_primary_route (lines 196-199) -- int64 buffers, not params
      shared expert  shared_gate / shared_up h -> shared_intermediate_size, shared_down back (lines 206-209)
      attention      q / k / v / o, no bias (lines 86-89), q_norm / k_norm RMSNorm(head_dim) under use_qk_norm
                     (lines 90-92); rotary inv_freq is a property, not stored (lines 59-64)
      head           tied to embed_tokens (get_output_embeddings, lines 327-331; no lm_head stored)
    Pacific-i64/TR-MOE-306 is the earlier native export of the same model ('use_shared_routed_gates': true, no
    modelling code in the repo; README "fixed top-2 lookup with 0.5/0.5 route weights", "learned scalar gates on
    shared and routed outputs"): its model.safetensors holds token_to_expert [V] (one table), the 0-d
    routed_output_gate / shared_output_gate per layer and a stored rotary inv_freq per layer; export_manifest.json
    "trainable_parameter_estimate": 306486564 is the count below."""
    h, V = p["hidden"], p["vocab"]
    E = int(d.get("num_experts") or 4)
    k = int(d.get("num_experts_per_tok") or d.get("top_k") or 2)
    ew = int(d.get("intermediate_size")) // E
    legacy = bool(d.get("use_shared_routed_gates"))
    n_moe = 0
    for g in p["groups"]:
        if g["moe"] is not None:
            g["moe"] = dict(g["moe"], E=E, k=k, ew=ew, no_router=True,
                            shared=int(d.get("shared_intermediate_size") or 0) if d.get("shared_expert", True) else 0)
            if legacy:
                g["norm"] += 2                              # routed_output_gate, shared_output_gate (0-d)
            n_moe += g["n"]
    rows = 1 if legacy else k + (1 if d.get("persist_primary_route") else 0)
    blocks = {"token-to-expert route tables": n_moe * rows * V}
    if legacy:
        blocks["0-d scalars (the Hub API counts none)"] = -2 * n_moe
    p["stored_blocks"] = dict(p.get("stored_blocks") or {}, **blocks)
    p["notes"].append(f"token-id routed MoE: top-{k} of {E} experts by a fixed token table, no router")


# ---------------------------------------------------------------------------------------------- meditor

def is_meditor(p, d, raw):
    return p["mt"] == "meditor" and bool(d.get("num_routed_experts"))


def _meditor(p, d, raw, cl):
    """Meditor-1 (Yo-Purple MeditorForCausalLM), transcribed from modeling_meditor.py / configuration_meditor.py,
    read as text, not run:
      attention  q / k / v with bias, o without (modeling_meditor.py lines 131-134); a layer is global when
                 (i + 1) % global_attn_every_n_layers == 0, else sliding with window sliding_window_size (lines
                 125-126; the cache keeps the last W keys, lines 164-168); rotary inv_freq persistent (line 73)
      MoE        shared SwiGLU (lines 207-209), router h -> E (line 212), packed routed experts routed_gate /
                 routed_up [E, h, I], routed_down [E, I, h] (lines 215-217), top-k = num_experts_per_tok, and one
                 routed_scale (line 219)
      head       lm_head tied to embed_tokens under tie_word_embeddings (lines 333-336); the checkpoint stores a copy"""
    every = int(d.get("global_attn_every_n_layers") or 0)
    W = int(d.get("sliding_window_size") or 0)

    def per(i, g):
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], qkv_bias=True, o_bias=False)
            if every and W and (i + 1) % every != 0:
                g["kind"], g["window"] = "sliding", W
        if g["moe"] is not None:
            g["norm"] += 1                                  # routed_scale
        return g
    _regroup_layers(p, d, raw, per)


# ---------------------------------------------------------------------------------------------- xing4_0

def is_xing_hc(p, d, raw):
    return p["mt"] == "xing4_0" and bool(d.get("hc_mult"))


def _xing_hc(p, d, raw, cl):
    """Xing-4.0 (XingChen-AGI Xing4_0ForCausalLM, a DeepSeek-V3-style MLA + MoE model) manifold hyper-connections,
    from modeling_xing4_0.py read as text, not run: every decoder layer has attn_hc and ffn_hc
    (lines 498-501), each a Xing4_0HyperConnection (lines 451-462) with hc_fn [(2 + hc) hc, hc x h] applied per
    token to the flattened hc residual streams (F.linear, line 470), hc_base [(2 + hc) hc] and hc_scale [3]; the
    residual is hc_mult streams of width h (line 621), averaged before the final norm (line 632). The stored MTP
    layer (model.layers.<n>: eh_proj, enorm / hnorm, shared_head, its own embedding copy) is not run by the model
    (layers[: num_hidden_layers], line 624) and has no hyper-connections, so they are counted outside the layer
    groups (the MTP block of hub_components is one plain decoder layer)."""
    h = p["hidden"]
    hc = int(d["hc_mult"])
    mix = (2 + hc) * hc
    n = sum(g["np"] for g in p["groups"])
    p["extra_W"] += n * 2 * mix * hc * h
    p["extra"] += n * 2 * (mix + 3)
    p["notes"].append(f"hyper-connections: {hc} residual streams, attn_hc and ffn_hc per layer")


def _gdn2_rec(h, inner, H, D, K, f_rank=None, g_rank=None):
    """One GDN-2 mixer (the GPT-S3 / GPT-X3 / KLM layout, fla's GatedDeltaNet2 parameter names): q / k / v
    h -> inner, depthwise q / k / v short convs (kernel K, no bias), a decay projection f (h -> inner, or low rank
    h -> f_rank -> inner), erase gate b_proj and write gate w_proj (h -> inner), A_log (H), dt_bias (inner), an
    output gate g (h -> inner, or low rank, with a bias of inner), o_norm RMSNorm(D), o_proj inner -> h. The cache
    holds the last K - 1 inputs of each conv and the H x D x D state."""
    f = h * inner if not f_rank else h * f_rank + f_rank * inner
    g = h * inner if not g_rank else h * g_rank + g_rank * inner
    W = 3 * h * inner + f + 2 * h * inner + g + inner * h
    conv = 3 * inner
    b = conv * K + H + inner + inner + D
    return {"family": "gdn2", "W": W, "b": b, "conv_state": conv * (K - 1), "ssm_state": H * D * D,
            "core_flops": 2 * conv * K + 6 * H * D * D, "conv_ch": conv, "conv_k": K}


def _drop_unknown_rec(p):
    u = p.get("unknown")
    if u and "rec" in u:
        u.discard("rec")


def is_gpt_s3x3(p, d, raw):
    return str(raw.get("model_type")) in ("gpts3_hybrid", "gptx3") and bool(raw.get("gdn_head_dim"))


def _gpt_s3x3(p, d, raw, cl):
    """AxiomicLabs GPT-S3 (gpts3_hybrid) and GPT-X3 (gptx3): every (n_gdn + 1)-th layer gated GQA, the rest GDN-2.
    GPT-X3 (modeling_gptx3.py, configuration_gptx3.py):
      layer kind   configuration_gptx3.py 48-49 is_full_attention_layer: (i + 1) % (gdn_layers_per_gqa + 1) == 0
      attention    75 q_gate_proj h -> 2 x heads x hd (q and a sigmoid output gate, 85 / 104), 76-78 k / v / o,
                   79-80 q_norm / k_norm OffsetRMSNorm(hd)
      GDN-2        113-127: q / k / v h -> h, q / k / v_conv1d (h, 1, conv_size), f_proj h -> gdn_head_dim -> h,
                   b_proj / w_proj h -> h, A_log (h / gdn_head_dim), dt_bias (h), g_proj h -> dim -> h (+ bias),
                   o_norm RMSNorm(dim), o_proj; conv history conv_size - 1 (160 / 172), state heads x dim x dim (208)
      block        249-253 ln_1, ln_2 RMSNorm, SwiGLU w_gate / w_up / w_down (238-240); 274-279 wte, ln_f, lm_head
                   tied (_tied_weights_keys, 265)
    GPT-S3 (modeling_gpts3.py, configuration_gpts3.py 49-50 the same layer rule over gdn_layers_per_attention):
      GPT-S3's GDN-2 has full-rank f_proj / g_proj (169-182, g_proj with bias), GatedAttention 257-262 as GPT-X3,
      and Full Attention Residuals: per block attn_res_1 / attn_res_2 (360-361), plus the model's attn_res_f (409),
      each AttnResAggregator a query and a key_norm_weight of h (336-337)."""
    mt = str(raw.get("model_type"))
    h = p["hidden"]
    n = p["layers"]
    s3 = mt == "gpts3_hybrid"
    per = int((d.get("gdn_layers_per_attention") if s3 else d.get("gdn_layers_per_gqa")) or 3)
    D = int(d["gdn_head_dim"])
    K = int((d.get("gdn_conv_size") if s3 else d.get("conv_size")) or 4)
    nh, nkv = int(d["num_attention_heads"]), int(d["num_key_value_heads"])
    hd = int(d.get("head_dim") or h // nh)
    I = int(d["intermediate_size"])
    rec = _gdn2_rec(h, h, h // D, D, K, None if s3 else D, None if s3 else D)
    attn = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": False, "o_bias": False, "qk_norm": 2 * hd,
            "out_gate": True, "gate_headwise": False}
    norm = 2 * h + (4 * h if s3 else 0)
    layers = []
    for i in range(n):
        g = _blank("full" if (i + 1) % (per + 1) == 0 else "rec")
        if g["kind"] == "full":
            g["attn"] = dict(attn)
        else:
            g["rec"] = dict(rec)
        g.update(ffn={"glu": True, "inter": I, "bias": False}, norm=norm)
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)
    p["final_norm"] = h
    if s3:
        p["extra"] += 2 * h                                 # attn_res_f
    _drop_unknown_rec(p)


def is_klm(p, d, raw):
    return str(raw.get("model_type")) == "klm" and bool(raw.get("gdn_heads")) and bool(raw.get("attention_every"))


def _klm(p, d, raw, cl):
    """MinimaLabs KLM (modeling_klm.py, configuration_klm.py), read as text:
      layer kind   configuration_klm.py 55-56: attention when i % attention_every == attention_every - 1, else GDN-2
      attention    141 head_dim = h // heads; 143-150 q h -> heads x hd, k / v h -> kv_heads x hd, gate_proj
                   h -> h (sigmoid output gate), o h -> h, q_norm / k_norm RMSNorm(hd), value_mix (2)
      GDN-2        213-230 inner = gdn_heads x gdn_head_dim: q / k / v h -> inner, q / k / v_conv1d (inner, 1, CONV = 4,
                   line 30), f_proj h -> head_dim -> inner, b_proj / w_proj h -> inner, A_log (heads), dt_bias (inner),
                   g_proj h -> head_dim -> inner (+ bias), o_norm RMSNorm(head_dim), o_proj inner -> h; the conv window
                   keeps CONV - 1 inputs (170-180), the state heads x head_dim x head_dim (193)
      block        271-274 mixer_norm, ffn_norm, SwiGLU gate / up / down (259-261)
      model        315-321 embed (tied: the head is F.linear(x, embed.weight), 355), NgramMemory 101-106: bigram
                   (n_bigrams + 1) x ngram_dim and trigram (n_trigrams + 1) x ngram_dim tables, bigram_proj /
                   trigram_proj ngram_dim -> h applied to every token, and the persistent int key buffers bigram_keys
                   (n_bigrams) / trigram_keys (n_trigrams); attn_res_query / attn_res_norm (2 n + 1) x h; norm"""
    h = p["hidden"]
    n = p["layers"]
    every = int(d["attention_every"])
    H, D = int(d["gdn_heads"]), int(d["gdn_head_dim"])
    nh, nkv = int(d["num_attention_heads"]), int(d["num_key_value_heads"])
    hd = h // nh
    I = int(d["intermediate_size"])
    rec = _gdn2_rec(h, H * D, H, D, 4, D, D)
    attn = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": False, "o_bias": False, "qk_norm": 2 * hd,
            "out_gate": True, "gate_headwise": False, "extra_b": 2}
    layers = []
    for i in range(n):
        g = _blank("full" if i % every == every - 1 else "rec")
        if g["kind"] == "full":
            g["attn"] = dict(attn)
        else:
            g["rec"] = dict(rec)
        g.update(ffn={"glu": True, "inter": I, "bias": False}, norm=2 * h)
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)
    nb, nt, nd = int(d["n_bigrams"]), int(d["n_trigrams"]), int(d["ngram_dim"])
    p["embedding"] += (nb + 1 + nt + 1) * nd
    p["extra_W"] += 2 * nd * h
    p["extra"] += 2 * (2 * n + 1) * h
    p["final_norm"] = h
    p["tied"], p["lm_head"] = True, 0
    p["stored_blocks"] = dict(p.get("stored_blocks") or {}, **{"n-gram key buffers": nb + nt})
    _drop_unknown_rec(p)


def is_cyrah(p, d, raw):
    return str(raw.get("model_type")) == "cyrah" and isinstance(raw.get("layer_types"), list)


def _cyrah_codecs(d):
    """Parameters of the four built-in codecs (modeling_cyrah.py 672-676 under use_builtin_codecs): audio 1057-1083,
    image 1128-1153, 3-D Gaussian 1232-1248, signal 1302-1328; _gn GroupNorm (weight + bias) 968-972, _ResBlock1d /
    _ResBlock2d two GroupNorms and two k = 3 convs with bias (975-996), CyrahRVQ n_q codebooks of size x dim (1007-1014),
    _CrossBlock two LayerNorms, q (dim), kv (2 dim), o (dim), a 2 x dim GELU MLP, all with bias (1191-1200)."""
    ch = int(d.get("codec_hidden") or 64)
    cap = int(d.get("codec_max_channels") or 192)
    L = int(d.get("codec_layers") or 4)
    gn = lambda c: 2 * c
    conv = lambda i, o, k: i * o * k + o

    def conv_codec(cin, kk, nd, rvq):
        """1-D (nd = 1) or 2-D (nd = 2) conv codec: kk the kernel elements of the first / last conv."""
        k3, k4 = 3 ** nd, 4 ** nd                         # the k = 3 convs, the stride-2 (transposed) k = 4 convs
        res = lambda c: 2 * gn(c) + 2 * conv(c, c, k3)
        s = conv(cin, ch, kk)
        cur = ch
        for _ in range(L):
            nxt = min(cur * 2, cap)
            s += conv(cur, nxt, k4) + res(nxt)
            cur = nxt
        s += conv(cur, ch, k3) + gn(ch)
        s += gn(ch) + conv(ch, cur, k3)
        for _ in range(L):
            nxt = max(ch, cur // 2)
            s += res(cur) + conv(cur, nxt, k4)
            cur = nxt
        s += conv(cur, cin, kk)
        return s + rvq

    audio = conv_codec(1, 7, 1, int(d.get("audio_codebooks") or 8) * int(d.get("audio_codebook_size") or 1024) * ch)
    image = conv_codec(3, 9, 2, int(d.get("image_codebook_size") or 8192) * ch)
    signal = conv_codec(int(d.get("signal_channels") or 8), 7, 1, int(d.get("signal_codebook_size") or 1024) * ch)
    c2 = 2 * ch
    C = int(d.get("gaussian_channels") or 14)
    T = max(1, int(d.get("gaussian_tokens") or 256))
    per = max(1, -(-max(1, int(d.get("gaussian_points") or 262144)) // T))
    lin = lambda i, o: i * o + o
    cross = 2 * gn(c2) + lin(c2, c2) + lin(c2, 2 * c2) + lin(c2, c2) + lin(c2, 2 * c2) + lin(2 * c2, c2)
    gauss = lin(C, c2) + lin(c2, c2) + 2 * cross + int(d.get("gaussian_codebook_size") or 8192) * c2 \
        + lin(c2, c2) + lin(c2, per * C)
    return audio + image + gauss + signal


def _cyrah(p, d, raw, cl):
    """Cyrah (Lilgarca/my-model-v01, modeling_cyrah.py), read as text:
      layer kinds  519-533 CyrahDecoderLayer: layer_types "conv" -> CyrahShortConv, "linear_attention" -> GDN-2,
                   otherwise CyrahMLA; a dense CyrahMLP while i < first_k_dense_replace or num_experts <= 1
      short conv   250-258 in_proj h -> 3h, depthwise conv (h, 1, conv_L_cache, bias conv_bias), out_proj h -> h
                   (the LFM2 layout: the _shortconv builder)
      attention    380-395 attn_impl "gqa": q h -> heads x (nope + rope), k h -> kv_heads x (nope + rope),
                   v h -> kv_heads x v_head_dim, q_norm / k_norm RMSNorm(nope + rope), o, and under use_output_gate a
                   gate h -> heads x v_head_dim with bias. Not MLA, although the config has kv_lora_rank
      MLP          176-178 w1 / w3 h -> I, w2 I -> h; 523 / 531 operator_norm, ffn_norm; no CyrahAttnRes while
                   attention_residuals is false (535-538)
      model        619-630 embed_tokens, modality_embed (num_modalities x h), norm, attnres_norm; lm_head tied
                   (_tied_weights_keys, 662); 1379 modality_head h -> len(MODALITIES) (= num_modalities), per token
      codecs       672-676: the audio / image / 3-D / signal codecs are stored in the checkpoint (_cyrah_codecs),
                   non-LM blocks"""
    h = p["hidden"]
    lt = raw["layer_types"]
    n = len(lt)
    nh = int(d["num_attention_heads"])
    nkv = max(1, int(d.get("num_key_value_heads") or 1))
    dqk = int(d.get("qk_nope_head_dim") or 0) + int(d.get("qk_rope_head_dim") or 0)
    dv = int(d.get("v_head_dim") or dqk)
    gate = d.get("use_output_gate", True) is not False
    I = int(d["intermediate_size"])
    sc = _shortconv(d, h, p["mt"])
    attn = {"nh": nh, "nkv": nkv, "dqk": dqk, "dv": dv, "qkv_bias": False, "o_bias": False, "qk_norm": 2 * dqk,
            "out_gate": gate, "gate_headwise": False, "extra_b": nh * dv if gate else 0}
    W = d.get("sliding_window")
    layers = []
    for t in lt:
        if t == "conv":
            g = _blank("rec")
            g["rec"] = dict(sc)
        else:
            g = _blank("sliding" if t == "sliding_attention" else "full")
            g["attn"] = dict(attn)
            if t == "sliding_attention" and W:
                g["window"] = int(W)
        g.update(ffn={"glu": True, "inter": I, "bias": False}, norm=2 * h)
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)
    p["layers"] = n
    M = int(d.get("num_modalities") or 95)
    p["embedding"] += M * h
    p["extra_W"] += M * h
    p["extra"] += h                                         # attnres_norm
    p["final_norm"] = h
    p["tied"], p["lm_head"] = True, 0
    if d.get("use_builtin_codecs"):
        p["stored_blocks"] = dict(p.get("stored_blocks") or {},
                                  **{"built-in codecs (audio / image / 3-D / signal)": _cyrah_codecs(d)})
    _drop_unknown_rec(p)


def _blank_group(kind):
    return {"kind": kind, "window": None, "attn": None, "mla": None, "rec": None, "ffn": None, "moe": None,
            "norm": 0, "k_eq_v": False}


# ------------------------------------------------------------------------------------------------ Dory

def is_dory(p, d, raw):
    lt = d.get("layer_types")
    return (p["mt"] == "dory" and isinstance(lt, list) and len(lt) == int(d.get("num_hidden_layers") or 0)
            and bool(d.get("n_recurrent_layers")))


def _dory(p, d, raw, cl):
    """Dory (NVIDIA Dory1.0, a looped normalized GPT), transcribed from the remote code read as text, not run:
    ProCreations/Dory-Beam-4B code/dory/modeling.py (a port of vLLM's dory.py whose parameter names match the
    checkpoint, line 15):
      DoryLayer, lines 193-206: every physical layer holds ONE sublayer, attention (layer_types sliding_attention /
        full_attention) or a SwiGLU MLP (layer_type "mlp"), and its slerp scale attn_alpha / mlp_alpha [h]; there
        are no norms anywhere (justnorm is weightless, lines 63-66) and no final norm (DoryBackbone, lines 216-221)
      DoryAttention, lines 129-139: q_proj h -> H x hd, k_proj / v_proj h -> KV x hd, g_proj h -> H x hd (a
        sigmoid output gate per channel, line 170), o_proj H x hd -> h, sqk [H x hd], s_gate [H x hd]
      DoryMLP, lines 177-183: gate_proj / up_proj h -> I, down_proj I -> h, suv_gate / suv_up [I]
      DoryForCausalLM, lines 263-269: an untied lm_head plus lm_head.logit_scale [V]
      DoryBackbone.forward, lines 246-259: layers [0, n_in) once, [n_in, n_in + n_rec) n_recurrent_loops times,
        the rest once; each loop attends over its own keys / values (line 8-9, vLLM's per_loop KV mode), so a
        recurrent attention layer caches n_recurrent_loops K / V sets.
    The groups count weight sets (np) and unrolled applications (n)."""
    h, V = p["hidden"], p["vocab"]
    lt = d["layer_types"]
    N = len(lt)
    n_in, n_rec = int(d.get("n_input_layers") or 0), int(d.get("n_recurrent_layers") or 0)
    loops = int(d.get("n_recurrent_loops") or 1)
    I = int(d["intermediate_size"])
    W = int(d.get("sliding_window") or 0) or None
    tmpl = next(g["attn"] for g in p["groups"] if g["attn"] is not None)
    attn = dict(tmpl, qkv_bias=False, o_bias=False, qk_norm=0, out_gate=True, gate_headwise=False,
                extra_b=2 * tmpl["nh"] * tmpl["dqk"])
    G = {}
    order = []
    for i, t in enumerate(lt):
        m = loops if n_in <= i < n_in + n_rec else 1
        kind = "none" if t == "mlp" else ("sliding" if "sliding" in str(t) and W else "full")
        key = (kind, m)
        if key not in G:
            g = _blank_group(kind)
            if kind == "none":
                g["ffn"] = {"glu": True, "inter": I, "bias": False}
                g["norm"] = h + 2 * I                       # mlp_alpha, suv_gate, suv_up
            else:
                g["attn"] = dict(attn)
                g["norm"] = h                               # attn_alpha
                g["window"] = W if kind == "sliding" else None
            g["n"] = g["np"] = 0
            G[key] = g
            order.append(key)
        G[key]["np"] += 1
        G[key]["n"] += m
    p["groups"] = [G[k] for k in order]
    p["layers"] = N + (loops - 1) * n_rec
    p["final_norm"] = 0
    p["extra"] += V                                         # lm_head.logit_scale
    p["notes"][:] = [x for x in p["notes"] if not x.startswith("looped:")]
    p["notes"].append(f"Dory: {N} single-sublayer layers (attention or MLP), {n_rec} of them looped {loops}x; "
                      f"no norms (nGPT)")


# ------------------------------------------------------------------------------------------------ ZAYA1-VL

def is_zaya1_vl(p, d, raw):
    return p["mt"] == "zaya1_vl" and bool(d.get("cca")) and bool(d.get("num_experts"))


def _qwen25_vision(v):
    """Parameters of the Qwen2.5-VL vision encoder of a vision_config (transformers modeling_qwen2_5_vl.py
    Qwen2_5_VisionTransformerPretrainedModel, read as text: patch_embed Conv3d without bias, per block norm1 / norm2
    RMSNorm, qkv with bias, proj with bias, a SwiGLU MLP with biases; the PatchMerger ln_q and its two-layer MLP),
    with the Qwen2_5_VLVisionConfig defaults for missing keys; equal to the stock class on the meta device."""
    hv = int(v.get("hidden_size") or 3584)
    depth = int(v.get("depth") or 32)
    Iv = int(v.get("intermediate_size") or 3420)
    inch = int(v.get("in_channels") or 3)
    ps = int(v.get("patch_size") or 14)
    tp = int(v.get("temporal_patch_size") or 2)
    sm = int(v.get("spatial_merge_size") or 2)
    out = int(v.get("out_hidden_size") or 3584)
    blk = 2 * hv + (3 * hv * hv + 3 * hv) + (hv * hv + hv) + 2 * (Iv * hv + Iv) + (hv * Iv + hv)
    mh = hv * sm * sm
    return hv * inch * tp * ps * ps + depth * blk + hv + (mh * mh + mh) + (out * mh + out)


def _zaya1_vl(p, d, raw, cl):
    """ZAYA1-VL's language model (Zaya1VLForConditionalGeneration: the ZAYA1 decoder, model.*, plus a Qwen2.5-VL
    vision tower and vision-only LoRA adapters). No modelling code ships with the repo; the structure is
    transcribed from transformers 5.18 models/zaya/modeling_zaya.py (read as text, not run) and the checkpoint's
    tensor names (model.safetensors.index.json):
      ZayaCCAProjection, lines 163-202 (qkv.linear_q h -> H x hd, linear_k h -> KV x hd, val_proj1 / val_proj2
        h -> KV x hd / 2, conv_qk.0 the depthwise Conv1d over C = (H + KV) x hd channels, kernel cca_time0,
        conv_qk.1 the grouped Conv1d, H + KV groups, kernel cca_time1; both nn.Conv1d with bias)
      ZayaQKNorm, lines 270-273: temp [KV];  ZayaAttention, lines 372-374: o_proj H x hd -> h
      ZayaResidualScaling, lines 481-487: hidden_states_scale / _bias, residual_scale / _bias [h]; the checkpoint
        stores the residual pair on every MLP block and on the attention blocks of layers 1.. (attn.res_scale of
        layer 0 has hidden_states_* only), and one 4-vector model.res_scale
      ZayaRouter, lines 512-536 + ZayaRouterMLP 496-502: down_proj h -> R (+ bias), router_states_scale [R] on
        layers 1.. (EDA), rmsnorm_eda [R], router_mlp fc1 / fc2 R -> R (+ bias), out_proj R -> E + 1 (the extra
        skip expert, no compute, lines 523-524), balancing_biases [E + 1] a buffer
      ZayaExperts, lines 579-585: E SwiGLU experts of width moe_intermediate_size; the VL checkpoint stores
        linear_fc1 [2 I, h], linear_fc2 [h, I] with I = ffn_hidden_size / 2
      two RMSNorms per layer (attn.input_norm, mlp.input_norm), final_norm, tied embedding.
    The EDA router's weights are counted with the layer's extra attention weights (a per-token matmul). The vision
    tower and the vision LoRA (lora_* on q / k / val_proj / o, rank vision_lora_rank_attn, and on every expert's
    fc1 / fc2, rank vision_lora_rank_mlp) are named stored blocks, not LM parameters."""
    h = p["hidden"]
    nh, nkv, hd = int(d["num_attention_heads"]), int(d["num_key_value_heads"]), int(d["head_dim"])
    E, k = int(d["num_experts"]), int(d.get("moe_router_topk") or d.get("num_experts_per_tok") or 1)
    I = int(d.get("ffn_hidden_size") or 2 * int(d.get("moe_intermediate_size") or 2048)) // 2
    R = int(d.get("zaya_mlp_expansion") or d.get("router_hidden_size") or 256)
    K0, K1 = int(d.get("cca_time0") or 2), int(d.get("cca_time1") or 2)
    C = (nh + nkv) * hd
    conv_W = C * K0 + C * (C // (nh + nkv)) * K1
    router_W = h * R + 2 * R * R + R * (E + 1)
    a0 = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": False, "o_bias": False, "qk_norm": 0,
          "out_gate": False, "gate_headwise": False,
          "extra_W": conv_W + router_W - h * (E + 1),
          "extra_b": 2 * C + nkv + R + 2 * R + R}           # conv biases, temp; router down bias, rmsnorm, fc biases
    moe = {"E": E, "k": k, "ew": I, "shared": 0, "shared_gate": False, "router_bias": False, "expert_bias": False,
           "glu": True, "out_bias": False, "dense_compute": False, "zero": 1}
    n = p["layers"]
    layers = []
    for i in range(n):
        g = _blank_group("full")
        a = dict(a0)
        if i > 0:
            a["extra_b"] += R                               # router_states_scale (EDA)
        g["attn"] = a
        g["moe"] = dict(moe)
        g["norm"] = 2 * h + 2 * h + 4 * h + (2 * h if i > 0 else 0)   # 2 input norms, attn / mlp res_scale
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)
    p["extra"] += 4 * h                                     # model.res_scale
    p["final_norm"] = h
    ra, rm = int(d.get("vision_lora_rank_attn") or 0), int(d.get("vision_lora_rank_mlp") or 0)
    blocks = p.setdefault("stored_blocks", {})
    if d.get("vision_lora") and (ra or rm):
        la = ra * (h + nh * hd) + ra * (h + nkv * hd) + 2 * ra * (h + nkv * hd // 2) + ra * (nh * hd + h)
        lm = E * (rm * (h + 2 * I) + rm * (I + h))
        blocks["vision LoRA adapters"] = n * (la + lm)
    if isinstance(raw.get("vision_config"), dict):
        blocks["vision tower (Qwen2.5-VL vision encoder)"] = _qwen25_vision(raw["vision_config"])
    p["notes"].append("ZAYA1: CCA attention (conv-mixed q / k, delayed half-values), an EDA-routed MoE with a skip "
                      "expert")


# ------------------------------------------------------------------------------------------------ Hybrid-Qwen3 GKA

def is_hybrid_gka(p, d, raw):
    lt = d.get("layer_types")
    return (p["mt"] == "hybrid_qwen3" and isinstance(d.get("gka_config"), dict) and isinstance(lt, list)
            and "GKA" in lt)


def _hybrid_gka(p, d, raw, cl):
    """Hybrid-Qwen3 with Gated KalmaNet layers (amazon GKA-primed-HQwen3), transcribed from amazon's
    hybrid-model-factory, read as text, not run: training/src/hmf/model/hybrid_zoo/models/hybrid_qwen3/
    modeling_hybrid_qwen3.py lines 643-651 (Qwen3GKADecoderLayer: the Qwen3 layer with self.gka as its mixer; the
    norms and the SwiGLU MLP stay) and layers/gated_kalmanet/gka.py GatedKalmaNet.__init__, lines 149-237:
      q_proj h -> Hq x hd, k_proj h -> Hk x hd, v_proj h -> Hv x hd (lines 183-185); E = max(Hq, Hk, Hv)
      a_proj h -> E (no bias, line 187), b_proj h -> E (+ bias, use_beta_gate, lines 189-193), alpha_proj h -> E
        (+ bias, use_alpha_connection, line 196), A_log [E], dt_bias [E] (lines 200-215)
      q / k / v ShortConvolution, depthwise, kernel conv_size, no bias (lines 220-230; v under use_v_conv)
      g_proj h -> E x hd and o_norm FusedRMSNormGated(hd) under use_gate, else RMSNorm(hd) (lines 232-236)
      o_proj E x hd -> h (line 237)
    The cache holds the three conv states and the recurrent pair (h_kk, h_kv), each E x hd x hd (lines 611-668).
    Per token the Chebyshev solve applies h_kk num_iter times (2 E hd^2 each) besides the h_kk / h_kv updates and
    the readout."""
    h = p["hidden"]
    gk = d["gka_config"]
    hd = int(gk.get("head_dim") or d.get("head_dim") or 128)
    Hq = int(gk.get("num_q_heads") or d["num_attention_heads"])
    Hk = int(gk.get("num_k_heads") or Hq)
    Hv = int(gk.get("num_v_heads") or Hq)
    E = max(Hq, Hk, Hv)
    K = int(gk.get("conv_size") or 4)
    it = int(gk.get("num_iter") or 30)
    qd, kd, vd, ed = Hq * hd, Hk * hd, Hv * hd, E * hd
    beta, alpha = gk.get("use_beta_gate", True), gk.get("use_alpha_connection", True)
    vconv, gate = gk.get("use_v_conv", True), gk.get("use_gate", True)
    W = h * (qd + kd + vd) + h * E + (h * E if beta else 0) + (h * E if alpha else 0) + (h * ed if gate else 0) \
        + ed * h
    conv = qd + kd + (vd if vconv else 0)
    b = (E if beta else 0) + (E if alpha else 0) + 2 * E + conv * K + hd
    rec = {"family": "gka", "W": W, "b": b, "conv_state": conv * K, "ssm_state": 2 * E * hd * hd,
           "core_flops": 2 * conv * K + E * hd * hd * (2 * it + 6), "conv_ch": conv, "conv_k": K}
    att = next(g for g in p["groups"] if g["attn"] is not None)
    layers = []
    for t in d["layer_types"]:
        g = {kk: v for kk, v in att.items() if kk not in ("n", "np")}
        if t == "GKA":
            g = dict(g, kind="rec", attn=None, mla=None, rec=dict(rec), window=None)
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)
    p.get("unknown", set()).discard("rec")


# ------------------------------------------------------------------------------------------------ Jet-Nemotron

def is_jet_nemotron(p, d, raw):
    lt = d.get("layer_types")
    ea = d.get("efficient_attention_config")
    return (p["mt"] == "jet_nemotron" and isinstance(lt, list) and isinstance(ea, dict)
            and isinstance(ea.get("jet"), dict))


def _jet_nemotron(p, d, raw, cl):
    """Jet-Nemotron (jet-ai), transcribed from the remote code read as text, not run:
    modeling_jet_nemotron.py lines 341-359 (JetNemotronDecoderLayer: layer_types "attn" a full attention, "swa" the
    same attention with efficient_attention_config.swa.window_size, "jet" a JetBlock; each with
    input_layernorm / post_attention_layernorm and the SwiGLU MLP of lines 67-74), lines 166-169 (q / k / v with
    bias, o_proj without) and jet_block.py JetBlock.__init__, lines 81-148:
      key_dim = heads x head_dim, value_dim = int(key_dim x expand_v), head_v_dim = int(head_dim x expand_v)
      q_proj / k_proj h -> key_dim, v_proj h -> value_dim, b_proj / a_proj h -> heads (no bias, lines 112-116)
      A_log [heads], dt_bias [heads] (lines 118-132)
      dynamic_conv1d (dynamic_conv.py DynamicShortConvolution, lines 55-71): kernel_generator w1 h -> value_dim /
        dconv_generator_reduction (no bias), w2 -> value_dim x conv_size (+ bias): a per-token generated depthwise
        kernel over v
      g_proj h -> value_dim, o_norm FusedRMSNormGated(head_v_dim), o_proj value_dim -> h (lines 146-148)
    The state is the conv cache of v and the gated-delta recurrent state heads x head_dim x head_v_dim."""
    h = p["hidden"]
    ea = d["efficient_attention_config"]
    jc = ea["jet"]
    nh, hk = int(jc.get("num_heads") or 6), int(jc.get("head_dim") or 256)
    ev = float(jc.get("expand_v") or 2.0)
    K = int(jc.get("conv_size") or 4)
    red = jc.get("dconv_generator_reduction")
    kd = nh * hk
    vd = int(kd * ev)
    hv = int(hk * ev)
    Gh = vd if red is None else vd // int(red)
    W = 2 * h * kd + h * vd + 2 * h * nh + h * Gh + Gh * vd * K + h * vd + vd * h
    b = 2 * nh + vd * K + hv
    rec = {"family": "jet", "W": W, "b": b, "conv_state": vd * K, "ssm_state": nh * hk * hv,
           "core_flops": 2 * vd * K + 6 * nh * hk * hv, "conv_ch": vd, "conv_k": K}
    swa = ea.get("swa") if isinstance(ea.get("swa"), dict) else {}
    Wn = int(swa.get("window_size") or 0) or None
    att = next(g for g in p["groups"] if g["attn"] is not None)
    layers = []
    for t in d["layer_types"]:
        g = {kk: v for kk, v in att.items() if kk not in ("n", "np")}
        if t == "jet":
            g = dict(g, kind="rec", attn=None, mla=None, rec=dict(rec), window=None)
        elif t == "swa" and Wn:
            g = dict(g, kind="sliding", window=Wn)
        else:
            g = dict(g, kind="full", window=None)
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)
    p.get("unknown", set()).discard("rec")


# ------------------------------------------------------------------------------------------------ SliderGPT

def is_slider_gpt(p, d, raw):
    return "SliderGPT" in (raw.get("architectures") or []) and bool(d.get("slider_on")) \
        and bool(d.get("slider_n_variables"))


def _slider_gpt(p, d, raw, cl):
    """CrystaLLM-pi SliderGPT (GitHub C-Bone-UCL/CrystaLLM-pi, _models/Slider_model.py at e75c9c7, read as text; the
    Hub repo ships no code): a GPT-2 whose every block holds a SliderEncoder when slider_on (lines 456-457). The
    encoder maps the n_variables condition scalars to extra attention keys / values, once per sequence (lines
    84-112): kv_size = 2 x head_dim x (heads / slider_n_heads_sharing_slider);
      encode_linear    Linear(1, V x kv_size)            weight + bias  (line 108)
      upscale_linear   Linear(kv_size, V x n_hidden)     weight + bias  (line 109)
      downscale_linear Linear(n_hidden, V x kv_size)     weight + bias  (line 110)
      attention_factor Linear(1, 1, bias=False)                         (line 112)
    It runs on the condition prefix, not per token, so its weights are parameters but not per-token matmuls."""
    h = p["hidden"]
    nh = int(d.get("n_head") or d.get("num_attention_heads"))
    hd = h // nh
    V = int(d["slider_n_variables"])
    H = int(d.get("slider_n_hidden") or 64)
    share = int(d.get("slider_n_heads_sharing_slider") or 1)
    kv = 2 * hd * (nh // share)
    per = (V * kv + V * kv) + (kv * V * H + V * H) + (H * V * kv + V * kv) + 1
    p["extra"] += p["layers"] * per
    p["notes"].append("SliderGPT: a condition encoder in every block (run once per sequence, not per token)")


# ------------------------------------------------------------------------------------------------ LFM3-RD

def is_lfm3_rd(p, d, raw):
    return p["mt"] == "lfm3_rd" and bool(d.get("num_motifs")) and bool(d.get("motif_size")) \
        and isinstance(d.get("layer_types"), list)


def _lfm3_rd(p, d, raw, cl):
    """LFM3-RD (cybertruck32489/LFM3-1.7B-RD modeling_lfm3_rd.py at fd41791, read as text; the -v2 file differs only
    in an import shim): num_prelude_layers ordinary LFM2-style layers, a recurrent-depth core of num_motifs phases of
    motif_size layers, num_coda_layers ordinary layers (lines 899-913), every layer a mixer (conv or attention) plus a
    SwiGLU w1 / w3 / w2 (lines 349-354) and four RMSNorms (operator / ffn, and post_operator / post_ffn under
    use_sandwich_norm, lines 633-637).
      attention (lines 418-445): q / k / v / o, q_norm / k_norm (head_dim each), a head-wise output gate gate_proj
                h -> heads
      conv      (lines 515-538): depthwise Conv1d (h x conv_L_cache, bias = conv_bias), in_proj h -> 3h, out_proj
    In the core (_core_pos, lines 410-415) the mixer projections are Lfm3SharedLoRALinear / GroupedLoRALinear: a
    weight in the shared bank (Lfm3SharedCoreBank, lines 77-90: one q / k / v / o set for motif position 0, one for
    position 2, one conv in / out set for positions 1 and 3) plus a per-layer LoRA B [out, r] / A [r, in] (lines
    93-103; grouped: B [g, h, r], A [g, r, h], lines 141-153; r = rank_attn for q / o, rank_kv for k / v, rank_conv
    for the conv). In eval the LoRA is merged into the shared weight (lines 120-135, 170-182), so each core layer
    applies one dense projection per token. The manual mode runs recurrent_loops steps, one phase per step (lines
    1102-1124): with recurrent_loops == num_motifs every layer runs once per token.
    The rd_controller (lines 195-225: act / vel / inj / ref norms, state_proj / vel_proj h -> 128, dyn_proj 4 -> 128,
    halting_head 128 -> 1, step_scales [32], inj_A / inj_B rank 64, inj_gates [6], ref_A / ref_B [num_motifs, 64, h],
    refine_gates [32], refine_token_proj h -> 1) only acts beyond num_motifs steps: parameters, not per-token
    matmuls."""
    h = p["hidden"]
    lt = d["layer_types"]
    n = len(lt)
    nh = int(d["num_attention_heads"])
    nkv = int(d.get("num_key_value_heads") or nh)
    hd = int(d.get("head_dim") or h // nh)
    q, kvd = nh * hd, nkv * hd
    K = int(d.get("conv_L_cache") or 3)
    cb = bool(d.get("conv_bias"))
    inter = next(g["ffn"]["inter"] for g in p["groups"] if g["ffn"] is not None)
    W = d.get("sliding_window")
    pre, M, S = int(d.get("num_prelude_layers") or 0), int(d["num_motifs"]), int(d["motif_size"])
    ra, rkv, rc = int(d["rank_attn"]), int(d["rank_kv"]), int(d["rank_conv"])
    G = int(d.get("conv_lora_groups") or 3)
    nn = 4 * h if d.get("use_sandwich_norm") else 2 * h
    ffn = {"glu": True, "inter": inter, "bias": False}
    conv_rec = {"family": "shortconv", "W": 3 * h * h + h * h, "b": h * K + ((3 * h + h) if cb else 0),
                "conv_state": h * K, "ssm_state": 0, "core_flops": 2 * h * K, "conv_ch": h, "conv_k": K, "W_raw": 0,
                "readout": 0}

    def kind(t):
        return "rec" if t not in ("full_attention", "sliding_attention") else (
            "sliding" if t == "sliding_attention" and W else "full")

    def attn(gate):
        return {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": False, "o_bias": False,
                "qk_norm": 2 * hd if gate else 0, "out_gate": gate, "gate_headwise": gate}

    def blank(k):
        return {"kind": k, "window": int(W) if k == "sliding" else None, "attn": None, "mla": None, "rec": None,
                "ffn": None, "moe": None, "norm": 0, "k_eq_v": False}
    layers = []
    for i in list(range(pre)) + list(range(pre + M * S, n)):
        k = kind(lt[i])
        g = blank(k)
        if k == "rec":
            g["rec"] = dict(conv_rec)
        else:
            g["attn"] = attn(True)
        g.update(ffn=dict(ffn), norm=nn)
        layers.append(g)
    G_ = _per_layer_groups(layers)
    for pos in range(S):
        k = kind(lt[pre + pos])
        mix = blank(k)                                     # the shared-bank projections: M applications, 1 weight set
        if k == "rec":
            mix["rec"] = dict(conv_rec, b=0)
            lora = G * (h * rc + rc * h) + (h * rc + rc * h)
            own_W, own_b = 0, h * K + ((3 * h + h) if cb else 0)
        else:
            mix["attn"] = attn(False)
            lora = (q * ra + ra * h) + 2 * (kvd * rkv + rkv * h) + (h * ra + ra * q)
            own_W, own_b = h * nh, 2 * hd                  # gate_proj (head-wise), q_norm / k_norm
        mix.update(n=M, np=1)
        own = blank("none")                                # per core layer: LoRA, gate / norms / conv, FFN, norms
        own.update(n=M, np=M, ffn=dict(ffn), norm=nn,
                   rec={"family": "glue", "W": own_W, "b": own_b + lora, "conv_state": 0, "ssm_state": 0,
                        "core_flops": 0})
        G_ += [mix, own]
    p["groups"] = G_
    p["layers"] = n
    R, ri, rr = 128, 64, 64                                # router_dim, inj_rank, ref_rank (lines 201, 213, 219)
    p["extra"] += (4 * h + 2 * h * R + (4 * R + R) + (R + 1) + 32 + 2 * ri * h + 6 + 2 * M * rr * h + 32 + (h + 1))
    p["notes"].append(f"LFM3-RD: a {M} x {S}-layer core on a shared weight bank with per-layer LoRA; "
                      f"{int(d.get('recurrent_loops') or M)} recurrent steps by default")


# ------------------------------------------------------------------------------------------------ Domyn-Edge

def is_domynedge(p, d, raw):
    hs = d.get("head_sizes")
    return p["mt"] == "domynedge" and isinstance(hs, list) and len(hs) == p["layers"]


def _domynedge(p, d, raw, cl):
    """Domyn-Edge (domyn/Domyn-Edge-Base modeling_domynedge.py / configuration_domynedge.py at cb83e26, read as
    text): DomynEdgeAttention takes head_dim = per_layer_config[i].head_dim, i.e. head_sizes[i] (configuration
    lines 122-124; modeling line 203), for q / k / v / o (lines 211-222); q_norm has no weight, k_norm one of
    head_dim (lines 223-224); four RMSNorms per layer, pre / post attention and feed-forward (lines 282-285)."""
    hs = d["head_sizes"]
    h = p["hidden"]

    def per(i, g):
        if g["attn"] is not None:
            hd = int(hs[i])
            g["attn"] = dict(g["attn"], dqk=hd, dv=hd, qk_norm=hd)
        g["norm"] = 4 * h
        return g
    _regroup_layers(p, d, raw, per)


# ------------------------------------------------------------------------------------------------ K2-Horizon MoVA

def is_k2_horizon_mova(p, d, raw):
    return p["mt"] == "k2_horizon" and int(d.get("mova_num_experts") or 0) > 0


def _k2_horizon_mova(p, d, raw, cl):
    """K2-Horizon (inception42/Jais-3-MoVA-36B-A4B modeling_k2_horizon.py at 44dd2b3, read as text). A sparse layer
    (not in mlp_only_layers, every decoder_sparse_step, lines 646-647) with mova_num_experts > 0 uses
    K2HorizonMoVAAttention (lines 351-410): v_proj is replaced by mova_num_experts value experts, each
    Linear(h, kv_heads x head_dim) without bias (lines 384-390), top mova_num_experts_per_tok routed per token by
    v_router Linear(h, mova_num_experts, bias = moe_gate_bias) (lines 380-383, 428-443). The sparse MoE gate also
    has a bias under moe_gate_bias (line 542)."""
    h = p["hidden"]
    Ev, kv_ = int(d["mova_num_experts"]), int(d.get("mova_num_experts_per_tok") or 1)
    rb = bool(d.get("moe_gate_bias"))
    for g in p["groups"]:
        if g["moe"] is None:
            continue
        g["moe"] = dict(g["moe"], router_bias=rb)
        a = g["attn"]
        if a is None:
            continue
        Wv = h * a["nkv"] * a["dv"]
        g["attn"] = dict(a, extra_W=a.get("extra_W", 0) + (Ev - 1) * Wv + h * Ev,
                         W_inactive=a.get("W_inactive", 0) + (Ev - kv_) * Wv,
                         extra_b=a.get("extra_b", 0) + (Ev if rb else 0))
    p["notes"].append(f"MoVA: {Ev} routed value experts (top {kv_}) replace v_proj on the sparse layers")


# ------------------------------------------------------------------------------------------------ Qwen4-Exp

def is_qwen4_exp(p, d, raw):
    # indexer_n_heads unset: a config the code cannot build (Qwen4ExpTextAttention reads it), left to the generic plan
    return p["mt"] in ("qwen4_exp_text", "qwen4_exp") and d.get("indexer_n_heads") is not None


def _is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    k = 3
    while k * k <= v:
        if v % k == 0:
            return False
        k += 2
    return True


def _ngram_rows(base, heads, layer_index, div):
    """Qwen4ExpTextNGramEmbedding rows (modeling_qwen4_exp.py lines 1093-1109, with _find_nth_prime_after / _is_prime
    lines 1056-1073): head j of PLE layer index l has the (l x heads + j + 1)-th prime after base - 1 rows; the total
    is padded up to a multiple of make_ngram_vocab_size_divisible_by."""
    total, prime, found = 0, base - 1, 0
    first = layer_index * heads
    while found < first + heads:
        prime += 1
        while not _is_prime(prime):
            prime += 1
        if found >= first:
            total += prime
        found += 1
    return -(-total // div) * div


def _qwen4_exp(p, d, raw, cl):
    """Qwen4-Exp text model (transformers main src/transformers/models/qwen4_exp/modeling_qwen4_exp.py, read as text,
    not run; the snapshot repos ship no modelling code):
      decoder layer   (lines 1256-1270) no input / post-attention RMSNorm: two Qwen4ExpTextGatedResidual
                      hyper-connections (attn_ / mlp_hyper_connection) instead, each (lines 999-1008): hc_norm over
                      hc_count x h, input_mix_weight_down (hc h -> hc_lowrank), input_mix_weight_up (hc_lowrank ->
                      hc h), block_inject_weight (hc h -> hc_count)
      attention       (lines 813-838) q_proj h -> 2 x heads x hd (the sigmoid output gate, always), k / v / o,
                      q_norm / k_norm (hd); the QSA indexer (lines 670-688): index_qk_proj h -> (indexer_n_heads +
                      indexer_kv_heads) x indexer_head_dim, q_layernorm / k_layernorm (indexer_head_dim)
      linear layers   Qwen4ExpTextGatedDeltaNet (lines 465-509) = the _gdn builder; MoE (lines 917-983) = experts,
                      router, shared expert and its sigmoid gate (the generic plan)
      PLE             (lines 1180-1210) on the layers i with i + 1 in ple_layer_ids: key_proj ple_embed_dim -> hc h,
                      value_proj ple_embed_dim -> h, norm_key / norm_query / norm_conv (hc h), a depthwise dilated
                      conv1d (hc h x ple_conv_kernel_size, no bias), and the hashed n-gram table
                      (Qwen4ExpTextNGramEmbedding, lines 1076-1109: (ngram_size - 1) x heads_per_ngram heads of
                      ple_embed_dim / heads columns; three int64 buffers layer_multipliers [ngram_size],
                      ngram_heads_vocab_sizes / ngram_heads_offsets [heads], not parameters)
      model           (lines 1370-1375) embed_tokens, hyper_connection_mixer (a GatedResidual without
                      block_inject_weight), no final norm (line 1476: the mixer's output is the last hidden state)
    Older checkpoints (whittleOrg/Whittle-Next-27B-A3B, config key num_embeddings_per_layer_input) keep one model-level
    ngram_embedding of ngram_vocab_size_base rows per head (bf16/model.safetensors.index.json:
    model.ngram_embedding.shard_{0,1} = 976,640 x 8 rows of 256), not the prime-sized per-layer table."""
    h, n = p["hidden"], p["layers"]
    # defaults: transformers main configuration_qwen4_exp.py lines 146-167 (hc_count 4, hc_lowrank 320,
    # ple_embed_dim = hidden_size, ple_conv_kernel_size 4, ngram_size 3, heads_per_ngram 8, ngram_vocab_size_base
    # 20,000,000, make_ngram_vocab_size_divisible_by 128; ple_layer_ids sorted and deduplicated)
    hc, r = int(d.get("hc_count") or 4), int(d.get("hc_lowrank") or 320)
    H = hc * h
    gr_W = H * r + r * H + H * hc                            # one GatedResidual's linears
    nh = int(d["num_attention_heads"])
    ih, ikv, idim = int(d.get("indexer_n_heads") or 0), int(d.get("indexer_kv_heads") or 0), \
        int(d.get("indexer_head_dim") or 0)
    for g in p["groups"]:
        g["norm"] = 2 * H                                    # the two hc_norms
        if g["attn"] is not None:
            a = g["attn"]
            g["attn"] = dict(a, out_gate=True, gate_headwise=False,
                             extra_W=a.get("extra_W", 0) + 2 * gr_W + h * (ih + ikv) * idim,
                             extra_b=a.get("extra_b", 0) + 2 * idim)
        elif g["rec"] is not None:
            g["rec"] = dict(g["rec"], W=g["rec"]["W"] + 2 * gr_W)
    p["final_norm"] = 0
    p["extra"] += H                                          # hyper_connection_mixer.hc_norm
    p["extra_W"] += 2 * H * r                                # its down / up
    ids = sorted(set(int(x) for x in (d.get("ple_layer_ids") or []) if isinstance(x, int) and 1 <= x <= n))
    if ids:
        E = int(d.get("ple_embed_dim") or h)
        K = int(d.get("ple_conv_kernel_size") or 4)
        ng = int(d.get("ngram_size") or 3)
        heads = (ng - 1) * int(d.get("heads_per_ngram") or 8)
        base = int(d.get("ngram_vocab_size_base") or 20000000)
        div = int(d.get("make_ngram_vocab_size_divisible_by") or 128)
        L = len(ids)
        p["extra_W"] += L * (E * H + E * h)                  # key_proj, value_proj
        p["extra"] += L * (3 * H + H * K)                    # norm_key / norm_query / norm_conv, conv1d
        if d.get("num_embeddings_per_layer_input") is not None:
            rows = L * base * heads                          # the older model-level table
        else:
            rows = sum(_ngram_rows(base, heads, k, div) for k in range(L))
        p["embedding"] += rows * (E // heads)
        p["stored_blocks"] = dict(p.get("stored_blocks") or {}, **{"n-gram index buffers (int64)": L * (ng + 2 * heads)})
    p["notes"].append(f"Qwen4-Exp: {hc} hyper-connection streams (low rank {r}), QSA indexer and PLE n-gram table "
                      f"counted; attention over the indexer's selected tokens not modelled")


# ------------------------------------------------------------------------------------------------ Step3

def is_step3(p, d, raw):
    return p["mt"] in ("step3_text", "step3_vl", "step3v") and bool(d.get("share_q_dim"))


def _step3_vision(raw, h):
    """Step3-VL vision tower and projector (stepfun-ai/step3 vision_encoder.py lines 51-75 StepCLIPVisionEmbeddings:
    class_embedding [hv], patch_embedding Conv2d(3 -> hv, P x P, bias), position_embedding (image / P)^2 + 1 rows;
    lines 98-140 per layer: qkv_proj (hv -> 3 hv, bias), out_proj (bias), fc1 / fc2 (bias), layer_norm1 / 2
    (LayerNorm, w + b); modeling_step3.py lines 686-703: vit_downsampler Conv2d(hv -> out, 2 x 2, bias),
    vit_downsampler2 Conv2d(out -> 2 out, 3 x 3, bias), vit_large_projector (2 out -> h, bias = projector_bias))."""
    v = raw.get("vision_config")
    if not isinstance(v, dict):
        return 0
    hv, I, Lv = int(v["hidden_size"]), int(v["intermediate_size"]), int(v["num_hidden_layers"])
    P, img, C = int(v.get("patch_size") or 14), int(v.get("image_size") or 728), int(v.get("num_channels") or 3)
    out = int(v.get("output_hidden_size") or hv)
    emb = hv + C * P * P * hv + hv + ((img // P) ** 2 + 1) * hv
    lay = 4 * hv + 3 * hv * hv + 3 * hv + hv * hv + hv + hv * I + I + I * hv + hv
    proj = hv * out * 4 + out + out * 2 * out * 9 + 2 * out + 2 * out * h + (h if raw.get("projector_bias") else 0)
    return emb + Lv * lay + proj


def _step3(p, d, raw, cl):
    """Step3 text model (stepfun-ai/step3 modeling_step3.py, read as text, not run): Step3vAttention lines 397-419 --
    one K / V head whatever the config (line 405: num_key_value_heads = 1), q_proj h -> share_q_dim, inter_norm
    (RMSNorm share_q_dim), wq share_q_dim -> heads x hd, k_proj / v_proj h -> hd, o_proj heads x hd -> h; the MoE
    layers (moe_layers_enum, lines 479-491) hold Step3vMoEMLP (lines 337-348: gate h -> E, up / gate / down
    MoELinear [E, I, h]) beside a share_expert MLP (share_expert_dim); the rest a dense MLP (intermediate_size);
    input / post-attention RMSNorm (lines 494-495). The vision tower is a named block (_step3_vision)."""
    h = p["hidden"]
    nh, hd, sq = int(d["num_attention_heads"]), int(d.get("head_dim") or h // int(d["num_attention_heads"])), \
        int(d["share_q_dim"])
    W = h * sq + sq * nh * hd + 2 * h * hd + nh * hd * h
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], nkv=1, W_override=W, b_override=sq)
    vt = _step3_vision(raw, h)
    if vt:
        p["stored_blocks"] = dict(p.get("stored_blocks") or {},
                                  **{"vision tower (Step3 StepCLIP + projector, from config)": vt})


# ------------------------------------------------------------------------------------------------ SDAR

def is_sdar_selfcond(p, d, raw):
    return p["mt"] == "sdar" and any(str(k).startswith("self_conditioning") for k in d)


def _sdar_selfcond(p, d, raw, cl):
    """SDAR self-conditioning (yifanyu/QwenDLM-4B-ORL modeling_sdar.py, read as text, not run: lines 601-614
    SDARSelfConditioning -- pre_norm RMSNorm(h), gate_proj / up_proj h -> I, down_proj I -> h, I =
    self_conditioning_intermediate_size else intermediate_size; line 1403 SDARForCausalLM.self_conditioning, applied
    to the input embeddings on self-conditioned steps, lines 1471-1475). The head is tied (line 1393
    _tied_weights_keys, tie_word_embeddings true); the checkpoint stores the copy."""
    h = p["hidden"]
    I = int(d.get("self_conditioning_intermediate_size") or d["intermediate_size"])
    p["extra_W"] += 3 * h * I
    p["extra"] += h


def _blank_g(kind, n, h=0):
    return {"n": n, "np": n, "kind": kind, "window": None, "attn": None, "mla": None, "rec": None, "ffn": None,
            "moe": None, "norm": 0, "k_eq_v": False}


# --------------------------------------------------------------------------------------------- SambaY / Phi-4-flash

def is_sambay(p, d, raw):
    return p["mt"] == "phi4flash" or p["arch"] == "Phi4FlashForCausalLM"


def _sambay(p, d, raw, cl):
    """SambaY (microsoft/Phi-4-mini-flash-reasoning, remote modeling_phi4flash.py and configuration_phi4flash.py,
    read as text). n layers (n % 4 == 0), hidden h, heads nh, head_dim h / nh (modeling 473), KV heads nkv:
      layer i < n/2, even   Phi3Mamba (modeling 784-849): in_proj h -> 2 di (di = mamba_expand x h), conv1d
                            depthwise kernel K with bias, x_proj di -> R + 2N, dt_proj R -> di (+ bias), A_log
                            [di, N], D [di], out_proj di -> h (no bias: mamba_proj_bias False)
      layer i < n/2, odd    SambaYAttention (modeling 457-493): Wqkv h -> (nh + 2 nkv) hd with bias, out_proj
                            nh hd -> h with bias, FlashDiffCustomAttention lambda_q1 / k1 / q2 / k2 [hd] and
                            subln RMSNorm(2 hd) (modeling 1980-2005); sliding window on these layers
                            (configuration 144-147: sliding_window for i < n/2 and i odd)
      layer n/2             the same full Mamba (yoco_mb: it emits the SSM output the GMUs reuse, modeling 1024-1036)
      layer n/2 + 1         full attention (sliding_window None from n/2 + 1, modeling 1028-1030), the one global
                            KV cache (SambaYCache global_attn_idx = n/2 + 1, modeling 239)
      layer >= n/2 + 2      yoco_cross: even -> GMU (Phi3Mamba yoco_cross: in_proj h -> di, out_proj di -> h, no
                            state, modeling 813-815); odd -> cross attention with Wqkv h -> nh hd (+ bias) and the
                            out_proj, reusing layer n/2 + 1's K / V (modeling 489-490, 554-557): no KV of its own
      every layer           SambaYMLP fc1 h -> 2 I, fc2 I -> h, no bias (modeling 438-439); input_layernorm and
                            post_attention_layernorm nn.LayerNorm (PHI_NORM_CLASS, modeling 409, 1017, 1042)
      model                 embed_tokens, final_layernorm (LayerNorm), lm_head tied (tie_word_embeddings)
    The Mamba cache holds conv [di, K] and ssm [di, N] per Mamba layer (SambaYCache, modeling 223-225, 250-262)."""
    h, n = p["hidden"], int(d["num_hidden_layers"])
    nh = int(d["num_attention_heads"])
    hd = h // nh
    nkv = int(d.get("num_key_value_heads") or nh)
    di = int(d.get("mamba_expand") or 2) * h
    N, K = int(d.get("mamba_d_state") or 16), int(d.get("mamba_d_conv") or 4)
    R = d.get("mamba_dt_rank")
    R = math.ceil(h / 16) if R in (None, "auto") else int(R)
    W = d.get("sliding_window")
    W = int(W) if isinstance(W, int) and not isinstance(W, bool) else 2047
    I = int(d["intermediate_size"])
    ln = 2 * h
    mamba = {"family": "mamba1", "W": h * 2 * di + di * (R + 2 * N) + R * di + di * h,
             "b": di * K + di + di + di * N + di, "conv_state": di * K, "ssm_state": di * N,
             "core_flops": 2 * di * K + 2 * di * N, "conv_ch": di, "conv_k": K}
    gmu = {"family": "gmu", "W": h * di + di * h, "b": 0, "conv_state": 0, "ssm_state": 0, "core_flops": 0}
    att = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": True, "o_bias": True, "qk_norm": 0,
           "out_gate": False, "gate_headwise": False, "extra_b": 4 * hd + 2 * hd}
    ffn = {"glu": True, "inter": I, "bias": False}
    G = {}
    for i in range(n):
        if i < n // 2 + 2:
            key = ("mamba",) if i % 2 == 0 else (("sliding",) if i < n // 2 else ("full",))
        else:
            key = ("gmu",) if i % 2 == 0 else ("cross",)
        G[key] = G.get(key, 0) + 1
    groups = []
    for key, cnt in G.items():
        g = _blank_g("rec", cnt)
        g["ffn"], g["norm"] = dict(ffn), 2 * ln
        if key[0] == "mamba":
            g["rec"] = dict(mamba)
        elif key[0] == "gmu":
            g["rec"] = dict(gmu)
        elif key[0] == "cross":
            g.update(kind="shared", attn=dict(att, no_kv_proj=True))
        else:
            g.update(kind=key[0], attn=dict(att), window=W if key[0] == "sliding" else None)
        groups.append(g)
    p["groups"] = groups
    p["layers"] = n
    p["final_norm"] = ln
    p.pop("unknown", None)
    p["notes"].append(f"SambaY: layers {n // 2 + 2}+ reuse layer {n // 2 + 1}'s KV (cross attention) and layer "
                      f"{n // 2}'s SSM output (GMU)")


# --------------------------------------------------------------------------------------------- Hymba

def is_hymba(p, d, raw):
    return p["mt"] == "hymba" or p["arch"] == "HymbaForCausalLM"


def _hymba(p, d, raw, cl):
    """Hymba (nvidia/Hymba-1.5B-*, remote modeling_hymba.py, read as text). Every layer is a HymbaDecoderLayer
    (modeling 1858-1876): a HymbaBlock with attention heads and a Mamba SSM in parallel, then a one-expert MoE
    (HymbaSparseMoeBlock with num_experts = 1: no router, one HymbaMLP gate / up h -> I, down I -> h; modeling
    1748-1750, 1786-1790), input_layernorm and pre_moe_layernorm (RMSNorm h).
      HymbaBlock (modeling 1479-1551): di = mamba_expand x h; k_hidden = nkv / nh x h, v_hidden = k_hidden x
      mamba_expand; in_proj h -> di + h (q) + k_hidden + v_hidden + di (the KV-reuse consumers drop k / v,
      modeling 1512-1515, 1520); out_proj di -> h, shared by the attention (v_head_dim = di / nh, modeling 1495) and
      the SSM outputs, each RMS-normed first (pre_avg_layernorm1 / 2 [di]); conv1d depthwise kernel K with bias;
      x_proj di -> R + 2N, dt_proj R -> di (+ bias), A_log [di, N], D [di], dt / B / C RMSNorms (R, N, N) under
      mamba_inner_layernorms. q / k head_dim h / nh (modeling 496), the attention class itself has no projections
      (attn_only_wo_proj, modeling 1500; 517-530).
      Layers: global_attn_idx full attention, the rest sliding_window (modeling 2126-2132); kv_reuse_group
      consumers (every member after the first) reuse the producer's K / V (modeling 2103-2113): no KV of their own.
      Model: embed_tokens, memory_tokens [num_memory_tokens, h] prepended to the prompt (modeling 2145-2146),
      final_layernorm, lm_head tied.
    The cache holds conv [di, K] and ssm [di, N] per layer (HybridMambaAttentionDynamicCache, modeling 384-407)."""
    h, n = p["hidden"], int(d["num_hidden_layers"])
    nh = int(d["num_attention_heads"])
    nkv = int(d.get("num_key_value_heads") or nh)
    hd = h // nh
    ex = int(d.get("mamba_expand") or 2)
    di = ex * h
    N, K = int(d.get("mamba_d_state") or 16), int(d.get("mamba_d_conv") or 4)
    R = d.get("mamba_dt_rank")
    R = math.ceil(h / 16) if R in (None, "auto") else int(R)
    I = int(d["intermediate_size"])
    W = d.get("sliding_window")
    W = int(W) if W else None
    glob = set(int(x) for x in (d.get("global_attn_idx") or []))
    cons = set()
    for grp in d.get("kv_reuse_group") or []:
        for x in list(grp)[1:]:
            cons.add(int(x))
    prod_win = {}
    for grp in d.get("kv_reuse_group") or []:
        for x in list(grp)[1:]:
            prod_win[int(x)] = int(list(grp)[0]) not in glob
    rec = {"family": "mamba1", "W": h * 2 * di + di * (R + 2 * N) + R * di,
           "b": di * K + (di if d.get("mamba_conv_bias", True) else 0) + di + di * N + di + 2 * di
                + ((R + 2 * N) if d.get("mamba_inner_layernorms", True) else 0),
           "conv_state": di * K, "ssm_state": di * N, "core_flops": 2 * di * K + 2 * di * N, "conv_ch": di,
           "conv_k": K}
    att = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": di // nh, "qkv_bias": False, "o_bias": False, "qk_norm": 0,
           "out_gate": False, "gate_headwise": False}
    G = {}
    for i in range(n):
        if i in cons:
            key = ("shared", bool(W) and prod_win.get(i, True))
        else:
            key = ("full" if (i in glob or not W) else "sliding", False)
        G.setdefault(key, []).append(i)
    groups = []
    for (kind, win), at in G.items():
        g = _blank_g(kind, len(at))
        g["at"] = at
        g["attn"] = dict(att, no_kv_proj=True) if kind == "shared" else dict(att)
        g["window"] = W if (kind == "sliding" or win) else None
        g["rec"] = dict(rec)
        g["ffn"] = {"glu": True, "inter": I, "bias": False}
        g["norm"] = 2 * h
        groups.append(g)
    p["groups"] = groups
    p["layers"] = n
    p["order_exact"] = True                                 # (the groups' layer indices, in order)
    p["final_norm"] = h
    M = int(d.get("num_memory_tokens") or 0)
    p["extra"] += M * h
    p.pop("unknown", None)
    if M:
        p["notes"].append(f"Hymba: {M} learned memory tokens are prepended to every prompt (not in the KV counts)")


# --------------------------------------------------------------------------------------------- Sundial

def is_sundial(p, d, raw):
    return p["mt"] == "sundial" or p["arch"] == "SundialForPrediction"


def _sundial(p, d, raw, cl):
    """Sundial (thuml/sundial-base-128m, a time-series model; remote modeling_sundial.py and flow_loss.py, read as
    text). No token vocabulary:
      SundialPatchEmbedding (modeling 28-39): hidden_layer 2P -> I, output_layer I -> h, residual_layer 2P -> h,
        all with bias (P = input_token_len; values and mask concatenated)
      SundialAttention (modeling 101-113): q / k / v h -> h with bias, o_proj without; head_dim h / heads
      SundialMLP (modeling 163-173): gate / up h -> I, down I -> h, no bias
      SundialDecoderLayer (modeling 180-191): norm1, norm2 nn.LayerNorm (weight + bias); SundialModel.norm
        LayerNorm (modeling 262)
      FlowLoss head (modeling 403-404 -> flow_loss.py SimpleMLPAdaLN 166-190): target C = output_token_lens[-1],
        width w = h, z = h, depth D = flow_loss_depth; time_embed MLP 256 -> w -> w (bias, flow_loss.py 63-68),
        cond_embed z -> w, input_proj C -> w; D ResBlocks (in_ln LayerNorm w, mlp w -> w -> w, adaLN w -> 3w, all
        with bias; flow_loss.py 107-123); FinalLayer (norm_final non-affine, linear w -> C, adaLN w -> 2w, bias;
        flow_loss.py 139-147)
    The patch embedding is counted as applied linear weights (extra_W) and the flow head as the 'LM head' (one
    sampling step per position: num_sampling_steps steps per generated patch in generation)."""
    h = p["hidden"]
    nh = int(d["num_attention_heads"])
    I = int(d["intermediate_size"])
    P = int(d.get("input_token_len") or 16)
    ol = d.get("output_token_lens") or [720]
    C = int(ol[-1])
    D = int(d.get("flow_loss_depth") or 3)
    w = h
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], nh=nh, nkv=nh, dqk=h // nh, dv=h // nh, qkv_bias=True, o_bias=False,
                             qk_norm=0)
        g["ffn"] = {"glu": True, "inter": I, "bias": False}
        g["norm"] = 2 * 2 * h
    p["final_norm"] = 2 * h
    p["vocab"], p["embedding"], p["tied"] = 0, 0, False
    p["extra_W"] += 2 * P * I + I * h + 2 * P * h
    p["extra"] += I + h + h
    head_W = 256 * w + w * w + h * w + C * w + D * (2 * w * w + w * 3 * w) + w * C + w * 2 * w
    head_b = 2 * w + w + w + D * (2 * w + 2 * w + 3 * w) + C + 2 * w
    p["lm_head_W"] = head_W
    p["lm_head"] = head_W + head_b
    p["notes"].append("Sundial (time series): patch embedding instead of a token table, the TimeFlow head as the "
                      "head (one sampling step)")


# --------------------------------------------------------------------------------------------- Ethoric

def is_ethoric(p, d, raw):
    return p["mt"] == "ethoric" or p["arch"] == "EthoricForCausalLM"


def _ethoric(p, d, raw, cl):
    """Ethoric (techdotus/Ethoric-10M, remote attention.py and modeling_ethoric.py, read as text):
      EthoricAttention (attention.py 144-185): q / k / v / o without bias; q_norm / k_norm RMSNorm(head_dim) when
        qk_norm == "per_head" (163-164); gate_proj h -> heads (+ bias) when attn_gate == "headwise", h -> heads x
        head_dim (+ bias) when "elementwise" (168-171), a sigmoid output gate; the layers in attn_window_layers
        attend the last attn_window positions (175-185)
      EthoricBlock (modeling_ethoric.py 116-132): Peri-LN, four RMSNorms (attn_in / attn_out / mlp_in / mlp_out);
        SwiGLU h -> ffn_hidden (100-102); Canon depthwise causal convs [h, 1, k], no bias (38-53), at the positions
        in canon.positions ("A" before attention, "C" before the MLP), each carrying k - 1 inputs in the cache
        (63-71)
      model: embed_tokens, final_norm (209), lm_head tied"""
    h = p["hidden"]
    n = int(d.get("n_layer") or d["num_hidden_layers"])
    hd = int(d["head_dim"])
    gate = str(d.get("attn_gate") or "")
    can = d.get("canon") if isinstance(d.get("canon"), dict) else {}
    npos = len([x for x in (can.get("positions") or []) if x in ("A", "C")]) if can.get("enabled") else 0
    k = int(can.get("kernel") or 4)
    win = int(d.get("attn_window") or 0)
    wl = set(int(x) for x in (d.get("attn_window_layers") or []))
    I = int(d.get("ffn_hidden") or d["intermediate_size"])

    def per(i, g):
        a = dict(g["attn"], qk_norm=2 * hd if str(d.get("qk_norm")) == "per_head" else 0)
        if gate == "headwise":
            a.update(out_gate=True, gate_headwise=True, extra_b=a.get("extra_b", 0) + a["nh"])
        elif gate == "elementwise":
            a.update(out_gate=True, gate_headwise=False, extra_b=a.get("extra_b", 0) + a["nh"] * a["dqk"])
        g = dict(g, attn=a, norm=4 * h, ffn=dict(g["ffn"], inter=I))
        if npos:
            g["rec"] = {"family": "glue", "W": 0, "b": npos * h * k, "conv_state": npos * h * (k - 1),
                        "ssm_state": 0, "core_flops": npos * 2 * h * k}
        if win and i in wl:
            g.update(kind="sliding", window=win)
        return g
    tmpl = {kk: v for kk, v in p["groups"][0].items() if kk not in ("n", "np")}
    p["groups"] = _per_layer_groups([per(i, dict(tmpl)) for i in range(n)])


def is_arloop(p, d, raw):
    return p["mt"] in ("arloop", "arloop_baseline") and isinstance(d.get("physical_layers"), int) \
        and len(p["groups"]) == 1 and p["groups"][0]["attn"] is not None


def _arloop(p, d, raw, cl):
    """osunlp ARLoop (model_type arloop, modeling_arloop.py, Hub snapshots 5cd28f3 / a29f310, identical files) and its
    fixed-loop PreNorm baseline (arloop_baseline, PreNormLoop modeling_baseline.py, snapshot e4ad59d), read as text:
      shared_value_embedding  nn.Embedding(V, nkv x head_dim) (modeling_arloop.py lines 110-111; baseline 110-111),
                              looked up once per token (line 188 / 164): an embedding table
      ve_gate                 Linear(32 -> nkv, no bias) on the physical layers i with i % 2 == (physical_layers - 1) % 2
                              (lines 37-38 in both files), applied on every loop of those layers (line 47)
      attnres_queries         [2 x num_hidden_layers, h] (ARLoop only, line 112): one routing query per unrolled
                              attention / MLP input (lines 194-202)
      attnres_norms           2 x num_hidden_layers RMSNorm(h) (lines 113-114), final_attnres_query [h] (line 115),
                              final_attnres_norm RMSNorm(h) (line 116); used by _route (lines 131-148), a softmax
                              over at most sw_blocks + 1 block states (no linear weights)
    The q / k RMSNorm is weightless (F.rms_norm, lines 64-66), and each block holds two RMSNorms (lines 90-92), as the
    generic plan already counts."""
    h, V = p["hidden"], p["vocab"]
    P = int(d["physical_layers"])
    t = p["groups"][0]
    nkv, hd = t["attn"]["nkv"], t["attn"]["dqk"]
    loop = max(1, t["n"] // max(1, t["np"]))
    G = sum(1 for i in range(P) if i % 2 == (P - 1) % 2)
    # unrolled layer loop x P + i runs physical block i (modeling_arloop.py lines 191-193)
    N_un = t["n"]
    gat = [j for j in range(N_un) if (j % P) % 2 == (P - 1) % 2]
    pla = [j for j in range(N_un) if (j % P) % 2 != (P - 1) % 2]
    gated = dict(t, n=G * loop, np=G, attn=dict(t["attn"], extra_W=t["attn"].get("extra_W", 0) + 32 * nkv), at=gat)
    plain = dict(t, n=(P - G) * loop, np=P - G, at=pla)
    p["groups"] = [g for g in (gated, plain) if g["np"]]
    p["embedding"] += V * nkv * hd
    if p["mt"] == "arloop":
        N = int(d.get("num_hidden_layers") or P * loop)
        p["extra"] += 2 * N * h + 2 * N * h + 2 * h
    p["notes"].append(f"value embedding (V x {nkv * hd}) gated into V on {G} of {P} physical layers")


def is_gptx2(p, d, raw):
    return p["mt"] == "gptx2" and bool(d.get("use_symplectic_rope"))


def _gptx2(p, d, raw, cl):
    """GPT-X2 Hamiltonian clocks (reaperdoesntknow modeling_gptx2.py, Hub snapshot 8c60ea3, read as text).
    SymplecticClock(k, h) (lines 754-787): raw_h_diag [2k] (line 774), context_to_force Linear(h -> 2k, bias) and
    Linear(2k -> 2k, bias) (lines 776-780), delta_to_p Linear(h -> k, no bias) (line 781); a metric projection
    Linear(2k -> 2, bias) beside each clock. One global clock (symplectic_global_k_dim, lines 1213-1222) and one local
    clock per block (symplectic_local_k_dim, lines 1068-1077). The clocks read the masked sequence MEAN of the hidden
    state (lines 1114-1116, 1445-1446) once per forward pass, not every token, so their weights are counted as other
    parameters, not per-token matmul weights; the clock state (2k per clock) is a few hundred floats per sequence."""
    h = p["hidden"]
    n = p["layers"]

    def clock(k):
        return 2 * k + (h * 2 * k + 2 * k) + (2 * k * 2 * k + 2 * k) + h * k + (2 * k * 2 + 2)
    kg = int(d.get("symplectic_global_k_dim") or 32)
    kl = int(d.get("symplectic_local_k_dim") or 16)
    p["extra"] += clock(kg) + n * clock(kl)
    p["notes"].append(f"Hamiltonian clocks: one global (k {kg}) and one per layer (k {kl}), on the sequence mean")


# config-reading gaps of graph/model_graph.py corrected here (round 3 fixes the classifier: dist/r3_from_zoo.md);
# each SETS what the code builds, so the result is unchanged once model_graph reads it right

def _set_tied(p, tied):
    p["tied"] = bool(tied)
    p["lm_head"] = 0 if p["tied"] else p["lm_head_W"]


# ------------------------------------------------------------------------------------------------ Inferno

def is_inferno(p, d, raw):
    return p["mt"] == "inferno" and "InfernoForCausalLM" in (raw.get("architectures") or [])


def _inferno(p, d, raw, cl):
    """Inferno (AlgoDriveAI/Inferno-Exp-1.0 modeling_inferno.py, Hub e13621c): the head is tied whatever the config
    says -- line 111 `self.lm_head.weight = self.embed.weight` (and _tied_weights_keys, line 102); the checkpoint holds
    embed.weight only. Attention qkv_proj h -> (heads + 2 kv) hd and o_proj (lines 56-57), the packed SwiGLU project
    h -> 2 I and down (lines 77-78), attn_norm / ffn_norm / ln_f RMSNorm (lines 91-92, 109): the generic plan."""
    _set_tied(p, True)


# ------------------------------------------------------------------------------------------------ BananaMind-2-MoE

def is_bananamind2_moe(p, d, raw):
    return p["mt"] == "bananamind2_moe" and bool(d.get("expert_intermediate_size"))


def _bananamind2_moe(p, d, raw, cl):
    """BananaMind-2-MoE (BananaMind/BananaMind-2-MoE modeling_bananamind2moe.py, Hub ca16baf): q / k / v / o
    (lines 57-60) with q_norm / k_norm RMSNorm(head_dim) always (lines 61-62); a top-1 MoE (lines 164-174:
    router h -> num_experts, no bias; num_experts BananaMind2MoEExpert) whose experts are SwiGLU of width
    expert_intermediate_size (lines 141-158), no shared expert; ln_1 / ln_2 / ln_f RMSNorm (lines 212-214, 266); the
    head is tied under tie_word_embeddings (lines 269-271) and the checkpoint stores a copy."""
    hd = int(d.get("head_dim") or p["hidden"] // int(d["num_attention_heads"]))
    E, k, ew = int(d["num_experts"]), int(d.get("num_experts_per_token") or 1), int(d["expert_intermediate_size"])
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], qk_norm=2 * hd)
        if g["moe"] is not None:
            g["moe"] = dict(g["moe"], E=E, k=k, ew=ew, shared=0, shared_gate=False)
    _set_tied(p, d.get("tie_word_embeddings", True))


# ------------------------------------------------------------------------------------------------ PIT

def is_pit(p, d, raw):
    return p["mt"] == "pit" and "PITForCausalLM" in (raw.get("architectures") or [])


def _pit(p, d, raw, cl):
    """PIT (Diamegs/PIT-1B-201912 modeling_pit.py, Hub 043069f), nanochat-style: c_q / c_k / c_v / c_proj, h x h,
    no bias (lines 94-97); q and k normed by F.rms_norm with no weight (lines 114-115); the MLP is relu^2 with two
    matrices, c_fc h -> 4 h and c_proj back (lines 147-152); the block norms and the final norm are F.rms_norm with no
    weight (lines 173, 177, 371); the head is tied (lines 213, 227) and the checkpoint stores both copies."""
    h = p["hidden"]
    p["final_norm"] = 0
    for g in p["groups"]:
        g["norm"] = 0
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], qk_norm=0)
        if g["ffn"] is not None:
            g["ffn"] = dict(g["ffn"], glu=False, inter=4 * h, bias=False)
    _set_tied(p, True)


# ------------------------------------------------------------------------------------------------ Ivme

def is_ivme(p, d, raw):
    return p["mt"] == "ivme" and d.get("ffn_mult") is not None


def _ivme(p, d, raw, cl):
    """Ivme (IvmeLabs/Ivme-Conversate-v2-Base modeling_ivme.py, Hub df4a019): the SwiGLU width is
    int(hidden_dim x ffn_mult x 2 / 3) (line 93; gate / up / down lines 95-97), q / k / v / out_proj hidden x hidden
    (lines 61-64), attn_norm / ffn_norm / final_norm RMSNorm (lines 106, 108, 141), the head tied under
    tie_word_embeddings (lines 176-184)."""
    hd_ = int(d.get("hidden_dim") or p["hidden"])
    inter = int(hd_ * float(d["ffn_mult"]) * 2 / 3)
    for g in p["groups"]:
        if g["ffn"] is not None:
            g["ffn"] = dict(g["ffn"], glu=True, inter=inter)
    _set_tied(p, d.get("tie_word_embeddings", True))


# ------------------------------------------------------------------------------------------------ Kohaku (TIPO)

def is_kohaku_moe(p, d, raw):
    return p["mt"] == "kohaku" and isinstance(d.get("first_k_dense"), int) and bool(d.get("n_routed_experts")) \
        and p["groups"] and p["groups"][0]["attn"] is not None


def _kohaku_moe(p, d, raw, cl):
    """Kohaku (KBlueLeaf/TIPO-v2.1-1B-A200M modeling_kohaku.py, Hub f5a3185): layers i < first_k_dense hold a dense
    SwiGLU KohakuMLP(h, intermediate_size), the rest a KohakuMoE (lines 180-186); KohakuMoE (lines 140-145): gate
    h -> n_routed_experts (no bias), the persistent expert_bias buffer [E] (line 141, the 'router bias buffers'
    block), packed gate / up / down experts [E, h, moe_intermediate_size], one shared SwiGLU of
    moe_intermediate_size x n_shared_experts; top num_experts_per_tok (line 159); q_norm / k_norm under qk_norm
    (lines 80-82)."""
    n, h = p["layers"], p["hidden"]
    kd = int(d["first_k_dense"])
    E, k, ew = int(d["n_routed_experts"]), int(d.get("num_experts_per_tok") or 1), int(d["moe_intermediate_size"])
    tmpl = next(g for g in p["groups"] if g["attn"] is not None)
    base = {kk: v for kk, v in tmpl.items() if kk not in ("n", "np")}
    moe = {"E": E, "k": k, "ew": ew, "shared": ew * int(d.get("n_shared_experts") or 0), "shared_gate": False,
           "router_bias": False, "expert_bias": False, "glu": True, "out_bias": False, "dense_compute": False}
    layers = []
    for i in range(n):
        g = dict(base)
        if i < kd:
            g.update(moe=None, ffn={"glu": True, "inter": int(d["intermediate_size"]), "bias": False})
        else:
            g.update(moe=dict(moe), ffn=None)
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)


# ------------------------------------------------------------------------------------------------ MemT (Qwen3.5-MoE)

def is_memt_qwen35_moe(p, d, raw):
    return p["mt"] == "memt_qwen3_5_moe" and isinstance(raw.get("memt"), dict)


def _memt_qwen35_moe(p, d, raw, cl):
    """MemT on a Qwen3.5-MoE backbone (MemTensor/MemT-1.0-Nano modeling_memt.py, Hub 30fe0b2): every
    full-attention layer is a MemTAttention = Qwen3_5MoeAttention (q_proj with the output gate, k / v / o,
    q_norm / k_norm of head_dim, lines 212-238) plus a MemTMemory (lines 55-69): in_proj h -> 5 x heads x head_dim
    + 2 x head_dim, f_down / g_down head_dim -> heads x head_dim (g_down bias under output_gate_bias), decay_bias
    (heads x head_dim), decay_scale_log (heads), o_norm (head_dim); every layer's mlp is a Qwen3_5MoeSparseMoeBlock
    (line 278), whose shared expert has its sigmoid shared_expert_gate (h -> 1)."""
    h = p["hidden"]
    t = (raw.get("backbone_configs") or {}).get("text_config") or d
    nh, hd = int(t["num_attention_heads"]), int(t["head_dim"])
    V = nh * hd
    gb = bool((raw.get("memt") or {}).get("output_gate_bias"))
    memW = h * (5 * V + 2 * hd) + 2 * hd * V
    memb = (V if gb else 0) + V + nh + hd
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], qk_norm=2 * hd, extra_W=memW, extra_b=memb)
        if g["moe"] is not None and g["moe"]["shared"]:
            g["moe"] = dict(g["moe"], shared_gate=True)


# ------------------------------------------------------------------------------------------------ Meditor-2 HMoE

def is_meditor2(p, d, raw):
    return p["mt"] == "meditor2" and bool(d.get("num_macro_domains")) and bool(d.get("num_sub_experts_per_domain"))


def _meditor2(p, d, raw, cl):
    """Meditor-2 hierarchical MoE (Yo-Purple/Meditor-2-3.4B-A1.8B-HMoE modeling_meditor2.py, Hub db3ea4c):
      attention  q / k / v / o, no bias (lines 138-141); global when (i + 1) % global_attn_every_n_layers == 0,
                 else sliding with window sliding_window_size (lines 130-131, 172-184); the rotary inv_freq is a
                 persistent buffer per layer (line 81)
      HMoE       (lines 194-230) a shared SwiGLU of shared_intermediate_size (lines 213-215), macro_router h ->
                 num_macro_domains and one micro router h -> num_sub_experts_per_domain per domain (lines 218-223),
                 num_macro_domains x num_sub_experts_per_domain packed SwiGLU experts of sub_intermediate_size
                 (lines 226-228) and routed_scale (line 230); a token takes the argmax domain and the argmax sub-expert
                 in it (lines 254-272): one routed expert active
      head       tied under tie_word_embeddings (lines 346, 360-361); no lm_head stored"""
    D, S = int(d["num_macro_domains"]), int(d["num_sub_experts_per_domain"])
    every, W = int(d.get("global_attn_every_n_layers") or 0), int(d.get("sliding_window_size") or 0)
    moe = {"E": D * S, "k": 1, "ew": int(d["sub_intermediate_size"]), "shared": int(d["shared_intermediate_size"]),
           "shared_gate": False, "router_bias": False, "expert_bias": False, "glu": True, "out_bias": False,
           "dense_compute": False, "zero": D}                   # zero: the macro router's D rows beside the D x S
    tmpl = next(g for g in p["groups"] if g["attn"] is not None)
    base = {kk: v for kk, v in tmpl.items() if kk not in ("n", "np")}
    layers = []
    for i in range(p["layers"]):
        g = dict(base, ffn=None, moe=dict(moe), norm=2 * p["hidden"] + 1)   # 2 RMSNorms, routed_scale
        glob = not (every and W) or (i + 1) % every == 0
        g["kind"], g["window"] = ("full", None) if glob else ("sliding", W)
        layers.append(g)
    p["groups"] = _per_layer_groups(layers)
    _set_tied(p, d.get("tie_word_embeddings", True))


# ------------------------------------------------------------------------------------------------ Nexus (flame)

def is_nexus(p, d, raw):
    return p["mt"] == "nexus" and "NexusForCausalLM" in (raw.get("architectures") or [])


def _nexus(p, d, raw, cl):
    """Nexus (ifx-pse-sys-ml/flame-27m-grounded modeling_nexus.py, Hub b31d984): q_norm / k_norm RMSNorm(head_dim)
    built always (lines 106-107); the head is tied under tie_word_embeddings (lines 294-295), whose default is True
    (configuration_nexus.py line 36; the config file has no key); a dense SwiGLU of intermediate_size unless use_moe
    (lines 155-161, 211)."""
    hd = int(d.get("head_dim") or p["hidden"] // int(d["num_attention_heads"]))
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], qk_norm=2 * hd)
    _set_tied(p, raw.get("tie_word_embeddings", True))


def _set_qk_norm(p, width):
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], qk_norm=width)


# ------------------------------------------------------------------------------------------------ Kanana-2 tiny

def is_kanana2_tiny(p, d, raw):
    return p["mt"] == "kanana2_tiny"


def _kanana2_tiny(p, d, raw, cl):
    """Kanana-2 tiny (kakaocorp/kanana-2-1.3b-instruct, remote modeling_kanana2_tiny.py / configuration_kanana2_tiny.py,
    read as text): q_norm / k_norm RMSNorm(head_dim) on every attention (modeling lines 230-231, applied per head at
    247-248); the head tied to embed_tokens (configuration line 59 tie_word_embeddings=True, passed on at line 155;
    modeling line 553 _tied_weights_keys) -- the config carries no tie key."""
    if d.get("tie_word_embeddings") is not False:
        _set_tied(p, True)
    hd = next((g["attn"]["dqk"] for g in p["groups"] if g["attn"] is not None), 0)
    _set_qk_norm(p, 2 * hd)


# ------------------------------------------------------------------------------------------------ Phi-3-small

def is_phi3small(p, d, raw):
    return p["mt"] == "phi3small" and raw.get("tie_word_embeddings") is None


def _phi3small(p, d, raw, cl):
    """Phi-3-small (microsoft/Phi-3-small-*-instruct, remote modeling_phi3_small.py / configuration_phi3_small.py, read
    as text): Phi3SmallForCausalLM declares _tied_weights_keys = ["lm_head.weight"] (modeling line 900) and its config
    sets no tie_word_embeddings (configuration line 234 passes only bos / eos to PretrainedConfig, whose default is
    True), so lm_head (line 906) is the embedding; the checkpoint stores no lm_head. Every Linear has its default bias
    (Phi3SmallMLP up_proj h -> 2 I and down_proj, lines 163-164; query_key_value and dense, lines 208-209) and every
    norm is a LayerNorm with bias (input / post_attention, lines 655-656; final_layernorm, line 748), whatever the
    config says (the -8k config is read by the classifier without them)."""
    h = p["hidden"]
    _set_tied(p, True)
    p["final_norm"] = 2 * h
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], qkv_bias=True, o_bias=True)
        if g["ffn"] is not None:
            g["ffn"] = dict(g["ffn"], bias=True)
        g["norm"] = 2 * 2 * h


# ------------------------------------------------------------------------------------------------ MiniCPM (remote)

def is_minicpm_remote_tied(p, d, raw):
    return (p["arch"] == "MiniCPMForCausalLM" and raw.get("tie_word_embeddings") is None
            and "configuration_minicpm" in str((raw.get("auto_map") or {}).get("AutoConfig") or ""))


def _minicpm_remote_tied(p, d, raw, cl):
    """MiniCPM remote configs with no tie key (openbmb/MiniCPM4-0.5B; also MiniCPM-1B / 2B-sft, BitCPM-CANN-0.5B,
    none with a model_type), read as text: configuration_minicpm.py's default tie_word_embeddings=True (MiniCPM4-0.5B
    line 128, passed on at line 175; MiniCPM-1B / 2B line 134; BitCPM-CANN-0.5B line 128) with
    MiniCPMForCausalLM._tied_weights_keys = ['lm_head.weight'] (MiniCPM4-0.5B modeling_minicpm.py line 1154): the
    head is the embedding, and MiniCPM4-0.5B's checkpoint stores no lm_head."""
    _set_tied(p, True)


# ------------------------------------------------------------------------------------------------ Mima

def is_mima(p, d, raw):
    return p["mt"] == "mima"


def _mima(p, d, raw, cl):
    """Mima (root-sys-n/Mima-Pico-15m, remote modeling_mima.py, read as text): SwiGLU w1 / w3 h -> ffn_hidden, w2 back
    (lines 108-110; MimaHFConfig ffn_hidden, lines 237 / 244 / 270, default 1024); fused c_qkv h -> 3 h and c_proj
    (lines 86-87); ln_1 / ln_2 RMSNorm (lines 119 / 121); lm_head tied to wte (line 139). The checkpoint stores the
    tied lm_head copy too (the config's own parametre = 15,537,792, parametre_toplam = 18,683,520)."""
    _set_tied(p, True)
    I = int(d.get("ffn_hidden") or 1024)
    for g in p["groups"]:
        if g["ffn"] is not None:
            g["ffn"] = dict(g["ffn"], inter=I, glu=True)


# ------------------------------------------------------------------------------------------------ Apertus-1.5

def is_apertus1p5(p, d, raw):
    return p["mt"] in ("apertus1p5_text", "apertus1p5") and str(d.get("hidden_act")) == "xielu"


def _apertus1p5(p, d, raw, cl):
    """Apertus-1.5 text model (swiss-ai/Apertus-v1.5-8B-base, gated, no modelling code in the repo; the layer is the
    stock Apertus one, transformers 4.57 models/apertus/modeling_apertus.py, read as text): ApertusMLP is ungated,
    up_proj h -> I, xIELU, down_proj I -> h (lines 43-54), whose XIELUActivation holds two parameters alpha_p / alpha_n
    (activations.py lines 238-241; beta / eps are 0-d buffers); q_norm / k_norm RMSNorm(head_dim) (lines 211-212);
    attention_layernorm / feedforward_layernorm (lines 268-269). The head maps to output_vocab_size, not the input
    vocabulary (the Hub's BF16 count, 8,609,083,456, is exactly this LM; the F32 files are the audio and image
    tokenizers)."""
    h = p["hidden"]
    Vo = int(d.get("output_vocab_size") or p["vocab"])
    p["lm_head_W"] = Vo * h
    p["tied"] = bool(d.get("tie_word_embeddings", False))
    p["lm_head"] = 0 if p["tied"] else Vo * h
    hd = next((g["attn"]["dqk"] for g in p["groups"] if g["attn"] is not None), 0)
    if d.get("qk_norm", True):
        _set_qk_norm(p, 2 * hd)
    for g in p["groups"]:
        if g["ffn"] is not None:
            g["ffn"] = dict(g["ffn"], glu=False)
            g["norm"] = 2 * h + 2                           # two RMSNorms + xIELU alpha_p / alpha_n


# ------------------------------------------------------------------------------------------------ Gemma-3 MoE

def is_gemma3_moe(p, d, raw):
    return p["mt"] == "gemma3_moe"


def _gemma3_moe(p, d, raw, cl):
    """gemma3_moe (tardellirs/gemma3moe-16x4-270m-starter, remote modeling_gemma3_moe.py, read as text): Gemma3MoeConfig
    subclasses Gemma3TextConfig (line 28; tie_word_embeddings default True, transformers configuration_gemma3.py line
    186) and Gemma3MoeForCausalLM subclasses Gemma3ForCausalLM (line 89; _tied_weights_keys, transformers
    modeling_gemma3.py line 603); its decoder layer is Gemma3DecoderLayer (line 72) with the MLP replaced by a router
    + num_experts Gemma3MLP experts (lines 41-49), so it keeps q_norm / k_norm RMSNorm(head_dim) (modeling_gemma3.py
    lines 292-293) and the four layer norms input / post_attention / pre_feedforward / post_feedforward (lines
    353-356)."""
    h = p["hidden"]
    if d.get("tie_word_embeddings", True):
        _set_tied(p, True)
    hd = next((g["attn"]["dqk"] for g in p["groups"] if g["attn"] is not None), 0)
    _set_qk_norm(p, 2 * hd)
    for g in p["groups"]:
        g["norm"] = 4 * h


# ------------------------------------------------------------------------------------------------ Bitsy

def is_bitsy(p, d, raw):
    return p["arch"] == "BitsyForCausalLM"


def _bitsy(p, d, raw, cl):
    """Bitsy (zarko1321/Bitsy500M, remote modeling_bitsy.py / configuration_bitsy.py, read as text): q_norm / k_norm
    BitsyRMSNorm(head_dim) on every attention (modeling lines 47-48, applied per head at 53-54); lm_head tied by
    BitsyConfig's default tie_word_embeddings=True (configuration lines 13 / 16) with _tied_weights_keys (modeling
    line 102); the checkpoint stores no lm_head."""
    if raw.get("tie_word_embeddings") is not False:
        _set_tied(p, True)
    hd = next((g["attn"]["dqk"] for g in p["groups"] if g["attn"] is not None), 0)
    _set_qk_norm(p, 2 * hd)

# ------------------------------------------------------------------------------------------------ Crowd

def is_crowd(p, d, raw):
    return p["mt"] == "crowd" and "CrowdForCausalLM" in (raw.get("architectures") or [])


def _crowd(p, d, raw, cl):
    """Crowd (Vxtzq/Crowd-v1 modeling_crowd.py, read as text, not run): q / k / v / o without bias (lines 29-32),
    a SwiGLU w1 / w2 / w3 without bias (lines 64-66), LayerNorm ln1 / ln2 (lines 75, 77) and the final ln (line 125),
    the head tied to wte (lines 115, 154-156), and CrowdEngram (lines 86-97): a hashed bigram table of engram_buckets
    rows of width h, looked up per token and added to the token embedding (line 183)."""
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], qkv_bias=False, o_bias=False)
        if g["ffn"] is not None:
            g["ffn"] = dict(g["ffn"], bias=False)
    if not p.get("crowd_engram"):
        p["crowd_engram"] = True
        p["embedding"] += int(d.get("engram_buckets") or 0) * p["hidden"]


# ------------------------------------------------------------------------------------------------ Nae1

def is_nae1(p, d, raw):
    return p["mt"] == "nae1" and bool(d.get("n_kv_heads"))


def _nae1(p, d, raw, cl):
    """NeuralAI Nae1 (Subject-Emu-5259/NeuralAI-Nae1 model/nae1_model.py, read as text, not run): Nae1GQA (lines
    101-121) q_proj h -> n_head x head_dim, k_proj / v_proj h -> n_kv_heads x head_dim (the config key n_kv_heads,
    which the generic reader misses), o_proj, no biases; K / V repeated n_query_groups times (lines 145-146); a
    per-layer Nae1RotaryEmbedding with a persistent inv_freq buffer (lines 61-73); LayerNorm pre-norms (lines
    203-204) and final_layernorm (line 236); the head tied (lines 298-305)."""
    nkv = int(d["n_kv_heads"])
    for g in p["groups"]:
        if g["attn"] is not None:
            g["attn"] = dict(g["attn"], nkv=nkv)


# snapshot-extension families (graph/snapshot_extend.py), the same contract

# ------------------------------------------------------------------------------------------------ BailingMoeV3

def is_bailing_v3(p, d, raw):
    return p["mt"] == "bailing_hybrid" and "BailingMoeV3ForCausalLM" in (raw.get("architectures") or []) \
        and bool(d.get("layer_group_size"))


def _bailing_v3(p, d, raw, cl):
    """Ling-3.x (BailingMoeV3ForCausalLM; inclusionAI/Ling-3.0-tiny-base modeling_bailing_moe_v3.py, read as text):
    a layer is MLA attention when (i + 1) % layer_group_size == 0 or it lies past the last full group, else Kimi
    Delta Attention (BailingMoeV3DecoderLayer lines 1004-1014). BailingMoeV3KimiDeltaAttention (lines 722-785):
    q / k / v projections h -> heads x head_dim (lines 745-747), three ShortConvolution(kernel short_conv_kernel_size,
    no bias; lines 749-763), A_log (heads, line 765), the decay f_proj h -> heads x hd under no_kda_lora (else the
    f_a / f_b LoRA through head_dim; lines 767-771), dt_bias (heads x hd, line 773), b_proj h -> heads (line 775),
    the output gate g_proj (or g_a / g_b; lines 777-781), o_norm FusedRMSNormGated(hd) (line 783), o_proj (line 784).
    The MLA output gate is head-wise when gated_attention_proj_granularity_type is head_wise (g_proj h -> heads, lines
    625-628). MoE and the stored MTP layer (BailingMoeV3MTPLayer, lines 927-940) are the generic plan's."""
    h = p["hidden"]
    H, hd, K = int(d["num_attention_heads"]), int(d.get("head_dim") or h // int(d["num_attention_heads"])), \
        int(d.get("short_conv_kernel_size") or 4)
    q = H * hd
    lora = not d.get("no_kda_lora")
    fg = (h * hd + hd * q) if lora else h * q              # f_proj / g_proj (or their LoRA pairs)
    W = 3 * h * q + 2 * fg + h * H + q * h
    b = 3 * q * K + H + q + hd
    rec = {"family": "kda", "W": W, "b": b, "conv_state": 3 * q * K, "ssm_state": H * hd * hd,
           "core_flops": 2 * 3 * q * K + 6 * H * hd * hd, "conv_ch": 3 * q, "conv_k": K}
    headwise = str(d.get("gated_attention_proj_granularity_type") or "") == "head_wise"
    for g in p["groups"]:
        if g["rec"] is not None:
            g["rec"] = dict(rec)
        if g["attn"] is not None and g["mla"] and not g["attn"].get("mla_gate"):
            # attn_params' MLA path counts no output gate: add g_proj (h -> heads, or heads x v_head_dim)
            a = g["attn"]
            g["attn"] = dict(a, mla_gate=True, gate_headwise=headwise,
                             extra_W=a.get("extra_W", 0) + h * a["nh"] * (1 if headwise else a["dv"]))
    u = p.get("unknown")
    if u:
        u.discard("rec")
        if not u:
            p.pop("unknown")


# ------------------------------------------------------------------------------------------------ LLaDA-MoE

def is_llada_moe(p, d, raw):
    return p["mt"] == "llada" and bool(d.get("expert_intermediate_size")) and bool(d.get("num_experts"))


def _llada_moe(p, d, raw, cl):
    """LLaDA-MoE (inclusionAI/LLaDA-MoE-7B-A1B-Instruct modeling_lladamoe.py, read as text): every routed expert is a
    SwiGLU of expert_intermediate_size (LLaDAMoEMLP lines 281-297; dense_intermediate_size is the dense MLP's), and
    under qk_layernorm q_norm / k_norm RMSNorm(head_dim) (lines 354-357); sets both (idempotent)."""
    ew = int(d["expert_intermediate_size"])
    for g in p["groups"]:
        if g["moe"] is not None:
            g["moe"] = dict(g["moe"], ew=ew)
        if g["attn"] is not None and d.get("qk_layernorm"):
            g["attn"] = dict(g["attn"], qk_norm=2 * g["attn"]["dqk"])


# ------------------------------------------------------------------------------------------------ Fuyu

def is_fuyu(p, d, raw):
    return raw.get("model_type") == "fuyu" and "FuyuForCausalLM" in (raw.get("architectures") or [])


def _fuyu(p, d, raw, cl):
    """Fuyu (transformers models/fuyu/modeling_fuyu.py and models/persimmon/modeling_persimmon.py, read as text): the
    language model is Persimmon, whose MLP is dense_h_to_4h -> relu^2 -> dense_4h_to_h with biases (PersimmonMLP, not
    gated), and FuyuModel adds vision_embed_tokens, a Linear(patch_size^2 x num_channels -> hidden, bias) applied to
    image patches only -- a named stored block, not the LM (sets glu False; idempotent)."""
    for g in p["groups"]:
        if g["ffn"] is not None:
            g["ffn"] = dict(g["ffn"], glu=False)
    P, C = int(raw.get("patch_size") or 30), int(raw.get("num_channels") or 3)
    p["stored_blocks"] = dict(p.get("stored_blocks") or {},
                              **{"vision patch projection (vision_embed_tokens)": P * P * C * p["hidden"] + p["hidden"]})


# ------------------------------------------------------------------------------------------------ Dynin-Omni

def is_dynin(p, d, raw):
    return p["mt"] == "dynin_omni" and isinstance(d.get("mlp_hidden_size"), int)


def _dynin(p, d, raw, cl):
    """Dynin-Omni (snu-aidas/Dynin-Omni modeling_llada.py, read as text): the LLaDA llama block (LLaDALlamaBlock,
    lines 844-875) has ff_proj and up_proj d_model -> hidden_size and ff_out back, with hidden_size = mlp_hidden_size
    when set, else mlp_ratio x d_model (LLaDABlock, lines 526-532); sets the SwiGLU width (idempotent)."""
    I = int(d["mlp_hidden_size"])
    for g in p["groups"]:
        if g["ffn"] is not None:
            g["ffn"] = dict(g["ffn"], inter=I)


# ------------------------------------------------------------------------------------------------ stock Mamba-2

def is_mamba2_stock(p, d, raw):
    return p["mt"] == "mamba2" and bool(p["stock"])


def _mamba2_stock(p, d, raw, cl):
    """Stock Mamba-2 (transformers models/mamba2/modeling_mamba2.py, read as text): every norm is Mamba2RMSNorm, a
    weight only -- the block norm (line 689) and norm_f (line 828); the classifier's LayerNorm reading counts a bias
    too. Sets RMSNorm widths (idempotent)."""
    h = p["hidden"]
    for g in p["groups"]:
        g["norm"] = h
    p["final_norm"] = h


# ------------------------------------------------------------------------------------------------ Intern-S2-Mobius

def is_mobius(p, d, raw):
    return p["mt"] in ("interns2_mobius_text", "interns2_mobius") and isinstance(d.get("num_blocks"), int) \
        and bool(d.get("num_experts"))


def _mobius(p, d, raw, cl):
    """Intern-S2-Mobius (internlm/Intern-S2-Mobius modeling_interns2_mobius.py, read as text): the routed experts are
    NOT per layer: InternS2MobiusTextModel holds num_blocks InternS2MobiusMetaMoeBlock (lines 1388-1390), each a
    TopKRouter (num_experts x h, line 782) and fused experts gate_up_proj [E, 2 I, h] / down_proj [E, h, I] (lines
    737-746); layer i uses meta_mlp[i % num_blocks] (lines 882-883). Every layer keeps its own shared expert
    (shared_expert_intermediate_size) and a sigmoid shared_expert_gate h -> 1 (InternS2MobiusSharedExpertBlock, lines
    806-818); the attention has q_norm / k_norm RMSNorm(head_dim) and the doubled q_proj output gate (lines
    650-686). The routed bank becomes one group of n = layers applications over np = num_blocks stored copies.
    The checkpoint's mtp.* (fc 2h -> h, pre_fc_norm_embedding / hidden, norm, one attention layer with its own
    mtp_num_experts unfused experts and router, a shared expert and gate) is not built by this code: a named block."""
    h, n = p["hidden"], p["layers"]
    Bk = int(d["num_blocks"])
    bank = None
    for g in p["groups"]:
        m = g["moe"]
        if m is None:
            continue
        bank = bank or dict(m, shared=0, shared_gate=False)
        g["moe"] = None
        g["ffn"] = {"glu": True, "inter": int(m["shared"] or d.get("shared_expert_intermediate_size") or 0), "bias": False}
        g["norm"] = 2 * h
        if g["attn"] is not None:
            a = g["attn"]
            g["attn"] = dict(a, qk_norm=2 * a["dqk"], extra_W=a.get("extra_W", 0) + h)       # + shared_expert_gate
        elif g["rec"] is not None:
            g["rec"] = dict(g["rec"], W=g["rec"]["W"] + h)
    if bank is None:
        return
    p["groups"].append({"n": n, "np": Bk, "kind": "none", "window": None, "attn": None, "mla": None, "rec": None,
                        "ffn": None, "moe": bank, "norm": 0, "k_eq_v": False})
    # the stored MTP block: one full-attention layer (its q / k / v / o, q / k norms, 2 norms) with its own experts
    Em, km = int(d.get("mtp_num_experts") or bank["E"]), int(d.get("mtp_num_experts_per_tok") or bank["k"])
    ga = next(g for g in p["groups"] if g["attn"] is not None)
    a = ga["attn"]
    attn = 2 * h * a["nh"] * a["dqk"] + 2 * h * a["nkv"] * a["dqk"] + a["nh"] * a["dv"] * h + 2 * a["dqk"]
    mlp = Em * 3 * h * bank["ew"] + Em * h + 3 * h * ga["ffn"]["inter"] + h
    n_mtp = int(d.get("mtp_num_hidden_layers") or 0)
    if n_mtp:
        p["stored_blocks"] = dict(p.get("stored_blocks") or {}, **{
            f"MTP layer ({Em} own experts, not run by the model code)": n_mtp * (attn + mlp + 2 * h + 2 * h * h + 3 * h)})
    p["notes"].append(f"routed experts shared across layers: {Bk} banks of {bank['E']} experts, layer i uses bank "
                      f"i mod {Bk}")


# ------------------------------------------------------------------------------------------------ Inkling

def is_inkling(p, d, raw):
    return p["mt"] in ("inkling_mm_model", "inkling") and isinstance(d.get("local_layer_ids"), list) \
        and bool(d.get("n_routed_experts"))


def _inkling_scales(tp, ps, nl, ch):
    """plan_out_scales (transformers 5.18 models/inkling/modeling_inkling.py lines 885-960) in plain Python: the
    (t, h, w, c) grid of each HMLP vision layer boundary; the linear_sum_assignment branch is an exhaustive search
    over the (few) scale choices."""
    import itertools
    import math

    def pf(x):
        f = []
        while x % 2 == 0:
            f.append(2)
            x //= 2
        q = 3
        while q * q <= x:
            while x % q == 0:
                f.append(q)
                x //= q
            q += 2
        if x > 1:
            f.append(x)
        return f
    h, t = [], []
    for v in pf(ps)[::-1]:
        h.append(v * (h[-1] if h else 1))
    for v in pf(tp)[::-1]:
        t.append(v * (t[-1] if t else 1))
    scales = [(1, 1, 1, ch)] + [(1, x, x, -(-x * x * ch // 64) * 64) for x in h] + \
        [(x, h[-1], h[-1], -(-(h[-1] ** 2 * ch * x) // 1) * 64) for x in t]
    red = [math.log(s[0] * s[1] * s[2]) for s in scales]
    tot = math.log(ps * ps * tp * ch)
    ideal = [tot * i / nl for i in range(nl + 1)]
    cost = [[abs(a - b) for b in red] for a in ideal]
    if nl >= len(scales):
        idx = [min(range(len(scales)), key=lambda j: c[j]) for c in cost]
    else:
        idx = list(min(itertools.permutations(range(len(scales)), nl + 1),
                       key=lambda pm: sum(cost[i][j] for i, j in enumerate(pm))))
    idx[0], idx[-1] = 0, len(scales) - 1
    return [scales[j] for j in idx]


def _inkling(p, d, raw, cl):
    """Inkling (thinkingmachines/Inkling, Inkling-Small; the repos ship no code: transformers 5.18
    models/inkling/modeling_inkling.py, read as text, with the checkpoint's own tensor names, model.llm.* / model.mtp.*):
      attention   (InklingAttention, lines 185-215) per layer local (local_layer_ids: swa_* heads, window
                  sliding_window_size, rel_extent = the window) or global (rel_extent): q / k / v / o, r_proj
                  h -> heads x d_rel (the hidden-state-conditioned relative bias, lines 119-135: proj
                  [d_rel, rel_extent]), q_norm / k_norm (head_dim), k_sconv / v_sconv depthwise convs (kernel
                  sconv_kernel_size, no bias; lines 483-500)
      layer       (lines 545-564) two RMSNorms and two hidden-width short convs (attn_sconv, mlp_sconv)
      FFN         the first dense_mlp_idx layers a dense SwiGLU of dense_intermediate_size with a global_scale
                  (lines 285-295); the rest InklingMoE (lines 407-420): n_routed_experts SwiGLU experts of
                  intermediate_size (top num_experts_per_tok), n_shared_experts shared experts of the same width
                  (lines 380-390), a router over E + n_shared rows with its global_scale (lines 342-353) and, with
                  use_gate_bias, a gate.bias of E
      model       embed_tokens with embed_norm (lines 594-600), final norm, an untied head (model.llm.unembed)
    Stored beside the LM: the mtp_config MTP layers (embed_norm, hidden_norm, input_proj 2h -> h, one dense
    transformer block each, local per mtp_config.local_layer_ids), the HMLP vision encoder (lines 966-995: one
    projection per scale step, an RMSNorm after each but the last, final_norm) and the dMel audio embedding
    (n_mel_bins x mel_vocab_size rows of h, final_norm)."""
    h, n = p["hidden"], p["layers"]
    K = int(d.get("sconv_kernel_size") or d.get("conv_kernel_size") or 4)
    loc = set(int(i) for i in d["local_layer_ids"] if isinstance(i, int))
    W = int(d.get("sliding_window_size") or 0)
    dr = int(d.get("d_rel") or 16)
    E, k, ew = int(d["n_routed_experts"]), int(d["num_experts_per_tok"]), int(d["intermediate_size"])
    S = int(d.get("n_shared_experts") or 0)
    dense_n, dense_w = int(d.get("dense_mlp_idx") or 0), int(d.get("dense_intermediate_size") or ew)

    def attn(local, extent):
        nh = int(d.get("swa_num_attention_heads") or d["num_attention_heads"]) if local else int(d["num_attention_heads"])
        nkv = int(d.get("swa_num_key_value_heads") or d["num_key_value_heads"]) if local else int(d["num_key_value_heads"])
        hd = int(d.get("swa_head_dim") or d["head_dim"]) if local else int(d["head_dim"])
        return {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": False, "o_bias": False, "qk_norm": 2 * hd,
                "out_gate": False, "gate_headwise": False, "extra_W": h * nh * dr,
                "extra_b": 2 * nkv * hd * K + dr * extent}

    def layer(i, local, extent, moe):
        g = {"kind": "sliding" if local else "full", "window": W if local else None, "attn": attn(local, extent),
             "mla": None, "rec": None, "ffn": None, "moe": None, "norm": 2 * h + 2 * h * K, "k_eq_v": False}
        if moe:
            g["moe"] = {"E": E, "k": k, "ew": ew, "shared": S * ew, "shared_gate": False,
                        "router_bias": bool(d.get("use_gate_bias", True)), "expert_bias": False, "glu": True,
                        "out_bias": False, "dense_compute": False}
            g["attn"]["extra_W"] += S * h                   # the router's n_shared rows
            g["norm"] += 1                                  # the router's global_scale
        else:
            g["ffn"] = {"glu": True, "inter": dense_w, "bias": False}
            g["norm"] += 1                                  # the MLP's global_scale
        return g
    ext = int(d.get("rel_extent") or 1024)
    p["groups"] = _per_layer_groups([layer(i, i in loc, W if i in loc else ext, i >= dense_n) for i in range(n)])
    p["extra"] += h                                         # embed_norm
    p["final_norm"] = h
    p["tied"], p["lm_head"] = False, p["lm_head_W"]
    blocks = {}
    mc = raw.get("mtp_config") if isinstance(raw.get("mtp_config"), dict) else {}
    nm = int(mc.get("num_nextn_predict_layers") or 0)
    if nm:
        ml = set(int(i) for i in mc.get("local_layer_ids") or [])
        tb = sum(_layer_total(layer(j, j in ml, W if j in ml else ext, False), h) for j in range(nm))
        blocks["MTP layers (Inkling, dense blocks)"] = tb + nm * (2 * h + 2 * h * h)
    v = raw.get("vision_config") if isinstance(raw.get("vision_config"), dict) else {}
    if v.get("n_layers"):
        sc = _inkling_scales(int(v.get("temporal_patch_size") or 2), int(v["patch_size"]), int(v["n_layers"]),
                             int(v.get("n_channels") or 3))
        vt = 0
        for i, (a, b) in enumerate(zip(sc[:-1], sc[1:])):
            mult = (b[0] // a[0]) * (b[1] // a[1]) * (b[2] // a[2])
            last = i == len(sc) - 2
            out = h if last else b[3]
            vt += a[3] * mult * out + (0 if last else out)
        blocks["vision encoder (HMLP)"] = vt + h
    a = raw.get("audio_config") if isinstance(raw.get("audio_config"), dict) else {}
    if a.get("n_mel_bins"):
        blocks["audio embedding (dMel)"] = int(a["n_mel_bins"]) * int(a.get("mel_vocab_size") or 16) * h + h
    p["stored_blocks"] = dict(p.get("stored_blocks") or {}, **blocks)


# ------------------------------------------------------------------------------------------------ CFRD (haru)

def is_cfrd(p, d, raw):
    return p["mt"] == "cfrd" and isinstance(d.get("physical_cells"), int)


def _cfrd(p, d, raw, cl):
    """CFRD, the Causal Folded Recurrent Decoder (alice-noa-chan/haru, haru_1.1, haru_2: model.py, configuration_cfrd.py,
    modeling_cfrd.py, read as text; three revisions, haru 5e722ae, haru_1.1 b834a95, haru_2 5f658fa).
      FoldedCell (haru model.py lines 412-432): attn_norm, ffn_norm RMSNorm(h); LocalCausalAttention (lines 199-264:
        q / k / v / o, no bias, causal attention inside fixed chunk_size chunks, global RoPE positions) or, with
        cell_attention "full" (haru_2 lines 512-519), FullCausalAttention and no memory; CausalSummaryMemory (lines
        267-316): summary_queries [S, m], summary_k / summary_v h -> m, read_q h -> m, read_k / read_v m -> m, read_o
        m -> h, null_memory [m], recency_decay_raw [memory_heads], and two persistent masks [ctx, 1 + chunks x S]
        (memory_allowed bool, memory_distance float); SwiGLU w1 / w3 / w2 (lines 400-406); attn_gate / ffn_gate
        Linear(h, 1, bias) (lines 426-427)
      CFRDLanguageModel (lines 480-536): token_embedding, the tokenizer-derived surface_feature_table [V, F]
        (persistent) with surface_projection F -> h and a 0-d surface_gain, physical_cells cells applied at each of
        the recurrences depths in turn (cell depth % cells), FiLM attn / ffn phase scale and shift [R, h] each,
        memory_gain [R] (local cells only), final_norm, rope_cos / rope_sin [ctx, hd / 2] (persistent); with
        use_binding_block (haru_1.1 lines 460-490) one more full-attention block (two norms, attention, SwiGLU, two
        gates) after the last depth. The LM head is the embedding in every revision (modeling_cfrd.py line 53),
        whatever tie_word_embeddings says (haru, haru_1.1 set it false).
    The phase shifts are the norm betas the classifier reads as bias_qkv / bias_out (cfrd_film_norm_shift): no
    projection has a bias."""
    h, V = p["hidden"], p["vocab"]
    nh, nkv = int(d["num_attention_heads"]), int(d.get("num_key_value_heads") or d["num_attention_heads"])
    hd = h // nh
    ff = int(d["intermediate_size"])
    m, S, mh = int(d["memory_dim"]), int(d["summary_slots"]), int(d["memory_heads"])
    R = int(d.get("inference_recurrences") or d["recurrences"])
    Rp = int(d["recurrences"])
    C = int(d["physical_cells"])
    ctx, ch = int(d.get("context_length") or d["max_position_embeddings"]), int(d["chunk_size"])
    local = str(d.get("cell_attention") or "local") == "local"
    att = {"nh": nh, "nkv": nkv, "dqk": hd, "dv": hd, "qkv_bias": False, "o_bias": False, "qk_norm": 0,
           "out_gate": False, "gate_headwise": False, "extra_W": 2 * h, "extra_b": 2}       # the two residual gates
    cell_att = dict(att)
    if local:
        cell_att["extra_W"] += 3 * h * m + 2 * m * m + m * h                              # summary k / v, read q / k / v / o
        cell_att["extra_b"] += S * m + m + mh
    ffn = {"glu": True, "inter": ff, "bias": False}
    base = {"mla": None, "rec": None, "moe": None, "ffn": ffn, "norm": 2 * h, "k_eq_v": False}
    cell = dict(base, n=R, np=C, at=list(range(R)), kind="sliding" if local else "full",
                window=ch if local else None, attn=cell_att)
    groups = [cell]
    if d.get("use_binding_block"):
        groups.append(dict(base, n=1, np=1, at=[R], kind="full", window=None, attn=dict(att)))
    p["groups"] = groups
    p["layers"] = R + (1 if d.get("use_binding_block") else 0)
    p["order_exact"] = True
    _set_tied(p, True)
    surf = bool(d.get("use_surface_features"))
    F = int(d.get("surface_feature_dim") or 0) if surf else 0
    p["extra_W"] += F * h                                   # surface_projection, once per token
    p["extra"] += 4 * Rp * h + (Rp if local else 0) + (1 if surf else 0)
    blocks = {"surface feature table + rope cache + memory masks (persistent buffers)":
              V * F + 2 * ctx * (hd // 2) + (C * 2 * ctx * (1 + (ctx // ch) * S) if local else 0)}
    if surf:
        blocks["0-d scalars (the Hub API counts none)"] = -1                       # surface_gain
    p["stored_blocks"] = dict(p.get("stored_blocks") or {}, **blocks)
    p["notes"].append(f"CFRD: {C} folded cells over {R} depths" + (", chunk-local attention + summary memory"
                      if local else ", full attention") + (", a final binding block" if d.get("use_binding_block") else ""))


BESPOKE = ((is_token_routed, _token_routed),
           (is_cfrd, _cfrd),
           (is_meditor, _meditor),
           (is_xing_hc, _xing_hc),
           (is_gpt_s3x3, _gpt_s3x3),
           (is_klm, _klm),
           (is_cyrah, _cyrah),
           (is_dory, _dory),
           (is_zaya1_vl, _zaya1_vl),
           (is_hybrid_gka, _hybrid_gka),
           (is_jet_nemotron, _jet_nemotron),
           (is_slider_gpt, _slider_gpt),
           (is_lfm3_rd, _lfm3_rd),
           (is_domynedge, _domynedge),
           (is_k2_horizon_mova, _k2_horizon_mova),
           (is_qwen4_exp, _qwen4_exp),
           (is_step3, _step3),
           (is_sdar_selfcond, _sdar_selfcond),
           (is_sambay, _sambay),
           (is_hymba, _hymba),
           (is_sundial, _sundial),
           (is_ethoric, _ethoric),
           (is_arloop, _arloop),
           (is_gptx2, _gptx2),
           (is_inferno, _inferno),
           (is_bananamind2_moe, _bananamind2_moe),
           (is_pit, _pit),
           (is_ivme, _ivme),
           (is_kohaku_moe, _kohaku_moe),
           (is_memt_qwen35_moe, _memt_qwen35_moe),
           (is_meditor2, _meditor2),
           (is_nexus, _nexus),
           (is_kanana2_tiny, _kanana2_tiny),
           (is_phi3small, _phi3small),
           (is_minicpm_remote_tied, _minicpm_remote_tied),
           (is_mima, _mima),
           (is_apertus1p5, _apertus1p5),
           (is_gemma3_moe, _gemma3_moe),
           (is_bitsy, _bitsy),
           (is_crowd, _crowd), (is_nae1, _nae1),
           (is_bailing_v3, _bailing_v3),
           (is_llada_moe, _llada_moe),
           (is_fuyu, _fuyu),
           (is_dynin, _dynin),
           (is_mamba2_stock, _mamba2_stock),
           (is_mobius, _mobius),
           (is_inkling, _inkling))


def _bespoke(p, d, raw, cl):
    for test, fn in BESPOKE:
        if test(p, d, raw):
            fn(p, d, raw, cl)
            return


# ------------------------------------------------------------------------------------------------ evaluation

def attn_params(a, mla, hidden, k_eq_v=False):
    """(linear weights, weights applied per token, other params) of one attention block."""
    if a.get("W_override") is not None:                        # a builder counted the block itself (DeepSeek-V4)
        return a["W_override"], a["W_override"], a.get("b_override", 0)
    nh, nkv, dqk, dv = a["nh"], a["nkv"], a["dqk"], a["dv"]
    extra_W, other = a.get("extra_W", 0), a.get("extra_b", 0) + a.get("sinks", 0)
    if mla:
        W = 0
        if mla["q_lora"]:
            W += hidden * mla["q_lora"] + mla["q_lora"] * nh * dqk
            other += mla["q_lora"]                             # q_a_layernorm
        else:
            W += hidden * nh * dqk
        W += hidden * (mla["kv_lora"] + mla["rope"]) + mla["kv_lora"] * nh * (mla["nope"] + dv)
        other += mla["kv_lora"]                                # kv_a_layernorm
        W += nh * dv * hidden + extra_W
        other += hidden if a["o_bias"] else 0
        return W, W, other
    moa = a.get("moa")
    if moa:                                                    # JetMoE mixture of attention experts
        E, k, D = moa["E"], moa["k"], moa["D"]
        W = E * 2 * hidden * D + 2 * hidden * D + hidden * E
        Wa = k * 2 * hidden * D + 2 * hidden * D + hidden * E
        return W, Wa, other + hidden
    xin = a.get("in_dim") or hidden                            # Zamba: the shared block reads concat(h, emb)
    Wq, Wk, Wv = xin * nh * dqk, xin * nkv * dqk, (0 if k_eq_v else xin * nkv * dv)
    if a.get("no_kv_proj"):                                    # Gemma-4 KV-shared layers: q and o only
        Wk = Wv = 0
    Wo = nh * dv * hidden
    Wg = (hidden * nh if a.get("gate_headwise") else hidden * nh * dv) if a.get("out_gate") else 0   # (head-wise: a scalar per head)
    W = Wq + Wk + Wv + Wo + Wg + extra_W
    Wa = W - a.get("W_inactive", 0)                           # MoVA: routed value experts not selected
    other += a["qk_norm"]
    if a["qkv_bias"]:
        other += nh * dqk + (0 if a.get("no_kv_proj") else nkv * dqk + (0 if k_eq_v else nkv * dv))
    if a["o_bias"]:
        other += hidden
    return W, Wa, other


def ffn_params(f, hidden):
    """(weights, other) of a dense FFN."""
    if f is None:
        return 0, 0
    if f.get("linear"):                                        # DeciLM replace_with_linear: one h x h map
        return hidden * hidden, 0
    if f.get("rwkv"):                                          # RWKV channel mix: key, receptance (h x h), value
        return hidden * f["inter"] + hidden * hidden + f["inter"] * hidden, 2 * hidden
    m = 3 if f["glu"] else 2
    W = m * hidden * f["inter"]
    other = ((2 if f["glu"] else 1) * f["inter"] + hidden) if f["bias"] else 0
    if f["glu"] and f["bias"]:
        other = 2 * f["inter"] + hidden
    return W, other


def moe_params(m, hidden):
    """(total weights, active weights, other params, routed weights) of a MoE FFN."""
    if m is None:
        return 0, 0, 0, 0
    xin = m.get("latent") or hidden                           # Nemotron-3 latent MoE: experts act in the latent
    per = (3 if m.get("glu", True) else 2) * xin * m["ew"]
    routed = m["E"] * per
    shared = (3 if m.get("shared_glu", m.get("glu", True)) else 2) * hidden * m["shared"]
    if m.get("latent"):
        shared += 2 * hidden * m["latent"]                     # fc1 / fc2 latent projections (always on)
    router = 0 if m.get("no_router") else hidden * (m["E"] + m.get("zero", 0))   # LongCat: + the zero experts;
    # no_router: a token-id routed MoE (a fixed token -> expert table, no router weight)
    gate = hidden if m["shared_gate"] else 0
    other = (m["E"] if m["router_bias"] else 0)
    if m["expert_bias"]:
        other += m["E"] * (2 * m["ew"] + hidden)
    if m.get("out_bias"):
        other += hidden                                        # JetMoE MoE: one output bias
    total = routed + shared + router + gate
    active = m["k"] * per + shared + router + gate
    return total, active, other, routed


def layer_params(g, hidden):
    """Per-layer dict of parameter counts and per-token matmul weights for one group."""
    out = {"attn_W": 0, "attn_Wa": 0, "attn_b": 0, "ffn_W": 0, "ffn_Wa": 0, "ffn_b": 0, "routed": 0, "rec_W": 0, "rec_b": 0,
           "norm": g["norm"]}
    if g["attn"] is not None:
        out["attn_W"], out["attn_Wa"], out["attn_b"] = attn_params(g["attn"], g["mla"], hidden, g["k_eq_v"])
    if g["ffn"] is not None:
        W, b = ffn_params(g["ffn"], hidden)
        out.update(ffn_W=W, ffn_Wa=W, ffn_b=b)
    if g["moe"] is not None:
        t, a, b, r = moe_params(g["moe"], hidden)
        out.update(ffn_W=t, ffn_Wa=a, ffn_b=b, routed=r)
    if g["rec"] is not None:
        out["rec_W"], out["rec_b"] = g["rec"].get("W", 0), g["rec"].get("b", 0)
    return out


def params(p):
    """Parameter counts of a plan (schema `params`, without the hub fields and bytes)."""
    h = p["hidden"]
    tot = p["embedding"] + p["lm_head"] + p["final_norm"] + p["extra"] + p["extra_W"]
    routed = inactive = 0
    for g in p["groups"]:
        lp = layer_params(g, h)
        per = (lp["attn_W"] + lp["attn_b"] + lp["ffn_W"] + lp["ffn_b"] + lp["rec_W"] + lp["rec_b"] + lp["norm"])
        tot += g["np"] * per
        routed += g["np"] * lp["routed"]
        inactive += g["np"] * (lp["ffn_W"] - lp["ffn_Wa"] + lp["attn_W"] - lp["attn_Wa"])
    non_emb = tot - p["embedding"] - p["lm_head"]
    return {"total": int(tot), "embedding": int(p["embedding"]), "lm_head": int(p["lm_head"]),
            "non_embedding": int(non_emb), "active": int(tot - inactive),
            "active_non_embedding": int(non_emb - inactive), "routed_experts": int(routed),
            "tied_embeddings": bool(p["tied"])}


def matmul_weights(p, code=False):
    """Linear weights applied per token, unrolled (MoE: active), without the head. code=True: what the transformers
    code multiplies, where it computes every expert on every token (Llama-4: Llama4TextExperts bmm over all experts,
    the router scores masking the output) -- the trace convention, not the architecture's active compute."""
    h = p["hidden"]
    s = 0
    for g in p["groups"]:
        lp = layer_params(g, h)
        ffn = lp["ffn_Wa"]
        if code and g["moe"] is not None and g["moe"].get("dense_compute"):
            ffn = lp["ffn_W"]
        s += g["n"] * (lp["attn_Wa"] + ffn + lp["rec_W"])
    return s


def _min(a, b):
    """min that stays symbolic for a sympy L (the symbolic proofs evaluate these formulas at a symbol)."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    import sympy
    return sympy.Min(a, b)


def _span_causal_sum(L, W):
    """sum_{i=1..L} min(i, W)."""
    if W is None or L <= W:
        return L * (L + 1) // 2
    return W * (W + 1) // 2 + (L - W) * W


def attn_flops(p, L, mode):
    """Attention score + value FLOPs per token: mode 'prefill' (causal, amortised), 'prefill_dense', 'decode'."""
    tot = 0
    for g in p["groups"]:
        a = g["attn"]
        if a is None:
            continue
        per_key = 2 * a["nh"] * (a["dqk"] + a["dv"])
        W = g["window"]
        dsa = p.get("dsa") if g["kind"] != "shared" else None
        if dsa and mode != "prefill_dense":
            # DeepSeek sparse attention: the indexer scores every key (index heads x index head dim), the attention
            # itself reads the top index_topk keys
            idx_key = 2 * dsa["heads"] * dsa["dim"] if g.get("dsa_index", True) else 0
            if mode == "decode":
                tot += g["n"] * (per_key * _min(L, dsa["topk"]) + idx_key * L)
            else:
                tot += g["n"] * (per_key * _span_causal_sum(L, dsa["topk"]) + idx_key * _span_causal_sum(L, None)) / L
            continue
        if mode == "decode":
            span = _min(L, W) if W else L
            tot += g["n"] * per_key * span
        elif mode == "prefill_dense":
            tot += g["n"] * per_key * L
        else:
            tot += g["n"] * per_key * _span_causal_sum(L, W) / L
    return tot


def _one_kv_tensor_hf(p, g):
    """Does the transformers cache hold ONE tensor for this K = V group? DeepSeek-V4's cache layer aliases values to
    keys (modeling_deepseek_v4.py update: self.values = self.keys). Gemma-4's attention_k_eq_v layers do not: K is
    RoPE(k_norm(k_proj x)) and V is v_norm(k_proj x), two different tensors, and cache.update stores both
    (modeling_gemma4.py Gemma4TextAttention; vLLM gemma4.py and llama.cpp gemma4.cpp also cache K and V)."""
    return bool(g["k_eq_v"]) and str(p["mt"]).lower() in VNORM_TYPES


def _kv_tensors(p, g):
    """K / V tensors a K = V group's cache holds per KV head: one for DeepSeek-V4 (the cache aliases V to K), two
    for Gemma-4 (K and V differ after their norms and RoPE, and every runtime stores both)."""
    return 1 if _one_kv_tensor_hf(p, g) else 2


def kv_per_token(p):
    """Elements cached per token per kind (window layers while L < W): the headline convention (as allocated by
    vLLM / SGLang / transformers / llama.cpp; MLA as the latent), the transformers-cache convention (MLA per-head
    K / V), and the theoretical minimum (one tensor for every K = V layer)."""
    e = {"global": 0, "sliding": 0, "mla_latent": 0, "k_eq_v": 0, "dsa_indexer": 0, "kv_shared": 0}
    hf = {"global": 0, "sliding": 0}
    mn = 0
    for g in p["groups"]:
        a = g["attn"]
        if a is None or g["kind"] == "shared":
            continue
        if g["mla"]:
            e["mla_latent"] += g["n"] * (g["mla"]["kv_lora"] + g["mla"]["rope"])
            mn += g["n"] * (g["mla"]["kv_lora"] + g["mla"]["rope"])
            per_hf = a["nh"] * (a["dqk"] + a["dv"])
        else:
            if g["k_eq_v"]:
                per = a["nkv"] * a["dqk"] * _kv_tensors(p, g)
                e["k_eq_v"] += g["n"] * per
                mn += g["n"] * a["nkv"] * a["dqk"]
            else:
                per = a["nkv"] * (a["dqk"] + a["dv"])
                e["global" if g["kind"] == "full" else "sliding"] += g["n"] * per
                mn += g["n"] * per
            per_hf = per
        hf["global" if g["kind"] == "full" else "sliding"] += g["n"] * per_hf
    return e, hf, mn


def kv_at(p, L, convention="headline"):
    """{kind: elements} cached after L tokens (one sequence). convention: 'headline' (as allocated, MLA latent),
    'hf' (the transformers cache objects: MLA per-head K / V), 'minimum' (headline with one tensor per K = V
    layer, the theoretical floor no runtime implements for Gemma-4)."""
    out = {"global": 0, "sliding": 0, "mla_latent": 0, "k_eq_v": 0, "dsa_indexer": 0}
    for g in p["groups"]:
        a = g["attn"]
        if a is None or g["kind"] == "shared":
            continue
        W = g["window"]
        toks = _min(L, W - 1) if W else L
        if convention == "hf":
            per = a["nh"] * (a["dqk"] + a["dv"]) if g["mla"] else (
                a["nkv"] * a["dqk"] * _kv_tensors(p, g) if g["k_eq_v"] else a["nkv"] * (a["dqk"] + a["dv"]))
            out["global" if not W else "sliding"] += g["n"] * per * toks
            continue
        if g["mla"]:
            out["mla_latent"] += g["n"] * (g["mla"]["kv_lora"] + g["mla"]["rope"]) * toks
        elif g["k_eq_v"]:
            out["k_eq_v"] += g["n"] * a["nkv"] * a["dqk"] * toks * (1 if convention == "minimum" else _kv_tensors(p, g))
        else:
            out["global" if not W else "sliding"] += g["n"] * a["nkv"] * (a["dqk"] + a["dv"]) * toks
    return out


def state_elems(p):
    conv = ssm = 0
    for g in p["groups"]:
        r = g["rec"]
        if r:
            conv += g["n"] * r.get("conv_state", 0)
            ssm += g["n"] * r.get("ssm_state", 0)
    return conv, ssm


def rec_flops(p, mode):
    s = 0
    for g in p["groups"]:
        r = g["rec"]
        if r:
            s += g["n"] * r.get("core_flops", 0)
    return s


# What transformers' CPU reference path EXECUTES, where it differs from the convention above (the trace check
# verifies these exactly; the headline numbers are the convention):
#   conv     nn.Conv1d with padding K - 1 computes L + K - 1 outputs in a prefill (then slices L); a cached decode
#            step convolves with torch.sum(conv_state * weight) (elementwise: no matmul FLOPs)
#   readout  Mamba-1 slow path: one matmul(ssm_state, C) per token (2 x d_inner x N)
#   W_raw    projections applied as raw parameters inside the mixer (Zamba x_proj / dt_proj, RG-LRU gates)
TRACE_CORE_FAMILIES = ("mamba1", "shortconv", "rglru")


def trace_core(p, L, mode):
    """The recurrent-mixer FLOPs the transformers CPU path counts outside its nn.Linear layers (total over the
    prefill's L tokens, or one decode step), or None when a family's reference path is not modelled."""
    s = 0
    for g in p["groups"]:
        r = g["rec"]
        if not r or r.get("family") == "glue":
            continue
        if r.get("family") not in TRACE_CORE_FAMILIES:
            return None
        if mode == "decode":
            s += g["n"] * (r.get("readout", 0) + 2 * r.get("W_raw", 0))
        else:
            conv = 2 * r["conv_ch"] * r["conv_k"] * (L + r["conv_k"] - 1)
            s += g["n"] * (conv + L * (r.get("readout", 0) + 2 * r.get("W_raw", 0)))
    return s


def code_matmul(p):
    """Per-token matmul FLOPs the transformers code executes (trace convention), without the recurrent raw weights."""
    return 2 * (matmul_weights(p, code=True) + p["lm_head_W"] + p["extra_W"] - raw_weights(p))


def raw_weights(p):
    """Per-token weights inside recurrent mixers applied as raw parameters (counted in trace_core)."""
    return sum(g["n"] * (g["rec"].get("W_raw", 0) if g["rec"] else 0) for g in p["groups"])


def trace_attn(p, L, mode):
    """Attention FLOPs the transformers path executes: a prefill of L tokens (total) or one decode step whose cache
    holds L - 1 tokens. Dense (masked) scores; RecurrentGemma scores every query against its fixed W-slot buffer."""
    if p["mt"] == "recurrent_gemma":
        per = sum(g["n"] * 2 * g["attn"]["nh"] * (g["attn"]["dqk"] + g["attn"]["dv"]) * g["window"]
                  for g in p["groups"] if g["attn"] is not None)
        return per * (L if mode == "prefill" else 1)
    if mode == "prefill":
        return L * attn_flops(p, L, "prefill_dense")
    extra = 0
    if p["mt"] in NO_CACHE_TYPES:
        return attn_flops(p, 1, "decode")
    if p["mt"] == "doge":
        # DogeAttention recomputes dt_proj over every cached value at each decode step (the convention counts each
        # token's dt once): L - 1 extra applications of the (KV heads x head_dim -> KV heads) map per layer
        extra = sum(g["n"] * 2 * g["attn"].get("extra_W", 0) * (L - 1) for g in p["groups"] if g["attn"] is not None)
    return attn_flops(p, L, "decode") + extra


NO_CACHE_TYPES = frozenset({"openai-gpt"})      # stock code without a KV cache (each decode step sees itself only)
# stock MoE code that multiplies every expert on every token in eval (router scores mask the outputs): Llama-4
# Llama4TextExperts, Qwen3-VL-MoE Qwen3VLMoeTextExperts (gpt-oss does too, but only off CPU)
DENSE_EXPERT_CODE = frozenset({"llama4", "llama4_text", "qwen3_vl_moe", "qwen3_vl_moe_text"})


def hf_alloc(p, L):
    """(kv, conv, ssm) elements the transformers cache objects hold after L tokens, where the code allocates more
    than the model needs: RecurrentGemma's attention cache is a fixed attention_window_size buffer; LFM2's
    Lfm2HybridConvCache allocates a conv state for every layer (attention layers included)."""
    kv = 0 if p["mt"] in NO_CACHE_TYPES else sum(kv_at(p, L, "hf").values())
    if p.get("falcon_new_arch"):
        # FalconAttention (new_decoder_architecture) broadcasts K / V to every query head before the cache update:
        # the cache holds num_attention_heads heads, not num_kv_heads
        kv = sum(g["n"] * g["attn"]["nh"] * (g["attn"]["dqk"] + g["attn"]["dv"]) * L
                 for g in p["groups"] if g["attn"] is not None)
    conv, ssm = state_elems(p)
    if p["mt"] == "recurrent_gemma":
        kv = sum(g["n"] * g["attn"]["nkv"] * (g["attn"]["dqk"] + g["attn"]["dv"]) * g["window"]
                 for g in p["groups"] if g["attn"] is not None)
    if p["mt"] == "lfm2":
        r = next(g["rec"] for g in p["groups"] if g["rec"])
        conv = p["layers"] * r["conv_state"]
    return kv, conv, ssm


def flops(p, L, mode):
    """{matmul, attn, recurrent, total} per token at context L (mode: 'prefill', 'prefill_dense', 'decode')."""
    mm = 2 * (matmul_weights(p) + p["lm_head_W"] + p["extra_W"])
    at = attn_flops(p, L, mode)
    rc = rec_flops(p, mode)
    return {"matmul": int(mm), "attn": int(round(at)), "recurrent": int(rc), "total": int(round(mm + at + rc))}


# ------------------------------------------------------------------------------------------------ the JSON record

def _kind_counts(p):
    c = {"total": int(p["layers"]), "global": 0, "sliding": 0, "window": None, "mla": 0, "kv_shared": 0,
         "k_eq_v": 0, "recurrent": 0, "dsa_indexer": 0}
    for g in p["groups"]:
        if g["kind"] == "none":
            continue
        if g["attn"] is not None:
            if g["kind"] == "shared":
                c["kv_shared"] += g["n"]
            elif g["kind"] == "full":
                c["global"] += g["n"]
            else:
                c["sliding"] += g["n"]
                c["window"] = g["window"]
            if g["mla"]:
                c["mla"] += g["n"]
            if g["k_eq_v"]:
                c["k_eq_v"] += g["n"]
            if p.get("dsa_dim") and g["kind"] != "shared" and g.get("dsa_index", True):
                c["dsa_indexer"] += g["n"]
        if g["rec"] is not None and g["rec"].get("family") != "glue":
            c["recurrent"] += g["n"]
    return c


def _kv_headline(p, L, convention="headline"):
    """{kind: elements} after L tokens, headline (or minimum) convention, with the DSA indexer keys."""
    k = kv_at(p, L, convention)
    if p.get("dsa_dim"):
        n = sum(g["n"] for g in p["groups"] if g["attn"] is not None and g["kind"] != "shared"
                and g.get("dsa_index", True))
        k["dsa_indexer"] = n * p["dsa_dim"] * L
    return k


def _bytes(elems, dt):
    b = elems * DTYPE_BYTES[dt]
    return int(b) if float(b).is_integer() else b


def contexts_for(p):
    ctx = dict(CONTEXTS)
    ctx["max"] = int(p["max_context"])
    return ctx


QUANT_MARKERS = ("nvfp4", "mxfp4", "fp8", "fp4", "int4", "int8", "4bit", "8bit", "awq", "gptq", "bnb", "gguf", "exl2",
                 "-q4", "-q8", "mlx", "quant", "w4a", "w8a", "3bit", "6bit", "5bit", "2bit", "jang", "oq4", "oq8", "oq2", "oq3", "oq5", "oq6", "bpw")


def is_quantised(repo, raw):
    q = raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config") or \
        (raw.get("quantization") if isinstance(raw.get("quantization"), dict) else None)   # mlx convert's key
    return bool(q) or any(m in (repo or "").lower() for m in QUANT_MARKERS)


def _layer_total(g, h):
    lp = layer_params(g, h)
    return lp["attn_W"] + lp["attn_b"] + lp["ffn_W"] + lp["ffn_b"] + lp["rec_W"] + lp["rec_b"] + lp["norm"]


# stored blocks of one checkpoint that are an upload artefact, not the architecture (computed from the plan): named so the
# Hub count is matched exactly, while the formula stays the architecture's
CHECKPOINT_BLOCKS = {
    # the root shards store every MoE layer's router (mlp.gate, E x h) and shared_expert_gate (1 x h) twice
    # (model-00002 and model-00011, model-00003 and model-00012, ...: the safetensors headers of the 13 root files,
    # read 2026-11-14); the bf16/ and bf16-v4.3/ folders hold them once
    "whittleOrg/Whittle-Next-27B-A3B": lambda p: {"router and shared-gate tensors stored twice (two shards)": sum(
        g["n"] * (g["moe"]["E"] + 1) * p["hidden"] for g in p["groups"] if g["moe"] is not None)},
}


def hub_components(p, raw, non_lm=None, repo=None):
    """Named, exactly computed parameter blocks a checkpoint may hold beyond the LM count: the stored copy of a tied
    head, MTP modules (DeepSeek / GLM style: a decoder layer + enorm / hnorm / eh_proj / head norm, optionally with
    their own embedding and head copies; Qwen-Next style: a layer + fc + 3 norms), stored router-bias buffers,
    causal-mask / rotary buffers of old checkpoints, and the non-LM towers (counted on the meta device for stock
    classes)."""
    h, V, n = p["hidden"], p["vocab"], p["layers"]
    comp = {}
    if p["tied"]:
        comp["tied head stored"] = V * h
    t = raw.get("text_config") or raw
    n_mtp = int(t.get("num_nextn_predict_layers") or t.get("mtp_num_hidden_layers") or t.get("num_mtp_layers") or 0)
    if p["mt"] in ("qwen3_next",) and not n_mtp:
        n_mtp = 1                                           # Qwen3-Next checkpoints ship one MTP layer (no config key)
    if n_mtp:
        lay = max(p["groups"], key=lambda g: (g["moe"] is not None, g["attn"] is not None and g["kind"] == "full",
                                             g.get("dsa_index", True)))
        L1 = _layer_total(lay, h)                           # the MTP block is one decoder layer of the main kind
        comp["MTP layers"] = n_mtp * (L1 + 2 * h * h + 3 * h)
        comp["MTP layers (+ embedding and head copies)"] = comp["MTP layers"] + n_mtp * 2 * V * h
    if p["arch"] in LONGCAT_ARCHS:
        # one stored MTP layer whatever mtp_num_layers says (LongCat-Flash / 2.0 / Lite safetensors: model.mtp.*):
        # embedding copy, enorm / hnorm, eh_proj (2h -> h), one MLA attention (+ indexer), one dense MLP, 2 norms,
        # final norm; with mtp_disable_over_tokenizer, the MTP's own over-tokenizer tables
        A = p["groups"][0]
        la = _layer_total(dict(A, ffn=None, norm=0), h)
        comp["MTP layer (LongCat)"] = V * h + h + 2 * h * h + 4 * h + la + 3 * h * A["ffn"]["inter"]
        if t.get("mtp_disable_over_tokenizer") and p.get("longcat_oe"):
            comp["MTP over-tokenizer copy"] = p["longcat_oe"]
    n_moe = sum(g["n"] for g in p["groups"] if g["moe"] is not None)
    if n_moe:
        E = next(g["moe"]["E"] + g["moe"].get("zero", 0) for g in p["groups"] if g["moe"] is not None)
        comp["router bias buffers"] = sum(g["n"] * (g["moe"]["E"] + g["moe"].get("zero", 0)) for g in p["groups"]
                                          if g["moe"] is not None)
        if n_mtp:
            comp["router bias buffers"] += n_mtp * E           # the MTP layers' routers too
            dense = [g for g in p["groups"] if g["moe"] is None and g["attn"] is not None and g["kind"] == "full"]
            if dense:
                # a dense MTP layer with its own head copy and no embedding copy, and so no MTP router bias
                # (jdopensource/JoyAI-LLM-Flash safetensors: model.layers.40.{mlp.gate_proj, ..., shared_head.head})
                comp["MTP layers (dense, + head copy)"] = n_mtp * (_layer_total(dense[0], h) + 2 * h * h + 3 * h
                                                                   + V * h)
                comp["router bias buffers (main layers)"] = n_moe * E
    if p.get("dsv4_hash_layers"):
        k = next((g["moe"]["k"] for g in p["groups"] if g["moe"] is not None), 0)
        comp["hash-routing tid2eid buffers"] = p["dsv4_hash_layers"] * V * k
    if str(p["mt"]).startswith("deepseek_v41") and n_moe:
        comp["router bias buffers (e_score_correction_bias, _vl)"] = 2 * n_moe * E
        en = t.get("engram_num_embeddings")
        if isinstance(en, list) and t.get("engram_head_dim"):
            comp["engram FP8 weight_scale_inv"] = sum(int(x) for x in en) * (int(t["engram_head_dim"]) // 32)
    P = int(raw.get("n_positions") or raw.get("max_position_embeddings") or 0)
    if p["mt"] in ("gpt2", "gpt_neox", "gptj", "codegen", "gpt_neo", "gpt_bigcode") and P:
        comp["attention-mask buffers"] = n * (P * P + 1)
    rot = sum(g["attn"]["dqk"] for g in p["groups"] if g["attn"] is not None) // 2
    if rot:
        comp["rotary inv_freq buffers"] = sum(g["n"] * g["attn"]["dqk"] // 2 for g in p["groups"]
                                              if g["attn"] is not None)
    comp.update(p.get("stored_blocks") or {})              # named by a bespoke builder
    if repo in CHECKPOINT_BLOCKS:
        comp.update(CHECKPOINT_BLOCKS[repo](p))
    if non_lm:
        comp["vision / audio towers"] = int(non_lm)
    else:
        vt = vision_tower_params(raw)
        if vt:
            comp[f"vision tower ({vt[1]})"] = vt[0]
    return comp


# vision configs of non-stock wrappers whose encoder is a stock class's (the Qwen3.5 / 3.6 / 3.8 vision encoder is
# Qwen3-VL's: the same config keys, patch embed, blocks, merger and deepstack mergers)
VISION_AS = {"interns2_mobius": "qwen3_vl", "qwen3_5": "qwen3_vl", "qwen3_5_vision": "qwen3_vl", "qwen3_5_moe": "qwen3_vl",
             "qwen3_5_moe_vision": "qwen3_vl", "qwen4_exp": "qwen3_vl", "qwen4_exp_vision": "qwen3_vl"}
_VT_CACHE = {}


def vision_tower_params(raw):
    """(params, how) of the vision encoder a config's vision_config describes, counted on the meta device with the
    stock transformers class (directly, or through VISION_AS), or None."""
    v = raw.get("vision_config")
    if not isinstance(v, dict):
        return None
    vt = str(v.get("model_type") or raw.get("model_type") or "")
    key = json.dumps(v, sort_keys=True, default=str)
    if key in _VT_CACHE:
        return _VT_CACHE[key]
    out = None
    try:
        import torch
        from transformers import CONFIG_MAPPING
        lg = logging.getLogger("transformers")
        lvl = lg.level
        lg.setLevel(logging.ERROR)
        try:
            if VISION_AS.get(vt) == "qwen3_vl" or VISION_AS.get(str(raw.get("model_type"))) == "qwen3_vl":
                from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
                from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
                vv = {k: x for k, x in v.items() if k != "model_type"}
                with torch.device("meta"):
                    m = Qwen3VLVisionModel._from_config(Qwen3VLVisionConfig(**vv))
                out = (sum(x.numel() for x in m.parameters()), "Qwen3-VL vision encoder class")
            elif vt in ("qwen2_5_vl", "qwen2_vl"):
                # their vision configs carry the wrapper's model_type: CONFIG_MAPPING would build the whole VL model
                # (a default text model included), so build the vision encoder class itself
                if vt == "qwen2_5_vl":
                    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig as VC
                    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
                        Qwen2_5_VisionTransformerPretrainedModel as VM
                else:
                    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig as VC
                    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel as VM
                vv = {k: x for k, x in v.items() if k != "model_type"}
                with torch.device("meta"):
                    m = VM._from_config(VC(**vv))
                out = (sum(x.numel() for x in m.parameters()), f"{VM.__name__}")
            elif vt in CONFIG_MAPPING:
                from transformers import AutoModel
                cfg = CONFIG_MAPPING[vt](**{k: x for k, x in v.items() if k != "model_type"})
                with torch.device("meta"):
                    m = AutoModel.from_config(cfg)
                out = (sum(x.numel() for x in m.parameters()), f"stock {type(m).__name__}")
        finally:
            lg.setLevel(lvl)
    except Exception as e:                                   # a vision config the stock class cannot build
        log.debug("vision tower %s: %s", vt, e)
        out = None
    _VT_CACHE[key] = out
    return out


TOWER_MARKERS = ("vision", "visual", "audio", "speech", "image", "acoustic", "tower", "vae", "dit_", "connector",
                 "projector", "encoder", "tts", "codec", "mm_", "video", "siglip", "clip", "whisper", "depth_decoder",
                 "diffusion_head", "point", "spatial")


def has_towers(raw):
    """The checkpoint carries a non-LM part (a vision / audio / speech / point-cloud tower, a projector, a codec
    decoder) that the LM count leaves out: a config key or the class name says so."""
    keys = " ".join(str(k).lower() for k in raw)
    arch = " ".join(str(a).lower() for a in raw.get("architectures") or [])
    mt = str(raw.get("model_type") or "").lower()
    return (any(m in keys for m in TOWER_MARKERS) or any(m in arch for m in ("vision", "vl", "audio", "tts", "speech",
                                                                            "conditionalgeneration", "omni", "asr"))
            or any(m in mt for m in ("tts", "asr", "audio", "_vl", "vl_", "omni", "spatiallm", "videollama", "ovis",
                                     "cambrian", "molmo", "phi3_v", "phi4mm", "hulumed", "nextstep")))


def hub_reason(p, raw, total, hub_total, repo=None, non_lm=None):
    """(status, reason) of the Hub safetensors count against the formula: 'hub-matched' with the exact blocks that
    account for the difference, else a best explanation (quantised storage, towers we cannot count, MTP) or
    'unexplained'."""
    if not hub_total:
        return None, None
    delta = hub_total - total
    if delta == 0:
        return "hub-matched", "exact"
    comp = hub_components(p, raw, non_lm, repo)
    names = list(comp)
    import itertools
    for r in range(1, min(4, len(names)) + 1):
        for combo in itertools.combinations(names, r):
            if sum(comp[c] for c in combo) == delta:
                return "hub-matched", " + ".join(combo)
    rel = delta / total if total else 0
    if is_quantised(repo, raw):
        return "hub-explained", f"quantised checkpoint: stored elements differ from parameters ({rel:+.1%})"
    has_tower = has_towers(raw)
    if delta > 0 and has_tower and non_lm is None:
        return "hub-explained", f"vision / audio / other towers in the checkpoint, not in the LM count ({rel:+.1%})"
    if delta > 0 and "MTP layers" in comp and abs(delta - comp["MTP layers"]) / max(delta, 1) < 0.05:
        return "hub-explained", f"MTP layers ({rel:+.1%}, within 5% of the estimate)"
    if 0 < delta and rel < 1e-3:
        return "hub-explained", f"stored buffers / small extra tensors ({delta:+,d} params, {rel:+.3%})"
    return "unexplained", f"{rel:+.2%} ({delta:+,d} params)"


def metrics(raw, hub_total=None, batch_B=BATCH_B, p=None):
    """The schema's Metrics record of one raw config (without prov / valid, filled by build())."""
    p = p or plan(raw)
    prm = params(p)
    h = p["hidden"]
    out = {"model_type": p["mt"], "arch": p["arch"], "max_context": int(p["max_context"]),
           "max_context_source": p.get("max_context_source"), "max_context_unbounded": bool(p.get("max_context_unbounded")),
           "layers": _kind_counts(p), "notes": list(p["notes"])}
    prm["extra_towers"] = None
    prm["hub_total"] = hub_total
    prm["hub_delta"] = (hub_total - prm["total"]) if hub_total else None
    prm["hub_delta_reason"] = None
    prm["weight_bytes"] = {dt: _bytes(prm["total"], dt) for dt in DTYPE_BYTES}
    prm["active_weight_bytes"] = {dt: _bytes(prm["active"], dt) for dt in DTYPE_BYTES}
    out["params"] = prm
    e, hf, mn = kv_per_token(p)
    if p.get("dsa_dim"):
        e["dsa_indexer"] = p["dsa_dim"] * out["layers"]["dsa_indexer"]
        mn += e["dsa_indexer"]
    tot = sum(e.values())
    out["kv_per_token"] = {"elems": e, "elems_total": tot, "bf16": _bytes(tot, "bf16"), "fp8": _bytes(tot, "fp8")}
    keqv_two = any(g["k_eq_v"] and g["attn"] is not None and not _one_kv_tensor_hf(p, g) for g in p["groups"])
    conv, ssm = state_elems(p)
    st = conv + ssm
    out["state"] = {"elems": st, "conv_elems": conv, "ssm_elems": ssm, "bf16": 2 * st, "fp32": 4 * st}
    ctx = contexts_for(p)
    out["kv"], out["kv_materialized"] = {}, {"per_token_elems": sum(hf.values())}
    out["kv_minimum"] = {"per_token_elems": mn, "note": (
        "K = V layers counted as one tensor: the theoretical minimum; transformers, vLLM and llama.cpp allocate K "
        "and V separately (the headline)") if keqv_two else "equal to the headline"}
    out["flops"] = {"matmul": flops(p, 1, "decode")["matmul"], "prefill": {}, "decode": {}}
    out["memory"], out["bandwidth"] = {}, {}
    weights_bf16 = 2 * prm["total"]
    read_params = prm["active_non_embedding"] + p["lm_head_W"] + h          # head (tied: the table) + one row
    for key, L in ctx.items():
        k = _kv_headline(p, L)
        ke = sum(k.values())
        win = {}
        for g in p["groups"]:
            if g["window"]:
                win["sliding"] = L <= g["window"] - 1
        out["kv"][key] = {"L": L, "beyond_max": L > p["max_context"] and not p.get("max_context_unbounded"),
                          "elems": k, "in_window": win,
                          "bf16": _bytes(ke, "bf16"), "fp8": _bytes(ke, "fp8")}
        kh = sum(kv_at(p, L, "hf").values())
        out["kv_materialized"][key] = {"bf16": _bytes(kh, "bf16"), "fp8": _bytes(kh, "fp8")}
        km = sum(_kv_headline(p, L, "minimum").values())
        out["kv_minimum"][key] = {"bf16": _bytes(km, "bf16"), "fp8": _bytes(km, "fp8")}
        fp = flops(p, L, "prefill")
        fpd = flops(p, L, "prefill_dense")
        out["flops"]["prefill"][key] = {"matmul": fp["matmul"], "attn": fp["attn"], "attn_dense": fpd["attn"],
                                        "recurrent": fp["recurrent"], "total": fp["total"]}
        fd = flops(p, L, "decode")
        out["flops"]["decode"][key] = fd
        kvb = 2 * ke
        out["memory"][key] = {"weights": weights_bf16, "kv": kvb, "state": 2 * st,
                              "batch1": weights_bf16 + kvb + 2 * st,
                              "batchB": weights_bf16 + batch_B * (kvb + 2 * st)}
        kv_read = 2 * sum(_kv_headline(p, max(L - 1, 0)).values())
        wr = 2 * read_params
        b1 = wr + kv_read + 2 * 2 * st
        bB = wr + batch_B * (kv_read + 2 * 2 * st)
        out["bandwidth"][key] = {"weights": wr, "kv": kv_read, "state": 2 * 2 * st, "bytes": b1,
                                 "flops": fd["total"], "intensity": round(fd["total"] / b1, 4) if b1 else None,
                                 "batchB": {"bytes": bB, "flops": batch_B * fd["total"],
                                            "intensity": round(batch_B * fd["total"] / bB, 4) if bB else None}}
    if keqv_two:
        out["notes"].append("K = V layers: the KV counts K and V as two tensors, as transformers, vLLM and llama.cpp "
                            "allocate them; kv_minimum gives the one-tensor theoretical minimum")
    if "rec" in p.get("unknown", set()):
        out["state"] = {k: None for k in out["state"]}
        out["notes"].append("recurrent family not modelled: state and recurrent FLOPs unknown")
    return out


# ------------------------------------------------------------------------------------------------ the whole file

# stock-code parameter traces that differ from the formula for a reason that is not a formula error
PARAM_TRACE_EXPLAINED = {
    "inference-optimization/Qwen3-1.6B-A0.9B":
        "transformers 4.57's Qwen3MoeConfig reads num_experts (default 128) and ignores num_local_experts = 16; the "
        "checkpoint (written by transformers 5.8, which maps the key) has 16 experts, as the formula counts",
}
STATUS = ("symbolic-proved", "trace-verified", "hub-matched", "formula-only", "failed")
GROUPS = ("params", "active", "kv", "state", "flops", "memory", "bandwidth")
TRACE_PARAMS = os.path.join(HERE, "data", "arch_trace_params.json")
TRACE_FORWARD = os.path.join(HERE, "data", "arch_trace_forward.json")
TRACE_SYMBOLIC = os.path.join(HERE, "data", "arch_trace_symbolic.json")
META = os.path.join(HERE, "model_meta.json")
# the repos whose params check failed at the competitive evaluation, each classified (graph/data/params_triage.json)
TRIAGE = os.path.join(HERE, "data", "params_triage.json")
TRIAGE_TEXT = {"A-open": "a formula gap in graph/arch_metrics.py, not fixed yet",
               "B": "the checkpoint is not the LM the config describes, not a formula error",
               "C": "a config-reading gap outside graph/arch_metrics.py"}


def weakest(*st):
    return max(st, key=lambda x: STATUS.index(x) if x in STATUS else len(STATUS))


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def type_status(forward, symbolic):
    """{model_type: {group: status, ...}} from the forward and symbolic trace files."""
    out = {}
    for mt, rec in forward.items():
        chk = rec.get("checked", [])
        if not chk:
            continue
        eq = lambda k: all(c["equal"].get(k) for c in chk)
        core = [c["equal"].get("core") for c in chk]
        st = {"flops": "trace-verified" if eq("flops_prefill") and eq("flops_decode") and all(
                  x is not False for x in core) else "failed",
              "kv": "trace-verified" if eq("kv") else "failed",
              "state": "trace-verified" if eq("state") else "failed",
              "core_modelled": all(x is True for x in core), "configs": [c.get("repo") for c in chk]}
        if st["flops"] == "trace-verified" and any(x is None for x in core):
            st["flops"] = "formula-only"                     # projections + attention verified, the core not
            st["flops_note"] = "projections, FFN, head and attention trace-verified; the recurrent core is formula-only"
        pr = (symbolic.get(mt) or {}).get("proof")
        if pr:
            if pr.get("flops_equal") and st["flops"] == "trace-verified":
                st["flops"] = "symbolic-proved"
            if pr.get("kv_equal") and st["kv"] == "trace-verified":
                st["kv"] = "symbolic-proved"
            st["symbolic"] = {"regimes": [{k: r[k] for k in ("range", "flops_trace", "kv_trace")}
                                          for r in pr.get("regimes", [])]}
        out[mt] = st
    return out


def build(models=None, configs=None, out=OUT):
    """Every zoo repo's Metrics, the shapes (zoo node order), the per-type validation and the coverage counts."""
    from graph.frontier import load
    from graph.model_graph import ZOO_SNAPSHOT, _dedup_points, iter_json_configs
    models = models if models is not None else load()
    configs = configs if configs is not None else dict(iter_json_configs(os.path.join(HERE, "hub_10k.json")))
    meta = _load(META, {})
    ptrace = _load(TRACE_PARAMS, {})
    p_of = {r: rec for rec in ptrace.values() for r in rec.get("repos", [])}
    tstat = type_status(_load(TRACE_FORWARD, {}), _load(TRACE_SYMBOLIC, {}))
    triage = _load(TRIAGE, {}).get("repos", {})
    repos, types = {}, {}
    for m in models:
        name = m["name"]
        raw = configs[name]
        mt = str(raw.get("model_type"))
        hub = (meta.get(name) or {}).get("hf_total")
        try:
            p = plan(raw)
            rec = metrics(raw, hub, p=p)
        except Exception as e:                               # a config the formulas cannot read: say so
            log.warning("metrics failed for %s: %s", name, e)
            repos[name] = {"model_type": mt, "error": f"{type(e).__name__}: {e}"[:200]}
            continue
        stock = stock_arch(raw)
        pt = p_of.get(name)
        non_lm = pt.get("non_lm") if pt else None
        status, reason = hub_reason(p, raw, rec["params"]["total"], hub, name, non_lm if non_lm else None)
        rec["params"]["hub_delta_reason"] = reason
        if non_lm:
            rec["params"]["extra_towers"] = int(non_lm)
        valid = {}
        if stock and pt and pt.get("equal"):
            valid["params"] = "trace-verified"
        elif stock and pt and "trace" in pt and not pt.get("equal") and name in PARAM_TRACE_EXPLAINED:
            valid["params"] = "hub-matched" if status == "hub-matched" else "formula-only"
            rec["notes"].append(f"stock-code parameter count {pt['trace']:,d}: {PARAM_TRACE_EXPLAINED[name]}")
        elif stock and pt and "trace" in pt and not pt.get("equal"):
            valid["params"] = "failed"
            rec["notes"].append(f"stock-code parameter count {pt['trace']:,d} differs from the formula")
        elif status == "hub-matched":
            valid["params"] = "hub-matched"
        elif status == "unexplained" and not is_quantised(name, raw) and hub and abs(
                hub - rec["params"]["total"]) > 0.005 * rec["params"]["total"]:
            valid["params"] = "failed"
            rec["notes"].append(f"Hub safetensors count differs: {reason}")
        else:
            valid["params"] = "formula-only"
        tri = triage.get(name)
        rec["params"]["checkpoint_anomaly"] = None
        if tri and tri["bucket"] == "B" and valid["params"] in ("failed", "formula-only"):
            # the upload is not the model its config describes: the numbers are the architecture's (the formula);
            # the Hub count is not a check of them, so the status is formula-only and the anomaly is shown
            valid["params"] = "formula-only"
            rec["params"]["checkpoint_anomaly"] = tri["reason"]
            hub_s = f"{hub:,d}" if hub else "none"
            rec["notes"].append(f"checkpoint anomaly: {tri['reason']}. Every figure here is the architecture's, from "
                                f"its config; the Hub upload counts {hub_s} parameters")
        elif valid["params"] == "failed" and tri:
            rec["notes"].append(f"params check, triaged ({TRIAGE_TEXT.get(tri['bucket'], tri['bucket'])}): "
                                f"{tri['reason']}")
        ts = tstat.get(mt, {}) if stock else {}
        moe = rec["params"]["routed_experts"] > 0
        for g in ("kv", "state", "flops"):
            valid[g] = ts.get(g, "formula-only")
        if rec["layers"]["mla"] and valid["kv"] in ("trace-verified", "symbolic-proved"):
            valid["kv"] = "formula-only"
            rec["notes"].append("MLA: the latent KV (headline) is a formula; kv_materialized (transformers' per-head "
                                "K / V cache) is " + ts.get("kv", "formula-only"))
        if p.get("dsa_dim"):
            rec["notes"].append("DSA: attention over the index_topk selected keys plus the indexer over all keys")
        if ts.get("flops_note"):
            rec["notes"].append(ts["flops_note"])
        if valid["params"] == "trace-verified" and (not moe or ts.get("flops") in ("trace-verified", "symbolic-proved")):
            valid["active"] = "trace-verified"
        else:
            valid["active"] = valid["params"] if not moe else weakest(valid["params"], "formula-only")
        valid["memory"] = weakest(valid["params"], valid["kv"], valid["state"])
        valid["bandwidth"] = weakest(valid["active"], valid["kv"], valid["state"], valid["flops"])
        rec["valid"] = valid
        rec["prov"] = {g: ("trace-verified" if valid[g] in ("trace-verified", "symbolic-proved") else "code")
                       for g in GROUPS}
        rec["stock_class"] = bool(stock)
        repos[name] = rec
        t = types.setdefault(mt, {"stock": stock_arch(raw) or _stock_config_class(mt) is not None, "repos": 0,
                                  "params": {"checked": 0, "equal": 0, "mismatches": []}, "notes": []})
        t["repos"] += 1
        if pt and "trace" in pt:
            t["params"]["checked"] += 1
            t["params"]["equal"] += int(bool(pt.get("equal")) or name in PARAM_TRACE_EXPLAINED)
            if not pt.get("equal") and len(t["params"]["mismatches"]) < 10:
                t["params"]["mismatches"].append({"repo": name, "formula": rec["params"]["total"],
                                                  "trace": pt["trace"], "reason": PARAM_TRACE_EXPLAINED.get(name),
                                                  "resolved": name in PARAM_TRACE_EXPLAINED})
    for mt, t in types.items():
        ts = tstat.get(mt, {})
        t["params"]["status"] = ("trace-verified" if t["params"]["checked"] and t["params"]["equal"] ==
                                 t["params"]["checked"] else "failed" if t["params"]["checked"] else "formula-only")
        for g in ("flops", "kv", "state"):
            t[g] = {"status": ts.get(g, "formula-only")}
        t["symbolic"] = ts.get("symbolic")
        t["trace_configs"] = ts.get("configs", [])
        t["method"] = ("params: the exact config on the meta device; flops / kv / state: tiny real-shaped configs "
                       "(<= 8 layers, <= 4 query heads, head dims kept, window 8, vocab 512) at L = 5 and 13, CPU fp32 "
                       "random weights, FlopCounterMode; symbolic: torch.export with a dynamic length, per window "
                       "regime") if t["stock"] else "formula only (no stock transformers class at 4.57)"
    # shapes in zoo order
    rank = {m["name"]: k for k, m in enumerate(models)}
    P = _dedup_points(models)
    shapes = []
    for p_ in P:
        mem = [x for x in p_["members"] if x in repos and "error" not in repos[x]]
        rep = min(mem or p_["members"], key=lambda x: rank.get(x, 1 << 30))
        rng = {}
        for key, get in (("params.total", lambda r: r["params"]["total"]),
                         ("params.active", lambda r: r["params"]["active"]),
                         ("kv_per_token.bf16", lambda r: r["kv_per_token"]["bf16"]),
                         ("max_context", lambda r: r["max_context"])):
            vals = [get(repos[x]) for x in mem]
            if vals:
                rng[key] = [min(vals), max(vals)]
        shapes.append({"rep": rep, "n_members": len(p_["members"]), "metrics": repos.get(rep), "range": rng})
    coverage = {g: {s: 0 for s in STATUS} for g in GROUPS}
    for r in repos.values():
        for g in GROUPS:
            if "valid" in r:
                coverage[g][r["valid"][g]] += 1
    type_cov = {g: {s: 0 for s in STATUS} for g in ("params", "flops", "kv", "state")}
    for t in types.values():
        for g in type_cov:
            type_cov[g][t[g]["status"]] += 1
    try:
        import transformers
        tv = transformers.__version__
    except Exception:          # pragma: no cover
        tv = None
    data = {"schema_version": SCHEMA_VERSION, "generated": datetime.date.today().isoformat(), "snapshot": ZOO_SNAPSHOT,
            "transformers_version": tv, "contexts": CONTEXTS, "batch_B": BATCH_B, "dtype_bytes": DTYPE_BYTES,
            "conventions": CONVENTIONS, "coverage": {"repos": coverage, "types": type_cov},
            "types": types, "shapes": shapes, "repos": repos}
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(data, f, separators=(",", ":"), default=_json_default)
    return data


def _json_default(x):
    if isinstance(x, set):
        return sorted(x)
    if hasattr(x, "item"):
        return x.item()
    return str(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    d = build()
    print(f"wrote {OUT}: {len(d['repos'])} repos, {len(d['shapes'])} shapes, {len(d['types'])} model types")
    for g, c in d["coverage"]["repos"].items():
        print(f"  {g:10}", {k: v for k, v in c.items() if v})
