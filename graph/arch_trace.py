"""Trace checks of graph/arch_metrics against the real (stock) transformers architecture code.

  params   every repo whose model_type has a stock transformers class: the model is instantiated from its exact
           config on the `meta` device (no weights, no download) and its LM parameters counted; they must equal
           arch_metrics.params exactly.
  flops    a tiny real-shaped config per (model type, config shape): real head dims, KV heads, MLA / window /
           expert structure, few layers of each kind, small vocab / FFN widths; a random-weight CPU forward under
           torch.utils.flop_counter.FlopCounterMode at several L (inside and beyond the window) must equal
           arch_metrics.flops(plan(tiny config)) in the `dense` attention convention (prefill) and in decode.
  kv       the same forward's cache object (DynamicCache / its sliding layers / the hybrid caches): elements held
           after L tokens must equal arch_metrics.kv_at(plan, L, "hf"); the recurrent state the cache holds must
           equal arch_metrics.state_elems.

Remote code is never executed: only model types with a stock class are traced, and the config goes through the
stock config class.

Run:  python -m graph.arch_trace params        # writes graph/data/arch_trace_params.json
      python -m graph.arch_trace forward       # writes graph/data/arch_trace_forward.json
"""
from __future__ import annotations

import copy
import json
import tempfile
import logging
import os
import sys
import time
import warnings

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_OUT = os.path.join(HERE, "data", "arch_trace_params.json")
FORWARD_OUT = os.path.join(HERE, "data", "arch_trace_forward.json")

# parameters of a multimodal checkpoint that are not the text LM (vision / audio towers, projectors)
NON_LM_MARKERS = ("vision", "visual", "multi_modal_projector", "mm_projector", "image_newline", "audio",
                  "connector", "patch_embed", "merger", "vqmodel", "img_", "depth_decoder", "image_")
# keys that never change the architecture (dedup of the configs to trace)
IGNORED_KEYS = {"_name_or_path", "transformers_version", "torch_dtype", "dtype", "bos_token_id", "eos_token_id",
                "pad_token_id", "use_cache", "rope_theta", "initializer_range", "task_specific_params",
                "quantization_config", "auto_map", "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon",
                "attention_dropout", "hidden_dropout", "resid_pdrop", "embd_pdrop", "attn_pdrop", "summary_type",
                "rope_scaling", "max_position_embeddings_comment", "_attn_implementation_autoset"}


def _quiet():
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)


def stock_types():
    from transformers import CONFIG_MAPPING
    return set(CONFIG_MAPPING.keys())


def stock_items(models=None, configs=None):
    """[(repo, raw config)] of the zoo models whose model_type has a stock transformers class."""
    from graph.frontier import load
    from graph.model_graph import iter_json_configs
    models = models if models is not None else load()
    configs = configs if configs is not None else dict(iter_json_configs(os.path.join(HERE, "hub_10k.json")))
    st = stock_types()
    return [(m["name"], configs[m["name"]]) for m in models if configs.get(m["name"], {}).get("model_type") in st]


def dedup_key(raw):
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in sorted(x.items()) if k not in IGNORED_KEYS and "dropout" not in k}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip(raw), sort_keys=True, default=str)


def model_class(raw, hc):
    import transformers
    from transformers import AutoModelForCausalLM
    arch = (raw.get("architectures") or [None])[0]
    names = {n.lower(): n for n in dir(transformers)}
    cls = getattr(transformers, names[arch.lower()], None) if isinstance(arch, str) and arch.lower() in names else None
    if cls is None or not hasattr(cls, "_from_config"):          # the Auto mapping for the config class
        from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
        cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(hc)]
    return cls


def build_meta(raw):
    """The stock model of a raw config on the meta device."""
    import torch
    from graph.arch_metrics import _stock_config_class
    _quiet()
    hc = _stock_config_class(raw["model_type"]).from_dict(copy.deepcopy(raw))
    cls = model_class(raw, hc)
    try:
        with torch.device("meta"):
            return cls._from_config(hc)
    except Exception as e:
        if "meta" not in str(e):
            raise
    # a module whose __init__ reads a tensor value (Apertus XIELU: .item()): parameters on meta, buffers on CPU
    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
        m = cls._from_config(hc)
    m.tie_weights()                                  # (from_pretrained ties; _from_config here does not)
    return m


def lm_param_split(model):
    """(LM params, non-LM params, {top-level name: count} of the non-LM part)."""
    lm, other, parts = 0, 0, {}
    for name, prm in model.named_parameters():          # tied parameters are yielded once
        if any(mk in name.lower() for mk in NON_LM_MARKERS):
            other += prm.numel()
            key = ".".join(name.split(".")[:2])
            parts[key] = parts.get(key, 0) + prm.numel()
        else:
            lm += prm.numel()
    return lm, other, parts


def check_params(raw):
    from graph.arch_metrics import params, plan
    t = time.time()
    m = build_meta(raw)
    lm, other, parts = lm_param_split(m)
    f = params(plan(raw))
    return {"trace": lm, "formula": f["total"], "non_lm": other, "non_lm_parts": parts, "equal": lm == f["total"],
            "seconds": round(time.time() - t, 2), "class": type(m).__name__}


def run_params(items=None, out=PARAMS_OUT, only_types=None, limit=None):
    items = items if items is not None else stock_items()
    groups = {}
    for repo, raw in items:
        if only_types and raw["model_type"] not in only_types:
            continue
        groups.setdefault(dedup_key(raw), []).append((repo, raw))
    res = {}
    for k, (key, reps) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0][1]["model_type"])):
        if limit and k >= limit:
            break
        repo, raw = reps[0]
        from graph.arch_metrics import stock_arch
        if not stock_arch(raw):                           # the class is remote code: no stock reference
            r = {"nonstock": (raw.get("architectures") or [None])[0]}
        else:
            try:
                r = check_params(raw)
            except Exception as e:                        # a config the stock code cannot build
                r = {"error": f"{type(e).__name__}: {e}"[:300]}
        r["model_type"] = raw["model_type"]
        r["repos"] = [x for x, _ in reps]
        res[repo] = r
        flag = "ok " if r.get("equal") else "ERR" if "error" in r else "NONSTOCK" if "nonstock" in r else "DIFF"
        print(f"{flag} {raw['model_type']:22} {repo:60} trace {r.get('trace')} formula {r.get('formula')} "
              f"{r.get('error', '')}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    return res


# ------------------------------------------------------------------------------------------------ forward traces

TINY_WINDOW = 8                     # every sliding / chunked window of a tiny config
TINY_L = (5, 13)                    # one L inside the window (L < W - 1), one beyond it
TINY_VOCAB = 512
_LAYER_LIST_KEYS = ("layer_types", "layers_block_type", "block_types", "_block_types", "attn_type_list",
                    "mlp_layer_types", "moe_layer_freq", "no_rope_layers", "layers_attention_type")
_LAYER_INDEX_KEYS = ("attn_layer_indices", "full_attn_idxs", "moe_layers", "mlp_only_layers", "global_attn_idx",
                     "full_attention_layers")
_WINDOW_KEYS = ("sliding_window", "attention_chunk_size", "attention_window_size", "window_size")
_TOKEN_KEYS = ("bos_token_id", "eos_token_id", "pad_token_id", "image_token_id", "image_token_index",
               "video_token_id", "vision_start_token_id", "vision_end_token_id", "decoder_start_token_id")


def _shrink_text(c, n_small):
    """Shrink one (text) config dict in place: few layers (per-layer lists truncated, index lists filtered), small
    vocab / FFN widths, at most 4 query heads (head dims kept), windows of TINY_WINDOW."""
    mt = str(c.get("model_type") or "")
    n = None
    for k in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if isinstance(c.get(k), int):
            n = c[k]
            c[k] = min(n, n_small)
    ns = min(n or n_small, n_small)
    for k in _LAYER_LIST_KEYS:
        if isinstance(c.get(k), list) and n and len(c[k]) == n:
            c[k] = c[k][:ns]
    for k in _LAYER_INDEX_KEYS:
        if isinstance(c.get(k), list):
            had = bool(c[k])
            c[k] = [i for i in c[k] if isinstance(i, int) and i < ns]
            if had and not c[k] and k != "mlp_only_layers":
                c[k] = [ns - 1]                              # keep one layer of the listed kind
    for k in ("layers_block_type",):                         # keep one attention / hybrid layer
        v = c.get(k)
        if isinstance(v, list) and v and n and len(v) < n:
            pass
        if isinstance(v, list) and v and all(x == "mamba" for x in v):
            full = raw_kinds = None
            c[k] = v[:-1] + [("hybrid" if mt in ("zamba", "zamba2") else "attention")]
    if mt == "gpt_neo" and isinstance(c.get("attention_types"), list):
        kinds = [k for pat, rep in c["attention_types"] for _ in range(rep) for k in pat][:ns]
        c["attention_types"] = [[[k], 1] for k in kinds]
        c.pop("attention_layers", None)
    for k in ("torch_dtype", "dtype"):
        c.pop(k, None)
    if isinstance(c.get("hybrid_override_pattern"), str):
        c["hybrid_override_pattern"] = c["hybrid_override_pattern"][:ns]
    for k in ("vocab_size", "n_vocab"):
        if isinstance(c.get(k), int):
            c[k] = TINY_VOCAB
    for k in _TOKEN_KEYS:
        v = c.get(k)
        if isinstance(v, int) and v >= TINY_VOCAB:
            c[k] = 1
        elif isinstance(v, list):
            c[k] = [x if not isinstance(x, int) or x < TINY_VOCAB else 1 for x in v]
    for k in list(c):
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, int):
            continue
        if (("intermediate_size" in k and "mamba" not in k) or k in ("ffn_hidden_size", "dff", "n_inner", "ffn_dim",
                                                                     "block_ff_dim",
                                                                     "moe_intermediate_size", "shared_intermediate_size")):
            if v > 64:
                c[k] = 64 if k != "ffn_dim" else 128
    for k in _WINDOW_KEYS:
        if isinstance(c.get(k), int) and not isinstance(c.get(k), bool) and c[k] > TINY_WINDOW:
            c[k] = TINY_WINDOW
    _shrink_width(c)
    return c


TINY_HIDDEN = 256
TINY_HEADS = 4


def _has_head_dim_attr(mt):
    from graph.arch_metrics import _stock_config_class
    cls = _stock_config_class(mt)
    try:
        return cls is not None and "head_dim" in cls().to_dict()
    except Exception:
        return False


def _shrink_width(c):
    """Hidden width TINY_HIDDEN and at most TINY_HEADS query heads, head dims kept (the KV / attention per head is the
    real one); the recurrent widths follow the hidden width, their per-head dims kept."""
    from graph.arch_metrics import HEAD_DIM_FROM_HIDDEN
    mt = str(c.get("model_type") or "")
    hk = next((k for k in ("hidden_size", "n_embd", "d_model") if isinstance(c.get(k), int)), None)
    nk = next((k for k in ("num_attention_heads", "n_head", "n_heads", "num_heads") if isinstance(c.get(k), int)), None)
    if hk is None:
        return
    h = c[hk]
    nh = c.get(nk) if nk else None
    kvk = next((k for k in ("num_key_value_heads", "num_kv_heads", "n_head_kv") if isinstance(c.get(k), int)), None)
    nkv = c.get(kvk) if kvk else nh
    if "use_mamba_kernels" in c or mt in ("jamba", "zamba"):
        c["use_mamba_kernels"] = False                      # the CUDA kernels: CPU runs the torch path
    if mt in ("mpt", "rwkv", "mamba", "falcon_mamba", "zamba", "zamba2", "moshi"):
        return                                              # (the families with their own width rules: kept real)
    if nh:
        hd = c.get("head_dim") if isinstance(c.get("head_dim"), int) else h // nh
        group = max(1, nh // max(nkv or nh, 1))
        nkv_s = max(1, min(nkv or nh, max(1, TINY_HEADS // group)))
        nh_s = nkv_s * min(group, TINY_HEADS)
        derived = (mt in HEAD_DIM_FROM_HIDDEN or mt in ("recurrent_gemma", "helium")
                   or not (_has_head_dim_attr(mt) or isinstance(c.get("head_dim"), int)))
        if c.get("kv_lora_rank"):
            derived = False
        h_s = nh_s * hd if derived else TINY_HIDDEN
        if h_s >= h:
            return
        c[hk] = h_s
        c[nk] = nh_s
        if kvk:
            c[kvk] = nkv_s
        if not derived and not c.get("kv_lora_rank"):
            c["head_dim"] = hd
        if mt == "falcon" and not c.get("new_decoder_architecture"):
            c["num_kv_heads"] = nh_s if not c.get("multi_query") else 1
    else:
        h_s = TINY_HIDDEN
        if h_s >= h:
            return
        c[hk] = h_s
    # recurrent widths
    if isinstance(c.get("mamba_d_head"), int) and (isinstance(c.get("mamba_n_heads"), int)):
        P = c["mamba_d_head"]
        I = int(c.get("mamba_expand", 2) * h_s)
        c["mamba_n_heads"] = max(1, I // P)
        if c.get("mamba_d_ssm"):
            c["mamba_d_ssm"] = c["mamba_n_heads"] * P
        if isinstance(c.get("mamba_n_groups"), int):
            g = c["mamba_n_groups"]
            while c["mamba_n_heads"] % g:
                g -= 1
            c["mamba_n_groups"] = g
    if isinstance(c.get("linear_num_value_heads"), int):
        hv, hkk = c["linear_num_value_heads"], c.get("linear_num_key_heads", c["linear_num_value_heads"])
        r = max(1, hv // max(hkk, 1))
        c["linear_num_key_heads"] = 2
        c["linear_num_value_heads"] = 2 * r
    if isinstance(c.get("lru_width"), int):
        c["lru_width"] = h_s
    if isinstance(c.get("attention_hidden_size"), int) and mt == "rwkv":
        c["attention_hidden_size"] = h_s


def tiny_config(raw, n_small=8):
    """A tiny real-shaped config of the same model type (see the module docstring); the formula is evaluated on it."""
    c = copy.deepcopy(raw)
    for k in ("quantization_config", "auto_map"):
        c.pop(k, None)
    targets = [c]
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(c.get(k), dict):
            targets.append(c[k])
    for t in targets:
        _shrink_text(t, n_small)
    _shrink_vision(c)
    for v in c.values():
        if isinstance(v, dict):
            for k in ("torch_dtype", "dtype"):
                v.pop(k, None)
    return c


def _shrink_vision(c):
    v = c.get("vision_config")
    if isinstance(v, dict):
        for k in ("num_hidden_layers", "depth"):
            if isinstance(v.get(k), int):
                v[k] = 1


def _collect(obj, out, path="", seen=None):
    """Tensor elements held by a cache object, by attribute path."""
    import torch
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        out[path] = out.get(path, 0) + obj.numel()
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            _collect(x, out, f"{path}[]", seen)
    elif isinstance(obj, dict):
        for k, x in obj.items():
            _collect(x, out, f"{path}.{k}", seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, x in vars(obj).items():
            if k.startswith("_") or k in ("config",):
                continue
            _collect(x, out, f"{path}.{k}", seen)


def cache_elems(out_obj, model=None):
    """{'kv': elements, 'conv': ..., 'ssm': ..., 'paths': {...}} from a forward's output (its cache)."""
    paths = {}
    for attr in ("past_key_values", "cache_params", "state"):
        v = getattr(out_obj, attr, None)
        if v is not None:
            _collect(v, paths, attr)
    if model is not None and not paths:                      # RecurrentGemma keeps its caches on the modules
        for name, mod in model.named_modules():
            for a in ("key_states", "value_states", "conv1d_state", "recurrent_states"):
                t = getattr(mod, a, None)
                if t is not None and hasattr(t, "numel"):
                    paths[f"{name}.{a}"] = t.numel()
    kv = conv = ssm = 0
    for pth, n in paths.items():
        last = [x for x in pth.replace("[]", "").split(".") if x][-1].lower()   # the holding attribute
        if "conv" in last:
            conv += n
        elif (any(x in last for x in ("key", "value")) or last in ("keys", "values")) and "recurrent" not in last:
            kv += n
        else:
            ssm += n
    return {"kv": kv, "conv": conv, "ssm": ssm, "paths": paths}


REC_MODULE_MARKERS = ("Mixer", "GatedDeltaNet", "ShortConv", "RecurrentBlock", "LightningAttention",
                      "RwkvSelfAttention", "MambaLayer")
ROPE_MODULE_MARKERS = ("Rotary",)


def _module_flops(fc, model):
    """(total, rope, recurrent core) FLOPs of a FlopCounterMode run: rope = inside rotary-embedding modules (RoPE is
    not counted by the convention), core = inside recurrent mixers minus their nn.Linear children."""
    import torch
    counts = {k: sum(v.values()) for k, v in fc.get_flop_counts().items()}
    total = counts.get("Global", 0)
    root = type(model).__name__
    mods = dict(model.named_modules())
    rope = core = 0
    for name, mod in mods.items():
        key = f"{root}.{name}" if name else root
        cname = type(mod).__name__
        if any(m in cname for m in ROPE_MODULE_MARKERS):
            parent_rope = any(any(m in type(mods[p]).__name__ for m in ROPE_MODULE_MARKERS)
                              for p in _parents(name) if p in mods)
            if not parent_rope:
                rope += counts.get(key, 0)
        if any(m in cname for m in REC_MODULE_MARKERS):
            c = counts.get(key, 0)
            for cn, child in mod.named_modules():
                if cn and isinstance(child, torch.nn.Linear):
                    c -= counts.get(f"{key}.{cn}", 0)
            core += c
    return total, rope, core


def _parents(name):
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def build_cpu(raw):
    import torch
    from graph.arch_metrics import _stock_config_class
    _quiet()
    torch.manual_seed(0)
    hc = _stock_config_class(raw["model_type"]).from_dict(copy.deepcopy(raw))
    hc._attn_implementation = "eager"
    for sub in ("text_config", "vision_config"):
        if hasattr(hc, sub) and getattr(hc, sub) is not None:
            getattr(hc, sub)._attn_implementation = "eager"
    cls = model_class(raw, hc)
    if raw["model_type"] == "moshi":
        # Moshi's text forward is its temporal transformer, MoshiForCausalLM (the conditional-generation wrapper needs
        # audio codes); the depth decoder is outside the formula
        from transformers import MoshiForCausalLM
        cls = MoshiForCausalLM
    ns = sys.modules.get(cls.__module__)
    for attr in ("FusedRMSNormGated",):              # fla Triton modules: GPU-only, use the torch module
        if ns is not None and getattr(ns, attr, None) is not None:
            setattr(ns, attr, None)
    m = cls._from_config(hc, torch_dtype=torch.float32)
    m.eval()
    _torch_paths(m)
    return m


def _torch_paths(model):
    """Run the reference torch paths of the gated-DeltaNet layers (the fla / causal-conv1d kernels are GPU-only)."""
    for mod in model.modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            ns = sys.modules[type(mod).__module__]
            mod.chunk_gated_delta_rule = ns.torch_chunk_gated_delta_rule
            mod.recurrent_gated_delta_rule = ns.torch_recurrent_gated_delta_rule
            mod.causal_conv1d_fn = None
            mod.causal_conv1d_update = ns.torch_causal_conv1d_update


def _cpu_sdpa():
    """FlopCounterMode counts the CUDA SDPA kernels but not the CPU flash path a model calling
    F.scaled_dot_product_attention directly takes (RecurrentGemma): the same formula (QK^T + AV, no causal saving)."""
    import torch
    from torch.utils.flop_counter import sdpa_flop_count

    def cpu_flash(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs):
        return sdpa_flop_count(query_shape, key_shape, value_shape)
    return {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: cpu_flash}


HYBRID_CACHES = ("HybridMambaAttentionDynamicCache", "FalconHybridMambaAttentionDynamicCache",
                 "ZambaHybridDynamicCache")


def _hybrid_cache(model):
    """The hybrid cache a Mamba-hybrid model needs the caller to build (Bamba / Jamba / Granite-4 / Falcon-H1 /
    Zamba: 'requires an initialized ... cache to return a cache'), or None."""
    import inspect
    import torch
    mod = sys.modules.get(type(model).__module__)
    for name in HYBRID_CACHES:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        sig = inspect.signature(cls.__init__).parameters
        kw = {"dtype": torch.float32}
        if "device" in sig:
            kw["device"] = "cpu"
        if "devices" in sig:
            kw["devices"] = ["cpu"] * model.config.num_hidden_layers
        return cls(model.config, 1, **kw)
    return None


def run_forward(model, L, vocab=TINY_VOCAB):
    """Prefill L tokens, then decode token L + 1: FLOPs (total, rope, core) of each, and the cache after each."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    g = torch.Generator().manual_seed(L)
    ids = torch.randint(2, vocab - 1, (1, L), generator=g)
    kw0 = {}
    cache = _hybrid_cache(model)
    if cache is not None:
        kw0["past_key_values"] = cache
    with torch.no_grad():
        with FlopCounterMode(display=False, custom_mapping=_cpu_sdpa()) as fc:
            out = model(input_ids=ids, use_cache=True, **kw0)
        pre = _module_flops(fc, model)
        c_pre = cache_elems(out, model)
        kw = {}
        if getattr(out, "past_key_values", None) is not None:
            kw["past_key_values"] = out.past_key_values
        if getattr(out, "cache_params", None) is not None:
            kw["cache_params"] = out.cache_params
        if getattr(out, "state", None) is not None:
            kw["state"] = out.state
        nxt = torch.randint(2, vocab - 1, (1, 1), generator=g)
        with FlopCounterMode(display=False, custom_mapping=_cpu_sdpa()) as fc:
            out2 = model(input_ids=nxt, use_cache=True, cache_position=torch.tensor([L]), **kw)
        dec = _module_flops(fc, model)
        c_dec = cache_elems(out2, model)
    return {"prefill": pre, "decode": dec, "cache_prefill": c_pre, "cache_decode": c_dec}


def check_forward(raw, Ls=TINY_L):
    """Tiny-config forward traces vs the formula, at each L (prefill: dense attention convention, total over the L
    tokens; decode: the step producing token L + 1). Returns the per-L record and `equal` flags."""
    from graph.arch_metrics import code_matmul, flops, hf_alloc, plan, trace_attn, trace_core
    t0 = time.time()
    tiny = tiny_config(raw)
    p = plan(tiny)
    m = build_cpu(tiny)
    res = {"tiny": {k: tiny.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                            "num_key_value_heads", "head_dim", "vocab_size")}, "L": {}}
    ok = {"flops_prefill": True, "flops_decode": True, "kv": True, "state": True, "core": True}
    mm = code_matmul(p)
    for L in Ls:
        tr = run_forward(m, L)
        f_pre = flops(p, L, "prefill_dense")
        f_dec = flops(p, L + 1, "decode")
        tot, rope, core = tr["prefill"]
        want_pre = L * mm + trace_attn(p, L, "prefill")
        got_pre = tot - rope - core
        dtot, drope, dcore = tr["decode"]
        want_dec = mm + trace_attn(p, L + 1, "decode")
        got_dec = dtot - drope - dcore
        kv_want, conv_w, ssm_w = hf_alloc(p, L)
        tc_pre, tc_dec = trace_core(p, L, "prefill"), trace_core(p, L + 1, "decode")
        cp = tr["cache_prefill"]
        rec = {"prefill": {"trace": got_pre, "formula": want_pre, "rope": rope, "core_trace": core,
                           "core_formula": tc_pre, "core_convention": L * f_pre["recurrent"]},
               "decode": {"trace": got_dec, "formula": want_dec, "rope": drope, "core_trace": dcore,
                          "core_formula": tc_dec, "core_convention": f_dec["recurrent"]},
               "kv": {"trace": cp["kv"], "formula": kv_want},
               "state": {"trace": cp["conv"] + cp["ssm"], "formula": conv_w + ssm_w, "conv_trace": cp["conv"],
                         "ssm_trace": cp["ssm"], "conv_formula": conv_w, "ssm_formula": ssm_w}}
        ok["flops_prefill"] &= got_pre == want_pre
        ok["flops_decode"] &= got_dec == want_dec
        ok["kv"] &= cp["kv"] == kv_want
        ok["state"] &= (cp["conv"] + cp["ssm"]) == (conv_w + ssm_w)
        if tc_pre is None:
            ok["core"] = None if (core == 0 and dcore == 0) is False else ok["core"]
        else:
            ok["core"] = bool(ok["core"]) and core == tc_pre and dcore == tc_dec
        res["L"][L] = rec
    res["equal"] = ok
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------------------------------------ symbolic proofs

def _sym(x):
    import sympy
    return sympy.expand(x.node.expr if hasattr(x, "node") else sympy.Integer(int(x)))


def symbolic_prefill(model, vocab=TINY_VOCAB, example_len=17):
    """Export a prefill with a symbolic sequence length (torch.export, Dim.DYNAMIC; core ATen after decomposition)
    and read off, as sympy expressions in that length: the matmul FLOPs of every node FlopCounterMode would count
    (minus those inside rotary-embedding modules, the convention), and the elements of every cache tensor returned.
    Returns (flops, rope, kv, symbol)."""
    import torch
    import torch.fx
    from torch.export import Dim, export
    from torch.utils.flop_counter import flop_registry

    class Prefill(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):
            out = self.m(input_ids=ids, use_cache=True)
            pkv = out.past_key_values
            return (out.logits, *[t for lay in pkv.layers for t in (lay.keys, lay.values)])

    ids = torch.randint(2, vocab - 1, (1, example_len))
    ep = export(Prefill(model), (ids,), dynamic_shapes={"ids": {1: Dim.DYNAMIC}}, strict=False)
    ep = ep.run_decompositions()
    val = lambda a: a.meta.get("val") if isinstance(a, torch.fx.Node) else a
    custom = _cpu_sdpa()
    reg = {**flop_registry}
    tot = rope = 0
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        pk = getattr(node.target, "overloadpacket", None)
        fn = reg.get(pk)
        if fn is None:
            continue
        f = fn(*torch.fx.node.map_arg(node.args, val), out_val=node.meta.get("val"),
               **torch.fx.node.map_arg(node.kwargs, val))
        stack = node.meta.get("nn_module_stack", {})
        if any(any(m in str(v[1]) for m in ROPE_MODULE_MARKERS) for v in stack.values()):
            rope = rope + f
        else:
            tot = tot + f
    outs = [n for n in ep.graph.nodes if n.op == "output"][0].args[0]
    kv = sum(o.meta["val"].numel() for o in outs[1:])
    syms = set()
    for x in (tot, kv):
        e = _sym(x)
        syms |= e.free_symbols
    if len(syms) > 1:
        raise ValueError(f"expected one symbol, got {syms}")
    sym = syms.pop() if syms else None
    rng = None
    for k, v in ep.range_constraints.items():               # the lengths the exported graph is valid for
        if sym is not None and str(k) == str(sym):
            rng = (int(v.lower), None if str(v.upper) in ("oo", "int_oo") else int(v.upper))
    return _sym(tot), _sym(rope), _sym(kv), sym, rng


def check_symbolic(raw):
    """The symbolic proof for one config's model type: the exported prefill's FLOP and KV-cache polynomials equal
    the closed forms (arch_metrics at a sympy symbol L) identically."""
    import sympy
    from graph.arch_metrics import code_matmul, hf_alloc, plan, trace_attn
    t0 = time.time()
    tiny = tiny_config(raw)
    p = plan(tiny)
    m = build_cpu(tiny)
    W = min([g["window"] for g in p["groups"] if g.get("window")] or [None]) if any(
        g.get("window") for g in p["groups"]) else None
    examples = (3, 17) if W else (17,)                      # one length per regime of the window cap
    out = {"regimes": [], "flops_equal": True, "kv_equal": True}
    for ex in examples:
        f_tr, rope, kv_tr, s, rng = symbolic_prefill(m, example_len=ex)
        L = s if s is not None else sympy.Symbol("L", positive=True, integer=True)
        mm = code_matmul(p)
        f_formula = sympy.expand(L * mm + trace_attn(p, L, "prefill"))
        kv_formula = sympy.expand(hf_alloc(p, L)[0])
        lo, hi = rng if rng else (None, None)

        def resolve_min(e):                                  # Min(c, L) inside the regime the graph holds for
            def mn(*a):
                consts = [x for x in a if x.is_number]
                if consts and lo is not None and lo >= max(consts):
                    return min(consts)
                if consts and hi is not None and hi <= min(consts):
                    return L
                return sympy.Min(*a)
            return e.replace(sympy.Min, mn)
        f_formula, kv_formula = resolve_min(f_formula), resolve_min(kv_formula)
        fe = bool(sympy.simplify(f_tr - f_formula) == 0)
        ke = bool(sympy.simplify(kv_tr - kv_formula) == 0)
        out["regimes"].append({"range": [lo, hi], "flops_trace": str(f_tr), "flops_formula": str(f_formula),
                               "kv_trace": str(kv_tr), "kv_formula": str(kv_formula), "rope": str(rope),
                               "flops_equal": fe, "kv_equal": ke})
        out["flops_equal"] &= fe
        out["kv_equal"] &= ke
    out["seconds"] = round(time.time() - t0, 1)
    return out


SYMBOLIC_OUT = os.path.join(HERE, "data", "arch_trace_symbolic.json")


def _by_type(items):
    """{model_type: [(repo, raw), ...]} of the stock-class repos, one per distinct config, most downloaded first."""
    from graph.arch_metrics import stock_arch
    seen, out = set(), {}
    for repo, raw in items:
        if not stock_arch(raw):
            continue
        k = dedup_key(raw)
        if k in seen:
            continue
        seen.add(k)
        out.setdefault(raw["model_type"], []).append((repo, raw))
    return out


def run_forward_all(items=None, out=FORWARD_OUT, per_type=2, tries=6, only_types=None):
    """Tiny-config forward traces for every stock model type: up to `per_type` configs (the most downloaded that
    build and run), each at TINY_L. Writes {model_type: {"checked": [...], "failed": [...]}}."""
    items = items if items is not None else stock_items()
    res = {}
    for mt, reps in sorted(_by_type(items).items()):
        if only_types and mt not in only_types:
            continue
        rec = {"checked": [], "errors": []}
        for repo, raw in reps[:tries]:
            if len(rec["checked"]) >= per_type:
                break
            try:
                r = check_forward(raw)
                r["repo"] = repo
                rec["checked"].append(r)
                print(mt, repo, r["equal"], flush=True)
            except Exception as e:
                rec["errors"].append({"repo": repo, "error": f"{type(e).__name__}: {e}"[:300]})
                print(mt, repo, "ERR", str(e)[:120], flush=True)
        res[mt] = rec
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    return res


def run_symbolic_all(items=None, out=SYMBOLIC_OUT, tries=3, only_types=None):
    """The symbolic proof for every stock model type whose prefill exports with a symbolic length (MoE routing and
    per-token recurrent loops are data dependent and do not)."""
    items = items if items is not None else stock_items()
    res = {}
    for mt, reps in sorted(_by_type(items).items()):
        if only_types and mt not in only_types:
            continue
        rec = {"errors": []}
        for repo, raw in reps[:tries]:
            try:
                r = check_symbolic(raw)
                r["repo"] = repo
                rec["proof"] = r
                print(mt, repo, r["flops_equal"], r["kv_equal"], flush=True)
                break
            except Exception as e:
                rec["errors"].append({"repo": repo, "error": f"{type(e).__name__}: {e}"[:300]})
                print(mt, repo, "ERR", str(e)[:120], flush=True)
        res[mt] = rec
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "params"
    types = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
    fn, path = {"params": (run_params, PARAMS_OUT), "forward": (run_forward_all, FORWARD_OUT),
                "symbolic": (run_symbolic_all, SYMBOLIC_OUT)}[what]
    if not types:
        fn(out=path)
    else:                                   # re-run some model types and merge them into the existing file
        tmp = os.path.join(tempfile.mkdtemp(), f"arch_trace_{what}_subset.json")
        new = fn(only_types=types, out=tmp)
        try:
            with open(path) as f:
                cur = json.load(f)
        except FileNotFoundError:
            cur = {}
        if what == "params":                # keyed by repo: drop the re-run types' old records first
            cur = {k: v for k, v in cur.items() if v.get("model_type") not in types}
        cur.update(new)
        with open(path, "w") as f:
            json.dump(cur, f, indent=1, sort_keys=True, default=str)
