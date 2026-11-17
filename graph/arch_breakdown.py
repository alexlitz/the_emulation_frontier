"""Detailed numbers per model, from the same PLAN as graph/arch_metrics.py: parameters by component and by layer kind,
FLOPs per token by component (prefill at L, decode at position L, the training approximation), memory (weights at
fp32 / bf16 / fp8 / int8 / int4 / nf4 and the llama.cpp GGUF levels, KV at bf16 / fp8 / q8_0 / q4_0, recurrent
state, peak prefill activations, total serving memory), bytes read per decode step, and roofline upper bounds on
common hardware. The schema is graph/arch_metrics_schema.md (section "arch_breakdown.json").

Everything is a fold over the plan's groups (arch_metrics.plan): one layer of a group is a list of TENSORS
(role, kind, in, out, experts, elements), and every number below sums those. The plan's own totals are the check:
the components sum to arch_metrics.params / flops / kv exactly (breakdown_checks, run on every config by the tests).

  roles      embedding (token table), embedding.other (position / per-layer / n-gram tables), lm_head, final_norm,
             other (stray params outside the layers), other.linear (linear weights outside the layers);
             attn.{q, k, v, o, gate, q_a, q_b, kv_a, kv_b, experts, kv_proj, router, extra, bias, qk_norm, lora_norm,
             block, misc}; mlp.{gate, up, down, linear, receptance, bias, misc}; moe.{router, routed.gate, routed.up,
             routed.down, shared.gate, shared.up, shared.down, latent, shared_gate, bias}; rec.{linear, misc}; norm
  kinds      linear (a weight matrix applied per token), lump (linear weights the plan holds as one number: the
             recurrent mixers, a builder's override, the DSA indexer; the shape is taken as hidden -> elements /
             hidden where a quantiser needs one), vector (biases, norms, scales: never matmul'd), embed (a table:
             one row read per token)

Run:  python -m graph.arch_breakdown            # writes graph/data/arch_breakdown.json.gz
"""
from __future__ import annotations

import datetime
import gzip
import json
import logging
import math
import os

from graph import arch_metrics as AM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "arch_breakdown.json.gz")
SCHEMA_VERSION = 1
CONTEXTS = {"1k": 1024, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}
BATCHES = (1, 8, 32)
ROOFLINE_L = ("4k", "32k", "max")                  # the contexts the JSON stores roofline bounds at (the page: any)
# KV-cache element formats: (bytes per block, elements per block). q8_0 / q4_0 are llama.cpp's KV cache types
# (--cache-type-k / -v), with the block layout of graph/data/llama_cpp_quant.json (32 elements, an fp16 scale)
KV_FORMATS = {"bf16": (2, 1), "fp8": (1, 1), "q8_0": (34, 32), "q4_0": (18, 32)}
# safetensors-style weight formats (llama.cpp's GGUF levels are added from graph/gguf_quant.py)
WEIGHT_FORMATS = ("fp32", "bf16", "fp8", "int8", "int4", "nf4", "nvfp4")
GGUF_LEVELS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")
CONVENTIONS = {
    "params": "per role, summed over the plan's groups (weight-distinct layers np); the roles sum to "
              "arch_metrics params.total exactly",
    "flops": "2 x multiply-accumulates of matrix products per token (FlopCounterMode convention): each linear role "
             "x the unrolled layer count n (MoE: the k routed experts of E), attention scores (QK^T) and AV per layer "
             "kind, the recurrent core, the LM head; decode = the step producing token L, prefill = amortised over "
             "an L-token prefill (causal: a query at i scores min(i, W) keys)",
    "elementwise": "not in any total: softmax 5 FLOPs per score (max, subtract, exp, sum, divide), RMSNorm 4 per "
                   "element (square, sum, scale, weight), LayerNorm 7 (mean, subtract, square, sum, scale, weight, "
                   "bias), counted from the norm weights (weightless norms are not counted), activation 4 per FFN "
                   "element (+1 for the GLU product)",
    "training": "approximation: forward + backward = 3 x the forward matmul and attention FLOPs of an L-token "
                "sequence (backward = 2 x forward); no recomputation, optimizer or communication",
    "weights": "fp32 4 B and bf16 2 B per parameter; fp8 = 1 B per element of every linear matrix in the layers "
               "(router and LM head excluded) + an fp32 scale per 128 x 128 block (DeepSeek-V3 / Qwen3-FP8 "
               "weight_block_size), the rest bf16; int8 = 1 B + a bf16 scale per output channel; int4 = GPTQ / AWQ "
               "style 4-bit with an fp16 scale and a 4-bit zero point per group of 128 inputs; nf4 = bitsandbytes "
               "NF4 (4 bits + an fp32 absmax per 64 elements, no double quantisation); nvfp4 = 4-bit E2M1 + an fp8 "
               "scale per 16 elements + an fp32 global scale per tensor (compressed-tensors nvfp4-pack-quantized); "
               "GGUF = llama.cpp "
               "llama-quantize's per-tensor types (graph/gguf_quant.py, read from llama.cpp's source)",
    "kv": "elements as arch_metrics kv (as allocated, MLA latent, window layers cap at W - 1 tokens) x the format: "
          "bf16 2 B, fp8 1 B, q8_0 34 B per 32, q4_0 18 B per 32; per sequence, x batch",
    "activation": "peak transient bytes of an L-token prefill of ONE sequence at bf16, no grad (inference), batch "
                  "scales linearly; the residual stream and the normed input, plus the largest per-layer transient: "
                  "the attention (q, k, v, RoPE temporaries, output; eager also the L x L scores: bf16 scores, fp32 "
                  "softmax and its bf16 copy alive together, and the L x L mask once per forward) or the FFN "
                  "(gate, act, up and their product alive together; MoE: the k selected experts' width); the RoPE "
                  "cos / sin tables; logits for the last position only (generate's logits_to_keep = 1). eager: the "
                  "masked bf16 scores, their fp32 copy and the fp32 softmax alive together (10 B per score) and the "
                  "L x L mask; flash = SDPA / FlashAttention (no L x L term). Kernel workspaces are not counted. "
                  "Measured against torch.cuda.max_memory_allocated (graph/arch_breakdown_trace.py memory)",
    "serving": "weights + batch x (KV(L) + recurrent state) + the flash activation peak of one L-token prefill",
    "bandwidth": "one decode step at position L for B sequences: weights read once (the routed experts touched by B "
                 "tokens: E x (1 - (1 - k / E)^B) of E per MoE layer, uniform routing), the LM head (a tied head "
                 "reads the table), B embedding rows, B x the KV over each layer's attended span (L - 1 cached "
                 "tokens, window-capped), B x the recurrent state read and written",
    "roofline": "UPPER BOUNDS, not benchmarks: decode tokens/s <= B / max(bytes / peak bandwidth, FLOPs / peak "
                "FLOPs) and prefill time >= max(FLOPs / peak FLOPs, weight bytes / peak bandwidth), from the vendors' "
                "published peaks (graph/data/hardware.json, dense, no sparsity). Real kernels reach a fraction of "
                "both peaks; graph/data/decode_benchmarks.json checks that published results fall below",
}
log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ tensors per group

def _t(role, kind, n_in, n_out, experts=1, elems=None, active=None):
    e = int(elems if elems is not None else n_in * n_out * experts)
    return {"role": role, "kind": kind, "in": n_in, "out": n_out, "experts": experts, "elems": e,
            "active": int(e if active is None else active)}


def _lump(role, elems, h):
    return _t(role, "lump", h, (elems // h) if h else 0, 1, elems)


def _vec(role, elems):
    return _t(role, "vector", None, None, 1, elems)


def attn_tensors(a, mla, h, k_eq_v=False):
    """One layer's attention tensors; their elements sum to arch_metrics.attn_params (W + other), the active ones to
    the per-token weights (attn_Wa) plus the vectors."""
    out = []
    if a.get("W_override") is not None:
        out.append(_lump("attn.block", a["W_override"], h))
        b = a.get("b_override", 0)
        if b:
            out.append(_vec("attn.misc", b))
        return out
    nh, nkv, dqk, dv = a["nh"], a["nkv"], a["dqk"], a["dv"]
    extra_W, other = a.get("extra_W", 0), a.get("extra_b", 0) + a.get("sinks", 0)
    if mla:
        if mla["q_lora"]:
            out.append(_t("attn.q_a", "linear", h, mla["q_lora"]))
            out.append(_t("attn.q_b", "linear", mla["q_lora"], nh * dqk))
            out.append(_vec("attn.lora_norm", mla["q_lora"]))
        else:
            out.append(_t("attn.q", "linear", h, nh * dqk))
        out.append(_t("attn.kv_a", "linear", h, mla["kv_lora"] + mla["rope"]))
        out.append(_t("attn.kv_b", "linear", mla["kv_lora"], nh * (mla["nope"] + dv)))
        out.append(_vec("attn.lora_norm", mla["kv_lora"]))
        out.append(_t("attn.o", "linear", nh * dv, h))
        if extra_W:
            out.append(_lump("attn.extra", extra_W, h))
        if a["o_bias"]:
            out.append(_vec("attn.bias", h))
        if other:
            out.append(_vec("attn.misc", other))
        return out
    moa = a.get("moa")
    if moa:                                                    # JetMoE mixture of attention experts
        E, k, D = moa["E"], moa["k"], moa["D"]
        out.append(_t("attn.experts", "linear", h, 2 * D, E, E * 2 * h * D, k * 2 * h * D))
        out.append(_t("attn.kv_proj", "linear", h, 2 * D))
        out.append(_t("attn.router", "linear", h, E))
        out.append(_vec("attn.bias", h))
        if other:
            out.append(_vec("attn.misc", other))
        return out
    xin = a.get("in_dim") or h
    out.append(_t("attn.q", "linear", xin, nh * dqk))
    if not a.get("no_kv_proj"):
        out.append(_t("attn.k", "linear", xin, nkv * dqk))
        if not k_eq_v:
            out.append(_t("attn.v", "linear", xin, nkv * dv))
    out.append(_t("attn.o", "linear", nh * dv, h))
    if a.get("out_gate"):
        out.append(_t("attn.gate", "linear", h, nh if a.get("gate_headwise") else nh * dv))
    if extra_W:
        e = _lump("attn.extra", extra_W, h)
        e["active"] = extra_W - a.get("W_inactive", 0)        # MoVA: the routed value experts not selected
        out.append(e)
    if a["qk_norm"]:
        out.append(_vec("attn.qk_norm", a["qk_norm"]))
    b = 0
    if a["qkv_bias"]:
        b += nh * dqk + (0 if a.get("no_kv_proj") else nkv * dqk + (0 if k_eq_v else nkv * dv))
    if a["o_bias"]:
        b += h
    if b:
        out.append(_vec("attn.bias", b))
    if other:
        out.append(_vec("attn.misc", other))
    return out


def ffn_tensors(f, h):
    if f is None:
        return []
    if f.get("linear"):
        return [_t("mlp.linear", "linear", h, h)]
    if f.get("rwkv"):
        return [_t("mlp.up", "linear", h, f["inter"]), _t("mlp.receptance", "linear", h, h),
                _t("mlp.down", "linear", f["inter"], h), _vec("mlp.misc", 2 * h)]
    out = []
    if f["glu"]:
        out.append(_t("mlp.gate", "linear", h, f["inter"]))
    out += [_t("mlp.up", "linear", h, f["inter"]), _t("mlp.down", "linear", f["inter"], h)]
    if f["bias"]:
        out.append(_vec("mlp.bias", (2 * f["inter"] + h) if f["glu"] else (f["inter"] + h)))
    return out


def moe_tensors(m, h):
    if m is None:
        return []
    xin = m.get("latent") or h
    E, k, ew = m["E"], m["k"], m["ew"]
    glu = m.get("glu", True)
    out = []
    parts = (("gate", xin, ew), ("up", xin, ew), ("down", ew, xin)) if glu else (("up", xin, ew), ("down", ew, xin))
    for nm, a, b in parts:
        out.append(_t("moe.routed." + nm, "linear", a, b, E, E * a * b, k * a * b))
    s = m["shared"]
    if s:
        sglu = m.get("shared_glu", glu)
        sp = (("gate", h, s), ("up", h, s), ("down", s, h)) if sglu else (("up", h, s), ("down", s, h))
        for nm, a, b in sp:
            out.append(_t("moe.shared." + nm, "linear", a, b))
    if m.get("latent"):
        out += [_t("moe.latent", "linear", h, m["latent"]), _t("moe.latent", "linear", m["latent"], h)]
    if not m.get("no_router"):                                 # (a token-id routed MoE: a fixed table, no router)
        out.append(_t("moe.router", "linear", h, E + m.get("zero", 0)))
    if m["shared_gate"]:
        out.append(_t("moe.shared_gate", "linear", h, 1))
    b = (E if m["router_bias"] else 0) + (E * (2 * ew + h) if m["expert_bias"] else 0) + (h if m.get("out_bias") else 0)
    if b:
        out.append(_vec("moe.bias", b))
    return out


def group_tensors(g, h):
    """The tensors of ONE layer of group g (elements sum to arch_metrics._layer_total(g, h))."""
    out = []
    if g["attn"] is not None:
        out += attn_tensors(g["attn"], g["mla"], h, g["k_eq_v"])
    if g["ffn"] is not None:
        out += ffn_tensors(g["ffn"], h)
    if g["moe"] is not None:
        out += moe_tensors(g["moe"], h)
    if g["rec"] is not None:
        if g["rec"].get("W", 0):
            out.append(_lump("rec.linear", g["rec"]["W"], h))
        if g["rec"].get("b", 0):
            out.append(_vec("rec.misc", g["rec"]["b"]))
    if g["norm"]:
        out.append(_vec("norm", g["norm"]))
    return out


def global_tensors(p):
    """The tensors outside the layers: token table, other tables, LM head, final norm, stray params / linears."""
    h, V = p["hidden"], p["vocab"]
    emb = p["embedding"]
    tok_cols = (p["lm_head_W"] // V) if V and p["lm_head_W"] and p["lm_head_W"] % V == 0 else h
    tok = min(emb, V * tok_cols)
    out = [_t("embedding", "embed", tok_cols, V, 1, tok)]
    if emb - tok:
        out.append(_t("embedding.other", "embed", None, None, 1, emb - tok))
    if p["lm_head"]:
        out.append(_t("lm_head", "linear", tok_cols, p["lm_head"] // tok_cols if tok_cols else 0, 1, p["lm_head"]))
    if p["final_norm"]:
        out.append(_vec("final_norm", p["final_norm"]))
    if p["extra"]:
        out.append(_vec("other", p["extra"]))
    if p["extra_W"]:
        out.append(_lump("other.linear", p["extra_W"], h))
    return out


# ------------------------------------------------------------------------------------------------ labels and order

def group_label(g):
    """A layer kind's label: the mixer (global / sliding / KV-shared / recurrent family / none) and the FFN."""
    k = g["kind"]
    if g["attn"] is not None:
        mix = {"full": "global", "sliding": "sliding", "shared": "KV-shared"}.get(k, k)
        if g["mla"]:
            mix += " MLA"
        if g["k_eq_v"]:
            mix += " K=V"
        if g.get("dsa_index") is False:
            mix += " (no indexer)"
        if g["rec"] is not None and g["rec"].get("family") != "glue":
            mix += " + " + str(g["rec"].get("family"))
    elif g["rec"] is not None:
        mix = "glue" if g["rec"].get("family") == "glue" else str(g["rec"].get("family"))
    else:
        mix = "no mixer"
    ffn = "MoE" if g["moe"] is not None else ("FFN" if g["ffn"] is not None else "")
    return mix + (" · " + ffn if ffn else "")


def true_kinds(d, n):
    """The per-layer mixer kinds ('full' / 'local' / 'rec') of a config whose plan read only their COUNTS (the
    branches of model_graph._attn_counts that carry a layer order), or None."""
    from graph.model_graph import _kind, _num
    g = lambda k: _num(d.get(k))
    mt = str(d.get("model_type") or "").lower()
    for key in ("layers_block_type", "layer_type"):
        v = d.get(key)
        if isinstance(v, list) and len(v) == n and all(isinstance(x, str) for x in v):
            return [_kind(x) for x in v]
    for key, soft in (("attn_type_list", 1), ("linear_attention_pattern", 0)):
        v = d.get(key)
        if isinstance(v, list) and len(v) == n:
            return ["full" if x == soft else "rec" for x in v]
    hlp = d.get("hybrid_layer_pattern")
    if isinstance(hlp, list) and len(hlp) == n and all(x in (0, 1) for x in hlp):
        return ["full" if x == 0 else "local" for x in hlp]
    bt = d.get("_block_types") or d.get("block_types")
    if isinstance(bt, list) and bt and n:
        kinds = [_kind(bt[i % len(bt)]) for i in range(n)]
        return ["local" if k == "full" and d.get("attention_window_size") else k for k in kinds]
    lac = d.get("linear_attn_config")
    if isinstance(lac, dict) and isinstance(lac.get("full_attn_layers"), list):
        idx = set(int(i) - 1 for i in lac["full_attn_layers"] if isinstance(i, int))     # 1-based (Kimi-Linear)
        return ["full" if i in idx else "rec" for i in range(n)]
    for key in ("full_attn_idxs", "attn_layer_indices", "full_attention_layers"):
        if isinstance(d.get(key), list):
            idx = set(i for i in d[key] if isinstance(i, int))
            return ["full" if i in idx else "rec" for i in range(n)]
    if g("attn_layer_period"):
        per, off = g("attn_layer_period"), g("attn_layer_offset")
        return ["full" if i % per == off else "rec" for i in range(n)]
    if mt == "gemma2":
        return ["local" if i % 2 == 0 else "full" for i in range(n)]
    if g("sliding_window_pattern"):
        q = g("sliding_window_pattern")
        return ["full" if (i + 1) % q == 0 else "local" for i in range(n)]
    if mt in ("qwen2", "qwen3", "qwen2_moe", "qwen3_moe") and d.get("max_window_layers") is not None:
        return ["local" if i >= g("max_window_layers") else "full" for i in range(n)]
    return None


def fix_order(p, d):
    """The plan with each group's `at` in the real layer order: as planned when p['order_exact'], else remapped
    from true_kinds(d) when the per-layer (kind, MoE) pairs it gives have the plan's group counts; else the plan
    with p['order_exact'] False (the per-layer table then shows the groups, not an order). The MoE flag of a
    layer is the plan's own (its MoE mask is indexed by the real layer)."""
    if p.get("order_exact", True) or d is None:
        return p
    n = p["layers"]
    tk = true_kinds(d, n)
    rows = layer_order(dict(p, order_exact=True))
    if tk is None or rows is None or len(tk) != n or any(len(r) != 1 for r in rows):
        return p
    G = p["groups"]
    att = [k for k in ("full", "sliding") if any(g["kind"] == k for g in G)]
    vocab = {"rec": "rec", "full": "full" if "full" in att else (att[0] if att else None),
             "local": "sliding" if "sliding" in att else (att[0] if att else None)}
    want = [(vocab.get(k), G[rows[i][0]]["moe"] is not None) for i, k in enumerate(tk)]
    key = lambda g: (g["kind"], g["moe"] is not None)
    keys = [key(g) for g in G]
    if len(set(keys)) != len(keys):
        return p
    from collections import Counter
    if Counter(want) == Counter({key(g): g["n"] for g in G}):
        G2 = [dict(g, at=[i for i, w in enumerate(want) if w == key(g)]) for g in G]
        return dict(p, groups=G2, order_exact=True)
    # the plan paired mixers and FFNs in count order (the totals are sums over layers, so the pairing does not
    # change them); re-pair them per layer when each kind has one mixer and each MoE flag one FFN
    MIX = ("kind", "window", "attn", "mla", "rec", "k_eq_v", "dsa_index")
    FF = ("ffn", "moe")
    sig = lambda g, ks: json.dumps({k: g.get(k) for k in ks}, sort_keys=True, default=str)
    mix, ff = {}, {}
    for g in G:
        if g["np"] != g["n"] or mix.setdefault(g["kind"], sig(g, MIX)) != sig(g, MIX) or \
                ff.setdefault(g["moe"] is not None, sig(g, FF)) != sig(g, FF) or g["norm"] != G[0]["norm"]:
            return p
    tm = {g["kind"]: g for g in G}
    tf = {g["moe"] is not None: g for g in G}
    if any(k not in tm or m not in tf for k, m in want):
        return p
    G2 = []
    for k, m in dict.fromkeys(want):
        at = [i for i, w in enumerate(want) if w == (k, m)]
        g = dict(tm[k], ffn=tf[m]["ffn"], moe=tf[m]["moe"], n=len(at), np=len(at), at=at)
        G2.append(g)
    return dict(p, groups=G2, order_exact=True, repaired=True)


def checkpoint_quant(raw):
    """The checkpoint's own quant_method (config quantization_config), or None."""
    q = raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config")
    return str(q.get("quant_method")).lower() if isinstance(q, dict) and q.get("quant_method") else None


def plan_ordered(raw):
    """arch_metrics.plan(raw) with the layer order fixed (fix_order on the resolved config) and the checkpoint's
    quant method (`ckpt_quant`, read by the GGUF sizes)."""
    p = AM.plan(raw)
    if not p.get("order_exact", True):
        p = fix_order(p, AM.resolve(raw)[0])
    p["ckpt_quant"] = checkpoint_quant(raw)
    p["n_mtp"] = n_mtp(raw, p)
    p["norm_kind"] = (AM.classify(raw) or {}).get("norm", "RMSNorm")
    return p


PAGE_PLAN_KEYS = ("mt", "arch", "hidden", "vocab", "tied", "embedding", "lm_head", "lm_head_W", "final_norm", "extra",
                  "extra_W", "layers", "max_context", "dsa", "dsa_dim", "groups", "order_exact", "ckpt_quant", "n_mtp",
                  "norm_kind", "head_stored")


def page_plan(p):
    """The plan as the zoo page ships it (JSON; the keys the breakdown reads): graph/templates/arch_breakdown.js
    computes the same breakdown from it as this module does from the plan."""
    q = {k: p[k] for k in PAGE_PLAN_KEYS if k in p}
    return json.loads(json.dumps(q, default=lambda x: sorted(x) if isinstance(x, (set, frozenset)) else str(x)))


def js_tables():
    """The tables graph/templates/arch_breakdown.js reads (setTables)."""
    from graph import gguf_quant as GQ
    with open(os.path.join(HERE, "data", "arch_metrics_js_tables.json")) as f:
        am = json.load(f)
    return {"am": am, "quant": GQ.tables(), "hardware": hardware().get("hardware", []), "calib": gguf_calibration()}


def n_mtp(raw, p):
    """MTP (next-token-prediction) layers the checkpoint ships beside the LM (arch_metrics.hub_components' keys)."""
    t = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    n = AM._g(t, "num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers")
    if not n and p["mt"] == "qwen3_next":
        n = 1                                               # Qwen3-Next ships one MTP layer (no config key)
    return int(n or 0)


def layer_order(p):
    """{layer index: [group indices]} when every group carries a valid `at` (the plan's layer indices) in the real
    order (p['order_exact']), else None."""
    if not p.get("order_exact", True):
        return None
    n = p["layers"]
    rows = [[] for _ in range(n)]
    for gi, g in enumerate(p["groups"]):
        at = g.get("at")
        if not isinstance(at, list) or len(at) != g["n"] or any((not isinstance(i, int)) or i < 0 or i >= n for i in at):
            return None
        for i in at:
            rows[i].append(gi)
    if any(not r for r in rows):
        return None
    return rows


# ------------------------------------------------------------------------------------------------ parameters

def _add(d, k, v):
    d[k] = d.get(k, 0) + v


def params_breakdown(p):
    """{components, active, groups: [...], by_kind, per_layer?}: exact integer parameter counts."""
    h = p["hidden"]
    comp, act = {}, {}
    for t in global_tensors(p):
        _add(comp, t["role"], t["elems"])
        _add(act, t["role"], t["active"])
    groups, by_kind = [], {}
    for gi, g in enumerate(p["groups"]):
        ts = group_tensors(g, h)
        per, per_act = {}, {}
        for t in ts:
            _add(per, t["role"], t["elems"])
            _add(per_act, t["role"], t["active"])
        for r, v in per.items():
            _add(comp, r, g["np"] * v)
        for r, v in per_act.items():
            _add(act, r, g["np"] * v)
        lt = sum(per.values())
        lab = group_label(g)
        a = g["attn"]
        mm = sum(2 * t["active"] for t in ts if t["kind"] in ("linear", "lump"))
        rec = {"label": lab, "kind": g["kind"], "n": g["n"], "np": g["np"], "window": g["window"],
               "heads": a["nh"] if a else 0, "kv_heads": (a["nkv"] if not g["mla"] else 0) if a else 0,
               "head_dim": a["dqk"] if a else 0, "v_head_dim": a["dv"] if a else 0,
               "experts": [g["moe"]["E"], g["moe"]["k"]] if g["moe"] is not None else None,
               "params": per, "layer_total": lt, "active_layer_total": sum(per_act.values()),
               "kv_per_token": kv_group_per_token(p, g), "flops_matmul": mm,
               "attn_per_key": 2 * a["nh"] * (a["dqk"] + a["dv"]) if a else 0,
               "rec_core": g["rec"].get("core_flops", 0) if g["rec"] else 0}
        groups.append(rec)
        bk = by_kind.setdefault(lab, {"layers": 0, "distinct": 0, "params": 0, "active": 0})
        bk["layers"] += g["n"]
        bk["distinct"] += g["np"]
        bk["params"] += g["np"] * lt
        bk["active"] += g["np"] * sum(per_act.values())
    out = {"components": comp, "active": act, "total": sum(comp.values()), "active_total": sum(act.values()),
           "groups": groups, "by_kind": by_kind, "tied": bool(p["tied"])}
    nm = p.get("n_mtp", 0)
    if nm and p["groups"]:
        lay = max(p["groups"], key=lambda g: (g["moe"] is not None, g["attn"] is not None and g["kind"] == "full",
                                             g.get("dsa_index", True)))
        L1 = sum(t["elems"] for t in group_tensors(lay, h))
        # stored beside the LM, not in the total (arch_metrics.hub_components: one decoder layer of the main kind,
        # eh_proj 2h -> h, enorm / hnorm / head norm)
        out["mtp"] = {"layers": nm, "params": nm * (L1 + 2 * h * h + 3 * h)}
    return out


# ------------------------------------------------------------------------------------------------ FLOPs

def _span_sum(L, W):
    return AM._span_causal_sum(L, W)


def _attn_group(p, g, L, mode):
    """(scores, av, indexer) FLOPs of ONE layer of group g: decode (one step at position L) or the TOTAL over an
    L-token causal prefill (exact integers)."""
    a = g["attn"]
    if a is None:
        return 0, 0, 0
    W = g["window"]
    sc, av = 2 * a["nh"] * a["dqk"], 2 * a["nh"] * a["dv"]
    dsa = p.get("dsa") if g["kind"] != "shared" else None
    if dsa:
        idx = 2 * dsa["heads"] * dsa["dim"] if g.get("dsa_index", True) else 0
        if mode == "decode":
            s = min(L, dsa["topk"])
            return sc * s, av * s, idx * L
        S = _span_sum(L, dsa["topk"])
        return sc * S, av * S, idx * _span_sum(L, None)
    if mode == "decode":
        s = min(L, W) if W else L
        return sc * s, av * s, 0
    S = _span_sum(L, W)
    return sc * S, av * S, 0


def matmul_components(p):
    """{role: FLOPs per token} of the linear roles (2 x active weights x the unrolled count n), the LM head and the
    other linears outside the layers; sums to arch_metrics flops matmul."""
    h = p["hidden"]
    out = {}
    for g in p["groups"]:
        for t in group_tensors(g, h):
            if t["kind"] in ("linear", "lump"):
                _add(out, t["role"], 2 * g["n"] * t["active"])
    out["lm_head"] = 2 * p["lm_head_W"]
    if p["extra_W"]:
        out["other.linear"] = 2 * p["extra_W"]
    return out


def rec_core(p):
    return sum(g["n"] * g["rec"].get("core_flops", 0) for g in p["groups"] if g["rec"])


def flops_at(p, L):
    """Decode at position L and an L-token prefill: per component, per layer kind; prefill totals over the L tokens
    (floats where they pass 2^53) and amortised per token."""
    mm = matmul_components(p)
    mm_tot = sum(mm.values())
    core = rec_core(p)
    dec = dict(mm)
    pre_tot = {}
    kinds_dec, kinds_pre = {}, {}
    att_d = [0, 0, 0]
    att_p = [0, 0, 0]
    for g in p["groups"]:
        if g["attn"] is None:
            continue
        lab = group_label(g)
        d3 = _attn_group(p, g, L, "decode")
        p3 = _attn_group(p, g, L, "prefill")
        for j in range(3):
            att_d[j] += g["n"] * d3[j]
            att_p[j] += g["n"] * p3[j]
        _add(kinds_dec, lab, g["n"] * sum(d3))
        _add(kinds_pre, lab, g["n"] * sum(p3))
    dec.update({"attn.scores": att_d[0], "attn.av": att_d[1]})
    if att_d[2]:
        dec["attn.indexer"] = att_d[2]
    if core:
        dec["rec.core"] = core
    dec_total = mm_tot + sum(att_d) + core
    pre_attn_total = sum(att_p)
    total_L = L * (mm_tot + core) + pre_attn_total                  # exact int over the L-token prefill
    pre = {r: v for r, v in mm.items()}                             # matmul per token is the same in prefill
    pre["attn.scores"] = att_p[0] / L
    pre["attn.av"] = att_p[1] / L
    if att_p[2]:
        pre["attn.indexer"] = att_p[2] / L
    if core:
        pre["rec.core"] = core
    return {
        "decode": {"components": dec, "attn_by_kind": kinds_dec, "total": dec_total},
        "prefill": {"components": pre, "attn_by_kind": {k: v / L for k, v in kinds_pre.items()},
                    "per_token": total_L / L, "total": float(total_L),
                    "attn_total": float(pre_attn_total)},
        "training_approx": float(3 * total_L),
    }


NORM_FLOPS = {"RMSNorm": 4, "LayerNorm": 7}


def elementwise_at(p, L, norm_kind):
    """Non-matmul FLOPs per decode token at position L (the stated convention; not in any total)."""
    per_elem = NORM_FLOPS.get(norm_kind, 4)
    div = 2 if norm_kind == "LayerNorm" else 1
    sm = norms = act = 0
    for g in p["groups"]:
        a = g["attn"]
        if a is not None:
            W = g["window"]
            dsa = p.get("dsa") if g["kind"] != "shared" else None
            span = min(L, dsa["topk"]) if dsa else (min(L, W) if W else L)
            sm += g["n"] * 5 * a["nh"] * span
        norms += g["n"] * per_elem * (g["norm"] // div)
        f = g["ffn"]
        if f is not None and not f.get("linear"):
            act += g["n"] * (5 if f.get("glu") else 4) * f["inter"]
        m = g["moe"]
        if m is not None:
            w = m["k"] * m["ew"] + m["shared"]
            act += g["n"] * (5 if m.get("glu", True) else 4) * w
    norms += per_elem * (p["final_norm"] // div)
    return {"softmax": sm, "norms": norms, "activations": act, "total": sm + norms + act}


# ------------------------------------------------------------------------------------------------ KV and state

def kv_group_per_token(p, g):
    """Elements ONE layer of group g caches per token (headline convention, DSA indexer keys included)."""
    a = g["attn"]
    if a is None or g["kind"] == "shared":
        return 0
    if g["mla"]:
        e = g["mla"]["kv_lora"] + g["mla"]["rope"]
    elif g["k_eq_v"]:
        e = a["nkv"] * a["dqk"] * AM._kv_tensors(p, g)
    else:
        e = a["nkv"] * (a["dqk"] + a["dv"])
    if p.get("dsa_dim") and g.get("dsa_index", True):
        e += p["dsa_dim"]
    return e


def kv_elems(p, L):
    """Elements cached after L tokens, one sequence (== sum of arch_metrics._kv_headline)."""
    s = 0
    for g in p["groups"]:
        e = kv_group_per_token(p, g)
        if not e:
            continue
        W = g["window"]
        toks = min(L, W - 1) if W else L
        if p.get("dsa_dim") and g.get("dsa_index", True):
            s += g["n"] * ((e - p["dsa_dim"]) * toks + p["dsa_dim"] * L)   # the indexer keys are never windowed
        else:
            s += g["n"] * e * toks
    return s


def kv_bytes(elems, fmt):
    b, per = KV_FORMATS[fmt]
    if per == 1:
        return elems * b
    return elems * b / per


def kv_by_kind(p, L):
    out = {}
    for g in p["groups"]:
        e = kv_group_per_token(p, g)
        if not e:
            continue
        W = g["window"]
        toks = min(L, W - 1) if W else L
        if p.get("dsa_dim") and g.get("dsa_index", True):
            v = g["n"] * ((e - p["dsa_dim"]) * toks + p["dsa_dim"] * L)
        else:
            v = g["n"] * e * toks
        _add(out, group_label(g), v)
    return out


def state_elems(p):
    c, s = AM.state_elems(p)
    return c + s


# ------------------------------------------------------------------------------------------------ weight formats

QUANT_ROLES_EXCLUDED = ("embedding", "embedding.other", "lm_head", "moe.router", "attn.router")


def _is_quant_target(t):
    return t["kind"] in ("linear", "lump") and t["role"] not in QUANT_ROLES_EXCLUDED and not t["role"].startswith(
        "other")


def _ceil(a, b):
    return -(-a // b)


def tensor_format_bytes(t, fmt):
    """Bytes of one tensor (all experts) in a safetensors-style weight format (the CONVENTIONS above)."""
    e = t["elems"]
    if fmt == "fp32":
        return 4 * e
    if fmt == "bf16" or not _is_quant_target(t):
        return 2 * e
    n_in = t["in"] or 1
    n_out = (e // (n_in * t["experts"])) if n_in and t["experts"] else 0
    X = t["experts"]
    if fmt == "fp8":
        return e + 4 * X * _ceil(n_in, 128) * _ceil(n_out, 128)
    if fmt == "int8":
        return e + 2 * X * n_out
    if fmt == "int4":
        return e // 2 + (e % 2) + X * _ceil(n_in, 128) * n_out * 5 // 2
    if fmt == "nf4":
        return e // 2 + (e % 2) + 4 * _ceil(e, 64)
    if fmt == "nvfp4":
        return e // 2 + (e % 2) + _ceil(e, 16) + 4 * X
    raise KeyError(fmt)


def all_tensors(p):
    """[(tensor, copies)] of the whole model: the globals once, each group's layer tensors np times."""
    out = [(t, 1) for t in global_tensors(p)]
    for g in p["groups"]:
        for t in group_tensors(g, p["hidden"]):
            out.append((t, g["np"]))
    return out


def weight_bytes(p, fmt, tensors=None):
    """{total, routed, embedding, lm_head} bytes in a safetensors-style format."""
    tensors = tensors or all_tensors(p)
    out = {"total": 0, "routed": 0, "embedding": 0, "lm_head": 0}
    for t, c in tensors:
        b = c * tensor_format_bytes(t, fmt)
        out["total"] += b
        if t["role"].startswith("moe.routed") or t["role"] == "attn.experts":
            out["routed"] += b
        if t["role"] in ("embedding", "embedding.other"):
            out["embedding"] += b
        if t["role"] == "lm_head":
            out["lm_head"] += b
    return out


def routed_fraction(p):
    """(E, k) of the routed blocks, or None: one (E, k) per model in every plan the zoo has."""
    for g in p["groups"]:
        if g["moe"] is not None and g["moe"]["E"] > 1:
            return g["moe"]["E"], g["moe"]["k"]
        a = g["attn"]
        if a is not None and a.get("moa"):
            return a["moa"]["E"], a["moa"]["k"]
    return None


def expert_coverage(E, k, B):
    """Expected fraction of E experts touched by B tokens choosing k each (uniform, independent routing)."""
    if B == 1:
        return k / E
    return 1 - (1 - k / E) ** B


# ------------------------------------------------------------------------------------------------ GGUF (llama.cpp)

# HF model_type -> llama.cpp GGUF architecture (the converter's model_arch, conversion/*.py); a type not listed
# converts under its own name for the rules' purposes (they only single out falcon and the 70B / MoE cases)
LLAMA_ARCH = {
    "llama": "llama", "mistral": "llama", "mixtral": "llama", "internlm3": "llama", "qwen2": "qwen2", "qwen3": "qwen3",
    "qwen3_moe": "qwen3moe", "qwen2_moe": "qwen2moe", "qwen3_next": "qwen3next", "gemma": "gemma", "gemma2": "gemma2",
    "gemma3": "gemma3", "gemma3_text": "gemma3", "phi3": "phi3", "phi": "phi2", "phimoe": "phimoe",
    "deepseek_v2": "deepseek2", "deepseek_v3": "deepseek2", "deepseek": "deepseek", "olmo": "olmo", "olmo2": "olmo2",
    "olmoe": "olmoe", "olmo3": "olmo2", "granite": "granite", "granitemoe": "granitemoe",
    "granitemoehybrid": "granitehybrid", "falcon": "falcon", "falcon_h1": "falcon-h1", "falcon_mamba": "mamba",
    "mamba": "mamba", "internlm2": "internlm2", "glm4": "glm4", "chatglm": "chatglm", "glm4_moe": "glm4moe",
    "exaone": "exaone", "exaone4": "exaone4", "cohere": "command-r", "cohere2": "cohere2", "starcoder2": "starcoder2",
    "gpt_bigcode": "starcoder", "gpt_oss": "gpt-oss", "smollm3": "smollm3", "stablelm": "stablelm", "gpt2": "gpt2",
    "gpt_neox": "gptneox", "bloom": "bloom", "lfm2": "lfm2", "hunyuan_v1_dense": "hunyuan-dense", "ernie4_5": "ernie4_5",
    "ernie4_5_moe": "ernie4_5-moe", "seed_oss": "seed_oss", "apertus": "apertus", "nemotron_h": "nemotron_h",
    "nemotron": "nemotron", "minimax_m2": "minimax-m2", "bailing_moe": "bailingmoe", "arcee": "arcee",
    "baichuan": "baichuan", "minicpm": "minicpm", "jamba": "jamba", "dots1": "dots1", "nemotron-nas": "deci",
    "mistral3": "llama", "llama4": "llama4"}
# HF layouts whose attention is ONE fused projection in the GGUF (gguf-py tensor_mapping.py MODEL_TENSOR.ATTN_QKV:
# query_key_value, c_attn, Wqkv, qkv_proj), and whose gate / up projections are one ffn_up (MODEL_TENSOR.FFN_UP:
# phi3 gate_up_proj, chatglm dense_h_to_4h)
GGUF_FUSED_QKV = frozenset({"gpt_neox", "falcon", "bloom", "persimmon", "chatglm", "gpt2", "gpt_bigcode", "qwen", "jais",
                            "mpt", "dbrx", "phi3", "openelm", "plamo2", "bailing_moe"})
GGUF_FUSED_GATE_UP = frozenset({"phi3", "chatglm"})
# converter classes that export the checkpoint's MTP / NextN layers by default (supports_mtp_export, block_count +=
# the NextN layers: conversion/glm.py, qwen.py, deepseek.py, mimo.py, step3.py, exaone.py, bailingmoe.py, mellum.py,
# command_r.py, qwen4exp.py at the pinned llama.cpp commit), by the HF architecture they register
GGUF_MTP_ARCHS = frozenset({
    "Glm4MoeForCausalLM", "Glm4vMoeForConditionalGeneration", "Glm4MoeLiteForCausalLM", "GlmMoeDsaForCausalLM",
    "Glm5NextForConditionalGeneration", "GlmOcrForConditionalGeneration", "Qwen3NextForCausalLM",
    "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5TextModel", "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM", "DeepseekV32ForCausalLM",
    "DeepseekV4ForCausalLM", "MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM", "Step3p5ForCausalLM",
    "Step3p7ForConditionalGeneration", "ExaoneMoEForCausalLM", "ExaoneMoeForCausalLM",
    "Exaone4_5_ForConditionalGeneration", "BailingMoeV2ForCausalLM", "MellumForCausalLM", "Cohere2MoeForCausalLM"})
GGUF_NAME = {"attn.q": "attn_q", "attn.k": "attn_k", "attn.v": "attn_v", "attn.o": "attn_output",
             "attn.gate": "attn_gate", "attn.q_a": "attn_q_a", "attn.q_b": "attn_q_b", "attn.kv_a": "attn_kv_a_mqa",
             "attn.extra": "attn_extra", "attn.experts": "attn_exps", "attn.kv_proj": "attn_kv_proj",
             "attn.router": "attn_router", "attn.block": "attn_block", "mlp.gate": "ffn_gate", "mlp.up": "ffn_up",
             "mlp.down": "ffn_down", "mlp.linear": "ffn_up", "mlp.receptance": "channel_mix_receptance",
             "moe.routed.gate": "ffn_gate_exps", "moe.routed.up": "ffn_up_exps", "moe.routed.down": "ffn_down_exps",
             "moe.shared.gate": "ffn_gate_shexp", "moe.shared.up": "ffn_up_shexp", "moe.shared.down": "ffn_down_shexp",
             "moe.latent": "ffn_latent", "moe.router": "ffn_gate_inp", "moe.shared_gate": "ffn_gate_inp_shexp",
             "rec.linear": "ssm_in"}
GGUF_VEC = {"attn.bias": "attn_qkv.bias", "attn.qk_norm": "attn_q_norm.weight", "attn.lora_norm": "attn_q_a_norm.weight",
            "attn.misc": "attn_sinks.weight", "mlp.bias": "ffn_up.bias", "mlp.misc": "channel_mix_lerp.weight",
            "moe.bias": "exp_probs_b.bias", "rec.misc": "ssm_d.weight", "norm": "attn_norm.weight"}


def gguf_mtp_layers(p):
    """MTP layers the GGUF holds (the converter exports them by default for GGUF_MTP_ARCHS)."""
    return p.get("n_mtp", 0) if p.get("arch") in GGUF_MTP_ARCHS else 0


def _layer_slots(g):
    """The layer indices a group's np weight copies sit at (the first np of its `at`, else 0..np-1)."""
    at = g.get("at")
    if isinstance(at, list) and len(at) >= g["np"]:
        return at[:g["np"]]
    return list(range(g["np"]))


def gguf_tensors(p):
    """The model's tensors as llama.cpp's converter writes them: names, GGML shapes [ne0 = the input dim, ne1,
    (ne2 = experts)]. The shapes the plan holds as lumps are [hidden, elements / hidden] (+ a 1-D remainder)."""
    mt = str(p["mt"]).lower()
    h = p["hidden"]
    out = []

    def lin(name, n_in, n_out, X=1):
        out.append({"name": name, "shape": [int(n_in), int(n_out)] + ([int(X)] if X > 1 else [])})

    def lump(name, e, base):
        if h and e // h:
            lin(base + ".weight", h, e // h)
        if h and e % h:
            out.append({"name": base + "_rest.weight", "shape": [int(e % h)]})
    for t in global_tensors(p):
        r = t["role"]
        if r == "embedding":
            lin("token_embd.weight", t["in"], t["out"])
        elif r == "lm_head":
            lin("output.weight", t["in"], t["out"])
        elif r == "embedding.other":
            lump(r, t["elems"], "per_layer_token_embd" if mt.startswith(("gemma3n", "gemma4")) else "position_embd")
        elif r == "other.linear":
            lump("output_extra", t["elems"], "output_extra")
        elif r == "final_norm":
            out.append({"name": "output_norm.weight", "shape": [int(t["elems"])]})
        else:
            out.append({"name": "output_misc.bias", "shape": [int(t["elems"])]})
    slots = [(g, _layer_slots(g)) for g in p["groups"]]
    nm = gguf_mtp_layers(p)
    if nm:
        lay = max(p["groups"], key=lambda g: (g["moe"] is not None, g["attn"] is not None and g["kind"] == "full",
                                             g.get("dsa_index", True)))
        slots.append((lay, list(range(p["layers"], p["layers"] + nm))))
        for i in range(p["layers"], p["layers"] + nm):
            lin(f"blk.{i}.nextn.eh_proj.weight", 2 * h, h)
            for n_ in ("enorm", "hnorm", "shared_head_norm"):
                out.append({"name": f"blk.{i}.nextn.{n_}.weight", "shape": [int(h)]})
    for g, at in slots:
        ts = group_tensors(g, h)
        for i in at:
            pre = f"blk.{i}."
            qkv = mt in GGUF_FUSED_QKV and all(any(t["role"] == r for t in ts) for r in ("attn.q", "attn.k"))
            gu = mt in GGUF_FUSED_GATE_UP and any(t["role"] == "mlp.gate" for t in ts)
            fused = {}
            for t in ts:
                r = t["role"]
                if t["kind"] == "vector":
                    out.append({"name": pre + GGUF_VEC.get(r, "misc.weight"), "shape": [int(t["elems"])]})
                    continue
                if qkv and r in ("attn.q", "attn.k", "attn.v"):
                    fused.setdefault("attn_qkv", [t["in"], 0])[1] += t["out"]
                    continue
                if gu and r in ("mlp.gate", "mlp.up"):
                    fused.setdefault("ffn_up", [t["in"], 0])[1] += t["out"]
                    continue
                if r == "attn.kv_b" and g["mla"]:                    # split for the MLA absorption (deepseek.py)
                    m = g["mla"]
                    nh = g["attn"]["nh"]
                    lin(pre + "attn_k_b.weight", m["nope"], m["kv_lora"], nh)
                    lin(pre + "attn_v_b.weight", m["kv_lora"], m["v"], nh)
                    continue
                if t["kind"] == "lump":
                    lump(r, t["elems"], pre + GGUF_NAME.get(r, "misc"))
                    continue
                if r == "moe.shared_gate":
                    out.append({"name": pre + "ffn_gate_inp_shexp.weight", "shape": [int(t["in"])]})
                    continue
                nm = GGUF_NAME.get(r, "misc")
                if r == "moe.latent":
                    nm = "ffn_latent_in" if t["in"] == h else "ffn_latent_out"
                lin(pre + nm + ".weight", t["in"], t["out"], t["experts"])
            for nm, (a, b) in fused.items():
                lin(pre + nm + ".weight", a, b)
    return out


GGUF_CALIBRATION = os.path.join(HERE, "data", "gguf_calibration.json")
_GC = None


def gguf_calibration():
    """The header terms fitted on the Hub's BF16 / F16 GGUF files (graph/arch_breakdown_trace.py hub)."""
    global _GC
    if _GC is None:
        try:
            with open(GGUF_CALIBRATION) as f:
                _GC = json.load(f)
        except FileNotFoundError:
            _GC = {"bytes_per_vocab": 28.0, "fixed": 4096}
    return _GC


def gguf_header(p, tensors):
    """The GGUF metadata bytes: the tensor infos exactly (name + dims + type + offset), the tokenizer and other KVs
    as a per-vocab-entry term fitted on the Hub's unquantised GGUF files (tokens, merges / scores, token types)."""
    c = gguf_calibration()
    infos = sum(8 + len(t["name"]) + 4 + 8 * len(t["shape"]) + 4 + 8 for t in tensors)
    return int(c["fixed"] + infos + c["bytes_per_vocab"] * p["vocab"])


def gguf_sizes(p, levels=GGUF_LEVELS, with_header=True):
    """{level: bytes} of the GGUF llama-quantize writes at each level (default params)."""
    from graph import gguf_quant as GQ
    mt = str(p["mt"]).lower()
    arch = LLAMA_ARCH.get(mt, mt)
    if p.get("head_stored") and p["tied"]:
        p = dict(p, tied=False, lm_head=p["lm_head_W"])
    ts = gguf_tensors(p)
    src = GQ.source_types(ts, arch)
    ts = [dict(t, src=s) for t, s in zip(ts, src)]
    kept = 0
    if p.get("ckpt_quant") == "mxfp4" or mt == "gpt_oss":
        # an MXFP4 checkpoint (gpt-oss): the converter writes the experts as MXFP4 and the GGUFs keep them so at
        # every level (the Hub's gpt-oss files barely change from Q8_0 to Q2_K): their bytes are added as MXFP4,
        # outside the type choice
        T = GQ.tables()
        ex = [t for t in ts if "_exps." in t["name"]]
        ts = [t for t in ts if "_exps." not in t["name"]]
        kept = sum(GQ._pad(GQ.tensor_bytes("MXFP4", GQ._n_elements(t["shape"]), t["shape"][0], T), T["alignment"])
                   for t in ex)
    a0 = next((g["attn"] for g in p["groups"] if g["attn"] is not None), None)
    E = next((g["moe"]["E"] for g in p["groups"] if g["moe"] is not None), 0)
    ctx = {"n_expert": E, "n_layer": p["layers"] + gguf_mtp_layers(p)}
    if a0:
        ctx.update(n_head=a0["nh"], n_head_kv=a0["nkv"])
    head = (gguf_header(p, ts) if with_header else 0) + kept
    out = {}
    for lv in levels:
        ftype = None if lv in ("BF16", "F16") else lv
        if ftype is None:
            tts = [dict(t, src=("F16" if lv == "F16" and t["src"] == "BF16" else t["src"])) for t in ts]
            b = GQ.breakdown(tts, None, arch, head, **ctx)
        else:
            b = GQ.breakdown(ts, ftype, arch, head, **ctx)
        out[lv] = b["total"]
    return out


# ------------------------------------------------------------------------------------------------ activations

def activation_bytes(p, L, attn="flash", b=1, a_bytes=2):
    """Peak transient bytes of an L-token prefill (see CONVENTIONS['activation'])."""
    h = p["hidden"]
    base = 2 * b * L * h * a_bytes                     # the residual stream and the normed input
    peak = 0
    for g in p["groups"]:
        at = 0
        a = g["attn"]
        if a is not None:
            q = b * L * a["nh"] * a["dqk"]
            kv = b * L * (a["nkv"] if not g["mla"] else a["nh"]) * (a["dqk"] + a["dv"])
            o = b * L * a["nh"] * a["dv"]
            at = (2 * q + kv + o + b * L * h) * a_bytes   # q (+ its RoPE copy), k / v, the output, o_proj's output
            if attn == "eager":
                # softmax(scores, dtype=float32) on bf16 scores: the masked bf16 scores, their fp32 copy and the
                # fp32 result are alive together (10 bytes per score; measured, graph/arch_breakdown_trace.py)
                at += b * a["nh"] * L * L * (2 + 4 + 4)
        ff = 0
        f = g["ffn"]
        if f is not None and not f.get("linear") and not f.get("rwkv"):
            ff = (3 if f["glu"] else 2) * b * L * f["inter"] * a_bytes
        m = g["moe"]
        if m is not None:
            # the k experts' width, fp32 router logits, and the int64 one-hot (E x k x tokens) transformers' MoE
            # blocks build to dispatch the tokens
            w = m["k"] * m["ew"] + m["shared"]
            ff = max(ff, 3 * b * L * w * a_bytes + b * L * m["E"] * 4 + 8 * b * L * m["k"] * m["E"])
        rc = 0
        r = g["rec"]
        if r is not None and r.get("W"):
            rc = 2 * b * L * (r["W"] // h if h else 0) * a_bytes
        peak = max(peak, at, ff, rc)
    mask = b * L * L * a_bytes if attn == "eager" else 0   # the 4-D float mask, built once per forward
    rope = 0
    a0 = next((g["attn"] for g in p["groups"] if g["attn"] is not None), None)
    if a0 is not None:
        rope = 2 * b * L * a0["dqk"] * a_bytes             # the cos / sin tables of the forward
    logits = b * p["lm_head_W"] // h * 4 if h else 0    # last position only, fp32
    return float(base + peak + mask + rope + logits)       # a float: L x L terms pass 2^53 at 10M-token contexts


# ------------------------------------------------------------------------------------------------ hardware

HARDWARE = os.path.join(HERE, "data", "hardware.json")
_HW = None


def hardware():
    global _HW
    if _HW is None:
        try:
            with open(HARDWARE) as f:
                _HW = json.load(f)
        except FileNotFoundError:
            _HW = {"hardware": []}
    return _HW


def hw_peak(hw, fmt):
    """(bytes/s, FLOP/s) of a hardware record for a weight format: fp8 weights use the fp8 peak where published."""
    fl = hw.get("flops") or {}
    t = None
    if fmt == "fp8":
        t = fl.get("fp8_dense")
    if t is None:
        t = fl.get("bf16_dense") or fl.get("fp16_dense")
    return hw["mem_bw_gbps"] * 1e9, (t or 0) * 1e12


def decode_bound(bw_rec, hw, fmt="bf16"):
    """Decode tokens/s upper bound of one bandwidth record ({bytes, flops, B})."""
    bps, fps = hw_peak(hw, fmt)
    t = bw_rec["bytes"] / bps
    if fps:
        t = max(t, bw_rec["flops"] / fps)
    return bw_rec["B"] / t if t else None


def prefill_bound(total_flops, weight_bytes_, hw, fmt="bf16"):
    """Prefill (time to first token) lower bound in seconds."""
    bps, fps = hw_peak(hw, fmt)
    t = weight_bytes_ / bps
    if fps:
        t = max(t, total_flops / fps)
    return t


# ------------------------------------------------------------------------------------------------ the record

def contexts_for(p):
    ctx = dict(CONTEXTS)
    ctx["max"] = int(p["max_context"])
    return ctx


def decode_read(p, L, B, wb, emb_row_bytes, kvfmt="bf16", state_bytes=2):
    """Bytes read by one decode step at position L for B sequences (weights wb = weight_bytes(p, fmt))."""
    ek = routed_fraction(p)
    routed = wb["routed"]
    if ek and routed:
        routed_read = routed * expert_coverage(ek[0], ek[1], B)
    else:
        routed_read = routed
    head = wb["lm_head"] if p["lm_head"] else (wb["embedding"] if p["tied"] else 0)
    if p["tied"] and wb["embedding"]:
        # the tied head reads the token table (not the other tables)
        head = p["lm_head_W"] * (wb["embedding"] / p["embedding"]) if p["embedding"] else 0
    w = wb["total"] - wb["embedding"] - wb["lm_head"] - routed + routed_read + head + B * emb_row_bytes
    kv = B * kv_bytes(kv_elems(p, max(L - 1, 0)), kvfmt)
    st = B * 2 * state_bytes * state_elems(p)
    return {"weights": w, "kv": kv, "state": st, "bytes": w + kv + st}


def breakdown(p, raw=None, hw_list=None):
    """The detailed record of one plan (the arch_breakdown.json per-repo entry, without prov / valid)."""
    h = p["hidden"]
    norm_kind = p.get("norm_kind") or ((AM.classify(raw) or {}).get("norm", "RMSNorm") if raw is not None else "RMSNorm")
    prm = params_breakdown(p)
    order = layer_order(p)
    tensors = all_tensors(p)
    wb = {fmt: weight_bytes(p, fmt, tensors) for fmt in WEIGHT_FORMATS}
    ctx = contexts_for(p)
    st = state_elems(p)
    out = {"max_context": int(p["max_context"]), "hidden": h, "vocab": p["vocab"], "layers": p["layers"],
           "layer_order": order is not None, "params": prm}
    if order is not None:
        out["per_layer"] = [r for r in order]
    out["weights"] = {fmt: wb[fmt]["total"] for fmt in WEIGHT_FORMATS}
    try:
        out["gguf"] = gguf_sizes(p, GGUF_LEVELS + ("BF16",))
    except Exception as e:                                   # a tensor shape llama-quantize cannot take
        out["gguf"] = None
        out.setdefault("notes", []).append(f"GGUF sizes unavailable: {e}"[:200])
    out["kv_per_token"] = {"elems": kv_elems(p, 1), "by_kind": kv_by_kind(p, 1)}
    out["kv_per_token"].update({fmt: kv_bytes(out["kv_per_token"]["elems"], fmt) for fmt in KV_FORMATS})
    out["state"] = {"elems": st, "bf16": 2 * st, "fp32": 4 * st}
    out["flops"], out["elementwise"], out["kv"], out["activation"] = {}, {}, {}, {}
    out["serving"], out["bandwidth"] = {}, {}
    emb_row = 2 * (p["lm_head_W"] // p["vocab"] if p["vocab"] and p["lm_head_W"] else h)
    for key, L in ctx.items():
        out["flops"][key] = flops_at(p, L)
        out["elementwise"][key] = elementwise_at(p, L, norm_kind)
        ke = kv_elems(p, L)
        out["kv"][key] = {"L": L, "elems": ke, "by_kind": kv_by_kind(p, L),
                          **{fmt: kv_bytes(ke, fmt) for fmt in KV_FORMATS}}
        act_f = activation_bytes(p, L, "flash")
        out["activation"][key] = {"flash": act_f, "eager": activation_bytes(p, L, "eager")}
        out["serving"][key] = {str(B): wb["bf16"]["total"] + B * (2 * ke + 2 * st) + act_f for B in BATCHES}
        dfl = out["flops"][key]["decode"]["total"]
        bw = {}
        for B in BATCHES:
            r = decode_read(p, L, B, wb["bf16"], emb_row)
            r["flops"] = B * dfl
            r["B"] = B
            r["intensity"] = r["flops"] / r["bytes"] if r["bytes"] else None
            bw[str(B)] = r
        out["bandwidth"][key] = bw
    hws = hw_list if hw_list is not None else hardware().get("hardware", [])
    if hws:
        rf = {}
        for hw in hws:
            d = {}
            for key in ROOFLINE_L:
                L = ctx[key]
                d[key] = {"decode": {B: decode_bound(out["bandwidth"][key][str(B)], hw) for B in map(str, BATCHES)},
                          "prefill_s": prefill_bound(out["flops"][key]["prefill"]["total"], wb["bf16"]["total"], hw)}
            rf[hw["id"]] = d
        out["roofline"] = rf
    return out


def breakdown_checks(p, bd=None):
    """[(what, breakdown value, arch_metrics value)] of every sum that must equal the plan's own totals."""
    bd = bd or breakdown(p, hw_list=[])
    prm = AM.params(p)
    bad = []

    def eq(what, a, b, tol=0):
        if (a != b) if not tol else abs(a - b) > tol * max(1, abs(b)):
            bad.append((what, a, b))
    eq("params.total", bd["params"]["total"], prm["total"])
    eq("params.active", bd["params"]["active_total"], prm["active"])
    eq("params.groups", sum(g["np"] * g["layer_total"] for g in bd["params"]["groups"]) + sum(
        v for r, v in bd["params"]["components"].items() if r in ("embedding", "embedding.other", "lm_head",
                                                                   "final_norm", "other", "other.linear")),
       prm["total"])
    for key, L in contexts_for(p).items():
        f = bd["flops"][key]
        ref_d = AM.flops(p, L, "decode")
        eq(f"flops.decode[{key}]", f["decode"]["total"], ref_d["total"])
        mm = sum(v for r, v in f["decode"]["components"].items() if not r.startswith("attn.scores")
                 and r not in ("attn.av", "attn.indexer", "rec.core"))
        eq(f"flops.matmul[{key}]", mm, ref_d["matmul"])
        ref_p = AM.flops(p, L, "prefill")
        eq(f"flops.prefill[{key}]", round(f["prefill"]["per_token"]), ref_p["total"], 1e-12)
        eq(f"kv[{key}]", bd["kv"][key]["elems"], sum(AM._kv_headline(p, L).values()))
    return bad


# ------------------------------------------------------------------------------------------------ the whole file

STATUS = ("symbolic-proved", "trace-verified", "hub-matched", "size-checked", "measured", "formula-only",
          "approximation", "bound", "failed")
VALID_GROUPS = ("params.components", "params.per_layer", "flops.components", "flops.elementwise", "flops.training",
                "kv", "state", "weights.dense", "weights.quantised", "weights.gguf", "activation", "serving",
                "bandwidth", "roofline")
TRACE_FILES = {k: os.path.join(HERE, "data", f"arch_breakdown_trace_{k}.json")
               for k in ("params", "flops", "symbolic", "memory")}
METRICS = os.path.join(HERE, "data", "arch_metrics.json")
MEM_TOL = 0.15                       # an activation check passes when the eager prediction is within 15% of measured


def weakest(*st):
    return max(st, key=lambda x: STATUS.index(x) if x in STATUS else len(STATUS))


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def validation_sources():
    """The trace / Hub / measurement files the per-repo statuses read."""
    tr = {k: _load(v, {}) for k, v in TRACE_FILES.items()}
    params_of = {r: rec for rec in tr["params"].values() for r in rec.get("repos", [])}
    flops_ok = {mt: bool(r.get("checked")) and all(c.get("all_equal") for c in r["checked"])
                for mt, r in tr["flops"].items()}
    sym_ok = {mt: bool((r.get("proof") or {}).get("all_equal")) for mt, r in tr["symbolic"].items()}
    mem_ok = {mt: all(abs(x["ratio"] - 1) <= MEM_TOL for x in r.get("runs", []) if x["impl"] == "eager")
              for mt, r in tr["memory"].items() if r.get("runs")}
    hub = _load(os.path.join(HERE, "data", "arch_breakdown_hub.json"), {})
    am = _load(METRICS, {})
    return {"params_of": params_of, "flops_ok": flops_ok, "sym_ok": sym_ok, "mem_ok": mem_ok, "hub": hub,
            "am": am.get("repos", {})}


def validate(name, raw, p, src):
    """{group: status} and notes of one repo's breakdown."""
    mt = str(raw.get("model_type"))
    am = (src["am"].get(name) or {}).get("valid") or {}
    v, notes = {}, []
    pt = src["params_of"].get(name)
    if pt and "error" not in pt and pt.get("all_buckets") and pt.get("all_subs"):
        v["params.components"] = "trace-verified"
        v["params.per_layer"] = "trace-verified" if pt.get("layers_equal") else "formula-only"
    elif pt and "error" not in pt and "buckets" in pt:
        v["params.components"] = v["params.per_layer"] = "failed"
        bad = {k: x for k, x in pt["buckets"].items() if x[0] != x[1]}
        notes.append(f"per-module parameter trace differs: {bad or {k: x for k, x in pt['subs'].items() if x[0] != x[1]}}")
    else:
        v["params.components"] = v["params.per_layer"] = "formula-only"
    if layer_order(p) is None:
        v["params.per_layer"] = "formula-only"
        notes.append("layer order not modelled: the per-layer table lists the layer kinds")
    if src["flops_ok"].get(mt) and AM.stock_arch(raw):
        f = "symbolic-proved" if src["sym_ok"].get(mt) else "trace-verified"
        v["flops.components"] = weakest(f, am.get("flops", "formula-only")) if am.get("flops") == "formula-only" \
            else f
    else:
        v["flops.components"] = "formula-only"
    v["flops.elementwise"] = "formula-only"
    v["flops.training"] = "approximation"
    v["kv"] = am.get("kv", "formula-only")
    v["state"] = am.get("state", "formula-only")
    v["weights.dense"] = am.get("params", "formula-only")
    base = v["weights.dense"] if v["weights.dense"] != "failed" else "failed"
    v["weights.quantised"] = weakest(base, "size-checked")
    v["weights.gguf"] = weakest(base, "size-checked")
    v["activation"] = "measured" if src["mem_ok"].get(mt) and AM.stock_arch(raw) else "formula-only"
    v["serving"] = weakest(v["weights.dense"], v["kv"], v["state"], v["activation"])
    v["bandwidth"] = weakest(am.get("active", "formula-only"), v["kv"], v["state"])
    v["roofline"] = "bound"
    return v, notes


def build(models=None, configs=None, out=OUT, metrics_path=None):
    from graph.frontier import load
    from graph.model_graph import ZOO_SNAPSHOT, iter_json_configs
    models = models if models is not None else load()
    configs = configs if configs is not None else dict(iter_json_configs(os.path.join(HERE, "hub_10k.json")))
    hw = hardware()
    src = validation_sources()
    repos = {}
    coverage = {g: {s: 0 for s in STATUS} for g in VALID_GROUPS}
    for m in models:
        name = m["name"]
        raw = configs[name]
        try:
            p = plan_ordered(raw)
            why = (((src["am"].get(name) or {}).get("params") or {}).get("hub_delta_reason") or "")
            if p["tied"] and "tied head stored" in why:
                p["head_stored"] = True                      # the checkpoint stores the head: GGUF output.weight
            rec = breakdown(p, raw)
        except Exception as e:                               # a config the formulas cannot read
            repos[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
            continue
        rec["valid"], nt = validate(name, raw, p, src)
        rec["notes"] = rec.get("notes", []) + nt
        for g, s in rec["valid"].items():
            coverage[g][s] += 1
        repos[name] = rec
    hub = src["hub"]
    data = {"schema_version": SCHEMA_VERSION, "generated": datetime.date.today().isoformat(),
            "snapshot": ZOO_SNAPSHOT, "contexts": CONTEXTS, "batches": list(BATCHES), "kv_formats": KV_FORMATS,
            "weight_formats": list(WEIGHT_FORMATS), "gguf_levels": list(GGUF_LEVELS), "roofline_contexts": list(ROOFLINE_L),
            "hardware": hw.get("hardware", []), "conventions": CONVENTIONS,
            "hub_checks": {"gguf": (hub.get("gguf") or {}).get("summary", {}).get("by_level"),
                           "gguf_all": (hub.get("gguf") or {}).get("summary", {}).get("all_quantised"),
                           "safetensors": (hub.get("safetensors") or {}).get("summary", {}).get("by_method")},
            "coverage": coverage, "repos": repos}
    if out:
        with gzip.open(out, "wt") as f:
            json.dump(data, f, separators=(",", ":"))
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    d = build()
    print(f"wrote {OUT}: {len(d['repos'])} repos")
