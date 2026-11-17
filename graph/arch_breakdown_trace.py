"""Trace checks of graph/arch_breakdown against the real (stock) transformers code, per module group.

  params   every distinct stock config of the zoo, instantiated from its exact config on the `meta` device (no weights,
           no download; graph/arch_trace.build_meta): each LM parameter is put in a module bucket by its name
           (the global embedding / head / rest; per layer the mixer (attention or recurrent), the FFN (dense or MoE)
           and the layer norms; inside them q / k / v / o, the routed experts, the router and the shared experts),
           and the bucket sums must equal the breakdown's roles mapped to the same buckets, for the whole model and
           per layer where the layer order is known.
  flops    tiny real-shaped configs (graph/arch_trace.tiny_config), a CPU fp32 forward under FlopCounterMode: the
           per-module counts (attention projections, the attention core = the attention module minus its linear
           children and rotary modules, the dense FFN, the routed experts, the router, the shared experts, the
           recurrent mixer, the LM head) must equal the breakdown's components in the trace convention (dense
           prefill scores, the code's matmuls) at L = 5 and 13, prefill and decode.
  symbolic the same per-module split read off torch.export's graph with a symbolic length (graph/arch_trace
           symbolic_prefill, bucketed by each node's nn_module_stack): per-bucket polynomials in L.
  memory   peak prefill activations of tiny real-shaped bf16 models on one GPU (torch.cuda.max_memory_allocated
           minus weights and cache) against arch_breakdown.activation_bytes, eager and SDPA.

Remote code is never executed: only model types with a stock class are traced.

Run:  python -m graph.arch_breakdown_trace params      # graph/data/arch_breakdown_trace_params.json
      python -m graph.arch_breakdown_trace flops       # graph/data/arch_breakdown_trace_flops.json
      python -m graph.arch_breakdown_trace symbolic    # graph/data/arch_breakdown_trace_symbolic.json
      CUDA_VISIBLE_DEVICES=1 python -m graph.arch_breakdown_trace memory   # ..._memory.json
"""
from __future__ import annotations

import json
import tempfile
import os
import re
import sys
import time

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {k: os.path.join(HERE, "data", f"arch_breakdown_trace_{k}.json") for k in ("params", "flops", "symbolic",
                                                                                 "memory")}

# ------------------------------------------------------------------------------------------------ buckets

LAYER_RX = re.compile(r"(?:^|\.)(?:layers|h|blocks|layer|block|decoder_layers)\.(\d+)\.(.*)$")
MIXER_SEG = ("self_attn", "attn", "attention", "self_attention", "multi_head_attention", "temporal_block", "mamba",
             "mixer", "linear_attn", "conv", "linear", "time_mix", "attention_block")
FFN_SEG = ("mlp", "feed_forward", "block_sparse_moe", "ffn", "mlp_block", "fc1", "fc2", "shared_mlp", "moe",
           "c_fc", "channel_mix", "experts")
WRAP_SEG = ("shared_transformer", "shared_transf", "mamba_decoder")     # Zamba: blocks wrapping a mixer / FFN / norms
EMB_MARK = ("embed", "wte", "wpe", "word_embeddings", "tokens_embed", "positions_embed", "embeddings")
HEAD_MARK = ("lm_head", "embed_out")
Q_NAMES = ("q_proj", "query", "wq", "q")
K_NAMES = ("k_proj", "key", "wk", "k")
V_NAMES = ("v_proj", "value", "wv", "v")
O_NAMES = ("o_proj", "out_proj", "dense", "c_proj", "wo", "o", "output")
QKV_NAMES = ("qkv_proj", "c_attn", "query_key_value", "wqkv", "qkv", "w_pack")
GATE_NAMES = ("g_proj", "output_gate", "gate", "gate_proj")
ATTN_SEG = ("self_attn", "attn", "attention", "self_attention", "multi_head_attention", "temporal_block",
            "attention_block")


def hf_bucket(name, attn_layers=None, moe_layers=None):
    """(layer or None, bucket, sub-bucket or None) of an LM parameter name. attn_layers: the layers whose mixer is
    softmax attention (q / k / v / o sub-buckets are read there only; None = everywhere). Conventions where the
    breakdown groups a tensor differently from the module tree: a block's LoRA adapters (Zamba-2) are the hybrid
    layer's glue (mixer); RWKV's pre_ln (layer 0 only) is a global; JetMoE's MoE input / output linears are the
    routed experts."""
    m = LAYER_RX.search(name)
    if m and m.group(2).split(".")[0] == "pre_ln":
        return None, "global.misc", None
    if name.endswith("dense_bias"):                         # GPT-NeoX-Japanese: the last layer's bias (a global)
        return None, "global.misc", None
    if not m:
        low = name.lower()
        if low.endswith("lm_head.bias") or low.endswith("head.bias"):
            return None, "global.misc", None                # an LM-head bias is a vector (role 'other')
        if low.startswith("transformer.w."):                # CTRL's token table
            return None, "global.embedding", None
        if any(x in low for x in HEAD_MARK) or low.split(".")[-2:-1] == ["head"]:
            return None, "global.lm_head", None
        if any(x in low for x in EMB_MARK) and "norm" not in low and "layernorm" not in low:
            return None, "global.embedding", None
        return None, "global.misc", None
    i, rest = int(m.group(1)), m.group(2)
    segs = rest.split(".")
    while segs and segs[0] in WRAP_SEG:
        segs = segs[1:]
    f = segs[0] if segs else ""
    leaf = segs[-1] if segs else ""
    is_bias = leaf == "bias" or leaf.endswith("_bias")
    if "adapter" in rest:
        return i, "mixer", None
    if "act_fn" in segs or "act" in segs:                   # learned activation scalars (Apertus XIELU): 'norm'
        return i, "norm", None
    if f in MIXER_SEG:
        sub = None
        if len(segs) >= 2 and not is_bias and f in ATTN_SEG and (attn_layers is None or i in attn_layers):
            lin = segs[2].lower() if len(segs) >= 4 and segs[1] == "attention" else segs[1].lower()   # GPT-Neo
            if lin in Q_NAMES:
                sub = "attn.q"
            elif lin in K_NAMES:
                sub = "attn.k"
            elif lin in V_NAMES:
                sub = "attn.v"
            elif lin in O_NAMES:
                sub = "attn.o"
            elif lin in QKV_NAMES:
                sub = "attn.qkv"
            elif lin in GATE_NAMES:
                sub = "attn.gate"
        return i, "mixer", sub
    if f in FFN_SEG:
        path = ".".join(segs).lower()
        sub = None
        if (("experts" in segs or "experts" in path.split(".")) and "shared" not in path and not is_bias) or (
                f in ("mlp", "block_sparse_moe") and len(segs) >= 2 and segs[1] in ("input_linear", "output_linear")
                and not is_bias):
            sub = "moe.routed"
        elif "shared" in path and "experts" not in path.split(".")[:1]:
            sub = "moe.shared"
        elif len(segs) >= 2 and segs[1] in ("gate", "router", "gate_inp", "wg") and not is_bias:
            if not (len(segs) >= 3 and segs[2] not in ("weight", "layer", "wg")):
                sub = "moe.router"
        if moe_layers is not None and i not in moe_layers:
            sub = None                                      # a dense layer's FFN (Granite-4 shared_mlp)
        return i, "ffn", sub
    if "norm" in f or f.startswith("ln") or f.endswith("_residual") or f in ("pre_ln",):
        return i, "norm", None
    return i, "layer.misc", None


def bd_bucket(role):
    """The hf_bucket (bucket, sub) a breakdown role lands in."""
    if role in ("embedding", "embedding.other"):
        return "global.embedding", None
    if role == "lm_head":
        return "global.lm_head", None
    if role in ("final_norm", "other", "other.linear"):
        return "global.misc", None
    if role.startswith("attn.") or role.startswith("rec."):
        sub = role if role in ("attn.q", "attn.k", "attn.v", "attn.o", "attn.gate") else None
        return "mixer", sub
    if role.startswith("mlp.") or role.startswith("moe."):
        if role.startswith("moe.routed"):
            return "ffn", "moe.routed"
        if role.startswith("moe.shared"):
            return "ffn", "moe.shared"
        if role == "moe.router":
            return "ffn", "moe.router"
        return "ffn", None
    if role == "norm":
        return "norm", None
    return "layer.misc", None


def attn_layer_set(p):
    """The layers whose groups hold a softmax attention, or None when the order is unknown."""
    from graph import arch_breakdown as BD
    rows = BD.layer_order(p)
    if rows is None:
        return None if not any(g["rec"] for g in p["groups"]) else set()
    return {i for i, gis in enumerate(rows) if any(p["groups"][gi]["attn"] is not None for gi in gis)}


def moe_layer_set(p):
    from graph import arch_breakdown as BD
    rows = BD.layer_order(p)
    if rows is None:
        return None if any(g["moe"] is not None for g in p["groups"]) else set()
    return {i for i, gis in enumerate(rows) if any(p["groups"][gi]["moe"] is not None for gi in gis)}


def trace_buckets(model, attn_layers=None, moe_layers=None):
    """{bucket: n}, {sub: n}, {(layer, bucket): n} of a model's LM parameters (tied parameters once)."""
    from graph.arch_trace import NON_LM_MARKERS
    tot, sub, per = {}, {}, {}
    for name, prm in model.named_parameters():
        if any(mk in name.lower() for mk in NON_LM_MARKERS):
            continue
        i, b, s = hf_bucket(name, attn_layers, moe_layers)
        n = prm.numel()
        tot[b] = tot.get(b, 0) + n
        if s:
            sub[s] = sub.get(s, 0) + n
        if i is not None:
            per[(i, b)] = per.get((i, b), 0) + n
    if sub.get("attn.qkv"):
        sub["attn.qkv"] += sum(sub.pop(k, 0) for k in ("attn.q", "attn.k", "attn.v"))
    return tot, sub, per


def formula_buckets(p):
    """The same three maps from the breakdown (per layer: each layer's groups' tensors)."""
    from graph import arch_breakdown as BD
    tot, sub, per = {}, {}, {}

    def add(d, k, v):
        d[k] = d.get(k, 0) + v
    for t in BD.global_tensors(p):
        b, _ = bd_bucket(t["role"])
        add(tot, b, t["elems"])
    order = BD.layer_order(p)
    for gi, g in enumerate(p["groups"]):
        for t in BD.group_tensors(g, p["hidden"]):
            b, s = bd_bucket(t["role"])
            add(tot, b, g["np"] * t["elems"])
            if s:
                add(sub, s, g["np"] * t["elems"])
    if order is not None:
        for i, gis in enumerate(order):
            for gi in gis:
                for t in BD.group_tensors(p["groups"][gi], p["hidden"]):
                    b, _ = bd_bucket(t["role"])
                    add(per, (i, b), t["elems"])
    return tot, sub, per


def check_params(raw):
    """Per-bucket parameter counts of the stock model on the meta device vs the breakdown."""
    from graph.arch_breakdown import plan_ordered
    from graph.arch_trace import build_meta
    t0 = time.time()
    m = build_meta(raw)
    p = plan_ordered(raw)
    T, S, P = trace_buckets(m, attn_layer_set(p), moe_layer_set(p))
    F, FS, FP = formula_buckets(p)
    if FS.get("attn.gate") and not S.get("attn.gate"):          # the output gate fused into q_proj (Qwen3-Next)
        FS["attn.q"] = FS.get("attn.q", 0) + FS.pop("attn.gate")
    has_qkv = "attn.qkv" in S
    if has_qkv:
        FS["attn.qkv"] = sum(FS.pop(k, 0) for k in ("attn.q", "attn.k", "attn.v"))
    buckets = {}
    for b in sorted(set(T) | set(F)):
        buckets[b] = [F.get(b, 0), T.get(b, 0)]
    subs = {}
    for s in sorted(set(S) | set(FS)):
        subs[s] = [FS.get(s, 0), S.get(s, 0)]
    layers_equal = None
    bad_layers = []
    tl = {i for i, _ in P}
    tied_blocks = any(g["np"] != g["n"] for g in p["groups"])        # a block shared by several layers (Zamba)
    if FP and tl and max(tl) + 1 == p["layers"] and not tied_blocks:
        layers_equal = True
        for k in sorted(set(P) | set(FP)):
            if P.get(k, 0) != FP.get(k, 0):
                layers_equal = False
                if len(bad_layers) < 6:
                    bad_layers.append([k[0], k[1], FP.get(k, 0), P.get(k, 0)])
    eq = lambda d: {k: v[0] == v[1] for k, v in d.items()}
    return {"buckets": buckets, "subs": subs, "bucket_equal": eq(buckets), "sub_equal": eq(subs),
            "all_buckets": all(v[0] == v[1] for v in buckets.values()),
            "all_subs": all(v[0] == v[1] for v in subs.values()),
            "total": [sum(F.values()), sum(T.values())], "layers_equal": layers_equal, "bad_layers": bad_layers,
            "seconds": round(time.time() - t0, 2)}


def _param_job(arg):
    repo, raw = arg
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    try:
        r = check_params(raw)
    except Exception as e:
        r = {"error": f"{type(e).__name__}: {e}"[:300]}
    r["model_type"] = raw.get("model_type")
    return repo, r


def run_params(items=None, out=OUT["params"], procs=4):
    """Every distinct stock config of the zoo (graph/arch_trace.run_params' dedup)."""
    from multiprocessing import get_context

    from graph.arch_metrics import stock_arch
    from graph.arch_trace import dedup_key, stock_items
    items = items if items is not None else stock_items()
    groups = {}
    for repo, raw in items:
        if stock_arch(raw):
            groups.setdefault(dedup_key(raw), []).append((repo, raw))
    jobs = [reps[0] for reps in groups.values()]
    rep_of = {reps[0][0]: [x for x, _ in reps] for reps in groups.values()}
    res = {}
    with get_context("spawn").Pool(procs) as pool:
        for k, (repo, r) in enumerate(pool.imap_unordered(_param_job, jobs, chunksize=4)):
            r["repos"] = rep_of[repo]
            res[repo] = r
            if k % 100 == 0:
                print(k, len(jobs), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=0, sort_keys=True)
    return res


# ------------------------------------------------------------------------------------------------ FLOPs per module

ROPE_MARK = ("Rotary",)


def _layer_children(model):
    """{layer index: [(relative child name, module), ...]} of the decoder layers, Zamba-style wrappers unwrapped."""
    out = {}
    for name, mod in model.named_modules():
        m = LAYER_RX.search(name + ".")
        if not m or m.group(2) == "" or "." in m.group(2).rstrip("."):
            continue
        child = m.group(2).rstrip(".")
        i = int(m.group(1))
        if child in WRAP_SEG:
            for cn, cm in mod.named_children():
                out.setdefault(i, []).append((cn, cm, f"{name}.{cn}"))
        else:
            out.setdefault(i, []).append((child, mod, name))
    return out


def module_buckets(fc, model, attn_layers=None):
    """{bucket: FLOPs} of one FlopCounterMode run: per decoder-layer child module (its subtree), the mixer split into
    its linear children (`*.proj`) and the rest (`*.core`, rotary modules excluded), the FFN split into the routed
    experts, the router, the shared experts and the rest (dense FFN); the LM head; everything else ('other')."""
    import torch
    counts = {k: sum(v.values()) for k, v in fc.get_flop_counts().items()}
    root = type(model).__name__
    key = lambda n: f"{root}.{n}" if n else root
    total = counts.get("Global", 0)
    out = {}

    def add(b, v):
        if v:
            out[b] = out.get(b, 0) + v

    def cnt(full, mod):                                     # a container never called (ModuleList): its children
        if key(full) in counts:
            return counts[key(full)]
        return sum(cnt(f"{full}.{n}", x) for n, x in mod.named_children())
    # GPT-2 Conv1D and JetMoE ParallelExperts are linear maps too
    is_lin = lambda x: isinstance(x, torch.nn.Linear) or type(x).__name__ == "Conv1D" or \
        type(x).__name__.endswith("ParallelExperts")
    seen = 0
    for i, kids in _layer_children(model).items():
        for cn, cm, full in kids:
            c = counts.get(key(full), 0)
            if not c:
                continue
            seen += c
            rope = sum(counts.get(key(f"{full}.{n}"), 0) for n, x in cm.named_modules()
                       if n and any(r in type(x).__name__ for r in ROPE_MARK))
            c -= rope
            seen -= 0
            adapt = sum(cnt(f"{full}.{n}", x) for n, x in cm.named_modules()
                        if n and "adapter" in n.split(".")[-1] and "adapter" not in ".".join(n.split(".")[:-1]))
            # (Zamba-2 LoRA adapters: the hybrid layer's glue)
            add("rec.proj", adapt)
            c -= adapt
            if cn in MIXER_SEG:
                lin = c if is_lin(cm) else sum(counts.get(key(f"{full}.{n}"), 0) for n, x in cm.named_modules()
                                               if n and is_lin(x) and "adapter" not in n)
                kind = "attn" if (cn in ATTN_SEG and (attn_layers is None or i in attn_layers)) else "rec"
                add(f"{kind}.proj", lin)
                add(f"{kind}.core", c - lin)
            elif cn in FFN_SEG:
                rest = c
                for n, x in cm.named_children():
                    v = cnt(f"{full}.{n}", x)
                    if n == "experts" or (cn in ("mlp", "block_sparse_moe") and n in ("input_linear", "output_linear")):
                        add("moe.experts", v)
                    elif n in ("gate", "router", "gate_inp"):
                        add("moe.router", v)
                    elif "shared" in n:
                        add("moe.shared", v)
                    elif "adapter" in n:
                        continue                            # counted as glue above
                    else:
                        continue
                    rest -= v
                add("mlp", rest)
            else:
                add("layer.other", c)
    lm = sum(counts.get(key(n), 0) for n, x in model.named_modules()
             if n.split(".")[-1] in ("lm_head", "embed_out", "output_projection") or n == "head")
    add("lm_head", lm)
    rope_top = sum(counts.get(key(n), 0) for n, x in model.named_modules()
                   if n and any(r in type(x).__name__ for r in ROPE_MARK) and not LAYER_RX.search(n + "."))
    add("other", total - seen - lm - rope_top)
    return out


def formula_module_buckets(p, L, mode):
    """The breakdown's components in the trace convention, mapped to module_buckets: a prefill of L tokens (total)
    or one decode step at position L (the cache holds L - 1 tokens)."""
    from graph import arch_breakdown as BD
    from graph import arch_metrics as AM
    h = p["hidden"]
    T = L if mode == "prefill" else 1
    out = {}

    def add(b, v):
        if v:
            out[b] = out.get(b, 0) + v
    for g in p["groups"]:
        for t in BD.group_tensors(g, h):
            if t["kind"] not in ("linear", "lump"):
                continue
            r = t["role"]
            w = t["active"]
            if r.startswith("moe.routed") and g["moe"].get("dense_compute"):
                w = t["elems"]
            if r.startswith("attn."):
                b = "attn.proj"
            elif r.startswith("rec."):
                b = "rec.proj"
                w -= g["rec"].get("W_raw", 0)
            elif r.startswith("moe.routed"):
                b = "moe.experts"
            elif r == "moe.router":
                b = "moe.router"
            elif r.startswith("moe.shared") or r == "moe.latent":
                b = "moe.shared"
            else:
                b = "mlp"
            add(b, g["n"] * 2 * w * T)
        a = g["attn"]
        if a is not None:
            per_key = 2 * a["nh"] * (a["dqk"] + a["dv"])
            W = g["window"]
            if p["mt"] == "recurrent_gemma":                # a fixed W-slot buffer (arch_metrics.trace_attn)
                add("attn.core", g["n"] * per_key * W * T)
            elif mode == "prefill":
                add("attn.core", g["n"] * per_key * L * L)
            elif p["mt"] in AM.NO_CACHE_TYPES:
                add("attn.core", g["n"] * per_key)
            else:
                add("attn.core", g["n"] * per_key * (min(L, W) if W else L))
            if p["mt"] == "doge" and mode == "decode":      # dt_proj over the whole cache (arch_metrics.trace_attn)
                add("attn.proj", g["n"] * 2 * a.get("extra_W", 0) * (L - 1))
    tc = AM.trace_core(p, L if mode == "prefill" else L, mode)
    if tc is not None:
        add("rec.core", tc)
    add("lm_head", 2 * p["lm_head_W"] * T)
    add("other", 2 * p["extra_W"] * T)
    return out, tc is not None


def check_flops(raw, Ls=(5, 13)):
    """Per-module FLOPs of a tiny real-shaped config (prefill and decode at each L) vs the breakdown."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from graph.arch_breakdown import plan_ordered
    from graph.arch_trace import TINY_VOCAB, _cpu_sdpa, _hybrid_cache, build_cpu, tiny_config
    t0 = time.time()
    tiny = tiny_config(raw)
    p = plan_ordered(tiny)
    m = build_cpu(tiny)
    al = attn_layer_set(p)
    res = {"L": {}, "equal": {}, "core_modelled": None}
    for L in Ls:
        g = torch.Generator().manual_seed(L)
        ids = torch.randint(2, TINY_VOCAB - 1, (1, L), generator=g)
        kw0 = {}
        cache = _hybrid_cache(m)
        if cache is not None:
            kw0["past_key_values"] = cache
        with torch.no_grad():
            with FlopCounterMode(display=False, custom_mapping=_cpu_sdpa()) as fc:
                out = m(input_ids=ids, use_cache=True, **kw0)
            pre = module_buckets(fc, m, al)
            kw = {}
            for a in ("past_key_values", "cache_params", "state"):
                if getattr(out, a, None) is not None:
                    kw[a] = getattr(out, a)
            nxt = torch.randint(2, TINY_VOCAB - 1, (1, 1), generator=g)
            with FlopCounterMode(display=False, custom_mapping=_cpu_sdpa()) as fc:
                m(input_ids=nxt, use_cache=True, cache_position=torch.tensor([L]), **kw)
            dec = module_buckets(fc, m, al)
        fp, core_ok = formula_module_buckets(p, L, "prefill")
        fd, _ = formula_module_buckets(p, L + 1, "decode")
        res["core_modelled"] = core_ok
        rec = {}
        for mode, tr, fo in (("prefill", pre, fp), ("decode", dec, fd)):
            for b in sorted(set(tr) | set(fo)):
                if b == "rec.core" and not core_ok:
                    continue
                rec[f"{mode}:{b}"] = [fo.get(b, 0), tr.get(b, 0)]
                res["equal"].setdefault(b, True)
                res["equal"][b] &= fo.get(b, 0) == tr.get(b, 0)
        res["L"][L] = rec
    res["all_equal"] = all(res["equal"].values())
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _flops_job(arg):
    mt, reps = arg
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import torch
    torch.set_num_threads(2)
    rec = {"checked": [], "errors": []}
    for repo, raw in reps[:6]:
        if len(rec["checked"]) >= 2:
            break
        try:
            r = check_flops(raw)
            r["repo"] = repo
            rec["checked"].append(r)
        except Exception as e:
            rec["errors"].append({"repo": repo, "error": f"{type(e).__name__}: {e}"[:300]})
    return mt, rec


def run_flops(items=None, out=OUT["flops"], procs=4, only_types=None):
    """Up to two configs per stock model type (graph/arch_trace._by_type order), as the forward traces."""
    from multiprocessing import get_context

    from graph.arch_trace import _by_type, stock_items
    items = items if items is not None else stock_items()
    jobs = [(mt, reps) for mt, reps in sorted(_by_type(items).items()) if not only_types or mt in only_types]
    res = {}
    with get_context("spawn").Pool(procs) as pool:
        for mt, rec in pool.imap_unordered(_flops_job, jobs):
            res[mt] = rec
            ok = [c["all_equal"] for c in rec["checked"]]
            print(mt, ok, [e["error"][:80] for e in rec["errors"]][:1], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=0, sort_keys=True, default=str)
    return res


# ------------------------------------------------------------------------------------------------ activation memory

MEM_L = (512, 2048)


def check_memory(raw, Ls=MEM_L, impls=("eager", "sdpa")):
    """Peak prefill memory of a tiny real-shaped bf16 model on the visible GPU: (max_memory_allocated - the weights)
    against arch_breakdown.activation_bytes + the KV cache the prefill writes (all layers, bf16). One sequence,
    no grad, logits for the last position (logits_to_keep = 1, as generate does)."""
    import copy as _copy

    import torch

    from graph import arch_breakdown as BD
    from graph.arch_metrics import _stock_config_class
    from graph.arch_trace import TINY_VOCAB, _hybrid_cache, _torch_paths, model_class, tiny_config
    tiny = tiny_config(raw)
    for k in ("max_position_embeddings", "n_positions", "n_ctx"):    # learned positions: room for the longest L
        if isinstance(tiny.get(k), int) and tiny[k] < max(Ls):
            tiny[k] = max(Ls)
    p = BD.plan_ordered(tiny)
    out = {"tiny": {k: tiny.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                            "num_key_value_heads", "head_dim", "intermediate_size")}, "runs": []}
    for impl in impls:
        hc = _stock_config_class(tiny["model_type"]).from_dict(_copy.deepcopy(tiny))
        hc._attn_implementation = impl
        cls = model_class(tiny, hc)
        torch.manual_seed(0)
        m = cls._from_config(hc, torch_dtype=torch.bfloat16).to("cuda").eval()
        _torch_paths(m)
        with torch.no_grad():                               # warm-up: cuBLAS workspaces and kernels allocate once
            kw = {}
            cache = _hybrid_cache(m)
            if cache is not None:
                kw["past_key_values"] = cache
            m(input_ids=torch.randint(2, TINY_VOCAB - 1, (1, 16), device="cuda"), use_cache=True, logits_to_keep=1,
              **kw)
        for L in Ls:
            ids = torch.randint(2, TINY_VOCAB - 1, (1, L), device="cuda")
            kw = {}
            cache = _hybrid_cache(m)
            if cache is not None:
                kw["past_key_values"] = cache
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            with torch.no_grad():
                o = m(input_ids=ids, use_cache=True, logits_to_keep=1, **kw)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            kv = 2 * BD.kv_elems(p, L)
            pred = BD.activation_bytes(p, L, "eager" if impl == "eager" else "flash") + kv
            out["runs"].append({"impl": impl, "L": L, "measured": int(peak), "predicted": int(pred),
                                "kv": int(kv), "ratio": pred / peak if peak else None})
            del o
        del m
        torch.cuda.empty_cache()
    return out


MEM_TYPES = ("llama", "qwen3", "qwen2", "mistral", "gemma2", "gemma3", "phi3", "olmo2", "granite", "mixtral",
             "qwen3_moe", "qwen2_moe", "deepseek_v3", "gpt2", "gpt_neox", "falcon", "starcoder2", "cohere", "smollm3",
             "glm4", "exaone4", "gpt_oss")


def _memory_job(arg):
    repo, raw = arg
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    try:
        r = check_memory(raw)
        r["repo"] = repo
        return r
    except Exception as e:
        return {"repo": repo, "error": f"{type(e).__name__}: {e}"[:300]}


def run_memory(out=OUT["memory"], types=MEM_TYPES):
    """One config per type (the most downloaded stock one that runs), on the visible GPU, each in its own process
    (a device-side assert poisons the CUDA context)."""
    from multiprocessing import get_context

    from graph.arch_trace import _by_type, stock_items
    res = {}
    ctx = get_context("spawn")
    for mt, reps in sorted(_by_type(stock_items()).items()):
        if mt not in types:
            continue
        for repo, raw in reps[:3]:
            with ctx.Pool(1) as pool:
                r = pool.apply(_memory_job, ((repo, raw),))
            if "error" in r:
                print(mt, repo, "ERR", r["error"][:150], flush=True)
                continue
            res[mt] = r
            print(mt, repo, [(x["impl"], x["L"], round(x["ratio"], 3)) for x in r["runs"]], flush=True)
            break
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return res


# ------------------------------------------------------------------------------------------------ symbolic, per module

def _stack_bucket(stack, attn_layers=None):
    """The module_buckets bucket of an exported node from its nn_module_stack (outer -> inner (fqn, class))."""
    ents = [(str(v[0]), str(v[1])) for v in stack.values()]
    fq = [f[2:] if f.startswith("m.") else f for f, _ in ents]
    cls = [c for _, c in ents]
    if any(any(r in c for r in ROPE_MARK) for c in cls):
        return None
    for f, c in zip(reversed(fq), reversed(cls)):
        last = f.split(".")[-1]
        if last in ("lm_head", "embed_out", "output_projection") or f == "head":
            return "lm_head"
    lay = next((LAYER_RX.search(f + ".") for f in fq if LAYER_RX.search(f + ".")), None)
    if lay is None:
        return "other"
    i = int(lay.group(1))
    rest = [f for f in fq if LAYER_RX.search(f + ".") and LAYER_RX.search(f + ".").group(2).rstrip(".")]
    if not rest:
        return "layer.other"
    segs = LAYER_RX.search(rest[-1] + ".").group(2).rstrip(".").split(".")
    while segs and segs[0] in WRAP_SEG:
        segs = segs[1:]
    if not segs:
        return "layer.other"
    if any("adapter" in s for s in segs):
        return "rec.proj"
    f0 = segs[0]
    tail = [c.rstrip("'>").split(".")[-1] for c in cls[-1:]]
    is_lin = any(t.endswith("Linear") or t == "Conv1D" or t.endswith("ParallelExperts") for t in tail)
    if f0 in MIXER_SEG:
        kind = "attn" if (f0 in ATTN_SEG and (attn_layers is None or i in attn_layers)) else "rec"
        return f"{kind}.proj" if is_lin else f"{kind}.core"
    if f0 in FFN_SEG:
        n = segs[1] if len(segs) > 1 else ""
        if n == "experts" or (f0 in ("mlp", "block_sparse_moe") and n in ("input_linear", "output_linear")):
            return "moe.experts"
        if n in ("gate", "router", "gate_inp"):
            return "moe.router"
        if "shared" in n:
            return "moe.shared"
        return "mlp"
    return "layer.other"


def symbolic_buckets(model, attn_layers=None, vocab=None, example_len=17):
    """torch.export a prefill with a symbolic length; {bucket: sympy FLOPs} by each node's nn_module_stack."""
    import torch
    import torch.fx
    from torch.export import Dim, export
    from torch.utils.flop_counter import flop_registry

    from graph.arch_trace import TINY_VOCAB, _cpu_sdpa, _sym

    class Prefill(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):
            out = self.m(input_ids=ids, use_cache=True)
            pkv = out.past_key_values
            return (out.logits, *[t for lay in pkv.layers for t in (lay.keys, lay.values)])

    ids = torch.randint(2, (vocab or TINY_VOCAB) - 1, (1, example_len))
    ep = export(Prefill(model), (ids,), dynamic_shapes={"ids": {1: Dim.DYNAMIC}}, strict=False)
    ep = ep.run_decompositions()
    val = lambda a: a.meta.get("val") if isinstance(a, torch.fx.Node) else a
    reg = {**flop_registry}
    out, syms = {}, set()
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        fn = reg.get(getattr(node.target, "overloadpacket", None))
        if fn is None:
            continue
        f = fn(*torch.fx.node.map_arg(node.args, val), out_val=node.meta.get("val"),
               **torch.fx.node.map_arg(node.kwargs, val))
        b = _stack_bucket(node.meta.get("nn_module_stack", {}), attn_layers)
        if b is None:
            continue
        out[b] = out.get(b, 0) + f
    res = {k: _sym(v) for k, v in out.items()}
    for e in res.values():
        syms |= e.free_symbols
    rng = None
    sym = syms.pop() if len(syms) == 1 else None
    for k, v in ep.range_constraints.items():
        if sym is not None and str(k) == str(sym):
            rng = (int(v.lower), None if str(v.upper) in ("oo", "int_oo") else int(v.upper))
    return res, sym, rng


def check_symbolic(raw):
    """Per-bucket FLOP polynomials of the exported prefill vs the breakdown's components at a symbolic L, per window
    regime (as graph/arch_trace.check_symbolic does for the totals)."""
    import sympy

    from graph.arch_breakdown import plan_ordered
    from graph.arch_trace import build_cpu, tiny_config
    t0 = time.time()
    tiny = tiny_config(raw)
    p = plan_ordered(tiny)
    m = build_cpu(tiny)
    al = attn_layer_set(p)
    W = min([g["window"] for g in p["groups"] if g.get("window")] or [0]) or None
    out = {"regimes": [], "equal": {}}
    for ex in ((3, 17) if W else (17,)):
        tr, s, rng = symbolic_buckets(m, al, example_len=ex)
        L = s if s is not None else sympy.Symbol("L", positive=True, integer=True)
        fo, core_ok = formula_module_buckets(p, L, "prefill")
        reg = {"range": list(rng) if rng else None, "buckets": {}}
        for b in sorted(set(tr) | set(fo)):
            if b == "rec.core" and not core_ok:
                continue
            ft, ff = sympy.expand(tr.get(b, 0)), sympy.expand(fo.get(b, 0))
            ok = bool(sympy.simplify(ft - ff) == 0)
            reg["buckets"][b] = {"trace": str(ft), "formula": str(ff), "equal": ok}
            out["equal"][b] = out["equal"].get(b, True) and ok
        out["regimes"].append(reg)
    out["all_equal"] = all(out["equal"].values())
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _sym_job(arg):
    mt, reps = arg
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import torch
    torch.set_num_threads(2)
    rec = {"errors": []}
    for repo, raw in reps[:3]:
        try:
            r = check_symbolic(raw)
            r["repo"] = repo
            rec["proof"] = r
            break
        except Exception as e:
            rec["errors"].append({"repo": repo, "error": f"{type(e).__name__}: {e}"[:300]})
    return mt, rec


def run_symbolic(items=None, out=OUT["symbolic"], procs=4, only_types=None):
    """The per-bucket proof for every stock type whose prefill exports (graph/arch_trace's symbolic set)."""
    from multiprocessing import get_context

    from graph.arch_trace import _by_type, stock_items
    items = items if items is not None else stock_items()
    try:
        with open(os.path.join(HERE, "data", "arch_trace_symbolic.json")) as f:
            have = {k for k, v in json.load(f).items() if v.get("proof")}
    except FileNotFoundError:
        have = None
    jobs = [(mt, reps) for mt, reps in sorted(_by_type(items).items())
            if (have is None or mt in have) and (not only_types or mt in only_types)]
    res = {}
    with get_context("spawn").Pool(procs) as pool:
        for mt, rec in pool.imap_unordered(_sym_job, jobs):
            res[mt] = rec
            pr = rec.get("proof")
            print(mt, pr["all_equal"] if pr else "ERR", [e["error"][:80] for e in rec["errors"]][:1], flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=0, sort_keys=True, default=str)
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "params"
    types = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
    if what == "flops" and types:                           # re-run some model types, merged into the file
        new = run_flops(only_types=types, out=os.path.join(tempfile.mkdtemp(), "arch_breakdown_trace_flops_subset.json"))
        with open(OUT["flops"]) as f:
            cur = json.load(f)
        cur.update(new)
        with open(OUT["flops"], "w") as f:
            json.dump(cur, f, indent=0, sort_keys=True, default=str)
    else:
        {"params": run_params, "flops": run_flops, "memory": run_memory, "symbolic": run_symbolic}[what]()
