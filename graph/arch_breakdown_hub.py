"""Hub-size checks of graph/arch_breakdown's weight formats: predicted file sizes against the ACTUAL sizes of
quantised checkpoints on the Hub (graph/data/quant_sizes.json, written by graph/fetch_quant_sizes.py from the HF
API's file-size metadata: nothing is downloaded).

  calibrate  the GGUF header's per-vocab term (tokenizer tokens, merges / scores, token types) and fixed part,
             fitted on the unquantised BF16 / F16 GGUF files: header = file size - the tensor data, which is exact
             for an unquantised file (arch_breakdown.gguf_sizes with no header). Written to
             graph/data/gguf_calibration.json. The quantised levels are then predicted with that header and never
             fitted.
  gguf       every quantised level of every record (Q8_0 ... Q2_K, Q4_0): predicted / actual - 1.
  safetensors  GPTQ, AWQ, FP8 (per-tensor / per-channel / block), compressed-tensors (int8, W4A16, 4-bit group 16),
             bitsandbytes NF4, MLX: the library's stored tensors per linear layer (packed weights, scales, zero
             points, g_idx) from the plan's tensor shapes, the rest at the checkpoint dtype, plus the safetensors
             JSON headers.

Run:  python -m graph.arch_breakdown_hub          # writes graph/data/gguf_calibration.json and
                                                  # graph/data/arch_breakdown_hub.json (the error statistics)
"""
from __future__ import annotations

import json
import logging
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = os.path.join(HERE, "data", "quant_sizes.json")
OUT = os.path.join(HERE, "data", "arch_breakdown_hub.json")
CAL = os.path.join(HERE, "data", "gguf_calibration.json")
QUANT_LEVELS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "Q4_0")
log = logging.getLogger(__name__)


def _plans(records):
    from graph import arch_breakdown as BD
    from graph.model_graph import iter_json_configs
    configs = dict(iter_json_configs(os.path.join(HERE, "hub_10k.json")))
    out = {}
    for r in records:
        b = r["base"]
        if b in out or b not in configs:
            continue
        try:
            out[b] = BD.plan_ordered(configs[b])
        except Exception as e:                               # a base the formulas cannot read
            log.warning("plan failed for %s: %s", b, e)
    return out


def _gguf_plan(p, rec):
    """The plan as the GGUF stores it: a tied head the checkpoint stores (and the converter writes as output.weight)
    gets its own output tensor, read off the GGUF's own parameter count (expand[]=gguf total)."""
    from graph import arch_breakdown as BD
    tot = BD.params_breakdown(p)["total"]
    g = (rec.get("gguf_meta") or {}).get("total")
    if p["tied"] and (rec.get("tied_output_duplicated") or (g and abs(g - tot - p["lm_head_W"]) < abs(g - tot))):
        return dict(p, tied=False, lm_head=p["lm_head_W"])
    return p


def stats(errs):
    a = sorted(abs(e) for e in errs)
    if not a:
        return None
    q = lambda f: a[min(len(a) - 1, int(f * len(a)))]
    return {"n": len(a), "median_abs": statistics.median(a), "p90_abs": q(0.9), "max_abs": a[-1],
            "mean": statistics.fmean(errs), "within_1pct": sum(x < 0.01 for x in a) / len(a),
            "within_2pct": sum(x < 0.02 for x in a) / len(a)}


def calibrate(data=None, write=True):
    """Fit header = fixed + bytes_per_vocab x vocab + tensor infos on the unquantised GGUF files."""
    from graph import arch_breakdown as BD
    data = data or json.load(open(SIZES))
    recs = data["gguf"]
    plans = _plans(recs)
    rows = []
    for r in recs:
        p = plans.get(r["base"])
        if p is None:
            continue
        p = _gguf_plan(p, r)
        for lv in ("BF16", "F16"):
            f = r["files"].get(lv)
            if not f:
                continue
            try:
                data_b = BD.gguf_sizes(p, (lv,), with_header=False)[lv]
            except Exception as e:
                log.warning("gguf %s %s: %s", r["repo"], lv, e)
                continue
            ts = BD.gguf_tensors(p)
            infos = sum(8 + len(t["name"]) + 4 + 8 * len(t["shape"]) + 4 + 8 for t in ts)
            rows.append({"repo": r["repo"], "base": r["base"], "level": lv, "vocab": p["vocab"],
                         "header": f["bytes"] - data_b - infos, "file": f["bytes"]})
    # robust fit: the per-vocab term is the median of (header - fixed) / vocab with fixed the median intercept
    per = [x["header"] / x["vocab"] for x in rows if x["vocab"]]
    bpv = statistics.median(per)
    fixed = statistics.median([x["header"] - bpv * x["vocab"] for x in rows])
    cal = {"bytes_per_vocab": round(bpv, 3), "fixed": int(fixed), "n_files": len(rows),
           "method": "header = file size - exact tensor data - tensor infos on the unquantised (BF16 / F16) GGUF files; "
                     "bytes_per_vocab = the median header / vocab, fixed = the median remainder",
           "residual_rel": stats([(bpv * x["vocab"] + fixed - x["header"]) / x["file"] for x in rows])}
    if write:
        with open(CAL, "w") as f:
            json.dump(cal, f, indent=1)
        BD._GC = None
    return cal, rows


def check_gguf(data=None):
    from graph import arch_breakdown as BD
    data = data or json.load(open(SIZES))
    recs = data["gguf"]
    plans = _plans(recs)
    out = []
    for r in recs:
        p = plans.get(r["base"])
        if p is None:
            continue
        p = _gguf_plan(p, r)
        lvls = [lv for lv in QUANT_LEVELS + ("BF16", "F16") if lv in r["files"]]
        try:
            pred = BD.gguf_sizes(p, lvls)
        except Exception as e:
            out.append({"repo": r["repo"], "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        for lv in lvls:
            a = r["files"][lv]["bytes"]
            out.append({"repo": r["repo"], "base": r["base"], "model_type": r.get("model_type"), "level": lv,
                        "actual": a, "pred": pred[lv], "err": pred[lv] / a - 1})
    return out


def summarize(rows):
    ok = [x for x in rows if "err" in x]
    s = {"all_quantised": stats([x["err"] for x in ok if x["level"] in QUANT_LEVELS]),
         "by_level": {lv: stats([x["err"] for x in ok if x["level"] == lv]) for lv in QUANT_LEVELS + ("BF16", "F16")},
         "by_type": {}}
    for mt in sorted({x["model_type"] for x in ok}):
        s["by_type"][mt] = stats([x["err"] for x in ok if x["model_type"] == mt and x["level"] in QUANT_LEVELS])
    s["errors"] = [x for x in rows if "error" in x]
    s["worst"] = sorted((x for x in ok if x["level"] in QUANT_LEVELS), key=lambda x: -abs(x["err"]))[:25]
    return s


# ------------------------------------------------------------------------------------------------ safetensors quants

ST_HEADER_PER_TENSOR = 100          # bytes of safetensors JSON header per tensor entry (name, dtype, shape, offsets)
# the HF module a breakdown role stands for (matched against quantisation ignore lists and config-group targets)
ROLE_MODULE = {"attn.q": "self_attn.q_proj", "attn.k": "self_attn.k_proj", "attn.v": "self_attn.v_proj",
               "attn.o": "self_attn.o_proj", "attn.gate": "self_attn.g_proj", "attn.q_a": "self_attn.q_a_proj",
               "attn.q_b": "self_attn.q_b_proj", "attn.kv_a": "self_attn.kv_a_proj_with_mqa",
               "attn.kv_b": "self_attn.kv_b_proj", "attn.extra": "self_attn.indexer.wk",
               "attn.block": "self_attn.wq_a", "mlp.gate": "mlp.gate_proj", "mlp.up": "mlp.up_proj",
               "mlp.down": "mlp.down_proj", "mlp.linear": "mlp.linear", "moe.routed.gate": "mlp.experts.0.gate_proj",
               "moe.routed.up": "mlp.experts.0.up_proj", "moe.routed.down": "mlp.experts.0.down_proj",
               "moe.shared.gate": "mlp.shared_experts.gate_proj", "moe.shared.up": "mlp.shared_experts.up_proj",
               "moe.shared.down": "mlp.shared_experts.down_proj", "moe.shared_gate": "mlp.shared_expert_gate",
               "moe.latent": "mlp.fc1_latent_proj", "moe.router": "mlp.gate", "rec.linear": "mamba.in_proj",
               "lm_head": "lm_head", "embedding": "model.embed_tokens", "embedding.other": "model.embed_positions",
               "other.linear": "model.extra_proj"}


def _module(role, layer=5):
    m = ROLE_MODULE.get(role, role)
    return m if role in ("lm_head", "embedding", "embedding.other", "other.linear") else f"model.layers.{layer}.{m}"


def _match(pattern, name):
    import re
    if pattern.startswith("re:"):
        return re.search(pattern[3:], name) is not None
    return name == pattern or name.endswith("." + pattern) or pattern in name.split(".")


def _ign(qc):
    ig = []
    for k in ("ignore", "modules_to_not_convert", "llm_int8_skip_modules"):
        v = qc.get(k)
        if isinstance(v, list):
            ig += [str(x) for x in v]
    return ig


def _cdiv(a, b):
    return -(-a // b)


def _ct_group(qc, role):
    """The compressed-tensors config group (format, weights args) a role's module falls in, or None (not quantised):
    a group whose targets match the module name (a regex 're:...', or 'Linear' for any linear)."""
    name = _module(role)
    hit = None
    for g in (qc.get("config_groups") or {}).values():
        for tg in g.get("targets") or []:
            if tg == "Linear" and hit is None:
                hit = g
            elif tg != "Linear" and _match(tg, name):
                return g.get("format") or qc.get("format"), g.get("weights") or {}
    return (hit.get("format") or qc.get("format"), hit.get("weights") or {}) if hit else None


def quantised(role, kind, method, qc, n_in):
    """Is a tensor of this role stored quantised: the library's defaults, the config's ignore / skip lists (by the
    role's module name), the config-group targets (compressed-tensors)."""
    if kind not in ("linear", "lump", "embed"):
        return False
    name = _module(role)
    if any(_match(x, name) for x in _ign(qc)):
        return False
    if role.startswith("other"):
        return False
    if method == "mlx":
        return (n_in or 0) % int(qc.get("group_size") or 64) == 0 and role != "embedding.other"
    if kind == "embed" or role == "lm_head":
        return False
    if role in ("moe.router", "attn.router"):
        return method == "bnb"                               # bnb converts every nn.Linear it is not told to skip
    if method in ("gptq", "awq") and role.startswith("rec."):
        return False                                         # AutoGPTQ / AutoAWQ quantise the attention and MLP lists
    if method == "compressed-tensors":
        return _ct_group(qc, role) is not None
    return True


def quant_tensor_bytes(t, method, qc, scale_bytes=2):
    """(bytes, safetensors entries) of one quantised tensor (all its experts)."""
    e, X = t["elems"], t["experts"]
    n_in = t["in"] or 1
    n_out = e // (n_in * X) if n_in and X else 0
    bits = int(qc.get("bits") or 4)
    g = qc.get("group_size")
    g = n_in if g in (None, -1) else int(g)
    if method == "gptq":
        b = e * bits // 8 + X * _cdiv(n_in, g) * n_out * bits // 8 + X * _cdiv(n_in, g) * n_out * 2 + X * 4 * n_in
        return b, 4 * X
    if method == "awq":
        return e // 2 + X * _cdiv(n_in, g) * n_out // 2 + X * _cdiv(n_in, g) * n_out * 2, 3 * X
    if method == "fp8":                                      # quant_method fp8: weight_block_size scales
        bs = qc.get("weight_block_size") or [128, 128]
        return e + X * scale_bytes * _cdiv(n_out, bs[0]) * _cdiv(n_in, bs[1]), 2 * X
    if method == "compressed-tensors":
        fmt, w = _ct_group(qc, t["role"])
        nb = int(w.get("num_bits") or 8)
        strat = w.get("strategy")
        gs = int(w.get("group_size") or n_in)
        sb = 1 if fmt in ("nvfp4-pack-quantized", "mxfp4-pack-quantized") else scale_bytes
        if strat == "channel":
            sc = X * n_out * sb
        elif strat in ("group", "tensor_group"):
            sc = X * n_out * _cdiv(n_in, gs) * sb
        elif strat == "block":
            bs = w.get("block_structure") or [128, 128]
            sc = X * scale_bytes * _cdiv(n_out, bs[0]) * _cdiv(n_in, bs[1])
        else:
            sc = X * 4
        extra = 0
        if fmt == "nvfp4-pack-quantized":
            extra = X * 8                                    # weight_global_scale, input_global_scale (f32 [1])
        elif fmt == "pack-quantized":
            extra = X * 8                                    # weight_shape (int32 [2])
            if not w.get("symmetric", True):
                extra += X * n_out * _cdiv(n_in, gs) * nb // 8
        elif fmt in ("float-quantized", "naive-quantized", "int-quantized"):
            ia = (next(iter((qc.get("config_groups") or {}).values()), {}) or {}).get("input_activations") or {}
            if ia and not ia.get("dynamic", True):
                extra = X * 4                                # a static input_scale
        return e * nb // 8 + sc + extra, 3 * X
    if method == "bnb":
        dq = bool(qc.get("bnb_4bit_use_double_quant"))
        nblk = _cdiv(e // X, 64)
        per = (nblk + 4 * _cdiv(nblk, 256) + 4 * 256 + 4 * 16 + 200) if dq else (4 * nblk + 4 * 16 + 200)
        return _cdiv(e, 2) + X * per, (6 if dq else 4) * X
    if method == "mlx":
        gm = int(qc.get("group_size") or 64)
        return e * bits // 8 + 2 * 2 * e // gm, 3 * X
    raise KeyError(method)


def _method(rec):
    qc = rec.get("quantization_config") or {}
    return "compressed-tensors" if qc.get("quant_method") == "compressed-tensors" else rec["method"]


def stored_layout(p, rec, raw=None):
    """(stored head, note): does the checkpoint store a tied head as its own tensor, read off the Hub's safetensors
    metadata (element counts per dtype): MLX reports logical counts, so its total says; the other formats keep the
    head unquantised, so their BF16 / F16 count says (the tied table alone, or twice). note: why a record is left
    out of the statistics (a checkpoint with vision / audio towers beside the LM)."""
    from graph import arch_breakdown as BD
    from graph.arch_metrics import has_towers
    if raw is not None and has_towers(raw):
        return False, "the checkpoint holds towers beside the LM"
    if not p["tied"]:
        return False, None
    sp = (rec.get("safetensors_params") or {})
    if _method(rec) == "mlx":
        n, tot = sp.get("total") or 0, BD.params_breakdown(p)["total"]
        return bool(n and abs(n - tot - p["lm_head_W"]) < abs(n - tot)), None
    dt = sp.get("parameters") or {}
    unq = dt.get("BF16", 0) + dt.get("F16", 0)
    emb = p["embedding"]
    return bool(unq and abs(unq - emb - p["lm_head_W"]) < abs(unq - emb)), None


def safetensors_bytes(p, rec, stored_head=False):
    """Predicted total bytes of a quantised safetensors checkpoint of plan p (record of quant_sizes.json)."""
    from graph import arch_breakdown as BD
    qc = rec.get("quantization_config") or {}
    method = _method(rec)
    dt = (rec.get("safetensors_params") or {}).get("parameters") or {}
    base_b = 2                                               # BF16 / F16 for everything not quantised
    scale_b = 4 if dt.get("F32", 0) > 1000 else 2
    tensors = BD.all_tensors(p)
    if stored_head and p["tied"]:
        emb = next(t for t, _ in tensors if t["role"] == "embedding")
        tensors.append((dict(emb, role="lm_head", kind="linear"), 1))
    total = entries = 0
    for t, c in tensors:
        if quantised(t["role"], t["kind"], method, qc, t["in"]):
            b, n = quant_tensor_bytes(t, method, qc, scale_b)
        else:
            b, n = base_b * t["elems"], t["experts"]
        total += c * b
        entries += c * n
    return total + ST_HEADER_PER_TENSOR * entries + 8 * max(1, rec.get("n_files") or 1)


def check_safetensors(data=None):
    from graph.model_graph import iter_json_configs
    data = data or json.load(open(SIZES))
    recs = data["safetensors"]
    plans = _plans(recs)
    configs = dict(iter_json_configs(os.path.join(HERE, "hub_10k.json")))
    out = []
    for r in recs:
        p = plans.get(r["base"])
        if p is None:
            continue
        head, note = stored_layout(p, r, configs.get(r["base"]))
        try:
            pred = safetensors_bytes(p, r, stored_head=head)
        except Exception as e:
            out.append({"repo": r["repo"], "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        qc = r.get("quantization_config") or {}
        fmt = None
        if qc.get("quant_method") == "compressed-tensors":
            fmts = {g.get("format") or qc.get("format") for g in (qc.get("config_groups") or {}).values()}
            fmt = "/".join(sorted(f for f in fmts if f)) or qc.get("format")
        out.append({"repo": r["repo"], "base": r["base"], "method": r["method"], "format": fmt,
                    "bits": r.get("bits"), "actual": r["bytes"], "pred": pred, "err": pred / r["bytes"] - 1,
                    "stored_head": head, "excluded": note})
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    cal, _ = calibrate()
    print("calibration", cal)
    rows = check_gguf()
    s = summarize(rows)
    print(json.dumps({k: s[k] for k in ("all_quantised", "by_level")}, indent=1))
    for x in s["worst"][:15]:
        print(f"{x['err']:+.3%} {x['level']:7} {x['repo']}")
    st = check_safetensors()
    ok = [x for x in st if "err" in x and not x.get("excluded")]
    ss = {"all": stats([x["err"] for x in ok]), "by_method": {},
          "excluded": [{k: x[k] for k in ("repo", "excluded", "err")} for x in st if x.get("excluded")]}
    for k in sorted({(x["method"], x.get("format") or "") for x in ok}):
        ss["by_method"]["/".join(filter(None, k))] = stats([x["err"] for x in ok
                                                           if (x["method"], x.get("format") or "") == k])
    ss["worst"] = sorted(ok, key=lambda x: -abs(x["err"]))[:25]
    ss["errors"] = [x for x in st if "error" in x]
    print(json.dumps({"all": ss["all"], **ss["by_method"]}, indent=0), len(ss["excluded"]), "excluded")
    for x in ss["worst"][:15]:
        print(f"{x['err']:+.3%} {x['method']:18} {x['repo']}")
    with open(OUT, "w") as f:
        json.dump({"gguf": {"calibration": cal, "summary": s, "rows": rows},
                   "safetensors": {"summary": ss, "rows": st}}, f, indent=0)
