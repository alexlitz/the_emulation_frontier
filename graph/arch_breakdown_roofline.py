"""Roofline UPPER BOUNDS against published decode benchmarks (graph/data/decode_benchmarks.json, with citations):
every published number must fall below the bound computed from the vendor's peak bandwidth and FLOPs
(graph/data/hardware.json) and the model's bytes and FLOPs per step (graph/arch_breakdown.py). The ratio
real / bound is how much of the roofline the framework reaches.

  batch-1 decode (llama.cpp tg128)   tokens/s <= 1 / max(bytes per step / bandwidth, FLOPs per step / peak),
                                    at the mean position of the generated tokens, the GGUF level's real bytes
  batched (TensorRT-LLM, B given)   tokens/s <= B / max(...) with the weights read once per step
  max load (B not given)            tokens/s <= 1 / max(KV bytes per sequence / bandwidth, FLOPs per token / peak),
                                    the B -> infinity limit (the weights amortised away)

The output tokens/s of an end-to-end benchmark also pays for the prefill, so the bound (decode only) is loose there.

Run:  python -m graph.arch_breakdown_roofline      # writes graph/data/roofline_check.json
"""
from __future__ import annotations

import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "data", "decode_benchmarks.json")
CONFIGS = os.path.join(HERE, "data", "benchmark_configs.json")
OUT = os.path.join(HERE, "data", "roofline_check.json")
# the KV-cache format of a benchmark: llama.cpp's default is f16; TensorRT-LLM's FP8 / FP4 runs cache in FP8 (the
# v0.7.1 performance.md engine builds pass --enable_fp8 --fp8_kv_cache; the ModelOpt checkpoints of the
# perf-overview page, e.g. nvidia/Llama-3.1-8B-Instruct-FP8 and nvidia/Llama-3.3-70B-Instruct-FP4, declare
# kv_cache_quant_algo FP8 in hf_quant_config.json)
def kv_format(b):
    if b["framework"] == "TensorRT-LLM" and (b["weights"] == "fp8" or str(b["weights"]).startswith("fp4")):
        return "fp8"
    return "bf16"


# a gated repo's architecture, read from a snapshot repo of the same architecture (identical config shapes)
SAME_ARCH = {"meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct"}


def benchmark_configs(names, refresh=False):
    """config.json of each benchmark model: the zoo snapshot, else fetched once from the Hub (metadata only) and
    kept in graph/data/benchmark_configs.json."""
    from graph.model_graph import iter_json_configs
    snap = dict(iter_json_configs(os.path.join(HERE, "hub_10k.json")))
    try:
        with open(CONFIGS) as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}
    out = {}
    for n in names:
        src = SAME_ARCH.get(n, n)
        if src in snap:
            out[n] = snap[src]
            continue
        if src not in cache or refresh:
            with urllib.request.urlopen(f"https://huggingface.co/{src}/resolve/main/config.json") as r:
                cache[src] = json.loads(r.read().decode())
        out[n] = cache[src]
    with open(CONFIGS, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    return out


def weight_read(p, fmt, B=1):
    """(bytes the weights cost per decode step, the bytes of the whole model) in a weight format: a GGUF level
    (the converter's tensors through llama-quantize), else a safetensors-style format (arch_breakdown)."""
    from graph import arch_breakdown as BD
    from graph import gguf_quant as GQ
    if fmt in ("F16", "BF16", "Q8_0", "Q4_0") + BD.GGUF_LEVELS:
        ts = BD.gguf_tensors(p)
        mt = str(p["mt"]).lower()
        arch = BD.LLAMA_ARCH.get(mt, mt)
        src = GQ.source_types(ts, arch)
        if fmt == "F16":
            src = ["F16" if s == "BF16" else s for s in src]
        ts = [dict(t, src=s) for t, s in zip(ts, src)]
        a0 = next((g["attn"] for g in p["groups"] if g["attn"] is not None), None)
        ctx = {"n_layer": p["layers"], "n_head": a0["nh"], "n_head_kv": a0["nkv"]} if a0 else {}
        b = GQ.breakdown(ts, None if fmt in ("F16", "BF16") else fmt, arch, 0, **ctx)
        total = b["data"]
        emb = next((r["bytes"] for r in b["tensors"] if r["name"] == "token_embd.weight"), 0)
        has_out = any(r["name"] == "output.weight" for r in b["tensors"])
        row = emb / p["vocab"] if p["vocab"] else 0
        read = total - (emb if has_out else 0) + B * row
        return read, total
    fm = {"bf16": "bf16", "fp8": "fp8", "fp4 (NVFP4)": "nvfp4"}.get(fmt, fmt)
    wb = BD.weight_bytes(p, fm)
    emb_row = (wb["embedding"] / p["embedding"]) * (p["lm_head_W"] // p["vocab"]) if p["embedding"] else 0
    r = BD.decode_read(p, 1, B, wb, emb_row)
    return r["weights"], wb["total"]


def bound(b, p, hw):
    """The roofline bound of one benchmark record: (tokens/s bound, regime, the inputs)."""
    from graph import arch_breakdown as BD
    B = b.get("batch")
    L = int((b.get("prompt_tokens") or 0) + max(1, (b.get("gen_tokens") or 1)) / 2)   # mean decode position
    fmt = b["weights"]
    bw = hw["mem_bw_gbps"] * 1e9
    fl = hw.get("flops") or {}
    key = "fp8_dense" if fmt == "fp8" else "fp4_dense" if fmt.startswith("fp4") else None
    peak = (fl.get(key) if key else None) or fl.get("bf16_dense") or fl.get("fp16_dense")
    peak *= 1e12
    f_tok = BD.flops_at(p, L)["decode"]["total"]
    kv_seq = BD.kv_bytes(BD.kv_elems(p, max(L - 1, 0)), kv_format(b))
    if B is None:
        t = max(kv_seq / bw, f_tok / peak)
        return 1 / t, ("compute" if f_tok / peak >= kv_seq / bw else "kv-bandwidth"), {
            "L": L, "flops_per_token": f_tok, "kv_bytes_per_seq": kv_seq, "peak_flops": peak, "bandwidth": bw}
    w, total = weight_read(p, fmt, B)
    byt = w + B * kv_seq
    fl_step = B * f_tok
    t = max(byt / bw, fl_step / peak)
    return B / t, ("bandwidth" if byt / bw >= fl_step / peak else "compute"), {
        "L": L, "weight_bytes_read": w, "weights_total": total, "kv_bytes_per_seq": kv_seq, "flops_per_step": fl_step,
        "bytes_per_step": byt, "peak_flops": peak, "bandwidth": bw}


def check(write=True):
    from graph import arch_breakdown as BD
    bench = json.load(open(BENCH))["benchmarks"]
    hws = {h["id"]: h for h in BD.hardware()["hardware"]}
    cfg = benchmark_configs(sorted({b["config_repo"] for b in bench}))
    rows = []
    for b in bench:
        p = BD.plan_ordered(cfg[b["config_repo"]])
        bd, regime, inp = bound(b, p, hws[b["hardware"]])
        rows.append({**{k: b.get(k) for k in ("hardware", "framework", "model", "config_repo", "weights", "batch",
                                              "prompt_tokens", "gen_tokens", "tokens_per_s", "metric", "url")},
                     "bound_tokens_per_s": bd, "ratio": b["tokens_per_s"] / bd, "regime": regime, "inputs": inp,
                     "below_bound": b["tokens_per_s"] < bd, "kv_cache": kv_format(b),
                     "same_arch_config": SAME_ARCH.get(b["config_repo"])})
    out = {"method": __doc__.split("\n\n")[1].strip(), "rows": rows,
           "all_below": all(r["below_bound"] for r in rows),
           "ratio_range": [min(r["ratio"] for r in rows), max(r["ratio"] for r in rows)]}
    if write:
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    o = check()
    for r in o["rows"]:
        print(f"{r['hardware']:14} {r['framework']:12} {r['weights']:11} B={str(r['batch']):5} real {r['tokens_per_s']:9.1f} "
              f"bound {r['bound_tokens_per_s']:10.1f}  ratio {r['ratio']:.3f}  {r['regime']}")
    print("all below:", o["all_below"], "ratio range", o["ratio_range"])
