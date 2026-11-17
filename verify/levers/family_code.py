"""Which shape-preserving config switches each model family's CODE reads (host-side capabilities).

For every model_type in the snapshot, the family's modeling file in transformers main (fetched once into
$TEF_SCRATCH/tfmain; version recorded) is grepped for the config reads that change behaviour without changing a
parameter tensor's shape. A family whose code is not in transformers (remote code) gets no capability:
nothing is claimed for it. Each capability records the matching source line as evidence.

Run:  python -m verify.levers.family_code     # writes verify/levers/out/family_code.json
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from verify.levers.rel import lm, points

CACHE = os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "tfmain")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
RAW = "https://raw." + "github" + "usercontent.com/huggingface/transformers/main/src/transformers/"
API = "https://api." + "github" + ".com/repos/huggingface/transformers/contents/src/transformers/models?ref=main"
ALIAS = {"mimo_v2": "mimo_v2_flash", "gemma4_unified_text": "gemma4_unified"}

# capability -> regex over the modeling file (a config READ that the forward pass uses)
CAPS = {
    # MLP activation looked up from the config string (any parameter-free ACT2FN entry can be set)
    "act_config": r"ACT2FN\[\s*config\.(hidden_act|hidden_activation|mlp_hidden_act|activation_function)",
    # q / k norm built only when a config flag says so (switch OFF without touching other tensors' shapes)
    "qknorm_switch": r"config\.(use_qk_norm|qk_layernorm|use_qk_layernorm|qk_norm\b|use_head_qk_norm|use_qknorm)",
    # rotary applied to a configurable slice of the head (rot / pass split)
    "partial_rope": r"\b(q_pass|query_pass|k_pass|key_pass|rotary_ndims|q_rot|query_rot)\b",
    # parallel vs sequential residual chosen by config
    "parallel_switch": r"config\.(use_parallel_residual|parallel_attn)\b",
    # MoE top-k read from config
    "moe_topk": r"config\.(num_experts_per_tok|moe_topk|moe_top_k|num_experts_per_token|moe_k)\b",
    # router score function chosen by config
    "scoring_switch": r"config\.(scoring_func|score_func|router_score_func|moe_router_activation_func)\b",
    # group-limited routing by config
    "router_group_cfg": r"config\.(n_group|topk_group)\b",
    # DSA indexer budget by config
    "index_topk": r"config\.(index_topk|index_top_k)\b",
    # RoPE / NoPE per layer by config
    "nope_cfg": r"config\.(no_rope_layers|no_rope_layer_interval|no_rope_layer_indices|use_rope)\b",
    # sliding window by config
    "window_cfg": r"config\.(sliding_window|attention_chunk_size|attention_window_size)\b",
    # attention softcap by config
    "softcap_cfg": r"config\.(attn_logit_softcapping|final_logit_softcapping)\b",
    # qk-norm / post-norm epsilon from config (the eps lever needs the eps read from config)
    "norm_eps_cfg": r"eps\s*=\s*config\.(rms_norm_eps|layer_norm_eps|norm_eps|layer_norm_epsilon)",
}


def _get(url):
    return urllib.request.urlopen(url, timeout=30).read().decode()


def modeling_source(mt, main_dirs):
    d = ALIAS.get(mt, mt)
    if d not in main_dirs and d.endswith("_text"):
        d = d[: -len("_text")]
    if d not in main_dirs:
        return None, None
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"modeling_{d}.py")
    if not os.path.exists(path):
        try:
            src = _get(RAW + f"models/{d}/modeling_{d}.py")
        except Exception:
            return d, None
        with open(path, "w") as f:
            f.write(src)
    with open(path) as f:
        return d, f.read()


def capabilities():
    P, cfg = points()
    mts = sorted({str(lm(cfg[n]).get("model_type")) for p in P for n in p["members"]})
    main_dirs = {x["name"] for x in json.loads(_get(API))}
    version = re.search(r'__version__ = "([^"]+)"', _get(RAW + "__init__.py")).group(1)
    out = {}
    for mt in mts:
        d, src = modeling_source(mt, main_dirs)
        if src is None:
            out[mt] = dict(code=None, caps={})
            continue
        caps = {}
        for cap, rx in CAPS.items():
            m = re.search(r"config\.sliding_window\b", src) if cap == "window_cfg" else None   # prefer the window itself
            m = m or re.search(rx, src)
            if m:
                line = src[: m.start()].count("\n") + 1
                caps[cap] = f"modeling_{d}.py:{line}: " + src.splitlines()[line - 1].strip()[:120]
        out[mt] = dict(code=f"models/{d}/modeling_{d}.py", caps=caps)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "family_code.json"), "w") as f:
        json.dump(dict(transformers=version, families=out), f, indent=1)
    return version, out


def load_caps():
    path = os.path.join(OUT, "family_code.json")
    if not os.path.exists(path):
        capabilities()
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    v, out = capabilities()
    have = [m for m, x in out.items() if x["code"]]
    print(f"transformers main {v}: {len(have)} of {len(out)} model types have in-library code")
    from collections import Counter
    c = Counter(cap for x in out.values() for cap in x["caps"])
    for cap, k in c.most_common():
        print(f"  {cap:18s} {k}")
