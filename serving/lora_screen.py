"""Is a derived checkpoint exactly a LoRA of its base? Screened by range-reading tensors from both.

A LoRA serves derived model D on base B exactly (to bf16 precision) iff every adaptable linear's
delta D - B has rank <= the provider cap and every non-adaptable tensor (embeddings, norms, biases,
routers, conv / decay parameters of linear-attention layers) is unchanged.

  quick: one mid-layer q/o/down, a norm and 4,096 embedding rows per pair (cheap triage).
  full:  every language-model tensor. Large matrices are read as their first --rows rows: a
         row slice of a rank-r delta has rank <= r, so the slice bounds the rank from below and
         settles any cap below --rows. Routed experts (fused or per-expert tensors) are
         sampled at the first, middle and last expert.

  python -m serving.lora_screen quick serving/data/pairs.json > serving/data/screen.jsonl
  python -m serving.lora_screen full  serving/data/full_pairs.json > serving/data/full.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from serving.st_reader import Repo, delta_rank

SKIP = re.compile(r"(^|\.)(visual|vision_tower|vision_model|audio_tower|mtp|multi_modal_projector|embed_vision)\.")
LINEAR = re.compile(r"(^|\.)(q|k|v|o|gate|up|down|gate_up|qkv|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z|"
                    r"in_proj_b|in_proj_a|out|dense|fc1|fc2|c_attn|c_proj|w1|w2|w3)(_proj)?\.weight$")
ROUTER = re.compile(r"(mlp\.gate\.weight|router\.|\.gate\.weight$|e_score_correction)")
EMBED = re.compile(r"(embed_tokens|wte)\.weight$")
HEAD = re.compile(r"lm_head\.weight$")
CACHE = os.path.expanduser("~/.cache/lora_screen")


def kind(name, shape):
    if EMBED.search(name):
        return "embed"
    if HEAD.search(name):
        return "head"                                   # adaptable only where the engine allows lm_head LoRA
    if ROUTER.search(name) and "experts" not in name:
        return "router"
    if len(shape) >= 2 and LINEAR.search(name.removesuffix(".weight") + ".weight") and "conv" not in name:
        return "linear"
    return "fixed"


def cached_read(repo, name, rows=None, lead=None):
    """Base-side reads are reused across pairs (many derived models share one base)."""
    path = os.path.join(CACHE, repo.repo.replace("/", "__"), f"{name}.{rows}.{lead}.pt")
    if os.path.exists(path):
        return torch.load(path)
    t = repo.read(name, rows, lead)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(t.clone(), path)
    return t


def compare(B, D, name, rows, experts, cache_base, bname=None):
    """Verdict for one tensor: {'kind', 'rank', 'rel', 'subulp' | 'rows_big', ...}. `bname` reads a
    differently named base tensor (an untied lm_head against the base's tied embeddings)."""
    bname = bname or name
    shape = B.shape(bname)
    if shape != D.shape(name):
        return {"kind": kind(name, shape), "shape": [shape, D.shape(name)]}
    k = kind(name, shape)
    rb = (lambda *a: cached_read(B, bname, *a)) if cache_base else (lambda *a: B.read(bname, *a))
    if k == "embed":
        b, d = rb(4096).float(), D.read(name, 4096).float()
        row = (d - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
        big = float((row > 0.01).float().mean())
        return {"kind": k, "rank": 0 if big == 0 else None, "rows_big": big, "rel": float((d - b).norm() / b.norm())}
    if k == "head":                                     # 4,096 vocab rows: edits often sit on a few tokens
        r, rel, sub = delta_rank(rb(4096), D.read(name, 4096))
        return {"kind": k, "rank": r, "rel": rel, "subulp": sub}
    if len(shape) == 3 and k == "linear":               # fused experts [E, out, in]
        leads = sorted({0, shape[0] // 2, shape[0] - 1})[:experts]
        rs = [delta_rank(rb(rows, e), D.read(name, rows, e)) for e in leads]
        r = None if any(x[0] is None for x in rs) else max(x[0] for x in rs)
        return {"kind": k, "rank": r, "rel": max(x[1] for x in rs), "subulp": all(x[2] for x in rs)}
    if len(shape) == 2 and k in ("linear", "router"):
        r, rel, sub = delta_rank(rb(rows), D.read(name, rows))
        return {"kind": k, "rank": r, "rel": rel, "subulp": sub}
    big = len(shape) == 2 and shape[0] * shape[1] > 50_000_000
    b, d = (rb(rows) if big else rb()).float(), (D.read(name, rows) if big else D.read(name)).float()
    # ^ norms, biases, conv, A_log, dt_bias, ...; huge non-linear tables are compared on a row slice
    if torch.equal(b, d):
        return {"kind": k, "rank": 0, "rel": 0.0}
    rel = float((d - b).norm() / b.norm().clamp_min(1e-30))
    return {"kind": k, "rank": 0 if rel < 1e-3 else None, "rel": rel}


def screen_full(base, der, rows=96, experts=3, cap=(16, 64), workers=8):
    B, D = Repo(base), Repo(der)
    names_b = {n for n in B.idx if not SKIP.search(n)}
    names_d = {n for n in D.idx if not SKIP.search(n)}
    common = sorted(names_b & names_d)
    n_exp = max((int(m[1]) + 1 for n in common for m in [re.search(r"experts\.(\d+)\.", n)] if m), default=0)
    if n_exp:                                           # per-expert tensors: sample experts like the fused case
        keep = {str(e) for e in sorted({0, n_exp // 2, n_exp - 1})[:experts]}
        common = [n for n in common if (m := re.search(r"experts\.(\d+)\.", n)) is None or m[1] in keep]
    extra = names_d - names_b
    alias = {}
    if "lm_head.weight" in extra and "embed_tokens.weight" in names_b:   # base ties the head to the embeddings
        extra.discard("lm_head.weight")
        alias["lm_head.weight"] = "embed_tokens.weight"
        common.append("lm_head.weight")
    out = {"derived": der, "base": base, "n_common": len(common),
           "missing": len(names_b - names_d), "extra": len(extra)}
    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip(common, ex.map(lambda n: _safe(compare, B, D, n, rows, experts, True, alias.get(n)), common)))
    worst = {}                                          # per module type (name with layer index dropped)
    for n, v in res.items():
        key = re.sub(r"\.\d+\.", ".N.", n)
        r = v.get("rank", "x") if "error" not in v and "shape" not in v else "x"
        cur = worst.get(key)
        worst[key] = r if cur is None else _max_rank(cur, r)
    changed_fixed = sorted(n for n, v in res.items()
                           if v.get("kind") in ("fixed", "router") and v.get("rank") != 0)
    emb = [v for v in res.values() if v.get("kind") == "embed"]
    head = [v for v in res.values() if v.get("kind") == "head"]
    max_lin = 0
    for v in res.values():
        if v.get("kind") in ("linear", "head"):
            max_lin = _max_rank(max_lin, v.get("rank", "x"))
    head_rank = max((v.get("rank") or 0 for v in head), default=0) if all(v.get("rank") is not None for v in head) else None
    exact_ok = (not changed_fixed and all(e.get("rows_big", 1) == 0 for e in emb)
                and not any(isinstance(v, dict) and ("shape" in v or "error" in v) for v in res.values())
                and out["extra"] == 0)
    fits = [c for c in cap if isinstance(max_lin, int) and max_lin <= c]
    n_sub = sum(1 for v in res.values() if v.get("subulp"))
    if not exact_ok or not fits:
        verdict = "not-lora"
    elif all(v.get("rel", 0) == 0 for v in res.values() if "rel" in v):
        verdict = "identical"
    elif max_lin == 0:
        verdict = "sub-ulp"     # every change is below one bf16 step per weight: a real but tiny update
                                # whose structure a row slice can't resolve (needs a behavioural check)
    else:
        verdict = f"lora<={fits[0]}" + ("+lm_head" if head_rank else "")
        if n_sub:   # a sub-ulp tensor's rank is only a lower bound: Selene-1-Mini reads as rank 2 in its
            verdict += "?sub-ulp"   # weights but needs rank ~64+ to reproduce behaviourally (lora_behaviour)
    out.update(verdict=verdict, max_linear_rank=max_lin, head_rank=head_rank, n_subulp=n_sub,
               n_changed_fixed=len(changed_fixed),
               changed_fixed=changed_fixed[:20], embed=emb, per_module=worst,
               errors=sorted(n for n, v in res.items() if "error" in v or "shape" in v)[:10])
    return out


def _max_rank(a, b):
    if "x" in (a, b):
        return "x"
    if a is None or b is None:
        return None
    return max(a, b)


def _safe(f, *a):
    try:
        return f(*a)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:160]}


def screen_quick(base, der):
    """One mid-layer q/o/down, the first input_layernorm at that layer, 4,096 embedding rows."""
    B, D = Repo(base), Repo(der)
    common = sorted((set(B.idx) & set(D.idx)) - {n for n in B.idx if SKIP.search(n)})
    out = {"derived": der, "base": base, "n_common": len(common), "n_base": len(B.idx), "n_der": len(D.idx)}
    layers = sorted({int(m.group(1)) for n in common for m in [re.search(r"layers\.(\d+)\.", n)] if m})
    mid = layers[len(layers) // 2] if layers else 0
    for tag in ("o_proj.weight", "q_proj.weight", "down_proj.weight", "input_layernorm.weight", "embed_tokens.weight"):
        cands = [n for n in common if n.endswith(tag) and "expert" not in n]
        if cands:
            n = min(cands, key=lambda n: abs(int((re.search(r"layers\.(\d+)\.", n) or [0, 0])[1]) - mid))
            out[tag] = _safe(compare, B, D, n, None, 1, False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("quick", "full"))
    ap.add_argument("pairs", help="JSON list of [base, derived]")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    pairs = json.load(open(a.pairs))
    if a.mode == "quick":
        with ThreadPoolExecutor(a.workers) as ex:
            for res in ex.map(lambda p: _safe(screen_quick, *p) | {"derived": p[1]}, pairs):
                print(json.dumps(res), flush=True)
    else:
        for base, der in pairs:                         # pairs in sequence, tensors in parallel
            print(json.dumps(_safe(screen_full, base, der, a.rows) | {"derived": der}), flush=True)
            print(f"done {der}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
