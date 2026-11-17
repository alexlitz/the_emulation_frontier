"""Random end-to-end spot checks of the emulation relation.

Each claimed edge "B emulates A" of the headline relation (graph.model_graph.emul_matrix(mode="payable") over
the zoo's shapes, graph.currency_curve.points) should be buildable as real weights that reproduce A. This draws
edges at random, stratified so that every kind of rule the relation uses is exercised, and for each one:

  real     both endpoints have a small member repo that elephantine/emulate.py builds: the host checkpoint is
           written from the target's real weights (the host's own stock class and config, only its config.json
           read) and checked against the target -- fp32 and fp64 (the models' fp32 casts lifted) max |d logits|,
           top-1 agreement over the prompt positions, and greedy 32-token continuations on 8 prompts;
  scaled   emulate.py builds the pair but a checkpoint is too big for this machine: a scaled-down pair -- the
           two real configs with every width (hidden, head_dim, FFN / expert widths) divided by one power of
           two f, windows shortened so the 8 + 32-token prompts cross them, vocab 49152 -- is checked to still be
           an edge with the same slot signature, a target with random weights is written, and the same
           verification runs. That tests the construction for the pair's exact rule combination;
  unsupported  emulate.py refuses the pair: the refusal (emulate.support / gate), the plan (graph.edge_plan) and
           the construction pieces that are missing, as categories, are recorded.

Pass: an exact edge must reproduce the target in fp64 (rel max |d logits| <= 1e-9, or 1e-6 where the RoPE
theta had to be rescaled for a non-integer head_dim ratio and inv_freq is fp32 at init) with every greedy
continuation identical; when fp64 does not fit, and for limit / regime edges, the high-agreement standard on
fp32: top-1 >= 99% and >= 90% of the continuations identical. Anything else is a FAILURE (a finding).

  python -m verify.spot.run_spot_checks --seed 0 --n 240               # sample, build, verify, write results
  python -m verify.spot.run_spot_checks --report                        # SPOT.md from results.jsonl
  python -m verify.spot.run_spot_checks --one TARGET HOST               # one pair (real repos or local dirs)

Scratch checkpoints go under $SPOT_DIR (default $TEF_SCRATCH/spot/) and are deleted after each check.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
SPOT_MD = HERE / "SPOT.md"
WORK = Path(os.environ.get("SPOT_DIR", os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "spot")))
TOK = "HuggingFaceTB/SmolLM2-135M"              # tokenizer of the scaled-down models (vocab 49152)
VOCAB = 49152
PROMPTS = ["The capital of France is", "def fibonacci(n):", "Once upon a time, there was a little",
           "The theory of general relativity states that", "import numpy as np\n", "Q: What is 17 times 3?\nA:",
           "In 1492, Columbus", "The best way to learn a new language is"]
NEW_TOKENS = 32
MAX_TARGET = 1.5e9                              # params: real targets this (shared) machine runs in fp64
MAX_HOST = 2.5e9
MAX_SCALED = 1.3e9                               # params of a scaled-down host
EXACT_TOL, THETA_TOL, FP32_TOL = 1e-9, 1e-6, 1e-4
SLOT_NAMES = ["norm", "pos", "kv", "ffn", "act", "block", "window", "qknorm", "softcap", "mixer", "sink",
              "crosslayer_kv", "mtp", "gen", "loop"]


# ================================================================== the relation and its edges

def load_relation():
    """(P, ok, exact, configs, meta): the shapes, the headline and exact matrices (ok[i, j]: host j emulates
    target i, diagonal off), every repo's raw config and the Hub metadata."""
    import numpy as np
    from graph.currency_curve import points
    from graph.model_graph import emul_matrix
    P = points()
    ok = emul_matrix(P, mode="payable")
    ex = emul_matrix(P, mode="exact")
    np.fill_diagonal(ok, False)
    np.fill_diagonal(ex, False)
    configs = {r["id"]: r["config"] for r in json.loads((ROOT / "graph/hub_10k.json").read_text())}
    meta = json.loads((ROOT / "graph/model_meta.json").read_text())
    return P, ok, ex & ok, configs, meta


def _types(p, configs):
    return {str(configs.get(m, {}).get("model_type")) for m in p["members"]}


def edge_tags(A, B, ta=None, tb=None, exact=True):
    """The kinds of rule the edge "B emulates A" spends (A, B: points). ta / tb: the shapes' model_types."""
    from graph.model_graph import FFN_RANK, MOE_SIG, ZERO_EDGES
    sa, sb, da, db = A["slots"], B["slots"], A["dims"], B["dims"]
    tags = []
    diffs = [s for s in SLOT_NAMES if sa.get(s) != sb.get(s)]
    fa, fb = sa.get("feats", ""), sb.get("feats", "")
    if not diffs and fa == fb:
        tags.append("capacity")
    plain = sa["mixer"] == "softmax" and sa["crosslayer_kv"] == "no"
    for s in diffs:
        tags.append(f"slot:{s}")
        tags.append(f"step:{s}:{sa[s]}->{sb[s]}")
        if (s, sa[s], sb[s]) in ZERO_EDGES and plain:
            tags.append("zero_edge")
    if fa != fb:
        tags.append("feats")
    # KV-head packing: the target's KV heads do not tile over the host's groups
    kvt, gt, kvh, gh = (int(da.get("kv_heads", 0)), int(da.get("q_per_kv", 0)), int(db.get("kv_heads", 0)),
                        int(db.get("q_per_kv", 0)))
    if kvt and gt and kvh and gh and kvt * -(-gt // gh) > kvh:
        tags.append("kv_pack")
    # MoE
    mt = dict(zip(MOE_SIG, da.get("moe_sig") or ())) if da.get("moe_sig") else {}
    mh = dict(zip(MOE_SIG, db.get("moe_sig") or ())) if db.get("moe_sig") else {}
    Et, Eh = int(mt.get("E", 0) or 0), int(mh.get("E", 0) or 0)
    if Et and Eh:
        tags.append("moe<-moe")
        kt, kh = int(mt.get("k", 0) or 0), int(mh.get("k", 0) or 0)
        if kt and kh and kh // max(kt, 1) > 1:
            tags.append("moe_dup")
    elif Et and not Eh:
        tags.append("dense_host_for_moe")
    elif Eh and not Et:
        tags.append("moe_host_for_dense")
    if FFN_RANK.get(sa["ffn"], 0) != FFN_RANK.get(sb["ffn"], 0) and not (Et or Eh):
        tags.append("ffn_kind")
    # recurrent capacity, masks, rotary tables
    if int(da.get("rc_rule", 0) or 0):
        tags.append("recurrent")
    if sa.get("gen") != sb.get("gen") or sa.get("gen") != "ar":
        tags.append("mask")
    if tuple(db.get("rope_cap", ()) or ()) != tuple(db.get("rope_sched", ()) or ()) or (
            sa["pos"] != sb["pos"] and "rope" in sb["pos"]):
        tags.append("rope_table")
    if "MLA" in sa["kv"]:
        tags.append("mla")
    tags += lever_tags(A, B)
    if ta is not None and tb is not None and not (ta & tb):
        tags.append("cross_family")
    if not exact:
        tags.append("approx")
    return tags


def lever_tags(A, B):
    """The round 2 pass 5 config levers and rescues the edge may spend (graph/model_graph.py P5_PORTS; dims_of's
    pass-5 dims): the eps lever, the window lever, the start-token rules, the activation width, looped hosts, the
    pre-norm compensators."""
    sa, sb, da, db = A["slots"], B["slots"], A["dims"], B["dims"]
    out = []
    fa = set(sa.get("feats", "").lstrip("*").split("+")) - {""}
    fb = set(sb.get("feats", "").lstrip("*").split("+")) - {""}
    if db.get("eps_lever") and ((sa["qknorm"] == "no" and sb["qknorm"] != "no") or ({"post_norm", "vnorm"} & (fb - fa))
                                or ("MLA" in sb["kv"] and "MLA" not in sa["kv"])):
        out.append("eps_lever")
    if db.get("win_free") and sb["window"] != "full" and sa["window"] != sb["window"]:
        out.append("window_lever")
    if (sa["pos"] == "learned_abs" and sb["pos"] != "learned_abs") or (sa["sink"] == "yes" and sb["sink"] == "no"):
        out.append("start_token")
    if sa["act"] != sb["act"]:
        try:
            from graph.model_graph import ACT_ORDER
            if not any(h == sb["act"] for h, _, _ in ACT_ORDER.get(sa["act"], [])):
                out.append("act_width")
        except ImportError:
            pass
    if sa["loop"] == "untied" and sb["loop"] == "looped":
        out.append("looped_host")
    if (sa["norm"] == "LayerNorm" and sb["norm"] == "RMSNorm") or (sa["block"] == "parallel" and sb["block"] == "sequential"):
        out.append("prenorm_rescue")
    return out


def all_edges(P, ok, ex, configs):
    """[(i, j, tags)] for every headline edge."""
    import numpy as np
    types = [_types(p, configs) for p in P]
    return [(int(i), int(j), edge_tags(P[i], P[j], types[i], types[j], bool(ex[i, j])))
            for i, j in np.argwhere(ok)]


STRATA_SKIP = ("step:",)        # the per-transition tags are recorded, the slot-level ones stratify


def sample(edges, n, seed, buildable=None, buildable_share=0.3):
    """Stratified: every rule kind gets an equal quota (or all its edges when it has fewer), a share of the
    sample is drawn from the edges emulate.py builds today (so the real / scaled builds are many), the rest
    uniformly. Deterministic in seed."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for k, (_, _, tags) in enumerate(edges):
        for t in tags:
            if not t.startswith(STRATA_SKIP):
                by[t].append(k)
    strata = sorted(by)
    quota = max(3, (n - int(n * buildable_share)) // max(len(strata), 1))
    chosen, why = [], {}

    def take(pool, q, label):
        pool = [k for k in pool if k not in why]
        for k in rng.sample(pool, min(q, len(pool))):
            why[k] = label
            chosen.append(k)
    for t in strata:
        take(by[t], quota, t)
    if buildable is not None:
        take([k for k, (i, j, _) in enumerate(edges) if (i, j) in buildable], int(n * buildable_share), "buildable")
    if len(chosen) < n:
        take(range(len(edges)), n - len(chosen), "uniform")
    return [(edges[k], why[k]) for k in chosen]


# ================================================================== member repos

def _cached(repo):
    from huggingface_hub import constants
    d = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not d.is_dir():
        return False
    return any(f.suffix in (".safetensors", ".bin") for s in d.iterdir() for f in s.iterdir())


def params_of(repo, cfg, meta):
    """Parameters of the model the config describes (emulate's count for a v1 family -- what a written host
    checkpoint holds; the Hub's count is that of a quantized / GGUF file for some repos), else the Hub's."""
    try:
        from elephantine.emulate import FAMILIES, _params, resolved
        c = resolved(cfg)
        if c.model_type in FAMILIES:
            return int(_params(c, FAMILIES[c.model_type]))
    except Exception:                                       # noqa: BLE001 -- an odd config: fall back
        pass
    n = (meta.get(repo) or {}).get("hf_total")
    if n:
        return int(n)
    h = int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model") or 0)
    L = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers") or 0)
    return 12 * L * h * h + 2 * h * int(cfg.get("vocab_size") or 0)


GATED_ORGS = ("meta-llama/", "google/gemma", "mistralai/")      # gated repos: only when already cached


def candidates(members, role, configs, meta):
    """The members emulate.py reads in this role, smallest first (cached repos first among equals), one
    list; a member it refuses is kept at the end (for the refusal reason)."""
    from elephantine.emulate import side_support
    good, bad = [], []
    for m in members:
        c = configs.get(m)
        if c is None:
            continue
        rs = side_support(c, role)
        if role == "target" and m.startswith(GATED_ORGS) and not _cached(m):
            rs = (rs or []) + ["gated repo, not cached"]
        p = params_of(m, c, meta)
        toy = any(x in m.lower() for x in ("tiny", "test", "random", "dummy", "debug"))
        (bad if rs else good).append((p > (MAX_TARGET if role == "target" else MAX_HOST), toy, not _cached(m), p, m,
                                      rs))
    good.sort()
    bad.sort()
    return good, bad


def pick_pair(A, B, configs, meta, limit=12):
    """(target repo, host repo, support reasons or None). Prefers a pair emulate.py builds with small repos."""
    from elephantine.emulate import support
    tg, tb = candidates(A["members"], "target", configs, meta)
    hg, hb = candidates(B["members"], "host", configs, meta)
    first = None
    for t in tg[:limit]:
        for h in hg[:limit]:
            rs = support(configs[t[4]], configs[h[4]])
            if rs is None:
                return t[4], h[4], None
            if first is None:
                first = (t[4], h[4], rs)
    if first is not None:
        return first
    t = (tg or tb or [(0, 0, 0, 0, A["members"][0], ["no config"])])[0]
    h = (hg or hb or [(0, 0, 0, 0, B["members"][0], ["no config"])])[0]
    return t[4], h[4], (t[5] or []) + (h[5] or []) or ["no member pair read by emulate.py"]


# ================================================================== what is missing

MISSING = [   # (needle in a refusal reason, the construction piece emulate.py lacks)
    ("not the stock transformers class", "remote-code family (no stock class to write)"),
    ("an MoE target", "MoE target (expert duplication / MoE <- MoE, dense <- top-1 MoE)"),
    ("KV-head packing", "KV-head packing (longrope table per packed head)"),
    ("several window sizes", "per-layer window sizes"),
    ("has no window setting", "window on a host family without a window setting"),
    ("mixes full and sliding", "per-layer windows on a one-window host family"),
    ("activation", "activation change (FFN rewrite)"),
    ("norm placement", "sandwich-norm change"),
    ("qk-norm", "qk-norm on one side only"),
    ("softcap", "softcap change"),
    ("bias", "bias the host lacks"),
    ("do not fit in order", "layer placement (full / sliding kinds)"),
    ("FFN: host layer", "FFN split over routed experts"),
    ("partial rotary", "partial-RoPE host / head_dim change with partial rotary"),
    ("partial_rotary_factor", "partial-RoPE host / head_dim change with partial rotary"),
    ("scaling depends on head_dim", "RoPE scaling that depends on head_dim"),
    ("hidden: host", "hidden narrower (relation width factor: mirror / carry)"),
    ("head_dim: host", "head_dim narrower"),
    ("quantized", "quantized target checkpoint"),
    ("gated repo", "gated repo, not cached"),
    ("is not a v1 family", None),                                     # filled from the model_type
    ("needs a limit", "limit / regime / floor edge (v1 writes exact constructions only)"),
    ("no stock transformers config", "no stock transformers config"),
    ("config not understood", "config not understood by emulate.py"),
]


def missing_pieces(reasons):
    out = []
    for r in reasons or []:
        for needle, piece in MISSING:
            if needle in r:
                if piece is None:
                    mt = r.split("model_type ")[1].split(" ")[0].strip("'") if "model_type " in r else "?"
                    piece = f"family not in v1: {mt}"
                out.append(piece)
                break
        else:
            out.append("other: " + r[:80])
    return sorted(set(out))


def _plan(A, B):
    from graph.edge_plan import edge_plan, format_plan
    plan = edge_plan({"slots": A["slots"], "dims": A["dims"]}, {"slots": B["slots"], "dims": B["dims"]})
    return plan, format_plan(plan, "target", "host")


def pieces_of(con, tc, hc):
    """The construction pieces emulate.py spends on a pair (for "does a big edge compose from verified small ones")."""
    from elephantine.emulate import _head_dim, _n_kv, _rope
    out = [f"family:{con.tfam.name}->{con.hfam.name}"]
    out.append("hidden:pad" if con.D > con.d else "hidden:same")
    if con.a != 1:
        out.append("resid_mult")
    L_t, L_h = tc.num_hidden_layers, hc.num_hidden_layers
    out.append("layers:same" if con.layer_pos == list(range(L_t)) else
               ("layers:spread" if con.layer_pos == [i * L_h // L_t for i in range(L_t)] else "layers:placed"))
    rep = len(con.kvmap) // max(_n_kv(tc), 1)
    out.append(f"kv:tile{'x' + str(rep) if rep > 1 else '1'}")
    if hc.num_attention_heads > len(con.qmap):
        out.append("q:pad")
    hdt, hdh = _head_dim(tc), _head_dim(hc)
    out.append("hd:same" if hdt == hdh else ("hd:int" if hdh % hdt == 0 else "hd:theta"))
    out.append(f"rope:{_rope(tc)[0]}")
    out += sorted({f"ffn:{m}" for m in con.ffn_mode})
    if con.host_window is not None:
        out.append("window:set")
    if "tie_word_embeddings" in con.config_changes:
        out.append("untie")
    return out


# ================================================================== scaled-down pairs

WIDTH_KEYS = ("hidden_size", "intermediate_size", "moe_intermediate_size", "shared_expert_intermediate_size",
              "head_dim")


def shrink(traw, hraw, target_hidden=64):
    """Both configs with every width divided by one power of two f (all ratios and comparisons kept), windows
    shortened, vocab 49152. Returns (t, h, f) or raises ValueError with the reason."""
    from elephantine.emulate import _head_dim, resolved
    t, h = copy.deepcopy(traw), copy.deepcopy(hraw)
    for c in (t, h):
        for k in ("quantization_config", "auto_map", "_name_or_path"):
            c.pop(k, None)
    hds = []
    for c in (traw, hraw):              # (the originals: emulate.resolved caches by object id, so never the copies)
        rc = resolved(c)
        hd = _head_dim(rc)
        rot = int(round(hd * float(getattr(rc, "partial_rotary_factor", 1.0) or 1.0)))
        hds += [hd // 2, rot // 2]
    vals = [int(c[k]) for c in (t, h) for k in WIDTH_KEYS if isinstance(c.get(k), int) and c.get(k)] + hds
    f = 1
    big = lambda f: (int(traw["hidden_size"]) // f > target_hidden or int(hraw["hidden_size"]) // f > 4 * target_hidden)
    while big(f) and all(v % (f * 2) == 0 for v in vals) and min(hds) // (f * 2) >= 2:
        f *= 2
    if f == 1 and int(traw["hidden_size"]) > 4 * target_hidden:
        raise ValueError("no common power-of-two factor of the widths")
    ws = [w for c in (t, h) for w in [c.get("sliding_window")] if isinstance(w, int) and w]
    for c in (t, h):
        for k in WIDTH_KEYS:
            if isinstance(c.get(k), int) and c.get(k):
                c[k] //= f
        rs = c.get("rope_scaling")
        if isinstance(rs, dict):
            for k in ("short_factor", "long_factor"):
                if isinstance(rs.get(k), list):
                    rs[k] = rs[k][::f]
        if isinstance(c.get("sliding_window"), int) and c["sliding_window"]:
            c["sliding_window"] = max(4, c["sliding_window"] * 8 // min(ws))
        c["vocab_size"] = VOCAB
        c["pad_token_id"], c["bos_token_id"], c["eos_token_id"] = None, 1, 2
        c["torch_dtype"] = "float32"
    from elephantine.emulate import FAMILIES, _params
    hc = resolved(copy.deepcopy(h))      # (a copy: resolved caches by object id and the caller may edit h)
    if hc.model_type in FAMILIES and _params(hc, FAMILIES[hc.model_type]) > MAX_SCALED:
        raise ValueError(f"the widths share no factor to shrink by (f = {f}): the host would keep "
                         f"{_params(hc, FAMILIES[hc.model_type]) / 1e6:.0f}M parameters")
    return t, h, f


def _point(raw):
    from graph.model_graph import classify, dims_of
    return {"slots": classify(raw), "dims": dims_of(raw)}


def write_target(d, raw, seed=0):
    """A target checkpoint with random weights (and the SmolLM2 tokenizer) at d."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, CONFIG_MAPPING
    from elephantine.emulate import warnings_off
    with warnings_off():
        cfg = CONFIG_MAPPING[raw["model_type"]].from_dict({k: v for k, v in raw.items() if k != "architectures"})
        cfg.architectures = raw.get("architectures")
        torch.manual_seed(seed)
        m = AutoModelForCausalLM.from_config(cfg).float()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.08)
    m.save_pretrained(d)
    AutoTokenizer.from_pretrained(TOK).save_pretrained(d)
    del m


def write_host_config(d, raw):
    os.makedirs(d, exist_ok=True)
    Path(d, "config.json").write_text(json.dumps(raw, indent=1))


# ================================================================== one check (in a subprocess)

def judge(v, exact, notes=()):
    """(passed, standard, details) for emulate.verify's result."""
    if v is None:
        return False, "none", "no verification"
    f32 = v["fp32"]
    n = len(f32["greedy"])
    f32_ok = (f32["argmax_match"] >= 0.99 and sum(g["identical"] for g in f32["greedy"]) >= 0.9 * n)
    if exact and "fp64" in v:
        tol = THETA_TOL if any("not an integer" in x for x in notes) else EXACT_TOL
        f64 = v["fp64"]
        ok = f64["rel"] <= tol and all(g["identical"] for g in f64["greedy"]) and f64["argmax_match"] == 1.0
        return ok, f"fp64 exact (rel <= {tol:g}, every continuation identical)", (
            f"fp64 rel {f64['rel']:.2e}, greedy {sum(g['identical'] for g in f64['greedy'])}/{n}; "
            f"fp32 rel {f32['rel']:.2e} top-1 {f32['argmax_match']:.1%} greedy "
            f"{sum(g['identical'] for g in f32['greedy'])}/{n}")
    return f32_ok, "high agreement (fp32 top-1 >= 99%, >= 90% identical)", (
        f"fp32 rel {f32['rel']:.2e} top-1 {f32['argmax_match']:.1%} greedy "
        f"{sum(g['identical'] for g in f32['greedy'])}/{n}")


def _summ(v):
    if v is None:
        return None
    out = {}
    for k in ("fp32", "fp64"):
        if k in v:
            r = v[k]
            out[k] = {"max_abs_diff": r["max_abs_diff"], "rel": r["rel"], "logit_scale": r["logit_scale"],
                      "top1": r["argmax_match"], "greedy_identical": sum(g["identical"] for g in r["greedy"]),
                      "greedy_n": len(r["greedy"]), "first_divergence": [g["first_divergence"] for g in r["greedy"]],
                      "seconds": r["seconds"]}
    if v.get("fp64_skipped"):
        out["fp64_skipped"] = v["fp64_skipped"]
    return out


def _on_device(dev):
    """Run emulate.verify's forward passes on dev (a GPU: the CPUs here are shared). Copies between GPUs are
    broken on this machine, so everything goes through the CPU."""
    import contextlib
    import torch
    import elephantine.emulate as E

    @torch.no_grad()
    def _run(path, dtype, inputs, new_tokens, lift):
        m = E.load_model(path, dtype).to(dev)
        ctx = E._lift() if lift else contextlib.nullcontext()
        with ctx:
            logits = [m(input_ids=x.to(dev)).logits.cpu().to(E.F64) for x in inputs]
            greedy = [m.generate(x.to(dev), attention_mask=torch.ones_like(x).to(dev), max_new_tokens=new_tokens,
                                 do_sample=False, num_beams=1, pad_token_id=0)[0, x.shape[1]:].cpu().tolist()
                      for x in inputs]
        del m
        torch.cuda.empty_cache()
        return logits, greedy
    E._run = _run


_T0 = time.time()


def _log(*a):
    print(f"[{time.time() - _T0:7.1f}s]", *a, file=sys.stderr, flush=True)


def one(target, host, out, fp64="auto", new_tokens=NEW_TOKENS, prompts=PROMPTS):
    """Build and verify one pair with emulate.py; returns a JSON-able dict (the worker's output)."""
    import torch
    torch.set_num_threads(int(os.environ.get("SPOT_THREADS", "1")))
    dev = os.environ.get("SPOT_DEVICE", "cpu")
    if dev != "cpu":
        _on_device(dev)
    from elephantine.emulate import Refused, emulate, fetch_config, resolved, plan_construction
    try:
        r = emulate(target, host, out, fp64=fp64, prompts=prompts, new_tokens=new_tokens, log=_log)
    except Refused as e:
        return {"refused": e.code, "reasons": e.reasons}
    finally:
        shutil.rmtree(out, ignore_errors=True)
    traw, hraw = fetch_config(target), fetch_config(host)
    con = plan_construction(traw, hraw)
    return {"verification": r["verification"], "exact": r["edge"]["exact"], "notes": r["notes"],
            "pieces": pieces_of(con, resolved(traw), resolved(hraw)), "lines": r["construction"]["lines"],
            "config_changes": {k: v for k, v in r["config_changes"].items() if k != "layer_types"}}


def one_scaled(traw, hraw, root):
    """The scaled-down check of a pair (raw configs): returns the worker dict, with "scaled" facts."""
    t, h, f = shrink(traw, hraw)
    A0, B0, A, B = _point(traw), _point(hraw), _point(t), _point(h)
    if A["slots"] is None or B["slots"] is None:
        return {"scaled_error": "the scaled configs do not classify"}
    from graph.model_graph import emulates
    if not emulates(B, A):
        return {"scaled_error": f"the scaled pair (f = {f}) is not an edge", "f": f}
    same = (A["slots"] == A0["slots"] and B["slots"] == B0["slots"])
    td, hd = os.path.join(root, "t"), os.path.join(root, "h")
    write_target(td, t)
    write_host_config(hd, h)
    try:
        r = one(td, hd, os.path.join(root, "out"), fp64=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    r["scaled"] = {"f": f, "same_slots": same, "hidden": [t["hidden_size"], h["hidden_size"]],
                   "layers": [t.get("num_hidden_layers"), h.get("num_hidden_layers")],
                   "exact": emulates(B, A, mode="exact")}
    return r


def _pick_gpu(min_free_mib=8000):
    """The GPU with the most free memory (one device per worker), or the CPU."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
        free = sorted((int(f), i) for i, f in (x.split(",") for x in q.strip().splitlines()))
        if free and free[-1][0] >= min_free_mib:
            return {"CUDA_VISIBLE_DEVICES": free[-1][1].strip(), "SPOT_DEVICE": "cuda:0"}
    except Exception:                                       # noqa: BLE001 -- no GPU: the CPU
        pass
    return {"SPOT_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": ""}


def run_worker(job, timeout, tries=3):
    """run_worker_once, retried on a CUDA out-of-memory (the GPUs are shared), then on the CPU."""
    for k in range(tries):
        res = run_worker_once(job, timeout)
        if "out of memory" not in res.get("error", ""):
            return res
        time.sleep(30 * (k + 1))
    return run_worker_once(job, timeout, {"SPOT_DEVICE": "cpu"})


def run_worker_once(job, timeout, extra_env=None):
    """Run one check in a subprocess (a fresh process per edge: memory is returned, a hang is killed)."""
    WORK.mkdir(parents=True, exist_ok=True)
    root = tempfile.mkdtemp(dir=WORK)
    jf = Path(root, "job.json")
    jf.write_text(json.dumps(job))
    env = dict(os.environ, PYTHONPATH=str(ROOT), **(extra_env or {}))
    if "SPOT_DEVICE" not in env:
        env.update(_pick_gpu())
    t0 = time.time()
    (WORK / "logs").mkdir(exist_ok=True)
    logf = WORK / "logs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{Path(root).name}.log"
    try:
        with open(logf, "w") as lf:
            subprocess.run([sys.executable, "-m", "verify.spot.run_spot_checks", "--worker", str(jf)], cwd=ROOT,
                           env=env, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rf = Path(root, "result.json")
        if rf.exists():
            res = json.loads(rf.read_text())
        else:
            res = {"error": logf.read_text()[-1500:]}
    except subprocess.TimeoutExpired:
        res = {"timeout": timeout, "log_tail": logf.read_text()[-400:]}
    finally:
        shutil.rmtree(root, ignore_errors=True)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def worker(jobfile):
    job = json.loads(Path(jobfile).read_text())
    root = str(Path(jobfile).parent)
    try:
        if job["mode"] == "real":
            host = job["host"]
            if job.get("hraw"):                     # the snapshot's config (only the host's config.json is read)
                host = os.path.join(root, "host-config")
                write_host_config(host, job["hraw"])
            res = one(job["target"], host, os.path.join(root, "out"))
        else:
            res = one_scaled(job["traw"], job["hraw"], root)
    except ValueError as e:
        res = {"scaled_error": str(e)} if job["mode"] == "scaled" else {"error": f"ValueError: {e}"[:1500]}
    except Exception as e:                                   # noqa: BLE001 -- recorded, not hidden
        msg = f"{type(e).__name__}: {e}"
        res = {"error": msg[:1500], "trace": traceback.format_exc()[-2500:]}
    if "verification" in res:
        res["verification"] = _summ(res["verification"])
    Path(root, "result.json").write_text(json.dumps(res, default=str))


# ================================================================== the driver

def check_edge(i, j, tags, why, P, ex, configs, meta, timeout, build=True, pre=None):
    rec = copy.deepcopy(pre) if pre is not None else prepare(i, j, tags, why, P, ex, configs, meta)
    if rec.get("status"):
        return rec
    if not build:
        rec.update(status="buildable")
        return rec
    t, h, exact = rec["target"], rec["host"], rec["exact"]
    small = (rec["target_params"] or 0) <= MAX_TARGET and (rec["host_params"] or 0) <= MAX_HOST
    if small:
        rec["mode"] = "real"
        res = run_worker({"mode": "real", "target": t, "host": h, "hraw": configs[h]}, timeout)
        if "error" in res and ("401" in res["error"] or "403" in res["error"] or "gated" in res["error"].lower()
                               or "Repository Not Found" in res["error"] or "does not appear to have" in res["error"]):
            rec["real_error"] = res["error"][:300]
            small = False
    if not small:
        rec["mode"] = "scaled"
        res = run_worker({"mode": "scaled", "traw": configs[t], "hraw": configs[h]}, timeout)
    return finish(rec, res, exact)


def prepare(i, j, tags, why, P, ex, configs, meta):
    """The config-level part of a check: the plan, the member pair, and the refusal when emulate.py lacks a piece."""
    A, B = P[i], P[j]
    exact = bool(ex[i, j])
    plan, plan_text = _plan(A, B)
    t, h, reasons = pick_pair(A, B, configs, meta)
    rec = {"target_shape": i, "host_shape": j, "target": t, "host": h, "stratum": why, "tags": tags,
           "exact": exact, "plan": {"exact": plan["exact"], "slots": [
               f"{r['slot']} {r['target']}->{r['host']} [{r['kind']}{': ' + r['currency'] if r['currency'] else ''}"
               f"{'' if r['exact'] else ', limit'}]" for r in plan["slots"]],
               "axes": [f"{r['axis']} need {r['need']} have {r['have']}" for r in plan["axes"] if r["need"] != r["have"]]},
           "target_params": params_of(t, configs[t], meta) if t in configs else None,
           "host_params": params_of(h, configs[h], meta) if h in configs else None}
    tm = configs.get(t, {}).get("model_type")
    hm = configs.get(h, {}).get("model_type")
    rec["families"] = f"{tm}->{hm}"
    if tm != hm and "cross_family" not in tags:
        rec["tags"] = tags + ["cross_family_members"]
    if not exact:
        from graph.edge_plan import approx_reason
        reasons = [approx_reason(plan) + "; needs a limit"] + (reasons or [])
    if reasons:
        rec.update(status="unsupported", reasons=reasons, missing=missing_pieces(reasons), plan_text=plan_text)
    return rec


def finish(rec, res, exact):
    rec["seconds"] = res.get("seconds")
    if "timeout" in res:
        rec.update(status="timeout")
    elif "refused" in res:
        rec.update(status="unsupported", reasons=res["reasons"], missing=missing_pieces(res["reasons"]))
    elif "scaled_error" in res:
        rec.update(status="scaled_error", reasons=[res["scaled_error"]])
    elif "error" in res:
        rec.update(status="error", error=res["error"], trace=res.get("trace", "")[-800:])
    else:
        v = res["verification"]
        ok, std, det = judge(_unsumm(v), exact and res.get("scaled", {}).get("exact", True), res.get("notes", []))
        rec.update(status="pass" if ok else "FAIL", standard=std, detail=det, verification=v,
                   pieces=res.get("pieces"), construction=res.get("lines"), notes=res.get("notes"),
                   scaled=res.get("scaled"))
    return rec


def _unsumm(v):
    """judge() reads emulate.verify's shape; rebuild the fields it uses from the summary."""
    if v is None:
        return None
    out = {}
    for k in ("fp32", "fp64"):
        if k in v:
            r = v[k]
            out[k] = {"rel": r["rel"], "argmax_match": r["top1"],
                      "greedy": [{"identical": i < r["greedy_identical"]} for i in range(r["greedy_n"])]}
    return out


def buildable_pairs(P, ok, ex, configs):
    """Shape pairs (i, j) of the headline relation that emulate.py builds today with some pair of members
    (config level: emulate.support on every exact edge, one member per model_type on each side)."""
    import numpy as np
    from elephantine.emulate import side_support, support
    tg, hs = {}, {}
    for i, p in enumerate(P):
        for role, d in (("target", tg), ("host", hs)):
            by = {}
            for m in p["members"]:
                c = configs.get(m)
                if c is not None and side_support(c, role) is None:
                    by.setdefault(c.get("model_type"), m)
            d[i] = list(by.values())
    out = set()
    for i, j in np.argwhere(ex & ok):
        if any(support(configs[a], configs[b]) is None for a in tg[i] for b in hs[j]):
            out.add((int(i), int(j)))
    return out


def run(seed=0, n=240, timeout=900, out=RESULTS, build=True, resume=True, log=print, jobs=1, retry=()):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    P, ok, ex, configs, meta = load_relation()
    edges = all_edges(P, ok, ex, configs)
    bp = buildable_pairs(P, ok, ex, configs)
    picks = sample(edges, n, seed, bp)
    done = {}
    if resume and Path(out).exists():
        keep = []
        for line in Path(out).read_text().splitlines():
            r = json.loads(line)
            if r.get("seed") == seed and r["status"] in retry:
                continue                                    # checked again (e.g. a timeout on a loaded machine)
            keep.append(line)
            if r.get("seed") == seed:
                done[(r["target_shape"], r["host_shape"])] = r
        Path(out).write_text("".join(x + "\n" for x in keep))
    log(f"{len(edges)} headline edges, {len(bp)} buildable by emulate.py; {len(picks)} sampled (seed {seed}); "
        f"{len(done)} already in {out}")
    lock = threading.Lock()
    # the pair choice (config level) is done up front, in this thread: emulate.resolved's cache is not thread-safe
    todo = [(k, (i, j, tags), why) for k, ((i, j, tags), why) in enumerate(picks) if (i, j) not in done]
    with open(out, "a") as fh:
        def go(item):
            k, (i, j, tags), why = item
            t0 = time.time()
            rec = check_edge(i, j, tags, why, P, ex, configs, meta, timeout, build, pre=pre[k])
            rec["seed"] = seed
            with lock:
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                log(f"[{k + 1}/{len(picks)}] {rec['status']:11} {rec.get('mode', ''):6} {rec['target']} <- "
                    f"{rec['host']}  ({why}; {time.time() - t0:.0f}s) "
                    f"{rec.get('detail', '') or '; '.join(rec.get('missing', []))[:120]}")
        pre = {k: prepare(i, j, tags, why, P, ex, configs, meta) for k, (i, j, tags), why in todo}
        with ThreadPoolExecutor(max(1, jobs)) as pool:
            list(pool.map(go, todo))
    return coverage_numbers(edges, bp)


def coverage_numbers(edges, bp):
    return {"edges": len(edges), "buildable": len(bp), "fraction": len(bp) / max(len(edges), 1)}


# ================================================================== the report

def load_results(path=RESULTS):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def report(results, cov=None, seed=None, whole=None):
    if seed is not None:
        results = [r for r in results if r.get("seed") == seed]
    st = collections.Counter(r["status"] for r in results)
    built = [r for r in results if r["status"] in ("pass", "FAIL")]
    L = ["# Spot checks of the emulation relation", "",
         "Random edges of the headline relation (`emul_matrix(mode=\"payable\")` over the zoo's shapes), stratified "
         "by the kind of rule each spends, built with `elephantine/emulate.py` where it can and verified end to end. "
         "Written by `python -m verify.spot.run_spot_checks` (raw rows: `verify/spot/results.jsonl`).", "",
         f"- edges sampled: **{len(results)}** (seed{'s' if len({r.get('seed') for r in results}) > 1 else ''} "
         f"{sorted({r.get('seed') for r in results})})",
         f"- built and verified: **{len(built)}** -- pass **{st['pass']}**, FAIL **{st['FAIL']}** "
         f"(real checkpoints {sum(r.get('mode') == 'real' for r in built)}, scaled-down "
         f"{sum(r.get('mode') == 'scaled' for r in built)})",
         f"- not built: unsupported by emulate.py {st['unsupported']}, scaled pair not an edge / not shrinkable "
         f"{st['scaled_error']}, error {st['error']}, timeout {st['timeout']}", ""]
    if cov:
        L += [f"**Directly buildable today:** {cov['buildable']:,} of the {cov['edges']:,} headline edges "
              f"(**{cov['fraction']:.1%}**) have a member pair that `emulate.py` builds (config level: "
              f"`emulate.support` on every exact edge, one member per model_type on each side).", ""]
    L += ["## Pass rate per rule kind", "",
          "A row counts every sampled edge carrying the tag (an edge carries several). *built* = real or scaled "
          "build verified; *pass* is over the built ones.", "",
          "| rule kind | sampled | built | pass | FAIL | unsupported | other |", "|---|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for r in results:
        for t in r["tags"]:
            if not t.startswith("step:"):
                by[t].append(r)
    for t in sorted(by, key=lambda t: (-len(by[t]), t)):
        rs = by[t]
        c = collections.Counter(r["status"] for r in rs)
        b = c["pass"] + c["FAIL"]
        pct = f" ({c['pass'] / b:.0%})" if b else ""
        L.append(f"| {t} | {len(rs)} | {b} | {c['pass']}{pct} | {c['FAIL']} | {c['unsupported']} | "
                 f"{len(rs) - b - c['unsupported']} |")
    L += ["", "Per slot transition (built edges only):", "",
          "| transition | built | pass |", "|---|---|---|"]
    tr = collections.defaultdict(list)
    for r in built:
        for t in r["tags"]:
            if t.startswith("step:"):
                tr[t[5:]].append(r)
    for t in sorted(tr):
        L.append(f"| {t} | {len(tr[t])} | {sum(r['status'] == 'pass' for r in tr[t])} |")
    if not tr:
        L.append("| (none) | | |")
    L += ["", "## Failures", ""]
    fails = [r for r in results if r["status"] == "FAIL"]
    if not fails:
        L.append("None: every edge that was built reproduced its target to the standard.")
    for r in fails:
        L += [f"### {r['target']} <- {r['host']} ({r.get('mode')})", "",
              f"- tags: {', '.join(r['tags'])}", f"- {r.get('detail')}",
              f"- plan: {'; '.join(r['plan']['slots']) or 'no slot differs'}",
              f"- construction pieces: {', '.join(r.get('pieces') or [])}",
              f"- minimal cause: {fail_cause(r)}",
              f"- repro: `python -m verify.spot.run_spot_checks --one {r['target']} {r['host']}"
              f"{' --scaled' if r.get('mode') == 'scaled' else ''}`", ""]
    p4 = HERE / "results_round2-p4.jsonl"
    if p4.exists():
        f4 = [r for r in load_results(p4) if r["status"] == "FAIL"]
        if f4:
            L += ["### Failures of the earlier pass on round2-p4 (the same sampler, seed 0, before pass 5)", "",
                  "Each is an `emulate.py` defect (two causes) on an edge the relation still claims; the round2-p5 sample did not "
                  "draw them (raw rows: `verify/spot/results_round2-p4.jsonl`).", ""]
            for r in f4:
                L += [f"- **{r['target']} <- {r['host']}** ({r.get('mode')}): {r.get('detail')}. Cause: {fail_cause(r)}. "
                      f"Repro: `python -m verify.spot.run_spot_checks --one {r['target']} {r['host']}`"]
            L.append("")
    L += ["## Built, by construction piece", "",
          "Every piece `emulate.py` used on a built edge, and how many of those edges passed. A large edge composes "
          "from these pieces: its plan is covered when each of its pieces passed somewhere at small scale.", "",
          "| piece | built | pass |", "|---|---|---|"]
    pc = collections.defaultdict(list)
    for r in built:
        for p in r.get("pieces") or []:
            pc[p].append(r)
    for p in sorted(pc):
        L.append(f"| {p} | {len(pc[p])} | {sum(r['status'] == 'pass' for r in pc[p])} |")
    real_ok = {p for r in built if r.get("mode") == "real" and r["status"] == "pass" for p in r.get("pieces") or []}
    sc = [r for r in built if r.get("mode") == "scaled"]
    comp = [r for r in sc if set(r.get("pieces") or []) <= real_ok]
    L += ["", f"Scaled-down edges (pairs too big to run here): {len(sc)}; of those, {len(comp)} use only pieces "
              f"also verified on a real small checkpoint pair; the rest need a piece verified only at scaled-down "
              f"size: {', '.join(sorted({p for r in sc for p in (r.get('pieces') or [])} - real_ok)) or 'none'}.", ""]
    L += ["## Why the rest is not built (ranked)", "",
          "The missing construction pieces behind each unsupported sampled edge (an edge may lack several), "
          "most frequent first -- the order in which to extend `emulate.py`.", "",
          "| missing piece | edges |", "|---|---|"]
    mc = collections.Counter(p for r in results if r["status"] == "unsupported" for p in r.get("missing", []))
    for p, k in mc.most_common():
        L.append(f"| {p} | {k} |")
    other = [r for r in results if r["status"] in ("error", "timeout", "scaled_error")]
    if other:
        L += ["", "## Not verified for another reason", "", "| target | host | status | why |", "|---|---|---|---|"]
        for r in other:
            why = r.get("error") or "; ".join(r.get("reasons", [])) or f"over {r.get('seconds')}s"
            L.append(f"| {r['target']} | {r['host']} | {r['status']} | {why[:160].replace('|', '/')} |")
    L += composition_section(results, whole)
    L += catalog_section()
    return "\n".join(L) + "\n"


def fail_cause(r):
    """The responsible rule of a FAIL, from what the record shows."""
    import re
    v = r.get("verification") or {}
    for line in r.get("construction") or []:
        m = re.search(r"theta [0-9.e+]+ -> ([0-9.e+]+)", line)
        if m and float(m.group(1)) > 3.4e38:
            return ("emulate.py, not the relation: the non-integer head_dim ratio rescales RoPE theta to "
                    f"{float(m.group(1)):.3g}, past fp32's range, and inv_freq is computed in fp32 at init, so the "
                    "host's frequencies past pair 0 are 0 (minimal repro: tests/test_spot_checks.py::"
                    "test_theta_rescale_overflows_fp32, llama head_dim 6 -> 128)")
    f64 = v.get("fp64") or v.get("fp32") or {}
    if f64 and f64.get("rel", 1) <= THETA_TOL and f64.get("greedy_identical", 0) < f64.get("greedy_n", 0):
        return ("not the construction: every prompt logit matches, yet greedy generation differs -- the target repo's "
                "generation_config.json (e.g. a second eos id) is not written to the host's directory by emulate.py, "
                "so the target stops where the host continues")
    return "see the construction pieces"


def edge_types(rec):
    """The catalog reduction types (verify/spot/reduction_types.py ids) a sampled edge's plan spends."""
    out = {"capacity:hidden"}
    for t in rec["tags"]:
        if t.startswith("step:"):
            _, slot, ab = t.split(":", 2)
            a, b = ab.split("->")
            tid = f"slot:{slot}:{a}->{b}"
            host_mt = (rec.get("families") or "->").split("->")[1]
            if tid == "slot:norm:LayerNorm->RMSNorm" and host_mt not in ("phi", "bunny-phi", "ctrl"):
                tid += " (host without an lm_head bias)"       # the catalog's finding: the final beta is lost
            if slot == "ffn" and a == "vanilla" and not rec.get("host_ffn_bias", 1):
                tid += " (host without an FFN bias)"          # the catalog's finding: the up branch cannot be held at 1
            out.add(tid)
    tags = set(rec["tags"])
    if "kv_pack" in tags:
        out.add("capacity:kv_pack:rope" if "rope" in json.dumps(rec.get("plan", {})) else "capacity:kv_pack:nope")
    if "moe<-moe" in tags:
        out.add("moe:dup:R:m" if "moe_dup" in tags else "moe:dup:R:1")
    if "dense_host_for_moe" in tags:
        out.add("moe:dense_host_top1")
    if "moe_host_for_dense" in tags:
        out.add("moe:host_for_dense")
    if "recurrent" in tags:
        out.add("recurrent:*")
    if "rope_table" in tags:
        out.add("capacity:rot_gate")
    return out


def composition_section(results, whole=None):
    """Does each sampled edge's plan compose from reduction types that have a verified catalog instance?"""
    from verify.spot.catalog import CATALOG, load_all
    load_all()
    f = HERE / "catalog_results.json"
    if not f.exists():
        return []
    rows = {r["entry"]: r for r in json.loads(f.read_text())}
    ok_types = {t for t, es in CATALOG.items() for e in es if rows.get(e.check.__name__, {}).get("status") == "pass"}
    ok_types |= {"recurrent:*"} if any(t.startswith("recurrent:") for t in ok_types) else set()
    cover, lack = 0, collections.Counter()
    for r in results:
        miss = edge_types(r) - ok_types
        cover += not miss
        lack.update(miss)
    wc, wlack = 0, collections.Counter()
    for r in whole or []:
        miss = edge_types(r) - ok_types
        wc += not miss
        wlack.update(miss)
    L = ["", "## Every sampled edge against the reduction catalog", "",
         f"Whether or not `emulate.py` builds it, a sampled edge's plan is a composition of reduction types; "
         f"**{cover} of {len(results)}** sampled edges ({cover / max(len(results), 1):.0%}) use only types with a "
         f"passing end-to-end catalog instance (below). (Composition is checked type by type, not as one build: the "
         f"types the rest use without a passing instance --)"]
    if whole:
        L[-1] += (f" Over the whole headline relation: **{wc:,} of {len(whole):,}** edges "
                  f"({wc / len(whole):.1%}) compose from verified types.")
    L += ["", "| reduction type without a passing instance | sampled edges | headline edges |", "|---|---|---|"]
    for t in sorted(set(lack) | set(wlack), key=lambda t: -wlack.get(t, 0)):
        L.append(f"| {t} | {lack.get(t, 0)} | {wlack.get(t, 0):,} |")
    return L


def catalog_section():
    """The reduction catalog table (verify/spot/catalog_results.json, verify/spot/reduction_types.json)."""
    from verify.spot.catalog import CATALOG, UNINSTANTIATED, load_all
    load_all()
    rows = {r["entry"]: r for r in json.loads((HERE / "catalog_results.json").read_text())} if (
        HERE / "catalog_results.json").exists() else {}
    types = json.loads((HERE / "reduction_types.json").read_text())
    used = [t for t in types if t["headline"]]
    L = ["", "## Reduction catalog: one end-to-end instance per reduction type", "",
         f"`verify/spot/reduction_types.py` lists every reduction type the headline relation uses, from the relation's "
         f"code ({len(used)} of {len(types)} listed types; the rest are one-hop orders the headline does not pay). "
         f"Each has an instance in `verify/spot/catalog/` that builds the host from the target's weights and runs "
         f"both (8 prompts, 32 greedy tokens, fp64); `tests/test_reduction_catalog.py` runs them and checks that "
         f"every type has one. *real* = a real checkpoint, else a small synthetic pair (stock transformers classes, "
         f"or the reference decoder `catalog/_ref.py` where no stock class has the slot value).", ""]
    unin = [t for t in used if t["id"] in UNINSTANTIATED]
    fails = [(t["id"], e) for t in used for e in CATALOG.get(t["id"], [])
             if rows.get(e.check.__name__, {}).get("status") in ("FAIL", "error")]
    finds = [(t["id"], e) for t in used for e in CATALOG.get(t["id"], []) if e.finding]
    L += ["### Findings", ""]
    if not (unin or fails or finds):
        L.append("None.")
    for t in unin:
        L.append(f"- **{t['id']}: no end-to-end instance.** {UNINSTANTIATED[t['id']]}")
    seen = set()
    for tid, e in finds:
        if e.check.__name__ in seen:
            continue
        seen.add(e.check.__name__)
        r = rows.get(e.check.__name__, {})
        L.append(f"- **{tid}** (`{e.check.__name__}`, {e.instance}): {e.finding}. Measured: {r.get('result', 'not run')}. "
                 f"Repro: `pytest tests/test_reduction_catalog.py -k {e.check.__name__}`")
    for tid, e in fails:
        r = rows.get(e.check.__name__, {})
        L.append(f"- **{tid}** (`{e.check.__name__}`): {r.get('status')}: {r.get('result')}")
    L += ["", "### Table", "", "| type | currency | instance (target -> host) | real / synthetic | standard | result |",
          "|---|---|---|---|---|---|"]
    for t in used:
        es = CATALOG.get(t["id"], [])
        cur = t.get("currency") or t.get("kind")
        if not es:
            L.append(f"| {t['id']} | {cur} | -- | -- | -- | **not instantiated** |")
            continue
        for e in es:
            r = rows.get(e.check.__name__, {})
            st = r.get("status", "not run")
            res = f"{st}: {r.get('result', '')}" if st != "pass" else f"pass ({r.get('result', '').split(' (')[0]})"
            L.append(f"| {t['id']} | {cur} | {e.instance.replace('|', '/')} | "
                     f"{'real' if e.real else 'synthetic'} | {e.standard} | {res.replace('|', '/')} |")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per edge")
    ap.add_argument("--jobs", type=int, default=4, help="edges checked at once (one subprocess each)")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--plan-only", action="store_true", help="sample and classify, build nothing")
    ap.add_argument("--fresh", action="store_true", help="ignore rows of this seed already in --out")
    ap.add_argument("--retry", default="", help="comma-separated statuses to check again (e.g. timeout,error)")
    ap.add_argument("--report", action="store_true", help="write SPOT.md from --out")
    ap.add_argument("--one", nargs=2, metavar=("TARGET", "HOST"), help="check one pair")
    ap.add_argument("--scaled", action="store_true", help="with --one: the scaled-down check")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        worker(a.worker)
        return 0
    if a.one:
        from elephantine.emulate import fetch_config
        t, h = a.one
        if a.scaled:
            res = run_worker({"mode": "scaled", "traw": fetch_config(t), "hraw": fetch_config(h)}, a.timeout)
        else:
            res = run_worker({"mode": "real", "target": t, "host": h}, a.timeout)
        rec = finish({"target": t, "host": h}, res, True)
        print(json.dumps(rec, indent=1, default=str))
        return 0 if rec["status"] == "pass" else 1
    if a.report:
        P, ok, ex, configs, meta = load_relation()
        edges = all_edges(P, ok, ex, configs)
        cov = coverage_numbers(edges, buildable_pairs(P, ok, ex, configs))
        types = [_types(p, configs) for p in P]
        fb = lambda j: int(P[j]["dims"].get("ffn_bias", 0) or 0)
        whole = [{"tags": tags, "families": "->" + ("phi" if types[j] & {"phi", "bunny-phi", "ctrl"} else "x"),
                  "host_ffn_bias": fb(j)} for i, j, tags in edges]
        res = load_results(a.out)
        for r in res:
            r["host_ffn_bias"] = fb(r["host_shape"])
        SPOT_MD.write_text(report(res, cov, whole=whole))
        print(f"wrote {SPOT_MD}")
        return 0
    if a.fresh and Path(a.out).exists():
        keep = [x for x in Path(a.out).read_text().splitlines() if x.strip() and json.loads(x).get("seed") != a.seed]
        Path(a.out).write_text("".join(x + "\n" for x in keep))
    cov = run(a.seed, a.n, a.timeout, a.out, build=not a.plan_only, jobs=a.jobs,
              retry=tuple(x for x in a.retry.split(",") if x))
    print(json.dumps(cov))
    return 0


if __name__ == "__main__":
    sys.exit(main())
