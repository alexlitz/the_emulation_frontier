"""Every number the blog post quotes, computed from the code, with provenance.

Writes post_numbers.json (default dist/post_numbers.json):

    {"meta": {...}, "numbers": {key: {"value", "kind", "source", "phrase", "category",
                                      "anchors", ["aliases"], ["optional"]}}}

  value     the computed value, in the unit the post prints it in (a count, a percent, T params, ...)
  kind      how graph/check_post.py compares the draft's text with it:
              int     exact integer
              approx  agrees at the draft's displayed precision (rounded or truncated: "~22T" vs 22.07,
                      "0.012" vs 0.0117, "~2,550" vs 2,546 -- a trailing zero is a displayed digit of 10)
              name    a model name: the draft's name matches if its tokens are a subset of one alias's
              bool    on / off the frontier (the draft's check / cross mark)
              tokens  a shape string: every token in value must appear in the draft's cell
  source    the function (module.function) that produced the value
  phrase    the draft phrase the value backs (human-readable)
  anchors   where graph/check_post.py looks for it in the draft:
              {"re": pattern}  one capture group = the value. "<N>" stands for a number capture
                               (3,933 / 3{,}933 / 0.012) with any leading $ * ~ \\mathbf{ \\sim\\! markup.
              {"table": header pattern, "row": index | {"col": c, "re": pattern}, "col": c,
               ["re": pattern inside the cell, one capture group; default the first number]}
  aliases   kind name: the accepted names (e.g. every model holding a tied record)
  optional  a missing anchor is not reported

Nothing here re-derives a result: each section calls the code that produced the number for the post
(graph.frontier, graph.currency_curve, graph.frontier_over_time, graph.figures, graph.minimal_emulator,
graph.model_graph.zoo_data's in-degree, act.act_swap_report, act.activation_cross, attn.mixer_gap_report).
The OpenRouter / Artificial Analysis top-ten lists are INPUT data (August 2026), mapped to Hub repos below.

Heavy sections are cached under dist/.post_numbers_cache, keyed by a hash of the source and data files
each depends on, so a re-run after an unrelated change is quick (the activation cross-matrix fit is the
slow one, ~6 min cold).

Run:  python -m graph.post_numbers [--out dist/post_numbers.json] [--no-cache]
Then: python -m graph.check_post DRAFT.md dist/post_numbers.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = "dist/post_numbers.json"
DEFAULT_RELEASE_OUT = "dist/release_numbers.json"
CACHE_DIR = "dist/.post_numbers_cache"
JSON = "graph/hub_10k.json"

# --- input data: the August 2026 top-ten OPEN models (draft table), mapped to the Hub repo that stands for
# each in graph/hub_10k.json (a quantized upload where that is the only full-size checkpoint on the Hub)
OPENROUTER_TOP10 = [
    ("Hunyuan-Hy3", "tencent/Hy3"),
    ("DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash"),
    ("DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro"),
    ("MiMo-V2.5", "XiaomiMiMo/MiMo-V2.5"),
    ("Nemotron-3-Ultra", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"),
    ("GLM-5.2", "zai-org/GLM-5.2"),
    ("Kimi-K2.6", "nvidia/Kimi-K2.6-NVFP4"),
    ("MiniMax-M3", "nvidia/MiniMax-M3-NVFP4"),
    ("Step-3.7-Flash", "stepfun-ai/Step-3.5-Flash"),      # 3.7 is not on the Hub; 3.5 is the closest
    ("gpt-oss-120b", "openai/gpt-oss-120b"),
]
AA_TOP10 = [
    ("Kimi-K3", "nvidia/Kimi-K3-NVFP4"),
    ("GLM-5.2", "zai-org/GLM-5.2"),
    ("DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro"),
    ("Nemotron-3-Ultra", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"),
    ("DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash"),
    ("MiMo-V2.5-Pro", "XiaomiMiMo/MiMo-V2.5-Pro"),
    ("MiniMax-M3", "nvidia/MiniMax-M3-NVFP4"),
    ("Qwen3-235B", "Qwen/Qwen3-235B-A22B"),
    ("gpt-oss-120b", "openai/gpt-oss-120b"),
    ("Mistral-Medium-3.5", "nvidia/Mistral-Medium-3.5-128B-NVFP4"),
]
OVER_TIME_DATES = ["2022-04", "2023-01", "2024-01", "2024-07", "2025-01", "2025-07", "2026-01", "2026-11"]
LLAMA_LADDER = [("8B", "meta-llama/Llama-3.1-8B"), ("70B", "meta-llama/Llama-3.1-70B"),
                ("405B", "meta-llama/Llama-3.1-405B-FP8")]
IN_DEGREE_MODELS = {"pythia-14m": "EleutherAI/pythia-14m", "mamba-130m": "state-spaces/mamba-130m-hf",
                    "rwkv-4-169m": "RWKV/rwkv-4-169m-pile", "Swedish0.1M": "qwrt/Swedish0.1M",
                    "SimpleStories-1.25M": "SimpleStories/SimpleStories-1.25M"}
TOP_N = 7

# --- anchor helpers ----------------------------------------------------------------------------------


def R(pattern):
    return {"re": pattern}


def T(header, row, col, inner=None):
    a = {"table": header, "row": row, "col": col}
    if inner:
        a["re"] = inner
    return a


def rowname(name):
    """a table row whose first cell is `name` (bold or not)"""
    return {"col": 0, "re": r"^\**" + name.replace(".", r"\.").replace("+", r"\+") + r"\**$"}


TOP_TABLE = r"\| frontier model \| shape \| emulates \|"
CURRENCY_TABLE = r"\| currency budget \| frontier \|"
TIME_TABLE = r"\| as of \| unique configs \| maximal set \|"
OR_TABLE = r"\| OpenRouter \(usage\) \| on frontier\?"
ACT_TABLE = r"\| ReLU \| relu² \| SiLU \|"
ACT_TABLE_NAMES = {"ReLU": "ReLU", "relu2": "relu²", "SiLU": "SiLU", "GeLU": "GeLU", "mish": "mish",
                   "swigluoai": "swigluoai", "situ": "situ", "xielu": "xIELU"}


def act_cross_numbers(N, x):
    """actx.<target><-<host> for every off-diagonal cell of the activation cross matrix. A cell whose target or
    host is not one of the post's 8 x 8 table (ACT_TABLE_NAMES; e.g. GeLU_tanh, which act.activation_cross added
    as its own activation) is recorded without an anchor: the draft has no cell for it to check."""
    c = "activation cross matrix"
    cols = list(ACT_TABLE_NAMES)
    for Y, row in x.items():
        for X, v in row.items():
            if X == Y:
                continue
            anchors = []
            if X in ACT_TABLE_NAMES and Y in ACT_TABLE_NAMES:
                anchors = [T(ACT_TABLE, rowname(r"\*\*" + ACT_TABLE_NAMES[Y] + r"\*\*"), 1 + cols.index(X),
                             r"^(?!.*\\to)[^\d]*(\d+(?:\.\d+)?)")]
            N.put(f"actx.{Y}<-{X}", v, "act.activation_cross.matrix(method='minimax')",
                  f"activation table: target {Y}, host {X}", c, anchors, kind="approx", optional=True)


class Numbers:
    def __init__(self):
        self.d = {}

    def put(self, key, value, source, phrase, category, anchors, kind="int", **extra):
        assert key not in self.d, key
        if kind == "int":
            value = int(value)
        elif kind == "approx":
            value = float(value)
        self.d[key] = {"value": value, "kind": kind, "source": source, "phrase": phrase,
                       "category": category, "anchors": anchors, **extra}


# --- cache -------------------------------------------------------------------------------------------


def _fingerprint(patterns, fn):
    """hash of the section function's own source, of every file it depends on, and of the version of each
    "pkg:<name>" package it depends on"""
    import importlib
    import inspect
    h = hashlib.sha256(inspect.getsource(fn).encode())
    for p in sorted(p for p in patterns if p.startswith("pkg:")):
        h.update(f"{p}=={importlib.import_module(p[4:]).__version__}".encode())
    files = sorted({f for p in patterns if not p.startswith("pkg:") for f in glob.glob(os.path.join(ROOT, p))})
    for f in files:
        h.update(os.path.relpath(f, ROOT).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:20]


def cached(name, deps, fn, use_cache=True):
    """fn() -> JSON-able, cached under CACHE_DIR keyed by the hash of the dependency files."""
    path = os.path.join(ROOT, CACHE_DIR, f"{name}-{_fingerprint(deps, fn)}.json")
    if use_cache and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    t = time.time()
    out = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for old in glob.glob(os.path.join(ROOT, CACHE_DIR, f"{name}-*.json")):
        os.remove(old)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"  [{name}] computed in {time.time() - t:.0f}s")
    return out


GRAPH_DEPS = ["graph/model_graph.py", "graph/frontier.py", "graph/currency_curve.py", "graph/figures.py",
              "graph/frontier_over_time.py", "graph/minimal_emulator.py", "graph/hub_10k.json",
              "graph/model_meta.json"]
# loaded_split -> elephantine.loader_coverage reads tests/test_stock_loaders.py (CASES / REAL / TF5_CASES) and
# asks the installed transformers which architectures are stock classes
ELEPH_DEPS = GRAPH_DEPS + ["elephantine/*.py", "tests/test_stock_loaders.py", "tests/conftest.py",
                           "pkg:transformers"]

# --- sections: each returns plain JSON data, cached -----------------------------------------------------


def graph_section():
    """Everything that comes from the headline relation over the snapshot."""
    import numpy as np

    import graph.frontier_over_time as fot
    import graph.model_graph as mg
    from graph.currency_curve import curve, n_frontier
    from graph.figures import real_prm
    from graph.frontier import bindings, frontier, reps_of

    rows, n_shapes, n_models, P = frontier(mode="payable")       # (cover, name, slots, dims, binding)
    M = fot.load()                                                # the same 3,933 (dated)
    out = {"n_shapes": n_shapes, "n_models": n_models}

    # funnel: configs -> non-decoder (SKIP_TYPES) / unplaceable -> decoder LMs
    cfgs = list(mg.iter_json_configs(JSON))
    out["n_configs"] = len(cfgs)
    out["n_non_decoder"] = sum(1 for n, c in cfgs if not mg._junky(n) and isinstance(c, dict)
                               and mg.skipped(c) == "non_decoder")
    out["n_drafter"] = sum(1 for n, c in cfgs if not mg._junky(n) and isinstance(c, dict)
                           and mg.skipped(c) == "drafter")
    out["n_junk"] = sum(1 for n, _ in cfgs if mg._junky(n))
    out["mixers"] = {k: sum(1 for m in M if m["slots"]["mixer"] == k) for k in ("softmax", "hybrid", "ssm", "linear")}
    out["n_slots"] = len(mg.SLOTS)

    # the currency curve; the AR / diffusion split of the headline; the opt-in axes
    out["curve"] = curve(P)
    out["frontier_diffusion"] = sum(1 for r in rows if r[2]["gen"] != "ar")   # diffusion + block-causal (REVIEW 1)
    out["frontier_len"] = len(rows)
    for flag in ("USE_VOCAB", "USE_CONTEXT"):
        setattr(mg, flag, True)
        try:
            out[flag] = n_frontier(mg.emul_matrix(mg._dedup_points(M), mode="payable"))
        finally:
            setattr(mg, flag, False)

    # bindings (RECORD / PARETO / WALL / PARADIGM)
    kinds = {}
    for r in rows:
        k = r[4].split()[0]
        kinds[k] = kinds.get(k, 0) + 1
    out["bindings"] = kinds

    # the relation, its in-degrees (zoo_data's `by`: models in OTHER shapes that emulate it) and ties
    ok = mg.emul_matrix(P, mode="payable")
    cnt = np.array([len(p["members"]) for p in P])
    cover = (ok * cnt[:, None]).sum(0)
    o = ok.copy()
    np.fill_diagonal(o, False)
    by = (o * cnt[None, :]).sum(1)
    out["mutual_pairs"] = int((o & o.T).sum()) // 2
    idx = {n: i for i, p in enumerate(P) for n in p["members"]}
    out["in_degree"] = {k: int(by[idx[v]]) for k, v in IN_DEGREE_MODELS.items() if v in idx}
    out["in_degree_median"] = float(np.median(np.repeat(by, cnt)))
    # the toy top: the most-emulated shapes, test fixtures (tiny-random / testing repos) aside
    real = np.array([not all("test" in n.lower() for n in p["members"]) for p in P])
    top = int(by[real].max())
    out["in_degree_top"] = top
    out["in_degree_top_models"] = sorted({n.split("/")[-1] for i in np.where(real & (by >= 0.97 * top))[0]
                                          for n in P[i]["members"]})

    # the top-N cover table
    def shape_tokens(s):
        pos = {"rope": "RoPE", "partial_rope": "RoPE", "alibi": "ALiBi", "nope": "NoPE",
               "learned_abs": "learned-abs"}.get(s["pos"], s["pos"])
        ffn = "MoE" if s["ffn"].startswith("moe") else ("SwiGLU" if s["ffn"] == "glu" and s["act"] == "SiLU"
                                                         else f"{s['act']}-{s['ffn']}")
        return [pos, s["kv"], ffn] + (["hybrid"] if s["mixer"] == "hybrid" else [])
    reps = reps_of(ok, P)
    pidx = {short(P[i]["members"][0]): i for i in reps}
    out["top"] = [{"name": r[1], "members": [short(n) for n in P[pidx[r[1]]]["members"]], "cover": r[0],
                   "h": r[3]["hidden"], "L": r[3]["eff_layers"], "kv": r[3]["kv_cap"],
                   "tokens": shape_tokens(r[2])} for r in rows[:TOP_N]]

    # coverage caption: the crown's cover, and how much larger the biggest model it out-covers is
    crown = pidx[rows[0][1]]
    pr = np.array([real_prm(p) for p in P], float)
    beaten = cover < cover[crown]
    out["coverage_ratio"] = float(pr[beaten].max() / pr[crown])
    out["coverage_ratio_model"] = short(P[int(np.argmax(np.where(beaten, pr, -1)))]["members"][0])

    # the OpenRouter / Artificial Analysis table: on the frontier (not strictly dominated), else the
    # highest-cover frontier representative that emulates it
    strict = o & ~o.T
    rep_cover = {int(i): int(cover[i]) for i in reps}

    def verdict(repo):
        if repo not in idx:
            return None
        i = idx[repo]
        if not strict[i].any():
            return {"on": True}
        hosts = [j for j in reps if ok[i, j]]
        j = max(hosts, key=lambda j: rep_cover[int(j)])
        return {"on": False, "by": short(P[j]["members"][0]),
                "by_members": [short(n) for n in P[j]["members"]]}
    out["openrouter"] = [verdict(r) for _, r in OPENROUTER_TOP10]
    out["aa"] = [verdict(r) for _, r in AA_TOP10]

    # the frontier over time (frontier_over_time.analyze reads its module-level models_by_name)
    fot.models_by_name = {m["name"]: m["date"] for m in M}
    _, _, _, series = fot.analyze()
    ot = {}
    for d in OVER_TIME_DATES:
        prior = [s for s in series if s[0] <= d]
        if prior:
            t, fsz, ncls, _, _ = prior[-1]
            ot[d] = {"month": t, "frontier": int(fsz), "configs": int(ncls)}
    out["over_time"] = ot

    # the Llama-3.1 ladder (raw configs)
    cfg = dict(cfgs)
    lad = {}
    for size, repo in LLAMA_LADDER:
        c = cfg.get(repo)
        if c:
            lad[size] = {"layers": c["num_hidden_layers"], "hidden": c["hidden_size"],
                         "heads": c["num_attention_heads"], "ffn": c["intermediate_size"],
                         "kv": c["num_key_value_heads"],
                         "head_dim": c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]}
    out["llama"] = lad
    return out


def short(name):
    return name.split("/")[-1]


def elephantine_section():
    """graph.minimal_emulator: the join, its size, coverage and the loaded-and-tested split."""
    from graph.minimal_emulator import MLA_PENALTY, coverage, join, load, loaded_split, size
    models = load()
    j = join(models)
    tot, act = size(j)
    big = max(models, key=lambda m: size(m["dims"])[0])
    llama = next(m for m in models if m["name"] == "meta-llama/Llama-3.1-405B-FP8")
    carried, _ = coverage(models)
    _, loaded = loaded_split(models)
    _, loaded_remote = loaded_split(models, include_remote=True)

    def holders(k):
        """every model holding the join's record on axis k (ties included), with the join's penalty"""
        pen = lambda m: MLA_PENALTY if (k == "kv_cap" and m["slots"]["kv"] == "MLA") else 1
        return sorted({short(m["name"]) for m in models if m["dims"][k] * pen(m) == j[k]})
    ffn_rec = next(m for m in models if m["name"] == j["who"]["ffn_total"])
    return {"n_models": len(models), "hidden": j["hidden"], "layers": j["layers"], "kv": j["kv_cap"],
            "qheads": j["qheads"], "ffn_total": j["ffn_total"], "ffn_units": j["ffn_units"],
            "total": tot, "active": act, "biggest": short(big["name"]), "biggest_total": size(big["dims"])[0],
            "llama405_active": size(llama["dims"])[1], "carried": carried, "loaded": loaded,
            "loaded_with_remote": loaded_remote,
            "holders": {k: holders(k) for k in ("hidden", "eff_layers", "kv_cap", "qheads", "ffn_total")},
            "ffn_record_experts": ffn_rec["dims"]["experts"]}


def join_hosted_section():
    """The real shapes the joined host hosts per mode: graph.minimal_emulator.host_point(join(zoo), gen) with its gen
    slot set per target (the one slot host_point sets per target), checked with graph.model_graph.emul_matrix."""
    import numpy as np

    import graph.model_graph as mg
    from graph.minimal_emulator import host_point, join
    from verify.z3_relation import real_points
    M, P = real_points()
    J = join(M)
    gens = sorted({p["slots"]["gen"] for p in P})
    out = {"shapes": len(P), "models": sum(len(p["members"]) for p in P)}
    for mode in ("exact", "payable", "mixgap", "capacity"):
        hosted = np.zeros(len(P), bool)
        for g in gens:
            col = mg.emul_matrix(list(P) + [host_point(J, g)], mode=mode)[:-1, -1]
            hosted |= col & np.array([p["slots"]["gen"] == g for p in P])
        out[mode] = {"shapes": int(hosted.sum()),
                     "models": int(sum(len(p["members"]) for p, h in zip(P, hosted) if h))}
    return out


def counts_section():
    """Counts read off data files: the snapshot models with no modelling code anywhere (the code-diff sweep's
    inventory), the models with a tested recipe in every framework (recipes.json), the zoo page's models."""
    from graph import zoo_ref
    inv = json.load(open(os.path.join(ROOT, "verify/sweep/inventory.json")))["types"]
    no_code = {t: v["models"] for t, v in inv.items() if v.get("no_code") or not v.get("code")}
    rc = json.load(open(os.path.join(ROOT, "graph/data/recipes.json")))["counts"]
    return {"no_code_models": sum(no_code.values()), "no_code_types": sorted(no_code),
            "recipe_models": min(c["tested_route"] for c in rc.values()),
            "recipe_by_framework": {fw: c["tested_route"] for fw, c in rc.items()},
            "zoo_models": len(zoo_ref.zoo_models())}


def act_swap_section():
    """act.act_swap_report over act/data/act_swap_agreement.jsonl."""
    from act.act_swap_report import DEFS, TARGET_ACT, load, swaps, threshold
    rows = load()
    sw = swaps(rows)
    cell = [r for r, _ in sw if TARGET_ACT.get(r["model"]) == "GeLU" and r["host"] == "SiLU"]
    rev = [r for r, _ in sw if r["model"] == "Qwen/Qwen2.5-1.5B" and r["host"] == "GeLU"]
    real = [r["floor"] for r, _ in sw if "*" not in r["host"] and r["floor"] > 1e-4]   # below: an exact fold
    return {"n_models": len({r["model"] for r in rows}),
            "gelu_silu_floor": min(r["floor"] for r in cell),
            "gelu_silu_top1": [min(r["top1"] for r in cell), max(r["top1"] for r in cell)],
            "gelu_silu_ex32_max": max(r["ex32"] for r in cell),
            "silu_gelu_qwen_top1": rev[0]["top1"] if rev else None,
            "min_nonzero_floor": min(real),
            "bf16_first_fail": threshold(sw, DEFS["bf16"])[1]}


def act_cross_section():
    """act.activation_cross.matrix(method="minimax"): the 8 x 8 single-op floors (slow: cached)."""
    from act.activation_cross import matrix
    return matrix(method="minimax")


def mixer_gap_section():
    """attn.mixer_gap_report rows + graph.model_graph.MIXER_GAP prices."""
    from attn.mixer_gap_report import load, whole
    from graph.minimal_emulator import load as mload
    from graph.model_graph import MIXER_GAP
    rows = whole(load())
    fixed = next(r for r in rows if r["family"] == "mamba" and r["variant"] == "fixed-decay")
    widest = max(m["dims"]["qheads"] for m in mload() if m["slots"]["mixer"] == "softmax")
    return {"mamba_heads": MIXER_GAP["mamba1"]["heads"], "fixed_top1": fixed["top1"],
            "fixed_ex32": fixed["ex32"], "widest_softmax_qheads": widest}


LEAN_REL = "proofs/lean/EmulationProofs"
LEAN_DIR = os.path.join(ROOT, LEAN_REL)


def release_section():
    """The draft's <N:...> placeholders that graph_section does not already hold (release-rel): the frontier study's
    itemised floors (graph.frontier_study.floor_numbers), the headline before the code-level reading (every round-3
    key off), the act-width multiples the headline spends, the mutual pairs' differences, and the catalog-only rule
    count of the edge table."""
    import collections
    import re

    import numpy as np

    import graph.model_graph as mg
    from graph import frontier_study as fs
    from graph.currency_curve import n_frontier, points
    out = {"study": fs.floor_numbers()}
    P = points()
    ok = mg.emul_matrix(P, mode="payable")
    # every round-3 key off: the pass-5 relation on today's snapshot reading
    ks = set()
    for t in mg._R3_TABLES:
        ks |= set(t.r3_add) | set(t.r3_drop)
    ks |= set(re.findall(r'_r3\(\s*"([A-Za-z0-9_]+)"', open(mg.__file__).read()))
    saved = set(mg.R3_OFF)
    mg.R3_OFF.update(ks)
    try:
        out["headline_before_code_reading"] = n_frontier(mg.emul_matrix(points(), mode="payable"))
    finally:
        mg.R3_OFF.clear()
        mg.R3_OFF.update(saved)
    out["n_r3_keys"] = len(ks)
    # the act-width multiples k the headline spends: payable edges whose activation is reached only through a k-unit fit
    act = [p["slots"]["act"] for p in P]
    fb = [int(bool(p["dims"].get("ffn_bias"))) for p in P]
    o = ok.copy()
    np.fill_diagonal(o, False)
    kset = collections.Counter()
    memo = {}
    for i, j in zip(*np.nonzero(o)):
        key = (act[i], act[j], fb[j])
        if key not in memo:
            memo[key] = (0 if mg.slot_reaches("act", act[i], act[j]) else mg.act_width_k(act[i], act[j], bool(fb[j])))
        if memo[key] > 1:
            kset[memo[key]] += 1
    out["act_k_spent"] = {int(k): int(v) for k, v in sorted(kset.items())}
    out["k_gelu_from_silu"] = int(mg.act_width_k("GeLU", "SiLU", False))
    # the mutual pairs and what each pair differs in (slots, then dims)
    pairs = [(i, j) for i, j in zip(*np.nonzero(o & o.T)) if i < j]
    diff, coarse = collections.Counter(), collections.Counter()
    for i, j in pairs:
        sd = [s for s in P[i]["slots"] if P[i]["slots"][s] != P[j]["slots"].get(s)]
        dd = [k for k in P[i]["dims"] if P[i]["dims"][k] != P[j]["dims"].get(k)]
        diff[", ".join(sorted([f"slot {s}" for s in sd] + [f"dim {k}" for k in dd])) or "nothing (dedup key only)"] += 1
        # coarse: what the two-way edge spends (the counts / vocab / context a lever or a tie moves are not listed)
        c = [{"window": "the window (a config window both ways)", "act": "GeLU erf <-> tanh",
              "feats": "a feature switched off both ways", "pos": "the position scheme", "mtp": "MTP heads"}.get(s, s)
             for s in sd if s != "mtp" or len(sd) == 1]
        if "window" not in sd and any(k in dd for k in ("rot_frac_free", "rope_cap", "rope_sched")):
            c.append("the rotary width (rotated dims equal)")
        if "moe_sig" in dd:
            c.append("the MoE routing signature")
        coarse["; ".join(sorted(set(c))) or "dims only"] += 1
    out["mutual_pairs"] = len(pairs)
    out["mutual_pairs_detail"] = dict(diff.most_common())
    diff = coarse
    # the cover (models emulated, its own shape's included) of Mistral-Medium-3.5, as the top table counts it
    cnt = np.array([len(p["members"]) for p in P])
    ix = {n: i for i, p in enumerate(P) for n in p["members"]}
    mi = ix.get("nvidia/Mistral-Medium-3.5-128B-NVFP4")
    out["mistral_cover"] = int((ok[:, mi] * cnt).sum()) if mi is not None else None
    out["mutual_pairs_breakdown"] = dict(diff.most_common())
    # snapshot facts
    meta = json.load(open(os.path.join(ROOT, "graph/model_meta.json")))
    names = [n for p in P for n in p["members"]]
    dates = [meta[n]["created_at"][:10] for n in names if n in meta and meta[n].get("created_at")]
    out["newest_model_created"] = max(dates) if dates else None
    cfg = dict(mg.iter_json_configs(JSON))
    p3 = cfg.get("microsoft/Phi-3-small-8k-instruct") or {}
    out["phi3_dense_ctx"] = (int(p3["blocksparse_num_local_blocks"] * p3["blocksparse_block_size"])
                             if p3.get("blocksparse_num_local_blocks") and p3.get("blocksparse_block_size") else None)
    # the edge table: headline rules per verdict (catalog-only = CATALOG)
    from verify import edge_table as E
    from verify import z3_relation as Z
    M, Pz = Z.real_points()
    rows = E.edge_rows(Z.Space(Pz, extra=[Z.join_host(M)]), Pz)
    hd = [r for r in rows if "payable" in r["modes"]]
    out["edge_verdicts"] = dict(collections.Counter(r["just"].verdict for r in hd))
    out["catalog_only_rules"] = sum(1 for r in hd if r["just"].verdict == "CATALOG")
    out["headline_gaps"] = len(E.headline_gaps(rows)[0])
    # the Lean library (proofs/lean): theorem / lemma declarations, as check.sh counts them
    n = None
    if os.path.isdir(LEAN_DIR):
        n = 0
        for f in sorted(os.listdir(LEAN_DIR)):
            if f.endswith(".lean"):
                n += len(re.findall(r"^(?:theorem|lemma) [A-Za-z0-9_.']+", open(os.path.join(LEAN_DIR, f)).read(), re.M))
    out["lean_theorems"] = n
    # the spot checks: sampled headline edges built end to end that passed (verify/spot/results.jsonl)
    sp = os.path.join(ROOT, "verify/spot/results.jsonl")
    rs = [json.loads(x) for x in open(sp)] if os.path.exists(sp) else []
    out["spot_built"] = sum(1 for r in rs if r.get("status") == "pass")
    out["spot_sampled"] = len(rs)
    return out


# --- assembling the keys ------------------------------------------------------------------------------


def build(use_cache=True):
    N = Numbers()
    g = cached("graph", GRAPH_DEPS, graph_section, use_cache)
    e = cached("elephantine", ELEPH_DEPS, elephantine_section, use_cache)
    a = cached("act_swap", ["act/act_swap_report.py", "act/data/act_swap_agreement.jsonl"], act_swap_section,
               use_cache)
    x = cached("act_cross", ["act/activation_cross.py"], act_cross_section, use_cache)
    m = cached("mixer_gap", ["attn/mixer_gap_report.py", "attn/data/mixer_gap_agreement.jsonl"] + GRAPH_DEPS,
               mixer_gap_section, use_cache)

    # funnel ---------------------------------------------------------------------------------------
    c = "funnel"
    src_f = "graph.frontier.frontier / graph.figures.fig_funnel"
    N.put("funnel.configs", g["n_configs"], "graph.model_graph.iter_json_configs(graph/hub_10k.json)",
          "4384 configs", c, [R(r"<N>\\text\{ configs\}"), R(r"The funnel: <N> configs")])
    N.put("funnel.configs_k", g["n_configs"] / 1000, "graph.model_graph.iter_json_configs (thousands)",
          "~4.4k deduped", c, [R(r"<N>\\text\{k\}\$ deduped")], kind="approx")
    N.put("funnel.models", g["n_models"], src_f, "N decoder LMs", c, [
        R(r"collapses <N> models"), R(r"<N> decoder LMs once duplicate"), R(r"<N>\\text\{ models\}"),
        R(r"leaving <N>\$ decoder LMs"), R(r"<N> classified decoder LMs"), R(r"carrying [\d,]+ of <N> models"),
        R(r"of the <N>\$ models; the other"), R(r"Across <N> distinct decoder-LM")])
    N.put("funnel.shapes", g["n_shapes"], src_f, "841 shapes", c, [
        R(r"collapsing to <N> distinct"), R(r"<N>\\text\{ shapes\}"), R(r"collapse to <N>\$ distinct arch"),
        R(r"<N> arch\+size shapes")])
    N.put("funnel.non_decoder", g["n_non_decoder"], "graph.model_graph.classify (model_type in SKIP_TYPES)",
          "drops 18 non-decoder types", c, [R(r"drops <N>\$ non-decoder")])
    N.put("funnel.drafter", g.get("n_drafter", 0), "graph.model_graph.skipped (drafter / assistant config keys)",
          "drops N speculative drafters", c, [R(r"drops <N>\$ (?:speculative )?drafters")])
    N.put("funnel.unplaced", g["n_configs"] - g["n_junk"] - g["n_non_decoder"] - g.get("n_drafter", 0) - g["n_models"],
          "graph.model_graph.classify / dims_of (no placement or no dims)", "433 that can't be placed", c,
          [R(r"and <N>\$ that can't be placed")])
    N.put("funnel.slots", g["n_slots"], "len(graph.model_graph.SLOTS)", "assigns 15 slots", c,
          [R(r"assigns <N>\$ slots")])
    c = "mixer split"
    for k, pat in (("softmax", r"\(<N>\$ softmax"), ("hybrid", r"<N>\$ hybrid, "), ("ssm", r"<N>\$ SSM, "),
                   ("linear", r"<N>\$ linear\)")):
        N.put(f"mixers.{k}", g["mixers"][k], "graph.model_graph.classify (slot mixer)", f"N {k}", c, [R(pat)])

    # the currency curve / headline -----------------------------------------------------------------
    c = "currency curve"
    cv = g["curve"]
    src_c = "graph.currency_curve.curve"
    nd = g["frontier_diffusion"]
    N.put("frontier.headline", cv["payable"], src_c + " ['payable'] (= graph.frontier.frontier rows)",
          "the headline frontier, 106", c, [
              R(r"no more than <N> of them"), R(r"frontier settles at <N>"), R(r"\\text\{frontier\}=(\d+)"),
              R(r"leaves <N>\$\*\* maximal shapes"), R(r"the frontier grows from <N>\$ to"),
              R(r"→ <N> frontier emulators"), R(r"any of the <N>\$ frontier specs"),
              R(r"hence the <N> element frontier"), T(CURRENCY_TABLE, {"col": 0, "re": "regimes / limits"}, 1)])
    N.put("frontier.ar", cv["payable"] - nd, "graph.frontier.frontier (gen == ar)", "103 AR", c, [
        R(r"\((\d+)\\text\{ AR\}"), R(r"\(<N>\$ AR \$\+"), R(r"maximal shapes, <N>\$ autoregressive")])
    N.put("frontier.diffusion", nd, "graph.frontier.frontier (gen != ar: diffusion and block-causal)", "3 diffusion", c, [
        R(r"\+(\d+)\\text\{ diffusion\}"), R(r"AR \$\+(\d+)\$ diffusion"), R(r"autoregressive plus <N>\$ diffusion"),
        R(r"so the <N>\$ diffusion models")])
    N.put("currency.exact", cv["exact"], src_c + " ['exact']", "exact only: 126", c, [
        T(CURRENCY_TABLE, {"col": 0, "re": "exact only"}, 1), R(r"grows it to <N>"),
        R(r"demand every edge exact and it's <N>")])
    N.put("currency.mixgap", cv["mixgap"], src_c + " ['mixgap']", "+ the never-exact carries: 85", c, [
        T(CURRENCY_TABLE, {"col": 0, "re": "never-exact carries"}, 1), R(r"frontier shrinks to <N>"),
        R(r"CLA carries and it's <N>")])
    N.put("currency.capacity", cv["capacity"], src_c + " ['capacity']", "+ fold the type walls: 40", c, [
        T(CURRENCY_TABLE, {"col": 0, "re": "fold the type walls"}, 1), R(r"it falls to <N>\$, the pure capacity"),
        R(r"Treat every wall as free and it falls to <N>")])
    N.put("currency.elephantine", cv["elephantine"], src_c + " ['elephantine']", "+ a bigger model: 1", c, [
        T(CURRENCY_TABLE, {"col": 0, "re": "a bigger model"}, 1), R(r"closes the last step to <N>")])
    c = "axes"
    N.put("axes.vocab", g["USE_VOCAB"], "graph.currency_curve.n_frontier(emul_matrix, USE_VOCAB=True)",
          "grows from 106 to 144", c, [R(r"grows from \$\d+\$ to <N>\$ as wide-vocab")])
    N.put("axes.context", g["USE_CONTEXT"], "graph.currency_curve.n_frontier(emul_matrix, USE_CONTEXT=True)",
          "grows from 106 to 132", c, [R(r"grows from \$\d+\$ to <N>\$ as short-context")])
    c = "frontier structure"
    b = g["bindings"]
    N.put("frontier.records_pareto", b.get("RECORD", 0) + b.get("PARETO", 0),
          "graph.frontier.bindings (RECORD + PARETO)", "The 41 Pareto records", c,
          [R(r"The <N>\$ Pareto records")], detail=b)
    N.put("frontier.mutual_pairs", g["mutual_pairs"], "graph.model_graph.emul_matrix (ok & ok.T, off-diagonal)",
          "25 pairs of shapes emulate each other both ways", c, [R(r"isn't quite a DAG: <N> pairs")])

    # the top-N cover table -----------------------------------------------------------------------
    c = "top cover table"
    for i, r in enumerate(g["top"]):
        src = f"graph.frontier.frontier (rank {i + 1} by cover)"
        anchors = [T(TOP_TABLE, i, 0)]
        if i == 0:
            anchors.append(R(r"The crown froze: ([\w.\-]+) the top emulator"))
        N.put(f"top.{i + 1}.name", r["name"], src, f"top-{TOP_N} row {i + 1}: model", c, anchors, kind="name",
              aliases=r["members"])
        anchors = [T(TOP_TABLE, i, 2)]
        if i == 0:
            anchors.append(R(r"Qwen-72B emulates ~<N>"))
        N.put(f"top.{i + 1}.cover", r["cover"], src, f"top-{TOP_N} row {i + 1}: emulates", c, anchors)
        for ax, pat in (("h", r"h(\d+)"), ("L", r"L(\d+)"), ("kv", r"\\mathrm\{kv\}(\d+)")):
            anchors = [T(TOP_TABLE, i, 1, pat)]
            if i == 0 and ax == "kv":
                anchors.append(R(r"full-MHA cache \(\$\\mathrm\{kv\}(\d+)"))
            N.put(f"top.{i + 1}.{ax}", r[ax], src, f"top-{TOP_N} row {i + 1}: {ax}", c, anchors,
                  optional=(ax == "kv" and i > 0))
        N.put(f"top.{i + 1}.shape", r["tokens"], src + " slots", f"top-{TOP_N} row {i + 1}: shape", c,
              [T(TOP_TABLE, i, 1, r"^(.*?),")], kind="tokens")
    N.put("coverage.ratio", g["coverage_ratio"], "graph.figures.real_prm over the points the crown out-covers",
          "models up to ~38× larger emulate far fewer", "coverage caption",
          [R(r"models up to ~<N>× larger")], kind="approx", detail=g["coverage_ratio_model"])

    # in-degrees ---------------------------------------------------------------------------------------
    c = "in-degree"
    src = "graph.model_graph.zoo_data 'by' (models in other shapes that emulate it)"
    N.put("indeg.toy_top", g["in_degree_top"], src + ", max", "subsets of ~2,550 models each", c,
          [R(r"subsets of ~<N>\$ models each")], kind="approx")
    N.put("indeg.toy_top_models", g["in_degree_top_models"][0], src + ", within 3% of the max",
          "tiny Llama-shaped checkpoints like Swedish0.1M and SimpleStories-1.25M", c,
          [R(r"checkpoints like `([^`]+)`"), R(r"and `([^`]+)` are subsets of ~")], kind="name",
          aliases=g["in_degree_top_models"])
    for k, pat in (("pythia-14m", r"`pythia-14m` is a subset of <N>"),
                   ("mamba-130m", r"are subsets of only <N>"),
                   ("rwkv-4-169m", r"are subsets of only \$\d+\$ and <N>")):
        N.put(f"indeg.{k}", g["in_degree"][k], src, f"{k} in-degree", c, [R(pat)])
    N.put("indeg.median", g["in_degree_median"], src + ", median over models", "the median model is emulable by 167",
          c, [R(r"median model is emulable by <N>")], kind="approx")

    # the OpenRouter / AA table -------------------------------------------------------------------------
    c = "openrouter / AA"
    for tag, lst, res, col in (("openrouter", OPENROUTER_TOP10, g["openrouter"], 0), ("aa", AA_TOP10, g["aa"], 2)):
        on = 0
        for i, ((name, repo), v) in enumerate(zip(lst, res)):
            src = f"graph.model_graph.emul_matrix: {repo} strictly dominated? (input list {tag.upper()}_TOP10)"
            N.put(f"{tag}.{i + 1}.name", name, f"graph.post_numbers.{tag.upper()}_TOP10 (input)",
                  f"{tag} row {i + 1}", c, [T(OR_TABLE, i, col)], kind="name", aliases=[name])
            if v is None:
                continue
            on += v["on"]
            N.put(f"{tag}.{i + 1}.on_frontier", v["on"], src, f"{name}: on frontier?", c,
                  [T(OR_TABLE, i, col + 1, r"^(✅|❌)")], kind="bool")
            if not v["on"]:
                N.put(f"{tag}.{i + 1}.emulator", v["by"], src + "; highest-cover frontier host",
                      f"{name}: ❌ → host", c, [T(OR_TABLE, i, col + 1, r"→\s*(.+?)\s*$")], kind="name",
                      aliases=v["by_members"])
        N.put(f"{tag}.on_frontier", on, "count of the above", f"N of the top-ten by "
              f"{'usage' if tag == 'openrouter' else 'intelligence'} sit on the frontier", c,
              [R(r"<N>\$ of the top-ten by usage") if tag == "openrouter" else R(r"and <N>\$ by intelligence")])

    # the frontier over time ---------------------------------------------------------------------------
    c = "over time"
    for d, v in g["over_time"].items():
        src = f"graph.frontier_over_time.analyze (month {v['month']})"
        N.put(f"time.{d}.configs", v["configs"], src, f"{d}: unique configs", c, [T(TIME_TABLE, rowname(d), 1)])
        N.put(f"time.{d}.frontier", v["frontier"], src, f"{d}: maximal set", c, [T(TIME_TABLE, rowname(d), 2)])

    # elephantine -----------------------------------------------------------------------------------
    c = "elephantine"
    src = "graph.minimal_emulator.join"
    hold = e["holders"]
    N.put("eleph.hidden", e["hidden"], src, "h=16384", c, [R(r"with \$h\{=\}(\d+)"), R(r"its own <N>\$ hidden width")])
    N.put("eleph.layers", e["layers"], src, "L=160", c, [R(r"h\{=\}\d+\$, \$L\{=\}(\d+)")])
    N.put("eleph.kv", e["kv"], src, "kv=14336", c, [R(r"\\mathrm\{kv\}\{=\}(\d+)"), R(r"bloom's \(<N>\$, full MHA")])
    N.put("eleph.qheads", e["qheads"], src, "128 query heads", c,
          [R(r"<N>\$ query heads, and"), R(r"most query heads \(<N>\$\)")])
    N.put("eleph.ffn_per_layer_M", e["ffn_total"] / 1e6, src + " ffn_total", "~2.76M expert units per layer", c,
          [R(r"<N>\\text\{M\}\$ expert units per layer")], kind="approx")
    N.put("eleph.ffn_units_M", e["ffn_units"] / 1e6, src + " ffn_units", "441M total", c,
          [R(r"\(<N>\\text\{M\}\$ total\)")], kind="approx")
    N.put("eleph.total_T", e["total"] / 1e12, "graph.minimal_emulator.size(join)", "~22T total", c,
          [R(r"~(\d+(?:\.\d+)?)T total"), R(r"one ~(\d+(?:\.\d+)?)T spec"), R(r"training a <N>\$-trillion")],
          kind="approx")
    N.put("eleph.active_T", e["active"] / 1e12, "graph.minimal_emulator.size(join)", "~3.8T active", c,
          [R(r"~(\d+(?:\.\d+)?)T active")], kind="approx")
    N.put("eleph.x_biggest", e["total"] / e["biggest_total"], "graph.minimal_emulator.size: join / biggest",
          "about 4x the biggest model", c, [R(r"about <N>\\times\$ the biggest")], kind="approx",
          detail=e["biggest"])
    N.put("eleph.x_llama405_active", e["active"] / e["llama405_active"],
          "graph.minimal_emulator.size: join active / Llama-3.1-405B active",
          "roughly 9x the per-token compute of Llama-3.1-405B", c,
          [R(r"roughly <N>\\times\$ the per-token")], kind="approx")
    N.put("eleph.carried", e["carried"], "graph.minimal_emulator.coverage", "carrying N of M models", c,
          [R(r"carrying (\d[\d,]*) of [\d,]+ models")])
    N.put("eleph.loaded_tested", e["loaded"], "graph.minimal_emulator.loaded_split", "N loaded and tested", c,
          [R(r"That holds for <N>\$ of the")])
    N.put("eleph.priced_only", e["carried"] - e["loaded"], "graph.minimal_emulator.loaded_split",
          "the other N (priced only)", c, [R(r"the other <N>\$, mostly")])
    for k, ax, pat in (("eleph.kv_who", "kv_cap", r"widest cache is ([\w.\-]+)'s"),
                       ("eleph.depth_who", "eff_layers", r"the most depth ([\w.\-]+)'s"),
                       ("eleph.ffn_who", "ffn_total", r"fattest FFN ([\w.\-]+)'s"),
                       ("eleph.qheads_who", "qheads", r"query heads \(\$\d+\$\) ([\w.\-]+)'s")):
        N.put(k, hold[ax][0], src + f" record holders ({ax})", f"the {ax} record holder", c, [R(pat)],
              kind="name", aliases=hold[ax])
    N.put("eleph.ffn_experts", e["ffn_record_experts"], "graph.model_graph.dims_of (experts) of the FFN record",
          "Kimi-K3.1-Pro's 896 experts", c, [R(r"'s <N>\$ experts")])

    # activation calibration / cross matrix ----------------------------------------------------------
    c = "activation calibration"
    src = "act.act_swap_report (act/data/act_swap_agreement.jsonl)"
    N.put("act.n_checkpoints", a["n_models"], src, "16 real checkpoints", c, [R(r"every FFN of <N> real checkpoints")])
    N.put("act.gelu_from_silu_floor", a["gelu_silu_floor"], src, "GeLU from SiLU at 0.012", c,
          [R(r"even GeLU from SiLU at <N>\$")], kind="approx")
    N.put("act.gelu_from_silu_top1_min", 100 * a["gelu_silu_top1"][0], src, "top-1 96-98%", c,
          [R(r"top-1 <N>\$–")], kind="approx")
    N.put("act.gelu_from_silu_top1_max", 100 * a["gelu_silu_top1"][1], src, "top-1 96-98%", c,
          [R(r"top-1 \$[\d.]+\$–\$(\d+)\\%")], kind="approx")
    N.put("act.gelu_from_silu_ex32_max", 100 * a["gelu_silu_ex32_max"], src, "at most 54% identical", c,
          [R(r"at most <N>\\%\$ of 32-token")], kind="approx")
    if a["silu_gelu_qwen_top1"] is not None:
        N.put("act.silu_from_gelu_qwen_top1", 100 * a["silu_gelu_qwen_top1"], src, "only 26% top-1 on Qwen2.5-1.5B",
              c, [R(r"manages only <N>\\%\$ top-1")], kind="approx")
    N.put("act.min_nonzero_floor", a["min_nonzero_floor"], src + " (smallest nonzero real-host floor)",
          "no measured cell between 0 and 0.009", c, [R(r"between \$0\$ and <N>\$")], kind="approx")
    N.put("act.bf16_first_fail", a["bf16_first_fail"], src + " threshold(DEFS['bf16'])",
          "even no worse than bf16 fails at 0.010", c, [R(r"fails at <N>\$\)")], kind="approx")
    act_cross_numbers(N, x)

    # the mixer gap ------------------------------------------------------------------------------------
    c = "mixer gap"
    N.put("mixer.mamba_heads_k", m["mamba_heads"] / 1000, "graph.model_graph.MIXER_GAP['mamba1'].heads",
          "26k heads for mamba-130m", c, [R(r"\(<N>\$k for mamba-130m")], kind="approx")
    N.put("mixer.widest_host_heads", m["widest_softmax_qheads"], "graph.minimal_emulator.load (max softmax qheads)",
          "against 128 for the widest real host", c, [R(r"against <N>\$ for the widest")])
    N.put("mixer.fixed_decay_top1", 100 * m["fixed_top1"], "attn.mixer_gap_report (mamba fixed-decay)",
          "one fixed-decay layer: 85% top-1", c, [R(r"fixed-decay layer: <N>\\%\$ top-1")], kind="approx")
    N.put("mixer.fixed_decay_ex32", 100 * m["fixed_ex32"], "attn.mixer_gap_report (mamba fixed-decay)",
          "4% of continuations identical", c, [R(r"<N>\\%\$ of continuations identical")], kind="approx")

    # the draft's <N:key> placeholders (release-rel) -------------------------------------------------------------
    # Each key is the placeholder's own name, with no anchor: graph/check_post.py lists every placeholder of the draft
    # with its value (PLACEHOLDERS). value None: not computable from this tree (the source says where it lives).
    rl = cached("release", GRAPH_DEPS + ["graph/frontier_study.py", "verify/edge_table.py", "verify/z3_relation.py",
                                         "verify/spot/results.jsonl"], release_section, use_cache)
    st = rl["study"]
    c = "placeholders"

    def ph(key, value, source, phrase, kind="int", **extra):
        N.put(key, value, source, phrase, c, [], kind=kind if value is not None else "text", placeholder=True, **extra)
    ph("headline", cv["payable"], src_c + " ['payable']", "the headline frontier")
    ph("headline_ar", cv["payable"] - nd, "graph.frontier.frontier (gen == ar)", "AR part of the headline")
    ph("headline_nonar", nd, "graph.frontier.frontier (gen != ar)", "non-AR part (diffusion, block-causal, prefix-LM)")
    ph("models", g["n_models"], src_f, "decoder LMs in the snapshot")
    ph("shapes", g["n_shapes"], src_f, "distinct shapes")
    ph("drafters", g.get("n_drafter", 0), "graph.model_graph.skipped (drafter)", "speculative drafters dropped")
    ph("mutual_pairs", g["mutual_pairs"], "graph.model_graph.emul_matrix (ok & ok.T, off-diagonal)",
       "pairs of shapes that emulate each other both ways")
    ph("mutual_pairs_breakdown", rl["mutual_pairs_breakdown"], "graph.post_numbers.release_section (the keys each "
       "mutual pair's two points differ in)", "what the mutual pairs differ in", kind="text")
    ph("or_on_frontier", sum(v["on"] for v in g["openrouter"] if v), "OPENROUTER_TOP10 on the frontier",
       "OpenRouter top-10 on the frontier")
    ph("aa_on_frontier", sum(v["on"] for v in g["aa"] if v), "AA_TOP10 on the frontier",
       "Artificial Analysis top-10 on the frontier")
    ph("top_cover", g["top"][0]["cover"], "graph.frontier.frontier (rank 1 cover)", "the top emulator's cover")
    ph("top_size_ratio", g["coverage_ratio"], "graph.figures.real_prm over the points the crown out-covers",
       "size ratio for the coverage caption", kind="approx", detail=g["coverage_ratio_model"])
    ph("mamba130m_in", g["in_degree"].get("mamba-130m"),
       "graph.model_graph.zoo_data 'by' (models in other shapes that emulate it)", "how many emulate mamba-130m")
    ph("mistral_medium_35_cover", rl["mistral_cover"],
       "graph.model_graph.emul_matrix: models nvidia/Mistral-Medium-3.5-128B-NVFP4 emulates (its cover)",
       "how many models Mistral-Medium-3.5 emulates")
    ks = sorted(rl["act_k_spent"], key=int)
    ph("act_width_k_range", f"x{ks[0]}-{ks[-1]}" if ks else None, "graph.post_numbers.release_section: payable edges "
       "whose activation is reached only by a k-unit fit (act_width_k)", "range of k the headline spends", kind="text",
       detail=rl["act_k_spent"])
    ph("k_gelu_from_silu", rl["k_gelu_from_silu"], "graph.model_graph.act_width_k('GeLU', 'SiLU') (cost_act_k_measured)",
       "FFN width multiple at which GeLU <- SiLU pays")
    ph("phi3_dense_ctx", rl["phi3_dense_ctx"], "Phi-3-small config blocksparse_num_local_blocks x blocksparse_block_size",
       "context up to which Phi-3-small's block-sparse attention is dense")
    ph("headline_before_code_reading", rl["headline_before_code_reading"],
       f"graph.currency_curve.n_frontier with every round-3 key off ({rl['n_r3_keys']} keys in R3_OFF)",
       "headline before the code-level reading")
    for k, phrase in (("floor_types_free", "every type wall waived; construction costs and the paradigm kept"),
                      ("floor_types_paradigm_free", "every type wall and the paradigm waived"),
                      ("max_single_kind_freed", "the most frontier classes one wall kind frees"),
                      ("unique_feature_frontier", "frontier classes with a feature no other shape has"),
                      ("headline_relaxed_bias_norm", "bias + norm + qk / v-norm + output gate + sink / softcap waived")):
        ph(k, st[k], f"graph.frontier_study.floor_numbers ['{k}']", phrase,
           **({"detail": st["max_single_kind"]} if k == "max_single_kind_freed" else {}))
    ph("catalog_only_rules", rl["catalog_only_rules"], "verify.edge_table: headline rules with verdict CATALOG",
       "simple switch-offs justified by a verified catalog instance only", detail=rl["edge_verdicts"])
    ph("lean_theorems", rl["lean_theorems"], f"theorem / lemma declarations in {LEAN_REL} (check.sh's count)",
       "Lean theorems and lemmas")
    ph("spot_built", rl["spot_built"], f"verify/spot/results.jsonl: status pass of {rl['spot_sampled']} sampled",
       "spot-checked headline edges built end to end")
    ph("snapshot_date", rl["newest_model_created"], "graph/model_meta.json: the newest snapshot model's created_at "
       "(the fetch date itself is not recorded)", "the snapshot date", kind="text")
    ph("transitivity_note", f"Reflexive in every mode; 0 transitivity violations on the real shapes in exact, payable, "
       f"mixgap and capacity; exact ⊆ payable ⊆ mixgap ⊆ capacity ({g['n_shapes']} shapes).",
       "python -m verify.r3.relcheck", "the final transitivity status sentence", kind="text")
    k = cached("counts", ["verify/sweep/inventory.json", "graph/data/recipes.json", "graph/zoo_ref.py",
                          "graph/model_graph.py", "graph/hub_10k.json"], counts_section, use_cache)
    ph("no_code_models", k["no_code_models"], "verify/sweep/inventory.json: models of the types with no modelling "
       "code anywhere (no_code)", "models with no code anywhere", detail=k["no_code_types"])
    ph("recipe_models", k["recipe_models"], "graph/data/recipes.json counts[framework].tested_route (min over the "
       "frameworks)", "models with a tested recipe in each framework", detail=k["recipe_by_framework"])
    ph("zoo_models", k["zoo_models"], "graph.zoo_ref.zoo_models (the models dist/zoo.html shows a card for)",
       "models on the zoo page")

    # release-only numbers (no draft anchor): the joined host per mode, loaded-and-tested with / without remote code
    c = "release"
    jh = cached("join_hosted", GRAPH_DEPS + ["elephantine/*.py", "verify/z3_relation.py"], join_hosted_section,
                use_cache)
    for mode in ("exact", "payable", "mixgap", "capacity"):
        N.put(f"join_hosted.{mode}", jh[mode]["shapes"], "graph.post_numbers.join_hosted_section (host_point(join(zoo), "
              "gen per target), emul_matrix", f"real shapes the joined host hosts ({mode}) of {jh['shapes']}", c, [],
              optional=True, models=jh[mode]["models"], of=jh["shapes"])
    N.put("eleph.loaded_tested_no_remote", e["loaded"], "graph.minimal_emulator.loaded_split (include_remote=False)",
          "loaded and tested, no remote-code test credited", c, [], optional=True)
    N.put("eleph.loaded_tested_with_remote", e["loaded_with_remote"],
          "graph.minimal_emulator.loaded_split (include_remote=True)", "loaded and tested, remote-code tests credited",
          c, [], optional=True)

    # the Llama-3.1 ladder ------------------------------------------------------------------------------
    c = "llama ladder"
    lines = {"layers": "layers", "hidden": "hidden", "heads": "query heads", "ffn": "FFN size", "kv": "KV heads",
             "head_dim": "head dim"}
    sizes = [s for s, _ in LLAMA_LADDER]
    if all(s in g["llama"] for s in sizes):
        for ax, word in lines.items():
            for pos, s in enumerate(sizes):
                cells = [r"[\d]+" if p != pos else r"(\d+)" for p in range(3)]
                pat = r"^- " + r"\s*\$\\rightarrow\$\s*".join(cells) + r"\s+" + word
                N.put(f"llama.{s}.{ax}", g["llama"][s][ax], f"graph/hub_10k.json config of {dict(LLAMA_LADDER)[s]}",
                      f"Llama-3.1 ladder: {word}", c, [R(pat)])
    return N.d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-cache", action="store_true", help="recompute every section")
    ap.add_argument("--release-out", default=DEFAULT_RELEASE_OUT,
                    help="key -> value of every placeholder and release key (default dist/release_numbers.json)")
    args = ap.parse_args()
    nums = build(use_cache=not args.no_cache)
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "snapshot": JSON, "n_keys": len(nums)}
    try:
        import subprocess
        meta["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip()
    except OSError:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "numbers": nums}, f, indent=1, ensure_ascii=False)
    rel = {k: v["value"] for k, v in nums.items() if v.get("placeholder") or v["category"] == "release"}
    os.makedirs(os.path.dirname(os.path.abspath(args.release_out)), exist_ok=True)
    with open(args.release_out, "w") as f:
        json.dump({"meta": meta, "numbers": rel}, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.release_out}: {len(rel)} keys ({sum(v is None for v in rel.values())} null)")
    cats = {}
    for v in nums.values():
        cats[v["category"]] = cats.get(v["category"], 0) + 1
    print(f"wrote {args.out}: {len(nums)} keys")
    for k, v in cats.items():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
