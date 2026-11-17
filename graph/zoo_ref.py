"""The zoo page as an architecture reference: graph/templates/zoo.html filled with zoo_data() (the emulation
relation, graph/model_graph.py) AND ref_data() below (what each shape IS, read the way the classifier reads it).

Every value on a spec card comes from one of:
  the snapshot          graph/hub_10k.json (the configs), graph/model_meta.json (Hub upload date, safetensors count)
  the classifier        graph/model_graph.py classify / dims_of / _feats and the per-type code tables it keeps
                        (NORM_BY_CODE, QKNORM_VARIANT_BY_TYPE, POST_NORM_TYPES, ...): a field whose value one of
                        those tables set is marked "from code" on the page
  the code-diff sweep   verify/sweep/inventory.json (model type -> modelling files, modular parent) and
                        verify/sweep/findings.json (where the code and the classifier disagree)
  the metrics pass      graph/data/arch_metrics.json, graph/data/frameworks.json (graph/arch_metrics_schema.md)
  the recipes pass      graph/data/recipes.json (recipes/RECIPES_SCHEMA.md)
The last two are optional: a missing file leaves its sections off the page. Nothing is typed in by hand; a field
the sources do not give is left blank.

Regenerate:  python -m graph.zoo_ref --out dist/zoo.html
"""
import argparse
import base64
import json
import os

from graph import model_graph as mg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
META = os.path.join(HERE, "model_meta.json")
INVENTORY = os.path.join(ROOT, "verify", "sweep", "inventory.json")
FINDINGS = os.path.join(ROOT, "verify", "sweep", "findings.json")
ARCH_METRICS = os.path.join(HERE, "data", "arch_metrics.json")
FRAMEWORKS = os.path.join(HERE, "data", "frameworks.json")
RECIPES = os.path.join(HERE, "data", "recipes.json")
EXTENSION = os.path.join(HERE, "data", "hub_extension.json")   # graph/snapshot_extend.py (a candidate)
REGISTRIES = os.path.join(HERE, "data", "framework_registries.json")
# the any-HF-id lookup: graph/arch_metrics.py ported to JS, and the tables it reads (python -m graph.arch_metrics_js)
AM_JS = os.path.join(HERE, "templates", "arch_metrics.js")
AM_TABLES = os.path.join(HERE, "data", "arch_metrics_js_tables.json")
# the detailed breakdown (graph/arch_breakdown.py) ported to JS, the llama.cpp quant tables it reads, and its per-repo
# statuses; the page ships each shape's plan gzip-compressed and computes the detail in the browser on demand
BD_JS = os.path.join(HERE, "templates", "arch_breakdown.js")
GQ_JS = os.path.join(HERE, "templates", "gguf_quant.js")
BREAKDOWN = os.path.join(HERE, "data", "arch_breakdown.json.gz")
TF_URL = "https://github.com/huggingface/transformers/blob/main/src/transformers/"
SIDE_PAIRS = "zoo_pairs.json"          # the lazy-loaded blocker file, written next to the page


def _load_json(path):
    """A JSON file, or None when it is missing or unreadable (the optional passes)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --- one config -> its spec ---------------------------------------------------------------------------------

def _first(d, *keys):
    for k in keys:
        v = d.get(k)
        if v not in (None, [], ""):
            return v
    return None


def _scalar(v):
    """A JSON-safe scalar: ints stay ints, floats keep their value, lists become their max (per-layer lists)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (list, tuple)):
        nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return max(nums) if nums else None
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    return None


def _rope(d):
    """RoPE facts of an LM config: base theta (global layers, and a second one for local layers when the config
    keeps one), the rotated fraction, the scaling type and factor, the pre-scaling context."""
    out = {}
    rp = d.get("rope_parameters") if isinstance(d.get("rope_parameters"), dict) else {}
    per = mg._rope_layer_types(rp)
    rs = d.get("rope_scaling") if isinstance(d.get("rope_scaling"), dict) else {}
    glob = per.get("full_attention") or per.get("main") or (rp if not per else {}) or {}
    loc = per.get("sliding_attention") or {}
    theta = _first(d, "rope_theta") or (glob or {}).get("rope_theta") or (d.get("attn_config") or {}).get("rope_theta")
    if theta:
        out["theta"] = _scalar(theta)
    lt = _first(d, "rope_local_base_freq", "swa_rope_theta", "rope_local_theta") or (loc or {}).get("rope_theta")
    if lt and _scalar(lt) != out.get("theta"):
        out["theta_local"] = _scalar(lt)
    f = mg._rotary_factor(d)
    if f is not None:
        out["rot"] = f
    if d.get("rotary_dim"):
        out["rot_dim"] = _scalar(d["rotary_dim"])
    src = rs or glob or {}
    typ = src.get("rope_type") or src.get("type")
    if typ and typ != "default":
        out["scaling"] = str(typ)
        if _scalar(src.get("factor")):
            out["factor"] = _scalar(src.get("factor"))
        omp = src.get("original_max_position_embeddings") or d.get("original_max_position_embeddings")
        if _scalar(omp):
            out["orig_ctx"] = _scalar(omp)
    return out


# layer kinds of the schedule strip: G full-window attention, S sliding-window attention, R recurrent mixer,
# F an FFN-only entry (Nemotron-H lists mixer and FFN separately), E a MoE-only entry, N no attention (a no-op
# or linear block), ? unknown
_KIND_CHAR = {"full": "G", "local": "S", "rec": "R", "noattn": "N", None: "?"}


def _schedule(d, n, kc):
    """The per-layer mixer schedule as a string, read from the config's per-layer list or index keys (the same
    keys model_graph._attn_counts reads, mapped by its _kind), or None when the order cannot be read. The
    counts must match _attn_counts' per-kind counts kc, else None (the page then shows the counts only)."""
    mt = str(d.get("model_type") or "").lower()
    g = lambda k: mg._num(d.get(k))
    kinds = None
    pat = d.get("hybrid_override_pattern")
    if isinstance(pat, str) and pat:
        kinds = ["E" if c == "E" else "F" if c == "-" else _KIND_CHAR.get(mg._kind(c), "?") for c in pat]
    bc = d.get("block_configs")
    if kinds is None and mt == "nemotron-nas" and isinstance(bc, list) and bc:
        kinds = []
        for b in bc:
            a, f = (b.get("attention") or {}), (b.get("ffn") or {})
            if a.get("no_op") and f.get("no_op"):
                continue
            kinds.append("N" if (a.get("no_op") or a.get("replace_with_linear"))
                         else ("S" if a.get("window_length") else "G"))
    for key in (() if kinds else ("layers_block_type", "layer_types", "layer_type")):
        v = d.get(key)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            kinds = []
            for x in v:
                k = mg._kind(x)
                kinds.append("E" if k == "ffn" and str(x).lower() in ("e", "moe") else
                             "F" if k == "ffn" else _KIND_CHAR.get(k, "?"))
            break
    if kinds is None:
        for key, soft in (("attn_type_list", 1), ("linear_attention_pattern", 0)):
            v = d.get(key)
            if isinstance(v, list) and v:
                kinds = ["G" if x == soft else "R" for x in v]
        hlp = d.get("hybrid_layer_pattern")
        if kinds is None and isinstance(hlp, list) and hlp and all(x in (0, 1) for x in hlp):
            kinds = ["G" if x == 0 else "S" for x in hlp]
    if kinds is None and n:
        sw_on = bool(d.get("sliding_window") or (d.get("attn_config") or {}).get("sliding_window")) \
            and d.get("use_sliding_window", True)
        idx = None
        for key in ("full_attn_idxs", "attn_layer_indices", "full_attention_layers"):
            if isinstance(d.get(key), list) and all(isinstance(i, int) for i in d[key]):
                idx = set(d[key])
        lac = d.get("linear_attn_config")
        if idx is None and isinstance(lac, dict) and isinstance(lac.get("full_attn_layers"), list):
            idx = {i - 1 for i in lac["full_attn_layers"] if isinstance(i, int)}   # Kimi-Linear: 1-based
        if idx is not None:
            kinds = ["G" if i in idx else "R" for i in range(n)]
        elif g("attn_layer_period"):
            per, off = g("attn_layer_period"), g("attn_layer_offset")
            kinds = ["G" if i % per == off else "R" for i in range(n)]
        elif g("full_attention_interval") and mt not in ("plamo2", "plamo2vl"):
            k = g("full_attention_interval")             # the config classes' default layer_types: every k-th full
            kinds = ["G" if (i + 1) % k == 0 else "R" for i in range(n)]
        elif mt in ("plamo2", "plamo2vl") and g("mamba_step"):
            st, fi = g("mamba_step"), set(d.get("full_attention_idx") or [])
            kinds = [("G" if i in fi else "S") if i % st == st // 2 else "R" for i in range(n)]
        elif mt in ("falcon_h1", "hymba"):
            gi = set(d.get("global_attn_idx") or []) if mt == "hymba" else set()
            kinds = [("PG" if (not sw_on or i in gi) else "PS") for i in range(n)]   # attention + SSM in each layer
        elif mt in mg.HYBRID_TYPES or mt == "phi4flash":
            kinds = None
        elif not sw_on:
            kinds = ["G"] * n
        elif mt == "gemma2":
            kinds = ["S" if i % 2 == 0 else "G" for i in range(n)]
        elif g("sliding_window_pattern") or mt in ("gemma3", "gemma3_text"):
            p = g("sliding_window_pattern") or 6
            kinds = ["G" if (i + 1) % p == 0 else "S" for i in range(n)]
        elif mt in ("qwen2", "qwen3", "qwen2_moe", "qwen3_moe") and d.get("max_window_layers") is not None:
            kinds = ["S" if i >= g("max_window_layers") else "G" for i in range(n)]
        else:
            kinds = ["S"] * n
    if not kinds or "?" in kinds:
        return None
    par = [k.startswith("P") for k in kinds]
    kinds = [k[-1] for k in kinds]
    if mt in ("falcon_h1", "hymba"):                     # attention and an SSM side by side in every layer
        par = [k in ("G", "S") for k in kinds]
    elif isinstance(d.get("pb_hybrid_layers"), list):
        pb = set(i for i in d["pb_hybrid_layers"] if isinstance(i, int))
        par = [p or (i in pb and k in ("G", "S")) for i, (p, k) in enumerate(zip(par, kinds))]
    # check against the classifier's own counts (kc: FU FT LU LT S R)
    attn = kc["FU"] + kc["FT"] + kc["LU"] + kc["LT"] + kc["S"]
    if mt not in ("falcon_h1", "hymba") and not isinstance(d.get("pb_hybrid_layers"), list):
        if kinds.count("R") != kc["R"]:
            return None
    if "F" not in kinds and "E" not in kinds:
        if kinds.count("G") + kinds.count("S") != attn:
            return None
        if kc["S"] == 0 and kinds.count("G") != kc["FU"] + kc["FT"]:
            return None
    return "".join("P" if p and k == "G" else "Q" if p else k for k, p in zip(kinds, par))


def _kv_shared_layers(d, n):
    """Per-layer: does the layer read another layer's K / V (cross-layer KV sharing)? Only the explicit keys."""
    nsh = mg._num(d.get("num_kv_shared_layers"))
    if nsh and n:
        return [i >= n - nsh for i in range(n)]
    ksl = d.get("kv_source_layers")
    if isinstance(ksl, list) and len(ksl) == n:
        return [not (x is None or x == i) for i, x in enumerate(ksl)]
    return None


def _nope_layers(d, n):
    """Per-layer NoPE flags from an explicit list (no_rope_layers: 1 = RoPE, 0 = NoPE; no_rope_layer_indices)."""
    nrl = d.get("no_rope_layers")
    if isinstance(nrl, list) and len(nrl) == n and len(set(nrl)) > 1:
        return [not x for x in nrl]
    idx = d.get("no_rope_layer_indices")
    if isinstance(idx, list) and idx and n:
        s = set(idx)
        return [i in s for i in range(n)]
    return None


def _code_fields(d, outer, slots, feats):
    """The spec fields whose value the classifier took from a model type's CODE (a per-type table in
    model_graph.py read from the modelling files), not from a config key."""
    mt = str(d.get("model_type") or "").lower()
    code = set()
    if mt in mg.NORM_BY_CODE or not any(k in d for k in ("rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps")):
        code.add("norm")
    if mt in mg.POST_NORM_TYPES or mt in mg.POST_ONLY_TYPES or "post_ln" in feats:
        code.add("place")
    if slots["qknorm"] != "no" and (mt in mg.QKNORM_ALWAYS or mt in mg.QKNORM_BY_CODE
                                    or mt in mg.QKNORM_VARIANT_BY_TYPE or mt in mg.QKNORM_HEAD_CHECKED):
        code.add("qknorm")
    if (mt in mg.RAW_ACT_BY_CODE or mt in mg.GELU_FORM_BY_CODE or mt in mg.ACT_KEY_BY_CODE
            or mt in ("gpt_oss", "gpt_oss_puzzle")):
        code.add("act")
    if mt in mg.UNGATED_BY_CODE or mt in mg.FUSED_GLU_HALF:
        code.add("ffn")
    if mt in mg.PARALLEL_BY_CODE:
        code.add("block")
    if slots["sink"] == "yes" and mt in mg.SINK_TYPES:
        code.add("sink")
    if slots["pos"] == "nope" and mt in mg.NOPE_BY_CODE:
        code.add("pos")
    if "bias_qkv" in feats and mt in mg.BIAS_QKV_TYPES and not any(
            d.get(k) is True for k in ("attention_bias", "use_qkv_bias", "qkv_bias", "add_qkv_bias", "bias")):
        code.add("bias_qkv")
    if "bias_out" in feats and mt in mg.BIAS_OUT_TYPES:
        code.add("bias_out")
    if not any(k in d or k in (outer or {}) for k in ("tie_word_embeddings", "tie_embedding", "weight_tying")):
        code.add("tied")                                   # the config class's default
    if "out_gate" in feats and mt in mg.OUT_GATE_TYPES:
        code.add("out_gate")
    if mg._k_eq_v_all(d):
        code.add("keqv")
    if mt in mg.SIGMOID_ROUTER_TYPES or mt in mg.RENORM_FIXED or mt in mg.GROUP_SCORE_TYPES:
        code.add("router")
    if mt == "gemma2" and slots["window"] == "local":
        code.add("sched")                                  # Gemma-2's alternation is fixed in code
    if "vnorm" in feats and (mt.startswith(("gemma4", "diffusion_gemma")) or mt in mg.VNORM_TYPES):
        code.add("vnorm")
    return code


def attn_kinds(cfg):
    """The attention shape of each layer kind, where the kinds differ in heads / KV heads / head dim / K = V
    (Gemma-4: sliding layers 16 KV x 256 with a window, global layers 4 KV x 512, full, K = V), from the metrics
    plan (graph/arch_metrics.plan, the same reading the KV numbers use). None when every attention layer has one
    shape (a window alone is shown by the Window row)."""
    from graph import arch_metrics as am
    try:
        p = am.plan(cfg)
    except Exception:                                   # a config the metrics pass cannot read
        return None
    kinds = {}
    for g in p["groups"]:
        a = g["attn"]
        if a is None or g["kind"] not in ("full", "sliding", "shared"):
            continue
        k = {"full": "global", "sliding": "sliding", "shared": "kv-shared"}[g["kind"]]
        key = (k, a["nh"], a["nkv"], a["dqk"], a["dv"], g["window"], bool(g["k_eq_v"]))
        kinds[key] = kinds.get(key, 0) + g["n"]
    shapes = {key[1:5] + key[6:] for key in kinds}
    if len(shapes) < 2 or len(kinds) > 6:              # one shape, or per-layer widths (OpenELM): the Heads row
        return None
    order = {"global": 0, "sliding": 1, "kv-shared": 2}
    out = []
    for key in sorted(kinds, key=lambda x: (order[x[0]], x[5] or 0)):
        k, nh, nkv, dqk, dv, w, keqv = key
        e = {"k": k, "n": kinds[key], "h": nh, "kvh": nkv, "hd": dqk}
        if dv != dqk:
            e["vd"] = dv
        if w:
            e["win"] = w
        if keqv:
            e["keqv"] = True
        out.append(e)
    return out


def spec_of(cfg):
    """The spec card of one config: what the classifier reads, in the page's field names (blank = not given)."""
    slots = mg.classify(cfg)
    dims = mg.dims_of(cfg)
    d = mg._lm_config(cfg)
    feats = set(filter(None, slots["feats"].split("+")))
    mt = str(d.get("model_type") or "").lower()
    omt = str(cfg.get("model_type") or "").lower()
    n = dims["layers"]
    s = {"mt": mt}
    if omt and omt != mt:
        s["omt"] = omt
    arch = cfg.get("architectures") or d.get("architectures")
    if isinstance(arch, list) and arch:
        s["arch"] = str(arch[0])
    for k, v in (("L", n), ("Le", dims["eff_layers"]), ("Lp", dims["param_layers"]), ("d", dims["hidden"]),
                 ("h", dims["heads"]), ("kvh", dims["kv_heads"]), ("hd", dims["head_dim"]),
                 ("vocab", dims["vocab"])):
        if v:
            s[k] = int(v)
    ctx = _first(d, "max_position_embeddings", "max_seq_len", "n_positions", "seq_length")
    if _scalar(ctx):
        s["ctx"] = _scalar(ctx)
    # FFN / MoE
    sig = dict(zip(mg.MOE_SIG, dims["moe_sig"]))
    if dims["experts"]:
        s["E"], s["k"], s["ew"] = int(sig["E"]), int(sig["k"]), int(sig["ew"])
        if sig["shared"]:
            s["shw"] = int(sig["shared"])
            nsh = _scalar(_first(d, "n_shared_experts", "num_shared_experts", "moe_num_shared_experts",
                                 "num_shared_expert"))
            if nsh:
                s["nsh"] = nsh
        s["score"] = sig["score"]
        if not str(sig["renorm"]).startswith("?"):
            s["renorm"] = sig["renorm"]
        if sig["n_group"]:
            s["grp"] = [int(sig["n_group"]), int(sig["topk_group"])]
        s["nmoe"], s["ndense"] = int(sig["n_moe"]), int(sig["n_dense"])
        if sig["dense_w"]:
            s["dw"] = int(sig["dense_w"])
        if sig["latent"]:
            s["moe_latent"] = int(sig["latent"])
        mask = mg._moe_layer_mask(d, n) if n else None
        if mask and len(mask) == n:
            s["moe_mask"] = "".join("M" if x else "D" for x in mask)
    elif dims["ffn_total"]:
        s["ffn"] = int(dims["ffn_total"])
    if dims["ffn_widths"]:
        s["ffn_widths"] = [min(dims["ffn_widths"]), max(dims["ffn_widths"])]
    # attention
    if slots["kv"] == "MLA":
        s["mla"] = {k: int(dims[v]) for k, v in (("kv", "kv_latent"), ("q", "q_latent"), ("rope", "mla_rope"),
                                                 ("nope", "mla_nope"), ("v", "mla_v")) if dims[v]}
    if dims["kv_cap"]:
        s["kv_cap"] = int(dims["kv_cap"])
    sw = _scalar(_first(d, "sliding_window", "sliding_window_size", "attention_window_size"))
    if slots["window"] == "local" and sw:
        s["win"] = sw
    _, _, _, _, kc, _ = mg._attn_counts(d, mg._num(_first(d, "num_hidden_layers", "n_layer", "num_layers",
                                                     "num_transformer_layers", "n_layers")) or mg._layer_count(d))
    s["kc"] = {k: int(v) for k, v in kc.items() if v}
    sched = _schedule(d, n, kc) if n else None
    if sched:
        s["sched"] = sched
    sh = _kv_shared_layers(d, len(sched) if sched else n)
    if sh and any(sh):
        s["kvshare"] = "".join("1" if x else "0" for x in sh)
    nl = _nope_layers(d, len(sched) if sched else n)
    if nl:
        s["nope"] = "".join("1" if x else "0" for x in nl)
    if mg._k_eq_v_all(d):
        s["keqv"] = "all"
    elif mg._k_eq_v(d):
        s["keqv"] = "full"
    ak = attn_kinds(cfg)
    if ak:
        s["akinds"] = ak
    ac = d["attn_logit_softcapping"] if "attn_logit_softcapping" in d else mg._cfg_value(d, "attn_logit_softcapping")
    fc = _first(d, "final_logit_softcapping", "logits_soft_cap", "logit_softcap")
    if slots["softcap"] != "no":
        if _scalar(ac) and slots["softcap"] == "attn":
            s["cap_attn"] = _scalar(ac)
        if _scalar(fc):
            s["cap_final"] = _scalar(fc)
    mtp = _scalar(_first(d, "num_nextn_predict_layers", "num_mtp_layers", "mtp_depth"))
    if mtp:
        s["mtp"] = mtp
    rf = mg._recurrent_family(d, slots["mixer"])
    if rf:
        s["rf"] = rf
    s.update(_rope(d) if slots["pos"] in ("rope", "partial_rope") else {})
    if slots["feats"]:
        s["feats"] = slots["feats"]
    s["code"] = sorted(_code_fields(d, cfg if cfg is not d else None, slots, feats))
    return s


# --- the sweep: lineage, code files, findings ----------------------------------------------------------------

def sweep_types(inv, findings):
    """model type -> {parent, how, files (transformers 5.18 paths), hub: {variant: files}, f: [finding ids]}."""
    out = {}
    if not inv:
        return out, {}
    for t, rec in inv.get("types", {}).items():
        e = {}
        for c in rec.get("code", []):
            v = c.get("variant", "")
            if v.startswith("tf5:"):
                e["files"] = [f[4:] for f in c.get("files", [])]
                if c.get("parent"):
                    e["parent"] = c["parent"]
                e["how"] = str(c.get("how", "")).split(" ")[0]
            elif v.startswith("hub:"):
                e.setdefault("hub", {})[v] = {"files": c.get("files", []), "parent": c.get("parent"),
                                              "how": c.get("how")}
        out[t] = e
    for k, f in enumerate(findings or []):
        for t in f.get("types", []):
            out.setdefault(t, {}).setdefault("f", []).append(k)
    repo_var = {}
    for v, repos in inv.get("var_models", {}).items():
        for r in repos:
            repo_var[r] = v
    return out, repo_var


def lineage(types, t, seen=None):
    """The modular-parent chain of a model type, nearest first (transformers 5.18 modular files)."""
    chain, seen = [], set(seen or ())
    while t in types and types[t].get("parent") and types[t]["parent"] not in seen and len(chain) < 12:
        t = types[t]["parent"]
        seen.add(t)
        chain.append(t)
    return chain


# --- the optional passes, compacted to what the page shows ---------------------------------------------------

def compact_metrics(am, nodes):
    """arch_metrics.json (schema_version 1) -> one compact record per page node, or None (shapes aligned by
    _aligned; a node the shapes miss falls back to repos[members[0]])."""
    if not isinstance(am, dict) or am.get("schema_version") != 1:
        return None
    shapes, repos = _aligned(am.get("shapes") or [], nodes), am.get("repos") or {}
    ctx = am.get("contexts") or {}
    Lk = [k for k in ("1k", "4k", "8k", "32k", "128k") if k in ctx] + ["max"]
    out = []
    for nd, sh in zip(nodes, shapes):
        m = (sh or {}).get("metrics") or repos.get(nd["members"][0])
        if not isinstance(m, dict):
            out.append(None)
            continue
        per = lambda sec, f: [(((m.get(sec) or {}).get(L) or {}).get(f)) for L in Lk]
        p = m.get("params") or {}
        lay = m.get("layers") or {}
        kvt = m.get("kv_per_token") or {}
        rec = {
            "p": [p.get(k) for k in ("total", "active", "embedding", "lm_head", "non_embedding",
                                     "active_non_embedding", "routed_experts", "hub_total")],
            "tied": p.get("tied_embeddings"), "hub_why": p.get("hub_delta_reason"),
            "wb": (p.get("weight_bytes") or {}), "awb": (p.get("active_weight_bytes") or {}),
            "lay": lay, "ctx": m.get("max_context"), "ctxsrc": m.get("max_context_source"),
            "ctxunb": bool(m.get("max_context_unbounded")), "anom": p.get("checkpoint_anomaly"),
            "kvt": {"elems": kvt.get("elems") or {}, "bf16": kvt.get("bf16"), "fp8": kvt.get("fp8")},
            "kv": per("kv", "bf16"), "kv8": per("kv", "fp8"),
            "kvL": [(((m.get("kv") or {}).get(L) or {}).get("L")) for L in Lk],
            "kvB": per("kv", "beyond_max"),
            "kvE": {kind: [((((m.get("kv") or {}).get(L) or {}).get("elems") or {}).get(kind)) for L in Lk]
                    for kind in (kvt.get("elems") or {})},
            "inwin": [((((m.get("kv") or {}).get(L) or {}).get("in_window") or {}).get("sliding")) for L in Lk],
            "kvm": (m.get("kv_materialized") or {}).get("per_token_elems"),
            **({"kvmin": per("kv_minimum", "bf16"), "kvmin_note": (m.get("kv_minimum") or {}).get("note")}
               if per("kv_minimum", "bf16") != per("kv", "bf16") and any(per("kv_minimum", "bf16")) else {}),
            "st": m.get("state") or {},
            "fpm": (m.get("flops") or {}).get("matmul"),
            "fp": [(((m.get("flops") or {}).get("prefill") or {}).get(L) or {}).get("total") for L in Lk],
            "fd": [(((m.get("flops") or {}).get("decode") or {}).get(L) or {}).get("total") for L in Lk],
            "fda": [(((m.get("flops") or {}).get("decode") or {}).get(L) or {}).get("attn") for L in Lk],
            "mem": per("memory", "batch1"), "memB": per("memory", "batchB"),
            "memW": per("memory", "weights"), "memK": per("memory", "kv"),
            "bw": per("bandwidth", "bytes"), "bwi": per("bandwidth", "intensity"),
            "valid": m.get("valid") or {}, "prov": m.get("prov") or {}, "notes": m.get("notes") or [],
            "range": (sh or {}).get("range") or {},
        }
        out.append(rec)
    if not any(out):
        return None
    return {"Lk": Lk, "ctx": ctx, "batch_B": am.get("batch_B"), "generated": am.get("generated"),
            "transformers": am.get("transformers_version"), "shapes": out,
            "types": {t: {g: (v.get(g) or {}).get("status") for g in ("params", "flops", "kv", "state")}
                      for t, v in (am.get("types") or {}).items() if isinstance(v, dict)}}


def _aligned(shapes, nodes):
    """A pass's per-shape list lined up with the page's nodes: by index when every entry's rep is a member of
    the node at that index (the passes promise the zoo order), else by the rep's node."""
    if len(shapes) == len(nodes) and all(isinstance(x, dict) and x.get("rep") in nd["members"]
                                         for x, nd in zip(shapes, nodes)):
        return list(shapes)
    at = {m: i for i, nd in enumerate(nodes) for m in nd["members"]}
    out = [None] * len(nodes)
    for x in shapes:
        if isinstance(x, dict) and x.get("rep") in at and out[at[x["rep"]]] is None:
            out[at[x["rep"]]] = x
    return out


def compact_frameworks(fw, nodes, node_of):
    """frameworks.json (schema_version 1) -> per node: native support of its most-downloaded repo ("nat", one
    character per framework), and for each framework that lacks it the shape's best backdoor host ("bd": framework
    index -> [n_hosts, n_exact, host node (or repo), exact, params_x, width_x, depth_x])."""
    if not isinstance(fw, dict) or fw.get("schema_version") != 1:
        return None
    names = list((fw.get("frameworks") or {}).keys())
    repos = fw.get("repos") or {}
    shapes = _aligned(fw.get("shapes") or [], nodes)
    out = []
    for nd, sh in zip(nodes, shapes):
        r = repos.get(nd["members"][0])
        if not isinstance(r, dict):
            out.append(None)
            continue
        nat = r.get("native") or {}
        bd = {}
        for f, b in ((sh or {}).get("backdoor") or {}).items():
            hosts = [h for h in (b or {}).get("hosts") or [] if isinstance(h, dict)]
            if not hosts or f not in names:
                continue
            h = hosts[0]                                   # the pass's best: practical, exact, own family, fewest params
            hi = node_of.get(h.get("host"))
            bd[names.index(f)] = [b.get("n_hosts") or 0, b.get("n_exact") or 0, hi if hi is not None else h.get("host"),
                                  int(h.get("edge") == "exact"), h.get("params_x"), h.get("width_x"), h.get("depth_x"),
                                  int(bool(h.get("impractical"))), int(bool(h.get("same_family"))),
                                  b.get("n_practical", b.get("n_hosts") or 0)]
        # native, one character per framework in names order: 1 yes, t via the framework's Transformers backend /
        # fallback (vLLM, TGI: a separate tier, not native), 0 no, ? the config names neither
        vt = r.get("via_transformers") or {}
        nat_s = "".join("1" if nat.get(f) is True else "t" if vt.get(f) else "0" if nat.get(f) is False else "?"
                        for f in names)
        why = {names.index("executorch"): r["executorch_why"]} if r.get("executorch_why") and "executorch" in names \
            and nat.get("executorch") is False else {}
        out.append({"nat": nat_s, **({"bd": bd} if bd else {}), **({"why": why} if why else {})})
    if not any(out):
        return None
    keep = ("repo", "version", "fetched", "key", "source_note", "urls")
    return {"names": names, "frameworks": {f: {k: v for k, v in (fw["frameworks"][f] or {}).items() if k in keep} for f in names},
            "generated": fw.get("generated"), "shapes": out, "counts": fw.get("counts") or {}}


def compact_registries(reg):
    """framework_registries.json -> what the page's any-HF-id lookup needs to say which frameworks run a config by
    its architecture class / model type: per framework the key, the native lists, the Transformers-backend tier
    (vLLM) / fallback (TGI), and ExecuTorch's per-model export configs and named repos."""
    if not isinstance(reg, dict) or not isinstance(reg.get("frameworks"), dict):
        return None
    out = {}
    for fw, r in reg["frameworks"].items():
        e = {"key": r.get("key"), "a": sorted(r.get("architectures") or []), "t": sorted(r.get("model_types") or []),
             "v": r.get("version")}
        if r.get("transformers_backend"):
            e["tb"] = sorted(r["transformers_backend"])
        if r.get("transformers_fallback"):
            e["fb"] = sorted(r["transformers_fallback"])
        if fw == "executorch":
            e["ec"] = [[c["model_type"], c["hidden"], c["layers"], c["heads"], c["kv_heads"], c["file"]]
                       for c in r.get("export_configs") or []]
            e["ids"] = [x["id"] for x in r.get("model_ids") or []]
        out[fw] = e
    return out


def compact_recipes(rc, nodes, node_of=None):
    """recipes.json (recipes/RECIPES_SCHEMA.md, schema_version 1) -> per node, the recipes of its most-downloaded
    repo. Strings that name the repo or its host (make commands, script paths, CLI lines, reasons) are stored once
    as templates with {repo} / {host} in a shared table "T"; the page substitutes them back, so the text shown is
    the file's own. Per node: "f": framework index -> [route, host, params_x, width_x, depth_x, script, make,
    status, validation, why] (strings as T indices, host as a node index when it is one), "cli": name -> [route,
    host, edge, cmd]. "V": the validation entries the nodes reference."""
    if not isinstance(rc, dict) or rc.get("schema_version") != 1 or not isinstance(rc.get("models"), dict):
        return None
    node_of = node_of or {m: i for i, nd in enumerate(nodes) for m in nd["members"]}
    names = list((rc.get("frameworks") or {}).keys())
    T, tid = [], {}

    def tpl(x, repo, host):
        if not isinstance(x, str):
            return None
        if repo:
            x = x.replace(repo, "{repo}")
        if host and host not in ("universal",):
            x = x.replace(host, "{host}")
        if x not in tid:
            tid[x] = len(T)
            T.append(x)
        return tid[x]

    hostref = lambda h: node_of.get(h, h) if h else None
    vals = {v.get("id"): v for v in rc.get("validation") or [] if isinstance(v, dict)}
    used, out = set(), []
    for nd in nodes:
        repo = nd["members"][0]
        m = rc["models"].get(repo)
        if not isinstance(m, dict):
            out.append(None)
            continue
        f = {}
        for k, name in enumerate(names):
            e = (m.get("frameworks") or {}).get(name)
            if not isinstance(e, dict):
                continue
            c, h = e.get("cost") or {}, e.get("host")
            if e.get("validated_by"):
                used.add(e["validated_by"])
            f[k] = [e.get("route"), hostref(h), c.get("params_x"), c.get("width_x"), c.get("depth_x"),
                    tpl(e.get("script"), repo, h), tpl(e.get("make"), repo, h), e.get("status"),
                    e.get("validated_by"), tpl(e.get("why"), repo, h)]
        cli = {}
        for name, e in (m.get("native_cli") or {}).items():
            if isinstance(e, dict):
                cli[name] = [e.get("route"), hostref(e.get("host")), e.get("edge"), tpl(e.get("cmd"), repo, e.get("host"))]
        out.append({"f": f, "cli": cli, **({"loader": m["loader"]} if m.get("loader") else {})})
    if not any(out):
        return None
    V = {i: [vals[i].get(k) for k in ("framework", "route", "kind", "max_abs_diff", "logit_scale", "greedy_tokens",
                                       "greedy_identical", "decode_max_abs_diff", "pass", "date", "target")]
         for i in sorted(used) if i in vals}
    return {"names": names, "frameworks": {n: {k: v for k, v in (rc["frameworks"][n] or {}).items() if k in ("package", "version")}
                                           for n in names},
            "routes": rc.get("routes") or {}, "generated": rc.get("generated"), "T": T, "V": V, "shapes": out,
            "counts": rc.get("counts") or {}}


# --- the glossary --------------------------------------------------------------------------------------------

GLOSSARY = {
    # norms
    "RMSNorm": ("RMSNorm", "x / rms(x) times a learned per-channel gain. No mean subtraction and no bias."),
    "LayerNorm": ("LayerNorm", "Subtracts the mean, divides by the standard deviation, then applies a learned gain "
                  "and bias (the GPT-2 / BERT norm)."),
    "AffineNorm": ("Affine norm", "The page's top norm value: a norm with gain and bias that both RMSNorm and "
                   "LayerNorm are special cases of."),
    "place:pre": ("Pre-norm", "Each sublayer reads norm(x) and its output is added to the un-normalised residual: "
                  "x + f(norm(x)). The default of modern decoders."),
    "place:sandwich": ("Sandwich norm", "A pre-norm and a second norm on each sublayer's output before the residual "
                       "add: x + norm2(f(norm1(x))) (Gemma-2 / 3 / 4, GLM-4-0414)."),
    "place:post-only": ("Post-only norm", "No pre-norm. Each sublayer reads the raw residual and its output is "
                        "normalised before the add: x + norm(f(x)) (OLMo-2 / 3, EXAONE-4)."),
    "place:post-LN": ("Post-LN", "The original Transformer block: norm(x + f(x)), the residual stream itself is "
                      "normalised (GPT-1, OPT-350m)."),
    "block:parallel": ("Parallel block", "Attention and MLP read the same input and their outputs are summed: "
                       "x + attn(n(x)) + mlp(n(x)) (GPT-J, Pythia, Falcon, Cohere)."),
    "block:sequential": ("Sequential block", "The MLP reads the stream after attention has been added: "
                         "x' = x + attn(n(x)); x'' = x' + mlp(n(x'))."),
    # positions
    "rope": ("RoPE", "Rotary position embedding: q and k are rotated pair by pair by an angle proportional to the "
             "position, so q.k depends on the distance between tokens. Theta sets the frequency base."),
    "partial_rope": ("Partial RoPE", "RoPE on only part of each head (rotary_pct / partial_rotary_factor / "
                     "rotary_dim); the remaining dimensions carry no position."),
    "nope": ("NoPE", "No positional encoding in the attention: order comes only from the causal mask (and from any "
             "recurrent layers)."),
    "alibi": ("ALiBi", "A per-head linear penalty on the attention score proportional to the distance "
              "(slope m_h), no rotation or embedding."),
    "learned_abs": ("Learned absolute positions", "A learned table of position vectors added to the token "
                    "embeddings (GPT-2, OPT)."),
    "sinusoidal": ("Sinusoidal positions", "A fixed sine / cosine table added to the embeddings (XGLM, CTRL)."),
    "theta": ("RoPE theta", "The base of RoPE's frequencies: pair i rotates at theta^(-2i/d) radians per token. "
              "Larger theta means slower rotation and longer reach."),
    "scaling": ("RoPE scaling", "A change to the RoPE frequencies for contexts beyond training: linear, dynamic "
                "NTK, YaRN, llama3, longrope, ... with a factor and the original context."),
    "nope_layers": ("NoPE layers", "Some layers rotate q and k and others use no position at all (SmolLM3, "
                    "Llama-4 style interleaving)."),
    # attention
    "MHA": ("MHA", "Multi-head attention: every query head has its own key and value head."),
    "GQA": ("GQA", "Grouped-query attention: groups of query heads share one key / value head, shrinking the KV "
            "cache by heads / KV heads."),
    "MQA": ("MQA", "Multi-query attention: all query heads share a single key / value head."),
    "MLA": ("MLA", "Multi-head latent attention (DeepSeek-V2 / V3): keys and values are decompressed from a small "
            "per-token latent (kv_lora_rank) plus a shared rotary key; the cache holds only the latent."),
    "kv:none": ("No attention KV", "A purely recurrent model: no softmax attention, so no key / value cache."),
    "window:local": ("Sliding window", "Some or all layers attend only to the last W tokens. The page counts "
                     "global (full-window) and sliding layers separately."),
    "window:full": ("Full attention", "Every layer can attend to the whole context."),
    "head_dim": ("Head dim", "The width of one query / key head (the rank bound of one attention pattern)."),
    "kv_heads": ("KV heads", "The number of distinct key / value heads per layer (MLA: every head reads its own K / V "
                 "from the latent)."),
    "keqv": ("K = V", "The layer has no v_proj: the value is computed from its key projection (Gemma-4 "
             "attention_k_eq_v on the global layers, DeepSeek-V4 on every layer). Gemma-4 still caches K and V as two "
             "tensors (V = v_norm(k_proj x), K = RoPE(k_norm(k_proj x))), as transformers, vLLM and llama.cpp "
             "allocate them, and the KV figures count both; the one-tensor theoretical minimum is shown beside them. "
             "DeepSeek-V4's cache holds one tensor."),
    "kvshare": ("Cross-layer KV", "A layer reads another layer's keys and values instead of computing its own "
                "(CLA, YOCO, Gemma-3n num_kv_shared_layers)."),
    "sink": ("Attention sinks", "A learned per-head logit in the softmax's denominator, so a head can attend to "
             "'nothing' (gpt-oss, MiMo-V2)."),
    "softcap": ("Logit softcap", "c tanh(x / c) applied to the attention scores (attn) and / or the output logits "
                "(final), bounding them by c (Gemma-2)."),
    "mtp": ("MTP", "Multi-token prediction heads trained to predict tokens further ahead (DeepSeek-V3). They do not "
            "change the main next-token logits."),
    "kv_cap": ("KV width", "Per-token key / value width a layer caches: KV heads x head dim, or the MLA latent "
               "plus its rotary key."),
    # qk-norm variants (model_graph.QKNORM_VARIANTS)
    "qknorm": ("qk-norm", "A norm on the queries and keys inside attention, before the dot product. The page names "
               "the flavour the code builds, not just yes / no."),
    "qknorm:head": ("qk-norm: per head", "RMSNorm over each head's q and k with one head_dim gain, before RoPE "
                    "(Qwen3, Gemma-3)."),
    "qknorm:head_distinct": ("qk-norm: per-head gains", "Per-head RMSNorm with a separate gain for every head "
                             "(PLaMo-2)."),
    "qknorm:head_free": ("qk-norm: weight-free", "Per-head RMSNorm with no gain (Llama-4 L2Norm, nanochat)."),
    "qknorm:head_post": ("qk-norm: after RoPE", "Per-head RMSNorm with a gain applied after the rotation "
                         "(Hunyuan)."),
    "qknorm:proj": ("qk-norm: whole projection", "One RMSNorm over the entire q (and k) projection, all heads "
                    "jointly (OLMo-2 / 3, OLMoE, MiniMax-M2)."),
    "qknorm:proj_ln": ("qk-norm: projection LayerNorm", "One LayerNorm over the whole projection (OLMo-1, Molmo)."),
    "qknorm:head_ln": ("qk-norm: per-head LayerNorm", "Per-head LayerNorm with per-head gains, no bias (Cohere, "
                       "StableLM-2-12B)."),
    "qknorm:head_ln_bias": ("qk-norm: per-head LayerNorm + bias", "Per-head LayerNorm with per-head gains and "
                            "biases (Chameleon)."),
    "qknorm:ln_shared_bias": ("qk-norm: shared LayerNorm", "Per-head LayerNorm with one shared gain and bias "
                              "(Persimmon)."),
    "qknorm:dsv4": ("qk-norm: DeepSeek-V4", "A weight-free per-head q norm and a weighted norm of the single "
                    "K = V head."),
    "qknorm:qkv_input": ("qk-norm: input norm", "A second RMSNorm on the attention input (CENO)."),
    "qknorm:unverified": ("qk-norm: unverified", "The config asks for a q / k norm but the modelling code was not "
                          "available to say which flavour."),
    # FFN
    "glu": ("Gated MLP (GLU)", "down(act(gate x) * up x): SwiGLU when act is SiLU, GeGLU with GeLU."),
    "vanilla": ("Plain MLP", "down(act(up x)), no gate (GPT-2, OPT, Falcon, Phi-2)."),
    "moe": ("MoE", "Mixture of experts: a router picks the top-k of E expert MLPs per token; only those run."),
    "moe_shared": ("MoE + shared expert", "A MoE layer plus one or more always-on shared expert MLPs."),
    "top_k": ("Top-k", "How many routed experts each token uses."),
    "score": ("Router score", "How router logits become expert weights: softmax, sigmoid (DeepSeek-V3), "
              "sqrt-softplus (DeepSeek-V4) or sparsemixer (Phi-MoE)."),
    "renorm": ("Top-k renormalisation", "Whether the selected experts' weights are renormalised to sum to 1 "
               "(norm_topk_prob)."),
    "grp": ("Group-limited routing", "Experts are split into n_group groups; each token routes only inside its "
            "topk_group best groups (DeepSeek-V3)."),
    "dense_layers": ("Dense layers in a MoE", "Layers whose FFN is a plain MLP although the model is MoE "
                     "(first_k_dense_replace, mlp_only_layers, ...)."),
    # activations
    "SiLU": ("SiLU / swish", "x sigmoid(x). The gate activation of SwiGLU."),
    "GeLU": ("GeLU", "x Phi(x) with the exact (erf) normal CDF."),
    "GeLU_tanh": ("GeLU (tanh)", "The tanh approximation of GeLU (gelu_new / gelu_pytorch_tanh)."),
    "ReLU": ("ReLU", "max(0, x)."),
    "relu2": ("Squared ReLU", "relu(x)^2 (Primer, nanochat, RWKV channel mix)."),
    "swigluoai": ("SwiGLU-OAI", "gpt-oss's clamped SwiGLU: (up + 1) g sigmoid(1.702 g), both clamped at +-7."),
    "mish": ("Mish", "x tanh(softplus(x))."),
    # mixers / generation
    "softmax": ("Softmax attention", "Every mixer layer is ordinary attention."),
    "hybrid": ("Hybrid", "Softmax attention layers interleaved with recurrent ones (SSM, linear attention, gated "
               "DeltaNet, short conv). The schedule strip shows which is where."),
    "ssm": ("SSM", "A state-space recurrent mixer only (Mamba, RWKV): a constant-size state instead of a KV cache."),
    "linear": ("Linear attention", "Attention without the softmax, run as a recurrence (RetNet, GLA)."),
    "ar": ("Autoregressive", "Causal next-token generation."),
    "diffusion": ("Diffusion LM", "Bidirectional denoising of masked tokens (LLaDA, Dream)."),
    "block": ("Block diffusion", "Bidirectional inside a block, causal across blocks (SDAR)."),
    "looped": ("Looped depth", "The same layers are run several times (recurrent depth): fewer distinct layers "
               "than the unrolled depth."),
    "untied": ("Untied depth", "Every layer has its own weights."),
    # embeddings
    "tied": ("Tied embeddings", "The output head is the transposed input embedding table (lm_head = E^T)."),
    "vocab": ("Vocab", "Rows of the embedding table (vocab_size, padding included)."),
    "ctx": ("Context", "max_position_embeddings (or the family's equivalent key) as the config states it."),
    # page terms
    "shape": ("Shape", "One architecture at one size. Fine-tunes, quantisations and re-uploads with the same "
              "architecture and dimensions share a shape."),
    "from_code": ("From code", "This value is not in the config: the classifier took it from the model type's "
                  "modelling code (a per-type table in graph/model_graph.py)."),
    "frontier": ("Frontier", "The shapes no other model emulates. Between them they emulate every model here."),
    "emulates": ("Emulates", "A emulates B when some setting of A's weights makes A compute what B computes, "
                 "exactly or within a measured, priced error."),
    "params": ("Params", "Hub safetensors count of the repo where published; the shape-level number falls back to "
               "the size in the repo name or an estimate from the config."),
    "active": ("Active params (incl. embeddings)", "Parameters used per token: dense layers, shared experts and the "
               "top-k routed experts, plus the input embedding table(s) and the LM head (from the metrics pass). This "
               "is what a decode step touches; vendors usually quote the number without the embeddings."),
    "active_ex": ("Active params (excl. embeddings)", "The vendor convention: active params minus the input embedding "
                  "tables (per-layer embeddings included); an untied LM head stays. gpt-oss-20b: 4.19B incl. "
                  "embeddings, 3.61B excl. (OpenAI's '3.6B active'); Gemma-4-E4B: 7.46B incl., 3.97B excl. "
                  "(Google's '4B effective')."),
    "date": ("First on the Hub", "The repo's creation date on the Hugging Face Hub. For a shape, its earliest "
             "repo. Not a paper date."),
}

# the features (model_graph.FEATURES) the glossary explains with the classifier's own description
FEATURE_TITLES = {
    "tied_emb": "Tied embeddings", "out_gate": "Attention output gate", "switch_lora": "Switched LoRA",
    "hyper_conn": "Hyper-connections", "engram": "Engram n-gram memory", "ple": "Per-layer embeddings",
    "value_emb": "Value embeddings", "attn_res": "Attention residuals", "vnorm": "V norm",
    "bias_qkv": "q / k / v biases", "bias_out": "Output biases", "post_norm": "Sandwich norm",
    "post_only": "Post-only norm", "router_sigmoid": "Sigmoid router", "router_bias": "Router selection bias",
    "shared_gate": "Shared-expert gate", "router_group": "Group-limited routing", "router_in_gate": "Input-gated router",
    "nope_layers": "NoPE layers", "dsa": "DeepSeek sparse attention", "kv_compress": "Compressed KV",
    "post_ln": "Post-LN", "clip_qkv": "Clipped q / k / v", "act_clip": "Clamped SwiGLU",
    "router_sqrtsoftplus": "sqrt-softplus router", "hash_route": "Hash routing", "selfcond": "Self-conditioning",
    "selfcond_norm": "Self-conditioning norm", "zamba_shared": "Zamba shared block",
}


def glossary():
    g = {k: {"t": t, "d": d} for k, (t, d) in GLOSSARY.items()}
    for k, (tw, hw, desc) in mg.FEATURES.items():
        g["f:" + k] = {"t": FEATURE_TITLES.get(k, k), "d": desc, "rule": [tw, hw]}
    return g


# --- the reference block -------------------------------------------------------------------------------------

def compact_extension(ext, nodes):
    """The candidate snapshot extension (graph/data/hub_extension.json) as search aliases: node index -> [[repo,
    downloads, params status, params total]] for the extension repos that JOIN an existing zoo shape (google/gemma-4-31B-it
    lands on its shape). Repos in new shapes are left to the merge (the relation changes) and to the Look up view."""
    if not isinstance(ext, dict):
        return None
    node_of = {m: i for i, nd in enumerate(nodes) for m in nd["members"]}
    out = {}
    for repo, r in (ext.get("repos") or {}).items():
        i = node_of.get(r.get("joins_shape"))
        if i is None or repo in node_of:
            continue
        out.setdefault(str(i), []).append([repo, r.get("downloads"), r.get("valid_params"),
                                           ((r.get("metrics") or {}).get("params") or {}).get("total")])
    for v in out.values():
        v.sort(key=lambda x: -(x[1] or 0))
    return {"generated": ext.get("generated"), "n": sum(len(v) for v in out.values()), "nodes": out}


def ref_data(nodes, configs, inv=None, findings=None, meta=None, am=None, fw=None, rc=None, reg=None, ext=None):
    """The reference half of the page, aligned with zoo_data()["nodes"]."""
    meta = meta or {}
    types, repo_var = sweep_types(inv, findings)
    node_of = {m: i for i, nd in enumerate(nodes) for m in nd["members"]}
    shapes, mdate, mpar = [], [], []
    for nd in nodes:
        rep = nd["members"][0]
        s = spec_of(configs[rep])
        v = repo_var.get(rep)
        if v:
            s["var"] = v
        shapes.append(s)
        mdate.append([((meta.get(m) or {}).get("created_at") or "")[:10] for m in nd["members"]])
        mpar.append([int((meta.get(m) or {}).get("hf_total") or 0) for m in nd["members"]])
    used = {s["mt"] for s in shapes} | {s.get("omt") for s in shapes if s.get("omt")}
    for s in shapes:
        used |= set(lineage(types, s["mt"]))
    tsub = {t: types[t] for t in sorted(used) if t in types}
    # keep only the hub variants the page's representatives use
    want = {s.get("var") for s in shapes if s.get("var")}
    for t, e in tsub.items():
        if "hub" in e:
            hub = {k: v for k, v in e["hub"].items() if k in want}
            tsub[t] = {**e, "hub": hub} if hub else {k: v for k, v in e.items() if k != "hub"}
    fidx = sorted({k for e in tsub.values() for k in e.get("f", [])})
    fl = {str(k): {"key": findings[k]["key"], "note": findings[k].get("note", ""),
                   "dir": findings[k].get("direction", ""), "n": findings[k].get("live_models", 0),
                   "suggest": (findings[k].get("suggest") or [""])[0]} for k in fidx} if findings else {}
    # members whose upload is not the model its config describes (params_triage bucket B): node -> [[member index,
    # reason, the architecture's params]]; the member list shows these instead of the upload's count
    arep = (am or {}).get("repos") or {} if isinstance(am, dict) else {}
    manom = {}
    for i, nd in enumerate(nodes):
        for k, m in enumerate(nd["members"]):
            pr = (arep.get(m) or {}).get("params") or {}
            if pr.get("checkpoint_anomaly"):
                manom.setdefault(str(i), []).append([k, pr["checkpoint_anomaly"], pr.get("total")])
    return {
        "shapes": shapes, "mdate": mdate, "mpar": mpar, "manom": manom,
        "types": tsub, "findings": fl, "tf_url": TF_URL, "glossary": glossary(),
        "sweep": bool(inv), "metrics": compact_metrics(am, nodes), "frameworks": compact_frameworks(fw, nodes, node_of),
        "recipes": compact_recipes(rc, nodes, node_of), "pairs_file": SIDE_PAIRS,
        "fwreg": compact_registries(reg), "ext": compact_extension(ext, nodes),
    }


# --- the lazy pair file: why a host that dominates on capacity still does not emulate a target ----------------

def pair_blockers(P, ok, cap):
    """For every (target i, host j) pair where j dominates i on raw capacity but does not emulate it:
    graph.frontier.blockers(P[i], P[j]), deduplicated. Returns {"strings", "combos", "bits", "idx"}:
    bits = the pairs (row i = targets, np.packbits), idx = one combo id per set bit, in row-major order."""
    import numpy as np

    from graph.frontier import blockers
    x = cap & ~ok
    np.fill_diagonal(x, False)
    strings, sid, combos, cid, idx = [], {}, [], {}, []
    for i, j in zip(*np.where(x)):
        bl = tuple(blockers(P[i], P[j]))
        ids = []
        for b in bl:
            if b not in sid:
                sid[b] = len(strings)
                strings.append(b)
            ids.append(sid[b])
        key = tuple(ids)
        if key not in cid:
            cid[key] = len(combos)
            combos.append(list(key))
        idx.append(cid[key])
    bits = np.packbits(x, axis=1)
    arr = np.array(idx, dtype="<u2" if len(combos) < 65536 else "<u4")
    return {"n": int(len(P)), "row": int(bits.shape[1]), "bits": base64.b64encode(bits.tobytes()).decode(),
            "idx": base64.b64encode(arr.tobytes()).decode(), "width": int(arr.itemsize),
            "strings": strings, "combos": combos}


def detail_blob(nodes, configs, breakdown=BREAKDOWN):
    """Per page node: the representative's plan as graph/templates/arch_breakdown.js reads it (arch_breakdown.page_plan),
    its per-field statuses and notes (graph/data/arch_breakdown.json.gz); gzip + base64, decompressed by the page on
    the first card that shows the detail. Returns (blob, bd tables) or (None, None)."""
    import gzip

    from graph import arch_breakdown as BD
    if not os.path.exists(breakdown):
        return None, None
    with gzip.open(breakdown, "rt") as f:
        bd = json.load(f)
    am = _load_json(ARCH_METRICS) or {}
    out = []
    for nd in nodes:
        rep = nd["members"][0]
        rec = bd["repos"].get(rep) or {}
        try:
            p = BD.plan_ordered(configs[rep])
        except Exception:
            out.append(None)
            continue
        why = ((((am.get("repos") or {}).get(rep) or {}).get("params") or {}).get("hub_delta_reason") or "")
        if p["tied"] and "tied head stored" in why:
            p["head_stored"] = True
        out.append({"r": rep, "p": BD.page_plan(p), "v": rec.get("valid"), "n": rec.get("notes") or []})
    raw = json.dumps(out, separators=(",", ":")).encode()
    blob = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode()
    hw = [{k: h.get(k) for k in ("id", "name", "mem_bw_gbps", "mem_gb", "flops", "notes")} | {
        "source": ((h.get("sources") or [{}])[0]).get("url")} for h in BD.hardware().get("hardware", [])]
    tables = {"quant": __import__("graph.gguf_quant", fromlist=["tables"]).tables(), "hardware": hw,
              "calib": BD.gguf_calibration(), "hub": bd.get("hub_checks"), "conventions": bd.get("conventions"),
              "roofline": _roofline_summary()}
    return blob, tables


def _roofline_summary():
    r = _load_json(os.path.join(HERE, "data", "roofline_check.json")) or {}
    return [{k: x.get(k) for k in ("hardware", "framework", "model", "weights", "batch", "tokens_per_s",
                                   "bound_tokens_per_s", "ratio", "url")} for x in r.get("rows", [])]


def zoo_models(items=None):
    """The models the page shows a card for: every classified, non-junk config of ZOO_CONFIGS."""
    if items is None:
        items = list(mg.iter_json_configs(mg.ZOO_CONFIGS))
    models = []
    for name, d in items:
        if mg._junky(name):
            continue
        c = mg.classify(d)
        if c is not None:
            models.append({"name": name, "slots": c, "dims": mg.dims_of(d)})
    return models


def build(out, configs=None, pairs=True, metrics=ARCH_METRICS, frameworks=FRAMEWORKS, recipes=RECIPES):
    """Write the page (and its pair file next to it)."""
    from graph.model_graph import ZOO_CONFIGS, ZOO_TEMPLATE, _dedup_points, emul_matrix, zoo_data
    items = list(mg.iter_json_configs(ZOO_CONFIGS))
    if configs is None:
        configs = dict(items)
    models = zoo_models(items)
    data = zoo_data(models, configs=configs)
    inv, findings, meta, fwj = _load_json(INVENTORY), _load_json(FINDINGS), _load_json(META), _load_json(frameworks)
    ref = ref_data(data["nodes"], configs, inv=inv, findings=findings, meta=meta, am=_load_json(metrics), fw=fwj,
                   rc=_load_json(recipes), reg=_load_json(REGISTRIES), ext=_load_json(EXTENSION))
    enc = lambda o: json.dumps(o, separators=(",", ":")).replace("</", "<\\/")
    am_js = ""
    if os.path.exists(AM_JS) and os.path.exists(AM_TABLES):
        with open(AM_JS) as f:
            am_js = f.read().replace("</script", "<\\/script")
    with open(ZOO_TEMPLATE) as f:
        html = f.read().replace("__ZOO_DATA__", enc(data)).replace("__ZOO_REF__", enc(ref))
    html = html.replace("__AM_TABLES__", enc(_load_json(AM_TABLES)) if am_js else "null").replace("__AM_JS__", am_js)
    blob, bdt = detail_blob(data["nodes"], configs) if am_js else (None, None)
    js = lambda p: open(p).read().replace("</script", "<\\/script") if os.path.exists(p) else ""
    html = html.replace("__BD_TABLES__", enc(bdt) if bdt else "null").replace("__BD_DETAIL__", blob or "")
    html = html.replace("__GQ_JS__", js(GQ_JS) if bdt else "").replace("__BD_JS__", js(BD_JS) if bdt else "")
    from graph import zoo_variants                     # the Variants view (graph/zoo_variants.py)
    hub = {r["id"]: r.get("downloads") for r in (_load_json(ZOO_CONFIGS) or []) if isinstance(r, dict)}
    html = zoo_variants.inject(html, data["nodes"], configs, meta=meta, inv=inv, findings=findings, fw=fwj, hub=hub)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    msg = f"wrote {out}: {data['n_shapes']} shapes ({data['n_models']} models), {len(html) / 1e3:.0f} kB"
    if pairs:
        P = _dedup_points(models)
        assert [frozenset(p["members"]) for p in P] == [frozenset(nd["members"]) for nd in data["nodes"]]
        pb = pair_blockers(P, emul_matrix(P, mode="payable"), emul_matrix(P, mode="capacity"))
        side = os.path.join(os.path.dirname(os.path.abspath(out)), SIDE_PAIRS)
        with open(side, "w") as f:
            f.write(json.dumps(pb, separators=(",", ":")))
        msg += f"; {side}: {len(pb['strings'])} blocker texts ({os.path.getsize(side) / 1e3:.0f} kB)"
    print(msg)
    return data, ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="dist/zoo.html")
    ap.add_argument("--no-pairs", action="store_true", help="skip the lazy blocker file (about a minute)")
    ap.add_argument("--metrics", default=ARCH_METRICS, help="arch_metrics.json (graph/arch_metrics_schema.md)")
    ap.add_argument("--frameworks", default=FRAMEWORKS, help="frameworks.json (graph/arch_metrics_schema.md)")
    ap.add_argument("--recipes", default=RECIPES, help="recipes.json (recipes/RECIPES_SCHEMA.md)")
    a = ap.parse_args()
    build(a.out, pairs=not a.no_pairs, metrics=a.metrics, frameworks=a.frameworks, recipes=a.recipes)


if __name__ == "__main__":
    main()
