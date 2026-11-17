"""Frontier study, part 3: the growth breakdown (from growth_raw.json), the presentation numbers, and the report
dist/frontier_study.md (tables; charts in dist/frontier_study/).

Run (after graph.frontier_study and graph.frontier_study_growth):
      python -m graph.frontier_study_report
"""
from __future__ import annotations

import collections
import json
import os
import re

OUT = "dist/frontier_study"
SPECS = "verify/r3/specs"

CONVERSION = re.compile(r"(gguf|mlx|awq|gptq|nvfp4|fp8|int4|int8|bnb|[-_]\d+bit|oq\d|optiq|exl2|quant|fp4|w4a16|"
                        r"-hf$|bf16|fp16mix)", re.I)
DERIVED = re.compile(r"(reap|cut\d|prune|pruned|slim|whittle|distill|merge|merged|upcycl|lobotom|\d+x\d+b|moe-a|"
                     r"puzzle|frankenmoe|-moe-patched|unleashed|abliterat|depth\d)", re.I)


SUMMARY = {   # what the pass-5 reading credited that the code does not do (from the specs and round-3 commit messages)
    "spot_vanilla_glu_bias": "a GLU target with an FFN up bias was played by up-bias-free GLU hosts; the up bias needs "
                             "a constant no pre-norm host forms (spot check; transitivity-forced after glu_ungated)",
    "spot_act_width_oai": "activation-width fits were credited on swigluoai hosts, ignoring gpt-oss's gate clamp",
    "per_layer_head_shapes": "per-layer head counts / sizes (Laguna, DeciLM, Domyn-Edge) read as one shape: the host's "
                             "widest shape was credited on every layer",
    "ngram_memory_unread": "n-gram memories (KLM, Crowd, AllGDN, needle Engram) not read: treated as plain stacks",
    "attn_res_keys_unread": "AttnRes depth-softmax residuals (ARLoop, KLM, GPT-S3) not read",
    "value_emb_keys_unread": "value-embedding inputs (ARLoop, PreNormLoop, sn38, KLM) not read",
    "memory_attention": "causal summary-memory attention (haru, Bluebird, EdgeRock) not read",
    "loop_depth_unread": "loop depth (Ouro total_ut_steps, CFRD, LFM3-RD, haru) not read: depth / looped flag wrong",
    "canon_conv_layers": "Canon causal depthwise convs (dhara, needle3, Ethoric) not read",
    "hadamard_mlp_unlisted": "needle's HadamardMLP read as a plain FFN",
    "local_rope_theta_unread": "two RoPE tables (OLMo-3 YaRN on full layers only, PLaMo-3 local theta) read as one",
    "loop_boundary_norm": "the loop-boundary norm (Ouro, HRM, Dory, Nanbeige4.2) not read",
    "moe_capacity_drop": "Hunyuan remote MoE capacity drops not read",
    "text_encoder_input_embeds": "Breeze-TTS text ids are a bidirectional encoder's output, read as token embeddings",
    "v_head_dim_narrow": "a V head width different from the QK width (MiMo-V2, Hymba, needle3) not compared",
    "per_token_early_exit": "per-token early exit (Ouro, LFM3-RD) not read",
    "grouped_rmsnorm": "grouped RMSNorm (K2-Horizon, Jais) read as full-width RMSNorm",
    "bolmo_byte_patch_hierarchy": "Bolmo's byte-patch hierarchy read as a plain byte LM",
    "block_routing": "LLaDA2.2 block routing (per-block expert capacity) read as per-token top-k",
    "cross_attention_decoder_unflagged": "Parler-TTS's decoder cross-attends to an encoder (plus its LayerNorm and "
                                         "sinusoidal positions)",
    "hrm_hierarchical_loop": "HRM's two-state hierarchical loop read as a plain loop",
    "out_gate_by_code_unlisted": "attention output gates fixed in code (17 types) not read",
    "bias_out_by_code_unlisted": "output biases fixed in code (RecurrentGemma, JetMoE, Nae1) not read",
    "moa_routed_attention_heads": "JetMoE's mixture-of-attention routed heads not read",
    "head_dim_key_unread": "head widths under attention_head_dim / kv_channels / ... not read",
    "hybrid_layer_schedule_unread": "hybrid layer schedules (GPT-X3, KLM, Solar-Open2) not read",
    "mova_routed_values": "Jais-3-MoVA routed values not read",
    "norm_placement_keys_unread": "sandwich / post-only norms fixed in code (PLaMo-3, Molmo, Bolmo) not read",
    "group_mask_zero_fill": "the first-release DeepSeek router 0-fills dropped groups (MAI-DS-R1, A.X-K2, JoyAI): a "
                            "different group code",
    "doge_dynamic_mask": "Doge's per-key dynamic mask not read",
    "residual_gain_vector": "residual gain vectors (Doge, ZAYA) not read",
    "content_warped_rope": "GPT-X2's content-warped, non-causal RoPE phase not read",
    "eos_document_mask": "eos document masking (Ethoric, KLM) not read",
    "parallel_by_code_unlisted": "parallel blocks fixed in code (Bitsy) not read",
    "qknorm_by_code_unlisted": "qk-norms fixed in code (23 types) not read",
    "nanochat_port_unlisted": "nanochat ports (relu2, ungated FFN, weight-free qk-norm) not recognised",
    "hybrid_mla_layers_unread": "Sabi's shared pre-RoPE MLA layers not read",
    "xsa_self_value_projection": "XSA (the self-value projection removal) not read",
    "prefix_lm_mask": "a prefix-LM mask (DIVE-Doc) read as causal",
    "no_model_type_longcat": "LongCat (no model_type) unplaced; its shortcut-connected MoE block",
    "identity_experts": "LongCat identity expert slots not read",
    "mm200_core_keys_unread": "MM200's gate after o_proj and core keys not read",
    "attn_output_saturation": "Qwen3-D-ROS attention output saturation not read",
    "fla_num_heads_unplaced": "fla DeltaNet heads / GatedMLP width unplaced",
    "qknorm_always_remote_lacks": "remote Doge has no q / k norm",
    "hierarchical_moe_unread": "Meditor-2's two-level argmax router read as flat top-k",
    "swa_prefill_key_truncation": "Meditor's K / V truncation before a causal SDPA not read",
    "not_a_causal_lm": "rerankers / non-LMs counted as LMs (removed)",
    "router_score_unread": "router scoring read from code (Mark-SLM-V3 now hosted)",
    "router_feats_on_dense_model": "router features on dense configs (cobalt) were spurious walls",
    "dsv4_layer_types_schema": "the tf5 DeepSeek-V4 layer_types schema",
    "remote_code_not_config_family": "remote wrappers placed in their config family",
    "renorm_by_code_unlisted": "router renormalisation fixed in code",
    "zoo_deep_topk_legacy": "DeepConfig's legacy top_k (TR-MOE-306: top-2)",
    "b_act_clip_host": "the act_clip host side is a config lever (swiglu_limit off): K2-Horizon-375B / K-EXAONE now "
                       "host more",
    "config_default_unread": "class defaults read from the config class",
}


def spec_summary(key):
    if key in SUMMARY:
        direction = ""
        p = f"{SPECS}/{key}.md"
        if os.path.exists(p):
            lines = open(p).read().splitlines()
            m = re.search(r"direction ([a-z\- ]+?)(;|$)", lines[1]) if len(lines) > 1 else None
            direction = m.group(1).strip() if m else ""
        return direction or ("over-credit (spot check)" if key.startswith("spot_") else ""), SUMMARY[key]
    p = f"{SPECS}/{key}.md"
    if not os.path.exists(p):
        return "", ""
    lines = open(p).read().splitlines()
    direction = ""
    if len(lines) > 1:
        m = re.search(r"direction ([a-z\- ]+?)(;|$)", lines[1])
        direction = m.group(1).strip() if m else ""
    body = []
    for ln in lines[2:]:
        if ln.startswith("## 1"):
            continue
        if ln.startswith("## "):
            break
        if ln.strip():
            body.append(ln.strip())
    txt = " ".join(body)
    txt = re.sub(r"`", "", txt)
    first = re.split(r"(?<=[.:])\s", txt, maxsplit=1)[0]
    return direction, first[:260]


def growth():
    raw = json.load(open(f"{OUT}/growth_raw.json"))
    on = raw["all_on"]["frontier"]
    off = raw["all_off"]["frontier"]
    names = lambda fr: {m for cl in fr for m in cl}
    N_on, N_off = names(on), names(off)
    new = [cl for cl in on if not set(cl) & N_off]               # frontier classes with no pass-5 frontier member
    gone = [cl for cl in off if not set(cl) & N_on]
    keys = sorted(k[4:] for k in raw if k.startswith("loo:"))
    loo = {k: names(raw[f"loo:{k}"]["frontier"]) for k in keys}
    loi = {k: names(raw[f"loi:{k}"]["frontier"]) for k in keys}
    rows = []
    for cl in new:
        need = [k for k in keys if not set(cl) & loo[k]]             # off alone -> the class leaves
        alone = [k for k in keys if set(cl) & loi[k]]               # on alone (from pass 5) -> the class joins
        rows.append(dict(name=cl[0], members=len(cl), necessary=need, sufficient_alone=alone))
    left = []
    for cl in gone:
        need = [k for k in keys if set(cl) & loo[k]]                 # off alone -> the class is back
        alone = [k for k in keys if not set(cl) & loi[k]]           # on alone -> it leaves
        left.append(dict(name=cl[0], members=len(cl), necessary=need, sufficient_alone=alone))
    per_key = {}
    for k in keys:
        h_loo, h_loi = len(raw[f"loo:{k}"]["frontier"]), len(raw[f"loi:{k}"]["frontier"])
        per_key[k] = dict(loo=h_loo - len(on), loi=h_loi - len(off),
                          adds=[r["name"] for r in rows if k in r["necessary"] or k in r["sufficient_alone"]],
                          removes=[r["name"] for r in left if k in r["necessary"] or k in r["sufficient_alone"]])
    return dict(n_on=len(on), n_off=len(off), new=rows, gone=left, per_key=per_key,
                shapes_on=raw["all_on"]["shapes"], shapes_off=raw["all_off"]["shapes"])


def load_meta():
    snap = json.load(open("graph/hub_10k.json"))
    dl = {r["id"]: r.get("downloads", 0) or 0 for r in snap}
    mt = {}
    for r in snap:
        c = r.get("config") or {}
        t = c.get("model_type") or ""
        for k in ("text_config", "llm_config", "language_config"):
            if isinstance(c.get(k), dict) and c[k].get("model_type"):
                t = c[k]["model_type"]
        mt[r["id"]] = str(t).lower()
    return dl, mt


def provenance(members):
    """'conversion' (quantised / re-packaged copy), 'derived shape' (pruned / merged / upcycled), or 'original'."""
    if any(not CONVERSION.search(m) and not DERIVED.search(m) for m in members):
        return "original"
    if all(CONVERSION.search(m) and not DERIVED.search(m) for m in members):
        return "conversion"
    return "derived shape"


def presentation(att, dl, mt):
    M = att["models"]
    fam = collections.defaultdict(list)
    lab = collections.defaultdict(list)
    for nm, r in M.items():
        types = sorted({mt.get(m, "") for m in r["all_members"]} - {""}) or ["?"]
        fam[types[0]].append(nm)
        lab[nm.split("/")[0].lower()].append(nm)
    downloads = {nm: sum(dl.get(m, 0) for m in r["all_members"]) for nm, r in M.items()}
    tot = sum(downloads.values())
    ordered = sorted(downloads.items(), key=lambda kv: -kv[1])
    cum, k50, k90 = 0, None, None
    for i, (nm, d) in enumerate(ordered, 1):
        cum += d
        if k50 is None and cum >= 0.5 * tot:
            k50 = i
        if k90 is None and cum >= 0.9 * tot:
            k90 = i
    prov = collections.Counter(provenance(r["all_members"]) for r in M.values())
    return dict(n_arch_families=len(fam), n_labs=len(lab), families=sorted(((k, len(v)) for k, v in fam.items()),
                                                                             key=lambda kv: -kv[1]),
                labs=sorted(((k, len(v)) for k, v in lab.items()), key=lambda kv: -kv[1]),
                downloads_total=tot, top_by_downloads=ordered[:25], k50=k50, k90=k90,
                provenance=dict(prov),
                prov_list={nm: provenance(r["all_members"]) for nm, r in M.items()},
                family_of={nm: (sorted({mt.get(m, "") for m in r["all_members"]} - {""}) or ["?"])[0]
                           for nm, r in M.items()})


CAPACITY_ITEMS = {"hidden", "kv_cap", "q_cap", "qheads", "head_dim", "ffn width", "depth (eff)", "depth (param)",
                  "ffn per-layer widths"}
BLUE, BLUE_L, INK, INK2, SURF = "#2a78d6", "#86b6ef", "#0b0b0b", "#52514e", "#fcfcfb"


def charts(att):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    # 1. greedy curve
    gr = att["greedy"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = list(range(len(gr) + 1))
    ys = [att["n_frontier"]] + [n for _, n in gr]
    ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=4)
    for x, (k, n) in zip(xs[1:7], gr[:6]):
        ax.annotate(k, (x, n), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.axhline(att["n_frontier_no_typewalls"], color=INK2, lw=1, ls=":")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.text(0, att["n_frontier_no_typewalls"] + 4, f"capacity + paradigm only: {att['n_frontier_no_typewalls']}",
            ha="left", fontsize=7.5, color=INK2)
    ax.set_xlabel("wall kinds granted (greedy order)")
    ax.set_ylabel("payable frontier size")
    ax.set_title("Granting wall kinds one at a time, largest drop first", loc="left", color=INK)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/greedy_curve.png", dpi=150)
    plt.close(fig)
    # 2. leave-one-out per kind (models leaving)
    ck = sorted(((k, len(v["leave"])) for k, v in att["cf_kind"].items()), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([k for k, _ in ck], [v for _, v in ck], color=BLUE, height=0.6)
    for i, (_, v) in enumerate(ck):
        ax.text(v + 0.4, i, str(v), va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("frontier models that leave if only this wall kind is granted")
    ax.set_title("Single-kind counterfactual (everything else unchanged)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kind_leave_one_out.png", dpi=150)
    plt.close(fig)
    # 3. multiplicity: minimum number of failing items per frontier model
    mc = collections.Counter(m["mlen"] for m in att["models"].values())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ks = sorted(mc)
    ax.bar([str(k) if k else "0 (no capacity\ndominator)" for k in ks], [mc[k] for k in ks], color=BLUE, width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, mc[k] + 1, str(mc[k]), ha="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("walls the closest would-be dominator still fails (minimum over hosts)")
    ax.set_ylabel("frontier models")
    ax.set_title("How many walls stand between each frontier model and its closest dominator", loc="left",
                 color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/wall_multiplicity.png", dpi=150)
    plt.close(fig)


def _t(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_md(att, g, pres, levers=None):
    M = att["models"]
    n = att["n_frontier"]
    L = []
    add = L.append
    add("# Frontier study: does the payable headline need to be that high?")
    add("")
    add(f"Snapshot relation: round 3 at f88b44a (branch frontier-study), payable mode, {att['n_shapes']} shapes, "
        f"**{n} frontier classes**. Generated by `python -m graph.frontier_study`, `graph.frontier_study_growth`, "
        "`graph.frontier_study_report` (data in dist/frontier_study/*.json).")
    add("")
    nocap = sum(1 for m in M.values() if m["n_capdom"] == 0)
    mc = collections.Counter(m["mlen"] for m in M.values())
    single = sum(1 for m in M.values() if m["mlen"] == 1)
    single_tw = sum(1 for m in M.values() if [x for x in m["singles"] if x not in CAPACITY_ITEMS])
    add("## 1. Short answer")
    add("")
    add(f"- With every type wall granted (capacity and the AR <= diffusion paradigm kept) the frontier is "
        f"**{att['n_frontier_no_typewalls']}**. So about {n - att['n_frontier_no_typewalls']} of the {n} are kept "
        f"by type walls, and {att['n_frontier_no_typewalls']} by size alone.")
    add(f"- {nocap} frontier classes have **no capacity dominator**: no host is at least as big on every axis, so no "
        "wall change can remove them.")
    add(f"- {single} classes are one refined check away from some capacity dominator. For {single - single_tw} of "
        f"them that check is a capacity one (a wall's charge or a per-shape fit), so {single_tw} are one TYPE wall away. "
        f"{sum(v for k, v in mc.items() if k >= 2)} classes are walled at least twice against every dominator "
        f"(distribution below), so no single wall change moves them.")
    big = sorted(att["cf_kind"].items(), key=lambda kv: -len(kv[1]["leave"]))[:4]
    add("- No wall kind dominates. Granting one kind alone removes at most "
        + ", ".join(f"{len(v['leave'])} ({k})" for k, v in big)
        + ". The headline is a long tail of independent walls. Many are features that almost nothing else has: "
          f"{att['rare_feature_classes']['1']} frontier classes carry a feature no other snapshot shape has, and "
          f"{att['rare_feature_classes']['5']} carry one at most 5 shapes have (a hyper-connection variant, an n-gram "
          "memory, a loop norm, a router form). Nothing can host such a feature, because no bigger model has it.")
    add(f"- Round 3's growth (248 -> {g['n_on']}) is {len(g['new'])} new frontier classes minus {len(g['gone'])} "
        "gone. Every new one comes from a classification or relation correction that removed over-credit (section "
        "4). Most are small or exotic models whose code has a feature the pass-5 reading missed.")
    add("")
    add("![greedy](frontier_study/greedy_curve.png)")
    add("")
    add("## 2. Attribution")
    add("")
    add("Method: every refusing check of emul_matrix(explain=...) is refined into **items**: a slot check into its "
        "value pair, the feature check into each offending feature, the mixer check into its slot pairs, and the MoE "
        "routing check into **every** failing moe_why condition, not only the first. The refinement is exact: a pair "
        "is refused iff it has at least one item. This is asserted in `graph.frontier_study.build`. Each item has a "
        "kind. For a frontier class i, the *would-be dominators* are its strict capacity dominators (capacity mode). "
        "Each one's failing items is a *blocker set*. Granting a set S of items makes every pair whose items lie "
        "in S pass. The counterfactual frontier is n_frontier of that relation.")
    add("")
    add("![multiplicity](frontier_study/wall_multiplicity.png)")
    add("")
    add("Minimum blocker-set size over each class's capacity dominators:")
    add("")
    L.extend(_t([(k if k else "0 (no capacity dominator)", mc[k]) for k in sorted(mc)], ["walls", "frontier classes"]))
    add("")
    add("### 2.1 Wall kinds: binding counts and counterfactuals")
    add("")
    add("*binds*: classes whose binding (minimum-cardinality) kind set contains the kind. *in a minimal set*: classes "
        "where the kind appears in some minimal blocker kind set. *grant this kind*: the frontier with that kind's "
        "items all granted, and how many current classes acquire a strict dominator. *only this wall*: every "
        f"other type wall granted (from the {att['n_frontier_no_typewalls']} floor), this kind kept.")
    add("")
    rows = []
    for k, v in sorted(att["cf_kind"].items(), key=lambda kv: -len(kv[1]["leave"])):
        rows.append((k, v["binds"], v["in_minimal"], v["n_frontier"], len(v["leave"]),
                     v["only_this_wall"] if v["only_this_wall"] is not None else "-"))
    L.extend(_t(rows, ["kind", "binds", "in a minimal set", "grant this kind: frontier", "leave", "only this wall"]))
    add("")
    add("![kinds](frontier_study/kind_leave_one_out.png)")
    add("")
    add("Greedy order (grant the kind that lowers the frontier most, repeatedly): "
        + " -> ".join(f"{k} {nn}" for k, nn in att["greedy"]) + ".")
    add("")
    add("### 2.2 Single items with the largest counterfactual")
    add("")
    add("*single*: classes for which this item is the only refusal against some capacity dominator. *leave*: the "
        "classes that leave when this item alone is granted.")
    add("")
    it = sorted(att["cf_item"].items(), key=lambda kv: (-len(kv[1]["leave"]), -kv[1]["in_min"]))
    rows = [(f"`{k}`", v["kind"], v["in_min"], v["single"], len(v["leave"]),
             ", ".join(x.split("/")[-1] for x in v["leave"][:6]) + (" ..." if len(v["leave"]) > 6 else ""))
            for k, v in it if len(v["leave"]) >= 2]
    L.extend(_t(rows, ["item", "kind", "in a minimal set", "single", "leave", "who"]))
    add("")
    add("### 2.3 The frontier tree (architecture family x binding wall kinds)")
    add("")
    add("Each frontier class is listed under its config model_type (the first member's) and its binding kind set: the "
        "minimum-cardinality set of wall kinds whose removal lets its widest-covering capacity dominator strictly "
        "dominate it, or *capacity-maximal*.")
    add("")
    tree = collections.defaultdict(lambda: collections.defaultdict(list))
    for nm, m in M.items():
        tree[pres["family_of"][nm]][" + ".join(m["binding"])].append(nm)
    for famk in sorted(tree, key=lambda f: -sum(len(v) for v in tree[f].values())):
        tot = sum(len(v) for v in tree[famk].values())
        add(f"- **{famk}** ({tot})")
        for b, names in sorted(tree[famk].items(), key=lambda kv: -len(kv[1])):
            add(f"  - {b}: " + ", ".join(f"`{x.split('/')[-1]}`" for x in sorted(names)))
    add("")
    bind = collections.Counter(" + ".join(m["binding"]) for m in M.values())
    add("Binding kind sets, aggregated:")
    add("")
    L.extend(_t(bind.most_common(30), ["binding kind set", "classes"]))
    add("")
    add("## 3. Legitimate reductions (constructions)")
    add("")
    if levers:
        L.extend(levers)
    else:
        add("(in progress)")
    add("")
    add("## 4. Where 248 -> " + str(g["n_on"]) + " came from")
    add("")
    add(f"Every round-3 change is keyed (R3_OFF, {len(g['per_key'])} keys). With all keys off the relation is pass "
        f"5's on today's snapshot: {g['n_off']} ({g['shapes_off']} shapes); all on: {g['n_on']} ({g['shapes_on']} "
        "shapes). A class is *new* if none of its members was on the pass-5 frontier. *Needs*: keys whose leave-"
        "one-out (that key off, the rest on) takes it off the frontier. *Alone*: keys that put it on the frontier when "
        "switched on by themselves from pass 5. Direction and finding come from verify/r3/specs/<key>.md.")
    add("")
    keyrows = collections.defaultdict(list)
    for r in g["new"]:
        ks = r["necessary"] or r["sufficient_alone"]
        keyrows[", ".join(ks) or "(interaction)"].append(r["name"].split("/")[-1])
    rows = []
    for ks, names in sorted(keyrows.items(), key=lambda kv: -len(kv[1])):
        k0 = ks.split(", ")[0]
        direction = spec_summary(k0)[0]
        what = "; ".join(f"{k}: {spec_summary(k)[1]}" if ", " in ks else spec_summary(k)[1] for k in ks.split(", "))
        rows.append((ks, len(names), ", ".join(names), direction, what.replace("|", "/")))
    L.extend(_t(rows, ["key(s)", "new", "classes", "direction", "what the pass-5 reading had wrong (spec)"]))
    add("")
    add("Classes that left the frontier in round 3:")
    add("")
    L.extend(_t([(r["name"], ", ".join(r["necessary"] or r["sufficient_alone"]),
                  "; ".join(spec_summary(k)[1] for k in (r["necessary"] or r["sufficient_alone"])))
                 for r in g["gone"]], ["class", "key(s)", "change"]))
    add("")
    add("Per-key leave-one-out (headline change with the key off) and leave-one-in (change from 248 with only the key "
        "on), keys that move either:")
    add("")
    L.extend(_t([(k, f"{v['loo']:+d}", f"{v['loi']:+d}") for k, v in sorted(g["per_key"].items(),
                                                                          key=lambda kv: kv[1]["loo"])
                 if v["loo"] or v["loi"]], ["key", "LOO (off)", "LOI (alone on)"]))
    add("")
    add("## 5. Presentation alternatives (rules unchanged)")
    add("")
    p = pres
    add(f"- **Distinct families**: the {n} classes span **{p['n_arch_families']} config model_types** and "
        f"**{p['n_labs']} Hub organisations**. Largest families: "
        + ", ".join(f"{k} {v}" for k, v in p["families"][:12]) + ".")
    add(f"- **Downloads**: {p['k50']} frontier classes carry half of the frontier's downloads and {p['k90']} carry "
        "90 %. Top 10: " + ", ".join(f"`{k.split('/')[-1]}` ({v:,})" for k, v in p["top_by_downloads"][:10]) + ".")
    pv = p["provenance"]
    add(f"- **Provenance** (name heuristics, graph/frontier_study_report.py): {pv.get('original', 0)} classes have an "
        f"original-release member, {pv.get('conversion', 0)} are only quantised / re-packaged conversions (their "
        f"shape differs from the original's in a read config value), {pv.get('derived shape', 0)} are derived shapes "
        "(pruned, merged, upcycled). Conversions and derived shapes are listed in growth_presentation.json.")
    it = sorted(att["cf_item"].items(), key=lambda kv: -len(kv[1]["leave"]))
    add(f"- **Frontier modulo one feature**: ignoring any single refined wall item removes at most "
        f"{len(it[0][1]['leave'])} classes ({it[0][0]}); the top five are "
        + ", ".join(f"{k} {len(v['leave'])}" for k, v in it[:5])
        + f". Ignoring a whole kind removes at most {max(len(v['leave']) for v in att['cf_kind'].values())}. "
          f"Only {sum(1 for _, v in it if len(v['leave']) >= 3)} items move 3 or more classes. Tables 2.1 and 2.2 "
          "have the full lists.")
    add(f"- **Fine-tunes and conversions**: {pv.get('conversion', 0) + pv.get('derived shape', 0)} of {n} classes "
        f"({100 * (pv.get('conversion', 0) + pv.get('derived shape', 0)) // n} %) have no original-release member. The "
        f"frontier is not a few base shapes in many copies: its {n} classes span {p['n_arch_families']} model_types.")
    return "\n".join(L) + "\n"


def levers_section(att):
    if not os.path.exists(f"{OUT}/levers.json"):
        return None
    lv = json.load(open(f"{OUT}/levers.json"))
    cs = json.load(open(f"{OUT}/constructions.json")) if os.path.exists(f"{OUT}/constructions.json") else {}
    L = ["This branch took the wall kinds bias, output / shared-expert gate, qk-norm / v-norm, attention layout, MLA, "
         "activation, tied / factorized embeddings, other attention features, sink / softcap, parallel block and FFN "
         "form. Positions, MoE, mixer / loops and residual / norm placement went to four parallel branches.",
         "",
         "**Upper bound.** Granting ALL of this branch's kinds at once takes the frontier from "
         f"{att['n_frontier']} to " + str(att["mine_all_granted"]) + ". Biases are the largest single kind "
         f"({len(att['cf_kind']['bias']['leave'])} classes), and every bias wall on a bias-free RMSNorm host is a real "
         "one (relation notes B).",
         "",
         "Constructions (block/frontier_constructions.py; graph/frontier_study_levers.py applies them to the relation "
         "without editing it):",
         ""]
    rows = []
    for tag in ("gate_softplus_off", "gate_bias_silu_off", "gate_nests", "tied_host", "mine_all"):
        v = lv[tag]
        rows.append((tag, v["note"], f"+{v['new_edges']}", f"{lv['base']['frontier']} -> {v['frontier']}",
                     len(v["leave"]), v["violations"]))
    L += _t(rows, ["lever", "construction", "payable edges", "frontier", "leave", "transitivity violations "
                   f"(base {lv['base']['violations']})"])
    if cs:
        L += ["", "Proofs (reference decoder, fp64) and negative controls:", ""]
        L += _t([(k, v) for k, v in cs.get("ref", {}).items()] + [("NEGATIVE: " + k, v) for k, v in
                                                                  cs.get("negative", {}).items()],
                ["check", "target vs host"])
        if cs.get("real"):
            L += ["", "Real checkpoints (fp32; gate levers on the checkpoint's own attention with the gate added, the "
                  "tied host a stock LlamaForCausalLM of width 2 d + 2 and L + 1 layers built from the untied "
                  "checkpoint; 8 random-id prompts, 32 greedy tokens):", ""]
            L += _t(list(cs["real"].items()), ["pair", "target vs host"])
    L += ["", "**Result.** The constructions are exact and add 3,987 payable edges without new transitivity violations, "
          "but they move no frontier class. Every class they newly reach still has another wall against the same "
          "host: the softplus hosts are Laguna MoEs that fail the top-k multiple, and the tied hosts lack 2x width. "
          "The remaining walls of these kinds are genuine (bias oddness, per-head qk-norm, the MLA q-latent rank, tied "
          "rank)."]
    return L


def main():
    att = json.load(open(f"{OUT}/attribution.json"))
    g = growth()
    dl, mt = load_meta()
    pres = presentation(att, dl, mt)
    json.dump(dict(growth=g, presentation=pres), open(f"{OUT}/growth_presentation.json", "w"), indent=1)
    charts(att)
    lev = levers_section(att)
    open("dist/frontier_study.md", "w").write(write_md(att, g, pres, lev))
    print(f"growth {g['n_off']} -> {g['n_on']}: {len(g['new'])} new classes, {len(g['gone'])} gone")
    print(f"families {pres['n_arch_families']}, labs {pres['n_labs']}, provenance {pres['provenance']}, "
          f"k50 {pres['k50']} k90 {pres['k90']}")


if __name__ == "__main__":
    main()
