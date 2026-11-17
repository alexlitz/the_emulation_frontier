"""Framework support for every model in the zoo: NATIVE (the framework's own registry, graph.framework_registries,
read as text at a pinned version) and BACKDOOR (a natively supported architecture in that framework that EMULATES
the model under the headline relation, so the model's weights can be rebuilt as that host and run there).

  native[fw]    True / False per repo: its architectures[0] (registries keyed by class name: vLLM, SGLang,
                llama.cpp) or its model_type (keyed by type: mlx-lm, TGI, ExecuTorch, ONNX, transformers.js), either
                for transformers. None = the config names neither.
  backdoor[fw]  per SHAPE (graph.model_graph._dedup_points order = the zoo's nodes): the shapes j (j = i included: a natively supported member of the same shape) with
                emul_matrix(P, mode="payable")[i, j] (j emulates i) that have a member natively supported in fw,
                exact edges (mode="exact") first, then payable ones (limit / floor / regime edges), each by host
                params. The host repo is the most-downloaded natively supported member of the host shape.
                width_x = host hidden / target hidden, depth_x = host unrolled depth / target's,
                params_x = host params / target params (params from graph/data/arch_metrics.json when present,
                else graph.figures.real_prm).

Run:  python -m graph.frameworks      # writes graph/data/frameworks.json (needs framework_registries.json)
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np

from graph.framework_registries import FRAMEWORKS, load as load_registries
from graph.model_graph import ZOO_SNAPSHOT, _dedup_points, emul_matrix, iter_json_configs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "frameworks.json")
METRICS = os.path.join(HERE, "data", "arch_metrics.json")
CONFIGS = os.path.join(HERE, "hub_10k.json")
TOP_HOSTS = 3                   # backdoor hosts kept per (shape, framework)
SCHEMA_VERSION = 1


def arch_of(cfg):
    a = cfg.get("architectures") or []
    return a[0] if a and isinstance(a[0], str) else None


def config_fits_arch(cfg):
    """(ok, reason): a config whose architectures[0] names a stock transformers class but whose model_type is not a
    stock type must still be readable by that class's config for a registry keyed by class name to load it. The
    stock config class (transformers' own code) reads the raw dict, and its width / depth / heads must be the ones
    the config states (graph.model_graph.dims_of), not the class defaults. NeuralAI-Nae1 names LlamaForCausalLM with
    a 'nae1' config of n_embd / n_layer / n_head: LlamaConfig reads hidden_size 4096 for its 768, so no framework
    that loads LlamaForCausalLM reads it. Kimi-K2 (DeepseekV3ForCausalLM, model_type kimi_k2) and LongCat-Flash
    (num_layers, which LongcatFlashConfig takes and reads as 2 x num_layers sub-layers) fit."""
    arch, mt = arch_of(cfg), cfg.get("model_type")
    if not arch:
        return True, None
    try:
        import logging
        import transformers
        from transformers import CONFIG_MAPPING
        from graph.model_graph import dims_of
    except Exception:          # pragma: no cover
        return True, None
    if mt in CONFIG_MAPPING:
        return True, None
    cc = getattr(getattr(transformers, arch, None), "config_class", None)
    if cc is None:
        return True, None
    lg = logging.getLogger("transformers")
    lvl = lg.level
    lg.setLevel(logging.ERROR)
    try:
        c = cc.from_dict({k: v for k, v in cfg.items() if k not in ("model_type", "auto_map")})
        t = c.get_text_config(decoder=True) if hasattr(c, "get_text_config") else c
        dm = dims_of(cfg)
        got = {"hidden": getattr(t, "hidden_size", None), "layers": getattr(t, "num_hidden_layers", None),
               "heads": getattr(t, "num_attention_heads", None)}
    except Exception:          # the class rejects the dict: say nothing either way
        return True, None
    finally:
        lg.setLevel(lvl)
    try:
        import inspect
        takes = set(inspect.signature(cc.__init__).parameters) | set(getattr(cc, "attribute_map", {}) or {})
    except (TypeError, ValueError):          # pragma: no cover
        takes = set()
    if any("layer" in k and k in takes for k in cfg):
        got.pop("layers")                    # the class reads its own depth key (LongCat: 2 x num_layers sub-layers)
    bad = [k for k, v in got.items() if isinstance(v, int) and dm.get(k) and v != dm[k]]
    if bad:
        return False, (f"architectures names {arch}, but its config class reads the {mt} config as "
                       + ", ".join(f"{k} {got[k]} (the config's {dm[k]})" for k in bad))
    return True, None


def native_support(cfg, reg):
    """{framework: True / False / None} for one raw config (ExecuTorch here by model type; build() narrows it to
    the models with an export recipe). A class-name hit whose config the class cannot read (config_fits_arch) is
    not native."""
    mt, arch = cfg.get("model_type"), arch_of(cfg)
    fits = config_fits_arch(cfg)[0]
    out = {}
    for fw in FRAMEWORKS:
        r = reg[fw]
        archs, types = set(r["architectures"]), set(r["model_types"])
        if fw == "transformers":          # a stock class: the model_type is registered, or the class name is
            hit = (mt in types) or (arch in archs and fits)
            out[fw] = bool(hit) if (mt or arch) else None
        elif r["key"] == "architecture":
            out[fw] = (arch in archs and fits) if arch else None
        else:
            out[fw] = (mt in types) if mt else None
    return out


def transformers_tier(cfg, reg, native):
    """{framework: True} where the framework runs the model through its Transformers backend / fallback but has no
    native implementation: vLLM's _TRANSFORMERS_SUPPORTED_MODELS (by architecture) and TGI's transformers fallback
    (by model_type, transformers' causal-LM mapping at the version TGI pins). Not counted as native."""
    mt, arch = cfg.get("model_type"), arch_of(cfg)
    out = {}
    if not native.get("vllm") and arch and arch in set(reg["vllm"].get("transformers_backend") or ()):
        out["vllm"] = True
    if not native.get("tgi") and mt and mt in set(reg["tgi"].get("transformers_fallback") or ()):
        out["tgi"] = True
    return out


def export_dims(cfg):
    """(hidden, layers, heads, kv_heads) the way ExecuTorch's export configs state them (graph.framework_registries
    ._dims_of_export), for matching a repo to a recipe."""
    from graph.framework_registries import _dims_of_export
    try:
        d = _dims_of_export(cfg)
    except (KeyError, TypeError, ValueError):
        return None
    return tuple(d[k] for k in ("hidden", "layers", "heads", "kv_heads")) if d else None


def _family(mt):
    """A model type's family for host ranking: the text sub-type and the wrapper count as one (gemma3 /
    gemma3_text, qwen2_5_vl / qwen2_5_vl_text)."""
    mt = str(mt or "").lower()
    for suf in ("_text", "_vl", "_moe"):
        if mt.endswith(suf):
            mt = mt[: -len(suf)]
    return mt


# a repo published as an embedding / reranker / reward / classifier model (pipeline tag), or whose class has no LM
# head (architectures). Tags generative LLMs also carry (question-answering, reinforcement-learning, robotics on
# Llama / Qwen fine-tunes in the snapshot) are not used.
NON_GENERATIVE_TAGS = frozenset({"feature-extraction", "sentence-similarity", "text-classification",
                                 "token-classification", "fill-mask", "zero-shot-classification", "text-ranking",
                                 "time-series-forecasting"})
NON_GENERATIVE_ARCH = ("ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering",
                       "ForMaskedLM", "RewardModel", "ForRewardModel", "ForReward", "EmbeddingModel", "ForEmbedding",
                       "EncoderModel", "PairRM", "ForPrediction", "ForSegmentation")
IMPRACTICAL_PARAMS_X = 10              # a host over 10x the target's params is listed, flagged, never the top pick
PIPELINE_TAGS = os.path.join(HERE, "data", "pipeline_tags.json")


def generative(cfg, tag):
    """(bool, reason): can this repo stand in as a text-generation host? False for embedding / reward /
    classifier models, read from the Hub pipeline tag and the config's architectures."""
    if tag in NON_GENERATIVE_TAGS:
        return False, f"pipeline tag {tag}"
    for a in cfg.get("architectures") or []:
        if isinstance(a, str) and any(m in a for m in NON_GENERATIVE_ARCH):
            return False, f"architecture {a}"
    return True, None


SPEECH_TAGS = frozenset({"automatic-speech-recognition", "text-to-speech", "text-to-audio", "audio-to-audio",
                         "audio-text-to-text", "audio-classification", "voice-activity-detection"})
SPEECH_ARCH = ("ASR", "Speech", "Moshi", "Audio", "TTS", "Whisper")
VL_ARCH = ("ConditionalGeneration", "VL", "Vision", "Llava", "LLaVA", "MultiModality", "Omni", "ImageText")
DRAFTER_ARCH = ("Eagle", "EAGLE", "Medusa", "Draft", "Assistant")
DRAFTER_NAME = ("-eagle", "eagle3", "-medusa", "-assistant", "-drafter", "dflash", "-mtpv", "speculator")
TRIAGE = os.path.join(HERE, "data", "params_triage.json")


def host_flags(repo, cfg, tag, anomalies=()):
    """(excluded, flags) of a repo as a backdoor host EXAMPLE. Excluded: a drafter (EAGLE / Medusa / assistant /
    MTP-only heads, not a standalone LM) or a checkpoint anomaly (params_triage.json bucket B: NPU binaries, partial
    uploads). Flags, shown with the host and used to rank it down: 'speech / audio model' (an ASR / TTS / speech
    checkpoint: its class or tag), 'vision-language wrapper' (the class wraps a vision or audio tower), 'quantised
    checkpoint' (the example's weights are quantised)."""
    from graph.arch_metrics import is_quantised
    archs = " ".join(a for a in (cfg.get("architectures") or []) if isinstance(a, str))
    low = repo.lower()
    if any(m in archs for m in DRAFTER_ARCH) or any(m in low for m in DRAFTER_NAME):
        return True, ["drafter"]
    if repo in anomalies:
        return True, ["checkpoint anomaly"]
    flags = []
    if tag in SPEECH_TAGS or any(m in archs for m in SPEECH_ARCH) or cfg.get("audio_encoder_config"):
        flags.append("speech / audio model")
    if any(m in archs for m in VL_ARCH) or isinstance(cfg.get("vision_config"), dict) or \
            isinstance(cfg.get("audio_config"), dict):
        flags.append("vision-language wrapper")
    if is_quantised(repo, cfg) or isinstance(cfg.get("quantization_config"), dict):
        flags.append("quantised checkpoint")
    return False, flags


def _params_lookup():
    try:
        with open(METRICS) as f:
            repos = json.load(f)["repos"]
        return lambda repo, p: (repos.get(repo, {}).get("params", {}).get("total") or None)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        from graph.figures import prm, real_prm

        def size(repo, p):                 # the zoo page's placement size (zoo_data): a published count far
            s = real_prm(p)                # below the config's own estimate is a bad safetensors total
            return float(prm(p["dims"]) if s < prm(p["dims"]) / 20 else s)
        return size


def build(models=None, configs=None, reg=None, top=TOP_HOSTS):
    from graph.frontier import load
    models = models if models is not None else load()
    configs = configs if configs is not None else dict(iter_json_configs(CONFIGS))
    reg = reg or load_registries()
    fws = reg["frameworks"]
    P = _dedup_points(models)
    n = len(P)
    ok = emul_matrix(P, mode="payable")                 # ok[i, j]: j emulates i
    exact = emul_matrix(P, mode="exact")
    rank = {m["name"]: k for k, m in enumerate(models)}  # snapshot order = downloads
    shape_of = {nm: i for i, p in enumerate(P) for nm in p["members"]}
    native = {nm: native_support(configs.get(nm, {}), fws) for nm in shape_of}
    # ExecuTorch per MODEL: the model type is registered AND the shape is one an export recipe builds (its dims
    # match an examples/models config) or holds a repo the recipes / optimum-executorch tests name ("X and its
    # variants" = X's shape)
    et = fws["executorch"]
    et_dims = {(_family(c["model_type"]), (c["hidden"], c["layers"], c["heads"], c["kv_heads"])): c["file"]
               for c in et.get("export_configs") or []}
    et_ids = {x["id"]: x["source"] for x in et.get("model_ids") or []}
    et_why = {}
    for i, p in enumerate(P):
        named = next((m for m in p["members"] if m in et_ids), None)
        for m in p["members"]:
            if not native[m]["executorch"]:
                continue
            dm = export_dims(configs.get(m, {}))
            hit = et_dims.get((_family(configs.get(m, {}).get("model_type")), dm)) if dm else None
            if named:
                et_why[m] = f"same shape as {named} ({et_ids[named]})"
            elif hit:
                et_why[m] = f"export recipe {hit}"
            else:
                native[m]["executorch"] = False
                et_why[m] = "model type registered, but no export recipe for this model (size / variant)"
    via_tf = {nm: transformers_tier(configs.get(nm, {}), fws, native[nm]) for nm in shape_of}
    tags = {}
    if os.path.isfile(PIPELINE_TAGS):
        with open(PIPELINE_TAGS) as f:
            tags = {k: (v or {}).get("pipeline_tag") for k, v in (json.load(f).get("repos") or {}).items()}
    gen = {nm: generative(configs.get(nm, {}), tags.get(nm)) for nm in shape_of}
    anomalies = set()
    if os.path.isfile(TRIAGE):
        with open(TRIAGE) as f:
            anomalies = {r for r, x in (json.load(f).get("repos") or {}).items() if x.get("bucket") == "B"}
    hfl = {nm: host_flags(nm, configs.get(nm, {}), tags.get(nm), anomalies) for nm in shape_of}
    prm = _params_lookup()
    # per shape and framework: the host repo, the natively supported GENERATIVE member that is not a drafter or a
    # checkpoint anomaly, with the fewest flags (a full-precision text-only member before a quantised or wrapped
    # one), then the most downloaded
    host_repo = [{fw: min((m for m in p["members"] if native[m][fw] and gen[m][0] and not hfl[m][0]),
                          key=lambda m: (len(hfl[m][1]), rank.get(m, 1 << 30)), default=None) for fw in FRAMEWORKS}
                 for p in P]
    pop = [min(p["members"], key=lambda m: rank.get(m, 1 << 30)) for p in P]
    shape_params = [prm(pop[i], P[i]) for i in range(n)]
    shapes = []
    for i, p in enumerate(P):
        d = p["dims"]
        bd = {}
        fam = {_family(configs.get(m, {}).get("model_type")) for m in p["members"]}
        for fw in FRAMEWORKS:
            if all(native[m][fw] for m in p["members"]):
                continue                                   # every member native: no backdoor needed
            cands = []
            for j in np.where(ok[i])[0]:
                j = int(j)
                h = host_repo[j][fw]
                if h is None:          # j == i: a natively supported member of the same shape (mutually emulable)
                    continue
                dh = P[j]["dims"]
                hp = prm(h, P[j])
                tp = shape_params[i]
                px = round(hp / tp, 4) if hp and tp else None
                hmt = configs.get(h, {}).get("model_type")
                cands.append({"host": h, "host_arch": arch_of(configs.get(h, {})),
                              "host_model_type": hmt, "host_shape": j,
                              "edge": "exact" if exact[i, j] else "payable", "exact": bool(exact[i, j]),
                              "width_x": round(dh["hidden"] / d["hidden"], 4) if d["hidden"] else None,
                              "depth_x": round(dh["eff_layers"] / d["eff_layers"], 4) if d["eff_layers"] else None,
                              "host_params": hp, "params_x": px, "same_family": _family(hmt) in fam,
                              "impractical": bool(px is not None and px > IMPRACTICAL_PARAMS_X),
                              "host_flags": hfl[h][1]})
            # practical text-LM hosts first (a speech / audio checkpoint last), exact edges first, then the target's
            # own family, then a plain host before a vision-language wrapper or a quantised-only example, then the
            # fewest params
            cands.sort(key=lambda c: (c["impractical"], "speech / audio model" in c["host_flags"], not c["exact"],
                                      not c["same_family"], "vision-language wrapper" in c["host_flags"],
                                      "quantised checkpoint" in c["host_flags"],
                                      c["params_x"] if c["params_x"] is not None else 1e18, c["host"]))
            bd[fw] = {"n_hosts": len(cands), "n_exact": sum(c["exact"] for c in cands),
                      "n_practical": sum(not c["impractical"] for c in cands), "hosts": cands[:top]}
        shapes.append({"rep": p["members"][0], "popular": pop[i], "n_members": len(p["members"]), "backdoor": bd})
    repos = {}
    for nm, i in shape_of.items():
        c = configs.get(nm, {})
        repos[nm] = {"model_type": c.get("model_type"), "arch": arch_of(c), "shape": i, "native": native[nm]}
        if via_tf[nm]:
            repos[nm]["via_transformers"] = via_tf[nm]
        if nm in et_why:
            repos[nm]["executorch_why"] = et_why[nm]
        if not gen[nm][0]:
            repos[nm]["not_generative"] = gen[nm][1]
        if hfl[nm][0]:
            repos[nm]["not_a_host"] = hfl[nm][1][0]
        fit = config_fits_arch(c)
        if not fit[0]:
            repos[nm]["config_mismatch"] = fit[1]
    counts = {}
    for fw in FRAMEWORKS:
        nat = sum(1 for r in repos.values() if r["native"][fw])
        unk = sum(1 for r in repos.values() if r["native"][fw] is None)
        miss = [r for r in repos.values() if r["native"][fw] is False or r["native"][fw] is None]
        bd = [shapes[r["shape"]]["backdoor"].get(fw, {}) for r in miss]
        counts[fw] = {"repos": len(repos), "native": nat, "missing": len(miss), "unknown": unk,
                      "via_transformers": sum(1 for r in repos.values() if (r.get("via_transformers") or {}).get(fw)),
                      "backdoor_exact": sum(1 for b in bd if b.get("n_exact")),
                      "backdoor_payable": sum(1 for b in bd if b.get("n_hosts") and not b.get("n_exact")),
                      "no_backdoor": sum(1 for b in bd if not b.get("n_hosts")),
                      "shapes_native": sum(1 for s in range(n) if host_repo[s][fw]),
                      "shapes": n}
    meta = {fw: {k: fws[fw][k] for k in ("repo", "version", "fetched", "urls", "key", "source_note")}
            | {"n_architectures": len(fws[fw]["architectures"]), "n_model_types": len(fws[fw]["model_types"])}
            for fw in FRAMEWORKS}
    return {"schema_version": SCHEMA_VERSION, "generated": datetime.date.today().isoformat(),
            "snapshot": ZOO_SNAPSHOT, "frameworks": meta, "counts": counts, "shapes": shapes, "repos": repos,
            "conventions": {
                "native": "registry hit at the pinned version: architectures[0] for vllm / sglang / llama.cpp, "
                          "model_type for mlx-lm / tgi / onnx / transformers.js, either for transformers; "
                          "executorch per model: the type is registered and the shape matches an export recipe's "
                          "dims or holds a repo the recipes / optimum-executorch tests name (repos[r].executorch_why)",
                "via_transformers": "repos[r].via_transformers[fw] = true: the framework runs the model through its "
                                    "Transformers backend (vLLM _TRANSFORMERS_SUPPORTED_MODELS) or fallback (TGI, a "
                                    "model_type in transformers' causal-LM mapping at TGI's pin); a separate tier, "
                                    "not counted as native",
                "backdoor": "shapes[i].backdoor[fw].hosts: shapes j that emulate shape i under "
                            "emul_matrix(mode='payable') with a natively supported, generative member (embedding / "
                            "reward / classifier repos excluded by pipeline tag and architectures, drafters and "
                            "checkpoint anomalies excluded: repos[r].not_a_host); the host repo of a shape is its "
                            "member with the fewest host_flags (speech / audio model, vision-language wrapper, "
                            "quantised checkpoint), then the most downloaded; practical hosts "
                            f"first (params_x <= {IMPRACTICAL_PARAMS_X}; larger ones carry impractical: true), a "
                            "speech / audio host last, exact edges first (mode='exact'), then the target's own model "
                            "family, then a host without the wrapper / quantised flags, then by params_x",
                "repo_backdoor": "a repo whose native[fw] is not true uses shapes[repos[repo].shape].backdoor[fw]"}}


def main():
    data = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    for fw, c in data["counts"].items():
        print(f"{fw:16} native {c['native']:5}  missing {c['missing']:5}  backdoor exact {c['backdoor_exact']:5} "
              f"payable {c['backdoor_payable']:5}  none {c['no_backdoor']:5}")


if __name__ == "__main__":
    main()
