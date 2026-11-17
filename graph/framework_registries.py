"""Native model support of each inference framework, read as TEXT from its source registry at a pinned
version on GitHub. Nothing is installed, imported or executed: each file is downloaded and parsed with regex.

  transformers     src/transformers/models/auto/{configuration,modeling}_auto.py: CONFIG_MAPPING_NAMES and the
                   causal-LM / image-text-to-text / multimodal-LM mappings (model_type -> class)
  vllm             vllm/model_executor/models/registry.py: the architecture keys of the text-generation and
                   multimodal dicts; the architectures vLLM routes to its Transformers backend
                   (_TRANSFORMERS_SUPPORTED_MODELS, e.g. "PhiForCausalLM": ("transformers", "TransformersForCausalLM"))
                   are recorded separately as `transformers_backend`: supported, but not native
  sglang           python/sglang/srt/models/*.py: `EntryClass = X` / `[X, Y]` class names
  llama.cpp        conversion/*.py (convert_hf_to_gguf.py's package): @ModelBase.register("Arch", ...) on a
                   TEXT model class (MmprojModel subclasses only convert a vision tower), and src/llama-arch.cpp
                   runtime arch strings
  mlx-lm           mlx_lm/models/<model_type>.py, plus MODEL_REMAPPING (model_type aliases) in mlx_lm/utils.py
  tgi              server/text_generation_server/models/__init__.py: the ModelType enum "type" strings; its get_model()
                   falls back to transformers (TransformersFlashCausalLM / CausalLM) for a model_type in
                   transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, read at the version TGI pins
                   (server/requirements_cuda.txt): `transformers_fallback`, supported but not native
  executorch       examples/models/<dir> LLM families and examples/models/llama/export_llama_lib.py, plus
                   optimum-executorch's tests/models/test_modeling_<type>.py (HF export path), as model_types
  onnx             optimum-onnx optimum/exporters/onnx/model_configs.py: @register_tasks_manager_onnx("type", ...)
                   registrations that include text-generation
  transformers.js  packages/transformers/src/models/registry.js: MODEL_FOR_CAUSAL_LM / IMAGE_TEXT_TO_TEXT /
                   MULTIMODALITY mapping names (model_type -> class)

Run:  python -m graph.framework_registries     # fetch and write graph/data/framework_registries.json
"""
from __future__ import annotations

import datetime
import json
import os
import re
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "framework_registries.json")
CACHE = os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "framework_registry_cache")

# pinned: the latest release tag on 2026-11-13, or a main commit where the last release is over a year stale
PINS = {
    "transformers": ("huggingface/transformers", "v5.23.0"),
    "vllm": ("vllm-project/vllm", "v0.33.0rc3"),
    "sglang": ("sgl-project/sglang", "v0.5.23"),
    "llama.cpp": ("ggml-org/llama.cpp", "v1.0.0"),
    "mlx-lm": ("ml-explore/mlx-lm", "v0.32.2"),
    "tgi": ("huggingface/text-generation-inference", "v3.3.7"),
    "executorch": ("pytorch/executorch", "v1.6.0"),
    "optimum-executorch": ("huggingface/optimum-executorch", "dc779b7fb1f2185d7c67f5f152e99b2ed8a0e315"),
    "onnx": ("huggingface/optimum-onnx", "46533fb66570021f52a39d77112b703762fdaada"),
    "transformers.js": ("huggingface/transformers.js", "4.3.2"),
}
FRAMEWORKS = ("transformers", "vllm", "sglang", "llama.cpp", "mlx-lm", "tgi", "executorch", "onnx",
              "transformers.js")

# executorch/examples/models/<dir> (and export_llama's EXECUTORCH_DEFINED_MODELS) -> HF model_type, read from
# each example's README / HF repo id; only the LLM decoders
EXECUTORCH_DIRS = {"llama": "llama", "llama2": "llama", "llama3": "llama", "smollm2": "llama",
                   "deepseek-r1-distill-llama-8B": "llama", "qwen2_5": "qwen2", "qwen3": "qwen3",
                   "qwen3_5": "qwen3_5", "qwen3_5_moe": "qwen3_5_moe", "phi_4_mini": "phi3", "phi-3-mini": "phi3",
                   "gemma": "gemma", "gemma2": "gemma2", "gemma3": "gemma3_text", "gemma4": "gemma4",
                   "gemma4_31b": "gemma4", "glm": "glm", "granite": "granite", "lfm2": "lfm2", "smollm3": "smollm3",
                   "codegen": "codegen", "spark_x2_5": "spark2_5", "llava": "llava"}
# optimum-executorch test_modeling_<name> -> model_type (decoder LMs only)
OPTIMUM_ET_TESTS = {"codegen": "codegen", "gemma": "gemma", "gemma2": "gemma2", "gemma3": "gemma3_text", "glm": "glm",
                    "gpt2": "gpt2", "gptj": "gptj", "gptneox": "gpt_neox", "gptneoxjapanese": "gpt_neox_japanese",
                    "granite": "granite", "llama": "llama", "mistral": "mistral", "olmo": "olmo", "phi": "phi",
                    "phi4": "phi3", "qwen2": "qwen2", "qwen3": "qwen3", "smollm": "llama", "smollm3": "smollm3",
                    "starcoder2": "starcoder2"}


def _get(url):
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", url)[-200:])
    if os.path.exists(fn):
        with open(fn) as f:
            return f.read()
    req = urllib.request.Request(url, headers={"User-Agent": "the-emulation-frontier"})
    text = urllib.request.urlopen(req, timeout=90).read().decode()
    with open(fn, "w") as f:
        f.write(text)
    return text


def _raw(fw, path):
    repo, tag = PINS[fw]
    url = f"https://raw.githubusercontent.com/{repo}/{tag}/{path}"
    return url, _get(url)


def _tree(fw):
    repo, tag = PINS[fw]
    return [x["path"] for x in json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/{tag}?recursive=1"))["tree"]]


def _dict_block(text, name):
    """The text of `NAME = {...}` / `NAME = OrderedDict([...])` / `new Map([...])`, by bracket matching."""
    m = re.search(rf"^\s*(?:export\s+const\s+)?{name}\s*(?::[^=]*)?=\s*", text, re.M)
    if not m:
        return ""
    i = m.end()
    opens = text.find("(", i), text.find("{", i), text.find("[", i)
    i = min(x for x in opens if x >= 0)
    depth, j = 0, i
    pair = {"(": ")", "{": "}", "[": "]"}
    stack = []
    while j < len(text):
        c = text[j]
        if c in pair:
            stack.append(pair[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return text[i:j + 1]
        elif c == "#":                              # skip python comments (may hold brackets)
            j = text.find("\n", j)
            continue
        j += 1
    return text[i:]


def _pairs(block):
    """('key', 'Value') pairs of a python mapping or a JS Map literal."""
    if block.lstrip().startswith("{"):              # {"Key": ("module", "Class"), ...} / {"k": "v"}
        return re.findall(r"""["']([\w.\-]+)["']\s*:\s*\(?\s*["']([\w.\-]+)["']""", block)
    return re.findall(r"""[\[(]\s*["']([\w.\-]+)["']\s*,\s*["']([\w.\-]+)["']""", block)


def fetch_transformers():
    u1, conf = _raw("transformers", "src/transformers/models/auto/auto_mappings.py")   # v5: generated mapping
    u0, conf0 = _raw("transformers", "src/transformers/models/auto/configuration_auto.py")
    u2, mod = _raw("transformers", "src/transformers/models/auto/modeling_auto.py")
    types = {k for k, _ in _pairs(_dict_block(conf, "CONFIG_MAPPING_NAMES"))}
    types |= set(re.findall(r"""["']([\w.\-]+)["']\s*:\s*["']\w+Config["']""", conf0))   # the .update({...}) extras
    gen_types, archs = set(), set()
    for name in ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
                 "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES"):
        for k, v in _pairs(_dict_block(mod, name)):
            gen_types.add(k)
            archs.add(v)
    return dict(urls=[u1, u0, u2], key="both", model_types=sorted(gen_types & types), architectures=sorted(archs),
                all_model_types=sorted(types),
                source_note="model_types = keys of the causal-LM / image-text-to-text / multimodal-LM mappings that "
                            "are also in CONFIG_MAPPING_NAMES (auto_mappings.py plus configuration_auto.py's update); architectures = those mappings' class names")


def fetch_vllm():
    u, reg = _raw("vllm", "vllm/model_executor/models/registry.py")
    archs = set()
    for name in ("_TEXT_GENERATION_MODELS", "_MULTIMODAL_MODELS"):
        archs |= {k for k, _ in _pairs(_dict_block(reg, name))}
    # the architectures the registry maps to the Transformers backend ("ArchForCausalLM": ("transformers",
    # "TransformersForCausalLM")); _TRANSFORMERS_BACKEND_MODELS holds only the backend's own class names
    tb = {k for k, v in _pairs(_dict_block(reg, "_TRANSFORMERS_SUPPORTED_MODELS")) if v == "transformers"}
    return dict(urls=[u], key="architecture", architectures=sorted(archs), model_types=[],
                transformers_backend=sorted(tb - archs),
                backend_classes=sorted(k for k, _ in _pairs(_dict_block(reg, "_TRANSFORMERS_BACKEND_MODELS"))),
                source_note="keys of _TEXT_GENERATION_MODELS and _MULTIMODAL_MODELS; transformers_backend = the "
                            "keys of _TRANSFORMERS_SUPPORTED_MODELS (vLLM runs them through its Transformers "
                            "backend: shown as a separate tier, not counted as native)")


def fetch_sglang():
    files = [p for p in _tree("sglang") if re.fullmatch(r"python/sglang/srt/models/[^/]+\.py", p)]
    archs, urls = set(), []
    for p in files:
        u, t = _raw("sglang", p)
        urls.append(u)
        for m in re.finditer(r"^EntryClass\s*=\s*(\[[^\]]*\]|\w+)", t, re.M):
            archs |= set(re.findall(r"[A-Za-z_]\w*", m.group(1)))
    return dict(urls=[f"https://github.com/{PINS['sglang'][0]}/tree/{PINS['sglang'][1]}/python/sglang/srt/models"],
                key="architecture", architectures=sorted(archs), model_types=[], files_read=len(urls),
                source_note="EntryClass names in python/sglang/srt/models/*.py")


def fetch_llamacpp():
    paths = _tree("llama.cpp")
    files = [p for p in paths if re.fullmatch(r"conversion/[^/]+\.py", p)] + ["convert_hf_to_gguf.py"]
    bases, regs = {}, {}
    for p in files:
        _, t = _raw("llama.cpp", p)
        for m in re.finditer(r"((?:@\w+\.register\((?:[^()]|\([^()]*\))*\)\s*(?:#[^\n]*\n\s*)*"
                             r"(?:@[\w.]+\((?:[^()]|\([^()]*\))*\)\s*(?:#[^\n]*\n\s*)*)*)*)class\s+(\w+)\(([^)]*)\)", t):
            dec, cls, base = m.group(1), m.group(2), m.group(3)
            bases[cls] = [b.strip() for b in base.split(",") if b.strip()]
            names = []
            for r in re.finditer(r"@(?:ModelBase|Model)\.register\(((?:[^()]|\([^()]*\))*)\)", dec):
                names += re.findall(r"""["']([\w.\-]+)["']""", r.group(1))
            if names:
                regs[cls] = names

    def root_is_mmproj(c, seen=()):
        if c in ("MmprojModel",) or "Mmproj" in c:
            return True
        return any(root_is_mmproj(b.split(".")[-1], seen + (c,)) for b in bases.get(c, []) if b not in seen)
    text_archs = sorted({n for c, ns in regs.items() if not root_is_mmproj(c) for n in ns})
    mm_archs = sorted({n for c, ns in regs.items() if root_is_mmproj(c) for n in ns})
    u, arch = _raw("llama.cpp", "src/llama-arch.cpp")
    runtime = sorted(set(re.findall(r"\{\s*LLM_ARCH_\w+\s*,\s*\"([\w.\-]+)\"\s*\}", arch)))
    return dict(urls=[f"https://github.com/{PINS['llama.cpp'][0]}/tree/{PINS['llama.cpp'][1]}/conversion", u],
                key="architecture", architectures=text_archs, model_types=[], mmproj_architectures=mm_archs,
                runtime_archs=runtime,
                source_note="@ModelBase.register names on text-model classes in conversion/*.py (the "
                            "convert_hf_to_gguf.py package); runtime_archs = LLM_ARCH_NAMES in src/llama-arch.cpp")


def fetch_mlx():
    files = [p for p in _tree("mlx-lm") if re.fullmatch(r"mlx_lm/models/[^/]+\.py", p)]
    non_models = {"__init__", "base", "cache", "activations", "bitlinear_layers", "rope_utils", "switch_layers",
                  "pipeline", "ssm", "gated_delta", "mla", "recurrent_gemma_utils", "su_rope", "kernels"}
    types = {os.path.basename(p)[:-3] for p in files} - non_models
    u, ut = _raw("mlx-lm", "mlx_lm/utils.py")
    remap = dict(_pairs(_dict_block(ut, "MODEL_REMAPPING")))
    types |= {k for k, v in remap.items() if v in types}
    return dict(urls=[f"https://github.com/{PINS['mlx-lm'][0]}/tree/{PINS['mlx-lm'][1]}/mlx_lm/models", u],
                key="model_type", architectures=[], model_types=sorted(types), remapping=remap,
                source_note="file names of mlx_lm/models/*.py (helper modules dropped) plus MODEL_REMAPPING aliases")


def fetch_tgi():
    u, t = _raw("tgi", "server/text_generation_server/models/__init__.py")
    blk = t[t.find("class ModelType"):]
    blk = blk[:blk.find("\n\n\n")]
    types = sorted(set(re.findall(r"""["']type["']\s*:\s*["']([\w.\-]+)["']""", blk)))
    # get_model()'s last branch: a model_type in transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES runs through
    # TransformersFlashCausalLM.fallback / CausalLM.fallback, with the transformers version TGI pins
    ur, req = _raw("tgi", "server/requirements_cuda.txt")
    pin = re.search(r"^transformers==([\w.]+)", req, re.M)
    tv = "v" + pin.group(1) if pin else PINS["transformers"][1]
    url = f"https://raw.githubusercontent.com/huggingface/transformers/{tv}/src/transformers/models/auto/modeling_auto.py"
    fb = {k for k, _ in _pairs(_dict_block(_get(url), "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES"))}
    return dict(urls=[u, ur, url], key="model_type", architectures=[], model_types=types,
                transformers_fallback=sorted(fb - set(types)), transformers_pin=tv,
                source_note="ModelType enum 'type' values; transformers_fallback = the other model types of "
                            f"transformers {tv} MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, which get_model() runs through "
                            "its transformers fallback (shown as a separate tier, not counted as native; remote "
                            "code through auto_map is not counted)")


# executorch examples/models/llama/README.md "Here are supported models:" (lines 4-10, v1.6.0) names sizes, not
# repo ids; these are the Meta repos those names refer to (the README's download page)
LLAMA_README_IDS = {"Llama 3.2 1B": "meta-llama/Llama-3.2-1B", "Llama 3.2 3B": "meta-llama/Llama-3.2-3B",
                    "Llama 3.1 8B": "meta-llama/Llama-3.1-8B", "Llama 3 8B": "meta-llama/Meta-Llama-3-8B",
                    "Llama 2 7B": "meta-llama/Llama-2-7b-hf"}
_ID = r"[A-Za-z0-9][\w\-]*/[\w.\-]*\w"
_NOT_ID = (".py", ".md", ".json", ".pte", ".bin", ".txt", ".yaml", ".pt", ".cpp", ".h")


def _dims_of_export(js):
    """(hidden, layers, heads, kv_heads) of an ExecuTorch export config: llama-style params.json keys or an HF
    config (top level or text_config)."""
    if not isinstance(js, dict):
        return None
    if "dim" in js and "n_layers" in js:
        return dict(hidden=int(js["dim"]), layers=int(js["n_layers"]), heads=int(js["n_heads"]),
                    kv_heads=int(js.get("n_kv_heads") or js["n_heads"]))
    t = js.get("text_config") if isinstance(js.get("text_config"), dict) else js
    if "hidden_size" in t and "num_hidden_layers" in t:
        return dict(hidden=int(t["hidden_size"]), layers=int(t["num_hidden_layers"]),
                    heads=int(t["num_attention_heads"]),
                    kv_heads=int(t.get("num_key_value_heads") or t["num_attention_heads"]))
    return None


def executorch_recipes(paths):
    """The models ExecuTorch has an export path for, per MODEL (not per model type):
      export_configs  the shapes the examples/models/<dir> recipes export: every <dir>/**/*config*.json with model
                      dims (llama-style params.json or an HF config), and the dims a <dir>/model.py dataclass
                      defaults to (gemma4_31b) -- the toy params_demo_config.json is dropped
      model_ids       the repos optimum-executorch tests (tests/models/test_modeling_<decoder>.py) and lists in its
                      README "Supported Models" ("X and its variants"), and the repos the example READMEs name
    A repo is ExecuTorch-native when its shape matches one of these (graph.frameworks)."""
    out_cfg, ids = [], {}
    for d, mt in EXECUTORCH_DIRS.items():
        pre = f"examples/models/{d}/"
        for p in sorted(x for x in paths if x.startswith(pre) and x.count("/") <= 4):
            name = os.path.basename(p)
            if name.endswith(".json") and "config" in name and "demo" not in name and "CMake" not in name:
                try:
                    dims = _dims_of_export(json.loads(_raw("executorch", p)[1]))
                except (ValueError, KeyError, TypeError):
                    dims = None
                if dims and dims["hidden"] >= 256:
                    out_cfg.append(dict(dir=d, model_type=mt, file=p, **dims))
            elif name == "model.py":
                t = _raw("executorch", p)[1]
                g = {k: re.search(rf"^\s*{k}\s*:\s*int\s*=\s*(\d+)", t, re.M) for k in
                     ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads")}
                if all(g.values()):
                    out_cfg.append(dict(dir=d, model_type=mt, file=p, hidden=int(g["hidden_size"].group(1)),
                                        layers=int(g["num_hidden_layers"].group(1)),
                                        heads=int(g["num_attention_heads"].group(1)),
                                        kv_heads=int(g["num_key_value_heads"].group(1))))
            if name == "README.md":
                t = _raw("executorch", p)[1]
                for m in re.findall(rf"huggingface\.co/({_ID})", t):
                    if not m.startswith(("collections/", "blog/", "docs/", "spaces/", "datasets/")):
                        ids.setdefault(m, f"executorch {p}")
                if d == "llama":
                    blk = t[t.find("Here are supported models:"):][:600]
                    for line in blk.splitlines()[1:]:
                        if not line.startswith("- "):
                            continue
                        for k, rid in LLAMA_README_IDS.items():
                            fam, size = k.rsplit(" ", 1)
                            if fam in line and re.search(rf"\b{size}\b", line):
                                ids.setdefault(rid, f"executorch {p} (supported models list)")
    r2, t2 = PINS["optimum-executorch"]
    for p in _tree("optimum-executorch"):
        m = re.fullmatch(r"tests/models/test_modeling_(\w+)\.py", p)
        if m and m.group(1) in OPTIMUM_ET_TESTS:
            for rid in re.findall(rf"""["']({_ID})["']""", _raw("optimum-executorch", p)[1]):
                if not rid.endswith(_NOT_ID):
                    ids.setdefault(rid, f"optimum-executorch {p}")
    u, readme = _raw("optimum-executorch", "README.md")
    sec = readme[readme.find("##### Decoder-only"):]           # the decoder LLM list only
    sec = sec[:sec.find("\n#", 5)]
    for rid in re.findall(rf"\(https://huggingface\.co/({_ID})\)", sec):
        ids.setdefault(rid, "optimum-executorch README.md Supported Models")
    return {"export_configs": out_cfg, "model_ids": [{"id": k, "source": v} for k, v in sorted(ids.items())]}


def fetch_executorch():
    paths = _tree("executorch")
    dirs = {p.split("/")[2] for p in paths if p.startswith("examples/models/") and p.count("/") >= 3}
    u, lib = _raw("executorch", "examples/models/llama/export_llama_lib.py")
    oet = {os.path.basename(p)[len("test_modeling_"):-3] for p in _tree("optimum-executorch")
           if "tests/models/test_modeling_" in p}
    types = {EXECUTORCH_DIRS[d] for d in dirs if d in EXECUTORCH_DIRS} | {OPTIMUM_ET_TESTS[t] for t in oet
                                                                         if t in OPTIMUM_ET_TESTS}
    r2, t2 = PINS["optimum-executorch"]
    recipes = executorch_recipes(paths)
    return dict(urls=[f"https://github.com/{PINS['executorch'][0]}/tree/{PINS['executorch'][1]}/examples/models", u,
                      f"https://github.com/{r2}/tree/{t2}/tests/models", f"https://github.com/{r2}/blob/{t2}/README.md"],
                key="model_type", architectures=[], model_types=sorted(types),
                example_dirs=sorted(d for d in dirs if d in EXECUTORCH_DIRS),
                optimum_executorch_tests=sorted(t for t in oet if t in OPTIMUM_ET_TESTS),
                export_configs=recipes["export_configs"], model_ids=recipes["model_ids"],
                source_note="LLM families with an executorch/examples/models/<dir> export (dir -> model_type by "
                            "hand, EXECUTORCH_DIRS) plus the decoder model types optimum-executorch tests "
                            "(tests/models/test_modeling_<type>.py, pinned main commit; its v0.1.0 release is a "
                            "year old)")


def fetch_onnx():
    u, t = _raw("onnx", "optimum/exporters/onnx/model_configs.py")
    types = set()
    for m in re.finditer(r"@register_tasks_manager_onnx\(((?:[^()]|\([^()]*\))*)\)", t):
        args = m.group(1)
        mt = re.match(r"""\s*["']([\w.\-]+)["']""", args)
        # a decoder registers COMMON_TEXT_GENERATION_TASKS or "text-generation" itself (the seq2seq lists name
        # "text2text-generation", which this does not match)
        if mt and ("COMMON_TEXT_GENERATION_TASKS" in args or re.search(r"""["']text-generation(-with-past)?["']""",
                                                                       args)):
            types.add(mt.group(1))
    return dict(urls=[u], key="model_type", architectures=[], model_types=sorted(types),
                source_note="@register_tasks_manager_onnx model types whose tasks include text-generation "
                            "(optimum-onnx pinned main commit; its v0.1.0 release predates most current models)")


def fetch_transformers_js():
    u, t = _raw("transformers.js", "packages/transformers/src/models/registry.js")
    types, archs = set(), set()
    for name in ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
                 "MODEL_FOR_MULTIMODALITY_MAPPING_NAMES"):
        for k, v in _pairs(_dict_block(t, name)):
            types.add(k)
            archs.add(v)
    return dict(urls=[u], key="model_type", architectures=sorted(archs), model_types=sorted(types),
                source_note="keys of MODEL_FOR_CAUSAL_LM / IMAGE_TEXT_TO_TEXT / MULTIMODALITY mapping names")


FETCH = {"transformers": fetch_transformers, "vllm": fetch_vllm, "sglang": fetch_sglang, "llama.cpp": fetch_llamacpp,
         "mlx-lm": fetch_mlx, "tgi": fetch_tgi, "executorch": fetch_executorch, "onnx": fetch_onnx,
         "transformers.js": fetch_transformers_js}

SANITY = {"transformers": ("LlamaForCausalLM", "llama"), "vllm": ("LlamaForCausalLM", "Qwen3ForCausalLM",
                                                                  "DeepseekV3ForCausalLM"),
          "sglang": ("LlamaForCausalLM",), "llama.cpp": ("LlamaForCausalLM", "Qwen3ForCausalLM"),
          "mlx-lm": ("llama", "qwen3"), "tgi": ("llama",), "executorch": ("llama",), "onnx": ("llama",),
          "transformers.js": ("llama",)}


def fetch_all(today=None):
    today = today or datetime.date.today().isoformat()
    out = {}
    for fw in FRAMEWORKS:
        rec = FETCH[fw]()
        repo, tag = PINS[fw]
        rec = {"repo": repo, "version": tag, "fetched": today, **rec}
        names = set(rec["architectures"]) | set(rec["model_types"])
        missing = [s for s in SANITY[fw] if s not in names]
        assert not missing, (fw, missing)
        out[fw] = rec
    return {"generated": today, "frameworks": out}


def load(path=OUT):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    data = fetch_all()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    for fw, r in data["frameworks"].items():
        print(f"{fw:16} {r['version']:12} archs {len(r['architectures']):4}  model_types {len(r['model_types']):4}")
