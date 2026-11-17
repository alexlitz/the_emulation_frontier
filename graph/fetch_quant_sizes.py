"""Ground-truth file sizes of quantised checkpoints on the Hub, for
validating a size model. Metadata only: Hub JSON endpoints (file trees with
sizes, gguf header summaries, safetensors dtype counts) and small
config.json files. No weights are downloaded and no repo code is run.

  python -m graph.fetch_quant_sizes            # writes graph/data/quant_sizes.json

Responses are cached under $TEF_SCRATCH/quant_sizes_cache/ (TEF_SCRATCH defaults to ~/.cache/the_emulation_frontier)
so a rerun is offline.

GGUF
----
Discovery: the full model listings of a few uploaders that publish plain
llama-quantize outputs (bartowski, lmstudio-community, QuantFactory,
second-state, ggml-org and the vendors' own -GGUF repos) are paged with
their tags; a repo is a candidate for snapshot base B when it carries the
tag base_model:quantized:B. For families those listings miss, the
quantised children of an official base are queried directly
(filter=base_model:quantized:<base>), which also surfaces mradermacher's
static repos. unsloth is never used (its "UD" dynamic quants override
tensor types) and mradermacher's -i1- repos are skipped in favour of the
static ones.

Acceptance, per candidate repo:
  * base_model tag names B exactly;
  * the gguf header summary (expand[]=gguf) exists and its architecture is
    the llama.cpp name of B's model_type (table GGUF_ARCH);
  * gguf.total is within 3% of B's safetensors parameter count (or B's
    count is unknown / packed and the check is reported as skipped);
    three explained exceptions are accepted and flagged: a tied-embedding
    base whose GGUF stores the embedding twice (gguf.total = base +
    vocab*hidden within 1%, tied_output_duplicated=True), a base with MTP
    layers that llama.cpp drops (ratio 0.85-0.97), and a multimodal base
    whose GGUF holds only the text tower (an mmproj file is present,
    ratio >= 0.5, text_only=True);
  * implausible files are dropped with a note: < 1 bit/weight (truncated
    upload) and "F16"/"BF16" files above 17 bits/weight (really F32);
  * file names parse to standard llama-quantize levels. Only exact level
    names are kept: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, Q4_0, BF16,
    F16. Variants with tensor overrides (bartowski _L / _XL, Q4_0_4_4 etc,
    anything with "UD") do not match the pattern and are ignored. Split
    files (-00001-of-0000N.gguf, or mradermacher .gguf.partXofN) are summed.
  * at least three of those levels are present.
At most two accepted repos are kept per base (cross-checks between
uploaders), and per model_type at most MAX_PER_TYPE bases (DIRECT_BASES first,
then official orgs, then snapshot order).

Safetensors
-----------
A curated list (SAFETENSORS_CANDIDATES) plus quantised children of the
accepted GGUF bases that publish safetensors with a quantization_config.
Acceptance: the quant repo's config.json has the same model_type,
hidden_size, num_hidden_layers and vocab_size as the snapshot base (text
config for multimodal ones), the base itself carries no quantization_config,
and a quantization_config whose method maps to gptq / awq / fp8 /
compressed-tensors / bnb / mlx. compressed-tensors schemes with float 8-bit
weights are labelled fp8 (group_size = strategy or block structure); schemes
mixing several weight formats get bits="mixed". MLX is kept only for uniform
4-bit and 8-bit recipes (no per-layer overrides), one repo per base and width.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

CACHE = os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "quant_sizes_cache")
OUT = "graph/data/quant_sizes.json"
HUB = "https://huggingface.co"
SLEEP = 0.25
MAX_PER_TYPE = 10
MAX_PER_BASE = 2

LISTED_UPLOADERS = ["bartowski", "lmstudio-community", "QuantFactory",
                    "second-state", "ggml-org"]
# vendor orgs that publish their own -GGUF repos
VENDOR_ORGS = ["Qwen", "google", "microsoft", "ibm-granite", "tiiuae",
               "LGAI-EXAONE", "LiquidAI", "HuggingFaceTB", "allenai",
               "mistralai", "internlm", "zai-org", "openai", "meta-llama",
               "deepseek-ai", "nvidia", "baidu", "tencent", "swiss-ai"]
UPLOADER_RANK = {"bartowski": 0, "lmstudio-community": 1, "vendor": 2,
                 "ggml-org": 3, "QuantFactory": 4, "mradermacher": 5,
                 "second-state": 6}

LEVELS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "Q4_0",
          "BF16", "F16"]
LEVEL_RE = re.compile(
    r"(?:^|[-_.])(" + "|".join(LEVELS) + r")"
    r"(?:-(\d{5})-of-(\d{5}))?\.gguf(?:\.part(\d+)of(\d+))?$", re.I)

# HF model_type -> llama.cpp gguf architecture
GGUF_ARCH = {
    "llama": {"llama"}, "mistral": {"llama"}, "mixtral": {"llama"},
    "qwen2": {"qwen2"}, "qwen3": {"qwen3"}, "qwen3_moe": {"qwen3moe"},
    "qwen2_moe": {"qwen2moe"}, "qwen3_next": {"qwen3next"},
    "gemma": {"gemma"}, "gemma2": {"gemma2"}, "gemma3": {"gemma3"},
    "gemma3_text": {"gemma3"}, "phi3": {"phi3"}, "phi": {"phi2"},
    "phimoe": {"phimoe"}, "deepseek_v2": {"deepseek2"},
    "deepseek_v3": {"deepseek2"}, "deepseek": {"deepseek"},
    "olmo": {"olmo"}, "olmo2": {"olmo2"}, "olmoe": {"olmoe"},
    "olmo3": {"olmo2"}, "granite": {"granite"},
    "granitemoe": {"granitemoe"},
    # granite-4.0 micro / 1b / 350m are attention-only and convert to "granite"
    "granitemoehybrid": {"granitehybrid", "granite"},
    "falcon": {"falcon"}, "falcon_h1": {"falcon-h1"},
    "falcon_mamba": {"mamba"}, "mamba": {"mamba"},
    "internlm2": {"internlm2"}, "internlm3": {"llama"},
    "glm4": {"glm4"}, "chatglm": {"chatglm"}, "glm4_moe": {"glm4moe"},
    "exaone": {"exaone"}, "exaone4": {"exaone4"},
    "cohere": {"command-r"}, "cohere2": {"cohere2"},
    "starcoder2": {"starcoder2"}, "gpt_bigcode": {"starcoder"},
    "gpt_oss": {"gpt-oss"}, "smollm3": {"smollm3"},
    "stablelm": {"stablelm"}, "gpt2": {"gpt2"}, "gpt_neox": {"gptneox"},
    "bloom": {"bloom"}, "lfm2": {"lfm2"}, "hunyuan_v1_dense": {"hunyuan-dense"},
    "ernie4_5": {"ernie4_5"}, "ernie4_5_moe": {"ernie4_5-moe"},
    "seed_oss": {"seed_oss"}, "apertus": {"apertus"},
    "nemotron_h": {"nemotron_h", "nemotron_h_moe"}, "nemotron": {"nemotron"},
    "minimax_m2": {"minimax-m2"}, "bailing_moe": {"bailingmoe", "bailingmoe2"},
    "arcee": {"arcee"}, "baichuan": {"baichuan"}, "minicpm": {"minicpm"},
    "jamba": {"jamba"}, "dots1": {"dots1"}, "nemotron-nas": {"deci"},
    "mistral3": {"llama", "mistral3"}, "llama4": {"llama4"},
}

# Official bases queried directly through their quantised children, to
# reach families/sizes the uploader listings miss (and mradermacher).
DIRECT_BASES = [
    "tiiuae/falcon-mamba-7b-instruct", "state-spaces/mamba-2.8b-hf",
    "state-spaces/mamba-130m-hf", "bigcode/starcoder2-3b",
    "bigcode/starcoder2-7b", "bigcode/starcoder2-15b",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "LGAI-EXAONE/EXAONE-3.5-32B-Instruct", "LGAI-EXAONE/EXAONE-4.0-32B",
    "internlm/internlm2_5-7b-chat", "internlm/internlm2_5-1_8b-chat",
    "internlm/internlm3-8b-instruct", "zai-org/GLM-4-9B-0414",
    "zai-org/GLM-Z1-32B-0414", "zai-org/GLM-4.5-Air", "zai-org/glm-4-9b",
    "deepseek-ai/DeepSeek-V2-Lite-Chat", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-V3-0324", "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/deepseek-moe-16b-chat", "openai/gpt-oss-20b",
    "openai/gpt-oss-120b", "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b-instruct",
    "allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct", "allenai/OLMo-2-0325-32B-Instruct",
    "allenai/OLMoE-1B-7B-0125-Instruct", "allenai/Olmo-3-7B-Instruct",
    "ibm-granite/granite-3.3-8b-instruct", "ibm-granite/granite-3.3-2b-instruct",
    "ibm-granite/granite-3.1-3b-a800m-instruct", "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-micro", "ibm-granite/granite-4.0-h-small",
    "HuggingFaceTB/SmolLM3-3B", "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "google/gemma-3-1b-it", "google/gemma-3-270m", "google/gemma-2-2b-it",
    "google/gemma-2-27b-it", "google/gemma-7b-it", "google/gemma-1.1-2b-it",
    "google/codegemma-7b-it", "microsoft/phi-4", "microsoft/Phi-4-mini-instruct",
    "microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3.5-MoE-instruct", "microsoft/phi-2",
    "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat", "Qwen/Qwen2-57B-A14B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct",
    "stabilityai/stablelm-2-1_6b-chat", "stabilityai/stablelm-zephyr-3b",
    "LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-350M", "tencent/Hunyuan-7B-Instruct",
    "baidu/ERNIE-4.5-0.3B-PT", "baidu/ERNIE-4.5-21B-A3B-PT",
    "ByteDance-Seed/Seed-OSS-36B-Instruct", "swiss-ai/Apertus-8B-Instruct-2509",
    "MiniMaxAI/MiniMax-M2", "inclusionAI/Ling-mini-2.0", "arcee-ai/AFM-4.5B",
    "tiiuae/Falcon-H1-1.5B-Instruct", "tiiuae/Falcon-H1-7B-Instruct",
    "tiiuae/Falcon3-7B-Instruct", "tiiuae/Falcon3-1B-Instruct",
    "01-ai/Yi-1.5-9B-Chat", "01-ai/Yi-1.5-34B-Chat-16K",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", "nvidia/Nemotron-Mini-4B-Instruct",
    "openai-community/gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "bigscience/bloom-560m", "EleutherAI/pythia-1.4b",
    "baichuan-inc/Baichuan2-7B-Chat", "adamo1139/aya-expanse-8b-ungated",
]

# Safetensors quantised repos, curated: (base, repo).
SAFETENSORS_CANDIDATES = [
    # GPTQ
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4"),
    ("Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4"),
    ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-GPTQ-Int4"),
    ("Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-GPTQ-Int4"),
    ("meta-llama/Llama-3.1-8B-Instruct", "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"),
    ("meta-llama/Llama-3.1-70B-Instruct", "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"),
    ("mistralai/Mistral-7B-Instruct-v0.1", "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"),
    ("HuggingFaceH4/zephyr-7b-beta", "TheBloke/zephyr-7B-beta-GPTQ"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"),
    ("deepseek-ai/deepseek-coder-6.7b-instruct", "TheBloke/deepseek-coder-6.7B-instruct-GPTQ"),
    # AWQ
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct-AWQ"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct-AWQ"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct-AWQ"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"),
    ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-AWQ"),
    ("Qwen/Qwen3-14B", "Qwen/Qwen3-14B-AWQ"),
    ("Qwen/Qwen3-32B", "Qwen/Qwen3-32B-AWQ"),
    ("Qwen/Qwen3-4B", "Qwen/Qwen3-4B-AWQ"),
    ("meta-llama/Llama-3.1-8B-Instruct", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"),
    ("meta-llama/Llama-3.1-70B-Instruct", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"),
    ("HuggingFaceH4/zephyr-7b-beta", "TheBloke/zephyr-7B-beta-AWQ"),
    # FP8
    ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-FP8"),
    ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-FP8"),
    ("Qwen/Qwen3-4B", "Qwen/Qwen3-4B-FP8"),
    ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-FP8"),
    ("Qwen/Qwen3-14B", "Qwen/Qwen3-14B-FP8"),
    ("Qwen/Qwen3-32B", "Qwen/Qwen3-32B-FP8"),
    ("Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-FP8"),
    ("Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-FP8"),
    ("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507-FP8"),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"),
    ("Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"),
    ("meta-llama/Llama-3.1-8B-Instruct", "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"),
    ("meta-llama/Llama-3.1-8B-Instruct", "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"),
    ("meta-llama/Llama-3.1-70B-Instruct", "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8"),
    ("meta-llama/Llama-3.2-1B-Instruct", "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic"),
    ("meta-llama/Llama-3.2-3B-Instruct", "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic"),
    ("microsoft/phi-4", "RedHatAI/phi-4-FP8-dynamic"),
    ("Qwen/Qwen2.5-7B-Instruct", "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic"),
    ("google/gemma-2-2b-it", "RedHatAI/gemma-2-2b-it-FP8"),
    ("google/gemma-2-27b-it", "RedHatAI/gemma-2-27b-it-FP8"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"),
    # compressed-tensors int
    ("meta-llama/Llama-3.1-8B-Instruct", "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"),
    ("meta-llama/Llama-3.1-8B-Instruct", "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"),
    ("meta-llama/Llama-3.1-8B-Instruct", "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"),
    ("meta-llama/Llama-3.2-1B-Instruct", "RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8"),
    ("meta-llama/Llama-3.1-70B-Instruct", "RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"),
    ("Qwen/Qwen2.5-7B-Instruct", "RedHatAI/Qwen2.5-7B-Instruct-quantized.w4a16"),
    ("Qwen/Qwen2.5-7B-Instruct", "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8"),
    ("microsoft/phi-4", "RedHatAI/phi-4-quantized.w4a16"),
    ("Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-quantized.w4a16"),
    ("Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-FP8-dynamic"),
    ("Qwen/Qwen3-30B-A3B", "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"),
    # bitsandbytes
    ("Qwen/Qwen3-8B", "unsloth/Qwen3-8B-bnb-4bit"),
    ("meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"),
    ("meta-llama/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"),
    ("meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
    ("Qwen/Qwen2.5-7B-Instruct", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"),
    ("microsoft/phi-4", "unsloth/phi-4-bnb-4bit"),
    # MLX
    ("Qwen/Qwen3-8B", "lmstudio-community/Qwen3-8B-MLX-4bit"),
    ("Qwen/Qwen3-8B", "lmstudio-community/Qwen3-8B-MLX-8bit"),
    ("Qwen/Qwen3-30B-A3B", "lmstudio-community/Qwen3-30B-A3B-MLX-4bit"),
    ("Qwen/Qwen3-30B-A3B", "lmstudio-community/Qwen3-30B-A3B-MLX-8bit"),
    ("meta-llama/Llama-3.2-1B-Instruct", "mlx-community/Llama-3.2-1B-Instruct-4bit"),
    ("meta-llama/Llama-3.2-3B-Instruct", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
    ("meta-llama/Llama-3.1-8B-Instruct", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
    ("meta-llama/Llama-3.1-8B-Instruct", "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"),
    ("Qwen/Qwen2.5-7B-Instruct", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
    ("openai/gpt-oss-20b", "lmstudio-community/gpt-oss-20b-MLX-8bit"),
]

# ---------------------------------------------------------------- http

_last = [0.0]


def _key(url):
    return os.path.join(CACHE, hashlib.sha1(url.encode()).hexdigest() + ".json")


def get(url, *, with_link=False):
    """GET a JSON (or text) URL with an on-disk cache; None on 4xx."""
    os.makedirs(CACHE, exist_ok=True)
    path = _key(url)
    if os.path.exists(path):
        rec = json.load(open(path))
    else:
        wait = SLEEP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": "quant-size-survey/1.0"})
        rec = {"url": url}
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read().decode("utf-8", "replace")
                    rec.update(status=r.status, body=body, link=r.headers.get("Link"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                rec.update(status=e.code, body=None, link=None)
                break
            except Exception as e:                      # network hiccup
                if attempt == 3:
                    raise
                time.sleep(3)
        _last[0] = time.time()
        json.dump(rec, open(path, "w"))
    if rec["status"] != 200 or rec["body"] is None:
        return (None, None) if with_link else None
    try:
        data = json.loads(rec["body"])
    except json.JSONDecodeError:
        data = None
    return (data, rec.get("link")) if with_link else data


def _next(link):
    if not link:
        return None
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return m.group(1) if m else None


def get_paged(url):
    out = []
    while url:
        data, link = get(url, with_link=True)
        if not data:
            break
        out.extend(data)
        url = _next(link)
    return out


def q(s):
    return urllib.parse.quote(s, safe="/")


def tree(repo):
    files = get_paged(f"{HUB}/api/models/{q(repo)}/tree/main?recursive=true")
    return [f for f in files if f.get("type") == "file"]


def info(repo, *expand):
    ex = "&".join(f"expand[]={e}" for e in expand)
    return get(f"{HUB}/api/models/{q(repo)}?{ex}") or {}


def children(base):
    return get(f"{HUB}/api/models?filter=base_model:quantized:{q(base)}"
               f"&limit=100&sort=downloads&expand[]=tags") or []


def config(repo):
    return get(f"{HUB}/{q(repo)}/resolve/main/config.json")

# ---------------------------------------------------------------- snapshot


def snapshot():
    from graph.frontier import load
    from graph.model_graph import iter_json_configs
    names = [m["name"] for m in load()]
    configs = dict(iter_json_configs("graph/hub_10k.json"))
    try:
        meta = json.load(open("graph/model_meta.json"))
    except FileNotFoundError:
        meta = {}
    return names, configs, meta


def text_cfg(cfg):
    cfg = cfg or {}
    tc = cfg.get("text_config") or cfg.get("llm_config") or {}
    return tc if tc and not cfg.get("hidden_size") and not cfg.get("n_embd") else cfg


def model_type(cfg):
    cfg = cfg or {}
    return cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type")


def shape(cfg):
    t = text_cfg(cfg)
    g = lambda *ks: next((t[k] for k in ks if t.get(k) is not None), None)
    return {"model_type": t.get("model_type") or model_type(cfg),
            "hidden_size": g("hidden_size", "n_embd", "n_embed", "d_model"),
            "num_hidden_layers": g("num_hidden_layers", "n_layer", "num_layers"),
            "vocab_size": g("vocab_size", "padded_vocab_size")}

# ---------------------------------------------------------------- gguf


def tagged_bases(tags):
    return {t.split(":", 2)[2] for t in tags or [] if t.startswith("base_model:quantized:")}


def uploader_of(repo, base):
    a = repo.split("/")[0]
    if a == base.split("/")[0] or a in VENDOR_ORGS:
        return "vendor"
    return a


def discover(snapset):
    """base -> {repo: uploader} from uploader listings."""
    cands = {}
    for author in LISTED_UPLOADERS + VENDOR_ORGS:
        rows = get_paged(f"{HUB}/api/models?author={author}&limit=1000&expand[]=tags")
        for r in rows:
            rid = r["id"]
            if "gguf" not in (r.get("tags") or []) and not rid.lower().endswith("gguf"):
                continue
            for b in tagged_bases(r.get("tags")):
                if b in snapset:
                    cands.setdefault(b, {})[rid] = uploader_of(rid, b)
    return cands


def parse_files(files):
    levels, skipped = {}, []
    for f in files:
        p = f["path"]
        name = p.rsplit("/", 1)[-1]
        if not re.search(r"\.gguf(\.part\d+of\d+)?$", name, re.I):
            continue
        if "mmproj" in name.lower():
            continue
        m = LEVEL_RE.search(name)
        if not m or re.search(r"(^|[-_.])UD[-_.]", name):
            skipped.append(name)
            continue
        lv = next(L for L in LEVELS if L.lower() == m.group(1).lower())
        stem = p[: m.start(1)]
        e = levels.setdefault(lv, {"bytes": 0, "parts": 0, "paths": [], "stems": set()})
        e["bytes"] += f.get("lfs", {}).get("size", f["size"])
        e["parts"] += 1
        e["paths"].append(p)
        e["stems"].add(stem)
        n_of = m.group(3) or m.group(5)
        if n_of:
            e["n_of"] = int(n_of)
    out, notes = {}, []
    for lv, e in levels.items():
        if len(e["stems"]) > 1:
            notes.append(f"{lv}: several files for one level {sorted(e['paths'])[:4]}; skipped")
            continue
        if e.get("n_of") and e["n_of"] != e["parts"]:
            notes.append(f"{lv}: {e['parts']} of {e['n_of']} parts present; skipped")
            continue
        paths = sorted(e["paths"])
        out[lv] = {"bytes": e["bytes"], "parts": e["parts"],
                   "path": paths[0] if len(paths) == 1 else paths}
    return out, notes, skipped


def check_gguf(base, repo, uploader, cfg, base_params, rejected):
    if repo.split("/")[0] == "unsloth":
        rejected.append({"repo": repo, "base": base, "why": "unsloth (dynamic UD quants override tensor types)"})
        return None
    if re.search(r"-i1-GGUF$", repo, re.I):
        rejected.append({"repo": repo, "base": base, "why": "mradermacher imatrix repo; static repo preferred"})
        return None
    mi = info(repo, "gguf", "tags", "lastModified", "cardData")
    if not mi:
        rejected.append({"repo": repo, "base": base, "why": "info endpoint failed (gated/missing)"})
        return None
    bases = tagged_bases(mi.get("tags"))
    if base not in bases:
        rejected.append({"repo": repo, "base": base, "why": f"base_model tag is {sorted(bases)}"})
        return None
    gm = mi.get("gguf") or {}
    meta = {k: gm.get(k) for k in ("total", "architecture", "context_length") if k in gm}
    if not meta.get("architecture"):
        rejected.append({"repo": repo, "base": base, "why": "no gguf header summary"})
        return None
    mt = model_type(cfg)
    want = GGUF_ARCH.get(mt)
    if want is not None and meta["architecture"] not in want:
        rejected.append({"repo": repo, "base": base,
                         "why": f"gguf architecture {meta['architecture']} != expected {sorted(want)} for {mt}"})
        return None
    files = tree(repo)
    has_mmproj = any("mmproj" in f["path"].lower() for f in files)
    ratio = (meta.get("total") / base_params) if meta.get("total") and base_params else None
    text_only = False
    sh = shape(cfg)
    tc = text_cfg(cfg)
    tied = tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True)) is not False
    emb = (sh["vocab_size"] or 0) * (sh["hidden_size"] or 0)
    mtp = tc.get("num_nextn_predict_layers") or tc.get("mtp_num_hidden_layers") or 0
    tied_dup = False
    if ratio is None:
        param_check = "skipped (no parameter count)"
    elif 0.97 <= ratio <= 1.03:
        param_check = "ok"
    elif tied and emb and abs(meta["total"] - (base_params + emb)) / base_params < 0.01:
        # converter wrote the tied embedding a second time as output.weight
        param_check = "ok (tied embedding stored twice: gguf.total = base + vocab*hidden)"
        tied_dup = True
    elif mtp and 0.85 <= ratio < 0.97:
        param_check = f"ok (llama.cpp drops the {mtp} MTP layer(s))"
    elif has_mmproj and 0.5 <= ratio < 0.97:
        param_check = "ok (text tower only; mmproj separate)"
        text_only = True
    else:
        rejected.append({"repo": repo, "base": base,
                         "why": f"gguf.total/base params = {ratio:.4f} (gguf {meta.get('total')}, base {base_params})"})
        return None
    lv, notes, skipped = parse_files(files)
    P = meta.get("total")
    for k in list(lv):
        bpw = 8 * lv[k]["bytes"] / P if P else None
        if bpw is None:
            continue
        if bpw < 1.0:
            notes.append(f"{k}: {bpw:.2f} bits/weight, truncated upload; dropped")
            del lv[k]
        elif k in ("F16", "BF16") and bpw > 17.0:
            notes.append(f"{k}: {bpw:.2f} bits/weight, file is really F32; dropped")
            del lv[k]
    std = [k for k in lv if k not in ("BF16", "F16")]
    if len(lv) < 3:
        rejected.append({"repo": repo, "base": base,
                         "why": f"fewer than 3 standard levels ({sorted(lv)}); other files {skipped[:6]}"})
        return None
    rec = {"base": base, "repo": repo, "uploader": uploader,
           "model_type": mt, "last_modified": mi.get("lastModified"),
           "base_params": base_params,
           "param_ratio": round(ratio, 5) if ratio else None, "param_check": param_check,
           "text_only": text_only, "tied_output_duplicated": tied_dup,
           "gguf_meta": meta,
           "files": {k: lv[k] for k in LEVELS if k in lv}}
    if notes:
        rec["notes"] = notes
    return rec


def collect_gguf(names, configs, meta, rejected):
    snapset = set(names)
    order = {n: i for i, n in enumerate(names)}      # snapshot order = popularity
    cands = discover(snapset)
    for b in DIRECT_BASES:
        if b not in snapset:
            rejected.append({"repo": None, "base": b, "why": "direct base not in snapshot"})
            continue
        for r in children(b):
            rid = r["id"]
            tags = r.get("tags") or []
            if "gguf" not in tags:
                continue
            up = uploader_of(rid, b)
            if up not in UPLOADER_RANK:
                continue
            cands.setdefault(b, {})[rid] = up
    by_type = {}
    for b in sorted(cands, key=lambda b: order[b]):
        mt = model_type(configs.get(b))
        by_type.setdefault(mt, []).append(b)
    out = []
    for mt, bases in by_type.items():
        # official orgs first, then snapshot order
        def official(b):
            return 0 if b.split("/")[0] in VENDOR_ORGS + ["LGAI-EXAONE", "bigcode", "state-spaces",
                                                         "01-ai", "stabilityai", "ByteDance-Seed",
                                                         "MiniMaxAI", "inclusionAI", "arcee-ai",
                                                         "TinyLlama", "HuggingFaceH4"] else 1
        kept = 0
        direct = set(DIRECT_BASES)
        for b in sorted(bases, key=lambda b: (b not in direct, official(b), order[b])):
            if kept >= MAX_PER_TYPE:
                break
            cfg = configs.get(b)
            bp = (meta.get(b) or {}).get("hf_total")
            good = 0
            for repo, up in sorted(cands[b].items(), key=lambda kv: (UPLOADER_RANK.get(kv[1], 9), kv[0])):
                if good >= MAX_PER_BASE:
                    break
                rec = check_gguf(b, repo, up, cfg, bp, rejected)
                if rec:
                    out.append(rec)
                    good += 1
            if good:
                kept += 1
    return out

# ---------------------------------------------------------------- safetensors


def classify(qc):
    m = (qc.get("quant_method") or "").lower()
    if m == "gptq":
        return "gptq", qc.get("bits"), qc.get("group_size")
    if m == "awq":
        return "awq", qc.get("bits") or qc.get("w_bit"), qc.get("group_size") or qc.get("q_group_size")
    if m == "fp8":
        wb = qc.get("weight_block_size")
        return "fp8", 8, (wb if wb else None)
    if m == "bitsandbytes":
        bits = 4 if qc.get("load_in_4bit") else 8 if qc.get("load_in_8bit") else None
        return "bnb", bits, qc.get("bnb_4bit_quant_type")
    if m == "compressed-tensors":
        ws = [(g.get("weights") or {}) for g in (qc.get("config_groups") or {}).values()]
        kinds = {(w.get("type"), w.get("num_bits"), w.get("group_size"), w.get("strategy"),
                  str(w.get("block_structure"))) for w in ws}
        if len(kinds) != 1:
            return "compressed-tensors", "mixed", "mixed"
        w = ws[0]
        if w.get("type") == "float" and w.get("num_bits") == 8:
            bs = w.get("block_structure")
            return "fp8", 8, (bs if bs else w.get("strategy"))
        return "compressed-tensors", w.get("num_bits"), w.get("group_size") or w.get("strategy")
    if "bits" in qc and "group_size" in qc and not m:
        if any(isinstance(v, dict) for v in qc.values()):
            return "mlx-mixed", qc.get("bits"), qc.get("group_size")
        return "mlx", qc.get("bits"), qc.get("group_size")
    return None, None, None


MLX_SEEN = set()


def check_safetensors(base, repo, configs, rejected):
    cfg = config(repo)
    if not cfg:
        rejected.append({"repo": repo, "base": base, "why": "no config.json (missing or gated)"})
        return None
    qc = cfg.get("quantization_config") or cfg.get("quantization") \
        or (cfg.get("text_config") or {}).get("quantization_config")
    if not qc:
        rejected.append({"repo": repo, "base": base, "why": "config.json has no quantization_config"})
        return None
    bq = (configs.get(base) or {}).get("quantization_config") \
        or ((configs.get(base) or {}).get("text_config") or {}).get("quantization_config")
    if bq:
        rejected.append({"repo": repo, "base": base,
                         "why": f"base is itself quantised ({bq.get('quant_method')}); not a quant of a dense base"})
        return None
    method, bits, group = classify(qc)
    if method is None:
        rejected.append({"repo": repo, "base": base, "why": f"unhandled quant_method {qc.get('quant_method')}"})
        return None
    if method == "mlx-mixed":
        rejected.append({"repo": repo, "base": base, "why": "MLX with per-layer bit overrides (mixed recipe)"})
        return None
    if method == "mlx" and bits not in (4, 8):
        rejected.append({"repo": repo, "base": base, "why": f"MLX {bits}-bit; only 4/8-bit kept"})
        return None
    if method == "mlx" and (base, bits) in MLX_SEEN:
        rejected.append({"repo": repo, "base": base, "why": f"MLX {bits}-bit already recorded for this base"})
        return None
    if method == "mlx":
        MLX_SEEN.add((base, bits))
    sb, sq = shape(configs.get(base)), shape(cfg)
    if sb != sq:
        rejected.append({"repo": repo, "base": base, "why": f"shape mismatch base {sb} vs quant {sq}"})
        return None
    files = tree(repo)
    st = [f for f in files if f["path"].endswith(".safetensors")]
    if not st:
        rejected.append({"repo": repo, "base": base, "why": "no .safetensors files"})
        return None
    mi = info(repo, "safetensors", "lastModified")
    sp = (mi.get("safetensors") or {})
    return {"base": base, "repo": repo, "method": method, "bits": bits, "group_size": group,
            "bytes": sum(f.get("lfs", {}).get("size", f["size"]) for f in st),
            "n_files": len(st), "shape": sq, "last_modified": mi.get("lastModified"),
            "quantization_config": qc,
            "safetensors_params": {"parameters": sp.get("parameters"), "total": sp.get("total")}}


def collect_safetensors(names, configs, gguf_bases, rejected):
    snapset = set(names)
    pairs, seen = [], set()
    for b, r in SAFETENSORS_CANDIDATES:
        if (b, r) not in seen:
            pairs.append((b, r)); seen.add((b, r))
    # quantised safetensors children of accepted gguf bases (official uploaders only)
    OK = {"Qwen", "RedHatAI", "neuralmagic", "hugging-quants", "TheBloke", "lmstudio-community",
          "mlx-community", "unsloth", "Intel", "AMead10", "solidrust", "casperhansen",
          "google", "microsoft", "ibm-granite", "nm-testing", "allenai", "tiiuae",
          "LGAI-EXAONE", "zai-org", "deepseek-ai", "meta-llama", "openai"}
    for b in gguf_bases:
        for r in children(b):
            rid, tags = r["id"], r.get("tags") or []
            if "gguf" in tags or "safetensors" not in tags or rid.split("/")[0] not in OK:
                continue
            if rid.split("/")[0] == "unsloth" and "unsloth-bnb" in rid:
                continue                              # dynamic: some layers kept 16-bit
            if (b, rid) not in seen:
                pairs.append((b, rid)); seen.add((b, rid))
    out = []
    for b, r in pairs:
        if b not in snapset:
            rejected.append({"repo": r, "base": b, "why": "base not in snapshot"})
            continue
        rec = check_safetensors(b, r, configs, rejected)
        if rec:
            out.append(rec)
    return out

# ---------------------------------------------------------------- main


def main():
    names, configs, meta = snapshot()
    rejected = []
    gg = collect_gguf(names, configs, meta, rejected)
    gbases = list(dict.fromkeys(r["base"] for r in gg))
    st = collect_safetensors(names, configs, gbases, rejected)
    doc = {
        "fetched": datetime.date.today().isoformat(),
        "method": ("Hub metadata only (api/models tree sizes incl. LFS, expand[]=gguf, "
                   "expand[]=safetensors, config.json). See graph/fetch_quant_sizes.py docstring "
                   "for discovery and acceptance rules. GGUF bytes are whole-file sizes "
                   "(header + tensors), summed over split parts."),
        "gguf": gg, "safetensors": st, "rejected": rejected,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1, default=list)
    from collections import Counter
    print(f"gguf: {len(gg)} records over {len(gbases)} bases; safetensors: {len(st)}; rejected {len(rejected)}")
    print(" types:", dict(Counter(r["model_type"] for r in gg)))
    print(" levels:", dict(Counter(l for r in gg for l in r["files"])))
    print(" st methods:", dict(Counter(r["method"] for r in st)))


if __name__ == "__main__":
    main()
