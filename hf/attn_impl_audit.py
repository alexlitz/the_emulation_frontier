"""eager vs sdpa on every model an agreement / measurement harness recorded (transformers 4.57).

For each checkpoint: the implementation transformers picks by default, and the sdpa model scored against
the eager model with the act_swap standard (act/act_swap_agreement.py: the eager model's own greedy
64-token continuations of the 24 prompts, fp32, TF32 off): top-1, 32-token identical continuations,
max |delta log p| teacher-forced. A difference at the fp32 rounding level (max |dlogp| ~1e-5 .. 1e-3,
100% / ~100%) is two kernels computing the same model; a large one is a feature one path drops
(hf/attn_pin.py explains which, from the code). Also two checkpoints no harness recorded, to show the
two known bugs: stanford-crfm/alias-gpt2-small-x21 (GPT-2 with scale_attn_by_inverse_layer_idx) and
tiiuae/falcon-rw-1b (Falcon with ALiBi).

  python -m hf.attn_impl_audit --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from act.act_swap_agreement import PROMPTS
from hf.attn_pin import attn_impl
from hf.eps_lever_agreement import reference, score, strict

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "attn_impl_audit.jsonl")
# model -> the harnesses (and branches) that recorded it
RECORDED = {
    "gpt2": "act_swap, act_width (act-width-fits), bos_constructions (bos-token)",
    "gpt2-medium": "act_swap, act_width",
    "EleutherAI/pythia-160m": "act_swap, act_width",
    "EleutherAI/pythia-410m": "act_swap, act_width",
    "EleutherAI/pythia-1.4b": "act_swap, act_width",
    "facebook/opt-125m": "act_swap, bos_constructions",
    "facebook/opt-350m": "act_swap",
    "facebook/opt-1.3b": "act_swap",
    "HuggingFaceTB/SmolLM2-135M": "act_swap, act_width, regime (regime-agreement), looped_host_prenorm (moe-loop-rescue)",
    "Qwen/Qwen2.5-0.5B": "act_swap, act_width, fold, regime, looped_host_prenorm",
    "Qwen/Qwen2.5-1.5B": "act_swap, act_width, fold",
    "meta-llama/Llama-3.2-1B": "act_swap, act_width, regime",
    "google/gemma-2-2b": "act_swap, act_width, fold, regime",
    "google/gemma-2-2b-it": "regime",
    "google/gemma-3-270m": "regime",
    "Qwen/Qwen3-0.6B": "fold, regime",
    "Qwen/Qwen3-1.7B": "fold",
    "allenai/OLMo-2-0425-1B": "fold, regime",
    "AlexKoff88/gpt-oss-tinystories": "act_swap, act_width, fold, bos_constructions, dense_host_prenorm",
    "swiss-ai/Apertus-v1.1-0.5B": "act_swap",
    "nanochat-students/nanochat-d20": "act_swap",
    "huihui-ai/Huihui-MoE-0.8B-2E": "dense_host_prenorm",
    "ggml-org/stories15M_MOE": "dense_host_prenorm",
    "allenai/OLMoE-1B-7B-0924": "dense_host_prenorm",
    "LiquidAI/LFM2-350M": "mixer_gap",
    "Qwen/Qwen3.5-0.8B-Base": "mixer_gap",
    "ibm-granite/granite-4.0-h-350m-base": "mixer_gap",
    "stanford-crfm/alias-gpt2-small-x21": "none (demonstrates the GPT-2 layer-idx bug)",
    "tiiuae/falcon-rw-1b": "none (demonstrates the Falcon ALiBi bug)",
}
# not run here: deepseek-ai/DeepSeek-V2-Lite (mla_norm; 64 GB in fp32, CPU only -- DeepseekV2Attention
# passes only `scaling`, which sdpa honours), google/gemma-2-9b and gemma-2-2b-jpn-it (regime; same code
# as gemma-2-2b), RWKV / Mamba (mixer_gap; no softmax attention).


def default_impl(model_id):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id)
    try:
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(cfg)
        return m.config._attn_implementation, cfg
    except Exception:                               # e.g. xIELU's init cannot run on the meta device
        return "sdpa" if AutoModelForCausalLM._model_mapping[type(cfg)]._supports_sdpa else "eager", cfg


def load(model_id, impl, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    for _ in range(30):
        try:
            return tok, AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                             attn_implementation=impl).to(device).eval()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            time.sleep(60)
    raise RuntimeError("OOM")


def audit(model_id, device, n_prompts=24):
    dflt, cfg = default_impl(model_id)
    tok, m = load(model_id, "eager", device)
    ref = reference(m, tok, device, n_prompts)
    del m
    torch.cuda.empty_cache()
    try:
        _, m = load(model_id, "sdpa", device)
    except ValueError as e:                         # no sdpa path: eager is the only (and default) one
        return dict(model=model_id, recorded_by=RECORDED.get(model_id, ""), default_impl=dflt,
                    pinned=attn_impl(cfg), sdpa_supported=False, note=str(e)[:120])
    r = score(m, ref, device)
    del m
    torch.cuda.empty_cache()
    flags = {k: getattr(cfg, k) for k in ("scale_attn_by_inverse_layer_idx", "scale_attn_weights",
                                         "reorder_and_upcast_attn", "attn_logit_softcapping", "alibi")
             if getattr(cfg, k, None) is not None}
    return dict(model=model_id, recorded_by=RECORDED.get(model_id, ""), default_impl=dflt, pinned=attn_impl(cfg),
                flags=flags, ref="eager", variant="sdpa", same_model=strict(r) and r["max_dlogp"] < 1e-2, **r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(RECORDED))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for mid in a.models:
        t0 = time.time()
        try:
            row = audit(mid, a.device)
        except Exception as e:                      # record and go on (remote code, missing weights)
            row = dict(model=mid, recorded_by=RECORDED.get(mid, ""), error=f"{type(e).__name__}: {str(e)[:200]}")
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{mid:40s} default {row.get('default_impl')!s:6s} top1 {row.get('top1', float('nan')):.4f} "
              f"ex32 {row.get('ex32', float('nan')):.2f} max|dlogp| {row.get('max_dlogp', float('nan')):.2e} "
              f"{row.get('error', '')} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
