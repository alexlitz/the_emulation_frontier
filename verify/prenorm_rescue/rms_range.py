"""Per-layer token RMS at every residual-width norm input (pre-norms, final norm; for OLMo-2 the
post-norms, whose inputs are the sublayer OUTPUTS). RMS is the target's own normaliser:
sqrt(mean(x^2) + eps) for RMSNorm, sqrt(var(x) + eps) for LayerNorm (the centred stream).
float32, ~32 wikitext sequences x <=128 tokens.

Run: PYENV_VERSION=3.12.6 python -m verify.prenorm_rescue.rms_range [model ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import harness  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = ["EleutherAI/pythia-160m", "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen3-0.6B",
          "meta-llama/Llama-3.2-1B", "allenai/OLMo-2-0425-1B", "allenai/OLMoE-1B-7B-0924",
          "AlexKoff88/gpt-oss-tinystories"]
QS = {"min": 0, "p0.1": .1, "p1": 1, "p50": 50, "p99": 99, "p99.9": 99.9, "max": 100}


def stats(a):
    q = {k: float(np.percentile(a, v)) for k, v in QS.items()}
    q["D_minmax"] = q["max"] / q["min"]
    q["D_p"] = q["p99.9"] / q["p0.1"]
    return q


def token_rms(module, x):
    eps = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or 1e-5
    x = x.float()
    if isinstance(module, torch.nn.LayerNorm):
        return torch.sqrt(x.var(-1, unbiased=False) + eps)
    return torch.sqrt(x.pow(2).mean(-1) + eps)


@torch.no_grad()
def run(name, device="cuda:0"):
    dtype = torch.bfloat16 if "OLMoE" in name else torch.float32
    model, tok, dev = harness.load(name, dtype=dtype, device=device)
    H = model.config.hidden_size
    batches = harness.eval_batches(tok, dev, max_length=128, num_samples=32)
    rec, hooks = {}, []
    for mname, m in model.named_modules():
        cls = type(m).__name__
        if "Norm" not in cls or "q_norm" in mname or "k_norm" in mname:
            continue
        w = getattr(m, "weight", None)
        if w is None or w.shape[-1] != H:
            continue

        def hook(mod, inp, _out, mname=mname):
            r = token_rms(mod, inp[0])[0]          # (T,)
            rec.setdefault(mname, []).append(r.cpu().numpy())
        hooks.append(m.register_forward_hook(hook))
    for ids in batches:
        model(ids)
    for h in hooks:
        h.remove()
    res = {}
    for mname, lst in rec.items():
        allp = np.concatenate(lst)
        no0 = np.concatenate([a[1:] for a in lst])
        res[mname] = {"all": stats(allp), "no_pos0": stats(no0)}
    out = {"model": name, "dtype": str(dtype), "n_seq": len(batches),
           "n_tok": int(sum(b.shape[1] for b in batches)), "d": H, "norms": res}
    fn = os.path.join(OUT, "rms_" + name.replace("/", "__") + ".json")
    json.dump(out, open(fn, "w"), indent=1)
    worst = max(res.items(), key=lambda kv: kv[1]["no_pos0"]["D_minmax"])
    print(f"{name}: {len(res)} norms; worst (no pos0) {worst[0]} D={worst[1]['no_pos0']['D_minmax']:.2f}"
          f" Dp={worst[1]['no_pos0']['D_p']:.2f}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for n in (sys.argv[1:] or MODELS):
        try:
            run(n)
        except Exception as e:  # report and continue
            print(f"{n}: FAILED {type(e).__name__}: {e}")
