"""The transformers reference a recipe is validated against (run in the repo's own environment, fp32):
prompt logits, the greedy continuation, and the logits at each greedy step.

    python -m recipes.reference TARGET OUT.npz [--ids 1,2,3 | --prompt TEXT] [--new-tokens 32]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

PROMPT = "The capital of France is"


def reference(target, ids=None, prompt=PROMPT, new_tokens=32, attn="eager", fp64=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    if ids is None:
        ids = AutoTokenizer.from_pretrained(target)(prompt).input_ids
    m = AutoModelForCausalLM.from_pretrained(target, dtype=torch.float32, attn_implementation=attn).eval()
    x = torch.tensor([ids])
    with torch.no_grad():
        out = m.generate(x, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, num_beams=1,
                         pad_token_id=0, eos_token_id=None)
        full = out[0].tolist()
        logits = m(torch.tensor([full[:-1]]), use_cache=False).logits[0].float().numpy()
    T = len(ids)
    new = full[T:]
    out = {"ids": np.array(ids), "new": np.array(new), "prefill": logits[:T],
           "steps": logits[T - 1:T - 1 + len(new)]}
    if fp64:                                            # the exact value the fp32 runs approximate
        del m
        m = AutoModelForCausalLM.from_pretrained(target, dtype=torch.float64, attn_implementation=attn).eval()
        with torch.no_grad():
            out["prefill64"] = m(x, use_cache=False).logits[0].numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("out")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--new-tokens", type=int, default=32)
    a = ap.parse_args(argv)
    ids = [int(t) for t in a.ids.split(",")] if a.ids else None
    r = reference(a.target, ids, a.prompt, a.new_tokens)
    np.savez(a.out, **r)
    print(a.target, "ids", r["ids"].tolist(), "->", r["new"].tolist())


if __name__ == "__main__":
    main()
