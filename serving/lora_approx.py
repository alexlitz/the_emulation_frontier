"""How well does a truncated-SVD LoRA reproduce a full fine-tune's behaviour?

For each rank r, every q/k/v/o/gate/up/down delta is replaced by its rank-r SVD truncation, and the
result is scored against the fine-tune on the fine-tune's own greedy continuations of 16 prompts:
KL(ft || model) per token, top-1 agreement, and recovered = 1 - KL / KL(ft || base).

  A_provider: embeddings / norms / biases stay at the base (what serverless LoRA allows).
  B_upper:    they are copied from the fine-tune (the most any linear-only adapter could add).

  python -m serving.lora_approx Qwen/Qwen2.5-0.5B-Instruct Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

import torch
import torch.nn.functional as F

RANKS = (0, 8, 16, 32, 64, 128, 256)
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROMPTS = ["Explain why the sky is blue.", "Write a short poem about autumn.", "What is 17 * 23? Show your work.",
           "Give three tips for learning a new language.", "Summarize the plot of Romeo and Juliet.",
           "Write a Python function that reverses a string.", "What are the causes of inflation?",
           "Describe how a transformer neural network works.", "Translate 'good morning' into French and German.",
           "List the planets of the solar system in order.", "How do vaccines train the immune system?",
           "Write a haiku about the ocean.", "What is the difference between weather and climate?",
           "Give a recipe for a simple tomato soup.", "Why do we have seasons on Earth?",
           "Explain recursion to a ten-year-old."]


def continuations(ft, tok, n_new=64):
    seqs = []
    with torch.no_grad():
        for p in PROMPTS:
            try:
                ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                              return_tensors="pt")
            except Exception:                           # no chat template
                ids = tok(p, return_tensors="pt").input_ids
            seqs.append(ft.generate(ids, max_new_tokens=n_new, do_sample=False))
    return seqs


def score(model, ft, seqs):
    kl, agree, n = 0.0, 0, 0
    with torch.no_grad():
        for s in seqs:
            a, b = model(s).logits[0, :-1], ft(s).logits[0, :-1]
            V = min(a.shape[-1], b.shape[-1])           # trimmed vocabularies: compare the shared ids
            lp, lq = F.log_softmax(a[:, :V], -1), F.log_softmax(b[:, :V], -1)
            kl += (lq.exp() * (lq - lp)).sum().item()
            agree += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
            n += lp.shape[0]
    return kl / n, agree / n


def run(base_id, ft_id, ranks=RANKS):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    try:
        tok = AutoTokenizer.from_pretrained(ft_id)
    except Exception:                                   # tokenizer configs newer transformers can't parse
        tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32).eval()
    ft = AutoModelForCausalLM.from_pretrained(ft_id, torch_dtype=torch.float32).eval()
    seqs = continuations(ft, tok)
    fsd, bsd = ft.state_dict(), base.state_dict()
    svd = {k: torch.linalg.svd((w - bsd[k]).double(), full_matrices=False)
           for k, w in fsd.items() if k.endswith(".weight") and any(t in k for t in LINEAR)}
    kl0, _ = score(base, ft, seqs)
    rows = []
    for variant in ("A_provider", "B_upper"):
        for r in ranks:
            m = copy.deepcopy(base)
            sd = m.state_dict()
            for k in sd:
                if k in svd:
                    U, S, Vh = svd[k]
                    sd[k] = (bsd[k].double() + (U[:, :r] * S[:r]) @ Vh[:r]).float() if r else bsd[k]
                elif variant == "B_upper":
                    if fsd[k].shape == sd[k].shape:
                        sd[k] = fsd[k]
                    else:                               # trimmed vocabulary: copy the shared rows
                        sd[k] = sd[k].clone()
                        sd[k][:fsd[k].shape[0]] = fsd[k]
            m.load_state_dict(sd)
            kl, ag = score(m, ft, seqs)
            rows.append(dict(variant=variant, rank=r, kl=kl, agree=ag, recovered=1 - kl / kl0))
            print(f"{variant} r{r}: KL {kl:.4f}  top1 {ag:.3f}  recovered {1 - kl / kl0:.1%}", file=sys.stderr, flush=True)
    return {"base": base_id, "ft": ft_id, "kl_base": kl0, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("ft")
    a = ap.parse_args()
    print(json.dumps(run(a.base, a.ft)))


if __name__ == "__main__":
    main()
