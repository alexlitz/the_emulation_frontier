"""Does the low-rank part of a fine-tune carry its behaviour? GPU version of lora_approx for 8B models.

The fine-tune runs on one GPU (weights range-read from the Hub, nothing written to disk); the base
runs on another with a LoRA applied as forward hooks, the way a LoRA server computes it:
y = W_base x + U_r S_r V_r^T x in fp32, with U S V^T the top-r SVD of each weight delta. Scored by
KL(ft || variant) per token on the fine-tune's own greedy continuations (generic + judge-style
prompts), against KL(ft || base).

  add:    base + rank-r adapter on the linears (q/k/v/o/gate/up/down), optionally lm_head too.
  remove: the fine-tune minus its rank-r part. If that falls back to the base's KL, the rank-r
          part is what makes the fine-tune differ; if it stays near 0, the behaviour lives in the
          remainder (e.g. a sub-ulp full-rank update).

  python -m serving.lora_behaviour meta-llama/Llama-3.1-8B-Instruct AtlaAI/Selene-1-Mini-Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from serving.lora_approx import PROMPTS
from serving.st_reader import Repo

RANKS = (0, 1, 2, 4, 8, 16, 64)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
JUDGE = [
    "Evaluate the following response for factual accuracy on a scale of 1-5 and explain your score.\n"
    "Question: What is the boiling point of water at sea level?\nResponse: 90 degrees Celsius.",
    "You are a strict grader. Score this answer from 1 to 10.\nQ: Name the largest planet.\nA: Jupiter, "
    "a gas giant about 11 times Earth's diameter.",
    "Which response is more helpful, A or B? Explain briefly.\nUser: How do I boil an egg?\nA: Put it in "
    "water.\nB: Place the egg in boiling water for 9 minutes, then cool it in cold water.",
    "Rate the harmlessness of this reply (1-5): 'I can't help with making weapons, but I can explain "
    "the history of chemistry.'",
    "Critique this code for bugs and give a score out of 10:\ndef avg(xs): return sum(xs) / len(xs)",
    "Is the following summary faithful to the text? Answer Yes or No with reasoning.\nText: The meeting "
    "moved to Tuesday.\nSummary: The meeting was cancelled.",
    "Grade the essay opening for clarity (1-5): 'In todays world, technology is everywhere and it "
    "effects everyone alot.'",
    "Does this response follow the instruction 'answer in one word'? Response: 'Paris is the capital.'",
]


def load_pair(base_id, ft_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    ft = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16).to("cuda:1").eval()
    R = Repo(ft_id)
    names = {n: n for n in ft.state_dict()}
    canon = {n.removeprefix("model."): n for n in names}
    with ThreadPoolExecutor(8) as ex:
        tensors = dict(ex.map(lambda c: (c, R.read(c)), [c for c in R.idx if c in canon]))
    missing = set(canon) - set(tensors)
    sd = {canon[c]: t.to(torch.bfloat16) for c, t in tensors.items()}
    ft.load_state_dict(sd, strict=False)
    if missing:
        print(f"kept base values for {sorted(missing)[:5]}", file=sys.stderr)
    return tok, base, ft


def deltas(base, ft, rmax=max(RANKS), head=True):
    """Top-rmax SVD factors of every target delta (and lm_head): name -> (US [out,r], V [in,r]).
    Computed on the CPU so the GPUs hold only the two models and the small factors."""
    fsd = ft.state_dict()
    out = {}
    for name, mod in base.named_modules():
        if isinstance(mod, torch.nn.Linear) and (name.split(".")[-1] in TARGETS or (head and name == "lm_head")):
            d = fsd[name + ".weight"].cpu().float() - mod.weight.cpu().float()
            if d.abs().max() == 0:
                continue
            out[name] = top_factors(d, rmax)
    return out


def top_factors(d, r, chunk=16384):
    """(U S [out, r], V [in, r]) of the top-r SVD of d, from the eigenvectors of the smaller Gram
    matrix. Randomized SVD and cuSOLVER's eigh both fail to converge on exactly low-rank deltas, so
    the eigendecomposition runs in CPU LAPACK; the Gram matrix is accumulated on the GPU in row
    chunks (double sums of fp32 products), so even a 128k-row lm_head needs ~300 MB extra."""
    wide = d.shape[0] < d.shape[1]
    m = d.T if wide else d                               # tall view: Gram is (cols x cols), the small side
    g = torch.zeros(m.shape[1], m.shape[1], dtype=torch.float64, device=d.device)
    for i in range(0, m.shape[0], chunk):
        c = m[i:i + chunk].float()
        g += (c.T @ c).double()
    _, vecs = torch.linalg.eigh(g.cpu())
    W = vecs[:, -r:].flip(1).float().to(d.device).contiguous()      # top right vectors of m
    if not wide:
        return (d @ W).contiguous(), W
    V = d.T @ W                                          # m = d^T: its right vectors are d's left ones
    n = V.norm(dim=0)
    V = (V / n.clamp_min(1e-30) * (n > 1e-6 * n.max())).contiguous()   # past the true rank: zero, not noise
    return (d @ V).contiguous(), V


class Adapter:
    """Forward hooks adding sign * U_r S_r V_r^T x to each adapted Linear of `model`."""

    def __init__(self, model, factors, device):
        self.r, self.sign, self.use_head, self.hooks = 0, 1.0, False, []
        f = {n: (us.to(device), v.to(device)) for n, (us, v) in factors.items()}
        for name, mod in model.named_modules():
            if name in f:
                self.hooks.append(mod.register_forward_hook(self._hook(name, *f[name])))

    def _hook(self, name, us, v):
        def h(mod, inp, out):
            if self.r == 0 or (name == "lm_head" and not self.use_head):
                return out
            x = inp[0].float()
            return out + (self.sign * (x @ v[:, :self.r]) @ us[:, :self.r].T).to(out.dtype)
        return h


def continuations(tok, ft, n_new=64):
    seqs = []
    with torch.no_grad():
        for p in PROMPTS[:8] + JUDGE:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)["input_ids"].to("cuda:1")
            seqs.append(ft.generate(ids, max_new_tokens=n_new, do_sample=False,
                                    attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id))
    return seqs


def ref_logprobs(ft, seqs):
    with torch.no_grad():
        return [F.log_softmax(ft(s).logits[0, :-1].float(), -1).cpu() for s in seqs]


def score(model, seqs, refs, device):
    kl, agree, n = 0.0, 0, 0
    with torch.no_grad():
        for s, lq in zip(seqs, refs):
            # via the CPU: direct GPU-to-GPU copies (P2P) silently return zeros on this machine
            lp = F.log_softmax(model(s.cpu().to(device)).logits[0, :-1].float(), -1).cpu()
            kl += (lq.exp() * (lq - lp)).sum().item()
            agree += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
            n += lp.shape[0]
    return kl / n, agree / n


def run(base_id, ft_id):
    tok, base, ft = load_pair(base_id, ft_id)
    seqs = continuations(tok, ft)
    refs = ref_logprobs(ft, seqs)
    fac = deltas(base, ft)
    print(f"factors done; peak GPU memory {[torch.cuda.max_memory_allocated(i) >> 20 for i in range(2)]} MiB",
          file=sys.stderr, flush=True)
    kl0, ag0 = score(base, seqs, refs, "cuda:0")
    rows = [dict(variant="base", rank=0, kl=kl0, agree=ag0, recovered=0.0)]
    print(f"base: KL {kl0:.5f} top1 {ag0:.3f}  ({len(fac)} adapted matrices)", file=sys.stderr, flush=True)
    add, rem = Adapter(base, fac, "cuda:0"), Adapter(ft, fac, "cuda:1")
    rem.sign = -1.0
    for variant, ad, model, dev in (("add_linear", add, base, "cuda:0"), ("add_linear+head", add, base, "cuda:0"),
                                    ("remove_from_ft", rem, ft, "cuda:1")):
        ad.use_head = variant != "add_linear"
        for r in RANKS[1:]:
            ad.r = r
            kl, ag = score(model, seqs, refs, dev)
            rows.append(dict(variant=variant, rank=r, kl=kl, agree=ag, recovered=1 - kl / kl0))
            print(f"{variant} r{r}: KL {kl:.5f} top1 {ag:.3f} recovered {1 - kl / kl0:.1%}", file=sys.stderr, flush=True)
        ad.r = 0
    return {"base": base_id, "ft": ft_id, "kl_base": kl0, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("ft")
    a = ap.parse_args()
    print(json.dumps(run(a.base, a.ft)))


if __name__ == "__main__":
    main()
