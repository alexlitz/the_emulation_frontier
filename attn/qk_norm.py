"""Simulating QK-norm (RMSNorm on Q,K inside attention).

QK-norm replaces the score  q.k  with  (q/||q||).(k/||k||) * gq*gk  -- it divides
out the per-token Q,K magnitude. Fixed weights can't reproduce an input-dependent
divide, so at the EXACT level a no-QK-norm model can't emulate a QK-norm one
(that's why the graph marks it must-match). BUT:

  RMSNorm(q_t) = q_t / rms(q_t).  If rms(q_t) is ~CONSTANT across tokens t (call
  it c_q), then RMSNorm folds into a constant scale 1/c_q, which absorbs into the
  W_q weights -- and QK-norm attention becomes ordinary attention with a fixed
  per-head scale. So QK-norm is APPROXIMATELY foldable, and the emulation currency
  is the token-to-token VARIANCE of ||q|| (its coefficient of variation, CV).

Part 1: exact gap (QK-norm != plain attention on random q,k).
Part 2: the fold -- residual of the constant-scale emulation vs the CV of ||q||.
Part 3 (optional): measure the real CV of ||q|| in a QK-norm model.

Run:
  python -m attn.qk_norm
  python -m attn.qk_norm --model <a Qwen3 / OLMo-2 checkpoint>
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def attend(scores):
    T = scores.shape[0]
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=scores.dtype), 1)
    return F.softmax(scores + mask, dim=-1)


def make_qk(T, hd, cv, seed):
    """q,k with unit-ish directions but per-token magnitude coeff-of-variation cv."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    qd, kd = F.normalize(r(T, hd), dim=-1), F.normalize(r(T, hd), dim=-1)
    mag_q = (1.0 + cv * r(T, 1)).abs()                   # per-token magnitude
    mag_k = (1.0 + cv * r(T, 1)).abs()
    return qd * mag_q, kd * mag_k


def part1_exact_gap():
    q, k = make_qk(48, 64, cv=0.6, seed=0)
    a_qk = attend(rmsnorm(q) @ rmsnorm(k).T)
    a_plain = attend((q @ k.T) / math.sqrt(64))
    print("1. exact: QK-norm attention != plain attention")
    print(f"     mean |A_qknorm - A_plain| = {(a_qk - a_plain).abs().mean().item():.3e}  "
          "(different attention -> must-match at exact level)\n")


def part2_fold_vs_variance(T=48, hd=64):
    print("2. the fold: emulate QK-norm with a CONSTANT per-head scale; residual vs CV(||q||):")
    print(f"     {'CV(||q||)':>10}  {'residual (attn weights)':>24}")
    for cv in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        errs = []
        for seed in range(6):
            q, k = make_qk(T, hd, cv, seed)
            a_qk = attend(rmsnorm(q) @ rmsnorm(k).T)                  # true QK-norm
            cq = q.pow(2).mean(-1).sqrt().mean()                      # constant fold
            ck = k.pow(2).mean(-1).sqrt().mean()
            a_fold = attend((q / cq) @ (k / ck).T)                   # plain attn, fixed scale
            errs.append((a_qk - a_fold).abs().mean().item())
        print(f"     {cv:>10.2f}  {sum(errs) / len(errs):>24.3e}")
    print("   -> at CV=0 the fold is EXACT (QK-norm = a fixed scale); the residual grows")
    print("      with magnitude variance. So QK-norm is an APPROX edge, penalty = CV(||q||).\n")


def part3_real(model_name):
    import harness as H
    model, tok, device = H.load(model_name, dtype=torch.float32)
    # find q_norm modules (Qwen3 / OLMo-2 name them q_norm inside attention)
    qnorms = [(n, m) for n, m in model.named_modules() if n.endswith("q_norm")]
    if not qnorms:
        print(f"3. {model_name}: no q_norm modules found (not a QK-norm model?)\n")
        return
    cvs = []

    def hook(mod, inp, out):
        x = inp[0].detach().float()                      # (..., hd) pre-norm q per head
        rms = x.pow(2).mean(-1).sqrt().flatten()         # per-token rms
        cvs.append((rms.std() / (rms.mean() + 1e-9)).item())

    handles = [m.register_forward_hook(hook) for _, m in qnorms]
    batches = H.eval_batches(tok, device, max_length=64, num_samples=8)
    with torch.no_grad():
        for ids in batches:
            model(ids)
    for h in handles:
        h.remove()
    import statistics
    print(f"3. {model_name}: measured CV(||q||) over {len(qnorms)} q_norm layers")
    print(f"     mean CV = {statistics.mean(cvs):.3f}   (low CV -> QK-norm nearly foldable)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    part1_exact_gap()
    part2_fold_vs_variance()
    if args.model:
        try:
            part3_real(args.model)
        except Exception as e:
            print(f"3. could not load {args.model} ({type(e).__name__}); synthetic parts stand.")
