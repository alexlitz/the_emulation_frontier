"""Simulating logit soft-capping (Gemma-2).

Soft-capping replaces a logit x with  softcap(x) = c * tanh(x / c)  (c the cap,
e.g. 50 for attention, 30 for final logits). For |x| << c this is tanh(u)~=u, so
softcap(x) ~= x -- an IDENTITY that folds away. It only bites when logits approach
the cap (saturation). So, like QK-norm, softcap is NOT a hard must-match: a plain
(no-softcap) model approximately emulates a softcap one, with the penalty growing
in the saturation ratio max|logit| / c.

Measures the residual of the plain emulation vs the saturation ratio.

Run:  python -m attn.softcap
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from attn.qk_norm import attend

F64 = torch.float64


def softcap(x, c):
    return c * torch.tanh(x / c)


def run(T=48, hd=64):
    print("softcap(x)=c*tanh(x/c): residual of the plain (no-softcap) emulation vs")
    print("saturation ratio max|logit|/c  (c=1; logit std set to the ratio):")
    print(f"     {'max|logit|/c':>14}  {'residual (attn weights)':>24}")
    for ratio in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        errs = []
        for seed in range(6):
            g = torch.Generator().manual_seed(seed)
            q = torch.randn(T, hd, dtype=F64, generator=g)
            k = torch.randn(T, hd, dtype=F64, generator=g)
            scores = (q @ k.T) / math.sqrt(hd)
            scores = scores / scores.abs().max() * ratio          # scale so max|logit|=ratio, c=1
            a_plain = attend(scores)                              # no-softcap model
            a_cap = attend(softcap(scores, 1.0))                 # softcap model
            errs.append((a_plain - a_cap).abs().mean().item())
        print(f"     {ratio:>14.2f}  {sum(errs) / len(errs):>24.3e}")
    print("   -> in the unsaturated regime (logits << cap) softcap ~= identity and folds")
    print("      away; residual grows only as logits approach the cap. Approx edge, not a wall.")
    print("   (Gemma-2 uses cap=50 for attention logits, which are typically O(1-10) ->")
    print("    mostly the linear regime, so its softcap is ~a no-op in the bulk.)")


if __name__ == "__main__":
    run()
