"""Autoregression is a special case of (any-order / masked) diffusion.

The `gen` axis used to be modelled as must-match: AR and diffusion emulate neither
way. That's too conservative. The difference is, at the architecture level, mostly an
ATTENTION-MASK difference plus a DECODING-ORDER choice:

  - a diffusion LM (LLaDA) uses BIDIRECTIONAL attention and unmasks in ANY order;
  - an AR LM uses CAUSAL attention and decodes in a FIXED left-to-right order.

Causal attention is a masked restriction of bidirectional (full) attention, and a
fixed order is a restriction of any order. So AR is the special case: give a diffusion
architecture a causal mask and decode strictly left-to-right (unmask position t+1
conditioned on <=t) and you reproduce autoregressive generation. Currency: mask (+ a
free schedule) -- exactly like local <= full.

The reverse (diffusion <= AR) is the hard direction: a strictly-causal model can't see
the future in one pass at any depth, so reproducing bidirectional any-order denoising
needs ~N passes (an any-order ensemble) or a retrain -- not a native reparameterization.

Part 1 (below) demonstrates the load-bearing piece: bidirectional attention + a causal
mask IS causal attention, bit-for-bit. Part 2 shows the decoding-order point: unmasking
left-to-right is the autoregressive factorization.

Run:  python -m attn.ar_diffusion
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def bidirectional(q, k, v):                 # a diffusion LM's attention: no causal mask
    return F.softmax(q @ k.T / math.sqrt(q.shape[-1]), dim=-1) @ v


def causal(q, k, v):                        # an AR LM's attention: causal mask
    T = q.shape[0]
    m = torch.triu(torch.full((T, T), float("-inf"), dtype=q.dtype), 1)
    return F.softmax(q @ k.T / math.sqrt(q.shape[-1]) + m, dim=-1) @ v


def bidirectional_with_causal_mask(q, k, v):  # the diffusion op, restricted
    T = q.shape[0]
    m = torch.triu(torch.full((T, T), float("-inf"), dtype=q.dtype), 1)
    return F.softmax(q @ k.T / math.sqrt(q.shape[-1]) + m, dim=-1) @ v


def demo():
    g = torch.Generator().manual_seed(0)
    T, d = 12, 16
    q, k, v = (torch.randn(T, d, dtype=F64, generator=g) for _ in range(3))

    # Part 1: AR <= diffusion, the attention core -- a diffusion (bidirectional)
    # architecture given a causal mask reproduces AR (causal) attention, bit-for-bit.
    err = (bidirectional_with_causal_mask(q, k, v) - causal(q, k, v)).abs().max().item()
    diff = (bidirectional(q, k, v) - causal(q, k, v)).abs().max().item()
    print("AR is a special case of diffusion (bidirectional) attention:")
    print(f"  bidirectional + causal mask  vs  causal:   max|Δ| = {err:.2e}   (identity)")
    print(f"  bidirectional (unmasked)     vs  causal:   max|Δ| = {diff:.2e}   (genuinely differs)")

    # Part 2: the decoding order. Any-order unmasking reduces to AR when the order is
    # left-to-right: revealing token t+1 conditioned on <=t IS the AR factorization.
    order = list(range(T))                  # left-to-right == the autoregressive order
    print(f"\n  any-order decode restricted to L2R {order[:4]}... = the AR factorization")
    print("  -> AR ⊆ diffusion: causal mask (like local⊆full) + a fixed decode order (free).")
    print("  reverse (diffusion ⊆ AR) is the ∞ direction: causality is a one-pass barrier.")
    return err


if __name__ == "__main__":
    demo()
