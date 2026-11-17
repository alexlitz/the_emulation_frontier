"""Cross-layer KV sharing (CLA / YOCO) is a RESTRICTION of standard attention.

A CLA layer does not compute its own K,V -- it REUSES an earlier layer's K,V
tensors (the memory saving). A standard stack computes independent K,V per layer,
so it cannot reproduce CLA by a weight tie alone (the hidden states differ). But it
CAN reproduce it exactly by CARRYING the shared K,V forward in reserved residual
channels and having the reusing layer read them out -- a WIDTH cost, not free and
not a KV tie. This is the honest currency for the CLA edge.

We show: a standard single-head layer whose k/v projections select carried channels
holding layer-0's K,V produces exactly the CLA layer's output.

SCOPE: this is the NORM-FREE toy. It hands layer 1 an augmented hidden [h1, K0, V0] directly
and reads it with no normalisation. In a real pre-norm layer (Llama input_layernorm) one rms is
taken over the WHOLE residual, carried channels included, so the read-back K,V (and layer 1's
own queries) come out rescaled per token. That is NOT exact at any width: on stock
LlamaDecoderLayers the copy-stream construction is off by ~1e-2. It becomes exact only with a
norm that normalises each channel group separately, which is a norm change, not width. See
block/prenorm_carry.py for the stock-layer test and the impossibility argument.
Two further gaps: (i) no stock sublayer writes K0 = W_k N(h0) into the residual. The real
construction carries a copy of the layer-0 INPUT h0 (the embedding writes it twice) and
recomputes K,V from it. (ii) Every sharing pair needs fresh width (see block/prenorm_carry.py).

Run:  python -m attn.cross_layer_kv
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def attend(q, k, v):
    T = q.shape[0]
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=F64), 1)
    return F.softmax(q @ k.T / math.sqrt(q.shape[1]) + mask, dim=-1) @ v


def build(T=12, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    h0 = r(T, d)
    Wk0, Wv0 = r(d, d), r(d, d)                      # layer-0 K,V projections
    Wq1 = r(d, d)                                    # layer-1 query
    K0, V0 = h0 @ Wk0, h0 @ Wv0                      # the shared K,V (computed once)
    h1 = r(T, d)                                     # layer-1 hidden (differs from h0)

    # CLA layer 1: reuse layer-0's K0,V0 (never recompute)
    cla = attend(h1 @ Wq1, K0, V0)

    # standard reproduction: carry K0,V0 in reserved channels; layer-1 k/v projections
    # are SELECTORS that read them back out -> k1==K0, v1==V0. width cost = 2d channels.
    h1_aug = torch.cat([h1, K0, V0], dim=1)          # reserved residual channels
    Sk = torch.cat([torch.zeros(d, d), torch.eye(d), torch.zeros(d, d)], 0).to(F64)  # picks K0
    Sv = torch.cat([torch.zeros(d, d), torch.zeros(d, d), torch.eye(d)], 0).to(F64)  # picks V0
    Wq1_aug = torch.cat([Wq1, torch.zeros(2 * d, d)], 0).to(F64)                      # query ignores them
    std = attend(h1_aug @ Wq1_aug, h1_aug @ Sk, h1_aug @ Sv)
    return (cla - std).abs().max().item()


if __name__ == "__main__":
    print("cross-layer KV (CLA) as a restriction of standard attention:")
    print(f"  standard layer carrying K,V in reserved channels vs CLA layer:  "
          f"max|Δ| = {build():.2e}")
    print("  -> CLA subset standard, EXACT, currency = WIDTH (carry the shared K,V),")
    print("     not a free KV tie -- the hidden states differ, so it must be carried.")
