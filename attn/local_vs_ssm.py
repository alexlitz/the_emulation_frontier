"""Local attention and SSMs as DUAL restrictions of full attention.

Full softmax attention:   unbounded reach, full resolution, O(N) cost.
Local (sliding-window W):  bounded REACH (last W tokens), full resolution -> fails LONG-RANGE.
Linear / SSM (state d):    unbounded reach, bounded RESOLUTION (fixed state)  -> fails at LOAD.

So local and SSM compress orthogonal axes of full attention:
  - local trades away *reach*      (can't see past the window)
  - SSM/linear trades away *resolution* (a fixed-capacity superposition state)
Both are <= full attention (local by masking; linear/SSM at the softmax near-gap),
and they are INCOMPARABLE to each other -- which is exactly why hybrids (Jamba,
Zamba, Nemotron-H) interleave local/SSM layers for cheap bulk with a few full
attention layers for the exact retrieval (induction / copying) neither can do.

Probe: a sequence of (key,value) pairs; query = one key, target = its value.
Measure retrieval error for each mixer vs (A) target distance and (B) load N.

Run:  python -m attn.local_vs_ssm
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def make(N, d, target, seed):
    g = torch.Generator().manual_seed(seed)
    K = F.normalize(torch.randn(N, d, dtype=F64, generator=g), dim=-1)   # keys
    V = torch.randn(N, d, dtype=F64, generator=g)                        # values
    q = K[target]                                                        # query = target key
    return K, V, q, V[target]


def full(K, V, q, beta=40):
    a = F.softmax(beta * (K @ q), dim=0)          # sharp softmax select
    return a @ V


def local(K, V, q, W, pos):                       # window: last W positions [pos-W+1, pos]
    s = K @ q
    mask = torch.arange(K.shape[0]) <= pos
    mask &= torch.arange(K.shape[0]) > pos - W
    s = s.masked_fill(~mask, float("-inf"))
    return F.softmax(40 * s, dim=0) @ V


def linear_ssm(K, V, q):                          # superposition memory (no softmax)
    return (K @ q) @ V                            # sum_i (q.k_i) v_i = q^T (sum k_i v_i^T)


def relerr(out, tgt):
    return ((out - tgt).norm() / tgt.norm()).item()


def sweep_reach(N=192, d=128, W=32):
    print(f"A. REACH: retrieval error vs target distance (N={N}, d={d}, window W={W}):")
    print(f"     {'distance':>9}  {'full':>9}  {'local(W)':>9}  {'linear/SSM':>11}")
    for dist in [1, 8, 24, 48, 96, 160]:
        target = N - 1 - dist
        if target < 0:
            continue
        ef = el = es = 0.0
        for seed in range(8):
            K, V, q, tgt = make(N, d, target, seed)
            ef += relerr(full(K, V, q), tgt)
            el += relerr(local(K, V, q, W, N - 1), tgt)
            es += relerr(linear_ssm(K, V, q), tgt)
        print(f"     {dist:>9}  {ef/8:>9.3f}  {el/8:>9.3f}  {es/8:>11.3f}")
    print("   -> local is exact inside the window, useless beyond it (bounded REACH).\n")


def sweep_load(d=128, W=256):
    print(f"B. RESOLUTION: retrieval error vs load N (d={d}, target at distance 4):")
    print(f"     {'N':>6}  {'full':>9}  {'local(W)':>9}  {'linear/SSM':>11}")
    for N in [16, 32, 64, 128, 256, 512]:
        target = N - 1 - 4
        ef = el = es = 0.0
        for seed in range(8):
            K, V, q, tgt = make(N, d, target, seed)
            ef += relerr(full(K, V, q), tgt)
            el += relerr(local(K, V, q, W, N - 1), tgt)
            es += relerr(linear_ssm(K, V, q), tgt)
        print(f"     {N:>6}  {ef/8:>9.3f}  {el/8:>9.3f}  {es/8:>11.3f}")
    print("   -> linear/SSM degrades as N grows (fixed-capacity state; bounded RESOLUTION);")
    print("      full stays exact. local is fine here (target is within the window).")


if __name__ == "__main__":
    sweep_reach()
    sweep_load()
