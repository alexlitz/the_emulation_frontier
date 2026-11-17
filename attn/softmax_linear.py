"""Softmax attention vs linear/kernel attention -- the ONE near-gap in the graph.

Linear attention replaces  softmax(QK^T) V  with  phi(Q) (phi(K)^T V), a kernel
feature map phi. Softmax attention is the kernel exp(q.k), whose exact feature
map is INFINITE-dimensional. So unlike every other pair, linear attention cannot
reproduce softmax exactly with finite resources -- it is exact only in a limit.
There are two different currencies for closing the gap, and they mirror the rest
of the graph:

  1. WIDTH (features): Performer / FAVOR+ approximates the softmax kernel with m
     random features; error decays ~1/sqrt(m). Exact only as m -> inf.

  2. MAGNITUDE (temperature): softmax's power is *hard selection* -- as the score
     scale beta -> inf it becomes a one-hot lookup (an exact argmax retrieval).
     A fixed-feature linear map cannot concentrate to one-hot, so it floors. This
     is the same magnitude-limit shape as ALiBi<=RoPE and ReLU<=Swish.

  3. DEPTH: one softmax layer can do associative recall / an induction head that
     linear-attention / SSMs need ~2 layers (or width ~ sequence length) for.
     (Documented; we show the single-layer selection mechanism that underlies it.)

Run:
  python -m attn.softmax_linear
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def favor_features(x, W):
    """Performer FAVOR+ positive random features: unbiased estimator of exp(q.k)."""
    # x: (T, d), W: (m, d) ~ N(0, I).  phi(x) = exp(Wx - ||x||^2/2) / sqrt(m)
    proj = x @ W.T                                   # (T, m)
    norm = (x ** 2).sum(-1, keepdim=True) / 2        # (T, 1)
    return torch.exp(proj - norm) / math.sqrt(W.shape[0])


def softmax_attention(Q, K, V, scale):
    A = F.softmax((Q @ K.T) * scale, dim=-1)
    return A @ V


def performer_attention(Q, K, V, W, scale):
    s = math.sqrt(scale)
    phiQ, phiK = favor_features(Q * s, W), favor_features(K * s, W)   # (T, m)
    num = phiQ @ (phiK.T @ V)                        # (T, dv)
    den = phiQ @ phiK.sum(0, keepdim=True).T         # (T, 1)
    return num / (den + 1e-6)


def feature_sweep(T=128, d=64, seed=0, repeats=8):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    # unit-norm Q,K keep the kernel argument q.k in [-1,1] so FAVOR+ variance is
    # controlled and the 1/sqrt(m) decay is visible (large ||q.k|| needs huge m).
    Q = F.normalize(rand(T, d), dim=-1)
    K = F.normalize(rand(T, d), dim=-1)
    V = rand(T, d)
    scale = 1.0
    exact = softmax_attention(Q, K, V, scale)
    print("1. WIDTH currency -- Performer random features approximate softmax:")
    print(f"     {'m features':>12}  {'rel err (output)':>18}")
    for m in [16, 64, 256, 1024, 4096, 16384]:
        errs = []
        for r in range(repeats):
            W = torch.randn(m, d, generator=g, dtype=torch.float64)
            approx = performer_attention(Q, K, V, W, scale)
            errs.append(((approx - exact).norm() / exact.norm()).item())
        print(f"     {m:>12}  {sum(errs) / len(errs):>18.3e}")
    print("   -> error ~ 1/sqrt(m); exact only as m -> inf (the infinite-feature wall)\n")


def selection_sharpness(T=64, d=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    # one query; key[target] aligned with query, the rest random unit vectors
    q = torch.zeros(d, dtype=torch.float64); q[0] = 1.0
    K = F.normalize(torch.randn(T, d, generator=g, dtype=torch.float64), dim=-1)
    target = 0
    K[target] = q
    scores = K @ q                                    # cosine-ish similarity
    print("2. MAGNITUDE currency -- softmax does exact hard selection; linear floors:")
    print(f"     {'beta':>8}  {'softmax mass on target':>24}")
    for beta in [1, 4, 16, 64, 256]:
        w = F.softmax(beta * scores, dim=0)
        print(f"     {beta:>8}  {w[target].item():>24.4f}")
    # linear attention (elu+1 feature map, Katharopoulos et al.), no temperature
    phi = lambda x: F.elu(x) + 1
    pq, pk = phi(q), phi(K)
    wlin = (pk @ pq); wlin = wlin / wlin.sum()
    print(f"     linear(elu+1) mass on target = {wlin[target].item():.4f}  "
          "(fixed features -> cannot concentrate)")
    print("   -> softmax -> one-hot as beta->inf (exact retrieval); linear attention")
    print("      has no temperature and floors. Same magnitude-limit shape as ALiBi<=RoPE.\n")


def note_depth():
    print("3. DEPTH currency -- associative recall / induction:")
    print("   A single softmax layer can implement an induction head (attend to the token")
    print("   that followed the previous occurrence of the current token) via the sharp")
    print("   selection above. Linear attention / SSMs need ~2 layers or width ~ sequence")
    print("   length for the same exact-copy behavior. So even the 'gap' is a resource")
    print("   trade (width / depth / magnitude), not a categorical impossibility for")
    print("   finite context -- but there is no free finite-feature exact reproduction.\n")


if __name__ == "__main__":
    feature_sweep()
    selection_sharpness()
    note_depth()
