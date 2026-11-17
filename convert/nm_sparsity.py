"""N:M structured sparsity <-> dense, BOTH directions (conserved, like MQA<->MHA).

  dense >= N:M-sparse  (EQUAL size): a dense matrix represents any N:M-sparse one
      by zeroing the pruned entries. Free restriction; sparsity saves FLOPs/memory.
      Going TO sparse (magnitude prune) is lossy -- measured below.

  N:M-sparse >= dense  (at M/N x WIDTH): a dense matmul embeds EXACTLY in a wider
      N:M-sparse matmul. Cut the dense columns into groups of N and give each group
      its own group of M slots: the N columns on the first N slots, zeros on the
      other M - N; the input is spread the same way (for 2:4, x -> [x0,x1,0,0,x2,x3,
      0,0,...]). Every group of M columns then has at most N nonzeros, for any N <= M
      (3:4 and 2:3 too, not only N | M): width ceil(cols/N) * M. So "dense fits in a
      bigger sparse." Conserved: the wide-sparse nonzeros == the dense parameter count,
      so the FLOP saving of sparsity is given back exactly when you widen to dense.

Run:  python -m convert.nm_sparsity
"""
from __future__ import annotations

import torch

F64 = torch.float64


def prune_nm(W, n=2, m=4):
    """Magnitude N:M prune along the input dim (keep top-n of every m weights)."""
    rows, cols = W.shape
    assert cols % m == 0
    Wg = W.view(rows, cols // m, m)
    idx = Wg.abs().topk(n, dim=-1).indices
    keep = torch.zeros_like(Wg, dtype=torch.bool).scatter_(-1, idx, True)
    return (Wg * keep).view(rows, cols)


def dense_to_wide_nm(W, n=2, m=4):
    """S (rows, ceil(cols/n)*m) that is N:M-sparse, with expand(): S @ expand(x) == W @ x. Dense column c
    goes to slot (c // n) * m + c % n: each group of n dense columns on the first n of its own m slots
    (Lean EP.dense_sub_nm), so it holds whether or not n divides m."""
    assert 1 <= n <= m
    rows, cols = W.shape
    width = -(-cols // n) * m
    idx = torch.arange(cols)
    idx = (idx // n) * m + idx % n
    S = torch.zeros(rows, width, dtype=W.dtype)
    S[:, idx] = W

    def expand(x):                                # x (..., cols) -> (..., width)
        z = torch.zeros(*x.shape[:-1], width, dtype=x.dtype)
        z[..., idx] = x
        return z
    return S, expand


def max_nonzeros_per_group(S, m=4):
    rows, cols = S.shape
    return int((S.view(rows, cols // m, m) != 0).sum(-1).max())


def run(rows=32, cols=64):
    torch.manual_seed(0)
    W = torch.randn(rows, cols, dtype=F64)
    x = torch.randn(8, cols, dtype=F64)
    ref = x @ W.T

    Ws = prune_nm(W, 2, 4)                         # direction 1: dense -> 2:4
    err1 = ((x @ Ws.T - ref).norm() / ref.norm()).item()
    print(f"1. dense -> 2:4 (magnitude prune, EQUAL size):  rel err {err1:.3e}  (lossy)")
    print(f"   nonzeros {int((Ws != 0).sum())}/{W.numel()} = {(Ws != 0).float().mean():.0%}"
          f"  -> dense >= 2:4-sparse (restriction, saves FLOPs/mem)\n")

    S, expand = dense_to_wide_nm(W, 2, 4)          # direction 2: dense -> wide 2:4 (EXACT)
    err2 = (expand(x) @ S.T - ref).abs().max().item()
    print(f"2. dense -> 2x-wide 2:4 (embed, at M/N x width):  max err {err2:.3e}  (EXACT)")
    print(f"   S width {S.shape[1]} = 2x{cols};  max nonzeros/group-of-4 = "
          f"{max_nonzeros_per_group(S)} (<=2 -> valid 2:4)")
    print(f"   conserved: wide-sparse nonzeros {int((S != 0).sum())} == dense params {W.numel()}")
    print("   -> 2:4-sparse >= dense at 2x width; the sparsity FLOP saving is handed back.\n")
    print("both directions => sparsity is a SIZE-based axis: dense and N:M-sparse are")
    print("mutually emulable given the M/N width factor. Currency: FLOPs / memory (conserved).")


if __name__ == "__main__":
    run()
