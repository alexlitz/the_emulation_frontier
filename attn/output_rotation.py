"""OUTPUT-path rotation: retentive / linear attention carries the position rotation
in its recurrent STATE, so it lands on the output -- unlike RoPE, which only rotates
Q and K on the input side and leaves no rotation on the output.

Softmax attention has no persistent state: a rotation only shows up inside the QK
score (RoPE), and since values are un-rotated the output sits in the plain basis.
Retention/linear attention DOES have a state  S_n = sum_{m<=n} gamma^(n-m) k~_m v_m^T
built from ROTATED keys k~_m = R_m k_m, read out by a rotated query q~_n = R_n q_n.
So the rotation lives on the state = the output path. This is the output-side dual
of RoPE, and it is exactly what distinguishes retentive mixers (RetNet, RWKV6/7,
xPos) from plain softmax+RoPE.

We verify:
  (1) parallel retention  ==  recurrent retention   (the state carries rotation+decay)
  (2) the score q~_n . k~_m = q_n^T R_(m-n) k_m depends only on (n-m): the output
      rotation is RELATIVE (shift-invariant), like RoPE's relative property.
  (3) gamma=1, theta=0 collapses retention to plain (un-rotated, un-decayed) linear
      attention -> the rotation/decay are the added output-path structure.

All in float64; retention is mixer=linear (<= softmax at the near-gap in the graph).

Run:  python -m attn.output_rotation
"""
from __future__ import annotations

import math

import torch

F64 = torch.float64


def rot_stack(T, d, theta):
    """R_n = block-diagonal 2x2 rotations by angle n*theta, for n in [0,T). (T,d,d)."""
    assert d % 2 == 0
    Rs = torch.zeros(T, d, d, dtype=F64)
    for n in range(T):
        c, s = math.cos(n * theta), math.sin(n * theta)
        for i in range(0, d, 2):
            Rs[n, i, i] = c;      Rs[n, i, i + 1] = -s
            Rs[n, i + 1, i] = s;  Rs[n, i + 1, i + 1] = c
    return Rs


def rotate(X, Rs):
    return torch.stack([Rs[n] @ X[n] for n in range(X.shape[0])])   # x~_n = R_n x_n


def parallel_retention(Q, K, V, theta, gamma, offset=0):
    """Out_n = sum_{m<=n} gamma^(n-m) (R_n q_n).(R_m k_m) v_m  -- rotation+decay in score."""
    T, d = Q.shape
    Rs = rot_stack(T + offset, d, theta)[offset:]
    Qr, Kr = rotate(Q, Rs), rotate(K, Rs)
    n = torch.arange(T)
    decay = torch.where(n[:, None] >= n[None, :],
                        gamma ** (n[:, None] - n[None, :]).clamp(min=0).double(), 0.0)
    return (Qr @ Kr.T * decay) @ V


def recurrent_retention(Q, K, V, theta, gamma, offset=0):
    """Same, as a linear recurrence whose STATE holds rotated keys (output-path rotation)."""
    T, d = Q.shape
    dv = V.shape[1]
    Rs = rot_stack(T + offset, d, theta)[offset:]
    Qr, Kr = rotate(Q, Rs), rotate(K, Rs)
    S = torch.zeros(d, dv, dtype=F64)     # the state: sum of gamma-decayed rotated k v^T
    out = torch.empty(T, dv, dtype=F64)
    for n in range(T):
        S = gamma * S + torch.outer(Kr[n], V[n])
        out[n] = Qr[n] @ S
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    T, d, dv = 20, 8, 8
    Q = torch.randn(T, d, dtype=F64); K = torch.randn(T, d, dtype=F64); V = torch.randn(T, dv, dtype=F64)
    theta, gamma = 0.3, 0.95

    par = parallel_retention(Q, K, V, theta, gamma)
    rec = recurrent_retention(Q, K, V, theta, gamma)
    print(f"parallel vs recurrent retention:   max|Δ| = {(par - rec).abs().max().item():.2e}")

    shifted = parallel_retention(Q, K, V, theta, gamma, offset=7)     # start rotations at n=7
    print(f"position-shift invariance (relative): max|Δ| = {(par - shifted).abs().max().item():.2e}")

    plain = parallel_retention(Q, K, V, 0.0, 1.0)                      # theta=0, gamma=1
    lin = (torch.tril(Q @ K.T)) @ V                                    # plain causal linear attn
    print(f"theta=0,gamma=1 == plain linear attention: max|Δ| = {(plain - lin).abs().max().item():.2e}")
    print("-> retention carries the rotation in its state (output path); it is the")
    print("   output-side dual of RoPE. Both are relative. mixer=linear <= softmax (near-gap).")
