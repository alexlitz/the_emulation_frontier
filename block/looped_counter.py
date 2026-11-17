"""Norm-free looped hosts for untied targets (Lean EP.looped_width2_neg, EP.looped_counter_two).

A looped host applies ONE residual block K times. At the target's width that cannot play an untied
target in general (EP.looped_same_width_fails: x -> -x is no g o g). Two constructions show extra
residual width fixes it, norm-free:

* width2_block(): the block p -> p + (R p - p), R the quarter rotation, on a width-2 residual. Looped
  twice it maps (x, 0) to (-x, 0), the 2-layer residual ReLU target x + (-2 relu x + 2 relu(-x)) then
  the identity (EP.negTarget).
* counter_block(a, v, a2, v2, B): one residual ReLU-MLP block on the residual [x; c; o] (a counter c,
  embedded 0, and a constant o, embedded 1) holding both target layers' units:

      layer-1 units  relu(a_j . x - B c)        write v_j   into x
      layer-2 units  relu(a2_k . x - B (o - c))  write v2_k  into x
      one unit       relu(o)                     writes 1    into c

  Every pre-activation is linear in [x; c; o], so this is an ordinary MLP (W_in, b_in = 0, W_out). On
  bounded inputs (a2_k . x <= B and a_j . f1(x) <= B) two iterations give exactly (f2(f1 x), 2, 1).

Run:  python -m block.looped_counter
"""
from __future__ import annotations

import torch

F64 = torch.float64


class ResidualMLP:
    """x -> x + W_out act(W_in x + b_in): a norm-free residual MLP block (ReLU by default)."""

    def __init__(self, W_in, b_in, W_out, act=torch.relu):
        self.W_in, self.b_in, self.W_out, self.act = W_in, b_in, W_out, act

    def __call__(self, s):
        return s + self.act(s @ self.W_in.T + self.b_in) @ self.W_out.T


class LinearResidual:
    """p -> p + (M p - p): a residual block whose update is linear."""

    def __init__(self, M):
        self.M = M

    def __call__(self, p):
        return p + (p @ self.M.T - p)


def loop(block, s, K):
    for _ in range(K):
        s = block(s)
    return s


def width2_block(dtype=F64):
    """EP.looped_width2_neg's block: the quarter rotation R (x, y) -> (-y, x) as a residual update."""
    return LinearResidual(torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype))


def neg_target(x):
    """EP.negTarget: layer 1 x + (-2 relu x + 2 relu(-x)), layer 2 the identity (a zeroed MLP)."""
    y = x + (-2 * torch.relu(x) + 2 * torch.relu(-x))
    return y + 0


def two_layer_target(a, v, a2, v2):
    """The untied target f2 o f1, f_l(y) = y + sum_j relu(a_j . y) v_j (a (m, d), v (m, d))."""
    def f(y, A, Vv):
        return y + torch.relu(y @ A.T) @ Vv
    return lambda x: f(f(x, a, v), a2, v2)


def counter_block(a, v, a2, v2, B):
    """EP.looped_counter_two's block on [x; c; o] (width d + 2): W_in (m + m' + 1, d + 2), W_out (d + 2, ...)."""
    m, d = a.shape
    m2 = a2.shape[0]
    dt = a.dtype
    W_in = torch.zeros(m + m2 + 1, d + 2, dtype=dt)
    W_in[:m, :d], W_in[:m, d] = a, -B                       # a_j . x - B c
    W_in[m:m + m2, :d], W_in[m:m + m2, d], W_in[m:m + m2, d + 1] = a2, B, -B   # a2_k . x - B (o - c)
    W_in[m + m2, d + 1] = 1.0                               # relu(o)
    W_out = torch.zeros(d + 2, m + m2 + 1, dtype=dt)
    W_out[:d, :m], W_out[:d, m:m + m2] = v.T, v2.T
    W_out[d, m + m2] = 1.0                                  # the counter += relu(o)
    return ResidualMLP(W_in, torch.zeros(m + m2 + 1, dtype=dt), W_out)


def embed_counter(x):
    """[x; 0; 1]: the counter starts at 0, the constant channel at 1."""
    return torch.cat([x, torch.zeros(*x.shape[:-1], 1, dtype=x.dtype), torch.ones(*x.shape[:-1], 1, dtype=x.dtype)], -1)


def main():
    g = torch.Generator().manual_seed(0)
    d, m = 6, 8
    a, v, a2, v2 = (torch.randn(m, d, dtype=F64, generator=g) for _ in range(4))
    x = torch.randn(16, d, dtype=F64, generator=g)
    f = two_layer_target(a, v, a2, v2)
    f1 = two_layer_target(a, v, torch.zeros_like(a2), v2)
    B = max((x @ a2.T).max().item(), (f1(x) @ a.T).max().item(), 0.0) + 1.0
    s = loop(counter_block(a, v, a2, v2, B), embed_counter(x), 2)
    print("counter host max |err| =", (s[:, :d] - f(x)).abs().max().item(), "counter, const:", s[0, d:].tolist())
    p = loop(width2_block(), torch.stack([x[:, 0], torch.zeros(16, dtype=F64)], -1), 2)
    print("width-2 host max |err| =", (p[:, 0] - neg_target(x[:, 0])).abs().max().item())


if __name__ == "__main__":
    main()
