"""Small numerical checks of the new-construction candidates (pure torch, random weights).

  A. token splitting, MHA target on an MQA host, k = n_heads sub-tokens per target token (block layout: sub-token
     (t, j) holds x_t in residual block j, so the host's ONE shared W_k / W_v maps block j by the target's head-j
     maps; a sub-index channel gives wrong-sub-index keys a logit gap that underflows exp in fp32). One layer:
     exact, with and without an RMS pre-norm. Two layers: the scratch sub-positions (t, j < n - 1) cannot see the
     later sub-positions of their own token, whose KV heads the target's self-attention needs, so they cannot hold
     x_t after layer 1 -- the best causal scheme (each sub-position runs the target layer on what it sees) is off.
  B. an output-side bias from the start token: a 'bias head' whose query reads a tiny constant channel and whose
     key at position 0 is huge attends ONLY to position 0 (the other weights underflow to 0 in fp32) under a
     pre-norm, and writes a constant -- o_proj / v biases exactly, without a bias tensor.
  C. KV-head packing: a GQA target with 2 KV heads (head_dim 8, 2 query heads each) on a host with ONE KV head of
     head_dim 16 and 4 query heads: K = [k_a; k_b], a query of group a is [q_a; 0], o_proj reads v_a's block. RoPE
     with the frequencies repeated on both blocks (an inv_freq buffer setting).

Run:  python -m verify.levers.constructions
"""
from __future__ import annotations

import json
import os

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def rms(x, g, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


def attn(q, k, v, mask=None):
    """q [H, T, d], k/v [H or 1, S, d]; causal unless mask given."""
    s = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is None:
        T, S = s.shape[-2:]
        mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
    s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, -1) @ v


# ---------------------------------------------------------------- A. token splitting
def token_split(prenorm=True, layers=1, dtype=torch.float32, n=4, dh=8, T=6, seed=0):
    torch.manual_seed(seed)
    d = n * dh
    Wq, Wk, Wv = (torch.randn(layers, n, dh, d, dtype=dtype) / d ** 0.5 for _ in range(3))
    Wo = torch.randn(layers, d, n * dh, dtype=dtype) / d ** 0.5
    g = torch.rand(layers, d, dtype=dtype) + 0.5
    x = torch.randn(T, d, dtype=dtype)
    # target: MHA, n heads
    h = x.clone()
    for l in range(layers):
        u = rms(h, g[l]) if prenorm else h
        q = torch.einsum("hed,td->hte", Wq[l], u)
        k = torch.einsum("hed,td->hte", Wk[l], u)
        v = torch.einsum("hed,td->hte", Wv[l], u)
        o = attn(q, k, v).transpose(0, 1).reshape(T, n * dh)
        h = h + o @ Wo[l].T
    target = h
    # host: MQA (ONE KV head of width dh + n), n query heads, residual = n blocks of d + 1 constant channel
    D = n * d + 1
    eps_c, Gc, B = 2.0 ** -40, 2.0 ** 60, 2.0 ** 7       # tiny constant channel, its huge gain, sub-index logit
    X = torch.zeros(T * n, D, dtype=dtype)
    for t in range(T):
        for j in range(n):
            X[t * n + j, j * d:(j + 1) * d] = x[t]
            X[t * n + j, -1] = eps_c * x[t].abs().max()
    H = X.clone()
    for l in range(layers):
        if prenorm:
            gh = torch.cat([g[l].repeat(n) * (d / D) ** 0.5, torch.full((1,), Gc, dtype=dtype)])
            U = rms(H, gh, eps=1e-6 * d / D)                # one nonzero block: rms over D dims -> fold sqrt(d / D)
        else:
            U = H.clone()
            U[:, -1] = U[:, -1] * Gc
        sub = torch.arange(T * n) % n
        blk = lambda m, j: m[:, j * d:(j + 1) * d]
        # shared key / value: block j -> the target's head-j maps; plus the one-hot sub-index (as channel values)
        K = torch.zeros(T * n, dh + n, dtype=dtype)
        V = torch.zeros(T * n, dh, dtype=dtype)
        for j in range(n):
            K[:, :dh] += blk(U, j) @ Wk[l, j].T
            V += blk(U, j) @ Wv[l, j].T
        K[:, dh:] = -1.0                                    # sub-index channels: 0 at the key's own sub-index,
        K[torch.arange(T * n), dh + sub] = 0.0              # -1 elsewhere (correct keys' logits untouched)
        # query head h at every sub-position reads that sub-position's own block (the best causal scheme), plus
        # B x (constant channel) on sub-index h
        Q = torch.zeros(n, T * n, dh + n, dtype=dtype)
        for hh in range(n):
            for j in range(n):
                rows = sub == j
                Q[hh, rows, :dh] = blk(U[rows], j) @ Wq[l, hh].T
            Q[hh, :, dh + hh] = B * U[:, -1]                # >> 0 on every token: wrong sub-index -> exp underflow
        Qs = Q.clone()
        Qs[:, :, :dh] = Q[:, :, :dh] * ((dh + n) / dh) ** 0.5   # host scale 1/sqrt(dh + n) -> target 1/sqrt(dh)
        o = attn(Qs, K[None], V[None]).transpose(0, 1).reshape(T * n, n * dh)
        delta = o @ Wo[l].T                                 # each sub-position writes into its own block
        for j in range(n):
            rows = sub == j
            H[rows, j * d:(j + 1) * d] += delta[rows]
    host = torch.stack([H[t * n + n - 1, (n - 1) * d:n * d] for t in range(T)])
    return float((host.double() - target.double()).abs().max()), float(target.abs().max())


# ---------------------------------------------------------------- B. bias head
def bias_head(dtype=torch.float32, d=16, T=12, seed=0):
    torch.manual_seed(seed)
    b = torch.randn(d, dtype=dtype)                         # the target's o_proj bias (added to every token)
    x = torch.randn(T, d, dtype=dtype) * torch.linspace(0.2, 5, T, dtype=dtype)[:, None]
    # host residual: [x ; c (tiny constant, every real token) ; s (start-token channel)]; position 0 = start token
    eps_c = 2.0 ** -40
    X = torch.zeros(T + 1, d + 2, dtype=dtype)
    X[1:, :d] = x
    X[1:, d] = eps_c * x.abs().max()
    X[0, d + 1] = 1.0
    g = torch.ones(d + 2, dtype=dtype)
    g[d] = 2.0 ** 60                                        # the constant channel's gain: c / rms -> O(1)
    U = rms(X, g)
    q = U[:, d:d + 1] * 2.0 ** 10                           # query reads the constant channel (> 0 on real tokens)
    k = U[:, d + 1:d + 2] * 2.0 ** 10                       # key: large only at the start token
    v = torch.zeros(T + 1, d, dtype=dtype)
    v[0] = b * U[0, d + 1] / U[0, d + 1]                    # W_v reads the start-token channel -> b at position 0
    w = torch.softmax((q @ k.T).masked_fill(~torch.ones(T + 1, T + 1, dtype=torch.bool).tril(), float("-inf")), -1)
    out = w @ v
    off0 = float(w[1:, 1:].max())                           # weight on any non-start position
    # pre-norm of the real channels unchanged by the tiny constant channel?
    drift = float((rms(X[1:, :d + 1], torch.ones(d + 1, dtype=dtype), eps=1e-6 * d / (d + 1))[:, :d] * (d / (d + 1)) ** 0.5
                   - rms(x, torch.ones(d, dtype=dtype))).abs().max())
    return float((out[1:] - b).abs().max()), off0, drift


# ---------------------------------------------------------------- C. KV-head packing
def rope(x, pos, inv):
    """half-split rotary on the last dim with frequencies inv (len = dim / 2)."""
    a = pos[:, None] * inv[None, :]
    c, s = torch.cos(a), torch.sin(a)
    x1, x2 = x[..., : inv.numel()], x[..., inv.numel():]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def kv_pack(dtype=torch.float64, dh=8, T=10, seed=0, nope=False, inv_host=None):
    """max |packed host - GQA target| over every query head and position. nope=True: a NoPE target (and
    zero host frequencies). inv_host: override the host's 2 dh / 2 frequencies (default: the target's,
    repeated on both blocks, the buffer write the construction needs)."""
    torch.manual_seed(seed)
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, dtype=dtype) / dh))
    if nope:
        inv = torch.zeros_like(inv)
    pos = torch.arange(T, dtype=dtype)
    q = torch.randn(4, T, dh, dtype=dtype)                  # 4 query heads: 0, 1 -> KV a ; 2, 3 -> KV b
    k = torch.randn(2, T, dh, dtype=dtype)
    v = torch.randn(2, T, dh, dtype=dtype)
    qr, kr = rope(q, pos, inv), rope(k, pos, inv)
    tgt = torch.cat([attn(qr[:2], kr[0:1], v[0:1]), attn(qr[2:], kr[1:2], v[1:2])])
    # host: one KV head of 2 dh; layout per block [x1 | x2] -> host half-split pairs: block a pairs then block b
    half = dh // 2
    inv_h = torch.cat([inv, inv]) if inv_host is None else inv_host.to(dtype)   # repeated (the inv_freq buffer)

    def pack(xa, xb):                                       # host half-split: [a1, b1 | a2, b2]
        return torch.cat([xa[..., :half], xb[..., :half], xa[..., half:], xb[..., half:]], -1)
    Kh = pack(k[0], k[1])[None]
    Vh = torch.cat([v[0], v[1]], -1)[None]
    z = torch.zeros_like(q[0])
    Qh = torch.stack([pack(q[0], z), pack(q[1], z), pack(z, q[2]), pack(z, q[3])])
    Qr, Kr = rope(Qh, pos, inv_h), rope(Kh, pos, inv_h)
    out = attn(Qr * (2 * dh / dh) ** 0.5, Kr, Vh)          # host scale 1/sqrt(2 dh) -> fold sqrt(2) into q
    host = torch.cat([out[:2, :, :dh], out[2:, :, dh:]])    # o_proj reads v_a for group a, v_b for group b
    return float((host - tgt).abs().max())


def main():
    rows = []
    for dt in (torch.float32, torch.float64):
        for pn in (False, True):
            for L in (1, 2):
                e, m = token_split(pn, L, dt)
                rows.append(dict(check=f"A token split MHA<-MQA, k=n=4, {L} layer(s), "
                                       f"{'RMS pre-norm' if pn else 'no norm'}", dtype=str(dt), max_abs_diff=e,
                                 max_abs_value=m))
    e, off0, drift = bias_head()
    rows.append(dict(check="B bias head: output - o_bias", dtype="torch.float32", max_abs_diff=e,
                     weight_off_start=off0, prenorm_drift=drift))
    rows.append(dict(check="C KV-head packing GQA(2 KV) <- 1 KV head of 2x head_dim, RoPE", dtype="torch.float64",
                     max_abs_diff=kv_pack()))
    for r in rows:
        extra = "".join(f"  {k}={r[k]:.2e}" for k in ("max_abs_value", "weight_off_start", "prenorm_drift") if k in r)
        print(f"{r['check']:62s} {r['dtype']:14s} max|d| = {r['max_abs_diff']:.3e}{extra}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "constructions.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
