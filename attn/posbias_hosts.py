"""Position transitions into additive-bias and learned-absolute hosts (Lean PosBias.lean).

* bias_attention(q, k, v, B, keep): an additive-bias head, softmax(q k^T * scale + B) v over the keys
  `keep` allows. A T5 relative bias is the table B[i, j] = b(bucket(j - i)) (t5_bias_table, the bucketing
  of transformers' T5Attention), so T5 is an additive-bias host (EP.t5_sub_additive_bias).
* alibi_learned_abs_host(Wq, Wk, slope, scale): a learned-absolute host for one ALiBi head, norm-free.
  The position table writes [0; 1; t] into two extra residual channels; the host query reads the 1, the
  host key reads slope * t / scale. The host logit is the ALiBi logit q.k scale - m (i - j) plus m i, a
  per-query constant that softmax ignores (EP.alibi_sub_learned_abs_causal).
* onehot_learned_abs_host(Wq, Wk, B, scale): any additive table over T positions on a learned-absolute
  host: one-hot position channels e_t, the query reads e_i, the key reads B^T e_j, so the extra
  coordinates add B[i, j] (EP.bias_sub_learned_abs).

The hosts are weight matrices on the residual [x; position channels]: q = Wq' r, k = Wk' r, v = Wv' r with
r = [x; 0] + P_t. Norm-free only (a pre-norm meets EP.prenorm_readout_obstruction).
"""
from __future__ import annotations

import math

import torch


def t5_bucket(rel, bidirectional=True, num_buckets=32, max_distance=128):
    """T5's relative-position bucket of rel = key - query (as transformers' T5Attention computes it)."""
    rel = torch.as_tensor(rel)
    buckets = torch.zeros_like(rel)
    if bidirectional:
        num_buckets //= 2
        buckets = buckets + (rel > 0).to(torch.long) * num_buckets
        rel = rel.abs()
    else:
        rel = -torch.clamp(rel, max=0)
    max_exact = num_buckets // 2
    small = rel < max_exact
    large = max_exact + (torch.log(rel.float() / max_exact) / math.log(max_distance / max_exact)
                         * (num_buckets - max_exact)).to(torch.long)
    large = torch.clamp(large, max=num_buckets - 1)
    return buckets + torch.where(small, rel, large)


def t5_bias_table(T, emb, bidirectional=True, num_buckets=32, max_distance=128):
    """B[h, i, j] = emb[bucket(j - i), h]: the additive table of a T5 relative bias (emb (num_buckets, H))."""
    pos = torch.arange(T)
    bk = t5_bucket(pos[None, :] - pos[:, None], bidirectional, num_buckets, max_distance)
    return emb[bk].permute(2, 0, 1)


def bias_attention(q, k, v, B, keep=None, scale=1.0):
    """softmax(q k^T scale + B) v, keys outside `keep` (bool (.., T, T)) removed. q, k (.., T, dh)."""
    s = q @ k.transpose(-1, -2) * scale + B
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    return torch.softmax(s, -1) @ v


def alibi_scores(c, slope):
    """EP.alibiScore: c[i, j] - m (i - j)."""
    T = c.shape[-1]
    i = torch.arange(T, dtype=c.dtype)
    return c - slope * (i[:, None] - i[None, :])


def _blk(*rows):
    return torch.cat([torch.cat(r, 1) for r in rows], 0)


def alibi_learned_abs_host(Wq, Wk, Wv, slope, scale, T):
    """The learned-absolute host of one ALiBi head (EP.alibiLAQ / alibiLAK). Wq, Wk, Wv (dh, d). Returns
    (P (T, d + 2) the position table [0; 1; t], Wq' (dh + 1, d + 2), Wk' (dh + 1, d + 2), Wv' (dv, d + 2))."""
    dh, d = Wq.shape
    dt = Wq.dtype
    z = lambda a, b: torch.zeros(a, b, dtype=dt)           # noqa: E731
    P = torch.zeros(T, d + 2, dtype=dt)
    P[:, d] = 1.0
    P[:, d + 1] = torch.arange(T, dtype=dt)
    one = torch.ones(1, 1, dtype=dt)
    Wq2 = _blk([Wq, z(dh, 2)], [z(1, d), one, z(1, 1)])
    Wk2 = _blk([Wk, z(dh, 2)], [z(1, d), z(1, 1), one * (slope / scale)])
    Wv2 = torch.cat([Wv, z(Wv.shape[0], 2)], 1)
    return P, Wq2, Wk2, Wv2


def onehot_learned_abs_host(Wq, Wk, Wv, B, scale):
    """The learned-absolute host of any additive table B (T, T) (EP.onehot, EP.bias_sub_learned_abs).
    Returns (P (T, d + T) = [0; e_t], Wq' (dh + T, d + T) reading e_i, Wk' reading B^T e_j / scale, Wv')."""
    dh, d = Wq.shape
    T = B.shape[-1]
    dt = Wq.dtype
    z = lambda a, b: torch.zeros(a, b, dtype=dt)           # noqa: E731
    P = torch.cat([z(T, d), torch.eye(T, dtype=dt)], 1)
    Wq2 = _blk([Wq, z(dh, T)], [z(T, d), torch.eye(T, dtype=dt)])
    Wk2 = _blk([Wk, z(dh, T)], [z(T, d), B / scale])     # (B e_j)_s ... key extra coords = column j of B
    Wv2 = torch.cat([Wv, z(Wv.shape[0], T)], 1)
    return P, Wq2, Wk2, Wv2


def learned_abs_head(x, P, Wq, Wk, Wv, keep=None, scale=1.0):
    """The host head on the residual r_t = [x_t; 0] + P_t: softmax(q k^T scale) v with q = Wq r etc."""
    r = torch.cat([x, torch.zeros(*x.shape[:-1], P.shape[-1] - x.shape[-1], dtype=x.dtype)], -1) + P
    return bias_attention(r @ Wq.T, r @ Wk.T, r @ Wv.T, 0.0, keep, scale)
