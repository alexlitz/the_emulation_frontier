"""Carrying state in reserved residual channels under a REAL pre-norm (stock transformers layers).

Two cookbook rows carry state in extra residual channels and read it back:
  CLA subset standard             carry the earlier layer's K/V source, read K,V from it
  sequential >= parallel (2x w)   carry the pre-attention x, the MLP reads it
attn/cross_layer_kv.py and block/parallel_residual.py verify these with sublayers that
normalise ONLY the channels they read. A stock pre-norm layer (Llama's input_layernorm /
post_attention_layernorm) computes ONE rms over the WHOLE residual, carried channels included.

The copy-stream trick that IS exact: if the reserved half holds an exact duplicate of the main
half, R = [h; h], then rms_{2d}(R) = rms_d(h) exactly (same eps), so a stock norm of width 2d
gives [N(h); N(h)]. Every writer before the carried state is frozen writes into both halves.
That makes layer 0 of the CLA host, and the attention of the parallel host, exact.

Impossibility (one block, any width D, any linear encoding). Let the host residual entering the
second reader be R = P h + Q a, where a is the first sublayer's output (attention output for
seq>=parallel; the producer layer's writes for CLA) and h the state the target wants that
reader to normalise (the pre-attention x; the K/V source). A stock pre-norm reader sees
    G R / rho(R),   rho(R) = sqrt(|R|^2 / D + eps),
while the target needs  Gamma h / sqrt(|h|^2 / d + eps)  for ALL (h, a) (with T >= 2 tokens a
varies independently of h). At h = 0 this forces G Q a = 0 for all a; then for fixed h != 0
the left side is G P h / rho(P h + Q a), and rho changes with a (scale a: |Q a|^2 grows
quadratically) unless Q = 0. So the first sublayer could not have written anything: no
construction of this form is exact at ANY width under a single-rms pre-norm. The extra
currency is a NORM change -- the reader's rms must be taken over the channels (more precisely
the linear readout) it reads, not the whole residual. GroupRMSNorm below is that currency;
with it both constructions are exact on otherwise stock Llama layers.

What survives a stock pre-norm unchanged: parallel >= sequential at 2x DEPTH (an attn-only
parallel block then an MLP-only one). Each sublayer normalises its own block input, which is
exactly what the sequential target normalises. Checked on stock GPT-NeoX layers
(use_parallel_residual True vs False).

Whole-model width. NORM-FREE, an L-block parallel model fits a sequential host of width 2d at every
L, with 2d extra MLP units per block (Lean EP.parallel_sub_seq_2x; block/parallel_residual
.parallel_stack_in_seq_2x): the attention writes the spare channel, and the MLP, which may read and
write the whole residual, moves that write into the main channel with identity units
phi(v) - phi(-v) = v and erases it, so the spare channel is 0 again at every block boundary. Only a
layout that never moves the write (each block's write merely ignored by its own MLP readout and seen
by every later one) needs a fresh channel per block, (L+1) d (parallel_stack_in_seq_wide). With a
per-group norm the identity units read the attention write normalised (v w / rms(v), not v), so the
norm-free move does not carry over as is; that case is not settled here. CLA sharing pairs cost
width the same way.

Run:  python -m block.prenorm_carry
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL, HEADS, HEAD_DIM, FFN, T = 32, 4, 8, 48, 12


def _cfg(hidden, dtype):
    from transformers import LlamaConfig
    c = LlamaConfig(hidden_size=hidden, intermediate_size=FFN, num_attention_heads=HEADS,
                    num_key_value_heads=HEADS, head_dim=HEAD_DIM, rms_norm_eps=1e-6, vocab_size=8,
                    max_position_embeddings=64)
    c._attn_implementation = "eager"
    return c


def _layers(n, hidden, dtype, seed):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    torch.manual_seed(seed)
    out = [LlamaDecoderLayer(_cfg(hidden, dtype), i).to(dtype).eval() for i in range(n)]
    with torch.no_grad():
        for L in out:                                   # non-trivial gains
            L.input_layernorm.weight.uniform_(0.5, 1.5)
            L.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    return out


def _rope_mask(dtype):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    rope = LlamaRotaryEmbedding(_cfg(D_MODEL, dtype))
    pos = torch.arange(T)[None]
    x = torch.zeros(1, T, HEAD_DIM, dtype=dtype)
    cos, sin = rope(x, pos)
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=dtype), 1)[None, None]
    return (cos.to(dtype), sin.to(dtype)), mask


class GroupRMSNorm(nn.Module):
    """RMSNorm taken separately over each channel group (the currency the stock norm lacks)."""

    def __init__(self, weight, eps, groups):
        super().__init__()
        self.weight, self.eps, self.groups = weight, eps, groups

    def forward(self, x):
        parts, lo = [], 0
        for g in self.groups:
            z = x[..., lo:lo + g]
            parts.append(z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.eps))
            lo += g
        return self.weight * torch.cat(parts, -1)


def _place(dst, src, rows=None, cols=None):
    """dst.zero_(); dst[rows, cols] = src (slices)."""
    dst.zero_()
    dst[rows or slice(None), cols or slice(None)] = src


def _attn(attn, xq, xkv, pe, mask, kv_from=None):
    """Llama attention with queries from xq and keys/values from xkv (projections of kv_from,
    default attn itself). Same eager math as transformers (softmax in fp32)."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    src = kv_from or attn
    sh = lambda z: z.view(1, T, -1, HEAD_DIM).transpose(1, 2)
    q, k, v = sh(attn.q_proj(xq)), sh(src.k_proj(xkv)), sh(src.v_proj(xkv))
    q, k = apply_rotary_pos_emb(q, k, *pe)
    w = F.softmax(q @ k.transpose(2, 3) * attn.scaling + mask, dim=-1, dtype=torch.float32).to(q.dtype)
    return attn.o_proj((w @ v).transpose(1, 2).reshape(1, T, -1))


@torch.no_grad()
def cla(dtype=torch.float64, group_norm=False, seed=0):
    """Target: 2 Llama layers, layer 1 REUSES layer 0's K,V (CLA). Host: 2 stock LlamaDecoderLayers
    of width 2d, residual [main; copy of layer-0 input]. Returns (max|host - target|, max|target|)."""
    d, pe, mask = D_MODEL, *_rope_mask(dtype)
    t0, t1 = _layers(2, d, dtype, seed)
    h0 = torch.randn(1, T, d, dtype=dtype, generator=torch.Generator().manual_seed(seed + 1))
    # target
    n0 = t0.input_layernorm(h0)
    h1 = h0 + _attn(t0.self_attn, n0, n0, pe, mask)
    h1 = h1 + t0.mlp(t0.post_attention_layernorm(h1))
    n1 = t1.input_layernorm(h1)
    h2 = h1 + _attn(t1.self_attn, n1, n0, pe, mask, kv_from=t0.self_attn)     # CLA: K,V of layer 0
    target = h2 + t1.mlp(t1.post_attention_layernorm(h2))
    # host
    H0, H1 = _layers(2, 2 * d, dtype, seed + 7)
    M, C = slice(0, d), slice(d, 2 * d)
    for Hl, tl in ((H0, t0), (H1, t1)):
        Hl.input_layernorm.weight[M] = tl.input_layernorm.weight
        Hl.post_attention_layernorm.weight[M] = tl.post_attention_layernorm.weight
        Hl.post_attention_layernorm.weight[C] = 1.0
        _place(Hl.self_attn.q_proj.weight, tl.self_attn.q_proj.weight, cols=M)
        _place(Hl.self_attn.o_proj.weight, tl.self_attn.o_proj.weight, rows=M)   # writes main only
        for a in ("gate_proj", "up_proj"):
            _place(getattr(Hl.mlp, a).weight, getattr(tl.mlp, a).weight, cols=M)
        _place(Hl.mlp.down_proj.weight, tl.mlp.down_proj.weight, rows=M)
    H0.input_layernorm.weight[C] = 1.0
    for a in ("k_proj", "v_proj"):
        _place(getattr(H0.self_attn, a).weight, getattr(t0.self_attn, a).weight, cols=M)
        _place(getattr(H1.self_attn, a).weight, getattr(t0.self_attn, a).weight, cols=C)  # read the copy
    H1.input_layernorm.weight[C] = t0.input_layernorm.weight                              # layer-0 gain
    if group_norm:
        for Hl in (H0, H1):
            for nm in ("input_layernorm", "post_attention_layernorm"):
                n = getattr(Hl, nm)
                setattr(Hl, nm, GroupRMSNorm(n.weight, n.variance_epsilon, (d, d)))
    R = torch.cat([h0, h0], -1)                         # the embedding writes both halves
    R = H0(R, attention_mask=mask, position_embeddings=pe)
    R = H1(R, attention_mask=mask, position_embeddings=pe)
    return (R[..., M] - target).abs().max().item(), target.abs().max().item()


@torch.no_grad()
def seq_hosts_parallel(dtype=torch.float64, group_norm=False, seed=0):
    """Target: a PARALLEL Llama block x + Attn(N1 x) + MLP(N2 x) (stock submodules). Host: one
    stock LlamaDecoderLayer of width 2d, residual [x; x]; attention reads/writes main, the MLP
    reads the copy and writes main. Returns (max|host - target|, max|target|)."""
    d, pe, mask = D_MODEL, *_rope_mask(dtype)
    (t,) = _layers(1, d, dtype, seed)
    x = torch.randn(1, T, d, dtype=dtype, generator=torch.Generator().manual_seed(seed + 1))
    nx = t.input_layernorm(x)
    target = x + _attn(t.self_attn, nx, nx, pe, mask) + t.mlp(t.post_attention_layernorm(x))
    (H,) = _layers(1, 2 * d, dtype, seed + 7)
    M, C = slice(0, d), slice(d, 2 * d)
    H.input_layernorm.weight[M] = t.input_layernorm.weight
    H.input_layernorm.weight[C] = 1.0
    H.post_attention_layernorm.weight[M] = 1.0
    H.post_attention_layernorm.weight[C] = t.post_attention_layernorm.weight
    for a in ("q_proj", "k_proj", "v_proj"):
        _place(getattr(H.self_attn, a).weight, getattr(t.self_attn, a).weight, cols=M)
    _place(H.self_attn.o_proj.weight, t.self_attn.o_proj.weight, rows=M)
    for a in ("gate_proj", "up_proj"):
        _place(getattr(H.mlp, a).weight, getattr(t.mlp, a).weight, cols=C)           # MLP reads the copy
    _place(H.mlp.down_proj.weight, t.mlp.down_proj.weight, rows=M)
    if group_norm:
        for nm in ("input_layernorm", "post_attention_layernorm"):
            n = getattr(H, nm)
            setattr(H, nm, GroupRMSNorm(n.weight, n.variance_epsilon, (d, d)))
    R = H(torch.cat([x, x], -1), attention_mask=mask, position_embeddings=pe)
    return (R[..., M] - target).abs().max().item(), target.abs().max().item()


@torch.no_grad()
def parallel_hosts_seq_2x_depth(dtype=torch.float64, seed=0):
    """Target: one SEQUENTIAL stock GPTNeoXLayer. Host: two stock PARALLEL GPTNeoXLayers (the first
    attention-only, the second MLP-only), real LayerNorm pre-norm throughout."""
    from transformers import GPTNeoXConfig
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer, GPTNeoXRotaryEmbedding
    cfg = GPTNeoXConfig(hidden_size=D_MODEL, num_attention_heads=HEADS, intermediate_size=FFN,
                        rotary_pct=0.5, use_parallel_residual=False, max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    tgt = GPTNeoXLayer(cfg, 0).to(dtype).eval()
    cfg_p = GPTNeoXConfig(**{**cfg.to_dict(), "use_parallel_residual": True})
    cfg_p._attn_implementation = "eager"
    h1, h2 = GPTNeoXLayer(cfg_p, 0).to(dtype).eval(), GPTNeoXLayer(cfg_p, 1).to(dtype).eval()
    for h in (h1, h2):
        h.load_state_dict(tgt.state_dict())
    for p in (h1.mlp.dense_4h_to_h.weight, h1.mlp.dense_4h_to_h.bias,       # block 1: attention only
              h2.attention.dense.weight, h2.attention.dense.bias):         # block 2: MLP only
        p.zero_()
    h2.post_attention_layernorm.load_state_dict(tgt.post_attention_layernorm.state_dict())
    rope = GPTNeoXRotaryEmbedding(cfg)
    x = torch.randn(1, T, D_MODEL, dtype=dtype, generator=torch.Generator().manual_seed(seed + 1))
    cos, sin = rope(x, torch.arange(T)[None])
    pe = (cos.to(dtype), sin.to(dtype))
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=dtype), 1)[None, None]
    o = lambda L, z: (lambda r: r[0] if isinstance(r, tuple) else r)(L(z, attention_mask=mask, position_embeddings=pe))
    ref = o(tgt, x)
    return (o(h2, o(h1, x)) - ref).abs().max().item(), ref.abs().max().item()


def main():
    rows = [("CLA, stock pre-norm, copy stream  ", lambda dt: cla(dt)),
            ("CLA, GroupRMSNorm readers         ", lambda dt: cla(dt, True)),
            ("seq>=parallel 2x width, stock norm", lambda dt: seq_hosts_parallel(dt)),
            ("seq>=parallel 2x width, GroupRMS  ", lambda dt: seq_hosts_parallel(dt, True)),
            ("parallel>=seq 2x depth, stock NeoX", lambda dt: parallel_hosts_seq_2x_depth(dt))]
    for dt in (torch.float64, torch.float32):
        n = str(dt).split(".")[-1]
        for name, f in rows:
            e, s = f(dt)
            print(f"[{n}] {name}  max|d| = {e:.3e}   (max|y| = {s:.2f}, rel {e / s:.1e})")


if __name__ == "__main__":
    main()
