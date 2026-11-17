"""A GQA / MQA / MHA target's value path on an MLA host, the one exact case (Lean EP.mla_host_sphere_full_rank).

An MLA host's values are W_UV RMSNorm(c), c = W_DKV x, and its latent norm (DeepSeek kv_a_layernorm) cannot be
switched off. On a single-token context an attention sublayer outputs its own value through o_proj: the host
writes M RMSNorm(W x), the target A x (A = W_O W_V). Along a ray the host is degree 0 and the target degree 1
(EP.mla_value_ray), so this only works when the inputs lie on a sphere (a target with pre-norm eps = 0) and the
down-projection is a scaled isometry, W^T W = alpha I (latent rank r >= n). Then the latent norm divides by the
constant D = sqrt(alpha R^2 / r + eps) and

    M = (D / alpha) A W^T diag(1/w)

reproduces A x exactly.

* `single_token_matrix(attn)`: A of a stock Llama-family attention (o_proj times v_proj, kv heads repeated).
* `mla_value_matrix(A, W, w, eps, alpha, R)`: M above.
* `write_mla_value_host(attn, A, R, w, alpha, seed)`: writes a stock DeepseekV3Attention so that its single-token
  output is M RMSNorm(W x): kv_a_proj_with_mqa's latent rows W = sqrt(alpha) Q (Q orthonormal columns, rope rows
  zero), kv_a_layernorm weight w, kv_b_proj's value rows and o_proj realise M (head h carries rows
  h dv .. (h+1) dv of M). Needs kv_lora_rank >= hidden_size and num_heads * v_head_dim >= hidden_size.
"""
from __future__ import annotations

import math

import torch


def single_token_matrix(attn):
    """A with attn(x) = A x on a single token (softmax over one key is 1): o_proj . repeat_kv(v_proj)."""
    H = attn.config.num_attention_heads
    hd = attn.head_dim
    Hkv = attn.config.num_key_value_heads
    Wv = attn.v_proj.weight.view(Hkv, hd, -1)
    Wv = Wv.repeat_interleave(H // Hkv, 0).reshape(H * hd, -1)
    return attn.o_proj.weight @ Wv


def mla_value_matrix(A, W, w, eps, alpha, R):
    r = W.shape[0]
    D = math.sqrt(alpha * R ** 2 / r + eps)
    return (D / alpha) * (A @ W.T @ torch.diag(1.0 / w))


@torch.no_grad()
def write_mla_value_host(attn, A, R, w, alpha=1.0, seed=0):
    """In place on a stock DeepseekV3Attention. Returns (W, M)."""
    cfg = attn.config
    n, r = cfg.hidden_size, attn.kv_lora_rank
    H, dv, dn = attn.num_heads, attn.v_head_dim, attn.qk_nope_head_dim
    m = A.shape[0]
    if r < n or H * dv < m:
        raise ValueError("needs kv_lora_rank >= hidden_size and num_heads * v_head_dim >= the output width")
    dt = attn.kv_b_proj.weight.dtype
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(r, n, dtype=dt, generator=g))
    W = math.sqrt(alpha) * Q
    eps = attn.kv_a_layernorm.variance_epsilon
    M = mla_value_matrix(A, W, w, eps, alpha, R)
    attn.kv_a_proj_with_mqa.weight.zero_()
    attn.kv_a_proj_with_mqa.weight[:r] = W
    if attn.kv_a_proj_with_mqa.bias is not None:
        attn.kv_a_proj_with_mqa.bias.zero_()
    attn.kv_a_layernorm.weight.copy_(w)
    kvb = attn.kv_b_proj.weight.view(H, dn + dv, r)
    kvb[:, dn:].zero_()
    Mp = torch.zeros(H * dv, r, dtype=dt)
    Mp[:m] = M
    kvb[:, dn:] = Mp.view(H, dv, r)
    attn.o_proj.weight.zero_()
    attn.o_proj.weight[:m, :m] = torch.eye(m, dtype=dt)
    if attn.o_proj.bias is not None:
        attn.o_proj.bias.zero_()
    return W, M
