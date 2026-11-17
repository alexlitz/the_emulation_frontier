"""Hymba forward pass, transcribed from the remote modelling code (read as text, never executed).

Source:   nvidia/Hymba-1.5B-Base, revision c085c63815412892a7af9d0b53fad3421b2abdf8 (nvidia/Hymba-1.5B-Instruct
          bc2d2b43bc73d62ad8bfc9e6a3dcc26ab0795e77 ships byte-identical files)
Files:    configuration_hymba.py, sha256 22b7283f1f12a71c987f126a4fadfcf15ef890a229fc68d5ee2faf443114accd
              lines 5-116 (HymbaConfig: the fields the forward reads)
          modeling_hymba.py,      sha256 4393c3e2097b58b6e1e137dad0ee6289481dcaab7db5b3415f8cfd5b7f0e45cf
Lines transcribed (modeling_hymba.py):
    172-186    HymbaRMSNorm: weight * x / sqrt(mean x^2 + eps)
    231-307    LlamaRotaryEmbedding, rope_type default: inv_freq = base^(-2i/d) over kq_head_dim
    310-349    rotate_half / apply_rotary_pos_emb, 352-362 repeat_kv
    476-562    HymbaAttention (attn_only_wo_proj: q / k / v come from the block's in_proj, no o_proj)
    1058-1176  HymbaSdpaAttention: RoPE on q and k, consumers of a KV group take the producer's post-RoPE (k, v),
               SDPA with is_causal (the model passes no mask: _prepare_4d_causal_attention_mask_for_sdpa returns
               None for an unpadded batch whose length is under sliding_window)
    1179-1456  HymbaFlexAttention: the block mask causal AND (q - kv <= sliding_window OR kv < num_memory_tokens)
               on layers outside global_attn_idx, causal on global ones (memory_tokens_interspersed_every 0);
               flex_attention's default scale 1 / sqrt(head_dim); v wider than q is two flex calls (same op)
    1471-1711  HymbaBlock.cuda_kernels_forward: in_proj -> [q | k | v | x] (or [q | x] when reuse_kv) and gate;
               causal depthwise conv + bias + silu on x (causal_conv1d_fn); attention on q / k / v; x_proj ->
               RMS-normed dt / B / C; dt_proj without its bias, the bias added inside the scan
               (selective_scan_fn = mamba_ssm 2.3.2.post1 selective_scan_ref, ops/selective_scan_interface.py
               115-181: delta = softplus(dt + bias), x_t = exp(delta A) x_{t-1} + delta B u, y = C . x_t + D u,
               y * silu(gate)); out = out_proj((norm1(attn) + norm2(ssm)) / 2)
    1738-1760  HymbaMLP (silu: down(silu(gate x) * up x)), 1763-1812 HymbaSparseMoeBlock with num_experts 1
    1858-1955  HymbaDecoderLayer: x + mixer(input_layernorm x), then x + moe(pre_moe_layernorm x)
    2077-2414  HymbaModel: embed, the learned memory (meta) tokens packed in front, position ids 0..M+T-1 over the
               packed sequence, the KV-reuse groups (producer = group[0]), final_layernorm, memory positions dropped
    2417-2530  HymbaForCausalLM: lm_head

Precision: the remote code computes its RMS norms, the selective scan and the RoPE table in fp32 and casts the
logits to fp32. This transcription runs everything in the parameters' dtype (fp64 in the tests): the fp32
islands are lifted, so the host can be compared at fp64 rounding.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class HymbaConfig:
    model_type = "hymba"

    def __init__(self, vocab_size=65536, tie_word_embeddings=False, hidden_size=4096, intermediate_size=14336,
                 num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8, rms_norm_eps=1e-6,
                 sliding_window=None, num_experts=16, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 mamba_dt_rank="auto", mamba_conv_bias=True, mamba_proj_bias=False, mamba_inner_layernorms=True,
                 kv_reuse_group=None, global_attn_idx=None, attn_implementation_new="sdpa", rope_type=None,
                 attn_hidden_size=-1, kq_head_dim=-1, v_head_dim=-1, kq_norm=None, rope=False, rope_theta=10000.0,
                 num_memory_tokens=0, memory_tokens_interspersed_every=0, mlp_hidden_act="silu", seq_length=None):
        self.vocab_size, self.tie_word_embeddings = vocab_size, tie_word_embeddings
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.num_hidden_layers, self.num_attention_heads = num_hidden_layers, num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.rms_norm_eps, self.sliding_window, self.num_experts = rms_norm_eps, sliding_window, num_experts
        self.mamba_d_state, self.mamba_d_conv, self.mamba_expand = mamba_d_state, mamba_d_conv, mamba_expand
        self.mamba_dt_rank = math.ceil(hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias, self.mamba_proj_bias = mamba_conv_bias, mamba_proj_bias
        self.mamba_inner_layernorms = mamba_inner_layernorms
        self.attn_hidden_size, self.kq_head_dim, self.v_head_dim = attn_hidden_size, kq_head_dim, v_head_dim
        self.kq_norm, self.rope, self.rope_theta, self.rope_type = kq_norm, rope, rope_theta, rope_type
        self.num_memory_tokens = num_memory_tokens
        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every
        self.kv_reuse_group, self.global_attn_idx = kv_reuse_group, global_attn_idx
        self.attn_implementation_new, self.mlp_hidden_act, self.seq_length = attn_implementation_new, mlp_hidden_act, seq_length
        assert kq_norm in (None, "none") and memory_tokens_interspersed_every == 0 and rope_type in (None, "default")
        assert attn_hidden_size in (-1, hidden_size) and kq_head_dim == -1 and mamba_inner_layernorms and rope


class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)),
                             persistent=False)

    def forward(self, x, position_ids):
        freqs = position_ids[:, :, None].to(self.inv_freq.dtype) * self.inv_freq            # (b, T, dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class HymbaAttention(nn.Module):
    """attn_only_wo_proj: q / k / v arrive from the block; RoPE, KV reuse, the masked softmax."""

    def __init__(self, config, layer_idx, reuse_kv=False):
        super().__init__()
        self.config, self.layer_idx, self.reuse_kv = config, layer_idx, reuse_kv
        self.num_heads = config.num_attention_heads
        self.head_dim = self.kq_head_dim = config.hidden_size // self.num_heads
        self.v_head_dim = config.v_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rotary_emb = LlamaRotaryEmbedding(self.kq_head_dim, base=config.rope_theta)

    def forward(self, q, k, v, position_ids, kv_last_layer=None):
        b, T, _ = q.shape
        q = q.view(b, T, self.num_heads, self.kq_head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(q, position_ids)
        cos, sin = cos[:, None], sin[:, None]
        q = q * cos + rotate_half(q) * sin
        if self.reuse_kv:
            k, v = kv_last_layer
        else:
            k = k.view(b, T, self.num_key_value_heads, self.kq_head_dim).transpose(1, 2)
            v = v.view(b, T, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)
            k = k * cos + rotate_half(k) * sin
        kv = (k, v)
        k = k.repeat_interleave(self.num_key_value_groups, 1)
        v = v.repeat_interleave(self.num_key_value_groups, 1)
        c = self.config
        t = torch.arange(T)
        allowed = t[:, None] >= t[None, :]
        if (c.attn_implementation_new == "flex" and c.sliding_window is not None and c.global_attn_idx is not None
                and self.layer_idx not in c.global_attn_idx):
            allowed = allowed & ((t[:, None] - t[None, :] <= c.sliding_window) | (t[None, :] < c.num_memory_tokens))
        elif c.attn_implementation_new == "sdpa":            # no mask is built: plain causal (see the header)
            assert c.sliding_window is None or T - c.num_memory_tokens < c.sliding_window
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.kq_head_dim)
        o = torch.softmax(s.masked_fill(~allowed, float("-inf")), -1) @ v
        return o.transpose(1, 2).reshape(b, T, self.v_head_dim * self.num_heads), kv


class HymbaBlock(nn.Module):
    def __init__(self, config, layer_idx, reuse_kv=False):
        super().__init__()
        self.config, self.layer_idx, self.reuse_kv = config, layer_idx, reuse_kv
        self.hidden_size = config.hidden_size
        self.ssm_state_size, self.conv_kernel_size = config.mamba_d_state, config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * config.hidden_size)
        self.attn_hidden_size = config.hidden_size
        self.num_attention_heads, self.num_key_value_heads = config.num_attention_heads, config.num_key_value_heads
        config.v_head_dim = self.intermediate_size // self.num_attention_heads
        self.k_hidden_size = int(self.num_key_value_heads / self.num_attention_heads * self.attn_hidden_size)
        self.v_hidden_size = int(self.num_key_value_heads / self.num_attention_heads * self.attn_hidden_size
                                 * config.mamba_expand)
        self.self_attn = HymbaAttention(config, layer_idx, reuse_kv=reuse_kv)
        self.time_step_rank = config.mamba_dt_rank
        I = self.intermediate_size
        self.latent_dim = I + self.attn_hidden_size + (0 if reuse_kv else self.k_hidden_size + self.v_hidden_size)
        eps = config.rms_norm_eps
        self.pre_avg_layernorm1 = HymbaRMSNorm(I, eps=eps)
        self.pre_avg_layernorm2 = HymbaRMSNorm(I, eps=eps)
        self.in_proj = nn.Linear(self.hidden_size, self.latent_dim + I, bias=config.mamba_proj_bias)
        self.out_proj = nn.Linear(I, self.hidden_size, bias=config.mamba_proj_bias)
        self.conv1d = nn.Conv1d(I, I, bias=config.mamba_conv_bias, kernel_size=self.conv_kernel_size, groups=I,
                                padding=self.conv_kernel_size - 1)
        R, N = self.time_step_rank, self.ssm_state_size
        self.x_proj = nn.ModuleList([nn.Linear(I, R + 2 * N, bias=False)])
        self.dt_proj = nn.ModuleList([nn.Linear(R, I, bias=True)])
        A = torch.arange(1, N + 1, dtype=torch.float32)[None, :].expand(I, -1).contiguous()
        self.A_log = nn.ParameterList([nn.Parameter(torch.log(A))])
        self.D = nn.ParameterList([nn.Parameter(torch.ones(I))])
        self.dt_layernorm = HymbaRMSNorm(R, eps=eps)
        self.B_layernorm = HymbaRMSNorm(N, eps=eps)
        self.C_layernorm = HymbaRMSNorm(N, eps=eps)

    def forward(self, h, position_ids, kv_last_layer=None):
        b, L, _ = h.shape
        I, R, N, K = self.intermediate_size, self.time_step_rank, self.ssm_state_size, self.conv_kernel_size
        p = self.in_proj(h)
        lat, gate = p[..., : self.latent_dim], p[..., self.latent_dim:]
        A0 = self.attn_hidden_size
        if self.reuse_kv:
            q, x = lat[..., :A0], lat[..., A0:]
            k = v = None
        else:
            kd, vd = self.k_hidden_size, self.v_hidden_size
            q, k, v, x = lat[..., :A0], lat[..., A0:A0 + kd], lat[..., A0 + kd:A0 + kd + vd], lat[..., A0 + kd + vd:]
        # causal depthwise conv + bias + silu
        xt = F.pad(x.transpose(1, 2), (K - 1, 0))
        w = self.conv1d.weight[:, 0, :]
        x = torch.stack([(xt[:, :, i: i + K] * w).sum(-1) for i in range(L)], -1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias[:, None]
        x = F.silu(x).transpose(1, 2)                                                    # (b, L, I)
        attn, kv = self.self_attn(q, k, v, position_ids, kv_last_layer)
        dt, B, C = self.x_proj[0](x).split([R, N, N], dim=-1)
        dt, B, C = self.dt_layernorm(dt), self.B_layernorm(B), self.C_layernorm(C)
        delta = F.softplus(F.linear(dt, self.dt_proj[0].weight) + self.dt_proj[0].bias)  # (b, L, I)
        A = -torch.exp(self.A_log[0])                                                    # (I, N)
        s = x.new_zeros(b, I, N)
        ys = []
        for i in range(L):
            s = torch.exp(delta[:, i, :, None] * A) * s + delta[:, i, :, None] * B[:, i, None, :] * x[:, i, :, None]
            ys.append((s * C[:, i, None, :]).sum(-1))
        y = (torch.stack(ys, 1) + x * self.D[0]) * F.silu(gate)
        return self.out_proj((self.pre_avg_layernorm1(attn) + self.pre_avg_layernorm2(y)) / 2), kv


class HymbaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mlp_hidden_act == "silu"
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HymbaSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([HymbaMLP(config)])                    # num_experts 1: no router

    def forward(self, x):
        return self.experts[0](x)


class HymbaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, reuse_kv=False):
        super().__init__()
        self.mamba = HymbaBlock(config, layer_idx, reuse_kv=reuse_kv)
        self.input_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = HymbaSparseMoeBlock(config)
        self.pre_moe_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h, position_ids, kv_last_layer=None):
        a, kv = self.mamba(self.input_layernorm(h), position_ids, kv_last_layer)
        h = h + a
        return h + self.moe(self.pre_moe_layernorm(h)), kv


class HymbaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        groups = config.kv_reuse_group or []
        self.kv_reuse_group = [{"producer": g[0], "consumer": g[1:]} for g in groups]
        self.layers = nn.ModuleList(
            HymbaDecoderLayer(config, i, reuse_kv=any(i in g["consumer"] for g in self.kv_reuse_group))
            for i in range(config.num_hidden_layers))
        self.final_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(config.num_memory_tokens, config.hidden_size))

    def forward(self, ids):
        h = self.embed_tokens(ids)
        M = self.config.num_memory_tokens
        if M > 0:
            h = torch.cat([self.memory_tokens.expand(h.shape[0], -1, -1), h], 1)
        position_ids = torch.arange(h.shape[1])[None]
        shared = {}
        for i, layer in enumerate(self.layers):
            kv_in = next((shared[j] for j, g in enumerate(self.kv_reuse_group) if i in g["consumer"]), None)
            h, kv = layer(h, position_ids, kv_in)
            for j, g in enumerate(self.kv_reuse_group):
                if i == g["producer"]:
                    shared[j] = kv
        h = self.final_layernorm(h)
        return h[:, M:]


class HymbaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = HymbaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, ids):
        return self.lm_head(self.model(ids))
