"""PLaMo-2 forward pass, transcribed from the remote modelling code (read as text, never executed).

Source:   pfnet/plamo-2-1b, revision 92c75fd6eea9018bcb9c33ee8921589febe071fa
File:     modeling_plamo.py, sha256 21c0e1d8ba2f777c69305259254da865cf79145c1e06fee4fd26ed926b82451e
Lines transcribed:
    43-46      _swiglu
    48-100     RotaryEmbedding, _rotate_half, _rotary_pos_emb (rotate-half RoPE, positions 0..T-1)
    109-210    Plamo2Config (the fields the forward reads) and is_mamba (1243-1252)
    426-454    _rms_norm / RMSNorm: (offset + weight) * x / sqrt(mean x^2 + eps)
    456-468    get_initial_dt_bias / get_initial_A (initialisation only)
    485-527    ssd_update_state -> mamba_ssm selective_state_update_ref (mamba_ssm 2.3.2.post1
               ops/triton/selective_state_update.py 224-284): dt = softplus(dt + dt_bias), state =
               exp(dt A) state + dt B x, y = C . state + D x, y * silu(z)
    529-566    _ssd_chunk_scan_combined_naive: the per-token scan the CPU path takes when an attention
               mask is present (it always is in eval: Plamo2Model.forward 1446-1454 builds one)
    729-768    _causal_conv1d_update / _causal_conv1d_naive: the per-token causal depthwise conv + silu
    826-1025   Mamba (in_proj per head [z | x], conv on x, bcdt_proj -> RMS-normed B, C, dt with learned
               weights, dt_proj per head, the scan, out_proj)
    1028-1035  swa_mask: key s visible to query t iff |t - s| <= attention_window_size (and causal)
    1038-1157  Attention (fused qkv, weight-free RMS q / k with eps 1e-6 times per-head q_weight / k_weight,
               RoPE with rope_theta on full_attention_idx layers and rope_local_theta elsewhere, GQA by
               repeat_interleave, SDPA under the causal (+ window) mask, o_proj)
    1160-1172  MLP (gate_up_proj -> silu(gate) * up -> down_proj)
    1175-1240  Plamo2DecoderLayer (sandwich norms: pre / post mixer, pre / post MLP; post offsets 1/5 and
               1/5^1.5)
    1255-1296  Plamo2Decoder (model.layers.layers), 1320-1497 Plamo2Model (embed, final norm, offset 1),
    1500-1627  Plamo2ForCausalLM (lm_head padded to a multiple of 16, logits sliced to vocab_size)

Precision: the remote code computes RMS norms in fp32 (_rms_norm), keeps the SSM state in fp32 and builds
its RoPE tables in fp32. This transcription runs everything in the parameters' dtype (fp64 in the tests): the
fp32 islands are lifted, so the host can be compared at fp64 rounding.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Plamo2Config:
    model_type = "plamo2"

    def __init__(self, hidden_size=4096, num_hidden_layers=32, rms_norm_eps=1e-6, tie_word_embeddings=True,
                 num_attention_heads=32, num_key_value_heads=4, hidden_size_per_head=128,
                 max_position_embeddings=2048, attention_window_size=2048, full_attention_idx=None,
                 rope_theta=10000, rope_local_theta=10000, mamba_d_state=64, mamba_d_conv=4, mamba_num_heads=64,
                 mamba_step=2, mamba_chunk_size=256, mamba_enabled=True, intermediate_size=13312,
                 vocab_size=32000):
        self.max_position_embeddings = max(10 * 1024 * 1024, max_position_embeddings)
        self.hidden_size, self.rms_norm_eps = hidden_size, rms_norm_eps
        self.num_hidden_layers, self.num_attention_heads = num_hidden_layers, num_attention_heads
        self.hidden_size_per_head, self.num_key_value_heads = hidden_size_per_head, num_key_value_heads
        self.attention_window_size = attention_window_size
        self.full_attention_idx = full_attention_idx if full_attention_idx is not None else []
        self.rope_theta, self.rope_local_theta = rope_theta, rope_local_theta
        self.mamba_d_state, self.mamba_d_conv, self.mamba_num_heads = mamba_d_state, mamba_d_conv, mamba_num_heads
        self.mamba_step, self.mamba_chunk_size, self.mamba_enabled = mamba_step, mamba_chunk_size, mamba_enabled
        self.intermediate_size, self.vocab_size = intermediate_size, vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.sliding_window = attention_window_size

    @property
    def layers_block_type(self):
        return ["mamba" if is_mamba(self, i) else "attention" for i in range(self.num_hidden_layers)]


def is_mamba(config, i):
    if not config.mamba_enabled:
        return False
    assert config.mamba_step > 1 and i < config.num_hidden_layers
    if config.num_hidden_layers <= (config.mamba_step // 2):
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


def _rms_norm(x, weight, eps, offset=1.0):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (offset + weight) * x if weight is not None else x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, offset=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon, self.offset = eps, offset

    def forward(self, x):
        return _rms_norm(x, self.weight, self.variance_epsilon, offset=self.offset)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim, self.base = dim, base
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)),
                             persistent=False)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, -1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)               # (T, dim)


def get_initial_dt_bias(num_heads):
    dt = torch.exp(torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(1e-4)
    return dt + torch.log(-torch.expm1(-dt))


class Mamba(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.hidden_size, self.d_state, self.d_conv = config.hidden_size, config.mamba_d_state, config.mamba_d_conv
        self.chunk_size, self.num_heads = config.mamba_chunk_size, config.mamba_num_heads
        self.hidden_size_per_head = config.hidden_size_per_head
        self.intermediate_size = self.num_heads * self.hidden_size_per_head
        self.in_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.conv1d = nn.Conv1d(self.intermediate_size, self.intermediate_size, bias=False, kernel_size=self.d_conv,
                                groups=self.intermediate_size, padding=0)
        self.dt_dim = max(64, self.hidden_size // 16)
        self.bcdt_proj = nn.Linear(self.intermediate_size, self.dt_dim + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_dim, self.num_heads, bias=False)
        self.dt_bias = nn.Parameter(get_initial_dt_bias(self.num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.dt_norm_weight = nn.Parameter(torch.ones(self.dt_dim))
        self.B_norm_weight = nn.Parameter(torch.ones(self.d_state))
        self.C_norm_weight = nn.Parameter(torch.ones(self.d_state))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.bcdt_norm = True                       # test switch (not in the remote code): the B / C / dt RMS norms

    def forward(self, h):
        b, L, _ = h.shape
        nh, P, N = self.num_heads, self.hidden_size_per_head, self.d_state
        z, x = self.in_proj(h).reshape(b, L, nh, 2 * P).split([P, P], dim=-1)
        # causal depthwise conv (width d_conv, zero initial state) + silu, token by token
        x = x.reshape(b, L, -1).transpose(1, 2)                           # (b, I, L)
        w = self.conv1d.weight[:, 0, :]                                   # (I, K)
        xp = F.pad(x, (self.d_conv - 1, 0))
        x = F.silu(torch.stack([(xp[:, :, i: i + self.d_conv] * w).sum(-1) for i in range(L)], -1))
        x = x.transpose(1, 2)                                             # (b, L, I)
        B, C, dt = self.bcdt_proj(x).split([N, N, self.dt_dim], dim=-1)
        x = x.reshape(b, L, nh, P)
        A = -torch.exp(self.A_log)
        eps = self.config.rms_norm_eps
        if self.bcdt_norm:
            dt = _rms_norm(dt, None, eps) * self.dt_norm_weight
            B = _rms_norm(B, None, eps) * self.B_norm_weight
            C = _rms_norm(C, None, eps) * self.C_norm_weight
        dt = self.dt_proj(dt)                                             # (b, L, nh)
        state = x.new_zeros(b, nh, P, N)
        ys = []
        for i in range(L):                                                # ssd_update_state, per token
            d = F.softplus(dt[:, i] + self.dt_bias)[:, :, None, None]     # (b, nh, 1, 1)
            state = state * torch.exp(d * A[:, None, None]) + d * B[:, i, None, None, :] * x[:, i, :, :, None]
            y = (state * C[:, i, None, None, :]).sum(-1) + x[:, i] * self.D[:, None]
            ys.append(y * F.silu(z[:, i]))
        return self.out_proj(torch.stack(ys, 1).reshape(b, L, -1))


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        hd = config.hidden_size_per_head
        self.q_num_heads, self.qk_dim, self.v_dim = config.num_attention_heads, hd, hd
        self.k_num_heads = self.v_num_heads = config.num_key_value_heads
        self.n_group = self.q_num_heads // self.k_num_heads
        self.q_proj_dim, self.k_proj_dim = self.q_num_heads * hd, self.k_num_heads * hd
        self.v_proj_dim = self.k_num_heads * hd
        self.qkv_proj = nn.Linear(config.hidden_size, self.q_proj_dim + self.k_proj_dim + self.v_proj_dim, bias=False)
        self.o_proj = nn.Linear(self.q_num_heads * hd, config.hidden_size, bias=False)
        self.q_weight = nn.Parameter(torch.ones(self.q_num_heads, hd))
        self.k_weight = nn.Parameter(torch.ones(self.k_num_heads, hd))
        self.full_attn = layer_idx in config.full_attention_idx
        self.rotary_emb = RotaryEmbedding(hd, base=config.rope_theta if self.full_attn else config.rope_local_theta)

    def forward(self, h):
        b, T, _ = h.shape
        q, k, v = self.qkv_proj(h).split([self.q_proj_dim, self.k_proj_dim, self.v_proj_dim], dim=-1)
        q = q.view(b, T, self.q_num_heads, self.qk_dim).transpose(1, 2)
        k = k.view(b, T, self.k_num_heads, self.qk_dim).transpose(1, 2)
        v = v.view(b, T, self.v_num_heads, self.v_dim).transpose(1, 2)
        q = _rms_norm(q, None, 1e-6) * self.q_weight[None, :, None]
        k = _rms_norm(k, None, 1e-6) * self.k_weight[None, :, None]
        cos, sin = self.rotary_emb(v, T)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        k = torch.repeat_interleave(k, self.n_group, dim=1)
        v = torch.repeat_interleave(v, self.n_group, dim=1)
        t = torch.arange(T)
        allowed = t[:, None] >= t[None, :]
        if not self.full_attn:
            allowed &= (t[:, None] - t[None, :]).abs() <= self.config.attention_window_size
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.qk_dim)
        o = torch.softmax(s.masked_fill(~allowed, float("-inf")), -1) @ v
        return self.o_proj(o.transpose(1, 2).reshape(b, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        g, u = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(g) * u)


class Plamo2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_mamba = config.layers_block_type[layer_idx] == "mamba"
        self.mixer = Mamba(config, layer_idx) if self.is_mamba else Attention(config, layer_idx)
        self.mlp = MLP(config)
        eps = config.rms_norm_eps
        self.pre_mixer_norm = RMSNorm(config.hidden_size, eps=eps, offset=1.0)
        self.post_mixer_norm = RMSNorm(config.hidden_size, eps=eps, offset=1.0 / 5)
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=eps, offset=1.0)
        self.post_mlp_norm = RMSNorm(config.hidden_size, eps=eps, offset=1.0 / (5 ** 1.5))

    def forward(self, h):
        h = h + self.post_mixer_norm(self.mixer(self.pre_mixer_norm(h)))
        return h + self.post_mlp_norm(self.mlp(self.pre_mlp_norm(h)))


class Plamo2Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(Plamo2DecoderLayer(config, i) for i in range(config.num_hidden_layers))


class Plamo2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = Plamo2Decoder(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, ids):
        h = self.embed_tokens(ids)
        for layer in self.layers.layers:
            h = layer(h)
        return self.norm(h)


class Plamo2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Plamo2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, ((config.vocab_size + 15) // 16) * 16, bias=False)

    def forward(self, ids):
        return self.lm_head(self.model(ids))[..., : self.vocab_size]
