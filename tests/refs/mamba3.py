"""REFERENCE transcription (read as text, never run) of the Mamba-3 MIMO token mixer.

Source: arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias0, snapshot 4695f9327c4150c0fa4e5862b373a11e3f27ac19,
modeling_qwen3_mamba3.py, sha256 6ca2cee4665ac88b28cb81470179f635287e983a28c4079afe2b085ceaf55f77.
Transcribed: l. 126-227 Qwen3Mamba3Config.__init__ (the mixer's fields; the Qwen3Config base contributes
rms_norm_eps = 1e-6, its default), l. 287-296 mamba3_rotary_dim_divisor / mamba3_num_rope_angles,
l. 338-344 heavy_tail_activation, l. 347-359 _rotate_half_partial, l. 362-463 mamba3_mimo_reference
(no initial states), l. 466-494 Mamba3GatedRMSNorm (its parameters; the reference path inlines the norm),
l. 516-570 Qwen3Mamba3Mixer.__init__ and l. 639-719 its forward on the reference path (CPU: the TileLang
kernel is CUDA-only; no cache, no padding mask). mamba_ssm 2.3.2 (the library) has no Mamba-3 module.
B_norm / C_norm are transformers' Qwen3RMSNorm (weight * x * rsqrt(mean x^2 + variance_epsilon)), written
out here. Every `.float()` / float32 cast (the reference's fp32 compute, Qwen3RMSNorm's fp32 upcast) is
lifted to the input dtype (fp64): that is the only change.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3Mamba3Config:
    def __init__(self, hidden_size=2560, mamba_n_heads=32, mamba_d_head=128, mamba_d_state=128, mamba_n_groups=8,
                 mamba_mimo_rank=4, mamba_expand=1.6, mamba_chunk_size=8, mamba_rope_fraction=0.5,
                 mamba_bc_bias_init=1.0, mamba_a_floor=1e-4, mamba_dt_min=0.001, mamba_dt_max=0.1,
                 mamba_dt_init_floor=1e-4, mamba_norm_eps=1e-5, mamba_bc_norm_eps=None, rms_norm_eps=1e-6, **kw):
        self.hidden_size = hidden_size
        self.mamba_n_heads, self.mamba_d_head, self.mamba_d_state = mamba_n_heads, mamba_d_head, mamba_d_state
        self.mamba_n_groups, self.mamba_mimo_rank, self.mamba_expand = mamba_n_groups, mamba_mimo_rank, mamba_expand
        self.mamba_chunk_size, self.mamba_rope_fraction = mamba_chunk_size, mamba_rope_fraction
        self.mamba_bc_bias_init, self.mamba_a_floor = mamba_bc_bias_init, mamba_a_floor
        self.mamba_dt_min, self.mamba_dt_max, self.mamba_dt_init_floor = mamba_dt_min, mamba_dt_max, mamba_dt_init_floor
        self.mamba_norm_eps, self.mamba_bc_norm_eps, self.rms_norm_eps = mamba_norm_eps, mamba_bc_norm_eps, rms_norm_eps
        self.__dict__.update(kw)
        assert int(self.mamba_expand * self.hidden_size) == self.mamba_n_heads * self.mamba_d_head
        assert self.mamba_n_heads % self.mamba_n_groups == 0 and self.mamba_rope_fraction in (0.5, 1.0)


def mamba3_rotary_dim_divisor(config):
    return int(round(2 / config.mamba_rope_fraction))


def mamba3_num_rope_angles(config):
    split = int(config.mamba_d_state * config.mamba_rope_fraction)
    if split % 2:
        split -= 1
    return split // 2


def heavy_tail_activation(x):
    return x.clamp(min=0) + 1.0 / (1.0 - x.clamp(max=0))


def _rotate_half_partial(t, cos, sin):
    N, A = t.shape[-1], cos.shape[-1]
    half = N // 2
    t0, mid, t1, rest = t[..., :A], t[..., A:half], t[..., half:half + A], t[..., half + A:]
    return torch.cat([t0 * cos - t1 * sin, mid, t0 * sin + t1 * cos, rest], dim=-1)


def mamba3_mimo_reference(C, B, x, z, adt, dt, trap, C_bias, B_bias, mimo_x, mimo_z, mimo_o, angles, D,
                          norm_weight, norm_eps):
    b, l, R, G, N = C.shape
    H, P = x.shape[2], x.shape[3]
    hpg = H // G
    A = angles.shape[-1]
    dtp = x.dtype
    Cf = C.repeat_interleave(hpg, dim=3) + C_bias.permute(1, 0, 2)[None, None]
    Bf = B.repeat_interleave(hpg, dim=3) + B_bias.permute(1, 0, 2)[None, None]
    v_all = torch.einsum("blhp,hrp->blhrp", x, mimo_x)
    z_all = torch.einsum("blhp,hrp->blhrp", z, mimo_z)
    sig = torch.sigmoid(trap)
    alpha = torch.exp(adt)
    gamma = sig * dt
    beta = (1.0 - sig) * dt * alpha
    w = norm_weight.view(H, P)
    angle_state = torch.zeros(b, H, A, dtype=dtp)
    S = torch.zeros(b, H, P, N, dtype=dtp)
    k_prev = torch.zeros(b, H, R, N, dtype=dtp)
    v_prev = torch.zeros(b, H, R, P, dtype=dtp)
    ys = []
    for t in range(l):
        q = Cf[:, t].permute(0, 2, 1, 3)
        k = Bf[:, t].permute(0, 2, 1, 3)
        v = v_all[:, t]
        angle_state = angle_state + torch.tanh(angles[:, t]) * dt[:, :, t, None] * math.pi
        cos, sin = angle_state.cos()[:, :, None, :], angle_state.sin()[:, :, None, :]
        q, k = _rotate_half_partial(q, cos, sin), _rotate_half_partial(k, cos, sin)
        prev_kv = torch.einsum("bhrn,bhrp->bhpn", k_prev, v_prev)
        curr_kv = torch.einsum("bhrn,bhrp->bhpn", k, v)
        S = alpha[:, :, t, None, None] * S + beta[:, :, t, None, None] * prev_kv + gamma[:, :, t, None, None] * curr_kv
        out = torch.einsum("bhpn,bhrn->bhrp", S, q) + D[None, :, None, None] * v
        out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + norm_eps)
        out = out * w[None, :, None, :] * F.silu(z_all[:, t])
        ys.append(torch.einsum("bhrp,hrp->bhp", out, mimo_o))
        k_prev, v_prev = k, v
    return torch.stack(ys, dim=1)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))


class Mamba3GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps, self.group_size = eps, group_size


class Qwen3Mamba3Mixer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads, self.head_dim = config.mamba_n_heads, config.mamba_d_head
        self.ssm_state_size, self.n_groups = config.mamba_d_state, config.mamba_n_groups
        self.mimo_rank = config.mamba_mimo_rank
        self.intermediate_size = int(config.mamba_expand * config.hidden_size)
        self.chunk_size, self.a_floor = config.mamba_chunk_size, config.mamba_a_floor
        self.rotary_dim_divisor = mamba3_rotary_dim_divisor(config)
        self.num_rope_angles = mamba3_num_rope_angles(config)
        H, R, N, P, G = self.num_heads, self.mimo_rank, self.ssm_state_size, self.head_dim, self.n_groups
        bc_eps = config.mamba_bc_norm_eps if config.mamba_bc_norm_eps is not None else config.rms_norm_eps
        self.split_sizes = [self.intermediate_size, self.intermediate_size, N * G * R, N * G * R, H, H, H,
                            self.num_rope_angles]
        self.in_proj = nn.Linear(self.hidden_size, sum(self.split_sizes), bias=False)
        self.dt_bias = nn.Parameter(torch.ones(H))
        self.B_bias = nn.Parameter(torch.zeros(H, R, N))
        self.C_bias = nn.Parameter(torch.zeros(H, R, N))
        self.B_norm = Qwen3RMSNorm(N, eps=bc_eps)
        self.C_norm = Qwen3RMSNorm(N, eps=bc_eps)
        self.mimo_x = nn.Parameter(torch.full((H, R, P), 1.0 / R))
        self.mimo_z = nn.Parameter(torch.ones(H, R, P))
        self.mimo_o = nn.Parameter(torch.full((H, R, P), 1.0 / R))
        self.D = nn.Parameter(torch.ones(H))
        self.norm = Mamba3GatedRMSNorm(self.intermediate_size, group_size=P, eps=config.mamba_norm_eps)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        b, l, _ = hidden_states.shape
        H, R, N, P, G = self.num_heads, self.mimo_rank, self.ssm_state_size, self.head_dim, self.n_groups
        proj = self.in_proj(hidden_states)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(proj, self.split_sizes, dim=-1)
        z, x = z.view(b, l, H, P), x.view(b, l, H, P)
        B = self.B_norm(B.view(b, l, R, G, N))
        C = self.C_norm(C.view(b, l, R, G, N))
        dt = F.softplus(dd_dt + self.dt_bias)
        A = torch.clamp(-heavy_tail_activation(dd_A), max=-self.a_floor)
        dt_t, adt_t, trap_t = dt.transpose(1, 2), (A * dt).transpose(1, 2), trap.transpose(1, 2)
        angles = angles.unsqueeze(2).expand(b, l, H, self.num_rope_angles)
        y = mamba3_mimo_reference(C, B, x, z, adt_t, dt_t, trap_t, self.C_bias, self.B_bias, self.mimo_x,
                                  self.mimo_z, self.mimo_o, angles, self.D, self.norm.weight, self.norm.eps)
        return self.out_proj(y.reshape(b, l, self.intermediate_size))
