"""REFERENCE transcription: Gated KalmaNet (GKA) and HybridQwen3 from awslabs hybrid-model-factory.

Source: https://github.com/awslabs/hybrid-model-factory, source tarball at commit
b9979c5ac74915dc529868acea8d35e522e2217c (the tree elephantine/fetch_upstream_refs.py unpacks into
~/.cache/upstream_refs/hybrid-model-factory). The files were READ AS TEXT ONLY, never imported or run.
Paths below are relative to training/src/hmf/model/hybrid_zoo/.

  layers/gated_kalmanet/gka.py                                 sha256 d3ced729a18bcf1099f2f60b81ffc6616cd0ff28119e40f8479165977a639b3d
      GatedKalmaNet.__init__        112-269   (kv_proj_rank / LinearLowRank branch 239-269 not transcribed: raises)
      GatedKalmaNet._expand_kv      271-331   (plain repeat branch 322-329)
      GatedKalmaNet.forward         333-696   (single-device path: 387-393, 415-449, 462-508, 614-631 prefill,
                                                641-662 decode, 673-684; the sequence-parallel branches are skipped)
  layers/gated_kalmanet/modules.py                             sha256 0990449e641490d39e70e67330a8d8cbfd8368fa8c0b17e323c0604cf70877de
      GatedKalmaNetLayer            17-96
  layers/gated_kalmanet/gka_config.py                          sha256 df5be31c0af58b462ff1a0a6bac3c6d64f605a3f07405f7697158d4eab2dfc24
      _construct_from_transformer   129-165   (the Qwen-GDN branch 90-126 is not used by HybridQwen3)
  layers/gated_kalmanet/ops/chebyshev/gka_chebyshev_solve.py   sha256 d93180d1801759c5a31a96a0c8e064ad6759104a497a3a1d0a7331b5f417dd86
      gka_chebyshev_gla             107-164   (prefill: ridge solve then chunk_simple_gla)
      chebyshev_iteration_fro_inference_forward_kernel 169-266 (the decode solve, transcribed as `_cheb_step`)
      torch_decoding_one_step       324-397
  layers/gated_kalmanet/ops/chebyshev/chebyshev_iteration.py   sha256 1d78d739d87cf099b1be542fc6632db86c6674d6f50a6adb7ee23c7523d6dee2
      chebyshev_iteration_fro_gating_forward_kernel 308-459 (the chunked prefill solve, transcribed as `_cheb_chunk`)
      chebyshev_iteration_forward_triton 462-557, ChebyshevIteration.forward 568-603
  models/hybrid_qwen3/modeling_hybrid_qwen3.py                 sha256 bd9a8761cd4df1e03bfd8856fc59ddeddb9d7572cd56e42aaf7dfe524188fc15
      Qwen3RMSNorm 85-99, Qwen3MLP 105-118, Qwen3RotaryEmbedding 121-183, rotate_half / apply_rotary_pos_emb
      186-216, repeat_kv 219-228, eager_attention_forward 231-253, Qwen3Attention 257-331, Qwen3DecoderLayer
      424-472, Qwen3HybridDecoderLayerBase 503-599, Qwen3GKADecoderLayer 643-651, HybridQwen3Model 1211-1324,
      HybridQwen3ForCausalLM 1328-1396
  models/hybrid_qwen3/configuration_hybrid_qwen3.py            sha256 954464d11b6693813bb6d44d85bc0d769c137aa62de3d150ea9317a2910b9d5d
      HybridQwen3Config.__init__    116-200

Library pieces the remote file imports from fla (fla.modules.ShortConvolution, FusedRMSNormGated, RMSNorm;
fla.ops.simple_gla.chunk_simple_gla / fused_recurrent_simple_gla) are written here as plain torch with the
same parameter names (ShortConvolution: a depthwise nn.Conv1d, bias=False, padding K-1, silu;
FusedRMSNormGated: x * rsqrt(mean x^2 + eps) * weight * silu(g); simple GLA: S_t = exp(g_t) S_{t-1} + k_t v_t^T,
o_t = scale S_t^T q_t, chunked like fla's naive_chunk_simple_gla).

Precision: every fp32 cast of the remote code (the Triton kernels' fp32 accumulators, Qwen3RMSNorm's float32,
the rotary's "Force float32", softmax dtype=float32) is lifted to fp64; the RoPE inv_freq itself is built
exactly as the remote does (int64 arange -> float32 -> base ** (.../dim)), then used in fp64.

Two GKA evaluation modes, both transcribed (not fused, not copied from the host):
  mode="recurrent": token by token, the decode path (torch_decoding_one_step + the inference kernel):
      H_t = exp(gk_t) H_{t-1} + k_t k_t^T;  lam = ridge_strength ||H_t||_F;  30 Chebyshev steps on
      (H_t + lam I) x = q_t;  q <- q + alpha (x - q);  S_t = exp(g_t) S_{t-1} + k_t v_t^T;  o = scale S_t^T q
  mode="chunk": the prefill path (gka_chebyshev_gla): per 64-token chunk, the chunk-start state h_kk, the
      within-chunk cumulative gate gkk, ||H_t||_F from the kernel's expanded formula, the kernel's matvec, and
      the chunked simple GLA.
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

F64 = torch.float64


# ---------------------------------------------------------------- fla library pieces (plain torch)
class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu"):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias, padding=kernel_size - 1)
        self.hidden_size, self.activation = hidden_size, activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        T = x.shape[1]
        y = F.conv1d(x.transpose(1, 2), self.weight, self.bias, padding=self.kernel_size[0] - 1,
                     groups=self.hidden_size)[..., :T].transpose(1, 2)
        if self.activation is not None:
            y = F.silu(y)
        return y, None


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish"):
        super().__init__()
        self.hidden_size, self.eps, self.activation = hidden_size, eps, activation
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward(self, x, g):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return y * g * torch.sigmoid(g)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size, self.eps = hidden_size, eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _expand_heads(x, h, g):           # einops "... (h d) -> ... h d" then "... h d -> ... (h g) d"
    x = x.reshape(*x.shape[:-1], h, -1)
    return x.repeat_interleave(g, dim=-2) if g > 1 else x


# ---------------------------------------------------------------- the ridge solve
def _cheb(matvec, q, fro, ridge_ratio, num_iter):
    """The Chebyshev loop shared by both kernels (inference kernel 228-255, chunk kernel 408-434)."""
    ridge_strength = ridge_ratio * fro
    stepsize = 2 / (2 * ridge_strength + fro)
    rho_sq_4 = fro / (2 * ridge_strength + fro) / 2.0
    rho_sq_4 = rho_sq_4 * rho_sq_4
    x_prev = torch.zeros_like(q)
    x = stepsize * q
    omega = torch.full_like(fro, 2.0)
    for _ in range(num_iter):
        omega = 1.0 / (1.0 - rho_sq_4 * omega)
        grad_and_x_prev = stepsize * omega * (matvec(x) + ridge_strength * x - q) + (omega - 1) * x_prev
        x_prev = x
        x = omega * x - grad_and_x_prev
    return x


def _cheb_step(prev_h_kk, k, q, gk, num_iter, ridge_ratio):
    """chebyshev_iteration_fro_inference_forward_kernel, one token, all (B, H) at once.
    prev_h_kk [B,H,D,D], k / q [B,H,D], gk [B,H] or None."""
    gk_exp = gk.exp()[..., None, None] if gk is not None else 1.0
    kk = k[..., :, None] * k[..., None, :]
    h_kk_ = prev_h_kk * gk_exp + kk
    fro = h_kk_.pow(2).sum((-2, -1)).sqrt()[..., None]
    ge = gk.exp()[..., None] if gk is not None else 1.0
    mv = lambda x: (prev_h_kk * (x * ge)[..., :, None]).sum(-2) + (kk * x[..., :, None]).sum(-2)
    return _cheb(mv, q, fro, ridge_ratio, num_iter), h_kk_


def _cheb_chunk(k, q, gk, num_iter, ridge_ratio, chunk_size):
    """ChebyshevIteration.forward -> chebyshev_iteration_forward_triton -> the gating kernel.
    k / q [B,T,H,D], gk [B,T,H] or None. Returns x [B,T,H,D] and the final h_kk [B,H,D,D]."""
    B, T, H, D = q.shape
    k, q = k.transpose(1, 2), q.transpose(1, 2)                     # [B,H,T,D]
    g = gk.transpose(1, 2) if gk is not None else torch.zeros(B, H, T, dtype=q.dtype)
    h_kk = torch.zeros(B, H, D, D, dtype=q.dtype)
    xs = []
    for c0 in range(0, T, chunk_size):
        kc, qc = k[:, :, c0:c0 + chunk_size], q[:, :, c0:c0 + chunk_size]
        gkk = g[:, :, c0:c0 + chunk_size].cumsum(-1)                # chunk_local_cumsum
        n = kc.shape[2]
        tri = torch.ones(n, n, dtype=torch.bool).tril()
        mask = torch.where(tri, (gkk[..., :, None] - gkk[..., None, :]).exp(), torch.zeros((), dtype=q.dtype))
        exp_gkk = gkk.exp()[..., None]                              # [B,H,C,1]
        kkT = kc @ kc.transpose(-1, -2)                             # C x C
        fro = ((mask @ (kkT * kkT)) * mask).sum(-1, keepdim=True)
        fro = fro + exp_gkk * (exp_gkk * h_kk.pow(2).sum((-2, -1))[..., None, None])
        fro = (fro + 2 * exp_gkk * (mask * (kc * (kc @ h_kk)).sum(-1)[..., None, :]).sum(-1, keepdim=True)).sqrt()
        mv = lambda x: (x * exp_gkk) @ h_kk + ((x @ kc.transpose(-1, -2)) * mask) @ kc
        xs.append(_cheb(mv, qc, fro, ridge_ratio, num_iter))
        # chunk_fwd_h(k, v=k, g=gkk): the next chunk's starting state
        last = gkk[..., -1:]
        h_kk = h_kk * last[..., None].exp() + (kc * (last - gkk).exp()[..., None]).transpose(-1, -2) @ kc
    return torch.cat(xs, 2).transpose(1, 2), h_kk


def _gla_recurrent(q, k, v, g, scale, S=None):
    """fused_recurrent_simple_gla. q/k/v [B,T,H,D], g [B,T,H] or None."""
    B, T, H, D = q.shape
    S = torch.zeros(B, H, D, v.shape[-1], dtype=q.dtype) if S is None else S
    o = []
    for t in range(T):
        if g is not None:
            S = S * g[:, t].exp()[..., None, None]
        S = S + k[:, t, :, :, None] * v[:, t, :, None, :]
        o.append(scale * torch.einsum("bhd,bhde->bhe", q[:, t], S))
    return torch.stack(o, 1), S


def _gla_chunk(q, k, v, g, scale, chunk_size):
    """chunk_simple_gla, in the form of fla's naive_chunk_simple_gla (fp64, unpadded last chunk)."""
    B, T, H, D = q.shape
    q, k, v = (t.transpose(1, 2) for t in (q * scale, k, v))
    g = g.transpose(1, 2) if g is not None else torch.zeros(B, H, T, dtype=q.dtype)
    S = torch.zeros(B, H, D, v.shape[-1], dtype=q.dtype)
    o = []
    for c0 in range(0, T, chunk_size):
        qi, ki, vi = q[:, :, c0:c0 + chunk_size], k[:, :, c0:c0 + chunk_size], v[:, :, c0:c0 + chunk_size]
        dec = g[:, :, c0:c0 + chunk_size].cumsum(-1)
        L = (dec[..., :, None] - dec[..., None, :]).tril().exp().tril()
        o.append((qi * dec[..., None].exp()) @ S + ((qi @ ki.transpose(-1, -2)) * L) @ vi)
        S = S * dec[..., -1:, None].exp() + (ki * (dec[..., -1:] - dec).exp()[..., None]).transpose(-1, -2) @ vi
    return torch.cat(o, 2).transpose(1, 2), S


# ---------------------------------------------------------------- GatedKalmaNet (gka.py)
class GatedKalmaNet(nn.Module):
    def __init__(self, hidden_size=2048, head_dim=128, num_q_heads=16, num_k_heads=None, num_v_heads=None,
                 use_beta_gate=True, use_alpha_connection=True, use_v_conv=True, use_forgetting_gate=True,
                 use_forgetting_gate_kk=True, gla_rescale=True, solver_type="chebyshev", bp_lambda=True,
                 num_iter=30, ridge_strength=0.02, use_gate=True, conv_size=4, layer_idx=None, norm_eps=1e-6,
                 chunk_size=64, kv_proj_rank=None, kv_learnable_residual=False, **kwargs):
        super().__init__()
        assert solver_type == "chebyshev"
        if kv_proj_rank is not None:
            raise NotImplementedError("REFERENCE: the LinearLowRank KV expansion is not transcribed")
        self.hidden_size, self.use_gate, self.conv_size, self.chunk_size = hidden_size, use_gate, conv_size, chunk_size
        self.sequence_parallel_group = None
        self.head_dim, self.num_q_heads = head_dim, num_q_heads
        self.num_k_heads = num_k_heads if num_k_heads is not None else num_q_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_q_heads
        self.expanded_num_heads = max(self.num_q_heads, self.num_k_heads, self.num_v_heads)
        self.num_q_groups = self.expanded_num_heads // self.num_q_heads
        self.num_k_groups = self.expanded_num_heads // self.num_k_heads
        self.num_v_groups = self.expanded_num_heads // self.num_v_heads
        self.query_dim, self.key_dim = self.num_q_heads * head_dim, self.num_k_heads * head_dim
        self.value_dim, self.expanded_dim = self.num_v_heads * head_dim, self.expanded_num_heads * head_dim
        self.layer_idx = layer_idx
        self.ridge_strength, self.num_iter = ridge_strength, num_iter
        self.use_beta_gate, self.use_alpha_connection = use_beta_gate, use_alpha_connection
        self.solver_type, self.bp_lambda, self.use_v_conv = solver_type, bp_lambda, use_v_conv
        self.use_forgetting_gate, self.use_forgetting_gate_kk = use_forgetting_gate, use_forgetting_gate_kk
        self.gla_scale = self.head_dim ** -0.5 if gla_rescale else 1.0

        self.q_proj = nn.Linear(hidden_size, self.query_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=True) if use_beta_gate else None
        if use_alpha_connection:
            self.alpha_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=True)
        A = torch.empty(self.expanded_num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(self.expanded_num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.q_conv1d = ShortConvolution(self.query_dim, conv_size, activation="silu")
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu")
        if use_v_conv:
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.expanded_dim, bias=False)
            self.o_norm = FusedRMSNormGated(head_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(head_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.expanded_dim, hidden_size, bias=False)
        self.kv_proj_rank, self.kv_learnable_residual = kv_proj_rank, kv_learnable_residual

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False,
                output_attentions=False, mode="recurrent", **kwargs):
        q, _ = self.q_conv1d(self.q_proj(hidden_states))
        k, _ = self.k_conv1d(self.k_proj(hidden_states))
        v = self.v_conv1d(self.v_proj(hidden_states))[0] if self.use_v_conv else self.v_proj(hidden_states)
        q = _expand_heads(q, self.num_q_heads, self.num_q_groups)
        k = _expand_heads(k, self.num_k_heads, self.num_k_groups)
        v = _expand_heads(v, self.num_v_heads, self.num_v_groups)
        qk_norm_eps = 1e-6
        q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + qk_norm_eps)
        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + qk_norm_eps)
        if self.use_beta_gate:
            beta = self.b_proj(hidden_states).sigmoid()
            eps = 1e-6
            k = (beta[..., None] + eps) * k
            v = (beta[..., None] + eps) * v
        alpha = self.alpha_proj(hidden_states).sigmoid() if self.use_alpha_connection else torch.ones_like(q[..., 0])
        g = (-self.A_log.exp() * F.softplus(self.a_proj(hidden_states) + self.dt_bias)
             if self.use_forgetting_gate else None)
        gk = g if self.use_forgetting_gate_kk else None

        if mode == "chunk":                                         # gka_chebyshev_gla
            if self.num_iter > 0:
                q_, _ = _cheb_chunk(k, q, gk, self.num_iter, self.ridge_strength, self.chunk_size)
                q = q + alpha[..., None] * (q_ - q)
            o, _ = _gla_chunk(q, k, v, g, self.gla_scale, self.chunk_size)
        else:                                                       # torch_decoding_one_step, token by token
            B, T, H, D = q.shape
            h_kk = torch.zeros(B, H, D, D, dtype=q.dtype)
            h_kv, o = None, []
            for t in range(T):
                qt = q[:, t]
                if self.num_iter > 0:
                    q_, h_kk = _cheb_step(h_kk, k[:, t], qt, gk[:, t] if gk is not None else None,
                                          self.num_iter, self.ridge_strength)
                    qt = qt + alpha[:, t, :, None] * (q_ - qt)
                ot, h_kv = _gla_recurrent(qt[:, None], k[:, t:t + 1], v[:, t:t + 1],
                                          g[:, t:t + 1] if g is not None else None, self.gla_scale, h_kv)
                o.append(ot)
            o = torch.cat(o, 1)

        if self.use_gate:
            gg = self.g_proj(hidden_states).reshape(*hidden_states.shape[:-1], -1, self.head_dim)
            o = self.o_norm(o, gg)
        else:
            o = self.o_norm(o)
        o = self.o_proj(o.reshape(*o.shape[:2], -1))
        return o, None, past_key_values


def construct_gka_config(config, gka_exp_config=None):
    """gka_config.py _construct_from_transformer (129-165)."""
    e = gka_exp_config or {}
    defaults = dict(use_alpha_connection=True, use_v_conv=True, use_forgetting_gate=True, gla_rescale=True,
                    solver_type="chebyshev", bp_lambda=True, num_iter=30, ridge_strength=0.02, use_gate=True,
                    conv_size=4, norm_eps=1e-6, use_forgetting_gate_kk=True, use_beta_gate=True, chunk_size=64,
                    kv_proj_rank=None, kv_learnable_residual=False)
    return SimpleNamespace(hidden_size=config.hidden_size,
                           head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                           num_q_heads=config.num_attention_heads, num_k_heads=config.num_key_value_heads,
                           num_v_heads=config.num_key_value_heads, **{n: e.get(n, d) for n, d in defaults.items()})


class GatedKalmaNetLayer(GatedKalmaNet):
    """modules.py 17-96."""

    def __init__(self, config, layer_idx):
        c = construct_gka_config(config, getattr(config, "gka_config", None))
        super().__init__(**vars(c), layer_idx=layer_idx)


# ---------------------------------------------------------------- HybridQwen3 (modeling_hybrid_qwen3.py)
class HybridQwen3Config(PretrainedConfig):
    model_type = "hybrid_qwen3"

    def __init__(self, vocab_size=151936, hidden_size=4096, intermediate_size=22016, num_hidden_layers=32,
                 num_attention_heads=32, num_key_value_heads=32, head_dim=128, hidden_act="silu",
                 max_position_embeddings=32768, initializer_range=0.02, rms_norm_eps=1e-6, use_cache=True,
                 tie_word_embeddings=False, rope_parameters=None, attention_bias=False, use_sliding_window=False,
                 sliding_window=4096, max_window_layers=28, layer_types=None, attention_dropout=0.0,
                 pad_token_id=None, bos_token_id=None, eos_token_id=None, hybrid_override_pattern=None, **kwargs):
        self.vocab_size, self.max_position_embeddings = vocab_size, max_position_embeddings
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.num_hidden_layers, self.num_attention_heads = num_hidden_layers, num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers, self.hybrid_override_pattern, self.layer_types = max_window_layers, hybrid_override_pattern, layer_types
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim, self.hidden_act, self.initializer_range = head_dim, hidden_act, initializer_range
        self.rms_norm_eps, self.use_cache = rms_norm_eps, use_cache
        self.attention_bias, self.attention_dropout = attention_bias, attention_dropout
        if rope_parameters is None and ("rope_theta" in kwargs or "rope_scaling" in kwargs):
            rope_parameters = {"rope_type": "default", "rope_theta": kwargs.pop("rope_theta", 10000.0)}
            rs = kwargs.pop("rope_scaling", None)
            if rs is not None:
                rope_parameters.update(rs)
        self.rope_parameters = rope_parameters
        if self.hybrid_override_pattern is not None:
            self.layer_types = self.hybrid_override_pattern.split("-")
        elif self.layer_types is None:
            self.layer_types = ["sliding_attention" if self.sliding_window is not None and i >= max_window_layers
                                else "full_attention" for i in range(num_hidden_layers)]
        assert len(self.layer_types) == self.num_hidden_layers
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        assert config.hidden_act == "silu"

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.rope_parameters["rope_type"] == "default"
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x, position_ids):
        freqs = (self.inv_freq.to(F64)[None, :, None] @ position_ids[:, None, :].to(F64)).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(h, n_rep):
    b, n, s, d = h.shape
    return h if n_rep == 1 else h[:, :, None].expand(b, n, n_rep, s, d).reshape(b, n * n_rep, s, d)


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        b = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=b)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        shp = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shp)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(shp)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shp).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        w = torch.matmul(q, k.transpose(2, 3)) * self.scaling + attention_mask
        o = torch.matmul(torch.softmax(w, dim=-1), v).transpose(1, 2).contiguous()
        return self.o_proj(o.reshape(*hidden_states.shape[:-1], -1))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h, attention_mask, position_embeddings, **kw):
        h = h + self.self_attn(self.input_layernorm(h), position_embeddings, attention_mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3GKADecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)
        self.gka = GatedKalmaNetLayer(config, layer_idx=layer_idx)

    def forward(self, h, attention_mask, position_embeddings, mode="recurrent"):
        h = h + self.gka(self.input_layernorm(h), mode=mode)[0]
        return h + self.mlp(self.post_attention_layernorm(h))


REGISTRY = {"*": Qwen3DecoderLayer, "GKA": Qwen3GKADecoderLayer}   # utils.set_decoder_layers_from_pattern


class HybridQwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        pat = config.hybrid_override_pattern.split("-")
        self.layers = nn.ModuleList(REGISTRY[pat[i]](config, i) for i in range(config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(self, input_ids, mode="recurrent"):
        h = self.embed_tokens(input_ids)
        T = h.shape[1]
        pos = torch.arange(T)[None]
        mask = torch.full((T, T), float("-inf"), dtype=h.dtype).triu(1)[None, None]
        pe = self.rotary_emb(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=mask, position_embeddings=pe, mode=mode)
        return self.norm(h)


class HybridQwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = HybridQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, mode="recurrent"):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids, mode=mode)))
