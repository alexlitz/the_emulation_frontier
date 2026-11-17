"""REFERENCE transcription (plain torch), not the remote code: jet-ai/Jet-Nemotron-2B.

Read as text from ~/.cache/huggingface/hub/models--jet-ai--Jet-Nemotron-2B/snapshots/
afd4ee28fbb9db42cfd66b0db32602aa031767a3/ and never executed:

  configuration_jet_nemotron.py  sha256 895d3c6e0d749aedfbf6b6dcbc1b2fee35e78d05541181867982e18a85f316c8
      JetNemotronConfig.__init__            l. 128-185
  dynamic_conv.py                sha256 11dacac0d05d78fe3142dbcc57b09c1e78935fc97cdd66f2491fa12ac926377e
      DynamicShortConvolution.__init__      l. 39-74   (_init_kernel_generator l. 76-88)
      get_kernel                            l. 90-98
      forward (no-cache path)               l. 135-184 (implementation "naive" only)
      _forward_naive                        l. 186-194
  jet_block.py                   sha256 2f10d2e7fb14dccb2276279dd187cdb89a2b15131a714d8224f52d25c339e504
      JetBlockConfig                        l. 41-50
      init_linear_conv1d                    l. 53-57
      JetBlock.__init__ / forward           l. 61-148 / l. 150-255 (no cache)
  modeling_jet_nemotron.py       sha256 d2dd2e4194366e767b37503f71616e946b5d1270d1f0018a0fc22b48de99370d
      JetNemotronMLP                        l. 67-78
      rotate_half / apply_rotary_pos_emb    l. 81-112
      JetNemotronAttention                  l. 153-313 (sdpa path, no cache)
      JetNemotronRMSNorm                    l. 316-333
      JetNemotronDecoderLayer               l. 341-406
      JetNemotronRotaryEmbedding            l. 409-440
      JetNemotronPreTrainedModel._init_weights  l. 478-487
      JetNemotronModel                      l. 569-585, forward l. 595-696 (no cache, no gradient checkpointing),
                                            _update_causal_mask l. 698-760, _prepare_4d_... l. 762-820
      JetNemotronForCausalLM                l. 830-837, forward l. 907-941 (no loss)

The fla pieces jet_block.py imports are transcribed from library sources (fla is library code):
  fla.layers.utils get_unpad_data / index_first_axis / pad_input: flash_linear_attention-0.4.0
      fla/layers/utils.py (sha256 57445d45bf193d414b89d7106c6bef42ace0b373c61383335099023614937b07) l. 75-98, 176-197
  fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule (jet_block takes this branch: q_len <= 64): fla_core-0.5.2
      fla/ops/gated_delta_rule/naive.py (sha256 d1cf1799...) naive_recurrent_gated_delta_rule l. 13-64, with the
      kernel's in-kernel l2norm x / sqrt(sum x^2 + 1e-6) and varlen bos/eos from fused_recurrent.py
      (sha256 9c15dc68...) l. 70-75, 116-118
  fla.modules.FusedRMSNormGated: fla_core-0.5.2 fla/modules/fused_norm_gate.py (sha256 d8654c43...)
      class l. 1001-1062, forward kernel l. 66-104: y = x rsqrt(mean x^2 + eps) w * g sigmoid(g)

Stock transformers / library pieces used where the remote file itself imports them: PretrainedConfig,
rope_config_validation, ROPE_INIT_FUNCTIONS, dynamic_rope_update, ACT2FN, AttentionMaskConverter,
ALL_ATTENTION_FUNCTIONS (sdpa_attention_forward), BaseModelOutputWithPast / CausalLMOutputWithPast.

fp32 casts lifted to fp64: JetNemotronRMSNorm (l. 327), the rotary embedding (l. 430-435), the JetBlock decay g
(l. 209: A_log.float(), a_proj(...).float()), the fla gated-delta recurrence and gated RMSNorm (fp32 in-kernel).
Not transcribed: the KV / conv-state cache (use_cache=False only), the Triton dconv implementations, chunk mode
(only reached for q_len > 64), flash_attention_2, training-mode dropout.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update, rope_config_validation
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

F64 = torch.float64


# ---- configuration_jet_nemotron.py l. 128-185 ----------------------------------------------------------------
class JetNemotronConfig(PretrainedConfig):
    model_type = "jet_nemotron"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
                 num_attention_heads=12, num_key_value_heads=2, hidden_act="silu", max_position_embeddings=32768,
                 initializer_range=0.02, rms_norm_eps=1e-6, use_cache=True, tie_word_embeddings=True,
                 rope_theta=1000000.0, rope_scaling=None, attention_dropout=0.0, layer_types=None,
                 efficient_attention_config=None, **kwargs):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        if layer_types is None:
            self.layer_types = ["attn"] * num_hidden_layers
        elif isinstance(layer_types, str):
            raise ValueError("string layer_types (eval'd by the remote config) not supported by the reference")
        else:
            self.layer_types = layer_types
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        self.efficient_attention_config = efficient_attention_config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# ---- fla.layers.utils (library) ------------------------------------------------------------------------------
def get_unpad_data(attention_mask):
    lens = attention_mask.sum(-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(lens, 0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, int(lens.max())


def index_first_axis(x, indices):
    return x[indices]


def pad_input(hidden_states, indices, batch_size, seq_len):
    out = hidden_states.new_zeros(batch_size * seq_len, *hidden_states.shape[1:])
    out[indices] = hidden_states
    return rearrange(out, "(b s) ... -> b s ...", b=batch_size)


# ---- fla gated delta rule, fused-recurrent semantics (library), fp64 ----------------------------------------
def fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                                     cu_seqlens=None, use_qk_l2norm_in_kernel=False, **kw):
    """q, k: [B, T, H, K]; v: [B, T, H, V]; g, beta: [B, T, H]. With cu_seqlens, B == 1 and the state restarts
    at every sequence boundary (the kernel's bos / eos)."""
    assert initial_state is None and not output_final_state
    B, T, H, K = k.shape
    if scale is None:
        scale = K ** -0.5
    cuts = cu_seqlens.tolist() if cu_seqlens is not None else None
    o = torch.zeros(*v.shape, dtype=F64)
    for b in range(B):
        segs = list(zip(cuts[:-1], cuts[1:])) if cuts is not None else [(0, T)]
        for bos, eos in segs:
            h = torch.zeros(H, K, v.shape[-1], dtype=F64)
            for t in range(bos, eos):
                b_q, b_k, b_v = q[b, t].to(F64), k[b, t].to(F64), v[b, t].to(F64)
                if use_qk_l2norm_in_kernel:
                    b_q = b_q / torch.sqrt((b_q * b_q).sum(-1, keepdim=True) + 1e-6)
                    b_k = b_k / torch.sqrt((b_k * b_k).sum(-1, keepdim=True) + 1e-6)
                b_q = b_q * scale
                h = h * g[b, t].to(F64).exp()[:, None, None]
                b_v = b_v - (h * b_k[..., None]).sum(-2)
                b_v = b_v * beta[b, t].to(F64)[:, None]
                h = h + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
                o[b, t] = torch.einsum("hd,hdm->hm", b_q, h)
    return o.to(v.dtype), None


# ---- fla.modules.FusedRMSNormGated (library), fp64 -----------------------------------------------------------
class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **kw):
        super().__init__()
        self.hidden_size, self.eps, self.activation = hidden_size, eps, activation
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward(self, x, g):
        dt = x.dtype
        x, g = x.to(F64), g.to(F64)
        y = x * (1 / torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps)) * self.weight.to(F64)
        y = y * g * torch.sigmoid(g) if self.activation in ("swish", "silu") else y * torch.sigmoid(g)
        return y.to(dt)


# ---- dynamic_conv.py -----------------------------------------------------------------------------------------
class DynamicShortConvolution(nn.Module):
    def __init__(self, hidden_size, kernel_size, generator_input_size=None, generator_reduction=None,
                 generator_activation="silu", activation="silu", static_conv_init=None, use_fast_conv1d=True,
                 implementation="naive"):
        super().__init__()
        self.hidden_size = hidden_size
        self.generator_input_size = hidden_size if generator_input_size is None else generator_input_size
        self.generator_hidden_size = hidden_size if generator_reduction is None else (hidden_size // generator_reduction)
        self.kernel_size = kernel_size
        self.activation = None
        self.use_fast_conv1d = use_fast_conv1d
        self.implementation = implementation
        if activation is not None:
            assert activation in ["silu", "swish"]
            self.activation = activation
        self.static_conv_init = static_conv_init
        self.kernel_generator = nn.Sequential(OrderedDict([
            ("w1", nn.Linear(self.generator_input_size, self.generator_hidden_size, bias=False)),
            ("act", ACT2FN[generator_activation]),
            ("w2", nn.Linear(self.generator_hidden_size, self.hidden_size * self.kernel_size, bias=True)),
        ]))
        for layer in self.kernel_generator:
            if isinstance(layer, nn.Linear):
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()
        if self.static_conv_init is not None:
            self.static_conv_init(self.kernel_generator.w2.bias)

    def get_kernel(self, x):
        flat_kernels = self.kernel_generator(x)
        if flat_kernels.dim() == 3:
            return rearrange(flat_kernels, "b t (d w) -> b t d w", w=self.kernel_size)
        if flat_kernels.dim() == 2:
            return rearrange(flat_kernels, "b (d w) -> b d w", w=self.kernel_size)
        raise ValueError(f"Invalid kernel shape: {flat_kernels.shape}")

    def forward(self, x, mask=None, cache=None, output_final_state=False, cu_seqlens=None, generator_input=None, **kw):
        assert cu_seqlens is None and cache is None and not output_final_state
        assert self.implementation in ("naive", "triton_decoding")
        input_dtype = x.dtype
        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))
        x = self._forward_naive(x, generator_input=generator_input)
        if self.activation is not None:
            x = ACT2FN[self.activation](x)
        return x.to(input_dtype), cache

    def _forward_naive(self, x, generator_input=None):
        W = self.kernel_size
        generator_input = x if generator_input is None else generator_input
        kernels = self.get_kernel(generator_input)
        x = F.pad(x.transpose(1, 2), (W - 1, 0))
        x = x.unfold(dimension=2, size=W, step=1)
        x = x.permute(0, 2, 1, 3)
        return (x * kernels).sum(dim=-1)


# ---- jet_block.py --------------------------------------------------------------------------------------------
@dataclass
class JetBlockConfig:
    mode: str = "chunk"
    expand_v: int = 2.0
    num_heads: int = 6
    head_dim: int = 256
    norm_eps: float = 1e-5
    conv_size: int = 4
    dconv_generator_reduction: int = 8
    dconv_implementation: str = "triton"


def init_linear_conv1d(weight, std, bias=None):
    weight.data.normal_(mean=0.0, std=std)
    if bias is not None and not getattr(bias, "_no_reinit", False):
        nn.init.zeros_(bias)


class JetBlock(nn.Module):
    def __init__(self, config=None, layer_type="jet", layer_idx=None, hidden_size=None, initializer_range=None,
                 jet_block_config=None):
        super().__init__()
        if jet_block_config is None:
            jet_block_config = JetBlockConfig(**config.efficient_attention_config[layer_type])
        hidden_size = hidden_size or config.hidden_size
        initializer_range = initializer_range or config.initializer_range
        self.mode = jet_block_config.mode
        self.hidden_size = hidden_size
        self.expand_v = jet_block_config.expand_v
        self.conv_size = jet_block_config.conv_size
        self.head_dim = jet_block_config.head_dim
        self.num_heads = jet_block_config.num_heads
        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_k_dim = jet_block_config.head_dim
        self.head_v_dim = int(jet_block_config.head_dim * self.expand_v)
        self.layer_idx = layer_idx
        assert math.isclose(self.key_dim * self.expand_v, self.value_dim, rel_tol=1e-5)
        assert math.isclose(self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5)
        assert self.mode in ["chunk", "fused_recurrent"]
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dynamic_conv1d = DynamicShortConvolution(
            hidden_size=self.value_dim, kernel_size=self.conv_size, generator_input_size=self.hidden_size,
            generator_reduction=jet_block_config.dconv_generator_reduction,
            static_conv_init=lambda x: init_linear_conv1d(x, std=initializer_range),
            implementation=jet_block_config.dconv_implementation)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=float(jet_block_config.norm_eps))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, use_cache=False, **kwargs):
        assert past_key_value is None and not use_cache
        if attention_mask is not None:
            if len(attention_mask.shape) > 2:
                attention_mask = attention_mask.squeeze(1)
                attention_mask = torch.where(attention_mask[:, -1] > -1, 1, 0)
            assert len(attention_mask.shape) == 2
        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if q_len <= 64 else self.mode
        if mode != "fused_recurrent":
            raise NotImplementedError("chunk mode (q_len > 64) is not transcribed")
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None and q_len > 1:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
        conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))
        v, _ = self.dynamic_conv1d(x=self.v_proj(hidden_states), generator_input=hidden_states, mask=conv_mask,
                                   cache=None, output_final_state=False)
        if attention_mask is not None and q_len > 1:
            q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            k = index_first_axis(rearrange(k, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            v = index_first_axis(rearrange(v, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        beta = self.b_proj(hidden_states).sigmoid()
        g = -self.A_log.to(F64).exp() * F.softplus(self.a_proj(hidden_states).to(F64) + self.dt_bias)  # .float() -> fp64
        o, _ = fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
                                                output_final_state=False, cu_seqlens=cu_seqlens,
                                                use_qk_l2norm_in_kernel=True)
        g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        o = self.o_norm(o, g)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None and q_len > 1:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, past_key_value


# ---- modeling_jet_nemotron.py --------------------------------------------------------------------------------
class JetNemotronMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size, self.intermediate_size = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class JetNemotronAttention(nn.Module):
    def __init__(self, config, layer_idx=None, sliding_window=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def _update_causal_mask_for_sliding_window(self, attention_mask, input_tensor, past_key_values):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        past_seen_tokens = 0                                           # no cache
        target_length = sequence_length + min(past_seen_tokens, self.sliding_window - 1)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + sequence_length, device=device)
        if attention_mask is not None:
            assert attention_mask.dim() == 4
            diagonal_attend_mask = attention_mask < -1
            diagonal_attend_mask = diagonal_attend_mask[:, :, :, -target_length:]
        else:
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            diagonal_attend_mask = diagonal_attend_mask[None, None, :, :]
        if past_key_values is None or target_length > self.sliding_window:
            sliding_attend_mask = torch.arange(past_seen_tokens + sequence_length, device=device)[-target_length:] <= (
                cache_position.reshape(-1, 1) - self.sliding_window)
            diagonal_attend_mask = diagonal_attend_mask | sliding_attend_mask[None, None, :, :]
        causal_mask = torch.full((1, 1, sequence_length, target_length), fill_value=min_dtype, dtype=dtype,
                                 device=device)
        causal_mask = causal_mask * diagonal_attend_mask
        return AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None,
                **kwargs):
        assert past_key_value is None
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self.sliding_window is not None and self.config._attn_implementation != "flash_attention_2":
            attention_mask = self._update_causal_mask_for_sliding_window(attention_mask, hidden_states, past_key_value)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        fa2_sliding_window = self.sliding_window - 1 if self.sliding_window is not None else None
        assert self.config._attn_implementation == "sdpa"
        if self.sliding_window is None and AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=hidden_states, past_key_values_length=0,
                sliding_window=self.sliding_window, is_training=self.training):
            attention_mask = None
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling,
            sliding_window=fa2_sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class JetNemotronRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(F64)                            # .to(torch.float32) -> fp64
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


EFFICIENT_ATTENTION_CLASSES = {"jet": JetBlock}


class JetNemotronDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        lt = config.layer_types[layer_idx]
        if lt == "attn":
            self.self_attn = JetNemotronAttention(config, layer_idx)
        elif lt == "swa":
            self.self_attn = JetNemotronAttention(config, layer_idx,
                                                  sliding_window=config.efficient_attention_config["swa"]["window_size"])
        else:
            self.self_attn = EFFICIENT_ATTENTION_CLASSES[lt](config, lt, layer_idx)
        self.mlp = JetNemotronMLP(config)
        self.input_layernorm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states,)


class JetNemotronRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if getattr(config, "rope_scaling", None) is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):                                # .float() -> fp64
        inv_freq_expanded = self.inv_freq[None, :, None].to(F64).expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].to(F64)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _init_weights(model, std):
    """JetNemotronPreTrainedModel._init_weights (l. 478-487), applied by post_init to every module."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class JetNemotronModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([JetNemotronDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation in ["sdpa", "flash_attention_2"]
        self.norm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = JetNemotronRotaryEmbedding(config=config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kw):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, False)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                                          past_key_value=None, output_attentions=False, use_cache=False,
                                          cache_position=cache_position, position_embeddings=position_embeddings)[0]
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states))

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        assert self.config._attn_implementation == "sdpa"
        past_seen_tokens = 0
        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor)
                         else past_seen_tokens + sequence_length + 1)
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask, sequence_length=sequence_length, target_length=target_length, dtype=dtype,
            cache_position=cache_position, batch_size=input_tensor.shape[0])
        if attention_mask is not None and attention_mask.device.type == "cuda" and not output_attentions:
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(attention_mask, sequence_length, target_length, dtype,
                                                              cache_position, batch_size, **kw):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype,
                                 device=cache_position.device)
        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask


class JetNemotronForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = JetNemotronModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        _init_weights(self, config.initializer_range)                   # post_init: init, then tie
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, use_cache=None,
                logits_to_keep=0, **kw):
        assert not use_cache
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                   inputs_embeds=inputs_embeds).last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return CausalLMOutputWithPast(logits=self.lm_head(hidden_states[:, slice_indices, :]))
