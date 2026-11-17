"""REFERENCE transcription (not the remote code) of Phi-4-mini-flash (SambaY), plain torch, fp64.

Source repo:  microsoft/Phi-4-mini-flash-reasoning
Snapshot:     1dff8163d28ec880ca2411c474ddc0a927792810
Files (read as text, never imported or executed):
  configuration_phi4flash.py  sha256 8f0bf8b2eb5aca929354c79bc0f40a183a962a87b0cfb8de0abcffcc63ff4546
  modeling_phi4flash.py       sha256 ed32f40d3811ef30aa934783ce3040d8a0110ca0349f811f57504073470a7718

Transcribed (line ranges of those files):
  Phi4FlashConfig                     configuration_phi4flash.py l. 25-173 (fields, per-layer sliding_window list,
                                      layers_block_type)
  swiglu                              modeling l. 72-388 (jiterator kernel: x * y / (1 + exp(-x)) = silu(x) * y)
  SambaYRMSNorm                       modeling l. 392-406
  PHI_NORM_CLASS = nn.LayerNorm       modeling l. 409
  SambaYMLP                           modeling l. 425-454
  SambaYAttention.__init__            modeling l. 457-495
  SambaYFlashAttention2.forward       modeling l. 530-637 (no padding mask, no cache)
  _flash_attention_forward            modeling l. 639-743 (the attention_mask=None branches)
  Phi3Mamba.__init__ / forward        modeling l. 784-954 (the yoco_cross GMU branch and the slow path:
                                      causal_conv1d_fn is None, selective_scan_fn; no inference cache)
  SambaYDecoderLayer                  modeling l. 1012-1117
  Phi4FlashPreTrainedModel            modeling l. 1141-1160 (_init_weights)
  Phi4FlashModel                      modeling l. 1237-1393 (use_cache=False, no attention mask)
  Phi4FlashForCausalLM                modeling l. 1397-1515 (no labels)
  lambda_init_fn / split_heads        modeling l. 1969-1978
  FlashDiffCustomAttention            modeling l. 1980-2098 (fa_og=True, padded (non-varlen) path)

Library pieces the remote file imports, used here:
  - transformers PretrainedConfig / PreTrainedModel (weight tying, post_init), nn.LayerNorm, ACT2FN, einops.
  - mamba_ssm selective_scan_fn: transcribed from the upstream pure-torch selective_scan_ref
    (mamba_ssm-2.3.2.post1 mamba_ssm/ops/selective_scan_interface.py, sha256
    543edd077f11c80cccd5c5e2f926420d9105fd238d9e8960e53660cfa234dce8, l. 115-182), real A, variable B/C (b n l) only.
  - flash_attn.flash_attn_func (CUDA-only): written from its documented semantics -- softmax_scale defaults to
    head_dim^-1/2, GQA by head groups, causal mask aligned bottom-right, window_size=(left, right) restricts key j of
    query i to [i + sk - sq - left, i + sk - sq + right].

Departures (precision only, no change of the computation):
  - fp32 casts lifted to fp64: SambaYRMSNorm's .to(float32), A_log/D/dt_proj.bias .float(), the lambda .float(),
    the decoder layer's `residual.to(torch.float32)` after a Mamba mixer, and selective_scan_ref's .float() casts.
  - FlashDiffCustomAttention's q/k/v .to(bfloat16) casts and its fp16/bf16 + is_cuda asserts are dropped (flash_attn
    computes in bf16 on CUDA; the reference runs the same arithmetic in the input dtype).
  - The fused fast path (mamba_inner_fn, use_fast_path=True) is the same math through CUDA kernels; it is not
    transcribed: set use_fast_path=False (as the remote-code test did).
"""
import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel


# --- configuration_phi4flash.py l. 25-173 -------------------------------------------------------------------------
class Phi4FlashConfig(PretrainedConfig):
    model_type = "phi4flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
        num_attention_heads=40,
        num_key_value_heads=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=2047,
        mb_per_layer=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.mb_per_layer = mb_per_layer
        self.sliding_window = [
            sliding_window if layer_idx < num_hidden_layers // 2 and layer_idx % 2 == 1 else None
            for layer_idx in range(num_hidden_layers)
        ]
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def layers_block_type(self):
        layer_block_types = []
        for i in range(self.num_hidden_layers):
            if i % 2 == 1:
                layer_block_type = "attention" if i <= (self.num_hidden_layers // 2 + 1) else "shared_attention"
            else:
                layer_block_type = "mamba"
            layer_block_types.append(layer_block_type)
        return layer_block_types


# --- modeling l. 72-388: swiglu(x, y) = x * y / (1 + exp(-x)) ------------------------------------------------------
def swiglu(x, y):
    return x * y / (1.0 + torch.exp(-x))


# --- mamba_ssm selective_scan_ref (l. 115-182), fp32 casts lifted to fp64; real A, B/C of shape (b, n, l) ----------
def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    dtype_in = u.dtype
    u = u.double()
    delta = delta.double()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].double()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    assert not A.is_complex() and B.dim() == 3 and C.dim() == 3
    B = B.double()
    C = C.double()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# --- flash_attn.flash_attn_func, documented semantics (B, S, H, D) layout -----------------------------------------
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1)):
    assert dropout_p == 0.0
    sq, sk, H, Dq = q.shape[1], k.shape[1], q.shape[2], q.shape[3]
    rep = H // k.shape[2]
    k, v = k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)
    scale = softmax_scale if softmax_scale is not None else Dq ** -0.5
    s = torch.einsum("bihd,bjhd->bhij", q, k) * scale
    i = torch.arange(sq)[:, None] + (sk - sq)
    j = torch.arange(sk)[None]
    ok = torch.ones(sq, sk, dtype=torch.bool)
    if causal:
        ok &= j <= i
    if window_size[0] >= 0:
        ok &= j >= i - window_size[0]
    if window_size[1] >= 0:
        ok &= j <= i + window_size[1]
    s = s.masked_fill(~ok, float("-inf"))
    return torch.einsum("bhij,bjhd->bihd", s.softmax(-1), v)


# --- modeling l. 392-409 -----------------------------------------------------------------------------------------
class SambaYRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float64)                 # fp32 lifted
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


PHI_NORM_CLASS = nn.LayerNorm


# --- modeling l. 425-454 -----------------------------------------------------------------------------------------
class SambaYMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        y = self.fc1(hidden_states)
        if self.config.hidden_act == "silu" and swiglu is not None:
            gate, y = y.chunk(2, dim=-1)
            y = swiglu(gate, y)
        else:
            gate, y = y.chunk(2, dim=-1)
            y = y * self.activation_fn(gate)
        return self.fc2(y)


# --- modeling l. 1969-2098 ---------------------------------------------------------------------------------------
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def split_heads(x):
    x = rearrange(x, "... (H two) D -> ... H two D", two=2)
    x1 = x[..., 0, :]
    x2 = x[..., 1, :]
    return x1, x2


class FlashDiffCustomAttention(nn.Module):
    def __init__(self, head_dim, depth, fa_og=True):
        super().__init__()
        self.head_dim = head_dim
        self.fa_og = fa_og
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.subln = SambaYRMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(self, q, k, v, dropout_p=0.0, softmax_scale=None, window_size=(-1, -1), causal=None):
        # (bf16 casts and the fp16/bf16 + is_cuda asserts dropped; padded path only)
        q1, q2 = split_heads(q)
        k1, k2 = split_heads(k)
        if self.fa_og:
            v1, v2 = split_heads(v)
        else:
            v = rearrange(v, "... (H two) D -> ... H (two D)", two=2)
        kwargs = {"dropout_p": dropout_p, "softmax_scale": softmax_scale, "causal": causal, "window_size": window_size}
        attn_func = flash_attn_func
        if self.fa_og:
            attn11 = attn_func(q1, k1, v1, **kwargs)
            attn12 = attn_func(q1, k1, v2, **kwargs)
            attn1 = torch.cat([attn11, attn12], dim=-1)
            attn21 = attn_func(q2, k2, v1, **kwargs)
            attn22 = attn_func(q2, k2, v2, **kwargs)
            attn2 = torch.cat([attn21, attn22], dim=-1)
        else:
            attn1 = attn_func(q1, k1, v, **kwargs)
            attn2 = attn_func(q2, k2, v, **kwargs)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).double()).type_as(q)   # fp32 lifted
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).double()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = rearrange(attn, "... H (two D) -> ... (H two) D", two=2)
        return attn


# --- modeling l. 457-495 / 512-743 -------------------------------------------------------------------------------
class SambaYAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None, yoco_cross: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.yoco_cross = yoco_cross
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")
        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        if yoco_cross:
            self.Wqkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        else:
            self.Wqkv = nn.Linear(self.hidden_size, op_size, bias=True)
        self.inner_cross_attn = FlashDiffCustomAttention(self.head_dim, self.layer_idx)


class SambaYFlashAttention2(SambaYAttention):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, yoco_key_values=None, **kwargs):
        assert attention_mask is None and past_key_value is None
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()
        if self.yoco_cross:
            q = self.Wqkv(hidden_states)
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim).transpose(1, 2)
            key_states, value_states = yoco_key_values
            query_states = q
            use_sliding_windows = False
        else:
            qkv = self.Wqkv(hidden_states)
            query_pos = self.num_heads * self.head_dim
            query_states = qkv[..., :query_pos]
            key_states = qkv[..., query_pos: query_pos + self.num_key_value_heads * self.head_dim]
            value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim:]
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            use_sliding_windows = (self.config.sliding_window is not None
                                   and self.config.sliding_window[self.layer_idx] is not None)
        yoco_key_values = key_states, value_states
        attn_dropout = self.attention_dropout if self.training else 0.0
        if query_states.dtype == torch.float32:
            target_dtype = self.Wqkv.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len,
                                                    dropout=attn_dropout, use_sliding_windows=use_sliding_windows)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, None, yoco_key_values

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length,
                                 dropout=0.0, softmax_scale=None, use_sliding_windows=False):
        causal = self.is_causal
        assert attention_mask is None
        if not use_sliding_windows:
            attn_output = self.inner_cross_attn(query_states, key_states, value_states, dropout_p=dropout,
                                                softmax_scale=softmax_scale, causal=causal)
        else:
            w = self.config.sliding_window[self.layer_idx]
            attn_output = self.inner_cross_attn(query_states, key_states, value_states, dropout_p=dropout,
                                                softmax_scale=softmax_scale, causal=causal, window_size=(w - 1, w - 1))
        return attn_output


# --- modeling l. 784-954 -----------------------------------------------------------------------------------------
class Phi3Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", conv_bias=True, bias=False,
                 use_fast_path=True, layer_idx=None, yoco_cross=False, yoco_kv=False, dtype=None):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        if self.yoco_cross:
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
                                    kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
            self.activation = "silu"
            self.act = nn.SiLU()
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, mask=None, yoco_key_values=None, cache_position=None):
        if self.yoco_cross:
            out = self.in_proj(hidden_states)
            out = swiglu(out, yoco_key_values)
            out = self.out_proj(out)
            return out, yoco_key_values
        assert inference_params is None
        batch, seqlen, _ = hidden_states.shape
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states.to(dtype=self.in_proj.weight.dtype),
                                                       "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.double())                                      # fp32 lifted
        if (not self.yoco_kv) and self.use_fast_path:
            raise NotImplementedError("fused mamba_inner_fn path not transcribed; set use_fast_path=False")
        x, z = xz.chunk(2, dim=1)
        if self.yoco_kv:
            z = z.transpose(-1, -2).contiguous()
        if mask is not None:
            x = x * mask.unsqueeze(1)
        x = self.act(self.conv1d(x)[..., :seqlen])                               # causal_conv1d_fn is None
        if mask is not None:
            x = x * mask.unsqueeze(1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, dt, A, B, C, self.D.double(), z=None if self.yoco_kv else z,
                              delta_bias=self.dt_proj.bias.double(), delta_softplus=True, return_last_state=False)
        y = rearrange(y, "b d l -> b l d")
        if self.yoco_kv:
            yoco_key_values = y
            y = swiglu(z, y)
        out = self.out_proj(y)
        return out, yoco_key_values


# --- modeling l. 1012-1117 ---------------------------------------------------------------------------------------
class SambaYDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.mlp = SambaYMLP(config)
        self.input_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)
        self.yoco_kv = False
        self.yoco_cross = False
        self.yoco_mb = False
        self.layer_idx = layer_idx
        assert config.num_hidden_layers % 4 == 0, "n_layer should be divisible by 4 for SambaY "
        if layer_idx >= config.num_hidden_layers // 2:
            self.yoco_mb = True
            self.yoco_kv = (layer_idx >= (config.num_hidden_layers // 2 + 1))
            self.yoco_cross = (layer_idx >= (config.num_hidden_layers // 2 + 2))
            if (layer_idx >= (config.num_hidden_layers // 2 + 1)):
                config = copy.deepcopy(config)
                config.sliding_window = None
        self.config = config
        self.use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        if self.use_mamba:
            factory_kwargs = {"d_conv": config.mamba_d_conv, "d_state": config.mamba_d_state,
                              "expand": config.mamba_expand, "dtype": None}
            self.attn = Phi3Mamba(config.hidden_size, layer_idx=layer_idx, yoco_cross=self.yoco_cross,
                                  yoco_kv=self.yoco_mb, **factory_kwargs)
        else:
            self.attn = SambaYFlashAttention2(config, layer_idx=layer_idx, yoco_cross=self.yoco_cross)
        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, ssm_output=None,
                yoco_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states.to(dtype=self.input_layernorm.weight.dtype))
        if self.use_mamba:
            attn_outputs, ssm_output = self.attn(hidden_states, inference_params=past_key_value,
                                                 mask=attention_mask, yoco_key_values=ssm_output,
                                                 cache_position=cache_position)
            residual = residual.to(torch.float64)                                # fp32 lifted
        else:
            attn_outputs, _, yoco_key_values = self.attn(
                hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, yoco_key_values=yoco_key_values)
        hidden_states = residual + self.resid_attn_dropout(attn_outputs)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        return (hidden_states, ssm_output, yoco_key_values)


# --- modeling l. 1141-1160 ---------------------------------------------------------------------------------------
class Phi4FlashPreTrainedModel(PreTrainedModel):
    config_class = Phi4FlashConfig
    base_model_prefix = "model"
    _no_split_modules = ["SambaYDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# --- modeling l. 1237-1393 ---------------------------------------------------------------------------------------
class Phi4FlashModel(Phi4FlashPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList([SambaYDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.final_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, use_cache=None, **kwargs):
        assert attention_mask is None and not use_cache, "reference: no padding mask, no cache"
        hidden_states = self.embed_tokens(input_ids)
        ssm_output = None
        yoco_key_values = None
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, ssm_output=ssm_output,
                                          yoco_key_values=yoco_key_values)
            hidden_states = layer_outputs[0]
            ssm_output = layer_outputs[1]
            yoco_key_values = layer_outputs[2]
        hidden_states = self.final_layernorm(hidden_states.to(dtype=self.final_layernorm.weight.dtype))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


# --- modeling l. 1397-1515 ---------------------------------------------------------------------------------------
class Phi4FlashForCausalLM(Phi4FlashPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Phi4FlashModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, use_cache=None, num_logits_to_keep=0, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        return CausalLMOutputWithPast(logits=logits)
