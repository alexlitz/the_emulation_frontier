"""REFERENCE transcription (text path only) of Agnes-AI/Agnes-3.0-Flash, read as text, never run.

Source repo: Agnes-AI/Agnes-3.0-Flash, snapshot 891ce4f9ffb89b22888aa7fcc2bb2f3618867684
  configuration_agnes.py  sha256 b504c8a6052933ef46f5e9df0d52a09a45b6df41f1f3a2168bfdb4718d7c2342
    LAYER_* constants (l. 23-25), AgnesTextConfig (l. 28-136)
  modeling_agnes.py       sha256 21eb5cf1c2f6686253000937c668030934bebb53cf1ce423ccd4c2eb47a074c5
    _RMSNormFn.forward / AgnesRMSNorm (l. 105-143), _GatedRMSNormFn.forward / AgnesGatedNorm (l. 146-194),
    AgnesMLP (l. 197-216, the parallel SwiGLU branch), _swap_halves / _apply_rope (l. 224-239),
    AgnesRotaryEmbedding (l. 242-298), _broadcast_kv / _dense_attention (l. 329-355),
    AgnesGlobalAttention (l. 358-411), _unit_normalize (l. 438-439), _delta_rule_chunked (l. 442-506),
    AgnesDeltaAttention (l. 542-661: the no-cache prefill path), AgnesDecoderLayer (l. 669-704),
    AgnesTextModel (l. 950-1022: no cache, no padding mask), AgnesForCausalLM (l. 1284-1331: logits only).

Deviations (all numerical-only or plumbing):
  * every `.float()` / `torch.float32` cast of the remote file (RMSNorm statistics, gated norm, rotary table,
    softmax, the delta-rule recurrence, the decay g) is lifted to fp64 (`_W`), so the reference is fp64-clean;
  * the fused fla / causal-conv1d kernels are absent (the remote falls back to the transcribed torch paths);
  * modules are plain nn.Module (no PreTrainedModel init / cache / generation plumbing, no vision tower);
    `create_causal_mask` (library) is replaced by its no-padding eager result, an additive causal mask, and the
    attention interface by the file's own `_dense_attention` (the eager path).
The config class is transcribed as written (library PreTrainedConfig + huggingface_hub strict dataclass).
Needs transformers 5.x (run it inside the TF5 subprocess, tests/test_universal_elephantine._tf5_run).
"""
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from huggingface_hub.dataclasses import strict
from transformers.activations import ACT2FN
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters

F64 = torch.float64
_W = lambda t: t.to(F64)                      # the remote's fp32 casts, lifted to fp64

LAYER_GLOBAL = "agnes_global_attention"
LAYER_DELTA = "agnes_delta_attention"
LAYER_TYPES = (LAYER_GLOBAL, LAYER_DELTA)


@strict
class AgnesTextConfig(PreTrainedConfig):
    model_type = "agnes_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 12288
    parallel_ffn_intermediate_size: int = 0
    vocab_size: int = 248320
    layer_types: list[str] | None = None
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    rope_parameters: RopeParameters | dict | None = None
    max_position_embeddings: int = 32768
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)
        if self.layer_types is None:
            period = kwargs.pop("global_attention_interval", kwargs.pop("full_attention_interval", 4))
            self.layer_types = [LAYER_GLOBAL if (i + 1) % period == 0 else LAYER_DELTA
                                for i in range(self.num_hidden_layers)]
        super().__post_init__(**kwargs)

    def validate_layer_type(self):
        plan = self.layer_types
        if plan is None:
            return
        unknown = sorted({t for t in plan if t not in LAYER_TYPES})
        if unknown:
            raise ValueError(f"`layer_types` entries must be in {LAYER_TYPES}, got {unknown}")
        if self.num_hidden_layers is not None and self.num_hidden_layers != len(plan):
            raise ValueError("`num_hidden_layers` must equal the number of `layer_types`")


class AgnesRMSNorm(nn.Module):
    """y = x / rms(x) * (1 + w) (l. 105-143)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = _W(x)
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * inv * (1.0 + _W(self.weight))).type_as(x)


class AgnesGatedNorm(nn.Module):
    """RMS norm, weight, then x silu(gate) (l. 146-194)."""

    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        in_dtype = hidden_states.dtype
        xf = _W(hidden_states)
        unit = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        scaled = self.weight * unit.to(in_dtype)
        return (scaled * F.silu(_W(gate))).to(in_dtype)


class AgnesMLP(nn.Module):
    """SwiGLU; with parallel_size > 0 a second, narrower SwiGLU on the same input is added (l. 197-216)."""

    def __init__(self, config, intermediate_size, parallel_size=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.parallel_ffn = AgnesMLP(config, parallel_size) if parallel_size > 0 else None

    def forward(self, x):
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if self.parallel_ffn is not None:
            y = y + self.parallel_ffn(x)
        return y


def _swap_halves(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    n_rot = cos.shape[-1]
    q_rot, q_rest = q[..., :n_rot], q[..., n_rot:]
    k_rot, k_rest = k[..., :n_rot], k[..., n_rot:]
    q_rot = (q_rot * cos) + (_swap_halves(q_rot) * sin)
    k_rot = (k_rot * cos) + (_swap_halves(k_rot) * sin)
    return torch.cat([q_rot, q_rest], dim=-1), torch.cat([k_rot, k_rest], dim=-1)


class AgnesRotaryEmbedding(nn.Module):
    """Three-axis rotary tables with interleaved sections (l. 242-298; default rope type only)."""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        assert self.rope_type == "default"
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(config=None, device=None, seq_len=None):
        theta = config.rope_parameters["rope_theta"]
        fraction = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        n_rot = int(head_dim * fraction)
        exponents = torch.arange(0, n_rot, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / n_rot
        return 1.0 / (theta ** exponents), 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        freq_col = _W(self.inv_freq[None, None, :, None]).expand(3, position_ids.shape[1], -1, 1)
        pos_row = _W(position_ids[:, :, None, :])
        angles = (freq_col @ pos_row).transpose(2, 3)
        angles = self._interleave_axes(angles, self.mrope_section)
        table = torch.cat((angles, angles), dim=-1)
        cos = table.cos() * self.attention_scaling
        sin = table.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def _interleave_axes(angles, section):
        merged = angles[0]
        for axis, offset in ((1, 1), (2, 2)):
            take = slice(offset, section[axis] * 3, 3)
            merged[..., take] = angles[axis, ..., take]
        return merged


def _broadcast_kv(x, groups):
    b, h, t, d = x.shape
    if groups == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, groups, t, d).reshape(b, h * groups, t, d)


def _dense_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    k = _broadcast_kv(key, module.num_key_value_groups)
    v = _broadcast_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = nn.functional.softmax(scores, dim=-1, dtype=F64).to(query.dtype)
    out = torch.matmul(probs, v).transpose(1, 2).contiguous()
    return out, probs


class AgnesGlobalAttention(nn.Module):
    """GQA softmax attention, (1 + w) q / k head norms, partial RoPE, sigmoid output gate (l. 358-411)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        q_width = config.num_attention_heads * self.head_dim
        kv_width = config.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=config.attention_bias)
        self.q_norm = AgnesRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = AgnesRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        lead = hidden_states.shape[:-1]
        per_head = (*lead, -1, self.head_dim)
        q, gate = torch.chunk(self.q_proj(hidden_states).view(*lead, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*lead, -1)
        q = self.q_norm(q.view(per_head)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(per_head)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(per_head).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rope(q, k, cos, sin)
        out, probs = _dense_attention(self, q, k, v, attention_mask, scaling=self.scaling)
        out = out.reshape(*lead, -1).contiguous()
        out = out * torch.sigmoid(gate)
        return self.o_proj(out), probs


def _unit_normalize(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _delta_rule_chunked(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=False, **kwargs):
    """Chunk-parallel gated delta rule (l. 442-506), fp32 lifted to fp64."""
    out_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _unit_normalize(query, dim=-1, eps=1e-6)
        key = _unit_normalize(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [_W(t.transpose(1, 2).contiguous()) for t in (query, key, value, beta, g)]
    bsz, heads, seq, dk = key.shape
    dv = value.shape[-1]
    pad = (chunk_size - seq % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    padded = seq + pad
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        t.reshape(t.shape[0], t.shape[1], -1, chunk_size, t.shape[-1]) for t in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    solve = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for i in range(1, chunk_size):
        row = solve[..., i, :i].clone()
        block = solve[..., :i, :i].clone()
        solve[..., i, :i] = row + (row.unsqueeze(-1) * block).sum(-2)
    solve = solve + torch.eye(chunk_size, dtype=solve.dtype, device=solve.device)
    value = solve @ v_beta
    k_decayed = solve @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(bsz, heads, dk, dv, dtype=value.dtype) if initial_state is None else initial_state.to(value)
    out = torch.zeros_like(value)
    for i in range(0, padded // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        local = q_i @ k_i.transpose(-1, -2) * decay[:, :, i]
        v_res = v_i - k_decayed[:, :, i] @ state
        carried = (q_i * g[:, :, i, :, None].exp()) @ state
        out[:, :, i] = carried + local @ v_res
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_res)
    if not output_final_state:
        state = None
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :seq]
    return out.transpose(1, 2).contiguous().to(out_dtype), state


class AgnesDeltaAttention(nn.Module):
    """Gated delta-rule attention: silu depthwise causal conv on qkv, l2-normed q / k, gated RMS norm (l. 542-661)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=self.conv_kernel_size,
                                groups=self.conv_dim, padding=self.conv_kernel_size - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = AgnesGatedNorm(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.causal_conv1d_fn = None
        self.chunk_gated_delta_rule = _delta_rule_chunked
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        assert cache_params is None and attention_mask is None
        bsz, seq, _ = hidden_states.shape
        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(bsz, seq, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        qkv = F.silu(self.conv1d(qkv)[:, :, : qkv.shape[-1]])
        qkv = qkv.transpose(1, 2)
        query, key, value = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(bsz, seq, -1, self.head_k_dim)
        key = key.reshape(bsz, seq, -1, self.head_k_dim)
        value = value.reshape(bsz, seq, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -_W(self.A_log).exp() * F.softplus(_W(a) + self.dt_bias)
        groups = self.num_v_heads // self.num_k_heads
        if groups > 1:
            query = query.repeat_interleave(groups, dim=2)
            key = key.repeat_interleave(groups, dim=2)
        mixed, _ = self.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None,
                                               output_final_state=False, use_qk_l2norm_in_kernel=True)
        mixed = self.norm(mixed.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(mixed.reshape(bsz, seq, -1))


class AgnesDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == LAYER_DELTA:
            self.delta_attn = AgnesDeltaAttention(config, layer_idx)
        elif self.layer_type == LAYER_GLOBAL:
            self.global_attn = AgnesGlobalAttention(config, layer_idx)
        self.mlp = AgnesMLP(config, config.intermediate_size, getattr(config, "parallel_ffn_intermediate_size", 0) or 0)
        self.input_layernorm = AgnesRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AgnesRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        skip = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.layer_type == LAYER_DELTA:
            h = self.delta_attn(hidden_states=h, cache_params=None, attention_mask=attention_mask)
        elif self.layer_type == LAYER_GLOBAL:
            h, _ = self.global_attn(hidden_states=h, attention_mask=attention_mask, position_embeddings=position_embeddings)
        h = skip + h
        skip = h
        h = self.mlp(self.post_attention_layernorm(h))
        return skip + h


class AgnesTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([AgnesDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = AgnesRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AgnesRotaryEmbedding(config=config)

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        T = inputs_embeds.shape[1]
        position_ids = torch.arange(T).view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        position_ids = position_ids[1:]                      # row 0 = text positions (mask only)
        # create_causal_mask, no padding, eager: additive causal mask
        global_mask = torch.full((T, T), torch.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype).triu(1)[None, None]
        delta_mask = None                                    # _recurrent_mask(None, None)
        h = inputs_embeds
        rope = self.rotary_emb(h, position_ids)
        for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            mask = delta_mask if self.config.layer_types[i] == LAYER_DELTA else global_mask
            h = layer(h, position_embeddings=rope, attention_mask=mask)
        return self.norm(h)


class AgnesForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AgnesTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, use_cache=False):
        assert not use_cache
        return self.lm_head(self.model(input_ids))
