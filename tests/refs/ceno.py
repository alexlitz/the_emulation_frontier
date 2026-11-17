"""CENO -- a labelled reference transcription of the remote modelling code (read as text, never executed).

Source: CladeTeam/CENO-1.3B-A450M-base, snapshot 18d80ea781f50b2918313d325fb56ca82d6d2b71 (refs/main)
  modeling_ceno.py       sha256 8b31ca257d03027bff242fbea7df521c3d8c141da726c72d2c502318d0bf0f62
  configuration_ceno.py  sha256 3a5bfb2bd7f96c954c0f6d974cc2843a2bd896dd675814473bac16890da0f594
(the same configuration file ships in every CENO repo snapshot; the older snapshot dabb3cab's modeling_ceno.py
differs only in kernel-side code, not in the torch forward transcribed here.)

Transcribed (modeling_ceno.py line ranges):
  34-36      squared_relu (relu2)
  169-217    pad_tensor_by_size, reshape_into_chunks, segment_sum
  366-394    MambaRMSNormGated (torch branch, rmsnorm_fn is None: gate before a per-group RMSNorm)
  404-475    CENOMamba2Mixer.__init__ / _maybe_apply_in_proj_norm
  701-895    CENOMamba2Mixer.torch_forward, the no-cache branch (811-878) and the gate (884-893)
  915-930    CENORMSNorm
  932-1064   CENOBlock (mamba / attention / mlp / moe residual structure)
  1067-1134  CENOMLP (ungated up / down; routing weight after the activation, before down_proj)
  1137-1229  CENOMoE (post-top-k softmax: top-k logits, then a softmax over them; pre-softmax optional)
  1232-1241  repeat_kv
  1244-1339  CENOAttention ("eager" = CENO_ATTENTION_CLASSES["eager"], 1524-1528: SDPA, is_causal, NoPE)
  1713-1724, 1745-1882  CENOModel (embeddings, blocks, norm_f; no mask -> SDPA is_causal)
  2033-2040, 2190-2191  CENOForCausalLM (lm_head)
configuration_ceno.py 155-303, 319-325: the config fields and layers_block_type.

Not transcribed: caches / generation, attention / padding masks, seq_idx segments, the CUDA kernel paths and the
bf16 training-precision mode (none is reached by an unpadded CPU forward without a cache).

fp64: the remote code casts to fp32 inside CENORMSNorm (.to(float32)), the Mamba scan (.float()), the router
logits (.to(float32)) and the logits (.float()). These casts are LIFTED here (computed in the input dtype), so the
reference is fp64-clean; everything else is the remote arithmetic in the remote order.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def squared_relu(x):
    return torch.pow(torch.relu(x), 2)


ACT = {"relu2": squared_relu, "squared_relu": squared_relu, "silu": F.silu, "relu": F.relu}


class CENOConfig:
    """configuration_ceno.py CENOConfig.__init__ (155-303): the defaults, and like PretrainedConfig the extra
    keyword arguments are set as attributes AFTER the named ones (so n_groups= / conv_kernel= / chunk_size=
    override mamba_n_groups / mamba_d_conv / mamba_chunk_size)."""

    def __init__(self, vocab_size=131072, tie_word_embeddings=False, hidden_size=4096, intermediate_size=21504,
                 num_hidden_layers=52, hybrid_override_pattern="MMMM*MMMMM*MMMMM*MMMMM*MMMMMM", num_attention_heads=32,
                 attention_head_dim=128, num_key_value_heads=8, mlp_hidden_act="relu2", mlp_use_swiglu=False,
                 num_experts=8, moe_top_k=2, moe_router_pre_softmax=False, attention_bias=False, qkv_layernorm=False,
                 mlp_bias=False, use_bias=False, layer_norm_epsilon=1e-5, residual_in_fp32=False,
                 ssm_state_size=128, mamba_num_heads=128, mamba_n_groups=8, mamba_head_dim=64, mamba_d_conv=4,
                 mamba_hidden_act="silu", mamba_dt_limit=(0.0, float("inf")), mamba_conv_bias=True,
                 mamba_in_proj_layernorm=False, mamba_chunk_size=256, ceno_mamba_gate_in_ssm=False, **kwargs):
        self.vocab_size, self.tie_word_embeddings, self.hidden_size = vocab_size, tie_word_embeddings, hidden_size
        self.intermediate_size, self.num_hidden_layers = intermediate_size, num_hidden_layers
        self.hybrid_override_pattern = hybrid_override_pattern
        assert len(hybrid_override_pattern) == num_hidden_layers
        self.num_attention_heads, self.attention_head_dim = num_attention_heads, attention_head_dim
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.mlp_hidden_act, self.mlp_use_swiglu = mlp_hidden_act, mlp_use_swiglu
        self.num_experts, self.moe_top_k, self.moe_router_pre_softmax = num_experts, moe_top_k, moe_router_pre_softmax
        self.attention_bias, self.qkv_layernorm, self.mlp_bias, self.use_bias = attention_bias, qkv_layernorm, mlp_bias, use_bias
        self.layer_norm_epsilon, self.residual_in_fp32 = layer_norm_epsilon, residual_in_fp32
        self.ssm_state_size, self.mamba_num_heads, self.n_groups = ssm_state_size, mamba_num_heads, mamba_n_groups
        self.mamba_head_dim, self.conv_kernel, self.mamba_hidden_act = mamba_head_dim, mamba_d_conv, mamba_hidden_act
        self.time_step_limit, self.use_conv_bias = mamba_dt_limit, mamba_conv_bias
        self.mamba_in_proj_layernorm, self.chunk_size = mamba_in_proj_layernorm, mamba_chunk_size
        self.ceno_mamba_gate_in_ssm = ceno_mamba_gate_in_ssm
        self.attention_dropout = 0.0
        for k, v in kwargs.items():                               # PretrainedConfig: extra kwargs -> attributes
            setattr(self, k, v)

    @property
    def layers_block_type(self):                                  # configuration_ceno.py 319-325
        p = self.hybrid_override_pattern
        return ["mamba" if c == "M" else "attention" if c == "*" else "mlp" if c == "-" else "moe" for c in p]


# --- 169-217 -----------------------------------------------------------------------------------------------------
def pad_tensor_by_size(input_tensor, pad_size):
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3])


def segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    return tensor_segsum.masked_fill(~mask, -torch.inf)


# --- 366-394, 915-930 ---------------------------------------------------------------------------------------------
class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, x, gate=None):
        if gate is not None:
            x = x * F.silu(gate)
        if self.group_size and x.shape[-1] % self.group_size == 0:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
            x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
            x = x.reshape(*x.shape[:-2], -1)
        else:
            x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return x * self.weight


class CENORMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, h):
        variance = h.pow(2).mean(-1, keepdim=True)
        return self.weight * (h * torch.rsqrt(variance + self.variance_epsilon))


# --- 404-475, 701-895 ---------------------------------------------------------------------------------------------
class CENOMamba2Mixer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.num_heads = config.mamba_num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.layer_idx = layer_idx
        self.act = ACT[config.mamba_hidden_act]
        self.in_proj_norm = (CENORMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
                             if config.mamba_in_proj_layernorm else None)
        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.chunk_size = config.chunk_size
        self.time_step_limit = config.time_step_limit
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=config.use_conv_bias, kernel_size=config.conv_kernel,
                                groups=self.conv_dim, padding=config.conv_kernel - 1)
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size + self.conv_dim + self.num_heads,
                                 bias=config.use_bias)
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32)))
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=config.layer_norm_epsilon,
                                      group_size=self.intermediate_size // self.n_groups)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def forward(self, input_states):
        batch_size, seq_len, _ = input_states.shape
        if self.in_proj_norm is not None:
            input_states = self.in_proj_norm(input_states)
        projected_states = self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size
                 - self.num_heads) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
        hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states, B, C = torch.split(hidden_states_B_C, [self.intermediate_size, self.n_groups * self.ssm_state_size,
                                                              self.n_groups * self.ssm_state_size], dim=-1)
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim)
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size)
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)
        hidden_states = hidden_states * dt[..., None]
        A = A.to(hidden_states.dtype) * dt
        hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)
        L = torch.exp(segment_sum(A))
        G = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
        M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)
        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
        new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        states = new_states[:, :-1]
        state_decay_out = torch.exp(A_cumsum)
        C_times_states = (C[..., None, :] * states[:, :, None, ...])
        Y_off = (C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None])
        y = (Y_diag + Y_off).reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)
        if self.config.ceno_mamba_gate_in_ssm:                    # 884-893: the same y * silu(z), then the norm
            scan_output = self.norm(y * F.silu(gate))
        else:
            scan_output = self.norm(y, gate)
        return self.out_proj(scan_output)


# --- 1067-1229 ----------------------------------------------------------------------------------------------------
class CENOMLP(nn.Module):
    def __init__(self, config, layer_idx=None, use_fc1_norm=True):
        super().__init__()
        self.fc1_norm = CENORMSNorm(config.hidden_size, eps=config.layer_norm_epsilon) if use_fc1_norm else None
        self.use_swiglu = config.mlp_use_swiglu
        assert not self.use_swiglu, "transcribed: the ungated MLP"
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT[config.mlp_hidden_act]

    def forward(self, x, per_token_scale=None):
        if self.fc1_norm is not None:
            x = self.fc1_norm(x)
        intermediate = self.act_fn(self.up_proj(x))
        if per_token_scale is not None:
            intermediate = intermediate * per_token_scale
        return self.down_proj(intermediate)


class CENOMoE(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.num_experts, self.top_k = config.num_experts, config.moe_top_k
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([CENOMLP(config, layer_idx, use_fc1_norm=False) for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(flat)
        if self.config.moe_router_pre_softmax:
            probs = F.softmax(router_logits, dim=-1)
            routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_logits, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
            routing_weights = F.softmax(topk_logits, dim=-1)
        final = torch.zeros_like(flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[e])
            if top_x.shape[0] == 0:
                continue
            cur = self.experts[e](flat[top_x], per_token_scale=routing_weights[top_x, idx].unsqueeze(-1))
            final.index_add_(0, top_x, cur)
        return final.view(batch_size, seq_len, hidden_dim)


# --- 1232-1339 ----------------------------------------------------------------------------------------------------
def repeat_kv(h, n_rep):
    b, nkv, s, hd = h.shape
    if n_rep == 1:
        return h
    return h[:, :, None].expand(b, nkv, n_rep, s, hd).reshape(b, nkv * n_rep, s, hd)


class CENOAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim or config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qkv_norm = CENORMSNorm(config.hidden_size, eps=config.layer_norm_epsilon) if config.qkv_layernorm else None
        b = config.attention_bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=b)

    def forward(self, h):
        bsz, q_len, _ = h.size()
        if self.qkv_norm is not None:
            h = self.qkv_norm(h)
        q = self.q_proj(h).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=q_len > 1)
        return self.o_proj(o.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim))


# --- 932-1064 -----------------------------------------------------------------------------------------------------
class CENOBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.block_type = config.layers_block_type[layer_idx]
        self.pre_mlp_norm = None
        eps = config.layer_norm_epsilon
        self.norm = None if (self.block_type == "mamba" and config.mamba_in_proj_layernorm) else CENORMSNorm(config.hidden_size, eps)
        if self.block_type == "mamba":
            self.mixer = CENOMamba2Mixer(config, layer_idx)
        elif self.block_type == "attention":
            self.mixer = CENOAttention(config, layer_idx)
            self.pre_mlp_norm = CENORMSNorm(config.hidden_size, eps)
        elif self.block_type == "mlp":
            self.mixer = CENOMLP(config, layer_idx, use_fc1_norm=True)
            self.pre_mlp_norm = CENORMSNorm(config.hidden_size, eps)
        else:
            self.mixer = CENOMoE(config, layer_idx)
            self.pre_mlp_norm = CENORMSNorm(config.hidden_size, eps)

    def forward(self, h):
        if self.block_type == "mamba":                            # x + mamba(norm(x))
            x = h if self.norm is None else self.norm(h)
            return h + self.mixer(x)
        if self.block_type == "attention":                        # pre_mlp_norm(x + attn(norm(x))): no 2nd residual
            return self.pre_mlp_norm(h + self.mixer(self.norm(h)))
        ln = self.norm(h)                                         # mlp / moe: ln(x) + ffn(pre_mlp_norm(ln(x)))
        return ln + self.mixer(self.pre_mlp_norm(ln))


# --- 1713-1882, 2033-2040, 2190-2191 ------------------------------------------------------------------------------
class CENOModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CENOBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm_f = CENORMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids):
        h = self.embeddings(input_ids)
        for block in self.layers:
            h = block(h)
        return self.norm_f(h)


class CENOForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = CENOModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.backbone(input_ids))
