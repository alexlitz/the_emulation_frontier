"""REFERENCE transcription (read as text, never run) of AliceAI: KDA + gated full attention, a sigmoid top-k
router with bias-corrected selection, a gated shared expert, and Block AttnRes.

Source  AMAImedia/AliceAI-Foundation-80B-A3B-NOESIS-BF16-GGUF. Two snapshots carry the code, byte-identical:
        cfe24f240e7bb13dc13d4bb4368d98febe2930c5 (with config.json; the one transcribed) and
        b5026459f58d37c59b137a44b811643b25f860cc.
        configuration_alice_ai.py  sha256 a45122b0d47765b08b710d1eb77305fd239148fcc5def7384d46b76876327f72
        modeling_alice_ai.py       sha256 20173b9430993d123969a12c0a003e23065cb83ea23c7ecc998d9532b6730180

Transcribed (line ranges in those files):
  configuration_alice_ai.py  l. 4-109   AliceAIConfig (a plain class here: PretrainedConfig's own fields are not read)
  modeling_alice_ai.py       l. 25-37   _rms_norm
                             l. 40-54   AliceAIRMSNorm
                             l. 57-67   _depth_softmax_mix
                             l. 70-83   AliceAIFinalBlockAttnRes
                             l. 86-105  AliceAIRotaryEmbedding
                             l. 108-121 _rotate_half, _apply_rotary
                             l. 124-148 eager_attention_forward (no dropout: eval)
                             l. 151-272 AliceAIAttention (no cache; _backend_mask l. 182-217 reduced to the
                                        branch taken with no padding mask: the causal bool mask, l. 211-214;
                                        the sdpa backend, which returns None there and runs is_causal=True,
                                        computes the same causal softmax attention)
                             l. 275-469 AliceAIKDA (no cache; _kda's CPU path l. 428-429 -> _torch_kda l. 338-384)
                             l. 472-485 AliceAIMLP
                             l. 488-527 AliceAIExperts
                             l. 530-555 AliceAISigmoidTopKRouter
                             l. 558-578 AliceAISparseMoEBlock
                             l. 581-648 AliceAIDecoderLayer
                             l. 669-704 AliceAIPreTrainedModel._init_weights (as init_weights_)
                             l. 707-802 AliceAIModel (no cache, no attention mask, no router-logit outputs)
                             l. 805-871 AliceAIForCausalLM.forward (logits only)

Library pieces: the remote file imports no model classes from transformers, only generic utilities (ACT2FN, Cache,
PreTrainedModel, output dataclasses). Here ACT2FN["silu"] is F.silu, the modules are plain nn.Modules, and the
outputs are SimpleNamespace(logits=...). Module tree, parameter / buffer names and the config attributes are the
remote ones, so elephantine.universal.from_alice loads this object as it loads the remote one.

Precision: every `.float()` / float32 cast of the remote code (rms norm, depth-softmax scores, rotary tables,
attention scores, KDA recurrence, router logits) is lifted to the input dtype (fp64 in the tests); that is the only
change.
"""
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch import nn

ACT2FN = {"silu": functional.silu}


# --- configuration_alice_ai.py l. 4-109 ---------------------------------------------------------------
class AliceAIConfig:
    model_type = "alice_ai"

    def __init__(self, vocab_size=129024, hidden_size=2048, num_hidden_layers=48, num_attention_heads=16,
                 num_key_value_heads=2, head_dim=256, linear_num_key_heads=32, linear_num_value_heads=32,
                 linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4, num_experts=512,
                 num_experts_per_tok=10, moe_intermediate_size=512, shared_expert_intermediate_size=512,
                 block_attn_res_block_size=4, router_score_function="sigmoid", router_bias_correction=True,
                 kda_allow_negative_eigenvalues=False, max_position_embeddings=262144, rope_theta=1_000_000.0,
                 partial_rotary_factor=0.25, rms_norm_eps=1e-6, hidden_act="silu", initializer_range=0.02,
                 attention_dropout=0.0, use_cache=True, output_router_logits=False, layer_types=None,
                 tie_word_embeddings=False, pad_token_id=None, bos_token_id=None, eos_token_id=None, **kwargs):
        if layer_types is None:
            layer_types = ["full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
                           for layer_idx in range(num_hidden_layers)]
        self.pad_token_id, self.bos_token_id, self.eos_token_id = pad_token_id, bos_token_id, eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.block_attn_res_block_size = block_attn_res_block_size
        self.router_score_function = router_score_function
        self.router_bias_correction = router_bias_correction
        self.kda_allow_negative_eigenvalues = kda_allow_negative_eigenvalues
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.output_router_logits = output_router_logits
        self.layer_types = layer_types
        self.number_of_conv_states = 3
        self._validate_fields()

    def _validate_fields(self):
        if self.block_attn_res_block_size <= 0:
            raise ValueError("block_attn_res_block_size must be positive")
        if self.linear_conv_kernel_dim < 2:
            raise ValueError("linear_conv_kernel_dim must be at least 2")
        if self.router_score_function != "sigmoid":
            raise ValueError("This architecture requires sigmoid routing")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")
        unknown = set(self.layer_types) - {"linear_attention", "full_attention"}
        if unknown:
            raise ValueError(f"Unsupported layer types: {sorted(unknown)}")


# --- modeling_alice_ai.py l. 25-121: norms, depth-softmax mix, rotary ---------------------------------
def _rms_norm(hidden_states, weight, eps, *, zero_centered=False):
    dtype = hidden_states.dtype                                         # remote: .float() (lifted)
    normalized = hidden_states * torch.rsqrt(hidden_states.square().mean(dim=-1, keepdim=True) + eps)
    scale = 1.0 + weight.to(dtype) if zero_centered else weight.to(dtype)
    return (normalized * scale).to(dtype)


class AliceAIRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps, *, zero_centered):
        super().__init__()
        initial_value = 0.0 if zero_centered else 1.0
        self.weight = nn.Parameter(torch.full((hidden_size,), initial_value))
        self.eps = eps
        self.zero_centered = zero_centered

    def forward(self, hidden_states):
        return _rms_norm(hidden_states, self.weight, self.eps, zero_centered=self.zero_centered)


def _depth_softmax_mix(sources, query_weight, norm_weight, eps):
    stacked = torch.stack(sources, dim=0)
    keys = _rms_norm(stacked, norm_weight, eps)
    scores = functional.linear(keys, query_weight.to(keys.dtype)).squeeze(-1)      # remote: .float() (lifted)
    weights = scores.softmax(dim=0).to(stacked.dtype).unsqueeze(-1)
    return (stacked * weights).sum(dim=0)


class AliceAIFinalBlockAttnRes(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.rms_norm_eps

    def forward(self, completed, partial=None):
        sources = completed if partial is None else [*completed, partial]
        return _depth_softmax_mix(sources, self.res_proj.weight, self.res_norm_weight, self.eps)


class AliceAIRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        if rotary_dim % 2:
            raise ValueError("The rotary dimension must be even")
        inv_freq = 1.0 / (config.rope_theta
                          ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))  # remote: float32
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        frequencies = torch.einsum("bi,j->bij", position_ids.to(dtype), self.inv_freq.to(dtype))  # remote: .float()
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def _rotate_half(hidden_states):
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(hidden_states, cos, sin):
    rotary_dim = cos.shape[-1]
    rotary, remainder = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary = rotary * cos + _rotate_half(rotary) * sin
    return torch.cat((rotary, remainder), dim=-1)


# --- l. 124-272: gated full attention -----------------------------------------------------------------
def eager_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **_):
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    scores = torch.matmul(query, key.transpose(-1, -2))                 # remote: .float() (lifted)
    scores = scores * (module.scaling if scaling is None else scaling)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attention_mask
    probabilities = scores.softmax(dim=-1).to(query.dtype)
    output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    return output, probabilities


class AliceAIAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, 2 * self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = AliceAIRMSNorm(self.head_dim, config.rms_norm_eps, zero_centered=True)
        self.k_norm = AliceAIRMSNorm(self.head_dim, config.rms_norm_eps, zero_centered=True)

    def forward(self, hidden_states, position_embeddings, cache_position):
        batch_size, sequence_length, _ = hidden_states.shape
        query, output_gate = (self.q_proj(hidden_states)
                              .view(batch_size, sequence_length, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1))
        output_gate = output_gate.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeddings
        query = _apply_rotary(query, cos, sin)
        key = _apply_rotary(key, cos, sin)
        key_length = key.shape[-2]                                      # _backend_mask, l. 211-214
        key_positions = torch.arange(key_length, device=cache_position.device)
        backend_mask = key_positions.view(1, 1, 1, -1) <= cache_position.view(1, 1, -1, 1)
        output, attention_weights = eager_attention_forward(self, query, key, value, backend_mask,
                                                            dropout=0.0, scaling=self.scaling, is_causal=True)
        output = output.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        output = output * torch.sigmoid(output_gate)
        return self.o_proj(output), attention_weights


# --- l. 275-469: KDA ---------------------------------------------------------------------------------
class AliceAIKDA(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.act_fn = ACT2FN[config.hidden_act]
        conv_kwargs = {"kernel_size": self.conv_kernel_size, "padding": self.conv_kernel_size - 1, "bias": False}
        self.q_conv1d = nn.Conv1d(self.key_dim, self.key_dim, groups=self.key_dim, **conv_kwargs)
        self.k_conv1d = nn.Conv1d(self.key_dim, self.key_dim, groups=self.key_dim, **conv_kwargs)
        self.v_conv1d = nn.Conv1d(self.value_dim, self.value_dim, groups=self.value_dim, **conv_kwargs)
        self.q_proj = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_v_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_v_dim, self.key_dim, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, self.num_k_heads, bias=False)
        self.g_a_proj = nn.Linear(config.hidden_size, self.head_v_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_v_dim, self.value_dim, bias=False)
        self.o_norm = AliceAIRMSNorm(self.head_v_dim, config.rms_norm_eps, zero_centered=False)
        self.o_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.a_log_bias = nn.Parameter(torch.empty(self.num_k_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim))

    def _causal_conv(self, hidden_states, convolution):
        channels_first = hidden_states.transpose(1, 2)
        sequence_length = channels_first.shape[-1]
        output = self.act_fn(convolution(channels_first)[..., : channels_first.shape[-1]])
        return output[..., -sequence_length:].transpose(1, 2)

    def _torch_kda(self, query, key, value, alpha, beta, initial_state):
        dt = value.dtype                                                # remote: .float() throughout (lifted)
        query = functional.normalize(query.to(dt), dim=-1) * self.head_k_dim ** -0.5
        key = functional.normalize(key.to(dt), dim=-1)
        gate = -self.a_log_bias.to(dt).exp().view(1, 1, self.num_k_heads, 1) * functional.softplus(
            alpha.to(dt) + self.dt_bias.to(dt).view(1, 1, self.num_k_heads, self.head_k_dim))
        beta = beta.to(dt).sigmoid()
        if self.config.kda_allow_negative_eigenvalues:
            beta = beta * 2.0
        state = (initial_state.to(dt) if initial_state is not None
                 else query.new_zeros(query.shape[0], self.num_v_heads, self.head_k_dim, self.head_v_dim))
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
            gate = gate.repeat_interleave(repeat, dim=2)
            beta = beta.repeat_interleave(repeat, dim=2)
        outputs = []
        for token_idx in range(query.shape[1]):
            query_i = query[:, token_idx]
            key_i = key[:, token_idx]
            value_i = value[:, token_idx].to(dt)
            state = state * gate[:, token_idx].exp().unsqueeze(-1)
            prediction = torch.einsum("bhk,bhkv->bhv", key_i, state)
            delta = (value_i - prediction) * beta[:, token_idx].unsqueeze(-1)
            state = state + torch.einsum("bhk,bhv->bhkv", key_i, delta)
            outputs.append(torch.einsum("bhk,bhkv->bhv", query_i, state))
        return torch.stack(outputs, dim=1).to(value.dtype), state

    def forward(self, hidden_states):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        alpha = self.f_b_proj(self.f_a_proj(hidden_states))
        beta = self.b_proj(hidden_states)
        query = self._causal_conv(query, self.q_conv1d)
        key = self._causal_conv(key, self.k_conv1d)
        value = self._causal_conv(value, self.v_conv1d)
        query = query.view(*query.shape[:2], self.num_k_heads, self.head_k_dim)
        key = key.view(*key.shape[:2], self.num_k_heads, self.head_k_dim)
        value = value.view(*value.shape[:2], self.num_v_heads, self.head_v_dim)
        alpha = alpha.view(*alpha.shape[:2], self.num_k_heads, self.head_k_dim)
        output, _ = self._torch_kda(query, key, value, alpha, beta, None)
        output = self.o_norm(output)
        output = output * torch.sigmoid(output_gate.view_as(output))
        return self.o_proj(output.flatten(-2))


# --- l. 472-578: MLP, experts, sigmoid router, sparse MoE block --------------------------------------
class AliceAIMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class AliceAIExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_up_proj = nn.Parameter(torch.empty(config.num_experts, 2 * config.moe_intermediate_size,
                                                     config.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(config.num_experts, config.hidden_size,
                                                  config.moe_intermediate_size))

    def forward(self, hidden_states, expert_indices, expert_weights):
        output = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            token_indices, slots = torch.where(expert_indices == expert_idx)
            if token_indices.numel() == 0:
                continue
            expert_input = hidden_states[token_indices]
            gate_up = functional.linear(expert_input, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = functional.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            expert_output = expert_output * expert_weights[token_indices, slots, None]
            output.index_add_(0, token_indices, expert_output.to(output.dtype))
        return output


class AliceAISigmoidTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.top_k = config.num_experts_per_tok
        if config.router_bias_correction:
            self.register_buffer("e_score_correction_bias", torch.zeros(config.num_experts))
        else:
            self.e_score_correction_bias = None

    def forward(self, hidden_states):
        router_logits = functional.linear(hidden_states, self.weight.to(hidden_states.dtype))  # remote: .float()
        scores = router_logits.sigmoid()
        selection_scores = (scores if self.e_score_correction_bias is None
                            else scores + self.e_score_correction_bias)
        indices = selection_scores.topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return router_logits, weights.to(hidden_states.dtype), indices


class AliceAISparseMoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = AliceAISigmoidTopKRouter(config)
        self.experts = AliceAIExperts(config)
        self.shared_expert = AliceAIMLP(config.hidden_size, config.shared_expert_intermediate_size,
                                        config.hidden_act)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, original_shape[-1])
        router_logits, weights, indices = self.gate(flattened)
        routed = self.experts(flattened, indices, weights)
        shared = self.shared_expert(flattened) * torch.sigmoid(self.shared_expert_gate(flattened))
        return (routed + shared).reshape(original_shape), router_logits


# --- l. 581-648: decoder layer with Block AttnRes -----------------------------------------------------
class AliceAIDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = AliceAIKDA(config, layer_idx)
        else:
            self.self_attn = AliceAIAttention(config, layer_idx)
        self.input_layernorm = AliceAIRMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=True)
        self.post_attention_layernorm = AliceAIRMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=True)
        self.mlp = AliceAISparseMoEBlock(config)
        if layer_idx != 0:
            self.attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.attn_res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.rms_norm_eps

    def _mix(self, completed, partial, projection, norm_weight):
        sources = completed if partial is None else [*completed, partial]
        return _depth_softmax_mix(sources, projection.weight, norm_weight, self.eps)

    def forward(self, embeddings, completed_blocks, partial, position_embeddings, cache_position):
        if self.layer_idx == 0:
            mixed = embeddings
        else:
            mixed = self._mix(completed_blocks, partial, self.attn_res_proj, self.attn_res_norm_weight)
        mixed = self.input_layernorm(mixed)
        if self.layer_type == "linear_attention":
            attention_output = self.linear_attn(mixed)
        else:
            attention_output, _ = self.self_attn(mixed, position_embeddings, cache_position)
        partial = attention_output if partial is None else partial + attention_output
        mixed = self._mix(completed_blocks, partial, self.mlp_res_proj, self.mlp_res_norm_weight)
        moe_output, router_logits = self.mlp(self.post_attention_layernorm(mixed))
        return partial + moe_output, router_logits


# --- l. 669-704: _init_weights ------------------------------------------------------------------------
@torch.no_grad()
def init_weights_(model, seed=0):
    """AliceAIPreTrainedModel._init_weights over every module (what post_init does; nothing is loaded, so every
    parameter `_needs_initialization`). The seed is ours."""
    torch.manual_seed(seed)
    std = model.config.initializer_range
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
        elif isinstance(module, AliceAIExperts):
            module.gate_up_proj.normal_(mean=0.0, std=std)
            module.down_proj.normal_(mean=0.0, std=std)
        elif isinstance(module, AliceAISigmoidTopKRouter):
            module.weight.normal_(mean=0.0, std=std)
        elif isinstance(module, AliceAIKDA):
            module.a_log_bias.uniform_(0, 16).log_()
            dt = torch.exp(torch.rand_like(module.dt_bias) * (math.log(0.1) - math.log(0.001))
                           + math.log(0.001)).clamp_min(1e-4)
            module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        elif isinstance(module, AliceAIFinalBlockAttnRes):
            module.res_proj.weight.zero_()
        elif isinstance(module, AliceAIDecoderLayer):
            if hasattr(module, "attn_res_proj"):
                module.attn_res_proj.weight.zero_()
            module.mlp_res_proj.weight.zero_()
    return model


# --- l. 707-871: model and causal LM ------------------------------------------------------------------
class AliceAIModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([AliceAIDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.attnres_final = AliceAIFinalBlockAttnRes(config)
        self.norm = AliceAIRMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=True)
        self.rotary_emb = AliceAIRotaryEmbedding(config)

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(position_ids, inputs_embeds.dtype)
        completed_blocks = [inputs_embeds]
        partial = None
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx > 0 and layer_idx % self.config.block_attn_res_block_size == 0:
                completed_blocks.append(partial)
                partial = None
            partial, _ = layer(inputs_embeds, completed_blocks=tuple(completed_blocks), partial=partial,
                               position_embeddings=position_embeddings, cache_position=cache_position)
        return self.norm(self.attnres_final(completed_blocks, partial))


class AliceAIForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AliceAIModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids)))
