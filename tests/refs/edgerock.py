"""REFERENCE transcription of rockmatrix-ai/EdgeRock (torch backend, no cache), read as text, never run.

Source repo: rockmatrix-ai/EdgeRock, snapshot f8c04fa0ddf8fab1aefdd4726535a8c3fa305972
  configuration_edgerock.py  sha256 dd763ef277f04bee2692b78e72dd32032804345ffad2245de68e869970e2c745
    EdgeRockConfig (l. 9-15; the SGLANG_EXTERNAL_MODEL_PACKAGE branches l. 5-6, 16-20 are not transcribed)
  modeling_edgerock.py       sha256 0e9a2ae7cc3ca20dc0d58b59ed2081052497817ece86af05b240c5a0ee2aa01e
    _gqa_tile / indexed_gqa_torch (l. 22-42), INDEPENDENT_MEMORY_REVISION (l. 44), _memory_rope (l. 46-49),
    select_topk / merge_topk / exclusive_prefix_topk (l. 51-79),
    IndependentGQAMemory: __init__ (l. 124-145), _write_kv (l. 147-166, unpacked path), _query_scores
    (l. 168-176), _sequence (l. 178-275, the state=None branches), forward (l. 315-336, cache_layer=None);
    Qwen3_5SegmentAwareGatedDeltaNet (l. 355-390, torch backend), reference_swa (l. 412-427),
    SUPPORTED_FIXED_MEMORY_REVISIONS / DEFAULT_FIXED_MEMORY_WINDOW / is_fixed_memory_revision (l. 450-462),
    Qwen3_5FixedWindowAttention (l. 823-889, past_key_values=None, torch backend),
    Qwen3_5FixedMemoryDecoderLayer (l. 891-929, unfused residual norm), Qwen3_5FixedMemoryTextModel
    (l. 931-978, no cache), Qwen3_5FixedMemoryForCausalLM (l. 980-1027, the unchunked path),
    _install_fixed_memory_modules (l. 1032-1058), EdgeRockModel / EdgeRockForCausalLM (l. 1084-1093).
Not transcribed: the cache classes and decode path (IndependentMemoryState, _decode, FixedMemory*Layer /
FixedMemoryHybridDynamicCache, l. 81-120, 277-313, 464-821), the CUDA / Triton / flash-attention kernels.

The remote file subclasses the stock transformers 5.x Qwen3.5 classes (Qwen3_5Attention, Qwen3_5DecoderLayer,
Qwen3_5ForCausalLM, Qwen3_5GatedDeltaNet, Qwen3_5TextModel, Qwen3_5RMSNorm, Qwen3_5MLP, ...; its l. 12, 342-345);
so does this reference (library code), and keeps those classes' own fp32 casts. The memory's admission
score cast (l. 174-175: features.float() @ admission_up.float()) is lifted to the input dtype (fp64).
Needs transformers 5.x (run it inside the TF5 subprocess, tests/test_universal_elephantine._tf5_run).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention, Qwen3_5DecoderLayer, Qwen3_5ForCausalLM, Qwen3_5GatedDeltaNet, Qwen3_5MLP,
    Qwen3_5ModelOutputWithPast, Qwen3_5PreTrainedModel, Qwen3_5RMSNorm, Qwen3_5RMSNormGated, Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb, rotate_half, torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule)


class EdgeRockConfig(Qwen3_5TextConfig):
    model_type = "edgerock"

    def __init__(self, backend="torch", prefill_chunk_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.prefill_chunk_size = prefill_chunk_size


def _gqa_tile(query, key, value, references):
    batch, blocks, chunk, heads, dim = query.shape
    groups, capacity = references.shape[-2:]
    key = F.pad(key, (0, 0, 0, 1)).reshape(batch * blocks * groups, 1, capacity + 1, dim)
    value = F.pad(value, (0, 0, 0, 1)).reshape_as(key)
    mask = F.pad(references >= 0, (0, 1), value=True)
    query = query.permute(0, 1, 3, 2, 4).reshape(batch * blocks * groups, 1, heads // groups * chunk, dim)
    mask = mask.reshape(batch * blocks * groups, 1, 1, capacity + 1)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=False)
    return output.reshape(batch, blocks, heads, chunk, dim).permute(0, 1, 3, 2, 4)


def indexed_gqa_torch(query, key, value, references):
    b = torch.arange(key.shape[0], device=query.device)[:, None, None, None]
    g = torch.arange(key.shape[2], device=query.device)[None, None, :, None]
    output = torch.empty_like(query)
    blocks_per_tile = max(1, (8 * 1024 * 1024) // (query.shape[0] * key.shape[2] * (references.shape[-1] + 1) * key.shape[-1]))
    for start in range(0, query.shape[1], blocks_per_tile):
        refs = references[:, start:start + blocks_per_tile]
        output[:, start:start + blocks_per_tile] = _gqa_tile(query[:, start:start + blocks_per_tile],
                                                             key[b, refs.clamp_min(0), g], value[b, refs.clamp_min(0), g], refs)
    return output


INDEPENDENT_MEMORY_REVISION = 'qwen35-text-original-gdn-swa-independent-gqa-memory-v1'


def _memory_rope(values, cos, sin):
    width = cos.shape[-1]
    rotated = values[..., :width]
    return torch.cat((rotated * cos + rotate_half(rotated) * sin, values[..., width:]), -1)


def select_topk(scores, positions, references, capacity):
    positions = positions.long().masked_fill(references < 0, -1)
    order = positions.masked_fill(references < 0, torch.iinfo(torch.long).max).argsort(dim=-1, stable=True)
    scores, positions, references = [x.gather(-1, order) for x in (scores, positions, references)]
    order = scores.argsort(dim=-1, descending=True, stable=True)[..., :capacity]
    result = [x.gather(-1, order) for x in (scores, positions, references)]
    missing = capacity - order.shape[-1]
    if missing:
        result = [F.pad(x, (0, missing), value=value) for x, value in zip(result, (-math.inf, -1, -1))]
    return tuple(result)


def merge_topk(left, right, capacity):
    return select_topk(*(torch.cat((a, b), -1) for a, b in zip(left, right)), capacity)


def exclusive_prefix_topk(scores, positions, references, capacity):
    count = scores.shape[1]
    size = 1 << (count - 1).bit_length()
    leaf = select_topk(scores.detach(), positions, references, capacity)
    levels = [tuple(F.pad(x, (0, 0, 0, 0, 0, size - count), value=v) for x, v in zip(leaf, (-math.inf, -1, -1)))]
    while levels[-1][0].shape[1] > 1:
        previous = levels[-1]
        levels.append(merge_topk(tuple(x[:, 0::2] for x in previous), tuple(x[:, 1::2] for x in previous), capacity))
    prefix = tuple(torch.full_like(x, v) for x, v in zip(levels[-1], (-math.inf, -1, -1)))
    for children in reversed(levels[:-1]):
        right = merge_topk(prefix, tuple(x[:, 0::2] for x in children), capacity)
        prefix = tuple(torch.stack((a, b), 2).flatten(1, 2) for a, b in zip(prefix, right))
    final = merge_topk(tuple(x[:, count - 1:count] for x in prefix), tuple(x[:, count - 1:count] for x in leaf), capacity)
    return tuple(torch.cat((x[:, :count], y), 1) for x, y in zip(prefix, final))


class IndependentGQAMemory(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backend = getattr(config, 'backend', 'torch')
        settings = config.independent_memory
        self.hidden_size, self.heads = (config.hidden_size, config.num_attention_heads)
        self.groups, self.dim = (config.num_key_value_heads, config.head_dim)
        self.capacity, self.max_chunk = (settings['capacity'], settings['max_commit_chunk'])
        self.window = config.fixed_memory_swa_window
        hidden, dim, rank, intermediate = (self.hidden_size, self.dim, settings['admission_rank'], settings['writer_intermediate_size'])
        self.input_norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.q_norm = Qwen3_5RMSNorm(dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(dim, eps=config.rms_norm_eps)
        for name, n_in, n_out in (('q_proj', hidden, self.heads * dim), ('k_proj', hidden, self.groups * dim),
                                  ('v_proj', hidden, self.groups * dim), ('o_proj', self.heads * dim, hidden),
                                  ('write_gate_proj', hidden, intermediate), ('write_up_proj', hidden, intermediate),
                                  ('write_down_proj', intermediate, hidden), ('admission_down', hidden, rank),
                                  ('admission_up', rank, self.groups)):
            setattr(self, name, nn.Linear(n_in, n_out, bias=False))
        self.read_scale = nn.Parameter(torch.zeros(self.heads))
        self.read_gate_weight = nn.Parameter(torch.zeros(self.heads, dim))
        self.read_gate_bias = nn.Parameter(torch.zeros(self.heads))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
        nn.init.zeros_(self.write_down_proj.weight)
        nn.init.zeros_(self.admission_up.weight)

    def _write_kv(self, u, position_embeddings):
        def write(x):
            return x + self.write_down_proj(F.silu(self.write_gate_proj(x)) * self.write_up_proj(x))
        z = torch.cat([write(part) for part in u.split(4096)], 0)
        key, value = (self.k_proj(z), self.v_proj(z))
        key = self.k_norm(key.unflatten(-1, (self.groups, self.dim)))
        value = value.unflatten(-1, (self.groups, self.dim)).contiguous()
        cos, sin = (x[:, None] for x in position_embeddings)
        return (_memory_rope(key, cos, sin), value)

    def _query_scores(self, u, position_embeddings):
        query = self.q_norm(self.q_proj(u).unflatten(-1, (self.heads, self.dim)))
        gate = self.read_scale.tanh() * 2 * torch.sigmoid((query * self.read_gate_weight).sum(-1) / math.sqrt(self.dim) + self.read_gate_bias)
        cos, sin = (x[:, :, None] for x in position_embeddings)
        query = _memory_rope(query, cos, sin)
        features = F.silu(self.admission_down(u.detach()))
        scores = F.linear(features, self.admission_up.weight)          # remote: both cast .float()
        return (query, scores, gate)

    def _sequence(self, u, query, scores, position_embeddings, valid, chunk, state=None):
        assert state is None
        batch, length, heads, dim = query.shape
        b = torch.arange(batch, device=query.device)[:, None]
        if valid is None:
            order = None
            q, s = (query, scores)
            counts = torch.full((batch,), length, device=query.device, dtype=torch.long)
        else:
            order = valid.long().argsort(dim=1, descending=True, stable=True)
            q, s = (query[b, order], scores[b, order])
            counts = valid.sum(1)
        old_count = torch.zeros_like(counts)
        old_total = torch.zeros_like(counts)
        prefix_length = 0
        extent = math.ceil((length + prefix_length) / chunk) * chunk
        at = torch.arange(extent, device=query.device)[None].expand(batch, -1)
        valid_extent = at < old_count[:, None] + counts[:, None]
        q = F.pad(q, (0, 0, 0, 0, 0, extent - length))
        s = F.pad(s, (0, 0, 0, extent - length), value=-math.inf)
        shift = 0
        s = s.masked_fill(~valid_extent[..., None], -math.inf)
        positions = (old_total - old_count)[:, None] + at
        refs = (at + shift).masked_fill(~valid_extent, -1)
        blocks = extent // chunk
        metadata = [x[:, :, None].expand(-1, -1, self.groups) if x.ndim == 2 else x for x in (s, positions, refs)]
        metadata = [x.reshape(batch, blocks, chunk, self.groups).transpose(2, 3) for x in metadata]
        prefix = exclusive_prefix_topk(*metadata, self.capacity)
        boundary = torch.arange(1, blocks + 1, device=q.device)[None, :, None, None] * chunk
        entrant = prefix[2][:, 1:] - shift
        used_boundary = boundary < (old_count + counts)[:, None, None, None]
        take = (entrant >= boundary - chunk) & (entrant < boundary) & used_boundary
        needed = torch.zeros((batch, extent + 1), device=q.device, dtype=torch.bool)
        needed.scatter_(1, entrant.masked_fill(~take, extent).flatten(1), True)
        needed = needed[:, :extent] & valid_extent & (at >= old_count[:, None])
        selected_batch, selected_position = needed.nonzero(as_tuple=True)
        original = selected_position - old_count[selected_batch]
        if order is not None:
            original = order[selected_batch, original]
        angles = tuple(x.expand(batch, -1, -1)[selected_batch, original] for x in position_embeddings)
        k, v = self._write_kv(u[selected_batch, original], angles)
        references = torch.full((batch, shift + extent), -1, device=q.device, dtype=torch.long)
        source_k, source_v = (k[None], v[None])
        offset = 0
        references[selected_batch, shift + selected_position] = offset + torch.arange(len(k), device=q.device)
        source_k, source_v = (F.pad(x, (0, 0, 0, 0, 0, 1)) for x in (source_k, source_v))
        read_refs = references[b[:, :, None, None], prefix[2][:, :-1].clamp_min(0)]
        read_refs = read_refs.masked_fill(prefix[2][:, :-1] < 0, -1)
        read = indexed_gqa_torch(q.reshape(batch, blocks, chunk, heads, dim), source_k, source_v, read_refs).flatten(1, 2)
        output = read[:, :length]
        if order is not None:
            output = output[b, order.argsort(1)] * valid[..., None, None]
        return output

    def forward(self, hidden, position_embeddings, *, window=None, attention_mask=None, cache_layer=None, state_indices=None):
        assert cache_layer is None
        chunk = min(self.max_chunk, self.window if window is None else window)
        valid = None if attention_mask is None else attention_mask[:, -hidden.shape[1]:].bool()
        u = self.input_norm(hidden)
        query, scores, gate = self._query_scores(u, position_embeddings)
        output = self._sequence(u, query, scores, position_embeddings, valid, chunk)
        return self.o_proj((output * gate[..., None]).flatten(-2))


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    values = torch.cat((conv_state, hidden_states), -1).to(weight.dtype)
    conv_state.copy_(values[:, :, -conv_state.shape[-1]:])
    output = F.conv1d(values, weight[:, None], bias, groups=hidden_states.shape[1])
    return F.silu(output[:, :, -hidden_states.shape[-1]:]).to(hidden_states.dtype)


class Qwen3_5SegmentAwareGatedDeltaNet(Qwen3_5GatedDeltaNet):

    def __init__(self, config, layer_idx):
        assert config.backend == 'torch'
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.num_v_heads, self.num_k_heads = config.linear_num_value_heads, config.linear_num_key_heads
        self.head_k_dim, self.head_v_dim = config.linear_key_head_dim, config.linear_value_head_dim
        self.key_dim, self.value_dim = self.head_k_dim * self.num_k_heads, self.head_v_dim * self.num_v_heads
        self.conv_kernel_size, self.layer_idx = config.linear_conv_kernel_dim, layer_idx
        self.activation, self.act = config.hidden_act, ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.conv_kernel_size,
                                groups=self.conv_dim, padding=self.conv_kernel_size - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log())
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.causal_conv1d_fn = None
        self.causal_conv1d_update = torch_causal_conv1d_update
        self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

    def forward(self, hidden_states: Tensor, cache_params=None, attention_mask: Tensor | None = None, **kwargs: Any) -> Tensor:
        if attention_mask is not None:
            current_mask = attention_mask[:, -hidden_states.shape[1]:]
            hidden_states = hidden_states * current_mask[..., None].to(hidden_states.dtype)
            attention_mask = None
        return super().forward(hidden_states=hidden_states, cache_params=cache_params, attention_mask=attention_mask)


def reference_swa(q: Tensor, k: Tensor, v: Tensor, *, window: int, key_valid: Tensor | None = None, query_start: int = 0) -> Tensor:
    batch, query_len, heads, head_dim = q.shape
    key_len = k.shape[1]
    query_pos = torch.arange(query_len, device=q.device) + int(query_start)
    key_pos = torch.arange(key_len, device=q.device)
    allowed = key_pos[None, :] <= query_pos[:, None]
    allowed &= key_pos[None, :] >= query_pos[:, None] - int(window) + 1
    allowed = allowed[None, None].expand(batch, 1, -1, -1)
    if key_valid is not None:
        allowed = allowed & key_valid[:, None, None, :].to(dtype=torch.bool)
    output = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed,
                                            dropout_p=0.0, is_causal=False, enable_gqa=heads != k.shape[2])
    output = output.transpose(1, 2).contiguous().view(batch, query_len, heads, head_dim)
    if key_valid is not None:
        query_valid = key_valid[:, query_start:query_start + query_len]
        output = output * query_valid[:, :, None, None].to(dtype=output.dtype)
    return output


SUPPORTED_FIXED_MEMORY_REVISIONS = (INDEPENDENT_MEMORY_REVISION,)
DEFAULT_FIXED_MEMORY_WINDOW = 1024


def is_fixed_memory_revision(value: Any) -> bool:
    if isinstance(value, str):
        revision = value
    elif isinstance(value, dict):
        revision = value.get('fixed_memory_revision')
    else:
        revision = getattr(value, 'fixed_memory_revision', None)
        if revision is None and hasattr(value, 'config'):
            revision = getattr(value.config, 'fixed_memory_revision', None)
    return revision in SUPPORTED_FIXED_MEMORY_REVISIONS


class Qwen3_5FixedWindowAttention(Qwen3_5Attention):

    def __init__(self, config: Any, layer_idx: int):
        super().__init__(config, layer_idx)
        self.swa_window = int(getattr(config, 'fixed_memory_swa_window', DEFAULT_FIXED_MEMORY_WINDOW))

    @classmethod
    def from_attention(cls, source: Qwen3_5Attention) -> 'Qwen3_5FixedWindowAttention':
        module = cls(source.config, source.layer_idx)
        module.to(device=source.q_proj.weight.device, dtype=source.q_proj.weight.dtype)
        incompatible = module.load_state_dict(source.state_dict(), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'unexpected SWA conversion keys: {incompatible}')
        return module

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        assert past_key_values is None and self.config.backend == 'torch'
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_projected = self.q_proj(hidden_states)
        k_projected = self.k_proj(hidden_states)
        v_projected = self.v_proj(hidden_states)
        query, gate = torch.chunk(q_projected.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(k_projected.view(hidden_shape)).transpose(1, 2)
        value = v_projected.view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        window = int(kwargs.get('swa_window_size', self.swa_window))
        if window <= 0 or window > self.swa_window:
            raise ValueError(f'runtime SWA window must be in [1, {self.swa_window}], got {window}')
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError('fixed-window Qwen attention expects a 2D padding mask')
        current_valid = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device) if attention_mask is None \
            else attention_mask[:, -query.shape[1]:].bool()
        local = reference_swa(query, key, value, window=window, key_valid=current_valid)
        output = local.reshape(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate)
        return (self.o_proj(output), None)


class Qwen3_5FixedMemoryDecoderLayer(Qwen3_5DecoderLayer):

    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.hidden_size, self.layer_type = config.hidden_size, config.layer_types[layer_idx]
        if self.layer_type == 'linear_attention':
            self.linear_attn = Qwen3_5SegmentAwareGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        def run_attention(values):
            if hasattr(self, 'linear_attn'):
                return self.linear_attn(hidden_states=values, cache_params=past_key_values, attention_mask=attention_mask, **kwargs)
            attention_output, _ = self.self_attn(hidden_states=values, attention_mask=attention_mask, position_ids=position_ids,
                                                 past_key_values=past_key_values, position_embeddings=position_embeddings, **kwargs)
            return attention_output
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = run_attention(hidden_states)
        if hasattr(self, 'memory'):
            memory_output = self.memory(residual, position_embeddings, window=kwargs.get('swa_window_size'),
                                        attention_mask=attention_mask, cache_layer=None)
            hidden_states = hidden_states + memory_output
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5FixedMemoryTextModel(Qwen3_5TextModel):

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen3_5FixedMemoryDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                use_cache=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('specify exactly one of input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        assert not use_cache and past_key_values is None, "the cache path is not transcribed"
        if position_ids is None:
            seen = 0
            position_ids = torch.arange(seen, seen + inputs_embeds.shape[1], device=inputs_embeds.device).view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids
        masks = attention_mask if isinstance(attention_mask, dict) else {'full_attention': attention_mask, 'linear_attention': attention_mask}
        kwargs.pop('output_hidden_states', None)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
        for index, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            layer_kwargs = {'position_embeddings': position_embeddings, 'attention_mask': masks[self.config.layer_types[index]],
                            'position_ids': text_position_ids, 'past_key_values': past_key_values, 'use_cache': use_cache, **kwargs}
            hidden_states = decoder_layer(hidden_states, **layer_kwargs)
        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Qwen3_5FixedMemoryForCausalLM(Qwen3_5ForCausalLM):
    config_class = EdgeRockConfig
    _no_split_modules = ['Qwen3_5FixedMemoryDecoderLayer', 'Qwen3_5FixedWindowAttention', 'IndependentGQAMemory']

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = EdgeRockModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                use_cache=True, logits_to_keep=0, **kwargs):
        assert not use_cache, "the chunked / cached prefill path (l. 998-1024) is not transcribed"
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                               logits_to_keep=logits_to_keep, **kwargs)


def _install_fixed_memory_modules(model):
    config = model.config
    if not is_fixed_memory_revision(config):
        raise ValueError('fixed-memory config revision is missing or unsupported')
    swa_layers = tuple(int(index) for index in config.fixed_memory_swa_layers)
    independent = config.fixed_memory_revision == INDEPENDENT_MEMORY_REVISION
    for index in swa_layers:
        source = model.model.layers[index].self_attn
        if type(source) is Qwen3_5Attention:
            model.model.layers[index].self_attn = Qwen3_5FixedWindowAttention.from_attention(source)
        elif not isinstance(source, Qwen3_5FixedWindowAttention):
            raise TypeError(f'layer {index} is not an original Qwen attention layer')
        if independent:
            layer = model.model.layers[index]
            layer.memory = IndependentGQAMemory(config).to(device=source.q_proj.weight.device, dtype=source.q_proj.weight.dtype)
    for index, layer_type in enumerate(config.layer_types):
        if layer_type != 'linear_attention':
            continue
        source = model.model.layers[index].linear_attn
        if type(source) is Qwen3_5GatedDeltaNet:
            source.__class__ = Qwen3_5SegmentAwareGatedDeltaNet
        elif not isinstance(source, Qwen3_5SegmentAwareGatedDeltaNet):
            raise TypeError(f'layer {index} is not an original Qwen GDN layer')
    for layer in model.model.layers:
        layer.__class__ = Qwen3_5FixedMemoryDecoderLayer
    if not isinstance(model.model, Qwen3_5FixedMemoryTextModel):
        model.model.__class__ = Qwen3_5FixedMemoryTextModel


class EdgeRockModel(Qwen3_5FixedMemoryTextModel):
    config_class = EdgeRockConfig

    def __init__(self, config):
        super().__init__(config)
        _install_fixed_memory_modules(SimpleNamespace(config=config, model=self))


class EdgeRockForCausalLM(Qwen3_5FixedMemoryForCausalLM):
    config_class = EdgeRockConfig
