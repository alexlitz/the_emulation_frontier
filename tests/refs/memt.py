"""MemT-1.0 (memt_qwen3_5_moe): a labelled transcription of the remote modelling code, read as text.

Source:   MemTensor/MemT-1.0-Nano, snapshot 30fe0b22422c4cc4c87475f98db5c8baa5c1a902
Files:    modeling_memt.py       sha256 b7a989249d651c62908383bc9a18ece91700c3ef590305753d700612263a5c8e
          configuration_memt.py  sha256 1fa524a4896924df78627d617e78f44af6e92a5adf20e9fa9df6e5da7eff14dc
Transcribed (modeling_memt.py):
    36-52    MemTRMSNormGated              -> RMSNormGated
    55-132   MemTMemory                    -> Memory (the session memory: a stable-GDN2 state read through a
                                              gated RMSNorm, written per token by commit events)
    206-209  _MemoryCall                   -> MemoryCall
    212-267  MemTAttention                 -> Attention (Qwen3.5 gated attention + the memory read)
    270-283  MemTDecoderLayer              -> DecoderLayer
    286-301  MemTTextModel                 -> TextModel
    304-307  MemTBackbone                  -> Backbone
    310-390  MemTQwen3_5MoeForCausalLM     -> ForCausalLM (__init__, forward; the session / commit API
                                              at 392-978 is not transcribed)
    153-203  MemTGatedDeltaNet is the stock Qwen3_5MoeGatedDeltaNet with a device-safe constructor (the
             same parameters, the native forward, torch kernels on CPU): the stock class is used.
configuration_memt.py 93-100 (get_text_config: a stock Qwen3_5MoeTextConfig from backbone text_config) is
the `text` argument; the field validation (11-91) is not transcribed.

Everything else is the stock transformers 5.x qwen3_5_moe module (library code), which the remote file
imports and subclasses. Departures, none of which changes a CPU forward: the attention interface is
eager_attention_forward (the remote looks it up by config._attn_implementation, "eager" here), the
autocast(enabled=False) around the memory read and the .to(device) moves are dropped. Import this module only under transformers >= 5 (the TF5 subprocess)."""
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen


class RMSNormGated(nn.Module):                                     # 36-52
    def __init__(self, head_dim, eps, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps
        self.softcap = config["output_gate_softcap_enabled"]
        self.limit = config["output_gate_softcap_limit"]

    def forward(self, value, gate):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        normalized = self.weight * normalized.to(value.dtype)
        activated = F.silu(gate.float())
        if self.softcap:
            activated = self.limit * torch.tanh(activated / self.limit)
        return (normalized * activated).to(value.dtype)


class Memory(nn.Module):                                           # 55-132
    def __init__(self, config, hidden_size, num_heads, head_dim, rms_norm_eps=1e-6):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.value_dim = num_heads * head_dim
        self.read_scale = head_dim ** -0.5
        self.in_proj_rows = (self.value_dim,) * 5 + (head_dim, head_dim)
        self.in_proj = nn.Linear(hidden_size, sum(self.in_proj_rows), bias=False)
        self.f_down = nn.Linear(head_dim, self.value_dim, bias=False)
        self.g_down = nn.Linear(head_dim, self.value_dim, bias=config["output_gate_bias"])
        self.decay_bias = nn.Parameter(torch.empty(self.value_dim))
        self.decay_scale_log = nn.Parameter(torch.zeros(num_heads))
        self.o_norm = RMSNormGated(head_dim, rms_norm_eps, config)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for block in self.in_proj.weight.split(self.in_proj_rows, dim=0):
            nn.init.xavier_uniform_(block, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.f_down.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_down.weight, gain=2 ** -2.5)
        if self.g_down.bias is not None:
            nn.init.zeros_(self.g_down.bias)
        nn.init.zeros_(self.decay_scale_log)
        half_life = torch.empty_like(self.decay_bias).uniform_(math.log(2048), math.log(32768)).exp()
        rate = math.log(2) / half_life
        self.decay_bias.copy_(rate + torch.log(-torch.expm1(-rate)))

    @staticmethod
    def _normalize(value, rounded):
        result = value.float() * torch.rsqrt(value.float().square().sum(-1, keepdim=True) + 1e-6)
        return result.to(value.dtype) if rounded else result

    def forward(self, hidden, state=None, commit=False, *, write_state=None):
        batch, length, _ = hidden.shape
        dim, width = self.head_dim, self.value_dim
        shape = (batch, length, self.num_heads, dim)
        if state is None and not commit:                           # 97-98: no state, no commit -> no read
            return hidden.new_zeros(batch, length, width), None
        snapshot = (hidden.new_zeros((batch, self.num_heads, dim, dim), dtype=torch.float32)
                    if state is None else state)
        projected = F.linear(hidden, self.in_proj.weight[: 3 * width if commit else width])
        query = self._normalize(projected[..., :width].reshape(shape), commit)
        read = self.read_scale * torch.einsum("blhk,bhkv->blhv", query.float(), snapshot.float())
        gate = self.g_down(F.linear(hidden, self.in_proj.weight[-dim:])).reshape(shape)
        read = self.o_norm(read.to(gate.dtype), gate).reshape(batch, length, width)
        if not commit:
            return read, state
        key, value = (item.reshape(shape) for item in projected[..., width:].split(width, -1))
        key = self._normalize(key, True)
        b, w, f = F.linear(hidden, self.in_proj.weight[3 * width: -dim]).split((width, width, dim), -1)
        sqrt_b = (0.5 * F.logsigmoid(b.float())).exp().to(b.dtype).reshape(shape)
        key = (key.float() * sqrt_b.float()).to(value.dtype).float()
        write_value = w.sigmoid().reshape(shape).float() * value.float()
        decay = (-self.decay_scale_log.float().exp().reshape(self.num_heads, 1)
                 * F.softplus(self.f_down(f).float() + self.decay_bias.float()).reshape(shape)).exp()
        updated = snapshot.float() if write_state is None else write_state.float()
        for index in range(length):
            u = key[:, index]
            updated = updated * decay[:, index].unsqueeze(-1)
            erase = (u.unsqueeze(-1) * updated).sum(-2)
            updated = updated + u.unsqueeze(-1) * (write_value[:, index] - erase).unsqueeze(-2)
        return read, updated


class MemoryCall:                                                  # 206-209
    def __init__(self, read, write=None):
        self.read, self.write, self.results = read, write, {}


class Attention(qwen.Qwen3_5MoeAttention):                          # 212-267
    def __init__(self, text, layer_idx, memory_config):
        super().__init__(text, layer_idx)
        self.memt = Memory(memory_config, text.hidden_size, text.num_attention_heads, text.head_dim, text.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                memory_call=None, **kwargs):
        shape = hidden_states.shape[:-1]
        heads = (*shape, -1, self.head_dim)
        query, gate = self.q_proj(hidden_states).view(*shape, -1, 2 * self.head_dim).chunk(2, dim=-1)
        query = self.q_norm(query.reshape(heads)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(heads)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(heads).transpose(1, 2)
        query, key = qwen.apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        output, weights = qwen.eager_attention_forward(self, query, key, value, attention_mask, dropout=0.0,
                                                       scaling=self.scaling, **kwargs)
        output = output.reshape(*shape, -1).contiguous() * torch.sigmoid(gate.reshape(*shape, -1))
        if memory_call is not None:
            state = memory_call.read.get(self.layer_idx)
            commit = memory_call.write is not None
            writer = memory_call.write.get(self.layer_idx) if commit else None
            read, updated = self.memt(hidden_states, state, commit, write_state=writer)
            output = output + read.to(output.dtype)
            if commit:
                memory_call.results[self.layer_idx] = updated
        return self.o_proj(output), weights


class DecoderLayer(qwen.Qwen3_5MoeDecoderLayer):                    # 270-283
    def __init__(self, config, layer_idx, memory_config):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.block_type = self.layer_type                          # the stock 5.x forward reads block_type
        if self.layer_type == "linear_attention":
            self.linear_attn = qwen.Qwen3_5MoeGatedDeltaNet(config, layer_idx)   # see 153-203 in the header
        else:
            self.self_attn = Attention(config, layer_idx, memory_config)
        self.mlp = qwen.Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = qwen.Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = qwen.Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class TextModel(qwen.Qwen3_5MoeTextModel):                          # 286-301
    def __init__(self, config, memory_config):
        qwen.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(config, i, memory_config) for i in range(config.num_hidden_layers)])
        self.norm = qwen.Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = qwen.Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Backbone(nn.Module):                                         # 304-307
    def __init__(self, text, memory_config):
        super().__init__()
        self.language_model = TextModel(text, memory_config)


class ForCausalLM(nn.Module):                                      # 310-390
    """`text` is the backbone's Qwen3_5MoeTextConfig (configuration_memt.py get_text_config), `memt` the
    config's memt dict. forward(memory_state=None) is the plain LM forward; a memory_state (layer index ->
    an fp32 (B, heads, hd, hd) state) is read by every attention layer (MemoryCall, read only)."""

    def __init__(self, text, memt):
        super().__init__()
        self.config = text
        self.memt = dict(memt)
        self.model = Backbone(text, self.memt)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def forward(self, input_ids, memory_state=None):
        out = self.model.language_model(input_ids=input_ids, use_cache=False,
                                        memory_call=MemoryCall(memory_state) if memory_state else None,
                                        return_dict=True)
        return self.lm_head(out.last_hidden_state)
