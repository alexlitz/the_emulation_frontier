"""REFERENCE transcription of GLM-5.3-SuperExpert-V2's own modelling code (read as text, never run).

Source repo:  shashanth-27xA9/GLM-5.3-SuperExpert-v2-9.2GB
Snapshot:     948ab036754e5c8cada135094037e88c2288639f
Files:
  configuration_glm_superexpert.py  sha256 e745dfe0d7534ad7419dad78961374ba6b5265b6e2a69df50ca2b0ded3e7e199
      GLMSuperExpertConfig                 lines 8-51
  modeling_glm_superexpert.py       sha256 5df456dd8d0ce3d6e9c66be7a82ee6bf48689790295acb16ebe9f07e4628dfa0
      DequantizedPackedLinear              lines 23-59
      DequantizedPackedEmbedding           lines 61-89
      RMSNorm                              lines 91-100
      GLMSuperExpertSingleMLP              lines 102-112
      GLMSuperExpertMoE                    lines 114-145
      GLMDenseMLP                          lines 147-157
      GLMSuperExpertAttention              lines 159-184
      GLMSuperExpertDecoderLayer           lines 186-208
      GLMSuperExpertModel                  lines 210-225
      GLMSuperExpertLanguageModelWrapper   lines 227-234
      GLMSuperExpertForCausalLM            lines 236-288

Class, attribute, buffer and parameter names mirror the remote module tree, so
elephantine.universal.from_glm_superexpert loads this object exactly as it loads the remote one. As in the
remote file, the config subclasses transformers' PretrainedConfig and the LM subclasses PreTrainedModel +
GenerationMixin (library code the remote file itself imports); construction order is the remote one, so the
same seed gives the same initial parameters and the same random packed nibbles.

fp64 lifts (the only deviations; everything else is verbatim):
  * DequantizedPackedEmbedding: the remote returns the fp32 dequantised rows, which would pin the whole residual
    stream to fp32. Here the identical fp32 dequantised values are cast (losslessly) to fp64 on return.
  * RMSNorm: the remote computes the variance in fp32 (hidden_states.to(torch.float32)); here it is computed in
    the input's dtype (fp64).
DequantizedPackedLinear keeps the remote's fp32 dequantisation ((nibble - 8) * (scale.float() / 7)) and casts the
dequantised weight to the input dtype, as the remote does; those weight values are exact in fp64.
"""
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel


class GLMSuperExpertConfig(PretrainedConfig):
    model_type = "glm_superexpert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size: int = 154880, hidden_size: int = 4096, intermediate_size: int = 12288,
                 moe_intermediate_size: int = 2048, num_hidden_layers: int = 45, num_attention_heads: int = 64,
                 num_key_value_heads: int = 64, n_routed_experts: int = 8, n_shared_experts: int = 1,
                 num_experts_per_tok: int = 8, first_k_dense_replace: int = 3, rms_norm_eps: float = 1e-5,
                 pad_token_id: int = 154820, eos_token_id: list = [154820, 154827, 154829],
                 tie_word_embeddings: bool = False, **kwargs):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.rms_norm_eps = rms_norm_eps

    def get_text_config(self, decoder=False):
        return self


class DequantizedPackedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        scale_key = prefix + "weight.scale"
        if scale_key in state_dict:
            self.scale.copy_(state_dict[scale_key])
            state_dict.pop(scale_key)
        elif prefix + "scale" in state_dict:
            self.scale.copy_(state_dict[prefix + "scale"])
            state_dict.pop(prefix + "scale")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if w.numel() == self.out_features * self.in_features:
            weight_f = (w.float() - 8.0) * (self.scale.float() / 7.0)
            weight_f = weight_f.reshape(self.out_features, self.in_features)
        else:
            low = (w & 0x0F).to(torch.float32)
            high = (w >> 4).to(torch.float32)
            unpacked = torch.stack([low, high], dim=-1).reshape(self.out_features, self.in_features)
            weight_f = (unpacked - 8.0) * (self.scale.float() / 7.0)
        return F.linear(x, weight_f.to(x.dtype), self.bias)


class DequantizedPackedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight", torch.zeros((num_embeddings, embedding_dim // 2), dtype=torch.uint8))
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float16))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        scale_key = prefix + "weight.scale"
        if scale_key in state_dict:
            self.scale.copy_(state_dict[scale_key])
            state_dict.pop(scale_key)
        elif prefix + "scale" in state_dict:
            self.scale.copy_(state_dict[prefix + "scale"])
            state_dict.pop(prefix + "scale")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        packed = self.weight[input_ids]
        if packed.shape[-1] == self.embedding_dim:
            return ((packed.float() - 8.0) * (self.scale.float() / 7.0)).to(self.scale.dtype)
        low = (packed & 0x0F).to(torch.float32)
        high = (packed >> 4).to(torch.float32)
        unpacked = torch.stack([low, high], dim=-1).reshape(*input_ids.shape, self.embedding_dim)
        return ((unpacked - 8.0) * (self.scale.float() / 7.0)).to(torch.float64)     # fp64 lift (lossless cast)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)                       # fp64 lift (remote: fp32)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(hidden_states.dtype)


class GLMSuperExpertSingleMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = DequantizedPackedLinear(hidden_size, intermediate_size)
        self.up_proj = DequantizedPackedLinear(hidden_size, intermediate_size)
        self.down_proj = DequantizedPackedLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GLMSuperExpertMoE(nn.Module):
    def __init__(self, config: GLMSuperExpertConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.super_experts = nn.ModuleList([
            GLMSuperExpertSingleMLP(self.hidden_size, self.intermediate_size)
            for _ in range(self.num_experts)
        ])

        if config.n_shared_experts > 0:
            self.shared_experts = GLMSuperExpertSingleMLP(self.hidden_size, config.intermediate_size)
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routing_logits = self.gate(hidden_states)
        routing_weights = F.softmax(routing_logits, dim=-1)

        final_output = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.super_experts):
            expert_out = expert(hidden_states)
            weight = routing_weights[..., i:i + 1]
            final_output = final_output + (weight * expert_out)

        if self.shared_experts is not None:
            final_output = final_output + self.shared_experts(hidden_states)

        return final_output


class GLMDenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = DequantizedPackedLinear(hidden_size, intermediate_size)
        self.up_proj = DequantizedPackedLinear(hidden_size, intermediate_size)
        self.down_proj = DequantizedPackedLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GLMSuperExpertAttention(nn.Module):
    def __init__(self, config: GLMSuperExpertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = DequantizedPackedLinear(self.hidden_size, self.hidden_size)
        self.k_proj = DequantizedPackedLinear(self.hidden_size, self.hidden_size)
        self.v_proj = DequantizedPackedLinear(self.hidden_size, self.hidden_size)
        self.o_proj = DequantizedPackedLinear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(context)


class GLMSuperExpertDecoderLayer(nn.Module):
    def __init__(self, config: GLMSuperExpertConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GLMSuperExpertAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx < config.first_k_dense_replace:
            self.mlp = GLMDenseMLP(config.hidden_size, config.intermediate_size)
        else:
            self.mlp = GLMSuperExpertMoE(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, attention_mask=attention_mask)
        hidden_states = residual + attn_out

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed)
        return residual + mlp_out


class GLMSuperExpertModel(nn.Module):
    def __init__(self, config: GLMSuperExpertConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = DequantizedPackedEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GLMSuperExpertDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return self.norm(hidden_states)


class GLMSuperExpertLanguageModelWrapper(nn.Module):
    def __init__(self, config: GLMSuperExpertConfig):
        super().__init__()
        self.language_model = GLMSuperExpertModel(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.language_model(input_ids, attention_mask=attention_mask)


class GLMSuperExpertForCausalLM(PreTrainedModel, GenerationMixin):
    _supports_flash_attn_2 = False
    _can_generate = True
    config_class = GLMSuperExpertConfig
    base_model_prefix = "model"

    def __init__(self, config: GLMSuperExpertConfig):
        super().__init__(config)
        self.model = GLMSuperExpertLanguageModelWrapper(config)
        self.lm_head = DequantizedPackedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = True,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None, hidden_states=None,
                                      attentions=None)
