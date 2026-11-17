"""REFERENCE transcription of Mini-K3 (Mamba-2 variants), text-only.

Source: Hub repo nkkbr/Mini-K3-1H-mamba2-n128-g8-v1_B, snapshot 60b095cc9d9e913dd44c48a732ea9f489101937f.
Files (read as text, never executed):
  configuration_mini_k3.py  sha256 94bd2c27173c0c65cf8ddb9315cd6ae1439ea93d41d80e37e41537a1667bb3e7
      MiniK3Config            lines 11-239 (fields, properties, is_*_layer, torch_dtype, validate,
                              tiny_for_test 198-239; the JSON helpers save_json / from_dict / from_json / bundled 174-196 are left out)
  mamba2_mixer.py           sha256 c30ccf113f83db2ef7bd10c05119694542ffa30600a276a2dea342a467c21ae3
      GroupRMSNormGated       lines 17-29
      Mamba2Mixer             lines 32-133 (doc_ids = None path only: _segmented_conv 69-83, forward 85-133)
      initialize_mamba2       lines 136-147
  modeling_mini_k3.py       sha256 74fb90a99cbd3473017239e8403a57b0ca7917e19a794de870a84a3601177067
      RMSNorm                 lines 54-64
      situ_glu                lines 101-104
      SiTUMLP                 lines 107-128
      RoutedExpert            lines 131-152
      GatedMLA                lines 293-377 (attention_mask = None path: causal SDPA, NoPE)
      MoERouter               lines 380-405 (forward; the training-only quantile_balancing_update is left out)
      StableLatentMoE         lines 430-520 (reference dispatcher; no router record)
      apply_attention_residual lines 523-544
      MiniK3DecoderLayer      lines 547-627
      MiniK3Model             lines 630-716 (no gradient checkpointing, no router records)
      MiniK3ForCausalLM       lines 719-839 (__init__, reset_parameters 744-785, forward without labels)
Left out: HeadRMSNorm / CausalDepthwiseConv1d / KimiDeltaAttention (lines 67-98, 155-290): the Mamba-2
variants have kda_layers = [] (validate() still allows KDA layers; this reference raises on them).

Precision: the remote code computes norms, the SSD recurrence, the router and AttnRes in fp32 via
`.float()` casts; here every such cast is lifted to fp64 (`_up`), and the model is meant to be cast
to fp64 as a whole, so the reference is fp64-clean.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _up(t: Tensor) -> Tensor:
    """The remote `.float()` cast, lifted to fp64."""
    return t.double()


# ---- configuration_mini_k3.py: MiniK3Config (lines 11-239) ----------------------------------------

@dataclass
class MiniK3Config:
    architecture: str = "MiniK3ForCausalLM"
    model_name: str = "Mini-K3-1H-mamba2-n128-g8"
    model_type: str = "mini_k3"
    schema_version: int = 1

    vocab_size: int = 163_840
    hidden_size: int = 1_024
    num_hidden_layers: int = 13
    kda_layers: list[int] = field(default_factory=list)
    mamba_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 6, 7, 9, 10, 11])
    mla_layers: list[int] = field(default_factory=lambda: [4, 8, 12, 13])

    mamba_expand: int = 2
    mamba_head_dim: int = 64
    mamba_d_state: int = 128
    mamba_ngroups: int = 8
    mamba_conv_kernel_size: int = 4
    mamba_chunk_size: int = 128

    num_attention_heads: int = 12
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0
    use_full_rank_output_gate: bool = True

    q_lora_rank: int = 256
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True

    first_k_dense_replace: int = 1
    dense_intermediate_size: int = 4_864
    routed_expert_hidden_size: int = 512
    moe_intermediate_size: int = 448
    num_experts: int = 64
    num_experts_per_token: int = 4
    num_shared_experts: int = 2
    moe_router_activation: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    use_quantile_balancing: bool = True

    attn_res_mode: str = "block"
    attn_res_block_size: int = 4
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    rms_norm_eps: float = 1.0e-5
    initializer_range: float = 0.02
    max_position_embeddings: int = 131_072

    pad_token_id: int = 163_839
    bos_token_id: int = 163_584
    eos_token_id: int = 163_586
    tie_word_embeddings: bool = False
    parameter_dtype: str = "bfloat16"
    kda_backend: str = "auto"
    attention_backend: str = "sdpa"
    gradient_checkpointing: bool = True

    expected_logical_parameters: int | None = 1_016_780_524
    expected_activated_parameters: int | None = 353_556_204

    @property
    def kda_qkv_width(self) -> int:
        return self.num_attention_heads * self.kda_head_dim

    @property
    def mla_q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mla_q_width(self) -> int:
        return self.num_attention_heads * self.mla_q_head_dim

    @property
    def mla_value_width(self) -> int:
        return self.num_attention_heads * self.v_head_dim

    @property
    def num_moe_layers(self) -> int:
        return self.num_hidden_layers - self.first_k_dense_replace

    def is_kda_layer(self, zero_based_layer_index: int) -> bool:
        return zero_based_layer_index + 1 in self.kda_layers

    def is_mamba_layer(self, zero_based_layer_index: int) -> bool:
        return zero_based_layer_index + 1 in self.mamba_layers

    def torch_dtype(self):
        names = {
            "float32": torch.float32, "float": torch.float32, "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16,
        }
        try:
            return names[self.parameter_dtype.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported parameter_dtype={self.parameter_dtype!r}") from exc

    def validate(self) -> None:
        all_layers = self.kda_layers + self.mamba_layers + self.mla_layers
        expected = list(range(1, self.num_hidden_layers + 1))
        if sorted(all_layers) != expected or len(set(all_layers)) != len(all_layers):
            raise ValueError("kda_layers, mamba_layers and mla_layers must partition decoder layers")
        if self.mamba_expand <= 0 or self.mamba_head_dim <= 0 or self.mamba_d_state <= 0:
            raise ValueError("Mamba-2 dimensions must be positive")
        if self.hidden_size * self.mamba_expand % self.mamba_head_dim:
            raise ValueError("Mamba-2 inner width must divide into heads")
        mamba_heads = self.hidden_size * self.mamba_expand // self.mamba_head_dim
        if self.mamba_ngroups <= 0 or mamba_heads % self.mamba_ngroups:
            raise ValueError("Mamba-2 B/C groups must divide the Mamba head count")
        if self.mamba_conv_kernel_size <= 0 or self.mamba_chunk_size <= 0:
            raise ValueError("Mamba-2 convolution and scan chunk sizes must be positive")
        if self.first_k_dense_replace < 0 or self.first_k_dense_replace > self.num_hidden_layers:
            raise ValueError("first_k_dense_replace is out of range")
        if self.num_experts_per_token <= 0 or self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be in [1, num_experts]")
        if self.moe_router_activation != "sigmoid":
            raise ValueError("This K3 definition requires a sigmoid MoE router")
        if not self.moe_renormalize:
            raise ValueError("This K3 definition requires selected router weights to be renormalized")
        if not self.use_quantile_balancing:
            raise ValueError("The baseline requires Quantile Balancing")
        if not self.mla_use_nope:
            raise ValueError("Baseline Mini-K3 uses NoPE MLA")
        if not self.mla_use_output_gate or not self.use_full_rank_output_gate:
            raise ValueError("The baseline requires full-rank KDA and MLA output gates")
        if self.tie_word_embeddings:
            raise ValueError("The baseline uses an untied LM head")
        if self.kda_head_dim != self.v_head_dim:
            raise ValueError("Reference KDA requires equal key/value head dimensions")
        if self.routed_expert_hidden_size <= 0 or self.moe_intermediate_size <= 0:
            raise ValueError("MoE dimensions must be positive")
        if self.attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be positive")
        if self.attn_res_mode != "block":
            raise ValueError("Mamba-2 variants retain baseline Block AttnRes")
        if self.short_conv_kernel_size <= 0:
            raise ValueError("short_conv_kernel_size must be positive")
        if self.kda_backend not in {"auto", "reference", "fla"}:
            raise ValueError("kda_backend must be auto, reference, or fla")
        if self.attention_backend != "sdpa":
            raise ValueError("Only PyTorch SDPA is implemented in the portable reference")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def tiny_for_test(cls) -> "MiniK3Config":
        """A CPU-sized config that exercises every architectural component (lines 198-239)."""
        config = cls(
            model_name="mini-k3-tiny-test",
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=5,
            kda_layers=[],
            mamba_layers=[1, 2, 3],
            mla_layers=[4, 5],
            mamba_head_dim=8,
            mamba_d_state=8,
            mamba_ngroups=4,
            num_attention_heads=2,
            kda_head_dim=8,
            short_conv_kernel_size=4,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=8,
            first_k_dense_replace=1,
            dense_intermediate_size=64,
            routed_expert_hidden_size=16,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_token=2,
            num_shared_experts=2,
            attn_res_block_size=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            parameter_dtype="float32",
            kda_backend="reference",
            gradient_checkpointing=False,
            expected_logical_parameters=None,
            expected_activated_parameters=None,
        )
        config.validate()
        return config


# ---- mamba2_mixer.py ---------------------------------------------------------------------------

class GroupRMSNormGated(nn.Module):                                     # lines 17-29
    def __init__(self, width: int, groups: int, eps: float, *, device=None, dtype=None):
        super().__init__()
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype))

    def forward(self, y: Tensor, gate: Tensor) -> Tensor:
        batch, seq, width = y.shape
        group_width = width // self.groups
        gated = (_up(y) * F.silu(_up(gate))).view(batch, seq, self.groups, group_width)
        normalized = gated * torch.rsqrt(gated.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized.reshape(batch, seq, width) * _up(self.weight)).to(y.dtype)


class Mamba2Mixer(nn.Module):                                           # lines 32-133
    def __init__(self, config, layer_index: int, *, device=None, dtype=None):
        super().__init__()
        self.layer_index = layer_index
        self.d_model = config.hidden_size
        self.d_inner = config.hidden_size * config.mamba_expand
        self.d_state = config.mamba_d_state
        self.headdim = config.mamba_head_dim
        self.nheads = self.d_inner // self.headdim
        self.ngroups = config.mamba_ngroups
        self.d_conv = config.mamba_conv_kernel_size
        self.chunk_size = config.mamba_chunk_size
        self.conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads,
                                 bias=False, device=device, dtype=dtype)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.d_conv, groups=self.conv_dim, bias=True,
                                device=device, dtype=dtype)
        self.A_log = nn.Parameter(torch.empty(self.nheads, device=device, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(self.nheads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, device=device, dtype=torch.float32))
        self.norm = GroupRMSNormGated(self.d_inner, self.ngroups, config.rms_norm_eps, device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, device=device, dtype=dtype)

    def _segmented_conv(self, x: Tensor) -> Tensor:                    # doc_ids = None
        batch, seq, channels = x.shape
        result = x.new_zeros(batch, seq, channels)
        weight = self.conv1d.weight[:, 0, :]
        # Conv1d's last kernel element multiplies the current token.
        for lag in range(self.d_conv):
            if lag >= seq:
                break
            source = x[:, : seq - lag]
            result[:, lag:] = result[:, lag:] + source * weight[:, self.d_conv - lag - 1]
        result = result + self.conv1d.bias
        return F.silu(result)

    def forward(self, hidden_states: Tensor, doc_ids: Tensor | None = None) -> Tensor:
        if doc_ids is not None:
            raise NotImplementedError("the doc_ids path is not transcribed")
        batch, seq, _ = hidden_states.shape
        raw = self.in_proj(hidden_states)
        gate, xbc_raw, dt_raw = torch.split(raw, [self.d_inner, self.conv_dim, self.nheads], dim=-1)
        xbc = self._segmented_conv(xbc_raw)
        x, b, c = torch.split(xbc, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x = _up(x.view(batch, seq, self.nheads, self.headdim))
        b = _up(b.view(batch, seq, self.ngroups, self.d_state))
        c = _up(c.view(batch, seq, self.ngroups, self.d_state))
        dt = F.softplus(_up(dt_raw) + self.dt_bias.view(1, 1, -1))
        a = torch.exp(dt * -self.A_log.exp().view(1, 1, -1))
        b = b.repeat_interleave(self.nheads // self.ngroups, dim=2)
        c = c.repeat_interleave(self.nheads // self.ngroups, dim=2)
        state = x.new_zeros(batch, self.nheads, self.headdim, self.d_state)
        values = []
        for token in range(seq):
            state = state * a[:, token, :, None, None] + (
                dt[:, token, :, None, None] * x[:, token, :, :, None] * b[:, token, :, None, :]
            )
            value = (state * c[:, token, :, None, :]).sum(dim=-1)
            value = value + self.D[None, :, None] * x[:, token]
            values.append(value)
        y = torch.stack(values, dim=1).reshape(batch, seq, self.d_inner)
        y = self.norm(y.to(hidden_states.dtype), gate)
        return self.out_proj(y)


@torch.no_grad()
def initialize_mamba2(module: Mamba2Mixer, *, std: float = 0.02) -> None:   # lines 136-147
    for linear in (module.in_proj, module.out_proj):
        nn.init.normal_(linear.weight, mean=0.0, std=std)
    nn.init.normal_(module.conv1d.weight, mean=0.0, std=std)
    nn.init.zeros_(module.conv1d.bias)
    nn.init.ones_(module.norm.weight)
    nn.init.ones_(module.D)
    module.A_log.uniform_(1.0, 16.0).log_()
    raw = torch.empty_like(module.dt_bias).uniform_(math.log(1e-3), math.log(1e-1))
    dt = raw.exp().clamp_min_(1e-4)
    module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))


# ---- modeling_mini_k3.py -----------------------------------------------------------------------

class RMSNorm(nn.Module):                                               # lines 54-64
    def __init__(self, size: int, eps: float, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        source_dtype = x.dtype
        x_float = _up(x)
        normalized = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * _up(self.weight)).to(source_dtype)


def situ_glu(gate: Tensor, up: Tensor, gate_beta: float, up_beta: float) -> Tensor:   # lines 101-104
    capped_gate = gate_beta * torch.tanh(gate / gate_beta)
    capped_up = up_beta * torch.tanh(up / up_beta)
    return capped_gate * torch.sigmoid(gate) * capped_up


class SiTUMLP(nn.Module):                                               # lines 107-128
    def __init__(self, hidden_size, intermediate_size, gate_beta, up_beta, *, device=None, dtype=None):
        super().__init__()
        self.gate_beta = gate_beta
        self.up_beta = up_beta
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(situ_glu(self.gate_proj(x), self.up_proj(x), self.gate_beta, self.up_beta))


class RoutedExpert(nn.Module):                                          # lines 131-152
    """K3-compatible routed-expert naming: w1=gate, w3=up, w2=down."""

    def __init__(self, hidden_size, intermediate_size, gate_beta, up_beta, *, device=None, dtype=None):
        super().__init__()
        self.gate_beta = gate_beta
        self.up_beta = up_beta
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(situ_glu(self.w1(x), self.w3(x), self.gate_beta, self.up_beta))


class GatedMLA(nn.Module):                                              # lines 293-377
    def __init__(self, config: MiniK3Config, layer_index: int, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_head_dim = config.mla_q_head_dim
        self.v_head_dim = config.v_head_dim
        kw = dict(bias=False, device=device, dtype=dtype)
        self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, **kw)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps, device=device, dtype=dtype)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, **kw)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, **kw)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps, device=device, dtype=dtype)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank,
                                   self.num_heads * (config.qk_nope_head_dim + config.v_head_dim), **kw)
        self.g_proj = nn.Linear(self.hidden_size, config.mla_value_width, **kw)
        self.o_proj = nn.Linear(config.mla_value_width, self.hidden_size, **kw)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if attention_mask is not None:
            raise NotImplementedError("the attention_mask path is not transcribed")
        batch, sequence, _ = hidden_states.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch, sequence, self.num_heads, self.q_head_dim).transpose(1, 2)

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        latent, shared_k_part = torch.split(compressed, [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(latent))
        kv = kv.view(batch, sequence, self.num_heads,
                     self.config.qk_nope_head_dim + self.config.v_head_dim).transpose(1, 2)
        k_content, value = torch.split(kv, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1)
        shared_k_part = shared_k_part[:, None, :, :].expand(batch, self.num_heads, sequence,
                                                            self.config.qk_rope_head_dim)
        key = torch.cat((k_content, shared_k_part), dim=-1)
        # No rotary transform is applied: this is K3's NoPE MLA.
        attended = F.scaled_dot_product_attention(q, key, value, attn_mask=None, dropout_p=0.0, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        attended = attended * torch.sigmoid(_up(self.g_proj(hidden_states))).to(attended.dtype)
        return self.o_proj(attended)


class MoERouter(nn.Module):                                             # lines 380-405
    def __init__(self, config: MiniK3Config, *, device=None, dtype=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.renormalize = config.moe_renormalize
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, device=device, dtype=dtype))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, device=device, dtype=torch.float32), requires_grad=False)

    def forward(self, hidden_states: Tensor):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores = torch.sigmoid(F.linear(_up(flat), _up(self.weight)))
        choice_scores = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(choice_scores, self.top_k, dim=-1, sorted=False).indices
        topk_weights = scores.gather(1, topk_indices)
        if self.renormalize and self.top_k > 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights, scores


class StableLatentMoE(nn.Module):                                       # lines 430-520
    def __init__(self, config: MiniK3Config, layer_index: int, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        kw = dict(bias=False, device=device, dtype=dtype)
        self.router = MoERouter(config, device=device, dtype=dtype)
        self.routed_expert_down_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, **kw)
        self.experts = nn.ModuleList([
            RoutedExpert(config.routed_expert_hidden_size, config.moe_intermediate_size,
                         config.activation_situ_beta, config.activation_situ_linear_beta, device=device, dtype=dtype)
            for _ in range(config.num_experts)
        ])
        self.routed_expert_norm = RMSNorm(config.routed_expert_hidden_size, config.rms_norm_eps,
                                          device=device, dtype=dtype)
        self.routed_expert_up_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, **kw)
        self.shared_experts = SiTUMLP(config.hidden_size, config.num_shared_experts * config.moe_intermediate_size,
                                      config.activation_situ_beta, config.activation_situ_linear_beta,
                                      device=device, dtype=dtype)

    def _reference_dispatch(self, latent: Tensor, topk_indices: Tensor, topk_weights: Tensor) -> Tensor:
        output = torch.zeros_like(latent)
        for expert_index, expert in enumerate(self.experts):
            selected = (topk_indices == expert_index).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            token_indices = selected[:, 0]
            slot_indices = selected[:, 1]
            expert_input = latent.index_select(0, token_indices)
            expert_output = expert(expert_input)
            weights = topk_weights[token_indices, slot_indices].to(expert_output.dtype).unsqueeze(-1)
            output = output.index_add(0, token_indices, expert_output * weights)
        return output

    def forward(self, hidden_states: Tensor):
        original_shape = hidden_states.shape
        topk_indices, topk_weights, scores = self.router(hidden_states)
        flat = hidden_states.reshape(-1, original_shape[-1])
        latent = self.routed_expert_down_proj(flat)
        routed = self._reference_dispatch(latent, topk_indices, topk_weights)
        routed = self.routed_expert_up_proj(self.routed_expert_norm(routed))
        shared = self.shared_experts(flat)
        return (routed + shared).view(original_shape), None


def apply_attention_residual(prefix_sum: Tensor, block_residual: Tensor, projection: nn.Linear,
                             norm: RMSNorm) -> Tensor:                  # lines 523-544
    values = torch.cat((block_residual, prefix_sum.unsqueeze(2)), dim=2)
    values_float = _up(values)
    normalized = values_float * torch.rsqrt(values_float.square().mean(dim=-1, keepdim=True) + norm.eps)
    score_weight = _up(norm.weight) * _up(projection.weight.squeeze(0))
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1).unsqueeze(-1)
    return (probabilities * values_float).sum(dim=2).to(prefix_sum.dtype)


class MiniK3DecoderLayer(nn.Module):                                    # lines 547-627
    def __init__(self, config: MiniK3Config, layer_index: int, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.is_kda = config.is_kda_layer(layer_index)
        self.is_mamba = config.is_mamba_layer(layer_index)
        if self.is_mamba:
            self.self_attn = Mamba2Mixer(config, layer_index, device=device, dtype=dtype)
        elif self.is_kda:
            raise NotImplementedError("KimiDeltaAttention is not transcribed (Mamba-2 variants have no KDA layers)")
        else:
            self.self_attn = GatedMLA(config, layer_index, device=device, dtype=dtype)

        if layer_index < config.first_k_dense_replace:
            self.feed_forward = SiTUMLP(config.hidden_size, config.dense_intermediate_size,
                                        config.activation_situ_beta, config.activation_situ_linear_beta,
                                        device=device, dtype=dtype)
            self.is_moe = False
        else:
            self.feed_forward = StableLatentMoE(config, layer_index, device=device, dtype=dtype)
            self.is_moe = True

        d, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RMSNorm(d, eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(d, eps, device=device, dtype=dtype)
        self.self_attention_res_norm = RMSNorm(d, eps, device=device, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(d, 1, bias=False, device=device, dtype=dtype)
        self.mlp_res_norm = RMSNorm(d, eps, device=device, dtype=dtype)
        self.mlp_res_proj = nn.Linear(d, 1, bias=False, device=device, dtype=dtype)

    def forward(self, prefix_sum: Tensor, block_residual: Tensor, attention_mask: Tensor | None = None):
        hidden_states = prefix_sum
        if block_residual.shape[2] > 0:
            hidden_states = apply_attention_residual(prefix_sum, block_residual, self.self_attention_res_proj,
                                                     self.self_attention_res_norm)

        if self.layer_index % self.config.attn_res_block_size == 0:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(2)), dim=2)
            prefix_sum = None

        attention_output = self.self_attn(self.input_layernorm(hidden_states), attention_mask)
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output

        hidden_states = apply_attention_residual(prefix_sum, block_residual, self.mlp_res_proj, self.mlp_res_norm)
        normalized = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            feed_forward_output, _ = self.feed_forward(normalized)
        else:
            feed_forward_output = self.feed_forward(normalized)
        prefix_sum = prefix_sum + feed_forward_output
        return prefix_sum, block_residual


class MiniK3Model(nn.Module):                                           # lines 630-716
    def __init__(self, config: MiniK3Config, *, device=None):
        super().__init__()
        config.validate()
        self.config = config
        dtype = config.torch_dtype()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id,
                                         device=device, dtype=dtype)
        self.layers = nn.ModuleList([MiniK3DecoderLayer(config, index, device=device, dtype=dtype)
                                     for index in range(config.num_hidden_layers)])
        self.output_attn_res_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False, device=device, dtype=dtype)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, input_ids: Tensor):
        hidden_states = self.embed_tokens(input_ids)
        batch, sequence, hidden = hidden_states.shape
        block_residual = hidden_states.new_empty(batch, sequence, 0, hidden)
        for layer in self.layers:
            hidden_states, block_residual = layer(hidden_states, block_residual, None)
        hidden_states = apply_attention_residual(hidden_states, block_residual, self.output_attn_res_proj,
                                                 self.output_attn_res_norm)
        return self.norm(hidden_states)


@dataclass
class MiniK3CausalLMOutput:
    logits: Tensor
    hidden_states: Tensor


class MiniK3ForCausalLM(nn.Module):                                     # lines 719-839
    def __init__(self, config: MiniK3Config, *, device=None):
        super().__init__()
        self.config = config
        self.model = MiniK3Model(config, device=device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device,
                                 dtype=config.torch_dtype())
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:                                 # lines 744-785
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight[module.padding_idx].zero_()
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, Mamba2Mixer):
                initialize_mamba2(module, std=std)
            elif isinstance(module, MoERouter):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.zeros_(module.e_score_correction_bias)
        # Attention Residuals use zero pseudo-query vectors.
        for name, parameter in self.named_parameters():
            if name.endswith(("self_attention_res_proj.weight", "mlp_res_proj.weight",
                              "output_attn_res_proj.weight")):
                parameter.zero_()

    def forward(self, input_ids: Tensor) -> MiniK3CausalLMOutput:
        hidden_states = self.model(input_ids)
        return MiniK3CausalLMOutput(logits=self.lm_head(hidden_states), hidden_states=hidden_states)
