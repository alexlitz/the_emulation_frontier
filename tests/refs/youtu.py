"""REFERENCE transcription of the Youtu-LLM / UTU-VL MoE text model (plain torch, fp64-clean).

Source (read as text, never executed):
  repo       tencent/Youtu-LLM-A7B-Base
  snapshot   6b0d02b2912c2975806db5a4922a1e091fa2f5d3
  modeling_youtu_vl_moe.py        sha256 9b5dc77938a5582d7b5133a717fb83f923b3b2ba1925a9ebfc55c9e42beb1ddc
  configuration_youtu_vl_moe.py   sha256 9224ac573046ec83dcbb93f4a4ab8cecb6b28dd6a673589d52b1790da606ba44

Transcribed (modeling_youtu_vl_moe.py line ranges):
  rotate_half 193-196, apply_rotary_pos_emb 199-221, repeat_kv 224-229, eager_attention_forward 257-278,
  YoutuVLRMSNorm 286-300, YoutuVLRotaryEmbedding 307-403, _quantile_balancing_enabled 406-409,
  YoutuVLMLP 416-427, YoutuVLTopkRouter 430-562, YoutuVLMoE 603-700 (single shared expert; the
  multimodal dual shared experts 565-600 are not transcribed and are refused), _is_kda_layer 708-709,
  YoutuVLMLAAttention 834-993 (eager path; the flash-attention branches 947-965 / 981-982 are inert under
  eager and omitted), YoutuVLCausalDepthwiseConv1d 996-1127 (the no-cache prefill path 1120-1127),
  YoutuVLKDAAttention 1130-1345 (no cache, no padding mask; the pure-torch short conv and the
  YoutuVLRMSNorm * sigmoid(gate) output path, i.e. kda_use_fla_short_conv=False,
  kda_use_fused_rmsnorm_gated=False as the remote test config sets), build_youtu_vl_attention 1348-1353,
  YoutuVLDecoderLayer 1441-1509, YoutuVLTextModel 1516-1659, YoutuVLModel.build_decoder_modality_id
  1873-1929 and YoutuVLModel.forward 1931-2003 (text-only: no pixel values, so position_ids stay None,
  1980-1990), YoutuVLForConditionalGeneration 2010-2096.
  configuration_youtu_vl_moe.py YoutuVLMoEConfig 8-262: the text fields only (the Qwen3-VL vision config
  and the vision tower are not built; from_youtu never reads them).

chunk_kda: the remote imports fla's Triton chunk_kda (fla 0.5.2 pinned). Here it is the fla-core 0.5.2
torch references, transcribed: fla/ops/kda/naive.py naive_recurrent_kda (lines 12-62, sha256
60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016), fla/ops/kda/gate.py
naive_kda_lowerbound_gate / naive_kda_gate (lines 27-70, sha256
04fc6aa1857eb216ddecc354480285014c25dded74fa646491e9c17e7e11ef35), and the kernel prologue of
fla/ops/kda/chunk.py (sha256 a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8, lines 54-62,
394-398): l2norm x / sqrt(sum x^2 + 1e-6) (fla/modules/l2norm.py eps=1e-6), beta = sigmoid(beta), scale K^-1/2,
and the safe_gate range check on lower_bound.

Library pieces the remote itself imports and that are reused here: torch.nn.Conv1d (the short conv's base
class), transformers' ACT2FN and CausalLMOutputWithPast / BaseModelOutputWithPast. The causal mask is the
eager additive mask create_causal_mask builds (dtype-min above the diagonal).

fp32 casts lifted to fp64 (none changes the math): YoutuVLRMSNorm's .to(float32), the rotary table
(inv_freq built from int64 .float(), the freqs matmul under autocast-off .float()), the eager softmax's
.float(), the router's F.linear(h.float(), W.float()) and the qb_beta / e_score_correction_bias buffers that
the remote's _apply pins to fp32, the beta .float(), the MLA / KDA output-gate sigmoid(.float()), and fla's
naive references' .float() / output_dtype=float32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

MODALITY_PAD = -1
MODALITY_TEXT = 0
MODALITY_VISION = 1


class YoutuVLMoEConfig:
    """configuration_youtu_vl_moe.py YoutuVLMoEConfig, text fields (same defaults)."""

    model_type = "youtu_vl_moe"

    def __init__(self, vocab_size=129280, hidden_size=3072, intermediate_size=12288, moe_intermediate_size=1024,
                 num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=32, head_dim=128,
                 multi_latent_attention=True, experimental_attention_variant="kda", linear_attention_pattern=None,
                 q_lora_rank=1024, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, qk_head_dim=128,
                 qk_pos_emb_head_dim=64, v_head_dim=128, attention_output_gate=True,
                 mla_attn_implementation="flash_attention_2", mla_output_gate_type="head_wise",
                 short_conv_kernel_size=4, use_full_rank_gate=True, kda_gate_lower_bound=-5.0,
                 kda_use_fla_short_conv=True, kda_use_fused_rmsnorm_gated=True,
                 n_routed_experts=256, num_experts_per_tok=8, first_k_dense_replace=1, n_shared_experts=1,
                 moe_shared_experts_intermediate_size=1024, enable_multimodal_shared_experts=False,
                 moe_router_modality_aware_expert_bias=True, moe_router_token_level_modality=True,
                 n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.83, scoring_func="sigmoid",
                 topk_method="quantile_balancing", moe_router_enable_expert_bias=True,
                 moe_router_expert_bias_update_method="quantile", moe_latent_size=None, moe_latent_rmsnorm=False,
                 hidden_act="silu", max_position_embeddings=262144, rms_norm_eps=1e-6, rope_theta=1600000.0,
                 rope_parameters=None, attention_bias=False, attention_dropout=0.0, rotary_percent=0.5,
                 mrope_section=None, image_token_id=128017, video_token_id=128040, pad_token_id=None,
                 tie_word_embeddings=False, **kwargs):
        self.vocab_size, self.hidden_size, self.intermediate_size = vocab_size, hidden_size, intermediate_size
        self.moe_intermediate_size, self.num_hidden_layers = moe_intermediate_size, num_hidden_layers
        self.num_attention_heads, self.num_key_value_heads, self.head_dim = num_attention_heads, num_key_value_heads, head_dim
        self.multi_latent_attention = multi_latent_attention
        self.experimental_attention_variant = experimental_attention_variant
        self.linear_attention_pattern = (linear_attention_pattern if linear_attention_pattern is not None
                                         else ([1, 1, 1, 0] * (num_hidden_layers // 4)))
        self.q_lora_rank, self.kv_lora_rank = q_lora_rank, kv_lora_rank
        self.qk_nope_head_dim, self.qk_rope_head_dim = qk_nope_head_dim, qk_rope_head_dim
        self.qk_head_dim, self.qk_pos_emb_head_dim, self.v_head_dim = qk_head_dim, qk_pos_emb_head_dim, v_head_dim
        self.attention_output_gate = attention_output_gate
        self.mla_attn_implementation, self.mla_output_gate_type = mla_attn_implementation, mla_output_gate_type
        self.short_conv_kernel_size, self.use_full_rank_gate = short_conv_kernel_size, use_full_rank_gate
        self.kda_gate_lower_bound = kda_gate_lower_bound
        self.kda_use_fla_short_conv, self.kda_use_fused_rmsnorm_gated = kda_use_fla_short_conv, kda_use_fused_rmsnorm_gated
        self.n_routed_experts, self.num_experts_per_tok = n_routed_experts, num_experts_per_tok
        self.first_k_dense_replace, self.n_shared_experts = first_k_dense_replace, n_shared_experts
        self.moe_shared_experts_intermediate_size = moe_shared_experts_intermediate_size
        self.enable_multimodal_shared_experts = enable_multimodal_shared_experts
        self.moe_router_modality_aware_expert_bias = moe_router_modality_aware_expert_bias
        self.moe_router_token_level_modality = moe_router_token_level_modality
        self.n_group, self.topk_group, self.norm_topk_prob = n_group, topk_group, norm_topk_prob
        self.routed_scaling_factor, self.scoring_func, self.topk_method = routed_scaling_factor, scoring_func, topk_method
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_router_expert_bias_update_method = moe_router_expert_bias_update_method
        self.moe_latent_size, self.moe_latent_rmsnorm = moe_latent_size, moe_latent_rmsnorm
        self.hidden_act, self.max_position_embeddings, self.rms_norm_eps = hidden_act, max_position_embeddings, rms_norm_eps
        self.rope_theta, self.attention_bias, self.attention_dropout = rope_theta, attention_bias, attention_dropout
        self.rotary_percent = rotary_percent
        self.mrope_section = mrope_section if mrope_section is not None else [12, 10, 10]
        self.rope_parameters = dict(rope_parameters) if rope_parameters is not None else {}
        self.rope_parameters.setdefault("rope_type", "default")
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters.setdefault("mrope_section", self.mrope_section)
        self.rope_parameters.setdefault("mrope_interleaved", True)
        self.image_token_id, self.video_token_id = image_token_id, video_token_id
        self.pad_token_id, self.tie_word_embeddings = pad_token_id, tie_word_embeddings
        self._attn_implementation = "eager"                 # YoutuVLForConditionalGeneration.__init__ (2025)

    @property
    def text_config(self):
        return self

    def get_text_config(self, *a, **k):
        return self


def _uses_token_level_router_groups(config):
    return bool(getattr(config, "moe_router_modality_aware_expert_bias", False)
                and getattr(config, "moe_router_token_level_modality", False))


def _needs_decoder_modality_id(config):
    return bool(getattr(config, "enable_multimodal_shared_experts", False) or _uses_token_level_router_groups(config))


# ---- fla-core 0.5.2 torch references (fp64) -------------------------------------------------------------

def naive_kda_gate(g, A_log, dt_bias=None, output_dtype=torch.float64):
    H, _ = g.shape[-2:]
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    return (-A_log.view(H, 1).exp() * F.softplus(g)).to(output_dtype)


def naive_kda_lowerbound_gate(g, A_log, dt_bias=None, lower_bound=-5.0, output_dtype=torch.float64):
    H, _ = g.shape[-2:]
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
              safe_gate=False, lower_bound=None, scale=None, cu_seqlens=None, **kw):
    """fla 0.5.2 chunk_kda with in-kernel gating, as its torch references (no varlen / state here)."""
    assert cu_seqlens is None and initial_state is None
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")
    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if use_beta_sigmoid_in_kernel:
        beta = beta.sigmoid()
    if use_gate_in_kernel:
        g = (naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound, output_dtype=q.dtype) if lower_bound is not None
             else naive_kda_gate(g, A_log, dt_bias, output_dtype=q.dtype))
    return naive_recurrent_kda(q, k, v, g, beta.to(q.dtype), scale=scale, initial_state=initial_state,
                               output_final_state=output_final_state)


# ---- modeling_youtu_vl_moe.py -----------------------------------------------------------------------------

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, multi_latent_attention=False):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if multi_latent_attention:                          # Megatron MLA interleave: cat(t[0::2], t[1::2])
        q = torch.cat((q[..., 0::2], q[..., 1::2]), dim=-1)
        k = torch.cat((k[..., 0::2], k[..., 1::2]), dim=-1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    if q_pass.shape[-1] > 0:
        q_rot = torch.cat((q_rot, q_pass), dim=-1)
    if k_pass.shape[-1] > 0:
        k_rot = torch.cat((k_rot, k_pass), dim=-1)
    return q_rot, k_rot


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query.dtype)       # .float() lifted
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class YoutuVLRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype                                            # .to(float32) lifted
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class YoutuVLRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.mrope_section = config.mrope_section
        self.rotary_percent = float(getattr(config, "rotary_percent", 1.0))
        if getattr(config, "experimental_attention_variant", None) == "kda":
            rotary_dim = int(config.qk_rope_head_dim)
        else:
            if self.rotary_percent <= 0.0 or self.rotary_percent > 1.0:
                raise ValueError(f"rotary_percent must be in (0, 1], got {self.rotary_percent}")
            rotary_dim = int(self.head_dim * self.rotary_percent)
            if rotary_dim % 2 != 0:
                rotary_dim -= 1
        self.rotary_dim = rotary_dim
        if sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError("mrope_section sum must equal rotary_dim//2")
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).double()
                                              / self.rotary_dim))                    # .float() lifted
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].double().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].double()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

    @staticmethod
    def _apply_interleaved_mrope(freqs, mrope_section):
        freqs_t = freqs[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]
        return freqs_t


def _quantile_balancing_enabled(config):
    method = getattr(config, "moe_router_expert_bias_update_method", "sign")
    topk_method = getattr(config, "topk_method", "noaux_tc")
    return method == "quantile" or topk_method in ("quantile_balancing", "quantile")


class YoutuVLMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class YoutuVLTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.use_expert_bias = getattr(config, "moe_router_enable_expert_bias", True)
        self.use_quantile_balancing = _quantile_balancing_enabled(config)
        if self.use_quantile_balancing and not self.use_expert_bias:
            raise ValueError("Quantile Balancing requires expert bias to be enabled.")
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        bias_shape = ((2, self.n_routed_experts) if getattr(config, "moe_router_modality_aware_expert_bias", False)
                      else (self.n_routed_experts,))
        if not self.use_expert_bias:
            self.qb_beta = None
            self.e_score_correction_bias = None
        elif self.use_quantile_balancing:
            self.register_buffer("qb_beta", torch.zeros(bias_shape))                # fp32 pin lifted
            self.e_score_correction_bias = None
        else:
            self.register_buffer("e_score_correction_bias", torch.zeros(bias_shape))
            self.qb_beta = None

    def _bias_for_tokens(self, bias, scores, modality_id):
        if bias.dim() == 2:
            if modality_id is None:
                return bias[0].unsqueeze(0)
            modality_id = modality_id.reshape(-1).to(device=scores.device)
            if modality_id.shape[0] != scores.shape[0]:
                raise ValueError("modality_id must align with router scores")
            bias_index = (modality_id == MODALITY_VISION).long()
            return bias.index_select(0, bias_index)
        return bias.unsqueeze(0)

    @torch.no_grad()
    def get_topk_indices(self, scores, modality_id=None, router_logits=None):
        scores = scores.view(-1, self.n_routed_experts)
        if self.use_quantile_balancing:
            logits = router_logits.view(-1, self.n_routed_experts)
            qb_beta = self._bias_for_tokens(self.qb_beta, logits, modality_id)
            return torch.topk(logits - qb_beta, k=self.top_k + 1, dim=-1).indices[:, : self.top_k]
        if self.e_score_correction_bias is None:
            scores_for_choice = scores
        else:
            scores_for_choice = scores + self._bias_for_tokens(self.e_score_correction_bias, scores, modality_id)
        if self.n_group > 1:
            group_scores = (scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
                            .topk(self.top_k // self.topk_group, dim=-1)[0].sum(dim=-1))
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (group_mask.unsqueeze(-1).expand(-1, self.n_group, self.n_routed_experts // self.n_group)
                          .reshape(-1, self.n_routed_experts))
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(self, hidden_states, modality_id=None):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states, self.weight)                         # .float() lifted
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores, modality_id=modality_id, router_logits=router_logits)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class YoutuVLMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe_latent_size = getattr(config, "moe_latent_size", None)
        expert_hidden_size = self.moe_latent_size if self.moe_latent_size else config.hidden_size
        self.fc1_latent_proj = None
        self.fc2_latent_proj = None
        self.latent_rmsnorm = None
        if self.moe_latent_size:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, self.moe_latent_size, bias=False)
            self.fc2_latent_proj = nn.Linear(self.moe_latent_size, config.hidden_size, bias=False)
            if getattr(config, "moe_latent_rmsnorm", False):
                self.latent_rmsnorm = YoutuVLRMSNorm(self.moe_latent_size, eps=config.rms_norm_eps)
        self.experts = nn.ModuleList([YoutuVLMLP(config, hidden_size=expert_hidden_size,
                                                 intermediate_size=config.moe_intermediate_size)
                                      for _ in range(config.n_routed_experts)])
        self.gate = YoutuVLTopkRouter(config)
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            if getattr(config, "enable_multimodal_shared_experts", False):
                raise NotImplementedError("YoutuVLMultiModalSharedExperts is not transcribed")
            self.shared_experts = YoutuVLMLP(config, intermediate_size=config.moe_shared_experts_intermediate_size)
        else:
            self.shared_experts = None

    def forward(self, hidden_states, modality_id=None):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states, modality_id=modality_id)
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        expert_input = hidden_states_flat
        if self.fc1_latent_proj is not None:
            expert_input = self.fc1_latent_proj(hidden_states_flat)
        final_hidden_states = torch.zeros(expert_input.shape[0], expert_input.shape[-1], dtype=topk_weights.dtype,
                                          device=expert_input.device)
        topk = topk_indices.shape[1]
        flat_expert_indices = topk_indices.reshape(-1)
        sorted_route_indices = flat_expert_indices.argsort(stable=True)
        tokens_per_expert = torch.bincount(flat_expert_indices, minlength=len(self.experts))
        start_idx = 0
        for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
            end_idx = start_idx + num_tokens
            if num_tokens:
                route_indices = sorted_route_indices[start_idx:end_idx]
                token_indices = torch.div(route_indices, topk, rounding_mode="floor")
                weight_indices = route_indices.remainder(topk)
                expert_output = self.experts[expert_idx](expert_input[token_indices])
                expert_weights = topk_weights[token_indices, weight_indices]
                final_hidden_states.index_add_(0, token_indices, expert_output * expert_weights.unsqueeze(-1))
            start_idx = end_idx
        if self.latent_rmsnorm is not None:
            final_hidden_states = self.latent_rmsnorm(final_hidden_states)
        if self.fc2_latent_proj is not None:
            final_hidden_states = self.fc2_latent_proj(final_hidden_states)
        final_hidden_states = final_hidden_states.view(*orig_shape).to(hidden_states.dtype)
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(residuals)
        return final_hidden_states


def _is_kda_layer(config, layer_idx):
    return config.linear_attention_pattern[layer_idx] == 1


class YoutuVLMLAAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = YoutuVLRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = YoutuVLRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.mla_output_gate_type = getattr(config, "mla_output_gate_type", "head_wise")
        if getattr(config, "attention_output_gate", False):
            if self.mla_output_gate_type == "channel_wise":
                self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)
            elif self.mla_output_gate_type == "head_wise":
                self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
            else:
                raise ValueError(f"Unsupported mla_output_gate_type={self.mla_output_gate_type!r}")
        else:
            self.g_proj = None
        self.attn_implementation = getattr(config, "mla_attn_implementation", getattr(config, "_attn_implementation", "eager"))
        assert self.attn_implementation == "eager", "only the eager MLA path is transcribed"
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        query_shape = (batch_size, seq_len, -1, self.q_head_dim)
        key_shape = (batch_size, seq_len, -1, self.qk_nope_head_dim + self.v_head_dim)
        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.view(batch_size, 1, seq_len, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, multi_latent_attention=True)
        k_rot = k_rot.expand(batch_size, self.num_heads, seq_len, self.qk_rope_head_dim)
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)
        attn_output, _ = eager_attention_forward(self, query_states, key_states, value_states, attention_mask,
                                                 dropout=0.0 if not self.training else self.attention_dropout,
                                                 scaling=self.scaling)
        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(hidden_states)).to(attn_output.dtype)       # .float() lifted
            if self.mla_output_gate_type == "head_wise":
                gate = gate.unsqueeze(-1)
            else:
                gate = gate.view(batch_size, seq_len, self.num_heads, self.v_head_dim)
            attn_output = attn_output * gate
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


class YoutuVLCausalDepthwiseConv1d(nn.Conv1d):
    def __init__(self, channels, kernel_size, bias=False, activation="silu"):
        super().__init__(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                         bias=bias, padding=kernel_size - 1)
        self.hidden_size = channels
        self.activation = activation
        if activation is not None and activation not in ("silu", "swish"):
            raise ValueError(f"Activation `{activation}` not supported by fallback short conv.")

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False, cu_seqlens=None, **unused):
        assert mask is None and cache is None and not output_final_state and cu_seqlens is None
        B, T, D = x.shape
        assert B * T != B, "the one-token decode step path is not transcribed"
        seq_len = x.size(1)
        y = super().forward(x.transpose(1, 2))[..., :seq_len].transpose(1, 2).contiguous()
        if residual is not None:
            y = y + residual
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        return y, None


class YoutuVLKDAAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = config.short_conv_kernel_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.kda_gate_lower_bound = config.kda_gate_lower_bound
        self.safe_gate = self.kda_gate_lower_bound is not None
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.projection_size, bias=False)
        assert not getattr(config, "kda_use_fla_short_conv", True), "fla ShortConvolution path not transcribed"
        self.use_fla_short_conv = False
        self.conv1d = YoutuVLCausalDepthwiseConv1d(channels=3 * self.projection_size, kernel_size=self.conv_size,
                                                   bias=False, activation="silu")
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads).uniform_(1, 16)))
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_size, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.use_full_rank_gate = getattr(config, "use_full_rank_gate", True)
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_size, bias=True)
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size))
        assert not getattr(config, "kda_use_fused_rmsnorm_gated", True), "FusedRMSNormGated path not transcribed"
        self.use_fused_rmsnorm_gated = False
        self.out_norm = YoutuVLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.projection_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask, cache_params=None, **kwargs):
        assert attention_mask is None and cache_params is None, "padding / cache paths not transcribed"
        qkv_proj_states = self.qkv_proj(hidden_states)
        qkv, _ = self.conv1d(qkv_proj_states, cache=None, output_final_state=False, cu_seqlens=None)
        batch_size, seq_len, _ = qkv.shape
        query, key, value = torch.split(qkv, [self.projection_size] * 3, dim=-1)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
        f_a = self.f_a_proj(hidden_states)
        g_decay = self.f_b_proj(f_a).view(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
        beta = self.b_proj(hidden_states)                                            # .float() lifted
        core_attn_out, _ = chunk_kda(q=query, k=key, v=value, g=g_decay, beta=beta, A_log=self.A_log,
                                     dt_bias=self.dt_bias, initial_state=None, output_final_state=False,
                                     use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                                     use_beta_sigmoid_in_kernel=True, safe_gate=self.safe_gate,
                                     lower_bound=self.kda_gate_lower_bound, transpose_state_layout=True,
                                     cu_seqlens=None)
        if self.use_full_rank_gate:
            output_gate = self.g_proj(hidden_states)
        else:
            output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        output_gate = output_gate.view(batch_size, seq_len, self.num_heads, self.head_dim)
        norm_out = self.out_norm(core_attn_out.reshape(-1, self.head_dim)).view_as(core_attn_out)
        norm_out = norm_out * torch.sigmoid(output_gate).to(norm_out.dtype)            # .float() lifted
        norm_out = norm_out.reshape(batch_size, seq_len, self.projection_size)
        return self.out_proj(norm_out)


def build_youtu_vl_attention(config, layer_idx):
    if getattr(config, "experimental_attention_variant", None) == "kda":
        if _is_kda_layer(config, layer_idx):
            return YoutuVLKDAAttention(config=config, layer_idx=layer_idx)
        return YoutuVLMLAAttention(config=config, layer_idx=layer_idx)
    raise NotImplementedError("the GQA + QK-norm YoutuVLAttention variant is not transcribed")


class YoutuVLDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = build_youtu_vl_attention(config=config, layer_idx=layer_idx)
        self.is_linear_attn = isinstance(self.self_attn, YoutuVLKDAAttention)
        self.mlp = YoutuVLMoE(config) if layer_idx >= config.first_k_dense_replace else YoutuVLMLP(config)
        self.input_layernorm = YoutuVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = YoutuVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, modality_id=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear_attn:
            hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, cache_params=None)
        else:
            attn_out = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                                      position_embeddings=position_embeddings)
            hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, YoutuVLMoE):
            hidden_states = self.mlp(hidden_states, modality_id=modality_id)
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def _causal_mask(inputs_embeds):
    """create_causal_mask's eager additive mask (no padding, no cache): dtype-min above the diagonal."""
    T, dt = inputs_embeds.shape[1], inputs_embeds.dtype
    m = torch.full((T, T), torch.finfo(dt).min, dtype=dt).triu(1)
    return m[None, None].expand(inputs_embeds.shape[0], 1, T, T)


class YoutuVLTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([YoutuVLDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = YoutuVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = YoutuVLRotaryEmbedding(config=config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None,
                use_cache=None, modality_id=None, **kwargs):
        assert attention_mask is None and not use_cache, "padding / cache paths not transcribed"
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = _causal_mask(inputs_embeds)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if (getattr(self.config, "enable_multimodal_shared_experts", False)
                or getattr(self.config, "moe_router_modality_aware_expert_bias", False)):
            if modality_id is None:
                modality_id = torch.full(hidden_states.shape[:2], MODALITY_TEXT, dtype=torch.int8)
            else:
                modality_id = modality_id.to(dtype=torch.int8)
        linear_attn_mask = None                          # flash_attn_mask is None without a 2-D attention mask
        for decoder_layer in self.layers:
            layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask
            hidden_states = decoder_layer(hidden_states, attention_mask=layer_mask,
                                          position_embeddings=position_embeddings, modality_id=modality_id)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class YoutuVLModel(nn.Module):
    """The multimodal backbone, text path only (no `visual`)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = YoutuVLTextModel(config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def build_decoder_modality_id(self, input_ids, inputs_embeds, attention_mask=None):
        if not _needs_decoder_modality_id(self.config):
            return None
        modality_id = torch.full_like(input_ids, fill_value=MODALITY_TEXT, dtype=torch.int8)
        vision_mask = input_ids == self.config.image_token_id
        if self.config.video_token_id is not None:
            vision_mask = vision_mask | (input_ids == self.config.video_token_id)
        modality_id = torch.where(vision_mask, torch.full_like(modality_id, MODALITY_VISION), modality_id)
        return modality_id.contiguous()

    def forward(self, input_ids=None, attention_mask=None, use_cache=None, **kwargs):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        decoder_modality_id = self.build_decoder_modality_id(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                                             attention_mask=attention_mask)
        return self.language_model(input_ids=None, attention_mask=attention_mask, position_ids=None,
                                   inputs_embeds=inputs_embeds, use_cache=use_cache, modality_id=decoder_modality_id)


class YoutuVLForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = YoutuVLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, use_cache=None, logits_to_keep=0, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return CausalLMOutputWithPast(logits=logits)
