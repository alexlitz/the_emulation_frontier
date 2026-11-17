"""GigaChat-3.5 (GigaChat35ForCausalLM), a labelled reference transcription in plain torch at fp64.

SOURCE (read as text, never executed):
    repo      ai-sage/GigaChat3.5-432B-A28B-Reasoning
    revision  02f55379ac0b20abd9378791bffd6254725e6f5c
    configuration_gigachat3_5.py  sha256 c14a2e75bf50466fa1987dafd1097782259db8a2050d9f8d5f9d77db380cc811
    modeling_gigachat3_5.py       sha256 58739e5eb8846feeefacb15132ea27f12cd70e9ca981e3c959c8837737326f7d

TRANSCRIBED (modeling_gigachat3_5.py line ranges):
    43-83      GigaChat35GatedDeltaRMSNorm (the pure-torch FusedRMSNormGated fallback: silu / sigmoid, zero-centered)
    87-104     GigaChat35RMSNorm
    107-124    GigaChat35ZeroCenteredRMSNorm (x * rsqrt(mean x^2 + eps) * (1 + w))
    127-170    GigaChat35ZeroCenteredGatedNorm / GigaChat35GatedNorm (rank-16 silu self-gate, sigmoid, x weight)
    311-394    l2norm, torch_chunk_gated_delta_rule (the CPU path: fla absent)
    439-441    _apply_sigmoid_gate_fp32
    456-474    _build_gated_delta_norm (the FusedRMSNormGated-absent branch)
    477-753    GigaChat35GatedDeltaNetWrapper (Qwen3-Next qkvz layout, silu short conv, beta / g, the four
               linear_gating_type output gates; no cache, no padding mask)
    773-878    rotate_half, apply_rotary_pos_emb, repeat_kv, eager_attention_forward, apply_rotary_pos_emb_interleave
    881-908    yarn_get_mscale, resolve_norm_class, build_norm
    911-1060   GigaChat35Attention (MLA: q / kv latent norms, alpha_q / alpha_kv scaling, interleaved RoPE,
               mscale^2 softmax scale, sigmoid output gate; eager)
    1063-1107  GigaChat35TopkRouter (sigmoid scores, group-limited top-k on scores + e_score_correction_bias,
               norm_topk_prob, routed_scaling_factor)
    1110-1137  GigaChat35MLP (hard SwiGLU clip: gate <= L, |up| <= L)
    1140-1195  GigaChat35MoE (+ shared expert, optional shared-expert sigmoid)
    1203-1293  GigaChat35DecoderLayer (pre / post / pre_post sandwich norms)
    1296-1329  GigaChat35RotaryEmbedding (ROPE_INIT_FUNCTIONS: the stock transformers yarn init, library code)
    1356-1449  GigaChat35Model (causal mask, no padding)
    1453-1529  GigaChat35ForCausalLM (lm_head, optional final logit softcapping)
configuration_gigachat3_5.py 6-213: GigaChat35Config (qk_head_dim, head_dim = qk_rope_head_dim, layer_types from
full_attention_layers).

DTYPE: the remote code rounds to fp32 in many places (.to(float32), .float(), .type(float32), softmax(dtype=
float32), the chunked delta rule's fp32 cast). Every such cast is lifted here: the reference computes in the
parameters' dtype (fp64). The rotary inv_freq table is the stock yarn init's (fp32), read in fp64.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class GigaChat35Config(PretrainedConfig):                                 # configuration_gigachat3_5.py 6-213
    model_type = "gigachat3_5"

    def __init__(self, vocab_size=129280, hidden_size=7168, intermediate_size=18432, moe_intermediate_size=2048,
                 num_hidden_layers=61, num_attention_heads=128, num_key_value_heads=128, n_shared_experts=1,
                 n_routed_experts=256, routed_scaling_factor=2.5, kv_lora_rank=512, q_lora_rank=1536,
                 qk_rope_head_dim=64, v_head_dim=128, qk_nope_head_dim=128, n_group=8, topk_group=4,
                 num_experts_per_tok=8, first_k_dense_replace=3, norm_topk_prob=True, hidden_act="silu",
                 max_position_embeddings=4096, initializer_range=0.02, rms_norm_eps=1e-20, pad_token_id=None,
                 bos_token_id=0, eos_token_id=1, tie_word_embeddings=False, rope_theta=10000.0, rope_scaling=None,
                 rope_interleave=True, attention_bias=False, attention_dropout=0.0, norm_type="LlamaRMSNorm",
                 layernorm_type="pre", layernorm_gating_weight=2.0, gated_attention=False,
                 use_shared_expert_sigmoid=False, use_mla_scaling_factor=False, linear_attention_type=None,
                 full_attention_layers=None, linear_key_head_dim=128, linear_value_head_dim=128,
                 linear_conv_kernel_dim=4, linear_num_key_heads=16, linear_num_value_heads=32,
                 linear_gating_type="gated_rmsnorm", linear_sigmoid_gate_scale=1.0, linear_attn_o_norm_eps=None,
                 swiglu_limit=0.0, final_logit_softcapping=None, **kwargs):
        layernorm_type = layernorm_type or "pre"
        assert layernorm_type in {"pre", "post", "pre_post"}
        self.vocab_size, self.max_position_embeddings = vocab_size, max_position_embeddings
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.moe_intermediate_size, self.num_hidden_layers = moe_intermediate_size, num_hidden_layers
        self.num_attention_heads, self.n_shared_experts = num_attention_heads, n_shared_experts
        self.n_routed_experts, self.routed_scaling_factor = n_routed_experts, routed_scaling_factor
        self.kv_lora_rank, self.q_lora_rank = kv_lora_rank, q_lora_rank
        self.qk_rope_head_dim, self.v_head_dim, self.qk_nope_head_dim = qk_rope_head_dim, v_head_dim, qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group, self.topk_group, self.num_experts_per_tok = n_group, topk_group, num_experts_per_tok
        self.first_k_dense_replace, self.norm_topk_prob = first_k_dense_replace, norm_topk_prob
        self.rope_interleave = rope_interleave
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act, self.initializer_range, self.rms_norm_eps = hidden_act, initializer_range, rms_norm_eps
        self.rope_theta, self.rope_scaling = rope_theta, rope_scaling
        self.attention_bias, self.attention_dropout = attention_bias, attention_dropout
        self.norm_type, self.layernorm_type, self.layernorm_gating_weight = norm_type, layernorm_type, layernorm_gating_weight
        self.gated_attention, self.use_shared_expert_sigmoid = gated_attention, use_shared_expert_sigmoid
        self.use_mla_scaling_factor = use_mla_scaling_factor
        assert linear_attention_type in (None, "GigaChat35GatedDeltaNet")
        self.linear_attention_type = linear_attention_type
        self.full_attention_layers = full_attention_layers or []
        if linear_attention_type is not None:
            self.layer_types = ["full_attention" if i in self.full_attention_layers else "linear_attention"
                                for i in range(num_hidden_layers)]
        else:
            self.full_attention_layers = list(range(num_hidden_layers))
            self.layer_types = ["full_attention"] * num_hidden_layers
        kwargs.pop("layer_types", None)
        self.linear_key_head_dim, self.linear_value_head_dim = linear_key_head_dim, linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads, self.linear_num_value_heads = linear_num_key_heads, linear_num_value_heads
        assert linear_gating_type in ("gated_rmsnorm", "gated_rmsnorm_sigmoid",
                                      "gated_rmsnorm_sigmoid_zero_centered", "sigmoid_gate")
        self.linear_gating_type, self.linear_sigmoid_gate_scale = linear_gating_type, linear_sigmoid_gate_scale
        self.linear_attn_o_norm_eps = linear_attn_o_norm_eps
        self.swiglu_limit, self.final_logit_softcapping = swiglu_limit, final_logit_softcapping
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)


def _act(name):
    return {"silu": F.silu}[name]


class GigaChat35GatedDeltaRMSNorm(nn.Module):                             # 43-83
    def __init__(self, hidden_size, eps=1e-6, activation="silu", zero_centered=False):
        super().__init__()
        assert activation in ("silu", "sigmoid")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon, self.activation, self.zero_centered = eps, activation, zero_centered

    def forward(self, h, gate=None):
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        w = 1.0 + self.weight if self.zero_centered else self.weight
        h = w * h
        if gate is not None:
            h = h * (F.silu(gate) if self.activation == "silu" else torch.sigmoid(gate))
        return h


class GigaChat35RMSNorm(nn.Module):                                       # 87-104
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, h):
        return self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))


class GigaChat35ZeroCenteredRMSNorm(nn.Module):                           # 107-124
    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight)


class _SelfGate:
    """127-170: the rank-16 silu -> sigmoid self-gate of GigaChat35ZeroCenteredGatedNorm / GigaChat35GatedNorm."""

    def _gate_init(self, hidden_size, layernorm_gating_weight):
        self.layernorm_gating_weight = layernorm_gating_weight
        self.gate_up_projection = nn.Linear(hidden_size, 16, bias=False)
        self.gate_down_projection = nn.Linear(16, hidden_size, bias=False)

    def _gate(self, h):
        g = self.gate_down_projection(F.silu(self.gate_up_projection(h)))
        return h * self.layernorm_gating_weight * torch.sigmoid(g)


class GigaChat35ZeroCenteredGatedNorm(GigaChat35ZeroCenteredRMSNorm, _SelfGate):
    def __init__(self, hidden_size, eps=1e-6, layernorm_gating_weight=2.0, **kwargs):
        super().__init__(hidden_size, eps=eps)
        self._gate_init(hidden_size, layernorm_gating_weight)

    def forward(self, h):
        return self._gate(super().forward(h))


class GigaChat35GatedNorm(GigaChat35RMSNorm, _SelfGate):
    def __init__(self, hidden_size, eps=1e-6, layernorm_gating_weight=2.0, **kwargs):
        super().__init__(hidden_size, eps=eps)
        self._gate_init(hidden_size, layernorm_gating_weight)

    def forward(self, h):
        return self._gate(super().forward(h))


def build_norm(config, hidden_size):                                     # 887-908
    cls = {"LlamaRMSNorm": GigaChat35RMSNorm, "ZeroCenteredRMSNorm": GigaChat35ZeroCenteredRMSNorm,
           "ZeroCenteredGatedNorm": GigaChat35ZeroCenteredGatedNorm, "GatedNorm": GigaChat35GatedNorm}[
        getattr(config, "norm_type", "LlamaRMSNorm")]
    return cls(hidden_size, eps=config.rms_norm_eps, layernorm_gating_weight=getattr(config, "layernorm_gating_weight", 2.0))


def l2norm(x, dim=-1, eps=1e-6):                                          # 311-314
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, use_qk_l2norm_in_kernel=False):
    """317-394 (initial_state None, no final state), the fp32 cast lifted."""
    if use_qk_l2norm_in_kernel:
        query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)]
    B, H, T, dk = key.shape
    dv = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    TT = T + pad
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row, sub = attn[..., i, :i].clone(), attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    S = torch.zeros(B, H, dk, dv, dtype=value.dtype)
    out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    for i in range(TT // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ S
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ S + a @ v_new
        S = S * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :T]
    return out.transpose(1, 2).contiguous()


class GigaChat35GatedDeltaNetWrapper(nn.Module):                          # 477-753
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads, self.num_k_heads = config.linear_num_value_heads, config.linear_num_key_heads
        self.head_k_dim, self.head_v_dim = config.linear_key_head_dim, config.linear_value_head_dim
        self.key_dim, self.value_dim = self.head_k_dim * self.num_k_heads, self.head_v_dim * self.num_v_heads
        self.conv_kernel_size, self.layer_idx = config.linear_conv_kernel_dim, layer_idx
        self.activation = config.hidden_act
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=self.conv_kernel_size,
                                groups=self.conv_dim, padding=self.conv_kernel_size - 1)
        self.in_proj_qkvz = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, self.num_v_heads * 2, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.linear_gating_type = getattr(config, "linear_gating_type", "gated_rmsnorm")
        self.linear_sigmoid_gate_scale = getattr(config, "linear_sigmoid_gate_scale", 1.0)
        eps = getattr(config, "linear_attn_o_norm_eps", None)
        eps = config.rms_norm_eps if eps is None else eps
        gt = self.linear_gating_type
        self.norm = (None if gt == "sigmoid_gate" else
                     GigaChat35GatedDeltaRMSNorm(self.head_v_dim, eps, activation="silu" if gt == "gated_rmsnorm" else "sigmoid",
                                                 zero_centered=gt == "gated_rmsnorm_sigmoid_zero_centered"))

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):         # 589-616, verbatim
        r = self.num_v_heads // self.num_k_heads
        mixed_qkvz = mixed_qkvz.view(*mixed_qkvz.size()[:-1], self.num_k_heads,
                                     2 * self.head_k_dim + 2 * self.head_v_dim * r)
        mixed_ba = mixed_ba.view(*mixed_ba.size()[:-1], self.num_k_heads, 2 * r)
        query, key, value, z = torch.split(mixed_qkvz, [self.head_k_dim, self.head_k_dim, r * self.head_v_dim,
                                                        r * self.head_v_dim], dim=3)
        b, a = torch.split(mixed_ba, [r, r], dim=3)
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(self, h):
        B, T, _ = h.shape
        q, k, v, z, b, a = self.fix_query_key_value_ordering(self.in_proj_qkvz(h), self.in_proj_ba(h))
        q, k, v = (x.reshape(x.shape[0], x.shape[1], -1) for x in (q, k, v))
        qkv = F.silu(self.conv1d(torch.cat((q, k, v), dim=-1).transpose(1, 2))[:, :, :T]).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, -1, self.head_k_dim)
        k = k.reshape(B, T, -1, self.head_k_dim)
        v = v.reshape(B, T, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        r = self.num_v_heads // self.num_k_heads
        if r > 1:
            q, k = q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)
        o = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        return self.out_proj(self._apply_output_gate(o, z))

    def _apply_output_gate(self, o, z):                                    # 729-753
        b, t = o.shape[:2]
        if self.linear_gating_type == "gated_rmsnorm":
            out = self.norm(o.reshape(-1, o.shape[-1]), z.reshape(-1, z.shape[-1]))
        elif self.linear_gating_type in ("gated_rmsnorm_sigmoid", "gated_rmsnorm_sigmoid_zero_centered"):
            out = self.norm(o.reshape(-1, o.shape[-1]), z.reshape(-1, z.shape[-1]))
            if self.linear_sigmoid_gate_scale != 1.0:
                out = out * self.linear_sigmoid_gate_scale
        else:
            out = o.reshape(b, t, -1) * torch.sigmoid(z.reshape(b, t, -1) * self.linear_sigmoid_gate_scale)
        return out.reshape(b, t, -1)


def rotate_half(x):                                                       # 773-777
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):                                 # 780-804
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def apply_rotary_pos_emb_interleave(q, k, cos, sin):                      # 845-878
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def yarn_get_mscale(scale=1, mscale=1):                                   # 881-884
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


class GigaChat35Attention(nn.Module):                                     # 911-1060
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.q_lora_rank, self.qk_rope_head_dim = config.q_lora_rank, config.qk_rope_head_dim
        self.kv_lora_rank, self.v_head_dim = config.kv_lora_rank, config.v_head_dim
        self.qk_nope_head_dim, self.qk_head_dim = config.qk_nope_head_dim, config.qk_head_dim
        d = config.hidden_size
        if getattr(config, "use_mla_scaling_factor", False):
            self.alpha_q = math.sqrt(d / (d if self.q_lora_rank is None else self.q_lora_rank))
            self.alpha_kv = math.sqrt(d / (d if self.kv_lora_rank is None else self.kv_lora_rank))
        else:
            self.alpha_q = self.alpha_kv = 1.0
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(d, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(d, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = build_norm(config, config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias)
        self.kv_a_layernorm = build_norm(config, self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, d, bias=config.attention_bias)
        self.scaling = self.qk_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            mad = config.rope_scaling.get("mscale_all_dim", 0)
            if mad:
                ms = yarn_get_mscale(config.rope_scaling["factor"], mad)
                self.scaling = self.scaling * ms * ms
        self.gated_attention = getattr(config, "gated_attention", False)
        if self.gated_attention:
            self.gate_proj = nn.Linear(d, self.num_heads * self.v_head_dim, bias=False)

    def forward(self, h, position_embeddings, mask):
        B, T = h.shape[:-1]
        if self.q_lora_rank is None:
            q = self.alpha_q * self.q_proj(h)
        else:
            q = self.q_b_proj(self.alpha_q * self.q_a_layernorm(self.q_a_proj(h)))
        q = q.view(B, T, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pass, k_rot = torch.split(self.kv_a_proj_with_mqa(h), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.alpha_kv * self.kv_a_layernorm(k_pass))
        k_pass = k_pass.view(B, T, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_pass, v = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.view(B, 1, T, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        rope = apply_rotary_pos_emb_interleave if self.config.rope_interleave else apply_rotary_pos_emb
        q_rot, k_rot = rope(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        q, k = torch.cat((q_pass, q_rot), dim=-1), torch.cat((k_pass, k_rot), dim=-1)
        k = k.repeat_interleave(self.num_key_value_groups, 1)               # repeat_kv (807-816)
        v = v.repeat_interleave(self.num_key_value_groups, 1)
        w = (q @ k.transpose(2, 3) * self.scaling + mask).softmax(-1)      # eager_attention_forward (819-842)
        o = (w @ v).transpose(1, 2).reshape(B, T, -1)
        if self.gated_attention:
            o = o * torch.sigmoid(self.gate_proj(h))                       # _apply_sigmoid_gate_fp32 (439-441)
        return self.o_proj(o)


class GigaChat35TopkRouter(nn.Module):                                    # 1063-1107
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k, self.n_routed_experts = config.num_experts_per_tok, config.n_routed_experts
        self.routed_scaling_factor, self.n_group, self.topk_group = config.routed_scaling_factor, config.n_group, config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.randn(self.n_routed_experts, config.hidden_size) * config.initializer_range)
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def get_topk_indices(self, scores):
        n, g = self.n_routed_experts, self.n_group
        sc = scores.view(-1, n) + self.e_score_correction_bias.unsqueeze(0)
        gs = sc.view(-1, g, n // g).topk(2, dim=-1)[0].sum(dim=-1)
        gi = torch.topk(gs, k=self.topk_group, dim=-1, sorted=False)[1]
        gm = torch.zeros_like(gs).scatter_(1, gi, 1)
        sm = gm.unsqueeze(-1).expand(-1, g, n // g).reshape(-1, n)
        return torch.topk(sc.masked_fill(~sm.bool(), 0.0), k=self.top_k, dim=-1, sorted=False)[1]

    def forward(self, h):
        scores = F.linear(h.view(-1, self.config.hidden_size), self.weight).sigmoid()
        idx = self.get_topk_indices(scores)
        w = scores.gather(1, idx)
        if self.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return idx, w * self.routed_scaling_factor


class GigaChat35MLP(nn.Module):                                           # 1110-1137
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        hs = config.hidden_size if hidden_size is None else hidden_size
        ff = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hs, ff, bias=False)
        self.up_proj = nn.Linear(hs, ff, bias=False)
        self.down_proj = nn.Linear(ff, hs, bias=False)
        self.act_fn = _act(config.hidden_act)

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        L = float(getattr(self.config, "swiglu_limit", 0.0))
        if L > 0:
            gate, up = gate.clamp(max=L), up.clamp(-L, L)
        return self.down_proj(self.act_fn(gate) * up)


class GigaChat35MoE(nn.Module):                                           # 1140-1195
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(GigaChat35MLP(config, intermediate_size=config.moe_intermediate_size)
                                     for _ in range(config.n_routed_experts))
        self.gate = GigaChat35TopkRouter(config)
        self.shared_experts = GigaChat35MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)
        self.use_shared_expert_sigmoid = bool(getattr(config, "use_shared_expert_sigmoid", False))
        if self.use_shared_expert_sigmoid:
            self.shared_expert_sigmoid = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, h):
        shape = h.shape
        idx, w = self.gate(h)
        x = h.view(-1, shape[-1])
        out = torch.zeros_like(x)
        mask = F.one_hot(idx, num_classes=len(self.experts)).permute(2, 0, 1)
        for e, ex in enumerate(self.experts):
            tok, wi = torch.where(mask[e])
            if tok.numel() > 0:
                out.index_add_(0, tok, ex(x[tok]) * w[tok, wi].unsqueeze(-1))
        sh = self.shared_experts(h)
        if self.use_shared_expert_sigmoid:
            sh = sh * torch.sigmoid(self.shared_expert_sigmoid(h))
        return out.view(*shape) + sh


class GigaChat35DecoderLayer(nn.Module):                                  # 1203-1293
    def __init__(self, config, layer_idx):
        super().__init__()
        self.mlp = GigaChat35MoE(config) if layer_idx >= config.first_k_dense_replace else GigaChat35MLP(config)
        lt = getattr(config, "layernorm_type", "pre")
        self._use_pre_layernorm, self._use_post_layernorm = lt in {"pre", "pre_post"}, lt in {"post", "pre_post"}
        if self._use_pre_layernorm:
            self.input_layernorm = build_norm(config, config.hidden_size)
            self.post_attention_layernorm = build_norm(config, config.hidden_size)
        else:
            self.input_layernorm = GigaChat35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = GigaChat35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self._use_post_layernorm:
            self.post_self_attn_layernorm = build_norm(config, config.hidden_size)
            self.post_feedforward_layernorm = build_norm(config, config.hidden_size)
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = (GigaChat35Attention(config, layer_idx) if layer_idx in config.full_attention_layers
                          else GigaChat35GatedDeltaNetWrapper(config, layer_idx))

    def forward(self, h, position_embeddings, mask):
        r = h
        if self._use_pre_layernorm:
            h = self.input_layernorm(h)
        h = self.self_attn(h) if self.layer_type == "linear_attention" else self.self_attn(h, position_embeddings, mask)
        if self._use_post_layernorm:
            h = self.post_self_attn_layernorm(h)
        h = r + h
        r = h
        if self._use_pre_layernorm:
            h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        if self._use_post_layernorm:
            h = self.post_feedforward_layernorm(h)
        return r + h


class GigaChat35RotaryEmbedding(nn.Module):                               # 1296-1329
    def __init__(self, config):
        super().__init__()
        rs = config.rope_scaling
        self.rope_type = rs.get("rope_type", rs.get("type")) if isinstance(rs, dict) else "default"
        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](config, None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        freqs = (self.inv_freq.to(x.dtype)[None, :, None] @ position_ids[:, None, :].to(x.dtype)).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling


class GigaChat35Model(nn.Module):                                         # 1356-1449
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(GigaChat35DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = build_norm(config, config.hidden_size)
        self.rotary_emb = GigaChat35RotaryEmbedding(config)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        T = input_ids.shape[1]
        pe = self.rotary_emb(h, torch.arange(T)[None])
        mask = torch.full((T, T), float("-inf"), dtype=h.dtype).triu(1)[None, None]   # causal (create_causal_mask)
        for layer in self.layers[: self.config.num_hidden_layers]:
            h = layer(h, pe, mask)
        return self.norm(h)


class GigaChat35ForCausalLM(nn.Module):                                   # 1453-1529
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GigaChat35Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        logits = self.lm_head(self.model(input_ids))
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits
