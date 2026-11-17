"""Motif-3 (MotifForCausalLM), a labelled reference transcription in plain torch at fp64.

SOURCE (read as text, never executed):
    repo      Motif-Technologies/Motif-3
    revision  2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a
    configuration_motif.py  sha256 cd6ee031056f20293d9cc2dc17dafc56479c27e62eac81d88c0d5aede42614d5
    modeling_motif.py       sha256 de9c73fd82712da3d7cb0951b8c1e2ad0f02fda5370fe2fb95ae5383a65c3b00

TRANSCRIBED (modeling_motif.py line ranges):
    49-87      PolyNormTorch (the torch PolyNorm; the hub-kernel PolyNorm is not used: `kernels` absent)
    89-136     GroupedPolyNorm (per-expert coefficients, bias clamp, hidden clamp, result clamp, output scale)
    139-156    MotifRMSNorm
    159-267    MHCLayer (manifold-constrained hyper-connections, Sinkhorn-Knopp on clamp(-20, 20).exp())
    270-306    _compute_yarn_inv_freq
    309-416    MotifRotaryEmbedding (default / yarn inv_freq in fp32, attention_scaling 1)
    419-470    rotate_half (roll form), apply_rotary_pos_emb_single
    473-501    MotifMLP (PolyNorm-GLU: clamp gate / up, forward_mul, output scale)
    534-775    MotifGDLAttention (diff_v2: q / kv LoRA latents, shared rope key, signal - sigmoid(lambda) noise,
               elementwise sigmoid output gate, mscale^2 on full layers when max_pos > original_seq_len,
               sliding window = config.sliding_window + 1 on interleaved layers with their own plain RoPE)
    778-879    TokenChoiceTopKRouter (sigmoid / softmax scores, top-k on scores + expert_bias, route_norm, route_scale)
    882-951    MotifExperts (eager per-expert loop, fused gate_up)
    954-1049   MoE (+ shared MotifMLP; score_before_experts False only)
    1052-1257  MotifDecoderLayer (forward and _forward_with_mhc)
    1289-1453  MotifModel (embed, rotary, MHC expand -> layers -> stream mean -> norm)
    1587-1690  MotifForCausalLM (lm_head)
configuration_motif.py 8-300: MotifConfig (the fields above and their defaults).

The attention kernel: Motif requires attn_implementation="flash_attention_2"; its semantics (transformers 4.57
modeling_flash_attention_utils.py 506-512: causal, GQA head h -> kv head h // (H / Hk), window_size =
(sliding_window - 1, sliding_window - 1) when the key length exceeds sliding_window) are written out here.

DTYPE: the remote code rounds to fp32 in many places (.float(), .to(float32), softmax(dtype=float32),
logits.float()). Every such cast is lifted: this reference computes in the parameters' dtype (fp64) throughout.
The rotary inv_freq tables stay as the source builds them (fp32 buffers); they are read in fp64.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


class MotifConfig:
    """configuration_motif.py 131-300 (only the fields the forward reads; unknown kwargs become attributes,
    as PretrainedConfig does: hidden_clamp, polynorm_*, mhc_h_post_alpha_end)."""

    def __init__(self, vocab_size=151936, hidden_size=4096, intermediate_size=22016, num_hidden_layers=32,
                 num_attention_heads=32, num_key_value_heads=32, hidden_act="silu", max_position_embeddings=32768,
                 rms_norm_eps=1e-6, tie_word_embeddings=False, rope_theta=1000000.0, rope_scaling=None,
                 use_sliding_window=False, sliding_window=4096, sliding_window_pattern="interleave",
                 sliding_window_period=2, attention_dropout=0.0, head_dim=None, num_noise_heads=0, num_experts=0,
                 experts_top_k=2, num_shared_experts=0, interleave_moe_layer_step=0, moe_intermediate_size=None,
                 score_func="softmax", route_norm=False, route_scale=1.0, load_balance_coeff=None,
                 score_before_experts=False, _debug_force_load_balance=False, mhc_enabled=False,
                 mhc_expansion_rate=4, mhc_identity_init=False, mhc_sinkhorn_iters=20, diff_v2=False,
                 attention_cls="basic", q_lora_rank=0, kv_lora_rank=0, qk_rope_head_dim=None, v_head_dim=None,
                 original_seq_len=32768, rope_factor=1.0, mscale=1.0, swa_rope_theta=None,
                 headwise_attn_output_gate=False, elementwise_attn_output_gate=False, n_dense_first_layers=0,
                 **kwargs):
        self.vocab_size, self.max_position_embeddings = vocab_size, max_position_embeddings
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.num_hidden_layers, self.num_attention_heads = num_hidden_layers, num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.sliding_window_pattern, self.sliding_window_period = sliding_window_pattern, sliding_window_period
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act, self.rms_norm_eps = hidden_act, rms_norm_eps
        self.rope_theta, self.rope_scaling, self.attention_dropout = rope_theta, rope_scaling, attention_dropout
        self.head_dim, self.num_noise_heads = head_dim, num_noise_heads
        self.num_experts, self.experts_top_k, self.num_shared_experts = num_experts, experts_top_k, num_shared_experts
        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.score_func, self.route_norm, self.route_scale = score_func, route_norm, route_scale
        self.load_balance_coeff, self.score_before_experts = load_balance_coeff, score_before_experts
        self._debug_force_load_balance = _debug_force_load_balance
        self.mhc_enabled, self.mhc_expansion_rate = mhc_enabled, mhc_expansion_rate
        self.mhc_identity_init, self.mhc_sinkhorn_iters = mhc_identity_init, mhc_sinkhorn_iters
        self.diff_v2, self.attention_cls = diff_v2, attention_cls
        self.q_lora_rank, self.kv_lora_rank = q_lora_rank, kv_lora_rank
        self.qk_rope_head_dim, self.v_head_dim = qk_rope_head_dim, v_head_dim
        self.original_seq_len, self.rope_factor, self.mscale = original_seq_len, rope_factor, mscale
        self.swa_rope_theta = swa_rope_theta
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        self.n_dense_first_layers = n_dense_first_layers
        self.tie_word_embeddings = tie_word_embeddings
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        if self.rope_scaling is not None and self.rope_scaling.get("rope_type") == "yarn":
            self.rope_scaling.setdefault("attention_factor", 1.0)
            self.rope_scaling.setdefault("apply_yarn_scaling", False)
        for k, v in kwargs.items():
            setattr(self, k, v)


class PolyNormTorch(nn.Module):                                           # 49-87
    def __init__(self, eps=1e-6, sigmoid_weight=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3) / 3)
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps
        self.sigmoid_weight = sigmoid_weight

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def _poly(self, x):
        w = torch.sigmoid(self.weight) if self.sigmoid_weight else self.weight
        return w[0] * self._norm(x ** 3) + w[1] * self._norm(x ** 2) + w[2] * self._norm(x) + self.bias

    def forward(self, x):
        return self._poly(x)

    def forward_mul(self, x, mul):
        return self._poly(x) * mul


class GroupedPolyNorm(nn.Module):                                         # 89-136
    def __init__(self, num_experts, eps=1e-6, sigmoid_weight=True, bias_clamp=None, output_scale=1.0,
                 hidden_clamp=None):
        super().__init__()
        self.num_experts, self.eps, self.sigmoid_weight = num_experts, eps, sigmoid_weight
        self.bias_clamp, self.output_scale, self.hidden_clamp = bias_clamp, output_scale, hidden_clamp
        self.weight = nn.Parameter(torch.ones(num_experts, 3) / 3)
        self.bias = nn.Parameter(torch.zeros(num_experts, 1))

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_single(self, x, mul, expert_idx):
        w = self.weight[expert_idx]
        if self.sigmoid_weight:
            w = torch.sigmoid(w)
        b = self.bias[expert_idx]
        if self.bias_clamp is not None:
            b = b.clamp(-self.bias_clamp, self.bias_clamp)
        xf, mf = x, mul
        if self.hidden_clamp is not None:
            xf = xf.clamp(-self.hidden_clamp, self.hidden_clamp)
            mf = mf.clamp(-self.hidden_clamp, self.hidden_clamp)
        poly = w[0] * self._norm(xf ** 3) + w[1] * self._norm(xf ** 2) + w[2] * self._norm(xf) + b
        result = poly * mf
        if self.hidden_clamp is not None:
            result = result.clamp(-self.hidden_clamp, self.hidden_clamp)
        return result * self.output_scale


class MotifRMSNorm(nn.Module):                                            # 139-156
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, h):
        return self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))


class MHCLayer(nn.Module):                                                # 159-267
    def __init__(self, expansion_rate, num_dim, sinkhorn_iters=20, h_post_coeff=2.0):
        super().__init__()
        self.expansion_rate, self.num_dim, self.sinkhorn_iters = expansion_rate, num_dim, sinkhorn_iters
        self.h_post_coeff = float(h_post_coeff)
        E, D = expansion_rate, num_dim
        self.proj_pre = nn.Linear(E * D, E, bias=False)
        self.proj_post = nn.Linear(E * D, E, bias=False)
        self.proj_res = nn.Linear(E * D, E * E, bias=False)
        self.rms_norm = MotifRMSNorm(E * D, eps=1e-6)
        # _init_weights(identity_init=False), 199-224
        self.alpha_pre = nn.Parameter(torch.randn(1) * 0.1)
        self.alpha_post = nn.Parameter(torch.randn(1) * 0.1)
        self.alpha_res = nn.Parameter(torch.randn(1) * 0.1)
        for lin in (self.proj_pre, self.proj_post, self.proj_res):
            nn.init.xavier_uniform_(lin.weight)
        self.bias_pre = nn.Parameter(torch.zeros(E))
        self.bias_post = nn.Parameter(torch.zeros(E))
        self.bias_res = nn.Parameter(torch.randn(E, E) * 0.1)

    def _sinkhorn_knopp_batch(self, matrix):
        m = matrix.clamp(-20.0, 20.0).exp()
        for _ in range(self.sinkhorn_iters):
            m = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            m = m / m.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        return m

    def forward(self, x):
        B, S, E, D = x.shape
        xn = self.rms_norm(x.reshape(B, S, E * D))
        pre, post = self.proj_pre(xn), self.proj_post(xn)
        res = self.proj_res(xn).reshape(B, S, E, E)
        h_pre = torch.sigmoid((self.alpha_pre * pre + self.bias_pre).clamp(-10.0, 10.0))
        h_post = self.h_post_coeff * torch.sigmoid((self.alpha_post * post + self.bias_post).clamp(-10.0, 10.0))
        h_res = self._sinkhorn_knopp_batch(self.alpha_res * res + self.bias_res)
        return h_pre, h_post, h_res

    @staticmethod
    def apply_h_pre(x, h_pre):
        return (x * h_pre.unsqueeze(-1)).sum(dim=2)


def _compute_yarn_inv_freq(dim, end, theta, original_seq_len, rope_factor, beta_fast=32.0, beta_slow=1.0):
    """270-306, verbatim (fp32 tables, as the source builds them)."""
    def find_correction_dim(num_rotations, d, base, max_seq_len):
        return d * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, d, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, d, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, d, base, max_seq_len))
        return max(low, 0), min(high, d - 1)

    def linear_ramp_factor(lo, hi, d):
        if lo == hi:
            hi += 0.001
        return torch.clamp((torch.arange(d, dtype=torch.float32) - lo) / (hi - lo), 0, 1)

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if end > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, theta, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth
    return freqs


class MotifRotaryEmbedding(nn.Module):                                    # 309-416
    def __init__(self, config, rope_head_dim=None):
        super().__init__()
        rs = config.rope_scaling if isinstance(config.rope_scaling, dict) else None
        self.rope_type = rs.get("rope_type", rs.get("type")) if rs else "default"
        dim = rope_head_dim if rope_head_dim is not None else (
            config.head_dim if config.head_dim is not None else config.hidden_size // config.num_attention_heads)
        if self.rope_type == "default":
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        elif self.rope_type == "yarn":
            inv_freq = _compute_yarn_inv_freq(
                dim, config.max_position_embeddings, float(rs.get("rope_theta", config.rope_theta)),
                int(rs.get("original_max_position_embeddings",
                           getattr(config, "original_seq_len", config.max_position_embeddings))),
                float(rs.get("factor", getattr(config, "rope_factor", 1.0))),
                float(rs.get("beta_fast", 32)), float(rs.get("beta_slow", 1)))
        else:
            raise NotImplementedError(self.rope_type)                     # (ROPE_INIT_FUNCTIONS branch: not transcribed)
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)

    def forward(self, x, position_ids):
        freqs = (self.inv_freq.to(x.dtype)[None, :, None] @ position_ids[:, None, :].to(x.dtype)).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling


def rotate_half(x):                                                       # 419-434 (roll form)
    half = x.shape[-1] // 2
    r = torch.roll(x, shifts=-half, dims=-1)
    r[..., :half] *= -1
    return r


def apply_rotary_pos_emb_single(x, cos, sin):                             # 458-470
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    return x * cos + rotate_half(x) * sin


class MotifMLP(nn.Module):                                                # 473-501
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        assert config.hidden_act == "poly_norm"                           # (other activations: ACT2FN, not used)
        self.act_fn = PolyNormTorch(sigmoid_weight=getattr(config, "polynorm_sigmoid_weight", True))
        self.hidden_clamp = getattr(config, "hidden_clamp", None)
        self.polynorm_output_scale = float(getattr(config, "polynorm_output_scale", 1.0))

    def forward(self, h):
        gate, up = self.gate_proj(h), self.up_proj(h)
        if self.hidden_clamp is not None:
            gate = gate.clamp(-self.hidden_clamp, self.hidden_clamp)
            up = up.clamp(-self.hidden_clamp, self.hidden_clamp)
        h = self.act_fn.forward_mul(gate, up)
        if self.polynorm_output_scale != 1.0:
            h = h * self.polynorm_output_scale
        return self.down_proj(h)


def flash_attention(q, k, v, scaling, sliding_window):
    """flash_attention_2 semantics (see the header). q (B,H,T,D), k / v (B,Hk,T,D) -> (B,T,H,D)."""
    B, H, T, D = q.shape
    rep = H // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = q @ k.transpose(-1, -2) * scaling
    t, j = torch.arange(T)[:, None], torch.arange(T)[None]
    ok = j <= t
    if sliding_window is not None and T > sliding_window:
        ok = ok & (t - j <= sliding_window - 1)
    return (s.masked_fill(~ok, float("-inf")).softmax(-1) @ v).transpose(1, 2)


class MotifGDLAttention(nn.Module):                                       # 534-775
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads, self.num_key_value_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.head_dim if config.head_dim is not None else self.hidden_size // self.num_heads
        self.num_noise_heads = config.num_noise_heads
        self.grouped_ratio = (self.num_heads - self.num_noise_heads) // self.num_noise_heads
        self.n_signal_heads = self.grouped_ratio * self.num_noise_heads
        self.q_lora_rank, self.kv_lora_rank = config.q_lora_rank, config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim if config.qk_rope_head_dim is not None else self.head_dim // 2
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim if config.v_head_dim is not None else self.head_dim
        self.diff_v2 = getattr(config, "diff_v2", True)
        assert self.diff_v2                                              # (V1 raises in the source)
        self.sliding_window, is_swa = None, False
        if config.use_sliding_window and getattr(config, "sliding_window", None) is not None:
            pattern = getattr(config, "sliding_window_pattern", "interleave")
            period = getattr(config, "sliding_window_period", 2)
            if pattern == "all" or (pattern == "interleave" and layer_idx % period != 0):
                self.sliding_window, is_swa = config.sliding_window + 1, True
        self.is_swa_layer = is_swa
        self.scaling = self.head_dim ** -0.5
        if (not is_swa) and config.max_position_embeddings > getattr(config, "original_seq_len", 32768):
            mv = 0.1 * getattr(config, "mscale", 1.0) * math.log(getattr(config, "rope_factor", 1.0)) + 1.0
            self.scaling = self.scaling * mv * mv
        self.elementwise_attn_output_gate = getattr(config, "elementwise_attn_output_gate", False)
        self.headwise_attn_output_gate = getattr(config, "headwise_attn_output_gate", False)
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = MotifRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wq_b_gate = (nn.Linear(self.q_lora_rank, self.n_signal_heads * self.v_head_dim, bias=False)
                          if self.elementwise_attn_output_gate else None)
        self.wkv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = MotifRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = nn.Linear(self.kv_lora_rank,
                               self.num_key_value_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.lambda_proj = nn.Linear(self.hidden_size, self.n_signal_heads, bias=False)
        self.wo = nn.Linear(self.n_signal_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.swa_rotary_emb = None
        if is_swa and getattr(config, "swa_rope_theta", None) is not None:
            sc = copy.copy(config)
            sc.rope_scaling, sc.rope_theta = None, config.swa_rope_theta
            self.swa_rotary_emb = MotifRotaryEmbedding(sc, rope_head_dim=self.qk_rope_head_dim)

    def forward(self, h, position_ids, position_embeddings):
        B, T, _ = h.shape
        q_latent = self.q_norm(self.wq_a(h))
        q = self.wq_b(q_latent).view(B, T, self.num_heads, self.head_dim)
        gate_score = None
        if self.elementwise_attn_output_gate and self.wq_b_gate is not None:
            gate_score = self.wq_b_gate(q_latent).view(B, T, -1, self.v_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_latent, k_pe = torch.split(self.wkv_a(h), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        cos, sin = self.swa_rotary_emb(h, position_ids) if self.swa_rotary_emb is not None else position_embeddings
        q_pe = apply_rotary_pos_emb_single(q_pe, cos, sin)
        k_pe = apply_rotary_pos_emb_single(k_pe.unsqueeze(2), cos, sin)
        q_total = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv_latent)).view(B, T, self.num_key_value_heads,
                                                      self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_full = torch.cat([k_nope, k_pe.expand(-1, -1, self.num_key_value_heads, -1)], dim=-1)
        lambda_full = self.lambda_proj(h)
        v_t = F.pad(v.transpose(1, 2), [0, self.head_dim - self.v_head_dim])
        out = flash_attention(q_total.transpose(1, 2), k_full.transpose(1, 2), v_t, self.scaling,
                              self.sliding_window)[..., :self.v_head_dim]
        g, gs = self.num_noise_heads, self.grouped_ratio + 1
        r = out.reshape(B, T, g, gs, self.v_head_dim)
        attn1 = r[:, :, :, :self.grouped_ratio].reshape(B, T, -1, self.v_head_dim)
        attn2 = r[:, :, :, self.grouped_ratio:].reshape(B, T, g, self.v_head_dim)
        attn2 = torch.repeat_interleave(attn2, dim=2, repeats=self.grouped_ratio)
        o = attn1 - torch.sigmoid(lambda_full).unsqueeze(-1) * attn2
        if gate_score is not None:
            o = o * torch.sigmoid(gate_score)
        return self.wo(o.reshape(B, T, -1))


class TokenChoiceTopKRouter(nn.Module):                                   # 778-879
    def __init__(self, hidden_size, num_experts, experts_top_k, score_func, route_norm, route_scale,
                 _debug_force_load_balance=False):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts, self.experts_top_k, self.score_func = num_experts, experts_top_k, score_func
        self.route_norm, self.route_scale = route_norm, route_scale
        self._debug_force_load_balance = _debug_force_load_balance
        assert not _debug_force_load_balance                             # (debug round-robin: not transcribed)

    def forward(self, x, expert_bias=None):
        scores = self.gate(x)
        scores = torch.sigmoid(scores) if self.score_func == "sigmoid" else F.softmax(scores, dim=1)
        if expert_bias is not None:
            _, idx = torch.topk(scores + expert_bias, k=self.experts_top_k, dim=1)
            top = scores.gather(dim=1, index=idx)
        else:
            top, idx = torch.topk(scores, k=self.experts_top_k, dim=1)
        if self.route_norm:
            top = top / (top.sum(dim=-1, keepdim=True) + 1e-20)
        return top * self.route_scale, idx


class MotifExperts(nn.Module):                                            # 882-951
    def __init__(self, config):
        super().__init__()
        self.num_experts, self.hidden_size = config.num_experts, config.hidden_size
        self.intermediate_dim = getattr(config, "moe_intermediate_size", config.intermediate_size)
        self.gate_up_proj = nn.Parameter(torch.randn(self.num_experts, 2 * self.intermediate_dim, self.hidden_size) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(self.num_experts, self.hidden_size, self.intermediate_dim) * 0.02)
        assert config.hidden_act == "poly_norm"
        self.act_fn = GroupedPolyNorm(self.num_experts, sigmoid_weight=getattr(config, "polynorm_sigmoid_weight", True),
                                      bias_clamp=getattr(config, "polynorm_bias_clamp", None),
                                      output_scale=float(getattr(config, "polynorm_output_scale", 1.0)),
                                      hidden_clamp=getattr(config, "hidden_clamp", None))

    def forward(self, h, top_k_index, top_k_weights):
        out = torch.zeros_like(h)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            pos, tok = torch.where(mask[e])
            if tok.shape[0] == 0:
                continue
            gate, up = (h[tok] @ self.gate_up_proj[e].T).chunk(2, dim=-1)
            cur = self.act_fn.forward_single(gate, up, e) @ self.down_proj[e].T
            out.index_add_(0, tok, cur * top_k_weights[tok, pos, None])
        return out


class MoE(nn.Module):                                                     # 954-1049
    def __init__(self, config):
        super().__init__()
        self.num_experts, self.experts_top_k = config.num_experts, config.experts_top_k
        self.experts = MotifExperts(config)
        self.router = TokenChoiceTopKRouter(config.hidden_size, config.num_experts, config.experts_top_k,
                                            config.score_func, config.route_norm, config.route_scale,
                                            config._debug_force_load_balance)
        mi = getattr(config, "moe_intermediate_size", config.intermediate_size)
        self.shared_experts = MotifMLP(config, intermediate_size=mi) if config.num_shared_experts > 0 else None
        self.score_before_experts = config.score_before_experts
        assert not self.score_before_experts                             # (_score_before_forward: not transcribed)
        self.expert_bias = (nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)
                            if config.load_balance_coeff is not None else None)

    def forward(self, x):
        B, S, D = x.shape
        x = x.view(-1, D)
        top, idx = self.router(x, self.expert_bias)
        out = self.experts(x, idx, top)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out.reshape(B, S, D), None


class MotifDecoderLayer(nn.Module):                                       # 1052-1257
    def __init__(self, config, layer_idx):
        super().__init__()
        assert getattr(config, "attention_cls", "basic") == "gdla"
        self.self_attn = MotifGDLAttention(config, layer_idx)
        nd = getattr(config, "n_dense_first_layers", 0)
        st = config.interleave_moe_layer_step
        self.moe_enabled = (layer_idx >= nd and (layer_idx + 1) % st == 0) if st != 0 else False
        if self.moe_enabled:
            self.moe = MoE(config)
        else:
            self.mlp = MotifMLP(config)
        self.input_layernorm = MotifRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MotifRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mhc_enabled = getattr(config, "mhc_enabled", False)
        if self.mhc_enabled:
            assert not getattr(config, "mhc_identity_init", False)
            c = 1.0 + float(getattr(config, "mhc_h_post_alpha_end", 0.0))
            kw = dict(expansion_rate=config.mhc_expansion_rate, num_dim=config.hidden_size,
                      sinkhorn_iters=getattr(config, "mhc_sinkhorn_iters", 20), h_post_coeff=c)
            self.mhc_attn, self.mhc_ffn = MHCLayer(**kw), MHCLayer(**kw)

    def _ffn(self, h):
        return self.moe(h)[0] if self.moe_enabled else self.mlp(h)

    def forward(self, x, position_ids, position_embeddings):
        if not self.mhc_enabled:
            x = x + self.self_attn(self.input_layernorm(x), position_ids, position_embeddings)
            return x + self._ffn(self.post_attention_layernorm(x))
        hp, hq, hr = self.mhc_attn(x)
        a = self.self_attn(self.input_layernorm(MHCLayer.apply_h_pre(x, hp)), position_ids, position_embeddings)
        h = torch.einsum("bsij,bsjd->bsid", hr, x) + hq.unsqueeze(-1) * a.unsqueeze(2)
        hp, hq, hr = self.mhc_ffn(h)
        f = self._ffn(self.post_attention_layernorm(MHCLayer.apply_h_pre(h, hp)))
        return torch.einsum("bsij,bsjd->bsid", hr, h) + hq.unsqueeze(-1) * f.unsqueeze(2)


class MotifModel(nn.Module):                                              # 1289-1453
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(MotifDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = MotifRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MotifRotaryEmbedding(config, rope_head_dim=getattr(config, "qk_rope_head_dim", None))
        self.mhc_enabled = getattr(config, "mhc_enabled", False)
        self.mhc_expansion_rate = getattr(config, "mhc_expansion_rate", 4)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        pos = torch.arange(input_ids.shape[1])[None]
        pe = self.rotary_emb(h, pos)
        if self.mhc_enabled:
            h = h.unsqueeze(2).expand(-1, -1, self.mhc_expansion_rate, -1).contiguous()
        for layer in self.layers:
            h = layer(h, pos, pe)
        if self.mhc_enabled:
            h = h.mean(dim=2)
        return self.norm(h)


class MotifForCausalLM(nn.Module):                                        # 1587-1690
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MotifModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        assert not config.tie_word_embeddings

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))
