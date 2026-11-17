"""LFM3 (Liquid AI): a labelled reference transcription of the remote modelling code, read as text, never run.

Source:   LiquidAI/LFM3-1.7B-Base, snapshot 7bca5038823c0d0f14efe06439109354a10e014c
  modeling_lfm3.py       sha256 e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab
  configuration_lfm3.py  sha256 176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5
Transcribed (modeling_lfm3.py):
   51-69    Lfm3RMSNorm
   72-124   Lfm3RotaryEmbedding, the default rope_type: zero-tail partial RoPE (the first
            int(partial_rotary_factor * head_dim) // 2 frequencies of the FULL head-dim progression, the rest 0),
            one inv_freq buffer per attention layer type ("{layer_type}_inv_freq")
  149-157   Lfm3MLP (SwiGLU w2(silu(w1 x) * w3 x))
  160-210   rotate_half, repeat_kv, eager_attention_forward, apply_rotary_pos_emb
  213-273   Lfm3Attention: per-head q / k RMSNorm, GQA, RoPE, a per-HEAD sigmoid output gate (gate_proj), the
            sliding window on sliding_attention layers
  331-430   Lfm3ShortConv on its CPU path (width 7 > 4: no causal_conv1d kernel; no Triton off CUDA): y = C * conv(B * x),
            the depthwise conv `_conv` (427-430), no cache, no padding mask, no packed documents (seq_idx None)
  445-498   Lfm3DecoderLayer: pre-norm, sandwich norms (post_operator_norm / post_ffn_norm) when use_sandwich_norm
  501-530   create_masks_by_layer_type: causal for full_attention; causal + sliding window for sliding_attention,
            whose overlay (transformers 5.18 masking_utils.sliding_window_overlay) keeps kv_idx > q_idx - sliding_window
  742-765   Lfm3Model.forward with no cache / attention_mask / packing: embed, layers, embedding_norm (the final norm)
  803-860   Lfm3ForCausalLM: lm_head on the last hidden state
configuration_lfm3.py 56-98: the fields and defaults (head_dim, rope_parameters, layer_types) as a plain class.

Deviations, all numerical: the remote code casts RMSNorm, the RoPE tables / rotation and the softmax to fp32; here
every op runs at the input dtype (the fp32 casts lifted), so an fp64 run is fp64 throughout and the host must agree
to ~1e-15 relative, not to fp32 rounding. The inv_freq buffers are computed in fp64 for the same reason.
"""
import torch
import torch.nn.functional as F
from torch import nn


class Lfm3Config:
    """configuration_lfm3.py 56-98 (defaults and __post_init__), without the PreTrainedConfig machinery."""
    model_type = "lfm3"
    _attention_layer_types = ("full_attention", "sliding_attention")

    def __init__(self, vocab_size=128_000, hidden_size=2048, intermediate_size=7680, num_hidden_layers=24,
                 num_attention_heads=16, num_key_value_heads=4, max_position_embeddings=128_000, norm_eps=1e-5,
                 pad_token_id=124_893, bos_token_id=124_894, eos_token_id=124_900, tie_word_embeddings=True,
                 rope_parameters=None, conv_bias=False, conv_L_cache=7, layer_types=None, head_dim=None,
                 sliding_window=256, use_sandwich_norm=True, **extra):
        self.vocab_size, self.hidden_size, self.intermediate_size = vocab_size, hidden_size, intermediate_size
        self.num_hidden_layers, self.num_attention_heads = num_hidden_layers, num_attention_heads
        self.num_key_value_heads, self.max_position_embeddings, self.norm_eps = num_key_value_heads, max_position_embeddings, norm_eps
        self.pad_token_id, self.bos_token_id, self.eos_token_id = pad_token_id, bos_token_id, eos_token_id
        self.tie_word_embeddings, self.conv_bias, self.conv_L_cache = tie_word_embeddings, conv_bias, conv_L_cache
        self.sliding_window, self.use_sandwich_norm = sliding_window, use_sandwich_norm
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.rope_parameters = rope_parameters if rope_parameters is not None else {
            "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0}}
        self.layer_types = layer_types if layer_types is not None else [
            "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
            for i in range(num_hidden_layers)]
        for k, v in extra.items():
            setattr(self, k, v)
        assert len(self.layer_types) == num_hidden_layers


class Lfm3RMSNorm(nn.Module):                                       # 51-69 (fp32 cast lifted)
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))


class Lfm3RotaryEmbedding(nn.Module):                               # 72-124, rope_type "default"
    def __init__(self, config):
        super().__init__()
        self.layer_types = sorted(set(config.layer_types) & set(config.rope_parameters))
        dim = config.head_dim
        for lt in self.layer_types:
            p = config.rope_parameters[lt]
            assert p.get("rope_type", "default") == "default", "only the default rope_type is transcribed"
            angles = int(p.get("partial_rotary_factor", 1.0) * dim) // 2
            inv = 1.0 / (p["rope_theta"] ** (torch.arange(0, 2 * angles, 2, dtype=torch.float64) / dim))
            inv = torch.cat((inv, torch.zeros(dim // 2 - angles, dtype=torch.float64)))
            self.register_buffer(f"{lt}_inv_freq", inv, persistent=False)

    def forward(self, x, position_ids, layer_type):
        inv = getattr(self, f"{layer_type}_inv_freq").to(x.dtype)
        freqs = position_ids[..., None].to(x.dtype) * inv
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Lfm3MLP(nn.Module):                                           # 149-157
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rotate_half(x):                                                 # 160-164
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(h, n_rep):                                            # 167-176
    return h if n_rep == 1 else h.repeat_interleave(n_rep, dim=1)


def apply_rotary_pos_emb(q, k, cos, sin):                           # 204-210 (fp32 casts lifted)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def layer_mask(T, sliding_window, dtype):
    """501-530 + masking_utils: causal, and kv_idx > q_idx - sliding_window on sliding layers (additive)."""
    q = torch.arange(T)[:, None]
    kv = torch.arange(T)[None, :]
    keep = kv <= q
    if sliding_window is not None:
        keep = keep & (kv > q - sliding_window)
    return torch.zeros(T, T, dtype=dtype).masked_fill(~keep, float("-inf"))


class Lfm3Attention(nn.Module):                                     # 213-273
    def __init__(self, config, layer_idx, make_linear=None):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        q_dim, kv_dim = config.num_attention_heads * self.head_dim, config.num_key_value_heads * self.head_dim
        lin = make_linear or (lambda i, o, which: nn.Linear(i, o, bias=False))
        self.q_proj = lin(config.hidden_size, q_dim, "q")
        self.k_proj = lin(config.hidden_size, kv_dim, "k")
        self.v_proj = lin(config.hidden_size, kv_dim, "v")
        self.o_proj = lin(q_dim, config.hidden_size, "o")
        self.q_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(self, x, position_embeddings):
        B, T, _ = x.shape
        hs = (B, T, -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(*hs)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(*hs)).transpose(1, 2)
        v = self.v_proj(x).view(*hs).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        w = torch.matmul(q, k.transpose(2, 3)) * self.scaling + layer_mask(T, self.sliding_window, x.dtype)
        o = torch.matmul(torch.softmax(w, dim=-1), v).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)       # one gate per head
        return self.o_proj((o * gate).reshape(B, T, -1))


class Lfm3ShortConv(nn.Module):                                     # 331-430 (the torch `_conv` path)
    def __init__(self, config, layer_idx, make_in=None, make_out=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.conv_L_cache
        self.bias = config.conv_bias
        h = config.hidden_size
        self.conv = nn.Conv1d(h, h, kernel_size=self.conv_kernel_size, groups=h, bias=self.bias,
                              padding=self.conv_kernel_size - 1)
        self.in_proj = make_in() if make_in else nn.Linear(h, 3 * h, bias=self.bias)
        self.out_proj = make_out() if make_out else nn.Linear(h, h, bias=self.bias)

    def forward(self, x):
        T = x.shape[1]
        Bm, C, xx = self.in_proj(x).transpose(-1, -2).chunk(3, dim=-2)
        y = C * self.conv(Bm * xx)[..., :T]
        return self.out_proj(y.transpose(-1, -2).contiguous())


class Lfm3DecoderLayer(nn.Module):                                  # 445-498
    def __init__(self, config, layer_idx, attn_factory=None, conv_factory=None):
        super().__init__()
        self.layer_idx = layer_idx
        lt = config.layer_types[layer_idx]
        self.is_attention_layer = lt in config._attention_layer_types
        self.layer_type = lt if self.is_attention_layer else "conv"
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.is_attention_layer:
            self.self_attn = (attn_factory or Lfm3Attention)(config, layer_idx)
        else:
            self.conv = (conv_factory or Lfm3ShortConv)(config, layer_idx)
        self.feed_forward = Lfm3MLP(config)
        self.operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, h, pos_emb):
        r = h
        if self.is_attention_layer:
            h = self.self_attn(self.operator_norm(h), pos_emb[self.layer_type])
        else:
            h = self.conv(self.operator_norm(h))
        if self.use_sandwich_norm:
            h = self.post_operator_norm(h)
        h = h + r
        f = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich_norm:
            f = self.post_ffn_norm(f)
        return h + f


class Lfm3Model(nn.Module):                                         # 592-603, 742-765
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(Lfm3DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.rotary_emb = Lfm3RotaryEmbedding(config)
        self.embedding_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)

    def pos_emb(self, h):
        pos = torch.arange(h.shape[1])[None]
        return {lt: self.rotary_emb(h, pos, lt) for lt in self.rotary_emb.layer_types}

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        pe = self.pos_emb(h)
        for layer in self.layers:
            h = layer(h, pe)
        return self.embedding_norm(h)


class Lfm3ForCausalLM(nn.Module):                                   # 803-860
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Lfm3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:                              # _tied_weights_keys (804)
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))
