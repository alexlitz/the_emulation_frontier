"""REFERENCE transcription (not the remote code) of GPT-S3.

Source: AxiomicLabs/GPT-S3-3M, snapshot 21a3399bbe888bdf65d9e59eb2d9a79164b4ba16, read as text, never executed.
  configuration_gpts3.py  sha256 ab820e4894597010eee2028001aa7c2b71a8020f8e787e737de18f18a2e4f8a9
      GPTS3Config.__init__ / is_attention_layer                  l. 9-50
  modeling_gpts3.py       sha256 85d8bb19a4eaeb884753c64e4be09a2d52cb74df54120a897a0569f6302c26b4
      GDN2_CHUNK_SIZE / L2_NORM_EPS                               l. 22-23
      RMSNorm / OffsetRMSNorm / GatedRMSNorm                      l. 26-61
      precompute_rope_cos_sin / apply_rotary_emb                  l. 64-87  (interleaved pairs, partial)
      l2_normalize / gdn2_recurrent / gdn2_chunked                l. 90-157
      GatedDeltaNet2Kernel / GatedDeltaNet2                       l. 160-243
      GatedAttention                                              l. 246-299
      SwiGLUMLP / AttnResSources / AttnResAggregator / GPTS3Block l. 302-370
      GPTS3PreTrainedModel._init_weights                         l. 380-386
      GPTS3ForCausalLM.__init__ / _get_rope / forward             l. 389-499 (logits only)

Differences from the source, all deliberate:
  * every fp32 cast (.float(), dtype=torch.float32 in the RoPE table) is lifted to fp64 (.double() / float64);
  * no KV / recurrent cache, no attention_mask / padding_mask, no labels / loss, no generation plumbing
    (the forward is the use_cache=False, attention_mask=None path);
  * the config is a plain object with the same attribute names (the source subclasses PretrainedConfig);
    the model is a plain nn.Module with the source's module tree and parameter names
    (transformer["wte" / "h" / "ln_f" / "attn_res_f"], lm_head, block.is_attention, block.attn.kernel, ...),
    so elephantine.universal.from_gpts3 loads it exactly as it loaded the remote model.
The source imports torch / torch.nn.functional only (plus transformers' PreTrainedModel / DynamicCache plumbing).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

GDN2_CHUNK_SIZE = 64
L2_NORM_EPS = 1e-6
DT = torch.float64


class GPTS3Config:
    model_type = "gpts3_hybrid"

    def __init__(self, vocab_size=4096, hidden_size=144, num_hidden_layers=8, gdn_layers_per_attention=3,
                 gdn_head_dim=48, gdn_conv_size=4, gdn_norm_eps=1e-5, num_attention_heads=3, num_key_value_heads=1,
                 head_dim=48, rotary_dim=24, intermediate_size=352, max_position_embeddings=1024,
                 rope_theta=10000.0, rms_norm_eps=1e-6, attn_res_eps=1e-6, tie_word_embeddings=True, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.gdn_layers_per_attention = gdn_layers_per_attention
        self.gdn_head_dim = gdn_head_dim
        self.gdn_conv_size = gdn_conv_size
        self.gdn_norm_eps = gdn_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attn_res_eps = attn_res_eps
        self.tie_word_embeddings = tie_word_embeddings

    def is_attention_layer(self, layer_idx):
        return (layer_idx + 1) % (self.gdn_layers_per_attention + 1) == 0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.double() * rms).type_as(x) * self.weight


class OffsetRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + self.eps)
        return ((x.double() * rms) * (1.0 + self.weight.double())).type_as(x)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate):
        x = x.double()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight.double() * torch.sigmoid(gate.double())


def precompute_rope_cos_sin(rotary_dim, seq_len, theta, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=DT, device=device) / rotary_dim))
    positions = torch.arange(seq_len, dtype=DT, device=device)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(q, k, rope_cos, rope_sin):
    """Interleaved-pair RoPE on the first 2 * rope_cos.size(-1) channels only."""
    rotary_dim = rope_cos.size(-1) * 2
    cos = rope_cos[None, None]
    sin = rope_sin[None, None]

    def rotate(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_even, x_odd = x_rot.double().reshape(*x_rot.shape[:-1], -1, 2).unbind(-1)
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
        return torch.cat((rotated.type_as(x), x_pass), dim=-1)

    return rotate(q), rotate(k)


def l2_normalize(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + L2_NORM_EPS)


def gdn2_recurrent(q, k, v, g, b, w, scale, state):
    """S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T ;  o_t = S_t^T (scale * q_t)"""
    outputs = []
    for t in range(q.size(1)):
        state = state * g[:, t].exp().unsqueeze(-1)
        erase = ((b[:, t] * k[:, t]).unsqueeze(-1) * state).sum(-2)
        v_new = w[:, t] * v[:, t] - erase
        state = state + k[:, t].unsqueeze(-1) * v_new.unsqueeze(-2)
        outputs.append(((q[:, t] * scale).unsqueeze(-1) * state).sum(-2))
    return torch.stack(outputs, dim=1), state


def gdn2_chunked(q, k, v, g, b, w, scale, state, chunk_size=GDN2_CHUNK_SIZE):
    batch, length, heads, key_dim = k.shape
    value_dim = v.size(-1)
    pad = (-length) % chunk_size
    if pad:
        q, k, v, g, b, w = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v, g, b, w))
    num_chunks = q.size(1) // chunk_size

    def chunks(x):
        return x.transpose(1, 2).reshape(batch, heads, num_chunks, chunk_size, -1)

    q, k, v, g, b, w = (chunks(x) for x in (q, k, v, g, b, w))
    q = q * scale
    g_cum = g.cumsum(-2)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    strict = causal.tril(-1)

    outputs = []
    for n in range(num_chunks):
        q_n, k_n, v_n, b_n, w_n, g_n = (x[:, :, n] for x in (q, k, v, b, w, g_cum))
        decay = (g_n.unsqueeze(-2) - g_n.unsqueeze(-3)).masked_fill(~causal[:, :, None], float("-inf")).exp()
        a_kk = torch.einsum("bhik,bhjk,bhijk->bhij", b_n * k_n, k_n, decay)
        a_kk = a_kk.masked_fill(~strict, 0.0)
        a_qk = torch.einsum("bhik,bhjk,bhijk->bhij", q_n, k_n, decay)
        system = a_kk + torch.eye(chunk_size, device=q.device, dtype=q.dtype)
        u = torch.linalg.solve_triangular(system, w_n * v_n, upper=False, unitriangular=True)
        w_wy = torch.linalg.solve_triangular(system, b_n * k_n * g_n.exp(), upper=False, unitriangular=True)
        v_new = u - w_wy @ state
        outputs.append(a_qk @ v_new + (q_n * g_n.exp()) @ state)
        g_last = g_n[:, :, -1]
        state = state * g_last.exp().unsqueeze(-1) + (k_n * (g_last.unsqueeze(-2) - g_n).exp()).transpose(-1, -2) @ v_new

    o = torch.stack(outputs, dim=2).reshape(batch, heads, -1, value_dim)
    return o[:, :, :length].transpose(1, 2), state


class GatedDeltaNet2Kernel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.head_dim = config.gdn_head_dim
        self.num_heads = hidden // config.gdn_head_dim
        self.conv_size = config.gdn_conv_size
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_conv1d = nn.Conv1d(hidden, hidden, self.conv_size, groups=hidden, bias=False)
        self.k_conv1d = nn.Conv1d(hidden, hidden, self.conv_size, groups=hidden, bias=False)
        self.v_conv1d = nn.Conv1d(hidden, hidden, self.conv_size, groups=hidden, bias=False)
        self.f_proj = nn.Linear(hidden, hidden, bias=False)
        self.b_proj = nn.Linear(hidden, hidden, bias=False)
        self.w_proj = nn.Linear(hidden, hidden, bias=False)
        self.g_proj = nn.Linear(hidden, hidden, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(hidden))
        self.o_norm = GatedRMSNorm(self.head_dim, eps=config.gdn_norm_eps)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def _short_conv(self, conv, x):
        conv_state = x.new_zeros(x.size(0), self.conv_size - 1, x.size(-1))
        x = torch.cat((conv_state, x), dim=1)
        y = F.conv1d(x.transpose(1, 2), conv.weight, groups=x.size(-1)).transpose(1, 2)
        return F.silu(y)

    def forward(self, x):
        batch, length, _ = x.shape
        q_in, k_in, v_in = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        b = self.b_proj(x).double().sigmoid()
        w = self.w_proj(x).double().sigmoid()
        g = F.softplus(self.f_proj(x).double() + self.dt_bias.double())
        q = self._short_conv(self.q_conv1d, q_in)
        k = self._short_conv(self.k_conv1d, k_in)
        v = self._short_conv(self.v_conv1d, v_in)

        def heads(t):
            return t.double().unflatten(-1, (self.num_heads, self.head_dim))

        q, k, v, g, b, w = (heads(t) for t in (q, k, v, g, b, w))
        q, k = l2_normalize(q), l2_normalize(k)
        g = -self.A_log.double().exp().unsqueeze(-1) * g
        recurrent = q.new_zeros(batch, self.num_heads, self.head_dim, self.head_dim)
        scale = self.head_dim ** -0.5
        mixer = gdn2_recurrent if length == 1 else gdn2_chunked
        o, recurrent = mixer(q, k, v, g, b, w, scale, recurrent)

        gate = self.g_proj(x).unflatten(-1, (self.num_heads, self.head_dim))
        o = self.o_norm(o, gate).to(x.dtype).flatten(-2)
        return self.o_proj(o)


class GatedDeltaNet2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kernel = GatedDeltaNet2Kernel(config)

    def forward(self, x, layer_idx):
        return self.kernel(x)


class GatedAttention(nn.Module):
    def __init__(self, config, cache_idx):
        super().__init__()
        self.cache_idx = cache_idx
        self.n_head = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_head // self.n_kv_heads
        hidden = config.hidden_size
        self.q_gate_proj = nn.Linear(hidden, 2 * self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, hidden, bias=False)
        self.q_norm = OffsetRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, rope_cos, rope_sin):
        batch, query_length, _ = x.shape
        q, gate = self.q_gate_proj(x).view(batch, query_length, self.n_head, 2 * self.head_dim).chunk(2, dim=-1)
        q = self.q_norm(q.transpose(1, 2))
        gate = gate.transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(batch, query_length, self.n_kv_heads, self.head_dim).transpose(1, 2))
        v = self.v_proj(x).view(batch, query_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        is_causal = query_length > 1
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=is_causal)
        y = y * torch.sigmoid(gate)
        y = y.transpose(1, 2).reshape(batch, query_length, self.n_head * self.head_dim)
        return self.o_proj(y)


class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class AttnResSources:
    def __init__(self, eps):
        self.eps = eps
        self.values = []
        self.inv_rms = []

    def append(self, source):
        source = source.double()
        self.values.append(source)
        self.inv_rms.append(torch.rsqrt(source.pow(2).mean(-1) + self.eps))


class AttnResAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(config.hidden_size))
        self.key_norm_weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, sources):
        query = self.query.double() * self.key_norm_weight.double()
        logits = torch.stack([(s * query).sum(-1) * r for s, r in zip(sources.values, sources.inv_rms)])
        weights = logits.softmax(dim=0).unsqueeze(-1)
        output = weights[0] * sources.values[0]
        for weight, source in zip(weights[1:], sources.values[1:]):
            output = output + weight * source
        return output


class GPTS3Block(nn.Module):
    def __init__(self, config, layer_idx, cache_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = config.is_attention_layer(layer_idx)
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GatedAttention(config, cache_idx) if self.is_attention else GatedDeltaNet2(config)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)
        self.attn_res_1 = AttnResAggregator(config)
        self.attn_res_2 = AttnResAggregator(config)

    def forward(self, sources, dtype, rope_cos, rope_sin):
        h = self.ln_1(self.attn_res_1(sources).to(dtype))
        if self.is_attention:
            h = self.attn(h, rope_cos, rope_sin)
        else:
            h = self.attn(h, self.layer_idx)
        sources.append(h)
        sources.append(self.mlp(self.ln_2(self.attn_res_2(sources).to(dtype))))


class _Out:
    def __init__(self, logits):
        self.logits = logits


class GPTS3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        attention_layers = [i for i in range(config.num_hidden_layers) if config.is_attention_layer(i)]
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.hidden_size),
            "h": nn.ModuleList([GPTS3Block(config, i, attention_layers.index(i) if i in attention_layers else None)
                                for i in range(config.num_hidden_layers)]),
            "ln_f": RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            "attn_res_f": AttnResAggregator(config),
        })
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer["wte"].weight
        # the source's _init_weights (post_init): Linear / Embedding weights N(0, 0.02), Linear biases 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_rope(self, seq_len, device):
        cos, sin = precompute_rope_cos_sin(self.config.rotary_dim, seq_len, self.config.rope_theta, device=device)
        return cos[:seq_len], sin[:seq_len]

    def forward(self, input_ids):
        _, query_length = input_ids.size()
        if query_length > self.config.max_position_embeddings:
            raise ValueError("sequence longer than max_position_embeddings")
        x = self.transformer["wte"](input_ids)
        rope_cos, rope_sin = self._get_rope(query_length, input_ids.device)
        sources = AttnResSources(self.config.attn_res_eps)
        sources.append(x)
        for block in self.transformer["h"]:
            block(sources, x.dtype, rope_cos, rope_sin)
        hidden = self.transformer["attn_res_f"](sources).to(x.dtype)
        return _Out(self.lm_head(self.transformer["ln_f"](hidden)))
