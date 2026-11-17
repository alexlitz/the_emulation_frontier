"""MiniCPM-SALA (model_type minicpm_sala, MiniCPMSALAForCausalLM) on the universal host: from_minicpm_sala.

No stock class exists (transformers 4.57, 5.18 and 5.22.2 have no minicpm_sala), and the family's own code is
remote (openbmb modeling_minicpm_sala.py, needs flash-attn / fla / infllm_v2 kernels): it was READ AS TEXT, NOT
EXECUTED. The model below is a REFERENCE: a line-by-line transcription of that file's prefill forward (no cache),
with the same module / parameter names, so from_minicpm_sala reads it exactly as it would a real checkpoint.
Transcribed (modeling_minicpm_sala.py, the copy in models--cyankiwi--MiniCPM-SALA-AWQ-8bit):
    rms_layernorm / MiniCPMRMSNorm                     l. 525-542   (variance in fp32)
    MiniCPMRotaryEmbedding, rotate_half,
      apply_rotary_pos_emb                             l. 548-587, 707-742   (fp32 table, q / k rotated in fp32)
    MiniCPMMLP (pretraining_tp = 1)                    l. 745-787
    MiniCPMAttention.forward (eager; the dense path)   l. 924-1061  (scores / sqrt(d), softmax in fp32, o_gate)
    _build_slope_tensor                                l. 2037-2058
    LightningAttention.__init__ / forward / attn_fn    l. 2064-2317 (simple GLA, g_gamma = -slopes, fp32)
    MiniCPMSALADecoderLayer.forward                    l. 2474-2539 (residual + f(x) * scale_depth / sqrt(L))
    MiniCPMSALAModel.forward                           l. 2697-2858 (embeddings * scale_emb, causal 4-D mask)
    MiniCPMSALAForCausalLM.forward                     l. 2895-2983 (lm_head(h / (hidden / dim_model_base)).float())
fla's fused_recurrent_simple_gla / chunk_simple_gla (the lightning kernel) is the recurrence
    S_t = exp(g_gamma_h) S_{t-1} + k_t v_t^T,  o_t = scale q_t^T S_t     (fp32 state)
-- written here as that recurrence. The minicpm4 layers' InfLLM-v2 sparse path (CUDA only, kv length >=
sparse_config.dense_len) is not transcribed: below dense_len the code runs dense causal flash attention, which is
the eager path above (no padding). `f32=True` keeps the code's fp32 casts; `f32=False` lifts them to fp64.
"""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

F64 = torch.float64


def _cfg(**kw):
    c = dict(vocab_size=89, hidden_size=48, intermediate_size=72, num_hidden_layers=4, num_attention_heads=4,
             num_key_value_heads=2, head_dim=None, lightning_nkv=None, rms_norm_eps=1e-6, rope_theta=10000.0,
             rope_scaling=None, max_position_embeddings=64, attention_bias=False, pretraining_tp=1,
             hidden_act="silu", scale_emb=12, scale_depth=1.4, dim_model_base=16,
             mixer_types=["minicpm4", "lightning-attn", "lightning-attn", "minicpm4"],
             attn_use_rope=False, lightning_use_rope=True, qk_norm=True, use_output_gate=True, use_output_norm=True,
             attn_use_output_gate=True, lightning_scale="1/sqrt(d)", model_type="minicpm_sala",
             sparse_config={"block_size": 64, "dense_len": 8192, "kernel_size": 32, "kernel_stride": 16,
                            "init_blocks": 1, "topk": 64, "window_size": 2048, "use_nope": False})
    c.update(kw)
    # MiniCPMSALAConfig.__init__ defaults (configuration_minicpm_sala.py)
    if c["head_dim"] is None:
        c["head_dim"] = c["hidden_size"] // c["num_attention_heads"]
    if c["lightning_nkv"] is None:
        c["lightning_nkv"] = c["num_key_value_heads"]
    return SimpleNamespace(**c)


# --- REFERENCE (transcribed from modeling_minicpm_sala.py; see the module docstring) ----------------

class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, f32=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon, self.f32 = eps, f32

    def forward(self, hidden):
        old_dtype = hidden.dtype
        variance = (hidden.to(torch.float32) if self.f32 else hidden).pow(2).mean(dim=-1, keepdim=True)
        hidden = (hidden * torch.rsqrt(variance + self.variance_epsilon)).to(old_dtype)
        return hidden * self.weight


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, f32=True):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.f32 = f32

    def forward(self, x, seq_len):
        dt = torch.float32 if self.f32 else F64
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype).to(dt)
        freqs = torch.outer(t, self.inv_freq.to(dt))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, f32=True):
    orig = k.dtype
    cos, sin = cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1)
    if f32:
        q, k = q.to(torch.float32), k.to(torch.float32)
    q_embed, k_embed = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig), k_embed.to(orig)


def repeat_kv(h, n_rep):
    b, nkv, s, d = h.shape
    return h if n_rep == 1 else h[:, :, None].expand(b, nkv, n_rep, s, d).reshape(b, nkv * n_rep, s, d)


class MiniCPMMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMAttention(nn.Module):
    def __init__(self, c, layer_idx, f32=True):
        super().__init__()
        self.config, self.f32 = c, f32
        self.num_heads, self.head_dim = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        self.num_key_value_heads = c.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        b = c.attention_bias
        self.q_proj = nn.Linear(c.hidden_size, self.num_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(c.hidden_size, self.num_key_value_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(c.hidden_size, self.num_key_value_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, c.hidden_size, bias=b)
        self.rotary_emb = MiniCPMRotaryEmbedding(self.head_dim, c.max_position_embeddings, c.rope_theta, f32)
        self.use_output_gate = c.attn_use_output_gate
        if self.use_output_gate:
            self.o_gate = nn.Linear(c.hidden_size, self.num_heads * self.head_dim, bias=b)

    def forward(self, hidden_states, attention_mask, position_ids):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = position_ids.max().item() + 1
        if self.config.attn_use_rope:
            cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, self.f32)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim) + attention_mask
        w = F.softmax(w, dim=-1, dtype=torch.float32 if self.f32 else F64).to(q.dtype)
        o = torch.matmul(w, v).transpose(1, 2).reshape(bsz, q_len, -1)
        if self.use_output_gate:
            o = o * torch.sigmoid(self.o_gate(hidden_states))
        return self.o_proj(o)


def _build_slope_tensor(nheads):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * start ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        cp = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(cp) + get_slopes(2 * cp)[0::2][: n - cp]
    return torch.tensor(get_slopes(nheads))


def simple_gla(q, k, v, g_gamma, scale):
    """fla fused_recurrent_simple_gla / chunk_simple_gla, no initial state: (b, t, h, d) in, (b, t, h, d) out."""
    b, t, h, dk = q.shape
    S = torch.zeros(b, h, dk, v.shape[-1], dtype=q.dtype)
    a = g_gamma.exp()[None, :, None, None]
    out = []
    for i in range(t):
        S = S * a + k[:, i, :, :, None] * v[:, i, :, None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, i] * scale, S))
    return torch.stack(out, 1)


class LightningAttention(nn.Module):
    def __init__(self, c, layer_idx, hidden_size, num_attention_heads, num_key_value_heads, head_dim,
                 use_output_gate, attention_bias, rms_norm_eps, use_rope, use_output_norm, qk_norm, scale, f32=True):
        super().__init__()
        self.config, self.f32 = c, f32
        self.num_attention_heads, self.num_key_value_heads, self.head_dim = num_attention_heads, num_key_value_heads, head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scale = head_dim ** -0.5 if scale == "1/sqrt(d)" else (head_dim ** -1.0 if scale == "1/d" else 1.0)
        self.use_output_gate, self.use_rope, self.qk_norm, self.use_output_norm = use_output_gate, use_rope, qk_norm, use_output_norm
        b = attention_bias
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=b)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=b)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=b)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=b)
        if use_output_norm:
            self.o_norm = MiniCPMRMSNorm(num_attention_heads * head_dim, rms_norm_eps, f32)
        if use_output_gate:
            self.z_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=b)
        if qk_norm:
            self.q_norm = MiniCPMRMSNorm(head_dim, rms_norm_eps, f32)
            self.k_norm = MiniCPMRMSNorm(head_dim, rms_norm_eps, f32)
        self.rotary_emb = MiniCPMRotaryEmbedding(head_dim, c.max_position_embeddings, c.rope_theta, f32)

    def forward(self, hidden_states, attention_mask, position_ids):
        bsz, seqlen, _ = hidden_states.shape
        sh = lambda t: t.view(bsz, seqlen, -1, self.head_dim).transpose(1, 2)      # b t (h d) -> b h t d
        q, k, v = sh(self.q_proj(hidden_states)), sh(self.k_proj(hidden_states)), sh(self.v_proj(hidden_states))
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            cos, sin = self.rotary_emb(v, seq_len=position_ids.max().item() + 1)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, self.f32)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        dt = torch.float32 if self.f32 else F64
        s = _build_slope_tensor(self.num_attention_heads).to(dtype=torch.float32) * -1.0
        q, k, v = (t.transpose(1, 2).to(dt) for t in (q, k, v))                   # b h t d -> b t h d
        o = simple_gla(q, k, v, s.to(dt), self.scale)
        o = o.reshape(bsz, seqlen, -1).to(hidden_states.dtype)
        if self.use_output_norm:
            o = self.o_norm(o)
        if self.use_output_gate:
            o = o * torch.sigmoid(self.z_proj(hidden_states))
        return self.o_proj(o)


class MiniCPMSALADecoderLayer(nn.Module):
    def __init__(self, c, layer_idx, f32=True):
        super().__init__()
        self.mixer_type = c.mixer_types[layer_idx]
        if self.mixer_type == "minicpm4":
            self.self_attn = MiniCPMAttention(c, layer_idx, f32)
        else:
            self.self_attn = LightningAttention(c, layer_idx, c.hidden_size, c.num_attention_heads, c.lightning_nkv,
                                                c.head_dim, c.use_output_gate, c.attention_bias, c.rms_norm_eps,
                                                c.lightning_use_rope, c.use_output_norm, c.qk_norm, c.lightning_scale, f32)
        self.mlp = MiniCPMMLP(c)
        self.input_layernorm = MiniCPMRMSNorm(c.hidden_size, c.rms_norm_eps, f32)
        self.post_attention_layernorm = MiniCPMRMSNorm(c.hidden_size, c.rms_norm_eps, f32)
        self.scale_depth, self.num_hidden_layers = c.scale_depth, c.num_hidden_layers

    def forward(self, h, attention_mask, position_ids):
        r = self.scale_depth / math.sqrt(self.num_hidden_layers)
        h = h + self.self_attn(self.input_layernorm(h), attention_mask, position_ids) * r
        return h + self.mlp(self.post_attention_layernorm(h)) * r


class MiniCPMSALAModel(nn.Module):
    def __init__(self, c, f32=True):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(MiniCPMSALADecoderLayer(c, i, f32) for i in range(c.num_hidden_layers))
        self.norm = MiniCPMRMSNorm(c.hidden_size, c.rms_norm_eps, f32)

    def forward(self, input_ids):
        b, T = input_ids.shape
        position_ids = torch.arange(T)[None]
        h = self.embed_tokens(input_ids) * self.config.scale_emb
        mask = torch.full((T, T), torch.finfo(h.dtype).min, dtype=h.dtype).triu(1)[None, None]   # 4-D causal
        for layer in self.layers:
            h = layer(h, mask, position_ids)
        return self.norm(h)


class MiniCPMSALAForCausalLM(nn.Module):
    def __init__(self, c, f32=True):
        super().__init__()
        self.config, self.f32 = c, f32
        self.model = MiniCPMSALAModel(c, f32)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, input_ids):
        h = self.model(input_ids)
        logits = self.lm_head(h / (self.config.hidden_size / self.config.dim_model_base))
        return logits.float() if self.f32 else logits


# --- tests --------------------------------------------------------------------------------------

def _ref(c, f32, seed=0):
    torch.manual_seed(seed)
    t = MiniCPMSALAForCausalLM(c, f32).to(F64).eval()
    with torch.no_grad():
        for p in t.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return t


CASES = {
    # the released layout: NoPE softmax layers with an output gate; lightning with qk-norm, rope, output norm + gate
    "released": dict(),
    # every switch flipped: rope on the softmax layers, no qk-norm / gate / output norm, MHA lightning, biases,
    # 1/d scale, 6 heads (the non-power-of-2 slope branch), a lightning head dim != hidden / heads
    "flipped": dict(num_attention_heads=6, num_key_value_heads=3, lightning_nkv=6, head_dim=10, attn_use_rope=True,
                    qk_norm=False, use_output_gate=False, use_output_norm=False, attn_use_output_gate=False,
                    attention_bias=True, lightning_scale="1/d",
                    mixer_types=["lightning-attn", "minicpm4", "lightning-attn", "lightning-attn"]),
    "gqa_lightning": dict(num_key_value_heads=1, lightning_nkv=2, attn_use_rope=True),
}


@pytest.mark.parametrize("case", list(CASES))
@torch.no_grad()
def test_same_host_carries_minicpm_sala(case):
    from elephantine.universal import from_minicpm_sala
    c = _cfg(**CASES[case])
    torch.manual_seed(1)
    x = torch.randint(0, c.vocab_size, (2, 11))
    for f32, tol in ((False, 1e-12), (True, 2e-6)):              # casts lifted: exact; the code's fp32: fp32 level
        t = _ref(c, f32)
        ref = t(x).to(F64)
        ours = from_minicpm_sala(t)(x)
        gap = (ref - ours).abs().max().item() / ref.abs().max().item()
        print(case, "f32" if f32 else "f64", gap)
        assert gap < tol, (case, f32, gap)
