"""Non-next-token models on the universal host (coverage metric M12): the snapshot's models whose output is
not a vocabulary distribution (score / reward / embedding heads) or whose input is not tokens (Sundial), each
checked against the family's own code where it is stock, else against a REFERENCE transcribed from the
family's remote code TEXT (the remote code is never executed; each reference names the file, its sha256 and
the line ranges it transcribes). Every new host branch is also switched off once to show it binds.

  Sundial                 thuml/sundial-base-128m: REFERENCE from modeling_sundial.py + flow_loss.py --
                          UPatchEmbed (continuous patch input, RevIN), identity readout at the last patch,
                          UFlowHead (the flow-matching velocity network); the sampler loop is runtime
  BGE MiniCPM layer-wise  BAAI/bge-reranker-v2-minicpm-layerwise: REFERENCE from modeling_minicpm_reranker.py --
                          early-exit score heads (set_exits)
  BGE cost-wise Gemma-2   BAAI/bge-reranker-v2.5-gemma2-lightweight: the stock Gemma-2 body + a REFERENCE of
                          gemma_model.py's layer-wise exits and CostWiseHead (+ the final softcap)
  GRM reward model        Ray2333/GRM-Gemma-2B-sftreg: the stock Gemma body + a REFERENCE of model.py's
                          ValueHead MLP read at the last real token (set_readout mlp=)
  sentence embeddings     Qwen3-Embedding, jina-code-embeddings (last token, Normalize), gte-Qwen2 (bidirectional,
                          last token, Normalize), stella / stella-pl (bidirectional, mean, Dense), voyage-4-nano
                          (REFERENCE of modeling_qwen3_bidirectional.py: bidirectional Qwen3 + a linear, mean,
                          Normalize): stock Qwen2Model / Qwen3Model + the sentence-transformers Pooling / Dense /
                          Normalize modules transcribed (the library is not installed)
  sequence classifier     stock Qwen3ForSequenceClassification (score at the last non-pad token): the general head

Run:  pytest tests/test_nonlm_host.py -q
"""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

F64 = torch.float64


def _rand_(model, seed=0, s=0.05):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * s)
    return model


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1e-30)


def _moves(ref, ours, tol=1e-3):
    return ours.shape != ref.shape or _rel(ref, ours) > tol


# =====================================================================================================
# Sundial -- REFERENCE, transcribed from thuml/sundial-base-128m (revision in models--thuml--sundial-base-128m):
#   modeling_sundial.py  sha256 4a7ce7defa6578d0ae84593587d164afb933cfa6e52aee53f709f236b00b85e4
#     rotate_half, apply_rotary_pos_emb        l. 14-25
#     SundialPatchEmbedding                     l. 28-58    (left pad, [x ; mask], MLP + residual)
#     SundialRotaryEmbedding                    l. 61-98    (fp32 inv_freq; cos / sin cache in the default dtype)
#     SundialAttention.forward (no cache)       l. 101-160  (SDPA under the 4-D additive causal mask)
#     SundialMLP, SundialDecoderLayer           l. 163-229
#     SundialModel.forward (no cache)           l. 254-395  (_prepare_4d_causal_attention_mask, final LayerNorm)
#     SundialForPrediction.forward (inference)  l. 398-508  (revin floor 1e-2, hidden_states[:, -1], flow sample)
#   flow_loss.py         sha256 fb33d3c3988015124c9f4e05728127d85afee8b416930c0e6d5097e4ced2ecf8
#     FlowLoss.__init__ / sample                l. 6-51     (Gaussian noise, Euler steps: x += (net - noise) dt)
#     modulate, TimestepEmbedder                l. 54-98    (fp32 sinusoid of t)
#     ResBlock, FinalLayer, SimpleMLPAdaLN      l. 101-236
# `f32=True` keeps the code's fp32 tensors (rotary table, timestep features, noise / t); an fp64 module needs
# them cast back to fp64 before its Linear layers (marked LIFT) -- the real fp32 checkpoint never casts.
# =====================================================================================================

def _sundial_cfg(**kw):
    c = dict(input_token_len=4, hidden_size=32, intermediate_size=48, output_token_lens=[12], num_hidden_layers=2,
             num_attention_heads=4, hidden_act="silu", rope_theta=10000, max_position_embeddings=64,
             dropout_rate=0.1, flow_loss_depth=2, num_sampling_steps=5, diffusion_batch_mul=4)
    c.update(kw)
    return SimpleNamespace(**c)


def s_rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def s_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos, sin = cos[position_ids].unsqueeze(unsqueeze_dim), sin[position_ids].unsqueeze(unsqueeze_dim)
    return (q * cos) + (s_rotate_half(q) * sin), (k * cos) + (s_rotate_half(k) * sin)


class SundialPatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.hidden_layer = nn.Linear(config.input_token_len * 2, config.intermediate_size)
        self.act = F.silu
        self.output_layer = nn.Linear(config.intermediate_size, config.hidden_size)
        self.residual_layer = nn.Linear(config.input_token_len * 2, config.hidden_size)
        self.input_token_len = config.input_token_len

    def forward(self, x):
        mask = torch.ones_like(x, dtype=torch.float32)
        padding_length = (self.input_token_len - (x.shape[-1] % self.input_token_len)) % self.input_token_len
        x = F.pad(x, (padding_length, 0))
        mask = F.pad(mask, (padding_length, 0))
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        mask = mask.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        x = torch.cat([x, mask], dim=-1)
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        return out + self.residual_layer(x)


class SundialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=10000, base=10000, f32=True):
        super().__init__()
        dt = torch.float32 if f32 else F64
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dt) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=torch.int64).type_as(inv_freq)
        emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dt), persistent=False)     # get_default_dtype(): fp32
        self.register_buffer("sin_cached", emb.sin().to(dt), persistent=False)

    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


class SundialAttention(nn.Module):
    def __init__(self, config, f32=True):
        super().__init__()
        self.hidden_size, self.num_heads = config.hidden_size, config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = SundialRotaryEmbedding(self.head_dim, config.max_position_embeddings, f32=f32)

    def forward(self, hidden_states, attention_mask, position_ids):
        bsz, q_len, _ = hidden_states.size()
        sh = lambda t: t.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = sh(self.q_proj(hidden_states)), sh(self.k_proj(hidden_states)), sh(self.v_proj(hidden_states))
        cos, sin = self.rotary_emb(v, seq_len=k.shape[-2])
        q, k = s_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        o = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        return self.o_proj(o.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size))


class SundialMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, h):
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class SundialDecoderLayer(nn.Module):
    def __init__(self, config, f32=True):
        super().__init__()
        self.self_attn = SundialAttention(config, f32)
        self.ffn_layer = SundialMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.norm1, self.norm2 = nn.LayerNorm(config.hidden_size), nn.LayerNorm(config.hidden_size)

    def forward(self, h, attention_mask, position_ids):
        h = h + self.self_attn(self.norm1(h), attention_mask, position_ids)
        return h + self.ffn_layer(self.norm2(h))


class SundialModel(nn.Module):
    def __init__(self, config, f32=True):
        super().__init__()
        self.embed_layer = SundialPatchEmbedding(config)
        self.layers = nn.ModuleList(SundialDecoderLayer(config, f32) for _ in range(config.num_hidden_layers))
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids):
        h = self.embed_layer(input_ids)
        T = h.shape[1]
        position_ids = torch.arange(T).view(-1, T)
        mask = torch.full((T, T), torch.finfo(h.dtype).min, dtype=h.dtype).triu(1)[None, None]   # 4-D causal
        for layer in self.layers:
            h = layer(h, mask, position_ids)
        return self.norm(h)


def s_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, f32=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size, self.f32 = frequency_embedding_size, f32

    def timestep_embedding(self, t, dim, max_period=10000):
        dt = torch.float32 if self.f32 else F64
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=dt) / half)
        args = t[:, None].to(dt) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))                 # LIFT (fp32 features -> fp64 module)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(channels, channels, bias=True), nn.SiLU(), nn.Linear(channels, channels, bias=True))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        return x + gate_mlp * self.mlp(s_modulate(self.in_ln(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(s_modulate(self.norm_final(x), shift, scale))


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, f32=True):
        super().__init__()
        self.in_channels, self.model_channels, self.out_channels = in_channels, model_channels, out_channels
        self.num_res_blocks = num_res_blocks
        self.time_embed = TimestepEmbedder(model_channels, f32=f32)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(ResBlock(model_channels) for _ in range(num_res_blocks))
        self.final_layer = FinalLayer(model_channels, out_channels)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        y = self.time_embed(t) + self.cond_embed(c)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


class FlowLoss(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, f32=True):
        super().__init__()
        self.in_channels, self.f32 = target_channels, f32
        self.net = SimpleMLPAdaLN(target_channels, width, target_channels, z_channels, depth, f32)
        self.num_sampling_steps = num_sampling_steps

    def sample(self, z, num_samples=1):
        z = z.repeat(num_samples, 1)
        noise = torch.randn(z.shape[0], self.in_channels)                 # fp32, the default dtype
        if not self.f32:
            noise = noise.to(F64)
        x = noise
        dt = 1.0 / self.num_sampling_steps
        for i in range(self.num_sampling_steps):
            t = torch.ones((x.shape[0])) * i / self.num_sampling_steps
            pred = self.net(x.to(z.dtype), (t * 1000).to(z.dtype) if not self.f32 else t * 1000, z)   # LIFT x
            x = x + (pred - noise) * dt
        return x.reshape(num_samples, -1, self.in_channels).transpose(0, 1)


class SundialForPrediction(nn.Module):
    def __init__(self, config, f32=True):
        super().__init__()
        self.config = config
        self.model = SundialModel(config, f32)
        self.flow_loss = FlowLoss(config.output_token_lens[-1], config.hidden_size, config.flow_loss_depth,
                                  config.hidden_size, config.num_sampling_steps, f32)

    def hidden(self, input_ids, revin=False):
        if revin:
            means = input_ids.mean(1, keepdim=True).detach()
            stdev = input_ids.std(dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.where(stdev > 1e-2, stdev, torch.tensor(1.0))
            input_ids = (input_ids - means) / stdev
        return self.model(input_ids)

    def forward(self, input_ids, revin=False, num_samples=1):
        if revin:
            means = input_ids.mean(1, keepdim=True).detach()
            stdev = input_ids.std(dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.where(stdev > 1e-2, stdev, torch.tensor(1.0))
            input_ids = (input_ids - means) / stdev
        hidden_states = self.model(input_ids)
        predictions = self.flow_loss.sample(hidden_states[:, -1, :], num_samples)
        if revin:
            predictions = predictions * stdev + means
        return predictions


def _sundial(f32, seed=0, **kw):
    return _rand_(SundialForPrediction(_sundial_cfg(**kw), f32).to(F64).eval(), seed, s=0.1)


def _series(B=2, S=18, seed=3):
    torch.manual_seed(seed)
    return (torch.randn(B, S) * 2.5 + 4.0).to(F64)


@pytest.mark.parametrize("revin", [False, True])
@torch.no_grad()
def test_same_host_carries_sundial(revin):
    from elephantine.universal import from_sundial
    x = _series()
    for f32, tol in ((False, 1e-12), (True, 1e-5)):
        t = _sundial(f32)
        u = from_sundial(t, revin=revin)
        # (b) the continuous-input body, read at the last patch (S = 18: 2 left-pad values, 5 patches)
        ref_z, z = t.hidden(x, revin)[:, -1], u(series=x)
        assert z.shape == ref_z.shape and _rel(ref_z, z) < tol, ("z", f32, _rel(ref_z, z))
        # (c) the head network v(x, t, z) at arbitrary inputs
        torch.manual_seed(5)
        xs, ts = torch.randn(2, 12, dtype=F64), torch.rand(2, dtype=F64) * 1000
        ref_v, v = t.flow_loss.net(xs, ts, ref_z), u.flow(xs, ts, ref_z)
        assert _rel(ref_v, v) < tol, ("net", f32, _rel(ref_v, v))
        # the whole inference forward, the sampler run for the same noise (B = 1 under revin: the code's
        # `predictions * stdev + means` broadcasts (B, n, C) against (B, 1), only shape-correct at B = 1)
        xb = x[:1] if revin else x
        torch.manual_seed(11)
        ref_p = t(xb, revin=revin, num_samples=2)
        torch.manual_seed(11)
        noise = torch.randn(xb.shape[0] * 2, 12)
        p = u.flow_sample(u(series=xb), noise, t.config.num_sampling_steps, num_samples=2)
        if revin:
            means, std = u.patch.stats
            p = p * std + means
        assert p.shape == ref_p.shape and _rel(ref_p, p) < tol, ("predict", f32, _rel(ref_p, p))
        print("sundial revin" if revin else "sundial", "f32" if f32 else "f64", _rel(ref_p, p))


@torch.no_grad()
def test_sundial_branches_bind():
    """Each new branch, switched off, moves the output: the patch MLP (mlp_c), the mask channel, RevIN, the
    last-token pooling, and in the flow network the adaLN modulation and the timestep features."""
    from elephantine.universal import from_sundial
    t, x = _sundial(False), _series()
    u = from_sundial(t, revin=True)
    z = u(series=x)
    flips = {
        "patch mlp": lambda v: v.patch.mlp_c.zero_(),
        "patch mask channel": lambda v: (v.patch.w1.weight[:, 4:].zero_(), v.patch.wr.weight[:, 4:].zero_()),
        "revin": lambda v: v.patch.rev.zero_(),
        "pool last -> mean": lambda v: v.pool.fill_(3.0),
    }
    for name, flip in flips.items():
        v = from_sundial(t, revin=True)
        flip(v)
        assert _moves(z, v(series=x)), name
    torch.manual_seed(5)
    xs, ts = torch.randn(2, 12, dtype=F64), torch.rand(2, dtype=F64) * 1000
    out = u.flow(xs, ts, z)
    for name, flip in {"adaLN": lambda v: [a.weight.zero_() for a in v.flow.ada],
                       "timestep features": lambda v: v.flow.t2.weight.zero_()}.items():
        v = from_sundial(t, revin=True)
        flip(v)
        assert _moves(out, v.flow(xs, ts, z)), name


# =====================================================================================================
# BGE layer-wise MiniCPM reranker -- REFERENCE, transcribed from BAAI/bge-reranker-v2-minicpm-layerwise:
#   modeling_minicpm_reranker.py  sha256 dbe978b271e0e05ba79dfd9e53811f7b2320cfd20b820954e2cde3bc3ff9473c
#     rms_layernorm / MiniCPMRMSNorm              l. 107-124  (variance in fp32)
#     MiniCPMRotaryEmbedding                      l. 130-164  (cos / sin cache built in fp32)
#     rotate_half, apply_rotary_pos_emb           l. 213-252  (q / k rotated in fp32)
#     MiniCPMMLP (pretraining_tp = 1), repeat_kv  l. 255-298
#     MiniCPMAttention.forward (eager, no cache)  l. 367-470  (rotary on value_states.to(fp32); softmax fp32)
#     MiniCPMDecoderLayer.forward                 l. 778-830  (residual + f(x) * scale_depth / sqrt(L))
#     LayerWiseMiniCPMModel.forward               l. 990-1131 (embeddings x scale_emb, norm(h) at each cutoff)
#     LayerWiseHead, LayerWiseMiniCPMForCausalLM  l. 1134-1209, forward l. 1210-1400 (head_type simple:
#                                                 l. 1362-1376, no / (hidden / dim_model_base))
#   configuration_minicpm_reranker.py sha256 c99d026cce2e4253aaea859c49f55f1a38645496535d74db63aa8d2a66d0bae0
#     defaults start_layer 8, head_multi True, head_type "simple" (l. 142-144)
# =====================================================================================================

def _mcfg(**kw):
    c = dict(vocab_size=89, hidden_size=48, intermediate_size=72, num_hidden_layers=5, num_attention_heads=4,
             num_key_value_heads=2, rms_norm_eps=1e-5, rope_theta=10000.0, rope_scaling=None,
             max_position_embeddings=64, attention_bias=False, pretraining_tp=1, hidden_act="silu", scale_emb=12,
             scale_depth=1.4, dim_model_base=16, start_layer=2, head_multi=True, head_type="simple",
             model_type="minicpm")
    c.update(kw)
    return SimpleNamespace(**c)


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
        dt = torch.float32 if f32 else F64
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).to(dt) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=inv_freq.dtype)
        emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dt), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dt), persistent=False)

    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


def m_apply_rotary_pos_emb(q, k, cos, sin, position_ids, f32=True):
    orig_dtype = k.dtype
    cos, sin = cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1)
    dt = torch.float32 if f32 else orig_dtype
    q32, k32 = q.to(dt), k.to(dt)
    return ((q32 * cos) + (s_rotate_half(q32) * sin)).to(orig_dtype), ((k32 * cos) + (s_rotate_half(k32) * sin)).to(orig_dtype)


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
    def __init__(self, c, f32=True):
        super().__init__()
        self.f32 = f32
        self.num_heads, self.head_dim = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        self.num_key_value_heads = c.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        b = c.attention_bias
        self.q_proj = nn.Linear(c.hidden_size, self.num_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(c.hidden_size, self.num_key_value_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(c.hidden_size, self.num_key_value_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, c.hidden_size, bias=b)
        self.rotary_emb = MiniCPMRotaryEmbedding(self.head_dim, c.max_position_embeddings, c.rope_theta, f32)
        self.hidden_size = c.hidden_size

    def forward(self, hidden_states, attention_mask, position_ids):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(v.to(torch.float32) if self.f32 else v, seq_len=k.shape[-2])
        q, k = m_apply_rotary_pos_emb(q, k, cos, sin, position_ids, self.f32)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim) + attention_mask
        w = F.softmax(w, dim=-1, dtype=torch.float32 if self.f32 else F64).to(q.dtype)
        o = torch.matmul(w, v).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(o)


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, c, f32=True):
        super().__init__()
        self.self_attn = MiniCPMAttention(c, f32)
        self.mlp = MiniCPMMLP(c)
        self.input_layernorm = MiniCPMRMSNorm(c.hidden_size, c.rms_norm_eps, f32)
        self.post_attention_layernorm = MiniCPMRMSNorm(c.hidden_size, c.rms_norm_eps, f32)
        self.scale_depth, self.num_hidden_layers = c.scale_depth, c.num_hidden_layers

    def forward(self, h, attention_mask, position_ids):
        r = self.scale_depth / math.sqrt(self.num_hidden_layers)
        h = h + self.self_attn(self.input_layernorm(h), attention_mask, position_ids) * r
        return h + self.mlp(self.post_attention_layernorm(h)) * r


class LayerWiseMiniCPMModel(nn.Module):
    def __init__(self, c, f32=True):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(MiniCPMDecoderLayer(c, f32) for _ in range(c.num_hidden_layers))
        self.norm = MiniCPMRMSNorm(c.hidden_size, c.rms_norm_eps, f32)

    def forward(self, input_ids, cutoff_layers):
        b, T = input_ids.shape
        position_ids = torch.arange(T)[None]
        h = self.embed_tokens(input_ids) * self.config.scale_emb
        mask = torch.full((T, T), torch.finfo(h.dtype).min, dtype=h.dtype).triu(1)[None, None]
        max_layer = max(cutoff_layers)
        all_hidden_states = ()
        for idx, layer in enumerate(self.layers):
            if idx in cutoff_layers:
                all_hidden_states += (self.norm(h),)
            if idx == max_layer:
                break
            h = layer(h, mask, position_ids)
        h = self.norm(h)
        if self.config.num_hidden_layers == max_layer:
            all_hidden_states += (h,)
        return all_hidden_states


class LayerWiseHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_head = nn.Linear(input_size, output_size, bias=False)


class LayerWiseMiniCPMForCausalLM(nn.Module):
    def __init__(self, c, f32=True):
        super().__init__()
        self.config = c
        self.model = LayerWiseMiniCPMModel(c, f32)
        width = 1 if c.head_type not in ("raw", "complex") else c.vocab_size
        mk = (lambda: nn.Linear(c.hidden_size, width, bias=False)) if c.head_type == "raw" else \
            (lambda: LayerWiseHead(c.hidden_size, width))
        self.lm_head = nn.ModuleList(mk() for _ in range(c.start_layer, c.num_hidden_layers + 1)) if c.head_multi else mk()

    def forward(self, input_ids, cutoff_layers=None):
        c = self.config
        if cutoff_layers is None:
            cutoff_layers = [c.num_hidden_layers]
        elif isinstance(cutoff_layers, int):
            cutoff_layers = [cutoff_layers]
        cutoff_layers = [i for i in cutoff_layers if c.start_layer <= i <= c.num_hidden_layers]
        hs = self.model(input_ids, cutoff_layers)
        all_logits = ()
        for i in range(len(hs)):
            head = self.lm_head[cutoff_layers[i] - c.start_layer] if c.head_multi else self.lm_head
            head = head if isinstance(head, nn.Linear) else head.linear_head
            x = hs[i] if c.head_type not in ("raw", "complex") else hs[i] / (c.hidden_size / c.dim_model_base)
            logits = head(x)
            logits = (logits.float() if self.f32 else logits).reshape(input_ids.shape[0], -1)
            all_logits += (logits,)
        return all_logits

    @property
    def f32(self):
        return self.model.norm.f32


MCASES = {
    "released": (dict(), None),                                     # the default call: one exit, the last layer
    "exits": (dict(), [2, 4, 5]),                                   # layer-wise: three exits
    "gqa_shared_head": (dict(num_key_value_heads=1, head_multi=False), [3, 5]),
    "complex_vocab": (dict(head_type="complex"), [2, 5]),           # a vocabulary head on h / (hidden / base)
}


@pytest.mark.parametrize("case", list(MCASES))
@torch.no_grad()
def test_same_host_carries_minicpm_layerwise_reranker(case):
    from elephantine.universal import from_minicpm_layerwise
    kw, cut = MCASES[case]
    c = _mcfg(**kw)
    torch.manual_seed(1)
    x = torch.randint(0, c.vocab_size, (2, 9))
    for f32, tol in ((False, 1e-12), (True, 1e-5)):
        t = _rand_(LayerWiseMiniCPMForCausalLM(c, f32).to(F64).eval(), s=0.1)
        ref = t(x, cut)
        ours = from_minicpm_layerwise(t, cut)(x)
        assert len(ours) == len(ref)
        for r, o in zip(ref, ours):
            o = o.reshape(x.shape[0], -1)
            assert _rel(r.to(F64), o) < tol, (case, f32, _rel(r.to(F64), o))
        print(case, "f32" if f32 else "f64", max(_rel(r.to(F64), o.reshape(2, -1)) for r, o in zip(ref, ours)))


@torch.no_grad()
def test_minicpm_layerwise_exits_bind():
    """The early exits bind: the same heads read at the final layer instead of at their own exits move the
    scores, and so does dropping a head's weights."""
    from elephantine.universal import from_minicpm_layerwise
    c = _mcfg()
    t = _rand_(LayerWiseMiniCPMForCausalLM(c, False).to(F64).eval(), s=0.1)
    x = torch.randint(0, c.vocab_size, (2, 9))
    u = from_minicpm_layerwise(t, [2, 4, 5])
    out = u(x)
    v = from_minicpm_layerwise(t, [2, 4, 5])
    v.cfg.exits = (5, 5, 5)                                          # every head at the final layer
    assert all(_moves(a, b) for a, b in zip(out[:2], v(x)[:2]))
    v = from_minicpm_layerwise(t, [2, 4, 5])
    v.exit_heads[1].weight.zero_()
    assert _moves(out[1], v(x)[1])


# =====================================================================================================
# BGE cost-wise Gemma-2 reranker -- the body is STOCK (gemma_model.py imports Gemma2DecoderLayer / Gemma2RMSNorm
# from transformers.models.gemma2 and runs them in CostWiseGemmaModel's loop: embeddings x sqrt(hidden), the layer
# stack, the final norm -- the stock Gemma2Model's loop), so the reference is the stock Gemma2Model plus a
# REFERENCE of the layer-wise readout, transcribed from BAAI/bge-reranker-v2.5-gemma2-lightweight:
#   gemma_model.py  sha256 920184d3e239160a061f797010cda29c634ae9a7c4d3bbc288028006330489b3
#     CostWiseGemmaModel.forward                  l. 273-438   (embeddings x sqrt(hidden) l. 340-341; layer_wise:
#                                                 norm(h) appended at each cutoff idx, break at max l. 366-374; the
#                                                 final norm(h) when max(cutoff) == L l. 416-426)
#     CostWiseHead                                l. 485-493   (Linear(hidden, 1), no bias)
#     CostWiseGemmaForCausalLM.__init__           l. 499-514   (one head per range(start_layer, L + 1, layer_sep))
#     .forward, layer_wise                        l. 589-600, 643-655  (cutoffs filtered to that range; output i
#                                                 -> lm_head[i].linear_head, final-logit softcap, float)
#   gemma_config.py sha256 7dae68c2514d25c475256912ca138b9e1d7b3f6cabdc48bf6e176a259f3fb9dc
# The token-compression path (compress_layer / compress_ratio, off by default) is not transcribed (not carried).
# =====================================================================================================

class CostWiseRef(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import Gemma2Model
        self.config = cfg
        self.model = Gemma2Model(cfg)
        n = len(range(cfg.start_layer, cfg.num_hidden_layers + 1, cfg.layer_sep))
        self.lm_head = nn.ModuleList(LayerWiseHead(cfg.hidden_size, 1) for _ in range(n))

    def forward(self, input_ids, cutoff_layers=None):
        c = self.config
        if cutoff_layers is None:
            cutoff_layers = [c.num_hidden_layers]
        elif isinstance(cutoff_layers, int):
            cutoff_layers = [cutoff_layers]
        can_use = list(range(c.start_layer, c.num_hidden_layers + 1, c.layer_sep))
        cutoff_layers = [i for i in cutoff_layers if i in can_use]
        out = self.model(input_ids, output_hidden_states=True)       # hidden_states[idx]: the stream before layer idx
        max_layer = max(cutoff_layers)
        hs = tuple(self.model.norm(out.hidden_states[idx]) for idx in range(c.num_hidden_layers) if idx in cutoff_layers)
        if max_layer == c.num_hidden_layers:
            hs += (out.last_hidden_state,)                            # = norm(the stream after L layers)
        logits = ()
        for i in range(len(hs)):
            z = self.lm_head[i].linear_head(hs[i])
            if c.final_logit_softcapping is not None:
                z = torch.tanh(z / c.final_logit_softcapping) * c.final_logit_softcapping
            logits += (z.reshape(hs[i].shape[0], -1),)
        return logits


def _costwise(**kw):
    from transformers import Gemma2Config
    cfg = Gemma2Config(vocab_size=97, hidden_size=48, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=4096,
                       attn_logit_softcapping=50.0, final_logit_softcapping=3.0, attn_implementation="eager", **kw)
    cfg.model_type = "cost_wise_gemma"
    cfg.layer_wise, cfg.start_layer, cfg.layer_sep = True, 2, 2
    return _rand_(CostWiseRef(cfg).to(F64).eval(), s=0.1)


@pytest.mark.parametrize("cut", [None, [2, 4], [4, 6], 3])
@torch.no_grad()
def test_same_host_carries_cost_wise_gemma(cut):
    from elephantine.universal import from_cost_wise_gemma
    t = _costwise()
    torch.manual_seed(2)
    x = torch.randint(0, 97, (2, 10))
    if cut == 3:                                                      # not in range(2, 7, 2): the code drops it
        with pytest.raises(AssertionError):
            from_cost_wise_gemma(t, cut)
        return
    ref, ours = t(x, cut), from_cost_wise_gemma(t, cut)(x)
    assert len(ours) == len(ref)
    for r, o in zip(ref, ours):
        assert _rel(r, o.reshape(2, -1)) < 1e-6, _rel(r, o.reshape(2, -1))   # stock Gemma-2: fp32 rotary / norms
    # the softcap and the exits bind
    u = from_cost_wise_gemma(t, cut)
    u.final_cap.fill_(float("inf"))
    assert _moves(ref[0], u(x)[0].reshape(2, -1))
    print(cut, max(_rel(r, o.reshape(2, -1)) for r, o in zip(ref, ours)))


# =====================================================================================================
# GRM reward model -- the body is the STOCK causal LM (model.py wraps AutoModelForCausalLM through trl's
# PreTrainedModelWrapper); the value head is a REFERENCE transcribed from Ray2333/GRM-Gemma-2B-sftreg:
#   model.py  sha256 61f58c525143a60d23de6d2c458779ff5f65ad876a0f3c943c84618b3f39cf4f
#     ValueHead.__init__ / forward                l. 9-64     (mlp: [Linear(hidden, n), ReLU] x num_layers, Linear(n, 1))
#     AutoModelForCausalLMWithValueHead.forward   l. 106-147  (v_head(hidden_states[-1]) at the last real token:
#                                                 left padding -> T - 1, else attention_mask.sum - 1)
#   config.json: vhead_layer_type "mlp", vhead_num_layers 1, vhead_num_neurons 1024
# =====================================================================================================

class ValueHead(nn.Module):
    def __init__(self, hidden, layer_type="mlp", num_neurons=24, num_layers=1):
        super().__init__()
        if layer_type == "linear":
            self.summary = nn.Linear(hidden, 1)
        else:
            mods, n_in = [], hidden
            for _ in range(num_layers):
                mods += [nn.Linear(n_in, num_neurons), nn.ReLU()]
                n_in = num_neurons
            mods.append(nn.Linear(num_neurons, 1))
            self.summary = nn.Sequential(*mods)

    def forward(self, h):
        return self.summary(h)


class GRMRef(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        from transformers import GemmaConfig, GemmaForCausalLM
        cfg = GemmaConfig(vocab_size=97, hidden_size=48, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                          num_key_value_heads=1, head_dim=16, attn_implementation="eager")
        self.pretrained_model = GemmaForCausalLM(cfg)
        self.v_head = ValueHead(48, layer_type)

    def forward(self, input_ids, attention_mask):
        out = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = out.hidden_states[-1]
        if torch.any(attention_mask[:, 0] == 0):
            last_index = attention_mask.shape[-1] - 1
        else:
            last_index = attention_mask.sum(dim=-1) - 1
        return self.v_head(last_hidden_state).squeeze(-1)[torch.arange(len(last_hidden_state)), last_index]


@pytest.mark.parametrize("layer_type", ["mlp", "linear"])
@torch.no_grad()
def test_same_host_carries_grm_value_head(layer_type):
    from elephantine.universal import from_grm
    t = _rand_(GRMRef(layer_type).to(F64).eval(), s=0.1)
    torch.manual_seed(4)
    x = torch.randint(0, 97, (2, 8))
    ref = t(x, torch.ones_like(x))
    u = from_grm(t)
    ours = u(x)[:, 0]
    assert _rel(ref, ours) < 1e-6, _rel(ref, ours)
    # right padding: the reward at the last real token (the host's pad_mask; the code's attention mask also
    # hides the padded keys, which a causal model's earlier positions never see)
    am = torch.ones_like(x)
    am[1, 5:] = 0
    ref = t(x, am)
    ours = u(x, pad_mask=am.bool())[:, 0]
    assert _rel(ref, ours) < 1e-6, _rel(ref, ours)
    if layer_type == "mlp":                                           # the hidden head layer binds
        v = from_grm(t)
        v.head_act.fill_(0.0)                                         # relu -> silu
        assert _moves(ref, v(x, pad_mask=am.bool())[:, 0])
    print(layer_type, _rel(ref, ours))


# =====================================================================================================
# Sentence-embedding models: the body is the stock decoder (Qwen3-Embedding / jina-code-embeddings: Qwen3Model /
# Qwen2Model, causal; gte-Qwen2 / stella / stella-pl: their remote modeling_qwen.py is transformers' Qwen2 with an
# is_causal switch, forward(is_causal=False) default -> _prepare_4d_attention_mask, i.e. a padding-only,
# bidirectional mask: run here as the stock Qwen2Model under a full 4-D mask), then sentence-transformers'
# Pooling / Dense / Normalize (modules.json + 1_Pooling / 2_Dense configs of each repo), transcribed:
#   Pooling lasttoken: the rightmost position with attention_mask 1 (flip(1).max(1)); mean: sum(h m) / clamp(sum m,
#   1e-9); Dense: activation(linear(x)) (Identity here); Normalize: F.normalize(x, p=2, dim=1).
# voyage-4-nano -- REFERENCE, transcribed from MongoDB/voyage-4-nano:
#   modeling_qwen3_bidirectional.py  sha256 425304da2571d17bf04add3b70fab46d4b7d4466abbe118ef553ccb6fe8fadfe
#     Qwen3BidirectionalModel  l. 30-91  (stock Qwen3Model, is_causal False, an or_mask_function that admits every
#                                         real key; last_hidden_state = linear(.) : Linear(hidden, num_labels), no bias)
#   modules.json: Transformer -> Pooling (mean) -> Normalize
# =====================================================================================================

def st_pool(h, mask, mode):
    if mode == "last":
        values, indices = mask.flip(1).max(1)
        indices = torch.where(values == 0, mask.shape[1] - 1, indices)
        return h[torch.arange(h.shape[0]), mask.shape[1] - indices - 1]
    m = mask[..., None].to(h.dtype)
    return (h * m).sum(1) / torch.clamp(m.sum(1), min=1e-9)


def _body(kind):
    from transformers import Qwen2Config, Qwen2Model, Qwen3Config, Qwen3Model
    kw = dict(vocab_size=97, hidden_size=48, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=2, attn_implementation="eager")
    m = Qwen3Model(Qwen3Config(**kw, head_dim=16)) if kind == "qwen3" else Qwen2Model(Qwen2Config(**kw))
    return _rand_(m.to(F64).eval(), s=0.1)


ECASES = {   # body, pool, Dense (out width) or None, Normalize, causal
    "qwen3_embedding":       ("qwen3", "last", None, True, True),
    "jina_code_embeddings":  ("qwen2", "last", None, True, True),
    "gte_qwen2_bidir":       ("qwen2", "last", None, True, False),
    "stella_bidir_dense":    ("qwen2", "mean", 32, False, False),
    "voyage_bidir_linear":   ("qwen3", "mean", "linear", True, False),
}


@pytest.mark.parametrize("case", list(ECASES))
@torch.no_grad()
def test_same_host_carries_sentence_embeddings(case):
    from elephantine.universal import from_embedding
    kind, pool, dense, l2, causal = ECASES[case]
    body = _body(kind)
    torch.manual_seed(6)
    x = torch.randint(0, 97, (2, 9))
    am = torch.ones_like(x)
    am[1, 6:] = 0                                                     # right padding on the second row
    T = x.shape[1]
    if causal:
        h = body(input_ids=x, attention_mask=am).last_hidden_state
    else:                                                             # bidirectional: every real key visible
        full = torch.zeros(2, 1, T, T, dtype=F64).masked_fill(~am.bool()[:, None, None, :], torch.finfo(F64).min)
        h = body(input_ids=x, attention_mask=full).last_hidden_state
    lin = None
    if dense == "linear":                                             # voyage: per-token linear, then pooling
        lin = _rand_(nn.Linear(48, 40, bias=False).to(F64), seed=7, s=0.3)
        h = lin(h)
    elif dense:
        lin = _rand_(nn.Linear(48, dense).to(F64), seed=7, s=0.3)
    e = st_pool(h, am, pool)
    if dense and dense != "linear":
        e = lin(e)
    ref = F.normalize(e, p=2, dim=1) if l2 else e
    u = from_embedding(body, pool=pool, l2=l2, dense=lin, causal=causal)
    # the host's attention mask: causal (padded keys sit after every real query) or full over the real keys
    kmask = None if causal else (am.bool()[:, None, :] & torch.ones(T, T, dtype=torch.bool)[None])
    ours = u(x, causal=causal, mask=kmask, pad_mask=am.bool())
    assert ours.shape == ref.shape and _rel(ref, ours) < 1e-6, (case, _rel(ref, ours))
    v = from_embedding(body, pool=pool, l2=l2, dense=lin, causal=causal)
    v.pool.fill_(1.0 if pool == "mean" else 3.0)                      # the pooling rule binds
    assert _moves(ref, v(x, causal=causal, mask=kmask, pad_mask=am.bool()))
    if l2:
        v = from_embedding(body, pool=pool, l2=l2, dense=lin, causal=causal)
        v.l2_out.zero_()
        assert _moves(ref, v(x, causal=causal, mask=kmask, pad_mask=am.bool()))
    print(case, _rel(ref, ours))


@torch.no_grad()
def test_same_host_carries_stock_sequence_classifier():
    """The general head: stock Qwen3ForSequenceClassification (score: Linear(hidden, num_labels), no bias) read at
    the rightmost non-pad token (pad_token_id)."""
    from transformers import Qwen3Config, Qwen3ForSequenceClassification
    from elephantine.universal import from_seq_cls
    cfg = Qwen3Config(vocab_size=97, hidden_size=48, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, num_labels=3, pad_token_id=0, attn_implementation="eager")
    t = _rand_(Qwen3ForSequenceClassification(cfg).to(F64).eval(), s=0.1)
    torch.manual_seed(8)
    x = torch.randint(1, 97, (2, 9))
    x[1, 6:] = 0
    ref = t(input_ids=x).logits
    ours = from_seq_cls(t)(x)
    assert ours.shape == (2, 3) and _rel(ref, ours) < 1e-6, _rel(ref, ours)
    v = from_seq_cls(t)
    v.pool_pad.fill_(-1.0)                                            # ignore padding: the last position
    assert _moves(ref, v(x))
