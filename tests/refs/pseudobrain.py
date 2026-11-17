"""REFERENCE transcription: PseudoBrain PB27Qwen3ForCausalLM / PB27Qwen3MoeForCausalLM (the pb27-vllm plugin).

Source repo:  defnoteatin/PseudoBrain-Qwen3-32B-BF16 (ships the plugin as pb27_vllm_plugin/; the -Qwen3-Coder-30B-A3B-BF16
              repo, snapshot 809d6ae6625d9f92886459313c9bf29ce2948456, uses the same plugin and ships no code itself)
Snapshot:     e8d0f9cc566d61b30d5fc2ecc70dc06542c13149 (pb27_vllm_plugin/pb27_vllm/*.py byte-identical to the copy in
              ~/.cache/upstream_refs/pb27_vllm/pb27_vllm_plugin)
Files (read as text, never imported or run):
  pb27_vllm/universal_models.py  sha256 c2f552e78fc10064ec5f675baebbfed9f6723f311a32e3d883c1ddb2a97bf50c
      108-115  GDN_STACKED_MAPPER (in_proj_qkv / in_proj_z -> in_proj_qkvz, in_proj_b / in_proj_a -> in_proj_ba)
      421-517  PB27Qwen3MoeAttention                      -> PB27Attention
      520-604  PB27Qwen3MoeDecoderLayer                   -> PB27DecoderLayer (moe)
      692-766  PB27Qwen3DecoderLayer (+ _forward_with_memory) -> PB27DecoderLayer (dense), PB27Attention
      777-834  PB27Qwen3Model / PB27Qwen3ForCausalLM, 608-684 PB27Qwen3MoeModel / -ForCausalLM -> PB27ForCausalLM
  pb27_vllm/model.py             sha256 6daa2746b55b7498573500c4e4f664a2ccb0002478876e232de2fb516ce2d8d5
      151-171  pb27_hybrid_config: hybrid layer set + per_layer_sliding_window = pb_attention_window on EVERY
               layer; pb_attention_sinks is only warned about ("vLLM serves ... a plain sliding window (no kept sink
               positions)"), so the reference has NO sinks.
  pb27_vllm/memory.py            sha256 f1f21324d622cc93694ee6a8a14b434d19a176ad86e09607369303c7e6094418
      33-34, 106-117, 1642-1836  the exact-memory tier (on by default: blocks = 256). windowed_attention_with_memory
               first runs vLLM's window attention (1706-1708: skip_base is False while p_first - W + 1 < ORIGIN +
               block = 4 + 64) and only overwrites a piece when retrieval finds archived blocks, which needs
               limit = q_min - W + 1 >= block = 64 (1812-1815). For prompts with T < W + 64 (here T <= 24, W = 8)
               it is therefore exactly vLLM's plain sliding-window attention, which is what is transcribed here.

vLLM 0.31 LIBRARY layers the plugin builds on (library code, semantics transcribed as torch; read as text from
~/.cache/vllm-env/lib/python3.12/site-packages/vllm):
  model_executor/layers/layernorm.py (sha256 d7e98b3f...) RMSNorm: x * rsqrt(mean x^2 + eps) * w, the fused
      residual form norm(x, r) -> (norm(x + r), x + r); RMSNormGated (216-329, norm_before_gate=True):
      rms(x) * w * silu(z)
  model_executor/layers/rotary_embedding (get_rope, is_neox_style=True, rope_theta from the config)
  model_executor/layers/attention Attention(per_layer_sliding_window=W): causal, key j visible iff i - j < W
  model_executor/models/qwen3.py / qwen3_moe.py (sha256 05a05b2e... / cac0cffc...): Qwen3 attention (q/k RMSNorm per
      head before RoPE, GQA), SiluAndMul MLP, Qwen3MoeSparseMoeBlock (softmax over experts, top-k, renormalize =
      norm_topk_prob, no shared expert)
  model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (sha256 3bd2f805...) QwenGatedDeltaNetAttention with
      gqa_interleaved_layout=False (386-538, 895-970, 1245-1521): flat [q|k|v|z] and [b|a]; depthwise causal conv1d
      (kernel K, no bias) + silu over [q|k|v]; third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py:
      q, k l2-normalised (x / sqrt(sum x^2 + 1e-6)), g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b);
      the chunk gated-delta rule (chunk_o.py / chunk_delta_h.py: v-head h reads k-head h // (Hv / Hk), scale
      dk^-0.5) written here as its exact sequential recurrence; then RMSNormGated (eps = rms_norm_eps) and out_proj.
The vLLM kernels compute in fp32 / bf16; everything here is lifted to fp64.

Switches for the tests: `window` (None = full causal attention) and `gdn_on` (False drops the gdn branch).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

F64 = torch.float64


class RMSNorm(nn.Module):
    def __init__(self, n, eps):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(n, dtype=F64)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RMSNormGated(RMSNorm):
    def forward(self, x, z):
        return super().forward(x) * F.silu(z)


def _lin(i, o):
    return nn.Linear(i, o, bias=False, dtype=F64)


class PB27Attention(nn.Module):
    """universal_models.py 421-517 / 717-738 (memory tier: see the header)."""

    def __init__(self, cfg, window):
        super().__init__()
        D, H, KV = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.H, self.KV, self.hd = H, KV, cfg.get("head_dim") or D // H
        self.scaling = self.hd ** -0.5
        self.sliding_window = window
        self.theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", cfg.get("rope_theta", 10000.0)))
        self.q_proj, self.k_proj, self.v_proj = _lin(D, H * self.hd), _lin(D, KV * self.hd), _lin(D, KV * self.hd)
        self.o_proj = _lin(H * self.hd, D)
        self.q_norm, self.k_norm = RMSNorm(self.hd, cfg["rms_norm_eps"]), RMSNorm(self.hd, cfg["rms_norm_eps"])

    def rotary_emb(self, positions, x):          # neox style
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=F64) / self.hd))
        f = positions.to(F64)[:, None] * inv[None]
        cos, sin = f.cos()[:, None], f.sin()[:, None]
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)

    def forward(self, positions, hidden_states):
        T = hidden_states.shape[0]
        q = self.q_norm(self.q_proj(hidden_states).view(T, self.H, self.hd))
        k = self.k_norm(self.k_proj(hidden_states).view(T, self.KV, self.hd))
        v = self.v_proj(hidden_states).view(T, self.KV, self.hd)
        q, k = self.rotary_emb(positions, q), self.rotary_emb(positions, k)
        rep = self.H // self.KV
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        s = torch.einsum("thd,shd->hts", q, k) * self.scaling
        i, j = positions[:, None], positions[None, :]
        vis = j <= i
        if self.sliding_window is not None:
            vis = vis & (i - j < self.sliding_window)
        s = s.masked_fill(~vis, float("-inf"))
        o = torch.einsum("hts,shd->thd", s.softmax(-1), v).reshape(T, self.H * self.hd)
        return self.o_proj(o)


class QwenGatedDeltaNet(nn.Module):
    """vLLM QwenGatedDeltaNetAttention, gqa_interleaved_layout=False (checkpoint names of the plugin's repos)."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg["hidden_size"]
        self.Hk, self.Hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        self.dk, self.dv, self.K = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"]
        kd, vd = self.Hk * self.dk, self.Hv * self.dv
        self.in_proj_qkv, self.in_proj_z = _lin(D, 2 * kd + vd), _lin(D, vd)
        self.in_proj_b, self.in_proj_a = _lin(D, self.Hv), _lin(D, self.Hv)
        self.conv1d = nn.Conv1d(2 * kd + vd, 2 * kd + vd, self.K, groups=2 * kd + vd, bias=False, dtype=F64)
        self.A_log, self.dt_bias = nn.Parameter(torch.zeros(self.Hv, dtype=F64)), nn.Parameter(torch.ones(self.Hv, dtype=F64))
        self.norm = RMSNormGated(self.dv, cfg["rms_norm_eps"])
        self.out_proj = _lin(vd, D)

    def forward(self, hidden_states):
        T = hidden_states.shape[0]
        kd, vd = self.Hk * self.dk, self.Hv * self.dv
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).view(T, self.Hv, self.dv)
        b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
        # causal_conv1d_fn (zero initial state), activation silu
        x = F.pad(mixed_qkv.T[None], (self.K - 1, 0))
        mixed_qkv = F.silu(F.conv1d(x, self.conv1d.weight, groups=2 * kd + vd))[0].T
        q, k, v = mixed_qkv.split([kd, kd, vd], -1)
        l2 = lambda t: t / torch.sqrt(t.pow(2).sum(-1, keepdim=True) + 1e-6)
        q, k = l2(q.view(T, self.Hk, self.dk)), l2(k.view(T, self.Hk, self.dk))
        v = v.view(T, self.Hv, self.dv)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        beta = b.sigmoid()
        rep = self.Hv // self.Hk
        q, k = q.repeat_interleave(rep, 1) * self.dk ** -0.5, k.repeat_interleave(rep, 1)
        S = torch.zeros(self.Hv, self.dk, self.dv, dtype=F64)
        out = []
        for t in range(T):                           # gated delta rule, sequential form of the chunk kernel
            S = S * g[t].exp()[:, None, None]
            kv = torch.einsum("hk,hkv->hv", k[t], S)
            S = S + torch.einsum("hk,hv->hkv", k[t], (v[t] - kv) * beta[t][:, None])
            out.append(torch.einsum("hk,hkv->hv", q[t], S))
        core = torch.stack(out)
        return self.out_proj(self.norm(core, z).reshape(T, vd))


class SiluAndMulMLP(nn.Module):
    def __init__(self, D, f):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = _lin(D, f), _lin(D, f), _lin(f, D)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D, E = cfg["hidden_size"], cfg["num_experts"]
        self.top_k, self.renormalize = cfg["num_experts_per_tok"], cfg.get("norm_topk_prob", True)
        self.gate = _lin(D, E)
        self.experts = nn.ModuleList(SiluAndMulMLP(D, cfg["moe_intermediate_size"]) for _ in range(E))

    def forward(self, x):
        p = self.gate(x).softmax(-1)
        w, idx = p.topk(self.top_k, -1)
        if self.renormalize:
            w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, ex in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out[tok] += w[tok, slot, None] * ex(x[tok])
        return out


class PB27DecoderLayer(nn.Module):
    """universal_models.py 520-604 (moe) / 692-766 (dense): attn(n x) [+ gdn(n x)] then the fused-residual norm."""

    def __init__(self, cfg, i, window, gdn_on):
        super().__init__()
        D, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.is_hybrid = i in set(cfg["pb_hybrid_layers"])
        self.gdn_on = gdn_on
        self.self_attn = PB27Attention(cfg, window)
        if self.is_hybrid:
            self.gdn = QwenGatedDeltaNet(cfg)
        moe = cfg.get("num_experts", 0) > 0 and i not in (cfg.get("mlp_only_layers") or []) \
            and (i + 1) % cfg.get("decoder_sparse_step", 1) == 0
        self.mlp = Qwen3MoeSparseMoeBlock(cfg) if moe else SiluAndMulMLP(D, cfg["intermediate_size"])
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(D, eps), RMSNorm(D, eps)

    def forward(self, positions, hidden_states, residual):
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_layernorm(residual)
        attn_out = self.self_attn(positions, hidden_states)
        if self.is_hybrid and self.gdn_on:
            attn_out = attn_out + self.gdn(hidden_states)
        residual = attn_out + residual
        return self.mlp(self.post_attention_layernorm(residual)), residual


class PB27Model(nn.Module):
    def __init__(self, cfg, window, gdn_on):
        super().__init__()
        D = cfg["hidden_size"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], D, dtype=F64)
        self.layers = nn.ModuleList(PB27DecoderLayer(cfg, i, window, gdn_on) for i in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(D, cfg["rms_norm_eps"])

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[0])
        h, r = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, r = layer(positions, h, r)
        return self.norm(h + r)


class PB27ForCausalLM(nn.Module):
    """PB27Qwen3ForCausalLM / PB27Qwen3MoeForCausalLM. `window` defaults to pb_attention_window; pass
    window=None for full causal attention (the window switched off)."""

    def __init__(self, cfg, window="config", gdn_on=True):
        super().__init__()
        self.config = cfg
        w = cfg.get("pb_attention_window") if window == "config" else window
        self.model = PB27Model(cfg, None if w is None or math.isinf(w) else int(w), gdn_on)
        self.lm_head = _lin(cfg["hidden_size"], cfg["vocab_size"])

    @classmethod
    def from_state_dict(cls, cfg, sd, **kw):
        m = cls(cfg, **kw)
        sd = {k: v.to(F64) for k, v in sd.items()}
        if cfg.get("tie_word_embeddings"):
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
        m.load_state_dict(sd, strict=True)
        return m

    def forward(self, input_ids):
        """input_ids: [T] (one sequence, as vLLM flattens a request). Returns logits [T, vocab]."""
        return self.lm_head(self.model(input_ids))
