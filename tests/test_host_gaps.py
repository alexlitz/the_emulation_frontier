"""Host gaps (coverage metric M11): the families the universal host did not carry or had no whole-model
loader for, each checked against the family's own code where a stock class exists, else against a
reference written from the family's remote code TEXT (the remote code itself is never executed; each
such reference is labelled REFERENCE and names the file it transcribes).

  SDAR / SDAR-MoE      block-causal / caller-supplied masks: stock Qwen3 / Qwen3-MoE (SDAR's body, read
                       from modeling_sdar*.py) run under SDAR's own mask, passed externally
  Ling-3.x (V3)        bailing_hybrid: stock BailingHybridForCausalLM (transformers 5.22.2, a subprocess)
  Ling-2.5 / 2.6       bailing_hybrid (BailingMoeV2_5): REFERENCE from modeling_bailing_moe_v2_5.py
  MiniCPM-SALA         minicpm_sala: REFERENCE from modeling_minicpm_sala.py
  RWKV-4               whole model vs stock RwkvForCausalLM
  PLaMo-2              the attention window (from_plamo2 now sets it)

Run:  pytest tests/test_host_gaps.py -q
"""
import pytest
import torch

F64 = torch.float64
TOL = 1e-6           # the stock classes round the rotary table and norms to fp32


def _rand_(model, seed=0, s=0.05):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 1:
                p.add_(torch.randn_like(p) * s)
    return model


def _ids(vocab, T=12, seed=0, B=2):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (B, T))


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1e-30)


# --- SDAR: block-causal --------------------------------------------------------------------------

def _block_mask(T, b):
    blk = torch.arange(T) // b
    return blk[None, :] <= blk[:, None]


def _qwen3(moe):
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
    kw = dict(vocab_size=97, hidden_size=48, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, rope_theta=1e6, attn_implementation="sdpa")
    if moe:   # SDAR-MoE: Qwen3-MoE's softmax top-k, norm_topk_prob, decoder_sparse_step 1
        cfg = Qwen3MoeConfig(**kw, num_experts=6, num_experts_per_tok=2, moe_intermediate_size=24,
                             norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[])
        return _rand_(Qwen3MoeForCausalLM(cfg).to(F64).eval())
    return _rand_(Qwen3ForCausalLM(Qwen3Config(**kw)).to(F64).eval())


@pytest.mark.parametrize("moe", [False, True])
@torch.no_grad()
def test_same_host_carries_sdar_block_causal(moe):
    """SDAR (modeling_sdar.py / modeling_sdar_moe.py, JetLM, remote code READ, not run): SDARAttention is
    Qwen3's (q_norm / k_norm on the head dim, then the rotary) and runs F.scaled_dot_product_attention
    (attn_mask=attention_mask.bool(), is_causal=False): the caller's mask is the whole mask. The decoder layer,
    MLP / SDARMoeSparseMoeBlock (softmax -> top-k -> renorm), norms and rotary are Qwen3 / Qwen3-MoE's. So the
    reference is the STOCK Qwen3 (-MoE) class (sdpa) given SDAR's block-causal mask as a 4-D bool
    attention_mask, the same call SDAR makes. The host (from_sdar) runs it from its block_size buffer, and any
    caller mask through forward(mask=)."""
    from elephantine.universal import from_sdar
    t = _qwen3(moe)
    T, b = 12, 4
    x = _ids(97, T)
    m = _block_mask(T, b)
    ref = t(x, attention_mask=m[None, None].expand(2, 1, T, T)).logits
    causal = t(x).logits
    assert _rel(ref, causal) > 1e-3                      # the block mask binds
    with pytest.raises(ValueError):                      # SDAR's code / configs carry no block size: no guess
        from_sdar(t)
    u = from_sdar(t, block_size=b)
    assert int(u.block_size) == b
    assert _rel(ref, u(x)) < TOL, _rel(ref, u(x))
    # a caller-supplied mask (per batch row): block 3 for row 0, an arbitrary self-visible mask for row 1
    torch.manual_seed(1)
    r = (torch.rand(T, T) < 0.5) | torch.eye(T, dtype=torch.bool)
    mm = torch.stack([_block_mask(T, 3), r])
    ref2 = t(x, attention_mask=mm[:, None]).logits
    assert _rel(ref2, u(x, mask=mm)) < TOL
    # block size 1 is the causal (AR) mask; causal=False (no block) is the bidirectional forward
    assert _rel(causal, u(x, block_size=1)) < TOL
    full = t(x, attention_mask=torch.ones(2, 1, T, T, dtype=torch.bool)).logits
    assert _rel(full, u(x, causal=False)) < TOL


# --- Ling-2.5 / 2.6 (BailingMoeV2_5): REFERENCE transcription ---------------------------------------------
# REFERENCE, not the remote code: a transcription of inclusionAI/Ling-2.6-flash modeling_bailing_moe_v2_5.py
# (md5 924a6ed1..., read as text, never imported or run): BailingMoeV2_5RMSNorm / GroupRMSNorm (l. 168-204), the
# rotary tables (l. 210-241, 538-571; transformers' default rope init: dim = head_dim * partial_rotary_factor),
# apply_rotary_pos_emb / _interleave (l. 253-285, 500-535), BailingMoeV2_5MLP / Gate / SparseMoeBlock (l. 288-441,
# eval moe_infer = the weighted sum over the top-k), the MLA (l. 580-702, eager attention), the linear attention
# (l. 705-899: fla chunk / fused_recurrent_simple_gla = fla's naive_recurrent_simple_gla, scale K^-1/2), the decoder
# layer (l. 974-1097) and the model forward (l. 1204-1380, the causal 4-D mask). MTP layers are not in the forward.
import math as _math
from types import SimpleNamespace as _NS

import torch.nn.functional as _F
from torch import nn as _nn


class _RMS(_nn.Module):
    def __init__(self, n, eps=1e-6, groups=None):
        super().__init__()
        self.weight, self.variance_epsilon, self.group_norm_size = _nn.Parameter(torch.ones(n)), eps, groups

    def forward(self, x):
        if self.group_norm_size is None:
            return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        s = x.shape
        g = x.view(*s[:-1], self.group_norm_size, s[-1] // self.group_norm_size)
        return self.weight * (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)).view(s)


class _Rot(_nn.Module):
    def __init__(self, dim, theta):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(F64) / dim)))
        self.attention_scaling = 1.0

    def forward(self, T):
        fr = torch.arange(T, dtype=F64)[:, None] * self.inv_freq[None]
        emb = torch.cat((fr, fr), -1)
        return emb.cos()[None], emb.sin()[None]


def _rot_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), -1)


def _apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    r = cos.shape[-1]
    qe = torch.cat([q[..., :r] * cos + _rot_half(q[..., :r]) * sin, q[..., r:]], -1)
    ke = torch.cat([k[..., :r] * cos + _rot_half(k[..., :r]) * sin, k[..., r:]], -1)
    return qe, ke


def _apply_rope_interleave(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return q * cos + _rot_half(q) * sin, k * cos + _rot_half(k) * sin


class _MLP(_nn.Module):
    def __init__(self, d, f, limit=None):
        super().__init__()
        self.gate_proj, self.up_proj = _nn.Linear(d, f, bias=False), _nn.Linear(d, f, bias=False)
        self.down_proj, self.swiglu_limit = _nn.Linear(f, d, bias=False), limit

    def forward(self, x):
        g, u = _F.silu(self.gate_proj(x)), self.up_proj(x)
        if self.swiglu_limit is not None:              # Ling-3.1 (BailingMoeV3MLP)
            g, u = g.clamp(max=self.swiglu_limit), u.clamp(-self.swiglu_limit, self.swiglu_limit)
        return self.down_proj(g * u)


class _Gate(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.top_k, self.num_experts, self.n_group, self.topk_group = c.num_experts_per_tok, c.num_experts, c.n_group, c.topk_group
        self.weight = _nn.Parameter(torch.randn(c.num_experts, c.hidden_size) * 0.1)
        self.routed_scaling_factor = c.routed_scaling_factor
        self.register_buffer("expert_bias", torch.zeros(c.num_experts))

    def forward(self, x):
        x = x.view(-1, x.shape[-1])
        logits = _F.linear(x, self.weight)
        scores = torch.sigmoid(logits)
        s4r = scores + self.expert_bias
        n = s4r.shape[0]
        gs = s4r.view(n, self.n_group, -1).topk(2, dim=-1)[0].sum(-1)
        gi = torch.topk(gs, k=self.topk_group, dim=-1, sorted=False)[1]
        gm = torch.zeros_like(gs).scatter_(1, gi, 1)
        sm = gm.unsqueeze(-1).expand(n, self.n_group, self.num_experts // self.n_group).reshape(n, -1)
        _, idx = torch.topk(s4r.masked_fill(~sm.bool(), float("-inf")), k=self.top_k, dim=-1)
        sc = torch.gather(scores, 1, idx)
        w = sc / (sc.sum(-1, keepdim=True) + 1e-20) if self.top_k > 1 else sc
        return idx, w * self.routed_scaling_factor


class _MoE(_nn.Module):
    def __init__(self, c, lim=None, slim=None):
        super().__init__()
        self.expert_swiglu_limit, self.share_expert_swiglu_limit = lim, slim
        self.experts = _nn.ModuleList(_MLP(c.hidden_size, c.moe_intermediate_size, lim) for _ in range(c.num_experts))
        self.gate = _Gate(c)
        self.shared_experts = _MLP(c.hidden_size, c.moe_shared_intermediate_size * c.num_shared_experts, slim)

    def forward(self, x):
        b, t, h = x.shape
        idx, w = self.gate(x)
        xf = x.reshape(-1, h)
        y = torch.zeros_like(xf)
        for j in range(idx.shape[1]):
            for e in range(len(self.experts)):
                sel = idx[:, j] == e
                if sel.any():
                    y[sel] += w[sel, j, None] * self.experts[e](xf[sel])
        return y.view(b, t, h) + self.shared_experts(x)


class _MLA(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.num_heads, self.qk_nope_head_dim, self.qk_rope_head_dim = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim
        self.kv_lora_rank, self.v_head_dim, self.qk_head_dim = c.kv_lora_rank, c.v_head_dim, c.qk_nope_head_dim + c.qk_rope_head_dim
        d, H = c.hidden_size, c.num_attention_heads
        if c.q_lora_rank:
            self.q_a_proj = _nn.Linear(d, c.q_lora_rank, bias=False)
            self.q_a_layernorm = _RMS(c.q_lora_rank)
            self.q_b_proj = _nn.Linear(c.q_lora_rank, H * self.qk_head_dim, bias=False)
        else:
            self.q_proj = _nn.Linear(d, H * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = _nn.Linear(d, c.kv_lora_rank + c.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = _RMS(c.kv_lora_rank)
        self.kv_b_proj = _nn.Linear(c.kv_lora_rank, H * (c.qk_nope_head_dim + c.v_head_dim), bias=False)
        gt = getattr(c, "gated_attention_proj_granularity_type", None)      # BailingMoeV3 (l. 644-649)
        self.g_proj = None if gt is None else _nn.Linear(d, H if gt == "head_wise" else H * c.v_head_dim, bias=False)
        self.dense = _nn.Linear(H * c.v_head_dim, d, bias=False)
        self.scaling = self.qk_head_dim ** -0.5

    def forward(self, x, pe, mask):
        B, T = x.shape[:2]
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))) if hasattr(self, "q_a_proj") else self.q_proj(x)
        q = q.view(B, T, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], -1)
        ckv = self.kv_a_proj_with_mqa(x)
        k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], -1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(B, T, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_pass, v = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], -1)
        k_rot = k_rot.view(B, 1, T, self.qk_rope_head_dim)
        q_rot, k_rot = _apply_rope_interleave(q_rot, k_rot, *pe)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        q, k = torch.cat((q_pass, q_rot), -1), torch.cat((k_pass, k_rot), -1)
        w = (q @ k.transpose(2, 3)) * self.scaling + mask
        o = (w.softmax(-1) @ v).transpose(1, 2)
        if self.g_proj is not None:
            g = torch.sigmoid(self.g_proj(x))
            o = o * (g[..., None] if g.shape[-1] == self.num_heads else g.view(B, T, self.num_heads, self.v_head_dim))
        return self.dense(o.reshape(B, T, -1))


def _slopes(n):
    def p2(n):
        start = 2 ** (-(2 ** -(_math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]
    if _math.log2(n).is_integer():
        return p2(n)
    c = 2 ** _math.floor(_math.log2(n))
    return p2(c) + _slopes(2 * c)[0::2][: n - c]


class _Lightning(_nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        d, H, hd = c.hidden_size, c.num_attention_heads, c.head_dim
        self.num_heads, self.head_dim, self.num_key_value_heads = H, hd, H
        self.query_key_value = _nn.Linear(d, 3 * H * hd, bias=c.use_qkv_bias)
        self.query_layernorm, self.key_layernorm = _RMS(hd, c.rms_norm_eps), _RMS(hd, c.rms_norm_eps)
        self.dense = _nn.Linear(H * hd, d, bias=c.use_bias)
        self.g_proj = _nn.Linear(d, H * hd, bias=False)
        self.g_norm = _RMS(H * hd, c.rms_norm_eps, groups=c.group_norm_size)
        slope = -torch.tensor(_slopes(H), dtype=torch.float) * (1 - (layer_idx - 1) / (c.num_hidden_layers - 1) + 1e-5)
        self.register_buffer("slope", slope.to(F64))

    def forward(self, x, pe):
        B, T, _ = x.shape
        qkv = self.query_key_value(x).view(B, T, 3 * self.num_heads, self.head_dim)
        q, k, v = qkv.split([self.num_heads] * 3, dim=-2)
        q, k = self.query_layernorm(q), self.key_layernorm(k)
        q, k = _apply_rope(q, k, *pe, unsqueeze_dim=2)
        # fla naive_recurrent_simple_gla (the kernels' reference): S <- exp(g) S + k^T v, o = (q K^-1/2) S
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q = q * self.head_dim ** -0.5
        S = q.new_zeros(B, self.num_heads, self.head_dim, self.head_dim)
        o = torch.zeros_like(v)
        g = self.slope.exp()[None, :, None, None]
        for i in range(T):
            S = S * g + k[:, :, i, :, None] * v[:, :, i, None, :]
            o[:, :, i] = (q[:, :, i, :, None] * S).sum(-2)
        o = self.g_norm(o.transpose(1, 2).reshape(B, T, -1)) * torch.sigmoid(self.g_proj(x))
        return self.dense(o)


class _Layer(_nn.Module):
    def __init__(self, c, i, linear_cls):
        super().__init__()
        G = c.layer_group_size
        self.kind = "attention" if (i + 1) % G == 0 or i >= c.num_hidden_layers // G * G else "linear_attention"
        self.attention = _MLA(c) if self.kind == "attention" else linear_cls(c, i)
        lims = [(getattr(c, n, None) or [None] * (i + 1))[i] or None
                for n in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list")]
        self.mlp = _MoE(c, *lims) if i >= c.first_k_dense_replace else _MLP(c.hidden_size, c.intermediate_size)
        self.input_layernorm, self.post_attention_layernorm = _RMS(c.hidden_size, c.rms_norm_eps), _RMS(c.hidden_size, c.rms_norm_eps)

    def forward(self, x, pe, pe_mla, mask):
        h = self.input_layernorm(x)
        x = x + (self.attention(h, pe_mla, mask) if self.kind == "attention" else self.attention(h, pe))
        return x + self.mlp(self.post_attention_layernorm(x))


class _Ling(_nn.Module):
    """REFERENCE BailingMoeV2_5ForCausalLM / BailingMoeV3ForCausalLM (see the headers); .model / .lm_head / .config
    named as the remote classes."""
    def __init__(self, c, linear_cls):
        super().__init__()
        m = _nn.Module()
        m.word_embeddings = _nn.Embedding(c.vocab_size, c.hidden_size)
        m.layers = _nn.ModuleList(_Layer(c, i, linear_cls) for i in range(c.num_hidden_layers))
        m.norm = _RMS(c.hidden_size, c.rms_norm_eps)
        # V2.5: transformers' default rope dim head_dim * partial_rotary_factor; V3 (l. 249-282): qk_rope_head_dim
        dim = int(c.head_dim * c.partial_rotary_factor) if linear_cls is _Lightning else c.qk_rope_head_dim
        m.rotary_emb = _Rot(dim, c.rope_theta)
        if linear_cls is _Lightning:
            m.rotary_emb_mla = _Rot(dim, c.rope_theta)
        self.model, self.config = m, c
        self.lm_head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, ids):
        m, T = self.model, ids.shape[1]
        x = m.word_embeddings(ids)
        mask = torch.full((T, T), torch.finfo(F64).min, dtype=F64).triu(1)[None, None]
        pe = m.rotary_emb(T)
        pe_mla = m.rotary_emb_mla(T) if hasattr(m, "rotary_emb_mla") else pe
        for l in m.layers:
            x = l(x, pe, pe_mla, mask)
        return self.lm_head(m.norm(x))


def _ling_cfg(**kw):
    c = dict(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16, moe_shared_intermediate_size=16,
             num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=4, head_dim=8, partial_rotary_factor=0.5,
             rope_theta=6e6, q_lora_rank=12, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
             rope_interleave=True, num_experts=8, num_experts_per_tok=2, n_group=4, topk_group=2,
             routed_scaling_factor=2.5, num_shared_experts=1, first_k_dense_replace=1, layer_group_size=4,
             group_norm_size=2, use_qkv_bias=False, use_bias=False, rms_norm_eps=1e-6, use_qk_norm=True,
             model_type="bailing_hybrid")
    c.update(kw)
    return _NS(**c)


def _ling_rand(t):
    _rand_(t, s=0.2)
    with torch.no_grad():
        for l in t.model.layers:
            if hasattr(l.mlp, "gate"):
                l.mlp.gate.expert_bias.normal_(0, 0.3)
    return t


@torch.no_grad()
def test_same_host_carries_ling25_lightning_mla():
    """Ling-2.5 / 2.6 (bailing_hybrid, BailingMoeV2_5) vs the REFERENCE transcription above: Lightning (simple-GLA)
    layers with per-head qk RMSNorm, partial rotate-half RoPE, per-head constant decays, GroupRMSNorm * sigmoid
    gate; interleaved-RoPE MLA with a q LoRA on every 4th layer and the trailing one; dense first layer, then the
    sigmoid group-limited MoE with expert-bias selection and a shared expert."""
    from elephantine.universal import from_bailing_hybrid
    t = _ling_rand(_Ling(_ling_cfg(), _Lightning).to(F64).eval())
    assert [l.kind for l in t.model.layers] == ["linear_attention"] * 3 + ["attention", "attention"]
    x = _ids(97, T=12)
    ref, ours = t(x), from_bailing_hybrid(t)(x)
    assert _rel(ref, ours) < 1e-12, _rel(ref, ours)


# --- Ling-3.x (BailingMoeV3): REFERENCE transcription of its KDA layer ----------------------------------------
# REFERENCE, not the remote code: inclusionAI/Ling-3.1-flash modeling_bailing_moe_v3.py (md5 f88b1bf8..., read as
# text): BailingMoeV3KimiDeltaAttention (l. 743-945; fla's ShortConvolution = a depthwise causal conv + silu, no bias;
# chunk_kda / fused_recurrent_kda with use_gate_in_kernel and lower_bound = the lower-bound gate
# lower * sigmoid(exp(A_log) (g + dt_bias)) (fla 0.5.2 ops/kda/gate.py naive_kda_lowerbound_gate) and the recurrence
# of fla's naive_recurrent_kda, l2-normed q / k (use_qk_l2norm_in_kernel), scale K^-1/2; FusedRMSNormGated with a
# sigmoid gate), the MLA's head-wise sigmoid output gate (l. 644-649, 730-736), the per-layer SwiGLU limits that
# clamp after the activation (l. 329-348, 420-450). Everything else is shared with BailingMoeV2_5 (above).
class _Conv(_nn.Module):
    def __init__(self, n, K):
        super().__init__()
        self.weight = _nn.Parameter(torch.randn(n, 1, K) * 0.3)

    def forward(self, x):                              # (B, T, n)
        K = self.weight.shape[-1]
        y = _F.conv1d(_F.pad(x.transpose(1, 2), (K - 1, 0)), self.weight, groups=x.shape[-1])
        return _F.silu(y).transpose(1, 2)


class _KDA(_nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        d, H, hd = c.hidden_size, c.num_attention_heads, c.head_dim
        n = H * hd
        self.num_heads, self.head_dim, self.safe_gate, self.lower_bound = H, hd, c.kda_safe_gate, c.kda_lower_bound
        self.q_proj, self.k_proj, self.v_proj = (_nn.Linear(d, n, bias=False) for _ in range(3))
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (_Conv(n, c.short_conv_kernel_size) for _ in range(3))
        self.A_log = _nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        self.f_proj = _nn.Linear(d, n, bias=False)                     # no_kda_lora
        self.dt_bias = _nn.Parameter(torch.randn(n) * 0.5)
        self.b_proj = _nn.Linear(d, H, bias=False)
        self.g_proj = _nn.Linear(d, n, bias=False)
        self.o_norm = _RMS(hd, c.rms_norm_eps)
        self.o_norm.eps = c.rms_norm_eps
        self.o_proj = _nn.Linear(n, d, bias=False)

    def forward(self, x, pe=None):
        B, T, _ = x.shape
        H, hd = self.num_heads, self.head_dim
        q, k, v = (cv(p(x)).view(B, T, H, hd) for cv, p in ((self.q_conv1d, self.q_proj), (self.k_conv1d, self.k_proj),
                                                            (self.v_conv1d, self.v_proj)))
        g = self.f_proj(x).view(B, T, H, hd) + self.dt_bias.view(H, hd)
        g = self.lower_bound * torch.sigmoid(self.A_log.exp().view(H, 1) * g)
        beta = torch.sigmoid(self.b_proj(x))
        l2 = lambda t: t / torch.sqrt(t.pow(2).sum(-1, keepdim=True) + 1e-6)   # fla l2norm (eps 1e-6)
        q, k = l2(q) * hd ** -0.5, l2(k)
        S = q.new_zeros(B, H, hd, hd)
        o = torch.zeros_like(v)
        for i in range(T):                                              # naive_recurrent_kda
            S = S * g[:, i, ..., None].exp()
            S = S + torch.einsum("bhk,bhv->bhkv", beta[:, i, :, None] * k[:, i], v[:, i] - (k[:, i, ..., None] * S).sum(-2))
            o[:, i] = torch.einsum("bhk,bhkv->bhv", q[:, i], S)
        o = self.o_norm(o) * torch.sigmoid(self.g_proj(x).view(B, T, H, hd))
        return self.o_proj(o.reshape(B, T, -1))


@torch.no_grad()
def test_same_host_carries_ling3_kda_gated_mla_swiglu_limit():
    """Ling-3.1 (bailing_hybrid, BailingMoeV3) vs the REFERENCE transcription above: KDA layers (separate silu short
    convs on q / k / v, lower-bound safe gate, full-rank forget / output gates), MLA with a q LoRA and a head-wise
    sigmoid output gate, per-layer SwiGLU limits on the routed and shared experts (the last layers), sigmoid
    group-limited MoE with expert-bias selection."""
    from elephantine.universal import from_bailing_hybrid
    c = _ling_cfg(kda_safe_gate=True, kda_lower_bound=-5.0, short_conv_kernel_size=4,
                  gated_attention_proj_granularity_type="head_wise",
                  expert_swiglu_limit_list=[0, 0, 0, 0.3, 0.2], share_expert_swiglu_limit_list=[0, 0, 0, 0.25, 0.3])
    t = _ling_rand(_Ling(c, _KDA).to(F64).eval())
    x = _ids(97, T=12)
    ref, ours = t(x), from_bailing_hybrid(t)(x)
    assert _rel(ref, ours) < 1e-12, _rel(ref, ours)
    t.model.layers[4].mlp.expert_swiglu_limit = None                   # the limits bind
    for e in t.model.layers[4].mlp.experts:
        e.swiglu_limit = None
    assert _rel(t(x), ref) > 1e-6


import os  # noqa: E402

TF522 = os.environ.get("TF522_SRC",                                    # transformers 5.22.2, unpacked (uv / pip)
                       os.path.expanduser("~/.cache/upstream_refs/transformers-5.22.2"))


def test_same_host_carries_ling3_stock_bailing_hybrid():
    """Ling-3.x in its STOCK transformers class (BailingHybridForCausalLM, transformers 5.22.2, a subprocess like
    _tf5_run; huggingface_hub 1.33 from the 5.18 unpack): KDA (Glm5Next's linear attention with the bounded forget
    gate, torch fallbacks), q-projection MLA with the head-wise gate, DeepSeek-V3 MoE."""
    import os, re, subprocess, sys
    from tests.test_universal_elephantine import TF5, TF5_PRELUDE
    if not (os.path.isdir(TF522) and os.path.isdir(TF5)):
        pytest.skip("transformers 5.22.2 / 5.18 not unpacked")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    body = '''
from transformers.models.bailing_hybrid import modeling_bailing_hybrid as bh, configuration_bailing_hybrid as bc
from elephantine.universal import from_bailing_hybrid
cfg = bc.BailingHybridConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
    num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=4, kv_lora_rank=16, qk_rope_head_dim=4,
    qk_nope_head_dim=8, v_head_dim=8, head_dim=8, n_group=4, topk_group=2, num_local_experts=8, num_experts_per_tok=2,
    n_shared_experts=1, first_k_dense_replace=1, short_conv_kernel_size=4, layer_group_size=2, num_mtp_layers=0,
    rms_norm_eps=1e-6, pad_token_id=0, eos_token_id=1, attn_implementation="eager", experts_implementation="eager")
t = rand_(bh.BailingHybridForCausalLM(cfg).to(F64).eval())
with torch.no_grad():
    for l in t.model.layers:
        g = getattr(l.mlp, "gate", None)
        if g is not None and hasattr(g, "e_score_correction_bias"):
            g.e_score_correction_bias.copy_(torch.randn_like(g.e_score_correction_bias) * 0.3)
    print("TYPES", cfg.layer_types, cfg.mlp_layer_types)
    x = ids(97, T=12)
    report(t(x).logits, from_bailing_hybrid(t)(x))
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF522, TF5, root]))
    r = subprocess.run([sys.executable, "-c", TF5_PRELUDE + body], env=env, capture_output=True, text=True, timeout=1800)
    m = re.search(r"GAP (\S+) SCALE (\S+)", r.stdout)
    assert m, r.stdout[-2000:] + r.stderr[-4000:]
    gap, scale = float(m.group(1)), float(m.group(2))
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


# --- PLaMo-2: the attention window ------------------------------------------------------------------------
# REFERENCE, not the remote code: pfnet/plamo-2-1b modeling_plamo.py (read as text): RotaryEmbedding (l. 48-83),
# _rotate_half / _rotary_pos_emb (l. 86-100), _rms_norm (l. 426-436), swa_mask (l. 1028-1035) and Attention.forward's
# masked path (l. 1068-1157; the model always passes a mask in eval, l. 1446-1458), a sliding (not full_attn) layer.
class _PlamoAttn(_nn.Module):
    def __init__(self, c, full):
        super().__init__()
        self.config, self.full_attn = c, full
        hd = c.hidden_size_per_head
        self.q_num_heads, self.qk_dim, self.v_dim, self.k_num_heads = c.num_attention_heads, hd, hd, c.num_key_value_heads
        self.n_group = self.q_num_heads // self.k_num_heads
        self.q_proj_dim, self.k_proj_dim = self.q_num_heads * hd, self.k_num_heads * hd
        self.v_proj_dim = self.k_proj_dim
        self.qkv_proj = _nn.Linear(c.hidden_size, self.q_proj_dim + 2 * self.k_proj_dim, bias=False)
        self.o_proj = _nn.Linear(self.q_num_heads * hd, c.hidden_size, bias=False)
        self.q_weight = _nn.Parameter(torch.ones(self.q_num_heads, hd))
        self.k_weight = _nn.Parameter(torch.ones(self.k_num_heads, hd))
        self.rotary_emb = _nn.Module()
        base = c.rope_theta if full else c.rope_local_theta
        self.rotary_emb.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, hd, 2).to(F64) / hd)))

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = torch.split(self.qkv_proj(x), [self.q_proj_dim, self.k_proj_dim, self.v_proj_dim], -1)
        q = q.view(B, T, self.q_num_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.k_num_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.k_num_heads, -1).transpose(1, 2)
        rms = lambda t: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
        q, k = rms(q) * self.q_weight[None, :, None], rms(k) * self.k_weight[None, :, None]
        fr = torch.arange(T, dtype=F64)[:, None] * self.rotary_emb.inv_freq[None]
        emb = torch.cat((fr, fr), -1)
        q, k = q * emb.cos() + _rot_half(q) * emb.sin(), k * emb.cos() + _rot_half(k) * emb.sin()
        k, v = k.repeat_interleave(self.n_group, 1), v.repeat_interleave(self.n_group, 1)
        mask = torch.zeros(T, T, dtype=F64).masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))[None, None]
        if not self.full_attn:
            W = self.config.attention_window_size
            m_swa = torch.ones(T, T, dtype=torch.bool).triu(diagonal=-W).tril(diagonal=W)
            mask = torch.where(m_swa[None, None], mask, float("-inf"))
        o = _F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))


@pytest.mark.parametrize("full", [False, True])
@torch.no_grad()
def test_plamo2_attention_window(full):
    """from_plamo2's attention (_plamo2_attn) now sets the sliding window on the non-full layers: T = 12 against
    a window of 3, so the window binds (the whole-model test, window 32 at T = 10, never did)."""
    from elephantine.universal import UAttention, UConfig, _plamo2_attn
    c = _NS(hidden_size=32, hidden_size_per_head=8, num_attention_heads=4, num_key_value_heads=2,
            attention_window_size=3, rope_theta=1e6, rope_local_theta=1e4)
    t = _rand_(_PlamoAttn(c, full).to(F64).eval(), s=0.3)
    a = UAttention(UConfig(vocab=2, d=32, L=1, heads=4, head_dim=8))
    _plamo2_attn(a, t)
    torch.manual_seed(3)
    x = torch.randn(2, 12, 32, dtype=F64)
    ref = t(x)
    assert _rel(ref, a(x, True)) < 1e-12
    a.window.fill_(float("inf"))
    assert (_rel(ref, a(x, True)) > 1e-3) != full                      # the window binds on sliding layers only


# --- RWKV-4: the whole model --------------------------------------------------------------------------------

@torch.no_grad()
def test_same_host_carries_rwkv4_whole_model():
    """RWKV-4 (stock RwkvForCausalLM): embedding LayerNorm (pre_ln, layer 0), per block LayerNorm -> time-mix (the
    SSM sub-block, rwkv4_block) and LayerNorm -> channel-mix  W_v relu(W_k xk)^2 * sigmoid(W_r xr)  on token-shifted
    mixes, final LayerNorm, untied head."""
    from transformers import RwkvConfig, RwkvForCausalLM
    from elephantine.universal import from_rwkv
    cfg = RwkvConfig(vocab_size=97, hidden_size=32, attention_hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                     context_length=64, rescale_every=0)
    t = RwkvForCausalLM(cfg).to(F64).eval()
    torch.manual_seed(0)
    for n, p in t.named_parameters():
        if "time_decay" in n or "time_first" in n:
            p.copy_(torch.randn_like(p) * 0.5)
        elif "time_mix" in n:
            p.copy_(torch.rand_like(p))
        else:
            p.add_(torch.randn_like(p) * 0.1)
    x = _ids(97, T=12)
    ref = t(x, use_cache=False).logits
    assert _rel(ref, from_rwkv(t)(x)) < TOL, _rel(ref, from_rwkv(t)(x))


# --- IQuest-Q1: the learned sink KEY ---------------------------------------------------------------------------
# REFERENCE, not the remote code: IQuestLab/vllm-iquest-q1 (read as text, verify/r3/specs/sink_key_query_dependent.md):
# model.py:354-365 (sink_k (num_kv_heads, head_dim) with enable_sink_attention), :393-414 (q_norm / k_norm per head,
# then RoPE on q and k; the sink key is not rotated), attention.py:66-89 (per query head h, its KV group's s:
# sink_logit = (q . s) * SCALE, out *= 1 / (1 + exp(sink_logit - lse)) -- softmax over [keys, s] with s's value 0;
# test_sink_attention.py:271-272: softmax(cat([q k^T, q . s]))). Full rotary here (rotary_dim = head_dim).
class _IQuestAttn(_nn.Module):
    def __init__(self, d, H, Hk, hd, theta=1e6, sink=True):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim, self.scaling = H, Hk, hd, hd ** -0.5
        self.q_proj, self.k_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(d, Hk * hd, bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, Hk * hd, bias=False), _nn.Linear(H * hd, d, bias=False)
        self.q_norm, self.k_norm = _RMS(hd), _RMS(hd)
        self.rotary_emb = _Rot(hd, theta)
        self.sink_k = _nn.Parameter(torch.zeros(Hk, hd)) if sink else None

    def forward(self, x):
        B, T, _ = x.shape
        H, Hk, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q = self.q_norm(self.q_proj(x).view(B, T, H, hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, Hk, hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hk, hd).transpose(1, 2)
        cos, sin = self.rotary_emb(T)
        q, k = q * cos[:, None] + _rot_half(q) * sin[:, None], k * cos[:, None] + _rot_half(k) * sin[:, None]
        g = H // Hk
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        s = q @ k.transpose(-1, -2) * self.scaling
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        if self.sink_k is not None:
            sk = self.sink_k.repeat_interleave(g, 0)                       # query head h -> its KV group's s
            s = torch.cat((s, (q @ sk[:, :, None]) * self.scaling), -1)    # (B, H, T, 1), value 0
        o = s.softmax(-1)[..., :T] @ v
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))


@torch.no_grad()
def test_iquest_q1_sink_key():
    """The learned sink key (minimal_emulator host_point sink 'key'): UAttention.null_key under _iquest_attn equals the
    REFERENCE to fp64 rounding, and the sink binds (zeroing it, or the plain host without it, moves the output)."""
    from elephantine.universal import UAttention, UConfig, _iquest_attn
    t = _rand_(_IQuestAttn(32, 4, 2, 8).to(F64).eval(), s=0.3)
    with torch.no_grad():
        t.sink_k.copy_(torch.randn(2, 8, dtype=F64) * 2.0)
    a = UAttention(UConfig(vocab=2, d=32, L=1, heads=4, head_dim=8, sink_key=True))
    _iquest_attn(a, t)
    torch.manual_seed(5)
    x = torch.randn(2, 10, 32, dtype=F64)
    ref = t(x)
    assert _rel(ref, a(x, True)) < 1e-12
    a.null_key.zero_()                               # a zero key is a logit-0 scalar sink, not no sink
    assert _rel(ref, a(x, True)) > 1e-3
    plain = UAttention(UConfig(vocab=2, d=32, L=1, heads=4, head_dim=8))
    t.sink_k = None
    _iquest_attn(plain, t)
    assert _rel(ref, plain(x, True)) > 1e-3 and _rel(t(x), plain(x, True)) < 1e-12


# --- fla DeltaNet: the ungated delta rule ----------------------------------------------------------------------
# REFERENCE, not run from the library: fla 0.4.0 (flash-linear-attention, ~/.cache/upstream_refs) layers/delta_net.py
# DeltaNet.__init__ / forward (l. 73-290; use_gate False, use_short_conv, qk_activation silu, qk_norm l2, the snapshot's
# fla-hub/delta_net-1.3B-100B settings), ShortConvolution (modules/convolution.py:794: a depthwise causal Conv1d, padding
# kernel_size - 1, then silu), l2norm_fwd (modules/l2norm.py: x / sqrt(sum x^2 + 1e-6)) and the recurrence of
# ops/delta_rule/naive.py delta_rule_recurrence (q x d_k^-0.5; v <- beta (v - S^T k); S <- S + k v^T; o = q S), in fp64.
class _ShortConv(_nn.Conv1d):
    def __init__(self, n, K, bias=False):
        super().__init__(n, n, K, groups=n, padding=K - 1, bias=bias)

    def forward(self, x):                                                 # (B, T, n)
        return _F.silu(super().forward(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))


class _DeltaNet(_nn.Module):
    def __init__(self, d, H, K=4, neg=False, eps=1e-6):
        super().__init__()
        self.hidden_size, self.num_heads, self.head_k_dim, self.head_v_dim = d, H, d // H, d // H
        self.use_gate, self.use_beta, self.use_short_conv, self.allow_neg_eigval = False, True, True, neg
        self.qk_activation, self.qk_norm = "silu", "l2"
        self.q_proj, self.k_proj, self.v_proj = (_nn.Linear(d, d, bias=False) for _ in range(3))
        self.b_proj = _nn.Linear(d, H, bias=False)
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (_ShortConv(d, K) for _ in range(3))
        self.o_norm = _RMS(d // H, eps)
        self.o_norm.eps = eps
        self.o_proj = _nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q, k, v = self.q_conv1d(self.q_proj(x)), self.k_conv1d(self.k_proj(x)), self.v_conv1d(self.v_proj(x))
        q, k, v = (t.view(B, T, H, -1).transpose(1, 2) for t in (q, k, v))           # (B, H, T, n)
        l2 = lambda t: t / torch.sqrt(t.pow(2).sum(-1, keepdim=True) + 1e-6)
        q, k = l2(q) * dk ** -0.5, l2(k)
        beta = self.b_proj(x).sigmoid().transpose(1, 2) * (2.0 if self.allow_neg_eigval else 1.0)   # (B, H, T)
        S = torch.zeros(B, H, dk, dv, dtype=x.dtype)
        o = torch.zeros(B, H, T, dv, dtype=x.dtype)
        for i in range(T):
            _k, _v = k[:, :, i], v[:, :, i] - (S * k[:, :, i, :, None]).sum(-2)
            S = S + _k[..., None] * (_v * beta[:, :, i, None])[..., None, :]
            o[:, :, i] = torch.einsum("bhd,bhdm->bhm", q[:, :, i], S)
        o = self.o_norm(o.transpose(1, 2))                                          # (B, T, H, dv)
        return self.o_proj(o.reshape(B, T, -1))


@pytest.mark.parametrize("neg", [False, True])
@torch.no_grad()
def test_deltanet_block(neg):
    """fla DeltaNet (minimal_emulator FAMILY_STATUS 'deltanet'): the SSM block under _deltanet_from equals the REFERENCE
    to fp64 rounding, with and without allow_neg_eigval; the delta rule binds (beta's erase off moves the output)."""
    from elephantine.universal import UConfig, USSM, _deltanet_from
    d, H, K = 32, 4, 4
    t = _rand_(_DeltaNet(d, H, K, neg).to(F64).eval(), s=0.3)
    s = USSM(UConfig(vocab=2, d=d, L=1, ssm_in=3 * d + H, conv_k=K, ssm_heads=H, dk=d // H, dv=d // H))
    _deltanet_from(s, t)
    torch.manual_seed(7)
    x = torch.randn(2, 12, d, dtype=F64)
    ref = t(x)
    assert _rel(ref, s(x)) < 1e-12
    s.cg.zero_()                                       # no erase: linear attention, not the delta rule
    assert _rel(ref, s(x)) > 1e-3


# --- GPT-X3: GPT-S3's GDN-2 and gated GQA on a plain residual stream -------------------------------------------
# REFERENCE, not the remote code: AxiomicLabs/GPT-X3-150M modeling_gptx3.py (snapshot 541bdc16, read as text):
# RMSNorm (l. 50-60, offset = (1 + w) for q_norm / k_norm), rotary (l. 63-67: pairs (2i, 2i + 1) on the first
# rotary_dim dims), GPTX3Attention (l. 70-105: q / gate from one projection, offset qk-norm, RoPE, causal SDPA with GQA,
# y * sigmoid(gate)), GPTX3GDN2Kernel's PyTorch path (l. 108-223: silu short convs, low-rank f_proj / g_proj, the
# per-key-channel decay -exp(A_log) softplus(f + dt_bias), l2 q / k (+1e-6), q * dim^-0.5, erase S^T (b * k), write
# w * v, sigmoid-gated per-head RMSNorm), GPTX3MLP / GPTX3Block (l. 235-258) and the forward (l. 309-359), in fp64.
class _X3Norm(_nn.Module):
    def __init__(self, n, eps=1e-6, offset=False):
        super().__init__()
        self.eps, self.offset = eps, offset
        self.weight = _nn.Parameter(torch.zeros(n) if offset else torch.ones(n))

    def forward(self, x):
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return y * (1.0 + self.weight) if self.offset else y * self.weight


def _x3_rotary(x, cos, sin):
    dim = cos.shape[-1] * 2
    a, b = x[..., :dim].reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.cat((torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2), x[..., dim:]), -1)


class _X3Attn(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        h, d, kv = c.num_attention_heads, c.head_dim, c.num_key_value_heads
        self.q_gate_proj = _nn.Linear(c.hidden_size, 2 * h * d, bias=False)
        self.k_proj = _nn.Linear(c.hidden_size, kv * d, bias=False)
        self.v_proj = _nn.Linear(c.hidden_size, kv * d, bias=False)
        self.o_proj = _nn.Linear(h * d, c.hidden_size, bias=False)
        self.q_norm, self.k_norm = _X3Norm(d, c.rms_norm_eps, True), _X3Norm(d, c.rms_norm_eps, True)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.config
        q, gate = self.q_gate_proj(x).view(B, T, c.num_attention_heads, 2 * c.head_dim).chunk(2, -1)
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(self.k_proj(x).view(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2))
        v = self.v_proj(x).view(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = _x3_rotary(q, cos, sin), _x3_rotary(k, cos, sin)
        y = _F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        y = y * torch.sigmoid(gate.transpose(1, 2))
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class _X3Kernel(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        h, dim = c.hidden_size, c.gdn_head_dim
        self.q_proj, self.k_proj, self.v_proj = (_nn.Linear(h, h, bias=False) for _ in range(3))
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (_nn.Conv1d(h, h, c.conv_size, groups=h, bias=False)
                                                       for _ in range(3))
        self.f_proj = _nn.Sequential(_nn.Linear(h, dim, bias=False), _nn.Linear(dim, h, bias=False))
        self.b_proj, self.w_proj = _nn.Linear(h, h, bias=False), _nn.Linear(h, h, bias=False)
        self.A_log, self.dt_bias = _nn.Parameter(torch.zeros(h // dim)), _nn.Parameter(torch.zeros(h))
        self.g_proj = _nn.Sequential(_nn.Linear(h, dim, bias=False), _nn.Linear(dim, h, bias=True))
        self.o_norm = _X3Norm(dim, c.gdn_norm_eps)
        self.o_proj = _nn.Linear(h, h, bias=False)

    def _conv(self, x, conv):
        B, T, C = x.shape
        comb = torch.cat((x.new_zeros(B, C, self.config.conv_size - 1), x.transpose(1, 2)), -1)
        return _F.silu(_F.conv1d(comb, conv.weight, groups=C)).transpose(1, 2)

    def forward(self, x):
        c = self.config
        B, T, _ = x.shape
        dim, heads = c.gdn_head_dim, c.hidden_size // c.gdn_head_dim
        q, k, v = [self._conv(p(x), cv).view(B, T, heads, dim)
                   for p, cv in ((self.q_proj, self.q_conv1d), (self.k_proj, self.k_conv1d),
                                 (self.v_proj, self.v_conv1d))]
        g = _F.softplus(self.f_proj(x) + self.dt_bias).view(B, T, heads, dim)
        g = -self.A_log.exp()[None, None, :, None] * g
        b, w = self.b_proj(x).sigmoid().view_as(q), self.w_proj(x).sigmoid().view_as(v)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        q = q * dim ** -0.5
        S = x.new_zeros(B, heads, dim, dim)
        out = []
        for t in range(T):
            dec = S * g[:, t].exp().unsqueeze(-1)
            erase = ((b[:, t] * k[:, t]).unsqueeze(-1) * dec).sum(-2)
            S = dec + k[:, t].unsqueeze(-1) * (w[:, t] * v[:, t] - erase).unsqueeze(-2)
            out.append((q[:, t].unsqueeze(-1) * S).sum(-2))
        o = torch.stack(out, 1)
        gate = self.g_proj(x).view(B, T, heads, dim)
        y = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + c.gdn_norm_eps) * self.o_norm.weight * torch.sigmoid(gate)
        return self.o_proj(y.flatten(-2))


class _X3Block(_nn.Module):
    def __init__(self, c, full):
        super().__init__()
        self.ln_1, self.ln_2 = _X3Norm(c.hidden_size, c.rms_norm_eps), _X3Norm(c.hidden_size, c.rms_norm_eps)
        self.full_attention = full
        if full:
            self.attn = _X3Attn(c)
        else:
            self.attn = _nn.Module()
            self.attn.kernel = _X3Kernel(c)
        self.mlp = _nn.Module()
        self.mlp.w_gate = _nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.mlp.w_up = _nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.mlp.w_down = _nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        h = self.ln_1(x)
        x = x + (self.attn(h, cos, sin) if self.full_attention else self.attn.kernel(h))
        m, h2 = self.mlp, self.ln_2(x)
        return x + m.w_down(_F.silu(m.w_gate(h2)) * m.w_up(h2))


class _GPTX3(_nn.Module):
    def __init__(self, c, kinds):
        super().__init__()
        self.config = c
        self.transformer = _nn.ModuleDict({"wte": _nn.Embedding(c.vocab_size, c.hidden_size),
                                           "h": _nn.ModuleList([_X3Block(c, f) for f in kinds]),
                                           "ln_f": _X3Norm(c.hidden_size, c.rms_norm_eps)})
        self.lm_head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, ids):
        c = self.config
        x = self.transformer["wte"](ids)
        freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.rotary_dim, 2, dtype=F64) / c.rotary_dim))
        ang = torch.outer(torch.arange(ids.shape[1], dtype=F64), freq)
        cos, sin = ang.cos()[None, None], ang.sin()[None, None]
        for blk in self.transformer["h"]:
            x = blk(x, cos, sin)
        return self.lm_head(self.transformer["ln_f"](x))


@torch.no_grad()
def test_same_host_carries_gptx3():
    """GPT-X3 (minimal_emulator FAMILY_STATUS 'gpts3', the GDN-2 family): from_gptx3 equals the REFERENCE whole model
    to fp64 rounding; GDN-2's erase gate binds."""
    from elephantine.universal import from_gptx3
    c = _NS(vocab_size=97, hidden_size=32, gdn_head_dim=16, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            rotary_dim=8, intermediate_size=48, conv_size=4, rms_norm_eps=1e-6, gdn_norm_eps=1e-6, rope_theta=1e4)
    t = _rand_(_GPTX3(c, [False, True, False, True]).to(F64).eval(), s=0.3)
    for b in t.transformer["h"]:
        if not b.full_attention:
            b.attn.kernel.A_log.copy_(torch.randn(2, dtype=F64) * 0.5)
    x = _ids(97, T=12)
    ref = t(x)
    host = from_gptx3(t)
    assert _rel(ref, host(x)) < 1e-12
    for l in host.layers:
        l.ssm.c_er.zero_()
    assert _rel(ref, host(x)) > 1e-4


# --- an attention-only post-norm (post_norm_attn: IQuest-Q1 attn_out_norm, Needle post_attn_norm) -----------------
@torch.no_grad()
def test_post_norm_attention_only():
    """The universal host's per-sub-block post-norms (ULayer.post_coef): the attention output post-normed, the MLP's
    not -- a stock Llama with an RMSNorm on each self_attn output (a forward hook: x + N(attn(n x)); x + mlp(n' x)) is
    reproduced by from_llama + post_a set, post_coef[0] = [0, 1], post_coef[2] = [1, 0] (minimal_emulator HOST_FEATS
    post_norm_attn)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from elephantine.universal import from_llama, _norm_from
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, attn_implementation="eager")
    t = _rand_(LlamaForCausalLM(cfg).to(F64).eval(), s=0.3)
    post = [_rand_(_RMS(32).to(F64), seed=11 + i, s=0.3) for i in range(2)]
    hooks = [l.self_attn.register_forward_hook(lambda m, a, o, n=n: (n(o[0]),) + tuple(o[1:]))
             for l, n in zip(t.model.layers, post)]
    x = _ids(97, T=10)
    ref = t(x).logits
    for h in hooks:
        h.remove()
    host = from_llama(t)
    assert _rel(ref, host(x)) > 1e-3                                  # the post-norm binds
    for ul, n in zip(host.layers, post):
        _norm_from(ul.post_a, n)
        ul.post_coef[0] = torch.tensor([0.0, 1.0], dtype=F64)
        assert ul.post_coef[2].tolist() == [1.0, 0.0]                 # the MLP output stays raw
    assert _rel(ref, host(x)) < TOL                                   # (stock Llama's fp32 norms / rotary table)


# --- query temperatures and constant qk gains (q_temp, qknorm head_free_gain) ----------------------------------
# REFERENCE, not run from the library: transformers main models/mistral_large4/modular_mistral_large4.py (read as text)
# MistralLarge4TextSoftmaxScaling (l. 278-291: q * (scale_h log(pos + 1) + bias_h), applied to the whole query after
# the rotary, l. 449-451); and the weight-free per-head q / k RMSNorm times a config constant c on the logit of
# head_free_gain (graph/model_graph.py QKNORM_VARIANTS: nanochat qk_sharpen_scale^2, Muse-Glimmer qk_scale_factor ...).
class _TempAttn(_nn.Module):
    def __init__(self, d, H, hd, c_gain=None, temp=False, theta=1e4):
        super().__init__()
        self.H, self.hd, self.c_gain = H, hd, c_gain
        self.q_proj, self.k_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(d, H * hd, bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(H * hd, d, bias=False)
        self.rotary_emb = _Rot(hd, theta)
        self.scale = _nn.Parameter(torch.zeros(H)) if temp else None
        self.bias = _nn.Parameter(torch.ones(H)) if temp else None

    def forward(self, x):
        B, T, _ = x.shape
        H, hd = self.H, self.hd
        q, k, v = (p(x).view(B, T, H, hd).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        if self.c_gain is not None:
            n = lambda t: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
            q, k = n(q), n(k)
        cos, sin = self.rotary_emb(T)
        q, k = q * cos[:, None] + _rot_half(q) * sin[:, None], k * cos[:, None] + _rot_half(k) * sin[:, None]
        if self.scale is not None:
            lp = (torch.arange(T, dtype=F64) + 1).log()
            q = q * (self.scale[None, :, None] * lp[None, None, :] + self.bias[None, :, None]).unsqueeze(-1)
        s = q @ k.transpose(-1, -2) * hd ** -0.5 * (self.c_gain if self.c_gain is not None else 1.0)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        return self.o_proj((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1))


@pytest.mark.parametrize("c_gain,temp", [(None, True), (2.5, False), (0.7, True)])
@torch.no_grad()
def test_query_temperature_and_constant_qk_gain(c_gain, temp):
    """UAttention.qt_s / qt_b carry Mistral-Large-4's learned per-head log-position query temperature (FEATURES q_temp),
    and its per-head qk-norm gains carry a weight-free norm times a constant logit gain c (q gain c, k gain 1: qknorm
    head_free_gain); both bind."""
    from elephantine.universal import UAttention, UConfig
    t = _rand_(_TempAttn(32, 4, 8, c_gain, temp).to(F64).eval(), s=0.3)
    if temp:
        t.scale.copy_(torch.randn(4, dtype=F64) * 0.5)
        t.bias.copy_(1 + torch.randn(4, dtype=F64) * 0.3)
    a = UAttention(UConfig(vocab=2, d=32, L=1, heads=4, head_dim=8))
    for dst, src in ((a.q_proj, t.q_proj), (a.k_proj, t.k_proj), (a.v_proj, t.v_proj), (a.o_proj, t.o_proj)):
        dst.weight.copy_(src.weight)
    a.inv_freq.copy_(t.rotary_emb.inv_freq)
    a.scale.fill_(8 ** -0.5)
    if c_gain is not None:
        a.q_coef.copy_(torch.tensor([0.0, 1.0])); a.k_coef.copy_(torch.tensor([0.0, 1.0]))
        a.q_nw.fill_(c_gain); a.k_nw.fill_(1.0); a.qk_eps.fill_(1e-6)
    if temp:
        a.qt_s.copy_(t.scale); a.qt_b.copy_(t.bias)
    torch.manual_seed(9)
    x = torch.randn(2, 11, 32, dtype=F64)
    ref = t(x)
    assert _rel(ref, a(x, True)) < 1e-12
    if temp:
        a.qt_s.zero_()
        assert _rel(ref, a(x, True)) > 1e-3
    else:
        a.q_nw.fill_(1.0)
        assert _rel(ref, a(x, True)) > 1e-3


# --- biased output gates and Beam's code-fixed query temperature (out_gate_bias, out_gate_head_bias, q_temp_nope) --
# REFERENCE (read as text): a biased sigmoid output gate o * sigmoid(W_g x + b) per channel (Beam modular_beam.py) or per
# head (W_g: hidden -> heads, Ethoric), and Beam's NoPE-layer query temperature q * (1 + (pos + 1) / 2048)^0.1
# (modular_beam.py:172-174, :207-208, :420-428).
class _GateAttn(_nn.Module):
    def __init__(self, d, H, hd, per_head, temp):
        super().__init__()
        self.H, self.hd, self.per_head, self.temp = H, hd, per_head, temp
        self.q_proj, self.k_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(d, H * hd, bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(H * hd, d, bias=False)
        self.g_proj = _nn.Linear(d, H if per_head else H * hd, bias=True)

    def forward(self, x):
        B, T, _ = x.shape
        H, hd = self.H, self.hd
        q, k, v = (p(x).view(B, T, H, hd).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        if self.temp:                                                    # a NoPE layer: no rotary
            q = q * ((1 + (torch.arange(T, dtype=F64) + 1) / 2048) ** 0.1)[None, None, :, None]
        s = (q @ k.transpose(-1, -2) * hd ** -0.5).masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2)                          # (B, T, H, hd)
        g = torch.sigmoid(self.g_proj(x))
        o = o * (g[..., None] if self.per_head else g.view(B, T, H, hd))
        return self.o_proj(o.reshape(B, T, -1))


@pytest.mark.parametrize("per_head,temp", [(False, True), (True, False)])
@torch.no_grad()
def test_biased_output_gate_and_power_temperature(per_head, temp):
    """UAttention's g_proj bias carries the biased gates (per channel; per head by repeating each head's row and bias
    over its head_dim), and qt_pow Beam's code-fixed (1 + (t + 1) / 2048)^0.1 query temperature; both bind."""
    from elephantine.universal import UAttention, UConfig
    t = _rand_(_GateAttn(32, 4, 8, per_head, temp).to(F64).eval(), s=0.3)
    a = UAttention(UConfig(vocab=2, d=32, L=1, heads=4, head_dim=8))
    for dst, src in ((a.q_proj, t.q_proj), (a.k_proj, t.k_proj), (a.v_proj, t.v_proj), (a.o_proj, t.o_proj)):
        dst.weight.copy_(src.weight)
    a.inv_freq.zero_()
    a.scale.fill_(8 ** -0.5)
    rep = (lambda w: w.repeat_interleave(8, 0)) if per_head else (lambda w: w)
    a.g_proj.weight.copy_(rep(t.g_proj.weight)); a.g_proj.bias.copy_(rep(t.g_proj.bias))
    a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
    if temp:
        a.qt_pow.copy_(torch.tensor([0.1, 1.0, 2048.0], dtype=F64))
    torch.manual_seed(13)
    x = torch.randn(2, 40, 32, dtype=F64) * 3
    ref = t(x)
    assert _rel(ref, a(x, True)) < 1e-12
    a.g_proj.bias.zero_()
    assert _rel(ref, a(x, True)) > 1e-3
    if temp:
        a.g_proj.bias.copy_(rep(t.g_proj.bias)); a.qt_pow[0] = 0.0
        assert _rel(ref, a(x, True)) > 1e-6


# --- activated output gates, biased or not, per channel or per head (out_gate_[head_][bias_]{silu,softplus}) ----------
# REFERENCE (read as text, not run): K2-Horizon's gated attention (IFM K2-Type / Jais-3 modeling_k2_horizon.py:225-230,
# :312-321): gate = gate_proj(x).view(..., -1, head_dim); gate = F.silu(gate) (attention_gate_func "silu") or
# F.softplus(gate, beta=log 2) ("softplus"); attn_output * gate. The per-head forms (W_g: hidden -> heads, Ethoric /
# Laguna) and the biased forms (W_g with a bias, Beam / Ethoric) are the same gate with a narrower or biased W_g.
class _ActGateAttn(_GateAttn):
    def __init__(self, d, H, hd, per_head, bias, act):
        super().__init__(d, H, hd, per_head, False)
        self.g_proj = _nn.Linear(d, H if per_head else H * hd, bias=bias)
        self.act = act

    def forward(self, x):
        B, T, _ = x.shape
        H, hd = self.H, self.hd
        q, k, v = (p(x).view(B, T, H, hd).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        s = (q @ k.transpose(-1, -2) * hd ** -0.5).masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2)                          # (B, T, H, hd)
        g = self.g_proj(x)
        g = _F.silu(g) if self.act == "silu" else _F.softplus(g, beta=_math.log(2))
        o = o * (g[..., None] if self.per_head else g.view(B, T, H, hd))
        return self.o_proj(o.reshape(B, T, -1))


@pytest.mark.parametrize("act", ["silu", "softplus"])
@pytest.mark.parametrize("per_head,bias", [(False, False), (True, False), (False, True), (True, True)])
@torch.no_grad()
def test_activated_output_gate(act, per_head, bias):
    """UAttention.gate_silu carries the silu output gate (and gate_sp the softplus one, softplus(z, beta) = softplus(beta
    z) / beta: the beta folds into g_proj's rows and the coefficient), per channel or per head (each head's row and bias
    repeated over its head_dim), with or without g_proj's bias; the branch binds."""
    from elephantine.universal import UAttention, UConfig
    t = _rand_(_ActGateAttn(32, 4, 8, per_head, bias, act).to(F64).eval(), s=0.3)
    a = UAttention(UConfig(vocab=2, d=32, L=1, heads=4, head_dim=8))
    for dst, src in ((a.q_proj, t.q_proj), (a.k_proj, t.k_proj), (a.v_proj, t.v_proj), (a.o_proj, t.o_proj)):
        dst.weight.copy_(src.weight)
    a.inv_freq.zero_()
    a.scale.fill_(8 ** -0.5)
    rep = (lambda w: w.repeat_interleave(8, 0)) if per_head else (lambda w: w)
    beta = 1.0 if act == "silu" else _math.log(2)
    a.g_proj.weight.copy_(rep(t.g_proj.weight) * beta)
    a.g_proj.bias.copy_(rep(t.g_proj.bias) * beta if bias else torch.zeros_like(a.g_proj.bias))
    a.gate_coef.copy_(torch.tensor([0.0, 0.0]))
    (a.gate_silu if act == "silu" else a.gate_sp).fill_(1.0 / beta)
    torch.manual_seed(17)
    x = torch.randn(2, 40, 32, dtype=F64) * 3
    ref = t(x)
    assert _rel(ref, a(x, True)) < 1e-12
    (a.gate_silu if act == "silu" else a.gate_sp).fill_(0.0)
    a.gate_coef.copy_(torch.tensor([0.0, 1.0]))                         # the sigmoid gate is not it
    assert _rel(ref, a(x, True)) > 1e-3
