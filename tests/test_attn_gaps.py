"""The round-3 ATTENTION features the universal host now carries (elephantine/branches_attn.py; the branches are
UAttention.add_branches in elephantine/universal.py, each off by default). Every test runs the host against the
target's own code where a stock transformers class exists, else against a REFERENCE transcription of the remote
code (read as text, never run; each names its file, snapshot, sha256 and the lines it transcribes), at fp64, and
shows that switching the branch off moves the output.

  xsa / xsa_scaled   GPT-X2.5 (gptx2), GPT-S2 (gpts2) whole models; Limite's tanh(alpha) and Grug's eps (blocks)
  out_gate_post      MM200 / Ying-K1.5 whole model
  dyn_mask           Doge: stock DogeForCausalLM, and Doge-320M's remote per-query-head attention (whole models)
  swa_trunc          Meditor-2's truncated sliding K / V (block)
  chunked attention  Llama-4 text (stock Llama4ForCausalLM, dense layers)
  attn_out_sat       Qwen3-D-ROS (stock Qwen3 + a REFERENCE of Qwen3DROSAttention)
  doc_mask           Ethoric's document mask (stock Llama under the REFERENCE dense_mask)
  moa                JetMoE (stock JetMoeForCausalLM)
  onorm              Muse-Spark's attention (REFERENCE of transformers 5.22.2 MuseSparkTextAttention)
  vha                ERNIE-5.5's VHAPostMix (REFERENCE from the round-3 spec's transcription)

Run:  pytest tests/test_attn_gaps.py -q
"""
import math

import pytest
import torch
import torch.nn as _nn
import torch.nn.functional as _F

F64 = torch.float64


def _rand_(model, seed=0, s=0.05):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.nan_to_num(p, 0.0, 0.0, 0.0) + torch.randn_like(p) * s) if p.dim() >= 1 else None
    return model


def _ids(vocab, T=12, seed=0, B=2):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (B, T))


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1e-30)


class _RMS(_nn.Module):
    def __init__(self, n, eps=1e-6, offset=False):
        super().__init__()
        self.weight, self.eps, self.offset = _nn.Parameter(torch.zeros(n) if offset else torch.ones(n)), eps, offset

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * ((1 + self.weight) if self.offset else self.weight)


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def _cos_sin(T, inv):
    f = torch.arange(T, dtype=F64)[:, None] * inv[None]
    return torch.cat((f.cos(), f.cos()), -1), torch.cat((f.sin(), f.sin()), -1)


# --- XSA: GPT-X2.5 / GPT-S2 ---------------------------------------------------------------------------------
# REFERENCE, not the remote code: AxiomicLabs/GPT-X2.5-135M modeling_gptx2.py (snapshot 53959ad5, sha256 16a8d45c...,
# read as text): RMSNorm (l. 16-24), precompute_rope_cos_sin / apply_rotary_emb (l. 27-51: adjacent pairs (2i, 2i+1)),
# GPTX2Attention (l. 54-116: GQA, causal SDPA, XSA l. 104-111: y - (y . v / max(|v|^2, 1e-6)) v with the token's own
# KV-group value), GPTX2SwiGLUMLP / GPTX2Block (l. 119-143), GPTX2ForCausalLM.forward (l. 214-241). And
# AxiomicLabs/GPT-S-5M modeling_gpts2.py (snapshot 3d7c6fb7, sha256 4cd4b27d...): the same block with complex RoPE
# (l. 31-44, the same adjacent pairs) and XSA by F.normalize(v) (l. 91-97: eps 1e-12). fp64, no cache, no padding.
class _GPTXCfg:
    def __init__(self, mt, **kw):
        self.model_type, self.vocab_size, self.hidden_size, self.num_hidden_layers = mt, 61, 32, 2
        self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.intermediate_size = 4, 2, 8, 48
        self.rope_theta, self.rms_norm_eps, self.xsa_projection = 2500.0, 1e-6, True
        self.__dict__.update(kw)


class _GPTXAttn(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c, d, H, Hk, hd = c, c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj, self.k_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(d, Hk * hd, bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, Hk * hd, bias=False), _nn.Linear(H * hd, d, bias=False)

    def forward(self, x):
        c, (B, T, _) = self.c, x.shape
        H, Hk, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        rep = H // Hk
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hk, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hk, hd).transpose(1, 2)
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, hd, 2, dtype=F64) / hd))
        f = torch.outer(torch.arange(T, dtype=F64), inv)
        cos, sin = f.cos(), f.sin()

        def rot(t):                                      # (even, odd) -> (e cos - o sin, e sin + o cos)
            e, o = t.reshape(*t.shape[:-1], -1, 2).unbind(-1)
            return torch.stack((e * cos - o * sin, e * sin + o * cos), -1).flatten(-2)
        q, k = rot(q), rot(k)
        y = _F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True)
        if c.xsa_projection:
            if c.model_type == "gptx2":                  # modeling_gptx2.py:106-111
                y = y.view(B, Hk, rep, T, hd)
                vg = v.unsqueeze(2)
                y = y - ((y * vg).sum(-1, keepdim=True) / vg.pow(2).sum(-1, keepdim=True).clamp_min(1e-6)) * vg
                y = y.view(B, H, T, hd)
            else:                                        # modeling_gpts2.py:91-97
                vn = _F.normalize(v.repeat_interleave(rep, 1), dim=-1)
                y = y - (y * vn).sum(-1, keepdim=True) * vn
        return self.o_proj(y.transpose(1, 2).reshape(B, T, H * hd))


class _GPTXMLP(_nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f = c.hidden_size, c.intermediate_size
        self.w_gate, self.w_up, self.w_down = _nn.Linear(d, f, bias=False), _nn.Linear(d, f, bias=False), _nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w_down(_F.silu(self.w_gate(x)) * self.w_up(x))


class _GPTXBlock(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1, self.attn = _RMS(c.hidden_size, c.rms_norm_eps), _GPTXAttn(c)
        self.ln_2, self.mlp = _RMS(c.hidden_size, c.rms_norm_eps), _GPTXMLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _GPTX(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.transformer = _nn.ModuleDict(dict(wte=_nn.Embedding(c.vocab_size, c.hidden_size),
                                               h=_nn.ModuleList(_GPTXBlock(c) for _ in range(c.num_hidden_layers)),
                                               ln_f=_RMS(c.hidden_size, c.rms_norm_eps)))
        self.lm_head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, ids):
        x = self.transformer["wte"](ids)
        for b in self.transformer["h"]:
            x = b(x)
        return self.lm_head(self.transformer["ln_f"](x))


@pytest.mark.parametrize("mt", ["gptx2", "gpts2"])
@torch.no_grad()
def test_same_host_carries_gptx2_xsa(mt):
    """xsa (model_graph FEATURES; minimal_emulator HOST_FEATS): from_gptx2 equals the REFERENCE whole model, and the
    XSA branch binds (xsa_c = 0 moves the output; the host with it off equals the reference without XSA)."""
    from elephantine.universal import from_gptx2
    t = _rand_(_GPTX(_GPTXCfg(mt)).to(F64).eval(), s=0.3)
    x = _ids(61)
    ref, host = t(x), from_gptx2(t)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))
    for ul in host.layers:
        ul.attn.xsa_c.zero_()
    off = host(x)
    assert _rel(ref, off) > 1e-3
    t.config.xsa_projection = False
    assert _rel(t(x), off) < 1e-12


# REFERENCE, not the remote code: paradigma-inc/limite-1b-violetto modeling_limite.py (snapshot 19569666, sha256
# 478d09ab..., read as text) l. 187 / 207-208 (has_xsa, a per-head xsa_alpha), l. 367-395 (the eval branch: y -
# tanh(alpha_h) (y . v_hat) v_hat, v_hat = F.normalize(current values, eps = xsa_normalize_eps), per KV group);
# and the Grug-MoE form the round-3 spec xsa_self_value_projection.md quotes from skyrl_train_models_grug_moe.py:743-745
# (o - (o . v / (|v|^2 + 1e-6)) v; the marin repo is not cached here). On a plain GQA attention (full rotate-half RoPE).
def _gqa_ref(x, wq, wk, wv, wo, H, Hk, hd, inv, post=None):
    B, T, _ = x.shape
    rep = H // Hk
    q = (x @ wq.T).view(B, T, H, hd).transpose(1, 2)
    k = (x @ wk.T).view(B, T, Hk, hd).transpose(1, 2)
    v = (x @ wv.T).view(B, T, Hk, hd).transpose(1, 2)
    cos, sin = _cos_sin(T, inv)
    q, k = q * cos + _rot_half(q) * sin, k * cos + _rot_half(k) * sin
    y = _F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True)
    if post is not None:
        y = post(y, v)
    return y.transpose(1, 2).reshape(B, T, H * hd) @ wo.T


@pytest.mark.parametrize("form", ["limite", "grug"])
@torch.no_grad()
def test_scaled_xsa_and_eps_forms(form):
    from elephantine.universal import UAttention, UConfig, _attn_any, set_xsa
    torch.manual_seed(3)
    d, H, Hk, hd = 24, 4, 2, 6
    rep = H // Hk
    wq, wk, wv, wo = (torch.randn(o, i, dtype=F64) * 0.3 for o, i in ((H * hd, d), (Hk * hd, d), (Hk * hd, d), (d, H * hd)))
    inv = 1.0 / (1e4 ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    alpha, eps = torch.randn(H, dtype=F64), 1e-4

    def limite(y, v):                                    # modeling_limite.py:375-394 (eval)
        B, _, T, _ = y.shape
        g = y.transpose(1, 2).reshape(B, T, Hk, rep, hd)
        vd = _F.normalize(v.transpose(1, 2), dim=-1, eps=eps).unsqueeze(3)
        g = g - torch.tanh(alpha).view(1, 1, Hk, rep, 1) * (g * vd).sum(-1, keepdim=True) * vd
        return g.reshape(B, T, H, hd).transpose(1, 2)

    def grug(y, v):
        v = v.repeat_interleave(rep, 1)
        return y - ((y * v).sum(-1, keepdim=True) / (v.pow(2).sum(-1, keepdim=True) + 1e-6)) * v
    x = torch.randn(2, 9, d, dtype=F64)
    ref = _gqa_ref(x, wq, wk, wv, wo, H, Hk, hd, inv, limite if form == "limite" else grug)
    a = UAttention(UConfig(vocab=2, d=d, L=1, heads=H, head_dim=hd))
    _attn_any(a, wq, wk, wv, wo, n_kv=Hk, scale=hd ** -0.5, inv_freq=inv)
    plain = a(x, True)
    if form == "limite":
        set_xsa(a, torch.tanh(alpha), clamp=eps ** 2)
    else:
        set_xsa(a, 1.0, clamp=0.0, add=1e-6)
    assert _rel(ref, a(x, True)) < 1e-12, _rel(ref, a(x, True))
    assert _rel(ref, plain) > 1e-3 and _rel(_gqa_ref(x, wq, wk, wv, wo, H, Hk, hd, inv), plain) < 1e-12


# --- MM200 (Ying-K1.5): the gate after o_proj -----------------------------------------------------------------
# REFERENCE, not the remote code: Karasumikura/Ying-K1.5-Lite-200M modeling_mm200.py (snapshot e63468e7, sha256
# 8759f1f4..., read as text): MM200Config (l. 53-87), ZeroCenteredRMSNorm / RMSNorm (l. 90-121), build_rope / apply_rope
# (l. 124-147: rotate-half on the first rot_dim dims), GatedAttention's SDPA path (l. 150-233: qk-norm, RoPE, the
# local layers' 0 <= t - s <= W mask, out = o_proj(attn) * sigmoid(gate_proj(x))), SwiGLU / Block / MM200.forward
# (l. 236-277, 383-401). fp64, use_weight_norm False (the Ying config), no MTP at inference.
class _MMCfg:
    vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads = 53, 32, 4, 4, 2
    head_dim, intermediate_size, rms_norm_eps, rope_theta, partial_rope_frac = 8, 40, 1e-6, 1e4, 0.25
    sliding_window, local_global_ratio, use_gated_attention, use_qk_norm = 3, 3, True, True
    use_zero_centered_norm, yarn_mscale = True, 1.0


class _MMAttn(_nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.c, d, self.nh, self.nkv, self.hd = c, c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj, self.k_proj = _nn.Linear(d, self.nh * self.hd, bias=False), _nn.Linear(d, self.nkv * self.hd, bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, self.nkv * self.hd, bias=False), _nn.Linear(self.nh * self.hd, d, bias=False)
        self.gate_proj = _nn.Linear(d, self.nh * self.hd, bias=False)
        self.q_norm, self.k_norm = _RMS(self.hd, c.rms_norm_eps), _RMS(self.hd, c.rms_norm_eps)
        self.is_local = (i % (c.local_global_ratio + 1)) != c.local_global_ratio

    def forward(self, x):
        b, s, _ = x.shape
        c = self.c
        q = self.q_proj(x).view(b, s, self.nh, self.hd).transpose(1, 2) * c.yarn_mscale
        k = self.k_proj(x).view(b, s, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.nkv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        rd = max(2, int(self.hd * c.partial_rope_frac) // 2 * 2)
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, rd, 2, dtype=F64) / rd))
        fr = torch.outer(torch.arange(s, dtype=F64), inv)
        cc, si = fr.cos(), fr.sin()

        def rope(t):
            r, p = t[..., :rd], t[..., rd:]
            t1, t2 = r.chunk(2, -1)
            return torch.cat([torch.cat([t1 * cc - t2 * si, t2 * cc + t1 * si], -1), p], -1)
        q, k = rope(q), rope(k)
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        m = None
        if self.is_local and s > c.sliding_window:
            idx = torch.arange(s)
            dl = idx[None, :] - idx[:, None]
            m = (dl <= 0) & (dl >= -c.sliding_window)
        out = _F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=m is None)
        out = self.o_proj(out.transpose(1, 2).reshape(b, s, -1))
        return out * torch.sigmoid(self.gate_proj(x)) if c.use_gated_attention else out


class _MMFFN(_nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f = c.hidden_size, c.intermediate_size
        self.gate, self.up, self.down = _nn.Linear(d, f, bias=False), _nn.Linear(d, f, bias=False), _nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(_F.silu(self.gate(x)) * self.up(x))


class _MMBlock(_nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.attn_norm, self.attn = _RMS(c.hidden_size, c.rms_norm_eps, True), _MMAttn(c, i)
        self.ffn_norm, self.ffn = _RMS(c.hidden_size, c.rms_norm_eps, True), _MMFFN(c)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class _MM200(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cfg = c
        self.embed = _nn.Embedding(c.vocab_size, c.hidden_size)
        self.blocks = _nn.ModuleList(_MMBlock(c, i) for i in range(c.num_hidden_layers))
        self.norm = _RMS(c.hidden_size, c.rms_norm_eps, True)
        self.lm_head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm(x))


@torch.no_grad()
def test_same_host_carries_mm200_post_gate():
    """out_gate_post: from_mm200 equals the REFERENCE whole model (T = 12 > W = 3: the local layers' window binds);
    the post-o_proj gate binds (pg_c = [1, 0] moves the output)."""
    from elephantine.universal import from_mm200
    t = _rand_(_MM200(_MMCfg()).to(F64).eval(), s=0.3)
    x = _ids(53)
    ref, host = t(x), from_mm200(t)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))
    for ul in host.layers:
        ul.attn.pg_c.copy_(torch.tensor([1.0, 0.0]))
    assert _rel(ref, host(x)) > 1e-3


# --- Doge: the dynamic mask (+ the per-channel residual gains) --------------------------------------------------
# The stock DogeForCausalLM (transformers 4.57: per-KV-head A / dt_proj, q / k RMS norms) and, for SmallDoge/Doge-320M's
# remote code, a REFERENCE (not run) of its DogeAttention: modeling_doge.py (snapshot 2249aa22, sha256 9692070e...,
# read as text) l. 243-363 -- no qk-norm, A and dt_proj per QUERY head (Linear(Hk hd, H)), the same prepare_dynamic_mask
# -- swapped into the stock model in place of the stock attention.
class _DogeRemoteAttn(_nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.config, self.layer_idx = c, layer_idx
        self.head_dim = c.hidden_size // c.num_attention_heads
        self.num_key_value_groups = c.num_attention_heads // c.num_key_value_heads
        self.scaling, self.keep_window_size = self.head_dim ** -0.5, c.keep_window_size
        H, Hk, hd, d = c.num_attention_heads, c.num_key_value_heads, self.head_dim, c.hidden_size
        self.q_proj, self.k_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(d, Hk * hd, bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, Hk * hd, bias=False), _nn.Linear(H * hd, d, bias=False)
        self.A = _nn.Parameter(torch.zeros(H))
        self.dt_proj = _nn.Linear(Hk * hd, H, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kw):
        from transformers.models.doge.modeling_doge import apply_rotary_pos_emb, repeat_kv
        sh = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(sh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(sh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(sh).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        dt = self.dt_proj(v.transpose(1, 2).reshape(v.shape[0], v.shape[-2], -1))
        dt = torch.exp(self.A * _F.softplus(dt)).transpose(-1, -2)                      # (B, H, S)
        mn = torch.finfo(hidden_states.dtype).min
        m = dt[:, :, None, :].expand(-1, -1, hidden_states.shape[1], -1)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.where(attention_mask, torch.tensor(0.0, dtype=hidden_states.dtype), mn)
            m = m.masked_fill(attention_mask[:, :, :, : m.shape[-1]] != 0, mn)
        if m.shape[-1] > self.keep_window_size:
            act = torch.zeros_like(m).scatter(-1, torch.topk(m, self.keep_window_size, dim=-1).indices, 1.0)
            m = m.masked_fill(act == 0.0, mn)
        w = (q @ repeat_kv(k, self.num_key_value_groups).transpose(2, 3)) * self.scaling + m
        o = w.softmax(-1) @ repeat_kv(v, self.num_key_value_groups)
        return self.o_proj(o.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)), None


@pytest.mark.parametrize("variant", ["stock", "remote"])
@torch.no_grad()
def test_same_host_carries_doge_dynamic_mask(variant):
    """dyn_mask (+ res_gain): from_doge equals Doge at T = 12 > keep_window_size = 5 (the top-k keys bind), A != 0;
    switching the dynamic mask off (dm_c = 0) or the residual gains (rgain.gx = 1) moves the output."""
    from transformers import DogeConfig, DogeForCausalLM
    from elephantine.universal import from_doge
    cfg = DogeConfig(vocab_size=67, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, keep_window_size=5, is_moe=False, attn_implementation="eager",
                     tie_word_embeddings=False)
    t = DogeForCausalLM(cfg)
    if variant == "remote":
        for i, l in enumerate(t.model.layers):
            l.self_attn = _DogeRemoteAttn(cfg, i)
    t = _rand_(t.to(F64).eval(), s=0.3)
    torch.manual_seed(2)
    for l in t.model.layers:
        l.self_attn.A.copy_(torch.randn_like(l.self_attn.A))
        l.input_residual.copy_(1 + 0.3 * torch.randn_like(l.input_residual))
        l.post_attention_residual.copy_(1 + 0.3 * torch.randn_like(l.post_attention_residual))
    x = _ids(67)
    from tests.test_universal_elephantine import _lift_fp32
    with _lift_fp32():                                   # DogeRMSNorm / the rotary round to fp32: lifted
        ref = t(x).logits
    host = from_doge(t)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))
    h2 = from_doge(t)
    for ul in h2.layers:
        ul.attn.dm_c.zero_()
    assert _rel(ref, h2(x)) > 1e-3
    h3 = from_doge(t)
    for ul in h3.layers:
        ul.rgain.gx.fill_(1.0)
    assert _rel(ref, h3(x)) > 1e-3


# --- Meditor: K / V truncated to the last W keys, top-left causal ---------------------------------------------
# REFERENCE, not the remote code: Yo-Purple/Meditor-2-3.4B-A1.8B-HMoE modeling_meditor2.py (snapshot 10e886b5, sha256
# c94299bd..., read as text) Meditor2Attention.forward l. 160-196 on a sliding (non-global) layer, no cache: RoPE at
# the true positions, then k, v <- k[-W:], v[-W:] when longer than W, then (kv_len = W, not > W) SDPA is_causal on
# the (T, W) scores -- PyTorch's top-left alignment. Meditor-1 (modeling_meditor.py:164-185) is the same with GQA.
@pytest.mark.parametrize("T", [12, 4])
@torch.no_grad()
def test_meditor_kv_truncation(T):
    from elephantine.universal import UAttention, UConfig, _attn_any, set_kv_trunc
    torch.manual_seed(4)
    d, H, hd, W = 24, 4, 6, 5
    wq, wk, wv, wo = (torch.randn(o, i, dtype=F64) * 0.3 for o, i in ((H * hd, d),) * 3 + ((d, H * hd),))
    inv = 1.0 / (1.3e5 ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    x = torch.randn(2, T, d, dtype=F64)
    B = 2
    q, k, v = ((x @ w.T).view(B, T, H, hd).transpose(1, 2) for w in (wq, wk, wv))
    cos, sin = _cos_sin(T, inv)
    q, k = q * cos + _rot_half(q) * sin, k * cos + _rot_half(k) * sin
    if k.size(2) > W:
        k, v = k[:, :, -W:], v[:, :, -W:]
    ref = _F.scaled_dot_product_attention(q, k, v, is_causal=T > 1).transpose(1, 2).reshape(B, T, -1) @ wo.T
    a = UAttention(UConfig(vocab=2, d=d, L=1, heads=H, head_dim=hd))
    _attn_any(a, wq, wk, wv, wo, n_kv=H, scale=hd ** -0.5, inv_freq=inv)
    plain = a(x, True)
    set_kv_trunc(a, W)
    assert _rel(ref, a(x, True)) < 1e-12, _rel(ref, a(x, True))
    if T > W:                                            # the truncation binds (a band or full causal layer differs)
        assert _rel(ref, plain) > 1e-3
    else:
        assert _rel(ref, plain) < 1e-12


# --- Llama-4: chunked attention -------------------------------------------------------------------------------
@torch.no_grad()
def test_same_host_carries_llama4_chunked_dense():
    """The chunked RoPE layers (attention_chunk_size 4 < T = 12), the NoPE layers' temperature (floor_scale 3) and
    the post-RoPE L2 qk-norm: from_llama4 vs the stock Llama4ForCausalLM (dense layers, its fp32 casts lifted);
    the chunk mask binds (chunk = inf moves the output), and so does the (t + 1) offset of the temperature."""
    from transformers import Llama4ForCausalLM, Llama4TextConfig
    from tests.test_universal_elephantine import _lift_fp32
    from elephantine.universal import from_llama4
    cfg = Llama4TextConfig(vocab_size=71, hidden_size=32, intermediate_size=40, intermediate_size_mlp=48,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                           moe_layers=[], no_rope_layers=[1, 1, 1, 0], attention_chunk_size=4, floor_scale=3,
                           attn_scale=0.5, use_qk_norm=True, attn_implementation="eager", rope_scaling=None)
    t = _rand_(Llama4ForCausalLM(cfg).to(F64).eval(), s=0.3)
    x = _ids(71)
    with _lift_fp32():
        ref = t(x, use_cache=False).logits
    host = from_llama4(t)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))
    h2 = from_llama4(t)
    for ul in h2.layers:
        ul.attn.chunk.fill_(float("inf"))
    assert _rel(ref, h2(x)) > 1e-3
    h3 = from_llama4(t)
    h3.layers[3].attn.qtemp_off.zero_()
    assert _rel(ref, h3(x)) > 1e-6


# --- Qwen3-D-ROS: attention-output saturation -------------------------------------------------------------------
# REFERENCE, not the remote code: aifeifei798/Qwen3-0.6B-D-ROS modeling_qwen3_eq.py (snapshot 8e1cf53e, sha256
# 7876ee15..., read as text): Qwen3DROSAttention l. 22-186 (a stock Qwen3Attention whose output passes _apply_dros --
# layer_ratio gate, multiband avg-pool crossover or the single band, _saturate_band / _apply_transfer_curve, makeup,
# dry / wet -- then the layer EQ gain), installed on a stock Qwen3ForCausalLM as Qwen3EQForCausalLM does (l. 189-198).
def _dros_ref_cls():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from elephantine.branches_attn import _layer_gain

    class _DROS(Qwen3Attention):
        def __init__(self, config, layer_idx):
            super().__init__(config, layer_idx)
            self.d_ros_config, self.total_layers = config.d_ros_config, config.num_hidden_layers
            self.layer_gain = _layer_gain(self.d_ros_config, layer_idx)   # l. 35-73 (transcribed in branches_attn)

        def _curve(self, e, knee, alpha, ct):
            if ct == "tanh":
                return knee * torch.tanh(e * alpha / max(knee, 1e-6))
            if ct == "sqrtsoftplus":
                return knee * torch.sqrt(_F.softplus(e * alpha / max(knee, 1e-6)))
            if ct == "hard":
                return torch.zeros_like(e)
            return e * (1.0 - alpha)

        def _band(self, x, b):
            th, kn, al = b.get("threshold", 26.0), b.get("knee", 6.0), b.get("alpha_max", 0.35)
            pol, ct = b.get("polarity", "positive_only"), b.get("curve_type", "tanh")
            if pol == "positive_only":
                return torch.where(x > th, th + self._curve(torch.clamp(x - th, min=0.0), kn, al, ct), x)
            if pol == "symmetric":
                ax = x.abs()
                nm = th + self._curve(torch.clamp(ax - th, min=0.0), kn, al, ct)
                return x * torch.where(ax > th, nm / torch.clamp(ax, min=1e-6), 1.0)
            return x

        def _multi(self, x, mb):
            b, s, d = x.shape
            r = x.view(b * s, 1, d)
            kl, km = mb.get("crossover_kernel_low", 9), mb.get("crossover_kernel_mid", 3)
            lo = _F.avg_pool1d(r, kernel_size=kl, stride=1, padding=kl // 2)
            hi = r - lo
            mi = _F.avg_pool1d(hi, kernel_size=km, stride=1, padding=km // 2)
            hi = hi - mi
            bd = mb.get("bands", {})
            return sum(self._band(t.view(b, s, d), bd.get(n, {})) for t, n in ((lo, "low"), (mi, "mid"), (hi, "high")))

        def _dros(self, x):
            c = self.d_ros_config
            if not c.get("enabled", True):
                return x
            r = c.get("layer_ratio", (0.60, 0.90))
            if not (int(self.total_layers * r[0]) <= self.layer_idx <= int(self.total_layers * r[1])):
                return x
            w, mk, mb = c.get("dry_wet", 0.70), c.get("makeup_gain", 1.0), c.get("multiband", {})
            sh = self._multi(x, mb) if mb.get("enabled", False) else self._band(x, c.get("single_band_fallback", c))
            if mk != 1.0:
                sh = sh * mk
            return sh if w >= 1.0 else (x if w <= 0.0 else (1.0 - w) * x + w * sh)

        def forward(self, *a, **k):
            o, *rest = super().forward(*a, **k)
            o = self._dros(o)
            return (o * self.layer_gain if self.layer_gain != 1.0 else o, *rest)
    return _DROS


_DROS_CFG = {"enabled": True, "layer_ratio": [0.3, 0.9], "dry_wet": 0.85, "makeup_gain": 1.1,
             "layer_eq": {"enabled": True, "layer_gains_db": {"1": 1.5, "3": -2.0}},
             "multiband": {"enabled": True, "crossover_kernel_low": 5, "crossover_kernel_mid": 3, "bands": {
                 "low": {"threshold": 0.02, "knee": 0.05, "alpha_max": 0.2, "curve_type": "tanh", "polarity": "symmetric"},
                 "mid": {"threshold": 0.01, "knee": 0.04, "alpha_max": 0.25, "curve_type": "sqrtsoftplus",
                         "polarity": "positive_only"},
                 "high": {"threshold": 0.01, "knee": 0.03, "alpha_max": 0.45, "curve_type": "tanh",
                          "polarity": "positive_only"}}},
             "single_band_fallback": {"threshold": 0.02, "knee": 0.05, "alpha_max": 0.35, "curve_type": "tanh"}}


@pytest.mark.parametrize("multiband", [True, False])
@torch.no_grad()
def test_same_host_carries_qwen3_dros(multiband):
    """attn_out_sat: from_qwen3_dros vs the stock Qwen3 under the REFERENCE D-ROS attention, thresholds set low enough
    that the saturation bites; switching the saturation off (sat.cfg[0] = 0) moves the output."""
    import copy
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from tests.test_universal_elephantine import _lift_fp32
    from elephantine.universal import from_qwen3_dros
    dc = copy.deepcopy(_DROS_CFG)
    dc["multiband"]["enabled"] = multiband
    cfg = Qwen3Config(vocab_size=59, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, attn_implementation="eager", tie_word_embeddings=False)
    cfg.d_ros_config = dc
    t = Qwen3ForCausalLM(cfg)
    R = _dros_ref_cls()
    for i, l in enumerate(t.model.layers):
        st = l.self_attn.state_dict()
        l.self_attn = R(cfg, i)
        l.self_attn.load_state_dict(st)
    t = _rand_(t.to(F64).eval(), s=0.3)
    x = _ids(59)
    with _lift_fp32():
        ref = t(x).logits
    host = from_qwen3_dros(t)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))
    for ul in host.layers:
        ul.attn.sat.cfg[0] = 0.0
    assert _rel(ref, host(x)) > 1e-4


# --- documents: Ethoric's eos mask ------------------------------------------------------------------------------
# REFERENCE, not the remote code: techdotus/Ethoric-10M (snapshot d8c245b1, read as text) modeling_ethoric.py
# document_ids l. 436-447 (sha256 e1c6fd3c...: cumsum(eos) - eos, an eos closes its own document) and attention.py
# dense_mask l. 79-99 (sha256 76eb2241...: causal & same document); the attention is the stock Llama's, run under that
# mask passed as a 4-D additive attention_mask.
@torch.no_grad()
def test_doc_mask_attention():
    """doc_mask (the attention part): the host's doc_eos mask equals the stock Llama under the REFERENCE document
    mask; without it (doc_eos = -1) the output moves."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from tests.test_universal_elephantine import _lift_fp32
    from elephantine.universal import from_llama, set_doc_mask
    cfg = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, attn_implementation="eager")
    t = _rand_(LlamaForCausalLM(cfg).to(F64).eval(), s=0.3)
    x = _ids(41, T=14)
    eos = 7
    x[0, 4], x[0, 9], x[1, 6] = eos, eos, eos
    e = (x == eos).to(torch.int32)
    doc = torch.cumsum(e, 1) - e
    pos = torch.arange(14)[None].expand(2, -1)
    m = ((pos[:, None, :] <= pos[:, :, None]) & (doc[:, :, None] == doc[:, None, :]))[:, None]
    with _lift_fp32():
        ref = t(x, attention_mask=torch.zeros(m.shape, dtype=F64).masked_fill(~m, torch.finfo(F64).min)).logits
    host = set_doc_mask(from_llama(t), eos)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))
    host.doc_eos.fill_(-1.0)
    assert _rel(ref, host(x)) > 1e-3


# --- JetMoE: the mixture of attention heads ---------------------------------------------------------------------
@torch.no_grad()
def test_same_host_carries_jetmoe_moa():
    """moa: from_jetmoe vs the stock JetMoeForCausalLM (top-2 of 4 attention experts over a shared GQA K / V, and the
    MoE FFN); routing no expert gate (hr_k = 0: every expert's heads at weight 1) moves the output."""
    from transformers import JetMoeConfig, JetMoeForCausalLM
    from tests.test_universal_elephantine import _lift_fp32
    from elephantine.universal import from_jetmoe
    cfg = JetMoeConfig(vocab_size=53, hidden_size=32, num_hidden_layers=2, num_key_value_heads=2, kv_channels=8,
                       intermediate_size=24, num_local_experts=4, num_experts_per_tok=2, attn_implementation="eager",
                       tie_word_embeddings=False)
    t = JetMoeForCausalLM(cfg)
    with torch.no_grad():
        for p in t.parameters():
            p.zero_()                                    # the expert biases are torch.empty
    t = _rand_(t.to(F64).eval(), s=0.3)
    x = _ids(53)
    with _lift_fp32():
        ref = t(x, use_cache=False).logits
    host = from_jetmoe(t)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))
    for ul in host.layers:
        ul.attn.hr_k.zero_()
    assert _rel(ref, host(x)) > 1e-3


# --- Muse-Spark's o_norm -----------------------------------------------------------------------------------------
# REFERENCE (written from the text, not imported): transformers 5.22.2 models/muse_spark/modeling_muse_spark.py
# (sha256 969ef27d...) MuseSparkRMSNorm l. 105-123 (with_scale False: x (mean x^2 + eps)^-0.5) and
# MuseSparkTextAttention l. 355-427: weight-free qk-norm, q * qk_scale_factor, RoPE, causal GQA, o_norm per head,
# o * sigmoid(gate_proj x), o_proj.
@torch.no_grad()
def test_muse_spark_attention_onorm():
    from elephantine.universal import UAttention, UConfig, _attn_any, set_onorm
    torch.manual_seed(6)
    d, H, Hk, hd, eps, qs = 24, 4, 2, 6, 1e-6, 1.7
    rep = H // Hk
    wq, wk, wv, wo, wg = (torch.randn(o, i, dtype=F64) * 0.3 for o, i in
                          ((H * hd, d), (Hk * hd, d), (Hk * hd, d), (d, H * hd), (H * hd, d)))
    inv = 1.0 / (1e4 ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    nrm = lambda t: t * torch.pow(t.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    x = torch.randn(2, 9, d, dtype=F64)
    B, T = 2, 9
    q = nrm((x @ wq.T).view(B, T, H, hd).transpose(1, 2)) * qs
    k = nrm((x @ wk.T).view(B, T, Hk, hd).transpose(1, 2))
    v = (x @ wv.T).view(B, T, Hk, hd).transpose(1, 2)
    cos, sin = _cos_sin(T, inv)
    q, k = q * cos + _rot_half(q) * sin, k * cos + _rot_half(k) * sin
    o = _F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True,
                                        scale=hd ** -0.5)
    o = nrm(o.transpose(1, 2)).reshape(B, T, -1) * torch.sigmoid(x @ wg.T)
    ref = o @ wo.T
    a = UAttention(UConfig(vocab=2, d=d, L=1, heads=H, head_dim=hd))
    _attn_any(a, wq, wk, wv, wo, n_kv=Hk, scale=hd ** -0.5, inv_freq=inv)
    for i, gain in enumerate((qs, 1.0)):                 # the weight-free qk-norm, q times the constant gain
        (a.q_nw, a.k_nw)[i].fill_(gain)
        a.qk_eps[i] = eps
        (a.q_coef, a.k_coef)[i].copy_(torch.tensor([0.0, 1.0]))
    a.g_proj.weight.copy_(wg)
    a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
    set_onorm(a, eps)
    assert _rel(ref, a(x, True)) < 1e-12, _rel(ref, a(x, True))
    a.o_coef.copy_(torch.tensor([1.0, 0.0]))
    assert _rel(ref, a(x, True)) > 1e-3


# --- ERNIE-5.5's VHAPostMix ---------------------------------------------------------------------------------------
# REFERENCE from the round-3 spec's transcription (verify/r3/specs/vha_head_mix.md, of baidu/vLLM-ERNIE attention.py
# l. 110-146 and the DSA layer's use l. 903-904; the GitHub repo is not cached here): m = I + V U^T (U, V (H, 16)),
# o[t, h] <- sum_g m[h, g] o[t, g] after the attention, then wo_b(o * sigmoid(gate)).
@torch.no_grad()
def test_ernie_vha_head_mix():
    from elephantine.universal import UAttention, UConfig, _attn_any, set_vha
    torch.manual_seed(7)
    d, H, hd, r = 24, 4, 6, 3
    wq, wk, wv, wo, wg = (torch.randn(o, i, dtype=F64) * 0.3 for o, i in ((H * hd, d),) * 3 + ((d, H * hd), (H * hd, d)))
    U, V = torch.randn(H, r, dtype=F64), torch.randn(H, r, dtype=F64)
    inv = 1.0 / (1e4 ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    x = torch.randn(2, 9, d, dtype=F64)

    def mix(y, v):
        m = torch.eye(H, dtype=F64) + V @ U.T
        return torch.einsum("hg,bgtd->bhtd", m, y)
    o = _gqa_ref(x, wq, wk, wv, torch.eye(H * hd, dtype=F64), H, H, hd, inv, mix)
    ref = (o * torch.sigmoid(x @ wg.T)) @ wo.T
    a = UAttention(UConfig(vocab=2, d=d, L=1, heads=H, head_dim=hd))
    _attn_any(a, wq, wk, wv, wo, n_kv=H, scale=hd ** -0.5, inv_freq=inv)
    a.g_proj.weight.copy_(wg)
    a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
    set_vha(a, U, V)
    assert _rel(ref, a(x, True)) < 1e-12, _rel(ref, a(x, True))
    set_vha(a, U * 0, V)
    assert _rel(ref, a(x, True)) > 1e-3
