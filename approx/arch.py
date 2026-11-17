"""Reference forwards for the two architectures, written on plain weight dicts so the real
checkpoints can be streamed one layer at a time (a 26B teacher and a 21B student do not fit
in one 24 GB GPU, and GPU->GPU copies are unreliable on this box).

  teacher  Gemma-4 text (transformers 5.x `Gemma4TextDecoderLayer`): per-head q/k RMSNorm with
           weights, v RMSNorm without, attention scale 1, sliding (head 256, kv 8) / global (head 512,
           kv 2, k == v, proportional partial RoPE) layers, sandwich norms, dense GeLU-tanh MLP +
           128-expert top-8 MoE in parallel, per-layer `layer_scalar`, sqrt(h) embedding scale,
           final logit softcap. (Gemma-4's RMSNorm is `x/rms * w` -- NOT the (1+w) of Gemma 1-3.)
  student  gpt-oss (`GptOssDecoderLayer`, transformers 4.57): pre-norm only, q/k/v/o biases,
           per-head attention sinks, 1/sqrt(64) scale, top-k router with bias + softmax over the
           top-k, experts with biases and the clamped swiglu-oai activation.

Names are canonical (no `model.` / `model.language_model.` prefix, see serving.st_reader.canon).
Both forwards are checked against the stock modules in tests/test_approx.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- configs

@dataclass
class TCfg:
    """Gemma-4 text config (the fields the forward uses)."""
    vocab: int
    h: int
    layer_types: list
    nq: int
    hd: int                 # sliding head dim
    nkv: int                # sliding kv heads
    hd_g: int               # global head dim
    nkv_g: int              # global kv heads
    k_eq_v: bool
    rope: dict              # layer_type -> {"theta", "partial", "type"}
    window: int
    E: int
    Ie: int
    topk: int
    Id: int
    softcap: float | None
    eps: float = 1e-6
    embed_scale: float = 1.0
    dtype: torch.dtype = torch.bfloat16

    @property
    def L(self):
        return len(self.layer_types)

    def head(self, li):
        """(head_dim, n_kv, k_eq_v) of layer li."""
        if self.layer_types[li] == "full_attention":
            return self.hd_g, self.nkv_g, self.k_eq_v
        return self.hd, self.nkv, False

    @classmethod
    def from_hf(cls, c, dtype=torch.bfloat16):
        c = c.get("text_config", c)
        rp = c["rope_parameters"]
        rope = {t: {"theta": float(rp[t]["rope_theta"]), "type": rp[t].get("rope_type", "default"),
                    "partial": float(rp[t].get("partial_rotary_factor", 1.0))} for t in rp}
        es = torch.tensor(c["hidden_size"] ** 0.5).to(dtype).item()   # HF casts the scale to the weight dtype
        nkv_g = c.get("num_global_key_value_heads") or c["num_key_value_heads"]
        return cls(vocab=c["vocab_size"], h=c["hidden_size"], layer_types=list(c["layer_types"]),
                   nq=c["num_attention_heads"], hd=c["head_dim"], nkv=c["num_key_value_heads"],
                   hd_g=c.get("global_head_dim") or c["head_dim"], nkv_g=nkv_g,
                   k_eq_v=bool(c.get("attention_k_eq_v", False)), rope=rope, window=c["sliding_window"],
                   E=c["num_experts"], Ie=c["moe_intermediate_size"], topk=c["top_k_experts"],
                   Id=c["intermediate_size"], softcap=c.get("final_logit_softcapping"),
                   eps=c["rms_norm_eps"], embed_scale=es, dtype=dtype)


@dataclass
class SCfg:
    """gpt-oss config. Shapes (h, L, heads, head_dim, experts, top-k, I, layer-type pattern) are the
    target architecture; vocab / rope / window / eps are free config values."""
    vocab: int
    h: int = 2880
    L: int = 24
    nq: int = 64
    nkv: int = 8
    hd: int = 64
    E: int = 32
    topk: int = 4
    I: int = 2880
    window: int = 128
    eps: float = 1e-5
    rope_theta: float = 150000.0
    rope_scaling: dict | None = None
    limit: float = 7.0
    alpha: float = 1.702
    layer_types: list = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(self.L)]

    def hf_config(self, **extra):
        d = dict(architectures=["GptOssForCausalLM"], model_type="gpt_oss", vocab_size=self.vocab,
                 hidden_size=self.h, num_hidden_layers=self.L, num_attention_heads=self.nq,
                 num_key_value_heads=self.nkv, head_dim=self.hd, num_local_experts=self.E,
                 num_experts_per_tok=self.topk, experts_per_token=self.topk, intermediate_size=self.I,
                 sliding_window=self.window, rms_norm_eps=self.eps, rope_theta=self.rope_theta,
                 rope_scaling=self.rope_scaling, swiglu_limit=self.limit, layer_types=self.layer_types,
                 attention_bias=True, attention_dropout=0.0, tie_word_embeddings=False, hidden_act="silu",
                 max_position_embeddings=131072, initial_context_length=4096, use_cache=True,
                 router_aux_loss_coef=0.0, output_router_logits=False)
        d.update(extra)
        return d

    @classmethod
    def from_hf(cls, c):
        return cls(vocab=c["vocab_size"], h=c["hidden_size"], L=c["num_hidden_layers"],
                   nq=c["num_attention_heads"], nkv=c["num_key_value_heads"], hd=c["head_dim"],
                   E=c["num_local_experts"], topk=c.get("num_experts_per_tok", c.get("experts_per_token")),
                   I=c["intermediate_size"], window=c["sliding_window"], eps=c["rms_norm_eps"],
                   rope_theta=c["rope_theta"], rope_scaling=c.get("rope_scaling"),
                   limit=c.get("swiglu_limit", 7.0), layer_types=list(c["layer_types"]))


# ----------------------------------------------------------------------------- rope

def teacher_inv_freq(cfg: TCfg, layer_type):
    """Per-pair angular frequency for a teacher layer type (pairs (j, j + hd/2), rotate_half)."""
    hd = cfg.hd_g if layer_type == "full_attention" else cfg.hd
    r = cfg.rope[layer_type]
    if r["type"] == "proportional":
        ra = int(r["partial"] * hd // 2)
        rot = 1.0 / (r["theta"] ** (torch.arange(0, 2 * ra, 2, dtype=torch.float) / hd))
        return torch.cat([rot, torch.zeros(hd // 2 - ra)])
    assert r["type"] == "default", r
    return 1.0 / (r["theta"] ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))


def student_inv_freq(scfg: SCfg):
    """(inv_freq[hd/2], attention_scaling) exactly as transformers' gpt-oss computes them."""
    from transformers import GptOssConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    c = GptOssConfig(**{k: v for k, v in scfg.hf_config().items() if k not in ("architectures", "model_type")})
    rt = (scfg.rope_scaling or {}).get("rope_type", "default")
    inv, att = ROPE_INIT_FUNCTIONS[rt](c, None)
    return inv.float(), float(att)


def rope_cs(inv_freq, T, device, scaling=1.0):
    pos = torch.arange(T, device=device, dtype=torch.float)
    ang = pos[:, None] * inv_freq.to(device)[None, :]
    return ang.cos() * scaling, ang.sin() * scaling           # (T, hd/2)


def apply_rope(x, cos, sin):
    """x (..., T, hd); rotate pairs (j, j + hd/2). cos/sin (T, hd/2)."""
    a, b = x.chunk(2, -1)
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([a * c - b * s, b * c + a * s], -1)


# ----------------------------------------------------------------------------- basics

def rms(x, eps):
    xf = x.float()
    return xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)


def rmsnorm(x, w, eps):
    y = rms(x, eps)
    if w is not None:
        y = y * w.float()
    return y.to(x.dtype)


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def swiglu_oai(g, u, limit=7.0, alpha=1.702):
    g = g.clamp(max=limit)
    u = u.clamp(-limit, limit)
    return (u + 1) * (g * torch.sigmoid(g * alpha))


def band_mask(T, window, device):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    m = j <= i
    if window:
        m = m & (j > i - window)
    return m


def attend(q, k, v, scale, window, sinks=None):
    """q (B,nq,T,d) k,v (B,nkv,T,d) -> (B,T,nq*d). GQA as in HF repeat_kv. Sinks are exact via an
    extra always-visible key with zero value whose logit is sinks[h] for every query."""
    B, nq, T, d = q.shape
    rep = nq // k.shape[1]
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    mask = band_mask(T, window, q.device)
    if sinks is not None:
        pad = (-(d + 1)) % 8
        one = torch.ones(B, nq, T, 1, dtype=q.dtype, device=q.device)
        q = torch.cat([q, one, q.new_zeros(B, nq, T, pad)], -1)
        k = torch.cat([k, k.new_zeros(B, nq, T, 1 + pad)], -1)
        ks = torch.zeros(B, nq, 1, d + 1 + pad, dtype=k.dtype, device=k.device)
        ks[..., d] = (sinks.to(k.dtype) / scale).view(1, nq, 1)
        k = torch.cat([ks, k], 2)
        v = torch.cat([v.new_zeros(B, nq, 1, d), v], 2)
        mask = torch.cat([torch.ones(T, 1, dtype=torch.bool, device=q.device), mask], 1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return o.transpose(1, 2).reshape(B, T, nq * d)


def moe(x, idx, wts, expert_fn, E):
    """x (N,h); idx/wts (N,k). sum_k wts * expert_fn(e, x[tokens of e])."""
    acc = torch.promote_types(x.dtype, torch.float32)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=acc, device=x.device)
    flat = idx.reshape(-1)
    order = flat.argsort()
    counts = torch.bincount(flat, minlength=E).tolist()
    tok = (order // idx.shape[1])
    wflat = wts.reshape(-1)[order]
    start = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        t = tok[start:start + n]
        y = expert_fn(e, x[t])
        out.index_add_(0, t, y.to(acc) * wflat[start:start + n, None].to(acc))
        start += n
    return out.to(x.dtype)


# ----------------------------------------------------------------------------- teacher

def t_embed(w, cfg: TCfg, ids):
    e = w["embed_tokens.weight"]
    return F.embedding(ids, e) * torch.tensor(cfg.embed_scale, dtype=e.dtype)


def t_logits(w, cfg: TCfg, h, softcap=True):
    x = rmsnorm(h, w["norm.weight"], cfg.eps)
    W = w.get("lm_head.weight", w["embed_tokens.weight"])
    z = (x @ W.T.to(x.dtype)).float()
    if softcap and cfg.softcap:
        z = torch.tanh(z / cfg.softcap) * cfg.softcap
    return z


def t_qkv(lw, cfg: TCfg, li, x):
    """x = input_layernorm(h). Returns pre-norm q (B,T,nq,hd), k, v (B,T,nkv,hd)."""
    hd, nkv, keqv = cfg.head(li)
    B, T, _ = x.shape
    q = (x @ lw["self_attn.q_proj.weight"].T).view(B, T, cfg.nq, hd)
    k = (x @ lw["self_attn.k_proj.weight"].T).view(B, T, nkv, hd)
    v = k if keqv else (x @ lw["self_attn.v_proj.weight"].T).view(B, T, nkv, hd)
    return q, k, v


def t_attn(lw, cfg: TCfg, li, h, cap=None):
    """Attention sub-block up to (excluding) post_attention_layernorm. Returns o_proj output."""
    lt = cfg.layer_types[li]
    x = rmsnorm(h, lw["input_layernorm.weight"], cfg.eps)
    q, k, v = t_qkv(lw, cfg, li, x)
    T = h.shape[1]
    cos, sin = rope_cs(teacher_inv_freq(cfg, lt), T, h.device)
    qn = rmsnorm(q, lw["self_attn.q_norm.weight"], cfg.eps)
    kn = rmsnorm(k, lw["self_attn.k_norm.weight"], cfg.eps)
    vn = rmsnorm(v, None, cfg.eps)
    qr = apply_rope(qn.transpose(1, 2), cos, sin)
    kr = apply_rope(kn.transpose(1, 2), cos, sin)
    win = cfg.window if lt == "sliding_attention" else None
    a = attend(qr, kr, vn.transpose(1, 2), 1.0, win)
    if cap is not None:
        cap.update(x=x, q=q, k=k, v=v, qr=qr, kr=kr, vn=vn, a=a)
    return a @ lw["self_attn.o_proj.weight"].T


def t_router(lw, cfg: TCfg, r):
    """r = residual (pre-FFN). Returns (probs[N,E] fp32, topk weights incl per_expert_scale, idx)."""
    n = rmsnorm(r, None, cfg.eps) * lw["router.scale"] * cfg.h ** -0.5      # in the model dtype, as HF
    logits = (n @ lw["router.proj.weight"].T).float()
    p = logits.softmax(-1)
    tw, ti = p.topk(cfg.topk, -1)
    tw = tw / tw.sum(-1, keepdim=True)
    return logits, p, tw * lw["router.per_expert_scale"].float()[ti], ti


def t_expert_fn(lw, cfg):
    gu, dn = lw["experts.gate_up_proj"], lw["experts.down_proj"]

    def f(e, x):
        g, u = (x @ gu[e].T).chunk(2, -1)
        return (gelu_tanh(g) * u) @ dn[e].T
    return f


def t_dense(lw, cfg, x):
    g = x @ lw["mlp.gate_proj.weight"].T
    u = x @ lw["mlp.up_proj.weight"].T
    return (gelu_tanh(g) * u) @ lw["mlp.down_proj.weight"].T


def t_ffn(lw, cfg: TCfg, r, cap=None):
    """FFN sub-block on residual r; returns post_feedforward_layernorm output (the residual update)."""
    B, T, h = r.shape
    m1 = t_dense(lw, cfg, rmsnorm(r, lw["pre_feedforward_layernorm.weight"], cfg.eps))
    m1n = rmsnorm(m1, lw["post_feedforward_layernorm_1.weight"], cfg.eps)
    rf = r.reshape(-1, h)
    logits, p, tw, ti = t_router(lw, cfg, rf)
    x2 = rmsnorm(rf, lw["pre_feedforward_layernorm_2.weight"], cfg.eps)
    m2 = moe(x2, ti, tw, t_expert_fn(lw, cfg), cfg.E).view(B, T, h)
    m2n = rmsnorm(m2, lw["post_feedforward_layernorm_2.weight"], cfg.eps)
    f = rmsnorm(m1n + m2n, lw["post_feedforward_layernorm.weight"], cfg.eps)
    if cap is not None:
        cap.update(m1=m1, m1n=m1n, m2=m2, m2n=m2n, logits=logits, p=p, tw=tw, ti=ti, f=f)
    return f


def t_layer(lw, cfg: TCfg, li, h, cap=None, skip_attn=False, skip_ffn=False):
    if not skip_attn:
        o = t_attn(lw, cfg, li, h, cap)
        h = h + rmsnorm(o, lw["post_attention_layernorm.weight"], cfg.eps)
    if cap is not None:
        cap["r"] = h
    if not skip_ffn:
        h = h + t_ffn(lw, cfg, h, cap)
    return h * lw["layer_scalar"].to(h.dtype)


# ----------------------------------------------------------------------------- student

def s_embed(w, ids):
    return F.embedding(ids, w["embed_tokens.weight"])


def s_logits(w, scfg: SCfg, h):
    x = rmsnorm(h, w["norm.weight"], scfg.eps)
    return (x @ w["lm_head.weight"].T.to(x.dtype)).float()


def s_attn(lw, scfg: SCfg, h, rope, cap=None):
    """returns the attention update (o_proj output, added to the residual)."""
    B, T, _ = h.shape
    x = rmsnorm(h, lw["input_layernorm.weight"], scfg.eps)
    q = (x @ lw["self_attn.q_proj.weight"].T + lw["self_attn.q_proj.bias"]).view(B, T, scfg.nq, scfg.hd)
    k = (x @ lw["self_attn.k_proj.weight"].T + lw["self_attn.k_proj.bias"]).view(B, T, scfg.nkv, scfg.hd)
    v = (x @ lw["self_attn.v_proj.weight"].T + lw["self_attn.v_proj.bias"]).view(B, T, scfg.nkv, scfg.hd)
    cos, sin = rope
    qr = apply_rope(q.transpose(1, 2), cos[:T], sin[:T])
    kr = apply_rope(k.transpose(1, 2), cos[:T], sin[:T])
    win = scfg.window if lw["_type"] == "sliding_attention" else None
    a = attend(qr, kr, v.transpose(1, 2), scfg.hd ** -0.5, win, lw["self_attn.sinks"])
    if cap is not None:
        cap.update(x=x, a=a)
    return a @ lw["self_attn.o_proj.weight"].T + lw["self_attn.o_proj.bias"]


def s_router(lw, scfg: SCfg, x):
    logits = F.linear(x, lw["mlp.router.weight"].to(x.dtype), lw["mlp.router.bias"].to(x.dtype))  # as HF
    tv, ti = logits.topk(scfg.topk, -1)
    return logits, tv.softmax(-1), ti


def s_expert_fn(lw, scfg: SCfg):
    gu, gub = lw["mlp.experts.gate_up_proj"], lw["mlp.experts.gate_up_proj_bias"]
    dn, dnb = lw["mlp.experts.down_proj"], lw["mlp.experts.down_proj_bias"]

    def f(e, x):
        z = x @ gu[e] + gub[e]
        return swiglu_oai(z[..., ::2], z[..., 1::2], scfg.limit, scfg.alpha) @ dn[e] + dnb[e]
    return f


def s_ffn(lw, scfg: SCfg, h, cap=None):
    B, T, H = h.shape
    x = rmsnorm(h, lw["post_attention_layernorm.weight"], scfg.eps).reshape(-1, H)
    logits, tw, ti = s_router(lw, scfg, x)
    out = moe(x, ti, tw, s_expert_fn(lw, scfg), scfg.E).view(B, T, H)
    if cap is not None:
        cap.update(fx=x, slogits=logits, stw=tw, sti=ti, sf=out)
    return out


def s_layer(lw, scfg: SCfg, h, rope, cap=None):
    h = h + s_attn(lw, scfg, h, rope, cap)
    if cap is not None:
        cap["r"] = h
    return h + s_ffn(lw, scfg, h, cap)


def student_rope(scfg: SCfg, T, device):
    inv, att = student_inv_freq(scfg)
    return rope_cs(inv, T, device, att)


# ----------------------------------------------------------------------------- whole-model helpers

def layer_view(w, li, prefix="layers."):
    p = f"{prefix}{li}."
    return {k[len(p):]: v for k, v in w.items() if k.startswith(p)}


def teacher_forward(w, cfg: TCfg, ids, softcap=True, hiddens=None):
    h = t_embed(w, cfg, ids)
    for li in range(cfg.L):
        if hiddens is not None:
            hiddens.append(h)
        h = t_layer(layer_view(w, li), cfg, li, h)
    return t_logits(w, cfg, h, softcap)


def student_forward(w, scfg: SCfg, ids, hiddens=None):
    h = s_embed(w, ids)
    rope = student_rope(scfg, ids.shape[1], ids.device)
    for li in range(scfg.L):
        if hiddens is not None:
            hiddens.append(h)
        lw = layer_view(w, li)
        lw["_type"] = scfg.layer_types[li]
        h = s_layer(lw, scfg, h, rope)
    return s_logits(w, scfg, h)


def random_student(scfg: SCfg, seed=0, std=0.02, dtype=torch.float32):
    """gpt-oss's own init (normal(0, initializer_range), zero biases, unit norms) -- baseline (a)."""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: (torch.randn(*s, generator=g) * std).to(dtype)
    z = lambda *s: torch.zeros(*s, dtype=dtype)
    h, w = scfg.h, {}
    w["embed_tokens.weight"] = rn(scfg.vocab, h)
    w["lm_head.weight"] = rn(scfg.vocab, h)
    w["norm.weight"] = torch.ones(h, dtype=dtype)
    for li in range(scfg.L):
        p = f"layers.{li}."
        for n, o in [("q", scfg.nq * scfg.hd), ("k", scfg.nkv * scfg.hd), ("v", scfg.nkv * scfg.hd)]:
            w[p + f"self_attn.{n}_proj.weight"], w[p + f"self_attn.{n}_proj.bias"] = rn(o, h), z(o)
        w[p + "self_attn.o_proj.weight"], w[p + "self_attn.o_proj.bias"] = rn(h, scfg.nq * scfg.hd), z(h)
        w[p + "self_attn.sinks"] = z(scfg.nq)
        w[p + "input_layernorm.weight"] = torch.ones(h, dtype=dtype)
        w[p + "post_attention_layernorm.weight"] = torch.ones(h, dtype=dtype)
        w[p + "mlp.router.weight"], w[p + "mlp.router.bias"] = rn(scfg.E, h), z(scfg.E)
        w[p + "mlp.experts.gate_up_proj"] = rn(scfg.E, h, 2 * scfg.I)
        w[p + "mlp.experts.gate_up_proj_bias"] = z(scfg.E, 2 * scfg.I)
        w[p + "mlp.experts.down_proj"] = rn(scfg.E, scfg.I, h)
        w[p + "mlp.experts.down_proj_bias"] = z(scfg.E, h)
    return w


def to_hf_student(w):
    """canonical student names -> GptOssForCausalLM state_dict names."""
    return {(k if k.startswith("lm_head") else "model." + k): v for k, v in w.items()}
