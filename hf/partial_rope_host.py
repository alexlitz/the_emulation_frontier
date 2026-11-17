"""A partial-RoPE host reproduces a full-RoPE target head exactly when it rotates enough dims.

A RoPE head's logit is a sum over its rotating PAIRS p of  Re(q_p conj(k_p) e^{i w_p (t - s)})  plus the
plain dot product of its pass (unrotated) dims. A partial-RoPE host head has rot_h / 2 rotating pairs
(the first rot_h dims of the head, frequencies in the model's `inv_freq` buffer) and hd_h - rot_h pass
dims. The construction, per target head (head_dim d_t, rot_t rotated dims, rot_t = d_t for full RoPE):

  * pairs: target pair j (frequency w_j) goes onto host pair j. The host's `inv_freq[j] = w_j` -- the
    same buffer setting that makes NoPE a special case of RoPE (hf/nope_rope.py), so no code change;
  * layout: which two host columns form pair j is a column PERMUTATION of q / k: (j, j + rot_h / 2)
    for the half-split rotation (GPT-NeoX, Phi, Qwen3-Next / Qwen3.5, Llama), (2j, 2j + 1) for the
    interleaved one (GLM, ChatGLM, GPT-J). The rotation's orientation is the same in both;
  * pass dims of a partial target go onto the host's pass dims, then onto spare host pairs whose
    frequency is set to 0 (a zero-frequency pair is the identity); every other host column is zero;
  * values: target v onto the host head's first d_v columns, o_proj reads those columns;
  * scales: the softmax scale s_t / s_h and the RoPE attention_scaling (YaRN's mscale m, which
    HF applies to cos and sin, so to the ROTATED dims only) fold into the q rows per column:
    q row *= (s_t f_t) / (s_h f_h), f = m^2 on rotated columns, 1 on pass columns;
  * heads: target query head i of KV group g -> host query head g * G_h + i (G = query heads per KV
    head): needs KV_h >= KV_t and G_h >= G_t (or an MHA host, tiling the KV heads). Spare host heads
    are zeroed (their o_proj columns are zero);
  * a host qk-norm (Qwen3-Next: RMS over the host head, before the rotary) over a zero-padded head
    gives rms_h = sqrt(d_t / hd_h) rms_t when eps_h = eps_t d_t / hd_h: the norm weight absorbs the
    constant (and the scale fold, which no longer fits in W_q); a target without qk-norm is the
    qknorm-slot regime edge, not this one;
  * gated attention (Qwen3-Next's sigmoid output gate): gate rows zero -> sigmoid(0) = 1/2, o_proj x2.

So the condition is  rot_h / 2 >= rot_t / 2  rotating pairs (rot_h >= rot_t) and hd_h >= d_t,
d_v,h >= d_v,t. The frequency schedule is free -- it is the buffer -- for every STATIC schedule
(default, linear, llama3, YaRN, deepseek-YaRN, M-RoPE with equal text positions): only its values and
the attention_scaling matter. With the config alone (no buffer write) a default-RoPE host matches a
default-RoPE target by rope_theta_h = rope_theta_t ** (rot_h / d_t): host pair j has frequency
theta_h^(-2j / rot_h) = theta_t^(-2j / d_t) (exact in exact arithmetic; the float32 power differs in
the last bits).

Where it fails:
  * rot_h < rot_t: the target needs rot_t / 2 pairs, each a distinct frequency (a distinct
    position-dependent bilinear form). Pass dims are position-independent, and splitting the pairs
    over MORE host heads does not help -- each host head runs its own softmax, so the logits do not
    add. A host with more heads but the same rotated dims per head is no better; only a host whose
    rotated dims per head reach rot_t (wider heads at the same factor, or a larger factor) is.
  * one buffer for all layers: a target whose layers use DIFFERENT schedules (Gemma-3 local / global
    thetas, MiMo's sliding-window theta) needs the union of its frequencies in the host's one buffer:
    rot_h / 2 >= the number of distinct frequencies (graph: dims_of(...)["rot_need"]).
  * runtime-switched schedules (dynamic NTK, LongRoPE) change the target's inv_freq with the sequence
    length; a fixed buffer holds one setting (exact below original_max_position_embeddings).

Run:  python -m hf.partial_rope_host          # fp64 module + whole-model checks
      python -m hf.partial_rope_host --real   # + SmolLM2-135M in a GLM partial-RoPE host
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch

F64 = torch.float64
# sdpa: the eager path rounds the softmax to fp32, sdpa keeps fp64 end to end
ATTN = "sdpa"


# --- the target head, read off a stock module -------------------------------------------------------

@dataclass
class Target:
    H: int
    KV: int
    d: int                     # q / k head dim
    rot: int                   # rotated dims per head (= d for full RoPE), half-split layout
    dv: int
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    inv_freq: torch.Tensor     # rot / 2 frequencies
    scale: float               # softmax scale
    m: float = 1.0             # RoPE attention_scaling (YaRN mscale), applied to cos / sin
    bq: torch.Tensor | None = None
    bk: torch.Tensor | None = None
    bv: torch.Tensor | None = None
    bo: torch.Tensor | None = None
    q_norm: torch.Tensor | None = None   # per-head RMSNorm weight on q / k (Qwen3), applied before the rotary
    k_norm: torch.Tensor | None = None
    eps: float = 0.0


def target_from(attn, rotary, cfg):
    """A Llama / Qwen2 / Qwen3 / Phi attention (separate q / k / v projections, half-split RoPE)."""
    d = attn.head_dim
    H = cfg.num_attention_heads
    KV = getattr(cfg, "num_key_value_heads", None) or H
    rot = 2 * rotary.inv_freq.numel()
    o = attn.o_proj if hasattr(attn, "o_proj") else attn.dense
    g = lambda lin: lin.bias.detach().clone() if lin.bias is not None else None
    qn, kn = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
    return Target(H=H, KV=KV, d=d, rot=rot, dv=d, wq=attn.q_proj.weight.detach().clone(),
                  wk=attn.k_proj.weight.detach().clone(), wv=attn.v_proj.weight.detach().clone(),
                  wo=o.weight.detach().clone(), inv_freq=rotary.inv_freq.detach().clone(),
                  scale=float(attn.scaling), m=float(rotary.attention_scaling),
                  bq=g(attn.q_proj), bk=g(attn.k_proj), bv=g(attn.v_proj), bo=g(o),
                  q_norm=qn.weight.detach().clone() if qn is not None else None,
                  k_norm=kn.weight.detach().clone() if kn is not None else None,
                  eps=float(getattr(qn, "variance_epsilon", getattr(qn, "eps", 0.0))) if qn is not None else 0.0)


# --- the host layout --------------------------------------------------------------------------------

@dataclass
class Host:
    H: int
    KV: int
    hd: int
    rot: int
    interleaved: bool          # (2j, 2j+1) pairs (GLM) vs (j, j + rot/2) (NeoX / Phi / Qwen3-Next)
    scale: float
    m: float = 1.0             # the host's own attention_scaling
    qk_norm: str | None = None  # None, or "one_plus" (Qwen3-Next: x_hat * (1 + w))
    gated: bool = False        # Qwen3-Next output gate


def pairs(rot, interleaved):
    r = rot // 2
    return [(2 * j, 2 * j + 1) for j in range(r)] if interleaved else [(j, j + r) for j in range(r)]


def column_map(t: Target, h: Host):
    """host column for each target q / k column, the host inv_freq, and the columns that rotate."""
    if h.rot < t.rot:
        raise ValueError(f"host rotates {h.rot} dims per head < the target's {t.rot}: no construction")
    if h.hd < t.d or h.hd < t.dv:
        raise ValueError("host head narrower than the target's")
    tp, hp = pairs(t.rot, False), pairs(h.rot, h.interleaved)
    m = [-1] * t.d
    for j, (a, b) in enumerate(tp):
        m[a], m[b] = hp[j]
    spare = list(range(h.rot, h.hd)) + [c for p in hp[len(tp):] for c in p]   # pass dims, then 0-freq pairs
    tpass = list(range(t.rot, t.d))
    if len(tpass) > len(spare):
        raise ValueError("not enough host dims for the target's pass dims")
    for c, hc in zip(tpass, spare):
        m[c] = hc
    inv = torch.zeros(h.rot // 2, dtype=t.inv_freq.dtype)
    inv[: t.rot // 2] = t.inv_freq
    return m, inv


def head_map(t: Target, h: Host):
    """target query head -> host query head, target KV head -> host KV head."""
    Gt, Gh = t.H // t.KV, h.H // h.KV
    if h.KV >= t.KV and Gh >= Gt:
        return [(i // Gt) * Gh + i % Gt for i in range(t.H)], list(range(t.KV))
    if h.KV == h.H and h.H >= t.H:          # MHA host: one KV head per target query head (tile)
        return list(range(t.H)), None
    raise ValueError("host head grouping cannot hold the target's")


def place(t: Target, h: Host):
    """Host-layout q / k / v / o weights (+ biases, qk-norm weights, inv_freq) reproducing the target."""
    m, inv = column_map(t, h)
    qh, kvh = head_map(t, h)
    hid = t.wq.shape[1]
    dt = t.wq.dtype
    Wq, Wk = torch.zeros(h.H * h.hd, hid, dtype=dt), torch.zeros(h.KV * h.hd, hid, dtype=dt)
    Wv, Wo = torch.zeros(h.KV * h.hd, hid, dtype=dt), torch.zeros(t.wo.shape[0], h.H * h.hd, dtype=dt)
    bq, bk, bv = (torch.zeros(h.H * h.hd, dtype=dt), torch.zeros(h.KV * h.hd, dtype=dt),
                  torch.zeros(h.KV * h.hd, dtype=dt))
    # per-column q factor: softmax scale and attention_scaling folds (rotated cols carry m^2)
    f = torch.tensor([(t.scale * (t.m ** 2 if c < t.rot else 1.0)) / (h.scale * (h.m ** 2 if m[c] < h.rot else 1.0))
                      for c in range(t.d)], dtype=F64).to(dt)
    mi = torch.tensor(m)
    Gt = t.H // t.KV
    for i in range(t.H):
        src = slice(i * t.d, (i + 1) * t.d)
        rows = qh[i] * h.hd + mi
        if h.qk_norm is None:
            Wq[rows] = t.wq[src] * f[:, None]
            if t.bq is not None:
                bq[rows] = t.bq[src] * f
        else:
            Wq[rows] = t.wq[src]
            if t.bq is not None:
                bq[rows] = t.bq[src]
        tk = i // Gt                                       # the target KV head this query reads
        hk = kvh[tk] if kvh is not None else qh[i]         # its host KV head (tiled on an MHA host)
        ks, krows = slice(tk * t.d, (tk + 1) * t.d), hk * h.hd + mi
        Wk[krows] = t.wk[ks]
        vs, vrows = slice(tk * t.dv, (tk + 1) * t.dv), hk * h.hd + torch.arange(t.dv)
        Wv[vrows] = t.wv[vs]
        for b_src, b_dst, s_, r_ in ((t.bk, bk, ks, krows), (t.bv, bv, vs, vrows)):
            if b_src is not None:
                b_dst[r_] = b_src[s_]
        Wo[:, qh[i] * h.hd + torch.arange(t.dv)] = t.wo[:, i * t.dv:(i + 1) * t.dv]
    out = dict(wq=Wq, wk=Wk, wv=Wv, wo=Wo, bq=bq, bk=bk, bv=bv, bo=t.bo, inv_freq=inv)
    if h.qk_norm is not None:
        if t.q_norm is None:
            raise ValueError("a qk-norm host for a target without qk-norm is the qknorm regime edge")
        c = math.sqrt(t.d / h.hd)                         # rms_h = c rms_t once eps_h = eps_t c^2
        wqn, wkn = torch.ones(h.hd, dtype=dt), torch.ones(h.hd, dtype=dt)
        wqn[mi], wkn[mi] = t.q_norm * c * f, t.k_norm * c
        sub = 1.0 if h.qk_norm == "one_plus" else 0.0
        out.update(q_norm=wqn - sub, k_norm=wkn - sub, eps=t.eps * c * c)
    if h.gated:
        out["wo"] = 2 * out["wo"]                          # gate rows 0 -> sigmoid = 1/2
    return out


# --- writers into stock host modules ----------------------------------------------------------------

@torch.no_grad()
def write_sep(attn, P, gated=False):
    """Separate q / k / v / o projections (GlmAttention, PhiAttention, Qwen3NextAttention, Llama)."""
    o = attn.o_proj if hasattr(attn, "o_proj") else attn.dense
    if gated:                                              # q_proj rows per head: [query | gate]
        hd = attn.head_dim
        wq = attn.q_proj.weight.view(-1, 2 * hd, attn.q_proj.weight.shape[1])
        wq.zero_()
        wq[:, :hd] = P["wq"].view(-1, hd, P["wq"].shape[1])
        if attn.q_proj.bias is not None:
            bq = attn.q_proj.bias.view(-1, 2 * hd)
            bq.zero_()
            bq[:, :hd] = P["bq"].view(-1, hd)
    else:
        attn.q_proj.weight.copy_(P["wq"])
        if attn.q_proj.bias is not None:
            attn.q_proj.bias.copy_(P["bq"])
    for lin, w, b in ((attn.k_proj, "wk", "bk"), (attn.v_proj, "wv", "bv")):
        lin.weight.copy_(P[w])
        if lin.bias is not None:
            lin.bias.copy_(P[b])
    o.weight.copy_(P["wo"])
    if o.bias is not None:
        o.bias.copy_(P["bo"] if P["bo"] is not None else torch.zeros_like(o.bias))
    elif P["bo"] is not None:
        raise ValueError("the target's o_proj bias has no place in this host")
    if "q_norm" in P:
        attn.q_norm.weight.copy_(P["q_norm"])
        attn.k_norm.weight.copy_(P["k_norm"])
        attn.q_norm.eps = attn.k_norm.eps = P["eps"]


@torch.no_grad()
def write_neox(attn, P):
    """GPT-NeoX fused query_key_value: per head [q | k | v] (MHA)."""
    hd = attn.head_size
    hid = P["wq"].shape[1]
    W = attn.query_key_value.weight.view(-1, 3, hd, hid)
    B = attn.query_key_value.bias.view(-1, 3, hd)
    for s, (w, b) in enumerate((("wq", "bq"), ("wk", "bk"), ("wv", "bv"))):
        W[:, s] = P[w].view(-1, hd, hid)
        B[:, s] = P[b].view(-1, hd)
    attn.dense.weight.copy_(P["wo"])
    attn.dense.bias.copy_(P["bo"] if P["bo"] is not None else torch.zeros_like(attn.dense.bias))


@torch.no_grad()
def set_inv_freq(rotary, inv):
    rotary.inv_freq = inv.to(rotary.inv_freq.dtype).clone()
    rotary.original_inv_freq = rotary.inv_freq


# --- module-level fp64 checks -----------------------------------------------------------------------

def causal(T):
    return torch.triu(torch.full((T, T), float("-inf"), dtype=F64), 1)[None, None]


@torch.no_grad()
def run_attn(attn, rotary, x):
    T = x.shape[1]
    pos = torch.arange(T)[None]
    return attn(hidden_states=x, attention_mask=causal(T), position_embeddings=rotary(x, pos))[0]


def _llama_target(d=16, H=4, KV=2, hidden=48, rope_scaling=None, theta=1e4, qwen3=False, bias=False, seed=0):
    """A Llama (bias=False) / Qwen2 (bias=True: q / k / v biases) / Qwen3 (qk-norm) target attention."""
    from transformers import LlamaConfig, Qwen2Config, Qwen3Config
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding
    torch.manual_seed(seed)
    kw = dict(hidden_size=hidden, num_attention_heads=H, num_key_value_heads=KV, head_dim=d,
              rope_theta=theta, rope_scaling=rope_scaling, max_position_embeddings=4096,
              attn_implementation=ATTN)
    if bias and not qwen3:
        cfg = Qwen2Config(**{k: v for k, v in kw.items() if k != "head_dim"})
        assert cfg.hidden_size // H == d
        attn, rot = Qwen2Attention(cfg, 0), Qwen2RotaryEmbedding(cfg)
    elif qwen3:
        cfg = Qwen3Config(**kw, rms_norm_eps=1e-6)
        attn, rot = Qwen3Attention(cfg, 0), Qwen3RotaryEmbedding(cfg)
        with torch.no_grad():
            attn.q_norm.weight.uniform_(0.5, 1.5)
            attn.k_norm.weight.uniform_(0.5, 1.5)
    else:
        cfg = LlamaConfig(**kw)
        attn, rot = LlamaAttention(cfg, 0), LlamaRotaryEmbedding(cfg)
    with torch.no_grad():
        for p in attn.parameters():
            if p.dim() == 1 and not qwen3:
                p.normal_(0, 0.3)
    return cfg, attn.double().eval(), rot.double()


def _host(family, hidden, H, KV, hd, factor, rope_scaling=None, theta=1e4, seed=1):
    """A stock partial-RoPE host attention + rotary of the given family (random weights, overwritten)."""
    torch.manual_seed(seed)
    if family == "glm":
        from transformers import GlmConfig
        from transformers.models.glm.modeling_glm import GlmAttention, GlmRotaryEmbedding
        cfg = GlmConfig(hidden_size=hidden, num_attention_heads=H, num_key_value_heads=KV, head_dim=hd,
                        partial_rotary_factor=factor, rope_theta=theta, rope_scaling=rope_scaling,
                        max_position_embeddings=4096, attn_implementation=ATTN)
        a, r = GlmAttention(cfg, 0), GlmRotaryEmbedding(cfg)
        spec = dict(interleaved=True)
    elif family == "phi":
        from transformers import PhiConfig
        from transformers.models.phi.modeling_phi import PhiAttention, PhiRotaryEmbedding
        assert hd * H == hidden
        cfg = PhiConfig(hidden_size=hidden, num_attention_heads=H, num_key_value_heads=KV,
                        partial_rotary_factor=factor, rope_theta=theta, rope_scaling=rope_scaling,
                        max_position_embeddings=4096, attn_implementation=ATTN)
        a, r = PhiAttention(cfg, 0), PhiRotaryEmbedding(cfg)
        spec = dict(interleaved=False)
    elif family == "neox":
        from transformers import GPTNeoXConfig
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXRotaryEmbedding
        assert hd * H == hidden and KV == H
        cfg = GPTNeoXConfig(hidden_size=hidden, num_attention_heads=H, rotary_pct=factor, rotary_emb_base=theta,
                            rope_scaling=rope_scaling, max_position_embeddings=4096, attn_implementation=ATTN)
        a, r = GPTNeoXAttention(cfg, 0), GPTNeoXRotaryEmbedding(cfg)
        spec = dict(interleaved=False)
    elif family == "qwen3_next":
        from transformers import Qwen3NextConfig
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention, Qwen3NextRotaryEmbedding
        cfg = Qwen3NextConfig(hidden_size=hidden, num_attention_heads=H, num_key_value_heads=KV, head_dim=hd,
                              partial_rotary_factor=factor, rope_theta=theta, rope_scaling=rope_scaling,
                              max_position_embeddings=4096, attn_implementation=ATTN)
        a, r = Qwen3NextAttention(cfg, 0), Qwen3NextRotaryEmbedding(cfg)
        spec = dict(interleaved=False, qk_norm="one_plus", gated=True)
    else:
        raise ValueError(family)
    a, r = a.double().eval(), r.double()
    rot = 2 * r.inv_freq.numel()
    return a, r, Host(H=H, KV=KV, hd=hd, rot=rot, scale=float(a.scaling), m=float(r.attention_scaling), **spec)


def module_check(family, d=16, H=4, KV=2, hd=32, factor=0.5, target_scaling=None, host_scaling=None,
                 qwen3=False, host_H=None, host_KV=None, T=40, seed=0):
    """max |host - target| of one attention block (fp64), the host a stock partial-RoPE module."""
    # glm: a Qwen2 target (q / k / v biases; hidden = H d); phi / neox: hidden = the host's H hd
    hidden = (host_H or H) * hd if family in ("phi", "neox") else H * d if family == "glm" else 48
    cfg, tattn, trot = _llama_target(d=d, H=H, KV=KV, hidden=hidden, rope_scaling=target_scaling,
                                     qwen3=qwen3, bias=family == "glm", seed=seed)
    t = target_from(tattn, trot, cfg)
    hattn, hrot, h = _host(family, hidden, host_H or H, host_KV or KV, hd, factor, rope_scaling=host_scaling)
    P = place(t, h)
    (write_neox if family == "neox" else lambda a, p: write_sep(a, p, gated=h.gated))(hattn, P)
    set_inv_freq(hrot, P["inv_freq"])
    x = torch.randn(2, T, hidden, dtype=F64, generator=torch.Generator().manual_seed(seed + 7))
    return (run_attn(tattn, trot, x) - run_attn(hattn, hrot, x)).abs().max().item()


def theta_only_check(d=16, hd=64, factor=0.5, theta=1e4, T=64, seed=0):
    """No buffer write: a default-RoPE GLM host with rope_theta_h = theta_t ** (rot_h / d_t)."""
    cfg, tattn, trot = _llama_target(d=d, theta=theta, seed=seed)
    t = target_from(tattn, trot, cfg)
    rot_h = int(hd * factor)
    hattn, hrot, h = _host("glm", 48, t.H, t.KV, hd, factor, theta=theta ** (rot_h / d))
    P = place(t, h)
    write_sep(hattn, P)
    x = torch.randn(1, T, 48, dtype=F64, generator=torch.Generator().manual_seed(seed + 3))
    freq_err = (hrot.inv_freq[: d // 2].double() - trot.inv_freq.double()).abs().max().item()
    return freq_err, (run_attn(tattn, trot, x) - run_attn(hattn, hrot, x)).abs().max().item()


# --- whole model: a Llama target in a GLM (partial, interleaved RoPE) host --------------------------

def glm_host_config(tcfg, factor=0.5, hd=None):
    """GlmConfig for a Llama / Qwen2 target: same widths, head_dim hd (default d_t / factor, so
    rot_h = d_t), RMSNorm eps, vocab; GLM's attention bias on (zeroed when the target has none)."""
    from transformers import GlmConfig
    d = getattr(tcfg, "head_dim", None) or tcfg.hidden_size // tcfg.num_attention_heads
    hd = hd or int(round(d / factor))
    return GlmConfig(vocab_size=tcfg.vocab_size, hidden_size=tcfg.hidden_size,
                     intermediate_size=tcfg.intermediate_size, num_hidden_layers=tcfg.num_hidden_layers,
                     num_attention_heads=tcfg.num_attention_heads, num_key_value_heads=tcfg.num_key_value_heads,
                     head_dim=hd, partial_rotary_factor=factor, rms_norm_eps=tcfg.rms_norm_eps,
                     rope_theta=getattr(tcfg, "rope_theta", 1e4), max_position_embeddings=tcfg.max_position_embeddings,
                     tie_word_embeddings=False, hidden_act=tcfg.hidden_act, attention_bias=True,
                     pad_token_id=None, eos_token_id=None, bos_token_id=None, attn_implementation=ATTN)


@torch.no_grad()
def glm_from_llama(target, factor=0.5, hd=None, dtype=F64):
    """Build a stock GlmForCausalLM (partial interleaved RoPE) that reproduces a Llama-family target."""
    from transformers import GlmForCausalLM
    tcfg = target.config
    hcfg = glm_host_config(tcfg, factor, hd)
    host = GlmForCausalLM(hcfg).to(dtype).eval()
    trot = target.model.rotary_emb
    hrot = host.model.rotary_emb
    h = Host(H=hcfg.num_attention_heads, KV=hcfg.num_key_value_heads, hd=hcfg.head_dim,
             rot=2 * hrot.inv_freq.numel(), interleaved=True, scale=hcfg.head_dim ** -0.5,
             m=float(hrot.attention_scaling))
    inv = None
    for lt, lh in zip(target.model.layers, host.model.layers):
        t = target_from(lt.self_attn, trot, tcfg)
        P = place(t, h)
        write_sep(lh.self_attn, P)
        inv = P["inv_freq"]
        lh.input_layernorm.weight.copy_(lt.input_layernorm.weight)
        lh.post_attention_layernorm.weight.copy_(lt.post_attention_layernorm.weight)
        lh.mlp.gate_up_proj.weight.copy_(torch.cat((lt.mlp.gate_proj.weight, lt.mlp.up_proj.weight), 0))
        lh.mlp.down_proj.weight.copy_(lt.mlp.down_proj.weight)
    set_inv_freq(hrot, inv)
    host.model.embed_tokens.weight.copy_(target.model.embed_tokens.weight)
    host.model.norm.weight.copy_(target.model.norm.weight)
    host.lm_head.weight.copy_(target.lm_head.weight)
    return host


def whole_model_check(rope_scaling=None, factor=0.5, T=24, seed=0):
    """A random fp64 LlamaForCausalLM and its GLM host: max |logit diff|, greedy-identical?"""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_theta=1e4,
                      rope_scaling=rope_scaling, max_position_embeddings=4096, attn_implementation=ATTN)
    tgt = LlamaForCausalLM(cfg).double().eval()
    host = glm_from_llama(tgt, factor=factor)
    ids = torch.randint(0, 97, (2, T), generator=torch.Generator().manual_seed(seed + 1))
    with torch.no_grad():
        lt, lh = tgt(ids).logits, host(ids).logits
        gt = tgt.generate(ids[:1, :8], max_new_tokens=24, do_sample=False, pad_token_id=0)
        gh = host.generate(ids[:1, :8], max_new_tokens=24, do_sample=False, pad_token_id=0)
    return (lt - lh).abs().max().item(), bool(torch.equal(gt, gh))


# --- a real checkpoint ------------------------------------------------------------------------------

@torch.no_grad()
def truncate_rotation(host, keep_pairs):
    """The natural shortfall: a host that rotates only `keep_pairs` of the target's pairs (the rest
    unrotated, frequency 0) -- what a host with rot_h = 2 keep_pairs < d_t could at best do."""
    inv = host.model.rotary_emb.inv_freq.clone()
    inv[keep_pairs:] = 0
    set_inv_freq(host.model.rotary_emb, inv)


def real_check(repo="HuggingFaceTB/SmolLM2-135M", n_prompts=8, n_new=64, dtype=torch.float32, keep_pairs=None):
    """The real checkpoint in a GLM partial-RoPE host (rot_h = d_t), scored like the other payable
    edges: teacher-forced top-1 on the target's own greedy continuations, and greedy identity of
    the first 32 / all n_new tokens. keep_pairs < d_t / 2: the truncated (rot_h < d_t) host."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from act.act_swap_agreement import PROMPTS, cont_logp, greedy
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    tgt = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True, dtype=dtype,
                                               attn_implementation=ATTN).eval()
    host = glm_from_llama(tgt, factor=0.5, dtype=dtype)
    if keep_pairs is not None:
        truncate_rotation(host, keep_pairs)
    top1 = n = ex32 = ex64 = 0
    maxdiff = 0.0
    prompts = PROMPTS[:n_prompts]
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        ref = greedy(tgt, ids, n_new)
        seq = torch.cat([ids, ref], 1)
        lt, lh = cont_logp(tgt, seq, ids.shape[1]), cont_logp(host, seq, ids.shape[1])
        maxdiff = max(maxdiff, (lt - lh).abs().max().item())
        top1 += (lt.argmax(-1) == lh.argmax(-1)).sum().item()
        n += lt.shape[0]
        g = greedy(host, ids, n_new)
        same = (g[0] == ref[0])
        ex32 += bool(same[:32].all())
        ex64 += bool(same.all())
    return dict(repo=repo, top1=top1 / n, ex32=ex32 / len(prompts), ex64=ex64 / len(prompts),
                max_logp_diff=maxdiff, keep_pairs=keep_pairs)


@torch.no_grad()
def long_context_check(repo="HuggingFaceTB/SmolLM2-135M", keep_pairs=None, n_ctx=2048, n_new=32, n_cuts=4,
                       text_files=("README.md", "FRONTIER.md", "VALIDATION.md", "TREE.md"), dtype=torch.float32):
    """The truncated host at LONG context: a real document cut at n_cuts points >= n_ctx / 2; top-1
    agreement over the second half of the window (teacher-forced on the document) and greedy identity
    of n_new tokens from each cut."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from act.act_swap_agreement import greedy
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    tgt = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True, dtype=dtype,
                                               attn_implementation=ATTN).eval()
    host = glm_from_llama(tgt, factor=0.5, dtype=dtype)
    if keep_pairs is not None:
        truncate_rotation(host, keep_pairs)
    text = "\n\n".join(open(f).read() for f in text_files)
    ids = tok(text, return_tensors="pt").input_ids[:, : n_ctx]
    lt, lh = tgt(ids).logits[0], host(ids).logits[0]
    half = ids.shape[1] // 2
    top1 = (lt[half:].argmax(-1) == lh[half:].argmax(-1)).double().mean().item()
    cuts = [half + k * (ids.shape[1] - half - 1) // max(n_cuts - 1, 1) for k in range(n_cuts)]
    ex = 0
    for c in cuts:
        p = ids[:, :c]
        ex += bool(torch.equal(greedy(tgt, p, n_new), greedy(host, p, n_new)))
    return dict(repo=repo, keep_pairs=keep_pairs, n_ctx=ids.shape[1], top1_second_half=top1,
                ex_greedy=ex / len(cuts), max_logit_diff=(lt - lh).abs().max().item())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024}
    l3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 1024}
    for fam, kw in [("glm", {}), ("glm", dict(hd=64, factor=0.5)), ("phi", dict(hd=32, factor=0.5, KV=4)),
                    ("neox", dict(hd=64, factor=0.25, KV=4)), ("qwen3_next", dict(hd=64, factor=0.25, qwen3=True)),
                    ("glm", dict(target_scaling=yarn)), ("glm", dict(target_scaling=l3, host_scaling=yarn))]:
        print(f"{fam:>11} {kw}: max|diff| = {module_check(fam, **kw):.2e}")
    print("theta-only (freq err, out err):", theta_only_check())
    print("whole Llama -> GLM:", whole_model_check(), "  yarn:", whole_model_check(rope_scaling=yarn))
    if a.real:
        print(real_check())
        print(real_check(keep_pairs=16))
