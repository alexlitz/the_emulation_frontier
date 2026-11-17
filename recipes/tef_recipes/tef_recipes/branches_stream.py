# Vendored from the_emulation_frontier/elephantine/branches_stream.py by `python -m recipes.vendor`. Do not edit: edit elephantine/branches_stream.py.
"""branches_stream.py -- the universal host's residual-stream, input, FFN / MoE, norm and loop branches added for
the round-3 features the host did not carry (the eleph-gaps-stream pass). Every branch is OFF by default (a UConfig
field or a coefficient at its off value) and is part of the same UniversalElephantine architecture: the modules here
are built by ULayer / UniversalElephantine from their UConfig (init_layer / init_model) and called from fixed points
of their forward passes. Each branch has a whole-model loader (from_*) here, re-exported by elephantine.universal,
and a test against the target's own code (a stock transformers class) or a labelled REFERENCE transcription of its
remote code (read as text, never run): tests/test_eleph_gaps_stream.py.

  hc_kind "gated"   Qwen4-Exp's GatedResidual hyper-connections (hc_qwen4_exp): the block reads
                    mean_i(sigmoid(up(silu(down(n) / N)))_i * n_i), n = the per-stream RMS-normed streams, and writes
                    X_i + 2 sigmoid(W_b n / N)_i f (no stream mixing); the final collapse is the same read
  ple_ng_rows       Qwen4-Exp's n-gram PLE on the hc streams (engram_qwen4): per-segment (EOS-reset) hashed n-grams
                    (XOR of token x odd multiplier products, mod a prime per head), per-stream key / one value,
                    gate sigmoid(signed_sqrt(<rms(k_i), rms(X_i)> / sqrt(d))), + a dilated depthwise silu conv of the
                    normed gated values
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

F64 = torch.float64


def _U():
    from tef_recipes import universal
    return universal


def _p(*shape, fill=0.0):
    return nn.Parameter(torch.full(shape, fill, dtype=F64))


def _lin(i, o, bias=False):
    return _U()._lin(i, o, bias=bias)


def _grms(x, w, eps, g):
    """RMS norm over groups of g channels of the last dim, times a (full-width) weight."""
    s = x.unflatten(-1, (-1, g))
    return (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2) * w


# --- hc_kind "gated": Qwen4-Exp GatedResidual -------------------------------------------------------------------

class UGatedResidual(nn.Module):
    """Qwen4-Exp hyper-connections around one sub-block (modular_qwen4_exp.py Qwen4ExpTextGatedResidual): from the N
    streams X, n = per-stream RMS(X) w; the block reads h = mean_i(m_i * n_i), m = sigmoid(up(silu(down(n) / N)));
    the streams become X_i + inj_i out, inj = 2 sigmoid(W_b n / N). Returned as (post = inj, comb = I, h), the
    interface of UHC, so ULayer.forward_hc runs it unchanged."""

    def __init__(self, c, combine=True):
        super().__init__()
        N, d, r = c.hc_mult, c.d, max(c.hc_rank, 1)
        self.N, self.d = N, d
        self.w = _p(N * d, fill=1.0)
        self.register_buffer("eps", torch.tensor(1e-6, dtype=F64))
        self.down, self.up = _lin(N * d, r), _lin(r, N * d)
        self.inject = _lin(N * d, N) if combine else None

    def forward(self, X, reg=None):
        B, T, N, d = X.shape
        n = _grms(X.reshape(B, T, N * d), self.w, self.eps, d)
        m = torch.sigmoid(self.up(F.silu(self.down(n) / N))).view(B, T, N, d)
        h = (m * n.view(B, T, N, d)).mean(-2)
        if self.inject is None:
            return None, None, h
        post = 2 * torch.sigmoid(self.inject(n) / N)
        comb = torch.eye(N, dtype=X.dtype).expand(B, T, N, N)
        return post, comb, h


class UHCVariant(nn.Module):
    """The Sinkhorn hyper-connection codes other than DeepSeek-V4's comb^T (UHC), each returned as (post, M, h) with
    the streams becoming post * out + M^T X (ULayer.forward_hc):
      "comb"   DeepSeek-V4's mixes (UHC) but the streams read comb X, not comb^T X (nanowhale's hc_post
               einsum("bsij,bsjd->bsid", comb, residual))
      "xing"   Xing4.0 (modeling_xing4_0.py Xing4_0HyperConnection): pre = sigmoid(.) with no eps, post = 2 sigmoid,
               comb logits clamped to [lo, hi], exp(l - rowmax), `iters` full (row, column) passes with + eps, comb X
      "nocomb" HY-V4 (modular_hy_v4.py HYV4HyperConnection): pre / post only, post = mag sigmoid(.) + eps, the
               streams carried unmixed (identity)
    The mixes are fn . rms(flat X) (the flat RMS of the N d stream features, eps[1]) times scale plus base."""

    def __init__(self, c):
        super().__init__()
        U = _U()
        N, d = c.hc_mult, c.d
        self.kind, self.N, self.iters = c.hc_kind, N, c.hc_iters
        rows = 2 * N if c.hc_kind == "nocomb" else (2 + N) * N
        self.fn, self.base = _p(rows, N * d), _p(rows)
        self.scale = _p(2 if c.hc_kind == "nocomb" else 3, fill=1.0)
        self.register_buffer("eps", torch.tensor([1e-6, 1e-6], dtype=F64))      # [hc_eps, input-norm eps]
        self.register_buffer("post_c", torch.tensor(2.0, dtype=F64))
        self.register_buffer("lim", torch.tensor([-30.0, 30.0], dtype=F64))
        self._uhc = U.UHC(c) if c.hc_kind == "comb" else None
        if self._uhc is not None:                          # "comb": UHC's own parameters, read transposed
            del self.fn, self.base, self.scale

    def forward(self, X, reg=None):
        if self._uhc is not None:
            post, comb, h = self._uhc(X, reg)
            return post, comb.transpose(-1, -2), h
        B, T, N, d = X.shape
        flat = X.reshape(B, T, N * d)
        flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps[1])
        w = flat @ self.fn.T
        e = self.eps[0]
        if self.kind == "nocomb":
            pw, qw = w.split([N, N], -1)
            pb, qb = self.base.split([N, N])
            pre = torch.sigmoid(pw * self.scale[0] + pb) + e
            post = self.post_c * torch.sigmoid(qw * self.scale[1] + qb) + e
            return post, torch.eye(N, dtype=X.dtype).expand(B, T, N, N), (pre[..., None] * X).sum(2)
        pw, qw, cw = w.split([N, N, N * N], -1)
        pb, qb, cb = self.base.split([N, N, N * N])
        pre = torch.sigmoid(pw * self.scale[0] + pb)
        post = self.post_c * torch.sigmoid(qw * self.scale[1] + qb)
        lg = (cw.view(B, T, N, N) * self.scale[2] + cb.view(N, N)).clamp(float(self.lim[0]), float(self.lim[1]))
        m = torch.exp(lg - lg.amax(-1, keepdim=True))
        for _ in range(self.iters):
            m = m / (m.sum(-1, keepdim=True) + e)
            m = m / (m.sum(-2, keepdim=True) + e)
        return post, m.transpose(-1, -2), (pre[..., None] * X).sum(2)


def hc_block(c):
    """The hyper-connection module of a ULayer sub-block for c.hc_kind ("mhc": UHC, the default; "gated": Qwen4-Exp's
    GatedResidual; "comb" / "xing" / "nocomb": UHCVariant)."""
    if c.hc_kind == "gated":
        return UGatedResidual(c)
    if c.hc_kind in ("comb", "xing", "nocomb"):
        return UHCVariant(c)
    return _U().UHC(c)


def hc_final(c):
    """The final stream collapse module (None: UniversalElephantine's own mean / HyperHead / lagged pre)."""
    return UGatedResidual(c, combine=False) if c.hc_mult and c.hc_kind == "gated" else None


# --- ple_ng_rows: Qwen4-Exp's n-gram PLE on the hc streams --------------------------------------------------------

def shift_right_ignore_eos(ids, shift, eos):
    """Token t's id `shift` positions back within its EOS-delimited segment (an EOS token ends its segment), else EOS
    (modular_qwen4_exp.py Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos, the history prefix being EOS)."""
    if shift == 0:
        return ids
    B, T = ids.shape
    pos = torch.arange(T)
    eos_pos = torch.where(ids == eos, pos, -1)
    prev_incl = torch.cummax(eos_pos, dim=1).values
    prev = torch.cat([eos_pos.new_full((B, 1), -1), prev_incl[:, :-1]], dim=1)
    in_seg = pos[None] - (prev + 1)
    src = pos - shift
    got = ids.gather(1, src.clamp_min(0)[None].expand(B, -1))
    ok = (in_seg >= shift) & (src[None] >= 0)
    return torch.where(ok, got, torch.full_like(ids, eos))


class UNgramPLE(nn.Module):
    """Per-layer n-gram embedding injected into every hc stream (Qwen4-Exp PLE): e = the concatenated rows of the
    hashed n-grams (orders 2 .. n, `heads` hash heads each); k_i = rms(key_i e) wk_i, v = val e, q_i = rms(X_i) wq_i,
    g_i = sigmoid(sign(s) sqrt(max(|s|, 1e-6))), s = <k_i, q_i> / sqrt(d); u_i = g_i v; the streams get
    u + silu(conv(rms(u) wc)), conv = a depthwise causal conv over the N * d channels with dilation `dil`."""

    def __init__(self, c):
        super().__init__()
        N, d, n, H, D = c.hc_mult, c.d, c.ple_ng_n, c.ple_ng_heads, c.ple_ng_dim
        cols = (n - 1) * H
        self.N, self.d, self.n, self.H = N, d, n, H
        self.table = _p(c.ple_ng_rows, D)
        self.key, self.val = _lin(cols * D, N * d), _lin(cols * D, d)
        self.wk, self.wq, self.wc = _p(N * d, fill=1.0), _p(N * d, fill=1.0), _p(N * d, fill=1.0)
        self.register_buffer("eps", torch.tensor(1e-6, dtype=F64))
        self.register_buffer("mult", torch.ones(n, dtype=torch.long))
        self.register_buffer("primes", torch.ones(cols, dtype=torch.long))
        self.register_buffer("offsets", torch.zeros(cols, dtype=torch.long))
        self.register_buffer("eos", torch.tensor(0))
        self.conv_w = _p(N * d, max(c.ple_ng_conv, 1))
        self.dil = max(c.ple_ng_dil, 1)
        self.c = _p(1)                                   # 0: off

    def hash_ids(self, ids):
        eos = int(self.eos)
        sh = [shift_right_ignore_eos(ids, s, eos) for s in range(self.n)]
        out = []
        for g in range(2, self.n + 1):
            mixed = sh[0] * self.mult[0]
            for j in range(1, g):
                mixed = torch.bitwise_xor(mixed, sh[j] * self.mult[j])
            a = (g - 2) * self.H
            out.append(torch.remainder(mixed[..., None], self.primes[a:a + self.H]) + self.offsets[a:a + self.H])
        return torch.cat(out, -1)

    def forward(self, X, ids):
        if self.c == 0:
            return X
        B, T, N, d = X.shape
        e = self.table[self.hash_ids(ids)].flatten(-2)
        k = _grms(self.key(e), self.wk, self.eps, d).view(B, T, N, d)
        q = _grms(X.reshape(B, T, N * d), self.wq, self.eps, d).view(B, T, N, d)
        s = (k * q).sum(-1, keepdim=True) / math.sqrt(d)
        s = s.abs().clamp_min(1e-6).sqrt() * s.sign()
        u = (torch.sigmoid(s) * self.val(e)[..., None, :]).reshape(B, T, N * d)
        un = _grms(u, self.wc, self.eps, d)
        K, dl = self.conv_w.shape[1], self.dil
        up = F.pad(un.transpose(1, 2), ((K - 1) * dl, 0))
        y = sum(self.conv_w[:, j, None] * up[:, :, j * dl: j * dl + T] for j in range(K)).transpose(1, 2)
        return X + self.c * (u + F.silu(y)).view(B, T, N, d)


def engram_block(c):
    """The hc-stream n-gram memory of a ULayer: DeepSeek-V4.1's (UEngramHC, c.engram_hc rows) or Qwen4-Exp's PLE."""
    if c.engram_hc:
        return _U().UEngramHC(c)
    return UNgramPLE(c) if c.ple_ng_rows else None


# --- loaders ----------------------------------------------------------------------------------------------------

@torch.no_grad()
def _moe_from(f, mlp):
    """A Qwen3-Next-style sparse MoE block (3-D fused or per-expert experts, softmax top-k, renorm, a sigmoid-gated
    shared expert) -> UFFN."""
    U = _U()
    U._cp(f.router.weight, mlp.gate.weight)
    ex = mlp.experts
    if hasattr(ex, "gate_up_proj"):
        g, up = ex.gate_up_proj.detach().to(F64).chunk(2, dim=1)
        f.w_gate.copy_(g); f.w_up.copy_(up); U._cp(f.w_down, ex.down_proj)
    else:
        for e, x in enumerate(ex):
            U._cp(f.w_gate[e], x.gate_proj.weight); U._cp(f.w_up[e], x.up_proj.weight); U._cp(f.w_down[e], x.down_proj.weight)
    f.renorm.fill_(1.0 if getattr(mlp.gate, "norm_topk_prob", getattr(mlp, "norm_topk_prob", True)) else 0.0)
    U._dense_ffn_from(f, mlp.shared_expert)
    U._cp(f.s_router.weight, mlp.shared_expert_gate.weight)
    f.s_coef.copy_(torch.tensor([0.0, 1.0]))


@torch.no_grad()
def _gr_from(dst: UGatedResidual, src):
    U = _U()
    U._cp(dst.w, 1 + src.hc_norm.weight.detach().to(F64))
    dst.eps.fill_(float(src.hc_norm.eps))
    U._cp(dst.down.weight, src.input_mix_weight_down.weight)
    U._cp(dst.up.weight, src.input_mix_weight_up.weight)
    if dst.inject is not None:
        U._cp(dst.inject.weight, src.block_inject_weight.weight)


@torch.no_grad()
def _ple_from(dst: UNgramPLE, src):
    U = _U()
    e = src.ple_embedding
    assert e.ngram_size == dst.n and e.heads_per_ngram == dst.H
    U._cp(dst.table, e.ngram_embedding.weight)
    dst.mult.copy_(e.layer_multipliers); dst.primes.copy_(e.ngram_heads_vocab_sizes)
    dst.offsets.copy_(e.ngram_heads_offsets); dst.eos.fill_(int(e.eos_token_id))
    U._cp(dst.key.weight, src.key_proj.weight); U._cp(dst.val.weight, src.value_proj.weight)
    for w, nm in ((dst.wk, src.norm_key), (dst.wq, src.norm_query), (dst.wc, src.norm_conv)):
        U._cp(w, 1 + nm.weight.detach().to(F64))
    dst.eps.fill_(float(src.norm_key.eps))
    assert src.conv1d.dilation[0] == dst.dil and src.conv1d.bias is None
    U._cp(dst.conv_w, src.conv1d.weight[:, 0])
    dst.c.fill_(1.0)


QWEN4_LINEAR = ("linear_attention",)


@torch.no_grad()
def from_qwen4_exp(target):
    """Qwen4-Exp (Qwen3.8-Flash-Next, Whittle-Next; transformers 5.18 models/qwen4_exp): hc_count GatedResidual
    streams (hc_kind "gated") with no sub-block input norms and no final norm (the hc norm is the only norm), Qwen3.5
    gated-DeltaNet layers whose output gate is output_gate_type (sigmoid: USSM.gate[1]), gated (1 + w) qk-normed
    partial-RoPE attention, a Qwen3-Next sparse MoE + gated shared expert, and the n-gram PLE on ple_layer_ids. The
    QSA token indexer is run as dense attention (the rule decision for DSA: exact while the budget covers the
    context, indexer_budget >= T; the indexer itself is an attention branch, not this pass's)."""
    U = _U()
    tc = target.config
    tc = tc.get_text_config() if hasattr(tc, "get_text_config") else getattr(tc, "text_config", tc)
    m = getattr(target.model, "language_model", target.model)
    hd = tc.head_dim
    rot = int(hd * ((tc.rope_parameters or {}).get("partial_rotary_factor", 1.0)))
    perm = U._partial_rope_perm(hd, rot)
    layers = list(m.layers)
    g0 = next(l.linear_attn for l in layers if l.layer_type in QWEN4_LINEAR)
    kd, vd = g0.num_k_heads * g0.head_k_dim, g0.num_v_heads * g0.head_v_dim
    mlp0 = layers[0].mlp
    ple0 = next((l.ple for l in layers if l.ple is not None), None)
    pk = {}
    if ple0 is not None:
        e = ple0.ple_embedding
        pk = dict(ple_ng_rows=e.ngram_embedding.num_embeddings, ple_ng_dim=e.ngram_embedding.embedding_dim,
                  ple_ng_n=e.ngram_size, ple_ng_heads=e.heads_per_ngram, ple_ng_conv=ple0.conv1d.kernel_size[0],
                  ple_ng_dil=ple0.conv1d.dilation[0])
    c = U.UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=len(layers), heads=tc.num_attention_heads, head_dim=hd,
                  ssm_in=2 * kd + 2 * vd + 2 * g0.num_v_heads, conv_k=g0.conv_kernel_size, ssm_heads=g0.num_v_heads,
                  dk=g0.head_k_dim, dv=g0.head_v_dim, experts=mlp0.experts.num_experts,
                  expert_ff=mlp0.experts.intermediate_dim, top_k=mlp0.gate.top_k,
                  shared_ff=mlp0.shared_expert.gate_proj.out_features,
                  hc_mult=tc.hc_count, hc_kind="gated", hc_rank=tc.hc_lowrank, **pk)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    u.norm.ident.fill_(1.0)                                   # no final norm: the mixer's hc norm
    _gr_from(u.hc_final, m.hyper_connection_mixer)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    gate_act = tc.output_gate_type or tc.hidden_act
    for ul, tl in zip(u.layers, layers):
        for n_ in (ul.norm_a, ul.norm_s, ul.norm_f):
            n_.ident.fill_(1.0)                               # the sub-blocks read the hc mix directly
        _gr_from(ul.hc_a, tl.attn_hyper_connection)
        _gr_from(ul.hc_f, tl.mlp_hyper_connection)
        if tl.ple is not None:
            _ple_from(ul.engram_hc, tl.ple)
        if tl.layer_type in QWEN4_LINEAR:
            U._gdn_from(ul.ssm, tl.linear_attn)
            if gate_act == "sigmoid":
                ul.ssm.gate.zero_(); ul.ssm.gate[1] = 1
            else:
                assert gate_act == "silu", gate_act
            ul.attn.o_proj.weight.zero_()
        else:
            U._attn_from(ul.attn, tl.self_attn, tc, m.rotary_emb.inv_freq, perm=perm, gated=True, qk_norm="one_plus")
            ul.ssm.out_proj.weight.zero_()
        _moe_from(ul.ffn, tl.mlp)
    return u


# --- cat_block: Zamba / Zamba2's shared transformer block ---------------------------------------------------------

class UCatBlock(nn.Module):
    """A sub-block over concat(x, x0) (x0 = the input embedding) whose output feeds the SSM's input but not the
    residual (Zamba / Zamba2 hybrid layers, modeling_zamba2.py Zamba2HybridLayer / Zamba2AttentionDecoderLayer):
    t = W_lin mlp(rms(attn(rms_2d([x, x0])))), no inner residual; the layer's SSM then reads norm_s(x + t). The
    attention has its own heads over the 2d-wide input (its output is the first d rows of a 2d o_proj); a shared
    (tied) block with per-invocation LoRA adapters is, on the untied host, the per-layer sum W + B A."""

    def __init__(self, c):
        super().__init__()
        U = _U()
        d = c.d
        self.d = d
        ca = U.UConfig(vocab=1, d=2 * d, L=1, heads=c.cat_heads, head_dim=c.cat_head_dim)
        self.norm_in, self.attn = U.UNorm(2 * d), U.UAttention(ca)
        cf = U.UConfig(vocab=1, d=d, L=1, shared_ff=c.cat_ff)
        self.norm_ff, self.ffn = U.UNorm(d), U.UFFN(cf)
        self.lin = _lin(d, d)
        self.c = _p(1)                                   # 0: off

    def forward(self, x, reg, causal):
        if self.c == 0:
            return 0
        a = self.attn(self.norm_in(torch.cat((x, reg["x0"]), -1)), causal)[..., :self.d]
        return self.c * self.lin(self.ffn(self.norm_ff(a)))


# --- value embeddings, smear, x0 lambdas, backout (nanochat / ARLoop / sn38) --------------------------------------

class UValueEmb(nn.Module):
    """Per-token value embeddings mixed into the attention values (ResFormer-style; nanochat value_embeds, ARLoop's
    shared table, sn38's aux tables): v_h <- v_h + c s sigmoid(W_g x[:gc])_{kv(h)} E[ids]_{kv(h)}, per KV head (the
    host's MHA heads read their group's), x = the attention input; s = the code's gate scale (3 nanochat, 2 ARLoop)."""

    def __init__(self, c, H, hd):
        super().__init__()
        nkv, gc = c.ve_heads, c.ve_gate_ch
        self.nkv, self.hd, self.gc = nkv, hd, gc
        self.table = _p(c.vocab, nkv * hd)
        self.gate = _lin(gc, nkv)
        self.register_buffer("head_kv", torch.arange(H) * nkv // H)
        self.register_buffer("scale", torch.tensor(3.0, dtype=F64))
        self.c = _p(1)                                   # 0: off

    def forward(self, v, x, reg):
        if self.c == 0:
            return v
        B, T = v.shape[:2]
        e = self.table[reg["ids"]].view(B, T, self.nkv, self.hd)[:, :, self.head_kv]
        g = self.scale * torch.sigmoid(self.gate(x[..., :self.gc]))[..., self.head_kv]
        return v + self.c * g[..., None] * e


class USmear(nn.Module):
    """The embedding smear (nanochat): x_t <- x_t + lam sigmoid(W x_t[:ch]) x_{t-1} for t >= 1."""

    def __init__(self, c):
        super().__init__()
        self.ch = c.smear_ch
        self.gate = _lin(c.smear_ch, 1)
        self.lam = _p(1)

    def forward(self, x):
        g = self.lam * torch.sigmoid(self.gate(x[:, 1:, :self.ch]))
        return torch.cat((x[:, :1], x[:, 1:] + g * x[:, :-1]), 1)


class UX0Mix(nn.Module):
    """x <- r x + s x0 at the layer start (nanochat resid_lambdas / x0_lambdas; x0 = the normed, smeared embedding)."""

    def __init__(self):
        super().__init__()
        self.r, self.s = _p(1, fill=1.0), _p(1)

    def forward(self, x, reg):
        return self.r * x + self.s * reg["x0"]

    def enter(self, x, reg):
        return self.forward(x, reg)

    def leave(self, x, reg):
        return x


class UEarlyExit(nn.Module):
    """Per-token early exit among the tapped candidates h_1 .. h_S (Ouro's UT steps, modeling_ouro.py:654-656,
    :770-781, :823-850): lam_s = sigmoid(W h_s + b), the exit pdf p_s = lam_s prod_{r<s} (1 - lam_r) (the last step
    takes the remainder); a token exits at the first s whose cumulative pdf reaches `threshold` (never: the last), or
    at the fixed step `step` (>= 0). threshold inf and step -1: always the last candidate."""

    def __init__(self, c):
        super().__init__()
        self.gate = _U()._lin(c.d, 1, bias=True)
        self.register_buffer("threshold", torch.tensor(float("inf"), dtype=F64))
        self.register_buffer("step", torch.tensor(-1.0, dtype=F64))

    def forward(self, hs):
        S = len(hs)
        if 0 <= self.step < S:
            return hs[int(self.step)]
        if not torch.isfinite(self.threshold):
            return hs[-1]
        rem, pdf = torch.ones_like(hs[0][..., 0]), []
        for i, h in enumerate(hs):
            lam = torch.sigmoid(self.gate(h)[..., 0])
            if i < S - 1:
                pdf.append(lam * rem)
                rem = rem * (1.0 - lam)
            else:
                pdf.append(rem)
        hit = torch.cumsum(torch.stack(pdf, 2), 2) >= self.threshold
        steps = torch.argmax(hit.to(F64), 2)
        steps[~hit.any(2)] = S - 1
        st = torch.stack(hs, 2)
        return st.gather(2, steps[..., None, None].expand(-1, -1, 1, st.shape[-1]))[:, :, 0]


class UTwoState(nn.Module):
    """Two residual states (HRM's z_H, z_L: modular_hrm_text.py HrmTextModel.forward) kept in the register
    reg["z"] = [z_H, z_L] (z_H starts as the embedding, z_L as the model's hrm_z0): a stack's first layer reads
    x = a z_H + b z_L (enter, a = mix[0], b = mix[1], on when on != 0), its last layer writes x into z_H (put = 1) or
    z_L (put = 2). Off: the plain residual stream."""

    def __init__(self):
        super().__init__()
        self.register_buffer("on", torch.tensor(0.0, dtype=F64))
        self.register_buffer("mix", torch.ones(2, dtype=F64))
        self.register_buffer("put", torch.tensor(0.0, dtype=F64))

    def enter(self, x, reg):
        if self.on == 0:
            return x
        z = reg["z"]
        return self.mix[0] * z[0] + self.mix[1] * z[1]

    def leave(self, x, reg):
        if self.put != 0:
            reg["z"][int(self.put) - 1] = x
        return x


class ULoopInject(nn.Module):
    """Loop input re-injection (Dory, modeling_dory.py DoryModel.forward l. 618-620): the layer that ends the input
    section snapshots reg["inj"] = n(x) (snap != 0); a pass's first layer starts from x <- n(x) + reg["inj"] (inject
    != 0); n = a weight-free norm (UNorm: Dory's L2 is RMS / sqrt(d) with eps / d)."""

    def __init__(self, c):
        super().__init__()
        self.norm = _U().UNorm(c.d)
        self.register_buffer("snap", torch.tensor(0.0, dtype=F64))
        self.register_buffer("inject", torch.tensor(0.0, dtype=F64))

    def enter(self, x, reg):
        return self.norm(x) + reg["inj"] if self.inject != 0 else x

    def leave(self, x, reg):
        if self.snap != 0:
            reg["inj"] = self.norm(x)
        return x


class URGain(nn.Module):
    """Per-channel residual gains (and a bias) per sub-block: x <- gx_i * x + go_i * out + gb_i (i = 0 the attention /
    SSM block, 1 the FFN); ones / zero = the plain residual. nGPT's interpolation h + r (f - h) is gx = 1 - r, go = r
    (Dory); Doge's scaled residual is gx; ZAYA's ResidualScaling merge s_r (x + b_r) + s_h (f + b_h) is gx = s_r,
    go = s_h, gb = s_r b_r + s_h b_h."""

    def __init__(self, c):
        super().__init__()
        self.gx, self.go, self.gb = _p(2, c.d, fill=1.0), _p(2, c.d, fill=1.0), _p(2, c.d)
        # a data-dependent scalar gate on the update: out <- sigmoid(<u, gw_i> + gc_i) out, u = the sub-block's normed
        # input (CFRD's attn_gate / ffn_gate); g_on = 0: off
        self.gw, self.gc = _p(2, c.d), _p(2)
        self.register_buffer("g_on", torch.zeros(2, dtype=F64))

    def mix(self, i, x, out, u=None):
        if self.g_on[i] != 0:
            out = out * torch.sigmoid(u @ self.gw[i] + self.gc[i])[..., None]
        return self.gx[i] * x + self.go[i] * out + self.gb[i]


class USelfCond(nn.Module):
    """Self-conditioning of a diffusion LM (QwenDLM-ORL SDARSelfConditioning, DiffusionGemmaSelfConditioning): the
    previous denoising step's soft embeddings s (forward(selfcond=s)) enter the input embeddings as
    x <- post(x + o SwiGLU(p rms(s) w)), post = a weight-free RMS norm times a scale (post_c 1) or the identity."""

    def __init__(self, c):
        super().__init__()
        U = _U()
        self.pre = U.UNorm(c.d)
        f = c.selfcond_ff
        self.gate, self.up, self.down = _lin(c.d, f), _lin(c.d, f), _lin(f, c.d)
        self.register_buffer("scales", torch.tensor([1.0, 1.0, 1.0], dtype=F64))   # [pre, out, post]
        self.register_buffer("post_c", torch.tensor(0.0, dtype=F64))
        self.register_buffer("post_eps", torch.tensor(1e-6, dtype=F64))
        self.register_buffer("act", torch.tensor(0.0, dtype=F64))   # 0: SiLU (QwenDLM), 1: tanh GeLU (DiffusionGemma)

    def forward(self, x, sig):
        s = self.pre(sig.to(F64)) * self.scales[0]
        g = F.silu(self.gate(s)) if self.act == 0 else F.gelu(self.gate(s), approximate="tanh")
        y = x + self.down(g * self.up(s)) * self.scales[1]
        if self.post_c != 0:
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.post_eps) * self.scales[2]
        return y


class UCanon(nn.Module):
    """A Canon layer (Allen-Zhu's Canon / dhara modeling_dhara_ar.py CanonLayer): a causal depthwise conv of kernel K
    over time per channel, x <- x + act(conv(x) + b) (res 1; act SiLU if act 1). c = 0: off (the identity)."""

    def __init__(self, C, K):
        super().__init__()
        self.w, self.b = _p(C, K), _p(C)
        self.register_buffer("res", torch.tensor(1.0, dtype=F64))
        self.register_buffer("act", torch.tensor(0.0, dtype=F64))
        self.c = _p(1)

    def forward(self, x):
        if self.c == 0:
            return x
        T, K = x.shape[1], self.w.shape[1]
        xp = F.pad(x.transpose(1, 2), (K - 1, 0))
        y = (sum(self.w[:, j, None] * xp[:, :, j:j + T] for j in range(K)) + self.b[:, None]).transpose(1, 2)
        if self.act != 0:
            y = F.silu(y)
        return self.res * x + y


_NORM_CANON = []


def UNormCanon(d, K, rank=0):
    """A pre-norm followed by a Canon conv (Canon-A / -C: dhara canon_a after input_layernorm, canon_c after
    post_attention_layernorm): a UNorm subclass (its weight / eps / gate load as any UNorm's) with .canon."""
    if not _NORM_CANON:
        U = _U()

        class _NormCanon(U.UNorm):
            def __init__(self, d, K, rank):
                super().__init__(d, rank)
                self.canon = UCanon(d, K)

            def forward(self, x):
                return self.canon(super().forward(x))
        _NORM_CANON.append(_NormCanon)
    return _NORM_CANON[0](d, K, rank)


class UCanonQKV(nn.Module):
    """Canon-B: a Canon conv on each of q, k, v (the host's H heads; GQA targets' k / v convs tiled), before the
    rotary and the qk-norm."""

    def __init__(self, C, K):
        super().__init__()
        self.q, self.k, self.v = UCanon(C, K), UCanon(C, K), UCanon(C, K)

    def forward(self, q, k, v):
        sh = q.shape
        f = lambda m, t: m(t.reshape(sh[0], sh[1], -1)).view(sh)
        return f(self.q, q), f(self.k, k), f(self.v, v)


class UBigramHash(nn.Module):
    """A hashed bigram embedding added to the token embedding (Crowd-v1 CrowdEngram, modeling_crowd.py l. 86-97):
    E[(prev P + id) mod buckets], prev = the previous token (pad at position 0)."""

    def __init__(self, c):
        super().__init__()
        self.table = _p(c.ngram_in_buckets, c.d)
        self.register_buffer("prime", torch.tensor(1000003))
        self.register_buffer("pad", torch.tensor(0))

    def forward(self, ids):
        prev = torch.cat([torch.full_like(ids[:, :1], int(self.pad)), ids[:, :-1]], dim=1)
        return self.table[(prev * self.prime + ids) % self.table.shape[0]]


class URouterMLP(nn.Module):
    """An MLP router: logits = W2 tanh(W1 x) (LLaMA-MoE TopKBalancedNoisyGate gate_network "mlp",
    modeling_llama_moe_hf.py l. 440-455; inference has no noise)."""

    def __init__(self, c):
        super().__init__()
        self.w1, self.w2 = _lin(c.d, c.experts), _lin(c.experts, c.experts)

    def forward(self, x):
        return self.w2(torch.tanh(self.w1(x)))


class UMultimax(nn.Module):
    """ERNIE-5.5's multimax on the logits (vLLM-ERNIE model.py Ernie5_5ParallelLMHead.multimax l. 93-101):
    l + t0 relu(r0 - l) + t1 relu(l - r1) + t2 relu(r2 - l)^2 + t3 relu(l - r3)^2 (t = 0: the identity)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("r", torch.zeros(4, dtype=F64))
        self.register_buffer("t", torch.zeros(4, dtype=F64))

    def forward(self, l):
        r, t = self.r, self.t
        return (l + t[0] * (r[0] - l).relu() + t[1] * (l - r[1]).relu() + t[2] * (r[2] - l).relu().square()
                + t[3] * (l - r[3]).relu().square())


class ULatentIn(nn.Module):
    """The routed experts' input as a normed latent (Muse-Spark MuseSparkMoE pre_expert_proj + the centered RMSNorm,
    transformers main modular_muse_spark.py l. 378-395): z = rms(P x) w over the m latent channels, zero-padded to d."""

    def __init__(self, d, m):
        super().__init__()
        self.d, self.m = d, m
        self.P, self.w = _p(m, d), _p(m, fill=1.0)
        self.register_buffer("eps", torch.tensor(1e-6, dtype=F64))

    def forward(self, x):
        z = x @ self.P.T
        z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.eps) * self.w
        return F.pad(z, (0, self.d - self.m))


class ULatentPostNorm(nn.Module):
    """A norm of each expert's latent output (MuseSparkTextInputScaledRMSNorm, l. 300-305: scale by s, then a
    weight-free RMS over the m latent channels): eo (..., E, d) -> the same, padded."""

    def __init__(self, d, m):
        super().__init__()
        self.d, self.m = d, m
        self.s = _p(m, fill=1.0)
        self.register_buffer("eps", torch.tensor(1e-6, dtype=F64))

    def forward(self, eo):
        z = eo[..., :self.m] * self.s
        z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.eps)
        return F.pad(z, (0, self.d - self.m))


def init_layer(layer, c):
    """Build a ULayer's stream-pass branches from its UConfig (each None / off unless the config asks for it)."""
    layer.cat = UCatBlock(c) if c.cat_block else None
    layer.attn.ve = UValueEmb(c, layer.attn.H, layer.attn.hd) if c.ve_dim else None
    layer.x0mix = UX0Mix() if c.x0_mix else None
    layer.register_buffer("bo_write", torch.tensor(0.0, dtype=F64))   # 1: publish this layer's output (backout)
    layer.register_buffer("ar_sw", torch.tensor(0.0, dtype=F64))     # AttnRes source window (0: every source)
    # ScMoE: [1: this layer's routed experts are deferred to the register, 1: add the deferred output after the FFN]
    layer.register_buffer("moe_defer", torch.zeros(2, dtype=F64))
    layer.register_buffer("ee_tap", torch.tensor(0.0, dtype=F64))    # 1: this layer's output is an exit candidate
    layer.register_buffer("moe_par", torch.tensor(0.0, dtype=F64))   # 1: with ffn_split, routed experts read the layer input
    layer.zstate = UTwoState() if c.two_state else None
    layer.loopinj = ULoopInject(c) if c.loop_inject else None
    layer.xhooks = nn.ModuleList(h for h in (layer.x0mix, layer.zstate, layer.loopinj) if h is not None)
    layer.rgain = URGain(c) if c.res_gain else None
    layer.attn.o_late = _U().UNorm(layer.attn.H * layer.attn.hd) if c.o_late_norm else None
    layer.ffn.s_hid = _U().UNorm(c.shared_ff) if c.ffn_hidden_norm else None
    layer.ffn.router_mlp = URouterMLP(c) if c.router_mlp else None
    layer.ffn.router2 = _lin(c.d, c.experts) if c.router_dual else None
    if c.latent_moe:
        layer.ffn.e_pre, layer.ffn.e_post = ULatentIn(c.d, c.latent_moe), ULatentPostNorm(c.d, c.latent_moe)
        layer.ffn.e_out = _lin(c.d, c.d)
    if c.canon_k:                              # Canon layers (off until a loader sets c = 1)
        K, at = c.canon_k, layer.attn
        if "A" in c.canon_sites:
            layer.norm_a = UNormCanon(c.d, K, c.norm_gate_rank)
        if "C" in c.canon_sites:
            layer.norm_f = UNormCanon(c.d, K, c.norm_gate_rank)
        if "B" in c.canon_sites:
            at.canon_b = UCanonQKV(at.H * at.hd, K)
        if "D" in c.canon_sites:
            layer.ffn.s_hid = UCanon(c.shared_ff, K)


def init_model(u, c):
    """Build the model-level stream-pass branches (the smear; the backout coefficient, 0 = off)."""
    u.smear = USmear(c) if c.smear_ch else None
    u.early_exit = UEarlyExit(c) if c.early_exit else None
    u.hrm_z0 = _p(c.d) if c.two_state else None
    u.selfcond = USelfCond(c) if c.selfcond_ff else None
    u.ngram_in = UBigramHash(c) if c.ngram_in_buckets else None
    u.multimax = UMultimax() if c.multimax else None
    if c.norm_gate_rank:                       # the gated-norm site on the embedding norm too (Grug's embed_gated_norm)
        u.emb_norm = _U().UNorm(c.d, c.norm_gate_rank)
    u.bo_c = _p(1)


@torch.no_grad()
def _zamba2_shared_into(cb: UCatBlock, blk, layer_idx, use_rope, inv_freq):
    """Zamba2AttentionDecoderLayer (block_id = layer_idx's shared block) at invocation layer_idx -> UCatBlock."""
    U = _U()
    at, ml, d = blk.self_attn, blk.feed_forward, cb.d
    a = cb.attn
    H, hd = a.H, a.hd
    n_kv = at.k_proj.out_features // hd
    W = lambda lin: lin.weight.detach().to(F64)
    q, k, v = W(at.q_proj), W(at.k_proj), W(at.v_proj)
    if at.config.use_shared_attention_adapter:
        i = at.layer_dic[layer_idx]
        lora = lambda seq: W(seq[1]) @ W(seq[0]) if isinstance(seq, nn.Sequential) else 0
        q = q + lora(at.linear_q_adapter_list[i])
        k = k + lora(at.linear_k_adapter_list[i])
        v = v + lora(at.linear_v_adapter_list[i])
    a.q_proj.weight.copy_(q)
    a.k_proj.weight.copy_(U._tile_kv(k, n_kv, H, hd))
    a.v_proj.weight.copy_(U._tile_kv(v, n_kv, H, hd))
    a.o_proj.weight[:d] = W(at.o_proj)
    a.scale.fill_(float(at.scaling))
    a.inv_freq.zero_()
    if use_rope:
        a.inv_freq.copy_(inv_freq.to(F64))
    U._norm_from(cb.norm_in, blk.input_layernorm)
    U._norm_from(cb.norm_ff, blk.pre_ff_layernorm)
    gu = W(ml.gate_up_proj)
    i = ml.layer_dic[layer_idx]
    ad = ml.gate_up_proj_adapter_list[i]
    if isinstance(ad, nn.Sequential):
        gu = gu + W(ad[1]) @ W(ad[0])
    g, up = gu.chunk(2, 0)
    f = cb.ffn
    f.s_gate.weight.copy_(g); f.s_up.weight.copy_(up); U._cp(f.s_down.weight, ml.down_proj.weight)
    if ml.gate_up_proj.bias is not None:
        bg, bu = ml.gate_up_proj.bias.detach().to(F64).chunk(2)
        f.s_gate.bias.copy_(bg); f.s_up.bias.copy_(bu); U._cp(f.s_down.bias, ml.down_proj.bias)
    U._set_act(f, ml.config.hidden_act, True)
    cb.c.fill_(1.0)


def _zamba2_mixer(mx):
    """Zamba2MambaMixer under mamba2_block's names: its prefill clamps dt below at time_step_min (torch_forward)."""
    from types import SimpleNamespace as NS
    keys = ("intermediate_size", "ssm_state_size", "num_heads", "head_dim", "n_groups", "conv_dim", "hidden_size",
            "conv_kernel_size", "in_proj", "conv1d", "dt_bias", "A_log", "D", "norm", "out_proj")
    return NS(**{k: getattr(mx, k) for k in keys}, time_step_limit=(float(mx.time_step_min), float("inf")))


@torch.no_grad()
def from_zamba2(target):
    """Zamba2 (transformers Zamba2ForCausalLM): Mamba-2 layers; on the hybrid layers a shared transformer block
    (num_mem_blocks of them, cycled) over concat(x, embedding) -> its own linear -> added to the Mamba-2 layer's input
    only (UCatBlock); the per-invocation LoRA adapters fold into the untied copy's weights."""
    U = _U()
    tc, m = target.config, target.model
    layers = list(m.layers)
    mx0 = (layers[0].mamba_decoder if hasattr(layers[0], "mamba_decoder") else layers[0]).mamba
    blk0 = next((l.shared_transformer for l in layers if hasattr(l, "shared_transformer")), None)
    I, cd, nh = mx0.intermediate_size, mx0.conv_dim, mx0.num_heads
    c = U.UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=len(layers), ssm_in=I + cd + nh, conv_k=mx0.conv_kernel_size,
                  ssm_heads=nh, dk=mx0.ssm_state_size, dv=mx0.head_dim,
                  cat_block=blk0 is not None, cat_heads=tc.num_attention_heads, cat_head_dim=tc.attention_head_dim,
                  cat_ff=tc.intermediate_size)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.final_layernorm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    inv = m.rotary_emb.inv_freq if tc.use_mem_rope else None
    for i, (ul, tl) in enumerate(zip(u.layers, layers)):
        md = tl.mamba_decoder if hasattr(tl, "mamba_decoder") else tl
        U._mamba2_into(ul, _zamba2_mixer(md.mamba), md.input_layernorm)
        if hasattr(tl, "shared_transformer"):
            _zamba2_shared_into(ul.cat, tl.shared_transformer, i, tc.use_mem_rope, inv)
            U._cp(ul.cat.lin.weight, tl.linear.weight)
    return u


def _rms_free(n, eps):
    """A weight-free RMSNorm (weight 1) with the target's eps."""
    with torch.no_grad():
        n.weight.fill_(1.0)
        n.eps.fill_(float(eps))


def _first(obj, *names):
    return next((getattr(obj, n) for n in names if getattr(obj, n, None) is not None), None)


@torch.no_grad()
def from_nanochat(target, *, eps=None):
    """nanochat ports -- nanochat_gpt (jkminder/d*_optimized, modeling_nanochat_gpt.py), gaggle (castorini/gaggle-
    nanochat, modeling_gaggle.py) and sn38-nanochrono (chronollm/sn38 modeling_nanochrono.py): weight-free RMSNorms
    (embedding, pre-attention, pre-MLP, final; eps = F.rms_norm's default, the dtype's machine epsilon unless `eps` is
    given), the smear, x <- r_i x + s_i x0 at every layer, GQA attention with nanochat's -theta rotary (the host's
    inv_freq negated), weight-free q / k norms AFTER the rotary times the qk sharpening scale, left windows per layer,
    gated value embeddings on the VE layers (scale 3), relu^2 MLP, the backout / tap x - lambda x_mid before the final
    norm, the logit softcap. Reads each port's module names."""
    U = _U()
    tc = target.config
    m = target.model
    gag = hasattr(m, "transformer")
    blocks = list(m.transformer["h"] if gag else m.layers)
    att = lambda b: b.attn if gag else b.self_attn
    a0 = att(blocks[0])
    H, nkv = _first(a0, "n_head", "num_heads"), _first(a0, "n_kv_head", "num_kv_heads")
    hd = a0.head_dim
    proj = lambda at: (_first(at, "c_q", "q_proj"), _first(at, "c_k", "k_proj"), _first(at, "c_v", "v_proj"),
                       _first(at, "c_proj", "o_proj"))
    mlp = lambda b: (_first(b.mlp, "c_fc", "up_proj"), _first(b.mlp, "down_proj", "c_proj"))
    d = proj(a0)[0].in_features
    emb = m.transformer["wte"] if gag else m.embed_tokens
    ves = _first(m, "value_embeds", "aux_embeds") or {}
    vgate = lambda at: _first(at, "ve_gate", "g_proj")
    gc = _first(a0, "ve_gate_channels", "gate_dim") or 1
    smear, lam = _first(m, "smear_gate", "mix_gate"), _first(m, "smear_lambda", "mix_scale")
    resid, x0l = _first(m, "resid_lambdas", "residual_scales"), _first(m, "x0_lambdas", "skip_scales")
    eps = float(torch.finfo(emb.weight.dtype).eps) if eps is None else eps
    c = U.UConfig(vocab=emb.num_embeddings, d=d, L=len(blocks), heads=H, head_dim=hd,
                  shared_ff=mlp(blocks[0])[0].out_features, ve_dim=nkv * hd if len(ves) else 0, ve_heads=nkv,
                  ve_gate_ch=gc, smear_ch=smear.in_features if smear is not None else 0,
                  x0_mix=resid is not None or x0l is not None)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, emb.weight)
    u.emb_norm_c.fill_(1.0); _rms_free(u.emb_norm, eps); _rms_free(u.norm, eps)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    cap = _first(tc, "logit_softcap", "final_logit_softcapping")
    if cap:
        u.final_cap.fill_(float(cap))
    if smear is not None:
        U._cp(u.smear.gate.weight, smear.weight); U._cp(u.smear.lam, lam)
    windows = list(_first(m, "window_sizes", "windows"))
    if gag:
        bo = tc.n_layer // 2
    else:
        bo = getattr(tc, "backout_layer", None) if hasattr(m, "backout_lambda") else getattr(tc, "tap_layer", None)
    bo_lam = _first(m, "backout_lambda", "tap_scale")
    if bo is not None and bo_lam is not None and 0 <= bo < len(blocks):
        U._cp(u.bo_c, bo_lam)
        u.layers[bo].bo_write.fill_(1.0)
    inv = 1.0 / (float(tc.rope_theta) ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    for i, (ul, b) in enumerate(zip(u.layers, blocks)):
        at = att(b)
        lq, lk, lv, lo = proj(at)
        a = ul.attn
        a.q_proj.weight.copy_(lq.weight.detach().to(F64))
        a.k_proj.weight.copy_(U._tile_kv(lk.weight.detach().to(F64), nkv, H, hd))
        a.v_proj.weight.copy_(U._tile_kv(lv.weight.detach().to(F64), nkv, H, hd))
        U._cp(a.o_proj.weight, lo.weight)
        a.inv_freq.copy_(-inv)                                # nanochat rotates by -theta
        a.qk_post.fill_(1.0)                                  # q / k norms after the rotary
        sc = float(_first(at, "qk_scale", "qk_sharpen_scale") or 1.0)
        for coef, w in ((a.q_coef, a.q_nw), (a.k_coef, a.k_nw)):
            coef.copy_(torch.tensor([0.0, 1.0])); w.fill_(sc)
        a.qk_eps.fill_(eps)
        a.window.fill_(float(windows[i]) if windows[i] >= 0 else float("inf"))
        a.scale.fill_(hd ** -0.5)
        _rms_free(ul.norm_a, eps); _rms_free(ul.norm_f, eps)
        if str(i) in ves:
            ve = a.ve
            U._cp(ve.table, ves[str(i)].weight); U._cp(ve.gate.weight, vgate(at).weight)
            ve.scale.fill_(3.0); ve.c.fill_(1.0)
        if ul.x0mix is not None:
            if resid is not None:
                ul.x0mix.r.fill_(float(resid[i]))
            if x0l is not None:
                ul.x0mix.s.fill_(float(x0l[i]))
        f = ul.ffn
        upw, dn = mlp(b)
        U._cp(f.s_up.weight, upw.weight); U._cp(f.s_down.weight, dn.weight)
        f.c_relu2.fill_(1.0)
    return u


@torch.no_grad()
def from_arloop(target, *, eps=None):
    """ARLoop and its PreNorm-loop baseline (osunlp/ARLoop-*, modeling_arloop.py; osunlp/PreNormLoop-0.6B,
    modeling_baseline.py), unrolled: loop_count passes over the physical layers (effective layer loop * P + i). Shared
    value embeddings (one table, every VE layer's UValueEmb a copy; gate 2 sigmoid(W x[:32])), adjacent-pair RoPE at
    positions / rope_condense_ratio, weight-free q / k norms after the rotary (F.rms_norm's dtype eps), GQA, a fused
    SwiGLU, a tied head with an embedding scale. ARLoop's residual is Block AttnRes over attnres_block_size / 2 layers
    with a source window (the embedding + the last sw_blocks: ULayer.ar_sw) and the first partial sum starting at the
    embedding (attnres_emb_partial); each route's output norm is the block's own norm; the final route has none."""
    U = _U()
    tc = target.config
    tr = target.transformer
    P, loops = tc.physical_layers, tc.loop_count
    L = P * loops
    H, nkv, hd, d = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.hidden_size
    ar = hasattr(target, "attnres_queries")
    blk0 = tr.h[0]
    ff = blk0.mlp.proj.in_features
    sve = target.shared_value_embedding
    gate0 = next(b.attn.ve_gate for b in tr.h if b.attn.ve_gate is not None)
    eps0 = float(torch.finfo(sve.weight.dtype).eps) if eps is None else eps
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=L, heads=H, head_dim=hd, shared_ff=ff, ve_dim=nkv * hd, ve_heads=nkv,
                  ve_gate_ch=gate0.in_features, attnres_block=tc.attnres_block_size // 2 if ar else 0,
                  attnres_emb_partial=ar)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, tr.wte.weight * float(tc.embedding_scale))
    U._cp(u.lm_head.weight, tr.wte.weight)
    U._norm_from(u.norm, tr.ln_f)
    if tc.logit_softcap is not None:
        u.final_cap.fill_(float(tc.logit_softcap))
    perm = U._interleaved_rope_perm(hd, hd)
    inv = 1.0 / (float(tc.rope_theta) ** (torch.arange(0, hd, 2, dtype=F64) / hd)) / float(tc.rope_condense_ratio)
    if ar:
        U._cp(u.ar_q_out, target.final_attnres_query); U._cp(u.ar_w_out, target.final_attnres_norm.weight)
    for e, ul in enumerate(u.layers):
        b = tr.h[e % P]
        at, a = b.attn, ul.attn
        Wp = lambda w: w.view(-1, hd, w.shape[-1])[:, perm].reshape(w.shape)
        a.q_proj.weight.copy_(Wp(at.c_q.weight.detach().to(F64)))
        a.k_proj.weight.copy_(Wp(U._tile_kv(at.c_k.weight.detach().to(F64), nkv, H, hd)))
        a.v_proj.weight.copy_(U._tile_kv(at.c_v.weight.detach().to(F64), nkv, H, hd))
        U._cp(a.o_proj.weight, at.c_proj.weight)
        a.inv_freq.copy_(inv)
        if tc.qk_norm:
            a.qk_post.fill_(1.0)
            for coef in (a.q_coef, a.k_coef):
                coef.copy_(torch.tensor([0.0, 1.0]))
            a.qk_eps.fill_(eps0)
        a.scale.fill_(hd ** -0.5)
        if at.ve_gate is not None:
            U._cp(a.ve.table, sve.weight); U._cp(a.ve.gate.weight, at.ve_gate.weight)
            a.ve.scale.fill_(2.0); a.ve.c.fill_(1.0)
        na, nf = (b.attn_norm, b.mlp_norm) if ar else (b.norm_1, b.norm_2)
        U._norm_from(ul.norm_a, na); U._norm_from(ul.norm_f, nf)
        g, up = b.mlp.fc.weight.detach().to(F64).chunk(2, 0)
        f = ul.ffn
        f.s_gate.weight.copy_(g); f.s_up.weight.copy_(up); U._cp(f.s_down.weight, b.mlp.proj.weight)
        if ar:
            U._cp(ul.ar_qa, target.attnres_queries[2 * e]); U._cp(ul.ar_wa, target.attnres_norms[2 * e].weight)
            U._cp(ul.ar_qf, target.attnres_queries[2 * e + 1]); U._cp(ul.ar_wf, target.attnres_norms[2 * e + 1].weight)
            ul.ar_eps.fill_(float(target.attnres_norms[2 * e].eps))
            ul.ar_sw.fill_(float(tc.sw_blocks))
    return u


@torch.no_grad()
def from_k2_horizon(target):
    """K2-Horizon / Jais-3 dense (IFM/K2-Horizon-*, modeling_k2_horizon.py): a Llama body (attention_gate_func None,
    query_key_norm False, rope_head_dim = head_dim: K2HorizonAttention is LlamaAttention; K2HorizonMLP a SwiGLU) whose
    RMSNorms -- input, post-attention, final -- normalise per group of hidden / layernorm_num_groups channels (UNorm
    groups: the grouped RMSNorm)."""
    U = _U()
    tc = target.config
    assert getattr(tc, "attention_gate_func", None) is None and not getattr(tc, "query_key_norm", False)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    assert (getattr(tc, "rope_head_dim", None) or hd) == hd and not getattr(tc, "num_experts", 0)
    u = U.from_llama(target)
    g = float(getattr(tc, "layernorm_num_groups", 1) or 1)
    for n_ in [u.norm] + [n for ul in u.layers for n in (ul.norm_a, ul.norm_f)]:
        n_.groups.fill_(g)
    return u


@torch.no_grad()
def from_longcat_flash(target):
    """LongCat-Flash (transformers LongcatFlashForCausalLM; the snapshot's LongCat configs, read by architecture): each
    decoder layer is two host layers -- [attn0, mlp0] and [attn1, mlp1] -- with the shortcut-connected MoE (ScMoE): the
    routed experts read layer 2i's FFN input (post_attention_layernorm[0] of x1) and their output joins the residual
    after layer 2i + 1's FFN (ULayer.moe_defer). Zero-computation experts return their input (UFFN.exp_ident); the
    router is a softmax over all experts selected on scores + e_score_correction_bias, not renormalised, times
    routed_scaling_factor. MLA with the q / kv LoRA scales sqrt(hidden / rank) folded into q_b / kv_b, interleaved
    rotary."""
    from types import SimpleNamespace as NS
    U = _U()
    tc, m = target.config, target.model
    layers = list(m.layers)[: tc.num_layers]
    H, nope, rope, vd = tc.num_attention_heads, tc.qk_nope_head_dim, tc.qk_rope_head_dim, tc.v_head_dim
    kl, ql, d = tc.kv_lora_rank, tc.q_lora_rank, tc.hidden_size
    hd = max(nope + rope, vd)
    hd += hd % 2
    moe0 = layers[0].mlp
    n_real = tc.n_routed_experts
    E = n_real + (tc.zero_expert_num or 0)
    ef = tc.expert_ffn_hidden_size
    sff = layers[0].mlps[0].gate_proj.out_features
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=2 * len(layers), heads=H, head_dim=hd, experts=E, expert_ff=ef,
                  top_k=moe0.router.top_k, shared_ff=sff, attn_norm_feats=ql + kl)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.norm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    re_ = m.rotary_emb

    class _L:                                                 # a linear with a folded scale
        def __init__(self, lin, s):
            self.weight = lin.weight.detach().to(F64) * s
            self.bias = None
    for i, tl in enumerate(layers):
        for j in (0, 1):
            ul = u.layers[2 * i + j]
            ml = tl.self_attn[j]
            sh = NS(q_a_proj=ml.q_a_proj, q_a_layernorm=ml.q_a_layernorm, kv_a_proj_with_mqa=ml.kv_a_proj_with_mqa,
                    kv_a_layernorm=ml.kv_a_layernorm, q_b_proj=_L(ml.q_b_proj, ml.mla_scale_q_lora),
                    kv_b_proj=_L(ml.kv_b_proj, ml.mla_scale_kv_lora), o_proj=ml.o_proj)
            U._mla_from(ul.attn, sh, d=d, H=H, nope=nope, rope=rope, vd=vd, kl=kl, ql=ql, inv_freq=re_.inv_freq,
                        att_s=float(re_.attention_scaling), interleave=True, scale=ml.scaling)
            U._norm_from(ul.norm_a, tl.input_layernorm[j])
            U._norm_from(ul.norm_f, tl.post_attention_layernorm[j])
            f = ul.ffn
            U._dense_ffn_from(f, tl.mlps[j])
            U._set_act(f, tc.hidden_act, True)
        ua, ub = u.layers[2 * i], u.layers[2 * i + 1]
        ua.moe_defer[0] = 1; ub.moe_defer[1] = 1
        f, mo = ua.ffn, tl.mlp
        U._cp(f.router.weight, mo.router.classifier.weight)
        if mo.router.classifier.bias is not None:
            U._cp(f.router.bias, mo.router.classifier.bias)
        U._cp(f.sel_bias, mo.router.e_score_correction_bias)
        f.route_scale.fill_(float(mo.router.routed_scaling_factor))
        for e in range(n_real):
            ex = mo.experts[e]
            U._cp(f.w_gate[e], ex.gate_proj.weight); U._cp(f.w_up[e], ex.up_proj.weight)
            U._cp(f.w_down[e], ex.down_proj.weight)
        f.exp_ident[n_real:] = 1
    return u


@torch.no_grad()
def from_ouro(target):
    """Ouro (ByteDance/Ouro-*, modeling_ouro.py) unrolled over total_ut_steps: each pass runs the same Llama-style
    layers with sandwich norms (input_layernorm_2 on the attention output, post_attention_layernorm_2 on the MLP
    output: post_coef), then the loop-boundary norm h <- norm(h) (the shared final norm as the last layer's stream
    norm: loop_bnorm), and that h is an early-exit candidate (ee_tap). The readout picks per token among the
    candidates by the early-exit gate's pdf and early_exit_threshold (or early_exit_step): UEarlyExit; the final norm
    is the identity (every candidate is already normed)."""
    U = _U()
    tc, m = target.config, target.model
    N, S = tc.num_hidden_layers, int(getattr(tc, "total_ut_steps", 4) or 4)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    ff = m.layers[0].mlp.gate_proj.out_features
    c = U.UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=N * S, heads=tc.num_attention_heads, head_dim=hd,
                  shared_ff=ff, stream_norms=True, early_exit=True)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    u.norm.ident.fill_(1.0)
    inv, att = U._rope_of(m.rotary_emb)
    for e, ul in enumerate(u.layers):
        i = e % N
        tl = m.layers[i]
        U._norm_any(ul.norm_a, tl.input_layernorm); U._norm_any(ul.post_a, tl.input_layernorm_2)
        U._norm_any(ul.norm_f, tl.post_attention_layernorm); U._norm_any(ul.post_f, tl.post_attention_layernorm_2)
        ul.post_coef[0] = torch.tensor([0.0, 1.0]); ul.post_coef[2] = torch.tensor([0.0, 1.0])
        a = tl.self_attn
        U._attn_any(ul.attn, *U._qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=inv, att_scale=att,
                    window=U._sliding(tc, i))
        U._dense_ffn_from(ul.ffn, tl.mlp)
        U._set_act(ul.ffn, tc.hidden_act, True)
        if i == N - 1:                                        # the end of a pass: h <- norm(h), an exit candidate
            U._norm_any(ul.sn_out, m.norm)
            ul.sn_c[1] = 1
            ul.ee_tap.fill_(1.0)
    ee = u.early_exit
    U._cp(ee.gate.weight, m.early_exit_gate.weight); U._cp(ee.gate.bias, m.early_exit_gate.bias)
    step = getattr(target, "early_exit_step", None)
    thr = getattr(target, "early_exit_threshold", None)
    ee.step.fill_(float(step) if step is not None else -1.0)
    ee.threshold.fill_(float(thr) if thr is not None else float("inf"))
    return u


@torch.no_grad()
def from_hrm_text(target):
    """HRM-Text (transformers 5.18 models/hrm_text, HrmTextForCausalLM): two residual states z_H (the scaled
    embedding) and z_L (z_L_init), H_cycles x [L_cycles x (z_L <- L(z_L + z_H)), z_H <- H(z_H + z_L)], each stack its
    num_layers_per_stack pre-norm layers (weight-free RMSNorms, MHA with an elementwise sigmoid output gate, SwiGLU)
    and a weight-free final norm -- unrolled: each stack invocation's first layer reads z_H + z_L (UTwoState.enter),
    its last layer's stream norm is the stack's final norm (loop_bnorm) and writes the state back (put); the head
    reads z_H."""
    U = _U()
    tc, m = target.config, target.model
    N, Hc, Lc = tc.num_layers_per_stack, tc.H_cycles, tc.L_cycles
    hd = tc.head_dim
    ff = m.H_module.layers[0].mlp.gate_proj.out_features
    c = U.UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=N * Hc * (Lc + 1), heads=tc.num_attention_heads,
                  head_dim=hd, shared_ff=ff, stream_norms=True, two_state=True)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight * float(m.embedding_scale))
    U._cp(u.lm_head.weight, target.lm_head.weight)
    U._cp(u.hrm_z0, m.z_L_init)
    u.norm.ident.fill_(1.0)
    inv, att = U._rope_of(m.rotary_emb)
    seq = [("L", 2)] * Lc + [("H", 1)]
    e = 0
    for _ in range(Hc):
        for name, put in seq:
            st = m.L_module if name == "L" else m.H_module
            for j, tl in enumerate(st.layers):
                ul = u.layers[e]
                e += 1
                _rms_free(ul.norm_a, tl.input_layernorm.eps); _rms_free(ul.norm_f, tl.post_attention_layernorm.eps)
                a = tl.self_attn
                U._attn_any(ul.attn, *U._qkv(a), n_kv=tc.num_attention_heads, scale=a.scaling, inv_freq=inv,
                            att_scale=att)
                U._cp(ul.attn.g_proj.weight, a.gate_proj.weight)
                if a.gate_proj.bias is not None:
                    U._cp(ul.attn.g_proj.bias, a.gate_proj.bias)
                ul.attn.gate_coef.copy_(torch.tensor([0.0, 1.0]))
                U._dense_ffn_from(ul.ffn, tl.mlp)
                U._set_act(ul.ffn, tc.hidden_act, True)
                if j == 0:
                    ul.zstate.on.fill_(1.0)
                if j == N - 1:
                    _rms_free(ul.sn_out, st.final_norm.eps)
                    ul.sn_c[1] = 1
                    ul.zstate.put.fill_(float(put))
    return u


@torch.no_grad()
def from_nanbeige(target):
    """Nanbeige4.2 (Nanbeige/Nanbeige4.2-3B, modeling_nanbeige.py) with its extras off (no n-gram embeddings, hyper-
    connections, double-loop split, shared-KV or depth attention, qk-layernorm): a Llama body run num_loops times
    (len(loop_loss_weights) + 1 when given), unrolled; after every pass h <- norm(h) with the final norm's weight
    (loop_bnorm_cfg: skip_loop_final_norm false), else once at the end."""
    U = _U()
    tc, m = target.config, target.model
    for k in ("enable_hyper_connection", "enable_double_loop_split", "loop_share_kv", "enable_depth_attention",
              "qk_layernorm"):
        assert not getattr(tc, k, False), k
    assert getattr(m, "ngram_embeddings", None) is None and not len(getattr(m, "ngram_layer_fusion", {}) or {})
    lw = getattr(tc, "loop_loss_weights", None) or []
    S = len(lw) + 1 if lw else int(getattr(tc, "num_loops", 1) or 1)
    N = len(m.layers)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    ff = m.layers[0].mlp.gate_proj.out_features
    skip = bool(getattr(tc, "skip_loop_final_norm", False))
    c = U.UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=N * S, heads=tc.num_attention_heads, head_dim=hd,
                  shared_ff=ff, stream_norms=not skip)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    if skip:
        U._norm_any(u.norm, m.norm)
    else:
        u.norm.ident.fill_(1.0)
    for e, ul in enumerate(u.layers):
        tl = m.layers[e % N]
        a = tl.self_attn
        U._norm_any(ul.norm_a, tl.input_layernorm); U._norm_any(ul.norm_f, tl.post_attention_layernorm)
        U._attn_any(ul.attn, *U._qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling,
                    inv_freq=a.rotary_emb.inv_freq, att_scale=1.0)
        U._dense_ffn_from(ul.ffn, tl.mlp)
        U._set_act(ul.ffn, tc.hidden_act, True)
        if not skip and e % N == N - 1:
            U._norm_any(ul.sn_out, m.norm)
            ul.sn_c[1] = 1
    return u


def _split_trailing_rope_perm(hd, rot):
    """Host <- target dim order for a rotary on the TRAILING `rot` dims of a head in split-half layout (pairs
    (n + i, n + rot / 2 + i), n = hd - rot) onto the host pairs (i, i + hd / 2); the leading dims fill the rest."""
    h, r2, n = hd // 2, rot // 2, hd - rot
    lead = list(range(n))
    first = [n + i for i in range(r2)] + lead[: h - r2]
    second = [n + r2 + i for i in range(r2)] + lead[h - r2:]
    return torch.tensor(first + second)


@torch.no_grad()
def from_nanowhale(target):
    """nanowhale (HuggingFaceTB/nanowhale-100m, its own modeling_deepseek_v4.py): a small DeepSeek-V4 re-implementation
    whose hyper-connections read comb X, not comb^T X (hc_kind "comb"), with the DSv4 HyperHead collapse; attention: a
    low-rank q with a weighted latent norm, a weight-free per-head q norm, one K = V head kv_norm(wkv x) with
    nanochat-style (-theta) split-half RoPE on the trailing qk_rope_head_dim (v rotated, the output de-rotated), full
    causal attention (its SDPA path omits attn_sink and has no window), the grouped low-rank output folded into o_proj;
    a sqrt-softplus router selected with its bias, renormalised (+ 1e-20), x routed_scaling_factor, + a shared expert."""
    U = _U()
    tc, m = target.config, target.model
    H, hd, ql, d, rot = tc.num_attention_heads, tc.head_dim, tc.q_lora_rank, tc.hidden_size, tc.qk_rope_head_dim
    eps = float(tc.rms_norm_eps)
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd, attn_norm_feats=ql,
                  experts=tc.n_routed_experts, expert_ff=tc.moe_intermediate_size, top_k=tc.num_experts_per_tok,
                  shared_ff=tc.moe_intermediate_size, hc_mult=tc.hc_mult, hc_iters=tc.hc_sinkhorn_iters, hc_kind="comb")
    u = U.UniversalElephantine(c)
    W = lambda lin: lin.weight.detach().to(F64)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.norm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    u.hc_head_c.fill_(1.0)
    U._cp(u.hc_fn, m.hc_head_fn); U._cp(u.hc_base, m.hc_head_base); U._cp(u.hc_scale, m.hc_head_scale)
    u.hc_head_eps.copy_(torch.tensor([tc.hc_eps, eps], dtype=F64))
    perm = _split_trailing_rope_perm(hd, rot)
    ph = lambda w, n: w.reshape(n, hd, *w.shape[1:])[:, perm].reshape(n * hd, *w.shape[1:])
    inv = 1.0 / (float(tc.rope_theta) ** (torch.arange(0, rot, 2, dtype=F64) / rot))
    for ul, tl in zip(u.layers, m.layers):
        for hc, nm in ((ul.hc_a, "attn"), (ul.hc_f, "ffn")):
            h_ = hc._uhc
            U._cp(h_.fn, getattr(tl, f"hc_{nm}_fn")); U._cp(h_.base, getattr(tl, f"hc_{nm}_base"))
            U._cp(h_.scale, getattr(tl, f"hc_{nm}_scale"))
            h_.eps.copy_(torch.tensor([tl.hc_eps, tl.norm_eps], dtype=F64))
            h_.iters = tl.hc_sinkhorn_iters
        U._norm_from(ul.norm_a, tl.attn_norm); U._norm_from(ul.norm_f, tl.ffn_norm)
        a, at = ul.attn, tl.attn
        a.pn.weight.copy_(W(at.wq_a))
        a.n_avg.fill_(1.0 / ql)
        U._cp(a.n_w, at.q_norm.weight); a.n_eps.fill_(float(at.q_norm.eps))
        a.q_proj.weight[:, d:] = ph(W(at.wq_b), H)
        kv = W(at.wkv)[perm]
        a.k_proj.weight[:, :d] = kv.repeat(H, 1)
        a.v_proj.weight[:, :d] = kv.repeat(H, 1)
        a.q_coef.copy_(torch.tensor([0.0, 1.0])); a.k_coef.copy_(torch.tensor([0.0, 1.0]))
        a.q_nw.fill_(1.0)
        kn = at.kv_norm.weight.detach().to(F64)[perm]
        a.k_nw.copy_(kn.expand(H, hd)); a.v_nw.copy_(kn.expand(H, hd)); a.v_c.fill_(1.0)
        a.qk_eps.copy_(torch.tensor([eps, float(at.kv_norm.eps)], dtype=F64))
        a.v_eps.fill_(float(at.kv_norm.eps))
        a.inv_freq.zero_()
        a.inv_freq[: rot // 2] = -inv                       # nanochat-style rotation: by -theta
        a.scale.fill_(float(at.scaling))
        a.v_rope.fill_(1.0); a.o_derot.fill_(1.0)
        G = at.o_groups
        oa = W(at.wo_a).view(G, at.o_lora_rank, -1)
        wo = W(at.wo_b) @ torch.block_diag(*oa)
        a.o_proj.weight.copy_(wo.view(d, H, hd)[:, :, perm].reshape(d, H * hd))
        f, mo = ul.ffn, tl.ffn
        g = mo.gate
        U._cp(f.router.weight, g.weight)
        sf = g.scoring_func
        if sf == "sqrtsoftplus":
            f.score_sp.fill_(1.0)
        elif sf == "sigmoid":
            f.score_c.fill_(1.0)
        if g.bias is not None:
            U._cp(f.sel_bias, g.bias)
        if sf != "softmax":
            f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
        f.route_scale.fill_(float(g.route_scale))
        lim = float(mo.experts[0].swiglu_limit or 0.0)
        if lim > 0:                                          # the routed experts clamp; the shared expert does not
            f.clip_lim.fill_(lim)
            raise NotImplementedError("a routed-only SwiGLU limit: not wired")
        for e_, ex in enumerate(mo.experts):
            U._cp(f.w_gate[e_], ex.w1.weight); U._cp(f.w_up[e_], ex.w3.weight); U._cp(f.w_down[e_], ex.w2.weight)
        sh = mo.shared_expert
        f.s_gate.weight.copy_(W(sh.w1)); f.s_up.weight.copy_(W(sh.w3)); f.s_down.weight.copy_(W(sh.w2))
    return u


@torch.no_grad()
def from_xing(target):
    """Xing4.0 (XingChen-AGI/Xing4.0-*, modeling_xing4_0.py): a DeepSeek-V3 body (MLA, the first_k_dense_replace dense
    layers, a sigmoid MoE selected on scores + e_score_correction_bias, renormalised (+ 1e-20), x the routed scaling,
    + shared experts) in hc_mult Xing hyper-connection streams (UHCVariant "xing": no pre eps, clamped comb logits,
    exp(l - rowmax), full Sinkhorn passes, comb X) collapsed by their mean before the final norm."""
    U = _U()
    tc, m = target.config, target.model
    H, nope, rope, vd = tc.num_attention_heads, tc.qk_nope_head_dim, tc.qk_rope_head_dim, tc.v_head_dim
    kl, ql, d = tc.kv_lora_rank, tc.q_lora_rank or 0, tc.hidden_size
    hd = max(nope + rope, vd)
    hd += hd % 2
    moe0 = next((l.mlp for l in m.layers if hasattr(l.mlp, "experts")), None)
    sff = max([tc.intermediate_size] + ([moe0.shared_experts.gate_proj.out_features] if moe0 is not None else []))
    hc0 = m.layers[0].attn_hc
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd, attn_norm_feats=ql + kl,
                  experts=len(moe0.experts) if moe0 is not None else 1,
                  expert_ff=moe0.experts[0].gate_proj.out_features if moe0 is not None else 1,
                  top_k=moe0.gate.top_k if moe0 is not None else 1, shared_ff=sff,
                  hc_mult=hc0.hc_mult, hc_iters=hc0.hc_sinkhorn_iters, hc_kind="xing")
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.norm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    re_ = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        for dst, src in ((ul.hc_a, tl.attn_hc), (ul.hc_f, tl.ffn_hc)):
            U._cp(dst.fn, src.hc_fn); U._cp(dst.base, src.hc_base); U._cp(dst.scale, src.hc_scale)
            dst.eps.copy_(torch.tensor([src.hc_eps, src.input_norm.eps], dtype=F64))
            dst.lim.copy_(torch.tensor([float(src.mhc_h_res_clamp_min), float(src.mhc_h_res_clamp_max)], dtype=F64))
            dst.iters = src.hc_sinkhorn_iters
        U._norm_from(ul.norm_a, tl.input_layernorm); U._norm_from(ul.norm_f, tl.post_attention_layernorm)
        U._mla_from(ul.attn, tl.self_attn, d=d, H=H, nope=nope, rope=rope, vd=vd, kl=kl, ql=ql, inv_freq=re_.inv_freq,
                    att_s=float(re_.attention_scaling), interleave=bool(getattr(tc, "rope_interleave", False)),
                    scale=tl.self_attn.scaling)
        f, mo = ul.ffn, tl.mlp
        if not hasattr(mo, "experts"):
            k = mo.gate_proj.out_features
            f.s_gate.weight[:k] = mo.gate_proj.weight.detach().to(F64); f.s_up.weight[:k] = mo.up_proj.weight.detach().to(F64)
            f.s_down.weight[:, :k] = mo.down_proj.weight.detach().to(F64)
            continue
        g = mo.gate
        U._cp(f.router.weight, g.weight); U._cp(f.sel_bias, g.e_score_correction_bias)
        f.score_c.fill_(1.0)
        if g.norm_topk_prob:
            f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
        f.route_scale.fill_(float(g.routed_scaling_factor))
        G = int(getattr(g, "num_group", None) or getattr(g, "n_group", None) or 1)
        assert G == 1 or int(g.topk_group) == G, "group-limited routing: not wired"
        for e_, ex in enumerate(mo.experts):
            U._cp(f.w_gate[e_], ex.gate_proj.weight); U._cp(f.w_up[e_], ex.up_proj.weight); U._cp(f.w_down[e_], ex.down_proj.weight)
        sh = mo.shared_experts
        k = sh.gate_proj.out_features
        f.s_gate.weight[:k] = sh.gate_proj.weight.detach().to(F64); f.s_up.weight[:k] = sh.up_proj.weight.detach().to(F64)
        f.s_down.weight[:, :k] = sh.down_proj.weight.detach().to(F64)
    return u


@torch.no_grad()
def from_hy_v4(target):
    """HY-V4 (transformers 5.18 models/hy_v4, HYV4ForCausalLM; tencent/Hy4-preview): hc_mult hyper-connection streams
    with pre / post maps only (UHCVariant "nocomb": post = hc_magnitude sigmoid(.) + eps, the streams carried unmixed)
    and the DeepSeek-V4 HyperHead collapse before the final norm; MLA with non-interleaved RoPE, an elementwise sigmoid
    output gate per head and v channel, learned per-head sinks (the null logit); the DSA indexer run dense (exact while
    index_topk covers the context, the rule decision for DSA); dense / sparse FFNs by mlp_layer_types, the sparse ones a
    sigmoid router selected on scores + e_score_correction_bias within the top topk_group groups (-inf masked),
    renormalised (+ 1e-20), x routed_scaling_factor, routed experts clamped at swiglu_limit (UFFN.clip_shared 0), and
    an unclamped shared expert."""
    U = _U()
    tc = target.config
    tc = getattr(tc, "text_config", tc)
    m = target.model
    H, nope, rope, vd = tc.num_attention_heads, tc.qk_nope_head_dim, tc.qk_rope_head_dim, tc.v_head_dim
    kl, ql, d = tc.kv_lora_rank, tc.q_lora_rank, tc.hidden_size
    hd = max(nope + rope, vd)
    hd += hd % 2
    moe0 = next((l.mlp for l in m.layers if hasattr(l.mlp, "experts")), None)
    dense0 = next((l.mlp for l in m.layers if not hasattr(l.mlp, "experts")), None)
    sff = max([moe0.shared_experts.gate_proj.out_features if moe0 is not None else 0,
               dense0.gate_proj.out_features if dense0 is not None else 0])
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=len(m.layers), heads=H, head_dim=hd, attn_norm_feats=ql + kl,
                  experts=moe0.experts.num_experts if moe0 is not None else 1,
                  expert_ff=moe0.experts.intermediate_dim if moe0 is not None else 1,
                  top_k=moe0.gate.top_k if moe0 is not None else 1, shared_ff=sff, hc_mult=tc.hc_mult, hc_kind="nocomb")
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.norm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    hh = m.hc_head
    u.hc_head_c.fill_(1.0)
    U._cp(u.hc_fn, hh.hc_fn); U._cp(u.hc_base, hh.hc_base); U._cp(u.hc_scale, hh.hc_scale)
    u.hc_head_eps.copy_(torch.tensor([hh.eps, hh.input_norm.eps], dtype=F64))
    re_ = m.rotary_emb
    W = lambda lin: lin.weight.detach().to(F64)
    for ul, tl in zip(u.layers, m.layers):
        for dst, src in ((ul.hc_a, tl.attn_hc), (ul.hc_f, tl.ffn_hc)):
            U._cp(dst.fn, src.fn); U._cp(dst.base, src.base); U._cp(dst.scale, src.scale)
            dst.post_c.fill_(float(src.hc_post_magnitude))
            dst.eps.copy_(torch.tensor([src.hc_eps, src.input_norm.eps], dtype=F64))
        U._norm_from(ul.norm_a, tl.input_layernorm); U._norm_from(ul.norm_f, tl.post_attention_layernorm)
        a, at = ul.attn, tl.self_attn
        U._mla_from(a, at, d=d, H=H, nope=nope, rope=rope, vd=vd, kl=kl, ql=ql, inv_freq=re_.inv_freq,
                    att_s=float(re_.attention_scaling), interleave=False, scale=at.scaling)
        g = W(at.gate_proj)
        for h in range(H):
            a.g_proj.weight[h * hd: h * hd + vd, :d] = g[h * vd:(h + 1) * vd]
        a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
        U._cp(a.null_logit, at.sinks)
        f, mo = ul.ffn, tl.mlp
        sh = mo.shared_experts if hasattr(mo, "experts") else mo
        k = sh.gate_proj.out_features
        f.s_gate.weight[:k] = W(sh.gate_proj); f.s_up.weight[:k] = W(sh.up_proj); f.s_down.weight[:, :k] = W(sh.down_proj)
        if not hasattr(mo, "experts"):
            continue
        rt = mo.gate
        U._cp(f.router.weight, rt.weight); U._cp(f.sel_bias, rt.e_score_correction_bias)
        f.score_c.fill_(1.0)
        G, kg = int(rt.num_group), int(rt.topk_group)
        if G > 1 and kg < G:
            f.grp.copy_(torch.tensor([float(G), float(kg), 0.0, float("-inf")], dtype=F64))
        if rt.norm_topk_prob:
            f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
        f.route_scale.fill_(float(rt.routed_scaling_factor))
        gt, up = mo.experts.gate_up_proj.detach().to(F64).chunk(2, dim=1)
        f.w_gate.copy_(gt); f.w_up.copy_(up); U._cp(f.w_down, mo.experts.down_proj)
        lim = getattr(mo.experts, "swiglu_limit", None)
        if lim is not None:
            f.c_clip.fill_(1.0); f.clip_lim.fill_(float(lim)); f.clip_shared.fill_(0.0)
    return u


@torch.no_grad()
def from_sdar_selfcond(target, block_size=None):
    """SDAR with self-conditioning (QwenDLM-4B-ORL, modeling_sdar.py SDARSelfConditioning, read as text): the SDAR body
    (from_sdar: Qwen3 under the caller's block mask) plus the self-conditioning input path, run as forward(ids,
    selfcond=s) with s the previous pass's soft embeddings (selfcond_signal): x <- rms(x + o SwiGLU(p rms(s) w)) * q,
    p / o / q = self_conditioning_prenorm_scale / reparam_out_scale / post_norm_scale; without s it is plain SDAR."""
    from types import SimpleNamespace as NS
    U = _U()
    u = U.from_sdar(target, block_size=block_size)
    tc, sc = target.config, target.self_conditioning
    u.cfg.selfcond_ff = sc.gate_proj.out_features
    u.selfcond = USelfCond(NS(d=u.cfg.d, selfcond_ff=u.cfg.selfcond_ff))
    s = u.selfcond
    U._norm_from(s.pre, sc.pre_norm)
    U._cp(s.gate.weight, sc.gate_proj.weight); U._cp(s.up.weight, sc.up_proj.weight); U._cp(s.down.weight, sc.down_proj.weight)
    s.scales.copy_(torch.tensor([float(getattr(tc, "self_conditioning_prenorm_scale", 1.0)),
                                 float(getattr(tc, "self_conditioning_reparam_out_scale", 1.0)),
                                 float(getattr(tc, "self_conditioning_post_norm_scale", 1.0))], dtype=F64))
    s.post_c.fill_(1.0)                                   # SDARSelfConditioning.forward always ends in F.rms_norm
    s.post_eps.fill_(float(sc.post_norm_eps))
    return u


@torch.no_grad()
def selfcond_signal(u, ids, mask=None, **kw):
    """RUNTIME, not architecture: the self-conditioning signal of a first pass (modeling_sdar.py
    _build_self_conditioning_signal): soft embeddings softmax(logits) E at the positions in `mask` (all if None), 0
    elsewhere."""
    logits = u(ids, **kw)
    s = logits.softmax(-1) @ u.embed.weight
    return s if mask is None else s * mask[..., None].to(s.dtype)


def _l2_unorm(n, width, eps):
    """A weight-free L2 norm x / sqrt(|x|^2 + eps) as a UNorm: RMS with weight 1 / sqrt(width), eps / width."""
    with torch.no_grad():
        n.weight.fill_(width ** -0.5)
        n.eps.fill_(float(eps) / width)


@torch.no_grad()
def from_dory(target):
    """Nemotron-Labs Dory (transformers PR 50099 models/dory, modeling_dory.py; read as text): nGPT on the unit sphere,
    each decoder layer as two host layers (attention, MLP). Each sub-block: the stream itself L2-normalised (ULayer
    sn_in: L2 = RMS / sqrt(d)), the sub-block's output L2-normalised (post_coef), and the nGPT update
    h + r (f - h) = (1 - r) h + r f with per-channel r = |lambda scale| (URGain); attention: unit-norm q / k (the per-head
    norm AFTER the rotary -- the same, the rotary is orthogonal -- with q's per-channel sqk^2 as its weight), GQA,
    sliding / full layers with their own rope theta, the sigmoid output gate with s_gate^2 folded into it, then the
    gated output L2-normalised (o_late) before o_proj; MLP: suv-scaled gate / up, SwiGLU, the hidden L2-normalised
    (s_hid) before down. Input layers, n_recurrent_loops passes of the recurrent layers (unrolled) each starting from
    L2(h) + L2(h_input) (ULoopInject), output layers, the final L2 norm, the lm_head with its per-token logit scale."""
    U = _U()
    tc, m = target.config, target.model
    H, nkv, hd, d = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.hidden_size
    n_in, n_rec, n_out, loops = tc.n_input_layers, tc.n_recurrent_layers, tc.n_output_layers, tc.n_recurrent_loops
    order = list(range(n_in)) + [n_in + j for _ in range(loops) for j in range(n_rec)] + \
        list(range(n_in + n_rec, n_in + n_rec + n_out))
    ff = m.layers[0].mlp.gate_proj.out_features
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=2 * len(order), heads=H, head_dim=hd, shared_ff=ff, stream_norms=True,
                  res_gain=True, loop_inject=True, o_late_norm=True, ffn_hidden_norm=True)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    eps = float(m.norm.eps)
    _l2_unorm(u.norm, d, eps)
    lh = target.lm_head
    U._cp(u.lm_head.weight, lh.weight.detach().to(F64) * (lh.logit_scale.detach().to(F64) * lh.logit_scale_factor)[:, None])
    W = lambda lin: lin.weight.detach().to(F64)
    for e, i in enumerate(order):
        tl = m.layers[i]
        ua, ub = u.layers[2 * e], u.layers[2 * e + 1]
        at = tl.self_attn
        # --- attention sub-block (host layer A)
        _l2_unorm(ua.sn_in, d, tl.input_layernorm.eps); ua.sn_c[0] = 1
        ua.norm_a.ident.fill_(1.0)
        a = ua.attn
        a.q_proj.weight.copy_(W(at.q_proj))
        a.k_proj.weight.copy_(U._tile_kv(W(at.k_proj), nkv, H, hd))
        a.v_proj.weight.copy_(U._tile_kv(W(at.v_proj), nkv, H, hd))
        U._cp(a.o_proj.weight, at.o_proj.weight)
        a.qk_post.fill_(1.0)
        sq = (at.sqk.detach().to(F64) * at.sqk_scale).view(H, hd).square()
        a.q_nw.copy_(sq * hd ** -0.5); a.k_nw.fill_(hd ** -0.5)
        a.q_coef.copy_(torch.tensor([0.0, 1.0])); a.k_coef.copy_(torch.tensor([0.0, 1.0]))
        a.qk_eps.fill_(float(at.q_norm.eps) / hd)
        lt = tl.layer_type
        a.inv_freq.copy_(getattr(m.rotary_emb, f"{lt}_inv_freq").detach().to(F64))
        a.scale.fill_(float(at.scaling))
        if at.sliding_window is not None:
            a.window.fill_(float(at.sliding_window - 1))
        sg = (at.s_gate.detach().to(F64) * at.s_gate_scale).square()
        a.g_proj.weight.copy_(W(at.gate_proj) * sg[:, None])
        a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
        _l2_unorm(a.o_late, H * hd, at.attn_sub_norm.eps)
        _l2_unorm(ua.post_a, d, tl.post_attention_layernorm.eps); ua.post_coef[0] = torch.tensor([0.0, 1.0])
        r1 = (tl.lambda_1.detach().to(F64) * tl.lambda_scale).abs()
        ua.rgain.gx[0] = 1 - r1; ua.rgain.go[0] = r1
        _l2_unorm(ua.ffn.s_hid, ff, 1e-6)
        # --- MLP sub-block (host layer B)
        _l2_unorm(ub.sn_in, d, tl.pre_feedforward_layernorm.eps); ub.sn_c[0] = 1
        ub.norm_f.ident.fill_(1.0)
        _l2_unorm(ub.attn.o_late, H * hd, 1e-6)
        mp, f = tl.mlp, ub.ffn
        f.s_gate.weight.copy_(W(mp.gate_proj) * (mp.suv_gate.detach().to(F64) * mp.suv_scale)[:, None])
        f.s_up.weight.copy_(W(mp.up_proj) * (mp.suv_up.detach().to(F64) * mp.suv_scale)[:, None])
        U._cp(f.s_down.weight, mp.down_proj.weight)
        U._set_act(f, tc.hidden_act, True)
        _l2_unorm(f.s_hid, ff, mp.ffn_sub_norm.eps)
        _l2_unorm(ub.post_f, d, tl.post_feedforward_layernorm.eps); ub.post_coef[2] = torch.tensor([0.0, 1.0])
        r2 = (tl.lambda_2.detach().to(F64) * tl.lambda_scale).abs()
        ub.rgain.gx[1] = 1 - r2; ub.rgain.go[1] = r2
        for ul in (ua, ub):
            _l2_unorm(ul.loopinj.norm, d, eps)
        if e == n_in - 1:
            ub.loopinj.snap.fill_(1.0)                       # the input section's output, normed: the injection
        if n_in <= e < n_in + loops * n_rec and (e - n_in) % n_rec == 0:
            ua.loopinj.inject.fill_(1.0)                     # every pass starts from L2(h) + L2(h_input)
    return u


@torch.no_grad()
def gated_norm_from(dst, norm, gate=None):
    """A gated pre-norm -> UNorm's low-rank gate branch (gate_c 1: rms(x) w * sigmoid(g_down(silu(g_up(rms(x) w))))):
    A.X-K2's AXK2GatedRMSNorm (`norm` = the module: .norm, .mlp.fc1 / .fc2; transformers 5.18 models/axk2) or Grug-MoE's
    RMSNorm + GrugMoeGatedNorm pair (`norm`, `gate` = down_proj / up_proj; marin skyrl_train_models_grug_moe.py
    l. 382-410). The gate rank r <= the host's rank: the rest of the rows stay zero."""
    U = _U()
    if gate is None:
        gate, norm = norm.mlp, norm.norm
        up, down = gate.fc1, gate.fc2
    else:
        up, down = gate.down_proj, gate.up_proj
    U._norm_any(dst, norm)
    r = up.weight.shape[0]
    dst.g_up.weight.zero_(); dst.g_down.weight.zero_()
    dst.g_up.weight[:r] = up.weight.detach().to(F64)
    dst.g_down.weight[:, :r] = down.weight.detach().to(F64)
    dst.gate_c.fill_(1.0)


@torch.no_grad()
def from_doge(target):
    """Doge (transformers DogeForCausalLM, dense MLP): a Llama-shaped body with per-head q / k RMSNorms and the
    per-channel residual gains h = g1 * r + attn(n r), g2 * h + mlp(n h) (input_residual / post_attention_residual:
    URGain gx). Its dynamic-mask attention (dt_proj, A: a key bias exp(A softplus(dt)), keep_window_size top-k) is an
    attention feature not carried here: this loader requires A = 0 (the bias a per-row constant, softmax-neutral) and
    runs while T <= keep_window_size."""
    from types import SimpleNamespace as NS
    U = _U()
    tc, m = target.config, target.model
    for tl in m.layers:
        assert not bool(tl.self_attn.A.detach().abs().max() > 0), "Doge's dynamic mask (A != 0) is not carried"
    assert not getattr(tc, "is_moe", False), "Doge's CDMoE is not wired"
    u = U.from_llama(target)
    for ul, tl in zip(u.layers, m.layers):
        ul.rgain = URGain(NS(d=tc.hidden_size))
        U._cp(ul.rgain.gx[0], tl.input_residual); U._cp(ul.rgain.gx[1], tl.post_attention_residual)
    return u


@torch.no_grad()
def zaya_residual_scaling_into(u, scalings):
    """ZAYA1's ResidualScaling merges (modeling_zaya1_vl.py l. 1512-1531, applied at each sub-layer's start and once
    before the final norm) onto the host's sub-blocks: `scalings` = the 2 L + 1 modules in order (attention, MLP per
    layer, then the final one; scalings[0] has no residual part). The first one's affine (e + b_h) s_h folds into the
    token table; merge k > 0 -- s_r (x + b_r) + s_h (f + b_h), f the previous sub-block's output -- is that sub-block's
    URGain (host layer (k - 1) // 2, slot (k - 1) % 2)."""
    from types import SimpleNamespace as NS
    W = lambda p: p.detach().to(F64)
    s0 = scalings[0]
    u.embed.weight.copy_((u.embed.weight + W(s0.hidden_states_bias)) * W(s0.hidden_states_scale))
    for k, sc in enumerate(scalings[1:], start=1):
        ul, i = u.layers[(k - 1) // 2], (k - 1) % 2
        if ul.rgain is None:
            ul.rgain = URGain(NS(d=u.cfg.d))
        sr, br, sh, bh = W(sc.residual_scale), W(sc.residual_bias), W(sc.hidden_states_scale), W(sc.hidden_states_bias)
        ul.rgain.gx[i] = sr; ul.rgain.go[i] = sh; ul.rgain.gb[i] = sr * br + sh * bh


@torch.no_grad()
def cfrd_binding_into(ul, blk, theta):
    """CFRD's BindingBlock (alice-noa-chan/haru*, model.py) -> one host layer: RMSNorm -> full causal GQA attention with
    interleaved (even / odd) RoPE -> x + c sigmoid(attn_gate(u)) attn(u), then the same with the SwiGLU (URGain's gate,
    go = residual_scale)."""
    from types import SimpleNamespace as NS
    U = _U()
    at = blk.attention
    H, nkv, hd = at.n_head, at.n_kv_head, at.head_dim
    U._norm_from(ul.norm_a, blk.attn_norm); U._norm_from(ul.norm_f, blk.ffn_norm)
    perm = U._interleaved_rope_perm(hd, hd)
    Wp = lambda w: w.view(-1, hd, w.shape[-1])[:, perm].reshape(w.shape)
    a = ul.attn
    a.q_proj.weight.copy_(Wp(at.q_proj.weight.detach().to(F64)))
    a.k_proj.weight.copy_(Wp(U._tile_kv(at.k_proj.weight.detach().to(F64), nkv, H, hd)))
    a.v_proj.weight.copy_(U._tile_kv(at.v_proj.weight.detach().to(F64), nkv, H, hd))
    U._cp(a.o_proj.weight, at.o_proj.weight)
    a.inv_freq.copy_(1.0 / (float(theta) ** (torch.arange(0, hd, 2, dtype=F64) / hd)))
    a.scale.fill_(hd ** -0.5)
    f = ul.ffn
    U._cp(f.s_gate.weight, blk.ffn.w1.weight); U._cp(f.s_up.weight, blk.ffn.w3.weight); U._cp(f.s_down.weight, blk.ffn.w2.weight)
    if ul.rgain is None:
        ul.rgain = URGain(NS(d=ul.norm_a.weight.numel()))
    g = ul.rgain
    for i, gate in enumerate((blk.attn_gate, blk.ffn_gate)):
        g.gw[i] = gate.weight.detach().to(F64)[0]; g.gc[i] = gate.bias.detach().to(F64)[0]
        g.go[i] = float(blk.residual_scale); g.g_on[i] = 1


@torch.no_grad()
def muse_residual_gate_into(g, i, gate, temperature, eps=1e-3):
    """Muse-Spark's MuseSparkResidualGate (transformers main modular_muse_spark.py l. 451-462: alpha r + beta h,
    beta = sigmoid(g / T), alpha = sqrt(max(sigmoid(-g / T) (1 + beta), eps))) -> URGain slot i (gx = alpha, go = beta)."""
    w = gate.detach().to(F64) / float(temperature)
    beta = torch.sigmoid(w)
    g.gx[i] = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=eps))
    g.go[i] = beta


@torch.no_grad()
def _canon_from(dst, src, tile=None):
    """A dhara CanonLayer (conv weight (C, 1, K), optional bias, residual / activation flags) -> UCanon."""
    w = src.conv.weight.detach().to(F64)[:, 0]
    b = src.conv.bias.detach().to(F64) if src.conv.bias is not None else torch.zeros(w.shape[0], dtype=F64)
    if tile is not None:
        w, b = tile(w), tile(b)
    dst.w.copy_(w); dst.b.copy_(b)
    dst.res.fill_(1.0 if src.use_residual else 0.0); dst.act.fill_(1.0 if src.use_activation else 0.0)
    dst.c.fill_(1.0)


@torch.no_grad()
def from_dhara(target):
    """Dhara-AR (codelion/dhara-250m, modeling_dhara_ar.py, read as text): a Llama-shaped GQA body with qk RMSNorms after
    the rotary and Canon layers (short causal depthwise convs with a residual) at canon_set positions: A after the input
    norm, B on q / k / v after the projections, C after the post-attention norm, D on the SwiGLU hidden; a logit
    softcap; tied embeddings. The AR forward (its diffusion trimode_bias path is the same weights under another mask)."""
    U = _U()
    tc, m = target.config, target.model
    H, nkv, d = tc.num_attention_heads, tc.num_key_value_heads, tc.hidden_size
    hd = d // H
    sites = str(tc.canon_set or "")
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd,
                  shared_ff=tc.intermediate_size, canon_k=int(tc.canon_kernel) if sites else 0, canon_sites=sites)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.norm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    if getattr(tc, "use_logit_softcap", False) and tc.logit_softcap > 0:
        u.final_cap.fill_(float(tc.logit_softcap))
    tile = lambda w: U._tile_kv(w, nkv, H, hd)
    for ul, tl in zip(u.layers, m.layers):
        at = tl.self_attn
        U._norm_from(ul.norm_a, tl.input_layernorm); U._norm_from(ul.norm_f, tl.post_attention_layernorm)
        assert float(getattr(at.rotary_emb, "_yarn_attn_factor", 1.0)) == 1.0, "YaRN's cos / sin factor: not wired"
        U._attn_any(ul.attn, *U._qkv(at), n_kv=nkv, scale=hd ** -0.5, inv_freq=at.rotary_emb.inv_freq,
                    q_norm=at.q_norm, k_norm=at.k_norm, qk_post=at.q_norm is not None)
        U._dense_ffn_from(ul.ffn, tl.mlp)
        if tl.canon_a is not None:
            _canon_from(ul.norm_a.canon, tl.canon_a)
        if tl.canon_c is not None:
            _canon_from(ul.norm_f.canon, tl.canon_c)
        if at.canon_b_q is not None:
            cb = ul.attn.canon_b
            _canon_from(cb.q, at.canon_b_q); _canon_from(cb.k, at.canon_b_k, tile); _canon_from(cb.v, at.canon_b_v, tile)
        if tl.mlp.canon_d is not None:
            _canon_from(ul.ffn.s_hid, tl.mlp.canon_d)
    return u


@torch.no_grad()
def from_crowd(target):
    """Crowd-v1 (Vxtzq/Crowd-v1, modeling_crowd.py, read as text): the token embedding plus a hashed bigram embedding
    (UBigramHash), LayerNorm (affine) pre-norm blocks with GQA rotary attention and a SwiGLU, the final LayerNorm, a
    tied head."""
    U = _U()
    tc = target.config
    H, nkv, hd, d = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.hidden_size
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd,
                  shared_ff=tc.intermediate_size, ngram_in_buckets=target.engram.table.num_embeddings)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, target.wte.weight)
    U._cp(u.ngram_in.table, target.engram.table.weight)
    U._norm_any(u.norm, target.ln, "ln")
    U._cp(u.lm_head.weight, target.head.weight)
    inv = 1.0 / (float(tc.rope_theta) ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    W = lambda lin: lin.weight.detach().to(F64)
    for ul, b in zip(u.layers, target.blocks):
        at = b.attn
        U._norm_any(ul.norm_a, b.ln1, "ln"); U._norm_any(ul.norm_f, b.ln2, "ln")
        U._attn_any(ul.attn, W(at.q), W(at.k), W(at.v), W(at.o), n_kv=nkv, scale=hd ** -0.5, inv_freq=inv)
        f = ul.ffn
        U._cp(f.s_gate.weight, b.mlp.w1.weight); U._cp(f.s_up.weight, b.mlp.w2.weight); U._cp(f.s_down.weight, b.mlp.w3.weight)
    return u


@torch.no_grad()
def from_llama_moe(target):
    """LLaMA-MoE v1 (llama-moe/LLaMA-MoE-v1-*, modeling_llama_moe_hf.py, read as text): a Llama body whose MLPs are
    split into experts (LinearGLUExperts, size_experts per layer: smaller experts zero-padded), routed by an MLP gate
    W2 tanh(W1 x) (URouterMLP), the top-k logits softmaxed (softmax + renorm), each expert's output times its score x
    score_scale_factor (UniversalCalculator)."""
    U = _U()
    tc, m = target.config, target.model
    H, nkv, d = tc.num_attention_heads, tc.num_key_value_heads, tc.hidden_size
    hd = d // H
    mo0 = m.layers[0].mlp
    E, k = mo0.gate.num_experts, mo0.gate.num_selects
    ef = max(int(w.shape[0]) for l in m.layers for w in l.mlp.calculator.experts.weight_gate)
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd, experts=E, expert_ff=ef,
                  top_k=k, router_mlp=mo0.gate.gate_network_type.lower() == "mlp")
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.norm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    inv = getattr(getattr(m.layers[0].self_attn, "rotary_emb", None), "inv_freq", None)
    if inv is None:
        inv = m.rotary_emb.inv_freq
    for ul, tl in zip(u.layers, m.layers):
        at = tl.self_attn
        U._norm_from(ul.norm_a, tl.input_layernorm); U._norm_from(ul.norm_f, tl.post_attention_layernorm)
        U._attn_any(ul.attn, *U._qkv(at), n_kv=nkv, scale=hd ** -0.5, inv_freq=inv)
        mo, f = tl.mlp, ul.ffn
        g = mo.gate
        assert g.use_softmax and mo.calculator.multiply_gate_scores
        gn = g.gate_network
        if f.router_mlp is not None:
            U._cp(f.router_mlp.w1.weight, gn[0].weight); U._cp(f.router_mlp.w2.weight, gn[2].weight)
        else:
            U._cp(f.router.weight, gn.weight)
        f.renorm.fill_(1.0)
        f.route_scale.fill_(float(mo.calculator.score_scale_factor))
        ex = mo.calculator.experts
        for e in range(E):
            n = ex.weight_gate[e].shape[0]
            f.w_gate[e, :n] = ex.weight_gate[e].detach().to(F64); f.w_up[e, :n] = ex.weight_up[e].detach().to(F64)
            f.w_down[e, :, :n] = ex.weight_down[e].detach().to(F64)
        f.s_down.weight.zero_()                                # no shared path
    return u


@torch.no_grad()
def from_arctic(target):
    """Snowflake Arctic (modeling_arctic.py, read as text): a Mixtral-attention body whose MoE layers run, with
    parallel_attn_mlp_res, a dense residual SwiGLU (width hidden_size) on the post-attention stream beside the routed
    experts on the normed LAYER input: out = h + mlp_res(n_r h) + moe(n_2 x), h = x + attn(n_1 x). Host: the split FFN
    (ffn_split, no branch post-norms) with ULayer.moe_par -- the shared path reads norm_f(h), the experts and router
    norm_f2 / norm_r of the layer input; softmax top-k with renorm."""
    U = _U()
    tc, m = target.config, target.model
    H, nkv, d = tc.num_attention_heads, tc.num_key_value_heads, tc.hidden_size
    hd = d // H
    moe = [l for l in m.layers if l.block_sparse_moe.is_moe_layer]
    E, k = tc.num_local_experts, tc.num_experts_per_tok
    sff = max([l.residual_mlp.w1.out_features for l in moe if l.parallel_attn_mlp_res] +
              [l.block_sparse_moe.mlp.w1.out_features for l in m.layers if not l.block_sparse_moe.is_moe_layer] + [1])
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd, experts=E,
                  expert_ff=tc.intermediate_size, top_k=k, shared_ff=sff)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.norm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    W = lambda lin: lin.weight.detach().to(F64)
    for ul, tl in zip(u.layers, m.layers):
        at = tl.self_attn
        inv = getattr(getattr(at, "rotary_emb", None), "inv_freq", None)
        inv = m.rotary_emb.inv_freq if inv is None else inv
        U._norm_from(ul.norm_a, tl.input_layernorm)
        U._attn_any(ul.attn, *U._qkv(at), n_kv=nkv, scale=hd ** -0.5, inv_freq=inv,
                    window=getattr(tc, "sliding_window", None))
        f, mo = ul.ffn, tl.block_sparse_moe
        put = lambda mlp: (f.s_gate.weight[:mlp.w1.out_features].copy_(W(mlp.w1)),
                           f.s_up.weight[:mlp.w3.out_features].copy_(W(mlp.w3)),
                           f.s_down.weight[:, :mlp.w2.in_features].copy_(W(mlp.w2)))
        if not mo.is_moe_layer:
            U._norm_from(ul.norm_f, tl.post_attention_layernorm)
            put(mo.mlp)
            continue
        U._cp(f.router.weight, mo.gate.weight)
        f.renorm.fill_(1.0 if k > 1 else 0.0)
        for e, ex in enumerate(mo.experts):
            U._cp(f.w_gate[e], ex.w1.weight); U._cp(f.w_up[e], ex.w3.weight); U._cp(f.w_down[e], ex.w2.weight)
        if tl.parallel_attn_mlp_res:
            ul.ffn_split.fill_(1.0); ul.moe_par.fill_(1.0)
            ul.post_f1.ident.fill_(1.0); ul.post_f2.ident.fill_(1.0)
            U._norm_from(ul.norm_f, tl.residual_layernorm)
            U._norm_from(ul.norm_f2, tl.post_attention_layernorm); U._norm_from(ul.norm_r, tl.post_attention_layernorm)
            put(tl.residual_mlp)
        else:
            U._norm_from(ul.norm_f, tl.post_attention_layernorm)
            f.s_down.weight.zero_()
    return u


def capacity_keep(topi, E):
    """Expert-capacity dropping (HunYuan-A13B modeling_hunyuan.py topkgating l. 74-140): capacity
    C = max(k, k S // E) over the S tokens of the whole forward (batch-major); assignments are queued per expert in
    (choice rank, token) order -- every token's first choice before any second choice -- and those at queue position
    >= C are dropped (their weight 0, the rest not renormalised). topi (B, T, k) -> keep (B, T, k) bool."""
    B, T, k = topi.shape
    S = B * T
    C = max(k, k * S // E)
    order = topi.reshape(S, k).transpose(0, 1).reshape(-1)               # rank-major
    oh = F.one_hot(order, E).to(torch.int64)
    prio = ((torch.cumsum(oh, 0) * oh) - 1).max(-1).values                # queue position in its expert
    return (prio < C).view(k, S).transpose(0, 1).reshape(B, T, k)


@torch.no_grad()
def from_hunyuan_capacity(target):
    """HunYuan-A13B (tencent/Hunyuan-A13B-Instruct, remote modeling_hunyuan.py, read as text): the stock HunYuan-MoE
    body (from_hunyuan: post-rotary qk-norms, softmax top-k renormalised, a shared MLP) whose remote router
    (topkgating) drops the assignments over each expert's per-forward capacity (UFFN.cap_drop, capacity_keep)."""
    u = _U().from_hunyuan(target)
    for ul in u.layers:
        if ul.ffn.w_gate.shape[0] > 1:
            ul.ffn.cap_drop.fill_(1.0)
    return u


def block_route_mask(sel, block, cap):
    """Block routing (LLaDA2.2-mini modeling_llada2_moe.py LLaDA2MoeGate.block_routing l. 244-275): the tokens of the
    whole forward (batch-major) in blocks of `block`; a block's score per expert is the max of its tokens' selection
    scores; only the block's top `cap` experts stay selectable (-inf for the rest)."""
    B, T, E = sel.shape
    flat = sel.reshape(-1, E)
    n = flat.shape[0]
    assert n % block == 0, (n, block)
    bs = flat.view(n // block, block, E).max(1).values
    allowed = torch.zeros(n // block, E, dtype=torch.bool).scatter_(1, bs.topk(cap, -1).indices, True)
    allowed = allowed[:, None].expand(-1, block, -1).reshape(n, E)
    return flat.masked_fill(~allowed, float("-inf")).view(B, T, E)


@torch.no_grad()
def llada2_block_moe_into(f, moe):
    """LLaDA2.2's LLaDA2MoeSparseMoeBlock (block-routed sigmoid gate with an expert bias, renormalised (+ 1e-20),
    x routed_scaling_factor, + shared experts) -> UFFN (blk_route)."""
    U = _U()
    g = moe.gate
    U._cp(f.router.weight, g.weight); U._cp(f.sel_bias, g.expert_bias)
    f.score_c.fill_(1.0)
    if g.top_k > 1:
        f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
    f.route_scale.fill_(float(g.routed_scaling_factor))
    f.blk_route.copy_(torch.tensor([float(g.block_size), float(g.expert_capacity)], dtype=F64))
    for e, ex in enumerate(moe.experts):
        U._cp(f.w_gate[e], ex.gate_proj.weight); U._cp(f.w_up[e], ex.up_proj.weight); U._cp(f.w_down[e], ex.down_proj.weight)
    sh = getattr(moe, "shared_experts", None)
    if sh is not None:
        U._dense_ffn_from(f, sh)
    else:
        f.s_down.weight.zero_()


@torch.no_grad()
def from_llama4(target):
    """Llama-4 text (transformers Llama4ForCausalLM / the text model of Llama4ForConditionalGeneration): GQA with the
    interleaved complex rotary on the rope layers (no_rope_layers 1) and a weight-free L2 (RMS) qk-norm after it, the
    query temperature 1 + attn_scale log1p(floor((t + 1) / floor_scale)) on the NoPE layers (UAttention.qtemp with
    qtemp_off 1), MoE layers whose router picks the top-k logits and whose experts READ the input scaled by
    sigmoid(logit) (UFFN.in_gate) beside a shared expert, dense layers between; the chunked layers (layer_types
    chunked_attention) attend within their attention_chunk_size block (UAttention.chunk, the mask_ext branch of
    eleph-gaps-attn), so the loader is exact at any length."""
    U = _U()
    tc = target.config
    tc = getattr(tc, "text_config", tc)
    m = getattr(target, "language_model", None) or target.model
    m = getattr(m, "language_model", m)
    H, nkv, d = tc.num_attention_heads, tc.num_key_value_heads, tc.hidden_size
    hd = getattr(tc, "head_dim", None) or d // H
    moe0 = next((l.feed_forward for l in m.layers if l.is_moe_layer), None)
    E = moe0.num_experts if moe0 is not None else 1
    ef = moe0.experts.expert_dim if moe0 is not None else 1
    sff = max([moe0.shared_expert.gate_proj.out_features if moe0 is not None else 0] +
              [l.feed_forward.gate_proj.out_features for l in m.layers if not l.is_moe_layer])
    c = U.UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd, experts=E, expert_ff=ef,
                  top_k=moe0.top_k if moe0 is not None else 1, shared_ff=sff, mask_ext=True)
    kinds = getattr(tc, "layer_types", None) or [None] * tc.num_hidden_layers
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.norm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    rot = m.rotary_emb
    W = lambda lin: lin.weight.detach().to(F64)
    for ul, tl, kind in zip(u.layers, m.layers, kinds):
        at = tl.self_attn
        U._norm_from(ul.norm_a, tl.input_layernorm); U._norm_from(ul.norm_f, tl.post_attention_layernorm)
        U._attn_any(ul.attn, *U._qkv(at), n_kv=nkv, scale=at.scaling, inv_freq=rot.inv_freq if at.use_rope else None,
                    att_scale=float(rot.attention_scaling) if at.use_rope else 1.0, layout="interleave")
        a = ul.attn
        if hasattr(at, "qk_norm"):
            a.qk_post.fill_(1.0)
            for coef, w in ((a.q_coef, a.q_nw), (a.k_coef, a.k_nw)):
                coef.copy_(torch.tensor([0.0, 1.0])); w.fill_(1.0)
            a.qk_eps.fill_(float(at.qk_norm.eps))
        if at.attn_temperature_tuning and not at.use_rope:
            a.qtemp.copy_(torch.tensor([float(at.attn_scale), float(at.floor_scale)], dtype=F64))
            a.qtemp_off.fill_(1.0)                                     # floor((t + 1) / floor_scale)
        if kind == "chunked_attention":
            a.chunk.fill_(float(tc.attention_chunk_size))
        f, ff = ul.ffn, tl.feed_forward
        if not tl.is_moe_layer:
            k = ff.gate_proj.out_features
            f.s_gate.weight[:k] = W(ff.gate_proj); f.s_up.weight[:k] = W(ff.up_proj); f.s_down.weight[:, :k] = W(ff.down_proj)
            continue
        U._cp(f.router.weight, ff.router.weight)
        f.score_c.fill_(1.0); f.in_gate.fill_(1.0)
        gu = ff.experts.gate_up_proj.detach().to(F64)                # (E, d, 2 f)
        f.w_gate.copy_(gu[:, :, :ef].transpose(1, 2)); f.w_up.copy_(gu[:, :, ef:].transpose(1, 2))
        f.w_down.copy_(ff.experts.down_proj.detach().to(F64).transpose(1, 2))
        sh = ff.shared_expert
        k = sh.gate_proj.out_features
        f.s_gate.weight[:k] = W(sh.gate_proj); f.s_up.weight[:k] = W(sh.up_proj); f.s_down.weight[:, :k] = W(sh.down_proj)
    return u


@torch.no_grad()
def ernie55_router_into(f, gate_weight, bias, scale, renormalize=True):
    """ERNIE-5.5's split-feature router (vLLM-ERNIE moe.py: gate = Linear(hidden, 2 E); scores = sigmoid(l[:E]) +
    sigmoid(l[E:]); ernie5_5_routing l. 66-88: top-k on scores + bias, unbiased weights renormalised, x the per-expert
    scale) -> UFFN: router / router2 (UConfig router_dual), sel_bias, exp_scale."""
    U = _U()
    E = f.router.weight.shape[0]
    w = gate_weight.detach().to(F64)
    f.router.weight.copy_(w[:E]); f.router2.weight.copy_(w[E:])
    U._cp(f.sel_bias, bias); U._cp(f.exp_scale, scale)
    f.renorm.fill_(1.0 if renormalize else 0.0)


@torch.no_grad()
def _zamba1_mixer_into(s, mx):
    """Zamba's multi-head Mamba-1 mixer (ZambaMambaMixer: in_proj rows interleaved [x, gate] per channel; per head h of
    P = I / n_mamba_heads channels its own x_proj / dt_proj / A_log / D) -> the SSM block, one host head per channel
    (dk = state, dv = 1): channel c of head h reads B / C / dt from its head's post-conv channels."""
    I, N, R, nh = mx.intermediate_size, mx.ssm_state_size, mx.time_step_rank, mx.n_mamba_heads
    P = I // nh
    D = s.D                                                       # host u = [x | z]
    w = mx.in_proj.weight.detach().to(F64).view(I, 2, -1)
    s.in_proj.weight.copy_(torch.cat([w[:, 0], w[:, 1]]))
    if mx.in_proj.bias is not None:
        b = mx.in_proj.bias.detach().to(F64).view(I, 2)
        s.in_proj.bias.copy_(torch.cat([b[:, 0], b[:, 1]]))
    s.conv_w[:I] = mx.conv1d.weight.detach().to(F64)[:, 0]
    if mx.conv1d.bias is not None:
        s.conv_b[:I] = mx.conv1d.bias.detach().to(F64)
    s.act_silu[:I] = 1
    xw = mx.x_proj_weight.detach().to(F64)                         # (nh, R + 2N, P)
    dtw, dtb = mx.dt_proj_weight.detach().to(F64), mx.dt_proj_bias.detach().to(F64)   # (nh, P, R), (nh, P)
    for h in range(nh):
        cols = slice(D + h * P, D + (h + 1) * P)
        WB, WC, Wdt = xw[h, R:R + N], xw[h, R + N:], xw[h, :R]
        for p in range(P):
            c = h * P + p
            rows = slice(c * N, (c + 1) * N)
            s.pk.weight[rows, cols] = WB
            s.pq.weight[rows, cols] = WC
            s.p1.weight[rows, cols] = (dtw[h, p] @ Wdt).expand(N, P)
            s.p1.bias[rows] = dtb[h, p]
    s.pv.weight[:, D:D + I] = torch.eye(I, dtype=F64)
    s.A1.copy_(mx.A_log.detach().to(F64).exp().reshape(I, N))
    s.cw[0] = 1
    s.Dskip[:, 0] = mx.D.detach().to(F64).reshape(I)
    s.pz.weight[:, I:2 * I] = torch.eye(I, dtype=F64)
    s.gate[0] = 1
    s.out_coef[2] = 1
    _U()._cp(s.out_proj.weight, mx.out_proj.weight)
    if mx.out_proj.bias is not None:
        _U()._cp(s.out_proj.bias, mx.out_proj.bias)


@torch.no_grad()
def from_zamba(target):
    """Zamba v1 (transformers ZambaForCausalLM): multi-head Mamba-1 layers (_zamba1_mixer_into) and, on the hybrid
    layers, the ONE shared transformer block (concat(x, embedding) -> 2d RMSNorm -> attention with scale (hd / 2)^-1/2,
    no rotary -> RMSNorm -> SwiGLU, no inner residual) -> the layer's linear -> the Mamba layer's input (UCatBlock)."""
    U = _U()
    tc, m = target.config, target.model
    layers = list(m.layers)
    md0 = layers[0].mamba_decoder if hasattr(layers[0], "mamba_decoder") else layers[0]
    mx0 = md0.mamba
    I, N = mx0.intermediate_size, mx0.ssm_state_size
    blk = next((l.shared_transf for l in layers if hasattr(l, "shared_transf")), None)
    c = U.UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=len(layers), ssm_in=2 * I, conv_k=mx0.conv_kernel_size,
                  ssm_heads=I, dk=N, dv=1, cat_block=blk is not None, cat_heads=tc.num_attention_heads,
                  cat_head_dim=tc.attention_head_dim, cat_ff=tc.intermediate_size)
    u = U.UniversalElephantine(c)
    U._cp(u.embed.weight, m.embed_tokens.weight)
    U._norm_from(u.norm, m.final_layernorm)
    U._cp(u.lm_head.weight, target.lm_head.weight)
    for ul, tl in zip(u.layers, layers):
        md = tl.mamba_decoder if hasattr(tl, "mamba_decoder") else tl
        _zamba1_mixer_into(ul.ssm, md.mamba)
        U._norm_from(ul.norm_s, md.input_layernorm)
        if hasattr(tl, "shared_transf"):
            cb, b = ul.cat, tl.shared_transf
            at, a = b.self_attn, cb.attn
            H, hd = a.H, a.hd
            nkv = at.k_proj.out_features // hd
            a.q_proj.weight.copy_(at.q_proj.weight.detach().to(F64))
            a.k_proj.weight.copy_(U._tile_kv(at.k_proj.weight.detach().to(F64), nkv, H, hd))
            a.v_proj.weight.copy_(U._tile_kv(at.v_proj.weight.detach().to(F64), nkv, H, hd))
            a.o_proj.weight[:cb.d] = at.o_proj.weight.detach().to(F64)
            a.scale.fill_(float(at.scaling))
            a.inv_freq.zero_()
            U._norm_from(cb.norm_in, b.input_layernorm); U._norm_from(cb.norm_ff, b.pre_ff_layernorm)
            U._dense_ffn_from(cb.ffn, b.feed_forward)
            U._set_act(cb.ffn, tc.hidden_act, True)
            cb.c.fill_(1.0)
            U._cp(cb.lin.weight, tl.linear.weight)
    return u


@torch.no_grad()
def muse_spark_moe_into(f, moe):
    """Muse-Spark's latent MoE (MuseSparkMoE: sigmoid router on x * output_multiplier, selected on scores + bias,
    renormalised (+ 1e-15), x routed_scaling_factor; experts on rms(pre_expert_proj x)(1 + w); each expert's output
    through the input-scaled RMSNorm; the weighted sum through post_expert_proj) -> UFFN's latent branches
    (UConfig latent_moe = moe_hidden_size)."""
    U = _U()
    g, ex = moe.gate, moe.experts
    m = ex.down_proj.shape[1]
    U._cp(f.router.weight, g.weight.detach().to(F64) * float(g.output_multiplier))
    U._cp(f.sel_bias, g.e_score_correction_bias)
    f.score_c.fill_(1.0); f.renorm.fill_(1.0); f.renorm_eps.fill_(float(g.norm_topk_prob_eps))
    f.route_scale.fill_(float(g.routed_scaling_factor))
    U._cp(f.e_pre.P, moe.pre_expert_proj.weight)
    U._cp(f.e_pre.w, 1 + moe.pre_expert_norm.weight.detach().to(F64)); f.e_pre.eps.fill_(float(moe.pre_expert_norm.eps))
    U._cp(f.e_post.s, 1 + ex.post_expert_norm.weight.detach().to(F64)); f.e_post.eps.fill_(float(ex.post_expert_norm.eps))
    gu = ex.gate_up_proj.detach().to(F64)                          # (E, 2 ff, m)
    ff = gu.shape[1] // 2
    f.w_gate.zero_(); f.w_up.zero_(); f.w_down.zero_()
    f.w_gate[:, :ff, :m] = gu[:, :ff]; f.w_up[:, :ff, :m] = gu[:, ff:]
    f.w_down[:, :m, :ff] = ex.down_proj.detach().to(F64)
    f.e_out.weight.zero_(); f.e_out.weight[:, :m] = moe.post_expert_proj.weight.detach().to(F64)
    f.s_down.weight.zero_()
