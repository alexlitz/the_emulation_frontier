# Vendored from the_emulation_frontier/elephantine/universal.py by `python -m recipes.vendor`. Do not edit: edit elephantine/universal.py.
"""universal.py — ONE elephantine architecture that carries every family in the zoo.

The minimal emulator is the *join* of the zoo: it carries the union of what every model does.
`make_elephantine` builds that join one bucket at a time (a Jamba host for softmax + Mamba, a
separate `GatedDeltaLM` for the other recurrences, a separate dense host for diffusion). This
module is the single architecture behind all of them. Every layer is

    x += Attn(norm_a(x))      softmax attention, RoPE (settable inv_freq), causal OR full mask
    x += SSM(norm_s(x))       short conv -> general gated-delta recurrence -> norm/gate -> out
    x += FFN(norm_f(x))       routed SwiGLU MoE + a gated shared expert

and every op a family may or may not use is a BRANCH WITH A COEFFICIENT, off by default:
  attention  qk-norm (per-head weights), output gate, output RMS norm, sliding window + sink
             positions, a cross-layer KV register (read / write), learned prefix tokens, gated
             latent norms; memory-read branches: a key/value writer MLP, a score-selected mask
             (top-capacity admission scores of earlier chunks), a null key, a per-head read gate;
             a DSA k-pool indexer mask (top-scoring pools of keys + the query's tail; indexer RoPE,
             a mask shared to later layers); grouped differential heads o_h - sigmoid(dl_proj x)_h
             o_src(h) (Motif-3 GDLA); a logit softcap cap tanh(s / cap), ALiBi slopes, learned
             per-head sinks (the null key per head), qk-norm over the whole projection / as a
             LayerNorm / after the rotary / over a narrower head width, a v-norm, a q / k / v clamp,
             the Llama-4 query temperature, a softplus output gate, per-layer-type KV register slots
  layer      post-sublayer norms (sandwich), parallel attention + SSM (seq = 0), gated norms
             (low-rank sigmoid self-gate) and two-stage norms (two RMSNorms in series), a parallel
             memory attention (mem_attn), a hashed n-gram memory added before the layer (UEngram:
             the token embedding generalised to n-grams), a residual snapshot x <- p + c (x - p),
             norms of the residual stream itself (stream_norms), hyper-connections (hc_mult
             Sinkhorn-mixed residual streams, UHC), Block or Full AttnRes (depth-softmax residuals),
             a split FFN (shared and routed paths with their own input / output norms and a router
             norm: Gemma-4), per-layer embedding inputs (PLE), residual coefficients x <- c x + f(x),
             a per-layer output scale, identity norms
  SSM        normalized knob features (RMS-normed B/C/dt), decay / delta / write formulas, a
             previous-token write (trapezoid), a cumulative or constant q/k rotation, output
             gates, a channel mix of y (sums heads, centres for GroupNorm), an RMS norm whose
             grouping is a weight plus a bias, the RWKV-4 ratio, the RG-LRU reset, a dynamic
             (per-token generated) conv on v, an erase gate on the delta term's read key, a ridge
             readout (a second, key-covariance state H_t and a fixed-step Chebyshev solve of
             (H_t + lam_t I) x = q_t, lam_t = r ||H_t||_F, mixed into the query: Gated KalmaNet),
             a floor on dt (Mamba-2 time_step_limit)
  FFN        routed top-k MoE (+ renorm, + eps; selection on scores or logits + bias), a
             sigmoid-gated shared expert, hard SwiGLU clipping, ungated squared-ReLU and xIELU
             (learned alpha_p / alpha_n) activations, PolyNorm-GLU (per-expert weights, clamps),
             exact GeLU (erf or tanh-approx; ungated, or gated GeGLU) and gated mish, ungated ReLU,
             biases on the shared / dense path, the routed experts and the router, the swigluoai
             up-shift with separate gate / up clip limits, group-limited top-k (DeepSeek), a router
             logit softcap, a per-expert scale, SparseMixer top-2 (Phi-3.5-MoE)
  model      an embedding norm (Bloom), a final logit softcap, an lm_head bias
  readout    (non-next-token models, M12) a linear head of any width (a score / class head, an embedding
             projection or the identity) after a pooling rule (last real token / first / mean) with an
             optional hidden layer (a value-head MLP) and L2 normalisation; early-exit heads (layer-wise
             rerankers); a flow-matching head network (UFlowHead: Sundial; its sampler loop is runtime)
  input      a continuous-series patch embedding instead of the token table (UPatchEmbed: Sundial)
A target zeros the branches it doesn't use, so each family is a *weight setting* of the same
architecture — the elephantine's usual move (it already carries learned-abs AND RoPE).

The SSM recurrence is `attn.gated_delta_layer.gd_recurrence`:
    S_t = Diag(a_t) S_{t-1} - gamma_t (S_{t-1} k_t) k_t^T + w_t v_t k_t^T,   y_t = S_t q_t
with (dt = softplus(p1))
    log a = -(A1 * softplus(p1) + A2 * sigmoid(p2) + A3 + A4 * dt * max(heavy_tail(pA), floor)),
    a <- mem * a
    gamma = cg * sigmoid(pb)
    w     = cw1 * softplus(p1) + cw2 * sigmoid(pb) + cw3 * exp(pe)
            + cw4 * sqrt(1 - a^2) * sigmoid(pi)   (RG-LRU; replaced by sigmoid(pi) at t=0 if reset)
            + cw5 + cw6 * dt * sigmoid(pb)
All p* are linear maps of [pre-conv stream u, post-conv stream c, normalized features nb].
Every recurrent family in the zoo is a setting of these knobs: see the `from_*` (whole model)
and `*_block` (mixer) loaders and tests/test_universal_elephantine.py, which checks each one
against the family's real code. The stock `transformers` decoders (Llama / Mistral / Qwen / Gemma /
Phi / Bloom / OPT / Falcon / GPT-J / OLMo / Granite / DeepSeek / GPT-OSS / Mamba hybrids / ...) have
`from_*` loaders built on a generic reader (_attn_any / _mlp_any / _ffn_any / _router_from) and are
checked in tests/test_stock_loaders.py; elephantine/loader_coverage.py inventories which zoo models a
tested loader accepts.

Runs in fp64. Real `transformers` modules round some internals to fp32 (norms, scans, rotary
tables), so comparisons against them agree to that fp32 level; see the tests. Where a family's only
runnable code is a vLLM plugin (PseudoBrain's pb27-vllm), the loader reads the checkpoint itself
(`from_pseudobrain(config, state_dict)`) and the test runs the plugin under vLLM in an isolated venv.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tef_recipes.gated_delta import gd_recurrence
from tef_recipes import branches_stream as _bs

F64 = torch.float64


@dataclass
class UConfig:
    vocab: int
    d: int                      # hidden width
    L: int                      # layers
    heads: int = 1              # attention heads (MHA; GQA targets tile their KV)
    head_dim: int = 2
    ssm_in: int = 1             # in_proj width (the conv runs over all of it)
    conv_k: int = 1
    ssm_heads: int = 1
    dk: int = 1
    dv: int = 1
    experts: int = 1
    expert_ff: int = 1
    top_k: int = 1
    shared_ff: int = 1
    ssm_norm_feats: int = 0     # normalized knob features (B/C/dt RMS-normed before use)
    ssm_sig_feats: int = 0      # sigmoid knob features (low-rank tanh / sigmoid gates)
    ssm_prod_feats: int = 0     # product features f = (Px r) * (Py r + b)  (RWKV-6 ddlerp)
    ssm_sig2_feats: int = 0     # a second sigmoid stage reading the product features
    n_prefix: int = 0           # learned prefix (meta) tokens, prepended then stripped
    attnres_block: int = 0      # Block AttnRes: depth-softmax residual over block sums (0 = off)
    attnres_full: bool = False  # Full AttnRes: every sub-block output is its own source (GPT-S3)
    attn_norm_feats: int = 0    # attention normalized-latent features (real MLA's q / kv latent norms)
    lora_adapters: int = 0      # switched LoRA adapters selected per token by a switch head (aLoRA)
    lora_rank: int = 1
    max_pos: int = 0            # learned absolute positions (0 = none; the spec carries both)
    ssm_dconv: int = 0          # dynamic (per-token generated) causal conv on v, window W (0 = off; Jet)
    ssm_dconv_rank: int = 1     # its kernel generator's hidden width
    norm_gate_rank: int = 0     # gated norms: rms(x) w * c * sigmoid(down(silu(up(.))))  (GigaChat-3.5)
    attn_norm_gate_rank: int = 0   # the same gate on the attention normalized-latent features
    engram_slots: int = 0       # hashed n-gram memory (Engram) per layer: slots per (order, hash head); 0 = none
    engram_orders: tuple = (2, 3)
    engram_heads: int = 1
    engram_dim: int = 1
    engram_conv: int = 1
    mem_attn: bool = False      # a second, parallel attention per layer over a score-selected memory (EdgeRock)
    mem_kv_ff: int = 0          # its key/value writer MLP width
    mem_sel_rank: int = 0       # its admission-score MLP rank
    stream_norms: bool = False  # per-layer norms OF the residual stream itself, at layer start / end (CENO)
    hc_mult: int = 0            # manifold-constrained hyper-connections: N residual streams (0 = one plain stream)
    hc_iters: int = 1           # their Sinkhorn iterations
    dsa_heads: int = 0          # DeepSeek sparse attention indexer heads (0 = dense attention)
    dsa_dim: int = 1            # its head dim
    ple_dim: int = 0            # per-layer embeddings (Gemma-3n / 4 PLE): width per layer (0 = none)
    ple_vocab: int = 0          # their token table's vocabulary (0: the model's)
    ffn_shift: bool = False     # the RWKV channel-mix branch's d x d receptance (UFFN.shift_c)
    cmp_pool: int = 0           # compressed-KV pools (DeepSeek-V4 CSA / HCA): largest pool size (0 = none)
    cmp_index_heads: int = 0    # their lightning indexer's heads (0 = no indexer: every complete pool visible)
    cmp_index_dim: int = 2      # its head dim
    cmp_ix_latent: bool = False # the indexer's keys read the pooled latent (DeepSeek-V4.1: wk: head_dim -> index dim)
    engram_hc: int = 0          # hc-stream Engram (DeepSeek-V4.1): its table rows (0 = none)
    engram_hc_cols: int = 1     # hash columns per position ((max n-gram - 1) x heads)
    engram_hc_dim: int = 1      # table row width
    engram_hc_n: int = 2        # max n-gram size
    sink_key: bool = False      # a learned zero-value sink KEY per head (IQuest-Q1): logit (R_t q) . s, s not rotated
    # attention branches of the round-3 features (eleph-gaps-attn; each off by default, and its weights exist only
    # when its flag is set, so a host without them has the state dict it always had)
    xsa: bool = False           # exclusive self attention o_h - c_h (o_h . v_h) / (max(|v_h|^2, e0) + e1) v_h (GPT-X2 / S2,
    #                             Grug-MoE, Limite's tanh(alpha_h)): v_h the token's OWN value of head h's KV group
    post_gate: bool = False     # an output gate AFTER o_proj: o_proj(o) * (c0 + c1 sigmoid(W x)) (MM200 / Ying)
    mask_ext: bool = False      # chunked attention (Llama-4: same attention_chunk_size block) and Meditor's K / V
    #                             truncation to the last W keys under a top-left-aligned causal mask
    dyn_mask: bool = False      # Doge's dynamic mask: logit bias exp(A_h softplus(W v_s)_h) per key, top keep_window keys
    out_sat: bool = False       # D-ROS: a multiband soft-knee saturation of the attention output, dry / wet, layer gain
    doc_mask: bool = False      # same-document attention, documents ended by an eos id (Ethoric doc_masking, KLM)
    head_route: int = 0         # JetMoE MoA: routed attention experts (this many), each owning H / head_route heads
    # non-next-token readouts and inputs (M12): see UniversalElephantine.readout / UPatchEmbed / UFlowHead
    head_out: int = 0           # readout width (0: the vocabulary): a score / class / regression head, or an embedding
    head_ff: int = 0            # a hidden layer before the readout head: head(act(head_in(h))) (0: none; a value-head MLP)
    exits: tuple = ()           # early-exit readouts: layer counts e, each norm(x after e layers) -> its own head
    patch_len: int = 0          # continuous input: a float series cut into patches of this many values (0: token ids)
    patch_mask: bool = False    # ... with each patch's padding mask concatenated to it (Sundial)
    patch_ff: int = 1           # ... the patch MLP's hidden width
    flow_out: int = 0           # a flow-matching head network v(x, t, z) (Sundial): its target width (0: none)
    flow_width: int = 1         # ... its width
    flow_depth: int = 1         # ... its adaLN residual blocks
    flow_freq: int = 256        # ... its sinusoidal timestep features
    # residual-stream / input / FFN / norm / loop branches (elephantine/branches_stream.py; all off by default)
    hc_kind: str = "mhc"        # hyper-connection code: "mhc" UHC (Sinkhorn), "gated" Qwen4-Exp GatedResidual
    hc_rank: int = 1            # ... its low-rank input-mix width (hc_lowrank)
    ple_ng_rows: int = 0        # Qwen4-Exp n-gram PLE on the hc streams: table rows (0 = none)
    ple_ng_dim: int = 1         # ... row width
    ple_ng_n: int = 3           # ... largest n-gram
    ple_ng_heads: int = 1       # ... hash heads per n-gram order
    ple_ng_conv: int = 1        # ... its dilated depthwise conv: kernel
    ple_ng_dil: int = 1         # ... dilation
    cat_block: bool = False     # a block over concat(x, x0) (x0 = the embedding) feeding the SSM input (Zamba shared)
    cat_heads: int = 1          # ... its attention heads (input width 2d)
    cat_head_dim: int = 2       # ... head dim
    cat_ff: int = 1             # ... its gated MLP width
    ve_dim: int = 0             # per-layer value-embedding tables (nanochat / ARLoop / sn38): width (0 = none)
    ve_heads: int = 1           # ... their gated heads (the target's KV heads)
    ve_gate_ch: int = 1         # ... the gate reads the first ve_gate_ch input channels
    smear_ch: int = 0           # the embedding smear (the previous token, gated by the first smear_ch channels; 0 = off)
    x0_mix: bool = False        # x <- r x + s x0 at every layer start (nanochat resid / x0 lambdas)
    attnres_emb_partial: bool = False   # Block AttnRes: the first block's partial sum starts at the embedding (ARLoop)
    early_exit: bool = False    # a per-token exit among the tapped layer outputs (Ouro: gated exit pdf, a threshold)
    two_state: bool = False     # two residual states z_H / z_L in registers, read / written at stack ends (HRM)
    selfcond_ff: int = 0        # a self-conditioning input path (forward(selfcond=)): its gated MLP width (0 = none)
    res_gain: bool = False      # per-channel residual gains x <- g_x x + g_o f per sub-block (nGPT / Doge / ZAYA)
    loop_inject: bool = False   # loop input re-injection x <- n(x) + reg["inj"] at a pass start (Dory)
    o_late_norm: bool = False   # a norm of the gated attention output (Dory attn_sub_norm)
    ffn_hidden_norm: bool = False   # a norm of the shared / dense FFN's hidden activation (Dory ffn_sub_norm)
    canon_k: int = 0            # Canon layers: short causal depthwise convs of this kernel (0 = none) at canon_sites
    canon_sites: str = ""       # ... "A" the attention input, "B" q / k / v, "C" the FFN input, "D" the FFN hidden
    ngram_in_buckets: int = 0   # a hashed (previous, current) token embedding added to the input: buckets (0 = none)
    router_mlp: bool = False    # the router as an MLP W2 tanh(W1 x) (LLaMA-MoE)
    router_dual: bool = False   # scores sigmoid(W1 x) + sigmoid(W2 x) (ERNIE-5.5 split-feature routing)
    multimax: bool = False      # the logits through ERNIE-5.5's multimax (UMultimax)
    latent_moe: int = 0         # routed experts on a normed latent of this width with per-expert output norms (Muse)


def _lin(i, o, bias=True):
    # skip PyTorch's default init: the weights are zeroed right away
    m = torch.nn.utils.skip_init(nn.Linear, i, o, bias=bias, dtype=F64)
    with torch.no_grad():
        m.weight.zero_()
        if bias:
            m.bias.zero_()
    return m


def _p(*shape, fill=0.0):
    return nn.Parameter(torch.full(shape, fill, dtype=F64))


def _lora(x, A, B, idx):
    """Switched LoRA: token t with adapter idx[t] = a > 0 gets  B[a-1] (A[a-1] x)."""
    if idx is None or A.shape[0] == 0 or not bool((idx > 0).any()):
        return 0
    out = 0
    for e in range(A.shape[0]):
        m = (idx == e + 1)[..., None].to(x.dtype)
        if m.any():
            out = out + m * ((x @ A[e].T) @ B[e].T)
    return out


def _rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _sig_gate(y, c, up, down):
    """Low-rank self-gate  y * c * sigmoid(down(silu(up(y))))  (off when c = 0)."""
    return y if c == 0 else y * c * torch.sigmoid(down(F.silu(up(y))))


class UNorm(nn.Module):
    """RMSNorm with a settable weight and eps; centre = 1 plus a bias makes it LayerNorm;
    gate_c != 0 multiplies by a low-rank sigmoid self-gate (GigaChat-3.5's gated norm)."""

    def __init__(self, d, gate_rank=0):
        super().__init__()
        self.weight = _p(d, fill=1.0)
        self.bias, self.centre = _p(d), _p(1)
        self.register_buffer("eps", torch.tensor(1e-6, dtype=F64))
        self.gate_c = _p(1)
        self.g_up, self.g_down = _lin(d, max(gate_rank, 1), bias=False), _lin(max(gate_rank, 1), d, bias=False)
        self.pre_c, self.pre_w = _p(1), _p(d, fill=1.0)  # a first RMSNorm stage (two norms in series)
        self.register_buffer("pre_eps", torch.tensor(1e-6, dtype=F64))
        self.ident = _p(1)                               # 1: no norm at all (post-norm-only layers: OLMo-2)
        self.register_buffer("groups", torch.tensor(1.0, dtype=F64))   # > 1: the RMS per group of d / groups (K2-Horizon)

    def forward(self, x):
        if self.ident != 0:
            return x
        if self.pre_c != 0:
            x = _rms(x, self.pre_eps) * self.pre_w
        if self.centre != 0:
            x = x - self.centre * x.mean(-1, keepdim=True)
        r = _rms(x, self.eps) if self.groups == 1 else _rms(x.unflatten(-1, (int(self.groups), -1)), self.eps).flatten(-2)
        return _sig_gate(r * self.weight + self.bias, self.gate_c, self.g_up, self.g_down)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def _sat_band(x, b):
    """D-ROS _saturate_band (one band b = [threshold, knee, alpha, curve, polarity]): above the threshold the
    excess e is replaced by curve(e) = knee tanh(e alpha / knee) (0), knee sqrt(softplus(e alpha / knee)) (1), 0 (2)
    or e (1 - alpha) (3); positive-only (0) on x, symmetric (1) on |x| as a gain; polarity 2 is the identity."""
    th, knee, alpha, curve, pol = (float(t) for t in b)
    if pol == 2:
        return x
    def tf(e):
        kn = max(knee, 1e-6)
        if curve == 0:
            return knee * torch.tanh(e * alpha / kn)
        if curve == 1:
            return knee * torch.sqrt(F.softplus(e * alpha / kn))
        if curve == 2:
            return torch.zeros_like(e)
        return e * (1.0 - alpha)
    if pol == 0:
        return torch.where(x > th, th + tf(torch.clamp(x - th, min=0.0)), x)
    ax = x.abs()
    gain = torch.where(ax > th, (th + tf(torch.clamp(ax - th, min=0.0))) / torch.clamp(ax, min=1e-6), 1.0)
    return x * gain


def _dros(x, cfg, bands):
    """D-ROS on an attention output x (B, T, d): cfg = [on, dry_wet, makeup, layer gain, multiband, k_low, k_mid].
    on: the saturation (multiband: complementary low / mid / high bands from two stride-1 average pools over the
    CHANNEL axis, each band saturated by its own row of `bands`, summed; else row 0), times makeup, mixed
    (1 - w) x + w shaped; then (on or not) the layer gain."""
    on, w, makeup, gain, mb, kl, km = (float(t) for t in cfg)
    if on:
        if mb:
            B, T, d = x.shape
            r = x.reshape(B * T, 1, d)
            lo = F.avg_pool1d(r, kernel_size=int(kl), stride=1, padding=int(kl) // 2)
            hi = r - lo
            mid = F.avg_pool1d(hi, kernel_size=int(km), stride=1, padding=int(km) // 2)
            hi = hi - mid
            sh = sum(_sat_band(t.view(B, T, d), bands[i]) for i, t in enumerate((lo, mid, hi)))
        else:
            sh = _sat_band(x, bands[0])
        if makeup != 1.0:
            sh = sh * makeup
        x = sh if w >= 1.0 else (x if w <= 0.0 else (1.0 - w) * x + w * sh)
    return x * gain if gain != 1.0 else x


def _select_mask(sc, chunk, cap):
    """Score-selected memory: key s is visible to query t iff s precedes t's chunk, s < c(t) =
    floor(t / chunk) chunk, and s is among the `cap` highest scores of the keys before c(t)
    (ties to the earlier position). sc (B,H,T) -> (B,H,T,S) bool."""
    T = sc.shape[-1]
    pos = torch.arange(T)
    before = pos[None, :] < (pos // chunk * chunk)[:, None]             # (T, S)
    a, b = sc[..., :, None], sc[..., None, :]                            # a: s', b: s
    beats = (a > b) | ((a == b) & (pos[:, None] < pos[None, :]))         # s' outranks s
    rank = torch.einsum("tu,bhus->bhts", before.to(sc.dtype), beats.to(sc.dtype))
    return before & (rank < cap)


class UAttention(nn.Module):
    def __init__(self, c: UConfig, mem=False):
        super().__init__()
        H, hd, d = c.heads, c.head_dim, c.d
        self.H, self.hd = H, hd
        Dn = c.attn_norm_feats
        self.Dn = Dn                                     # q/k/v/g read [x, nb], nb = rms_grouped(pn x) * n_w
        self.pn = _lin(d, max(Dn, 1), bias=False)
        self.n_avg, self.n_w = _p(max(Dn, 1), max(Dn, 1)), _p(max(Dn, 1), fill=1.0)
        self.register_buffer("n_eps", torch.tensor(1e-6, dtype=F64))
        gr = max(c.attn_norm_gate_rank, 1)               # gated latent norms (block-diagonal up / down)
        self.n_gate_c, self.n_up, self.n_down = _p(1), _lin(max(Dn, 1), gr, bias=False), _lin(gr, max(Dn, 1), bias=False)
        din = d + Dn
        self.q_proj, self.k_proj, self.v_proj = _lin(din, H * hd), _lin(din, H * hd), _lin(din, H * hd)
        self.g_proj, self.o_proj = _lin(din, H * hd), _lin(H * hd, d)
        E, r = c.lora_adapters, c.lora_rank
        self.la = nn.ParameterDict({n: _p(E, r, i) for n, i in (("q", din), ("k", din), ("v", din), ("o", H * hd))})
        self.lb = nn.ParameterDict({n: _p(E, o, r) for n, o in (("q", H * hd), ("k", H * hd), ("v", H * hd), ("o", d))})
        self.q_nw, self.k_nw = _p(H, hd, fill=1.0), _p(H, hd, fill=1.0)   # per-head qk-norm weights
        self.register_buffer("qk_eps", torch.tensor([1e-6, 1e-6], dtype=F64))
        self.q_coef, self.k_coef = _p(2), _p(2)          # [plain, normed]
        self.gate_coef = _p(2)                           # [constant 1, sigmoid(g_proj x)]
        with torch.no_grad():
            self.q_coef[0] = self.k_coef[0] = self.gate_coef[0] = 1.0
        self.inv_freq = _p(hd // 2)                      # 0 = NoPE on that pair
        self.scale = _p(1, fill=1 / math.sqrt(hd))
        C = H * hd
        self.o_hmix = nn.Parameter(torch.eye(H, dtype=F64))   # head mix before the output norm (DIFF attn)
        self.o_coef = _p(2)                              # [raw, RMS-normed] attention output
        self.o_avg, self.o_w = _p(C, C), _p(C, fill=1.0)
        self.register_buffer("o_eps", torch.tensor(1e-6, dtype=F64))
        self.kv_read, self.kv_write = _p(1), _p(1)       # cross-layer KV register (post-RoPE k, v)
        self.register_buffer("window", torch.tensor(float("inf"), dtype=F64))   # t - s <= window
        self.register_buffer("n_sink", torch.tensor(0.0, dtype=F64))            # s < n_sink always visible
        # memory-read branches (EdgeRock's independent memory; off by default):
        f, r = max(c.mem_kv_ff if mem else 0, 1), max(c.mem_sel_rank if mem else 0, 1)
        self.kv_c = _p(1)                                # k, v read z = x + kv_c down(silu(gate x) * up x)
        self.kv_gate, self.kv_up, self.kv_down = _lin(din, f, bias=False), _lin(din, f, bias=False), _lin(f, din, bias=False)
        self.sel_down, self.sel_up = _lin(din, r, bias=False), _lin(r, H, bias=False)   # score up(silu(down x))
        self.register_buffer("sel_cap", torch.tensor(float("inf"), dtype=F64))         # keys kept (inf = off)
        self.register_buffer("sel_chunk", torch.tensor(1.0, dtype=F64))
        self.register_buffer("null_logit", torch.full((H,), float("-inf"), dtype=F64))   # a zero-value null key
        #                                                  per head (learned sinks: GPT-OSS; -inf = off)
        if c.sink_key:                                   # a zero-value null KEY per head, scored by the query:
            self.null_key = _p(H, hd)                    # logit scale (q_t . s_h), q normed / rotated, s not (IQuest-Q1)
        self.register_buffer("cap", torch.tensor(float("inf"), dtype=F64))   # logit softcap cap tanh(s / cap)
        self.alibi = _p(H)                               # ALiBi slopes: s += alibi_h (s_pos - t_pos)
        self.register_buffer("qkv_clip", torch.tensor(float("inf"), dtype=F64))   # q, k, v clamped to +-clip
        # qk-norm variants: qk_full = 1 normalises over the WHOLE projection (all heads: OLMo-2 / 3), qk_centre
        # [q, k] = 1 makes it a LayerNorm (mean-centred, + q_nb / k_nb biases: StableLM, Persimmon, Phi);
        # qk_post = 1 applies it AFTER the rotary (HunYuan)
        self.qk_full, self.qk_centre, self.qk_post = _p(1), _p(2), _p(1)
        self.q_nb, self.k_nb = _p(H, hd), _p(H, hd)
        # the width the per-head q / k / v norms average over (< hd when a narrower target head is zero-padded)
        self.register_buffer("qk_width", torch.tensor(float(hd), dtype=F64))
        self.v_c, self.v_nw = _p(1), _p(H, hd, fill=1.0)   # v-norm: v <- rms_head(v) v_nw (Gemma-4)
        self.register_buffer("v_eps", torch.tensor(1e-6, dtype=F64))
        self.register_buffer("kv_slot", torch.tensor(0.0, dtype=F64))   # which KV register (per layer type)
        # Llama-4 / Ministral-3 query temperature: q_t *= 1 + beta log(1 + floor(t / M)), [beta, M] (beta = 0: off)
        self.register_buffer("qtemp", torch.tensor([0.0, 1.0], dtype=F64))
        # a learned per-head query temperature q * (qt_s log(t + 1) + qt_b) (Mistral-Large-4's softmax_scaling; off: 0 / 1)
        self.register_buffer("qt_s", torch.zeros(H, dtype=F64))
        self.register_buffer("qt_b", torch.ones(H, dtype=F64))
        # a code-fixed power-law query temperature q * (1 + (t + a) / c)^e, [e, a, c] (Beam's NoPE layers; e = 0: off)
        self.register_buffer("qt_pow", torch.tensor([0.0, 1.0, 2048.0], dtype=F64))
        self.gate_sp = _p(1)                             # + gate_sp softplus(g_proj x) in the output gate (Laguna)
        # + gate_silu silu(g_proj x) in the output gate (K2-Horizon attention_gate_func "silu"; 0: off)
        self.register_buffer("gate_silu", torch.tensor(0.0, dtype=F64))
        self.hg_c, self.hg_w, self.hg_b = _p(H), _p(H, hd), _p(H)   # head gate c sigmoid(<q_normed, w> + b)
        # grouped differential heads (Motif-3 GDLA): o_h <- o_h - dl_c m_h sigmoid(dl_proj x)_h o_{src(h)}
        self.dl_c, self.dl_proj = _p(1), _lin(din, H, bias=False)
        self.register_buffer("dl_src", torch.arange(H))
        self.register_buffer("dl_m", torch.zeros(H, dtype=F64))
        # DSA indexer (GLM-5.3 / DeepSeek-V3.2 with k-pool compression): a key is visible iff it is in one of
        # the top-scoring complete pools of `dsa_pool` keys, or in the query's incomplete tail pool
        nI, dI = max(c.dsa_heads, 1), c.dsa_dim
        self.dsa_on = c.dsa_heads > 0
        self.iq, self.ik, self.igate = _lin(din, nI * dI, bias=False), _lin(din, dI, bias=False), _lin(din, dI, bias=False)
        self.ik_w, self.ik_b, self.i_ape = _p(dI, fill=1.0), _p(dI), _p(1, dI)
        self.iw = _lin(din, nI, bias=False)
        self.register_buffer("ik_eps", torch.tensor(1e-6, dtype=F64))
        self.register_buffer("dsa_pool", torch.tensor(1.0, dtype=F64))
        self.register_buffer("dsa_topk", torch.tensor(float("inf"), dtype=F64))
        self.register_buffer("dsa_tail", torch.tensor(1.0, dtype=F64))
        self.i_inv = _p(max(dI // 2, 1))                 # RoPE on the indexer q / k (pairs (i, i + dI/2); 0 = NoPE)
        self.register_buffer("dsa_share", torch.tensor(0.0, dtype=F64))   # 1: publish the mask, 2: reuse it
        # K = V with a rotated value (DeepSeek-V4): v_rope = 1 rotates v like k (at its key position), o_derot = 1
        # rotates the output back at the query position (rotary angle -t), so each value enters at s - t
        self.register_buffer("v_rope", torch.tensor(0.0, dtype=F64))
        self.register_buffer("o_derot", torch.tensor(0.0, dtype=F64))
        # compressed-KV pools (DeepSeek-V4 CSA / HCA): every `cmp_rate` tokens one extra key = value per pool,
        # c_w = rms(sum_j softmax_j(gate_j + pb_j) * kv_j) w rotated at position w * rate, over the pool's own
        # tokens (the second half of cmp_kv / cmp_gate) and, with cmp_ovl = 1, the previous pool's (the first
        # half: CSA's overlapping windows); visible to query t iff its pool is complete, w < (t + 1) // rate,
        # and (with an indexer, ix_topk finite) among the top ix_topk pools by sum_h w_h relu(<q_h, k_w>)
        self.cmp = None
        P = c.cmp_pool if not mem else 0
        if P:
            nI, dI = max(c.cmp_index_heads, 1), c.cmp_index_dim
            self.cmp = nn.Module()
            m = self.cmp
            m.kv, m.gate, m.pb = _lin(din, 2 * hd, bias=False), _lin(din, 2 * hd, bias=False), _p(P, 2 * hd)
            m.nw = _p(hd, fill=1.0)
            m.ix_kv, m.ix_gate, m.ix_pb = _lin(din, 2 * dI, bias=False), _lin(din, 2 * dI, bias=False), _p(P, 2 * dI)
            m.ix_nw, m.ix_q, m.ix_w = _p(dI, fill=1.0), _lin(din, nI * dI, bias=False), _lin(din, nI, bias=False)
            m.ix_inv = _p(dI // 2)
            # DeepSeek-V4.1: index keys from the pooled latent (ix_wk), the pools / index keys / chosen pools /
            # candidate blocks handed between layers through the register; share = [kv: 0 own, 1 publish, 2 read;
            # index keys: 0 own pools (V4), 1 from the latent + publish, 2 read; picks: 0 / 1 own indexer (+ publish),
            # 2 read; candidates: 0 none, 1 publish, 2 mask the scores by them], cand = [top blocks, block size]
            m.ix_wk = _lin(hd if c.cmp_ix_latent else 1, dI, bias=False)
            m.register_buffer("share", torch.zeros(4, dtype=F64))
            m.register_buffer("cand", torch.tensor([0.0, 1.0], dtype=F64))
            # [rate (0 = off), overlap, norm eps, indexer top-k (inf = no indexer), indexer eps, q.k scale, w scale]
            m.register_buffer("cfg", torch.tensor([0.0, 0.0, 1e-6, float("inf"), 1e-6, 1.0, 1.0], dtype=F64))
        # --- the round-3 attention branches (eleph-gaps-attn); None = not built (UConfig flag off) ---
        self.xsa_c = self.pg_proj = self.chunk = self.dm_proj = self.sat = self.hr_proj = None
        self.add_branches(xsa=c.xsa, post_gate=c.post_gate, mask_ext=c.mask_ext, dyn_mask=c.dyn_mask,
                          out_sat=c.out_sat, head_route=c.head_route)
        with torch.no_grad():
            self.o_coef[0] = 1.0

    def add_branches(self, xsa=False, post_gate=False, mask_ext=False, dyn_mask=False, out_sat=False, head_route=0):
        """Build the round-3 attention branches named (each starts OFF: a zero coefficient, an infinite size);
        a loader may add them to an already-built host (from_llama + one branch)."""
        H, hd = self.H, self.hd
        din, d = self.q_proj.in_features, self.o_proj.out_features
        tensor = lambda v: torch.tensor(v, dtype=F64)
        if xsa and self.xsa_c is None:                   # XSA strength per head (1: unit XSA, tanh(alpha): Limite)
            self.xsa_c = _p(H)
            self.register_buffer("xsa_eps", tensor([1e-6, 0.0]))    # [clamp_min of |v|^2, + eps]
        if post_gate and self.pg_proj is None:           # out <- o_proj(o) * (pg_c0 + pg_c1 sigmoid(pg_proj x))
            self.pg_proj, self.pg_c = _lin(din, d, bias=False), _p(2)
            with torch.no_grad():
                self.pg_c[0] = 1.0
        if mask_ext and self.chunk is None:
            # chunk: key s visible to query t only within t's block, floor(s / chunk) = floor(t / chunk) (inf = off);
            # trunc: Meditor's sliding layers keep the last W keys and run a TOP-LEFT aligned causal mask over them,
            # so for T > W query t sees keys T - W .. T - W + t (inf = off)
            del self.chunk
            self.register_buffer("chunk", tensor(float("inf")))
            self.register_buffer("trunc", tensor(float("inf")))
            # Llama-4's query temperature reads floor((t + qtemp_off) / M) (Ministral-3: offset 0)
            self.register_buffer("qtemp_off", tensor(0.0))
        if dyn_mask and self.dm_proj is None:            # Doge: s_ts += dm_c exp(A_h softplus(dm_proj(v_s))_h)
            self.dm_proj, self.dm_A, self.dm_c = _lin(H * hd, H), _p(H), _p(1)
            self.register_buffer("dm_keep", tensor(float("inf")))      # keys kept per query (the top biases)
        if out_sat and self.sat is None:                 # D-ROS (_dros): [on, dry_wet, makeup, layer gain,
            self.sat = nn.Module()                       #   multiband, low kernel, mid kernel]
            self.sat.register_buffer("cfg", tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]))
            # per band (single band: row 0) [threshold, knee, alpha, curve (0 tanh, 1 sqrt-softplus, 2 hard,
            # 3 linear), polarity (0 positive only, 1 symmetric, 2 none)]
            self.sat.register_buffer("bands", tensor([[26.0, 6.0, 0.35, 0.0, 0.0]] * 3))
        if head_route and self.hr_proj is None:          # JetMoE MoA: head h belongs to expert h // (H / E)
            self.hr_proj = _lin(din, head_route, bias=False)
            self.register_buffer("hr_k", tensor(0.0))   # experts per token (0 = off)
        return self

    def _qk_norm(self, t, i):
        """[plain, normed] mix of q (i = 0) or k (i = 1): RMS-normed per head, or over every head
        (qk_full), optionally mean-centred with a bias (qk_centre: a LayerNorm)."""
        coef = (self.q_coef, self.k_coef)[i]
        if coef[1] == 0:
            return coef[0] * t
        dims = (-2, -1) if self.qk_full != 0 else (-1,)
        n = t - t.mean(dims, keepdim=True) if self.qk_centre[i] != 0 else t
        width = self.qk_width * (t.shape[-2] if self.qk_full != 0 else 1)
        n = n * torch.rsqrt(n.pow(2).sum(dims, keepdim=True) / width + self.qk_eps[i]) * (self.q_nw, self.k_nw)[i]
        if self.qk_centre[i] != 0:
            n = n + (self.q_nb, self.k_nb)[i]
        return coef[0] * t + coef[1] * n

    def forward(self, x, causal, reg=None):
        B, T, _ = x.shape
        sh = lambda t: t.view(B, T, self.H, self.hd)
        if self.Dn:
            n = self.pn(x)
            n = n * torch.rsqrt(n.pow(2) @ self.n_avg + self.n_eps) * self.n_w
            x = torch.cat((x, _sig_gate(n, self.n_gate_c, self.n_up, self.n_down)), -1)
        idx = reg.get("adapter") if reg is not None else None
        L = lambda n, t: _lora(t, self.la[n], self.lb[n], idx)
        xk = x if self.kv_c == 0 else x + self.kv_c * self.kv_down(F.silu(self.kv_gate(x)) * self.kv_up(x))
        q, k, v = sh(self.q_proj(x) + L("q", x)), sh(self.k_proj(xk) + L("k", xk)), sh(self.v_proj(xk) + L("v", xk))
        if getattr(self, "ve", None) is not None:        # gated per-token value embeddings (branches_stream.UValueEmb)
            v = self.ve(v, x, reg)
        if getattr(self, "canon_b", None) is not None:   # short causal convs on q / k / v (Canon-B)
            q, k, v = self.canon_b(q, k, v)
        if self.v_c != 0:
            v = v * torch.rsqrt(v.pow(2).sum(-1, keepdim=True) / self.qk_width + self.v_eps) * self.v_nw
        ang = torch.arange(T, dtype=F64)[:, None] * self.inv_freq[None]
        cos = torch.cat((ang.cos(), ang.cos()), -1)[None, :, None]
        sin = torch.cat((ang.sin(), ang.sin()), -1)[None, :, None]
        rope = lambda t: t * cos + _rotate_half(t) * sin
        if self.qk_post != 0:
            q, k = rope(q), rope(k)
        q, k = self._qk_norm(q, 0), self._qk_norm(k, 1)
        if torch.isfinite(self.qkv_clip):                # after the qk-norm (OLMoE), before the rotary
            c = self.qkv_clip
            q, k, v = q.clamp(-c, c), k.clamp(-c, c), v.clamp(-c, c)
        qn = q
        if self.qk_post == 0:
            q, k = rope(q), rope(k)
        if self.v_rope != 0:
            v = rope(v)
        if self.qtemp[0] != 0:
            t_ = torch.arange(T, dtype=F64) + float(getattr(self, "qtemp_off", 0.0))   # (Llama-4: floor((t + 1) / M))
            q = q * (1 + self.qtemp[0] * torch.log1p(torch.floor(t_ / self.qtemp[1])))[None, :, None, None]
        if self.qt_s.any() or (self.qt_b != 1).any():
            lp = torch.log(torch.arange(T, dtype=F64) + 1)
            q = q * (self.qt_s[None] * lp[:, None] + self.qt_b[None])[None, :, :, None]
        if self.qt_pow[0] != 0:
            e_, a_, c_ = self.qt_pow
            q = q * ((1 + (torch.arange(T, dtype=F64) + a_) / c_) ** e_)[None, :, None, None]
        if reg is not None:
            sk, sv = ("k", "v") if self.kv_slot == 0 else (f"k{int(self.kv_slot)}", f"v{int(self.kv_slot)}")
            if self.kv_read != 0:
                k = (1 - self.kv_read) * k + self.kv_read * reg[sk]
                v = (1 - self.kv_read) * v + self.kv_read * reg[sv]
            if self.kv_write != 0:
                reg[sk], reg[sv] = k, v
        s = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        if torch.isfinite(self.cap):
            s = self.cap * torch.tanh(s / self.cap)
        tt, ss = torch.arange(T)[:, None], torch.arange(T)[None]
        if self.alibi.any():
            s = s + self.alibi[:, None, None] * (ss - tt).to(F64)
        allowed = (tt - ss <= self.window) | (ss < self.n_sink)
        top_left = False
        if self.chunk is not None:                       # (mask_ext) Llama-4 chunks; Meditor's truncated K / V
            if torch.isfinite(self.chunk):
                allowed = allowed & (torch.div(tt, self.chunk, rounding_mode="floor")
                                     == torch.div(ss, self.chunk, rounding_mode="floor"))
            top_left = bool(torch.isfinite(self.trunc)) and T > self.trunc
            if top_left:
                assert not isinstance(causal, torch.Tensor) and causal, "K / V truncation is a causal-prefill rule"
                off = T - int(self.trunc)
                allowed = allowed & (ss >= off) & (ss - off <= tt)
        if isinstance(causal, torch.Tensor):             # a caller-supplied / block-causal mask (B|1, 1, T, S)
            allowed = allowed & causal
        elif causal and not top_left:
            allowed = allowed & (ss <= tt)
        if torch.isfinite(self.sel_cap):
            sc = self.sel_up(F.silu(self.sel_down(x))).transpose(1, 2)
            allowed = allowed & _select_mask(sc, int(self.sel_chunk), self.sel_cap)
        if self.dsa_on and (torch.isfinite(self.dsa_topk) or self.dsa_share != 0):
            if self.dsa_share == 2:                                       # GLM-5 'shared' indexer layers
                vis = reg["dsa"]
            elif torch.isfinite(self.dsa_topk):
                vis = self._dsa_mask(x)
            else:
                vis = torch.ones(B, T, T, dtype=torch.bool)
            if self.dsa_share == 1:
                reg["dsa"] = vis
            allowed = allowed & vis[:, None]
        if self.dm_proj is not None and self.dm_c != 0:  # Doge's dynamic mask: a per-key bias from the values
            dm = self.dm_c * torch.exp(self.dm_A * F.softplus(self.dm_proj(v.reshape(B, T, -1))))   # (B, T, H)
            dm = dm.transpose(1, 2)[:, :, None, :]                                                 # (B, H, 1, S)
            s = s + dm
            if T > self.dm_keep:                         # only the keep_window highest-bias visible keys per query
                rank = dm.expand(B, -1, T, -1).masked_fill(~allowed, float("-inf"))
                top = rank.topk(int(self.dm_keep), -1).indices
                allowed = allowed & torch.zeros_like(rank, dtype=torch.bool).scatter(-1, top, True)
        s = s.masked_fill(~allowed, float("-inf"))
        kc = None
        if self.cmp is not None and self.cmp.cfg[0] > 0:
            kc, vis = self._cmp_pools(x, reg)                             # (B, W, hd), (B, T, W)
            sc = torch.einsum("bthd,bsd->bhts", q, kc) * self.scale
            s = torch.cat((s.expand(B, -1, -1, -1), sc.masked_fill(~vis[:, None], float("-inf"))), -1)
        if torch.isfinite(self.null_logit).any():
            s = torch.cat((s, self.null_logit[:, None, None].expand(*s.shape[:-1], 1)), -1)
        if getattr(self, "null_key", None) is not None:
            ns = torch.einsum("bthd,hd->bht", q, self.null_key) * self.scale
            s = torch.cat((s.expand(B, -1, -1, -1), ns[..., None]), -1)
        p_ = s.softmax(-1)
        o = torch.einsum("bhts,bshd->bthd", p_[..., :T], v)
        if kc is not None:                               # K = V: the pools are their own values
            o = o + torch.einsum("bhts,bsd->bthd", p_[..., T:T + kc.shape[1]], kc)
        if self.xsa_c is not None and self.xsa_c.any():  # XSA: drop each head's component along its own value
            den = v.pow(2).sum(-1, keepdim=True).clamp_min(self.xsa_eps[0]) + self.xsa_eps[1]
            o = o - self.xsa_c[:, None] * ((o * v).sum(-1, keepdim=True) / den) * v
        if self.hr_proj is not None and self.hr_k > 0:  # MoA: each expert's heads times its top-k softmax gate
            tv, ti = self.hr_proj(x).topk(int(self.hr_k), -1)
            g = torch.zeros(B, T, self.hr_proj.out_features, dtype=o.dtype).scatter(-1, ti, tv.softmax(-1))
            o = o * g.repeat_interleave(self.H // g.shape[-1], -1)[..., None]
        if self.o_derot != 0:
            o = o * cos - _rotate_half(o) * sin
        if self.hg_c.any():
            o = o * (self.hg_c * torch.sigmoid((qn * self.hg_w).sum(-1) + self.hg_b))[..., None]
        if self.dl_c != 0:                               # signal heads minus a gated copy of their noise head
            o = o - self.dl_c * (self.dl_m * torch.sigmoid(self.dl_proj(x)))[..., None] * o[:, :, self.dl_src]
        if not torch.equal(self.o_hmix, torch.eye(self.H, dtype=F64)):
            o = torch.einsum("bthd,hg->btgd", o, self.o_hmix)
        o = o.reshape(B, T, -1)
        if self.o_coef[1] != 0:
            o = self.o_coef[0] * o + self.o_coef[1] * o * torch.rsqrt(o.pow(2) @ self.o_avg + self.o_eps) * self.o_w
        if self.gate_silu != 0:
            g = self.g_proj(x)
            o = o * (self.gate_coef[0] + self.gate_coef[1] * torch.sigmoid(g) + self.gate_sp * F.softplus(g)
                     + self.gate_silu * F.silu(g))
        elif self.gate_sp != 0:
            o = o * (self.gate_coef[0] + self.gate_coef[1] * torch.sigmoid(self.g_proj(x)) + self.gate_sp * F.softplus(self.g_proj(x)))
        else:
            o = o * (self.gate_coef[0] + self.gate_coef[1] * torch.sigmoid(self.g_proj(x)))
        if getattr(self, "o_late", None) is not None:    # a norm of the GATED output (Dory's attn_sub_norm)
            o = self.o_late(o)
        out = self.o_proj(o) + L("o", o)
        if self.pg_proj is not None and self.pg_c[1] != 0:   # MM200: the gate after o_proj
            out = out * (self.pg_c[0] + self.pg_c[1] * torch.sigmoid(self.pg_proj(x)))
        if self.sat is not None:
            out = _dros(out, self.sat.cfg, self.sat.bands)
        return out


def _dsa_mask(self, x):
    """DSA k-pool indexer -> visible keys (B,T,S). Pools of P consecutive keys; a pool's key is a
    per-channel softmax(gate x + ape) average of LayerNormed key projections; the pool score for a
    query is sum_h w_h relu(<q_h, pool key> / sqrt(dI)); the top floor(topk / P) complete pools whose
    last key precedes the query are kept, plus the query's own incomplete tail."""
    B, T, _ = x.shape
    P, nI, dI = int(self.dsa_pool), self.iw.out_features, self.ik.out_features
    q = self.iq(x).view(B, T, nI, dI)
    k = F.layer_norm(self.ik(x), (dI,), self.ik_w, self.ik_b, float(self.ik_eps))
    if self.i_inv.any():                                                    # indexer RoPE (DeepSeek-V3.2 / GLM-5)
        ang = torch.arange(T, dtype=F64)[:, None] * self.i_inv[None]
        cos, sin = torch.cat((ang.cos(), ang.cos()), -1), torch.cat((ang.sin(), ang.sin()), -1)
        q = q * cos[None, :, None] + _rotate_half(q) * sin[None, :, None]
        k = k * cos[None] + _rotate_half(k) * sin[None]
    gs = self.igate(x)
    tt = torch.arange(T)
    vis = torch.zeros(B, T, T, dtype=torch.bool)
    npool = T // P
    if npool and int(self.dsa_topk) // P:
        kk = k[:, :npool * P].view(B, npool, P, dI)
        gg = gs[:, :npool * P].view(B, npool, P, dI) + self.i_ape
        pk = (gg.softmax(2) * kk).sum(2)                                       # (B, pools, dI)
        sc = F.relu(torch.einsum("bthd,bpd->bthp", q, pk) / math.sqrt(dI))
        isc = torch.einsum("bth,bthp->btp", self.iw(x) / math.sqrt(nI), sc)
        ok = ((torch.arange(npool) * P + P - 1)[None, :] <= tt[:, None]).expand(B, T, npool)
        isc = isc.masked_fill(~ok, torch.finfo(isc.dtype).min)
        sel = isc.topk(min(int(self.dsa_topk) // P, npool), -1).indices
        keep = torch.zeros(B, T, npool, dtype=torch.bool).scatter_(-1, sel, True) & ok
        vis[:, :, :npool * P] = keep.repeat_interleave(P, -1)
    if self.dsa_tail != 0:                                                   # the incomplete visible pool
        start = (tt + 1) - (tt + 1) % P
        vis |= ((tt[None, :] >= start[:, None]) & (tt[None, :] <= tt[:, None]))[None]
    return vis


UAttention._dsa_mask = _dsa_mask


def _pool_kv(kv, g, pb, m, ovl, nw, eps, inv, raw=False):
    """Compressed pools of m tokens (DeepSeek-V4's compressors): kv, g (B, T, 2n) -> (B, T // m, n). Pool w is
    sum_j softmax_j(g_j + pb_j) * kv_j per channel over its own tokens' second halves and, if ovl, the previous
    pool's first halves (none for w = 0); RMS-normed (weight nw) and rotated (host pairs (i, i + n/2), inv) at
    position w * m."""
    B, T, n2 = kv.shape
    n, W = n2 // 2, T // m
    if W == 0:
        return kv.new_zeros(B, 0, n)
    kv, g = kv[:, :W * m].view(B, W, m, n2), g[:, :W * m].view(B, W, m, n2) + pb[:m]
    ck, cg = kv[..., n:], g[..., n:]
    if ovl:
        pk = torch.cat((kv.new_zeros(B, 1, m, n), kv[:, :-1, :, :n]), 1)
        pg = torch.cat((g.new_full((B, 1, m, n), float("-inf")), g[:, :-1, :, :n]), 1)
        ck, cg = torch.cat((pk, ck), 2), torch.cat((pg, cg), 2)
    c = (ck * cg.softmax(2)).sum(2)
    c = c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + eps) * nw
    rc = _rot_pos(c, inv, m)
    return (rc, c) if raw else rc


def _rot_pos(c, inv, m):
    """Rotate pool w's vector (host pairs (i, i + n/2)) at position w * m."""
    W = c.shape[1]
    ang = (torch.arange(W, dtype=F64) * m)[:, None] * inv[None]
    cos, sin = torch.cat((ang.cos(), ang.cos()), -1), torch.cat((ang.sin(), ang.sin()), -1)
    return c * cos + _rotate_half(c) * sin


def _candidate_blocks(logits, lens, topk_blocks, bs):
    """DeepSeek-V4.1 select_candidate_blocks: per query keep the top_blocks blocks of bs pools by their best score,
    the block holding the query's newest visible pool pinned in. logits (B, T, W) (-inf = unreachable), lens (T,)."""
    W = logits.shape[-1]
    sc = F.pad(logits, (0, -W % bs), value=float("-inf")).unflatten(-1, (-1, bs)).amax(-1)
    nb = sc.shape[-1]
    last = (lens - 1).div(bs, rounding_mode="floor")
    sc = sc.masked_fill(torch.arange(nb)[None, :] == last[:, None], float("inf"))
    top = sc.topk(min(topk_blocks, nb), -1)
    keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(bs, -1)[..., :W]


def _cmp_pools(self, x, reg=None):
    """-> (pool keys = values (B, W, hd), visibility (B, T, W)) of the compressed-KV branch."""
    mm, (rate, ovl, eps, topk, ieps, s0, s1) = self.cmp, self.cmp.cfg
    kvm, ixm, pkm, cdm = (int(v) for v in mm.share)
    m = int(rate)
    B, T, _ = x.shape
    lat = None
    if kvm == 2:                                                       # a later layer of a V4.1 source's group
        kc = reg["cmp_kc"]
    else:
        kc, lat = _pool_kv(mm.kv(x), mm.gate(x), mm.pb, m, ovl != 0, mm.nw, eps, self.inv_freq, raw=True)
        if kvm == 1:
            reg["cmp_kc"] = kc
    W = kc.shape[1]
    lens = (torch.arange(T) + 1) // m
    valid = (torch.arange(W)[None, :] < lens[:, None]).expand(B, T, W)
    if pkm == 2:                                                       # reuse the last index source's picks
        return kc, reg["cmp_vis"]
    if not torch.isfinite(topk) or W == 0:
        if pkm == 1:
            reg["cmp_vis"] = valid
        return kc, valid
    nI = mm.ix_w.out_features
    if ixm == 1:                                                       # V4.1: index keys from the latent
        ik = mm.ix_wk(lat)
        ik = _rot_pos(ik * torch.rsqrt(ik.pow(2).mean(-1, keepdim=True) + ieps) * mm.ix_nw, mm.ix_inv, m)
        reg["cmp_ik"] = ik
    elif ixm == 2:
        ik = reg["cmp_ik"]
    else:
        ik = _pool_kv(mm.ix_kv(x), mm.ix_gate(x), mm.ix_pb, m, True, mm.ix_nw, ieps, mm.ix_inv)
    qi = mm.ix_q(x).view(B, T, nI, -1)
    ang = torch.arange(T, dtype=F64)[:, None] * mm.ix_inv[None]
    cos, sin = torch.cat((ang.cos(), ang.cos()), -1)[None, :, None], torch.cat((ang.sin(), ang.sin()), -1)[None, :, None]
    qi = qi * cos + _rotate_half(qi) * sin
    sc = F.relu(torch.einsum("bthd,bwd->bthw", qi, ik)) * s0
    score = torch.einsum("bth,bthw->btw", mm.ix_w(x) * s1, sc).masked_fill(~valid, float("-inf"))
    if cdm == 1:
        reg["cmp_cand"] = _candidate_blocks(score, lens, int(mm.cand[0]), int(mm.cand[1]))
    elif cdm == 2:
        score = score.masked_fill(~reg["cmp_cand"], float("-inf"))
    sel = score.topk(min(int(topk), W), -1).indices
    vis = torch.zeros(B, T, W, dtype=torch.bool).scatter_(-1, sel, True) & valid
    if pkm == 1:
        reg["cmp_vis"] = vis
    return kc, vis


UAttention._cmp_pools = _cmp_pools


class UHC(nn.Module):
    """Manifold-constrained hyper-connections around one sub-block: from the N streams X (RMS-normed
    flat) a linear map gives pre (sigmoid), post (2 sigmoid) and comb (softmax, then Sinkhorn to doubly
    stochastic); the block reads sum_i pre_i X_i and the streams become post * out + comb^T X."""

    def __init__(self, c: UConfig):
        super().__init__()
        N, d = c.hc_mult, c.d
        self.N, self.iters = N, c.hc_iters
        self.fn, self.base, self.scale = _p((2 + N) * N, N * d), _p((2 + N) * N), _p(3, fill=1.0)
        self.register_buffer("eps", torch.tensor([1e-6, 1e-6], dtype=F64))     # [hc_eps, input-norm eps]
        self.register_buffer("post_c", torch.tensor(2.0, dtype=F64))           # post = post_c sigmoid(.)
        # Motif-3 variant (sk = 1): pre / post logits clamped to +-lim[0]; comb = Sinkhorn of exp(clamp(logits,
        # +-lim[1])), iters x (row, column) normalisation with sums clamped >= 1e-8; read transposed
        # (the streams become post * out + H_res X)
        self.register_buffer("sk", torch.tensor(0.0, dtype=F64))
        self.register_buffer("lim", torch.tensor([10.0, 20.0], dtype=F64))
        # lag = 1 (DeepSeek-V4.1): the block reads the streams with the pre computed by the PREVIOUS sub-block (from
        # the state one step earlier: reg["hc_pre"], one-hot on stream 0 before the first) and publishes its own
        self.register_buffer("lag", torch.tensor(0.0, dtype=F64))

    def forward(self, X, reg=None):
        B, T, N, d = X.shape
        flat = X.reshape(B, T, N * d)
        flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps[1])
        w = flat @ self.fn.T
        pw, qw, cw = w.split([N, N, N * N], -1)
        pb, qb, cb = self.base.split([N, N, N * N])
        if self.sk != 0:
            l0, l1 = self.lim
            pre = torch.sigmoid((pw * self.scale[0] + pb).clamp(-l0, l0))
            post = self.post_c * torch.sigmoid((qw * self.scale[1] + qb).clamp(-l0, l0))
            m = (cw * self.scale[2] + cb).view(B, T, N, N).clamp(-l1, l1).exp()
            for _ in range(self.iters):
                m = m / m.sum(-1, keepdim=True).clamp(min=1e-8)
                m = m / m.sum(-2, keepdim=True).clamp(min=1e-8)
            return post, m.transpose(-1, -2), (pre[..., None] * X).sum(2)
        e = self.eps[0]
        pre = torch.sigmoid(pw * self.scale[0] + pb) + e
        post = self.post_c * torch.sigmoid(qw * self.scale[1] + qb)
        comb = torch.softmax((cw * self.scale[2] + cb).view(B, T, N, N), -1) + e
        comb = comb / (comb.sum(-2, keepdim=True) + e)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + e)
            comb = comb / (comb.sum(-2, keepdim=True) + e)
        if self.lag != 0:
            prev = reg.get("hc_pre")
            if prev is None:
                prev = torch.zeros_like(pre)
                prev[..., 0] = 1.0
            reg["hc_pre"] = pre
            pre = prev
        return post, comb, (pre[..., None] * X).sum(2)


def _rot_pairs(t, theta):
    """Rotate key-dim pairs (j, j + dk/2) by theta (B,H,dk/2)."""
    h = t.shape[-1] // 2
    c, s_ = theta.cos(), theta.sin()
    t0, t1 = t[..., :h], t[..., h:2 * h]
    return torch.cat((t0 * c - t1 * s_, t0 * s_ + t1 * c, t[..., 2 * h:]), -1)


def _recurrence(q, k, v, a, gamma, w, w_prev, dth, rm=None, pre=None, er=None):
    """gd_recurrence plus: a previous-token write added before the decay (Mamba-3's trapezoid),
    q/k rotated by a cumulative angle (Mamba-3, RoPE), and RWKV-7's generalized removal
    S -= c (S_{t-1} kap)(kap * beta)^T read from the PRE-decay state (rm = (c, kap, rmv)), and a
    per-channel erase gate on the delta term's READ key (er: S -= gamma (S (er * k)) k^T, GDN-2)."""
    B, H, T, dk = k.shape
    S = torch.zeros(B, H, v.shape[-1], dk, dtype=q.dtype)
    theta = torch.zeros(B, H, dk // 2, dtype=q.dtype)
    kp = torch.zeros(B, H, dk, dtype=q.dtype); vp = torch.zeros(B, H, v.shape[-1], dtype=q.dtype)
    out = []
    for t in range(T):
        qt, kt = q[:, :, t], k[:, :, t]
        S0 = S
        if dth is not None:
            theta = theta + dth[:, :, t]
            qt, kt = _rot_pairs(qt, theta), _rot_pairs(kt, theta)
        if w_prev is not None:
            S = S + w_prev[:, :, t, None, None] * vp[..., None] * kp[:, :, None, :]
        if rm is not None:
            Skap = (S * rm[1][:, :, t, None, :]).sum(-1)              # pre-decay S kap
        S = S * a[:, :, t, None, :]
        if rm is not None:
            S = S - rm[0][None, :, None, None] * Skap[..., None] * rm[2][:, :, t, None, :]
        Sk = (S * (kt if er is None else kt * er[:, :, t])[:, :, None, :]).sum(-1)
        S = S - gamma[:, :, t, None, None] * Sk[..., None] * kt[:, :, None, :]
        S = S + w[:, :, t, None, None] * v[:, :, t, :, None] * kt[:, :, None, :]
        yt = (S * qt[:, :, None, :]).sum(-1)
        if pre is not None:                                            # RWKV-6 reads S_{t-1}
            yt = yt + pre[None, :, None] * ((S0 * qt[:, :, None, :]).sum(-1) - yt)
        out.append(yt)
        kp, vp = kt, v[:, :, t]
    return torch.stack(out, 2)


def _ridge_readout(q, k, a, ratio, n_iter):
    """Gated KalmaNet's readout: the key-covariance state H_t = Diag(a_t) H_{t-1} + k_t k_t^T (the
    gated-delta state with v = k, write 1, no delta term) and a fixed n_iter-step Chebyshev iteration
    for (H_t + lam_t I) x = q_t, lam_t = ratio ||H_t||_F (spectrum bound ||H_t||_F):
        step = 2 / (2 lam + |H|_F),  rho = (|H|_F / (2 lam + |H|_F) / 2)^2,  x_0 = step q,  x_-1 = 0,  w = 2
        w <- 1 / (1 - rho w);  x <- w x - (step w (x^T H + lam x - q) + (w - 1) x_prev)
    q, k, a: (B,H,T,dk) -> x: (B,H,T,dk)."""
    B, Hn, T, dk = k.shape
    S = torch.zeros(B, Hn, dk, dk, dtype=q.dtype)
    lam_r = ratio[None, :, None]
    out = []
    for t in range(T):
        kt, qt = k[:, :, t], q[:, :, t]
        S = S * a[:, :, t, None, :] + kt[..., :, None] * kt[..., None, :]
        fro = S.pow(2).sum((-2, -1)).sqrt()[..., None]                   # (B,H,1)
        lam = lam_r * fro
        step = 2 / (2 * lam + fro)
        rho = (fro / (2 * lam + fro) / 2) ** 2
        x_prev, x, w = torch.zeros_like(qt), step * qt, 2.0
        for _ in range(n_iter):
            w = 1 / (1 - rho * w)
            g = step * w * ((x[..., :, None] * S).sum(-2) + lam * x - qt) + (w - 1) * x_prev
            x_prev, x = x, w * x - g
        out.append(x)
    return torch.stack(out, 2)


class USSM(nn.Module):
    def __init__(self, c: UConfig):
        super().__init__()
        D, K, H, dk, dv, d = c.ssm_in, c.conv_k, c.ssm_heads, c.dk, c.dv, c.d
        self.D, self.K, self.H, self.dk, self.dv = D, K, H, dk, dv
        C = H * dv
        self.in_proj = _lin(d, D)
        self.conv_w, self.conv_b = _p(D, K), _p(D)
        self.act_silu, self.act_lin = _p(D), _p(D)       # c' = act_silu*silu(c) + act_lin*c
        Dn = c.ssm_norm_feats
        self.Dn = Dn
        self.pn = _lin(2 * D, max(Dn, 1))                # normalized knob features nb
        self.n_avg, self.n_w = _p(max(Dn, 1), max(Dn, 1)), _p(max(Dn, 1), fill=1.0)
        self.register_buffer("n_eps", torch.tensor(1e-6, dtype=F64))
        Dh, Dp, Dh2 = c.ssm_sig_feats, c.ssm_prod_feats, c.ssm_sig2_feats
        self.Dh, self.Dp, self.Dh2 = Dh, Dp, Dh2
        R0 = 2 * D + Dn
        self.ph = _lin(R0, max(Dh, 1))                   # stage 1: h = sigmoid(ph r)
        self.pfx, self.pfy = _lin(R0 + Dh, max(Dp, 1)), _lin(R0 + Dh, max(Dp, 1))   # stage 2: f
        self.ph2 = _lin(R0 + Dh + Dp, max(Dh2, 1))       # stage 3: h2 = sigmoid(ph2 r)
        R = R0 + Dh + Dp + Dh2                           # knobs read [u, c', nb, h, f, h2]
        self.pq, self.pk = _lin(R, H * dk), _lin(R, H * dk)
        self.pv, self.pz = _lin(R, C), _lin(R, C)
        self.q_coef, self.k_coef = _p(2), _p(2)          # [plain, l2-normed]
        with torch.no_grad():
            self.q_coef[0] = self.k_coef[0] = 1.0
        # per-channel gains on the l2-normed q / k (a per-head RMSNorm weight applied BEFORE the q / k rotation:
        # MiniCPM-SALA lightning; 1 = off)
        self.q_nw, self.k_nw = _p(H, dk, fill=1.0), _p(H, dk, fill=1.0)
        self.register_buffer("l2_eps", torch.tensor(1e-6, dtype=F64))
        self.p1, self.p2 = _lin(R, H * dk), _lin(R, H * dk)
        self.A1, self.A2, self.A3 = _p(H, dk), _p(H, dk), _p(H, dk)
        self.mem = _p(H, dk, fill=1.0)
        self.pb, self.pe, self.pi = _lin(R, H), _lin(R, H), _lin(R, H)
        self.pA, self.A4, self.a_floor = _lin(R, H), _p(H, dk), _p(H)   # Mamba-3 decay term
        self.p5, self.A5 = _lin(R, H * dk), _p(H, dk)   # log a -= A5 exp(p5)  (RWKV-6)
        self.c_pre = _p(H)                               # read S_{t-1} q (before this step's update)
        self.cg = _p(H)
        self.cg0 = _p(H)                                 # constant delta strength: gamma = cg sigmoid(pb) + cg0
        self.c_er, self.pbv = _p(H), _lin(R, H * dk)     # erase gate on the read key: er = sigmoid(pbv r) (GDN-2)
        self.cw = _p(6, H)                               # softplus / sigmoid / exp / RG-LRU / const / dt*sigmoid
        self.cw_prev = _p(H)                             # previous-token write dt*(1-sigmoid(pb))
        self.pang, self.rot_c = _lin(R, H * max(dk // 2, 1)), _p(H, dk // 2)  # cumulative q/k rotation
        self.rot_k = _p(H, dk // 2)                      # constant angle step per pair (= RoPE)
        # RWKV-7 generalized delta: removal reads the PRE-decay state along kap (l2-normed) and
        # writes out along kap * beta; the write key is modulated by beta.
        self.pkap, self.pbeta = _lin(R, H * dk), _lin(R, H * dk)
        self.c_rm, self.c_ka = _p(H), _p(H, dk)
        self.register_buffer("kap_eps", torch.tensor(1e-24, dtype=F64))
        # value register (RWKV-7 v_first): write v, or read  v <- v + read * gate * (reg - v)
        self.pvr = _lin(R, H * dv)
        self.v_write, self.v_read = _p(1), _p(1)
        self.y_write, self.y_read = _p(1), _p(1)       # SSM output register (SambaY GMU memory)
        # dynamic conv on v (Jet-Nemotron): v <- act(sum_j ker_t[c, j] v_{t-W+1+j}[c]), ker = w2 silu(w1 x) + b2
        Wd = c.ssm_dconv
        self.dcW = Wd
        self.dc_w1 = _lin(d, max(c.ssm_dconv_rank, 1), bias=False)
        self.dc_w2 = _lin(max(c.ssm_dconv_rank, 1), C * max(Wd, 1))
        self.dc_act = _p(1)                              # 1 = silu after the dynamic conv
        self.reset = _p(H)                               # RG-LRU: first token writes sigmoid(pi)
        # ridge readout (Gated KalmaNet): a second, key-covariance state H_t = Diag(a_t) H_{t-1} + k_t k_t^T
        # and a FIXED `ridge_iters`-step Chebyshev solve x_t ~ (H_t + lam_t I)^-1 q_t, lam_t = ridge_ratio ||H_t||_F;
        # the main state then reads q <- q + ridge_c * (pal r) * (x - q)
        self.ridge_c, self.ridge_ratio, self.pal = _p(H), _p(H), _lin(R, H)
        self.register_buffer("ridge_iters", torch.tensor(0))
        self.register_buffer("l2_eps_out", torch.tensor(0.0, dtype=F64))   # != 0: l2 is t / (|t| + eps_out)
        self.Dskip = _p(H, dv)
        self.gate = _p(5, C)                             # silu / sigmoid / gelu-tanh / identity / const
        self.avg = _p(C, C)                              # RMS grouping as a weight (block averages)
        self.norm_w, self.norm_b = _p(C, fill=1.0), _p(C)
        self.mix = nn.Parameter(torch.eye(C, dtype=F64))   # channel mix of y before the output branches
        self.register_buffer("norm_eps", torch.tensor(1e-6, dtype=F64))
        self.out_coef = _p(4, C)                         # norm-then-gate / gate-then-norm / plain / ratio
        self.register_buffer("dt_min", torch.tensor(0.0, dtype=F64))   # dt = max(softplus(p1), dt_min)
        self.ratio_num, self.ratio_den = _p(C, C), _p(C, C)
        self.out_proj = _lin(C, d)

    def _norm(self, t):
        return t * torch.rsqrt(t.pow(2) @ self.avg + self.norm_eps) * self.norm_w + self.norm_b

    def forward(self, x, reg=None):
        B, T, _ = x.shape
        H, dk, dv, K = self.H, self.dk, self.dv, self.K
        u = self.in_proj(x)                                               # (B,T,D)
        up = F.pad(u.transpose(1, 2), (K - 1, 0))                        # causal depthwise conv
        cv = sum(self.conv_w[:, j, None] * up[:, :, j:j + T] for j in range(K)) + self.conv_b[:, None]
        cv = cv.transpose(1, 2)
        cv = self.act_silu * F.silu(cv) + self.act_lin * cv
        r = torch.cat((u, cv), -1)
        if self.Dn:
            n = self.pn(r)
            r = torch.cat((r, n * torch.rsqrt(n.pow(2) @ self.n_avg + self.n_eps) * self.n_w), -1)
        if self.Dh:
            r = torch.cat((r, torch.sigmoid(self.ph(r))), -1)
        if self.Dp:
            r = torch.cat((r, self.pfx(r) * self.pfy(r)), -1)
        if self.Dh2:
            r = torch.cat((r, torch.sigmoid(self.ph2(r))), -1)
        heads = lambda t, n: t.view(B, T, H, n).transpose(1, 2)          # (B,H,T,n)
        q, k = heads(self.pq(r), dk), heads(self.pk(r), dk)
        l2 = lambda t: t * torch.rsqrt(t.pow(2).sum(-1, keepdim=True) + self.l2_eps)
        if self.l2_eps_out != 0:
            l2 = lambda t: t / ((t.pow(2).sum(-1, keepdim=True) + self.l2_eps).sqrt() + self.l2_eps_out)
        q = self.q_coef[0] * q + self.q_coef[1] * l2(q) * self.q_nw[:, None]
        k = self.k_coef[0] * k + self.k_coef[1] * l2(k) * self.k_nw[:, None]
        v = heads(self.pv(r), dv)
        if self.dcW:
            Wd = self.dcW
            ker = self.dc_w2(F.silu(self.dc_w1(x))).view(B, T, H * dv, Wd)
            vr = F.pad(v.transpose(1, 2).reshape(B, T, H * dv).transpose(1, 2), (Wd - 1, 0)).unfold(2, Wd, 1)
            vr = (vr.permute(0, 2, 1, 3) * ker).sum(-1)
            vr = vr + self.dc_act * (F.silu(vr) - vr)
            v = heads(vr, dv)
        if reg is not None and self.v_write != 0:
            reg["v_ssm"] = v
        if reg is not None and self.v_read != 0:
            v = v + self.v_read * heads(self.pvr(r), dv) * (reg["v_ssm"] - v)
        kap = rmv = None
        if self.c_rm.any() or self.c_ka.any():
            beta = heads(self.pbeta(r), dk)
            kap = heads(self.pkap(r), dk)
            kap = kap * torch.rsqrt(kap.pow(2).sum(-1, keepdim=True) + self.kap_eps)
            rmv = kap * beta
            k = k * (1 + self.c_ka[:, None] * (beta - 1))
        p1 = heads(self.p1(r), dk)
        sp = F.softplus(p1)
        if self.dt_min > 0:                                              # Mamba-2 time_step_limit floor
            sp = sp.clamp(min=self.dt_min)
        loga = -(self.A1[:, None] * sp
                 + self.A2[:, None] * torch.sigmoid(heads(self.p2(r), dk)) + self.A3[:, None])
        per_head = lambda p: p(r).transpose(1, 2)                        # (B,H,T)
        dt = sp[..., 0]                                                  # (B,H,T)
        if self.A4.any():
            pa = per_head(self.pA)
            ht = pa.clamp(min=0) + 1 / (1 - pa.clamp(max=0))
            loga = loga - self.A4[:, None] * (dt * torch.maximum(ht, self.a_floor[:, None]))[..., None]
        if self.A5.any():
            loga = loga - self.A5[:, None] * heads(self.p5(r), dk).exp()
        a = self.mem[:, None] * loga.exp()                                # (B,H,T,dk)
        sb, si = torch.sigmoid(per_head(self.pb)), torch.sigmoid(per_head(self.pi))
        gamma = self.cg[:, None] * sb + self.cg0[:, None]
        er = (1 + self.c_er[:, None, None] * (torch.sigmoid(heads(self.pbv(r), dk)) - 1)) if self.c_er.any() else None
        lru = torch.sqrt((1 - a[..., 0] ** 2).clamp_min(0))
        first = torch.zeros(T, dtype=F64); first[0] = 1
        lru = lru + self.reset[:, None] * first * (1 - lru)
        cw = lambda i: self.cw[i][:, None]
        w = (cw(0) * dt + cw(1) * sb + cw(2) * per_head(self.pe).exp()
             + cw(3) * lru * si + cw(4) + cw(5) * dt * sb)
        w_prev = self.cw_prev[:, None] * dt * (1 - sb) if self.cw_prev.any() else None
        dth = None
        if self.rot_c.any() or self.rot_k.any():
            dth = self.rot_c[:, None] * torch.tanh(heads(self.pang(r), dk // 2)) * dt[..., None] + self.rot_k[:, None]
        if self.ridge_c.any():
            x_r = _ridge_readout(q, k, a, self.ridge_ratio, int(self.ridge_iters))
            q = q + (self.ridge_c[:, None] * per_head(self.pal))[..., None] * (x_r - q)
        if w_prev is None and dth is None and kap is None and er is None and not self.c_pre.any():
            y = gd_recurrence(q, k, v, a, gamma, w)
        else:
            rm = (self.c_rm, kap, rmv) if kap is not None else None
            y = _recurrence(q, k, v, a, gamma, w, w_prev, dth, rm, self.c_pre if self.c_pre.any() else None, er)
        y = y + self.Dskip[:, None] * v
        if reg is not None and self.y_read != 0:
            y = (1 - self.y_read) * y + self.y_read * reg["y_ssm"]
        if reg is not None and self.y_write != 0:
            reg["y_ssm"] = y
        y = y.transpose(1, 2).reshape(B, T, H * dv) @ self.mix
        z = self.pz(r)
        g = self.gate
        G = g[0] * F.silu(z) + g[1] * torch.sigmoid(z) + g[2] * F.gelu(z, approximate="tanh") + g[3] * z + g[4]
        oc = self.out_coef
        out = oc[0] * self._norm(y) * G + oc[1] * self._norm(y * G) + oc[2] * y * G
        if oc[3].any():
            on = oc[3] != 0                                  # channels that use the ratio branch
            den = torch.where(on, y @ self.ratio_den, torch.ones_like(y))
            out = out + oc[3] * (y @ self.ratio_num) / den * G
        return self.out_proj(out)


class UFFN(nn.Module):
    def __init__(self, c: UConfig):
        super().__init__()
        d, E, f, fs = c.d, c.experts, c.expert_ff, c.shared_ff
        self.top_k = c.top_k
        self.router = _lin(d, E)                         # a router bias (zero unless a target has one: GPT-OSS)
        self.w_gate, self.w_up, self.w_down = _p(E, f, d), _p(E, f, d), _p(E, d, f)
        self.b_gate, self.b_up, self.b_down = _p(E, f), _p(E, f), _p(E, d)   # routed-expert biases (GPT-OSS)
        # group-limited top-k (DeepSeek-V2 / V3): [groups, groups kept, group score: 0 sum of the top-2
        # selection scores, 1 their max, the value experts outside the kept groups are masked to before the
        # top-k: 0 (DeepSeek) or -inf (Nemotron-H)]
        self.register_buffer("grp", torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=F64))   # [..., masked value]
        self.renorm = _p(1)                              # 1 = renormalize the top-k weights
        self.register_buffer("renorm_eps", torch.tensor(0.0, dtype=F64))   # ... by (sum + eps)
        self.score_c = _p(1)                             # router scores: 0 softmax, 1 sigmoid
        self.register_buffer("score_sp", torch.tensor(0.0, dtype=F64))   # 1: sqrt(softplus(logits)) (DeepSeek-V4)
        # hash routing (DeepSeek-V4 hash_moe layers): a frozen token id -> top-k experts table replaces the
        # selection (the router scores still weight the picks); empty = off
        self.register_buffer("hash_tab", torch.zeros(0, c.top_k, dtype=torch.long))
        self.sel_bias = _p(E)                            # added to scores for top-k SELECTION only
        self.sel_logit = _p(1)                           # 1: select on the raw logits + sel_bias (Youtu QB)
        self.register_buffer("router_cap", torch.tensor(float("inf"), dtype=F64))   # router logit softcap (Laguna)
        self.route_scale = _p(1, fill=1.0)
        self.exp_scale = _p(E, fill=1.0)                 # the top-k weights times a per-expert scale (Gemma-4)
        self.register_buffer("exp_ident", torch.zeros(E, dtype=F64))   # 1: an identity (zero-computation) expert
        self.register_buffer("clip_shared", torch.tensor(1.0, dtype=F64))   # 0: c_clip on the routed experts only
        self.register_buffer("cap_drop", torch.tensor(0.0, dtype=F64))   # 1: per-forward expert capacity drops
        self.register_buffer("blk_route", torch.zeros(2, dtype=F64))   # [block size (0 = off), experts kept per block]
        self.register_buffer("in_gate", torch.tensor(0.0, dtype=F64))   # 1: routing weights scale the expert INPUT
        # SparseMixer top-2 (Phi-3.5-MoE, eval): each pick's weight is a softmax over the logits within a relative
        # 2 eps of that pick's maximum; smix_eps = 0: off
        self.register_buffer("smix_eps", torch.tensor(0.0, dtype=F64))
        # the shared / dense path carries biases (zero unless a target has them: GPT-2, GPT-NeoX, ...)
        self.s_gate, self.s_up, self.s_down = _lin(d, fs), _lin(d, fs), _lin(fs, d)
        self.s_router = _lin(d, 1, bias=False)
        E, r = c.lora_adapters, c.lora_rank
        self.la = nn.ParameterDict({"g": _p(E, r, d), "u": _p(E, r, d), "d": _p(E, r, fs)})
        self.lb = nn.ParameterDict({"g": _p(E, fs, r), "u": _p(E, fs, r), "d": _p(E, d, r)})
        self.c_situ = _p(1)                              # SiTU-GLU: capped gate/up (Mini-K3)
        self.register_buffer("situ_beta", torch.tensor([4.0, 25.0], dtype=F64))
        self.c_clip = _p(1)                              # hard SwiGLU clip: gate <= L, |up| <= L
        self.c_relu2 = _p(1)                             # ungated squared ReLU: relu(up x)^2
        self.c_relu = _p(1)                              # ungated ReLU: relu(up x) (OPT)
        self.up_shift = _p(1)                            # u <- u + up_shift after the clip (swigluoai's (u + 1))
        # exact GeLU / mish branches: c_gelu = 1 ungated gelu(u) (GPT-2, NeoX, ...), c_geglu = 1 gated
        # gelu(g) * u (Gemma GeGLU); gelu_tanh = 1 the tanh approximation (gelu_new / gelu_pytorch_tanh /
        # gelu_fast), 0 the erf GeLU; c_mish = 1 gated mish(g) * u
        self.c_gelu, self.c_geglu, self.gelu_tanh, self.c_mish = _p(1), _p(1), _p(1), _p(1)
        self.register_buffer("clip_lim", torch.full((2,), float("inf"), dtype=F64))   # [gate, up] limits
        # a clamp AFTER the activation (Ling-3.1): min(silu(g), L) * clamp(u, -L, L), L per path [routed, shared]
        self.register_buffer("clip_post", torch.full((2,), float("inf"), dtype=F64))
        # ungated xIELU (Apertus): [alpha_p, alpha_n] pre-softplus (learned), [beta, eps]
        self.c_xielu, self.xielu_a = _p(1), _p(2)
        self.register_buffer("xielu_be", torch.tensor([0.5, -1e-6], dtype=F64))
        # PolyNorm-GLU (Motif-3): (w0 n(g^3) + w1 n(g^2) + w2 n(g) + b) * u, n = RMS over the width;
        # weights per routed expert / for the shared path, sigmoid(w) if poly_sig
        self.c_poly, self.poly_sig = _p(1), _p(1)
        self.poly_w, self.poly_b = _p(c.experts, 3), _p(c.experts, 1)
        self.s_poly_w, self.s_poly_b = _p(3), _p(1)
        # [eps, hidden clamp, output scale, routed bias clamp, routed result clamp, shared active width]
        self.register_buffer("poly_cfg", torch.tensor([1e-6, float("inf"), 1.0, float("inf"), float("inf"),
                                                       float(fs)], dtype=F64))
        self.r_c = _p(1)                                 # post-routing: routed <- r_up(rms_grouped(routed) r_w)
        self.r_avg, self.r_w = _p(d, d), _p(d, fill=1.0)
        self.r_up = _lin(d, d, bias=False)
        self.register_buffer("r_eps", torch.tensor(1e-6, dtype=F64))
        self.s_coef = _p(2)                              # [constant 1, sigmoid(s_router x)]
        with torch.no_grad():
            self.s_coef[0] = 1.0
        # RWKV channel mix on the shared path (shift_c = 1): token-shifted inputs x_mu = prev + mu (x - prev) (prev = the
        # previous token, 0 at t = 0) for the up / gate rows (mix_k) and a receptance gate sigmoid(r_gate x_mr) (mix_r)
        self.shift_c, self.mix_k, self.mix_r = _p(1), _p(d, fill=1.0), _p(d, fill=1.0)
        self.r_gate = _lin(d, d if c.ffn_shift else 1, bias=False)

    def _xielu(self, u):
        """xIELU (HF XIELUActivation):  a_p = softplus(alpha_p), a_n = beta + softplus(alpha_n),
        u > 0: a_p u^2 + beta u;  else: a_n (expm1(min(u, eps)) - u) + beta u."""
        ap, an = F.softplus(self.xielu_a[0]), self.xielu_be[0] + F.softplus(self.xielu_a[1])
        beta, eps = self.xielu_be
        return torch.where(u > 0, ap * u * u + beta * u, (torch.expm1(torch.min(u, eps)) - u) * an + beta * u)

    def _poly(self, g, u, routed):
        """PolyNorm-GLU (Motif-3 PolyNormTorch / GroupedPolyNorm): g, u clamped to the hidden clamp,
        (w0 n(g^3) + w1 n(g^2) + w2 n(g) + b) * u with n(x) = x / sqrt(sum(x^2) / width + eps) over the
        FFN width; routed experts (own w, b per expert; b clamped) also clamp the result; then * scale.
        `width` is the shared path's active width, so a zero-padded shared bank normalises exactly."""
        eps, hc, scale, bc, rc, ws = self.poly_cfg
        g, u = g.clamp(-hc, hc), u.clamp(-hc, hc)
        if routed:                                       # h: (B, T, E, f); per-expert (E, 1) coefficients
            w, b, n = self.poly_w, self.poly_b.clamp(-bc, bc), g.shape[-1]
            w = [w[:, k, None] for k in range(3)]
        else:
            w, b, n = self.s_poly_w, self.s_poly_b, ws
        w = [wk + self.poly_sig * (torch.sigmoid(wk) - wk) for wk in w]
        nrm = lambda x: x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) / n + eps)
        y = (w[0] * nrm(g ** 3) + w[1] * nrm(g ** 2) + w[2] * nrm(g) + b) * u
        if routed:
            y = y.clamp(-rc, rc)
        return y * scale

    def _act(self, g, u, routed=False):
        """silu(g) * u, or SiTU-GLU  bg tanh(g/bg) sigmoid(g) * bu tanh(u/bu)  (c_situ = 1);
        c_clip = 1 first clips g <= L and u to [-L, L]; c_relu2 = 1 is the ungated relu(u)^2;
        c_xielu = 1 the ungated xIELU(u); c_poly = 1 PolyNorm-GLU (routed: per-expert weights);
        c_gelu = 1 the ungated gelu(u), c_geglu = 1 gelu(g) * u (erf, or tanh if gelu_tanh = 1);
        c_mish = 1 mish(g) * u."""
        if self.c_relu2 != 0:
            return F.relu(u) ** 2
        if self.c_relu != 0:
            return F.relu(u)
        if self.c_gelu != 0 or self.c_geglu != 0:
            gelu = lambda t: F.gelu(t, approximate="tanh" if self.gelu_tanh != 0 else "none")
            return gelu(u) if self.c_gelu != 0 else gelu(g) * u
        if self.c_mish != 0:
            return F.mish(g) * u
        if self.c_xielu != 0:
            return self._xielu(u)
        if self.c_poly != 0:
            return self._poly(g, u, routed)
        if self.c_clip != 0 and (routed or self.clip_shared != 0):   # (clip_shared 0: the routed experts only, HY-V4)
            Lg, Lu = self.clip_lim
            g = g + self.c_clip * (g.clamp(max=Lg) - g)
            u = u + self.c_clip * (u.clamp(-Lu, Lu) - u)
        u = u + self.up_shift
        if self.c_situ == 0:
            Lp = self.clip_post[0 if routed else 1]
            if torch.isfinite(Lp):
                return F.silu(g).clamp(max=Lp) * u.clamp(-Lp, Lp)
            return F.silu(g) * u
        c, (bg, bu) = self.c_situ, self.situ_beta
        return (((1 - c) * g + c * bg * torch.tanh(g / bg)) * torch.sigmoid(g)
                * ((1 - c) * u + c * bu * torch.tanh(u / bu)))

    def forward(self, x, reg=None, xe=None, xg=None, parts=False):
        """routed(x) + shared(x) * gate. xe / xg: separate inputs of the routed experts and of the router
        (Gemma-4 normalises each differently); parts = True returns (routed, gated shared)."""
        idx = reg.get("adapter") if reg is not None else None
        L = lambda n, t: _lora(t, self.la[n], self.lb[n], idx)
        xs = x
        x = x if xe is None else xe
        logits = self.router(x if xg is None else xg)
        if getattr(self, "router_mlp", None) is not None:   # an MLP router W2 tanh(W1 x) (LLaMA-MoE's gate_network)
            logits = self.router_mlp(x if xg is None else xg)
        if torch.isfinite(self.router_cap):
            logits = self.router_cap * torch.tanh(logits / self.router_cap)
        p = logits.softmax(-1) if self.score_c == 0 else (
            (1 - self.score_c) * logits.softmax(-1) + self.score_c * torch.sigmoid(logits))
        if self.score_sp != 0:
            p = torch.sqrt(F.softplus(logits))
        if getattr(self, "router2", None) is not None:   # two summed sigmoid routers (ERNIE-5.5 split-feature routing)
            p = torch.sigmoid(logits) + torch.sigmoid(self.router2(x if xg is None else xg))
        sel = (logits if self.sel_logit != 0 else p) + self.sel_bias
        G, kg = int(self.grp[0]), int(self.grp[1])
        if G > 1 and kg < G:                             # keep the top-kg groups' experts
            sg = sel.view(*sel.shape[:-1], G, -1)
            gs = sg.max(-1).values if self.grp[2] != 0 else sg.topk(2, -1).values.sum(-1)
            keep = torch.zeros_like(gs).scatter_(-1, gs.topk(kg, -1).indices, 1.0)
            sel = sel.masked_fill(keep.repeat_interleave(sel.shape[-1] // G, -1) == 0, float(self.grp[3]))
        if self.blk_route[0] > 0:                        # block routing (LLaDA2.2): each block of tokens keeps its
            sel = _bs.block_route_mask(sel, int(self.blk_route[0]), int(self.blk_route[1]))   # top experts
        topi = sel.topk(self.top_k, -1).indices if not self.hash_tab.numel() else self.hash_tab[reg["ids"]]
        topv = p.gather(-1, topi)
        if self.smix_eps > 0:                            # SparseMixer: argmax, then the next, each own-masked softmax
            e2, s_, picks, ws = 2 * self.smix_eps, logits, [], []
            for _ in range(2):
                m_, i_ = s_.max(-1, keepdim=True)
                fac = logits.abs().clamp(min=m_)
                ws.append(s_.masked_fill((m_ - logits) / fac > e2, float("-inf")).softmax(-1).gather(-1, i_))
                picks.append(i_)
                s_ = s_.scatter(-1, i_, float("-inf"))
            topi, topv = torch.cat(picks, -1), torch.cat(ws, -1)
        topv = (topv + self.renorm * (topv / (topv.sum(-1, keepdim=True) + self.renorm_eps) - topv)) * self.route_scale
        topv = topv * self.exp_scale[topi]
        if self.cap_drop != 0:                           # expert capacity: over-capacity assignments dropped (HunYuan)
            topv = topv * _bs.capacity_keep(topi, self.w_gate.shape[0]).to(topv.dtype)
        xe_ = self.e_pre(x) if getattr(self, "e_pre", None) is not None else x   # a normed latent (Muse-Spark)
        if self.in_gate != 0:                            # input gating (Llama-4): expert e reads s_e x, s = the
            sg = torch.zeros(*topi.shape[:-1], self.w_gate.shape[0], dtype=x.dtype).scatter(-1, topi, topv)   # weights
            h = self._act(torch.einsum("efd,btd->btef", self.w_gate, xe_) * sg[..., None] + self.b_gate,
                          torch.einsum("efd,btd->btef", self.w_up, xe_) * sg[..., None] + self.b_up, True)
            topv = torch.ones_like(topv)                 # ... and its output enters unweighted
        else:
            h = self._act(torch.einsum("efd,btd->btef", self.w_gate, xe_) + self.b_gate,
                          torch.einsum("efd,btd->btef", self.w_up, xe_) + self.b_up, True)
        eo = torch.einsum("edf,btef->bted", self.w_down, h) + self.b_down   # every expert, then select
        if self.exp_ident.any():                         # zero-computation experts: the expert's output is its input
            eo = torch.where(self.exp_ident[:, None] != 0, x[:, :, None, :].expand_as(eo), eo)   # (LongCat)
        if getattr(self, "e_post", None) is not None:    # a norm of each expert's output (Muse-Spark)
            eo = self.e_post(eo)
        routed = (eo.gather(2, topi[..., None].expand(*topi.shape, x.shape[-1])) * topv[..., None]).sum(2)
        if getattr(self, "e_out", None) is not None:     # the latent back to the hidden width (Muse-Spark)
            routed = self.e_out(routed)
        if self.r_c != 0:
            rn = routed * torch.rsqrt(routed.pow(2) @ self.r_avg + self.r_eps) * self.r_w
            routed = (1 - self.r_c) * routed + self.r_c * self.r_up(rn)
        x = xr = xs
        if self.shift_c != 0:                            # RWKV channel mix: token shift + receptance gate
            prev = F.pad(xs, (0, 0, 1, 0))[:, :-1]
            x, xr = prev + self.mix_k * (xs - prev), prev + self.mix_r * (xs - prev)
        hs = self._act(self.s_gate(x) + L("g", x), self.s_up(x) + L("u", x))
        if getattr(self, "s_hid", None) is not None:     # a module on the hidden activation (Dory's ffn_sub_norm,
            hs = self.s_hid(hs)                          # Canon-D's conv)
        shared = self.s_down(hs) + L("d", hs)
        if self.shift_c != 0:
            shared = shared * torch.sigmoid(self.r_gate(xr))
            x = xs
        sg = self.s_coef[0] + self.s_coef[1] * torch.sigmoid(self.s_router(x))
        return (routed, shared * sg) if parts else routed + shared * sg


class UEngram(nn.Module):
    """Hashed n-gram memory (Engram): the token embedding generalised from the current token to
    hashed n-grams of the input ids. Each (order n, hash head h) folds the last n ids with a prime,
    x <- x P_h + id + 1, and reads row (x ^ (x >> shift)) mod slots of its own table; the rows are
    concatenated into e. Added to the residual stream before the layer:
        g = sigmoid(<rms(x) w_h, rms(key e) w_k> / sqrt(d)),  v = g * val(e),  out = v + causal_conv(v).
    Off (c = 0) unless a target sets it; ids are the model's input ids (left-padded with `pad`)."""

    PRIMES = (1000003, 999983, 1000033, 999979, 1000037, 999961, 1000039, 999953)

    def __init__(self, c: UConfig):
        super().__init__()
        d, S, Hh, D = c.d, c.engram_slots, c.engram_heads, c.engram_dim
        self.orders, self.heads, self.slots, self.pad, self.shift = tuple(c.engram_orders), Hh, S, c.vocab, 17
        self.register_buffer("primes", torch.tensor(self.PRIMES, dtype=torch.long))
        self.table = _p(len(self.orders) * Hh * S, D)
        mem = len(self.orders) * Hh * D
        self.key, self.val = _lin(mem, d, bias=False), _lin(mem, d, bias=False)
        self.w_h, self.w_k = _p(d, fill=1.0), _p(d, fill=1.0)
        self.register_buffer("eps", torch.tensor([1e-6, 1e-6], dtype=F64))
        self.conv_w = _p(d, c.engram_conv)
        self.c = _p(1)

    def hash_ids(self, ids):
        B, T = ids.shape
        maxn = max(self.orders)
        p = F.pad(ids, (maxn - 1, 0), value=self.pad)
        out, off = [], 0
        for n in self.orders:
            grams = [p[:, maxn - 1 - j: maxn - 1 - j + T] for j in range(n)]
            for h in range(self.heads):
                P = self.primes[h % len(self.primes)]
                x = torch.zeros_like(ids)
                for g in grams:
                    x = x * P + g + 1
                out.append(torch.remainder(x ^ (x >> self.shift), self.slots) + off)
                off += self.slots
        return torch.stack(out, -1)

    def forward(self, x, ids):
        T = x.shape[1]
        e = self.table[self.hash_ids(ids)].flatten(-2)
        k, v = self.key(e), self.val(e)
        g = torch.sigmoid((_rms(x, self.eps[0]) * self.w_h * _rms(k, self.eps[1]) * self.w_k).sum(-1, keepdim=True)
                          / math.sqrt(x.shape[-1]))
        v = g * v
        K = self.conv_w.shape[1]
        vp = F.pad(v.transpose(1, 2), (K - 1, 0))
        y = sum(self.conv_w[:, j, None] * vp[:, :, j:j + T] for j in range(K)).transpose(1, 2)
        return self.c * (v + y)


class UEngramHC(nn.Module):
    """Engram on the hc streams (DeepSeek-V4.1 Engram + NgramHashState): the ids mapped through a token table,
    each position's n-grams (2 .. max_n, look-back padded with pad) hashed by XOR-ing the token x odd multiplier
    products, mod a prime per (n-gram size, head), plus that pair's row offset; the rows' concatenation e gives a
    key per stream and one value, [k_1 .. k_N, v] = wkv e, and each stream X_i gets gate_i v,
    gate_i = sigmoid(signed_sqrt(max(|dot|, 1e-6))), dot = <X_i, w_i k_i> rms(X_i)^-1 rms(k_i)^-1 / sqrt(d)."""

    def __init__(self, c: UConfig):
        super().__init__()
        N, d, cols, D, n = c.hc_mult, c.d, c.engram_hc_cols, c.engram_hc_dim, c.engram_hc_n
        self.N, self.d = N, d
        self.table = _p(c.engram_hc, D)
        self.wkv = _lin(cols * D, d * (N + 1), bias=False)
        self.w = _p(N, d, fill=1.0)                      # q_weight * k_weight
        self.register_buffer("token_map", torch.arange(c.vocab))
        self.register_buffer("mult", torch.ones(n, dtype=torch.long))
        self.register_buffer("primes", torch.ones(n - 1, cols // max(n - 1, 1), dtype=torch.long))
        self.register_buffer("offsets", torch.zeros(cols, dtype=torch.long))
        self.register_buffer("pad", torch.tensor(0))
        self.register_buffer("eps", torch.tensor(1e-6, dtype=F64))

    def hash_ids(self, ids):
        B, T = ids.shape
        tok = self.token_map[ids]
        pos = torch.arange(T)
        prod = []
        for sh in range(self.mult.numel()):
            src = tok.gather(1, (pos - sh).clamp(min=0).expand(B, T))
            prod.append(torch.where((pos < sh)[None], self.pad, src) * self.mult[sh])
        roll, out = prod[0], []
        for i in range(1, len(prod)):
            roll = torch.bitwise_xor(roll, prod[i])
            out.append(roll[..., None] % self.primes[i - 1])
        return torch.cat(out, -1) + self.offsets

    def forward(self, X, ids):
        N, d = self.N, self.d
        kv = self.wkv(self.table[self.hash_ids(ids)].flatten(-2))
        k, v = kv.split([N * d, d], -1)
        k = k.unflatten(-1, (N, d))
        rstd = torch.rsqrt(X.pow(2).mean(-1) + self.eps) * torch.rsqrt(k.pow(2).mean(-1) + self.eps)
        dot = (X * self.w * k).sum(-1) * rstd * d ** -0.5
        g = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return X + g[..., None] * v[..., None, :]


class ULayer(nn.Module):
    def __init__(self, c: UConfig):
        super().__init__()
        self.engram = UEngram(c) if c.engram_slots else None
        self.engram_hc = _bs.engram_block(c)                         # hc-stream Engram / n-gram PLE, on X before the layer
        self.mem = UAttention(c, mem=True) if c.mem_attn else None     # parallel memory read, own norm
        self.norm_m = UNorm(c.d) if c.mem_attn else None
        self.sn_in = UNorm(c.d) if c.stream_norms else None      # x <- sn_in(x) at the start (sn_c[0])
        self.sn_out = UNorm(c.d) if c.stream_norms else None     # x <- sn_out(x) at the end (sn_c[1])
        self.sn_c = _p(2)
        self.hc_a = _bs.hc_block(c) if c.hc_mult else None          # hyper-connections (attn+SSM / FFN)
        self.hc_f = _bs.hc_block(c) if c.hc_mult else None
        N = lambda: UNorm(c.d, c.norm_gate_rank)
        self.norm_a, self.norm_s, self.norm_f = N(), N(), N()
        self.attn, self.ssm, self.ffn = UAttention(c), USSM(c), UFFN(c)
        self.post_a, self.post_s, self.post_f = N(), N(), N()
        self.post_coef = _p(3, 2)                        # per sub-block [raw, post-normed] output
        # split FFN (Gemma-4): the shared path reads norm_f(x) -> post_f1, the routed experts norm_f2(x) ->
        # post_f2, the router norm_r(x); their sum goes through post_f (post_coef[2]). ffn_split = 0: off
        self.ffn_split = _p(1)
        self.norm_f2, self.norm_r, self.post_f1, self.post_f2 = UNorm(c.d), UNorm(c.d), UNorm(c.d), UNorm(c.d)
        # per-layer embedding input (Gemma-4 PLE): x <- x + ple_post(ple_proj(gelu(ple_gate x) * pli_l))
        dp = max(c.ple_dim, 1)
        self.ple_c, self.ple_gate, self.ple_proj = _p(1), _lin(c.d, dp, bias=False), _lin(dp, c.d, bias=False)
        self.ple_post = UNorm(c.d)
        self.register_buffer("ple_tanh", torch.tensor(1.0, dtype=F64))   # its GeLU: 1 tanh-approx, 0 erf
        self.register_buffer("out_scale", torch.tensor(1.0, dtype=F64))  # x <- out_scale x at the end (layer_scalar)
        self.register_buffer("res_c", torch.ones(2, dtype=F64))  # residual coefficients x <- c x + f(x) (MiniMax alpha)
        self.idx = 0
        self.res_write, self.res_mix = _p(1), _p(1)      # residual snapshot: write x; x <- x + mix (snap - x)
        self.seq = _p(1, fill=1.0)                       # 1: SSM reads x + attn (sequential); 0: parallel
        self.par = _p(1)                                 # 1: FFN reads the LAYER INPUT (a parallel block,
        #                                                  GPT-J / NeoX / Cohere: x + attn(n x) + ffn(n' x))
        d = c.d                                         # Block AttnRes queries / key-norm weights
        self.ar_qa, self.ar_wa, self.ar_qf, self.ar_wf = _p(d), _p(d, fill=1.0), _p(d), _p(d, fill=1.0)
        self.register_buffer("ar_eps", torch.tensor(1e-6, dtype=F64))
        with torch.no_grad():
            self.post_coef[:, 0] = 1.0
        _bs.init_layer(self, c)                          # the stream / input / FFN / loop branches (off by default)

    def _out(self, i, o, post):
        pc = self.post_coef[i]
        return pc[0] * o + (pc[1] * post(o) if pc[1] != 0 else 0)

    def mix(self, sources, q, w):
        """Depth-softmax over residual sources: sum_s softmax_s(q . rms(src_s) * w) src_s."""
        if self.ar_sw > 0 and len(sources) > self.ar_sw + 1:   # a sliding window over the sources (ARLoop): the
            sources = [sources[0]] + sources[-int(self.ar_sw):]  # embedding + the last ar_sw
        st = torch.stack(sources)
        sc = (_rms(st, self.ar_eps) * w) @ q
        return (st * sc.softmax(0)[..., None]).sum(0)

    def forward_attnres(self, blocks, partial, causal, reg=None):
        """Block AttnRes: no running residual; inputs are depth-softmax mixes of the block sums."""
        xa = self.mix(blocks + ([partial] if partial is not None else []), self.ar_qa, self.ar_wa)
        a = self._out(0, self.attn(self.norm_a(xa), causal, reg), self.post_a)
        s = self._out(1, self.ssm(self.norm_s(xa), reg), self.post_s)
        partial = a + s if partial is None else partial + a + s
        xf = self.mix(blocks + [partial], self.ar_qf, self.ar_wf)
        return partial + self._out(2, self.ffn(self.norm_f(xf), reg), self.post_f)

    def forward_fullres(self, sources, causal, reg=None):
        """Full AttnRes: each sub-block reads a depth-softmax mix over the embedding and every earlier
        sub-block output, and appends its own output as a new source."""
        xa = self.mix(sources, self.ar_qa, self.ar_wa)
        a = self._out(0, self.attn(self.norm_a(xa), causal, reg), self.post_a)
        s = self._out(1, self.ssm(self.norm_s(xa + self.seq * a), reg), self.post_s)
        sources.append(a + s)
        sources.append(self._out(2, self.ffn(self.norm_f(self.mix(sources, self.ar_qf, self.ar_wf)), reg), self.post_f))

    def forward_hc(self, X, causal, reg=None):
        """N residual streams (mHC): each sub-block reads a pre-weighted collapse of the streams and
        writes post * out + comb^T X."""
        if self.engram_hc is not None:
            X = self.engram_hc(X, reg["ids"])
        post, comb, h = self.hc_a(X, reg)
        a = self._out(0, self.attn(self.norm_a(h), causal, reg), self.post_a)
        s = self._out(1, self.ssm(self.norm_s(h + self.seq * a), reg), self.post_s)
        X = post[..., None] * (a + s)[..., None, :] + comb.transpose(-1, -2) @ X
        post, comb, h = self.hc_f(X, reg)
        f = self._out(2, self.ffn(self.norm_f(h), reg), self.post_f)
        return post[..., None] * f[..., None, :] + comb.transpose(-1, -2) @ X

    def forward(self, x, causal, reg=None):
        for hk in self.xhooks:                           # stream ops at the layer start (branches_stream: the x0 mix,
            x = hk.enter(x, reg)                         # the two-state loops, the loop input re-injection)
        if self.engram is not None and self.engram.c != 0:
            x = x + self.engram(x, reg["ids"])
        if self.res_mix != 0:                            # x_snap + c (x - x_snap), mix = 1 - c (LFM3-RD)
            x = x + self.res_mix * (reg["x_snap"] - x)
        if self.res_write != 0:
            reg["x_snap"] = x
        if self.sn_in is not None and self.sn_c[0] != 0:
            x = self.sn_in(x)
        a = self._out(0, self.attn(self.norm_a(x), causal, reg), self.post_a)
        xs_ = x + self.seq * a
        if self.cat is not None:                         # a concat(x, x0) block feeding the SSM input only (Zamba)
            xs_ = xs_ + self.cat(x, reg, causal)
        s = self._out(1, self.ssm(self.norm_s(xs_), reg), self.post_s)
        m = self.mem(self.norm_m(x), causal, reg) if self.mem is not None else 0
        x_in = x
        if self.rgain is None:
            x = (x if self.res_c[0] == 1 else self.res_c[0] * x) + a + s + m
        else:                                            # per-channel residual gains g_x x + g_o out (nGPT, Doge, ...)
            x = self.rgain.mix(0, x, a + s + m, self.norm_a(x) if self.rgain.g_on[0] != 0 else None)
        xf = x_in if self.par != 0 else x
        if self.res_c[1] != 1:
            x = self.res_c[1] * x
        if self.ffn_split != 0:
            xr = x_in if self.moe_par != 0 else xf       # (Arctic: the routed experts read the LAYER input, the dense
            r_, s_ = self.ffn(self.norm_f(xf), reg, xe=self.norm_f2(xr), xg=self.norm_r(xr), parts=True)   # path x)
            x = x + self._out(2, self.post_f1(s_) + self.post_f2(r_), self.post_f)
        elif self.moe_defer[0] != 0:                     # ScMoE (LongCat): the routed output joins a later layer
            r_, s_ = self.ffn(self.norm_f(xf), reg, parts=True)
            reg["moe_defer"] = r_
            x = x + self._out(2, s_, self.post_f)
        elif self.rgain is not None:
            x = self.rgain.mix(1, x, self._out(2, self.ffn(self.norm_f(xf), reg), self.post_f),
                               self.norm_f(xf) if self.rgain.g_on[1] != 0 else None)
        else:
            x = x + self._out(2, self.ffn(self.norm_f(xf), reg), self.post_f)
        if self.moe_defer[1] != 0:                       # ... added here, after this layer's own FFN
            x = x + reg.pop("moe_defer")
        if self.ple_c != 0:
            g = F.gelu(self.ple_gate(x), approximate="tanh" if self.ple_tanh != 0 else "none")
            x = x + self.ple_c * self.ple_post(self.ple_proj(g * reg["ple"][:, :, self.idx]))
        if self.sn_out is not None and self.sn_c[1] != 0:
            x = self.sn_out(x)
        x = x * self.out_scale if self.out_scale != 1 else x
        if self.bo_write != 0:                           # the backout residual's mid-layer snapshot
            reg["x_bo"] = x
        if self.ee_tap != 0:                             # an early-exit candidate (the end of a loop pass: Ouro)
            reg.setdefault("ee", []).append(x)
        for hk in self.xhooks:                           # ... and at its end (a state write, an injection snapshot)
            x = hk.leave(x, reg)
        return x


_ACT_CODES = ("silu", "gelu", "gelu_tanh", "relu")


def _coded_act(x, code):
    a = _ACT_CODES[int(code)]
    if a == "silu":
        return F.silu(x)
    if a == "relu":
        return F.relu(x)
    return F.gelu(x, approximate="tanh" if a == "gelu_tanh" else "none")


class UPatchEmbed(nn.Module):
    """A continuous-input embedding, the alternative to the token table (cfg.patch_len = P > 0):
    a float series s (B, S) is optionally instance-normalised (RevIN: (s - mean) / std, std = std_biased
    + rev_eps, replaced by 1 where it is <= rev_floor; rev_on = 0: off), LEFT-padded with zeros to a
    multiple of P, cut into N = ceil(S / P) patches u (B, N, P) and, with cfg.patch_mask, each patch is
    concatenated with its padding mask (1 = a real value, 0 = padding) -> (B, N, 2P). The token
    vector is then
        mlp_c * w2(act(w1 u)) + wr u       (biases on all three; act: silu / gelu / gelu_tanh / relu)
    -- Sundial's SundialPatchEmbedding is mlp_c = 1 (an MLP plus a linear residual), a plain linear
    patch projection is mlp_c = 0. The RevIN statistics are kept (self.stats) for the caller's
    de-normalisation of the outputs."""

    def __init__(self, c: UConfig):
        super().__init__()
        P = c.patch_len
        i = 2 * P if c.patch_mask else P
        self.w1, self.w2, self.wr = _lin(i, c.patch_ff), _lin(c.patch_ff, c.d), _lin(i, c.d)
        self.mlp_c = _p(1)
        self.register_buffer("act", torch.tensor(0.0, dtype=F64))          # index into _ACT_CODES
        self.register_buffer("rev", torch.tensor([0.0, 0.0, 0.0], dtype=F64))   # [rev_on, rev_eps, rev_floor]
        self.P, self.mask = P, c.patch_mask
        self.stats = None

    def revin(self, s):
        means = s.mean(-1, keepdim=True)
        std = s.std(-1, keepdim=True, unbiased=False) + self.rev[1]
        std = torch.where(std > self.rev[2], std, torch.ones_like(std))
        return (s - means) / std, means, std

    def forward(self, s):
        s = s.to(F64)
        if self.rev[0] != 0:
            s, means, std = self.revin(s)
            self.stats = (means, std)
        else:
            self.stats = None
        P = self.P
        pad = (P - s.shape[-1] % P) % P
        m = F.pad(torch.ones_like(s), (pad, 0))
        u = F.pad(s, (pad, 0)).unfold(-1, P, P)
        if self.mask:
            u = torch.cat((u, m.unfold(-1, P, P)), -1)
        out = self.wr(u)
        return out + self.mlp_c * self.w2(_coded_act(self.w1(u), self.act)) if self.mlp_c != 0 else out


def _timestep_features(t, dim, max_period):
    """Sinusoidal timestep features [cos(t f), sin(t f)], f_j = exp(-ln(max_period) j / (dim / 2)) (GLIDE / DiT)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=F64) / half)
    a = t.to(F64)[:, None] * freqs[None]
    e = torch.cat((torch.cos(a), torch.sin(a)), -1)
    return torch.cat((e, torch.zeros_like(e[:, :1])), -1) if dim % 2 else e


class UFlowHead(nn.Module):
    """A conditional flow-matching head NETWORK v = net(x, t, z) (cfg.flow_out = C > 0): Sundial's
    SimpleMLPAdaLN (a DiT-style adaLN MLP), conditioned on z, the backbone's readout (width d):
        y = t2(silu(t1(feat(t)))) + cond(z);  h = inp(x)
        per block:  shift, scale, gate = ada(silu(y));  h = h + gate * m2(silu(m1(ln(h) (1 + scale) + shift)))
        out = fin(fin_norm(h) (1 + scale) + shift),  shift, scale = fin_ada(silu(y))
    (ln: LayerNorm with weight / bias; fin_norm: affine-free; eps settable, 1e-6 in Sundial).
    THE BOUNDARY: this deterministic network is what the host reproduces. The flow SAMPLER that calls
    it -- draw Gaussian noise, Euler-step x through `steps` calls of the network, de-normalise -- is a
    runtime procedure, like greedy or top-p decoding of an LM's logits; `flow_sample` below is that
    loop for a GIVEN noise tensor (so a test can run it), not part of the architecture."""

    def __init__(self, c: UConfig):
        super().__init__()
        C, W = c.flow_out, c.flow_width
        self.t1, self.t2 = _lin(c.flow_freq, W), _lin(W, W)
        self.cond, self.inp = _lin(c.d, W), _lin(C, W)
        self.ln = nn.ModuleList(UNorm(W) for _ in range(c.flow_depth))
        self.m1 = nn.ModuleList(_lin(W, W) for _ in range(c.flow_depth))
        self.m2 = nn.ModuleList(_lin(W, W) for _ in range(c.flow_depth))
        self.ada = nn.ModuleList(_lin(W, 3 * W) for _ in range(c.flow_depth))
        self.fin_norm, self.fin_ada, self.fin = UNorm(W), _lin(W, 2 * W), _lin(W, C)
        for n in list(self.ln) + [self.fin_norm]:
            n.centre.fill_(1.0)
        self.register_buffer("max_period", torch.tensor(10000.0, dtype=F64))
        self.freq = c.flow_freq

    def forward(self, x, t, z):
        y = self.t2(F.silu(self.t1(_timestep_features(t, self.freq, float(self.max_period))))) + self.cond(z)
        h = self.inp(x.to(F64))
        sy = F.silu(y)
        for ln, m1, m2, ada in zip(self.ln, self.m1, self.m2, self.ada):
            shift, scale, gate = ada(sy).chunk(3, -1)
            h = h + gate * m2(F.silu(m1(ln(h) * (1 + scale) + shift)))
        shift, scale = self.fin_ada(sy).chunk(2, -1)
        return self.fin(self.fin_norm(h) * (1 + scale) + shift)


class UniversalElephantine(nn.Module):
    def __init__(self, c: UConfig):
        super().__init__()
        self.cfg = c
        self.embed = nn.Embedding(c.vocab, c.d).to(F64)
        self.pos = _p(c.max_pos, c.d)                    # learned-abs table, added to the embedding
        self.prefix = _p(c.n_prefix, c.d)                # meta tokens
        self.ar_q_out, self.ar_w_out = _p(c.d), _p(c.d, fill=1.0)   # final AttnRes mix
        self.sw_score, self.sw_val = _p(c.vocab), _p(c.vocab)   # switch head: per-token key / value
        self.emb_norm_c, self.emb_norm = _p(1), UNorm(c.d)   # a norm of the embedding itself (Bloom)
        self.layers = nn.ModuleList(ULayer(c) for _ in range(c.L))
        for i, layer in enumerate(self.layers):
            layer.idx = i
        if c.ple_dim:                                    # per-layer embeddings: pli = rms_l(P x0) w + E[ids]
            self.ple_E = _p(c.ple_vocab or c.vocab, c.L * c.ple_dim)
            self.ple_P = _lin(c.d, c.L * c.ple_dim, bias=False)
            self.ple_w = _p(c.ple_dim, fill=1.0)
            self.register_buffer("ple_eps", torch.tensor(1e-6, dtype=F64))
        self.norm = UNorm(c.d, c.norm_gate_rank)
        # the readout: an lm_head over the vocabulary, or (head_out) any-width linear head -- a score / class /
        # regression head, or an embedding (the identity) -- with a bias (zero unless a target has one: Phi)
        self.head_in = _lin(c.d, c.head_ff) if c.head_ff else None
        self.register_buffer("head_act", torch.tensor(3.0, dtype=F64))   # index into _ACT_CODES (relu)
        self.lm_head = _lin(c.head_ff or c.d, c.head_out or c.vocab)
        # early exits (layer-wise rerankers): exit j reads norm(x after exits[j] layers) through exit_heads[j]
        self.exit_heads = nn.ModuleList(_lin(c.d, c.head_out or c.vocab) for _ in c.exits)
        # pooling before the head: 0 none (per token), 1 the last real token, 2 the first (CLS), 3 the mean over
        # real tokens. Real tokens: the forward's pad_mask, else ids != pool_pad (pool_pad >= 0), else all
        self.register_buffer("pool", torch.tensor(0.0, dtype=F64))
        self.register_buffer("pool_pad", torch.tensor(-1.0, dtype=F64))
        self.register_buffer("l2_out", torch.tensor(0.0, dtype=F64))   # 1: L2-normalise the readout (embeddings)
        self.patch = UPatchEmbed(c) if c.patch_len else None          # continuous input instead of the table
        self.flow = UFlowHead(c) if c.flow_out else None              # a flow-matching head network on the readout
        self.register_buffer("final_cap", torch.tensor(float("inf"), dtype=F64))   # cap tanh(logits / cap)
        # block-causal attention (SDAR): key s visible to query t iff floor(s / b) <= floor(t / b); 0 = off
        # (then `causal` picks the causal or the full mask). A forward's `mask` overrides both.
        self.register_buffer("block_size", torch.tensor(0.0, dtype=F64))
        if c.doc_mask:    # same-document attention: the eos id that ENDS a document (it belongs to the one it closes)
            self.register_buffer("doc_eos", torch.tensor(-1.0, dtype=F64))   # -1: off
        if c.hc_mult:     # the final stream collapse: mean (hc_head_c = 0) or DeepSeek-V4's HyperHead
            N = c.hc_mult  # sum_i (sigmoid(rms(flat X) fn_i * scale + base_i) + eps) X_i
            self.hc_head_c, self.hc_fn, self.hc_base, self.hc_scale = _p(1), _p(N, N * c.d), _p(N), _p(1, fill=1.0)
            self.register_buffer("hc_head_eps", torch.tensor([1e-6, 1e-6], dtype=F64))   # [hc eps, norm eps]
            self.register_buffer("hc_lag", torch.tensor(0.0, dtype=F64))   # 1: collapse with the last lagged pre
        self.hc_final = _bs.hc_final(c)                  # a collapse module of its own (hc_kind "gated"), else None
        _bs.init_model(self, c)                          # the stream / input / FFN / loop branches (off by default)

    def attn_mask(self, T, causal=True, mask=None, block_size=None):
        """The attention mask every layer runs: a bool tensor (B | 1, 1, T, T) (True = visible), or the
        flag `causal` itself (True: causal, False: full) when no block size / mask is set. `mask` is a
        caller-supplied bool (T, T), (B, T, T) or (B, 1, T, T) mask (SDAR's attention_mask: its code runs
        non-causal attention under exactly this mask); block_size (default: the block_size buffer) builds
        the block-causal mask the SDAR generation code passes."""
        if mask is not None:
            m = mask.to(torch.bool)
            m = m[None, None] if m.dim() == 2 else (m[:, None] if m.dim() == 3 else m)
            assert m.shape[-2:] == (T, T), (m.shape, T)
            return m
        b = int(self.block_size) if block_size is None else int(block_size)
        if b > 0 and causal:
            blk = torch.arange(T) // b
            return (blk[None, :] <= blk[:, None])[None, None]
        return causal

    def _pool(self, h, ids, pad_mask):
        p = int(self.pool)
        if p == 0:
            return h
        if p == 2:
            return h[:, 0]
        if pad_mask is None and ids is not None and self.pool_pad >= 0:
            pad_mask = ids != int(self.pool_pad)
        B, T = h.shape[:2]
        if pad_mask is None:
            pad_mask = torch.ones(B, T, dtype=torch.bool)
        if p == 1:                                       # the rightmost real token (HF *ForSequenceClassification)
            idx = (torch.arange(T) * pad_mask.long()).argmax(-1)
            return h[torch.arange(B), idx]
        m = pad_mask[..., None].to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def readout(self, h, head=None, ids=None, pad_mask=None):
        """The output head on a residual stream h: the final norm, the pooling rule, a linear head of any
        width (+ bias), the logit softcap, an optional L2 normalisation. An LM is pool 0, head = lm_head
        over the vocabulary; a score head is the same with a narrow head and a pooling rule."""
        z = self._pool(self.norm(h), ids, pad_mask)
        if head is None and self.head_in is not None:
            z = _coded_act(self.head_in(z), self.head_act)
        out = (self.lm_head if head is None else head)(z)
        if getattr(self, "multimax", None) is not None:  # an elementwise piecewise map of the logits (ERNIE-5.5)
            out = self.multimax(out)
        if torch.isfinite(self.final_cap):
            out = self.final_cap * torch.tanh(out / self.final_cap)
        return F.normalize(out, p=2, dim=-1) if self.l2_out != 0 else out

    def flow_sample(self, z, noise, steps, num_samples=1):
        """RUNTIME, not architecture: Sundial's FlowLoss.sample Euler loop for a given noise tensor
        (B * num_samples, C): x <- x + (net(x, 1000 i / steps, z) - noise) / steps, i = 0 .. steps - 1;
        returns (B, num_samples, C)."""
        z = z.repeat(num_samples, 1)
        x, dt = noise.to(F64), 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i / steps, dtype=F64)
            x = x + (self.flow(x, t * 1000, z) - noise.to(F64)) * dt
        return x.reshape(num_samples, -1, x.shape[-1]).transpose(0, 1)

    def forward(self, ids=None, causal=True, mask=None, block_size=None, *, series=None, pad_mask=None,
                selfcond=None):
        if series is not None:                           # continuous input: patches -> vectors (UPatchEmbed)
            assert ids is None and self.patch is not None
            x = self.patch(series)
            assert not (self.cfg.ple_dim or self.cfg.lora_adapters or self.cfg.engram_slots or self.cfg.engram_hc
                        or self.pool_pad >= 0), "token-keyed branches need token ids"
        else:
            x = self.embed(ids)
        if selfcond is not None:                         # a self-conditioning signal (the previous step's soft
            x = self.selfcond(x, selfcond)               # embeddings, diffusion LMs): branches_stream.USelfCond
        if self.ngram_in is not None:                    # a hashed bigram embedding added to the input (Crowd)
            x = x + self.ngram_in(ids)
        T = x.shape[1]
        if mask is not None or block_size or self.block_size > 0:
            assert not self.cfg.n_prefix, "a block / caller-supplied mask with prefix tokens is not defined"
            causal = self.attn_mask(T, causal, mask, block_size)
        if self.cfg.doc_mask and self.doc_eos >= 0:     # documents: doc_t = #eos before t (Ethoric document_ids)
            assert ids is not None and not self.cfg.n_prefix
            e = (ids == int(self.doc_eos)).long()
            doc = e.cumsum(1) - e
            same = (doc[:, :, None] == doc[:, None, :])[:, None]
            if not isinstance(causal, torch.Tensor):
                causal = torch.ones(T, T, dtype=torch.bool).tril() if causal else torch.ones(T, T, dtype=torch.bool)
            causal = causal & same
        if self.cfg.max_pos:
            x = x + self.pos[:T]
        if self.emb_norm_c != 0:
            x = self.emb_norm(x)
        if self.smear is not None:                       # the previous token's gated embedding (nanochat smear)
            x = self.smear(x)
        n = self.cfg.n_prefix
        if n:
            x = torch.cat((self.prefix.expand(x.shape[0], -1, -1), x), 1)
        reg = {"ids": ids, "x0": x}                      # x0: the input embedding (branches that re-read it)
        if self.hrm_z0 is not None:                      # two-state loops: z_H = the embedding, z_L = a learned start
            reg["z"] = [x, self.hrm_z0.expand_as(x)]
        if self.cfg.ple_dim:
            B_, T_ = ids.shape
            pp = self.ple_P(x).view(B_, T_, self.cfg.L, self.cfg.ple_dim)
            reg["ple"] = _rms(pp, self.ple_eps) * self.ple_w + self.ple_E[ids].view(B_, T_, self.cfg.L, self.cfg.ple_dim)
        if self.cfg.lora_adapters:                      # adapter index = round(causal softmax-avg of values)
            sc, val = self.sw_score[ids], self.sw_val[ids]
            T_ = ids.shape[1]
            m = torch.ones(T_, T_, dtype=torch.bool).tril()
            w = sc[:, None, :].expand(-1, T_, -1).masked_fill(~m, float("-inf")).softmax(-1)
            avg = (w * val[:, None, :]).sum(-1)
            reg["adapter"] = avg.round().long().clamp(0, self.cfg.lora_adapters)
        B = self.cfg.attnres_block
        if self.cfg.exits:                              # early exits: one readout per exit layer count
            assert not (self.cfg.attnres_full or self.cfg.hc_mult or B), "exits read the plain residual stream"
            at, last = {}, max(self.cfg.exits)
            for i, layer in enumerate(self.layers):
                if i in self.cfg.exits:
                    at[i] = x
                if i == last:
                    break
                x = layer(x, causal, reg)
            else:
                at[len(self.layers)] = x
            return tuple(self.readout(at[e][:, n:], hd, ids, pad_mask) for e, hd in zip(self.cfg.exits, self.exit_heads))
        if self.cfg.attnres_full:
            sources = [x]
            for layer in self.layers:
                layer.forward_fullres(sources, causal, reg)
            x = self.layers[0].mix(sources, self.ar_q_out, self.ar_w_out)
        elif self.cfg.hc_mult:                          # N streams, collapsed by their mean at the end
            X = x[:, :, None].expand(-1, -1, self.cfg.hc_mult, -1)
            for layer in self.layers:
                X = layer.forward_hc(X, causal, reg)
            if self.hc_final is not None:               # Qwen4-Exp: the gated-residual read of the streams
                x = self.hc_final(X)[2]
            elif self.hc_lag != 0:                      # DeepSeek-V4.1: the last FFN's pre, no separate head
                x = (reg["hc_pre"][..., None] * X).sum(2)
            elif self.hc_head_c != 0:
                flat = X.flatten(2)
                flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.hc_head_eps[1])
                pre = torch.sigmoid(flat @ self.hc_fn.T * self.hc_scale + self.hc_base) + self.hc_head_eps[0]
                x = (pre[..., None] * X).sum(2)
            else:
                x = X.mean(2)
        elif B:
            blocks, partial = [x], (x if self.cfg.attnres_emb_partial else None)   # ARLoop: the first partial
            #                                                                     sum starts at the embedding
            for i, layer in enumerate(self.layers):
                if i > 0 and i % B == 0:
                    blocks.append(partial); partial = None
                partial = layer.forward_attnres(blocks, partial, causal, reg)
            x = self.layers[0].mix(blocks + [partial], self.ar_q_out, self.ar_w_out)
        else:
            for layer in self.layers:
                x = layer(x, causal, reg)
        if self.bo_c != 0:                               # backout: x - lambda x_mid before the final norm
            x = x - self.bo_c * reg["x_bo"]
        if self.early_exit is not None and reg.get("ee"):   # per-token exit among the tapped candidates
            x = self.early_exit(reg["ee"])
        return self.readout(x[:, n:], None, ids, pad_mask)


# --- loaders: each family is a weight setting of the one architecture -----------------------

def _cp(dst, src):
    with torch.no_grad():
        dst.copy_(src.detach().to(F64))


def _norm_from(dst: UNorm, src, one_plus=False):
    w = src.weight.detach().to(F64)
    _cp(dst.weight, 1 + w if one_plus else w)
    eps = getattr(src, "variance_epsilon", getattr(src, "eps", 1e-6))
    dst.eps.fill_(float(eps))


def _tile_kv(w, n_kv, H, hd):
    """GQA/MQA -> MHA: repeat each KV head across its query group (rows of a projection)."""
    return w.view(n_kv, hd, *w.shape[1:]).repeat_interleave(H // n_kv, 0).reshape(H * hd, *w.shape[1:])


def _attn_from(dst: UAttention, src, cfg, inv_freq, perm=None, gated=False, qk_norm=False):
    H, hd = dst.H, dst.hd
    n_kv = cfg.num_key_value_heads
    P = (lambda w: w.view(-1, hd, *w.shape[1:])[:, perm].reshape(w.shape)) if perm is not None else (lambda w: w)
    qw = src.q_proj.weight.detach()
    if gated:                                    # Qwen3-Next: per head [query | gate]
        qw = qw.view(H, 2 * hd, -1)
        _cp(dst.g_proj.weight, qw[:, hd:].reshape(H * hd, -1))
        qw = qw[:, :hd].reshape(H * hd, -1)
        with torch.no_grad():
            dst.gate_coef.copy_(torch.tensor([0.0, 1.0]))
    _cp(dst.q_proj.weight, P(qw))
    _cp(dst.k_proj.weight, P(_tile_kv(src.k_proj.weight.detach(), n_kv, H, hd)))
    _cp(dst.v_proj.weight, _tile_kv(src.v_proj.weight.detach(), n_kv, H, hd))
    for name, tile in (("q_proj", False), ("k_proj", True), ("v_proj", True)):
        b = getattr(src, name).bias
        if b is not None:
            b = _tile_kv(b.detach(), n_kv, H, hd) if tile else b.detach()
            _cp(getattr(dst, name).bias, P(b) if name != "v_proj" else b)
    _cp(dst.o_proj.weight, src.o_proj.weight)
    if src.o_proj.bias is not None:
        _cp(dst.o_proj.bias, src.o_proj.bias)
    if qk_norm:
        for i, (w_dst, s_n, coef) in enumerate(((dst.q_nw, src.q_norm, dst.q_coef), (dst.k_nw, src.k_norm, dst.k_coef))):
            w = s_n.weight.detach().to(F64)
            w = 1 + w if qk_norm == "one_plus" else w
            if perm is not None:
                w = w[perm]
            with torch.no_grad():
                w_dst.copy_(w.expand(H, hd))
                dst.qk_eps[i] = float(getattr(s_n, "variance_epsilon", getattr(s_n, "eps", 1e-6)))
                coef.copy_(torch.tensor([0.0, 1.0]))
    with torch.no_grad():
        dst.inv_freq.zero_()
        dst.inv_freq[: inv_freq.numel()] = inv_freq.to(F64)
        dst.scale.fill_(float(src.scaling))


def _partial_rope_perm(hd, rot):
    """Target RoPE pairs (j, j+rot/2) within the first `rot` dims; the host pairs (i, i+hd/2).
    Return host<-target dim order putting each target pair on a host pair, pass dims after."""
    h, r = hd // 2, rot // 2
    passes = list(range(rot, hd))
    first = list(range(r)) + passes[: h - r]
    second = [j + r for j in range(r)] + passes[h - r:]
    return torch.tensor(first + second)


def _dense_ffn_from(dst: UFFN, mlp):
    _cp(dst.s_gate.weight, mlp.gate_proj.weight)
    _cp(dst.s_up.weight, mlp.up_proj.weight)
    _cp(dst.s_down.weight, mlp.down_proj.weight)


# transformers ACT2FN names -> the exact UFFN branch (the tanh-approx GeLUs are one function written
# with different constants / association: they agree to fp64 rounding)
_GELU_ERF = ("gelu", "gelu_python")
_GELU_TANH = ("gelu_new", "gelu_pytorch_tanh", "gelu_fast", "gelu_accurate", "gelu_tanh")


@torch.no_grad()
def _set_act(f: UFFN, act, gated):
    """Select the UFFN activation branch that IS the target's `hidden_act` / `activation_function`:
    gated (act(gate) * up) or ungated (act(up)). SiLU is the default gated branch."""
    a = str(act).lower()
    if a in ("silu", "swish") and gated:
        return
    if a in _GELU_ERF + _GELU_TANH:
        (f.c_geglu if gated else f.c_gelu).fill_(1.0)
        f.gelu_tanh.fill_(float(a in _GELU_TANH))
        return
    if a == "mish" and gated:
        f.c_mish.fill_(1.0)
        return
    if a in ("relu2", "relu_squared", "squared_relu") and not gated:
        f.c_relu2.fill_(1.0)
        return
    if a == "relu" and not gated:
        f.c_relu.fill_(1.0)
        return
    raise NotImplementedError(f"no UFFN branch for {'gated' if gated else 'ungated'} activation {act!r}")


@torch.no_grad()
def _ungated_ffn_from(f: UFFN, w_up, b_up, w_down, b_down, act):
    """An ungated up / down MLP  down(act(up x + b)) + b  (nn.Linear layout: (out, in))."""
    _cp(f.s_up.weight, w_up)
    _cp(f.s_down.weight, w_down)
    if b_up is not None:
        _cp(f.s_up.bias, b_up)
    if b_down is not None:
        _cp(f.s_down.bias, b_down)
    _set_act(f, act, gated=False)


@torch.no_grad()
def _mha_from(dst: UAttention, wq, wk, wv, bq, bk, bv, wo, bo, scale, inv_freq=None, perm=None):
    """Plain multi-head attention from its (out, in) projections: q / k rows permuted by `perm` onto
    the host's RoPE pairs, inv_freq on the host's first pairs (None: NoPE)."""
    hd = dst.hd
    P = (lambda w: w.view(-1, hd, *w.shape[1:])[:, perm].reshape(w.shape)) if perm is not None else (lambda w: w)
    keep = lambda w: w
    for lin, w, b, p in ((dst.q_proj, wq, bq, P), (dst.k_proj, wk, bk, P), (dst.v_proj, wv, bv, keep),
                         (dst.o_proj, wo, bo, keep)):
        _cp(lin.weight, p(w.detach()))
        if b is not None:
            _cp(lin.bias, p(b.detach()))
    dst.inv_freq.zero_()
    if inv_freq is not None:
        dst.inv_freq[: inv_freq.numel()] = inv_freq.to(F64)
    dst.scale.fill_(float(scale))


def _ln_from(dst: UNorm, src):
    """nn.LayerNorm (weight, bias, eps) -> UNorm with centre = 1."""
    _norm_from(dst, src)
    dst.centre.fill_(1.0)
    if getattr(src, "bias", None) is not None:
        _cp(dst.bias, src.bias)


@torch.no_grad()
def from_llama(target):
    """Any Llama-shaped decoder (model.layers of self_attn q / k / v / o + mlp + two pre-RMSNorms):
    Llama, Mistral / Ministral (sliding window), Ministral-3 (the Llama-4 query temperature: qtemp),
    Qwen2 (q / k / v biases, sliding layers), Qwen3 (per-head qk-norm), Helium (interleaved RoPE),
    Seed-OSS, ERNIE-4.5 (interleaved RoPE, biases), SmolLM3 (NoPE layers), Arcee (ungated relu^2 MLP);
    with a routed FFN (_ffn_any / _router_from): Mixtral, Qwen2/3-MoE, OLMoE (whole-projection qk-norm,
    clip_qkv), ERNIE-4.5-MoE, GLM-4-MoE, HY-V3, Mellum, MiniMax-M2, Solar-Open, EXAONE-MoE (NoPE full
    layers beside its sliding ones). RoPE scaling is the target's inv_freq; its attention_scaling (YaRN,
    LongRoPE) is folded into the softmax scale. Diffusion LMs with this body use the same weights and
    run with causal=False."""
    tc, m = target.config, target.model
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=sff,
               experts=E, expert_ff=ef, top_k=k)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    rope = m.rotary_emb
    layout = "interleave" if tc.model_type in ("ernie4_5", "ernie4_5_moe", "helium") else "half"
    rp = getattr(tc, "rope_parameters", None) or {}
    beta = rp.get("llama_4_scaling_beta") if isinstance(rp, dict) else None
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        use_rope = getattr(a, "use_rope", True)                              # SmolLM3 NoPE layers
        if tc.model_type == "exaone_moe" and tc.sliding_window is not None:  # rotary on sliding layers only
            use_rope = tc.layer_types[i] == "sliding_attention"
        if beta:
            ul.attn.qtemp.copy_(torch.tensor([float(beta), float(rp["original_max_position_embeddings"])], dtype=F64))
        inv, att = _rope_of(rope, (getattr(tc, "layer_types", None) or [None] * tc.num_hidden_layers)[i])
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=getattr(a, "scaling", hd ** -0.5),
                  inv_freq=inv if use_rope else None,
                  att_scale=att, layout=layout, window=_sliding(tc, i),
                  q_norm=getattr(a, "q_norm", None), k_norm=getattr(a, "k_norm", None),
                  clip=getattr(tc, "clip_qkv", None))
        # SwiGLU, a mish GLU, relu^2, or a routed MoE (+ shared expert)
        _ffn_any(ul.ffn, _mlp_of(tl), tc, getattr(tc, "hidden_act", None) or "silu")
    return u


@torch.no_grad()
def from_gpt2(target):
    """GPT-2 (GPT2LMHeadModel): learned absolute positions (the host's pos table), NoPE attention with
    a biased fused c_attn (Conv1D: y = x W + b), biased pre-LayerNorms, an ungated biased MLP under the
    config's activation_function (gelu_new: the tanh GeLU branch; gelu: erf), tied lm_head."""
    tc, m = target.config, target.transformer
    d, H = tc.n_embd, tc.n_head
    hd = d // H
    c = UConfig(vocab=tc.vocab_size, d=d, L=tc.n_layer, heads=H, head_dim=hd, shared_ff=tc.n_inner or 4 * d,
                max_pos=tc.n_positions)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.wte.weight)
    _cp(u.pos, m.wpe.weight)
    _ln_from(u.norm, m.ln_f)
    _cp(u.lm_head.weight, target.lm_head.weight)
    for i, (ul, tl) in enumerate(zip(u.layers, m.h)):
        a = tl.attn
        assert not a.is_cross_attention
        _ln_from(ul.norm_a, tl.ln_1)
        _ln_from(ul.norm_f, tl.ln_2)
        W, b = a.c_attn.weight.detach().T, a.c_attn.bias.detach()               # Conv1D -> (out, in)
        scale = (1 / math.sqrt(hd) if a.scale_attn_weights else 1.0) / (i + 1 if a.scale_attn_by_inverse_layer_idx else 1)
        _mha_from(ul.attn, W[:d], W[d:2 * d], W[2 * d:], b[:d], b[d:2 * d], b[2 * d:],
                  a.c_proj.weight.detach().T, a.c_proj.bias, scale)
        _ungated_ffn_from(ul.ffn, tl.mlp.c_fc.weight.detach().T, tl.mlp.c_fc.bias,
                          tl.mlp.c_proj.weight.detach().T, tl.mlp.c_proj.bias, tc.activation_function)
    return u


@torch.no_grad()
def from_gpt_neox(target):
    """GPT-NeoX / Pythia (GPTNeoXForCausalLM): fused per-head [q | k | v] query_key_value, partial
    rotary (rotary_pct of each head, rotate-half within those dims: permuted onto the host's pairs),
    biased LayerNorms and projections, an ungated biased MLP under hidden_act (gelu: the erf GeLU
    branch; gelu_fast / gelu_new: tanh), and the parallel residual x + attn(ln1 x) + mlp(ln2 x)
    (ULayer.par = 1) when use_parallel_residual."""
    tc, m = target.config, target.gpt_neox
    d, H = tc.hidden_size, tc.num_attention_heads
    hd = d // H
    c = UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd,
                shared_ff=tc.intermediate_size)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_in.weight)
    _ln_from(u.norm, m.final_layer_norm)
    _cp(u.lm_head.weight, target.embed_out.weight)
    inv_freq = m.rotary_emb.inv_freq
    assert m.rotary_emb.attention_scaling == 1.0
    for ul, tl in zip(u.layers, m.layers):
        a = tl.attention
        assert inv_freq.numel() == a.rotary_ndims // 2
        _ln_from(ul.norm_a, tl.input_layernorm)
        _ln_from(ul.norm_f, tl.post_attention_layernorm)
        ul.par.fill_(float(tc.use_parallel_residual))
        W, b = a.query_key_value.weight.detach().view(H, 3, hd, d), a.query_key_value.bias
        qkv = [W[:, j].reshape(H * hd, d) for j in range(3)]
        bqkv = [b.detach().view(H, 3, hd)[:, j].reshape(H * hd) for j in range(3)] if b is not None else [None] * 3
        _mha_from(ul.attn, *qkv, *bqkv, a.dense.weight, a.dense.bias, a.scaling,
                  inv_freq=inv_freq, perm=_partial_rope_perm(hd, a.rotary_ndims))
        _ungated_ffn_from(ul.ffn, tl.mlp.dense_h_to_4h.weight, tl.mlp.dense_h_to_4h.bias,
                          tl.mlp.dense_4h_to_h.weight, tl.mlp.dense_4h_to_h.bias, tc.hidden_act)
    return u


@torch.no_grad()
def from_gemma(target):
    """Gemma (1), Gemma-2, VaultGemma and Gemma-3 text (GemmaForCausalLM / Gemma2ForCausalLM /
    VaultGemmaForCausalLM / Gemma3ForCausalLM): embeddings scaled by sqrt(d) (folded into the table),
    offset RMSNorms (1 + w), GQA, a GeGLU MLP under hidden_act (Gemma) / hidden_activation (Gemma-2 / 3)
    (gelu_pytorch_tanh: the tanh GeLU branch, gated; gelu: erf), tied lm_head.
    Gemma-2 adds the attention-logit softcap cap tanh(s / cap) (UAttention.cap) and the final-logit
    softcap (UniversalElephantine.final_cap), sandwich norms (post-norms on each sub-block's output),
    query_pre_attn_scalar scaling and alternating sliding-window layers (visible iff q - k <
    sliding_window: host window = sliding_window - 1); Gemma-3 adds per-head (1 + w) qk-norm and a
    local RoPE base for the sliding layers (and, like the reference, no attention softcap even when the
    config sets attn_logit_softcapping); VaultGemma is Gemma-2 without the post-norms. The rotary's
    attention_scaling (Gemma-3 YaRN) is folded into the scale."""
    tc, m = target.config, target.model
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    # the field the target's MLP reads: GemmaMLP ACT2FN[hidden_act], Gemma2 / 3 MLP ACT2FN[hidden_activation]
    act = tc.hidden_act if tc.model_type == "gemma" else tc.hidden_activation
    # Gemma-3's ScaledWordEmbedding keeps sqrt(d) in an fp32-created buffer; Gemma (1) / 2 scale in the
    # activation dtype: use the target's own rounding of the constant
    scale = getattr(m.embed_tokens, "embed_scale", None)
    scale = torch.tensor(tc.hidden_size ** 0.5, dtype=F64) if scale is None else scale.detach().to(F64)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, "rms1p", emb_scale=scale)
    if getattr(tc, "final_logit_softcapping", None):
        u.final_cap.fill_(float(tc.final_logit_softcapping))
    # only Gemma-2 / VaultGemma attention passes softcap= to the attention function; Gemma3Attention stores
    # attn_logit_softcapping but never applies it (transformers 4.57), and Gemma (1) has none
    cap = (getattr(tc, "attn_logit_softcapping", None)
           if tc.model_type in ("gemma2", "vaultgemma", "cost_wise_gemma") else None)   # (cost-wise: Gemma-2 layers)
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        sliding = getattr(a, "sliding_window", None) is not None
        rope = m.rotary_emb_local if sliding and hasattr(m, "rotary_emb_local") else m.rotary_emb
        _norm_any(ul.norm_a, tl.input_layernorm, "rms1p")
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, window=a.sliding_window if sliding else None,
                  q_norm=getattr(a, "q_norm", None), k_norm=getattr(a, "k_norm", None))
        if hasattr(a, "q_norm"):                       # Gemma-3 qk-norms are (1 + w) too
            ul.attn.q_nw.add_(1.0); ul.attn.k_nw.add_(1.0)
        if cap:
            ul.attn.cap.fill_(float(cap))
        if hasattr(tl, "pre_feedforward_layernorm"):
            _norm_any(ul.norm_f, tl.pre_feedforward_layernorm, "rms1p")
            if hasattr(tl, "post_feedforward_layernorm"):
                _norm_any(ul.post_a, tl.post_attention_layernorm, "rms1p")
                _norm_any(ul.post_f, tl.post_feedforward_layernorm, "rms1p")
                ul.post_coef[0] = torch.tensor([0.0, 1.0])
                ul.post_coef[2] = torch.tensor([0.0, 1.0])
        else:
            _norm_any(ul.norm_f, tl.post_attention_layernorm, "rms1p")
        _mlp_any(ul.ffn, tl.mlp, act)
    return u


@torch.no_grad()
def from_apertus(target):
    """Apertus (swiss-ai): a Llama body (RoPE incl. llama3 / linear scaling, GQA, RMSNorm) with
    per-head qk-norm and an UNGATED up / down MLP whose activation is xIELU with its own learned
    alpha_p / alpha_n (and beta / eps buffers) in every layer: the c_xielu branch on the shared path."""
    tc, m = target.config, target.model
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    assert tc.hidden_act == "xielu"
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers,
                heads=tc.num_attention_heads, head_dim=hd, shared_ff=tc.intermediate_size)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    for ul, tl in zip(u.layers, m.layers):
        _norm_from(ul.norm_a, tl.attention_layernorm)
        _norm_from(ul.norm_f, tl.feedforward_layernorm)
        _attn_from(ul.attn, tl.self_attn, tc, m.rotary_emb.inv_freq, qk_norm=True)
        f, mlp, act = ul.ffn, tl.mlp, tl.mlp.act_fn
        assert mlp.up_proj.bias is None and mlp.down_proj.bias is None
        _cp(f.s_up.weight, mlp.up_proj.weight)
        _cp(f.s_down.weight, mlp.down_proj.weight)
        f.s_gate.weight.zero_()                                   # ungated: xIELU reads u only
        f.c_xielu.fill_(1.0)
        _cp(f.xielu_a, torch.cat((act.alpha_p.reshape(1), act.alpha_n.reshape(1))))
        _cp(f.xielu_be, torch.stack((act.beta.reshape(()), act.eps.reshape(()))))
    return u


@torch.no_grad()
def motif_ffn_block(mod, config):
    """Motif-3 FFN -> a standalone UFFN with the PolyNorm-GLU branch (see _motif_ffn_from)."""
    dense = not hasattr(mod, "router")
    sh = mod if dense else mod.shared_experts
    if dense:
        c = UConfig(vocab=1, d=config.hidden_size, L=1, shared_ff=mod.intermediate_size)
    else:
        ex = mod.experts
        c = UConfig(vocab=1, d=config.hidden_size, L=1, experts=ex.num_experts, expert_ff=ex.intermediate_dim,
                    top_k=mod.experts_top_k, shared_ff=sh.intermediate_size if sh is not None else 1)
    f = UFFN(c)
    _motif_ffn_from(f, mod, config)
    return f


@torch.no_grad()
def _motif_ffn_from(f: UFFN, mod, config):
    """Motif-3 FFN -> UFFN's PolyNorm-GLU branch. `mod` is a dense MotifMLP or a MoE
    (TokenChoiceTopKRouter: sigmoid or softmax scores, top-k on scores + expert_bias, route_norm
    (sum + 1e-20), route_scale; MotifExperts with per-expert GroupedPolyNorm (bias clamp, hidden
    clamp, result clamp, output scale); an optional PolyNorm-GLU shared expert, added with weight 1).
    The dense / shared bank may be narrower than the host's: it fills the first w rows and the
    PolyNorm RMS runs over those w (poly_cfg[5])."""
    assert config.hidden_act == "poly_norm"
    dense = not hasattr(mod, "router")
    sh = mod if dense else mod.shared_experts
    f.c_poly.fill_(1.0)
    hc = getattr(config, "hidden_clamp", None)
    sig = getattr(config, "polynorm_sigmoid_weight", True)
    f.poly_sig.fill_(1.0 if sig else 0.0)
    if sh is not None:
        an, w = sh.act_fn, sh.intermediate_size
        assert type(an).__name__ == "PolyNormTorch" and bool(an.sigmoid_weight) == bool(sig)
        f.s_gate.weight.zero_(); f.s_up.weight.zero_(); f.s_down.weight.zero_()
        _cp(f.s_gate.weight[:w], sh.gate_proj.weight); _cp(f.s_up.weight[:w], sh.up_proj.weight)
        _cp(f.s_down.weight[:, :w], sh.down_proj.weight)
        _cp(f.s_poly_w, an.weight); _cp(f.s_poly_b, an.bias)
        f.poly_cfg[0] = float(an.eps)
        f.poly_cfg[2] = float(sh.polynorm_output_scale)
        f.poly_cfg[5] = float(w)
        assert sh.hidden_clamp == hc
    else:
        f.s_coef.zero_()                                         # no shared expert
    if hc is not None:
        f.poly_cfg[1] = float(hc)
    if not dense:
        ex, r, gp = mod.experts, mod.router, mod.experts.act_fn
        assert type(gp).__name__ == "GroupedPolyNorm" and not mod.score_before_experts
        assert not r._debug_force_load_balance and r.score_func in ("sigmoid", "softmax")
        I = ex.intermediate_dim
        assert tuple(f.w_gate.shape) == (ex.num_experts, I, config.hidden_size) and f.top_k == mod.experts_top_k
        _cp(f.router.weight, r.gate.weight)
        f.score_c.fill_(1.0 if r.score_func == "sigmoid" else 0.0)
        if mod.expert_bias is not None:
            _cp(f.sel_bias, mod.expert_bias)
        if r.route_norm:
            f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
        f.route_scale.fill_(float(r.route_scale))
        _cp(f.w_gate, ex.gate_up_proj[:, :I]); _cp(f.w_up, ex.gate_up_proj[:, I:])
        _cp(f.w_down, ex.down_proj)
        _cp(f.poly_w, gp.weight); _cp(f.poly_b, gp.bias)
        assert float(gp.eps) == float(f.poly_cfg[0]) and gp.hidden_clamp == hc and bool(gp.sigmoid_weight) == bool(sig)
        assert sh is None or float(gp.output_scale) == float(f.poly_cfg[2])
        f.poly_cfg[0] = float(gp.eps); f.poly_cfg[2] = float(gp.output_scale)
        if gp.bias_clamp is not None:
            f.poly_cfg[3] = float(gp.bias_clamp)
        if hc is not None:
            f.poly_cfg[4] = float(hc)


def _gdla_perm(hd, nope, r):
    """GDLA q / k head dims are [nope | rope], RoPE pairs (nope + j, nope + j + r/2); the host pairs
    (i, i + hd/2). Host dim order (indices into the target's) putting each rope pair on a host pair."""
    h, rh = hd // 2, r // 2
    np_ = list(range(nope))
    first = [nope + j for j in range(rh)] + np_[: h - rh]
    second = [nope + rh + j for j in range(rh)] + np_[h - rh:]
    return torch.tensor(first + second)


@torch.no_grad()
def _gdla_from(a: UAttention, at, inv_freq):
    """Motif-3 GDLA (grouped differential latent attention, diff_v2) -> the attention block.
    q = wq_b q_norm(wq_a x) and [k_nope | v] = wkv_b kv_norm(kv_latent) read the latent-norm features
    (block-averaged RMS over [q latent | kv latent]); k_pe = wkv_a's rope rows on x, shared by the kv
    heads; [nope | rope] dims permuted so the rope pairs are host RoPE pairs (the rest NoPE); GQA tiled;
    v zero-padded to head_dim. Heads come in groups of (ratio signal + 1 noise): each signal head's
    output is o_i - sigmoid(lambda_proj x)_i o_noise (the dl branch), times sigmoid(wq_b_gate q latent)
    elementwise (the output gate), then wo over the signal heads (noise-head columns zero). Sliding
    window layers see t - s <= sliding_window (flash window_size = (W - 1, W - 1), W = window + 1)."""
    H, hd, Hk = at.num_heads, at.head_dim, at.num_key_value_heads
    nope, r, vd, qr, kr = at.qk_nope_head_dim, at.qk_rope_head_dim, at.v_head_dim, at.q_lora_rank, at.kv_lora_rank
    assert at.diff_v2 and a.H == H and a.hd == hd and a.Dn == qr + kr and hd % 2 == 0 and r % 2 == 0 and vd <= hd
    assert not at.headwise_attn_output_gate                     # (the HF forward does not apply it)
    W = lambda lin: lin.weight.detach().to(F64)
    d = at.hidden_size
    a.pn.weight.copy_(torch.cat((W(at.wq_a), W(at.wkv_a)[:kr])))
    a.n_avg.copy_(torch.block_diag(torch.full((qr, qr), 1 / qr, dtype=F64), torch.full((kr, kr), 1 / kr, dtype=F64)))
    a.n_w.copy_(torch.cat((at.q_norm.weight.detach().to(F64), at.kv_norm.weight.detach().to(F64))))
    assert at.q_norm.variance_epsilon == at.kv_norm.variance_epsilon
    a.n_eps.fill_(float(at.q_norm.variance_epsilon))
    QL, KL = slice(d, d + qr), slice(d + qr, d + qr + kr)       # columns of [x, nb]
    perm = _gdla_perm(hd, nope, r)
    rep = H // Hk
    wkv_b = W(at.wkv_b).view(Hk, nope + vd, kr)
    q = torch.zeros(H, hd, d + qr + kr, dtype=F64); k = torch.zeros_like(q); v = torch.zeros_like(q)
    q[:, :, QL] = W(at.wq_b).view(H, hd, qr)
    k[:, :nope, KL] = wkv_b[:, :nope].repeat_interleave(rep, 0)
    k[:, nope:, :d] = W(at.wkv_a)[kr:kr + r]                    # shared rope key, every head
    v[:, :vd, KL] = wkv_b[:, nope:].repeat_interleave(rep, 0)
    a.q_proj.weight.copy_(q[:, perm].reshape(H * hd, -1)); a.k_proj.weight.copy_(k[:, perm].reshape(H * hd, -1))
    a.v_proj.weight.copy_(v.reshape(H * hd, -1))
    a.inv_freq.zero_(); a.inv_freq[: r // 2] = inv_freq.detach().to(F64)
    a.scale.fill_(float(at.scaling))
    a.window.fill_(float(at.sliding_window - 1) if at.sliding_window is not None else float("inf"))
    g, ratio = at.num_noise_heads, at.grouped_ratio
    assert H == g * (ratio + 1)
    o = torch.zeros(d, H, hd, dtype=F64)
    gate = torch.zeros(H, hd, d + qr + kr, dtype=F64)
    wo = W(at.wo).view(d, at.n_signal_heads, vd)
    wg = W(at.wq_b_gate).view(at.n_signal_heads, vd, qr) if at.wq_b_gate is not None else None
    a.dl_proj.weight.zero_(); a.dl_m.zero_()
    for h in range(H):
        grp, pos = divmod(h, ratio + 1)
        if pos == ratio:                                         # a noise head: no output columns
            continue
        i = grp * ratio + pos                                    # its signal index
        o[:, h, :vd] = wo[:, i]
        if wg is not None:
            gate[h, :vd, QL] = wg[i]
        a.dl_proj.weight[h, :d] = W(at.lambda_proj)[i]
        a.dl_src[h] = grp * (ratio + 1) + ratio
        a.dl_m[h] = 1.0
    a.o_proj.weight.copy_(o.reshape(d, H * hd))
    a.dl_c.fill_(1.0)
    if wg is not None:
        a.g_proj.weight.copy_(gate.reshape(H * hd, -1))
        a.gate_coef.copy_(torch.tensor([0.0, 1.0]))


@torch.no_grad()
def _motif_hc_from(u: UHC, mh):
    """Motif MHCLayer -> UHC (sk variant): the input RMSNorm weight folds into the projections."""
    nw = mh.rms_norm.weight.detach().to(F64)
    assert u.N == mh.expansion_rate and u.iters == mh.sinkhorn_iters
    u.fn.copy_(torch.cat([lin.weight.detach().to(F64) for lin in (mh.proj_pre, mh.proj_post, mh.proj_res)]) * nw)
    u.base.copy_(torch.cat((mh.bias_pre.detach(), mh.bias_post.detach(), mh.bias_res.detach().flatten())).to(F64))
    u.scale.copy_(torch.cat((mh.alpha_pre.detach(), mh.alpha_post.detach(), mh.alpha_res.detach())).to(F64))
    u.eps.copy_(torch.tensor([0.0, float(mh.rms_norm.variance_epsilon)], dtype=F64))
    u.post_c.fill_(mh.h_post_coeff); u.sk.fill_(1.0)


@torch.no_grad()
def from_motif(target):
    """Motif-3 (MotifForCausalLM, its own remote code): GDLA attention (_gdla_from; full layers on the
    model's (YaRN) rotary table with the mscale^2 softmax factor, sliding-window layers on their own
    plain-RoPE table), PolyNorm-GLU dense / MoE FFNs (_motif_ffn_from), manifold-constrained
    hyper-connections (_motif_hc_from) or a plain residual, final RMSNorm over the stream mean."""
    tc, m = target.config, target.model
    assert getattr(tc, "attention_cls", "basic") == "gdla"
    a0 = m.layers[0].self_attn
    moe = [l.moe for l in m.layers if l.moe_enabled]
    widths = [l.mlp.intermediate_size for l in m.layers if not l.moe_enabled]
    widths += [x.shared_experts.intermediate_size for x in moe if x.shared_experts is not None]
    mhc = bool(getattr(tc, "mhc_enabled", False))
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers, heads=a0.num_heads,
                head_dim=a0.head_dim, attn_norm_feats=a0.q_lora_rank + a0.kv_lora_rank,
                experts=moe[0].num_experts if moe else 1, expert_ff=moe[0].experts.intermediate_dim if moe else 1,
                top_k=moe[0].experts_top_k if moe else 1, shared_ff=max(widths + [1]),
                hc_mult=tc.mhc_expansion_rate if mhc else 0,
                hc_iters=getattr(tc, "mhc_sinkhorn_iters", 20) if mhc else 1)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    for ul, tl in zip(u.layers, m.layers):
        at = tl.self_attn
        _norm_from(ul.norm_a, tl.input_layernorm)
        _norm_from(ul.norm_f, tl.post_attention_layernorm)
        rot = at.swa_rotary_emb if at.swa_rotary_emb is not None else m.rotary_emb
        assert float(rot.attention_scaling) == 1.0
        _gdla_from(ul.attn, at, rot.inv_freq)
        _motif_ffn_from(ul.ffn, tl.moe if tl.moe_enabled else tl.mlp, tc)
        if mhc:
            _motif_hc_from(ul.hc_a, tl.mhc_attn); _motif_hc_from(ul.hc_f, tl.mhc_ffn)
    return u


@torch.no_grad()
def from_cohere(target):
    """Cohere (Command-R / R+): a PARALLEL block, x + attn(ln x) + mlp(ln x) with ONE input
    LayerNorm (mean-centred, weight only), SwiGLU, interleaved RoPE pairs (2i, 2i+1), tied
    embeddings and a logit_scale. The host runs it with par = 1 (the FFN reads the layer input
    through norm_f = the same LayerNorm), the pairs permuted onto its (i, i + hd/2) layout and
    logit_scale folded into lm_head. use_qk_norm: per-head LayerNorms with their own weights per
    head (qk_centre)."""
    tc, m = target.config, target.model
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, "ln", logit_scale=float(tc.logit_scale))
    rope = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm, "ln")
        _norm_any(ul.norm_f, tl.input_layernorm, "ln")
        ul.par.fill_(1.0)
        qk = getattr(a, "use_qk_norm", False)
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, layout="interleave", q_norm=a.q_norm if qk else None,
                  k_norm=a.k_norm if qk else None, qk_kind="ln")
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def _mamba1_from(s: USSM, mx):
    """Mamba-1 mixer -> the gated-delta SSM block (one head per channel, dk = state, dv = 1)."""
    di, N, Rr = mx.intermediate_size, mx.ssm_state_size, mx.time_step_rank
    D = s.D                                                     # u = [x | z]
    _cp(s.in_proj.weight, mx.in_proj.weight)
    if mx.in_proj.bias is not None:
        _cp(s.in_proj.bias, mx.in_proj.bias)
    _cp(s.conv_w[:di], mx.conv1d.weight[:, 0])
    if mx.conv1d.bias is not None:
        _cp(s.conv_b[:di], mx.conv1d.bias)
    s.act_silu[:di] = 1
    cx = slice(D, D + di)                                       # post-conv x inside r = [u, c']
    W = mx.x_proj.weight.detach().to(F64)
    Wdt, WB, WC = W[:Rr], W[Rr:Rr + N], W[Rr + N:]
    s.pk.weight[:, cx] = WB.repeat(di, 1)
    s.pq.weight[:, cx] = WC.repeat(di, 1)
    s.pv.weight[:, cx] = torch.eye(di, dtype=F64)
    s.p1.weight[:, cx] = (mx.dt_proj.weight.detach().to(F64) @ Wdt).repeat_interleave(N, 0)
    s.p1.bias.copy_(mx.dt_proj.bias.detach().to(F64).repeat_interleave(N))
    s.A1.copy_(mx.A_log.detach().to(F64).exp())
    s.cw[0] = 1                                                 # write = dt
    s.Dskip[:, 0] = mx.D.detach().to(F64)
    s.pz.weight[:, di:2 * di] = torch.eye(di, dtype=F64)        # gate z from the pre-conv stream
    s.gate[0] = 1                                               # silu(z)
    s.out_coef[2] = 1                                           # plain y * gate
    _cp(s.out_proj.weight, mx.out_proj.weight)
    if mx.out_proj.bias is not None:
        _cp(s.out_proj.bias, mx.out_proj.bias)


@torch.no_grad()
def from_mamba(target):
    tc, bb = target.config, target.backbone
    di = tc.intermediate_size
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers, ssm_in=2 * di,
                conv_k=tc.conv_kernel, ssm_heads=di, dk=tc.state_size, dv=1)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, bb.embeddings.weight)
    _norm_from(u.norm, bb.norm_f)
    _cp(u.lm_head.weight, target.lm_head.weight)
    for ul, tl in zip(u.layers, bb.layers):
        _norm_from(ul.norm_s, tl.norm)
        _mamba1_from(ul.ssm, tl.mixer)
    return u


@torch.no_grad()
def _gdn_from(s: USSM, g):
    """Qwen3-Next gated-DeltaNet -> the SSM block (q/k shared across v-heads, l2 + scale,
    scalar decay, beta-gated delta and write, per-head RMS then silu gate)."""
    Hk, Hv, dk, dv = g.num_k_heads, g.num_v_heads, g.head_k_dim, g.head_v_dim
    kd, vd = Hk * dk, Hv * dv
    # u = [q | k | v | z | b | a] in canonical order
    if hasattr(g, "in_proj_qkv"):                               # Qwen3.5: separate projections
        _cp(s.in_proj.weight, torch.cat([g.in_proj_qkv.weight, g.in_proj_z.weight,
                                         g.in_proj_b.weight, g.in_proj_a.weight], 0))
    else:                                                       # Qwen3-Next: run its own split on W^T
        q, k, v, z, b, a = g.fix_query_key_value_ordering(g.in_proj_qkvz.weight.detach().T[None],
                                                          g.in_proj_ba.weight.detach().T[None])
        flat = lambda t: t.reshape(t.shape[1], -1).T
        _cp(s.in_proj.weight, torch.cat([flat(t) for t in (q, k, v, z, b, a)], 0))
    _cp(s.conv_w[:2 * kd + vd], g.conv1d.weight[:, 0])
    if g.conv1d.bias is not None:
        _cp(s.conv_b[:2 * kd + vd], g.conv1d.bias)
    s.act_silu[:2 * kd + vd] = 1
    D, rep = s.D, Hv // Hk
    khead = torch.arange(Hv).repeat_interleave(dk) // rep * dk + torch.arange(dk).repeat(Hv)
    rows = torch.arange(Hv * dk)
    s.pq.weight[rows, D + khead] = 1
    s.pk.weight[rows, D + kd + khead] = 1
    s.q_coef.copy_(torch.tensor([0.0, dk ** -0.5], dtype=F64))
    s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    s.pv.weight[:, D + 2 * kd: D + 2 * kd + vd] = torch.eye(vd, dtype=F64)
    s.pz.weight[:, 2 * kd + vd: 2 * kd + 2 * vd] = torch.eye(vd, dtype=F64)
    zb = 2 * kd + 2 * vd
    s.pb.weight[:, zb: zb + Hv] = torch.eye(Hv, dtype=F64)
    s.p1.weight[:, zb + Hv: zb + 2 * Hv] = torch.eye(Hv, dtype=F64).repeat_interleave(dk, 0)
    s.p1.bias.copy_(g.dt_bias.detach().to(F64).repeat_interleave(dk))
    s.A1.copy_(g.A_log.detach().to(F64).exp()[:, None].expand(Hv, dk))
    s.cg.fill_(1)
    s.cw[1] = 1                                                 # write = beta
    s.avg.copy_(torch.block_diag(*[torch.full((dv, dv), 1 / dv, dtype=F64)] * Hv))
    s.norm_w.copy_(g.norm.weight.detach().to(F64).repeat(Hv))
    s.norm_eps.fill_(float(getattr(g.norm, "variance_epsilon", getattr(g.norm, "eps", 1e-6))))
    s.gate[0] = 1
    s.out_coef[0] = 1                                           # norm(y) * silu(z)
    _cp(s.out_proj.weight, g.out_proj.weight)


@torch.no_grad()
def _deltanet_from(s: USSM, g):
    """fla DeltaNet (fla 0.4.0 layers/delta_net.py, use_gate False: fla-hub/delta_net-*) -> the SSM block: silu short
    convs on q / k / v (q / k: qk_activation silu), l2-normed q (x dk^-0.5) and k, beta = sigmoid(b_proj x) (x 2 with
    allow_neg_eigval), the ungated delta rule S <- S - beta (S k) k^T + beta v k^T (no decay: a = 1), a per-head
    RMSNorm o_norm, o_proj."""
    assert not g.use_gate and g.qk_activation == "silu" and g.qk_norm == "l2" and g.use_short_conv
    H, dk, dv = g.num_heads, g.head_k_dim, g.head_v_dim
    kd, vd = H * dk, H * dv
    D = s.D
    b = g.b_proj.weight if g.use_beta else torch.zeros(H, g.hidden_size, dtype=F64)
    _cp(s.in_proj.weight, torch.cat([g.q_proj.weight, g.k_proj.weight, g.v_proj.weight, b], 0))   # u = [q | k | v | b]
    for off, cv in ((0, g.q_conv1d), (kd, g.k_conv1d), (2 * kd, g.v_conv1d)):
        n = cv.weight.shape[0]
        _cp(s.conv_w[off:off + n], cv.weight[:, 0])
        if cv.bias is not None:
            _cp(s.conv_b[off:off + n], cv.bias)
    s.act_silu[:2 * kd + vd] = 1
    s.pq.weight[:, D: D + kd] = torch.eye(kd, dtype=F64)
    s.pk.weight[:, D + kd: D + 2 * kd] = torch.eye(kd, dtype=F64)
    s.q_coef.copy_(torch.tensor([0.0, dk ** -0.5], dtype=F64))
    s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    s.pv.weight[:, D + 2 * kd: D + 2 * kd + vd] = torch.eye(vd, dtype=F64)
    if g.use_beta:
        s.pb.weight[:, 2 * kd + vd: 2 * kd + vd + H] = torch.eye(H, dtype=F64)   # sigmoid(b), from u (unconvolved)
    else:
        s.pb.bias.fill_(float("inf"))                                            # beta = 1
    two = 2.0 if g.allow_neg_eigval else 1.0
    s.cg.fill_(two)
    s.cw[1] = two                                                               # write = erase strength = beta
    s.avg.copy_(torch.block_diag(*[torch.full((dv, dv), 1 / dv, dtype=F64)] * H))
    s.norm_w.copy_(g.o_norm.weight.detach().to(F64).repeat(H))
    s.norm_eps.fill_(float(g.o_norm.eps))
    s.gate[4] = 1                                                               # no output gate: G = 1
    s.out_coef[0] = 1                                                           # norm(y)
    _cp(s.out_proj.weight, g.o_proj.weight)


@torch.no_grad()
def from_qwen3next(target):
    tc, m = target.config, target.model
    hd = tc.head_dim
    rot = int(hd * tc.partial_rotary_factor)
    perm = _partial_rope_perm(hd, rot)
    kd, vd = tc.linear_num_key_heads * tc.linear_key_head_dim, tc.linear_num_value_heads * tc.linear_value_head_dim
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers,
                heads=tc.num_attention_heads, head_dim=hd,
                ssm_in=2 * kd + 2 * vd + 2 * tc.linear_num_value_heads, conv_k=tc.linear_conv_kernel_dim,
                ssm_heads=tc.linear_num_value_heads, dk=tc.linear_key_head_dim, dv=tc.linear_value_head_dim,
                experts=tc.num_experts, expert_ff=tc.moe_intermediate_size, top_k=tc.num_experts_per_tok,
                shared_ff=tc.shared_expert_intermediate_size)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm, one_plus=True)
    _cp(u.lm_head.weight, target.lm_head.weight)
    for ul, tl in zip(u.layers, m.layers):
        if tl.layer_type == "linear_attention":
            _norm_from(ul.norm_s, tl.input_layernorm, one_plus=True)
            _gdn_from(ul.ssm, tl.linear_attn)
        else:
            _norm_from(ul.norm_a, tl.input_layernorm, one_plus=True)
            _attn_from(ul.attn, tl.self_attn, tc, m.rotary_emb.inv_freq, perm=perm, gated=True, qk_norm="one_plus")
        _norm_from(ul.norm_f, tl.post_attention_layernorm, one_plus=True)
        mlp, f = tl.mlp, ul.ffn
        _cp(f.router.weight, mlp.gate.weight)
        for e, ex in enumerate(mlp.experts):
            _cp(f.w_gate[e], ex.gate_proj.weight); _cp(f.w_up[e], ex.up_proj.weight)
            _cp(f.w_down[e], ex.down_proj.weight)
        f.renorm.fill_(1.0 if mlp.norm_topk_prob else 0.0)
        _dense_ffn_from(f, mlp.shared_expert)
        _cp(f.s_router.weight, mlp.shared_expert_gate.weight)
        f.s_coef.copy_(torch.tensor([0.0, 1.0]))
    return u


# --- SSM-block loaders for the other recurrent families (block-level) -----------------------

def ssm_config(d, D, K, H, dk, dv):
    return UConfig(vocab=1, d=d, L=1, ssm_in=D, conv_k=K, ssm_heads=H, dk=dk, dv=dv)


def _lag(s: USSM, ch, lag):
    """Make conv channel(s) `ch` output the stream delayed by `lag` steps (identity act)."""
    s.conv_w[ch] = 0
    s.conv_w[ch, s.K - 1 - lag] = 1
    s.act_lin[ch] = 1


@torch.no_grad()
def mamba2_block(mx):
    """Mamba-2 (SSD): scalar decay per head, B/C shared per group, D skip, rmsnorm(y*silu(z))."""
    I, N, nh, P, ng = mx.intermediate_size, mx.ssm_state_size, mx.num_heads, mx.head_dim, mx.n_groups
    cd = mx.conv_dim
    s = USSM(ssm_config(mx.hidden_size, I + cd + nh, mx.conv_kernel_size, nh, N, P))
    _cp(s.in_proj.weight, mx.in_proj.weight)                         # u = [z | xBC | dt]
    if mx.in_proj.bias is not None:
        _cp(s.in_proj.bias, mx.in_proj.bias)
    xbc = slice(I, I + cd)
    _cp(s.conv_w[xbc], mx.conv1d.weight[:, 0])
    if mx.conv1d.bias is not None:
        _cp(s.conv_b[xbc], mx.conv1d.bias)
    s.act_silu[xbc] = 1
    D = s.D
    cx, cB, cC = D + I, D + I + I, D + I + I + ng * N                # post-conv x | B | C
    grp = torch.arange(nh) // (nh // ng)
    for h in range(nh):
        rows = slice(h * N, (h + 1) * N)
        s.pk.weight[rows, cB + grp[h] * N: cB + (grp[h] + 1) * N] = torch.eye(N, dtype=F64)
        s.pq.weight[rows, cC + grp[h] * N: cC + (grp[h] + 1) * N] = torch.eye(N, dtype=F64)
        s.p1.weight[rows, I + cd + h] = 1
    s.pv.weight[:, cx: cx + I] = torch.eye(I, dtype=F64)
    s.p1.bias.copy_(mx.dt_bias.detach().to(F64).repeat_interleave(N))
    s.A1.copy_(mx.A_log.detach().to(F64).exp()[:, None].expand(nh, N))
    s.cw[0] = 1
    s.Dskip.copy_(mx.D.detach().to(F64)[:, None].expand(nh, P))
    s.pz.weight[:, :I] = torch.eye(I, dtype=F64)
    s.gate[0] = 1
    s.avg.fill_(1 / I)                                               # norm over the full width
    s.norm_w.copy_(mx.norm.weight.detach().to(F64))
    s.norm_eps.fill_(float(mx.norm.variance_epsilon))
    s.out_coef[1] = 1                                                # norm(y * silu(z))
    lo, hi = getattr(mx, "time_step_limit", (0.0, float("inf")))     # dt clamp (Nemotron-H: time_step_min)
    assert hi == float("inf"), hi
    s.dt_min.fill_(float(lo))
    _cp(s.out_proj.weight, mx.out_proj.weight)
    if mx.out_proj.bias is not None:
        _cp(s.out_proj.bias, mx.out_proj.bias)
    return s


@torch.no_grad()
def lfm2_block(sc):
    """LFM2 gated short conv  y = C * conv(B * x): shifted copies from the conv, products from
    memoryless (decay 0) heads, the sum from out_proj, C from the identity gate."""
    h, K = sc.conv.in_channels, sc.L_cache
    has_b = sc.conv.bias is not None
    H = h * K + (h if has_b else 0)
    s = USSM(ssm_config(sc.in_proj.in_features, 2 * K * h + h, K, H, 1, 1))
    Wb, Wc, Wx = sc.in_proj.weight.detach().to(F64).chunk(3, 0)
    bias = sc.in_proj.bias.detach().to(F64).chunk(3) if sc.in_proj.bias is not None else (None,) * 3
    s.in_proj.weight.copy_(torch.cat([Wb] * K + [Wx] * K + [Wc], 0))
    if bias[0] is not None:
        s.in_proj.bias.copy_(torch.cat([bias[0]] * K + [bias[2]] * K + [bias[1]], 0))
    D, cw = s.D, sc.conv.weight.detach().to(F64)[:, 0]               # (h, K)
    s.mem.zero_()                                                    # no memory: S_t = w v k^T
    s.cw[4] = 1
    s.gate[3] = 1                                                    # identity gate (C)
    s.out_coef[2] = 1
    Wout = sc.out_proj.weight.detach().to(F64)
    for j in range(K):
        _lag(s, slice(j * h, (j + 1) * h), j)                        # B_{t-j}
        _lag(s, slice((K + j) * h, (K + j + 1) * h), j)              # x_{t-j}
        heads = torch.arange(j * h, (j + 1) * h)
        s.pk.weight[heads, D + j * h + torch.arange(h)] = 1
        s.pv.weight[heads, D + (K + j) * h + torch.arange(h)] = 1
        s.pq.bias[heads] = cw[:, K - 1 - j]                          # conv tap for lag j
        s.pz.weight[heads, 2 * K * h + torch.arange(h)] = 1
        s.out_proj.weight[:, heads] = Wout
    if has_b:                                                        # conv bias, gated by C
        heads = torch.arange(h * K, H)
        s.pk.bias[heads] = 1; s.pv.bias[heads] = 1
        s.pq.bias[heads] = sc.conv.bias.detach().to(F64)
        s.pz.weight[heads, 2 * K * h + torch.arange(h)] = 1
        s.out_proj.weight[:, heads] = Wout
    if sc.out_proj.bias is not None:
        _cp(s.out_proj.bias, sc.out_proj.bias)
    return s


@torch.no_grad()
def minimax_block(la):
    """MiniMax lightning: silu(q,k,v), constant decay exp(-slope) per head, full-width RMSNorm,
    sigmoid output gate."""
    H, hd = la.num_attention_heads, la.head_dim
    d = la.qkv_proj.in_features
    s = USSM(ssm_config(d, 4 * H * hd, 1, H, hd, hd))
    qkv = la.qkv_proj.weight.detach().to(F64).view(H, 3, hd, d)
    s.in_proj.weight.copy_(torch.cat([qkv[:, 0].reshape(-1, d), qkv[:, 1].reshape(-1, d),
                                      qkv[:, 2].reshape(-1, d), la.output_gate.weight.detach().to(F64)], 0))
    n = H * hd
    s.conv_w[:3 * n, 0] = 1
    s.act_silu[:3 * n] = 1
    D = s.D
    s.pq.weight[:, D: D + n] = torch.eye(n, dtype=F64)
    s.pk.weight[:, D + n: D + 2 * n] = torch.eye(n, dtype=F64)
    s.pv.weight[:, D + 2 * n: D + 3 * n] = torch.eye(n, dtype=F64)
    s.A3.copy_(la.slope_rate.detach().to(F64).view(H, 1).expand(H, hd))
    s.cw[4] = 1
    s.pz.weight[:, 3 * n: 4 * n] = torch.eye(n, dtype=F64)
    s.gate[1] = 1                                                    # sigmoid(output_gate x)
    s.avg.fill_(1 / n)
    s.norm_w.copy_(la.norm.weight.detach().to(F64))
    s.norm_eps.fill_(float(la.norm.variance_epsilon))
    s.out_coef[0] = 1                                                # sigmoid(g) * norm(y)
    _cp(s.out_proj.weight, la.out_proj.weight)
    return s


@torch.no_grad()
def rwkv4_block(at):
    """RWKV-4 time-mix: token shift (lag-1 conv), WKV = num/den with the time_first bonus, both
    linear in four heads per channel (decaying + current-token, for num and den), sigma(r) gate."""
    h = at.key.in_features
    s = USSM(ssm_config(h, 6 * h, 2, 4 * h, 1, 1))
    ch = lambda i: slice(i * h, (i + 1) * h)
    rows = []
    for lin, mix in ((at.key, at.time_mix_key), (at.value, at.time_mix_value), (at.receptance, at.time_mix_receptance)):
        W, mu = lin.weight.detach().to(F64), mix.detach().to(F64).view(-1)
        rows += [W * mu, W * (1 - mu)]                               # current | previous token
    s.in_proj.weight.copy_(torch.cat(rows, 0))
    for i in range(6):
        _lag(s, ch(i), i % 2)
    D, e = s.D, torch.arange(h)
    num_r, num_c, den_r, den_c = (torch.arange(i * h, (i + 1) * h) for i in range(4))
    w = -at.time_decay.detach().to(F64).exp()                        # log-decay per channel
    for heads in (num_r, num_c, den_r, den_c):
        s.pk.bias[heads] = 1; s.pq.bias[heads] = 1
        s.pe.weight[heads, D + e] = 1; s.pe.weight[heads, D + h + e] = 1       # k_t
    for heads in (num_r, num_c):
        s.pv.weight[heads, D + 2 * h + e] = 1; s.pv.weight[heads, D + 3 * h + e] = 1   # v_t
    for heads in (den_r, den_c):
        s.pv.bias[heads] = 1
    s.A3[num_r, 0] = -w; s.A3[den_r, 0] = -w
    s.mem[num_c] = 0; s.mem[den_c] = 0
    s.cw[2] = 1                                                      # write = exp(k)
    u = at.time_first.detach().to(F64).view(-1)
    cur = u.exp() - (-w).exp()
    s.ratio_num[num_r, num_r] = (-w).exp(); s.ratio_num[num_c, num_r] = cur
    s.ratio_den[den_r, num_r] = (-w).exp(); s.ratio_den[den_c, num_r] = cur
    s.out_coef[3, num_r] = 1
    s.pz.weight[num_r, D + 4 * h + e] = 1; s.pz.weight[num_r, D + 5 * h + e] = 1  # receptance
    s.gate[1] = 1
    s.out_proj.weight[:, num_r] = at.output.weight.detach().to(F64)
    return s


@torch.no_grad()
def rglru_block(rb):
    """RecurrentGemma recurrent block: conv -> RG-LRU (decay exp(-8 softplus(c) sigma(r)),
    write sqrt(1-a^2) sigma(i), reset at t=0) -> * gelu_tanh(y-branch) -> linear_out."""
    lru, K = rb.lru_width, rb.conv1d_width
    g = rb.rg_lru
    s = USSM(ssm_config(rb.hidden_size, 2 * lru, K, lru, 1, 1))     # u = [x-branch | y-branch]
    s.in_proj.weight.copy_(torch.cat([rb.linear_x.weight, rb.linear_y.weight]).detach().to(F64))
    s.in_proj.bias.copy_(torch.cat([rb.linear_x.bias, rb.linear_y.bias]).detach().to(F64))
    _cp(s.conv_w[:lru], rb.conv_1d.weight[:, 0])
    _cp(s.conv_b[:lru], rb.conv_1d.bias)
    s.act_lin[:lru] = 1
    D = s.D
    blk = lambda W: torch.block_diag(*W.detach().to(F64).transpose(1, 2))   # x_h @ W_h -> rows
    s.p2.weight[:, D: D + lru] = blk(g.recurrent_gate_weight)
    s.p2.bias.copy_(g.recurrent_gate_bias.detach().to(F64).reshape(-1))
    s.A2[:, 0] = 8 * F.softplus(g.recurrent_param.detach().to(F64))
    s.pi.weight[:, D: D + lru] = blk(g.input_gate_weight)
    s.pi.bias.copy_(g.input_gate_bias.detach().to(F64).reshape(-1))
    s.cw[3] = 1
    s.reset.fill_(1)
    s.pk.bias.fill_(1); s.pq.bias.fill_(1)
    s.pv.weight[:, D: D + lru] = torch.eye(lru, dtype=F64)
    s.pz.weight[:, lru: 2 * lru] = torch.eye(lru, dtype=F64)
    s.gate[2] = 1                                                    # gelu_tanh(y-branch)
    s.out_coef[2] = 1
    _cp(s.out_proj.weight, rb.linear_out.weight)
    _cp(s.out_proj.bias, rb.linear_out.bias)
    return s


@torch.no_grad()
def mamba3_block(m):
    """Mamba-3 MIMO mixer. One host head per (head h, rank r, channel p) with dk = N, dv = 1:
    the state row S[p] is shared across ranks (writes are elementwise in p), so each host head
    carries k_p = sum_r' mimo_x[h,r',p] k_r', and reads with its own rank's q_r. B/C are RMS-normed
    knob features; decay -dt*max(heavy_tail(dd_A), floor); trapezoid writes sigmoid(trap)*dt now
    and (1 - sigmoid(trap))*dt on the previous token; q/k rotate by sum tanh(angles)*dt*pi."""
    H, R, N, P, G = m.num_heads, m.mimo_rank, m.ssm_state_size, m.head_dim, m.n_groups
    I, A = m.intermediate_size, m.num_rope_angles
    RGN = R * G * N
    Du = sum(m.split_sizes)
    Hs = H * R * P
    s = USSM(UConfig(vocab=1, d=m.hidden_size, L=1, ssm_in=Du, conv_k=1, ssm_heads=Hs, dk=N, dv=1,
                     ssm_norm_feats=2 * RGN))
    _cp(s.in_proj.weight, m.in_proj.weight)
    oz, ox, oB, oC = 0, I, 2 * I, 2 * I + RGN
    odt, oA, otr, oang = 2 * I + 2 * RGN, 2 * I + 2 * RGN + H, 2 * I + 2 * RGN + 2 * H, 2 * I + 2 * RGN + 3 * H
    s.pn.weight[:, oB: oB + 2 * RGN] = torch.eye(2 * RGN, dtype=F64)       # nb = [B blocks | C blocks]
    s.n_avg.copy_(torch.block_diag(*[torch.full((N, N), 1 / N, dtype=F64)] * (2 * R * G)))
    s.n_w.copy_(torch.cat([m.B_norm.weight.detach().to(F64).repeat(R * G), m.C_norm.weight.detach().to(F64).repeat(R * G)]))
    s.n_eps.fill_(float(m.B_norm.variance_epsilon))
    nb = 2 * Du
    mx, mz, mo = (t.detach().to(F64) for t in (m.mimo_x, m.mimo_z, m.mimo_o))
    Bb, Cb, Dp = m.B_bias.detach().to(F64), m.C_bias.detach().to(F64), m.D.detach().to(F64)
    Wout = m.out_proj.weight.detach().to(F64)
    eyeN = torch.eye(N, dtype=F64)
    for h in range(H):
        g = h // (H // G)
        for r in range(R):
            for p in range(P):
                hh = (h * R + r) * P + p
                rows = slice(hh * N, (hh + 1) * N)
                for r2 in range(R):
                    c0 = nb + (r2 * G + g) * N
                    s.pk.weight[rows, c0: c0 + N] = mx[h, r2, p] * eyeN
                s.pk.bias[rows] = (mx[h, :, p, None] * Bb[h]).sum(0)
                c0 = nb + RGN + (r * G + g) * N
                s.pq.weight[rows, c0: c0 + N] = eyeN
                s.pq.bias[rows] = Cb[h, r]
                s.pv.weight[hh, ox + h * P + p] = 1
                s.p1.weight[rows, odt + h] = 1
                s.p1.bias[rows] = float(m.dt_bias[h])
                s.pA.weight[hh, oA + h] = 1
                s.pb.weight[hh, otr + h] = 1
                for j in range(A):
                    s.pang.weight[hh * (N // 2) + j, oang + j] = 1
                s.Dskip[hh, 0] = Dp[h] * mx[h, r, p]
                s.pz.weight[hh, oz + h * P + p] = mz[h, r, p]
                s.norm_w[hh] = float(m.norm.weight[h * P + p])
                s.out_proj.weight[:, hh] = Wout[:, h * P + p] * mo[h, r, p]
    s.A4.fill_(1)
    s.a_floor.fill_(float(m.a_floor))
    s.cw[5] = 1
    s.cw_prev.fill_(1)
    s.rot_c[:, :A] = math.pi
    s.avg.copy_(torch.block_diag(*[torch.full((P, P), 1 / P, dtype=F64)] * (H * R)))
    s.norm_eps.fill_(float(m.norm.eps))
    s.gate[0] = 1
    s.out_coef[0] = 1                                                # norm_P(y) * w * silu(z)
    return s


def _offset_norm(dst: UNorm, src, offset):
    with torch.no_grad():
        dst.weight.copy_(offset + src.weight.detach().to(F64))
        dst.eps.fill_(float(src.variance_epsilon))


@torch.no_grad()
def _plamo2_mamba(s: USSM, mx, eps):
    nh, P, N, dtd, I = mx.num_heads, mx.hidden_size_per_head, mx.d_state, mx.dt_dim, mx.intermediate_size
    W = mx.in_proj.weight.detach().to(F64).view(nh, 2 * P, -1)        # per head [z | x]
    s.in_proj.weight.copy_(torch.cat([W[:, :P].reshape(I, -1), W[:, P:].reshape(I, -1)]))
    D = s.D
    _cp(s.conv_w[I:2 * I], mx.conv1d.weight[:, 0])
    s.act_silu[I:2 * I] = 1
    s.pn.weight[:, D + I: D + 2 * I] = mx.bcdt_proj.weight.detach().to(F64)   # nb = [B | C | dt]
    blk = lambda n: torch.full((n, n), 1 / n, dtype=F64)
    s.n_avg.copy_(torch.block_diag(blk(N), blk(N), blk(dtd)))
    s.n_w.copy_(torch.cat([mx.B_norm_weight, mx.C_norm_weight, mx.dt_norm_weight]).detach().to(F64))
    s.n_eps.fill_(float(eps))
    nb = 2 * D
    eyeN = torch.eye(N, dtype=F64)
    for h in range(nh):
        rows = slice(h * N, (h + 1) * N)
        s.pk.weight[rows, nb: nb + N] = eyeN
        s.pq.weight[rows, nb + N: nb + 2 * N] = eyeN
        s.p1.weight[rows, nb + 2 * N: nb + 2 * N + dtd] = mx.dt_proj.weight[h].detach().to(F64)
    s.p1.bias.copy_(mx.dt_bias.detach().to(F64).repeat_interleave(N))
    s.A1.copy_(mx.A_log.detach().to(F64).exp()[:, None].expand(nh, N))
    s.cw[0] = 1
    s.pv.weight[:, D + I: D + 2 * I] = torch.eye(I, dtype=F64)
    s.Dskip.copy_(mx.D.detach().to(F64)[:, None].expand(nh, P))
    s.pz.weight[:, :I] = torch.eye(I, dtype=F64)
    s.gate[0] = 1
    s.out_coef[2] = 1
    _cp(s.out_proj.weight, mx.out_proj.weight)


@torch.no_grad()
def _plamo2_attn(a: UAttention, at):
    H, hd, Hk = at.q_num_heads, at.qk_dim, at.k_num_heads
    q, k, v = at.qkv_proj.weight.detach().to(F64).split([at.q_proj_dim, at.k_proj_dim, at.v_proj_dim])
    a.q_proj.weight.copy_(q)
    a.k_proj.weight.copy_(_tile_kv(k, Hk, H, hd)); a.v_proj.weight.copy_(_tile_kv(v, Hk, H, hd))
    _cp(a.o_proj.weight, at.o_proj.weight)
    a.q_nw.copy_(at.q_weight.detach().to(F64))
    a.k_nw.copy_(at.k_weight.detach().to(F64).repeat_interleave(H // Hk, 0))
    a.qk_eps.fill_(1e-6)
    a.q_coef.copy_(torch.tensor([0.0, 1.0])); a.k_coef.copy_(torch.tensor([0.0, 1.0]))
    a.inv_freq.copy_(at.rotary_emb.inv_freq.detach().to(F64))
    a.scale.fill_(hd ** -0.5)
    if not getattr(at, "full_attn", True):     # sliding layers: swa_mask |t - s| <= attention_window_size (& causal)
        a.window.fill_(float(at.config.attention_window_size))


@torch.no_grad()
def _iquest_attn(a: UAttention, at):
    """IQuest-Q1's attention (IQuestAttention, vllm-iquest-q1 model.py / attention.py: read as text) -> UAttention:
    GQA q / k / v / o, per-head weighted RMS q_norm / k_norm, then RoPE on q and k; with enable_sink_attention a learned
    sink KEY sink_k (num_kv_heads, head_dim) of value 0, scored (R_t q) . s per query head (its KV group's s, not
    rotated) at the attention scale: UAttention.null_key (UConfig.sink_key). (Its sliding layers' window and partial
    rotary are not set here: no whole-model loader yet.)"""
    H, Hk, hd = at.num_heads, at.num_kv_heads, at.head_dim
    _cp(a.q_proj.weight, at.q_proj.weight)
    a.k_proj.weight.copy_(_tile_kv(_W(at.k_proj), Hk, H, hd)); a.v_proj.weight.copy_(_tile_kv(_W(at.v_proj), Hk, H, hd))
    _cp(a.o_proj.weight, at.o_proj.weight)
    a.q_nw.copy_(at.q_norm.weight.detach().to(F64).expand(H, hd))
    a.k_nw.copy_(at.k_norm.weight.detach().to(F64).expand(H, hd))
    a.qk_eps.fill_(float(at.q_norm.variance_epsilon))
    a.q_coef.copy_(torch.tensor([0.0, 1.0])); a.k_coef.copy_(torch.tensor([0.0, 1.0]))
    a.inv_freq.copy_(at.rotary_emb.inv_freq.detach().to(F64))
    a.scale.fill_(float(at.scaling))
    if getattr(at, "sink_k", None) is not None:
        a.null_key.copy_(at.sink_k.detach().to(F64).repeat_interleave(H // Hk, 0))


@torch.no_grad()
def from_plamo2(target):
    """PLaMo-2: Mamba-2 with RMS-normed B/C/dt (the normalized-feature branch) alternating with
    per-head-qk-normed attention, in sandwich-norm layers (pre + post norms, offset weights). The attention
    layers outside config.full_attention_idx slide: key s visible to query t iff t - s <= attention_window_size
    (_plamo2_attn sets the window; their rotary base is rope_local_theta, read off each layer's table)."""
    tc, m = target.config, target.model
    mx = next(l.mixer for l in m.layers.layers if l.is_mamba)
    V = max(m.embed_tokens.num_embeddings, target.lm_head.out_features)      # PLaMo pads the vocab
    c = UConfig(vocab=V, d=tc.hidden_size, L=tc.num_hidden_layers,
                heads=tc.num_attention_heads, head_dim=tc.hidden_size_per_head,
                ssm_in=2 * mx.intermediate_size, conv_k=mx.d_conv, ssm_heads=mx.num_heads,
                dk=mx.d_state, dv=mx.hidden_size_per_head,
                ssm_norm_feats=2 * mx.d_state + mx.dt_dim, shared_ff=tc.intermediate_size)
    u = UniversalElephantine(c)
    u.embed.weight[: m.embed_tokens.num_embeddings] = m.embed_tokens.weight.detach().to(F64)
    _offset_norm(u.norm, m.norm, m.norm.offset)
    u.lm_head.weight[: target.lm_head.out_features] = target.lm_head.weight.detach().to(F64)
    for ul, tl in zip(u.layers, m.layers.layers):
        i = 1 if tl.is_mamba else 0
        _offset_norm(ul.norm_s if tl.is_mamba else ul.norm_a, tl.pre_mixer_norm, tl.pre_mixer_norm.offset)
        _offset_norm(ul.post_s if tl.is_mamba else ul.post_a, tl.post_mixer_norm, tl.post_mixer_norm.offset)
        ul.post_coef[i] = torch.tensor([0.0, 1.0])
        if tl.is_mamba:
            _plamo2_mamba(ul.ssm, tl.mixer, tc.rms_norm_eps)
        else:
            _plamo2_attn(ul.attn, tl.mixer)
        _offset_norm(ul.norm_f, tl.pre_mlp_norm, tl.pre_mlp_norm.offset)
        _offset_norm(ul.post_f, tl.post_mlp_norm, tl.post_mlp_norm.offset)
        ul.post_coef[2] = torch.tensor([0.0, 1.0])
        g, up = tl.mlp.gate_up_proj.weight.detach().to(F64).chunk(2)
        ul.ffn.s_gate.weight.copy_(g); ul.ffn.s_up.weight.copy_(up)
        _cp(ul.ffn.s_down.weight, tl.mlp.down_proj.weight)
    return u


@torch.no_grad()
def from_hymba(target):
    """Hymba: attention and a Mamba-1 SSM run in PARALLEL on one input (seq = 0), each output
    full-width RMS-normed and averaged into one out_proj (the attention output-norm branch and
    the SSM's gate-then-norm branch, out_proj / 2 on each side); RMS-normed dt/B/C; q/k head 64
    vs v head 128 (q/k zero-padded onto the host's 128-dim heads); cross-layer KV sharing
    (the KV register); learned meta tokens (the prefix); sliding window on non-global layers
    with the meta tokens always visible (window + sinks)."""
    tc, m = target.config, target.model
    blk0 = m.layers[0].mamba
    H, Hk, I, N, K = blk0.num_attention_heads, blk0.num_key_value_heads, blk0.intermediate_size, blk0.ssm_state_size, blk0.conv_kernel_size
    A0, R = blk0.attn_hidden_size, blk0.time_step_rank
    kq, vh = A0 // H, I // H
    hd = max(kq, vh)
    perm = _partial_rope_perm(hd, kq)
    nmem = getattr(tc, "num_memory_tokens", 0) or 0
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers, heads=H, head_dim=hd,
                ssm_in=2 * I, conv_k=K, ssm_heads=I, dk=N, dv=1, ssm_norm_feats=R + 2 * N,
                shared_ff=tc.intermediate_size, n_prefix=nmem)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.final_layernorm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    if nmem:
        _cp(u.prefix, m.memory_tokens)
    producers = {g[0] for g in (tc.kv_reuse_group or [])}
    pad = lambda w, n: torch.cat([w.view(n, kq, -1), torch.zeros(n, hd - kq, w.shape[-1], dtype=w.dtype)], 1)
    for ul, tl in zip(u.layers, m.layers):
        b, a, s = tl.mamba, ul.attn, ul.ssm
        _norm_from(ul.norm_a, tl.input_layernorm)
        _norm_from(ul.norm_s, tl.input_layernorm)
        ul.seq.fill_(0)                                                    # parallel attn + SSM
        W = b.in_proj.weight.detach().to(F64)
        lat = b.latent_dim
        if b.reuse_kv:
            q, xw = W[:A0], W[A0:lat]
            a.kv_read.fill_(1)
        else:
            kd, vd = b.k_hidden_size, b.v_hidden_size
            q, kw, vw, xw = W[:A0], W[A0:A0 + kd], W[A0 + kd:A0 + kd + vd], W[A0 + kd + vd:lat]
            a.k_proj.weight.copy_(_tile_kv(pad(kw, Hk).reshape(Hk * hd, -1), Hk, H, hd).view(H, hd, -1)[:, perm].reshape(H * hd, -1))
            a.v_proj.weight.copy_(_tile_kv(vw, Hk, H, vh))
            if b.layer_idx in producers:
                a.kv_write.fill_(1)
        a.q_proj.weight.copy_(pad(q, H)[:, perm].reshape(H * hd, -1))
        inv = b.self_attn.rotary_emb.inv_freq.detach().to(F64)
        a.inv_freq.zero_(); a.inv_freq[: inv.numel()] = inv
        a.scale.fill_(kq ** -0.5)
        if tc.sliding_window is not None and b.layer_idx not in (tc.global_attn_idx or []):
            a.window.fill_(float(tc.sliding_window))
            a.n_sink.fill_(float(nmem))
        Wout = b.out_proj.weight.detach().to(F64)
        a.o_coef.copy_(torch.tensor([0.0, 1.0]))
        a.o_avg.fill_(1 / I)
        a.o_w.copy_(b.pre_avg_layernorm1.weight.detach().to(F64))
        a.o_eps.fill_(float(b.pre_avg_layernorm1.variance_epsilon))
        a.o_proj.weight.copy_(Wout / 2)
        # SSM: Mamba-1 on [x | gate] with normalized dt/B/C
        s.in_proj.weight.copy_(torch.cat([xw, W[lat: lat + I]]))
        D = s.D
        _cp(s.conv_w[:I], b.conv1d.weight[:, 0])
        if b.conv1d.bias is not None:
            _cp(s.conv_b[:I], b.conv1d.bias)
        s.act_silu[:I] = 1
        s.pn.weight[:, D: D + I] = b.x_proj[0].weight.detach().to(F64)      # nb = [dt | B | C]
        bl = lambda n: torch.full((n, n), 1 / n, dtype=F64)
        s.n_avg.copy_(torch.block_diag(bl(R), bl(N), bl(N)))
        s.n_w.copy_(torch.cat([b.dt_layernorm.weight, b.B_layernorm.weight, b.C_layernorm.weight]).detach().to(F64))
        s.n_eps.fill_(float(b.dt_layernorm.variance_epsilon))
        nb = 2 * D
        s.pk.weight[:, nb + R: nb + R + N] = torch.eye(N, dtype=F64).repeat(I, 1)
        s.pq.weight[:, nb + R + N: nb + R + 2 * N] = torch.eye(N, dtype=F64).repeat(I, 1)
        s.p1.weight[:, nb: nb + R] = b.dt_proj[0].weight.detach().to(F64).repeat_interleave(N, 0)
        s.p1.bias.copy_(b.dt_proj[0].bias.detach().to(F64).repeat_interleave(N))
        s.A1.copy_(b.A_log[0].detach().to(F64).exp())
        s.cw[0] = 1
        s.pv.weight[:, D: D + I] = torch.eye(I, dtype=F64)
        s.Dskip[:, 0] = b.D[0].detach().to(F64)
        s.pz.weight[:, I: 2 * I] = torch.eye(I, dtype=F64)
        s.gate[0] = 1
        s.avg.fill_(1 / I)
        s.norm_w.copy_(b.pre_avg_layernorm2.weight.detach().to(F64))
        s.norm_eps.fill_(float(b.pre_avg_layernorm2.variance_epsilon))
        s.out_coef[1] = 1                                                 # norm(y * silu(gate))
        s.out_proj.weight.copy_(Wout / 2)
        _norm_from(ul.norm_f, tl.pre_moe_layernorm)
        _dense_ffn_from(ul.ffn, tl.moe.experts[0])
    return u


@torch.no_grad()
def rwkv5_block(at, eps=1e-5):
    """RWKV-5 time-mix: per-head matrix state with per-key decay w = exp(-exp(time_decay)) and
    bonus u. With the host's read-after-write state S_t, out = S_t (r/w) + v (k . r(u - 1/w)): a
    decaying head plus a memoryless current-token head, SUMMED by the mix matrix (which also
    mean-centres for GroupNorm), then per-head RMS * w + b, times silu(gate)."""
    H, S = at.time_decay.shape
    h = at.key.in_features
    Hs, C = 2 * H, 2 * H * S
    s = USSM(UConfig(vocab=1, d=h, L=1, ssm_in=8 * h, conv_k=2, ssm_heads=Hs, dk=S, dv=S))
    rows = []
    for lin, mix in ((at.key, at.time_mix_key), (at.value, at.time_mix_value),
                     (at.receptance, at.time_mix_receptance), (at.gate, at.time_mix_gate)):
        W, mu = lin.weight.detach().to(F64), mix.detach().to(F64).view(-1)
        rows += [W * mu, W * (1 - mu)]                               # current | previous token
    s.in_proj.weight.copy_(torch.cat(rows, 0))
    for i in range(8):
        _lag(s, slice(i * h, (i + 1) * h), i % 2)
    D, A = s.D, h                                                    # A: attention width (= h here)
    stream = lambda j: (slice(D + 2 * j * A, D + (2 * j + 1) * A), slice(D + (2 * j + 1) * A, D + (2 * j + 2) * A))
    td = at.time_decay.detach().to(F64)                              # (H, S)
    inv_w = td.exp().exp().reshape(-1)                               # 1 / w per (head, key)
    u = at.time_faaaa.detach().to(F64).reshape(-1)
    for half, qscale in ((0, inv_w), (1, u - inv_w)):               # recurrent heads | current heads
        hs = slice(half * A, (half + 1) * A)
        for j, P in ((0, s.pk), (1, s.pv)):                          # k, v streams
            for sl in stream(j):
                P.weight[hs, sl] = torch.eye(A, dtype=F64)
        for sl in stream(2):                                         # q = r scaled per key
            s.pq.weight[hs, sl] = torch.diag(qscale)
        s.pz.weight[hs, stream(3)[0]] = torch.eye(A, dtype=F64)
        s.pz.weight[hs, stream(3)[1]] = torch.eye(A, dtype=F64)
    s.A3[:H] = td.exp()                                              # log a = -exp(time_decay)
    s.mem[H:] = 0
    s.cw[4] = 1
    blockavg = torch.block_diag(*[torch.full((S, S), 1 / S, dtype=F64)] * H)
    centre = torch.eye(A, dtype=F64) - blockavg
    s.mix.zero_()
    s.mix[:A, :A] = centre; s.mix[A:, :A] = centre                   # sum both halves, then centre
    s.avg.zero_(); s.avg[:A, :A] = blockavg
    s.norm_w[:A] = at.ln_x.weight.detach().to(F64)
    s.norm_b[:A] = at.ln_x.bias.detach().to(F64)
    s.norm_eps.fill_(eps)
    s.gate[0] = 1
    s.out_coef[0, :A] = 1                                            # (groupnorm(y) * w + b) * silu(g)
    s.out_proj.weight[:, :A] = at.output.weight.detach().to(F64)
    return s


@torch.no_grad()
def _fla_qkv(s: USSM, layer, conv: bool, kv_rep: int, silu_no_conv: bool = False):
    """q | k | v streams from an fla layer into u (optionally short-conv + silu), shared-KV repeat."""
    H, dk, dv = s.H, s.dk, s.dv
    Wq, Wk, Wv = (getattr(layer, n).weight.detach().double().cpu() for n in ("q_proj", "k_proj", "v_proj"))
    nq, nk, nv = Wq.shape[0], Wk.shape[0], Wv.shape[0]
    s.in_proj.weight[: nq + nk + nv] = torch.cat([Wq, Wk, Wv])
    D = s.D
    if conv:
        for off, cv in ((0, layer.q_conv1d), (nq, layer.k_conv1d), (nq + nk, layer.v_conv1d)):
            n = cv.weight.shape[0]
            s.conv_w[off: off + n] = cv.weight.detach().double().cpu()[:, 0]
            if cv.bias is not None:
                s.conv_b[off: off + n] = cv.bias.detach().double().cpu()
        s.act_silu[: nq + nk + nv] = 1
    else:
        s.conv_w[: nq + nk + nv, s.K - 1] = 1                          # identity tap
        (s.act_silu if silu_no_conv else s.act_lin)[: nq + nk + nv] = 1
    # head h reads q head h, k/v group h // kv_rep
    for h in range(H):
        g = h // kv_rep
        s.pq.weight[h * dk:(h + 1) * dk, D + h * dk: D + (h + 1) * dk] = torch.eye(dk, dtype=F64)
        s.pk.weight[h * dk:(h + 1) * dk, D + nq + g * dk: D + nq + (g + 1) * dk] = torch.eye(dk, dtype=F64)
        s.pv.weight[h * dv:(h + 1) * dv, D + nq + nk + g * dv: D + nq + nk + (g + 1) * dv] = torch.eye(dv, dtype=F64)
    return nq + nk + nv


def _fla_out(s: USSM, layer, norm, gate_rows, gate_idx):
    """per-head RMSNorm * w, then gate(g) (silu or sigmoid), then o_proj."""
    H, dv = s.H, s.dv
    s.avg.copy_(torch.block_diag(*[torch.full((dv, dv), 1 / dv, dtype=F64)] * H))
    s.norm_w.copy_(norm.weight.detach().double().cpu().repeat(H))
    s.norm_eps.fill_(float(norm.eps))
    s.gate[gate_idx] = 1
    s.out_coef[0] = 1
    s.out_proj.weight.copy_(layer.o_proj.weight.detach().double().cpu())


@torch.no_grad()
def gla_block(layer):
    """fla GatedLinearAttention: log-decay logsigmoid(gk)/16 = -softplus(-gk)/16 per key, write 1,
    q scaled dk^-1/2, per-head RMSNorm * silu(g)."""
    H, dk, dv = layer.num_heads, layer.head_k_dim, layer.head_v_dim
    conv = layer.use_short_conv
    nqkv = layer.key_dim + layer.key_dim_per_group + layer.value_dim_per_group
    Dg = layer.gk_proj[0].out_features
    s = USSM(ssm_config(layer.hidden_size, nqkv + Dg + layer.value_dim, layer.conv_size if conv else 1, H, dk, dv))
    n = _fla_qkv(s, layer, conv, layer.num_kv_groups)
    W1 = layer.gk_proj[0].weight.detach().double().cpu()
    s.in_proj.weight[n: n + Dg] = W1                                   # low-rank gate input
    s.in_proj.weight[n + Dg:] = layer.g_proj.weight.detach().double().cpu()
    W2, b2 = layer.gk_proj[1].weight.detach().double().cpu(), layer.gk_proj[1].bias.detach().double().cpu()
    kv_rep, kdg = layer.num_kv_groups, layer.key_dim_per_group
    for h in range(H):
        g = h // kv_rep
        rows = slice(h * dk, (h + 1) * dk)
        s.p1.weight[rows, n: n + Dg] = -W2[g * dk:(g + 1) * dk]       # softplus(-gk)
        s.p1.bias[rows] = -b2[g * dk:(g + 1) * dk]
    s.A1.fill_(1 / layer.gate_logit_normalizer)
    s.q_coef.copy_(torch.tensor([dk ** -0.5, 0.0], dtype=F64))
    s.cw[4] = 1
    s.pz.weight[:, n + Dg: n + Dg + layer.value_dim] = torch.eye(layer.value_dim, dtype=F64)
    _fla_out(s, layer, layer.g_norm_swish_gate, None, 0)
    return s


@torch.no_grad()
def retnet_block(layer):
    """fla MultiScaleRetention: RoPE on q/k (the constant-angle rotation), fixed decay
    1 - 2^(-5-h) per head, write 1, q scaled dk^-1/2, per-head RMSNorm * silu(g)."""
    H, dk, dv = layer.num_heads, layer.head_k_dim, layer.head_v_dim
    conv = layer.use_short_conv
    nqkv = layer.key_dim + layer.key_dim_per_group + layer.value_dim_per_group
    s = USSM(ssm_config(layer.hidden_size, nqkv + layer.value_dim, layer.conv_size if conv else 1, H, dk, dv))
    n = _fla_qkv(s, layer, conv, layer.num_kv_groups)
    s.in_proj.weight[n:] = layer.g_proj.weight.detach().double().cpu()
    s.A3.copy_(-(1 - 2.0 ** (-5.0 - torch.arange(H, dtype=F64))).log()[:, None].expand(H, dk))
    s.rot_k.copy_(layer.rotary.inv_freq.detach().double().cpu()[None].expand(H, dk // 2))
    s.q_coef.copy_(torch.tensor([dk ** -0.5, 0.0], dtype=F64))
    s.cw[4] = 1
    s.pz.weight[:, n: n + layer.value_dim] = torch.eye(layer.value_dim, dtype=F64)
    _fla_out(s, layer, layer.g_norm_swish_gate, None, 0)
    return s


@torch.no_grad()
def kda_block(layer):
    """fla KimiDeltaAttention: short conv + silu on q/k/v, per-channel decay
    -exp(A_log[h]) softplus(f(x) + dt_bias), beta-gated delta rule, l2 q/k, per-head RMSNorm *
    sigmoid(g(x))."""
    H, Hv, dk, dv = layer.num_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim
    conv = layer.use_short_conv
    nqkv = 2 * layer.key_dim + layer.value_dim
    Df, Dg = layer.f_proj[0].out_features, layer.g_proj[0].out_features
    s = USSM(ssm_config(layer.hidden_size, nqkv + Df + H + Dg, layer.conv_size if conv else 1, Hv, dk, dv))
    n = _fla_qkv(s, layer, conv, 1, silu_no_conv=True) if Hv == H else None
    assert n is not None, "num_v_heads > num_heads not mapped here"
    cpu = lambda t: t.detach().double().cpu()
    s.in_proj.weight[n: n + Df] = cpu(layer.f_proj[0].weight)
    s.in_proj.weight[n + Df: n + Df + H] = cpu(layer.b_proj.weight)
    s.in_proj.weight[n + Df + H:] = cpu(layer.g_proj[0].weight)
    s.p1.weight[:, n: n + Df] = cpu(layer.f_proj[1].weight)
    s.p1.bias.copy_(cpu(layer.dt_bias))
    s.A1.copy_(cpu(layer.A_log).exp()[:, None].expand(H, dk))
    s.pb.weight[:, n + Df: n + Df + H] = torch.eye(H, dtype=F64)
    s.cg.fill_(2.0 if layer.allow_neg_eigval else 1.0)
    s.cw[1] = 2.0 if layer.allow_neg_eigval else 1.0
    s.q_coef.copy_(torch.tensor([0.0, dk ** -0.5], dtype=F64)); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    s.pz.weight[:, n + Df + H: n + Df + H + Dg] = cpu(layer.g_proj[1].weight)
    s.pz.bias.copy_(cpu(layer.g_proj[1].bias))
    _fla_out(s, layer, layer.o_norm, None, 1)
    return s


@torch.no_grad()
def rwkv7_block(at):
    """RWKV-7 time-mix as one SSM block. u = [x | x] with lag 0 / lag 1 conv copies, so every
    token-shift lerp is a linear map of c' = [x_t | x_{t-1}]. Sigmoid features carry the low-rank
    gates (decay via tanh z = 2 sigmoid(2z) - 1, a, g, v_gate). H main heads run the generalized
    delta rule (pre-decay removal along kk, out along kk*a, write key k*(1+(a-1)k_a)); H memoryless
    heads carry the r.k.r_k bonus. GroupNorm on the main half (mix centres), bonus added after,
    both times g. Layer 0 writes the value register (v_first); later layers read it."""
    C, H, N = at.hidden_size, at.num_heads, at.head_dim
    cpu = lambda t: t.detach().to(F64)
    v1 = lambda t: t.detach().to(F64).reshape(-1)
    Dw, Da, Dg = at.w1.shape[1], at.a1.shape[1], at.g1.shape[1]
    first = at.layer_id == 0
    Dh = Dw + C + Dg + (0 if first else C)                          # [h_w | h_a | h_g | h_v]
    s = USSM(UConfig(vocab=1, d=C, L=1, ssm_in=2 * C, conv_k=2, ssm_heads=2 * H, dk=N, dv=N,
                     ssm_sig_feats=Dh))
    I = torch.eye(C, dtype=F64)
    s.in_proj.weight.copy_(torch.cat([I, I]))
    _lag(s, slice(0, C), 0); _lag(s, slice(C, 2 * C), 1)
    D = s.D
    cx = slice(D, D + 2 * C)                                        # c' = [x_t | x_{t-1}]
    L = lambda mu: torch.cat([torch.diag(1 - v1(mu)), torch.diag(v1(mu))], 1)   # (C, 2C) lerp
    W = lambda lin: lin.weight.detach().to(F64)
    # sigmoid features
    o = 0
    s.ph.weight[o:o + Dw, cx] = 2 * cpu(at.w1).T @ L(at.x_w); o += Dw       # h_w = sig(2 xw w1)
    s.ph.weight[o:o + C, cx] = (cpu(at.a1) @ cpu(at.a2)).T @ L(at.x_a)
    s.ph.bias[o:o + C] = v1(at.a0); oa = o; o += C                          # h_a = a
    s.ph.weight[o:o + Dg, cx] = cpu(at.g1).T @ L(at.x_g); og = o; o += Dg     # h_g
    if not first:
        s.ph.weight[o:o + C, cx] = (cpu(at.v1) @ cpu(at.v2)).T @ L(at.x_v)
        s.ph.bias[o:o + C] = v1(at.v0); ov = o; o += C                        # h_v = v_gate
    hb = 2 * D + s.Dn                                                       # h offset in r
    halves = (slice(0, C), slice(C, 2 * C))                                 # main | bonus heads
    rk = v1(at.r_k)
    for i, hs in enumerate(halves):
        s.pq.weight[hs, cx] = (torch.diag(rk) if i else I) @ W(at.receptance) @ L(at.x_r)
        s.pk.weight[hs, cx] = W(at.key) @ L(at.x_k)
        s.pv.weight[hs, cx] = W(at.value) @ L(at.x_v)
        s.pbeta.weight[hs, hb + oa: hb + oa + C] = I                        # beta = a
        s.pkap.weight[hs, cx] = torch.diag(v1(at.k_k)) @ W(at.key) @ L(at.x_k)
        s.pz.weight[hs, hb + og: hb + og + Dg] = cpu(at.g2).T               # g
        s.out_proj.weight[:, hs] = W(at.output)
        if not first:
            s.pvr.weight[hs, hb + ov: hb + ov + C] = I
    s.c_ka.copy_(v1(at.k_a).view(H, N).repeat(2, 1))
    s.c_rm[:H] = 1                                                          # removal on main heads
    # decay: log a = -e^-1/2 sigmoid(w0 + w2 tanh(.)) = -A2 sigmoid((w0 - 1 w2) + 2 h_w w2)
    s.A2[:H] = math.exp(-0.5)
    w2 = cpu(at.w2)
    s.p2.weight[:C, hb: hb + Dw] = 2 * w2.T
    s.p2.bias[:C] = v1(at.w0) - w2.sum(0)
    s.mem[H:] = 0
    s.cw[4] = 1
    blockavg = torch.block_diag(*[torch.full((N, N), 1 / N, dtype=F64)] * H)
    s.mix.zero_(); s.mix[:C, :C] = I - blockavg; s.mix[C:, C:] = I
    s.avg.zero_(); s.avg[:C, :C] = blockavg
    s.norm_w[:C] = at.ln_x.weight.detach().to(F64); s.norm_b[:C] = at.ln_x.bias.detach().to(F64)
    s.norm_eps.fill_(float(at.ln_x.eps))
    s.gate[3] = 1                                                           # identity gate (g)
    s.out_coef[0, :C] = 1; s.out_coef[2, C:] = 1
    (s.v_write if first else s.v_read).fill_(1)
    return s


LINEAR_TYPES = ("linear_attention", "agnes_delta_attention")


def _engram_cfg(target):
    """UConfig fields for an Engram-carrying target (AllGDN: `target.engrams`, one per hooked layer)."""
    eg = getattr(target, "engrams", None)
    if not eg:
        return {}
    e0 = next(iter(eg.values()))
    return dict(engram_slots=e0.slots, engram_orders=tuple(e0.orders), engram_heads=e0.heads,
                engram_dim=e0.table.embedding_dim, engram_conv=e0.conv.kernel_size[0])


def _memory_cfg(m):
    """UConfig fields for EdgeRock's independent memory (a `memory` module on its SWA layers)."""
    mem = next((l.memory for l in m.layers if hasattr(l, "memory")), None)
    if mem is None:
        return {}
    return dict(mem_attn=True, mem_kv_ff=mem.write_gate_proj.out_features, mem_sel_rank=mem.admission_down.out_features)


@torch.no_grad()
def _memory_from(ul, mem, inv_freq, perm):
    """EdgeRock IndependentGQAMemory -> the layer's parallel memory attention: its own (1 + w) input
    norm; q / k (1 + w) head norms + partial RoPE; keys and values written from z = u + SwiGLU(u);
    a key is readable from the chunk after its own (chunk = min(max_commit_chunk, window)) while it
    ranks in the top `capacity` admission scores up(silu(down u)) of all earlier chunks (per KV group);
    a zero null key (logit 0); per-head read gate 2 tanh(read_scale) sigmoid(<q, w> / sqrt(dim) + b)."""
    a, H, G, hd = ul.mem, mem.heads, mem.groups, mem.dim
    assert (a.H, a.hd) == (H, hd)
    _norm_from(ul.norm_m, mem.input_norm, one_plus=True)
    P = lambda w: w.view(-1, hd, *w.shape[1:])[:, perm].reshape(w.shape)
    W = lambda lin: lin.weight.detach().to(F64)
    a.q_proj.weight.copy_(P(W(mem.q_proj)))
    a.k_proj.weight.copy_(P(_tile_kv(W(mem.k_proj), G, H, hd)))
    a.v_proj.weight.copy_(_tile_kv(W(mem.v_proj), G, H, hd))
    a.o_proj.weight.copy_(W(mem.o_proj))
    for i, (w_dst, s_n, coef) in enumerate(((a.q_nw, mem.q_norm, a.q_coef), (a.k_nw, mem.k_norm, a.k_coef))):
        w_dst.copy_((1 + s_n.weight.detach().to(F64))[perm].expand(H, hd))
        a.qk_eps[i] = float(s_n.eps)
        coef.copy_(torch.tensor([0.0, 1.0]))
    a.kv_c.fill_(1.0)
    a.kv_gate.weight.copy_(W(mem.write_gate_proj)); a.kv_up.weight.copy_(W(mem.write_up_proj))
    a.kv_down.weight.copy_(W(mem.write_down_proj))
    a.sel_down.weight.copy_(W(mem.admission_down))
    a.sel_up.weight.copy_(W(mem.admission_up).repeat_interleave(H // G, 0))
    a.sel_cap.fill_(float(mem.capacity))
    a.sel_chunk.fill_(float(min(mem.max_chunk, mem.window)))
    a.null_logit.fill_(0.0)
    a.hg_c.copy_(2 * torch.tanh(mem.read_scale.detach().to(F64)))
    a.hg_w.copy_(mem.read_gate_weight.detach().to(F64)[:, perm] / math.sqrt(hd))
    a.hg_b.copy_(mem.read_gate_bias.detach().to(F64))
    a.inv_freq.zero_()
    a.inv_freq[: inv_freq.numel()] = inv_freq.to(F64)
    a.scale.fill_(hd ** -0.5)


@torch.no_grad()
def _engram_from(u, target):
    for i, e in (getattr(target, "engrams", None) or {}).items():
        he = u.layers[int(i)].engram
        assert list(e.PRIMES) == list(he.PRIMES)
        he.pad = e.pad
        _cp(he.table, e.table.weight)
        _cp(he.key.weight, e.key.weight); _cp(he.val.weight, e.val.weight)
        _cp(he.w_h, e.nh.w); _cp(he.w_k, e.nk.w)
        he.eps.copy_(torch.tensor([e.nh.eps, e.nk.eps], dtype=F64))
        _cp(he.conv_w, e.conv.weight[:, 0])
        he.c.fill_(1.0)


@torch.no_grad()
def from_qwen35(target):
    """Qwen3.5-family hybrids (Qwen3.5 / Diba / Agnes / dQwen3.5): Qwen3-Next's gated-DeltaNet +
    gated attention layout with separate in-projections and a dense (or MoE) MLP. Agnes adds a
    parallel SwiGLU, folded exactly into the shared expert by concatenating hidden channels.
    dQwen3.5 is the same body run as a diffusion LM: use causal=False (its gated-delta layers stay
    causal, as in the reference; only attention goes bidirectional). AllGDN-Engram (all layers
    gated-DeltaNet) adds hashed n-gram memories before chosen layers (UEngram); EdgeRock makes its
    attention layers sliding-window and adds a parallel score-selected memory read (mem_attn)."""
    tc, m = target.config, target.model
    tc = tc.get_text_config() if hasattr(tc, "get_text_config") else getattr(tc, "text_config", tc)
    m = getattr(m, "language_model", m)                         # MemT nests the text model
    hd = tc.head_dim
    rot = int(hd * getattr(tc, "partial_rotary_factor", None) or tc.rope_parameters["partial_rotary_factor"])
    perm = _partial_rope_perm(hd, rot)
    lt = lambda l: getattr(l, "layer_type", None) or getattr(l, "block_type")
    mixer = lambda l: next(getattr(l, n) for n in ("linear_attn", "delta_attn", "self_attn", "global_attn") if hasattr(l, n))
    g0 = next(mixer(l) for l in m.layers if lt(l) in LINEAR_TYPES)
    kd, vd = g0.num_k_heads * g0.head_k_dim, g0.num_v_heads * g0.head_v_dim
    mlp0 = m.layers[0].mlp
    moe = hasattr(mlp0, "experts")
    fused = moe and hasattr(mlp0.experts, "gate_up_proj")      # transformers 5: 3-D expert tensors
    par = next((getattr(mlp0, n) for n in ("parallel_mlp", "parallel", "parallel_ffn") if hasattr(mlp0, n)), None)
    main = next((getattr(mlp0, n) for n in ("main_mlp", "main", "mlp") if hasattr(mlp0, n)), mlp0)
    sff = (main.gate_proj.out_features + (par.gate_proj.out_features if par is not None else 0)) if not moe \
        else mlp0.shared_expert.gate_proj.out_features
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers, heads=tc.num_attention_heads,
                head_dim=hd, ssm_in=2 * kd + 2 * vd + 2 * g0.num_v_heads, conv_k=g0.conv_kernel_size,
                ssm_heads=g0.num_v_heads, dk=g0.head_k_dim, dv=g0.head_v_dim,
                experts=(mlp0.experts.num_experts if fused else len(mlp0.experts)) if moe else 1,
                expert_ff=(mlp0.experts.intermediate_dim if fused else mlp0.experts[0].gate_proj.out_features) if moe else 1,
                top_k=getattr(mlp0, "top_k", None) or mlp0.gate.top_k if moe else 1, shared_ff=sff,
                **_engram_cfg(target), **_memory_cfg(m))
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm, one_plus=True)
    _cp(u.lm_head.weight, target.lm_head.weight)
    _engram_from(u, target)
    for ul, tl in zip(u.layers, m.layers):
        if lt(tl) in LINEAR_TYPES:
            _norm_from(ul.norm_s, tl.input_layernorm, one_plus=True)
            _gdn_from(ul.ssm, mixer(tl))
        else:
            _norm_from(ul.norm_a, tl.input_layernorm, one_plus=True)
            _attn_from(ul.attn, mixer(tl), tc, m.rotary_emb.inv_freq, perm=perm, gated=True, qk_norm="one_plus")
            if hasattr(mixer(tl), "swa_window"):                    # EdgeRock: s >= t - W + 1
                ul.attn.window.fill_(float(mixer(tl).swa_window - 1))
        if hasattr(tl, "memory"):
            _memory_from(ul, tl.memory, m.rotary_emb.inv_freq, perm)
        _norm_from(ul.norm_f, tl.post_attention_layernorm, one_plus=True)
        mlp, f = tl.mlp, ul.ffn
        if moe:
            _cp(f.router.weight, mlp.gate.weight)
            if fused:
                g, up = mlp.experts.gate_up_proj.detach().to(F64).chunk(2, dim=1)
                f.w_gate.copy_(g); f.w_up.copy_(up); _cp(f.w_down, mlp.experts.down_proj)
            else:
                for e, ex in enumerate(mlp.experts):
                    _cp(f.w_gate[e], ex.gate_proj.weight); _cp(f.w_up[e], ex.up_proj.weight)
                    _cp(f.w_down[e], ex.down_proj.weight)
            f.renorm.fill_(1.0 if getattr(mlp, "norm_topk_prob", True) else 0.0)   # tf5 router always renorms
            _dense_ffn_from(f, mlp.shared_expert)
            _cp(f.s_router.weight, mlp.shared_expert_gate.weight)
            f.s_coef.copy_(torch.tensor([0.0, 1.0]))
        else:
            mains = [next((getattr(mlp, n) for n in ("main_mlp", "main", "mlp") if hasattr(mlp, n)), mlp)]
            mains += [getattr(mlp, n) for n in ("parallel_mlp", "parallel", "parallel_ffn") if hasattr(mlp, n)]
            f.s_gate.weight.copy_(torch.cat([x.gate_proj.weight for x in mains]).to(F64))
            f.s_up.weight.copy_(torch.cat([x.up_proj.weight for x in mains]).to(F64))
            f.s_down.weight.copy_(torch.cat([x.down_proj.weight for x in mains], 1).to(F64))
    return u


def _kda_shim(k, cfg):
    """Present a KDA module with differently named parts (AliceAI) under kda_block's names."""
    from types import SimpleNamespace as NS
    z = torch.zeros(k.g_b_proj.out_features)
    return NS(num_heads=k.num_k_heads, num_v_heads=k.num_v_heads, head_k_dim=k.head_k_dim,
              head_v_dim=k.head_v_dim, key_dim=k.key_dim, value_dim=k.value_dim, use_short_conv=True,
              conv_size=k.conv_kernel_size, hidden_size=k.q_proj.in_features,
              q_proj=k.q_proj, k_proj=k.k_proj, v_proj=k.v_proj,
              q_conv1d=k.q_conv1d, k_conv1d=k.k_conv1d, v_conv1d=k.v_conv1d,
              f_proj=[k.f_a_proj, k.f_b_proj], g_proj=[k.g_a_proj, NS(weight=k.g_b_proj.weight, bias=z)],
              b_proj=k.b_proj, A_log=k.a_log_bias, dt_bias=k.dt_bias,
              allow_neg_eigval=bool(getattr(cfg, "kda_allow_negative_eigenvalues", False)),
              o_norm=NS(weight=k.o_norm.weight, eps=k.o_norm.eps), o_proj=k.o_proj)


@torch.no_grad()
def from_alice(target):
    """AliceAI: KDA + gated-attention hybrid (zero-centered norms = (1 + w)) with a sigmoid top-k
    router (bias-corrected selection) and Block AttnRes (depth-softmax residual over block sums)."""
    tc, m = target.config, target.model
    hd = tc.head_dim
    perm = _partial_rope_perm(hd, int(hd * tc.partial_rotary_factor))
    k0 = next(l.linear_attn for l in m.layers if l.layer_type == "linear_attention")
    ex = m.layers[0].mlp.experts
    f = ex.gate_up_proj.shape[1] // 2
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers, heads=tc.num_attention_heads,
                head_dim=hd, ssm_in=2 * k0.key_dim + k0.value_dim + k0.f_a_proj.out_features + k0.num_k_heads
                + k0.g_a_proj.out_features, conv_k=k0.conv_kernel_size, ssm_heads=k0.num_v_heads,
                dk=k0.head_k_dim, dv=k0.head_v_dim, experts=ex.num_experts, expert_ff=f,
                top_k=m.layers[0].mlp.gate.top_k,
                shared_ff=m.layers[0].mlp.shared_expert.gate_proj.out_features,
                attnres_block=tc.block_attn_res_block_size)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm, one_plus=True)
    _cp(u.lm_head.weight, target.lm_head.weight)
    _cp(u.ar_q_out, m.attnres_final.res_proj.weight[0]); _cp(u.ar_w_out, m.attnres_final.res_norm_weight)
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        _norm_from(ul.norm_a, tl.input_layernorm, one_plus=True)
        _norm_from(ul.norm_s, tl.input_layernorm, one_plus=True)
        _norm_from(ul.norm_f, tl.post_attention_layernorm, one_plus=True)
        if tl.layer_type == "linear_attention":
            sh = kda_block(_kda_shim(tl.linear_attn, tc))
            sh.l2_eps.fill_(1e-24)                                   # F.normalize
            ul.ssm.load_state_dict(sh.state_dict())
        else:
            _attn_from(ul.attn, tl.self_attn, tc, m.rotary_emb.inv_freq, perm=perm, gated=True, qk_norm="one_plus")
        if i > 0:
            _cp(ul.ar_qa, tl.attn_res_proj.weight[0]); _cp(ul.ar_wa, tl.attn_res_norm_weight)
        _cp(ul.ar_qf, tl.mlp_res_proj.weight[0]); _cp(ul.ar_wf, tl.mlp_res_norm_weight)
        ul.ar_eps.fill_(float(tl.eps))
        mo, fl = tl.mlp, ul.ffn
        _cp(fl.router.weight, mo.gate.weight)
        fl.score_c.fill_(1.0); fl.renorm.fill_(1.0)
        if mo.gate.e_score_correction_bias is not None:
            _cp(fl.sel_bias, mo.gate.e_score_correction_bias)
        g, up = mo.experts.gate_up_proj.detach().to(F64).chunk(2, dim=1)
        fl.w_gate.copy_(g); fl.w_up.copy_(up); _cp(fl.w_down, mo.experts.down_proj)
        _dense_ffn_from(fl, mo.shared_expert)
        _cp(fl.s_router.weight, mo.shared_expert_gate.weight)
        fl.s_coef.copy_(torch.tensor([0.0, 1.0]))
    u.layers[0].ar_eps.fill_(float(m.attnres_final.eps))       # the final mix reuses layer 0's eps
    return u


def _mk3_mamba2_shim(mx, eps):
    from types import SimpleNamespace as NS
    return NS(intermediate_size=mx.d_inner, ssm_state_size=mx.d_state, num_heads=mx.nheads, head_dim=mx.headdim,
              n_groups=mx.ngroups, conv_dim=mx.conv_dim, hidden_size=mx.d_model, conv_kernel_size=mx.d_conv,
              in_proj=mx.in_proj, conv1d=mx.conv1d, dt_bias=mx.dt_bias, A_log=mx.A_log, D=mx.D,
              norm=NS(weight=mx.norm.weight, variance_epsilon=mx.norm.eps), out_proj=mx.out_proj)


@torch.no_grad()
def from_mini_k3(target):
    """Mini-K3 (Mamba-2 variants): Mamba-2 with per-group gated norm, NoPE MLA with q / kv latent
    RMSNorms (the attention normalized-latent branch), a SiTU-GLU dense layer, a latent MoE (sigmoid
    router with selection bias, experts on a down-projected latent, RMSNorm + up after routing,
    SiTU experts and shared experts), and Block AttnRes."""
    tc, m = target.config, target.model
    H, nope, rope, vd = tc.num_attention_heads, tc.qk_nope_head_dim, tc.qk_rope_head_dim, tc.v_head_dim
    qhd, ql, kl = nope + rope, tc.q_lora_rank, tc.kv_lora_rank
    hd = max(qhd, vd) + (max(qhd, vd) % 2)
    mx0 = next(l.self_attn for l in m.layers if l.is_mamba)
    moe0 = next(l.feed_forward for l in m.layers if l.is_moe)
    Dl, f = tc.routed_expert_hidden_size, tc.moe_intermediate_size
    dense0 = next((l.feed_forward for l in m.layers if not l.is_moe), None)
    sff = max(moe0.shared_experts.gate_proj.out_features, dense0.gate_proj.out_features if dense0 else 0)
    d = tc.hidden_size
    c = UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd,
                ssm_in=mx0.d_inner + mx0.conv_dim + mx0.nheads, conv_k=mx0.d_conv, ssm_heads=mx0.nheads,
                dk=mx0.d_state, dv=mx0.headdim, experts=tc.num_experts, expert_ff=f,
                top_k=tc.num_experts_per_token, shared_ff=sff, attnres_block=tc.attn_res_block_size,
                attn_norm_feats=ql + kl)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    _cp(u.ar_q_out, m.output_attn_res_proj.weight[0]); _cp(u.ar_w_out, m.output_attn_res_norm.weight)
    W = lambda lin: lin.weight.detach().to(F64)
    for ul, tl in zip(u.layers, m.layers):
        for un, tn in ((ul.norm_a, tl.input_layernorm), (ul.norm_s, tl.input_layernorm), (ul.norm_f, tl.post_attention_layernorm)):
            _norm_from(un, tn)
        _cp(ul.ar_qa, tl.self_attention_res_proj.weight[0]); _cp(ul.ar_wa, tl.self_attention_res_norm.weight)
        _cp(ul.ar_qf, tl.mlp_res_proj.weight[0]); _cp(ul.ar_wf, tl.mlp_res_norm.weight)
        ul.ar_eps.fill_(float(tl.mlp_res_norm.eps))
        if tl.is_mamba:
            s = mamba2_block(_mk3_mamba2_shim(tl.self_attn, tc.rms_norm_eps))
            g, I = tl.self_attn.ngroups, tl.self_attn.d_inner
            s.avg.copy_(torch.block_diag(*[torch.full((I // g, I // g), g / I, dtype=F64)] * g))
            ul.ssm.load_state_dict(s.state_dict())
        else:
            a, ml = ul.attn, tl.self_attn
            a.pn.weight.copy_(torch.cat([W(ml.q_a_proj), W(ml.kv_a_proj_with_mqa)[:kl]]))
            a.n_avg.copy_(torch.block_diag(torch.full((ql, ql), 1 / ql, dtype=F64), torch.full((kl, kl), 1 / kl, dtype=F64)))
            a.n_w.copy_(torch.cat([ml.q_a_layernorm.weight, ml.kv_a_layernorm.weight]).detach().to(F64))
            a.n_eps.fill_(float(ml.q_a_layernorm.eps))
            qb, kva, kvb = W(ml.q_b_proj), W(ml.kv_a_proj_with_mqa), W(ml.kv_b_proj)
            nq, nkv = slice(d, d + ql), slice(d + ql, d + ql + kl)
            for h in range(H):
                r = h * hd
                a.q_proj.weight[r: r + qhd, nq] = qb[h * qhd:(h + 1) * qhd]
                a.k_proj.weight[r: r + nope, nkv] = kvb[h * (nope + vd): h * (nope + vd) + nope]
                a.k_proj.weight[r + nope: r + qhd, :d] = kva[kl: kl + rope]
                a.v_proj.weight[r: r + vd, nkv] = kvb[h * (nope + vd) + nope:(h + 1) * (nope + vd)]
                a.g_proj.weight[r: r + vd, :d] = W(ml.g_proj)[h * vd:(h + 1) * vd]
                a.o_proj.weight[:, r: r + vd] = W(ml.o_proj)[:, h * vd:(h + 1) * vd]
            a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
            a.inv_freq.zero_()
            a.scale.fill_(qhd ** -0.5)
        fl, ff = ul.ffn, tl.feed_forward
        fl.c_situ.fill_(1.0)
        fl.situ_beta.copy_(torch.tensor([tc.activation_situ_beta, tc.activation_situ_linear_beta], dtype=F64))
        sh = ff.shared_experts if tl.is_moe else ff
        n = sh.gate_proj.out_features
        fl.s_gate.weight[:n] = W(sh.gate_proj); fl.s_up.weight[:n] = W(sh.up_proj)
        fl.s_down.weight[:, :n] = W(sh.down_proj)
        if tl.is_moe:
            rt = ff.router
            _cp(fl.router.weight, rt.weight); _cp(fl.sel_bias, rt.e_score_correction_bias)
            fl.score_c.fill_(1.0)
            fl.renorm.fill_(1.0 if (rt.renormalize and rt.top_k > 1) else 0.0)
            fl.route_scale.fill_(float(rt.routed_scaling_factor))
            Wd = W(ff.routed_expert_down_proj)
            for e, ex in enumerate(ff.experts):
                fl.w_gate[e] = W(ex.w1) @ Wd; fl.w_up[e] = W(ex.w3) @ Wd
                fl.w_down[e].zero_(); fl.w_down[e][:Dl] = W(ex.w2)
            fl.r_c.fill_(1.0)
            fl.r_avg.zero_(); fl.r_avg[:Dl, :Dl] = 1 / Dl
            fl.r_w.zero_(); fl.r_w[:Dl] = ff.routed_expert_norm.weight.detach().to(F64)
            fl.r_eps.fill_(float(ff.routed_expert_norm.eps))
            fl.r_up.weight.zero_(); fl.r_up.weight[:, :Dl] = W(ff.routed_expert_up_proj)
    u.layers[0].ar_eps.fill_(float(m.output_attn_res_norm.eps))
    return u


@torch.no_grad()
def deltanet_block(layer):
    """fla-style DeltaNet (Nemotron-Flash): short conv + silu on q/k/v, l2 q/k, q scaled dk^-1/2,
    beta = sigmoid(b_proj) (x2 with negative eigenvalues), plain delta rule (no decay), per-head
    RMSNorm, no output gate."""
    H, dk, dv = layer.num_heads, layer.head_k_dim, layer.head_v_dim
    conv = layer.use_short_conv
    nqkv = 2 * layer.key_dim + layer.value_dim
    s = USSM(ssm_config(layer.hidden_size, nqkv + H, layer.conv_size if conv else 1, H, dk, dv))
    n = _fla_qkv(s, layer, conv, 1, silu_no_conv=True)
    s.in_proj.weight[n: n + H] = layer.b_proj.weight.detach().double().cpu()
    s.pb.weight[:, n: n + H] = torch.eye(H, dtype=F64)
    beta = 2.0 if layer.allow_neg_eigval else 1.0
    s.cg.fill_(beta); s.cw[1] = beta
    s.q_coef.copy_(torch.tensor([0.0, dk ** -0.5], dtype=F64)); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    _fla_out(s, layer, layer.o_norm, None, 4)                     # constant gate
    return s


def mamba_ssm_mamba2_shim(mx):
    """Present an upstream mamba_ssm Mamba2 module (e.g. Nemotron-Flash's) under mamba2_block's names."""
    from types import SimpleNamespace as NS
    return NS(intermediate_size=mx.d_ssm, ssm_state_size=mx.d_state, num_heads=mx.nheads, head_dim=mx.headdim,
              n_groups=mx.ngroups, conv_dim=mx.conv1d.in_channels, hidden_size=mx.d_model,
              conv_kernel_size=mx.d_conv, in_proj=mx.in_proj, conv1d=mx.conv1d, dt_bias=mx.dt_bias,
              A_log=mx.A_log, D=mx.D, norm=NS(weight=mx.norm.weight, variance_epsilon=mx.norm.eps),
              out_proj=mx.out_proj)


def _phi3mamba_shim(mx):
    from types import SimpleNamespace as NS
    return NS(intermediate_size=mx.d_inner, ssm_state_size=mx.d_state, time_step_rank=mx.dt_rank,
              in_proj=mx.in_proj, conv1d=mx.conv1d, x_proj=mx.x_proj, dt_proj=mx.dt_proj, A_log=mx.A_log,
              D=mx.D, out_proj=mx.out_proj)


@torch.no_grad()
def from_phi4flash(target):
    """Phi-4-mini-flash (SambaY). Even layers Mamba-1; odd layers sliding-window DIFFERENTIAL attention
    (NoPE) in the first half. Layer L/2 is a Mamba that publishes its scan output (SSM output register),
    L/2+1 a full attention that publishes its K/V (KV register); then GMU layers
    out_proj(silu(in_proj x) * y_{L/2}) alternate with cross-attention reading that K/V. Each diff-head
    j becomes two host heads (q zero-padded to 2*hd, v = [v1 | v2]); the head mix forms
    o1 - lambda o2 before the grouped subln norm. LayerNorm everywhere (centre + bias)."""
    tc, m = target.config, target.model
    att = next(l.attn for l in m.layers if not l.use_mamba)
    mb = next(l.attn for l in m.layers if l.use_mamba and not l.attn.yoco_cross)
    H, Hk, hd0 = att.num_heads, att.num_key_value_heads, att.head_dim
    hd = 2 * hd0
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers, heads=H, head_dim=hd,
                ssm_in=2 * mb.d_inner, conv_k=mb.d_conv, ssm_heads=mb.d_inner, dk=mb.d_state, dv=1,
                shared_ff=tc.intermediate_size)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _ln_from(u.norm, m.final_layernorm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    g = H // Hk
    W = lambda t: t.detach().to(F64)
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        _ln_from(ul.norm_a, tl.input_layernorm); _ln_from(ul.norm_s, tl.input_layernorm)
        _ln_from(ul.norm_f, tl.post_attention_layernorm)
        gu, up = W(tl.mlp.fc1.weight).chunk(2)
        ul.ffn.s_gate.weight.copy_(gu); ul.ffn.s_up.weight.copy_(up); _cp(ul.ffn.s_down.weight, tl.mlp.fc2.weight)
        mx = tl.attn
        if tl.use_mamba:
            s = ul.ssm
            if mx.yoco_cross:                                       # GMU: silu(in_proj x) * y_{L/2}
                di = mx.d_inner
                s.in_proj.weight[:di] = W(mx.in_proj.weight)
                if mx.in_proj.bias is not None:
                    s.in_proj.bias[:di] = W(mx.in_proj.bias)
                s.pz.weight[:, :di] = torch.eye(di, dtype=F64)
                s.gate[0] = 1; s.out_coef[2] = 1; s.y_read.fill_(1)
                _cp(s.out_proj.weight, mx.out_proj.weight)
                if mx.out_proj.bias is not None:
                    _cp(s.out_proj.bias, mx.out_proj.bias)
            else:
                _mamba1_from(s, _phi3mamba_shim(mx))
                if mx.yoco_kv:
                    s.y_write.fill_(1)
            continue
        a, ia = ul.attn, mx.inner_cross_attn
        Wq = W(mx.Wqkv.weight); bq = W(mx.Wqkv.bias)
        nq = H * hd0
        q, qb = Wq[:nq].view(H, hd0, -1), bq[:nq].view(H, hd0)
        for h in range(H):                                          # host head h <- q head h
            a.q_proj.weight[h * hd: h * hd + hd0] = q[h]; a.q_proj.bias[h * hd: h * hd + hd0] = qb[h]
        if not mx.yoco_cross:
            nk = Hk * hd0
            k, kb = Wq[nq: nq + nk].view(Hk, hd0, -1), bq[nq: nq + nk].view(Hk, hd0)
            v, vb = Wq[nq + nk:].view(Hk, hd0, -1), bq[nq + nk:].view(Hk, hd0)
            for h in range(H):
                j, s_ = divmod(h, 2)
                mk = j // g                                         # kv diff-head for diff-head j
                kh = 2 * mk + s_
                a.k_proj.weight[h * hd: h * hd + hd0] = k[kh]; a.k_proj.bias[h * hd: h * hd + hd0] = kb[kh]
                for half in (0, 1):
                    r = h * hd + half * hd0
                    a.v_proj.weight[r: r + hd0] = v[2 * mk + half]; a.v_proj.bias[r: r + hd0] = vb[2 * mk + half]
            if i == tc.num_hidden_layers // 2 + 1:
                a.kv_write.fill_(1)
        else:
            a.kv_read.fill_(1)
        lam = (torch.exp((ia.lambda_q1 * ia.lambda_k1).sum().double())
               - torch.exp((ia.lambda_q2 * ia.lambda_k2).sum().double()) + ia.lambda_init).item()
        a.o_hmix.zero_()
        for j in range(H // 2):
            a.o_hmix[2 * j, 2 * j] = 1; a.o_hmix[2 * j + 1, 2 * j] = -lam
        a.o_coef.copy_(torch.tensor([0.0, 1.0]))
        a.o_avg.copy_(torch.block_diag(*[torch.full((hd, hd), 1 / hd, dtype=F64)] * H))
        a.o_w.zero_()
        for j in range(H // 2):
            a.o_w[2 * j * hd: (2 * j + 1) * hd] = W(ia.subln.weight) * (1 - ia.lambda_init)
        a.o_eps.fill_(float(getattr(ia.subln, "variance_epsilon", getattr(ia.subln, "eps", 1e-5))))
        Wo = W(mx.out_proj.weight)
        a.o_proj.weight.zero_()
        for j in range(H // 2):
            a.o_proj.weight[:, 2 * j * hd: (2 * j + 1) * hd] = Wo[:, 2 * j * hd0: (2 * j + 2) * hd0]
        _cp(a.o_proj.bias, mx.out_proj.bias)
        a.scale.fill_(hd0 ** -0.5)
        sw = mx.config.sliding_window[i] if mx.config.sliding_window is not None else None
        if sw is not None:
            a.window.fill_(float(sw - 1))
    return u


@torch.no_grad()
def rwkv6_block(at):
    """RWKV-6 time-mix. c' = [x_t | x_{t-1}] from lag-0 / lag-1 conv copies. ddlerp in three feature
    stages: h1 = sigmoid(2 xxx W1) (tanh = 2 sigmoid(2z) - 1), f_j = xx * (maa_j + m_j) (product
    features), h2 = sigmoid(2 (x + f_w) dW1) (decay LoRA). Decay log a = -exp(td) (exp branch).
    Main heads read S_{t-1} (pre-update readout); memoryless heads carry the u-bonus; GroupNorm on
    their sum (mix sums then centres); silu gate; output."""
    C = at.key.in_features
    H, S = at.time_faaaa.shape
    E = at.time_maa_w1.shape[1] // 5
    Dd = at.time_decay_w1.shape[1]
    s = USSM(UConfig(vocab=1, d=C, L=1, ssm_in=2 * C, conv_k=2, ssm_heads=2 * H, dk=S, dv=S,
                     ssm_sig_feats=5 * E, ssm_prod_feats=5 * C, ssm_sig2_feats=Dd))
    I = torch.eye(C, dtype=F64)
    s.in_proj.weight.copy_(torch.cat([I, I]))
    _lag(s, slice(0, C), 0); _lag(s, slice(C, 2 * C), 1)
    D = s.D
    xt, xp = slice(D, D + C), slice(D + C, D + 2 * C)              # x_t | x_{t-1} in r
    oh = 2 * D; of = oh + 5 * E; oh2 = of + 5 * C
    v1 = lambda t: t.detach().to(F64).reshape(-1)
    T_ = lambda t: t.detach().to(F64)
    mx = v1(at.time_maa_x)
    s.ph.weight[:, xt] = 2 * T_(at.time_maa_w1).T * (1 - mx)          # 2 W1^T L(maa_x)
    s.ph.weight[:, xp] = 2 * T_(at.time_maa_w1).T * mx
    W2 = T_(at.time_maa_w2)                                         # (5, E, C)
    maa = [v1(getattr(at, "time_maa_" + n)) for n in "wkvrg"]
    for j in range(5):
        rows = slice(j * C, (j + 1) * C)
        s.pfx.weight[rows, xt] = -I; s.pfx.weight[rows, xp] = I     # xx = x_{t-1} - x_t
        s.pfy.weight[rows, oh + j * E: oh + (j + 1) * E] = 2 * W2[j].T
        s.pfy.bias[rows] = maa[j] - W2[j].sum(0)
    fcol = lambda j: slice(of + j * C, of + (j + 1) * C)               # f_w, f_k, f_v, f_r, f_g
    dW1, dW2 = T_(at.time_decay_w1), T_(at.time_decay_w2)
    s.ph2.weight[:, xt] = 2 * dW1.T; s.ph2.weight[:, fcol(0)] = 2 * dW1.T
    s.p5.weight[:C, oh2: oh2 + Dd] = 2 * dW2.T
    s.p5.bias[:C] = v1(at.time_decay) - dW2.sum(0)
    s.A5[:H] = 1
    u = v1(at.time_faaaa)
    Wr, Wk, Wv, Wg = (T_(l.weight) for l in (at.receptance, at.key, at.value, at.gate))
    for half in (0, 1):
        hs = slice(half * C, (half + 1) * C)
        Q = torch.diag(u) @ Wr if half else Wr
        for P, Wm, j in ((s.pq, Q, 3), (s.pk, Wk, 1), (s.pv, Wv, 2), (s.pz, Wg, 4)):
            P.weight[hs, xt] = Wm; P.weight[hs, fcol(j)] = Wm
    s.mem[H:] = 0
    s.c_pre[:H] = 1
    s.cw[4] = 1
    blockavg = torch.block_diag(*[torch.full((S, S), 1 / S, dtype=F64)] * H)
    centre = I - blockavg
    s.mix.zero_(); s.mix[:C, :C] = centre; s.mix[C:, :C] = centre
    s.avg.zero_(); s.avg[:C, :C] = blockavg
    s.norm_w[:C] = T_(at.ln_x.weight); s.norm_b[:C] = T_(at.ln_x.bias); s.norm_eps.fill_(float(at.ln_x.eps))
    s.gate[0] = 1
    s.out_coef[0, :C] = 1
    s.out_proj.weight[:, :C] = T_(at.output.weight)
    return s


@torch.no_grad()
def from_granite_switch(target):
    """Granite-Switch: a dense Granite-4 attention model plus E LoRA adapters chosen PER TOKEN by a
    switch head (control tokens carry key +gain and value = adapter id; everything else key -gain,
    value 0; the adapter index is the rounded causal softmax-average), so an aLoRA adapter switches on
    mid-sequence. Control tokens are exchanged for substitute tokens before embedding (here: their
    embedding rows). Granite's embedding / residual / logit multipliers fold into the weights."""
    tc, m = target.config, target.model
    a0 = m.layers[0].self_attn
    H, Hk, hd = a0.num_heads, a0.num_key_value_heads, a0.head_dim
    E = tc.num_adapters
    r = a0.o_proj.lora_A.shape[2]
    sff = m.layers[0].shared_mlp.output_linear.base_layer.in_features
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=len(m.layers), heads=H, head_dim=hd,
                shared_ff=sff, lora_adapters=E, lora_rank=r)
    u = UniversalElephantine(c)
    W = lambda t: t.detach().to(F64)
    u.embed.weight.copy_(W(m.embed_tokens.weight) * tc.embedding_multiplier)
    for ctrl, sub in zip(tc.adapter_token_ids, tc.adapter_substitute_token_ids):
        u.embed.weight[ctrl] = u.embed.weight[sub]                     # token exchange
    _norm_from(u.norm, m.norm)
    u.lm_head.weight.copy_(W(target.lm_head.weight) / tc.logits_scaling)
    gain = float(m.switch.control_token_gain)
    u.sw_score.fill_(-gain)
    for i, ctrl in enumerate(tc.adapter_token_ids):
        u.sw_score[ctrl] = gain; u.sw_val[ctrl] = i + 1
    rm = tc.residual_multiplier
    for ul, tl in zip(u.layers, m.layers):
        a, at = ul.attn, tl.self_attn
        _norm_from(ul.norm_a, tl.input_layernorm); _norm_from(ul.norm_f, tl.post_attention_layernorm)
        qkv = at.qkv_proj
        Wq, Wk, Wv = W(qkv.base_layer.weight).split(list(qkv.output_slices))
        a.q_proj.weight.copy_(Wq)
        a.k_proj.weight.copy_(_tile_kv(Wk, Hk, H, hd)); a.v_proj.weight.copy_(_tile_kv(Wv, Hk, H, hd))
        for i, n in enumerate("qkv"):
            A, B = W(qkv.lora_A_slices[i])[:, 0], W(qkv.lora_B_slices[i])[:, 0]     # (E,r,in), (E,out,r)
            a.la[n].copy_(A)
            a.lb[n].copy_(B if n == "q" else torch.stack([_tile_kv(B[e], Hk, H, hd) for e in range(E)]))
        a.o_proj.weight.copy_(W(at.o_proj.base_layer.weight) * rm)
        a.la["o"].copy_(W(at.o_proj.lora_A)[:, 0]); a.lb["o"].copy_(W(at.o_proj.lora_B)[:, 0] * rm)
        _cp(a.inv_freq, m.rotary_emb.inv_freq)
        a.scale.fill_(float(at.scaling))
        f, ml = ul.ffn, tl.shared_mlp
        g, up = W(ml.input_linear.base_layer.weight).chunk(2)
        f.s_gate.weight.copy_(g); f.s_up.weight.copy_(up)
        f.s_down.weight.copy_(W(ml.output_linear.base_layer.weight) * rm)
        f.la["g"].copy_(W(ml.input_linear.lora_A_slices[0])[:, 0]); f.lb["g"].copy_(W(ml.input_linear.lora_B_slices[0])[:, 0])
        f.la["u"].copy_(W(ml.input_linear.lora_A_slices[1])[:, 0]); f.lb["u"].copy_(W(ml.input_linear.lora_B_slices[1])[:, 0])
        f.la["d"].copy_(W(ml.output_linear.lora_A)[:, 0]); f.lb["d"].copy_(W(ml.output_linear.lora_B)[:, 0] * rm)
    return u


def _gc_norm(dst: UNorm, src):
    """GigaChat-3.5 norms: RMSNorm or zero-centered (1 + w), optionally gated by a low-rank
    sigmoid self-gate scaled by `layernorm_gating_weight`."""
    with torch.no_grad():
        w = src.weight.detach().to(F64)
        zero_centered = not hasattr(src, "variance_epsilon")          # ZeroCentered* keep `eps`
        dst.weight.copy_(1 + w if zero_centered else w)
        dst.eps.fill_(float(getattr(src, "variance_epsilon", getattr(src, "eps", 1e-6))))
        if hasattr(src, "gate_up_projection"):
            dst.gate_c.fill_(float(src.layernorm_gating_weight))
            _cp(dst.g_up.weight, src.gate_up_projection.weight)
            _cp(dst.g_down.weight, src.gate_down_projection.weight)


@torch.no_grad()
def _gc_gdn(s: USSM, g):
    """GigaChat-3.5 gated-DeltaNet: Qwen3-Next's grouped qkvz layout; output gating per
    `linear_gating_type` (silu / sigmoid / zero-centered sigmoid gated RMSNorm, times a scale, or a
    bare sigmoid gate)."""
    from types import SimpleNamespace as NS
    gt, sc = g.linear_gating_type, float(g.linear_sigmoid_gate_scale)
    src = g if g.norm is not None else NS(**{n: getattr(g, n) for n in (
        "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim", "fix_query_key_value_ordering", "in_proj_qkvz",
        "in_proj_ba", "conv1d", "dt_bias", "A_log", "out_proj")}, norm=NS(weight=torch.ones(g.head_v_dim), eps=1e-6))
    _gdn_from(s, src)
    if gt == "gated_rmsnorm":
        return
    s.gate.zero_(); s.gate[1] = 1                                    # sigmoid(z)
    if gt == "sigmoid_gate":
        s.pz.weight.mul_(sc)
        s.out_coef.zero_(); s.out_coef[2] = 1
        return
    w = g.norm.weight.detach().to(F64)
    s.norm_w.copy_(((1 + w) if getattr(g.norm, "zero_centered", False) else w).repeat(g.num_v_heads) * sc)


@torch.no_grad()
def from_gigachat35(target):
    """GigaChat-3.5: gated-DeltaNet (Qwen3-Next qkvz layout, zero-centered sigmoid gated norm x 2)
    alternating with gated MLA (q / kv latent norms, interleaved RoPE with YaRN scaling, MLA alpha
    scaling), sandwich (pre + post) zero-centered GATED norms (the UNorm self-gate branch), a sigmoid
    top-k router with selection bias + routed scaling and a shared expert, hard SwiGLU clipping."""
    tc, m = target.config, target.model
    assert tc.n_group == 1 or tc.topk_group == tc.n_group, "group-limited routing is not a host branch"
    assert getattr(tc, "final_logit_softcapping", None) is None
    assert tc.layernorm_type in ("pre", "pre_post")
    H, nope, rope, vd = tc.num_attention_heads, tc.qk_nope_head_dim, tc.qk_rope_head_dim, tc.v_head_dim
    qhd, kl = nope + rope, tc.kv_lora_rank
    ql = tc.q_lora_rank or 0
    hd = max(qhd, vd) + (max(qhd, vd) % 2)
    d = tc.hidden_size
    g0 = next(l.self_attn for l in m.layers if l.layer_type == "linear_attention")
    moe0 = next((l.mlp for l in m.layers if hasattr(l.mlp, "experts")), None)
    dense0 = next((l.mlp for l in m.layers if not hasattr(l.mlp, "experts")), None)
    sff = max(moe0.shared_experts.gate_proj.out_features if moe0 else 0, dense0.gate_proj.out_features if dense0 else 0)
    gated = hasattr(m.norm, "gate_up_projection")
    gr = m.norm.gate_up_projection.out_features if gated else 0
    c = UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd,
                ssm_in=2 * g0.key_dim + 2 * g0.value_dim + 2 * g0.num_v_heads, conv_k=g0.conv_kernel_size,
                ssm_heads=g0.num_v_heads, dk=g0.head_k_dim, dv=g0.head_v_dim,
                experts=tc.n_routed_experts if moe0 else 1, expert_ff=tc.moe_intermediate_size,
                top_k=tc.num_experts_per_tok if moe0 else 1, shared_ff=sff,
                attn_norm_feats=ql + kl, norm_gate_rank=gr, attn_norm_gate_rank=(2 if ql else 1) * gr)
    u = UniversalElephantine(c)
    W = lambda lin: lin.weight.detach().to(F64)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _gc_norm(u.norm, m.norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    # host head layout: target rope dims (de-interleaved) on host RoPE pairs (i, i + hd/2), i < rope/2;
    # nope dims on the remaining (NoPE) dims
    r2, h2 = rope // 2, hd // 2
    rope_host = list(range(r2)) + [h2 + i for i in range(r2)]
    nope_host = [i for i in range(hd) if i not in rope_host][:nope]
    if tc.rope_interleave:
        rope_src = [nope + 2 * i for i in range(r2)] + [nope + 2 * i + 1 for i in range(r2)]
    else:
        rope_src = list(range(nope, qhd))
    att_s = float(m.rotary_emb.attention_scaling)                     # YaRN mscale on cos / sin
    for ul, tl in zip(u.layers, m.layers):
        lin = tl.layer_type == "linear_attention"
        i = 1 if lin else 0
        _gc_norm(ul.norm_s if lin else ul.norm_a, tl.input_layernorm)
        _gc_norm(ul.norm_f, tl.post_attention_layernorm)
        if tl._use_post_layernorm:
            _gc_norm(ul.post_s if lin else ul.post_a, tl.post_self_attn_layernorm)
            _gc_norm(ul.post_f, tl.post_feedforward_layernorm)
            ul.post_coef[i] = torch.tensor([0.0, 1.0]); ul.post_coef[2] = torch.tensor([0.0, 1.0])
        if lin:
            _gc_gdn(ul.ssm, tl.self_attn)
        else:
            a, ml = ul.attn, tl.self_attn
            assert ml.kv_a_proj_with_mqa.bias is None and ml.o_proj.bias is None
            aq, akv = float(ml.alpha_q), float(ml.alpha_kv)
            kva = W(ml.kv_a_proj_with_mqa)
            norms = ([ml.q_a_layernorm] if ql else []) + [ml.kv_a_layernorm]
            a.pn.weight.copy_(torch.cat(([W(ml.q_a_proj)] if ql else []) + [kva[:kl]]))
            blk = lambda n: torch.full((n, n), 1 / n, dtype=F64)
            a.n_avg.copy_(torch.block_diag(*([blk(ql)] if ql else []), blk(kl)))
            zc = lambda nm: nm.weight.detach().to(F64) + (0 if hasattr(nm, "variance_epsilon") else 1)
            a.n_w.copy_(torch.cat([zc(nm) for nm in norms]))
            a.n_eps.fill_(float(getattr(norms[0], "variance_epsilon", getattr(norms[0], "eps", 1e-6))))
            if gated:
                a.n_gate_c.fill_(float(norms[0].layernorm_gating_weight))
                a.n_up.weight.copy_(torch.block_diag(*[W(nm.gate_up_projection) for nm in norms]))
                a.n_down.weight.copy_(torch.block_diag(*[W(nm.gate_down_projection) for nm in norms]))
            nq, nkv = slice(d, d + ql), slice(d + ql, d + ql + kl)
            kvb = W(ml.kv_b_proj) * akv
            qsrc, qcols = (W(ml.q_b_proj) * aq, nq) if ql else (W(ml.q_proj) * aq, slice(0, d))
            for h in range(H):
                r0 = h * hd
                qh = qsrc[h * qhd:(h + 1) * qhd]
                kh = kvb[h * (nope + vd): h * (nope + vd) + nope]
                for j, hj in enumerate(nope_host):
                    a.q_proj.weight[r0 + hj, qcols] = qh[j]
                    a.k_proj.weight[r0 + hj, nkv] = kh[j]
                for src_row, hj in zip(rope_src, rope_host):
                    a.q_proj.weight[r0 + hj, qcols] = qh[src_row] * att_s
                    a.k_proj.weight[r0 + hj, :d] = kva[kl + src_row - nope] * att_s
                a.v_proj.weight[r0: r0 + vd, nkv] = kvb[h * (nope + vd) + nope:(h + 1) * (nope + vd)]
                if ml.gated_attention:
                    a.g_proj.weight[r0: r0 + vd, :d] = W(ml.gate_proj)[h * vd:(h + 1) * vd]
                a.o_proj.weight[:, r0: r0 + vd] = W(ml.o_proj)[:, h * vd:(h + 1) * vd]
            if ml.gated_attention:
                a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
            a.inv_freq.zero_()
            a.inv_freq[:r2] = m.rotary_emb.inv_freq.detach().to(F64)
            a.scale.fill_(float(ml.scaling))
        fl, mo = ul.ffn, tl.mlp
        if float(getattr(tc, "swiglu_limit", 0.0) or 0.0) > 0:
            fl.c_clip.fill_(1.0); fl.clip_lim.fill_(float(tc.swiglu_limit))
        sh = mo.shared_experts if hasattr(mo, "experts") else mo
        n = sh.gate_proj.out_features
        fl.s_gate.weight[:n] = W(sh.gate_proj); fl.s_up.weight[:n] = W(sh.up_proj)
        fl.s_down.weight[:, :n] = W(sh.down_proj)
        if hasattr(mo, "experts"):
            rt = mo.gate
            _cp(fl.router.weight, rt.weight); _cp(fl.sel_bias, rt.e_score_correction_bias)
            fl.score_c.fill_(1.0)
            fl.renorm.fill_(1.0 if rt.norm_topk_prob else 0.0)
            fl.route_scale.fill_(float(rt.routed_scaling_factor))
            for e, ex in enumerate(mo.experts):
                _cp(fl.w_gate[e], ex.gate_proj.weight); _cp(fl.w_up[e], ex.up_proj.weight)
                _cp(fl.w_down[e], ex.down_proj.weight)
            if mo.use_shared_expert_sigmoid:
                _cp(fl.s_router.weight, mo.shared_expert_sigmoid.weight)
                fl.s_coef.copy_(torch.tensor([0.0, 1.0]))
    return u


def _eff_w(lin):
    """A linear layer's effective weight (LFM3-RD's shared-core + LoRA layers merge exactly as the
    reference does at inference)."""
    if hasattr(lin, "_get_eval_weight"):
        return lin._get_eval_weight(F64, torch.device("cpu")).detach().to(F64)
    return lin.weight.detach().to(F64)


def _lfm_conv_shim(cv):
    from types import SimpleNamespace as NS
    W = _eff_w(cv.in_proj)
    b = getattr(cv.in_proj, "bias", None)
    return NS(conv=cv.conv, L_cache=getattr(cv, "L_cache", getattr(cv, "conv_kernel_size", None)),
              in_proj=NS(weight=W, bias=b, in_features=W.shape[1]),
              out_proj=NS(weight=_eff_w(cv.out_proj), bias=getattr(cv.out_proj, "bias", None)))


@torch.no_grad()
def _lfm_attn(a: UAttention, at, inv_freq):
    """LFM2 / LFM2-MoE / LFM3 attention: per-head q / k RMSNorm, GQA, RoPE (LFM3: zero-tail partial
    RoPE, a theta per layer type), LFM3's per-HEAD sigmoid output gate and sliding window."""
    H, hd = a.H, a.hd
    Wq, Wk, Wv = _eff_w(at.q_proj), _eff_w(at.k_proj), _eff_w(at.v_proj)
    Hk = Wk.shape[0] // hd
    a.q_proj.weight.copy_(Wq)
    a.k_proj.weight.copy_(_tile_kv(Wk, Hk, H, hd)); a.v_proj.weight.copy_(_tile_kv(Wv, Hk, H, hd))
    a.o_proj.weight.copy_(_eff_w(at.o_proj if hasattr(at, "o_proj") else at.out_proj))
    qn = at.q_norm if hasattr(at, "q_norm") else at.q_layernorm
    kn = at.k_norm if hasattr(at, "k_norm") else at.k_layernorm
    for i, (w_dst, s_n, coef) in enumerate(((a.q_nw, qn, a.q_coef), (a.k_nw, kn, a.k_coef))):
        w_dst.copy_(s_n.weight.detach().to(F64).expand(H, hd))
        a.qk_eps[i] = float(getattr(s_n, "variance_epsilon", getattr(s_n, "eps", 1e-6)))
        coef.copy_(torch.tensor([0.0, 1.0]))
    if hasattr(at, "gate_proj"):                                     # one sigmoid gate per head
        a.g_proj.weight.copy_(at.gate_proj.weight.detach().to(F64).repeat_interleave(hd, 0))
        a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
    if getattr(at, "sliding_window", None):                         # t - s < W
        a.window.fill_(float(at.sliding_window - 1))
    a.inv_freq.zero_()
    a.inv_freq[: inv_freq.numel()] = inv_freq.detach().to(F64)
    a.scale.fill_(float(at.scaling))


@torch.no_grad()
def from_lfm(target):
    """Liquid LFM family: LFM2 / LFM2-MoE / LFM3 / LFM3-RD. Gated short-conv layers (y = C conv(B x),
    any L_cache) are the lfm2_block weight setting of the SSM block; attention layers use per-head
    qk-norm (LFM3: per-head sigmoid gate, sliding window, a RoPE per layer type); LFM3's sandwich
    norms are the post-sublayer norms; LFM2-MoE's sigmoid router selects with an expert bias and
    renormalises by (sum + 1e-6). LFM3-RD (looped, recurrent depth) at its shipped manual setting
    (recurrent_loops == num_motifs) is its prelude, the motifs unrolled once with their shared-core
    + LoRA weights merged, the residual-snapshot rescale x <- p + step_scale (x - p) around them,
    and the coda."""
    tc, m = target.config, target.model
    rd = hasattr(m, "core")
    if rd:
        assert tc.reasoning_mode == "manual" and tc.recurrent_loops == tc.num_motifs, \
            "only the unrolled manual setting is a fixed weight setting (auto mode halts data-dependently)"
        layers = list(m.prelude) + [b for ph in m.core.phases for b in ph.blocks] + list(m.coda)
    else:
        layers = list(m.layers)
    att = lambda l: getattr(l, "self_attn", None)
    a0 = next(att(l) for l in layers if att(l) is not None)
    hd = a0.head_dim
    H = _eff_w(a0.q_proj).shape[0] // hd
    cv0 = next(l.conv for l in layers if att(l) is None)
    h, K = cv0.conv.in_channels, cv0.conv.kernel_size[0]
    Hs = h * K + (h if cv0.conv.bias is not None else 0)
    ffs = [l.feed_forward for l in layers]
    moe = next((f for f in ffs if hasattr(f, "experts")), None)
    dense = [f for f in ffs if not hasattr(f, "experts")]
    sff = max([f.w1.out_features for f in dense], default=1)
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=len(layers), heads=H, head_dim=hd,
                ssm_in=2 * K * h + h, conv_k=K, ssm_heads=Hs, dk=1, dv=1, shared_ff=sff,
                experts=moe.experts.num_experts if moe else 1,
                expert_ff=moe.experts.intermediate_dim if moe else 1,
                top_k=moe.gate.top_k if moe else 1)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.embedding_norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    rot = getattr(m, "rotary_emb", None) or getattr(m, "pos_emb")
    W = lambda lin: lin.weight.detach().to(F64)
    for ul, tl in zip(u.layers, layers):
        at = att(tl)
        i = 0 if at is not None else 1
        _norm_from(ul.norm_a if at is not None else ul.norm_s, tl.operator_norm)
        _norm_from(ul.norm_f, tl.ffn_norm)
        if getattr(tl, "use_sandwich_norm", False):
            _norm_from(ul.post_a if at is not None else ul.post_s, tl.post_operator_norm)
            _norm_from(ul.post_f, tl.post_ffn_norm)
            ul.post_coef[i] = torch.tensor([0.0, 1.0]); ul.post_coef[2] = torch.tensor([0.0, 1.0])
        if at is not None:
            lt = getattr(tc, "layer_types", [None])[tl.self_attn.layer_idx] if hasattr(tl.self_attn, "layer_idx") else None
            inv = getattr(rot, f"{lt}_inv_freq", None) if lt is not None else None
            _lfm_attn(ul.attn, at, inv if inv is not None else rot.inv_freq)
        else:
            ul.ssm.load_state_dict(lfm2_block(_lfm_conv_shim(tl.conv)).state_dict())
        f, ff = ul.ffn, tl.feed_forward
        if hasattr(ff, "experts"):
            _cp(f.router.weight, ff.gate.weight)
            f.score_c.fill_(1.0)
            if getattr(ff, "use_expert_bias", False):
                _cp(f.sel_bias, ff.expert_bias)
            if ff.gate.norm_topk_prob:
                f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-6)
            f.route_scale.fill_(float(ff.gate.routed_scaling_factor))
            g, up = ff.experts.gate_up_proj.detach().to(F64).chunk(2, dim=1)
            f.w_gate.copy_(g); f.w_up.copy_(up); _cp(f.w_down, ff.experts.down_proj)
        else:
            n = ff.w1.out_features
            f.s_gate.weight[:n] = W(ff.w1); f.s_up.weight[:n] = W(ff.w3); f.s_down.weight[:, :n] = W(ff.w2)
    if rd:                                                           # p + step_scale (x - p) around the core
        p0, p1 = tc.num_prelude_layers, tc.num_prelude_layers + tc.num_motifs * tc.motif_size
        u.layers[p0].res_write.fill_(1.0)
        c_ = float(m.rd_controller.step_scales[tc.recurrent_loops - 1])
        assert p1 < len(u.layers), "the rescale lands at the first coda layer"
        u.layers[p1].res_mix.fill_(1.0 - c_)
    return u


def _norm_chain(dst: UNorm, first, second=None):
    """dst <- first (one RMSNorm) or second(first(.)) (two RMSNorms in series, the pre stage)."""
    if second is None:
        _norm_from(dst, first)
        return
    with torch.no_grad():
        dst.pre_c.fill_(1.0); _cp(dst.pre_w, first.weight)
        dst.pre_eps.fill_(float(getattr(first, "variance_epsilon", getattr(first, "eps", 1e-6))))
    _norm_from(dst, second)


@torch.no_grad()
def from_ceno(target):
    """CENO (a Megatron-faithful Nemotron-H-style hybrid). Mamba-2 blocks (x + mamba(in_proj_norm(x)),
    y * silu(z) then a per-group RMSNorm) are mamba2_block. The other blocks REPLACE the residual
    stream by a norm of it, which the stream-norm branches carry: MoE / MLP blocks give
    x_ln + ffn(pre_mlp_norm(x_ln)) with x_ln = norm(x) (sn_in), attention blocks give
    pre_mlp_norm(x + attn(qkv_norm(norm(x)))) (two norms in series, then sn_out). Experts are
    ungated squared-ReLU MLPs; the router takes the top-k logits then a softmax over them (= global
    softmax renormalised over the top k). Attention is NoPE MHA/GQA."""
    tc, bb = target.config, target.backbone
    blocks = list(bb.layers)
    mx0 = next((b.mixer for b in blocks if b.block_type == "mamba"), None)
    at0 = next((b.mixer for b in blocks if b.block_type == "attention"), None)
    moe0 = next((b.mixer for b in blocks if b.block_type == "moe"), None)
    mlp0 = next((b.mixer for b in blocks if b.block_type == "mlp"), None)
    assert all(b.block_type in ("mamba", "attention", "moe", "mlp") for b in blocks)
    for e in ([moe0.experts[0]] if moe0 else []) + ([mlp0] if mlp0 else []):
        assert not e.use_swiglu and tc.mlp_hidden_act in ("relu2", "squared_relu"), "carried: relu^2 MLPs"
    H = at0.num_heads if at0 else 1
    hd = at0.head_dim if at0 else 2
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=len(blocks), heads=H, head_dim=hd,
                ssm_in=(mx0.intermediate_size + mx0.conv_dim + mx0.num_heads) if mx0 else 1,
                conv_k=mx0.conv_kernel_size if mx0 else 1, ssm_heads=mx0.num_heads if mx0 else 1,
                dk=mx0.ssm_state_size if mx0 else 1, dv=mx0.head_dim if mx0 else 1,
                experts=moe0.num_experts if moe0 else 1,
                expert_ff=moe0.experts[0].up_proj.out_features if moe0 else 1,
                top_k=moe0.top_k if moe0 else 1, shared_ff=mlp0.up_proj.out_features if mlp0 else 1,
                stream_norms=True)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, bb.embeddings.weight)
    _norm_from(u.norm, bb.norm_f)
    _cp(u.lm_head.weight, target.lm_head.weight)
    W = lambda lin: lin.weight.detach().to(F64)
    for ul, b in zip(u.layers, blocks):
        f = ul.ffn
        if b.block_type == "mamba":
            mx = b.mixer
            assert b.norm is None and mx.in_proj_norm is not None
            _norm_from(ul.norm_s, mx.in_proj_norm)
            s = mamba2_block(mx)
            g, I = mx.n_groups, mx.intermediate_size
            s.avg.copy_(torch.block_diag(*[torch.full((I // g, I // g), g / I, dtype=F64)] * g))
            ul.ssm.load_state_dict(s.state_dict())
        elif b.block_type == "attention":
            at = b.mixer
            _norm_chain(ul.norm_a, b.norm, at.qkv_norm)
            Hk = at.num_key_value_heads
            ul.attn.q_proj.weight.copy_(W(at.q_proj))
            ul.attn.k_proj.weight.copy_(_tile_kv(W(at.k_proj), Hk, H, hd))
            ul.attn.v_proj.weight.copy_(_tile_kv(W(at.v_proj), Hk, H, hd))
            ul.attn.o_proj.weight.copy_(W(at.o_proj))
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                bias = getattr(at, n).bias
                if bias is not None:
                    getattr(ul.attn, n).bias.copy_(bias.detach().to(F64) if n in ("q_proj", "o_proj")
                                                   else _tile_kv(bias.detach().to(F64), Hk, H, hd))
            ul.attn.inv_freq.zero_()                                 # NoPE
            ul.attn.scale.fill_(hd ** -0.5)
            _norm_from(ul.sn_out, b.pre_mlp_norm); ul.sn_c[1] = 1
        else:
            _norm_from(ul.sn_in, b.norm); ul.sn_c[0] = 1
            f.c_relu2.fill_(1.0)
            if b.block_type == "moe":
                mo = b.mixer
                _norm_from(ul.norm_f, b.pre_mlp_norm)
                _cp(f.router.weight, mo.gate.weight)
                if getattr(tc, "moe_router_pre_softmax", False):
                    f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
                else:
                    f.renorm.fill_(1.0)                              # softmax over the top-k logits
                for e, ex in enumerate(mo.experts):
                    _cp(f.w_up[e], ex.up_proj.weight); _cp(f.w_down[e], ex.down_proj.weight)
                    assert ex.up_proj.bias is None and ex.down_proj.bias is None
            else:
                ml = b.mixer
                _norm_chain(ul.norm_f, b.pre_mlp_norm, ml.fc1_norm)
                n = ml.up_proj.out_features
                f.s_up.weight[:n] = W(ml.up_proj); f.s_down.weight[:, :n] = W(ml.down_proj)
                f.s_gate.weight.zero_()
    return u


@torch.no_grad()
def _glm_kda(s: USSM, la):
    """GLM-5.3 KDA layer -> the SSM block: silu short conv on q / k / v, l2 q / k (q * dk^-1/2),
    per-channel decay log a = lower * sigmoid(exp(A_log) (f_b f_a x + dt_bias)) (or the softplus form),
    beta = sigmoid(b x), per-head RMSNorm * sigmoid(g_b g_a x)."""
    H, hd = la.num_heads, la.head_dim
    n = H * hd
    W = lambda lin: lin.weight.detach().to(F64)
    fg = la.forget_gate
    s.in_proj.weight.copy_(torch.cat([W(la.q_proj), W(la.k_proj), W(la.v_proj), W(fg.f_a_proj), W(la.b_proj),
                                      W(la.g_a_proj)]))
    _cp(s.conv_w[:3 * n], la.conv1d.weight[:, 0])
    s.act_silu[:3 * n] = 1
    D, eye = s.D, torch.eye(n, dtype=F64)
    s.pq.weight[:, D: D + n] = eye; s.pk.weight[:, D + n: D + 2 * n] = eye; s.pv.weight[:, D + 2 * n: D + 3 * n] = eye
    s.q_coef.copy_(torch.tensor([0.0, hd ** -0.5], dtype=F64)); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    fa, b0, ga = slice(3 * n, 3 * n + hd), 3 * n + hd, slice(3 * n + hd + H, 3 * n + 2 * hd + H)
    rate = fg.A_log.detach().to(F64).exp().repeat_interleave(hd)            # exp(A_log) per (head, channel)
    Wf, dtb = W(fg.f_b_proj), fg.dt_bias.detach().to(F64)
    if getattr(fg, "safe_gate_lower_bound", None) is not None:
        s.p2.weight[:, fa] = rate[:, None] * Wf; s.p2.bias.copy_(rate * dtb)
        s.A2.fill_(-float(fg.safe_gate_lower_bound))
    else:
        s.p1.weight[:, fa] = Wf; s.p1.bias.copy_(dtb)
        s.A1.copy_(rate.view(H, hd))
    s.pb.weight[:, b0: b0 + H] = torch.eye(H, dtype=F64)
    s.cg.fill_(1); s.cw[1] = 1
    s.pz.weight[:, ga] = W(la.g_b_proj)
    s.gate[1] = 1
    s.avg.copy_(torch.block_diag(*[torch.full((hd, hd), 1 / hd, dtype=F64)] * H))
    s.norm_w.copy_(la.o_norm.weight.detach().to(F64).repeat(H))
    s.norm_eps.fill_(float(la.o_norm.variance_epsilon))
    s.out_coef[0] = 1
    _cp(s.out_proj.weight, la.o_proj.weight)


def _hc_from(dst: UHC, src):
    with torch.no_grad():
        _cp(dst.fn, src.fn); _cp(dst.base, src.base); _cp(dst.scale, src.scale)
        dst.eps.copy_(torch.tensor([src.hc_eps, src.input_norm.eps], dtype=F64))
        dst.iters = src.hc_sinkhorn_iters


@torch.no_grad()
def from_glm5_next(target):
    """GLM-5.3-Flash (glm5_next): manifold-constrained hyper-connections (hc_mult residual streams,
    Sinkhorn-mixed; UHC), KDA linear layers (the SSM block), NoPE MLA with DeepSeek sparse attention
    whose k-pool indexer picks the visible keys (the DSA mask branch), a sigmoid top-k router with a
    selection bias, a shared expert, hard SwiGLU clipping."""
    tc = getattr(target.config, "text_config", target.config)
    m = getattr(target.model, "language_model", target.model)
    assert tc.n_group == 1 or tc.topk_group == tc.n_group, "group-limited routing is not a host branch"
    assert all(t == "full" for t in tc.indexer_types), "cross-layer shared top-k is not wired"
    H, nope, rope, vd = tc.num_attention_heads, tc.qk_nope_head_dim, tc.qk_rope_head_dim, tc.v_head_dim
    qhd, ql, kl, d = nope + rope, tc.q_lora_rank, tc.kv_lora_rank, tc.hidden_size
    hd = max(qhd, vd) + (max(qhd, vd) % 2)
    layers = list(m.layers)
    la0 = next(l.self_attn for l in layers if l.block_type == "linear_attention")
    at0 = next((l.self_attn for l in layers if l.block_type != "linear_attention"), None)
    moe0 = next((l.mlp for l in layers if hasattr(l.mlp, "experts")), None)
    dense0 = next((l.mlp for l in layers if not hasattr(l.mlp, "experts")), None)
    sff = max(moe0.shared_experts.gate_proj.out_features if moe0 else 0, dense0.gate_proj.out_features if dense0 else 0)
    n, hk = la0.num_heads * la0.head_dim, la0.head_dim
    ix0 = at0.indexer if at0 is not None else None
    c = UConfig(vocab=tc.vocab_size, d=d, L=len(layers), heads=H, head_dim=hd, attn_norm_feats=ql + kl,
                ssm_in=3 * n + 2 * hk + la0.num_heads, conv_k=la0.conv_kernel_size, ssm_heads=la0.num_heads,
                dk=hk, dv=hk, experts=moe0.experts.num_experts if moe0 else 1,
                expert_ff=moe0.experts.intermediate_dim if moe0 else 1,
                top_k=moe0.gate.top_k if moe0 else 1, shared_ff=sff, hc_mult=tc.hc_mult, hc_iters=tc.hc_sinkhorn_iters,
                dsa_heads=ix0.n_heads if ix0 is not None else 0, dsa_dim=ix0.head_dim if ix0 is not None else 1)
    u = UniversalElephantine(c)
    W = lambda lin: lin.weight.detach().to(F64)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    for ul, tl in zip(u.layers, layers):
        lin = tl.block_type == "linear_attention"
        _hc_from(ul.hc_a, tl.attn_hc); _hc_from(ul.hc_f, tl.ffn_hc)
        _norm_from(ul.norm_s if lin else ul.norm_a, tl.input_layernorm)
        _norm_from(ul.norm_f, tl.post_attention_layernorm)
        if lin:
            _glm_kda(ul.ssm, tl.self_attn)
        else:
            a, ml = ul.attn, tl.self_attn
            assert ml.kv_a_proj_with_mqa.bias is None and ml.o_proj.bias is None
            kva = W(ml.kv_a_proj_with_mqa)
            a.pn.weight.copy_(torch.cat([W(ml.q_a_proj), kva[:kl]]))
            blk = lambda k: torch.full((k, k), 1 / k, dtype=F64)
            a.n_avg.copy_(torch.block_diag(blk(ql), blk(kl)))
            a.n_w.copy_(torch.cat([ml.q_a_layernorm.weight, ml.kv_a_layernorm.weight]).detach().to(F64))
            a.n_eps.fill_(float(ml.q_a_layernorm.variance_epsilon))
            nq, nkv = slice(d, d + ql), slice(d + ql, d + ql + kl)
            qb, kvb = W(ml.q_b_proj), W(ml.kv_b_proj)
            for h in range(H):                                       # NoPE: k_rot is used unrotated
                r = h * hd
                a.q_proj.weight[r: r + qhd, nq] = qb[h * qhd:(h + 1) * qhd]
                a.k_proj.weight[r: r + nope, nkv] = kvb[h * (nope + vd): h * (nope + vd) + nope]
                if rope:
                    a.k_proj.weight[r + nope: r + qhd, :d] = kva[kl: kl + rope]
                a.v_proj.weight[r: r + vd, nkv] = kvb[h * (nope + vd) + nope:(h + 1) * (nope + vd)]
                a.o_proj.weight[:, r: r + vd] = W(ml.o_proj)[:, h * vd:(h + 1) * vd]
            a.inv_freq.zero_()
            a.scale.fill_(float(ml.scaling))
            ix = ml.indexer
            a.iq.weight[:, nq] = W(ix.wq_b)
            a.ik.weight[:, :d] = W(ix.wk)
            _cp(a.ik_w, ix.k_norm.weight); _cp(a.ik_b, ix.k_norm.bias); a.ik_eps.fill_(float(ix.k_norm.eps))
            a.igate.weight[:, :d] = ix.index_kpool_compress_gate.detach().to(F64)
            a.i_ape = nn.Parameter(ix.index_kpool_compress_ape.detach().to(F64).clone())
            a.iw.weight[:, :d] = W(ix.weights_proj)
            a.dsa_pool.fill_(float(ix.index_kpool)); a.dsa_topk.fill_(float(ix.index_topk))
            a.dsa_tail.fill_(1.0 if ix.index_kpool_always_select_tail else 0.0)
        f, mo = ul.ffn, tl.mlp
        lim = float(getattr(tc, "swiglu_limit", 0.0) or 0.0)
        if lim > 0:
            f.c_clip.fill_(1.0); f.clip_lim.fill_(lim)
        sh = mo.shared_experts if hasattr(mo, "experts") else mo
        k = sh.gate_proj.out_features
        f.s_gate.weight[:k] = W(sh.gate_proj); f.s_up.weight[:k] = W(sh.up_proj); f.s_down.weight[:, :k] = W(sh.down_proj)
        if hasattr(mo, "experts"):
            rt = mo.gate
            _cp(f.router.weight, rt.weight); _cp(f.sel_bias, rt.e_score_correction_bias)
            f.score_c.fill_(1.0)
            if rt.norm_topk_prob:
                f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
            f.route_scale.fill_(float(rt.routed_scaling_factor))
            g, up = mo.experts.gate_up_proj.detach().to(F64).chunk(2, dim=1)
            f.w_gate.copy_(g); f.w_up.copy_(up); _cp(f.w_down, mo.experts.down_proj)
    return u


def _dense_w(lin):
    """The effective weight of any linear-like module (e.g. a 4-bit dequantising one): W = lin(I)^T."""
    n = lin.in_features
    return (lin(torch.eye(n, dtype=F64)) - (lin.bias.detach().to(F64) if getattr(lin, "bias", None) is not None else 0)).T


@torch.no_grad()
def from_glm_superexpert(target):
    """GLM-5.3-SuperExpert-V2's own modeling code: a dense NoPE MHA decoder whose attention has NO
    causal mask (run the host with causal=False), 4-bit packed weights (dequantised: W = lin(I)^T),
    a dense softmax mixture over ALL experts (top-k = E, no renorm) plus a shared expert."""
    tc, m = target.config, target.model.language_model
    H, d = tc.num_attention_heads, tc.hidden_size
    hd = d // H
    moe0 = next((l.mlp for l in m.layers if hasattr(l.mlp, "super_experts")), None)
    dense0 = next((l.mlp for l in m.layers if not hasattr(l.mlp, "super_experts")), None)
    sh = [x for x in (dense0, moe0.shared_experts if moe0 is not None else None) if x is not None]
    c = UConfig(vocab=tc.vocab_size, d=d, L=len(m.layers), heads=H, head_dim=hd,
                experts=moe0.num_experts if moe0 else 1, expert_ff=moe0.intermediate_size if moe0 else 1,
                top_k=moe0.num_experts if moe0 else 1, shared_ff=max(x.gate_proj.out_features for x in sh))
    u = UniversalElephantine(c)
    emb = m.embed_tokens
    u.embed.weight.copy_(emb(torch.arange(emb.num_embeddings)).to(F64))
    _norm_from(u.norm, m.norm)
    u.lm_head.weight.copy_(_dense_w(target.lm_head))
    for ul, tl in zip(u.layers, m.layers):
        _norm_from(ul.norm_a, tl.input_layernorm); _norm_from(ul.norm_f, tl.post_attention_layernorm)
        a, at = ul.attn, tl.self_attn
        a.q_proj.weight.copy_(_dense_w(at.q_proj)); a.k_proj.weight.copy_(_dense_w(at.k_proj))
        a.v_proj.weight.copy_(_dense_w(at.v_proj)); a.o_proj.weight.copy_(_dense_w(at.o_proj))
        a.inv_freq.zero_(); a.scale.fill_(hd ** -0.5)
        f, ml = ul.ffn, tl.mlp
        s_ = ml.shared_experts if hasattr(ml, "super_experts") else ml
        if s_ is not None:
            k = s_.gate_proj.out_features
            f.s_gate.weight[:k] = _dense_w(s_.gate_proj); f.s_up.weight[:k] = _dense_w(s_.up_proj)
            f.s_down.weight[:, :k] = _dense_w(s_.down_proj)
        if hasattr(ml, "super_experts"):
            _cp(f.router.weight, ml.gate.weight)                     # softmax over all experts, no renorm
            for e, ex in enumerate(ml.super_experts):
                f.w_gate[e] = _dense_w(ex.gate_proj); f.w_up[e] = _dense_w(ex.up_proj); f.w_down[e] = _dense_w(ex.down_proj)
    return u


@torch.no_grad()
def _kda_lowrank_gate(s: USSM, *, qkv, conv_w, f_a, f_b, b, gate, A_log, dt_bias, lower, norm_w, norm_eps, out):
    """A KDA layer with fused q / k / v (one silu short conv), a low-rank forget gate f_b f_a x, a
    full-rank (or any linear) output gate; decay log a = lower sigmoid(exp(A_log) (f + dt_bias)) or
    -exp(A_log) softplus(f + dt_bias); beta = sigmoid(b x); per-head RMSNorm * sigmoid(gate)."""
    H, hd = A_log.numel(), f_a.shape[0]
    n = H * hd
    s.in_proj.weight.copy_(torch.cat([qkv, f_a, b, gate]))
    s.conv_w[:3 * n] = conv_w
    s.act_silu[:3 * n] = 1
    D, eye = s.D, torch.eye(n, dtype=F64)
    s.pq.weight[:, D: D + n] = eye; s.pk.weight[:, D + n: D + 2 * n] = eye; s.pv.weight[:, D + 2 * n: D + 3 * n] = eye
    s.q_coef.copy_(torch.tensor([0.0, hd ** -0.5], dtype=F64)); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    fa, b0, g0 = slice(3 * n, 3 * n + hd), 3 * n + hd, 3 * n + hd + H
    rate = A_log.exp().repeat_interleave(hd)
    if lower is not None:
        s.p2.weight[:, fa] = rate[:, None] * f_b; s.p2.bias.copy_(rate * dt_bias)
        s.A2.fill_(-float(lower))
    else:
        s.p1.weight[:, fa] = f_b; s.p1.bias.copy_(dt_bias)
        s.A1.copy_(rate.view(H, hd))
    s.pb.weight[:, b0: b0 + H] = torch.eye(H, dtype=F64)
    s.cg.fill_(1); s.cw[1] = 1
    s.pz.weight[:, g0: g0 + n] = eye
    s.gate[1] = 1
    s.avg.copy_(torch.block_diag(*[torch.full((hd, hd), 1 / hd, dtype=F64)] * H))
    s.norm_w.copy_(norm_w.repeat(H)); s.norm_eps.fill_(float(norm_eps))
    s.out_coef[0] = 1
    s.out_proj.weight.copy_(out)


@torch.no_grad()
def from_youtu(target):
    """Youtu-LLM (youtu_vl_moe), the text model: KDA layers (fused qkv + one short conv, low-rank
    forget gate with the lower-bound sigmoid decay, full-rank sigmoid output gate) and MLA layers
    (latent RMSNorms, Megatron-interleaved RoPE, a head-wise sigmoid gate), a dense first layer, then a
    sigmoid MoE whose experts are SELECTED on the raw logits minus the quantile-balancing bias (text
    row) and weighted by renormalised sigmoid scores x routed_scaling, plus a shared expert. Image /
    video tokens (vision tower, vision bias row) are not carried."""
    tc = target.config.get_text_config() if hasattr(target.config, "get_text_config") else target.config
    m = target.model.language_model
    layers = list(m.layers)
    W = lambda lin: lin.weight.detach().to(F64)
    k0 = next(l.self_attn for l in layers if l.is_linear_attn)
    a0 = next((l.self_attn for l in layers if not l.is_linear_attn), None)
    moe0 = next((l.mlp for l in layers if hasattr(l.mlp, "experts")), None)
    dense0 = next((l.mlp for l in layers if not hasattr(l.mlp, "experts")), None)
    assert moe0 is None or (moe0.fc1_latent_proj is None and moe0.gate.n_group == 1)
    H, nope, rope, vd = tc.num_attention_heads, tc.qk_nope_head_dim, tc.qk_rope_head_dim, tc.v_head_dim
    qhd, ql, kl, d = nope + rope, tc.q_lora_rank, tc.kv_lora_rank, tc.hidden_size
    hd = max(qhd, vd) + (max(qhd, vd) % 2)
    n = k0.num_heads * k0.head_dim
    sh = [x for x in ((moe0.shared_experts if moe0 else None), dense0) if x is not None]
    c = UConfig(vocab=tc.vocab_size, d=d, L=len(layers), heads=H, head_dim=hd, attn_norm_feats=ql + kl,
                ssm_in=4 * n + k0.head_dim + k0.num_heads, conv_k=k0.conv_size, ssm_heads=k0.num_heads,
                dk=k0.head_dim, dv=k0.head_dim, experts=len(moe0.experts) if moe0 else 1,
                expert_ff=moe0.experts[0].gate_proj.out_features if moe0 else 1,
                top_k=moe0.gate.top_k if moe0 else 1, shared_ff=max(x.gate_proj.out_features for x in sh))
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    r2, h2 = rope // 2, hd // 2                                       # rope dims on host pairs, nope after
    rope_host = list(range(r2)) + [h2 + i for i in range(r2)]
    nope_host = [i for i in range(hd) if i not in rope_host][:nope]
    rope_src = [nope + 2 * i for i in range(r2)] + [nope + 2 * i + 1 for i in range(r2)]   # Megatron interleave
    for ul, tl in zip(u.layers, layers):
        _norm_from(ul.norm_s if tl.is_linear_attn else ul.norm_a, tl.input_layernorm)
        _norm_from(ul.norm_f, tl.post_attention_layernorm)
        at = tl.self_attn
        if tl.is_linear_attn:
            assert at.use_full_rank_gate
            _kda_lowrank_gate(ul.ssm, qkv=W(at.qkv_proj), conv_w=at.conv1d.weight.detach().to(F64)[:, 0],
                              f_a=W(at.f_a_proj), f_b=W(at.f_b_proj), b=W(at.b_proj), gate=W(at.g_proj),
                              A_log=at.A_log.detach().to(F64), dt_bias=at.dt_bias.detach().to(F64),
                              lower=at.kda_gate_lower_bound, norm_w=at.out_norm.weight.detach().to(F64),
                              norm_eps=getattr(at.out_norm, "variance_epsilon", getattr(at.out_norm, "eps", 1e-6)),
                              out=W(at.out_proj))
        else:
            a = ul.attn
            kva = W(at.kv_a_proj_with_mqa)
            a.pn.weight.copy_(torch.cat([W(at.q_a_proj), kva[:kl]]))
            blk = lambda k: torch.full((k, k), 1 / k, dtype=F64)
            a.n_avg.copy_(torch.block_diag(blk(ql), blk(kl)))
            a.n_w.copy_(torch.cat([at.q_a_layernorm.weight, at.kv_a_layernorm.weight]).detach().to(F64))
            a.n_eps.fill_(float(at.q_a_layernorm.variance_epsilon))
            nq, nkv = slice(d, d + ql), slice(d + ql, d + ql + kl)
            qb, kvb = W(at.q_b_proj), W(at.kv_b_proj)
            for h in range(H):
                r0 = h * hd
                qh, kh = qb[h * qhd:(h + 1) * qhd], kvb[h * (nope + vd): h * (nope + vd) + nope]
                for j, hj in enumerate(nope_host):
                    a.q_proj.weight[r0 + hj, nq] = qh[j]; a.k_proj.weight[r0 + hj, nkv] = kh[j]
                for src, hj in zip(rope_src, rope_host):
                    a.q_proj.weight[r0 + hj, nq] = qh[src]; a.k_proj.weight[r0 + hj, :d] = kva[kl + src - nope]
                a.v_proj.weight[r0: r0 + vd, nkv] = kvb[h * (nope + vd) + nope:(h + 1) * (nope + vd)]
                a.o_proj.weight[:, r0: r0 + vd] = W(at.o_proj)[:, h * vd:(h + 1) * vd]
            if at.g_proj is not None:
                g = W(at.g_proj)
                if at.mla_output_gate_type == "head_wise":
                    g = g.repeat_interleave(vd, 0)
                for h in range(H):
                    a.g_proj.weight[h * hd: h * hd + vd, :d] = g[h * vd:(h + 1) * vd]
                a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
            a.inv_freq.zero_(); a.inv_freq[:r2] = m.rotary_emb.inv_freq.detach().to(F64)
            a.scale.fill_(float(at.scaling))
        f, mo = ul.ffn, tl.mlp
        s_ = mo.shared_experts if hasattr(mo, "experts") else mo
        k = s_.gate_proj.out_features
        f.s_gate.weight[:k] = W(s_.gate_proj); f.s_up.weight[:k] = W(s_.up_proj); f.s_down.weight[:, :k] = W(s_.down_proj)
        if hasattr(mo, "experts"):
            rt = mo.gate
            _cp(f.router.weight, rt.weight)
            f.score_c.fill_(1.0)
            if rt.use_quantile_balancing:                            # select on logits - qb_beta (text row)
                qb = rt.qb_beta.detach().to(F64)
                f.sel_logit.fill_(1.0); f.sel_bias.copy_(-(qb[0] if qb.dim() == 2 else qb))
            elif rt.e_score_correction_bias is not None:
                eb = rt.e_score_correction_bias.detach().to(F64)
                f.sel_bias.copy_(eb[0] if eb.dim() == 2 else eb)
            if rt.norm_topk_prob:
                f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
            f.route_scale.fill_(float(rt.routed_scaling_factor))
            for e, ex in enumerate(mo.experts):
                _cp(f.w_gate[e], ex.gate_proj.weight); _cp(f.w_up[e], ex.up_proj.weight); _cp(f.w_down[e], ex.down_proj.weight)
    return u


@torch.no_grad()
def _jet_block(s: USSM, jb):
    """Jet-Nemotron JetBlock -> the SSM block: q = silu(Wq x), k = silu(Wk x) (no conv), v from the
    DYNAMIC conv branch over v_proj x (kernel w2 silu(w1 x) + b2, then silu), gated-delta rule with l2
    q / k (q * dk^-1/2), Mamba-2 decay, beta = sigmoid(b x), per-head RMSNorm * silu(g x)."""
    H, dk, dv = jb.num_heads, jb.head_k_dim, jb.head_v_dim
    kd, vd = H * dk, H * dv
    W = lambda lin: lin.weight.detach().to(F64)
    s.in_proj.weight.copy_(torch.cat([W(jb.q_proj), W(jb.k_proj), W(jb.v_proj), W(jb.g_proj), W(jb.b_proj), W(jb.a_proj)]))
    s.conv_w.zero_(); s.conv_w[:2 * kd, s.K - 1] = 1; s.act_silu[:2 * kd] = 1       # silu(q), silu(k) at lag 0
    D = s.D
    s.pq.weight[:, D: D + kd] = torch.eye(kd, dtype=F64); s.pk.weight[:, D + kd: D + 2 * kd] = torch.eye(kd, dtype=F64)
    s.q_coef.copy_(torch.tensor([0.0, dk ** -0.5], dtype=F64)); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    s.pv.weight[:, 2 * kd: 2 * kd + vd] = torch.eye(vd, dtype=F64)                 # v_raw, pre-conv stream
    dc = jb.dynamic_conv1d
    assert s.dcW == dc.kernel_size and dc.activation in ("silu", "swish")
    _cp(s.dc_w1.weight, dc.kernel_generator.w1.weight)
    _cp(s.dc_w2.weight, dc.kernel_generator.w2.weight); _cp(s.dc_w2.bias, dc.kernel_generator.w2.bias)
    s.dc_act.fill_(1.0)
    s.pz.weight[:, 2 * kd + vd: 2 * kd + 2 * vd] = torch.eye(vd, dtype=F64)
    zb = 2 * kd + 2 * vd
    s.pb.weight[:, zb: zb + H] = torch.eye(H, dtype=F64)
    s.p1.weight[:, zb + H: zb + 2 * H] = torch.eye(H, dtype=F64).repeat_interleave(dk, 0)
    s.p1.bias.copy_(jb.dt_bias.detach().to(F64).repeat_interleave(dk))
    s.A1.copy_(jb.A_log.detach().to(F64).exp()[:, None].expand(H, dk))
    s.cg.fill_(1); s.cw[1] = 1
    s.gate[0] = 1
    s.avg.copy_(torch.block_diag(*[torch.full((dv, dv), 1 / dv, dtype=F64)] * H))
    s.norm_w.copy_(jb.o_norm.weight.detach().to(F64).repeat(H)); s.norm_eps.fill_(float(jb.o_norm.eps))
    s.out_coef[0] = 1
    _cp(s.out_proj.weight, jb.o_proj.weight)


@torch.no_grad()
def from_jet_nemotron(target):
    """Jet-Nemotron: Qwen2-style decoder (biased q / k / v, RoPE, GQA) whose layers are full
    attention, sliding-window attention (t - s < W) or JetBlocks (the SSM block with the dynamic conv
    branch on v)."""
    tc, m = target.config, target.model
    layers = list(m.layers)
    jb0 = next(l.self_attn for l in layers if hasattr(l.self_attn, "dynamic_conv1d"))
    at0 = next(l.self_attn for l in layers if not hasattr(l.self_attn, "dynamic_conv1d"))
    H, hd = tc.num_attention_heads, at0.head_dim
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=len(layers), heads=H, head_dim=hd,
                ssm_in=2 * jb0.key_dim + 2 * jb0.value_dim + 2 * jb0.num_heads, conv_k=1, ssm_heads=jb0.num_heads,
                dk=jb0.head_k_dim, dv=jb0.head_v_dim, shared_ff=tc.intermediate_size,
                ssm_dconv=jb0.dynamic_conv1d.kernel_size,
                ssm_dconv_rank=jb0.dynamic_conv1d.kernel_generator.w1.out_features)
    u = UniversalElephantine(c)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    for ul, tl in zip(u.layers, layers):
        at = tl.self_attn
        if hasattr(at, "dynamic_conv1d"):
            _norm_from(ul.norm_s, tl.input_layernorm)
            _jet_block(ul.ssm, at)
        else:
            _norm_from(ul.norm_a, tl.input_layernorm)
            _attn_from(ul.attn, at, tc, m.rotary_emb.inv_freq)
            if at.sliding_window is not None:
                ul.attn.window.fill_(float(at.sliding_window - 1))
        _norm_from(ul.norm_f, tl.post_attention_layernorm)
        _dense_ffn_from(ul.ffn, tl.mlp)
    return u


def _interleaved_rope_perm(hd, rot):
    """Target RoPE pairs (2i, 2i+1) within the first `rot` dims -> host pairs (i, i + hd/2); the
    unrotated dims fill the rest in order. Returns host <- target dim order."""
    h, r = hd // 2, rot // 2
    passes = list(range(rot, hd))
    first = [2 * i for i in range(r)] + passes[: h - r]
    second = [2 * i + 1 for i in range(r)] + passes[h - r:]
    return torch.tensor(first + second)


@torch.no_grad()
def _gdn2_from(s: USSM, kn):
    """GPT-S3 GDN-2 -> the SSM block: separate silu short convs on q / k / v, l2 q / k, per-channel
    Mamba-2 decay, an erase gate b on the delta term's READ key (S -= (S (b * k)) k^T), the written
    value w * v (sigmoid feature x post-conv v, the product-feature branch), sigmoid-gated per-head
    RMSNorm with a biased gate."""
    C, H, dk = kn.q_proj.out_features, kn.num_heads, kn.head_dim
    W = lambda lin: lin.weight.detach().to(F64)
    s.in_proj.weight.copy_(torch.cat([W(kn.q_proj), W(kn.k_proj), W(kn.v_proj), W(kn.b_proj), W(kn.w_proj),
                                      W(kn.f_proj), W(kn.g_proj)]))
    s.in_proj.bias[6 * C:] = kn.g_proj.bias.detach().to(F64)
    for i, cv in enumerate((kn.q_conv1d, kn.k_conv1d, kn.v_conv1d)):
        _cp(s.conv_w[i * C:(i + 1) * C], cv.weight[:, 0])
    s.act_silu[:3 * C] = 1
    D, eye, ar = s.D, torch.eye(C, dtype=F64), torch.arange(C)
    R0 = 2 * D
    s.pq.weight[:, D: D + C] = eye; s.pk.weight[:, D + C: D + 2 * C] = eye
    s.q_coef.copy_(torch.tensor([0.0, dk ** -0.5], dtype=F64)); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    s.ph.weight[:, 4 * C: 5 * C] = eye                              # h = sigmoid(w_proj x)
    s.pfx.weight[ar, D + 2 * C + ar] = 1; s.pfy.weight[ar, R0 + ar] = 1   # f = v_conv * h
    s.pv.weight[:, R0 + C: R0 + 2 * C] = eye                        # v <- w * v
    s.p1.weight[:, 5 * C: 6 * C] = eye; s.p1.bias.copy_(kn.dt_bias.detach().to(F64))
    s.A1.copy_(kn.A_log.detach().to(F64).exp()[:, None].expand(H, dk))
    s.c_er.fill_(1.0); s.pbv.weight[:, 3 * C: 4 * C] = eye          # erase read key b * k, b = sigmoid(b_proj x)
    s.cg0.fill_(1.0); s.cw[4] = 1
    s.pz.weight[:, 6 * C: 7 * C] = eye; s.gate[1] = 1
    s.avg.copy_(torch.block_diag(*[torch.full((dk, dk), 1 / dk, dtype=F64)] * H))
    s.norm_w.copy_(kn.o_norm.weight.detach().to(F64).repeat(H)); s.norm_eps.fill_(float(kn.o_norm.eps))
    s.out_coef[0] = 1
    _cp(s.out_proj.weight, kn.o_proj.weight)


@torch.no_grad()
def from_gpts3(target):
    """GPT-S3: GDN-2 layers and gated GQA layers (offset qk-norm, interleaved partial RoPE, sigmoid
    output gate) with Full Attention Residuals instead of a residual stream (the attnres_full
    branch: every sub-block output is a depth-softmax source)."""
    tc, tr = target.config, target.transformer
    blocks = list(tr["h"])
    kn0 = next(b.attn.kernel for b in blocks if not b.is_attention)
    at0 = next((b.attn for b in blocks if b.is_attention), None)
    C = tc.hidden_size
    H, hd = (at0.n_head, at0.head_dim) if at0 is not None else (1, 2)
    c = UConfig(vocab=tc.vocab_size, d=C, L=len(blocks), heads=H, head_dim=hd, ssm_in=7 * C,
                conv_k=kn0.conv_size, ssm_heads=kn0.num_heads, dk=kn0.head_dim, dv=kn0.head_dim,
                ssm_sig_feats=C, ssm_prod_feats=C, shared_ff=tc.intermediate_size, attnres_full=True)
    u = UniversalElephantine(c)
    W = lambda lin: lin.weight.detach().to(F64)
    _cp(u.embed.weight, tr["wte"].weight)
    _norm_from(u.norm, tr["ln_f"])
    _cp(u.lm_head.weight, target.lm_head.weight)
    _cp(u.ar_q_out, tr["attn_res_f"].query); _cp(u.ar_w_out, tr["attn_res_f"].key_norm_weight)
    rd = tc.rotary_dim
    perm = _interleaved_rope_perm(hd, rd)
    inv = 1.0 / (tc.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    for ul, b in zip(u.layers, blocks):
        _cp(ul.ar_qa, b.attn_res_1.query); _cp(ul.ar_wa, b.attn_res_1.key_norm_weight)
        _cp(ul.ar_qf, b.attn_res_2.query); _cp(ul.ar_wf, b.attn_res_2.key_norm_weight)
        ul.ar_eps.fill_(float(tc.attn_res_eps))
        _norm_from(ul.norm_a if b.is_attention else ul.norm_s, b.ln_1)
        _norm_from(ul.norm_f, b.ln_2)
        if b.is_attention:
            a, at = ul.attn, b.attn
            P = lambda w: w.view(-1, hd, *w.shape[1:])[:, perm].reshape(w.shape)
            qg = W(at.q_gate_proj).view(H, 2 * hd, -1)
            a.q_proj.weight.copy_(P(qg[:, :hd].reshape(H * hd, -1)))
            a.g_proj.weight.copy_(qg[:, hd:].reshape(H * hd, -1)); a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
            a.k_proj.weight.copy_(P(_tile_kv(W(at.k_proj), at.n_kv_heads, H, hd)))
            a.v_proj.weight.copy_(_tile_kv(W(at.v_proj), at.n_kv_heads, H, hd))
            a.o_proj.weight.copy_(W(at.o_proj))
            for i, (w_dst, nrm, coef) in enumerate(((a.q_nw, at.q_norm, a.q_coef), (a.k_nw, at.k_norm, a.k_coef))):
                w_dst.copy_((1 + nrm.weight.detach().to(F64))[perm].expand(H, hd))
                a.qk_eps[i] = float(nrm.eps); coef.copy_(torch.tensor([0.0, 1.0]))
            a.inv_freq.zero_(); a.inv_freq[: rd // 2] = inv.to(F64)
            a.scale.fill_(hd ** -0.5)
        else:
            _gdn2_from(ul.ssm, b.attn.kernel)
        f, ml = ul.ffn, b.mlp
        f.s_gate.weight.copy_(W(ml.w_gate)); f.s_up.weight.copy_(W(ml.w_up)); f.s_down.weight.copy_(W(ml.w_down))
    u.layers[0].ar_eps.fill_(float(tc.attn_res_eps))
    return u


@torch.no_grad()
def from_gptx3(target):
    """GPT-X3 (AxiomicLabs modeling_gptx3.py, read as text): GPT-S3's two mixers on a plain pre-norm residual stream --
    GDN-2 layers (_gdn2_from; its forget-gate and output-gate projections are low-rank, Linear(h, dim) then
    Linear(dim, h) (+ bias on the gate's), folded here into one h x h map) and gated GQA layers (offset qk-norm,
    interleaved partial RoPE on the first rotary_dim dims, an elementwise sigmoid output gate), SiLU-GLU MLPs."""
    from types import SimpleNamespace as NS
    tc, tr = target.config, target.transformer
    blocks = list(tr["h"])
    kn0 = next(b.attn.kernel for b in blocks if not b.full_attention)
    C, dim = tc.hidden_size, tc.gdn_head_dim
    H, hd, Hk = tc.num_attention_heads, tc.head_dim, tc.num_key_value_heads
    c = UConfig(vocab=tc.vocab_size, d=C, L=len(blocks), heads=H, head_dim=hd, ssm_in=7 * C, conv_k=tc.conv_size,
                ssm_heads=C // dim, dk=dim, dv=dim, ssm_sig_feats=C, ssm_prod_feats=C, shared_ff=tc.intermediate_size)
    u = UniversalElephantine(c)
    W = lambda lin: lin.weight.detach().to(F64)
    _cp(u.embed.weight, tr["wte"].weight)
    _norm_from(u.norm, tr["ln_f"])
    _cp(u.lm_head.weight, target.lm_head.weight)
    rd = tc.rotary_dim
    perm = _interleaved_rope_perm(hd, rd)
    inv = 1.0 / (tc.rope_theta ** (torch.arange(0, rd, 2, dtype=F64) / rd))
    P = lambda w: w.view(-1, hd, *w.shape[1:])[:, perm].reshape(w.shape)
    for ul, b in zip(u.layers, blocks):
        _norm_from(ul.norm_a if b.full_attention else ul.norm_s, b.ln_1)
        _norm_from(ul.norm_f, b.ln_2)
        if b.full_attention:
            a, at = ul.attn, b.attn
            qg = W(at.q_gate_proj).view(H, 2 * hd, -1)
            a.q_proj.weight.copy_(P(qg[:, :hd].reshape(H * hd, -1)))
            a.g_proj.weight.copy_(qg[:, hd:].reshape(H * hd, -1)); a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
            a.k_proj.weight.copy_(P(_tile_kv(W(at.k_proj), Hk, H, hd)))
            a.v_proj.weight.copy_(_tile_kv(W(at.v_proj), Hk, H, hd))
            a.o_proj.weight.copy_(W(at.o_proj))
            for i, (w_dst, nrm, coef) in enumerate(((a.q_nw, at.q_norm, a.q_coef), (a.k_nw, at.k_norm, a.k_coef))):
                w_dst.copy_((1 + nrm.weight.detach().to(F64))[perm].expand(H, hd))
                a.qk_eps[i] = float(nrm.eps); coef.copy_(torch.tensor([0.0, 1.0]))
            a.inv_freq.zero_(); a.inv_freq[: rd // 2] = inv
            a.scale.fill_(hd ** -0.5)
        else:
            kn = b.attn.kernel
            lin = lambda w, bias=None: NS(weight=w, bias=bias)
            f1, f2 = kn.f_proj[0], kn.f_proj[1]
            g1, g2 = kn.g_proj[0], kn.g_proj[1]
            _gdn2_from(ul.ssm, NS(q_proj=kn.q_proj, k_proj=kn.k_proj, v_proj=kn.v_proj, b_proj=kn.b_proj,
                                  w_proj=kn.w_proj, f_proj=lin(W(f2) @ W(f1)),
                                  g_proj=lin(W(g2) @ W(g1), g2.bias.detach().to(F64)),
                                  q_conv1d=kn.q_conv1d, k_conv1d=kn.k_conv1d, v_conv1d=kn.v_conv1d,
                                  dt_bias=kn.dt_bias, A_log=kn.A_log, o_norm=kn.o_norm, o_proj=kn.o_proj,
                                  num_heads=C // dim, head_dim=dim))
        f, ml = ul.ffn, b.mlp
        f.s_gate.weight.copy_(W(ml.w_gate)); f.s_up.weight.copy_(W(ml.w_up)); f.s_down.weight.copy_(W(ml.w_down))
    return u


def gka_config(layer):
    """USSM shape for a Gated KalmaNet layer: u = [q | k | v | beta | alpha | a | g], H expanded heads,
    sigmoid features [beta | alpha]."""
    H, hd = layer.expanded_num_heads, layer.head_dim
    D = layer.query_dim + layer.key_dim + layer.value_dim + 3 * H + H * hd
    return dict(ssm_in=D, conv_k=layer.conv_size, ssm_heads=H, dk=hd, dv=hd, ssm_sig_feats=2 * H)


@torch.no_grad()
def _gka_from(s: USSM, layer):
    """Gated KalmaNet (awslabs hybrid-model-factory) -> the SSM block. Short conv + silu on q/k/v (GQA
    repeat to the expanded heads), q/k l2-normed as t / (|t| + 1e-6), k and v scaled by sigmoid(b) + 1e-6
    (k through the c_ka key modulation, v through the write w = sigmoid(b) + 1e-6), one scalar decay
    g = -exp(A_log) softplus(a + dt_bias) for both states; the ridge readout solves (H_t + lam I) x = q_t
    with lam = ridge_strength ||H_t||_F in num_iter Chebyshev steps and feeds q + alpha (x - q) (alpha =
    sigmoid(alpha_proj x)) to the GLA state, scaled head_dim^-1/2 (carried on q before the solve, which is
    linear in q); per-head RMSNorm * silu(g_proj x)."""
    assert layer.kv_proj_rank is None, "low-rank KV expansion not mapped"
    assert layer.use_forgetting_gate_kk == layer.use_forgetting_gate, "H_t shares the GLA decay here"
    cpu = lambda t: t.detach().double().cpu()
    H, dk = layer.expanded_num_heads, layer.head_dim
    nq, nk, nv = layer.query_dim, layer.key_dim, layer.value_dim
    D = s.D
    ob, oal, oa, og = nq + nk + nv, nq + nk + nv + H, nq + nk + nv + 2 * H, nq + nk + nv + 3 * H
    W = s.in_proj.weight
    W[:nq], W[nq:nq + nk], W[nq + nk:ob] = cpu(layer.q_proj.weight), cpu(layer.k_proj.weight), cpu(layer.v_proj.weight)
    if layer.use_beta_gate:
        W[ob:oal] = cpu(layer.b_proj.weight); s.in_proj.bias[ob:oal] = cpu(layer.b_proj.bias)
    if layer.use_alpha_connection:
        W[oal:oa] = cpu(layer.alpha_proj.weight); s.in_proj.bias[oal:oa] = cpu(layer.alpha_proj.bias)
    W[oa:og] = cpu(layer.a_proj.weight)
    if layer.use_gate:
        W[og:] = cpu(layer.g_proj.weight)
    convs = [(0, layer.q_conv1d), (nq, layer.k_conv1d)] + ([(nq + nk, layer.v_conv1d)] if layer.use_v_conv else [])
    for off, cv in convs:
        n = cv.weight.shape[0]
        s.conv_w[off:off + n] = cpu(cv.weight)[:, 0]
        if cv.bias is not None:
            s.conv_b[off:off + n] = cpu(cv.bias)
        s.act_silu[off:off + n] = 1
    eye = torch.eye(dk, dtype=F64)
    vsrc = D + nq + nk if layer.use_v_conv else nq + nk                 # no v conv: v read pre-conv (u)
    for h in range(H):
        rows = slice(h * dk, (h + 1) * dk)
        gq, gk, gv = h // layer.num_q_groups, h // layer.num_k_groups, h // layer.num_v_groups
        s.pq.weight[rows, D + gq * dk: D + (gq + 1) * dk] = eye
        s.pk.weight[rows, D + nq + gk * dk: D + nq + (gk + 1) * dk] = eye
        s.pv.weight[rows, vsrc + gv * dk: vsrc + (gv + 1) * dk] = eye
    s.q_coef.copy_(torch.tensor([0.0, layer.gla_scale], dtype=F64)); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    s.l2_eps.fill_(0.0); s.l2_eps_out.fill_(1e-6)
    R0 = 2 * D + s.Dn                                                # sigmoid features [beta | alpha]
    hrows = torch.arange(H)
    s.ph.weight[hrows, ob + hrows] = 1
    s.ph.weight[H + hrows, oal + hrows] = 1
    if layer.use_beta_gate:
        s.pbeta.weight[torch.arange(H * dk), R0 + hrows.repeat_interleave(dk)] = 1
        s.pbeta.bias.fill_(1e-6)
        s.c_ka.fill_(1.0)                                            # k <- (sigmoid(b) + 1e-6) k
        s.pb.weight[hrows, ob + hrows] = 1
        s.cw[1] = 1; s.cw[4] = 1e-6                                  # write sigmoid(b) + 1e-6 (the v scale)
    else:
        s.cw[4] = 1
    if layer.use_forgetting_gate:
        s.p1.weight[torch.arange(H * dk), oa + hrows.repeat_interleave(dk)] = 1
        s.p1.bias.copy_(cpu(layer.dt_bias).repeat_interleave(dk))
        s.A1.copy_(cpu(layer.A_log).exp()[:, None].expand(H, dk))
    if layer.num_iter > 0:
        s.ridge_c.fill_(1.0); s.ridge_ratio.fill_(layer.ridge_strength); s.ridge_iters.fill_(layer.num_iter)
        if layer.use_alpha_connection:
            s.pal.weight[hrows, R0 + H + hrows] = 1                  # alpha = sigmoid(alpha_proj x)
        else:
            s.pal.bias.fill_(1.0)
    s.avg.copy_(torch.block_diag(*[torch.full((dk, dk), 1 / dk, dtype=F64)] * H))
    s.norm_w.copy_(cpu(layer.o_norm.weight).repeat(H))
    s.norm_eps.fill_(float(layer.o_norm.eps))
    if layer.use_gate:
        s.pz.weight[:, og:og + H * dk] = torch.eye(H * dk, dtype=F64)
        s.gate[0] = 1                                                # silu(g)
    else:
        s.gate[4] = 1
    s.out_coef[0] = 1
    s.out_proj.weight.copy_(cpu(layer.o_proj.weight))


@torch.no_grad()
def gka_block(layer):
    """One Gated KalmaNet mixer as a USSM block."""
    s = USSM(UConfig(vocab=1, d=layer.hidden_size, L=1, **gka_config(layer)))
    _gka_from(s, layer)
    return s


@torch.no_grad()
def from_hybrid_qwen3(target):
    """hybrid-model-factory HybridQwen3 (GKA-primed HQwen3): Qwen3 attention layers ('*': qk-norm,
    RoPE, GQA) and Gated KalmaNet layers ('GKA') in the hybrid_override_pattern, dense SwiGLU MLPs."""
    tc, m = target.config, target.model
    g0 = next((tl.gka for tl in m.layers if hasattr(tl, "gka")), None)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    c = UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers, heads=tc.num_attention_heads,
                head_dim=hd, shared_ff=tc.intermediate_size, **(gka_config(g0) if g0 is not None else {}))
    u = UniversalElephantine(c)
    cpu = lambda t: t.detach().cpu()
    _cp(u.embed.weight, cpu(m.embed_tokens.weight))
    _norm_from(u.norm, m.norm)
    _cp(u.lm_head.weight, cpu(target.lm_head.weight))
    inv = m.rotary_emb.inv_freq.detach().cpu()
    for ul, tl in zip(u.layers, m.layers):
        if hasattr(tl, "gka"):
            _norm_from(ul.norm_s, tl.input_layernorm)
            _gka_from(ul.ssm, tl.gka)
        else:
            _norm_from(ul.norm_a, tl.input_layernorm)
            _attn_from(ul.attn, tl.self_attn, tc, inv, qk_norm=True)
        _norm_from(ul.norm_f, tl.post_attention_layernorm)
        _dense_ffn_from(ul.ffn, tl.mlp)
    return u


@torch.no_grad()
def from_pseudobrain(cfg, sd):
    """PseudoBrain PB27Qwen3 / PB27Qwen3Moe (pb27-vllm plugin, universal_models.py), from the
    checkpoint itself (HF config dict + state dict: the only reference is the vLLM plugin). A stock
    Qwen3 (-MoE) decoder whose EVERY attention layer is sliding-window (vLLM per_layer_sliding_window
    = pb_attention_window, q - k < W; pb_attention_sinks is listed but not served) and whose
    pb_hybrid_layers add a parallel Qwen3.5 gated-DeltaNet (QwenGatedDeltaNetAttention, flat qkv
    layout) on the same input_layernorm output, summed before post_attention_layernorm:
        x += attn(n(x)) + gdn(n(x))   -> the host's parallel attention + SSM (seq = 0, norm_s = norm_a)."""
    from types import SimpleNamespace as NS
    T = lambda k: sd[k].detach().to(F64)
    lin = lambda k: NS(weight=T(k + ".weight"), bias=None)
    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    hd = cfg.get("head_dim") or D // H
    Hk, Hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    moe = cfg.get("num_experts", 0) > 0
    hyb, W = set(cfg["pb_hybrid_layers"]), cfg.get("pb_attention_window")
    eps = cfg["rms_norm_eps"]
    c = UConfig(vocab=cfg["vocab_size"], d=D, L=cfg["num_hidden_layers"], heads=H, head_dim=hd,
                ssm_in=2 * Hk * dk + 2 * Hv * dv + 2 * Hv, conv_k=cfg["linear_conv_kernel_dim"], ssm_heads=Hv,
                dk=dk, dv=dv, experts=cfg["num_experts"] if moe else 1,
                expert_ff=cfg["moe_intermediate_size"] if moe else 1,
                top_k=cfg["num_experts_per_tok"] if moe else 1, shared_ff=1 if moe else cfg["intermediate_size"])
    u = UniversalElephantine(c)
    nrm = lambda k: NS(weight=T(k + ".weight"), eps=eps)
    _cp(u.embed.weight, T("model.embed_tokens.weight"))
    _norm_from(u.norm, nrm("model.norm"))
    _cp(u.lm_head.weight, T("model.embed_tokens.weight" if cfg.get("tie_word_embeddings") else "lm_head.weight"))
    theta = (cfg.get("rope_parameters") or {}).get("rope_theta", cfg.get("rope_theta", 10000.0))
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    for i, ul in enumerate(u.layers):
        p = f"model.layers.{i}."
        sa = NS(**{n: lin(p + "self_attn." + n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
                q_norm=nrm(p + "self_attn.q_norm"), k_norm=nrm(p + "self_attn.k_norm"), scaling=hd ** -0.5)
        _norm_from(ul.norm_a, nrm(p + "input_layernorm"))
        _attn_from(ul.attn, sa, NS(num_key_value_heads=KV), inv, qk_norm=True)
        if W is not None:
            ul.attn.window.fill_(float(W - 1))
        if i in hyb:
            g = p + "gdn."
            gd = NS(num_k_heads=Hk, num_v_heads=Hv, head_k_dim=dk, head_v_dim=dv,
                    **{n: lin(g + n) for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")},
                    conv1d=NS(weight=T(g + "conv1d.weight"), bias=None), dt_bias=T(g + "dt_bias"),
                    A_log=T(g + "A_log"), norm=nrm(g + "norm"))
            _norm_from(ul.norm_s, nrm(p + "input_layernorm"))
            _gdn_from(ul.ssm, gd)
            ul.seq.fill_(0.0)
        _norm_from(ul.norm_f, nrm(p + "post_attention_layernorm"))
        f = ul.ffn
        if moe:
            _cp(f.router.weight, T(p + "mlp.gate.weight"))
            for e in range(c.experts):
                x = f"{p}mlp.experts.{e}."
                _cp(f.w_gate[e], T(x + "gate_proj.weight")); _cp(f.w_up[e], T(x + "up_proj.weight"))
                _cp(f.w_down[e], T(x + "down_proj.weight"))
            f.renorm.fill_(1.0 if cfg.get("norm_topk_prob", True) else 0.0)
        else:
            _dense_ffn_from(f, NS(**{n: lin(p + "mlp." + n) for n in ("gate_proj", "up_proj", "down_proj")}))
    return u


# --- stock transformers decoders: one generic reader, one from_* per architecture ------------------
#
# The loaders below read the stock `transformers` modules of each family (attention projections,
# fused or not; RMS / LayerNorm / offset norms; gated or ungated MLPs; rotary layout and partial
# factor; per-layer sliding windows) into the same host. Family-specific ops are host branches:
# logit softcaps (UAttention.cap, UniversalElephantine.final_cap), ALiBi (UAttention.alibi), learned
# per-head sinks (UAttention.null_logit), qk-norm over the whole projection / as a LayerNorm / after the
# rotary (qk_full, qk_centre, qk_post), a q / k / v clamp (qkv_clip), an embedding norm (emb_norm),
# identity norms (UNorm.ident), ungated ReLU (UFFN.c_relu), expert / router biases, the swigluoai
# up-shift and gate limit (up_shift, clip_lim), group-limited top-k (UFFN.grp), an lm_head bias.
# Scalar multipliers (Granite, MiniCPM, Cohere) are folded into weights.

def _W(lin):
    return lin.weight.detach().to(F64)


def _B(lin):
    b = getattr(lin, "bias", None)
    return None if b is None else b.detach().to(F64)


@torch.no_grad()
def _norm_any(dst: UNorm, src, kind="rms"):
    """A target norm -> UNorm. kind: rms (w x / rms), rms1p ((1 + w) x / rms: Gemma), ln (LayerNorm:
    centred, weight / bias optional, eps 1e-5 when the module keeps none), ln1p (Nemotron LayerNorm1P:
    1 + w), none (no norm: UNorm.ident)."""
    if kind == "none" or src is None:
        dst.ident.fill_(1.0)
        return
    w = getattr(src, "weight", None)
    w = torch.ones(dst.weight.shape, dtype=F64) if w is None else w.detach().to(F64)
    dst.weight.copy_(w + 1 if kind.endswith("1p") else w)
    eps = getattr(src, "variance_epsilon", getattr(src, "eps", None))
    dst.eps.fill_(float(1e-5 if eps is None else eps))
    if kind.startswith("ln"):
        dst.centre.fill_(1.0)
        b = getattr(src, "bias", None)
        if b is not None:
            dst.bias.copy_(b.detach().to(F64))


def _rope_perm(hd, rot, layout):
    """host <- target head-dim order for a target rotating its first `rot` dims, either as
    rotate-half pairs (j, j + rot/2) ('half') or adjacent pairs (2j, 2j + 1) ('interleave')."""
    if not rot:
        return None
    perm = _partial_rope_perm(hd, rot) if layout == "half" else _interleaved_rope_perm(hd, rot)
    return None if torch.equal(perm, torch.arange(hd)) else perm


def _sliding(tc, i):
    """The sliding window of layer i as the stock model's mask builder applies it (None: full).
    Families with layer_types slide only on 'sliding_attention' layers; the others (Mistral,
    Mixtral, Phi-3, StarCoder2) apply config.sliding_window to every layer when it is set."""
    sw = getattr(tc, "sliding_window", None)
    lt = getattr(tc, "layer_types", None)
    if sw is None:
        return None
    if lt is not None:
        return sw if lt[i] == "sliding_attention" else None
    return sw


@torch.no_grad()
def _attn_any(dst: UAttention, wq, wk, wv, wo, bq=None, bk=None, bv=None, bo=None, *, n_kv, scale,
              inv_freq=None, att_scale=1.0, layout="half", window=None, q_norm=None, k_norm=None,
              qk_kind="rms", qk_post=False, clip=None):
    """GQA attention from (out, in) projections (k / v with n_kv heads, tiled onto the host's H).
    The rotary (inv_freq; its length gives the rotated dims) is permuted onto the host pairs; the
    rotary's attention_scaling multiplies cos and sin of q and k: folded into the scale (att^2).
    q_norm / k_norm: per-head (weight of head_dim) or whole-projection (weight of H * hd: qk_full);
    qk_kind 'ln' makes them LayerNorms; per-head-distinct norms pass (H, hd) weights."""
    H, hd = dst.H, dst.hd
    rot = 0 if inv_freq is None else 2 * inv_freq.numel()
    perm = _rope_perm(hd, rot, layout)
    tile = lambda w: None if w is None else _tile_kv(w, n_kv, H, hd)
    _mha_from(dst, wq, tile(wk), tile(wv), bq, tile(bk), tile(bv), wo, bo, float(scale) * float(att_scale) ** 2,
              inv_freq=inv_freq, perm=perm)
    if window is not None:
        dst.window.fill_(float(window - 1))
    if clip is not None:
        dst.qkv_clip.fill_(float(clip))
    P = (lambda w: w[..., perm]) if perm is not None else (lambda w: w)
    for i, (nm, w_dst, b_dst, coef, nh) in enumerate(((q_norm, dst.q_nw, dst.q_nb, dst.q_coef, H),
                                                      (k_norm, dst.k_nw, dst.k_nb, dst.k_coef, n_kv))):
        if nm is None:
            continue
        def full(t):                                     # (hd,) | (nh * hd,) | (nh, hd) -> (H, hd)
            t = t.detach().to(F64)
            if t.numel() == hd:
                return t.reshape(1, hd).expand(H, hd)
            t = t.reshape(nh, hd)
            return t.repeat_interleave(H // nh, 0) if nh != H else t
        ws = getattr(nm, "norms", None)                  # StableLM: one LayerNorm per head
        if ws is not None:
            w = torch.stack([n.weight.detach().to(F64) for n in ws])
            b = torch.stack([n.bias.detach().to(F64) if n.bias is not None else torch.zeros(hd, dtype=F64) for n in ws])
            eps = ws[0].eps
        else:
            w = nm.weight if getattr(nm, "weight", None) is not None else torch.ones(hd)
            b = getattr(nm, "bias", None)
            b = torch.zeros(hd) if b is None else b
            eps = getattr(nm, "variance_epsilon", getattr(nm, "eps", 1e-6))
            if w.dim() == 1 and w.numel() > hd:             # one norm over the whole projection
                dst.qk_full.fill_(1.0)
        w_dst.copy_(P(full(w)))
        if qk_kind == "ln":
            b_dst.copy_(P(full(b)))
            dst.qk_centre[i] = 1.0
        dst.qk_eps[i] = float(eps)
        coef.copy_(torch.tensor([0.0, 1.0]))
    if qk_post:
        dst.qk_post.fill_(1.0)


def _qkv(a):
    """(wq, wk, wv, wo, bq, bk, bv, bo) of an attention with separate q / k / v / o projections."""
    o = getattr(a, "o_proj", None) or getattr(a, "dense", None) or getattr(a, "out_proj", None)
    return (_W(a.q_proj), _W(a.k_proj), _W(a.v_proj), _W(o), _B(a.q_proj), _B(a.k_proj), _B(a.v_proj), _B(o))


@torch.no_grad()
def _mlp_any(f: UFFN, mlp, act, n=None):
    """A dense MLP -> the shared path (rows [:n]): gated gate / up / down (or a fused gate_up, gate
    first), or an ungated up / down pair under any of the stock names; biases copied."""
    def put(lin, w, b):
        rows = w.shape[0]
        lin.weight[:rows, : w.shape[1]] = w
        if b is not None:
            lin.bias[:rows] = b
    if hasattr(mlp, "gate_up_proj") or isinstance(getattr(mlp, "input_linear", None), nn.Linear):
        gu = mlp.gate_up_proj if hasattr(mlp, "gate_up_proj") else mlp.input_linear   # Granite shared_mlp
        w, b = _W(gu), _B(gu)
        h = w.shape[0] // 2
        put(f.s_gate, w[:h], None if b is None else b[:h]); put(f.s_up, w[h:], None if b is None else b[h:])
        down, gated = getattr(mlp, "down_proj", None) or mlp.output_linear, True
    elif hasattr(mlp, "gate_proj"):
        put(f.s_gate, _W(mlp.gate_proj), _B(mlp.gate_proj)); put(f.s_up, _W(mlp.up_proj), _B(mlp.up_proj))
        down, gated = mlp.down_proj, True
    else:
        up, down = next((getattr(mlp, u), getattr(mlp, d_)) for u, d_ in
                        (("up_proj", "down_proj"), ("c_fc", "c_proj"), ("fc1", "fc2"), ("fc_in", "fc_out"),
                         ("dense_h_to_4h", "dense_4h_to_h")) if hasattr(mlp, u))
        put(f.s_up, _W(up), _B(up))
        gated = False
    wd, bd = _W(down), _B(down)
    f.s_down.weight[:, : wd.shape[1]] = wd
    if bd is not None:
        f.s_down.bias.copy_(bd)
    _set_act(f, act, gated)


def _shell(tc, *, d, L, H, hd, ff, vocab=None, **kw):
    return UniversalElephantine(UConfig(vocab=vocab or tc.vocab_size, d=d, L=L, heads=H, head_dim=hd,
                                        shared_ff=ff, **kw))


@torch.no_grad()
def _head_from(u, embed, norm, lm_head, norm_kind="rms", emb_scale=None, logit_scale=None):
    w = _W(embed)
    _cp(u.embed.weight, w if emb_scale is None else w * emb_scale)
    _norm_any(u.norm, norm, norm_kind)
    if lm_head is None:                                  # a headless body: the caller sets the readout
        return
    lw = _W(lm_head)
    _cp(u.lm_head.weight, lw if logit_scale is None else lw * logit_scale)
    if _B(lm_head) is not None:
        _cp(u.lm_head.bias, _B(lm_head) if logit_scale is None else _B(lm_head) * logit_scale)


@torch.no_grad()
def _fold_residual(ul, r):
    """Granite / MiniCPM residual multiplier x + r f(x): scale both sub-block output projections."""
    for lin in (ul.attn.o_proj, ul.ffn.s_down):
        lin.weight.mul_(r); lin.bias.mul_(r)
    ul.ffn.w_down.mul_(r); ul.ffn.b_down.mul_(r)


@torch.no_grad()
def from_granite(target):
    """Granite / Granite-MoE / Granite-MoE-Shared / Granite-SWA (sliding layers + a learned per-head sink:
    the null-key branch) (GraniteForCausalLM, GraniteMoe*, GraniteSWA*): a Llama body with four scalar multipliers, all folded into
    weights: embedding_multiplier (the table), attention_multiplier (the softmax scale),
    residual_multiplier (x + r f(x): both output projections) and logits_scaling (lm_head / s)."""
    tc, m = target.config, target.model
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    hd = tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=sff,
               experts=E, expert_ff=ef, top_k=k)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, emb_scale=float(tc.embedding_multiplier),
               logit_scale=1.0 / float(tc.logits_scaling))
    rope = m.rotary_emb
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, window=getattr(a, "sliding_window", None))
        if getattr(a, "sinks", None) is not None:                          # Granite-SWA learned sinks
            _cp(ul.attn.null_logit, a.sinks)
        _ffn_any(ul.ffn, _mlp_of(tl), tc, tc.hidden_act, shared=getattr(tl, "shared_mlp", None))
        _fold_residual(ul, float(tc.residual_multiplier))
    return u


@torch.no_grad()
def from_phi3(target):
    """Phi-3 / Phi-4-mini (Phi3ForCausalLM): a fused [q | k | v] qkv_proj and a fused [gate | up]
    gate_up_proj, partial rotary (partial_rotary_factor, rotate-half within the rotated dims), LongRoPE
    (the short-context inv_freq; its attention factor folded into the scale), a sliding window on every
    layer when config.sliding_window is set."""
    tc, m = target.config, target.model
    H, n_kv = tc.num_attention_heads, tc.num_key_value_heads
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // H
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    rope = m.rotary_emb
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        W, b = _W(a.qkv_proj), _B(a.qkv_proj)
        sp = [H * hd, n_kv * hd, n_kv * hd]
        wq, wk, wv = W.split(sp)
        bq, bk, bv = b.split(sp) if b is not None else (None,) * 3
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _attn_any(ul.attn, wq, wk, wv, _W(a.o_proj), bq, bk, bv, _B(a.o_proj), n_kv=n_kv, scale=a.scaling,
                  inv_freq=rope.inv_freq, att_scale=rope.attention_scaling, window=_sliding(tc, i))
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def from_glm(target):
    """GLM-4 (GlmForCausalLM, glm-4-9b) and GLM-4-0414 (Glm4ForCausalLM): partial INTERLEAVED rotary
    (pairs (2i, 2i + 1) in the first partial_rotary_factor of each head), q / k / v biases, a fused
    [gate | up] MLP; GLM-4-0414 adds sandwich post-norms (post_self_attn / post_mlp layernorm)."""
    tc, m = target.config, target.model
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    rope = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        if hasattr(tl, "post_self_attn_layernorm"):
            _norm_any(ul.post_a, tl.post_self_attn_layernorm)
            _norm_any(ul.post_f, tl.post_mlp_layernorm)
            ul.post_coef[0] = torch.tensor([0.0, 1.0])
            ul.post_coef[2] = torch.tensor([0.0, 1.0])
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, layout="interleave")
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def from_olmo(target):
    """OLMo (1), OLMo-2, OLMo-3 and EXAONE-4. OLMo (1): non-parametric LayerNorms (no weight / bias,
    eps 1e-5) and an optional clip_qkv clamp of q, k, v (UAttention.qkv_clip). OLMo-2 / 3: NO pre-norms
    (UNorm.ident), post-norms on each sub-block's output, and qk-norm over the WHOLE q / k projection
    (UAttention.qk_full); OLMo-3 alternates sliding / full layers, each on its own rotary. EXAONE-4: the
    OLMo-2 layout with per-head qk-norm, its full layers NoPE when a sliding window is configured."""
    tc, m = target.config, target.model
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=sff,
               experts=E, expert_ff=ef, top_k=k)
    olmo1 = tc.model_type == "olmo"
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, "ln" if olmo1 else "rms")
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        lt = (getattr(tc, "layer_types", None) or ["full_attention"] * tc.num_hidden_layers)[i]
        rope = m.rotary_embs[lt] if hasattr(m, "rotary_embs") else m.rotary_emb
        use_rope = not (tc.model_type == "exaone4" and tc.sliding_window is not None and lt != "sliding_attention")
        if olmo1:
            _norm_any(ul.norm_a, tl.input_layernorm, "ln")
            _norm_any(ul.norm_f, tl.post_attention_layernorm, "ln")
        else:
            _norm_any(ul.norm_a, None, "none")
            _norm_any(ul.norm_f, None, "none")
            _norm_any(ul.post_a, tl.post_attention_layernorm)
            _norm_any(ul.post_f, tl.post_feedforward_layernorm)
            ul.post_coef[0] = torch.tensor([0.0, 1.0])
            ul.post_coef[2] = torch.tensor([0.0, 1.0])
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling,
                  inv_freq=rope.inv_freq if use_rope else None, att_scale=rope.attention_scaling,
                  window=_sliding(tc, i), q_norm=getattr(a, "q_norm", None), k_norm=getattr(a, "k_norm", None),
                  clip=getattr(tc, "clip_qkv", None))
        _ffn_any(ul.ffn, _mlp_of(tl), tc, tc.hidden_act)
    return u


@torch.no_grad()
def from_cohere2(target):
    """Cohere-2 (Command-R7B / A): Cohere's parallel block (one mean-centred LayerNorm read by both
    attention and MLP: ULayer.par), interleaved RoPE on the sliding-window layers only (the full layers
    are NoPE), logit_scale folded into lm_head."""
    tc, m = target.config, target.model
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, "ln", logit_scale=float(tc.logit_scale))
    rope = m.rotary_emb
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        w = _sliding(tc, i)
        _norm_any(ul.norm_a, tl.input_layernorm, "ln")
        _norm_any(ul.norm_f, tl.input_layernorm, "ln")
        ul.par.fill_(1.0)
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling,
                  inv_freq=rope.inv_freq if w is not None else None, att_scale=rope.attention_scaling,
                  layout="interleave", window=w)
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def from_hunyuan(target):
    """HunYuan dense / MoE (HunYuanDenseV1ForCausalLM / HunYuanMoEV1ForCausalLM): a Llama body whose per-head qk-norms
    (query_layernorm / key_layernorm) are applied AFTER the rotary (UAttention.qk_post); dynamic-alpha
    NTK RoPE is the target's inv_freq."""
    tc, m = target.config, target.model
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=sff,
               experts=E, expert_ff=ef, top_k=k)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    rope = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, q_norm=getattr(a, "query_layernorm", None),
                  k_norm=getattr(a, "key_layernorm", None), qk_post=True)
        _ffn_any(ul.ffn, tl.mlp, tc, tc.hidden_act)
    return u


@torch.no_grad()
def from_starcoder2(target):
    """StarCoder2: biased q / k / v / o, biased LayerNorms, an ungated biased c_fc / c_proj MLP
    (gelu_pytorch_tanh: the tanh GeLU branch), RoPE, a sliding window on every layer when set."""
    tc, m = target.config, target.model
    hd = tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, "ln")
    rope = m.rotary_emb
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm, "ln")
        _norm_any(ul.norm_f, tl.post_attention_layernorm, "ln")
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, window=_sliding(tc, i))
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def from_nemotron(target):
    """Nemotron (NemotronForCausalLM, Minitron): LayerNorm1P norms ((1 + w), bias), partial rotary
    (rotate-half within partial_rotary_factor), an ungated up / down MLP under relu^2 (c_relu2)."""
    tc, m = target.config, target.model
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, "ln1p")
    rope = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm, "ln1p")
        _norm_any(ul.norm_f, tl.post_attention_layernorm, "ln1p")
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=hd ** -0.5, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling)
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def from_stablelm(target):
    """StableLM / StableLM-2 (StableLmForCausalLM): biased LayerNorms, partial rotary (rotate-half
    within partial_rotary_factor), optional q / k / v biases, optional per-head LayerNorm qk-norm
    (StableLmLayerNormPerHead: one LayerNorm per head, qk_centre) and the optional parallel residual
    (one input LayerNorm read by attention and MLP: ULayer.par)."""
    tc, m = target.config, target.model
    hd = tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, "ln")
    rope = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm, "ln")
        if tl.use_parallel_residual:
            _norm_any(ul.norm_f, tl.input_layernorm, "ln")
            ul.par.fill_(1.0)
        else:
            _norm_any(ul.norm_f, tl.post_attention_layernorm, "ln")
        qk = getattr(a, "qk_layernorm", False)
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=hd ** -0.5, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, q_norm=a.q_layernorm if qk else None,
                  k_norm=a.k_layernorm if qk else None, qk_kind="ln")
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def from_persimmon(target):
    """Persimmon / Fuyu text (PersimmonForCausalLM): a per-head fused [q | k | v] query_key_value,
    LayerNorm qk-norm with biases (qk_centre), partial rotary, biased LayerNorms, an ungated relu^2 MLP."""
    tc, m = target.config, target.model
    d, H = tc.hidden_size, tc.num_attention_heads
    hd = d // H
    u = _shell(tc, d=d, L=tc.num_hidden_layers, H=H, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.final_layernorm, target.lm_head, "ln")
    rope = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        W, b = _W(a.query_key_value).view(H, 3, hd, d), _B(a.query_key_value).view(H, 3, hd)
        qkv = [W[:, j].reshape(H * hd, d) for j in range(3)]
        bqkv = [b[:, j].reshape(H * hd) for j in range(3)]
        _norm_any(ul.norm_a, tl.input_layernorm, "ln")
        _norm_any(ul.norm_f, tl.post_attention_layernorm, "ln")
        qk = getattr(a, "qk_layernorm", False)
        _attn_any(ul.attn, *qkv, _W(a.dense), *bqkv, _B(a.dense), n_kv=H, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, q_norm=a.q_layernorm if qk else None,
                  k_norm=a.k_layernorm if qk else None, qk_kind="ln")
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def from_phi(target):
    """Phi-1 / 1.5 / 2 (PhiForCausalLM): a PARALLEL block with ONE shared LayerNorm, x + attn(ln x) +
    mlp(ln x) (ULayer.par, norm_f = the same LayerNorm), biased q / k / v / dense, partial rotary,
    optional LayerNorm qk-norm, an ungated biased fc1 / fc2 MLP (gelu_new), a biased lm_head."""
    tc, m = target.config, target.model
    hd = tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.final_layernorm, target.lm_head, "ln")
    rope = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm, "ln")
        _norm_any(ul.norm_f, tl.input_layernorm, "ln")
        ul.par.fill_(1.0)
        qk = getattr(a, "qk_layernorm", False)
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, q_norm=a.q_layernorm if qk else None,
                  k_norm=a.k_layernorm if qk else None, qk_kind="ln")
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


def _sincos_inv_freq(rot):
    """GPT-J / CodeGen's create_sinusoidal_positions frequencies, rounded as the target builds them
    (fp32 at init)."""
    return (1.0 / (10000 ** (torch.arange(0, rot, 2, dtype=torch.int64) / rot))).to(torch.float32).to(F64)


@torch.no_grad()
def from_gptj(target):
    """GPT-J and CodeGen (GPTJForCausalLM / CodeGenForCausalLM): a parallel block with ONE shared
    LayerNorm (ULayer.par), interleaved rotary (rotate_every_two) on the first rotary_dim of each head,
    an ungated biased fc_in / fc_out MLP (gelu_new), a biased lm_head. CodeGen fuses q / v / k into one
    qkv_proj in mp_num = 4 blocks of [q | v | k]."""
    tc, m = target.config, target.transformer
    d, H = tc.n_embd, tc.n_head
    hd = d // H
    u = _shell(tc, d=d, L=tc.n_layer, H=H, hd=hd, ff=tc.n_inner or 4 * d)
    _head_from(u, m.wte, m.ln_f, target.lm_head, "ln")
    rot = tc.rotary_dim or hd
    inv = _sincos_inv_freq(rot)
    for ul, tl in zip(u.layers, m.h):
        a = tl.attn
        if hasattr(a, "qkv_proj"):                      # CodeGen
            mp = 4
            W = _W(a.qkv_proj).view(mp, 3, d // mp, d)
            wq, wv, wk = (W[:, j].reshape(d, d) for j in range(3))
        else:
            wq, wk, wv = _W(a.q_proj), _W(a.k_proj), _W(a.v_proj)
        _norm_any(ul.norm_a, tl.ln_1, "ln")
        _norm_any(ul.norm_f, tl.ln_1, "ln")
        ul.par.fill_(1.0)
        _attn_any(ul.attn, wq, wk, wv, _W(a.out_proj), n_kv=H, scale=1 / float(a.scale_attn), inv_freq=inv,
                  layout="interleave")
        _mlp_any(ul.ffn, tl.mlp, tc.activation_function)
    return u


@torch.no_grad()
def from_gpt_bigcode(target):
    """GPT-BigCode / StarCoder-1 / SantaCoder (GPTBigCodeForCausalLM): learned absolute positions,
    NoPE, a fused c_attn that is multi-query ([q | k | v] with ONE k / v head, tiled onto every host
    head) or per-head MHA, biased LayerNorms, an ungated biased c_fc / c_proj MLP, tied lm_head."""
    tc, m = target.config, target.transformer
    d, H = tc.n_embd, tc.n_head
    hd = d // H
    u = _shell(tc, d=d, L=tc.n_layer, H=H, hd=hd, ff=tc.n_inner or 4 * d, max_pos=tc.n_positions)
    _head_from(u, m.wte, m.ln_f, target.lm_head, "ln")
    _cp(u.pos, m.wpe.weight)
    for ul, tl in zip(u.layers, m.h):
        a = tl.attn
        W, b = _W(a.c_attn), _B(a.c_attn)
        if a.multi_query:
            (wq, wk, wv), (bq, bk, bv), n_kv = W.split([d, hd, hd]), b.split([d, hd, hd]), 1
        else:
            W, b = W.view(H, 3, hd, d), b.view(H, 3, hd)
            wq, wk, wv = (W[:, j].reshape(d, d) for j in range(3))
            bq, bk, bv = (b[:, j].reshape(d) for j in range(3))
            n_kv = H
        _norm_any(ul.norm_a, tl.ln_1, "ln")
        _norm_any(ul.norm_f, tl.ln_2, "ln")
        scale = 1 / math.sqrt(hd) if a.scale_attn_weights else 1.0
        _attn_any(ul.attn, wq, wk, wv, _W(a.c_proj), bq, bk, bv, _B(a.c_proj), n_kv=n_kv, scale=scale)
        _mlp_any(ul.ffn, tl.mlp, tc.activation_function)
    return u


def _alibi_slopes(H):
    """Bloom / Falcon / MPT ALiBi slopes, as the targets build them (fp32)."""
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor
    return build_alibi_tensor(torch.ones(1, 2), H, torch.float32)[:, 0, 1].to(F64)


@torch.no_grad()
def from_bloom(target):
    """BLOOM (BloomForCausalLM): a LayerNorm on the word embeddings (the host's emb_norm branch), ALiBi
    (UAttention.alibi: s += slope_h (s - t), softmax-equal to Bloom's slope_h * s), a per-head fused
    [q | k | v] query_key_value, biased LayerNorms, an ungated biased MLP under Bloom's tanh GeLU."""
    tc, m = target.config, target.transformer
    assert not tc.apply_residual_connection_post_layernorm
    d, H = tc.hidden_size, tc.n_head
    hd = d // H
    u = _shell(tc, d=d, L=tc.n_layer, H=H, hd=hd, ff=4 * d)
    _head_from(u, m.word_embeddings, m.ln_f, target.lm_head, "ln")
    u.emb_norm_c.fill_(1.0)
    _norm_any(u.emb_norm, m.word_embeddings_layernorm, "ln")
    slopes = _alibi_slopes(H)
    for ul, tl in zip(u.layers, m.h):
        a = tl.self_attention
        W, b = _W(a.query_key_value).view(H, 3, hd, d), _B(a.query_key_value).view(H, 3, hd)
        _norm_any(ul.norm_a, tl.input_layernorm, "ln")
        _norm_any(ul.norm_f, tl.post_attention_layernorm, "ln")
        _attn_any(ul.attn, *(W[:, j].reshape(d, d) for j in range(3)), _W(a.dense),
                  *(b[:, j].reshape(d) for j in range(3)), _B(a.dense), n_kv=H, scale=a.inv_norm_factor)
        ul.attn.alibi.copy_(slopes * a.beta)
        _mlp_any(ul.ffn, tl.mlp, "gelu_new")              # BloomGelu: the tanh approximation
    return u


@torch.no_grad()
def from_opt(target):
    """OPT (OPTForCausalLM): learned positions OFFSET BY 2 (the host's pos table = rows 2:), biased
    q / k / v / out projections, an ungated biased fc1 / fc2 MLP under ReLU (UFFN.c_relu), a project_in /
    project_out pair (word_embed_proj_dim != hidden) folded into the embedding table and lm_head.
    Pre-LayerNorm (do_layer_norm_before) layers map one-to-one. Post-LayerNorm OPT (opt-350m:
    x <- LN1(x + attn(x)), x <- LN2(x + ffn(x))) runs each layer as TWO host layers whose stream norm
    (sn_out, the CENO stream-norm branch) is that LayerNorm: [attention, LN1] then [FFN, LN2], their
    pre-norms the identity."""
    tc, m = target.config, target.model.decoder
    post = not tc.do_layer_norm_before
    d, H = tc.hidden_size, tc.num_attention_heads
    hd = d // H
    u = _shell(tc, d=d, L=tc.num_hidden_layers * (2 if post else 1), H=H, hd=hd, ff=tc.ffn_dim,
               max_pos=tc.max_position_embeddings, stream_norms=post)
    emb, head = _W(m.embed_tokens), _W(target.lm_head)
    if m.project_in is not None:
        emb = emb @ _W(m.project_in).T
    if m.project_out is not None:
        head = head @ _W(m.project_out)
    _cp(u.embed.weight, emb)
    _cp(u.lm_head.weight, head)
    _norm_any(u.norm, m.final_layer_norm, "ln" if m.final_layer_norm is not None else "none")
    off = m.embed_positions.offset
    _cp(u.pos, m.embed_positions.weight[off: off + tc.max_position_embeddings])
    for i, tl in enumerate(m.layers):
        a = tl.self_attn
        ua, uf = (u.layers[2 * i], u.layers[2 * i + 1]) if post else (u.layers[i], u.layers[i])
        if post:
            _norm_any(ua.norm_a, None, "none")
            _norm_any(uf.norm_f, None, "none")
            _norm_any(ua.sn_out, tl.self_attn_layer_norm, "ln"); ua.sn_c[1] = 1
            _norm_any(uf.sn_out, tl.final_layer_norm, "ln"); uf.sn_c[1] = 1
        else:
            _norm_any(ua.norm_a, tl.self_attn_layer_norm, "ln")
            _norm_any(uf.norm_f, tl.final_layer_norm, "ln")
        _attn_any(ua.attn, *_qkv(a), n_kv=H, scale=a.scaling)
        _ungated_ffn_from(uf.ffn, _W(tl.fc1), _B(tl.fc1), _W(tl.fc2), _B(tl.fc2), tc.activation_function)
    return u


@torch.no_grad()
def from_falcon(target):
    """Falcon (FalconForCausalLM): the fused query_key_value in all three layouts (new decoder
    architecture: per KV group [q ... | k | v]; multi-query [q | k | v] with one k / v head; per-head
    MHA), a parallel attention + MLP block (parallel_attn: ONE input LayerNorm, or ln_attn / ln_mlp in
    the new architecture: ULayer.par) or sequential, RoPE (rotate-half) or ALiBi (Falcon rounds the
    slopes to bf16 and scales the bias by 1 / sqrt(hd) too), biased or bias-free projections, an
    ungated MLP."""
    tc, m = target.config, target.transformer
    d, H = tc.hidden_size, tc.num_attention_heads
    hd = d // H
    u = _shell(tc, d=d, L=tc.num_hidden_layers, H=H, hd=hd, ff=getattr(tc, "ffn_hidden_size", None) or 4 * d)
    _head_from(u, m.word_embeddings, m.ln_f, target.lm_head, "ln")
    rope = getattr(m, "rotary_emb", None)
    slopes = _alibi_slopes(H).to(torch.bfloat16).to(F64) if tc.alibi else None
    for ul, tl in zip(u.layers, m.h):
        a = tl.self_attention
        W, b = _W(a.query_key_value), _B(a.query_key_value)
        if a.new_decoder_architecture:
            n_kv = a.num_kv_heads
            Wg = W.view(n_kv, H // n_kv + 2, hd, d)
            wq, wk, wv = Wg[:, :-2].reshape(H * hd, d), Wg[:, -2].reshape(n_kv * hd, d), Wg[:, -1].reshape(n_kv * hd, d)
            bq = bk = bv = None
            if b is not None:
                bg = b.view(n_kv, H // n_kv + 2, hd)
                bq, bk, bv = bg[:, :-2].reshape(-1), bg[:, -2].reshape(-1), bg[:, -1].reshape(-1)
        elif a.multi_query:
            n_kv = 1
            wq, wk, wv = W.split([H * hd, hd, hd])
            bq, bk, bv = b.split([H * hd, hd, hd]) if b is not None else (None,) * 3
        else:
            n_kv = H
            Wg = W.view(H, 3, hd, d)
            wq, wk, wv = (Wg[:, j].reshape(d, d) for j in range(3))
            bq, bk, bv = (b.view(H, 3, hd)[:, j].reshape(d) for j in range(3)) if b is not None else (None,) * 3
        if a.new_decoder_architecture and hasattr(tl, "ln_attn"):
            _norm_any(ul.norm_a, tl.ln_attn, "ln")
            _norm_any(ul.norm_f, tl.ln_mlp, "ln")
            ul.par.fill_(1.0)
        else:
            _norm_any(ul.norm_a, tl.input_layernorm, "ln")
            if tc.parallel_attn:
                _norm_any(ul.norm_f, tl.input_layernorm, "ln")
                ul.par.fill_(1.0)
            else:
                _norm_any(ul.norm_f, tl.post_attention_layernorm, "ln")
        _attn_any(ul.attn, wq, wk, wv, _W(a.dense), bq, bk, bv, _B(a.dense), n_kv=n_kv, scale=a.inv_norm_factor,
                  inv_freq=None if tc.alibi else rope.inv_freq, att_scale=1.0 if tc.alibi else rope.attention_scaling)
        if tc.alibi:
            ul.attn.alibi.copy_(slopes * a.inv_norm_factor)
        _mlp_any(ul.ffn, tl.mlp, tc.activation)
    return u


# --- mixture-of-experts FFNs: the routed bank + the shared path of UFFN -------------------------------

def _mlp_of(tl):
    return next(m for m in (getattr(tl, n, None) for n in ("mlp", "block_sparse_moe", "feed_forward", "shared_mlp"))
                if m is not None)


def _expert_ws(mlp):
    """[(gate, up, down)] (out, in) weights of a routed bank, or None for a dense MLP."""
    ex = getattr(mlp, "experts", None)
    if isinstance(ex, nn.ModuleList):
        e0 = ex[0]
        if hasattr(e0, "w1"):                              # Mixtral: w1 gate, w3 up, w2 down
            return [(_W(e.w1), _W(e.w3), _W(e.w2)) for e in ex]
        return [(_W(e.gate_proj), _W(e.up_proj), _W(e.down_proj)) for e in ex]
    if ex is not None and isinstance(getattr(ex, "gate_up_proj", None), torch.Tensor) and ex.gate_up_proj.dim() == 3:
        gu, dn = ex.gate_up_proj.detach().to(F64), ex.down_proj.detach().to(F64)   # transformers 5: (E, 2f, d), (E, d, f)
        import inspect
        if "@ self.gate_up_proj" in inspect.getsource(type(ex)):                 # x @ W layout: (E, d, 2f), (E, f, d)
            gu, dn = gu.transpose(1, 2), dn.transpose(1, 2)
        f = gu.shape[1] // 2
        return [(gu[e, :f], gu[e, f:], dn[e]) for e in range(gu.shape[0])]
    if hasattr(mlp, "input_linear") and hasattr(mlp, "router"):   # Granite-MoE parallel experts
        wi, wo = mlp.input_linear.weight.detach().to(F64), mlp.output_linear.weight.detach().to(F64)
        f = wi.shape[1] // 2
        return [(wi[e, :f], wi[e, f:], wo[e]) for e in range(wi.shape[0])]
    return None


def _expert_bs(mlp):
    """[(gate, up, down)] biases of a routed bank (None where absent)."""
    ex = getattr(mlp, "experts", None)
    if ex is not None and getattr(ex, "gate_up_proj_bias", None) is not None:
        gb, db = ex.gate_up_proj_bias.detach().to(F64), ex.down_proj_bias.detach().to(F64)
        f = gb.shape[1] // 2
        return [(gb[e, :f], gb[e, f:], db[e]) for e in range(gb.shape[0])]
    if not isinstance(ex, nn.ModuleList) or hasattr(ex[0], "w1"):
        return None
    return [(_B(e.gate_proj), _B(e.up_proj), _B(e.down_proj)) for e in ex]


def _shared_of(mlp):
    for n in ("shared_expert", "shared_experts", "shared_mlp"):
        s = getattr(mlp, n, None)
        if s is not None:
            return s
    return None


def _ffn_dims(layers, tc):
    """(experts, expert_ff, top_k, shared_ff) over every layer's FFN: the host's UFFN widths."""
    E = f = k = 1
    sff = 1
    for tl in layers:
        mlp = _mlp_of(tl)
        ws = _expert_ws(mlp)
        if ws is None:
            dense = mlp
        else:
            E, f = len(ws), ws[0][0].shape[0]
            k = int(getattr(mlp, "top_k", None) or getattr(getattr(mlp, "router", None), "top_k", None)
                    or getattr(getattr(mlp, "gate", None), "top_k", None) or tc.num_experts_per_tok)
            dense = _shared_of(mlp) or getattr(tl, "shared_mlp", None)
        if dense is not None:
            for n in ("gate_up_proj", "gate_proj", "up_proj", "c_fc", "fc1", "fc_in", "dense_h_to_4h", "input_linear"):
                if hasattr(dense, n):
                    w = getattr(dense, n).weight
                    sff = max(sff, w.shape[0] // (2 if n in ("gate_up_proj", "input_linear") else 1))
                    break
    return E, f, k, sff


# softmax top-k routers that always / never renormalise their k weights (the others read norm_topk_prob)
_RENORM_ALWAYS = ("granitemoe", "granitemoeshared", "granitemoehybrid", "granitemoe_swa", "qwen3_vl_moe_text", "mixtral",
                  "ernie4_5_moe", "hunyuan_v1_moe")
_RENORM_NEVER = ("jamba",)


@torch.no_grad()
def _router_from(f: UFFN, mlp, tc):
    """The router of a routed bank, read off the target's router (and MoE block) module and its code:
      softmax top-k, renormalised or not (Mixtral, Qwen-MoE, OLMoE, HunYuan-MoE, Mellum; Granite-MoE's
      softmax over the top-k logits = softmax + renorm; ERNIE-4.5-MoE selects on softmax + bias);
      DeepSeek-V2: softmax top-k, greedy or group-limited greedy (group score = max), times
      routed_scaling_factor, no renorm;
      sigmoid scores selected on scores + e_score_correction_bias (DeepSeek-V3, GLM-4-MoE, Nemotron-H,
      EXAONE-MoE, Solar-Open, HY-V3, MiniMax-M2, Laguna), optionally within the top topk_group groups
      (group score = sum of the top-2; the masked experts set to 0 or -inf as the target's code does),
      renormalised by (sum + eps), times the routed scaling; Laguna softcaps the router logits first."""
    import inspect
    gate = getattr(mlp, "gate", None)
    if gate is None:
        gate = mlp.router
    lin = (getattr(gate, "layer", None) or getattr(gate, "wg", None)           # Granite-MoE / HunYuan / AFMoE wrappers
           or (gate.gate if isinstance(getattr(gate, "gate", None), nn.Linear) else gate))
    _cp(f.router.weight, lin.weight)
    if getattr(lin, "bias", None) is not None:
        _cp(f.router.bias, lin.bias)
    try:
        src = inspect.getsource(type(gate)) + inspect.getsource(type(mlp))
    except (OSError, TypeError):
        src = ""
    get = lambda *names: next((getattr(o, n) for n in names for o in (gate, mlp) if getattr(o, n, None) is not None), None)
    if hasattr(gate, "topk_method"):                                     # DeepSeek-V2
        f.route_scale.fill_(float(gate.routed_scaling_factor))
        if gate.topk_method == "group_limited_greedy":
            f.grp.copy_(torch.tensor([float(gate.num_group), float(gate.topk_group), 1.0, 0.0], dtype=F64))
        else:
            assert gate.topk_method == "greedy", gate.topk_method
        return
    cap = get("router_logit_softcapping")
    if cap:
        f.router_cap.fill_(float(cap))
    bias = get("e_score_correction_bias", "expert_bias")
    st = getattr(mlp, "moe_statics", None)
    if st is not None:                                                   # ERNIE-4.5: selection on softmax + bias
        bias = st.e_score_correction_bias.reshape(-1)
    if bias is not None:
        _cp(f.sel_bias, bias)
    if "sigmoid" in "\n".join(l for l in src.split("\n") if "shared" not in l):   # (not the shared-expert gate)
        f.score_c.fill_(1.0)
    G = get("n_group", "num_group")
    if G and int(G) > 1 and int(get("topk_group")) < int(G):
        fill = float("-inf") if 'masked_fill(~score_mask.bool(), float("-inf"))' in src else 0.0
        f.grp.copy_(torch.tensor([float(G), float(get("topk_group")), 0.0, fill], dtype=F64))
    if tc.model_type in _RENORM_ALWAYS:
        renorm = True
    elif tc.model_type in _RENORM_NEVER:
        renorm = False
    else:
        renorm = get("norm_topk_prob")
        if renorm is None:
            renorm = getattr(tc, "norm_topk_prob", None)
        if renorm is None:                                               # unconditional in the code
            renorm = "sum(dim=-1, keepdim=True)" in src
    f.renorm.fill_(float(bool(renorm)))
    if "1e-20" in src:
        f.renorm_eps.fill_(1e-20)
    scale = get("routed_scaling_factor", "router_scaling_factor", "route_scale")
    if scale is not None and any(n in src.split("def forward", 1)[-1] for n in ("scaling_factor", "route_scale")):
        f.route_scale.fill_(float(scale))


@torch.no_grad()
def _ffn_any(f: UFFN, mlp, tc, act, shared=None):
    """A layer's FFN: a dense MLP (the shared path), or a routed MoE (+ shared expert): softmax router
    (Mixtral, Qwen-MoE, OLMoE, ERNIE-4.5-MoE with its selection bias on the softmax scores) with or
    without top-k renormalisation, a sigmoid-gated shared expert (Qwen2-MoE: s_coef [0, 1]); Granite-MoE's
    softmax over the top-k logits is the softmax + renorm branch."""
    ws = _expert_ws(mlp)
    if ws is None:
        _mlp_any(f, mlp, act)
        return
    _router_from(f, mlp, tc)
    for e, (g, up, dn) in enumerate(ws):
        _cp(f.w_gate[e], g); _cp(f.w_up[e], up); _cp(f.w_down[e], dn)
    for e, bs in enumerate(_expert_bs(mlp) or []):          # routed-expert biases (ERNIE-4.5 use_bias)
        for dst, b in zip((f.b_gate, f.b_up, f.b_down), bs):
            if b is not None:
                _cp(dst[e], b)
    sh = shared if shared is not None else _shared_of(mlp)
    if sh is not None:
        _mlp_any(f, sh, act)
        if getattr(mlp, "shared_expert_gate", None) is not None:
            _cp(f.s_router.weight, _W(mlp.shared_expert_gate))
            f.s_coef.copy_(torch.tensor([0.0, 1.0]))
    else:
        _set_act(f, act, gated=True)


@torch.no_grad()
def _mla_from(a: UAttention, ml, *, d, H, nope, rope, vd, kl, ql, inv_freq, att_s, interleave, scale):
    """Multi-head latent attention (DeepSeek-V2 / V3, GLM-4-MoE-Lite, Kimi-K2) -> the host's MHA over
    [x, nb]: nb = the q / kv latents (q_a_proj, kv_a_proj[:kv_lora_rank]) RMS-normed per group by the
    attention normalized-latent branch (pn, n_avg, n_w). Per host head: the nope query / key rows read
    nb through q_b / kv_b, the rope rows read x (the shared k_pe is every head's key rope part, scaled
    by the rotary attention_scaling on q and k), value rows read nb through kv_b."""
    hd = a.hd
    qhd = nope + rope
    assert getattr(ml.kv_a_proj_with_mqa, "bias", None) is None
    kva = _W(ml.kv_a_proj_with_mqa)
    norms = ([ml.q_a_layernorm] if ql else []) + [ml.kv_a_layernorm]
    eps = {float(getattr(n, "variance_epsilon", getattr(n, "eps", 1e-6))) for n in norms}
    assert len(eps) == 1, eps
    a.pn.weight.copy_(torch.cat(([_W(ml.q_a_proj)] if ql else []) + [kva[:kl]]))
    blk = lambda n: torch.full((n, n), 1 / n, dtype=F64)
    a.n_avg.copy_(torch.block_diag(*([blk(ql)] if ql else []), blk(kl)))
    a.n_w.copy_(torch.cat([n.weight.detach().to(F64) for n in norms]))
    a.n_eps.fill_(eps.pop())
    r2, h2 = rope // 2, hd // 2
    rope_host = list(range(r2)) + [h2 + i for i in range(r2)]
    nope_host = [i for i in range(hd) if i not in rope_host][:nope]
    rope_src = ([nope + 2 * i for i in range(r2)] + [nope + 2 * i + 1 for i in range(r2)] if interleave
                else list(range(nope, qhd)))
    nq, nkv = slice(d, d + ql), slice(d + ql, d + ql + kl)
    kvb = _W(ml.kv_b_proj)
    qsrc, qcols = (_W(ml.q_b_proj), nq) if ql else (_W(ml.q_proj), slice(0, d))
    wo = _W(ml.o_proj)
    for h in range(H):
        r0 = h * hd
        qh = qsrc[h * qhd:(h + 1) * qhd]
        kh = kvb[h * (nope + vd): h * (nope + vd) + nope]
        for j, hj in enumerate(nope_host):
            a.q_proj.weight[r0 + hj, qcols] = qh[j]
            a.k_proj.weight[r0 + hj, nkv] = kh[j]
        for src_row, hj in zip(rope_src, rope_host):
            a.q_proj.weight[r0 + hj, qcols] = qh[src_row] * att_s
            a.k_proj.weight[r0 + hj, :d] = kva[kl + src_row - nope] * att_s
        a.v_proj.weight[r0: r0 + vd, nkv] = kvb[h * (nope + vd) + nope:(h + 1) * (nope + vd)]
        a.o_proj.weight[:, r0: r0 + vd] = wo[:, h * vd:(h + 1) * vd]
    if _B(ml.o_proj) is not None:
        _cp(a.o_proj.bias, _B(ml.o_proj))
    a.inv_freq.zero_()
    a.inv_freq[:r2] = inv_freq.detach().to(F64)
    a.scale.fill_(float(scale))


def _mla_dims(tc):
    H, nope, rope, vd = tc.num_attention_heads, tc.qk_nope_head_dim, tc.qk_rope_head_dim, tc.v_head_dim
    hd = max(nope + rope, vd)
    return H, nope, rope, vd, tc.kv_lora_rank, tc.q_lora_rank or 0, hd + hd % 2


@torch.no_grad()
def from_deepseek(target):
    """DeepSeek-V2 / V3 (DeepseekV2ForCausalLM / DeepseekV3ForCausalLM; transformers 5 also GLM-4-MoE-Lite,
    Youtu and MiniCPM3 with its scale_emb / scale_depth / logits_scaling folded): MLA (_mla_from; V2's complex
    rotary and V3's rope_interleave are the interleaved layout, V3's YaRN mscale^2 is in its scaling),
    dense first layers, then a routed MoE with shared experts: V2's softmax top-k (greedy or
    group-limited greedy, group score = max) times routed_scaling_factor; V3's sigmoid scores selected on
    scores + e_score_correction_bias within the top topk_group groups (UFFN.grp), renormalised by
    (sum + 1e-20), times routed_scaling_factor."""
    tc, m = target.config, target.model
    H, nope, rope, vd, kl, ql, hd = _mla_dims(tc)
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    d = tc.hidden_size
    u = _shell(tc, d=d, L=tc.num_hidden_layers, H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k,
               attn_norm_feats=ql + kl)
    es = getattr(m.embed_tokens, "embed_scale", None)                    # MiniCPM3 scale_emb
    ls = getattr(tc, "logits_scaling", None) if tc.model_type == "minicpm3" else None
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, emb_scale=None if es is None else es.detach().to(F64),
               logit_scale=None if ls is None else 1.0 / float(ls))
    re_ = m.rotary_emb
    interleave = tc.model_type == "deepseek_v2" or bool(getattr(tc, "rope_interleave", False))
    for ul, tl in zip(u.layers, m.layers):
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _mla_from(ul.attn, tl.self_attn, d=d, H=H, nope=nope, rope=rope, vd=vd, kl=kl, ql=ql,
                  inv_freq=re_.inv_freq, att_s=float(re_.attention_scaling), interleave=interleave,
                  scale=tl.self_attn.scaling)
        _ffn_any(ul.ffn, tl.mlp, tc, tc.hidden_act)
        rp = getattr(tc, "rope_parameters", None) or {}
        if isinstance(rp, dict) and rp.get("llama_4_scaling_beta"):         # Mistral-4: the Llama-4 query temperature
            ul.attn.qtemp.copy_(torch.tensor([float(rp["llama_4_scaling_beta"]),
                                              float(rp["original_max_position_embeddings"])], dtype=F64))
        if getattr(tl.mlp, "post_mlp_layernorm", None) is not None:        # AXK1: a post-norm on its MoE output
            _norm_any(ul.post_f, tl.mlp.post_mlp_layernorm)
            ul.post_coef[2] = torch.tensor([0.0, 1.0])
        if getattr(tl, "residual_scale", None) is not None:                # MiniCPM3 scale_depth / sqrt(L)
            _fold_residual(ul, float(tl.residual_scale))
    return u


@torch.no_grad()
def from_gpt_oss(target):
    """GPT-OSS (GptOssForCausalLM): biased GQA with a learned per-head attention sink (a zero-value
    null key whose logit is the sink: UAttention.null_logit), alternating sliding / full layers, YaRN
    (attention_scaling folded into the scale); a router WITH a bias, top-k on the logits and a softmax
    over those k (= the softmax + renorm branch); biased experts (UFFN.b_gate / b_up / b_down) whose
    fused gate_up interleaves gate (even) and up (odd) columns, under swigluoai:
        (clamp(u, -L, L) + 1) * g' sigmoid(alpha g'),  g' = min(g, L)
    = (u_clip + up_shift) * silu(alpha g'') / alpha with g'' = alpha g clipped at alpha L: the gate rows
    are scaled by alpha (clip_lim [alpha L, L]), up_shift = 1, and 1 / alpha folds into w_down."""
    tc, m = target.config, target.model
    H, n_kv, hd = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    ex0 = m.layers[0].mlp.experts
    E, f = ex0.gate_up_proj.shape[0], ex0.gate_up_proj.shape[2] // 2
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=1, experts=E, expert_ff=f,
               top_k=tc.num_experts_per_tok)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    rope = m.rotary_emb
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _attn_any(ul.attn, *_qkv(a), n_kv=n_kv, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling, window=_sliding(tc, i))
        _cp(ul.attn.null_logit, a.sinks)
        fl, ex, rt = ul.ffn, tl.mlp.experts, tl.mlp.router
        alpha, lim = float(ex.alpha), float(ex.limit)
        _cp(fl.router.weight, rt.weight); _cp(fl.router.bias, rt.bias)
        fl.renorm.fill_(1.0)
        gu, gb = ex.gate_up_proj.detach().to(F64), ex.gate_up_proj_bias.detach().to(F64)
        _cp(fl.w_gate, gu[:, :, ::2].transpose(1, 2) * alpha); _cp(fl.b_gate, gb[:, ::2] * alpha)
        _cp(fl.w_up, gu[:, :, 1::2].transpose(1, 2)); _cp(fl.b_up, gb[:, 1::2])
        _cp(fl.w_down, ex.down_proj.detach().to(F64).transpose(1, 2) / alpha); _cp(fl.b_down, ex.down_proj_bias)
        fl.c_clip.fill_(1.0)
        fl.clip_lim.copy_(torch.tensor([alpha * lim, lim], dtype=F64))
        fl.up_shift.fill_(1.0)
    return u


# --- Mamba hybrids: the SSM sub-block on mamba layers, attention on the others --------------------------

def _ssm_dims(mixers):
    """UConfig SSM widths from a Mamba-2 mixer (every mamba layer of a model has the same shape)."""
    mx = mixers[0] if mixers else None
    if mx is None:
        return {}
    return dict(ssm_in=mx.intermediate_size + mx.conv_dim + mx.num_heads, conv_k=mx.conv_kernel_size,
                ssm_heads=mx.num_heads, dk=mx.ssm_state_size, dv=mx.head_dim)


@torch.no_grad()
def _mamba2_into(ul, mx, norm):
    """A Mamba-2 mixer (HF Mamba2 / Bamba / Granite-4 / Falcon-H1 / Nemotron-H naming) into the layer's
    SSM sub-block (mamba2_block; its gated RMSNorm per group when n_groups > 1 and the target groups)."""
    s = mamba2_block(mx)
    g, I = mx.n_groups, mx.intermediate_size
    gs = getattr(mx.norm, "group_size", I)
    if gs != I:
        s.avg.copy_(torch.block_diag(*[torch.full((gs, gs), 1 / gs, dtype=F64)] * (I // gs)))
    ul.ssm.load_state_dict(s.state_dict())
    _norm_any(ul.norm_s, norm)


@torch.no_grad()
def from_bamba(target):
    """Bamba and Granite-4.0-H (BambaForCausalLM / GraniteMoeHybridForCausalLM): Mamba-2 layers (the SSM
    sub-block, mamba2_block) and GQA attention layers (partial rotary, or NoPE: Granite's
    position_embedding_type), each followed by the FFN (Bamba: a dense SwiGLU; Granite-4: a routed MoE with
    its softmax over the top-k logits + the shared_mlp, or the shared_mlp alone), Granite's multipliers
    folded into weights (embedding, attention scale, residual: attention / SSM / FFN outputs, logits)."""
    tc, m = target.config, target.model
    mixers = [tl.mamba for tl in m.layers if getattr(tl, "mamba", None) is not None]
    at0 = next((tl.self_attn for tl in m.layers if getattr(tl, "self_attn", None) is not None), None)
    H = tc.num_attention_heads
    hd = getattr(at0, "head_dim", None) or tc.hidden_size // H
    granite = tc.model_type == "granitemoehybrid"
    E, ef, k, sff = _ffn_dims(m.layers, tc) if granite else (1, 1, 1, tc.intermediate_size)
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k,
               **_ssm_dims(mixers))
    em = float(getattr(tc, "embedding_multiplier", 1.0) or 1.0)
    ls = float(getattr(tc, "logits_scaling", 1.0) or 1.0)
    _head_from(u, m.embed_tokens, getattr(m, "norm", None) or m.final_layernorm, target.lm_head, emb_scale=em,
               logit_scale=1.0 / ls)
    rope = getattr(m, "rotary_emb", None)
    for ul, tl in zip(u.layers, m.layers):
        if getattr(tl, "mamba", None) is not None:
            _mamba2_into(ul, tl.mamba, tl.input_layernorm)
        else:
            a = tl.self_attn
            _norm_any(ul.norm_a, tl.input_layernorm)
            _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=getattr(a, "scaling", hd ** -0.5),
                      inv_freq=None if rope is None else rope.inv_freq,
                      att_scale=1.0 if rope is None else rope.attention_scaling)
        if granite:
            _norm_any(ul.norm_f, tl.post_attention_layernorm)
            moe = getattr(tl, "block_sparse_moe", None)
            if moe is not None:
                _ffn_any(ul.ffn, moe, tc, tc.hidden_act, shared=tl.shared_mlp)
            else:
                _mlp_any(ul.ffn, tl.shared_mlp, tc.hidden_act)
            r = float(tc.residual_multiplier)
            _fold_residual(ul, r)
            ul.ssm.out_proj.weight.mul_(r); ul.ssm.out_proj.bias.mul_(r)
        else:
            _norm_any(ul.norm_f, tl.pre_ff_layernorm)
            _mlp_any(ul.ffn, tl.feed_forward, tc.hidden_act)
    return u


@torch.no_grad()
def from_falcon_h1(target):
    """Falcon-H1 (FalconH1ForCausalLM): every layer runs a Mamba-2 mixer and GQA attention IN PARALLEL on
    one input norm (ULayer.seq = 0), then a SwiGLU. Its muP multipliers are all folds: the in_proj rows
    by mup_vector (per z / x / B / C / dt section) * ssm_in_multiplier, out_proj by ssm_out_multiplier,
    q / k / v by attention_in_multiplier and k by key_multiplier, o_proj by attention_out_multiplier,
    the gate rows / down by mlp_multipliers, the embedding and lm_head multipliers. Its gated RMSNorm
    (per group, norm-then-gate or gate-then-norm by mamba_norm_before_gate) or none (mamba_rms_norm off)
    is the SSM output branch."""
    from types import SimpleNamespace as NS
    tc, m = target.config, target.model
    mixers = [tl.mamba for tl in m.layers]
    H, hd = tc.num_attention_heads, tc.head_dim
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=tc.intermediate_size, **_ssm_dims(mixers))
    _head_from(u, m.embed_tokens, m.final_layernorm, target.lm_head, emb_scale=float(tc.embedding_multiplier),
               logit_scale=float(tc.lm_head_multiplier))
    rope = m.rotary_emb
    gm, dm = tc.mlp_multipliers
    for ul, tl in zip(u.layers, m.layers):
        mx, a = tl.mamba, tl.self_attn
        I = mx.intermediate_size
        norm = mx.norm if mx.mamba_rms_norm else NS(weight=torch.ones(I), variance_epsilon=1e-6)
        sh = NS(**{k: getattr(mx, k) for k in ("intermediate_size", "ssm_state_size", "num_heads", "head_dim", "n_groups",
                                               "conv_dim", "hidden_size", "conv_kernel_size", "conv1d", "dt_bias",
                                               "A_log", "D", "out_proj")},
                in_proj=NS(weight=_W(mx.in_proj) * (mx.mup_vector.reshape(-1, 1).to(F64) * float(mx.ssm_in_multiplier)),
                           bias=None if _B(mx.in_proj) is None else _B(mx.in_proj) * mx.mup_vector.reshape(-1).to(F64)),
                norm=NS(weight=norm.weight, variance_epsilon=norm.variance_epsilon, group_size=I // mx.n_groups))
        _mamba2_into(ul, sh, tl.input_layernorm)
        s = ul.ssm
        s.out_coef.zero_()
        if not mx.mamba_rms_norm:
            s.out_coef[2] = 1                                        # y * silu(z)
        elif mx.norm.norm_before_gate:
            s.out_coef[0] = 1                                        # norm(y) * silu(z)
        else:
            s.out_coef[1] = 1                                        # norm(y * silu(z))
        s.out_proj.weight.mul_(float(tl.ssm_out_multiplier)); s.out_proj.bias.mul_(float(tl.ssm_out_multiplier))
        ul.seq.fill_(0.0)                                            # attention and the SSM read the same input
        _norm_any(ul.norm_a, tl.input_layernorm)
        wq, wk, wv, wo, bq, bk, bv, bo = _qkv(a)
        aim, km, aom = float(tl.attention_in_multiplier), float(a.key_multiplier), float(tl.attn_out_multiplier)
        _attn_any(ul.attn, wq * aim, wk * aim * km, wv * aim, wo * aom, bq, None if bk is None else bk * km, bv,
                  None if bo is None else bo * aom, n_kv=tc.num_key_value_heads, scale=a.scaling,
                  inv_freq=rope.inv_freq, att_scale=rope.attention_scaling)
        _norm_any(ul.norm_f, tl.pre_ff_layernorm)
        _mlp_any(ul.ffn, tl.feed_forward, tc.hidden_act)
        ul.ffn.s_gate.weight.mul_(float(gm)); ul.ffn.s_gate.bias.mul_(float(gm))
        ul.ffn.s_down.weight.mul_(float(dm)); ul.ffn.s_down.bias.mul_(float(dm))
    return u


@torch.no_grad()
def _mamba1_normed_from(s: USSM, mx, w_dt=None, w_b=None, w_c=None, eps=1e-6):
    """A Mamba-1 mixer whose dt / B / C (the x_proj outputs) are RMS-normed before use (Jamba: weighted
    RMSNorms; Falcon-Mamba: weight-free rms_forward) -> _mamba1_from with the knobs read from the
    normalized-feature branch: nb = [B | C | dt] = rms_grouped(x_proj x_conv) * n_w."""
    _mamba1_from(s, mx)
    di, N, Rr = mx.intermediate_size, mx.ssm_state_size, mx.time_step_rank
    D = s.D
    cx = slice(D, D + di)
    W = _W(mx.x_proj)
    s.pk.weight.zero_(); s.pq.weight.zero_(); s.p1.weight.zero_()
    s.pn.weight[:, cx] = torch.cat([W[Rr:Rr + N], W[Rr + N:], W[:Rr]])
    blk = lambda n: torch.full((n, n), 1 / n, dtype=F64)
    s.n_avg.copy_(torch.block_diag(blk(N), blk(N), blk(Rr)))
    one = lambda n, w: torch.ones(n, dtype=F64) if w is None else w.detach().to(F64)
    s.n_w.copy_(torch.cat([one(N, w_b), one(N, w_c), one(Rr, w_dt)]))
    s.n_eps.fill_(float(eps))
    nb = 2 * D
    s.pk.weight[:, nb: nb + N] = torch.eye(N, dtype=F64).repeat(di, 1)
    s.pq.weight[:, nb + N: nb + 2 * N] = torch.eye(N, dtype=F64).repeat(di, 1)
    s.p1.weight[:, nb + 2 * N: nb + 2 * N + Rr] = _W(mx.dt_proj).repeat_interleave(N, 0)


def _mamba1_cfg(mx):
    di, N, Rr = mx.intermediate_size, mx.ssm_state_size, mx.time_step_rank
    return dict(ssm_in=2 * di, conv_k=mx.conv_kernel_size, ssm_heads=di, dk=N, dv=1, ssm_norm_feats=2 * N + Rr)


@torch.no_grad()
def from_falcon_mamba(target):
    """Falcon-Mamba (FalconMambaForCausalLM): Mamba-1 whose B, C and dt are weight-free RMS-normed
    (rms_forward, mixer_rms_eps): the normalized-feature branch of the SSM sub-block."""
    tc, bb = target.config, target.backbone
    mx0 = bb.layers[0].mixer
    u = UniversalElephantine(UConfig(vocab=tc.vocab_size, d=tc.hidden_size, L=tc.num_hidden_layers, **_mamba1_cfg(mx0)))
    _head_from(u, bb.embeddings, bb.norm_f, target.lm_head)
    for ul, tl in zip(u.layers, bb.layers):
        _norm_any(ul.norm_s, tl.norm)
        _mamba1_normed_from(ul.ssm, tl.mixer, eps=tl.mixer.rms_eps)
    return u


@torch.no_grad()
def from_jamba(target):
    """Jamba (JambaForCausalLM): Mamba-1 layers with RMS-normed dt / B / C (weighted JambaRMSNorms: the
    normalized-feature branch) and NoPE GQA attention layers, each followed by a dense SwiGLU or a routed MoE
    (softmax top-k, NOT renormalised)."""
    tc, m = target.config, target.model
    mx0 = next(tl.mamba for tl in m.layers if hasattr(tl, "mamba"))
    at0 = next((tl.self_attn for tl in m.layers if hasattr(tl, "self_attn")), None)
    H = tc.num_attention_heads
    hd = tc.hidden_size // H
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k,
               **_mamba1_cfg(mx0))
    _head_from(u, m.embed_tokens, m.final_layernorm, target.lm_head)
    for ul, tl in zip(u.layers, m.layers):
        if hasattr(tl, "mamba"):
            mx = tl.mamba
            _norm_any(ul.norm_s, tl.input_layernorm)
            _mamba1_normed_from(ul.ssm, mx, mx.dt_layernorm.weight, mx.b_layernorm.weight, mx.c_layernorm.weight,
                                mx.b_layernorm.variance_epsilon)
        else:
            a = tl.self_attn
            _norm_any(ul.norm_a, tl.input_layernorm)
            _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=getattr(a, "scaling", hd ** -0.5))
        _norm_any(ul.norm_f, tl.pre_ff_layernorm)
        _ffn_any(ul.ffn, tl.feed_forward, tc, tc.hidden_act)
    return u


@torch.no_grad()
def from_nemotron_h(target):
    """Nemotron-H / Nemotron-3 (NemotronHForCausalLM, transformers 5): a stack of single-mixer blocks
    x + mixer(norm(x)), the mixer a Mamba-2 (the SSM sub-block, its grouped gated RMSNorm and the
    time_step_limit floor on dt), NoPE GQA attention, an ungated relu^2 MLP, or a latent MoE: sigmoid
    router, selection on scores + e_score_correction_bias within the top topk_group groups (masked to
    -inf), renorm + routed scaling, relu^2 experts on a latent projection (fc1 / fc2 folded into each
    expert's up / down exactly), plus a relu^2 shared expert. One host layer per block."""
    tc, bb = target.config, target.model if hasattr(target, "model") else target.backbone
    blocks = bb.layers
    mixers = [b.mixer for b in blocks if b.block_type == "linear_attention"]
    at0 = next((b.mixer for b in blocks if b.block_type == "full_attention"), None)
    moe0 = next((b.mixer for b in blocks if b.block_type == "moe"), None)
    mlps = [b.mixer for b in blocks if b.block_type == "mlp"] + ([moe0.shared_experts] if moe0 else [])
    H = tc.num_attention_heads
    hd = at0.head_dim if at0 else 2
    sff = max([m_.up_proj.out_features for m_ in mlps] or [1])
    E, ef = (moe0.experts.up_proj.shape[0], moe0.experts.up_proj.shape[1]) if moe0 else (1, 1)
    u = _shell(tc, d=tc.hidden_size, L=len(blocks), H=H, hd=hd, ff=sff, experts=E, expert_ff=ef,
               top_k=moe0.gate.top_k if moe0 else 1, **_ssm_dims(mixers))
    _head_from(u, bb.embeddings, bb.norm_f, target.lm_head)
    for ul, b in zip(u.layers, blocks):
        mx, f = b.mixer, ul.ffn
        if b.block_type == "linear_attention":
            _mamba2_into(ul, mx, b.norm)
        elif b.block_type == "full_attention":
            _norm_any(ul.norm_a, b.norm)
            _attn_any(ul.attn, *_qkv(mx), n_kv=tc.num_key_value_heads, scale=mx.scaling)
        elif b.block_type == "mlp":
            _norm_any(ul.norm_f, b.norm)
            _mlp_any(f, mx, tc.mlp_hidden_act)
        else:
            assert b.block_type == "moe", b.block_type
            _norm_any(ul.norm_f, b.norm)
            _mlp_any(f, mx.shared_experts, tc.mlp_hidden_act)
            g = mx.gate
            _cp(f.router.weight, g.weight)
            f.score_c.fill_(1.0)
            _cp(f.sel_bias, g.e_score_correction_bias)
            f.renorm.fill_(float(bool(g.norm_topk_prob))); f.renorm_eps.fill_(1e-20)
            f.route_scale.fill_(float(g.routed_scaling_factor))
            f.grp.copy_(torch.tensor([float(g.num_group), float(g.topk_group), 0.0, float("-inf")], dtype=F64))
            up, dn = mx.experts.up_proj.detach().to(F64), mx.experts.down_proj.detach().to(F64)
            l1, l2 = mx.fc1_latent_proj, mx.fc2_latent_proj
            if isinstance(l1, nn.Linear):                       # latent experts: fold the projections
                b1 = _B(l1)
                if b1 is not None:
                    f.b_up.copy_(torch.einsum("efl,l->ef", up, b1))
                up = up @ _W(l1)
                dn = _W(l2) @ dn
                if _B(l2) is not None:
                    f.s_down.bias.add_(_B(l2))
            f.w_up.copy_(up); f.w_down.copy_(dn)
    return u


def _rope_of(rotary, layer_type=None):
    """(inv_freq, attention_scaling) of a rotary module, per layer type when it keeps one per type
    (transformers 5: `{layer_type}_inv_freq`)."""
    if layer_type is not None and hasattr(rotary, f"{layer_type}_inv_freq"):
        return getattr(rotary, f"{layer_type}_inv_freq"), getattr(rotary, f"{layer_type}_attention_scaling")
    return rotary.inv_freq, rotary.attention_scaling


@torch.no_grad()
def from_afmoe(target):
    """AFMoE (AfmoeForCausalLM, Arcee Trinity): dual (sandwich) norms around attention and FFN, per-head
    qk-norm, an elementwise sigmoid output gate (gate_proj: the attention gate branch), RoPE on the
    sliding-window layers only (full layers NoPE), muP embedding scale sqrt(d) (folded), dense first layers
    then sigmoid top-k routing on scores + expert_bias, renormalised by (sum + 1e-20), times route_scale,
    plus shared experts."""
    tc, m = target.config, target.model
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=sff,
               experts=E, expert_ff=ef, top_k=k)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head,
               emb_scale=tc.hidden_size ** 0.5 if getattr(tc, "mup_enabled", False) else None)
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.post_a, tl.post_attention_layernorm)
        _norm_any(ul.norm_f, tl.pre_mlp_layernorm)
        _norm_any(ul.post_f, tl.post_mlp_layernorm)
        ul.post_coef[0] = torch.tensor([0.0, 1.0])
        ul.post_coef[2] = torch.tensor([0.0, 1.0])
        inv, att = _rope_of(m.rotary_emb, tc.layer_types[i])
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling,
                  inv_freq=inv if a.is_local_attention else None, att_scale=att, window=a.sliding_window,
                  q_norm=a.q_norm, k_norm=a.k_norm)
        _cp(ul.attn.g_proj.weight, a.gate_proj.weight)
        ul.attn.gate_coef.copy_(torch.tensor([0.0, 1.0]))
        _ffn_any(ul.ffn, tl.mlp, tc, tc.hidden_act)
    return u


@torch.no_grad()
def from_laguna(target):
    """Laguna (LagunaForCausalLM): a per-layer query-head count (num_attention_heads_per_layer: the host
    runs the widest, narrower layers zero-padded, their KV groups tiled per layer), per-head qk-norm,
    sliding / full layers each on its own rotary, a SOFTPLUS output gate per head or per channel
    (UAttention.gate_sp), dense or routed FFNs: router logits softcapped (UFFN.router_cap), sigmoid scores
    selected on scores + e_score_correction_bias, renormalised, the routed sum times the routed scaling
    factor, plus a shared expert."""
    tc, m = target.config, target.model
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    hd = tc.head_dim
    Hs = list(tc.num_attention_heads_per_layer)
    H, n_kv = max(Hs), tc.num_key_value_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    d = tc.hidden_size
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a, Hl = tl.self_attn, Hs[i]
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        wq, wk, wv, wo, bq, bk, bv, bo = _qkv(a)
        pad = lambda w, rows: torch.cat([w, w.new_zeros(rows - w.shape[0], *w.shape[1:])])
        tile = lambda w: None if w is None else pad(_tile_kv(w, n_kv, Hl, hd), H * hd)
        wq = pad(wq, H * hd)
        bq = None if bq is None else pad(bq, H * hd)
        wo = torch.cat([wo, wo.new_zeros(d, (H - Hl) * hd)], 1)
        inv, att = _rope_of(m.rotary_emb, tc.layer_types[i])
        _attn_any(ul.attn, wq, tile(wk), tile(wv), wo, bq, tile(bk), tile(bv), bo, n_kv=H, scale=a.scaling,
                  inv_freq=inv, att_scale=att, window=a.sliding_window, q_norm=a.q_norm, k_norm=a.k_norm)
        g = _W(a.g_proj)
        g = g.repeat_interleave(hd, 0) if a.gate_per_head else g
        ul.attn.g_proj.weight[: g.shape[0]] = g
        ul.attn.gate_coef.zero_()
        ul.attn.gate_sp.fill_(1.0)
        _ffn_any(ul.ffn, tl.mlp, tc, tc.hidden_act)
    return u


@torch.no_grad()
def from_gemma4(target):
    """Gemma-4 text and Gemma-4-Unified text (Gemma4ForCausalLM / Gemma4UnifiedForCausalLM, transformers 5): sqrt(d)-scaled embeddings, sandwich RMSNorms,
    per-head q / k norms and a weight-free v-norm (UAttention.v_c) with softmax scale 1, a different
    head_dim / KV-head count per layer type (sliding vs full: the host runs the wider head, narrower heads
    zero-padded with the norms averaging over their own width, qk_width), proportional (partly NoPE)
    RoPE per layer type, K = V on full layers (attention_k_eq_v: v_proj = the raw k_proj), the last
    num_kv_shared_layers reading the K / V of the last unshared layer of their type (the KV register, one
    slot per type: kv_slot), a dense GeGLU MLP with an optional parallel routed MoE whose paths each have
    their own input / output norms and whose router reads its own scaled RMSNorm (ULayer.ffn_split),
    softmax top-k renormalised then times a per-expert scale (UFFN.exp_scale); per-layer embeddings
    (PLE: UniversalElephantine.ple_*, ULayer.ple_*), a per-layer output scalar (out_scale) and the final
    logit softcap."""
    from types import SimpleNamespace as NS
    tc, m = target.config, target.model
    L, H, d = tc.num_hidden_layers, tc.num_attention_heads, tc.hidden_size
    hds = [tl.self_attn.head_dim for tl in m.layers]
    hd = max(hds)
    E, ef, k, sff = 1, 1, 1, 1
    for tl in m.layers:
        sff = max(sff, tl.mlp.gate_proj.out_features)
        if getattr(tl, "enable_moe_block", False):
            E, ef, k = tl.experts.num_experts, tl.experts.intermediate_dim, tc.top_k_experts
    dp = getattr(tc, "hidden_size_per_layer_input", 0) or 0
    u = UniversalElephantine(UConfig(vocab=tc.vocab_size, d=d, L=L, heads=H, head_dim=hd, shared_ff=sff, experts=E,
                                     expert_ff=ef, top_k=k, ple_dim=dp,
                                     ple_vocab=getattr(tc, "vocab_size_per_layer_input", 0) if dp else 0))
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, emb_scale=m.embed_tokens.embed_scale.detach().to(F64))
    if getattr(tc, "final_logit_softcapping", None):
        u.final_cap.fill_(float(tc.final_logit_softcapping))
    if dp:
        c = float(m.per_layer_input_scale)
        _cp(u.ple_E, _W(m.embed_tokens_per_layer) * m.embed_tokens_per_layer.embed_scale.detach().to(F64) * c)
        _cp(u.ple_P.weight, _W(m.per_layer_model_projection) * float(m.per_layer_model_projection_scale))
        _cp(u.ple_w, m.per_layer_projection_norm.weight.detach().to(F64) * c)
        u.ple_eps.fill_(float(m.per_layer_projection_norm.eps))
    act = tc.hidden_activation
    types = sorted(set(tc.layer_types))
    pad_rows = lambda w, n, h: torch.cat([w.view(n, h, -1), w.new_zeros(n, hd - h, w.shape[-1])], 1).reshape(n * hd, -1)
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a, h = tl.self_attn, hds[i]
        lt = tc.layer_types[i]
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.post_a, tl.post_attention_layernorm)
        _norm_any(ul.norm_f, tl.pre_feedforward_layernorm)
        _norm_any(ul.post_f, tl.post_feedforward_layernorm)
        ul.post_coef[0] = torch.tensor([0.0, 1.0])
        ul.post_coef[2] = torch.tensor([0.0, 1.0])
        wq = pad_rows(_W(a.q_proj), H, h)
        wo = torch.cat([_W(a.o_proj).view(d, H, h), torch.zeros(d, H, hd - h, dtype=F64)], 2).reshape(d, H * hd)
        if a.is_kv_shared_layer:
            n_kv = 1
            wk = wv = torch.zeros(hd, d, dtype=F64)
            kn = None
        else:
            n_kv = a.k_proj.out_features // h
            wk = pad_rows(_W(a.k_proj), n_kv, h)
            wv = pad_rows(_W(a.v_proj) if a.v_proj is not None else _W(a.k_proj), n_kv, h)   # K = V layers
            kn = NS(weight=torch.cat([a.k_norm.weight.detach().to(F64), torch.zeros(hd - h, dtype=F64)]), eps=a.k_norm.eps)
        qn = NS(weight=torch.cat([a.q_norm.weight.detach().to(F64), torch.zeros(hd - h, dtype=F64)]), eps=a.q_norm.eps)
        inv, att = _rope_of(m.rotary_emb, lt)
        _attn_any(ul.attn, wq, wk, wv, wo, n_kv=n_kv, scale=a.scaling, inv_freq=inv, att_scale=att,
                  window=a.sliding_window, q_norm=qn, k_norm=kn)
        at = ul.attn
        at.qk_width.fill_(float(h))
        if not a.is_kv_shared_layer:
            at.v_c.fill_(1.0)
            at.v_eps.fill_(float(a.v_norm.eps))
            if getattr(a.v_norm, "with_scale", False):
                at.v_nw.copy_(torch.cat([a.v_norm.weight.detach().to(F64), torch.zeros(hd - h, dtype=F64)]).expand(H, hd))
        at.kv_slot.fill_(float(types.index(lt)))
        if a.is_kv_shared_layer:
            at.kv_read.fill_(1.0)
        elif a.store_full_length_kv:
            at.kv_write.fill_(1.0)
        f = ul.ffn
        _mlp_any(f, tl.mlp, act)
        if getattr(tl, "enable_moe_block", False):
            ul.ffn_split.fill_(1.0)
            _norm_any(ul.norm_f2, tl.pre_feedforward_layernorm_2)
            _norm_any(ul.post_f1, tl.post_feedforward_layernorm_1)
            _norm_any(ul.post_f2, tl.post_feedforward_layernorm_2)
            rt = tl.router
            ul.norm_r.weight.copy_(rt.scale.detach().to(F64) * float(rt.scalar_root_size))
            ul.norm_r.eps.fill_(float(rt.norm.eps))
            _cp(f.router.weight, rt.proj.weight)
            f.renorm.fill_(1.0)
            _cp(f.exp_scale, rt.per_expert_scale)
            gu, dn = tl.experts.gate_up_proj.detach().to(F64), tl.experts.down_proj.detach().to(F64)
            _cp(f.w_gate, gu[:, :ef]); _cp(f.w_up, gu[:, ef:]); _cp(f.w_down, dn)
        if dp:
            ul.ple_c.fill_(1.0)
            _cp(ul.ple_gate.weight, tl.per_layer_input_gate.weight)
            _cp(ul.ple_proj.weight, tl.per_layer_projection.weight)
            _norm_any(ul.ple_post, tl.post_per_layer_input_norm)
            ul.ple_tanh.fill_(float(str(act) in _GELU_TANH))
        ul.out_scale.fill_(float(tl.layer_scalar))
    return u


@torch.no_grad()
def _dsa_from(a: UAttention, ix, *, d, ql, inv_freq, att_s, interleave):
    """A DeepSeek-V3.2 / GLM-5 DSA indexer (per-token top-k: the host's k-pool indexer with pool 1) -> the
    attention's DSA branch: q = wq_b(q latent) (the q-latent columns of [x, nb]), k = LayerNorm(wk x),
    RoPE on the first qk_rope_head_dim of both (half or interleaved: permuted onto the host's pairs; the
    rotary attention_scaling on the rotated rows / LayerNorm affine), score sum_h w_h relu(q_h k / sqrt(dI))
    with w = weights_proj x / sqrt(nI), the top index_topk earlier keys visible."""
    nI, dI, r = ix.n_heads, ix.head_dim, ix.qk_rope_head_dim
    perm = _rope_perm(dI, r, "interleave" if interleave else "half")
    perm = torch.arange(dI) if perm is None else perm
    sc = torch.ones(dI, dtype=F64)
    sc[: r // 2] = att_s
    sc[dI // 2: dI // 2 + r // 2] = att_s
    a.iq.weight.zero_(); a.ik.weight.zero_(); a.iw.weight.zero_()
    a.iq.weight[:, d: d + ql] = (_W(ix.wq_b).view(nI, dI, ql)[:, perm] * sc[None, :, None]).reshape(nI * dI, ql)
    a.ik.weight[:, :d] = _W(ix.wk)[perm]
    a.ik_w.copy_(ix.k_norm.weight.detach().to(F64)[perm] * sc)
    a.ik_b.copy_(ix.k_norm.bias.detach().to(F64)[perm] * sc)
    a.ik_eps.fill_(float(ix.k_norm.eps))
    a.iw.weight[:, :d] = _W(ix.weights_proj)
    a.i_inv.zero_()
    a.i_inv[: r // 2] = inv_freq.detach().to(F64)
    a.dsa_pool.fill_(1.0); a.dsa_topk.fill_(float(ix.index_topk)); a.dsa_tail.fill_(0.0)


@torch.no_grad()
def from_dsa(target):
    """DeepSeek-V3.2 and GLM-5 (DeepseekV32ForCausalLM / GlmMoeDsaForCausalLM, transformers 5): MLA
    (_mla_from) under DeepSeek sparse attention, each query seeing only the index_topk earlier keys its
    lightning indexer scores highest (_dsa_from; GLM-5's 'shared' indexer layers reuse the previous
    layer's selection: dsa_share), dense first layers then the DeepSeek-V3 sigmoid group-limited MoE."""
    tc, m = target.config, target.model
    H, nope, rope, vd, kl, ql, hd = _mla_dims(tc)
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    d = tc.hidden_size
    ix0 = next(tl.self_attn.indexer for tl in m.layers if tl.self_attn.indexer is not None)
    u = _shell(tc, d=d, L=tc.num_hidden_layers, H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k,
               attn_norm_feats=ql + kl, dsa_heads=ix0.n_heads, dsa_dim=ix0.head_dim)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    re_ = m.rotary_emb
    glm = tc.model_type == "glm_moe_dsa"
    interleave = True                       # both run the MLA rotary interleaved (V3.2's rope_interleave is gone)
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _mla_from(ul.attn, a, d=d, H=H, nope=nope, rope=rope, vd=vd, kl=kl, ql=ql, inv_freq=re_.inv_freq,
                  att_s=float(re_.attention_scaling), interleave=interleave, scale=a.scaling)
        if a.indexer is not None:
            _dsa_from(ul.attn, a.indexer, d=d, ql=ql, inv_freq=re_.inv_freq, att_s=float(re_.attention_scaling),
                      interleave=glm)
            ul.attn.dsa_share.fill_(1.0)
        else:
            ul.attn.dsa_share.fill_(2.0)
        _ffn_any(ul.ffn, tl.mlp, tc, tc.hidden_act)
    return u

def _trailing_rope_perm(hd, rot):
    """Target RoPE pairs (n + 2i, n + 2i + 1) on the TRAILING `rot` dims (n = hd - rot; DeepSeek-V4's
    [nope | rope] heads, interleaved) -> host pairs (i, i + hd/2); the nope dims fill the rest in order.
    Returns host <- target dim order."""
    h, r, n = hd // 2, rot // 2, hd - rot
    first = [n + 2 * i for i in range(r)] + list(range(0, h - r))
    second = [n + 2 * i + 1 for i in range(r)] + list(range(h - r, n))
    return torch.tensor(first + second)


@torch.no_grad()
def from_deepseek_v4(target):
    """DeepSeek-V4 (DeepseekV4ForCausalLM, transformers 5.x): every layer is shared-KV MQA with K = V
    (kv_proj -> weighted RMSNorm -> the rotary; the host's k- and v-norms with the same weight, v_rope),
    a low-rank q (q_a -> weighted RMSNorm: the attention latent features; q_b -> an unweighted per-head
    RMSNorm), partial interleaved RoPE on the trailing qk_rope_head_dim of each head (permuted onto the host
    pairs; sliding layers the 'main' table, compressed layers the 'compress' one), the output de-rotated at
    the query position (o_derot), learned per-head sinks (the null key), a sliding window over the tokens,
    and on CSA / HCA layers the compressed-KV pools (UAttention.cmp: HCA non-overlapping pools, CSA
    overlapping ones picked per query by the lightning indexer's top-k). The grouped low-rank output
    projection o_b(blockdiag(o_a)) is linear and is folded into o_proj. mHC streams (UHC) with the final
    HyperHead collapse (hc_head_c); a sqrt(softplus) router (score_sp) with selection bias and renorm,
    hash-routed layers (a frozen token -> experts table: hash_tab), clamped SwiGLU on the routed and the
    shared experts (c_clip). MTP layers are not part of the forward."""
    if not hasattr(target, "config") and hasattr(target, "layers") and hasattr(target.layers[0], "hc_attn_fn"):
        return from_deepseek_v41(target)                     # the official V4.1 inference module layout
    tc, m = target.config, target.model
    assert tc.n_shared_experts == 1 and not tc.attention_bias and not tc.mlp_bias
    H, hd, ql, d, rot = tc.num_attention_heads, tc.head_dim, tc.q_lora_rank, tc.hidden_size, tc.qk_rope_head_dim
    rates = [tc.compress_rates[t] for t in tc.layer_types if t != "sliding_attention"]
    ix0 = next((l.self_attn.compressor.indexer for l in m.layers
                if getattr(l.self_attn.compressor, "indexer", None) is not None), None)
    c = UConfig(vocab=tc.vocab_size, d=d, L=tc.num_hidden_layers, heads=H, head_dim=hd, attn_norm_feats=ql,
                experts=tc.n_routed_experts, expert_ff=tc.moe_intermediate_size, top_k=tc.num_experts_per_tok,
                shared_ff=tc.moe_intermediate_size, hc_mult=tc.hc_mult, hc_iters=tc.hc_sinkhorn_iters,
                cmp_pool=max(rates, default=0), cmp_index_heads=ix0.num_heads if ix0 is not None else 0,
                cmp_index_dim=ix0.head_dim if ix0 is not None else 2)
    u = UniversalElephantine(c)
    W = lambda lin: lin.weight.detach().to(F64)
    _cp(u.embed.weight, m.embed_tokens.weight)
    _norm_from(u.norm, m.norm)
    _cp(u.lm_head.weight, target.lm_head.weight)
    hh = m.hc_head
    u.hc_head_c.fill_(1.0)
    _cp(u.hc_fn, hh.hc_fn); _cp(u.hc_base, hh.hc_base); _cp(u.hc_scale, hh.hc_scale)
    u.hc_head_eps.copy_(torch.tensor([hh.eps, hh.input_norm.eps], dtype=F64))
    perm = _trailing_rope_perm(hd, rot)
    ph = lambda w, n: w.reshape(n, hd, *w.shape[1:])[:, perm].reshape(n * hd, *w.shape[1:])   # rows, per head
    rope = m.rotary_emb

    def host_inv(kind, width):
        inv, att = _rope_of(rope, kind)
        assert float(att) == 1.0, "a rotary attention_scaling != 1 is not wired (V4 forces attention_factor 1)"
        out = torch.zeros(width // 2, dtype=F64)
        out[: inv.numel()] = inv.detach().to(F64)
        return out

    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        _hc_from(ul.hc_a, tl.attn_hc); _hc_from(ul.hc_f, tl.ffn_hc)
        _norm_from(ul.norm_a, tl.input_layernorm)
        _norm_from(ul.norm_f, tl.post_attention_layernorm)
        a, at = ul.attn, tl.self_attn
        a.pn.weight.copy_(W(at.q_a_proj))
        a.n_avg.fill_(1.0 / ql)
        _cp(a.n_w, at.q_a_norm.weight); a.n_eps.fill_(float(at.q_a_norm.variance_epsilon))
        a.q_proj.weight[:, d:] = ph(W(at.q_b_proj), H)
        kv = W(at.kv_proj)[perm]                                       # one KV head, broadcast to all
        a.k_proj.weight[:, :d] = kv.repeat(H, 1)
        a.v_proj.weight[:, :d] = kv.repeat(H, 1)
        a.q_coef.copy_(torch.tensor([0.0, 1.0])); a.k_coef.copy_(torch.tensor([0.0, 1.0]))
        a.q_nw.fill_(1.0)
        kn = at.kv_norm.weight.detach().to(F64)[perm]
        a.k_nw.copy_(kn.expand(H, hd)); a.v_nw.copy_(kn.expand(H, hd)); a.v_c.fill_(1.0)
        a.qk_eps.copy_(torch.tensor([at.q_b_norm.eps, at.kv_norm.variance_epsilon], dtype=F64))
        a.v_eps.fill_(float(at.kv_norm.variance_epsilon))
        a.inv_freq.copy_(host_inv(at.rope_layer_type, hd))
        a.scale.fill_(float(at.scaling))
        a.window.fill_(float(at.sliding_window - 1))
        _cp(a.null_logit, at.sinks)
        a.v_rope.fill_(1.0); a.o_derot.fill_(1.0)
        G = tc.o_groups
        oa = W(at.o_a_proj).view(G, -1, H * hd // G)                   # (G, rank, in per group)
        wo = W(at.o_b_proj) @ torch.block_diag(*oa)                    # (d, H hd): the grouped projection, folded
        a.o_proj.weight.copy_(wo.view(d, H, hd)[:, :, perm].reshape(d, H * hd))
        cm = at.compressor
        if cm is not None:
            ac, rate = a.cmp, cm.compress_rate
            csa = getattr(cm, "indexer", None) is not None
            halves = lambda w, n, p: torch.cat([t[p] for t in w.detach().to(F64).view(2, n, *w.shape[1:])]) \
                if w.shape[0] == 2 * n else torch.cat([torch.zeros_like(w.detach().to(F64)), w.detach().to(F64)[p]])
            ac.kv.weight[:, :d] = halves(cm.kv_proj.weight, hd, perm)
            ac.gate.weight[:, :d] = halves(cm.gate_proj.weight, hd, perm)
            ac.pb[:rate] = halves(cm.position_bias.T, hd, perm).T
            _cp(ac.nw, cm.kv_norm.weight.detach()[perm])
            cfg = [rate, 1.0 if csa else 0.0, cm.kv_norm.variance_epsilon, float("inf"), 1e-6, 1.0, 1.0]
            if csa:
                ix = cm.indexer
                dI, nI = ix.head_dim, ix.num_heads
                pi = _trailing_rope_perm(dI, rot)
                ac.ix_kv.weight[:, :d] = halves(ix.kv_proj.weight, dI, pi)
                ac.ix_gate.weight[:, :d] = halves(ix.gate_proj.weight, dI, pi)
                ac.ix_pb[:rate] = halves(ix.position_bias.T, dI, pi).T
                _cp(ac.ix_nw, ix.kv_norm.weight.detach()[pi])
                ac.ix_q.weight[:, d:] = W(ix.q_b_proj).view(nI, dI, ql)[:, pi].reshape(nI * dI, ql)
                ac.ix_w.weight[:, :d] = W(ix.scorer.weights_proj)
                ac.ix_inv.copy_(host_inv(ix.rope_layer_type, dI))
                cfg[3:] = [ix.index_topk, ix.kv_norm.variance_epsilon, ix.scorer.softmax_scale, ix.scorer.weights_scaling]
            ac.cfg.copy_(torch.tensor(cfg, dtype=F64))
        f, mo = ul.ffn, tl.mlp
        f.c_clip.fill_(1.0); f.clip_lim.fill_(float(tc.swiglu_limit))
        sh = mo.shared_experts
        f.s_gate.weight.copy_(W(sh.gate_proj)); f.s_up.weight.copy_(W(sh.up_proj)); f.s_down.weight.copy_(W(sh.down_proj))
        rt = mo.gate
        _cp(f.router.weight, rt.weight)
        sf = tc.scoring_func
        if sf == "sqrtsoftplus":
            f.score_sp.fill_(1.0)
        elif sf == "sigmoid":
            f.score_c.fill_(1.0)
        else:
            assert sf == "softmax", sf
        if hasattr(rt, "e_score_correction_bias"):
            _cp(f.sel_bias, rt.e_score_correction_bias)
        if mo.is_hash:
            f.hash_tab = rt.tid2eid.detach().long().clone()
        f.renorm.fill_(1.0); f.renorm_eps.fill_(1e-20)
        f.route_scale.fill_(float(rt.routed_scaling_factor))
        g, up = mo.experts.gate_up_proj.detach().to(F64).chunk(2, dim=1)
        f.w_gate.copy_(g); f.w_up.copy_(up); _cp(f.w_down, mo.experts.down_proj)
    return u


@torch.no_grad()
def from_deepseek_v41(target, args=None):
    """DeepSeek-V4.1 (deepseek_v41 / deepseek_v41_text; no transformers class: the official inference/model.py
    Transformer, deepseek-ai/DeepSeek-V4.1-Flash, read as text) -- DeepSeek-V4 (from_deepseek_v4) with
      (a) no per-head q norm: q = wq_b(q_norm(wq_a x)) goes straight to the rotary (q_coef [1, 0]);
      (b) lagged hyper-connections: each sub-block's pre (collapse) weights come from the PREVIOUS sub-block's
          mixes (the state one step earlier; layer 0 reads stream 0 alone), and the final collapse is the last FFN's
          pre with no separate head (UHC.lag, hc_lag);
      (c) its compressed KV: ratio > 1 pools of `ratio` tokens with no overlap and no position bias, ratio 1 a
          per-token key norm(wkv_c x), built only by kv_source_layers and read by the later layers of the group
          (cmp.share); index keys k_norm(wk(latent)) from the pooled latent (ix_wk), owned by the kv sources and
          read by the other index sources; layers that are not index sources reuse the last source's picks; the
          candidate pre-filter (top candidate_topk_blocks blocks of candidate_block_size pools, the newest block
          pinned) published by candidate_source_layer and masking the later index sources' scores;
    plus its Engram on the hc streams (UEngramHC). The sigmoid / softmax / sqrt-softplus router with
    gate_temp folded into the router weight; SwiGLU limits on the routed and the shared experts. args: the
    ModelArgs (default target.args). Not carried: the fp8 / fp4 weight and activation quantisation (the host is
    the dequantised function), the vision tower and image-span routing bias, the DSpark draft head."""
    a_ = args if args is not None else target.args
    L0 = target.layers[0]
    H, hd, rd, ql, d = a_.n_heads, a_.head_dim, a_.rope_head_dim, a_.q_lora_rank, a_.dim
    N = a_.hc_mult
    ix0 = next((l.attn.indexer for l in target.layers if l.attn.indexer is not None), None)
    eg0 = next((l.engram for l in target.layers if l.engram is not None), None)
    eh = getattr(target, "engram_hash", None)
    ratios = [a_.compress_ratios[i] for i in range(a_.n_layers)]
    c = UConfig(vocab=a_.vocab_size, d=d, L=a_.n_layers, heads=H, head_dim=hd, attn_norm_feats=ql,
                experts=a_.n_routed_experts, expert_ff=a_.moe_inter_dim, top_k=a_.n_activated_experts,
                shared_ff=a_.moe_inter_dim, hc_mult=N, hc_iters=a_.hc_sinkhorn_iters,
                cmp_pool=max(ratios, default=0), cmp_index_heads=ix0.n_heads if ix0 is not None else 0,
                cmp_index_dim=ix0.index_head_dim if ix0 is not None else 2, cmp_ix_latent=True,
                engram_hc=max(l.engram.embed.weight.shape[0] for l in target.layers if l.engram is not None)
                if eg0 is not None else 0,
                engram_hc_cols=(a_.engram_max_ngram_size - 1) * a_.engram_n_heads if eg0 is not None else 1,
                engram_hc_dim=a_.engram_head_dim if eg0 is not None else 1,
                engram_hc_n=a_.engram_max_ngram_size if eg0 is not None else 2)
    u = UniversalElephantine(c)
    W = lambda lin: lin.weight.detach().to(F64)
    _cp(u.embed.weight, target.embed.weight)
    _norm_from(u.norm, target.norm)
    _cp(u.lm_head.weight, target.head.weight)
    u.hc_lag.fill_(1.0)
    perm = _trailing_rope_perm(hd, rd)
    ph = lambda w, n: w.reshape(n, hd, *w.shape[1:])[:, perm].reshape(n * hd, *w.shape[1:])
    inv_of = lambda fc, width: torch.cat([fc[1].angle().to(F64), torch.zeros(width // 2 - fc.shape[1], dtype=F64)])
    eps = float(a_.norm_eps)
    for i, (ul, tl) in enumerate(zip(u.layers, target.layers)):
        for hc, nm in ((ul.hc_a, "attn"), (ul.hc_f, "ffn")):
            _cp(hc.fn, getattr(tl, f"hc_{nm}_fn")); _cp(hc.base, getattr(tl, f"hc_{nm}_base"))
            _cp(hc.scale, getattr(tl, f"hc_{nm}_scale"))
            hc.eps.copy_(torch.tensor([a_.hc_eps, eps], dtype=F64))
            hc.iters = a_.hc_sinkhorn_iters
            hc.lag.fill_(1.0)
        _norm_from(ul.norm_a, tl.attn_norm)
        _norm_from(ul.norm_f, tl.ffn_norm)
        a, at = ul.attn, tl.attn
        a.pn.weight.copy_(W(at.wq_a))
        a.n_avg.fill_(1.0 / ql)
        _cp(a.n_w, at.q_norm.weight); a.n_eps.fill_(float(at.q_norm.eps))
        a.q_proj.weight[:, d:] = ph(W(at.wq_b), H)
        kv = W(at.wkv)[perm]
        a.k_proj.weight[:, :d] = kv.repeat(H, 1)
        a.v_proj.weight[:, :d] = kv.repeat(H, 1)
        a.q_coef.copy_(torch.tensor([1.0, 0.0])); a.k_coef.copy_(torch.tensor([0.0, 1.0]))   # (a): no q head norm
        kn = at.kv_norm.weight.detach().to(F64)[perm]
        a.k_nw.copy_(kn.expand(H, hd)); a.v_nw.copy_(kn.expand(H, hd)); a.v_c.fill_(1.0)
        a.qk_eps.fill_(float(at.kv_norm.eps)); a.v_eps.fill_(float(at.kv_norm.eps))
        a.inv_freq.copy_(inv_of(at.freqs_cis, hd))
        a.scale.fill_(float(at.softmax_scale))
        a.window.fill_(float(at.window_size - 1))
        _cp(a.null_logit, at.attn_sink)
        a.v_rope.fill_(1.0); a.o_derot.fill_(1.0)
        G = at.n_groups
        oa = W(at.wo_a).view(G, at.o_lora_rank, -1)
        wo = W(at.wo_b) @ torch.block_diag(*oa)
        a.o_proj.weight.copy_(wo.view(d, H, hd)[:, :, perm].reshape(d, H * hd))
        r = at.compress_ratio
        if r:
            ac = a.cmp
            cm, ix = at.compressor, at.indexer
            if cm is not None:                                       # a kv source: its own pools, published
                ac.kv.weight[hd:, :d] = W(cm.wkv)[perm]
                if r > 1:
                    ac.gate.weight[hd:, :d] = W(cm.wgate)[perm]
                _cp(ac.nw, cm.norm.weight.detach()[perm])
            cfg = [r, 0.0, float(cm.norm.eps) if cm is not None else eps, float("inf"), eps, 1.0, 1.0]
            ac.share.copy_(torch.tensor([1.0 if cm is not None else 2.0, 0.0, 2.0, 0.0], dtype=F64))
            if ix is not None:                                       # an index source
                dI, nI = ix.index_head_dim, ix.n_heads
                pi = _trailing_rope_perm(dI, rd)
                if ix.owns_k:
                    ac.ix_wk.weight.copy_(W(ix.wk)[pi][:, perm])
                    _cp(ac.ix_nw, ix.k_norm.weight.detach()[pi])
                    cfg[4] = float(ix.k_norm.eps)
                ac.ix_q.weight[:, d:] = W(ix.wq_b).view(nI, dI, ql)[:, pi].reshape(nI * dI, ql)
                ac.ix_w.weight[:, :d] = W(ix.weights_proj)
                ac.ix_inv.copy_(inv_of(at.freqs_cis, dI))
                cfg[3], cfg[6] = float(ix.index_topk), float(ix.softmax_scale * ix.n_heads ** -0.5)
                cd = 1.0 if ix.is_candidate_source else (2.0 if ix.uses_candidates else 0.0)
                ac.share[1:] = torch.tensor([1.0 if ix.owns_k else 2.0, 1.0, cd], dtype=F64)
                ac.cand.copy_(torch.tensor([float(ix.candidate_topk_blocks), float(max(ix.candidate_block_size, 1))],
                                           dtype=F64))
            ac.cfg.copy_(torch.tensor(cfg, dtype=F64))
        eg = tl.engram
        if eg is not None:
            e = ul.engram_hc
            e.table[: eg.embed.weight.shape[0]] = eg.embed.weight.detach().to(F64)
            _cp(e.wkv.weight, eg.wkv.weight)
            _cp(e.w, eg.q_weight.detach().to(F64) * eg.k_weight.detach().to(F64))
            e.eps.fill_(float(eg.eps))
            j = eg.layer_hash_index
            e.token_map.copy_(eh.token_map); e.mult.copy_(eh.multipliers[j])
            e.primes.copy_(eh.primes[j]); e.offsets.copy_(eh.offsets[j].reshape(-1)); e.pad.fill_(int(eh.pad_id))
        f, mo = ul.ffn, tl.ffn
        g = mo.gate
        _cp(f.router.weight, W(g) / float(g.gate_temp) if hasattr(g, "gate_temp") else W(g))
        sf = g.score_func
        if sf == "sqrtsoftplus":
            f.score_sp.fill_(1.0)
        elif sf == "sigmoid":
            f.score_c.fill_(1.0)
        _cp(f.sel_bias, g.bias)
        f.renorm.fill_(1.0 if (g.norm_topk_prob and g.topk > 1) else 0.0); f.renorm_eps.fill_(1e-20)
        f.route_scale.fill_(float(g.route_scale))
        for e_, ex in enumerate(mo.experts):
            _cp(f.w_gate[e_], ex.w1.weight); _cp(f.w_up[e_], ex.w3.weight); _cp(f.w_down[e_], ex.w2.weight)
        sh = mo.shared_experts
        f.s_gate.weight.copy_(W(sh.w1)); f.s_up.weight.copy_(W(sh.w3)); f.s_down.weight.copy_(W(sh.w2))
        if a_.swiglu_limit > 0:
            f.c_clip.fill_(1.0); f.clip_lim.fill_(float(a_.swiglu_limit))
    return u


@torch.no_grad()
def from_recurrent_gemma(target):
    """RecurrentGemma / Griffin (RecurrentGemmaForCausalLM): recurrent layers (the RG-LRU SSM sub-block,
    rglru_block) and local MQA attention layers (partial rotary on the first half of each head; the
    prefill forward masks causally, the window only bounds the decoding cache), each followed by a biased
    GeGLU-tanh MLP, (1 + w) RMSNorms, the bf16-rounded sqrt(d) embedding normalizer (folded), the final
    logit softcap."""
    tc, m = target.config, target.model
    rb0 = next(tl.temporal_block for tl in m.layers if hasattr(tl.temporal_block, "rg_lru"))
    H, hd = tc.num_attention_heads, tc.head_dim
    lru = rb0.lru_width
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=tc.intermediate_size // 2,
               ssm_in=2 * lru, conv_k=rb0.conv1d_width, ssm_heads=lru, dk=1, dv=1)
    _head_from(u, m.embed_tokens, m.final_norm, target.lm_head, "rms1p", emb_scale=m.normalizer.detach().to(F64))
    if getattr(tc, "logits_soft_cap", None):
        u.final_cap.fill_(float(tc.logits_soft_cap))
    for ul, tl in zip(u.layers, m.layers):
        tb = tl.temporal_block
        if hasattr(tb, "rg_lru"):
            ul.ssm.load_state_dict(rglru_block(tb).state_dict())
            _norm_any(ul.norm_s, tl.temporal_pre_norm, "rms1p")
        else:
            _norm_any(ul.norm_a, tl.temporal_pre_norm, "rms1p")
            _attn_any(ul.attn, *_qkv(tb), n_kv=tb.num_key_value_heads, scale=hd ** -0.5,
                      inv_freq=tb.rotary_emb.inv_freq)
        _norm_any(ul.norm_f, tl.channel_pre_norm, "rms1p")
        _mlp_any(ul.ffn, tl.mlp_block, tc.hidden_activation)
    return u


@torch.no_grad()
def from_minimax(target):
    """MiniMax-Text-01 / M1 (MiniMaxForCausalLM): lightning-attention layers (the SSM sub-block,
    minimax_block) and softmax GQA layers, each REPLACING the stream by its norm and scaling the residual:
    h = n1(x); h = alpha_a h + beta_a attn(h); h = n2(h); x' = alpha_m h + beta_m moe(h). Each layer runs as
    two host layers whose stream norm (sn_in) is n1 / n2 and whose residual coefficient (res_c) is alpha,
    beta folded into the output projections; the Mixtral-style softmax MoE renormalises its top-k."""
    tc, m = target.config, target.model
    la0 = next((tl.self_attn for tl in m.layers if tl.layer_type == "linear_attention"), None)
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    H = tc.num_attention_heads
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // H
    ssm = dict(ssm_in=4 * la0.num_attention_heads * la0.head_dim, conv_k=1, ssm_heads=la0.num_attention_heads,
               dk=la0.head_dim, dv=la0.head_dim) if la0 is not None else {}
    u = _shell(tc, d=tc.hidden_size, L=2 * tc.num_hidden_layers, H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k,
               stream_norms=True, **ssm)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    rope = m.rotary_emb
    for i, tl in enumerate(m.layers):
        ua, uf = u.layers[2 * i], u.layers[2 * i + 1]
        _norm_any(ua.sn_in, tl.input_layernorm); ua.sn_c[0] = 1
        _norm_any(uf.sn_in, tl.post_attention_layernorm); uf.sn_c[0] = 1
        ba, bm = float(tl.attn_beta_factor), float(tl.mlp_beta_factor)
        ua.res_c[0] = float(tl.attn_alpha_factor)
        uf.res_c[1] = float(tl.mlp_alpha_factor)
        if tl.layer_type == "linear_attention":
            _norm_any(ua.norm_s, None, "none")
            ua.ssm.load_state_dict(minimax_block(tl.self_attn).state_dict())
            ua.ssm.out_proj.weight.mul_(ba); ua.ssm.out_proj.bias.mul_(ba)
        else:
            a = tl.self_attn
            _norm_any(ua.norm_a, None, "none")
            wq, wk, wv, wo, bq, bk, bv, bo = _qkv(a)
            _attn_any(ua.attn, wq, wk, wv, wo * ba, bq, bk, bv, None if bo is None else bo * ba,
                      n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                      att_scale=rope.attention_scaling, window=getattr(tc, "sliding_window", None))
        _norm_any(uf.norm_f, None, "none")
        _ffn_any(uf.ffn, tl.block_sparse_moe, tc, tc.hidden_act)
        uf.ffn.w_down.mul_(bm); uf.ffn.b_down.mul_(bm)
        uf.ffn.s_down.weight.mul_(bm); uf.ffn.s_down.bias.mul_(bm)
    return u


@torch.no_grad()
def from_kimi_linear(target):
    """Kimi-Linear (KimiLinearForCausalLM, transformers 5): KDA layers (the SSM sub-block, _glm_kda: silu
    short conv on q / k / v, l2 q / k, per-channel softplus decay, sigmoid beta, per-head RMSNorm * sigmoid
    low-rank gate) and NoPE MLA layers (_mla_from with no rotation), dense first layers then the
    DeepSeek-V3-style sigmoid group-limited MoE with shared experts."""
    tc, m = target.config, target.model
    la0 = next(tl.self_attn for tl in m.layers if tl.block_type == "linear_attention")
    H, nope, rope, vd, kl, ql, hd = _mla_dims(tc)
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    d, n, hk = tc.hidden_size, la0.qkv_dim, la0.head_dim
    u = _shell(tc, d=d, L=tc.num_hidden_layers, H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k,
               attn_norm_feats=ql + kl, ssm_in=3 * n + 2 * hk + la0.num_heads, conv_k=la0.conv_kernel_size,
               ssm_heads=la0.num_heads, dk=hk, dv=hk)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        if tl.block_type == "linear_attention":
            _norm_any(ul.norm_s, tl.input_layernorm)
            _glm_kda(ul.ssm, a)
        else:
            _norm_any(ul.norm_a, tl.input_layernorm)
            _mla_from(ul.attn, a, d=d, H=H, nope=nope, rope=rope, vd=vd, kl=kl, ql=ql,
                      inv_freq=torch.zeros(rope // 2, dtype=F64), att_s=1.0, interleave=False, scale=a.scaling)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _ffn_any(ul.ffn, tl.mlp, tc, tc.hidden_act)
    return u


@torch.no_grad()
def _sala_lightning(s: USSM, la, H, hd, n_kv):
    """MiniCPM-SALA LightningAttention (modeling_minicpm_sala.py, remote code read): q / k / v projections (no
    activation, no conv), per-head RMSNorm on q and k (qk_norm) THEN the rotary (lightning_use_rope; rotate-half
    pairs), GQA k / v repeated, simple GLA with a constant per-head log-decay -slope_h (ALiBi slopes over the
    num_attention_heads heads): S_t = exp(-slope_h) S_{t-1} + k_t v_t^T, o_t = scale S_t q_t; then an RMSNorm over
    the whole H * hd width (use_output_norm), a sigmoid gate sigmoid(z_proj x) (use_output_gate), o_proj.
    The per-head RMSNorm rms(q) w is the l2 branch: q / sqrt(|q|^2 / hd + eps) = sqrt(hd) l2(q) with l2_eps = hd eps,
    the weight (and sqrt(hd), and the scale on q) the l2 gains q_nw. The rotary is the constant angle step rot_k
    = inv_freq (the cumulative angle starts one step on, a common offset that cancels in q . k)."""
    n = H * hd
    tile = lambda w: None if w is None else _tile_kv(w.detach().to(F64), n_kv, H, hd)
    W = lambda lin: lin.weight.detach().to(F64)
    gate = getattr(la, "z_proj", None)
    rows = [W(la.q_proj), tile(W(la.k_proj)), tile(W(la.v_proj)), W(gate) if gate is not None else torch.zeros(n, s.in_proj.in_features, dtype=F64)]
    s.in_proj.weight.copy_(torch.cat(rows, 0))
    bias = lambda lin: lin.bias.detach().to(F64) if getattr(lin, "bias", None) is not None else torch.zeros(lin.out_features, dtype=F64)
    if la.q_proj.bias is not None:
        s.in_proj.bias.copy_(torch.cat([bias(la.q_proj), tile(bias(la.k_proj)), tile(bias(la.v_proj)),
                                        bias(gate) if gate is not None else torch.zeros(n, dtype=F64)]))
    eye = torch.eye(n, dtype=F64)
    s.pq.weight[:, 0:n] = eye                                         # read the pre-conv stream u directly
    s.pk.weight[:, n:2 * n] = eye
    s.pv.weight[:, 2 * n:3 * n] = eye
    sc = float(la.scale)
    if la.qk_norm:
        eps_q, eps_k = float(la.q_norm.variance_epsilon), float(la.k_norm.variance_epsilon)
        assert eps_q == eps_k, "one l2 eps for q and k"
        s.l2_eps.fill_(hd * eps_q)
        s.q_coef.copy_(torch.tensor([0.0, 1.0])); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
        s.q_nw.copy_((W(la.q_norm) * math.sqrt(hd) * sc).expand(H, hd))
        s.k_nw.copy_((W(la.k_norm) * math.sqrt(hd)).expand(H, hd))
    else:
        s.q_coef.copy_(torch.tensor([sc, 0.0], dtype=F64))
    if la.use_rope:
        inv = la.rotary_emb.inv_freq.detach().to(F64)
        assert inv.numel() == hd // 2
        s.rot_k.copy_(inv.expand(H, hd // 2))
    slopes = _sala_slopes(H)
    s.A3.copy_(slopes.view(H, 1).expand(H, hd))                       # log a = -slope_h
    s.cw[4] = 1                                                       # write 1
    if gate is not None:
        s.pz.weight[:, 3 * n:4 * n] = eye
        s.gate[1] = 1                                                 # sigmoid(z_proj x)
    else:
        s.gate[4] = 1
    if la.use_output_norm:
        s.avg.fill_(1 / n)
        s.norm_w.copy_(W(la.o_norm))
        s.norm_eps.fill_(float(la.o_norm.variance_epsilon))
        s.out_coef[0] = 1                                             # norm(y) * gate
    else:
        s.out_coef[2] = 1                                             # y * gate
    _cp(s.out_proj.weight, la.o_proj.weight)
    if la.o_proj.bias is not None:
        _cp(s.out_proj.bias, la.o_proj.bias)


def _sala_slopes(n):
    """modeling_minicpm_sala._build_slope_tensor: the ALiBi slopes (the lightning decay rates)."""
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]

    def get(n):
        if math.log2(n).is_integer():
            return pow2(n)
        c = 2 ** math.floor(math.log2(n))
        return pow2(c) + get(2 * c)[0::2][: n - c]
    return torch.tensor(get(n), dtype=torch.float32).to(F64)          # the code builds them in fp32


@torch.no_grad()
def from_minicpm_sala(target):
    """MiniCPM-SALA (MiniCPMSALAForCausalLM, openbmb remote code modeling_minicpm_sala.py, read not run):
    a MiniCPM-4 body (muP: embeddings x scale_emb, each sub-block output x scale_depth / sqrt(L), logits of
    h / (hidden / dim_model_base)) whose mixer_types pick per layer either
      minicpm4        softmax GQA (head dim hidden / heads), RoPE iff attn_use_rope, a sigmoid output gate
                      o_gate (attn_use_output_gate) -- the host attention; or
      lightning-attn  _sala_lightning (simple GLA) -- the SSM sub-block,
    then a SwiGLU MLP. The minicpm4 layers run InfLLM-v2 sparse attention (sparse_config) only on CUDA AND at
    kv length >= sparse_config.dense_len (8,192); below that, and always off CUDA, they are dense causal
    attention, which is what the host carries. The sparse regime (top-k 64-token blocks scored by
    mean-pooled compressed keys at two strides through the infllm_v2 CUDA kernels, + init / local window
    blocks) is NOT carried."""
    tc, m = target.config, target.model
    assert not getattr(tc, "rope_scaling", None), "MiniCPM-SALA rope_scaling: only the plain rotary is mapped"
    d, L, H = tc.hidden_size, tc.num_hidden_layers, tc.num_attention_heads
    hd = d // H                                                        # MiniCPMAttention's head dim
    hl, nkv_l = tc.head_dim, tc.lightning_nkv                          # LightningAttention's
    ff = tc.intermediate_size
    u = _shell(tc, d=d, L=L, H=H, hd=hd, ff=ff, ssm_in=4 * H * hl, conv_k=1, ssm_heads=H, dk=hl, dv=hl)
    tp = getattr(tc, "pretraining_tp", 1) or 1                         # tp > 1 skips the logit division in code
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, emb_scale=float(tc.scale_emb),
               logit_scale=1.0 if tp > 1 else float(tc.dim_model_base) / d)
    r = float(tc.scale_depth) / math.sqrt(L)
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        mt = tc.mixer_types[i]
        a = tl.self_attn
        if mt == "minicpm4":
            _norm_any(ul.norm_a, tl.input_layernorm)
            inv = a.rotary_emb.inv_freq if tc.attn_use_rope else None
            wq, wk, wv, wo, bq, bk, bv, bo = _qkv(a)
            _attn_any(ul.attn, wq, wk, wv, wo * r, bq, bk, bv, None if bo is None else bo * r,
                      n_kv=tc.num_key_value_heads, scale=hd ** -0.5, inv_freq=inv)
            if getattr(a, "use_output_gate", False):
                _cp(ul.attn.g_proj.weight, a.o_gate.weight)
                if a.o_gate.bias is not None:
                    _cp(ul.attn.g_proj.bias, a.o_gate.bias)
                ul.attn.gate_coef.copy_(torch.tensor([0.0, 1.0]))
        elif mt in ("lightning", "lightning_attn", "lightning-attn"):
            _norm_any(ul.norm_s, tl.input_layernorm)
            _sala_lightning(ul.ssm, a, H, hl, nkv_l)
            ul.ssm.out_proj.weight.mul_(r); ul.ssm.out_proj.bias.mul_(r)
        else:
            raise ValueError(f"MiniCPM-SALA mixer type {mt!r}")
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _ffn_any(ul.ffn, tl.mlp, tc, tc.hidden_act)
        ul.ffn.s_down.weight.mul_(r); ul.ffn.s_down.bias.mul_(r)
        ul.ffn.w_down.mul_(r); ul.ffn.b_down.mul_(r)
    return u


@torch.no_grad()
def from_olmo_hybrid(target):
    """OLMo-Hybrid (OlmoHybridForCausalLM, transformers 5): gated-DeltaNet layers (the SSM sub-block,
    _gdn_from on its separate q / k / v / a / b / g projections; allow_neg_eigval doubles beta: the
    delta and write coefficients) under pre-norms, and OLMo-2-style attention layers (no pre-norm,
    post-norms, whole-projection qk-norm, RoPE or none) in between."""
    from types import SimpleNamespace as NS
    tc, m = target.config, target.model
    la0 = next(tl.linear_attn for tl in m.layers if hasattr(tl, "linear_attn"))
    H = tc.num_attention_heads
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // H
    kd, vd = la0.key_dim, la0.value_dim
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=tc.intermediate_size,
               ssm_in=2 * kd + 2 * vd + 2 * la0.num_v_heads, conv_k=la0.conv_kernel_size, ssm_heads=la0.num_v_heads,
               dk=la0.head_k_dim, dv=la0.head_v_dim)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    rope = getattr(m, "rotary_emb", None)
    for ul, tl in zip(u.layers, m.layers):
        if hasattr(tl, "linear_attn"):
            g = tl.linear_attn
            sh = NS(num_k_heads=g.num_k_heads, num_v_heads=g.num_v_heads, head_k_dim=g.head_k_dim, head_v_dim=g.head_v_dim,
                    in_proj_qkv=NS(weight=torch.cat([_W(g.q_proj), _W(g.k_proj), _W(g.v_proj)])), in_proj_z=g.g_proj,
                    in_proj_b=g.b_proj, in_proj_a=g.a_proj, conv1d=g.conv1d, dt_bias=g.dt_bias, A_log=g.A_log.reshape(-1),
                    norm=g.o_norm, out_proj=g.o_proj)
            _gdn_from(ul.ssm, sh)
            if g.allow_neg_eigval:
                ul.ssm.cg.mul_(2.0); ul.ssm.cw[1].mul_(2.0)
            _norm_any(ul.norm_s, tl.input_layernorm)
            _norm_any(ul.norm_f, tl.post_attention_layernorm)
        else:
            a = tl.self_attn
            _norm_any(ul.norm_a, None, "none")
            _norm_any(ul.norm_f, None, "none")
            _norm_any(ul.post_a, tl.post_attention_layernorm)
            _norm_any(ul.post_f, tl.post_feedforward_layernorm)
            ul.post_coef[0] = torch.tensor([0.0, 1.0])
            ul.post_coef[2] = torch.tensor([0.0, 1.0])
            _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling,
                      inv_freq=None if rope is None else rope.inv_freq,
                      att_scale=1.0 if rope is None else rope.attention_scaling,
                      q_norm=a.q_norm, k_norm=a.k_norm)
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


@torch.no_grad()
def from_hyperclovax(target):
    """HyperCLOVA X (HyperCLOVAXForCausalLM, transformers 5): a Llama body with muP multipliers (embedding,
    attention scale, residual, logits) and optional post-norms on each sub-block output (use_post_norm:
    the sandwich branch, the residual multiplier folded into their weights; without them into the
    output projections)."""
    tc, m = target.config, target.model
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, emb_scale=float(tc.embedding_multiplier),
               logit_scale=float(getattr(tc, "logits_scaling", 1.0) or 1.0))    # (multiplies, unlike Granite)
    rope = m.rotary_emb
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=a.scaling, inv_freq=rope.inv_freq,
                  att_scale=rope.attention_scaling)
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
        r = float(tl.residual_multiplier)
        if isinstance(tl.post_norm1, nn.Identity):
            _fold_residual(ul, r)
        else:
            _norm_any(ul.post_a, tl.post_norm1); _norm_any(ul.post_f, tl.post_norm2)
            ul.post_a.weight.mul_(r); ul.post_f.weight.mul_(r)
            ul.post_coef[0] = torch.tensor([0.0, 1.0])
            ul.post_coef[2] = torch.tensor([0.0, 1.0])
    return u


@torch.no_grad()
def from_qwen_vl(target):
    """The language model of Qwen2.5-VL / Qwen3-VL / Qwen3-VL-MoE (…ForConditionalGeneration) on text: its
    decoder is Qwen2 / Qwen3 / Qwen3-MoE-shaped (from_llama), multimodal RoPE (mrope sections) reduces to
    the plain rotary when the three position streams coincide (text tokens); image / video tokens and
    Qwen3-VL's deepstack visual injections are not carried."""
    from types import SimpleNamespace as NS
    lm = target.model.language_model
    u = from_llama(NS(config=lm.config, model=lm, lm_head=target.lm_head))
    return u


@torch.no_grad()
def from_phimoe(target):
    """Phi-3.5-MoE (PhimoeForCausalLM): biased LayerNorms, biased GQA attention, LongRoPE, a Mixtral
    expert bank routed by SparseMixer (UFFN.smix_eps: top-2 picks, each weighted by a softmax over the
    logits within 2 router_jitter_noise of it, relative), an lm_head bias."""
    tc, m = target.config, target.model
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=tc.num_attention_heads, hd=hd, ff=sff,
               experts=E, expert_ff=ef, top_k=k)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head, "ln")
    rope = m.rotary_emb                     # it builds inv_freq per call: the short-context (T <= original) setting
    inv, att = rope.rope_init_fn(rope.config, None, 1)
    if tc.rope_scaling:
        att = rope.short_mscale if rope.short_mscale is not None else att
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm, "ln")
        _norm_any(ul.norm_f, tl.post_attention_layernorm, "ln")
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=getattr(a, "scaling", hd ** -0.5), inv_freq=inv,
                  att_scale=att, window=_sliding(tc, i))
        f, mo = ul.ffn, tl.block_sparse_moe
        _cp(f.router.weight, mo.gate.weight)
        for e, (g, up, dn) in enumerate(_expert_ws(mo)):
            _cp(f.w_gate[e], g); _cp(f.w_up[e], up); _cp(f.w_down[e], dn)
        f.smix_eps.fill_(float(mo.router_jitter_noise))
        _set_act(f, tc.hidden_act, gated=True)
    return u


def _pad_heads(w, n, h, hd):
    """(n * h, ...) rows (n heads of width h) -> (n * hd, ...), each head zero-padded to hd."""
    w = w.view(n, h, *w.shape[1:])
    return torch.cat([w, w.new_zeros(n, hd - h, *w.shape[2:])], 1).reshape(n * hd, *w.shape[2:])


@torch.no_grad()
def from_mimo_v2_flash(target):
    """MiMo-V2-Flash (MiMoV2FlashForCausalLM, transformers 5): sliding / full layers with their own rotary
    (partial: ~1/3 of each head), the sliding layers with twice the KV heads and a learned per-head sink
    (the null-key branch), a value head width v_head_dim != the q / k head (both zero-padded onto the
    host head) and a value scale (folded into v), dense first layers then the DeepSeek-V3-style sigmoid
    group-limited MoE."""
    tc, m = target.config, target.model
    E, ef, k, sff = _ffn_dims(m.layers, tc)
    H, hq, hv = tc.num_attention_heads, tc.head_dim, tc.v_head_dim
    hd = max(hq, hv)
    hd += hd % 2
    d = tc.hidden_size
    u = _shell(tc, d=d, L=tc.num_hidden_layers, H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    for i, (ul, tl) in enumerate(zip(u.layers, m.layers)):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        n_kv = a.k_proj.out_features // hq
        wq, wk, wv, wo, bq, bk, bv, bo = _qkv(a)
        pad = lambda w, n, h: None if w is None else _pad_heads(w, n, h, hd)
        wo = torch.cat([wo.view(d, H, hv), wo.new_zeros(d, H, hd - hv)], 2).reshape(d, H * hd)
        vs = float(a.v_scale)
        inv, att = _rope_of(m.rotary_emb, tc.layer_types[i])
        _attn_any(ul.attn, pad(wq, H, hq), pad(wk, n_kv, hq), pad(wv * vs, n_kv, hv), wo, pad(bq, H, hq), pad(bk, n_kv, hq),
                  None if bv is None else pad(bv * vs, n_kv, hv), bo, n_kv=n_kv, scale=a.scaling, inv_freq=inv,
                  att_scale=att, window=a.sliding_window)
        if a.sinks is not None:
            _cp(ul.attn.null_logit, a.sinks)
        _ffn_any(ul.ffn, tl.mlp, tc, tc.hidden_act)
    return u


@torch.no_grad()
def from_rwkv(target):
    """RWKV-4 (stock RwkvForCausalLM): the embedding LayerNorm (block 0's pre_ln: the host's emb_norm), per block
    LayerNorm -> time mix (the SSM sub-block: rwkv4_block, WKV as a ratio of two linear recurrences) and LayerNorm ->
    channel mix W_v relu(W_k x_mk)^2 * sigmoid(W_r x_mr) (UFFN's shift_c branch: token-shifted mixes, the ungated
    relu^2 shared path, the receptance gate), the final LayerNorm and the untied head. rescale_every > 0: the stock
    eval forward divides the output / value weights by 2^(i // R) and halves the stream every R blocks; the host
    reads the unrescaled weights (undoing a rescale already applied) and runs the trained function, which the
    halved stream matches up to the LayerNorms' eps."""
    tc, m = target.config, target.rwkv
    h = tc.hidden_size
    assert (tc.attention_hidden_size or h) == h, "attention_hidden_size != hidden_size not mapped"
    R = int(getattr(tc, "rescale_every", 0) or 0)
    rescaled = bool(getattr(m, "layers_are_rescaled", False)) and R > 0     # an eval forward divided the weights
    ff = m.blocks[0].feed_forward.key.out_features
    u = _shell(tc, d=h, L=tc.num_hidden_layers, H=1, hd=2, ff=ff, ssm_in=6 * h, conv_k=2, ssm_heads=4 * h, dk=1, dv=1,
               ffn_shift=True)
    _head_from(u, m.embeddings, m.ln_out, target.head, "ln")
    _norm_any(u.emb_norm, m.blocks[0].pre_ln, "ln")
    u.emb_norm_c.fill_(1.0)
    for ul, tl in zip(u.layers, m.blocks):
        _norm_any(ul.norm_s, tl.ln1, "ln")
        ul.ssm.load_state_dict(rwkv4_block(tl.attention).state_dict())
        _norm_any(ul.norm_f, tl.ln2, "ln")
        fw, f = tl.feed_forward, ul.ffn
        _cp(f.s_up.weight, fw.key.weight)
        _cp(f.s_down.weight, fw.value.weight)
        f.c_relu2.fill_(1.0)
        f.shift_c.fill_(1.0)
        _cp(f.mix_k, fw.time_mix_key.view(-1)); _cp(f.mix_r, fw.time_mix_receptance.view(-1))
        _cp(f.r_gate.weight, fw.receptance.weight)
        if rescaled:                                     # undo _rescale_layers: output / value weights / 2^(i // R)
            k = 2.0 ** (ul.idx // R)
            ul.ssm.out_proj.weight.mul_(k); f.s_down.weight.mul_(k)
    return u


# --- host gaps (M11): block-causal masks, Ling hybrids, MiniCPM-SALA, whole-model RWKV ----------------

@torch.no_grad()
def from_sdar(target, block_size=None):
    """SDAR / SDAR-MoE (JetLM, remote code modeling_sdar*.py, read not run): a Qwen3 (-MoE) body -- per-head
    q / k RMSNorm before the rotary, GQA, SwiGLU or a softmax top-k MoE with norm_topk_prob -- whose attention
    is NON-causal under a caller-supplied mask: SDARAttention.forward (modeling_sdar.py, JetLM/SDAR-8B-Chat-b32
    @76fd433, l. 275-297; modeling_sdar_moe.py, JetLM/SDAR-30B-A3B-Chat @f5add2a, l. 218-237) runs
    F.scaled_dot_product_attention(attn_mask=attention_mask, is_causal=False), and SDARModel.forward (l. 517-519)
    has the causal-mask builder commented out. The block size is NOT in the model: neither modeling_sdar.py,
    configuration_sdar.py, configuration_sdar_moe.py nor any released config.json has a block size or a default
    for it (the generation code that builds the block mask is not in the repos), so the caller must give it:
    block_size, else config.block_size if a config carries one; otherwise this raises. The host then runs the
    block-causal mask (key s visible to query t iff floor(s / b) <= floor(t / b)) from its block_size buffer;
    a forward's `mask=` runs any other caller mask, block_size 1 is the causal AR mask and causal=False the
    bidirectional one."""
    b = block_size or getattr(target.config, "block_size", None)
    if not b:
        raise ValueError("SDAR's block size is a generation-time input (its code and configs set none): pass "
                         "from_sdar(target, block_size=...)")
    u = from_llama(target)
    u.block_size.fill_(float(b))
    return u


def _bl_attn(tl):
    return getattr(tl, "attention", None) or tl.self_attn


def _bl_kind(a):
    """A Ling (bailing) mixer's kind: MLA softmax, KDA, or lightning (simple GLA)."""
    if hasattr(a, "kv_b_proj"):
        return "mla"
    if hasattr(a, "query_key_value"):
        return "lightning"
    return "kda"


@torch.no_grad()
def _bailing_kda(s: USSM, a):
    """A Ling-3 KDA layer (remote BailingMoeV3KimiDeltaAttention: q / k / v each with its own silu short conv,
    a full-rank f_proj / g_proj or the low-rank f_a f_b / g_a g_b pair; stock BailingHybridDeltaAttention: one
    fused conv1d, forget_gate.f_proj) -> the SSM block: l2 q / k (q * dk^-1/2), per-channel decay
    log a = lower * sigmoid(exp(A_log) (f + dt_bias)) (safe gate) or -exp(A_log) softplus(f + dt_bias),
    beta = sigmoid(b x) as both the delta strength and the write, per-head RMSNorm * sigmoid(g)."""
    H, hd = a.num_heads, a.head_dim
    n = H * hd
    fg = getattr(a, "forget_gate", None) or a
    if hasattr(fg, "f_proj"):
        fw, fb = _W(fg.f_proj), torch.eye(n, dtype=F64)
    else:
        fw, fb = _W(a.f_a_proj), _W(a.f_b_proj)
    if hasattr(a, "g_proj"):
        gw, gb = _W(a.g_proj), torch.eye(n, dtype=F64)
    else:
        gw, gb = _W(a.g_a_proj), _W(a.g_b_proj)
    nf, ng = fw.shape[0], gw.shape[0]
    s.in_proj.weight[:3 * n + nf + H + ng] = torch.cat([_W(a.q_proj), _W(a.k_proj), _W(a.v_proj), fw, _W(a.b_proj), gw])
    if hasattr(a, "conv1d"):
        cw = a.conv1d.weight.detach().to(F64)[:, 0]
    else:
        cw = torch.cat([c.weight.detach().to(F64)[:, 0] for c in (a.q_conv1d, a.k_conv1d, a.v_conv1d)])
    s.conv_w[:3 * n, : cw.shape[1]] = cw
    s.act_silu[:3 * n] = 1
    D, eye = s.D, torch.eye(n, dtype=F64)
    s.pq.weight[:, D: D + n] = eye; s.pk.weight[:, D + n: D + 2 * n] = eye; s.pv.weight[:, D + 2 * n: D + 3 * n] = eye
    s.q_coef.copy_(torch.tensor([0.0, hd ** -0.5], dtype=F64)); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
    f0, b0, g0 = 3 * n, 3 * n + nf, 3 * n + nf + H
    rate = fg.A_log.detach().to(F64).exp().repeat_interleave(hd)
    dtb = fg.dt_bias.detach().to(F64).reshape(-1)
    safe = getattr(fg, "safe_gate", True)
    lower = getattr(fg, "lower_bound", None)
    if safe and lower is not None:
        s.p2.weight[:, f0: f0 + nf] = rate[:, None] * fb; s.p2.bias.copy_(rate * dtb)
        s.A2.fill_(-float(lower))
    else:
        s.p1.weight[:, f0: f0 + nf] = fb; s.p1.bias.copy_(dtb)
        s.A1.copy_(rate.view(H, hd))
    s.pb.weight[:, b0: b0 + H] = torch.eye(H, dtype=F64)
    s.cg.fill_(1); s.cw[1] = 1
    s.pz.weight[:, g0: g0 + ng] = gb
    s.gate[1] = 1
    s.avg.copy_(torch.block_diag(*[torch.full((hd, hd), 1 / hd, dtype=F64)] * H))
    on = a.o_norm
    s.norm_w.copy_(on.weight.detach().to(F64).repeat(H))
    s.norm_eps.fill_(float(getattr(on, "variance_epsilon", getattr(on, "eps", 1e-6))))
    s.out_coef[0] = 1
    _cp(s.out_proj.weight, a.o_proj.weight)


@torch.no_grad()
def _bailing_lightning(s: USSM, a, inv_freq):
    """A Ling-2.5 / 2.6 linear-attention layer (remote BailingMoeV2_5LinearAttention: Lightning Attention-2 via fla's
    simple GLA) -> the SSM block: fused q / k / v, per-head RMSNorm of q and k (the normalized knob features),
    partial RoPE on the first rope_dim channels of each head (rotate-half layout; the head's channels permuted so
    the rotated pairs are (j, j + hd / 2): the constant per-pair angle step rot_k), the per-head constant decay
    exp(slope) (slope = -ALiBi-style slope x the layer factor), unit write, q * hd^-1/2; output GroupRMSNorm over
    group_norm_size groups of the full width, times sigmoid(g_proj x)."""
    H, hd = a.num_heads, a.head_dim
    assert a.num_key_value_heads == H
    n = H * hd
    r = inv_freq.numel() * 2                                           # rope_dim
    h1 = (hd - r) // 2
    perm = list(range(r // 2)) + [r + i for i in range(h1)] + [r // 2 + i for i in range(r // 2)] \
        + [r + h1 + i for i in range(hd - r - h1)]
    P = torch.tensor(perm)                                             # host channel j <- target channel P[j]
    hp = (torch.arange(H)[:, None] * hd + P[None]).reshape(-1)         # the same per head
    qkv, bqkv = _W(a.query_key_value), _B(a.query_key_value)
    s.in_proj.weight[:4 * n] = torch.cat([qkv, _W(a.g_proj)])
    if bqkv is not None:
        s.in_proj.bias[:3 * n] = bqkv
    D = s.D
    eye = torch.eye(n, dtype=F64)
    s.pn.weight.zero_()
    s.pn.weight[:n, :n] = eye[hp]                                      # q (permuted), read off the pre-conv u
    s.pn.weight[n: 2 * n, n: 2 * n] = eye[hp]                          # k (permuted)
    blk = torch.full((hd, hd), 1 / hd, dtype=F64)
    s.n_avg.zero_()
    s.n_avg[:2 * n, :2 * n] = torch.block_diag(*[blk] * (2 * H))
    eps = {float(a.query_layernorm.variance_epsilon), float(a.key_layernorm.variance_epsilon)}
    assert len(eps) == 1
    s.n_eps.fill_(eps.pop())
    s.n_w[:2 * n] = torch.cat([a.query_layernorm.weight.detach().to(F64)[P].repeat(H),
                               a.key_layernorm.weight.detach().to(F64)[P].repeat(H)])
    R0 = 2 * D
    s.pq.weight[:, R0: R0 + n] = eye; s.pk.weight[:, R0 + n: R0 + 2 * n] = eye
    s.pv.weight[:, 2 * n: 3 * n] = eye                                 # v from u (no conv)
    s.q_coef.copy_(torch.tensor([hd ** -0.5, 0.0], dtype=F64)); s.k_coef.copy_(torch.tensor([1.0, 0.0]))
    s.rot_k.zero_()
    s.rot_k[:, : r // 2] = inv_freq.detach().to(F64)[None]
    s.A3.copy_((-a.slope.detach().to(F64)).view(H, 1).expand(H, hd))
    s.cw[4] = 1
    s.pz.weight[:, 3 * n: 4 * n] = eye
    s.gate[1] = 1
    G = a.g_norm.group_norm_size
    s.avg.copy_(torch.block_diag(*[torch.full((n // G, n // G), G / n, dtype=F64)] * G))
    s.norm_w.copy_(a.g_norm.weight.detach().to(F64))
    s.norm_eps.fill_(float(a.g_norm.variance_epsilon))
    s.out_coef[0] = 1
    _cp(s.out_proj.weight, a.dense.weight)
    if _B(a.dense) is not None:
        _cp(s.out_proj.bias, a.dense.bias)


@torch.no_grad()
def from_bailing_hybrid(target):
    """Ling / Ring 2.5 - 3.x (model_type bailing_hybrid): MLA softmax layers every layer_group_size-th layer (and the
    trailing ones), linear layers elsewhere, dense first layers then a sigmoid group-limited MoE with a shared expert.
      Ling-3.x (BailingMoeV3, and the stock transformers BailingHybridForCausalLM): KDA linear layers (_bailing_kda),
        MLA with (or without) a q LoRA and a head-wise sigmoid output gate sigmoid(g_proj x)_h (each head's g_proj row
        repeated over the host head), interleaved RoPE on the rope part; Ling-3.1's per-layer SwiGLU limits on the
        routed and shared experts clamp AFTER the activation (UFFN.clip_post).
      Ling-2.5 / 2.6 (BailingMoeV2_5): Lightning (simple GLA) linear layers (_bailing_lightning), plain MLA.
    The router: sigmoid scores, selection on scores + expert_bias within the top topk_group groups (group score =
    sum of the top 2, the other experts masked to -inf), renormalised by (sum + 1e-20) when top_k > 1, times
    routed_scaling_factor. The MTP layers are not part of the forward."""
    tc, m = target.config, target.model
    layers = list(m.layers)[: tc.num_hidden_layers]
    kinds = [_bl_kind(_bl_attn(tl)) for tl in layers]
    mla = next(_bl_attn(tl) for tl, k in zip(layers, kinds) if k == "mla")
    lin = [_bl_attn(tl) for tl, k in zip(layers, kinds) if k != "mla"]
    H, d = mla.num_heads, tc.hidden_size
    nope, rope, vd, kl = mla.qk_nope_head_dim, mla.qk_rope_head_dim, mla.v_head_dim, mla.kv_lora_rank
    ql = mla.q_a_proj.out_features if hasattr(mla, "q_a_proj") else 0
    hd = max(nope + rope, vd)
    hd += hd % 2
    E, ef, k, sff = _ffn_dims(layers, tc)
    ssm = {}
    if lin:
        a0 = lin[0]
        Hs, hs = a0.num_heads, a0.head_dim
        ns = Hs * hs
        if _bl_kind(a0) == "kda":
            fg = getattr(a0, "forget_gate", None) or a0
            nf = fg.f_proj.out_features if hasattr(fg, "f_proj") else a0.f_a_proj.out_features
            ng = a0.g_proj.out_features if hasattr(a0, "g_proj") else a0.g_a_proj.out_features
            K = a0.conv1d.weight.shape[-1] if hasattr(a0, "conv1d") else a0.q_conv1d.weight.shape[-1]
            ssm = dict(ssm_in=3 * ns + nf + Hs + ng, conv_k=K, ssm_heads=Hs, dk=hs, dv=hs)
        else:
            ssm = dict(ssm_in=4 * ns, conv_k=1, ssm_heads=Hs, dk=hs, dv=hs, ssm_norm_feats=2 * ns)
    u = _shell(tc, d=d, L=len(layers), H=H, hd=hd, ff=sff, experts=E, expert_ff=ef, top_k=k,
               attn_norm_feats=ql + kl, **ssm)
    emb = getattr(m, "word_embeddings", None) or m.embed_tokens
    _head_from(u, emb, m.norm, target.lm_head)
    rot_mla = getattr(m, "rotary_emb_mla", None) or m.rotary_emb
    rot_lin = m.rotary_emb
    # the remote code branches on config.rope_interleave (its other branch raises); the stock class always interleaves
    interleave = bool(tc.__dict__.get("rope_interleave", True))
    from types import SimpleNamespace as NS
    for ul, tl, kind in zip(u.layers, layers, kinds):
        a = _bl_attn(tl)
        if kind == "mla":
            _norm_any(ul.norm_a, tl.input_layernorm)
            sh = NS(q_a_proj=getattr(a, "q_a_proj", None), q_a_layernorm=getattr(a, "q_a_layernorm", None),
                    q_b_proj=getattr(a, "q_b_proj", None), q_proj=getattr(a, "q_proj", None),
                    kv_a_proj_with_mqa=a.kv_a_proj_with_mqa, kv_a_layernorm=a.kv_a_layernorm, kv_b_proj=a.kv_b_proj,
                    o_proj=getattr(a, "dense", None) or a.o_proj)
            _mla_from(ul.attn, sh, d=d, H=H, nope=nope, rope=rope, vd=vd, kl=kl, ql=ql,
                      inv_freq=rot_mla.inv_freq, att_s=float(rot_mla.attention_scaling), interleave=interleave,
                      scale=a.scaling)
            g = getattr(a, "g_proj", None)
            if g is not None:                                          # sigmoid output gate, head- or element-wise
                gw = _W(g)
                rows = gw.repeat_interleave(hd, 0) if gw.shape[0] == H else torch.cat(
                    [torch.cat([gw[h * vd:(h + 1) * vd], gw.new_zeros(hd - vd, d)]) for h in range(H)])
                ul.attn.g_proj.weight[:, :d] = rows
                ul.attn.gate_coef.copy_(torch.tensor([0.0, 1.0], dtype=F64))
        else:
            _norm_any(ul.norm_s, tl.input_layernorm)
            if kind == "kda":
                _bailing_kda(ul.ssm, a)
            else:
                _bailing_lightning(ul.ssm, a, rot_lin.inv_freq)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        mlp = tl.mlp
        _ffn_any(ul.ffn, mlp, tc, getattr(tc, "hidden_act", "silu"))
        if _expert_ws(mlp) is not None:
            f, gate = ul.ffn, getattr(mlp, "gate", None)
            if hasattr(gate, "expert_bias"):                           # the remote Ling router (read off its code)
                f.score_c.fill_(1.0)
                _cp(f.sel_bias, gate.expert_bias)
                f.grp.copy_(torch.tensor([float(gate.n_group), float(gate.topk_group), 0.0, float("-inf")], dtype=F64))
                f.renorm.fill_(1.0 if gate.top_k > 1 else 0.0)
                f.renorm_eps.fill_(1e-20)
                f.route_scale.fill_(float(gate.routed_scaling_factor))
            lims = (getattr(mlp, "expert_swiglu_limit", None), getattr(mlp, "share_expert_swiglu_limit", None))
            for i, lim in enumerate(lims):
                if lim:
                    f.clip_post[i] = float(lim)
    return u


# --- non-next-token readouts and inputs (M12): score / class / value heads, embeddings, continuous series ---
#
# A model whose output is not a next-token distribution is the same body with a different READOUT
# (UniversalElephantine.readout): the final norm, a pooling rule (none / last real token / first / mean), a
# linear head of any width with an optional bias (a score head is a narrow lm_head; an embedding is the
# identity or a projection), optionally a hidden layer before it (a value-head MLP), the logit softcap and an
# L2 normalisation; early exits (layer-wise rerankers) are readouts of the stream after fewer layers, each with
# its own head. A model whose INPUT is not tokens swaps the token table for UPatchEmbed. A flow-matching
# forecaster adds UFlowHead, the network its sampler calls; the sampler loop itself is runtime.

_POOLS = {"none": 0, "last": 1, "first": 2, "cls": 2, "mean": 3}


@torch.no_grad()
def set_readout(u, weight, bias=None, *, pool="none", pad_id=None, l2=False, mlp=None):
    """Replace u's readout head by a linear head `weight` (out, in) (+ bias) of any width, read after the
    pooling rule `pool` (real tokens: the forward's pad_mask, else ids != pad_id); mlp = (w1, b1, act) puts
    a hidden layer act(w1 h + b1) before it; l2 L2-normalises the output."""
    w = weight.detach().to(F64)
    if mlp is not None:
        w1, b1, act = mlp
        u.cfg.head_ff = w1.shape[0]
        u.head_in = _lin(w1.shape[1], w1.shape[0])
        _cp(u.head_in.weight, w1)
        if b1 is not None:
            _cp(u.head_in.bias, b1)
        u.head_act.fill_(float(_ACT_CODES.index(act)))
    u.cfg.head_out = w.shape[0]
    u.lm_head = _lin(w.shape[1], w.shape[0])
    _cp(u.lm_head.weight, w)
    if bias is not None:
        _cp(u.lm_head.bias, bias)
    u.pool.fill_(float(_POOLS[pool]))
    u.pool_pad.fill_(-1.0 if pad_id is None else float(pad_id))
    u.l2_out.fill_(float(l2))
    return u


@torch.no_grad()
def set_exits(u, exits, heads, scale=1.0):
    """Early-exit readouts: exit j reads norm(x after exits[j] layers) through heads[j] (an nn.Linear or a
    weight (out, d), times `scale`); the forward then returns one output per exit."""
    u.cfg.exits = tuple(int(e) for e in exits)
    assert len(heads) == len(exits) and all(0 <= e <= u.cfg.L for e in u.cfg.exits)
    mods = []
    for h in heads:
        w = (h if torch.is_tensor(h) else h.weight).detach().to(F64)
        lin = _lin(w.shape[1], w.shape[0])
        _cp(lin.weight, w * scale)
        b = None if torch.is_tensor(h) else _B(h)
        if b is not None:
            _cp(lin.bias, b * scale)
        mods.append(lin)
    u.exit_heads = nn.ModuleList(mods)
    u.cfg.head_out = mods[0].weight.shape[0]
    return u


@torch.no_grad()
def from_embedding(model, *, pool, l2=False, dense=None, causal=True):
    """A sentence-embedding model (sentence-transformers Transformer -> Pooling -> [Dense] -> [Normalize]) on a
    stock Llama-shaped body (Qwen2Model / Qwen3Model: the headless decoder, no lm_head): the readout is the
    pooled final hidden state (pool: 'last' = pooling_mode_lasttoken, 'mean' = pooling_mode_mean_tokens,
    'cls'), through the Dense projection (an nn.Linear or a (weight, bias) pair; identity if None), L2-
    normalised when the pipeline ends in Normalize. causal = False marks the bidirectional encoders (gte-Qwen2
    / stella's is_causal = False Qwen2Model, voyage's Qwen3BidirectionalModel): the CALLER runs them with
    causal=False, the host's full mask (u.bidirectional records it)."""
    from types import SimpleNamespace
    u = from_llama(SimpleNamespace(config=model.config, model=model, lm_head=None))
    d = model.config.hidden_size
    if dense is None:
        w, b = torch.eye(d, dtype=F64), None
    elif isinstance(dense, nn.Module):
        w, b = _W(dense), _B(dense)
    else:
        w, b = dense
    set_readout(u, w, b, pool=pool, l2=l2)
    u.bidirectional = not causal
    return u


@torch.no_grad()
def from_seq_cls(target):
    """A stock *ForSequenceClassification on a Llama-shaped body (LlamaForSequenceClassification,
    Qwen2/3ForSequenceClassification, ...): the body, then score (Linear(hidden, num_labels), no bias) read at
    the last non-pad token (the rightmost position whose id != config.pad_token_id; the last position when
    pad_token_id is None)."""
    from types import SimpleNamespace
    u = from_llama(SimpleNamespace(config=target.config, model=target.model, lm_head=None))
    set_readout(u, _W(target.score), _B(target.score), pool="last", pad_id=target.config.pad_token_id)
    return u


@torch.no_grad()
def from_grm(target):
    """GRM reward models (Ray2333 GRM-*, remote model.py AutoModelForCausalLMWithValueHead, read not run):
    a stock causal LM (`pretrained_model`) whose reward is v_head(hidden_states[-1]) at the last real token,
    v_head = ValueHead: vhead_layer_type 'linear' -> Linear(hidden, 1); 'mlp' -> [Linear, ReLU] x
    vhead_num_layers then Linear(neurons, 1) (dropout: an eval no-op). hidden_states[-1] is the final-normed
    stream (transformers' last hidden state). The host output is the reward (B, 1); the LM logits of the
    same weights are the plain loader of the body (from_gemma / from_llama)."""
    from types import SimpleNamespace
    lm = target.pretrained_model
    body = SimpleNamespace(config=lm.config, model=lm.model, lm_head=None)
    u = from_gemma(body) if lm.config.model_type.startswith("gemma") else from_llama(body)
    summ = target.v_head.summary
    if isinstance(summ, nn.Linear):
        return set_readout(u, _W(summ), _B(summ), pool="last")
    lins = [mod for mod in summ if isinstance(mod, nn.Linear)]
    assert len(lins) == 2 and isinstance(summ[1], nn.ReLU), "value-head MLPs of depth > 1: not mapped"
    set_readout(u, _W(lins[1]), _B(lins[1]), pool="last", mlp=(_W(lins[0]), _B(lins[0]), "relu"))
    return u


def _cutoffs(cutoff_layers, L):
    if cutoff_layers is None:
        return [L]
    return [cutoff_layers] if isinstance(cutoff_layers, int) else list(cutoff_layers)


@torch.no_grad()
def from_minicpm_layerwise(target, cutoff_layers=None):
    """BGE layer-wise MiniCPM reranker (LayerWiseMiniCPMForCausalLM, BAAI remote modeling_minicpm_reranker.py,
    read not run): a MiniCPM body (embeddings x scale_emb, MHA / GQA with RoPE, SwiGLU, sub-block outputs x
    scale_depth / sqrt(L), RMSNorm) whose outputs are EARLY EXITS: for each cutoff layer c in
    [start_layer, L] (the call's cutoff_layers, default [L]) the stream after c layers, final-normed, goes
    through its own head lm_head[c - start_layer] (head_multi; one shared head otherwise). head_type 'simple'
    (the released checkpoint): LayerWiseHead = Linear(hidden, 1), no bias, NOT divided by hidden /
    dim_model_base -- one relevance score per token per exit (the caller reads the last token); 'raw' /
    'complex': a vocabulary head on h / (hidden / dim_model_base). Outputs, as in the code, are ordered by
    layer; output i uses the head of cutoff_layers[i]."""
    tc, m = target.config, target.model
    assert not getattr(tc, "rope_scaling", None), "MiniCPM rope_scaling: only the plain rotary is mapped"
    assert (getattr(tc, "pretraining_tp", 1) or 1) == 1, "pretraining_tp > 1 (split heads): not mapped"
    d, L, H = tc.hidden_size, tc.num_hidden_layers, tc.num_attention_heads
    hd = d // H
    u = _shell(tc, d=d, L=L, H=H, hd=hd, ff=tc.intermediate_size)
    _head_from(u, m.embed_tokens, m.norm, None, emb_scale=float(tc.scale_emb))
    r = float(tc.scale_depth) / math.sqrt(L)
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _attn_any(ul.attn, *_qkv(a), n_kv=tc.num_key_value_heads, scale=hd ** -0.5, inv_freq=a.rotary_emb.inv_freq)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
        _fold_residual(ul, r)
    cut = [c for c in _cutoffs(cutoff_layers, L) if tc.start_layer <= c <= L]
    assert cut, f"cutoff layers must be in [{tc.start_layer}, {L}]"
    exits = sorted(set(cut))
    multi, ht = tc.head_multi, tc.head_type
    pick = lambda c: target.lm_head[c - tc.start_layer] if multi else target.lm_head
    lin = lambda h: h if isinstance(h, nn.Linear) else h.linear_head
    scale = 1.0 if ht not in ("raw", "complex") else float(tc.dim_model_base) / d
    set_exits(u, exits, [lin(pick(cut[i])) for i in range(len(exits))], scale)
    return u


@torch.no_grad()
def from_cost_wise_gemma(target, cutoff_layers=None):
    """BGE cost-wise Gemma-2 reranker (CostWiseGemmaForCausalLM, BAAI remote gemma_model.py, read not run): the
    stock Gemma-2 body (its layers ARE transformers' Gemma2DecoderLayer; from_gemma) with, when layer_wise,
    EARLY EXITS: for each cutoff layer c in range(start_layer, L + 1, layer_sep) (the call's cutoff_layers,
    default [L]) the stream after c layers, final-normed, through a score head CostWiseHead = Linear(hidden,
    1), no bias, then the final-logit softcap. As in the code, output i (ordered by layer) uses lm_head[i]
    -- the i-th head, not the head of layer c. layer_wise false is an LM (from_gemma). The token-compression
    path (compress_layer / compress_ratio: passage tokens mean-pooled in groups mid-network) is an opt-in
    call argument, off by default, and not carried."""
    from types import SimpleNamespace
    tc = target.config
    if not getattr(tc, "layer_wise", False):
        return from_gemma(target)
    u = from_gemma(SimpleNamespace(config=tc, model=target.model, lm_head=None))
    L = tc.num_hidden_layers
    ok = set(range(tc.start_layer, L + 1, tc.layer_sep))
    cut = [c for c in _cutoffs(cutoff_layers, L) if c in ok]
    assert cut, f"cutoff layers must be in {sorted(ok)}"
    exits = sorted(set(cut))
    set_exits(u, exits, [target.lm_head[i].linear_head for i in range(len(exits))])
    return u


@torch.no_grad()
def from_sundial(target, revin=False):
    """Sundial (thuml SundialForPrediction, remote modeling_sundial.py + flow_loss.py, read not run): a
    time-series forecaster. INPUT: a float series, not tokens -- SundialPatchEmbedding (UPatchEmbed): left-pad to
    a multiple of input_token_len, patches concatenated with their padding mask, hidden_layer -> act ->
    output_layer plus residual_layer; with forward(revin=True) the series is first instance-normalised
    (std floor 1e-2 -> 1). BODY: pre-LayerNorm decoder, MHA with q / k / v biases (no o bias), full rotate-half
    RoPE, SwiGLU, final LayerNorm -- the host's softmax AR layers. READOUT: the last position's hidden state z
    (identity head, pool last). HEAD: FlowLoss.net, a SimpleMLPAdaLN velocity network v(x, t, z) (UFlowHead).
    The sampler FlowLoss.sample (Gaussian noise, num_sampling_steps Euler steps of the network, then the
    RevIN de-normalisation and the generate() roll-out of further patches) is runtime: flow_sample runs it for
    a given noise tensor."""
    tc, m = target.config, target.model
    d, L, H = tc.hidden_size, tc.num_hidden_layers, tc.num_attention_heads
    hd = d // H
    net = target.flow_loss.net
    c = UConfig(vocab=1, d=d, L=L, heads=H, head_dim=hd, shared_ff=tc.intermediate_size, head_out=d,
                patch_len=tc.input_token_len, patch_mask=True, patch_ff=tc.intermediate_size,
                flow_out=net.out_channels, flow_width=net.model_channels, flow_depth=net.num_res_blocks,
                flow_freq=net.time_embed.frequency_embedding_size)
    u = UniversalElephantine(c)
    pe, P = m.embed_layer, u.patch
    for dst, src in ((P.w1, pe.hidden_layer), (P.w2, pe.output_layer), (P.wr, pe.residual_layer)):
        _cp(dst.weight, src.weight)
        _cp(dst.bias, src.bias)
    P.mlp_c.fill_(1.0)
    P.act.fill_(float(_ACT_CODES.index(str(tc.hidden_act).lower().replace("swish", "silu"))))
    if revin:
        P.rev.copy_(torch.tensor([1.0, 0.0, 1e-2], dtype=F64))
    _ln_from(u.norm, m.norm)
    _cp(u.lm_head.weight, torch.eye(d, dtype=F64))
    u.pool.fill_(float(_POOLS["last"]))
    for ul, tl in zip(u.layers, m.layers):
        a = tl.self_attn
        _ln_from(ul.norm_a, tl.norm1)
        _ln_from(ul.norm_f, tl.norm2)
        _attn_any(ul.attn, *_qkv(a), n_kv=H, scale=hd ** -0.5, inv_freq=a.rotary_emb.inv_freq)
        _mlp_any(ul.ffn, tl.ffn_layer, tc.hidden_act)
    fh = u.flow
    pairs = [(fh.t1, net.time_embed.mlp[0]), (fh.t2, net.time_embed.mlp[2]), (fh.cond, net.cond_embed),
             (fh.inp, net.input_proj), (fh.fin_ada, net.final_layer.adaLN_modulation[1]), (fh.fin, net.final_layer.linear)]
    for i, blk in enumerate(net.res_blocks):
        pairs += [(fh.m1[i], blk.mlp[0]), (fh.m2[i], blk.mlp[2]), (fh.ada[i], blk.adaLN_modulation[1])]
        _ln_from(fh.ln[i], blk.in_ln)
    for dst, src in pairs:
        _cp(dst.weight, src.weight)
        _cp(dst.bias, src.bias)
    fh.fin_norm.eps.fill_(float(net.final_layer.norm_final.eps))
    return u


# the round-3 attention features' loaders and setters (elephantine/branches_attn.py; tests/test_attn_gaps.py)
from tef_recipes.branches_attn import *  # noqa: E402,F401,F403
# the residual-stream / input / FFN / norm / loop branches' loaders (eleph-gaps-stream)
from tef_recipes.branches_stream import (  # noqa: E402,F401
    from_qwen4_exp, from_zamba2, from_nanochat, from_arloop, from_k2_horizon, from_longcat_flash,
    from_ouro, from_hrm_text, from_nanbeige, from_nanowhale, from_xing, from_hy_v4, from_sdar_selfcond,
    selfcond_signal, from_dory, gated_norm_from, zaya_residual_scaling_into, cfrd_binding_into,
    muse_residual_gate_into, from_dhara, from_crowd, from_llama_moe, from_arctic, from_hunyuan_capacity,
    llada2_block_moe_into, from_llama4, ernie55_router_into, from_zamba, muse_spark_moe_into)
