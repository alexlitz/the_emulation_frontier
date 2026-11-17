"""Is the mixer near-gap PAYABLE?  softmax attention >= SSM / linear / hybrid, measured on real models.

graph/model_graph.py MIXER_ORDER lets a softmax host count as emulating an SSM / linear-attention /
hybrid target at "near-gap" cost: no width, depth or norm charge beyond plain capacity dominance.
This module builds the best EXPLICIT constructions (weight settings, no training) by which softmax
attention reproduces the recurrent mixer of a real checkpoint, swaps them into the model, and scores
generation with the act/act_swap_agreement.py standard: on the ORIGINAL model's own greedy 64-token
continuations of the same 24 prompts, fp32, TF32 off,

  payable  <=>  top-1 agreement >= 99%  AND  >= 90% of 32-token greedy continuations identical
  (looser: no worse than the original model run in bf16; top-1 >= 98% and >= 50% identical).

THE CONSTRUCTIONS (each is computed literally: real fp32 softmaxes over the logits a host would form)

  sink-normaliser (undoes the softmax normaliser). A head with logits l_s and an extra zero-value key
      (a gpt-oss-style per-head sink logit, position-free) at logit sigma_t returns
      sum_s e^{l_s - sigma_t} v_s / (1 + sum_s e^{l_s - sigma_t}); with sigma_t = (row term) + L the
      output times e^L is the UNNORMALISED kernel sum up to a relative leak e^{-L} * sum_s e^{l_s - row}.
      The leak is bounded whenever the kernel decays (every SSM here), so L -> inf is a LIMIT.
  decay as a logit. exp(A (c_t - c_s)) = exp(A c_t) exp(-A c_s): key logit -A c_s, sink logit
      -A c_t + L. EXACT for any diagonal decay -- fixed (RetNet / RWKV-4: c_s = s, an ALiBi slope) or
      data-dependent (Mamba: c = cumsum(dt)) -- PROVIDED c_s is already in the residual stream. One
      softmax layer's logit q_t.k_s sees only tokens t and s, while prod_{r=s+1..t} exp(A dt_r) depends
      on every token in between, so a data-dependent decay costs a PRIOR layer that writes the prefix
      sum: a sink head with constant logits returns e^{-L'} sum_{r<=t} dt_r (relative leak t e^{-L'}).
  small temperature (linear attention). exp(eps q.k) = 1 + eps q.k + O(eps^2): two heads at +-eps with
      the same decay logits and sinks give (H(eps) - H(-eps)) / (2 eps) = sum_s decay * (q.k_s) v_s up to
      O(eps^2) truncation and an fp32 cancellation ~ 1e-7 / eps: Mamba-2 (SSD) / scalar-decay linear
      attention with q = C, k = B in ONE head pair per head.
  hard-selection shift (short convolutions, token shift). Logit -beta (t - s - j)^2 plus a zero-value
      sink at -beta/2 selects x_{t-j} (zero before the start) up to e^{-beta/2}: a kernel-k conv is k
      heads, exact as beta -> inf.
  diagonal bonus (RWKV's u). A logit term g(t - s) for t - s < m is exactly a cosine series in m RoPE
      frequencies 2 pi i / m (a DFT): (1/m) sum_i cos(2 pi i D / m) = [D = 0 mod m]. Exact within a
      context of m tokens at ~m extra q/k dims per head; aliases at D = m, 2m, ...
  GLU products. x * y = silu(x) y - silu(-x) y exactly, so every product of residual-stream features
      (B_s dt_s x_s, C_t . h_t, gates) is two GLU units -- exact, but each product costs an FFN stage.
  scalar nonlinearities (softplus, sigmoid) need K SiLU units: least-squares fits, error shrinks with K.
  delta rule (gated DeltaNet / Qwen3.5 / Qwen3-Next): S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t
      k_t^T has NON-COMMUTING transitions, so the kernel is not a product of per-token factors and no
      single logit carries it. Writing u_t = b_t (v_t - sum_{s<t} D(t,s) (k_t.k_s) u_s) turns it into
      linear attention on pseudo-values u, a strictly lower-triangular system (I + diag(b) M) u = b v;
      its Neumann / Jacobi iteration is one decayed linear-attention hop per host layer and is exact
      only after T - 1 hops: DEPTH proportional to CONTEXT. (Products of such transitions solve state
      tracking that fixed-depth softmax stacks are not believed to -- Grazzi et al. 2024; Merrill et
      al. 2024.)

What is NOT charged by the graph edge but the constructions need: heads (Mamba-1: one per (channel,
state) pair, d_inner * N = 24,576 for mamba-130m; RWKV-4: one per channel), residual width (the
states must sit in the residual for the C_t product), depth (prefix sums, products, shifts), and a
norm change (carrying state across sublayers under a stock single-rms pre-norm is impossible at any
width, block/prenorm_carry.py). The budget-matched variants (fixed decay = no prefix-sum layer;
H < d_inner * N heads; Neumann order p << T; a distilled softmax layer of the target's width) are
what a host that pays only capacity dominance can do.

MEASURED (2026-11-12, 24 prompts, every recurrent mixer swapped; python -m attn.mixer_gap_report):
  complete constructions pass -- mamba-130m 99.9% top-1 / 96% identical (26,116 heads x 4 layers,
  softplus by 256 SiLU units per channel: |A| reaches ~1e6, 64 units give 98.4% / 79%), RWKV-4-169m
  100% / 100% (769 heads x 2 layers), granite-4.0-h-350m Mamba-2 100% / 100% (101 heads x 3
  layers), LFM2-350M 99.9% / 100% in ONE stock layer (12,288 heads) or 3 heads x 3 layers; gated
  DeltaNet (Qwen3.5-0.8B) only at T - 1 Jacobi hops (32 hops 92% / 54%, 64 hops 98.9% / 100%).
  Budget-matched variants fail: fixed decay 85% / 4 - 12%, 16 - 4,096 shared decay rates 57 - 95%
  / <= 25%, RWKV-4 without the bonus 92% / 4%, distilled layers 2 - 16% top-1.

  python -m attn.mixer_gap_agreement mamba --device cuda:0           # one family, all its variants
  python -m attn.mixer_gap_agreement rwkv4 --only exact,softmax-m128
  python -m attn.mixer_gap_agreement qwen35 --tf5 --device cuda:0     # transformers 5.18
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from hf.attn_pin import attn_impl_for            # eager pinned (hf/attn_pin.py)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

N_NEW = 64
OUT = "attn/data/mixer_gap_agreement.jsonl"

MODELS = {
    "mamba": "state-spaces/mamba-130m-hf",
    "rwkv4": "RWKV/rwkv-4-169m-pile",
    "granite": "ibm-granite/granite-4.0-h-350m-base",     # Mamba-2 hybrid (28 mamba2 + 4 attention)
    "lfm2": "LiquidAI/LFM2-350M",                         # gated short-conv hybrid (10 conv + 6 attention)
    "qwen35": "Qwen/Qwen3.5-0.8B-Base",                   # gated-DeltaNet hybrid (18 GDN + 6 attention)
}


def _prompts():
    from act.act_swap_agreement import PROMPTS
    return PROMPTS


# =============================================================================================
# softmax primitives (literal fp32 softmax over the logits a host head would form)
# =============================================================================================
def causal_mask(T, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def sink_softmax(logits, sink, values, log_scale=0.0):
    """e^{log_scale} * sum_s softmax([sink, logits])_s v_s.  logits [..., T, S] (-inf where masked),
    sink broadcastable to [..., T] (the zero-value sink key's logit), values [..., S, dv]."""
    sink = torch.broadcast_to(sink, logits.shape[:-1])
    w = torch.softmax(torch.cat([sink[..., None], logits], -1), -1)[..., 1:]
    out = w @ values
    return out * math.exp(log_scale) if log_scale else out


def prefix_sum_layer(x, Lp):
    """A sink head with constant (zero) logits over s <= t and a sink at Lp, rescaled by e^{Lp}:
    e^{Lp} sum_{r<=t} x_r / (e^{Lp} + t + 1). x [T, C] -> [T, C] (relative leak (t+1) e^{-Lp})."""
    T = x.shape[0]
    lg = torch.zeros(T, T, device=x.device, dtype=x.dtype).masked_fill(~causal_mask(T, x.device), float("-inf"))
    return sink_softmax(lg, torch.full((T,), Lp, device=x.device, dtype=x.dtype), x, Lp)


def prefix_sum(x, Lp=None):
    return torch.cumsum(x, 0) if Lp is None else prefix_sum_layer(x, Lp)


def shift_heads(x, K, beta):
    """Hard-selection shift heads: [x_{t-j}] for j = 0..K-1 (zero before the start), each from logits
    -beta (t - s - j)^2 and a zero-value sink at -beta/2.  x [T, C] -> [K, T, C]."""
    T = x.shape[0]
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    D = t[:, None] - t[None, :]
    m = causal_mask(T, x.device)
    out = []
    for j in range(K):
        lg = (-beta * (D - j) ** 2).masked_fill(~m, float("-inf"))
        out.append(sink_softmax(lg, torch.full((T,), -beta / 2, device=x.device, dtype=x.dtype), x))
    return torch.stack(out)


def causal_conv(x, weight, bias, beta=None):
    """Depthwise causal conv (HF Conv1d, padding K-1, first T outputs). x [T, C], weight [C, 1, K].
    beta None: the exact conv; else the hard-selection shift-head construction."""
    K = weight.shape[-1]
    if beta is None:
        y = F.conv1d(x.T[None], weight, bias, padding=K - 1, groups=x.shape[1])[0, :, :x.shape[0]].T
        return y
    sh = shift_heads(x, K, beta)                                  # sh[j] = x_{t-j}
    w = weight[:, 0, :]                                           # out_t = sum_i w_i x_{t-(K-1)+i}
    y = sum(sh[K - 1 - i] * w[:, i] for i in range(K))
    return y + bias if bias is not None else y


def glu_product(x, y):
    """x * y as two SiLU-GLU units (silu(x) y - silu(-x) y): what a host FFN computes exactly."""
    return F.silu(x) * y - F.silu(-x) * y


# --- K-unit SiLU fits of a scalar nonlinearity (what a width-K host FFN computes) ---------------
class UnitFit(torch.nn.Module):
    """sum_i c_i silu(a (x - b_i)) + e x + f, least squares on the inputs the host actually sees."""

    def __init__(self, a, b, coef):
        super().__init__()
        self.a, self.b, self.coef = float(a), torch.tensor(b), torch.tensor(coef)

    def forward(self, x):
        b, c = self.b.to(x), self.coef.to(x)
        y = x * c[-2] + c[-1]
        if len(b):
            y = y + (F.silu(self.a * (x[..., None] - b)) * c[:-2]).sum(-1)
        return y


def fit_units(fn, xs, K, n_max=200_000, rel=False):
    """Least-squares K-unit SiLU fit of fn on samples xs (a 1-D tensor); returns (UnitFit, max err on xs).
    rel: minimise the RELATIVE error (rows weighted by 1 / |fn|) -- what a decay rate dt needs when
    it multiplies |A| up to ~1e6 (mamba-130m), where an absolute 1e-4 error flips the sign of dt."""
    xs = xs.detach().double().reshape(-1).cpu()
    if xs.numel() > n_max:
        xs = xs[torch.randperm(xs.numel(), generator=torch.Generator().manual_seed(0))[:n_max]]
    lo, hi = xs.min().item(), xs.max().item()
    grid = torch.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 4001, dtype=torch.float64)
    X = torch.cat([xs, grid])
    Y = fn(X)
    best = None
    for s in ([1.0] if K == 0 else [0.25, 0.5, 1, 2, 4, 8]):
        b = torch.linspace(lo, hi, K, dtype=torch.float64) if K else torch.zeros(0, dtype=torch.float64)
        a = s * max(K, 1) / max(hi - lo, 1e-6)
        feats = [F.silu(a * (X[:, None] - b))] if K else []
        Phi = torch.cat(feats + [X[:, None], torch.ones_like(X)[:, None]], 1)
        wt = 1.0 / Y.abs().clamp(min=1e-30) if rel else torch.ones_like(Y)
        coef = torch.linalg.lstsq(Phi * wt[:, None], (Y * wt)[:, None]).solution[:, 0]
        err = (Phi @ coef - Y).abs() * (wt if rel else 1.0)
        e = err[: xs.numel()].max().item()
        if best is None or e < best[0]:
            best = (e, a, b.tolist(), coef.tolist())
    e, a, b, coef = best
    return UnitFit(a, b, coef), e


def _hs(a, k):
    """The hidden-states argument of a mixer forward, positional or keyword."""
    if a:
        return a[0]
    for n in ("hidden_states", "input_states", "hidden", "x"):
        if n in k:
            return k[n]
    raise TypeError("no hidden states")


# =============================================================================================
# Mamba-1 (state-spaces/mamba-*-hf, HF MambaMixer)
# =============================================================================================
def mamba_parts(mod, h, cfg):
    """Everything up to the scan: x (post-conv silu), z (gate), dt, B, C. h [T, hidden]."""
    proj = mod.in_proj(h)                                        # [T, 2 d_inner]
    x, z = proj.chunk(2, -1)
    x = F.silu(causal_conv(x, mod.conv1d.weight, mod.conv1d.bias if mod.use_conv_bias else None,
                           cfg.get("conv_beta")))
    ts, B, C = torch.split(mod.x_proj(x), [mod.time_step_rank, mod.ssm_state_size, mod.ssm_state_size], -1)
    pre = mod.dt_proj(ts)
    sp = cfg.get("softplus_fit", {}).get(mod.layer_idx)
    dt = sp(pre) if sp is not None else F.softplus(pre)
    return x, z, dt, B, C, pre


def mamba_scan_exact(x, dt, A, B, C):
    """The selective scan, sequential fp32 (HF slow_forward order)."""
    T, d = x.shape
    S = torch.zeros(d, A.shape[1], device=x.device, dtype=x.dtype)
    ys = []
    for t in range(T):
        S = torch.exp(A * dt[t][:, None]) * S + (dt[t][:, None] * B[t][None, :]) * x[t][:, None]
        ys.append(S @ C[t])
    return torch.stack(ys)


def mamba_scan_sink(x, dt, A, B, C, L=30.0, Lp=None, decay="data", dt_mean=None, groups=None, chunk=192):
    """The scan as sink-normalised softmax heads: head (d,n) has key logit -A[d,n] c_s[d], sink
    -A[d,n] c_t[d] + L, value dt_s B_s[n] x_s[d]; y_t[d] = sum_n C_t[n] out_{(d,n)} (GLU products).
    decay="data": c = prefix sum of dt (oracle cumsum if Lp None, else the sink prefix-sum layer);
    decay="fixed": c_s = s * dt_mean[d] (no prefix-sum layer: what ONE softmax layer per mixer can do --
    a data-independent ALiBi decay; the data-dependent dt still enters the value).
    groups: (labels [d*N], H) -- H heads; head h's rate is the mean of A[d,n] dt[d] over its members."""
    T, d = x.shape
    N = A.shape[1]
    m = causal_mask(T, x.device)
    V = (dt[:, :, None] * B[:, None, :]) * x[:, :, None]            # [T, d, N]
    if groups is not None:
        lab, H = groups
        rate = (A[None] * dt[:, :, None]).reshape(T, d * N)         # per-(d,n) log-decay per step
        oh = torch.zeros(d * N, H, device=x.device, dtype=x.dtype)
        oh[torch.arange(d * N, device=x.device), lab] = 1.0
        rate_h = rate @ (oh / oh.sum(0).clamp(min=1))               # [T, H] centroid rates (linear in dt)
        c = -prefix_sum(rate_h, Lp)                                  # -sum rate >= 0 ; [T, H]
        lg = c.T[:, None, :].expand(-1, T, -1).masked_fill(~m, float("-inf"))   # [H, T, S]: key logit c_s
        W = torch.softmax(torch.cat([(c.T + L)[..., None], lg], -1), -1)[..., 1:]    # sink c_t + L
        Vf = V.reshape(T, d * N)
        out = torch.empty(T, d * N, device=x.device, dtype=x.dtype)
        for i0 in range(0, d * N, 2048):
            sl = slice(i0, min(d * N, i0 + 2048))
            out[:, sl] = torch.einsum("cts,sc->tc", W[lab[sl]], Vf[:, sl]) * math.exp(L)
        out = out.reshape(T, d, N)
        return (out * C[:, None, :]).sum(-1)
    if decay == "fixed":
        cc = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)[:, None] * dt_mean[None, :]
    else:
        cc = prefix_sum(dt, Lp)                                      # [T, d]
    y = torch.empty(T, d, device=x.device, dtype=x.dtype)
    for d0 in range(0, d, chunk):
        ds = slice(d0, min(d, d0 + chunk))
        lgk = (-A[ds][:, :, None] * cc[:, ds].T[:, None, :])       # [dc, N, S]: -A c_s
        lg = lgk[:, :, None, :].expand(-1, -1, T, -1).masked_fill(~m, float("-inf"))
        sink = lgk + L                                               # sink at row t: -A c_t + L
        out = sink_softmax(lg, sink, V[:, ds].permute(1, 2, 0)[..., None], L)[..., 0]   # [dc, N, T]
        y[:, ds] = (out * C.T[None]).sum(1).T
    return y


def mamba_forward(mod, hidden_states, cfg):
    outs = []
    A = -torch.exp(mod.A_log.float())
    for h in hidden_states:                                       # batch of 1 in practice
        x, z, dt, B, C, _ = mamba_parts(mod, h, cfg)
        scan = cfg.get("scan", "exact")
        if scan == "exact":
            y = mamba_scan_exact(x, dt, A, B, C)
        else:
            y = mamba_scan_sink(x, dt, A, B, C, L=cfg.get("L", 30.0), Lp=cfg.get("Lp"),
                                decay="fixed" if scan == "fixed" else "data",
                                dt_mean=cfg.get("dt_mean", {}).get(mod.layer_idx),
                                groups=cfg.get("groups", {}).get(mod.layer_idx))
        y = (y + x * mod.D) * F.silu(z)
        outs.append(mod.out_proj(y))
    return torch.stack(outs)


def mamba_install(model, cfg, layers=None):
    for i, blk in enumerate(model.backbone.layers):
        mod = blk.mixer
        if layers is not None and i not in layers:
            mod.forward = types.MethodType(type(mod).forward, mod)
            continue
        mod.forward = types.MethodType(lambda self, hs, *a, **k: mamba_forward(self, hs, cfg), mod)


def mamba_uninstall(model):
    for blk in model.backbone.layers:
        blk.mixer.forward = types.MethodType(type(blk.mixer).forward, blk.mixer)


def mamba_probe(model, seqs):
    """Per layer: the softplus inputs, mean dt per channel and per-(d,n) rate profiles (for groups)."""
    store = {}

    def make(mod):
        def f(self, hs, *a, **k):
            x, z, dt, B, C, pre = mamba_parts(self, hs[0], {})
            st = store.setdefault(self.layer_idx, {"pre": [], "dt": []})
            st["pre"].append(pre.detach().reshape(-1)[::7].cpu())
            st["dt"].append(dt.detach().cpu())
            return type(self).forward(self, hs, *a, **k)
        return f
    for blk in model.backbone.layers:
        blk.mixer.forward = types.MethodType(make(blk.mixer), blk.mixer)
    with torch.no_grad():
        for s in seqs:
            model(s, use_cache=False)
    mamba_uninstall(model)
    return store


def kmeans_groups(feat, H, iters=25, seed=0):
    """Cluster rows of feat [n, f] into H groups (cosine-free plain k-means on the GPU)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = feat.shape[0]
    if H >= n:
        return torch.arange(n, device=feat.device)
    cen = feat[torch.randperm(n, generator=g)[:H].to(feat.device)].clone()
    for _ in range(iters):
        lab = torch.cdist(feat, cen).argmin(1)
        cen = torch.zeros_like(cen).index_add_(0, lab, feat) / torch.bincount(lab, minlength=H).clamp(min=1)[:, None].float()
    return torch.cdist(feat, cen).argmin(1)


# =============================================================================================
# RWKV-4 (RWKV/rwkv-4-*-pile, HF RwkvSelfAttention)
# =============================================================================================
def rwkv4_wkv_exact(w, u, k, v):
    """wkv_t = (sum_{s<t} e^{-(t-1-s) w + k_s} v_s + e^{u + k_t} v_t) / (same with v = 1), fp32."""
    T, C = k.shape
    t = torch.arange(T, device=k.device, dtype=k.dtype)
    D = t[:, None] - t[None, :]
    lg = k.T[:, None, :] - w[:, None, None] * (D - 1)[None]
    lg = torch.where(D[None] == 0, (u[:, None] + k.T)[:, :, None].expand(-1, -1, T), lg)
    lg = lg.masked_fill(~causal_mask(T, k.device)[None], float("-inf"))
    return (torch.softmax(lg, -1) @ v.T[:, :, None])[..., 0].T


def rwkv4_wkv_softmax(w, u, k, v, m=128):
    """One softmax head per channel: logits k_s - w (t - s) (an ALiBi slope w per head) plus the
    diagonal bonus (u - w) [t - s = 0 mod m] written as a DFT cosine series in m RoPE frequencies.
    m=0: no bonus (plain ALiBi)."""
    T, C = k.shape
    t = torch.arange(T, device=k.device, dtype=k.dtype)
    D = t[:, None] - t[None, :]
    lg = k.T[:, None, :] - w[:, None, None] * D[None]
    if m:
        i = torch.arange(m, device=k.device, dtype=k.dtype)
        dirich = torch.cos(2 * math.pi * i[:, None, None] * D[None] / m).mean(0)   # [T, T]
        lg = lg + (u - w)[:, None, None] * dirich[None]
    lg = lg.masked_fill(~causal_mask(T, k.device)[None], float("-inf"))
    return (torch.softmax(lg, -1) @ v.T[:, :, None])[..., 0].T


def rwkv4_forward(mod, hidden, cfg):
    outs = []
    for h in hidden:
        beta = cfg.get("shift_beta")
        sh = torch.cat([torch.zeros_like(h[:1]), h[:-1]]) if beta is None else shift_heads(h, 2, beta)[1]
        mix = lambda mu: h * mu[0, 0] + sh * (1 - mu[0, 0])
        k = mod.key(mix(mod.time_mix_key)).float()
        v = mod.value(mix(mod.time_mix_value)).float()
        rpre = mod.receptance(mix(mod.time_mix_receptance))
        sg = cfg.get("sigmoid_fit", {}).get(mod.layer_id)
        r = sg(rpre) if sg is not None else torch.sigmoid(rpre)
        w, u = torch.exp(mod.time_decay.float()), mod.time_first.float()
        if cfg.get("scan", "exact") == "exact":
            wkv = rwkv4_wkv_exact(w, u, k, v)
        else:
            wkv = rwkv4_wkv_softmax(w, u, k, v, m=cfg.get("m", 128))
        outs.append(mod.output(r * wkv))
    return torch.stack(outs), None


def rwkv4_install(model, cfg, layers=None):
    for i, blk in enumerate(model.rwkv.blocks):
        mod = blk.attention
        if layers is None or i in layers:
            mod.forward = types.MethodType(lambda self, hs, state=None, use_cache=False: rwkv4_forward(self, hs, cfg), mod)
        else:
            mod.forward = types.MethodType(type(mod).forward, mod)


def rwkv4_uninstall(model):
    for blk in model.rwkv.blocks:
        blk.attention.forward = types.MethodType(type(blk.attention).forward, blk.attention)


def rwkv4_probe(model, seqs):
    store = {}

    def make(mod):
        def f(self, hs, state=None, use_cache=False):
            h = hs[0]
            sh = torch.cat([torch.zeros_like(h[:1]), h[:-1]])
            rpre = self.receptance(h * self.time_mix_receptance[0, 0] + sh * (1 - self.time_mix_receptance[0, 0]))
            store.setdefault(self.layer_id, []).append(rpre.detach().reshape(-1)[::3].cpu())
            return type(self).forward(self, hs, state=state, use_cache=use_cache)
        return f
    for blk in model.rwkv.blocks:
        blk.attention.forward = types.MethodType(make(blk.attention), blk.attention)
    with torch.no_grad():
        for s in seqs:
            model(s, use_cache=False)
    rwkv4_uninstall(model)
    return store


# =============================================================================================
# Mamba-2 (granite-4.0-h: GraniteMoeHybridMambaLayer, SSD)
# =============================================================================================
def mamba2_parts(mod, h, cfg):
    proj = mod.in_proj(h)
    gate, xBC, dtr = proj.split([mod.intermediate_size, mod.conv_dim, mod.num_heads], -1)
    xBC = mod.act(causal_conv(xBC, mod.conv1d.weight, mod.conv1d.bias if mod.use_conv_bias else None,
                              cfg.get("conv_beta")))
    x, B, C = torch.split(xBC, [mod.intermediate_size, mod.n_groups * mod.ssm_state_size,
                                mod.n_groups * mod.ssm_state_size], -1)
    pre = dtr + mod.dt_bias
    sp = cfg.get("softplus_fit", {}).get(mod.layer_idx)
    dt = sp(pre) if sp is not None else F.softplus(pre)
    dt = torch.clamp(dt, mod.time_step_limit[0], mod.time_step_limit[1])
    T = h.shape[0]
    rep = mod.num_heads // mod.n_groups
    x = x.reshape(T, mod.num_heads, mod.head_dim).float()
    B = B.reshape(T, mod.n_groups, -1).float().repeat_interleave(rep, 1)
    C = C.reshape(T, mod.n_groups, -1).float().repeat_interleave(rep, 1)
    return gate, x, B, C, dt.float(), pre


def mamba2_scan_exact(x, dt, A, B, C):
    T, H, P = x.shape
    S = torch.zeros(H, P, B.shape[-1], device=x.device, dtype=x.dtype)
    ys = []
    for t in range(T):
        S = torch.exp(dt[t] * A)[:, None, None] * S + (dt[t][:, None, None] * x[t][:, :, None]) * B[t][:, None, :]
        ys.append((S * C[t][:, None, :]).sum(-1))
    return torch.stack(ys)


def smallT_pair(G, dec, V, eps, L, m, order=2):
    """sum_s e^{dec_s - dec_t} G_ts v_s from softmax heads at small temperature: logits sgn*eps*G + dec_s,
    sink dec_t + L, rescaled by e^L. order 2: heads at +-eps, (H+ - H-)/(2 eps), error O(eps^2 G^2);
    order 4: also +-2 eps, (8 (H(e) - H(-e)) - (H(2e) - H(-2e))) / (12 eps), error O(eps^4 G^4).
    G [H, T, S], dec [H, S] (the sink of row t sits at dec_t + L), V [H, S, dv]."""
    def head(e):
        lg = (e * G + dec[:, None, :]).masked_fill(~m[None], float("-inf"))
        return sink_softmax(lg, dec + L, V, L)
    d1 = head(eps) - head(-eps)
    if order == 2:
        return d1 / (2 * eps)
    d2 = head(2 * eps) - head(-2 * eps)
    return (8 * d1 - d2) / (12 * eps)


def mamba2_scan_smallT(x, dt, A, B, C, eps=1e-2, L=30.0, Lp=None, decay="data", dt_mean=None, order=2):
    """SSD as softmax: per mamba head a pair (order 2) or quadruple (order 4) of softmax heads at
    small temperature with logits +-eps C_t.B_s - A c_s and sink -A c_t + L (c = prefix sum of dt),
    values dt_s x_s (smallT_pair). decay="fixed": c_s = s * dt_mean[h] (no prefix-sum layer)."""
    T, H, P = x.shape
    m = causal_mask(T, x.device)
    if decay == "fixed":
        cc = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)[:, None] * dt_mean[None, :]
    else:
        cc = prefix_sum(dt, Lp)                                       # [T, H]
    dec = (-A[None, :] * cc).T                                         # [H, S]: -A c_s >= 0
    G = torch.einsum("thn,shn->hts", C, B)                             # C_t . B_s
    V = (dt[:, :, None] * x).permute(1, 0, 2)                          # [H, S, P]
    return smallT_pair(G, dec, V, eps, L, m, order).permute(1, 0, 2)


def mamba2_forward(mod, hidden, cfg):
    outs = []
    A = -torch.exp(mod.A_log.float())
    for h in hidden:
        gate, x, B, C, dt, _ = mamba2_parts(mod, h, cfg)
        scan = cfg.get("scan", "exact")
        if scan == "exact":
            y = mamba2_scan_exact(x, dt, A, B, C)
        else:
            eps = cfg["eps_layer"][mod.layer_idx] if "eps_layer" in cfg else cfg.get("eps", 1e-2)
            y = mamba2_scan_smallT(x, dt, A, B, C, eps=eps, L=cfg.get("L", 30.0),
                                   Lp=cfg.get("Lp"), decay="fixed" if scan == "fixed" else "data",
                                   dt_mean=cfg.get("dt_mean", {}).get(mod.layer_idx), order=cfg.get("order", 2))
        y = y + x * mod.D[:, None]
        y = y.reshape(h.shape[0], -1)
        outs.append(mod.out_proj(mod.norm(y[None], gate[None])[0].to(h.dtype)))
    return torch.stack(outs)


def _granite_mixers(model):
    return [(i, l.mamba) for i, l in enumerate(model.model.layers) if getattr(l, "mamba", None) is not None]


def mamba2_install(model, cfg, layers=None):
    for i, mod in _granite_mixers(model):
        if layers is None or i in layers:
            mod.forward = types.MethodType(lambda self, *a, **k: mamba2_forward(self, _hs(a, k), cfg), mod)
        else:
            mod.forward = types.MethodType(type(mod).forward, mod)


def mamba2_uninstall(model):
    for i, mod in _granite_mixers(model):
        mod.forward = types.MethodType(type(mod).forward, mod)


def mamba2_probe(model, seqs):
    store = {}

    def make(mod):
        def f(self, *a, **k):
            _, _, B, C, dt, pre = mamba2_parts(self, _hs(a, k)[0], {})
            st = store.setdefault(self.layer_idx, {"pre": [], "dt": [], "cb": 0.0})
            st["pre"].append(pre.detach().reshape(-1).cpu())
            st["dt"].append(dt.detach().cpu())
            st["cb"] = max(st["cb"], torch.einsum("thn,shn->hts", C, B).abs().max().item())
            return type(self).forward(self, *a, **k)
        return f
    for i, mod in _granite_mixers(model):
        mod.forward = types.MethodType(make(mod), mod)
    with torch.no_grad():
        for s in seqs:
            model(s, use_cache=False)
    mamba2_uninstall(model)
    return store


# =============================================================================================
# LFM2 gated short conv (Lfm2ShortConv): y = out_proj(C * conv(B * x)), kernel 3
# =============================================================================================
STENCIL = {2: ((1.0, 0.5), (-1.0, -0.5)),                              # f'(0) ~ sum w f(a eps) / eps
           4: ((1.0, 8 / 12), (-1.0, -8 / 12), (2.0, -1 / 12), (-2.0, 1 / 12))}


def lfm2_onelayer(Bm, Cm, xx, weight, beta, L, ea, eb, order=2):
    """C_t * sum_j w_j B_{t-j} x_{t-j} in ONE softmax layer, no carried state: per (channel, offset j,
    stencil point (a, b)) a head with logits -beta (t - s - j)^2 + a B_s[c], a sink at L - b C_t[c] and
    value x_s[c] returns ~ e^{a B_{t-j} + b C_t - L} x_{t-j}; the mixed finite difference over (a, b)
    (order^2 heads) is B_{t-j} C_t x_{t-j} + O(eps^order).  Heads: hidden * K * order^2."""
    T, Cd = xx.shape
    t = torch.arange(T, device=xx.device, dtype=xx.dtype)
    D = t[:, None] - t[None, :]
    m = causal_mask(T, xx.device)
    K = weight.shape[-1]
    w = weight[:, 0, :]
    st = STENCIL[order]
    y = torch.zeros_like(xx)
    for j in range(K):
        pos = (-beta * (D - j) ** 2).masked_fill(~m, float("-inf"))            # [T, S]
        acc = torch.zeros_like(xx)
        for ai, wa in st:
            lg = pos[None] + (ai * ea) * Bm.T[:, None, :]                       # [C, T, S]
            for bi, wb in st:
                sink = L - (bi * eb) * Cm.T                                     # [C, T]
                f = sink_softmax(lg, sink, xx.T[:, :, None], L)[..., 0].T       # [T, C]
                acc = acc + (wa * wb) * f
        y = y + acc / (ea * eb) * w[:, K - 1 - j]
    return y


def lfm2_forward(mod, x, cfg):
    outs = []
    for h in x:
        Bm, Cm, xx = mod.in_proj(h).chunk(3, -1)
        if "kappa" in cfg:
            ea, eb = cfg["kappa"] / cfg["bmax"][mod.layer_idx], cfg["kappa"] / cfg["cmax"][mod.layer_idx]
            y = lfm2_onelayer(Bm, Cm, xx, mod.conv.weight, cfg["conv_beta"], cfg.get("L", 30.0), ea, eb,
                              cfg.get("order", 2))
            if mod.conv.bias is not None:
                y = y + Cm * mod.conv.bias
        else:
            conv = causal_conv(Bm * xx, mod.conv.weight, mod.conv.bias, cfg.get("conv_beta"))
            y = Cm * conv
        outs.append(mod.out_proj(y))
    return torch.stack(outs)


def lfm2_probe(model, seqs):
    store = {"bmax": {}, "cmax": {}}

    def make(mod):
        def f(self, *a, **k):
            Bm, Cm, _ = self.in_proj(_hs(a, k)[0]).chunk(3, -1)
            i = self.layer_idx
            store["bmax"][i] = max(store["bmax"].get(i, 0.0), Bm.abs().max().item())
            store["cmax"][i] = max(store["cmax"].get(i, 0.0), Cm.abs().max().item())
            return type(self).forward(self, *a, **k)
        return f
    for i, mod in _lfm2_mixers(model):
        mod.forward = types.MethodType(make(mod), mod)
    with torch.no_grad():
        for s in seqs:
            model(s, use_cache=False)
    lfm2_uninstall(model)
    return store


def _lfm2_mixers(model):
    return [(i, l.conv) for i, l in enumerate(model.model.layers) if getattr(l, "conv", None) is not None]


def lfm2_install(model, cfg, layers=None):
    for i, mod in _lfm2_mixers(model):
        if layers is None or i in layers:
            mod.forward = types.MethodType(lambda self, *a, **k: lfm2_forward(self, _hs(a, k), cfg), mod)
        else:
            mod.forward = types.MethodType(type(mod).forward, mod)


def lfm2_uninstall(model):
    for i, mod in _lfm2_mixers(model):
        mod.forward = types.MethodType(type(mod).forward, mod)


# =============================================================================================
# gated DeltaNet (Qwen3.5 / Qwen3-Next: Qwen3_5GatedDeltaNet)
# =============================================================================================
def gdn_parts(mod, h):
    T = h.shape[0]
    qkv = F.silu(causal_conv(mod.in_proj_qkv(h), mod.conv1d.weight, mod.conv1d.bias))
    q, k, v = torch.split(qkv, [mod.key_dim, mod.key_dim, mod.value_dim], -1)
    q = q.reshape(T, -1, mod.head_k_dim).float()
    k = k.reshape(T, -1, mod.head_k_dim).float()
    v = v.reshape(T, -1, mod.head_v_dim).float()
    z = mod.in_proj_z(h).reshape(T, -1, mod.head_v_dim)
    beta = mod.in_proj_b(h).sigmoid().float()
    g = -mod.A_log.float().exp() * F.softplus(mod.in_proj_a(h).float() + mod.dt_bias)
    rep = mod.num_v_heads // mod.num_k_heads
    if rep > 1:
        q, k = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
    q = _l2(q) / math.sqrt(mod.head_k_dim)                           # HF l2norm: x rsqrt(|x|^2 + 1e-6)
    k = _l2(k)
    return q, k, v, z, beta, g


def _l2(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gdn_exact(q, k, v, beta, g):
    """torch_recurrent_gated_delta_rule (state [H, dk, dv])."""
    T, H, dk = k.shape
    S = torch.zeros(H, dk, v.shape[-1], device=k.device, dtype=k.dtype)
    out = []
    for t in range(T):
        S = S * g[t].exp()[:, None, None]
        mem = (S * k[t][:, :, None]).sum(1)
        S = S + k[t][:, :, None] * ((v[t] - mem) * beta[t][:, None])[:, None, :]
        out.append((S * q[t][:, :, None]).sum(1))
    return torch.stack(out)


def decayed_linear_attn(a, b, x, G, strict, soft=None):
    """sum_s D(t,s) (a_t . b_s) x_s with D(t,s) = exp(G_t - G_s), over s <= t (strict: s < t).
    a, b [T, H, dk], x [T, H, dv], G [T, H] (cumulative log decay, non-increasing).
    soft=(eps, L): the small-temperature softmax pair with sinks (decay logit -G_s ... as in Mamba-2)."""
    T = a.shape[0]
    m = causal_mask(T, a.device)
    if strict:
        m = m & ~torch.eye(T, dtype=torch.bool, device=a.device)
    S = torch.einsum("thd,shd->hts", a, b)
    if soft is None:
        Dm = torch.exp((G.T[:, :, None] - G.T[:, None, :]).masked_fill(~m[None], float("-inf")))
        return torch.einsum("hts,hts,shv->thv", Dm, S, x)
    eps, L = soft[:2]
    order = soft[2] if len(soft) > 2 else 2
    return smallT_pair(S, (-G).T, x.permute(1, 0, 2), eps, L, m, order).permute(1, 0, 2)


def gdn_neumann(q, k, v, beta, g, p, soft=None):
    """Jacobi / Neumann iteration on u = beta (v - M u), M[t,s] = D(t,s)(k_t.k_s), s < t: u^(0) =
    beta v, p hops, then o = sum_{s<=t} D(t,s)(q_t.k_s) u_s. Exact at p >= T - 1. Each hop and the
    readout is one decayed linear-attention layer. soft=(kappa, L, order): each hop as small-temperature
    softmax heads; |k.k| <= 1 and |q.k| <= 1/sqrt(dk) a priori (l2-normalised), so the fixed scales are
    eps = kappa for the hops and kappa sqrt(dk) for the readout."""
    G = torch.cumsum(g, 0)
    sk = sq = None
    if soft is not None:
        kap, L, order = soft
        sk, sq = (kap, L, order), (kap * math.sqrt(k.shape[-1]), L, order)
    u = beta[:, :, None] * v
    for _ in range(p):
        u = beta[:, :, None] * (v - decayed_linear_attn(k, k, u, G, True, sk))
    return decayed_linear_attn(q, k, u, G, False, sq)


def gdn_forward(mod, hidden, cfg):
    outs = []
    for h in hidden:
        q, k, v, z, beta, g = gdn_parts(mod, h)
        scan = cfg.get("scan", "exact")
        if scan == "exact":
            o = gdn_exact(q, k, v, beta, g)
        else:
            soft = (cfg["kappa"], cfg.get("L", 30.0), cfg.get("order", 4)) if "kappa" in cfg else None
            o = gdn_neumann(q, k, v, beta, g, cfg.get("p", 0), soft)
        T = h.shape[0]
        o = mod.norm(o.reshape(-1, mod.head_v_dim).to(h.dtype), z.reshape(-1, mod.head_v_dim))
        outs.append(mod.out_proj(o.reshape(T, -1)))
    return torch.stack(outs)


def _gdn_mixers(model):
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return [(i, l.linear_attn) for i, l in enumerate(lm.layers) if getattr(l, "linear_attn", None) is not None]


def gdn_install(model, cfg, layers=None):
    for i, mod in _gdn_mixers(model):
        if layers is None or i in layers:
            mod.forward = types.MethodType(lambda self, *a, **k: gdn_forward(self, _hs(a, k), cfg), mod)
        else:
            mod.forward = types.MethodType(type(mod).forward, mod)


def gdn_uninstall(model):
    for i, mod in _gdn_mixers(model):
        mod.forward = types.MethodType(type(mod).forward, mod)


# =============================================================================================
# per-family variant lists:  name -> (cfg, cost note)
# =============================================================================================
def variants(family, probe=None, model=None):
    """[(name, cfg dict, cost dict)] -- cost: what the host pays per target mixer layer:
    heads, q/k dim, residual width (as a multiple of the target hidden), extra depth (host layers per
    target layer), and whether a group norm (non-stock pre-norm) is needed to carry state."""
    V = []
    if family == "mamba":
        d_in, N = 1536, 16
        V.append(("exact", dict(scan="exact"), dict(note="own fp32 scan (control)")))
        for L in (2, 4, 8, 16, 30):
            V.append((f"sink-L{L}", dict(scan="sink", L=float(L)),
                      dict(heads=d_in * N, width=d_in * N, depth=4, groupnorm=True,
                           note="oracle prefix sum, exact conv/softplus")))
        for Lp in (8, 16, 30):
            V.append((f"sink-L30-cum{Lp}", dict(scan="sink", L=30.0, Lp=float(Lp)),
                      dict(heads=d_in * N + d_in, width=d_in * N, depth=4, groupnorm=True,
                           note="prefix sum by a sink head")))
        for b in (4, 8, 16, 32):
            V.append((f"sink-L30-conv{b}", dict(scan="sink", L=30.0, conv_beta=float(b)),
                      dict(heads=d_in * N + 4, width=d_in * N, depth=4, groupnorm=True,
                           note="conv by hard-selection shift heads")))
        for K in (4, 16, 64):
            V.append((f"sink-L30-softplus{K}", dict(scan="sink", L=30.0, softplus_K=K),
                      dict(heads=d_in * N, width=d_in * N, ffn_units=K * d_in, depth=4, groupnorm=True,
                           note=f"softplus by {K} SiLU units")))
        V.append(("sink-L30-softplus256", dict(scan="sink", L=30.0, softplus_K=256),
                  dict(heads=d_in * N, width=d_in * N, ffn_units=256 * d_in, depth=4, groupnorm=True,
                       note="softplus by 256 SiLU units")))
        for K in (64, 256):
            V.append((f"sink-L30-softplusrel{K}", dict(scan="sink", L=30.0, softplus_K=K, softplus_rel=True),
                      dict(heads=d_in * N, width=d_in * N, ffn_units=K * d_in, depth=4, groupnorm=True,
                           note=f"softplus by {K} SiLU units, relative-error fit (keeps dt > 0)")))
        V.append(("full-construction", dict(scan="sink", L=30.0, Lp=30.0, conv_beta=32.0, softplus_K=64),
                  dict(heads=d_in * N + d_in + 4, width=d_in * N, ffn_units=64 * d_in, depth=4, groupnorm=True,
                       note="every piece built: sink L30, prefix-sum L'30, conv beta32, softplus 64 units")))
        V.append(("full-construction-sp256", dict(scan="sink", L=30.0, Lp=30.0, conv_beta=32.0, softplus_K=256),
                  dict(heads=d_in * N + d_in + 4, width=d_in * N, ffn_units=256 * d_in, depth=4, groupnorm=True,
                       note="every piece built, softplus by 256 SiLU units (|A| reaches ~1e6 in mamba-130m)")))
        for H in (16, 64, 256, 1024, 4096):
            V.append((f"heads{H}", dict(scan="sink", L=30.0, groups_H=H),
                      dict(heads=H, width=d_in * N, depth=4, groupnorm=True,
                           note="head budget H (k-means-shared decay rates)")))
        V.append(("fixed-decay", dict(scan="fixed", L=30.0),
                  dict(heads=d_in * N, width=d_in * N, depth=3, groupnorm=True,
                       note="no prefix-sum layer: data-independent decay (ALiBi slope dt_mean)")))
    elif family == "rwkv4":
        C = 768
        V.append(("exact", dict(scan="exact"), dict(note="own fp32 wkv (control)")))
        for m in (128, 64, 32, 16, 0):
            V.append((f"softmax-m{m}", dict(scan="softmax", m=m),
                      dict(heads=C, qk_dim=2 + 2 * m, width=1, depth=1, groupnorm=False,
                           note=f"one ALiBi head per channel; diagonal bonus by {m} DFT RoPE frequencies"
                                + (" (exact for T <= m)" if m >= 128 else ""))))
        for b in (4, 8, 16, 32):
            V.append((f"softmax-m128-shift{b}", dict(scan="softmax", m=128, shift_beta=float(b)),
                      dict(heads=C + 1, qk_dim=258, width=2, depth=2, groupnorm=True,
                           note="token shift by a hard-selection head")))
        for K in (4, 16, 64):
            V.append((f"softmax-m128-sigmoid{K}", dict(scan="softmax", m=128, sigmoid_K=K),
                      dict(heads=C, qk_dim=258, width=1, ffn_units=K * C, depth=2, groupnorm=True,
                           note=f"receptance sigmoid by {K} SiLU units")))
        V.append(("full-construction", dict(scan="softmax", m=128, shift_beta=32.0, sigmoid_K=64),
                  dict(heads=C + 1, qk_dim=258, width=2, ffn_units=64 * C, depth=2, groupnorm=True,
                       note="every piece built")))
    elif family == "granite":
        H = 48
        V.append(("exact", dict(scan="exact"), dict(note="own fp32 SSD scan (control)")))
        for eps in (1e-1, 1e-2, 1e-3):
            V.append((f"smallT-eps{eps:g}", dict(scan="smallT", eps=eps, L=30.0),
                      dict(heads=2 * H, qk_dim=128 + 2, width=1, depth=3, groupnorm=True,
                           note="2-head pair, absolute eps; oracle prefix sum, exact conv/softplus")))
        for order in (2, 4):
            for kap in (1.0, 0.3, 0.1, 0.03):
                V.append((f"smallT-o{order}-k{kap:g}", dict(scan="smallT", kappa=kap, order=order, L=30.0),
                          dict(heads=order * H, qk_dim=128 + 2, width=1, depth=3, groupnorm=True,
                               note=f"{order} heads per mamba head, eps = {kap:g} / max|C.B| per layer")))
        best = dict(scan="smallT", kappa=0.3, order=4, L=30.0)
        for Lp in (16, 30):
            V.append((f"smallT-o4-k0.3-cum{Lp}", dict(best, Lp=float(Lp)),
                      dict(heads=4 * H + 1, qk_dim=130, width=1, depth=3, groupnorm=True,
                           note="prefix sum by a sink head")))
        for b in (8, 16, 32):
            V.append((f"smallT-o4-k0.3-conv{b}", dict(best, conv_beta=float(b)),
                      dict(heads=4 * H + 4, qk_dim=130, width=1, depth=3, groupnorm=True,
                           note="conv by shift heads")))
        for K in (4, 16, 64):
            V.append((f"smallT-o4-k0.3-softplus{K}", dict(best, softplus_K=K),
                      dict(heads=4 * H, qk_dim=130, width=1, ffn_units=K * H, depth=3, groupnorm=True,
                           note=f"softplus by {K} SiLU units")))
        V.append(("full-construction", dict(best, Lp=30.0, conv_beta=32.0, softplus_K=64),
                  dict(heads=4 * H + 5, qk_dim=130, width=1, ffn_units=64 * H, depth=3, groupnorm=True,
                       note="every piece built")))
        V.append(("fixed-decay", dict(best, scan="fixed"),
                  dict(heads=4 * H, qk_dim=130, width=1, depth=2, groupnorm=True,
                       note="no prefix-sum layer: data-independent decay")))
        o2 = dict(scan="smallT", kappa=0.1, order=2, L=30.0)
        V.append(("full-construction-o2", dict(o2, Lp=30.0, conv_beta=32.0, softplus_K=64),
                  dict(heads=2 * H + 5, qk_dim=130, width=1, ffn_units=64 * H, depth=3, groupnorm=True,
                       note="every piece built, 2 heads per mamba head (kappa 0.1)")))
        V.append(("fixed-decay-o2", dict(o2, scan="fixed"),
                  dict(heads=2 * H, qk_dim=130, width=1, depth=2, groupnorm=True,
                       note="no prefix-sum layer, 2 heads per mamba head")))
    elif family == "lfm2":
        V.append(("exact", dict(), dict(note="own fp32 conv (control)")))
        for b in (2, 4, 8, 16, 32):
            V.append((f"conv{b}", dict(conv_beta=float(b)),
                      dict(heads=3, width=2, depth=3, groupnorm=True, note="conv by 3 hard-selection shift heads")))
        for order in (2, 4):
            for kap in (0.3, 0.1, 0.03):
                V.append((f"onelayer-o{order}-k{kap:g}", dict(kappa=kap, order=order, conv_beta=32.0, L=30.0),
                          dict(heads=1024 * 3 * order ** 2, width=1, depth=1, groupnorm=False,
                               note=f"ONE stock softmax layer: per (channel, offset) {order}x{order} stencil heads "
                                    "(B, C products in the logits, hard-selection shift beta 32)")))
    elif family == "qwen35":
        V.append(("exact", dict(scan="exact"), dict(note="own fp32 gated delta rule (control)")))
        for p in (0, 1, 2, 4, 8, 16, 32, 64, 96):
            V.append((f"neumann-p{p}", dict(scan="neumann", p=p),
                      dict(heads=16, depth=p + 1, groupnorm=True,
                           note=f"{p} Jacobi hops, exact linear-attention hops (p >= T-1 is exact)")))
        for kap in (0.3, 0.1):
            V.append((f"neumann-p96-o4-k{kap:g}", dict(scan="neumann", p=96, kappa=kap, order=4, L=30.0),
                      dict(heads=4 * 16, depth=97, groupnorm=True,
                           note="every hop as 4 small-temperature softmax heads (oracle prefix sum of g)")))
    return V


# =============================================================================================
# harness
# =============================================================================================
@torch.no_grad()
def greedy(model, ids, n_new=N_NEW):
    """Greedy continuation, no EOS stop, full recompute (the constructions keep no cache)."""
    seq = ids
    for _ in range(n_new):
        nxt = model(seq, use_cache=False).logits[:, -1:].argmax(-1)
        seq = torch.cat([seq, nxt], 1)
    return seq[:, ids.shape[1]:]


@torch.no_grad()
def cont_logp(model, seq, P):
    return F.log_softmax(model(seq, use_cache=False).logits[0, P - 1:-1].double(), -1).cpu()


def score(model, enc, ref_seq, ref_lp, ref_cont, gen=True):
    top1 = kl = 0.0
    n = ex32 = ex64 = 0
    div = []
    for ids, s, lq, rc in zip(enc, ref_seq, ref_lp, ref_cont):
        lp = cont_logp(model, s, ids.shape[1])
        kl += (lq.exp() * (lq - lp)).sum().item()
        top1 += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
        n += lp.shape[0]
        if gen:
            g = greedy(model, ids)
            same = (g[0] == rc[0]).cpu().numpy()
            ex32 += bool(same[:32].all())
            ex64 += bool(same.all())
            div.append(int(np.argmin(same)) if not same.all() else N_NEW)
    r = dict(top1=top1 / n, kl=kl / n)
    if gen:
        r.update(ex32=ex32 / len(enc), ex64=ex64 / len(enc), div=float(np.mean(div)))
    return r


FAMILY = {
    "mamba": (mamba_install, mamba_uninstall),
    "rwkv4": (rwkv4_install, rwkv4_uninstall),
    "granite": (mamba2_install, mamba2_uninstall),
    "lfm2": (lfm2_install, lfm2_uninstall),
    "qwen35": (gdn_install, gdn_uninstall),
}


def n_mixers(model, family):
    if family == "mamba":
        return list(range(len(model.backbone.layers)))
    if family == "rwkv4":
        return list(range(len(model.rwkv.blocks)))
    return [i for i, _ in {"granite": _granite_mixers, "lfm2": _lfm2_mixers, "qwen35": _gdn_mixers}[family](model)]


def prepare(family, model, cfg, store, device):
    """Resolve fits / groups / means the cfg asks for from the reference-pass probe."""
    cfg = dict(cfg)
    if "softplus_K" in cfg:
        K = cfg.pop("softplus_K")
        rel = cfg.pop("softplus_rel", False)
        cfg["softplus_fit"] = {i: fit_units(F.softplus, torch.cat(st["pre"]), K, rel=rel)[0].to(device)
                               for i, st in store.items()}
    if "sigmoid_K" in cfg:
        K = cfg.pop("sigmoid_K")
        cfg["sigmoid_fit"] = {i: fit_units(torch.sigmoid, torch.cat(xs), K)[0].to(device) for i, xs in store.items()}
    if family == "lfm2" and "kappa" in cfg:   # eps = kappa / max |B|, kappa / max |C| per layer
        cfg["bmax"], cfg["cmax"] = store["bmax"], store["cmax"]
    elif "kappa" in cfg:                  # eps = kappa / max |C.B| per layer (a fixed weight scale)
        cfg["eps_layer"] = {i: cfg["kappa"] / max(st["cb"], 1e-6) for i, st in store.items()}
    if cfg.get("scan") == "fixed":
        cfg["dt_mean"] = {i: torch.cat(st["dt"]).mean(0).to(device) for i, st in store.items()}
    if "groups_H" in cfg:
        H = cfg.pop("groups_H")
        cfg["groups"] = {}
        for blk in model.backbone.layers:
            mod = blk.mixer
            A = -torch.exp(mod.A_log.float())
            dt = torch.cat(store[mod.layer_idx]["dt"]).to(device)                # [Ttot, d]
            feat = (A[None] * dt[:, :, None]).reshape(dt.shape[0], -1).T        # [d*N, Ttot]
            cfg["groups"][mod.layer_idx] = (kmeans_groups(feat, H), H)
    return cfg


def load(family, device, model_id=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = model_id or MODELS[family]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32,
                                                 attn_implementation=attn_impl_for(mid)).to(device).eval()
    return mid, tok, model


def run(family, device="cuda:0", only=None, layers=None, n_prompts=24, sink=None, model_id=None,
        gen=True, label=None, bf16=True, per_layer=False):
    t0 = time.time()
    mid, tok, model = load(family, device, model_id)
    install, uninstall = FAMILY[family]
    mix = n_mixers(model, family)
    prompts = _prompts()[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    ref_cont = [greedy(model, ids) for ids in enc]
    ref_seq = [torch.cat([i, c], 1) for i, c in zip(enc, ref_cont)]
    ref_lp = [cont_logp(model, s, i.shape[1]) for s, i in zip(ref_seq, enc)]
    probe = {"mamba": mamba_probe, "rwkv4": rwkv4_probe, "granite": mamba2_probe, "lfm2": lfm2_probe}.get(family)
    store = probe(model, ref_seq) if probe else {}
    print(f"{mid}: {len(mix)} recurrent mixer layers; reference in {time.time() - t0:.0f}s",
          file=sys.stderr, flush=True)
    rows = []
    layer_sets = [{i} for i in mix] if per_layer else [layers]
    vs = variants(family)
    if bf16:
        vs = vs + [("bf16", None, dict(note="the ORIGINAL model in bf16 (numerics control)"))]
    for layers in layer_sets:
        for name, cfg, cost in vs:
            if only and name not in only or (cfg is None and layers is not None):
                continue
            t1 = time.time()
            if cfg is None and layers is None:                   # bf16 control (whole model)
                sd = {n: p.data.cpu() for n, p in model.named_parameters()}
                for n, p in model.named_parameters():
                    p.data = sd[n].to(torch.bfloat16).to(device)
                r = score(model, enc, ref_seq, ref_lp, ref_cont, gen)
                for n, p in model.named_parameters():
                    p.data = sd[n].to(device)
            else:
                c = prepare(family, model, cfg, store, device)
                install(model, c, layers)
                try:
                    r = score(model, enc, ref_seq, ref_lp, ref_cont, gen)
                finally:
                    uninstall(model)
            row = dict(model=label or mid, family=family, variant=name, layers=None if layers is None else sorted(layers),
                       n_mixers=len(mix), n_prompts=len(enc), cfg={k: v for k, v in (cfg or {}).items()},
                       cost=cost, **r, secs=round(time.time() - t1, 1))
            rows.append(row)
            if sink:
                sink(row)
            print(f"  {name:>26} layers={'all' if layers is None else sorted(layers)}: top1 {r['top1']:.4f}  "
                  f"ex32 {r.get('ex32', float('nan')):.2f}  ex64 {r.get('ex64', float('nan')):.2f}  KL {r['kl']:.2e}  "
                  f"({time.time() - t1:.0f}s)", file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=sorted(FAMILY))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", help="comma-separated variant names")
    ap.add_argument("--layers", help="comma-separated mixer layer indices to swap (default: all)")
    ap.add_argument("--per-layer", action="store_true", help="swap one mixer layer at a time")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--no-gen", action="store_true", help="teacher-forced metrics only")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--tf5", action="store_true", help="run under transformers 5.18 (Qwen3.5)")
    a = ap.parse_args()
    if a.tf5 and not os.environ.get("MIXER_GAP_TF5"):
        import subprocess
        tf5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prelude = ("import sys, types\n"
                   "m = types.ModuleType('transformers.dependency_versions_check'); m.dep_version_check = lambda *a, **k: None\n"
                   "sys.modules['transformers.dependency_versions_check'] = m\n"
                   "sys.modules['fla'] = None\n"
                   f"sys.argv = {['mixer_gap'] + sys.argv[1:]!r}\n"
                   "import runpy; runpy.run_module('attn.mixer_gap_agreement', run_name='__main__')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([tf5, root]), MIXER_GAP_TF5="1")
        sys.exit(subprocess.run([sys.executable, "-c", prelude], env=env).returncode)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    only = a.only.split(",") if a.only else None
    layers = {int(x) for x in a.layers.split(",")} if a.layers else None
    run(a.family, a.device, only=only, layers=layers, n_prompts=a.n_prompts, sink=sink, gen=not a.no_gen,
        bf16=not a.no_bf16 and not a.per_layer, per_layer=a.per_layer)


if __name__ == "__main__":
    main()
