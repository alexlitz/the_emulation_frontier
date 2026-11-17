"""Approximate, performance-preserving conversion Gemma-4 (teacher) -> gpt-oss (student).

Sequential, layer by layer (GPTQ-style): student layer i is built from teacher layer pi(i) with
closed-form maps, then its linear maps are refit by least squares on the *student's own* input
stream against the teacher's residual stream (so later layers correct upstream error and the
dropped layers' updates are absorbed). See approx/__init__.py for the method summary and
approx/plan.py for which steps are exact.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F

from approx.arch import (SCfg, TCfg, attend, t_ffn, s_ffn, apply_rope, gelu_tanh, rms, rmsnorm, rope_cs, s_attn,
                         s_logits, student_inv_freq, swiglu_oai, t_attn, t_dense, t_expert_fn,
                         t_logits, t_router, teacher_inv_freq)
from approx.importance import Teacher, final_logprobs, kl_rows, stream


@dataclass
class Opts:
    layer_select: str = "importance"     # importance (ablation dKL, type-aware DP) | first
    attn_pairs: str = "importance"       # importance (score-variance x freq-matched assignment) | first
    attn_kv: str = "shared"              # shared: every student kv group of a teacher kv head keeps the
                                         #   same rope pairs (one pattern), different V subspaces
                                         # split: disjoint pairs (each student head sees a partial score)
    attn_alloc: str = "uniform"          # uniform | importance (student kv groups / q slots by head importance)
    attn_heads: str = "temp"             # replicate | temp (spare same-group copies at other temperatures)
    temps: tuple = (1.0, 0.6, 1.6, 0.35, 2.5, 0.8, 1.25, 0.2)
    v_subspace: str = "eig"              # eig (output-weighted PCA) | first
    attn_refit: bool = True              # LS refit of o_proj (+bias) on the student stream
    ffn_group: str = "modularity"        # modularity (excess co-activation) | coact | consecutive
    ffn_router: str = "lse"              # lse (weighted LS onto log-sum-exp of member logits) | mean
    ffn_dense: str = "trade"             # none | spare (top units in the 64 spare slots) | trade
    drop_experts: int = 0                # drop the k least important experts before grouping
    unit_select: str = "importance"      # importance | first
    act_fit: bool = True                 # per-unit gate scale (GeLU -> swiglu-oai, clamp) + up range
    ffn_refit: bool = True               # router LS, per-expert down LS, per-channel output fit
    grad_steps: int = 0                  # >0: gradient block reconstruction of the FFN after the LS refits
    grad_lr: float = 3e-3
    norm_lin: str = "ls"                 # ls (per-channel LS) | scalar (mean 1/rms per norm)
    target: str = "residual"             # residual (track teacher stream) | local
    softcap_fit: bool = True
    importance: str = "attr"             # attr (grad x act) | act (activation energy)
    rope_theta: float | None = None      # None: teacher sliding theta
    ridge: float = 1e-2                  # per-feature (standardized) ridge of the LS refits (if cv_frac = 0)
    cv_frac: float = 0.25                # last fraction of calibration sequences held out: every LS refit
                                         #   picks its ridge from `ridges` by held-out error, and the
                                         #   gradient refit early-stops on it
    ridges: tuple = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    ffn_down_ls: bool = True             # per-expert down_proj LS (underdetermined: ~2k tokens / 2881 units)
    cv_keep_prior: bool = False          # o_proj: the closed form is a candidate of the held-out selection
                                         #   (added after the reported real runs: global layers' LS refit
                                         #   hurt held-out attention error, see README)
    store_dtype: str = "bfloat16"        # each finished layer is rounded to this before the stream moves on
    pair_seqs: int = 4
    pair_queries: int = 64
    bs: int = 4
    verbose: bool = True


def naive_opts(**kw):
    """baseline (b'): exact folds only; everything approximate done by truncation, no calibration."""
    return replace(Opts(), layer_select="first", attn_pairs="first", attn_heads="replicate",
                   v_subspace="first", attn_refit=False, ffn_group="consecutive", ffn_router="mean",
                   ffn_dense="none", unit_select="first", act_fit=False, ffn_refit=False, norm_lin="scalar",
                   softcap_fit=False, importance="act", **kw)


# ----------------------------------------------------------------------------- small utils

def _lsq(G, B, ridge, prior=None):
    """argmin ||Phi W - Y||^2 + lam ||W - prior||^2 from G = Phi^T Phi, B = Phi^T Y (float64)."""
    lam = ridge * G.diagonal().mean().clamp_min(1e-12)
    A = G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    rhs = B + (lam * prior if prior is not None else 0)
    return torch.linalg.solve(A, rhs)


def _lsq_aug(G, B, ridge, prior=None):
    """Ridge LS with an intercept, from the (weighted) Gram of [phi, 1]: G = sum [phi,1][phi,1]^T,
    B = sum [phi,1] y^T. Solved on CENTERED features with a per-feature ridge lam * Var(phi_j)
    (scale-free), and the intercept recovered from the means. Without centering, near-constant
    feature directions (the residual stream's "massive activation" dims) are collinear with the
    intercept and the solution splits them into huge canceling bias / weight pairs that do not
    survive bf16 storage. The ridge is lam * (Var_j + mean_j Var_j): scale-aware per feature, with a
    floor so near-constant features (whose part belongs in the intercept) cannot get huge
    coefficients. prior (optional) pulls the weights (not the intercept) toward it."""
    n = G[-1, -1].clamp_min(1e-12)
    s = G[:-1, -1]
    Cov = G[:-1, :-1] - torch.outer(s, s) / n
    Cr = B[:-1] - torch.outer(s, B[-1]) / n
    d = Cov.diagonal().clamp_min(0) + Cov.diagonal().clamp_min(0).mean() + 1e-10   # per-feature + global floor
    A = Cov + ridge * torch.diag(d)
    rhs = Cr + (ridge * d[:, None] * prior[:-1] if prior is not None else 0)
    W = torch.linalg.solve(A, rhs)
    b = (B[-1] - s @ W) / n
    return torch.cat([W, b[None]])


def _chunks(N, nfit, ch):
    """(start, stop, held_out) chunks of [0, N) that never straddle the fit / held-out boundary."""
    for lo, hi, ho in ((0, nfit, False), (nfit, N, True)):
        for b in range(lo, hi, ch):
            yield b, min(b + ch, hi), ho


def _cv_lsq(Gf, Bf, Gh, Bh, yyh, ridges, prior=None, keep_prior=False):
    """_lsq_aug on the fit Gram for each ridge; pick the one with the lowest held-out squared error
    (computed in closed form from the held-out Gram: yy - 2<W, Bh> + <W, Gh W>). keep_prior: the
    prior itself (no refit) is a candidate too (ridge reported as inf)."""
    best = None
    if keep_prior and prior is not None:
        W = prior.to(Gf.dtype)
        best = ((yyh - 2 * (W * Bh).sum() + (W * (Gh @ W)).sum()).item(), float("inf"), W)
    for r in ridges:
        W = _lsq_aug(Gf, Bf, r, prior)
        err = (yyh - 2 * (W * Bh).sum() + (W * (Gh @ W)).sum()).item()
        if best is None or err < best[0]:
            best = (err, r, W)
    return best[2], best[1], best[0] / max(float(yyh), 1e-30)


def _largest_remainder(weights, total, floor=0):
    w = np.maximum(np.asarray(weights, dtype=float), 0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    n = len(w)
    out = np.full(n, floor)
    rest = total - floor * n
    q = w / w.sum() * rest
    out += np.floor(q).astype(int)
    for i in np.argsort(-(q - np.floor(q)))[: total - out.sum()]:
        out[i] += 1
    return out


def layer_map(tcfg: TCfg, scfg: SCfg, dkl=None, mode="importance"):
    """Choose L_S of L_T teacher layers (order kept) minimizing the summed ablation dKL of the
    dropped ones, with teacher global layers only on student full-attention positions (a sliding
    student layer would truncate them); teacher sliding layers on full positions cost a little
    (exact for contexts <= window). Returns pi (list of teacher indices)."""
    LT, LS = tcfg.L, scfg.L
    if mode == "first" or dkl is None:
        return list(range(LS))
    INF = float("inf")
    cost = np.full((LT + 1, LS + 1), INF)
    back = {}
    cost[0, 0] = 0.0
    d = np.asarray(dkl, dtype=float)
    eps = 1e-3 * (d.mean() + 1e-12)
    for t in range(LT):
        for s in range(LS + 1):
            if cost[t, s] == INF:
                continue
            c = cost[t, s] + d[t]                                   # drop t
            if c < cost[t + 1, s]:
                cost[t + 1, s], back[(t + 1, s)] = c, (t, s, False)
            if s < LS:                                              # keep t at student slot s
                tg, sf = tcfg.layer_types[t] == "full_attention", scfg.layer_types[s] == "full_attention"
                if tg and not sf:
                    continue
                c = cost[t, s] + (eps if (sf and not tg) else 0.0)
                if c < cost[t + 1, s + 1]:
                    cost[t + 1, s + 1], back[(t + 1, s + 1)] = c, (t, s, True)
    if cost[LT, LS] == INF:
        raise ValueError("no type-compatible layer map")
    pi, key = [], (LT, LS)
    while key != (0, 0):
        t, s, kept = back[key]
        if kept:
            pi.append(t)
        key = (t, s)
    return pi[::-1]


# ----------------------------------------------------------------------------- attention

def pair_stats(qn, kn, f_t, f_s, window, rep, omega, nseq=4, M=64, ncand=3):
    """Per teacher kv head g and rope pair p: I[g,p] = sum_t omega_t E_m Var_{n~A_t(m,.)}[s_tp(m,n)]
    (the attention-weighted variance of the pair's score contribution -- a per-query constant does
    not change the softmax), and Err[g,p,i] = the same variance of the change when pair p rotates
    at student frequency f_s[i] instead of f_t[p] (only for the `ncand` nearest student
    frequencies; others inf)."""
    B, T, nq, hd = qn.shape
    nkv, P, PS = kn.shape[2], hd // 2, f_s.shape[0]
    dev = qn.device
    lf = lambda f: torch.log(f.clamp_min(1e-9))
    dist = (lf(f_t)[:, None] - lf(f_s)[None, :]).abs()                       # (P, PS)
    cand = dist.topk(min(ncand, PS), dim=1, largest=False).indices         # (P, ncand)
    I = torch.zeros(nkv, P, device=dev)
    Err = torch.full((nkv, P, PS), float("inf"), device=dev)
    Ec = torch.zeros(nkv, P, cand.shape[1], device=dev)
    qc = torch.complex(qn[..., :P].float(), qn[..., P:].float())
    kc = torch.complex(kn[..., :P].float(), kn[..., P:].float())
    g = torch.Generator(device="cpu").manual_seed(0)
    nm = 0
    for b in range(min(nseq, B)):
        lo = min(T - 1, T // 8)
        m = torch.randperm(T - lo, generator=g)[:M].to(dev) + lo
        n = torch.arange(T, device=dev)
        D = (m[:, None] - n[None, :]).float()
        valid = D >= 0
        if window:
            valid = valid & (D < window)
        ph = torch.polar(torch.ones_like(D)[..., None].expand(-1, -1, P).contiguous(), D[..., None] * f_t)
        phs = [torch.polar(torch.ones_like(D)[..., None].expand(-1, -1, P).contiguous(), D[..., None] * f_s[cand[:, c]])
               for c in range(cand.shape[1])]
        for t in range(nq):
            gk = t // rep
            prod = qc[b, m, t][:, None, :] * kc[b, :, gk].conj()[None]        # (M,T,P)
            sp = (prod * ph).real
            s = sp.sum(-1).masked_fill(~valid, float("-inf"))
            A = s.softmax(-1)[..., None]                                    # (M,T,1)
            var = lambda z: ((A * z * z).sum(1) - (A * z).sum(1) ** 2).sum(0)
            I[gk] += omega[t] * var(sp)
            for c in range(cand.shape[1]):
                Ec[gk, :, c] += omega[t] * var((prod * (phs[c] - ph)).real)
        nm += m.numel()
    I, Ec = I / nm, Ec / nm
    for gk in range(nkv):
        Err[gk].scatter_(1, cand, Ec[gk])
    return I, Err


def assign_pairs(I, Err, PS, exclude=()):
    """Hungarian: student slot i <- teacher pair p maximizing sum(I_p - Err[p,i]); slots whose best
    gain is <= 0 stay empty (-1)."""
    from scipy.optimize import linear_sum_assignment
    gain = (I[:, None] - Err).cpu().double().numpy()
    gain[list(exclude), :] = -1e30
    gain[~np.isfinite(gain)] = -1e30
    r, c = linear_sum_assignment(-gain)
    slot = np.full(PS, -1)
    for p, i in zip(r, c):
        if gain[p, i] > 0:
            slot[i] = p
    return slot


def _ls_scalar(y, z, dims):
    """per-group LS scalar alpha minimizing ||y - alpha z||^2 summed over `dims`."""
    return (y.float() * z.float()).sum(dims) / (z.float() ** 2).sum(dims).clamp_min(1e-12)


def attn_convert(lw, tcfg: TCfg, scfg: SCfg, l, st, f_s, att_scale, C, opts: Opts, head_imp, pair_data):
    """Closed-form student attention weights for teacher layer l. Returns (weights, info)."""
    hd, nkv, keqv = tcfg.head(l)
    nq, P, PS, hs_ = tcfg.nq, hd // 2, scfg.hd // 2, scfg.hd
    rep = nq // nkv
    dev = lw["self_attn.q_proj.weight"].device
    Wq = lw["self_attn.q_proj.weight"].float().view(nq, hd, -1)
    Wk = lw["self_attn.k_proj.weight"].float().view(nkv, hd, -1)
    Wv = Wk if keqv else lw["self_attn.v_proj.weight"].float().view(nkv, hd, -1)
    Wo = lw["self_attn.o_proj.weight"].float().view(-1, nq, hd)              # (h, nq, hd)
    wqn, wkn = lw["self_attn.q_norm.weight"].float(), lw["self_attn.k_norm.weight"].float()
    Ht = Wq.shape[-1]
    # rms-norm linearization of q/k/v (per head scalar)
    # q/k/v RMSNorm linearization, as affine maps of the normalized input x (last column = bias):
    #   ls:     LS fit of the teacher's normalized q / k / v on [x, 1] (captures how rms(Wq x) varies
    #           with the direction of x)
    #   scalar: one LS scalar per head (q_hat ~ alpha * w * Wq x)
    if opts.norm_lin == "ls" and "Qfit" in st:
        Q = st["Qfit"].to(dev).view(nq, hd, Ht + 1)
        K = st["Kfit"].to(dev).view(nkv, hd, Ht + 1)
        Vt = st["Vfit"].to(dev).view(nkv, hd, Ht + 1)
    else:
        aq, ak, av = st["aq"].to(dev), st["ak"].to(dev), st["av"].to(dev)
        z1 = lambda M: F.pad(M, (0, 1))
        Q = z1(aq[:, None, None] * wqn[None, :, None] * Wq)                     # (nq, hd, H+1)
        K = z1(ak[:, None, None] * wkn[None, :, None] * Wk)
        Vt = z1(av[:, None, None] * Wv)
    omega = head_imp.to(dev).float().clamp_min(0)
    if omega.sum() <= 0:
        omega = torch.ones(nq, device=dev)
    omegaG = omega.view(nkv, rep).sum(1)
    # student kv groups per teacher kv head
    NS = scfg.nkv
    if opts.attn_alloc == "importance":
        ns = _largest_remainder(omegaG.cpu().numpy(), NS)
    else:
        ns = _largest_remainder(np.ones(nkv), NS)
    # pairs per student group
    if pair_data is not None:
        I, Err = pair_data
    groups = []                                    # (teacher kv head G, slot->pair array)
    for G in range(nkv):
        used = []
        for j in range(ns[G]):
            if opts.attn_pairs == "first":
                start = (j * PS) if opts.attn_kv == "split" else 0
                slot = np.array([p if p < P else -1 for p in range(start, start + PS)])
            else:
                if opts.attn_kv == "split" or not groups or groups[-1][0] != G:
                    slot = assign_pairs(I[G], Err[G], PS, exclude=used if opts.attn_kv == "split" else ())
                else:
                    slot = groups[-1][1]
            used += [p for p in slot if p >= 0]
            groups.append((G, slot))
    # V subspaces per teacher kv head: top-(64 * ns) of C^1/2 M C^1/2 (output-weighted PCA)
    venc, vdec = {}, {}
    for G in range(nkv):
        r = hs_ * ns[G]
        if r == 0:
            continue
        heads = list(range(G * rep, (G + 1) * rep))
        if opts.v_subspace == "first":
            E = torch.eye(hd, device=dev)
            venc[G], vdec[G] = E[:, :r], E[:r, :]
            continue
        w = omega[heads] / omega[heads].sum().clamp_min(1e-12)
        Cv = sum(w[i] * st["C"][t].to(dev) for i, t in enumerate(heads)) + 1e-6 * torch.eye(hd, device=dev)
        Mo = sum(w[i] * (Wo[:, t, :].T @ Wo[:, t, :]) for i, t in enumerate(heads))
        ev, U = torch.linalg.eigh(Cv)
        Ch, Cih = (U * ev.clamp_min(1e-10).sqrt()) @ U.T, (U * ev.clamp_min(1e-10).rsqrt()) @ U.T
        ew, Uw = torch.linalg.eigh(Ch @ Mo @ Ch)
        Ur = Uw[:, ew.argsort(descending=True)[:r]]
        venc[G], vdec[G] = Cih @ Ur, Ur.T @ Ch                                   # v' = v @ enc ; v ~ v' @ dec
    # q slots per student group
    q_w = torch.zeros(scfg.nq, hs_, Ht + 1, device=dev)
    k_w = torch.zeros(NS, hs_, Ht + 1, device=dev)
    v_w = torch.zeros(NS, hs_, Ht + 1, device=dev)
    o_w = torch.zeros(Ht, scfg.nq, hs_, device=dev)
    beta = st["beta"].to(dev) if opts.norm_lin == "ls" else lw["post_attention_layernorm.weight"].float() * st["irms_o"]
    qscale = (scfg.hd ** 0.5) / (att_scale ** 2)
    slots_info = []
    jg = {}
    for j, (G, slot) in enumerate(groups):
        jg.setdefault(G, []).append(j)
    for j, (G, slot) in enumerate(groups):
        jj = jg[G].index(j)
        for i, p in enumerate(slot):
            if p < 0:
                continue
            k_w[j, i], k_w[j, i + PS] = K[G, p], K[G, p + P]
        enc = venc[G][:, jj * hs_:(jj + 1) * hs_]                               # (hd, 64)
        dec = vdec[G][jj * hs_:(jj + 1) * hs_, :]                               # (64, hd)
        v_w[j] = enc.T @ Vt[G]
        heads = list(range(G * rep, (G + 1) * rep))
        per = scfg.nq // NS
        if opts.attn_alloc == "importance":
            cnt = _largest_remainder(omega[heads].cpu().numpy(), per)
        else:
            cnt = _largest_remainder(np.ones(rep), per)
        s0 = j * per
        for hi, t in enumerate(heads):
            for c in range(cnt[hi]):
                tau = opts.temps[c % len(opts.temps)] if opts.attn_heads == "temp" else 1.0
                for i, p in enumerate(slot):
                    if p < 0:
                        continue
                    q_w[s0, i], q_w[s0, i + PS] = Q[t, p] * qscale * tau, Q[t, p + P] * qscale * tau
                if c == 0:
                    o_w[:, s0, :] = (beta[:, None] * (Wo[:, t, :] @ dec.T)) / C
                slots_info.append((s0, t, j, tau))
                s0 += 1
    H = scfg.h
    pad = lambda M: F.pad(M, (0, H - Ht))
    q_w, k_w, v_w = q_w.reshape(-1, Ht + 1), k_w.reshape(-1, Ht + 1), v_w.reshape(-1, Ht + 1)
    out = {"self_attn.q_proj.weight": pad(q_w[:, :Ht]), "self_attn.q_proj.bias": q_w[:, Ht].contiguous(),
           "self_attn.k_proj.weight": pad(k_w[:, :Ht]), "self_attn.k_proj.bias": k_w[:, Ht].contiguous(),
           "self_attn.v_proj.weight": pad(v_w[:, :Ht]), "self_attn.v_proj.bias": v_w[:, Ht].contiguous(),
           "self_attn.o_proj.weight": F.pad(o_w.reshape(Ht, -1), (0, 0, 0, H - Ht)),
           "self_attn.o_proj.bias": torch.zeros(H, device=dev),
           "self_attn.sinks": torch.full((scfg.nq,), -1e4, device=dev)}
    return out, {"ns": ns.tolist(), "slots": slots_info, "groups": [(G, s.tolist()) for G, s in groups]}


# ----------------------------------------------------------------------------- FFN

def group_experts(coact, usage, nE, mode="coact", drop=None):
    """Partition kept teacher experts into nE groups (sizes balanced) maximizing within-group
    co-activation (so a token's top-k teacher experts fall into few student experts)."""
    E = coact.shape[0]
    keep = [e for e in range(E) if drop is None or e not in set(drop)]
    if mode == "consecutive":
        sz = math.ceil(len(keep) / nE)
        return [keep[i * sz:(i + 1) * sz] for i in range(nE)]
    Co = coact.double().cpu().numpy().copy()
    if mode == "modularity":           # excess co-activation over independence (balances group usage)
        u = np.asarray(usage, dtype=float)
        u = u / max(u.sum(), 1e-12) * np.trace(Co) / max(np.trace(Co), 1e-12)
        Co = Co - np.outer(u, u) * (Co.sum() / max(np.outer(u, u).sum(), 1e-12))
    np.fill_diagonal(Co, 0)
    Co = Co + Co.T
    sizes = [len(keep) // nE + (1 if i < len(keep) % nE else 0) for i in range(nE)]
    rem = sorted(keep, key=lambda e: -float(usage[e]))
    groups = []
    for sz in sizes:                     # greedy: seed with the most used remaining expert, add best partners
        if not rem:
            groups.append([])
            continue
        gset = [rem.pop(0)]
        while len(gset) < sz and rem:
            sc = [Co[e, gset].sum() for e in rem]
            gset.append(rem.pop(int(np.argmax(sc))))
        groups.append(gset)
    gid = {e: gi for gi, gs in enumerate(groups) for e in gs}
    for _ in range(20):                  # pairwise swap local search
        improved = False
        for a in keep:
            for b in keep:
                ga, gb = gid[a], gid[b]
                if ga >= gb:
                    continue
                A = [x for x in groups[ga] if x != a]
                Bg = [x for x in groups[gb] if x != b]
                d = Co[b, A].sum() + Co[a, Bg].sum() - Co[a, A].sum() - Co[b, Bg].sum()
                if d > 1e-12:
                    groups[ga], groups[gb] = A + [b], Bg + [a]
                    gid[a], gid[b] = gb, ga
                    improved = True
        if not improved:
            break
    return groups


def unit_plan(groups, unit_imp, dense_imp, I_s, Ie, Id, mode, dense_mode):
    """Which teacher units fill each student expert's I_s slots: members' units + D dense units
    replicated in every expert (the routing weights sum to 1, so a unit present in all experts acts
    as a dense unit). D is chosen to maximize the kept importance (a dense unit costs one slot in
    each of the nE experts)."""
    nE = len(groups)
    need = max(len(g) for g in groups) * Ie
    if dense_mode == "none":
        Ds = [0]
    elif dense_mode == "spare":
        Ds = [max(0, min(Id, I_s - need))]
    else:
        Ds = sorted(set([max(0, min(Id, I_s - need))] + list(range(0, min(Id, I_s) + 1, max(1, Id // 32)))))
    dorder = (torch.argsort(dense_imp, descending=True) if mode == "importance" else torch.arange(Id)).tolist()
    best = None
    for D in Ds:
        val = float(dense_imp[dorder[:D]].sum()) if D else 0.0
        plan = []
        for gs in groups:
            cand = [(float(unit_imp[e, u].detach()), e, u) for e in gs for u in range(Ie)]
            if mode == "importance":
                cand.sort(key=lambda c: -c[0])
            k = I_s - D
            plan.append([(e, u) for _, e, u in cand[:k]])
            val += sum(c[0] for c in cand[:k])
        if best is None or val > best[0] + 1e-12:
            best = (val, D, plan)
    _, D, plan = best
    return plan, dorder[:D]


def fit_act(gate, up, grid=(1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3), limit=7.0, alpha=1.702):
    """squared error per unit of the student activation for each gate scale c in `grid`:
    student unit = (1/c) * clamp(c g, <=7) * sigmoid(alpha * clamp(c g)) * u  vs  gelu_tanh(g) * u.
    (The up projection is separately scaled by 6/max|u| so u stays inside the up clamp.)"""
    ref = gelu_tanh(gate) * up
    errs = []
    for c in grid:
        gs = (c * gate).clamp(max=limit)
        errs.append(((gs * torch.sigmoid(alpha * gs) / c * up - ref) ** 2).sum(0))
    errs = torch.stack(errs)
    return errs


def ffn_closed(lw, tcfg: TCfg, scfg: SCfg, groups, plan, dense_units, st, C, opts: Opts):
    """student MoE weights (HF layout) from the plan."""
    dev = lw["experts.gate_up_proj"].device
    Ht, H, Is, Ie = tcfg.h, scfg.h, scfg.I, tcfg.Ie
    gu = lw["experts.gate_up_proj"]                               # (E, 2Ie, Ht)  (rows upcast when gathered)
    dn = lw["experts.down_proj"]                                  # (E, Ht, Ie)
    w2 = lw["pre_feedforward_layernorm_2.weight"].float()
    w1 = lw["pre_feedforward_layernorm.weight"].float()
    pes = lw["router.per_expert_scale"].float()
    a_out, b_out = st["a_out"].to(dev), st["b_out"].to(dev)       # f ~ a * m1 + b * m2 (per channel)
    Gd, Ud = lw["mlp.gate_proj.weight"].float() * w1, lw["mlp.up_proj.weight"].float() * w1
    Dd = lw["mlp.down_proj.weight"].float() * a_out[:, None] / C  # (Ht, Id)
    nE = scfg.E
    gate = torch.zeros(nE, Is, Ht, device=dev)
    up = torch.zeros(nE, Is, Ht, device=dev)
    upb = torch.full((nE, Is), -1.0, device=dev)
    down = torch.zeros(nE, Is, Ht, device=dev)
    cE, rE = st.get("c_e"), st.get("r_e")
    cD, rD = st.get("c_d"), st.get("r_d")
    du = torch.tensor(list(dense_units), dtype=torch.long, device=dev)
    for E_, units in enumerate(plan):
        ne = len(units)
        if ne:
            ee = torch.tensor([e for e, _ in units], device=dev)
            uu = torch.tensor([u for _, u in units], device=dev)
            g_, u_ = gu[ee, uu].float() * w2, gu[ee, Ie + uu].float() * w2
            d_ = dn[ee, :, uu].float() * b_out[None] * (pes[ee] / C)[:, None]
            if opts.act_fit and cE is not None:
                c, r = cE[ee, uu].float(), rE[ee, uu].float()
            else:
                c, r = torch.ones(ne, device=dev), None
            kap = 6.0 / r.clamp_min(1e-6) if r is not None else torch.ones(ne, device=dev)
            gate[E_, :ne], up[E_, :ne], down[E_, :ne] = c[:, None] * g_, kap[:, None] * u_, d_ / (c * kap)[:, None]
        nd = du.numel()
        if nd:
            g_, u_, d_ = Gd[du], Ud[du], Dd[:, du].T
            if opts.act_fit and cD is not None:
                c, r = cD[du].float(), rD[du].float()
                kap = 6.0 / r.clamp_min(1e-6)
            else:
                c, kap = torch.ones(nd, device=dev), torch.ones(nd, device=dev)
            gate[E_, ne:ne + nd], up[E_, ne:ne + nd] = c[:, None] * g_, kap[:, None] * u_
            down[E_, ne:ne + nd] = d_ / (c * kap)[:, None]
    # router
    Wr = lw["router.proj.weight"].float() * lw["router.scale"].float()[None] * Ht ** -0.5   # logits = Wr @ n
    rw = torch.zeros(nE, Ht, device=dev)
    rb = torch.zeros(nE, device=dev)
    for E_, gs in enumerate(groups):
        if gs:
            rw[E_] = Wr[gs].mean(0)
            rb[E_] = math.log(len(gs))
        else:
            rb[E_] = -1e4
    padh = lambda M: F.pad(M, (0, H - Ht))
    gu_s = torch.zeros(nE, H, 2 * Is, device=dev)
    gu_s[:, :Ht, ::2] = gate.transpose(1, 2)
    gu_s[:, :Ht, 1::2] = up.transpose(1, 2)
    del gate, up
    gub = torch.zeros(nE, 2 * Is, device=dev)
    gub[:, 1::2] = upb
    return {"post_attention_layernorm.weight": F.pad(torch.full((Ht,), math.sqrt(Ht / H), device=dev), (0, H - Ht)),
            "mlp.router.weight": padh(rw), "mlp.router.bias": rb,
            "mlp.experts.gate_up_proj": gu_s, "mlp.experts.gate_up_proj_bias": gub,
            "mlp.experts.down_proj": F.pad(down, (0, H - Ht)), "mlp.experts.down_proj_bias": torch.zeros(nE, H, device=dev)}


@torch.enable_grad()
def grad_refit_ffn(fw, sc: SCfg, gm, tgt, steps, lr, dev, seqs_per_step=4, log=print, nval=None):
    """Block reconstruction: Adam on the student FFN's router (W, b), gate/up biases, per-unit
    gate and up scales, down projection and bias, minimizing ||FFN(g) - target||^2 / ||target||^2
    on the calibration stream (last 1/4 of the sequences held out; the best held-out step is kept).
    Unlike the least-squares refits this can use the student's nonlinearities (top-k, clamps,
    gating) -- e.g. to approximate the teacher's per-token RMS normalization of its FFN output."""
    import torch.nn as nn
    N = gm.shape[0]
    nval = nval if nval else max(1, N // 4)
    Ht = tgt(0, 1).shape[-1]
    gu = fw["mlp.experts.gate_up_proj"].to(dev).to(torch.bfloat16)     # frozen; bf16 = its storage precision
    fw["mlp.experts.gate_up_proj"] = None
    P = {"rw": fw["mlp.router.weight"].to(dev).float(), "rb": fw["mlp.router.bias"].to(dev).float(),
         "gub": fw["mlp.experts.gate_up_proj_bias"].to(dev).float(),
         "gs": torch.ones(sc.E, sc.I, device=dev), "us": torch.ones(sc.E, sc.I, device=dev),
         "dn": fw["mlp.experts.down_proj"][..., :Ht].to(dev).float(), "dnb": fw["mlp.experts.down_proj_bias"][:, :Ht].to(dev).float()}
    P = {k: nn.Parameter(v.clone()) for k, v in P.items()}
    rms_ = lambda t: t.detach().pow(2).mean().sqrt().clamp_min(1e-8).item()
    groups = [{"params": [P["gs"], P["us"]], "lr": lr},
              {"params": [P["gub"]], "lr": lr * max(rms_(P["gub"]), 0.1)},
              {"params": [P["rw"]], "lr": lr * rms_(P["rw"])}, {"params": [P["rb"]], "lr": lr},
              {"params": [P["dn"]], "lr": lr * rms_(P["dn"])}, {"params": [P["dnb"]], "lr": lr * max(rms_(P["dnb"]), 1e-3)}]
    opt = torch.optim.Adam(groups)
    wpost = fw["post_attention_layernorm.weight"].to(dev).float()

    def fwd(g):
        H = g.shape[-1]
        x = rmsnorm(g, wpost, sc.eps).reshape(-1, H)
        logits = x @ P["rw"].T + P["rb"]
        tv, ti = logits.topk(sc.topk, -1)
        tw = tv.softmax(-1)

        def ef(e, xe):
            z = xe @ gu[e].float()
            gg = z[:, ::2] * P["gs"][e] + P["gub"][e, ::2]
            uu = z[:, 1::2] * P["us"][e] + P["gub"][e, 1::2]
            return swiglu_oai(gg, uu, sc.limit, sc.alpha) @ P["dn"][e] + P["dnb"][e]
        out = torch.zeros(x.shape[0], Ht, device=dev)
        flat = ti.reshape(-1)
        for e in flat.unique().tolist():
            tok, kk = (ti == e).nonzero(as_tuple=True)
            out.index_add_(0, tok, ef(e, x[tok]) * tw[tok, kk, None])
        return out

    def val_err():
        num = den = 0.0
        with torch.no_grad():
            for b in range(N - nval, N, seqs_per_step):
                e = min(N, b + seqs_per_step)
                t = tgt(b, e).reshape(-1, Ht)
                num += (fwd(gm[b:e].to(dev).float()) - t).pow(2).sum().item()
                den += t.pow(2).sum().item()
        return math.sqrt(num / max(den, 1e-30))
    best = (val_err(), {k: v.detach().clone() for k, v in P.items()})
    first = best[0]
    gen = torch.Generator().manual_seed(0)
    for it in range(steps):
        b = int(torch.randint(0, max(1, N - nval - seqs_per_step + 1), (1,), generator=gen))
        e = b + seqs_per_step
        t = tgt(b, e).reshape(-1, Ht)
        loss = (fwd(gm[b:e].to(dev).float()) - t).pow(2).sum() / t.pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % max(1, steps // 8) == 0 or it == steps - 1:
            v = val_err()
            if v < best[0]:
                best = (v, {k: p.detach().clone() for k, p in P.items()})
    Pb, best_err = best[1], best[0]
    del opt, P, best
    fw["mlp.router.weight"] = Pb["rw"]
    fw["mlp.router.bias"] = Pb["rb"]
    gu = gu.float()
    gu[..., ::2] *= Pb["gs"][:, None, :]
    gu[..., 1::2] *= Pb["us"][:, None, :]
    fw["mlp.experts.gate_up_proj"] = gu
    fw["mlp.experts.gate_up_proj_bias"] = Pb["gub"]
    fw["mlp.experts.down_proj"] = None
    fw["mlp.experts.down_proj"] = F.pad(Pb.pop("dn"), (0, sc.h - Ht))
    fw["mlp.experts.down_proj_bias"] = F.pad(Pb["dnb"], (0, sc.h - Ht))
    log(f"    grad refit: held-out FFN err {first:.3f} -> {best_err:.3f}")
    return {"before": first, "after": best_err}


# ----------------------------------------------------------------------------- the driver

def eps_plan(eps, Ht, H, C):
    """The student's rms_norm_eps and the per-stream constant v of the eps channel (residual dim Ht, the
    first padded one), for a teacher with eps `eps`, width Ht, cumulative layer scalars C (C_l at the
    stream entering teacher layer l, C_L at the final norm), and a student of width H.

    The student stream is [h / C_l, v_l, 0, ...] and every student RMSNorm weight is the teacher's
    x sqrt(Ht / H), zero on the padded dims (so v_l never reaches a linear map). Then
        RMSNorm_s = (h / C_l) w sqrt(Ht/H) / sqrt((Ht ms(h) / C_l^2 + v_l^2) / H + eps_s)
    equals the teacher's h w / sqrt(ms(h) + eps) iff  v_l^2 = Ht eps / C_l^2 - H eps_s  (Lean
    rmsNorm_smul_eps: RMSNorm_eps(h / C) = RMSNorm_{eps C^2}(h), so layer l needs eps / C_l^2; and
    hidden_pad_needs_eps: the zero pad needs eps Ht / H). One config eps cannot give a different eps per
    layer; the constant channel can. eps_s = Ht eps / (H max C^2) keeps every v_l^2 >= 0 (it is eps Ht / H,
    the plain pad fold, when no C exceeds 1 -- then v_0 = 0 at C = 1). v_l is written by the embedding
    (v at C_0 = 1) and changed by each layer's down_proj_bias (the top-k weights sum to 1). Without a
    padded dim (H == Ht) there is no channel: eps_s = eps, exact only where C_l = 1.
    Returns (eps_s, [v_l for each entry of C])."""
    C = [float(c) for c in C]
    if H <= Ht:
        return float(eps), None
    eps_s = Ht * eps / (H * max(c * c for c in C))
    return eps_s, [math.sqrt(max(Ht * eps / (c * c) - H * eps_s, 0.0)) for c in C]


class Converter:
    def __init__(self, teacher: Teacher, scfg: SCfg, opts: Opts, imp=None, dkl=None, log=print):
        self.T, self.scfg, self.o, self.imp, self.dkl, self.log = teacher, scfg, opts, imp, dkl, log
        self.tc = teacher.cfg
        self.dev = teacher.device
        self.info = {"layers": []}
        self.limit = None

    def _say(self, *a):
        if self.o.verbose:
            self.log(*a)

    @torch.no_grad()
    def run(self, ids, hs=None, sink=None, done=None):
        """ids: calibration tokens (N, T). sink(i, weights) receives each finished student layer
        (canonical names, CPU); returns (globals, info). done: {i: weights} of already converted
        layers (resume): they are only run, to rebuild the student stream."""
        tc, sc, o, dev = self.tc, self.scfg, self.o, self.dev
        if hs is None:
            hs = stream(self.T, ids, o.bs)
        pi = layer_map(tc, sc, self.dkl, o.layer_select)
        self.info["pi"] = pi
        self._say("layer map", pi)
        scal = [float(self.T.layer(l)["layer_scalar"].float()) if self.T.w is not None else
                float(self.T._get(f"layers.{l}.layer_scalar").float()) for l in range(tc.L)]
        Cum = np.concatenate([[1.0], np.cumprod(scal)])                    # C_l = prod_{j<l} s_j
        self.info["C"] = Cum.tolist()
        sc.rope_theta = o.rope_theta or tc.rope["sliding_attention"]["theta"]
        sc.window = tc.window
        f_s, att = student_inv_freq(sc)
        f_s = f_s.to(dev)
        rope = rope_cs(f_s, ids.shape[1], dev, att)
        H, Ht = sc.h, tc.h
        # eps: the pad needs eps Ht / H, and the layer_scalar fold eps / C_l^2 per layer (eps_plan)
        sc.eps, V = eps_plan(tc.eps, Ht, H, Cum)
        self.info["eps_s"], self.info["eps_v"] = sc.eps, V
        g0 = self.T.glob()
        emb = (g0["embed_tokens.weight"] * torch.tensor(tc.embed_scale, dtype=g0["embed_tokens.weight"].dtype))
        fold = math.sqrt(Ht / H)
        glob = {"embed_tokens.weight": F.pad(emb.float(), (0, H - Ht)).cpu(),
                "norm.weight": F.pad(g0["norm.weight"].float() * fold, (0, H - Ht)).cpu(),
                "lm_head.weight": F.pad(g0["embed_tokens.weight"].float(), (0, H - Ht)).cpu()}
        g = F.pad(torch.cat([F.embedding(ids[i:i + o.bs].to(dev), emb).cpu() for i in range(0, ids.shape[0], o.bs)]).float(), (0, H - Ht))
        if V is not None:                      # the eps channel, at the first kept layer's C (every embedding row)
            glob["embed_tokens.weight"][:, Ht] = V[pi[0]]
            g[..., Ht] = V[pi[0]]
        nb = lambda X: range(0, X.shape[0], o.bs)
        for i, l in enumerate(pi):
            if self.limit is not None and i >= self.limit:
                break
            t_layer0 = time.time()
            nxt = pi[i + 1] if i + 1 < len(pi) else tc.L
            if done and i in done:
                sw = {k: v.to(dev).float() for k, v in done[i].items()}
                sw["_type"] = sc.layer_types[i]
                gm = torch.cat([(g[b:b + o.bs].to(dev) + s_attn(sw, sc, g[b:b + o.bs].to(dev), rope)).cpu() for b in nb(g)])
                g = torch.cat([(gm[b:b + o.bs].to(dev) + s_ffn(sw, sc, gm[b:b + o.bs].to(dev))).cpu() for b in nb(gm)])
                ref = hs[nxt].float() / float(Cum[nxt])
                rel = ((g[..., :Ht] - ref).norm() / ref.norm()).item()
                self._say(f"layer {i:2d} <- teacher {l:2d}: resumed from disk, stream err {rel:.3f}")
                self.info["layers"].append({"i": i, "l": l, "resumed": True, "stream_err": rel})
                del sw, gm
                continue
            for attempt in range(30):          # other jobs share the GPUs: on OOM wait and redo this layer
                try:
                    self._cvlog = {}
                    lw = self.T.layer(l)
                    Cl = float(Cum[l])
                    nxt = pi[i + 1] if i + 1 < len(pi) else tc.L
                    # ---- teacher-stream statistics for layer l
                    st, hmid = self._teacher_stats(lw, l, hs[l], ids)
                    sw = {"input_layernorm.weight": F.pad(lw["input_layernorm.weight"].float() * fold, (0, H - Ht)),
                          "post_attention_layernorm.weight": F.pad(torch.full((Ht,), fold, device=dev), (0, H - Ht)),
                          "_type": sc.layer_types[i]}
                    # ---- attention
                    head_imp = self._head_imp(l, st)
                    pdata = None
                    if o.attn_pairs == "importance":
                        pdata = self._pair_data(lw, l, hs[l], f_s, head_imp)
                    aw, ainfo = attn_convert(lw, tc, sc, l, st, f_s, att, Cl, o, head_imp, pdata)
                    sw.update(aw)
                    if o.attn_refit:
                        self._refit_attn(sw, g, hs[l], hmid, Cl, rope, o.target)
                    if V is not None:              # attention leaves the eps channel alone
                        sw["self_attn.o_proj.weight"][Ht] = 0
                        sw["self_attn.o_proj.bias"][Ht] = 0
                    self._round(sw)
                    gm = torch.cat([(g[b:b + o.bs].to(dev) + s_attn(sw, sc, g[b:b + o.bs].to(dev), rope)).cpu() for b in nb(g)])
                    # ---- FFN
                    fw, finfo = self._ffn(lw, l, st, Cl, gm, hs[nxt], float(Cum[nxt]), hmid)
                    sw.update(fw)
                    if V is not None:              # the FFN moves the eps channel from v(C_l) to v(C_next)
                        sw["mlp.experts.down_proj"][..., Ht] = 0
                        sw["mlp.experts.down_proj_bias"][:, Ht] = V[nxt] - V[l]
                    self._round(sw)
                    gn = torch.cat([(gm[b:b + o.bs].to(dev) + s_ffn(sw, sc, gm[b:b + o.bs].to(dev))).cpu() for b in nb(gm)])
                    # ---- per-layer fidelity. local: student sub-block on the TEACHER's input vs the teacher's
                    # update (the approximation error of this layer alone); stream: student residual after
                    # this layer vs the teacher's residual at the next kept layer (accumulated).
                    P_ = lambda X: F.pad(X.float(), (0, H - Ht))
                    ea = self._rel(lambda b: s_attn(sw, sc, P_(hs[l][b:b + o.bs].to(dev)) / Cl, rope)[..., :Ht],
                                   lambda b: (hmid[b:b + o.bs].to(dev).float() - hs[l][b:b + o.bs].to(dev).float()) / Cl, g.shape[0])
                    C1 = float(Cum[l + 1])
                    ef = self._rel(lambda b: s_ffn(sw, sc, P_(hmid[b:b + o.bs].to(dev)) / Cl)[..., :Ht],
                                   lambda b: hs[l + 1][b:b + o.bs].to(dev).float() / C1 - hmid[b:b + o.bs].to(dev).float() / Cl, g.shape[0])
                    ref = hs[nxt].float() / float(Cum[nxt])
                    gr = gn[..., :Ht]
                    rel = ((gr - ref).norm() / ref.norm()).item()
                    nf = self._nfit(gn.shape[0])
                    rel_ho = ((gr[nf:] - ref[nf:]).norm() / ref[nf:].norm()).item() if nf < gn.shape[0] else float("nan")
                    self._say(f"layer {i:2d} <- teacher {l:2d} ({tc.layer_types[l][:4]}): local attn err {ea:.3f}  "
                              f"local ffn err {ef:.3f}  stream err {rel:.3f} (held-out seqs {rel_ho:.3f})  {finfo.get('summary', '')}  ({time.time() - t_layer0:.0f}s)")
                    for k_, v_ in st.items():
                        if k_.endswith("_cv"):
                            self._cvlog[k_] = v_
                    self.info["layers"].append({"i": i, "l": l, "attn_err": ea, "ffn_err": ef, "stream_err": rel, "stream_err_ho": rel_ho, **ainfo,
                                                **{k: v for k, v in finfo.items() if k != "summary"}, "cv": dict(self._cvlog)})
                    break
                except torch.OutOfMemoryError:
                    self._say(f'layer {i}: CUDA OOM (attempt {attempt}), retrying in 60 s')
                    lw = st = hmid = sw = None
                    import gc; gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(60)
            else:
                raise RuntimeError('gave up after repeated OOM')
            g = gn
            out = {k: v.cpu() for k, v in sw.items() if not k.startswith("_")}
            if sink is not None:
                sink(i, out)
            del lw, st, hmid, sw, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # ---- final: softcap fit (scale of lm_head)
        if o.softcap_fit and tc.softcap:
            glob["lm_head.weight"] *= self._fit_softcap(glob, g, hs[-1])
        return glob, self.info

    # -- statistics on the teacher stream
    def _teacher_stats(self, lw, l, hin, ids):
        tc, o, dev = self.tc, self.o, self.dev
        hd, nkv, keqv = tc.head(l)
        st = {}
        acc = lambda k, v: st.__setitem__(k, st[k] + v if k in st else v)
        hmid = []
        for b in range(0, hin.shape[0], o.bs):
            h = hin[b:b + o.bs].to(dev)
            cap = {}
            ob = t_attn(lw, tc, l, h, cap)
            q, k, v = cap["q"], cap["k"], cap["v"]
            qn = rmsnorm(q, lw["self_attn.q_norm.weight"], tc.eps)
            kn = rmsnorm(k, lw["self_attn.k_norm.weight"], tc.eps)
            zq, zk = q.float() * lw["self_attn.q_norm.weight"].float(), k.float() * lw["self_attn.k_norm.weight"].float()
            acc("aq_n", (qn.float() * zq).sum((0, 1, 3))); acc("aq_d", (zq * zq).sum((0, 1, 3)))
            if self.o.norm_lin == "ls":
                xa = torch.cat([cap["x"].float(), torch.ones_like(cap["x"][..., :1]).float()], -1).reshape(-1, tc.h + 1).double()
                sfx = "_h" if b >= self._nfit(hin.shape[0]) else ""
                acc("XX" + sfx, xa.T @ xa)
                for nm_, t_ in (("QX", qn), ("KX", kn), ("VX", cap["vn"])):
                    t_ = t_.reshape(xa.shape[0], -1).double()
                    acc(nm_ + sfx, t_.T @ xa)
                    if sfx:
                        acc(nm_ + "yy", (t_ * t_).sum())
            acc("ak_n", (kn.float() * zk).sum((0, 1, 3))); acc("ak_d", (zk * zk).sum((0, 1, 3)))
            vn = cap["vn"].float()
            acc("av_n", (vn * v.float()).sum((0, 1, 3))); acc("av_d", (v.float() ** 2).sum((0, 1, 3)))
            a = cap["a"].float().view(h.shape[0], h.shape[1], tc.nq, hd)
            acc("C", torch.einsum("btnd,btne->nde", a, a))
            on = rmsnorm(ob, lw["post_attention_layernorm.weight"], tc.eps).float()
            acc("beta_n", (on * ob.float()).sum((0, 1))); acc("beta_d", (ob.float() ** 2).sum((0, 1)))
            acc("head_en", (a ** 2).sum((0, 1, 3)))
            acc("irms_o", (ob.float().pow(2).mean(-1) + tc.eps).rsqrt().sum())
            hm = h + on.to(h.dtype)
            hmid.append(hm.cpu())
            # FFN stats
            capf = {}
            t_ffn(lw, tc, hm, capf)
            f, m1, m2 = capf["f"].float(), capf["m1"].float(), capf["m2"].float()
            for nm_, z in (("11", m1 * m1), ("22", m2 * m2), ("12", m1 * m2), ("f1", f * m1), ("f2", f * m2)):
                acc("S" + nm_, z.sum((0, 1)))
            ir = lambda z: (z.float().pow(2).mean(-1) + tc.eps).rsqrt().sum()
            acc("irms_1", ir(m1)); acc("irms_2", ir(m2)); acc("irms_s", ir(capf["m1n"] + capf["m2n"]))
            self._unit_stats(lw, l, hm, capf, acc)
        n = hin.shape[0] * hin.shape[1]
        st["aq"] = st["aq_n"] / st["aq_d"].clamp_min(1e-12)
        st["ak"] = st["ak_n"] / st["ak_d"].clamp_min(1e-12)
        st["av"] = st["av_n"] / st["av_d"].clamp_min(1e-12)
        if "XX" in st:
            for k_, n_ in (("QX", "Qfit"), ("KX", "Kfit"), ("VX", "Vfit")):
                if "XX_h" in st:
                    W, r, e = _cv_lsq(st["XX"], st[k_].T, st["XX_h"], st[k_ + "_h"].T, st[k_ + "yy"], o.ridges)
                    st[n_ + "_cv"] = (r, e)
                else:
                    W = _lsq_aug(st["XX"], st[k_].T, 1e-3)
                st[n_] = W.T.float()
            for k_ in [k for k in st if k[:2] in ("XX", "QX", "KX", "VX")]:
                del st[k_]
        st["beta"] = st["beta_n"] / st["beta_d"].clamp_min(1e-12)
        st["C"] = st["C"] / n
        det = st["S11"] * st["S22"] - st["S12"] ** 2
        ok = det.abs() > 1e-12 * (st["S11"] * st["S22"]).clamp_min(1e-30)
        a_ = torch.where(ok, (st["Sf1"] * st["S22"] - st["Sf2"] * st["S12"]) / det.where(ok, torch.ones_like(det)), st["Sf1"] / st["S11"].clamp_min(1e-12))
        b_ = torch.where(ok, (st["Sf2"] * st["S11"] - st["Sf1"] * st["S12"]) / det.where(ok, torch.ones_like(det)), st["Sf2"] / st["S22"].clamp_min(1e-12))
        for k in ("irms_o", "irms_1", "irms_2", "irms_s"):
            st[k] = st[k] / n
        if self.o.norm_lin == "ls":           # per-channel LS of f on (m1, m2)
            st["a_out"], st["b_out"] = a_, b_
        else:                                 # one mean 1/rms scalar per norm
            wf = lw["post_feedforward_layernorm.weight"].float() * st["irms_s"]
            st["a_out"] = wf * lw["post_feedforward_layernorm_1.weight"].float() * st["irms_1"]
            st["b_out"] = wf * lw["post_feedforward_layernorm_2.weight"].float() * st["irms_2"]
        grid = torch.tensor((1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3), device=dev)
        st["c_e"] = grid[st["c_e_err"].argmin(0)]
        st["c_d"] = grid[st["c_d_err"].argmin(0)]
        st["unit_en"] = st["unit_en"] / n
        st["dense_en"] = st["dense_en"] / n
        for k in ("freq", "mass", "coact"):
            st[k] = st[k] / n
        return st, torch.cat(hmid)

    def _unit_stats(self, lw, l, hm, capf, acc):
        tc, dev = self.tc, self.dev
        H = tc.h
        rf = hm.reshape(-1, H)
        ti, tw = capf["ti"], capf["tw"]
        x2 = rmsnorm(rf, lw["pre_feedforward_layernorm_2.weight"], tc.eps)
        gu, dn = lw["experts.gate_up_proj"], lw["experts.down_proj"]
        E, Ie = tc.E, tc.Ie
        en = torch.zeros(E, Ie, device=dev); umax = torch.zeros(E, Ie, device=dev)
        cerr = torch.zeros(9, E, Ie, device=dev)
        dnorm = dn.float().pow(2).sum(1)                                         # (E, Ie)
        for e in ti.unique().tolist():
            tok, kk = (ti == e).nonzero(as_tuple=True)
            gg, uu = (x2[tok] @ gu[e].T).float().chunk(2, -1)
            act = gelu_tanh(gg) * uu
            wv = tw[tok, kk].float()[:, None]
            en[e] = ((act * wv) ** 2).sum(0) * dnorm[e]
            umax[e] = uu.abs().amax(0)
            cerr[:, e] = fit_act(gg, uu * wv)
        acc("unit_en", en); acc("c_e_err", cerr)
        # dense units
        xd = rmsnorm(hm, lw["pre_feedforward_layernorm.weight"], tc.eps).reshape(-1, H)
        gd = (xd @ lw["mlp.gate_proj.weight"].T).float()
        ud = (xd @ lw["mlp.up_proj.weight"].T).float()
        dd = lw["mlp.down_proj.weight"].float().pow(2).sum(0)
        acc("dense_en", ((gelu_tanh(gd) * ud) ** 2).sum(0) * dd)
        acc("c_d_err", fit_act(gd, ud))
        # running max (not additive): stash in a side dict
        self._maxes = getattr(self, "_maxes", {})
        key = l
        if key not in self._maxes:
            self._maxes[key] = [umax, ud.abs().amax(0)]
        else:
            self._maxes[key][0] = torch.maximum(self._maxes[key][0], umax)
            self._maxes[key][1] = torch.maximum(self._maxes[key][1], ud.abs().amax(0))
        oh = torch.zeros(ti.shape[0], E, device=dev).scatter_(1, ti, (tw / tw.sum(-1, keepdim=True)).float())
        acc("freq", (oh > 0).float().sum(0)); acc("mass", oh.sum(0)); acc("coact", oh.T @ oh)

    def _nfit(self, N):
        """calibration sequences [0, nfit) fit the LS refits, [nfit, N) are held out."""
        return N - int(round(N * self.o.cv_frac)) if self.o.cv_frac > 0 and N > 1 else N

    def _round(self, sw):
        """round to the storage dtype (the stream then sees exactly the weights that get saved)."""
        dt = getattr(torch, self.o.store_dtype)
        for k, v in sw.items():
            if torch.is_tensor(v) and v.is_floating_point():
                sw[k] = v.to(dt).float()

    def _head_imp(self, l, st):
        if self.o.importance == "attr" and self.imp is not None:
            return self.imp["head"][l].detach().float()
        return st["head_en"].detach().float().cpu()

    def _pair_data(self, lw, l, hin, f_s, head_imp):
        tc, o, dev = self.tc, self.o, self.dev
        hd, nkv, _ = tc.head(l)
        h = hin[: o.pair_seqs].to(dev)
        cap = {}
        t_attn(lw, tc, l, h, cap)
        qn = rmsnorm(cap["q"], lw["self_attn.q_norm.weight"], tc.eps)
        kn = rmsnorm(cap["k"], lw["self_attn.k_norm.weight"], tc.eps)
        f_t = teacher_inv_freq(tc, tc.layer_types[l]).to(dev)
        win = tc.window if tc.layer_types[l] == "sliding_attention" else None
        w = head_imp.to(dev).float().clamp_min(0)
        w = w / w.mean().clamp_min(1e-12)
        return pair_stats(qn, kn, f_t, f_s, win, tc.nq // nkv, w, nseq=o.pair_seqs, M=o.pair_queries)

    def _attn_feats(self, sw, gb, rope):
        cap = {}
        s_attn(sw, self.scfg, gb, rope, cap)
        return cap["a"].float()

    def _refit_attn(self, sw, g, hin, hmid, Cl, rope, target):
        """LS: o_proj (+bias) on the student's per-head outputs -> teacher attention update (residual
        mode: teacher h_mid / C - student g, i.e. also correcting the upstream drift)."""
        sc, o, dev, Ht = self.scfg, self.o, self.dev, self.tc.h
        D = sc.nq * sc.hd + 1
        z = lambda *s_: torch.zeros(*s_, dtype=torch.float64, device=dev)
        G, Bm, Gh, Bh, yyh = z(D, D), z(D, Ht), z(D, D), z(D, Ht), z(())
        nfit = self._nfit(g.shape[0])
        for b in range(0, g.shape[0], o.bs):
            gb = g[b:b + o.bs].to(dev)
            a = self._attn_feats(sw, gb, rope)
            phi = torch.cat([a, torch.ones_like(a[..., :1])], -1).reshape(-1, D).double()
            if target == "residual":
                y = hmid[b:b + o.bs].to(dev).float() / Cl - gb[..., :Ht]
            else:
                y = (hmid[b:b + o.bs].to(dev).float() - hin[b:b + o.bs].to(dev).float()) / Cl
            y = y.reshape(-1, Ht).double()
            if b >= nfit:
                Gh += phi.T @ phi; Bh += phi.T @ y; yyh += (y * y).sum()
            else:
                G += phi.T @ phi; Bm += phi.T @ y
        prior = torch.cat([sw["self_attn.o_proj.weight"][:Ht].T, sw["self_attn.o_proj.bias"][None, :Ht]]).double()
        if nfit < g.shape[0]:
            W, r, e = _cv_lsq(G, Bm, Gh, Bh, yyh, o.ridges, prior, keep_prior=o.cv_keep_prior)
            self._cvlog["o_proj"] = (r, e)
            W = W.float()
        else:
            W = _lsq_aug(G, Bm, o.ridge, prior).float()
        sw["self_attn.o_proj.weight"] = F.pad(W[:-1].T, (0, 0, 0, sc.h - Ht))
        sw["self_attn.o_proj.bias"] = F.pad(W[-1], (0, sc.h - Ht))

    def _rel(self, f_s, f_t, N):
        num = den = 0.0
        for b in range(0, N, self.o.bs):
            ys, yt = f_s(b).float(), f_t(b).float()
            num += (ys - yt).pow(2).sum().item()
            den += yt.pow(2).sum().item()
        return math.sqrt(num / max(den, 1e-30))

    # -- FFN conversion + refits
    def _ffn(self, lw, l, st, Cl, gm, hnext, Cn, hmid):
        tc, sc, o, dev = self.tc, self.scfg, self.o, self.dev
        mx = self._maxes.pop(l)
        st["r_e"], st["r_d"] = mx[0] * 1.15, mx[1] * 1.15
        if o.importance == "attr" and self.imp is not None:
            uimp, dimp = self.imp["unit"][l].float(), self.imp["dense"][l].float()
            eimp = self.imp["expert"][l].float()
            coact = self.imp["coact"][l]
        else:
            uimp, dimp = st["unit_en"].cpu(), st["dense_en"].cpu()
            eimp = uimp.sum(1)
            coact = st["coact"].cpu()
        usage = st["mass"].cpu()
        drop = torch.argsort(eimp)[: o.drop_experts].tolist() if o.drop_experts else None
        groups = group_experts(coact, usage, sc.E, o.ffn_group, drop)
        plan, dense_units = unit_plan(groups, uimp, dimp, sc.I, tc.Ie, tc.Id, o.unit_select, o.ffn_dense)
        fw = ffn_closed(lw, tc, sc, groups, plan, dense_units, st, Cl, o)
        info = {"groups": groups, "dense_units": len(dense_units)}
        # routing-mass recall of the grouping on the teacher stream
        M = torch.zeros(tc.E, sc.E)
        for E_, gs in enumerate(groups):
            M[gs, E_] = 1
        if o.ffn_refit:
            self._refit_ffn(fw, lw, l, groups, gm, hnext, Cl, Cn, dense_units, st, hmid)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if o.grad_steps:
            Ht = tc.h
            if o.target == "residual":
                tgt = lambda b, e: hnext[b:e].to(dev).float() / Cn - gm[b:e, ..., :Ht].to(dev).float()
            else:
                tgt = lambda b, e: t_ffn(lw, tc, hmid[b:e].to(dev)).float() / Cl
            big = ("experts.gate_up_proj", "experts.down_proj")
            for k in big:                                   # make room: teacher experts to the CPU meanwhile
                lw[k] = lw[k].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            info["grad_refit"] = grad_refit_ffn(fw, sc, gm, tgt, o.grad_steps, o.grad_lr, dev, log=self._say,
                                                nval=gm.shape[0] - self._nfit(gm.shape[0]))
            for k in big:
                lw[k] = lw[k].to(dev)
        rec = self._recall(fw, lw, gm, M.to(dev))
        info["mass_recall"] = rec
        info["summary"] = f"mass recall {rec:.3f} dense units {len(dense_units)}"
        return fw, info

    def _teacher_ffn_at(self, lw, l, x, st, Cl):
        """teacher FFN (linear surrogate of its output norms) evaluated at student FFN input x (N, H):
        returns dense part (N,Ht), per-token top-k idx, weights, and expert outputs (N,k,Ht)."""
        tc = self.tc
        Ht = tc.h
        n = rms(x[:, :Ht], tc.eps)          # x[:Ht] is already the unit-rms teacher residual (fold)
        w1 = lw["pre_feedforward_layernorm.weight"].float()
        w2 = lw["pre_feedforward_layernorm_2.weight"].float()
        nd = n.to(lw["mlp.gate_proj.weight"].dtype)
        dense = t_dense(lw, tc, (n * w1).to(nd.dtype)).float() * st["a_out"] / Cl
        logits, p, tw, ti = t_router(lw, tc, nd)
        ef = t_expert_fn(lw, tc)
        x2 = (n * w2).to(nd.dtype)
        outs = torch.zeros(n.shape[0], tc.topk, Ht, device=n.device)
        for e in ti.unique().tolist():
            tok, kk = (ti == e).nonzero(as_tuple=True)
            outs[tok, kk] = ef(e, x2[tok]).float() * st["b_out"] / Cl
        return dense, ti, tw, outs, logits

    def _refit_ffn(self, fw, lw, l, groups, gm, hnext, Cl, Cn, dense_units, st, hmid):
        tc, sc, o, dev = self.tc, self.scfg, self.o, self.dev
        H, Ht, nE, Is = sc.h, tc.h, sc.E, sc.I
        gidx = torch.full((tc.E,), -1, dtype=torch.long, device=dev)
        for E_, gs in enumerate(groups):
            gidx[gs] = E_
        xs = []
        fw_post = F.pad(torch.full((Ht,), math.sqrt(Ht / H), device=dev), (0, H - Ht))
        for b in range(0, gm.shape[0], o.bs):
            xs.append(rmsnorm(gm[b:b + o.bs].to(dev), fw_post, sc.eps).reshape(-1, H).cpu())
        X = torch.cat(xs)
        N = X.shape[0]
        # 1) router: weighted LS of each group's logit onto log-sum-exp of its members' teacher logits
        Tq = gm.shape[1]
        ntok_fit = self._nfit(gm.shape[0]) * Tq
        if o.ffn_router == "lse":
            zf = lambda *s_: torch.zeros(*s_, dtype=torch.float32, device=dev)
            Gs = [zf(Ht + 1, Ht + 1) for _ in range(nE)]
            Bs = [zf(Ht + 1) for _ in range(nE)]
            Gsh = [zf(Ht + 1, Ht + 1) for _ in range(nE)] if ntok_fit < N else None
            Bsh = [zf(Ht + 1) for _ in range(nE)] if ntok_fit < N else None
            yys = [0.0] * nE
            for b, e_, ho in _chunks(N, ntok_fit, 2048):
                x = X[b:e_].to(dev)
                n = rms(x[:, :Ht], tc.eps).to(lw["router.proj.weight"].dtype)
                logits, p, _, _ = t_router(lw, tc, n)
                phi = torch.cat([x[:, :Ht].float(), torch.ones(x.shape[0], 1, device=dev)], -1)
                for E_, gs in enumerate(groups):
                    if not gs:
                        continue
                    z = torch.logsumexp(logits[:, gs], -1).float()
                    wgt = (1.0 + nE * p[:, gs].sum(-1)).float()
                    if ho:
                        Gsh[E_] += (phi * wgt[:, None]).T @ phi
                        Bsh[E_] += (phi * wgt[:, None]).T @ z
                        yys[E_] += float((wgt * z * z).sum())
                    else:
                        Gs[E_] += (phi * wgt[:, None]).T @ phi
                        Bs[E_] += (phi * wgt[:, None]).T @ z
            errs = []
            for E_, gs in enumerate(groups):
                if not gs:
                    continue
                prior = torch.cat([fw["mlp.router.weight"][E_, :Ht], fw["mlp.router.bias"][E_:E_ + 1]]).double()
                if Gsh is not None:
                    wv, r, e = _cv_lsq(Gs[E_].double(), Bs[E_][:, None].double(), Gsh[E_].double(), Bsh[E_][:, None].double(),
                                       torch.tensor(yys[E_], dtype=torch.float64, device=dev), o.ridges, prior[:, None])
                    wv = wv[:, 0].float()
                    errs.append(r)
                else:
                    wv = _lsq_aug(Gs[E_].double(), Bs[E_][:, None].double(), o.ridge, prior[:, None])[:, 0].float()
                fw["mlp.router.weight"][E_, :Ht], fw["mlp.router.bias"][E_] = wv[:-1], wv[-1]
            del Gs, Bs, Gsh, Bsh
            if errs:
                self._cvlog["router_ridge_median"] = float(np.median(errs))
        # 2) per-expert down_proj LS: student expert E must produce t_E = its members' share of the
        #    teacher FFN + w_E * (dense + mass the student's top-k groups missed)
        Gd = {}
        from approx.arch import s_router
        gu, gub = fw["mlp.experts.gate_up_proj"], fw["mlp.experts.gate_up_proj_bias"]
        dn = fw["mlp.experts.down_proj"]
        GE, BE, GEh, BEh, yyE = [None] * nE, [None] * nE, [None] * nE, [None] * nE, [0.0] * nE
        for b, e_, ho in (_chunks(N, ntok_fit, 1024) if o.ffn_down_ls else []):
            x = X[b:e_].to(dev)
            dense, ti, tw, outs, _ = self._teacher_ffn_at(lw, l, x, st, Cl)
            _, sw_, si = s_router(fw, sc, x)
            grp_t = gidx[ti]                                                       # (n, kT)
            sel = (grp_t[:, :, None] == si[:, None, :])                            # (n, kT, kS)
            covered = sel.any(-1)
            lost = (outs * (~covered)[..., None]).sum(1)                           # (n, Ht)
            for kk in range(sc.topk):
                E_all = si[:, kk]
                share = (outs * sel[:, :, kk, None]).sum(1)                        # members of the chosen group
                tE = share + sw_[:, kk, None] * (dense + lost)
                for E_ in E_all.unique().tolist():
                    m = E_all == E_
                    xe = x[m]
                    z = xe @ gu[E_] + gub[E_]
                    act = swiglu_oai(z[:, ::2], z[:, 1::2], sc.limit, sc.alpha)
                    phi = torch.cat([act, torch.ones_like(act[:, :1])], -1) * sw_[m, kk, None]
                    phi = phi.float()
                    yv = tE[m].float()
                    G_, B_ = (GEh, BEh) if ho else (GE, BE)
                    if G_[E_] is None:                  # fp32 accumulation (memory: 32 experts), fp64 solve
                        G_[E_] = torch.zeros(Is + 1, Is + 1, dtype=torch.float32, device=dev)
                        B_[E_] = torch.zeros(Is + 1, Ht, dtype=torch.float32, device=dev)
                    G_[E_] += phi.T @ phi
                    B_[E_] += phi.T @ yv
                    if ho:
                        yyE[E_] += float((yv * yv).sum())
        rid = []
        for E_ in range(nE):
            if GE[E_] is None:
                continue
            prior = torch.cat([dn[E_, :, :Ht], fw["mlp.experts.down_proj_bias"][E_:E_ + 1, :Ht]]).double()
            if GEh[E_] is not None:
                W, r, e = _cv_lsq(GE[E_].double(), BE[E_].double(), GEh[E_].double(), BEh[E_].double(),
                                  torch.tensor(yyE[E_], dtype=torch.float64, device=dev), o.ridges, prior)
                rid.append(r)
                W = W.float()
            else:
                W = _lsq_aug(GE[E_].double(), BE[E_].double(), max(o.ridge, 3e-2), prior).float()
            GE[E_] = BE[E_] = GEh[E_] = BEh[E_] = None
            dn[E_, :, :Ht], fw["mlp.experts.down_proj_bias"][E_, :Ht] = W[:-1], W[-1]
        if rid:
            self._cvlog["down_ridge_median"] = float(np.median(rid))
        # 3) per-channel gain + bias of the whole FFN update against the (residual) target
        S = {k: torch.zeros(Ht, dtype=torch.float64, device=dev) for k in ("yy", "y", "ty", "t", "n")}
        for b in range(0, self._nfit(gm.shape[0]), o.bs):
            gb = gm[b:b + o.bs].to(dev)
            y = s_ffn(fw, sc, gb)[..., :Ht].reshape(-1, Ht).double()
            if o.target == "residual":
                t = (hnext[b:b + o.bs].to(dev).float() / Cn - gb[..., :Ht]).reshape(-1, Ht).double()
            else:   # local: the teacher's own FFN update on its own stream
                hm_ = hmid[b:b + o.bs].to(dev)
                t = (t_ffn(lw, tc, hm_).float() / Cl).reshape(-1, Ht).double()
                y = s_ffn(fw, sc, F.pad(hm_.float() / Cl, (0, H - Ht)))[..., :Ht].reshape(-1, Ht).double()
            S["yy"] += (y * y).sum(0); S["y"] += y.sum(0); S["ty"] += (t * y).sum(0); S["t"] += t.sum(0)
            S["n"] += y.shape[0]
        n_ = S["n"]
        var = S["yy"] - S["y"] ** 2 / n_
        cov = S["ty"] - S["t"] * S["y"] / n_
        gam = torch.where(var > 1e-12, cov / var.clamp_min(1e-30), torch.ones_like(var)).clamp(0.25, 4.0)
        bet = (S["t"] - gam * S["y"]) / n_
        gam, bet = gam.float(), bet.float()
        dn[:, :, :Ht] *= gam
        fw["mlp.experts.down_proj_bias"][:, :Ht] = fw["mlp.experts.down_proj_bias"][:, :Ht] * gam + bet

    def _recall(self, fw, lw, gm, M):
        """fraction of the teacher's (top-k, renormalized) routing mass whose teacher expert sits in
        one of the student's chosen experts, on the student stream."""
        from approx.arch import s_router
        tc, sc, o, dev = self.tc, self.scfg, self.o, self.dev
        H, Ht = sc.h, tc.h
        tot = got = 0.0
        for b in range(0, gm.shape[0], o.bs):
            x = rmsnorm(gm[b:b + o.bs].to(dev), F.pad(torch.full((Ht,), math.sqrt(Ht / H), device=dev), (0, H - Ht)), sc.eps).reshape(-1, H)
            n = rms(x[:, :Ht], tc.eps).to(lw["router.proj.weight"].dtype)
            _, p, tw, ti = t_router(lw, tc, n)
            twn = tw / tw.sum(-1, keepdim=True)
            _, _, si = s_router(fw, sc, x)
            grp = M[ti].argmax(-1)                                                   # (n, kT)
            ok = (grp[:, :, None] == si[:, None, :]).any(-1).float()
            got += (twn * ok).sum().item()
            tot += twn.sum().item()
        return got / max(tot, 1e-12)

    def _fit_softcap(self, glob, g, hT):
        """scale s of lm_head minimizing KL(teacher (softcapped) || student) on the calibration set."""
        tc, sc, dev = self.tc, self.scfg, self.dev
        gg = {k: v.to(dev) for k, v in glob.items()}
        tg = self.T.glob()
        idx = torch.randperm(g.shape[0] * g.shape[1], generator=torch.Generator().manual_seed(0))[:2048]
        hs_ = g.reshape(-1, g.shape[-1])[idx]
        ht = hT.reshape(-1, hT.shape[-1])[idx]
        grid = np.linspace(0.2, 1.4, 31)
        kl = np.zeros(len(grid))
        for c in range(0, idx.numel(), 256):
            lt = t_logits(tg, tc, ht[c:c + 256].to(dev)).log_softmax(-1)
            z = s_logits(gg, sc, hs_[c:c + 256].to(dev))
            for j, s_ in enumerate(grid):
                kl[j] += kl_rows(lt, (z * float(s_)).log_softmax(-1)).sum().item()
            del lt, z
        j = int(kl.argmin())
        best = (kl[j] / idx.numel(), float(grid[j]))
        del gg
        self._say(f"softcap fit: lm_head scale {best[1]:.3f} (KL {best[0]:.4f})")
        self.info["lm_scale"] = best[1]
        return best[1]
