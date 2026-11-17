"""fr-resnorm: catalog instances of the residual-stream and norm rules this branch adds (dist/fr_resnorm.md), on the
reference decoder (verify/spot/catalog/_ref.py, fp64). Each builds the host from the target's weights and runs both on
the same token ids --

  hyper-connection hosts switched off (FEATURES hyper_conn / hc_<variant> host sides free for the four stock variants
    dsv4, glm5_next, hy4, qwen4_exp): every stream carries the target's residual (Lean EP.hc_identical_streams,
    EP.hc_stack); the dsv4 / glm5_next comb needs hc_eps = 0 (a config value) -- the finding fr_hc_dsv4_eps shows
    the shrink at hc_eps > 0;
  the LayerNorm-beta fold WITHOUT an lm_head bias by a dominated final readout (port ln_beta_readout): one extra host
    layer whose MLP (a constant gate from its gate bias) writes M [gamma x_hat + beta; kappa] from its pre-norm, so the
    final norm divides by ~ M kappa / sqrt(D), a constant (Lean EP.dominated_readout_limit);
  DiffusionGemma's always-on embedding norm switched off (FEATURES selfcond_norm host side limit): the residual
    scaled by alpha with host eps = alpha^2 eps, the embedding rows lambda_v e_v, lambda_v = alpha^2 sqrt(eps) /
    sqrt(1 - alpha^2 ms(e_v)) (Lean EP.selfcond_norm_row, EP.rmsNorm_smul_eps_sq): exact on an untied host, a limit
    (alpha -> 0) on a tied one."""
from __future__ import annotations

import math

import torch

from verify.spot.catalog import entry
from verify.spot.catalog._ref import F64, RefLM
from verify.spot.catalog.ref_slots import REF, _check, _host, _t

N_HC = 4


def _logit(q):
    return math.log(q / (1 - q))


def _hc_host(t, var, hc_eps, seed=5):
    """A plain target's host with N_HC streams of variant var: the target's weights, the HC weights random, then the
    construction (static mixes with the bases below)."""
    h = _host(t, hc=(N_HC, var), hc_eps=hc_eps)
    tmp = RefLM(**h.c).init(seed)
    for k, v in tmp.p.items():
        if k not in h.p:
            h.p[k] = v
    n, e = N_HC, hc_eps
    sites = [f"l{i}.hc{k}" for i in range(t.c["L"]) for k in (1, 2)]
    for s in sites:
        if var == "qwen4":
            g = t.p[s.replace(".hc", ".n") + "_w"]                  # the target pre-norm gain of this sublayer
            h.p[s + "_up"] = torch.zeros_like(h.p[s + "_up"])       # gate sigmoid(0) = 1/2 ...
            h.p[s + "_nw"] = (2 * g - 1).repeat(n)                  # ... undone by the (1 + w) gain 2 gamma
            h.p[s + "_inj"] = torch.zeros_like(h.p[s + "_inj"])     # injection 2 sigmoid(0) = 1
            continue
        h.p[s + "_fn"] = torch.zeros_like(h.p[s + "_fn"])
        b = torch.zeros_like(h.p[s + "_base"])
        b[:n] = _logit(1.0 / n - e)                                 # sum_i pre_i = 1
        b[n:2 * n] = _logit((1.0 - e) / h.c["hc_mag"]) if var == "hy4" else 0.0   # post_i = 1
        h.p[s + "_base"] = b                                        # comb logits constant (0)
    if var == "qwen4":
        h.p["hch_up"] = torch.zeros_like(h.p["hch_up"])
        h.p["hch_nw"] = (2 * t.p["nf_w"] - 1).repeat(n)
        h.p.pop("nf_w", None)
    elif var in ("dsv4", "hy4"):
        h.p["hch_fn"] = torch.zeros_like(h.p["hch_fn"])
        h.p["hch_base"] = torch.full_like(h.p["hch_base"], _logit(1.0 / n - e))
    return h


@entry("port:hc_off", "feature:hyper_conn:host", "feature:hc_dsv4:host",
       instance=f"plain pre-norm decoder -> a DeepSeek-V4 mHC host (4 streams, Sinkhorn comb, learned HyperHead): fn = 0, "
                f"pre 1/4, post 1, constant comb logits, hc_eps = 0 (config): every stream the target's residual ({REF})")
def fr_hc_dsv4_off():
    t = _t()
    return _check(t, _hc_host(t, "dsv4", 0.0))


@entry("feature:hc_dsv4:host",
       instance=f"the same DeepSeek-V4 host keeping hc_eps = 1e-6: the Sinkhorn rows sum to n c / (n c + hc_eps) < 1, "
                f"the streams shrink each layer ({REF})",
       finding="hc_eps > 0 in the Sinkhorn normalisers makes the identical streams decay (comb row sums < 1): the "
               "switch-off needs hc_eps = 0, a shape-preserving config value")
def fr_hc_dsv4_eps():
    t = _t()
    return _check(t, _hc_host(t, "dsv4", 1e-3))


@entry("port:hc_off", "feature:hc_glm5_next:host",
       instance=f"plain -> a GLM-5-Next mHC host (comb^T, mean collapse): fn = 0, pre 1/4, post 1, hc_eps = 0 ({REF})")
def fr_hc_glm5_next_off():
    t = _t()
    return _check(t, _hc_host(t, "glm5_next", 0.0))


@entry("port:hc_off", "feature:hc_hy4:host",
       instance=f"plain -> a Hy4 HC host (pre / post only, identity carry, hc_eps = 1e-6 kept, hc_magnitude 2): fn = 0, "
                f"sigmoid(pre) = 1/4 - hc_eps, sigmoid(post) = (1 - hc_eps) / 2 ({REF})")
def fr_hc_hy4_off():
    t = _t()
    return _check(t, _hc_host(t, "hy4", 1e-6))


@entry("port:hc_off", "feature:hc_qwen4_exp:host",
       instance=f"plain -> a Qwen4-exp gated residual host (per-stream (1 + w) RMSNorm, low-rank sigmoid mix, injection "
                f"2 sigmoid): mix up-projection 0 (gate 1/2), norm gain 2 gamma, injection weights 0 ({REF})")
def fr_hc_qwen4_off():
    t = _t()
    return _check(t, _hc_host(t, "qwen4", 1e-6))


# ------------------------------------------------------------------ the LayerNorm-beta fold, dominated final readout

def ln_beta_readout(M=2.0 ** 26, kappa=2.0 ** 20, c0=20.0, seed=0):
    """(target, host): a LayerNorm target with betas (biased q / k / v, o, MLP) on an RMSNorm host WITHOUT an lm_head
    bias, one layer and one channel more. Layers 0 .. L-1: the exact fold (writers centred, betas into the
    consumers' biases, the hidden pad's eps d / D and gains sqrt(d / D)). Layer L: attention off (o = 0); the MLP's
    gate is the constant SiLU(c0) (gate weights 0, gate bias c0), its up-projection the identity, so it writes the
    affine M (gamma_f x_hat + beta_f) of its pre-norm output x_hat = RMS(P x) into the stream and M kappa into the
    extra channel (down bias). The final norm then sees M [y; kappa] + P x: as M -> oo and kappa / |y| -> oo it
    divides by M kappa / sqrt(D), and lm_head = (kappa / sqrt(D)) [U; 0] gives y . U, the target's logits."""
    t = _t(seed, norm="ln", qkv_bias=True, out_bias=True)
    tc = t.c
    d, L, Fw, eps = tc["d"], tc["L"], tc["F"], tc["eps"]
    D = d + 1
    h = RefLM(**{**tc, "norm": "rms", "d": D, "L": L + 1, "F": max(Fw, d), "eps": eps * d / D, "lm_bias": False})
    h.init(seed + 1)
    P = {k: torch.zeros_like(v) if v is not None else None for k, v in h.p.items()}
    pad = math.sqrt(d / D)
    cen = lambda W: W - W.mean(0, keepdim=True)
    padr = lambda W: torch.cat([W, torch.zeros(1, *W.shape[1:], dtype=F64)], 0)       # an extra output row
    padc = lambda W: torch.cat([W, torch.zeros(W.shape[0], 1, dtype=F64)], 1)         # an extra input column
    padv = lambda v: torch.cat([v, torch.zeros(1, dtype=F64)])
    padF = lambda W, rows: torch.cat([W, torch.zeros(rows - W.shape[0], *W.shape[1:], dtype=F64)], 0)
    P["emb"] = padc(t.p["emb"] - t.p["emb"].mean(1, keepdim=True))
    for i in range(L):
        pre = f"l{i}."
        g1, b1, g2, b2 = (t.p[pre + k] for k in ("n1_w", "n1_b", "n2_w", "n2_b"))
        P[pre + "n1_w"] = torch.full((D,), pad, dtype=F64)
        P[pre + "n2_w"] = torch.full((D,), pad, dtype=F64)
        for key in ("q", "k", "v"):
            P[pre + key] = padc(t.p[pre + key] * g1)
            P[pre + key + "b"] = t.p[pre + key + "b"] + t.p[pre + key] @ b1
        P[pre + "o"] = padr(cen(t.p[pre + "o"]))
        P[pre + "ob"] = padv(t.p[pre + "ob"] - t.p[pre + "ob"].mean())
        Fh = P[pre + "up"].shape[0]
        P[pre + "up"] = padF(padc(t.p[pre + "up"] * g2), Fh)
        P[pre + "fgate"] = padF(padc(t.p[pre + "fgate"] * g2), Fh)
        P[pre + "upb"] = padF(t.p[pre + "upb"] + t.p[pre + "up"] @ b2, Fh)
        P[pre + "fgateb"] = padF(t.p[pre + "fgate"] @ b2, Fh)
        down = torch.zeros(D, Fh, dtype=F64)
        down[:d, :Fw] = cen(t.p[pre + "down"])
        P[pre + "down"] = down
        P[pre + "downb"] = padv(t.p[pre + "downb"] - t.p[pre + "downb"].mean())
    pre = f"l{L}."                                                   # the readout layer
    P[pre + "n1_w"] = torch.full((D,), pad, dtype=F64)
    P[pre + "n2_w"] = torch.full((D,), pad, dtype=F64)
    s = c0 * torch.sigmoid(torch.tensor(c0, dtype=F64)).item()      # SiLU(c0)
    Fh = P[pre + "up"].shape[0]
    up = torch.zeros(Fh, D, dtype=F64)
    up[:d, :d] = torch.eye(d, dtype=F64)
    P[pre + "up"], P[pre + "upb"] = up, torch.zeros(Fh, dtype=F64)
    P[pre + "fgate"], P[pre + "fgateb"] = torch.zeros(Fh, D, dtype=F64), torch.full((Fh,), c0, dtype=F64)
    down = torch.zeros(D, Fh, dtype=F64)
    down[:d, :d] = torch.diag(t.p["nf_w"]) * (M / s)
    P[pre + "down"] = down
    P[pre + "downb"] = torch.cat([t.p["nf_b"] * M, torch.tensor([M * kappa], dtype=F64)])
    P["nf_w"] = torch.ones(D, dtype=F64)
    P["lm"] = padc(t.p["lm"]) * (kappa / math.sqrt(D))
    h.p = P
    return t, h


@entry("port:ln_beta_readout",
       instance=f"LayerNorm with betas (biased q / k / v, o, MLP) -> an RMSNorm host WITHOUT an lm_head bias, +1 layer "
                f"+1 channel: the exact fold below, the final LayerNorm read out by a constant-gate MLP at scale M = 2^26 "
                f"beside a pad kappa = 2^20 that fixes the final norm's divisor ({REF})", standard="limit")
def fr_ln_beta_readout():
    t, h = ln_beta_readout()
    return _check(t, h, standard="limit", tol=1e-7, note="limit M, kappa -> oo: rel ~ R/M + |y|^2/(2 kappa^2)")


# ------------------------------------------------------------------ DiffusionGemma's embedding norm switched off

def selfcond_norm_host(t, alpha=2.0 ** -12):
    """The host of target t with an always-on weight-free embedding RMSNorm (eps = the config eps): host eps =
    alpha^2 eps, every residual writer x alpha (the stream is alpha x the target's), embedding row v = lambda_v e_v.
    The pre-norms and the final norm read alpha x at eps alpha^2 eps = the target's norm of x, exactly."""
    eps = t.c["eps"]
    h = _host(t, selfcond_norm=True, eps=alpha ** 2 * eps)
    e = t.p["emb"] * t.c["emb_scale"]
    ms = e.pow(2).mean(-1, keepdim=True)
    assert (alpha ** 2 * ms < 1).all()
    lam = alpha ** 2 * math.sqrt(eps) / torch.sqrt(1 - alpha ** 2 * ms)
    h.p["emb"] = t.p["emb"] * lam
    for i in range(t.n_param_layers()):
        for k in ("o", "down"):
            h.p[f"l{i}.{k}"] = t.p[f"l{i}.{k}"] * alpha
    if t.c["tied"]:                       # lm_head IS the scaled table: logits_v x lambda_v -> rescale the final gain
        h.p["lm"] = h.p["emb"]
        h.p["nf_w"] = t.p["nf_w"] / (alpha ** 2 * math.sqrt(eps))
    return h


@entry("port:sc_norm_off", "feature:selfcond_norm:host",
       instance=f"plain untied decoder -> a host whose input embedding passes an always-on weight-free RMSNorm "
                f"(DiffusionGemma self-conditioning post_norm, zero signal): residual x 2^-12, host eps x 2^-24, "
                f"embedding rows lambda_v e_v -- exact ({REF})")
def fr_selfcond_norm_untied():
    t = _t()
    return _check(t, selfcond_norm_host(t))


@entry("port:sc_norm_off", "feature:selfcond_norm:host",
       instance=f"the same on a TIED target / host (DiffusionGemma ties lm_head): logits_v x 1 / sqrt(1 - alpha^2 "
                f"ms(e_v)), a limit alpha -> 0 (alpha = 2^-12) ({REF})", standard="limit")
def fr_selfcond_norm_tied():
    t = _t(tied=True)
    return _check(t, selfcond_norm_host(t), standard="limit", tol=1e-6,
                  note="limit alpha -> 0: rel ~ alpha^2 ms(e) / 2")
