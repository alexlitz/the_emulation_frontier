"""End-to-end, pythia-160m: the LayerNorm -> RMSNorm fold WITHOUT an lm_head bias, at the final norm.

Target logits = W_U (gamma * LN(x) + beta), LN over the final residual x (embed_out has no bias).
Host final residual [P x; zeta; pad] (P = centring, the host keeps its stream centred), one RMSNorm
with eps_h = 0 over all m = d + 2 channels, pad^2 = d * eps_t (the pad IS the target epsilon, so
the stream readout is exactly sqrt(m/d) * Px / sigma, sigma = sqrt(var + eps_t)).
The beta term W_U beta needs a channel whose NORMED value is constant, i.e. zeta ∝ sigma.

Variants: sanity (beta term added by hand: must match native), (i) drop the beta term,
(ii) a constant carried channel (beta term arrives times r_ref / sigma; r_ref = median or best),
(iii) oracle zeta = lam * sqrt(|Px|^2 + d eps_t), (iv) zeta = lam * sqrt(m) * kappa * f(t),
f ≈ 1/t from fitlib (t = sqrt(m) kappa / |h| is the pad's normed coordinate, read by an extra
MLP before the final norm), fitted on the observed t range of this eval set.

Model forward in float32 on cuda:0; host arithmetic in float64 on the collected final residuals.
Run: PYENV_VERSION=3.12.6 python verify/prenorm_rescue/ln_fold_e2e.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import harness  # noqa: E402
import fitlib  # noqa: E402

LAM = 1e-2   # small: a fit error e moves the beta term by ~e and the stream term by ~lam^2 e


def metrics(native, host):
    top1 = (native.argmax(-1) == host.argmax(-1)).double().mean().item()
    la, lb = Fn.log_softmax(native, -1), Fn.log_softmax(host, -1)
    kl = (la.exp() * (la - lb)).sum(-1)
    return dict(top1=top1, kl_mean=kl.mean().item(), kl_max=kl.max().item(),
                max_dlogit=(native - host).abs().max().item())


@torch.no_grad()
def main():
    model, tok, dev = harness.load("EleutherAI/pythia-160m", dtype=torch.float32, device="cuda:0")
    ln = model.gpt_neox.final_layer_norm
    eps, d = ln.eps, model.config.hidden_size
    xs, nat = [], []
    h = ln.register_forward_hook(lambda m, i, o: xs.append(i[0][0].double().cpu()))
    for ids in harness.eval_batches(tok, dev, max_length=128, num_samples=32):
        nat.append(model(ids).logits[0].double().cpu())
    h.remove()
    X, native = torch.cat(xs), torch.cat(nat)
    WU = model.embed_out.weight.double().cpu()
    g, b = ln.weight.double().cpu(), ln.bias.double().cpu()
    bterm = WU @ b                                     # the per-vocab constant W_U beta

    Px = X - X.mean(-1, keepdim=True)
    sq = Px.pow(2).sum(-1)                             # |Px|^2
    sigma = torch.sqrt(sq / d + eps)
    kappa = (d * eps) ** 0.5

    def host_logits(zeta):                             # zeta: (N,) channel; m = d + 2
        m = d + 2
        r = torch.sqrt((sq + zeta ** 2 + kappa ** 2) / m)          # eps_h = 0
        stream = Px / r[:, None]                       # normed stream
        zn = zeta / r                                  # normed zeta
        return stream, zn, m

    out = {"model": "EleutherAI/pythia-160m", "n_tok": int(X.shape[0]), "eps_t": eps,
           "sigma_range": [sigma.min().item(), sigma.max().item()], "lam": LAM}
    # sanity: the host stream readout (zeta = 0) times sqrt(d/m) is LN(x) - beta; add beta by hand
    stream, _, m = host_logits(torch.zeros_like(sq))
    base = (stream * (d / m) ** 0.5 * g) @ WU.T
    out["sanity_beta_by_hand"] = metrics(native, base + bterm)
    out["i_drop_beta"] = metrics(native, base)
    # (ii) a constant carried channel (inside the pad budget): beta term times r_ref / sigma
    rr = {"median": sigma.median().item()}
    grid = np.geomspace(sigma.min().item(), sigma.max().item(), 60)
    kls = [metrics(native, base + bterm * (r / sigma)[:, None])["kl_mean"] for r in grid]
    rr["best_kl"] = float(grid[int(np.argmin(kls))])
    out["ii_constant_channel"] = {k: dict(r_ref=v, **metrics(native, base + bterm * (v / sigma)[:, None]))
                                  for k, v in rr.items()}

    # (iii) oracle zeta and (iv) fitted zeta; readout weights fixed from the oracle design
    def with_zeta(zeta):
        stream, zn, m = host_logits(zeta)
        s_scale = (1 + LAM ** 2) ** 0.5 * (d / m) ** 0.5          # exact for the oracle zeta
        z_target = LAM * m ** 0.5 / (1 + LAM ** 2) ** 0.5         # the oracle's normed zeta
        return (stream * s_scale * g) @ WU.T + bterm * (zn / z_target)[:, None]

    zeta_star = LAM * torch.sqrt(sq + d * eps)
    out["iii_oracle_zeta"] = metrics(native, with_zeta(zeta_star))
    m = d + 2
    t = (m ** 0.5 * kappa / torch.sqrt(sq + kappa ** 2)).numpy()    # pad's normed coord, zeta = 0
    out["t_range"] = [float(t.min()), float(t.max()), float(t.max() / t.min())]
    out["iv_fitted_zeta"] = {}
    for fam in ("glu", "relu"):
        for k in (2, 4, 8, 16):
            comp = fitlib.Comp(fam, k, float(t.min()), float(t.max()))
            zeta = torch.from_numpy(LAM * m ** 0.5 * kappa * comp(t))
            rel = (zeta / zeta_star - 1).abs().max().item()
            out["iv_fitted_zeta"][f"{fam}_k{k}"] = dict(fit_sup_rel=rel, **metrics(native, with_zeta(zeta)))
            print(fam, k, out["iv_fitted_zeta"][f"{fam}_k{k}"], flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "ln_fold_e2e.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "iv_fitted_zeta"}, indent=1))


if __name__ == "__main__":
    main()
