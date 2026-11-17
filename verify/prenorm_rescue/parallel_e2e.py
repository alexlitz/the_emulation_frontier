"""End-to-end, pythia-160m (parallel residual: x + Attn(LN1 x) + MLP(LN2 x)) emulated by a
SEQUENTIAL pre-norm host at ~2x width. Each GPTNeoXLayer.forward is replaced by the host's
computation, using the layer's own attention / MLP modules (so the host MLP computes the target's
GELU MLP exactly: this isolates the carry error).

Host residual [P x; c; pad] (centred stream; every host norm is an RMSNorm over ALL channels with
eps_h = 0 and pad^2 = d * eps_t). Its stream readout, rescaled by the known sqrt(d/m), is
P x / s with s = sqrt((|Px|^2 + |c|^2 + d eps_t) / d); with c = 0, s = sqrt(var + eps_t), the
target's LayerNorm denominator (LN = gamma * Px / s + beta; beta folded into the host's biases).

Modes:
  target      parallel block (sanity: must equal HF logits)
  seq         plain sequential, no carry: the MLP reads LN2(x + A)
  naive       carry c = P A; the MLP reads the polluted norm; A is moved into the stream by
              identity units reading P A / s, times a fixed per-layer constant (median s, pass 1)
  vanish      c = delta P A; MLP read polluted by O(delta^2); A moved with the ORACLE rescale s/delta
  fit         delta = 1e-3, rescale s taken from a fitted 1D compensator f(t) ≈ 1/t of the pad's
              normed coordinate t (needs product units, i.e. a GLU host)
float64 on cuda:1. Run: PYENV_VERSION=3.12.6 python verify/prenorm_rescue/parallel_e2e.py
"""
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import harness  # noqa: E402
import fitlib  # noqa: E402

DEV = "cuda:1"
CFG = {}          # mode, delta, per-layer s_ref / compensators, recorder


def ln_with(mod, x, s):
    """LayerNorm `mod` applied with an arbitrary denominator s (per token)."""
    Px = x - x.mean(-1, keepdim=True)
    return mod.weight * Px / s[..., None] + mod.bias


def host_forward(layer, idx):
    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kw):
        x = hidden_states
        d = x.shape[-1]
        eps = self.input_layernorm.eps
        m = 2 * d + 1
        attn = lambda inp: self.attention(inp, attention_mask=attention_mask,
                                          position_embeddings=position_embeddings)[0]
        Px = x - x.mean(-1, keepdim=True)
        sq = Px.pow(2).sum(-1)
        A = attn(self.input_layernorm(x))                       # exact: carry empty at this read
        PA = A - A.mean(-1, keepdim=True)
        mode = CFG["mode"]
        if mode == "target":
            return (x + A + self.mlp(self.post_attention_layernorm(x)),)
        if mode == "seq":
            y = x + A
            return (y + self.mlp(self.post_attention_layernorm(y)),)
        delta = 1.0 if mode == "naive" else CFG["delta"]
        c2 = (delta ** 2) * PA.pow(2).sum(-1)
        s = torch.sqrt((sq + c2 + d * eps) / d)                 # host denominator at the MLP read
        M = self.mlp(ln_with(self.post_attention_layernorm, x, s))
        if "rec" in CFG:
            CFG["rec"].setdefault(idx, []).append(s.flatten().cpu())
        if mode == "naive":
            Ahat = PA / s[..., None] * CFG["s_ref"][idx]
        elif mode == "vanish":
            Ahat = PA                                           # oracle: (delta PA / s) * s / delta
        else:                                                   # fitted compensator of t
            kappa = (d * eps) ** 0.5
            t = (m ** 0.5 * kappa / torch.sqrt(sq + c2 + kappa ** 2)).cpu().numpy()
            s_hat = m ** 0.5 * kappa * CFG["comp"][idx](t) / d ** 0.5
            Ahat = PA / s[..., None] * torch.from_numpy(s_hat).to(x)[..., None]
        return (x + Ahat + M,)
    return forward


def metrics(a, b):
    a, b = torch.cat(a), torch.cat(b)
    la, lb = Fn.log_softmax(a, -1), Fn.log_softmax(b, -1)
    kl = (la.exp() * (la - lb)).sum(-1)
    return dict(top1=(a.argmax(-1) == b.argmax(-1)).double().mean().item(),
                kl_mean=kl.mean().item(), kl_max=kl.max().item(),
                max_dlogit=(a - b).abs().max().item())


@torch.no_grad()
def main():
    model, tok, dev = harness.load("EleutherAI/pythia-160m", dtype=torch.float64, device=DEV)
    batches = harness.eval_batches(tok, dev, max_length=128, num_samples=32)
    native = [model(ids).logits[0].cpu() for ids in batches]
    for i, L in enumerate(model.gpt_neox.layers):
        L.forward = types.MethodType(host_forward(L, i), L)
    run = lambda: [model(ids).logits[0].cpu() for ids in batches]
    out = {"model": "EleutherAI/pythia-160m", "dtype": "float64", "n_tok": sum(b.shape[1] for b in batches)}
    CFG["mode"] = "target"; out["target_sanity"] = metrics(native, run())
    CFG["mode"] = "seq"; out["o_sequential"] = metrics(native, run())
    # pass 1 (oracle vanish delta=1e-3) records the host denominators -> s_ref, t ranges
    CFG.update(mode="vanish", delta=1e-3, rec={})
    run()
    rec = {i: torch.cat(v).numpy() for i, v in CFG.pop("rec").items()}
    CFG["s_ref"] = {i: float(np.median(v)) for i, v in rec.items()}
    CFG["mode"] = "naive"; out["i_naive_carry"] = metrics(native, run())
    out["ii_vanish_oracle"] = {}
    for delta in (1e-1, 1e-2, 1e-3):
        CFG.update(mode="vanish", delta=delta)
        out["ii_vanish_oracle"][str(delta)] = metrics(native, run())
    d, eps = model.config.hidden_size, model.config.layer_norm_eps
    m, kappa = 2 * d + 1, (d * eps) ** 0.5
    # t = sqrt(m) kappa / |h| and |h| = sqrt(d) s  ->  t range per layer from the recorded s
    trange = {i: (m ** 0.5 * kappa / (d ** 0.5 * v.max()), m ** 0.5 * kappa / (d ** 0.5 * v.min()))
              for i, v in rec.items()}
    out["s_dynamic_range_per_layer"] = {i: float(v.max() / v.min()) for i, v in rec.items()}
    out["iii_fitted"] = {}
    CFG.update(mode="fit", delta=1e-3)
    for fam in ("glu", "relu"):
        for k in (2, 4, 8, 16):
            CFG["comp"] = {i: fitlib.Comp(fam, k, lo * 0.999, hi * 1.001) for i, (lo, hi) in trange.items()}
            r = metrics(native, run())
            r["worst_layer_fit_err"] = max(float(c.err) for c in CFG["comp"].values())
            out["iii_fitted"][f"{fam}_k{k}"] = r
            print(fam, k, r, flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "parallel_e2e.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "iii_fitted"}, indent=1))


if __name__ == "__main__":
    main()
