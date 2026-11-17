"""Is MLA <-> plain (GQA / MHA) attention PAYABLE?  The per-token norm of the compressed latent, measured.

graph/model_graph.py KV_ORDER used to read MQA <= GQA <= MLA <= MHA, all exact: an MLA target was a
plain attention layer with a low-rank K / V factorisation (attn/mla_rank.py absorbs the up-projections
exactly, in the norm-free toy). Real MLA (DeepSeek-V2 / V3, Kimi-K2, GLM-4.7-Flash / GLM-5, Mistral-4,
MiniCPM3, Youtu, ...) does NOT compute K, V linearly in the normed residual x:

    c_t   = W_dkv x_t                     (rank kv_lora_rank)
    k_t   = W_uk RMSNorm(c_t),  v_t = W_uv RMSNorm(c_t)        <- kv_a_layernorm, per token
    q_t   = W_uq RMSNorm(W_dq x_t)                             <- q_a_layernorm (when q_lora_rank)

RMSNorm(c_t) = g * c_t / rms(c_t): every key and value of token t is scaled by 1 / rms(c_t), a factor
that varies from token to token. A plain host's k_proj / v_proj are linear in x_t, so the best it can do
is fold the norm into a CONSTANT per-layer scale (the qk-norm regime fold, attn/qk_norm.py): its keys and
values are W_uk g alpha c_t with one alpha per layer. This module computes exactly that host's forward
pass on a real MLA checkpoint: every kv_a_layernorm (and q_a_layernorm) replaced by g * alpha_l * c with
alpha_l the least-squares optimum on calibration text (alpha = sum |c|^2 / r / sum |c|^2 over the
calibration tokens, r = rms(c)), everything else untouched (the up-projections absorb exactly, the
decoupled RoPE key is an extra shared rotary head: both exact in a plain host of enough kv_cap).

Scored with the act/act_swap_agreement.py standard on the ORIGINAL model's own greedy 64-token
continuations of the same 24 prompts, fp32:

  payable  <=>  top-1 agreement >= 99%  AND  >= 90% of 32-token greedy continuations identical.

Also recorded: the per-layer coefficient of variation of rms(c_t) on the calibration text (how far
the latent norm is from a constant), and the per-layer relative error of the attention output.

The REVERSE direction (an MLA host for a plain GQA / MHA target): the host's latent norm CAN be switched
off -- the eps lever, graph/eps_lever.py: scale the latent rows of kv_a_proj_with_mqa by 2^-a so that
ms(c) << eps, the norm is then the constant 1 / sqrt(eps) and its gain undoes it (measured exact on
gemma-2b in a DeepseekV2 host, hf/data/eps_lever_agreement.jsonl). What still blocks it: the rotary
part is ONE key head of qk_rope_head_dim dims shared by every query head, so a RoPE target with
n_kv > 1 rotating KV heads (or rotary dims > qk_rope_head_dim) has rotating keys the host cannot form
at all; an MQA RoPE target (gemma-2b) fits.

MEASURED (2026-11-12, attn/data/mla_norm_agreement.jsonl): DeepSeek-V2-Lite (no q latent), all 27 KV latent
norms folded: top-1 81.0%, 4.2% of 32-token continuations identical (KL 0.29, first divergence ~4.5 tokens);
the latent rms varies 0.4 - 11x across tokens (per-layer cv 0.17 - 1.58). Not payable.

  python -m attn.mla_norm_agreement deepseek-ai/DeepSeek-V2-Lite --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from act.act_swap_agreement import N_NEW, PROMPTS, cont_logp, greedy
from hf.attn_pin import attn_impl_for            # eager pinned (hf/attn_pin.py)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# calibration text: disjoint from PROMPTS
CALIB = [
    "The committee met on Tuesday to discuss the annual budget, and after a long debate the members "
    "agreed to postpone the vote until more information about the expected revenue was available.",
    "def merge_sort(xs):\n    if len(xs) <= 1:\n        return xs\n    mid = len(xs) // 2\n"
    "    return merge(merge_sort(xs[:mid]), merge_sort(xs[mid:]))\n",
    "Volcanoes form where magma from the mantle reaches the surface. Some erupt explosively, throwing "
    "ash high into the atmosphere, while others release slow rivers of lava.",
    "Le petit prince demanda au renard pourquoi il fallait des rites, et le renard lui expliqua que "
    "c'est ce qui rend un jour different des autres jours.",
    "1. Preheat the oven to 180 C. 2. Mix the flour, sugar and butter. 3. Add two eggs and stir until "
    "the batter is smooth. 4. Bake for 35 minutes.",
    "In 1969 the Apollo 11 mission landed the first humans on the Moon; Neil Armstrong and Buzz Aldrin "
    "spent about two and a half hours outside the lunar module.",
]


class ConstNorm(torch.nn.Module):
    """g * alpha * x: the latent norm folded to one constant -- what a plain host's projections compute."""

    def __init__(self, weight, alpha):
        super().__init__()
        self.weight, self.alpha = weight, alpha

    def forward(self, x):
        return self.weight * (x.float() * self.alpha).to(x.dtype)


def latent_norms(model):
    """[(attention module, attr)] for every per-token latent norm: kv_a_layernorm, q_a_layernorm."""
    out = []
    for name, mod in model.named_modules():
        for attr in ("kv_a_layernorm", "q_a_layernorm"):
            if isinstance(getattr(mod, attr, None), torch.nn.Module):
                out.append((name, mod, attr))
    return out


@torch.no_grad()
def calibrate(model, tok, norms, device):
    """Per latent norm: alpha (LS-optimal constant 1/rms), cv of rms over tokens."""
    acc = {(n, a): [0.0, 0.0, []] for n, _, a in norms}
    hooks = []
    for n, mod, a in norms:
        nm = getattr(mod, a)
        eps = getattr(nm, "variance_epsilon", getattr(nm, "eps", 1e-6))

        def pre(m, args, key=(n, a), eps=eps):
            c = args[0].detach().double().reshape(-1, args[0].shape[-1])
            r = (c.pow(2).mean(-1) + eps).sqrt()
            s = acc[key]
            s[0] += (c.pow(2).sum(-1) / r).sum().item()
            s[1] += c.pow(2).sum().item()
            s[2].append(r.cpu())
        hooks.append(nm.register_forward_pre_hook(pre))
    for t in CALIB:
        model(tok(t, return_tensors="pt").input_ids.to(device))
    for h in hooks:
        h.remove()
    res = {}
    for k, (num, den, rs) in acc.items():
        r = torch.cat(rs)
        res[k] = dict(alpha=num / den, cv=(r.std() / r.mean()).item(), rmin=r.min().item(), rmax=r.max().item())
    return res


def run(model_id, device="cpu", n_prompts=24, label=None, variants=("self", "kv", "q", "kv+q"), sink=None,
        trust_remote_code=False, threads=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if threads:
        torch.set_num_threads(threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 attn_implementation=attn_impl_for(model_id, trust_remote_code),
                                                 trust_remote_code=trust_remote_code).to(device).eval()
    norms = latent_norms(model)
    assert norms, "no latent norm (kv_a_layernorm / q_a_layernorm) found"
    cal = calibrate(model, tok, norms, device)
    for (n, a), v in sorted(cal.items()):
        print(f"  {n}.{a}: alpha {v['alpha']:.4g}  cv(rms) {v['cv']:.3f}  rms [{v['rmin']:.3g}, {v['rmax']:.3g}]",
              file=sys.stderr, flush=True)
    orig = {(n, a): getattr(mod, a) for n, mod, a in norms}
    const = {(n, a): ConstNorm(getattr(mod, a).weight, cal[(n, a)]["alpha"]) for n, mod, a in norms}

    def set_variant(v):
        kinds = set() if v == "self" else set(v.split("+"))
        for n, mod, a in norms:
            use = ("kv" in kinds and a == "kv_a_layernorm") or ("q" in kinds and a == "q_a_layernorm")
            setattr(mod, a, const[(n, a)] if use else orig[(n, a)])

    prompts = PROMPTS[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    set_variant("self")
    ref_cont = [greedy(model, ids) for ids in enc]
    ref_seq = [torch.cat([ids, c], 1) for ids, c in zip(enc, ref_cont)]
    ref_lp = [cont_logp(model, s, ids.shape[1]) for s, ids in zip(ref_seq, enc)]
    print(f"  reference done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    has_q = any(a == "q_a_layernorm" for _, _, a in norms)
    rows = []
    for v in variants:
        if "q" in v.split("+") and not has_q:
            continue
        set_variant(v)
        top1 = kl = 0.0
        n = ex32 = ex64 = 0
        div = []
        for ids, s, lq, rc in zip(enc, ref_seq, ref_lp, ref_cont):
            lp = cont_logp(model, s, ids.shape[1])
            kl += (lq.exp() * (lq - lp)).sum().item()
            top1 += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
            n += lp.shape[0]
            g = greedy(model, ids)
            same = (g[0] == rc[0]).cpu().numpy()
            ex32 += bool(same[:32].all())
            ex64 += bool(same.all())
            div.append(int(np.argmin(same)) if not same.all() else N_NEW)
        kv_cv = [c["cv"] for (nn, a), c in cal.items() if a == "kv_a_layernorm"]
        q_cv = [c["cv"] for (nn, a), c in cal.items() if a == "q_a_layernorm"]
        row = dict(model=label or model_id, variant=v, n_norms=len(norms), top1=top1 / n, kl=kl / n,
                   ex32=ex32 / len(enc), ex64=ex64 / len(enc), div=float(np.mean(div)), n_prompts=len(enc),
                   kv_cv_mean=float(np.mean(kv_cv)) if kv_cv else None, kv_cv_max=float(np.max(kv_cv)) if kv_cv else None,
                   q_cv_mean=float(np.mean(q_cv)) if q_cv else None)
        rows.append(row)
        if sink is not None:
            sink(row)
        print(f"  {v:>6}: top1 {row['top1']:.4f}  ex32 {row['ex32']:.2f}  ex64 {row['ex64']:.2f}  "
              f"KL {row['kl']:.2e}  div {row['div']:.1f}  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    set_variant("self")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="attn/data/mla_norm_agreement.jsonl")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--variants", default="self,kv,q,kv+q")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--label")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.model, a.device, a.n_prompts, a.label, tuple(a.variants.split(",")), sink, a.trust_remote_code, a.threads)


if __name__ == "__main__":
    main()
