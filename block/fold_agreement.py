"""Is a per-token op PAYABLE by a host that lacks it?  Swap in what the host computes instead, measure.

Several forward-pass features have no exact construction in a host that does not carry them: a
per-token normalisation inside a sublayer (an MLA latent norm, a qk-norm, a v-norm, a post-sublayer
norm) is not linear in the residual, a logit softcap is not the identity, a learned attention sink is a
query-independent logit no bias-free RoPE projection forms, a bias needs a constant input channel no
pre-norm residual offers. The best a host WITHOUT the feature can do is a weight setting:

  latent / qknorm / vnorm / postnorm   the norm folded to a constant scale: g * alpha * x, alpha the
                    least-squares optimum on calibration text (alpha = sum |x|^2 / r / sum |x|^2, r = rms(x),
                    one alpha per head where the norm is per head) -- the regime fold of attn/qk_norm.py
  softcap           the cap removed (logits as they are; attention and final caps)
  sink              the learned per-head sink logits removed (-> -inf)
  bias_q / bias_k / bias_v   that projection's bias dropped (a key bias would shift every logit of a query
                    by the same q.b_k -- softmax-invariant -- were it not rotated: under RoPE the term is
                    q_t . R_{s-t} b_k, position-dependent, so it is not free either)

and the same expression bounds the reverse direction (a host WITH the feature playing a target without
it: its norm folds to the same constant only in the same regime, its cap is the identity only where
logits are small, ...). Every listed module of the real checkpoint is swapped, then generation is scored
with the act/act_swap_agreement.py standard on the ORIGINAL model's own greedy 64-token continuations of
24 prompts, fp32, TF32 off:

  payable  <=>  top-1 agreement >= 99%  AND  >= 90% of 32-token greedy continuations identical.

Recorded per variant: top1, KL, ex32 / ex64, mean first divergence, the number of swapped modules, and
for the norm folds the per-module coefficient of variation of rms(x) on the calibration text (how far the
norm is from a constant).

  python -m block.fold_agreement Qwen/Qwen3-0.6B --folds qknorm
  python -m block.fold_agreement deepseek-ai/DeepSeek-V2-Lite --folds latent --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from act.act_swap_agreement import N_NEW, PROMPTS, STORY_PROMPTS, cont_logp, greedy

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# calibration text: disjoint from PROMPTS / STORY_PROMPTS
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
CALIB_STORY = [
    "Once upon a time, there was a boy named Leo who had a blue kite. He went to the hill with his dad "
    "and the wind carried the kite up into the sky.",
    "Lily and her cat played in the garden. The cat saw a butterfly and jumped, but the butterfly flew "
    "away. Lily laughed and gave the cat a hug.",
    "One day, a little duck could not find his mom. He asked the frog and the fish for help, and at last "
    "he found her by the big tree.",
    "Tim had a toy car. It was red and fast. He raced it on the floor and it crashed into the wall. Tim "
    "was sad, but his sister helped him fix it.",
]

# norm modules per fold kind: attribute names on any submodule
NORM_ATTRS = {
    "latent": ("kv_a_layernorm", "q_a_layernorm"),
    "qknorm": ("q_norm", "k_norm", "q_layernorm", "k_layernorm"),
    "vnorm": ("v_norm",),
    "postnorm": ("post_attention_layernorm", "post_feedforward_layernorm"),
}
# families whose post_attention_layernorm IS a post-sublayer norm (elsewhere it is the FFN pre-norm)
POSTNORM_TYPES = {"olmo2", "olmo3", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text"}


class ConstFold(torch.nn.Module):
    """What a host without the norm computes: the target norm's gain times alpha * x (alpha per head
    when the norm runs per head), i.e. orig(x) * rms_eps(x) * alpha."""

    def __init__(self, orig, alpha, eps):
        super().__init__()
        self.orig, self.alpha, self.eps = orig, alpha, eps

    def forward(self, x):
        xf = x.float()
        r = (xf.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return (self.orig(x).float() * r * self.alpha.to(x.device)).to(x.dtype)


def _eps(m):
    for a in ("variance_epsilon", "eps", "epsilon"):
        v = getattr(m, a, None)
        if isinstance(v, (int, float)):
            return float(v)
    return 1e-6


def norm_modules(model, kind):
    mt = getattr(getattr(model.config, "text_config", model.config), "model_type", "")
    if kind == "postnorm" and mt not in POSTNORM_TYPES:
        return []
    out = []
    for name, mod in model.named_modules():
        for attr in NORM_ATTRS[kind]:
            m = getattr(mod, attr, None)
            if isinstance(m, torch.nn.Module) and not isinstance(m, ConstFold):
                out.append((f"{name}.{attr}", mod, attr))
    return out


@torch.no_grad()
def calibrate(model, tok, mods, device, texts):
    """alpha (LS-optimal 1/rms, per head for 4-d inputs) and the cv of rms, per module."""
    acc = {n: [0, 0, []] for n, _, _ in mods}
    hooks = []
    for n, mod, attr in mods:
        nm = getattr(mod, attr)
        eps = _eps(nm)

        def pre(m, args, key=n, eps=eps):
            x = args[0].detach().double()
            r = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
            dims = tuple(range(x.dim() - 2)) if x.dim() >= 4 else tuple(range(x.dim() - 1))
            num = (x.pow(2) / r).sum(dims + (x.dim() - 1,))           # per head (4-d) or scalar
            den = x.pow(2).sum(dims + (x.dim() - 1,))
            s = acc[key]
            s[0] = s[0] + num.cpu()
            s[1] = s[1] + den.cpu()
            s[2].append(r.reshape(-1).cpu())
        hooks.append(nm.register_forward_pre_hook(pre))
    for t in texts:
        model(tok(t, return_tensors="pt").input_ids.to(device))
    for h in hooks:
        h.remove()
    res = {}
    for n, (num, den, rs) in acc.items():
        r = torch.cat(rs)
        a = (num / den.clamp_min(1e-30)).float()
        res[n] = dict(alpha=a.reshape(-1, 1) if a.dim() else a, cv=(r.std() / r.mean()).item(),
                      rmin=r.min().item(), rmax=r.max().item())
    return res


def _softcap_setter(model):
    """Turn every attention / final logit softcap off (and back): returns setter(on: bool) or None."""
    cfgs = [model.config] + ([model.config.text_config] if hasattr(model.config, "text_config") else [])
    saved = []
    for c in cfgs:
        for k in ("attn_logit_softcapping", "final_logit_softcapping", "logit_softcap"):
            if getattr(c, k, None):
                saved.append((c, k, getattr(c, k)))
    for _, mod in model.named_modules():
        for k in ("attn_logit_softcapping", "softcap", "logit_softcap"):
            v = getattr(mod, k, None)
            if isinstance(v, (int, float)) and v:
                saved.append((mod, k, v))
    if not saved:
        return None

    def set_on(on):
        for o, k, v in saved:
            setattr(o, k, v if on else None)
    return set_on


def _sink_setter(model):
    params = [(n, p) for n, p in model.named_parameters() if n.endswith("sinks")]
    if not params:
        return None
    orig = {n: p.data.clone() for n, p in params}

    def set_on(on):
        for n, p in params:
            p.data.copy_(orig[n] if on else torch.full_like(orig[n], -1e4))
    return set_on


def _bias_setter(model, which):
    """Drop the q / k / v projection biases (which in 'q', 'k', 'v')."""
    names = {"q": ("q_proj",), "k": ("k_proj",), "v": ("v_proj",)}[which]
    mods = [m for n, m in model.named_modules()
            if n.split(".")[-1] in names and isinstance(m, torch.nn.Linear) and m.bias is not None]
    if not mods:
        return None
    orig = [m.bias.data.clone() for m in mods]

    def set_on(on):
        for m, b in zip(mods, orig):
            m.bias.data.copy_(b if on else torch.zeros_like(b))
    return set_on


def run(model_id, folds, device="cuda:0", n_prompts=24, story=False, label=None, sink=None,
        trust_remote_code=False, threads=None, dtype="float32", attn="eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if threads:
        torch.set_num_threads(threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), attn_implementation=attn,
                                                 trust_remote_code=trust_remote_code).to(device).eval()
    texts = CALIB_STORY if story else CALIB
    setters, info = {}, {}
    for f in folds:
        if f in NORM_ATTRS:
            mods = norm_modules(model, f)
            if not mods:
                print(f"  {f}: no module found, skipped", file=sys.stderr, flush=True)
                continue
            cal = calibrate(model, tok, mods, device, texts)
            orig = {n: getattr(mod, a) for n, mod, a in mods}
            const = {n: ConstFold(getattr(mod, a), cal[n]["alpha"], _eps(getattr(mod, a))) for n, mod, a in mods}

            def set_on(on, mods=mods, orig=orig, const=const):
                for n, mod, a in mods:
                    setattr(mod, a, orig[n] if on else const[n])
            setters[f] = set_on
            cvs = [c["cv"] for c in cal.values()]
            info[f] = dict(n_mod=len(mods), cv_mean=float(np.mean(cvs)), cv_max=float(np.max(cvs)),
                           rms_ratio_max=float(max(c["rmax"] / max(c["rmin"], 1e-30) for c in cal.values())))
            print(f"  {f}: {len(mods)} modules, cv(rms) mean {info[f]['cv_mean']:.3f} max {info[f]['cv_max']:.3f}",
                  file=sys.stderr, flush=True)
        elif f == "softcap":
            s = _softcap_setter(model)
            if s:
                setters[f], info[f] = s, {}
        elif f == "sink":
            s = _sink_setter(model)
            if s:
                setters[f], info[f] = s, {}
        elif f.startswith("bias_"):
            s = _bias_setter(model, f[5:])
            if s:
                setters[f], info[f] = s, {}
        else:
            raise ValueError(f)
    for f in folds:
        if f not in setters:
            print(f"  {f}: not present in {model_id}", file=sys.stderr, flush=True)

    def set_variant(v):
        on = set() if v == "self" else set(v.split("+"))
        for f, s in setters.items():
            s(f not in on)

    prompts = (STORY_PROMPTS if story else PROMPTS)[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    set_variant("self")
    ref_cont = [greedy(model, ids) for ids in enc]
    ref_seq = [torch.cat([ids, c], 1) for ids, c in zip(enc, ref_cont)]
    ref_lp = [cont_logp(model, s, ids.shape[1]) for s, ids in zip(ref_seq, enc)]
    rows = []
    variants = ["self"] + [f for f in folds if f in setters]
    for v in variants:
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
        row = dict(model=label or model_id, fold=v, top1=top1 / n, kl=kl / n, ex32=ex32 / len(enc),
                   ex64=ex64 / len(enc), div=float(np.mean(div)), n_prompts=len(enc), dtype=dtype,
                   **({k: val for k, val in info.get(v, {}).items()}))
        rows.append(row)
        if sink is not None:
            sink(row)
        print(f"  {v:>10}: top1 {row['top1']:.4f}  ex32 {row['ex32']:.2f}  ex64 {row['ex64']:.2f}  "
              f"KL {row['kl']:.2e}  div {row['div']:.1f}  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    set_variant("self")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--folds", required=True, help="comma-separated: latent, qknorm, vnorm, postnorm, softcap, "
                                                   "sink, bias_q, bias_k, bias_v")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="block/data/fold_agreement.jsonl")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--story", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--label")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.model, a.folds.split(","), a.device, a.n_prompts, a.story, a.label, sink, a.trust_remote_code, a.threads)


if __name__ == "__main__":
    main()
