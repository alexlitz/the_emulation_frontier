"""Achievability probe (NOT a construction): distil a softmax-attention layer onto each recurrent mixer
layer of a real model and score the swapped model with the mixer-gap standard.

Per mixer layer, a softmax layer of a given head budget -- q, k, v, o projections, RoPE, causal
softmax with a learned per-head sink logit -- is trained for a few hundred Adam steps to map the
mixer's own input (the normalised hidden state the real model feeds it) to the mixer's output, on
wikitext-103 train text (teacher hooks, all layers at once, each layer its own loss: relative MSE).
Held-out relative MSE is reported on wikitext-103 validation; then every mixer is replaced by its
student and the model is scored on the 24 prompts exactly as attn/mixer_gap_agreement.py does.

It is an upper-bound probe of what a budget-matched softmax layer can reach with fitting, distinct
from the explicit constructions; the graph edge is about constructions.

  python -m attn.mixer_gap_distill mamba --heads 12 --steps 600 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import attn.mixer_gap_agreement as M

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SoftmaxMixer(nn.Module):
    def __init__(self, hidden, heads, head_dim=64, base=10000.0):
        super().__init__()
        self.h, self.d = heads, head_dim
        self.q = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k = nn.Linear(hidden, heads * head_dim, bias=False)
        self.v = nn.Linear(hidden, heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, hidden, bias=False)
        self.sink = nn.Parameter(torch.zeros(heads))
        self.register_buffer("inv", 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim), persistent=False)
        nn.init.normal_(self.o.weight, std=0.02 / math.sqrt(2))

    def rope(self, x):                                   # x [B, H, T, d]
        T = x.shape[2]
        f = torch.arange(T, device=x.device).float()[:, None] * self.inv[None]
        cos, sin = f.cos(), f.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    def forward(self, x):                                # x [B, T, hidden]
        B, T, _ = x.shape
        sh = lambda y: y.view(B, T, self.h, self.d).transpose(1, 2)
        q, k, v = self.rope(sh(self.q(x))), self.rope(sh(self.k(x))), sh(self.v(x))
        lg = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        lg = lg.masked_fill(~M.causal_mask(T, x.device), float("-inf"))
        sink = self.sink.view(1, -1, 1, 1).expand(B, -1, T, 1)
        w = torch.softmax(torch.cat([sink, lg], -1), -1)[..., 1:]
        return self.o((w @ v).transpose(1, 2).reshape(B, T, -1))


def mixers(model, family):
    if family == "mamba":
        return [blk.mixer for blk in model.backbone.layers]
    raise ValueError(family)


def token_stream(tok, split, n_tokens):
    from datasets import load_dataset
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ids, i = [], 0
    while len(ids) < n_tokens and i < len(ds):
        t = ds[i]["text"]
        if t.strip():
            ids.extend(tok(t).input_ids)
        i += 1
    return torch.tensor(ids[:n_tokens])


def capture(model, family, seqs):
    """Run the teacher; return per layer (inputs, outputs) of every mixer."""
    store = {}
    mods = mixers(model, family)

    def make(i, mod):
        def f(self, *a, **k):
            hs = M._hs(a, k)
            out = type(self).forward(self, *a, **k)
            store[i] = (hs.detach(), out.detach())
            return out
        return f
    for i, mod in enumerate(mods):
        mod.forward = types.MethodType(make(i, mod), mod)
    with torch.no_grad():
        model(seqs, use_cache=False)
    for mod in mods:
        mod.forward = types.MethodType(type(mod).forward, mod)
    return store


def train(family, heads, steps, device, bs=8, T=256, lr=1e-3, seed=0):
    mid, tok, model = M.load(family, device)
    for p in model.parameters():
        p.requires_grad_(False)
    hidden = model.config.hidden_size
    mods = mixers(model, family)
    torch.manual_seed(seed)
    studs = nn.ModuleList([SoftmaxMixer(hidden, heads) for _ in mods]).to(device)
    opt = torch.optim.AdamW(studs.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 3 / steps))
    tr = token_stream(tok, "train", bs * T * steps // 4 + bs * T)      # each token seen ~4 times
    va = token_stream(tok, "validation", 16 * T)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        st = torch.randint(0, len(tr) - T, (bs,), generator=g)
        seqs = torch.stack([tr[s:s + T] for s in st]).to(device)
        cap = capture(model, family, seqs)
        loss = 0.0
        for i, s in enumerate(studs):
            x, y = cap[i]
            loss = loss + ((s(x) - y) ** 2).mean() / (y ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step}: mean rel MSE {loss.item() / len(studs):.4f} ({time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
    # held-out
    with torch.no_grad():
        seqs = va.view(16, T).to(device)
        cap = capture(model, family, seqs)
        held = [(((s(cap[i][0]) - cap[i][1]) ** 2).mean() / (cap[i][1] ** 2).mean()).item()
                for i, s in enumerate(studs)]
    return mid, tok, model, studs, held


def evaluate(family, mid, tok, model, studs, device, n_prompts=24):
    prompts = M._prompts()[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    ref_cont = [M.greedy(model, ids) for ids in enc]
    ref_seq = [torch.cat([i, c], 1) for i, c in zip(enc, ref_cont)]
    ref_lp = [M.cont_logp(model, s, i.shape[1]) for s, i in zip(ref_seq, enc)]
    mods = mixers(model, family)
    for mod, s in zip(mods, studs):
        mod.forward = types.MethodType(lambda self, *a, _s=s, **k: _s(M._hs(a, k)), mod)
    try:
        r = M.score(model, enc, ref_seq, ref_lp, ref_cont)
    finally:
        for mod in mods:
            mod.forward = types.MethodType(type(mod).forward, mod)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["mamba"])
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=M.OUT)
    a = ap.parse_args()
    t0 = time.time()
    mid, tok, model, studs, held = train(a.family, a.heads, a.steps, a.device)
    r = evaluate(a.family, mid, tok, model, studs, a.device)
    row = dict(model=mid, family=a.family, variant=f"distill-h{a.heads}-s{a.steps}", layers=None,
               n_mixers=len(studs), n_prompts=24, cfg=dict(heads=a.heads, head_dim=64, steps=a.steps, bs=8, T=256),
               cost=dict(heads=a.heads, qk_dim=64, width=1, depth=1, groupnorm=False,
                         note="PROBE: per-layer distilled softmax layer (RoPE + sink), not a construction"),
               heldout_rel_mse=held, **r, secs=round(time.time() - t0, 1))
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  distill-h{a.heads}: held-out rel MSE median {sorted(held)[len(held) // 2]:.4f}  top1 {r['top1']:.4f}  "
          f"ex32 {r['ex32']:.2f}  KL {r['kl']:.2e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
