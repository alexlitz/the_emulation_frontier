"""Logit-KL distillation of a converted student from the teacher.

What fits on 2x24 GB (and what this does):
  * the student's 24 x 32 expert matrices are FROZEN and stored int8 (absmax per output column and
    block of 32 inputs), dequantized per expert on the fly; everything else in fp32 (the converted
    student needs fp32 activations). With
    offload=True the int8 experts live in pinned host memory and stream to the GPU per layer
    (the GPUs here are shared with other jobs)
  * trainable: LoRA (rank r) on q/k/v/o of every layer, optional per-expert LoRA on down_proj,
    and in full: RMSNorm weights, attention sinks and biases, router weight+bias, expert biases
  * layers [0, split) live on cuda:0, the rest + lm_head on cuda:1; activations / gradients cross
    between the GPUs only through the CPU (direct GPU->GPU copies are broken on this box)
  * loss = KL(teacher || student) over the FULL vocabulary per token, teacher logits recomputed
    from stored teacher final hidden states (so no top-k truncation); optional final-hidden MSE
  * per-layer gradient checkpointing

The teacher's final hidden states for the training / held-out windows come from one streamed
teacher pass (approx.importance.stream).
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from approx.arch import (SCfg, TCfg, attend, apply_rope, moe, rmsnorm, student_rope, swiglu_oai, t_logits)


def q8(w, dim):
    """symmetric int8 along `dim` reduced (scale per remaining index)."""
    s = w.float().abs().amax(dim, keepdim=True).clamp_min(1e-12) / 127.0
    return (w.float() / s).round().clamp(-127, 127).to(torch.int8), s.to(torch.bfloat16)


def q8g(w, group=32):
    """group-wise symmetric int8 of (E, a, b) along a (one scale per `group` inputs and output
    column, like MXFP4's 32-blocks). Per-column int8 costs ~4% FFN error per converted layer (the
    columns also carry large weights for the residual stream's massive dims); groups of 32: ~1%."""
    E, a, b = w.shape
    g = group if a % group == 0 else a
    wg = w.float().view(E, a // g, g, b)
    sc = wg.abs().amax(2, keepdim=True).clamp_min(1e-12) / 127.0
    q = (wg / sc).round().clamp(-127, 127).to(torch.int8)
    return q.view(E, a, b), sc.squeeze(2).to(torch.bfloat16)


def dq8g(q, sc, dt):
    a, b = q.shape
    ng = sc.shape[0]
    return (q.view(ng, a // ng, b).to(dt) * sc[:, None, :].to(dt)).view(a, b)


class LoRA(nn.Module):
    def __init__(self, din, dout, r, dev):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, din, device=dev) / math.sqrt(din))
        self.B = nn.Parameter(torch.zeros(dout, r, device=dev))

    def delta(self):
        return self.B @ self.A


class SLayer(nn.Module):
    def __init__(self, lw, scfg: SCfg, dev, r=16, r_exp=0, dtype=torch.bfloat16, offload=False, int8=True):
        super().__init__()
        self.sc, self.dev, self.dt, self.type, self.offload = scfg, dev, dtype, lw["_type"], offload
        self.int8 = int8
        P = lambda k, train=True: nn.Parameter(lw[k].to(dev).float(), requires_grad=train)
        B = lambda k: lw[k].to(dev).to(dtype)
        for n in ("q", "k", "v", "o"):
            self.register_buffer(f"W{n}", B(f"self_attn.{n}_proj.weight"))
            setattr(self, f"b{n}", P(f"self_attn.{n}_proj.bias"))
            W = lw[f"self_attn.{n}_proj.weight"]
            setattr(self, f"L{n}", LoRA(W.shape[1], W.shape[0], r, dev) if r else None)
        self.sinks = P("self_attn.sinks")
        self.n_in = P("input_layernorm.weight")
        self.n_post = P("post_attention_layernorm.weight")
        self.rw, self.rb = P("mlp.router.weight"), P("mlp.router.bias")
        if int8:
            gu, s1 = q8g(lw["mlp.experts.gate_up_proj"])        # (E, h, 2I), scales (E, h/32, 2I)
            dn, s2 = q8g(lw["mlp.experts.down_proj"])           # (E, I, h),  scales (E, I/32, h)
        else:                                                   # numerically fragile layer: keep bf16
            gu, dn = lw["mlp.experts.gate_up_proj"].to(torch.bfloat16), lw["mlp.experts.down_proj"].to(torch.bfloat16)
            s1 = s2 = torch.zeros(1)
        if offload:     # int8 experts stay in pinned host memory and stream in per forward
            self.gu_host, self.dn_host = gu.pin_memory(), dn.pin_memory()
            self.gu = self.dn = None
        else:
            self.register_buffer("gu", gu.to(dev)); self.register_buffer("dn", dn.to(dev))
        self.register_buffer("gus", s1.to(dev)); self.register_buffer("dns", s2.to(dev))
        self.gub, self.dnb = P("mlp.experts.gate_up_proj_bias"), P("mlp.experts.down_proj_bias")
        E, I, H = lw["mlp.experts.down_proj"].shape
        self.LA = nn.Parameter(torch.randn(E, I, r_exp, device=dev) / math.sqrt(I)) if r_exp else None
        self.LB = nn.Parameter(torch.zeros(E, r_exp, H, device=dev)) if r_exp else None

    def W(self, n):
        w = getattr(self, f"W{n}")
        L = getattr(self, f"L{n}")
        return w + L.delta().to(w.dtype) if L is not None else w

    def forward(self, h, rope):
        sc, dt = self.sc, self.dt
        B_, T, H = h.shape
        x = rmsnorm(h, self.n_in, sc.eps)
        q = F.linear(x, self.W("q"), self.bq.to(dt)).view(B_, T, sc.nq, sc.hd).transpose(1, 2)
        k = F.linear(x, self.W("k"), self.bk.to(dt)).view(B_, T, sc.nkv, sc.hd).transpose(1, 2)
        v = F.linear(x, self.W("v"), self.bv.to(dt)).view(B_, T, sc.nkv, sc.hd).transpose(1, 2)
        cos, sin = rope
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        a = attend(q, k, v, sc.hd ** -0.5, sc.window if self.type == "sliding_attention" else None, self.sinks.to(dt))
        h = h + F.linear(a, self.W("o"), self.bo.to(dt))
        xf = rmsnorm(h, self.n_post, sc.eps).reshape(-1, H)
        logits = F.linear(xf.float(), self.rw, self.rb)
        tv, ti = logits.topk(sc.topk, -1)
        tw = tv.softmax(-1)

        GU = self.gu_host.to(self.dev) if self.offload else self.gu
        DN = self.dn_host.to(self.dev) if self.offload else self.dn

        def ef(e, xe):
            Wg = dq8g(GU[e], self.gus[e], dt) if self.int8 else GU[e].to(dt)
            Wd = dq8g(DN[e], self.dns[e], dt) if self.int8 else DN[e].to(dt)
            z = xe @ Wg + self.gub[e].to(dt)
            act = swiglu_oai(z[..., ::2], z[..., 1::2], sc.limit, sc.alpha)
            y = act @ Wd + self.dnb[e].to(dt)
            if self.LA is not None:
                y = y + (act @ self.LA[e].to(dt)) @ self.LB[e].to(dt)
            return y
        return h + moe(xf, ti, tw.to(dt), ef, sc.E).view(B_, T, H)

    def export(self):
        """trained values in canonical student names (LoRA merged)."""
        out = {}
        for n in ("q", "k", "v", "o"):
            out[f"self_attn.{n}_proj.weight"] = self.W(n).detach().float().cpu()
            out[f"self_attn.{n}_proj.bias"] = getattr(self, f"b{n}").detach().cpu()
        out.update({"self_attn.sinks": self.sinks.detach().cpu(), "input_layernorm.weight": self.n_in.detach().cpu(),
                    "post_attention_layernorm.weight": self.n_post.detach().cpu(), "mlp.router.weight": self.rw.detach().cpu(),
                    "mlp.router.bias": self.rb.detach().cpu(), "mlp.experts.gate_up_proj_bias": self.gub.detach().cpu(),
                    "mlp.experts.down_proj_bias": self.dnb.detach().cpu()})
        return out


class PipeStudent(nn.Module):
    """student split over devices; hops through the CPU."""

    def __init__(self, stu, devices=("cuda:0", "cuda:1"), split=None, r=16, r_exp=0, dtype=torch.bfloat16, T=512, log=print,
                 offload=False, probe=None, max_q_err=0.02):
        """probe: a few token windows; each layer's int8 FFN is compared with the stored bf16 one on
        the student's own stream and kept bf16 if its relative output error exceeds max_q_err."""
        super().__init__()
        sc = stu.scfg
        self.sc, self.devs, self.dt = sc, list(devices), dtype
        split = split if split is not None else (sc.L // 2 if len(self.devs) > 1 else sc.L)
        self.dev_of = [self.devs[0] if i < split else self.devs[-1] for i in range(sc.L)]
        g = stu.glob()
        self.embed = g["embed_tokens.weight"].to(dtype).cpu()                 # looked up on the CPU
        self.n_f = nn.Parameter(g["norm.weight"].float().to(self.devs[-1]))
        self.register_buffer("lm", g["lm_head.weight"].to(self.devs[-1]).to(dtype))
        self.layers = nn.ModuleList()
        self.ropes = {d: student_rope(sc, T, d) for d in set(self.dev_of)}
        hp = F.embedding(probe.cpu(), self.embed).float() if probe is not None else None
        self.q_err = []
        for i in range(sc.L):
            t0 = time.time()
            lw = stu.layer(i)
            int8, err = True, float("nan")
            if hp is not None:
                from approx.arch import s_attn, s_ffn
                d = self.dev_of[i]
                lwd = {k: (v.to(d).float() if torch.is_tensor(v) else v) for k, v in lw.items()}
                with torch.no_grad():
                    hm = hp.to(d) + s_attn(lwd, sc, hp.to(d), self.ropes[d])
                    ref = s_ffn(lwd, sc, hm)
                    q = {**lwd}
                    for k in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                        qq, ss = q8g(lwd[k].cpu())
                        q[k] = torch.stack([dq8g(qq[e], ss[e], torch.float32) for e in range(qq.shape[0])]).to(d)
                    err = ((s_ffn(q, sc, hm) - ref).norm() / ref.norm()).item()
                    hp = (hm + ref).cpu()
                del lwd, q
                int8 = err <= max_q_err
            self.q_err.append(err)
            self.layers.append(SLayer(lw, sc, self.dev_of[i], r, r_exp, dtype, offload=offload, int8=int8))
            log(f"  student layer {i} -> {self.dev_of[i]} ({time.time() - t0:.1f}s)  int8 FFN err {err:.4f}"
                f"{'' if int8 else '  -> kept bf16'}")

    def forward(self, ids, ckpt=True):
        """returns final normed hidden on the last device. Cross-device hops are explicit
        autograd-preserving .cpu().to(dev) copies."""
        h = F.embedding(ids.cpu(), self.embed).to(self.dev_of[0])
        cur = self.dev_of[0]
        for i, layer in enumerate(self.layers):
            d = self.dev_of[i]
            if d != cur:
                h = h.cpu().to(d)                                              # autograd tracks both copies
                cur = d
            h = checkpoint(layer, h, self.ropes[d], use_reentrant=False) if (ckpt and self.training) else layer(h, self.ropes[d])
        if cur != self.devs[-1]:
            h = h.cpu().to(self.devs[-1])
        return rmsnorm(h, self.n_f, self.sc.eps)

    def logits(self, x):
        return F.linear(x.to(self.dt), self.lm).float()


class TeacherHead:
    """teacher logits from stored teacher final hidden states."""

    def __init__(self, tcfg: TCfg, glob, dev):
        self.cfg, self.dev = tcfg, dev
        self.g = {k: v.to(dev) for k, v in glob.items()}

    def logprobs(self, hT):
        return t_logits(self.g, self.cfg, hT.to(self.dev)).log_softmax(-1)


def kl_chunked(model: PipeStudent, th: TeacherHead, xs, hT, chunk=512, grad=True):
    """sum over tokens of KL(teacher || student); xs (n, H) student normed final hidden, hT (n, Ht)."""
    tot = 0.0
    n = xs.shape[0]

    def f(xc, ht):
        lt = th.logprobs(ht)
        ls = model.logits(xc).log_softmax(-1)
        return (lt.exp() * (lt - ls)).sum()
    out = []
    for i in range(0, n, chunk):
        xc, ht = xs[i:i + chunk], hT[i:i + chunk]
        out.append(checkpoint(f, xc, ht, use_reentrant=False) if grad else f(xc, ht))
    return torch.stack(out).sum()


@torch.no_grad()
def evaluate(model: PipeStudent, th: TeacherHead, ids, hT, bs=4, chunk=512):
    model.eval()
    kl = agree = nll = 0.0
    n = 0
    for b in range(0, ids.shape[0], bs):
        x = model(ids[b:b + bs], ckpt=False)[:, :-1].reshape(-1, model.sc.h)
        ht = hT[b:b + bs, :-1].reshape(-1, hT.shape[-1])
        y = ids[b:b + bs, 1:].reshape(-1).to(x.device)
        for i in range(0, x.shape[0], chunk):
            lt = th.logprobs(ht[i:i + chunk])
            ls = model.logits(x[i:i + chunk]).log_softmax(-1)
            kl += (lt.exp() * (lt - ls)).sum().item()
            agree += (lt.argmax(-1) == ls.argmax(-1)).sum().item()
            nll += F.nll_loss(ls, y[i:i + chunk], reduction="sum").item()
        n += x.shape[0]
    model.train()
    return {"kl": kl / n, "top1": agree / n, "ppl": math.exp(nll / n)}


def distill(model: PipeStudent, th: TeacherHead, train_ids, train_hT, eval_ids, eval_hT=None, steps=200, bs=4,
            lr=2e-4, lr_other=2e-5, eval_every=25, hidden_w=0.0, log=print, seed=0):
    """AdamW on LoRA (lr) and the small full parameters (lr_other). eval_ids: a tensor (with
    eval_hT) or a dict name -> (ids, hT). Returns the eval curve."""
    evals = eval_ids if isinstance(eval_ids, dict) else {"eval": (eval_ids, eval_hT)}
    ev = lambda: {k: v for name, (i_, h_) in evals.items()
                  for k, v in [(f"{name}_{m}" if len(evals) > 1 else m, x) for m, x in evaluate(model, th, i_, h_).items()]}
    lora = [p for n, p in model.named_parameters() if p.requires_grad and (".L" in n or n.startswith("L") or ".LA" in n or ".LB" in n)]
    lid = {id(p) for p in lora}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in lid]
    opt = torch.optim.AdamW([{"params": lora, "lr": lr}, {"params": other, "lr": lr_other}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.1, 1 - s / steps))
    g = torch.Generator().manual_seed(seed)
    curve = [{"step": 0, **ev()}]
    log(f"step 0 {curve[-1]}")
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, train_ids.shape[0], (bs,), generator=g)
        ids = train_ids[idx]
        x = model(ids)[:, :-1].reshape(-1, model.sc.h)
        ht = train_hT[idx][:, :-1].reshape(-1, train_hT.shape[-1])
        loss = kl_chunked(model, th, x, ht) / x.shape[0]
        if hidden_w:
            Ht = ht.shape[-1]
            tn = rmsnorm(ht.to(x.device).float(), None, 1e-6)
            loss = loss + hidden_w * (x[:, :Ht].float() / model.n_f[:Ht].detach().clamp_min(1e-6) - tn).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        if step % 5 == 0:
            log(f"step {step} train KL {loss.item():.4f} ({(time.time() - t0) / step:.2f}s/step)")
        if step % eval_every == 0 or step == steps:
            curve.append({"step": step, "train_kl": loss.item(), **ev()})
            log(f"step {step} {curve[-1]}")
    return curve
