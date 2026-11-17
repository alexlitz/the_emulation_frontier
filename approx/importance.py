"""Teacher-side passes over a calibration set, streamed layer by layer:

  stream()        the teacher's residual stream h[0..L] (inputs of every layer + final), on CPU
  layer_ablation  per-layer ablation dKL: KL(teacher || teacher with layer l's updates removed),
                  all 30 variants advanced together so each layer's weights are read once
  attribution()   gradient x activation ("attribution patching") scores from one reverse
                  layer-by-layer pass: per q-head, per expert, per expert unit, per dense-MLP unit,
                  plus routing frequency / routed mass / co-activation per expert.

The scores are what the conversion spends its capacity on: computed on a *domain* calibration
set they say what that domain uses. |a . dL/da| per token (summed, /tokens) estimates the loss
change from zero-ablating component a for that token (first order).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from approx.arch import (TCfg, attend, apply_rope, gelu_tanh, rmsnorm, rope_cs, t_embed, t_layer,
                         t_logits, t_router, teacher_inv_freq)


class Teacher:
    """Teacher weights: an in-memory dict (tiny) or a local safetensors checkpoint (real), loaded
    one layer at a time onto `device`."""

    def __init__(self, cfg: TCfg, weights=None, path=None, device="cpu"):
        self.cfg, self.w, self.device = cfg, weights, device
        self.files = {}
        if path is not None:
            import json, os
            from serving.st_reader import canon
            idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
            for n, f in idx.items():
                if "vision" in n or "audio" in n or "embed_vision" in n or "embed_audio" in n:
                    continue
                self.files[canon(n)] = (os.path.join(path, f), n)
        self._open = {}

    def _get(self, name):
        if self.w is not None:
            return self.w[name]
        from safetensors import safe_open
        f, n = self.files[name]
        if f not in self._open:
            self._open[f] = safe_open(f, framework="pt", device="cpu")
        return self._open[f].get_tensor(n)

    def layer(self, li, device=None):
        p = f"layers.{li}."
        names = [k for k in (self.w.keys() if self.w is not None else self.files.keys()) if k.startswith(p)]
        return {k[len(p):]: self._get(k).to(device or self.device) for k in names}

    def glob(self, device=None):
        return {k: self._get(k).to(device or self.device) for k in ("embed_tokens.weight", "norm.weight")}


def _batches(ids, bs):
    for i in range(0, ids.shape[0], bs):
        yield i, ids[i:i + bs]


@torch.no_grad()
def stream(teacher: Teacher, ids, bs=4, store_dtype=None):
    """h[l] for l = 0..L (CPU). ids (N, T)."""
    cfg, dev = teacher.cfg, teacher.device
    g = teacher.glob()
    hs = [torch.cat([t_embed(g, cfg, b.to(dev)).cpu() for _, b in _batches(ids, bs)])]
    for li in range(cfg.L):
        lw = teacher.layer(li)
        out = [t_layer(lw, cfg, li, hs[-1][i:i + bs].to(dev)).cpu() for i in range(0, ids.shape[0], bs)]
        hs.append(torch.cat(out).to(store_dtype or out[0].dtype))
        del lw
    return hs


def kl_rows(lp_ref, lp):
    """mean over rows of KL(ref || other), both log-probs."""
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1)


@torch.no_grad()
def final_logprobs(teacher: Teacher, h, chunk=1024):
    cfg, g = teacher.cfg, teacher.glob()
    out = []
    flat = h.reshape(-1, h.shape[-1])
    for i in range(0, flat.shape[0], chunk):
        out.append(t_logits(g, cfg, flat[i:i + chunk].to(teacher.device)).log_softmax(-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def layer_ablation(teacher: Teacher, ids, bs=4, hs=None, chunk=512):
    """dKL[l] = mean_token KL(teacher || teacher without layer l's attention+FFN updates)
    (its layer_scalar kept: removing a layer from the student drops its updates, the residual
    scale is bookkept separately)."""
    cfg, dev = teacher.cfg, teacher.device
    g = teacher.glob()
    base = torch.cat([t_embed(g, cfg, b.to(dev)).cpu() for _, b in _batches(ids, bs)])
    var = {}
    for li in range(cfg.L):
        lw = teacher.layer(li)
        run = lambda h: torch.cat([t_layer(lw, cfg, li, h[i:i + bs].to(dev)).cpu() for i in range(0, h.shape[0], bs)])
        for l in var:
            var[l] = run(var[l])
        var[li] = (base.to(dev) * lw["layer_scalar"].to(base.dtype)).cpu()
        base = run(base)
        del lw
    T = base.shape[-2]
    ref = final_logprobs(teacher, base, chunk)
    return torch.tensor([kl_rows(ref, final_logprobs(teacher, var[l], chunk)).mean().item() for l in range(cfg.L)])


# ----------------------------------------------------------------------------- attribution

def _t_layer_instr(lw, cfg: TCfg, li, h):
    """Teacher layer with handles on the components we attribute. Returns (out, parts) where every
    part is a tensor in the graph whose (value . grad) we read."""
    lt = cfg.layer_types[li]
    hd, nkv, keqv = cfg.head(li)
    B, T, H = h.shape
    parts = {}
    x = rmsnorm(h, lw["input_layernorm.weight"], cfg.eps)
    q = (x @ lw["self_attn.q_proj.weight"].T).view(B, T, cfg.nq, hd)
    k = (x @ lw["self_attn.k_proj.weight"].T).view(B, T, nkv, hd)
    v = k if keqv else (x @ lw["self_attn.v_proj.weight"].T).view(B, T, nkv, hd)
    cos, sin = rope_cs(teacher_inv_freq(cfg, lt), T, h.device)
    qr = apply_rope(rmsnorm(q, lw["self_attn.q_norm.weight"], cfg.eps).transpose(1, 2), cos, sin)
    kr = apply_rope(rmsnorm(k, lw["self_attn.k_norm.weight"], cfg.eps).transpose(1, 2), cos, sin)
    vn = rmsnorm(v, None, cfg.eps).transpose(1, 2)
    a = attend(qr, kr, vn, 1.0, cfg.window if lt == "sliding_attention" else None)
    parts["heads"] = a                                            # (B,T,nq*hd)
    o = a @ lw["self_attn.o_proj.weight"].T
    h = h + rmsnorm(o, lw["post_attention_layernorm.weight"], cfg.eps)
    r = h
    xd = rmsnorm(r, lw["pre_feedforward_layernorm.weight"], cfg.eps)
    dact = gelu_tanh(xd @ lw["mlp.gate_proj.weight"].T) * (xd @ lw["mlp.up_proj.weight"].T)
    parts["dense"] = dact                                         # (B,T,Id)
    m1 = dact @ lw["mlp.down_proj.weight"].T
    rf = r.reshape(-1, H)
    _, p, tw, ti = t_router(lw, cfg, rf)
    x2 = rmsnorm(rf, lw["pre_feedforward_layernorm_2.weight"], cfg.eps)
    gu, dn = lw["experts.gate_up_proj"], lw["experts.down_proj"]
    m2 = torch.zeros_like(rf, dtype=torch.float32)
    eparts = {}
    flat = ti.reshape(-1)
    for e in flat.unique().tolist():
        tok, kk = (ti == e).nonzero(as_tuple=True)
        gg, uu = (x2[tok] @ gu[e].T).chunk(2, -1)
        act = gelu_tanh(gg) * uu
        y = (act @ dn[e].T).float() * tw[tok, kk, None]
        eparts[e] = (tok, act, y)
        m2 = m2.index_add(0, tok, y)
    parts["experts"] = eparts
    m2 = m2.to(h.dtype).view(B, T, H)
    m1n = rmsnorm(m1, lw["post_feedforward_layernorm_1.weight"], cfg.eps)
    m2n = rmsnorm(m2, lw["post_feedforward_layernorm_2.weight"], cfg.eps)
    f = rmsnorm(m1n + m2n, lw["post_feedforward_layernorm.weight"], cfg.eps)
    out = (r + f) * lw["layer_scalar"].to(h.dtype)
    stats = dict(p=p.detach(), ti=ti.detach(), tw=tw.detach())
    return out, parts, stats


def _retain(parts):
    parts["heads"].retain_grad()
    parts["dense"].retain_grad()
    for e, (tok, act, y) in parts["experts"].items():
        act.retain_grad()
        y.retain_grad()


def attribution(teacher: Teacher, ids, hs=None, bs=2, chunk=256):
    """grad x activation scores (mean over tokens of |a . dL/da|) with L = next-token NLL of the
    teacher on `ids`, from one reverse layer-by-layer pass (each layer recomputed under autograd
    from its stored input). Also routing frequency / mass / co-activation per expert."""
    cfg, dev = teacher.cfg, teacher.device
    if hs is None:
        hs = stream(teacher, ids, bs)
    N, T = ids.shape
    ntok = N * T
    g = teacher.glob()
    # dL/dh_L
    grads = []
    for i in range(0, N, bs):
        h = hs[-1][i:i + bs].to(dev).float().requires_grad_(True)
        z = t_logits({k: v.float() for k, v in g.items()}, cfg, h)
        tgt = ids[i:i + bs, 1:].to(dev)
        loss = F.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]), tgt.reshape(-1), reduction="sum") / ntok
        loss.backward()
        grads.append(h.grad.cpu())
    gcur = torch.cat(grads)
    L = cfg.L
    res = {"head": torch.zeros(L, cfg.nq), "expert": torch.zeros(L, cfg.E),
           "unit": torch.zeros(L, cfg.E, cfg.Ie), "dense": torch.zeros(L, cfg.Id),
           "freq": torch.zeros(L, cfg.E), "mass": torch.zeros(L, cfg.E), "coact": torch.zeros(L, cfg.E, cfg.E)}
    for li in reversed(range(L)):
        lw = teacher.layer(li)
        hd = cfg.head(li)[0]
        newg = []
        for i in range(0, N, bs):
            hin = hs[li][i:i + bs].to(dev).requires_grad_(True)
            out, parts, st = _t_layer_instr(lw, cfg, li, hin)
            _retain(parts)
            out.backward(gcur[i:i + bs].to(dev).to(out.dtype))
            newg.append(hin.grad.float().cpu())
            torch.set_grad_enabled(False)
            a, ga = parts["heads"].detach(), parts["heads"].grad
            Bq = a.shape[0]
            ah = (a.float() * ga.float()).view(Bq, T, cfg.nq, hd).sum(-1).abs().sum((0, 1))
            res["head"][li] += ah.cpu()
            res["dense"][li] += (parts["dense"].detach().float() * parts["dense"].grad.float()).abs().sum((0, 1)).cpu()
            for e, (tok, act, y) in parts["experts"].items():
                res["unit"][li, e] += (act.detach().float() * act.grad.float()).abs().sum(0).cpu()
                res["expert"][li, e] += (y.detach().float() * y.grad.float()).sum(-1).abs().sum().cpu()
            ti, tw = st["ti"], st["tw"]
            oh = torch.zeros(ti.shape[0], cfg.E, device=ti.device).scatter_(1, ti, tw / tw.sum(-1, keepdim=True))
            res["freq"][li] += (oh > 0).float().sum(0).cpu()
            res["mass"][li] += oh.sum(0).cpu()
            res["coact"][li] += (oh.T @ oh).cpu()
            torch.set_grad_enabled(True)
            del out, parts
        gcur = torch.cat(newg)
        del lw
    for k in res:
        res[k] /= ntok
    return res
