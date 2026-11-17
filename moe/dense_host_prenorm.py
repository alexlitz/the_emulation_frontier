"""Dense host for a real MoE target UNDER A REAL PRE-NORM: the construction and its measured agreement.

The norm-free construction (`moe_two_layer_margin`, Lean) spends two FFN layers per MoE layer: a select
FFN writes s_e = sum_a relu(z_a - z_e) into E carried channels (0 for the winner), then an expert FFN of
masked units relu(p - lambda s_e). Under a pre-norm the carried s cannot be read exactly
(`select_carry_prenorm_fails`): it enters the next RMSNorm's mean square, so the expert layer reads the
target's normalised stream scaled by a token-dependent factor, and it reads s itself divided by the
token's rms.

The pre-norm construction here keeps the two layers and fixes both readings:

  * the carry is SMALL: the select FFN writes gamma * m_e with gamma = 1e-8, so |carry|^2 / |h|^2 ~ 1e-12
    and the expert layer's normalised stream is the target's to fp32 noise (at gamma = 1e-4 the first
    layers, where |h| is small, already drift);
  * the mask is a SCALE-ROBUST THRESHOLD: m_e = sum_a t_a^2 sigma(beta t_a), t_a = z_a - z_e (two SwiGLU
    units per pair: silu(beta t) * t / beta, bias-free), is ~0 for the winner and >= margin^2 for every
    loser; the expert layer shifts each of expert e's gate pre-activations by -(Lambda / gamma) * c_e / r
    (r = the token's rms), so only the sign/size of c_e / r matters, never its exact value;
  * the carry is ERASED in the same expert FFN by recomputing m from the expert layer's own input
    (2 E(E-1) more units writing -gamma m'). The residue is gamma |m| O(|carry|^2 / |h|^2), i.e. O(gamma^3):
    the next layer reads it at strength Lambda / gamma, so a small gamma is what keeps a stale loser
    channel from masking the next layer's winner;
  * the host RMSNorm over d + E channels uses eps' = eps d / (d + E) and weight w sqrt(d / (d + E)): with
    the carried channels zero it equals the target's norm exactly (the learned-abs padding identity).

Host cost per target MoE layer: 2 layers (the target attention + the select FFN; a zero attention + the
expert FFN), E extra residual channels, E*e_w + 2 E(E-1) FFN units (2 E(E-1) select units in the first
FFN, E*e_w expert units + 2 E(E-1) erase units in the second). The unit gate is exact for TOP-1
renormalised routing (gate 1). A top-k target (k >= 2) also needs (a) a top-k MEMBERSHIP mask, not
one-layer separable bias-free (relu(min(t_a, t_b)) is not a one-hidden-layer function), and (b) the gate
weight, a third multiplicative factor a SwiGLU unit silu(a) * b does not have; neither survives being
carried (any carried value is read divided by the token's rms). The `oracle` variant supplies both
(membership by the k-th logit, the target's own gate weights) to isolate what the carry machinery
costs; it is an upper bound, not a host.

Variants (each one a full host simulation: every RMSNorm reads the carried channels, generation runs
the host end to end):
  self         the target itself (control)
  top1         the realizable construction: unit-gate top-1 masks (exact for a top-1 target)
  top1-carry1  as top1 with gamma = 1 (the norm-free construction's carry, verbatim scale)
  top1-noerase as top1 without the erase units (the carry accumulates over layers)
  oracle       top-k membership mask + the target's gate weights (not realizable; upper bound)
  uniform      top-k membership mask (oracle) + gate 1/k (what masks alone give a top-k target)
  stairL       top-k membership (oracle) + the gate rounded to 1/L: what an L-level staircase of masked
               expert copies (L x the expert units) would give

On the ORIGINAL (target) model's greedy 64-token continuations of 24 prompts, fp32, TF32 off:
top1, kl, ex32, ex64, div as in act/act_swap_agreement.py. Router margins (z_(1) - z_(2) and the top-k
boundary z_(k) - z_(k+1)) are recorded on the reference pass.

  python -m moe.dense_host_prenorm ggml-org/stories15M_MOE --story --device cuda:0
  python -m moe.dense_host_prenorm AlexKoff88/gpt-oss-tinystories --story --device cuda:0
  python -m moe.dense_host_prenorm allenai/OLMoE-1B-7B-0924 --device cpu --margins-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from act.act_swap_agreement import N_NEW, PROMPTS, STORY_PROMPTS, cont_logp, greedy

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

GAMMA = 1e-8        # carry scale (relative to the residual: |carry|^2 / |h|^2 ~ 1e-12)
LAMBDA = 1e12       # mask strength: shift = (LAMBDA / GAMMA) * c / r on every gate pre-activation
BETA = 1e7          # select-unit sharpness: t^2 sigma(beta t). Soft zone: router margins |t| <~ 1e-5
#                     (the moderate setting LAMBDA = 1e6, BETA = 2e3 has a soft zone of ~1e-2)
DELTAS = (1e-3, 1e-2, 3e-2, 1e-1)

VARIANTS = {          # name: (selection, gate, gamma, erase)
    "top1": ("top1", "unit", GAMMA, True),
    "top1-carry1": ("top1", "unit", 1.0, True),
    "top1-noerase": ("top1", "unit", GAMMA, False),
    "oracle": ("topk", "target", GAMMA, True),
    "uniform": ("topk", "uniform", GAMMA, True),
    # the gate as a staircase of L levels (L masked copies of every expert, one threshold mask per level):
    # the target's gate rounded to a multiple of 1/L, membership still the oracle's
    **{f"stair{L}": ("topk", f"stair{L}", GAMMA, True) for L in (2, 4, 8, 16, 64)},
}


# --- the MoE families: router logits, the target's gates, every expert with a gate shift ----------------
class Adapter:
    """Uniform access to one MoE block. experts(x, shift) -> (N, E, d): every expert's output with
    shift[:, e] subtracted from all of expert e's gate pre-activations (the masked dense host units)."""

    def __init__(self, block, kind, cfg):
        self.b, self.kind = block, kind
        self.norm_topk = bool(getattr(cfg, "norm_topk_prob", kind in ("mixtral", "gpt_oss")))
        if kind == "gpt_oss":
            self.E = block.router.num_experts
        else:
            self.E = block.gate.out_features if hasattr(block.gate, "out_features") else block.gate.weight.shape[0]
        self.shared = getattr(block, "shared_expert", None)

    @property
    def k(self):
        return self.b.router.top_k if self.kind == "gpt_oss" else self.b.top_k

    @k.setter
    def k(self, v):
        if self.kind == "gpt_oss":
            self.b.router.top_k = v
        else:
            self.b.top_k = v

    def logits(self, x):
        if self.kind == "gpt_oss":
            return F.linear(x, self.b.router.weight, self.b.router.bias)
        return self.b.gate(x)

    def gates(self, z, k):
        """The target's dense (N, E) gate weights for top-k routing."""
        if self.kind == "gpt_oss":                            # top-k on logits, softmax over the k
            v, i = torch.topk(z, k, -1)
            return torch.zeros_like(z).scatter_(1, i, F.softmax(v, -1))
        p = F.softmax(z, -1, dtype=torch.float)
        v, i = torch.topk(p, k, -1)
        if self.norm_topk:
            v = v / v.sum(-1, keepdim=True)
        return torch.zeros_like(z).scatter_(1, i, v.to(z.dtype))

    def experts(self, x, shift):
        ex = self.b.experts
        if self.kind == "gpt_oss":
            gu = torch.einsum("nd,edh->enh", x, ex.gate_up_proj) + ex.gate_up_proj_bias[:, None, :]
            gate, up = gu[..., ::2] - shift.T[:, :, None], gu[..., 1::2]
            gate = gate.clamp(max=ex.limit)
            up = up.clamp(min=-ex.limit, max=ex.limit)
            hid = (up + 1) * (gate * torch.sigmoid(gate * ex.alpha))
            out = torch.einsum("enh,ehd->end", hid, ex.down_proj)
            # the expert's down bias rides on one more masked unit: glu(L - shift) / glu(L), L = the clamp
            def glu(g):
                return g * torch.sigmoid(g * ex.alpha)
            L = torch.tensor(ex.limit, dtype=x.dtype, device=x.device)
            on = glu((L - shift.T).clamp(max=ex.limit)) / glu(L)
            out = out + on[:, :, None] * ex.down_proj_bias[:, None, :]
            return out.permute(1, 0, 2)
        outs = []
        for e, m in enumerate(ex):
            if self.kind == "mixtral":
                g, u, dn = m.w1, m.w3, m.w2
            else:
                g, u, dn = m.gate_proj, m.up_proj, m.down_proj
            outs.append(dn(F.silu(g(x) - shift[:, e:e + 1]) * u(x)))
        return torch.stack(outs, 1)

    def shared_out(self, x):
        if self.shared is None:
            return 0.0
        out = self.shared(x)
        sg = getattr(self.b, "shared_expert_gate", None)
        return out * torch.sigmoid(sg(x)) if sg is not None else out


def moe_blocks(model):
    """[(decoder layer, attribute name, Adapter)] for every MoE layer."""
    cfg = getattr(model.config, "text_config", model.config)
    kind = cfg.model_type
    out = []
    for layer in model.model.layers:
        for attr in ("block_sparse_moe", "mlp"):
            blk = getattr(layer, attr, None)
            if blk is not None and (hasattr(blk, "experts")):
                out.append((layer, attr, Adapter(blk, "gpt_oss" if kind == "gpt_oss" else
                                                 ("mixtral" if kind == "mixtral" else "glu"), cfg)))
                break
    return out


# --- the host: carried channels, host norms, the two-layer MoE replacement -----------------------------
class Side:
    """The host's extra residual channels (N tokens x E), shared by every module of one forward."""

    def __init__(self, n_ch):
        self.n_ch, self.c, self.h, self.r = n_ch, None, None, None
        self.stats = dict(max_c=0.0, soft=0, n=0)

    def reset(self, shape, device):
        self.c = torch.zeros(*shape[:-1], self.n_ch, device=device, dtype=torch.float32)


class HostNorm(torch.nn.Module):
    """RMSNorm over [h; c] (d + n_ch channels, eps' = eps d / (d + n), weight w sqrt(d / (d + n))):
    algebraically w h / sqrt((|h|^2 + |c|^2) / d + eps)."""

    def __init__(self, orig, side):
        super().__init__()
        self.orig, self.side = orig, side
        self.w = orig.weight
        self.eps = getattr(orig, "variance_epsilon", getattr(orig, "eps", 1e-6))

    def forward(self, h):
        hf = h.float()
        ms = hf.pow(2).sum(-1, keepdim=True)
        if self.side.c is not None:
            ms = ms + self.side.c.pow(2).sum(-1, keepdim=True)
        r = torch.sqrt(ms / hf.shape[-1] + self.eps)
        self.side.h, self.side.r = hf, r
        return (self.w * (hf / r)).to(h.dtype)


def select_m(z, mode, k, beta):
    """The select FFN's write (before gamma): (N, E), ~0 on the kept experts, >= margin^2 elsewhere.
    top1: sum_a t^2 sigma(beta t), t = z_a - z_e (SwiGLU units silu(beta t) t / beta, 2 E(E-1) units).
    topk: t = z_(k) - z_e (the k-th logit: NOT one-layer computable; the oracle)."""
    if mode == "top1":
        t = z[:, None, :] - z[:, :, None]                    # [n, e, a] = z_a - z_e
        return (t.pow(2) * torch.sigmoid(beta * t)).sum(-1)  # the a = e term is 0
    zk = torch.topk(z, k, -1).values[:, -1:]
    t = zk - z
    return t.pow(2) * torch.sigmoid(beta * t)


class HostMoE(torch.nn.Module):
    """Replaces a target MoE block by the dense host's two layers (select FFN, zero attention, expert
    FFN), carrying gamma * m in the side channels. Returns the (h-channel) update of both layers."""

    def __init__(self, ad, side, variant, k_target, lam=LAMBDA, beta=BETA, check=False):
        super().__init__()
        self.ad, self.side, self.k = ad, side, k_target
        self.sel, self.gate, self.gamma, self.erase = VARIANTS[variant]
        self.lam, self.beta, self.check = lam, beta, check
        self.err = [0.0, 0.0, 0]          # max |host - target|, sum sq, count (block level, if check)

    def forward(self, x1):
        shp = x1.shape
        side = self.side
        x1 = x1.reshape(-1, shp[-1]).float()
        r1 = side.r.reshape(-1, 1)
        c = side.c.reshape(-1, side.n_ch)
        d = shp[-1]
        # host layer 2l: the select FFN (reads x1) writes gamma m into the E carried channels
        z1 = self.ad.logits(x1)
        m = self.gamma * select_m(z1, self.sel, self.k, self.beta)
        c_mid = c + m
        # host layer 2l+1: zero attention, then its RMSNorm over [h; c_mid] -> the expert FFN's input
        r2 = torch.sqrt(r1.pow(2) + (c_mid.pow(2).sum(-1, keepdim=True) - c.pow(2).sum(-1, keepdim=True)) / d)
        x2 = x1 * (r1 / r2)
        shift = (self.lam / self.gamma) * c_mid / r2
        Y = self.ad.experts(x2, shift)                                  # (N, E, d)
        if self.gate == "unit":
            w = torch.ones_like(z1)
        elif self.gate == "uniform":
            w = torch.full_like(z1, 1.0 / self.k)
        else:                                                           # the target's gate (oracle)
            g = self.ad.gates(z1, self.k)
            if self.gate.startswith("stair"):                           # L-level staircase of the gate
                L = int(self.gate[5:])
                g = torch.round(g * L) / L
            w = torch.where(self.ad.gates(z1, self.k) > 0, g, torch.ones_like(g))   # unselected: leakage at 1
        out = (w[:, :, None] * Y).sum(1) + self.ad.shared_out(x2)
        if self.erase:                                                  # E^2 units: -gamma m(x2)
            c_mid = c_mid - self.gamma * select_m(self.ad.logits(x2), self.sel, self.k, self.beta)
        side.c = c_mid.reshape(*shp[:-1], side.n_ch)
        side.stats["max_c"] = max(side.stats["max_c"], c_mid.abs().max().item())
        if self.check:
            ref = target_moe(self.ad, x1, self.k)
            e = (out - ref).abs()
            self.err[0] = max(self.err[0], e.max().item())
            self.err[1] += e.pow(2).sum().item()
            self.err[2] += e.numel()
        return out.reshape(shp).to(x1.dtype), z1


def target_moe(ad, x, k):
    """The target MoE block's own output (gates x experts, + shared expert)."""
    g = ad.gates(ad.logits(x), k)
    Y = ad.experts(x, torch.zeros_like(g))
    return (g[:, :, None] * Y).sum(1) + ad.shared_out(x)


class Installed:
    """Swap a model to the host (HostNorm everywhere, HostMoE per MoE layer) and back."""

    def __init__(self, model, blocks, variant, k_target, check=False, lam=LAMBDA, beta=BETA):
        self.model, self.blocks = model, blocks
        self.side = Side(blocks[0][2].E)
        self.saved = []
        self.hosts = []
        inner = model.model
        for layer in inner.layers:
            for a in ("input_layernorm", "post_attention_layernorm"):
                self.saved.append((layer, a, getattr(layer, a)))
                setattr(layer, a, HostNorm(getattr(layer, a), self.side))
        self.saved.append((inner, "norm", inner.norm))
        inner.norm = HostNorm(inner.norm, self.side)
        for layer, attr, ad in blocks:
            self.saved.append((layer, attr, getattr(layer, attr)))
            hm = HostMoE(ad, self.side, variant, k_target, lam=lam, beta=beta, check=check)
            self.hosts.append(hm)
            setattr(layer, attr, hm)
        side = self.side
        self.hook = inner.embed_tokens.register_forward_hook(lambda m, i, o: side.reset(o.shape, o.device))

    def remove(self):
        self.hook.remove()
        for obj, a, mod in reversed(self.saved):
            setattr(obj, a, mod)


# --- measurement ----------------------------------------------------------------------------------------
def margin_stats(zs, k):
    """zs: list of (N, E) router logits. Top-1 margin z_(1) - z_(2) and top-k boundary z_(k) - z_(k+1)."""
    z = torch.cat([q.reshape(-1, q.shape[-1]).double() for q in zs], 0)
    srt = torch.sort(z, -1, descending=True).values
    out = {}
    for name, (a, b) in (("m1", (0, 1)), ("mk", (k - 1, k))):
        if b >= srt.shape[1]:
            continue
        g = (srt[:, a] - srt[:, b]).cpu().numpy()
        out[name] = dict(q01=float(np.quantile(g, 0.01)), q05=float(np.quantile(g, 0.05)),
                         median=float(np.median(g)), **{f"lt{dl:g}": float((g < dl).mean()) for dl in DELTAS},
                         min=float(g.min()), n=int(g.size))
    return out


def load(model_id, device, dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from hf.attn_pin import attn_impl
    cfg = AutoConfig.from_pretrained(model_id)
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and cfg.model_type == "gpt_oss":      # a transformers-5 config read by 4.x
        cfg.rope_theta, cfg.rope_scaling = rp.get("rope_theta", 10000.0), None
    tok = AutoTokenizer.from_pretrained(model_id)
    for attempt in range(30):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, config=cfg, dtype=dtype,  # eager: hf/attn_pin.py
                                                         attn_implementation=attn_impl(cfg)).to(device).eval()
            return model, tok
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            time.sleep(60)
    raise RuntimeError("GPU busy")


@torch.no_grad()
def score(model, enc, ref_seq, ref_lp, ref_cont):
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
    return dict(top1=top1 / n, kl=kl / n, ex32=ex32 / len(enc), ex64=ex64 / len(enc), div=float(np.mean(div)))


@torch.no_grad()
def score_tf(model, enc, ref_seq, ref_lp):
    """Teacher-forced metrics only (top1, kl): for targets too large to generate with in fp32."""
    top1 = kl = 0.0
    n = 0
    for ids, s, lq in zip(enc, ref_seq, ref_lp):
        lp = cont_logp(model, s, ids.shape[1])
        kl += (lq.exp() * (lq - lp)).sum().item()
        top1 += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
        n += lp.shape[0]
    return dict(top1=top1 / n, kl=kl / n, ex32=None, ex64=None, div=None)


@torch.no_grad()
def run(model_id, device="cuda:0", story=False, n_prompts=24, ks=None, variants=None, margins_only=False,
        sink=None, lam=LAMBDA, beta=BETA, ref_device=None):
    """ref_device: generate the 64-token continuations with a bf16 copy there (the fp32 target on `device`,
    e.g. the CPU, then only scores them teacher-forced: top1 and kl, no ex32 / ex64)."""
    model, tok = load(model_id, device)
    blocks = moe_blocks(model)
    rmodel = rblocks = None
    if ref_device:
        rmodel, _ = load(model_id, ref_device, dtype=torch.bfloat16)
        rblocks = moe_blocks(rmodel)
    assert blocks, "no MoE block found"
    E, k0 = blocks[0][2].E, blocks[0][2].k
    prompts = (STORY_PROMPTS if story else PROMPTS)[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    rows = []
    for k in ks or (k0, 1):
        for _, _, ad in blocks + (rblocks or []):
            ad.k = k
        t0 = time.time()
        if rmodel is not None:
            ref_cont = [greedy(rmodel, ids.to(ref_device)).to(device) for ids in enc]
        else:
            ref_cont = [greedy(model, ids) for ids in enc]
        ref_seq = [torch.cat([ids, c], 1) for ids, c in zip(enc, ref_cont)]
        ref_lp = [cont_logp(model, s, ids.shape[1]) for s, ids in zip(ref_seq, enc)]
        zs = []
        hooks = []
        for _, _, ad in blocks:                                          # router logits, reference pass
            mod = ad.b.router if ad.kind == "gpt_oss" else ad.b.gate
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, ad=ad: zs.append(
                    (ad.logits(i[0].reshape(-1, i[0].shape[-1])) if ad.kind == "gpt_oss" else o).float().cpu())))
        for s in ref_seq:
            model(s)
        for hk in hooks:
            hk.remove()
        ms = margin_stats(zs, k)
        base = dict(model=model_id, E=E, k_native=k0, k=k, n_prompts=len(enc), margins=ms,
                    lam=lam, gamma=GAMMA, beta=beta, refs="bf16 greedy, fp32 scored" if rmodel is not None else "fp32")
        print(f"{model_id} E={E} k={k} (native {k0}): ref in {time.time() - t0:.0f}s; margins "
              + json.dumps({a: {kk: round(v, 4) for kk, v in b.items()} for a, b in ms.items()}),
              file=sys.stderr, flush=True)
        if margins_only:
            row = dict(base, variant="margins")
            rows.append(row)
            sink and sink(row)
            continue
        names = ["self"] + [v for v in (variants or VARIANTS) if not (k == 1 and VARIANTS[v][0] == "topk")]
        for v in names:
            inst = None
            if v != "self":
                inst = Installed(model, blocks, v, k, check=True, lam=lam, beta=beta)
            try:
                r = (score_tf(model, enc, ref_seq, ref_lp) if rmodel is not None
                     else score(model, enc, ref_seq, ref_lp, ref_cont))
                blk = {}
                if inst is not None:
                    errs = [hm.err for hm in inst.hosts]
                    blk = dict(block_err_max=max(e[0] for e in errs),
                               block_err_rms=math.sqrt(sum(e[1] for e in errs) / max(sum(e[2] for e in errs), 1)),
                               carry_max=inst.side.stats["max_c"])
            finally:
                if inst is not None:
                    inst.remove()
            row = dict(base, variant=v, **r, **blk)
            rows.append(row)
            sink and sink(row)
            print(f"  k={k} {v:>13}: top1 {r['top1']:.4f} ex32 {r['ex32']} ex64 {r['ex64']} "
                  f"KL {r['kl']:.2e} div {r['div']}  block max|d| {blk.get('block_err_max', 0):.2e}",
                  file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--story", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--ks", help="comma-separated top-k settings to run (default: native, 1)")
    ap.add_argument("--variants", help="comma-separated variants (default: all)")
    ap.add_argument("--margins-only", action="store_true")
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--ref-device", help="generate the continuations with a bf16 copy here; score fp32 "
                                          "teacher-forced only (top1, kl)")
    ap.add_argument("--out", default="moe/data/dense_host_prenorm.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.model, a.device, a.story, a.n_prompts, ks=[int(x) for x in a.ks.split(",")] if a.ks else None,
        variants=a.variants.split(",") if a.variants else None, margins_only=a.margins_only, sink=sink,
        lam=a.lam, beta=a.beta, ref_device=a.ref_device)


if __name__ == "__main__":
    main()
