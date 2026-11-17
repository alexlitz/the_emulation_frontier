"""Are the softcap / qk-norm REGIME edges payable?  Measured on real checkpoints, both directions.

graph/model_graph.py SOFTCAP_ORDER / QKNORM_ORDER let a plain host play a softcapped or qk-normed
target, and the reverse, at "approx" cost: exact only while the scores stay inside a range. This
module swaps the construction into a real model and scores it on the ORIGINAL model's own greedy
64-token continuations of the 24 act/act_swap_agreement.py prompts, fp32 with TF32 off -- the same
standard as ACT_FLOOR_MAX / MIXER_GAP_PAYABLE / CLA_CARRY_PAYABLE:

  top1  teacher-forced argmax agreement over the 64 continuation positions
  ex32  fraction of prompts whose free-running greedy continuation matches for 32 tokens

Variants (one per row; "self" and "bf16" are the controls):

  softcap target on a plain host   (google/gemma-2-2b: attention cap 50, final-logit cap 30)
    nocap_attn / nocap_final / nocap_both   drop the cap(s): the plain host computes z, not c tanh(z/c)
  plain target on a softcap host   (a Llama / Qwen2 checkpoint gets a cap it never had)
    cap_attn:c    c tanh(z / c) on every attention score (z = q.k * scaling, before the mask)
    cap_final:c   c tanh(z / c) on the output logits (monotone: greedy argmax unchanged)
    ...@bf16      the same in bf16 (does precision limit how large c can be?)
  qk-norm target on a plain host   (Qwen3 / OLMo-2 / Gemma-3: drop the norm, fold a FIXED scale)
    qkfold        g * x / c, c FIXED per (layer, head): the least-squares constant sum r^2 / sum r
                  (r = rms(x_t)) on the evaluation text itself -- the most favourable calibration
    qkfold_story  c calibrated on held-out text (the story prompts and their greedy continuations)
  plain target on a qk-norm host   (a Llama / Qwen2 checkpoint gets a qk-norm with gain c)
    qkhost_head   per-head RMSNorm of q and k (Qwen3 / Gemma-3 style), gain c = mean_t rms per head
                  (the least-squares gain), calibrated on the evaluation text itself
    qkhost_joint  one RMSNorm over the whole projection (OLMo-2 style), least-squares gain per head

The score ranges are recorded on the reference pass: per layer max |z| of the attention scores the
cap acts on (unmasked entries only), their 99.9th percentile, max |final logit|, and for qk-norm the
coefficient of variation of rms(q), rms(k) over tokens (with and without position 0). The Lean bound
|c tanh(z/c) - z| <= |z|^3 / (3 c^2) (proofs/PROOFS.md) turns max |z| into a worst
score perturbation.

  python -m attn.regime_agreement google/gemma-2-2b --device cuda:0
  python -m attn.regime_agreement HuggingFaceTB/SmolLM2-135M --device cuda:1
  python -m attn.regime_agreement google/gemma-2-9b --device split    # fp32 over two GPUs, via the CPU
  python -m attn.regime_agreement --report                            # table, smallest caps, verdicts

graph/model_graph.py REGIME_PAYABLE records the verdicts; tests/test_regime_agreement.py checks them
against the rows.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from act.act_swap_agreement import N_NEW, PROMPTS, STORY_PROMPTS, cont_logp, greedy

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

EPS = 1e-6
CAPS = (5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000)
FINAL_CAPS = (5, 15, 30)


# --- one attention function for every family: optional softcap, and the score statistics ---------
class State:
    cap = "native"        # "native" (the module's own softcap kwarg), None, or a float
    collect = False
    zmax = {}             # layer -> max |z| over unmasked scores
    zq = {}               # layer -> [99.9th percentile per call]


S = State()


def _repeat_kv(x, n):
    return x if n == 1 else x[:, :, None].expand(x.shape[0], x.shape[1], n, *x.shape[2:]).reshape(
        x.shape[0], x.shape[1] * n, *x.shape[2:])


def regime_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                     softcap=None, **kwargs):
    """eager attention (the Llama / Gemma-2 definition) with the softcap chosen by S.cap."""
    if scaling is None:
        scaling = module.head_dim ** -0.5
    n_rep = query.shape[1] // key.shape[1]
    k, v = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
    z = torch.matmul(query, k.transpose(2, 3)) * scaling
    mask = attention_mask[:, :, :, : k.shape[-2]] if attention_mask is not None else None
    if S.collect:
        zz = z.detach().double().abs()
        if mask is not None:
            zz = zz[(mask == 0).expand_as(zz)]
        li = getattr(module, "layer_idx", -1)
        S.zmax[li] = max(S.zmax.get(li, 0.0), zz.max().item())
        flat = zz.reshape(-1)
        S.zq.setdefault(li, []).append(flat.kthvalue(max(1, int(0.999 * flat.numel()))).values.item())
    c = softcap if S.cap == "native" else S.cap
    if c is not None:
        z = c * torch.tanh(z / c)
    if mask is not None:
        z = z + mask
    w = F.softmax(z, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, v).transpose(1, 2).contiguous()
    return out, w


def install_attention(model):
    """Every attention module of the model calls regime_attention (eager path, patched in place)."""
    mods = {sys.modules[type(m).__module__] for _, m in model.named_modules()
            if type(m).__name__.endswith("Attention")}
    for mod in mods:
        if hasattr(mod, "eager_attention_forward"):
            mod.eager_attention_forward = regime_attention
    assert mods, "no attention module"


# --- final-logit cap ----------------------------------------------------------------------------------
class CapHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head, self.c, self.collect, self.zmax = head, None, False, 0.0

    def forward(self, x):
        y = self.head(x)
        if self.collect:
            self.zmax = max(self.zmax, y.detach().abs().max().item())
        return y if self.c is None else self.c * torch.tanh(y / self.c)

    @property
    def weight(self):
        return self.head.weight


# --- qk-norm: the fold (target has it, host does not) and the host norm (host has it) --------------
def _rms(x, eps=EPS):
    return torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)


class FoldNorm(torch.nn.Module):
    """A qk-normed target on a plain host: orig(x) = g * x / rms(x) becomes g * x / c, c FIXED."""

    def __init__(self, orig, c):
        super().__init__()
        self.orig, self.c = orig, c
        self.eps = getattr(orig, "variance_epsilon", getattr(orig, "eps", EPS))   # the norm's own eps

    def forward(self, x):
        return (self.orig(x).float() * _rms(x, self.eps) / self.c.to(x.device)).to(x.dtype)


class HostNorm(torch.nn.Module):
    """A plain target on a qk-norm host: the projection output gets an RMSNorm with gain c (per head)."""

    def __init__(self, proj, c, head_dim, joint):
        super().__init__()
        self.proj, self.c, self.hd, self.joint = proj, c, head_dim, joint

    def forward(self, x):
        y = self.proj(x)
        sh = y.shape
        if self.joint:                                  # OLMo-2: one rms over the whole projection
            out = y.float() / _rms(y)
            out = out.view(*sh[:-1], -1, self.hd) * self.c.to(y.device)[:, None]
        else:                                           # Qwen3 / Gemma-3: one rms per head
            yh = y.view(*sh[:-1], -1, self.hd).float()
            out = yh / _rms(yh) * self.c.to(y.device)[:, None]
        return out.reshape(sh).to(y.dtype)


def _head_axis(model_type):
    return {"qwen3": -2, "gemma3_text": -3, "gemma3": -3}.get(model_type)   # olmo2: joint norm


def _attn_modules(model):
    return [m for _, m in model.named_modules()
            if type(m).__name__.endswith("Attention") and hasattr(m, "q_proj")]


def _stats_rms(r, axis_head):
    """r: per-token rms with the head axis at axis_head (None: no head axis). -> (mean per head, cv
    per head, cv per head without position 0)."""
    if axis_head is None:
        r = r.unsqueeze(-1)
        axis_head = -1
    r = r.movedim(axis_head, -1)                        # (..., T, H) or (B, T, H)
    r = r.reshape(-1, r.shape[-2], r.shape[-1])         # (B, T, H)
    return r


class RmsCollector:
    """per (layer, q / k): the per-token rms per head over every reference pass."""

    def __init__(self):
        self.r = {}

    def add(self, key, r):                              # r: (B, T, H)
        self.r.setdefault(key, []).append(r.detach().double().cpu())

    def summary(self, ls_fold=False):
        """mean[key]: the constant per head. ls_fold=False: mean_t r_t, the least-squares gain of a
        qk-norm HOST playing x (min sum |c x / r - x|^2); True: sum r^2 / sum r, the least-squares
        constant of the FOLD (min sum |x / r - x / c|^2), as block/fold_agreement.py."""
        mean, cv, cv1, ratio = {}, {}, {}, {}
        for key, rs in self.r.items():
            per_tok = torch.cat([x.reshape(-1, x.shape[-1]) for x in rs])           # (N, H)
            no0 = torch.cat([x[:, 1:].reshape(-1, x.shape[-1]) for x in rs])
            mean[key] = per_tok.pow(2).mean(0) / per_tok.mean(0) if ls_fold else per_tok.mean(0)
            cv[key] = (per_tok.std(0) / per_tok.mean(0)).max().item()
            cv1[key] = (no0.std(0) / no0.mean(0)).max().item()
            ratio[key] = (per_tok.max(0).values / per_tok.min(0).values).max().item()
        for key in [k for k in self.r if k[1].endswith("_joint")]:
            # OLMo-2-style host: out_h = c_h y_h / r_joint; least squares c_h = sum r_h^2 / r_j over
            # sum r_h^2 / r_j^2 (|y_h|^2 = head_dim r_h^2)
            rh = torch.cat([x.reshape(-1, x.shape[-1]) for x in self.r[(key[0], key[1][:-6])]])
            rj = torch.cat([x.reshape(-1, 1) for x in self.r[key]])
            mean[key] = (rh.pow(2) / rj).sum(0) / (rh.pow(2) / rj.pow(2)).sum(0)
        return mean, cv, cv1, ratio


def collect_qk(model, seqs, mode):
    """mode "norm": hook the target's q_norm / k_norm INPUTS; "proj": the plain q_proj / k_proj outputs.
    Returns RmsCollector summary keyed (layer, "q" / "k"); per head for "proj" (and per-head norms)."""
    mt = getattr(model.config, "text_config", model.config).model_type
    col, hs = RmsCollector(), []
    for li, a in enumerate(_attn_modules(model)):
        for which in ("q", "k"):
            if mode == "norm":
                m = getattr(a, f"{which}_norm")
                ax = _head_axis(mt)

                def hook(mod, inp, out, li=li, which=which, ax=ax):
                    x = inp[0]
                    r = _rms(x).squeeze(-1)
                    col.add((li, which), _stats_rms(r, None if ax is None else ax + 1))
            else:
                m = getattr(a, f"{which}_proj")
                hd = a.head_dim

                def hook(mod, inp, out, li=li, which=which, hd=hd):
                    y = out.view(*out.shape[:-1], -1, hd)
                    col.add((li, which), _rms(y).squeeze(-1))                       # (B, T, H)
                    col.add((li, which + "_joint"), _rms(out).squeeze(-1).unsqueeze(-1))
            hs.append(m.register_forward_hook(hook))
    with torch.no_grad():
        for s in seqs:
            model(s)
    for h in hs:
        h.remove()
    return col.summary(ls_fold=mode == "norm")


# --- a model too big for one GPU in fp32: layers split over two, every hop routed via the CPU ----------
def _move(x, d):
    if isinstance(x, torch.Tensor):
        return x if x.device == d else x.cpu().to(d)          # GPU-to-GPU copies are broken: via the CPU
    if isinstance(x, tuple):
        return tuple(_move(y, d) for y in x)
    if isinstance(x, list):
        return [_move(y, d) for y in x]
    return x


def place_split(model, devs=("cuda:0", "cuda:1"), frac0=0.5):
    """Embedding (tied head) and the first frac0 of the layers on devs[0], the rest and the final norm
    on devs[1]; a pre-hook moves every tensor argument of a layer / the norm / the head to its device."""
    inner = model.model
    layers = inner.layers
    k = int(round(len(layers) * frac0))
    d0, d1 = torch.device(devs[0]), torch.device(devs[1])
    for name, ch in inner.named_children():
        if name != "layers":
            ch.to(d1 if name == "norm" else d0)
    for i, layer in enumerate(layers):
        layer.to(d0 if i < k else d1)
    model.lm_head.to(d0)

    def pre(mod, args, kwargs):
        d = next(mod.parameters()).device
        return _move(args, d), {kk: _move(v, d) for kk, v in kwargs.items()}
    for m in list(layers) + [inner.norm, model.lm_head]:
        m.register_forward_pre_hook(pre, with_kwargs=True)
    return d0


# --- the run ----------------------------------------------------------------------------------------
def run(model_id, device="cuda:0", only=None, n_prompts=24, sink=None, label=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    if device == "split":
        device = place_split(model)
    else:
        model.to(device)
    install_attention(model)
    cfg = getattr(model.config, "text_config", model.config)
    mt = cfg.model_type
    attn_cap = getattr(cfg, "attn_logit_softcapping", None)
    final_cap = getattr(cfg, "final_logit_softcapping", None)
    attns = _attn_modules(model)
    qk_target = bool(attns) and hasattr(attns[0], "q_norm")
    head = CapHead(model.lm_head)
    model.lm_head = head
    if final_cap is not None:                     # the model's own final cap moves into CapHead
        cfg.final_logit_softcapping = None
        head.c = float(final_cap)

    prompts = PROMPTS[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    S.cap = "native"
    ref_cont = [greedy(model, ids) for ids in enc]
    ref_seq = [torch.cat([ids, c], 1) for ids, c in zip(enc, ref_cont)]
    ref_lp = [cont_logp(model, s, ids.shape[1]) for s, ids in zip(ref_seq, enc)]

    # score ranges on the reference pass
    S.zmax, S.zq, S.collect, head.collect = {}, {}, True, True
    with torch.no_grad():
        for s in ref_seq:
            model(s)
    S.collect, head.collect = False, False
    ranges = dict(zmax=max(S.zmax.values()), zq999=float(np.median([q for v in S.zq.values() for q in v])),
                  zmax_layer=[S.zmax[k] for k in sorted(S.zmax)], logit_max=head.zmax,
                  attn_cap=attn_cap, final_cap=final_cap)
    qk = {}
    if qk_target:
        mean, cv, cv1, ratio = collect_qk(model, ref_seq, "norm")
        story = [tok(p, return_tensors="pt").input_ids.to(device) for p in STORY_PROMPTS[:n_prompts]]
        story_seq = [torch.cat([ids, greedy(model, ids)], 1) for ids in story]
        mean_story = collect_qk(model, story_seq, "norm")[0]
    else:
        mean, cv, cv1, ratio = collect_qk(model, ref_seq, "proj")
    qk = dict(cv_max=max(cv.values()), cv_max_no_pos0=max(cv1.values()), ratio_max=max(ratio.values()),
              cv_mean=float(np.mean(list(cv.values()))), cv_mean_no_pos0=float(np.mean(list(cv1.values()))))
    ranges["qk"] = qk
    print(f"{model_id}: max|z| {ranges['zmax']:.1f} (q999 {ranges['zq999']:.1f}), max|logit| "
          f"{head.zmax:.1f}, caps attn {attn_cap} final {final_cap}, qk {'target' if qk_target else 'plain'}"
          f" cv {qk['cv_mean']:.3f} (max {qk['cv_max']:.3f}, no pos0 {qk['cv_mean_no_pos0']:.3f})",
          file=sys.stderr, flush=True)

    # the variants
    variants = ["self", "bf16"]
    if attn_cap is not None or final_cap is not None:
        variants += ["nocap_attn", "nocap_final", "nocap_both", "nocap_both@bf16"]
    else:
        variants += [f"cap_attn:{c}" for c in CAPS] + [f"cap_final:{c}" for c in FINAL_CAPS]
        variants += [f"cap_attn:{c}@bf16" for c in (30, 100, 1000)]
    variants += ["qkfold", "qkfold_story", "qkfold@bf16"] if qk_target else ["qkhost_head", "qkhost_joint",
                                                                             "qkhost_head@bf16"]
    if only:                                    # any variant spelled out (e.g. cap_attn:5000@bf16)
        variants = list(only)

    norms = {}
    if qk_target:
        for li, a in enumerate(attns):
            norms[li] = (a.q_norm, a.k_norm)
    projs = {li: (a.q_proj, a.k_proj) for li, a in enumerate(attns)}
    params = list({id(q): q for q in model.parameters()}.values())    # shared by every wrapper
    sd = None
    rows = []
    for v in variants:
        name, _, prec = v.partition("@")
        # reset
        S.cap = "native"
        head.c = None if final_cap is None else float(final_cap)
        for li, a in enumerate(attns):
            if qk_target:
                a.q_norm, a.k_norm = norms[li]
            a.q_proj, a.k_proj = projs[li]
        # set
        if name in ("nocap_attn", "nocap_both"):
            S.cap = None
        if name in ("nocap_final", "nocap_both"):
            head.c = None
        if name.startswith("cap_attn:"):
            S.cap = float(name.split(":")[1])
        if name.startswith("cap_final:"):
            head.c = float(name.split(":")[1])
        if name.startswith("qkfold"):
            m = mean_story if name == "qkfold_story" else mean
            for li, a in enumerate(attns):
                cq, ck = m[(li, "q")], m[(li, "k")]
                shape = (-1, 1, 1) if _head_axis(mt) == -3 else (-1, 1)
                if _head_axis(mt) is None:
                    cq, ck, shape = cq.reshape(()), ck.reshape(()), ()
                a.q_norm = FoldNorm(norms[li][0], cq.float().reshape(shape))
                a.k_norm = FoldNorm(norms[li][1], ck.float().reshape(shape))
        if name.startswith("qkhost"):
            joint = name == "qkhost_joint"
            for li, a in enumerate(attns):
                cq, ck = mean[(li, "q")], mean[(li, "k")]
                a.q_proj = HostNorm(projs[li][0], _joint_gain(mean, li, "q") if joint else cq.float(),
                                    a.head_dim, joint)
                a.k_proj = HostNorm(projs[li][1], _joint_gain(mean, li, "k") if joint else ck.float(),
                                    a.head_dim, joint)
        if prec == "bf16" or name == "bf16":
            if sd is None:
                sd = [(q.data.cpu(), q.device) for q in params]           # via the CPU (no P2P)
            for q, (d, dev) in zip(params, sd):
                q.data = d.to(torch.bfloat16).to(dev)
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
        if prec == "bf16" or name == "bf16":
            for q, (d, dev) in zip(params, sd):
                q.data = d.to(dev)
        cap = float(name.split(":")[1]) if ":" in name else None
        row = dict(model=label or model_id, model_type=mt, variant=v, top1=top1 / n, kl=kl / n,
                   ex32=ex32 / len(enc), ex64=ex64 / len(enc), div=float(np.mean(div)),
                   n_prompts=len(enc), ranges=ranges)
        if name.startswith("cap_attn:"):
            row["lean_bound"] = ranges["zmax"] ** 3 / (3 * cap ** 2)    # max |c tanh(z/c) - z|
        if name in ("nocap_attn", "nocap_both") and attn_cap:
            zc = min(ranges["zmax"], 1e9)
            row["cap_residual_max"] = abs(attn_cap * math.tanh(zc / attn_cap) - zc)
        rows.append(row)
        if sink is not None:
            sink(row)
        print(f"  {v:>18}: top1 {row['top1']:.4f}  ex32 {row['ex32']:.2f}  ex64 {row['ex64']:.2f}  "
              f"KL {row['kl']:.2e}", file=sys.stderr, flush=True)
    print(f"  done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return rows


def _joint_gain(mean, li, which):
    """OLMo-2-style host: out_h = c_h * y_h / rms_joint(y), c_h the least-squares gain per head."""
    return mean[(li, which + "_joint")].float()


# --- the standard and the report ------------------------------------------------------------------
DATA = "attn/data/regime_agreement.jsonl"
STRICT = (0.99, 0.90)          # top-1, ex32: the payable standard (ACT_FLOOR_MAX / MIXER_GAP_PAYABLE)
LOOSE = (0.98, 0.50)           # the looser variant

# the regime edge (slot, target value, host value) each variant measures (round 2, pass 2: softcap split into
# attn / final / no, qk-norm into variants -- graph/model_graph.py SOFTCAP_ORDER, QKNORM_VARIANTS)
EDGE = {"nocap_attn": ("softcap", "attn", "no"), "nocap_final": ("softcap", "final", "no"),
        "nocap_both": ("softcap", "attn", "no"), "cap_attn": ("softcap", "no", "attn"),
        "cap_final": ("softcap", "no", "final"), "qkfold": ("qknorm", "head", "no"),
        "qkfold_story": ("qknorm", "head", "no"), "qkhost_head": ("qknorm", "no", "head"),
        "qkhost_joint": ("qknorm", "no", "proj")}


def edge_of(variant):
    return EDGE.get(variant.partition("@")[0].split(":")[0])


def load_rows(path=DATA):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(r["model"], r["variant"])] = r          # the last measurement of a variant wins
    return rows


def judge(r, bf16):
    """(strict, loose, no worse than bf16) for one row; bf16 = the model's own bf16 control row."""
    strict = r["top1"] >= STRICT[0] and r["ex32"] >= STRICT[1]
    loose = r["top1"] >= LOOSE[0] and r["ex32"] >= LOOSE[1]
    nwb = r["top1"] >= bf16["top1"] and r["ex32"] >= bf16["ex32"]
    return strict, loose, nwb


def _passes(r):
    return r["top1"] >= STRICT[0] and r["ex32"] >= STRICT[1]


def min_passing_cap(rows):
    """[(model, smallest attention cap c from which every larger measured c passes (fp32), max |z|)]
    for every plain model given an attention-softcap host."""
    out = []
    for m in dict.fromkeys(mm for mm, _ in rows):
        caps = sorted((float(v.split(":")[1]), _passes(r)) for (mm, v), r in rows.items()
                      if mm == m and v.startswith("cap_attn:") and "@" not in v)
        if not caps:
            continue
        c_min = None
        for c, ok in reversed(caps):
            if not ok:
                break
            c_min = c
        out.append((m, c_min, rows[(m, "self")]["ranges"]["zmax"]))
    return out


def verdicts(rows):
    """{(slot, target, host): (payable, why)} under the strict standard, fp32 rows only.
      softcap attn -> no    the plain host drops the attention cap (nocap_attn) and passes on every capped model;
      softcap final -> no   dropping the final-logit cap alone (nocap_final) passes on every capped model;
      softcap no -> attn    every plain model passes from some attention cap c on (the host sets c);
      softcap no -> final   a final-logit cap of 15 or more passes on every plain model;
      qknorm head -> no     some fixed-scale fold passes on every qk-normed model;
      qknorm no -> head     some qk-norm host passes on every plain model."""
    fp32 = {k: r for k, r in rows.items() if "@" not in k[1]}
    by_model = lambda pred: {m: [r for (mm, v), r in fp32.items() if mm == m and pred(v)]
                             for m in dict.fromkeys(mm for mm, _ in fp32)}
    out = {}
    for edge, var in ((("softcap", "attn", "no"), "nocap_attn"), (("softcap", "final", "no"), "nocap_final")):
        nb = {m: rs for m, rs in by_model(lambda v: v == var).items() if rs}
        out[edge] = (bool(nb) and all(_passes(rs[0]) for rs in nb.values()),
                     {m: (rs[0]["top1"], rs[0]["ex32"]) for m, rs in nb.items()})
    caps = min_passing_cap(rows)
    fin = [r for (m, v), r in fp32.items() if v.startswith("cap_final:") and float(v.split(":")[1]) >= 15]
    out[("softcap", "no", "attn")] = (bool(caps) and all(c is not None for _, c, _ in caps), caps)
    out[("softcap", "no", "final")] = (bool(fin) and all(_passes(r) for r in fin), len(fin))
    for edge, pre in ((("qknorm", "head", "no"), "qkfold"), (("qknorm", "no", "head"), "qkhost")):
        per = {m: rs for m, rs in by_model(lambda v: v.startswith(pre)).items() if rs}
        out[edge] = (bool(per) and all(any(_passes(r) for r in rs) for rs in per.values()),
                     {m: max((r["top1"], r["ex32"]) for r in rs) for m, rs in per.items()})
    return out


def report(path=DATA):
    rows = load_rows(path)
    order = list(dict.fromkeys(m for m, _ in rows))
    rows = dict(sorted(rows.items(), key=lambda kv: order.index(kv[0][0])))
    print(f"{'model':<24} {'variant':<18} {'edge':<16} {'top1':>7} {'ex32':>6} {'KL':>9}  str loo bf "
          f"  max|z|  q999 max|logit|  cv rms(q,k)")
    for (m, v), r in rows.items():
        e = edge_of(v)
        bf = rows.get((m, "bf16"))
        js = judge(r, bf) if bf else (False,) * 3
        rg = r["ranges"]
        print(f"{m.split('/')[-1]:<24} {v:<18} {':'.join(e) if e else '-':<16} {r['top1']:>7.4f} "
              f"{r['ex32']:>6.3f} {r['kl']:>9.2e}  " + "   ".join("Y" if j else "." for j in js)
              + f"  {rg['zmax']:>6.1f} {rg['zq999']:>5.1f} {rg['logit_max']:>9.1f}  "
              f"{rg['qk']['cv_mean']:.3f} (max {rg['qk']['cv_max']:.2f})")
    print("\nsmallest passing attention cap per plain model (c, its max |z|):")
    for m, c, z in min_passing_cap(rows):
        print(f"  {m.split('/')[-1]:<24} c >= {c}   max|z| {z:.1f}")
    print("\nverdicts (strict standard, fp32):")
    for e, (ok, why) in verdicts(rows).items():
        print(f"  {':'.join(e):<16} {'PAYABLE' if ok else 'not payable'}   {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--device", default="cuda:0", help='a device, or "split" (cuda:0 + cuda:1 via the CPU)')
    ap.add_argument("--out", default=DATA)
    ap.add_argument("--only", help="comma-separated variants")
    ap.add_argument("--n-prompts", type=int, default=24)
    a = ap.parse_args()
    if a.report or not a.model:
        return report(a.out)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.model, a.device, only=a.only.split(",") if a.only else None, n_prompts=a.n_prompts, sink=sink)


if __name__ == "__main__":
    main()
