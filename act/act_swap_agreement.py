"""Is a single-op activation floor PAYABLE?  Swap a real model's FFN activation for a host's best fit
and measure whether generation still agrees with the original model.

For a model whose FFN activation is Y and a host activation X, every FFN activation is replaced by
the best single-op fit  a*X(c*x + d) + e*x + f  of Y on [-6, 6] (act/activation_cross.py
fit_minimax via fit_host -- exactly the function a host with activation X computes, its floor is the
max error on the interval). One exception: a ReLU target on a swigluoai host runs the REAL clamped
gpt-oss unit along Lean relu_sub_swigluoai (gate c x, up a x - 1, down 1/(a K), a = 6 / max|x| of the
model's pre-activations), since the single-op scaled-gate route is refuted on the clamped unit. Then, on the ORIGINAL model's greedy 64-token continuations of N prompts:

  top1    teacher-forced: fraction of the 64 continuation positions where the swapped model's argmax
          equals the original's
  kl      mean KL(original || swapped) per continuation position (nats)
  ex32 / ex64  free-running: fraction of prompts whose swapped-model greedy continuation is
          token-identical to the original's for the first 32 / all 64 tokens
  div     mean first-divergence position (64 when identical)

Also recorded per (model, host): the floor on [-6, 6], and the error the fit actually makes on the
model's OWN pre-activations (max / rms over every FFN activation input on the reference pass) --
the floor is a statement about [-6, 6], the model's pre-activations can leave it.

The activation's own module (host = target) is the control (must be top1 = 1, kl = 0).
fp32 throughout, TF32 off, each model on one device.

  python -m act.act_swap_agreement gpt2 --device cuda:0 --out act/data/act_swap_agreement.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from act.activation_cross import fit_host, relu_swigluoai_params
from hf.attn_pin import EAGER as ATTN, attn_impl_for      # eager pinned (sdpa drops GPT-2 / Gemma-2 features)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HOSTS = ("ReLU", "relu2", "SiLU", "GeLU", "mish", "swigluoai", "situ", "xielu")
N_NEW = 64

PROMPTS = [
    "The history of the Roman Empire is", "In mathematics, a prime number is",
    "The best way to learn a new programming language is", "Photosynthesis is the process by which",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "The capital of France is Paris. The capital of Germany is", "Once upon a time, in a small village,",
    "The main causes of the First World War were", "To make a simple tomato soup, you will need",
    "Climate change refers to", "The theory of relativity, developed by Albert Einstein,",
    "Here is a list of the planets in our solar system:", "The stock market fell sharply today after",
    "A neural network is a computational model that", "import numpy as np\n\n# compute the mean of each row\n",
    "The recipe for a good night's sleep includes", "Water boils at 100 degrees Celsius because",
    "In the year 2050, cities will", "The French Revolution began in 1789 when",
    "Dear hiring manager, I am writing to apply for", "The human heart has four chambers:",
    "Q: What is the largest ocean on Earth?\nA:", "The quick brown fox jumps over the lazy dog. This sentence",
    "Machine learning models are trained by",
]
STORY_PROMPTS = [
    "Once upon a time, there was a little girl named Lily.", "One day, a big dog named Max",
    "Tom and his mom went to the park.", "There was a small cat who loved to",
    "Once upon a time, in a big forest, there lived", "Sara had a red ball. She",
    "The sun was shining and the birds were singing.", "Ben wanted to build a tall tower.",
    "Once there was a boy who did not like to share.", "Mia found a shiny stone in the garden.",
    "The little bird was sad because", "One morning, Timmy woke up and saw",
    "Anna and Ben were best friends.", "There was a big, scary bear in the woods.",
    "Lucy liked to paint pictures of", "Once upon a time, a tiny fish lived in",
    "The old man had a magic hat.", "Sam was hungry, so he went to the kitchen.",
    "It was a rainy day, and Jack could not go outside.", "A friendly frog sat on a lily pad.",
    "Molly lost her favorite toy.", "The princess looked out of the window and",
    "Max and Sue found a box in the attic.", "One day, the wind blew very hard.",
]


# --- torch versions of act/activation_cross.ACTS (same definitions) -------------------------------
def _xielu_t(z, alpha_p=0.8, alpha_n=0.8, beta=0.5, eps=-1e-6):
    return torch.where(z > 0, alpha_p * z * z + beta * z,
                       alpha_n * (torch.expm1(torch.clamp(z, max=eps)) - z) + beta * z)


TACTS = {
    "ReLU": torch.relu,
    "relu2": lambda z: torch.relu(z) ** 2,
    "SiLU": F.silu,
    "GeLU": lambda z: F.gelu(z),
    "GeLU_tanh": lambda z: F.gelu(z, approximate="tanh"),
    "mish": F.mish,
    "swigluoai": lambda z: z * torch.sigmoid(1.702 * z),
    "situ": lambda z: 4.0 * torch.tanh(z / 4.0) * torch.sigmoid(z),
    "xielu": _xielu_t,
}


class FnMod(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class BlendAct(torch.nn.Module):
    """Y + t (fit - Y): a hypothetical host whose error is t times the fit's, everywhere (floor and
    in-distribution error both scale by t) -- the dose-response between the target and a real fit."""

    def __init__(self, orig, fit, t):
        super().__init__()
        self.orig, self.fit, self.t = orig, fit, t

    def forward(self, x):
        y = self.orig(x)
        return y + self.t * (self.fit(x) - y)


def swigluoai_unit_t(gate, up, limit=7.0, alpha=1.702):
    """The clamped gpt-oss unit (clip(up, -L, L) + 1) g' sigmoid(alpha g'), g' = min(gate, L) (torch)."""
    g = gate.clamp(max=limit)
    return (up.clamp(-limit, limit) + 1) * g * torch.sigmoid(alpha * g)


class FitAct(torch.nn.Module):
    """a * X(c x + d) + e x + f -- what a host with activation X computes for the target. With dict
    params (route "relu_sub_swigluoai", activation_cross.fit_relu_on_swigluoai): the real clamped
    swigluoai unit fed gate row c x and up row a x + up_bias, times the down scale."""

    def __init__(self, host, p):
        super().__init__()
        self.host, self.p = host, p

    def forward(self, x):
        if isinstance(self.p, dict):
            q = self.p
            return swigluoai_unit_t(q["c"] * x, q["a"] * x + q["up_bias"], q["limit"], q["alpha"]) * q["down"]
        a, c, d, e, f = self.p
        return a * TACTS[self.host](c * x + d) + e * x + f


def _bounded(p, bound):
    """relu_sub_swigluoai needs |a z| <= L - 1: refit a (and the down scale) to the inputs' bound."""
    if not isinstance(p, dict):
        return p
    return relu_swigluoai_params(bound, p["c"], p["limit"], p["alpha"])


def _params_list(p):
    return dict(p) if isinstance(p, dict) else list(map(float, p))


# --- where each family keeps its FFN activation --------------------------------------------------
def act_slots(model):
    """[(parent, attr, fn)] for every FFN activation; fn(x) is the original activation."""
    slots = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls.endswith("Experts") and hasattr(mod, "_apply_gate"):      # gpt-oss: gate*sigmoid(1.702 gate)
            slots.append((mod, "_gate_act", lambda z: z * torch.sigmoid(z * 1.702)))
            continue
        if cls == "SituAndMul":
            slots.append((mod, "_gate_act", lambda z, b=mod.beta: b * torch.tanh(z / b) * torch.sigmoid(z)))
            continue
        for attr in ("act_fn", "act", "activation_fn"):
            ch = getattr(mod, attr, None)
            if isinstance(ch, torch.nn.Module) and ("MLP" in cls or "DecoderLayer" in cls):
                slots.append((mod, attr, ch))
    return slots


def _gpt_oss_apply_gate(self, gate_up):
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    return (up + 1) * self._gate_act(gate)


def _situ_forward(self, x):
    d = x.shape[-1] // 2
    gate, up = x[..., :d].to(torch.float32), x[..., d:].to(torch.float32)
    if self.linear_beta is not None:
        up = self.linear_beta * torch.tanh(up / self.linear_beta)
    return (self._gate_act(gate) * up).to(x.dtype)


def install(slots):
    """Make every slot swappable: returns setter(list_of_fns) (None = original)."""
    orig = []
    for mod, attr, fn in slots:
        if attr == "_gate_act":
            object.__setattr__(mod, "_gate_act", fn)
            if hasattr(mod, "_apply_gate"):
                mod._apply_gate = types.MethodType(_gpt_oss_apply_gate, mod)
            else:
                mod.forward = types.MethodType(_situ_forward, mod)
        orig.append(fn)

    def set_fns(fns):
        for (mod, attr, _), o, f in zip(slots, orig, fns or orig):
            if attr == "_gate_act":
                object.__setattr__(mod, "_gate_act", f if f is not None else o)
            else:
                f = f if f is not None else o
                setattr(mod, attr, f if isinstance(f, torch.nn.Module) else FnMod(f))
    return orig, set_fns


def target_numpy(fn):
    """The model's own activation as a numpy function (float64 on the CPU)."""
    import copy
    g = copy.deepcopy(fn).to("cpu").double() if isinstance(fn, torch.nn.Module) else fn
    if isinstance(g, torch.nn.Module) and hasattr(g, "with_vector_loads"):   # xIELU: python path
        g.with_vector_loads = False
    def Y(z):
        with torch.no_grad():
            return g(torch.as_tensor(np.asarray(z, dtype=np.float64))).numpy()
    return Y


def signature(fn):
    if isinstance(fn, torch.nn.Module):
        ps = [round(float(p.detach().float().reshape(-1)[0]), 6) for p in fn.parameters()]
        bs = [round(float(b.detach().float().reshape(-1)[0]), 6) for b in fn.buffers()]
        return type(fn).__name__ + str(ps + bs)
    return "fn:" + repr(fn(torch.linspace(-3, 3, 7, dtype=torch.float64)).tolist())


# --- generation and scoring ----------------------------------------------------------------------
@torch.no_grad()
def greedy(model, ids, n_new=N_NEW):
    """Greedy continuation, no EOS stop, KV cache (falls back to full recompute)."""
    try:
        out = model(ids, use_cache=True)
        past, toks = out.past_key_values, []
        nxt = out.logits[:, -1].argmax(-1)
        for _ in range(n_new):
            toks.append(nxt)
            if len(toks) == n_new:
                break
            out = model(nxt[:, None], past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
        return torch.stack(toks, 1)
    except Exception:
        seq = ids
        for _ in range(n_new):
            seq = torch.cat([seq, model(seq).logits[:, -1:].argmax(-1)], 1)
        return seq[:, ids.shape[1]:]


@torch.no_grad()
def cont_logp(model, seq, P):
    """log-softmax over the positions that predict the continuation tokens seq[P:]."""
    return F.log_softmax(model(seq).logits[0, P - 1:-1].double(), -1).cpu()     # via the CPU: frees the GPU


def run(model_id, device="cuda:0", hosts=HOSTS, story=False, n_prompts=24, fit_cache=None,
        trust_remote_code=False, label=None, load_kw=None, only=None, dose=None, sink=None, save_fits=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 trust_remote_code=trust_remote_code,
                                                 **{"attn_implementation": attn_impl_for(model_id, trust_remote_code),
                                                    **(load_kw or {})}).to(device).eval()
    slots = act_slots(model)
    assert slots, "no FFN activation found"
    orig, set_fns = install(slots)
    cfg = getattr(model.config, "text_config", model.config)
    n_layers = cfg.num_hidden_layers
    sigs = [signature(fn) for fn in orig]
    print(f"{model_id}: {len(slots)} activation slots over {n_layers} layers, "
          f"{len(set(sigs))} distinct target functions ({sigs[0][:60]})", file=sys.stderr, flush=True)

    # fits (cached by target signature): (floor, params) per distinct target function and host
    fit_cache = {} if fit_cache is None else fit_cache
    Ys = {}
    for s, fn in zip(sigs, orig):
        Ys.setdefault(s, target_numpy(fn))
    fits = {}
    for h in hosts:
        for s, Y in Ys.items():
            key = f"{s}|{h}"
            if key not in fit_cache:
                fl, p = fit_host(h, Y)
                fit_cache[key] = (fl, p)
            fits[(s, h)] = fit_cache[key]
    if save_fits is not None:
        save_fits(fit_cache)

    prompts = (STORY_PROMPTS if story else PROMPTS)[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    set_fns(None)
    ref_cont = [greedy(model, ids) for ids in enc]
    ref_seq = [torch.cat([ids, c], 1) for ids, c in zip(enc, ref_cont)]
    ref_lp = [cont_logp(model, s, ids.shape[1]) for s, ids in zip(ref_seq, enc)]

    # pre-activation statistics and the fits' in-distribution error, on the reference pass
    stats = {h: [0.0, 0.0, 0] for h in hosts}       # max err, sum sq err, count
    pre = {"max_abs": 0.0, "frac_out6": 0.0, "n": 0, "q999": []}
    hooks_state = {}

    def make_probe(i):
        fn = orig[i]

        def probe(x):
            y = fn(x)
            xd = x.detach().double()
            yd = y.detach().double()
            pre["max_abs"] = max(pre["max_abs"], xd.abs().max().item())
            pre["frac_out6"] += (xd.abs() > 6).sum().item()
            pre["n"] += xd.numel()
            flat = xd.abs().reshape(-1)
            k = max(1, int(0.999 * flat.numel()))
            pre["q999"].append(flat.kthvalue(k).values.item())
            for h in hosts:
                fl, p = fits[(sigs[i], h)]
                p = _bounded(p, max(xd.abs().max().item(), 1e-30))     # this call's own input bound
                err = (FitAct(h, p)(xd) - yd).abs()
                st = stats[h]
                st[0] = max(st[0], err.max().item())
                st[1] += (err ** 2).sum().item()
                st[2] += err.numel()
            return y
        return probe

    set_fns([make_probe(i) for i in range(len(orig))])
    for s in ref_seq:
        with torch.no_grad():
            model(s)
    set_fns(None)
    pre_stats = dict(max_abs=pre["max_abs"], frac_out6=pre["frac_out6"] / max(pre["n"], 1),
                     q999=float(np.median(pre["q999"])))

    rows = []
    variants = [("self", None)] + [(h, h) for h in hosts] + [("bf16", None), ("fp16", None)]
    if only:
        variants = [v for v in variants if v[0] in only]
    for spec in dose or ():                    # "SiLU*0.1": Y + 0.1 (SiLU fit - Y)
        h, t = spec.split("*")
        variants.append((spec, h))
    for vname, h in variants:
        if vname in ("bf16", "fp16"):       # the numerics controls: the ORIGINAL model in bf16 / fp16
            set_fns(None)
            # parameters only: non-persistent buffers (RoPE inv_freq, masks) stay fp32, as in a
            # from_pretrained(dtype=bfloat16) load
            sd = {n: p.data.cpu() for n, p in model.named_parameters()}    # via the CPU (no P2P)
            dt = torch.bfloat16 if vname == "bf16" else torch.float16
            for n, p in model.named_parameters():          # one tensor at a time, via the CPU
                p.data = sd[n].to(dt).to(device)
            floor_max = floor_mean = 0.0
        elif h is None:
            set_fns(None)
            floor_max = floor_mean = 0.0
        else:
            t = float(vname.split("*")[1]) if "*" in vname else 1.0
            # a bounded-input route (relu_sub_swigluoai) is built for the reference pass's input bound
            fns = [FitAct(h, _bounded(fits[(s, h)][1], pre_stats["max_abs"])) for s in sigs]
            if "*" in vname:
                fns = [BlendAct(o if isinstance(o, torch.nn.Module) else FnMod(o), f, t) for o, f in zip(orig, fns)]
            set_fns(fns)
            fls = [t * fits[(s, h)][0] for s in sigs]
            floor_max, floor_mean = max(fls), float(np.mean(fls))
        top1 = kl = 0.0
        n = 0
        ex32 = ex64 = 0
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
        row = dict(model=label or model_id, layers=n_layers, slots=len(slots), host=vname,
                   floor=floor_max, floor_mean=floor_mean, top1=top1 / n, kl=kl / n,
                   ex32=ex32 / len(enc), ex64=ex64 / len(enc), div=float(np.mean(div)),
                   n_prompts=len(enc))
        if h is not None:
            st = stats[h]
            t = float(vname.split("*")[1]) if "*" in vname else 1.0
            row.update(err_indist_max=t * st[0], err_indist_rms=t * math.sqrt(st[1] / max(st[2], 1)),
                       fit=[_params_list(_bounded(fits[(s, h)][1], pre_stats["max_abs"]))
                            for s in sorted(set(sigs))][:1])
        row.update(pre=pre_stats)
        rows.append(row)
        if sink is not None:
            sink(row)
        if vname in ("bf16", "fp16"):
            for n, p in model.named_parameters():
                p.data = sd[n].to(device)
        print(f"  {vname:>10}: floor {floor_max:.3f}  top1 {row['top1']:.4f}  ex32 {row['ex32']:.2f}  "
              f"ex64 {row['ex64']:.2f}  KL {row['kl']:.2e}  in-dist max err "
              f"{row.get('err_indist_max', 0):.3g}", file=sys.stderr, flush=True)
    set_fns(None)
    print(f"  done in {time.time() - t0:.0f}s; pre-acts max|x| {pre_stats['max_abs']:.1f}, "
          f"frac |x|>6 {pre_stats['frac_out6']:.2e}", file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="act/data/act_swap_agreement.jsonl")
    ap.add_argument("--story", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--label")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--dose", help="comma-separated HOST*t dose-response variants, e.g. SiLU*0.1,SiLU*0.3")
    ap.add_argument("--only", help="comma-separated variants to run (self, a host name, bf16)")
    ap.add_argument("--tf5", action="store_true",
                    help="run under transformers 5.18 from ~/.cache/upstream_refs (nanochat, gpt-oss 5.x configs)")
    a = ap.parse_args()
    if a.tf5 and not os.environ.get("ACT_SWAP_TF5"):
        import subprocess
        tf5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prelude = ("import sys, types\n"
                   "m = types.ModuleType('transformers.dependency_versions_check'); m.dep_version_check = lambda *a, **k: None\n"
                   "sys.modules['transformers.dependency_versions_check'] = m\n"
                   "sys.modules['fla'] = None\n"
                   f"sys.argv = {['act_swap'] + sys.argv[1:]!r}\n"
                   "import runpy; runpy.run_module('act.act_swap_agreement', run_name='__main__')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([tf5, root]), ACT_SWAP_TF5="1")
        sys.exit(subprocess.run([sys.executable, "-c", prelude], env=env).returncode)
    cache_path = os.path.join(os.path.dirname(a.out), "act_fits_cache.json")
    try:
        cache = {k: tuple(v) for k, v in json.load(open(cache_path)).items()}
    except Exception:
        cache = {}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def sink(r):                          # each row as soon as it is measured
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    def save_fits(cache):                 # merge (other processes may have written meanwhile)
        try:
            disk = json.load(open(cache_path))
        except Exception:
            disk = {}
        disk.update({k: list(v) for k, v in cache.items()})
        json.dump(disk, open(cache_path, "w"))
    rows = run(a.model, a.device, story=a.story, n_prompts=a.n_prompts, fit_cache=cache,
               trust_remote_code=a.trust_remote_code, label=a.label,
               only=a.only.split(",") if a.only else None,
               dose=a.dose.split(",") if a.dose else None, sink=sink, save_fits=save_fits)
    return rows


if __name__ == "__main__":
    main()
