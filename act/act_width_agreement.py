"""Is a k-unit activation fit PAYABLE?  The act/act_swap_agreement.py test, with every target FFN unit
rebuilt from k host units (act/act_width_fit.py) instead of one.

For a model with FFN activation Y and a host activation X: a reference pass records the range of the
model's own pre-activations (max |a| over every FFN activation input), R = that max (rounded up);
then for each k the best k-term fit  sum_i alpha_i X(beta_i a + gamma_i)  of the model's OWN Y on
[-R, R] replaces every FFN activation, and the swapped model is scored exactly as in
act_swap_agreement.py on the original model's greedy 64-token continuations:
  top1 (teacher-forced), kl, ex32 / ex64 (free-running, token-identical), div.
Strict payable: top1 >= 0.99 and ex32 >= 0.90 (the harness definition). Looser: top1 >= 0.98 and
ex32 >= 0.50.

  python -m act.act_width_agreement gpt2 --hosts SiLU --ks 3 4 5 6 --device cuda:0
Rows go to act/data/act_width_agreement.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

from act.act_swap_agreement import (N_NEW, PROMPTS, STORY_PROMPTS, act_slots, cont_logp, greedy, install,
                                    signature)
from act.act_width_fit import TA, k_curve

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class KFitAct(torch.nn.Module):
    """sum_i alpha_i X(beta_i x + gamma_i): what k host units with activation X compute for one target
    unit (gate rows scaled / shifted, down-projection columns scaled)."""

    def __init__(self, host, p, device):
        super().__init__()
        self.host = host
        f = lambda v: torch.tensor(v, dtype=torch.float64, device=device)
        self.a, self.b, self.g = f(p["alpha"]), f(p["beta"]), f(p["gamma"])

    def forward(self, x):
        z = x.double()[..., None] * self.b + self.g
        return (TA[self.host](z) @ self.a).to(x.dtype)


def target_torch(fn):
    """The model's own activation as a float64 CPU torch function."""
    g = copy.deepcopy(fn).to("cpu").double() if isinstance(fn, torch.nn.Module) else fn
    if isinstance(g, torch.nn.Module) and hasattr(g, "with_vector_loads"):   # xIELU: python path
        g.with_vector_loads = False

    def Y(z):
        with torch.no_grad():
            return g(z)
    return Y


def payable(row, strict=True):
    return row["top1"] >= (0.99 if strict else 0.98) and row["ex32"] >= (0.90 if strict else 0.50)


def run(model_id, hosts, ks, device="cuda:0", shift=False, story=False, n_prompts=24, label=None,
        trust_remote_code=False, sink=None, kmax=None, table=None):
    """table: (target name, R, path) -- use the stored K-table fits of that target on [-R, R] instead of
    fitting the model's own activation on its own range (the in-distribution error is still measured
    against the model's own activation)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    from hf.attn_pin import attn_impl_for         # eager: sdpa drops GPT-2 layer-idx scaling, Gemma-2 softcap
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, low_cpu_mem_usage=True,
                                                 attn_implementation=attn_impl_for(model_id, trust_remote_code),
                                                 trust_remote_code=trust_remote_code,
                                                 device_map={"": device}).eval()
    slots = act_slots(model)
    assert slots, "no FFN activation found"
    orig, set_fns = install(slots)
    sigs = [signature(fn) for fn in orig]
    n_layers = getattr(model.config, "text_config", model.config).num_hidden_layers
    prompts = (STORY_PROMPTS if story else PROMPTS)[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    set_fns(None)
    ref_cont = [greedy(model, ids) for ids in enc]
    ref_seq = [torch.cat([ids, c], 1) for ids, c in zip(enc, ref_cont)]
    ref_lp = [cont_logp(model, s, ids.shape[1]) for s, ids in zip(ref_seq, enc)]

    # the model's own pre-activation range (reference pass) and the pre-activations for in-dist error
    pre = {"max_abs": 0.0, "n": 0, "out6": 0}
    samples = {s: [] for s in set(sigs)}

    def probe_of(i):
        fn = orig[i]

        def probe(x):
            xd = x.detach()
            pre["max_abs"] = max(pre["max_abs"], xd.abs().max().item())
            pre["out6"] += (xd.abs() > 6).sum().item()
            pre["n"] += xd.numel()
            flat = xd.reshape(-1)
            samples[sigs[i]].append(flat[torch.randperm(flat.numel(), device=flat.device)[:4096]].double().cpu())
            samples[sigs[i]].append(torch.stack([flat.min(), flat.max()]).double().cpu())
            return fn(x)
        return probe
    set_fns([probe_of(i) for i in range(len(orig))])
    with torch.no_grad():
        for s in ref_seq:
            model(s)
    set_fns(None)
    R = float(math.ceil(pre["max_abs"]))
    print(f"{model_id}: {len(slots)} slots, {len(set(sigs))} distinct targets, pre-acts max |a| = "
          f"{pre['max_abs']:.1f} -> fit range [-{R:g}, {R:g}]", file=sys.stderr, flush=True)
    Ys = {}
    for s, fn in zip(sigs, orig):
        Ys.setdefault(s, target_torch(fn))

    def score():
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

    rows = []
    for h in hosts:
        if table:                                   # the K-table fits (act/act_width_fit.py), one target
            tname, tR, tpath = table
            c = json.load(open(tpath))[f"{tname}|{h}|{tR:g}|{'shift' if shift else 'noshift'}"]
            curves = {s: c for s in Ys}
        else:                                       # fit the model's own activation on its own range
            curves = {s: k_curve(h, None, R, shift, kmax or max(ks), Y=Y) for s, Y in Ys.items()}
        for k in ks:
            fits = {s: c[k - 1] for s, c in curves.items() if len(c) >= k}
            if len(fits) < len(Ys):
                continue
            fns = [KFitAct(h, fits[s][2], device) for s in sigs]
            set_fns(fns)
            errs = []
            for s in Ys:                             # in-distribution error on the model's own a
                xs = torch.cat(samples[s])
                errs.append(float((KFitAct(h, fits[s][2], "cpu")(xs) - Ys[s](xs)).abs().max()))
            row = dict(model=label or model_id, layers=n_layers, slots=len(slots), host=h, k=k,
                       shift=shift, R=(table[1] if table else R), R_model=R,
                       fit_target=(table[0] if table else "own"), fit_err=max(f[1] for f in fits.values()),
                       err_indist_max=max(errs), **score(),
                       pre=dict(max_abs=pre["max_abs"], frac_out6=pre["out6"] / max(pre["n"], 1)),
                       fit=fits[sigs[0]][2], n_prompts=len(enc))
            row["payable"], row["payable_loose"] = payable(row), payable(row, strict=False)
            rows.append(row)
            if sink:
                sink(row)
            print(f"  {h} x{k}{' shift' if shift else ''}: fit err {row['fit_err']:.2e} (in-dist "
                  f"{row['err_indist_max']:.2e})  top1 {row['top1']:.4f}  ex32 {row['ex32']:.2f}  "
                  f"ex64 {row['ex64']:.2f}  KL {row['kl']:.2e}  {'PAYABLE' if row['payable'] else ''}",
                  file=sys.stderr, flush=True)
    set_fns(None)
    print(f"  done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--hosts", nargs="+", required=True)
    ap.add_argument("--ks", type=int, nargs="+", required=True)
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--story", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--label")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--out", default="act/data/act_width_agreement.jsonl")
    ap.add_argument("--table", help="use the K-table fits of this target (e.g. GeLU, gelu_tanh, SiLU)")
    ap.add_argument("--table-R", type=float, default=160.0)
    ap.add_argument("--table-path", default="act/data/act_width_fits.json")
    ap.add_argument("--tf5", action="store_true",
                    help="run under transformers 5.18 (gpt-oss, nanochat), as act/act_swap_agreement.py --tf5")
    a = ap.parse_args()
    if a.tf5 and not os.environ.get("ACT_SWAP_TF5"):
        import subprocess
        tf5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prelude = ("import sys, types\n"
                   "m = types.ModuleType('transformers.dependency_versions_check'); m.dep_version_check = lambda *a, **k: None\n"
                   "sys.modules['transformers.dependency_versions_check'] = m\n"
                   "sys.modules['fla'] = None\n"
                   f"sys.argv = {['act_width'] + sys.argv[1:]!r}\n"
                   "import runpy; runpy.run_module('act.act_width_agreement', run_name='__main__')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([tf5, root]), ACT_SWAP_TF5="1")
        sys.exit(subprocess.run([sys.executable, "-c", prelude], env=env).returncode)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.model, a.hosts, a.ks, a.device, a.shift, a.story, a.n_prompts, a.label, a.trust_remote_code, sink,
        table=(a.table, a.table_R, a.table_path) if a.table else None)


if __name__ == "__main__":
    main()
