"""NoPE on a RoPE host as a LIMIT: linear rope scaling to infinity, measured on a real NoPE target.

graph/model_graph.py credits NoPE <- RoPE EXACTLY only where the host's config writes inv_freq = 0 (a pairs-free
longrope table, a rotary fraction of 0, or pass dims: zero_freq_room); rope_theta alone cannot (Lean
rope_theta_cannot_reach_nope: the first pair's frequency is theta^0 = 1). Round 3 (nope_rope_limit) adds the LIMIT
route: a host whose rotary reads linear scaling (transformers ROPE_INIT_FUNCTIONS["linear"]: inv_freq / factor,
modeling_rope_utils.py _compute_linear_scaling_rope_parameters) sets factor = s; every angle (i / s) omega_p -> 0 and
the logit tends to the unrotated q . k (Lean rope_scale_limit_identity / rope_scale_limit_bound / rope_scale_limit_attn,
proofs/lean/EmulationProofs/Position.lean).

The measurement: the target is a stock NoPE checkpoint (granite-4.0-h: GraniteMoeHybrid, position_embedding_type
"nope" on its attention layers); the host is the SAME family with its config switched to position_embedding_type
"rope", rope_scaling {"rope_type": "linear", "factor": s}: the rotary module that config builds
(GraniteMoeHybridRotaryEmbedding, modeling_granitemoehybrid.py:1250-1263, :1306) is installed on the target's weights
(the rotary has no parameters). s = 1 is the plain-RoPE control (the host config untouched but for the position type).
Scored with the act/act_swap_agreement.py standard on the target's own greedy 64-token continuations of the 24
prompts, fp32:

  payable  <=>  top-1 agreement >= 99%  AND  >= 90% of 32-token greedy continuations identical.

Also recorded per factor: the max |logit difference| over the scored positions and the largest angle the host applies
(max position x max inv_freq / s) -- fp32 resolves cos(x) = 1 for x below ~3e-4 and sin(x) = x far below that, so
the error is ~ the angle until it underflows to the exact identity.

MEASURED (2026-11-15, attn/data/nope_rope_limit_agreement.jsonl, fp32, cuda): granite-4.0-h-350m-base (4 NoPE
attention layers of 32, the rest Mamba-2), 24 prompts x 64 tokens:

  factor   top-1    ex32   ex64   KL        max |dlogit|   max angle
  1        71.8%    0.00   0.00   5.4e-1    1.9e2          77        (plain RoPE: the control)
  1e2      97.98%   0.75   0.33   6.0e-3    1.6e1          0.77
  1e4      100%     1.00   1.00   6.2e-7    1.9e-1         7.7e-3    payable
  1e6      100%     1.00   1.00   2.2e-10   1.9e-3         7.7e-5    payable
  1e9      100%     1.00   1.00   4.1e-11   3.3e-4         7.7e-8    payable (near the fp32 floor)
  1e12     100%     1.00   1.00   3.9e-14   5.7e-5         7.7e-11   payable
  inf      100%     1.00   1.00   0         0              0         (inv_freq / inf = 0: the limit point)

The error falls ~ 1/s until fp32 rounding (~1e-4 on logits of size ~10-50); 1e6 .. 1e12 is the useful range (no
underflow: inv_freq / s stays a normal fp32 number down to ~1e-38). The mixed-layer targets (SmolLM3, EXAONE-4,
Command-R7B, ...) have no weights in the local cache: not measured end to end (the catalog instance and the theorem
cover the per-table route).

  python -m attn.nope_rope_limit_agreement ibm-granite/granite-4.0-h-350m-base --device cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

from act.act_swap_agreement import N_NEW, PROMPTS, cont_logp

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FACTORS = (1.0, 1e2, 1e4, 1e6, 1e9, 1e12, float("inf"))


def greedy(model, ids, n_new=N_NEW):
    """Greedy continuation, no EOS stop, with the hybrid's own cache (HybridMambaAttentionDynamicCache: the Mamba
    states and the attention KV; act_swap_agreement.greedy's past_key_values is None for this family)."""
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache
    cache = HybridMambaAttentionDynamicCache(model.config, ids.shape[0], dtype=torch.float32, device=ids.device)
    T = ids.shape[1]
    pos = torch.arange(T, device=ids.device)
    out = model(ids, past_key_values=cache, use_cache=True, cache_position=pos, position_ids=pos[None])
    toks, nxt = [], out.logits[:, -1].argmax(-1)
    for t in range(n_new):
        toks.append(nxt)
        if len(toks) == n_new:
            break
        p = torch.tensor([T + t], device=ids.device)
        out = model(nxt[:, None], past_key_values=cache, use_cache=True, cache_position=p, position_ids=p[None])
        nxt = out.logits[:, -1].argmax(-1)
    return torch.stack(toks, 1)


def host_rotary(model, factor):
    """The rotary the host config builds: position_embedding_type "rope", rope_scaling linear x factor."""
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridRotaryEmbedding
    cfg = copy.deepcopy(model.config)
    cfg.position_embedding_type = "rope"
    cfg.rope_scaling = {"rope_type": "linear", "factor": factor}
    dev = next(model.parameters()).device
    return GraniteMoeHybridRotaryEmbedding(cfg, device=dev).to(dev)


def run(model_id, device="cpu", n_prompts=24, factors=FACTORS, sink=None, threads=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if threads:
        torch.set_num_threads(threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="eager").to(device)
    model.eval()
    body = model.model
    assert getattr(model.config, "position_embedding_type", None) == "nope" and body.rotary_emb is None, \
        "the target must be a NoPE checkpoint (position_embedding_type nope)"
    n_attn = sum(1 for t in model.config.layer_types if t == "attention")
    prompts = PROMPTS[:n_prompts]
    enc = [tok(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    with torch.no_grad():
        ref_cont = [greedy(model, ids) for ids in enc]
        ref_seq = [torch.cat([ids, c], 1) for ids, c in zip(enc, ref_cont)]
        ref_lp = [cont_logp(model, s, ids.shape[1]) for s, ids in zip(ref_seq, enc)]
        ref_logits = [model(s).logits[0].double().cpu() for s in ref_seq]
    t_max = max(s.shape[1] for s in ref_seq)
    print(f"{model_id}: {n_attn} NoPE attention layers of {model.config.num_hidden_layers}; reference done in "
          f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    rows = []
    for f in factors:
        body.rotary_emb = host_rotary(model, f)
        inv = body.rotary_emb.inv_freq.double()
        top1 = kl = 0.0
        n = ex32 = ex64 = 0
        div, mx = [], 0.0
        with torch.no_grad():
            for ids, s, lq, rc, rl in zip(enc, ref_seq, ref_lp, ref_cont, ref_logits):
                lp = cont_logp(model, s, ids.shape[1])
                kl += (lq.exp() * (lq - lp)).sum().item()
                top1 += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
                n += lp.shape[0]
                mx = max(mx, (model(s).logits[0].double().cpu() - rl).abs().max().item())
                g = greedy(model, ids)
                same = (g[0] == rc[0]).cpu().numpy()
                ex32 += bool(same[:32].all())
                ex64 += bool(same.all())
                div.append(int(np.argmin(same)) if not same.all() else N_NEW)
        row = dict(model=model_id, factor=f if np.isfinite(f) else "inf", n_attn_nope=n_attn,
                   max_angle=float(t_max * inv.max().item()), top1=top1 / n, kl=kl / n, ex32=ex32 / len(enc),
                   ex64=ex64 / len(enc), div=float(np.mean(div)), max_logit_diff=mx, n_prompts=len(enc),
                   dtype="float32",
                   payable=bool(top1 / n >= 0.99 and ex32 / len(enc) >= 0.90))
        rows.append(row)
        if sink is not None:
            sink(row)
        print(f"  factor {row['factor']!s:>8}: top1 {row['top1']:.4f}  ex32 {row['ex32']:.2f}  ex64 {row['ex64']:.2f}  "
              f"KL {row['kl']:.2e}  max|dlogit| {mx:.3e}  max angle {row['max_angle']:.2e}  "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    body.rotary_emb = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="attn/data/nope_rope_limit_agreement.jsonl")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--factors", default=",".join(str(f) for f in FACTORS))
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.model, a.device, a.n_prompts, tuple(float(x) for x in a.factors.split(",")), sink, a.threads)


if __name__ == "__main__":
    main()
