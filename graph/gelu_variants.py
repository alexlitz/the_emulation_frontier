"""GeLU erf <-> tanh, measured on real models.

PORTED in round 2, pass 5: graph/model_graph.ACT_FLOORS holds the 0.0003 cells both ways (approximate floors).

(A proposal module: graph/model_graph.py is not edited here.)

The activation classifier splits exact GeLU (erf: GELUActivation, nn.GELU()) from the tanh
approximation (gelu_new / gelu_pytorch_tanh / NewGELUActivation / PytorchGELUTanh). Their single-op
distance on [-6, 6]: 4.73e-4 swapped as is, 2.69e-4 with the best fit a X(c x + d) + e x + f -- both under
ACT_FLOOR_MAX = 0.0005, neither measured on a model until now.

MEASURED (2026-11-13, act/data/gelu_swap_agreement.jsonl; act/gelu_swap_agreement.py; the act_swap standard:
the original model's own greedy 64-token continuations of 24 prompts, fp32, TF32 off, eager attention;
payable = top-1 >= 99% AND >= 90% of 32-token continuations identical). Every FFN activation replaced:

  target (its GeLU)          host        swap     top-1    32-tok identical
  gpt2 (tanh)                erf         fit      99.87%   91.7%
                             erf         raw      99.93%   95.8%
  gpt2-medium (tanh)         erf         fit      99.93%   95.8%
                             erf         raw     100.00%  100.0%
  pythia-160m (erf)          tanh        fit      99.87%   95.8%
                             tanh        raw     100.00%   91.7%
  pythia-410m (erf)          tanh        fit/raw 100.00%  100.0%
  pythia-1.4b (erf)          tanh        fit/raw 100.00%  100.0%
  gemma-2-2b (tanh)          erf         fit      99.67%   91.7%
                             erf         raw      99.87%   95.8%

Payable in BOTH directions on all six models, raw swap included. The margin is thin: 22 / 24 identical on
three cells (90% is the bar), and gemma-2-2b's erf fit read 87.5% (FAIL) in the earlier act_swap run,
which loaded Gemma-2 with the default sdpa attention -- sdpa drops Gemma-2's attention-logit softcap
(hf/attn_pin.py), so that run scored a different model. Under eager (the checkpoint's own model) it passes.
For comparison the original model in bf16 fails on every one of them (top-1 61 - 98%, 0 - 38% identical).

PROPOSED: erf and tanh GeLU are one activation class for the graph -- an edge both ways at currency
"free" (the raw swap: the host's own activation, no refit), labelled APPROXIMATE (max error 4.7e-4, the
single-op floor), not exact. This matches ACT_FLOOR_MAX = 0.0005 (the threshold already admits both cells);
what was missing was the measurement, and it agrees.
"""
from __future__ import annotations

GELU_FLOOR_RAW = 4.73e-4
GELU_FLOOR_FIT = 2.69e-4
# (model, host) -> payable, as recorded (hosts: GeLU = erf fit, GeLU_tanh = tanh fit, *_raw = as is)
GELU_SWAP = {
    ("gpt2", "GeLU"): True, ("gpt2", "GeLU_raw"): True,
    ("gpt2-medium", "GeLU"): True, ("gpt2-medium", "GeLU_raw"): True,
    ("EleutherAI/pythia-160m", "GeLU_tanh"): True, ("EleutherAI/pythia-160m", "GeLU_tanh_raw"): True,
    ("EleutherAI/pythia-410m", "GeLU_tanh"): True, ("EleutherAI/pythia-410m", "GeLU_tanh_raw"): True,
    ("EleutherAI/pythia-1.4b", "GeLU_tanh"): True, ("EleutherAI/pythia-1.4b", "GeLU_tanh_raw"): True,
    ("google/gemma-2-2b", "GeLU"): True, ("google/gemma-2-2b", "GeLU_raw"): True,
}
# the rule: one class, both directions, free, approximate (not exact)
GELU_EDGE = {("GeLU_tanh", "GeLU"): ("free", False), ("GeLU", "GeLU_tanh"): ("free", False)}
