"""GeLU erf <-> tanh: is the swap PAYABLE?  Measured on real checkpoints, both directions.

The activation classifier splits exact (erf) GeLU from the tanh approximation (gelu_new /
gelu_pytorch_tanh). Their single-op distance is below ACT_FLOOR_MAX = 0.0005 but had never been
measured on a model:
  * raw swap (a = c = 1, d = e = f = 0, the other function as is): max |GeLU_erf - GeLU_tanh| on
    [-6, 6] = 4.7e-4;
  * best single-op fit a X(c x + d) + e x + f (act/activation_cross.py fit_minimax): 2.7e-4.

Targets: GPT-2 / GPT-2-medium (NewGELUActivation = tanh), Pythia-160m / 410m / 1.4b (GELUActivation =
erf), Gemma-2-2b (PytorchGELUTanh = tanh). Every FFN activation is replaced by the other variant (raw,
and the best fit), scored with the act_swap standard (act/act_swap_agreement.py: the original model's own
greedy 64-token continuations of 24 prompts, fp32, TF32 off, eager attention):
payable = top-1 >= 99% AND >= 90% of 32-token continuations identical.

Hosts (rows' "host"):  GeLU = erf fit, GeLU_tanh = tanh fit, GeLU_raw = erf as is,
GeLU_tanh_raw = tanh as is; "self" is the control, "bf16" the original in bf16.

  python -m act.gelu_swap_agreement gpt2 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import act.act_swap_agreement as asw
from act.activation_cross import ACTS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gelu_swap_agreement.jsonl")
MODELS = ("gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b",
          "google/gemma-2-2b")
HOSTS = ("GeLU", "GeLU_tanh", "GeLU_raw", "GeLU_tanh_raw")


def gelu_tanh_np(z):
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))


class _Raw:
    """marks a host used as is (no fit): fit_minimax returns the identity parameters."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, z):
        return self.fn(z)


def register():
    ACTS.setdefault("GeLU_tanh", gelu_tanh_np)
    ACTS.setdefault("GeLU_raw", _Raw(ACTS["GeLU"]))
    ACTS.setdefault("GeLU_tanh_raw", _Raw(gelu_tanh_np))
    asw.TACTS.setdefault("GeLU_tanh", lambda z: F.gelu(z, approximate="tanh"))
    asw.TACTS.setdefault("GeLU_raw", asw.TACTS["GeLU"])
    asw.TACTS.setdefault("GeLU_tanh_raw", lambda z: F.gelu(z, approximate="tanh"))
    if getattr(asw.fit_minimax, "_raw_aware", False):
        return
    fit = asw.fit_minimax

    def fit_or_raw(X, Y, *a, **k):
        if isinstance(X, _Raw):
            x = np.linspace(-6.0, 6.0, 120001)
            return float(np.max(np.abs(X(x) - Y(x)))), [1.0, 1.0, 0.0, 0.0, 0.0]
        return fit(X, Y, *a, **k)
    fit_or_raw._raw_aware = True
    asw.fit_minimax = fit_or_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    register()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def sink(r):
        r = dict(r, attn=asw.ATTN)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    for m in a.models:
        asw.run(m, a.device, hosts=HOSTS, fit_cache={}, only=["self", *HOSTS, "bf16"], sink=sink)
        torch.cuda.empty_cache()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
