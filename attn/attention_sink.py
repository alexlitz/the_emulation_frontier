"""Attention sinks (gpt-oss, StreamingLLM) = standard softmax + a zero-value null token.

gpt-oss's distinctive primitive: a learned per-head bias `b` in the softmax
DENOMINATOR (off-by-one attention), letting a head attend to "nothing":

    a_j = exp(s_j) / (exp(b) + Σ_k exp(s_k))          (mass exp(b)/(...) is dumped)

This is EXACTLY a standard-softmax attention with one extra "null" key whose score
is `b` and whose value is 0: softmax over [s ; b] applied to [V ; 0] reproduces it,
because the null token absorbs probability but contributes 0 to the output.

So attention-sink and standard attention are mutually emulable:
  standard >= sink :  set b -> -inf   (free)
  sink emulates a target with a sink : it already has one
  standard emulates sink : append one zero-value null token  (currency: +1 KV slot)
A near-free width edge -- a sink is just an always-present register that holds nothing.

Run:  python -m attn.attention_sink
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def sink_attention(scores, V, b):
    a = torch.exp(scores) / (math.exp(b) + torch.exp(scores).sum())
    return a @ V


def emulate_with_null_token(scores, V, b):
    s_aug = torch.cat([scores, torch.tensor([b], dtype=F64)])
    V_aug = torch.cat([V, torch.zeros(1, V.shape[1], dtype=F64)])
    return F.softmax(s_aug, dim=0) @ V_aug


def run(N=32, dv=16):
    torch.manual_seed(0)
    scores = torch.randn(N, dtype=F64)
    V = torch.randn(N, dv, dtype=F64)
    print("attention sink (gpt-oss) == standard softmax + one zero-value null token:")
    for b in [-4.0, 0.0, 2.0, 5.0]:
        err = (sink_attention(scores, V, b) - emulate_with_null_token(scores, V, b)).abs().max().item()
        dumped = math.exp(b) / (math.exp(b) + torch.exp(scores).sum().item())
        print(f"     sink logit b={b:>5.1f}  (mass dumped {dumped:5.1%})   max err = {err:.3e}")
    # reverse: sink -> standard by b -> -inf
    err = (sink_attention(scores, V, -50.0) - F.softmax(scores, 0) @ V).abs().max().item()
    print(f"   reverse: b -> -inf recovers standard softmax exactly (err {err:.3e}) -- free.")
    print("   -> sinks are a +1-KV-slot width edge; standard attention emulates them with a")
    print("      null register token, and drops them for free. Not a new mechanism, a fold.")


if __name__ == "__main__":
    run()
