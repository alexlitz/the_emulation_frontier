"""Compile a GeLU-FFN model into a SiLU-FFN model (weight-level conversion).

A SiLU FFN reaches Swish_b(x) = SiLU(b x)/b by folding b into adjacent linears:
  down( gelu(up @ x) )  ~=  (down/b) ( SiLU( (b*up) @ x ) )
with b* = 1.7729 the best-fit (max-abs floor 0.0139). So the conversion is:
  up   *= b        (weight + bias)
  act  := SiLU
  down *= 1/b      (weight only; the additive down-bias is unchanged)
This is exact up to the GeLU~Swish floor -- the emulation currency is that floor.

Run:
  python -m convert.gelu_to_silu --model gpt2            # verify
  python -m convert.gelu_to_silu --model gpt2 --save out/gpt2-silu
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import harness as H

BETA = 1.772934   # best-fit Swish_beta to GeLU (see act/activations.py:gelu_silu_floor)


def convert_gpt2(model, beta=BETA):
    for blk in model.transformer.h:
        mlp = blk.mlp
        mlp.c_fc.weight.data.mul_(beta)          # up-projection (Conv1D: x @ W + b)
        mlp.c_fc.bias.data.mul_(beta)
        mlp.c_proj.weight.data.mul_(1.0 / beta)  # down-projection weight; bias untouched
        mlp.act = nn.SiLU()


def run(model_name="gpt2", beta=BETA, save=None, max_length=128, num_samples=48):
    assert "gpt2" in model_name, "this converter implements gpt2's Conv1D MLP"
    model, tok, device = H.load(model_name, dtype=torch.float32)
    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    print(f"compile {model_name}: GeLU-FFN -> SiLU-FFN  (beta={beta})")
    H.print_row("native GeLU", H.compare(base, base))
    convert_gpt2(model, beta)
    new = H.logits_for(model, batches)
    H.print_row("converted SiLU", H.compare(base, new))
    print("reading: agreement is high but not 1.0 -- the residual is exactly the GeLU~Swish")
    print("floor (0.0139 max-abs), the one emulation edge with no currency to reach exact.")
    if save:
        model.config.activation_function = "silu"
        model.save_pretrained(save)
        tok.save_pretrained(save)
        print(f"saved converted model to {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    run(args.model, args.beta, args.save)
