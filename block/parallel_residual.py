"""Block structure: parallel vs sequential attention+FFN.

  sequential (GPT-2, Llama):  h = x + Attn(N1 x);   y = h + MLP(N2 h)
  parallel   (GPT-J, NeoX):   y = x + Attn(N1 x) + MLP(N2 x)

Parallel is a RESTRICTION of sequential: it drops the attn->MLP dependency (the MLP
reads the PRE-attention x) so attention and MLP run concurrently -- a throughput
optimization. So SEQUENTIAL is the more general structure, and both simulate each
other, at different currencies:

  sequential >= parallel  at 2x WIDTH, SAME depth:
     carry pre-attention x in reserved channels; wire the MLP to read those ->
     M(x), giving x + A(x) + M(x) in ONE block.
  parallel >= sequential  at 2x DEPTH:
     an attn-only block then an mlp-only block reproduces one sequential block:
     b1 = x + A(x);  b2 = b1 + M(b1) = x + A(x) + M(x+A(x)) == sequential.

The depth-preserving direction (sequential emulating parallel) is why sequential
is "more general": it hosts parallel at 1:1 layers; parallel must double its layers.

Part 1 verifies BOTH constructions (exact) in the NORM-FREE toy: each sublayer here normalises
only its own input z, so the "reserved copy" read in 1b is normalised by its own rms.
Under a REAL pre-norm (one rms over the whole residual, reserved channels included) the two
directions split (block/prenorm_carry.py, stock transformers layers):
  parallel >= sequential at 2x DEPTH  stays EXACT (stock GPT-NeoX layers, max|d| = 0).
  sequential >= parallel at 2x WIDTH  is NOT exact at any width. The MLP's norm sees the
     attention write through the shared rms. On a stock LlamaDecoderLayer it is off by ~1e-2,
     and exact only with a per-group norm (a norm change, not width).
  For a whole L-block model, NORM-FREE, the width stays 2d at every L (Lean EP.parallel_sub_seq_2x,
  parallel_stack_in_seq_2x below): the MLP sublayer, which may read and write the whole residual, moves
  the attention write out of the spare channel with 2d identity units phi(v) - phi(-v) = v and erases it,
  so the spare channel is 0 again at every block boundary. The price is 2d extra MLP units per block
  (MLP width f + 2d), not extra width. Without that move (the attention write only dropped by later
  readouts, parallel_stack_in_seq_wide) every block needs fresh channels: (L+1) d. Under a per-group norm
  the identity units read the attention write normalised, so the move is not the norm-free one; that case
  is not covered here. Part 2 flips a real parallel model (Pythia) to sequential and measures the drift --
  genuinely different ops.

Run:
  python -m block.parallel_residual --model EleutherAI/pythia-160m
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import harness as H

F64 = torch.float64


def _rms(z, w, eps=1e-5):
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * w


def parallel_stack(x, As, Ms):
    """The target: L parallel blocks  x <- x + A_l(x) + M_l(x)."""
    for A, M in zip(As, Ms):
        x = x + A(x) + M(x)
    return x


def parallel_stack_in_seq_wide(x, As, Ms):
    """L parallel blocks in L SEQUENTIAL blocks of width (L+1) d (norm-free: each sublayer reads a
    linear readout of the residual and normalises only that). Channel 0 is the main stream, channel
    l+1 is reserved for block l's attention write. Block l: the attention reads the sum of all channels
    and writes channel l+1; the MLP then reads every channel EXCEPT l+1 -- the block's input -- and
    writes channel 0. The target residual is the sum of the channels. In THIS layout (the attention
    write is only ever dropped by readouts, never moved) every block's write needs its own channel. It
    is not the minimum: parallel_stack_in_seq_2x does any L at width 2d (Lean EP.parallel_sub_seq_2x),
    paying 2d extra MLP units per block instead."""
    ch = [x] + [torch.zeros_like(x) for _ in As]
    for l, (A, M) in enumerate(zip(As, Ms)):
        ch[l + 1] = ch[l + 1] + A(sum(ch))
        ch[0] = ch[0] + M(sum(c for i, c in enumerate(ch) if i != l + 1))
    return sum(ch)


def parallel_stack_in_seq_2x(x, As, Ms, phi=F.relu):
    """L parallel blocks in L SEQUENTIAL blocks of width 2d, norm-free (Lean EP.parallel_sub_seq_2x).
    Residual (c0, c1), c1 = 0 at every block boundary. Block l: the attention reads c0 and writes
    A_l(c0) into c1; the MLP reads c0 (still the block input) and writes M_l(c0) into c0, and with 2d
    identity units phi(v) - phi(-v) = v (ReLU exactly; SiLU, GeLU up to rounding) moves c1 into c0 and
    erases c1. Returns the list of (c0, c1) after every block; the target after block l is c0."""
    c0, c1, out = x, torch.zeros_like(x), []
    for A, M in zip(As, Ms):
        c1 = c1 + A(c0)                                   # attention sublayer: reads c0, writes c1
        idu = phi(c1) - phi(-c1)                          # 2d identity units on c1
        c0, c1 = c0 + M(c0) + idu, c1 - idu               # MLP sublayer: reads (c0, c1), writes both
        out.append((c0, c1))
    return out


def verify_both_constructions(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    T, d, m = 8, 64, 128
    x = r(T, d)
    n1, n2, Wa, Wm1, Wm2 = r(d), r(d), r(d, d), r(d, m), r(m, d)
    A = lambda z: _rms(z, n1) @ Wa                        # an "attention" sublayer
    M = lambda z: F.silu(_rms(z, n2) @ Wm1) @ Wm2         # an "MLP" sublayer
    par = x + A(x) + M(x)                                 # parallel block
    seq = x + A(x) + M(x + A(x))                          # sequential block

    # parallel >= sequential at 2x DEPTH: two parallel blocks (attn-only, mlp-only)
    b1 = x + A(x)
    b2 = b1 + M(b1)
    print("1a. parallel >= sequential at 2x DEPTH (two parallel blocks == one sequential):")
    print(f"      max err = {(seq - b2).abs().max().item():.3e}   (exact)")

    # sequential >= parallel at 2x WIDTH: reserved channels carry pre-attention x,
    # the MLP reads those -> M(x). One sequential block, 2x width.
    main = x + A(x)                                       # attn writes to main
    reserved = x                                          # residual carries pristine x
    seq_wide = main + M(reserved)                         # MLP reads reserved (pre-attn x)
    print("1b. sequential >= parallel at 2x WIDTH, same depth (MLP reads a reserved copy of x):")
    print(f"      max err = {(par - seq_wide).abs().max().item():.3e}   (exact)")
    print("    -> both simulate each other; sequential is more general (width, not depth)\n")


def run_real(model_name, max_length=128, num_samples=48):
    model, tok, device = H.load(model_name, dtype=torch.float32)
    layers = model.gpt_neox.layers
    assert getattr(layers[0], "use_parallel_residual", False), "model is not parallel-residual"
    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)                  # native parallel
    print(f"2. real model {model_name} ({len(layers)} parallel-residual blocks):")
    H.print_row("native (parallel)", H.compare(base, base))
    for L in layers:                                     # flip to sequential in place
        L.use_parallel_residual = False
    seq = H.logits_for(model, batches)
    for L in layers:
        L.use_parallel_residual = True
    H.print_row("forced sequential", H.compare(base, seq))
    print("reading: same weights, same depth -- but reading the MLP off the post-attention")
    print("stream diverges hard. Parallel and sequential are a real structural axis; each")
    print("emulates the other (sequential->parallel needs 2x width; parallel->seq 2x depth).\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    args = ap.parse_args()
    verify_both_constructions()
    run_real(args.model)
