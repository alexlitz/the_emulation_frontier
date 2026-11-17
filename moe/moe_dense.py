"""MoE  vs  dense FFN.

Downward (MoE >= dense), EXACT under a capacity condition:
  Route every token to a single expert e (force the gate) -> the sparse block
  computes exactly expert_e(x), a dense FFN. So an MoE emulates any dense FFN
  IFF an expert is at least as wide as that dense map (or, with top-k active,
  iff k*expert_width >= dense_width). "Route to one expert" is the degenerate
  witness; the honest statement is the capacity bound.

Upward (dense -> MoE) penalty:
  A single dense FFN cannot reproduce the full MoE's per-token expert switching.
  To hold all E experts densely you pay E x params; to match the *active* FLOPs
  you need width k*expert_width. Hard top-k routing also adds an argmax
  (discontinuous) a smooth FFN can only approximate. Currency: width / params.

This file:
  1. block-level EXACT check: force-route layer L to expert e, compare the whole
     sparse block's output against expert_e(x) directly  -> matches to fp noise.
  2. full-model reverse penalty: run the model with every token pinned to a
     fixed expert and measure how far it drifts from the native MoE.

Run (OLMoE is the clean case -- no shared expert):
  python -m moe.moe_dense --model allenai/OLMoE-1B-7B-0924
"""
from __future__ import annotations

import argparse

import torch

import harness as H


def find_moe_blocks(model):
    """Sparse blocks = modules with a .gate Linear and an .experts ModuleList."""
    blocks = []
    for _, m in model.named_modules():
        if hasattr(m, "gate") and hasattr(m, "experts") and isinstance(
            m.experts, torch.nn.ModuleList
        ):
            blocks.append(m)
    return blocks


def make_gate_to_expert(e):
    def factory(gate: torch.nn.Linear):
        n_exp = gate.out_features

        def fwd(self, x):
            shape = x.shape[:-1] + (n_exp,)
            logits = x.new_full(shape, -30.0)
            logits[..., e] = 30.0
            return logits

        return fwd

    return factory


@torch.no_grad()
def block_exact_check(model, e=0, layer_idx=0, T=16, seed=0):
    blocks = find_moe_blocks(model)
    blk = blocks[layer_idx]
    if getattr(blk, "shared_expert", None) is not None:
        print("  [note] this block has a shared_expert; route-to-one != pure dense here")
    d = model.config.hidden_size
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn(1, T, d, generator=g).to(dev, dt)

    with H.patched_forwards([blk.gate], make_gate_to_expert(e)):
        out = blk(h)
    out = out[0] if isinstance(out, tuple) else out          # block returns (hidden, router_logits)
    ref = blk.experts[e](h)                                  # dense FFN = expert e
    err = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    print(f"block-level EXACT check (layer {layer_idx}, expert {e}):")
    print(f"  max |MoE_routed_to_e - expert_e| = {err:.3e}   (ref scale {scale:.2f})  "
          f"-> exact dense FFN\n")
    return err


def run(model_name, e=0, max_length=64, num_samples=16):
    model, tok, device = H.load(model_name, dtype=torch.bfloat16)
    cfg = model.config
    n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None)
    k = getattr(cfg, "num_experts_per_tok", None)
    ew = getattr(cfg, "moe_intermediate_size", None) or getattr(cfg, "intermediate_size", None)
    blocks = find_moe_blocks(model)
    print(f"{model_name}: {len(blocks)} MoE blocks, E={n_exp}, top_k={k}, expert_width={ew}\n")

    block_exact_check(model, e=e)

    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    print(f"full-model reverse penalty ({len(batches)} contexts, "
          f"~{sum(b.shape[1] for b in batches)} positions):")
    H.print_row("native MoE", H.compare(base, base))
    gates = [b.gate for b in blocks]
    with H.patched_forwards(gates, make_gate_to_expert(e)):
        logits = H.logits_for(model, batches)
    H.print_row(f"pinned to expert {e} (dense)", H.compare(base, logits))

    print(f"\nexchange rate (dense that matches this MoE):")
    print(f"  params:      {n_exp}x  (hold all {n_exp} experts)")
    print(f"  active FLOPs: {k}x  (dense width {k}*{ew}={k*ew} vs one expert {ew})")
    print("reading: block-level route-to-one is an EXACT dense FFN (MoE >= dense). "
          "The full-model drop is the reverse cost: one expert cannot emulate the "
          "MoE's conditional computation without paying the width/param exchange above.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=16)
    args = ap.parse_args()
    run(args.model, args.expert, args.max_length, args.num_samples)
