"""A dense FFN  is a special case of  MoE  -- on the REAL MixtralSparseMoeBlock.

A Mixtral MoE layer routes each token to top-k of E expert MLPs and combines them
with gate weights that are normalized to sum to 1. If EVERY expert holds the same
dense SwiGLU MLP f, then for every token the combination is (sum of k normalized
gate weights) * f(x) = f(x) -- so the unmodified MoE block computes exactly the
dense MLP, for ANY router. Set all experts equal and check forward == one expert.

(This is the "dense fits inside MoE" direction; the reverse -- a dense FFN holding
all E experts -- is the capacity/depth cost measured in moe/moe_dense.py.)

Run:  python -m hf.moe_dense
"""
from __future__ import annotations

import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

F64 = torch.float64


def build(E=8, k=2, hidden=64, inter=128, seq=16, seed=0):
    torch.manual_seed(seed)
    cfg = MixtralConfig(hidden_size=hidden, intermediate_size=inter,
                        num_local_experts=E, num_experts_per_tok=k, hidden_act="silu")
    moe = MixtralSparseMoeBlock(cfg).double().eval()
    # tie every expert to expert 0 -> the block collapses to that dense SwiGLU MLP
    dense = moe.experts[0]
    for e in moe.experts[1:]:
        e.load_state_dict(dense.state_dict())
    x = torch.randn(1, seq, hidden, dtype=F64)
    with torch.no_grad():
        out = moe(x)
        out = out[0] if isinstance(out, tuple) else out          # (hidden[, router_logits])
        ref = dense(x.view(-1, hidden)).view(1, seq, hidden)     # the dense MLP alone
    return (out - ref).abs().max().item()


if __name__ == "__main__":
    for k in (1, 2, 4):
        print(f"MoE(E=8, top-{k}) with tied experts -> dense SwiGLU:  max|Δ| = {build(k=k):.2e}")
    print("-> the unmodified Mixtral MoE block, experts tied, IS a dense FFN (any router).")
