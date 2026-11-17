"""The REVERSE direction: a dense stack reproducing a hard-top-1 MoE.

MoE >= dense is easy (route to one expert; see moe/moe_dense.py). The load-bearing
claim is the reverse -- what a dense network must spend to reproduce the MoE's
per-token expert switching. We build it honestly and pin down the currency:

  (1) EXACT, with an explicit hard-SELECT layer. Hold ALL E experts densely
      (width E*ew) and select the routed one with the MoE's own argmax mask ->
      matches the MoE bit-for-bit. Currency: E x params (hold every expert) + a
      routing/select layer (depth).

  (2) The select is an ARGMAX -- discontinuous. A smooth (ReLU/softmax) router
      only APPROACHES it as sharpness beta->inf (a limit, exactly like ReLU<=SiLU),
      and is exact only away from routing boundaries (a margin). So dense->MoE is a
      capacity + LIMIT reduction, NOT a free/exact one. That is the honest currency.

Run:  python -m moe.dense_emulates_moe
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

F64 = torch.float64


def experts_all(x, W1, W2):
    """Every expert's output, densely. W1 (E,d,h), W2 (E,h,d) -> (T,E,d)."""
    H = F.silu(torch.einsum("td,edh->teh", x, W1))
    return torch.einsum("teh,ehd->ted", H, W2)


def moe_top1(x, W1, W2, Wg):
    """Reference MoE: route each token to argmax expert, output that expert (top-1)."""
    Y = experts_all(x, W1, W2)
    e = (x @ Wg).argmax(-1)
    return Y[torch.arange(x.shape[0]), e]


def dense_hardselect(x, W1, W2, Wg):
    """Dense witness: hold all experts (E*ew wide) + a hard argmax select layer."""
    Y = experts_all(x, W1, W2)                                  # E x params held densely
    mask = F.one_hot((x @ Wg).argmax(-1), Wg.shape[1]).to(F64)  # the routing/select layer
    return torch.einsum("te,ted->td", mask, Y)


def dense_softrouter(x, W1, W2, Wg, beta):
    """Smooth router: softmax(beta*logits). -> hard select as beta->inf (a limit)."""
    Y = experts_all(x, W1, W2)
    w = F.softmax(beta * (x @ Wg), -1)
    return torch.einsum("te,ted->td", w, Y)


def dense_masked_depth(x, W1, W2, Wg):
    """The DEPTH realization: instead of one wide select, process experts across layers.
    A masking layer computes the router's hard top-1 mask; then E residual FFN sublayers
    each add mask_e * expert_e(x) to the stream. This is the "2x layers + masking" form --
    the extra depth is the routing/mask circuit the graph charges 2x depth for. Exact,
    with the same caveat: producing the hard mask is the discontinuous step (a limit if
    you build it from smooth comparators instead of a hard argmax)."""
    mask = F.one_hot((x @ Wg).argmax(-1), Wg.shape[1]).to(x.dtype)   # the mask layer
    acc = torch.zeros(x.shape[0], W2.shape[2], dtype=x.dtype)
    for e in range(Wg.shape[1]):                                     # one FFN sublayer per expert
        acc = acc + mask[:, e:e + 1] * (F.silu(x @ W1[e]) @ W2[e])   # masked residual add
    return acc


def dense_two_layer_margin(z, P, V, lam):
    """The norm-free 2-FFN-layer dense host of a top-1 unit-gate MoE with ReLU experts (Lean
    EP.moe_two_layer_margin). z (..., E) router logits, P (..., E, ew) expert unit pre-activations,
    V (E, ew, d) down columns. Select FFN (E^2 ReLU units): s_e = sum_a relu(z_a - z_e) into E extra
    channels (0 for the winner, >= delta for a loser on a margin delta). Expert FFN: masked units
    relu(p_ej - lam s_e) v_ej, plus E units relu(s_e) writing -s_e (the erase). Returns (output,
    the select channels after the erase). Exact when the winner beats every loser by delta and
    lam delta >= every loser's pre-activation."""
    s = F.relu(z[..., None, :] - z[..., :, None]).sum(-1)          # [.., e] = sum_a relu(z_a - z_e)
    out = torch.einsum("...ej,ejd->...d", F.relu(P - lam * s[..., None]), V)
    return out, s - F.relu(s)


def build(T=32, d=16, ew=24, E=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    x, W1, W2, Wg = r(T, d), r(E, d, ew), r(E, ew, d), r(d, E)
    ref = moe_top1(x, W1, W2, Wg)
    exact = (dense_hardselect(x, W1, W2, Wg) - ref).abs().max().item()      # width + select
    masked = (dense_masked_depth(x, W1, W2, Wg) - ref).abs().max().item()   # 2x layers + masking
    errs = [(b, (dense_softrouter(x, W1, W2, Wg, b) - ref).abs().max().item())
            for b in (1, 4, 16, 64, 256, 1024)]
    return exact, masked, errs


if __name__ == "__main__":
    exact, masked, errs = build()
    print("dense (hold all E experts + hard argmax select) vs top-1 MoE:")
    print(f"  width + select : max|Δ| = {exact:.2e}   (E x width, one select layer)")
    print(f"  2x layers+mask : max|Δ| = {masked:.2e}   (E masked FFN sublayers -> depth)\n")
    print("smooth router (softmax beta) -> hard select as beta->inf (a LIMIT):")
    for b, e in errs:
        print(f"  beta={b:>5}   max|Δ| = {e:.3e}")
    print("  -> the argmax is discontinuous; a smooth dense router only reaches it in the")
    print("     limit (exact away from routing boundaries). dense->MoE = capacity + limit.")
