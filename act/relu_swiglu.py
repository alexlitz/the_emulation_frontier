"""Activation / FFN subset relations.

Exact constructions (verified numerically), plus a real-model reverse penalty.

1. ReLU is the large-beta limit of Swish.
     Swish_b(x) = x * sigmoid(b x)  ->  ReLU(x)  as b -> inf.
   Same shape as ALiBi = small-angle RoPE: a discrete op recovered as a limit of
   a continuous family. Currency: MAGNITUDE (b -> inf).

2. GLU-family >= vanilla FFN (same activation), exact, ~free.
   SwiGLU(x) = SiLU(x Wg) (.) (x Wu); pin the value branch (x Wu) to the constant
   1 (Wu=0, bias=1, or one constant input dim) -> SiLU(x Wg) -> vanilla SiLU FFN.
   Currency: one bias / one constant dim.

3. Bilinear-GLU  <=  squared-activation FFN, exact, 2x width.
   Product identity  a*b = 1/4[(a+b)^2 - (a-b)^2].  Two square units per output
   coordinate ((a+b) and (a-b)) reproduce the multiplicative gate exactly.
   Currency: 2x hidden width + a square activation.

4. SwiGLU  <=  TWO-layer squared FFN, exact -> DEPTH penalty.
   SwiGLU coord = SiLU(a)*b. SiLU(a) is nonlinear in x, so (SiLU(a)+b) is not
   producible by one linear layer before the square: you must compute SiLU(a)
   in layer 1, then the product in layer 2. Bilinear (no inner act) collapses
   back to ONE layer -- the inner nonlinearity is exactly what costs the depth.

Real model (Qwen2-0.5B, native SwiGLU): ablate the multiplicative gate and
measure the drift -- the reverse cost of the structure vanilla FFN lacks.

Run:
  python -m act.relu_swiglu                       # constructions (synthetic)
  python -m act.relu_swiglu --model Qwen/Qwen2-0.5B-Instruct   # + real ablation
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import harness as H


def sigmoid(x):
    return torch.sigmoid(x)


def swish_beta(x, b):
    return x * sigmoid(b * x)


# --- 1. ReLU = lim_{b->inf} Swish_b -------------------------------------------

def check_swish_to_relu():
    x = torch.linspace(-6, 6, 4001, dtype=torch.float64)
    relu = F.relu(x)
    print("1. ReLU as large-beta limit of Swish_b(x)=x*sigmoid(b x):")
    for b in [1, 4, 16, 64, 256, 1024]:
        err = (swish_beta(x, b) - relu).abs().max().item()
        print(f"     beta={b:>5}   max|Swish_b - ReLU| = {err:.3e}")
    print("   -> currency is MAGNITUDE beta, exactly like ALiBi = small-angle RoPE\n")


# --- 2. GLU-family >= vanilla FFN ---------------------------------------------

def check_glu_superset_vanilla(d=64, m=256, T=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, d, dtype=torch.float64, generator=g)
    Wg = torch.randn(d, m, dtype=torch.float64, generator=g)
    Wd = torch.randn(m, d, dtype=torch.float64, generator=g)
    vanilla = F.silu(x @ Wg) @ Wd
    # SwiGLU with value branch pinned to constant 1 (Wu=0, bias_u=1)
    up = torch.zeros(T, m, dtype=torch.float64) + 1.0
    swiglu_reduced = (F.silu(x @ Wg) * up) @ Wd
    err = (vanilla - swiglu_reduced).abs().max().item()
    print(f"2. GLU >= vanilla FFN (pin value branch to 1):  max err = {err:.3e}  (exact)\n")


# --- 3. Bilinear-GLU <= squared-activation FFN (2x width) ---------------------

def check_bilinear_as_square(d=64, m=128, T=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, d, dtype=torch.float64, generator=g)
    Wg = torch.randn(d, m, dtype=torch.float64, generator=g)
    Wu = torch.randn(d, m, dtype=torch.float64, generator=g)
    Wd = torch.randn(m, d, dtype=torch.float64, generator=g)
    a, b = x @ Wg, x @ Wu
    bilinear = (a * b) @ Wd
    # squared-activation FFN, 2m hidden units: [(a+b), (a-b)] -> square -> +1/4,-1/4
    h = torch.cat([a + b, a - b], dim=-1)
    prod = 0.25 * (h[:, :m] ** 2) - 0.25 * (h[:, m:] ** 2)   # == a*b
    square_ffn = prod @ Wd
    err = (bilinear - square_ffn).abs().max().item()
    print(f"3. Bilinear-GLU == squared-FFN at 2x width:  max err = {err:.3e}  (exact)\n")


# --- 4. SwiGLU <= two-layer squared FFN (depth) -------------------------------

def check_swiglu_needs_depth(d=64, m=128, T=32, seed=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, d, dtype=torch.float64, generator=g)
    Wg = torch.randn(d, m, dtype=torch.float64, generator=g)
    Wu = torch.randn(d, m, dtype=torch.float64, generator=g)
    Wd = torch.randn(m, d, dtype=torch.float64, generator=g)
    a, b = x @ Wg, x @ Wu
    swiglu = (F.silu(a) * b) @ Wd
    # layer 1: p = SiLU(a)  (nonlinear) ; b passes through linearly
    p = F.silu(a)
    # layer 2: product p*b via the square trick, then down-project
    h = torch.cat([p + b, p - b], dim=-1)
    prod = 0.25 * (h[:, :m] ** 2) - 0.25 * (h[:, m:] ** 2)   # == p*b
    two_layer = prod @ Wd
    err = (swiglu - two_layer).abs().max().item()
    print(f"4. SwiGLU == TWO-layer squared FFN:  max err = {err:.3e}  (exact)")
    print("   the inner SiLU forces layer 1; bilinear (no inner act) needs only 1 layer")
    print("   -> the inner nonlinearity is exactly what costs the DEPTH\n")


# --- real model: ablate the SwiGLU gate --------------------------------------

def _mlp_variant(kind):
    def factory(mlp):
        gate, up, down, act = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn

        def fwd(self, x):
            if kind == "no_gate":      # vanilla: drop the multiplicative up-branch
                return down(act(gate(x)))
            if kind == "bilinear":     # drop the inner activation (identity act)
                return down(gate(x) * up(x))
            raise ValueError(kind)

        return fwd

    return factory


def run_real(model_name, max_length=128, num_samples=32):
    model, tok, device = H.load(model_name, dtype=torch.float32)
    mlp_cls = type(model.model.layers[0].mlp).__name__
    mlps = H.modules_of_type(model, mlp_cls)
    print(f"real model {model_name}: {len(mlps)} {mlp_cls} (SwiGLU) blocks")
    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    H.print_row("native SwiGLU", H.compare(base, base))
    for kind in ["bilinear", "no_gate"]:
        with H.patched_forwards(mlps, _mlp_variant(kind)):
            logits = H.logits_for(model, batches)
        H.print_row(kind, H.compare(base, logits))
    print("reading: 'bilinear' removes the inner SiLU (keeps the gate); 'no_gate' "
          "removes the multiplicative gate (vanilla FFN). Both are the reverse cost "
          "of the structure the simpler FFN lacks.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="optional native-SwiGLU model to ablate")
    args = ap.parse_args()
    check_swish_to_relu()
    check_glu_superset_vanilla()
    check_bilinear_as_square()
    check_swiglu_needs_depth()
    if args.model:
        run_real(args.model)
