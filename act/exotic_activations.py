"""Exotic activations: which fold into SiLU, and which genuinely don't.

Four non-standard activations sit on the emulation frontier. Two REDUCE to SiLU (so a
plain SiLU/SwiGLU model emulates them and the model loses its activation-categorical
frontier status); two RESIST.

  swigluoai (gpt-oss / MiniMax-M3):  gate = g * sigmoid(1.702 g)
      == SiLU(1.702 g) / 1.702        -- EXACT fold: put 1.702 into W_gate and 1/1.702
                                         into W_down. (The +1 up-bias and +-7 clamp are
                                         the GLU-structure / regime parts, not the act.)
  mish (DanTagGen):  x * tanh(softplus(x))  ~= Swish_beta   -- APPROX, ~0.10 floor
                                         (same shape as GeLU ~= SiLU, a bit looser).

  situ (Kimi-K3):  beta*tanh(g/beta)*sigmoid(g) -> SiLU only as beta->inf, so situ
      GENERALIZES SiLU (SiLU is its limit); a SiLU model can't reproduce situ's
      saturation at the trained beta=4. Does NOT reduce -- Kimi-K3 stays frontier.
  xIELU (Apertus):  alpha_p*x^2 + beta*x  on x>0 -- a quadratic (super-linear) positive
      branch, unlike the sub-linear SiLU/GeLU/ReLU; alpha_n*(expm1(min(x,eps)) - x) + beta*x
      on x<=0 (nonlinear, -> +0.3|x|). HF's XIELUActivation defaults: alpha_p = alpha_n = 0.8,
      beta = 0.5, eps = -1e-6. Does NOT reduce to SiLU, and is NOT an exact relu^2 fold
      either (relu^2 -> xIELU single-op floor 0.166 on [-6,6]; see act/activation_cross.py).

Run:  python -m act.exotic_activations
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

F64 = torch.float64


def swigluoai_folds_into_silu(seed=0):
    g = torch.randn(4096, dtype=F64, generator=torch.Generator().manual_seed(seed)) * 2
    oai = g * torch.sigmoid(1.702 * g)              # gpt-oss / MiniMax-M3 gate
    silu_fold = F.silu(1.702 * g) / 1.702           # 1.702 folded into W_gate, 1/1.702 into W_down
    return (oai - silu_fold).abs().max().item()


def situ_limit_to_silu(beta):
    g = torch.randn(4096, dtype=F64, generator=torch.Generator().manual_seed(0)) * 2
    situ = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    return (situ - F.silu(g)).abs().max().item()


def mish_silu_floor():
    x = torch.linspace(-8, 8, 4001, dtype=F64)
    mish = x * torch.tanh(F.softplus(x))
    return min((mish - x * torch.sigmoid(b * x)).abs().max().item()
               for b in torch.linspace(0.5, 2.5, 400))


def xielu(x, alpha_p=0.8, alpha_n=0.8, beta=0.5, eps=-1e-6):
    """xIELU as transformers' XIELUActivation computes it, at its default init
    (alpha_p = softplus(log expm1 0.8) = 0.8, alpha_n = beta + softplus(log expm1 0.3) = 0.8,
    beta = 0.5, eps = -1e-6). Negative branch: alpha_n*(expm1(min(x, eps)) - x) + beta*x."""
    eps_t = torch.tensor(eps, dtype=x.dtype)
    return torch.where(x > 0, alpha_p * x * x + beta * x,
                       alpha_n * (torch.expm1(torch.minimum(x, eps_t)) - x) + beta * x)


def xielu_matches_hf(dtype=F64):
    """max|xielu - transformers.XIELUActivation(dtype)| on [-8, 8] (None if transformers missing)."""
    try:
        from transformers.activations import XIELUActivation
    except Exception:
        return None
    m = XIELUActivation(dtype=dtype)
    x = torch.linspace(-8, 8, 4001, dtype=dtype)
    with torch.no_grad():
        return (m._xielu_python(x).to(F64) - xielu(x.to(F64))).abs().max().item()


def xielu_positive_vs_silu():
    # xIELU positive branch alpha_p*x^2 + beta*x grows super-linearly; SiLU -> x (linear).
    x = torch.linspace(0, 8, 2001, dtype=F64)
    return (xielu(x) - F.silu(x)).abs().max().item()   # diverges -> not a reduction


if __name__ == "__main__":
    print(f"swigluoai == SiLU(1.702x)/1.702       max|d| = {swigluoai_folds_into_silu():.2e}   FOLD (exact)")
    for b in (4.0, 64.0, 256.0):
        tag = "  (trained beta=4: does NOT)" if b == 4.0 else ""
        print(f"situ(beta={b:5.0f}) -> SiLU            max|d| = {situ_limit_to_silu(b):.3e}   limit{tag}")
    print(f"mish  best-Swish floor                = {mish_silu_floor():.4f}   APPROX (like GeLU)")
    print(f"xIELU(+ branch) vs SiLU               max|d| = {xielu_positive_vs_silu():.1f}   NO reduction (quadratic)")


# --- ReLU on a swigluoai (gpt-oss) host unit: the up-bias route (Lean EP.relu_sub_swigluoai) ---------
SWIGLUOAI_LIMIT = 7.0
SWIGLUOAI_K = SWIGLUOAI_LIMIT / (1.0 + float(torch.exp(torch.tensor(-1.702 * SWIGLUOAI_LIMIT, dtype=F64))))
#   K = 7 sigmoid(1.702 * 7): the clamped gate branch's value once the gate is >= 7


def relu_swigluoai_rows(a, c):
    """The weights that make one clamped swigluoai unit  (clip(up, -7, 7) + 1) * g' sigmoid(1.702 g'),
    g' = min(gate, 7), play relu(z) (z the target unit's pre-activation): gate row c z, up row a z with
    up bias -1 (cancelling the unit's +1), down column 1 / (a K). Returns (gate_scale, up_scale,
    up_bias, down_scale). For |a z| <= 6 the up clamp is inactive; the gate branch is exactly K once
    c z >= 7, so the unit is relu(z) exactly for z >= 7 / c and tends to 0 for z < 0 as c -> inf.
    Scaling the gate alone (the route for SiLU, relu_sub_silu) does NOT work here: the gate clamp
    caps it at K, so swigluoai(c z, 0) / c = K / c -> 0 (EP.swigluoai_scaled_gate_vanishes)."""
    return float(c), float(a), -1.0, 1.0 / (float(a) * SWIGLUOAI_K)
