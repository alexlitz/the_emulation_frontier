"""Constant-norm extra channels under an RMSNorm (Lean EP.const_norm_carry, EP.rmsNorm_zero_sq_sum).

* The target's eps is a constant-norm budget. A target RMSNorm divides by sqrt(|x|^2/d + eps): the d eps is
  a virtual channel of fixed squared norm. A host residual [x; z] whose extra channels have a FIXED squared
  norm S reads the stream exactly with eps' = (d eps - S)/(d + e) and weight [sqrt(d/(d+e)) w; wz]; the
  extra channels come out as wz z / R, R = sqrt(d/(d+e)) sqrt(|x|^2/d + eps). `const_norm_host` returns that
  host norm's (weight, eps'); it needs S <= d eps for eps' >= 0.
* With eps = 0 and weight 1 the normed vector has squared norm exactly m (its width). A bias-free GLU with
  2m units silu(u_i) u_i - silu(-u_i) u_i = u_i^2 therefore writes the exact constant m:
  `glu_const_writer(mlp, channel)` sets a stock gated MLP (transformers LlamaMLP layout: gate_proj, up_proj,
  down_proj, SiLU) to write m into `channel`.
"""
from __future__ import annotations

import math

import torch


def const_norm_eps(d, e, eps, S):
    """eps' = (d eps - S) / (d + e)."""
    return (d * eps - S) / (d + e)


def const_norm_host(w, eps, wz, S):
    """The host RMSNorm (weight [sqrt(d/(d+e)) w; wz], eps') of EP.const_norm_carry for extra channels of
    squared norm S. Raises if eps' < 0 (S > d eps)."""
    d, e = w.shape[-1], wz.shape[-1]
    eps2 = const_norm_eps(d, e, eps, S)
    if eps2 < 0:
        raise ValueError(f"S = {S} exceeds the budget d eps = {d * eps}")
    return torch.cat([math.sqrt(d / (d + e)) * w, wz], -1), eps2


def carried_factor(x, eps, e):
    """R = sqrt(d/(d+e)) sqrt(|x|^2/d + eps): the extra channels come out as wz z / R."""
    d = x.shape[-1]
    return math.sqrt(d / (d + e)) * torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


@torch.no_grad()
def glu_const_writer(mlp, channel=0):
    """Set a stock gated MLP of width m (input) and intermediate >= 2m to write sum_i u_i^2 into `channel`:
    unit 2i is silu(u_i) * u_i, unit 2i + 1 is silu(-u_i) * u_i, down_proj +1 / -1. After an eps = 0, weight-1
    RMSNorm that is exactly m (EP.rmsNorm_zero_sq_sum, EP.silu_identity_pair). Units past 2m are zeroed."""
    m = mlp.gate_proj.in_features
    assert mlp.gate_proj.out_features >= 2 * m, "needs 2m units"
    for p in (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight):
        p.zero_()
    i = torch.arange(m)
    mlp.gate_proj.weight[2 * i, i] = 1.0
    mlp.gate_proj.weight[2 * i + 1, i] = -1.0
    mlp.up_proj.weight[2 * i, i] = 1.0
    mlp.up_proj.weight[2 * i + 1, i] = 1.0
    mlp.down_proj.weight[channel, 2 * i] = 1.0
    mlp.down_proj.weight[channel, 2 * i + 1] = -1.0
    return mlp
