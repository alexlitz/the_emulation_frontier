"""Value embeddings / value residuals on a stock attention module, and the switch that turns them off
(Lean EP.value_emb_off).

A value-embedding head (modded-nanogpt's value embeddings, ResFormer's value residual) adds a mixed-in
term to every value: v_j + lam * u_j, where u_j is a per-token embedding (value embeddings) or the first
layer's value (value residual). `add_value_mix(attn, lam)` installs that on a stock transformers attention
module through a forward hook on its v_proj; the extra term is set per forward with `set_mix(attn, u)`.
`value_emb_off(attn)` is the weight setting lam = 0: then the head's values are the plain values, so a
value-embedding host plays a target without them (free, exact).
"""
from __future__ import annotations

import torch


class ValueMix:
    """The v_proj hook: out + lam * u (u (.., T, kv_dim), set before each forward)."""

    def __init__(self, lam):
        self.lam = torch.as_tensor(lam)
        self.u = None

    def __call__(self, module, args, out):
        if self.u is None:
            return out
        return out + self.lam.to(out.dtype) * self.u.to(out.dtype)


def add_value_mix(attn, lam):
    """Install the value mix on `attn.v_proj` (in place). Returns the ValueMix."""
    vm = ValueMix(lam)
    attn._value_mix = vm
    attn._value_mix_handle = attn.v_proj.register_forward_hook(vm)
    return vm


def set_mix(attn, u):
    attn._value_mix.u = u


def value_emb_off(attn):
    """EP.value_emb_off's weight setting: the mixing weight lam = 0."""
    attn._value_mix.lam = torch.zeros_like(attn._value_mix.lam)
    return attn
