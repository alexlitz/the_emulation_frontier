"""A start token's stream on its own channel (Lean EP.bias_free_ffn_zero).

A bias-free FFN (any activation with phi(0) = 0, or a gated unit phi(g) u) reading zero content writes
zero. So a start token whose embedding lives only on an extra residual channel that the target layers do
not read keeps that embedding at every layer: its normed content is zero on the channels the q/k/v and
MLP projections read, its value is zero, and at position 0 its attention sees only itself. That is the
stream half of the start-token sink construction.

* `is_bias_free(module)`: every nn.Linear inside has no bias.
* `marker_host(target, e, kappa)`: the hidden-padded host of a stock Llama-family target
  (elephantine/hidden_width.widen, width d + e, readers with zero columns on the pad) with token id
  `marker_id` re-embedded as kappa * e_{d} (the first pad channel). Returns (host, marker_id).
"""
from __future__ import annotations

import torch


def is_bias_free(module):
    return all(m.bias is None for m in module.modules() if isinstance(m, torch.nn.Linear))


@torch.no_grad()
def marker_host(target, e, kappa=1.0, marker_id=0):
    from elephantine import hidden_width as hw
    d = target.config.hidden_size
    host = hw.widen(target, d + e)
    emb = host.get_input_embeddings().weight
    emb[marker_id].zero_()
    emb[marker_id, d] = kappa
    return host, marker_id
