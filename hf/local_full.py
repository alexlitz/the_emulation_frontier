"""Local (sliding-window) attention  is a special case of  full attention.

A sliding-window layer is just full attention with a BAND mask (query i sees keys
in (i-W, i]). So the unmodified full-attention Qwen2Model, handed that band as its
native `attention_mask` argument, computes exactly what a Qwen2Model *configured*
with sliding_window=W computes -- same class, same weights, only the mask differs.

We build a full model (sliding disabled) and a windowed model (sliding on every
layer) with identical weights, and check  full(x, band) == windowed(x).

Run:  python -m hf.local_full
"""
from __future__ import annotations

import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

F64 = torch.float64


def _cfg(sliding, W, hidden=64, layers=2, heads=8, inter=128, vocab=32):
    return Qwen2Config(hidden_size=hidden, num_hidden_layers=layers,
                       num_attention_heads=heads, num_key_value_heads=heads,
                       intermediate_size=inter, vocab_size=vocab,
                       max_position_embeddings=64, rms_norm_eps=1e-6,
                       use_sliding_window=sliding, sliding_window=W,
                       max_window_layers=0, attn_implementation="eager")


def band_mask(seq, W):
    """additive causal band: query i attends to keys j with i-W < j <= i."""
    i = torch.arange(seq)[:, None]
    j = torch.arange(seq)[None, :]
    allow = (j <= i) & (j > i - W)
    m = torch.zeros(seq, seq, dtype=F64)
    m.masked_fill_(~allow, torch.finfo(F64).min)
    return m.view(1, 1, seq, seq)


def build(W=4, seq=12, seed=0):
    torch.manual_seed(seed)
    full = Qwen2Model(_cfg(False, W)).double().eval()          # full-attention config
    windowed = Qwen2Model(_cfg(True, W)).double().eval()       # sliding-window config
    windowed.load_state_dict(full.state_dict())                # identical weights
    x = torch.randn(1, seq, 64, dtype=F64)
    with torch.no_grad():
        o_full_banded = full(inputs_embeds=x, attention_mask=band_mask(seq, W)).last_hidden_state
        o_windowed = windowed(inputs_embeds=x).last_hidden_state
        o_full_plain = full(inputs_embeds=x).last_hidden_state  # sanity: full != windowed
    return ((o_full_banded - o_windowed).abs().max().item(),
            (o_full_plain - o_windowed).abs().max().item())


if __name__ == "__main__":
    for W in (2, 4, 8):
        eq, diff = build(W=W)
        print(f"W={W}: full(x, band) vs windowed(x)  max|Δ| = {eq:.2e}   "
              f"(full plain differs by {diff:.2e})")
    print("-> the unmodified full Qwen2Model + a band mask IS the sliding-window model.")
