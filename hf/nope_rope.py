"""NoPE (no positional encoding) is a special case of RoPE -- on the real LlamaModel.

RoPE rotates q,k by angle position*inv_freq. Set inv_freq = 0 and the rotation is
the identity: the model computes NoPE attention (scores depend on content only, not
position). So NoPE ⊆ RoPE, reached by a buffer parameterization -- no code edits.

We verify by the defining property: NoPE attention ignores positions entirely, so
STRETCHING the position ids (changing the relative distances) leaves a NoPE model's
output unchanged, while a real RoPE model's output changes (RoPE encodes relative
distance). Same unmodified LlamaModel both times. (A global *shift* would not work --
RoPE is relative, so it is shift-invariant; we must change relative distances.)

Run:  python -m hf.nope_rope
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

F64 = torch.float64


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=8, intermediate_size=128, vocab_size=32,
                      max_position_embeddings=128, rms_norm_eps=1e-6,
                      attn_implementation="eager")
    return LlamaModel(cfg).double().eval()


def _run(m, x, pos):
    with torch.no_grad():
        return m(inputs_embeds=x, position_ids=pos.unsqueeze(0)).last_hidden_state


def build(seq=12, stretch=3, seed=0):
    m = _model(seed)
    x = torch.randn(1, seq, 64, dtype=F64)
    p0 = torch.arange(seq)
    p1 = torch.arange(seq) * stretch                                      # different RELATIVE distances
    rope_change = (_run(m, x, p0) - _run(m, x, p1)).abs().max().item()    # RoPE: differs
    m.rotary_emb.inv_freq = torch.zeros_like(m.rotary_emb.inv_freq)       # inv_freq -> 0 = NoPE
    m.rotary_emb.original_inv_freq = m.rotary_emb.inv_freq
    nope_change = (_run(m, x, p0) - _run(m, x, p1)).abs().max().item()    # NoPE: invariant
    return rope_change, nope_change


if __name__ == "__main__":
    rope_change, nope_change = build()
    print("NoPE ⊆ RoPE on the real LlamaModel (inv_freq -> 0):")
    print(f"  RoPE  output change when relative distances stretch: {rope_change:.3e}   (position-dependent)")
    print(f"  NoPE  output change when relative distances stretch: {nope_change:.3e}   (position-INVARIANT)")
    print("  -> zeroing the rotary frequency turns the unmodified RoPE model into NoPE.")
