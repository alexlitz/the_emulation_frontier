"""FULL-MODEL emulation: a whole `LlamaForCausalLM` reproduces a DIFFERENT, more
special `LlamaForCausalLM` end-to-end -- token embeddings through every decoder
layer to the LM-head logits -- by COMPOSING reductions in the weights alone, with
no code edits to the model.

The target is a "compressed" model: GQA (few KV heads) + a NARROW FFN. The host is
the general superset: MHA (full KV heads) + a WIDE FFN. We build the host's weights
from the target's by

  1. tiling the shared K/V heads to fill every query head   (GQA subset MHA, KV), and
  2. zero-padding the narrow FFN into the wide one so the extra hidden units are
     inert -- their up-projection is 0, so SiLU(gate)*up = 0 and the down-projection
     adds exactly nothing                                    (narrow-FFN subset wide-FFN, capacity).

Both reductions are exact and they compose across every layer, so the host's whole
forward equals the target's to floating-point noise. This is the in-suite, minimal
version of the `elephantine` demo: a real, unmodified `transformers` model doing
another model's entire job by weight-setting alone.

Run:  python -m hf.full_emulation
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hf.gqa_mha import expand_kv

HIDDEN, VOCAB = 64, 48


def _cfg(n_heads, n_kv, inter, layers=3):
    return LlamaConfig(hidden_size=HIDDEN, num_hidden_layers=layers,
                       num_attention_heads=n_heads, num_key_value_heads=n_kv,
                       intermediate_size=inter, vocab_size=VOCAB,
                       max_position_embeddings=64, rms_norm_eps=1e-6,
                       tie_word_embeddings=True, attn_implementation="eager")


def host_state_from_target(target, n_heads, n_kv, head_dim, pad):
    """Remap a (GQA, narrow-FFN) state_dict onto a (MHA, wide-FFN) host: tile K/V,
    zero-pad the FFN so the extra hidden units contribute nothing."""
    sd = {}
    for k, v in target.state_dict().items():
        if k.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
            sd[k] = expand_kv(v, n_kv, n_heads, head_dim)
        elif k.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
            sd[k] = torch.cat([v, torch.zeros(pad, v.shape[1], dtype=v.dtype)], 0)
        elif k.endswith("mlp.down_proj.weight"):
            sd[k] = torch.cat([v, torch.zeros(v.shape[0], pad, dtype=v.dtype)], 1)
        else:
            sd[k] = v.clone()
    return sd


def build(n_heads=8, n_kv=2, inter_t=96, inter_h=192, layers=3, seq=16, seed=0,
          dtype=torch.float64):
    """Return max|Δ logits| between the special target and the host built to emulate it."""
    torch.manual_seed(seed)
    head_dim = HIDDEN // n_heads
    tgt = LlamaForCausalLM(_cfg(n_heads, n_kv, inter_t, layers)).to(dtype).eval()
    host = LlamaForCausalLM(_cfg(n_heads, n_heads, inter_h, layers)).to(dtype).eval()
    host.load_state_dict(
        host_state_from_target(tgt, n_heads, n_kv, head_dim, inter_h - inter_t), strict=True)
    ids = torch.randint(0, VOCAB, (1, seq))
    with torch.no_grad():
        lt = tgt(input_ids=ids).logits
        lh = host(input_ids=ids).logits
    return (lt - lh).abs().max().item()


if __name__ == "__main__":
    print("full LlamaForCausalLM emulation -- (GQA + narrow FFN) target reproduced by")
    print("an (MHA + wide FFN) host, composed across all layers, checked to the logits:")
    for dt, nm in [(torch.float64, "fp64"), (torch.float32, "fp32")]:
        for nk in (1, 2, 4):
            e = build(n_kv=nk, dtype=dt)
            print(f"  {nm}  n_kv={nk} (MHA<-{'MQA' if nk==1 else 'GQA'}): max|Δ logits| = {e:.2e}")
