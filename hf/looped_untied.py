"""A looped / recurrent-depth transformer  is a special case of  an untied stack.

A looped model applies one block of b layers K times with TIED weights (forward
depth K*b, but only b layers of parameters). An untied LlamaModel of depth K*b
whose layers are periodic with period b (layer j == block layer j % b) computes
EXACTLY that loop. We build such a tied deep model and check its unmodified forward
equals running the b-layer block's own decoder layers K times in a loop.

This is the construction backing the `looped -> untied` edge in the graph: untied
emulates looped for FREE in params (repeat the block's weights), paid in DEPTH
(K*b physical layers). The block/rotary/mask are the model's own; we only tie
weights and orchestrate the loop.

Run:  python -m hf.looped_untied
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

F64 = torch.float64


def _cfg(layers, hidden=64, heads=8, inter=128, vocab=32):
    return LlamaConfig(hidden_size=hidden, num_hidden_layers=layers,
                       num_attention_heads=heads, num_key_value_heads=heads,
                       intermediate_size=inter, vocab_size=vocab,
                       max_position_embeddings=64, rms_norm_eps=1e-6,
                       attn_implementation="eager")


def build(b=2, K=4, seq=12, seed=0):
    torch.manual_seed(seed)
    deep = LlamaModel(_cfg(b * K)).double().eval()
    # tie weights periodically: layer j uses block layer j % b -> a b-layer loop x K
    for j in range(b, b * K):
        deep.layers[j].load_state_dict(deep.layers[j % b].state_dict())

    x = torch.randn(1, seq, 64, dtype=F64)
    with torch.no_grad():
        untied = deep(inputs_embeds=x).last_hidden_state          # unmodified forward, K*b layers

        # reproduce the LOOP with the block's own layers (deep.layers[:b]) applied K times
        pos = torch.arange(seq).unsqueeze(0)
        cos, sin = deep.rotary_emb(x, pos)
        neg = torch.finfo(F64).min
        mask = torch.triu(torch.full((seq, seq), neg, dtype=F64), 1).view(1, 1, seq, seq)
        h = x
        for _ in range(K):
            for i in range(b):
                out = deep.layers[i](h, position_embeddings=(cos, sin), attention_mask=mask)
                h = out[0] if isinstance(out, tuple) else out    # 4.57 returns a bare tensor
        looped = deep.norm(h)                                     # same final RMSNorm
    return (untied - looped).abs().max().item()


if __name__ == "__main__":
    for b, K in [(1, 6), (2, 4), (3, 3)]:
        print(f"block b={b} looped K={K}  (eff depth {b*K})  vs tied untied stack:  "
              f"max|Δ| = {build(b, K):.2e}")
    print("-> a tied-weight untied LlamaModel IS the b-layer block looped K times.")
