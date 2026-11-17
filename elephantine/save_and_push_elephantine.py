"""Save elephantine-2026-08-08 locally, write its model card, and push to the Hub.

Re-runs the verification (fp64 + fp32), then saves config.json + model.safetensors,
writes README.md with the exact measured max|Δ|, and pushes to
alexlitz/elephantine-2026-08-08.
"""
from __future__ import annotations

import pathlib

import torch
from huggingface_hub import create_repo, upload_file

from build_elephantine import (VOCAB, SEQ, build_elephantine, build_target,
                               weight_from_target)

REPO = "alexlitz/elephantine-2026-08-08"
OUT = pathlib.Path("elephantine-2026-08-08")


@torch.no_grad()
def measure(dtype):
    target = build_target(dtype)
    eleph, cfg = build_elephantine(dtype)
    weight_from_target(eleph, target, cfg)
    torch.manual_seed(123)
    ids = torch.randint(0, VOCAB, (2, SEQ))
    d = (target(input_ids=ids).logits - eleph(input_ids=ids).logits).abs().max().item()
    return d, eleph, cfg


def main():
    d64, _, _ = measure(torch.float64)
    d32, eleph, cfg = measure(torch.float32)          # the model we save is fp32
    assert d64 <= 1e-9 and d32 <= 1e-4, (d64, d32)
    n_params = sum(p.numel() for p in eleph.parameters())
    print(f"fp64 max|Δ|={d64:.3e}  fp32 max|Δ|={d32:.3e}  params={n_params:,}")

    OUT.mkdir(exist_ok=True)
    eleph.save_pretrained(OUT)
    print(f"saved to {OUT}/")

    readme = MODEL_CARD.format(
        d64=f"{d64:.3e}", d32=f"{d32:.3e}", n_params=f"{n_params:,}",
        schedule=cfg.layers_block_type, experts=cfg.layers_num_experts,
        top_k=cfg.num_experts_per_tok, n_experts=cfg.num_experts,
    )
    (OUT / "README.md").write_text(readme)
    print("wrote README.md")

    create_repo(REPO, exist_ok=True)
    eleph.push_to_hub(REPO)
    upload_file(path_or_fileobj=str(OUT / "README.md"),
                path_in_repo="README.md", repo_id=REPO)
    print(f"pushed -> https://huggingface.co/{REPO}")


MODEL_CARD = """---
library_name: transformers
pipeline_tag: text-generation
tags:
- elephantine
- architecture-emulation
- demonstration
- jamba
- hybrid-ssm
- mixture-of-experts
license: apache-2.0
---

# elephantine-2026-08-08

**A demonstration that a big, general "universal-emulator" architecture, correctly
weighted, *is* a smaller model — token-for-token.**

`elephantine-2026-08-08` is a **hybrid Jamba** causal LM: interleaved
**state-space-model (Mamba/SSM) layers and attention layers**, with a
**Mixture-of-Experts** FFN on every attention block. Its weights are *set* (not
trained) so that its forward pass **reproduces the smallest SiLU/RMSNorm decoder
LM in this dataset exactly**.

> Idea ("SSM layers to contain the FFNs"): a hybrid can be deep with cheap SSM
> mixing, and every block — SSM or attention — carries an FFN, so **depth holds
> FFN capacity**. Parameterize that general architecture into a special case and
> it becomes the target model itself.

## Bit-for-bit result

Running identical `input_ids` through elephantine and the target:

| precision | `max|Δ logits|` |
|-----------|-----------------|
| **fp64**  | **{d64}** (exactly zero) |
| **fp32**  | {d32} |

## Architecture

| property | value |
|----------|-------|
| class | `JambaForCausalLM` |
| hidden size | 64 |
| layers | 4 — schedule `{schedule}` |
| attention | 4 heads, 4 KV heads (MHA), **no positional encoding (NoPE)** |
| MoE | {n_experts} experts / expert-layer, top-k = {top_k} (experts per layer: `{experts}`) |
| FFN width | 128 (SwiGLU, SiLU) |
| vocab | 256 |
| params | ~{n_params} |

## The target it emulates

The smallest **RMSNorm + SiLU** decoder LM: a 2-layer **NoPE `LlamaForCausalLM`**
(hidden 64, 4 heads, FFN 128, vocab 256, `rms_norm_eps=1e-5`) with fixed random
weights. RMSNorm + SiLU (not LayerNorm/GeLU) is what elephantine can reproduce
*exactly*. NoPE because Jamba's attention carries no positional encoding — and
NoPE is a special case of RoPE (zero the rotary frequency), so a NoPE target is
exactly what a NoPE-attention hybrid reproduces.

## How the weights are set

- **Attention layers** copy the target's `q/k/v/o` projections and its two
  RMSNorms verbatim (MHA ← MHA, direct copy).
- **MoE FFNs** collapse to the target's dense SwiGLU MLP: every expert is tied to
  that MLP, and **top-k = num_experts** so all experts are active. The router
  weight is zeroed, so the softmax is exactly uniform (`0.25` each, a power of
  two); the four selected weights sum to **exactly 1.0**, and the tied experts
  accumulate to exactly `f(x)` — bit-exact even though Jamba computes the router
  softmax in fp32.
- **Mamba (SSM) layers** are exact identity: zeroing a mixer's `out_proj` makes
  the SSM block compute `x + 0 = x`; their plain MLP is zeroed too (`down_proj =
  0`), so the whole block is a full passthrough.
- Embeddings, `lm_head`, and the final RMSNorm are copied from the target.

Net: `elephantine.forward == target.forward`.

## Caveat

This model is a **demonstration, not a trained model**. Its logits are meaningful
only as a reproduction of the target; it is the general elephantine architecture
parameterized into a special case. Do not expect language-model quality.

## Reproduction

```python
import torch
from transformers import JambaForCausalLM
m = JambaForCausalLM.from_pretrained("alexlitz/elephantine-2026-08-08").eval()
ids = torch.randint(0, 256, (2, 8))
print(m(input_ids=ids).logits.shape)
```

Built with the reductions validated in
[github.com/alexlitz/the_emulation_frontier](https://github.com/alexlitz/the_emulation_frontier)
(`hf/gqa_mha.py`, `hf/moe_dense.py`, `hf/nope_rope.py`, and the Mamba
identity-passthrough trick).
"""


if __name__ == "__main__":
    main()
