"""GQA / MQA  is a special case of  MHA  -- proven on the REAL LlamaModel.

MHA is the general operator; GQA/MQA restrict it by TYING key/value projections
across groups of query heads. So an unmodified MHA `LlamaModel`, with its K/V
projection weights set to REPLICATE the shared heads, reproduces a GQA/MQA model
exactly. We build a GQA model, copy its state_dict into an MHA-configured model
(K/V rows repeat_interleave'd to fill every head), and check the full forward.

This mirrors HF's own `repeat_kv`: expanded kv-head index = kv*n_rep + r, so query
head q pairs with source kv head q // n_rep -> repeat_interleave along the kv axis.

Run:  python -m hf.gqa_mha
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

F64 = torch.float64


def _cfg(n_heads, n_kv, hidden=64, layers=2, inter=128, vocab=32):
    return LlamaConfig(hidden_size=hidden, num_hidden_layers=layers,
                       num_attention_heads=n_heads, num_key_value_heads=n_kv,
                       intermediate_size=inter, vocab_size=vocab,
                       max_position_embeddings=64, rms_norm_eps=1e-6,
                       attn_implementation="eager")


def expand_kv(w, n_kv, n_heads, head_dim):
    """(n_kv*head_dim, hidden) -> (n_heads*head_dim, hidden): replicate each kv head
    n_heads//n_kv times, in HF `repeat_kv` order (query head q <- kv head q//n_rep)."""
    n_rep = n_heads // n_kv
    return (w.view(n_kv, head_dim, -1).repeat_interleave(n_rep, dim=0)
            .reshape(n_heads * head_dim, -1).contiguous())


def mha_state_from_gqa(gqa: LlamaModel, n_heads, n_kv, head_dim):
    """Remap a GQA state_dict onto an MHA model: copy everything, tile K/V."""
    sd = {}
    for k, v in gqa.state_dict().items():
        if k.endswith("self_attn.k_proj.weight") or k.endswith("self_attn.v_proj.weight"):
            sd[k] = expand_kv(v, n_kv, n_heads, head_dim)
        else:
            sd[k] = v.clone()
    return sd


def build(n_heads=8, n_kv=2, seq=12, seed=0):
    torch.manual_seed(seed)
    head_dim = 64 // n_heads
    gqa = LlamaModel(_cfg(n_heads, n_kv)).double().eval()
    mha = LlamaModel(_cfg(n_heads, n_heads)).double().eval()      # unmodified, MHA config
    missing, unexpected = mha.load_state_dict(
        mha_state_from_gqa(gqa, n_heads, n_kv, head_dim), strict=True)
    x = torch.randn(1, seq, 64, dtype=F64)
    with torch.no_grad():
        og = gqa(inputs_embeds=x).last_hidden_state
        om = mha(inputs_embeds=x).last_hidden_state
    return (og - om).abs().max().item()


if __name__ == "__main__":
    for n_kv in (1, 2, 4):     # MQA (1), GQA (2,4)
        err = build(n_heads=8, n_kv=n_kv)
        kind = "MQA" if n_kv == 1 else "GQA"
        print(f"{kind:3} (kv={n_kv}) -> MHA via tiled K/V:  max|Δ| = {err:.2e}")
    print("-> the unmodified MHA LlamaModel, with replicated K/V weights, IS the GQA model.")
