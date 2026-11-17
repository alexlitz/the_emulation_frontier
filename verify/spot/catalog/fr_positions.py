"""fr-positions: the positions reductions of branch fr-positions (graph/model_graph.py fp_*), built with
hf/fr_positions.py on small random stock models: a fixed sinusoid table on a learned-abs host (table copy) and through
the start-token rule (hf/bos_constructions.learned_abs_host on the sinusoid's rows), and a RoPE + NoPE host playing
an all-RoPE target on its RoPE layers."""
from __future__ import annotations

import torch

from verify.spot.catalog import compare, entry, hf_fn, prompts
from verify.spot.catalog._ref import F64

V = 97


def _xglm(seed=0, layers=2, d=32, heads=4, ffn=64, T=64):
    """A random XGLM (the sinusoid table). fp32: transformers 4.57's XGLM 4-D mask overflows in fp64."""
    from transformers import XGLMConfig, XGLMForCausalLM
    torch.manual_seed(seed)
    cfg = XGLMConfig(vocab_size=V, d_model=d, num_layers=layers, attention_heads=heads, ffn_dim=ffn,
                     max_position_embeddings=T, activation_function="gelu", dropout=0.0, attention_dropout=0.0,
                     activation_dropout=0.0, layerdrop=0.0, scale_embedding=True, pad_token_id=1, bos_token_id=0,
                     eos_token_id=2)
    cfg._attn_implementation = "eager"
    m = XGLMForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if "norm" not in n else 0.1) + (1.0 if n.endswith("norm.weight") else 0.0))
    return m


@entry("slot:pos:sinusoidal->learned_abs",
       instance="xglm (fixed sinusoid table) -> biogpt (learned table, offset 2): the sinusoid's rows copied into the "
                "learned table (hf/fr_positions.sinusoid_host; Lean sinusoid_sub_learned_abs)")
def fr_sinusoid_learned():
    from hf.fr_positions import sinusoid_host
    tgt = _xglm()
    host = sinusoid_host(tgt)
    return compare(hf_fn(tgt), hf_fn(host), V, tol=1e-6, note="fp32 (same weights, same ops)")


@entry("slot:pos:sinusoidal->rope", "slot:pos:sinusoidal->nope", "slot:pos:sinusoidal->partial_rope",
       instance="xglm (fixed sinusoid table) -> llama at inv_freq 0, +2 layers, +5 channels: the start-token rule "
                "with the sinusoid's rows as the table (hf/bos_constructions.learned_abs_host on hf/fr_positions."
                "xglm_spec); a RoPE host at inv_freq 0 is the NoPE host, a partial-RoPE host the same function")
def fr_sinusoid_bos():
    from hf.bos_constructions import learned_abs_host
    from hf.fr_positions import xglm_spec
    tgt = _xglm(T=24)
    host = learned_abs_host(xglm_spec(tgt))
    ids = prompts(V, n=6, T=10, lo=3)                     # (no pad id 1: XGLM skips pads in its positions)
    for x in ids:
        x[:, 0] = tgt.config.bos_token_id
    tfn = lambda x: tgt(input_ids=x).logits.to(F64)
    hfn = lambda x: host(input_ids=x).logits
    return compare(tfn, hfn, V, ids=ids, tol=1e-4, new_tokens=8, note="inputs start with the BOS; fp32 target")


@entry("port:fp_nope_layer_host", "feature:nope_layers:host", "feature:q_temp_nope:host",
       instance="llama (RoPE, 3 layers) -> smollm3 (4 layers, NoPE on layer 3): the NoPE layer zeroed, the target on "
                "the RoPE layers (hf/fr_positions.nope_host; Lean nope_mix_host_skip). Beam's q_temp_nope runs only on "
                "the zeroed NoPE layers (zeroed_seqBlock_id)")
def fr_nope_host():
    from transformers import LlamaConfig, LlamaForCausalLM
    from hf.fr_positions import nope_host
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128, rope_theta=1e4, tie_word_embeddings=False,
                      bos_token_id=0, eos_token_id=1, pad_token_id=None)
    cfg._attn_implementation = "eager"
    tgt = LlamaForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():
        for p in tgt.parameters():
            p.add_(0.2 * torch.randn_like(p))
    host = nope_host(tgt)
    return compare(hf_fn(tgt), hf_fn(host), V, note=host.desc)
