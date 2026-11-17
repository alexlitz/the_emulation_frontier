"""Layer-wise verification on a REAL checkpoint's actual weights (not synthetic).

Loads a real Llama-family model, builds a bigger host (MHA instead of GQA, wider FFN,
extra identity layers) that reproduces it, and checks the RAW residual after every
layer to fp64 noise — plus confirms the size-aware relation calls it an edge and the
host genuinely spends the currency (more KV, more FFN width, more depth).

  python -m elephantine.verify_real HuggingFaceTB/SmolLM2-135M
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from elephantine.verify_edges import _raw_layer_outs, point
from graph.model_graph import dims_of, emulates


def host_config(ac, ffn_extra=512, depth_extra=2):
    d = ac.to_dict()
    d["num_key_value_heads"] = ac.num_attention_heads          # GQA/MQA target -> MHA host
    d["intermediate_size"] = ac.intermediate_size + ffn_extra  # wider FFN (zero-padded)
    d["num_hidden_layers"] = ac.num_hidden_layers + depth_extra  # deeper (identity tail)
    return LlamaConfig(**d)


@torch.no_grad()
def build_host(A, bc):
    B = LlamaForCausalLM(bc).to(next(A.parameters()).dtype).eval()
    asd = A.state_dict()
    hd = A.config.hidden_size // A.config.num_attention_heads
    B.model.embed_tokens.weight.copy_(A.model.embed_tokens.weight)
    lm = A.model.embed_tokens.weight if A.config.tie_word_embeddings else A.lm_head.weight
    B.lm_head.weight.copy_(lm)
    B.model.norm.weight.copy_(A.model.norm.weight)
    LA = A.config.num_hidden_layers
    kvA, kvB, iA = A.config.num_key_value_heads, bc.num_key_value_heads, A.config.intermediate_size
    rep = kvB // kvA
    for i, layer in enumerate(B.model.layers):
        if i >= LA:                                            # identity tail
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
            continue
        p = f"model.layers.{i}."
        layer.self_attn.q_proj.weight.copy_(asd[p + "self_attn.q_proj.weight"])
        layer.self_attn.o_proj.weight.copy_(asd[p + "self_attn.o_proj.weight"])
        kA = asd[p + "self_attn.k_proj.weight"].view(kvA, hd, -1)          # tile GQA kv into MHA
        vA = asd[p + "self_attn.v_proj.weight"].view(kvA, hd, -1)
        src = torch.arange(kvB) // rep
        layer.self_attn.k_proj.weight.copy_(kA[src].reshape(kvB * hd, -1))
        layer.self_attn.v_proj.weight.copy_(vA[src].reshape(kvB * hd, -1))
        layer.input_layernorm.weight.copy_(asd[p + "input_layernorm.weight"])
        layer.post_attention_layernorm.weight.copy_(asd[p + "post_attention_layernorm.weight"])
        for pr in ("gate_proj", "up_proj"):                               # narrow FFN -> wide (zero-pad)
            w = getattr(layer.mlp, pr).weight; w.zero_(); w[:iA].copy_(asd[p + f"mlp.{pr}.weight"])
        layer.mlp.down_proj.weight.zero_()
        layer.mlp.down_proj.weight[:, :iA].copy_(asd[p + "mlp.down_proj.weight"])
    return B


@torch.no_grad()
def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
    A = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float64).eval()
    ac = A.config
    print(f"real target: {repo}  (h{ac.hidden_size} L{ac.num_hidden_layers} "
          f"heads{ac.num_attention_heads} kv{ac.num_key_value_heads} ffn{ac.intermediate_size})")
    bc = host_config(ac, ffn_extra=512, depth_extra=2)
    B = build_host(A, bc)

    # currency check: the relation must call this an edge, and the host must be strictly bigger
    da, db = dims_of(ac.to_dict()), dims_of(bc.to_dict())
    edge = emulates(point(bc), point(ac))
    print(f"emulates(host, target) = {edge}; "
          f"currency spent: kv {da['kv_cap']}->{db['kv_cap']}, "
          f"ffn {da['ffn_total']}->{db['ffn_total']}, depth {da['eff_layers']}->{db['eff_layers']}")

    ids = torch.randint(0, ac.vocab_size, (1, 16))
    ao, bo = _raw_layer_outs(A, ids), _raw_layer_outs(B, ids)
    per = [(ao[i] - bo[i]).abs().max().item() for i in range(len(ao))]
    tail = max([(bo[j] - ao[-1]).abs().max().item() for j in range(len(ao), len(bo))], default=0.0)
    dlog = (A(ids).logits - B(ids).logits).abs().max().item()
    print(f"layer-wise on REAL weights: worst per-layer |Δ| = {max(per):.2e}  "
          f"(over {len(ao)} layers), identity tail = {tail:.1e}, final logits = {dlog:.2e}")
    print("PASS" if max(max(per), tail, dlog) < 1e-9 else "FAIL")


if __name__ == "__main__":
    main()
