"""Hidden-width emulation: a wider-hidden host reproduces a narrower model EXACTLY.

This is the one capacity axis `verify_edges.py` did not build, because it "needs norm
compensation." For an RMSNorm decoder it IS exact. Embed A's residual stream in the
first d_A coordinates of B's d_B-wide stream and keep the extra d_B - d_A coordinates
at 0 everywhere. Then RMSNorm over d_B dims with the tail zeroed satisfies

    rms_B(x) = sqrt( (sum_{i<d_A} x_i^2) / d_B + eps_B )
             = sqrt( d_A/d_B ) * rms_A(x)      iff   eps_B = eps_A * d_A/d_B,

so the norm is reproduced by

    eps_B   = eps_A * d_A/d_B
    gamma_B = [ gamma_A * sqrt(d_A/d_B) ,  0 ... 0 ]

and every linear zero-pads: readers pad input columns, writers pad output rows,
embed / lm_head pad the hidden columns. The attention head structure (n_heads,
head_dim, kv) is UNCHANGED -- only the residual width grows. Result: B(x) == A(x),
token-for-token.

The model's own `LlamaRMSNorm` upcasts to fp32 internally, so on the native forward the
match is fp32-level (~1e-6) with an identical greedy token stream; patching the norm to
fp64 shows the construction itself is exact (~1e-13).

  python -m elephantine.hidden_width          # verify on Llama-3.2-1B
"""
from __future__ import annotations

import math

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import harness as H

F64 = torch.float64


def widen_config(ca, dB):
    """Config for a d_B-wide host: same heads/head_dim/kv/FFN, compensated norm eps."""
    dA = ca.hidden_size
    return LlamaConfig(**{**ca.to_dict(), "hidden_size": dB, "head_dim": ca.head_dim,
                          "num_attention_heads": ca.num_attention_heads,
                          "num_key_value_heads": ca.num_key_value_heads,
                          "intermediate_size": ca.intermediate_size,
                          "rms_norm_eps": ca.rms_norm_eps * dA / dB})


@torch.no_grad()
def widen_state_dict(A, dB):
    """Remap A's weights into a d_B-wide host (zero-pad + RMSNorm gamma rescale)."""
    dA = A.config.hidden_size
    r = math.sqrt(dA / dB)
    sd = {}
    for k, v in A.state_dict().items():
        v = v.double()
        if k.endswith("embed_tokens.weight") or k == "lm_head.weight":     # (vocab, dA) -> pad hidden cols
            w = torch.zeros(v.shape[0], dB, dtype=F64); w[:, :dA] = v
        elif k.endswith("norm.weight"):                                    # RMSNorm gamma: rescale + zero-pad
            w = torch.zeros(dB, dtype=F64); w[:dA] = v * r
        elif k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight",
                         "gate_proj.weight", "up_proj.weight")):
            w = torch.zeros(v.shape[0], dB, dtype=F64); w[:, :dA] = v      # readers: pad input columns
        elif k.endswith(("o_proj.weight", "down_proj.weight")):
            w = torch.zeros(dB, v.shape[1], dtype=F64); w[:dA, :] = v      # writers: pad output rows
        else:
            w = v.clone()
        sd[k] = w
    return sd


@torch.no_grad()
def widen(A, dB):
    """Build a d_B-wide LlamaForCausalLM host whose forward reproduces A."""
    B = LlamaForCausalLM(widen_config(A.config, dB)).to(F64).eval()
    B.load_state_dict(widen_state_dict(A, dB), strict=True)
    return B


@torch.no_grad()
def build_caps(A, kv=None, inter=None, layers=None):
    """Build a host with A's REAL dims but bigger KV-sharing / FFN width / depth, whose
    forward reproduces A. Generalizes elephantine.verify_edges.build to a real config
    (A's head_dim / hidden / vocab): expand K/V heads, zero-pad the FFN, extra depth =
    identity. Assumes a bias-free Llama-family RMSNorm model (Llama-3.x)."""
    ca = A.config
    hd = ca.head_dim
    kvA, HH, LA, iA = ca.num_key_value_heads, ca.num_attention_heads, ca.num_hidden_layers, ca.intermediate_size
    kvB = kv or kvA
    iB = inter or iA
    LB = layers or LA
    B = LlamaForCausalLM(LlamaConfig(**{**ca.to_dict(), "num_key_value_heads": kvB,
                                        "intermediate_size": iB, "num_hidden_layers": LB})).to(F64).eval()
    asd = A.state_dict()
    B.model.embed_tokens.weight.copy_(A.model.embed_tokens.weight.double())
    if "lm_head.weight" in asd:
        B.lm_head.weight.copy_(asd["lm_head.weight"].double())
    B.model.norm.weight.copy_(A.model.norm.weight.double())
    rep = kvB // kvA
    for i, layer in enumerate(B.model.layers):
        if i >= LA:                                             # extra depth -> identity (adds 0)
            layer.self_attn.o_proj.weight.zero_(); layer.mlp.down_proj.weight.zero_()
            continue
        p = f"model.layers.{i}."
        layer.self_attn.q_proj.weight.copy_(asd[p + "self_attn.q_proj.weight"].double())
        layer.self_attn.o_proj.weight.copy_(asd[p + "self_attn.o_proj.weight"].double())
        kA = asd[p + "self_attn.k_proj.weight"].double().view(kvA, hd, -1)   # replicate each kv head
        vA = asd[p + "self_attn.v_proj.weight"].double().view(kvA, hd, -1)
        src = torch.arange(kvB) // rep
        layer.self_attn.k_proj.weight.copy_(kA[src].reshape(kvB * hd, -1))
        layer.self_attn.v_proj.weight.copy_(vA[src].reshape(kvB * hd, -1))
        layer.input_layernorm.weight.copy_(asd[p + "input_layernorm.weight"].double())
        layer.post_attention_layernorm.weight.copy_(asd[p + "post_attention_layernorm.weight"].double())
        n = min(iA, iB)
        for pr in ("gate_proj", "up_proj"):                    # narrow FFN into wide (zero-pad rows)
            w = getattr(layer.mlp, pr).weight; w.zero_(); w[:n].copy_(asd[p + f"mlp.{pr}.weight"][:n].double())
        layer.mlp.down_proj.weight.zero_()
        layer.mlp.down_proj.weight[:, :n].copy_(asd[p + "mlp.down_proj.weight"][:, :n].double())
    return B


@torch.no_grad()
def convert(A, kv=None, inter=None, layers=None, hidden=None):
    """Convert a real model A into a host bigger on ANY of the four capacity axes at once:
    KV-sharing (kv), FFN width (inter), depth (layers), hidden width (hidden). B reproduces
    A token-for-token."""
    B = build_caps(A, kv=kv, inter=inter, layers=layers)
    return B if not hidden or hidden == B.config.hidden_size else widen(B, hidden)


def _rms_f64(mod):
    """A make_forward that runs LlamaRMSNorm in fp64 (the stock one upcasts to fp32)."""
    w, eps = mod.weight, mod.variance_epsilon

    def fwd(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w
    return fwd


@torch.no_grad()
def verify(model_name="meta-llama/Llama-3.2-1B", add=256, seed=0):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    A, _, _ = H.load(model_name, dtype=F64, device="cpu")
    dB = A.config.hidden_size + add
    B = widen(A, dB)
    ids = torch.randint(0, 2000, (2, 16), generator=torch.Generator().manual_seed(seed))
    native = (A(ids).logits.double() - B(ids).logits.double()).abs().max().item()
    na, nb = H.modules_of_type(A, LlamaRMSNorm), H.modules_of_type(B, LlamaRMSNorm)
    with H.patched_forwards(na, _rms_f64), H.patched_forwards(nb, _rms_f64):   # isolate the construction
        exact = (A(ids).logits.double() - B(ids).logits.double()).abs().max().item()
    gk = dict(do_sample=False, max_new_tokens=20, pad_token_id=0,
              attention_mask=torch.ones_like(ids[:1]))
    same = torch.equal(A.generate(ids[:1], **gk), B.generate(ids[:1], **gk))
    return dict(dA=A.config.hidden_size, dB=dB, native=native, exact=exact, tokens_identical=same)


def main():
    r = verify()
    print(f"hidden-width emulation on Llama-3.2-1B: {r['dA']} -> {r['dB']}")
    print(f"  native forward (model's fp32 RMSNorm):  max|Δ logits| = {r['native']:.2e}")
    print(f"  construction (RMSNorm in fp64):         max|Δ logits| = {r['exact']:.2e}   (exact)")
    print(f"  greedy token stream identical:          {r['tokens_identical']}")
    print("  -> a wider-hidden host reproduces the narrower model token-for-token; the")
    print("     residual on the native forward is only the model's own fp32-internal norm.")


if __name__ == "__main__":
    main()
