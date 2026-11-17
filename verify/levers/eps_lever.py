"""The eps lever: a host's norm the target LACKS is switched off by config + weights alone.

All norms of a model share one config epsilon E (rms_norm_eps). A norm whose input is scaled by S computes
S x / sqrt(S^2 ms(x) + E) = x / sqrt(ms(x) + E / S^2): its EFFECTIVE epsilon is E / S^2, and S is set by the
weights that write the norm's input. So one shared E gives every norm its own effective eps:
  * the residual is scaled by S_res = sqrt(E / eps_t) (embedding, o_proj, down_proj x S_res): every pre-norm
    and the final norm compute exactly the target's norm (eps_eff = eps_t);
  * a norm the target lacks reads an input scaled DOWN (q_proj / k_proj / o_proj x 2^-a) so that ms << E:
    fl(ms + E) = E in floating point, the norm is the constant 1 / sqrt(E) -- a LINEAR map -- and its gain
    (or o_proj, for a weight-free norm) restores the scale.

Checks, fp32 / fp64, random weights (transformers installed version):
  1. Qwen3 host (q / k RMS norms, per head, before RoPE) == Llama target  (qk-norm switched off)
  2. Gemma-2 host (sandwich: pre + post norms, softcaps None, query_pre_attn_scalar = head_dim, window >=
     context) == Gemma target  (post-norms switched off)
  3. a weight-free RMS norm (Gemma-3n / Gemma-4 v_norm) in the linear regime == identity / sqrt(E)

Run:  python -m verify.levers.eps_lever
"""
from __future__ import annotations

import json
import math
import os

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def _copy(dst, src):
    with torch.no_grad():
        dst.copy_(src)


def qwen3_from_llama(dtype=torch.float32, eps_t=1e-6, S_res=2.0 ** 20, a=8, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    kw = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, max_position_embeddings=64, rope_theta=10000.0,
              tie_word_embeddings=False, attention_bias=False)
    tgt = LlamaForCausalLM(LlamaConfig(rms_norm_eps=eps_t, **kw)).to(dtype).eval()
    for n, p in tgt.named_parameters():                     # non-trivial norm gains
        if "norm" in n:
            with torch.no_grad():
                p.uniform_(0.5, 1.5)
    E = eps_t * S_res ** 2
    host = Qwen3ForCausalLM(Qwen3Config(rms_norm_eps=E, use_sliding_window=False, **kw)).to(dtype).eval()
    sd = tgt.state_dict()
    g = 2.0 ** a * math.sqrt(E)                             # the q / k norm gain that undoes 2^-a / sqrt(E)
    for n, p in host.named_parameters():
        if n.endswith("q_norm.weight") or n.endswith("k_norm.weight"):
            _copy(p, torch.full_like(p, g))
            continue
        v = sd[n].clone()
        if n.endswith(("q_proj.weight", "k_proj.weight")):
            v = v * 2.0 ** -a
        if n.endswith(("o_proj.weight", "down_proj.weight", "embed_tokens.weight")):
            v = v * S_res
        _copy(p, v)
    return tgt, host, E


def gemma2_from_gemma(dtype=torch.float32, eps_t=1e-6, S_res=2.0 ** 20, a=8, seed=0):
    from transformers import Gemma2Config, Gemma2ForCausalLM, GemmaConfig, GemmaForCausalLM
    torch.manual_seed(seed)
    kw = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, max_position_embeddings=64, rope_theta=10000.0,
              hidden_activation="gelu_pytorch_tanh", tie_word_embeddings=True)
    tgt = GemmaForCausalLM(GemmaConfig(rms_norm_eps=eps_t, **kw)).to(dtype).eval()
    for n, p in tgt.named_parameters():
        if "norm" in n:
            with torch.no_grad():
                p.uniform_(-0.5, 0.5)                       # Gemma gain is (1 + w)
    E = eps_t * S_res ** 2
    host = Gemma2ForCausalLM(Gemma2Config(rms_norm_eps=E, attn_logit_softcapping=None, final_logit_softcapping=None,
                                          query_pre_attn_scalar=16, sliding_window=4096, **kw)).to(dtype).eval()
    sd = tgt.state_dict()
    g = S_res * 2.0 ** a * math.sqrt(E)                     # post-norm gain: o x 2^-a / sqrt(E) -> S_res o
    for n, p in host.named_parameters():
        if n.endswith(("post_attention_layernorm.weight", "post_feedforward_layernorm.weight")):
            _copy(p, torch.full_like(p, g - 1.0))           # Gemma2RMSNorm multiplies by (1 + w)
            continue
        if n.endswith("pre_feedforward_layernorm.weight"):  # host pre-MLP norm = target's post_attention_layernorm
            _copy(p, sd[n.replace("pre_feedforward_layernorm", "post_attention_layernorm")])
            continue
        v = sd[n].clone()
        if n.endswith(("o_proj.weight", "down_proj.weight")):
            v = v * 2.0 ** -a                               # small: the post-norm runs linear
        if n.endswith("embed_tokens.weight"):
            v = v * S_res                                   # (tied: lm_head reads the final-normed residual,
        _copy(p, v)                                         #  so scale the logits back below)
    return tgt, host, E


def compare(tgt, host, dtype, logit_scale=1.0, T=24, seed=1):
    torch.manual_seed(seed)
    ids = torch.randint(0, tgt.config.vocab_size, (2, T))
    with torch.no_grad():
        a = tgt(ids).logits.double()
        b = host(ids).logits.double() / logit_scale
    return float((a - b).abs().max()), float(a.abs().max())


def weightless_norm(dtype=torch.float32, E=2.0 ** 40):
    x = torch.randn(4, 8, 16, dtype=dtype) * 3
    var = x.float().pow(2).mean(-1, keepdim=True)
    y = (x.float() * torch.rsqrt(var + E)).to(dtype)        # Gemma3n / Gemma4 RMSNorm(with_scale=False)
    return float((y * math.sqrt(E) - x).abs().max())


def main():
    rows = []
    for dtype in (torch.float32, torch.float64):
        a = 8 if dtype == torch.float32 else 20     # fp64 needs ms << E 2^-53: a larger input scale-down
        t, h, E = qwen3_from_llama(dtype, a=a)
        d, m = compare(t, h, dtype)
        # reference: the same target computed in the other precision (rounding scale)
        rows.append(dict(check="Qwen3 host (qk-norm off) == Llama target", dtype=str(dtype), E=E, max_abs_diff=d,
                         max_logit=m))
        t, h, E = gemma2_from_gemma(dtype, a=a)
        S = 2.0 ** 20
        d, m = compare(t, h, dtype, logit_scale=S)          # tied head: host logits = S x target logits
        rows.append(dict(check="Gemma-2 host (post-norms off, softcap None) == Gemma target", dtype=str(dtype), E=E,
                         max_abs_diff=d, max_logit=m))
        rows.append(dict(check="weight-free RMS norm, linear regime == x / sqrt(E)", dtype=str(dtype), E=2.0 ** 40,
                         max_abs_diff=weightless_norm(dtype), max_logit=None))
    # the rounding scale: the Llama target itself, fp32 vs fp64
    import copy
    t32, _, _ = qwen3_from_llama(torch.float32)
    t64 = copy.deepcopy(t32).double()
    torch.manual_seed(1)
    ids = torch.randint(0, 97, (2, 24))
    with torch.no_grad():
        ref = float((t32(ids).logits.double() - t64(ids).logits).abs().max())
    rows.append(dict(check="reference: Llama target fp32 vs fp64 (rounding scale)", dtype="-", E=None,
                     max_abs_diff=ref, max_logit=None))
    for r in rows:
        ml = f"  (max |logit| {r['max_logit']:.2f})" if r["max_logit"] else ""
        print(f"{r['check']:62s} {r['dtype']:14s} max|d| = {r['max_abs_diff']:.3e}{ml}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "eps_lever.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
