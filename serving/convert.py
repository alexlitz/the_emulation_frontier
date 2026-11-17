"""Exact conversion of Llama-equivalent architectures into a plain Llama (or Mistral) checkpoint, so
a provider that only serves Llama-family classes can serve them. The output then feeds
serving.embed into any bigger host.

  granite  Llama + four scalars: embedding_multiplier (into the embedding rows), attention_multiplier
           (replaces 1/sqrt(hd): into q as attention_multiplier * sqrt(hd)), residual_multiplier
           (into o_proj and down_proj), logits_scaling (into lm_head). The scaled embedding and head
           differ, so the result is untied.
  phi3     fused qkv_proj / gate_up_proj are split; a sliding window makes it a Mistral. Partial
           RoPE (partial_rotary_factor < 1, e.g. Phi-4-mini) has no Llama equivalent and is rejected.

  to_llama(state_dict, config) -> (state_dict, config dict with model_type llama | mistral)
"""
from __future__ import annotations

import math

import torch

CONVERTIBLE = ("granite", "phi3", "llama", "mistral")


def _g(cfg, k, default=None):
    v = cfg.get(k, default) if isinstance(cfg, dict) else getattr(cfg, k, default)
    return default if v is None else v


def why_not(cfg):
    """None if to_llama can convert this config exactly, else the reason."""
    mt = _g(cfg, "model_type")
    if mt not in CONVERTIBLE:
        return f"{mt} not handled"
    if mt == "phi3" and _g(cfg, "partial_rotary_factor", 1.0) != 1.0:
        return "partial RoPE"
    if _g(cfg, "attention_bias", False) or _g(cfg, "mlp_bias", False):
        return None if mt in ("llama", "mistral") else "biases"
    return None


def _base_cfg(cfg, model_type):
    keys = ("hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "num_hidden_layers",
            "vocab_size", "rms_norm_eps", "rope_theta", "rope_scaling", "max_position_embeddings", "hidden_act",
            "bos_token_id", "eos_token_id", "pad_token_id")
    out = {k: _g(cfg, k) for k in keys}
    out["head_dim"] = _g(cfg, "head_dim", out["hidden_size"] // out["num_attention_heads"])
    out["model_type"] = model_type
    out["architectures"] = ["MistralForCausalLM" if model_type == "mistral" else "LlamaForCausalLM"]
    return out


def to_llama(sd, cfg):
    mt = _g(cfg, "model_type")
    why = why_not(cfg)
    if why:
        raise ValueError(why)
    if mt in ("llama", "mistral"):
        return dict(sd), {**(cfg if isinstance(cfg, dict) else cfg.to_dict())}
    L = _g(cfg, "num_hidden_layers")
    out = {}
    if mt == "granite":
        c = _base_cfg(cfg, "llama")
        c["tie_word_embeddings"] = False
        em, am, rm, ls = (_g(cfg, "embedding_multiplier", 1.0), _g(cfg, "attention_multiplier", 1.0),
                          _g(cfg, "residual_multiplier", 1.0), _g(cfg, "logits_scaling", 1.0))
        qs = am * math.sqrt(c["head_dim"])
        for k, v in sd.items():
            if k == "model.embed_tokens.weight":
                v = v * em
            elif k.endswith("self_attn.q_proj.weight"):
                v = v * qs
            elif k.endswith("self_attn.o_proj.weight") or k.endswith("mlp.down_proj.weight"):
                v = v * rm
            out[k] = v
        head = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
        out["lm_head.weight"] = head / ls
        return out, c
    if mt == "phi3":
        sw = _g(cfg, "sliding_window")
        c = _base_cfg(cfg, "mistral" if sw else "llama")
        c["sliding_window"] = sw
        c["tie_word_embeddings"] = bool(_g(cfg, "tie_word_embeddings", False))
        H, kv, hd, ff = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["intermediate_size"]
        for k, v in sd.items():
            if k.endswith("self_attn.qkv_proj.weight"):
                p = k.removesuffix("qkv_proj.weight")
                out[p + "q_proj.weight"], out[p + "k_proj.weight"], out[p + "v_proj.weight"] = \
                    v.split([H * hd, kv * hd, kv * hd], 0)
            elif k.endswith("mlp.gate_up_proj.weight"):
                p = k.removesuffix("gate_up_proj.weight")
                out[p + "gate_proj.weight"], out[p + "up_proj.weight"] = v.split([ff, ff], 0)
            else:
                out[k] = v
        assert sum(1 for k in out if k.endswith("q_proj.weight")) == L
        return out, c
