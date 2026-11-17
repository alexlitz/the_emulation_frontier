"""Granite and Phi-3 checkpoints convert exactly into plain Llama / Mistral checkpoints (stock
transformers classes reproduce their logits), and then embed into a bigger Llama host.

Run:  python -m pytest tests/test_convert.py -q
"""
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, GraniteConfig, GraniteForCausalLM, Phi3Config,
                          Phi3ForCausalLM)

from serving.convert import to_llama, why_not
from serving.embed import embed_state, plan

EXACT = 1e-6        # transformers' RMSNorm runs in fp32 even in float64 models


def _rand_norms(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.3 * torch.randn_like(p))
    return m


def _logp(m, ids):
    with torch.no_grad():
        return m(ids).logits.log_softmax(-1)


def _from_cfg(c):
    cfg = AutoConfig.for_model(**{**c, "attn_implementation": "eager"})
    return AutoModelForCausalLM.from_config(cfg).double().eval(), cfg


def test_granite_multipliers_fold_into_llama():
    torch.manual_seed(0)
    cfg = GraniteConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, intermediate_size=48,
                        num_hidden_layers=2, vocab_size=50, tie_word_embeddings=True, embedding_multiplier=12.0,
                        attention_multiplier=0.015625, residual_multiplier=0.22, logits_scaling=8.0,
                        attn_implementation="eager")
    g = _rand_norms(GraniteForCausalLM(cfg).double().eval())
    sd, lc = to_llama(g.state_dict(), cfg)
    m, _ = _from_cfg(lc)
    m.load_state_dict(sd, strict=True)
    ids = torch.randint(0, 50, (2, 20))
    assert (_logp(g, ids) - _logp(m, ids)).abs().max().item() < EXACT


def test_phi3_splits_into_mistral_and_rejects_partial_rope():
    torch.manual_seed(1)
    cfg = Phi3Config(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, intermediate_size=48,
                     num_hidden_layers=2, vocab_size=50, sliding_window=8, max_position_embeddings=64,
                     pad_token_id=0, attn_implementation="eager")
    p = _rand_norms(Phi3ForCausalLM(cfg).double().eval())
    sd, lc = to_llama(p.state_dict(), cfg)
    assert lc["model_type"] == "mistral"
    m, _ = _from_cfg(lc)
    m.load_state_dict(sd, strict=False)
    ids = torch.randint(0, 50, (2, 24))                 # longer than the window: the window must carry over
    assert (_logp(p, ids) - _logp(m, ids)).abs().max().item() < EXACT
    assert why_not(Phi3Config(partial_rotary_factor=0.75)) == "partial RoPE"


def test_granite_then_embed_into_a_bigger_llama():
    torch.manual_seed(2)
    cfg = GraniteConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, intermediate_size=48,
                        num_hidden_layers=2, vocab_size=50, tie_word_embeddings=True, embedding_multiplier=12.0,
                        attention_multiplier=0.015625, residual_multiplier=0.22, logits_scaling=8.0,
                        attn_implementation="eager")
    g = _rand_norms(GraniteForCausalLM(cfg).double().eval())
    sd, lc = to_llama(g.state_dict(), cfg)
    host = dict(model_type="llama", hidden_size=48, num_attention_heads=8, num_key_value_heads=4, head_dim=16,
                intermediate_size=80, num_hidden_layers=3, vocab_size=64, tie_word_embeddings=False,
                max_position_embeddings=lc["max_position_embeddings"])
    pl = plan(lc, host)
    assert pl["ok"], pl["why"]
    host.update({k: v for k, v in pl["config"].items() if v is not None})
    m, _ = _from_cfg(host)
    m.load_state_dict(embed_state(sd, lc, host), strict=False)
    ids = torch.randint(0, 50, (2, 20))
    lp = _logp(m, ids)
    assert (_logp(g, ids) - lp[..., :50]).abs().max().item() < EXACT
    assert lp[..., 50:].exp().sum(-1).max().item() < 1e-12
