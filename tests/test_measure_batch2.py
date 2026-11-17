"""Measurement batch 2: the eps lever on stock modules, GeLU erf <-> tanh, and the pinned attention."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch

from hf import eps_lever_agreement as el

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _target(fam, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    kw = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
              head_dim=16, max_position_embeddings=64, rope_theta=10000.0, rms_norm_eps=1e-6,
              attn_implementation="eager", tie_word_embeddings=True)
    if fam == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        m = LlamaForCausalLM(LlamaConfig(num_key_value_heads=2, **kw))
    else:                                            # Gemma-1, MQA (as gemma-2b)
        from transformers import GemmaConfig, GemmaForCausalLM
        m = GemmaForCausalLM(GemmaConfig(num_key_value_heads=1, hidden_act="gelu", **kw))
    m = m.to(dtype).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5) if fam == "llama" else p.uniform_(-0.5, 0.5)
            elif "embed" in n:
                p.normal_(0, 1.0)
            else:
                p.normal_(0, 0.3)                     # big enough that every norm is far from linear at a = 0
    return m


def _host(kind, tgt, a, s_log2=0, dtype=torch.float64):
    t = el.tdims(tgt.config)
    sd = {n: p.detach() for n, p in tgt.state_dict().items()}
    host = el._causal_lm(kind, el.host_config(kind, t, t["eps"] * 4.0 ** s_log2)).to(dtype).eval()
    el.write_host(host, el.host_weights(kind, el.tfamily(tgt.config), t, sd, a, s_log2), dtype)
    return host


def _diff(tgt, host, T=20, seed=1):
    torch.manual_seed(seed)
    ids = torch.randint(0, 97, (2, T))
    with torch.no_grad():
        a = torch.log_softmax(tgt(ids).logits.double(), -1)
        b = torch.log_softmax(host(ids).logits.double(), -1)
    return (a - b).abs().max().item()


@pytest.mark.parametrize("kind,fam", [("qwen3", "llama"), ("gemma2", "gemma"), ("gemma3", "gemma"),
                                      ("mla", "gemma")])
def test_eps_lever_switches_off_host_norms(kind, fam):
    """the host's extra norms run linear at large a: the host IS the target (to fp32 rounding -- HF RMS
    norms upcast to fp32 only); at a = 0 the same weights give a different model."""
    tgt = _target(fam)
    off = _diff(tgt, _host(kind, tgt, a=0))
    on = _diff(tgt, _host(kind, tgt, a=30))
    assert on < 1e-4, (kind, on)          # the rounding floor: HF RMS norms upcast to fp32 only
    assert off > 1.0, (kind, off)


def test_eps_lever_residual_scaled_variant():
    """frontier-levers' split (residual x 2^20, host eps = e_t 2^40) is the same lever on a plain-gain host;
    on a (1 + w) host with a tied head it is not: the final norm needs gain 1 / S, and Gemma2RMSNorm forms
    1 + w in fp32, so w = 2^-20 - 1 loses the gain. S = 1 (the residual untouched) is the split to use."""
    tgt = _target("llama")
    assert _diff(tgt, _host("qwen3", tgt, a=10, s_log2=20)) < 1e-4
    tgt = _target("gemma")
    assert _diff(tgt, _host("gemma3", tgt, a=10, s_log2=20)) > 1e-1
    assert _diff(tgt, _host("gemma3", tgt, a=30, s_log2=0)) < 1e-4


def test_interleave_perm():
    p = el._interleave_perm(8)
    assert p.tolist() == [0, 4, 1, 5, 2, 6, 3, 7]


def test_a_for():
    a = el.a_for(1e6, 1e-6, -30)
    assert 2.0 ** (-2 * a) * 1e12 <= 2.0 ** -30 < 2.0 ** (-2 * (a - 1)) * 1e12


# --- recorded measurements -------------------------------------------------------------------------
def _rows(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        pytest.skip(f"{path} not recorded")
    return [json.loads(l) for l in open(p)]


def test_eps_lever_rule_matches_data():
    from graph import eps_lever as rule
    rows = _rows("hf/data/eps_lever_agreement.jsonl")
    for pair, rec in rule.MEASURED.items():
        prim = [r for r in rows if r["pair"] == pair and r["variant"] == f"host log2rho<={el.PRIMARY}"]
        assert prim, pair
        for dt in ("float32", "float64"):
            got = [r for r in prim if r["dtype"] == dt]
            if dt in rec:
                assert got and all(r["payable"] for r in got), (pair, dt)
                assert max(r["max_dlogp"] for r in got) <= rec[dt]["max_dlogp"] * 1.0001
    # every switched-off norm kind the rule names was measured on some real pair
    measured = {n for r in rows if r.get("payable") and r["dtype"] == "float32" for n in r["switched_off"]}
    assert set(rule.SWITCHED_OFF) <= measured


def test_eps_lever_rule_shape():
    from graph import eps_lever as rule
    assert set(rule.SWITCHED_OFF) == {"q_norm", "k_norm", "v_norm", "post_attention_layernorm",
                                      "post_feedforward_layernorm", "kv_a_layernorm"}
    assert rule.CURRENCY == "free"
    assert rule.host_norm_off("qk_norm") and rule.host_norm_off("post_norm") and rule.host_norm_off("v_norm")
    assert rule.host_norm_off("mla_latent_norm")
    assert not rule.host_norm_off("qk_norm", eps=0.0)            # eps = 0: no linear regime
    assert not rule.host_norm_off("qk_norm", dtype="float16")    # 2^-14 normal floor


def test_gelu_floors():
    from act.activation_cross import ACTS
    from act.gelu_swap_agreement import gelu_tanh_np
    x = np.linspace(-6, 6, 120001)
    raw = np.max(np.abs(ACTS["GeLU"](x) - gelu_tanh_np(x)))
    assert 4.5e-4 < raw < 4.9e-4
    from graph.gelu_variants import GELU_FLOOR_RAW
    assert abs(raw - GELU_FLOOR_RAW) < 1e-6


def test_gelu_swap_verdicts():
    from graph import gelu_variants as rule
    rows = _rows("act/data/gelu_swap_agreement.jsonl")
    for (model, host), payable in rule.GELU_SWAP.items():
        got = [r for r in rows if r["model"] == model and r["host"] == host]
        assert got, (model, host)
        r = got[-1]
        assert (r["top1"] >= 0.99 and r["ex32"] >= 0.90) == payable, (model, host, r["top1"], r["ex32"])


def test_attn_pin():
    from types import SimpleNamespace as NS
    from hf.attn_pin import attn_impl
    assert attn_impl(NS(model_type="gpt2", scale_attn_by_inverse_layer_idx=True)) == "eager"
    assert attn_impl(NS(model_type="gemma2", attn_logit_softcapping=50.0)) == "eager"
    assert attn_impl(NS(model_type="falcon", alibi=True)) == "sdpa"
    assert attn_impl(NS(model_type="falcon", alibi=False)) == "eager"


def test_gpt2_sdpa_drops_layer_idx_scaling():
    """the reason for the pin, on a 2-layer random GPT-2: sdpa != eager iff the layer-idx scaling is on."""
    from transformers import GPT2Config, GPT2LMHeadModel
    out = {}
    for flag in (False, True):
        torch.manual_seed(0)
        cfg = GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=50, n_positions=32,
                         scale_attn_by_inverse_layer_idx=flag)
        m = GPT2LMHeadModel(cfg).double().eval()
        with torch.no_grad():
            for p in m.parameters():
                p.normal_(0, 0.5)
        ids = torch.randint(0, 50, (1, 12))
        lo = {}
        for impl in ("eager", "sdpa"):
            m.config._attn_implementation = impl
            with torch.no_grad():
                lo[impl] = m(ids).logits
        out[flag] = (lo["eager"] - lo["sdpa"]).abs().max().item()
    assert out[False] < 1e-6 and out[True] > 1e-3, out


def test_attn_audit_only_known_differences():
    """eager == sdpa on every recorded model except Gemma-2 (softcap) and the two bug demonstrators."""
    rows = [r for r in _rows("hf/data/attn_impl_audit.jsonl") if "top1" in r]
    differ = {r["model"] for r in rows if not r["same_model"]}
    assert differ == {"google/gemma-2-2b", "google/gemma-2-2b-it", "stanford-crfm/alias-gpt2-small-x21",
                      "tiiuae/falcon-rw-1b"}, differ


def test_eager_reruns_verdicts():
    rows = _rows("hf/data/eager_reruns.jsonl")
    changed = {(r["harness"].split("/")[-1], r["variant"]) for r in rows if r["verdict_changed"]}
    assert changed == {("act_swap_agreement.py", "GeLU"), ("act_swap_agreement.py", "fp16")}, changed
