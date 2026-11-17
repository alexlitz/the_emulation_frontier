"""The proposed start-marker (BOS) graph rules (graph/bos_rules.py): the charges, the host conditions,
their agreement with the stock-module construction (hf/bos_constructions.py), and the headline
estimate's structure on the zoo snapshot.

Run:  pytest tests/test_bos_rules.py -q
"""
import os

import pytest

from graph.bos_rules import (BOS_EXTRA_HIDDEN, BOS_EXTRA_LAYERS, BOS_LEARNED_ABS_HOSTS_ALIBI, SINK_EXTRA_HIDDEN,
                             bos_learned_abs_charges, bos_learned_abs_host_ok, bos_learned_abs_ok,
                             bos_learned_abs_target_dims, bos_learned_abs_units, bos_sink_ok, has_q_bias,
                             sink_target_dims)

HUB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph", "hub_10k.json")

GPT2 = dict(hidden=768, eff_layers=12, param_layers=12, layers=12, ffn_active=3072, ffn_total=3072,
            max_context=1024, head_dim=64, q_cap=768, kv_cap=768)
LLAMA = dict(norm="RMSNorm", pos="rope", act="SiLU", ffn="glu", mixer="softmax", window="full", sink="no")


def _host(**kw):
    d = dict(hidden=2048, eff_layers=16, param_layers=16, ffn_active=8192, full_eff=16, head_dim=64)
    d.update(kw)
    return d


def test_learned_abs_charges():
    lean = bos_learned_abs_charges(GPT2, LLAMA)
    assert lean["layers"] == 2 and lean["hidden"] == 5 and lean["T"] == 1023
    assert lean["ffn_units"] == 5 + 768 * (1023 + 3)                    # the Lean construction as written
    shared = bos_learned_abs_charges(GPT2, LLAMA, shared=True)
    assert shared["ffn_units"] == 1023 + 1 + 5                           # shared kinks + 5 erasers
    assert bos_learned_abs_charges(GPT2, {**LLAMA, "norm": "LayerNorm"})["hidden"] == 10   # mirrored
    assert bos_learned_abs_units(768, 1023, shared=True, gated=False) == (1024, 1034)    # 2 units / eraser


def test_learned_abs_host_conditions():
    assert bos_learned_abs_host_ok(LLAMA)
    assert bos_learned_abs_host_ok({**LLAMA, "pos": "nope"}) and bos_learned_abs_host_ok({**LLAMA, "pos": "partial_rope"})
    assert not bos_learned_abs_host_ok({**LLAMA, "pos": "alibi"})                       # no stock slope buffer
    assert bos_learned_abs_host_ok({**LLAMA, "pos": "alibi"}, hosts=BOS_LEARNED_ABS_HOSTS_ALIBI)
    assert not bos_learned_abs_host_ok({**LLAMA, "pos": "t5_relbias"})
    assert bos_learned_abs_host_ok({**LLAMA, "ffn": "vanilla", "act": "GeLU"})          # gelu(z) - gelu(-z) = z
    assert not bos_learned_abs_host_ok({**LLAMA, "ffn": "vanilla", "act": "mish"})      # no exact linear unit
    assert not bos_learned_abs_host_ok({**LLAMA, "act": "poly_norm"})
    assert not bos_learned_abs_host_ok({**LLAMA, "window": "local"}, _host(full_eff=0))  # block 1 sees the marker
    assert bos_learned_abs_host_ok({**LLAMA, "window": "local"}, _host(full_eff=4))


def test_learned_abs_capacity():
    assert bos_learned_abs_ok(GPT2, _host(), LLAMA, shared=True)
    assert not bos_learned_abs_ok(GPT2, _host(), LLAMA)                                  # Lean: 788k units
    assert not bos_learned_abs_ok(GPT2, _host(eff_layers=13, param_layers=13), LLAMA, shared=True)   # +2 layers
    assert bos_learned_abs_ok(GPT2, _host(eff_layers=14, param_layers=14), LLAMA, shared=True)
    assert not bos_learned_abs_ok(GPT2, _host(hidden=772), LLAMA, shared=True)           # +5 channels
    assert bos_learned_abs_ok(GPT2, _host(hidden=773), LLAMA, shared=True)
    assert not bos_learned_abs_ok(GPT2, _host(hidden=777), {**LLAMA, "norm": "LayerNorm"}, shared=True)
    assert not bos_learned_abs_ok(GPT2, _host(ffn_active=1028), LLAMA, shared=True)      # T + 6 units
    t = bos_learned_abs_target_dims(GPT2, LLAMA, shared=True)
    assert (t["hidden"], t["eff_layers"], t["ffn_active"]) == (773, 14, 3072)


def test_charges_match_the_construction():
    """The stock-module build uses exactly the charged extra layers / channels and the shared unit count."""
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    from hf.bos_constructions import F64, gpt2_spec, learned_abs_host
    torch.manual_seed(0)
    tgt = GPT2LMHeadModel(GPT2Config(vocab_size=97, n_embd=32, n_layer=2, n_head=4, n_positions=64,
                                     bos_token_id=96)).to(F64).eval()
    host = learned_abs_host(gpt2_spec(tgt))
    d, T = 32, 63
    assert host.config.hidden_size == d + BOS_EXTRA_HIDDEN["RMSNorm"]
    assert host.config.num_hidden_layers == 2 + BOS_EXTRA_LAYERS
    used = [int((lay.mlp.gate_proj.weight.abs().sum(1) > 0).sum()) for lay in host.model.layers[:2]]
    assert used == list(bos_learned_abs_units(d, T, shared=True))


def test_has_q_bias():
    assert has_q_bias({"model_type": "qwen2"})                              # always biased q / k / v
    assert not has_q_bias({"model_type": "llama", "attention_bias": False})
    assert has_q_bias({"model_type": "llama", "attention_bias": True})
    assert not has_q_bias({"model_type": "mistral"})
    assert has_q_bias({"model_type": "gpt_oss", "attention_bias": True})
    assert not has_q_bias({"model_type": "internlm2", "bias": False})
    assert has_q_bias({"model_type": "qwen", "no_bias": True})              # Qwen v1: c_attn keeps its bias


def test_sink_rule():
    oss = dict(hidden=2880, head_dim=64, q_cap=64 * 64, kv_cap=8 * 64)
    rope = dict(LLAMA)
    assert bos_sink_ok(oss, dict(hidden=4096, head_dim=128), {**rope, "sink": "yes"}, False)   # a sink host
    assert not bos_sink_ok(oss, dict(hidden=4096, head_dim=128), rope, host_q_bias=False)     # needs a q bias
    assert bos_sink_ok(oss, dict(hidden=4096, head_dim=128), rope, host_q_bias=True)
    assert not bos_sink_ok(oss, dict(hidden=4096, head_dim=65), rope, True)                   # a whole pair
    assert bos_sink_ok(oss, dict(hidden=4096, head_dim=66), rope, True)
    assert bos_sink_ok(oss, dict(hidden=4096, head_dim=65), {**rope, "pos": "nope"}, True)    # no rotation
    assert not bos_sink_ok(oss, dict(hidden=2880, head_dim=128), rope, True)                  # + sigma
    t = sink_target_dims(oss, rope)
    assert t["head_dim"] == 66 and t["q_cap"] == 64 * 66 and t["kv_cap"] == 8 * 66
    assert t["hidden"] == 2880 + SINK_EXTRA_HIDDEN


@pytest.mark.skipif(not os.path.exists(HUB), reason="no zoo snapshot")
def test_estimate_structure():
    """On the snapshot: the Lean charges change nothing, a model can only LEAVE the frontier under the
    learned-abs rule if it is learned-abs, and only a sink model can ENTER under the sink rule."""
    from graph.bos_rules import _points, estimate
    est = estimate(HUB)
    P = {p["members"][0]: p for p in _points(HUB)}
    assert est["learned_abs (Lean charges)"]["frontier"] == est["baseline"]
    for k, v in est.items():
        if not isinstance(v, dict):
            continue
        if k.startswith("learned_abs"):
            assert not v["enter"]
            assert all(P[n]["slots"]["pos"] == "learned_abs" for n in v["leave"])
        if k == "sink rule":
            assert not v["leave"]
            assert all(P[n]["slots"]["sink"] == "yes" for n in v["enter"])
