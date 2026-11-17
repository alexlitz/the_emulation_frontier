"""A fixed handful of the random spot checks (verify/spot/run_spot_checks.py), fast: the sampler's stratification,
the refusal bookkeeping, and three edges built end to end at scaled-down size (the pair's real configs with every
width divided by one power of two, a random target, emulate.py's construction in memory, fp64 exact).

  pytest tests/test_spot_checks.py -q
"""
import json
from pathlib import Path

import pytest
import torch

from verify.spot import run_spot_checks as S

SNAP = Path(__file__).resolve().parent.parent / "graph" / "hub_10k.json"


@pytest.fixture(scope="module")
def configs():
    return {r["id"]: r["config"] for r in json.loads(SNAP.read_text())}


def test_sampler_is_stratified_and_deterministic():
    tags = ["capacity", "slot:kv", "slot:window", "mask", "recurrent"]
    edges = [(i, i + 1, [tags[i % len(tags)]] + (["approx"] if i % 7 == 0 else [])) for i in range(500)]
    a = S.sample(edges, 40, seed=3)
    assert a == S.sample(edges, 40, seed=3) and len(a) >= 40
    got = {t for (_, _, ts), _ in a for t in ts}
    assert got >= set(tags) | {"approx"}
    assert S.sample(edges, 40, seed=4) != a


def test_edge_tags():
    from graph.model_graph import classify, dims_of
    from verify.spot.catalog._emu import SMALL, BIG, raw
    p = lambda d: {"slots": classify(d), "dims": dims_of(d)}
    A, B = p(raw("llama", dict(SMALL, num_key_value_heads=1))), p(raw("qwen2", BIG))
    tags = S.edge_tags(A, B, {"llama"}, {"qwen2"})
    assert "slot:kv" in tags and "step:kv:MQA->GQA" in tags and "cross_family" in tags and "capacity" not in tags


def test_missing_pieces_are_categorised():
    assert S.missing_pieces(["target: model_type 'gpt_oss' is not a v1 family"]) == ["family not in v1: gpt_oss"]
    assert S.missing_pieces(["heads: ... (KV-head packing -- a longrope table ...) is not built by v1"]) == [
        "KV-head packing (longrope table per packed head)"]


@pytest.mark.parametrize("target,host", [
    ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M"),          # every capacity axis, same family
    ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen1.5-1.8B"),                            # GQA -> MHA, head_dim 64 -> 128
    ("HuggingFaceTB/SmolLM2-135M", "ibm-granite/granite-3.3-2b-base"),     # across families: Granite multipliers
])
def test_scaled_down_edge_is_exact(target, host, configs):
    import elephantine.emulate as E
    from verify.spot.catalog import compare, hf_fn
    from verify.spot.catalog._emu import emulate_pair
    t, h, f = S.shrink(configs[target], configs[host], target_hidden=32)
    assert f > 1
    for c in (t, h):
        c["vocab_size"] = 97
        c["num_hidden_layers"] = min(c["num_hidden_layers"], 4 if c is t else 6)
        for k in ("layer_types", "max_window_layers"):
            c.pop(k, None)
    target_m, host_m, con, plan = emulate_pair(t, h)
    assert plan["exact"]
    with E._lift():
        r = compare(hf_fn(target_m), hf_fn(host_m), 97, n=4, new_tokens=16,
                    tol=1e-6 if any("not an integer" in n for n in con.notes) else 1e-9)
    assert r.passed, r.line()


def test_unsupported_pair_records_the_missing_piece(configs):
    from graph.model_graph import classify, dims_of
    p = lambda n: {"slots": classify(configs[n]), "dims": dims_of(configs[n]), "members": [n]}
    A, B = p("HuggingFaceTB/SmolLM2-135M"), p("Qwen/Qwen2.5-3B")
    t, h, reasons = S.pick_pair(A, B, configs, {})
    # emulate v3 builds this pair (KV-head packing over a longrope table): no refusal; a refusal's reason still maps
    # to the piece the script lacks
    assert not reasons, reasons
    assert S.missing_pieces(["heads: KV-head packing is not exact for a per-head qk-norm target"]) == \
        ["KV-head packing (longrope table per packed head)"]
    assert S.missing_pieces(["windows: the target uses several window sizes [4, 8]; the host has one"]) == \
        ["per-layer window sizes"]


@pytest.mark.xfail(strict=True, reason="FINDING (emulate.py): a non-integer head_dim ratio rescales theta to "
                   "theta^(hd_h / hd_t), computed in fp32 at init; 10000^(128 / 6) = 2e85 overflows fp32, so the host's "
                   "inv_freq is 0 past pair 0 (spot check MaxJeblick/llama2-0b-unit-test <- BSC-LT/salamandra-2b)")
def test_theta_rescale_overflows_fp32():
    import elephantine.emulate as E
    from verify.spot.catalog import compare, hf_fn
    from verify.spot.catalog._emu import SMALL, BIG, emulate_pair, raw
    t = raw("llama", dict(SMALL, hidden_size=12, num_attention_heads=2, num_key_value_heads=2, head_dim=6,
                          intermediate_size=24))
    h = raw("llama", dict(BIG, num_attention_heads=6, num_key_value_heads=6, head_dim=128, hidden_size=96))
    target, host, con, plan = emulate_pair(t, h)
    assert any("not an integer" in n for n in con.notes)
    with E._lift():
        r = compare(hf_fn(target), hf_fn(host), 97, n=4, new_tokens=8, tol=1e-6)
    assert r.passed, r.line()
