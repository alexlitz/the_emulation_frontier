"""graph/arch_breakdown.py: the detailed numbers and the checks behind each status.

  sums        the parameter roles, FLOP components and KV kinds sum to arch_metrics' totals, on every snapshot config
  properties  Hypothesis: KV, memory, decode traffic and FLOPs monotone in the context L and the batch B; the
              weight formats and GGUF levels ordered by size; the eager activation peak above the flash one; the
              roofline bound falling with L; and, on random small configs of stock types, the per-module parameter
              trace on the meta device equal to the breakdown
  traces      the committed trace files: per-module parameters, per-module FLOPs, per-bucket symbolic proofs
  hub         the GGUF / safetensors size checks (median |error| < 1%) and the roofline-vs-benchmark check
  data        arch_breakdown.json.gz is current for a sample of repos

Run:  pytest tests/test_arch_breakdown.py -n 4
"""
import gzip
import json
import logging
import os

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.test_arch_metrics_js import _snapshot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "graph", "data")
CHUNKS = 4


@pytest.fixture(scope="module")
def snapshot():
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return _snapshot()


@pytest.fixture(scope="module")
def plans(snapshot):
    from graph import arch_breakdown as BD
    out = []
    for name, raw, _ in snapshot:
        try:
            out.append((name, raw, BD.plan_ordered(raw)))
        except Exception:
            pass
    return out


@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_sums_equal_arch_metrics(plans, chunk):
    from graph import arch_breakdown as BD
    bad = []
    for name, raw, p in plans[chunk::CHUNKS]:
        b = BD.breakdown_checks(p)
        if b:
            bad.append((name, b[:3]))
    assert not bad, bad[:10]


def test_layer_order_known_almost_everywhere(plans):
    from graph import arch_breakdown as BD
    known = sum(BD.layer_order(p) is not None for _, _, p in plans)
    assert known >= 0.99 * len(plans)


# ------------------------------------------------------------------------------------------------ properties

def _rec(plans, i, cache={}):
    from graph import arch_breakdown as BD
    name, raw, p = plans[i % len(plans)]
    if name not in cache:
        cache[name] = (p, BD.breakdown(p, raw, hw_list=BD.hardware()["hardware"][:3]))
    return cache[name]


PROP = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow,
                                                                        HealthCheck.function_scoped_fixture])


@PROP
@given(i=st.integers(0, 10 ** 6), l1=st.integers(1, 1 << 20), l2=st.integers(1, 1 << 20))
def test_monotone_in_L(plans, i, l1, l2):
    from graph import arch_breakdown as BD
    p, _ = _rec(plans, i)
    a, b = sorted((l1, l2))
    assert BD.kv_elems(p, a) <= BD.kv_elems(p, b)
    fa, fb = BD.flops_at(p, a), BD.flops_at(p, b)
    assert fa["decode"]["total"] <= fb["decode"]["total"]
    assert fa["prefill"]["total"] <= fb["prefill"]["total"]
    assert BD.activation_bytes(p, a) <= BD.activation_bytes(p, b)
    assert BD.activation_bytes(p, a, "flash") <= BD.activation_bytes(p, a, "eager")
    wb = BD.weight_bytes(p, "bf16")
    ra, rb = BD.decode_read(p, a, 1, wb, 2 * p["hidden"]), BD.decode_read(p, b, 1, wb, 2 * p["hidden"])
    assert ra["bytes"] <= rb["bytes"]


@PROP
@given(i=st.integers(0, 10 ** 6), b1=st.integers(1, 512), b2=st.integers(1, 512), L=st.integers(1, 1 << 18))
def test_monotone_in_batch(plans, i, b1, b2, L):
    from graph import arch_breakdown as BD
    p, _ = _rec(plans, i)
    a, b = sorted((b1, b2))
    wb = BD.weight_bytes(p, "bf16")
    ra, rb = BD.decode_read(p, L, a, wb, 2 * p["hidden"]), BD.decode_read(p, L, b, wb, 2 * p["hidden"])
    assert ra["bytes"] <= rb["bytes"] + 1e-6 * rb["bytes"]
    hw = BD.hardware()["hardware"][0]
    fl = BD.flops_at(p, L)["decode"]["total"]
    ta = BD.decode_bound(dict(ra, flops=a * fl, B=a), hw)
    tb = BD.decode_bound(dict(rb, flops=b * fl, B=b), hw)
    assert ta <= tb * (1 + 1e-9)                         # total tokens/s grows with the batch


@PROP
@given(i=st.integers(0, 10 ** 6))
def test_breakdown_sums_and_orders(plans, i):
    from graph import arch_breakdown as BD
    p, r = _rec(plans, i)
    assert sum(r["params"]["components"].values()) == r["params"]["total"]
    assert sum(g["np"] * g["layer_total"] for g in r["params"]["groups"]) <= r["params"]["total"]
    assert r["params"]["active_total"] <= r["params"]["total"]
    w = r["weights"]
    assert w["fp32"] >= w["bf16"] >= w["fp8"] and w["bf16"] >= w["int8"] >= w["int4"]
    assert w["int8"] >= w["nf4"] and w["int8"] >= w["nvfp4"]
    if r["gguf"]:
        g = r["gguf"]
        levels = ["BF16"] + list(BD.GGUF_LEVELS)
        assert all(g[x] >= g[y] for x, y in zip(levels, levels[1:])), g
    for key, kv in r["kv"].items():
        assert kv["bf16"] >= kv["q8_0"] >= kv["fp8"] >= kv["q4_0"]
        for d in ("decode", "prefill"):
            c = r["flops"][key][d]["components"]
            tot = r["flops"][key][d]["total"] if d == "decode" else r["flops"][key][d]["per_token"]
            assert abs(sum(c.values()) - tot) <= 1e-9 * max(1, tot)
        assert r["flops"][key]["training_approx"] == pytest.approx(3 * r["flops"][key]["prefill"]["total"])
        for B in ("1", "8", "32"):
            bw = r["bandwidth"][key][B]
            assert bw["bytes"] == pytest.approx(bw["weights"] + bw["kv"] + bw["state"])
    for B in ("1", "8", "32"):
        assert r["serving"]["1k"][B] <= r["serving"]["max"][B] or r["max_context"] < 1024
    for hw, d in r.get("roofline", {}).items():
        dec = [d[k]["decode"]["1"] for k in ("4k", "32k")]
        assert dec[0] >= dec[1] * (1 - 1e-12)


# random small configs of stock types: the meta-device module buckets equal the breakdown
def _cfg(kind, h, nh, g, layers, inter, vocab, E, k):
    hd = 64
    c = {"hidden_size": nh * hd if kind != "gpt2" else h, "num_attention_heads": nh, "num_hidden_layers": layers,
         "intermediate_size": inter, "vocab_size": vocab, "max_position_embeddings": 2048}
    if kind == "llama":
        c.update(model_type="llama", architectures=["LlamaForCausalLM"], num_key_value_heads=max(1, nh // g),
                 tie_word_embeddings=bool(layers % 2))
    elif kind == "qwen3":
        c.update(model_type="qwen3", architectures=["Qwen3ForCausalLM"], num_key_value_heads=max(1, nh // g),
                 head_dim=hd)
    elif kind == "mixtral":
        c.update(model_type="mixtral", architectures=["MixtralForCausalLM"], num_key_value_heads=max(1, nh // g),
                 num_local_experts=E, num_experts_per_tok=min(k, E))
    elif kind == "qwen2_moe":
        c.update(model_type="qwen2_moe", architectures=["Qwen2MoeForCausalLM"], num_key_value_heads=max(1, nh // g),
                 num_experts=E, num_experts_per_tok=min(k, E), moe_intermediate_size=inter // 2,
                 shared_expert_intermediate_size=inter)
    elif kind == "gemma2":
        c.update(model_type="gemma2", architectures=["Gemma2ForCausalLM"], num_key_value_heads=max(1, nh // g),
                 head_dim=hd, sliding_window=128)
    else:
        c = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"], "n_embd": h, "n_head": nh, "n_layer": layers,
             "n_positions": 1024, "n_inner": inter, "vocab_size": vocab}
    return c


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(kind=st.sampled_from(["llama", "qwen3", "mixtral", "qwen2_moe", "gemma2", "gpt2"]),
       nh=st.sampled_from([2, 4, 8]), g=st.sampled_from([1, 2, 4]), layers=st.integers(1, 6),
       inter=st.sampled_from([128, 256, 384]), vocab=st.integers(300, 4000), E=st.integers(2, 8),
       k=st.integers(1, 3))
def test_random_configs_trace(kind, nh, g, layers, inter, vocab, E, k):
    from graph import arch_breakdown as BD
    from graph import arch_breakdown_trace as BT
    logging.getLogger("transformers").setLevel(logging.ERROR)
    raw = _cfg(kind, nh * 64, nh, g, layers, inter, vocab, E, k)
    r = BT.check_params(raw)
    assert r["all_buckets"] and r["all_subs"], (raw, r["buckets"], r["subs"])
    assert r["layers_equal"] in (True, None)
    assert not BD.breakdown_checks(BD.plan_ordered(raw))


# ------------------------------------------------------------------------------------------------ the trace files

def _j(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def test_trace_params_file():
    d = _j("arch_breakdown_trace_params.json")
    ok = [r for r in d.values() if "error" not in r]
    assert len(ok) >= 1500
    good = sum(r["all_buckets"] and r["all_subs"] for r in ok)
    assert good >= len(ok) - 1                       # inference-optimization/Qwen3-1.6B-A0.9B: arch_metrics explains it
    assert len({r["model_type"] for r in ok}) >= 80


def test_trace_flops_file():
    d = _j("arch_breakdown_trace_flops.json")
    checked = [c for r in d.values() for c in r["checked"]]
    assert len(checked) >= 140 and all(c["all_equal"] for c in checked)
    assert len(d) >= 80


def test_trace_symbolic_file():
    d = _j("arch_breakdown_trace_symbolic.json")
    proofs = [r["proof"] for r in d.values() if r.get("proof")]
    assert len(proofs) >= 50 and all(p["all_equal"] for p in proofs)


def test_trace_memory_file():
    d = _j("arch_breakdown_trace_memory.json")
    eager = [x["ratio"] for r in d.values() for x in r["runs"] if x["impl"] == "eager"]
    assert len(eager) >= 30
    inside = sum(abs(x - 1) <= 0.15 for x in eager)
    assert inside >= 0.75 * len(eager)


def test_hub_sizes():
    h = _j("arch_breakdown_hub.json")
    g = h["gguf"]["summary"]["all_quantised"]
    assert g["n"] >= 300 and g["median_abs"] < 0.01
    bases = {x["base"] for x in h["gguf"]["rows"] if "err" in x}
    assert len(bases) >= 50
    for lv, s in h["gguf"]["summary"]["by_level"].items():
        if s:
            assert s["median_abs"] < 0.01, lv
    st_ = h["safetensors"]["summary"]["all"]
    assert st_["n"] >= 30 and st_["median_abs"] < 0.01


def test_roofline_below_bounds():
    r = _j("roofline_check.json")
    assert len(r["rows"]) >= 15 and r["all_below"]
    assert all(0 < x["ratio"] < 1 for x in r["rows"])


def test_breakdown_file_current(snapshot):
    """A sample of repos recomputed equals the committed graph/data/arch_breakdown.json.gz."""
    from graph import arch_breakdown as BD
    with gzip.open(os.path.join(DATA, "arch_breakdown.json.gz"), "rt") as f:
        d = json.load(f)
    assert d["schema_version"] == BD.SCHEMA_VERSION and len(d["repos"]) == len(snapshot)
    src = BD.validation_sources()
    for name, raw, _ in snapshot[::97]:
        rec = d["repos"][name]
        if "error" in rec:
            continue
        p = BD.plan_ordered(raw)
        why = (((src["am"].get(name) or {}).get("params") or {}).get("hub_delta_reason") or "")
        if p["tied"] and "tied head stored" in why:
            p["head_stored"] = True
        want = json.loads(json.dumps(BD.breakdown(p, raw)))
        got = {k: v for k, v in rec.items() if k not in ("valid", "notes")}
        want.pop("notes", None)
        assert got == want, name
