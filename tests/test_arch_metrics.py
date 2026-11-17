"""The architecture metrics (graph/arch_metrics.py): the JSON the zoo reads, exact values for known models, the
formulas against the real transformers code (meta-device parameter counts, tiny-config FLOP / cache traces, one
symbolic proof), and Hypothesis property tests over random configs.

Run:  pytest tests/test_arch_metrics.py -n 2
"""
import json
import os

import pytest

from graph import arch_metrics as AM

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "..", "graph", "data", "arch_metrics.json")
CONFIGS = os.path.join(HERE, "..", "graph", "hub_10k.json")


@pytest.fixture(scope="module")
def data():
    with open(JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def configs():
    from graph.model_graph import iter_json_configs
    return dict(iter_json_configs(CONFIGS))


# --- the file the zoo reads ------------------------------------------------------------------------------------

def test_top_level(data):
    for k in ("schema_version", "generated", "snapshot", "transformers_version", "contexts", "batch_B", "dtype_bytes",
              "conventions", "coverage", "types", "shapes", "repos"):
        assert k in data, k
    assert data["contexts"] == {"1k": 1024, "4k": 4096, "8k": 8192, "32k": 32768, "128k": 131072}
    assert data["dtype_bytes"] == {"bf16": 2, "fp8": 1, "int4": 0.5}


def test_shapes_line_up_with_the_zoo(data):
    from graph.frontier import load
    from graph.model_graph import _dedup_points
    P = _dedup_points(load())
    assert len(data["shapes"]) == len(P)
    for s, p in zip(data["shapes"], P):
        assert s["rep"] in p["members"]
        assert s["n_members"] == len(p["members"])
        assert s["metrics"] == data["repos"][s["rep"]]


def test_every_repo_has_provenance_and_status(data):
    for name, r in data["repos"].items():
        if "error" in r:
            continue
        for g in AM.GROUPS:
            assert r["prov"][g] in ("config", "code", "trace-verified"), (name, g)
            assert r["valid"][g] in AM.STATUS, (name, g)
        assert set(r["kv"]) == set(data["contexts"]) | {"max"}


def test_coverage_counts(data):
    n = sum(1 for r in data["repos"].values() if "valid" in r)
    for g, c in data["coverage"]["repos"].items():
        assert sum(c.values()) == n, g


# --- exact values ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("repo,total,active,kv_bf16", [
    ("Qwen/Qwen3-8B", 8190735360, 8190735360, 36 * 8 * 128 * 2 * 2),
    ("meta-llama/Llama-3.1-8B-Instruct", 8030261248, 8030261248, 32 * 8 * 128 * 2 * 2),
    ("deepseek-ai/DeepSeek-R1", 671026404352, 37552282624, 61 * (512 + 64) * 2),
])
def test_known_models(data, repo, total, active, kv_bf16):
    r = data["repos"][repo]
    assert r["params"]["total"] == total
    assert r["params"]["active"] == active
    assert r["kv_per_token"]["bf16"] == kv_bf16
    assert r["kv"]["4k"]["bf16"] == kv_bf16 * 4096


def test_sliding_window_flat(data):
    r = data["repos"]["google/gemma-3-1b-it"] if "google/gemma-3-1b-it" in data["repos"] else next(
        x for x in data["repos"].values() if x.get("layers", {}).get("sliding"))
    W = r["layers"]["window"]
    small = [k for k, v in r["kv"].items() if v["L"] < W - 1]
    big = sorted((v["L"], k) for k, v in r["kv"].items() if v["L"] > W)
    for k in small:
        assert r["kv"][k]["in_window"]["sliding"] is True
    for (L1, k1), (L2, k2) in zip(big, big[1:]):
        assert r["kv"][k1]["in_window"]["sliding"] is False
        # beyond the window only the global layers grow
        assert r["kv"][k2]["elems"]["sliding"] == r["kv"][k1]["elems"]["sliding"]


# --- the formulas against the stock code -------------------------------------------------------------------------

SMALL = ["openai-community/gpt2", "Qwen/Qwen3-0.6B", "state-spaces/mamba-130m-hf", "EleutherAI/pythia-160m",
         "tiiuae/falcon-mamba-tiny-dev"]


@pytest.mark.parametrize("repo", SMALL)
def test_meta_params(configs, repo):
    from graph.arch_trace import check_params
    if repo not in configs:
        pytest.skip("not in the snapshot")
    r = check_params(configs[repo])
    assert r["equal"], r


@pytest.mark.parametrize("repo", ["Qwen/Qwen3-0.6B", "state-spaces/mamba-130m-hf"])
def test_tiny_forward(configs, repo):
    from graph.arch_trace import check_forward
    r = check_forward(configs[repo])
    assert all(v is not False for v in r["equal"].values()), r


def test_tiny_forward_sliding(configs):
    from graph.arch_trace import check_forward
    repo = next(n for n, c in configs.items() if c.get("model_type") == "gemma3_text"
                and AM.stock_arch(c))
    r = check_forward(configs[repo])
    assert all(r["equal"].values()), r


def test_symbolic_prefill(configs):
    from graph.arch_trace import check_symbolic
    r = check_symbolic(configs["Qwen/Qwen3-0.6B"])
    assert r["flops_equal"] and r["kv_equal"], r


# --- properties ---------------------------------------------------------------------------------------------------

from hypothesis import given, settings, strategies as st


@st.composite
def llama_like(draw):
    nkv = draw(st.sampled_from([1, 2, 4, 8]))
    group = draw(st.sampled_from([1, 2, 4]))
    hd = draw(st.sampled_from([64, 128]))
    h = draw(st.sampled_from([256, 512, 1024]))
    c = {"model_type": "mistral", "architectures": ["MistralForCausalLM"], "hidden_size": h,
         "num_attention_heads": nkv * group, "num_key_value_heads": nkv, "head_dim": hd,
         "intermediate_size": draw(st.sampled_from([512, 1024, 2048])),
         "num_hidden_layers": draw(st.integers(1, 12)), "vocab_size": draw(st.sampled_from([1000, 32000])),
         "max_position_embeddings": 32768, "rms_norm_eps": 1e-5, "hidden_act": "silu",
         "sliding_window": draw(st.sampled_from([None, 64, 4096])), "tie_word_embeddings": draw(st.booleans())}
    if draw(st.booleans()):
        c.update(model_type="mixtral", architectures=["MixtralForCausalLM"],
                 num_local_experts=draw(st.sampled_from([2, 4, 8])), num_experts_per_tok=draw(st.sampled_from([1, 2])))
    return c


@settings(max_examples=40, deadline=None)
@given(llama_like(), st.integers(1, 200000), st.integers(1, 200000))
def test_properties(c, L1, L2):
    p = AM.plan(c)
    prm = AM.params(p)
    assert prm["active"] <= prm["total"]
    assert prm["active_non_embedding"] <= prm["non_embedding"]
    a, b = sorted((L1, L2))
    ka, kb = sum(AM.kv_at(p, a).values()), sum(AM.kv_at(p, b).values())
    assert ka <= kb                                         # monotone in L
    W = c["sliding_window"]
    if W and a >= W - 1:
        assert ka == kb                                     # all layers windowed: flat beyond the window
    fa, fb = AM.flops(p, a, "decode"), AM.flops(p, b, "decode")
    assert fa["matmul"] == fb["matmul"] and fa["attn"] <= fb["attn"]
    # causal prefill never exceeds the dense (masked) count
    assert AM.attn_flops(p, b, "prefill") <= AM.attn_flops(p, b, "prefill_dense") + 1e-6
    m = AM.metrics(c, p=p)
    for k, v in m["memory"].items():
        assert v["batchB"] >= v["batch1"]
    for k, v in m["bandwidth"].items():
        assert v["bytes"] >= v["weights"] and v["intensity"] > 0


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 5000), st.integers(1, 5000))
def test_span_closed_form(W, L):
    assert AM._span_causal_sum(L, W) == sum(min(i, W) for i in range(1, L + 1))


def test_repo_properties(data):
    """Every repo: active <= total, KV and memory monotone over the contexts, bytes = elements x dtype."""
    for name, r in data["repos"].items():
        if "error" in r:
            continue
        p = r["params"]
        assert p["active"] <= p["total"], name
        assert p["weight_bytes"]["bf16"] == 2 * p["total"]
        rows = sorted(r["kv"].values(), key=lambda v: v["L"])
        for x, y in zip(rows, rows[1:]):
            assert x["bf16"] <= y["bf16"], name
        for v in r["kv"].values():
            assert v["bf16"] == 2 * v["fp8"]
            assert v["bf16"] == 2 * sum(v["elems"].values())
