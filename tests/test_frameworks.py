"""Framework support (graph/frameworks.py): the native registries read at pinned versions, the per-repo native
flags, and the emulation backdoors (every host natively supported, every edge in the headline relation).

Run:  pytest tests/test_frameworks.py -n 2
"""
import json
import os

import pytest

from graph.framework_registries import FRAMEWORKS, load as load_registries
from graph.frameworks import native_support

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "..", "graph", "data", "frameworks.json")


@pytest.fixture(scope="module")
def data():
    with open(JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def reg():
    return load_registries()["frameworks"]


def test_registries_pinned(reg):
    assert set(reg) == set(FRAMEWORKS)
    for fw, r in reg.items():
        assert r["version"] and r["fetched"] and r["urls"], fw
        assert all(u.startswith("https://") for u in r["urls"]), fw
        assert r["architectures"] or r["model_types"], fw
    assert "LlamaForCausalLM" in reg["vllm"]["architectures"]
    assert "LlamaForCausalLM" in reg["llama.cpp"]["architectures"]
    assert {"llama", "qwen3"} <= set(reg["mlx-lm"]["model_types"])
    assert len(reg["vllm"]["architectures"]) > 150 and len(reg["llama.cpp"]["architectures"]) > 100


def test_native_support_keys(reg):
    llama = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    nat = native_support(llama, reg)
    assert set(nat) == set(FRAMEWORKS)
    assert nat["vllm"] and nat["llama.cpp"] and nat["mlx-lm"] and nat["transformers"]
    made_up = {"model_type": "no_such_type_xyz", "architectures": ["NoSuchForCausalLM"]}
    assert not any(native_support(made_up, reg).values())
    assert native_support({}, reg)["vllm"] is None          # no class name: cannot tell


def test_repos_and_counts(data, reg):
    assert set(data["frameworks"]) == set(FRAMEWORKS)
    for fw, c in data["counts"].items():
        assert c["native"] + c["missing"] == c["repos"] == len(data["repos"])
        assert c["backdoor_exact"] + c["backdoor_payable"] + c["no_backdoor"] == c["missing"]


def test_backdoor_hosts_are_native(data):
    for i, s in enumerate(data["shapes"]):
        for fw, b in s["backdoor"].items():
            assert b["n_exact"] <= b["n_hosts"]
            for h in b["hosts"]:
                assert data["repos"][h["host"]]["native"][fw] is True, (i, fw, h["host"])
                assert data["repos"][h["host"]]["shape"] == h["host_shape"]
                assert h["edge"] in ("exact", "payable") and h["exact"] == (h["edge"] == "exact")
            # practical text-LM hosts first, a speech / audio checkpoint last, and exact edges first within each of
            # those classes (graph/frameworks.py's ranking)
            key = [(h["impractical"], "speech / audio model" in h["host_flags"], not h["exact"]) for h in b["hosts"]]
            assert key == sorted(key)


def test_backdoor_edges_hold(data):
    """A sample of backdoor edges: the host shape emulates the target shape under the headline relation (and
    exactly, where marked exact)."""
    from graph.frontier import load
    from graph.model_graph import _dedup_points, emulates
    P = _dedup_points(load())
    assert len(P) == len(data["shapes"])
    seen = 0
    for i, s in enumerate(data["shapes"]):
        for fw, b in s["backdoor"].items():
            for h in b["hosts"][:1]:
                j = h["host_shape"]
                assert emulates(P[j], P[i]), (i, j)
                if h["exact"]:
                    assert emulates(P[j], P[i], mode="exact"), (i, j)
                seen += 1
        if seen > 60:
            break
    assert seen


# --- the Transformers-backend tier, ExecuTorch per model, the host ranking (zoo-fixes 3, 4, 6) -----------------

def _repo(data, name):
    return data["repos"][name]


def test_vllm_transformers_backend_tier(data, reg):
    """vLLM's registry maps these architectures to its Transformers backend (_TRANSFORMERS_SUPPORTED_MODELS):
    shown as their own tier, never counted as native."""
    tb = set(reg["vllm"]["transformers_backend"])
    assert {"PhiForCausalLM", "GPTNeoXForCausalLM", "OlmoForCausalLM", "Olmo2ForCausalLM", "SmolLM3ForCausalLM",
            "Starcoder2ForCausalLM", "GPTBigCodeForCausalLM", "GlmForCausalLM"} <= tb
    assert not tb & set(reg["vllm"]["architectures"])
    assert not any(a.startswith("Transformers") for a in tb)          # the backend's own classes are separate
    phi = _repo(data, "microsoft/phi-2")
    assert phi["native"]["vllm"] is False and phi["via_transformers"]["vllm"] is True
    n = data["counts"]["vllm"]["via_transformers"]
    assert 200 <= n <= 230, n                                         # the eval's 209 of 581 vLLM-missing repos
    for r in data["repos"].values():
        if (r.get("via_transformers") or {}).get("vllm"):
            assert r["native"]["vllm"] is not True and r["arch"] in tb


def test_tgi_transformers_fallback_tier(data, reg):
    t = reg["tgi"]
    assert t["transformers_pin"].startswith("v4.") and t["transformers_fallback"]
    assert not set(t["transformers_fallback"]) & set(t["model_types"])
    hits = [r for r in data["repos"].values() if (r.get("via_transformers") or {}).get("tgi")]
    assert hits and all(r["native"]["tgi"] is not True and r["model_type"] in t["transformers_fallback"] for r in hits)
    assert data["counts"]["tgi"]["via_transformers"] == len(hits)


def test_executorch_per_model(data, reg):
    """ExecuTorch is native only where an export recipe / test exists for that model (its shape), not per type."""
    et = reg["executorch"]
    assert any(c["dir"] == "gemma4" for c in et["export_configs"])
    assert any(c["dir"] == "gemma4_31b" for c in et["export_configs"])
    assert _repo(data, "nvidia/Gemma-4-31B-IT-NVFP4")["native"]["executorch"] is True          # gemma4_31b recipe
    assert _repo(data, "OBLITERATUS/gemma-4-E4B-it-OBLITERATED")["native"]["executorch"] is True  # e4b config
    moe = _repo(data, "nvidia/Gemma-4-26B-A4B-NVFP4")
    assert moe["native"]["executorch"] is False and "no export recipe" in moe["executorch_why"]
    assert _repo(data, "meta-llama/Llama-3.1-70B")["native"]["executorch"] is False
    assert _repo(data, "meta-llama/Llama-3.1-8B")["native"]["executorch"] is True              # README list
    assert _repo(data, "Qwen/Qwen3-0.6B")["native"]["executorch"] is True                      # tests + config
    for r in data["repos"].values():
        if r["native"]["executorch"]:
            assert r["model_type"] in et["model_types"] and r.get("executorch_why")


def test_backdoor_host_ranking(data):
    """No embedding / reward / classifier host, no drafter or checkpoint-anomaly host; practical before impractical
    (> 10x params), a speech / audio host last; exact, then the target's own family, then a plain host before a
    vision-language wrapper or a quantised-only example, then the fewest params."""
    bad = {r for r, v in data["repos"].items() if v.get("not_generative") or v.get("not_a_host")}
    assert "Alibaba-NLP/gte-Qwen2-7B-instruct" in bad
    for s in data["shapes"]:
        for fw, b in s["backdoor"].items():
            hs = b["hosts"]
            assert not {h["host"] for h in hs} & bad
            for h in hs:
                assert h["impractical"] == (h["params_x"] is not None and h["params_x"] > 10)
            fl = lambda h, f: f in h["host_flags"]
            keys = [(h["impractical"], fl(h, "speech / audio model"), not h["exact"], not h["same_family"],
                     fl(h, "vision-language wrapper"), fl(h, "quantised checkpoint"),
                     h["params_x"] if h["params_x"] is not None else 1e18) for h in hs]
            assert keys == sorted(keys)
    q = data["shapes"][_repo(data, "Qwen/Qwen2.5-7B")["shape"]]["backdoor"]
    assert all(b["hosts"][0]["host"] != "Alibaba-NLP/gte-Qwen2-7B-instruct" for b in q.values() if b["hosts"])
    l8 = data["shapes"][_repo(data, "meta-llama/Llama-3.1-8B")["shape"]]["backdoor"]["transformers"]["hosts"][0]
    assert l8["host"] != "kmhf/hf-moshika" and l8["same_family"]


def test_host_repo_is_the_cleanest_member(data):
    """A host shape's example repo is a full-precision, text-only member where the shape has one (the most
    downloaded member was often a GGUF / MLX / NVFP4 quant or a VL wrapper), and never a drafter or an anomaly."""
    from graph.frameworks import host_flags
    assert host_flags("openbmb/MiniCPM4-8B-Eagle-FRSpec", {"architectures": ["LlamaForCausalLM"]}, None) == \
        (True, ["drafter"])
    assert host_flags("x/y", {"architectures": ["MoshiForConditionalGeneration"]}, None)[1][0] == "speech / audio model"
    assert host_flags("x/y-GGUF", {"architectures": ["Qwen3ForCausalLM"]}, None) == (False, ["quantised checkpoint"])
    assert host_flags("x/y", {"architectures": ["Qwen3ForCausalLM"]}, "text-to-speech") == \
        (False, ["speech / audio model"])
    assert host_flags("mobilint/EXAONE-4.0-1.2B", {}, None, {"mobilint/EXAONE-4.0-1.2B"}) == \
        (True, ["checkpoint anomaly"])
    hosts = [h for s in data["shapes"] for b in s["backdoor"].values() for h in b["hosts"]]
    assert all(not data["repos"][h["host"]].get("not_a_host") for h in hosts)
    names = {h["host"] for h in hosts}
    for q in ("prism-ml/Ternary-Bonsai-27B-mlx-2bit", "Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
              "reaperdoesntknow/Qwen3.5-0.8-Cyber-GGUF"):
        assert q not in names, q                       # their shapes hold a full-precision member


def test_config_that_its_class_cannot_read_is_not_native(data):
    """architectures[0] names a stock class, but the config is another model's (NeuralAI-Nae1: 'nae1' keys under
    LlamaForCausalLM): no class-keyed registry can load it. Kimi-K2 and LongCat-Flash (their own keys, which the
    class takes) stay native."""
    r = _repo(data, "Subject-Emu-5259/NeuralAI-Nae1")
    assert r["config_mismatch"].startswith("architectures names LlamaForCausalLM")
    assert not any(r["native"][fw] for fw in ("vllm", "sglang", "llama.cpp", "transformers"))
    for ok in ("moonshotai/Kimi-K2-Instruct", "meituan-longcat/LongCat-Flash-Chat"):
        assert "config_mismatch" not in _repo(data, ok) and _repo(data, ok)["native"]["vllm"], ok
