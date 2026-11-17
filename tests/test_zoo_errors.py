"""zoo-errors: the data fixes outside the parameter builders (those are in tests/test_params_triage.py).

  max context    arch_metrics reads the limit itself, with its provenance, instead of the classifier's 4096 default
  Hub counts     graph/fetch_meta.py keeps the API's safetensors total unless it is broken
"""
import os
from types import SimpleNamespace

import pytest

from graph import arch_metrics as AM
from graph.fetch_meta import param_total
from graph.model_graph import iter_json_configs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = {
    # repo: (max_context, source prefix, unbounded)
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": (1_048_576, "config: model_max_length", False),
    "huggyllama/llama-65b": (2048, "config: max_sequence_length", False),
    "tiiuae/falcon-7b": (2048, "class default: max_position_embeddings", False),
    "state-spaces/mamba-130m-hf": (131_072, "unbounded", True),
    "RWKV/rwkv-4-169m-pile": (131_072, "unbounded", True),
    "google/recurrentgemma-2b": (131_072, "unbounded", True),
    "bigscience/bloom-560m": (4096, "default:", False),
    "root-sys-n/Mima-Pico-15m": (512, "config: block_size", False),
    # llama3 rope: 8,192 x 32 is not the limit; the positions key (131,072) is
    "meta-llama/Llama-3.2-1B-Instruct": (131_072, "config: max_position_embeddings", False),
    "Qwen/Qwen3-8B": (40_960, "config: max_position_embeddings", False),
}


@pytest.fixture(scope="module")
def cfgs():
    want = set(CTX) | {"OpenMOSS-Team/Qwen3-8B-ABC"}
    return {k: v for k, v in iter_json_configs(os.path.join(HERE, "graph", "hub_10k.json")) if k in want}


@pytest.mark.parametrize("repo", sorted(CTX))
def test_max_context_and_its_source(cfgs, repo):
    val, src, unb = CTX[repo]
    m = AM.metrics(cfgs[repo])
    assert m["max_context"] == val
    assert m["max_context_source"].startswith(src), m["max_context_source"]
    assert m["max_context_unbounded"] is unb
    if unb:                                                  # an unbounded model is never "beyond its max"
        assert not any(k["beyond_max"] for k in m["kv"].values())


def test_yarn_window_is_named_not_applied(cfgs):
    """A YaRN rope_scaling that declares more than the positions key is named in the source, the key is kept."""
    m = AM.metrics(cfgs["OpenMOSS-Team/Qwen3-8B-ABC"])
    assert m["max_context"] == 40_960
    assert "x factor 4 = 131,072 declared" in m["max_context_source"]


def test_rope_window_without_a_positions_key():
    raw = {"model_type": "x_custom", "architectures": ["XForCausalLM"], "hidden_size": 256, "num_attention_heads": 4,
           "num_hidden_layers": 2, "intermediate_size": 512, "vocab_size": 1000,
           "rope_scaling": {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}}
    m = AM.metrics(raw)
    assert (m["max_context"], m["max_context_unbounded"]) == (32_768, False)
    assert m["max_context_source"].startswith("rope_scaling yarn: original_max_position_embeddings 4,096 x factor 8")
    raw["rope_scaling"]["rope_type"] = "llama3"            # not a window extension: the default, marked as such
    m = AM.metrics(raw)
    assert m["max_context"] == 4096 and m["max_context_source"].startswith("default:")


def test_no_zoo_repo_has_a_silent_default():
    """Every zoo repo's max_context says where it comes from; the 4,096 default is used only where neither the config
    nor the code has a limit, and is marked."""
    import json
    if not os.path.exists(AM.OUT):
        pytest.skip("graph/data/arch_metrics.json not built")
    with open(AM.OUT) as f:
        reps = json.load(f)["repos"]
    srcs = [r.get("max_context_source") for r in reps.values() if "error" not in r]
    assert all(srcs)
    defaults = [r for r in reps.values() if "error" not in r and r["max_context_source"].startswith("default")]
    assert len(defaults) <= 12 and all(r["max_context"] == 4096 for r in defaults)
    assert all(r["model_type"] == "bloom" for r in defaults)     # ALiBi with no stated length: BLOOM only


@pytest.mark.parametrize("api,per,want", [
    (3_054_832, {"BF16": 27_356_728_560}, (27_356_728_560, {"hf_total_api": 3_054_832})),   # broken API total
    (11_907_350_272, {"U32": 1_488_418_784, "BF16": 471_311_440}, (11_907_350_272, {"hf_total_dtype_sum": 1_959_730_224})),
    (751_632_384, {"BF16": 751_632_384}, (751_632_384, None)),
    (None, {"F32": 1_000}, (1_000, None)),
])
def test_param_total(api, per, want):
    assert param_total(SimpleNamespace(total=api, parameters=per)) == want
    assert param_total(None) == (None, None)


# --- the page (headless browser, the session page of tests/zoo_harness.py) ------------------------------------

pw = pytest.importorskip("playwright.sync_api")

from tests.zoo_harness import enc  # noqa: E402

ROWS_JS = """() => { const rows = {}; document.querySelectorAll('#panel .spec-body > dl.spec > dt').forEach(dt => {
  const dd = dt.nextElementSibling; rows[dt.textContent] = [...dd.childNodes].filter(n => !(n.classList && n.classList.contains('fc')))
  .map(n => n.textContent).join(''); }); return rows; }"""


def _open(p, zoo, repo):
    i = next(k for k, nd in enumerate(zoo.data["nodes"]) if repo in nd["members"])
    p.goto_hash("#m=" + enc(repo))
    p.pg.wait_for_function(f"(() => {{ const g = document.querySelector('#svg g.node.me'); return g && +g.dataset.i === {i}; }})()")
    return i


def test_page_shows_checkpoint_anomalies(open_page, zoo):
    """An NPU-compiled upload (model.safetensors holds only the embedding) is listed with the architecture's count and
    the anomaly, not the upload's 210M as if it were the model."""
    p = open_page()
    _open(p, zoo, "mobilint/EXAONE-4.0-1.2B")
    txt = p.pg.inner_text("#panel")
    assert "Checkpoint anomaly in mobilint/EXAONE-4.0-1.2B" in txt
    assert "checkpoint anomaly: NPU-compiled .mxq binary" in txt and "(architecture)" in txt


def test_page_context_with_its_source(open_page, zoo):
    p = open_page()
    _open(p, zoo, "state-spaces/mamba-130m-hf")
    assert p.pg.evaluate(ROWS_JS)["Context"].startswith("unbounded (a recurrent / window-only stack")
    _open(p, zoo, "moonshotai/Kimi-Linear-48B-A3B-Instruct")
    assert p.pg.evaluate(ROWS_JS)["Context"] == "1,048,576 (1,024K) (config: model_max_length)"


def test_page_finds_extension_repos(open_page, zoo):
    """google/gemma-4-31B-it is not in the snapshot; the candidate extension names it as a member of its shape, so a
    permalink to it opens that shape, and the card lists it under 'not in the snapshot'."""
    if not os.path.exists(os.path.join(HERE, "graph", "data", "hub_extension.json")):
        pytest.skip("graph/data/hub_extension.json not built")
    p = open_page()
    rep = "nvidia/Gemma-4-31B-IT-NVFP4"
    i = next(k for k, nd in enumerate(zoo.data["nodes"]) if rep in nd["members"])
    p.goto_hash("#m=" + enc("google/gemma-4-31B-it"))
    p.pg.wait_for_function(f"(() => {{ const g = document.querySelector('#svg g.node.me'); return g && +g.dataset.i === {i}; }})()")
    txt = p.pg.inner_text("#panel")
    assert "not in the snapshot" in txt.lower() and "google/gemma-4-31B-it" in txt
