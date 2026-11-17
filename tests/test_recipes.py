"""The recipes (recipes/make_recipe.py): framework references, the tef-ucore export, generated scripts run end
to end in each framework's own venv, and graph/data/recipes.json.

The end-to-end tests need the per-framework venvs ($TEF_RECIPES_VENVS, default $TEF_SCRATCH/recipes-venvs/<fw>;
each is skipped when missing) and the checkpoints in the HF cache (skipped when not cached).

Run:  pytest tests/test_recipes.py -q -n 2
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "recipes", "tef_recipes"))

from recipes.make_recipe import FRAMEWORKS, OUT_JSON, emit, plan  # noqa: E402

F64 = torch.float64


def _cached(repo):
    from recipes.common.runtime import _cached as c
    return c(repo, "config.json") is not None


# --- vendoring, export, specs (in this environment) --------------------------------------------------

def test_vendored_copies_are_current():
    from recipes.vendor import main
    assert main(["--check"]) == 0, "python -m recipes.vendor"


def test_vendored_graph_parts_match_the_repo():
    """tef_recipes/graph_parts.py holds the graph definitions the construction reads, equal to the repo's; any
    other graph name raises ImportError (the relation gate is not shipped), and no vendored file imports a
    repository package."""
    from graph import model_graph as MG
    from recipes import vendor
    from tef_recipes import graph_parts as G
    assert G.ROPE_PAIRS_FREE_TYPES == MG.ROPE_PAIRS_FREE_TYPES
    for args in [(4, 3, 2, 1), (2, 7, 5, 3), (8, 4, 16, 4), (3, 5, 4, 2), (0, 1, 1, 1)]:
        assert G.kv_hosts_needed(*args) == MG.kv_hosts_needed(*args), args
    with pytest.raises(ImportError, match="does not ship"):
        from tef_recipes.graph_parts import edge_plan  # noqa: F401
    for f in os.listdir(vendor.PKG):
        if f.endswith(".py"):
            assert not vendor.repo_imports(open(os.path.join(vendor.PKG, f)).read()), f


STANDALONE = r"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"graph", "elephantine", "hf", "attn", "recipes"}:
            raise ModuleNotFoundError("not in a pip install: " + name)
sys.meta_path.insert(0, Block())
from tef_recipes import emulate as E
T = dict(model_type="qwen2", architectures=["Qwen2ForCausalLM"], vocab_size=151936, hidden_size=896,
         intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
         rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, max_position_embeddings=32768)
H = dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=49152, hidden_size=960,
         intermediate_size=2560, num_hidden_layers=32, num_attention_heads=15, num_key_value_heads=5,
         rope_theta=100000.0, rms_norm_eps=1e-5, tie_word_embeddings=True, max_position_embeddings=8192)
try:
    con = E.plan_construction(T, H)
    print("built", con.units)
except E.Refused as e:
    print("refused", e.code, e)
small = dict(H, num_attention_heads=2, num_key_value_heads=1, hidden_size=128)
try:
    E.plan_construction(T, small)
    print("built-small")
except E.Refused as e:
    print("refused-small", e.code, e)
"""


def test_vendored_construction_runs_without_the_repository(tmp_path):
    """The review's case (Qwen2.5-0.5B's config -> SmolLM2-360M's): with the repository's packages unimportable,
    as in a pip install of recipes/tef_recipes, plan_construction builds or refuses -- it never reaches for
    graph (the KV-head and longrope-table checks read tef_recipes.graph_parts)."""
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "recipes", "tef_recipes"))
    r = subprocess.run([sys.executable, "-c", STANDALONE], cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    lines = r.stdout.strip().splitlines()
    assert lines[0].startswith(("built", "refused")), r.stdout
    assert lines[-1].startswith("refused-small 4") and "host KV groups" in lines[-1], r.stdout


@pytest.mark.parametrize("rope", [None, {"rope_type": "linear", "factor": 2.0},
                                  {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                   "high_freq_factor": 4.0, "original_max_position_embeddings": 16},
                                  {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}])
def test_rope_tables_match_transformers(rope):
    """llama_spec's RoPE table (no transformers at run time) is transformers' own, bit for bit."""
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from recipes.common.llama_spec import rope_table
    kw = dict(hidden_size=64, num_attention_heads=4, max_position_embeddings=128, rope_theta=10000.0)
    cfg = LlamaConfig(**kw, rope_scaling=rope)
    typ = (rope or {}).get("rope_type", "default")
    inv, att = ROPE_INIT_FUNCTIONS[typ](cfg, "cpu")
    raw = dict(kw, model_type="llama", rope_scaling=rope)
    mine, matt, rot = rope_table(raw, 16)
    assert rot == 16 and np.array_equal(mine, inv.numpy()), (mine, inv)
    assert abs(matt - att) < 1e-12


def _tiny_host(case):
    from tests.test_stock_loaders import CASES, _build
    import elephantine.universal as U
    cfg_name, model_name, loader, kw, opt = CASES[case]
    return getattr(U, loader)(_build(cfg_name, model_name, kw, opt)).eval()


@pytest.mark.parametrize("case", ["gpt_bigcode-mqa", "gemma2-softcaps", "falcon_h1-gate-norm", "deepseek_v3"])
def test_ucore_export_runs_in_plain_torch(case, tmp_path):
    """export -> recipes/frameworks/pytorch/universal.py (in this env): the host's logits, prefill and decode."""
    from tef_recipes.ucore_export import save
    from recipes.frameworks.pytorch.universal import load_model
    host = _tiny_host(case)
    save(str(tmp_path), host)
    m = load_model(str(tmp_path))
    ids = [3, 14, 15, 92, 65, 35, 89, 79]
    with torch.no_grad():
        ref = host(torch.tensor([ids]))[0].numpy()
    got = m.prefill_logits(ids)
    assert np.abs(got - ref).max() < 1e-4 * max(1.0, np.abs(ref).max())
    cache = m.new_cache(len(ids))
    m.step(ids[:5], cache, 0)
    last = [m.step([t], cache, 5 + j)[-1] for j, t in enumerate(ids[5:])]   # KV / SSM cache == full forward
    assert np.abs(np.stack(last) - ref[5:]).max() < 1e-4 * max(1.0, np.abs(ref).max())


def test_ucore_export_refuses_branches_outside_the_core():
    from tef_recipes.ucore_export import Unsupported, export
    host = _tiny_host("phimoe-sparsemixer")
    with pytest.raises(Unsupported, match="SparseMixer"):
        export(host)
    host = _tiny_host("llama-biases")
    with torch.no_grad():
        host.layers[1].attn.dl_c.fill_(0.5)                 # differential heads switched on
    with pytest.raises(Unsupported, match="differential"):
        export(host)


def test_ucore_export_prunes_only_zero_sub_blocks():
    from tef_recipes.ucore_export import export
    cfg, t = export(_tiny_host("llama-biases"))
    assert all(L["ssm"] is None for L in cfg["layers"])              # a softmax model: the SSM is off
    assert all(L["attn"] is not None and L["ffn"]["shared"] is not None for L in cfg["layers"])


LLAMA_CFG = dict(model_type="llama", vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3,
                 num_attention_heads=6, num_key_value_heads=2)


@pytest.mark.parametrize("fw", FRAMEWORKS)
@pytest.mark.parametrize("route", ["native", "universal"])
def test_emitted_script_is_self_contained(fw, route):
    """A generated script imports nothing from this repository (its build stage imports tef_recipes, the pip
    package) and compiles under the oldest Python a framework needs (3.8 for DyNet: no 3.9+ syntax)."""
    import ast
    import re
    p = plan("x/tiny-llama", fw, route=route, cfg=dict(LLAMA_CFG))
    text = emit("x/tiny-llama", fw, p)
    bad = re.findall(r"^\s*(?:from|import)\s+(?:recipes|elephantine|graph|attn|hf)\b.*$", text, re.M)
    assert not bad, bad
    ast.parse(text, feature_version=(3, 8))
    assert "# requirements-run:" in text and "# requirements-build:" in text


# --- end to end in each framework's venv ---------------------------------------------------------------

E2E = [  # (validation id, target, route, host)
    ("native/HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M", None, None),
    ("universal/EleutherAI/pythia-14m", "EleutherAI/pythia-14m", None, None),
]
if os.environ.get("TEF_RECIPES_SLOW"):
    E2E += [("universal/openai-community/gpt2", "openai-community/gpt2", None, None),
            ("exact-host/HuggingFaceTB/SmolLM2-135M@HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-135M",
             "exact-host", "HuggingFaceTB/SmolLM2-360M")]


@pytest.mark.parametrize("fw", FRAMEWORKS)
@pytest.mark.parametrize("vid,target,route,host", E2E, ids=[e[0] for e in E2E])
def test_generated_script_end_to_end(fw, vid, target, route, host):
    from recipes import validate as V
    if V.venv_python(fw) is None:
        pytest.skip(f"no {fw} venv under {V.VENVS}")
    if not _cached(target) or (host and not _cached(host)):
        pytest.skip(f"{target} not in the HF cache")
    v = {"entries": {}, "frameworks": {}}
    e = V.run_one(fw, vid, target, route, host, v, force=True, save=False)
    assert "error" not in e, e.get("error")
    assert e["greedy_identical"], e
    assert e["pass"] or e["pass_floor"], e


# --- recipes.json ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recipes():
    if not os.path.isfile(OUT_JSON):
        pytest.skip("graph/data/recipes.json not built (python -m recipes.make_recipe --index)")
    return json.load(open(OUT_JSON))


def test_every_route_kind_has_an_end_to_end_tested_instance(recipes):
    """The meta-test: each route kind any model uses has, in some framework, a real checkpoint that went through
    its generated script end to end and passed."""
    used = {(fw, e["route"]) for m in recipes["models"].values() for fw, e in m["frameworks"].items() if e["route"]}
    kinds = {r for _, r in used}
    tested = {e["route"] for e in recipes["validation"] if e.get("kind") == "checkpoint"
              and (e.get("pass") or e.get("pass_floor"))}
    assert kinds <= tested, f"route kinds without an end-to-end tested instance: {sorted(kinds - tested)}"
    for fw in FRAMEWORKS:                                 # ... and per framework, for every kind it is offered
        mine = {r for f, r in used if f == fw}
        t = {e["route"] for e in recipes["validation"] if e["framework"] == fw and e.get("kind") == "checkpoint"
             and (e.get("pass") or e.get("pass_floor"))}
        if recipes["frameworks"][fw].get("version"):
            assert mine <= t, (fw, sorted(mine - t))


def test_recipes_json_is_consistent(recipes):
    ids = {e["id"] for e in recipes["validation"]}
    for repo, m in recipes["models"].items():
        for fw, e in m["frameworks"].items():
            assert e["route"] in (None, "native", "exact-host", "universal"), (repo, e)
            assert e["status"] in (None, "e2e", "family", "untested"), (repo, e)
            assert (e["route"] is None) == (e["status"] is None), (repo, fw, e)
            if e["status"] == "e2e":
                assert e["validated_by"] in ids
            if e["script"]:
                assert os.path.isfile(os.path.join(ROOT, e["script"])), e["script"]
    for fw, c in recipes["counts"].items():
        assert c["models"] == len(recipes["models"])
        assert c.get("native", 0) + c.get("exact-host", 0) + c.get("universal", 0) + c.get("none", 0) == c["models"]
        assert c["tested_route"] == c.get("e2e", 0) + c.get("family", 0)


def test_checked_in_scripts_are_current(recipes):
    """Every checked-in script is what make_recipe writes today (the references are inlined)."""
    for e in recipes["validation"]:
        if e.get("kind") != "checkpoint" or not e.get("script"):
            continue
        path = os.path.join(ROOT, e["script"])
        text = open(path).read()
        d = eval(text.split("DEFAULTS = ", 1)[1].split("\n", 1)[0])
        p = {"route": d["route"], "host": d["host"], "loader": d["loader"]}
        assert emit(d["target"], d["framework"], p) == text, f"{e['script']} is stale: python -m recipes.validate"


# --- the recipe command line's error paths (no network: urlopen is stubbed) -----------------------------

@pytest.mark.parametrize("code,want", [(401, "gated or private"), (404, "gated or private"), (500, "failed")])
def test_hub_file_http_errors_are_messages(code, want, tmp_path, monkeypatch):
    """A missing / gated repo ends with one line naming HF_TOKEN, not an HTTPError traceback; no .part left."""
    import io
    import urllib.error
    import urllib.request
    from recipes.common import runtime as R
    monkeypatch.setattr(R, "_cached", lambda *a, **k: None)

    def fail(req):
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b""))
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(SystemExit) as e:
        R.hub_file("nobody/nothing", "config.json", str(tmp_path))
    assert want in str(e.value) and ("HF_TOKEN" in str(e.value)) == (code != 500)
    assert not any(p.name.endswith(".part") for p in tmp_path.rglob("*"))
    if code == 404:                                         # an optional file that is missing is just None
        assert R.hub_file("nobody/nothing", "x.json", str(tmp_path), optional=True) is None


def test_run_cli_error_paths(tmp_path, capsys):
    from recipes.common.runtime import run_cli
    with pytest.raises(SystemExit, match="not a run directory"):
        run_cli(lambda d: None, "pytorch", ["run", "--dir", str(tmp_path / "missing")])
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        run_cli(lambda d: None, "pytorch", ["run", "--dir", str(tmp_path), "--ids", "1,x"])
    assert e.value.code == 2 and "comma-separated integers" in capsys.readouterr().err


def test_make_recipe_host_needs_exact_host_route(capsys):
    from recipes.make_recipe import main
    with pytest.raises(SystemExit) as e:
        main(["x/tiny", "--framework", "pytorch", "--host", "y/host", "--route", "native"])
    assert e.value.code == 2 and "--host only applies" in capsys.readouterr().err

