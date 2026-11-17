"""The candidate snapshot extension (graph/snapshot_extend.py -> graph/data/hub_extension.json): the selection rules
on synthetic inputs, and the written file against the snapshot and the zoo. Offline (no network)."""
import json
import os

import pytest

from graph import snapshot_extend as S

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(HERE, "graph", "data", "hub_extension.json")


# ------------------------------------------------------------------------------------------------ selection rules

SNAP = [{"id": "nvidia/Gemma-4-31B-IT-NVFP4", "downloads": 50_000},
        {"id": "acme/kept-model", "downloads": 9_000},
        {"id": "acme/small", "downloads": 1_000}]


def test_cutoff_is_the_snapshots_smallest():
    assert S.snapshot_cutoff(SNAP) == 1_000


def test_original_publisher_shadows_a_derivative():
    pool = [{"id": "google/gemma-4-31B-it", "downloads": 6_500_000}]
    (r, k, sh), = S.select(pool, SNAP)
    assert r["id"] == "google/gemma-4-31B-it" and k == "gemma-4-31b-it" and sh == "nvidia/Gemma-4-31B-IT-NVFP4"


def test_snapshot_holder_keeps_a_key_it_wins():
    pool = [{"id": "other/kept-model-AWQ", "downloads": 2_000}]   # same key, fewer downloads than the holder
    assert S.select(pool, SNAP) == []


def test_cutoff_junk_snapshot_members_and_pool_dedup():
    pool = [{"id": "acme/small", "downloads": 10**6},                 # already in the snapshot
            {"id": "x/tiny-random-llama", "downloads": 10**6},          # junk
            {"id": "x/below", "downloads": 999},                         # below the cutoff
            {"id": "a/NewModel-8B", "downloads": 5_000},
            {"id": "b/NewModel-8B-GGUF", "downloads": 7_000},           # same key, more downloads: wins
            {"id": "b/NewModel-8B-GGUF", "downloads": 7_000}]           # listed twice (tag + publisher)
    got = S.select(pool, SNAP)
    assert [(r["id"], k, sh) for r, k, sh in got] == [("b/NewModel-8B-GGUF", "newmodel-8b", None)]


def test_original_publisher_kept_beside_a_pool_derivative():
    pool = [{"id": "gaunernst/gemma-3-27b-it-int4-awq", "downloads": 800_000},
            {"id": "google/gemma-3-27b-it", "downloads": 335_000},
            {"id": "unsloth/gemma-3-27b-it-GGUF", "downloads": 50_000}]
    assert [r["id"] for r, k, sh in S.select(pool, SNAP)] == ["gaunernst/gemma-3-27b-it-int4-awq"]
    rk = S.ranked(pool, SNAP)["gemma-3-27b-it"][0]          # the fall-backs when the winner has no config.json
    assert [r["id"] for r in rk] == [x["id"] for x in pool]
    (r, k, sh), = S.originals(pool, SNAP)
    assert (r["id"], sh) == ("google/gemma-3-27b-it", "gaunernst/gemma-3-27b-it-int4-awq")
    assert S.is_original("google/gemma-3-27b-it") and not S.is_original("google/gemma-3-27b-it-qat-q4_0-gguf")


def test_text_lm_filter():
    llama = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 256,
             "num_attention_heads": 4, "num_hidden_layers": 2, "intermediate_size": 512, "vocab_size": 1000,
             "max_position_embeddings": 2048}
    assert S.text_lm(llama, "a/b") == (True, None)
    assert S.text_lm(None, "a/b")[0] is False
    assert S.text_lm(dict(llama, architectures=["LlamaForSequenceClassification"]), "a/b")[0] is False
    assert S.text_lm(llama, "a/b", "feature-extraction")[0] is False
    assert S.text_lm({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}, "a/b")[0] is False


# ------------------------------------------------------------------------------------------------ the written file

@pytest.fixture(scope="module")
def ext():
    if not os.path.exists(EXT):
        pytest.skip("graph/data/hub_extension.json not built")
    with open(EXT) as f:
        return json.load(f)


def test_criteria_applied(ext):
    snap = json.load(open(S.SNAP))
    ids = {x["id"] for x in snap}
    cut = ext["criteria"]["downloads_cutoff"]
    assert cut == S.snapshot_cutoff(snap)
    from graph.fetch_configs import JUNK, unique_key
    keys = {}
    for nm, e in ext["repos"].items():
        assert nm not in ids
        assert e["downloads"] >= cut
        assert not any(j in nm.lower() for j in JUNK)
        assert e["key"] == unique_key(nm)
        if e["original_beside_derivative"]:                 # the publisher's repo beside the derivative winner
            assert S.is_original(nm) and e["shadows"] and not S.is_original(e["shadows"])
            assert unique_key(e["shadows"]) == e["key"]
            continue
        assert e["key"] not in keys, (nm, keys.get(e["key"]))
        keys[e["key"]] = nm
        if e["shadows"]:
            h = next(x for x in snap if x["id"] == e["shadows"])
            assert unique_key(h["id"]) == e["key"] and h["downloads"] < e["downloads"]
    assert sum(e["original_beside_derivative"] for e in ext["repos"].values()) == \
        ext["counts"]["originals_beside_a_derivative"]


def test_every_repo_is_a_text_lm(ext):
    for nm, e in ext["repos"].items():
        assert S.text_lm(e["config"], nm, e["pipeline_tag"]) == (True, None), nm


def test_shapes_point_at_zoo_representatives(ext):
    from graph import arch_metrics as AM
    with open(AM.OUT) as f:
        reps = {s["rep"] for s in json.load(f)["shapes"]}
    for nm, e in ext["repos"].items():
        assert e["new_shape"] == (e["joins_shape"] is None)
        if e["joins_shape"] is not None:
            assert e["joins_shape"] in reps, nm
            assert isinstance(e["frameworks"]["backdoor_shape"], int)
    assert ext["counts"]["repos_in_new_shapes"] + ext["counts"]["repos_joining_zoo_shapes"] == len(ext["repos"])


def test_metrics_and_frameworks_present(ext):
    for nm, e in ext["repos"].items():
        assert e["metrics"] is not None, (nm, e.get("metrics_error"))
        assert e["metrics"]["params"]["total"] > 0
        assert set(e["frameworks"]["native"]) >= {"transformers", "vllm", "llama.cpp"}
        assert e["valid_params"] in ("hub-matched", "formula-only", "failed")


# the 17 checks that failed at the first run, triaged: builders (Hub-exact or exact up to named blocks), quantised
# uploads the markers missed, and checkpoint anomalies (EXT_ANOMALIES); recomputed from the stored configs
EXT_FIXED = {
    "thinkingmachines/Inkling": (952_377_623_626, "MTP layers (Inkling, dense blocks) + vision encoder (HMLP) + "
                                                  "audio embedding (dMel)"),
    "thinkingmachines/Inkling-Small": (265_956_439_090, "MTP layers (Inkling, dense blocks) + vision encoder (HMLP) + "
                                                        "audio embedding (dMel)"),
    "mistralai/Mamba-Codestral-7B-v0.1": (7_285_403_648, "exact"),
    "inclusionAI/LLaDA-MoE-7B-A1B-Instruct": (7_356_880_896, "exact"),
    "adept/fuyu-8b": (9_408_238_592, "vision patch projection (vision_embed_tokens)"),
    "inclusionAI/Ling-3.0-tiny-base": (8_209_997_600, "MTP layers + router bias buffers"),
    "snu-aidas/Dynin-Omni": (8_116_244_480, "exact"),
    "internlm/Intern-S2-Mobius": (35_964_405_616, "MTP layer (256 own experts, not run by the model code) + vision "
                                                  "tower (Qwen3-VL vision encoder class)"),
}
EXT_QUANT = ("sanasol2008/Qwen3.8-Flash-Next-Abliterated-Sushi-2bpw", "Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp",
             "Jundot/DeepSeek-V4.1-Flash-oQ2.5e-mtp")


@pytest.mark.parametrize("repo", sorted(EXT_FIXED))
def test_extension_builders_match_the_hub(ext, repo):
    from graph import arch_metrics as AM
    hub, why = EXT_FIXED[repo]
    raw = ext["repos"][repo]["config"]
    p = AM.plan(raw)
    assert AM.hub_reason(p, raw, AM.params(p)["total"], hub, repo) == ("hub-matched", why)


def test_extension_has_no_failed_check(ext):
    """No params check fails on the extension: the quantised uploads the markers missed are recognised, and the
    checkpoint anomalies carry their evidence (formula-only, the architecture's figures)."""
    from graph import arch_metrics as AM
    R = ext["repos"]
    assert not [r for r, e in R.items() if e["valid_params"] == "failed"]
    for r in EXT_QUANT:
        assert AM.is_quantised(r, R[r]["config"]) and R[r]["hub_reason"].startswith("quantised checkpoint"), r
    for r, why in S.EXT_ANOMALIES.items():
        assert R[r]["valid_params"] == "formula-only" and R[r]["checkpoint_anomaly"] == why, r
    # Nemotron-3.5-Bolt: the Puzzle blocks' own expert counts; the LM count is the language_model.* headers less the
    # 13,824 router-bias buffers (the RADIO vision tower is not countable from the config)
    assert R["nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16"]["metrics"]["params"]["total"] == 74_042_706_944


def test_flagships_covered(ext):
    f = ext["flagships"]
    assert "google/gemma-4-31B-it" in f["google/gemma-4-"]
    assert any(x.startswith("mistralai/Mixtral-8x7B") for x in f["mistralai/Mixtral-"])
