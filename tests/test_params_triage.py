"""The 105 repos whose parameter check failed at the competitive evaluation (zoo-eval, 5d1baf4), plus three that
failed later: each one is triaged in graph/data/params_triage.json. The formula fixes in graph/arch_metrics.py
(A-fixed, and the classifier gaps corrected there, C-fixed) reproduce the Hub totals exactly, or up to named stored
blocks; the checkpoint anomalies (B) keep the architecture's figures and say so on the page; no repo fails.
"""
import json
import os

import pytest

from graph import arch_metrics as AM
from graph.model_graph import iter_json_configs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAGE = os.path.join(HERE, "graph", "data", "params_triage.json")

# fixed here, with the Hub safetensors total each must reproduce and the stored blocks that explain any difference
FIXED = {
    "jkminder/d12_optimized_286m_seed2": (286_261_730, "exact"),
    "jkminder/d16_optimized_537m_seed3": (536_871_738, "exact"),
    "jkminder/d20_optimized_897m_seed3": (896_533_746, "exact"),
    "castorini/gaggle-nanochat-pretrained-climbmix-20260924": (3_286_633_962, "exact"),
    "kingbear329/sn38-r16-c12-step687-alpha05": (2_018_511_234, "exact"),
    "sapientinc/HRM-Text-1B": (1_182_795_264, "exact"),
    "danish-foundation-models/DFM-Mimir": (1_786_775_040, "exact"),
    "jgeuter/Jeff-1.0-Large": (30_697_345_280, "exact"),
    "poolside/Laguna-XS.2": (33_442_617_088, "router bias buffers"),
    "Snowflake/snowflake-arctic-instruct": (478_584_608_768, "exact"),
    "defnoteatin/PseudoBrain-Qwen3-32B-BF16": (33_604_001_280, "exact"),
    "jdopensource/JoyAI-LLM-Flash": (49_286_094_592, "MTP layers (dense, + head copy) + router bias buffers (main layers)"),
    "zai-org/GLM-5.2": (753_329_940_480, "MTP layers + router bias buffers"),
    "meituan-longcat/LongCat-Flash-Chat": (561_862_880_256, "MTP layer (LongCat) + router bias buffers"),
    "meituan-longcat/LongCat-2.0": (1_775_560_491_136, "MTP layer (LongCat) + MTP over-tokenizer copy + router bias buffers"),
    "HuggingFaceTB/nanowhale-100m": (110_369_621, "exact"),
    "endless-frontier/Fx-Bio": (284_334_567_511, "hash-routing tid2eid buffers"),
    "MemTensor/MemT-1.0-Flash": (287_943_175_511, "hash-routing tid2eid buffers"),
    "OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated": (754_638_836_864, "engram FP8 weight_scale_inv"),
    "nvidia/Nemotron-H-4B-Base-8K": (4_489_223_040, "exact"),
    "Nanbeige/Nanbeige4.2-3B": (4_169_800_704, "exact"),
    "fla-hub/gla-1.3B-100B": (1_365_514_240, "exact"),
    "fla-hub/rwkv7-0.1B-g1": (191_034_624, "exact"),
    "summerAIO/Cyber-9B-Qwen-AllGDN-Engram": (9_174_312_960, "exact"),
    "IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct": (39_794_696_320, "exact"),
    # zoo-errors: the 30 bespoke builders (A-open), the 15 classifier gaps corrected on the arch_metrics side (C), and
    # the two repos whose Hub count only appeared at the recheck (Crowd-v1, NeuralAI-Nae1)
    "AETHORIA-AI/TR-HASH-MOE-500M-HF": (493_633_536, "token-to-expert route tables"),
    "AlgoDriveAI/Inferno-Exp-1.0": (76_163_200, "exact"),
    "AxiomicLabs/GPT-S3-3M": (2_973_666, "exact"),
    "AxiomicLabs/GPT-X3-150M": (149_400_930, "exact"),
    "BananaMind/BananaMind-2-MoE": (26_135_168, "tied head stored"),
    "Diamegs/PIT-1B-201912": (1_626_734_592, "tied head stored"),
    "IvmeLabs/Ivme-Conversate-v2-Base": (23_846_784, "exact"),
    "KBlueLeaf/TIPO-v2.1-1B-A200M": (990_785_216, "router bias buffers"),
    "Lilgarca/my-model-v01": (275_963_052, "built-in codecs (audio / image / 3-D / signal)"),
    "MemTensor/MemT-1.0-Nano": (35_111_542_048, "exact"),
    "MinimaLabs/KLM-1.5": (98_975_308, "n-gram key buffers"),
    "Pacific-i64/TR-MOE-306": (307_063_104, "rotary inv_freq buffers + token-to-expert route tables + 0-d scalars (the Hub API counts none)"),
    "ProCreations/Dory-Beam-4B": (3_975_229_440, "exact"),
    "Subject-Emu-5259/NeuralAI-Nae1": (100_112_256, "rotary inv_freq buffers"),
    "Thox-ai/ThoxOS-Kernel-27B": (27_356_728_560, "vision tower (Qwen3-VL vision encoder class)"),
    "Vxtzq/Crowd-v1": (1_007_788_032, "exact"),
    "XingChen-AGI/Xing4.0-29B-Intent": (31_215_031_088, "MTP layers (+ embedding and head copies) + router bias buffers"),
    "Yo-Purple/Meditor-1-2.2B-A500M-MoE": (2_356_466_328, "tied head stored + rotary inv_freq buffers"),
    "Yo-Purple/Meditor-2-3.4B-A1.8B-HMoE": (3_410_955_032, "rotary inv_freq buffers"),
    "Zyphra/ZAYA1-VL-8B": (9_722_232_312, "router bias buffers + vision LoRA adapters + vision tower (Qwen2.5-VL vision encoder)"),
    "amazon/GKA-primed-HQwen3-8B-Instruct": (8_500_245_504, "exact"),
    "c-bone/CrystaLLM-pi_Mattergen-XRD": (47_420_936, "exact"),
    "cybertruck32489/LFM3-1.7B-RD": (1_696_306_760, "exact"),
    "cybertruck32489/LFM3-1.7B-RD-v2": (1_958_450_760, "tied head stored"),
    "domyn/Domyn-Edge-Base": (3_928_787_968, "exact"),
    "ifx-pse-sys-ml/flame-27m-grounded": (27_125_248, "exact"),
    "inception42/Jais-3-MoVA-36B-A4B": (37_444_792_020, "exact"),
    "jakeatx/slimder-qwen38-reap384-depth40-mb4-5": (131_026_159_680, "exact"),
    "jet-ai/Jet-Nemotron-2B": (1_964_806_208, "exact"),
    "kakaocorp/kanana-2-1.3b-instruct": (1_291_478_272, "exact"),
    "microsoft/Phi-3-small-128k-instruct": (7_392_272_384, "exact"),
    "microsoft/Phi-4-mini-flash-reasoning": (3_852_562_944, "exact"),
    "nvidia/Hymba-1.5B-Base": (1_522_797_824, "exact"),
    "nvidia/Hymba-1.5B-Instruct": (1_522_797_824, "exact"),
    "openbmb/MiniCPM4-0.5B": (433_873_920, "exact"),
    "osunlp/ARLoop-0.6B": (583_138_944, "exact"),
    "osunlp/ARLoop-1.2B": (1_208_655_360, "exact"),
    "osunlp/PreNormLoop-0.6B": (582_804_096, "exact"),
    "reaperdoesntknow/GPT-X2-125M-CIx-Long-Context": (126_006_398, "exact"),
    "root-sys-n/Mima-Pico-15m": (18_683_520, "tied head stored"),
    "stepfun-ai/step3": (320_969_912_832, "vision tower (Step3 StepCLIP + projector, from config)"),
    "tardellirs/gemma3moe-16x4-270m-starter": (1_329_965_696, "exact"),
    "techdotus/Ethoric-10M": (9_965_664, "exact"),
    "thuml/sundial-base-128m": (128_329_680, "exact"),
    "whittleOrg/Whittle-Next-27B-A3B": (27_568_288_915, "n-gram index buffers (int64) + router and shared-gate tensors stored twice (two shards)"),
    "yifanyu/QwenDLM-4B-ORL": (4_486_137_856, "tied head stored"),
    "zarko1321/Bitsy500M": (509_943_936, "exact"),
}


@pytest.fixture(scope="module")
def cfgs():
    want = set(FIXED)
    return {k: v for k, v in iter_json_configs(os.path.join(HERE, "graph", "hub_10k.json")) if k in want}


@pytest.mark.parametrize("repo", sorted(FIXED))
def test_fixed_repo_matches_the_hub(cfgs, repo):
    hub, why = FIXED[repo]
    raw = cfgs[repo]
    p = AM.plan(raw)
    tot = AM.params(p)["total"]
    st = AM.hub_reason(p, raw, tot, hub, repo)
    assert st == ("hub-matched", why), (repo, tot, hub, hub - tot, st)


def _am():
    if not os.path.exists(AM.OUT):
        pytest.skip("graph/data/arch_metrics.json not built")
    with open(AM.OUT) as f:
        return json.load(f)["repos"]


def test_every_triaged_repo_is_resolved():
    with open(TRIAGE) as f:
        tri = json.load(f)
    reps = tri["repos"]
    assert len(reps) == 108
    assert set(x["bucket"] for x in reps.values()) == {"A-fixed", "B", "C-fixed"} == set(tri["buckets"])
    assert all(x["reason"] for x in reps.values())
    am = _am()
    for r, x in reps.items():
        v = am[r]["valid"]["params"]
        assert v != "failed", (r, x)
        if x["bucket"] == "B":
            assert v == "formula-only" and am[r]["params"]["checkpoint_anomaly"] == x["reason"], r
            assert any(n.startswith("checkpoint anomaly: ") for n in am[r]["notes"]), r
        else:
            assert am[r]["params"]["checkpoint_anomaly"] is None, r
    fixed = [r for r, x in reps.items() if x["bucket"] != "B"]
    assert sum(am[r]["valid"]["params"] == "hub-matched" for r in fixed) >= len(fixed) - 3
    # no failure anywhere in the zoo
    assert not [r for r, x in am.items() if "valid" in x and x["valid"]["params"] == "failed"]
    assert sum(1 for x in reps.values() if x["bucket"] == "A-fixed") == 74
    assert sum(1 for x in reps.values() if x["bucket"] == "C-fixed") == 16


def test_checkpoint_anomaly_figures_are_the_architectures(cfgs_b):
    """A B repo's numbers come from its config, not the upload: an NPU-compiled checkpoint whose model.safetensors
    holds only the embedding still shows the full LM."""
    am = _am()
    r = "mobilint/EXAONE-4.0-1.2B"
    raw = cfgs_b[r]
    assert am[r]["params"]["total"] == AM.params(AM.plan(raw))["total"] > 5 * am[r]["params"]["hub_total"]


@pytest.fixture(scope="module")
def cfgs_b():
    want = {"mobilint/EXAONE-4.0-1.2B", "whittleOrg/Whittle-Next-27B-A3B", "jakeatx/slimder-qwen38-reap384-depth40-mb4-5"}
    return {k: v for k, v in iter_json_configs(os.path.join(HERE, "graph", "hub_10k.json")) if k in want}


def test_bespoke_predicates_fire_on_their_families_only():
    """Every bespoke builder fires on its own model types and nothing else in the snapshot or the candidate extension
    (a builder written for one remote family must not rewrite another's plan); each repo is rebuilt by at most one."""
    seen = {}
    saved = AM.BESPOKE
    cfgs = list(iter_json_configs(os.path.join(HERE, "graph", "hub_10k.json")))
    ext = os.path.join(HERE, "graph", "data", "hub_extension.json")
    if os.path.exists(ext):                                 # the candidate extension's families too
        with open(ext) as f:
            cfgs += [(r, x["config"]) for r, x in json.load(f)["repos"].items()]
    for name, raw in cfgs:
        AM.BESPOKE = ()                                     # the plan the dispatcher sees: before any builder
        try:
            p = AM.plan(raw)
        except Exception:
            continue
        finally:
            AM.BESPOKE = saved
        d, _ = AM.resolve(raw)
        hits = [fn.__name__ for test, fn in AM.BESPOKE if test(p, d, raw)]
        assert len(hits) <= 1, (name, hits)
        for h in hits:
            seen.setdefault(h, set()).add(p["mt"])
    assert len(seen) == len(AM.BESPOKE)                     # each builder is used
    for fn, mts in seen.items():
        assert len(mts) <= 2, (fn, mts)                     # a family and its text / wrapper type at most


def test_classifier_gap_corrections_are_idempotent():
    """The C-fixed builders SET what the code builds: run twice, the plan's counts do not move (so the zoo is unchanged
    once round 3 fixes graph/model_graph.py)."""
    import copy
    with open(TRIAGE) as f:
        reps = [r for r, x in json.load(f)["repos"].items() if x["bucket"] == "C-fixed" and "arch_metrics" in
                x.get("fixed_in", "")]
    cfgs = {k: v for k, v in iter_json_configs(os.path.join(HERE, "graph", "hub_10k.json")) if k in set(reps)}
    assert len(cfgs) == 15
    for r, raw in cfgs.items():
        p = AM.plan(raw)
        d, _ = AM.resolve(raw)
        cl = AM.classify(raw) or {}
        fn = next(fn for test, fn in AM.BESPOKE if test(p, d, raw))
        once = AM.params(p)
        q = copy.deepcopy(p)
        fn(q, d, raw, cl)
        assert AM.params(q) == once, r


def test_duplicated_shard_tensors_are_named(cfgs_b):
    """Whittle-Next's root shards store the 40 routers and shared-expert gates twice: that block is named for that
    checkpoint only, and the same builder on slimder (same family, a clean upload) is exact."""
    raw = cfgs_b["whittleOrg/Whittle-Next-27B-A3B"]
    p = AM.plan(raw)
    comp = AM.hub_components(p, raw, repo="whittleOrg/Whittle-Next-27B-A3B")
    assert comp["router and shared-gate tensors stored twice (two shards)"] == 40 * (180 + 1) * 2048
    assert "router and shared-gate tensors stored twice (two shards)" not in AM.hub_components(p, raw)


def test_dsa_indexer_only_on_full_layers(cfgs):
    """GLM-5.x indexer_types: only the "full" layers hold an indexer and cache its keys."""
    raw = cfgs["zai-org/GLM-5.2"]
    m = AM.metrics(raw)
    it = raw["indexer_types"]
    assert m["layers"]["dsa_indexer"] == sum(1 for x in it if x == "full") == 21
    assert m["kv_per_token"]["elems"]["dsa_indexer"] == 21 * raw["index_head_dim"]


def test_longcat_layer_structure(cfgs):
    p = AM.plan(cfgs["meituan-longcat/LongCat-Flash-Chat"])
    A, B, C = p["groups"]
    assert A["n"] == B["n"] == C["n"] == 28 and A["mla"] and B["mla"] and C["attn"] is None
    assert B["moe"]["E"] == 512 and B["moe"]["zero"] == 256 and A["ffn"]["inter"] == C["ffn"]["inter"] == 12288
