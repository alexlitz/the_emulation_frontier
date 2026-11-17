"""Rescue rules for dense >= MoE and looped >= untied (graph/rescue_rules.py).

1. the verdicts on configs: E = 1 exact, top-1 unit gate payable, top-k / no renormalisation / a gated
   shared expert walls; plain looped families payable, the rest walls;
2. the constructions on tiny random models (fp32, CPU): the pre-norm dense host reproduces a top-1 MoE
   and the oracle a top-2 one, the unit-scale carry and the top-1 masks on a top-2 target do not; the
   looped host (counter + start-token marker) reproduces an untied stack, naive looping does not;
3. the recorded real-model rows (moe/data/dense_host_prenorm.jsonl, hf/data/looped_host_prenorm.jsonl)
   agree with the verdicts under the payable standard;
4. apply() only touches the two slices.

Run:  python -m pytest tests/test_rescue.py -q
"""
import json
import os

import pytest
import torch

import graph.rescue_rules as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOE_DATA = os.path.join(ROOT, "moe/data/dense_host_prenorm.jsonl")
LOOP_DATA = os.path.join(ROOT, "hf/data/looped_host_prenorm.jsonl")
SHARP = (R.MOE_MASK_STRENGTH, R.MOE_SELECT_BETA)


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- 1. verdicts ------------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg, verdict", [
    (dict(model_type="mixtral", num_local_experts=1, num_experts_per_tok=1), "exact"),
    (dict(model_type="jamba", num_experts=1, num_experts_per_tok=1), "exact"),
    (dict(model_type="qwen3_moe", num_experts=2, num_experts_per_tok=1, norm_topk_prob=True), "payable"),
    (dict(model_type="mixtral", num_local_experts=8, num_experts_per_tok=1), "payable"),
    (dict(model_type="mixtral", num_local_experts=8, num_experts_per_tok=2), "wall"),
    (dict(model_type="olmoe", num_experts=64, num_experts_per_tok=1, norm_topk_prob=False), "wall"),
    (dict(model_type="qwen2_moe", num_experts=4, num_experts_per_tok=1, norm_topk_prob=True,
          shared_expert_intermediate_size=8960), "wall"),
    (dict(model_type="deepseek_v2", n_routed_experts=8, num_experts_per_tok=1, norm_topk_prob=True,
          n_shared_experts=2), "payable"),
    (dict(model_type="custom_moe", num_experts=8), "wall"),
])
def test_dense_moe_verdict(cfg, verdict):
    assert R.dense_moe_verdict(cfg)[0] == verdict


def test_dense_moe_cost():
    assert R.dense_moe_cost(dict(model_type="mixtral", num_local_experts=1, num_experts_per_tok=1)) == \
        dict(depth=1, channels=0, units=0)
    c = R.dense_moe_cost(dict(model_type="qwen3_moe", num_experts=4, num_experts_per_tok=1, norm_topk_prob=True))
    assert c == dict(depth=R.MOE_DENSE_DEPTH, channels=4, units=2 * 4 * 3)
    assert R.dense_moe_cost(dict(model_type="mixtral", num_local_experts=8, num_experts_per_tok=2)) is None
    assert not R.DENSE_MOE_TOPK_PAYABLE and R.DENSE_MOE_TOP1_PAYABLE


def test_looped_verdict():
    assert R.looped_untied_verdict(dict(model_type="nanbeige"))[0] == "payable"
    for mt in ("iquestloopcoder", "lfm3_rd", "arloop", "kotodama", "dory", "arloop_baseline", "qwen3loop",
               "never_seen"):
        assert R.looped_untied_verdict(dict(model_type=mt))[0] == "wall", mt
    assert not set(R.LOOP_PLAIN_TYPES) & set(R.LOOP_NOT_COVERED)
    assert R.loop_units(576) == 1154 and R.LOOP_EXTRA_CHANNELS == 2


def test_passes():
    assert R.passes(dict(top1=0.995, ex32=0.92)) and not R.passes(dict(top1=0.98, ex32=1.0))
    assert not R.passes(dict(top1=1.0, ex32=0.875))


# --- 2. constructions on tiny random models ---------------------------------------------------------
def _tiny_mixtral(k, seed=0):
    from transformers import MixtralConfig, MixtralForCausalLM
    torch.manual_seed(seed)
    cfg = MixtralConfig(hidden_size=48, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=4, num_local_experts=4, num_experts_per_tok=k, vocab_size=64,
                        max_position_embeddings=64, initializer_range=0.3, rms_norm_eps=1e-5)
    return MixtralForCausalLM(cfg).eval()


@torch.no_grad()
def _moe_err(model, variant, k, lam=SHARP[0], beta=SHARP[1]):
    import moe.dense_host_prenorm as D
    ids = torch.randint(0, 64, (1, 40), generator=torch.Generator().manual_seed(1))
    ref = model(ids).logits
    blocks = D.moe_blocks(model)
    inst = D.Installed(model, blocks, variant, k, lam=lam, beta=beta)
    try:
        out = model(ids).logits
    finally:
        inst.remove()
    return ((out - ref).abs().max() / ref.abs().max()).item()


def test_dense_host_top1_exact_under_prenorm():
    m = _tiny_mixtral(1)
    assert _moe_err(m, "top1", 1) < 1e-5                      # the pre-norm construction
    assert _moe_err(m, "top1-carry1", 1) > 1e-2               # the norm-free carry scale fails
    assert _moe_err(m, "top1-noerase", 1) > 1e-2              # so does leaving the carry in place


def test_dense_host_topk_needs_the_oracle():
    m = _tiny_mixtral(2)
    assert _moe_err(m, "oracle", 2) < 1e-5                    # membership + gates given: exact
    assert _moe_err(m, "top1", 2) > 1e-2                      # top-1 masks on a top-2 target
    assert _moe_err(m, "uniform", 2) > 1e-2                   # membership alone, gate 1/k


def _tiny_llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=64, max_position_embeddings=64, initializer_range=0.2,
                      rms_norm_eps=1e-5, bos_token_id=1)
    return LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def test_looped_host_exact_under_prenorm():
    import hf.looped_host_prenorm as LH
    m = _tiny_llama()
    net = LH.Llama(m)
    ids = torch.cat([torch.tensor([[1]]), torch.randint(2, 64, (1, 30), generator=torch.Generator().manual_seed(2))], 1)
    tgt = net.forward(ids)
    assert (tgt - m(ids).logits[0]).abs().max() < 1e-4             # own forward == transformers
    scale = tgt[1:].abs().max()
    err = lambda name, K: ((net.forward(ids, K=K, v=LH.VARIANTS[name], bos_id=1) - tgt)[1:].abs().max() / scale).item()
    for K in (2, 3):
        assert err("host", K) < 1e-4                                # every position but the start token
        assert err("host-internorm", K) < 1e-3                     # a norm between loops, refreshed counter
        assert err("naive", K) > 1e-2
        assert err("ffn-only", K) > 1e-3
    assert err("host", 1) < 1e-6


@torch.no_grad()
def test_looped_host_split_and_value_bias():
    import hf.looped_host_prenorm as LH
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=64, max_position_embeddings=64, initializer_range=0.2,
                      rms_norm_eps=1e-6, use_sliding_window=False)
    m = Qwen2ForCausalLM(cfg).eval()                                # q / k / v biases
    with torch.no_grad():
        for layer in m.model.layers:
            layer.self_attn.v_proj.bias.normal_(0, 0.5)
    net = LH.Llama(m)
    ids = torch.cat([torch.tensor([[1]]), torch.randint(2, 64, (1, 30), generator=torch.Generator().manual_seed(3))], 1)
    tgt = net.forward(ids)
    scale = tgt[1:].abs().max()
    err = lambda name, K: ((net.forward(ids, K=K, v=LH.VARIANTS[name], bos_id=1) - tgt)[1:].abs().max() / scale).item()
    for K in (2, 3):
        assert err("host", K) < 1e-4                                # the value bias cancelled from iteration 2
        assert err("host-split", K) < 1e-4                          # prelude / loop / coda, coda counter columns
        assert err("host-split-noerase", K) > 1e-3


# --- 3. recorded rows ---------------------------------------------------------------------------------
def test_recorded_dense_moe_rows():
    allr = rows(MOE_DATA)
    rs = [r for r in allr if r.get("refs", "fp32") == "fp32"]       # generated and scored in fp32
    assert all(r["top1"] == 1.0 and r["kl"] == 0.0 for r in rs if r["variant"] == "self")
    sharp = [r for r in rs if (r["lam"], r["beta"]) == SHARP]
    # top-1 targets (native or forced): the construction passes at the sharp constants
    top1 = [r for r in sharp if r["variant"] == "top1" and r["k"] == 1]
    assert top1 and all(R.passes(r) for r in top1)
    # a top-k target is not reproduced by any realizable variant; the oracle passes
    topk = [r for r in sharp if r["k"] >= 2]
    assert topk and not any(R.passes(r) for r in topk if r["variant"] in ("top1", "uniform")
                            and r["model"] != "ggml-org/stories15M_MOE")      # stories15M: identical experts
    assert all(R.passes(r) for r in topk if r["variant"] == "oracle")
    # the norm-free carry scale and the missing erase fail everywhere
    assert not any(R.passes(r) for r in rs if r["variant"] in ("top1-carry1", "top1-noerase"))
    # at the moderate constants the soft zone (~1e-2) already breaks a real top-1 target
    mod = [r for r in rs if r["variant"] == "top1" and r["k"] == 1 and r["lam"] < SHARP[0]
           and r["model"] == "huihui-ai/Huihui-MoE-0.8B-2E"]
    assert mod and not any(R.passes(r) for r in mod)
    # the gate as an L-level staircase (oracle membership): 16 levels fail, 64 pass on gpt-oss top-2
    stair = {r["variant"]: r for r in sharp if r["variant"].startswith("stair") and r["model"].endswith("gpt-oss-tinystories")}
    assert not R.passes(stair["stair16"]) and R.passes(stair["stair64"])
    # teacher-forced only (bf16-generated references, fp32 scored): OLMoE, top-8 of 64 WITHOUT
    # renormalisation. The oracle passes; the unit-gate masks fail at top-8 and also forced to top-1,
    # where the gate is p_max(x), not 1 (the third factor: dense_moe_verdict walls it)
    tf = [r for r in allr if r.get("refs", "fp32") != "fp32" and r["variant"] != "self"]
    assert tf
    for r in tf:
        if r["variant"] == "oracle":
            assert r["top1"] >= R.PAYABLE_TOP1, r
        if r["variant"] == "top1":
            assert r["top1"] < 0.5, r
    assert R.dense_moe_verdict(dict(model_type="olmoe", num_experts=64, num_experts_per_tok=1,
                                    norm_topk_prob=False))[0] == "wall"


def test_recorded_loop_rows():
    rs = rows(LOOP_DATA)
    for r in rs:
        if r["variant"] in ("host", "host-internorm", "host-split", "big-carry"):
            assert R.passes(r), r
        if r["variant"] in ("naive", "naive-internorm", "ffn-only", "host-internorm-norefresh"):
            assert not R.passes(r), r
        if r["variant"] == "all-huge":                              # logits flattened (argmax kept only
            assert r["kl"] > 1.0                                    # without a value bias)
        if r["variant"] == "host":                                  # exact off the start-token id
            assert r["kl_ex_start"] < 1e-8 and r["ex64"] == 1.0
    assert {r["model"] for r in rs if r["variant"] == "host"} >= {"HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B"}


# --- 4. apply() touches only the slices ----------------------------------------------------------------
def test_apply_only_touches_slices():
    import numpy as np

    from graph.frontier import load
    from graph.model_graph import _dedup_points, _lm_config, emul_matrix, iter_json_configs
    P = _dedup_points(load(400))
    cfgs = {n: _lm_config(c) for n, c in iter_json_configs("graph/hub_10k.json")}
    ok = emul_matrix(P, mode="payable")
    out = R.apply(ok, P, cfgs)
    sl = R.dense_moe_slice(P) | R.loop_slice(P)
    assert (out[~sl] == ok[~sl]).all()
    walled = [i for i, p in enumerate(P) if p["slots"]["ffn"].startswith("moe")
              and R._point_verdict(p, cfgs, R.dense_moe_verdict) == "wall"]
    assert not out[np.ix_(walled, np.where(R.dense_moe_slice(P).any(0))[0])].any()
