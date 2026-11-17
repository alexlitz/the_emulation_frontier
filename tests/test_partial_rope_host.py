"""A partial-RoPE host reproduces a full-RoPE target head exactly when it rotates enough dims per head
(hf/partial_rope_host.py), and the graph pays pos rope <- partial_rope exactly there (the rot gate).

  * stock modules, fp64: Llama / Qwen2 / Qwen3 targets in GLM (interleaved), Phi, GPT-NeoX and
    Qwen3-Next (gated, qk-norm) partial-RoPE hosts; YaRN / llama3 scalings; spare heads; the
    config-only route (rope_theta_h = rope_theta_t ** (rot_h / d_t));
  * a whole LlamaForCausalLM in a GlmForCausalLM, and the real SmolLM2-135M in a GLM host, scored
    like the other payable edges (top-1 on the target's own greedy continuations, greedy identity);
  * where the construction stops (rot_h < rot_t) and the graph's rot gate (classify / dims_of /
    rope_fits / emulates) on synthetic points and on the zoo's own configs.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from hf import partial_rope_host as prh
from graph.model_graph import (POS_ORDER, _dedup_points, classify, dims_of, edge_cost, emul_matrix, emulates,
                               rope_fits)

YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024}
LLAMA3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 1024}
EXACT = 1e-12
# HF builds cos / sin in float32 (times attention_scaling) and its qk RMSNorms upcast to float32: where
# host and target differ in those, they agree to that float32 rounding
FP32_TABLE = 1e-6


# --- the construction on stock modules (fp64) -------------------------------------------------------

@pytest.mark.parametrize("family,kw,tol", [
    ("glm", dict(hd=32, factor=0.5), EXACT),                 # Qwen2 target (q/k/v bias), rot_h = d_t, interleaved
    ("glm", dict(hd=64, factor=0.5), EXACT),                 # rot_h = 2 d_t: spare pairs unused
    ("glm", dict(hd=32, factor=0.5, host_H=8, host_KV=4), EXACT),   # spare host heads zeroed
    ("phi", dict(hd=32, factor=0.5, KV=4), EXACT),           # half-split partial rotary
    ("neox", dict(hd=64, factor=0.25, KV=4), EXACT),         # fused per-head qkv, rotary_pct
    ("glm", dict(target_scaling=YARN, host_scaling=YARN), EXACT),   # same attention_scaling: exact
    ("glm", dict(target_scaling=YARN), FP32_TABLE),          # target mscale folded into W_q
    ("glm", dict(target_scaling=LLAMA3, host_scaling=YARN), FP32_TABLE),   # host mscale folded out
    ("qwen3_next", dict(hd=64, factor=0.25, qwen3=True), FP32_TABLE),     # Qwen3.8-style host: gate + qk-norm
])
def test_partial_host_reproduces_full_rope_head(family, kw, tol):
    assert prh.module_check(family, **kw) < tol


def test_theta_only_route():
    # no buffer write: rope_theta_h = theta_t ** (rot_h / d_t) puts the target's frequencies on the host's
    # first pairs (here the float32 powers even agree bit for bit)
    freq_err, out_err = prh.theta_only_check()
    assert freq_err < 1e-6 and out_err < FP32_TABLE


@pytest.mark.parametrize("scaling,tol", [(None, EXACT), (YARN, FP32_TABLE)])
def test_whole_llama_in_glm_host(scaling, tol):
    diff, same_greedy = prh.whole_model_check(rope_scaling=scaling)
    assert diff < tol and same_greedy


def test_construction_needs_rot_host_ge_rot_target():
    t = prh.Target(H=2, KV=2, d=16, rot=16, dv=16, wq=torch.zeros(32, 8), wk=torch.zeros(32, 8),
                   wv=torch.zeros(32, 8), wo=torch.zeros(8, 32), inv_freq=torch.ones(8), scale=0.25)
    for rot_h, ok in ((16, True), (32, True), (8, False)):
        h = prh.Host(H=2, KV=2, hd=64, rot=rot_h, interleaved=False, scale=0.125)
        if ok:
            m, inv = prh.column_map(t, h)
            assert len(set(m)) == 16 and inv[:8].eq(1).all() and inv[8:].eq(0).all()
        else:
            with pytest.raises(ValueError, match="rotates"):
                prh.column_map(t, h)
    # a partial target: its pass dims go to the host's pass dims, then to zero-frequency pairs
    tp = prh.Target(H=1, KV=1, d=16, rot=8, dv=16, wq=torch.zeros(16, 8), wk=torch.zeros(16, 8),
                    wv=torch.zeros(16, 8), wo=torch.zeros(8, 16), inv_freq=torch.ones(4), scale=0.25)
    m, inv = prh.column_map(tp, prh.Host(H=1, KV=1, hd=16, rot=16, interleaved=True, scale=0.25))
    assert sorted(m) == list(range(16)) and inv[4:].eq(0).all()


def test_more_heads_do_not_replace_rotated_dims():
    # the head grouping must hold the target's; extra heads are spare, they never carry another head's pairs
    t = prh.Target(H=4, KV=2, d=16, rot=16, dv=16, wq=torch.zeros(64, 8), wk=torch.zeros(32, 8),
                   wv=torch.zeros(32, 8), wo=torch.zeros(8, 64), inv_freq=torch.ones(8), scale=0.25)
    with pytest.raises(ValueError, match="rotates"):
        prh.place(t, prh.Host(H=64, KV=64, hd=32, rot=8, interleaved=False, scale=32 ** -0.5))
    with pytest.raises(ValueError, match="grouping"):
        prh.head_map(t, prh.Host(H=4, KV=1, hd=32, rot=16, interleaved=False, scale=32 ** -0.5))


# --- a real checkpoint ------------------------------------------------------------------------------

def _cached(repo):
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(repo, "config.json")
    return isinstance(p, str) and os.path.exists(p)


@pytest.mark.skipif(not _cached("HuggingFaceTB/SmolLM2-135M"), reason="SmolLM2-135M not in the HF cache")
def test_real_smollm2_in_glm_host():
    # the payable standard (act/act_swap_agreement.py): top-1 >= 99% and >= 90% of 32-token greedy
    # continuations identical. The exact construction meets it with every token identical.
    r = prh.real_check(n_prompts=8)
    assert r["top1"] == 1.0 and r["ex32"] == 1.0 and r["ex64"] == 1.0
    assert r["max_logp_diff"] < 1e-3                       # float32 run


# --- the graph: classify, dims_of, the rot gate -----------------------------------------------------

def _cfg(**kw):
    base = dict(model_type="llama", hidden_size=2048, num_attention_heads=16, num_key_value_heads=4,
                head_dim=128, num_hidden_layers=24, intermediate_size=8192, rms_norm_eps=1e-6,
                rope_theta=1e4, hidden_act="silu", vocab_size=32000)
    base.update(kw)
    return base


def test_classify_and_dims_record_the_rotated_dims():
    # Qwen3.5 / Qwen3.8: the factor nested in rope_parameters only (it used to read as full RoPE)
    q35 = _cfg(model_type="qwen3_5_text", head_dim=256,
               rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                "mrope_section": [11, 11, 10], "mrope_interleaved": True})
    assert classify(q35)["pos"] == "partial_rope"
    assert dims_of(q35)["rope_sched"] == (64,) and dims_of(q35)["rope_cap"] == (64,)
    # Llama: one schedule over the whole head
    assert classify(_cfg())["pos"] == "rope" and dims_of(_cfg())["rope_sched"] == (128,)
    # MLA: the decoupled RoPE key; none at all (qk_rope_head_dim = 0) is NoPE
    mla = _cfg(kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128)
    assert dims_of(mla)["rope_sched"] == (64,)
    assert classify(_cfg(kv_lora_rank=512, qk_rope_head_dim=0, qk_nope_head_dim=256))["pos"] == "nope"
    # Gemma-3: local and global thetas are two tables
    g3 = _cfg(model_type="gemma3_text", head_dim=256, rope_theta=1e6, rope_local_base_freq=1e4)
    assert dims_of(g3)["rope_sched"] == (256, 256)
    # Gemma-4: "proportional" RoPE spans the whole (global) head with a NoPE tail -> its table drives 512
    g4 = _cfg(model_type="gemma4_text", head_dim=256, global_head_dim=512, rope_parameters={
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})
    assert classify(g4)["pos"] == "partial_rope"
    assert dims_of(g4)["rope_sched"] == (256, 128) and dims_of(g4)["rope_cap"] == (512, 256)
    # GPT-NeoX rotary_pct, MiniMax rotary_dim
    assert dims_of(_cfg(model_type="gpt_neox", rotary_pct=0.25, head_dim=None, hidden_size=2048))["rope_sched"] == (32,)
    assert dims_of(_cfg(rotary_dim=64))["rope_sched"] == (64,)
    # RoPE + NoPE layer mixes keep their NoPE layers as a zero-frequency schedule (a negative entry)
    smol3 = _cfg(model_type="smollm3", no_rope_layers=[1, 1, 1, 0] * 6, no_rope_layer_interval=4)
    assert dims_of(smol3)["rope_sched"] == (128, -128) and "nope_layers" in classify(smol3)["feats"]
    beam = _cfg(rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
                                 "full_attention": None})
    assert dims_of(beam)["rope_sched"] == (128, -128) and "nope_layers" in classify(beam)["feats"]
    # use_rope = false is NoPE (the rot gate never sees it)
    assert classify(_cfg(use_rope=False))["pos"] == "nope"


def test_rope_fits():
    assert rope_fits((128,), (128,)) and rope_fits((64,), (128,)) and not rope_fits((128,), (64,))
    assert rope_fits((), (64,)) and rope_fits((0,), (64,))          # NoPE needs nothing
    assert not rope_fits((256, 256), (256,))                        # two schedules in one table
    assert rope_fits((256, 256), (512,)) and rope_fits((256, 256), (256, 256))
    assert rope_fits((256, 128), (512, 256)) and not rope_fits((288, 288), (512, 256))
    # NoPE layers in a RoPE mix (-r: r zero-frequency slots) are not free: a one-table host whose table
    # rotates the whole head cannot also hold them; a wider table, a second table or a NoPE table can
    assert not rope_fits((128, -128), (128,))
    assert rope_fits((128, -128), (256,)) and rope_fits((128, -128), (128, 128))
    assert rope_fits((128, -128), (128, -128)) and not rope_fits((128, 128), (128, -128))
    assert rope_fits((-128,), (64,))                                # nothing rotates: inv_freq = 0
    # transitive: a packing of packings is a packing (A -> B -> C)
    assert rope_fits((128, -128), (128, 128)) and rope_fits((128, 128), (256,)) and rope_fits((128, -128), (256,))


def _pt(pos, sched, cap=None, **dims):
    from tests.test_graph_rules import mk
    return mk(pos=pos, rope_sched=sched, rope_cap=cap or sched, **dims)


def test_rot_gate_in_emulates():
    assert edge_cost(POS_ORDER, "rope", "partial_rope") == (["rot"], True)
    full64, part64, part128 = _pt("rope", (64,)), _pt("partial_rope", (64,)), _pt("partial_rope", (128,))
    full_t = _pt("rope", (128,), head_dim=128)
    host64 = _pt("partial_rope", (64,), hidden=4096, layers=48, q_cap=4096, kv_cap=4096, head_dim=256)
    host128 = _pt("partial_rope", (128,), hidden=4096, layers=48, q_cap=4096, kv_cap=4096, head_dim=256)
    assert emulates(host128, full_t) and not emulates(host64, full_t)           # rope <- partial_rope, gated
    assert emulates(host128, full_t, mode="exact")                              # the construction is exact
    assert emulates(host64, full_t, mode="capacity")                            # capacity ignores pos
    assert emulates(host64, full64) and emulates(host64, part64)                # 64 rotated dims suffice
    assert emulates(host128, part128) and not emulates(host64, part128)        # partial <- partial, gated
    # a full-RoPE host is gated too (its table spans the head): one table holds one schedule of up to its
    # head's dims, or several schedules that fit together
    big_full = _pt("rope", (256,), hidden=4096, layers=48, q_cap=4096, kv_cap=4096, head_dim=256)
    assert emulates(big_full, part128) and emulates(big_full, full_t)
    two_tables = _pt("rope", (256, 256), head_dim=256)                  # Gemma-3: local + global theta
    assert not emulates(big_full, two_tables)
    wide = dict(hidden=4096, layers=48, q_cap=4096, kv_cap=4096, head_dim=512)
    # round 2, pass 5: two schedules in one table need a frequency per pair (a longrope-capable rotary); a fixed
    # rotary sets rope_theta only, one geometric schedule
    assert emulates(_pt("rope", (512,), rope_pairs_free=1, **wide), two_tables)
    assert not emulates(_pt("rope", (512,), **wide), two_tables)
    rope_nope = _pt("rope", (128, -128), head_dim=128)                 # SmolLM3: RoPE + NoPE layers
    assert emulates(big_full, rope_nope, mode="capacity")
    assert not emulates(_pt("rope", (128,), hidden=4096, layers=48, q_cap=4096, kv_cap=4096, head_dim=128),
                        rope_nope)


HUB = "graph/hub_10k.json"


@pytest.mark.skipif(not os.path.exists(HUB), reason="no zoo snapshot")
def test_zoo_partial_rope_walls():
    """Qwen3.8-2.4T-A95B (Qwen3.5 MoE hybrid: head_dim 256 x 0.25 = 64 rotated dims) now hosts the
    frontier models whose own RoPE fits in 64 dims -- ThoxOS-Kernel-27B (also Qwen3.5: 64; GLM-5.3-Flash,
    MLA without a RoPE key, passes the rot gate but not the edge-case audit's walls) -- and still not the full-RoPE head_dim-128 / 256
    targets (Hy3, Ling-plus, step3, K-EXAONE-2.0, snowflake-arctic, Beam-1.0, Trinity-Large)."""
    want = {"Qwen/Qwen3.8-2.4T-A95B", "Thox-ai/ThoxOS-Kernel-27B", "avlp12/GLM-5.3-Flash-Alis-MLX-4bit",
            "tencent/Hy3", "inclusionAI/Ling-plus", "stepfun-ai/step3"}
    rows = {r["id"]: r["config"] for r in json.load(open(HUB)) if r["id"] in want}
    assert set(rows) == want
    pts = {n: {"name": n, "slots": classify(c), "dims": dims_of(c)} for n, c in rows.items()}
    host = pts["Qwen/Qwen3.8-2.4T-A95B"]
    assert host["slots"]["pos"] == "partial_rope" and host["dims"]["rope_cap"] == (64,)
    assert emulates(host, pts["Thox-ai/ThoxOS-Kernel-27B"])
    # GLM-5.3-Flash's position passes (MLA without a RoPE key: NoPE), but the edge-case audit's walls stop
    # it: an MLA target on a plain host, hyper-connections, the KDA recurrence (DSA was one too until round 2,
    # pass 3, which treats DSA as dense attention: exact while the context is <= index_topk)
    from graph.frontier import blockers
    glm = pts["avlp12/GLM-5.3-Flash-Alis-MLX-4bit"]
    assert glm["slots"]["pos"] == "nope" and not emulates(host, glm)
    why = blockers(glm, host)
    assert not any(w.startswith("pos ") for w in why) and not any("dsa" in w for w in why), why
    assert any("hyper_conn" in w for w in why) and any("MLA" in w for w in why), why
    for t, rot in (("tencent/Hy3", 128), ("inclusionAI/Ling-plus", 128), ("stepfun-ai/step3", 256)):
        assert pts[t]["dims"]["rope_sched"] == (rot,)
        assert not rope_fits(pts[t]["dims"]["rope_sched"], host["dims"]["rope_cap"])
        assert not emulates(host, pts[t]), t
    # the dedup key carries the schedules, so self-emulation holds on every point
    P = _dedup_points(list(pts.values()))
    assert emul_matrix(P, mode="payable").diagonal().all()
