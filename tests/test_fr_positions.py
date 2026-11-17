"""fr-positions: the positions-wall rules of branch fr-positions (graph/model_graph.py), catalog style -- each rule's
construction (hf/fr_positions.py, Lean FrontierPositions.lean), its reading of the configs, and the relation's verdict
on synthetic points and on the zoo's own configs.

  fp_nope_layer_host      RoPE + NoPE host for a target without NoPE layers: NoPE layers zeroed, rotating-layer Hall
  fp_sinusoid_table       a fixed sinusoid table on a learned-abs host (and through the start-token rule)
  fp_sinusoid_same_table  sinusoidal <- sinusoidal only for the same table (formula and width)
  fp_table_length         learned <- learned only with the target's rows (transitivity once tables ride on learned)
  fp_gemma4_plc_rope      Gemma-4's full-layer table width from per_layer_config
"""
from __future__ import annotations

import json
import os

import pytest

from graph import model_graph as mg
from graph.model_graph import _dedup_points, classify, dims_of, emul_matrix, iter_json_configs

HERE = os.path.dirname(os.path.abspath(__file__))


def _pts(*cfgs):
    ms = []
    for i, c in enumerate(cfgs):
        ms.append({"name": f"m{i}", "slots": classify(c), "dims": dims_of(c)})
    return [{"slots": m["slots"], "dims": m["dims"], "members": [m["name"]]} for m in ms]


def _why(t, h, mode="payable"):
    why = {}
    ok = emul_matrix(_pts(t, h), mode=mode, explain=why)
    return bool(ok[0, 1]), sorted(nm for nm, m in why.items() if m[0, 1])


def _llama(**kw):
    base = dict(model_type="llama", architectures=["LlamaForCausalLM"], hidden_size=1024, num_attention_heads=16,
                num_key_value_heads=4, head_dim=64, num_hidden_layers=12, intermediate_size=4096, rms_norm_eps=1e-6,
                rope_theta=1e4, hidden_act="silu", vocab_size=32000, max_position_embeddings=4096)
    base.update(kw)
    return base


def _smollm3(n, every=4, **kw):
    return _llama(model_type="smollm3", architectures=["SmolLM3ForCausalLM"], num_hidden_layers=n,
                  hidden_size=2048, intermediate_size=8192, num_attention_heads=32, num_key_value_heads=8,
                  no_rope_layers=[int((i + 1) % every != 0) for i in range(n)], tie_word_embeddings=False, **kw)


_ZOO = {}


def _zoo(name):
    if not _ZOO:
        want = {"arcee-ai/Trinity-Large-Base", "reflection/Beam-1.0-501B-A23B", "mlx-community/flint-oQ4e",
                "noman-asif/Gemma-4-26B-A4B-Arabic-It", "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit-REAP-14B",
                "facebook/incoder-1B", "facebook/xglm-564M", "oblivious/ruGPT-3.5-13B-GGUF", "Salesforce/ctrl",
                "sshleifer/tiny-ctrl", "togethercomputer/RedPajama-INCITE-7B-Base"}
        for n, c in iter_json_configs(os.path.join(HERE, "..", "graph", "hub_10k.json")):
            if n in want:
                _ZOO[n] = c
    return _ZOO[name]


# --- fp_nope_layer_host ---------------------------------------------------------------------------------------------

def test_nope_flags_read_the_codes_switch():
    d = mg._normalize(_smollm3(8))
    assert mg._nope_flags(d, 8) == [False, False, False, True, False, False, False, True]
    beam = mg._lm_config(_zoo("reflection/Beam-1.0-501B-A23B"))
    fl = mg._nope_flags(beam, 52)
    assert sum(fl) == 9 and all(t == "full_attention" for t, f in zip(beam["layer_types"], fl) if f)
    flint = mg._lm_config(_zoo("mlx-community/flint-oQ4e"))
    assert sum(mg._nope_flags(flint, 49)) == 14
    assert mg._nope_flags(mg._normalize(_llama()), 12) is None          # not a mix


def test_rotating_layer_counts():
    # Trinity (afmoe): RoPE only on the 45 sliding layers, its 15 full layers NoPE
    d = dims_of(_zoo("arcee-ai/Trinity-Large-Base"))
    assert (d["kFU_eff"], d["kLU_eff"], d["rkFU_eff"], d["rkLU_eff"]) == (15, 45, 0, 45)
    d = dims_of(_llama())
    assert all(d[f"rk{k}_eff"] == d[f"k{k}_eff"] for k in ("FU", "FT", "LU", "LT"))


def test_nope_host_plays_a_rope_target_on_its_rope_layers():
    # 16 layers, NoPE every 4th: 12 RoPE layers host a 12-layer RoPE target, not a 13-layer one
    ok, why = _why(_llama(num_hidden_layers=12), _smollm3(16))
    assert ok, why
    ok, why = _why(_llama(num_hidden_layers=13), _smollm3(16))
    assert not ok and why == ["pos rotating layers"]
    # (with the rule off the host-side feature is the wall)
    mg.R3_OFF.add("fp_nope_layer_host")
    try:
        ok, why = _why(_llama(num_hidden_layers=12), _smollm3(16))
        assert not ok and why == ["feature"]
    finally:
        mg.R3_OFF.discard("fp_nope_layer_host")


def test_mixed_target_keeps_its_walls():
    # a target WITH NoPE layers on a one-table host: no construction (one table cannot both rotate and freeze)
    ok, why = _why(_smollm3(8), _llama(num_hidden_layers=24, hidden_size=4096, intermediate_size=16384,
                                       num_attention_heads=32, num_key_value_heads=8))
    assert not ok and "feature" in why
    # on a mixed host its rotating layers need the host's rotating layers
    ok, why = _why(_smollm3(16), _smollm3(12))
    assert not ok


# --- fp_sinusoid_table / fp_sinusoid_same_table / fp_table_length ---------------------------------------------------

def _pair(t, h):
    return _why(_zoo(t), _zoo(h))


def test_sinusoid_rides_on_a_learned_table():
    ok, why = _pair("facebook/incoder-1B", "oblivious/ruGPT-3.5-13B-GGUF")
    assert ok, why
    ok, why = _pair("facebook/incoder-1B", "togethercomputer/RedPajama-INCITE-7B-Base")   # the start-token rule
    assert ok, why


def test_sinusoid_needs_its_rows():
    # CTRL's table spans n_positions = 50000 rows; a 2048-row learned table does not hold it
    d = dims_of(_zoo("Salesforce/ctrl"))
    assert d["pos_table"] == 50000
    ok, why = _pair("Salesforce/ctrl", "oblivious/ruGPT-3.5-13B-GGUF")
    assert not ok and "slot:pos" in why


def test_sinusoid_host_holds_only_its_own_table():
    assert dims_of(_zoo("facebook/xglm-564M"))["sin_sig"] == "xglm:1024"
    ok, why = _pair("facebook/xglm-564M", "facebook/incoder-1B")      # xglm sinusoid at 1024 vs 2048 wide
    assert not ok and why == ["slot:pos"]
    ok, why = _pair("sshleifer/tiny-ctrl", "Salesforce/ctrl")
    assert not ok and "slot:pos" in why
    ok, _ = _pair("facebook/xglm-564M", "facebook/xglm-564M")
    assert ok


def test_learned_table_length():
    gpt2 = dict(model_type="gpt2", architectures=["GPT2LMHeadModel"], n_embd=768, n_head=12, n_layer=12,
                vocab_size=50257, layer_norm_epsilon=1e-5, activation_function="gelu_new")
    long_t, short_h = dict(gpt2, n_positions=4096), dict(gpt2, n_positions=1024, n_embd=1024, n_head=16, n_layer=24)
    ok, why = _why(long_t, short_h)
    assert not ok and "slot:pos" in why
    ok, why = _why(dict(gpt2, n_positions=1024), dict(short_h, n_positions=4096))
    assert ok, why


# --- fp_gemma4_plc_rope ---------------------------------------------------------------------------------------------

def test_gemma4_per_layer_config_width():
    a = dims_of(_zoo("noman-asif/Gemma-4-26B-A4B-Arabic-It"))
    b = dims_of(_zoo("mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit-REAP-14B"))
    assert a["rope_sched"] == b["rope_sched"] == (256, 128)
    assert a["rope_cap"] == b["rope_cap"] == (512, 256)


# --- the measurements (hf/fr_positions.py) ----------------------------------------------------------------------------

def test_measured_at_the_standard():
    p = os.path.join(HERE, "..", "hf", "data", "fr_positions.jsonl")
    rows = [json.loads(x) for x in open(p)]
    for edge in ("rope target <- RoPE + NoPE host (NoPE layers zeroed)", "sinusoidal <- learned_abs (table copied)"):
        r32 = [r for r in rows if r["edge"] == edge and r["dtype"] == "float32"]
        assert len({r["target"] for r in r32}) >= 2
        assert all(r["top1"] >= 0.99 and r["ex32"] >= 0.90 for r in r32)
