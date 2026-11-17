"""Soundness (wall audit, model_graph key wa_pack_zero_room): KV-head packing of a position-free target (NoPE / ALiBi /
learned-abs) on a fixed-rotary (partial) RoPE host. Each packed block must sit at frequency 0 (Lean kvPack_nope needs
the blocks unrotated), so q blocks need q * head_dim_t zero-frequency host dims (zero_freq_room), not one block's worth.

Negative control: ibm-granite/granite-4.0-h-350m (NoPE attention, 4 KV heads x 3 q x 64) on zai-org/glm-4-9b (partial
RoPE: 64 of 128 dims rotate, 2 KV heads x 16 q) was granted in mixgap by packing 2 blocks into one 128-dim head with
room for 1.

Run:  python -m pytest tests/test_pack_zero_room.py -q
"""
import contextlib
import functools

import numpy as np
import pytest

from graph import model_graph as mg


@contextlib.contextmanager
def wa_off(*keys):
    added = [k for k in keys if k not in mg.WA_OFF]
    mg.WA_OFF.update(added)
    try:
        yield
    finally:
        mg.WA_OFF.difference_update(added)


def _pt(pos, kv, q_per_kv, hd, cap, pf=0):
    return {"slots": {"pos": pos, "kv": "GQA" if q_per_kv > 1 else "MHA"},
            "dims": {"kv_heads": kv, "q_per_kv": q_per_kv, "head_dim": hd, "rope_sched": cap, "rope_cap": cap,
                     "rope_pairs_free": pf, "kFU_eff": 2, "qheads": kv * q_per_kv}}


def test_pack_needs_zero_room_per_block():
    """NoPE target, 4 KV heads of 32 dims; partial-RoPE hosts with 64 pass dims per 128-dim head (2 blocks each)."""
    tgt = _pt("nope", 4, 2, 32, ())
    host_two = _pt("partial_rope", 2, 4, 128, (64,))      # 2 heads x 2 blocks = 4 target heads: fits
    host_one = _pt("partial_rope", 1, 8, 128, (64,))      # 1 head x 2 blocks < 4
    P = [tgt, host_two, host_one]
    ok = mg.kv_heads_fit(P, "payable")
    assert ok[0, 1] and not ok[0, 2]
    with wa_off("wa_pack_zero_room"):
        assert mg.kv_heads_fit(P, "payable")[0, 2]        # the old rule: 4 blocks of 32 in 128 dims, 64 rotating


def test_pairs_free_host_unchanged():
    tgt = _pt("nope", 4, 2, 32, ())
    host = _pt("partial_rope", 1, 8, 128, (64,), pf=1)    # a pairs-free host sets every frequency to 0
    assert mg.kv_heads_fit([tgt, host], "payable")[0, 1]


@functools.lru_cache(maxsize=None)
def _points():
    from graph.currency_curve import points
    return points()


def test_negative_control_granite_on_glm4_mixgap():
    P = _points()
    names = {m: k for k, p in enumerate(P) for m in p["members"]}
    t, h = names.get("ibm-granite/granite-4.0-h-350m"), names.get("zai-org/glm-4-9b")
    if t is None or h is None:
        pytest.skip("pair not in the snapshot")
    sub = [P[t], P[h]]
    assert not mg.kv_heads_fit(sub, "mixgap")[0, 1]
    assert not mg.emul_matrix(P, mode="mixgap")[t, h]
    with wa_off("wa_pack_zero_room"):
        assert mg.kv_heads_fit(sub, "mixgap")[0, 1]
        assert mg.emul_matrix(P, mode="mixgap")[t, h]
