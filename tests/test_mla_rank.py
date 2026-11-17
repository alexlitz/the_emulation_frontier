"""KV-sharing and MLA rank, checked ANALYTICALLY (numpy, fp64; no SGD, SVD only):

  * MQA/GQA -> MHA by tiling shared KV heads (repeat_interleave) is EXACT (0.0);
  * MLA's non-positional absorption fold is exact (~fp64 eps), the RoPE part is not;
  * the optimal rank-r (SVD) K,V gives a monotone-decreasing attention-output error
    that reaches ~0 at full rank;
  * KV-cache slots/token order MHA >= GQA >= MLA >= MQA.

Run:  python -m pytest tests/test_mla_rank.py -q
"""
import numpy as np

from attn.mla_rank import (emulation_matrix, gqa_to_mha_error, kv_memory_table,
                           mla_absorption_error, rank_sweep, rope_fold_gap)


def test_mqa_gqa_to_mha_exact():
    # tiling the shared KV heads up to n_head heads reproduces GQA/MQA EXACTLY, 0.0
    for n_kv in (1, 2, 4):
        err = gqa_to_mha_error(n_kv, n_head=8, head_dim=8)
        assert err == 0.0, (n_kv, err)


def test_mla_absorption_fold_is_exact():
    # non-positional score/output fold: (W_UK^T q).c and (A c) W_UV -> ~fp64 eps
    ms, mo = mla_absorption_error()
    assert ms < 1e-9, ms
    assert mo < 1e-9, mo


def test_rope_part_does_not_fold():
    # the positional (RoPE) part cannot absorb into a static bilinear form -> nonzero
    gap = rope_fold_gap()
    assert gap > 1.0, gap


def test_rank_error_decreasing_to_zero_at_full_rank():
    # optimal rank-r [K|V] (SVD): attention-output error is monotone-nonincreasing
    # in r and hits ~0 at full rank. Checked on both flat and decaying spectra.
    for decay in (None, 1.0):
        rows, full = rank_sweep(n_head=8, head_dim=16, seq=128, decay=decay)
        errs = [e for _, e, _ in rows]
        ranks = [r for r, _, _ in rows]
        # monotone non-increasing (SVD residual shrinks as r grows)
        for a, b in zip(errs, errs[1:]):
            assert b <= a + 1e-12, (decay, errs)
        # full rank present and error there is ~0 (fp64)
        assert ranks[-1] == full
        assert errs[-1] < 1e-10, (decay, errs[-1])
        # and a strict improvement from smallest to full rank
        assert errs[-1] < errs[0]


def test_decaying_spectrum_beats_flat_at_low_rank():
    # realistic decaying spectrum: low-rank cut is far cheaper than worst-case flat
    flat, _ = rank_sweep(n_head=8, head_dim=16, seq=128, decay=None)
    powr, _ = rank_sweep(n_head=8, head_dim=16, seq=128, decay=1.0)
    r_flat = dict((r, e) for r, e, _ in flat)
    r_pow = dict((r, e) for r, e, _ in powr)
    for r in (4, 8, 16, 32):
        assert r_pow[r] < r_flat[r], (r, r_pow[r], r_flat[r])


def test_kv_relative_residual_monotone_and_zero_at_full():
    # the SVD tail energy (relative KV residual) is strictly decreasing -> 0 at full
    rows, full = rank_sweep(n_head=8, head_dim=16, seq=128, decay=1.0)
    resids = [resid for _, _, resid in rows]
    for a, b in zip(resids, resids[1:]):
        assert b < a, resids
    assert resids[-1] == 0.0 or resids[-1] < 1e-14


def test_kv_memory_ordering():
    # cache slots/token: MHA (all heads) >= GQA (n_kv groups) >= MQA (1 head);
    # MLA caches its rank r. With these dims MHA >= GQA >= MLA(=GQA here) >= MQA.
    kv = kv_memory_table(n_head=8, head_dim=16, n_kv=2, rank=32)
    assert kv["MHA"] == 8 * 16
    assert kv["GQA"] == 2 * 16
    assert kv["MQA"] == 16
    assert kv["MLA"] == 32
    assert kv["MHA"] >= kv["GQA"] >= kv["MQA"]
    assert kv["MHA"] > kv["MLA"]
    # MLA rank below full n_head*head_dim is a real cache saving vs MHA
    assert kv["MLA"] < kv["MHA"]


def test_emulation_matrix_conserved_currency():
    # MHA (widest host) exactly emulates every target, paying the TARGET's cache;
    # thriftier hosts cannot exactly realize a fatter target.
    kv = kv_memory_table(n_head=8, head_dim=16, n_kv=2, rank=32)
    order, reach, kv = emulation_matrix(kv)
    # MHA reaches all
    for Y in order:
        assert reach["MHA"][Y] is True or reach["MHA"][Y] == "r=full" or reach["MHA"][Y] is True
    assert reach["MHA"]["MQA"] is True
    assert reach["MHA"]["GQA"] is True
    # MQA (thriftiest discrete host) cannot exactly emulate MHA or GQA
    assert reach["MQA"]["MHA"] is False
    assert reach["MQA"]["GQA"] is False
    # MLA reaches the discrete family only at full rank
    assert reach["MLA"]["MHA"] == "r=full"
    assert reach["MLA"]["MLA"] is True
