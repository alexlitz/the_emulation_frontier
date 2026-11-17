"""Local/sliding/sparse attention ⊆ full attention — checked COMBINATORIALLY
(numpy boolean reachability `(I ∨ A)^L`, no training):

  * sliding-window reach composes EXACTLY as L*(w-1)+1 (verified by mask composition);
  * the depth-to-global law L* = ⌈(T-1)/(w-1)⌉ matches the actual depth at which the
    composed mask becomes fully causally dense;
  * sparse patterns reach full at their graph DIAMETER: block-local = (N-1)//b,
    local+strided = 2, strided/dilated alone = NEVER (residue-class gaps).

Run:  python -m pytest tests/test_mask_reach.py -q
"""
from math import ceil

import numpy as np

from block.mask_reach import (block_local, causal, depth_to_global,
                              depth_to_global_formula, diameter, dilated, reach,
                              reach_count, sliding, strided)


def test_pattern_is_subset_of_full_for_free():
    # every causal pattern is a subset of full attention's edges (mask, exact, no cost)
    N = 48
    C = causal(N)
    for A in (sliding(N, 5), strided(N, 4), block_local(N, 8), dilated(N, 2, 4)):
        assert (A & C == A).all()          # already causal
        assert (A <= C).all()              # ⊆ full


def test_sliding_reach_is_L_times_wminus1_plus1():
    # THE law, checked by actually composing the boolean mask L times (not the formula)
    N = 64
    for w in (2, 3, 5, 8):
        for L in (1, 2, 3, 4):
            rc = reach_count(sliding(N, w), L)      # composed (I∨A)^L, last row
            assert rc == min(N, L * (w - 1) + 1), (w, L, rc)


def test_one_layer_reaches_exactly_w():
    # sanity: a single sliding layer sees exactly w tokens (self + w-1 previous)
    N = 40
    for w in (1, 2, 5, 8):
        assert reach_count(sliding(N, w), 1) == w


def test_depth_to_global_matches_formula_via_composition():
    # depth at which composed mask becomes fully causally dense == ⌈(T-1)/(w-1)⌉
    T = 64
    for w in (2, 3, 5, 8, 9, 16, 33):
        Lc = depth_to_global(sliding(T, w), T)      # measured by mask composition
        Lf = depth_to_global_formula(T, w)          # closed form
        assert Lc == Lf == ceil((T - 1) / (w - 1)), (w, Lc, Lf)


def test_full_attention_reaches_global_in_one_layer():
    N = 32
    assert diameter(causal(N), N) == 1
    assert depth_to_global_formula(N, N) == 1       # w = T corner


def test_window_depth_tradeoff_is_monotone():
    # smaller window strictly needs (weakly) more depth to go global
    T = 64
    ds = [depth_to_global(sliding(T, w), T) for w in (2, 3, 5, 9, 17, 33)]
    assert ds == sorted(ds, reverse=True)           # non-increasing in w
    assert ds[0] == T - 1                            # w=2 needs T-1 layers
    assert ds[-1] == 2                               # w=33 (> T/2) needs 2


def test_block_local_diameter_is_floor():
    # block-local (own + previous block) diameter = (N-1)//b, verified by composition
    for N in (32, 48, 64):
        for b in (4, 8, 16):
            assert diameter(block_local(N, b), N) == (N - 1) // b, (N, b)


def test_local_plus_strided_factorizes_to_two_hops():
    # Sparse-Transformer √N factorization: local(w) ∪ strided(w) reaches full in 2 layers
    N = 64
    w = 8                                            # w = √N
    assert diameter(sliding(N, w) | strided(N, w), N) == 2


def test_strided_and_dilated_alone_never_reach_full():
    # every edge shares a common stride -> reachable set is a residue class -> NEVER dense
    N = 64
    assert diameter(strided(N, 2), N) is None
    assert diameter(strided(N, 4), N) is None
    assert diameter(dilated(N, 2, 8), N) is None


def test_reach_is_monotone_in_depth():
    # more layers can only add reachable pairs (boolean OR grows)
    N = 48
    A = sliding(N, 4)
    prev = reach(A, 1)
    for L in range(2, 8):
        cur = reach(A, L)
        assert (prev <= cur).all()                  # monotone growth
        prev = cur
