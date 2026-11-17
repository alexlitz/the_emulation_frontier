"""The positional-encoding reduction structure, analytically (numpy, no SGD):
NoPE=RoPE(inv_freq->0) exact, ALiBi=a fixed RoPE pair (quadratic-in-angle floor),
RoPE->hard window Gibbs-floors at the edge but converges in the interior, and the
cross-emulation matrix has the claimed span structure.

Run:  python -m pytest tests/test_positional_cross.py -q
"""
import numpy as np

from attn.positional_cross import (alibi_identity_error, alibi_smallangle_error,
                                    cross_matrix, nope_from_rope_error,
                                    rope_pair_score, window_errors)


# 1. NoPE = RoPE at inv_freq -> 0 : EXACT ------------------------------------

def test_nope_is_rope_at_zero_freq_exact():
    assert nope_from_rope_error(seq=16) == 0.0
    assert nope_from_rope_error(seq=32, seed=3) == 0.0


# 2. ALiBi = a fixed RoPE pair (seed) ----------------------------------------

def test_alibi_is_fixed_rope_pair_exact_identity():
    # <R_i q, R_j k> == -N sin(theta (i-j)) at fp noise
    assert alibi_identity_error(N=7.3, theta=0.01, D=40) < 1e-10
    # spot-check the closed form directly
    N, th = 5.0, 0.02
    assert abs(rope_pair_score(N, th, 9, 4) - (-N * np.sin(th * 5))) < 1e-12


def test_alibi_smallangle_error_is_quadratic_in_theta():
    # halving theta (slope fixed) must QUARTER the error -> ratio -> 4 (sin remainder)
    errs = [alibi_smallangle_error(1.0, th, D=64) for th in (1e-2, 5e-3, 2.5e-3, 1.25e-3)]
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for r in ratios:
        assert 3.7 < r < 4.3, ratios
    # and the error is monotonically shrinking toward 0 (exact -> limit)
    assert errs[0] > errs[-1] and errs[-1] < 0.1


# 3. RoPE -> hard local window: Gibbs FLOOR at the edge, converges elsewhere --

def test_rope_window_sup_norm_is_a_gibbs_floor():
    # sup-norm error over the WHOLE domain (includes the discontinuity) does NOT ->0:
    # the partial sum hits the mid-value 0.5 at the jump. It stays >~0.25 across N.
    linfs = [window_errors(256, 16, N)[0] for N in (8, 16, 32, 64)]
    assert min(linfs) > 0.2                               # a real floor, never reaches a hard mask
    # the Gibbs OVERSHOOT settles to a constant ~0.09 (does not vanish with N)
    ov = [window_errors(256, 16, N)[3] for N in (16, 32, 64)]
    assert all(0.05 < o < 0.2 for o in ov)


def test_rope_window_interior_and_L2_converge():
    # away from the edge the error SHRINKS with N (direction of the scaling law)
    inter = [window_errors(256, 16, N)[1] for N in (8, 16, 32, 64)]
    assert inter[0] > inter[1] > inter[2] > inter[3]      # monotone decreasing
    assert inter[-1] < 0.5 * inter[0]                     # ~1/N -> at least halves
    # L2/RMS decreases too (~1/sqrt(N))
    rms = [window_errors(256, 16, N)[2] for N in (8, 64)]
    assert rms[1] < rms[0]


def test_smooth_edge_window_decays_fast():
    # a SMOOTH (raised-cosine) edge has no discontinuity -> spectral decay, beats the hard edge
    hard = [window_errors(256, 16, N)[0] for N in (16, 64)]
    soft = [window_errors(256, 16, N, smooth=8)[0] for N in (16, 64)]
    assert soft[1] < 1e-2                                  # smooth edge reaches ~1e-3 at N=64
    assert soft[1] < hard[1]                               # far below the hard-edge sup-norm floor


# 4. cross-emulation matrix structure ----------------------------------------

def test_cross_matrix_structure():
    names, M = cross_matrix(L=24)
    assert set(names) == {"nope", "alibi", "rope", "t5_relbias", "learned_abs", "sinusoidal"}

    # diagonal is exact
    for n in names:
        assert M[n][n] == 0.0

    # NoPE is the BOTTOM: every scheme emulates NoPE exactly (zero its field, free reverse)
    for X in names:
        assert M["nope"][X] == 0.0

    # T5 relbias (arbitrary relative g(d)) emulates every RELATIVE scheme exactly (fp noise)
    assert M["alibi"]["t5_relbias"] < 1e-10
    assert M["rope"]["t5_relbias"] < 1e-10

    # RoPE emulates ALiBi in the small-angle LIMIT: a small floor, not incomparable
    assert 0.0 < M["alibi"]["rope"] < 0.2

    # RoPE emulates the (bucketed) T5 relative bias only up to a band-limited floor (small, nonzero)
    assert 0.0 < M["t5_relbias"]["rope"] < 0.2

    # ABSOLUTE vs RELATIVE incomparability: purely-relative schemes CANNOT make an
    # absolute a(i)+b(j) field -> learned_abs floors near its full range under them
    for X in ("nope", "alibi", "rope", "t5_relbias"):
        assert M["learned_abs"][X] > 0.5
    # ...and a purely-relative RoPE field is out of reach of NoPE/ALiBi/absolute hosts
    for X in ("nope", "alibi", "learned_abs"):
        assert M["rope"][X] > 0.5

    # sinusoidal has an absolute part, so it is the CLOSEST host to learned_abs
    assert M["learned_abs"]["sinusoidal"] < min(
        M["learned_abs"][X] for X in ("nope", "alibi", "rope", "t5_relbias"))
