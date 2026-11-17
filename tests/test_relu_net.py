"""Every activation reduces to a ReLU network — checked ANALYTICALLY (explicit
weights, no training):

  * ungated curves need only ONE hidden layer, error ~1/N^2;
  * the gated product (SwiGLU/GeGLU) reduces via 1/4((p+b)^2-(p-b)^2) with x^2 from a
    depth-L Yarotsky sawtooth, error ~4^-L.

Run:  python -m pytest tests/test_relu_net.py -q
"""
import numpy as np

from act.relu_net import (gate_error, gelu, mish, silu, square_error, tanh,
                          ungated_error)


def test_ungated_one_layer_is_enough():
    # a SINGLE hidden layer of N ReLUs; error falls ~1/N^2, so 16x more width -> ~256x smaller
    for phi in (silu, gelu, mish, tanh):
        e16, e256 = ungated_error(phi, 16), ungated_error(phi, 256)
        assert e256 < e16 / 30, (phi, e16, e256)
        assert e256 < 1e-3


def test_squaring_shrinks_4_to_minus_L():
    e2, e4, e6 = square_error(2), square_error(4), square_error(6)
    assert e6 < e4 < e2
    assert 10 < e2 / e6 < 400            # ~4^-L over 4 extra depth -> ~256x
    assert e6 < 1e-4


def test_gated_gate_converges_with_depth():
    # analytic gate approximation improves monotonically with product depth, no training
    e2, e6 = gate_error(silu, 256, 2), gate_error(silu, 256, 6)
    assert e6 < e2 / 20
    assert e6 < 5e-3
    assert gate_error(gelu, 256, 6) < 5e-3


def test_exotic_activations_are_near_gap_not_walls():
    # xIELU (quadratic branch) and situ (Kimi-K3, SiLU's finite-beta generalization) don't
    # fold as a single op, but they are pointwise curves -- a wider FFN folds them ~1/N^2,
    # so they are width-priced near-gap edges, NOT hard walls like absolute position.
    from act.activation_cross import ACTS, floor
    situ, xielu, relu2 = ACTS["situ"], ACTS["xielu"], ACTS["relu2"]
    for phi in (situ, xielu):
        e8, e32 = ungated_error(phi, 8, -6, 6), ungated_error(phi, 32, -6, 6)
        assert e32 < e8 / 10, (phi, e8, e32)          # 4x width -> ~16x smaller (~1/N^2)
    # xIELU (the real HF definition, default init alpha_p = alpha_n = 0.8) is a pure quadratic
    # 0.8 x^2 + 0.5 x on the positive branch, which dominates the interpolation error on [-6,6]:
    # err = 0.8 * (12/N)^2 / 4 = 28.8 / N^2 EXACTLY (the old simplified xIELU, alpha_p =
    # softplus(0.8) = 1.17, gave 42/N^2).
    cN2 = [ungated_error(xielu, N, -6, 6) * N * N for N in (2, 4, 8, 16, 32, 64)]
    assert max(abs(c - 28.8) for c in cN2) < 1e-3, cN2
    # situ closes as ~9/N^2 once N >= 16
    sN2 = [ungated_error(situ, N, -6, 6) * N * N for N in (16, 32, 64)]
    assert all(8.0 < c < 9.5 for c in sN2), sN2
    # ...and the real xIELU is NOT an exact squared-ReLU fold: its negative branch
    # alpha_n (e^x - 1 - x) + beta x is nonlinear, so relu^2 hosts it only to a 0.166 floor
    # (the reverse, xIELU hosting relu^2, is near-exact: the negative branch is linear far out).
    assert 0.15 < floor(relu2, xielu) < 0.18
    assert floor(xielu, relu2) < 0.01


def test_free_knot_minimax_small_K():
    # a TRUE minimax free-knot fit (LP for the coefficients, global search over the knots) on
    # [-8,8]. K=1 tanh reproduces the post's 0.605; K=2 is far below the post's 0.133/0.147.
    from act.relu_net import free_knot_minimax
    e1_tanh, t1, _ = free_knot_minimax(tanh, 1)
    assert abs(e1_tanh - 0.6048) < 2e-3, e1_tanh
    e1, _, _ = free_knot_minimax(silu, 1)
    e2, t2, _ = free_knot_minimax(silu, 2)
    assert abs(e1 - 0.1396) < 2e-3 and abs(e2 - 0.0497) < 2e-3, (e1, e2)
    assert np.allclose(sorted(t2), [-0.60, 0.60], atol=0.05)      # symmetric knots near +-0.6
    # 2 optimal units already beat 16 uniform ones (uniform N=16 on [-8,8] is 0.054 for SiLU)
    assert e2 < ungated_error(silu, 16)