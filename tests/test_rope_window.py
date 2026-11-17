"""local window ⊆ global attention with RoPE, by Fourier analysis: N RoPE frequency
blocks weighted by the window's Fourier coefficients give a Dirichlet-kernel score
that attends locally. Currency = width (N blocks) + magnitude (softmax beta); floor =
Gibbs ringing at the edge.

Run:  python -m pytest tests/test_rope_window.py -q
"""
import numpy as np

from attn.rope_window import (build_qk, fourier_coeffs, leakage, real_rope_score,
                             score, window)


def test_real_rope_reproduces_the_designed_score():
    # the ACTUAL RoPE rotation == the cosine series it was built from, to fp64 noise
    T, w = 128, 8
    d, W = window(T, w)
    for N in (4, 16, 64):
        a, th = fourier_coeffs(W, N, T)
        s = score(a, th, T)
        q, kk, td = build_qk(a, th)
        assert np.abs(real_rope_score(q, kk, td, T) - s).max() < 1e-12, N


def test_leakage_falls_with_more_frequencies():
    T, w = 128, 8
    d, W = window(T, w)

    def leak(N):
        a, th = fourier_coeffs(W, N, T)
        return leakage(score(a, th, T), d, w, 8.0)

    assert leak(16) < leak(4)          # more frequency blocks -> tighter window
    assert leak(16) < 0.01             # <1% attention mass outside the window at N=16


def test_dirichlet_out_of_phase():
    T, N = 128, 32
    th = np.pi * np.arange(1, N + 1) / T
    assert abs(np.cos(th * 0).sum() - N) < 1e-9     # in phase at Delta=0 -> peak = N
    assert abs(np.cos(th * 16).sum()) < 1.0         # out of phase far away -> cancels
