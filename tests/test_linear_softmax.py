"""Softmax attention -> linear attention is the ONE true infinite-gap edge, QUANTIFIED
(ANALYTIC numpy, fp64, no SGD -- closed-form random features only):

  * random features approximate the exp(q.k) softmax kernel with error ~ 1/sqrt(m)
    (Monte-Carlo rate), so error DECREASES with m but is bounded away from 0 at any
    finite m -- the honest content of the "infinite currency";
  * a genuine EXACT reduction (linear kernel q.k via identity features) hits fp64 0.0;
  * positive FAVOR+ features stay >= 0 (attention-safe); plain RFF can go negative.

Run:  python -m pytest tests/test_linear_softmax.py -q
"""
import numpy as np

from attn.linear_softmax import (MS, exact_reduction_residual, favor_features,
                                 fit_rate, kernel_error, output_error,
                                 rff_features, true_kernel)


def test_favor_features_are_unbiased_for_exp_kernel():
    # E[phi(q).phi(k)] = exp(q.k); average over a huge m -> matches the exact kernel.
    rng = np.random.default_rng(0)
    d = 32
    Q = rng.standard_normal((4, d)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    K = rng.standard_normal((4, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    W = rng.standard_normal((400000, d))
    est = favor_features(Q, W) @ favor_features(K, W).T
    # unbiased MC estimate: small at large m, but NOT exact (residual variance ~1/m).
    assert np.abs(est - true_kernel(Q, K)).max() < 2e-2


def test_favor_features_are_nonnegative():
    # positivity is the whole point of FAVOR+: the estimated kernel matrix is >= 0.
    rng = np.random.default_rng(1)
    d = 64
    Q = rng.standard_normal((8, d)); K = rng.standard_normal((8, d))
    W = rng.standard_normal((128, d))
    fQ, fK = favor_features(Q, W), favor_features(K, W)
    assert (fQ >= 0).all() and (fK >= 0).all()
    assert (fQ @ fK.T >= 0).all()


def test_error_decreases_with_m():
    # more features -> smaller error, monotonically, for both feature maps.
    for kind in ("favor", "rff"):
        errs = [kernel_error(kind, m)[0] for m in MS]
        assert all(a > b for a, b in zip(errs, errs[1:])), (kind, errs)


def test_monte_carlo_rate_is_one_over_sqrt_m():
    # fitted log-log exponent ~ -0.5 (the O(m^{-1/2}) MC wall) for both maps.
    for kind in ("favor", "rff"):
        errs = [kernel_error(kind, m)[0] for m in MS]
        p, _ = fit_rate(MS, errs)
        assert -0.60 < p < -0.40, (kind, p)


def test_not_exact_at_finite_m_bounded_away_from_zero():
    # at a fixed MODEST m the error is well away from 0 -- NOT an exact reduction.
    err_modest, _ = kernel_error("favor", 256)
    assert err_modest > 1e-2                          # nowhere near fp64 0
    # even at a large m the residual is still orders of magnitude above fp64 zero.
    err_large, _ = kernel_error("favor", 16384)
    assert err_large > 1e-4
    assert err_large < err_modest                     # but shrinking


def test_exact_reduction_contrast_hits_fp64_zero():
    # the honest contrast: a TRUE finite reduction (linear kernel, identity features)
    # is 0.0 in fp64 -- softmax's exp(q.k) is the one edge that never is.
    resid = exact_reduction_residual()
    assert resid == 0.0
    # and the softmax random-feature residual is many orders of magnitude larger.
    assert kernel_error("favor", 256)[0] > 1e6 * max(resid, 1e-16)


def test_rff_can_go_negative_at_large_scale():
    # plain Fourier features produce an INVALID (signed) kernel as scores grow;
    # FAVOR+ never does. This is why Performer uses positive features.
    _, neg_favor = kernel_error("favor", 1024, scale=3.0)
    _, neg_rff = kernel_error("rff", 1024, scale=3.0)
    assert neg_favor == 0.0
    assert neg_rff > 0.1


def test_output_error_also_decays():
    # the attention OUTPUT (not just the kernel) improves with m, same MC rate.
    e_small = output_error("favor", 64)
    e_big = output_error("favor", 4096)
    assert e_big < e_small / 2
