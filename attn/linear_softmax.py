"""Softmax attention -> linear attention is the ONE true INFINITE-gap in the graph,
made CONCRETE and QUANTIFIED here (ANALYTIC numpy, fp64, no SGD -- closed-form
random-feature maps only).

Softmax attention is  softmax(Q K^T / sqrt d) V.  Dropping the row-normalisation,
the unnormalised weights are the EXPONENTIAL (a.k.a. softmax) kernel
      k(q, k) = exp(q . k)                     (fold the 1/sqrt d into q, k scale)
Linear / kernel attention replaces this with an explicit feature map,
      k(q, k) ~ phi(q) . phi(k),   phi : R^d -> R^m .
The exact feature map of exp(q.k) is INFINITE-dimensional (it is the Gaussian /
exponential kernel), so NO finite m reproduces it exactly. Every other reduction in
this repo hits fp64 0.0 for free; this one cannot. We quantify the approach.

Two closed-form random-feature maps, both UNBIASED estimators of exp(q.k):

  * POSITIVE random features (Performer / FAVOR+):
        phi(x) = exp(W x - ||x||^2 / 2) / sqrt(m),   W ~ N(0, I_d),  m rows.
    Identity: E_w[ exp(w.(q+k) - (||q||^2+||k||^2)/2) ] = exp(q.k), so
    E[phi(q).phi(k)] = exp(q.k) EXACTLY. Features are >= 0, so the estimated kernel
    matrix is >= 0 -- a valid (attention-safe) kernel at every m.

  * PLAIN random Fourier features (RFF, Rahimi-Recht) target the shift-invariant
    RBF kernel exp(-||q-k||^2/2) via  z(x) = sqrt(2/m) cos(W x + b),  b ~ U[0,2pi).
    Since  exp(q.k) = exp(-||q-k||^2/2) * exp((||q||^2+||k||^2)/2),  a per-point
    positive rescale  exp(||x||^2/2)  turns the RBF estimate into an estimate of the
    SAME softmax kernel. Fair head-to-head with FAVOR+.

The MONTE-CARLO rate is O(m^{-1/2}) (variance ~ 1/m): averaging m i.i.d. features.
That exponent is the "infinite currency" -- to halve the error you 4x the features,
forever; the error is bounded away from 0 at any FINITE m. We fit the exponent, show
the residual floor, and contrast with an EXACT reduction (which is 0.0 in fp64).

Run:
  python -m attn.linear_softmax
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# closed-form random-feature maps for the softmax (exponential) kernel exp(q.k)
# --------------------------------------------------------------------------- #


def favor_features(X, W):
    """Performer FAVOR+ positive random features (unbiased for exp(q.k)).

    X: (T, d), W: (m, d) ~ N(0, I).  phi(x) = exp(W x - ||x||^2/2) / sqrt(m) >= 0.
    """
    proj = X @ W.T                                      # (T, m)
    half_sq = (X ** 2).sum(1, keepdims=True) / 2.0      # (T, 1)
    return np.exp(proj - half_sq) / np.sqrt(W.shape[0])


def rff_features(X, W, b):
    """Plain random Fourier features (Rahimi-Recht), rescaled to the softmax kernel.

    z(x) = sqrt(2/m) cos(W x + b) estimates the RBF kernel exp(-||q-k||^2/2);
    multiplying by exp(||x||^2/2) per point retargets it to exp(q.k). Features are
    signed, so the estimated kernel matrix can go NEGATIVE (invalid for attention).
    """
    m = W.shape[0]
    z = np.sqrt(2.0 / m) * np.cos(X @ W.T + b[None, :])  # (T, m)
    corr = np.exp((X ** 2).sum(1, keepdims=True) / 2.0)  # (T, 1)
    return z * corr


def true_kernel(Q, K):
    """The exact softmax kernel matrix, exp(Q K^T) (unnormalised softmax weights)."""
    return np.exp(Q @ K.T)


# --------------------------------------------------------------------------- #
# attention with a feature map:  softmax(QK^T) V  ~  phi(Q) (phi(K)^T V) / Z
# --------------------------------------------------------------------------- #


def softmax_attention(Q, K, V):
    S = Q @ K.T
    S = S - S.max(1, keepdims=True)
    A = np.exp(S)
    A = A / A.sum(1, keepdims=True)
    return A @ V


def feature_attention(Q, K, V, featureQ, featureK):
    """Linear attention with the given random features (numerator / denominator)."""
    num = featureQ @ (featureK.T @ V)                   # (T, dv)
    den = featureQ @ featureK.sum(0, keepdims=True).T   # (T, 1)
    return num / (den + 1e-12)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def _draw_qk(rng, T, d, scale):
    """Unit-norm rows scaled by `scale` (controls the kernel argument q.k range)."""
    Q = rng.standard_normal((T, d)); Q *= scale / np.linalg.norm(Q, axis=1, keepdims=True)
    K = rng.standard_normal((T, d)); K *= scale / np.linalg.norm(K, axis=1, keepdims=True)
    return Q, K


def kernel_error(kind, m, T=64, d=64, scale=1.0, reps=20, seed=0):
    """Mean relative Frobenius error of the estimated kernel matrix over `reps` draws
    of the random features (Q, K held fixed).  kind in {'favor', 'rff'}."""
    rng = np.random.default_rng(seed)
    Q, K = _draw_qk(rng, T, d, scale)
    Ktrue = true_kernel(Q, K)
    denom = np.linalg.norm(Ktrue)
    errs, negfrac = [], []
    for _ in range(reps):
        W = rng.standard_normal((m, d))
        if kind == "favor":
            A = favor_features(Q, W) @ favor_features(K, W).T
        else:
            b = rng.uniform(0.0, 2.0 * np.pi, m)
            A = rff_features(Q, W, b) @ rff_features(K, W, b).T
        errs.append(np.linalg.norm(A - Ktrue) / denom)
        negfrac.append(float((A < 0).mean()))
    return float(np.mean(errs)), float(np.mean(negfrac))


def output_error(kind, m, T=64, d=64, scale=1.0, reps=20, seed=0):
    """Mean relative error of the ATTENTION OUTPUT (not just the kernel) vs exact."""
    rng = np.random.default_rng(seed)
    Q, K = _draw_qk(rng, T, d, scale)
    V = rng.standard_normal((T, d))
    exact = softmax_attention(Q, K, V)
    denom = np.linalg.norm(exact)
    errs = []
    for _ in range(reps):
        W = rng.standard_normal((m, d))
        if kind == "favor":
            fQ, fK = favor_features(Q, W), favor_features(K, W)
        else:
            b = rng.uniform(0.0, 2.0 * np.pi, m)
            fQ, fK = rff_features(Q, W, b), rff_features(K, W, b)
        errs.append(np.linalg.norm(feature_attention(Q, K, V, fQ, fK) - exact) / denom)
    return float(np.mean(errs))


def fit_rate(ms, errs):
    """Fit err ~ C * m^p in log-log; return the exponent p (expect ~ -0.5)."""
    p, logC = np.polyfit(np.log(ms), np.log(errs), 1)
    return float(p), float(np.exp(logC))


# an EXACT reduction for contrast: identity feature map reproduces a LINEAR kernel
# with ZERO error (fp64) -- the whole point is that exp(q.k) canNOT be done this way.
def exact_reduction_residual(T=64, d=64, seed=0):
    """A genuine finite exact reduction: the *linear* kernel q.k IS phi(q).phi(k) with
    phi = identity, so the residual is 0.0 in fp64. Softmax's exp(q.k) is not."""
    rng = np.random.default_rng(seed)
    Q, K = _draw_qk(rng, T, d, 1.0)
    linear_kernel = Q @ K.T
    phiQ, phiK = Q, K                       # identity feature map, m = d, EXACT
    return float(np.abs(phiQ @ phiK.T - linear_kernel).max())


MS = (16, 64, 256, 1024, 4096, 16384)


def main():
    print("SOFTMAX -> LINEAR attention: the exp(q.k) kernel has an INFINITE exact")
    print("feature map, so finite-m linear attention only APPROXIMATES it. Quantified:\n")

    print("1. WIDTH currency -- random features vs exact kernel matrix (rel Frobenius,")
    print("   d=64, unit-norm q,k, mean over 20 feature draws):")
    print(f"   {'m':>7}  {'FAVOR+ (positive)':>18}  {'RFF (Fourier)':>14}")
    favor_errs, rff_errs = [], []
    for m in MS:
        ef, _ = kernel_error("favor", m)
        er, _ = kernel_error("rff", m)
        favor_errs.append(ef); rff_errs.append(er)
        print(f"   {m:>7}  {ef:>18.3e}  {er:>14.3e}")
    pf, _ = fit_rate(MS, favor_errs)
    pr, _ = fit_rate(MS, rff_errs)
    print(f"   fitted exponent  FAVOR+ p={pf:+.3f}   RFF p={pr:+.3f}   (theory -0.500,")
    print("   the Monte-Carlo 1/sqrt(m) rate; halve the error -> 4x the features).\n")

    print("2. NEVER 0 at finite m (the 'infinite currency'):")
    print("   error keeps shrinking but stays > 0 at every finite m -- residual variance.")
    ef_big, _ = kernel_error("favor", 65536, reps=8)
    print(f"     FAVOR+ at a HUGE m=65536:            {ef_big:.3e}   (still > 0)")
    print(f"     extrapolate to fp64 0 (~1e-16): m ~ {int((favor_errs[0]*16**0.5/1e-16)**2):.1e}"
          "  (astronomically infeasible)")
    resid = exact_reduction_residual()
    print(f"   CONTRAST -- an EXACT reduction (linear kernel q.k, identity features):")
    print(f"     residual = {resid:.1e}  == fp64 0.0.  THAT is a true finite reduction;")
    print("     softmax's exp(q.k) is the one edge that is not.\n")

    print("3. FAVOR+ vs RFF -- which is the better construction? Depends on score scale:")
    print(f"   {'scale s (||q||=||k||=s)':>26}  {'FAVOR+':>10}  {'RFF':>10}  {'RFF neg%':>9}")
    for s in (1.0, 2.0, 3.0):
        ef, _ = kernel_error("favor", 1024, scale=s)
        er, negf = kernel_error("rff", 1024, scale=s)
        print(f"   {s:>26.1f}  {ef:>10.3e}  {er:>10.3e}  {100*negf:>8.1f}%")
    print("   -> at small scale the two tie; as q.k grows RFF makes the kernel NEGATIVE")
    print("      (invalid attention weights) at up to ~50% of entries and eventually blows")
    print("      up, while FAVOR+ stays >= 0 by construction -- the better/safer feature")
    print("      map, exactly Performer's argument for POSITIVE features.\n")

    print("4. Attention OUTPUT error tracks the kernel error (~1/sqrt m), FAVOR+ (m sweep):")
    out = [output_error("favor", m) for m in (64, 256, 1024, 4096)]
    print("     m=64/256/1024/4096 -> " + " / ".join(f"{e:.3e}" for e in out))
    po, _ = fit_rate((64, 256, 1024, 4096), out)
    print(f"     fitted output-error exponent p={po:+.3f}  (~ -0.5, same MC wall).")


if __name__ == "__main__":
    main()
