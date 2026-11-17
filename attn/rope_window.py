"""Local (windowed) attention ⊆ global attention with RoPE — by Fourier analysis.

The blog mused that "RoPE on some extra dims can very closely approximate a hard
local mask." Here is the actual construction. A RoPE 2-D block with angular frequency
theta contributes to the query·key score a term  a * cos(theta * Delta),
Delta = n - m the relative position (an exact fact of the rotation:
(R(theta n) q)·(R(theta m) k) = a*cos(theta*(n-m)) when q=(sqrt|a|,0), k=(sign(a)sqrt|a|,0)).

Stack N frequency blocks theta_k = pi k / T and the score becomes a cosine series
    s(Delta) = sum_k a_k cos(theta_k Delta).
Choose a_k = the Fourier-cosine coefficients of the target window boxcar
    W(Delta) = 1 for Delta <= w, else 0
and s(Delta) is the truncated Fourier series of that window: a Dirichlet kernel,
sharply peaked at Delta = 0 and — because the high frequencies go OUT OF PHASE —
collapsing to ~0 outside the window. Feed it through softmax (magnitude beta to
sharpen) and the head attends locally.

Currency: WIDTH (N frequency blocks = 2N head dims) + MAGNITUDE (beta). It is a
FLOOR, not exact: Gibbs ringing at the window edge never fully vanishes (~9% overshoot
that only narrows with N) — but the softmax attention LEAKAGE outside the window still
falls fast in N.

Run:  python -m act.rope_window   (or python -m attn.rope_window)
"""
import numpy as np

relu = lambda x: np.maximum(0.0, x)


def window(T, w):
    d = np.arange(T)
    return d, (d <= w).astype(float)                       # causal window: attend Delta in [0, w]


def fourier_coeffs(W, N, T):
    """a_k, theta_k for the truncated Fourier-cosine series of W over Delta in [0, T)."""
    d = np.arange(T)
    theta = np.pi * np.arange(N + 1) / T
    a = np.array([(W * np.cos(tk * d)).mean() * (1 if k == 0 else 2) for k, tk in enumerate(theta)])
    return a, theta


def score(a, theta, T):
    d = np.arange(T)
    return (a[:, None] * np.cos(theta[:, None] * d[None, :])).sum(0)    # s(Delta)


def build_qk(a, theta):
    """Actual RoPE query/key blocks whose rotated dot-product reproduces sum_k a_k cos(theta_k Delta).
    Block k: q_k = (sqrt|a_k|, 0), k_k = (sign(a_k) sqrt|a_k|, 0). Returns (q, k, theta_per_dim)."""
    s = np.sqrt(np.abs(a))
    q = np.stack([s, np.zeros_like(s)], 1).ravel()                     # (2(N+1),)
    kk = np.stack([np.sign(a) * s, np.zeros_like(s)], 1).ravel()
    theta_dim = np.repeat(theta, 2)                                    # per-dim angular freq
    return q, kk, theta_dim


def rope_apply(vec, pos, theta_dim):
    """Rotate each 2-D block of `vec` by theta*pos (standard RoPE)."""
    out = vec.astype(float).copy()
    c, s = np.cos(theta_dim * pos), np.sin(theta_dim * pos)
    x0, x1 = out[0::2].copy(), out[1::2].copy()          # copy: slices are views, avoid in-place aliasing
    out[0::2] = x0 * c[0::2] - x1 * s[0::2]
    out[1::2] = x0 * s[0::2] + x1 * c[0::2]
    return out


def real_rope_score(q, kk, theta_dim, T):
    """Score(n, m) from ACTUAL RoPE rotation, as a function of Delta=n-m (take n=T-1)."""
    n = T - 1
    qn = rope_apply(q, n, theta_dim)
    return np.array([qn @ rope_apply(kk, n - dd, theta_dim) for dd in range(T)])


def leakage(s, d, w, beta):
    e = np.exp(beta * (s - s.max()))
    p = e / e.sum()
    return float(p[d > w].sum())                          # attention mass OUTSIDE the window


def main():
    T, w = 128, 8
    d, W = window(T, w)
    print(f"local window (w={w}, T={T}) via N RoPE frequency blocks:")
    print(f"{'N':>4} {'max|s-W| (Gibbs)':>16} {'softmax leak (beta=8)':>22} {'real-RoPE match':>16}")
    for N in (2, 4, 8, 16, 32, 64):
        a, theta = fourier_coeffs(W, N, T)
        s = score(a, theta, T)
        q, kk, td = build_qk(a, theta)
        match = np.abs(real_rope_score(q, kk, td, T) - s).max()       # exact: RoPE == cosine series
        print(f"{N:>4} {np.abs(s - W).max():>16.3f} {leakage(s, d, w, 8.0):>22.4f} {match:>16.1e}")
    print("\nDirichlet 'out of phase' (equal weights): Sum_k cos(theta_k Delta) peaks at 0, cancels away:")
    N = 32
    th = np.pi * np.arange(1, N + 1) / T
    for dd in (0, 1, 2, 4, 8, 32):
        print(f"  Delta={dd:>3}:  Sum cos = {np.cos(th * dd).sum():+6.1f}")


if __name__ == "__main__":
    main()
