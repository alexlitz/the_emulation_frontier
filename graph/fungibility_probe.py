"""Microcosm probes: which capacity "fungibilities" actually facilitate an EXACT emulation,
and what they really cost — tested for a layer or a few, so the frontier only ever credits a
trade we can actually build.

The load-bearing finding (it corrects an over-claim): a wide FFN is NOT a free deep stack of
narrow ones. In a residual stack each sub-FFN reads the *accumulated* residual, not the
original block input, so naive depth-splitting diverges. It becomes exact only when every
spread sublayer reads the ORIGINAL input — which costs reserved channels (~2x hidden width) to
carry it. So FFN/MoE width<->depth is a **depth + 2x-width** trade, not free depth; almost no
model has the spare width, which is why crediting it moves the frontier by only one model.

Query-head COUNT (independent softmaxes) and KV cache (bytes of memory) have no such trade —
they are genuinely non-fungible.

Run:  python -m graph.fungibility_probe
"""
from __future__ import annotations

import numpy as np

_silu = lambda z: z / (1 + np.exp(-np.clip(z, -30, 30)))


def ffn_width_to_depth(L=3, d=6, H=12, n=5, seed=0):
    """Split one wide FFN (hidden H) into L narrow FFNs (hidden H/L). Returns
    (naive_err, carried_err, width_cost) — naive residual stack vs input-carried."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    W1 = rng.standard_normal((d, H)) * 0.6
    W2 = rng.standard_normal((H, d)) * 0.6
    wide = x + _silu(x @ W1) @ W2                     # target: one wide FFN (residual)
    h = H // L
    parts = [(W1[:, i * h:(i + 1) * h], W2[i * h:(i + 1) * h, :]) for i in range(L)]

    r = x.copy()                                      # naive: each sublayer reads accumulated r
    for w1, w2 in parts:
        r = r + _silu(r @ w1) @ w2
    naive = float(np.abs(r - wide).max())

    acc = x.copy()                                    # input-carried: each reads the frozen x
    for w1, w2 in parts:
        acc = acc + _silu(x @ w1) @ w2
    carried = float(np.abs(acc - wide).max())
    return naive, carried, d                          # width_cost = +d reserved channels


def moe_experts_to_depth(E=4, d=6, He=5, n=5, seed=1):
    """A shared-MoE (E summed experts) as E masked residual sublayers. Returns
    (naive_err, carried_err)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    gates = _silu(rng.standard_normal((n, E)))
    Ws = [(rng.standard_normal((d, He)) * 0.6, rng.standard_normal((He, d)) * 0.6) for _ in range(E)]
    moe = x + sum(gates[:, e:e + 1] * (_silu(x @ w1) @ w2) for e, (w1, w2) in enumerate(Ws))

    r = x.copy()
    for e, (w1, w2) in enumerate(Ws):
        r = r + gates[:, e:e + 1] * (_silu(r @ w1) @ w2)
    naive = float(np.abs(r - moe).max())

    acc = x.copy()
    for e, (w1, w2) in enumerate(Ws):
        acc = acc + gates[:, e:e + 1] * (_silu(x @ w1) @ w2)
    carried = float(np.abs(acc - moe).max())
    return naive, carried


def heads_width_to_depth(H=6, dh=2, n=5, seed=0):
    """Spread H attention heads over depth (each head-sublayer adds its own contribution).
    Returns (naive_err, carried_err, carry_full, carry_per_head) — the win of carrying only a
    head's low-dim projection (3*dh) instead of the full input (d = H*dh)."""
    rng = np.random.default_rng(seed)
    d = H * dh
    x = rng.standard_normal((n, d))
    WQ = [rng.standard_normal((d, dh)) * 0.6 for _ in range(H)]
    WK = [rng.standard_normal((d, dh)) * 0.6 for _ in range(H)]
    WV = [rng.standard_normal((d, dh)) * 0.6 for _ in range(H)]
    WO = [rng.standard_normal((dh, d)) * 0.6 for _ in range(H)]

    def head(h, inp):
        Q, K, V = inp @ WQ[h], inp @ WK[h], inp @ WV[h]
        s = Q @ K.T / np.sqrt(dh)
        s = s - s.max(1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(1, keepdims=True)
        return a @ V @ WO[h]

    full = x + sum(head(h, x) for h in range(H))      # H-head MHA (residual)
    r = x.copy()                                      # naive: reads accumulated residual
    for h in range(H):
        r = r + head(h, r)
    naive = float(np.abs(r - full).max())
    acc = x.copy()                                    # each head-sublayer reads the frozen x
    for h in range(H):
        acc = acc + head(h, x)
    carried = float(np.abs(acc - full).max())
    return naive, carried, d, 3 * dh                  # carry the whole input vs one head's Q/K/V


def ffn_subset_to_depth(d=8, H=32, k=4, n=5, seed=0):
    """Spread only k of a wide FFN's H units over depth (the rest stay wide in the base layer).
    Each unit reads x through ONE row of W1, so its pre-activation is a single scalar -> carrying
    k units costs k channels, NOT 2x the whole width. Returns (err, carry_channels=k)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    W1 = rng.standard_normal((d, H)) * 0.5
    W2 = rng.standard_normal((H, d)) * 0.5
    Z = x @ W1                                        # pre-activations (one scalar per unit)
    full = x + _silu(Z) @ W2
    acc = x + _silu(Z[:, :H - k]) @ W2[:H - k, :]     # base layer keeps H-k units (reads x)
    for i in range(H - k, H):                         # k spread units read their carried pre-act
        acc = acc + _silu(Z[:, i:i + 1]) @ W2[i:i + 1, :]
    return float(np.abs(acc - full).max()), k


def residual_rotation(d=8, n=6, seed=0):
    """A transformer only ADDS to the residual (h -> h + f(h)); it has no native rotate. To
    install R@x you add f(x)=(R-I)@x -- the -I CANCELS the incoming residual -- realized as a
    self-attention layer whose value·output = R-I. Returns (rot_err, rank_needed). Exact for
    linear attention; softmax reaches self-attention (matrix = I) only in the large-logit limit."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))  # a random rotation
    M = R - np.eye(d)
    r = int(np.linalg.matrix_rank(M))
    U, S, Vt = np.linalg.svd(M)
    WV, WO = U[:, :r] * S[:r], Vt[:r, :]              # M = WV @ WO (rank r = head-dims needed)
    out = x + (x @ WV) @ WO                           # self-attn returns x; V,O apply M
    return float(np.abs(out - x @ R).max()), r


def main():
    n, c, w = ffn_width_to_depth()
    mn, mc = moe_experts_to_depth()
    hn, hc, cfull, cper = heads_width_to_depth()
    se, sk = ffn_subset_to_depth()
    re, rr = residual_rotation()
    print("microcosm fungibility probes (max|err| over a few tokens):\n")
    print(f"  FFN width->depth   naive stack   {n:8.2e}   (each sublayer reads the accumulated residual -> FAILS)")
    print(f"                     input-carried {c:8.2e}   (EXACT, at +{w} reserved channels ~ 2x width to carry x)")
    print(f"  MoE experts->depth naive stack   {mn:8.2e}   (FAILS)")
    print(f"                     input-carried {mc:8.2e}   (EXACT, at ~2x width)")
    print(f"  heads->depth       naive stack   {hn:8.2e}   (FAILS)")
    print(f"                     input-carried {hc:8.2e}   (EXACT; carry a head's projection = 3*dh = {cper}")
    print(f"                                                vs the whole input d = {cfull} -> cheaper for < H/3 heads)")
    print("  KV cache           no trade       (bytes of memory; can't be spread over depth)")
    print(f"\n  FFN SUBSET (k of H units):  spread {sk} units, err {se:.2e}   -> carry only k={sk} channels")
    print("     (each unit reads x through ONE row of W1 -> one scalar; NOT a flat 2x width).")
    print(f"  RESIDUAL ROTATION:  install R@x, err {re:.2e}   -> 1 self-attn layer, rank(R-I)={rr} head-dims")
    print("     the residual is TIED across layers (only added to); reusing carry channels means")
    print("     CLEARING them = a rotation, cost = the cleared dim in head-dims (exact for linear")
    print("     attention; softmax self-attention is a large-logit / magnitude limit).")
    print("\n  => carry cost = the operator's READ dimension (FFN unit = 1 scalar, head = d_head, MLA")
    print("     = rank); but the residual tie adds a per-reuse rotation/clear (attention + a")
    print("     magnitude limit). Cheap for a few units/heads; prohibitive for the big records.")


if __name__ == "__main__":
    main()
