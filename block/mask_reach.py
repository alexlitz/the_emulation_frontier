"""Local / sliding / sparse attention are NATIVE SUBSETS of full attention — the
mask is a subset of full attention's edges, so `pattern ⊆ full` for free (full masks
down to any pattern, EXACTLY). The reverse — recovering full attention's wide
receptive field from a narrow one — costs DEPTH, and the amount is a pure COMBINATORIAL
fact about boolean reachability. No SGD, no fitting: everything here is `(I ∨ A)^L`.

  * SLIDING WINDOW width w (causal: self + previous w-1 tokens). One masked layer
    reaches w tokens. With a residual stream, stacking L layers COMPOSES the reach:
    a token at the end of an L-deep stack can be influenced by a token that is a chain
    of ≤ L window-hops away. Each hop advances the front by (w-1), so

        reach(L) = L*(w-1) + 1                          [verified by mask composition]

    To cover a context of length T (i.e. match global attention's reach over T tokens)
    you need reach ≥ T, hence the exact depth-to-global law

        L*  =  ⌈ (T-1) / (w-1) ⌉    (the CONSTANT is 1; slope 1/(w-1)).

    Small window -> more depth: window and depth trade off on a hyperbola.

  * SPARSE PATTERNS (strided, block-local, dilated, local+strided). Reach = (I ∨ A)^L
    boolean; the depth to reach FULL causal connectivity is the graph DIAMETER of the
    pattern. Concrete closed forms we confirm by composition:
        block-local (own + previous block, size b):  diameter = (N-1)//b
        local(w) + strided(w) (√N factorization):    diameter = 2   (Sparse Transformer)
        strided ALONE / dilated ALONE:               NEVER (residue-class gaps: a token
            at a distance not divisible by the stride is unreachable at any depth).

The currency throughout is DEPTH; the "never" cases are the honest edge — a pattern
whose edges all share a common stride can only ever reach a sublattice, so it needs at
least one distance-1 (local) edge to be completable at any finite depth.

Run:  python -m block.mask_reach
"""
from __future__ import annotations

from math import ceil

import numpy as np

# --------------------------------------------------------------------------- masks
# All masks are causal (decoder): row i = query at position i, col j = key j, j <= i.


def causal(N):
    i, j = np.indices((N, N))
    return j <= i


def sliding(N, w):
    """Causal sliding window of width w: self + previous w-1 tokens (w total)."""
    i, j = np.indices((N, N))
    return (j <= i) & (j > i - w)


def strided(N, s):
    """Attend to every s-th token back (Sparse-Transformer strided head)."""
    i, j = np.indices((N, N))
    return (j <= i) & ((i - j) % s == 0)


def block_local(N, b):
    """Blockwise local: own block + the previous block (block size b)."""
    i, j = np.indices((N, N))
    return (j <= i) & (i // b - j // b <= 1)


def dilated(N, d, w):
    """Dilated window: w taps spaced d apart, causal (a gappy window)."""
    i, j = np.indices((N, N))
    return (j <= i) & ((i - j) % d == 0) & (i - j < d * w)


# ----------------------------------------------------------------- reachability
# A residual stream keeps a self-loop, so one layer's reach is (I ∨ A). Stacking L
# layers composes it: (I ∨ A)^L (boolean). This is EXACTLY where information can flow.


def reach(A, L):
    """(I ∨ A)^L, boolean. Which (query, key) pairs can influence each other in L layers."""
    R = A | np.eye(A.shape[0], dtype=bool)
    out = R.copy()
    for _ in range(1, L):
        out = (out.astype(np.int64) @ R.astype(np.int64)) > 0
    return out


def reach_count(A, L, row=None):
    """How many tokens the query at `row` (default: the last, widest) reaches in L layers."""
    R = reach(A, L)
    row = R.shape[0] - 1 if row is None else row
    return int(R[row].sum())


def diameter(A, N, cap=None):
    """Min depth L for (I ∨ A)^L to cover the full causal mask (the graph diameter).
    Returns None if the pattern can never reach full connectivity (residue-class gaps)."""
    C = causal(N)
    R = A | np.eye(N, dtype=bool)
    out = R.copy()
    L = 1
    cap = cap or (2 * N)
    while not (out | ~C).all():                 # out covers every causal pair?
        out = (out.astype(np.int64) @ R.astype(np.int64)) > 0
        L += 1
        if L > cap:
            return None
    return L


def depth_to_global(A, N, cap=None):            # alias: diameter measured on a length-N context
    return diameter(A, N, cap)


def depth_to_global_formula(T, w):
    """Exact sliding-window law:  L* = ⌈(T-1)/(w-1)⌉  (w >= 2)."""
    return ceil((T - 1) / (w - 1))


# --------------------------------------------------------------------------- main


def main():
    T = 64
    print(f"MASK REDUCTIONS — local/sliding/sparse ⊆ full attention; currency = DEPTH.")
    print(f"reach = (I ∨ A)^L boolean composition; context length T = {T}.\n")

    print("1. SLIDING WINDOW reach composes as L*(w-1)+1  (checked by mask composition):")
    print(f"     {'w':>3}  {'L':>2}  {'reach(composed)':>15}  {'L*(w-1)+1':>10}")
    for w in (2, 3, 5, 8):
        for L in (1, 2, 3, 4):
            rc = reach_count(sliding(T, w), L)
            pred = min(T, L * (w - 1) + 1)
            flag = "" if rc == pred else "  <-- MISMATCH"
            print(f"     {w:>3}  {L:>2}  {rc:>15}  {pred:>10}{flag}")
        print()

    print(f"2. WINDOW ↔ DEPTH-TO-GLOBAL law over context T={T}")
    print("   (depth for a window to reach full/global attention's receptive field):")
    print(f"     {'window w':>9}  {'depth L* (composed)':>19}  {'⌈(T-1)/(w-1)⌉':>14}  {'density/layer':>13}")
    for w in (2, 3, 5, 8, 9, 16, 33, 64):
        Lc = depth_to_global(sliding(T, w), T)
        Lf = depth_to_global_formula(T, w)
        m = sliding(T, w) & causal(T)
        dens = m.sum() / causal(T).sum()
        flag = "" if Lc == Lf else "  <-- MISMATCH"
        print(f"     {w:>9}  {Lc:>19}  {Lf:>14}  {dens:>12.0%}{flag}")
    print("   -> small window buys cheap layers but needs MORE of them: L* ~ (T-1)/(w-1).")
    print("      Full attention is the w=T corner (depth 1). The product (window x depth) to")
    print("      cover T is roughly constant — a hyperbola, the exact reach/depth tradeoff.\n")

    print(f"3. SPARSE-PATTERN DIAMETERS (depth to reach full connectivity, N={T}):")
    N = T
    b = 8
    s = 8
    pats = {
        f"sliding w=8":                 sliding(N, 8),
        f"block-local b={b}":           block_local(N, b),
        f"local(8)+strided(8) (√N)":    sliding(N, 8) | strided(N, 8),
        f"strided s={s} (alone)":       strided(N, s),
        f"dilated d=2,w=8 (alone)":     dilated(N, 2, 8),
    }
    print(f"     {'pattern':28} {'density':>8}  {'diameter (depth to full)':>24}")
    for name, m in pats.items():
        m = m & causal(N)
        dens = m.sum() / causal(N).sum()
        dia = diameter(m, N)
        d = "NEVER (residue gaps)" if dia is None else f"{dia}"
        print(f"     {name:28} {dens:>7.0%}  {d:>24}")
    print(f"   closed forms (all verified by composition):")
    print(f"     block-local b:  diameter = (N-1)//b            -> b={b}: {(N - 1) // b}")
    print(f"     local(w)+strided(w):  diameter = 2  (Sparse Transformer √N factorization)")
    print(f"     strided/dilated ALONE: NEVER — every edge shares a stride, so the reachable")
    print(f"       set is a residue class; a distance-1 (local) edge is required to complete it.")


if __name__ == "__main__":
    main()
