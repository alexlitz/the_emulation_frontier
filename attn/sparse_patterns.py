"""Sparse-attention patterns (strided, dilated, block, global, sinks) as
restrictions of full attention -- currency = DEPTH (the pattern's graph diameter).

Every fixed sparse attention pattern is a MASK: a subset of full attention's
edges. So `pattern ⊆ full` for free (full masks down to any pattern, exact). The
reverse -- emulating FULL reach with a sparse pattern -- costs the number of layers
for information to hop across the pattern, i.e. its graph DIAMETER (with residual
connections). Cheap patterns (strided+local, global tokens) reach full in ~2 hops;
naive local needs O(N/W). This generalizes local→full to any mask.

Reach after L residual layers = (mask | I)^L (boolean). We report the diameter
(min L that covers all causal pairs) and per-layer density (FLOPs proxy).

Run:  python -m attn.sparse_patterns
"""
from __future__ import annotations

import numpy as np


def causal(N):
    i, j = np.indices((N, N))
    return j <= i


def local(N, W):
    i, j = np.indices((N, N))
    return (j <= i) & (j > i - W)


def strided(N, k):                      # attend to every k-th token back (Sparse Transformer)
    i, j = np.indices((N, N))
    return (j <= i) & ((i - j) % k == 0)


def block_sparse(N, b):                 # blockwise local: own block + previous block (BigBird-ish)
    i, j = np.indices((N, N))
    return (j <= i) & (i // b - j // b <= 1)


def global_tokens(N, g, W=1):           # first g tokens are global (BigBird/Longformer) + self
    i, j = np.indices((N, N))
    return ((j <= i) & (j > i - W)) | ((j < g) & (j <= i)) | ((i < g) & (j <= i))


def sink(N, s, W):                      # StreamingLLM: first s "sink" tokens + local window
    i, j = np.indices((N, N))
    return ((j <= i) & (j > i - W)) | ((j < s) & (j <= i))


def diameter(mask, N, cap=None):
    full = causal(N)
    R = mask | np.eye(N, dtype=bool)                 # residual: self-loop
    step = R.copy()
    for L in range(1, (cap or N) + 1):
        if (R | ~full).all():                        # R covers every causal pair
            return L
        R = (R.astype(np.int32) @ step.astype(np.int32)) > 0
    return None


def run(N=64):
    W = 8; k = 8; s = 4; g = 4
    pats = {
        "full (causal)":              causal(N),
        f"local W={W}":               local(N, W),
        f"strided k={k} (alone)":     strided(N, k),
        f"local+strided (√N={W})":    local(N, W) | strided(N, W),
        f"block-sparse b={W}":        block_sparse(N, W),
        f"global g={g} + local":      global_tokens(N, g, W),
        f"sinks s={s} + local W={W}": sink(N, s, W),
    }
    print(f"sparse attention patterns (N={N}), all subset of full by masking:")
    print(f"     {'pattern':26} {'density':>8}  {'depth to reach full':>20}")
    for name, m in pats.items():
        m = m & causal(N)
        dens = m.sum() / causal(N).sum()
        dia = diameter(m, N)
        d = "1  (is full)" if name.startswith("full") else \
            "never*" if dia is None else f"{dia} layers"
        print(f"     {name:26} {dens:>7.0%}  {d:>20}")
    print("\n   -> every pattern is `⊆ full` (mask, free); FULL is recovered at DEPTH = the")
    print("      pattern's graph diameter. local+strided factorizes to ~2 hops (Sparse")
    print("      Transformer); naive local/block needs O(N/W); *strided ALONE never reaches")
    print("      full (residue-class gaps — it needs a distance-1/local edge).")
    print("      Note: in CAUSAL decoders 'global' tokens don't bridge forward, so global/")
    print("      sink patterns stay local-limited — the O(1) BigBird diameter is bidirectional-only.")


if __name__ == "__main__":
    run()
