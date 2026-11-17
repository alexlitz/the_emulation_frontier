# Round 3, non-relation pass: notes for the relation pass

These are graph-side consequences of fixes made on branch `r3-nonrelation`. graph/model_graph.py was not
edited here.

## W3-1: ACT_LIMIT for (ReLU, swigluoai) needs a host up bias and bounded inputs

`ACT_LIMIT ∋ ("ReLU", "swigluoai")` (graph/model_graph.py, line ~156) is right as a limit only with two
conditions that the relation does not check yet:

- **The host needs an up-projection bias.** The construction that works on the real, clamped swigluoai
  unit `(clip(up, -7, 7) + 1) · g' σ(1.702 g')` with `g' = min(gate, 7)` is Lean `relu_sub_swigluoai`. It
  uses gate row `c z`, up row `a z` with **up bias −1** (which cancels the unit's +1), and down scale
  `1/(a K)` with `K = 7 σ(11.914)`. The route that scales only the gate (`relu_sub_silu`) is refuted:
  the gate clamp caps it at `K/c → 0` (`swigluoai_scaled_gate_vanishes`). With no up bias the unit computes
  `(a z + 1) · …`, and the +1 adds a step of height 1/a at z = 0 (`(a z + 1) K/(a K) = z + 1/a` for z ≥ 7/c, and → 0 for z < 0). No linear map can remove that. So
  the edge should require that the host has a bias on its gate/up projection. gpt-oss always has one
  (`gate_up_proj_bias`). For a MiniMax-style swigluoai host, check whether it carries one. FEATURES
  `bias_out` is the nearest existing flag, but it means an output-side (o_proj / down) bias. The gpt-oss
  unit needs a bias on the gate_up input projection, so a separate `bias_up` predicate (or a per-family
  fact) is cleaner.
- **Bounded inputs.** It needs `|a z| ≤ 6` (the up clamp stays inactive), so `a = 6/R` for an
  input bound R. For any finite R this is a magnitude regime, the same kind as the ±7 clamp note on
  (SiLU, swigluoai). In fp32 a smaller `a` costs relative precision `2^-24/(a|z|)` through the
  `(a z − 1) + 1` cancellation. That stays harmless at the R ≈ 16–20 seen on OPT.
- It is exact for `z ≥ 7/c` and a limit (`c → ∞`) below that. The sup error is `M/c` with `M ≈ 1.75`.

Measured after the fix (act/act_swap_agreement.py with the real clamped unit, `c = 1e5`, `a = 6/max|z|`):
OPT-125m / 350m / 1.3b with a swigluoai host give top-1 1.0, ex32 1.0, ex64 1.0, and KL 4e-12 / 4e-12 / 1e-11.
The old rows (the unclamped, refuted route) gave top-1 1.0 and ex64 1.0. **No verdict changes**, and
the floor goes from 0.0 to 1.75e-5 (still ≤ ACT_FLOOR_MAX = 5e-4).

## W1-1: charge the 2d extra MLP units for a sequential host of a parallel target

Lean `parallel_sub_seq_2x`: norm-free, an L-block parallel model fits a sequential host of width **2d at
every L** (not (L+1)·d). Each block's MLP gets **2d extra units**, the identity pairs
`φ(v) − φ(−v) = v` that move the attention write out of the spare channel. So the graph's `wf = 2` width
factor is right norm-free, but the host's FFN width check should also charge `ffn_target + 2·d_target`
units per block (for a GLU host, 2d gate/up rows; for MoE, per routed expert, or in a shared expert).
Also fix the comment at graph/model_graph.py line ~3792: "costs (L+1) d there, not 2d: Lean
parallel_sub_seq_wide" is wrong. It costs 2d width plus 2d MLP units (parallel_sub_seq_2x). The
decision there rests on the pre-norm failure (`parallel_sub_seq_prenorm_fails`), which stands, so with
`PAR_CARRY_PAYABLE` False only the comment and the capacity-mode charge change.

The docstrings in block/parallel_residual.py and block/prenorm_carry.py are fixed on this branch.
