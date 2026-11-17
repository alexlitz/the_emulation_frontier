# Proofs

Lean 4 + Mathlib proofs of the constructions and walls the emulation relation uses, over exact real
arithmetic (no floating point). Each theorem (namespace `EP`) states that a host component with
constructed weights equals the target component on every input (exact), converges to it (limit), stays
within a stated distance (bound), or cannot match it (impossibility).

- `lean/`: the Lean project. `lean/EmulationProofs/*.lean` (44 files) is the library, `lean/EmulationProofs.lean`
  its root, `lean/NegativeControl/FalseVariants.lean` the negative control, `lean/check.sh` the checker.
- `PROOFS.md`: the index, theorem names per rule with their status (PROVED exact / limit / bound,
  PROVED impossible, REFUTED, OPEN).

## Toolchain

`leanprover/lean4:v4.36.0-rc2` (`lean/lean-toolchain`), Mathlib at tag `v4.36.0-rc2`, pinned in
`lean/lake-manifest.json`.

## Running the check

```bash
# elan installs the pinned toolchain on first use
cd proofs/lean
lake exe cache get      # Mathlib's prebuilt .olean files
./check.sh
```

`check.sh`:

1. builds the library (`lake build`);
2. rejects any `sorry`, `admit`, `axiom` or `native_decide` in `EmulationProofs/*.lean`;
3. runs `#print axioms` on every `theorem` / `lemma` and fails on any axiom outside
   `propext`, `Classical.choice`, `Quot.sound`;
4. runs `NegativeControl/FalseVariants.lean` (false variants of the polarisation identity and of the
   partial-RoPE base, given the true proofs' scripts) and requires an error at each.

Current output: 740 theorems and lemmas, all on the three standard axioms; the negative control rejected
with 2 errors.

## PROOFS.md index

- The cookbook table
- Swing table, "other direction", and other rows
- Capacity rules used by the relation
- Every allowed rule (with the walls of the edge-case audit)
- Limits at the whole-model level
- The eps lever: host norms switched off
- KV-head packing
- MoE ← MoE routing
- Round-2 rules
- fr-mixer: mixer / recurrence walls paid
- Where the formal statement needs a condition the cookbook does not state
- Pre-norm rescue attempts
- Residual streams and norms paid
- Target norms
- Constants through a bias-free pre-norm
- Not done, with the reason

## Where the repo uses them

`verify/edge_table.py` checks that every theorem name it cites exists in `lean/EmulationProofs/`
(`verify/EDGE_JUSTIFICATION.md`). `verify/BRIDGES.md` and `tests/test_lean_bridges*.py` check theorems
against the repo functions that build the constructions. `verify/coverage_report.py` reads `PROOFS.md` and
the Lean sources for metrics M4, M5 and M12. `graph/post_numbers.py` counts the theorems (`lean_theorems`).
