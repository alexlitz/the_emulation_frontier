# Full coverage: the definition, the metrics, and the plan

"Full coverage" means every claim the post makes is backed at every layer below, and a script reports each metric. Target: 100% on every row, or an explicit, justified exclusion list.

## The metrics (each computed by `verify/coverage_report.py`, to be built)

| # | metric | numerator / denominator | source of truth | at plan time (current values: `verify/COVERAGE.md`) | owner |
|---|---|---|---|---|---|
| M1 | **Model types reviewed** | model types whose code diff vs parent is fully classified / all model types in the snapshot | `verify/sweep/` | ~150 / 290 (family reviews) | code-diff sweep (running) |
| M2 | **Feature variants handled** | variants read ∧ (host branch ∨ wall) ∧ (rule as target) ∧ (rule as host) / all variants | the sweep's coverage matrix | — | code-diff sweep (running) |
| M3 | **Rules proved** | allowed relation rules with a Lean theorem or a recorded measurement / all allowed rules | `EDGE_JUSTIFICATION.md` (generated from code) | 25/38 slot + 10/19 capacity (pre-audit) | Z3 update (running), then Lean |
| M4 | **Walls classified** | walls labelled proved-impossible or open-with-measurement / all walls | PROOFS.md walls table | most | Lean |
| M5 | **Theorems bridged to code** | proved theorems with a test on the repo function that builds them / all construction theorems | `BRIDGES.md` | 76 / ~330 | bridges round 2 (to schedule) |
| M6 | **Reduction types with an end-to-end instance** | rule types with a non-skipped catalog test that builds and verifies / all rule types | `tests/test_reduction_catalog.py` + meta-test | 0 (queued) | spot checks + catalog (queued) |
| M7 | **Random edges verified** | spot-checked edges passing / sampled (≥ 200, stratified) | `verify/spot/` | — | spot checks (queued) |
| M8 | **Edges directly buildable** | headline edges the emulate script builds / all headline edges | emulate.support | ~1,445 / ~14k (pre-round-2) | emulate v3 (to schedule from the spot-check ranking) |
| M9 | **Relation model fidelity** | pairs where the Z3 model equals emul_matrix / all pairs × modes | `z3_relation.py` | ~95% (stale) | Z3 update (running) |
| M10 | **Relation properties proved** | {reflexive, transitive, monotone, join} × 4 modes proved by Z3 on the final code | `z3_relation.py` | 0 / 16 on final code | Z3 update (running) |
| M11 | **Models carried end to end** | models whose family loader runs against its own code / all models | `loader_coverage` | 3,531 / 3,906 | the rest needs remote code (TEF_ALLOW_REMOTE_CODE=1) |
| M12 | **Limits fully grounded** | limit rules with (a) a Lean proof that the limit exists and equals the target, (b) a Lean convergence rate, and (c) the setting used measured on real models at the agreement standard (top-1 ≥ 99%, ≥ 90% identical 32-token greedy continuations, fp32) / all limit rules | PROOFS.md + the agreement measurements | redefined 2026-11-14 | Lipschitz measurement done (lipschitz-2): worst-case whole-model bounds (layer products 10^23–10^39) cannot certify practical settings, so settings are certified by measurement |
| M13 | **Zoo behaviour tested** | page features with an automated browser test / all page features | `tests/test_zoo_browser.py` | 0 | zoo tests (running) |
| M14 | **Post numbers verified** | draft numbers matching computed values (strict checker) / all numbers in the draft | `check_post.py` | pending fixes | tooling fixes (running), then the numbers pass |
| M15 | **Known bugs fixed** | confirmed bugs with a passing regression test / all confirmed bugs | `test_review_findings.py` | 0 / 10 | pass 5 + tooling fixes (running) |

## Enforcement (so coverage can't regress)
- Each metric's script becomes a test that fails below its current level and is ratcheted up as items land. M1, M3, M6, M10 and M15 are hard-gated at 100%.
- New model type in the snapshot → M1 test fails until classified.
- New rule in the relation → M3 + M6 tests fail until it has a theorem/measurement and an end-to-end instance.

## Accepted exclusions (to be stated in the post)
- Remote-code models we may not execute (M11), unless remote code is explicitly allowed (TEF_ALLOW_REMOTE_CODE=1).
- Models with no code available anywhere (M1 "no code" list).
- Limit settings are certified by measurement, not by the whole-model theorem: the worst-case thresholds exceed the settings used by many orders of magnitude (M12; see lipschitz-2 verify/lipschitz/LIPSCHITZ.md). Only masks and sinks meet their theorem thresholds.
- GeLU fits can't be certified in Lean, because Mathlib has no computable Gaussian CDF (M5/M12 note).

## Order of work
1. Running now: pass 5; Z3 update; code-diff sweep; tooling fixes; zoo tests.
2. After pass 5 lands: spot checks + catalog (M6, M7); re-run the Z3 differential and proofs on the final code (M9, M10); regenerate EDGE_JUSTIFICATION (M3).
3. Then: Lean for any UNJUSTIFIED rules (M3); bridges round 2 for the ~250 new theorems (M5); emulate v3 from the spot-check ranking (M8); measure Lipschitz constants on sample models (M12).
4. Build `verify/coverage_report.py` + the ratchet tests; produce the final dashboard.
5. Numbers pass (M14) → prose → release.
