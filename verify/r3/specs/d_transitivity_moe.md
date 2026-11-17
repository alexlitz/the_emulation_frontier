# d_transitivity_moe
live 0 / placed 0 (relation rule; 0 violations among the real shapes); direction property (both edges real); batch bd

## 1. What the code computes
- MoE <- MoE (p5:model_graph.py `moe_why` :954-1010, applied by `_moe_ok` :4546; Z3 `moe_ok`, verify/z3_relation.py:591-624): the target's router score must be one the HOST can be set to (`T["score"] not in H["score_cap"]` refuses, :996), its renormalisation one the host can be set to (`T["renorm"] in H["renorm_cap"]`), plus copies / widths / groups. `score_cap` / `renorm_cap` are the values the host family's code reads from config (SCORE_CONFIG_TYPES, RENORM_KEYS).
- The docstring states the policy (:4568-4574): "a host can be SET to a renorm / score / group shape its code allows, but as a target it is its own config, so config freedom on the host side can break X <= Y <= Z ... The capability-superset guards ... are not needed on the snapshot (0 violations without them) and are not applied." The one forced guard: an unrenormalised softmax target needs a host whose own router is unrenormalised softmax (:1000-1005).
- Counterexample (EDGE_JUSTIFICATION exact / payable / mixgap; rebuilt minimal, confirmed on emul_matrix in all three modes, verify/r3/d_work/triples.py):
  - A: MoE E = 2, top-1, expert width 4, renorm R, score sigmoid (score_cap {sigmoid}).
  - B: the same MoE with score softmax, score_cap {sigmoid, softmax} (its code reads scoring_func).
  - C: the same MoE with score softmax, score_cap {softmax} (softmax fixed in code).
  - A -> B: sigmoid in B's cap. B -> C: softmax in C's cap. A -> C: sigmoid not in {softmax}: "ffn MoE routing".

## 2. Classifier change
Classifier change: none (relation rule).

## 3. Relation rule
- cause: a host-side config relaxation that is not inherited: B hosts A as B' (scoring_func = sigmoid); C plays B as configured (softmax), not B'. All three edges claimed are real constructions: A <- B' (expert copies, `moeR_*` rows), B <- C (same score), and A <- C is genuinely impossible (a softmax gate cannot equal sigmoid weights for linear logits, FEATURES router_sigmoid note). The relation is "h, at some free config, emulates t as configured", which does not compose through a reconfigured middle point.
- decision: correctly non-transitive here; do NOT add the superset guard. Applying "T score_cap <= H score_cap and T renorm_cap <= H renorm_cap" would remove 30 / 32 / 39 real MoE <- MoE edges (exact / payable / mixgap; 60-77 target repos, verify/r3/d_work/fixcost.py), e.g. Qwen/Qwen3-30B-A3B (renorm R, settable P / R) on QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ (R fixed): an exact edge (same renorm, same score) lost only because the target COULD be configured otherwise.
- property test: keep "moe" in EXPECTED_NOT_TRANSITIVE, with the reason; replace the blanket expectation by the class-closed statement the relation does satisfy -- add to the Z3 query `B.score_cap == {B.score} and B.renorm_cap == {B.renorm}` (the middle point configured to its whole class) and expect PROVED (or the bounded check), plus the existing 0-violation check on the real shapes.
- cost: none (no rule change).
- kind: property (test weakened to the class-closed form).
- theorem: none new.

## 4. Test fixtures
- synthetic triple (Z._plain_point, hidden 8, ffn_total 4, kFU 1, ffn=moe, moe_sig in G.MOE_SIG order): A `(2, 1, 4, 0, "R", ("R",), "sigmoid", ("sigmoid",), 0, 0, "", 1, 0, 0, 0, 0)`; B `(2, 1, 4, 0, "R", ("R",), "softmax", ("sigmoid", "softmax"), 0, 0, "", 1, 0, 0, 0, 0)`; C `(2, 1, 4, 0, "R", ("R",), "softmax", ("softmax",), 0, 0, "", 1, 0, 0, 0, 0)`. Expected (and kept): A->B, B->C, not A->C. With B's score_cap = ("softmax",) the triple is no longer a chain (A->B fails).
- real shapes: none violate; the closest real middle points are hosts with a settable renorm / score (Qwen3-MoE renorm P / R, DeepSeek-V4 sigmoid / sqrtsoftplus).

## Verdict
non-issue (correct non-transitivity from host-side config freedom; the guard costs 30-39 real exact edges): keep
the rule, weaken the property test to the class-closed form. Frontier unchanged.
