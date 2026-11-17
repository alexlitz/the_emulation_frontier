# d_transitivity_dense_host
live 0 / placed 0 (relation rule; 0 violations among the real shapes); direction property (both edges real); batch bd

## 1. What the code computes
- Dense host for a MoE target (p5:model_graph.py `moe_why` :964-971): only a top-1 unit-gate target -- `T["k"] == 1 and T["renorm"] == "R"` -- with hidden >= ht + E_t select channels and ffn >= ft + 2 E_t (E_t - 1) erase units; emul_matrix also requires the target's `moe_unit_top1` and not exact mode (:4456, the "dense_host" conjunct; Z3 verify/z3_relation.py:842). The check reads the target's OWN renorm, not its renorm_cap.
- MoE <- MoE (`moe_why` :990-1010): the target's renorm / score need only be settable on the host (`renorm_cap`, `score_cap`); see d_transitivity_moe for the policy (:4568-4574: superset guards not applied).
- Counterexample (EDGE_JUSTIFICATION payable / mixgap; rebuilt minimal, confirmed on emul_matrix, verify/r3/d_work/triples.py):
  - A: MoE E = 2, top-1, renorm P (unrenormalised), score sigmoid: the expert output is scaled by sigmoid(l), not 1.
  - B: MoE E = 2, top-1, renorm R, score softmax, moe_unit_top1 1, renorm_cap {P, R}, score_cap {sigmoid, softmax}.
  - C: dense vanilla FFN, hidden 16 (>= 8 + 2), ffn 8 (>= 4 + 4), 2 layers.
  - A -> B: B set to (P, sigmoid). B -> C: B as configured is a top-1 unit gate, the dense construction applies. A -> C: A is not a unit gate: "ffn MoE<-dense" (payable / mixgap; exact never allows a dense host).

## 2. Classifier change
Classifier change: none (relation rule).

## 3. Relation rule
- cause: host-side config freedom not inherited (as d_transitivity_moe): B hosts A as B' = (P, sigmoid); C plays B as (R, softmax). A <- C is genuinely impossible by the dense construction: a hard select needs a unit gate (`hard_select_needs_unit_gate`, PROOFS.md "Where the formal statement needs a condition", item 1).
- decision: correctly non-transitive; keep the rule. The guard that would restore transitivity ("a dense host plays a MoE target only if every renorm the target's family can be set to is R": `"P" not in renorm_cap`) removes 5 real payable / mixgap edges, all of one target: huihui-ai/Huihui-MoE-0.8B-2E (renorm R, settable P / R) on Qwen/Qwen3-32B, QCRI/Fanar-2-27B-Instruct, ig1/medgemma-27b-it-FP8-Dynamic, ... (verify/r3/d_work/guard_ex.py) -- exact-by-construction edges for the configured target.
- property test: keep "dense_host" in EXPECTED_NOT_TRANSITIVE (payable, mixgap) with the reason; test the class-closed form instead: with the middle point's renorm_cap == {renorm} and score_cap == {score}, expect transitivity.
- cost: none (no rule change).
- kind: property (test weakened to the class-closed form).
- theorem: `dense_sub_moe_shared`, `hard_select_needs_unit_gate` (unchanged).

## 4. Test fixtures
- synthetic triple (Z._plain_point, moe_sig in G.MOE_SIG order): A `hidden=8, ffn_total=4, ffn_active=4, kFU_eff=1, kFU_param=1, ffn=moe, moe_sig=(2, 1, 4, 0, "P", ("P",), "sigmoid", ("sigmoid",), 0, 0, "", 1, 0, 0, 0, 0)`; B the same with `moe_unit_top1=1, moe_sig=(2, 1, 4, 0, "R", ("P", "R"), "softmax", ("sigmoid", "softmax"), 0, 0, "", 1, 0, 0, 0, 0)`; C `hidden=16, ffn_total=8, ffn_active=8, eff_layers=2, param_layers=2, kFU_eff=2, kFU_param=2, ffn=vanilla`. Expected (kept): payable A->B, B->C, not A->C.
- real shapes: none violate; closest real middle point: huihui-ai/Huihui-MoE-0.8B-2E (top-1 of 2, renorm settable).

## Verdict
non-issue (correct non-transitivity from the middle MoE's settable renorm; the guard costs 5 real edges of one
target): keep the rule, weaken the property test to the class-closed form. Frontier unchanged.
