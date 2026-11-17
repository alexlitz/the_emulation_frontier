# d_transitivity_depth
live 0 / placed 0 (relation rule; wf-only counterexample, PROVED on realistic shapes); direction over-credit (synthetic only); batch bd

## 1. What the code computes
- Depth (p5:model_graph.py emul_matrix :4253-4268, :4310-4311): a PLAIN target (softmax, own-KV) uses the ZERO_EDGES depth `zero_e` -- on a non-plain host (KV-sharing / hybrid) only the host's own-KV attention layers, `attn_eff` / `attn_param` (full ones for a full target); a NON-plain target uses the host's whole `eff_layers`, its layer kinds matched by `kind_match` (:3632, applied :4503-4504, over k{FU,FT,LU,LT,S,R}; ZERO_EDGES :394-396).
- dims_of builds `attn_eff = scale(a_cnt, eff_layers)` and the kind counts `k{kind}_eff = scale(kc[kind], eff_layers)` separately from `_attn_counts` (:3256-3262): two roundings of the same layer list.
- Counterexample (EDGE_JUSTIFICATION exact / payable / mixgap; rebuilt by hand and confirmed on emul_matrix, verify/r3/d_work/triples.py):
  - A: plain, window local, 1 layer (kLU 1).
  - B: KV-sharing (crosslayer_kv yes), 1 layer, attn_eff 1.
  - C: KV-sharing, 1 layer, attn_eff 0 but kFU_eff 1 (its kinds say one own-KV layer, its zero-edge count says none).
  - A -> B: plain target, zero_e = B.attn_eff = 1 >= 1. B -> C: non-plain target, C.eff_layers 1 >= 1, kinds 1 <= 1. A -> C: zero_e = C.attn_eff = 0 < 1: "depth (eff)", "depth (param)".

## 2. Classifier change
Classifier change: none (relation rule).

## 3. Relation rule
- cause: two counts of the same thing. The plain-target path charges `attn_eff`, the non-plain path the kind counts; a point whose attn_eff is below its own-KV kind total lets B -> C pass on kinds while A -> C fails on attn_eff. A real stack always has attn_eff = kFU + kFT + kLU + kLT (eff and param): 0 exceptions among the 931 real shapes, and no KV-sharing shape has attn_eff 0 (verify/r3/d_work/fixcost.py). EDGE_JUSTIFICATION: "PROVED on realistic shapes".
- fix (make the domain say so, no edge change): build attn_eff / attn_param as the kind sum so it is an invariant, and state it as a "built" wf clause of the Z3 model:
```python
# dims_of (r3_mg_p5.py :3256-3262), after kind_dims
attn_eff = sum(kind_dims[f"k{k}_eff"] for k in ("FU", "FT", "LU", "LT"))
attn_param = sum(kind_dims[f"k{k}_param"] for k in ("FU", "FT", "LU", "LT"))
full_eff = kind_dims["kFU_eff"] + kind_dims["kFT_eff"]
full_param = kind_dims["kFU_param"] + kind_dims["kFT_param"]
# verify/z3_relation.py wf_clauses (:916): ("zero-edge counts are the own-KV kinds", "built",
#   And(d["attn_eff"] == Sum(k{FU,FT,LU,LT}_eff), d["attn_param"] == Sum(..._param),
#       d["full_eff"] == kFU_eff + kFT_eff, d["full_param"] == kFU_param + kFT_param))
```
  (both halves hold on all 931 real shapes, eff and param: verify/r3/d_work/fullchk.py.)
- not "2x depth twice is 4x": MOE_DEPTH_FACTOR doubling is applied per hop and composes (each hop's host has the depth the next hop charges); the refuted case is the count mismatch only.
- cost: 0 real edges (the rebuilt counts equal the current ones on every real shape).
- kind: invariant / test-domain fix.
- theorem: none (ZERO_EDGES layer zeroing).
- property test: add the clause, then "depth" leaves EXPECTED_NOT_TRANSITIVE (z3-final tests/test_z3_relation.py) in every mode.

## 4. Test fixtures
- synthetic triple (Z._plain_point, hidden 8): A `window=local, kFU_eff=0, kFU_param=0, kLU_eff=1, kLU_param=1`; B `crosslayer_kv=yes, attn_eff=1, attn_param=1`; C `crosslayer_kv=yes, attn_eff=0, attn_param=0` (kFU 1 default). Now A->B, B->C, not A->C. C violates the new wf clause, so it leaves the domain.
- real shapes: none.

## Verdict
non-issue for the real relation (wf-only; transitive on realistic shapes): make the invariant built and drop depth
from the expected-non-transitive set. Frontier unchanged.
