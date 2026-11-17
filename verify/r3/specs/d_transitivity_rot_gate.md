# d_transitivity_rot_gate
live 0 / placed 0 (relation rule; 0 violations among the 931 real shapes, 0 real shapes of the middle kind); direction over-credit (synthetic only); batch bd

## 1. What the code computes
- Rot gate (p5:model_graph.py emul_matrix :4277-4310): for a RoPE target i on a RoPE host j, `nd = cap[i] if (pf[i] or ff[i]) else need[i]` -- "a target is its family's config class" -- then a pairs-free host checks `rope_fits(nd, cap[j])`, a fixed host `fixed_rot_fits(need[i], hd_t, kind_t, cap[j], hd_h, ff[j], kind_j, need[j])` (:3892-3909: one table of d_t rotated dims fits a host table c >= d_t with hd_t - d_t <= hd_h - c).
- `cap` = `host_rope_cap` (:3881-3889) = dims rope_cap, which `_rope_schedules` sets to the WHOLE head for a "proportional" table and to need otherwise (:1255-1259, :3278-3280). So a fixed (not pairs-free, not fraction-free) point can have cap > sched: as a HOST it rotates up to cap; as a TARGET it is held to its sched.
- Counterexample (confirmed on emul_matrix, every mode; verify/r3/d_work/triples.py):
  - A: rope, head_dim 4, sched (4,), cap (4,) (full RoPE, fixed).
  - B: partial_rope, head_dim 4, sched (2,), cap (4,) (proportional table, fixed).
  - C: partial_rope, head_dim 4, sched (2,), cap (2,) (fixed partial RoPE).
  - A -> B: fixed_rot_fits((4,), cap (4,)) holds (c = 4). B -> C: (2,) on cap (2,) holds. A -> C: 4 rotated dims on a 2-dim table: refused by "pos rot gate".

## 2. Classifier change
Classifier change: none (relation rule).

## 3. Relation rule
- cause: a host-side relaxation not inherited. B hosts A with its table raised to cap (config), but as a target B is held to its own sched, so C (which matches only B's sched) plays B without playing B's class.
- fix (make it compose; the pass-5 "TRANSITIVITY (forced)" pattern already used for pairs-free / fraction-free targets): a target whose cap differs from its sched stands for its cap.
```python
# emul_matrix, rot gate (r3_mg_p5.py :4296-4307)
cls_i = pf[i] or ff[i] or tuple(cap[i]) != tuple(need[i])
nd = cap[i] if cls_i else need[i]
...
elif pf[j]:
    memo[key] = rope_fits(nd, cap[j])
else:
    memo[key] = (not pf[i] and (ff[j] or not ff[i])
                 and fixed_rot_fits(nd, int(hd[i]), kind[i], cap[j], int(hd[j]), bool(ff[j]), kind[j], need[j]))
```
  and the same class in `_kv_fit_shape`'s packing need (:4014, `need_s = host_rope_cap(p) if (pf or ff or cap != sched) else sched`) and in the Z3 model's `tgt_class` (verify/z3_relation.py:461-465: `cls = Or(pf, ff, Or(*[c != s for c, s in zip(cap, sched)]))`).
- cost: 0 real edges (no real shape is fixed with cap != sched: verify/r3/d_work/fixcost.py). On the synthetic box with proportional middle points (kvbox2.py, N = 4, 1,104 points): 7,184 violating (A, C) -> 0 in exact / payable / mixgap; edges 154,130 -> 146,188 (the proportional targets lose the hosts that only match their sched).
- kind: exact rule unchanged; this is a guard (fewer edges, synthetic only).
- theorem: none new (the guard removes edges); the kept edges use `defaultFreq_match`, `partial_rope_host_config`.
- property test: after the fix remove "rot_gate" from EXPECTED_NOT_TRANSITIVE (z3-final tests/test_z3_relation.py) in every mode.

## 4. Test fixtures
- synthetic triple (Z._plain_point, all other dims default): A `pos=rope, head_dim=4, kv_heads=1, q_per_kv=1, qheads=1, kv=MHA, rope_sched=(4,), rope_cap=(4,)`; B the same with `pos=partial_rope, rope_sched=(2,), rope_cap=(4,)`; C the same with `pos=partial_rope, rope_sched=(2,), rope_cap=(2,)`. Now: A->B, B->C, not A->C. After the fix: B->C refused (B's class needs 4 rotating dims), no violation.
- real shapes: none (0 real violations; no real fixed point with cap != sched).

## Verdict
real (synthetic), fix it: apply the class rule -- the relation becomes transitive on the rot gate at 0 real-edge
cost (frontier unchanged). Also fixes d_transitivity_kv_heads.
