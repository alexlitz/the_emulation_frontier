# d_transitivity_kv_heads
live 0 / placed 0 (relation rule; 0 violations among the real shapes); direction over-credit (synthetic only); batch bd

## 1. What the code computes
- KV-head packing (p5:model_graph.py `_kv_fit_shape`, :3990-4035): a host KV head of head_dim hd_h holds up to q = min(hd_h // hd_t, q_per_kv_h) target KV heads side by side; on a RoPE pair q = 1 unless the host is pairs-free (:4013), and then q is cut to the largest m with `rope_fits(m x need_s[i], cap_s[j])` (:4014-4030). `need_s[i]` is the target's class = its cap only when pairs-free or fraction-free (:4014-4016), else its sched. `kv_hosts_needed(kt, gt, gh, q)` (:3925-3940) counts the host heads.
- EDGE_JUSTIFICATION (z3-final verify/EDGE_JUSTIFICATION.md "Properties"): kv_heads UNKNOWN (timeout) with the whole relation, REFUTED alone; the old bounded box: 6,398 kv_heads violations. Re-run on the current p5 code (z3-final worktree, identical model_graph.py): `Z.box_family("heads")` has 0 violations in exact / payable / mixgap (the box only has pairs-free cap > sched points, which the pass-5 fixed-rotary packing guard :4312-4317 now covers -- the recorded 6,398 predate it). Adding FIXED cap > sched (proportional) points to the box (verify/r3/d_work/kvbox2.py, N = 4): 44 kv_heads violations (beside 7,140 rot-gate ones).
- Counterexample (exact, confirmed on emul_matrix):
  - A: rope MHA, 2 KV heads x 1 query, head_dim 4, sched (4,), cap (4,), fixed.
  - B: rope MHA, 2 KV heads x 1 query, head_dim 4, sched (2,), cap (4,), fixed (proportional table).
  - C: rope MQA, 1 KV head x 2 queries, head_dim 8, sched (2,), cap (4,), pairs-free.
  - A -> B head for head (rot gate: 4 on cap 4). B -> C: C packs B's two heads into its one 8-dim head, rope_fits(2 x (2,), (4,)) holds (q = 2). A -> C: rope_fits(2 x (4,), (4,)) fails, q = 1, A needs 2 KV heads, C has 1: refused by "kv heads".

## 2. Classifier change
Classifier change: none (relation rule).

## 3. Relation rule
- cause: the same host-side relaxation as d_transitivity_rot_gate: B plays A at its cap, but as a target B's packing need is its sched, so C packs B tighter than B's class allows.
- fix: the class rule of d_transitivity_rot_gate in the packing need:
```python
# _kv_fit_shape (r3_mg_p5.py :4014)
need_s = [host_rope_cap(p) if (p["dims"].get("rope_pairs_free")
                               or (p["dims"].get("rot_frac_free") and _p5("partial"))
                               or tuple(p["dims"].get("rope_cap", ())) != tuple(p["dims"].get("rope_sched", ())))
          else tuple(p["dims"].get("rope_sched", ())) for p in P]
```
  (Z3 model: `tgt_class` and `kv_shape_fit`'s `mstar(self.tgt_class(t), ...)`, verify/z3_relation.py:461-465, :494.)
- cost: 0 real edges (no real fixed shape has cap != sched); the box with proportional points: 44 kv_heads violations -> 0 (with the rot-gate class rule, 0 violations of any conjunct in every mode).
- kind: guard (fewer edges, synthetic only).
- theorem: none new; packing = `kvPack` rows (PROOFS.md), unchanged.
- property test: after the fix kv_heads stays UNKNOWN in Z3 (nonlinear), so keep the bounded exhaustive box as its check and add the proportional variants ("rope", (2,), (4,), 0), ("partial_rope", (2,), (4,), 0), ("rope", (4,), (4,), 0), ("partial_rope", (2,), (2,), 0) to `box_family("heads")` (verify/z3_relation.py:2026) so the box can see the case at all.

## 4. Test fixtures
- synthetic triple (Z._plain_point): A `kv=MHA, kv_heads=2, q_per_kv=1, qheads=2, head_dim=4, kv_cap=8, q_cap=8, rope_sched=(4,), rope_cap=(4,)`; B the same with `rope_sched=(2,), rope_cap=(4,)`; C `kv=MQA, kv_heads=1, q_per_kv=2, qheads=2, head_dim=8, kv_cap=8, q_cap=16, rope_sched=(2,), rope_cap=(4,), rope_pairs_free=1`. Now A->B, B->C, not A->C ("kv heads"); after the fix B->C refused.
- real shapes: none.

## Verdict
real (synthetic), fixed by the same class rule as rot_gate (0 real-edge cost, frontier unchanged); the recorded
6,398 box violations are stale -- already fixed in round2-p5 for pairs-free middles (:4312-4317), not for fixed
proportional ones.
