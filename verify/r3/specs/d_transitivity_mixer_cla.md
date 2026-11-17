# d_transitivity_mixer_cla
live 0 / placed 0 (relation rule, mixgap only; 0 violations among the real shapes); direction over-wall (synthetic only); batch bd

## 1. What the code computes
- p5:model_graph.py emul_matrix :4239-4271: `plain = (mixer == "softmax") & (cla == "no")`; a plain target passes the mixer / cla check outright and is charged by the ZERO_EDGES depth (`zero_e`: a hybrid / KV-sharing host's own-KV attention layers, :4253-4268, :4320); a non-plain target needs the categorical orders `cat_ok(mixer) & cat_ok(cla)` (:4242-4244), where MIXER_ORDER has no softmax -> hybrid entry (:310-311: softmax <- hybrid is a ZERO_EDGE only, :394) and CLA_ORDER yes -> no is the CLA carry (:355), granted free in mixgap (`cla_strict` false, :4243). In mixgap the near-gap adds any softmax-layer host for a NON-softmax target (:4245-4250), and a cla ZERO_EDGE for non-sharing recurrent targets.
- Counterexample (EDGE_JUSTIFICATION mixgap; rebuilt minimal, confirmed on emul_matrix, verify/r3/d_work/triples.py):
  - A: softmax, KV-sharing (crosslayer_kv yes), 1 layer, attn_eff 1.
  - B: plain softmax (cla no), 1 layer.
  - C: hybrid (one Mamba-2 layer, one own-KV softmax layer), KV-sharing, attn_eff 1.
  - A -> B: the CLA carry (mixgap). B -> C: plain target, ZERO_EDGE on C's own-KV softmax layer. A -> C: A is not plain; mixer softmax -> hybrid has no categorical edge: "mixer/cla". (payable: A -> B is already refused, no chain.)

## 2. Classifier change
Classifier change: none (relation rule).

## 3. Relation rule
- cause: two granted constructions that do not compose in the code: the CLA carry turns a KV-sharing softmax target into a plain one (in mixgap), but only the literal `plain` mask opens the ZERO_EDGES; a KV-sharing softmax target never reaches a hybrid host although "carry, then zero the recurrent layers" is exactly the chain.
- fix (make it compose; mixgap only, where the carry is free): in mixgap a softmax target whose only non-plainness is KV sharing is routed as plain.
```python
# emul_matrix (r3_mg_p5.py :4239), after cla_strict is known
plain = (mixer == "softmax") & (cla == "no")
plain_t = plain | ((mixer == "softmax") & (cla == "yes") & (not cla_strict))   # the CLA carry, then ZERO_EDGES
ok &= _why("mixer/cla", np.where(plain_t[:, None], True, cat))                  # (:4271)
usable_e = np.where(plain_t[:, None], zero_e, el[None, :])                      # (:4320; and usable_p, kinds)
# Z3 parts (verify/z3_relation.py:763): P["mixer_cla"] = Or(t_plain_eff, cat), t_plain_eff = Or(t_plain,
#   And(mixer softmax, cla yes, Not(cla_strict))), and use_e / use_p on t_plain_eff
```
  (the target-side `plain` for zero_e / kind matching only; the host-side `h_plain` stays literal.)
- cost: depth = the host's own-KV attention layers >= the target's layers (the chain's own charge: B.eff >= A.eff, C.attn >= B.eff). +0 real edges in mixgap (8 KV-sharing softmax target shapes, 53 repos, none newly hosted: verify/r3/d_work/fixcost.py); payable / exact unchanged (cla_strict).
- kind: more edges (mixgap only); the edge kind is the carry's (granted) plus ZERO_EDGES (exact).
- theorem: ZERO_EDGES zeroing (residual identity) + the CLA carry as granted by mixgap (MIXER_GAP / CLA_CARRY_PAYABLE note, not a theorem: mixgap is the pre-measurement mode).
- property test: after the fix remove "mixer_cla" from EXPECTED_NOT_TRANSITIVE["mixgap"].

## 4. Test fixtures
- synthetic triple (Z._plain_point, hidden 8): A `crosslayer_kv=yes, attn_eff=1, attn_param=1`; B default plain; C `mixer=hybrid, crosslayer_kv=yes, feats=rf_mamba2_nonorm, eff_layers=2, param_layers=2, attn_eff=1, attn_param=1, kR_eff=1, kR_param=1, rc_rule=G.RC_M2`. Now (mixgap) A->B, B->C, not A->C; A with cla no on C passes (the fix's target), so after the fix A->C holds.
- real shapes: none.

## Verdict
real (synthetic, mixgap only): make it compose by routing a carry-able KV-sharing softmax target as plain in
mixgap. 0 real edges change; the headline (payable) is untouched.
