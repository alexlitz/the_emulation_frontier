import EmulationProofs.MoERouting

/-!
# Null experts: a top-k remainder filled by always-selected, zero-weight experts (branch fr-moe)

`moeR_dup_needs_divisibility` shows that within the duplication family a host needs `k_t ∣ k_h`.
A host with a SELECTION-ONLY bias (DeepSeek-V3 / GLM-4.5 `e_score_correction_bias`: top-k on
score + bias, gates from the unbiased score) can leave that family: `k_h = m k_t + r`, `m` copies
of every target expert and `r` NULL experts whose selection score is above every copy's (a bias
above the target's) and whose output is `0`.

* `moeR_null_topk`: if the target's top-k set is strict, every null scores above every target
  expert and every extra below every selected one (and below the nulls), then EVERY host
  top-`(k m + r)` set, whatever its tie-break, is the copies plus all the nulls.
* `moeR_null_renorm`: renormalised gates over that set are the target's times `m A / (m A + B)`,
  `A` the target's selected gate mass and `B` the nulls' (`φ` = σ: their scores `σ(z_n)`).
* `moeR_null_renorm_exact`: `B = 0` is exact. Over the reals `σ > 0`, so it is a limit:
  `moeR_null_renorm_bound` bounds the relative error by `B / (m A)`; the host's null rows read
  `-M` off a constant channel (`const_norm_carry`), `σ(-M c / R) ≤ exp(-M c / R)`, which is `0` in
  fp32 / fp64 once `M c / R` passes the underflow (catalog `fm_null_experts`).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

section nulls

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- The host's scores: copy `(e, r)` reads target expert `e`'s score, nulls read `un`, extras `ud`. -/
def nullScore {ι N D : Type*} {m : ℕ} (u : ι → ℝ) (un : N → ℝ) (ud : D → ℝ) :
    (ι × Fin m ⊕ N) ⊕ D → ℝ :=
  Sum.elim (Sum.elim (fun h => u h.1) un) ud

/-- The copies of a target set together with every null expert. -/
def copiesNulls {ι N D : Type*} [Fintype N] (m : ℕ) (S : Finset ι) :
    Finset ((ι × Fin m ⊕ N) ⊕ D) :=
  ((S ×ˢ (univ : Finset (Fin m))).disjSum univ).disjSum ∅

lemma moeR_card_copiesNulls {ι N D : Type*} [Fintype N] (m : ℕ) (S : Finset ι) :
    (copiesNulls (D := D) (N := N) m S).card = S.card * m + Fintype.card N := by
  simp [copiesNulls, Finset.card_product]

/-- **The host's top-(k·m + r) set is the copies and the nulls** (selection, exact). -/
theorem moeR_null_topk {ι N D : Type*} [Fintype ι] [Fintype N] [Fintype D] [DecidableEq ι]
    [DecidableEq N] [DecidableEq D] {m k : ℕ} {u : ι → ℝ} {un : N → ℝ} {ud : D → ℝ}
    {S : Finset ι} (hS : IsTopKOn univ u k S) (hstr : StrictTopOn univ u S)
    (hn : ∀ n, ∀ i, u i < un n) (hd : ∀ d, ∀ i ∈ S, ud d < u i) (hnd : ∀ n d, ud d < un n)
    {Sh : Finset ((ι × Fin m ⊕ N) ⊕ D)}
    (hSh : IsTopKOn univ (nullScore (m := m) u un ud) (k * m + Fintype.card N) Sh) :
    Sh = copiesNulls m S := by
  have hstrict : StrictTopOn univ (nullScore (m := m) u un ud) (copiesNulls m S) := by
    intro i hi j _ hj
    rcases i with (⟨e, r⟩ | n) | d
    · rcases j with (⟨j, s⟩ | n) | d
      · simp only [copiesNulls, Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ,
          and_true] at hi hj
        exact hstr e hi j (Finset.mem_univ _) hj
      · simp [copiesNulls] at hj
      · simp only [copiesNulls, Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ,
          and_true] at hi
        exact hd d e hi
    · rcases j with (⟨j, s⟩ | n') | d
      · exact hn n j
      · simp [copiesNulls] at hj
      · exact hnd n d
    · simp [copiesNulls] at hi
  refine moeR_topk_unique ⟨Finset.subset_univ _, by rw [moeR_card_copiesNulls, hS.2.1], ?_⟩
    hstrict hSh
  intro i hi j hj hjS
  exact (hstrict i hi j hj hjS).le

/-- **Null experts, renormalised: the factor.** Copies whose outputs sum to `m · Y e`, nulls with
output `0`, renormalised gates with score map `φ`: the host is the target times
`m A / (m A + B)`, `A = Σ_{e∈S} φ(z e)`, `B = Σ_n φ(zn n)`. -/
theorem moeR_null_renorm {ι N D : Type*} [Fintype N] {m : ℕ} (φ : ℝ → ℝ) (z : ι → ℝ)
    (zn : N → ℝ) (zd : D → ℝ) (S : Finset ι) (Y : ι → V) (Yh : ι × Fin m → V) (Yd : D → V)
    (hY : ∀ e ∈ S, ∑ r, Yh (e, r) = (m : ℝ) • Y e) (hA : ∑ e ∈ S, φ (z e) ≠ 0) :
    renormOut φ (nullScore z zn zd) (copiesNulls m S) (Sum.elim (Sum.elim Yh (fun _ => 0)) Yd) =
      ((m * ∑ e ∈ S, φ (z e)) / (m * ∑ e ∈ S, φ (z e) + ∑ n, φ (zn n))) • renormOut φ z S Y := by
  unfold renormOut
  simp only [copiesNulls, Finset.sum_disjSum, Finset.sum_product, nullScore, Sum.elim_inl,
    Sum.elim_inr, Finset.sum_empty, add_zero, smul_zero, Finset.sum_const_zero, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hnum : ∑ e ∈ S, ∑ r : Fin m, φ (z e) • Yh (e, r) = (m : ℝ) • ∑ e ∈ S, φ (z e) • Y e := by
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl (fun e he => ?_)
    rw [← Finset.smul_sum, hY e he, smul_comm]
  rw [hnum, ← Finset.mul_sum, smul_smul, smul_smul]
  congr 1
  rw [mul_comm (m : ℝ) (∑ e ∈ S, φ (z e))]
  field_simp

/-- **…exact when the nulls' gate mass is `0`** (`m ≠ 0`). -/
theorem moeR_null_renorm_exact {ι N D : Type*} [Fintype N] {m : ℕ} (hm : m ≠ 0) (φ : ℝ → ℝ)
    (z : ι → ℝ) (zn : N → ℝ) (zd : D → ℝ) (S : Finset ι) (Y : ι → V) (Yh : ι × Fin m → V)
    (Yd : D → V) (hY : ∀ e ∈ S, ∑ r, Yh (e, r) = (m : ℝ) • Y e) (hA : ∑ e ∈ S, φ (z e) ≠ 0)
    (hB : ∑ n, φ (zn n) = 0) :
    renormOut φ (nullScore z zn zd) (copiesNulls m S) (Sum.elim (Sum.elim Yh (fun _ => 0)) Yd) =
      renormOut φ z S Y := by
  rw [moeR_null_renorm φ z zn zd S Y Yh Yd hY hA, hB, add_zero]
  have : (m : ℝ) * ∑ e ∈ S, φ (z e) ≠ 0 := mul_ne_zero (by exact_mod_cast hm) hA
  rw [div_self this, one_smul]

/-- **…and within `B / (m A)` relative error** (positive scores, `m ≠ 0`, `S` nonempty). -/
theorem moeR_null_renorm_bound {ι N D : Type*} [Fintype N] {m : ℕ} (hm : m ≠ 0) (φ : ℝ → ℝ)
    (hφ : ∀ t, 0 < φ t) (z : ι → ℝ) (zn : N → ℝ) (zd : D → ℝ) (S : Finset ι) (hS : S.Nonempty)
    (Y : ι → V) (Yh : ι × Fin m → V) (Yd : D → V)
    (hY : ∀ e ∈ S, ∑ r, Yh (e, r) = (m : ℝ) • Y e) :
    ‖renormOut φ (nullScore z zn zd) (copiesNulls m S) (Sum.elim (Sum.elim Yh (fun _ => 0)) Yd) -
        renormOut φ z S Y‖ ≤
      (∑ n, φ (zn n)) / (m * ∑ e ∈ S, φ (z e)) * ‖renormOut φ z S Y‖ := by
  have hA : 0 < ∑ e ∈ S, φ (z e) := Finset.sum_pos (fun _ _ => hφ _) hS
  rw [moeR_null_renorm φ z zn zd S Y Yh Yd hY hA.ne']
  have hm' : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  set A := (m : ℝ) * ∑ e ∈ S, φ (z e)
  set B := ∑ n, φ (zn n)
  have hA' : 0 < A := by positivity
  have hB : 0 ≤ B := Finset.sum_nonneg (fun _ _ => (hφ _).le)
  have e1 : (A / (A + B)) • renormOut φ z S Y - renormOut φ z S Y =
      (-(B / (A + B))) • renormOut φ z S Y := by
    have : A / (A + B) - 1 = -(B / (A + B)) := by field_simp; ring
    rw [← this, sub_smul, one_smul]
  rw [e1, norm_smul, norm_neg, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
  exact div_le_div_of_nonneg_left hB hA' (by linarith)

/-- The null score `σ(-a)` is below `exp(-a)`: the bound's `B` falls as `exp(-M c / R)`. -/
theorem moeR_null_sigmoid_le (a : ℝ) : sigmoid (-a) ≤ exp (-a) := by
  rw [Real.sigmoid_def]
  have h1 : 0 < 1 + exp (- -a) := by positivity
  rw [inv_le_iff_one_le_mul₀ h1, mul_add, mul_one, ← Real.exp_add, neg_neg, neg_add_cancel, Real.exp_zero]
  linarith [Real.exp_pos (-a)]

end nulls

end EP
