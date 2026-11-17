import Mathlib

/-!
# Positive random features for `exp(q·k)`: unbiased, variance exactly `∝ 1/m`

The positive random-feature estimator of the softmax kernel (Performer / FAVOR+) draws `m`
independent standard Gaussian vectors `ω_s ∈ ℝᵈ` and sets
`Ŷ = (1/m) ∑_s φ_q(ω_s) φ_k(ω_s)`, `φ_x(ω) = exp(ω·x - |x|²/2)`.
We prove, over an arbitrary probability space carrying `m·d` independent `N(0,1)` coordinates:

* it is unbiased: `E Ŷ = exp(q·k)` (`rf_unbiased`);
* its variance is exactly `(exp(|q|² + |k|² + 4 q·k) - exp(2 q·k)) / m`
  (`rf_mse`, `rf_variance`), so the standard deviation is `C(q,k)/√m`;
* Chebyshev: `P(|Ŷ - exp(q·k)| ≥ δ) ≤ Var/δ²` (`rf_chebyshev`), the `O(m^{-1/2})` rate.
-/

open Real Finset MeasureTheory ProbabilityTheory

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {κ : Type*} [Fintype κ]

/-- `exp` of a linear form in an independent standard Gaussian family is integrable. -/
lemma gauss_exp_integrable [IsProbabilityMeasure μ] {X : κ → Ω → ℝ} (hind : iIndepFun X μ)
    (hmeas : ∀ i, Measurable (X i)) (hlaw : ∀ i, HasLaw (X i) (gaussianReal 0 1) μ)
    (a : κ → ℝ) : Integrable (fun ω => exp (∑ i, a i * X i ω)) μ := by
  have hY : iIndepFun (fun i => (fun x => a i * x) ∘ X i) μ :=
    hind.comp _ (fun i => measurable_const.mul measurable_id)
  have := hY.integrable_exp_mul_sum (t := 1) (s := univ)
    (fun i => (measurable_const.mul measurable_id).comp (hmeas i)) (fun i _ => by
      have h := (mgf_pos_iff (X := X i) (μ := μ) (t := a i)).mp (by
        rw [mgf_gaussianReal (hlaw i)]; exact exp_pos _)
      simpa [Function.comp] using h)
  simpa [Function.comp, Finset.sum_apply] using this

/-- **Gaussian moment-generating function of a linear form**:
`E exp(∑ a_i X_i) = exp(∑ a_i² / 2)` for independent `X_i ~ N(0,1)`. -/
lemma gauss_exp_integral [IsProbabilityMeasure μ] {X : κ → Ω → ℝ} (hind : iIndepFun X μ)
    (hmeas : ∀ i, Measurable (X i)) (hlaw : ∀ i, HasLaw (X i) (gaussianReal 0 1) μ)
    (a : κ → ℝ) : ∫ ω, exp (∑ i, a i * X i ω) ∂μ = exp ((∑ i, a i ^ 2) / 2) := by
  have hY : iIndepFun (fun i => (fun x => a i * x) ∘ X i) μ :=
    hind.comp _ (fun i => measurable_const.mul measurable_id)
  have h := hY.mgf_sum (t := 1)
    (fun i => (measurable_const.mul measurable_id).comp (hmeas i)) univ
  have h1 : ∀ i, ∫ ω, exp (a i * X i ω) ∂μ = exp (a i ^ 2 / 2) := by
    intro i
    have := mgf_gaussianReal (hlaw i) (a i)
    simp only [mgf] at this
    rw [this]; congr 1; push_cast; ring
  simp only [mgf, Finset.sum_apply, Function.comp, one_mul] at h
  rw [h, Finset.prod_congr rfl (fun i _ => h1 i), ← Real.exp_sum, Finset.sum_div]

variable {m d : ℕ}

/-- The positive random-feature estimate of `exp(q·k)` from the draws `X (s, i)`
(feature `s`, coordinate `i`). -/
def rfEst (X : Fin m × Fin d → Ω → ℝ) (q k : Fin d → ℝ) (ω : Ω) : ℝ :=
  (1 / m : ℝ) * ∑ s : Fin m, exp (∑ i, X (s, i) ω * q i - (∑ i, q i ^ 2) / 2) *
    exp (∑ i, X (s, i) ω * k i - (∑ i, k i ^ 2) / 2)

lemma sum_sel (s : Fin m) (F : Fin m → Fin d → ℝ) :
    ∑ u : Fin m, ∑ i : Fin d, (if u = s then F u i else 0) = ∑ i, F s i := by
  rw [Finset.sum_eq_single s (fun u _ hu => by simp [hu]) (by simp)]
  simp

/-- Coefficients selecting feature `s`. -/
def rfCoef (p : Fin d → ℝ) (s : Fin m) : Fin m × Fin d → ℝ :=
  fun ui => if ui.1 = s then p ui.2 else 0

lemma rf_feature (X : Fin m × Fin d → Ω → ℝ) (q k : Fin d → ℝ) (s : Fin m) (ω : Ω) :
    exp (∑ i, X (s, i) ω * q i - (∑ i, q i ^ 2) / 2) *
        exp (∑ i, X (s, i) ω * k i - (∑ i, k i ^ 2) / 2) =
      exp (-((∑ i, q i ^ 2) + ∑ i, k i ^ 2) / 2) *
        exp (∑ ui, rfCoef (q + k) s ui * X ui ω) := by
  rw [← exp_add, ← exp_add]
  congr 1
  rw [Fintype.sum_prod_type]
  simp only [rfCoef, ite_mul, zero_mul]
  rw [sum_sel]
  simp only [Pi.add_apply, add_mul, Finset.sum_add_distrib, mul_comm (X _ ω)]
  ring

lemma exp_mul_exp_sum (a b : Fin m × Fin d → ℝ) (X : Fin m × Fin d → Ω → ℝ) (ω : Ω) :
    exp (∑ ui, a ui * X ui ω) * exp (∑ ui, b ui * X ui ω) =
      exp (∑ ui, (a + b) ui * X ui ω) := by
  rw [← exp_add, ← Finset.sum_add_distrib]
  simp [add_mul]

lemma rfCoef_sq (p : Fin d → ℝ) (s : Fin m) :
    ∑ ui, rfCoef p s ui ^ 2 = ∑ i, p i ^ 2 := by
  rw [Fintype.sum_prod_type]
  have : ∀ u i, rfCoef p s (u, i) ^ 2 = if u = s then p i ^ 2 else 0 := by
    intro u i; by_cases hu : u = s <;> simp [rfCoef, hu]
  simp only [this]
  exact sum_sel s _

lemma rfCoef_add_sq (p : Fin d → ℝ) (s t : Fin m) :
    ∑ ui, (rfCoef p s + rfCoef p t) ui ^ 2 =
      if s = t then 4 * ∑ i, p i ^ 2 else 2 * ∑ i, p i ^ 2 := by
  rw [Fintype.sum_prod_type]
  split_ifs with hst
  · subst hst
    have : ∀ u i, (rfCoef p s + rfCoef p s) (u, i) ^ 2 = if u = s then 4 * p i ^ 2 else 0 := by
      intro u i; by_cases hu : u = s <;> simp [rfCoef, hu]; ring
    simp only [this]
    rw [sum_sel, Finset.mul_sum]
  · have : ∀ u i, (rfCoef p s + rfCoef p t) (u, i) ^ 2 =
        (if u = s then p i ^ 2 else 0) + (if u = t then p i ^ 2 else 0) := by
      intro u i
      simp only [Pi.add_apply, rfCoef]
      by_cases hs : u = s
      · subst hs; simp [hst]
      · by_cases ht : u = t
        · subst ht; simp [hs]
        · simp [hs, ht]
    simp only [this, Finset.sum_add_distrib]
    rw [sum_sel, sum_sel]
    ring

lemma sq_sum_add (q k : Fin d → ℝ) :
    ∑ i, (q + k) i ^ 2 = ∑ i, q i ^ 2 + 2 * ∑ i, q i * k i + ∑ i, k i ^ 2 := by
  simp only [Pi.add_apply, add_sq, Finset.sum_add_distrib, Finset.mul_sum]
  refine congrArg₂ _ (congrArg₂ _ rfl (Finset.sum_congr rfl fun i _ => by ring)) rfl

section Main

variable [IsProbabilityMeasure μ] {X : Fin m × Fin d → Ω → ℝ} (hind : iIndepFun X μ)
  (hmeas : ∀ i, Measurable (X i)) (hlaw : ∀ i, HasLaw (X i) (gaussianReal 0 1) μ)
include hind hmeas hlaw

lemma rfEst_eq (q k : Fin d → ℝ) :
    rfEst X q k = fun ω => (1 / m : ℝ) * ∑ s : Fin m,
      exp (-((∑ i, q i ^ 2) + ∑ i, k i ^ 2) / 2) * exp (∑ ui, rfCoef (q + k) s ui * X ui ω) := by
  funext ω; simp only [rfEst, rf_feature]

/-- **Random features are unbiased**: `E Ŷ = exp(q·k)`. -/
theorem rf_unbiased (hm : 0 < m) (q k : Fin d → ℝ) :
    ∫ ω, rfEst X q k ω ∂μ = exp (∑ i, q i * k i) := by
  rw [rfEst_eq hind hmeas hlaw, integral_const_mul, integral_finsetSum _ (fun s _ =>
    (gauss_exp_integrable hind hmeas hlaw _).const_mul _)]
  simp only [integral_const_mul, gauss_exp_integral hind hmeas hlaw, rfCoef_sq, sq_sum_add,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [← exp_add]
  have : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  field_simp
  congr 1; ring

/-- **The mean squared error of `m` random features is exactly `C(q,k)/m`**:
`E (Ŷ - exp(q·k))² = (exp(|q|² + |k|² + 4 q·k) - exp(2 q·k)) / m`. -/
theorem rf_mse (hm : 0 < m) (q k : Fin d → ℝ) :
    ∫ ω, (rfEst X q k ω - exp (∑ i, q i * k i)) ^ 2 ∂μ =
      (exp (∑ i, q i ^ 2 + ∑ i, k i ^ 2 + 4 * ∑ i, q i * k i) -
        exp (2 * ∑ i, q i * k i)) / m := by
  set c := exp (-((∑ i, q i ^ 2) + ∑ i, k i ^ 2) / 2)
  set e := exp (∑ i, q i * k i)
  set g : (Fin m × Fin d → ℝ) → Ω → ℝ := fun a ω => exp (∑ ui, a ui * X ui ω)
  have hg : ∀ a, Integrable (g a) μ := fun a => gauss_exp_integrable hind hmeas hlaw a
  have hgi : ∀ a, ∫ ω, g a ω ∂μ = exp ((∑ i, a i ^ 2) / 2) :=
    fun a => gauss_exp_integral hind hmeas hlaw a
  set p := q + k
  have hpt : ∀ ω, (rfEst X q k ω - e) ^ 2 =
      (1 / m : ℝ) ^ 2 * c ^ 2 * ∑ s : Fin m, ∑ t : Fin m, g (rfCoef p s + rfCoef p t) ω -
        2 * e * (1 / m : ℝ) * c * ∑ s : Fin m, g (rfCoef p s) ω + e ^ 2 := by
    intro ω
    rw [rfEst_eq hind hmeas hlaw]
    simp only [g, ← exp_mul_exp_sum, ← Finset.mul_sum, ← Finset.sum_mul, ← Finset.mul_sum]
    ring
  simp_rw [hpt]
  have hI1 : Integrable (fun ω => (1 / m : ℝ) ^ 2 * c ^ 2 *
      ∑ s : Fin m, ∑ t : Fin m, g (rfCoef p s + rfCoef p t) ω) μ :=
    (integrable_finsetSum _ (fun s _ => integrable_finsetSum _ (fun t _ => hg _))).const_mul _
  have hI2 : Integrable (fun ω => 2 * e * (1 / m : ℝ) * c * ∑ s : Fin m, g (rfCoef p s) ω) μ :=
    (integrable_finsetSum _ (fun s _ => hg _)).const_mul _
  have hI12 : Integrable (fun ω => (1 / m : ℝ) ^ 2 * c ^ 2 *
      ∑ s : Fin m, ∑ t : Fin m, g (rfCoef p s + rfCoef p t) ω -
        2 * e * (1 / m : ℝ) * c * ∑ s : Fin m, g (rfCoef p s) ω) μ := hI1.sub hI2
  rw [integral_add hI12 (integrable_const _), integral_sub hI1 hI2,
    integral_const_mul, integral_const_mul, integral_const, integral_finsetSum _
      (fun s _ => integrable_finsetSum _ (fun t _ => hg _)),
    integral_finsetSum _ (fun s _ => hg _)]
  simp only [integral_finsetSum _ (fun t _ => hg _), hgi, rfCoef_add_sq, rfCoef_sq]
  simp only [probReal_univ, smul_eq_mul, one_mul]
  -- the double sum: `m` diagonal terms and `m² - m` off-diagonal ones
  have hdiag : ∀ A B : ℝ, ∑ s : Fin m, ∑ t : Fin m, exp ((if s = t then A else B) / 2) =
      m * exp (A / 2) + (m * m - m) * exp (B / 2) := by
    intro A B
    have : ∀ s : Fin m, ∑ t : Fin m, exp ((if s = t then A else B) / 2) =
        exp (A / 2) + (m - 1) * exp (B / 2) := by
      intro s
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ s)]
      simp only [ite_true]
      rw [Finset.sum_congr rfl (fun t ht => (show exp ((if s = t then A else B) / 2) = exp (B / 2) by simp [(Finset.ne_of_mem_erase ht).symm]))]
      simp [Finset.card_erase_of_mem, Nat.cast_sub (Nat.succ_le_of_lt hm)]
    simp only [this, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  rw [hdiag]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  set Q := ∑ i, q i ^ 2
  set K := ∑ i, k i ^ 2
  set D := ∑ i, q i * k i
  have hS : ∑ i, p i ^ 2 = Q + 2 * D + K := sq_sum_add q k
  rw [hS]
  have hc2 : c ^ 2 = exp (-(Q + K)) := by
    simp only [c]; rw [← exp_nat_mul]; congr 1; push_cast; ring
  have h1 : c ^ 2 * exp (4 * (Q + 2 * D + K) / 2) = exp (Q + K + 4 * D) := by
    rw [hc2, ← exp_add]; congr 1; ring
  have h2 : c ^ 2 * exp (2 * (Q + 2 * D + K) / 2) = exp (2 * D) := by
    rw [hc2, ← exp_add]; congr 1; ring
  have h3 : c * exp ((Q + 2 * D + K) / 2) = e := by
    simp only [c, e]; rw [← exp_add]; congr 1; ring
  have he2 : e ^ 2 = exp (2 * D) := by simp only [e, D]; rw [← exp_nat_mul]; push_cast; ring_nf
  have : (1 / m : ℝ) ^ 2 * c ^ 2 * (m * exp (4 * (Q + 2 * D + K) / 2) +
        (m * m - m) * exp (2 * (Q + 2 * D + K) / 2)) -
      2 * e * (1 / m) * c * (m * exp ((Q + 2 * D + K) / 2)) + e ^ 2 =
      (1 / m : ℝ) * (c ^ 2 * exp (4 * (Q + 2 * D + K) / 2)) +
        (1 - 1 / m) * (c ^ 2 * exp (2 * (Q + 2 * D + K) / 2)) -
        2 * e * (c * exp ((Q + 2 * D + K) / 2)) + e ^ 2 := by
    field_simp
  rw [this, h1, h2, h3, he2]
  field_simp
  rw [he2, mul_comm 2 D]
  ring

lemma rfEst_measurable (q k : Fin d → ℝ) : Measurable (rfEst X q k) := by
  rw [rfEst_eq hind hmeas hlaw]
  exact (Finset.measurable_sum _ fun s _ => ((Finset.measurable_sum _ fun ui _ =>
    (hmeas ui).const_mul _).exp).const_mul _).const_mul _

lemma rfEst_memLp (q k : Fin d → ℝ) : MemLp (rfEst X q k) 2 μ := by
  rw [memLp_two_iff_integrable_sq (rfEst_measurable hind hmeas hlaw q k).aestronglyMeasurable]
  have hpt : (fun ω => rfEst X q k ω ^ 2) = fun ω =>
      (1 / m : ℝ) ^ 2 * exp (-((∑ i, q i ^ 2) + ∑ i, k i ^ 2) / 2) ^ 2 *
        ∑ s : Fin m, ∑ t : Fin m, exp (∑ ui, (rfCoef (q + k) s + rfCoef (q + k) t) ui * X ui ω) := by
    funext ω
    rw [rfEst_eq hind hmeas hlaw]
    simp only [← exp_mul_exp_sum, ← Finset.mul_sum, ← Finset.sum_mul, ← Finset.mul_sum]
    ring
  rw [hpt]
  exact (integrable_finsetSum _ (fun s _ => integrable_finsetSum _ (fun t _ =>
    gauss_exp_integrable hind hmeas hlaw _))).const_mul _

/-- **The variance of `m` random features is exactly `C(q,k)/m`** (so the standard deviation is
`O(m^{-1/2})`), with `C(q,k) = exp(|q|² + |k|² + 4 q·k) - exp(2 q·k) = e^{2q·k}(e^{|q+k|²} - 1)`. -/
theorem rf_variance (hm : 0 < m) (q k : Fin d → ℝ) :
    variance (rfEst X q k) μ =
      (exp (∑ i, q i ^ 2 + ∑ i, k i ^ 2 + 4 * ∑ i, q i * k i) -
        exp (2 * ∑ i, q i * k i)) / m := by
  rw [variance_eq_integral (rfEst_measurable hind hmeas hlaw q k).aemeasurable,
    rf_unbiased hind hmeas hlaw hm]
  exact rf_mse hind hmeas hlaw hm q k

/-- **Chebyshev, the `1/√m` rate**: the estimate misses `exp(q·k)` by `δ` or more with
probability at most `C(q,k)/(m δ²)`. -/
theorem rf_chebyshev (hm : 0 < m) (q k : Fin d → ℝ) {δ : ℝ} (hδ : 0 < δ) :
    μ {ω | δ ≤ |rfEst X q k ω - exp (∑ i, q i * k i)|} ≤
      ENNReal.ofReal ((exp (∑ i, q i ^ 2 + ∑ i, k i ^ 2 + 4 * ∑ i, q i * k i) -
        exp (2 * ∑ i, q i * k i)) / (m * δ ^ 2)) := by
  have := meas_ge_le_variance_div_sq (rfEst_memLp hind hmeas hlaw q k) hδ
  rw [rf_unbiased hind hmeas hlaw hm, rf_variance hind hmeas hlaw hm, div_div] at this
  exact this

end Main

end EP
