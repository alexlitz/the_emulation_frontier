import EmulationProofs.Norm

/-!
# RMSNorm ⊆ LayerNorm at width `n + 1` (branch fr-cost): the mean-zero isometry

`Norm.lean` proves RMSNorm ⊆ LayerNorm on the mirror `[h; -h]` (2× width). The mirror is one linear map that makes
the host's centring a no-op and keeps the norm; ANY linear `U : ℝⁿ → ℝᵐ` whose image is mean-zero
(`∑ᵢ (U x)ᵢ = 0`) and which preserves the sum of squares does the same (`layerNorm_subspace`), with the host epsilon
rescaled, `m · eps_h = n · eps_t`. Such a `U` exists as soon as `m ≥ n + 1` (`helmert`, `pad`), so the whole-model
emulation (`rmsNorm_sub_layerNorm_subspace`) costs `n + 1` hidden, not `2 n`. Every residual writer writes
`U (target write)`; every reader applies `Uᵀ`, the target norm gain and the scale `√(n / m)`.

The target norm gains are folded into the readers, so this covers pre-norm stacks (and the final norm read by an
untied head). A norm whose gained output is written into the residual (sandwich post-norms) or a tied head needs the
gain to commute with `U`, which the mirror provides and a generic `U` does not; the relation keeps the mirror there.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {n m : ℕ}

/-- **The norm** (exact): on a mean-zero, square-sum-preserving image `z = U x`, a LayerNorm of width `m` (weight 1,
bias 0, epsilon `eps_h` with `m eps_h = n eps_t`) is `√(m / n)` times `U` of the width-`n` RMSNorm (epsilon
`eps_t`). -/
theorem layerNorm_subspace (hn : 0 < n) (hm : 0 < m) (eps_t eps_h : ℝ) (heps : (m : ℝ) * eps_h = n * eps_t)
    (U : Matrix (Fin m) (Fin n) ℝ) (hmean : ∀ x, ∑ i, U.mulVec x i = 0)
    (hsq : ∀ x, ∑ i, (U.mulVec x i) ^ 2 = ∑ j, x j ^ 2) (x : Fin n → ℝ) :
    layerNorm eps_h 1 0 (U.mulVec x) = (√((m : ℝ) / n)) • U.mulVec (rmsNorm eps_t 1 x) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  have hmean0 : mean (U.mulVec x) = 0 := by simp [mean, hmean x]
  have h1 : (∑ l, U.mulVec x l ^ 2) / m + eps_h = (∑ j, x j ^ 2 + n * eps_t) / m := by
    rw [hsq x]; field_simp; linarith [heps]
  have h2 : (∑ l, x l ^ 2) / n + eps_t = (∑ j, x j ^ 2 + n * eps_t) / m * (m / n) := by
    field_simp
  have hr : rmsNorm eps_t 1 x = (1 / √((∑ l, x l ^ 2) / n + eps_t)) • x := by
    funext i; simp [rmsNorm, div_eq_inv_mul, mul_comm]
  funext i
  rw [hr, Matrix.mulVec_smul]
  simp only [layerNorm, hmean0, sub_zero, Pi.one_apply, one_mul, Pi.zero_apply, add_zero, Pi.smul_apply,
    smul_eq_mul]
  rw [h1, h2]
  have hmn : 0 < (m : ℝ) / n := by positivity
  rcases le_or_gt 0 ((∑ j, x j ^ 2 + n * eps_t) / m) with hA0 | hA0
  · rw [Real.sqrt_mul hA0]
    rcases eq_or_lt_of_le hA0 with hz | hpos
    · rw [← hz]; simp
    · have : √((∑ j, x j ^ 2 + n * eps_t) / m) ≠ 0 := (Real.sqrt_pos.mpr hpos).ne'
      have : √((m : ℝ) / n) ≠ 0 := (Real.sqrt_pos.mpr hmn).ne'
      field_simp
  · have hneg : (∑ j, x j ^ 2 + n * eps_t) / m * (m / n) < 0 := mul_neg_of_neg_of_pos hA0 hmn
    rw [Real.sqrt_eq_zero'.mpr hA0.le, Real.sqrt_eq_zero'.mpr hneg.le]
    simp

/-- **RMSNorm ⊆ LayerNorm, the whole model, at the width of any mean-zero isometry** (exact). Target: an RMSNorm stack
of width `n` (gains `γ l`, epsilon `eps_t`, sublayers `f l`, embedding `e`, head `head` reading the final RMSNorm
`γf`). Host: a LayerNorm stack of width `m` (weights 1, biases 0, epsilon `eps_h`, `m eps_h = n eps_t`) whose
writers write `U (target write)` and whose readers read `γ ⊙ (√(n / m) V y)`, `V` a left inverse of `U` (`Uᵀ`). The
host residual is `U` of the target's at every layer and the head sees the target's final RMSNorm. -/
theorem rmsNorm_sub_layerNorm_subspace {T : ℕ} (hn : 0 < n) (hm : 0 < m) (eps_t eps_h : ℝ)
    (heps : (m : ℝ) * eps_h = n * eps_t) (U : Matrix (Fin m) (Fin n) ℝ) (V : Matrix (Fin n) (Fin m) ℝ)
    (hmean : ∀ x, ∑ i, U.mulVec x i = 0) (hsq : ∀ x, ∑ i, (U.mulVec x i) ^ 2 = ∑ j, x j ^ 2)
    (hinv : ∀ x, V.mulVec (U.mulVec x) = x) (γ : ℕ → Fin n → ℝ)
    (f : ℕ → (Fin T → Fin n → ℝ) → (Fin T → Fin n → ℝ)) (e : Fin T → Fin n → ℝ)
    (γf : Fin n → ℝ) {Out : Type*} (head : (Fin n → ℝ) → Out) (L : ℕ) :
    let read : (Fin n → ℝ) → (Fin m → ℝ) → (Fin n → ℝ) := fun g y i => g i * (√((n : ℝ) / m) * V.mulVec y i)
    let hostNorm : ℕ → (Fin m → ℝ) → (Fin m → ℝ) := fun _ => layerNorm eps_h 1 0
    let hostF : ℕ → (Fin T → Fin m → ℝ) → (Fin T → Fin m → ℝ) :=
      fun l z t => U.mulVec (f l (fun s => read (γ l) (z s)) t)
    let hostE : Fin T → Fin m → ℝ := fun t => U.mulVec (e t)
    (∀ l, stack hostNorm hostF hostE l = fun t => U.mulVec (stack (fun l => rmsNorm eps_t (γ l)) f e l t)) ∧
    (∀ t, head (read γf (layerNorm eps_h 1 0 (stack hostNorm hostF hostE L t))) =
      head (rmsNorm eps_t γf (stack (fun l => rmsNorm eps_t (γ l)) f e L t))) := by
  intro read hostNorm hostF hostE
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  have hsc : √((n : ℝ) / m) * √((m : ℝ) / n) = 1 := by
    rw [← Real.sqrt_mul (by positivity)]
    rw [show (n : ℝ) / m * (m / n) = 1 by field_simp]; simp
  -- reading the host norm of `U h` gives the target's gained RMSNorm of `h`
  have hread : ∀ (g : Fin n → ℝ) (h : Fin n → ℝ),
      read g (layerNorm eps_h 1 0 (U.mulVec h)) = rmsNorm eps_t g h := by
    intro g h
    rw [layerNorm_subspace hn hm eps_t eps_h heps U hmean hsq h]
    funext i
    simp only [read]
    rw [Matrix.mulVec_smul, hinv]
    simp only [Pi.smul_apply, smul_eq_mul, rmsNorm, Pi.one_apply, one_mul]
    set r := √((∑ l, h l ^ 2) / n + eps_t)
    linear_combination (g i * (h i / r)) * hsc
  have key : ∀ l, stack hostNorm hostF hostE l =
      fun t => U.mulVec (stack (fun l => rmsNorm eps_t (γ l)) f e l t) := by
    intro l
    induction l with
    | zero => rfl
    | succ l ih =>
      funext t
      simp only [stack, ih, hostF, hostNorm, hread, Pi.add_apply]
      rw [Matrix.mulVec_add]
  refine ⟨key, fun t => ?_⟩
  rw [key, hread]

/-! ## A witness at width `n + 1` and zero padding -/

/-- The Helmert-type map `x ↦ (x - a S, -b S)`, `S = ∑ x`, `b = 1 / √(n + 1)`, `a = (1 - b) / n`. -/
def helmert (n : ℕ) : Matrix (Fin (n + 1)) (Fin n) ℝ :=
  fun i j => Fin.lastCases (-(1 / √((n : ℝ) + 1)))
    (fun i : Fin n => (if i = j then 1 else 0) - (1 - 1 / √((n : ℝ) + 1)) / n) i

lemma helmert_apply (x : Fin n → ℝ) :
    (helmert n).mulVec x = Fin.snoc (fun i => x i - (1 - 1 / √((n : ℝ) + 1)) / n * ∑ j, x j)
      (-(1 / √((n : ℝ) + 1)) * ∑ j, x j) := by
  funext i
  refine Fin.lastCases ?_ (fun i => ?_) i
  · simp [helmert, Matrix.mulVec, dotProduct, Finset.mul_sum]
  · simp [helmert, Matrix.mulVec, dotProduct, sub_mul, Finset.sum_sub_distrib, Finset.mul_sum]

lemma helmert_mean (hn : 0 < n) (x : Fin n → ℝ) : ∑ i, (helmert n).mulVec x i = 0 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [helmert_apply, Fin.sum_univ_castSucc]
  simp only [Fin.snoc_castSucc, Fin.snoc_last, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

lemma helmert_sq (hn : 0 < n) (x : Fin n → ℝ) : ∑ i, ((helmert n).mulVec x i) ^ 2 = ∑ j, x j ^ 2 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hb : √((n : ℝ) + 1) ^ 2 = n + 1 := Real.sq_sqrt (by positivity)
  have hb0 : √((n : ℝ) + 1) ≠ 0 := (Real.sqrt_pos.mpr (by positivity)).ne'
  rw [helmert_apply, Fin.sum_univ_castSucc]
  simp only [Fin.snoc_castSucc, Fin.snoc_last]
  set S := ∑ j, x j
  have : ∑ i : Fin n, (x i - (1 - 1 / √((n : ℝ) + 1)) / n * S) ^ 2 =
      ∑ j, x j ^ 2 - 2 * ((1 - 1 / √((n : ℝ) + 1)) / n) * S * S
        + n * ((1 - 1 / √((n : ℝ) + 1)) / n * S) ^ 2 := by
    simp only [sub_sq, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, ← Finset.sum_mul, ← Finset.mul_sum]
    ring
  rw [this]
  generalize √((n : ℝ) + 1) = r at hb hb0 ⊢
  field_simp
  linear_combination (-S ^ 2) * hb

/-- `helmert`'s left inverse is its transpose. -/
lemma helmert_inv (hn : 0 < n) (x : Fin n → ℝ) : (helmert n).transpose.mulVec ((helmert n).mulVec x) = x := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hb : √((n : ℝ) + 1) ^ 2 = n + 1 := Real.sq_sqrt (by positivity)
  have hb0 : √((n : ℝ) + 1) ≠ 0 := (Real.sqrt_pos.mpr (by positivity)).ne'
  funext j
  rw [helmert_apply]
  simp only [Matrix.mulVec, dotProduct, Matrix.transpose_apply, Fin.sum_univ_castSucc, helmert,
    Fin.lastCases_castSucc, Fin.lastCases_last, Fin.snoc_castSucc, Fin.snoc_last]
  simp only [sub_mul, Finset.sum_sub_distrib, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ,
    ite_true, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  generalize √((n : ℝ) + 1) = r at hb hb0 ⊢
  field_simp
  linear_combination (-∑ j, x j) * hb

/-- Zero padding keeps both properties: `U` of width `m` extended by `p` zero rows. -/
def pad (p : ℕ) (U : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin (m + p)) (Fin n) ℝ :=
  fun i j => Fin.addCases (fun i => U i j) (fun _ => 0) i

lemma pad_apply (p : ℕ) (U : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    (pad p U).mulVec x = Fin.append (U.mulVec x) 0 := by
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · simp [pad, Matrix.mulVec, dotProduct]
  · simp [pad, Matrix.mulVec, dotProduct]

/-- **Existence at width `≥ n + 1`**: a mean-zero, square-sum-preserving `U` with a left inverse. So the host needs
`n + 1` hidden for an RMSNorm target of width `n` (the relation's `cost_norm_subspace`). -/
theorem subspace_exists (hn : 0 < n) (p : ℕ) :
    ∃ (U : Matrix (Fin (n + 1 + p)) (Fin n) ℝ) (V : Matrix (Fin n) (Fin (n + 1 + p)) ℝ),
      (∀ x, ∑ i, U.mulVec x i = 0) ∧ (∀ x, ∑ i, (U.mulVec x i) ^ 2 = ∑ j, x j ^ 2) ∧
      (∀ x, V.mulVec (U.mulVec x) = x) := by
  refine ⟨pad p (helmert n), fun j i => Fin.addCases (fun i => helmert n i j) (fun _ => 0) i, ?_, ?_, ?_⟩
  · intro x
    rw [pad_apply, Fin.sum_univ_add]
    simp [helmert_mean hn x]
  · intro x
    rw [pad_apply, Fin.sum_univ_add]
    simp [helmert_sq hn x]
  · intro x
    have h := helmert_inv hn x
    rw [pad_apply]
    funext j
    have := congrFun h j
    simp only [Matrix.mulVec, dotProduct, Fin.sum_univ_add, Fin.addCases_left, Fin.addCases_right,
      Fin.append_left, Fin.append_right, zero_mul, Finset.sum_const_zero, add_zero] at this ⊢
    simpa [Matrix.transpose_apply] using this

end EP
