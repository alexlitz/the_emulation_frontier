import EmulationProofs.Norm
import EmulationProofs.PreNormRescue
import EmulationProofs.ResNorm

/-!
# Target norms (branch fr-targetnorm)

Norm placements on the TARGET side that a plain pre-norm host has to reproduce.

* **LayerNorm qk-norm from a mirror** (exact). The unpadded mirror `[x; −x]` is `Norm.lean`:
  `mean_mirror` (mean 0), `sumsq_mirror` (variance = mean square of `x`), `layerNorm_mirror`
  (LayerNorm with the same ε is the mirror of RMSNorm). Here the padded form `[x; −x; 0_e]`
  over `D = 2d + e` entries: mean 0, variance `(2/D) ∑ xᵢ²`, and LayerNorm with
  `ε · 2d / D`, times `√(2d / D)`, is `RMSNorm_ε(x)` on the first block (`ln_mirror_pad_*`).
* **The RoPE pair split** (exact). `R(−α) diag(a, b) R(β) = ((a+b)/2) R(β−α) + ((a−b)/2) J R(α+β)`
  (`rot_diag_split`): a gain that is constant on each rotation pair commutes with the rotation
  (`pair_const_comm`) and keeps the logit relative (`pair_const_relative`); a gain that splits a
  pair (`a ≠ b`) makes it depend on absolute position (`post_gain_not_relative`).
* **The dominated post-norm block** (limit, bound, exact identities). A block `Λ y` beside the
  stream `r` and a pad `κ`, read by an ε-free host norm over `D` entries, tends to
  `(√D/‖y‖) y` at rate `O(1/Λ²)` (`dominant_block_limit`, `dominant_block_bound`); the target's
  ε enters through a 1D factor of `‖y‖` (`postnorm_eps_factor`); the block is erased exactly by
  the pad's reciprocal (`pad_readout_erase`) and a relative fit error `e` of that reciprocal
  leaves a residue of relative size `e` (`fit_erase_residue`, `fit_scalar_residue`).
* **The loop-boundary norm write** (exact). With host ε = 0 and pad channels of squared norm
  `d ε`, the write `u ⊙ (g √(d/D) − κ/t)` turns the residual `x` into `RMSNorm_{ε,g}(x)`
  (`bnorm_pad_write`, from `const_norm_carry`).
* **q-latent pad isometry** (exact). An ε-free norm puts every input on the sphere `‖u‖² = D`
  (`rmsNorm_zero_sq_sum`, PreNormRescue.lean); after an isometry `W` a target latent norm is a
  FIXED multiple of `W u` (`qlatent_isometry_const`), so it folds into the next linear map
  (`qlatent_factor`).
* **V-head split** (exact): two heads with the same logits are one head with the concatenated
  values (`vhead_split`, `vhead_split_attn`).
* **Embedding-norm fold** (exact, untied): `E' = N ∘ E` (`emb_fold_untied`).

Euclidean norms are written out as `l2 x = √(∑ xᵢ²)` (the `Fin n → ℝ` norm in Mathlib is the
sup norm).
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-- The Euclidean norm, written out. -/
def l2 {n : ℕ} (x : Fin n → ℝ) : ℝ := √(∑ i, x i ^ 2)

lemma l2_nonneg {n : ℕ} (x : Fin n → ℝ) : 0 ≤ l2 x := Real.sqrt_nonneg _

lemma l2_smul {n : ℕ} (c : ℝ) (x : Fin n → ℝ) : l2 (c • x) = |c| * l2 x := by
  unfold l2
  simp only [Pi.smul_apply, smul_eq_mul, mul_pow, ← Finset.mul_sum]
  rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq_eq_abs]

/-! ## 1. LayerNorm qk-norm from a padded mirror -/

/-- The padded mirror `[x; −x; 0_e]`, width `d + d + e`. -/
def mirrorPad {d : ℕ} (e : ℕ) (x : Fin d → ℝ) : Fin (d + d + e) → ℝ := Fin.append (mirror x) 0

@[simp] lemma mirrorPad_left {d e : ℕ} (x : Fin d → ℝ) (i : Fin d) :
    mirrorPad e x (Fin.castAdd e (Fin.castAdd d i)) = x i := by
  simp [mirrorPad]

@[simp] lemma mirrorPad_mid {d e : ℕ} (x : Fin d → ℝ) (i : Fin d) :
    mirrorPad e x (Fin.castAdd e (Fin.natAdd d i)) = -x i := by
  rw [mirrorPad, Fin.append_left, mirror_right]

@[simp] lemma mirrorPad_right {d e : ℕ} (x : Fin d → ℝ) (j : Fin e) :
    mirrorPad e x (Fin.natAdd (d + d) j) = 0 := by
  simp [mirrorPad]

/-- **The padded mirror has mean 0** (exact; unpadded: `mean_mirror`). -/
theorem ln_mirror_pad_mean_zero {d e : ℕ} (x : Fin d → ℝ) : mean (mirrorPad e x) = 0 := by
  simp only [mean, Fin.sum_univ_add, mirrorPad_left, mirrorPad_mid, mirrorPad_right,
    Finset.sum_neg_distrib, add_neg_cancel, Finset.sum_const_zero, add_zero, zero_div]

/-- **The padded mirror's variance is `(2/D) ∑ xᵢ²`**, `D = 2d + e` (exact; unpadded:
`sumsq_mirror`). -/
theorem ln_mirror_pad_var {d e : ℕ} (x : Fin d → ℝ) :
    (∑ l, (mirrorPad e x l - mean (mirrorPad e x)) ^ 2) / (d + d + e : ℕ) =
      2 * (∑ i, x i ^ 2) / (d + d + e : ℕ) := by
  rw [ln_mirror_pad_mean_zero]
  simp only [sub_zero, Fin.sum_univ_add, mirrorPad_left, mirrorPad_mid, mirrorPad_right, neg_sq]
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, Finset.sum_const_zero,
    add_zero]
  ring

/-- **LayerNorm over `[x; −x; 0_e]` is RMSNorm** (exact; unpadded, same ε: `layerNorm_mirror`).
A LayerNorm of width `D = 2d + e` (weight 1, bias 0, epsilon `ε · 2d / D`) on the padded mirror,
multiplied by `√(2d / D)`, returns `RMSNorm_ε(x)` on the first block, `−RMSNorm_ε(x)` on the
second, and `0` on the pad. -/
theorem ln_mirror_pad_eq_rmsNorm {d e : ℕ} (hd : d ≠ 0) (eps : ℝ) (x : Fin d → ℝ) :
    (∀ i, √(((d + d : ℕ) : ℝ) / (d + d + e : ℕ)) *
        layerNorm (eps * (d + d : ℕ) / (d + d + e : ℕ)) 1 0 (mirrorPad e x)
          (Fin.castAdd e (Fin.castAdd d i)) = rmsNorm eps 1 x i) ∧
    (∀ i, √(((d + d : ℕ) : ℝ) / (d + d + e : ℕ)) *
        layerNorm (eps * (d + d : ℕ) / (d + d + e : ℕ)) 1 0 (mirrorPad e x)
          (Fin.castAdd e (Fin.natAdd d i)) = -rmsNorm eps 1 x i) ∧
    (∀ j, layerNorm (eps * (d + d : ℕ) / (d + d + e : ℕ)) 1 0 (mirrorPad e x)
          (Fin.natAdd (d + d) j) = 0) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hD : (0 : ℝ) < (d + d + e : ℕ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hd).trans_le (by omega)
  have hc : 0 < √(((d + d : ℕ) : ℝ) / (d + d + e : ℕ)) := Real.sqrt_pos.mpr (by positivity)
  have key : √((∑ l, (mirrorPad e x l - mean (mirrorPad e x)) ^ 2) / (d + d + e : ℕ) +
      eps * (d + d : ℕ) / (d + d + e : ℕ)) =
      √(((d + d : ℕ) : ℝ) / (d + d + e : ℕ)) * √((∑ l, x l ^ 2) / d + eps) := by
    rw [ln_mirror_pad_var, ← Real.sqrt_mul (by positivity)]
    congr 1
    push_cast
    field_simp
    ring
  have hfin : ∀ (a : ℝ), √(((d + d : ℕ) : ℝ) / (d + d + e : ℕ)) * (a /
      (√(((d + d : ℕ) : ℝ) / (d + d + e : ℕ)) * √((∑ l, x l ^ 2) / d + eps))) =
      a / √((∑ l, x l ^ 2) / d + eps) := by
    intro a
    rcases (Real.sqrt_nonneg ((∑ l, x l ^ 2) / d + eps)).lt_or_eq with h | h
    · field_simp
    · rw [← h]; simp
  refine ⟨fun i => ?_, fun i => ?_, fun j => ?_⟩
  · simp only [layerNorm, Pi.one_apply, one_mul, Pi.zero_apply, add_zero]
    rw [key, ln_mirror_pad_mean_zero, sub_zero, mirrorPad_left, hfin]
    simp [rmsNorm]
  · simp only [layerNorm, Pi.one_apply, one_mul, Pi.zero_apply, add_zero]
    rw [key, ln_mirror_pad_mean_zero, sub_zero, mirrorPad_mid, hfin]
    simp [rmsNorm, neg_div]
  · simp [layerNorm, ln_mirror_pad_mean_zero]

/-! ## 2. The RoPE pair split -/

/-- The 2×2 rotation `R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]` (the matrix of `rot θ`,
`rotM_mulVec`). -/
def rotM (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![cos θ, -sin θ; sin θ, cos θ]

/-- The reflection `J = diag(1, −1)`. -/
def reflJ : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]

/-- A gain on one rotation pair, `diag(a, b)`. -/
def diag2 (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, 0; 0, b]

lemma rotM_mulVec (θ : ℝ) (p : ℝ × ℝ) :
    (rotM θ).mulVec ![p.1, p.2] = ![(rot θ p).1, (rot θ p).2] := by
  ext i; fin_cases i <;> simp [rotM, rot, Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> ring

/-- **The pair split** (exact): `R(−α) diag(a, b) R(β) = ((a+b)/2) R(β−α) + ((a−b)/2) J R(α+β)`.
The mean gain rides the rotation (relative), the half-difference reads `α + β` (absolute). -/
theorem rot_diag_split (a b α β : ℝ) :
    rotM (-α) * diag2 a b * rotM β =
      ((a + b) / 2) • rotM (β - α) + ((a - b) / 2) • (reflJ * rotM (α + β)) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rotM, diag2, reflJ, Matrix.mul_apply, Fin.sum_univ_two, cos_add, sin_add, cos_sub,
      sin_sub] <;> ring

/-- **A pair-constant gain commutes with every rotation** (exact). -/
theorem pair_const_comm (c θ : ℝ) : diag2 c c * rotM θ = rotM θ * diag2 c c := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [rotM, diag2, Matrix.mul_apply, Fin.sum_univ_two] <;> ring

/-- The bilinear form `qᵀ R(−α) diag(a, b) R(β) k` of one pair. -/
def rotForm (a b : ℝ) (q k : Fin 2 → ℝ) (α β : ℝ) : ℝ :=
  ∑ i, q i * ((rotM (-α) * diag2 a b * rotM β).mulVec k) i

/-- **A pair-constant gain keeps the logit relative** (exact): it depends on `β − α` only. -/
theorem pair_const_relative (c : ℝ) (q k : Fin 2 → ℝ) (α β δ : ℝ) :
    rotForm c c q k (α + δ) (β + δ) = rotForm c c q k α β := by
  unfold rotForm
  rw [rot_diag_split, rot_diag_split]
  have : β + δ - (α + δ) = β - α := by ring
  simp only [this, sub_self, zero_div, zero_smul, add_zero]

/-- **A gain that splits a pair breaks relativity** (exact counterexample): if `a ≠ b`, the
form is not a function of `β − α`. Witness `q = k = e₁`, `α = β = 0`, `δ = π/2`: the values are
`a` and `b`. -/
theorem post_gain_not_relative (a b : ℝ) (hab : a ≠ b) :
    ∃ (q k : Fin 2 → ℝ) (α β δ : ℝ), rotForm a b q k α β ≠ rotForm a b q k (α + δ) (β + δ) := by
  refine ⟨![1, 0], ![1, 0], 0, 0, π / 2, ?_⟩
  simp [rotForm, rotM, diag2, Matrix.mul_apply, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  exact hab

/-! ## 3. The dominated post-norm block -/

section dominated

variable {m d : ℕ}

/-- **The dominated block** (limit). Host residual `[r; κ; Λ y]` (the stream `r`, a pad `κ`, the
target's post-norm input `y ≠ 0` written at scale `Λ`), read by an ε-free host norm over `D`
entries: the block's read `Λ y / R(Λ)`, `R(Λ) = √((‖r‖² + κ² + Λ²‖y‖²)/D)`, tends to
`(√D/‖y‖) y`, the normalised direction of `y`. -/
theorem dominant_block_limit (r : Fin m → ℝ) (y : Fin d → ℝ) (hy : y ≠ 0) (κ D : ℝ) (hD : 0 < D) :
    Tendsto (fun Λ : ℝ => (Λ / √((∑ i, r i ^ 2 + κ ^ 2 + Λ ^ 2 * ∑ i, y i ^ 2) / D)) • y) atTop
      (𝓝 ((√D / l2 y) • y)) := by
  apply Tendsto.smul_const
  set A := ∑ i, r i ^ 2 + κ ^ 2
  set B := ∑ i, y i ^ 2
  have hA : 0 ≤ A := by positivity
  have hB : 0 < B := sum_sq_pos_of_ne_zero y hy
  have hsB : 0 < √B := Real.sqrt_pos.mpr hB
  have hlim := (pad_ratio_limit A (√D / √B) hA).comp (tendsto_id.atTop_mul_const hsB)
  apply hlim.congr'
  filter_upwards [eventually_gt_atTop 0] with Λ hΛ
  simp only [Function.comp, id]
  have e1 : (Λ * √B) ^ 2 = Λ ^ 2 * B := by rw [mul_pow, Real.sq_sqrt hB.le]
  rw [e1, Real.sqrt_div (by positivity)]
  have : 0 < √(A + Λ ^ 2 * B) := Real.sqrt_pos.mpr (by positivity)
  have : 0 < √D := Real.sqrt_pos.mpr hD
  field_simp

/-- **The dominated block, rate** (bound): for `Λ > 0` the read is within
`√D (‖r‖² + κ²) / (2 Λ² ‖y‖²)` of `(√D/‖y‖) y` in Euclidean norm. -/
theorem dominant_block_bound (r : Fin m → ℝ) (y : Fin d → ℝ) (hy : y ≠ 0) (κ D Λ : ℝ)
    (hD : 0 < D) (hΛ : 0 < Λ) :
    l2 ((Λ / √((∑ i, r i ^ 2 + κ ^ 2 + Λ ^ 2 * ∑ i, y i ^ 2) / D)) • y - (√D / l2 y) • y) ≤
      √D * (∑ i, r i ^ 2 + κ ^ 2) / (2 * Λ ^ 2 * ∑ i, y i ^ 2) := by
  set A := ∑ i, r i ^ 2 + κ ^ 2
  set B := ∑ i, y i ^ 2
  have hA : 0 ≤ A := by positivity
  have hB : 0 < B := sum_sq_pos_of_ne_zero y hy
  have hl2 : l2 y = √B := rfl
  set sB := √B with hsBdef
  have hsB : 0 < sB := Real.sqrt_pos.mpr hB
  have hsB2 : sB ^ 2 = B := Real.sq_sqrt hB.le
  set s := √(A + Λ ^ 2 * B) with hsdef
  have hs2 : s ^ 2 = A + Λ ^ 2 * B := Real.sq_sqrt (by positivity)
  have hs : 0 < s := Real.sqrt_pos.mpr (by positivity)
  have ht : 0 < Λ * sB := by positivity
  have ht2 : (Λ * sB) ^ 2 = Λ ^ 2 * B := by rw [mul_pow, hsB2]
  have hts : Λ * sB ≤ s := by
    rw [hsdef]; apply Real.le_sqrt_of_sq_le; rw [ht2]; linarith
  have hsD : 0 < √D := Real.sqrt_pos.mpr hD
  have hR : √((A + Λ ^ 2 * B) / D) = s / √D := Real.sqrt_div (by positivity) D
  rw [← sub_smul, l2_smul, hR, hl2]
  have hst : 0 < s + Λ * sB := by positivity
  have hdiff : Λ * sB - s = -A / (s + Λ * sB) := by
    rw [eq_div_iff hst.ne']
    linear_combination Λ ^ 2 * hsB2 - hs2
  have hcoef : Λ / (s / √D) - √D / sB = -(√D * A / (s * sB * (s + Λ * sB))) := by
    have : Λ / (s / √D) - √D / sB = √D * (Λ * sB - s) / (s * sB) := by
      field_simp
    rw [this, hdiff]
    field_simp
  rw [hcoef, abs_neg, abs_of_nonneg (by positivity)]
  have e2 : √D * A / (s * sB * (s + Λ * sB)) * sB = √D * A / (s * (s + Λ * sB)) := by
    field_simp
  rw [e2, show 2 * Λ ^ 2 * B = 2 * (Λ * sB) ^ 2 by rw [ht2]; ring]
  apply div_le_div_of_nonneg_left (by positivity) (by positivity)
  nlinarith

/-- **The target's ε from a 1D factor** (exact). The normalised direction `(√D/‖y‖) y`, scaled by
`√(d/D)` and by the 1D factor `‖y‖/√(‖y‖² + d ε)` of `‖y‖`, is the target's post-norm
`y / √(‖y‖²/d + ε)`. -/
theorem postnorm_eps_factor (hd : d ≠ 0) {eps : ℝ} (heps : 0 ≤ eps) {D : ℝ} (hD : 0 < D)
    (y : Fin d → ℝ) (hy : y ≠ 0) :
    (fun i => √((d : ℝ) / D) * ((√D / l2 y) * y i) * (l2 y / √(∑ l, y l ^ 2 + d * eps))) =
      rmsNorm eps 1 y := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hB : 0 < ∑ l, y l ^ 2 := sum_sq_pos_of_ne_zero y hy
  have h1 : 0 < l2 y := Real.sqrt_pos.mpr hB
  have h2 : 0 < √(∑ l, y l ^ 2 + d * eps) := Real.sqrt_pos.mpr (by positivity)
  have h3 : 0 < √D := Real.sqrt_pos.mpr hD
  have h4 : 0 < √(d : ℝ) := Real.sqrt_pos.mpr hd'
  funext i
  have e : (∑ l, y l ^ 2) / d + eps = (∑ l, y l ^ 2 + d * eps) / d := by field_simp
  simp only [rmsNorm, Pi.one_apply, one_mul]
  rw [e, Real.sqrt_div (by positivity : (0 : ℝ) ≤ ∑ l, y l ^ 2 + d * eps) (d : ℝ),
    Real.sqrt_div hd'.le D]
  field_simp

/-- **The pad erases the block exactly** (exact). With the pad's read `t = κ/R` (`κ ≠ 0`,
`R ≠ 0`), the 1D reciprocal `κ/t = R` turns the block's read `B/R` back into `B`. -/
theorem pad_readout_erase {n : ℕ} {κ R : ℝ} (hκ : κ ≠ 0) (hR : R ≠ 0) (B : Fin n → ℝ) :
    (κ / (κ / R)) • (R⁻¹ • B) = B ∧ B - (κ / (κ / R)) • (R⁻¹ • B) = 0 := by
  have h : (κ / (κ / R)) • (R⁻¹ • B) = B := by
    rw [smul_smul]
    have : κ / (κ / R) * R⁻¹ = 1 := by field_simp
    rw [this, one_smul]
  exact ⟨h, by rw [h, sub_self]⟩

/-- **A fitted reciprocal leaves a proportional residue** (bound). If the fitted `φ` is within
relative error `e` of `κ/t` (`t = κ/R`), the erase leaves at most `e ‖B‖`. -/
theorem fit_erase_residue {n : ℕ} {κ R : ℝ} (hκ : κ ≠ 0) (hR : 0 < R) (B : Fin n → ℝ) (φ e : ℝ)
    (hφ : |φ - κ / (κ / R)| ≤ e * (κ / (κ / R))) :
    l2 (B - φ • (R⁻¹ • B)) ≤ e * l2 B := by
  have hk : κ / (κ / R) = R := by field_simp
  rw [hk] at hφ
  have e1 : B - φ • (R⁻¹ • B) = (-((φ - R) / R)) • B := by
    rw [smul_smul]; funext i; simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
    field_simp; ring
  rw [e1, l2_smul, abs_neg, abs_div, abs_of_pos hR]
  apply mul_le_mul_of_nonneg_right _ (l2_nonneg B)
  rw [div_le_iff₀ hR]; exact hφ

/-- **A scalar fit error is a relative error of the scaled vector** (bound):
`|ψ − ψ*| ≤ e |ψ*|` gives `‖ψ v − ψ* v‖ ≤ e ‖ψ* v‖`. -/
theorem fit_scalar_residue {n : ℕ} (v : Fin n → ℝ) (ψ ψs e : ℝ) (h : |ψ - ψs| ≤ e * |ψs|) :
    l2 (ψ • v - ψs • v) ≤ e * l2 (ψs • v) := by
  rw [← sub_smul, l2_smul, l2_smul, ← mul_assoc]
  exact mul_le_mul_of_nonneg_right h (l2_nonneg v)

end dominated

/-! ## 4. The loop-boundary norm write -/

/-- **The loop-boundary renorm write** (exact, with exact 1D factors). Host residual `[x; z]` over
`D = d + e` entries, host ε = 0, where the pad channels `z` have squared norm `d ε` (`κ = z j₀ ≠ 0`
one of them, the others e.g. zero). With `u` the host-normed residual and `t = u_{κ} = κ/R`, the
write `u ⊙ (g √(d/D) − κ/t)` added to the stream gives the target's `RMSNorm_{ε,g}(x)`
(per-channel gain `g`). The key step is `const_norm_carry` with `S = d ε`. -/
theorem bnorm_pad_write {d e : ℕ} (hd : d ≠ 0) (eps : ℝ) (x g : Fin d → ℝ) (z : Fin e → ℝ)
    (hz : ∑ j, z j ^ 2 = d * eps) (j0 : Fin e) (hj0 : z j0 ≠ 0) (i : Fin d) :
    x i + rmsNorm 0 1 (Fin.append x z) (Fin.castAdd e i) *
        (g i * √((d : ℝ) / (d + e : ℕ)) - z j0 / rmsNorm 0 1 (Fin.append x z) (Fin.natAdd d j0)) =
      rmsNorm eps g x i := by
  have hc := (const_norm_carry hd eps (d * eps) 1 x z 1 hz).1 i
  simp only [sub_self, zero_div] at hc
  have hpos : 0 < (∑ l : Fin (d + e), (Fin.append x z l) ^ 2) / (d + e : ℕ) + 0 := by
    have hde : (0 : ℝ) < (d + e : ℕ) := by
      exact_mod_cast (Nat.pos_of_ne_zero hd).trans_le (by omega)
    rw [add_zero]
    apply div_pos _ hde
    simp only [Fin.sum_univ_add, Fin.append_left, Fin.append_right]
    have : 0 < z j0 ^ 2 := by positivity
    have : z j0 ^ 2 ≤ ∑ j, z j ^ 2 :=
      Finset.single_le_sum (f := fun j => z j ^ 2) (fun _ _ => sq_nonneg _) (Finset.mem_univ j0)
    have : 0 ≤ ∑ l, x l ^ 2 := by positivity
    linarith
  have hR := Real.sqrt_pos.mpr hpos
  simp only [rmsNorm, Fin.append_left, Fin.append_right, Pi.one_apply, one_mul, mul_one] at hc ⊢
  set R := √((∑ l : Fin (d + e), (Fin.append x z l) ^ 2) / (d + e : ℕ) + 0)
  have hzR : z j0 / (z j0 / R) = R := by field_simp
  rw [hzR]
  calc x i + x i / R * (g i * √((d : ℝ) / (d + e : ℕ)) - R)
      = g i * (√((d : ℝ) / (d + e : ℕ)) * x i / R) := by field_simp; ring
    _ = g i * x i / √((∑ l, x l ^ 2) / d + eps) := by rw [hc]; ring

/-- **The loop-boundary write, scalar gain** (exact): `x + u ⊙ (g √(d/D) − κ/t) =
g x / √(‖x‖²/d + ε)`. -/
theorem bnorm_pad_write_const {d e : ℕ} (hd : d ≠ 0) (eps g : ℝ) (x : Fin d → ℝ) (z : Fin e → ℝ)
    (hz : ∑ j, z j ^ 2 = d * eps) (j0 : Fin e) (hj0 : z j0 ≠ 0) (i : Fin d) :
    x i + rmsNorm 0 1 (Fin.append x z) (Fin.castAdd e i) *
        (g * √((d : ℝ) / (d + e : ℕ)) - z j0 / rmsNorm 0 1 (Fin.append x z) (Fin.natAdd d j0)) =
      g * x i / √((∑ l, x l ^ 2) / d + eps) :=
  bnorm_pad_write hd eps x (fun _ => g) z hz j0 hj0 i

/-! ## 5. The q-latent pad isometry -/

/-- **After an ε-free norm and an isometry, a latent norm is a fixed scale** (exact). Host ε = 0
puts `u = RMSNorm₀(z)` on the sphere `∑ uᵢ² = D` (`rmsNorm_zero_sq_sum`); an isometry `W` into
`r` coordinates keeps it there, so `RMSNorm_{ε'}(W u) = w ⊙ W u / √(D/r + ε')` for every `z ≠ 0`. -/
theorem qlatent_isometry_const {D r : ℕ} (eps' : ℝ) (w : Fin r → ℝ)
    (W : (Fin D → ℝ) →ₗ[ℝ] (Fin r → ℝ)) (hW : ∀ v, ∑ i, (W v i) ^ 2 = ∑ i, v i ^ 2)
    (z : Fin D → ℝ) (hz : z ≠ 0) :
    rmsNorm eps' w (W (rmsNorm 0 1 z)) = fun i => w i * W (rmsNorm 0 1 z) i / √((D : ℝ) / r + eps') := by
  set u := rmsNorm 0 1 z with hu
  have hsq : ∑ i, (W u i) ^ 2 = D := by rw [hW, hu]; exact rmsNorm_zero_sq_sum z hz
  funext i
  unfold rmsNorm
  rw [hsq]

/-- **The latent norm folds into the next map** (exact). For every linear `Q` there is a linear
`Wb` with `Wb(RMSNorm_{ε'}(W u)) = Q u` for every host-normed `u = RMSNorm₀(z)`, `z ≠ 0`
(`Wb = √(D/r + ε') · Q ∘ W⁺`, `W⁺` a left inverse of the isometry). -/
theorem qlatent_factor {D r p : ℕ} (eps' : ℝ) (heps : 0 ≤ eps')
    (W : (Fin D → ℝ) →ₗ[ℝ] (Fin r → ℝ)) (hW : ∀ v, ∑ i, (W v i) ^ 2 = ∑ i, v i ^ 2)
    (Q : (Fin D → ℝ) →ₗ[ℝ] (Fin p → ℝ)) :
    ∃ Wb : (Fin r → ℝ) →ₗ[ℝ] (Fin p → ℝ), ∀ z : Fin D → ℝ, z ≠ 0 →
      Wb (rmsNorm eps' 1 (W (rmsNorm 0 1 z))) = Q (rmsNorm 0 1 z) := by
  rcases Nat.eq_zero_or_pos D with hD | hD
  · subst hD
    refine ⟨0, fun z hz => absurd (funext fun i => Fin.elim0 i) hz⟩
  have hker : LinearMap.ker W = ⊥ := by
    rw [LinearMap.ker_eq_bot']
    intro v hv
    have h0 : ∑ i, v i ^ 2 = 0 := by rw [← hW, hv]; simp
    funext i
    have := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => sq_nonneg (v j))).mp h0 i
      (Finset.mem_univ i)
    simpa using this
  obtain ⟨g, hg⟩ := LinearMap.exists_leftInverse_of_injective W hker
  have hr : 0 < r := by
    rcases Nat.eq_zero_or_pos r with h0 | h0
    · subst h0
      have := hW (fun _ => 1)
      simp at this
      norm_cast at this
      omega
    · exact h0
  set c := √((D : ℝ) / r + eps') with hcdef
  have hc : 0 < c := Real.sqrt_pos.mpr (by positivity)
  refine ⟨c • (Q ∘ₗ g), fun z hz => ?_⟩
  rw [qlatent_isometry_const eps' 1 W hW z hz]
  have e1 : (fun i => (1 : Fin r → ℝ) i * W (rmsNorm 0 1 z) i / c) = c⁻¹ • W (rmsNorm 0 1 z) := by
    funext i; simp [div_eq_inv_mul]
  have hgu : g (W (rmsNorm 0 1 z)) = rmsNorm 0 1 z := by
    simpa using LinearMap.congr_fun hg (rmsNorm 0 1 z)
  rw [e1, LinearMap.smul_apply, LinearMap.comp_apply, map_smul, map_smul, hgu, smul_smul,
    mul_inv_cancel₀ hc.ne', one_smul]

/-! ## 6. The V-head split -/

section vhead

variable {a b : ℕ} {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The output map of a head with concatenated values: `O [v₁; v₂] = O₁ v₁ + O₂ v₂`. -/
def vcat (O1 : (Fin a → ℝ) →ₗ[ℝ] V) (O2 : (Fin b → ℝ) →ₗ[ℝ] V) : (Fin (a + b) → ℝ) →ₗ[ℝ] V :=
  O1 ∘ₗ LinearMap.funLeft ℝ ℝ (Fin.castAdd b) + O2 ∘ₗ LinearMap.funLeft ℝ ℝ (Fin.natAdd a)

lemma vcat_append (O1 : (Fin a → ℝ) →ₗ[ℝ] V) (O2 : (Fin b → ℝ) →ₗ[ℝ] V) (v1 : Fin a → ℝ)
    (v2 : Fin b → ℝ) : vcat O1 O2 (Fin.append v1 v2) = O1 v1 + O2 v2 := by
  simp only [vcat, LinearMap.add_apply, LinearMap.comp_apply]
  congr 2 <;> funext i <;> simp

/-- **Two heads with the same weights are one head with concatenated values** (exact): any
weights `α` over any key set (no normalisation needed). -/
theorem vhead_split {ι : Type*} (S : Finset ι) (α : ι → ℝ) (v1 : ι → Fin a → ℝ)
    (v2 : ι → Fin b → ℝ) (O1 : (Fin a → ℝ) →ₗ[ℝ] V) (O2 : (Fin b → ℝ) →ₗ[ℝ] V) :
    O1 (∑ j ∈ S, α j • v1 j) + O2 (∑ j ∈ S, α j • v2 j) =
      vcat O1 O2 (∑ j ∈ S, α j • Fin.append (v1 j) (v2 j)) := by
  simp only [map_sum, map_smul, vcat_append, smul_add, Finset.sum_add_distrib]

/-- **The V-head split for softmax attention** (exact): two heads sharing the logits `s`. -/
theorem vhead_split_attn {ι : Type*} [Fintype ι] (S : Finset ι) (s : ι → ℝ) (v1 : ι → Fin a → ℝ)
    (v2 : ι → Fin b → ℝ) (O1 : (Fin a → ℝ) →ₗ[ℝ] V) (O2 : (Fin b → ℝ) →ₗ[ℝ] V) :
    O1 (attn S s v1) + O2 (attn S s v2) = vcat O1 O2 (attn S s (fun j => Fin.append (v1 j) (v2 j))) := by
  simp only [attn, map_smul, ← smul_add]
  rw [vhead_split S (fun j => exp (s j))]

end vhead

/-! ## 7. The embedding-norm fold -/

/-- **An embedding norm folds into an untied table** (exact). A target that applies `Nemb` to the
looked-up row (BLOOM / Gemma-style embedding norm) has the same first-layer input, hence the same
residual stream at every layer of any `stack`, as a host with table `E' v = Nemb (E v)` and no
embedding norm. A TIED head still reads `E`, so the host needs an untied head equal to `E`. -/
theorem emb_fold_untied {n T : ℕ} {Vocab : Type*} (Nemb : (Fin n → ℝ) → (Fin n → ℝ))
    (E : Vocab → Fin n → ℝ) (N : ℕ → (Fin n → ℝ) → (Fin n → ℝ))
    (f : ℕ → (Fin T → Fin n → ℝ) → (Fin T → Fin n → ℝ)) (X : Fin T → Vocab) :
    let E' : Vocab → Fin n → ℝ := fun v => Nemb (E v)
    (∀ v, E' v = Nemb (E v)) ∧
      ∀ l, stack N f (fun t => E' (X t)) l = stack N f (fun t => Nemb (E (X t))) l :=
  ⟨fun _ => rfl, fun _ => rfl⟩

end EP
