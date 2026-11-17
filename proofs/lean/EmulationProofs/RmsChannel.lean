import EmulationProofs.PreNormRescue

/-!
# The RMS channel: constants through a bias-free pre-norm (branch fr-bias)

A bias-free pre-norm host reads only `h / R(h)`, so a constant channel comes out divided by the
token's RMS (`const_readout_channel`): a constant readout needs a channel PROPORTIONAL to the RMS,
and no analytic write is one exactly (`no_prenorm_homogeneous_channel`). Two ways around it:

* **The RMS channel** (an approximation, measured end to end in model_subsets
  `norm/rms_channel.py`). The host residual is `[x; κ; ζ]` with host `ε = 0`, a pad `κ² = d ε`
  (the target's `ε`, `const_norm_carry`) and `ζ = λ r(x)`, `r(x) = √(‖x‖²/d + ε)`. Then the host
  RMS is `r · s` with `s = √((d + λ²)/(d + 2))` a CONSTANT (`rmsChannel_fresh_read`): the stream
  reads the target's normalisation and `ζ` reads the constant `λ / s`, a bias column for every
  linear map of the sublayer. After the stream moves, a bias-free GLU "refresh" rewrites `ζ` from
  the two normed coordinates `t = κ/R` and `a = ζ₀/R` (`rmsChannel_refresh`: the exact identity it
  approximates, `R = κ/t` being a 1-D reciprocal). Its fit errors do not accumulate
  (`rmsChannel_refresh_error`), a relative error `η` in `ζ` moves the constant read by at most
  `|η|` and the stream read by at most `λ² |(1+η)² − 1| / d` (`rmsChannel_perturb`).
* **The dominant constant channel** (a limit, REFUTED as a construction for pre-norm targets).
  Residual `[x; C]`, host `ε = 0`: the scaled stream read is `w x / √(1 + ‖x‖²/C²)` and the
  constant read `w_C / √(1 + ‖x‖²/C²)` (`dominant_channel_read`), within `(‖x‖²/C²)/2` of the
  affine, NORM-FREE read `w x` and of `w_C` (`dominant_channel_rate`). That limit is homogeneous of
  degree 1 in `x`, so it is never a fixed multiple of an RMSNorm read on both `x` and `2x`
  (`dominant_limit_not_normalised`): the error plateaus at the drop-the-norm error.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-- The target's RMS `r(x) = √(‖x‖²/d + ε)`. -/
def tRms {d : ℕ} (eps : ℝ) (x : Fin d → ℝ) : ℝ := √((∑ l, x l ^ 2) / d + eps)

lemma tRms_pos {d : ℕ} {eps : ℝ} (heps : 0 < eps) (x : Fin d → ℝ) : 0 < tRms eps x := by
  unfold tRms
  apply Real.sqrt_pos.mpr
  have : 0 ≤ (∑ l, x l ^ 2) / d := div_nonneg (sum_nonneg fun _ _ => sq_nonneg _) (Nat.cast_nonneg _)
  linarith

lemma tRms_sq {d : ℕ} {eps : ℝ} (heps : 0 ≤ eps) (x : Fin d → ℝ) :
    tRms eps x ^ 2 = (∑ l, x l ^ 2) / d + eps := by
  unfold tRms
  rw [Real.sq_sqrt]
  have : 0 ≤ (∑ l, x l ^ 2) / d := div_nonneg (sum_nonneg fun _ _ => sq_nonneg _) (Nat.cast_nonneg _)
  linarith

/-- The two extra channels `[κ; ζ]`. -/
def padZeta (κ ζ : ℝ) : Fin 2 → ℝ := ![κ, ζ]

/-- **A fresh RMS channel makes every read of the sublayer exact, and `ζ` a constant** (exact).
Host residual `[x; κ; λ r(x)]`, `κ² = d ε`, host `ε = 0` over `d + 2` channels: the host RMS is
`r(x) · s` with `s = √((d + λ²)/(d + 2))`, independent of `x`. So the stream coordinate `i` reads
`w i * x i / (r(x) s)` (the target's normalisation up to the fixed `s`) and `ζ` reads
`w_ζ λ / s`, the same for every token. -/
theorem rmsChannel_fresh_read {d : ℕ} (hd : d ≠ 0) {eps : ℝ} (heps : 0 ≤ eps) (lam : ℝ)
    (w x : Fin d → ℝ) (wz : Fin 2 → ℝ) :
    let r := tRms eps x
    let s := √((d + lam ^ 2) / (d + 2 : ℕ))
    let h : Fin (d + 2) → ℝ := Fin.append x (padZeta (√(d * eps)) (lam * r))
    √((∑ l, h l ^ 2) / (d + 2 : ℕ) + 0) = r * s ∧
    (∀ i, rmsNorm 0 (Fin.append w wz) h (Fin.castAdd 2 i) = w i * x i / (r * s)) ∧
    rmsNorm 0 (Fin.append w wz) h (Fin.natAdd d 1) = wz 1 * (lam * r) / (r * s) := by
  intro r s h
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hD : (0 : ℝ) < (d + 2 : ℕ) := by positivity
  have hsq : (∑ l, h l ^ 2) = d * r ^ 2 + lam ^ 2 * r ^ 2 := by
    have hr2 : r ^ 2 = (∑ l, x l ^ 2) / d + eps := tRms_sq heps x
    simp only [h, Fin.sum_univ_add, Fin.append_left, Fin.append_right, Fin.sum_univ_two, padZeta,
      Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [Real.sq_sqrt (by positivity), mul_pow, hr2]
    field_simp
    ring
  have key : √((∑ l, h l ^ 2) / (d + 2 : ℕ) + 0) = r * s := by
    rw [add_zero, hsq, show (d * r ^ 2 + lam ^ 2 * r ^ 2) / ((d + 2 : ℕ) : ℝ) =
      r ^ 2 * ((d + lam ^ 2) / (d + 2 : ℕ)) by ring]
    rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (show 0 ≤ r from Real.sqrt_nonneg _)]
  refine ⟨key, fun i => ?_, ?_⟩
  · simp only [rmsNorm, Fin.append_left]
    rw [key]
    simp [h]
  · simp only [rmsNorm, Fin.append_right]
    rw [key]
    simp [h, padZeta]

/-- The constant read of a fresh channel: `w_ζ λ / s` (the factor `r` cancels; `ε > 0`). -/
theorem rmsChannel_const {d : ℕ} (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) (lam : ℝ)
    (w x : Fin d → ℝ) (wz : Fin 2 → ℝ) :
    rmsNorm 0 (Fin.append w wz) (Fin.append x (padZeta (√(d * eps)) (lam * tRms eps x)))
      (Fin.natAdd d 1) = wz 1 * lam / √((d + lam ^ 2) / (d + 2 : ℕ)) := by
  rw [(rmsChannel_fresh_read hd heps.le lam w x wz).2.2]
  have := tRms_pos heps x
  field_simp

/-- **The refresh identity** (exact). With the pad `κ > 0`, a stale `ζ₀` and the host RMS
`R = √((‖x‖² + κ² + ζ₀²)/D)` (`κ² = d ε`, `D = d + 2`), the two normed coordinates the refresh
reads, `t = κ/R` and `a = ζ₀/R`, determine the target's RMS: `r(x) = (κ/t) √((D − a²)/d)`; so
`ζ₀ + (κ/t)(λ √((D − a²)/d) − a) = λ r(x)`. The refresh FFN approximates `κ/t` (a reciprocal of
one normed coordinate) by bias-free GLU units. -/
theorem rmsChannel_refresh {d : ℕ} (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) (lam ζ₀ : ℝ)
    (x : Fin d → ℝ) :
    let κ := √(d * eps)
    let D : ℝ := (d + 2 : ℕ)
    let R := √(((∑ l, x l ^ 2) + κ ^ 2 + ζ₀ ^ 2) / D)
    let t := κ / R
    let a := ζ₀ / R
    (κ / t) * √((D - a ^ 2) / d) = tRms eps x ∧
    ζ₀ + (κ / t) * (lam * √((D - a ^ 2) / d) - a) = lam * tRms eps x := by
  intro κ D R t a
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hD : (0 : ℝ) < D := by simp only [D]; positivity
  have hκ : 0 < κ := Real.sqrt_pos.mpr (by positivity)
  have hS : 0 ≤ ∑ l, x l ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
  have hκ2 : κ ^ 2 = d * eps := Real.sq_sqrt (by positivity)
  have hR : 0 < R := Real.sqrt_pos.mpr (by rw [hκ2]; positivity)
  have hR2 : R ^ 2 = ((∑ l, x l ^ 2) + κ ^ 2 + ζ₀ ^ 2) / D :=
    Real.sq_sqrt (by rw [hκ2]; positivity)
  have hkt : κ / t = R := by simp only [t]; field_simp
  have hinner : (D - a ^ 2) / d = tRms eps x ^ 2 / R ^ 2 := by
    rw [tRms_sq heps.le]
    simp only [a]
    rw [div_pow]
    field_simp
    rw [hR2, hκ2]
    field_simp
    ring
  have hfirst : (κ / t) * √((D - a ^ 2) / d) = tRms eps x := by
    rw [hkt, hinner, Real.sqrt_div (sq_nonneg _), Real.sqrt_sq (tRms_pos heps x).le,
      Real.sqrt_sq hR.le]
    field_simp
  refine ⟨hfirst, ?_⟩
  have : (κ / t) * (lam * √((D - a ^ 2) / d) - a) = lam * tRms eps x - ζ₀ := by
    rw [mul_sub, mul_left_comm, hfirst, hkt]
    simp only [a]
    field_simp
  linarith

/-- **Refresh errors do not accumulate** (exact). If the refresh computes its two terms with
relative errors `e₂` (the `λ r` term) and `e₁` (the `ζ₀` term), the new channel is
`λ r (1 + e₂) − ζ₀ e₁`: the old channel enters only through `e₁`, so the error is set by the fit,
not by the number of refreshes. -/
theorem rmsChannel_refresh_error (lamr ζ₀ e₁ e₂ : ℝ) :
    ζ₀ + (lamr * (1 + e₂) - ζ₀ * (1 + e₁)) = lamr * (1 + e₂) - ζ₀ * e₁ := by ring

/-- **A relative error in the channel moves the constant read by at most that much, and the
stream read by `O(λ²/d)`** (exact bounds). With `ζ = λ r u` (`u = 1 + η > 0`) instead of `λ r`,
the host RMS is `r √((d + λ² u²)/D)`; relative to the fresh read the constant read is multiplied by
`f = u √((d + λ²)/(d + λ² u²))` and the stream read by `g = √((d + λ²)/(d + λ² u²))`, with
`|f − 1| ≤ |u − 1|` and `|g − 1| ≤ λ² |u² − 1| / d`. -/
theorem rmsChannel_perturb {d lam u : ℝ} (hd : 0 < d) (hu : 0 < u) :
    let g := √((d + lam ^ 2) / (d + lam ^ 2 * u ^ 2))
    |u * g - 1| ≤ |u - 1| ∧ |g - 1| ≤ lam ^ 2 * |u ^ 2 - 1| / d := by
  intro g
  have hl : 0 ≤ lam ^ 2 := sq_nonneg _
  have hden : 0 < d + lam ^ 2 * u ^ 2 := by positivity
  have hg0 : 0 ≤ g := Real.sqrt_nonneg _
  have hg2 : g ^ 2 = (d + lam ^ 2) / (d + lam ^ 2 * u ^ 2) := Real.sq_sqrt (by positivity)
  constructor
  · rcases le_total 1 u with h1 | h1
    · -- u ≥ 1: g ≤ 1 and u g ≥ 1
      have hg1 : g ≤ 1 := by
        rw [show g = √((d + lam ^ 2) / (d + lam ^ 2 * u ^ 2)) from rfl, Real.sqrt_le_one]
        rw [div_le_one hden]
        nlinarith [mul_le_mul_of_nonneg_left (one_le_pow₀ h1 : (1 : ℝ) ≤ u ^ 2) hl]
      have hug : 1 ≤ u * g := by
        have : 1 ≤ (u * g) ^ 2 := by
          rw [mul_pow, hg2, mul_div_assoc', le_div_iff₀ hden]
          nlinarith [mul_le_mul_of_nonneg_left (one_le_pow₀ h1 : (1 : ℝ) ≤ u ^ 2) hd.le]
        nlinarith [mul_nonneg hu.le hg0]
      rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)]
      nlinarith
    · -- u ≤ 1: g ≥ 1 and u g ≤ 1
      have hg1 : 1 ≤ g := by
        rw [show g = √((d + lam ^ 2) / (d + lam ^ 2 * u ^ 2)) from rfl, Real.one_le_sqrt]
        rw [one_le_div hden]
        nlinarith [mul_le_mul_of_nonneg_left (pow_le_one₀ hu.le h1 : u ^ 2 ≤ 1) hl]
      have hug : u * g ≤ 1 := by
        have : (u * g) ^ 2 ≤ 1 := by
          rw [mul_pow, hg2, mul_div_assoc', div_le_iff₀ hden]
          nlinarith [mul_le_mul_of_nonneg_left (pow_le_one₀ hu.le h1 : u ^ 2 ≤ 1) hd.le]
        nlinarith [mul_nonneg hu.le hg0]
      rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)]
      nlinarith
  · have hgp : 1 ≤ g + 1 := by linarith
    have hdiff : g ^ 2 - 1 = lam ^ 2 * (1 - u ^ 2) / (d + lam ^ 2 * u ^ 2) := by
      rw [hg2]; field_simp; ring
    have e : g - 1 = (g ^ 2 - 1) / (g + 1) := by field_simp; ring
    rw [e, abs_div, abs_of_pos (by linarith : (0 : ℝ) < g + 1), hdiff, abs_div,
      abs_of_pos hden, abs_mul, abs_of_nonneg hl, abs_sub_comm 1]
    calc lam ^ 2 * |u ^ 2 - 1| / (d + lam ^ 2 * u ^ 2) / (g + 1)
        ≤ lam ^ 2 * |u ^ 2 - 1| / (d + lam ^ 2 * u ^ 2) :=
          div_le_self (by positivity) hgp
      _ ≤ lam ^ 2 * |u ^ 2 - 1| / d :=
          div_le_div_of_nonneg_left (by positivity) hd (by nlinarith)

/-! ## The dominant constant channel: a bias, but no normalisation -/

/-- **The dominated read** (exact). Host residual `[x; C]` (`C > 0`), host `ε = 0` over `d + 1`
channels: the stream read times `C/√(d+1)` is `w i x i / √(1 + ‖x‖²/C²)`, and the constant read
divided by `√(d+1)` is `w_C / √(1 + ‖x‖²/C²)`. -/
theorem dominant_channel_read {d : ℕ} (w x : Fin d → ℝ) (wC C : ℝ) (hC : 0 < C) :
    let h : Fin (d + 1) → ℝ := Fin.append x ![C]
    let W : Fin (d + 1) → ℝ := Fin.append w ![wC]
    let u := (∑ l, x l ^ 2) / C ^ 2
    (∀ i, C / √((d + 1 : ℕ) : ℝ) * rmsNorm 0 W h (Fin.castAdd 1 i) = w i * x i / √(1 + u)) ∧
    rmsNorm 0 W h (Fin.natAdd d 0) / √((d + 1 : ℕ) : ℝ) = wC / √(1 + u) := by
  intro h W u
  have hD : (0 : ℝ) < (d + 1 : ℕ) := by positivity
  have hS : 0 ≤ ∑ l, x l ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
  have hu : 0 ≤ u := div_nonneg hS (sq_nonneg _)
  have key : √((∑ l, h l ^ 2) / (d + 1 : ℕ) + 0) = C / √((d + 1 : ℕ) : ℝ) * √(1 + u) := by
    have : (∑ l, h l ^ 2) / ((d + 1 : ℕ) : ℝ) + 0 =
        (C / √((d + 1 : ℕ) : ℝ)) ^ 2 * (1 + u) := by
      simp only [h, u, Fin.sum_univ_add, Fin.append_left, Fin.append_right, Fin.sum_univ_one,
        Matrix.cons_val_zero, add_zero]
      rw [div_pow, Real.sq_sqrt hD.le]
      field_simp
      ring
    rw [this, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by positivity)]
  have hsq : 0 < √(1 + u) := Real.sqrt_pos.mpr (by linarith)
  have hsD : 0 < √((d + 1 : ℕ) : ℝ) := Real.sqrt_pos.mpr hD
  refine ⟨fun i => ?_, ?_⟩
  · simp only [rmsNorm, Fin.append_left, W, h]
    rw [key]
    field_simp
  · simp only [rmsNorm, Fin.append_right, W, h, Matrix.cons_val_zero]
    rw [key]
    field_simp

/-- `|1 − 1/√(1+u)| ≤ u/2` for `u ≥ 0`. -/
lemma one_sub_inv_sqrt_le {u : ℝ} (hu : 0 ≤ u) : |1 - 1 / √(1 + u)| ≤ u / 2 := by
  have hs : 1 ≤ √(1 + u) := Real.one_le_sqrt.mpr (by linarith)
  have hpos : 0 < √(1 + u) := by linarith
  have hle : √(1 + u) ≤ 1 + u / 2 := by
    rw [Real.sqrt_le_left (by linarith)]
    nlinarith
  have h0 : 0 ≤ 1 - 1 / √(1 + u) := by rw [sub_nonneg, div_le_one hpos]; exact hs
  rw [abs_of_nonneg h0, show 1 - 1 / √(1 + u) = (√(1 + u) - 1) / √(1 + u) by field_simp]
  calc (√(1 + u) - 1) / √(1 + u) ≤ √(1 + u) - 1 := div_le_self (by linarith) hs
    _ ≤ u / 2 := by linarith

/-- **The dominated read tends to the norm-free affine read at rate `‖x‖²/(2C²)`**. -/
theorem dominant_channel_rate {d : ℕ} (w x : Fin d → ℝ) (wC C : ℝ) (hC : 0 < C) (i : Fin d) :
    let h : Fin (d + 1) → ℝ := Fin.append x ![C]
    let W : Fin (d + 1) → ℝ := Fin.append w ![wC]
    let u := (∑ l, x l ^ 2) / C ^ 2
    |C / √((d + 1 : ℕ) : ℝ) * rmsNorm 0 W h (Fin.castAdd 1 i) - w i * x i| ≤ |w i * x i| * (u / 2) ∧
    |rmsNorm 0 W h (Fin.natAdd d 0) / √((d + 1 : ℕ) : ℝ) - wC| ≤ |wC| * (u / 2) := by
  intro h W u
  obtain ⟨h1, h2⟩ := dominant_channel_read w x wC C hC
  have hu : 0 ≤ u := div_nonneg (sum_nonneg fun _ _ => sq_nonneg _) (sq_nonneg _)
  have hb := one_sub_inv_sqrt_le hu
  constructor
  · rw [h1 i, show w i * x i / √(1 + u) - w i * x i = -(w i * x i) * (1 - 1 / √(1 + u)) by ring,
      abs_mul, abs_neg]
    exact mul_le_mul_of_nonneg_left hb (abs_nonneg _)
  · rw [h2, show wC / √(1 + u) - wC = -wC * (1 - 1 / √(1 + u)) by ring, abs_mul, abs_neg]
    exact mul_le_mul_of_nonneg_left hb (abs_nonneg _)

/-- **The dominant-channel limit is not a normalisation** (the refutation). The limit read
`x ↦ c · w ⊙ x` is homogeneous of degree 1; an RMSNorm read (any `ε ≥ 0`) is not: no constant `c`
matches it on both `x` and `2x` at a coordinate where `w i x i ≠ 0`. So the dominated host computes
the target's sublayer on the un-normalised stream, and its error tends to the drop-the-norm error,
not to 0. -/
theorem dominant_limit_not_normalised {d : ℕ} (hd : d ≠ 0) {eps : ℝ} (heps : 0 ≤ eps)
    (w x : Fin d → ℝ) (i : Fin d) (hwx : w i * x i ≠ 0) (c : ℝ) :
    ¬ (c * (w i * x i) = rmsNorm eps w x i ∧ c * (w i * (2 * x i)) = rmsNorm eps w ((2 : ℝ) • x) i) := by
  rintro ⟨h1, h2⟩
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  set S := ∑ l, x l ^ 2 with hSdef
  have hxi : x i ≠ 0 := by intro h; apply hwx; rw [h, mul_zero]
  have hS : 0 < S := lt_of_lt_of_le (by positivity) (Finset.single_le_sum
    (f := fun l => x l ^ 2) (fun l _ => sq_nonneg _) (Finset.mem_univ i))
  have hA : 0 < S / d + eps := by positivity
  have hB : 0 < 4 * S / d + eps := by positivity
  simp only [rmsNorm, Pi.smul_apply, smul_eq_mul] at h1 h2
  have h2' : c * (w i * (2 * x i)) = w i * (2 * x i) / √(4 * S / d + eps) := by
    rw [h2]; congr 2
    rw [show ∑ l, (2 * x l) ^ 2 = 4 * S by rw [hSdef, Finset.mul_sum]; congr 1; funext l; ring]
  -- from h1: c = 1/√A; from h2': c = 1/√B; so A = B, i.e. S = 0
  have hc1 : c = 1 / √(S / d + eps) := by
    have : c * (w i * x i) = (1 / √(S / d + eps)) * (w i * x i) := by rw [h1]; ring
    exact mul_right_cancel₀ hwx this
  have hc2 : c = 1 / √(4 * S / d + eps) := by
    have : c * (w i * (2 * x i)) = (1 / √(4 * S / d + eps)) * (w i * (2 * x i)) := by
      rw [h2']; ring
    exact mul_right_cancel₀ (by rw [show w i * (2 * x i) = 2 * (w i * x i) by ring]; exact mul_ne_zero two_ne_zero hwx) this
  rw [hc1] at hc2
  have hsq := congrArg (fun z => (1 / z) ^ 2) hc2
  simp only [one_div, inv_inv] at hsq
  rw [Real.sq_sqrt hA.le, Real.sq_sqrt hB.le] at hsq
  have : S / d = 4 * S / d := by linarith
  have : S = 0 := by field_simp at this; linarith
  linarith

end EP
