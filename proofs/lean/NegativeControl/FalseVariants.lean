import EmulationProofs

/-!
# Negative control: false variants of two cookbook identities

Each theorem below is a deliberately wrong version of a proved identity, given the SAME proof
script as the true one. The checker must reject this file (`check.sh` asserts a nonzero exit and
an error at each theorem). The library proves the negations (`EP.polarization_half_false`,
`EP.partial_rope_wrong_base_false`), so these are not merely unproved but false.
-/

open Real

/-- WRONG: the polarisation identity with `/2` instead of `/4` (SwiGLU = sq-FFN). -/
theorem polarization_half (p b : ℝ) : p * b = ((p + b) ^ 2 - (p - b) ^ 2) / 2 := by ring

/-- WRONG: the partial-RoPE base with the exponent inverted, `θ_t^(d_t / rot_h)`. -/
theorem partial_rope_wrong_base (θt r d i : ℝ) (hθ : 0 < θt) (hr : r ≠ 0) (hd : d ≠ 0) :
    (θt ^ (d / r)) ^ (-2 * i / r) = θt ^ (-2 * i / d) := by
  rw [← Real.rpow_mul hθ.le]
  congr 1
  field_simp
