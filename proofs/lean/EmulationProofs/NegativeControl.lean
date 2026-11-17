import Mathlib

/-!
# The negative control's variants are false

`NegativeControl/FalseVariants.lean` (outside the library) states two wrong variants of proved
identities and must FAIL to check. Here we prove that they are in fact false.
-/

open Real

namespace EP

/-- The polarisation identity with `/2` is false (`p = b = 1`: `1 ≠ 2`). -/
theorem polarization_half_false : ¬ ∀ p b : ℝ, p * b = ((p + b) ^ 2 - (p - b) ^ 2) / 2 := by
  intro h
  have := h 1 1
  norm_num at this

/-- The partial-RoPE base with the inverted exponent is false (`θ_t = e`, `rot_h = 2`, `d_t = 1`,
pair `i = 1`: `e^(-1/2) ≠ e^(-2)`). -/
theorem partial_rope_wrong_base_false :
    ¬ ∀ θt r d i : ℝ, 0 < θt → r ≠ 0 → d ≠ 0 →
      (θt ^ (d / r)) ^ (-2 * i / r) = θt ^ (-2 * i / d) := by
  intro h
  have := h (exp 1) 2 1 1 (exp_pos 1) (by norm_num) (by norm_num)
  rw [← Real.rpow_mul (exp_pos 1).le, Real.exp_one_rpow, Real.exp_one_rpow] at this
  have := Real.exp_injective this
  norm_num at this

end EP
