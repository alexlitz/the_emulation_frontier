import EmulationProofs.Activation

/-!
# Activation swaps paid in FFN width: the shift-free odd-part obstruction

A host with FFN width `k ×` the target's can rebuild one target unit `Y(a)·b` from `k` host
units `∑ᵢ αᵢ φ(βᵢ a + γᵢ)·b` (gate rows scaled and shifted, down columns scaled). The shifts
`γᵢ` need a gate bias; a host without one (no constant channel survives an RMSNorm,
`prenorm_readout_obstruction`) computes `F(a) = ∑ᵢ αᵢ φ(βᵢ a)`.

For a swish host, `φ(t) − φ(−t) = t` (SiLU: `silu_identity_pair`; GeLU and swigluoai too), so
the odd part of every shift-free fit is LINEAR (`shiftfree_odd_part`). Hence a target whose
odd part is not linear cannot be fitted below a fixed error by any number of shift-free units
(`shiftfree_odd_obstruction`): if `|F − Y| ≤ ε` at `±x₀, ±x₁` then
`|x₁ (Y x₀ − Y (−x₀)) − x₀ (Y x₁ − Y (−x₁))| ≤ 2 (x₀ + x₁) ε`. This is why mish, situ,
squared ReLU and xIELU stay walls for bias-free swish hosts at every width
(model_subsets graph/act_width.py; numerically mish's odd part is 0.10 from a line on
[−160, 160]), while SiLU / GeLU / swigluoai targets (odd part exactly `a/2`) are reachable.
-/

open Finset

noncomputable section

namespace EP

/-- A shift-free k-unit fit: `∑ᵢ αᵢ φ(βᵢ a)`. -/
def shiftFreeFit {ι : Type*} (s : Finset ι) (φ : ℝ → ℝ) (α β : ι → ℝ) (a : ℝ) : ℝ :=
  ∑ i ∈ s, α i * φ (β i * a)

/-- **The odd part of a shift-free swish fit is linear**: with `φ t − φ (−t) = t`,
`F a − F (−a) = (∑ᵢ αᵢ βᵢ) a`. -/
theorem shiftfree_odd_part {ι : Type*} (s : Finset ι) (φ : ℝ → ℝ)
    (hφ : ∀ t, φ t - φ (-t) = t) (α β : ι → ℝ) (a : ℝ) :
    shiftFreeFit s φ α β a - shiftFreeFit s φ α β (-a) = (∑ i ∈ s, α i * β i) * a := by
  unfold shiftFreeFit
  rw [← sum_sub_distrib, sum_mul]
  refine sum_congr rfl fun i _ => ?_
  have h := hφ (β i * a)
  rw [show β i * -a = -(β i * a) by ring, ← mul_sub, h]
  ring

/-- SiLU satisfies the identity-pair hypothesis. -/
theorem silu_odd_pair (t : ℝ) : silu t - silu (-t) = t := by
  simp only [silu, Real.sigmoid_neg]; ring

/-- **Shift-free obstruction.** Any function `F` with linear odd part `F a − F (−a) = c a` (every
shift-free swish fit, `shiftfree_odd_part`) that is within `ε` of a target `Y` at `±x₀, ±x₁`
forces `|x₁ (Y x₀ − Y (−x₀)) − x₀ (Y x₁ − Y (−x₁))| ≤ 2 (x₀ + x₁) ε`. -/
theorem odd_obstruction (F Y : ℝ → ℝ) (c ε x₀ x₁ : ℝ) (h₀ : 0 ≤ x₀) (h₁ : 0 ≤ x₁)
    (hF : ∀ a, F a - F (-a) = c * a)
    (e₀ : |F x₀ - Y x₀| ≤ ε) (e₀' : |F (-x₀) - Y (-x₀)| ≤ ε)
    (e₁ : |F x₁ - Y x₁| ≤ ε) (e₁' : |F (-x₁) - Y (-x₁)| ≤ ε) :
    |x₁ * (Y x₀ - Y (-x₀)) - x₀ * (Y x₁ - Y (-x₁))| ≤ 2 * (x₀ + x₁) * ε := by
  have k₀ := hF x₀
  have k₁ := hF x₁
  -- x₁ (Y x₀ − Y(−x₀)) − x₀ (Y x₁ − Y(−x₁))
  --   = x₁ [(Y x₀ − F x₀) − (Y(−x₀) − F(−x₀))] − x₀ [(Y x₁ − F x₁) − (Y(−x₁) − F(−x₁))]
  have key : x₁ * (Y x₀ - Y (-x₀)) - x₀ * (Y x₁ - Y (-x₁))
      = x₁ * ((Y x₀ - F x₀) - (Y (-x₀) - F (-x₀))) - x₀ * ((Y x₁ - F x₁) - (Y (-x₁) - F (-x₁))) := by
    have : x₁ * (F x₀ - F (-x₀)) = x₀ * (F x₁ - F (-x₁)) := by rw [k₀, k₁]; ring
    linarith
  rw [key]
  have a₀ : |(Y x₀ - F x₀) - (Y (-x₀) - F (-x₀))| ≤ 2 * ε := by
    have := abs_sub (Y x₀ - F x₀) (Y (-x₀) - F (-x₀))
    rw [abs_sub_comm (Y x₀) (F x₀), abs_sub_comm (Y (-x₀)) (F (-x₀))] at this
    linarith
  have a₁ : |(Y x₁ - F x₁) - (Y (-x₁) - F (-x₁))| ≤ 2 * ε := by
    have := abs_sub (Y x₁ - F x₁) (Y (-x₁) - F (-x₁))
    rw [abs_sub_comm (Y x₁) (F x₁), abs_sub_comm (Y (-x₁)) (F (-x₁))] at this
    linarith
  calc |x₁ * ((Y x₀ - F x₀) - (Y (-x₀) - F (-x₀))) - x₀ * ((Y x₁ - F x₁) - (Y (-x₁) - F (-x₁)))|
      ≤ |x₁ * ((Y x₀ - F x₀) - (Y (-x₀) - F (-x₀)))| + |x₀ * ((Y x₁ - F x₁) - (Y (-x₁) - F (-x₁)))| :=
        abs_sub _ _
    _ = x₁ * |(Y x₀ - F x₀) - (Y (-x₀) - F (-x₀))| + x₀ * |(Y x₁ - F x₁) - (Y (-x₁) - F (-x₁))| := by
        rw [abs_mul, abs_mul, abs_of_nonneg h₀, abs_of_nonneg h₁]
    _ ≤ x₁ * (2 * ε) + x₀ * (2 * ε) := by gcongr
    _ = 2 * (x₀ + x₁) * ε := by ring

/-- **The shift-free swish obstruction, assembled**: every bias-free k-unit SiLU fit
`∑ᵢ αᵢ silu(βᵢ a)` (any k, any weights) within `ε` of `Y` at `±x₀, ±x₁` has
`|x₁ (Y x₀ − Y (−x₀)) − x₀ (Y x₁ − Y (−x₁))| ≤ 2 (x₀ + x₁) ε`. -/
theorem shiftfree_silu_obstruction {ι : Type*} (s : Finset ι) (α β : ι → ℝ) (Y : ℝ → ℝ)
    (ε x₀ x₁ : ℝ) (h₀ : 0 ≤ x₀) (h₁ : 0 ≤ x₁)
    (e : ∀ a ∈ ({x₀, -x₀, x₁, -x₁} : Set ℝ), |shiftFreeFit s silu α β a - Y a| ≤ ε) :
    |x₁ * (Y x₀ - Y (-x₀)) - x₀ * (Y x₁ - Y (-x₁))| ≤ 2 * (x₀ + x₁) * ε :=
  odd_obstruction (shiftFreeFit s silu α β) Y _ ε x₀ x₁ h₀ h₁
    (shiftfree_odd_part s silu silu_odd_pair α β)
    (e _ (by simp)) (e _ (by simp)) (e _ (by simp)) (e _ (by simp))

end EP
