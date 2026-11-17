import EmulationProofs.Common

/-!
# Activations: SwiGLU = squared FFN, swigluoai ⊆ SiLU, ReLU ⊆ SiLU / GeLU, GLU → ungated,
# softcap and qk-norm regimes
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-- SiLU. -/
def silu (x : ℝ) : ℝ := x * sigmoid x

/-- ReLU. -/
def relu (x : ℝ) : ℝ := max x 0

/-! ## SwiGLU = squared-activation FFN -/

/-- The polarisation identity. -/
theorem polarization (p b : ℝ) : p * b = ((p + b) ^ 2 - (p - b) ^ 2) / 4 := by ring

/-- **SwiGLU = sq-FFN** (exact, width ×2 + depth). A gated layer `∑_u p_u b_u w_u` (with
`p_u = SiLU(a_u)` and `b_u` computed by a previous layer) equals an FFN with squared activation
and `2 |U|` units: unit `(u, +)` reads `p_u + b_u` with down column `w_u / 4`, unit `(u, -)`
reads `p_u - b_u` with down column `-w_u / 4`. -/
theorem swiglu_eq_sq_ffn {U V : Type*} [Fintype U] [AddCommGroup V] [Module ℝ V]
    (p b : U → ℝ) (w : U → V) :
    ∑ u, (p u * b u) • w u =
      ∑ ub : U × Bool, (p ub.1 + (if ub.2 then 1 else -1) * b ub.1) ^ 2 •
        ((if ub.2 then (1 / 4 : ℝ) else -1 / 4) • w ub.1) := by
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun u _ => ?_)
  simp only [Fintype.sum_bool, Bool.false_eq_true, ↓reduceIte, smul_smul, ← add_smul]
  congr 1
  ring

/-! ## swigluoai ⊆ SiLU -/

/-- gpt-oss's gated activation: `(clamp(up, -7, 7) + 1) · g σ(1.702 g)`, `g = min(gate, 7)`. -/
def swigluoai (gate up : ℝ) : ℝ :=
  (max (min up 7) (-7) + 1) * (min gate 7 * sigmoid (1.702 * min gate 7))

/-- The fold, for any `α ≠ 0`: `g σ(α g) = SiLU(α g) / α`. -/
theorem sigmoid_scale_fold (α g : ℝ) (hα : α ≠ 0) : g * sigmoid (α * g) = silu (α * g) / α := by
  simp only [silu]; field_simp

/-- **swigluoai ⊆ SiLU** (exact fold, in the clamp regime). While `gate ≤ 7` and
`|up| ≤ 7`, swigluoai is a SwiGLU unit with `1.702` folded into the gate weights, `+1` into the
up bias and `1/1.702` into the down weights. Outside the regime the clamps bite. -/
theorem swigluoai_sub_silu (gate up : ℝ) (hg : gate ≤ 7) (hu : |up| ≤ 7) :
    swigluoai gate up = (up + 1) * silu (1.702 * gate) / 1.702 := by
  have h1 : min gate 7 = gate := min_eq_left hg
  have h2 : max (min up 7) (-7) = up := by
    rw [min_eq_left (abs_le.mp hu).2, max_eq_left (abs_le.mp hu).1]
  rw [swigluoai, h1, h2, sigmoid_scale_fold 1.702 gate (by norm_num)]
  ring

/-! ## ReLU ⊆ SiLU (limit) -/

/-- **ReLU ⊆ SiLU, the rate** (uniform). `SiLU(βx)/β = x σ(βx)` (the `β` folds into the
neighbouring weights) misses `ReLU x` by at most `|x| e^{-β|x|} ≤ 1/β`, for every `x`. -/
theorem relu_silu_bound (β : ℝ) (hβ : 0 < β) (x : ℝ) :
    |silu (β * x) / β - relu x| ≤ 1 / β := by
  have hfold : silu (β * x) / β = x * sigmoid (β * x) := by
    simp only [silu]; field_simp
  rw [hfold]
  -- t e^{-β t} ≤ 1/β for t ≥ 0, since e^{β t} ≥ 1 + β t > β t
  have key : ∀ t : ℝ, 0 ≤ t → t * exp (-(β * t)) ≤ 1 / β := by
    intro t ht
    rw [exp_neg, ← div_eq_mul_inv, div_le_div_iff₀ (exp_pos _) hβ]
    nlinarith [add_one_le_exp (β * t)]
  rcases le_or_gt 0 x with hx | hx
  · rw [relu, max_eq_left hx]
    have hs : 1 - sigmoid (β * x) = sigmoid (-(β * x)) := (sigmoid_neg _).symm
    have : x * sigmoid (β * x) - x = -(x * sigmoid (-(β * x))) := by rw [← hs]; ring
    rw [this, abs_neg, abs_of_nonneg (mul_nonneg hx (sigmoid_nonneg _))]
    calc x * sigmoid (-(β * x)) ≤ x * exp (-(β * x)) :=
          mul_le_mul_of_nonneg_left (sigmoid_le_exp _) hx
      _ ≤ 1 / β := key x hx
  · rw [relu, max_eq_right hx.le, sub_zero, abs_mul, abs_of_neg hx,
      abs_of_pos (sigmoid_pos _)]
    calc -x * sigmoid (β * x) ≤ -x * exp (β * x) :=
          mul_le_mul_of_nonneg_left (sigmoid_le_exp _) (by linarith)
      _ = -x * exp (-(β * -x)) := by ring_nf
      _ ≤ 1 / β := key (-x) (by linarith)

/-- **ReLU ⊆ SiLU** (limit `β → ∞`, uniformly in `x`). -/
theorem relu_sub_silu (x : ℝ) :
    Tendsto (fun β => silu (β * x) / β) atTop (𝓝 (relu x)) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  refine squeeze_zero_norm' (a := fun β : ℝ => 1 / β) ?_
    ((tendsto_const_nhds (x := (1 : ℝ))).div_atTop tendsto_id)
  filter_upwards [eventually_gt_atTop 0] with β hβ
  simpa [Real.norm_eq_abs] using relu_silu_bound β hβ x

/-- GeLU with the exact Gaussian CDF `Φ`: `GeLU x = x Φ(x)`. -/
def gelu (x : ℝ) : ℝ := x * ProbabilityTheory.cdf (ProbabilityTheory.gaussianReal 0 1) x

/-- **ReLU ⊆ GeLU** (limit `β → ∞`; swing table, construction B). `GeLU(βx)/β = x Φ(βx) → ReLU x`.
Only the CDF's limits at `±∞` are used, so this holds for every CDF-gated unit. -/
theorem relu_sub_gelu (x : ℝ) :
    Tendsto (fun β => gelu (β * x) / β) atTop (𝓝 (relu x)) := by
  set Φ := ProbabilityTheory.cdf (ProbabilityTheory.gaussianReal 0 1)
  have hfold : ∀ᶠ β in atTop, gelu (β * x) / β = x * Φ (β * x) := by
    filter_upwards [eventually_gt_atTop 0] with β hβ
    simp only [gelu, Φ]; field_simp
  refine Tendsto.congr' (EventuallyEq.symm hfold) ?_
  rcases lt_trichotomy x 0 with hx | hx | hx
  · have : Tendsto (fun β : ℝ => Φ (β * x)) atTop (𝓝 0) :=
      (ProbabilityTheory.tendsto_cdf_atBot _).comp (tendsto_id.atTop_mul_const_of_neg hx)
    rw [relu, max_eq_right hx.le]
    simpa using this.const_mul x
  · subst hx; simp [relu]
  · have : Tendsto (fun β : ℝ => Φ (β * x)) atTop (𝓝 1) :=
      (ProbabilityTheory.tendsto_cdf_atTop _).comp (tendsto_id.atTop_mul_const hx)
    rw [relu, max_eq_left hx.le]
    simpa using this.const_mul x

/-! ## GLU → ungated -/

/-- **GLU → ungated** (exact): with the up branch held at `1` (zero weights, bias `1`) a gated
unit is the plain activation. -/
theorem glu_ungated (σ : ℝ → ℝ) (a : ℝ) : σ a * (0 * a + 1) = σ a := by ring

/-! ## softcap ⊆ plain (regime) -/

lemma hasDerivAt_tanh (x : ℝ) : HasDerivAt tanh (1 - tanh x ^ 2) x := by
  have hc := (cosh_pos x).ne'
  have h := (hasDerivAt_sinh x).div (hasDerivAt_cosh x) hc
  have e : sinh / cosh = tanh := funext fun y => (tanh_eq_sinh_div_cosh y).symm
  rw [e] at h
  convert h using 1
  rw [tanh_eq_sinh_div_cosh]
  field_simp

lemma tanh_le_self {u : ℝ} (hu : 0 ≤ u) : tanh u ≤ u := by
  have hmono : Monotone (fun y => y - tanh y) := by
    apply monotone_of_deriv_nonneg
    · intro y; exact ((hasDerivAt_id' y).sub (hasDerivAt_tanh y)).differentiableAt
    · intro y
      have hd : HasDerivAt (fun y => y - tanh y) (1 - (1 - tanh y ^ 2)) y :=
        (hasDerivAt_id' y).sub (hasDerivAt_tanh y)
      rw [hd.deriv]
      nlinarith [sq_nonneg (tanh y)]
  have := hmono hu
  simp at this
  linarith

lemma tanh_nonneg {u : ℝ} (hu : 0 ≤ u) : 0 ≤ tanh u := by
  rw [tanh_eq_sinh_div_cosh]
  exact div_nonneg (sinh_nonneg_iff.mpr hu) (cosh_pos u).le

lemma self_sub_cube_le_tanh {u : ℝ} (hu : 0 ≤ u) : u - u ^ 3 / 3 ≤ tanh u := by
  have hmono : MonotoneOn (fun y => tanh y - y + y ^ 3 / 3) (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
    · have hd : Differentiable ℝ tanh := fun y => (hasDerivAt_tanh y).differentiableAt
      exact ((hd.continuous.sub continuous_id).add
        ((continuous_pow 3).div_const 3)).continuousOn
    · intro y _
      exact (((hasDerivAt_tanh y).sub (hasDerivAt_id' y)).add
        ((hasDerivAt_pow 3 y).div_const 3)).differentiableAt.differentiableWithinAt
    · intro y hy
      rw [interior_Ici] at hy
      have hy0 : 0 ≤ y := le_of_lt hy
      have hd : HasDerivAt (fun y => tanh y - y + y ^ 3 / 3)
          (1 - tanh y ^ 2 - 1 + (3 : ℕ) * y ^ (3 - 1) / 3) y :=
        ((hasDerivAt_tanh y).sub (hasDerivAt_id' y)).add ((hasDerivAt_pow 3 y).div_const 3)
      rw [hd.deriv]
      have h1 := tanh_le_self hy0
      have h2 := tanh_nonneg hy0
      push_cast
      nlinarith
  have := hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hu) hu
  simp at this
  linarith

/-- Softcap: `c tanh(z / c)`. -/
def softcap (c z : ℝ) : ℝ := c * tanh (z / c)

/-- **softcap ⊆ plain, the regime bound.** `|softcap_c(z) − z| ≤ |z|³ / (3c²)` for every `z`
(`c > 0`). So a plain host's logit `z` is within `R³/(3c²)` of the capped one while `|z| ≤ R`. -/
theorem softcap_bound (c : ℝ) (hc : 0 < c) (z : ℝ) : |softcap c z - z| ≤ |z| ^ 3 / (3 * c ^ 2) := by
  have main : ∀ z, 0 ≤ z → |softcap c z - z| ≤ z ^ 3 / (3 * c ^ 2) := by
    intro z hz
    have hu : 0 ≤ z / c := div_nonneg hz hc.le
    have h1 := tanh_le_self hu
    have h2 := self_sub_cube_le_tanh hu
    have e1 : softcap c z - z = c * (tanh (z / c) - z / c) := by
      simp only [softcap]; field_simp
    rw [e1, abs_mul, abs_of_pos hc, abs_of_nonpos (by linarith)]
    have : c * (-(tanh (z / c) - z / c)) ≤ c * ((z / c) ^ 3 / 3) := by
      apply mul_le_mul_of_nonneg_left _ hc.le; linarith
    calc c * (-(tanh (z / c) - z / c)) ≤ c * ((z / c) ^ 3 / 3) := this
      _ = z ^ 3 / (3 * c ^ 2) := by field_simp
  rcases le_or_gt 0 z with hz | hz
  · rw [abs_of_nonneg hz]; exact main z hz
  · have := main (-z) (by linarith)
    have e : softcap c (-z) - -z = -(softcap c z - z) := by
      simp [softcap, neg_div, tanh_neg]; ring
    rw [e, abs_neg] at this
    rw [abs_of_neg hz]
    simpa [neg_pow, Odd.neg_one_pow (by decide : Odd 3)] using this

/-- **softcap ⊆ plain at the attention weights.** If every logit satisfies `|z| ≤ R`, the plain
host's softmax weights are within a factor `exp (2 R³/(3c²))` of the capped model's. -/
theorem softcap_attn_regime {ι : Type*} [Fintype ι] [Nonempty ι] (c R : ℝ) (hc : 0 < c)
    (z : ι → ℝ) (hz : ∀ j, |z j| ≤ R) (j : ι) :
    softmax z j ≤ exp (2 * (R ^ 3 / (3 * c ^ 2))) * softmax (fun l => softcap c (z l)) j ∧
    softmax (fun l => softcap c (z l)) j ≤ exp (2 * (R ^ 3 / (3 * c ^ 2))) * softmax z j := by
  have hb : ∀ l, |softcap c (z l) - z l| ≤ R ^ 3 / (3 * c ^ 2) := by
    intro l
    refine (softcap_bound c hc (z l)).trans ?_
    gcongr
    exact hz l
  constructor
  · exact softmax_ratio_bound _ _ _ (fun l => by rw [abs_sub_comm]; exact hb l) j
  · exact softmax_ratio_bound _ _ _ hb j

/-! ## qk-norm ⊆ plain (regime) -/

/-- The RMS (with epsilon) of a vector. -/
def rms {n : ℕ} (eps : ℝ) (x : Fin n → ℝ) : ℝ := √((∑ l, x l ^ 2) / n + eps)

/-- **qk-norm ⊆ plain, the identity.** The qk-norm logit is the plain logit of the weight-scaled
`q, k` with FIXED constants `r̄_q, r̄_k` folded into `W_q` (all a plain host can do), times the
token-dependent factor `r̄_q r̄_k / (rms q · rms k)`. -/
theorem qknorm_logit {n : ℕ} (eps : ℝ) (wq wk q k : Fin n → ℝ) (rq rk : ℝ) (hq : rq ≠ 0)
    (hk : rk ≠ 0) (hq' : rms eps q ≠ 0) (hk' : rms eps k ≠ 0) :
    rmsNorm eps wq q ⬝ᵥ rmsNorm eps wk k =
      (rq * rk) / (rms eps q * rms eps k) *
        ((fun i => wq i * q i / rq) ⬝ᵥ (fun i => wk i * k i / rk)) := by
  simp only [rmsNorm, dotProduct, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  simp only [rms] at hq' hk'
  simp only [rms]
  field_simp

/-- **qk-norm ⊆ plain** (exact in-regime). On every token pair whose `q, k` RMS equal the folded
constants, the plain host's logit equals the qk-norm logit. -/
theorem qknorm_sub_plain_regime {n : ℕ} (eps : ℝ) (wq wk q k : Fin n → ℝ) (rq rk : ℝ)
    (hq : rms eps q = rq) (hk : rms eps k = rk) (hq0 : rq ≠ 0) (hk0 : rk ≠ 0) :
    rmsNorm eps wq q ⬝ᵥ rmsNorm eps wk k =
      (fun i => wq i * q i / rq) ⬝ᵥ (fun i => wk i * k i / rk) := by
  rw [qknorm_logit eps wq wk q k rq rk hq0 hk0 (hq ▸ hq0) (hk ▸ hk0), hq, hk,
    div_self (mul_ne_zero hq0 hk0), one_mul]

end EP
