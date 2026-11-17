import EmulationProofs.Activation

/-!
# Activation transitions: relu² ⊆ xIELU, ReLU ⊆ mish, ReLU ⊆ swigluoai

* **relu² ⊆ xIELU** (limit, with a rate; never exact): `xIELU(c x) / (α_p c²) → relu(x)²`,
  within `(α_n + (|β| + α_n)|x|) / (α_p c)` for `c ≥ 1` (`relu2_xielu_bound`,
  `relu2_sub_xielu`); with `α_n > 0` (HF: `α_n = β + softplus(·) > 0`) no scaling is exact
  (`xielu_ne_relu2`).
* **ReLU ⊆ mish** (limit, uniform rate `1/β`): `mish(β x)/β → relu x` (`relu_mish_bound`,
  `relu_sub_mish`).
* **ReLU ⊆ swigluoai**: the route the relation names (scale the gate, as for SiLU) FAILS: the
  gate clamp at 7 makes `swigluoai(β z, 0)/β → 0` for `z > 0` (`swigluoai_scaled_gate_vanishes`).
  A different construction is a limit on bounded inputs: the gate saturates to a step
  (`c → ∞`, clamp value `K = 7 σ(11.914)`), and the up branch with bias `−1` carries `a z`
  (`relu_sub_swigluoai`, needs `|a z| ≤ 6`, an up-projection bias, which gpt-oss has).
-/

open Real Filter Topology

noncomputable section

namespace EP

/-! ## xIELU -/

/-- xIELU (HF `XIELUActivation`): `α_p x² + β x` for `x > 0`,
`α_n (expm1(min(x, ε)) − x) + β x` otherwise. -/
def xielu (αp αn β ε x : ℝ) : ℝ :=
  if 0 < x then αp * x ^ 2 + β * x else αn * (exp (min x ε) - 1 - x) + β * x

/-- **relu² ⊆ xIELU, the rate** (bound). For `α_p > 0`, `α_n ≥ 0`, `ε ≤ 0` and `c ≥ 1`,
`|xIELU(c x)/(α_p c²) − relu(x)²| ≤ (α_n + (|β| + α_n)|x|) / (α_p c)`. -/
theorem relu2_xielu_bound (αp αn β ε : ℝ) (hp : 0 < αp) (hn : 0 ≤ αn) (hε : ε ≤ 0) (c : ℝ)
    (hc : 1 ≤ c) (x : ℝ) :
    |xielu αp αn β ε (c * x) / (αp * c ^ 2) - relu x ^ 2| ≤
      (αn + (|β| + αn) * |x|) / (αp * c) := by
  have hc0 : 0 < c := by linarith
  have hD : 0 < αp * c ^ 2 := by positivity
  rcases lt_or_ge 0 x with hx | hx
  · have hcx : 0 < c * x := mul_pos hc0 hx
    rw [xielu, relu, max_eq_left hx.le]; simp only [hcx, ↓reduceIte]
    have : (αp * (c * x) ^ 2 + β * (c * x)) / (αp * c ^ 2) - x ^ 2 = β * x / (αp * c) := by
      field_simp; ring
    rw [this, abs_div, abs_mul, abs_of_pos hx, abs_of_pos (by positivity : 0 < αp * c)]
    gcongr
    nlinarith [abs_nonneg β]
  · have hcx : ¬ 0 < c * x := by nlinarith
    rw [xielu, relu, max_eq_right hx]; simp only [hcx, ↓reduceIte]
    norm_num
    -- |expm1 (min (c x) ε)| ≤ 1
    have hm : min (c * x) ε ≤ 0 := (min_le_right _ _).trans hε
    have he1 : |exp (min (c * x) ε) - 1| ≤ 1 := by
      rw [abs_le]
      constructor
      · linarith [exp_pos (min (c * x) ε)]
      · linarith [exp_le_one_iff.mpr hm]
    have hnum : |αn * (exp (min (c * x) ε) - 1 - c * x) + β * (c * x)| ≤
        αn + (|β| + αn) * (c * |x|) := by
      have : αn * (exp (min (c * x) ε) - 1 - c * x) + β * (c * x) =
          αn * (exp (min (c * x) ε) - 1) + (β - αn) * (c * x) := by ring
      rw [this]
      calc |αn * (exp (min (c * x) ε) - 1) + (β - αn) * (c * x)|
          ≤ |αn * (exp (min (c * x) ε) - 1)| + |(β - αn) * (c * x)| := abs_add_le _ _
        _ = αn * |exp (min (c * x) ε) - 1| + |β - αn| * (c * |x|) := by
          rw [abs_mul, abs_mul, abs_mul, abs_of_nonneg hn, abs_of_pos hc0]
        _ ≤ αn * 1 + (|β| + αn) * (c * |x|) := by
          gcongr
          calc |β - αn| ≤ |β| + |αn| := abs_sub _ _
            _ = |β| + αn := by rw [abs_of_nonneg hn]
        _ = αn + (|β| + αn) * (c * |x|) := by ring
    rw [abs_div, abs_of_pos hD]
    calc |αn * (exp (min (c * x) ε) - 1 - c * x) + β * (c * x)| / (αp * c ^ 2)
        ≤ (αn + (|β| + αn) * (c * |x|)) / (αp * c ^ 2) := by gcongr
      _ = (αn / c + (|β| + αn) * |x|) / (αp * c) := by field_simp
      _ ≤ (αn + (|β| + αn) * |x|) / (αp * c) := by
          gcongr
          exact div_le_self hn hc

/-- **relu² ⊆ xIELU** (limit `c → ∞`, uniformly on bounded `x`). -/
theorem relu2_sub_xielu (αp αn β ε : ℝ) (hp : 0 < αp) (hn : 0 ≤ αn) (hε : ε ≤ 0) (x : ℝ) :
    Tendsto (fun c => xielu αp αn β ε (c * x) / (αp * c ^ 2)) atTop (𝓝 (relu x ^ 2)) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  have hlim : Tendsto (fun c : ℝ => (αn + (|β| + αn) * |x|) / αp / c) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_id
  refine squeeze_zero_norm' ?_ hlim
  filter_upwards [eventually_ge_atTop 1] with c hc
  simp only [Real.norm_eq_abs, abs_abs]
  rw [div_div]
  exact relu2_xielu_bound αp αn β ε hp hn hε c hc x

/-- **relu² ⊆ xIELU is never exact** (impossibility). With `α_n > 0` no input scale `c > 0` and
output scale `a ≠ 0` make `xIELU(c x)/a` equal `relu(x)² = 0` on the negative half-line: there
it is `α_n (eʸ − 1 − y) + β y` (`y = c x`), and `exp` is strictly convex. -/
theorem xielu_ne_relu2 (αp αn β ε : ℝ) (hn : 0 < αn) (c a : ℝ) (hc : 0 < c) (ha : a ≠ 0) :
    ¬ ∀ x < 0, xielu αp αn β ε (c * x) / a = relu x ^ 2 := by
  intro h
  set y0 := min 0 ε
  have hy : ∀ y < y0, αn * (exp y - 1 - y) + β * y = 0 := by
    intro y hy
    have hy0 : y < 0 := lt_of_lt_of_le hy (min_le_left _ _)
    have hyε : y ≤ ε := (lt_of_lt_of_le hy (min_le_right _ _)).le
    have hx : y / c < 0 := div_neg_of_neg_of_pos hy0 hc
    have := h (y / c) hx
    have hny : ¬ 0 < y := not_lt.mpr hy0.le
    rw [relu, max_eq_right hx.le, mul_div_cancel₀ y hc.ne', xielu, min_eq_left hyε] at this
    simp only [hny, ↓reduceIte] at this
    norm_num at this
    rcases this with h1 | h1
    · exact h1
    · exact absurd h1 ha
  have h1 := hy (y0 - 3) (by linarith)
  have h2 := hy (y0 - 2) (by linarith)
  have h3 := hy (y0 - 1) (by linarith)
  have hconv := strictConvexOn_exp.2 (Set.mem_univ (y0 - 3)) (Set.mem_univ (y0 - 1))
    (by linarith : y0 - 3 ≠ y0 - 1) (by norm_num : (0 : ℝ) < 1 / 2) (by norm_num : (0 : ℝ) < 1 / 2)
    (by norm_num)
  simp only [smul_eq_mul] at hconv
  have harg : 1 / 2 * (y0 - 3) + 1 / 2 * (y0 - 1) = y0 - 2 := by ring
  rw [harg] at hconv
  nlinarith

/-! ## mish -/

/-- softplus `log(1 + eˣ)`. -/
def softplus (x : ℝ) : ℝ := log (1 + exp x)

/-- mish `x tanh(softplus x)`. -/
def mish (x : ℝ) : ℝ := x * tanh (softplus x)

lemma softplus_ge (x : ℝ) : x ≤ softplus x := by
  rw [softplus, ← exp_le_exp, exp_log (by positivity)]
  linarith

lemma softplus_pos (x : ℝ) : 0 < softplus x := by
  rw [softplus]; apply log_pos; linarith [exp_pos x]

lemma softplus_le_exp (x : ℝ) : softplus x ≤ exp x := by
  rw [softplus]
  have := log_le_sub_one_of_pos (by positivity : (0 : ℝ) < 1 + exp x)
  linarith

lemma tanh_lt_one' (y : ℝ) : tanh y < 1 := by
  rw [tanh_eq_sinh_div_cosh, div_lt_one (cosh_pos y), sinh_eq, cosh_eq]
  linarith [exp_pos (-y)]

lemma one_sub_tanh_le (y : ℝ) : 1 - tanh y ≤ 2 * exp (-(2 * y)) := by
  rw [tanh_eq_sinh_div_cosh, sinh_eq, cosh_eq]
  have h1 := exp_pos y
  have h2 := exp_pos (-y)
  have h3 : exp (-(2 * y)) = exp (-y) * exp (-y) := by rw [← exp_add]; ring_nf
  have h4 : exp y * exp (-y) = 1 := by rw [← exp_add]; simp
  rw [h3, one_sub_div (by positivity)]
  rw [div_le_iff₀ (by positivity)]
  nlinarith [mul_pos h2 h2, mul_pos (mul_pos h2 h2) h2]

/-- `t e^{-βt} ≤ 1/β` for `t ≥ 0`. -/
lemma mul_exp_neg_le (β t : ℝ) (hβ : 0 < β) (_ht : 0 ≤ t) : t * exp (-(β * t)) ≤ 1 / β := by
  rw [exp_neg, ← div_eq_mul_inv, div_le_div_iff₀ (exp_pos _) hβ]
  nlinarith [add_one_le_exp (β * t)]

/-- **ReLU ⊆ mish, the rate** (uniform). `mish(βx)/β = x tanh(softplus(βx))` misses `relu x`
by at most `1/β` for every `x`. -/
theorem relu_mish_bound (β : ℝ) (hβ : 0 < β) (x : ℝ) : |mish (β * x) / β - relu x| ≤ 1 / β := by
  have hfold : mish (β * x) / β = x * tanh (softplus (β * x)) := by
    simp only [mish]; field_simp
  rw [hfold]
  rcases le_or_gt 0 x with hx | hx
  · rw [relu, max_eq_left hx]
    have ht1 := tanh_lt_one' (softplus (β * x))
    have hb := one_sub_tanh_le (softplus (β * x))
    have hsp := softplus_ge (β * x)
    rw [abs_of_nonpos (by nlinarith)]
    calc -(x * tanh (softplus (β * x)) - x) = x * (1 - tanh (softplus (β * x))) := by ring
      _ ≤ x * (2 * exp (-(2 * (β * x)))) := by
          apply mul_le_mul_of_nonneg_left _ hx
          calc 1 - tanh (softplus (β * x)) ≤ 2 * exp (-(2 * softplus (β * x))) := hb
            _ ≤ 2 * exp (-(2 * (β * x))) := by gcongr
      _ = (2 * x) * exp (-(β * (2 * x))) := by ring_nf
      _ ≤ 1 / β := mul_exp_neg_le β (2 * x) hβ (by linarith)
  · rw [relu, max_eq_right hx.le, sub_zero]
    have hsp := softplus_pos (β * x)
    have h0 := tanh_nonneg hsp.le
    have h1 := tanh_le_self hsp.le
    have h2 := softplus_le_exp (β * x)
    rw [abs_mul, abs_of_neg hx, abs_of_nonneg h0]
    calc -x * tanh (softplus (β * x)) ≤ -x * exp (β * x) := by
          apply mul_le_mul_of_nonneg_left (h1.trans h2) (by linarith)
      _ = (-x) * exp (-(β * (-x))) := by ring_nf
      _ ≤ 1 / β := mul_exp_neg_le β (-x) hβ (by linarith)

/-- **ReLU ⊆ mish** (limit `β → ∞`, uniformly in `x`). -/
theorem relu_sub_mish (x : ℝ) : Tendsto (fun β => mish (β * x) / β) atTop (𝓝 (relu x)) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  refine squeeze_zero_norm' (a := fun β : ℝ => 1 / β) ?_
    ((tendsto_const_nhds (x := (1 : ℝ))).div_atTop tendsto_id)
  filter_upwards [eventually_gt_atTop 0] with β hβ
  simpa [Real.norm_eq_abs] using relu_mish_bound β hβ x

/-! ## swigluoai -/

/-- The clamp value of the gate branch: `K = 7 σ(1.702 · 7)`. -/
def swK : ℝ := 7 * sigmoid (1.702 * 7)

lemma swK_pos : 0 < swK := by unfold swK; have := sigmoid_pos (1.702 * 7); positivity

/-- **The naive route fails** (refutation). Scaling the gate (`relu_sub_silu`'s `β → ∞`) inside
swigluoai's clamp gives `swigluoai(βz, 0)/β = K/β → 0` for every `z > 0`, not `relu z = z`. -/
theorem swigluoai_scaled_gate_vanishes (z : ℝ) (hz : 0 < z) :
    Tendsto (fun β => swigluoai (β * z) 0 / β) atTop (𝓝 0) := by
  have hev : ∀ᶠ β in atTop, swigluoai (β * z) 0 / β = swK / β := by
    filter_upwards [eventually_ge_atTop (7 / z)] with β hβ
    have : 7 ≤ β * z := by rwa [div_le_iff₀ hz] at hβ
    rw [swigluoai, min_eq_right this]
    norm_num [swK]
  exact Tendsto.congr' (EventuallyEq.symm hev) (tendsto_const_nhds.div_atTop tendsto_id)

/-- **ReLU ⊆ swigluoai** (limit `c → ∞`, bounded inputs, up bias). Gate `c z`, up `a z − 1`
(bias `−1`), down scale `1/(a K)`. For `|a z| ≤ 6` the up clamp is inactive and `up + 1 = a z`;
the gate branch is exactly `K` once `c z ≥ 7` (`z > 0`), `0` at `z = 0`, and
`c z σ(1.702 c z) → 0` for `z < 0`. So the unit tends to `relu z` (eventually EXACT for `z ≥ 0`). -/
theorem relu_sub_swigluoai (a z : ℝ) (ha : 0 < a) (hz : |a * z| ≤ 6) :
    Tendsto (fun c => swigluoai (c * z) (a * z - 1) / (a * swK)) atTop (𝓝 (relu z)) := by
  have hup : max (min (a * z - 1) 7) (-7) + 1 = a * z := by
    have := abs_le.mp hz
    rw [min_eq_left (by linarith), max_eq_left (by linarith)]; ring
  have hK := swK_pos
  rcases lt_trichotomy z 0 with hz0 | hz0 | hz0
  · rw [relu, max_eq_right hz0.le]
    have hev : ∀ᶠ c in atTop, swigluoai (c * z) (a * z - 1) / (a * swK) =
        (z / swK) * ((c * z) * sigmoid (1.702 * (c * z))) := by
      filter_upwards [eventually_gt_atTop 0] with c hc
      have : c * z ≤ 7 := by nlinarith
      rw [swigluoai, hup, min_eq_left this]
      field_simp
    refine Tendsto.congr' (EventuallyEq.symm hev) ?_
    rw [show (0 : ℝ) = z / swK * 0 by ring]
    refine Tendsto.const_mul _ ?_
    -- |g σ(1.702 g)| ≤ |g| e^{1.702 g} → 0 as g = c z → -∞
    have hT : Tendsto (fun c : ℝ => (c * -z) * exp (-(c * -z))) atTop (𝓝 0) := by
      have := (Real.tendsto_pow_mul_exp_neg_atTop_nhds_zero 1).comp
        (tendsto_id.atTop_mul_const (by linarith : 0 < -z))
      refine this.congr (fun c => ?_)
      simp [Function.comp]
    refine squeeze_zero_norm' ?_ hT
    filter_upwards [eventually_gt_atTop 0] with c hc
    have hg : c * z < 0 := by nlinarith
    rw [Real.norm_eq_abs, abs_mul, abs_of_neg hg, abs_of_pos (sigmoid_pos _)]
    have hs : sigmoid (1.702 * (c * z)) ≤ exp (c * z) := by
      calc sigmoid (1.702 * (c * z)) ≤ exp (1.702 * (c * z)) := sigmoid_le_exp _
        _ ≤ exp (c * z) := by apply exp_le_exp.mpr; nlinarith
    calc -(c * z) * sigmoid (1.702 * (c * z)) ≤ -(c * z) * exp (c * z) :=
          mul_le_mul_of_nonneg_left hs (by linarith)
      _ = c * -z * exp (-(c * -z)) := by ring_nf
  · subst hz0
    simp [swigluoai, relu]
  · rw [relu, max_eq_left hz0.le]
    refine tendsto_const_nhds.congr' ?_
    filter_upwards [eventually_ge_atTop (7 / z)] with c hc
    have : 7 ≤ c * z := by rwa [div_le_iff₀ hz0] at hc
    rw [swigluoai, hup, min_eq_right this]
    simp only [swK]
    field_simp
    exact (div_self (sigmoid_pos _).ne').symm

end EP
