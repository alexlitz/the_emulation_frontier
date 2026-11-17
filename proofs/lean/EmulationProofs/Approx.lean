import EmulationProofs.Activation

/-!
# ReLU-net approximations: `φ ⊆ ReLU-net` (width, `~1/N²`) and Yarotsky's `x²` (depth, `4^{-L}`)
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## Linear interpolation error -/

section interp

variable {f : ℝ → ℝ} (hd : Differentiable ℝ f) (hd2 : Differentiable ℝ (deriv f))

include hd hd2 in
/-- `f` plus a quadratic has second derivative `f'' + 2α`. -/
lemma deriv2_add_quad (α β γ : ℝ) :
    Differentiable ℝ (fun t => f t + α * t ^ 2 + β * t + γ) ∧
    Differentiable ℝ (deriv (fun t => f t + α * t ^ 2 + β * t + γ)) ∧
    ∀ t, deriv^[2] (fun t => f t + α * t ^ 2 + β * t + γ) t = deriv (deriv f) t + 2 * α := by
  have h1 : ∀ t, HasDerivAt (fun t => f t + α * t ^ 2 + β * t + γ)
      (deriv f t + α * (2 * t) + β) t := by
    intro t
    have := (((hd t).hasDerivAt.add ((hasDerivAt_pow 2 t).const_mul α)).add
      ((hasDerivAt_id' t).const_mul β)).add_const γ
    convert this using 1; push_cast; ring
  have hD : deriv (fun t => f t + α * t ^ 2 + β * t + γ) =
      fun t => deriv f t + α * (2 * t) + β := funext fun t => (h1 t).deriv
  have h2 : ∀ t, HasDerivAt (fun t => deriv f t + α * (2 * t) + β)
      (deriv (deriv f) t + 2 * α) t := by
    intro t
    have := ((hd2 t).hasDerivAt.add (((hasDerivAt_id' t).const_mul 2).const_mul α)).add_const β
    convert this using 1; ring
  refine ⟨fun t => (h1 t).differentiableAt, ?_, fun t => ?_⟩
  · rw [hD]; exact fun t => (h2 t).differentiableAt
  · simp only [Function.iterate_succ, Function.iterate_zero, Function.comp_apply, id]
    rw [hD]; exact (h2 t).deriv

include hd hd2 in
/-- **Linear interpolation error.** If `|f''| ≤ M`, the chord through `(u, f u)`, `(v, f v)` misses
`f` on `[u, v]` by at most `M (x - u)(v - x) / 2`. -/
theorem interp_error (M : ℝ) (hM : ∀ x, |deriv (deriv f) x| ≤ M) {u v : ℝ} (huv : u < v)
    {x : ℝ} (hx : x ∈ Set.Icc u v) :
    |f x - (f u + (f v - f u) / (v - u) * (x - u))| ≤ M * (x - u) * (v - x) / 2 := by
  set s := (f v - f u) / (v - u)
  have hvu : v - u ≠ 0 := (sub_pos.mpr huv).ne'
  have hseg : x ∈ segment ℝ u v := by rw [segment_eq_Icc huv.le]; exact hx
  -- φ_σ(t) = f t - (f u + s (t - u)) + σ (M/2) (t - u)(t - v), written as f + quadratic
  have form : ∀ σ t, f t - (f u + s * (t - u)) + σ * (M / 2) * ((t - u) * (t - v)) =
      f t + (σ * (M / 2)) * t ^ 2 + (-s - σ * (M / 2) * (u + v)) * t +
        (-f u + s * u + σ * (M / 2) * (u * v)) := by intro σ t; ring
  have hfu : f u - (f u + s * (u - u)) = 0 := by ring
  have hfv : f v - (f u + s * (v - u)) = 0 := by simp only [s]; field_simp; ring
  -- upper bound: σ = +1 is convex
  obtain ⟨d1, d2, dd⟩ := deriv2_add_quad hd hd2 (1 * (M / 2)) (-s - 1 * (M / 2) * (u + v))
    (-f u + s * u + 1 * (M / 2) * (u * v))
  have hconv := convexOn_univ_of_deriv2_nonneg d1 d2 (fun t => by
    rw [dd]; linarith [(abs_le.mp (hM t)).1])
  have hup := hconv.le_on_segment (Set.mem_univ u) (Set.mem_univ v) hseg
  simp only [← form 1] at hup
  rw [hfu, hfv] at hup
  simp only [sub_self, mul_zero, zero_mul, add_zero, max_self] at hup
  -- lower bound: σ = -1 is concave
  obtain ⟨e1, e2, ee⟩ := deriv2_add_quad hd hd2 (-1 * (M / 2)) (-s - -1 * (M / 2) * (u + v))
    (-f u + s * u + -1 * (M / 2) * (u * v))
  have hconc := concaveOn_univ_of_deriv2_nonpos e1 e2 (fun t => by
    rw [ee]; linarith [(abs_le.mp (hM t)).2])
  have hlo := hconc.ge_on_segment (Set.mem_univ u) (Set.mem_univ v) hseg
  simp only [← form (-1)] at hlo
  rw [hfu, hfv] at hlo
  simp only [sub_self, mul_zero, zero_mul, add_zero, min_self] at hlo
  rw [abs_le]
  constructor <;> nlinarith [hx.1, hx.2]

include hd hd2 in
/-- The same, as the familiar `M h² / 8` with `h = v - u`. -/
theorem interp_error_h2 (M : ℝ) (hM : ∀ x, |deriv (deriv f) x| ≤ M) {u v : ℝ} (huv : u < v)
    {x : ℝ} (hx : x ∈ Set.Icc u v) :
    |f x - (f u + (f v - f u) / (v - u) * (x - u))| ≤ M * (v - u) ^ 2 / 8 := by
  refine (interp_error hd hd2 M hM huv hx).trans ?_
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have : (x - u) * (v - x) ≤ (v - u) ^ 2 / 4 := by nlinarith [sq_nonneg (2 * x - u - v)]
  have h2 : M * ((x - u) * (v - x)) ≤ M * ((v - u) ^ 2 / 4) := mul_le_mul_of_nonneg_left this hM0
  nlinarith

end interp

/-! ## The ReLU net is the piecewise-linear interpolant -/

/-- One hidden layer of ReLUs on knots `t 0 < t 1 < …`: bias `y 0`, a linear (skip / first-knot)
term `s 0 (x - t 0)`, and units `ReLU(x - t i)` with output weights `s i - s (i-1)` (the second
differences), `i = 1 … n`. -/
def reluNet (t y : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  let s : ℕ → ℝ := fun i => (y (i + 1) - y i) / (t (i + 1) - t i)
  y 0 + s 0 * (x - t 0) + ∑ i ∈ range n, (s (i + 1) - s i) * relu (x - t (i + 1))

/-- On the `k`-th piece `[t k, t (k+1)]` the ReLU net is the chord through `(t k, y k)` and
`(t (k+1), y (k+1))`. -/
theorem reluNet_on_piece (t y : ℕ → ℝ) (ht : StrictMono t) (n k : ℕ) (hk : k ≤ n) (x : ℝ)
    (hx : x ∈ Set.Icc (t k) (t (k + 1))) :
    reluNet t y n x = y k + (y (k + 1) - y k) / (t (k + 1) - t k) * (x - t k) := by
  set s : ℕ → ℝ := fun i => (y (i + 1) - y i) / (t (i + 1) - t i)
  -- units beyond the piece are off; units up to it are linear
  have hsplit : ∑ i ∈ range n, (s (i + 1) - s i) * relu (x - t (i + 1)) =
      ∑ i ∈ range k, (s (i + 1) - s i) * (x - t (i + 1)) := by
    rw [← Finset.sum_range_add_sum_Ico _ hk]
    rw [Finset.sum_eq_zero (s := Ico k n) (fun i hi => by
      have : x - t (i + 1) ≤ 0 := by
        have := ht.monotone (show k + 1 ≤ i + 1 by have := (Finset.mem_Ico.mp hi).1; omega)
        linarith [hx.2]
      simp [relu, max_eq_right this])]
    rw [add_zero]
    refine Finset.sum_congr rfl (fun i hi => ?_)
    have : 0 ≤ x - t (i + 1) := by
      have := ht.monotone (show i + 1 ≤ k by have := Finset.mem_range.mp hi; omega)
      linarith [hx.1]
    simp [relu, max_eq_left this]
  -- telescoping
  have tele : ∀ k, y 0 + s 0 * (x - t 0) + ∑ i ∈ range k, (s (i + 1) - s i) * (x - t (i + 1)) =
      y k + s k * (x - t k) := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      rw [Finset.sum_range_succ, ← add_assoc, ih]
      have hne : t (k + 1) - t k ≠ 0 := (sub_pos.mpr (ht (Nat.lt_succ_self k))).ne'
      simp only [s]
      field_simp
      ring
  simp only [reluNet]
  rw [hsplit]
  exact tele k

/-- **φ ⊆ ReLU-net, the width law** (bound, proved). Uniform knots `t i = a + i h`,
`h = (b - a)/N`, values `f (t i)`: the one-hidden-layer ReLU net with `N` units (`reluNet` with
`n = N - 1` inner knots plus the first-knot term) is within `M h² / 8 = M (b-a)² / (8 N²)` of `f`
on all of `[a, b]`, whenever `|f''| ≤ M`. -/
theorem relu_net_approx {f : ℝ → ℝ} (hd : Differentiable ℝ f) (hd2 : Differentiable ℝ (deriv f))
    (M : ℝ) (hM : ∀ x, |deriv (deriv f) x| ≤ M) (a b : ℝ) (hab : a < b) (N : ℕ) (hN : 0 < N)
    (x : ℝ) (hx : x ∈ Set.Icc a b) :
    |f x - reluNet (fun i => a + i * ((b - a) / N)) (fun i => f (a + i * ((b - a) / N))) (N - 1) x|
      ≤ M * (b - a) ^ 2 / (8 * N ^ 2) := by
  set h := (b - a) / N
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  have hh : 0 < h := div_pos (sub_pos.mpr hab) hN'
  set t : ℕ → ℝ := fun i => a + i * h
  have ht : StrictMono t := fun i j hij => by
    simp only [t]; have : (i : ℝ) < j := by exact_mod_cast hij
    nlinarith
  -- the piece containing x
  set k := min (⌊(x - a) / h⌋₊) (N - 1)
  have hkN : k ≤ N - 1 := min_le_right _ _
  have hxk : x ∈ Set.Icc (t k) (t (k + 1)) := by
    have hq : 0 ≤ (x - a) / h := div_nonneg (by linarith [hx.1]) hh.le
    have hfl := Nat.floor_le hq
    have hlt := Nat.lt_floor_add_one ((x - a) / h)
    constructor
    · simp only [t]
      have : (k : ℝ) ≤ ⌊(x - a) / h⌋₊ := by exact_mod_cast min_le_left _ _
      have := (le_div_iff₀ hh).mp (this.trans hfl)
      linarith
    · simp only [t]
      rcases le_total (⌊(x - a) / h⌋₊) (N - 1) with hle | hle
      · have hk : k = ⌊(x - a) / h⌋₊ := min_eq_left hle
        rw [hk]; push_cast
        have := (div_lt_iff₀ hh).mp hlt
        linarith
      · have hk : k = N - 1 := min_eq_right hle
        rw [hk]
        have : ((N - 1 : ℕ) : ℝ) + 1 = N := by
          rw [Nat.cast_sub hN]; push_cast; ring
        push_cast [this]
        simp only [h]; field_simp; linarith [hx.2]
  rw [reluNet_on_piece t (fun i => f (t i)) ht (N - 1) k hkN x hxk]
  have := interp_error_h2 hd hd2 M hM (ht (Nat.lt_add_one k)) hxk
  have hstep : t (k + 1) - t k = h := by simp only [t]; push_cast; ring
  rw [hstep] at this ⊢
  refine this.trans (le_of_eq ?_)
  simp only [h]; field_simp

/-! ## Yarotsky's `x²` -/

/-- The tooth `g = 2 ReLU(x) - 4 ReLU(x - 1/2)`: one ReLU layer of width 2. -/
def tooth (x : ℝ) : ℝ := 2 * relu x - 4 * relu (x - 1 / 2)

lemma tooth_mem {x : ℝ} (hx : x ∈ Set.Icc (0 : ℝ) 1) : tooth x ∈ Set.Icc (0 : ℝ) 1 := by
  obtain ⟨h0, h1⟩ := hx
  simp only [tooth, relu, Set.mem_Icc]
  rcases le_total x (1 / 2) with h | h
  · rw [max_eq_left h0, max_eq_right (by linarith)]; constructor <;> linarith
  · rw [max_eq_left h0, max_eq_left (by linarith)]; constructor <;> linarith

lemma iterate_tooth_mem (m : ℕ) {x : ℝ} (hx : x ∈ Set.Icc (0 : ℝ) 1) :
    tooth^[m] x ∈ Set.Icc (0 : ℝ) 1 := by
  induction m with
  | zero => exact hx
  | succ m ih => rw [Function.iterate_succ_apply']; exact tooth_mem ih

/-- The self-similarity of `q(x) = x (1 - x)` under the tooth. -/
lemma q_tooth {x : ℝ} (hx : x ∈ Set.Icc (0 : ℝ) 1) :
    x * (1 - x) = tooth x / 4 + tooth x * (1 - tooth x) / 4 := by
  obtain ⟨h0, h1⟩ := hx
  simp only [tooth, relu]
  rcases le_total x (1 / 2) with h | h
  · rw [max_eq_left h0, max_eq_right (by linarith)]; ring
  · rw [max_eq_left h0, max_eq_left (by linarith)]; ring

/-- Yarotsky's depth-`m` ReLU approximation of `x²` on `[0, 1]`:
`f_m(x) = x - ∑_{s=1}^m g^{∘s}(x) / 4^s`. -/
def yarotskySq (m : ℕ) (x : ℝ) : ℝ := x - ∑ s ∈ range m, tooth^[s + 1] x / 4 ^ (s + 1)

lemma yarotsky_exact (m : ℕ) {x : ℝ} (hx : x ∈ Set.Icc (0 : ℝ) 1) :
    yarotskySq m x - x ^ 2 = tooth^[m] x * (1 - tooth^[m] x) / 4 ^ m := by
  induction m with
  | zero => simp [yarotskySq]; ring
  | succ m ih =>
    have hq := q_tooth (iterate_tooth_mem m hx)
    rw [← Function.iterate_succ_apply' tooth m x] at hq
    simp only [yarotskySq, Finset.sum_range_succ] at ih ⊢
    rw [pow_succ]
    have h4 : (4 : ℝ) ^ m ≠ 0 := by positivity
    field_simp at ih ⊢
    nlinarith [ih, hq]

/-- **Yarotsky's `x²`** (bound, proved). On `[0, 1]`,
`0 ≤ f_m(x) - x² ≤ 4^{-m} / 4 = 2^{-2m-2}`: the error falls as `4^{-L}` in the depth. -/
theorem yarotsky_sq_bound (m : ℕ) {x : ℝ} (hx : x ∈ Set.Icc (0 : ℝ) 1) :
    0 ≤ yarotskySq m x - x ^ 2 ∧ yarotskySq m x - x ^ 2 ≤ 1 / (4 * 4 ^ m) := by
  rw [yarotsky_exact m hx]
  obtain ⟨h0, h1⟩ := iterate_tooth_mem m hx
  have h4 : (0 : ℝ) < 4 ^ m := by positivity
  constructor
  · apply div_nonneg _ h4.le; nlinarith
  · rw [div_le_div_iff₀ h4 (by positivity)]
    nlinarith [sq_nonneg (tooth^[m] x - 1 / 2)]

/-- `x²` on `[-1, 1]` through `|x| = ReLU x + ReLU (-x)`. -/
def yarotskySqSym (m : ℕ) (x : ℝ) : ℝ := yarotskySq m (relu x + relu (-x))

lemma relu_add_relu_neg (x : ℝ) : relu x + relu (-x) = |x| := by
  simp only [relu]
  rcases le_total 0 x with h | h
  · rw [max_eq_left h, max_eq_right (by linarith), abs_of_nonneg h]; ring
  · rw [max_eq_right h, max_eq_left (by linarith), abs_of_nonpos h]; ring

lemma yarotsky_sym_bound (m : ℕ) {x : ℝ} (hx : |x| ≤ 1) :
    |yarotskySqSym m x - x ^ 2| ≤ 1 / (4 * 4 ^ m) := by
  rw [yarotskySqSym, relu_add_relu_neg, ← sq_abs x]
  obtain ⟨h0, h1⟩ := yarotsky_sq_bound m (x := |x|) ⟨abs_nonneg x, hx⟩
  rw [abs_of_nonneg h0]; exact h1

/-- **SwiGLU / GeGLU gate ⊆ ReLU-net, the depth law** (bound, proved). For `|p|, |b| ≤ 1`, the
product `p b = ((p+b)/2)² - ((p-b)/2)²` computed with two depth-`m` Yarotsky squares is within
`1 / (2 · 4^m)` of `p b`. -/
theorem yarotsky_product_bound (m : ℕ) {p b : ℝ} (hp : |p| ≤ 1) (hb : |b| ≤ 1) :
    |(yarotskySqSym m ((p + b) / 2) - yarotskySqSym m ((p - b) / 2)) - p * b| ≤
      1 / (2 * 4 ^ m) := by
  have h1 : |(p + b) / 2| ≤ 1 := by
    rw [abs_div, abs_two, div_le_one two_pos]; exact (abs_add_le p b).trans (by linarith)
  have h2 : |(p - b) / 2| ≤ 1 := by
    rw [abs_div, abs_two, div_le_one two_pos]; exact (abs_sub p b).trans (by linarith)
  have e1 := yarotsky_sym_bound m h1
  have e2 := yarotsky_sym_bound m h2
  have hpb : p * b = ((p + b) / 2) ^ 2 - ((p - b) / 2) ^ 2 := by ring
  rw [hpb]
  calc |(yarotskySqSym m ((p + b) / 2) - yarotskySqSym m ((p - b) / 2)) -
        (((p + b) / 2) ^ 2 - ((p - b) / 2) ^ 2)|
      = |(yarotskySqSym m ((p + b) / 2) - ((p + b) / 2) ^ 2) -
          (yarotskySqSym m ((p - b) / 2) - ((p - b) / 2) ^ 2)| := by ring_nf
    _ ≤ |yarotskySqSym m ((p + b) / 2) - ((p + b) / 2) ^ 2| +
          |yarotskySqSym m ((p - b) / 2) - ((p - b) / 2) ^ 2| := abs_sub _ _
    _ ≤ 1 / (4 * 4 ^ m) + 1 / (4 * 4 ^ m) := add_le_add e1 e2
    _ = 1 / (2 * 4 ^ m) := by field_simp; ring

end EP
