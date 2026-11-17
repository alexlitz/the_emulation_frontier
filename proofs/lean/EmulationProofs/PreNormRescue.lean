import EmulationProofs.PreNorm
import EmulationProofs.CapacityDepth
import Mathlib.Analysis.Analytic.Uniqueness
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Pre-norm rescue attempts: compensating channels, the ε budget, and what stays impossible

`prenorm_readout_obstruction` (in `PreNorm.lean`) refutes carries whose carried vector does not
move with the stream. It leaves open hosts whose earlier layers COMPUTE compensating channels, as
`learned_abs_two_block` does. This file settles what such channels can and cannot do.

**What works (exact).**

* **The target's ε is a constant-norm budget** (`const_norm_carry`). A target RMSNorm divides by
  `√(‖x‖²/d + ε)`: the `d ε` is a virtual channel of fixed squared norm. A host residual
  `[x; z]` whose extra channels have a FIXED squared norm `S` reads the stream exactly with
  `ε' = (d ε − S)/(d + e)` (`≥ 0` iff `S ≤ d ε`; with `ε' = 0` the extra channels are a pad that
  IS the target's ε). The carried channels themselves come out as `z / R`, `R` a positive
  multiple of the target's RMS: exact up to a positive per-token factor.
* **Consequence: the MoE select survives a pre-norm on a strong router margin**
  (`moe_select_prenorm_exact`). Clamped pair units `relu(t) − relu(t − δ)` take only the values
  `0` and `δ` when every pairwise router gap is at least `δ`, so the select vector has a fixed
  norm. A pad channel holding the same norm before the select keeps every read exact, and the
  masks only need signs, which the factor `1/R` preserves. This needs constants (MLP biases)
  in the host.

**What does not (impossibility, on any nonempty open set of stream states).**

* With host `ε = 0` and no pad, every sublayer output is invariant under scaling of the token's
  residual, so no sublayer (attention, MLP, anything) makes the residual's RMS constant on a
  segment of a ray (`eps_zero_no_equaliser`).
* With `ε > 0` (or a pad), a write that is a real-analytic function of the normed residual
  (SiLU / GeLU / GLU MLPs, softmax attention as a function of one token) or an affine one (a ReLU
  MLP on any open set where it is affine) is bounded and analytic in the stream state. Such a
  write cannot make the residual's RMS constant on any open set of states
  (`no_prenorm_rms_equaliser`), nor write the channel `λ √(‖x‖² + c)` that a constant readout of
  the next norm requires (`no_prenorm_homogeneous_channel`). `learned_abs_two_block` escapes
  because its residuals take finitely many values (tokens × positions), where an interpolant is
  exact.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## The ε budget: constant-norm extra channels are free -/

/-- **ε is a constant-norm budget** (exact). Host residual `[x; z]`: the target stream `x`
(`d` channels) and any extra channels `z` (`e` channels) of squared norm `S`. A host RMSNorm with
weight `√(d/(d+e))·w` on the stream (any weight `wz` on the extra channels) and
`ε' = (d ε − S)/(d + e)` returns the target's `RMSNorm_ε(x)` on the stream, and `wz ⊙ z / R` on
the extra channels, where `R = √(d/(d+e)) · √(‖x‖²/d + ε)` is a fixed multiple of the target's
RMS. With `S = 0` this is `qknorm_pad` / the hidden pad. A real host needs `ε' ≥ 0`, i.e.
`S ≤ d ε`; scaling the whole stream by `λ` (and `ε` by `λ²`, `rmsNorm_smul_eps`) multiplies the
budget by `λ²`, so it is not a real limit. The identity needs `S` to be the SAME number for every
input that reaches this norm: a content-dependent `‖z‖²` breaks it
(`prenorm_readout_obstruction`). -/
theorem const_norm_carry {d e : ℕ} (hd : d ≠ 0) (eps S : ℝ)
    (w x : Fin d → ℝ) (z wz : Fin e → ℝ) (hS : ∑ j, z j ^ 2 = S) :
    (∀ i, rmsNorm ((d * eps - S) / (d + e : ℕ))
        (Fin.append (fun i => √((d : ℝ) / (d + e : ℕ)) * w i) wz) (Fin.append x z)
        (Fin.castAdd e i) = rmsNorm eps w x i) ∧
    (∀ j, rmsNorm ((d * eps - S) / (d + e : ℕ))
        (Fin.append (fun i => √((d : ℝ) / (d + e : ℕ)) * w i) wz) (Fin.append x z)
        (Fin.natAdd d j) =
      wz j * z j / (√((d : ℝ) / (d + e : ℕ)) * √((∑ l, x l ^ 2) / d + eps))) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hde : (0 : ℝ) < (d + e : ℕ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hd).trans_le (by omega)
  have key : √((∑ l : Fin (d + e), (Fin.append x z l) ^ 2) / (d + e : ℕ) +
      (d * eps - S) / (d + e : ℕ)) =
      √((d : ℝ) / (d + e : ℕ)) * √((∑ l, x l ^ 2) / d + eps) := by
    rw [← Real.sqrt_mul (by positivity)]
    congr 1
    simp only [Fin.sum_univ_add, Fin.append_left, Fin.append_right, hS]
    field_simp
    ring
  refine ⟨fun i => ?_, fun j => ?_⟩
  · simp only [rmsNorm, Fin.append_left]
    rw [key]
    have : 0 < √((d : ℝ) / (d + e : ℕ)) := Real.sqrt_pos.mpr (by positivity)
    rcases (Real.sqrt_nonneg ((∑ l, x l ^ 2) / d + eps)).lt_or_eq with h | h
    · field_simp
    · rw [← h]; simp
  · simp only [rmsNorm, Fin.append_right]
    rw [key]

/-! ## Impossibility 1: with `ε = 0` nothing equalises the RMS -/

/-- With `ε = 0` the RMSNorm is invariant under positive scaling. -/
lemma rmsNorm_zero_smul {m : ℕ} (w h : Fin m → ℝ) {ρ : ℝ} (hρ : 0 < ρ) :
    rmsNorm 0 w (ρ • h) = rmsNorm 0 w h := by
  funext i
  simp only [rmsNorm, Pi.smul_apply, smul_eq_mul, add_zero]
  have hs : (∑ l, (ρ * h l) ^ 2) / m = ρ ^ 2 * ((∑ l, h l ^ 2) / m) := by
    rw [mul_div_assoc', Finset.mul_sum]; congr 1
    exact Finset.sum_congr rfl (fun l _ => by ring)
  rw [hs, Real.sqrt_mul (sq_nonneg ρ), Real.sqrt_sq hρ.le]
  rcases (Real.sqrt_nonneg ((∑ l, h l ^ 2) / m)).lt_or_eq with h0 | h0
  · field_simp
  · rw [← h0]; simp

/-- **With host `ε = 0`, no sublayer equalises the RMS** (any sublayer, no analyticity). The
token's host residual `ρ h` (`h ≠ 0`; a stream state on a ray, with no constant pad), plus the
sublayer's write `F(RMSNorm₀(ρ h))`, where `F` is ANY function of the normed residual (an MLP,
or an attention whose other positions are held fixed). The write does not depend on `ρ`, so
the squared norm of the new residual is a quadratic in `ρ` with leading coefficient `‖h‖² > 0`.
It is constant on no interval of scales. -/
theorem eps_zero_no_equaliser {m : ℕ} (w h : Fin m → ℝ) (hh : h ≠ 0)
    (F : (Fin m → ℝ) → Fin m → ℝ) (C a b : ℝ) (ha : 0 < a) (hab : a < b)
    (hC : ∀ ρ, a < ρ → ρ < b → ∑ i, (ρ * h i + F (rmsNorm 0 w (ρ • h)) i) ^ 2 = C) : False := by
  set G := F (rmsNorm 0 w h)
  have q : ∀ ρ, a < ρ → ρ < b →
      ρ ^ 2 * (∑ i, h i ^ 2) + 2 * ρ * (∑ i, h i * G i) + ∑ i, G i ^ 2 = C := by
    intro ρ h1 h2
    have := hC ρ h1 h2
    rw [rmsNorm_zero_smul w h (ha.trans h1)] at this
    rw [← this, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl (fun i _ => by ring)
  have hH := sum_sq_pos_of_ne_zero h hh
  set s := (b - a) / 4
  have hs : 0 < s := by simp only [s]; linarith
  have k1 := q (a + s) (by linarith) (by simp only [s]; linarith)
  have k2 := q (a + 2 * s) (by linarith) (by simp only [s]; linarith)
  have k3 := q (a + 3 * s) (by linarith) (by simp only [s]; linarith)
  have : 2 * s ^ 2 * (∑ i, h i ^ 2) = 0 := by linear_combination k1 - 2 * k2 + k3
  have : 0 < 2 * s ^ 2 * (∑ i, h i ^ 2) := by positivity
  linarith

/-- **With `ε = 0` the normed vector has a fixed squared norm `m`** (weight `1`, `h ≠ 0`). A
bias-free GLU unit pair `act(u_i) u_i − act(−u_i) u_i = u_i²` (`silu_identity_pair`; also ReLU,
GeLU) summed over `i` therefore writes the exact constant `m`: a bias-free GLU host with `ε = 0`
has exact constant WRITES (`2m` units), though not constants inside its activations. A bias-free
ReLU / SiLU MLP has none: it is positively homogeneous, or analytic with value `0` at `0`. -/
lemma rmsNorm_zero_sq_sum {m : ℕ} (h : Fin m → ℝ) (hh : h ≠ 0) :
    ∑ i, (rmsNorm 0 1 h i) ^ 2 = m := by
  have hH := sum_sq_pos_of_ne_zero h hh
  have hm : (0 : ℝ) < m := by
    rcases Nat.eq_zero_or_pos m with h0 | h0
    · subst h0; exact absurd hH (by simp)
    · exact_mod_cast h0
  simp only [rmsNorm, Pi.one_apply, one_mul, add_zero, div_pow]
  rw [← Finset.sum_div, Real.sq_sqrt (by positivity)]
  field_simp

/-! ## Impossibility 2: analytic (or affine) writes, any `ε ≥ 0` -/

lemma analyticAt_coord {n : ℕ} (j : Fin n) (x : Fin n → ℝ) :
    AnalyticAt ℝ (fun x : Fin n → ℝ => x j) x :=
  (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin n => ℝ) j).analyticAt x

lemma analyticAt_affine_coord {n : ℕ} (A : Fin n → ℝ) (b : ℝ) (x : Fin n → ℝ) :
    AnalyticAt ℝ (fun x : Fin n → ℝ => ∑ j, A j * x j + b) x :=
  (Finset.analyticAt_fun_sum _ (fun j _ => analyticAt_const.mul (analyticAt_coord j x))).add
    analyticAt_const

/-- `√f` is analytic where `f` is analytic and positive. -/
lemma AnalyticAt.sqrt_of_pos {E : Type} [NormedAddCommGroup E] [NormedSpace ℝ E] {f : E → ℝ}
    {x : E} (hf : AnalyticAt ℝ f x) (hx : 0 < f x) : AnalyticAt ℝ (fun y => √(f y)) x := by
  have hev : (fun y => exp (log (f y) / 2)) =ᶠ[𝓝 x] fun y => √(f y) := by
    filter_upwards [hf.continuousAt.eventually (lt_mem_nhds hx)] with y hy
    rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos hy]
    ring_nf
  have hlog : AnalyticAt ℝ (fun y => log (f y)) x :=
    AnalyticAt.comp (g := Real.log) (f := f) (analyticAt_log hx) hf
  have hl : AnalyticAt ℝ (fun y => log (f y) / 2) x := hlog.div_const
  exact hl.rexp'.congr hev

/-- An analytic function on `ℝⁿ` that is constant on a nonempty open set is constant. -/
lemma analytic_const_of_open {n : ℕ} {f : (Fin n → ℝ) → ℝ}
    (hf : AnalyticOnNhd ℝ f Set.univ) {U : Set (Fin n → ℝ)} (hU : IsOpen U) {x₀ : Fin n → ℝ}
    (hx₀ : x₀ ∈ U) {C : ℝ} (hC : ∀ x ∈ U, f x = C) (x : Fin n → ℝ) : f x = C := by
  have hev : f =ᶠ[𝓝 x₀] fun _ => C := by
    filter_upwards [hU.mem_nhds hx₀] with y hy using hC y hy
  exact AnalyticOnNhd.eqOn_of_preconnected_of_eventuallyEq hf analyticOnNhd_const
    convex_univ.isPreconnected (Set.mem_univ x₀) hev (Set.mem_univ x)

/-- **A bounded analytic write never equalises the RMS** (the core lemma). Stream states
`x ∈ ℝⁿ` enter the host residual linearly, `A x + b` (`A ≠ 0`), and the host's sublayers add a
write `Y x` that is real-analytic on all of `ℝⁿ` and bounded. Then the squared norm of the
residual `A x + b + Y x` is constant on NO nonempty open set of states. Proof: it is analytic,
so constancy on an open set propagates to all of `ℝⁿ`; along a line where `A x` grows, the
bounded write cannot keep up. -/
theorem no_bounded_analytic_equaliser {n m : ℕ} (A : Fin m → Fin n → ℝ) (b : Fin m → ℝ)
    (Y : (Fin n → ℝ) → Fin m → ℝ) (hY : ∀ i, AnalyticOnNhd ℝ (fun x => Y x i) Set.univ)
    (B : ℝ) (hB : ∀ x i, |Y x i| ≤ B) (hA : ∃ i j, A i j ≠ 0)
    {U : Set (Fin n → ℝ)} (hU : IsOpen U) (hne : U.Nonempty) (C : ℝ)
    (h : ∀ x ∈ U, ∑ i, (∑ j, A i j * x j + b i + Y x i) ^ 2 = C) : False := by
  obtain ⟨x₀, hx₀⟩ := hne
  have hf : AnalyticOnNhd ℝ (fun x => ∑ i, (∑ j, A i j * x j + b i + Y x i) ^ 2) Set.univ :=
    fun x _ => Finset.analyticAt_fun_sum _
      (fun i _ => ((analyticAt_affine_coord (A i) (b i) x).add (hY i x trivial)).pow 2)
  have hall := analytic_const_of_open hf hU hx₀ h
  obtain ⟨i₀, j₀, hij⟩ := hA
  have hpos : 0 < |A i₀ j₀| := abs_pos.mpr hij
  set t := (|b i₀| + B + |C| + 1) / A i₀ j₀
  set x : Fin n → ℝ := Pi.single j₀ t
  have hx : ∑ j, A i₀ j * x j = |b i₀| + B + |C| + 1 := by
    simp only [x, Pi.single_apply, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ,
      ite_true, t]
    field_simp
  have hB0 : 0 ≤ B := (abs_nonneg _).trans (hB x i₀)
  have hterm : (|C| + 1) ^ 2 ≤ (∑ j, A i₀ j * x j + b i₀ + Y x i₀) ^ 2 := by
    rw [hx]
    have h1 := neg_abs_le (b i₀)
    have h2 := neg_abs_le (Y x i₀)
    have h2' := hB x i₀
    have : |C| + 1 ≤ |b i₀| + B + |C| + 1 + b i₀ + Y x i₀ := by linarith
    exact pow_le_pow_left₀ (by positivity) this 2
  have hle : (∑ j, A i₀ j * x j + b i₀ + Y x i₀) ^ 2 ≤
      ∑ i, (∑ j, A i j * x j + b i + Y x i) ^ 2 :=
    Finset.single_le_sum (f := fun i => (∑ j, A i j * x j + b i + Y x i) ^ 2)
      (fun i _ => sq_nonneg _) (Finset.mem_univ i₀)
  rw [hall x] at hle
  have : |C| < (|C| + 1) ^ 2 := by nlinarith [abs_nonneg C]
  linarith [le_abs_self C]

/-- RMSNorm is bounded: every coordinate is at most `|w i| √m` (any `ε ≥ 0`). -/
lemma rmsNorm_abs_le {m : ℕ} (eps : ℝ) (heps : 0 ≤ eps) (w y : Fin m → ℝ) (i : Fin m) :
    |rmsNorm eps w y i| ≤ |w i| * √m := by
  unfold rmsNorm
  rcases (Finset.sum_nonneg (fun l (_ : l ∈ Finset.univ) => sq_nonneg (y l))).lt_or_eq with hS | hS
  · have hm : (0 : ℝ) < m := by
      rcases Nat.eq_zero_or_pos m with h0 | h0
      · subst h0; exact absurd hS (by simp)
      · exact_mod_cast h0
    have hyi : |y i| ≤ √(∑ l, y l ^ 2) := by
      rw [← Real.sqrt_sq_eq_abs]
      exact Real.sqrt_le_sqrt (Finset.single_le_sum (f := fun l => y l ^ 2)
        (fun l _ => sq_nonneg _) (Finset.mem_univ i))
    have hden : √(∑ l, y l ^ 2) / √m ≤ √((∑ l, y l ^ 2) / m + eps) := by
      rw [← Real.sqrt_div' _ hm.le]
      exact Real.sqrt_le_sqrt (by linarith)
    have hsq : 0 < √(∑ l, y l ^ 2) := Real.sqrt_pos.mpr hS
    have hsm : 0 < √(m : ℝ) := Real.sqrt_pos.mpr hm
    have hden0 : 0 < √((∑ l, y l ^ 2) / m + eps) := lt_of_lt_of_le (by positivity) hden
    rw [abs_div, abs_mul, abs_of_pos hden0, div_le_iff₀ hden0]
    calc |w i| * |y i| ≤ |w i| * √(∑ l, y l ^ 2) := by gcongr
      _ = |w i| * √m * (√(∑ l, y l ^ 2) / √m) := by field_simp
      _ ≤ |w i| * √m * √((∑ l, y l ^ 2) / m + eps) := by gcongr
  · have : y i = 0 := by
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun l (_ : l ∈ Finset.univ) =>
        sq_nonneg (y l))).mp hS.symm i (Finset.mem_univ i)
      exact pow_eq_zero_iff (by norm_num) |>.mp this
    simp [this]; positivity

/-- **Pre-norm writes are bounded and analytic in the stream state.** Stream state `x`, host
residual `A x + b`, host RMSNorm (any weight, `ε ≥ 0`, positive argument everywhere: `ε > 0`, or a
pad channel `b` outside the range of `A`), and a sublayer `F` analytic on all of `ℝᵐ` (a
SiLU / GeLU / GLU MLP, an attention head as a function of this token's normed input, or an affine
map: a ReLU MLP on one of its linear pieces). The write `x ↦ F(RMSNorm(A x + b))` is analytic and
bounded on `ℝⁿ`. -/
theorem prenorm_write_analytic_bounded {n m : ℕ} (eps : ℝ) (heps : 0 ≤ eps) (w : Fin m → ℝ)
    (A : Fin m → Fin n → ℝ) (b : Fin m → ℝ)
    (hpos : ∀ x : Fin n → ℝ, 0 < (∑ l, (∑ j, A l j * x j + b l) ^ 2) / m + eps)
    (F : (Fin m → ℝ) → Fin m → ℝ) (hF : ∀ i, AnalyticOnNhd ℝ (fun u => F u i) Set.univ) :
    (∀ i, AnalyticOnNhd ℝ
      (fun x : Fin n → ℝ => F (rmsNorm eps w (fun l => ∑ j, A l j * x j + b l)) i) Set.univ) ∧
    ∃ B, ∀ (x : Fin n → ℝ) i, |F (rmsNorm eps w (fun l => ∑ j, A l j * x j + b l)) i| ≤ B := by
  constructor
  · intro i x _
    have hN : AnalyticAt ℝ (fun x => rmsNorm eps w (fun l => ∑ j, A l j * x j + b l)) x := by
      rw [analyticAt_pi_iff]
      intro l
      simp only [rmsNorm]
      refine AnalyticAt.div (analyticAt_const.mul (analyticAt_affine_coord (A l) (b l) x)) ?_
        (Real.sqrt_pos.mpr (hpos x)).ne'
      refine AnalyticAt.sqrt_of_pos ?_ (hpos x)
      exact ((Finset.analyticAt_fun_sum _ (fun k _ =>
        (analyticAt_affine_coord (A k) (b k) x).pow 2)).div_const).add analyticAt_const
    have hFi : AnalyticAt ℝ (fun u => F u i) (rmsNorm eps w (fun l => ∑ j, A l j * x j + b l)) :=
      hF i _ trivial
    exact AnalyticAt.comp (g := fun u => F u i)
      (f := fun x : Fin n → ℝ => rmsNorm eps w (fun l => ∑ j, A l j * x j + b l)) hFi hN
  · set R := (∑ i, |w i|) * √m
    have hK : IsCompact (Metric.closedBall (0 : Fin m → ℝ) R) := isCompact_closedBall _ _
    have hFc : ContinuousOn F (Metric.closedBall (0 : Fin m → ℝ) R) :=
      (continuous_pi (fun i => continuousOn_univ.mp (hF i).continuousOn)).continuousOn
    obtain ⟨B, hB⟩ := hK.exists_bound_of_continuousOn hFc
    refine ⟨B, fun x i => ?_⟩
    have hmem : rmsNorm eps w (fun l => ∑ j, A l j * x j + b l) ∈
        Metric.closedBall (0 : Fin m → ℝ) R := by
      rw [mem_closedBall_zero_iff, pi_norm_le_iff_of_nonneg (by positivity)]
      intro l
      rw [Real.norm_eq_abs]
      refine (rmsNorm_abs_le eps heps w _ l).trans ?_
      show |w l| * √m ≤ (∑ i, |w i|) * √m
      refine mul_le_mul_of_nonneg_right ?_ (Real.sqrt_nonneg _)
      exact Finset.single_le_sum (f := fun i => |w i|) (fun i _ => abs_nonneg _)
        (Finset.mem_univ l)
    exact (norm_le_pi_norm _ i).trans (hB _ hmem) |>.trans' (le_of_eq (Real.norm_eq_abs _).symm)

/-- **No analytic sublayer equalises a pre-norm host's RMS** (impossibility). Host residual
`A x + b` for stream state `x` (`A ≠ 0`), one host RMSNorm (any weight, `ε ≥ 0` with a positive
argument everywhere), a sublayer `F` analytic on all of `ℝᵐ` (SiLU / GeLU / GLU MLP; an affine
map, i.e. a ReLU MLP restricted to one linear piece). The residual after the write,
`A x + b + F(RMSNorm(A x + b))`, has constant squared norm on NO nonempty open set of stream
states. So the compensator `√(c − ‖x‖²)` that `learned_abs_two_block` writes is not computable
for content-dependent residuals; it works there because the residuals take finitely many
values. Deeper stacks of such sublayers are covered by `no_bounded_analytic_equaliser`, as their
total write stays bounded and analytic. -/
theorem no_prenorm_rms_equaliser {n m : ℕ} (eps : ℝ) (heps : 0 ≤ eps) (w : Fin m → ℝ)
    (A : Fin m → Fin n → ℝ) (b : Fin m → ℝ)
    (hpos : ∀ x : Fin n → ℝ, 0 < (∑ l, (∑ j, A l j * x j + b l) ^ 2) / m + eps)
    (F : (Fin m → ℝ) → Fin m → ℝ) (hF : ∀ i, AnalyticOnNhd ℝ (fun u => F u i) Set.univ)
    (hA : ∃ i j, A i j ≠ 0) {U : Set (Fin n → ℝ)} (hU : IsOpen U) (hne : U.Nonempty) (C : ℝ)
    (h : ∀ x ∈ U, ∑ i, (∑ j, A i j * x j + b i +
      F (rmsNorm eps w (fun l => ∑ j, A l j * x j + b l)) i) ^ 2 = C) : False := by
  obtain ⟨hY, B, hB⟩ := prenorm_write_analytic_bounded eps heps w A b hpos F hF
  exact no_bounded_analytic_equaliser A b _ hY B hB hA hU hne C h

/-- **A constant readout of a pre-norm needs a channel proportional to the RMS** (algebra).
Host residual `[y; ζ]` (one extra channel `ζ`, weight `wζ`, any `ε ≥ 0`). If the normed `ζ`
channel reads a fixed `t ≠ 0` (a q bias, a sink logit, LayerNorm's `β`, an ALiBi slope), then
`ζ² (m wζ² − t²) = t² (‖y‖² + m ε)`: `ζ` is `λ √(‖y‖² + m ε)`, the token's ε-regularised norm up
to a fixed factor. No analytic write produces that (`no_prenorm_homogeneous_channel`). -/
theorem const_readout_channel {k : ℕ} (eps wζ ζ t : ℝ) (y : Fin k → ℝ) (ht : t ≠ 0)
    (h : wζ * ζ / √((∑ i, y i ^ 2 + ζ ^ 2) / (k + 1 : ℕ) + eps) = t) :
    ζ ^ 2 * ((k + 1 : ℕ) * wζ ^ 2 - t ^ 2) = t ^ 2 * (∑ i, y i ^ 2 + (k + 1 : ℕ) * eps) := by
  set r := √((∑ i, y i ^ 2 + ζ ^ 2) / (k + 1 : ℕ) + eps)
  have hr : r ≠ 0 := by intro h0; rw [h0, div_zero] at h; exact ht h.symm
  have hnn : 0 ≤ (∑ i, y i ^ 2 + ζ ^ 2) / (k + 1 : ℕ) + eps := by
    by_contra hneg
    exact hr (Real.sqrt_eq_zero'.mpr (le_of_lt (not_le.mp hneg)))
  have h1 : wζ * ζ = t * r := by rw [div_eq_iff hr] at h; exact h
  have h2 : (wζ * ζ) ^ 2 = t ^ 2 * ((∑ i, y i ^ 2 + ζ ^ 2) / (k + 1 : ℕ) + eps) := by
    rw [h1, mul_pow, Real.sq_sqrt hnn]
  have hk : (0 : ℝ) < (k + 1 : ℕ) := by positivity
  field_simp at h2
  linear_combination h2

/-- **No analytic write is proportional to the token's RMS** (impossibility). A constant readout
of the next pre-norm (a q bias, a sink logit, LayerNorm's `β`, the ALiBi key term `m j`) needs a
channel `ζ` with `ζ / RMS = const`, i.e. `ζ = λ √(‖x‖² + c)` (`c > 0` the ε / pad term). A
bounded analytic write `G` (`prenorm_write_analytic_bounded`) equals it on no nonempty open set
of stream states: by analyticity it would equal it everywhere, and it is unbounded. -/
theorem no_prenorm_homogeneous_channel {n : ℕ} (hn : 0 < n) (G : (Fin n → ℝ) → ℝ)
    (hG : AnalyticOnNhd ℝ G Set.univ) (B : ℝ) (hB : ∀ x, |G x| ≤ B) (lam c : ℝ)
    (hlam : lam ≠ 0) (hc : 0 < c) {U : Set (Fin n → ℝ)} (hU : IsOpen U) (hne : U.Nonempty)
    (h : ∀ x ∈ U, G x = lam * √(∑ i, x i ^ 2 + c)) : False := by
  obtain ⟨x₀, hx₀⟩ := hne
  have hH : AnalyticOnNhd ℝ (fun x : Fin n → ℝ => lam * √(∑ i, x i ^ 2 + c)) Set.univ := by
    intro x _
    refine analyticAt_const.mul (AnalyticAt.sqrt_of_pos ?_ (by positivity))
    exact (Finset.analyticAt_fun_sum _ (fun i _ => (analyticAt_coord i x).pow 2)).add
      analyticAt_const
  have hev : G =ᶠ[𝓝 x₀] fun x => lam * √(∑ i, x i ^ 2 + c) := by
    filter_upwards [hU.mem_nhds hx₀] with y hy using h y hy
  have hall := AnalyticOnNhd.eqOn_of_preconnected_of_eventuallyEq hG hH
    convex_univ.isPreconnected (Set.mem_univ x₀) hev
  set i₀ : Fin n := ⟨0, hn⟩
  set t := (|B| + 1) / |lam|
  have hl : 0 < |lam| := abs_pos.mpr hlam
  have h1 : G (Pi.single i₀ t) = lam * √(∑ i, (Pi.single i₀ t : Fin n → ℝ) i ^ 2 + c) :=
    hall (Set.mem_univ (Pi.single i₀ t : Fin n → ℝ))
  have h2 := hB (Pi.single i₀ t : Fin n → ℝ)
  rw [h1] at h2
  have hsum : ∑ i, (Pi.single i₀ t : Fin n → ℝ) i ^ 2 = t ^ 2 := by
    simp [Pi.single_apply]
  rw [hsum, abs_mul] at h2
  have ht : 0 ≤ t := by positivity
  have hsq : t ≤ √(t ^ 2 + c) := Real.le_sqrt_of_sq_le (by linarith)
  have : |lam| * t = |B| + 1 := by simp only [t]; field_simp
  have : |lam| * t ≤ |lam| * |√(t ^ 2 + c)| := by
    rw [abs_of_nonneg (Real.sqrt_nonneg _)]; gcongr
  linarith [le_abs_self B]

/-! ## The MoE select under a pre-norm, on a strong router margin -/

/-- A clamp from two ReLU units, one with a bias: `relu t − relu (t − δ)`. -/
def clampU (δ t : ℝ) : ℝ := relu t - relu (t - δ)

lemma clampU_nonneg {δ : ℝ} (hδ : 0 ≤ δ) (t : ℝ) : 0 ≤ clampU δ t := by
  unfold clampU relu
  exact sub_nonneg.mpr (max_le_max (by linarith) le_rfl)

lemma clampU_of_nonpos {δ t : ℝ} (hδ : 0 ≤ δ) (ht : t ≤ 0) : clampU δ t = 0 := by
  unfold clampU
  rw [relu_of_nonpos ht, relu_of_nonpos (by linarith)]; ring

lemma clampU_of_ge {δ t : ℝ} (hδ : 0 ≤ δ) (ht : δ ≤ t) : clampU δ t = δ := by
  unfold clampU
  rw [relu_of_nonneg (by linarith), relu_of_nonneg (by linarith)]; ring

/-- The select layer's pair features: `g e e' = clamp_δ(Λ (z e' − z e))`, one clamp (two ReLU
units) per ordered pair of experts. -/
def pairSel {E : ℕ} (δ Λ : ℝ) (z : Fin E → ℝ) (e e' : Fin E) : ℝ :=
  clampU δ (Λ * (z e' - z e))

/-- A strong router margin: every pairwise gap of the scaled logits is at least `δ`. -/
def StrongMargin {E : ℕ} (δ Λ : ℝ) (z : Fin E → ℝ) : Prop :=
  ∀ e e', e ≠ e' → δ ≤ Λ * |z e - z e'|

lemma pairSel_pair_sq {E : ℕ} {δ Λ : ℝ} {z : Fin E → ℝ} (hδ : 0 < δ)
    (hz : StrongMargin δ Λ z) {e e' : Fin E} (hee : e ≠ e') :
    pairSel δ Λ z e e' ^ 2 + pairSel δ Λ z e' e ^ 2 = δ ^ 2 := by
  have hm := hz e e' hee
  unfold pairSel
  rcases le_total (z e) (z e') with h | h
  · rw [abs_of_nonpos (by linarith)] at hm
    rw [clampU_of_ge hδ.le (by linarith), clampU_of_nonpos hδ.le (by nlinarith)]; ring
  · rw [abs_of_nonneg (by linarith)] at hm
    rw [clampU_of_nonpos hδ.le (by nlinarith), clampU_of_ge hδ.le (by linarith)]; ring

/-- **On a strong margin the select vector has a fixed norm**: `∑ g² = E(E−1)/2 · δ²`, whatever
the input. -/
theorem pairSel_sq_sum {E : ℕ} {δ Λ : ℝ} {z : Fin E → ℝ} (hδ : 0 < δ)
    (hz : StrongMargin δ Λ z) :
    ∑ e, ∑ e', pairSel δ Λ z e e' ^ 2 = (E : ℝ) * ((E : ℝ) - 1) / 2 * δ ^ 2 := by
  have hdiag : ∀ e, pairSel δ Λ z e e = 0 := fun e => by
    simp [pairSel, clampU_of_nonpos hδ.le]
  have h2 : 2 * ∑ e, ∑ e', pairSel δ Λ z e e' ^ 2 =
      ∑ e, ∑ e', (pairSel δ Λ z e e' ^ 2 + pairSel δ Λ z e' e ^ 2) := by
    simp only [Finset.sum_add_distrib]
    rw [Finset.sum_comm (f := fun e e' => pairSel δ Λ z e' e ^ 2)]; ring
  have h3 : ∀ e e', pairSel δ Λ z e e' ^ 2 + pairSel δ Λ z e' e ^ 2 =
      δ ^ 2 - if e = e' then δ ^ 2 else 0 := by
    intro e e'
    by_cases hee : e = e'
    · subst hee; simp [hdiag]
    · simp [hee, pairSel_pair_sq hδ hz hee]
  simp only [h3, Finset.sum_sub_distrib, Finset.sum_ite_eq, Finset.mem_univ, ite_true,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h2
  linarith

lemma pairSel_winner {E : ℕ} {δ Λ : ℝ} {z : Fin E → ℝ} (hδ : 0 ≤ δ) (hΛ : 0 ≤ Λ)
    (estar : Fin E) (hwin : ∀ e, e ≠ estar → z e < z estar) :
    ∑ e', pairSel δ Λ z estar e' = 0 := by
  refine Finset.sum_eq_zero (fun e' _ => clampU_of_nonpos hδ ?_)
  rcases eq_or_ne e' estar with h | h
  · simp [h]
  · exact mul_nonpos_of_nonneg_of_nonpos hΛ (by linarith [hwin e' h])

lemma pairSel_loser {E : ℕ} {δ Λ : ℝ} {z : Fin E → ℝ} (hδ : 0 ≤ δ)
    (hz : StrongMargin δ Λ z) (estar : Fin E) (hwin : ∀ e, e ≠ estar → z e < z estar)
    {e : Fin E} (he : e ≠ estar) : δ ≤ ∑ e', pairSel δ Λ z e e' := by
  have hm := hz e estar he
  rw [abs_of_neg (by linarith [hwin e he])] at hm
  calc δ = pairSel δ Λ z e estar := (clampU_of_ge hδ (by linarith)).symm
    _ ≤ ∑ e', pairSel δ Λ z e e' :=
      Finset.single_le_sum (f := fun e' => pairSel δ Λ z e e')
        (fun _ _ => clampU_nonneg hδ _) (Finset.mem_univ estar)

/-- Masked expert units: if the mask is `0` for the winner and at least `σ` for every loser, and
`λ σ` dominates every loser's pre-activation, the masked layer is the winner's FFN (exact). -/
lemma masked_experts {E ew : ℕ} {V : Type*} [AddCommGroup V] [Module ℝ V]
    (s : Fin E → ℝ) (p : Fin E → Fin ew → ℝ) (v : Fin E → Fin ew → V) (estar : Fin E)
    (σ lam : ℝ) (hlam : 0 ≤ lam) (hs0 : s estar = 0) (hsσ : ∀ e, e ≠ estar → σ ≤ s e)
    (hmask : ∀ e, e ≠ estar → ∀ j, p e j ≤ lam * σ) :
    ∑ e, ∑ j, relu (p e j - lam * s e) • v e j = ∑ j, relu (p estar j) • v estar j := by
  rw [Finset.sum_eq_single estar]
  · simp [hs0]
  · intro e _ he
    refine Finset.sum_eq_zero (fun j _ => ?_)
    rw [relu_of_nonpos, zero_smul]
    have := hmask e he j
    have := mul_le_mul_of_nonneg_left (hsσ e he) hlam
    linarith
  · simp

/-- The select channels, flattened: channel `(e, e')` of `E·E`. -/
def pairFlat {E : ℕ} (δ Λ : ℝ) (z : Fin E → ℝ) (k : Fin (E * E)) : ℝ :=
  pairSel δ Λ z (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2

lemma pairFlat_sq_sum {E : ℕ} (δ Λ : ℝ) (z : Fin E → ℝ) :
    ∑ k, pairFlat δ Λ z k ^ 2 = ∑ e, ∑ e', pairSel δ Λ z e e' ^ 2 := by
  rw [← Fintype.sum_prod_type']
  exact Equiv.sum_comp finProdFinEquiv.symm (fun q => pairSel δ Λ z q.1 q.2 ^ 2)

/-- **The MoE select survives a pre-norm on a strong router margin** (exact). Target: a
pre-norm (RMSNorm, `ε > 0`, weight `w`) top-1 unit-gate MoE with ReLU experts, router logits
`z = W_r N(x)`, expert pre-activations `p e j = P_{ej} · N(x)`. Host: residual `[x; g; κ]`
(`E²` select channels `g`, one pad `κ`), ONE RMSNorm over all of it with
`ε' = (d ε − S)/(d + E² + 1)`, `S = E(E−1)/2 · δ²`, and two FFN layers with biases:

* the select layer reads `[x; 0; √S]`, writes the clamped pair features `g` (`2E²` ReLU units,
  one bias each) and `−√S` on the pad (a bias);
* the expert layer reads `[x; g; 0]`, masks expert `e` by `λ ∑_{e'} (normed g)_{e e'}`, recomputes
  `g` from the same exact logits to erase it, and writes `+√S` back on the pad.

On inputs with a strong margin (`StrongMargin δ Λ z`, every pairwise logit gap `≥ δ/Λ`) both
reads give the target's `N(x)` exactly (the extra channels have squared norm `S` at both,
`const_norm_carry`, `pairSel_sq_sum`), the masks are `0` for the winner and `≥ δ/R` for the
losers (`R` the target RMS times `√(d/(d+E²+1))`), and the expert layer's output is the
winner's FFN. Conditions: `S ≤ d ε` for `ε' ≥ 0` (take `δ` small, or scale the stream);
`λ δ / R` above every loser pre-activation, a bound on `‖x‖` (bounded inputs); MLP biases (a
bias-free ReLU MLP writes no nonzero constant on an open set). Contrast
`select_carry_prenorm_fails`: there the select vector's norm varies with the input. -/
theorem moe_select_prenorm_exact {d E ew : ℕ} {V : Type*} [AddCommGroup V] [Module ℝ V]
    (hd : d ≠ 0) (eps : ℝ) (heps : 0 < eps) (w x : Fin d → ℝ) (Wr : Fin E → Fin d → ℝ)
    (P : Fin E → Fin ew → Fin d → ℝ) (v : Fin E → Fin ew → V) (δ Λ lam : ℝ) (hδ : 0 < δ)
    (hΛ : 0 ≤ Λ) (hlam : 0 ≤ lam) (estar : Fin E) :
    let N := rmsNorm eps w x
    let z : Fin E → ℝ := fun e => ∑ i, Wr e i * N i
    let p : Fin E → Fin ew → ℝ := fun e j => ∑ i, P e j i * N i
    let S : ℝ := (E : ℝ) * ((E : ℝ) - 1) / 2 * δ ^ 2
    let eps' : ℝ := (d * eps - S) / (d + (E * E + 1) : ℕ)
    let W : Fin (d + (E * E + 1)) → ℝ :=
      Fin.append (fun i => √((d : ℝ) / (d + (E * E + 1) : ℕ)) * w i) (fun _ => 1)
    let before : Fin (E * E + 1) → ℝ := Fin.append (0 : Fin (E * E) → ℝ) (fun _ => √S)
    let after : Fin (E * E + 1) → ℝ := Fin.append (pairFlat δ Λ z) (fun _ => 0)
    let R : ℝ := √((d : ℝ) / (d + (E * E + 1) : ℕ)) * √((∑ l, x l ^ 2) / d + eps)
    let s : Fin E → ℝ := fun e => ∑ e', rmsNorm eps' W (Fin.append x after)
      (Fin.natAdd d (Fin.castAdd 1 (finProdFinEquiv (e, e'))))
    StrongMargin δ Λ z → (∀ e, e ≠ estar → z e < z estar) →
    (∀ e, e ≠ estar → ∀ j, p e j ≤ lam * (δ / R)) →
    (∀ i, rmsNorm eps' W (Fin.append x before) (Fin.castAdd _ i) = N i) ∧
    (∀ i, rmsNorm eps' W (Fin.append x after) (Fin.castAdd _ i) = N i) ∧
    (∑ e, ∑ j, relu (p e j - lam * s e) • v e j = ∑ j, relu (p estar j) • v estar j) := by
  intro N z p S eps' W before after R s hz hwin hmask
  have hS0 : 0 ≤ S := by
    have := pairSel_sq_sum hδ hz
    simp only [S]; rw [← this]; positivity
  have hb : ∑ j, before j ^ 2 = S := by
    simp only [before, Fin.sum_univ_add, Fin.append_left, Fin.append_right, Pi.zero_apply]
    simp [Real.sq_sqrt hS0]
  have ha : ∑ j, after j ^ 2 = S := by
    simp only [after, Fin.sum_univ_add, Fin.append_left, Fin.append_right]
    simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, Finset.sum_const_zero,
      add_zero, pairFlat_sq_sum]
    exact pairSel_sq_sum hδ hz
  obtain ⟨hb1, -⟩ := const_norm_carry hd eps S w x before (fun _ => 1) hb
  obtain ⟨ha1, ha2⟩ := const_norm_carry hd eps S w x after (fun _ => 1) ha
  refine ⟨hb1, ha1, ?_⟩
  have hR : 0 < R := by
    have : 0 < (d : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hd
    have : (0 : ℝ) < (d + (E * E + 1) : ℕ) := by positivity
    exact mul_pos (Real.sqrt_pos.mpr (by positivity)) (Real.sqrt_pos.mpr (by positivity))
  have hs : ∀ e, s e = (∑ e', pairSel δ Λ z e e') / R := by
    intro e
    simp only [s]
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl (fun e' _ => ?_)
    rw [ha2]
    simp [after, pairFlat, R]
  refine masked_experts s p v estar (δ / R) lam hlam ?_ ?_ hmask
  · rw [hs, pairSel_winner hδ.le hΛ estar hwin, zero_div]
  · intro e he
    rw [hs]
    exact div_le_div_of_nonneg_right (pairSel_loser hδ.le hz estar hwin he) hR.le

/-! ## The vanishing-scale carry: magnitude-insensitive consumers survive in the limit -/

/-- **A carry at vanishing scale is a limit construction** (any `ε > 0`). Host residual
`[x; δ s]`: the carried vector `s` written at scale `δ`. As `δ → 0` the host RMSNorm (hidden-pad
epsilon `ε d/(d+e)`) converges to the target's on the stream, and the carried channels, read
and multiplied by `1/δ`, converge to `wz ⊙ s / R` with `R` the target RMS times `√(d/(d+e))`:
the carry is recovered up to a positive per-token factor. Consumers that only need signs or a
threshold (the MoE masks, `moe_two_layer_margin` with `λ ∝ 1/δ`) are therefore rescued as a
limit for every pre-norm host, bias-free and analytic (SiLU, SwiGLU) ones included; consumers
that need the magnitude (CLA K/V, the parallel block's attention output, a constant readout) are
not, as the factor `1/R` remains (`no_prenorm_homogeneous_channel`). -/
theorem vanishing_carry_limit {d e : ℕ} (hd : d ≠ 0) (eps : ℝ) (heps : 0 < eps)
    (w x : Fin d → ℝ) (s wz : Fin e → ℝ) :
    let W : Fin (d + e) → ℝ := Fin.append (fun i => √((d : ℝ) / (d + e : ℕ)) * w i) wz
    let R : ℝ := √((d : ℝ) / (d + e : ℕ)) * √((∑ l, x l ^ 2) / d + eps)
    (∀ i, Tendsto (fun δ : ℝ => rmsNorm (d * eps / (d + e : ℕ)) W (Fin.append x (δ • s))
        (Fin.castAdd e i)) (𝓝 0) (𝓝 (rmsNorm eps w x i))) ∧
    (∀ j, Tendsto (fun δ : ℝ => δ⁻¹ * rmsNorm (d * eps / (d + e : ℕ)) W
        (Fin.append x (δ • s)) (Fin.natAdd d j)) (𝓝[≠] 0) (𝓝 (wz j * s j / R))) := by
  intro W R
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hde : (0 : ℝ) < (d + e : ℕ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hd).trans_le (by omega)
  have heps' : 0 < d * eps / (d + e : ℕ) := by positivity
  have hcont : Continuous (fun δ : ℝ => rmsNorm (d * eps / (d + e : ℕ)) W
      (Fin.append x (δ • s))) := by
    refine (rmsNorm_continuous _ heps' W).comp ?_
    refine continuous_pi (fun i => ?_)
    refine Fin.addCases (fun i => ?_) (fun i => ?_) i
    · simp only [Fin.append_left]; exact continuous_const
    · simp only [Fin.append_right, Pi.smul_apply, smul_eq_mul]; fun_prop
  have h0 : Fin.append x ((0 : ℝ) • s) = Fin.append x (0 : Fin e → ℝ) := by simp
  have hS0 : ∑ j, (0 : Fin e → ℝ) j ^ 2 = 0 := by simp
  have hb := (const_norm_carry hd eps 0 w x 0 wz hS0).1
  simp only [sub_zero] at hb
  refine ⟨fun i => ?_, fun j => ?_⟩
  · have := ((continuous_apply (Fin.castAdd e i)).comp hcont).tendsto 0
    simp only [Function.comp_apply, h0] at this
    rwa [hb i] at this
  · -- the carried read is `wz j * (δ s j) / r(δ)`; divide by `δ` and let `δ → 0`
    have hr : Continuous (fun δ : ℝ => √((∑ l : Fin (d + e), (Fin.append x (δ • s) l) ^ 2) /
        (d + e : ℕ) + d * eps / (d + e : ℕ))) := by
      refine Real.continuous_sqrt.comp ?_
      refine Continuous.add (Continuous.div_const (continuous_finsetSum _ (fun l _ => ?_)) _)
        continuous_const
      refine Continuous.pow ?_ 2
      refine Fin.addCases (fun i => ?_) (fun i => ?_) l
      · simp only [Fin.append_left]; exact continuous_const
      · simp only [Fin.append_right, Pi.smul_apply, smul_eq_mul]; fun_prop
    have hr0 : √((∑ l : Fin (d + e), (Fin.append x ((0 : ℝ) • s) l) ^ 2) / (d + e : ℕ) +
        d * eps / (d + e : ℕ)) = R := by
      simp only [R, zero_smul, Fin.sum_univ_add, Fin.append_left, Fin.append_right,
        Pi.zero_apply, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
        Finset.sum_const_zero, add_zero]
      rw [← Real.sqrt_mul (by positivity)]
      congr 1
      field_simp
    have hRpos : 0 < R := mul_pos (Real.sqrt_pos.mpr (by positivity))
      (Real.sqrt_pos.mpr (by positivity))
    have hlim : Tendsto (fun δ : ℝ => wz j * s j / √((∑ l : Fin (d + e),
        (Fin.append x (δ • s) l) ^ 2) / (d + e : ℕ) + d * eps / (d + e : ℕ))) (𝓝 0)
        (𝓝 (wz j * s j / R)) := by
      have := (hr.tendsto 0)
      rw [hr0] at this
      exact tendsto_const_nhds.div this hRpos.ne'
    refine (hlim.mono_left nhdsWithin_le_nhds).congr' ?_
    filter_upwards [self_mem_nhdsWithin] with δ hδ
    have hδ' : δ ≠ 0 := hδ
    simp only [rmsNorm, W, Fin.append_right, Pi.smul_apply, smul_eq_mul]
    field_simp

end EP
