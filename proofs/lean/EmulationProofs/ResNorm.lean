import EmulationProofs.AuditRules
import EmulationProofs.Round2Rec

/-!
# Residual streams and norms paid (branch fr-resnorm)

Three walls of the headline relation get a construction.

* **Hyper-connection hosts switched off** (`hc_identical_streams`, `hc_stack`, `sinkhorn_const`,
  `sinkhorn_const_iter`, `sinkhorn_eps_rowsum_lt`). A host layer with `n` residual streams reads
  `Σ_j pre_j s_j`, writes `post_i f(·)` into stream `i` and mixes the streams by a matrix `C`. With
  every stream equal to `x`, `Σ_j pre_j = 1`, `post_i = 1` and every row of `C` summing to `1`, all
  streams become `x + f x` -- the plain residual update, for ANY sublayer `f` of the whole state
  (attention included). With the dynamic projections zero the mixes are their static bases, and a
  constant comb matrix Sinkhorn-normalises to exactly `1/n` when the normaliser has no `ε`
  (`sinkhorn_const`); with `ε > 0` the row sums are `< 1` (`sinkhorn_eps_rowsum_lt`), so the
  DeepSeek-V4 / GLM-5-Next form needs `hc_eps = 0` (a config value).
* **The LayerNorm-β fold without an lm_head bias, by a dominated final readout**
  (`dominated_readout_limit`, `pad_ratio_limit`, `pos_scale_le_iff`). A last host layer writes
  `M • v` (an affine function of its pre-norm output) into a stream that already holds `x`; the final
  RMSNorm of `x + M • v` tends to the ε-free RMSNorm of `v` as `M → ∞`. With `v = [y; κ]` (`y` the
  target's final LayerNorm output, `κ` a constant pad) the head `(κ/√m)[U; 0]` gives
  `(U·y) κ / √(‖y‖² + κ²)`, which tends to `U·y` as `κ → ∞` and is a positive multiple of it at every
  `κ` (so the argmax, hence greedy text, is the target's).
* **An always-on weight-free embedding norm switched off** (`selfcond_norm_row`,
  `rmsNorm_smul_eps_sq`). Scale the whole residual stream by `α` with host `ε' = α² ε` (exact for
  every norm reading a linear image of the stream, `rmsNorm_smul_eps_sq`); the embedding norm then
  returns `α e` from the row `λ e`, `λ = α² √ε / √(1 − α² ms(e))`, for every `e` once
  `α² ms(e) < 1` -- no condition on the target's embeddings (cf. `selfcond_norm_escape`, which needs
  `ms(e) < 1`).
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## Hyper-connections -/

section hc

variable {V : Type*} [AddCommGroup V] [Module ℝ V] {n : ℕ}

/-- One hyper-connection site: streams `s`, mixing matrix `C`, collapse weights `pre`, expansion
weights `post`, sublayer `f` (a map of the whole state). -/
def hcLayer (C : Fin n → Fin n → ℝ) (pre post : Fin n → ℝ) (f : V → V) (s : Fin n → V) :
    Fin n → V :=
  fun i => ∑ j, C i j • s j + post i • f (∑ j, pre j • s j)

/-- **Identical streams stay identical, and carry the plain residual update** (exact). -/
theorem hc_identical_streams (C : Fin n → Fin n → ℝ) (pre post : Fin n → ℝ) (f : V → V) (x : V)
    (hC : ∀ i, ∑ j, C i j = 1) (hpre : ∑ j, pre j = 1) (hpost : ∀ i, post i = 1) :
    hcLayer C pre post f (fun _ => x) = fun _ => x + f x := by
  funext i
  simp only [hcLayer, ← Finset.sum_smul, hC i, hpre, hpost i, one_smul]

/-- The plain pre-norm stack `x ↦ x + f_l x`, `L` sublayers. -/
def plainRun (f : ℕ → V → V) : ℕ → V → V
  | 0, x => x
  | l + 1, x => plainRun f l x + f l (plainRun f l x)

/-- The hyper-connection stack: `L` sites, the embedding copied to every stream. -/
def hcRun (C : ℕ → Fin n → Fin n → ℝ) (pre post : ℕ → Fin n → ℝ) (f : ℕ → V → V) :
    ℕ → V → Fin n → V
  | 0, x => fun _ => x
  | l + 1, x => hcLayer (C l) (pre l) (post l) (f l) (hcRun C pre post f l x)

/-- **The whole stack** (exact): after every site all `n` streams are the plain stack's residual, and
a head collapsing with weights summing to `1` reads it. -/
theorem hc_stack (C : ℕ → Fin n → Fin n → ℝ) (pre post : ℕ → Fin n → ℝ) (f : ℕ → V → V)
    (hC : ∀ l i, ∑ j, C l i j = 1) (hpre : ∀ l, ∑ j, pre l j = 1) (hpost : ∀ l i, post l i = 1)
    (head : Fin n → ℝ) (hhead : ∑ j, head j = 1) (L : ℕ) (x : V) :
    hcRun C pre post f L x = (fun _ => plainRun f L x) ∧
      ∑ j, head j • hcRun C pre post f L x j = plainRun f L x := by
  have key : ∀ l, hcRun C pre post f l x = fun _ => plainRun f l x := by
    intro l
    induction l with
    | zero => rfl
    | succ l ih =>
      simp only [hcRun, plainRun, ih]
      exact hc_identical_streams _ _ _ _ _ (hC l) (hpre l) (hpost l)
  refine ⟨key L, ?_⟩
  rw [key L]
  simp only [← Finset.sum_smul, hhead, one_smul]

/-- Column normalisation with an epsilon in the normaliser (the Sinkhorn step of the stock code). -/
def colNorm (eps : ℝ) (M : Fin n → Fin n → ℝ) : Fin n → Fin n → ℝ :=
  fun i j => M i j / (∑ k, M k j + eps)

/-- Row normalisation with an epsilon in the normaliser. -/
def rowNorm (eps : ℝ) (M : Fin n → Fin n → ℝ) : Fin n → Fin n → ℝ :=
  fun i j => M i j / (∑ k, M i k + eps)

/-- **A constant positive matrix normalises to `1/n`** (exact) when the normaliser has no epsilon,
by rows or by columns. -/
theorem sinkhorn_const (hn : n ≠ 0) (a : ℝ) (ha : 0 < a) :
    colNorm 0 (fun (_ : Fin n) (_ : Fin n) => a) = (fun _ _ => (1 : ℝ) / n) ∧
      rowNorm 0 (fun (_ : Fin n) (_ : Fin n) => a) = (fun _ _ => (1 : ℝ) / n) := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  constructor <;> funext i j <;>
    simp only [colNorm, rowNorm, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      add_zero] <;> field_simp

/-- **…and stays there under any number of Sinkhorn steps** (exact): the `1/n` matrix is a fixed
point, so every row and column sums to `1`. -/
theorem sinkhorn_const_iter (hn : n ≠ 0) (k : ℕ) :
    (fun M => colNorm 0 (rowNorm 0 M))^[k] (fun (_ : Fin n) (_ : Fin n) => (1 : ℝ) / n) =
      (fun _ _ => (1 : ℝ) / n) := by
  have hpos : (0 : ℝ) < 1 / n := by
    have : (0 : ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn)
    positivity
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih, (sinkhorn_const hn _ hpos).2, (sinkhorn_const hn _ hpos).1]

/-- The `1/n` matrix has unit row sums. -/
theorem uniform_rowsum (hn : n ≠ 0) (_i : Fin n) : ∑ _j : Fin n, (1 : ℝ) / n = 1 := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- **With `ε > 0` in the normaliser the rows sum to less than `1`** (the converse): after a column
step of a constant matrix every entry is `a / (n a + ε)`, a row sums to `n a / (n a + ε) < 1`, so
identical streams shrink by that factor each layer. -/
theorem sinkhorn_eps_rowsum_lt (hn : n ≠ 0) (a eps : ℝ) (ha : 0 < a) (he : 0 < eps) (i : Fin n) :
    ∑ j, colNorm eps (fun (_ : Fin n) (_ : Fin n) => a) i j < 1 := by
  have hn0 : (0 : ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn)
  simp only [colNorm, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [mul_div_assoc', div_lt_one (by positivity)]
  linarith

end hc

/-! ## The dominated final readout -/

section readout

variable {m : ℕ}

/-- `RMSNorm_ε(M • y) = RMSNorm_{ε/M²}(y)` for `M > 0` (the stream-scale fold, stated for the readout). -/
lemma rmsNorm_smul_eps_sq (eps α : ℝ) (hα : 0 < α) (w x : Fin m → ℝ) :
    rmsNorm (α ^ 2 * eps) w (α • x) = rmsNorm eps w x := by
  rw [rmsNorm_smul_eps _ _ hα]
  congr 1
  field_simp

/-- **The dominated readout** (limit): the final RMSNorm of `x + M • v`, any `ε ≥ 0`, tends to the
ε-free RMSNorm of `v` as `M → ∞` (`v ≠ 0`). -/
theorem dominated_readout_limit (hm : m ≠ 0) (eps : ℝ) (w x v : Fin m → ℝ)
    (hv : v ≠ 0) :
    Tendsto (fun M : ℝ => rmsNorm eps w (x + M • v)) atTop (𝓝 (rmsNorm 0 w v)) := by
  have hm0 : (0 : ℝ) < m := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hm)
  have hS : 0 < (∑ l, v l ^ 2) / m := by
    apply div_pos _ hm0
    obtain ⟨k, hk⟩ : ∃ k, v k ≠ 0 := by
      by_contra h; push Not at h; exact hv (funext h)
    exact lt_of_lt_of_le (by positivity) (Finset.single_le_sum (f := fun l => v l ^ 2)
      (fun l _ => sq_nonneg _) (Finset.mem_univ k))
  -- for M > 0 the readout is G(1/M), G continuous at 0
  set G : ℝ → Fin m → ℝ := fun t i =>
    w i * (t * x i + v i) / √((∑ l, (t * x l + v l) ^ 2) / m + eps * t ^ 2) with hG
  have hval : ∀ M : ℝ, 0 < M → rmsNorm eps w (x + M • v) = G M⁻¹ := by
    intro M hM
    funext i
    have e1 : x + M • v = M • (M⁻¹ • x + v) := by
      rw [smul_add, smul_smul, mul_inv_cancel₀ hM.ne', one_smul]
    rw [e1, rmsNorm_smul_eps _ _ hM]
    simp only [rmsNorm, hG, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    congr 2
    field_simp
  have hG0 : G 0 = rmsNorm 0 w v := by
    funext i; simp [hG, rmsNorm]
  have hcont : Tendsto G (𝓝 0) (𝓝 (G 0)) := by
    apply tendsto_pi_nhds.mpr
    intro i
    have hnum : Tendsto (fun t : ℝ => w i * (t * x i + v i)) (𝓝 0) (𝓝 (w i * (0 * x i + v i))) :=
      ((continuous_const.mul ((continuous_id.mul continuous_const).add continuous_const)).tendsto 0)
    have hden : Tendsto (fun t : ℝ => √((∑ l, (t * x l + v l) ^ 2) / m + eps * t ^ 2)) (𝓝 0)
        (𝓝 (√((∑ l, (0 * x l + v l) ^ 2) / m + eps * 0 ^ 2))) := by
      apply Continuous.tendsto
      apply Real.continuous_sqrt.comp
      apply Continuous.add
      · apply Continuous.div_const
        apply continuous_finsetSum
        intro l _
        exact ((continuous_id.mul continuous_const).add continuous_const).pow 2
      · exact continuous_const.mul (continuous_id.pow 2)
    have hpos : √((∑ l, (0 * x l + v l) ^ 2) / m + eps * 0 ^ 2) ≠ 0 := by
      simp only [zero_mul, zero_add, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
        mul_zero, add_zero]
      exact (Real.sqrt_pos.mpr hS).ne'
    exact hnum.div hden hpos
  have hinv : Tendsto (fun M : ℝ => M⁻¹) atTop (𝓝 0) := tendsto_inv_atTop_zero
  rw [← hG0]
  apply (hcont.comp hinv).congr'
  filter_upwards [eventually_gt_atTop 0] with M hM
  simp only [Function.comp]
  rw [hval M hM]

/-- **The pad fixes the divisor** (limit): `(U·y) κ / √(‖y‖² + κ²) → U·y` as `κ → ∞`. -/
theorem pad_ratio_limit (s c : ℝ) (hs : 0 ≤ s) :
    Tendsto (fun κ : ℝ => c * κ / √(s + κ ^ 2)) atTop (𝓝 c) := by
  have hf : ∀ κ : ℝ, 0 < κ → c * κ / √(s + κ ^ 2) = c / √(s * (κ⁻¹) ^ 2 + 1) := by
    intro κ hκ
    have e : s + κ ^ 2 = κ ^ 2 * (s * (κ⁻¹) ^ 2 + 1) := by field_simp
    rw [e, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hκ.le]
    have : 0 < √(s * (κ⁻¹) ^ 2 + 1) := Real.sqrt_pos.mpr (by positivity)
    field_simp
  have hlim : Tendsto (fun κ : ℝ => c / √(s * (κ⁻¹) ^ 2 + 1)) atTop (𝓝 (c / √(s * 0 ^ 2 + 1))) := by
    apply Tendsto.div tendsto_const_nhds _ (by simp)
    apply (Real.continuous_sqrt.tendsto _).comp
    apply Tendsto.add_const
    apply Tendsto.const_mul
    exact (tendsto_inv_atTop_zero.pow 2)
  have h1 : c / √(s * 0 ^ 2 + 1) = c := by simp
  rw [h1] at hlim
  apply hlim.congr'
  filter_upwards [eventually_gt_atTop 0] with κ hκ
  rw [hf κ hκ]

/-- **A positive multiple keeps every comparison** (exact): the dominated readout's logits are
`ρ · (target logits)` with `ρ = κ / √(‖y‖² + κ²) > 0`, so the argmax (any tie-break that compares
logits) and hence the greedy text are the target's at every `M`-limit, every `κ`. -/
theorem pos_scale_le_iff {ι : Type*} (ρ : ℝ) (hρ : 0 < ρ) (z : ι → ℝ) (a b : ι) :
    ρ * z a ≤ ρ * z b ↔ z a ≤ z b :=
  mul_le_mul_iff_of_pos_left hρ

end readout

/-! ## The always-on embedding norm -/

section selfcond

variable {d : ℕ}

/-- **The embedding norm returns `α e`** (exact): with host `ε' = α² ε` and the row
`λ e`, `λ = α² √ε / √(1 − α² ms(e))`, for every `e` with `α² ms(e) < 1` -- which a small `α` gives
for every embedding row. With the stream scaled by `α` everywhere (`rmsNorm_smul_eps_sq`) the host is
the target exactly on an untied head; a tied head multiplies logit `v` by `λ_v / (α² √ε)`
`= 1/√(1 − α² ms(e_v)) → 1` as `α → 0`. -/
theorem selfcond_norm_row (hd : d ≠ 0) (eps α : ℝ) (he : 0 < eps) (hα : 0 < α) (e : Fin d → ℝ)
    (hms : α ^ 2 * ((∑ l, e l ^ 2) / d) < 1) :
    rmsNorm (α ^ 2 * eps) 1 ((α ^ 2 * √eps / √(1 - α ^ 2 * ((∑ l, e l ^ 2) / d))) • e) = α • e := by
  set s := (∑ l, e l ^ 2) / d with hs
  have hs0 : 0 ≤ s := by positivity
  have h1 : 0 < 1 - α ^ 2 * s := by linarith
  have hsq : √(1 - α ^ 2 * s) ^ 2 = 1 - α ^ 2 * s := Real.sq_sqrt h1.le
  have hq : 0 < √(1 - α ^ 2 * s) := Real.sqrt_pos.mpr h1
  have hse : √eps ^ 2 = eps := Real.sq_sqrt he.le
  set c := α ^ 2 * √eps / √(1 - α ^ 2 * s) with hc
  have hcpos : 0 < c := by positivity
  have hc2 : c ^ 2 = α ^ 4 * eps / (1 - α ^ 2 * s) := by
    rw [hc, div_pow, mul_pow, hsq, hse]; ring
  have hsum : (∑ l, (c * e l) ^ 2) / d + α ^ 2 * eps = (c / α) ^ 2 := by
    have : (∑ l, (c * e l) ^ 2) / d = c ^ 2 * s := by
      rw [hs, mul_div_assoc', Finset.mul_sum]; simp only [mul_pow]
    have h2 : (c / α) ^ 2 = c ^ 2 / α ^ 2 := div_pow c α 2
    rw [this, h2, hc2]
    field_simp
    ring
  funext i
  simp only [rmsNorm, Pi.one_apply, one_mul, Pi.smul_apply, smul_eq_mul]
  rw [hsum, Real.sqrt_sq (by positivity)]
  field_simp

end selfcond

end EP
