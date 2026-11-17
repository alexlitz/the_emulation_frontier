import EmulationProofs.Attention

/-!
# A GQA / MQA / MHA target on an MLA host: the latent RMSNorm on the VALUE path

An MLA host's values are `W_UV · RMSNorm(c)`, `c = W_DKV x`, where the latent norm (DeepSeek
`kv_a_layernorm`) cannot be switched off; the decoupled RoPE key has no value counterpart. On a
single-token context (position 0 of a causal model) an attention sublayer outputs its own value,
through the output projection: the host writes `M · RMSNorm(W x)` (`M = W_O W_UV`, all heads
share the latent), the target writes `A x` (`A = W_O W_V`), linear.

* `mla_value_ray`: if the two agree at two distinct positive multiples of an input `x₀`, then
  `A x₀ = 0` (the host is degree-0 along the ray, the target degree-1).
* `mla_host_not_prenorm_target`: a target with a pre-norm `ε > 0` (every zoo model) feeds its
  attention every scale along every ray, so an MLA host reproduces its single-token attention
  output only if that output is `0`, at ANY latent rank.
* `mla_host_sphere_rank`: with `ε = 0` the inputs lie on a sphere; if the latent rank `r` is below
  the input width `n`, again only `A = 0` works.
* `mla_host_sphere_full_rank`: …and with `r ≥ n` (an isometric down-projection) the latent norm
  divides by a constant on the sphere, so the host is exact: the only case where it is possible.
-/

open Real Finset Matrix

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {n r m : ℕ}

lemma rmsNorm_smul (eps : ℝ) (w c : Fin r → ℝ) (s : ℝ) :
    rmsNorm eps w (s • c) =
      (s / √((s ^ 2 * ∑ l, c l ^ 2) / r + eps)) • (fun i => w i * c i) := by
  funext i
  simp only [rmsNorm, Pi.smul_apply, smul_eq_mul, mul_pow, ← Finset.mul_sum]
  ring

/-- **Along a ray the MLA host's value is degree 0, the target's degree 1.** If
`M · RMSNorm_{ε,w}(W (s x₀)) = A (s x₀)` at two distinct positive scales `s₁ ≠ s₂`, then
`A x₀ = 0` (any latent rank, any weight, any `ε ≥ 0`). -/
theorem mla_value_ray (hr : 0 < r) (eps : ℝ) (heps : 0 ≤ eps) (w : Fin r → ℝ)
    (W : Matrix (Fin r) (Fin n) ℝ) (M : Matrix (Fin m) (Fin r) ℝ) (A : Matrix (Fin m) (Fin n) ℝ)
    (x₀ : Fin n → ℝ) (s₁ s₂ : ℝ) (h1 : 0 < s₁) (h2 : 0 < s₂) (h12 : s₁ ≠ s₂)
    (h : ∀ s ∈ ({s₁, s₂} : Set ℝ), M *ᵥ rmsNorm eps w (W *ᵥ (s • x₀)) = A *ᵥ (s • x₀)) :
    A *ᵥ x₀ = 0 := by
  set c := W *ᵥ x₀
  set K := ∑ l, c l ^ 2
  set y := M *ᵥ (fun i => w i * c i)
  have hrR : (0 : ℝ) < r := by exact_mod_cast hr
  have key : ∀ s, 0 < s → s ∈ ({s₁, s₂} : Set ℝ) →
      (1 / √((s ^ 2 * K) / r + eps)) • y = A *ᵥ x₀ := by
    intro s hs hmem
    have := h s hmem
    rw [Matrix.mulVec_smul, Matrix.mulVec_smul, rmsNorm_smul, Matrix.mulVec_smul] at this
    have hs' : s ≠ 0 := hs.ne'
    have : s • ((1 / √((s ^ 2 * K) / r + eps)) • y) = s • (A *ᵥ x₀) := by
      rw [← this, smul_smul]; congr 1; ring
    exact smul_right_injective _ hs' this
  have k1 := key s₁ h1 (by simp)
  have k2 := key s₂ h2 (by simp)
  by_cases hK : K = 0
  · -- the latent is zero: the host value is zero
    have hc : ∀ l, c l = 0 := by
      intro l
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun l _ => sq_nonneg (c l))).mp hK l
        (Finset.mem_univ _)
      exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this
    have hy : y = 0 := by
      have : (fun i => w i * c i) = 0 := by funext i; simp [hc i]
      simp only [y, this, Matrix.mulVec_zero]
    rw [← k1, hy, smul_zero]
  · have hKpos : 0 < K := lt_of_le_of_ne (Finset.sum_nonneg (fun l _ => sq_nonneg _)) (Ne.symm hK)
    have hD : ∀ s, 0 < s → 0 < √((s ^ 2 * K) / r + eps) := fun s hs =>
      Real.sqrt_pos.mpr (by positivity)
    have hne : √((s₁ ^ 2 * K) / r + eps) ≠ √((s₂ ^ 2 * K) / r + eps) := by
      intro he
      rw [Real.sqrt_inj (by positivity) (by positivity)] at he
      have : s₁ ^ 2 = s₂ ^ 2 := by
        have := congrArg (fun t => (t - eps) * r / K) he
        simp only [add_sub_cancel_right] at this
        field_simp at this
        exact this
      have hm : (s₁ - s₂) * (s₁ + s₂) = 0 := by ring_nf; linarith
      rcases mul_eq_zero.mp hm with h | h
      · exact h12 (by linarith)
      · linarith
    have hy : y = 0 := by
      have := k1.trans k2.symm
      rw [← sub_eq_zero, ← sub_smul] at this
      rcases smul_eq_zero.mp this with h | h
      · exfalso
        rw [sub_eq_zero, one_div, one_div, _root_.inv_inj] at h
        exact hne h
      · exact h
    rw [← k1, hy, smul_zero]

/-- **A pre-norm target with `ε > 0` on an MLA host** (impossibility, any latent rank). The target
attention reads `x = RMSNorm_{ε_t, w_t}(h)`; the host (zero-padded or not, its own pre-norm
folded into `W`) reads the same `x`. If the host's single-token output `M · RMSNorm(W x)` equals
the target's `A x` for every residual `h`, then the target's single-token attention output is
`0` for every `h`: no nonzero GQA / MQA / MHA attention sublayer is reproduced. -/
theorem mla_host_not_prenorm_target (hr : 0 < r) (eps : ℝ) (heps : 0 ≤ eps) (w : Fin r → ℝ)
    (W : Matrix (Fin r) (Fin n) ℝ) (M : Matrix (Fin m) (Fin r) ℝ) (A : Matrix (Fin m) (Fin n) ℝ)
    (epsT : ℝ) (hepsT : 0 < epsT) (wT : Fin n → ℝ)
    (h : ∀ hres : Fin n → ℝ,
      M *ᵥ rmsNorm eps w (W *ᵥ rmsNorm epsT wT hres) = A *ᵥ rmsNorm epsT wT hres) :
    ∀ hres, A *ᵥ rmsNorm epsT wT hres = 0 := by
  intro hres
  set x₀ : Fin n → ℝ := fun i => wT i * hres i
  set K := ∑ l, hres l ^ 2
  have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  -- RMSNorm along the ray `λ hres` is `σ(λ) x₀`
  have hray : ∀ lam : ℝ, rmsNorm epsT wT (lam • hres) =
      (lam / √((lam ^ 2 * K) / n + epsT)) • x₀ := by
    intro lam; rw [rmsNorm_smul]
  set σ : ℝ → ℝ := fun lam => lam / √((lam ^ 2 * K) / n + epsT)
  have hσpos : ∀ lam, 0 < lam → 0 < σ lam := fun lam hl =>
    div_pos hl (Real.sqrt_pos.mpr (by positivity))
  have hσ12 : σ 1 ≠ σ 2 := by
    intro he
    simp only [σ] at he
    have hD1 : 0 < √((1 ^ 2 * K) / n + epsT) := Real.sqrt_pos.mpr (by positivity)
    have hD2 : 0 < √((2 ^ 2 * K) / n + epsT) := Real.sqrt_pos.mpr (by positivity)
    rw [div_eq_div_iff hD1.ne' hD2.ne'] at he
    have hsq := congrArg (fun t => t ^ 2) he
    simp only [mul_pow, Real.sq_sqrt (by positivity : (0 : ℝ) ≤ (1 ^ 2 * K) / n + epsT),
      Real.sq_sqrt (by positivity : (0 : ℝ) ≤ (2 ^ 2 * K) / n + epsT)] at hsq
    ring_nf at hsq
    linarith
  have hx₀ : A *ᵥ x₀ = 0 := by
    refine mla_value_ray hr eps heps w W M A x₀ (σ 1) (σ 2) (hσpos 1 one_pos) (hσpos 2 two_pos)
      hσ12 ?_
    intro s hs
    rcases hs with rfl | rfl
    · have := h ((1 : ℝ) • hres); rwa [hray] at this
    · have := h ((2 : ℝ) • hres); rwa [hray] at this
  have := hray 1
  rw [one_smul] at this
  rw [this, Matrix.mulVec_smul, hx₀, smul_zero]

/-- **The sphere case, latent rank below the input width** (impossibility). With the target's
`ε = 0` its attention inputs lie on a sphere `∑ xᵢ² = R²` (the norm weight folded into `W` and
`A`). If `r < n`, an MLA host whose single-token output `M · RMSNorm(W x)` equals `A x` on the
whole sphere has `A = 0`. (A kernel vector `k` of `W` gives `A k = 0`; for any `v`, the points
`s v⊥ + t k` of the sphere put the host on a ray of `v⊥`, and `mla_value_ray` applies.) -/
theorem mla_host_sphere_rank (hr : 0 < r) (hrn : r < n) (eps : ℝ) (heps : 0 ≤ eps)
    (w : Fin r → ℝ) (W : Matrix (Fin r) (Fin n) ℝ) (M : Matrix (Fin m) (Fin r) ℝ)
    (A : Matrix (Fin m) (Fin n) ℝ) (R : ℝ) (hR : 0 < R)
    (h : ∀ x : Fin n → ℝ, ∑ i, x i ^ 2 = R ^ 2 → M *ᵥ rmsNorm eps w (W *ᵥ x) = A *ᵥ x) :
    A = 0 := by
  -- a kernel vector
  obtain ⟨k, hkker, hk0⟩ : ∃ k ∈ LinearMap.ker (Matrix.mulVecLin W), k ≠ 0 :=
    (Submodule.ne_bot_iff _).mp (LinearMap.ker_ne_bot_of_finrank_lt (by simpa using hrn))
  have hWk : W *ᵥ k = 0 := by simpa using hkker
  set Kk := ∑ i, k i ^ 2
  have hKk : 0 < Kk := by
    obtain ⟨i, hi⟩ := Function.ne_iff.mp hk0
    exact Finset.sum_pos' (fun j _ => sq_nonneg _)
      ⟨i, Finset.mem_univ _, lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hi))⟩
  have hnorm0 : rmsNorm eps w (0 : Fin r → ℝ) = 0 := by funext i; simp [rmsNorm]
  -- the sphere points `s v + t k` for `v ⊥ k`
  have hsph : ∀ (v : Fin n → ℝ), v ⬝ᵥ k = 0 → ∀ s t : ℝ,
      ∑ i, (s • v + t • k) i ^ 2 = s ^ 2 * ∑ i, v i ^ 2 + t ^ 2 * Kk := by
    intro v hv s t
    have : ∑ i, v i * k i = 0 := by simpa [dotProduct] using hv
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Kk, Finset.mul_sum]
    have e : ∀ i, (s * v i + t * k i) ^ 2 =
        s ^ 2 * v i ^ 2 + t ^ 2 * k i ^ 2 + 2 * s * t * (v i * k i) := fun i => by ring
    simp only [e, Finset.sum_add_distrib, ← Finset.mul_sum, this, mul_zero, add_zero]
  -- step 1: `A k = 0`
  have hAk : A *ᵥ k = 0 := by
    have hsK : 0 < √Kk := Real.sqrt_pos.mpr hKk
    have hpt := h ((R / √Kk) • k) (by
      have e : ∀ i, ((R / √Kk) • k) i ^ 2 = (R / √Kk) ^ 2 * k i ^ 2 := fun i => by
        simp only [Pi.smul_apply, smul_eq_mul]; ring
      simp only [e, ← Finset.mul_sum]
      rw [show ∑ i, k i ^ 2 = Kk from rfl, div_pow, Real.sq_sqrt hKk.le]
      field_simp)
    rw [Matrix.mulVec_smul, hWk, smul_zero, hnorm0, Matrix.mulVec_zero, Matrix.mulVec_smul] at hpt
    have hc : R / √Kk ≠ 0 := div_ne_zero hR.ne' (Real.sqrt_pos.mpr hKk).ne'
    exact (smul_eq_zero.mp hpt.symm).resolve_left hc
  -- step 2: every `v`
  ext i j
  set v : Fin n → ℝ := Pi.single j 1
  set vp : Fin n → ℝ := v - ((v ⬝ᵥ k) / Kk) • k
  have hkk : k ⬝ᵥ k = Kk := by simp [dotProduct, Kk, sq]
  have hvp : vp ⬝ᵥ k = 0 := by
    have : vp ⬝ᵥ k = v ⬝ᵥ k - ((v ⬝ᵥ k) / Kk) * (k ⬝ᵥ k) := by
      simp only [vp, dotProduct, Pi.sub_apply, Pi.smul_apply, smul_eq_mul, sub_mul,
        Finset.sum_sub_distrib, Finset.mul_sum, mul_assoc]
    rw [this, hkk]; field_simp; ring
  have hAv : A *ᵥ v = A *ᵥ vp := by
    simp only [vp, Matrix.mulVec_sub, Matrix.mulVec_smul, hAk, smul_zero, sub_zero]
  have hWv : W *ᵥ vp = W *ᵥ v := by
    simp only [vp, Matrix.mulVec_sub, Matrix.mulVec_smul, hWk, smul_zero, sub_zero]
  have hAvp : A *ᵥ vp = 0 := by
    set P := ∑ i, vp i ^ 2
    by_cases hP : P = 0
    · have : vp = 0 := by
        funext l
        have := (Finset.sum_eq_zero_iff_of_nonneg (fun l _ => sq_nonneg (vp l))).mp hP l
          (Finset.mem_univ _)
        simpa using this
      rw [this, Matrix.mulVec_zero]
    have hPpos : 0 < P := lt_of_le_of_ne (Finset.sum_nonneg (fun l _ => sq_nonneg _)) (Ne.symm hP)
    set s₁ := R / (2 * √P)
    set s₂ := R / (4 * √P)
    have hsP : 0 < √P := Real.sqrt_pos.mpr hPpos
    refine mla_value_ray hr eps heps w W M A vp s₁ s₂ (by positivity) (by positivity) ?_ ?_
    · intro he
      simp only [s₁, s₂] at he
      rw [div_eq_div_iff (by positivity) (by positivity)] at he
      nlinarith
    · intro s hs
      have hsq : s ^ 2 * P < R ^ 2 := by
        rcases hs with rfl | rfl
        · simp only [s₁, div_pow, mul_pow, Real.sq_sqrt hPpos.le]
          field_simp; nlinarith
        · simp only [s₂, div_pow, mul_pow, Real.sq_sqrt hPpos.le]
          field_simp; nlinarith
      set t := √((R ^ 2 - s ^ 2 * P) / Kk)
      have hpt := h (s • vp + t • k) (by
        rw [hsph vp hvp s t, Real.sq_sqrt (div_nonneg (by linarith) hKk.le)]
        field_simp; ring)
      simp only [Matrix.mulVec_add, Matrix.mulVec_smul, hWk, hAk, smul_zero, add_zero] at hpt
      simpa only [Matrix.mulVec_smul] using hpt
  have := congrFun (hAv.trans hAvp) i
  simpa [v, Matrix.mulVec, dotProduct, Pi.single_apply] using this

/-- **The sphere case, full latent rank** (exact: the only case where it works). If the
down-projection is a scaled isometry (`Wᵀ W = α I`, `α > 0`, needs `r ≥ n`) and the inputs lie on
the sphere `∑ xᵢ² = R²`, the latent norm divides by the constant `√(α R²/r + ε)`, and the host
with `M = A Wᵀ diag(1/w) · √(αR²/r + ε)/α` (weights `w ≠ 0`) reproduces `A x` exactly. -/
theorem mla_host_sphere_full_rank (eps : ℝ) (heps : 0 ≤ eps) (w : Fin r → ℝ)
    (hw : ∀ i, w i ≠ 0) (W : Matrix (Fin r) (Fin n) ℝ) (α : ℝ) (hα : 0 < α)
    (hW : Wᵀ * W = α • (1 : Matrix (Fin n) (Fin n) ℝ)) (A : Matrix (Fin m) (Fin n) ℝ) (R : ℝ)
    (x : Fin n → ℝ) (hx : ∑ i, x i ^ 2 = R ^ 2) :
    let D := √(α * R ^ 2 / r + eps)
    (D / α) • (A * Wᵀ * Matrix.diagonal (fun i => (w i)⁻¹)) *ᵥ rmsNorm eps w (W *ᵥ x) =
      A *ᵥ x := by
  intro D
  have hc : ∑ l, (W *ᵥ x) l ^ 2 = α * R ^ 2 := by
    have : ∑ l, (W *ᵥ x) l ^ 2 = x ⬝ᵥ ((Wᵀ * W) *ᵥ x) := by
      rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]
      simp [dotProduct, sq]
    rw [this, hW, Matrix.smul_mulVec, Matrix.one_mulVec, dotProduct_smul, smul_eq_mul, ← hx]
    simp [dotProduct, sq]
  have hD : 0 < D ∨ D = 0 := (Real.sqrt_nonneg _).lt_or_eq.imp id Eq.symm
  have hnorm : rmsNorm eps w (W *ᵥ x) = D⁻¹ • (fun i => w i * (W *ᵥ x) i) := by
    funext i; simp only [rmsNorm, hc, Pi.smul_apply, smul_eq_mul, D]; ring
  have hdiag : Matrix.diagonal (fun i => (w i)⁻¹) *ᵥ (fun i => w i * (W *ᵥ x) i) = W *ᵥ x := by
    funext i; simp [Matrix.mulVec_diagonal, hw i]
  have key : (A * Wᵀ * Matrix.diagonal (fun i => (w i)⁻¹)) *ᵥ (fun i => w i * (W *ᵥ x) i) =
      α • (A *ᵥ x) := by
    rw [← Matrix.mulVec_mulVec, hdiag, Matrix.mulVec_mulVec, Matrix.mul_assoc, hW,
      Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mulVec]
  rw [hnorm, Matrix.mulVec_smul, key, smul_smul, smul_smul]
  rcases hD with hD | hD
  · rw [show D / α * D⁻¹ * α = 1 by field_simp, one_smul]
  · -- `D = 0` forces `α R² / r + ε ≤ 0`; then `R = 0` or `r = 0` cases collapse
    rw [hD]
    simp only [_root_.inv_zero, zero_div, zero_mul, zero_smul]
    have h0 : α * R ^ 2 / r + eps ≤ 0 := Real.sqrt_eq_zero'.mp hD
    rcases Nat.eq_zero_or_pos r with hr0 | hr0
    · -- r = 0: `Wᵀ W = 0 = α I` forces `n = 0`
      subst hr0
      have : (α • (1 : Matrix (Fin n) (Fin n) ℝ)) = 0 := by
        rw [← hW]; ext a b; simp [Matrix.mul_apply]
      funext i
      simp only [Pi.zero_apply]
      symm
      simp only [Matrix.mulVec, dotProduct]
      refine Finset.sum_eq_zero (fun l _ => ?_)
      have := congrFun (congrFun this l) l
      simp [hα.ne'] at this
    · have hrR : (0 : ℝ) < r := by exact_mod_cast hr0
      have : α * R ^ 2 / r ≥ 0 := by positivity
      have hR0 : R ^ 2 = 0 := by
        have : α * R ^ 2 / r = 0 := by linarith
        rcases div_eq_zero_iff.mp this with h | h
        · exact (mul_eq_zero.mp h).resolve_left hα.ne'
        · exact absurd h hrR.ne'
      have hx0 : x = 0 := by
        funext l
        have := (Finset.sum_eq_zero_iff_of_nonneg (fun l _ => sq_nonneg (x l))).mp (hx.trans hR0) l
          (Finset.mem_univ _)
        simpa using this
      rw [hx0, Matrix.mulVec_zero]

end EP
