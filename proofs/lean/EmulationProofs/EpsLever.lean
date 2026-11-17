import EmulationProofs.LimitKit

/-!
# The eps lever: a host norm with a small input is a fixed linear map

A host RMSNorm keeps its own `ε > 0` (a config value), but its weights are free. Feed it `x / S`
(fold `1/S` into the writer of `x`) and multiply its output by `S √ε` (fold into the norm weight):

  `S √ε · N_ε(w, x/S) = w ⊙ x / √(1 + u)`,  `u = ‖x‖₂² / (n S² ε)`   (`epsLever_eq`, exact).

So the host norm is the fixed linear map `x ↦ w ⊙ x` up to the scalar `1/√(1+u) ∈ (0, 1]`, and
`|S√ε N_ε(w, x/S)_i − w_i x_i| ≤ |w_i x_i| u / 2` (`epsLever_coord_bound`), i.e. in the sup norm
`≤ ‖w‖ ‖x‖³ / (2 S² ε)` (`epsLever_bound`). It tends to `w ⊙ x` as `S → ∞`, uniformly on bounded
sets (`epsLever_tendsto`, `epsLever_uniform`) but NOT uniformly on all of `ℝⁿ` (for `‖x‖ ≫ S√ε`
the norm still normalises). `ε > 0` is needed: with `ε = 0` the norm is scale invariant.

Instances, "host norm switched off" (a host that has a norm where the target has none):

* attention with any of a q-norm, a k-norm, a v-norm (`leverAttnLayer`, flags `bq bk bv`); the
  MLA latent norm is the case where the key and value vectors are the same latent `c`
  (`kc = vv = lat`, flags `bk = bv = true`, `bq` for the q-latent norm). One layer is within
  `‖O‖ V (2 C_B Q K (Q² + K²) + V²/2) / (ε S²)` of the plain layer once
  `C_B Q K (Q² + K²) ≤ ε S²` (`leverAttn_layer_bound`), on states where the query / key / value
  vectors are bounded by `Q, K, V`;
* a sandwich post-norm on a sublayer output (`postNormLayer`): within `R³/(2 S² ε)` when the
  sublayer output is bounded by `R` (`postNorm_layer_bound`).

Whole model (`lever_model_bound`, `lever_greedy`, `lever_model_limit`, and the instances
`leverAttn_model_bound` / `_greedy` / `_model_limit`, `postNorm_model_bound` / `_greedy` /
`_model_limit`): logits within `Lh (Π L_i)(Σ C_i)/S²`; identical greedy generation once
`S² > 2 Lh (Π L_i)(Σ C_i)/δ`; uniform convergence as `S → ∞` on inputs whose target trajectory
keeps a fixed neighbourhood inside the bounded region.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## The per-vector lever -/

section lever

variable {d : ℕ}

/-- The host norm with the lever: input `x / S`, weight `S √ε · w`. -/
def epsLeverNorm (eps S : ℝ) (w x : Fin d → ℝ) : Fin d → ℝ :=
  fun i => S * √eps * rmsNorm eps w (fun l => x l / S) i

/-- `u = ‖x‖₂² / (d S² ε)`. -/
def leverU (eps S : ℝ) (x : Fin d → ℝ) : ℝ := (∑ l, x l ^ 2) / (d * S ^ 2 * eps)

/-- The scalar the levered norm multiplies by: `1/√(1 + u)`. -/
def leverFactor (eps S : ℝ) (x : Fin d → ℝ) : ℝ := 1 / √(1 + leverU eps S x)

lemma leverU_nonneg {eps : ℝ} (heps : 0 ≤ eps) (S : ℝ) (x : Fin d → ℝ) : 0 ≤ leverU eps S x := by
  unfold leverU; positivity

/-- **The eps lever, exact form.** `S √ε · N_ε(w, x/S) = w ⊙ x / √(1 + ‖x‖²/(d S² ε))`. -/
theorem epsLever_eq (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : S ≠ 0)
    (w x : Fin d → ℝ) (i : Fin d) :
    epsLeverNorm eps S w x i = leverFactor eps S x * (w i * x i) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hsum : (∑ l, (x l / S) ^ 2) / d + eps = eps * (1 + leverU eps S x) := by
    unfold leverU
    simp only [div_pow, ← Finset.sum_div]
    field_simp
    ring
  have hu := leverU_nonneg heps.le S x
  unfold epsLeverNorm rmsNorm leverFactor
  rw [hsum, Real.sqrt_mul heps.le]
  have h1 : 0 < √eps := Real.sqrt_pos.mpr heps
  have h2 : 0 < √(1 + leverU eps S x) := Real.sqrt_pos.mpr (by linarith)
  field_simp

lemma leverFactor_pos (eps S : ℝ) (heps : 0 ≤ eps) (x : Fin d → ℝ) : 0 < leverFactor eps S x := by
  have := leverU_nonneg heps S x
  unfold leverFactor
  exact div_pos one_pos (Real.sqrt_pos.mpr (by linarith))

lemma leverFactor_le_one (eps S : ℝ) (heps : 0 ≤ eps) (x : Fin d → ℝ) : leverFactor eps S x ≤ 1 := by
  have := leverU_nonneg heps S x
  unfold leverFactor
  rw [div_le_one (Real.sqrt_pos.mpr (by linarith))]
  exact Real.one_le_sqrt.mpr (by linarith)

/-- `1 − 1/√(1+u) ≤ u/2`. -/
lemma one_sub_leverFactor_le (eps S : ℝ) (heps : 0 ≤ eps) (x : Fin d → ℝ) :
    1 - leverFactor eps S x ≤ leverU eps S x / 2 := by
  have hu := leverU_nonneg heps S x
  unfold leverFactor
  set u := leverU eps S x
  set r := √(1 + u) with hr
  have hr1 : 1 ≤ r := Real.one_le_sqrt.mpr (by linarith)
  have hru : r ≤ 1 + u / 2 := by
    rw [hr, Real.sqrt_le_iff]; constructor
    · linarith
    · nlinarith [sq_nonneg u]
  have key : 2 - r ≤ 1 / r := by
    rw [le_div_iff₀ (by linarith)]; nlinarith [sq_nonneg (r - 1)]
  linarith

/-- `u ≤ ‖x‖²/(S² ε)` in the sup norm. -/
lemma leverU_le (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : S ≠ 0) (x : Fin d → ℝ) :
    leverU eps S x ≤ ‖x‖ ^ 2 / (S ^ 2 * eps) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hsq : ∑ l, x l ^ 2 ≤ d * ‖x‖ ^ 2 := by
    calc ∑ l, x l ^ 2 ≤ ∑ _l : Fin d, ‖x‖ ^ 2 := by
          refine Finset.sum_le_sum (fun l _ => ?_)
          rw [← sq_abs]
          exact pow_le_pow_left₀ (abs_nonneg _) (by
            have := norm_le_pi_norm x l; rwa [Real.norm_eq_abs] at this) 2
      _ = d * ‖x‖ ^ 2 := by simp
  have hS2 : 0 < S ^ 2 := by positivity
  unfold leverU
  calc (∑ l, x l ^ 2) / (d * S ^ 2 * eps) ≤ (d * ‖x‖ ^ 2) / (d * S ^ 2 * eps) :=
        div_le_div_of_nonneg_right hsq (by positivity)
    _ = ‖x‖ ^ 2 / (S ^ 2 * eps) := by field_simp

/-- **The eps lever, per coordinate.** `|S√ε N_ε(w, x/S)_i − w_i x_i| ≤ |w_i x_i| · u/2`. -/
theorem epsLever_coord_bound (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : S ≠ 0)
    (w x : Fin d → ℝ) (i : Fin d) :
    |epsLeverNorm eps S w x i - w i * x i| ≤ |w i * x i| * (leverU eps S x / 2) := by
  rw [epsLever_eq hd heps hS]
  have h1 := one_sub_leverFactor_le eps S heps.le x
  have h2 := leverFactor_le_one eps S heps.le x
  have : leverFactor eps S x * (w i * x i) - w i * x i = -((1 - leverFactor eps S x) * (w i * x i)) :=
    by ring
  rw [this, abs_neg, abs_mul, abs_of_nonneg (by linarith), mul_comm]
  exact mul_le_mul_of_nonneg_left h1 (abs_nonneg _)

/-- **The eps lever, sup norm.** `‖S√ε N_ε(w, x/S) − w ⊙ x‖ ≤ ‖w‖ ‖x‖³ / (2 S² ε)`. -/
theorem epsLever_bound (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : S ≠ 0)
    (w x : Fin d → ℝ) :
    ‖epsLeverNorm eps S w x - fun i => w i * x i‖ ≤ ‖w‖ * ‖x‖ ^ 3 / (2 * S ^ 2 * eps) := by
  have hS2 : 0 < S ^ 2 := by positivity
  refine (pi_norm_le_iff_of_nonneg (by positivity)).mpr (fun i => ?_)
  rw [Pi.sub_apply, Real.norm_eq_abs]
  refine (epsLever_coord_bound hd heps hS w x i).trans ?_
  have hu := leverU_le hd heps hS x
  have hwx : |w i * x i| ≤ ‖w‖ * ‖x‖ := by
    rw [abs_mul]
    have a := norm_le_pi_norm w i; have b := norm_le_pi_norm x i
    rw [Real.norm_eq_abs] at a b
    exact mul_le_mul a b (abs_nonneg _) (norm_nonneg _)
  calc |w i * x i| * (leverU eps S x / 2) ≤ (‖w‖ * ‖x‖) * (‖x‖ ^ 2 / (S ^ 2 * eps) / 2) :=
        mul_le_mul hwx (by linarith) (by linarith [leverU_nonneg heps.le S x]) (by positivity)
    _ = ‖w‖ * ‖x‖ ^ 3 / (2 * S ^ 2 * eps) := by field_simp

lemma tendsto_const_div_sq (c : ℝ) : Tendsto (fun S : ℝ => c / S ^ 2) atTop (𝓝 0) :=
  tendsto_const_nhds.div_atTop (tendsto_pow_atTop two_ne_zero)

/-- **The eps lever, the limit.** As `S → ∞` the levered norm tends to the linear map `w ⊙ x`. -/
theorem epsLever_tendsto (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) (w x : Fin d → ℝ) :
    Tendsto (fun S => epsLeverNorm eps S w x) atTop (𝓝 (fun i => w i * x i)) := by
  rw [tendsto_iff_norm_sub_tendsto_zero]
  refine squeeze_zero' (Eventually.of_forall (fun _ => norm_nonneg _)) ?_
    ((tendsto_const_div_sq (‖w‖ * ‖x‖ ^ 3 / (2 * eps))))
  filter_upwards [eventually_gt_atTop 0] with S hS
  refine (epsLever_bound hd heps hS.ne' w x).trans (le_of_eq ?_)
  field_simp

/-- **…uniformly on every ball** (`‖x‖ ≤ R`); not on all of `ℝⁿ`. -/
theorem epsLever_uniform (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) (w : Fin d → ℝ) (R : ℝ) :
    TendstoUniformlyOn (fun S x => epsLeverNorm eps S w x) (fun x i => w i * x i) atTop
      (Metric.closedBall 0 R) := by
  rw [Metric.tendstoUniformlyOn_iff]
  intro e he
  filter_upwards [eventually_gt_atTop 0,
    (tendsto_const_div_sq (‖w‖ * R ^ 3 / (2 * eps))).eventually (gt_mem_nhds he)] with S hS h x hx
  rw [mem_closedBall_zero_iff] at hx
  rw [dist_comm, dist_eq_norm]
  have hR : 0 ≤ R := (norm_nonneg _).trans hx
  refine (epsLever_bound hd heps hS.ne' w x).trans_lt (lt_of_le_of_lt ?_ h)
  rw [show ‖w‖ * R ^ 3 / (2 * eps) / S ^ 2 = ‖w‖ * R ^ 3 / (2 * S ^ 2 * eps) by field_simp]
  gcongr

/-! ### A switchable lever (weight 1) -/

/-- The host vector: levered (`b = true`, host norm present, weight `S√ε`) or untouched. -/
def optLever (b : Bool) (eps S : ℝ) (x : Fin d → ℝ) : Fin d → ℝ :=
  if b then epsLeverNorm eps S (fun _ => 1) x else x

/-- Its scalar factor. -/
def optFactor (b : Bool) (eps S : ℝ) (x : Fin d → ℝ) : ℝ := if b then leverFactor eps S x else 1

lemma optLever_eq_smul (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : S ≠ 0) (b : Bool)
    (x : Fin d → ℝ) : optLever b eps S x = optFactor b eps S x • x := by
  cases b
  · simp [optLever, optFactor]
  · funext i
    simp [optLever, optFactor, epsLever_eq hd heps hS]

lemma optFactor_pos (b : Bool) {eps : ℝ} (heps : 0 ≤ eps) (S : ℝ) (x : Fin d → ℝ) :
    0 < optFactor b eps S x := by
  cases b
  · simp [optFactor]
  · exact leverFactor_pos eps S heps x

lemma optFactor_le_one (b : Bool) {eps : ℝ} (heps : 0 ≤ eps) (S : ℝ) (x : Fin d → ℝ) :
    optFactor b eps S x ≤ 1 := by
  cases b
  · simp [optFactor]
  · exact leverFactor_le_one eps S heps x

lemma one_sub_optFactor_le (hd : d ≠ 0) (b : Bool) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : S ≠ 0)
    (x : Fin d → ℝ) : 1 - optFactor b eps S x ≤ ‖x‖ ^ 2 / (2 * S ^ 2 * eps) := by
  cases b
  · simp only [optFactor]; norm_num; positivity
  · have h1 := one_sub_leverFactor_le eps S heps.le x
    have := leverU_le hd heps hS x
    calc 1 - optFactor true eps S x ≤ leverU eps S x / 2 := h1
      _ ≤ ‖x‖ ^ 2 / (S ^ 2 * eps) / 2 := by linarith
      _ = _ := by field_simp

end lever

/-! ## Attention with a convex combination: a value shift moves the output by at most the shift -/

lemma epsLever_attn_value {ι V : Type*} [Fintype ι] [NormedAddCommGroup V] [NormedSpace ℝ V]
    (S : Finset ι) (hS : S.Nonempty) (s : ι → ℝ) (v v' : ι → V) (η : ℝ)
    (h : ∀ j ∈ S, ‖v' j - v j‖ ≤ η) : ‖attn S s v' - attn S s v‖ ≤ η := by
  have hw : ∑ j ∈ S, exp (s j) / ∑ l ∈ S, exp (s l) = 1 := by
    rw [← Finset.sum_div]; exact div_self (sum_exp_pos hS s).ne'
  rw [attn_eq_weights, attn_eq_weights, ← Finset.sum_sub_distrib]
  calc ‖∑ j ∈ S, ((exp (s j) / ∑ l ∈ S, exp (s l)) • v' j - (exp (s j) / ∑ l ∈ S, exp (s l)) • v j)‖
      ≤ ∑ j ∈ S, ‖(exp (s j) / ∑ l ∈ S, exp (s l)) • (v' j - v j)‖ := by
        refine (norm_sum_le _ _).trans (le_of_eq (Finset.sum_congr rfl (fun j _ => ?_)))
        rw [smul_sub]
    _ ≤ ∑ j ∈ S, (exp (s j) / ∑ l ∈ S, exp (s l)) * η := by
        refine Finset.sum_le_sum (fun j hj => ?_)
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
        exact mul_le_mul_of_nonneg_left (h j hj) (by positivity)
    _ = η := by rw [← Finset.sum_mul, hw, one_mul]

/-! ## Instance: q-norm / k-norm / v-norm / MLA latent norm switched off -/

section attn

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {T dq dk dv : ℕ}

/-- The target: one plain attention head per layer (output projection `Ov`), key sets `ks t`
(causal, windowed, anything nonempty), logits `B t j (q_t, k_j) + ex t j` with `B t j` bilinear
(a dot product, RoPE, the attention scale, the MLA absorbed `W_UK`) and `ex` any extra term
(a decoupled RoPE part, a bias), values `Ov (v_j)`, plus the residual. -/
def plainAttnLayer (ks : Fin T → Finset (Fin T))
    (B : Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ) (ex : Fin T → Fin T → V → V → ℝ)
    (qa : V → Fin dq → ℝ) (kc : V → Fin dk → ℝ) (vv : V → Fin dv → ℝ) (Ov : (Fin dv → ℝ) →L[ℝ] V)
    (x : Fin T → V) : Fin T → V :=
  fun t => x t + attn (ks t) (fun j => B t j (qa (x t)) (kc (x j)) + ex t j (x t) (x j))
    (fun j => Ov (vv (x j)))

/-- The host: the same head with a norm on the query vector (`bq`), the key vector (`bk`), the
value vector (`bv`), each levered (`optLever`). qk-norm: `bq = bk = true`. v-norm: `bv = true`.
MLA latent norm: `kc = vv = lat` (the latent `c_KV`, normed once and read by both `W_UK` and
`W_UV`), `bk = bv = true`, and `bq = true` for the q-latent norm (`q_lora_rank`). -/
def leverAttnLayer (bq bk bv : Bool) (eps S : ℝ) (ks : Fin T → Finset (Fin T))
    (B : Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ) (ex : Fin T → Fin T → V → V → ℝ)
    (qa : V → Fin dq → ℝ) (kc : V → Fin dk → ℝ) (vv : V → Fin dv → ℝ) (Ov : (Fin dv → ℝ) →L[ℝ] V)
    (x : Fin T → V) : Fin T → V :=
  fun t => x t + attn (ks t) (fun j => B t j (optLever bq eps S (qa (x t)))
      (optLever bk eps S (kc (x j))) + ex t j (x t) (x j))
    (fun j => Ov (optLever bv eps S (vv (x j))))

/-- The bounded region: every query, key and value vector bounded by `Q, K, Vb`. -/
def leverRegion (qa : V → Fin dq → ℝ) (kc : V → Fin dk → ℝ) (vv : V → Fin dv → ℝ)
    (Q K Vb : ℝ) : Set (Fin T → V) :=
  {x | ∀ t, ‖qa (x t)‖ ≤ Q ∧ ‖kc (x t)‖ ≤ K ∧ ‖vv (x t)‖ ≤ Vb}

/-- The per-layer constant: `‖O‖ Vb (2 C_B Q K (Q² + K²) + Vb²/2) / ε`. -/
def leverAttnConst (eps Ob Cb Q K Vb : ℝ) : ℝ := Ob * Vb * (2 * Cb * Q * K * (Q ^ 2 + K ^ 2) + Vb ^ 2 / 2) / eps

/-- **qk-norm / v-norm / MLA latent norm switched off, one layer.** On the bounded region, with
`‖B t j‖ ≤ C_B`, `‖Ov‖ ≤ O_b`, and `S` past the threshold `C_B Q K (Q² + K²) ≤ ε S²`, the host
layer is within `leverAttnConst/S² = O_b Vb (2 C_B Q K (Q² + K²) + Vb²/2)/(ε S²)` of the plain
layer. (The logits move by at most `η = C_B Q K (Q²+K²)/(2 S² ε)`, since the levered logit is the
plain one times `γ_q γ_k ∈ (0, 1]`; the values move by at most `O_b Vb³/(2 S² ε)`.) -/
theorem leverAttn_layer_bound (hdq : dq ≠ 0) (hdk : dk ≠ 0) (hdv : dv ≠ 0) (bq bk bv : Bool)
    {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : 0 < S) (ks : Fin T → Finset (Fin T))
    (hks : ∀ t, (ks t).Nonempty)
    (B : Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ) (ex : Fin T → Fin T → V → V → ℝ)
    (qa : V → Fin dq → ℝ) (kc : V → Fin dk → ℝ) (vv : V → Fin dv → ℝ) (Ov : (Fin dv → ℝ) →L[ℝ] V)
    (Ob Cb Q K Vb : ℝ) (hOb : ‖Ov‖ ≤ Ob) (hCb0 : 0 ≤ Cb) (hQ0 : 0 ≤ Q) (hK0 : 0 ≤ K)
    (hV0 : 0 ≤ Vb) (hB : ∀ t j, ‖B t j‖ ≤ Cb)
    (hth : Cb * Q * K * (Q ^ 2 + K ^ 2) ≤ eps * S ^ 2) (x : Fin T → V)
    (hx : x ∈ leverRegion qa kc vv Q K Vb) :
    dist (leverAttnLayer bq bk bv eps S ks B ex qa kc vv Ov x) (plainAttnLayer ks B ex qa kc vv Ov x)
      ≤ leverAttnConst eps Ob Cb Q K Vb / S ^ 2 := by
  have hS2 : 0 < S ^ 2 := by positivity
  have hO0 : 0 ≤ Ob := (norm_nonneg _).trans hOb
  set η := Cb * Q * K * (Q ^ 2 + K ^ 2) / (2 * S ^ 2 * eps) with hηdef
  have hη0 : 0 ≤ η := by positivity
  have hη1 : 2 * η ≤ 1 := by
    rw [hηdef]
    rw [show 2 * (Cb * Q * K * (Q ^ 2 + K ^ 2) / (2 * S ^ 2 * eps)) =
      Cb * Q * K * (Q ^ 2 + K ^ 2) / (eps * S ^ 2) by field_simp]
    rw [div_le_one (by positivity)]; exact hth
  have hC0 : 0 ≤ leverAttnConst eps Ob Cb Q K Vb / S ^ 2 := by
    unfold leverAttnConst; positivity
  refine (dist_pi_le_iff hC0).mpr (fun t => ?_)
  rw [dist_eq_norm, leverAttnLayer, plainAttnLayer, add_sub_add_left_eq_sub]
  obtain ⟨hqt, -, -⟩ := hx t
  -- the logits
  have hlog : ∀ j ∈ ks t, |(B t j (optLever bq eps S (qa (x t))) (optLever bk eps S (kc (x j))) +
      ex t j (x t) (x j)) - (B t j (qa (x t)) (kc (x j)) + ex t j (x t) (x j))| ≤ η := by
    intro j _
    obtain ⟨-, hkj, -⟩ := hx j
    rw [optLever_eq_smul hdq heps hS.ne', optLever_eq_smul hdk heps hS.ne']
    simp only [map_smul, FunLike.coe_smul, Pi.smul_apply, smul_eq_mul, add_sub_add_right_eq_sub]
    have hq1 := optFactor_le_one bq heps.le S (qa (x t))
    have hk1 := optFactor_le_one bk heps.le S (kc (x j))
    have hq0 := optFactor_pos bq heps.le S (qa (x t))
    have hk0 := optFactor_pos bk heps.le S (kc (x j))
    have hqs := one_sub_optFactor_le hdq bq heps hS.ne' (qa (x t))
    have hks' := one_sub_optFactor_le hdk bk heps hS.ne' (kc (x j))
    set gq := optFactor bq eps S (qa (x t))
    set gk := optFactor bk eps S (kc (x j))
    have hQ2 : ‖qa (x t)‖ ^ 2 ≤ Q ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hqt 2
    have hK2 : ‖kc (x j)‖ ^ 2 ≤ K ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hkj 2
    have hfac : |gq * gk - 1| ≤ (Q ^ 2 + K ^ 2) / (2 * S ^ 2 * eps) := by
      rw [abs_of_nonpos (by nlinarith)]
      have e1 : ‖qa (x t)‖ ^ 2 / (2 * S ^ 2 * eps) ≤ Q ^ 2 / (2 * S ^ 2 * eps) :=
        div_le_div_of_nonneg_right hQ2 (by positivity)
      have e2 : ‖kc (x j)‖ ^ 2 / (2 * S ^ 2 * eps) ≤ K ^ 2 / (2 * S ^ 2 * eps) :=
        div_le_div_of_nonneg_right hK2 (by positivity)
      rw [add_div]; nlinarith
    have hBv : |B t j (qa (x t)) (kc (x j))| ≤ Cb * Q * K := by
      rw [← Real.norm_eq_abs]
      refine ((B t j).le_opNorm₂ _ _).trans ?_
      have := hB t j
      gcongr
    rw [show gk * (gq * B t j (qa (x t)) (kc (x j))) - B t j (qa (x t)) (kc (x j)) =
      (gq * gk - 1) * B t j (qa (x t)) (kc (x j)) by ring, abs_mul]
    calc |gq * gk - 1| * |B t j (qa (x t)) (kc (x j))|
        ≤ (Q ^ 2 + K ^ 2) / (2 * S ^ 2 * eps) * (Cb * Q * K) :=
          mul_le_mul hfac hBv (abs_nonneg _) (by positivity)
      _ = η := by rw [hηdef]; ring
  -- the values
  have hval : ∀ j ∈ ks t, ‖Ov (optLever bv eps S (vv (x j))) - Ov (vv (x j))‖ ≤
      Ob * Vb ^ 3 / (2 * S ^ 2 * eps) := by
    intro j _
    obtain ⟨-, -, hvj⟩ := hx j
    rw [optLever_eq_smul hdv heps hS.ne', map_smul]
    set g := optFactor bv eps S (vv (x j))
    have hg1 : g ≤ 1 := optFactor_le_one bv heps.le S (vv (x j))
    have hgs : 1 - g ≤ ‖vv (x j)‖ ^ 2 / (2 * S ^ 2 * eps) :=
      one_sub_optFactor_le hdv bv heps hS.ne' (vv (x j))
    rw [show g • Ov (vv (x j)) - Ov (vv (x j)) = -((1 - g) • Ov (vv (x j))) by
      rw [sub_smul, one_smul]; abel, norm_neg, norm_smul, Real.norm_eq_abs,
      abs_of_nonneg (by linarith)]
    have hOv : ‖Ov (vv (x j))‖ ≤ Ob * Vb :=
      (Ov.le_opNorm _).trans (mul_le_mul hOb hvj (norm_nonneg _) hO0)
    have hv3 : ‖vv (x j)‖ ^ 2 / (2 * S ^ 2 * eps) ≤ Vb ^ 2 / (2 * S ^ 2 * eps) :=
      div_le_div_of_nonneg_right (pow_le_pow_left₀ (norm_nonneg _) hvj 2) (by positivity)
    calc (1 - g) * ‖Ov (vv (x j))‖ ≤ Vb ^ 2 / (2 * S ^ 2 * eps) * (Ob * Vb) :=
          mul_le_mul (hgs.trans hv3) hOv (norm_nonneg _) (by positivity)
      _ = Ob * Vb ^ 3 / (2 * S ^ 2 * eps) := by ring
  have hM : ∀ j ∈ ks t, ‖Ov (vv (x j))‖ ≤ Ob * Vb := fun j _ =>
    (Ov.le_opNorm _).trans (mul_le_mul hOb (hx j).2.2 (norm_nonneg _) hO0)
  set s' := fun j => B t j (optLever bq eps S (qa (x t))) (optLever bk eps S (kc (x j))) +
      ex t j (x t) (x j)
  set s0 := fun j => B t j (qa (x t)) (kc (x j)) + ex t j (x t) (x j)
  set v' := fun j => Ov (optLever bv eps S (vv (x j)))
  set v0 := fun j => Ov (vv (x j))
  have h1 := epsLever_attn_value (ks t) (hks t) s' v0 v' _ hval
  have h2 := attn_perturb (ks t) (hks t) s' s0 η hlog v0 (Ob * Vb) hM
  have h3 := exp_two_mul_sub_one_le η hη0 hη1
  calc ‖attn (ks t) s' v' - attn (ks t) s0 v0‖
      ≤ ‖attn (ks t) s' v' - attn (ks t) s' v0‖ + ‖attn (ks t) s' v0 - attn (ks t) s0 v0‖ :=
        norm_sub_le_norm_sub_add_norm_sub _ _ _
    _ ≤ Ob * Vb ^ 3 / (2 * S ^ 2 * eps) + 4 * η * (Ob * Vb) := by
        have : (exp (2 * η) - 1) * (Ob * Vb) ≤ 4 * η * (Ob * Vb) :=
          mul_le_mul_of_nonneg_right h3 (by positivity)
        linarith
    _ = leverAttnConst eps Ob Cb Q K Vb / S ^ 2 := by
        rw [hηdef]; unfold leverAttnConst; field_simp; ring

end attn

/-! ## Instance: a sandwich post-norm on a sublayer output switched off -/

section postnorm

variable {T D : ℕ}

/-- The target: a residual sublayer `x + y(x)` (`y` any sublayer: attention, MLP, anything). -/
def plainResLayer (y : (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ) (x : Fin T → Fin D → ℝ) :
    Fin T → Fin D → ℝ :=
  fun t => x t + y x t

/-- The host: the same sublayer with a post-norm on its output, per token, levered (the sublayer
writes `y/S`, the post-norm weight is `S√ε`). -/
def postNormLayer (eps S : ℝ) (y : (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ)
    (x : Fin T → Fin D → ℝ) : Fin T → Fin D → ℝ :=
  fun t => x t + epsLeverNorm eps S (fun _ => 1) (y x t)

/-- **Post-norm switched off, one layer.** If the sublayer output is bounded by `R` at every
token of the state, the host is within `R³/(2 ε S²)`. -/
theorem postNorm_layer_bound (hD : D ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : 0 < S)
    (y : (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ) (R : ℝ) (hR0 : 0 ≤ R) (x : Fin T → Fin D → ℝ)
    (hy : ∀ t, ‖y x t‖ ≤ R) :
    dist (postNormLayer eps S y x) (plainResLayer y x) ≤ R ^ 3 / (2 * eps) / S ^ 2 := by
  refine (dist_pi_le_iff (by positivity)).mpr (fun t => ?_)
  rw [dist_eq_norm, postNormLayer, plainResLayer, add_sub_add_left_eq_sub]
  have h := epsLever_bound hD heps hS.ne' (fun _ => 1) (y x t)
  have hw : ‖(fun _ => (1 : ℝ) : Fin D → ℝ)‖ ≤ 1 :=
    (pi_norm_le_iff_of_nonneg zero_le_one).mpr (fun _ => by simp)
  have he : (fun i => (fun _ => (1 : ℝ) : Fin D → ℝ) i * y x t i) = y x t := by funext i; simp
  rw [he] at h
  refine h.trans ?_
  have h3 : ‖y x t‖ ^ 3 ≤ R ^ 3 := pow_le_pow_left₀ (norm_nonneg _) (hy t) 3
  calc ‖(fun _ => (1 : ℝ) : Fin D → ℝ)‖ * ‖y x t‖ ^ 3 / (2 * S ^ 2 * eps)
      ≤ 1 * R ^ 3 / (2 * S ^ 2 * eps) := by gcongr
    _ = R ^ 3 / (2 * eps) / S ^ 2 := by field_simp

end postnorm

/-! ## Whole model: per-layer errors `C i / S²` -/

section stack

variable {X : Type*} [PseudoMetricSpace X]

lemma lever_sum_div (L C : ℕ → ℝ) (S : ℝ) (n : ℕ) :
    (∏ i ∈ range n, L i) * ∑ i ∈ range n, C i / S ^ 2 =
      (∏ i ∈ range n, L i) * (∑ i ∈ range n, C i) / S ^ 2 := by
  rw [← Finset.sum_div]; ring

/-- **The lever, whole model, the bound.** Host layer `i` (with lever `S`) is within `C i / S²` of
target layer `i` on the region `K i` once `S² ≥ A`; target layers `L i`-Lipschitz on `K i`; head
`Lh`-Lipschitz on `K n`; the `(Π L)(Σ C)/S²`-neighbourhood of the target trajectory stays in the
region. Then the logits are within `Lh (Π L i)(Σ C i)/S²`. -/
theorem lever_model_bound {Y : Type*} [PseudoMetricSpace Y] (F : ℝ → ℕ → X → X) (f : ℕ → X → X)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i) (C : ℕ → ℝ) (hC0 : ∀ i, 0 ≤ C i) (A : ℝ) (K : ℕ → Set X)
    (hL : ∀ i, ∀ a ∈ K i, ∀ b ∈ K i, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ S, 0 < S → A ≤ S ^ 2 → ∀ i, ∀ y ∈ K i, dist (F S i y) (f i y) ≤ C i / S ^ 2)
    (head : X → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ K n, ∀ b ∈ K n, dist (head a) (head b) ≤ Lh * dist a b)
    (S : ℝ) (hS : 0 < S) (hA : A ≤ S ^ 2) (x : X)
    (hK : ∀ i ≤ n, ∀ y, dist y (run f i x) ≤ (∏ j ∈ range n, L j) * (∑ j ∈ range n, C j) / S ^ 2 →
      y ∈ K i) :
    dist (head (run (F S) n x)) (head (run f n x)) ≤
      Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, C i) / S ^ 2 := by
  have h := model_bound_on_prod (F S) f L (fun i => C i / S ^ 2) hL1
    (fun i => div_nonneg (hC0 i) (sq_nonneg S)) K hL (hε S hS hA) head Lh hLh0 n hLh x
    (fun i hi y hy => hK i hi y (by rwa [lever_sum_div] at hy))
  rw [lever_sum_div] at h
  refine h.trans (le_of_eq ?_); ring

/-- **The lever, whole model, identical generation.** With the target's greedy margin `δ` along
its trajectory, every `S > 0` with `S² ≥ A`, `(Π L)(Σ C) ≤ r S²` (the region radius `r`) and
`S² > 2 Lh (Π L i)(Σ C i)/δ` generates exactly the target's tokens. -/
theorem lever_greedy {Tok : Type*} [Fintype Tok] (F : ℝ → ℕ → X → X) (f : ℕ → X → X)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i) (C : ℕ → ℝ) (hC0 : ∀ i, 0 ≤ C i) (A : ℝ) (K : ℕ → Set X)
    (hL : ∀ i, ∀ a ∈ K i, ∀ b ∈ K i, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ S, 0 < S → A ≤ S ^ 2 → ∀ i, ∀ y ∈ K i, dist (F S i y) (f i y) ≤ C i / S ^ 2)
    (head : X → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ K n, ∀ b ∈ K n, dist (head a) (head b) ≤ Lh * dist a b) (embed : List Tok → X)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ r : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run f n (embed (greedy pickT
      (fun q => head (run f n (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run f i (embed (greedy pickT
      (fun q => head (run f n (embed q))) p k))) ≤ r → y ∈ K i)
    (S : ℝ) (hS : 0 < S) (hA : A ≤ S ^ 2)
    (hr : (∏ i ∈ range n, L i) * (∑ i ∈ range n, C i) ≤ r * S ^ 2)
    (hsmall : 2 * Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, C i) < δ * S ^ 2) :
    ∀ k, greedy pickH (fun q => head (run (F S) n (embed q))) p k =
      greedy pickT (fun q => head (run f n (embed q))) p k := by
  have hS2 : 0 < S ^ 2 := by positivity
  refine greedy_of_layer_bounds_on (F S) f L (fun i => C i / S ^ 2) hL1
    (fun i => div_nonneg (hC0 i) (sq_nonneg S)) K hL (hε S hS hA) head Lh hLh0 n hLh embed hT hH
    p δ hm (fun k i hi y hy => hK k i hi y (hy.trans ?_)) ?_
  · rw [lever_sum_div, div_le_iff₀ hS2]; exact hr
  · rw [lever_sum_div]
    calc 2 * (Lh * ((∏ i ∈ range n, L i) * (∑ i ∈ range n, C i) / S ^ 2))
        = (2 * Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, C i)) / S ^ 2 := by ring
      _ < δ := by rw [div_lt_iff₀ hS2]; exact hsmall

/-- **The lever, whole model, the limit.** Uniform convergence as `S → ∞` on every set `Aset` of
inputs whose target trajectory keeps an `r`-neighbourhood (`r > 0`) inside the region. -/
theorem lever_model_limit (F : ℝ → ℕ → X → X) (f : ℕ → X → X)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i) (C : ℕ → ℝ) (hC0 : ∀ i, 0 ≤ C i) (A : ℝ) (K : ℕ → Set X)
    (hL : ∀ i, ∀ a ∈ K i, ∀ b ∈ K i, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ S, 0 < S → A ≤ S ^ 2 → ∀ i, ∀ y ∈ K i, dist (F S i y) (f i y) ≤ C i / S ^ 2)
    (n : ℕ) (Aset : Set X) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ Aset, ∀ i ≤ n, ∀ y, dist y (run f i x) ≤ r → y ∈ K i) :
    TendstoUniformlyOn (fun S => run (F S) n) (run f n) atTop Aset := by
  rw [Metric.tendstoUniformlyOn_iff]
  intro e he
  set c := (∏ i ∈ range n, L i) * (∑ i ∈ range n, C i)
  filter_upwards [eventually_gt_atTop 0, (tendsto_pow_atTop (two_ne_zero)).eventually_ge_atTop A,
    (tendsto_const_div_sq c).eventually (gt_mem_nhds (lt_min he hr))] with S hS hA hc x hx
  rw [dist_comm]
  have hc' : c / S ^ 2 < min e r := hc
  have h := lever_model_bound F f L hL1 C hC0 A K hL hε id 1 zero_le_one n
    (fun a _ b _ => by simp) S hS hA x
    (fun i hi y hy => hK x hx i hi y (hy.trans (hc'.le.trans (min_le_right _ _))))
  simp only [id, one_mul] at h
  exact h.trans_lt (hc'.trans_le (min_le_left _ _))

end stack

/-! ## The instances, whole model -/

section attn_stack

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {T dq dk dv : ℕ}

/-- Host layer `i`: the levered head, then any map `post i` (the MLP sub-block, unchanged). -/
def leverAttnStack (bq bk bv : Bool) (eps : ℝ) (ks : ℕ → Fin T → Finset (Fin T))
    (B : ℕ → Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ)
    (ex : ℕ → Fin T → Fin T → V → V → ℝ) (qa : ℕ → V → Fin dq → ℝ) (kc : ℕ → V → Fin dk → ℝ)
    (vv : ℕ → V → Fin dv → ℝ) (Ov : ℕ → (Fin dv → ℝ) →L[ℝ] V) (post : ℕ → (Fin T → V) → Fin T → V)
    (S : ℝ) (i : ℕ) (x : Fin T → V) : Fin T → V :=
  post i (leverAttnLayer bq bk bv eps S (ks i) (B i) (ex i) (qa i) (kc i) (vv i) (Ov i) x)

/-- Target layer `i`: the plain head, then `post i`. -/
def plainAttnStack (ks : ℕ → Fin T → Finset (Fin T))
    (B : ℕ → Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ)
    (ex : ℕ → Fin T → Fin T → V → V → ℝ) (qa : ℕ → V → Fin dq → ℝ) (kc : ℕ → V → Fin dk → ℝ)
    (vv : ℕ → V → Fin dv → ℝ) (Ov : ℕ → (Fin dv → ℝ) →L[ℝ] V) (post : ℕ → (Fin T → V) → Fin T → V)
    (i : ℕ) (x : Fin T → V) : Fin T → V :=
  post i (plainAttnLayer (ks i) (B i) (ex i) (qa i) (kc i) (vv i) (Ov i) x)

lemma leverAttnConst_nonneg {eps Ob Cb Q K Vb : ℝ} (heps : 0 < eps) (hO : 0 ≤ Ob) (hCb0 : 0 ≤ Cb)
    (hQ0 : 0 ≤ Q) (hK0 : 0 ≤ K) (hV0 : 0 ≤ Vb) : 0 ≤ leverAttnConst eps Ob Cb Q K Vb := by
  unfold leverAttnConst; positivity

/-- The per-layer error of the stack: `P i · leverAttnConst / S²` (`post i` is `P i`-Lipschitz). -/
theorem leverAttnStack_layer (hdq : dq ≠ 0) (hdk : dk ≠ 0) (hdv : dv ≠ 0) (bq bk bv : Bool)
    {eps : ℝ} (heps : 0 < eps) (ks : ℕ → Fin T → Finset (Fin T)) (hks : ∀ i t, (ks i t).Nonempty)
    (B : ℕ → Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ)
    (ex : ℕ → Fin T → Fin T → V → V → ℝ) (qa : ℕ → V → Fin dq → ℝ) (kc : ℕ → V → Fin dk → ℝ)
    (vv : ℕ → V → Fin dv → ℝ) (Ov : ℕ → (Fin dv → ℝ) →L[ℝ] V) (post : ℕ → (Fin T → V) → Fin T → V)
    (P : ℕ → ℝ) (hP0 : ∀ i, 0 ≤ P i) (hP : ∀ i a b, dist (post i a) (post i b) ≤ P i * dist a b)
    (Ob Cb Q K Vb : ℝ) (hOb : ∀ i, ‖Ov i‖ ≤ Ob) (hCb0 : 0 ≤ Cb) (hQ0 : 0 ≤ Q) (hK0 : 0 ≤ K)
    (hV0 : 0 ≤ Vb) (hB : ∀ i t j, ‖B i t j‖ ≤ Cb) :
    ∀ S, 0 < S → Cb * Q * K * (Q ^ 2 + K ^ 2) / eps ≤ S ^ 2 → ∀ i,
      ∀ y ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb,
        dist (leverAttnStack bq bk bv eps ks B ex qa kc vv Ov post S i y)
          (plainAttnStack ks B ex qa kc vv Ov post i y) ≤
          P i * leverAttnConst eps Ob Cb Q K Vb / S ^ 2 := by
  intro S hS hA i y hy
  have hth : Cb * Q * K * (Q ^ 2 + K ^ 2) ≤ eps * S ^ 2 := by
    rw [div_le_iff₀ heps] at hA; linarith
  refine (hP i _ _).trans ?_
  rw [mul_div_assoc]
  exact mul_le_mul_of_nonneg_left (leverAttn_layer_bound hdq hdk hdv bq bk bv heps hS (ks i)
    (hks i) (B i) (ex i) (qa i) (kc i) (vv i) (Ov i) Ob Cb Q K Vb (hOb i) hCb0 hQ0 hK0 hV0 (hB i)
    hth y hy) (hP0 i)

/-- **qk-norm / v-norm / MLA latent norm switched off, whole model, the bound.**
`Lh (Π L i)(Σ P i) · leverAttnConst / S²`, for `S² ≥ C_B Q K (Q²+K²)/ε`. -/
theorem leverAttn_model_bound {Y : Type*} [PseudoMetricSpace Y] (hdq : dq ≠ 0) (hdk : dk ≠ 0)
    (hdv : dv ≠ 0) (bq bk bv : Bool) {eps : ℝ} (heps : 0 < eps) (ks : ℕ → Fin T → Finset (Fin T))
    (hks : ∀ i t, (ks i t).Nonempty)
    (B : ℕ → Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ)
    (ex : ℕ → Fin T → Fin T → V → V → ℝ) (qa : ℕ → V → Fin dq → ℝ) (kc : ℕ → V → Fin dk → ℝ)
    (vv : ℕ → V → Fin dv → ℝ) (Ov : ℕ → (Fin dv → ℝ) →L[ℝ] V) (post : ℕ → (Fin T → V) → Fin T → V)
    (P : ℕ → ℝ) (hP0 : ∀ i, 0 ≤ P i) (hP : ∀ i a b, dist (post i a) (post i b) ≤ P i * dist a b)
    (Ob Cb Q K Vb : ℝ) (hOb : ∀ i, ‖Ov i‖ ≤ Ob) (hCb0 : 0 ≤ Cb) (hQ0 : 0 ≤ Q) (hK0 : 0 ≤ K)
    (hV0 : 0 ≤ Vb) (hB : ∀ i t j, ‖B i t j‖ ≤ Cb)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb,
      ∀ b ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb,
      dist (plainAttnStack ks B ex qa kc vv Ov post i a) (plainAttnStack ks B ex qa kc vv Ov post i b)
        ≤ L i * dist a b)
    (head : (Fin T → V) → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ leverRegion (T := T) (qa n) (kc n) (vv n) Q K Vb,
      ∀ b ∈ leverRegion (T := T) (qa n) (kc n) (vv n) Q K Vb, dist (head a) (head b) ≤ Lh * dist a b)
    (S : ℝ) (hS : 0 < S) (hA : Cb * Q * K * (Q ^ 2 + K ^ 2) / eps ≤ S ^ 2) (x : Fin T → V)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (plainAttnStack ks B ex qa kc vv Ov post) i x) ≤
      (∏ j ∈ range n, L j) * (∑ j ∈ range n, P j * leverAttnConst eps Ob Cb Q K Vb) / S ^ 2 →
      y ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb) :
    dist (head (run (leverAttnStack bq bk bv eps ks B ex qa kc vv Ov post S) n x))
        (head (run (plainAttnStack ks B ex qa kc vv Ov post) n x)) ≤
      Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, P i * leverAttnConst eps Ob Cb Q K Vb) / S ^ 2 :=
  lever_model_bound _ _ L hL1 (fun i => P i * leverAttnConst eps Ob Cb Q K Vb)
    (fun i => mul_nonneg (hP0 i) (leverAttnConst_nonneg heps ((norm_nonneg _).trans (hOb 0)) hCb0 hQ0
      hK0 hV0))
    _ _ hL (leverAttnStack_layer hdq hdk hdv bq bk bv heps ks hks B ex qa kc vv Ov post P hP0 hP
      Ob Cb Q K Vb hOb hCb0 hQ0 hK0 hV0 hB) head Lh hLh0 n hLh S hS hA x hK

/-- **qk-norm / v-norm / MLA latent norm switched off, identical generation, explicit threshold.**
Every `S` with `S² ≥ C_B Q K (Q²+K²)/ε`, `S² ≥ (Π L)(Σ P)·leverAttnConst / r` and
`S² > 2 Lh (Π L i)(Σ P i)·leverAttnConst / δ` generates exactly the target's tokens. -/
theorem leverAttn_greedy {Tok : Type*} [Fintype Tok] (hdq : dq ≠ 0) (hdk : dk ≠ 0)
    (hdv : dv ≠ 0) (bq bk bv : Bool) {eps : ℝ} (heps : 0 < eps) (ks : ℕ → Fin T → Finset (Fin T))
    (hks : ∀ i t, (ks i t).Nonempty)
    (B : ℕ → Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ)
    (ex : ℕ → Fin T → Fin T → V → V → ℝ) (qa : ℕ → V → Fin dq → ℝ) (kc : ℕ → V → Fin dk → ℝ)
    (vv : ℕ → V → Fin dv → ℝ) (Ov : ℕ → (Fin dv → ℝ) →L[ℝ] V) (post : ℕ → (Fin T → V) → Fin T → V)
    (P : ℕ → ℝ) (hP0 : ∀ i, 0 ≤ P i) (hP : ∀ i a b, dist (post i a) (post i b) ≤ P i * dist a b)
    (Ob Cb Q K Vb : ℝ) (hOb : ∀ i, ‖Ov i‖ ≤ Ob) (hCb0 : 0 ≤ Cb) (hQ0 : 0 ≤ Q) (hK0 : 0 ≤ K)
    (hV0 : 0 ≤ Vb) (hB : ∀ i t j, ‖B i t j‖ ≤ Cb)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb,
      ∀ b ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb,
      dist (plainAttnStack ks B ex qa kc vv Ov post i a) (plainAttnStack ks B ex qa kc vv Ov post i b)
        ≤ L i * dist a b)
    (head : (Fin T → V) → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ leverRegion (T := T) (qa n) (kc n) (vv n) Q K Vb,
      ∀ b ∈ leverRegion (T := T) (qa n) (kc n) (vv n) Q K Vb, dist (head a) (head b) ≤ Lh * dist a b)
    (embed : List Tok → Fin T → V) {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT)
    (hH : IsPick pickH) (p : List Tok) (δ r : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (plainAttnStack ks B ex qa kc vv Ov post) n
      (embed (greedy pickT (fun q => head (run (plainAttnStack ks B ex qa kc vv Ov post) n
        (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run (plainAttnStack ks B ex qa kc vv Ov post) i
      (embed (greedy pickT (fun q => head (run (plainAttnStack ks B ex qa kc vv Ov post) n
        (embed q))) p k))) ≤ r → y ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb)
    (S : ℝ) (hS : 0 < S) (hA : Cb * Q * K * (Q ^ 2 + K ^ 2) / eps ≤ S ^ 2)
    (hr : (∏ i ∈ range n, L i) * (∑ i ∈ range n, P i * leverAttnConst eps Ob Cb Q K Vb) ≤ r * S ^ 2)
    (hsmall : 2 * Lh * (∏ i ∈ range n, L i) *
      (∑ i ∈ range n, P i * leverAttnConst eps Ob Cb Q K Vb) < δ * S ^ 2) :
    ∀ k, greedy pickH (fun q => head (run (leverAttnStack bq bk bv eps ks B ex qa kc vv Ov post S) n
        (embed q))) p k =
      greedy pickT (fun q => head (run (plainAttnStack ks B ex qa kc vv Ov post) n (embed q))) p k :=
  lever_greedy _ _ L hL1 (fun i => P i * leverAttnConst eps Ob Cb Q K Vb)
    (fun i => mul_nonneg (hP0 i) (leverAttnConst_nonneg heps ((norm_nonneg _).trans (hOb 0)) hCb0 hQ0
      hK0 hV0))
    _ _ hL (leverAttnStack_layer hdq hdk hdv bq bk bv heps ks hks B ex qa kc vv Ov post P hP0 hP
      Ob Cb Q K Vb hOb hCb0 hQ0 hK0 hV0 hB) head Lh hLh0 n hLh embed hT hH p δ r hm hK S hS hA hr
    hsmall

/-- **qk-norm / v-norm / MLA latent norm switched off, whole model, the limit** (`S → ∞`),
uniformly on inputs whose target trajectory keeps an `r`-neighbourhood in the bounded region. -/
theorem leverAttn_model_limit (hdq : dq ≠ 0) (hdk : dk ≠ 0)
    (hdv : dv ≠ 0) (bq bk bv : Bool) {eps : ℝ} (heps : 0 < eps) (ks : ℕ → Fin T → Finset (Fin T))
    (hks : ∀ i t, (ks i t).Nonempty)
    (B : ℕ → Fin T → Fin T → (Fin dq → ℝ) →L[ℝ] (Fin dk → ℝ) →L[ℝ] ℝ)
    (ex : ℕ → Fin T → Fin T → V → V → ℝ) (qa : ℕ → V → Fin dq → ℝ) (kc : ℕ → V → Fin dk → ℝ)
    (vv : ℕ → V → Fin dv → ℝ) (Ov : ℕ → (Fin dv → ℝ) →L[ℝ] V) (post : ℕ → (Fin T → V) → Fin T → V)
    (P : ℕ → ℝ) (hP0 : ∀ i, 0 ≤ P i) (hP : ∀ i a b, dist (post i a) (post i b) ≤ P i * dist a b)
    (Ob Cb Q K Vb : ℝ) (hOb : ∀ i, ‖Ov i‖ ≤ Ob) (hCb0 : 0 ≤ Cb) (hQ0 : 0 ≤ Q) (hK0 : 0 ≤ K)
    (hV0 : 0 ≤ Vb) (hB : ∀ i t j, ‖B i t j‖ ≤ Cb)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb,
      ∀ b ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb,
      dist (plainAttnStack ks B ex qa kc vv Ov post i a) (plainAttnStack ks B ex qa kc vv Ov post i b)
        ≤ L i * dist a b) (n : ℕ) (Aset : Set (Fin T → V)) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ Aset, ∀ i ≤ n, ∀ y, dist y (run (plainAttnStack ks B ex qa kc vv Ov post) i x) ≤ r →
      y ∈ leverRegion (T := T) (qa i) (kc i) (vv i) Q K Vb) :
    TendstoUniformlyOn (fun S => run (leverAttnStack bq bk bv eps ks B ex qa kc vv Ov post S) n)
      (run (plainAttnStack ks B ex qa kc vv Ov post) n) atTop Aset :=
  lever_model_limit _ _ L hL1 (fun i => P i * leverAttnConst eps Ob Cb Q K Vb)
    (fun i => mul_nonneg (hP0 i) (leverAttnConst_nonneg heps ((norm_nonneg _).trans (hOb 0)) hCb0 hQ0
      hK0 hV0))
    _ _ hL (leverAttnStack_layer hdq hdk hdv bq bk bv heps ks hks B ex qa kc vv Ov post P hP0 hP
      Ob Cb Q K Vb hOb hCb0 hQ0 hK0 hV0 hB) n Aset r hr hK

end attn_stack

section post_stack

variable {T D : ℕ}

/-- The post-norm region: every token of the sublayer output bounded by `R`. -/
def postRegion (y : (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ) (R : ℝ) : Set (Fin T → Fin D → ℝ) :=
  {x | ∀ t, ‖y x t‖ ≤ R}

lemma postNorm_layer_on (hD : D ≠ 0) {eps : ℝ} (heps : 0 < eps)
    (y : ℕ → (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ) (R : ℝ) (hR0 : 0 ≤ R) :
    ∀ S, 0 < S → (0 : ℝ) ≤ S ^ 2 → ∀ i, ∀ x ∈ postRegion (y i) R,
      dist (postNormLayer eps S (y i) x) (plainResLayer (y i) x) ≤ R ^ 3 / (2 * eps) / S ^ 2 :=
  fun S hS _ i x hx => postNorm_layer_bound hD heps hS (y i) R hR0 x hx

/-- **Post-norm switched off, whole model, the bound.** A stack of residual sublayers `x + y i x`
whose outputs are bounded by `R` on the region; the host puts a levered post-norm on every
sublayer output. Logits within `Lh (Π L i) · n R³/(2ε) / S²`. -/
theorem postNorm_model_bound {Y : Type*} [PseudoMetricSpace Y] (hD : D ≠ 0) {eps : ℝ}
    (heps : 0 < eps) (y : ℕ → (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ) (R : ℝ) (hR0 : 0 ≤ R)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ postRegion (y i) R, ∀ b ∈ postRegion (y i) R,
      dist (plainResLayer (y i) a) (plainResLayer (y i) b) ≤ L i * dist a b)
    (head : (Fin T → Fin D → ℝ) → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ postRegion (y n) R, ∀ b ∈ postRegion (y n) R, dist (head a) (head b) ≤ Lh * dist a b)
    (S : ℝ) (hS : 0 < S) (x : Fin T → Fin D → ℝ)
    (hK : ∀ i ≤ n, ∀ z, dist z (run (fun i => plainResLayer (y i)) i x) ≤
      (∏ j ∈ range n, L j) * (∑ _j ∈ range n, R ^ 3 / (2 * eps)) / S ^ 2 → z ∈ postRegion (y i) R) :
    dist (head (run (fun i => postNormLayer eps S (y i)) n x))
        (head (run (fun i => plainResLayer (y i)) n x)) ≤
      Lh * (∏ i ∈ range n, L i) * (∑ _i ∈ range n, R ^ 3 / (2 * eps)) / S ^ 2 :=
  lever_model_bound (fun S i => postNormLayer eps S (y i)) _ L hL1 (fun _ => R ^ 3 / (2 * eps))
    (fun _ => by positivity) 0 (fun i => postRegion (y i) R) hL
    (postNorm_layer_on hD heps y R hR0) head Lh hLh0 n hLh S hS (sq_nonneg S) x hK

/-- **Post-norm switched off, identical generation.** Threshold
`S² > 2 Lh (Π L i) · n R³/(2ε) / δ` (and `(Π L) n R³/(2ε) ≤ r S²` for the region radius `r`). -/
theorem postNorm_greedy {Tok : Type*} [Fintype Tok] (hD : D ≠ 0) {eps : ℝ}
    (heps : 0 < eps) (y : ℕ → (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ) (R : ℝ) (hR0 : 0 ≤ R)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ postRegion (y i) R, ∀ b ∈ postRegion (y i) R,
      dist (plainResLayer (y i) a) (plainResLayer (y i) b) ≤ L i * dist a b)
    (head : (Fin T → Fin D → ℝ) → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ postRegion (y n) R, ∀ b ∈ postRegion (y n) R, dist (head a) (head b) ≤ Lh * dist a b)
    (embed : List Tok → Fin T → Fin D → ℝ) {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT)
    (hH : IsPick pickH) (p : List Tok) (δ r : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => plainResLayer (y i)) n
      (embed (greedy pickT (fun q => head (run (fun i => plainResLayer (y i)) n (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ z, dist z (run (fun i => plainResLayer (y i)) i
      (embed (greedy pickT (fun q => head (run (fun i => plainResLayer (y i)) n (embed q))) p k)))
        ≤ r → z ∈ postRegion (y i) R)
    (S : ℝ) (hS : 0 < S)
    (hr : (∏ i ∈ range n, L i) * (∑ _i ∈ range n, R ^ 3 / (2 * eps)) ≤ r * S ^ 2)
    (hsmall : 2 * Lh * (∏ i ∈ range n, L i) * (∑ _i ∈ range n, R ^ 3 / (2 * eps)) < δ * S ^ 2) :
    ∀ k, greedy pickH (fun q => head (run (fun i => postNormLayer eps S (y i)) n (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => plainResLayer (y i)) n (embed q))) p k :=
  lever_greedy (fun S i => postNormLayer eps S (y i)) _ L hL1 (fun _ => R ^ 3 / (2 * eps))
    (fun _ => by positivity) 0 (fun i => postRegion (y i) R) hL
    (postNorm_layer_on hD heps y R hR0) head Lh hLh0 n hLh embed hT hH p δ r hm hK S hS
    (sq_nonneg S) hr hsmall

/-- **Post-norm switched off, whole model, the limit** (`S → ∞`). -/
theorem postNorm_model_limit (hD : D ≠ 0) {eps : ℝ}
    (heps : 0 < eps) (y : ℕ → (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ) (R : ℝ) (hR0 : 0 ≤ R)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ postRegion (y i) R, ∀ b ∈ postRegion (y i) R,
      dist (plainResLayer (y i) a) (plainResLayer (y i) b) ≤ L i * dist a b)
    (n : ℕ) (Aset : Set (Fin T → Fin D → ℝ)) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ Aset, ∀ i ≤ n, ∀ z, dist z (run (fun i => plainResLayer (y i)) i x) ≤ r →
      z ∈ postRegion (y i) R) :
    TendstoUniformlyOn (fun S => run (fun i => postNormLayer eps S (y i)) n)
      (run (fun i => plainResLayer (y i)) n) atTop Aset :=
  lever_model_limit (fun S i => postNormLayer eps S (y i)) _ L hL1 (fun _ => R ^ 3 / (2 * eps))
    (fun _ => by positivity) 0 (fun i => postRegion (y i) R) hL
    (postNorm_layer_on hD heps y R hR0) n Aset r hr hK

end post_stack

end EP
