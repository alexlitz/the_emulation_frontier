import EmulationProofs.LimitKit
import EmulationProofs.ActLimits

/-!
# Whole-model limit instances: activation swaps and the final softcap

Every per-unit activation limit gives a whole-model statement through the kit of
`LimitKit.lean`. A layer is `A x + Σ_u σ(pre_u x) w_u` (`actLayer`): an arbitrary map `A` (the
residual, attention, norms, …) plus an MLP with activation `σ`. If the host activation is within
`η` of the target's at every pre-activation (or at every pre-activation of size `≤ R`), each layer
is within `η Σ_u ‖w_u‖` (`actLayer_dist`), and the kit gives

* the whole-model bound `Lh (Π L i) (Σ_i Σ_u ‖w_iu‖) η`,
* identical greedy generation once `2 Lh (Π L i)(Σ_i Σ_u ‖w_iu‖) η < δ`,
* convergence of the stack as `η → 0`.

Instances:
* **ReLU ← mish** (`relu_mish_bound`, `η = 1/β`, uniform);
* **ReLU ← GeLU**: `relu_gelu_bound` is new, `|GeLU(βx)/β − ReLU x| ≤ 1/β` for every `x`, from
  Chebyshev's inequality for the standard Gaussian (`P(|Z| ≥ t) ≤ 1/t²`);
* **relu² ← xIELU** (`relu2_xielu_bound`, `η = (α_n + (|β| + α_n) R)/(α_p c)`, only on
  pre-activations `|t| ≤ R`, so a region statement);
* **the final logit softcap** is exact for greedy generation at every `c > 0`: a strictly
  increasing map of the logits does not move the argmax (`final_softcap_greedy`).
-/

open Real Finset Filter Topology MeasureTheory ProbabilityTheory

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## A uniform GeLU bound from Chebyshev -/

/-- Chebyshev for the standard Gaussian: `P(|Z| ≥ t) ≤ 1/t²`. -/
lemma gauss_tail_abs (t : ℝ) (ht : 0 < t) :
    (gaussianReal 0 1).real {ω | t ≤ |ω|} ≤ 1 / t ^ 2 := by
  have h := meas_ge_le_variance_div_sq (μ := gaussianReal 0 1) (X := id)
    (memLp_id_gaussianReal 2) ht
  have hm : (gaussianReal 0 1)[id] = 0 := by simp
  rw [hm, variance_id_gaussianReal] at h
  simp only [id, sub_zero] at h
  rw [Measure.real]
  have := ENNReal.toReal_mono ENNReal.ofReal_ne_top h
  rw [ENNReal.toReal_ofReal (by positivity)] at this
  exact this

lemma gauss_cdf_neg_le (t : ℝ) (ht : 0 < t) :
    cdf (gaussianReal 0 1) (-t) ≤ 1 / t ^ 2 := by
  rw [cdf_eq_real]
  refine le_trans (measureReal_mono (fun ω (hω : ω ≤ -t) => ?_)) (gauss_tail_abs t ht)
  show t ≤ |ω|; rw [abs_of_neg (by linarith)]; linarith

lemma gauss_one_sub_cdf_le (t : ℝ) (ht : 0 < t) :
    1 - cdf (gaussianReal 0 1) t ≤ 1 / t ^ 2 := by
  rw [cdf_eq_real, ← probReal_compl_eq_one_sub₀ measurableSet_Iic.nullMeasurableSet]
  refine le_trans (measureReal_mono (fun ω (hω : ω ∈ (Set.Iic t)ᶜ) => ?_)) (gauss_tail_abs t ht)
  simp only [Set.mem_compl_iff, Set.mem_Iic, not_le] at hω
  show t ≤ |ω|; rw [abs_of_pos (by linarith)]; linarith

lemma mul_tail_le (β a p : ℝ) (hβ : 0 < β) (ha : 0 < a) (hp1 : p ≤ 1)
    (hpt : p ≤ 1 / (β * a) ^ 2) : a * p ≤ 1 / β := by
  rcases le_or_gt (β * a) 1 with h | h
  · calc a * p ≤ a * 1 := mul_le_mul_of_nonneg_left hp1 ha.le
      _ ≤ 1 / β := by rw [le_div_iff₀ hβ]; linarith
  · calc a * p ≤ a * (1 / (β * a) ^ 2) := mul_le_mul_of_nonneg_left hpt ha.le
      _ = 1 / β * (1 / (β * a)) := by field_simp
      _ ≤ 1 / β * 1 := by
          gcongr; rw [div_le_one (by positivity)]; exact h.le
      _ = 1 / β := mul_one _

/-- **ReLU ⊆ GeLU, the uniform rate.** `|GeLU(βx)/β − ReLU x| = |x| P(Z ≥ β|x|) ≤ 1/β` for every
`x` (`β > 0`): the tail is `≤ 1` and, by Chebyshev, `≤ 1/(β x)²`. -/
theorem relu_gelu_bound (β : ℝ) (hβ : 0 < β) (x : ℝ) : |gelu (β * x) / β - relu x| ≤ 1 / β := by
  set Φ := cdf (gaussianReal 0 1)
  have hfold : gelu (β * x) / β = x * Φ (β * x) := by simp only [gelu, Φ]; field_simp
  rw [hfold]
  have h0 := cdf_nonneg (gaussianReal 0 1) (β * x)
  have h1 := cdf_le_one (gaussianReal 0 1) (β * x)
  rcases lt_trichotomy x 0 with hx | hx | hx
  · rw [relu, max_eq_right hx.le, sub_zero, abs_mul, abs_of_neg hx, abs_of_nonneg h0]
    have := gauss_cdf_neg_le (β * -x) (by nlinarith)
    rw [show -(β * -x) = β * x by ring] at this
    exact mul_tail_le β (-x) _ hβ (by linarith) h1 this
  · subst hx; simp only [mul_zero, zero_mul, relu, max_self, sub_self, abs_zero]; positivity
  · rw [relu, max_eq_left hx.le, show x * Φ (β * x) - x = -(x * (1 - Φ (β * x))) by ring,
      abs_neg, abs_mul, abs_of_pos hx, abs_of_nonneg (by linarith)]
    exact mul_tail_le β x _ hβ hx (by linarith) (gauss_one_sub_cdf_le _ (by positivity))

/-! ## Activation swaps in a whole stack -/

section actswap

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- A layer: an arbitrary map `A` plus an MLP with activation `σ`, pre-activations `pre u` and
down-projection columns `w u`. (`reluLayer A pre w = actLayer relu A pre w` by definition.) -/
def actLayer {m : ℕ} (σ : ℝ → ℝ) (A : V → V) (pre : Fin m → V → ℝ) (w : Fin m → V) (x : V) : V :=
  A x + ∑ u, σ (pre u x) • w u

lemma reluLayer_eq_actLayer {m : ℕ} (A : V → V) (pre : Fin m → V → ℝ) (w : Fin m → V) :
    reluLayer A pre w = actLayer relu A pre w := rfl

/-- **One layer.** Activations within `η` at this state's pre-activations ⇒ the layers are within
`η Σ_u ‖w u‖`. -/
theorem actLayer_dist {m : ℕ} (σ τ : ℝ → ℝ) (A : V → V) (pre : Fin m → V → ℝ) (w : Fin m → V)
    (x : V) (η : ℝ) (h : ∀ u, |σ (pre u x) - τ (pre u x)| ≤ η) :
    dist (actLayer σ A pre w x) (actLayer τ A pre w x) ≤ η * ∑ u, ‖w u‖ := by
  rw [dist_eq_norm, actLayer, actLayer, add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib,
    Finset.mul_sum]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum (fun u _ => ?_))
  rw [← sub_smul, norm_smul, Real.norm_eq_abs]
  exact mul_le_mul_of_nonneg_right (h u) (norm_nonneg _)

lemma sum_eta_eq (n : ℕ) (W : ℕ → ℝ) (η : ℝ) :
    ∑ i ∈ range n, η * W i = (∑ i ∈ range n, W i) * η := by
  rw [← Finset.mul_sum, mul_comm]

lemma wsum_nonneg (m : ℕ → ℕ) (w : ∀ i, Fin (m i) → V) (i : ℕ) : 0 ≤ ∑ u, ‖w i u‖ :=
  Finset.sum_nonneg (fun u _ => norm_nonneg _)

/-- **Activation swap, the whole model.** Host activation within `η` of the target's
everywhere: logits within `Lh (Π L i)(Σ_i Σ_u ‖w i u‖) η`. -/
theorem act_swap_model_bound {Y : Type*} [PseudoMetricSpace Y] (σH σT : ℝ → ℝ) (η : ℝ)
    (hη0 : 0 ≤ η) (hη : ∀ t, |σH t - σT t| ≤ η)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (actLayer σT (A i) (pre i) (w i) a) (actLayer σT (A i) (pre i) (w i) b) ≤
      L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b)
    (n : ℕ) (x : V) :
    dist (head (run (fun i => actLayer σH (A i) (pre i) (w i)) n x))
        (head (run (fun i => actLayer σT (A i) (pre i) (w i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * ((∑ i ∈ range n, ∑ u, ‖w i u‖) * η)) := by
  rw [← sum_eta_eq]
  exact model_bound_prod _ _ L (fun i => η * ∑ u, ‖w i u‖) hL1
    (fun i => mul_nonneg hη0 (wsum_nonneg m w i)) hL
    (fun i y => actLayer_dist _ _ _ _ _ y η (fun u => hη _)) head Lh hLh0 hLh n x

/-- **Activation swap, identical generation.** Threshold `2 Lh (Π L i)(Σ_i Σ_u ‖w i u‖) η < δ`. -/
theorem act_swap_greedy {Tok : Type*} [Fintype Tok] (σH σT : ℝ → ℝ) (η : ℝ) (hη0 : 0 ≤ η)
    (hη : ∀ t, |σH t - σT t| ≤ η)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (actLayer σT (A i) (pre i) (w i) a) (actLayer σT (A i) (pre i) (w i) b) ≤
      L i * dist a b)
    (head : V → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh)
    (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b) (n : ℕ) (embed : List Tok → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => actLayer σT (A i) (pre i) (w i)) n
      (embed (greedy pickT (fun q => head (run (fun i => actLayer σT (A i) (pre i) (w i)) n
        (embed q))) p k)))) a)
    (hsmall : 2 * Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, ∑ u, ‖w i u‖) * η < δ) :
    ∀ k, greedy pickH (fun q => head (run (fun i => actLayer σH (A i) (pre i) (w i)) n (embed q)))
        p k =
      greedy pickT (fun q => head (run (fun i => actLayer σT (A i) (pre i) (w i)) n (embed q)))
        p k :=
  greedy_of_layer_bounds _ _ L (fun i => η * ∑ u, ‖w i u‖) hL1
    (fun i => mul_nonneg hη0 (wsum_nonneg m w i)) hL
    (fun i y => actLayer_dist _ _ _ _ _ y η (fun u => hη _)) head Lh hLh0 hLh n embed hT hH p δ
    hm (by rw [sum_eta_eq]; linarith)

/-- **Activation swap, the limit.** Host activations `σH β` within `η β → 0` of a continuous
target activation, continuous `A i` and pre-activations: the host stack converges to the target
stack locally uniformly (`model_limit`; no Lipschitz constant needed). -/
theorem act_swap_model_limit {ι : Type*} {pf : Filter ι} (σH : ι → ℝ → ℝ) (σT : ℝ → ℝ)
    (hσT : Continuous σT) (η : ι → ℝ) (hη : ∀ β t, |σH β t - σT t| ≤ η β)
    (hηlim : Tendsto η pf (𝓝 0))
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (hA : ∀ i, Continuous (A i)) (hpre : ∀ i u, Continuous (pre i u)) (n : ℕ) :
    TendstoLocallyUniformly (fun β => run (fun i => actLayer (σH β) (A i) (pre i) (w i)) n)
      (run (fun i => actLayer σT (A i) (pre i) (w i)) n) pf := by
  refine model_limit _ _ (fun i => ?_) (fun i => ?_) n
  · exact (hA i).add (continuous_finsetSum _ (fun u _ =>
      (hσT.comp (hpre i u)).smul continuous_const))
  · refine TendstoUniformly.tendstoLocallyUniformly ?_
    rw [Metric.tendstoUniformly_iff]
    intro e he
    have hC : Tendsto (fun β => η β * ∑ u, ‖w i u‖) pf (𝓝 0) := by
      simpa using hηlim.mul_const (∑ u, ‖w i u‖)
    filter_upwards [hC.eventually (gt_mem_nhds he)] with β h1 x
    rw [dist_comm]
    exact (actLayer_dist _ _ _ _ _ x _ (fun u => hη β _)).trans_lt h1

/-! ### On a region: activations close only on bounded pre-activations -/

/-- The states whose pre-activations are all bounded by `R`. -/
def preBox {m : ℕ} (pre : Fin m → V → ℝ) (R : ℝ) : Set V := {y | ∀ u, |pre u y| ≤ R}

/-- **Activation swap on a region, the whole model.** Activations within `η` on `[−R, R]`,
target layers Lipschitz on their `preBox`, head Lipschitz on the last one, and every state within
`B = (Π L i)(Σ_i Σ_u ‖w i u‖) η` of the target trajectory inside the boxes: logits within
`Lh B`. -/
theorem act_swap_model_bound_on {Y : Type*} [PseudoMetricSpace Y] (σH σT : ℝ → ℝ) (η R : ℝ)
    (hη0 : 0 ≤ η) (hη : ∀ t, |t| ≤ R → |σH t - σT t| ≤ η)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ preBox (pre i) R, ∀ b ∈ preBox (pre i) R,
      dist (actLayer σT (A i) (pre i) (w i) a) (actLayer σT (A i) (pre i) (w i) b) ≤
        L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ preBox (pre n) R, ∀ b ∈ preBox (pre n) R, dist (head a) (head b) ≤ Lh * dist a b)
    (x : V)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (fun i => actLayer σT (A i) (pre i) (w i)) i x) ≤
      (∏ j ∈ range n, L j) * ((∑ j ∈ range n, ∑ u, ‖w j u‖) * η) → y ∈ preBox (pre i) R) :
    dist (head (run (fun i => actLayer σH (A i) (pre i) (w i)) n x))
        (head (run (fun i => actLayer σT (A i) (pre i) (w i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * ((∑ i ∈ range n, ∑ u, ‖w i u‖) * η)) := by
  rw [← sum_eta_eq] at hK ⊢
  exact model_bound_on_prod _ _ L (fun i => η * ∑ u, ‖w i u‖) hL1
    (fun i => mul_nonneg hη0 (wsum_nonneg m w i)) (fun i => preBox (pre i) R) hL
    (fun i y hy => actLayer_dist _ _ _ _ _ y η (fun u => hη _ (hy u))) head Lh hLh0 n hLh x hK

/-- **Activation swap on a region, identical generation.** -/
theorem act_swap_greedy_on {Tok : Type*} [Fintype Tok] (σH σT : ℝ → ℝ) (η R : ℝ) (hη0 : 0 ≤ η)
    (hη : ∀ t, |t| ≤ R → |σH t - σT t| ≤ η)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ preBox (pre i) R, ∀ b ∈ preBox (pre i) R,
      dist (actLayer σT (A i) (pre i) (w i) a) (actLayer σT (A i) (pre i) (w i) b) ≤
        L i * dist a b)
    (head : V → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ preBox (pre n) R, ∀ b ∈ preBox (pre n) R, dist (head a) (head b) ≤ Lh * dist a b)
    (embed : List Tok → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => actLayer σT (A i) (pre i) (w i)) n
      (embed (greedy pickT (fun q => head (run (fun i => actLayer σT (A i) (pre i) (w i)) n
        (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run (fun i => actLayer σT (A i) (pre i) (w i)) i
      (embed (greedy pickT (fun q => head (run (fun i => actLayer σT (A i) (pre i) (w i)) n
        (embed q))) p k))) ≤
      (∏ j ∈ range n, L j) * ((∑ j ∈ range n, ∑ u, ‖w j u‖) * η) → y ∈ preBox (pre i) R)
    (hsmall : 2 * Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, ∑ u, ‖w i u‖) * η < δ) :
    ∀ k, greedy pickH (fun q => head (run (fun i => actLayer σH (A i) (pre i) (w i)) n (embed q)))
        p k =
      greedy pickT (fun q => head (run (fun i => actLayer σT (A i) (pre i) (w i)) n (embed q)))
        p k := by
  rw [← sum_eta_eq] at hK
  exact greedy_of_layer_bounds_on _ _ L (fun i => η * ∑ u, ‖w i u‖) hL1
    (fun i => mul_nonneg hη0 (wsum_nonneg m w i)) (fun i => preBox (pre i) R) hL
    (fun i y hy => actLayer_dist _ _ _ _ _ y η (fun u => hη _ (hy u))) head Lh hLh0 n hLh embed
    hT hH p δ hm hK (by rw [sum_eta_eq]; linarith)

/-- **Activation swap on a region, the limit.** Uniform convergence on every input set `A₀` whose
target trajectories keep an `r`-neighbourhood inside the boxes. -/
theorem act_swap_model_limit_on {ι : Type*} {pf : Filter ι} (σH : ι → ℝ → ℝ) (σT : ℝ → ℝ)
    (η : ι → ℝ) (R : ℝ) (hη0 : ∀ β, 0 ≤ η β) (hη : ∀ β t, |t| ≤ R → |σH β t - σT t| ≤ η β)
    (hηlim : Tendsto η pf (𝓝 0))
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ preBox (pre i) R, ∀ b ∈ preBox (pre i) R,
      dist (actLayer σT (A i) (pre i) (w i) a) (actLayer σT (A i) (pre i) (w i) b) ≤
        L i * dist a b)
    (n : ℕ) (A₀ : Set V) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ A₀, ∀ i ≤ n, ∀ y, dist y (run (fun i => actLayer σT (A i) (pre i) (w i)) i x) ≤ r →
      y ∈ preBox (pre i) R) :
    TendstoUniformlyOn (fun β => run (fun i => actLayer (σH β) (A i) (pre i) (w i)) n)
      (run (fun i => actLayer σT (A i) (pre i) (w i)) n) pf A₀ :=
  model_limit_on (fun β i => actLayer (σH β) (A i) (pre i) (w i)) _ L hL1
    (fun β i => η β * ∑ u, ‖w i u‖) (fun β i => mul_nonneg (hη0 β) (wsum_nonneg m w i))
    (fun i => by simpa using hηlim.mul_const (∑ u, ‖w i u‖)) (fun i => preBox (pre i) R) hL
    (fun β i y hy => actLayer_dist _ _ _ _ _ y _ (fun u => hη β _ (hy u))) n A₀ r hr hK

end actswap

/-! ## Instances -/

section instances

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {Tok : Type*} [Fintype Tok]

lemma tendstoLocallyUniformly_congr_eventually {ι X Y : Type*} [TopologicalSpace X]
    [UniformSpace Y] {pf : Filter ι} {F G : ι → X → Y} {f : X → Y}
    (h : TendstoLocallyUniformly F f pf) (hFG : ∀ᶠ β in pf, F β = G β) :
    TendstoLocallyUniformly G f pf := fun u hu x => by
  obtain ⟨t, ht, h'⟩ := h u hu x
  exact ⟨t, ht, by filter_upwards [h', hFG] with β h1 h2; rwa [← h2]⟩

lemma inv_mul_lt_of_lt (C δ β : ℝ) (hδ : 0 < δ) (hC : 0 ≤ C) (hβ : C / δ < β) :
    0 < β ∧ C * (1 / β) < δ := by
  have hβ0 : 0 < β := lt_of_le_of_lt (div_nonneg hC hδ.le) hβ
  refine ⟨hβ0, ?_⟩
  rw [div_lt_iff₀ hδ] at hβ
  rw [mul_one_div, div_lt_iff₀ hβ0]; linarith

lemma model_const_nonneg (Lh : ℝ) (hLh0 : 0 ≤ Lh) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i) (n : ℕ)
    (W : ℕ → ℝ) (hW : ∀ i, 0 ≤ W i) :
    0 ≤ 2 * Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, W i) := by
  have : 0 ≤ ∏ i ∈ range n, L i := Finset.prod_nonneg (fun i _ => by linarith [hL1 i])
  have : 0 ≤ ∑ i ∈ range n, W i := Finset.sum_nonneg (fun i _ => hW i)
  positivity

/-- **ReLU ⊆ mish, the whole model.** Logits within `Lh (Π L i)(Σ_i Σ_u ‖w i u‖)/β`. -/
theorem relu_mish_model_bound {Y : Type*} [PseudoMetricSpace Y] (β : ℝ) (hβ : 0 < β)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (reluLayer (A i) (pre i) (w i) a) (reluLayer (A i) (pre i) (w i) b) ≤
      L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b)
    (n : ℕ) (x : V) :
    dist (head (run (fun i => actLayer (fun t => mish (β * t) / β) (A i) (pre i) (w i)) n x))
        (head (run (fun i => reluLayer (A i) (pre i) (w i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * ((∑ i ∈ range n, ∑ u, ‖w i u‖) * (1 / β))) :=
  act_swap_model_bound _ relu _ (by positivity) (relu_mish_bound β hβ) m A pre w L hL1 hL head Lh
    hLh0 hLh n x

/-- **ReLU ⊆ mish, identical generation, explicit `β*`.** For
`β > β* = 2 Lh (Π L i)(Σ_i Σ_u ‖w i u‖)/δ` the mish host generates the target's tokens. -/
theorem relu_mish_greedy (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ)
    (w : ∀ i, Fin (m i) → V) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (reluLayer (A i) (pre i) (w i) a) (reluLayer (A i) (pre i) (w i) b) ≤
      L i * dist a b)
    (head : V → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh)
    (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b) (n : ℕ) (embed : List Tok → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ) (hδ : 0 < δ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => reluLayer (A i) (pre i) (w i)) n
      (embed (greedy pickT (fun q => head (run (fun i => reluLayer (A i) (pre i) (w i)) n
        (embed q))) p k)))) a)
    (β : ℝ) (hβ : 2 * Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, ∑ u, ‖w i u‖) / δ < β) :
    ∀ k, greedy pickH (fun q => head (run (fun i => actLayer (fun t => mish (β * t) / β) (A i)
        (pre i) (w i)) n (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => reluLayer (A i) (pre i) (w i)) n (embed q))) p k :=
  have h := inv_mul_lt_of_lt _ δ β hδ
    (model_const_nonneg Lh hLh0 L hL1 n _ (wsum_nonneg m w)) hβ
  act_swap_greedy _ relu _ (by have := h.1; positivity) (relu_mish_bound β h.1) m A pre w L hL1 hL
    head Lh hLh0 hLh n embed hT hH p δ hm h.2

/-- **ReLU ⊆ mish, the whole-model limit** (locally uniform as `β → ∞`). -/
theorem relu_mish_model_limit (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ)
    (w : ∀ i, Fin (m i) → V) (hA : ∀ i, Continuous (A i)) (hpre : ∀ i u, Continuous (pre i u))
    (n : ℕ) :
    TendstoLocallyUniformly
      (fun β : ℝ => run (fun i => actLayer (fun t => mish (β * t) / β) (A i) (pre i) (w i)) n)
      (run (fun i => reluLayer (A i) (pre i) (w i)) n) atTop := by
  have hlim : Tendsto (fun β : ℝ => |1 / β|) atTop (𝓝 0) := by
    simpa using (tendsto_const_nhds.div_atTop tendsto_id : Tendsto (fun β : ℝ => 1 / β) atTop
      (𝓝 0)).abs
  refine tendstoLocallyUniformly_congr_eventually (act_swap_model_limit
    (fun β t => if 0 < β then mish (β * t) / β else relu t) relu
    continuous_relu (fun β => |1 / β|) (fun β t => ?_) hlim m A pre w hA hpre n) ?_
  · split_ifs with h
    · rw [abs_of_pos (show (0 : ℝ) < 1 / β by positivity)]; exact relu_mish_bound β h t
    · simp
  · filter_upwards [eventually_gt_atTop 0] with β hβ
    simp [hβ]

/-- **ReLU ⊆ GeLU, the whole model.** Logits within `Lh (Π L i)(Σ_i Σ_u ‖w i u‖)/β`. -/
theorem relu_gelu_model_bound {Y : Type*} [PseudoMetricSpace Y] (β : ℝ) (hβ : 0 < β)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (reluLayer (A i) (pre i) (w i) a) (reluLayer (A i) (pre i) (w i) b) ≤
      L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b)
    (n : ℕ) (x : V) :
    dist (head (run (fun i => actLayer (fun t => gelu (β * t) / β) (A i) (pre i) (w i)) n x))
        (head (run (fun i => reluLayer (A i) (pre i) (w i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * ((∑ i ∈ range n, ∑ u, ‖w i u‖) * (1 / β))) :=
  act_swap_model_bound _ relu _ (by positivity) (relu_gelu_bound β hβ) m A pre w L hL1 hL head Lh
    hLh0 hLh n x

/-- **ReLU ⊆ GeLU, identical generation, explicit `β*`** (the same `β*` as SiLU and mish). -/
theorem relu_gelu_greedy (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ)
    (w : ∀ i, Fin (m i) → V) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (reluLayer (A i) (pre i) (w i) a) (reluLayer (A i) (pre i) (w i) b) ≤
      L i * dist a b)
    (head : V → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh)
    (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b) (n : ℕ) (embed : List Tok → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ) (hδ : 0 < δ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => reluLayer (A i) (pre i) (w i)) n
      (embed (greedy pickT (fun q => head (run (fun i => reluLayer (A i) (pre i) (w i)) n
        (embed q))) p k)))) a)
    (β : ℝ) (hβ : 2 * Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, ∑ u, ‖w i u‖) / δ < β) :
    ∀ k, greedy pickH (fun q => head (run (fun i => actLayer (fun t => gelu (β * t) / β) (A i)
        (pre i) (w i)) n (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => reluLayer (A i) (pre i) (w i)) n (embed q))) p k :=
  have h := inv_mul_lt_of_lt _ δ β hδ
    (model_const_nonneg Lh hLh0 L hL1 n _ (wsum_nonneg m w)) hβ
  act_swap_greedy _ relu _ (by have := h.1; positivity) (relu_gelu_bound β h.1) m A pre w L hL1 hL
    head Lh hLh0 hLh n embed hT hH p δ hm h.2

/-- **ReLU ⊆ GeLU, the whole-model limit** (locally uniform as `β → ∞`). -/
theorem relu_gelu_model_limit (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ)
    (w : ∀ i, Fin (m i) → V) (hA : ∀ i, Continuous (A i)) (hpre : ∀ i u, Continuous (pre i u))
    (n : ℕ) :
    TendstoLocallyUniformly
      (fun β : ℝ => run (fun i => actLayer (fun t => gelu (β * t) / β) (A i) (pre i) (w i)) n)
      (run (fun i => reluLayer (A i) (pre i) (w i)) n) atTop := by
  have hlim : Tendsto (fun β : ℝ => |1 / β|) atTop (𝓝 0) := by
    simpa using (tendsto_const_nhds.div_atTop tendsto_id : Tendsto (fun β : ℝ => 1 / β) atTop
      (𝓝 0)).abs
  refine tendstoLocallyUniformly_congr_eventually (act_swap_model_limit
    (fun β t => if 0 < β then gelu (β * t) / β else relu t) relu
    continuous_relu (fun β => |1 / β|) (fun β t => ?_) hlim m A pre w hA hpre n) ?_
  · split_ifs with h
    · rw [abs_of_pos (show (0 : ℝ) < 1 / β by positivity)]; exact relu_gelu_bound β h t
    · simp
  · filter_upwards [eventually_gt_atTop 0] with β hβ
    simp [hβ]

/-- The relu² target activation. -/
def relu2 (t : ℝ) : ℝ := relu t ^ 2

/-- The xIELU host activation, rescaled: `xIELU(c t)/(α_p c²)`. -/
def xieluHost (αp αn β ε c t : ℝ) : ℝ := xielu αp αn β ε (c * t) / (αp * c ^ 2)

lemma xielu_host_close (αp αn β ε R c : ℝ) (hp : 0 < αp) (hn : 0 ≤ αn) (hε : ε ≤ 0) (hc : 1 ≤ c)
    (t : ℝ) (ht : |t| ≤ R) :
    |xieluHost αp αn β ε c t - relu2 t| ≤ (αn + (|β| + αn) * R) / (αp * c) := by
  refine (relu2_xielu_bound αp αn β ε hp hn hε c hc t).trans ?_
  have : 0 < αp * c := by nlinarith
  gcongr

/-- **relu² ⊆ xIELU, the whole model, on bounded pre-activations.** Every MLP pre-activation
within `R` along the error-neighbourhood of the target trajectory (`hK`), `c ≥ 1`: logits within
`Lh (Π L i)(Σ_i Σ_u ‖w i u‖)(α_n + (|β| + α_n) R)/(α_p c)`. -/
theorem relu2_xielu_model_bound {Y : Type*} [PseudoMetricSpace Y] (αp αn β ε R c : ℝ)
    (hp : 0 < αp) (hn : 0 ≤ αn) (hε : ε ≤ 0) (hR : 0 ≤ R) (hc : 1 ≤ c)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ preBox (pre i) R, ∀ b ∈ preBox (pre i) R,
      dist (actLayer relu2 (A i) (pre i) (w i) a) (actLayer relu2 (A i) (pre i) (w i) b) ≤
        L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ preBox (pre n) R, ∀ b ∈ preBox (pre n) R, dist (head a) (head b) ≤ Lh * dist a b)
    (x : V)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (fun i => actLayer relu2 (A i) (pre i) (w i)) i x) ≤
      (∏ j ∈ range n, L j) * ((∑ j ∈ range n, ∑ u, ‖w j u‖) *
        ((αn + (|β| + αn) * R) / (αp * c))) → y ∈ preBox (pre i) R) :
    dist (head (run (fun i => actLayer (xieluHost αp αn β ε c) (A i) (pre i) (w i)) n x))
        (head (run (fun i => actLayer relu2 (A i) (pre i) (w i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * ((∑ i ∈ range n, ∑ u, ‖w i u‖) *
        ((αn + (|β| + αn) * R) / (αp * c)))) :=
  act_swap_model_bound_on _ _ _ R (by have : 0 < αp * c := by nlinarith
                                      positivity)
    (xielu_host_close αp αn β ε R c hp hn hε hc) m A pre w L hL1 hL head Lh hLh0 n hLh x hK

/-- **relu² ⊆ xIELU, identical generation, explicit `c*`.** For `c ≥ 1` with
`c > c* = 2 Lh (Π L i)(Σ_i Σ_u ‖w i u‖)(α_n + (|β| + α_n) R)/(α_p δ)` (stated multiplied out). -/
theorem relu2_xielu_greedy (αp αn β ε R c : ℝ)
    (hp : 0 < αp) (hn : 0 ≤ αn) (hε : ε ≤ 0) (hR : 0 ≤ R) (hc : 1 ≤ c)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ preBox (pre i) R, ∀ b ∈ preBox (pre i) R,
      dist (actLayer relu2 (A i) (pre i) (w i) a) (actLayer relu2 (A i) (pre i) (w i) b) ≤
        L i * dist a b)
    (head : V → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ preBox (pre n) R, ∀ b ∈ preBox (pre n) R, dist (head a) (head b) ≤ Lh * dist a b)
    (embed : List Tok → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => actLayer relu2 (A i) (pre i) (w i)) n
      (embed (greedy pickT (fun q => head (run (fun i => actLayer relu2 (A i) (pre i) (w i)) n
        (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run (fun i => actLayer relu2 (A i) (pre i) (w i)) i
      (embed (greedy pickT (fun q => head (run (fun i => actLayer relu2 (A i) (pre i) (w i)) n
        (embed q))) p k))) ≤
      (∏ j ∈ range n, L j) * ((∑ j ∈ range n, ∑ u, ‖w j u‖) *
        ((αn + (|β| + αn) * R) / (αp * c))) → y ∈ preBox (pre i) R)
    (hc' : 2 * Lh * (∏ i ∈ range n, L i) * (∑ i ∈ range n, ∑ u, ‖w i u‖) *
      (αn + (|β| + αn) * R) < αp * c * δ) :
    ∀ k, greedy pickH (fun q => head (run (fun i => actLayer (xieluHost αp αn β ε c) (A i)
        (pre i) (w i)) n (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => actLayer relu2 (A i) (pre i) (w i)) n
        (embed q))) p k := by
  have hpc : 0 < αp * c := by nlinarith
  refine act_swap_greedy_on _ _ _ R (by positivity) (xielu_host_close αp αn β ε R c hp hn hε hc)
    m A pre w L hL1 hL head Lh hLh0 n hLh embed hT hH p δ hm hK ?_
  rw [mul_div_assoc', div_lt_iff₀ hpc]; linarith

/-- **relu² ⊆ xIELU, the whole-model limit** (`c → ∞`), uniformly on every input set whose target
trajectories keep an `r`-neighbourhood with pre-activations `≤ R`. -/
theorem relu2_xielu_model_limit (αp αn β ε R : ℝ) (hp : 0 < αp) (hn : 0 ≤ αn) (hε : ε ≤ 0)
    (hR : 0 ≤ R)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ preBox (pre i) R, ∀ b ∈ preBox (pre i) R,
      dist (actLayer relu2 (A i) (pre i) (w i) a) (actLayer relu2 (A i) (pre i) (w i) b) ≤
        L i * dist a b)
    (n : ℕ) (A₀ : Set V) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ A₀, ∀ i ≤ n, ∀ y, dist y (run (fun i => actLayer relu2 (A i) (pre i) (w i)) i x) ≤
      r → y ∈ preBox (pre i) R) :
    TendstoUniformlyOn
      (fun c : ℝ => run (fun i => actLayer (xieluHost αp αn β ε (max c 1)) (A i) (pre i) (w i)) n)
      (run (fun i => actLayer relu2 (A i) (pre i) (w i)) n) atTop A₀ := by
  set C := (αn + (|β| + αn) * R) / αp
  have hC : 0 ≤ C := by positivity
  have hlim : Tendsto (fun c : ℝ => C / max c 1) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop (tendsto_atTop_mono (fun c => le_max_left c 1) tendsto_id)
  refine act_swap_model_limit_on (fun c => xieluHost αp αn β ε (max c 1)) relu2
    (fun c => C / max c 1) R (fun c => div_nonneg hC (by positivity)) (fun c t ht => ?_) hlim
    m A pre w L hL1 hL n A₀ r hr hK
  refine (xielu_host_close αp αn β ε R _ hp hn hε (le_max_right c 1) t ht).trans (le_of_eq ?_)
  simp only [C]; field_simp

end instances

/-! ## The final logit softcap is exact for greedy generation -/

section final_cap

variable {Tok : Type*}

/-- A pick through a strictly increasing map of the logits is a pick. -/
lemma isPick_comp_strictMono {pick : (Tok → ℝ) → Tok} (hp : IsPick pick) {φ : ℝ → ℝ}
    (hφ : StrictMono φ) : IsPick (fun z => pick (fun b => φ (z b))) := fun z b =>
  hφ.le_iff_le.mp (hp (fun b => φ (z b)) b)

lemma tanh_strictMono : StrictMono tanh :=
  strictMono_of_deriv_pos (fun x => by
    rw [(hasDerivAt_tanh x).deriv]
    have := abs_tanh_lt_one x
    have : tanh x ^ 2 < 1 := by rw [← sq_abs]; nlinarith [abs_nonneg (tanh x)]
    linarith)

lemma softcap_strictMono (c : ℝ) (hc : 0 < c) : StrictMono (softcap c) := fun a b h => by
  simp only [softcap]
  exact mul_lt_mul_of_pos_left (tanh_strictMono (div_lt_div_of_pos_right h hc)) hc

lemma greedy_comp (pick : (Tok → ℝ) → Tok) (φ : ℝ → ℝ) (lg : List Tok → Tok → ℝ) (p : List Tok)
    (k : ℕ) : greedy pick (fun q b => φ (lg q b)) p k = greedy (fun z => pick (fun b => φ (z b)))
      lg p k := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [greedy, ih]

/-- **The final logit softcap, identical generation, every `c > 0`.** A host whose logits are a
softcap `c·tanh(z/c)` of logits within `< δ/2` of the target's (margin `δ` along the target's
trajectory) generates the target's tokens: the cap is strictly increasing, so it never moves the
argmax. With `lgH = lgT` this says dropping (or adding) the final cap is exact for greedy text. -/
theorem final_softcap_greedy {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT)
    (hH : IsPick pickH) (c : ℝ) (hc : 0 < c) (lgT lgH : List Tok → Tok → ℝ) (p : List Tok)
    (δ : ℝ) (hm : ∀ k, ∃ a, MarginAt δ (lgT (greedy pickT lgT p k)) a)
    (hclose : ∀ k b, |lgH (greedy pickT lgT p k) b - lgT (greedy pickT lgT p k) b| < δ / 2) :
    ∀ k, greedy pickH (fun q b => softcap c (lgH q b)) p k = greedy pickT lgT p k := by
  intro k
  rw [greedy_comp]
  exact greedy_preserved hT (isPick_comp_strictMono hH (softcap_strictMono c hc)) lgT lgH p δ hm
    hclose k

end final_cap

end EP
