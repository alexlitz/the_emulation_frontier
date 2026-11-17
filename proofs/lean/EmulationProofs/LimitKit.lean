import EmulationProofs.ModelLimit

/-!
# A kit for whole-model limit instances

`ModelLimit.lean` proves the whole-model bound (`model_error_bound`) with GLOBAL per-layer errors
and Lipschitz constants. Most per-operation limits only hold on bounded states (the eps lever,
softcap, the sink, hard masks, the soft router under a margin, relu² ← xIELU), so this file adds:

* `model_bound_prod`: the global bound in product form through a head;
* `model_error_bound_on`, `model_bound_on_prod`: the same bound when the per-layer error and the
  target's Lipschitz constant hold only on sets `K i` that contain the error-neighbourhood of the
  target's trajectory (so the host never leaves the region where the per-layer bound holds);
* `greedy_of_layer_bounds`, `greedy_of_layer_bounds_on`: identical greedy generation from
  per-layer errors, with the explicit threshold `2 Lh (Π L i)(Σ ε i) < δ`;
* `model_limit_of_rates`, `model_limit_on`: per-layer errors `ε β i → 0` give uniform
  convergence of the whole stack (over all inputs, resp. over a set `A` of inputs whose target
  trajectories keep an `r`-neighbourhood inside the `K i`).
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

section kit

variable {X : Type*} [PseudoMetricSpace X]

lemma one_le_prod_range (L : ℕ → ℝ) (hL : ∀ i, 1 ≤ L i) (n : ℕ) : 1 ≤ ∏ i ∈ range n, L i := by
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.prod_range_succ]; nlinarith [hL n]

lemma one_le_prod_finset (L : ℕ → ℝ) (hL : ∀ i, 1 ≤ L i) (s : Finset ℕ) :
    1 ≤ ∏ i ∈ s, L i := by
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih => rw [Finset.prod_insert ha]; nlinarith [hL a]

lemma errBound_nonneg (L ε : ℕ → ℝ) (hL0 : ∀ i, 0 ≤ L i) (hε : ∀ i, 0 ≤ ε i) (n : ℕ) :
    0 ≤ errBound L ε n := by
  induction n with
  | zero => simp [errBound]
  | succ n ih => rw [errBound]; have := hL0 n; have := hε n; positivity

/-- The product bound at `n` dominates `errBound` at every `i ≤ n`. -/
lemma errBound_le_prod_sum_of_le (L ε : ℕ → ℝ) (hL : ∀ i, 1 ≤ L i) (hε : ∀ i, 0 ≤ ε i)
    {i n : ℕ} (hin : i ≤ n) :
    errBound L ε i ≤ (∏ j ∈ range n, L j) * ∑ j ∈ range n, ε j := by
  refine (errBound_le_prod_sum L ε hL hε i).trans ?_
  have hP : ∏ j ∈ range i, L j ≤ ∏ j ∈ range n, L j := by
    rw [← Finset.prod_range_mul_prod_Ico _ hin]
    have h1 : 1 ≤ ∏ j ∈ Ico i n, L j := one_le_prod_finset L hL _
    have h0 : 0 ≤ ∏ j ∈ range i, L j := (zero_le_one.trans (one_le_prod_range L hL i))
    nlinarith
  have hS : ∑ j ∈ range i, ε j ≤ ∑ j ∈ range n, ε j :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hin) (fun j _ _ => hε j)
  have hS0 : 0 ≤ ∑ j ∈ range i, ε j := Finset.sum_nonneg (fun j _ => hε j)
  have hP1 := one_le_prod_range L hL i
  nlinarith

/-- **The global whole-model bound, product form, through a head.** -/
theorem model_bound_prod {Y : Type*} [PseudoMetricSpace Y] (F f : ℕ → X → X) (L ε : ℕ → ℝ)
    (hL1 : ∀ i, 1 ≤ L i) (hε0 : ∀ i, 0 ≤ ε i)
    (hL : ∀ i a b, dist (f i a) (f i b) ≤ L i * dist a b) (hε : ∀ i y, dist (F i y) (f i y) ≤ ε i)
    (head : X → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b)
    (n : ℕ) (x : X) :
    dist (head (run F n x)) (head (run f n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, ε i) :=
  (model_error_bound_head F f L ε (fun i => by linarith [hL1 i]) hL hε head Lh hLh0 hLh n x).trans
    (mul_le_mul_of_nonneg_left (errBound_le_prod_sum L ε hL1 hε0 n) hLh0)

/-- **The whole-model bound on a region.** The target layer `i` is `L i`-Lipschitz on `K i` and
the host layer `i` is within `ε i` of it on `K i`; every state within `errBound L ε i` of the
target's state at layer `i` lies in `K i`. Then the host stays in the region and its output is
within `errBound L ε n`. -/
theorem model_error_bound_on (F f : ℕ → X → X) (L ε : ℕ → ℝ) (hL0 : ∀ i, 0 ≤ L i)
    (hε0 : ∀ i, 0 ≤ ε i) (K : ℕ → Set X)
    (hL : ∀ i, ∀ a ∈ K i, ∀ b ∈ K i, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ i, ∀ y ∈ K i, dist (F i y) (f i y) ≤ ε i) (n : ℕ) (x : X)
    (hK : ∀ i < n, ∀ y, dist y (run f i x) ≤ errBound L ε i → y ∈ K i) :
    dist (run F n x) (run f n x) ≤ errBound L ε n := by
  induction n with
  | zero => simp [run, errBound]
  | succ n ih =>
    have ih' := ih (fun i hi => hK i (by omega))
    have hF : run F n x ∈ K n := hK n (by omega) _ ih'
    have hf : run f n x ∈ K n := hK n (by omega) _ (by
      rw [dist_self]; exact errBound_nonneg L ε hL0 hε0 n)
    show dist (F n (run F n x)) (f n (run f n x)) ≤ L n * errBound L ε n + ε n
    calc dist (F n (run F n x)) (f n (run f n x))
        ≤ dist (F n (run F n x)) (f n (run F n x)) + dist (f n (run F n x)) (f n (run f n x)) :=
          dist_triangle _ _ _
      _ ≤ ε n + L n * errBound L ε n :=
          add_le_add (hε _ _ hF) ((hL _ _ hF _ hf).trans (mul_le_mul_of_nonneg_left ih' (hL0 n)))
      _ = _ := by ring

/-- **…in product form, through a head that is `Lh`-Lipschitz on `K n`.** The region condition is
now stated with the single radius `B = (Π L i)(Σ ε i)`. -/
theorem model_bound_on_prod {Y : Type*} [PseudoMetricSpace Y] (F f : ℕ → X → X) (L ε : ℕ → ℝ)
    (hL1 : ∀ i, 1 ≤ L i) (hε0 : ∀ i, 0 ≤ ε i) (K : ℕ → Set X)
    (hL : ∀ i, ∀ a ∈ K i, ∀ b ∈ K i, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ i, ∀ y ∈ K i, dist (F i y) (f i y) ≤ ε i)
    (head : X → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ K n, ∀ b ∈ K n, dist (head a) (head b) ≤ Lh * dist a b) (x : X)
    (hK : ∀ i ≤ n, ∀ y, dist y (run f i x) ≤ (∏ j ∈ range n, L j) * ∑ j ∈ range n, ε j →
      y ∈ K i) :
    dist (head (run F n x)) (head (run f n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, ε i) := by
  have hL0 : ∀ i, 0 ≤ L i := fun i => by linarith [hL1 i]
  have hK' : ∀ i ≤ n, ∀ y, dist y (run f i x) ≤ errBound L ε i → y ∈ K i := fun i hi y hy =>
    hK i hi y (hy.trans (errBound_le_prod_sum_of_le L ε hL1 hε0 hi))
  have hb := model_error_bound_on F f L ε hL0 hε0 K hL hε n x (fun i hi => hK' i hi.le)
  have hF := hK' n le_rfl _ hb
  have hf := hK' n le_rfl _ (by rw [dist_self]; exact errBound_nonneg L ε hL0 hε0 n)
  exact (hLh _ hF _ hf).trans (mul_le_mul_of_nonneg_left
    (hb.trans (errBound_le_prod_sum L ε hL1 hε0 n)) hLh0)

end kit

/-! ## Identical generation from per-layer errors -/

section greedy_kit

variable {X : Type*} [PseudoMetricSpace X] {Tok : Type*} [Fintype Tok]

/-- **Identical greedy generation from per-layer errors (global).** Threshold:
`2 Lh (Π L i)(Σ ε i) < δ`, with `δ` the target's margin along its own greedy trajectory. -/
theorem greedy_of_layer_bounds (F f : ℕ → X → X) (L ε : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hε0 : ∀ i, 0 ≤ ε i) (hL : ∀ i a b, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ i y, dist (F i y) (f i y) ≤ ε i) (head : X → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh)
    (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b) (n : ℕ) (embed : List Tok → X)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run f n (embed (greedy pickT
      (fun q => head (run f n (embed q))) p k)))) a)
    (hsmall : 2 * (Lh * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, ε i)) < δ) :
    ∀ k, greedy pickH (fun q => head (run F n (embed q))) p k =
      greedy pickT (fun q => head (run f n (embed q))) p k :=
  greedy_of_bound hT hH _ _ p δ (Lh * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, ε i))
    (by linarith) hm (fun q b => by
    rw [← Real.dist_eq]
    exact (dist_le_pi_dist _ _ b).trans
      (model_bound_prod F f L ε hL1 hε0 hL hε head Lh hLh0 hLh n (embed q)))

/-- **Identical greedy generation from per-layer errors on a region.** As
`greedy_of_layer_bounds`, with the per-layer errors and Lipschitz constants only on `K i`, and the
region condition only along the target's greedy trajectory. -/
theorem greedy_of_layer_bounds_on (F f : ℕ → X → X) (L ε : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hε0 : ∀ i, 0 ≤ ε i) (K : ℕ → Set X)
    (hL : ∀ i, ∀ a ∈ K i, ∀ b ∈ K i, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ i, ∀ y ∈ K i, dist (F i y) (f i y) ≤ ε i) (head : X → Tok → ℝ) (Lh : ℝ)
    (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ K n, ∀ b ∈ K n, dist (head a) (head b) ≤ Lh * dist a b) (embed : List Tok → X)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run f n (embed (greedy pickT
      (fun q => head (run f n (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run f i (embed (greedy pickT
      (fun q => head (run f n (embed q))) p k))) ≤
        (∏ j ∈ range n, L j) * ∑ j ∈ range n, ε j → y ∈ K i)
    (hsmall : 2 * (Lh * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, ε i)) < δ) :
    ∀ k, greedy pickH (fun q => head (run F n (embed q))) p k =
      greedy pickT (fun q => head (run f n (embed q))) p k :=
  greedy_preserved hT hH _ _ p δ hm (fun k b => by
    rw [← Real.dist_eq]
    exact ((dist_le_pi_dist _ _ b).trans (model_bound_on_prod F f L ε hL1 hε0 K hL hε head Lh
      hLh0 n hLh _ (hK k))).trans_lt (by linarith))

end greedy_kit

/-! ## Limits from per-layer rates -/

section limit_kit

variable {X : Type*} [PseudoMetricSpace X] {ι : Type*} {p : Filter ι}

lemma tendsto_prod_sum_rates (L : ℕ → ℝ) (ε : ι → ℕ → ℝ)
    (hεlim : ∀ i, Tendsto (fun β => ε β i) p (𝓝 0)) (n : ℕ) :
    Tendsto (fun β => (∏ i ∈ range n, L i) * ∑ i ∈ range n, ε β i) p (𝓝 0) := by
  have : Tendsto (fun β => ∑ i ∈ range n, ε β i) p (𝓝 (∑ _i ∈ range n, (0 : ℝ))) :=
    tendsto_finsetSum _ (fun i _ => hεlim i)
  simpa using this.const_mul (∏ i ∈ range n, L i)

/-- **Whole-model limit from per-layer rates, on a region.** Target layers `L i`-Lipschitz on
`K i`, host layers within `ε β i` of them on `K i`, `ε β i → 0`. If every input in `A` keeps an
`r`-neighbourhood (`r > 0`) of its target trajectory inside the `K i`, the host stack converges to
the target stack uniformly on `A`. -/
theorem model_limit_on (F : ι → ℕ → X → X) (f : ℕ → X → X) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (ε : ι → ℕ → ℝ) (hε0 : ∀ β i, 0 ≤ ε β i) (hεlim : ∀ i, Tendsto (fun β => ε β i) p (𝓝 0))
    (K : ℕ → Set X) (hL : ∀ i, ∀ a ∈ K i, ∀ b ∈ K i, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ β i, ∀ y ∈ K i, dist (F β i y) (f i y) ≤ ε β i) (n : ℕ) (A : Set X) (r : ℝ)
    (hr : 0 < r) (hK : ∀ x ∈ A, ∀ i ≤ n, ∀ y, dist y (run f i x) ≤ r → y ∈ K i) :
    TendstoUniformlyOn (fun β => run (F β) n) (run f n) p A := by
  rw [Metric.tendstoUniformlyOn_iff]
  intro e he
  have hT := tendsto_prod_sum_rates L ε hεlim n
  filter_upwards [hT.eventually (gt_mem_nhds (lt_min he hr))] with β hβ x hx
  rw [dist_comm]
  have hB : (∏ i ∈ range n, L i) * ∑ i ∈ range n, ε β i < min e r := hβ
  have hb := model_bound_on_prod (F β) f L (ε β) hL1 (hε0 β) K hL (hε β) id 1 zero_le_one
    n (fun a _ b _ => by simp) x (fun i hi y hy => hK x hx i hi y (hy.trans
      (hB.le.trans (min_le_right _ _))))
  simp only [id, one_mul] at hb
  exact hb.trans_lt (hB.trans_le (min_le_left _ _))

/-- **Whole-model limit from per-layer rates (global).** Uniform over all inputs. -/
theorem model_limit_of_rates (F : ι → ℕ → X → X) (f : ℕ → X → X) (L : ℕ → ℝ)
    (hL1 : ∀ i, 1 ≤ L i) (ε : ι → ℕ → ℝ) (hε0 : ∀ β i, 0 ≤ ε β i)
    (hεlim : ∀ i, Tendsto (fun β => ε β i) p (𝓝 0))
    (hL : ∀ i a b, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ β i y, dist (F β i y) (f i y) ≤ ε β i) (n : ℕ) :
    TendstoUniformly (fun β => run (F β) n) (run f n) p := by
  rw [← tendstoUniformlyOn_univ]
  exact model_limit_on F f L hL1 ε hε0 hεlim (fun _ => Set.univ) (fun i a _ b _ => hL i a b)
    (fun β i y _ => hε β i y) n Set.univ 1 one_pos (fun _ _ _ _ _ _ => Set.mem_univ _)

end limit_kit

end EP
