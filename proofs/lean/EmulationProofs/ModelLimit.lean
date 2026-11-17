import EmulationProofs.Activation
import EmulationProofs.Position

/-!
# Limits at the whole-model level, and identical greedy generation

The per-operation limits (`relu_silu_bound`, `relu_sub_gelu`, `alibi_sub_rope_bound`,
`sink_to_plain_limit`, …) say one sublayer converges. This file lifts them to the model:

* **qualitative** (`model_limit`): a stack of layers, each target layer continuous and each host
  layer family converging locally uniformly (e.g. uniformly) to it, converges locally uniformly,
  hence pointwise and uniformly on every compact set of inputs;
* **quantitative** (`model_error_bound`, `errBound_le_prod_sum`): target layers `L_i`-Lipschitz,
  host layer `i` within `ε_i` of target layer `i` everywhere: the output is within
  `Σ_i ε_i Π_{j>i} L_j ≤ (Π L_i)(Σ ε_i)` (all `L_i ≥ 1`);
* **identical generation** (`greedy_preserved`): if along the target's own greedy trajectory the
  top-1 logit beats every other by `δ > 0`, and the host's logits are within `< δ/2` of the
  target's at every prefix, the greedy generations are identical, token for token, forever;
* **the tie caveat** (`tie_flips`): at a margin-0 step perturbations of any size `η > 0` flip the
  pick either way, so no limit construction fixes the generated token.

A layer is an arbitrary map of the whole state (the residual of every position, so attention is
included); the stack is the plain composition `run f n = f (n-1) ∘ ⋯ ∘ f 0`.
-/

open Real Finset Filter Topology Uniformity

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-- The composition of layers `0, …, n - 1`. -/
def run {X : Type*} (f : ℕ → X → X) : ℕ → X → X
  | 0 => id
  | n + 1 => fun x => f n (run f n x)

lemma continuous_run {X : Type*} [TopologicalSpace X] (f : ℕ → X → X)
    (hf : ∀ i, Continuous (f i)) (n : ℕ) : Continuous (run f n) := by
  induction n with
  | zero => exact continuous_id
  | succ n ih => exact (hf n).comp ih

/-! ## Qualitative: locally uniform convergence passes through the stack -/

section qualitative

variable {ι X : Type*} [UniformSpace X] {p : Filter ι}

/-- The joint form of locally uniform convergence, through one more layer. -/
lemma tendsto_layer_comp {F : ι → X → X} {f : X → X} (hF : TendstoLocallyUniformly F f p)
    (hf : Continuous f) {α : Type*} {q : Filter α} (j : α → ι) (hj : Tendsto j q p)
    (A : α → X) (z : X) (hA : Tendsto A q (𝓝 z)) :
    Tendsto (fun a => F (j a) (A a)) q (𝓝 (f z)) := by
  have h1 : Tendsto (fun a => (j a, A a)) q (p ×ˢ 𝓝 z) := hj.prodMk hA
  have h2 := ((tendstoLocallyUniformly_iff_forall_tendsto.mp hF) z).comp h1
  exact ((hf.tendsto z).comp hA).congr_uniformity h2

/-- **Whole-model limit** (locally uniform). If every target layer `f i` is continuous and every
host layer family `F β i` converges to `f i` locally uniformly as `β → p` (uniform convergence,
e.g. `relu_silu_bound`, is a special case), then the host stack of any depth `n` converges to the
target stack locally uniformly — even jointly: `F β` applied at inputs `x' → x` tends to
`run f n x`. -/
theorem model_limit (F : ι → ℕ → X → X) (f : ℕ → X → X) (hf : ∀ i, Continuous (f i))
    (hF : ∀ i, TendstoLocallyUniformly (fun β => F β i) (f i) p) (n : ℕ) :
    TendstoLocallyUniformly (fun β => run (F β) n) (run f n) p := by
  have joint : ∀ x, Tendsto (fun y : ι × X => run (F y.1) n y.2) (p ×ˢ 𝓝 x) (𝓝 (run f n x)) := by
    induction n with
    | zero => intro x; exact tendsto_snd
    | succ n ih =>
      intro x
      exact tendsto_layer_comp (hF n) (hf n) Prod.fst tendsto_fst _ _ (ih x)
  refine tendstoLocallyUniformly_iff_forall_tendsto.mpr (fun x => ?_)
  have hc : Tendsto (fun y : ι × X => run f n y.2) (p ×ˢ 𝓝 x) (𝓝 (run f n x)) :=
    ((continuous_run f hf n).tendsto x).comp tendsto_snd
  exact (hc.prodMk_nhds (joint x)).mono_right (nhds_le_uniformity _)

/-- **…pointwise, at every input.** -/
theorem model_limit_pointwise (F : ι → ℕ → X → X) (f : ℕ → X → X) (hf : ∀ i, Continuous (f i))
    (hF : ∀ i, TendstoLocallyUniformly (fun β => F β i) (f i) p) (n : ℕ) (x : X) :
    Tendsto (fun β => run (F β) n x) p (𝓝 (run f n x)) :=
  (model_limit F f hf hF n).tendsto_comp ((continuous_run f hf n).continuousAt) tendsto_const_nhds

/-- **…uniformly on every compact set of inputs.** -/
theorem model_limit_compact (F : ι → ℕ → X → X) (f : ℕ → X → X) (hf : ∀ i, Continuous (f i))
    (hF : ∀ i, TendstoLocallyUniformly (fun β => F β i) (f i) p) (n : ℕ) {K : Set X}
    (hK : IsCompact K) : TendstoUniformlyOn (fun β => run (F β) n) (run f n) p K :=
  (tendstoLocallyUniformlyOn_iff_tendstoUniformlyOn_of_compact hK).mp
    ((tendstoLocallyUniformlyOn_univ.mpr (model_limit F f hf hF n)).mono (Set.subset_univ _))

/-- **…and through a continuous head** (the logits), at every input. -/
theorem model_limit_head {Y : Type*} [TopologicalSpace Y] (F : ι → ℕ → X → X) (f : ℕ → X → X)
    (hf : ∀ i, Continuous (f i)) (hF : ∀ i, TendstoLocallyUniformly (fun β => F β i) (f i) p)
    (head : X → Y) (hh : Continuous head) (n : ℕ) (x : X) :
    Tendsto (fun β => head (run (F β) n x)) p (𝓝 (head (run f n x))) :=
  (hh.tendsto _).comp (model_limit_pointwise F f hf hF n x)

end qualitative

/-! ## Quantitative: Lipschitz layers, per-layer errors -/

section quantitative

variable {X : Type*} [PseudoMetricSpace X]

/-- The propagated error: `E 0 = 0`, `E (n+1) = L n · E n + ε n`, i.e.
`E n = Σ_{i<n} ε i Π_{i<j<n} L j`. -/
def errBound (L ε : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => L n * errBound L ε n + ε n

/-- **Whole-model error bound.** Target layers `f i` are `L i`-Lipschitz; the host layer `F i`
is within `ε i` of `f i` at every state. Then the host stack is within `errBound L ε n` of the
target stack at every input. -/
theorem model_error_bound (F f : ℕ → X → X) (L ε : ℕ → ℝ) (hL0 : ∀ i, 0 ≤ L i)
    (hL : ∀ i a b, dist (f i a) (f i b) ≤ L i * dist a b) (hε : ∀ i y, dist (F i y) (f i y) ≤ ε i)
    (n : ℕ) (x : X) : dist (run F n x) (run f n x) ≤ errBound L ε n := by
  induction n with
  | zero => simp [run, errBound]
  | succ n ih =>
    show dist (F n (run F n x)) (f n (run f n x)) ≤ L n * errBound L ε n + ε n
    calc dist (F n (run F n x)) (f n (run f n x))
        ≤ dist (F n (run F n x)) (f n (run F n x)) + dist (f n (run F n x)) (f n (run f n x)) :=
          dist_triangle _ _ _
      _ ≤ ε n + L n * errBound L ε n :=
          add_le_add (hε _ _) ((hL _ _ _).trans (mul_le_mul_of_nonneg_left ih (hL0 n)))
      _ = _ := by ring

/-- **The product form.** With every `L i ≥ 1` and `ε i ≥ 0`,
`errBound L ε n ≤ (Π_{i<n} L i) · Σ_{i<n} ε i`. -/
theorem errBound_le_prod_sum (L ε : ℕ → ℝ) (hL : ∀ i, 1 ≤ L i) (hε : ∀ i, 0 ≤ ε i) (n : ℕ) :
    errBound L ε n ≤ (∏ i ∈ range n, L i) * ∑ i ∈ range n, ε i := by
  induction n with
  | zero => simp [errBound]
  | succ n ih =>
    have hP : 1 ≤ ∏ i ∈ range n, L i := by
      clear ih
      induction n with
      | zero => simp
      | succ n ihn => rw [Finset.prod_range_succ]; nlinarith [hL n]
    have hS : 0 ≤ ∑ i ∈ range n, ε i := Finset.sum_nonneg (fun i _ => hε i)
    have hLn := hL n
    rw [errBound, Finset.prod_range_succ, Finset.sum_range_succ]
    have h1 : L n * errBound L ε n ≤ L n * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, ε i) :=
      mul_le_mul_of_nonneg_left ih (by linarith)
    have h2 : ε n ≤ (∏ i ∈ range n, L i) * L n * ε n := by
      have : 1 ≤ (∏ i ∈ range n, L i) * L n := by nlinarith
      nlinarith [hε n]
    nlinarith

/-- **…through a Lipschitz head** (the logits). -/
theorem model_error_bound_head {Y : Type*} [PseudoMetricSpace Y] (F f : ℕ → X → X)
    (L ε : ℕ → ℝ) (hL0 : ∀ i, 0 ≤ L i) (hL : ∀ i a b, dist (f i a) (f i b) ≤ L i * dist a b)
    (hε : ∀ i y, dist (F i y) (f i y) ≤ ε i) (head : X → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh)
    (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b) (n : ℕ) (x : X) :
    dist (head (run F n x)) (head (run f n x)) ≤ Lh * errBound L ε n :=
  (hLh _ _).trans (mul_le_mul_of_nonneg_left (model_error_bound F f L ε hL0 hL hε n x) hLh0)

end quantitative

/-! ## Instance: ReLU ⊆ SiLU in every MLP of the stack -/

section silu

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- A target layer: an arbitrary map `A` (the residual, attention, norms, …) plus a ReLU MLP with
pre-activations `pre u x` and down-projection columns `w u`. -/
def reluLayer {m : ℕ} (A : V → V) (pre : Fin m → V → ℝ) (w : Fin m → V) (x : V) : V :=
  A x + ∑ u, relu (pre u x) • w u

/-- Its host: the same layer with every `ReLU` replaced by `SiLU(β ·)/β` (the `β`, `1/β` fold
into the MLP's first and second matrices). -/
def siluLayer {m : ℕ} (β : ℝ) (A : V → V) (pre : Fin m → V → ℝ) (w : Fin m → V) (x : V) : V :=
  A x + ∑ u, (silu (β * pre u x) / β) • w u

/-- **One layer, uniformly in the state:** `‖siluLayer − reluLayer‖ ≤ (Σ ‖w u‖)/β`. -/
theorem silu_layer_bound {m : ℕ} (β : ℝ) (hβ : 0 < β) (A : V → V) (pre : Fin m → V → ℝ)
    (w : Fin m → V) (x : V) :
    dist (siluLayer β A pre w x) (reluLayer A pre w x) ≤ (∑ u, ‖w u‖) / β := by
  rw [dist_eq_norm, siluLayer, reluLayer, add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib,
    Finset.sum_div]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum (fun u _ => ?_))
  rw [← sub_smul, norm_smul, Real.norm_eq_abs]
  calc |silu (β * pre u x) / β - relu (pre u x)| * ‖w u‖ ≤ 1 / β * ‖w u‖ :=
        mul_le_mul_of_nonneg_right (relu_silu_bound β hβ _) (norm_nonneg _)
    _ = ‖w u‖ / β := by ring

lemma continuous_relu : Continuous relu := continuous_id.max continuous_const

/-- **ReLU ⊆ SiLU, the whole model, the rate.** A stack of `n` layers `A i + ReLU-MLP i`, target
layer `i` `L i`-Lipschitz (`L i ≥ 1`), head `Lh`-Lipschitz: the SiLU host's logits are within
`Lh (Π L i)(Σ_i Σ_u ‖w i u‖)/β` of the target's, at every input. -/
theorem relu_silu_model_bound {Y : Type*} [PseudoMetricSpace Y] (β : ℝ) (hβ : 0 < β)
    (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ) (w : ∀ i, Fin (m i) → V)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (reluLayer (A i) (pre i) (w i) a) (reluLayer (A i) (pre i) (w i) b) ≤
      L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b)
    (n : ℕ) (x : V) :
    dist (head (run (fun i => siluLayer β (A i) (pre i) (w i)) n x))
        (head (run (fun i => reluLayer (A i) (pre i) (w i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, (∑ u, ‖w i u‖) / β) := by
  refine (model_error_bound_head _ _ L (fun i => (∑ u, ‖w i u‖) / β)
    (fun i => by linarith [hL1 i]) hL (fun i y => silu_layer_bound β hβ _ _ _ y) head Lh hLh0 hLh
    n x).trans (mul_le_mul_of_nonneg_left ?_ hLh0)
  exact errBound_le_prod_sum _ _ hL1
    (fun i => div_nonneg (Finset.sum_nonneg (fun u _ => norm_nonneg _)) hβ.le) n

/-- **ReLU ⊆ SiLU, the whole model, the limit.** With continuous `A i` and pre-activations the
SiLU host stack converges to the ReLU target stack locally uniformly as `β → ∞` (so pointwise,
and uniformly on compact input sets, by `model_limit_pointwise`, `model_limit_compact`). -/
theorem relu_silu_model_limit (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ)
    (w : ∀ i, Fin (m i) → V) (hA : ∀ i, Continuous (A i))
    (hpre : ∀ i u, Continuous (pre i u)) (n : ℕ) :
    TendstoLocallyUniformly (fun β => run (fun i => siluLayer β (A i) (pre i) (w i)) n)
      (run (fun i => reluLayer (A i) (pre i) (w i)) n) atTop := by
  refine model_limit _ _ (fun i => ?_) (fun i => ?_) n
  · exact (hA i).add (continuous_finsetSum _ (fun u _ =>
      (continuous_relu.comp (hpre i u)).smul continuous_const))
  · refine TendstoUniformly.tendstoLocallyUniformly ?_
    rw [Metric.tendstoUniformly_iff]
    intro ε hε
    have hC : Tendsto (fun β : ℝ => (∑ u, ‖w i u‖) / β) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop tendsto_id
    filter_upwards [hC.eventually (gt_mem_nhds hε), eventually_gt_atTop 0] with β h1 h2 x
    rw [dist_comm]
    exact (silu_layer_bound β h2 _ _ _ x).trans_lt h1

end silu

/-! ## Identical greedy generation under a margin -/

section greedy

variable {Tok : Type*}

/-- A pick rule returns a maximiser (any tie-break). -/
def IsPick (pick : (Tok → ℝ) → Tok) : Prop := ∀ (z : Tok → ℝ) (b : Tok), z b ≤ z (pick z)

/-- `a` beats every other token by at least `δ`. -/
def MarginAt (δ : ℝ) (z : Tok → ℝ) (a : Tok) : Prop := ∀ b, b ≠ a → z b + δ ≤ z a

/-- With a margin `δ > 0`, every pick of every `z'` within `< δ/2` of `z` is the target's
winner. -/
theorem pick_eq_of_margin {pick : (Tok → ℝ) → Tok} (hp : IsPick pick) {δ : ℝ} {z z' : Tok → ℝ}
    {a : Tok} (hm : MarginAt δ z a) (hz : ∀ b, |z' b - z b| < δ / 2) : pick z' = a := by
  by_contra hc
  have h1 := hm _ hc
  have h2 := hp z' a
  have h3 := (abs_lt.mp (hz a)).1
  have h4 := (abs_lt.mp (hz (pick z'))).2
  linarith

/-- Greedy generation: append the picked token, `k` times. -/
def greedy (pick : (Tok → ℝ) → Tok) (lg : List Tok → Tok → ℝ) (p : List Tok) : ℕ → List Tok
  | 0 => p
  | k + 1 => greedy pick lg p k ++ [pick (lg (greedy pick lg p k))]

/-- **Identical greedy generation** (`greedy_preserved`). Along the target's own greedy trajectory,
suppose every step has a winner with margin `δ > 0`, and the host's logits there are within
`< δ/2` of the target's (sup norm). Then the host's greedy generation equals the target's at every
length, whatever tie-breaks either side uses. -/
theorem greedy_preserved {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH)
    (lgT lgH : List Tok → Tok → ℝ) (p : List Tok) (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (lgT (greedy pickT lgT p k)) a)
    (hclose : ∀ k b, |lgH (greedy pickT lgT p k) b - lgT (greedy pickT lgT p k) b| < δ / 2) :
    ∀ k, greedy pickH lgH p k = greedy pickT lgT p k := by
  intro k
  induction k with
  | zero => rfl
  | succ k ih =>
    obtain ⟨a, ha⟩ := hm k
    have hδ : 0 < δ := by
      have := hclose k a; have := abs_nonneg (lgH (greedy pickT lgT p k) a -
        lgT (greedy pickT lgT p k) a); linarith
    have e1 : pickH (lgH (greedy pickT lgT p k)) = a := pick_eq_of_margin hH ha (hclose k)
    have e2 : pickT (lgT (greedy pickT lgT p k)) = a :=
      pick_eq_of_margin hT ha (fun b => by simpa using half_pos hδ)
    simp only [greedy, ih, e1, e2]

/-- **…from a uniform logit bound.** If the host's logits are within `E < δ/2` of the target's at
every prefix (e.g. `E` = a whole-model bound) and the target has margin `δ` along its trajectory,
the generations are identical. -/
theorem greedy_of_bound {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH)
    (lgT lgH : List Tok → Tok → ℝ) (p : List Tok) (δ E : ℝ) (hE : E < δ / 2)
    (hm : ∀ k, ∃ a, MarginAt δ (lgT (greedy pickT lgT p k)) a)
    (hb : ∀ q b, |lgH q b - lgT q b| ≤ E) :
    ∀ k, greedy pickH lgH p k = greedy pickT lgT p k :=
  greedy_preserved hT hH lgT lgH p δ hm (fun _ b => (hb _ b).trans_lt hE)

/-- **The tie caveat.** At a margin-0 step (`z a = z b`, `a ≠ b`), for every `η > 0` there are
logits within `η` of `z` on which a pick cannot return `b`, and others on which it cannot return
`a`: a host converging to the target can approach from either side, so the limit does not fix the
generated token (cf. `soft_router_tie`). -/
theorem tie_flips [DecidableEq Tok] {pick : (Tok → ℝ) → Tok} (hp : IsPick pick) (z : Tok → ℝ)
    {a b : Tok} (hab : a ≠ b) (htie : z a = z b) (η : ℝ) (hη : 0 < η) :
    ∃ z₁ z₂ : Tok → ℝ, (∀ c, |z₁ c - z c| < η) ∧ (∀ c, |z₂ c - z c| < η) ∧
      pick z₁ ≠ b ∧ pick z₂ ≠ a := by
  refine ⟨fun c => z c + if c = a then η / 2 else 0, fun c => z c + if c = b then η / 2 else 0,
    fun c => ?_, fun c => ?_, fun h => ?_, fun h => ?_⟩
  · show |z c + (if c = a then η / 2 else 0) - z c| < η
    split_ifs <;> simp [abs_of_pos (half_pos hη), hη]
  · show |z c + (if c = b then η / 2 else 0) - z c| < η
    split_ifs <;> simp [abs_of_pos (half_pos hη), hη]
  · have := hp (fun c => z c + if c = a then η / 2 else 0) a
    rw [h] at this
    simp [hab.symm, htie] at this; linarith
  · have := hp (fun c => z c + if c = b then η / 2 else 0) b
    rw [h] at this
    simp [hab, htie] at this; linarith

end greedy

/-! ## Explicit thresholds: ReLU ⊆ SiLU -/

section silu_greedy

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {Tok : Type*} [Fintype Tok]

/-- **ReLU ⊆ SiLU, identical generation, explicit `β*`.** Target: the ReLU stack of
`relu_silu_model_bound`, read by an `Lh`-Lipschitz head into logits (sup norm) from an arbitrary
embedding of the prefix. If along the target's greedy trajectory every step has margin `δ`, then
for every `β > β* = 2 Lh (Π L i)(Σ_i Σ_u ‖w i u‖)/δ` the SiLU host generates exactly the same
tokens, at every length. -/
theorem relu_silu_greedy (m : ℕ → ℕ) (A : ℕ → V → V) (pre : ∀ i, Fin (m i) → V → ℝ)
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
    (β : ℝ) (hβ : 2 * Lh * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, ∑ u, ‖w i u‖) / δ < β) :
    ∀ k, greedy pickH (fun q => head (run (fun i => siluLayer β (A i) (pre i) (w i)) n (embed q)))
        p k =
      greedy pickT (fun q => head (run (fun i => reluLayer (A i) (pre i) (w i)) n (embed q))) p k := by
  set C := Lh * ((∏ i ∈ range n, L i) * ∑ i ∈ range n, ∑ u, ‖w i u‖)
  have hC0 : 0 ≤ C := by
    have : 0 ≤ ∏ i ∈ range n, L i := Finset.prod_nonneg (fun i _ => by linarith [hL1 i])
    have : 0 ≤ ∑ i ∈ range n, ∑ u, ‖w i u‖ :=
      Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun u _ => norm_nonneg _))
    positivity
  have hβpos : 0 < β := lt_of_le_of_lt (by rw [mul_assoc]; positivity) hβ
  have hE : C / β < δ / 2 := by
    rw [div_lt_iff₀ hβpos]
    rw [mul_assoc, div_lt_iff₀ hδ] at hβ
    linarith
  refine greedy_of_bound hT hH _ _ p δ (C / β) hE hm (fun q b => ?_)
  rw [← Real.dist_eq]
  refine (dist_le_pi_dist _ _ b).trans ((relu_silu_model_bound β hβpos m A pre w L hL1 hL head
    Lh hLh0 hLh n (embed q)).trans (le_of_eq ?_))
  simp only [C, ← Finset.sum_div]
  ring

end silu_greedy

/-! ## Instance: ALiBi ⊆ RoPE in every attention layer of a bounded-context stack -/

section alibi

/-- Softmax weights over `S` move by at most a factor `e^{2η}` when every logit moves by `≤ η`. -/
lemma attn_weight_le {ι : Type*} [Fintype ι] (S : Finset ι) (hS : S.Nonempty) (s t : ι → ℝ) (η : ℝ)
    (h : ∀ j ∈ S, |s j - t j| ≤ η) (j : ι) (hj : j ∈ S) :
    exp (s j) / ∑ l ∈ S, exp (s l) ≤ exp (2 * η) * (exp (t j) / ∑ l ∈ S, exp (t l)) := by
  have ht := sum_exp_pos hS t
  have hs := sum_exp_pos hS s
  have hnum : exp (s j) ≤ exp η * exp (t j) := by
    rw [← exp_add]; apply exp_le_exp.mpr; linarith [(abs_le.mp (h j hj)).2]
  have hden : exp (-η) * ∑ l ∈ S, exp (t l) ≤ ∑ l ∈ S, exp (s l) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum (fun l hl => ?_)
    rw [← exp_add]; apply exp_le_exp.mpr; linarith [(abs_le.mp (h l hl)).1]
  rw [div_le_iff₀ hs, two_mul, exp_add]
  calc exp (s j) ≤ exp η * exp (t j) := hnum
    _ = exp η * exp η * (exp (t j) / ∑ l ∈ S, exp (t l)) * (exp (-η) * ∑ l ∈ S, exp (t l)) := by
        rw [exp_neg]; field_simp
    _ ≤ exp η * exp η * (exp (t j) / ∑ l ∈ S, exp (t l)) * ∑ l ∈ S, exp (s l) := by
        gcongr

lemma attn_eq_weights {ι V : Type*} [Fintype ι] [NormedAddCommGroup V] [NormedSpace ℝ V] (S : Finset ι)
    (s : ι → ℝ) (v : ι → V) :
    attn S s v = ∑ j ∈ S, (exp (s j) / ∑ l ∈ S, exp (s l)) • v j := by
  rw [attn, Finset.smul_sum]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [smul_smul, div_eq_inv_mul]

/-- **Logit perturbation → attention-output perturbation.** If every logit over `S` moves by at
most `η` and every value has norm `≤ M`, the attention output moves by at most `(e^{2η} − 1) M`. -/
theorem attn_perturb {ι V : Type*} [Fintype ι] [NormedAddCommGroup V] [NormedSpace ℝ V] (S : Finset ι)
    (hS : S.Nonempty) (s t : ι → ℝ) (η : ℝ) (h : ∀ j ∈ S, |s j - t j| ≤ η) (v : ι → V) (M : ℝ)
    (hv : ∀ j ∈ S, ‖v j‖ ≤ M) :
    ‖attn S s v - attn S t v‖ ≤ (exp (2 * η) - 1) * M := by
  have hη : 0 ≤ η := by
    obtain ⟨j, hj⟩ := hS; exact (abs_nonneg _).trans (h j hj)
  have h' : ∀ j ∈ S, |t j - s j| ≤ η := fun j hj => by rw [abs_sub_comm]; exact h j hj
  set ws := fun j => exp (s j) / ∑ l ∈ S, exp (s l)
  set wt := fun j => exp (t j) / ∑ l ∈ S, exp (t l)
  have hwt : ∑ j ∈ S, wt j = 1 := by
    simp only [wt, ← Finset.sum_div]; exact div_self (sum_exp_pos hS t).ne'
  have hE : 1 ≤ exp (2 * η) := one_le_exp (by linarith)
  have hinv : exp (-(2 * η)) * exp (2 * η) = 1 := by rw [← exp_add]; simp
  have hdiff : ∀ j ∈ S, |ws j - wt j| ≤ (exp (2 * η) - 1) * wt j := by
    intro j hj
    have h1 := attn_weight_le S hS s t η h j hj
    have h2 := attn_weight_le S hS t s η h' j hj
    have hwt0 : 0 ≤ wt j := by positivity
    have hws0 : 0 ≤ ws j := by positivity
    change ws j ≤ exp (2 * η) * wt j at h1
    change wt j ≤ exp (2 * η) * ws j at h2
    rw [abs_le]; constructor
    · -- wt ≤ e^{2η} ws, so wt - ws ≤ (1 - e^{-2η}) wt ≤ (e^{2η} - 1) wt
      have hpos := exp_pos (2 * η)
      nlinarith [mul_le_mul_of_nonneg_left h2 hwt0, sq_nonneg (exp (2 * η) - 1)]
    · linarith
  rw [attn_eq_weights, attn_eq_weights, ← Finset.sum_sub_distrib]
  calc ‖∑ j ∈ S, ((ws j) • v j - (wt j) • v j)‖ ≤ ∑ j ∈ S, ‖(ws j - wt j) • v j‖ := by
        refine (norm_sum_le _ _).trans (le_of_eq (Finset.sum_congr rfl (fun j _ => ?_)))
        rw [sub_smul]
    _ ≤ ∑ j ∈ S, (exp (2 * η) - 1) * wt j * M := by
        refine Finset.sum_le_sum (fun j hj => ?_)
        rw [norm_smul, Real.norm_eq_abs]
        exact mul_le_mul (hdiff j hj) (hv j hj) (norm_nonneg _) (by
          have : 0 ≤ wt j := by positivity
          nlinarith)
    _ = (exp (2 * η) - 1) * M := by
        rw [← Finset.sum_mul, ← Finset.mul_sum, hwt, mul_one]

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {Pr : Type*} [Fintype Pr] {T : ℕ}

/-- The causal key set of query `t`. -/
def causalSet (t : Fin T) : Finset (Fin T) := univ.filter (fun j => j ≤ t)

lemma causalSet_nonempty (t : Fin T) : (causalSet t).Nonempty := ⟨t, by simp [causalSet]⟩

/-- A causal ALiBi attention layer (one head; the output projection is folded into `val`) on a
context of `T` positions, slope `m`, query / key maps `q, k` (pairs of channels), plus the
residual. -/
def alibiLayer (m : ℝ) (q k : V → Pr → ℝ × ℝ) (val : V → V) (x : Fin T → V) : Fin T → V :=
  fun t => x t + attn (causalSet t)
    (fun j => alibiScore (∑ p, dot2 (q (x t) p) (k (x j) p)) m (t : ℕ) (j : ℕ)) (fun j => val (x j))

/-- Its RoPE host: the frozen-pair construction of `alibi_host_score` at frequency `θ`. -/
def ropeAlibiLayer (θ m : ℝ) (q k : V → Pr → ℝ × ℝ) (val : V → V) (x : Fin T → V) :
    Fin T → V :=
  fun t => x t + attn (causalSet t)
    (fun j => ropeScore (alibiHostFreq θ) (alibiHostQ m θ (q (x t))) (alibiHostK (k (x j)))
      (t : ℕ) (j : ℕ)) (fun j => val (x j))

/-- **One ALiBi layer from RoPE, uniformly in the state.** With values bounded by `M`, the host
layer is within `(e^{2η} − 1) M`, `η = |m| T³ θ² / 6`, of the target at every state. -/
theorem rope_alibi_layer_bound (θ : ℝ) (hθ : θ ≠ 0) (m : ℝ) (q k : V → Pr → ℝ × ℝ)
    (val : V → V) (M : ℝ) (hM : ∀ y, ‖val y‖ ≤ M) (x : Fin T → V) :
    dist (ropeAlibiLayer θ m q k val x) (alibiLayer m q k val x) ≤
      (exp (2 * (|m| * (T : ℝ) ^ 3 * θ ^ 2 / 6)) - 1) * M := by
  have hM0 : 0 ≤ M := by
    rcases T.eq_zero_or_pos with hT | hT
    · exact (norm_nonneg _).trans (hM 0)
    · exact (norm_nonneg _).trans (hM 0)
  have hη : 0 ≤ |m| * (T : ℝ) ^ 3 * θ ^ 2 / 6 := by positivity
  refine (dist_pi_le_iff (mul_nonneg (by linarith [one_le_exp (by linarith : 0 ≤ 2 *
    (|m| * (T : ℝ) ^ 3 * θ ^ 2 / 6))]) hM0)).mpr (fun t => ?_)
  rw [dist_eq_norm, ropeAlibiLayer, alibiLayer, add_sub_add_left_eq_sub]
  refine attn_perturb _ (causalSet_nonempty t) _ _ _ (fun j _ => ?_) _ M (fun j _ => hM _)
  refine (alibi_sub_rope_bound m θ hθ _ _ _ _).trans ?_
  have hd : |((t : ℕ) : ℝ) - ((j : ℕ) : ℝ)| ≤ T := by
    rw [abs_le]
    have := t.isLt; have := j.isLt
    constructor
    · have : ((t : ℕ) : ℝ) ≥ 0 := by positivity
      have : ((j : ℕ) : ℝ) < T := by exact_mod_cast j.isLt
      linarith
    · have : ((j : ℕ) : ℝ) ≥ 0 := by positivity
      have : ((t : ℕ) : ℝ) < T := by exact_mod_cast t.isLt
      linarith
  have : |((t : ℕ) : ℝ) - ((j : ℕ) : ℝ)| ^ 3 ≤ (T : ℝ) ^ 3 :=
    pow_le_pow_left₀ (abs_nonneg _) hd 3
  have := sq_nonneg θ
  have := abs_nonneg m
  gcongr

/-- `e^{2η} − 1 ≤ 4η` for `0 ≤ 2η ≤ 1`. -/
lemma exp_two_mul_sub_one_le (η : ℝ) (h0 : 0 ≤ η) (h1 : 2 * η ≤ 1) :
    exp (2 * η) - 1 ≤ 4 * η := by
  have := Real.abs_exp_sub_one_le (x := 2 * η) (by rw [abs_of_nonneg (by linarith)]; exact h1)
  have he : 0 ≤ exp (2 * η) - 1 := by linarith [one_le_exp (by linarith : 0 ≤ 2 * η)]
  rw [abs_of_nonneg he, abs_of_nonneg (by linarith : (0 : ℝ) ≤ 2 * η)] at this
  linarith

/-- **ALiBi ⊆ RoPE, the whole model, the rate.** A stack of `n` causal ALiBi layers over `T`
positions (slopes `|m i| ≤ m̄`, values bounded by `M i ≤ M̄`, target layer `i` `L i`-Lipschitz,
`L i ≥ 1`), read by an `Lh`-Lipschitz head. For `θ ≠ 0` with `m̄ T³ θ² ≤ 3`, the RoPE host's logits
are within `Lh (Π L i) · n · (2/3) m̄ T³ θ² M̄` of the target's, at every input: `O(θ²)`. -/
theorem alibi_rope_model_bound {Y : Type*} [PseudoMetricSpace Y] (θ : ℝ) (hθ : θ ≠ 0)
    (m : ℕ → ℝ) (q k : ℕ → V → Pr → ℝ × ℝ) (val : ℕ → V → V) (M : ℕ → ℝ)
    (hM : ∀ i y, ‖val i y‖ ≤ M i) (mbar Mbar : ℝ) (hm : ∀ i, |m i| ≤ mbar) (hMb : ∀ i, M i ≤ Mbar)
    (hsmall : mbar * (T : ℝ) ^ 3 * θ ^ 2 ≤ 3) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (alibiLayer (T := T) (m i) (q i) (k i) (val i) a)
      (alibiLayer (m i) (q i) (k i) (val i) b) ≤ L i * dist a b)
    (head : (Fin T → V) → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh)
    (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b) (n : ℕ) (x : Fin T → V) :
    dist (head (run (fun i => ropeAlibiLayer θ (m i) (q i) (k i) (val i)) n x))
        (head (run (fun i => alibiLayer (m i) (q i) (k i) (val i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * (n * (2 / 3 * mbar * (T : ℝ) ^ 3 * θ ^ 2 * Mbar))) := by
  set ε : ℕ → ℝ := fun i => (exp (2 * (|m i| * (T : ℝ) ^ 3 * θ ^ 2 / 6)) - 1) * M i
  have hM0 : ∀ i, 0 ≤ M i := fun i => (norm_nonneg _).trans (hM i 0)
  have hε0 : ∀ i, 0 ≤ ε i := fun i => mul_nonneg (by linarith [one_le_exp (by positivity :
    0 ≤ 2 * (|m i| * (T : ℝ) ^ 3 * θ ^ 2 / 6))]) (hM0 i)
  have hεle : ∀ i, ε i ≤ 2 / 3 * mbar * (T : ℝ) ^ 3 * θ ^ 2 * Mbar := by
    intro i
    have hη0 : 0 ≤ |m i| * (T : ℝ) ^ 3 * θ ^ 2 / 6 := by positivity
    have hTθ : 0 ≤ (T : ℝ) ^ 3 * θ ^ 2 := by positivity
    have hmi : |m i| * ((T : ℝ) ^ 3 * θ ^ 2) ≤ mbar * ((T : ℝ) ^ 3 * θ ^ 2) :=
      mul_le_mul_of_nonneg_right (hm i) hTθ
    have hη1 : 2 * (|m i| * (T : ℝ) ^ 3 * θ ^ 2 / 6) ≤ 1 := by nlinarith
    have he := exp_two_mul_sub_one_le _ hη0 hη1
    have hpos : 0 ≤ exp (2 * (|m i| * (T : ℝ) ^ 3 * θ ^ 2 / 6)) - 1 := by
      linarith [one_le_exp (by positivity : 0 ≤ 2 * (|m i| * (T : ℝ) ^ 3 * θ ^ 2 / 6))]
    calc ε i ≤ (4 * (|m i| * (T : ℝ) ^ 3 * θ ^ 2 / 6)) * Mbar :=
          mul_le_mul he (hMb i) (hM0 i) (by linarith)
      _ ≤ 2 / 3 * mbar * (T : ℝ) ^ 3 * θ ^ 2 * Mbar := by
          have : 0 ≤ Mbar := (hM0 i).trans (hMb i)
          nlinarith
  refine (model_error_bound_head _ _ L ε (fun i => by linarith [hL1 i]) hL
    (fun i y => rope_alibi_layer_bound θ hθ _ _ _ _ _ (hM i) y) head Lh hLh0 hLh n x).trans
    (mul_le_mul_of_nonneg_left ?_ hLh0)
  refine (errBound_le_prod_sum L ε hL1 hε0 n).trans (mul_le_mul_of_nonneg_left ?_
    (Finset.prod_nonneg (fun i _ => by linarith [hL1 i])))
  calc ∑ i ∈ range n, ε i ≤ ∑ _i ∈ range n, (2 / 3 * mbar * (T : ℝ) ^ 3 * θ ^ 2 * Mbar) :=
        Finset.sum_le_sum (fun i _ => hεle i)
    _ = _ := by simp

/-- **ALiBi ⊆ RoPE, the whole model, the limit.** Under the hypotheses of
`alibi_rope_model_bound` (bounded values, Lipschitz target layers, context `T`), the RoPE host
stack converges to the ALiBi target stack uniformly over ALL inputs as `θ → 0`. -/
theorem alibi_rope_model_limit (m : ℕ → ℝ) (q k : ℕ → V → Pr → ℝ × ℝ) (val : ℕ → V → V)
    (M : ℕ → ℝ) (hM : ∀ i y, ‖val i y‖ ≤ M i) (mbar Mbar : ℝ) (hm : ∀ i, |m i| ≤ mbar)
    (hMb : ∀ i, M i ≤ Mbar) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (alibiLayer (T := T) (m i) (q i) (k i) (val i) a)
      (alibiLayer (m i) (q i) (k i) (val i) b) ≤ L i * dist a b) (n : ℕ) :
    TendstoUniformly (fun θ => run (fun i => ropeAlibiLayer (T := T) θ (m i) (q i) (k i) (val i)) n)
      (run (fun i => alibiLayer (m i) (q i) (k i) (val i)) n) (𝓝[≠] 0) := by
  rw [Metric.tendstoUniformly_iff]
  intro ε hε
  set C := 1 * ((∏ i ∈ range n, L i) * (n * (2 / 3 * mbar * (T : ℝ) ^ 3 * Mbar)))
  have hcont : ∀ c : ℝ, Tendsto (fun θ : ℝ => c * θ ^ 2) (𝓝[≠] 0) (𝓝 0) := fun c => by
    have : Tendsto (fun θ : ℝ => c * θ ^ 2) (𝓝 0) (𝓝 (c * 0 ^ 2)) :=
      (continuous_const.mul (continuous_pow 2)).tendsto 0
    simpa using this.mono_left nhdsWithin_le_nhds
  filter_upwards [self_mem_nhdsWithin, (hcont C).eventually (gt_mem_nhds hε),
    (hcont (mbar * (T : ℝ) ^ 3)).eventually (gt_mem_nhds (by norm_num : (0 : ℝ) < 3))]
    with θ hθ h1 h2 x
  rw [dist_comm]
  have hb := alibi_rope_model_bound θ hθ m q k val M hM mbar Mbar hm hMb h2.le L hL1 hL id 1
    zero_le_one (fun a b => by simp) n x
  simp only [id] at hb
  refine hb.trans_lt (lt_of_eq_of_lt ?_ h1)
  simp only [C]; ring

end alibi

/-! ## Explicit thresholds: ALiBi ⊆ RoPE -/

section alibi_greedy

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {Pr : Type*} [Fintype Pr] {T : ℕ}
  {Tok : Type*} [Fintype Tok]

/-- **ALiBi ⊆ RoPE, identical generation, explicit `θ*`.** In the setting of
`alibi_rope_model_bound` (context `T`, logits through an `Lh`-Lipschitz head), if along the
target's greedy trajectory every step has margin `δ`, then every `θ ≠ 0` with
`m̄ T³ θ² ≤ 3` and `θ² < θ*² = 3δ / (4 Lh (Π L i) n m̄ T³ M̄)` gives exactly the same tokens. -/
theorem alibi_rope_greedy (θ : ℝ) (hθ : θ ≠ 0) (m : ℕ → ℝ) (q k : ℕ → V → Pr → ℝ × ℝ)
    (val : ℕ → V → V) (M : ℕ → ℝ) (hM : ∀ i y, ‖val i y‖ ≤ M i) (mbar Mbar : ℝ)
    (hm : ∀ i, |m i| ≤ mbar) (hMb : ∀ i, M i ≤ Mbar) (hsmall : mbar * (T : ℝ) ^ 3 * θ ^ 2 ≤ 3)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i a b, dist (alibiLayer (T := T) (m i) (q i) (k i) (val i) a)
      (alibiLayer (m i) (q i) (k i) (val i) b) ≤ L i * dist a b)
    (head : (Fin T → V) → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh)
    (hLh : ∀ a b, dist (head a) (head b) ≤ Lh * dist a b) (n : ℕ) (embed : List Tok → Fin T → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hmar : ∀ k', ∃ a, MarginAt δ (head (run (fun i => alibiLayer (m i) (q i) (k i) (val i)) n
      (embed (greedy pickT (fun q' => head (run (fun i => alibiLayer (m i) (q i) (k i) (val i)) n
        (embed q'))) p k')))) a)
    (hθ2 : 4 * Lh * (∏ i ∈ range n, L i) * n * mbar * (T : ℝ) ^ 3 * Mbar * θ ^ 2 < 3 * δ) :
    ∀ k', greedy pickH (fun q' => head (run (fun i => ropeAlibiLayer θ (m i) (q i) (k i) (val i)) n
        (embed q'))) p k' =
      greedy pickT (fun q' => head (run (fun i => alibiLayer (m i) (q i) (k i) (val i)) n
        (embed q'))) p k' := by
  refine greedy_of_bound hT hH _ _ p δ
    (Lh * ((∏ i ∈ range n, L i) * (n * (2 / 3 * mbar * (T : ℝ) ^ 3 * θ ^ 2 * Mbar))))
    (by nlinarith) hmar (fun q' b => ?_)
  rw [← Real.dist_eq]
  exact (dist_le_pi_dist _ _ b).trans (alibi_rope_model_bound θ hθ m q k val M hM mbar Mbar hm hMb
    hsmall L hL1 hL head Lh hLh0 hLh n (embed q'))

end alibi_greedy

end EP
