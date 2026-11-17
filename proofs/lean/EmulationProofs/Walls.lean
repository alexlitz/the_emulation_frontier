import EmulationProofs.Position
import EmulationProofs.Activation
import EmulationProofs.Attention

/-!
# Walls and depth laws (the "irreducible" section and the exchange-rate table)

* No analytic host activation (SiLU, swigluoai, any `a·SiLU(cx+d) + ex + f`) equals ReLU or
  squared-ReLU EXACTLY on any interval around `0`: ReLU ⊆ SiLU can only be a limit, and every
  single-op fit of ReLU / relu² by a swish curve has a nonzero floor.
* A relative (RoPE) logit is shift-invariant, so a single RoPE layer cannot produce a learned
  absolute logit field; but a causal uniform head computes `1/(t+1)`, so absolute position is
  recoverable in a deeper causal host (built exactly in `LearnedAbs.lean`, given a BOS marker).
* A causal model cannot reproduce a bidirectional one.
* Stacked local windows: `L` layers of window `w` reach exactly `L (w - 1)` positions back, so
  covering a context `T` needs `L ≥ ⌈(T-1)/(w-1)⌉`.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## Analytic hosts never hit ReLU or relu² exactly -/

/-- An analytic function that vanishes on `(-r, 0)` cannot equal a target that is nonzero
somewhere in `(0, r)`. -/
theorem analytic_ne_kink (g t : ℝ → ℝ) (hg : AnalyticOnNhd ℝ g Set.univ) (r : ℝ) (hr : 0 < r)
    (ht0 : ∀ x ∈ Set.Ioo (-r) 0, t x = 0) (x₀ : ℝ) (hx₀ : x₀ ∈ Set.Ioo 0 r) (ht : t x₀ ≠ 0) :
    ¬ ∀ x ∈ Set.Ioo (-r) r, g x = t x := by
  intro h
  have hev : g =ᶠ[𝓝 (-r / 2)] (fun _ => (0 : ℝ)) := by
    have : Set.Ioo (-r) 0 ∈ 𝓝 (-r / 2) := Ioo_mem_nhds (by linarith) (by linarith)
    filter_upwards [this] with x hx
    rw [h x ⟨hx.1, by linarith [hx.2]⟩, ht0 x hx]
  have := hg.eqOn_of_preconnected_of_eventuallyEq (analyticOnNhd_const) isPreconnected_univ
    (Set.mem_univ _) hev (Set.mem_univ x₀)
  rw [h x₀ ⟨by linarith [hx₀.1], hx₀.2⟩] at this
  exact ht this

/-- The SiLU single-op family `a·SiLU(cx + d) + ex + f` is analytic. -/
lemma silu_family_analytic (a c d e f : ℝ) :
    AnalyticOnNhd ℝ (fun x => a * silu (c * x + d) + e * x + f) Set.univ := by
  intro x _
  simp only [silu]
  fun_prop

/-- **ReLU ⊆ SiLU is necessarily a limit** (impossibility, proved). No member of the SiLU
single-op family (affine maps before and after; this includes swigluoai) equals ReLU on any
interval `(-r, r)`. -/
theorem relu_not_exact_silu (a c d e f r : ℝ) (hr : 0 < r) :
    ¬ ∀ x ∈ Set.Ioo (-r) r, a * silu (c * x + d) + e * x + f = relu x :=
  analytic_ne_kink _ relu (silu_family_analytic a c d e f) r hr
    (fun x hx => max_eq_right hx.2.le) (r / 2) ⟨by linarith, by linarith⟩
    (by simp only [relu]; rw [max_eq_left (by linarith)]; linarith)

/-- **relu² from a swish curve has a nonzero floor** (impossibility of exactness, proved). -/
theorem relu_sq_not_exact_silu (a c d e f r : ℝ) (hr : 0 < r) :
    ¬ ∀ x ∈ Set.Ioo (-r) r, a * silu (c * x + d) + e * x + f = relu x ^ 2 :=
  analytic_ne_kink _ (fun x => relu x ^ 2) (silu_family_analytic a c d e f) r hr
    (fun x hx => by simp [relu, max_eq_right hx.2.le]) (r / 2) ⟨by linarith, by linarith⟩
    (by simp only [relu]; rw [max_eq_left (by linarith)]; positivity)

/-! ## Absolute positions on a relative host -/

/-- RoPE logits are invariant under a joint shift of the query and key positions. -/
theorem rope_shift_invariant {Pr : Type*} [Fintype Pr] (ω : Pr → ℝ) (q k : Pr → ℝ × ℝ)
    (i j c : ℝ) : ropeScore ω q k (i + c) (j + c) = ropeScore ω q k i j := by
  simp only [ropeScore, pairScore_eq, add_sub_add_right_eq_sub]

/-- **A single RoPE layer cannot play a learned-absolute layer** (impossibility, proved). With the
same token at every position (so the RoPE host sees the same `q, k` everywhere) and a learned
table `P 0 = 0, P 1 = P 2 = 1` added to a zero embedding, the absolute logit `P i · P j` is `0` at
`(1, 0)` but `1` at `(2, 1)`; every RoPE logit is equal at those two pairs. -/
theorem rope_cannot_learned_abs {Pr : Type*} [Fintype Pr] :
    ¬ ∃ (ω : Pr → ℝ) (q k : Pr → ℝ × ℝ), ∀ i j : ℕ,
      ropeScore ω q k i j = (if i = 0 then 0 else 1) * (if j = 0 then 0 else 1) := by
  rintro ⟨ω, q, k, h⟩
  have h1 := h 1 0
  have h2 := h 2 1
  have hs := rope_shift_invariant ω q k 1 0 1
  norm_num at h1 h2 hs
  rw [hs] at h2
  linarith

/-- **…but absolute position is recoverable in depth.** A causal head with all-zero logits and a
value marking position `0` (e.g. a BOS channel) outputs exactly `1/(t+1)` at position `t`. -/
theorem causal_uniform_head_position {T : ℕ} (t : Fin (T + 1)) :
    attn (univ.filter (· ≤ t)) (fun _ => (0 : ℝ)) (fun s : Fin (T + 1) => if s = 0 then (1 : ℝ) else 0)
      = 1 / ((t : ℝ) + 1) := by
  have hcard : (univ.filter (fun s : Fin (T + 1) => s ≤ t)).card = t + 1 := by
    have : univ.filter (fun s : Fin (T + 1) => s ≤ t) = Finset.Iic t := by ext; simp
    rw [this, Fin.card_Iic]
  simp only [attn, exp_zero, Finset.sum_const, nsmul_eq_mul, mul_one, hcard, smul_eq_mul,
    one_mul]
  rw [Finset.sum_ite_eq' (univ.filter (· ≤ t)) 0]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fin.zero_le, if_true]
  push_cast
  field_simp

/-! ## AR cannot play diffusion -/

/-- **diffusion ⊄ AR** (impossibility, proved). The two-token bidirectional uniform-attention
layer's output at position `0` depends on the token at position `1`, so no causal model equals
it. -/
theorem causal_ne_bidirectional (F : (Fin 2 → ℝ) → (Fin 2 → ℝ)) (hF : Causal F) :
    F ≠ fun X _ => attn univ (fun _ => (0 : ℝ)) X := by
  intro h
  have := hF ![0, 0] ![0, 1] 0 (fun s hs => by
    have : s = 0 := le_antisymm hs (Fin.zero_le _)
    subst this; rfl)
  rw [h] at this
  simp [attn, Fin.sum_univ_two] at this

/-! ## Stacked local windows: the depth law -/

/-- One window step: position `i` attends to `k` iff `k ≤ i < k + w`. -/
def winStep (w i k : ℕ) : Prop := k ≤ i ∧ i - k ≤ w - 1

/-- `L`-fold reachability through window steps. -/
def winReach (w : ℕ) : ℕ → ℕ → ℕ → Prop
  | 0, i, j => i = j
  | L + 1, i, j => ∃ k, winStep w i k ∧ winReach w L k j

/-- **Stacked windows reach exactly `L (w - 1)` back.** -/
theorem winReach_iff (w L i j : ℕ) : winReach w L i j ↔ j ≤ i ∧ i - j ≤ L * (w - 1) := by
  induction L generalizing i with
  | zero => simp only [winReach, zero_mul]; omega
  | succ L ih =>
    simp only [winReach, winStep, ih]
    set m := w - 1
    have e : (L + 1) * m = L * m + m := by ring
    rw [e]
    set P := L * m
    constructor
    · rintro ⟨k, ⟨h1, h2⟩, h3, h4⟩; omega
    · rintro ⟨h1, h2⟩
      refine ⟨max j (i - m), ⟨?_, ?_⟩, ?_, ?_⟩ <;> omega

/-- A sequence map is window-local if position `t`'s output depends only on the window. -/
def WinLocal {A B : Type*} (w : ℕ) (F : (ℕ → A) → (ℕ → B)) : Prop :=
  ∀ X Y t, (∀ s, winStep w t s → X s = Y s) → F X t = F Y t

/-- **Locality composes into reach.** After `L` window-local layers, position `t` depends only on
the inputs it reaches in `L` steps, i.e. on positions `≥ t - L (w - 1)`. So a host of window `w`
needs `L (w - 1) ≥ T - 1` layers before position `T - 1` can see token `0` — the
`⌈(T-1)/(w-1)⌉` depth of the exchange-rate table. -/
theorem winLocal_stack {A : Type*} (w : ℕ) (layers : List ((ℕ → A) → (ℕ → A)))
    (h : ∀ F ∈ layers, WinLocal w F) :
    ∀ X Y t, (∀ s, winReach w layers.length t s → X s = Y s) →
      layers.foldr (· ∘ ·) id X t = layers.foldr (· ∘ ·) id Y t := by
  induction layers with
  | nil => intro X Y t hXY; exact hXY t rfl
  | cons F rest ih =>
    intro X Y t hXY
    simp only [List.foldr_cons, Function.comp_apply]
    apply h F (by simp)
    intro s hs
    apply ih (fun G hG => h G (List.mem_cons_of_mem _ hG))
    intro u hu
    exact hXY u ⟨s, hs, by simpa using hu⟩

end EP
