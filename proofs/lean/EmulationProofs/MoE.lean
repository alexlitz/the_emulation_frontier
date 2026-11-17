import EmulationProofs.Common

/-!
# Mixture of experts: dense ⊆ MoE, MoE → dense, dense ⊇ top-1 MoE

A hidden unit is anything with a parameter `θ : Θ` and an activation `act θ x` (a plain
`σ(w·x)`, a gated `σ(w·x)(u·x)`, …); an FFN sums its units' activations times their down-
projection columns. An MoE layer sums experts (each an FFN with `ew` units) times gates.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {X Θ : Type*} {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- A dense FFN with units indexed by `U`. -/
def ffn {U : Type*} [Fintype U] (act : Θ → X → ℝ) (θ : U → Θ) (W2 : U → V) (x : X) : V :=
  ∑ u, act (θ u) x • W2 u

/-- An MoE layer: `E` experts of `ew` units each, combined with gate weights `gate x e`. -/
def moe {E ew : ℕ} (act : Θ → X → ℝ) (gate : X → Fin E → ℝ) (θ : Fin E → Fin ew → Θ)
    (W2 : Fin E → Fin ew → V) (x : X) : V :=
  ∑ e, gate x e • ffn act (θ e) (W2 e) x

/-- With zero router weights every token gets the router bias as logits, so the gates —
whatever function `G` of the logits the implementation uses (softmax, top-k, tie-break,
renormalisation, scaling) — are the same for every token. -/
theorem zero_router_constant_gate {E n : ℕ} (G : (Fin E → ℝ) → Fin E → ℝ)
    (br : Fin E → ℝ) (x : Fin n → ℝ) :
    G ((0 : Matrix (Fin E) (Fin n) ℝ).mulVec x + br) = G br := by simp

/-- Softmax over constant logits is uniform: `1/E` (the `× E` without renormalisation). -/
theorem softmax_const {E : ℕ} (c : ℝ) (e : Fin E) :
    softmax (fun _ : Fin E => c) e = 1 / E := by
  have : (E : ℝ) ≠ 0 := by exact_mod_cast (Fin.pos e).ne'
  simp [softmax]
  field_simp

/-- **dense ⊆ MoE** (exact, active capacity `d_ff ≤ k e_w`). Suppose the gates are constant
(`zero_router_constant_gate`), nonzero on `k` distinct experts `sel b` (whichever the
implementation's tie-break picks) and zero on every other expert. Deal the dense FFN's `k e_w` units (pad with zero units
if `d_ff < k e_w`) into those experts — unit `(b, u)` into slot `u` of expert `sel b` — and
divide each expert's down-projection by its gate weight. The MoE equals the dense FFN on every
input. -/
theorem dense_sub_moe {E k ew : ℕ} (act : Θ → X → ℝ) (θ : Fin (k * ew) → Θ)
    (W2 : Fin (k * ew) → V) (g : Fin E → ℝ) (sel : Fin k → Fin E)
    (hsel : Function.Injective sel) (hg : ∀ b, g (sel b) ≠ 0)
    (hoff : ∀ e, e ∉ Set.range sel → g e = 0) (θ₀ : Θ) :
    let θE : Fin E → Fin ew → Θ := fun e u =>
      if h : ∃ b, sel b = e then θ (finProdFinEquiv (h.choose, u)) else θ₀
    let W2E : Fin E → Fin ew → V := fun e u =>
      if h : ∃ b, sel b = e then (g e)⁻¹ • W2 (finProdFinEquiv (h.choose, u)) else 0
    ∀ x, moe act (fun _ => g) θE W2E x = ffn act θ W2 x := by
  intro θE W2E x
  have hch : ∀ b, (⟨b, rfl⟩ : ∃ b', sel b' = sel b).choose = b := fun b =>
    hsel (⟨b, rfl⟩ : ∃ b', sel b' = sel b).choose_spec
  unfold moe ffn
  symm
  rw [← Equiv.sum_comp finProdFinEquiv, Fintype.sum_prod_type]
  refine Fintype.sum_of_injective sel hsel _ _ (fun e he => by simp [hoff e he]) (fun b => ?_)
  have hex : ∃ b', sel b' = sel b := ⟨b, rfl⟩
  simp only [θE, W2E, hex, hch, dite_true, Finset.smul_sum, smul_smul]
  refine Finset.sum_congr rfl (fun u _ => ?_)
  congr 1
  field_simp [hg b]

/-- **MoE → dense by tying experts** (exact). If every expert is the same FFN and the gate
weights sum to `1` on every token (renormalised top-k), the router drops out. Without
renormalisation (`∑ gates < 1`, varying per token) this is false; see the report. -/
theorem tied_moe_is_dense {E ew : ℕ} (act : Θ → X → ℝ) (gate : X → Fin E → ℝ)
    (θ : Fin ew → Θ) (W2 : Fin ew → V) (hsum : ∀ x, ∑ e, gate x e = 1) (x : X) :
    moe act gate (fun _ => θ) (fun _ => W2) x = ffn act θ W2 x := by
  simp [moe, ← Finset.sum_smul, hsum]

/-! ## dense ⊇ top-1 MoE -/

/-- **dense ⊇ top-1 MoE, with a hard select layer** (exact, capacity + depth). All `E` experts are
held in width; the select layer's one-hot mask of the argmax `e*` picks the routed expert's
output (top-1 with a renormalised, i.e. unit, gate). -/
theorem dense_hard_select {E : ℕ} (Y : Fin E → V) (estar : Fin E) :
    ∑ e, (if e = estar then (1 : ℝ) else 0) • Y e = Y estar := by
  simp

/-- **dense ⊇ top-1 MoE, the smooth router** (limit `β → ∞`). If the router logits `z` have a
UNIQUE maximiser `e*`, then `∑_e softmax(β z)_e Y_e → Y_{e*}`. -/
theorem dense_soft_router_limit {E : ℕ} (z : Fin E → ℝ) (Y : Fin E → V) (estar : Fin E)
    (huniq : ∀ e, e ≠ estar → z e < z estar) :
    Tendsto (fun β : ℝ => ∑ e, softmax (fun e => β * z e) e • Y e) atTop (𝓝 (Y estar)) := by
  have hlim := attn_bias_limit (fun _ => (0 : ℝ)) (fun e => z e - z estar) Y
    (fun e => by
      rcases eq_or_ne e estar with h | h
      · simp [h]
      · linarith [huniq e h])
    ⟨estar, by simp⟩
  have hS : univ.filter (fun e => z e - z estar = 0) = {estar} := by
    ext e
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    constructor
    · intro h; by_contra hne; linarith [huniq e hne]
    · intro h; simp [h]
  rw [hS] at hlim
  have hval : attn {estar} (fun _ => (0 : ℝ)) Y = Y estar := by simp [attn]
  rw [hval] at hlim
  refine hlim.congr (fun β => ?_)
  rw [← attn_univ_eq_softmax]
  have := attn_add_const univ (fun e => 0 + β * (z e - z estar)) (β * z estar) Y
  rw [← this]
  congr 1
  funext e; ring

/-- **…but not at a tie.** With two experts tied for the max, `softmax (β z)` is `(1/2, 1/2)`
for every `β`, so the smooth router's output is the average of the two experts, never the
tie-broken argmax's single expert (when the two differ). -/
theorem soft_router_tie (Y : Fin 2 → V) (β : ℝ) :
    ∑ e, softmax (fun _ : Fin 2 => β * 0) e • Y e = (1 / 2 : ℝ) • (Y 0 + Y 1) := by
  simp only [softmax_const, Fin.sum_univ_two, smul_add]
  norm_num

/-- `SiLU (a - B) · b → 0` as `B → ∞`: masking an expert's units by shifting their gate
pre-activations by `-B` (`B` written by the select layer when the expert is unselected) is a
limit in a SiLU host. -/
theorem silu_shift_mask (a b : ℝ) :
    Tendsto (fun B : ℝ => (a - B) * sigmoid (a - B) * b) atTop (𝓝 0) := by
  have hsig : ∀ t, sigmoid t ≤ exp t := by
    intro t
    rw [sigmoid_def, ← one_div, div_le_iff₀ (by positivity), mul_add, mul_one, ← exp_add,
      add_neg_cancel, exp_zero]
    linarith [exp_pos t]
  -- |t σ(t)| ≤ (-t) e^t for t ≤ 0, and (-t) e^t → 0 as t → -∞
  have hT : Tendsto (fun B : ℝ => (B - a) * exp (-(B - a))) atTop (𝓝 0) := by
    have := (Real.tendsto_pow_mul_exp_neg_atTop_nhds_zero 1).comp
      (tendsto_atTop_add_const_right _ (-a) tendsto_id)
    refine this.congr (fun B => ?_)
    simp only [Function.comp, id, pow_one, neg_add, neg_neg]
    ring_nf
  rw [show (0 : ℝ) = 0 * b by simp]
  refine Tendsto.mul_const b ?_
  refine squeeze_zero_norm' ?_ hT
  filter_upwards [eventually_ge_atTop a] with B hB
  rw [Real.norm_eq_abs, abs_mul, abs_of_nonpos (by linarith : a - B ≤ 0),
    abs_of_pos (sigmoid_pos _)]
  have : 0 ≤ B - a := by linarith
  calc -(a - B) * sigmoid (a - B) = (B - a) * sigmoid (a - B) := by ring
    _ ≤ (B - a) * exp (a - B) := mul_le_mul_of_nonneg_left (hsig _) this
    _ = (B - a) * exp (-(B - a)) := by rw [neg_sub]

end EP
