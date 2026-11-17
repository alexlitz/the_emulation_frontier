import EmulationProofs.Position

/-!
# A partial-RoPE host with fewer rotating pairs cannot play a RoPE target

A RoPE pair of frequency `ω`, seen from a key at position `0`, contributes
`a cos(n ω) + b sin(n ω)` at query position `n`. Every such sequence satisfies the recurrence
`g (n+2) = 2 cos ω · g (n+1) - g n`, so the operator `g ↦ g (n+2) - 2x g (n+1) + g n` with
`x = cos ω` kills it, and with any other `x` multiplies it by `2 (cos ω - x)` (and shifts it).
Applying the operators of all the host's frequencies (each twice, and `x = 1` for its pass
pairs) kills the host's logit; on the target's logit it leaves a sum of squares that is positive
as soon as one target frequency is not among the host's. With fewer rotating pairs than the
target has distinct frequencies, one always is not (pigeonhole).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-- The second-order difference operator of frequency cosine `x`. -/
def recOp (x : ℝ) (g : ℕ → ℝ) : ℕ → ℝ := fun n => g (n + 2) - 2 * x * g (n + 1) + g n

/-- A list of such operators, applied right to left. -/
def recOps : List ℝ → (ℕ → ℝ) → (ℕ → ℝ)
  | [], g => g
  | x :: xs, g => recOp x (recOps xs g)

/-- The operators are linear: they commute with finite sums. -/
lemma recOps_sum {α : Type*} (s : Finset α) (xs : List ℝ) (g : α → ℕ → ℝ) (n : ℕ) :
    recOps xs (fun m => ∑ a ∈ s, g a m) n = ∑ a ∈ s, recOps xs (g a) n := by
  induction xs generalizing n with
  | nil => rfl
  | cons x xs ih =>
    show recOps xs _ (n + 2) - 2 * x * recOps xs _ (n + 1) + recOps xs _ n = _
    rw [ih, ih, ih, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    rfl

/-- On a sequence with `g (n+2) = 2c g (n+1) - g n`, the operators multiply by
`∏ 2 (c - x)` and shift by the number of operators. -/
lemma recOps_rec (c : ℝ) (g : ℕ → ℝ) (hg : ∀ n, g (n + 2) = 2 * c * g (n + 1) - g n)
    (xs : List ℝ) (n : ℕ) :
    recOps xs g n = (xs.map (fun x => 2 * (c - x))).prod * g (n + xs.length) := by
  induction xs generalizing n with
  | nil => simp [recOps]
  | cons x xs ih =>
    show recOps xs g (n + 2) - 2 * x * recOps xs g (n + 1) + recOps xs g n = _
    rw [ih, ih, ih, List.map_cons, List.prod_cons, List.length_cons]
    have e1 : n + 2 + xs.length = (n + xs.length) + 2 := by omega
    have e2 : n + 1 + xs.length = (n + xs.length) + 1 := by omega
    have e3 : n + (xs.length + 1) = (n + xs.length) + 1 := by omega
    rw [e1, e2, e3, hg]
    ring

/-- A RoPE pair's logit along query positions satisfies the frequency's recurrence. -/
lemma sinusoid_rec (a b ω : ℝ) (n : ℕ) :
    a * cos (((n + 2 : ℕ) : ℝ) * ω) + b * sin (((n + 2 : ℕ) : ℝ) * ω) =
      2 * cos ω * (a * cos (((n + 1 : ℕ) : ℝ) * ω) + b * sin (((n + 1 : ℕ) : ℝ) * ω)) -
        (a * cos ((n : ℝ) * ω) + b * sin ((n : ℝ) * ω)) := by
  push_cast
  have h1 : ((n : ℝ) + 2) * ω = ((n : ℝ) + 1) * ω + ω := by ring
  have h2 : (n : ℝ) * ω = ((n : ℝ) + 1) * ω - ω := by ring
  rw [h1, h2, cos_add, sin_add, cos_sub, sin_sub]
  ring

/-- A pair's logit against a key at position `0`. -/
lemma pairScore_key0 (ω : ℝ) (n : ℕ) (q k : ℝ × ℝ) :
    pairScore ω n 0 q k = dot2 q k * cos ((n : ℝ) * ω) + (q.1 * k.2 - q.2 * k.1) * sin ((n : ℝ) * ω) := by
  rw [pairScore_eq, sub_zero]

/-- **A partial-RoPE host with fewer rotating pairs cannot play a RoPE target** (impossibility,
proved). The target head has `Pt` pairs whose frequencies have distinct cosines, none equal to
`1` (no frequency is `0` mod `2π`). The host head has `Ph < Pt` rotating pairs with ANY
frequencies (buffer writes allowed) plus any number `Z` of pass (zero-frequency) pairs. There
is a target query/key such that no host query/key reproduces the target's logit at every
causal pair of positions; in fact the host already fails on queries at positions
`0, …, 2(Ph+1)` against a key at position `0`. -/
theorem partial_rope_too_few {Ph Z Pt : ℕ} (hlt : Ph < Pt) (ωt : Fin Pt → ℝ)
    (hinj : Function.Injective (fun p => cos (ωt p))) (hne : ∀ p, cos (ωt p) ≠ 1) :
    ∃ q k : Fin Pt → ℝ × ℝ, ∀ (ωh : Fin Ph → ℝ) (qh kh : Fin Ph ⊕ Fin Z → ℝ × ℝ),
      ¬ ∀ i j : ℕ, j ≤ i →
        ropeScore (Sum.elim ωh (fun _ => 0)) qh kh i j = ropeScore ωt q k i j := by
  set N : ℕ := 2 * (Ph + 1)
  refine ⟨fun _ => (1, 0), fun p => (cos (N * ωt p), sin (N * ωt p)), ?_⟩
  intro ωh qh kh h
  set ωH : Fin Ph ⊕ Fin Z → ℝ := Sum.elim ωh (fun _ => 0)
  set L : List ℝ := 1 :: List.ofFn (fun q => cos (ωh q))
  set L2 := L ++ L
  have hlen : L2.length = N := by
    simp only [L2, L, N, List.length_append, List.length_cons, List.length_ofFn]; ring
  -- the host's and the target's logits along query positions, key at `0`
  have hH : (fun n : ℕ => ropeScore ωH qh kh n 0) = fun n : ℕ => ∑ p, (dot2 (qh p) (kh p) *
      cos ((n : ℝ) * ωH p) + ((qh p).1 * (kh p).2 - (qh p).2 * (kh p).1) * sin ((n : ℝ) * ωH p)) := by
    funext n; simp only [ropeScore, pairScore_key0]
  have hT : (fun n : ℕ => ropeScore ωt (fun _ => ((1 : ℝ), (0 : ℝ)))
      (fun p => (cos (N * ωt p), sin (N * ωt p))) n 0) = fun n : ℕ => ∑ p, (cos (N * ωt p) *
      cos ((n : ℝ) * ωt p) + sin (N * ωt p) * sin ((n : ℝ) * ωt p)) := by
    funext n; simp only [ropeScore, pairScore_key0, dot2]; congr 1; funext p; ring
  have heq : (fun n : ℕ => ropeScore ωH qh kh n 0) = (fun n : ℕ => ropeScore ωt
      (fun _ => ((1 : ℝ), (0 : ℝ))) (fun p => (cos (N * ωt p), sin (N * ωt p))) n 0) := by
    funext n; simpa using h n 0 (Nat.zero_le n)
  have key := congrArg (fun g => recOps L2 g 0) heq
  simp only [hH, hT] at key
  rw [recOps_sum, recOps_sum] at key
  -- host side: every pair is killed
  have hhost : ∀ p, recOps L2 (fun n : ℕ => dot2 (qh p) (kh p) * cos ((n : ℝ) * ωH p) +
      ((qh p).1 * (kh p).2 - (qh p).2 * (kh p).1) * sin ((n : ℝ) * ωH p)) 0 = 0 := by
    intro p
    rw [recOps_rec (cos (ωH p)) _ (fun n => sinusoid_rec _ _ _ n)]
    have hmem : cos (ωH p) ∈ L2 := by
      refine List.mem_append_left _ ?_
      rcases p with q | z
      · exact List.mem_cons_of_mem _ (List.mem_ofFn.mpr ⟨q, rfl⟩)
      · simp [ωH, L]
    have : (L2.map (fun x => 2 * (cos (ωH p) - x))).prod = 0 :=
      List.prod_eq_zero (List.mem_map.mpr ⟨_, hmem, by ring⟩)
    rw [this, zero_mul]
  -- target side: a sum of squares
  set P : Fin Pt → ℝ := fun p => (L.map (fun x => 2 * (cos (ωt p) - x))).prod
  have htarget : ∀ p, recOps L2 (fun n : ℕ => cos (N * ωt p) * cos ((n : ℝ) * ωt p) +
      sin (N * ωt p) * sin ((n : ℝ) * ωt p)) 0 = P p ^ 2 := by
    intro p
    rw [recOps_rec (cos (ωt p)) _ (fun n => sinusoid_rec _ _ _ n), zero_add, hlen]
    simp only [L2, List.map_append, List.prod_append, P]
    rw [← sq, ← sq, ← sq, cos_sq_add_sin_sq, mul_one]
  simp only [hhost, htarget, Finset.sum_const_zero] at key
  -- pigeonhole: some target frequency is not among the host's
  have hex : ∃ p, P p ≠ 0 := by
    by_contra hall
    push Not at hall
    have hσ : ∀ p, ∃ q, cos (ωh q) = cos (ωt p) := by
      intro p
      obtain ⟨x, hx, hx0⟩ := List.mem_map.mp ((List.prod_eq_zero_iff).mp (hall p))
      have hxe : x = cos (ωt p) := by linarith
      rcases List.mem_cons.mp hx with h1 | h1
      · exact absurd (hxe ▸ h1) (hne p)
      · obtain ⟨q, hq⟩ := List.mem_ofFn.mp h1
        exact ⟨q, hq.trans hxe⟩
    choose σ hσ using hσ
    have : Function.Injective σ := fun a b hab => hinj (by
      simp only; rw [← hσ a, ← hσ b, hab])
    have := Fintype.card_le_of_injective σ this
    simp at this
    omega
  obtain ⟨p0, hp0⟩ := hex
  have hpos : 0 < ∑ p, P p ^ 2 :=
    Finset.sum_pos' (fun p _ => sq_nonneg _) ⟨p0, Finset.mem_univ _, by positivity⟩
  linarith

/-- **The default schedule: `rot_h < rot_t` cannot work** (impossibility, proved). A default-RoPE
target with `Pt` pairs and base `θ > 1` (frequencies `θ^(-p/Pt) ∈ (0, 1]`, distinct) cannot be
reproduced by any host head with fewer rotating pairs, whatever its frequencies and however
many pass pairs it has. -/
theorem partial_rope_too_few_default {Ph Z Pt : ℕ} (hlt : Ph < Pt) (θ : ℝ) (hθ : 1 < θ) :
    ∃ q k : Fin Pt → ℝ × ℝ, ∀ (ωh : Fin Ph → ℝ) (qh kh : Fin Ph ⊕ Fin Z → ℝ × ℝ),
      ¬ ∀ i j : ℕ, j ≤ i →
        ropeScore (Sum.elim ωh (fun _ => 0)) qh kh i j =
          ropeScore (defaultFreq θ Pt) q k i j := by
  have hPt : (0 : ℝ) < Pt := by exact_mod_cast (Nat.zero_le Ph).trans_lt hlt
  have hpos : ∀ p, 0 < defaultFreq θ Pt p := fun p => Real.rpow_pos_of_pos (by linarith) _
  have hle : ∀ p, defaultFreq θ Pt p ≤ 1 := fun p =>
    Real.rpow_le_one_of_one_le_of_nonpos hθ.le (div_nonpos_of_nonpos_of_nonneg
      (by have : (0 : ℝ) ≤ (p : ℕ) := Nat.cast_nonneg _; linarith) (by positivity))
  have hpi : (1 : ℝ) ≤ π := by linarith [Real.pi_gt_three]
  refine partial_rope_too_few hlt _ (fun a b hab => ?_) (fun p => ?_)
  · have hab' := Real.injOn_cos ⟨(hpos a).le, (hle a).trans hpi⟩ ⟨(hpos b).le, (hle b).trans hpi⟩ hab
    simp only [defaultFreq] at hab'
    have := le_antisymm ((Real.rpow_le_rpow_left_iff hθ).mp hab'.le)
      ((Real.rpow_le_rpow_left_iff hθ).mp hab'.ge)
    field_simp at this
    have : ((a : ℕ) : ℝ) = (b : ℕ) := by linarith
    exact Fin.ext (by exact_mod_cast this)
  · intro h1
    have := (Real.cos_eq_one_iff_of_lt_of_lt (x := defaultFreq θ Pt p)
      (by linarith [hpos p, Real.pi_pos]) (by linarith [hle p, Real.pi_gt_three])).mp h1
    exact (hpos p).ne' this

end EP
