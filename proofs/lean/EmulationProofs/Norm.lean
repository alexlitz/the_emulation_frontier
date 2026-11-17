import EmulationProofs.Common

/-!
# Norms: LayerNorm ⊆ RMSNorm (fold) and RMSNorm ⊆ LayerNorm (the `[h; -h]` mirror)

Vectors are `Fin n → ℝ`. A pre-norm network is modelled faithfully at the level of the residual
stream: a sequence of `T` token vectors, layers `l` whose sublayer `f l` is an ARBITRARY map of
the whole normed sequence (attention mixes tokens, an MLP does not; both are covered), and a
final norm read by the head.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {n : ℕ}

/-- Mean of a vector. -/
def mean (x : Fin n → ℝ) : ℝ := (∑ i, x i) / n

/-- The centring projection `P = I - (1/n) 1 1ᵀ`. -/
def center (x : Fin n → ℝ) : Fin n → ℝ := fun i => x i - mean x

/-- LayerNorm with weight `γ`, bias `β`, epsilon `eps`. -/
def layerNorm (eps : ℝ) (γ β x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => γ i * (x i - mean x) / √((∑ l, (x l - mean x) ^ 2) / n + eps) + β i

lemma mean_center (x : Fin n → ℝ) : mean (center x) = 0 := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp [mean]
  have : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  simp only [mean, center, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

lemma center_center (x : Fin n → ℝ) : center (center x) = center x := by
  funext i
  show center x i - mean (center x) = center x i
  rw [mean_center, sub_zero]

lemma center_add (x y : Fin n → ℝ) : center (x + y) = center x + center y := by
  funext i
  simp only [center, mean, Pi.add_apply, Finset.sum_add_distrib]
  ring

/-- **LayerNorm = γ ⊙ RMSNorm(P x) + β** (exact, same epsilon). -/
theorem layerNorm_eq_rms_center (eps : ℝ) (γ β x : Fin n → ℝ) :
    layerNorm eps γ β x = fun i => γ i * rmsNorm eps 1 (center x) i + β i := by
  funext i
  simp only [layerNorm, rmsNorm, center, Pi.one_apply, one_mul, mul_div_assoc]

/-- The affine part of a norm folds into the next linear layer:
`W (γ ⊙ z + β) = (W · diag γ) z + W β`. -/
theorem affine_fold {m : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (γ β z : Fin n → ℝ) :
    W.mulVec (fun i => γ i * z i + β i) = (W * Matrix.diagonal γ).mulVec z + W.mulVec β := by
  rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_add]
  congr 1
  funext i
  simp [Matrix.mulVec_diagonal]

/-! ## The whole-model fold -/

/-- A pre-norm residual stack with a per-token norm `N`: `h₀ = e`,
`h_{l+1} = h_l + f l (N l h_l)`. -/
def stack {T : ℕ} (N : ℕ → (Fin n → ℝ) → (Fin n → ℝ))
    (f : ℕ → (Fin T → Fin n → ℝ) → (Fin T → Fin n → ℝ)) (e : Fin T → Fin n → ℝ) :
    ℕ → Fin T → Fin n → ℝ
  | 0 => e
  | l + 1 => stack N f e l + f l (fun t => N l (stack N f e l t))

/-- **LayerNorm ⊆ RMSNorm, the whole model** (exact, fold + biases). Target: a LayerNorm model
with weights `γ l`, biases `β l`, sublayers `f l`, embedding `e` and head `head` reading the
final LayerNorm `(γf, βf)`. Host: an RMSNorm model (weight `1`, same `eps`) whose every
residual writer — the embedding and every sublayer's output projection — is post-composed with
the centring `P`, and whose sublayers read `γ ⊙ z + β` (which `affine_fold` folds into their
first matrices and biases; for the head this is the lm_head bias). Then the host residual is
`P` of the target residual at every layer, and the logits agree for every input. -/
theorem layerNorm_sub_rmsNorm {T : ℕ} (eps : ℝ) (γ β : ℕ → Fin n → ℝ)
    (f : ℕ → (Fin T → Fin n → ℝ) → (Fin T → Fin n → ℝ)) (e : Fin T → Fin n → ℝ)
    (γf βf : Fin n → ℝ) {Out : Type*} (head : (Fin n → ℝ) → Out) (L : ℕ) :
    let hostNorm : ℕ → (Fin n → ℝ) → (Fin n → ℝ) := fun _ => rmsNorm eps 1
    let hostF : ℕ → (Fin T → Fin n → ℝ) → (Fin T → Fin n → ℝ) :=
      fun l z t => center (f l (fun s i => γ l i * z s i + β l i) t)
    let hostE : Fin T → Fin n → ℝ := fun t => center (e t)
    (∀ l, stack hostNorm hostF hostE l = fun t => center (stack (fun l => layerNorm eps (γ l) (β l))
        f e l t)) ∧
    (∀ t, head (fun i => γf i * rmsNorm eps 1 (stack hostNorm hostF hostE L t) i + βf i) =
      head (layerNorm eps γf βf (stack (fun l => layerNorm eps (γ l) (β l)) f e L t))) := by
  intro hostNorm hostF hostE
  have key : ∀ l, stack hostNorm hostF hostE l =
      fun t => center (stack (fun l => layerNorm eps (γ l) (β l)) f e l t) := by
    intro l
    induction l with
    | zero => rfl
    | succ l ih =>
      have harg : (fun s i => γ l i * rmsNorm eps 1
          (center (stack (fun l => layerNorm eps (γ l) (β l)) f e l s)) i + β l i) =
          (fun s => layerNorm eps (γ l) (β l) (stack (fun l => layerNorm eps (γ l) (β l)) f e l s)) := by
        funext s; rw [layerNorm_eq_rms_center]
      funext t
      simp only [stack, ih, hostF, hostNorm, harg, center_add, Pi.add_apply]
  refine ⟨key, fun t => ?_⟩
  rw [key, layerNorm_eq_rms_center]

/-! ## RMSNorm ⊆ LayerNorm at 2× width: the mirror -/

/-- The mirror `[h; -h]` of a width-`n` vector, width `n + n`. -/
def mirror (x : Fin n → ℝ) : Fin (n + n) → ℝ := Fin.append x (-x)

/-- The first half of a width-`n + n` vector. -/
def firstHalf (y : Fin (n + n) → ℝ) : Fin n → ℝ := fun i => y (Fin.castAdd n i)

@[simp] lemma mirror_left (x : Fin n → ℝ) (i : Fin n) : mirror x (Fin.castAdd n i) = x i := by
  simp [mirror]

@[simp] lemma mirror_right (x : Fin n → ℝ) (i : Fin n) : mirror x (Fin.natAdd n i) = -x i := by
  rw [mirror, Fin.append_right]; rfl

lemma mean_mirror (x : Fin n → ℝ) : mean (mirror x) = 0 := by
  simp only [mean, Fin.sum_univ_add, mirror_left, mirror_right, Finset.sum_neg_distrib,
    add_neg_cancel, zero_div]

lemma sumsq_mirror (x : Fin n → ℝ) :
    (∑ l, (mirror x l - mean (mirror x)) ^ 2) / ((n + n : ℕ) : ℝ) = (∑ l, x l ^ 2) / n := by
  rw [mean_mirror]
  simp only [sub_zero, Fin.sum_univ_add, mirror_left, mirror_right, neg_sq]
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp
  have : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  push_cast
  field_simp

/-- **RMSNorm ⊆ LayerNorm, the norm** (exact, 2× width). On the mirrored vector the mean is `0`
(centring is a no-op) and the variance is the target's mean square, so a LayerNorm with weight
`[γ; γ]`, bias `0` and the SAME epsilon returns the mirror of the RMSNorm. -/
theorem layerNorm_mirror (eps : ℝ) (γ x : Fin n → ℝ) :
    layerNorm eps (Fin.append γ γ) 0 (mirror x) = mirror (rmsNorm eps γ x) := by
  funext i
  have hs := sumsq_mirror x
  simp only [mean_mirror, sub_zero] at hs
  simp only [layerNorm, mean_mirror, sub_zero, hs, Pi.zero_apply, add_zero]
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · simp only [Fin.append_left, mirror_left, rmsNorm, mul_div_assoc]
  · simp only [Fin.append_right, mirror_right, rmsNorm, mul_div_assoc, neg_div, mul_neg]

/-- **RMSNorm ⊆ LayerNorm, the whole model** (exact, 2× hidden). Target: an RMSNorm stack.
Host: a LayerNorm stack of width `n + n` with weights `[γ; γ]`, zero biases, the same epsilon,
the mirrored embedding, and sublayers that read the first half of their (normed) input and write
the mirror of the target's write (every writer writes `[w; -w]`). The host residual is the
mirror of the target residual at every layer, and a head reading the first half of the final
LayerNorm sees exactly the target's final RMSNorm. -/
theorem rmsNorm_sub_layerNorm_mirror {T : ℕ} (eps : ℝ) (γ : ℕ → Fin n → ℝ)
    (f : ℕ → (Fin T → Fin n → ℝ) → (Fin T → Fin n → ℝ)) (e : Fin T → Fin n → ℝ)
    (γf : Fin n → ℝ) {Out : Type*} (head : (Fin n → ℝ) → Out) (L : ℕ) :
    let hostNorm : ℕ → (Fin (n + n) → ℝ) → (Fin (n + n) → ℝ) :=
      fun l => layerNorm eps (Fin.append (γ l) (γ l)) 0
    let hostF : ℕ → (Fin T → Fin (n + n) → ℝ) → (Fin T → Fin (n + n) → ℝ) :=
      fun l z t => mirror (f l (fun s => firstHalf (z s)) t)
    let hostE : Fin T → Fin (n + n) → ℝ := fun t => mirror (e t)
    (∀ l, stack hostNorm hostF hostE l =
        fun t => mirror (stack (fun l => rmsNorm eps (γ l)) f e l t)) ∧
    (∀ t, head (firstHalf (layerNorm eps (Fin.append γf γf) 0 (stack hostNorm hostF hostE L t))) =
      head (rmsNorm eps γf (stack (fun l => rmsNorm eps (γ l)) f e L t))) := by
  intro hostNorm hostF hostE
  have firstHalf_mirror : ∀ x : Fin n → ℝ, firstHalf (mirror x) = x := by
    intro x; funext i; simp [firstHalf]
  have mirror_add : ∀ x y : Fin n → ℝ, mirror (x + y) = mirror x + mirror y := by
    intro x y; funext i
    refine Fin.addCases (fun i => ?_) (fun i => ?_) i
    · simp only [Pi.add_apply, mirror_left]
    · simp only [Pi.add_apply, mirror_right]; ring
  have key : ∀ l, stack hostNorm hostF hostE l =
      fun t => mirror (stack (fun l => rmsNorm eps (γ l)) f e l t) := by
    intro l
    induction l with
    | zero => rfl
    | succ l ih =>
      funext t
      simp only [stack, ih, hostF, hostNorm, layerNorm_mirror, firstHalf_mirror, Pi.add_apply,
        mirror_add]
  refine ⟨key, fun t => ?_⟩
  rw [key, layerNorm_mirror, firstHalf_mirror]

end EP
