import EmulationProofs.Position
import EmulationProofs.Attention

/-!
# KV-head packing: several target KV heads in one wider host KV head

A GQA target has `G` KV heads; query head `h` of group `g` attends with KV head `g`. A host KV
head is wider when its pair set `Q` (key/query dims, in RoPE pairs) and value set `Qv` hold an
injective copy of every target KV head: `σ : Fin G × Pt → Q`, `τ : Fin G × Dv → Qv` (disjoint
slices; any leftover host dims are zero padding). The host key / value write target head `g`'s
key / value into slice `g`; a host query head serving target group `g` is zero outside slice `g`;
`o_proj` reads value slice `g`. Then:

* **no RoPE** (`kvPack_nope`): exact for every input, every key set (any mask);
* **RoPE** (`kvPack_rope_attn`): exact when every slice pair's host frequency equals the target
  head's frequency of that pair, `ω_h(σ(g, p)) = ω_t(g, p)` (written into the `inv_freq`
  buffer, as for NoPE);
* **…and only then** (`kvPack_rope_iff`): exactness for all target weights at all integer
  positions holds iff `cos` and `sin` of the two frequencies agree (`ω_h ≡ ω_t mod 2π`) on every
  slice pair;
* **the config-only default schedule cannot pack two RoPE heads** (`kvPack_default_fails`):
  the default schedule's pair 0 has frequency `θ^0 = 1` in every target head, while a default
  host schedule `θ_h^(-2p/2P_h)` with `θ_h > 1` has distinct frequencies in `(0, 1]`, so only one
  host pair matches frequency 1.

The softmax normalises per QUERY head, not per KV head, so the relation's older justification
"softmax is per head: two target KV heads cannot share one wider host head" is wrong for NoPE
targets and for hosts with written frequency buffers; it is right for config-only RoPE hosts.
The scale `1/√d_h` vs `1/√d_t` folds into `W_q` (`attn_scale_fold`).
-/

open Real Finset Function

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {G : ℕ} {Pt Q : Type*} [Fintype Pt] [Fintype Q] [DecidableEq Q]

/-- The packed host key: target head `g`'s key pairs in slice `g`, zero on unused host pairs. -/
def kvPackK (σ : Fin G × Pt → Q) (k : Fin G → Pt → ℝ × ℝ) : Q → ℝ × ℝ :=
  extend σ (fun x => k x.1 x.2) 0

/-- The packed host query of a query head of target group `g`: zero outside slice `g`. -/
def kvPackQ (σ : Fin G × Pt → Q) (g : Fin G) (q : Pt → ℝ × ℝ) : Q → ℝ × ℝ :=
  extend σ (fun x => if x.1 = g then q x.2 else 0) 0

/-- The packed host value: target head `g`'s value in value slice `g`. -/
def kvPackV {Dv Qv : Type*} (τ : Fin G × Dv → Qv) (v : Fin G → Dv → ℝ) : Qv → ℝ :=
  extend τ (fun x => v x.1 x.2) 0

lemma kvPack_pairScore_zero (ω i j : ℝ) (k : ℝ × ℝ) : pairScore ω i j 0 k = 0 := by
  simp [pairScore, rot, dot2]

/-- A host logit over extended-by-zero pairs is the sum over the slices. -/
lemma kvPack_score_slices (σ : Fin G × Pt → Q) (hσ : Injective σ) (ωh : Q → ℝ)
    (a b : Fin G × Pt → ℝ × ℝ) (i j : ℝ) :
    ropeScore ωh (extend σ a 0) (extend σ b 0) i j =
      ∑ x, pairScore (ωh (σ x)) i j (a x) (b x) := by
  classical
  rw [ropeScore, ← Finset.sum_subset (Finset.subset_univ (Finset.univ.image σ)), Finset.sum_image
    (fun x _ y _ h => hσ h)]
  · simp only [hσ.extend_apply]
  · intro c _ hc
    have hn : ¬ ∃ x, σ x = c := fun ⟨x, hx⟩ => hc (by simp [← hx])
    rw [extend_apply' _ _ _ hn]
    exact kvPack_pairScore_zero _ _ _ _

/-- **KV packing, the logit** (exact). With slice frequencies equal to the target's, the packed
host logit of a query head of group `g` is target head `g`'s RoPE logit, at every position. -/
theorem kvPack_rope_score (σ : Fin G × Pt → Q) (hσ : Injective σ) (ωh : Q → ℝ)
    (ωt : Fin G → Pt → ℝ) (hω : ∀ g p, ωh (σ (g, p)) = ωt g p) (g : Fin G)
    (q : Pt → ℝ × ℝ) (k : Fin G → Pt → ℝ × ℝ) (i j : ℝ) :
    ropeScore ωh (kvPackQ σ g q) (kvPackK σ k) i j = ropeScore (ωt g) q (k g) i j := by
  rw [kvPackQ, kvPackK, kvPack_score_slices σ hσ, Fintype.sum_prod_type, Finset.sum_eq_single g]
  · simp [ropeScore, hω]
  · intro b _ hb
    simp [hb, kvPack_pairScore_zero]
  · simp

lemma kvPack_attn_apply {ι Qv : Type*} [Fintype ι] [Fintype Qv] (S : Finset ι) (s : ι → ℝ) (v : ι → Qv → ℝ)
    (c : Qv) : attn S s v c = attn S s (fun j => v j c) := by
  simp [attn, smul_eq_mul]

/-- **KV-head packing, RoPE** (exact). Target: GQA head of group `g`, query map `qt`, KV maps
`kt g`, `vt g` (pairs `Pt`, value dims `Dv`), RoPE frequencies `ωt g`. Host: ONE KV head with
pairs `Q ⊇ σ(Fin G × Pt)` and value dims `Qv ⊇ τ(Fin G × Dv)`, every target head in its own
slice, frequencies `ωh` matching on every slice. The packed host head's output, read at value
slice `g`, equals the target head's output, for every input, key set and positions. -/
theorem kvPack_rope_attn {ι X Dv Qv : Type*} [Fintype ι] [Fintype Dv] [Fintype Qv] (σ : Fin G × Pt → Q) (hσ : Injective σ)
    (τ : Fin G × Dv → Qv) (hτ : Injective τ) (ωh : Q → ℝ) (ωt : Fin G → Pt → ℝ)
    (hω : ∀ g p, ωh (σ (g, p)) = ωt g p) (g : Fin G) (qt : X → Pt → ℝ × ℝ)
    (kt : Fin G → X → Pt → ℝ × ℝ) (vt : Fin G → X → Dv → ℝ) (S : Finset ι) (xq : X)
    (xs : ι → X) (i : ℝ) (pos : ι → ℝ) (t : Dv) :
    attn S (fun j => ropeScore ωh (kvPackQ σ g (qt xq)) (kvPackK σ (fun g' => kt g' (xs j)))
        i (pos j)) (fun j => kvPackV τ (fun g' => vt g' (xs j))) (τ (g, t)) =
      attn S (fun j => ropeScore (ωt g) (qt xq) (kt g (xs j)) i (pos j))
        (fun j => vt g (xs j)) t := by
  rw [kvPack_attn_apply, kvPack_attn_apply]
  simp only [kvPack_rope_score σ hσ ωh ωt hω, kvPackV, hτ.extend_apply]

/-- **KV-head packing without RoPE** (exact). NoPE targets on a NoPE host (or zero `inv_freq` on
the packed slices): one host KV head of width `≥ Σ` target KV-head widths reproduces every target
head, for every input and key set. -/
theorem kvPack_nope {ι X Dv Qv : Type*} [Fintype ι] [Fintype Dv] [Fintype Qv] (σ : Fin G × Pt → Q) (hσ : Injective σ)
    (τ : Fin G × Dv → Qv) (hτ : Injective τ) (g : Fin G) (qt : X → Pt → ℝ × ℝ)
    (kt : Fin G → X → Pt → ℝ × ℝ) (vt : Fin G → X → Dv → ℝ) (S : Finset ι) (xq : X)
    (xs : ι → X) (t : Dv) :
    attn S (fun j => ∑ c, dot2 (kvPackQ σ g (qt xq) c) (kvPackK σ (fun g' => kt g' (xs j)) c))
        (fun j => kvPackV τ (fun g' => vt g' (xs j))) (τ (g, t)) =
      attn S (fun j => ∑ p, dot2 (qt xq p) (kt g (xs j) p)) (fun j => vt g (xs j)) t := by
  have := kvPack_rope_attn σ hσ τ hτ (fun _ => 0) (fun _ _ => 0) (fun _ _ => rfl) g qt kt vt S
    xq xs 0 (fun _ => 0) t
  simpa [nope_sub_rope] using this

/-! ## When RoPE packing works: the frequency condition is necessary -/

lemma kvPack_cos_sin_nat (a b : ℝ) (hc : cos a = cos b) (hs : sin a = sin b) (n : ℕ) :
    cos (n * a) = cos (n * b) ∧ sin (n * a) = sin (n * b) := by
  induction n with
  | zero => simp
  | succ n ih =>
    push_cast
    rw [add_mul, add_mul, one_mul, one_mul, cos_add, cos_add, sin_add, sin_add, ih.1, ih.2, hc, hs]
    exact ⟨rfl, rfl⟩

lemma kvPack_cos_sin_int (a b : ℝ) (hc : cos a = cos b) (hs : sin a = sin b) (n : ℤ) :
    cos (n * a) = cos (n * b) ∧ sin (n * a) = sin (n * b) := by
  rcases Int.eq_nat_or_neg n with ⟨m, rfl | rfl⟩
  · exact_mod_cast kvPack_cos_sin_nat a b hc hs m
  · have := kvPack_cos_sin_nat a b hc hs m
    push_cast
    simp only [neg_mul, cos_neg, sin_neg, this.1, this.2, and_self]

/-- One pair, at integer positions: equal logits for every query / key iff `ω' ≡ ω (mod 2π)`. -/
theorem kvPack_pair_iff (ω' ω : ℝ) :
    (∀ (q k : ℝ × ℝ) (i j : ℕ), pairScore ω' i j q k = pairScore ω i j q k) ↔
      cos ω' = cos ω ∧ sin ω' = sin ω := by
  constructor
  · intro h
    have h1 := h (1, 0) (1, 0) 1 0
    have h2 := h (1, 0) (0, 1) 1 0
    simp [pairScore_eq, dot2] at h1 h2
    exact ⟨h1, h2⟩
  · rintro ⟨hc, hs⟩ q k i j
    have := kvPack_cos_sin_int ω' ω hc hs ((i : ℤ) - j)
    push_cast at this
    rw [pairScore_eq, pairScore_eq, this.1, this.2]

/-- **RoPE packing: exactly when the slice frequencies match** (iff). The packed host logit equals
every target head's logit for all target query / key weights at all integer positions iff, on
every slice pair, the host frequency equals the target head's modulo `2π`. -/
theorem kvPack_rope_iff (σ : Fin G × Pt → Q) (hσ : Injective σ) (ωh : Q → ℝ)
    (ωt : Fin G → Pt → ℝ) :
    (∀ (g : Fin G) (q : Pt → ℝ × ℝ) (k : Fin G → Pt → ℝ × ℝ) (i j : ℕ),
        ropeScore ωh (kvPackQ σ g q) (kvPackK σ k) i j = ropeScore (ωt g) q (k g) i j) ↔
      ∀ g p, cos (ωh (σ (g, p))) = cos (ωt g p) ∧ sin (ωh (σ (g, p))) = sin (ωt g p) := by
  classical
  have hslice : ∀ g (q : Pt → ℝ × ℝ) (k : Fin G → Pt → ℝ × ℝ) (i j : ℝ),
      ropeScore ωh (kvPackQ σ g q) (kvPackK σ k) i j =
        ∑ p, pairScore (ωh (σ (g, p))) i j (q p) (k g p) := by
    intro g q k i j
    rw [kvPackQ, kvPackK, kvPack_score_slices σ hσ, Fintype.sum_prod_type, Finset.sum_eq_single g]
    · simp
    · intro b _ hb; simp [hb, kvPack_pairScore_zero]
    · simp
  constructor
  · intro h g p
    refine (kvPack_pair_iff _ _).mp (fun a b i j => ?_)
    have := h g (fun p' => if p' = p then a else 0) (fun _ p' => if p' = p then b else 0) i j
    rw [hslice, ropeScore, Finset.sum_eq_single p, Finset.sum_eq_single p] at this
    · simpa using this
    all_goals first
      | (intro b _ hb; simp [hb, kvPack_pairScore_zero])
      | simp
  · intro h g q k i j
    rw [hslice, ropeScore]
    exact Finset.sum_congr rfl (fun p _ => (kvPack_pair_iff _ _).mpr (h g p) _ _ i j)

/-! ## The config-only default schedule cannot pack two RoPE heads -/

lemma kvPack_defaultFreq_mem {θ : ℝ} (hθ : 1 < θ) {P : ℕ} (p : Fin P) :
    0 < defaultFreq θ P p ∧ defaultFreq θ P p ≤ 1 := by
  refine ⟨Real.rpow_pos_of_pos (by linarith) _, Real.rpow_le_one_of_one_le_of_nonpos hθ.le ?_⟩
  have : (0 : ℝ) ≤ 2 * (p : ℝ) := by positivity
  have : (0 : ℝ) ≤ 2 * (P : ℝ) := by positivity
  exact div_nonpos_of_nonpos_of_nonneg (by linarith) this

lemma kvPack_defaultFreq_inj {θ : ℝ} (hθ : 1 < θ) {P : ℕ} : Injective (defaultFreq θ P) := by
  intro a b h
  by_contra hab
  have hP : (0 : ℝ) < 2 * P := by
    have : 0 < P := Fin.pos a
    positivity
  have key : ∀ x y : Fin P, (x : ℕ) < y → defaultFreq θ P y < defaultFreq θ P x := by
    intro x y hxy
    apply Real.rpow_lt_rpow_of_exponent_lt hθ
    have : (x : ℝ) < y := by exact_mod_cast hxy
    rw [div_lt_div_iff_of_pos_right hP]
    linarith
  rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hab) with hl | hl
  · exact (key a b hl).ne' h
  · exact (key b a hl).ne h

/-- **The default schedule refutes config-only RoPE packing.** Two target RoPE heads (default
schedule, any base `θ_t`, at least one pair) cannot share one host KV head whose frequencies are
the default schedule with any base `θ_h > 1`, under any placement `σ`: pair 0 of every target head
has frequency 1, and only one host pair has a frequency ≡ 1 (mod 2π). With
`kvPack_rope_iff`, the packed host is then wrong for some target weights and positions. -/
theorem kvPack_default_fails {Ph P : ℕ} (θh θt : ℝ) (hθh : 1 < θh) (hP : 0 < P)
    (σ : Fin G × Fin P → Fin Ph) (hσ : Injective σ) (g₁ g₂ : Fin G) (hg : g₁ ≠ g₂) :
    ¬ ∀ g p, cos (defaultFreq θh Ph (σ (g, p))) = cos (defaultFreq θt P p) := by
  intro h
  have h0 : ∀ g, defaultFreq θh Ph (σ (g, ⟨0, hP⟩)) = 1 := by
    intro g
    have hc := h g ⟨0, hP⟩
    simp only [defaultFreq, Nat.cast_zero, mul_zero, neg_zero, zero_div,
      Real.rpow_zero] at hc ⊢
    have hm := kvPack_defaultFreq_mem hθh (σ (g, ⟨0, hP⟩))
    simp only [defaultFreq] at hm
    have hpi : (1 : ℝ) ≤ π := by linarith [Real.pi_gt_three]
    exact Real.injOn_cos ⟨hm.1.le, hm.2.trans hpi⟩ ⟨zero_le_one, hpi⟩ hc
  have := kvPack_defaultFreq_inj hθh ((h0 g₁).trans (h0 g₂).symm)
  exact hg (congrArg Prod.fst (hσ this))

/-- **…as a statement about the head.** With the default host schedule (`θ_h > 1`) and two default
target RoPE heads packed in one host KV head, some target weights and integer positions give a
different logit. -/
theorem kvPack_default_not_exact {Ph P : ℕ} (θh θt : ℝ) (hθh : 1 < θh) (hP : 0 < P)
    (σ : Fin G × Fin P → Fin Ph) (hσ : Injective σ) (g₁ g₂ : Fin G) (hg : g₁ ≠ g₂) :
    ¬ ∀ (g : Fin G) (q : Fin P → ℝ × ℝ) (k : Fin G → Fin P → ℝ × ℝ) (i j : ℕ),
        ropeScore (defaultFreq θh Ph) (kvPackQ σ g q) (kvPackK σ k) i j =
          ropeScore (defaultFreq θt P) q (k g) i j := by
  intro h
  exact kvPack_default_fails θh θt hθh hP σ hσ g₁ g₂ hg (fun g p =>
    (((kvPack_rope_iff σ hσ (defaultFreq θh Ph) (fun _ => defaultFreq θt P)).mp h) g p).1)

end EP
