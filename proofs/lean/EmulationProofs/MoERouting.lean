import EmulationProofs.MoE

/-!
# MoE ← MoE: expert duplication and the routing conditions

A router produces, per token, SELECTION scores `u` (logits, plus a selection-only bias such as
DeepSeek-V3's `e_score_correction_bias`) and GATE scores `z`. A top-k selection is any set of `k`
experts whose selection scores are `≥` every other's (`IsTopKOn`): the tie-break is a choice.
The gates are either

* renormalised (`renormOut`): `φ(z_e) / Σ_{j∈S} φ(z_j)` on the selected set (`φ = exp`:
  softmax-after-top-k or `norm_topk_prob = True`; `φ = σ`: DeepSeek-V3-style sigmoid scores), or
* not renormalised (`plainOut`): `exp(z_e) / Σ_{all j} exp(z_j)` (softmax-then-top-k with
  `norm_topk_prob = False`: Qwen2-MoE, OLMoE, DeepSeek-V2, …),

and `routed_scaling_factor` multiplies the result. The expert outputs at this token are vectors
`Y e`.

**Expert duplication** (the host of `serving/embed.py`): the host's experts are `ι × Fin m`
(copy `r` of target expert `e`) plus extras `D`. Copies have the target's router rows, so the same
selection and gate scores. Their outputs satisfy `Σ_r Yh(e, r) = m · Y e`: whole copies (`Yh = Y`),
or `m` slices of the expert with down-projections `×m` (so expert width `m · f_h ≥ f_t` suffices).

* `moeR_copies_topk`: if the target's top-k set is strict (no tie across its boundary) and every
  extra scores below every selected expert, EVERY host top-(k·m) set is exactly the copies.
* `moeR_dup_renorm`: renormalised, the copies reproduce the target exactly, whatever the extras.
* `moeR_dup_plain_factor`: not renormalised, the host is the target times `mZ / (mZ + Z_D)`
  (`Z`: the target's softmax denominator, `Z_D = Σ_d exp(z_d)`). Exact iff there are no extras
  (`moeR_dup_plain`, `E_h = m E_t`), never exact with extras on a nonzero output
  (`moeR_dup_plain_extras_ne`), within `Z_D / (m Z)` relative error (`moeR_dup_plain_extras_bound`).
* `moeR_scaling_fold`: `routed_scaling_factor` folds into the down-projections.
* `moeR_tie_split`: at a tie across the target's boundary, the host top-(k·m) set may split
  the copies of the tied experts; its output then equals no target tie-break.
* `moeR_group_copies`: group-limited routing (`n_group`, `topk_group`) survives duplication if
  copy `r` of group `g` forms its own host group `(g, r)`, the host has `m·n_group` groups and keeps
  `m·topk_group`, and both levels are strict. `moeR_group_top2sum_dup_flips`: with copies placed
  in their original group (same `n_group`, `topk_group`), a top-2-sum group score (DeepSeek-V3)
  ranks the groups differently, so the "same group shape" rule is not enough.

**Necessity, within the duplication family** (host experts are copies of target experts with
one common constant down-projection scale; the target's expert outputs at the token linearly
independent):

* `moeR_dup_needs_divisibility`: if the host matches at a token, it selects the same number `n`
  of copies of every selected target expert and none of the others, so `k_h = n k_t`: `k_t ∣ k_h`.
* `moeR_dup_needs_experts`: if every target expert is selected at some matched token,
  `k_h E_t ≤ k_t E_h`, i.e. `E_h ≥ (k_h/k_t) E_t`.
* `moeR_renorm_host_needs_const_mass`, `moeR_plain_host_needs_const_mass`: a renormalising
  duplication host cannot match a non-renormalising target, nor the reverse, once the target's
  selected softmax mass `P_S` takes two values on tokens with nonzero output
  (`moeR_mass_varies`: it does for two experts, top-1).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

section routing

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- `S` is a top-`k` set of the scores `u` within the allowed experts `U` (any tie-break). -/
def IsTopKOn {ι : Type*} (U : Finset ι) (u : ι → ℝ) (k : ℕ) (S : Finset ι) : Prop :=
  S ⊆ U ∧ S.card = k ∧ ∀ i ∈ S, ∀ j ∈ U, j ∉ S → u j ≤ u i

/-- No tie across the boundary of `S`: every selected expert beats every unselected one. -/
def StrictTopOn {ι : Type*} (U : Finset ι) (u : ι → ℝ) (S : Finset ι) : Prop :=
  ∀ i ∈ S, ∀ j ∈ U, j ∉ S → u j < u i

/-- **A strict top-k set is the only top-k set.** -/
theorem moeR_topk_unique {ι : Type*} [DecidableEq ι] {U : Finset ι} {u : ι → ℝ} {k : ℕ}
    {S T : Finset ι} (hS : IsTopKOn U u k S) (hstr : StrictTopOn U u S) (hT : IsTopKOn U u k T) :
    T = S := by
  have hTS : T ⊆ S := by
    intro t ht
    by_contra htS
    have hex : ∃ s ∈ S, s ∉ T := by
      by_contra hno
      push Not at hno
      have : S = T := Finset.eq_of_subset_of_card_le hno (by rw [hS.2.1, hT.2.1])
      exact htS (this ▸ ht)
    obtain ⟨s, hs, hsT⟩ := hex
    have h1 := hT.2.2 t ht s (hS.1 hs) hsT
    have h2 := hstr s hs t (hT.1 ht) htS
    linarith
  exact Finset.eq_of_subset_of_card_le hTS (by rw [hS.2.1, hT.2.1])

/-- Renormalised gates over the selected set, score map `φ` (`exp`: softmax; `σ`: sigmoid). -/
def renormOut {ι : Type*} (φ : ℝ → ℝ) (z : ι → ℝ) (S : Finset ι) (Y : ι → V) : V :=
  (∑ j ∈ S, φ (z j))⁻¹ • ∑ e ∈ S, φ (z e) • Y e

/-- Softmax over ALL experts, restricted to the selected set (no renormalisation). -/
def plainOut {ι : Type*} [Fintype ι] (z : ι → ℝ) (S : Finset ι) (Y : ι → V) : V :=
  (∑ j, exp (z j))⁻¹ • ∑ e ∈ S, exp (z e) • Y e

/-- The host's scores: copy `(e, r)` reads target expert `e`'s router row; extras read `zd`. -/
def dupScore {ι D : Type*} {m : ℕ} (z : ι → ℝ) (zd : D → ℝ) : ι × Fin m ⊕ D → ℝ :=
  Sum.elim (fun h => z h.1) zd

/-- The copies of a target set. -/
def copies {ι D : Type*} (m : ℕ) (S : Finset ι) : Finset (ι × Fin m ⊕ D) :=
  (S ×ˢ (univ : Finset (Fin m))).disjSum ∅

lemma moeR_card_copies {ι D : Type*} (m : ℕ) (S : Finset ι) :
    (copies (D := D) m S).card = S.card * m := by
  simp [copies, Finset.card_product]

lemma moeR_mem_copies_inl {ι D : Type*} (m : ℕ) (S : Finset ι) (e : ι) (r : Fin m) :
    (Sum.inl (e, r) : ι × Fin m ⊕ D) ∈ copies m S ↔ e ∈ S := by
  simp [copies]

lemma moeR_not_mem_copies_inr {ι D : Type*} (m : ℕ) (S : Finset ι) (d : D) :
    (Sum.inr d : ι × Fin m ⊕ D) ∉ copies m S := by
  simp [copies]

/-- **The host's top-(k·m) set is the copies** (selection, exact). If the target's top-k set `S`
is strict and every extra expert scores below every selected one (a router bias or a free
hidden dimension reading `−M`), every host top-(k·m) set, whatever its tie-break, is the copies
of `S`. -/
theorem moeR_copies_topk {ι D : Type*} [Fintype ι] [Fintype D] [DecidableEq ι] [DecidableEq D]
    {m k : ℕ} {u : ι → ℝ} {ud : D → ℝ} {S : Finset ι} (hS : IsTopKOn univ u k S)
    (hstr : StrictTopOn univ u S) (hd : ∀ d, ∀ i ∈ S, ud d < u i)
    {Sh : Finset (ι × Fin m ⊕ D)} (hSh : IsTopKOn univ (dupScore u ud) (k * m) Sh) :
    Sh = copies m S := by
  refine moeR_topk_unique ⟨Finset.subset_univ _, by rw [moeR_card_copies, hS.2.1], ?_⟩ ?_ hSh
  · intro i hi j _ hj
    rcases i with ⟨e, r⟩ | d
    · rw [moeR_mem_copies_inl] at hi
      rcases j with ⟨j, s⟩ | d
      · rw [moeR_mem_copies_inl] at hj
        exact (hstr e hi j (Finset.mem_univ _) hj).le
      · exact (hd d e hi).le
    · exact absurd hi (moeR_not_mem_copies_inr _ _ _)
  · intro i hi j _ hj
    rcases i with ⟨e, r⟩ | d
    · rw [moeR_mem_copies_inl] at hi
      rcases j with ⟨j, s⟩ | d
      · rw [moeR_mem_copies_inl] at hj
        exact hstr e hi j (Finset.mem_univ _) hj
      · exact hd d e hi
    · exact absurd hi (moeR_not_mem_copies_inr _ _ _)

/-- The copies' gate sums: `Σ_{h ∈ copies} f h = Σ_{e∈S} Σ_r f (e, r)`. -/
lemma moeR_sum_copies {ι D : Type*} {M : Type*} [AddCommMonoid M] {m : ℕ} (S : Finset ι)
    (f : ι × Fin m ⊕ D → M) :
    ∑ h ∈ copies (D := D) m S, f h = ∑ e ∈ S, ∑ r : Fin m, f (Sum.inl (e, r)) := by
  simp [copies, Finset.sum_product]

/-- **Expert duplication, renormalised: exact.** Host top-(k·m) on the copies (forced by
`moeR_copies_topk`), copies' outputs summing to `m · Y e` (whole copies, or `m` slices with
down-projections `×m`), renormalised gates with the same score map `φ`: the host equals the
target, whatever the extras. Each copy gets `p_e / m`. -/
theorem moeR_dup_renorm {ι D : Type*} {m : ℕ} (hm : m ≠ 0) (φ : ℝ → ℝ) (z : ι → ℝ)
    (zd : D → ℝ) (S : Finset ι) (Y : ι → V) (Yh : ι × Fin m → V) (Yd : D → V)
    (hY : ∀ e ∈ S, ∑ r, Yh (e, r) = (m : ℝ) • Y e) :
    renormOut φ (dupScore z zd) (copies m S) (Sum.elim Yh Yd) = renormOut φ z S Y := by
  unfold renormOut
  rw [moeR_sum_copies, moeR_sum_copies]
  simp only [dupScore, Sum.elim_inl, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  rw [← Finset.mul_sum]
  have : ∑ e ∈ S, ∑ r : Fin m, φ (z e) • Yh (e, r) = (m : ℝ) • ∑ e ∈ S, φ (z e) • Y e := by
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl (fun e he => ?_)
    rw [← Finset.smul_sum, hY e he, smul_comm]
  rw [this, smul_smul]
  congr 1
  have : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  rw [mul_inv]; field_simp

/-- **Expert duplication, not renormalised: the dilution factor.** With softmax over ALL host
experts, the copies' total gate for expert `e` is `m e^{z_e} / (m Z + Z_D)`, so the host is the
target times `m Z / (m Z + Z_D)`, with `Z = Σ_e e^{z_e}` and `Z_D = Σ_d e^{z_d}` over the extras.
(Copies with unscaled down-projections; a constant rescale cannot fix a token-dependent factor.) -/
theorem moeR_dup_plain_factor {ι D : Type*} [Fintype ι] [Nonempty ι] [Fintype D] {m : ℕ}
    (hm : m ≠ 0) (z : ι → ℝ) (zd : D → ℝ) (S : Finset ι) (Y : ι → V) (Yh : ι × Fin m → V)
    (Yd : D → V) (hY : ∀ e ∈ S, ∑ r, Yh (e, r) = (m : ℝ) • Y e) :
    plainOut (dupScore z zd) (copies m S) (Sum.elim Yh Yd) =
      ((m * ∑ j, exp (z j)) / (m * ∑ j, exp (z j) + ∑ d, exp (zd d))) • plainOut z S Y := by
  unfold plainOut
  rw [moeR_sum_copies]
  simp only [dupScore, Sum.elim_inl, Fintype.sum_sum_type, Sum.elim_inr, Fintype.sum_prod_type,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hZ : 0 < ∑ j, exp (z j) := Finset.sum_pos (fun _ _ => exp_pos _) Finset.univ_nonempty
  have hD : 0 ≤ ∑ d, exp (zd d) := Finset.sum_nonneg (fun _ _ => (exp_pos _).le)
  have hm' : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  have this : ∑ e ∈ S, ∑ r : Fin m, exp (z e) • Yh (e, r) = (m : ℝ) • ∑ e ∈ S, exp (z e) • Y e := by
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl (fun e he => ?_)
    rw [← Finset.smul_sum, hY e he, smul_comm]
  rw [← Finset.mul_sum, this, smul_smul, smul_smul]
  congr 1
  field_simp

/-- **…exact with no extras** (`E_h = m E_t`). -/
theorem moeR_dup_plain {ι D : Type*} [Fintype ι] [Nonempty ι] [Fintype D] [IsEmpty D] {m : ℕ}
    (hm : m ≠ 0) (z : ι → ℝ) (zd : D → ℝ) (S : Finset ι) (Y : ι → V) (Yh : ι × Fin m → V)
    (Yd : D → V) (hY : ∀ e ∈ S, ∑ r, Yh (e, r) = (m : ℝ) • Y e) :
    plainOut (dupScore z zd) (copies m S) (Sum.elim Yh Yd) = plainOut z S Y := by
  rw [moeR_dup_plain_factor hm z zd S Y Yh Yd hY]
  have hZ : 0 < ∑ j, exp (z j) := Finset.sum_pos (fun _ _ => exp_pos _) Finset.univ_nonempty
  have hm' : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  simp only [Finset.univ_eq_empty, Finset.sum_empty, add_zero]
  rw [div_self (by positivity), one_smul]

/-- **…never exact with an extra expert** (on a token with nonzero output): `Z_D > 0`, so the
factor is `< 1`. The `−M` dummy rows of `serving/embed.py` are exact only through floating-point
underflow; over the reals they are the limit `Z_D → 0` (`moeR_dup_plain_extras_bound`). -/
theorem moeR_dup_plain_extras_ne {ι D : Type*} [Fintype ι] [Nonempty ι] [Fintype D] [Nonempty D]
    {m : ℕ} (hm : m ≠ 0) (z : ι → ℝ) (zd : D → ℝ) (S : Finset ι) (Y : ι → V)
    (Yh : ι × Fin m → V) (Yd : D → V) (hY : ∀ e ∈ S, ∑ r, Yh (e, r) = (m : ℝ) • Y e)
    (hne : plainOut z S Y ≠ 0) :
    plainOut (dupScore z zd) (copies m S) (Sum.elim Yh Yd) ≠ plainOut z S Y := by
  rw [moeR_dup_plain_factor hm z zd S Y Yh Yd hY]
  intro h
  have hZ : 0 < ∑ j, exp (z j) := Finset.sum_pos (fun _ _ => exp_pos _) Finset.univ_nonempty
  have hD : 0 < ∑ d, exp (zd d) := Finset.sum_pos (fun _ _ => exp_pos _) Finset.univ_nonempty
  have hm' : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  have h1 : ((m * ∑ j, exp (z j)) / (m * ∑ j, exp (z j) + ∑ d, exp (zd d)) - 1) •
      plainOut z S Y = 0 := by rw [sub_smul, one_smul, h, sub_self]
  rcases smul_eq_zero.mp h1 with h2 | h2
  · rw [sub_eq_zero, div_eq_one_iff_eq (by positivity)] at h2
    linarith
  · exact hne h2

/-- **…and within `Z_D/(m Z)` relative error.** -/
theorem moeR_dup_plain_extras_bound {ι D : Type*} [Fintype ι] [Nonempty ι] [Fintype D] {m : ℕ}
    (hm : m ≠ 0) (z : ι → ℝ) (zd : D → ℝ) (S : Finset ι) (Y : ι → V) (Yh : ι × Fin m → V)
    (Yd : D → V) (hY : ∀ e ∈ S, ∑ r, Yh (e, r) = (m : ℝ) • Y e) :
    ‖plainOut (dupScore z zd) (copies m S) (Sum.elim Yh Yd) - plainOut z S Y‖ ≤
      (∑ d, exp (zd d)) / (m * ∑ j, exp (z j)) * ‖plainOut z S Y‖ := by
  rw [moeR_dup_plain_factor hm z zd S Y Yh Yd hY]
  have hZ : 0 < ∑ j, exp (z j) := Finset.sum_pos (fun _ _ => exp_pos _) Finset.univ_nonempty
  have hD : 0 ≤ ∑ d, exp (zd d) := Finset.sum_nonneg (fun _ _ => (exp_pos _).le)
  have hm' : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  set A := (m : ℝ) * ∑ j, exp (z j)
  set B := ∑ d, exp (zd d)
  have hA : 0 < A := by positivity
  have e1 : (A / (A + B)) • plainOut z S Y - plainOut z S Y = (-(B / (A + B))) • plainOut z S Y := by
    have : A / (A + B) - 1 = -(B / (A + B)) := by field_simp; ring
    rw [← this, sub_smul, one_smul]
  rw [e1, norm_smul, norm_neg, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
  exact div_le_div_of_nonneg_left hD hA (by linarith)

lemma moeR_renormOut_smul {ι : Type*} (φ : ℝ → ℝ) (z : ι → ℝ) (S : Finset ι) (Y : ι → V)
    (a : ℝ) : renormOut φ z S (fun e => a • Y e) = a • renormOut φ z S Y := by
  simp only [renormOut, Finset.smul_sum, smul_smul]
  refine Finset.sum_congr rfl (fun e _ => ?_)
  ring_nf

lemma moeR_plainOut_smul {ι : Type*} [Fintype ι] (z : ι → ℝ) (S : Finset ι) (Y : ι → V)
    (a : ℝ) : plainOut z S (fun e => a • Y e) = a • plainOut z S Y := by
  simp only [plainOut, Finset.smul_sum, smul_smul]
  refine Finset.sum_congr rfl (fun e _ => ?_)
  ring_nf

/-- **`routed_scaling_factor` folds into the down-projections** (exact): a host scale `s_h ≠ 0`
with expert outputs `×(s_t/s_h)` equals a target scale `s_t`, for either gate kind. -/
theorem moeR_scaling_fold {ι : Type*} (φ : ℝ → ℝ) (z : ι → ℝ) (S : Finset ι) (Y : ι → V)
    (sh st : ℝ) (hs : sh ≠ 0) :
    sh • renormOut φ z S (fun e => (st / sh) • Y e) = st • renormOut φ z S Y := by
  rw [moeR_renormOut_smul, smul_smul, mul_div_cancel₀ _ hs]

theorem moeR_scaling_fold_plain {ι : Type*} [Fintype ι] (z : ι → ℝ) (S : Finset ι) (Y : ι → V)
    (sh st : ℝ) (hs : sh ≠ 0) :
    sh • plainOut z S (fun e => (st / sh) • Y e) = st • plainOut z S Y := by
  rw [moeR_plainOut_smul, smul_smul, mul_div_cancel₀ _ hs]

/-! ## Ties -/

/-- **At a tie the host may split the copies** (two target experts with equal scores, `k = 1`,
`m = 2`). Taking one copy of each is a valid host top-2 set, and its renormalised output
`(Y 0 + Y 1)/2` equals the target's output under NO tie-break once `Y 0 ≠ Y 1`. Exactness at
ties needs the host's tie-break to keep copy classes together (then `moeR_dup_renorm` applies to
whichever top-k set the target chose); `torch.topk`'s tie order is not documented. -/
theorem moeR_tie_split (φ : ℝ → ℝ) (hφ : φ 0 ≠ 0) (Y : Fin 2 → V) (hY : Y 0 ≠ Y 1) :
    IsTopKOn univ (dupScore (m := 2) (fun _ : Fin 2 => (0 : ℝ)) (fun _ : Empty => 0)) (1 * 2)
      {Sum.inl (0, 0), Sum.inl (1, 0)} ∧
    ∀ T, IsTopKOn univ (fun _ : Fin 2 => (0 : ℝ)) 1 T →
      renormOut φ (dupScore (m := 2) (fun _ : Fin 2 => (0 : ℝ)) (fun _ : Empty => 0))
        {Sum.inl (0, 0), Sum.inl (1, 0)} (Sum.elim (fun h => Y h.1) (fun _ => 0)) ≠
      renormOut φ (fun _ => 0) T Y := by
  refine ⟨⟨Finset.subset_univ _, by decide, fun i _ j _ _ => ?_⟩, fun T hT => ?_⟩
  · rcases i with i | i <;> rcases j with j | j <;> simp [dupScore]
  · obtain ⟨a, rfl⟩ := Finset.card_eq_one.mp hT.2.1
    have hne : (Sum.inl (0, 0) : Fin 2 × Fin 2 ⊕ Empty) ≠ Sum.inl (1, 0) := by decide
    simp only [renormOut, Finset.sum_pair hne, Finset.sum_singleton, dupScore, Sum.elim_inl]
    rw [smul_smul, inv_mul_cancel₀ hφ, one_smul]
    intro h
    have hh : (φ 0 + φ 0)⁻¹ • (φ 0 • Y 0 + φ 0 • Y 1) = (1 / 2 : ℝ) • (Y 0 + Y 1) := by
      rw [← smul_add, smul_smul]; congr 1; field_simp; ring
    rw [hh] at h
    have h2 : Y 0 + Y 1 = (2 : ℝ) • Y a := by rw [← h, smul_smul]; norm_num
    fin_cases a
    · apply hY
      have : Y 1 = Y 0 := by
        have := h2; simp only [Fin.zero_eta, two_smul] at this
        exact add_left_cancel this
      exact this.symm
    · apply hY
      have := h2; simp only [Fin.mk_one, two_smul] at this
      exact add_right_cancel this

/-! ## Necessity within the duplication family -/

/-- **Divisibility is necessary** (duplication family). Host experts `κ`, each a copy of target
expert `c h`, one common nonzero scale (`λ`, which absorbs the down-projection scale and the
host's normaliser); target gates `μ w(z_e)` on `S`; `w` never 0 (`exp`, `σ`). If the expert outputs
at this token are linearly independent and the host matches, then the host selected the same
number `n` of copies of every expert in `S` and none of any other expert, so
`k_h = |Sh| = |S| · n = k_t n`. Renormalised (`λ = c/Σ_{Sh} w`, `μ = 1/Σ_S w`) or not
(`λ = c/Z_h`, `μ = 1/Z`). -/
theorem moeR_dup_needs_divisibility {ι κ : Type*} [Fintype ι] [DecidableEq ι] (c : κ → ι)
    (w : ℝ → ℝ) (hw : ∀ t, w t ≠ 0) (z : ι → ℝ) (lam mu : ℝ) (hlam : lam ≠ 0) (S : Finset ι)
    (Sh : Finset κ) (Y : ι → V) (hli : LinearIndependent ℝ Y)
    (hmatch : ∑ h ∈ Sh, (lam * w (z (c h))) • Y (c h) = ∑ e ∈ S, (mu * w (z e)) • Y e) :
    ∃ n : ℕ, (∀ e ∈ S, (Sh.filter (fun h => c h = e)).card = n) ∧
      (∀ e ∉ S, (Sh.filter (fun h => c h = e)).card = 0) ∧ Sh.card = S.card * n := by
  set N : ι → ℕ := fun e => (Sh.filter (fun h => c h = e)).card
  have hL : ∑ h ∈ Sh, (lam * w (z (c h))) • Y (c h) =
      ∑ e, ((N e : ℝ) * (lam * w (z e))) • Y e := by
    rw [← Finset.sum_fiberwise Sh c]
    refine Finset.sum_congr rfl (fun e _ => ?_)
    rw [Finset.sum_congr rfl (fun h hh => by rw [(Finset.mem_filter.mp hh).2]), Finset.sum_const,
      ← Nat.cast_smul_eq_nsmul ℝ, smul_smul]
  have hR : ∑ e ∈ S, (mu * w (z e)) • Y e =
      ∑ e, (if e ∈ S then mu * w (z e) else 0) • Y e := by
    simp only [ite_smul, zero_smul]
    rw [← Finset.sum_filter]
    congr 1
    ext e; simp
  have hcoef : ∀ e, (N e : ℝ) * (lam * w (z e)) = if e ∈ S then mu * w (z e) else 0 := by
    have := Fintype.linearIndependent_iff.mp hli
      (fun e => (N e : ℝ) * (lam * w (z e)) - if e ∈ S then mu * w (z e) else 0)
      (by simp only [sub_smul, Finset.sum_sub_distrib]; rw [← hL, ← hR, hmatch, sub_self])
    intro e; exact sub_eq_zero.mp (this e)
  have hout : ∀ e ∉ S, N e = 0 := by
    intro e he
    have := hcoef e
    simp only [he, ↓reduceIte] at this
    rcases mul_eq_zero.mp this with h | h
    · exact_mod_cast h
    · exact absurd h (mul_ne_zero hlam (hw _))
  have hin : ∀ e ∈ S, (N e : ℝ) = mu / lam := by
    intro e he
    have := hcoef e
    simp only [he, ↓reduceIte] at this
    field_simp [hw (z e)]
    have hw' := hw (z e)
    apply mul_right_cancel₀ hw'
    linarith [this]
  by_cases hS : S.Nonempty
  · obtain ⟨e0, he0⟩ := hS
    refine ⟨N e0, fun e he => ?_, hout, ?_⟩
    · exact_mod_cast (hin e he).trans (hin e0 he0).symm
    · rw [Finset.card_eq_sum_card_fiberwise (f := c) (t := univ) (fun _ _ => Finset.mem_coe.mpr
        (Finset.mem_univ _))]
      change ∑ e, N e = S.card * N e0
      rw [← Finset.sum_subset (Finset.subset_univ S) (fun e _ he => hout e he)]
      rw [Finset.sum_congr rfl (fun e he => (show N e = N e0 by
        exact_mod_cast (hin e he).trans (hin e0 he0).symm)), Finset.sum_const, smul_eq_mul]
  · rw [Finset.not_nonempty_iff_eq_empty] at hS
    subst hS
    refine ⟨0, by simp, hout, ?_⟩
    rw [Finset.card_eq_sum_card_fiberwise (f := c) (t := univ) (fun _ _ => Finset.mem_coe.mpr
        (Finset.mem_univ _))]
    simp only [Finset.card_empty, zero_mul]
    exact Finset.sum_eq_zero (fun e _ => hout e (Finset.notMem_empty e))

/-- **Enough experts is necessary** (duplication family). If every target expert `e` is selected
at some matched token, where (by `moeR_dup_needs_divisibility`) the host picks `k_h / k_t` of its
copies, then `k_h E_t ≤ k_t E_h`. -/
theorem moeR_dup_needs_experts {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq ι]
    (c : κ → ι) (kt kh : ℕ)
    (h : ∀ e, ∃ Sh : Finset κ, kt * (Sh.filter (fun h => c h = e)).card = kh) :
    kh * Fintype.card ι ≤ kt * Fintype.card κ := by
  rw [← Finset.card_univ (α := κ), Finset.card_eq_sum_card_fiberwise (f := c) (t := univ)
    (fun _ _ => Finset.mem_coe.mpr (Finset.mem_univ _)), Finset.mul_sum, mul_comm,
    ← Finset.card_univ, ← smul_eq_mul, ← Finset.sum_const]
  refine Finset.sum_le_sum (fun e _ => ?_)
  obtain ⟨Sh, hSh⟩ := h e
  rw [← hSh]
  exact Nat.mul_le_mul_left _ (Finset.card_le_card (Finset.filter_subset_filter _
    (Finset.subset_univ _)))

/-- The selected softmax mass `P_S = Σ_{e∈S} e^{z_e} / Σ_e e^{z_e}`. -/
def selMass {ι : Type*} [Fintype ι] (z : ι → ℝ) (S : Finset ι) : ℝ :=
  (∑ e ∈ S, exp (z e)) / ∑ e, exp (z e)

/-- No renormalisation = the selected mass times the renormalised output. -/
lemma moeR_plain_eq_mass {ι : Type*} [Fintype ι] [Nonempty ι] (z : ι → ℝ) (S : Finset ι)
    (Y : ι → V) : plainOut z S Y = selMass z S • renormOut exp z S Y := by
  unfold plainOut renormOut selMass
  rw [smul_smul]
  by_cases hS : S = ∅
  · subst hS; simp
  · have : 0 < ∑ e ∈ S, exp (z e) :=
      Finset.sum_pos (fun _ _ => exp_pos _) (Finset.nonempty_iff_ne_empty.mpr hS)
    congr 1
    field_simp

/-- **A renormalising duplication host cannot play a non-renormalising target.** Host copies with
any common down-projection scale `c` and the target's router rows give `c ·` (the renormalised
output) (`moeR_dup_renorm`), while the target gives `P_S ·` it. Matching at two tokens with
nonzero output forces the target's selected mass to be the same at both. -/
theorem moeR_renorm_host_needs_const_mass {ι D : Type*} [Fintype ι] [Nonempty ι] {m : ℕ}
    (hm : m ≠ 0) (c : ℝ) (z₁ z₂ : ι → ℝ) (zd₁ zd₂ : D → ℝ) (S₁ S₂ : Finset ι) (Y₁ Y₂ : ι → V)
    (Yd : D → V) (hR₁ : renormOut exp z₁ S₁ Y₁ ≠ 0) (hR₂ : renormOut exp z₂ S₂ Y₂ ≠ 0)
    (h₁ : renormOut exp (dupScore z₁ zd₁) (copies m S₁) (Sum.elim (fun h => c • Y₁ h.1) Yd) =
      plainOut z₁ S₁ Y₁)
    (h₂ : renormOut exp (dupScore z₂ zd₂) (copies m S₂) (Sum.elim (fun h => c • Y₂ h.1) Yd) =
      plainOut z₂ S₂ Y₂) :
    selMass z₁ S₁ = selMass z₂ S₂ := by
  have key : ∀ (z : ι → ℝ) (zd : D → ℝ) (S : Finset ι) (Y : ι → V),
      renormOut exp z S Y ≠ 0 →
      renormOut exp (dupScore z zd) (copies m S) (Sum.elim (fun h => c • Y h.1) Yd) =
        plainOut z S Y → c = selMass z S := by
    intro z zd S Y hR h
    rw [moeR_dup_renorm hm exp z zd S (fun e => c • Y e) _ Yd (fun e _ => by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, Nat.cast_smul_eq_nsmul]),
      moeR_renormOut_smul,
      moeR_plain_eq_mass] at h
    have : (c - selMass z S) • renormOut exp z S Y = 0 := by rw [sub_smul, h, sub_self]
    exact sub_eq_zero.mp ((smul_eq_zero.mp this).resolve_right hR)
  rw [← key z₁ zd₁ S₁ Y₁ hR₁ h₁, ← key z₂ zd₂ S₂ Y₂ hR₂ h₂]

/-- **…nor the reverse: a non-renormalising duplication host (no extras) cannot play a
renormalising target.** The host gives `c P_S ·` (the renormalised output); matching at two
tokens with nonzero output forces equal selected masses. -/
theorem moeR_plain_host_needs_const_mass {ι D : Type*} [Fintype ι] [Nonempty ι] [Fintype D]
    [IsEmpty D] {m : ℕ} (hm : m ≠ 0) (c : ℝ) (z₁ z₂ : ι → ℝ) (zd : D → ℝ) (S₁ S₂ : Finset ι)
    (Y₁ Y₂ : ι → V) (Yd : D → V) (hR₁ : renormOut exp z₁ S₁ Y₁ ≠ 0)
    (hR₂ : renormOut exp z₂ S₂ Y₂ ≠ 0)
    (h₁ : plainOut (dupScore z₁ zd) (copies m S₁) (Sum.elim (fun h => c • Y₁ h.1) Yd) =
      renormOut exp z₁ S₁ Y₁)
    (h₂ : plainOut (dupScore z₂ zd) (copies m S₂) (Sum.elim (fun h => c • Y₂ h.1) Yd) =
      renormOut exp z₂ S₂ Y₂) :
    selMass z₁ S₁ = selMass z₂ S₂ := by
  have key : ∀ (z : ι → ℝ) (S : Finset ι) (Y : ι → V), renormOut exp z S Y ≠ 0 →
      plainOut (dupScore z zd) (copies m S) (Sum.elim (fun h => c • Y h.1) Yd) =
        renormOut exp z S Y → c * selMass z S = 1 := by
    intro z S Y hR h
    rw [moeR_dup_plain hm z zd S (fun e => c • Y e) _ Yd (fun e _ => by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, Nat.cast_smul_eq_nsmul]),
      moeR_plainOut_smul, moeR_plain_eq_mass,
      smul_smul] at h
    have : (c * selMass z S - 1) • renormOut exp z S Y = 0 := by
      rw [sub_smul, one_smul, h, sub_self]
    exact sub_eq_zero.mp ((smul_eq_zero.mp this).resolve_right hR)
  have e1 := key z₁ S₁ Y₁ hR₁ h₁
  have e2 := key z₂ S₂ Y₂ hR₂ h₂
  have hc : c ≠ 0 := by rintro rfl; simp at e1
  have := e1.trans e2.symm
  exact mul_left_cancel₀ hc this

/-- **The selected mass varies** (two experts, top-1, expert 0 selected): `e/(e+1) ≠ e²/(e²+1)`.
So the two theorems above apply to every such target with nonzero outputs. -/
theorem moeR_mass_varies :
    selMass (fun i : Fin 2 => if i = 0 then (1 : ℝ) else 0) {0} ≠
      selMass (fun i : Fin 2 => if i = 0 then (2 : ℝ) else 0) {0} := by
  simp only [selMass, Finset.sum_singleton, Fin.sum_univ_two]
  norm_num
  intro h
  have h1 : exp 1 ≠ 1 := by
    have := Real.add_one_lt_exp (x := 1) one_ne_zero; linarith
  have hp1 : 0 < exp 1 + 1 := by positivity
  have hp2 : 0 < exp 2 + 1 := by positivity
  rw [div_eq_div_iff hp1.ne' hp2.ne'] at h
  have h2 : exp 2 = exp 1 * exp 1 := by rw [← exp_add]; norm_num
  rw [h2] at h
  have : exp 1 * (exp 1 - 1) = 0 := by nlinarith
  rcases mul_eq_zero.mp this with h3 | h3
  · exact (exp_pos 1).ne' h3
  · exact h1 (by linarith)

/-! ## Group-limited routing -/

/-- Group-limited selection: keep a top-`tg` set `A` of groups by group score `gs`, then a top-`k`
set of experts inside the kept groups. -/
def IsGroupTopK {ι G : Type*} [Fintype ι] [Fintype G] [DecidableEq G] (grp : ι → G)
    (gs : G → ℝ) (tg : ℕ) (u : ι → ℝ) (k : ℕ) (S : Finset ι) : Prop :=
  ∃ A, IsTopKOn univ gs tg A ∧ IsTopKOn (univ.filter (fun e => grp e ∈ A)) u k S

lemma moeR_prod_top {ι : Type*} [DecidableEq ι] {U : Finset ι} {u : ι → ℝ} {k : ℕ}
    {S : Finset ι} (hS : IsTopKOn U u k S) (hstr : StrictTopOn U u S) (m : ℕ) :
    IsTopKOn (U ×ˢ (univ : Finset (Fin m))) (fun h => u h.1) (k * m) (S ×ˢ univ) ∧
      StrictTopOn (U ×ˢ (univ : Finset (Fin m))) (fun h => u h.1) (S ×ˢ univ) := by
  refine ⟨⟨Finset.product_subset_product_left hS.1, by simp [hS.2.1], ?_⟩, ?_⟩
  · intro i hi j hj hjS
    simp only [Finset.mem_product, Finset.mem_univ, and_true] at hi hj hjS
    exact hS.2.2 _ hi _ hj hjS
  · intro i hi j hj hjS
    simp only [Finset.mem_product, Finset.mem_univ, and_true] at hi hj hjS
    exact hstr _ hi _ hj hjS

/-- **Group-limited routing under duplication: a sufficient condition** (exact selection).
Copy `r` of target group `g` is host group `(g, r)`, with the target's group score (any score
computed from the group's logits, e.g. max or top-2 sum, is the same on each copy group); the host
has `m · n_group` groups and keeps `m · topk_group`, and selects `m k` experts. If both levels of
the target's selection are strict, every host group-limited selection is the copies of the
target's, so `moeR_dup_renorm` / `moeR_dup_plain` apply. -/
theorem moeR_group_copies {ι G : Type*} [Fintype ι] [Fintype G] [DecidableEq ι] [DecidableEq G]
    (grp : ι → G) (gs : G → ℝ) (tg k : ℕ) (u : ι → ℝ) (A : Finset G) (S : Finset ι)
    (hA : IsTopKOn univ gs tg A) (hAs : StrictTopOn univ gs A)
    (hS : IsTopKOn (univ.filter (fun e => grp e ∈ A)) u k S)
    (hSs : StrictTopOn (univ.filter (fun e => grp e ∈ A)) u S) (m : ℕ)
    (Sh : Finset (ι × Fin m))
    (hSh : IsGroupTopK (fun h : ι × Fin m => (grp h.1, h.2)) (fun g : G × Fin m => gs g.1)
      (tg * m) (fun h => u h.1) (k * m) Sh) :
    Sh = S ×ˢ univ := by
  obtain ⟨Ah, hAh, hSh'⟩ := hSh
  have hpA := moeR_prod_top hA hAs m
  rw [Finset.univ_product_univ] at hpA
  have hAeq : Ah = A ×ˢ univ := moeR_topk_unique hpA.1 hpA.2 hAh
  subst hAeq
  have hU : (univ.filter (fun h : ι × Fin m => (grp h.1, h.2) ∈ A ×ˢ (univ : Finset (Fin m)))) =
      (univ.filter (fun e => grp e ∈ A)) ×ˢ univ := by
    ext h; simp
  rw [hU] at hSh'
  have hpS := moeR_prod_top hS hSs m
  exact moeR_topk_unique hpS.1 hpS.2 hSh'

/-- **Same group shape is not enough under duplication** (top-2-sum group score, DeepSeek-V3 /
Ling). Target: 4 experts, scores `(3, 0, 2, 2)`, group A = experts 0, 1, group B = experts 2, 3.
Every top-2 set of A sums to 3 and of B to 4, so `topk_group = 1` keeps B. Host: 2 copies of each
expert, kept in the original group (same `n_group`, `topk_group`): every top-2 set of A sums to 6
and of B to 4, so the host keeps A. The host then selects only A's experts, the target only B's. -/
theorem moeR_group_top2sum_dup_flips :
    let u : Fin 4 → ℝ := ![3, 0, 2, 2]
    let inA : Fin 4 → Prop := fun e => e.val < 2
    (∀ T, IsTopKOn (univ.filter (fun e => inA e)) u 2 T → ∑ e ∈ T, u e = 3) ∧
    (∀ T, IsTopKOn (univ.filter (fun e => ¬ inA e)) u 2 T → ∑ e ∈ T, u e = 4) ∧
    (∀ T, IsTopKOn (univ.filter (fun h : Fin 4 × Fin 2 => inA h.1)) (fun h => u h.1) 2 T →
      ∑ h ∈ T, u h.1 = 6) ∧
    (∀ T, IsTopKOn (univ.filter (fun h : Fin 4 × Fin 2 => ¬ inA h.1)) (fun h => u h.1) 2 T →
      ∑ h ∈ T, u h.1 = 4) := by
  intro u inA
  refine ⟨fun T hT => ?_, fun T hT => ?_, fun T hT => ?_, fun T hT => ?_⟩
  · have : T = univ.filter (fun e => inA e) :=
      Finset.eq_of_subset_of_card_le hT.1 (by rw [hT.2.1]; decide)
    subst this
    have hA : univ.filter (fun e : Fin 4 => inA e) = {0, 1} := by decide
    rw [hA, Finset.sum_pair (by decide)]; simp [u]
  · have : T = univ.filter (fun e => ¬ inA e) :=
      Finset.eq_of_subset_of_card_le hT.1 (by rw [hT.2.1]; decide)
    subst this
    have hB : univ.filter (fun e : Fin 4 => ¬ inA e) = {2, 3} := by decide
    rw [hB, Finset.sum_pair (by decide)]; simp [u]; norm_num
  · have hS : IsTopKOn (univ.filter (fun h : Fin 4 × Fin 2 => inA h.1)) (fun h => u h.1) 2
        {(0, 0), (0, 1)} := by
      refine ⟨by decide, by decide, fun i hi j _ hj => ?_⟩
      simp only [Finset.mem_insert, Finset.mem_singleton] at hi
      rcases hi with rfl | rfl <;> fin_cases j <;> simp_all [u, inA]
    have hstr : StrictTopOn (univ.filter (fun h : Fin 4 × Fin 2 => inA h.1)) (fun h => u h.1)
        {(0, 0), (0, 1)} := by
      intro i hi j hjU hj
      simp only [Finset.mem_insert, Finset.mem_singleton] at hi
      rcases hi with rfl | rfl <;> fin_cases j <;> simp_all [u, inA]
    rw [moeR_topk_unique hS hstr hT]
    simp [u]; norm_num
  · have hval : ∀ h ∈ T, u h.1 = 2 := by
      intro h hh
      have := hT.1 hh
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, inA] at this
      rcases h with ⟨e, r⟩
      fin_cases e <;> simp_all [u]
    rw [Finset.sum_congr rfl hval, Finset.sum_const, hT.2.1]; norm_num

end routing


end EP
