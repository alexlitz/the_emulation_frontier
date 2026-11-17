import EmulationProofs.Position

/-!
# Position transitions into additive-bias and learned-absolute hosts

An additive-bias host adds an arbitrary per-(query, key) table `B i j` to the content logit.

* **trivial ones**: `nope_sub_alibi` (slope `0`), `nope_sub_learned_abs` (table `0`),
  `alibi_sub_additive_bias`, `t5_sub_additive_bias`, `nope_sub_additive_bias`.
* **alibi → learned_abs, norm-free** (exact, bounded context): a learned-absolute host writes the
  position into extra residual channels, and the head reads them with its (linear) q, k
  projections. Causal ALiBi needs two channels (a constant `1` and the position `t`): the key-side
  term `m j` is the ALiBi logit up to the per-query constant `−m i`, which softmax ignores
  (`alibi_sub_learned_abs_causal`). ANY additive table over `T` positions (bidirectional ALiBi
  `−m|i−j|`, T5 buckets) needs `T` channels, a one-hot position (`bias_sub_learned_abs`).
  The channels are carried in the residual; under a pre-norm the carry meets the scale obstruction
  (`prenorm_readout_obstruction`), so the pre-norm case is NOT proved.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {ι : Type*} [Fintype ι] {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- An additive-bias head at query `i`: content logits `c j` plus the bias row `B i j`. -/
def biasAttn {P : Type*} (S : Finset ι) (c : ι → ℝ) (B : P → ι → ℝ) (i : P) (v : ι → V) : V :=
  attn S (fun j => c j + B i j) v

/-- **NoPE ⊆ ALiBi** (exact): slope `0`. -/
theorem nope_sub_alibi (c i j : ℝ) : alibiScore c 0 i j = c := by simp [alibiScore]

/-- **NoPE ⊆ learned-absolute** (exact): a zero position table. -/
theorem nope_sub_learned_abs {Tok A : Type*} [AddCommGroup A] {T : ℕ} (e : Tok → A)
    (X : Fin T → Tok) : (fun t => e (X t) + (0 : Fin T → A) t) = fun t => e (X t) := by simp

/-- **ALiBi ⊆ additive bias** (exact): the table is the ramp `B i j = −m (i − j)`. -/
theorem alibi_sub_additive_bias {T : ℕ} (S : Finset (Fin T)) (c : Fin T → ℝ) (m : ℝ)
    (i : Fin T) (v : Fin T → V) :
    biasAttn S c (fun i j : Fin T => -(m * (((i : ℕ) : ℝ) - (j : ℕ)))) i v =
      attn S (fun j => alibiScore (c j) m (i : ℕ) (j : ℕ)) v := by
  simp only [biasAttn, alibiScore, sub_eq_add_neg]

/-- **T5 relative bias ⊆ additive bias** (exact): the table is the bucketed bias
`B i j = b (bucket (i − j))`, any bucketing. -/
theorem t5_sub_additive_bias {K : Type*} {T : ℕ} (S : Finset (Fin T)) (c : Fin T → ℝ)
    (b : K → ℝ) (bucket : ℤ → K) (i : Fin T) (v : Fin T → V) :
    biasAttn S c (fun i j : Fin T => b (bucket (((i : ℕ) : ℤ) - (j : ℕ)))) i v =
      attn S (fun j => c j + b (bucket (((i : ℕ) : ℤ) - (j : ℕ)))) v := rfl

/-- **NoPE ⊆ additive bias** (exact): the zero table. -/
theorem nope_sub_additive_bias {P : Type*} (S : Finset ι) (c : ι → ℝ) (i : P) (v : ι → V) :
    biasAttn S c (fun _ _ => 0) i v = attn S c v := by simp [biasAttn]

/-! ## alibi → learned_abs, norm-free -/

/-- The host query and key for causal ALiBi: content `q`, `k` plus one extra coordinate: the
query reads the constant channel (`1`), the key reads `m` times the position channel. Both are
linear in the residual `[x; 1; t]` that a learned-absolute table `P_t = [0; 1; t]` produces. -/
def alibiLAQ {κ : Type*} (q : κ → ℝ) : κ ⊕ Unit → ℝ := Sum.elim q (fun _ => 1)

def alibiLAK {κ : Type*} (m : ℝ) (k : κ → ℝ) (j : ℕ) : κ ⊕ Unit → ℝ :=
  Sum.elim k (fun _ => m * j)

/-- **ALiBi ⊆ learned-absolute, norm-free, one head** (exact). With the two extra coordinates the
host logit is `q·k_j + m j`, the ALiBi logit `q·k_j − m (i − j)` plus the per-query constant
`m i`; softmax ignores it (`attn_add_const`), so the attention output is ALiBi's, for any key set
(causal or not: the identity `q·k − m(i−j) + m i = q·k + m j` holds for every pair). -/
theorem alibi_sub_learned_abs_causal {κ : Type*} [Fintype κ] {T : ℕ} (S : Finset (Fin T))
    (m : ℝ) (i : Fin T) (q : κ → ℝ) (k : Fin T → κ → ℝ) (v : Fin T → V) :
    attn S (fun j => alibiLAQ q ⬝ᵥ alibiLAK m (k j) j) v =
      attn S (fun j => alibiScore (q ⬝ᵥ k j) m (i : ℕ) (j : ℕ)) v := by
  have h : (fun j : Fin T => alibiLAQ q ⬝ᵥ alibiLAK m (k j) j) =
      fun j => alibiScore (q ⬝ᵥ k j) m (i : ℕ) (j : ℕ) + m * (i : ℕ) := by
    funext j
    simp only [alibiLAQ, alibiLAK, dotProduct, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
      Finset.univ_unique, Finset.sum_singleton, one_mul, alibiScore]
    ring
  rw [h, attn_add_const]

/-- The one-hot position channels `e_t` of a learned-absolute table over `T` positions. -/
def onehot {T : ℕ} (t : Fin T) : Fin T → ℝ := fun s => if s = t then 1 else 0

/-- **Any additive bias ⊆ learned-absolute, norm-free** (exact, bounded context `T`). The table
writes the one-hot position `e_t` into `T` extra channels. The query reads them unchanged
(`e_i`), the key reads them through the matrix `Bᵀ` (`(Bᵀ e_j)_s = B s j`), both linear. The
extra coordinates contribute `e_i · Bᵀ e_j = B i j`: bidirectional ALiBi, T5 buckets, anything. -/
theorem bias_sub_learned_abs {κ : Type*} [Fintype κ] {T : ℕ} (S : Finset (Fin T))
    (B : Fin T → Fin T → ℝ) (i : Fin T) (q : κ → ℝ) (k : Fin T → κ → ℝ) (v : Fin T → V) :
    attn S (fun j => Sum.elim q (onehot i) ⬝ᵥ
        Sum.elim (k j) (fun s => ∑ t, B s t * onehot j t)) v =
      biasAttn S (fun j => q ⬝ᵥ k j) B i v := by
  have h : ∀ j, Sum.elim q (onehot i) ⬝ᵥ Sum.elim (k j) (fun s => ∑ t, B s t * onehot j t) =
      q ⬝ᵥ k j + B i j := by
    intro j
    simp only [dotProduct, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr, onehot]
    congr 1
    simp [Finset.sum_ite_eq']
  simp only [biasAttn, h]

end EP
