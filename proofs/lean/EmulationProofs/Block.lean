import EmulationProofs.Common

/-!
# Block structure and model-level rows: seq ⊆ parallel, parallel ⊆ seq (norm-free), looped ⊆
# untied, MTP ⊆ plain, N:M ↔ dense

Sublayers are ARBITRARY maps of the residual (each with its own pre-norm inside, if any).
-/

open Finset Matrix

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {R : Type*} [AddCommGroup R]

/-- A sequential block: `h = x + A x`, `y = h + M h`. -/
def seqBlock (A M : R → R) (x : R) : R := x + A x + M (x + A x)

/-- A parallel block: `y = x + A x + M x`. -/
def parBlock (A M : R → R) (x : R) : R := x + A x + M x

/-- A sublayer whose output projection is zeroed writes nothing. -/
theorem zeroed_sublayer {S : Type*} [AddCommGroup S] [Module ℝ S] [Module ℝ R]
    (inner : R → S) : (fun x => (0 : S →ₗ[ℝ] R) (inner x)) = fun _ => 0 := by
  funext x; simp

/-- **seq ⊆ parallel at 2× depth** (exact). An attention-only parallel block (MLP zeroed) and
then an MLP-only one (attention zeroed) reproduce one sequential block, for every input and
whatever the sublayers (pre-norms included). -/
theorem seq_sub_parallel (A M : R → R) (x : R) :
    parBlock (fun _ => 0) M (parBlock A (fun _ => 0) x) = seqBlock A M x := by
  simp [parBlock, seqBlock]

/-- The whole stack: `L` sequential blocks are `2L` parallel blocks. -/
theorem seq_stack_sub_parallel (blocks : List ((R → R) × (R → R))) (x : R) :
    (blocks.map (fun AM => fun y => parBlock (fun _ => 0) AM.2 (parBlock AM.1 (fun _ => 0) y))).foldr
        (· ∘ ·) id x =
      (blocks.map (fun AM => seqBlock AM.1 AM.2)).foldr (· ∘ ·) id x := by
  induction blocks generalizing x with
  | nil => rfl
  | cons AM rest ih =>
    simp only [List.map_cons, List.foldr_cons, Function.comp_apply, seq_sub_parallel]

/-- **parallel ⊆ seq, norm-free, at `(L+1)×` width** (exact). The host residual has `L + 1`
channels of width `d` (channel `0` the main stream, channel `l + 1` reserved for block `l`'s
attention write); the target residual is their sum. In block `l` the attention reads the sum and
writes channel `l + 1`; the MLP reads every channel except `l + 1` — i.e. the block's INPUT —
and writes channel `0`. Every read is linear, which is exactly what a shared pre-norm breaks. -/
theorem parallel_sub_seq_wide (L : ℕ) (A M : ℕ → R → R) (x : R) :
    let target : ℕ → R := fun l => Nat.rec x (fun l y => parBlock (A l) (M l) y) l
    let A' : ℕ → (Fin (L + 1) → R) → (Fin (L + 1) → R) := fun l s i =>
      if (i : ℕ) = l + 1 then A l (∑ i, s i) else 0
    let M' : ℕ → (Fin (L + 1) → R) → (Fin (L + 1) → R) := fun l s i =>
      if (i : ℕ) = 0 then M l (∑ i : Fin (L + 1), if (i : ℕ) = l + 1 then 0 else s i) else 0
    let host : ℕ → Fin (L + 1) → R := fun l =>
      Nat.rec (fun i => if (i : ℕ) = 0 then x else 0) (fun l s => seqBlock (A' l) (M' l) s) l
    ∀ l ≤ L, ∑ i, host l i = target l ∧ ∀ i : Fin (L + 1), l < i → host l i = 0 := by
  intro target A' M' host l hl
  -- a vector supported on one channel sums to that channel
  have single_sum : ∀ (k : Fin (L + 1)) (a : R),
      (∑ i : Fin (L + 1), if (i : ℕ) = k then a else 0) = a := by
    intro k a
    rw [Finset.sum_eq_single k (fun b _ hb => if_neg (fun h => hb (Fin.ext h))) (by simp)]
    simp
  induction l with
  | zero =>
    refine ⟨?_, fun i hi => ?_⟩
    · exact single_sum 0 x
    · show (if (i : ℕ) = 0 then x else 0) = 0
      rw [if_neg (by omega)]
  | succ l ih =>
    obtain ⟨hsum, hzero⟩ := ih (by omega)
    have hl' : l + 1 < L + 1 := by omega
    set s := host l
    set j : Fin (L + 1) := ⟨l + 1, hl'⟩
    have hsj : s j = 0 := hzero j (by simp [j])
    have hA : ∑ i, A' l s i = A l (target l) := by
      rw [← hsum]; exact single_sum j _
    -- the MLP's read: every channel but `j` of `s + A' l s`, i.e. the block input
    have hread : (∑ i : Fin (L + 1), if (i : ℕ) = l + 1 then 0 else (s + A' l s) i) = target l := by
      rw [← hsum]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      by_cases hi : (i : ℕ) = l + 1
      · rw [if_pos hi]
        have : i = j := Fin.ext hi
        rw [this, hsj]
      · simp [hi, A']
    have hstep : host (l + 1) = seqBlock (A' l) (M' l) s := rfl
    refine ⟨?_, fun i hi => ?_⟩
    · rw [hstep, seqBlock]
      simp only [Pi.add_apply, Finset.sum_add_distrib]
      rw [hsum, hA]
      have hM : ∑ i, M' l (s + A' l s) i = M l (target l) := by
        simp only [M']; rw [hread]; exact single_sum 0 _
      rw [hM]
      rfl
    · rw [hstep, seqBlock]
      have h1 : (i : ℕ) ≠ l + 1 := by omega
      have h0 : (i : ℕ) ≠ 0 := by omega
      simp only [Pi.add_apply, A', M', h1, h0, if_false, add_zero]
      exact hzero i (by omega)

/-! ## looped ⊆ untied -/

/-- An untied stack of `n` layers, layer `t` with parameters `ψ t`. -/
def untied {P : Type*} (f : P → R → R) (ψ : ℕ → P) : ℕ → R → R
  | 0 => id
  | n + 1 => fun x => f (ψ n) (untied f ψ n x)

theorem untied_add {P : Type*} (f : P → R → R) (ψ : ℕ → P) (m n : ℕ) (x : R) :
    untied f ψ (m + n) x = untied f (fun t => ψ (m + t)) n (untied f ψ m x) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show f (ψ (m + n)) (untied f ψ (m + n) x) =
      f (ψ (m + n)) (untied f (fun t => ψ (m + t)) n (untied f ψ m x))
    rw [ih]

theorem untied_congr {P : Type*} (f : P → R → R) (ψ φ : ℕ → P) (n : ℕ)
    (h : ∀ t < n, ψ t = φ t) (x : R) : untied f ψ n x = untied f φ n x := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [untied]
    rw [h n (by omega), ih (fun t ht => h t (by omega))]

/-- **looped ⊆ untied** (exact, depth). `K` loops of a `b`-layer block with parameters `θ` are an
untied stack of `K b` layers with `θ_{t+b} = θ_t`. -/
theorem looped_sub_untied {P : Type*} (f : P → R → R) (θ : ℕ → P) (b K : ℕ) (x : R) :
    untied f (fun t => θ (t % b)) (K * b) x = (untied f θ b)^[K] x := by
  induction K generalizing x with
  | zero => simp [untied]
  | succ K ih =>
    rw [Function.iterate_succ_apply', add_mul, one_mul, untied_add, ih]
    apply untied_congr
    intro t ht
    simp [Nat.add_mod, Nat.mul_mod_left, Nat.mod_eq_of_lt ht]

/-! ## MTP ⊆ plain -/

/-- An MTP model: a trunk, the main head, and auxiliary predict modules reading the trunk's
hidden states (and the inputs); they never write back into the trunk. -/
def mtpForward {Xs H O A : Type*} (trunk : Xs → H) (head : H → O) (aux : List (H → Xs → A))
    (x : Xs) : O × List A :=
  (head (trunk x), aux.map (fun m => m (trunk x) x))

/-- **MTP ⊆ plain** (exact, free): dropping the auxiliary heads leaves the main next-token
logits unchanged — whatever the auxiliary modules are (so also `plain → MTP` by appending
zero-init heads). -/
theorem mtp_sub_plain {Xs H O A : Type*} (trunk : Xs → H) (head : H → O)
    (aux : List (H → Xs → A)) (x : Xs) : (mtpForward trunk head aux x).1 = head (trunk x) := rfl

/-! ## N:M ↔ dense -/

/-- A matrix whose input columns are grouped as `(group, slot)` is `N:M` sparse if every row
has at most `N` nonzeros in each group of `M` slots. -/
def IsNM {m G M : ℕ} (N : ℕ) (S : Matrix (Fin m) (Fin G × Fin M) ℝ) : Prop :=
  ∀ r g, (univ.filter (fun p : Fin M => S r (g, p) ≠ 0)).card ≤ N

/-- **dense ⊆ N:M at `M/N×` width** (exact). The dense `N G`-input matrix `W` (columns grouped
`(group, slot < N)`) is placed on the first `N` slots of each group of `M`, and the input is
spread the same way (zeros in the other slots). The result is `N:M` sparse and computes `W x`
exactly; for `2:4`, each dense pair goes into 2 of 4 slots. -/
theorem dense_sub_nm {m G N M : ℕ} (hNM : N ≤ M) (W : Matrix (Fin m) (Fin G × Fin N) ℝ)
    (x : Fin G × Fin N → ℝ) :
    let S : Matrix (Fin m) (Fin G × Fin M) ℝ := fun r gp =>
      if h : (gp.2 : ℕ) < N then W r (gp.1, ⟨gp.2, h⟩) else 0
    let x' : Fin G × Fin M → ℝ := fun gp => if h : (gp.2 : ℕ) < N then x (gp.1, ⟨gp.2, h⟩) else 0
    IsNM N S ∧ S *ᵥ x' = W *ᵥ x := by
  intro S x'
  constructor
  · intro r g
    calc (univ.filter (fun p : Fin M => S r (g, p) ≠ 0)).card
        ≤ (univ.map (Fin.castLEEmb hNM)).card := by
          apply Finset.card_le_card
          intro p hp
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, S] at hp
          have h : (p : ℕ) < N := by by_contra h; simp [h] at hp
          exact Finset.mem_map.mpr ⟨⟨p, h⟩, Finset.mem_univ _, Fin.ext (by simp)⟩
      _ = N := by simp
  · funext r
    simp only [Matrix.mulVec, dotProduct, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl (fun g _ => ?_)
    symm
    refine Fintype.sum_of_injective (Fin.castLE hNM) (Fin.castLE_injective hNM) _ _
      (fun p hp => ?_) (fun p => ?_)
    · have h : ¬ (p : ℕ) < N := by
        intro h; exact hp ⟨⟨p, h⟩, Fin.ext (by simp)⟩
      simp [S, x', h]
    · simp [S, x']

/-- **N:M ⊆ dense** (exact, equal size): an N:M-sparse matrix is a dense matrix. -/
theorem nm_sub_dense {m G M N : ℕ} (S : Matrix (Fin m) (Fin G × Fin M) ℝ) (_ : IsNM N S)
    (x : Fin G × Fin M → ℝ) : ∃ W : Matrix (Fin m) (Fin G × Fin M) ℝ, W *ᵥ x = S *ᵥ x :=
  ⟨S, rfl⟩

end EP
