import EmulationProofs.Approx
import EmulationProofs.Position
import EmulationProofs.Attention

/-!
# Learned-absolute positions on a relative host

* **The construction** (`learned_abs_two_block`, exact). Over a bounded context of `T + 1`
  positions, with a BOS token at position `0` (and only there), a host with NO positional
  encoding (any relative scheme set to the identity, e.g. NoPE) reproduces a learned-absolute
  embedding `e(x_t) + P_t` exactly with two pre-norm (RMSNorm) blocks and five extra residual
  channels `s, b, κ, u, v`:
  - the embedding writes `[e x; s(x), b(x), 1, 0, 0]`, with `b` the BOS flag and `s(x)` a
    per-token pad making every embedding's squared norm the same constant `C`;
  - block 1's attention (causal, zero logits, rank-1 value `ρ (κ - b)`) writes `t/(t+1)` into
    `u` (`causal_uniform_head_position` with a pre-norm);
  - block 1's MLP (bias-free ReLU units; the constant channel `κ` plays the bias) reads `u` and
    writes `v = √(1 - u²)` into `v`, so the residual's squared norm is `C + 1` at every position;
  - block 2's attention writes nothing; block 2's MLP, reading at constant RMS, writes the
    table `P_t` (a ReLU interpolant through the knots `i/(i+1)`, exact at the knots) and erases
    the five extra channels.
  After the two blocks the host residual is `[e(x_t) + P_t; 0]` at every position, for every
  admissible sequence, every `eps ≥ 0`.
* **Downstream** (`padded_prenorm_sublayer`, `padded_prenorm_stack`): the rest of the target
  (any pre-norm sublayers, mixing tokens or not) runs on the zero-padded residual exactly, with
  host `eps' = eps d/(d+5)` and weight `√(d/(d+5)) w` (`qknorm_pad`).
* **Without a marker it is impossible** (`no_learned_abs_without_marker`): plain softmax
  attention with ANY logits (relative, absolute, masked), per-position maps, residual sums and
  compositions send constant sequences to constant sequences, so no stack of them, of any depth,
  equals `x_t ↦ e(x_t) + P_t` once `P_0 ≠ P_1`.
* **A sink also breaks the symmetry** (`sink_uniform_position`): zero logits with a sink logit
  `0` give `(t+1)/(t+2)` times a constant value, with no BOS marker.
-/

open Real Finset Matrix

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## Host vectors: `d` content channels and five extra channels `s, b, κ, u, v` -/

/-- A host residual vector `[z; w]`: content `z`, extra channels `w = (s, b, κ, u, v)`. -/
def hostVec {d : ℕ} (z : Fin d → ℝ) (w : Fin 5 → ℝ) : Fin (d + 5) → ℝ := Fin.append z w

section hostVecLemmas

variable {d : ℕ}

@[simp] lemma hostVec_left (z : Fin d → ℝ) (w : Fin 5 → ℝ) (i : Fin d) :
    hostVec z w (Fin.castAdd 5 i) = z i := by simp [hostVec]

@[simp] lemma hostVec_right (z : Fin d → ℝ) (w : Fin 5 → ℝ) (i : Fin 5) :
    hostVec z w (Fin.natAdd d i) = w i := by simp [hostVec]

lemma hostVec_ext {y y' : Fin (d + 5) → ℝ} (h1 : ∀ i, y (Fin.castAdd 5 i) = y' (Fin.castAdd 5 i))
    (h2 : ∀ i, y (Fin.natAdd d i) = y' (Fin.natAdd d i)) : y = y' := by
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · exact h1 i
  · exact h2 i

lemma hostVec_add (z z' : Fin d → ℝ) (w w' : Fin 5 → ℝ) : hostVec z w + hostVec z' w' = hostVec (z + z') (w + w') :=
  hostVec_ext (by simp) (by simp)

lemma hostVec_smul (c : ℝ) (z : Fin d → ℝ) (w : Fin 5 → ℝ) : c • hostVec z w = hostVec (c • z) (c • w) :=
  hostVec_ext (by simp) (by simp)

lemma hostVec_sum {K : Type*} [Fintype K] (z : K → Fin d → ℝ) (w : K → Fin 5 → ℝ) :
    ∑ k, hostVec (z k) (w k) = hostVec (∑ k, z k) (∑ k, w k) :=
  hostVec_ext (fun i => by simp [Finset.sum_apply]) (fun i => by simp [Finset.sum_apply])

lemma dot_hostVec (a z : Fin d → ℝ) (b w : Fin 5 → ℝ) : hostVec a b ⬝ᵥ hostVec z w = a ⬝ᵥ z + b ⬝ᵥ w := by
  simp [dotProduct, Fin.sum_univ_add]

lemma sumsq_hostVec (z : Fin d → ℝ) (w : Fin 5 → ℝ) :
    ∑ i, hostVec z w i ^ 2 = ∑ i, z i ^ 2 + ∑ i, w i ^ 2 := by
  simp [Fin.sum_univ_add]

lemma padVec_eq_hostVec (z : Fin d → ℝ) : padVec (e := 5) z = hostVec z 0 :=
  hostVec_ext (by simp) (by simp)

end hostVecLemmas

/-- RMSNorm with unit weight is division by the RMS. -/
lemma rmsNorm_one {m : ℕ} (eps : ℝ) (y : Fin m → ℝ) :
    rmsNorm eps (fun _ => 1) y = (rms eps y)⁻¹ • y := by
  funext i
  simp only [rmsNorm, rms, Pi.smul_apply, smul_eq_mul]
  ring

/-! ## Bias-free ReLU MLPs -/

/-- A one-hidden-layer ReLU MLP without biases: unit `k` reads `A k ⬝ᵥ y` and writes the
column `W k`. (A bias is a weight on a constant channel.) -/
def reluMLP {m : ℕ} {K : Type*} [Fintype K] (A W : K → Fin m → ℝ) (y : Fin m → ℝ) :
    Fin m → ℝ :=
  ∑ k, relu (A k ⬝ᵥ y) • W k

lemma relu_mul_of_nonneg {c : ℝ} (hc : 0 ≤ c) (a : ℝ) : relu (c * a) = c * relu a := by
  simp only [relu]; rw [mul_max_of_nonneg _ _ hc, mul_zero]

lemma relu_sub_relu_neg (a : ℝ) : relu a - relu (-a) = a := by
  simp only [relu]
  rcases le_total a 0 with h | h
  · rw [max_eq_right h, max_eq_left (by linarith)]; ring
  · rw [max_eq_left h, max_eq_right (by linarith)]; ring

/-- **Positive homogeneity**: a bias-free ReLU MLP commutes with a positive rescaling of its
input, so a pre-norm's `1/rms` passes through it. -/
lemma reluMLP_smul {m : ℕ} {K : Type*} [Fintype K] (A W : K → Fin m → ℝ) {c : ℝ} (hc : 0 ≤ c)
    (y : Fin m → ℝ) : reluMLP A W (c • y) = c • reluMLP A W y := by
  simp only [reluMLP, dotProduct_smul, smul_eq_mul, relu_mul_of_nonneg hc, Finset.smul_sum,
    smul_smul]

lemma reluMLP_sum_elim {m : ℕ} {K J : Type*} [Fintype K] [Fintype J] (A W : K → Fin m → ℝ)
    (A' W' : J → Fin m → ℝ) (y : Fin m → ℝ) :
    reluMLP (Sum.elim A A') (Sum.elim W W') y = reluMLP A W y + reluMLP A' W' y := by
  simp [reluMLP, Fintype.sum_sum_type]

lemma reluMLP_prod {m : ℕ} {I K : Type*} [Fintype I] [Fintype K] (A W : I → K → Fin m → ℝ)
    (y : Fin m → ℝ) :
    reluMLP (fun p : I × K => A p.1 p.2) (fun p => W p.1 p.2) y = ∑ i, reluMLP (A i) (W i) y := by
  simp [reluMLP, Fintype.sum_prod_type]

/-! ## A piecewise-linear readout of channel `u`, biased through channel `κ` -/

/-- The knots `i/(i+1)`: the values a causal uniform head produces. -/
def posKnot (i : ℕ) : ℝ := i / (i + 1)

lemma posKnot_strictMono : StrictMono posKnot := by
  intro i j hij
  have : (i : ℝ) < j := by exact_mod_cast hij
  simp only [posKnot]
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  nlinarith

lemma posKnot_nonneg (i : ℕ) : 0 ≤ posKnot i := by unfold posKnot; positivity

lemma posKnot_lt_one (i : ℕ) : posKnot i < 1 := by
  unfold posKnot; rw [div_lt_one (by positivity)]; linarith

/-- Slopes of the interpolant (as in `reluNet`). -/
def pwlSlope (t y : ℕ → ℝ) (i : ℕ) : ℝ := (y (i + 1) - y i) / (t (i + 1) - t i)

/-- The affine part `y 0 + s 0 (u - t 0)`, as a functional of `(s, b, κ, u, v)` with `κ = 1`. -/
def pwlLin (t y : ℕ → ℝ) : Fin 5 → ℝ := ![0, 0, y 0 - pwlSlope t y 0 * t 0, pwlSlope t y 0, 0]

/-- The units of `reluNet t y n` reading channel `u` (index 3) with `κ` (index 2) as bias:
two units `±` for the affine part, and `ReLU(u - t (i+1) κ)` for each inner knot. -/
def pwlA {d : ℕ} (t y : ℕ → ℝ) (n : ℕ) : Fin 2 ⊕ Fin n → Fin (d + 5) → ℝ :=
  Sum.elim ![hostVec 0 (pwlLin t y), hostVec 0 (-pwlLin t y)]
    (fun i => hostVec 0 ![0, 0, -t ((i : ℕ) + 1), 1, 0])

/-- Their output columns, all along the direction `D`. -/
def pwlW {d : ℕ} (t y : ℕ → ℝ) (n : ℕ) (D : Fin (d + 5) → ℝ) : Fin 2 ⊕ Fin n → Fin (d + 5) → ℝ :=
  Sum.elim ![D, -D] (fun i => (pwlSlope t y ((i : ℕ) + 1) - pwlSlope t y i) • D)

/-- The piecewise-linear readout computes `reluNet t y n (u) • D` whenever `κ = 1`. -/
lemma pwl_apply {d : ℕ} (t y : ℕ → ℝ) (n : ℕ) (D : Fin (d + 5) → ℝ) (z : Fin d → ℝ)
    (w : Fin 5 → ℝ) (hκ : w 2 = 1) :
    reluMLP (pwlA t y n) (pwlW t y n D) (hostVec z w) = reluNet t y n (w 3) • D := by
  have hlin : pwlLin t y ⬝ᵥ w = y 0 + pwlSlope t y 0 * (w 3 - t 0) := by
    simp [pwlLin, dotProduct, Fin.sum_univ_five, hκ]; ring
  have hunit : ∀ i : Fin n,
      (![0, 0, -t ((i : ℕ) + 1), 1, 0] : Fin 5 → ℝ) ⬝ᵥ w = w 3 - t ((i : ℕ) + 1) := by
    intro i; simp [dotProduct, Fin.sum_univ_five, hκ]; ring
  simp only [reluMLP, pwlA, pwlW, Fintype.sum_sum_type, Fin.sum_univ_two, Sum.elim_inl,
    Sum.elim_inr, Matrix.cons_val_zero, Matrix.cons_val_one, dot_hostVec,
    zero_dotProduct, zero_add, neg_dotProduct, hlin, hunit, smul_smul, smul_neg]
  rw [← Finset.sum_smul, ← neg_smul, ← add_smul, ← add_smul]
  congr 1
  have h := relu_sub_relu_neg (y 0 + pwlSlope t y 0 * (w 3 - t 0))
  simp only [reluNet]
  rw [Fin.sum_univ_eq_sum_range (fun i => relu (w 3 - t (i + 1)) *
    (pwlSlope t y (i + 1) - pwlSlope t y i)) n]
  simp only [pwlSlope] at h ⊢
  rw [show ∀ a b c : ℝ, a + -b + c = (a - b) + c from fun a b c => by ring, h]
  congr 1
  refine Finset.sum_congr rfl (fun i _ => by ring)

/-- At a knot the interpolant is the table value. -/
lemma reluNet_posKnot (y : ℕ → ℝ) (n k : ℕ) (hk : k ≤ n) :
    reluNet posKnot y n (posKnot k) = y k := by
  rw [reluNet_on_piece posKnot y posKnot_strictMono n k hk (posKnot k)
    ⟨le_rfl, (posKnot_strictMono (Nat.lt_succ_self k)).le⟩]
  ring

/-! ## The two-block construction -/

section construction

variable {d T : ℕ} {Tok : Type*} [Fintype Tok] [DecidableEq Tok]

/-- The squared norm every host embedding is padded to: `C = ∑_x |e x|² + 2`. -/
def padC (e : Tok → Fin d → ℝ) : ℝ := ∑ x, ∑ i, e x i ^ 2 + 2

/-- The BOS flag. -/
def bosFlag (bos x : Tok) : ℝ := if x = bos then 1 else 0

/-- The per-token pad `s(x) = √(C - |e x|² - b(x)² - 1)`. -/
def padS (e : Tok → Fin d → ℝ) (bos x : Tok) : ℝ :=
  √(padC e - ∑ i, e x i ^ 2 - bosFlag bos x ^ 2 - 1)

/-- The host embedding `[e x; s(x), b(x), 1, 0, 0]` (a token-embedding table, width `d + 5`). -/
def hostEmb (e : Tok → Fin d → ℝ) (bos x : Tok) : Fin (d + 5) → ℝ :=
  hostVec (e x) ![padS e bos x, bosFlag bos x, 1, 0, 0]

lemma bosFlag_nonneg (bos x : Tok) : 0 ≤ bosFlag bos x := by
  unfold bosFlag; split_ifs <;> norm_num

lemma bosFlag_sq_le (bos x : Tok) : bosFlag bos x ^ 2 ≤ 1 := by
  unfold bosFlag; split_ifs <;> norm_num

lemma padS_sq (e : Tok → Fin d → ℝ) (bos x : Tok) :
    padS e bos x ^ 2 = padC e - ∑ i, e x i ^ 2 - bosFlag bos x ^ 2 - 1 := by
  unfold padS
  rw [sq_sqrt]
  have h1 : ∑ i, e x i ^ 2 ≤ ∑ x', ∑ i, e x' i ^ 2 :=
    Finset.single_le_sum (f := fun x' => ∑ i, e x' i ^ 2)
      (fun x' _ => Finset.sum_nonneg (fun i _ => sq_nonneg _)) (Finset.mem_univ x)
  have := bosFlag_sq_le bos x
  unfold padC; linarith

lemma padC_pos (e : Tok → Fin d → ℝ) : 0 < padC e := by
  unfold padC
  have : 0 ≤ ∑ x, ∑ i, e x i ^ 2 :=
    Finset.sum_nonneg (fun x _ => Finset.sum_nonneg (fun i _ => sq_nonneg _))
  linarith

/-- Every host embedding has squared norm `C`. -/
lemma hostEmb_sumsq (e : Tok → Fin d → ℝ) (bos x : Tok) :
    ∑ i, hostEmb e bos x i ^ 2 = padC e := by
  rw [hostEmb, sumsq_hostVec]
  simp only [Fin.sum_univ_five, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.cons_val_three, Matrix.cons_val_four, Matrix.tail_cons]
  rw [padS_sq]; ring

/-- The unit vector of extra channel `j`. -/
def eCh {d : ℕ} (j : Fin 5) : Fin (d + 5) → ℝ := hostVec 0 (Pi.single j 1)

/-- Block 1's RMS (`eps` included): at the embedding, after the attention, after the MLP. -/
def rms0 (e : Tok → Fin d → ℝ) (eps : ℝ) : ℝ := √(padC e / (d + 5 : ℕ) + eps)
def rms1 (e : Tok → Fin d → ℝ) (eps : ℝ) (i : ℕ) : ℝ :=
  √((padC e + posKnot i ^ 2) / (d + 5 : ℕ) + eps)
def rms2 (e : Tok → Fin d → ℝ) (eps : ℝ) : ℝ := √((padC e + 1) / (d + 5 : ℕ) + eps)

/-- Block 1's value/output projection (rank 1): read `ρ (κ - b)`, write channel `u`. -/
def val1 (e : Tok → Fin d → ℝ) (eps : ℝ) (y : Fin (d + 5) → ℝ) : Fin (d + 5) → ℝ :=
  (hostVec (0 : Fin d → ℝ) ![0, -rms0 e eps, rms0 e eps, 0, 0] ⬝ᵥ y) • eCh 3

/-- Block 1's MLP table: `rms1 i · √(1 - (i/(i+1))²)`, written to channel `v`. -/
def vals1 (e : Tok → Fin d → ℝ) (eps : ℝ) (i : ℕ) : ℝ :=
  rms1 e eps i * √(1 - posKnot i ^ 2)

/-- Block 2's MLP table for content coordinate `c`: `rms2 · P_i c`. -/
def vals2 (e : Tok → Fin d → ℝ) (eps : ℝ) (P : Fin (T + 1) → Fin d → ℝ) (c : Fin d) (i : ℕ) : ℝ :=
  if h : i < T + 1 then rms2 e eps * P ⟨i, h⟩ c else 0

/-- Block 2's MLP units: one eraser per extra channel, then the table interpolant for each
content coordinate. -/
def mlp2A (e : Tok → Fin d → ℝ) (eps : ℝ) (P : Fin (T + 1) → Fin d → ℝ) :
    Fin 5 ⊕ (Fin d × (Fin 2 ⊕ Fin (T + 1))) → Fin (d + 5) → ℝ :=
  Sum.elim (fun j => eCh j) (fun p => pwlA posKnot (vals2 e eps P p.1) (T + 1) p.2)

def mlp2W (e : Tok → Fin d → ℝ) (eps : ℝ) (P : Fin (T + 1) → Fin d → ℝ) :
    Fin 5 ⊕ (Fin d × (Fin 2 ⊕ Fin (T + 1))) → Fin (d + 5) → ℝ :=
  Sum.elim (fun j => (-rms2 e eps) • eCh j)
    (fun p => pwlW posKnot (vals2 e eps P p.1) (T + 1) (hostVec (Pi.single p.1 1) 0) p.2)

/-- The host's pre-norm: RMSNorm, unit weight (any weight folds into the next linear maps). -/
def hostNorm {m : ℕ} (eps : ℝ) (y : Fin m → ℝ) : Fin m → ℝ := rmsNorm eps (fun _ => 1) y

/-- Host residual after the embedding. -/
def hostR0 (e : Tok → Fin d → ℝ) (bos : Tok) (X : Fin (T + 1) → Tok) (t : Fin (T + 1)) :
    Fin (d + 5) → ℝ :=
  hostEmb e bos (X t)

/-- After block 1's attention: causal, all logits `0`, value `val1` of the normed input. -/
def hostR1 (e : Tok → Fin d → ℝ) (eps : ℝ) (bos : Tok) (X : Fin (T + 1) → Tok)
    (t : Fin (T + 1)) : Fin (d + 5) → ℝ :=
  hostR0 e bos X t + attn (univ.filter (· ≤ t)) (fun _ => (0 : ℝ))
    (fun s => val1 e eps (hostNorm eps (hostR0 e bos X s)))

/-- After block 1's MLP (`vals1` interpolant on channel `u`, written to channel `v`). -/
def hostR2 (e : Tok → Fin d → ℝ) (eps : ℝ) (bos : Tok) (X : Fin (T + 1) → Tok)
    (t : Fin (T + 1)) : Fin (d + 5) → ℝ :=
  hostR1 e eps bos X t + reluMLP (pwlA posKnot (vals1 e eps) (T + 1))
    (pwlW posKnot (vals1 e eps) (T + 1) (eCh 4)) (hostNorm eps (hostR1 e eps bos X t))

/-- After block 2 (attention writes nothing; the MLP writes the table and erases the extra
channels). -/
def hostR3 (e : Tok → Fin d → ℝ) (eps : ℝ) (bos : Tok) (P : Fin (T + 1) → Fin d → ℝ)
    (X : Fin (T + 1) → Tok) (t : Fin (T + 1)) : Fin (d + 5) → ℝ :=
  hostR2 e eps bos X t + reluMLP (mlp2A e eps P) (mlp2W e eps P) (hostNorm eps (hostR2 e eps bos X t))

/-- The squared norm of a host residual of the form `[e x; s(x), b(x), 1, a, c]`. -/
lemma stream_sumsq (e : Tok → Fin d → ℝ) (bos x : Tok) (a c : ℝ) :
    ∑ i, hostVec (e x) ![padS e bos x, bosFlag bos x, 1, a, c] i ^ 2 = padC e + a ^ 2 + c ^ 2 := by
  rw [sumsq_hostVec]
  simp only [Fin.sum_univ_five, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.cons_val_three, Matrix.cons_val_four, Matrix.tail_cons]
  rw [padS_sq]; ring

lemma rms_stream (e : Tok → Fin d → ℝ) (eps : ℝ) (bos x : Tok) (a c : ℝ) :
    rms eps (hostVec (e x) ![padS e bos x, bosFlag bos x, 1, a, c]) =
      √((padC e + a ^ 2 + c ^ 2) / (d + 5 : ℕ) + eps) := by
  rw [rms, stream_sumsq]

lemma rms_arg_pos (e : Tok → Fin d → ℝ) {eps : ℝ} (heps : 0 ≤ eps) (a : ℝ) (ha : 0 ≤ a) :
    0 < √((padC e + a) / (d + 5 : ℕ) + eps) := by
  have := padC_pos e
  apply Real.sqrt_pos.mpr
  have : (0 : ℝ) < (d + 5 : ℕ) := by positivity
  positivity

/-- `∑_j w_j • (c • eCh j) = c • [0; w]`. -/
lemma sum_smul_eCh {d : ℕ} (w : Fin 5 → ℝ) (c : ℝ) :
    ∑ j, w j • (c • eCh (d := d) j) = hostVec 0 (c • w) := by
  simp only [eCh, hostVec_smul, smul_zero, hostVec_sum, Finset.sum_const_zero]
  congr 1
  funext i
  simp [Finset.sum_apply, Pi.single_apply, mul_comm]

/-- `∑_c a_c • [e_c; 0] = [a; 0]`. -/
lemma sum_smul_content {d : ℕ} (a : Fin d → ℝ) :
    ∑ c, a c • hostVec (Pi.single c 1) (0 : Fin 5 → ℝ) = hostVec a 0 := by
  simp only [hostVec_smul, smul_zero, hostVec_sum, Finset.sum_const_zero]
  congr 1
  funext i
  simp [Finset.sum_apply, Pi.single_apply]

/-- **Learned-absolute positions from a BOS marker, in two pre-norm blocks** (exact). For every
token embedding `e`, every table `P` over `T + 1` positions, every RMSNorm `eps ≥ 0`, and every
sequence with the BOS token at position `0` and nowhere else, the host above (no positional
encoding at all; embedding `hostEmb`, block 1 = causal zero-logit attention + ReLU MLP, block 2
= ReLU MLP, every sublayer reading through RMSNorm) ends with residual `[e(x_t) + P_t; 0]` at
every position `t`: the learned-absolute target's residual, zero-padded. -/
theorem learned_abs_two_block (eps : ℝ) (heps : 0 ≤ eps) (e : Tok → Fin d → ℝ) (bos : Tok)
    (P : Fin (T + 1) → Fin d → ℝ) (X : Fin (T + 1) → Tok) (hX0 : X 0 = bos)
    (hX : ∀ j, j ≠ 0 → X j ≠ bos) (t : Fin (T + 1)) :
    hostR3 e eps bos P X t = padVec (e (X t) + P t) := by
  -- the embedding stage
  have hr0 : 0 < rms0 e eps := by simpa [rms0] using rms_arg_pos e heps 0 le_rfl
  have hN0 : ∀ x, hostNorm eps (hostEmb e bos x) = (rms0 e eps)⁻¹ • hostEmb e bos x := by
    intro x
    rw [hostNorm, rmsNorm_one, rms, hostEmb_sumsq, rms0]
  have hval : ∀ x, val1 e eps ((rms0 e eps)⁻¹ • hostEmb e bos x) = (1 - bosFlag bos x) • eCh 3 := by
    intro x
    rw [val1, dotProduct_smul, hostEmb, dot_hostVec, zero_dotProduct, zero_add]
    congr 1
    simp only [dotProduct, Fin.sum_univ_five, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.cons_val_three, Matrix.cons_val_four,
      Matrix.tail_cons, smul_eq_mul]
    field_simp
    ring
  have hb : ∀ s, bosFlag bos (X s) = if s = 0 then 1 else 0 := by
    intro s
    by_cases hs : s = 0
    · simp [bosFlag, hs, hX0]
    · simp [bosFlag, hs, hX s hs]
  -- block 1's attention writes `t/(t+1)` into `u`
  have hcard : (univ.filter (fun s : Fin (T + 1) => s ≤ t)).card = t + 1 := by
    have : univ.filter (fun s : Fin (T + 1) => s ≤ t) = Finset.Iic t := by ext; simp
    rw [this, Fin.card_Iic]
  have hattn : attn (univ.filter (· ≤ t)) (fun _ => (0 : ℝ))
      (fun s => val1 e eps (hostNorm eps (hostR0 e bos X s))) = posKnot t • eCh 3 := by
    simp only [hostR0, hN0, hval, attn, exp_zero, one_smul, Finset.sum_const, hcard,
      nsmul_eq_mul, mul_one]
    rw [← Finset.sum_smul, smul_smul, Finset.sum_sub_distrib, Finset.sum_const, hcard]
    simp only [hb]
    rw [Finset.sum_ite_eq' (univ.filter (· ≤ t)) 0]
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fin.zero_le, ite_true, nsmul_eq_mul,
      mul_one, posKnot]
    congr 1
    push_cast
    field_simp
    ring
  set s := padS e bos (X t)
  set b := bosFlag bos (X t)
  set k := posKnot t
  have h1 : hostR1 e eps bos X t = hostVec (e (X t)) ![s, b, 1, k, 0] := by
    rw [hostR1, hattn, hostR0, hostEmb, eCh, hostVec_smul, hostVec_add]
    congr 1
    · simp
    · funext j; fin_cases j <;> simp [s, b]
  -- block 1's MLP writes `√(1 - k²)` into `v`
  have hk0 : 0 ≤ k := posKnot_nonneg _
  have hk1 : k < 1 := posKnot_lt_one _
  set v := √(1 - k ^ 2)
  have hv2 : v ^ 2 = 1 - k ^ 2 := sq_sqrt (by nlinarith)
  have hr1 : 0 < rms1 e eps t := by simpa [rms1] using rms_arg_pos e heps (k ^ 2) (sq_nonneg _)
  have hmlp1 : reluMLP (pwlA posKnot (vals1 e eps) (T + 1))
      (pwlW posKnot (vals1 e eps) (T + 1) (eCh 4)) (hostNorm eps (hostR1 e eps bos X t)) = v • eCh 4 := by
    rw [hostNorm, rmsNorm_one, h1, rms_stream, reluMLP_smul _ _ (inv_nonneg.mpr (Real.sqrt_nonneg _)),
      pwl_apply _ _ _ _ _ _ (by simp)]
    simp only [Matrix.cons_val_three, Matrix.tail_cons, Matrix.head_cons]
    rw [reluNet_posKnot _ _ _ (by omega), smul_smul, vals1]
    congr 1
    have : rms1 e eps t = √((padC e + k ^ 2 + 0 ^ 2) / (d + 5 : ℕ) + eps) := by
      simp [rms1, k]
    rw [← this]
    field_simp
    rfl
  have h2 : hostR2 e eps bos X t = hostVec (e (X t)) ![s, b, 1, k, v] := by
    rw [hostR2, hmlp1, h1, eCh, hostVec_smul, hostVec_add]
    congr 1
    · simp
    · funext j; fin_cases j <;> simp
  -- block 2 reads at constant RMS, writes the table and erases the extra channels
  set w : Fin 5 → ℝ := ![s, b, 1, k, v]
  have hw : ∀ j, 0 ≤ w j := by
    intro j
    fin_cases j
    · exact Real.sqrt_nonneg _
    · exact bosFlag_nonneg _ _
    · simp [w]
    · exact hk0
    · exact Real.sqrt_nonneg _
  have hr2 : 0 < rms2 e eps := by simpa [rms2] using rms_arg_pos e heps 1 zero_le_one
  have hrms2 : rms eps (hostR2 e eps bos X t) = rms2 e eps := by
    rw [h2, rms_stream, rms2, hv2]; congr 3; ring
  have hmlp2 : reluMLP (mlp2A e eps P) (mlp2W e eps P) (hostVec (e (X t)) w) =
      hostVec (rms2 e eps • P t) ((-rms2 e eps) • w) := by
    rw [mlp2A, mlp2W, reluMLP_sum_elim]
    have hprod := reluMLP_prod (fun c q => pwlA posKnot (vals2 e eps P c) (T + 1) q)
      (fun c q => pwlW posKnot (vals2 e eps P c) (T + 1) (hostVec (Pi.single c 1) 0) q) (hostVec (e (X t)) w)
    rw [hprod]
    have hcont : ∀ c, reluMLP (pwlA posKnot (vals2 e eps P c) (T + 1))
        (pwlW posKnot (vals2 e eps P c) (T + 1) (hostVec (Pi.single c 1) 0)) (hostVec (e (X t)) w) =
        (rms2 e eps * P t c) • hostVec (Pi.single c 1) (0 : Fin 5 → ℝ) := by
      intro c
      rw [pwl_apply _ _ _ _ _ _ (by simp [w])]
      have : w 3 = posKnot t := by simp [w, k]
      rw [this, reluNet_posKnot _ _ _ (by omega), vals2]
      simp [t.isLt]
    simp only [hcont, sum_smul_content]
    have herase : reluMLP (fun j => eCh (d := d) j) (fun j => (-rms2 e eps) • eCh j)
        (hostVec (e (X t)) w) = hostVec 0 ((-rms2 e eps) • w) := by
      rw [reluMLP, ← sum_smul_eCh]
      refine Finset.sum_congr rfl (fun j _ => ?_)
      congr 1
      rw [eCh, dot_hostVec, zero_dotProduct, zero_add]
      simp only [relu]
      rw [show Pi.single j 1 ⬝ᵥ w = w j by simp, max_eq_left (hw j)]
    rw [herase, hostVec_add]
    simp
    rfl
  rw [hostR3, hostNorm, rmsNorm_one, hrms2, h2, reluMLP_smul _ _ (inv_nonneg.mpr hr2.le), hmlp2,
    hostVec_smul, hostVec_add, padVec_eq_hostVec]
  congr 1
  · funext c; simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]; field_simp
  · funext j; simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]; field_simp
    ring

end construction

/-! ## The rest of the target runs on the padded residual -/

section downstream

variable {d : ℕ} {ι : Type*}

/-- The content channels of a host vector. -/
def contentOf (y : Fin (d + 5) → ℝ) : Fin d → ℝ := fun i => y (Fin.castAdd 5 i)

@[simp] lemma contentOf_padVec (z : Fin d → ℝ) : contentOf (padVec (e := 5) z) = z := by
  funext i; simp [contentOf]

/-- A target pre-norm residual sublayer: `Z ↦ Z + F (RMSNorm_{eps,w} ∘ Z)`. `F` is arbitrary
and may mix positions (attention) or not (MLP). -/
def preLayer (eps : ℝ) (w : Fin d → ℝ) (F : (ι → Fin d → ℝ) → ι → Fin d → ℝ)
    (Z : ι → Fin d → ℝ) : ι → Fin d → ℝ :=
  fun t => Z t + F (fun s => rmsNorm eps w (Z s)) t

/-- Its host copy at width `d + 5`: RMSNorm over all `d + 5` channels with `eps d/(d+5)` and
weight `√(d/(d+5)) w` on the content channels (`0` on the extra ones), the same `F` on the
content channels, writing only content channels. -/
def padLayer (eps : ℝ) (w : Fin d → ℝ) (F : (ι → Fin d → ℝ) → ι → Fin d → ℝ)
    (Y : ι → Fin (d + 5) → ℝ) : ι → Fin (d + 5) → ℝ :=
  fun t => Y t + padVec (F (fun s => contentOf (rmsNorm (eps * d / (d + 5 : ℕ))
    (padVec (fun i => √((d : ℝ) / (d + 5 : ℕ)) * w i)) (Y s))) t)

/-- **A pre-norm sublayer acts on the padded residual exactly.** -/
theorem padded_prenorm_sublayer (hd : d ≠ 0) (eps : ℝ) (w : Fin d → ℝ)
    (F : (ι → Fin d → ℝ) → ι → Fin d → ℝ) (Z : ι → Fin d → ℝ) :
    padLayer eps w F (fun s => padVec (Z s)) = fun t => padVec (preLayer eps w F Z t) := by
  funext t
  simp only [padLayer, preLayer]
  simp only [qknorm_pad hd, contentOf_padVec]
  simp only [padVec_eq_hostVec, hostVec_add, add_zero]

/-- **…and so does a whole stack of them.** -/
theorem padded_prenorm_stack (hd : d ≠ 0)
    (layers : List (ℝ × (Fin d → ℝ) × ((ι → Fin d → ℝ) → ι → Fin d → ℝ))) (Z : ι → Fin d → ℝ) :
    (layers.map (fun L => padLayer L.1 L.2.1 L.2.2)).foldr (· ∘ ·) id (fun s => padVec (Z s)) =
      fun t => padVec ((layers.map (fun L => preLayer L.1 L.2.1 L.2.2)).foldr (· ∘ ·) id Z t) := by
  induction layers with
  | nil => rfl
  | cons L rest ih =>
    simp only [List.map_cons, List.foldr_cons, Function.comp_apply, ih]
    exact padded_prenorm_sublayer hd _ _ _ _

end downstream

/-- **Learned-absolute ⊆ relative host, end to end** (exact, bounded context, BOS marker).
The two-block prefix followed by the padded copies of the target's pre-norm sublayers reproduces
the learned-absolute target (embedding `e(x_t) + P_t`, then the same sublayers), zero-padded, at
every position of every sequence with BOS at position `0` only. -/
theorem learned_abs_model {d T : ℕ} {Tok : Type*} [Fintype Tok] [DecidableEq Tok] (hd : d ≠ 0)
    (eps : ℝ) (heps : 0 ≤ eps) (e : Tok → Fin d → ℝ) (bos : Tok) (P : Fin (T + 1) → Fin d → ℝ)
    (layers : List (ℝ × (Fin d → ℝ) × ((Fin (T + 1) → Fin d → ℝ) → Fin (T + 1) → Fin d → ℝ)))
    (X : Fin (T + 1) → Tok) (hX0 : X 0 = bos) (hX : ∀ j, j ≠ 0 → X j ≠ bos) :
    (layers.map (fun L => padLayer L.1 L.2.1 L.2.2)).foldr (· ∘ ·) id
        (fun t => hostR3 e eps bos P X t) =
      fun t => padVec ((layers.map (fun L => preLayer L.1 L.2.1 L.2.2)).foldr (· ∘ ·) id
        (fun s => e (X s) + P s) t) := by
  have : (fun t => hostR3 e eps bos P X t) = fun s => padVec (e (X s) + P s) :=
    funext (learned_abs_two_block eps heps e bos P X hX0 hX)
  rw [this, padded_prenorm_stack hd]

/-! ## Without a marker: constant sequences stay constant -/

/-- A sequence map sends constant sequences to constant sequences. -/
def PresConst {ι A B : Type*} (F : (ι → A) → (ι → B)) : Prop :=
  ∀ a : A, ∃ b : B, F (fun _ => a) = fun _ => b

/-- Attention over identical values returns that value, whatever the logits. -/
lemma attn_const_value {κ V : Type*} [Fintype κ] [NormedAddCommGroup V] [NormedSpace ℝ V]
    (S : Finset κ) (hS : S.Nonempty) (s : κ → ℝ) (c : V) : attn S s (fun _ => c) = c := by
  rw [attn, ← Finset.sum_smul, smul_smul, inv_mul_cancel₀ (sum_exp_pos hS s).ne', one_smul]

theorem PresConst.pointwise {ι A B : Type*} (g : A → B) : PresConst (fun (X : ι → A) t => g (X t)) :=
  fun a => ⟨g a, rfl⟩

theorem PresConst.comp {ι A B C : Type*} {F : (ι → A) → (ι → B)} {G : (ι → B) → (ι → C)}
    (hF : PresConst F) (hG : PresConst G) : PresConst (G ∘ F) := by
  intro a
  obtain ⟨b, hb⟩ := hF a
  obtain ⟨c, hc⟩ := hG b
  exact ⟨c, by simp [hb, hc]⟩

theorem PresConst.residual {ι A : Type*} [Add A] {G : (ι → A) → (ι → A)} (hG : PresConst G) :
    PresConst (fun X t => X t + G X t) := by
  intro a
  obtain ⟨b, hb⟩ := hG a
  exact ⟨a + b, by funext t; simp [hb]⟩

/-- **Softmax attention with ANY logits is blind to position on a constant sequence.** The key
set `S t` (causal, windowed, …) and the logits `logit X t s` may depend on the positions in any
way (RoPE, ALiBi, even absolute-position logit biases) and on the inputs; values and the output
map are per-token. -/
theorem PresConst.attention {κ A V : Type*} [Fintype κ] [NormedAddCommGroup V] [NormedSpace ℝ V]
    (S : κ → Finset κ) (hS : ∀ t, (S t).Nonempty) (logit : (κ → A) → κ → κ → ℝ) (val : A → V)
    (out : A → V → A) :
    PresConst (fun X t => out (X t) (attn (S t) (logit X t) (fun s => val (X s)))) := by
  intro a
  exact ⟨out a (val a), by funext t; simp [attn_const_value _ (hS t)]⟩

theorem PresConst.stack {ι A : Type*} (layers : List ((ι → A) → (ι → A)))
    (h : ∀ F ∈ layers, PresConst F) : PresConst (layers.foldr (· ∘ ·) id) := by
  induction layers with
  | nil => intro a; exact ⟨a, rfl⟩
  | cons F rest ih =>
    simp only [List.foldr_cons]
    exact PresConst.comp (ih (fun G hG => h G (List.mem_cons_of_mem _ hG))) (h F (by simp))

/-- **Without a marker, no depth helps** (impossibility). If the host (embedding, any number of
attention layers with any logits, MLPs, residual sums, readouts — anything `PresConst`) is
applied to a constant sequence, its output is constant; the learned-absolute embedding
`e(x_t) + P_t` is not, once `P 0 ≠ P 1`. So the BOS marker (or a sink,
`sink_uniform_position`) is necessary. -/
theorem no_learned_abs_without_marker {T : ℕ} {Tok A : Type*} [Nonempty Tok] [AddCommGroup A]
    (e : Tok → A) (P : Fin (T + 2) → A) (hP : P 0 ≠ P 1) (F : (Fin (T + 2) → Tok) → (Fin (T + 2) → A))
    (hF : PresConst F) : F ≠ fun X t => e (X t) + P t := by
  intro h
  obtain ⟨x⟩ := ‹Nonempty Tok›
  obtain ⟨b, hb⟩ := hF x
  have h0 := congrFun hb 0
  have h1 := congrFun hb 1
  rw [h] at h0 h1
  exact hP (add_left_cancel (h0.trans h1.symm))

/-! ## A sink breaks the symmetry too -/

/-- **A sink recovers position without a marker.** Zero logits on the `t + 1` visible keys,
sink logit `0`, the same value `c` everywhere (e.g. a constant channel): the output is
`(t+1)/(t+2) · c`, i.e. the knot `posKnot (t+1)`, distinct at every position. -/
theorem sink_uniform_position {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (t : ℕ)
    (c : V) :
    sinkAttn (fun _ : Fin (t + 1) => (0 : ℝ)) 0 (fun _ => c) = posKnot (t + 1) • c := by
  simp only [sinkAttn, exp_zero, one_smul, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_one, posKnot]
  rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul]
  congr 1
  push_cast
  field_simp
  ring

end EP
