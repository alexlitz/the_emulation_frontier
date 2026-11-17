import EmulationProofs.Position
import EmulationProofs.Norm
import EmulationProofs.Attention
import EmulationProofs.MoE
import EmulationProofs.Activation

/-!
# The conditions are needed: each construction fails when its condition is violated

`PROOFS.md` lists the conditions under which the cookbook constructions are exact. For each one
whose converse is not proved elsewhere, this file shows a violation that breaks it:

* top-1 select with a router-probability gate `p ≠ 1` (`hard_select_needs_unit_gate`);
* tied MoE without renormalisation, `∑ p ≠ 1` (`tied_moe_not_dense`);
* ALiBi's frozen pair cannot give bidirectional ALiBi (`alibi_host_not_bidirectional`), and a
  content pair at a nonzero frequency is not the plain dot product (`content_pair_rotates`);
* a sink / constant channel in a rotating pair depends on the query position
  (`sink_needs_zero_freq`);
* swigluoai outside the clamp regime (`swigluoai_gate_out_of_regime`,
  `swigluoai_up_out_of_regime`) and without an up bias (`swigluoai_needs_up_bias`);
* LayerNorm ⊆ RMSNorm with a different epsilon (`rmsNorm_eps_inj`, `layerNorm_ne_rms_other_eps`);
* qk-norm ⊆ plain outside its regime (`qknorm_out_of_regime`).

Elsewhere: the tie for the soft router (`soft_router_tie`), the learned-absolute wall without a
position-0 marker (`LearnedAbs.lean`), the pre-norm failures of parallel ⊆ seq and the CLA carry
(`PreNorm.lean`), and too few rotated dims for partial RoPE (`RopeRank.lean`).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## MoE gates -/

/-- **The hard select needs a unit gate.** The select layer outputs `Y_{e*}`; a top-1 MoE that
multiplies by its router probability `p ≠ 1` outputs `p • Y_{e*}`, which differs whenever
`Y_{e*} ≠ 0`. -/
theorem hard_select_needs_unit_gate {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {E : ℕ}
    (Y : Fin E → V) (estar : Fin E) (p : ℝ) (hp : p ≠ 1) (hY : Y estar ≠ 0) :
    ∑ e, (if e = estar then (1 : ℝ) else 0) • Y e ≠ p • Y estar := by
  rw [dense_hard_select]
  intro h
  have : (1 - p) • Y estar = 0 := by rw [sub_smul, one_smul, ← h, sub_self]
  rcases smul_eq_zero.mp this with h1 | h1
  · exact hp (by linarith [sub_eq_zero.mp h1])
  · exact hY h1

/-- **Tied MoE without renormalisation is not the dense FFN.** If on some token the gates do
not sum to `1` and the expert FFN is nonzero there, the tied MoE outputs `(∑ p) • F(x) ≠ F(x)`. -/
theorem tied_moe_not_dense {X Θ V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {E ew : ℕ}
    (act : Θ → X → ℝ) (gate : X → Fin E → ℝ) (θ : Fin ew → Θ) (W2 : Fin ew → V) (x : X)
    (hsum : ∑ e, gate x e ≠ 1) (hF : ffn act θ W2 x ≠ 0) :
    moe act gate (fun _ => θ) (fun _ => W2) x ≠ ffn act θ W2 x := by
  have e : moe act gate (fun _ => θ) (fun _ => W2) x = (∑ e, gate x e) • ffn act θ W2 x := by
    simp [moe, Finset.sum_smul]
  rw [e]
  intro h
  have : ((∑ e, gate x e) - 1) • ffn act θ W2 x = 0 := by rw [sub_smul, one_smul, h, sub_self]
  rcases smul_eq_zero.mp this with h1 | h1
  · exact hsum (sub_eq_zero.mp h1)
  · exact hF h1

/-! ## ALiBi and constant channels on a RoPE host -/

/-- **ALiBi via the frozen pair is causal only.** For every frequency `θ` (and so also in the
`θ → 0` limit, whose logit `-m (i - j)` is odd), the host logit cannot equal the bidirectional
ALiBi logit `q·k - m |i - j|` at both `(0, 1)` and `(1, 0)` unless `m = 0`. -/
theorem alibi_host_not_bidirectional {Pr : Type*} [Fintype Pr] (m θ : ℝ) (hm : m ≠ 0)
    (q k : Pr → ℝ × ℝ) :
    ¬ ∀ i j : ℕ, ropeScore (alibiHostFreq θ) (alibiHostQ m θ q) (alibiHostK k) i j =
      ∑ p, dot2 (q p) (k p) - m * |(i : ℝ) - j| := by
  intro h
  have h1 := h 0 1
  have h2 := h 1 0
  rw [alibi_host_score] at h1 h2
  norm_num at h1 h2
  have : m * 2 = 0 := by linarith
  exact hm (by linarith)

/-- The limit logit `-m (i - j)` is not the bidirectional `-m |i - j|` either. -/
theorem alibi_limit_not_bidirectional (m : ℝ) (hm : m ≠ 0) :
    ¬ ∀ i j : ℕ, alibiScore 0 m i j = -(m * |(i : ℝ) - j|) := by
  intro h
  have := h 0 1
  simp [alibiScore] at this
  exact hm (by linarith)

/-- **Content pairs must be at frequency 0.** A content pair at a frequency with `cos ω ≠ 1`
gives `q·k cos((i - j) ω)`, not `q·k`: with `q = k = (1, 0)` the logit at distance `1` is
`cos ω ≠ 1`. -/
theorem content_pair_rotates (ω : ℝ) (hω : cos ω ≠ 1) :
    ¬ ∀ i j : ℕ, j ≤ i → pairScore ω i j (1, 0) (1, 0) = dot2 (1, 0) (1, 0) := by
  intro h
  have := h 1 0 (by norm_num)
  rw [pairScore_eq] at this
  simp [dot2] at this
  exact hω this

/-- **The sink's constant channel must be unrotated.** In a pair with `cos ω ≠ 1`, a sink key
`(s, 0)`, `s ≠ 0`, at position `p` gets a different logit from queries at `p` and `p + 1`, so
the sink logit is not a constant. -/
theorem sink_needs_zero_freq (ω p s : ℝ) (hω : cos ω ≠ 1) (hs : s ≠ 0) :
    pairScore ω (p + 1) p (1, 0) (s, 0) ≠ pairScore ω p p (1, 0) (s, 0) := by
  rw [const_channel_rotates, const_channel_rotates]
  simp only [add_sub_cancel_left, one_mul, sub_self, zero_mul, cos_zero, mul_one]
  intro h
  exact hω (mul_left_cancel₀ hs (by rw [h, mul_one]))

/-! ## swigluoai outside its regime -/

/-- `g σ(α g)` is positive for `g > 0`, `α > 0`. -/
lemma gate_pos {α g : ℝ} (hg : 0 < g) : 0 < g * sigmoid (α * g) := mul_pos hg (sigmoid_pos _)

/-- **swigluoai ⊆ SiLU fails above the gate clamp.** For `gate > 7` (and `up = 0`) gpt-oss's
unit is `7 σ(1.702·7)`, strictly less than the unclamped `gate σ(1.702 gate)`. -/
theorem swigluoai_gate_out_of_regime (gate : ℝ) (hg : 7 < gate) :
    swigluoai gate 0 ≠ (0 + 1) * silu (1.702 * gate) / 1.702 := by
  rw [mul_div_assoc, ← sigmoid_scale_fold 1.702 gate (by norm_num), swigluoai, min_eq_right hg.le]
  norm_num
  have h1 : sigmoid (1.702 * 7) < sigmoid (1.702 * gate) := sigmoid_lt (by linarith)
  have h2 := sigmoid_pos (1.702 * 7)
  intro h
  nlinarith

/-- **…and outside the up clamp.** For `|up| > 7` and a gate in `(0, 7]` the clamp changes the
`up + 1` factor, and the gate factor is nonzero. -/
theorem swigluoai_up_out_of_regime (gate up : ℝ) (hg0 : 0 < gate) (hg : gate ≤ 7)
    (hu : 7 < |up|) : swigluoai gate up ≠ (up + 1) * silu (1.702 * gate) / 1.702 := by
  rw [mul_div_assoc, ← sigmoid_scale_fold 1.702 gate (by norm_num), swigluoai, min_eq_left hg]
  have hpos := gate_pos (α := 1.702) hg0
  intro h
  have hc : max (min up 7) (-7) + 1 = up + 1 := mul_right_cancel₀ hpos.ne' (by rw [h])
  rcases lt_abs.mp hu with h1 | h1
  · rw [min_eq_right h1.le, max_eq_left (by norm_num)] at hc; linarith
  · rw [min_eq_left (by linarith), max_eq_right (by linarith)] at hc; linarith

/-- **swigluoai needs an up bias (or constant channel).** A bias-free up branch `a·up` cannot
absorb the `+1`: at `up = 0` the unit is `g σ(1.702 g) ≠ 0` but the host gives `0`. -/
theorem swigluoai_needs_up_bias (gate a : ℝ) (hg0 : 0 < gate) (hg : gate ≤ 7) :
    swigluoai gate 0 ≠ (a * 0) * silu (1.702 * gate) / 1.702 := by
  rw [swigluoai, min_eq_left hg]
  have := gate_pos (α := 1.702) hg0
  norm_num
  exact ⟨hg0.ne', (sigmoid_pos _).ne'⟩

/-! ## Norms: the same epsilon, and the qk-norm regime -/

/-- **RMSNorm determines its epsilon.** On a vector with a nonzero coordinate where the weight
is nonzero, two RMSNorms with nonnegative epsilons agree only if the epsilons are equal. -/
theorem rmsNorm_eps_inj {n : ℕ} (eps eps' : ℝ) (he : 0 ≤ eps) (he' : 0 ≤ eps') (w z : Fin n → ℝ)
    (i : Fin n) (hz : z i ≠ 0) (hw : w i ≠ 0) (h : rmsNorm eps w z i = rmsNorm eps' w z i) :
    eps = eps' := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  have hS : 0 < (∑ l, z l ^ 2) / n :=
    div_pos (lt_of_lt_of_le (by positivity) (Finset.single_le_sum
      (fun l _ => sq_nonneg (z l)) (Finset.mem_univ i))) hn
  have hs1 : 0 < √((∑ l, z l ^ 2) / n + eps) := Real.sqrt_pos.mpr (by linarith)
  have hs2 : 0 < √((∑ l, z l ^ 2) / n + eps') := Real.sqrt_pos.mpr (by linarith)
  simp only [rmsNorm] at h
  rw [div_eq_div_iff hs1.ne' hs2.ne'] at h
  have hs : √((∑ l, z l ^ 2) / n + eps') = √((∑ l, z l ^ 2) / n + eps) :=
    mul_left_cancel₀ (mul_ne_zero hw hz) h
  have := (Real.sqrt_inj (by linarith) (by linarith)).mp hs
  linarith

/-- **LayerNorm ⊆ RMSNorm needs the same epsilon.** With `eps' ≠ eps`, on any input whose
centred coordinate `i` is nonzero (and `γ i ≠ 0`), `γ ⊙ RMSNorm_{eps'}(P x) + β` misses the
LayerNorm output at `i`. -/
theorem layerNorm_ne_rms_other_eps {n : ℕ} (eps eps' : ℝ) (he : 0 ≤ eps) (he' : 0 ≤ eps')
    (hne : eps ≠ eps') (γ β x : Fin n → ℝ) (i : Fin n) (hγ : γ i ≠ 0)
    (hx : center x i ≠ 0) :
    layerNorm eps γ β x i ≠ γ i * rmsNorm eps' 1 (center x) i + β i := by
  rw [layerNorm_eq_rms_center]
  intro h
  have h' : rmsNorm eps 1 (center x) i = rmsNorm eps' 1 (center x) i :=
    mul_left_cancel₀ hγ (by linarith)
  exact hne (rmsNorm_eps_inj eps eps' he he' 1 (center x) i hx one_ne_zero h')

/-- **qk-norm ⊆ plain fails outside its regime.** If a token pair's RMS product differs from the
folded constants' and the folded plain logit is nonzero, the plain host's logit is not the
qk-norm logit. -/
theorem qknorm_out_of_regime {n : ℕ} (eps : ℝ) (wq wk q k : Fin n → ℝ) (rq rk : ℝ)
    (hq : rq ≠ 0) (hk : rk ≠ 0) (hq' : rms eps q ≠ 0) (hk' : rms eps k ≠ 0)
    (hreg : rms eps q * rms eps k ≠ rq * rk)
    (hl : (fun i => wq i * q i / rq) ⬝ᵥ (fun i => wk i * k i / rk) ≠ 0) :
    rmsNorm eps wq q ⬝ᵥ rmsNorm eps wk k ≠
      (fun i => wq i * q i / rq) ⬝ᵥ (fun i => wk i * k i / rk) := by
  rw [qknorm_logit eps wq wk q k rq rk hq hk hq' hk']
  intro h
  have h1 : (rq * rk) / (rms eps q * rms eps k) = 1 := by
    have := mul_right_cancel₀ hl (h.trans (one_mul _).symm)
    exact this
  rw [div_eq_one_iff_eq (mul_ne_zero hq' hk')] at h1
  exact hreg h1.symm

end EP
