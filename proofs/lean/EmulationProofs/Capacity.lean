import EmulationProofs.Norm
import EmulationProofs.Position
import EmulationProofs.MoE

/-!
# Capacity rules: hidden-width padding, wider FFNs, a shared-expert host

* **Hidden padding** (`hidden_pad_stack`, `hidden_pad_blocks`, `hidden_pad_stack_layerNorm`): a
  pre-norm target of width `d` runs exactly on a host of width `d + e` whose residual is the
  zero-padded target residual. Under RMSNorm the padded mean square is `d/(d+e)` of the target's,
  so every host norm takes weight `√(d/(d+e)) w` and `ε' = ε d/(d+e)` (`qknorm_pad`). Under
  LayerNorm zero-padding alone fails (the mean and the variance both change), so the host keeps
  its residual CENTRED (every writer post-composed with the centring, as in
  `layerNorm_sub_rmsNorm`); then the host LayerNorm is the padded RMSNorm of the centred vector.
* **Wider FFN** (`ffn_extra_units`, `ffn_extra_zero_input`, `glu_zero_up_unit`,
  `moe_wider_experts`, `moe_extra_experts`).
* **A shared-expert host** (`moe_sub_moe_shared`, `moe_sub_moe_shared_gated`).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## Hidden padding -/

section pad

variable {d e : ℕ}

/-- The first `d` coordinates of a width-`d + e` vector. -/
def unpad (y : Fin (d + e) → ℝ) : Fin d → ℝ := fun i => y (Fin.castAdd e i)

@[simp] lemma unpad_padVec (x : Fin d → ℝ) : unpad (padVec (e := e) x) = x := by
  funext i; simp [unpad]

lemma padVec_add' (x y : Fin d → ℝ) : padVec (e := e) (x + y) = padVec x + padVec y := by
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i <;> simp

/-- The host RMSNorm for a zero-padded residual: weight `√(d/(d+e)) w` on the real channels
(`0` on the padding), epsilon `ε d/(d+e)`. -/
def padNorm (eps : ℝ) (w : Fin d → ℝ) : (Fin (d + e) → ℝ) → (Fin (d + e) → ℝ) :=
  rmsNorm (eps * d / (d + e : ℕ)) (padVec (fun i => √((d : ℝ) / (d + e : ℕ)) * w i))

/-- **Hidden padding, the whole pre-norm RMSNorm model** (exact). Target: a pre-norm stack
`h_{l+1} = h_l + f_l(RMSNorm_{ε_l, w_l}(h))` of width `d` (each `f l` an ARBITRARY sequence map:
an attention sublayer, an FFN, anything), embedding `emb`, final norm `(ε_f, w_f)` and any head.
Host: width `d + e`, embedding `[emb; 0]`, norms `padNorm` (weight `√(d/(d+e)) w`,
`ε d/(d+e)`), and each sublayer reading the real channels of its normed input (zero columns on
the padding) and writing only real channels (zero rows). Then the host residual is the padded
target residual at every layer, and the head sees exactly the target's final norm. -/
theorem hidden_pad_stack (hd : d ≠ 0) {T : ℕ} (eps : ℕ → ℝ) (w : ℕ → Fin d → ℝ)
    (f : ℕ → (Fin T → Fin d → ℝ) → (Fin T → Fin d → ℝ)) (emb : Fin T → Fin d → ℝ)
    (epsf : ℝ) (wf : Fin d → ℝ) {Out : Type*} (head : (Fin d → ℝ) → Out) (L : ℕ) :
    let hostNorm : ℕ → (Fin (d + e) → ℝ) → (Fin (d + e) → ℝ) := fun l => padNorm (eps l) (w l)
    let hostF : ℕ → (Fin T → Fin (d + e) → ℝ) → (Fin T → Fin (d + e) → ℝ) :=
      fun l z t => padVec (f l (fun s => unpad (z s)) t)
    let hostE : Fin T → Fin (d + e) → ℝ := fun t => padVec (emb t)
    (∀ l, stack hostNorm hostF hostE l =
        fun t => padVec (stack (fun l => rmsNorm (eps l) (w l)) f emb l t)) ∧
    (∀ t, head (unpad (padNorm (e := e) epsf wf (stack hostNorm hostF hostE L t))) =
      head (rmsNorm epsf wf (stack (fun l => rmsNorm (eps l) (w l)) f emb L t))) := by
  intro hostNorm hostF hostE
  have key : ∀ l, stack hostNorm hostF hostE l =
      fun t => padVec (stack (fun l => rmsNorm (eps l) (w l)) f emb l t) := by
    intro l
    induction l with
    | zero => rfl
    | succ l ih =>
      funext t
      show stack hostNorm hostF hostE l t + hostF l (fun s => hostNorm l (stack hostNorm hostF hostE l s)) t =
        padVec (stack (fun l => rmsNorm (eps l) (w l)) f emb l t +
          f l (fun s => rmsNorm (eps l) (w l) (stack (fun l => rmsNorm (eps l) (w l)) f emb l s)) t)
      rw [ih]
      simp only [hostF, hostNorm, padNorm, qknorm_pad hd, unpad_padVec, padVec_add']
  refine ⟨key, fun t => ?_⟩
  rw [key, padNorm, qknorm_pad hd, unpad_padVec]

/-- **The padded epsilon is forced** (converse). If a host RMSNorm over the padded width `d + e`
(ANY weight, ANY `ε' ≥ 0`) reproduces the target's RMSNorm coordinate `i` (`w i ≠ 0`) on every
input, then `ε' = ε d/(d+e)`. So with `e > 0` and `ε > 0` a host that keeps its own epsilon
`ε' = ε` is not exact; the relation's hidden rule is exact only when the host's epsilon can be
set (or both are `0`). -/
theorem hidden_pad_needs_eps (hd : d ≠ 0) (eps eps' : ℝ) (heps : 0 ≤ eps) (heps' : 0 ≤ eps')
    (w : Fin d → ℝ) (wh : Fin (d + e) → ℝ) (i : Fin d) (hw : w i ≠ 0)
    (h : ∀ x : Fin d → ℝ, rmsNorm eps' wh (padVec x) (Fin.castAdd e i) = rmsNorm eps w x i) :
    eps' = eps * d / (d + e : ℕ) := by
  have hdR : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hDR : (0 : ℝ) < ((d + e : ℕ) : ℝ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hd).trans_le (Nat.le_add_right d e)
  have hsum : ∀ t : ℝ, ∑ l, (padVec (e := e) (t • Pi.single i (1 : ℝ))) l ^ 2 = t ^ 2 ∧
      ∑ l, (t • Pi.single i (1 : ℝ)) l ^ 2 = t ^ 2 := by
    intro t
    have h1 : ∑ l, (t • Pi.single i (1 : ℝ)) l ^ 2 = t ^ 2 := by
      simp [Pi.single_apply, Finset.sum_ite_eq']
    refine ⟨?_, h1⟩
    simp only [Fin.sum_univ_add, padVec_left, padVec_right]
    simpa using h1
  have hval : ∀ t : ℝ, 0 < t →
      (wh (Fin.castAdd e i)) ^ 2 * (t ^ 2 / d + eps) =
        (w i) ^ 2 * (t ^ 2 / ((d + e : ℕ) : ℝ) + eps') := by
    intro t ht
    have := h (t • Pi.single i 1)
    simp only [rmsNorm] at this
    rw [(hsum t).1, (hsum t).2] at this
    simp only [padVec_left, Pi.smul_apply, Pi.single_eq_same, smul_eq_mul, mul_one] at this
    have hA : 0 < t ^ 2 / ((d + e : ℕ) : ℝ) + eps' := by positivity
    have hB : 0 < t ^ 2 / d + eps := by positivity
    rw [div_eq_div_iff (Real.sqrt_pos.mpr hA).ne' (Real.sqrt_pos.mpr hB).ne'] at this
    have hsq := congrArg (fun z => z ^ 2) this
    simp only [mul_pow, Real.sq_sqrt hA.le, Real.sq_sqrt hB.le] at hsq
    have ht2 : t ^ 2 ≠ 0 := by positivity
    have hc : t ^ 2 * ((wh (Fin.castAdd e i)) ^ 2 * (t ^ 2 / d + eps)) =
        t ^ 2 * ((w i) ^ 2 * (t ^ 2 / ((d + e : ℕ) : ℝ) + eps')) := by
      linear_combination hsq
    exact mul_left_cancel₀ ht2 hc
  have h1 := hval 1 one_pos
  have h2 := hval 2 two_pos
  have hwd : (w i) ^ 2 * d ≠ 0 := mul_ne_zero (pow_ne_zero 2 hw) hdR.ne'
  have hwh : (wh (Fin.castAdd e i)) ^ 2 * ((d + e : ℕ) : ℝ) = (w i) ^ 2 * d := by
    field_simp at h1 h2
    nlinarith
  have key : (w i) ^ 2 * d * (d * eps) = (w i) ^ 2 * d * (((d + e : ℕ) : ℝ) * eps') := by
    field_simp at h1
    linear_combination h1 - (1 + d * eps) * hwh
  have := mul_left_cancel₀ hwd key
  field_simp
  linarith

/-- Interleave attention and FFN sublayers: stack layer `2l` is block `l`'s attention, layer
`2l + 1` its FFN. -/
def interleave {α : Type*} (a m : ℕ → α) (l : ℕ) : α := if l % 2 = 0 then a (l / 2) else m (l / 2)

/-- **Hidden padding, attention + FFN blocks** (exact). The same statement for a target of `L`
pre-norm transformer blocks (attention sublayer with norm `(εa, wa)`, FFN sublayer with norm
`(εm, wm)`), written as the `2L`-sublayer stack. -/
theorem hidden_pad_blocks (hd : d ≠ 0) {T : ℕ} (εa εm : ℕ → ℝ) (wa wm : ℕ → Fin d → ℝ)
    (A M : ℕ → (Fin T → Fin d → ℝ) → (Fin T → Fin d → ℝ)) (emb : Fin T → Fin d → ℝ)
    (l : ℕ) :
    stack (fun l => padNorm (e := e) (interleave εa εm l) (interleave wa wm l))
        (fun l z t => padVec (interleave A M l (fun s => unpad (z s)) t)) (fun t => padVec (emb t))
        l =
      fun t => padVec (stack (fun l => rmsNorm (interleave εa εm l) (interleave wa wm l))
        (interleave A M) emb l t) :=
  (hidden_pad_stack (e := e) hd (interleave εa εm) (interleave wa wm) (interleave A M) emb 0 0
    id 0).1 l

/-! ### LayerNorm hosts -/

lemma mean_padVec_center (x : Fin d → ℝ) : mean (padVec (e := e) (center x)) = 0 := by
  have h := mean_center x
  simp only [mean, Fin.sum_univ_add, padVec_left, padVec_right, sum_const_zero, add_zero] at h ⊢
  rcases Nat.eq_zero_or_pos d with hd | hd
  · subst hd; simp
  · have : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
    have hs : ∑ i, center x i = 0 := by
      rcases (div_eq_zero_iff.mp h) with h | h
      · exact h
      · exact absurd h this
    rw [hs, zero_div]

/-- The host LayerNorm on a padded CENTRED vector is the padded target LayerNorm. -/
theorem layerNorm_pad_center (hd : d ≠ 0) (eps : ℝ) (γ β x : Fin d → ℝ) :
    layerNorm (eps * d / (d + e : ℕ)) (padVec (e := e) (fun i => √((d : ℝ) / (d + e : ℕ)) * γ i))
        (padVec β) (padVec (center x)) = padVec (layerNorm eps γ β x) := by
  have hrms1 := qknorm_pad (e := e) hd eps (fun _ => 1) (center x)
  have hm := mean_padVec_center (e := e) x
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · have h1 := congrFun hrms1 (Fin.castAdd e i)
    simp only [rmsNorm, padVec_left, mul_one, one_mul] at h1
    simp only [layerNorm, hm, sub_zero, padVec_left]
    rw [show ∀ a b c S : ℝ, a * b * c / S = b * (a * c / S) from fun a b c S => by ring, h1]
    simp only [center, mul_div_assoc]
  · simp [layerNorm, hm]

/-- **Hidden padding, the whole pre-norm LayerNorm model** (exact). Target: a LayerNorm stack
(weights `γ l`, biases `β l`, same `ε`). Host: width `d + e`, LayerNorm with weight
`√(d/(d+e)) γ`, bias `[β; 0]`, epsilon `ε d/(d+e)`; embedding `[P emb; 0]` and every writer
post-composed with the centring `P` (a linear map folded into the output projections, as in
`layerNorm_sub_rmsNorm`). Then the host residual is `[P h; 0]` at every layer, and the host's
final LayerNorm is exactly the padded target final LayerNorm. Zero-padding WITHOUT the centring is
not exact: the pad channels shift the mean. -/
theorem hidden_pad_stack_layerNorm (hd : d ≠ 0) {T : ℕ} (eps : ℝ) (γ β : ℕ → Fin d → ℝ)
    (f : ℕ → (Fin T → Fin d → ℝ) → (Fin T → Fin d → ℝ)) (emb : Fin T → Fin d → ℝ)
    (γf βf : Fin d → ℝ) (L : ℕ) :
    let lnPad : (Fin d → ℝ) → (Fin d → ℝ) → (Fin (d + e) → ℝ) → (Fin (d + e) → ℝ) := fun γ β =>
      layerNorm (eps * d / (d + e : ℕ)) (padVec (fun i => √((d : ℝ) / (d + e : ℕ)) * γ i))
        (padVec β)
    let hostF : ℕ → (Fin T → Fin (d + e) → ℝ) → (Fin T → Fin (d + e) → ℝ) :=
      fun l z t => padVec (center (f l (fun s => unpad (z s)) t))
    let hostE : Fin T → Fin (d + e) → ℝ := fun t => padVec (center (emb t))
    let target := stack (fun l => layerNorm eps (γ l) (β l)) f emb
    (∀ l, stack (fun l => lnPad (γ l) (β l)) hostF hostE l = fun t => padVec (center (target l t))) ∧
    (∀ t, lnPad γf βf (stack (fun l => lnPad (γ l) (β l)) hostF hostE L t) =
      padVec (layerNorm eps γf βf (target L t))) := by
  intro lnPad hostF hostE target
  have hLN : ∀ (γ β x : Fin d → ℝ), lnPad γ β (padVec (center x)) = padVec (layerNorm eps γ β x) :=
    fun γ β x => layerNorm_pad_center hd eps γ β x
  have key : ∀ l, stack (fun l => lnPad (γ l) (β l)) hostF hostE l =
      fun t => padVec (center (target l t)) := by
    intro l
    induction l with
    | zero => rfl
    | succ l ih =>
      funext t
      have harg : (fun s => unpad (lnPad (γ l) (β l) (padVec (center (target l s))))) =
          fun s => layerNorm eps (γ l) (β l) (target l s) := by
        funext s; rw [hLN, unpad_padVec]
      show stack (fun l => lnPad (γ l) (β l)) hostF hostE l t +
          hostF l (fun s => lnPad (γ l) (β l) (stack (fun l => lnPad (γ l) (β l)) hostF hostE l s)) t =
        padVec (center (target l t + f l (fun s => layerNorm eps (γ l) (β l) (target l s)) t))
      rw [ih]
      simp only [hostF, harg, center_add, padVec_add']
  refine ⟨key, fun t => ?_⟩
  rw [key, hLN]

end pad

/-! ## Wider FFN -/

section wider

variable {X Θ : Type*} {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- **Wider FFN, extra units with a zero down-projection** (exact, ANY activation, gated or not).
A host FFN with units `U ⊕ U'` holding the target's units and extra units whose down-projection
columns are `0` (their input weights and activation are arbitrary) equals the target FFN. -/
theorem ffn_extra_units {U U' : Type*} [Fintype U] [Fintype U'] (act : Θ → X → ℝ) (θ : U → Θ)
    (θ' : U' → Θ) (W2 : U → V) (x : X) :
    ffn act (Sum.elim θ θ') (Sum.elim W2 (fun _ => 0)) x = ffn act θ W2 x := by
  simp [ffn, Fintype.sum_sum_type]

/-- **Wider FFN, extra units with zero input weights** (exact): if the extra units' activation
vanishes identically (`act θ₀ x = 0`, e.g. an ungated unit with `φ(0) = 0`, or a gated unit whose
up branch is zeroed, `glu_zero_up_unit`), their down-projection is arbitrary. -/
theorem ffn_extra_zero_input {U U' : Type*} [Fintype U] [Fintype U'] (act : Θ → X → ℝ)
    (θ : U → Θ) (θ₀ : Θ) (h0 : ∀ x, act θ₀ x = 0) (W2 : U → V) (W2' : U' → V) (x : X) :
    ffn act (Sum.elim θ (fun _ : U' => θ₀)) (Sum.elim W2 W2') x = ffn act θ W2 x := by
  simp [ffn, Fintype.sum_sum_type, h0]

/-- A gated unit `φ(g · x) (u · x)`. -/
def gluUnit {n : ℕ} (φ : ℝ → ℝ) (p : (Fin n → ℝ) × (Fin n → ℝ)) (x : Fin n → ℝ) : ℝ :=
  φ (p.1 ⬝ᵥ x) * (p.2 ⬝ᵥ x)

/-- An ungated unit `φ(w · x)`. -/
def plainUnit {n : ℕ} (φ : ℝ → ℝ) (w : Fin n → ℝ) (x : Fin n → ℝ) : ℝ := φ (w ⬝ᵥ x)

/-- **The GLU case** (exact, any `φ`): a gated unit with a zero up-projection outputs
`φ(g·x) · 0 = 0`, whatever its gate weights and whatever `φ`. -/
theorem glu_zero_up_unit {n : ℕ} (φ : ℝ → ℝ) (g x : Fin n → ℝ) : gluUnit φ (g, 0) x = 0 := by
  simp [gluUnit]

/-- …and with BOTH projections zeroed it is `φ(0) · 0 = 0`. -/
theorem glu_zero_unit {n : ℕ} (φ : ℝ → ℝ) (x : Fin n → ℝ) : gluUnit φ (0, 0) x = 0 := by
  simp [gluUnit]

/-- **The ungated case needs `φ(0) = 0` for the zero-input route** (or a zero down-projection,
`ffn_extra_units`): a zero-input ungated unit outputs the constant `φ(0)`. -/
theorem plain_zero_unit {n : ℕ} (φ : ℝ → ℝ) (x : Fin n → ℝ) : plainUnit φ 0 x = φ 0 := by
  simp [plainUnit]

/-- **Wider experts** (exact): each expert widened from `ew` to `ew + ew'` units, the extra units
with zero down-projection, gives the same MoE for any gates. -/
theorem moe_wider_experts {E ew ew' : ℕ} (act : Θ → X → ℝ) (gate : X → Fin E → ℝ)
    (θ : Fin E → Fin ew → Θ) (θ' : Fin E → Fin ew' → Θ) (W2 : Fin E → Fin ew → V) (x : X) :
    moe act gate (fun e => Fin.append (θ e) (θ' e)) (fun e => Fin.append (W2 e) (fun _ => 0)) x =
      moe act gate θ W2 x := by
  simp [moe, ffn, Fin.sum_univ_add]

/-- **More experts, gated off** (exact, conditional on the router): extra experts whose gate is
`0` on every token leave the MoE unchanged. A learned top-k router must therefore keep them out
of the top-k on every token; this is a hypothesis, not proved for a given router. -/
theorem moe_extra_experts {E E' ew : ℕ} (act : Θ → X → ℝ) (gate : X → Fin E → ℝ)
    (θ : Fin E → Fin ew → Θ) (θ' : Fin E' → Fin ew → Θ) (W2 : Fin E → Fin ew → V)
    (W2' : Fin E' → Fin ew → V) (x : X) :
    moe act (fun x => Fin.append (gate x) (fun _ => 0)) (Fin.append θ θ') (Fin.append W2 W2') x =
      moe act gate θ W2 x := by
  simp [moe, Fin.sum_univ_add]

/-! ## moe → moe_shared -/

/-- A MoE layer with a shared expert behind a scalar gate `sg x` (`sg = 1` for an ungated shared
expert, `sg x = σ(w · x)` for Qwen-style). -/
def moeShared {E ew sw : ℕ} (act : Θ → X → ℝ) (gate : X → Fin E → ℝ) (θ : Fin E → Fin ew → Θ)
    (W2 : Fin E → Fin ew → V) (sg : X → ℝ) (θs : Fin sw → Θ) (W2s : Fin sw → V) (x : X) : V :=
  moe act gate θ W2 x + sg x • ffn act θs W2s x

/-- **moe ⊆ moe_shared** (exact): with the shared expert's down-projection zeroed, the host is the
routed MoE, whatever the shared gate (ungated or sigmoid) and the shared up/gate weights. -/
theorem moe_sub_moe_shared {E ew sw : ℕ} (act : Θ → X → ℝ) (gate : X → Fin E → ℝ)
    (θ : Fin E → Fin ew → Θ) (W2 : Fin E → Fin ew → V) (sg : X → ℝ) (θs : Fin sw → Θ) (x : X) :
    moeShared act gate θ W2 sg θs (fun _ => 0) x = moe act gate θ W2 x := by
  simp [moeShared, ffn]

/-- **moe ⊆ moe_shared with a sigmoid-gated shared expert** (exact). Zero the gate's input
weights (gate `σ(0) = 1/2` on every token) and the shared down-projection. A sigmoid gate is
never exactly `0` (`σ > 0`), so "gate at 0" alone is only a limit; the zero down-projection is
what makes it exact. -/
theorem moe_sub_moe_shared_gated {n E ew sw : ℕ} (act : Θ → (Fin n → ℝ) → ℝ)
    (gate : (Fin n → ℝ) → Fin E → ℝ) (θ : Fin E → Fin ew → Θ) (W2 : Fin E → Fin ew → V)
    (θs : Fin sw → Θ) (x : Fin n → ℝ) :
    moeShared act gate θ W2 (fun x => sigmoid ((0 : Fin n → ℝ) ⬝ᵥ x)) θs (fun _ => 0) x =
      moe act gate θ W2 x :=
  moe_sub_moe_shared act gate θ W2 _ θs x

/-- The sigmoid gate is `1/2` with zero input weights and never `0`. -/
theorem shared_gate_half {n : ℕ} (x : Fin n → ℝ) :
    sigmoid ((0 : Fin n → ℝ) ⬝ᵥ x) = 1 / 2 ∧ ∀ t, sigmoid t ≠ 0 := by
  refine ⟨by simp [sigmoid_zero], fun t => (sigmoid_pos t).ne'⟩

end wider

end EP
