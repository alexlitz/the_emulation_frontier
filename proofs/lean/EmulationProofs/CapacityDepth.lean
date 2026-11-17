import EmulationProofs.MoE
import EmulationProofs.Block
import EmulationProofs.PreNorm
import EmulationProofs.Activation

/-!
# Capacity rules that cost depth: MoE on a dense host, untied on a looped host,
# parallel on a sequential host

* **dense host ⊇ top-1 MoE.** A top-1, unit-gate MoE is discontinuous on its routing boundary
  (`top1_moe_not_continuous`), so NO continuous host (any depth, any width: RMSNorm with `ε > 0`,
  softmax attention, ReLU / SiLU / GeLU MLPs are all continuous) equals it on every input
  (`continuous_host_ne_top1_moe`). Exactness is REFUTED at every depth, `dense_hard_select`'s
  "select layer" included. On the inputs whose router margin is at least `δ`, two norm-free
  FFN layers DO reproduce it exactly (`moe_two_layer_margin`): a select layer computes
  `s_e = ∑_{e'} relu(z_{e'} - z_e)` (`0` for the winner, `≥ δ` for every loser), and the expert
  layer masks every unit with `relu(p - λ s_e)`. Under a pre-norm the carried select channel is
  not readable exactly (`select_carry_prenorm_fails`).
* **looped host ⊇ untied target.** Exact when the target is periodic with the host's period
  (`untied_sub_looped_periodic`). Not at the target's width in general
  (`looped_same_width_fails`). With two extra channels (a counter and a constant) and both
  layers' units in the block, norm-free and on bounded inputs, it is exact
  (`looped_counter_two`), so "no looped host of any width" is false.
* **sequential host ⊇ parallel target at 2× width** (norm-free), with `2 d` identity units in
  each MLP (`parallel_sub_seq_2x`).
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## dense ⊇ top-1 MoE: not exact at any depth -/

/-- **RMSNorm is continuous** for `ε > 0` (every input, including `0`). -/
theorem rmsNorm_continuous {n : ℕ} (eps : ℝ) (heps : 0 < eps) (w : Fin n → ℝ) :
    Continuous (rmsNorm eps w) := by
  refine continuous_pi (fun i => ?_)
  unfold rmsNorm
  refine Continuous.div (by fun_prop) (by fun_prop) (fun x => ?_)
  have : 0 < (∑ l, x l ^ 2) / n + eps := by positivity
  exact (Real.sqrt_pos.mpr this).ne'

/-- The smallest top-1 MoE: input `u ∈ ℝ²`, router logits `z = (u₀, -u₀)`, expert `0` a single
ReLU unit `relu(u₁)`, expert `1` the zero FFN, combined with the gates `G z` (any rule). -/
def toyTop1 (G : (Fin 2 → ℝ) → Fin 2 → ℝ) (u : Fin 2 → ℝ) : ℝ :=
  G ![u 0, -u 0] 0 * relu (u 1) + G ![u 0, -u 0] 1 * 0

/-- A unit-gate top-1 router: the strict winner gets gate `1`, a strict loser gate `0`. Ties
may be broken in any way. -/
def UnitTop1 (G : (Fin 2 → ℝ) → Fin 2 → ℝ) : Prop :=
  (∀ z : Fin 2 → ℝ, z 1 < z 0 → G z 0 = 1) ∧ (∀ z : Fin 2 → ℝ, z 0 < z 1 → G z 0 = 0)

lemma toyTop1_pos {G : (Fin 2 → ℝ) → Fin 2 → ℝ} (hG : UnitTop1 G) {t : ℝ} (ht : 0 < t) :
    toyTop1 G ![t, 1] = 1 := by
  have := hG.1 ![t, -t] (by simp; linarith)
  simp [toyTop1, this, relu]

lemma toyTop1_neg {G : (Fin 2 → ℝ) → Fin 2 → ℝ} (hG : UnitTop1 G) {t : ℝ} (ht : 0 < t) :
    toyTop1 G ![-t, 1] = 0 := by
  have := hG.2 ![-t, t] (by simp; linarith)
  simp only [toyTop1]
  simp only [Matrix.cons_val_zero, neg_neg]
  simp [this]

/-- **A top-1 unit-gate MoE is discontinuous** (any tie-break). At `u = (0, 1)` the router
switches: for `u₀ > 0` the output is `relu(1) = 1`, for `u₀ < 0` it is `0`. -/
theorem top1_moe_not_continuous (G : (Fin 2 → ℝ) → Fin 2 → ℝ) (hG : UnitTop1 G) :
    ¬ ContinuousAt (toyTop1 G) ![0, 1] := by
  intro hc
  obtain ⟨δ, hδ, h⟩ := Metric.continuousAt_iff.mp hc (1 / 2) (by norm_num)
  have near : ∀ s : ℝ, |s| < δ → dist (![s, 1] : Fin 2 → ℝ) ![0, 1] < δ := by
    intro s hs
    rw [dist_pi_lt_iff hδ]
    intro i; fin_cases i <;> simp [hs, hδ]
  have h1 := h (near (δ / 2) (by rw [abs_of_pos (by positivity)]; linarith))
  have h2 := h (near (-(δ / 2)) (by rw [abs_neg, abs_of_pos (by positivity)]; linarith))
  rw [toyTop1_pos hG (by positivity), Real.dist_eq] at h1
  rw [toyTop1_neg hG (by positivity), Real.dist_eq] at h2
  have := abs_lt.mp h1; have := abs_lt.mp h2
  linarith

/-- **No continuous host equals a top-1 MoE, at any depth or width** (impossibility). A host
built from continuous layers (RMSNorm with `ε > 0` (`rmsNorm_continuous`), softmax attention,
ReLU / SiLU / GeLU MLPs, residual sums, linear maps) is a continuous function of its input;
the top-1 MoE `toyTop1 G` is not. So the "exact" mode cannot use `MOE_DEPTH_FACTOR`, nor the
`E×` hard-select construction: the select is never exactly computable. -/
theorem continuous_host_ne_top1_moe (G : (Fin 2 → ℝ) → Fin 2 → ℝ) (hG : UnitTop1 G)
    (g : (Fin 2 → ℝ) → ℝ) (hg : Continuous g) : g ≠ toyTop1 G := by
  rintro rfl
  exact top1_moe_not_continuous G hG hg.continuousAt

/-! ## …but on a router margin, two norm-free FFN layers are exact -/

lemma relu_nonneg' (t : ℝ) : 0 ≤ relu t := le_max_right _ _

lemma relu_of_nonpos {t : ℝ} (ht : t ≤ 0) : relu t = 0 := max_eq_right ht

lemma relu_of_nonneg {t : ℝ} (ht : 0 ≤ t) : relu t = t := max_eq_left ht

/-- **dense ⊇ top-1 MoE at 2 FFN layers, on a router margin** (exact, norm-free). Router logits
`z`, expert `e`'s unit pre-activations `p e j` and down columns `v e j`; the target is the
winner `e*`'s FFN `∑_j relu(p e* j) v e* j` (top-1, unit gate). The host's first FFN (the select
layer, `E²` ReLU units reading the linear logits) writes `s_e = ∑_{e'} relu(z_{e'} - z_e)` into
`E` extra residual channels (residual width `d + E`); the second FFN (`E · e_w` masked units
`relu(p e j - λ s_e)`, plus `E` units `relu(s_e) = s_e` that write `-s_e` to erase the select
channels) is the expert layer. If `e*` wins by a margin `δ > 0` and `λ δ ≥ p e j` for every
loser's unit, the expert layer's output is exactly the target, and the select channels are
erased exactly. Inputs with a margin below `δ` (the routing boundary) are excluded, as they must
be (`continuous_host_ne_top1_moe`). -/
theorem moe_two_layer_margin {E ew : ℕ} {V : Type*} [AddCommGroup V] [Module ℝ V]
    (z : Fin E → ℝ) (p : Fin E → Fin ew → ℝ) (v : Fin E → Fin ew → V) (estar : Fin E)
    (δ lam : ℝ) (hδ : 0 < δ) (hlam : 0 ≤ lam)
    (hmargin : ∀ e, e ≠ estar → z e + δ ≤ z estar)
    (hmask : ∀ e, e ≠ estar → ∀ j, p e j ≤ lam * δ) :
    let s : Fin E → ℝ := fun e => ∑ e', relu (z e' - z e)
    (∑ e, ∑ j, relu (p e j - lam * s e) • v e j = ∑ j, relu (p estar j) • v estar j) ∧
      (∀ e, relu (s e) = s e) := by
  intro s
  have hs0 : s estar = 0 := by
    refine Finset.sum_eq_zero (fun e' _ => relu_of_nonpos ?_)
    rcases eq_or_ne e' estar with h | h
    · simp [h]
    · linarith [hmargin e' h]
  have hsδ : ∀ e, e ≠ estar → δ ≤ s e := by
    intro e he
    calc δ ≤ relu (z estar - z e) := by
          rw [relu_of_nonneg (by linarith [hmargin e he])]; linarith [hmargin e he]
      _ ≤ s e := Finset.single_le_sum (f := fun e' => relu (z e' - z e))
          (fun e' _ => relu_nonneg' _) (Finset.mem_univ estar)
  refine ⟨?_, fun e => relu_of_nonneg (Finset.sum_nonneg (fun _ _ => relu_nonneg' _))⟩
  rw [Finset.sum_eq_single estar]
  · simp [hs0]
  · intro e _ he
    refine Finset.sum_eq_zero (fun j _ => ?_)
    rw [relu_of_nonpos, zero_smul]
    have := hmask e he j
    have := mul_le_mul_of_nonneg_left (hsδ e he) hlam
    linarith
  · simp

/-- The layout `[ρ u; s]` (stream `ρ u` in `d` channels, a carried `s` in `k` more) is
`ρ [u; 0] + [0; s]`. -/
lemma append_smul_split' {d k : ℕ} (ρ : ℝ) (u : Fin d → ℝ) (s : Fin k → ℝ) :
    Fin.append (ρ • u) s = ρ • Fin.append u (0 : Fin k → ℝ) + Fin.append (0 : Fin d → ℝ) s := by
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · simp
  · simp only [Fin.append_right, Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]
    ring

/-- **The select channel does not survive a pre-norm** (the 2-layer construction is norm-free
only). Pre-norm host: residual `[ρ u; s]`, the stream `ρ u` (scaled by any `ρ > 0`, as the
target's residual may be) next to the carried, scale-free select vector `s ≠ 0` (it is a
function of the normed input). The expert layer must read a fixed pre-activation `t ≠ 0` (the
target's, also scale-free). No linear readout `c` of the host RMSNorm (any weight, any
`ε ≥ 0`, any widths) returns `t` for every `ρ > 0` (`prenorm_readout_obstruction`). -/
theorem select_carry_prenorm_fails {d k : ℕ} (eps : ℝ) (heps : 0 ≤ eps)
    (w c : Fin (d + k) → ℝ) (u : Fin d → ℝ) (s : Fin k → ℝ) (hu : u ≠ 0) (hs : s ≠ 0)
    (t : ℝ) (ht : t ≠ 0) :
    ¬ ∀ ρ : ℝ, 0 < ρ → ∑ i, c i * rmsNorm eps w (Fin.append (ρ • u) s) i = t := by
  intro h
  have hd : 0 < d + k := by
    rcases Nat.eq_zero_or_pos d with hd | hd
    · subst hd; exact absurd (funext fun i => i.elim0) hu
    · omega
  simp only [append_smul_split'] at h
  have hob := prenorm_readout_obstruction hd eps heps w c _ _ t ht h
  simp only [Fin.sum_univ_add, Fin.append_left, Fin.append_right, Pi.zero_apply, mul_zero,
    zero_mul, Finset.sum_const_zero, add_zero, zero_add, ne_eq, OfNat.ofNat_ne_zero,
    not_false_eq_true, zero_pow] at hob
  have hU := sum_sq_pos_of_ne_zero u hu
  have hV := sum_sq_pos_of_ne_zero s hs
  have : 0 ≤ ((d + k : ℕ) : ℝ) * eps := by positivity
  have : 0 < (∑ i, u i ^ 2) * (∑ i, s i ^ 2 + ((d + k : ℕ) : ℝ) * eps) := by positivity
  linarith

/-! ## looped host ⊇ untied target -/

section loop

variable {R : Type*} [AddCommGroup R]

lemma untied_identity {P : Type*} (f : P → R → R) (ψ : ℕ → P) (p₀ : P) (hp₀ : f p₀ = id)
    (n : ℕ) (h : ∀ t < n, ψ t = p₀) (x : R) : untied f ψ n x = x := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show f (ψ n) (untied f ψ n x) = x
    rw [h n (by omega), hp₀, id, ih (fun t ht => h t (by omega))]

/-- **looped host ⊇ periodic untied target** (exact). The target's `K b` layers repeat with
period `b` (`ψ t = θ (t mod b)`). The host loops `K` times over `b' ≥ b` blocks: the first `b`
carry `θ`, the rest the identity parameter `p₀` (`f p₀ = id`, e.g. zeroed output projections,
`zeroed_sublayer`). The host equals the target on every input. The alignment is necessary
in general (`looped_same_width_fails`). -/
theorem untied_sub_looped_periodic {P : Type*} (f : P → R → R) (θ ψ : ℕ → P) (p₀ : P)
    (hp₀ : f p₀ = id) (b b' K : ℕ) (hbb : b ≤ b') (hψ : ∀ t < K * b, ψ t = θ (t % b)) (x : R) :
    (untied f (fun i => if i < b then θ i else p₀) b')^[K] x = untied f ψ (K * b) x := by
  have hblk : untied f (fun i => if i < b then θ i else p₀) b' = untied f θ b := by
    funext y
    obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le hbb
    rw [untied_add, untied_identity f _ p₀ hp₀ m (fun t _ => by simp)]
    exact untied_congr f _ _ b (fun t ht => by simp [ht]) y
  rw [hblk, ← looped_sub_untied]
  exact untied_congr f _ _ _ (fun t ht => (hψ t ht).symm) x

/-- The 2-layer untied residual ReLU target: layer 1 is `x + (-2 relu x + 2 relu (-x))`,
layer 2 is the identity (zeroed MLP). -/
def negTarget (x : ℝ) : ℝ :=
  (fun y : ℝ => y + 0) ((fun y : ℝ => y + (-2 * relu y + 2 * relu (-y))) x)

lemma negTarget_eq (x : ℝ) : negTarget x = -x := by
  simp only [negTarget, relu]
  rcases le_total 0 x with h | h
  · rw [max_eq_left h, max_eq_right (by linarith)]; ring
  · rw [max_eq_right h, max_eq_left (by linarith)]; ring

/-- **untied ⊄ looped at the target's width** (impossibility). The 2-layer untied residual ReLU
target `negTarget` (width 1) computes `x ↦ -x`. A 1-block host looped twice, at width 1 with the
residual stream as its interface (host residual = target residual, as in every other
construction here, `hidden_pad_stack` included), computes `g ∘ g` for its block `g`; no
continuous `g` (any pre-norm with `ε > 0`, attention, MLP, any width of MLP) has `g ∘ g = -x`:
`g` is injective, so strictly monotone, so `g ∘ g` is increasing. What this does NOT say: with
free extra residual width it fails — a rotation at width 2 squares to `-1`
(`looped_width2_neg`), and `looped_counter_two` hosts any 2-layer target. -/
theorem looped_same_width_fails :
    (∀ x, negTarget x = -x) ∧ ¬ ∃ g : ℝ → ℝ, Continuous g ∧ ∀ x, g (g x) = negTarget x := by
  refine ⟨negTarget_eq, ?_⟩
  rintro ⟨g, hg, hgg⟩
  have hinj : Function.Injective g := by
    intro a b hab
    have := congrArg g hab
    rw [hgg, hgg, negTarget_eq, negTarget_eq] at this
    linarith
  have hmono : StrictMono (g ∘ g) := by
    rcases hg.strictMono_of_inj hinj with h | h
    · exact h.comp h
    · exact fun a b hab => h (h hab)
  have := hmono (show (0 : ℝ) < 1 by norm_num)
  simp only [Function.comp, hgg, negTarget_eq] at this
  linarith

/-- **…but width 2 suffices for it**: the residual block `p ↦ p + (R p - p)` with `R` the
quarter rotation, looped twice, maps `(x, 0)` to `(-x, 0)`. -/
theorem looped_width2_neg (x : ℝ) :
    let g : ℝ × ℝ → ℝ × ℝ := fun p => p + ((-p.2, p.1) - p)
    g (g (x, 0)) = (-x, 0) := by
  simp

/-- **looped host ⊇ untied target with a counter** (exact, norm-free, bounded inputs). Target:
two untied residual ReLU-MLP layers `f₁ x = x + ∑_j relu(a_j·x) v_j`,
`f₂ x = x + ∑_k relu(a'_k·x) v'_k`. Host: ONE residual ReLU-MLP block looped twice, on the
residual `(x, c, o)` with two extra channels, a counter `c` (embedded `0`) and a constant `o`
(embedded `1`). Its MLP holds both layers' units, masked by `-B c` and `-B (o - c)`, and one
unit `relu(o)` that adds `1` to the counter. Every pre-activation is linear in `(x, c, o)`. If
`a'_k·x ≤ B` and `a_j·f₁(x) ≤ B` (bounded inputs), the two iterations give exactly
`(f₂ (f₁ x), 2, 1)`. Cost: 2 channels and the units of both layers in one block. Under a
pre-norm the counter is a carried channel (`select_carry_prenorm_fails`), so this is
norm-free only. -/
theorem looped_counter_two {d m m' : ℕ} (a : Fin m → Fin d → ℝ) (v : Fin m → Fin d → ℝ)
    (a' : Fin m' → Fin d → ℝ) (v' : Fin m' → Fin d → ℝ) (B : ℝ) (x : Fin d → ℝ) :
    let f₁ : (Fin d → ℝ) → (Fin d → ℝ) := fun y => y + ∑ j, relu (a j ⬝ᵥ y) • v j
    let f₂ : (Fin d → ℝ) → (Fin d → ℝ) := fun y => y + ∑ k, relu (a' k ⬝ᵥ y) • v' k
    let blk : (Fin d → ℝ) × ℝ × ℝ → (Fin d → ℝ) × ℝ × ℝ := fun s =>
      (s.1 + ∑ j, relu (a j ⬝ᵥ s.1 - B * s.2.1) • v j +
          ∑ k, relu (a' k ⬝ᵥ s.1 - B * (s.2.2 - s.2.1)) • v' k,
        s.2.1 + relu s.2.2, s.2.2)
    (∀ k, a' k ⬝ᵥ x ≤ B) → (∀ j, a j ⬝ᵥ f₁ x ≤ B) →
      blk (blk (x, 0, 1)) = (f₂ (f₁ x), 2, 1) := by
  intro f₁ f₂ blk h1 h2
  have r1 : relu (1 : ℝ) = 1 := relu_of_nonneg (by norm_num)
  have e1 : blk (x, 0, 1) = (f₁ x, 1, 1) := by
    simp only [blk, f₁, mul_zero, sub_zero, mul_one, r1, zero_add]
    have : ∑ k, relu (a' k ⬝ᵥ x - B) • v' k = 0 :=
      Finset.sum_eq_zero (fun k _ => by rw [relu_of_nonpos (by linarith [h1 k]), zero_smul])
    rw [this, add_zero]
  rw [e1]
  simp only [blk, f₂, mul_one, sub_self, mul_zero, sub_zero, r1]
  have : ∑ j, relu (a j ⬝ᵥ f₁ x - B) • v j = 0 :=
    Finset.sum_eq_zero (fun j _ => by rw [relu_of_nonpos (by linarith [h2 j]), zero_smul])
  rw [this, add_zero]
  norm_num

end loop

/-! ## sequential host ⊇ parallel target at 2× width (norm-free) -/

/-- ReLU is an exact identity unit pair: `relu t - relu (-t) = t`. -/
theorem relu_identity_pair (t : ℝ) : relu t - relu (-t) = t := by
  simp only [relu]
  rcases le_total 0 t with h | h
  · rw [max_eq_left h, max_eq_right (by linarith)]; ring
  · rw [max_eq_right h, max_eq_left (by linarith)]; ring

/-- SiLU is an exact identity unit pair: `silu t - silu (-t) = t (σ t + σ (-t)) = t`. -/
theorem silu_identity_pair (t : ℝ) : silu t - silu (-t) = t := by
  simp only [silu, Real.sigmoid_neg]; ring

/-- **parallel ⊆ seq at 2× width, norm-free** (exact, whole stack). Host residual `(c₀, c₁)`,
invariant `c₁ = 0` at every block boundary. In host block `l` the attention reads `c₀` and
writes `A_l(c₀)` into `c₁`; the MLP reads `c₀` (still the block INPUT) and writes `M_l(c₀)`
into `c₀`, and with `2 d` identity units `φ(v) - φ(-v) = v` (ReLU `relu_identity_pair`, SiLU
`silu_identity_pair`; any `φ(v) = v s(v)` with `s(v) + s(-v) = 1`, GeLU included) it moves `c₁`
into `c₀` and erases `c₁`. After `l` blocks the host residual is `(target_l, 0)` for every `l`.
Sublayers act on a residual of any shape `ι → ℝ` (a sequence of tokens included), arbitrary
maps. Cost: 2× residual width, `+2d` MLP units per block. Under a pre-norm this layout fails
(`parallel_sub_seq_prenorm_fails`); whether some other pre-norm 2× host exists is open. -/
theorem parallel_sub_seq_2x {ι : Type*} (φ : ℝ → ℝ) (hφ : ∀ t, φ t - φ (-t) = t)
    (A M : ℕ → (ι → ℝ) → (ι → ℝ)) (x : ι → ℝ) :
    let target : ℕ → ι → ℝ := fun l => Nat.rec x (fun l y => parBlock (A l) (M l) y) l
    let idU : (ι → ℝ) → (ι → ℝ) := fun v i => φ (v i) - φ (-(v i))
    let A' : ℕ → (ι → ℝ) × (ι → ℝ) → (ι → ℝ) × (ι → ℝ) := fun l s => (0, A l s.1)
    let M' : ℕ → (ι → ℝ) × (ι → ℝ) → (ι → ℝ) × (ι → ℝ) := fun l s =>
      (M l s.1 + idU s.2, -idU s.2)
    let host : ℕ → (ι → ℝ) × (ι → ℝ) := fun l =>
      Nat.rec (x, 0) (fun l s => seqBlock (A' l) (M' l) s) l
    ∀ l, host l = (target l, 0) := by
  intro target idU A' M' host l
  have hid : ∀ v, idU v = v := fun v => funext fun i => hφ (v i)
  induction l with
  | zero => rfl
  | succ l ih =>
    show seqBlock (A' l) (M' l) (host l) = (parBlock (A l) (M l) (target l), 0)
    rw [ih]
    simp only [seqBlock, parBlock, A', M', hid, Prod.mk_add_mk, add_zero, zero_add,
      add_neg_cancel, Prod.mk.injEq, and_true]
    abel

end EP
