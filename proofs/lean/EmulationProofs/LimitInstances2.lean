import EmulationProofs.LimitKit
import EmulationProofs.Attention
import EmulationProofs.MoE

/-!
# Whole-model limit instances: hard masks, the sink, the softcap host, the soft router

One lemma does most of the work (`attn_extra_keys_bound`): attention over `S ⊇ S'` whose logits
agree with the target's on `S'` is within `2M · (Σ_{S∖S'} e^{t_j}) / (Σ_{S'} e^{s_j})` of the
target's attention over `S'` (values bounded by `M`). Then

* **hard mask ← additive mask `−B`**: per layer `≤ 2 M T e^{2R − B}` (logits `|s| ≤ R`, `T` keys);
* **plain ← sink host, sink logit `b → −∞`**: per layer `≤ 2 M e^{b + R}`;
* **dense ⊇ top-1 MoE, the soft router `softmax(β z)`**: per layer `≤ 2 M (E − 1) e^{−β δ_r}`
  on states whose router has margin `δ_r` (none at a tie: `soft_router_tie`);
* **plain ← attention-softcap host, `c → ∞`**: per layer `≤ (e^{2R³/(3c²)} − 1) M
  ≤ (4/3) R³ M / c²` (`softcap_bound`, `attn_perturb`).

Each is lifted to a whole-model bound, an identical-generation threshold and a limit through
`LimitKit.lean`, on the region of states where the logits are bounded (resp. the router has its
margin). The region condition is a hypothesis along the target's trajectory (`hK`).
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

section core

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Mixing in extra mass `Z₂` with numerator `N₂` (`‖N₂‖ ≤ Z₂ M`) moves a normalised average
`A = Z₁⁻¹ N₁` (`‖A‖ ≤ M`) by at most `2 M Z₂ / Z₁`. -/
lemma mix_bound (Z₁ Z₂ : ℝ) (N₁ N₂ : V) (M : ℝ) (hZ₁ : 0 < Z₁) (hZ₂ : 0 ≤ Z₂)
    (hN₂ : ‖N₂‖ ≤ Z₂ * M) (hA : ‖Z₁⁻¹ • N₁‖ ≤ M) :
    ‖(Z₁ + Z₂)⁻¹ • (N₁ + N₂) - Z₁⁻¹ • N₁‖ ≤ 2 * M * (Z₂ / Z₁) := by
  set A := Z₁⁻¹ • N₁
  have hM : 0 ≤ M := (norm_nonneg _).trans hA
  have hN1 : N₁ = Z₁ • A := by simp only [A, smul_smul, mul_inv_cancel₀ hZ₁.ne', one_smul]
  have hc : (Z₁ + Z₂)⁻¹ * Z₁ = 1 - (Z₁ + Z₂)⁻¹ * Z₂ := by field_simp; ring
  clear_value A
  subst hN1
  have key : (Z₁ + Z₂)⁻¹ • (Z₁ • A + N₂) - A = (Z₁ + Z₂)⁻¹ • (N₂ - Z₂ • A) := by
    rw [smul_add, smul_smul, hc, sub_smul, one_smul, smul_sub, smul_smul]; abel
  rw [key, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity)]
  have h2 : ‖N₂ - Z₂ • A‖ ≤ 2 * Z₂ * M := by
    refine (norm_sub_le _ _).trans ?_
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hZ₂]
    nlinarith
  have hinv : (Z₁ + Z₂)⁻¹ ≤ Z₁⁻¹ := inv_anti₀ hZ₁ (by linarith)
  calc (Z₁ + Z₂)⁻¹ * ‖N₂ - Z₂ • A‖ ≤ Z₁⁻¹ * (2 * Z₂ * M) :=
        mul_le_mul hinv h2 (norm_nonneg _) (inv_nonneg.mpr hZ₁.le)
    _ = 2 * M * (Z₂ / Z₁) := by ring

/-- Attention output is a convex combination: bounded by the value bound. -/
lemma attn_norm_le {ι : Type*} [Fintype ι] (S : Finset ι) (hS : S.Nonempty) (s : ι → ℝ)
    (v : ι → V) (M : ℝ) (hv : ∀ j ∈ S, ‖v j‖ ≤ M) : ‖attn S s v‖ ≤ M := by
  rw [attn_eq_weights]
  have hZ := sum_exp_pos hS s
  refine (norm_sum_le _ _).trans ?_
  calc ∑ j ∈ S, ‖(exp (s j) / ∑ l ∈ S, exp (s l)) • v j‖
      ≤ ∑ j ∈ S, (exp (s j) / ∑ l ∈ S, exp (s l)) * M := by
        refine Finset.sum_le_sum (fun j hj => ?_)
        rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity)]
        exact mul_le_mul_of_nonneg_left (hv j hj) (by positivity)
    _ = M := by rw [← Finset.sum_mul, ← Finset.sum_div, div_self hZ.ne', one_mul]

/-- **Attention with extra keys.** `S' ⊆ S`, `S'` nonempty, the host logits `t` equal the target
logits `s` on `S'`, values bounded by `M` on `S`: the host's attention over `S` is within
`2 M (Σ_{S∖S'} e^{t}) / (Σ_{S'} e^{s})` of the target's attention over `S'`. -/
theorem attn_extra_keys_bound {ι : Type*} [Fintype ι] [DecidableEq ι] (S S' : Finset ι)
    (hsub : S' ⊆ S) (hS' : S'.Nonempty) (s t : ι → ℝ) (hst : ∀ j ∈ S', t j = s j) (v : ι → V)
    (M : ℝ) (hv : ∀ j ∈ S, ‖v j‖ ≤ M) :
    ‖attn S t v - attn S' s v‖ ≤
      2 * M * ((∑ j ∈ S \ S', exp (t j)) / ∑ j ∈ S', exp (s j)) := by
  have hZ := (sum_exp_pos hS' s)
  have e1 : ∑ j ∈ S, exp (t j) = ∑ j ∈ S', exp (s j) + ∑ j ∈ S \ S', exp (t j) := by
    rw [← Finset.sum_sdiff hsub, add_comm]
    congr 1; exact Finset.sum_congr rfl (fun j hj => by rw [hst j hj])
  have e2 : ∑ j ∈ S, exp (t j) • v j =
      ∑ j ∈ S', exp (s j) • v j + ∑ j ∈ S \ S', exp (t j) • v j := by
    rw [← Finset.sum_sdiff hsub, add_comm]
    congr 1; exact Finset.sum_congr rfl (fun j hj => by rw [hst j hj])
  have hA : attn S' s v = (∑ j ∈ S', exp (s j))⁻¹ • ∑ j ∈ S', exp (s j) • v j := rfl
  rw [show attn S t v = (∑ j ∈ S, exp (t j))⁻¹ • ∑ j ∈ S, exp (t j) • v j from rfl, e1, e2, hA]
  refine mix_bound _ _ _ _ M hZ (Finset.sum_nonneg (fun j _ => (exp_pos _).le)) ?_ ?_
  · refine (norm_sum_le _ _).trans ?_
    rw [Finset.sum_mul]
    refine Finset.sum_le_sum (fun j hj => ?_)
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (exp_pos _)]
    exact mul_le_mul_of_nonneg_left (hv j (Finset.mem_sdiff.mp hj).1) (exp_pos _).le
  · rw [← hA]; exact attn_norm_le S' hS' s v M (fun j hj => hv j (hsub hj))

/-- Residual layers that differ by at most `C` at every position are within `C`. -/
lemma resid_dist_le {T : ℕ} (x a b : Fin T → V) (C : ℝ) (hC : 0 ≤ C)
    (h : ∀ t, ‖a t - b t‖ ≤ C) : dist (fun t => x t + a t) (fun t => x t + b t) ≤ C :=
  (dist_pi_le_iff hC).mpr (fun t => by rw [dist_eq_norm, add_sub_add_left_eq_sub]; exact h t)

end core

/-! ## Attention layers over a context of `T` positions -/

section layers

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {T : ℕ}

/-- An attention layer (one head, output projection folded into `val`): query `t` attends over
`S t` with logits `sc x t j` (any function of the state: content, RoPE, ALiBi, …). -/
def attnLayer (S : Fin T → Finset (Fin T)) (sc : (Fin T → V) → Fin T → Fin T → ℝ) (val : V → V)
    (x : Fin T → V) : Fin T → V :=
  fun t => x t + attn (S t) (sc x t) (fun j => val (x j))

/-- Its sink host: one extra key with logit `b` and value `0`. -/
def sinkLayer (S : Fin T → Finset (Fin T)) (sc : (Fin T → V) → Fin T → Fin T → ℝ) (val : V → V)
    (b : ℝ) (x : Fin T → V) : Fin T → V :=
  fun t => x t + (exp b + ∑ j ∈ S t, exp (sc x t j))⁻¹ • ∑ j ∈ S t, exp (sc x t j) • val (x j)

/-- The additive-mask host of a hard mask `S`: full attention with `−B` on keys outside `S t`. -/
def maskHostLogits (S : Fin T → Finset (Fin T)) (sc : (Fin T → V) → Fin T → Fin T → ℝ) (B : ℝ) :
    (Fin T → V) → Fin T → Fin T → ℝ :=
  fun x t j => sc x t j + B * (if j ∈ S t then 0 else -1)

/-- The softcap host's logits. -/
def capLogits (sc : (Fin T → V) → Fin T → Fin T → ℝ) (c : ℝ) :
    (Fin T → V) → Fin T → Fin T → ℝ :=
  fun x t j => softcap c (sc x t j)

/-- The states whose logits are all bounded by `R`. -/
def logitBox (sc : (Fin T → V) → Fin T → Fin T → ℝ) (R : ℝ) : Set (Fin T → V) :=
  {x | ∀ t j, |sc x t j| ≤ R}

/-- **Hard mask ← additive mask, one layer.** With logits bounded by `R` and values by `M`, the
full host with `−B` outside the mask is within `2 M T e^{2R − B}` of the masked target. -/
theorem mask_layer_bound (S : Fin T → Finset (Fin T)) (hS : ∀ t, (S t).Nonempty)
    (sc : (Fin T → V) → Fin T → Fin T → ℝ) (val : V → V) (M R B : ℝ)
    (hM : ∀ y, ‖val y‖ ≤ M) (x : Fin T → V) (hx : x ∈ logitBox sc R) :
    dist (attnLayer (fun _ => univ) (maskHostLogits S sc B) val x) (attnLayer S sc val x) ≤
      2 * M * (T * exp (2 * R - B)) := by
  have hM0 : 0 ≤ M := (norm_nonneg _).trans (hM 0)
  refine resid_dist_le _ _ _ _ (by positivity) (fun t => ?_)
  refine (attn_extra_keys_bound univ (S t) (subset_univ _) (hS t) (sc x t) _
    (fun j hj => by simp [maskHostLogits, hj]) _ M (fun j _ => hM _)).trans ?_
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  obtain ⟨j₀, hj₀⟩ := hS t
  have hZ1 : exp (-R) ≤ ∑ j ∈ S t, exp (sc x t j) :=
    (exp_le_exp.mpr (by linarith [(abs_le.mp (hx t j₀)).1])).trans
      (Finset.single_le_sum (fun j _ => (exp_pos _).le) hj₀)
  have hZ2 : ∑ j ∈ univ \ S t, exp (maskHostLogits S sc B x t j) ≤ T * exp (R - B) := by
    calc ∑ j ∈ univ \ S t, exp (maskHostLogits S sc B x t j)
        ≤ ∑ _j ∈ (univ : Finset (Fin T)), exp (R - B) := by
          refine (Finset.sum_le_sum (fun j hj => ?_)).trans
            (Finset.sum_le_sum_of_subset_of_nonneg sdiff_subset (fun _ _ _ => (exp_pos _).le))
          have hj : j ∉ S t := (Finset.mem_sdiff.mp hj).2
          simp only [maskHostLogits, hj, ite_false]
          exact exp_le_exp.mpr (by linarith [(abs_le.mp (hx t j)).2])
      _ = T * exp (R - B) := by simp
  rw [div_le_iff₀ (by positivity : (0 : ℝ) < ∑ j ∈ S t, exp (sc x t j))]
  calc ∑ j ∈ univ \ S t, exp (maskHostLogits S sc B x t j) ≤ T * exp (R - B) := hZ2
    _ = T * exp (2 * R - B) * exp (-R) := by rw [mul_assoc, ← exp_add]; ring_nf
    _ ≤ T * exp (2 * R - B) * ∑ j ∈ S t, exp (sc x t j) :=
        mul_le_mul_of_nonneg_left hZ1 (by positivity)

/-- **Plain ← sink host, one layer.** With logits `≥ −R` (here `|·| ≤ R`) and values bounded by
`M`, the sink host is within `2 M e^{b + R}` of the plain target. -/
theorem sink_layer_bound (S : Fin T → Finset (Fin T)) (hS : ∀ t, (S t).Nonempty)
    (sc : (Fin T → V) → Fin T → Fin T → ℝ) (val : V → V) (M R b : ℝ)
    (hM : ∀ y, ‖val y‖ ≤ M) (x : Fin T → V) (hx : x ∈ logitBox sc R) :
    dist (sinkLayer S sc val b x) (attnLayer S sc val x) ≤ 2 * M * exp (b + R) := by
  have hM0 : 0 ≤ M := (norm_nonneg _).trans (hM 0)
  refine resid_dist_le _ _ _ _ (by positivity) (fun t => ?_)
  have hZ := sum_exp_pos (hS t) (sc x t)
  have := mix_bound (∑ j ∈ S t, exp (sc x t j)) (exp b) (∑ j ∈ S t, exp (sc x t j) • val (x j))
    0 M hZ (exp_pos b).le (by simp; positivity)
    (attn_norm_le (S t) (hS t) (sc x t) (fun j => val (x j)) M (fun j _ => hM _))
  rw [add_zero, add_comm] at this
  refine this.trans (mul_le_mul_of_nonneg_left ?_ (by positivity))
  obtain ⟨j₀, hj₀⟩ := hS t
  have hZ1 : exp (-R) ≤ ∑ j ∈ S t, exp (sc x t j) :=
    (exp_le_exp.mpr (by linarith [(abs_le.mp (hx t j₀)).1])).trans
      (Finset.single_le_sum (fun j _ => (exp_pos _).le) hj₀)
  rw [div_le_iff₀ hZ]
  calc exp b = exp (b + R) * exp (-R) := by rw [← exp_add]; ring_nf
    _ ≤ exp (b + R) * ∑ j ∈ S t, exp (sc x t j) := mul_le_mul_of_nonneg_left hZ1 (exp_pos _).le

/-- **Plain ← attention-softcap host, one layer.** With logits bounded by `R` and values by `M`,
the capped host is within `(e^{2R³/(3c²)} − 1) M` of the plain target (`c > 0`). -/
theorem cap_layer_bound (S : Fin T → Finset (Fin T)) (hS : ∀ t, (S t).Nonempty)
    (sc : (Fin T → V) → Fin T → Fin T → ℝ) (val : V → V) (M R c : ℝ) (hc : 0 < c) (hR : 0 ≤ R)
    (hM : ∀ y, ‖val y‖ ≤ M) (x : Fin T → V) (hx : x ∈ logitBox sc R) :
    dist (attnLayer S (capLogits sc c) val x) (attnLayer S sc val x) ≤
      (exp (2 * (R ^ 3 / (3 * c ^ 2))) - 1) * M := by
  have hM0 : 0 ≤ M := (norm_nonneg _).trans (hM 0)
  refine resid_dist_le _ _ _ _ (mul_nonneg (by linarith [one_le_exp (by positivity :
    (0 : ℝ) ≤ 2 * (R ^ 3 / (3 * c ^ 2)))]) hM0) (fun t => ?_)
  refine attn_perturb _ (hS t) _ _ _ (fun j _ => ?_) _ M (fun j _ => hM _)
  refine (softcap_bound c hc _).trans ?_
  have h3 : |sc x t j| ^ 3 ≤ R ^ 3 := pow_le_pow_left₀ (abs_nonneg _) (hx t j) 3
  gcongr

end layers

/-! ## The soft router -/

section router

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- A dense host layer with the soft router: `A x + Σ_e softmax(β z(x))_e Y_e(x)`. -/
def softRouterLayer {E : ℕ} (β : ℝ) (A : V → V) (z : V → Fin E → ℝ) (Y : Fin E → V → V)
    (x : V) : V :=
  A x + ∑ e, softmax (fun e => β * z x e) e • Y e x

/-- The top-1 target layer: `A x + Y_{e*(x)}(x)`, `e*` the target's pick (any tie-break). -/
def top1Layer {E : ℕ} (A : V → V) (Y : Fin E → V → V) (estar : V → Fin E) (x : V) : V :=
  A x + Y (estar x) x

/-- The states where the target's pick beats every other router logit by `δr`. -/
def marginSet {E : ℕ} (z : V → Fin E → ℝ) (estar : V → Fin E) (δr : ℝ) : Set V :=
  {x | ∀ e, e ≠ estar x → z x e + δr ≤ z x (estar x)}

lemma attn_singleton {ι : Type*} [Fintype ι] (j : ι) (s : ι → ℝ) (v : ι → V) :
    attn {j} s v = v j := by
  simp [attn, smul_smul, (exp_pos (s j)).ne']

/-- **The soft router under a margin, one layer.** On states with router margin `δr`, expert
outputs bounded by `M` and `β ≥ 0`, the soft-router layer is within `2 M (E − 1) e^{−β δr}` of the
top-1 layer. (At a tie there is no such bound: `soft_router_tie`.) -/
theorem soft_router_layer_bound {E : ℕ} (β : ℝ) (hβ : 0 ≤ β) (A : V → V) (z : V → Fin E → ℝ)
    (Y : Fin E → V → V) (estar : V → Fin E) (δr M : ℝ) (hM : ∀ e y, ‖Y e y‖ ≤ M) (x : V)
    (hx : x ∈ marginSet z estar δr) :
    dist (softRouterLayer β A z Y x) (top1Layer A Y estar x) ≤
      2 * M * ((E - 1) * exp (-(β * δr))) := by
  have hM0 : 0 ≤ M := (norm_nonneg _).trans (hM (estar x) x)
  rw [dist_eq_norm, softRouterLayer, top1Layer, add_sub_add_left_eq_sub]
  set e₀ := estar x
  have h1 : ∑ e, softmax (fun e => β * z x e) e • Y e x =
      attn univ (fun e => β * z x e) (fun e => Y e x) := (attn_univ_eq_softmax _ _).symm
  rw [h1, ← attn_singleton e₀ (fun e => β * z x e) (fun e => Y e x)]
  refine (attn_extra_keys_bound univ {e₀} (subset_univ _) (singleton_nonempty _) _ _
    (fun j _ => rfl) _ M (fun j _ => hM _ _)).trans (mul_le_mul_of_nonneg_left ?_ (by positivity))
  rw [Finset.sum_singleton, div_le_iff₀ (exp_pos _)]
  calc ∑ j ∈ univ \ {e₀}, exp (β * z x j) ≤ ∑ _j ∈ univ \ {e₀}, exp (β * z x e₀ - β * δr) := by
        refine Finset.sum_le_sum (fun j hj => exp_le_exp.mpr ?_)
        have hj : j ≠ e₀ := by simpa using hj
        have := hx j hj
        nlinarith
    _ = (E - 1) * exp (-(β * δr)) * exp (β * z x e₀) := by
        rw [Finset.sum_const, Finset.card_univ_diff, Fintype.card_fin,
          card_singleton, nsmul_eq_mul, mul_assoc, ← exp_add]
        rcases Nat.eq_zero_or_pos E with hE | hE
        · subst hE; exact e₀.elim0
        · rw [Nat.cast_sub (by omega)]; push_cast; ring_nf

end router

/-! ## Whole-model statements -/

section whole

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {T : ℕ} {Tok : Type*}
  [Fintype Tok]

lemma sum_const_range (n : ℕ) (e : ℝ) : ∑ _i ∈ range n, e = n * e := by simp

/-- **Hard mask ← additive mask, the whole model.** Every layer `i` masks with `S i`; the host
uses `−B` instead. Logits bounded by `R` and values by `M̄` along the error-neighbourhood of the
target trajectory: logits within `Lh (Π L i) · n · 2 M̄ T e^{2R − B}`. -/
theorem mask_model_bound {Y : Type*} [PseudoMetricSpace Y] (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R B : ℝ) (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (head : (Fin T → V) → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ logitBox (sc n) R, ∀ b ∈ logitBox (sc n) R,
      dist (head a) (head b) ≤ Lh * dist a b) (x : Fin T → V)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i x) ≤
      (∏ j ∈ range n, L j) * (n * (2 * Mb * (T * exp (2 * R - B)))) → y ∈ logitBox (sc i) R) :
    dist (head (run (fun i => attnLayer (fun _ => univ) (maskHostLogits (S i) (sc i) B) (val i))
        n x)) (head (run (fun i => attnLayer (S i) (sc i) (val i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * (n * (2 * Mb * (T * exp (2 * R - B))))) := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  have h := model_bound_on_prod _ _ L (fun _ => 2 * Mb * (T * exp (2 * R - B))) hL1
    (fun _ => by positivity) (fun i => logitBox (sc i) R) hL
    (fun i y hy => mask_layer_bound (S i) (hS i) (sc i) (val i) Mb R B (hM i) y hy) head Lh hLh0 n
    hLh x (fun i hi y hy => hK i hi y (by rwa [sum_const_range] at hy))
  rwa [sum_const_range] at h

/-- **Hard mask ← additive mask, identical generation, explicit `B*`.** Threshold
`4 Lh (Π L i) n M̄ T e^{2R − B} < δ`, i.e. `B > B* = 2R + log(4 Lh (Π L i) n M̄ T / δ)`. -/
theorem mask_greedy (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R B : ℝ) (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (head : (Fin T → V) → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ logitBox (sc n) R, ∀ b ∈ logitBox (sc n) R,
      dist (head a) (head b) ≤ Lh * dist a b) (embed : List Tok → Fin T → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => attnLayer (S i) (sc i) (val i)) n
      (embed (greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n
        (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i
      (embed (greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n
        (embed q))) p k))) ≤
      (∏ j ∈ range n, L j) * (n * (2 * Mb * (T * exp (2 * R - B)))) → y ∈ logitBox (sc i) R)
    (hB : 4 * Lh * (∏ i ∈ range n, L i) * n * Mb * T * exp (2 * R - B) < δ) :
    ∀ k, greedy pickH (fun q => head (run (fun i => attnLayer (fun _ => univ)
        (maskHostLogits (S i) (sc i) B) (val i)) n (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n (embed q)))
        p k := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  refine greedy_of_layer_bounds_on _ _ L (fun _ => 2 * Mb * (T * exp (2 * R - B))) hL1
    (fun _ => by positivity) (fun i => logitBox (sc i) R) hL
    (fun i y hy => mask_layer_bound (S i) (hS i) (sc i) (val i) Mb R B (hM i) y hy) head Lh hLh0 n
    hLh embed hT hH p δ hm (fun k i hi y hy => hK k i hi y (by rwa [sum_const_range] at hy)) ?_
  rw [sum_const_range]; linarith

/-- **Hard mask ← additive mask, the whole-model limit** (`B → ∞`), uniformly on every input set
whose target trajectories keep an `r`-neighbourhood with logits `≤ R`. -/
theorem mask_model_limit (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R : ℝ) (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (n : ℕ) (A₀ : Set (Fin T → V)) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ A₀, ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i x) ≤ r →
      y ∈ logitBox (sc i) R) :
    TendstoUniformlyOn (fun B : ℝ => run (fun i => attnLayer (fun _ => univ)
        (maskHostLogits (S i) (sc i) B) (val i)) n)
      (run (fun i => attnLayer (S i) (sc i) (val i)) n) atTop A₀ := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  refine model_limit_on _ _ L hL1 (fun B _ => 2 * Mb * (T * exp (2 * R - B)))
    (fun _ _ => by positivity) (fun i => ?_) (fun i => logitBox (sc i) R) hL
    (fun B i y hy => mask_layer_bound (S i) (hS i) (sc i) (val i) Mb R B (hM i) y hy) n A₀ r hr hK
  have : Tendsto (fun B : ℝ => exp (2 * R - B)) atTop (𝓝 0) :=
    tendsto_exp_atBot.comp ((tendsto_atBot_add_const_left _ (2 * R) tendsto_neg_atTop_atBot).congr
      (fun B => by ring))
  simpa using (this.const_mul (T : ℝ)).const_mul (2 * Mb)

/-- **Plain ← sink host, the whole model.** Logits within `Lh (Π L i) · n · 2 M̄ e^{b + R}`. -/
theorem sink_model_bound {Y : Type*} [PseudoMetricSpace Y] (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R b : ℝ) (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (head : (Fin T → V) → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ logitBox (sc n) R, ∀ b ∈ logitBox (sc n) R,
      dist (head a) (head b) ≤ Lh * dist a b) (x : Fin T → V)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i x) ≤
      (∏ j ∈ range n, L j) * (n * (2 * Mb * exp (b + R))) → y ∈ logitBox (sc i) R) :
    dist (head (run (fun i => sinkLayer (S i) (sc i) (val i) b) n x))
        (head (run (fun i => attnLayer (S i) (sc i) (val i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * (n * (2 * Mb * exp (b + R)))) := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  have h := model_bound_on_prod _ _ L (fun _ => 2 * Mb * exp (b + R)) hL1
    (fun _ => by positivity) (fun i => logitBox (sc i) R) hL
    (fun i y hy => sink_layer_bound (S i) (hS i) (sc i) (val i) Mb R b (hM i) y hy) head Lh hLh0 n
    hLh x (fun i hi y hy => hK i hi y (by rwa [sum_const_range] at hy))
  rwa [sum_const_range] at h

/-- **Plain ← sink host, identical generation, explicit `b*`.** Threshold
`4 Lh (Π L i) n M̄ e^{b + R} < δ`, i.e. `b < b* = log(δ / (4 Lh (Π L i) n M̄)) − R`. -/
theorem sink_greedy (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R b : ℝ) (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (head : (Fin T → V) → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ logitBox (sc n) R, ∀ b ∈ logitBox (sc n) R,
      dist (head a) (head b) ≤ Lh * dist a b) (embed : List Tok → Fin T → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => attnLayer (S i) (sc i) (val i)) n
      (embed (greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n
        (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i
      (embed (greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n
        (embed q))) p k))) ≤
      (∏ j ∈ range n, L j) * (n * (2 * Mb * exp (b + R))) → y ∈ logitBox (sc i) R)
    (hb : 4 * Lh * (∏ i ∈ range n, L i) * n * Mb * exp (b + R) < δ) :
    ∀ k, greedy pickH (fun q => head (run (fun i => sinkLayer (S i) (sc i) (val i) b) n
        (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n (embed q)))
        p k := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  refine greedy_of_layer_bounds_on _ _ L (fun _ => 2 * Mb * exp (b + R)) hL1
    (fun _ => by positivity) (fun i => logitBox (sc i) R) hL
    (fun i y hy => sink_layer_bound (S i) (hS i) (sc i) (val i) Mb R b (hM i) y hy) head Lh hLh0 n
    hLh embed hT hH p δ hm (fun k i hi y hy => hK k i hi y (by rwa [sum_const_range] at hy)) ?_
  rw [sum_const_range]; linarith

/-- **Plain ← sink host, the whole-model limit** (`b → −∞`). -/
theorem sink_model_limit (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R : ℝ) (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (n : ℕ) (A₀ : Set (Fin T → V)) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ A₀, ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i x) ≤ r →
      y ∈ logitBox (sc i) R) :
    TendstoUniformlyOn (fun b : ℝ => run (fun i => sinkLayer (S i) (sc i) (val i) b) n)
      (run (fun i => attnLayer (S i) (sc i) (val i)) n) atBot A₀ := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  refine model_limit_on _ _ L hL1 (fun b _ => 2 * Mb * exp (b + R))
    (fun _ _ => by positivity) (fun i => ?_) (fun i => logitBox (sc i) R) hL
    (fun b i y hy => sink_layer_bound (S i) (hS i) (sc i) (val i) Mb R b (hM i) y hy) n A₀ r hr hK
  have : Tendsto (fun b : ℝ => exp (b + R)) atBot (𝓝 0) :=
    tendsto_exp_atBot.comp (tendsto_atBot_add_const_right _ _ tendsto_id)
  simpa using this.const_mul (2 * Mb)

/-- `e^{2η} − 1 ≤ 4η` turned into the softcap per-layer constant. -/
lemma cap_const_le (R c Mb : ℝ) (hc : 0 < c) (hR : 0 ≤ R) (hMb : 0 ≤ Mb)
    (hsmall : 2 * R ^ 3 ≤ 3 * c ^ 2) :
    (exp (2 * (R ^ 3 / (3 * c ^ 2))) - 1) * Mb ≤ 4 / 3 * R ^ 3 * Mb / c ^ 2 := by
  have hc2 : 0 < c ^ 2 := by positivity
  have h0 : 0 ≤ R ^ 3 / (3 * c ^ 2) := by positivity
  have h1 : 2 * (R ^ 3 / (3 * c ^ 2)) ≤ 1 := by
    rw [← mul_div_assoc, div_le_one (by positivity)]; linarith
  have := exp_two_mul_sub_one_le _ h0 h1
  calc (exp (2 * (R ^ 3 / (3 * c ^ 2))) - 1) * Mb ≤ 4 * (R ^ 3 / (3 * c ^ 2)) * Mb :=
        mul_le_mul_of_nonneg_right this hMb
    _ = 4 / 3 * R ^ 3 * Mb / c ^ 2 := by field_simp

/-- **Plain ← attention-softcap host, the whole model.** For `2R³ ≤ 3c²`, logits within
`Lh (Π L i) · n · (4/3) R³ M̄ / c²`: `O(1/c²)`. -/
theorem cap_model_bound {Y : Type*} [PseudoMetricSpace Y] (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R c : ℝ) (hc : 0 < c) (hR : 0 ≤ R) (hsmall : 2 * R ^ 3 ≤ 3 * c ^ 2)
    (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (head : (Fin T → V) → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ logitBox (sc n) R, ∀ b ∈ logitBox (sc n) R,
      dist (head a) (head b) ≤ Lh * dist a b) (x : Fin T → V)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i x) ≤
      (∏ j ∈ range n, L j) * (n * (4 / 3 * R ^ 3 * Mb / c ^ 2)) → y ∈ logitBox (sc i) R) :
    dist (head (run (fun i => attnLayer (S i) (capLogits (sc i) c) (val i)) n x))
        (head (run (fun i => attnLayer (S i) (sc i) (val i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * (n * (4 / 3 * R ^ 3 * Mb / c ^ 2))) := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  have h := model_bound_on_prod _ _ L (fun _ => 4 / 3 * R ^ 3 * Mb / c ^ 2) hL1
    (fun _ => by positivity) (fun i => logitBox (sc i) R) hL
    (fun i y hy => (cap_layer_bound (S i) (hS i) (sc i) (val i) Mb R c hc hR (hM i) y hy).trans
      (cap_const_le R c Mb hc hR hM0 hsmall)) head Lh hLh0 n
    hLh x (fun i hi y hy => hK i hi y (by rwa [sum_const_range] at hy))
  rwa [sum_const_range] at h

/-- **Plain ← attention-softcap host, identical generation, explicit `c*`.** For `2R³ ≤ 3c²` and
`(8/3) Lh (Π L i) n R³ M̄ < δ c²`, i.e. `c > c* = √(8 Lh (Π L i) n R³ M̄ / (3δ))`. -/
theorem cap_greedy (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R c : ℝ) (hc : 0 < c) (hR : 0 ≤ R) (hsmall : 2 * R ^ 3 ≤ 3 * c ^ 2)
    (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (head : (Fin T → V) → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ logitBox (sc n) R, ∀ b ∈ logitBox (sc n) R,
      dist (head a) (head b) ≤ Lh * dist a b) (embed : List Tok → Fin T → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => attnLayer (S i) (sc i) (val i)) n
      (embed (greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n
        (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i
      (embed (greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n
        (embed q))) p k))) ≤
      (∏ j ∈ range n, L j) * (n * (4 / 3 * R ^ 3 * Mb / c ^ 2)) → y ∈ logitBox (sc i) R)
    (hc' : 8 / 3 * Lh * (∏ i ∈ range n, L i) * n * R ^ 3 * Mb < δ * c ^ 2) :
    ∀ k, greedy pickH (fun q => head (run (fun i => attnLayer (S i) (capLogits (sc i) c) (val i))
        n (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => attnLayer (S i) (sc i) (val i)) n (embed q)))
        p k := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  refine greedy_of_layer_bounds_on _ _ L (fun _ => 4 / 3 * R ^ 3 * Mb / c ^ 2) hL1
    (fun _ => by positivity) (fun i => logitBox (sc i) R) hL
    (fun i y hy => (cap_layer_bound (S i) (hS i) (sc i) (val i) Mb R c hc hR (hM i) y hy).trans
      (cap_const_le R c Mb hc hR hM0 hsmall)) head Lh hLh0 n
    hLh embed hT hH p δ hm (fun k i hi y hy => hK k i hi y (by rwa [sum_const_range] at hy)) ?_
  rw [sum_const_range]
  have hc2 : 0 < c ^ 2 := by positivity
  have : 2 * (Lh * ((∏ i ∈ range n, L i) * (n * (4 / 3 * R ^ 3 * Mb / c ^ 2)))) =
      (8 / 3 * Lh * (∏ i ∈ range n, L i) * n * R ^ 3 * Mb) / c ^ 2 := by ring
  rw [this, div_lt_iff₀ hc2]; exact hc'

/-- **Plain ← attention-softcap host, the whole-model limit** (`c → ∞`; the host uses
`max c 1`). -/
theorem cap_model_limit (S : ℕ → Fin T → Finset (Fin T))
    (hS : ∀ i t, (S i t).Nonempty) (sc : ℕ → (Fin T → V) → Fin T → Fin T → ℝ)
    (val : ℕ → V → V) (Mb R : ℝ) (hR : 0 ≤ R) (hM : ∀ i y, ‖val i y‖ ≤ Mb) (L : ℕ → ℝ)
    (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ logitBox (sc i) R, ∀ b ∈ logitBox (sc i) R,
      dist (attnLayer (S i) (sc i) (val i) a) (attnLayer (S i) (sc i) (val i) b) ≤ L i * dist a b)
    (n : ℕ) (A₀ : Set (Fin T → V)) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ A₀, ∀ i ≤ n, ∀ y, dist y (run (fun i => attnLayer (S i) (sc i) (val i)) i x) ≤ r →
      y ∈ logitBox (sc i) R) :
    TendstoUniformlyOn (fun c : ℝ => run (fun i => attnLayer (S i) (capLogits (sc i) (max c 1))
        (val i)) n)
      (run (fun i => attnLayer (S i) (sc i) (val i)) n) atTop A₀ := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 0)
  refine model_limit_on _ _ L hL1
    (fun c _ => (exp (2 * (R ^ 3 / (3 * (max c 1) ^ 2))) - 1) * Mb)
    (fun c _ => mul_nonneg (by linarith [one_le_exp (by positivity :
      (0 : ℝ) ≤ 2 * (R ^ 3 / (3 * (max c 1) ^ 2)))]) hM0) (fun i => ?_)
    (fun i => logitBox (sc i) R) hL
    (fun c i y hy => cap_layer_bound (S i) (hS i) (sc i) (val i) Mb R (max c 1)
      (by positivity) hR (hM i) y hy) n A₀ r hr hK
  have h1 : Tendsto (fun c : ℝ => max c 1) atTop atTop :=
    tendsto_atTop_mono (fun c => le_max_left c 1) tendsto_id
  have h2 : Tendsto (fun c : ℝ => R ^ 3 / (3 * (max c 1) ^ 2)) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop (Tendsto.const_mul_atTop (by norm_num)
      (tendsto_pow_atTop (by norm_num) |>.comp h1))
  have h3 : Tendsto (fun c : ℝ => (exp (2 * (R ^ 3 / (3 * (max c 1) ^ 2))) - 1) * Mb) atTop
      (𝓝 ((exp (2 * 0) - 1) * Mb)) :=
    (((continuous_exp.tendsto _).comp (h2.const_mul 2)).sub_const 1).mul_const Mb
  simpa using h3

/-- **The soft router under a margin, the whole model.** Every layer's router has margin `δr`
along the error-neighbourhood of the target trajectory (`hK`), `β ≥ 0`: logits within
`Lh (Π L i) · n · 2 M̄ (E − 1) e^{−β δr}`. -/
theorem soft_router_model_bound {Y : Type*} [PseudoMetricSpace Y] {E : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (A : ℕ → V → V) (z : ℕ → V → Fin E → ℝ) (Yx : ℕ → Fin E → V → V) (estar : ℕ → V → Fin E)
    (δr Mb : ℝ) (hM : ∀ i e y, ‖Yx i e y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ marginSet (z i) (estar i) δr, ∀ b ∈ marginSet (z i) (estar i) δr,
      dist (top1Layer (A i) (Yx i) (estar i) a) (top1Layer (A i) (Yx i) (estar i) b) ≤
        L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ marginSet (z n) (estar n) δr, ∀ b ∈ marginSet (z n) (estar n) δr,
      dist (head a) (head b) ≤ Lh * dist a b) (x : V) (hE : 1 ≤ E)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (fun i => top1Layer (A i) (Yx i) (estar i)) i x) ≤
      (∏ j ∈ range n, L j) * (n * (2 * Mb * ((E - 1) * exp (-(β * δr))))) →
        y ∈ marginSet (z i) (estar i) δr) :
    dist (head (run (fun i => softRouterLayer β (A i) (z i) (Yx i)) n x))
        (head (run (fun i => top1Layer (A i) (Yx i) (estar i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * (n * (2 * Mb * ((E - 1) * exp (-(β * δr)))))) := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 ⟨0, by omega⟩ x)
  have hE' : (0 : ℝ) ≤ E - 1 := by have : (1 : ℝ) ≤ E := by exact_mod_cast hE
                                   linarith
  have h := model_bound_on_prod _ _ L (fun _ => 2 * Mb * ((E - 1) * exp (-(β * δr)))) hL1
    (fun _ => by positivity) (fun i => marginSet (z i) (estar i) δr) hL
    (fun i y hy => soft_router_layer_bound β hβ (A i) (z i) (Yx i) (estar i) δr Mb (hM i) y hy)
    head Lh hLh0 n hLh x (fun i hi y hy => hK i hi y (by rwa [sum_const_range] at hy))
  rwa [sum_const_range] at h

/-- **The soft router, identical generation, explicit `β*`.** Threshold
`4 Lh (Π L i) n M̄ (E − 1) e^{−β δr} < δ`, i.e.
`β > β* = log(4 Lh (Π L i) n M̄ (E − 1)/δ)/δr`. -/
theorem soft_router_greedy {E : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (A : ℕ → V → V) (z : ℕ → V → Fin E → ℝ) (Yx : ℕ → Fin E → V → V) (estar : ℕ → V → Fin E)
    (δr Mb : ℝ) (hM : ∀ i e y, ‖Yx i e y‖ ≤ Mb) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ marginSet (z i) (estar i) δr, ∀ b ∈ marginSet (z i) (estar i) δr,
      dist (top1Layer (A i) (Yx i) (estar i) a) (top1Layer (A i) (Yx i) (estar i) b) ≤
        L i * dist a b)
    (head : V → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ marginSet (z n) (estar n) δr, ∀ b ∈ marginSet (z n) (estar n) δr,
      dist (head a) (head b) ≤ Lh * dist a b) (embed : List Tok → V) (hE : 1 ≤ E)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => top1Layer (A i) (Yx i) (estar i)) n
      (embed (greedy pickT (fun q => head (run (fun i => top1Layer (A i) (Yx i) (estar i)) n
        (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run (fun i => top1Layer (A i) (Yx i) (estar i)) i
      (embed (greedy pickT (fun q => head (run (fun i => top1Layer (A i) (Yx i) (estar i)) n
        (embed q))) p k))) ≤
      (∏ j ∈ range n, L j) * (n * (2 * Mb * ((E - 1) * exp (-(β * δr))))) →
        y ∈ marginSet (z i) (estar i) δr)
    (hβ' : 4 * Lh * (∏ i ∈ range n, L i) * n * Mb * (E - 1) * exp (-(β * δr)) < δ) :
    ∀ k, greedy pickH (fun q => head (run (fun i => softRouterLayer β (A i) (z i) (Yx i)) n
        (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => top1Layer (A i) (Yx i) (estar i)) n (embed q)))
        p k := by
  have hM0 : 0 ≤ Mb := (norm_nonneg _).trans (hM 0 ⟨0, by omega⟩ (embed p))
  have hE' : (0 : ℝ) ≤ E - 1 := by have : (1 : ℝ) ≤ E := by exact_mod_cast hE
                                   linarith
  refine greedy_of_layer_bounds_on _ _ L (fun _ => 2 * Mb * ((E - 1) * exp (-(β * δr)))) hL1
    (fun _ => by positivity) (fun i => marginSet (z i) (estar i) δr) hL
    (fun i y hy => soft_router_layer_bound β hβ (A i) (z i) (Yx i) (estar i) δr Mb (hM i) y hy)
    head Lh hLh0 n hLh embed hT hH p δ hm
    (fun k i hi y hy => hK k i hi y (by rwa [sum_const_range] at hy)) ?_
  rw [sum_const_range]; linarith

/-- **The soft router, the whole-model limit** (`β → ∞`; the host uses `max β 0`), uniformly on
every input set whose target trajectories keep an `r`-neighbourhood with router margin `δr > 0`.
-/
theorem soft_router_model_limit {E : ℕ} (A : ℕ → V → V) (z : ℕ → V → Fin E → ℝ)
    (Yx : ℕ → Fin E → V → V) (estar : ℕ → V → Fin E) (δr Mb : ℝ) (hδr : 0 < δr)
    (hM : ∀ i e y, ‖Yx i e y‖ ≤ Mb) (hMb : 0 ≤ Mb) (hE : 1 ≤ E) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ marginSet (z i) (estar i) δr, ∀ b ∈ marginSet (z i) (estar i) δr,
      dist (top1Layer (A i) (Yx i) (estar i) a) (top1Layer (A i) (Yx i) (estar i) b) ≤
        L i * dist a b)
    (n : ℕ) (A₀ : Set V) (r : ℝ) (hr : 0 < r)
    (hK : ∀ x ∈ A₀, ∀ i ≤ n, ∀ y, dist y (run (fun i => top1Layer (A i) (Yx i) (estar i)) i x) ≤
      r → y ∈ marginSet (z i) (estar i) δr) :
    TendstoUniformlyOn (fun β : ℝ => run (fun i => softRouterLayer (max β 0) (A i) (z i) (Yx i)) n)
      (run (fun i => top1Layer (A i) (Yx i) (estar i)) n) atTop A₀ := by
  have hE' : (0 : ℝ) ≤ E - 1 := by have : (1 : ℝ) ≤ E := by exact_mod_cast hE
                                   linarith
  refine model_limit_on _ _ L hL1 (fun β _ => 2 * Mb * ((E - 1) * exp (-(max β 0 * δr))))
    (fun _ _ => by positivity) (fun i => ?_) (fun i => marginSet (z i) (estar i) δr) hL
    (fun β i y hy => soft_router_layer_bound (max β 0) (le_max_right _ _) (A i) (z i) (Yx i)
      (estar i) δr Mb (hM i) y hy) n A₀ r hr hK
  have h1 : Tendsto (fun β : ℝ => max β 0 * δr) atTop atTop :=
    (tendsto_atTop_mono (fun c => le_max_left c 0) tendsto_id).atTop_mul_const hδr
  have h2 : Tendsto (fun β : ℝ => exp (-(max β 0 * δr))) atTop (𝓝 0) :=
    tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp h1)
  simpa using (h2.const_mul ((E : ℝ) - 1)).const_mul (2 * Mb)

end whole

/-! ## Relative-scale reads: the vanishing carry and the 1-D reciprocal compensator -/

section scaled

lemma one_sub_ratio_sqrt (R u : ℝ) (hR : 0 < R) (hu : 0 ≤ u) :
    0 ≤ 1 - R / √(R ^ 2 + u) ∧ 1 - R / √(R ^ 2 + u) ≤ u / (2 * R ^ 2) := by
  have hD : 0 < √(R ^ 2 + u) := Real.sqrt_pos.mpr (by positivity)
  have hRD : R ≤ √(R ^ 2 + u) := Real.le_sqrt_of_sq_le (by linarith)
  have hsq : √(R ^ 2 + u) ^ 2 = R ^ 2 + u := Real.sq_sqrt (by positivity)
  set a := R / √(R ^ 2 + u)
  have ha0 : 0 < a := div_pos hR hD
  have ha1 : a ≤ 1 := (div_le_one hD).mpr hRD
  refine ⟨by linarith, ?_⟩
  set v := u / R ^ 2
  have hv : 0 ≤ v := by positivity
  have hva : a ^ 2 * (1 + v) = 1 := by
    simp only [a, v, div_pow, hsq]; field_simp
  have : u / (2 * R ^ 2) = v / 2 := by simp only [v]; field_simp
  rw [this]
  rcases le_or_gt 2 v with h2 | h2
  · linarith
  · -- (1 - v/2)^2 (1 + v) ≤ 1 = a^2 (1+v)
    have h1 : (1 - v / 2) ^ 2 ≤ a ^ 2 := by
      have : (1 - v / 2) ^ 2 * (1 + v) ≤ a ^ 2 * (1 + v) := by
        rw [hva]; nlinarith [sq_nonneg v]
      exact le_of_mul_le_mul_right this (by linarith)
    have h3 := (sq_le_sq₀ (by linarith) ha0.le).mp h1
    linarith


/-- **The vanishing-scale carry, quantitative** (`vanishing_carry_limit` with a rate). Host
residual `[x; δ s]`, host RMSNorm as in `vanishing_carry_limit`. Both the stream read and the
carried read ×`1/δ` are within relative error `δ²‖s‖²/(2dε)` of their limits: `O(δ²)`. (In the
proof each equals its limit times the same per-token factor `λ = R/√(R² + δ²‖s‖²/(d+e))`, with
`0 ≤ 1 − λ ≤ δ²‖s‖²/(2dε)`, which is the `scaledReadLayer` form below.) -/
theorem vanishing_carry_rate {d e : ℕ} (hd : d ≠ 0) (eps : ℝ) (heps : 0 < eps)
    (w x : Fin d → ℝ) (s wz : Fin e → ℝ) (δ : ℝ) (hδ : δ ≠ 0) :
    let W : Fin (d + e) → ℝ := Fin.append (fun i => √((d : ℝ) / (d + e : ℕ)) * w i) wz
    let R : ℝ := √((d : ℝ) / (d + e : ℕ)) * √((∑ l, x l ^ 2) / d + eps)
    (∀ i, |rmsNorm (d * eps / (d + e : ℕ)) W (Fin.append x (δ • s)) (Fin.castAdd e i) -
        rmsNorm eps w x i| ≤ |rmsNorm eps w x i| * (δ ^ 2 * (∑ j, s j ^ 2) / (2 * d * eps))) ∧
    (∀ j, |δ⁻¹ * rmsNorm (d * eps / (d + e : ℕ)) W (Fin.append x (δ • s)) (Fin.natAdd d j) -
        wz j * s j / R| ≤ |wz j * s j / R| * (δ ^ 2 * (∑ j, s j ^ 2) / (2 * d * eps))) := by
  intro W R
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hde : (0 : ℝ) < (d + e : ℕ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hd).trans_le (by omega)
  set Sx := ∑ l, x l ^ 2
  set Ss := ∑ j, s j ^ 2
  have hSx : 0 ≤ Sx := Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  have hSs : 0 ≤ Ss := Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  have hA : 0 < Sx / d + eps := by positivity
  have hRpos : 0 < R := mul_pos (Real.sqrt_pos.mpr (by positivity)) (Real.sqrt_pos.mpr hA)
  set u := δ ^ 2 * Ss / (d + e : ℕ)
  have hu : 0 ≤ u := by positivity
  have hR2 : R ^ 2 = (Sx + d * eps) / (d + e : ℕ) := by
    have hA' : 0 ≤ (∑ l, x l ^ 2) / d + eps := hA.le
    simp only [R, mul_pow, Real.sq_sqrt (by positivity : (0 : ℝ) ≤ (d : ℝ) / (d + e : ℕ)),
      Real.sq_sqrt hA']
    field_simp
    rfl
  have hden : (∑ l : Fin (d + e), (Fin.append x (δ • s) l) ^ 2) / (d + e : ℕ) +
      d * eps / (d + e : ℕ) = R ^ 2 + u := by
    rw [hR2, Fin.sum_univ_add]
    simp only [Fin.append_left, Fin.append_right, Pi.smul_apply, smul_eq_mul, mul_pow,
      ← Finset.mul_sum, u, Sx, Ss]
    field_simp; ring
  obtain ⟨h0, h1⟩ := one_sub_ratio_sqrt R u hRpos hu
  have hrel : u / (2 * R ^ 2) ≤ δ ^ 2 * Ss / (2 * d * eps) := by
    rw [hR2]
    simp only [u]
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have : 0 ≤ δ ^ 2 * Ss := by positivity
    field_simp
    nlinarith [mul_nonneg this hSx]
  have hD : 0 < √(R ^ 2 + u) := Real.sqrt_pos.mpr (by positivity)
  have key : ∀ t : ℝ, |t * (R / √(R ^ 2 + u)) - t| ≤ |t| * (δ ^ 2 * Ss / (2 * d * eps)) := by
    intro t
    rw [show t * (R / √(R ^ 2 + u)) - t = -(t * (1 - R / √(R ^ 2 + u))) by ring, abs_neg,
      abs_mul, abs_of_nonneg h0]
    exact mul_le_mul_of_nonneg_left (h1.trans hrel) (abs_nonneg _)
  refine ⟨fun i => ?_, fun j => ?_⟩
  · have e1 : rmsNorm (d * eps / (d + e : ℕ)) W (Fin.append x (δ • s)) (Fin.castAdd e i) =
        rmsNorm eps w x i * (R / √(R ^ 2 + u)) := by
      simp only [rmsNorm, W, Fin.append_left]
      rw [hden]
      simp only [R]
      field_simp
    rw [e1]; exact key _
  · have e1 : δ⁻¹ * rmsNorm (d * eps / (d + e : ℕ)) W (Fin.append x (δ • s)) (Fin.natAdd d j) =
        wz j * s j / R * (R / √(R ^ 2 + u)) := by
      simp only [rmsNorm, W, Fin.append_right, Pi.smul_apply, smul_eq_mul]
      rw [hden]
      field_simp
    rw [e1]; exact key _


variable {V W : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] [NormedAddCommGroup W]
  [NormedSpace ℝ W] {Tok : Type*} [Fintype Tok]

/-- A layer whose read `r x` reaches its consumer `G` scaled by a per-token factor `λ x`. The target
is `G x (r x)`. -/
def scaledReadLayer (G : V → W → V) (r : V → W) (lam : V → ℝ) (x : V) : V := G x (lam x • r x)

/-- The target layer. -/
def readLayer (G : V → W → V) (r : V → W) (x : V) : V := G x (r x)

/-- **One layer.** Consumer `K`-Lipschitz in the read, read bounded by `M`, `|λ − 1| ≤ ρ`: within
`K M ρ`. -/
theorem scaled_read_layer_bound (G : V → W → V) (r : V → W) (lam : V → ℝ) (K M ρ : ℝ) (x : V)
    (hK0 : 0 ≤ K) (hG : ∀ y y', dist (G x y) (G x y') ≤ K * ‖y - y'‖) (hr : ‖r x‖ ≤ M)
    (hlam : |lam x - 1| ≤ ρ) :
    dist (scaledReadLayer G r lam x) (readLayer G r x) ≤ K * (M * ρ) := by
  refine (hG _ _).trans (mul_le_mul_of_nonneg_left ?_ hK0)
  rw [← one_smul ℝ (r x), smul_smul, mul_one, ← sub_smul, norm_smul, Real.norm_eq_abs, mul_comm]
  exact mul_le_mul hr hlam (abs_nonneg _) ((norm_nonneg _).trans hr)

/-- **Relative-scale reads, the whole model.** On regions `Kset i` where `|λ_i − 1| ≤ ρ`, consumers
`K̄`-Lipschitz in the read, reads bounded by `M̄`: logits within `Lh (Π L i) · n · K̄ M̄ ρ`. -/
theorem scaled_read_model_bound {Y : Type*} [PseudoMetricSpace Y] (G : ℕ → V → W → V)
    (r : ℕ → V → W) (lam : ℕ → V → ℝ) (Kset : ℕ → Set V) (Kb Mb ρ : ℝ) (hK0 : 0 ≤ Kb)
    (hM0 : 0 ≤ Mb) (hρ : 0 ≤ ρ)
    (hG : ∀ i x y y', dist (G i x y) (G i x y') ≤ Kb * ‖y - y'‖) (hr : ∀ i x, ‖r i x‖ ≤ Mb)
    (hlam : ∀ i, ∀ x ∈ Kset i, |lam i x - 1| ≤ ρ) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ Kset i, ∀ b ∈ Kset i,
      dist (readLayer (G i) (r i) a) (readLayer (G i) (r i) b) ≤ L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ Kset n, ∀ b ∈ Kset n, dist (head a) (head b) ≤ Lh * dist a b) (x : V)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (fun i => readLayer (G i) (r i)) i x) ≤
      (∏ j ∈ range n, L j) * (n * (Kb * (Mb * ρ))) → y ∈ Kset i) :
    dist (head (run (fun i => scaledReadLayer (G i) (r i) (lam i)) n x))
        (head (run (fun i => readLayer (G i) (r i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * (n * (Kb * (Mb * ρ)))) := by
  have h := model_bound_on_prod _ _ L (fun _ => Kb * (Mb * ρ)) hL1 (fun _ => by positivity)
    Kset hL (fun i y hy => scaled_read_layer_bound (G i) (r i) (lam i) Kb Mb ρ y hK0 (hG i y)
      (hr i y) (hlam i y hy)) head Lh hLh0 n hLh x
    (fun i hi y hy => hK i hi y (by rwa [sum_const_range] at hy))
  rwa [sum_const_range] at h

/-- **Relative-scale reads, identical generation.** Threshold `2 Lh (Π L i) n K̄ M̄ ρ < δ`, i.e.
`ρ < ρ* = δ / (2 Lh (Π L i) n K̄ M̄)`. -/
theorem scaled_read_greedy (G : ℕ → V → W → V)
    (r : ℕ → V → W) (lam : ℕ → V → ℝ) (Kset : ℕ → Set V) (Kb Mb ρ : ℝ) (hK0 : 0 ≤ Kb)
    (hM0 : 0 ≤ Mb) (hρ : 0 ≤ ρ)
    (hG : ∀ i x y y', dist (G i x y) (G i x y') ≤ Kb * ‖y - y'‖) (hr : ∀ i x, ‖r i x‖ ≤ Mb)
    (hlam : ∀ i, ∀ x ∈ Kset i, |lam i x - 1| ≤ ρ) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ Kset i, ∀ b ∈ Kset i,
      dist (readLayer (G i) (r i) a) (readLayer (G i) (r i) b) ≤ L i * dist a b)
    (head : V → Tok → ℝ) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ Kset n, ∀ b ∈ Kset n, dist (head a) (head b) ≤ Lh * dist a b)
    (embed : List Tok → V)
    {pickT pickH : (Tok → ℝ) → Tok} (hT : IsPick pickT) (hH : IsPick pickH) (p : List Tok)
    (δ : ℝ)
    (hm : ∀ k, ∃ a, MarginAt δ (head (run (fun i => readLayer (G i) (r i)) n
      (embed (greedy pickT (fun q => head (run (fun i => readLayer (G i) (r i)) n
        (embed q))) p k)))) a)
    (hK : ∀ k, ∀ i ≤ n, ∀ y, dist y (run (fun i => readLayer (G i) (r i)) i
      (embed (greedy pickT (fun q => head (run (fun i => readLayer (G i) (r i)) n
        (embed q))) p k))) ≤ (∏ j ∈ range n, L j) * (n * (Kb * (Mb * ρ))) → y ∈ Kset i)
    (hρ' : 2 * Lh * (∏ i ∈ range n, L i) * n * Kb * Mb * ρ < δ) :
    ∀ k, greedy pickH (fun q => head (run (fun i => scaledReadLayer (G i) (r i) (lam i)) n
        (embed q))) p k =
      greedy pickT (fun q => head (run (fun i => readLayer (G i) (r i)) n (embed q))) p k := by
  refine greedy_of_layer_bounds_on _ _ L (fun _ => Kb * (Mb * ρ)) hL1 (fun _ => by positivity)
    Kset hL (fun i y hy => scaled_read_layer_bound (G i) (r i) (lam i) Kb Mb ρ y hK0 (hG i y)
      (hr i y) (hlam i y hy)) head Lh hLh0 n hLh embed hT hH p δ hm
    (fun k i hi y hy => hK k i hi y (by rwa [sum_const_range] at hy)) ?_
  rw [sum_const_range]; linarith

/-- **Relative-scale reads, the limit.** A family `λ_β` with `|λ_β − 1| ≤ ρ β → 0` on the regions:
uniform convergence on every input set whose target trajectories keep an `r₀`-neighbourhood inside
the regions. For the vanishing carry take `ρ δ = δ² S̄ /(2 d ε)` (`vanishing_carry_rate`, `S̄` a
bound on the carried vector's squared norm), `δ → 0`; for the compensator, fits with sup relative
error `ρ_k → 0`. -/
theorem scaled_read_model_limit {ι : Type*} {pf : Filter ι} (G : ℕ → V → W → V)
    (r : ℕ → V → W) (lam : ι → ℕ → V → ℝ) (Kset : ℕ → Set V) (Kb Mb : ℝ) (ρ : ι → ℝ)
    (hK0 : 0 ≤ Kb) (hM0 : 0 ≤ Mb) (hρ : ∀ β, 0 ≤ ρ β) (hρlim : Tendsto ρ pf (𝓝 0))
    (hG : ∀ i x y y', dist (G i x y) (G i x y') ≤ Kb * ‖y - y'‖) (hr : ∀ i x, ‖r i x‖ ≤ Mb)
    (hlam : ∀ β i, ∀ x ∈ Kset i, |lam β i x - 1| ≤ ρ β) (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ Kset i, ∀ b ∈ Kset i,
      dist (readLayer (G i) (r i) a) (readLayer (G i) (r i) b) ≤ L i * dist a b)
    (n : ℕ) (A₀ : Set V) (r₀ : ℝ) (hr₀ : 0 < r₀)
    (hK : ∀ x ∈ A₀, ∀ i ≤ n, ∀ y, dist y (run (fun i => readLayer (G i) (r i)) i x) ≤ r₀ →
      y ∈ Kset i) :
    TendstoUniformlyOn (fun β => run (fun i => scaledReadLayer (G i) (r i) (lam β i)) n)
      (run (fun i => readLayer (G i) (r i)) n) pf A₀ := by
  refine model_limit_on _ _ L hL1 (fun β _ => Kb * (Mb * ρ β)) (fun β _ => by
    have := hρ β; positivity) (fun i => ?_) Kset hL
    (fun β i y hy => scaled_read_layer_bound (G i) (r i) (lam β i) Kb Mb (ρ β) y hK0 (hG i y)
      (hr i y) (hlam β i y hy)) n A₀ r₀ hr₀ hK
  simpa using (hρlim.const_mul Mb).const_mul Kb

/-- The compensator's factor: the carried read arrives scaled by `t/κ` (`t` the pad's normed
coordinate, `∝ 1/RMS`), and the host multiplies it by the fitted `g(t) ≈ κ/t`. -/
def compFactor (g : ℝ → ℝ) (t : V → ℝ) (κ : ℝ) (x : V) : ℝ := g (t x) * (t x / κ)

/-- **The 1-D reciprocal compensator, the whole model.** A fit `g` with sup relative error `ρ` on
the layer's pad range `[t_min, t_max]` (`|g(τ)τ/κ − 1| ≤ ρ`, the quantity tabulated in
PROOFS.md's compensator table), carried magnitudes `≤ M̄` written into the residual (`G x y = A x + y`,
`K̄ = 1`): logits within `Lh (Π L i) · n · M̄ ρ`, on states whose pad coordinate stays in range. -/
theorem compensator_model_bound {Y : Type*} [PseudoMetricSpace Y] (A : ℕ → V → V)
    (u : ℕ → V → V) (g : ℕ → ℝ → ℝ) (t : ℕ → V → ℝ) (κ : ℕ → ℝ) (tmin tmax Mb ρ : ℝ)
    (hM0 : 0 ≤ Mb) (hρ : 0 ≤ ρ) (hu : ∀ i x, ‖u i x‖ ≤ Mb)
    (hfit : ∀ i τ, tmin ≤ τ → τ ≤ tmax → |g i τ * (τ / κ i) - 1| ≤ ρ)
    (L : ℕ → ℝ) (hL1 : ∀ i, 1 ≤ L i)
    (hL : ∀ i, ∀ a ∈ {x | tmin ≤ t i x ∧ t i x ≤ tmax}, ∀ b ∈ {x | tmin ≤ t i x ∧ t i x ≤ tmax},
      dist (readLayer (fun x y => A i x + y) (u i) a) (readLayer (fun x y => A i x + y) (u i) b) ≤
        L i * dist a b)
    (head : V → Y) (Lh : ℝ) (hLh0 : 0 ≤ Lh) (n : ℕ)
    (hLh : ∀ a ∈ {x | tmin ≤ t n x ∧ t n x ≤ tmax}, ∀ b ∈ {x | tmin ≤ t n x ∧ t n x ≤ tmax},
      dist (head a) (head b) ≤ Lh * dist a b) (x : V)
    (hK : ∀ i ≤ n, ∀ y, dist y (run (fun i => readLayer (fun x y => A i x + y) (u i)) i x) ≤
      (∏ j ∈ range n, L j) * (n * (1 * (Mb * ρ))) → tmin ≤ t i y ∧ t i y ≤ tmax) :
    dist (head (run (fun i => scaledReadLayer (fun x y => A i x + y) (u i)
        (compFactor (g i) (t i) (κ i))) n x))
        (head (run (fun i => readLayer (fun x y => A i x + y) (u i)) n x)) ≤
      Lh * ((∏ i ∈ range n, L i) * (n * (1 * (Mb * ρ)))) :=
  scaled_read_model_bound (fun i x y => A i x + y) u (fun i => compFactor (g i) (t i) (κ i))
    (fun i => {x | tmin ≤ t i x ∧ t i x ≤ tmax}) 1 Mb ρ zero_le_one hM0 hρ
    (fun i x y y' => by rw [dist_eq_norm, add_sub_add_left_eq_sub, one_mul]) hu
    (fun i x hx => hfit i _ hx.1 hx.2) L hL1 hL head Lh hLh0 n hLh x hK

end scaled

end EP
