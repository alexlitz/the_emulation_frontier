import Mathlib

/-!
# Shared definitions: softmax attention over a finite key set

All models are over exact real arithmetic. A query's attention output over a finite set `S` of
keys, with logits `s j` and values `v j` in a real normed space `V`, is
`(∑_{j∈S} exp (s j))⁻¹ • ∑_{j∈S} exp (s j) • v j`. A hard `{0, -∞}` mask is the restriction of
the sum to the unmasked keys; a finite additive bias `-B` is its `B → ∞` limit
(`attn_bias_limit`). The same lemma gives `softmax (β z) → one-hot (argmax z)`.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {ι : Type*} [Fintype ι] {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Softmax attention of one query over the key set `S`, logits `s`, values `v`. -/
def attn (S : Finset ι) (s : ι → ℝ) (v : ι → V) : V :=
  (∑ j ∈ S, exp (s j))⁻¹ • ∑ j ∈ S, exp (s j) • v j

/-- Softmax probabilities. -/
def softmax (s : ι → ℝ) (i : ι) : ℝ := exp (s i) / ∑ j, exp (s j)

lemma sum_exp_pos {S : Finset ι} (hS : S.Nonempty) (s : ι → ℝ) : 0 < ∑ j ∈ S, exp (s j) :=
  Finset.sum_pos (fun _ _ => exp_pos _) hS

/-- Attention is softmax-weighted averaging. -/
lemma attn_univ_eq_softmax (s : ι → ℝ) (v : ι → V) :
    attn univ s v = ∑ j, softmax s j • v j := by
  simp only [attn, softmax, Finset.smul_sum, smul_smul, div_eq_inv_mul]

/-- Adding the same constant to every logit does not change attention. -/
lemma attn_add_const (S : Finset ι) (s : ι → ℝ) (c : ℝ) (v : ι → V) :
    attn S (fun j => s j + c) v = attn S s v := by
  simp only [attn, exp_add, ← Finset.sum_mul, Finset.smul_sum, smul_smul]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  have := (exp_pos c).ne'
  congr 1
  rw [mul_inv]; field_simp

/-- An exponential with a negative rate dies. -/
lemma tendsto_exp_neg_rate (a c : ℝ) (hc : c < 0) :
    Tendsto (fun B : ℝ => exp (a + B * c)) atTop (𝓝 0) := by
  apply Real.tendsto_exp_atBot.comp
  apply tendsto_atBot_add_const_left
  exact tendsto_id.atTop_mul_const_of_neg hc

lemma tendsto_exp_rate (a c : ℝ) (hc : c ≤ 0) :
    Tendsto (fun B : ℝ => exp (a + B * c)) atTop (𝓝 (if c = 0 then exp a else 0)) := by
  rcases hc.lt_or_eq with h | h
  · simp only [h.ne, ite_false]; exact tendsto_exp_neg_rate a c h
  · subst h; simp

/-- **The hard mask as a limit.** Logits `a j + B * c j` with `c ≤ 0`: as `B → ∞` the attention
over all keys tends to the attention over the keys with `c j = 0` only. With `c j ∈ {0, -1}`
this is an additive mask `-B` on the masked keys (the `-∞` mask is its limit); with
`c j = z j - max z` it is `softmax (B z)` concentrating on the argmax. -/
theorem attn_bias_limit (a c : ι → ℝ) (v : ι → V) (hc : ∀ j, c j ≤ 0) (hS : ∃ j, c j = 0) :
    Tendsto (fun B : ℝ => attn univ (fun j => a j + B * c j) v) atTop
      (𝓝 (attn (univ.filter (fun j => c j = 0)) a v)) := by
  classical
  unfold attn
  have hden : Tendsto (fun B : ℝ => ∑ j, exp (a j + B * c j)) atTop
      (𝓝 (∑ j ∈ univ.filter (fun j => c j = 0), exp (a j))) := by
    rw [Finset.sum_filter]
    exact tendsto_finsetSum _ (fun j _ => tendsto_exp_rate (a j) (c j) (hc j))
  have hnum : Tendsto (fun B : ℝ => ∑ j, exp (a j + B * c j) • v j) atTop
      (𝓝 (∑ j ∈ univ.filter (fun j => c j = 0), exp (a j) • v j)) := by
    rw [Finset.sum_filter]
    refine tendsto_finsetSum _ (fun j _ => ?_)
    have := (tendsto_exp_rate (a j) (c j) (hc j)).smul_const (v j)
    by_cases h : c j = 0 <;> simp_all
  obtain ⟨j0, hj0⟩ := hS
  have hpos : 0 < ∑ j ∈ univ.filter (fun j => c j = 0), exp (a j) :=
    sum_exp_pos ⟨j0, by simp [hj0]⟩ a
  exact (hden.inv₀ hpos.ne').smul hnum

end EP

namespace EP

variable {ι : Type*} [Fintype ι] {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Attention is continuous in its logits: if every logit converges, the output converges. -/
theorem tendsto_attn {α : Type*} {l : Filter α} (S : Finset ι) (hS : S.Nonempty)
    (sn : α → ι → ℝ) (s : ι → ℝ) (v : ι → V) (h : ∀ j, Tendsto (fun n => sn n j) l (𝓝 (s j))) :
    Tendsto (fun n => attn S (sn n) v) l (𝓝 (attn S s v)) := by
  unfold attn
  have hden : Tendsto (fun n => ∑ j ∈ S, exp (sn n j)) l (𝓝 (∑ j ∈ S, exp (s j))) :=
    tendsto_finsetSum _ (fun j _ => (Real.continuous_exp.tendsto _).comp (h j))
  have hnum : Tendsto (fun n => ∑ j ∈ S, exp (sn n j) • v j) l (𝓝 (∑ j ∈ S, exp (s j) • v j)) :=
    tendsto_finsetSum _ (fun j _ => ((Real.continuous_exp.tendsto _).comp (h j)).smul_const (v j))
  exact (hden.inv₀ (sum_exp_pos hS s).ne').smul hnum

/-- RMSNorm with weight `w` and epsilon `eps` over `n` coordinates. -/
def rmsNorm {n : ℕ} (eps : ℝ) (w x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => w i * x i / √((∑ l, x l ^ 2) / n + eps)

/-- `σ(t) ≤ eᵗ`. -/
lemma sigmoid_le_exp (t : ℝ) : sigmoid t ≤ exp t := by
  rw [sigmoid_def, ← one_div, div_le_iff₀ (by positivity), mul_add, mul_one, ← exp_add,
    add_neg_cancel, exp_zero]
  linarith [exp_pos t]

/-- **Logit perturbation → attention perturbation.** If every logit moves by at most `δ`, every
softmax weight changes by at most a factor `e^{2δ}`. -/
theorem softmax_ratio_bound [Nonempty ι] (s t : ι → ℝ) (δ : ℝ) (h : ∀ j, |s j - t j| ≤ δ)
    (j : ι) : softmax s j ≤ exp (2 * δ) * softmax t j := by
  unfold softmax
  have ht := sum_exp_pos (S := (univ : Finset ι)) univ_nonempty t
  have hs := sum_exp_pos (S := (univ : Finset ι)) univ_nonempty s
  have hnum : exp (s j) ≤ exp δ * exp (t j) := by
    rw [← exp_add]; apply exp_le_exp.mpr; linarith [(abs_le.mp (h j)).2]
  have hden : exp (-δ) * ∑ l, exp (t l) ≤ ∑ l, exp (s l) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum (fun l _ => ?_)
    rw [← exp_add]; apply exp_le_exp.mpr; linarith [(abs_le.mp (h l)).1]
  rw [div_le_iff₀ hs, two_mul, exp_add]
  calc exp (s j) ≤ exp δ * exp (t j) := hnum
    _ = exp δ * exp δ * (exp (t j) / ∑ l, exp (t l)) * (exp (-δ) * ∑ l, exp (t l)) := by
        rw [exp_neg]; field_simp
    _ ≤ exp δ * exp δ * (exp (t j) / ∑ l, exp (t l)) * ∑ l, exp (s l) := by
        gcongr

end EP
