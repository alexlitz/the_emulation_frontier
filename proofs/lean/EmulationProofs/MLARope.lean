import EmulationProofs.Position

/-!
# MLA with the decoupled RoPE key ⊆ MHA with partial RoPE

DeepSeek-style MLA splits each head's logit into a content part and a decoupled RoPE part:

* content: `q_C,h · k_C,h(j)` with `k_C,h(j) = W_UK,h c_j`, the latent `c_j = W_DKV x_j` shared
  by all heads;
* RoPE: `⟨R(i ω) q_R,h, R(pos_j ω) k_R(j)⟩` with ONE rotated key `k_R(j) = W_KR x_j` shared by
  all heads;
* values `W_UV,h c_j`.

An MHA host head with partial RoPE holds the content part in pass (frequency-0) pairs and the
RoPE part in rotating pairs: `W_K,h = [W_UK,h W_DKV ; W_KR]` (the shared RoPE key copied into
every head), `W_Q,h = [W_qC,h ; W_qR,h]`, `W_V,h = W_UV,h W_DKV`. Head dims are modelled as
pairs (content width `2 Pc`, RoPE width `2 Pr`). Norm-free: the latent RMSNorm is refuted
separately (`mla_latent_norm_not_mha`), and a query-latent norm would make the query nonlinear.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {ι : Type*} [Fintype ι] {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  {n r Pc Pr H : ℕ}

/-- A partial-RoPE MHA head: pairs `Pp` with frequencies `ω` (pass pairs have frequency `0`),
linear query, key and value projections, query at position `i`, key `j` at position `pos j`. -/
def ropeHead {Pp : Type*} [Fintype Pp] (ω : Pp → ℝ) (Wq Wk : (Fin n → ℝ) →ₗ[ℝ] (Pp → ℝ × ℝ))
    (Wv : (Fin n → ℝ) →ₗ[ℝ] V) (S : Finset ι) (i : ℝ) (pos : ι → ℝ) (xq : Fin n → ℝ)
    (xs : ι → Fin n → ℝ) : V :=
  attn S (fun j => ropeScore ω (Wq xq) (Wk (xs j)) i (pos j)) (fun j => Wv (xs j))

/-- An MLA head `h` with the decoupled RoPE key (norm-free). The latent `W_DKV` and the RoPE key
`W_KR` are shared by all heads; the rest is per head. -/
def mlaRopeHead (ω : Fin Pr → ℝ) (WqC : Fin H → (Fin n → ℝ) →ₗ[ℝ] (Fin Pc → ℝ × ℝ))
    (WqR : Fin H → (Fin n → ℝ) →ₗ[ℝ] (Fin Pr → ℝ × ℝ)) (Wdkv : (Fin n → ℝ) →ₗ[ℝ] (Fin r → ℝ))
    (Wuk : Fin H → (Fin r → ℝ) →ₗ[ℝ] (Fin Pc → ℝ × ℝ)) (Wkr : (Fin n → ℝ) →ₗ[ℝ] (Fin Pr → ℝ × ℝ))
    (Wuv : Fin H → (Fin r → ℝ) →ₗ[ℝ] V) (h : Fin H) (S : Finset ι) (i : ℝ) (pos : ι → ℝ)
    (xq : Fin n → ℝ) (xs : ι → Fin n → ℝ) : V :=
  attn S (fun j => (∑ p, dot2 (WqC h xq p) (Wuk h (Wdkv (xs j)) p)) +
      ropeScore ω (WqR h xq) (Wkr (xs j)) i (pos j)) (fun j => Wuv h (Wdkv (xs j)))

/-- Stack two pair-valued linear maps into one over `Pc ⊕ Pr` pairs. -/
def stackPairs {α β : Type*} (f : (Fin n → ℝ) →ₗ[ℝ] (α → ℝ × ℝ))
    (g : (Fin n → ℝ) →ₗ[ℝ] (β → ℝ × ℝ)) : (Fin n → ℝ) →ₗ[ℝ] (α ⊕ β → ℝ × ℝ) :=
  LinearMap.pi (fun o => o.elim (fun p => (LinearMap.proj p).comp f)
    (fun p => (LinearMap.proj p).comp g))

@[simp] lemma stackPairs_inl {α β : Type*} (f : (Fin n → ℝ) →ₗ[ℝ] (α → ℝ × ℝ))
    (g : (Fin n → ℝ) →ₗ[ℝ] (β → ℝ × ℝ)) (x : Fin n → ℝ) (p : α) :
    stackPairs f g x (Sum.inl p) = f x p := rfl

@[simp] lemma stackPairs_inr {α β : Type*} (f : (Fin n → ℝ) →ₗ[ℝ] (α → ℝ × ℝ))
    (g : (Fin n → ℝ) →ₗ[ℝ] (β → ℝ × ℝ)) (x : Fin n → ℝ) (p : β) :
    stackPairs f g x (Sum.inr p) = g x p := rfl

/-- **MLA with the decoupled shared RoPE key ⊆ MHA with partial RoPE** (exact). For every head
`h`, the MHA head with `Pc` pass pairs (frequency `0`) and `Pr` rotating pairs (the MLA RoPE
frequencies), query `[W_qC,h ; W_qR,h]`, key `[W_UK,h W_DKV ; W_KR]` and value `W_UV,h W_DKV`
equals the MLA head, for every query, key set, input and positions. -/
theorem mla_rope_sub_mha_partial (ω : Fin Pr → ℝ)
    (WqC : Fin H → (Fin n → ℝ) →ₗ[ℝ] (Fin Pc → ℝ × ℝ))
    (WqR : Fin H → (Fin n → ℝ) →ₗ[ℝ] (Fin Pr → ℝ × ℝ)) (Wdkv : (Fin n → ℝ) →ₗ[ℝ] (Fin r → ℝ))
    (Wuk : Fin H → (Fin r → ℝ) →ₗ[ℝ] (Fin Pc → ℝ × ℝ)) (Wkr : (Fin n → ℝ) →ₗ[ℝ] (Fin Pr → ℝ × ℝ))
    (Wuv : Fin H → (Fin r → ℝ) →ₗ[ℝ] V) (h : Fin H) (S : Finset ι) (i : ℝ) (pos : ι → ℝ)
    (xq : Fin n → ℝ) (xs : ι → Fin n → ℝ) :
    ropeHead (Sum.elim (fun _ : Fin Pc => (0 : ℝ)) ω) (stackPairs (WqC h) (WqR h))
        (stackPairs ((Wuk h).comp Wdkv) Wkr) ((Wuv h).comp Wdkv) S i pos xq xs =
      mlaRopeHead ω WqC WqR Wdkv Wuk Wkr Wuv h S i pos xq xs := by
  simp only [ropeHead, mlaRopeHead, ropeScore, Fintype.sum_sum_type, Sum.elim_inl,
    Sum.elim_inr, stackPairs_inl, stackPairs_inr, pairScore_zero_freq, LinearMap.comp_apply]

end EP
