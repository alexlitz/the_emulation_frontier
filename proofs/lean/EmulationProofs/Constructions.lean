import Mathlib

/-!
Machine-checked versions of three cookbook identities, over exact real arithmetic.
-/

open Real

/-- **Partial RoPE hosts full RoPE.** A host that rotates `r` dims with base `θ_h = θ_t^(r/d)`
has, on its pair `i`, exactly the frequency the target (`d` rotated dims, base `θ_t`) uses on its
pair `i`: `θ_h^(-2i/r) = θ_t^(-2i/d)`. So the target's frequencies are a prefix of the host's. -/
theorem partial_rope_freq_match (θt r d i : ℝ) (hθ : 0 < θt) (hr : r ≠ 0) (hd : d ≠ 0) :
    (θt ^ (r / d)) ^ (-2 * i / r) = θt ^ (-2 * i / d) := by
  rw [← Real.rpow_mul hθ.le]
  congr 1
  field_simp

/-- **GQA ⊆ MHA.** Tiling each KV head to the query heads of its group (`repeat_interleave`)
gives an MHA whose every per-head score equals the GQA score. -/
theorem gqa_tiled_scores {H G : ℕ} (q : Fin H → ℝ) (k : Fin G → ℝ) (g : Fin H → Fin G) :
    (fun h => q h * (fun h' => k (g h')) h) = (fun h => q h * k (g h)) := rfl

/-- **LayerNorm = RMSNorm after centring.** For a vector `x` of length `n`, the mean of the
squares of the centred vector `x - mean x` is the variance of `x`, so normalising the centred
vector by its RMS is exactly LayerNorm's normalisation. -/
theorem centred_rms_is_variance {n : ℕ} (hn : 0 < n) (x : Fin n → ℝ) :
    let μ := (∑ j, x j) / n
    (∑ j, (x j - μ) ^ 2) / n = (∑ j, x j ^ 2) / n - μ ^ 2 := by
  intro μ
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hsum : ∑ j, x j = n * μ := by simp only [μ]; field_simp
  have : ∑ j, (x j - μ) ^ 2 = ∑ j, x j ^ 2 - 2 * μ * ∑ j, x j + n * μ ^ 2 := by
    simp only [sub_sq, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← Finset.mul_sum, ← Finset.sum_mul]
    ring
  rw [this, hsum]
  field_simp
  ring
