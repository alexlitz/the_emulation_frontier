import EmulationProofs.Position
import EmulationProofs.Norm
import EmulationProofs.Attention
import EmulationProofs.MoE
import EmulationProofs.Capacity
import EmulationProofs.AuditRules
import EmulationProofs.Converses

/-!
# Round-2 rules of the relation

Round 2 of `graph/model_graph.py` (branches `round2` … `round2-p4`) added rules that the earlier
files do not cover. This file proves each one, or refutes it with a counterexample theorem.

1. **qk-norm nestings** (`QKNORM_NEST`): `qkn_gain_one`, `qkn_equal_gains`, `qkn_ln_zero_bias`,
   `qkn_ln_equal_gains`, `weightFree_comm`, `rmsNorm_one_rot_comm` (a weight-free norm commutes
   with RoPE, so its position relative to RoPE does not matter).
2. **ALiBi slope schemes**: `alibiExps_mono`, `alibiSlopes_mono` (the BLOOM / MPT / Baichuan set
   nests by head count), `heads_inject` (extra heads zeroed, any head permutation);
   `alibi_sqrt_half_mem`, `alibi_bloom_not_rational` and `alibi_falcon_slope_not_mem` (Falcon's
   bf16 set and the real set do not nest either way); `mpt_bias_max_limit`.
3. **The window lever**: `band_ge_causal`, `window_ge_full`.
4. **DSA as dense**: `topk_all`, `dsa_dense`, `dsa_dense_causal`.
5. **Recurrent capacity** (`_rec_ok`): `dscan_inert`, `dscan_embed`, `scan_pad`, `mamba_state_pad`,
   `mamba_channel_inert`; `deltaRow_inert`, `deltaRow_pad`, `delta_out_pad`; `cconv_pad`;
   `rmsNorm_rep`, `grouped_norm_zero_pad_fails`, `rmsNorm_pad_scaled`; `gdn_out_scale`,
   `rmsNorm_scale_forces_one`, `gdn_dk_needed`; `fullrank_plays_lowrank`, `lowrank_pad`.
6. **Block-causal and diffusion under the mask currency**: `block_one_causal`, `block_ge_full`,
   `diffusion_sub_block`, `block_sub_diffusion`.
7. **Self-conditioning**: `selfcond_zero_path`; `rmsNorm_one_ms_lt`, `selfcond_norm_not_exact`,
   `selfcond_norm_not_exact_eps0` (the counterexample); `selfcond_norm_escape` (where the wall is
   not tight).
8. **Zamba2 shared blocks**: `zamba_shared_index`, `zamba_shared_needs_multiple`; adapters by
   `lowrank_pad`.
9. **Free numeric config**: `partial_rotary_raise`, `partial_rotary_raise_buffer`, `dt_floor_off`.
10. **Dense-first MoE layers**: `dense_sub_moe_shared`.
-/

open Real Finset Filter Topology Matrix

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## 1. qk-norm nestings -/

/-- Per-head RMSNorm with a gain per head (`head_distinct`, PLaMo-2); one gain shared by every
head is `head`, gain `1` is `head_free`. -/
def headNorm {H d : ℕ} (eps : ℝ) (g : Fin H → Fin d → ℝ) (q : Fin H → Fin d → ℝ) :
    Fin H → Fin d → ℝ :=
  fun h => rmsNorm eps (g h) (q h)

/-- **head ← head_distinct** (exact): per-head gains all equal to `w` are one shared gain. -/
theorem qkn_equal_gains {H d : ℕ} (eps : ℝ) (w : Fin d → ℝ) (q : Fin H → Fin d → ℝ) :
    headNorm eps (fun _ => w) q = fun h => rmsNorm eps w (q h) := rfl

/-- **head_free ← head / head_post / head_distinct** (exact): gain `1` is the weight-free norm
`x · (mean x² + ε)^{-1/2}` (Llama-4 `L2Norm`, nanochat), with the same `ε`. -/
theorem qkn_gain_one {H d : ℕ} (eps : ℝ) (q : Fin H → Fin d → ℝ) :
    headNorm eps (fun _ => 1) q =
      fun h => (√((∑ l, q h l ^ 2) / d + eps))⁻¹ • q h := by
  funext h i
  simp [headNorm, rmsNorm, div_eq_inv_mul, mul_comm]

/-- **head_ln ← head_ln_bias** (exact): a zero bias drops the bias. -/
theorem qkn_ln_zero_bias {H d : ℕ} (eps : ℝ) (γ : Fin H → Fin d → ℝ) (q : Fin H → Fin d → ℝ)
    (h : Fin H) : layerNorm eps (γ h) 0 (q h) = fun i => γ h i * rmsNorm eps 1 (center (q h)) i := by
  funext i
  simp only [layerNorm, rmsNorm, center, Pi.one_apply, one_mul, Pi.zero_apply, add_zero]
  ring

/-- **ln_shared_bias ← head_ln_bias** (exact): per-head gains and biases all equal are one shared
gain and bias. -/
theorem qkn_ln_equal_gains {H d : ℕ} (eps : ℝ) (γ β : Fin d → ℝ) (q : Fin H → Fin d → ℝ) :
    (fun h => layerNorm eps ((fun _ : Fin H => γ) h) ((fun _ : Fin H => β) h) (q h)) =
      fun h => layerNorm eps γ β (q h) := rfl

/-- A weight-free norm for any mean-square functional `ms`. -/
def weightFree {W : Type*} [AddCommGroup W] [Module ℝ W] (ms : W → ℝ) (eps : ℝ) (x : W) : W :=
  (√(ms x + eps))⁻¹ • x

/-- **A weight-free norm commutes with any linear map that keeps the mean square** (exact). -/
theorem weightFree_comm {W : Type*} [AddCommGroup W] [Module ℝ W] (ms : W → ℝ) (eps : ℝ)
    (R : W →ₗ[ℝ] W) (hR : ∀ x, ms (R x) = ms x) (x : W) :
    weightFree ms eps (R x) = R (weightFree ms eps x) := by
  simp [weightFree, hR]

/-- `rmsNorm eps 1` is the weight-free norm with `ms x = (∑ x²)/n`. -/
lemma rmsNorm_one_eq_weightFree {n : ℕ} (eps : ℝ) (x : Fin n → ℝ) :
    rmsNorm eps 1 x = weightFree (fun y : Fin n → ℝ => (∑ l, y l ^ 2) / n) eps x := by
  funext i; simp [rmsNorm, weightFree, div_eq_inv_mul, mul_comm]

/-- A RoPE rotation keeps each pair's squared norm. -/
lemma rot_sumsq (φ : ℝ) (p : ℝ × ℝ) : (rot φ p).1 ^ 2 + (rot φ p).2 ^ 2 = p.1 ^ 2 + p.2 ^ 2 := by
  simp only [rot]
  nlinarith [sin_sq_add_cos_sq φ]

/-- **A weight-free qk-norm commutes with RoPE** (exact). On a head of `P` pairs, any linear map of
the flattened head that rotates each pair (RoPE at any positions and frequencies) keeps the sum
of squares, so `norm (R q) = R (norm q)`. Hence a weight-free target normed AFTER RoPE (Llama-4,
nanochat) is played by a host normed BEFORE it with gain 1 (`head`) and vice versa (`head_post`
with gain 1); the order matters only with a gain that is not constant on each pair. -/
theorem rmsNorm_one_rot_comm {n : ℕ} (eps : ℝ) (R : (Fin n → ℝ) →ₗ[ℝ] (Fin n → ℝ))
    (hR : ∀ x, ∑ l, (R x) l ^ 2 = ∑ l, x l ^ 2) (x : Fin n → ℝ) :
    rmsNorm eps 1 (R x) = R (rmsNorm eps 1 x) := by
  rw [rmsNorm_one_eq_weightFree, rmsNorm_one_eq_weightFree]
  exact weightFree_comm _ eps R (fun y => by rw [hR]) x

/-! ## 2. ALiBi slope schemes -/

/-- The power of two at or above the head count: `N = 2^⌈log₂ n⌉`. -/
def alibiN (n : ℕ) : ℕ := 2 ^ Nat.clog 2 n

/-- The exponents of the ALiBi slopes of `n` heads, slope `= 2^(-B x)`. MPT's `gen_slopes`
(`2^(-k B / N)`, `k = 1 … N`, reordered `[1::2] + [::2]` and cut to `n`) keeps every even `k`
(those are `S(N/2)`) and the odd `k` below `2 (n − N/2)`. BLOOM's `build_alibi_tensor` and
Baichuan's `_get_interleave` give the same set (`B = 8`; checked against the relation's
`alibi_slopes` for `n = 1 … 512`). -/
def alibiExps (n : ℕ) : Set ℝ :=
  {x | ∃ k : ℕ, 1 ≤ k ∧ k ≤ alibiN n ∧ (2 ∣ k ∨ k < 2 * (n - alibiN n / 2)) ∧
    x = (k : ℝ) / alibiN n}

/-- The slope set `S_B(n)`. -/
def alibiSlopes (B : ℝ) (n : ℕ) : Set ℝ := (fun x => (2 : ℝ) ^ (-(B * x))) '' alibiExps n

/-- **The slope sets nest by head count** (exact): `S(n_t) ⊆ S(n_h)` whenever `n_t ≤ n_h`. -/
theorem alibiExps_mono {nt nh : ℕ} (h : nt ≤ nh) : alibiExps nt ⊆ alibiExps nh := by
  rintro x ⟨k, hk1, hkN, hk, rfl⟩
  have hp : Nat.clog 2 nt ≤ Nat.clog 2 nh := Nat.clog_mono_right 2 h
  rcases hp.lt_or_eq with hlt | heq
  · set a := Nat.clog 2 nt
    set b := Nat.clog 2 nh
    have hab : 2 ^ b = 2 ^ a * 2 ^ (b - a) := by rw [← pow_add]; congr 1; omega
    refine ⟨k * 2 ^ (b - a), ?_, ?_, Or.inl ?_, ?_⟩
    · exact Nat.one_le_iff_ne_zero.mpr (by positivity)
    · simp only [alibiN] at hkN ⊢
      rw [hab]; exact Nat.mul_le_mul_right _ hkN
    · exact Dvd.dvd.mul_left (dvd_pow_self 2 (by omega)) k
    · simp only [alibiN]
      rw [hab]; push_cast
      field_simp
      rfl
  · refine ⟨k, hk1, ?_, ?_, ?_⟩
    · simpa [alibiN, heq] using hkN
    · simp only [alibiN, heq] at hk ⊢
      omega
    · simp [alibiN, heq]

theorem alibiSlopes_mono (B : ℝ) {nt nh : ℕ} (h : nt ≤ nh) :
    alibiSlopes B nt ⊆ alibiSlopes B nh :=
  Set.image_mono (alibiExps_mono h)

/-- **Heads: extra host heads zeroed, any permutation** (exact). If target head `i` runs on host
head `σ i` (`σ` injective: MPT's reorder is a permutation of the same set) and every other host
head writes `0` (zero `o_proj` columns), the head sum is the target's. -/
theorem heads_inject {Ht Hh : ℕ} {V : Type*} [AddCommGroup V] (σ : Fin Ht → Fin Hh)
    (hσ : Function.Injective σ) (f : Fin Hh → V) (g : Fin Ht → V) (hon : ∀ i, f (σ i) = g i)
    (hoff : ∀ j, j ∉ Set.range σ → f j = 0) : ∑ j, f j = ∑ i, g i :=
  (Fintype.sum_of_injective σ hσ g f hoff (fun i => (hon i).symm)).symm

/-- `1/16` is an exponent of every head count `n ≥ 9`, so `2^{-1/2}` is a slope (`B = 8`). -/
theorem alibi_sqrt_half_mem {n : ℕ} (hn : 9 ≤ n) : (2 : ℝ) ^ (-(8 * (1 / 16 : ℝ))) ∈ alibiSlopes 8 n := by
  refine ⟨1 / 16, ?_, rfl⟩
  obtain ⟨p, hp⟩ : ∃ p, Nat.clog 2 n = p := ⟨_, rfl⟩
  have hN : alibiN n = 2 ^ p := by simp [alibiN, hp]
  have hp4 : 4 ≤ p := by
    have : 3 < p := by
      rw [← hp]
      exact (Nat.lt_clog_iff_pow_lt (by norm_num)).mpr (by norm_num; omega)
    omega
  have hnN : n ≤ 2 ^ p := hp ▸ Nat.le_pow_clog (by norm_num) n
  have hsplit : 2 ^ p = 2 ^ (p - 4) * 16 := by
    rw [show (16 : ℕ) = 2 ^ 4 by norm_num, ← pow_add]; congr 1; omega
  refine ⟨2 ^ (p - 4), Nat.one_le_two_pow, ?_, ?_, ?_⟩
  · rw [hN, hsplit]; omega
  · rcases (show 4 = p ∨ 4 < p by omega) with h4 | h4
    · right
      subst h4
      rw [hN]
      norm_num at hnN ⊢
      omega
    · left; exact dvd_pow_self 2 (by omega)
  · rw [hN, hsplit]; push_cast
    field_simp

/-- **A rational-slope host does not hold the real set** (no nesting, Falcon host). Falcon rounds
its slopes to bf16, so every Falcon slope is rational, while for `n ≥ 9` heads the BLOOM / MPT /
Baichuan set contains `2^{-1/2}`, which is irrational. So no Falcon head count holds a
BLOOM-type target with `n_t ≥ 9`. -/
theorem alibi_bloom_not_rational {n : ℕ} (hn : 9 ≤ n) :
    ¬ alibiSlopes 8 n ⊆ Set.range ((↑) : ℚ → ℝ) := by
  intro h
  obtain ⟨q, hq⟩ := h (alibi_sqrt_half_mem hn)
  have e : (2 : ℝ) ^ (-(8 * (1 / 16 : ℝ))) = (√2)⁻¹ := by
    rw [show -(8 * (1 / 16 : ℝ)) = -(1 / 2) by norm_num, Real.rpow_neg (by norm_num),
      Real.sqrt_eq_rpow]
  rw [e] at hq
  have : Irrational (√2)⁻¹ := irrational_sqrt_two.inv
  exact this ⟨q, hq⟩

/-- **The real set does not hold Falcon's** (no nesting, Falcon target). Falcon's slope set for
`n ≥ 9` contains `bf16(2^{-1/2}) = 181/256` (round to nearest even; the rounding is checked in
Python, not formalised). No BLOOM / MPT / Baichuan head count has that slope:
`2^{-8k/N} = 181/256` would give `181^N · 2^{8k} = 2^{8N}`, and `181` is odd. -/
theorem alibi_falcon_slope_not_mem (n : ℕ) : (181 / 256 : ℝ) ∉ alibiSlopes 8 n := by
  rintro ⟨x, ⟨k, hk1, hkN, -, rfl⟩, hx⟩
  set N := alibiN n with hNdef
  have hN : 1 ≤ N := Nat.one_le_two_pow
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  have hpow : ((2 : ℝ) ^ (-(8 * ((k : ℝ) / N)))) ^ N = 2 ^ (-(8 * (k : ℝ))) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul (by norm_num)]
    congr 1; field_simp
  have hx' : (2 : ℝ) ^ (-(8 * ((k : ℝ) / N))) = 181 / 256 := hx
  have h2 : (2 : ℝ) ^ (-(8 * (k : ℝ))) = ((2 : ℝ) ^ (8 * k))⁻¹ := by
    rw [Real.rpow_neg (by norm_num)]; norm_cast
  rw [hx', h2, div_pow] at hpow
  have e1 : (181 : ℝ) ^ N * 2 ^ (8 * k) = 256 ^ N := by
    have : (0 : ℝ) < 256 ^ N := by positivity
    have : (0 : ℝ) < 2 ^ (8 * k) := by positivity
    field_simp at hpow
    linear_combination hpow
  have e2 : (256 : ℝ) ^ N = 2 ^ (8 * N) := by rw [pow_mul]; norm_num
  have hnat : 181 ^ N * 2 ^ (8 * k) = 2 ^ (8 * N) := by exact_mod_cast e1.trans e2
  have hdvd : 181 ^ N ∣ 2 ^ (8 * N) := ⟨_, hnat.symm⟩
  have hcop : Nat.Coprime (181 ^ N) (2 ^ (8 * N)) :=
    Nat.Coprime.pow _ _ (by norm_num)
  have h1 : 181 ^ N = 1 := Nat.Coprime.eq_one_of_dvd hcop hdvd
  have : N = 0 := by
    rcases Nat.eq_zero_or_pos N with h | h
    · exact h
    · have : 181 ≤ 181 ^ N := Nat.le_self_pow h.ne' 181
      omega
  omega

/-- **The remote MPT's `alibi_bias_max` switch is a limit in exact arithmetic.** Each slope
`2^{-kB/N}` tends to `0` as `B → ∞`; it is exactly `0` only by floating-point underflow. -/
theorem mpt_bias_max_limit (x : ℝ) (hx : 0 < x) :
    Tendsto (fun B : ℝ => (2 : ℝ) ^ (-(B * x))) atTop (𝓝 0) := by
  have : Tendsto (fun B : ℝ => -(B * x)) atTop atBot :=
    tendsto_neg_atTop_atBot.comp (tendsto_id.atTop_mul_const hx)
  exact (tendsto_rpow_atBot_of_base_gt_one 2 (by norm_num)).comp this

/-! ## 3. The window lever -/

/-- **A window at least as long as the sequence is the causal mask** (exact): for `T ≤ w`, the
band of width `w` at every query position of a length-`T` sequence is the whole causal prefix. -/
theorem band_ge_causal {T w : ℕ} (hw : T ≤ w) (i : Fin T) :
    band w i = univ.filter (fun j : Fin T => j ≤ i) := by
  ext j
  simp only [band, mem_filter, mem_univ, true_and, and_iff_left_iff_imp]
  intro _; have := i.isLt; omega

/-- …so sliding-window attention with `w ≥ T` is full causal attention on sequences of length
`≤ T`. -/
theorem window_ge_full {T w : ℕ} {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (hw : T ≤ w) (i : Fin T) (s : Fin T → ℝ) (v : Fin T → V) :
    attn (band w i) s v = attn (univ.filter (fun j : Fin T => j ≤ i)) s v := by
  rw [band_ge_causal hw]

/-! ## 4. DSA as dense attention -/

/-- **Top-k of at most k candidates keeps them all** (exact). Whatever the scores and the
tie-break, a selection `Sel ⊆ C` of `min k |C|` candidates is `C` once `|C| ≤ k`. -/
theorem topk_all {α : Type*} [DecidableEq α] (C Sel : Finset α) (k : ℕ) (hsub : Sel ⊆ C)
    (hcard : min k C.card ≤ Sel.card) (hk : C.card ≤ k) : Sel = C :=
  Finset.eq_of_subset_of_card_le hsub (by omega)

/-- **DSA is dense attention while the context is at most `index_topk`** (exact): the attention
over the indexer's top-k keys is the attention over all candidate keys (for block top-k, apply
`topk_all` to the blocks). -/
theorem dsa_dense {ι V : Type*} [Fintype ι] [DecidableEq ι] [NormedAddCommGroup V]
    [NormedSpace ℝ V] (C Sel : Finset ι) (k : ℕ) (hsub : Sel ⊆ C)
    (hcard : min k C.card ≤ Sel.card) (hk : C.card ≤ k) (s : ι → ℝ) (v : ι → V) :
    attn Sel s v = attn C s v := by
  rw [topk_all C Sel k hsub hcard hk]

/-- The causal form: at query `i` of a length-`T ≤ k` sequence the candidates are `Iic i`. -/
theorem dsa_dense_causal {T : ℕ} {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (k : ℕ) (hk : T ≤ k) (i : Fin T) (Sel : Finset (Fin T)) (hsub : Sel ⊆ Iic i)
    (hcard : min k (i + 1) ≤ Sel.card) (s : Fin T → ℝ) (v : Fin T → V) :
    attn Sel s v = attn (Iic i) s v :=
  dsa_dense (Iic i) Sel k hsub (by rwa [Fin.card_Iic]) (by rw [Fin.card_Iic]; omega) s v

/-! ## 6. Block-causal and diffusion under the mask currency -/

/-- The block-causal mask of block size `b`: key `j` is visible from query `i` iff its block is
not later. -/
def blockMask (b i j : ℕ) : Prop := j / b ≤ i / b

instance (b i j : ℕ) : Decidable (blockMask b i j) := by unfold blockMask; infer_instance

/-- Block size `1` is the causal mask (AR ⊆ block). -/
theorem block_one_causal (i j : ℕ) : blockMask 1 i j ↔ j ≤ i := by simp [blockMask]

/-- A block at least as long as the context is the full mask. -/
theorem block_ge_full {b i j : ℕ} (hi : i < b) (hj : j < b) : blockMask b i j := by
  simp [blockMask, Nat.div_eq_of_lt hi, Nat.div_eq_of_lt hj]

/-- **diffusion ⊆ block** (exact, mask currency): a block-causal host given an all-ones mask, or a
block size `b ≥ T`, is bidirectional attention. -/
theorem diffusion_sub_block {T b : ℕ} {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (hb : T ≤ b) (i : Fin T) (s : Fin T → ℝ) (v : Fin T → V) :
    attn (univ.filter (fun j : Fin T => blockMask b i j)) s v = attn univ s v := by
  congr 1
  ext j; simp only [mem_filter, mem_univ, true_and, iff_true]
  exact block_ge_full (by omega) (by omega)

/-- **block ⊆ diffusion** (exact, mask currency): a full-attention host run under the
caller-supplied block mask is the block-causal target, for any block size the caller uses. -/
theorem block_sub_diffusion {T : ℕ} {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (b : ℕ) (i : Fin T) (s : Fin T → ℝ) (v : Fin T → V) :
    maskedAttn (fun j : Fin T => blockMask b i j) s v =
      attn (univ.filter (fun j : Fin T => blockMask b i j)) s v := rfl

end EP
