import EmulationProofs.Position
import EmulationProofs.Norm
import EmulationProofs.MoE
import EmulationProofs.Capacity
import EmulationProofs.AuditRules

/-!
# Round-2 rules: recurrent capacity, self-conditioning, Zamba2, numeric config, dense-first MoE

Continues `Round2.lean` (items 5, 7–10 of the round-2 list).

* **Recurrent capacity** (`_rec_ok`, `RC_GE` / `RC_M2` / `RC_GDN` / `RC_KDA`).
  - A diagonal scan `s_{t+1} = a_t ⊙ s_t + b_t` (Mamba-1, Mamba-2: decay `exp(Δ A)`, input
    `Δ B x`): `dscan_inert`, `dscan_embed`, `scan_pad`, `mamba_state_pad`, `mamba_channel_inert`.
    Zero-padded states, channels and heads are inert.
  - The delta rule, row form, with a per-key-channel decay (GDN: constant; KDA: a vector):
    `deltaRow_inert` (zero value rows / heads), `deltaRow_pad` and `delta_out_pad` (zero-padded
    key dims), `gdn_out_scale` (the hard-coded `1/√dk` factors out).
  - Causal conv: `cconv_pad` (zero extra taps).
  - Grouped RMSNorm: `rmsNorm_rep` (m whole copies), `grouped_norm_zero_pad_fails` (zero padding
    with the same ε is not exact), `rmsNorm_pad_scaled` (a padded norm input that is a linear map's
    output: scale the rows, same ε).
  - Key-head-dim equality: `rmsNorm_scale_forces_one`, `gdn_dk_needed` (converse).
  - Gates and adapters: `fullrank_plays_lowrank`, `lowrank_pad`.
* **Self-conditioning**: `selfcond_zero_path`, `rmsNorm_one_ms_lt`, `rmsNorm_one_ms_le`,
  `selfcond_norm_not_exact`, `selfcond_norm_not_exact_eps0`, `selfcond_norm_escape`.
* **Zamba2 shared blocks**: `zamba_shared_index`, `zamba_shared_needs_multiple`,
  `joint_norm_depends` (the wall against non-Zamba hosts, for the carry construction).
* **Free numeric config**: `partial_rotary_raise`, `partial_rotary_raise_buffer`, `dt_floor_off`.
* **Dense-first MoE layers**: `dense_sub_moe_shared`.
-/

open Real Finset Filter Topology Matrix

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## 5a. Diagonal scans (Mamba-1 / Mamba-2) -/

/-- The diagonal linear scan from a zero state: `s_0 = 0`, `s_{t+1} = a_t ⊙ s_t + b_t`. -/
def dscan {ι : Type*} (a b : ℕ → ι → ℝ) : ℕ → ι → ℝ
  | 0 => 0
  | t + 1 => fun i => a t i * dscan a b t i + b t i

/-- **A state coordinate with zero input stays zero** (exact), whatever its decay. -/
theorem dscan_inert {ι : Type*} (a b : ℕ → ι → ℝ) (i : ι) (hb : ∀ t, b t i = 0) :
    ∀ t, dscan a b t i = 0 := by
  intro t; induction t with
  | zero => rfl
  | succ t ih => simp [dscan, ih, hb]

/-- **The scan is coordinatewise**: target coordinates embedded in the host (same decay, same
input) follow the target's scan. -/
theorem dscan_embed {κ ι : Type*} (e : κ → ι) (a b : ℕ → ι → ℝ) (a' b' : ℕ → κ → ℝ)
    (ha : ∀ t j, a t (e j) = a' t j) (hb : ∀ t j, b t (e j) = b' t j) :
    ∀ t j, dscan a b t (e j) = dscan a' b' t j := by
  intro t; induction t with
  | zero => intro j; rfl
  | succ t ih => intro j; simp [dscan, ih, ha, hb]

/-- **Zero-padded scan states are inert** (exact). A host scan over `ι ⊇ e(κ)` whose decay and
input agree with the target's on `e(κ)` and whose input is `0` elsewhere (a zero `B` row, a zero
input channel), read out by any `C` that agrees on `e(κ)` (arbitrary elsewhere), equals the
target's readout at every step. -/
theorem scan_pad {κ ι : Type*} [Fintype κ] [Fintype ι] (e : κ → ι) (he : Function.Injective e)
    (a b C : ℕ → ι → ℝ) (a' b' C' : ℕ → κ → ℝ)
    (ha : ∀ t j, a t (e j) = a' t j) (hb : ∀ t j, b t (e j) = b' t j)
    (hC : ∀ t j, C t (e j) = C' t j) (hoff : ∀ t i, i ∉ Set.range e → b t i = 0) (t : ℕ) :
    ∑ i, C t i * dscan a b t i = ∑ j, C' t j * dscan a' b' t j := by
  symm
  refine Fintype.sum_of_injective e he (fun j => C' t j * dscan a' b' t j)
    (fun i => C t i * dscan a b t i) (fun i hi => ?_) (fun j => ?_)
  · beta_reduce
    rw [dscan_inert a b i (fun t => hoff t i hi) t, mul_zero]
  · beta_reduce
    rw [hC, dscan_embed e a b a' b' ha hb]

/-- One channel of a selective scan: state `σ`, step `Δ_t`, decay rates `A`, input `x_t`,
`h_{t+1} = exp(Δ_t A) ⊙ h_t + Δ_t B_t x_t`, `y_t = C_t · h_t + D x_t`. Mamba-1 has `A` per
(channel, state); Mamba-2 has one `A` per head (a constant function of the state). -/
def mambaChannel {σ : Type*} [Fintype σ] (Δ : ℕ → ℝ) (A : σ → ℝ) (B C : ℕ → σ → ℝ)
    (x : ℕ → ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  ∑ n, C t n * dscan (fun t n => exp (Δ t * A n)) (fun t n => Δ t * B t n * x t) t n + D * x t

/-- **Zero-padded state dims (`d_state`) are inert** (exact): host state `σ' ⊇ e(σ)`, `B` zero-padded
(a zero `B` row; `A` and `C` arbitrary on the extra states). -/
theorem mamba_state_pad {σ σ' : Type*} [Fintype σ] [Fintype σ'] (e : σ → σ')
    (he : Function.Injective e) (Δ : ℕ → ℝ) (A : σ → ℝ) (A' : σ' → ℝ) (B C : ℕ → σ → ℝ)
    (B' C' : ℕ → σ' → ℝ) (x : ℕ → ℝ) (D : ℝ) (hA : ∀ n, A' (e n) = A n)
    (hB : ∀ t n, B' t (e n) = B t n) (hC : ∀ t n, C' t (e n) = C t n)
    (hoff : ∀ t m, m ∉ Set.range e → B' t m = 0) (t : ℕ) :
    mambaChannel Δ A' B' C' x D t = mambaChannel Δ A B C x D t := by
  unfold mambaChannel
  congr 1
  exact scan_pad e he _ _ _ _ _ _ (fun t j => by simp [hA]) (fun t j => by simp [hB])
    (fun t j => hC t j) (fun t m hm => by simp [hoff t m hm]) t

/-- **A zero-input channel / head is inert** (exact): with zero `in_proj` rows and conv taps its
post-conv input is `silu(0) = 0`, its state stays `0` and it outputs `0`, so `out_proj` sees
nothing from it. -/
theorem mamba_channel_inert {σ : Type*} [Fintype σ] (Δ : ℕ → ℝ) (A : σ → ℝ) (B C : ℕ → σ → ℝ)
    (D : ℝ) (t : ℕ) : mambaChannel Δ A B C (fun _ => 0) D t = 0 := by
  unfold mambaChannel
  have h0 : ∀ n, dscan (fun t n => exp (Δ t * A n)) (fun _ _ => (0 : ℝ)) t n = 0 :=
    fun n => dscan_inert _ _ n (fun _ => rfl) t
  simp [h0]

/-! ## 5b. The delta rule (gated DeltaNet, KDA), row form -/

/-- One value row `r` of a delta-rule state `S ∈ ℝ^{dv×dk}` (rows evolve independently):
`u = a_t ⊙ row`, `row' = u − β_t (u · k_t) k_t + β_t v_t(r) k_t`. GDN: `a_t` a constant vector
(`exp g_t`); KDA: a per-key-channel decay. -/
def deltaRow {dk : ℕ} (a : ℕ → Fin dk → ℝ) (β : ℕ → ℝ) (k : ℕ → Fin dk → ℝ) (v : ℕ → ℝ) :
    ℕ → Fin dk → ℝ
  | 0 => 0
  | t + 1 =>
    let u := a t * deltaRow a β k v t
    u - (β t * (u ⬝ᵥ k t)) • k t + (β t * v t) • k t

/-- **A value row (or head) with zero values stays zero** (exact): zero `v` rows of `in_proj` give
inert extra value channels and heads; replicated `v` channels give replicated rows. -/
theorem deltaRow_inert {dk : ℕ} (a : ℕ → Fin dk → ℝ) (β : ℕ → ℝ) (k : ℕ → Fin dk → ℝ) :
    ∀ t, deltaRow a β k (fun _ => 0) t = 0 := by
  intro t; induction t with
  | zero => rfl
  | succ t ih => simp [deltaRow, ih]

lemma padVec_dot {d e : ℕ} (x y : Fin d → ℝ) : padVec (e := e) x ⬝ᵥ padVec y = x ⬝ᵥ y := by
  simp [dotProduct, Fin.sum_univ_add]

/-- **Zero-padded key dims are inert in the recurrence** (exact): with the keys zero-padded and the
decay agreeing on the real key dims (anything on the extra ones), the host row is the target row,
zero-padded. -/
theorem deltaRow_pad {dk e : ℕ} (a : ℕ → Fin dk → ℝ) (a' : ℕ → Fin (dk + e) → ℝ)
    (ha : ∀ t j, a' t (Fin.castAdd e j) = a t j) (β : ℕ → ℝ) (k : ℕ → Fin dk → ℝ) (v : ℕ → ℝ) :
    ∀ t, deltaRow a' β (fun t => padVec (k t)) v t = padVec (deltaRow a β k v t) := by
  have hmul : ∀ t (s : Fin dk → ℝ), a' t * padVec (e := e) s = padVec (a t * s) := by
    intro t s; funext i
    refine Fin.addCases (fun j => ?_) (fun j => ?_) i
    · simp [ha]
    · simp
  intro t; induction t with
  | zero => funext i; refine Fin.addCases (fun j => ?_) (fun j => ?_) i <;> simp [deltaRow]
  | succ t ih =>
    simp only [deltaRow]
    rw [ih, hmul, padVec_dot]
    funext i
    refine Fin.addCases (fun j => ?_) (fun j => ?_) i <;> simp

/-- …and the readout `row · q` with a zero-padded query is the target's. -/
theorem delta_out_pad {dk e : ℕ} (a : ℕ → Fin dk → ℝ) (a' : ℕ → Fin (dk + e) → ℝ)
    (ha : ∀ t j, a' t (Fin.castAdd e j) = a t j) (β : ℕ → ℝ) (k : ℕ → Fin dk → ℝ) (v : ℕ → ℝ)
    (q : Fin dk → ℝ) (t : ℕ) :
    deltaRow a' β (fun t => padVec (k t)) v t ⬝ᵥ padVec q = deltaRow a β k v t ⬝ᵥ q := by
  rw [deltaRow_pad a a' ha, padVec_dot]

/-- **The hard-coded scale factors out**: the output with query scale `s` (`1/√dk` in code) is
`s` times the unscaled readout. A host with zero-padded key dims (`deltaRow_pad`; the l2 norms of
`q, k` are unchanged by zero padding) therefore outputs `√(dk_t/dk_h)` times the target's. -/
theorem gdn_out_scale {dk : ℕ} (row q : Fin dk → ℝ) (s : ℝ) : row ⬝ᵥ (s • q) = s * (row ⬝ᵥ q) := by
  rw [dotProduct_smul, smul_eq_mul]

/-- **A per-head RMSNorm (ε > 0) does not absorb a constant scale on its input** (converse). If
`rmsNorm ε w' (c • o)` equals `rmsNorm ε w o` at coordinate `i` (`w i ≠ 0`) for every `o`, with
any host weight `w'`, then `c = 1` (`c > 0`). -/
theorem rmsNorm_scale_forces_one {n : ℕ} (eps c : ℝ) (he : 0 < eps) (hc : 0 < c)
    (w w' : Fin n → ℝ) (i : Fin n) (hw : w i ≠ 0)
    (h : ∀ o : Fin n → ℝ, rmsNorm eps w' (c • o) i = rmsNorm eps w o i) : c = 1 := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  have hval : ∀ t : ℝ, 0 < t →
      (w' i) ^ 2 * (t ^ 2 / n + eps) = (w i) ^ 2 * (t ^ 2 / n + eps / c ^ 2) := by
    intro t ht
    have := h (t • Pi.single i 1)
    rw [rmsNorm_smul_eps eps c hc] at this
    simp only [rmsNorm] at this
    have hs : ∑ l, ((t • Pi.single i (1 : ℝ)) l) ^ 2 = t ^ 2 := by
      simp [Pi.single_apply, Finset.sum_ite_eq']
    rw [hs] at this
    simp only [Pi.smul_apply, Pi.single_eq_same, smul_eq_mul, mul_one] at this
    have hA : 0 < t ^ 2 / n + eps / c ^ 2 := by positivity
    have hB : 0 < t ^ 2 / n + eps := by positivity
    rw [div_eq_div_iff (Real.sqrt_pos.mpr hA).ne' (Real.sqrt_pos.mpr hB).ne'] at this
    have hsq := congrArg (fun z => z ^ 2) this
    simp only [mul_pow, Real.sq_sqrt hA.le, Real.sq_sqrt hB.le] at hsq
    have ht2 : t ^ 2 ≠ 0 := by positivity
    exact mul_left_cancel₀ ht2 (by linear_combination hsq)
  have h1 := hval 1 one_pos
  have h2 := hval 2 two_pos
  have hww : (w' i) ^ 2 = (w i) ^ 2 := by
    have : (w' i) ^ 2 * 3 / n = (w i) ^ 2 * 3 / n := by linear_combination h2 - h1
    field_simp at this; linarith
  rw [hww] at h1
  have hw2 : (w i) ^ 2 ≠ 0 := pow_ne_zero 2 hw
  have : eps = eps / c ^ 2 := by
    have := mul_left_cancel₀ hw2 h1; linarith
  have hc2 : c ^ 2 = 1 := by
    field_simp at this; nlinarith
  nlinarith

/-- **GDN / KDA need equal key head dims** (converse, for the padding construction). A host with
zero-padded key dims outputs `c = √(dk_t/dk_h)` times the target (`gdn_out_scale`); its per-head
output RMSNorm (the config ε > 0, any gain) reproduces the target's only if `c = 1`, i.e.
`dk_h = dk_t`. -/
theorem gdn_dk_needed {n : ℕ} (eps : ℝ) (he : 0 < eps) (dkt dkh : ℕ) (ht : 0 < dkt) (hh : 0 < dkh)
    (w w' : Fin n → ℝ) (i : Fin n) (hw : w i ≠ 0)
    (h : ∀ o : Fin n → ℝ, rmsNorm eps w' (√((dkt : ℝ) / dkh) • o) i = rmsNorm eps w o i) :
    dkt = dkh := by
  have hc : 0 < √((dkt : ℝ) / dkh) := Real.sqrt_pos.mpr (by positivity)
  have h1 := rmsNorm_scale_forces_one eps _ he hc w w' i hw h
  have : (dkt : ℝ) / dkh = 1 := by
    have h2 : (√((dkt : ℝ) / dkh)) ^ 2 = 1 := by rw [h1]; norm_num
    rwa [Real.sq_sqrt (by positivity)] at h2
  have hh' : (dkh : ℝ) ≠ 0 := by positivity
  field_simp at this
  exact_mod_cast this

/-! ## 5c. Causal conv -/

/-- A causal depthwise conv of `K` taps: tap `j` reads the input `j` steps back (`0` before the
start). -/
def cconv {K : ℕ} (w : Fin K → ℝ) (x : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ j : Fin K, if (j : ℕ) ≤ t then w j * x (t - j) else 0

/-- **Zero extra taps give the shorter kernel** (exact): a `K + e`-tap host kernel with the target's
taps and zeros on the oldest `e` positions. -/
theorem cconv_pad {K e : ℕ} (w : Fin K → ℝ) (x : ℕ → ℝ) (t : ℕ) :
    cconv (padVec (e := e) w) x t = cconv w x t := by
  simp [cconv, Fin.sum_univ_add]

/-! ## 5d. Grouped RMSNorm (Mamba-2's gated norm) -/

/-- `m` whole copies of a width-`w` vector. -/
def rep {w : ℕ} (m : ℕ) (x : Fin w → ℝ) : Fin (m * w) → ℝ := fun j => x (finProdFinEquiv.symm j).2

lemma sum_rep {w : ℕ} (m : ℕ) (f : Fin w → ℝ) :
    ∑ j : Fin (m * w), f (finProdFinEquiv.symm j).2 = m * ∑ i, f i := by
  rw [← Equiv.sum_comp finProdFinEquiv, Fintype.sum_prod_type]
  simp

/-- **m whole copies keep the group's RMSNorm** (exact, same ε): a host group holding `m` copies of
a target group (inputs and gains copied) outputs `m` copies of the target group's output. This is
`RC_M2`'s construction (and GDN's replicated value channels). -/
theorem rmsNorm_rep {w : ℕ} (m : ℕ) (hm : m ≠ 0) (eps : ℝ) (g x : Fin w → ℝ) :
    rmsNorm eps (rep m g) (rep m x) = rep m (rmsNorm eps g x) := by
  funext j
  simp only [rmsNorm, rep]
  congr 2
  rw [sum_rep m (fun i => x i ^ 2)]
  have : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  push_cast
  field_simp

/-- **Zero padding a group is not exact at the same ε** (why `RC_M2` asks for whole copies): a host
group of width `d + e` holding the target group zero-padded, under the same `ε > 0` and any gains,
misses the target's norm somewhere when `e > 0` (the group mean square changes; the epsilon that
would fix it is `ε d/(d+e)`, `hidden_pad_needs_eps`). -/
theorem grouped_norm_zero_pad_fails {d e : ℕ} (hd : d ≠ 0) (he : e ≠ 0) (eps : ℝ) (heps : 0 < eps)
    (w : Fin d → ℝ) (wh : Fin (d + e) → ℝ) (i : Fin d) (hw : w i ≠ 0) :
    ¬ ∀ x : Fin d → ℝ, rmsNorm eps wh (padVec x) (Fin.castAdd e i) = rmsNorm eps w x i := by
  intro h
  have := hidden_pad_needs_eps hd eps eps heps.le heps.le w wh i hw h
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hde : ((d + e : ℕ) : ℝ) = d + e := by push_cast; ring
  have he' : (0 : ℝ) < e := by exact_mod_cast Nat.pos_of_ne_zero he
  rw [hde] at this
  field_simp at this
  nlinarith

/-- **A padded norm input that is a linear map's output: scale the rows** (exact, same ε). With
`c² d = d + e`, `RMSNorm_ε` of the zero-padded `c x` is `c` times the padded `RMSNorm_ε(x)`; the
constant `c` folds into the next linear map (Falcon-Mamba / Jamba / PLaMo-2 `B`, `C`, `dt` norms:
`RC_GE`). -/
theorem rmsNorm_pad_scaled {d e : ℕ} (hd : d ≠ 0) (eps c : ℝ) (hc : 0 < c)
    (hcd : c ^ 2 * d = (d + e : ℕ)) (w x : Fin d → ℝ) :
    rmsNorm eps (padVec (e := e) w) (padVec (c • x)) = padVec (c • rmsNorm eps w x) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hde : (0 : ℝ) < ((d + e : ℕ) : ℝ) := by rw [← hcd]; positivity
  have key : (∑ l : Fin (d + e), (padVec (e := e) (c • x) l) ^ 2) / ((d + e : ℕ) : ℝ) =
      (∑ l, x l ^ 2) / d := by
    simp only [Fin.sum_univ_add, padVec_left, padVec_right, Pi.smul_apply, smul_eq_mul]
    simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sum_const_zero, add_zero,
      mul_pow, ← Finset.mul_sum]
    rw [← hcd]; field_simp
  funext i
  refine Fin.addCases (fun j => ?_) (fun j => ?_) i
  · simp only [rmsNorm, padVec_left, Pi.smul_apply, smul_eq_mul]
    have : ((d + e : ℕ) : ℝ) = (Fintype.card (Fin (d + e)) : ℝ) := by simp
    rw [key]; ring
  · simp [rmsNorm]

/-! ## 5e. Gates and adapters: full rank plays low rank -/

/-- **A full-rank gate plays a low-rank one** (exact): `W := W_b W_a`. -/
theorem fullrank_plays_lowrank {m r n : ℕ} (Wb : Matrix (Fin m) (Fin r) ℝ)
    (Wa : Matrix (Fin r) (Fin n) ℝ) (x : Fin n → ℝ) : (Wb * Wa) *ᵥ x = Wb *ᵥ (Wa *ᵥ x) :=
  (Matrix.mulVec_mulVec x Wb Wa).symm

/-- **A higher rank plays a lower one** (exact): rank `r + e` with the extra rank rows of `W_a` and
columns of `W_b` zero (KDA gate rank, Mamba dt rank, Zamba2 LoRA adapters, an expert latent of
rank ≥ the hidden size with `W_a = 1`). -/
theorem lowrank_pad {m r e n : ℕ} (Wb : Matrix (Fin m) (Fin r) ℝ) (Wa : Matrix (Fin r) (Fin n) ℝ)
    (x : Fin n → ℝ) :
    (Matrix.of fun i (j : Fin (r + e)) => padVec (e := e) (Wb i) j) *ᵥ
        ((Matrix.of fun (j : Fin (r + e)) l => Fin.addCases (fun j => Wa j l) (fun _ => 0) j) *ᵥ x) =
      Wb *ᵥ (Wa *ᵥ x) := by
  funext i
  simp [Matrix.mulVec, dotProduct, Fin.sum_univ_add]

/-! ## 7. Self-conditioning -/

/-- **Zeroing a self-conditioning path is exact when no always-on norm follows it** (LLaDA2.1-ORL):
`down_proj = 0` adds exactly `0` to the embedding. -/
theorem selfcond_zero_path {m n : ℕ} (emb : Fin m → ℝ) (h : Fin n → ℝ) :
    emb + (0 : Matrix (Fin m) (Fin n) ℝ) *ᵥ h = emb := by simp

lemma rmsNorm_one_sq_sum {n : ℕ} (eps : ℝ) (x : Fin n → ℝ) (hA : 0 < (∑ l, x l ^ 2) / n + eps) :
    ∑ l, (rmsNorm eps 1 x l) ^ 2 = (∑ l, x l ^ 2) / ((∑ l, x l ^ 2) / n + eps) := by
  simp only [rmsNorm, Pi.one_apply, one_mul, div_pow, Real.sq_sqrt hA.le, ← Finset.sum_div]

/-- A weight-free RMSNorm with `ε > 0` outputs a mean square `< 1`. -/
theorem rmsNorm_one_ms_lt {n : ℕ} (hn : n ≠ 0) (eps : ℝ) (he : 0 < eps) (x : Fin n → ℝ) :
    (∑ l, (rmsNorm eps 1 x l) ^ 2) / n < 1 := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero hn
  have hS : 0 ≤ ∑ l, x l ^ 2 := Finset.sum_nonneg (fun l _ => sq_nonneg _)
  have hA : 0 < (∑ l, x l ^ 2) / n + eps := by positivity
  rw [rmsNorm_one_sq_sum eps x hA, div_div, div_lt_one (by positivity)]
  have : (∑ l, x l ^ 2) / n * n = ∑ l, x l ^ 2 := by field_simp
  nlinarith

/-- …and `≤ 1` with `ε = 0`. -/
theorem rmsNorm_one_ms_le {n : ℕ} (hn : n ≠ 0) (x : Fin n → ℝ) :
    (∑ l, (rmsNorm 0 1 x l) ^ 2) / n ≤ 1 := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero hn
  have hS : 0 ≤ ∑ l, x l ^ 2 := Finset.sum_nonneg (fun l _ => sq_nonneg _)
  rcases hS.lt_or_eq with hS | hS
  · rw [rmsNorm_one_sq_sum 0 x (by positivity), add_zero]
    apply le_of_eq
    field_simp
  · have hx : ∀ l, x l = 0 := fun l =>
      pow_eq_zero_iff (n := 2) (by norm_num) |>.mp
        ((Finset.sum_eq_zero_iff_of_nonneg (fun l _ => sq_nonneg (x l))).mp hS.symm l (mem_univ l))
    simp [rmsNorm, hx]

/-- **With an always-on weight-free RMSNorm of the embeddings, zeroing the signal is NOT exact**
(counterexample, DiffusionGemma `post_norm`). The zeroed host's first residual is
`RMSNorm_ε(e' + 0)` for its own embedding row `e'` (any row), whose mean square is `< 1` for
`ε > 0`; a target token whose first residual `e` has mean square `≥ 1` is out of reach. -/
theorem selfcond_norm_not_exact {n : ℕ} (hn : n ≠ 0) (eps : ℝ) (he : 0 < eps) (e : Fin n → ℝ)
    (hms : 1 ≤ (∑ l, e l ^ 2) / n) (e' : Fin n → ℝ) : rmsNorm eps 1 (e' + 0) ≠ e := by
  intro h
  have := rmsNorm_one_ms_lt hn eps he (e' + 0)
  rw [h] at this; linarith

/-- …with `ε = 0`: a target token with mean square `> 1` is out of reach. -/
theorem selfcond_norm_not_exact_eps0 {n : ℕ} (hn : n ≠ 0) (e : Fin n → ℝ)
    (hms : 1 < (∑ l, e l ^ 2) / n) (e' : Fin n → ℝ) : rmsNorm 0 1 (e' + 0) ≠ e := by
  intro h
  have := rmsNorm_one_ms_le hn (e' + 0)
  rw [h] at this; linarith

/-- **Where the wall is not tight** (exact, `ε > 0`): a target token whose first residual has mean
square `< 1` is reached by the embedding row `c e` with `c = √(ε/(1 − ms))`. So a self-conditioning
host with an UNTIED head plays a target whose embedding rows all have mean square `< 1`; with
a tied head the per-token `c` also scales the logits, and is not exact. -/
theorem selfcond_norm_escape {n : ℕ} (eps : ℝ) (he : 0 < eps) (e : Fin n → ℝ)
    (hms : (∑ l, e l ^ 2) / n < 1) :
    rmsNorm eps 1 (√(eps / (1 - (∑ l, e l ^ 2) / n)) • e) = e := by
  set m := (∑ l, e l ^ 2) / n with hm
  have h1 : 0 < 1 - m := by linarith
  set c := √(eps / (1 - m))
  have hc : 0 < c := Real.sqrt_pos.mpr (by positivity)
  have hc2 : c ^ 2 = eps / (1 - m) := Real.sq_sqrt (by positivity)
  have hsum : (∑ l, c ^ 2 * e l ^ 2) / n + eps = c ^ 2 := by
    rw [← Finset.mul_sum, mul_div_assoc, ← hm, hc2, div_mul_eq_mul_div, div_add' _ _ _ h1.ne',
      div_left_inj' h1.ne']
    ring
  funext i
  simp only [rmsNorm, Pi.one_apply, one_mul, Pi.smul_apply, smul_eq_mul, mul_pow]
  rw [hsum, Real.sqrt_sq hc.le]
  field_simp

/-! ## 8. Zamba2 shared blocks -/

/-- **A multiple of the target's shared-block count** (exact): host block `b` holds target block
`b mod M_t`; invocation `i` runs host block `i mod M_h`, which is target block `i mod M_t`. -/
theorem zamba_shared_index {β : Type*} (Mt Mh : ℕ) (hd : Mt ∣ Mh) (tgt : ℕ → β) (i : ℕ) :
    (fun b => tgt (b % Mt)) (i % Mh) = tgt (i % Mt) := by
  simp [Nat.mod_mod_of_dvd i hd]

/-- **…and a multiple is needed** (converse): if every invocation's host block must be the target's
block (`hostW (i mod M_h) = tgt (i mod M_t)` for all `i`; the invocations reach `M_h`) and the
target's blocks are distinct, then `M_t ∣ M_h`. -/
theorem zamba_shared_needs_multiple {β : Type*} (Mt Mh : ℕ) (hMt : 0 < Mt) (tgt hostW : ℕ → β)
    (hdist : ∀ r s, r < Mt → s < Mt → tgt r = tgt s → r = s)
    (h : ∀ i, hostW (i % Mh) = tgt (i % Mt)) : Mt ∣ Mh := by
  have h1 := h Mh
  have h0 := h 0
  rw [Nat.mod_self] at h1
  simp only [Nat.zero_mod] at h0
  rw [h0] at h1
  have := hdist _ _ (Nat.mod_lt _ hMt) hMt h1.symm
  exact Nat.dvd_of_mod_eq_zero this

/-- **The shared block's joint norm depends on the embedding** (the wall against a non-Zamba host,
for the carry construction). Zamba reads `RMSNorm([x; emb])`; its `x` part changes with the
embedding's size (any weights, any `ε ≥ 0`), so no norm of `x` alone reproduces it, and a plain
host that carries `emb` in its stream feeds it to every other layer's norm as well. -/
theorem joint_norm_depends {d m : ℕ} (eps : ℝ) (he : 0 ≤ eps) (w : Fin (d + m) → ℝ) (x : Fin d → ℝ)
    (i : Fin d) (hx : x i ≠ 0) (hw : w (Fin.castAdd m i) ≠ 0) (emb : Fin m → ℝ) (hemb : emb ≠ 0) :
    rmsNorm eps w (Fin.append x emb) (Fin.castAdd m i) ≠
      rmsNorm eps w (Fin.append x 0) (Fin.castAdd m i) := by
  have hdm : (0 : ℝ) < ((d + m : ℕ) : ℝ) := by
    have := Fin.pos (Fin.castAdd m i); exact_mod_cast this
  obtain ⟨j, hj⟩ : ∃ j, emb j ≠ 0 := by
    by_contra hc; push Not at hc; exact hemb (funext hc)
  have hE : 0 < ∑ l, emb l ^ 2 :=
    lt_of_lt_of_le (by positivity) (Finset.single_le_sum (fun l _ => sq_nonneg (emb l)) (mem_univ j))
  have hX : 0 < ∑ l, x l ^ 2 :=
    lt_of_lt_of_le (by positivity) (Finset.single_le_sum (fun l _ => sq_nonneg (x l)) (mem_univ i))
  have s1 : ∑ l, (Fin.append x emb l) ^ 2 = ∑ l, x l ^ 2 + ∑ l, emb l ^ 2 := by
    simp [Fin.sum_univ_add]
  have s2 : ∑ l, (Fin.append x (0 : Fin m → ℝ) l) ^ 2 = ∑ l, x l ^ 2 := by
    simp [Fin.sum_univ_add]
  simp only [rmsNorm, Fin.append_left]
  rw [s1, s2]
  intro h
  have hA : 0 < (∑ l, x l ^ 2 + ∑ l, emb l ^ 2) / ((d + m : ℕ) : ℝ) + eps := by positivity
  have hB : 0 < (∑ l, x l ^ 2) / ((d + m : ℕ) : ℝ) + eps := by positivity
  rw [div_eq_div_iff (Real.sqrt_pos.mpr hA).ne' (Real.sqrt_pos.mpr hB).ne'] at h
  have h2 := mul_left_cancel₀ (mul_ne_zero hw hx) h
  have h3 := congrArg (· ^ 2) h2
  simp only [Real.sq_sqrt hA.le, Real.sq_sqrt hB.le] at h3
  have : (∑ l, emb l ^ 2) / ((d + m : ℕ) : ℝ) = 0 := by linarith [add_div (∑ l, x l ^ 2) (∑ l, emb l ^ 2) ((d + m : ℕ) : ℝ)]
  have : 0 < (∑ l, emb l ^ 2) / ((d + m : ℕ) : ℝ) := by positivity
  linarith

/-! ## 9. Free numeric config -/

/-- **Raising `partial_rotary_factor`, extra pairs on zero-padded dims** (exact). A partial-RoPE
target logit (rotating pairs `q_r, k_r` at frequencies `ω_t`, pass dims `q_p, k_p`) equals a host
that rotates `P_t + e` pairs (its first `P_t` frequencies the target's, `defaultFreq_match` with base
`θ_t^(rot_h/rot_t)`; any frequencies on the other `e`) with the target's pairs zero-padded, and
pass dims zero-padded. Needs `rot_h ≥ rot_t` and host pass width ≥ target pass width. -/
theorem partial_rotary_raise {Pt e p f : ℕ} (ωt : Fin Pt → ℝ) (ωh : Fin (Pt + e) → ℝ)
    (hω : ∀ j, ωh (Fin.castAdd e j) = ωt j) (qr kr : Fin Pt → ℝ × ℝ) (qp kp : Fin p → ℝ)
    (i j : ℝ) :
    ropeScore ωh (padPairs qr) (padPairs kr) i j + padVec (e := f) qp ⬝ᵥ padVec kp =
      ropeScore ωt qr kr i j + qp ⬝ᵥ kp := by
  rw [partial_rope_host ωt ωh hω]
  congr 1
  simp [dotProduct, Fin.sum_univ_add]

/-- **…or the target's pass dims on extra host pairs at frequency 0** (exact, the inv_freq buffer as
NoPE ⊆ RoPE sets it): no pass width needed. -/
theorem partial_rotary_raise_buffer {Pt Pp : ℕ} (ωt : Fin Pt → ℝ) (ωh : Fin (Pt + Pp) → ℝ)
    (hω : ∀ j, ωh (Fin.castAdd Pp j) = ωt j) (h0 : ∀ j, ωh (Fin.natAdd Pt j) = 0)
    (qr kr : Fin Pt → ℝ × ℝ) (qp kp : Fin Pp → ℝ × ℝ) (i j : ℝ) :
    ropeScore ωh (Fin.append qr qp) (Fin.append kr kp) i j =
      ropeScore ωt qr kr i j + ∑ l, dot2 (qp l) (kp l) := by
  simp [ropeScore, Fin.sum_univ_add, hω, h0]

/-- **A dt floor at or below 0 never binds** (exact): `softplus > 0`, so `max(lo, softplus x)` is
`softplus x` for `lo ≤ 0`. A positive `time_step_min` is a regime (it binds where
`softplus x < lo`); the code reads it from the config, so a host sets it to `0`. -/
theorem dt_floor_off (lo x : ℝ) (hlo : lo ≤ 0) :
    max lo (Real.log (1 + exp x)) = Real.log (1 + exp x) := by
  have : 0 < Real.log (1 + exp x) := Real.log_pos (by linarith [exp_pos x])
  exact max_eq_right (by linarith)

/-! ## 10. Dense-first MoE layers -/

/-- **A dense target layer on a host MoE layer with a shared expert** (exact, the relation's
`k e_w + s ≥ d_ff`). Zero router weights (the gates are a constant `g`, nonzero on the `k` experts
`sel` the tie-break picks: `zero_router_constant_gate`); deal `k e_w` of the dense units into
them (`dense_sub_moe`) and the other `s` into the shared expert, its down-projection divided by its
constant gate `c ≠ 0` (`1`, or `σ(0) = 1/2` with zero gate weights). A narrower dense FFN pads with
zero units first (`ffn_extra_units`). -/
theorem dense_sub_moe_shared {X Θ V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    {E k ew sw : ℕ} (act : Θ → X → ℝ) (θ : Fin (k * ew) → Θ) (W2 : Fin (k * ew) → V)
    (θs : Fin sw → Θ) (W2s : Fin sw → V) (g : Fin E → ℝ) (sel : Fin k → Fin E)
    (hsel : Function.Injective sel) (hg : ∀ b, g (sel b) ≠ 0)
    (hoff : ∀ e, e ∉ Set.range sel → g e = 0) (θ₀ : Θ) (c : ℝ) (hc : c ≠ 0) (x : X) :
    moeShared act (fun _ => g)
        (fun e u => if h : ∃ b, sel b = e then θ (finProdFinEquiv (h.choose, u)) else θ₀)
        (fun e u => if h : ∃ b, sel b = e then (g e)⁻¹ • W2 (finProdFinEquiv (h.choose, u)) else 0)
        (fun _ => c) θs (fun u => c⁻¹ • W2s u) x =
      ffn act (Sum.elim θ θs) (Sum.elim W2 W2s) x := by
  unfold moeShared
  rw [dense_sub_moe act θ W2 g sel hsel hg hoff θ₀ x]
  simp only [ffn, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr, Finset.smul_sum, smul_smul]
  congr 1
  refine Finset.sum_congr rfl (fun u _ => ?_)
  rw [mul_comm c, mul_assoc, inv_mul_cancel₀ hc, mul_one]

end EP
