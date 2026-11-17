import EmulationProofs.Common

/-!
# Positional encodings: RoPE, NoPE, ALiBi, partial RoPE, retention

A RoPE head's logit between a query at position `i` and a key at position `j` is a sum over
its rotating pairs `p` (frequency `ω p`) of `⟨R(i ω_p) q_p, R(j ω_p) k_p⟩`, where `R φ` is the
2-D rotation by `φ`. A pass (unrotated) dimension pair is a pair with frequency `0`
(`pairScore_zero_freq`), so we model a head as a family of pairs indexed by a finite type.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-- The rotation of a 2-vector by angle `φ`. -/
def rot (φ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * cos φ - p.2 * sin φ, p.1 * sin φ + p.2 * cos φ)

/-- Dot product of 2-vectors. -/
def dot2 (p q : ℝ × ℝ) : ℝ := p.1 * q.1 + p.2 * q.2

/-- One RoPE pair's contribution to the logit (query at `i`, key at `j`, frequency `ω`). -/
def pairScore (ω i j : ℝ) (q k : ℝ × ℝ) : ℝ := dot2 (rot (i * ω) q) (rot (j * ω) k)

/-- A RoPE head's logit: the sum of its pairs. -/
def ropeScore {Pr : Type*} [Fintype Pr] (ω : Pr → ℝ) (q k : Pr → ℝ × ℝ) (i j : ℝ) : ℝ :=
  ∑ p, pairScore (ω p) i j (q p) (k p)

/-- RoPE is relative: a pair's logit depends on the positions only through `i - j`. -/
theorem pairScore_eq (ω i j : ℝ) (q k : ℝ × ℝ) :
    pairScore ω i j q k =
      dot2 q k * cos ((i - j) * ω) + (q.1 * k.2 - q.2 * k.1) * sin ((i - j) * ω) := by
  simp only [pairScore, dot2, rot, sub_mul, cos_sub, sin_sub]
  ring

/-- A zero-frequency pair is the plain dot product (identity rotation). -/
@[simp] theorem pairScore_zero_freq (i j : ℝ) (q k : ℝ × ℝ) : pairScore 0 i j q k = dot2 q k := by
  simp [pairScore, rot, dot2]

/-- **NoPE ⊆ RoPE** (exact, free). With `inv_freq = 0` every rotation is the identity, so the
RoPE logit is the plain (NoPE) dot product at every pair of positions. -/
theorem nope_sub_rope {Pr : Type*} [Fintype Pr] (q k : Pr → ℝ × ℝ) (i j : ℝ) :
    ropeScore (fun _ => 0) q k i j = ∑ p, dot2 (q p) (k p) := by
  simp [ropeScore]

/-- `rope_theta` cannot reach NoPE: the default schedule's first frequency is `θ^0 = 1` for every
base `θ`, so `inv_freq = 0` has to be written into the buffer directly. -/
theorem rope_theta_cannot_reach_nope (θ : ℝ) (P : ℕ) : θ ^ (-(0 : ℝ) / P) = 1 := by simp

/-! ## NoPE ⊆ RoPE as a limit: linear RoPE scaling to infinity

Linear RoPE scaling by a factor `s` (transformers' `"linear"` rope type, `inv_freq / s`) divides every
position by `s`: pair `p` rotates by `(i / s) ω_p`, i.e. it runs at frequency `ω_p / s`. A config can set `s`
but not `inv_freq = 0` (`rope_theta_cannot_reach_nope`); as `s → ∞` every angle tends to `0`. -/

/-- The frequencies under linear RoPE scaling by `s`. -/
def linScaledFreq {Pr : Type*} (ω : Pr → ℝ) (s : ℝ) : Pr → ℝ := fun p => ω p / s

/-- One pair misses the unrotated dot product by at most `|q·k| x²/2 + |q × k| |x|`, `x = (i - j) ω`. -/
theorem pairScore_sub_dot2_le (ω i j : ℝ) (q k : ℝ × ℝ) :
    |pairScore ω i j q k - dot2 q k| ≤
      |dot2 q k| * (((i - j) * ω) ^ 2 / 2) + |q.1 * k.2 - q.2 * k.1| * |(i - j) * ω| := by
  rw [pairScore_eq]
  set x := (i - j) * ω
  have hc1 : cos x ≤ 1 := cos_le_one x
  have hc2 : 1 - x ^ 2 / 2 ≤ cos x := one_sub_sq_div_two_le_cos
  have hcos : |cos x - 1| ≤ x ^ 2 / 2 := by
    rw [abs_le]; constructor <;> nlinarith [sq_nonneg x]
  have hsin : |sin x| ≤ |x| := abs_sin_le_abs
  have e : dot2 q k * cos x + (q.1 * k.2 - q.2 * k.1) * sin x - dot2 q k =
      dot2 q k * (cos x - 1) + (q.1 * k.2 - q.2 * k.1) * sin x := by ring
  rw [e]
  calc |dot2 q k * (cos x - 1) + (q.1 * k.2 - q.2 * k.1) * sin x|
      ≤ |dot2 q k * (cos x - 1)| + |(q.1 * k.2 - q.2 * k.1) * sin x| := abs_add_le _ _
    _ = |dot2 q k| * |cos x - 1| + |q.1 * k.2 - q.2 * k.1| * |sin x| := by
      rw [abs_mul, abs_mul]
    _ ≤ |dot2 q k| * (x ^ 2 / 2) + |q.1 * k.2 - q.2 * k.1| * |x| := by gcongr

/-- **NoPE ⊆ RoPE, the limit (`rope_scale_limit_identity`).** Under linear scaling by `s` the RoPE logit
tends to the unrotated (NoPE) logit `Σ_p q_p · k_p` as `s → ∞`, at every pair of positions. -/
theorem rope_scale_limit_identity {Pr : Type*} [Fintype Pr] (ω : Pr → ℝ) (q k : Pr → ℝ × ℝ) (i j : ℝ) :
    Tendsto (fun s => ropeScore (linScaledFreq ω s) q k i j) atTop (𝓝 (∑ p, dot2 (q p) (k p))) := by
  unfold ropeScore
  refine tendsto_finsetSum _ (fun p _ => ?_)
  have hω : Tendsto (fun s : ℝ => ω p / s) atTop (𝓝 0) := tendsto_const_nhds.div_atTop tendsto_id
  have hc : Continuous (fun w : ℝ => pairScore w i j (q p) (k p)) := by
    unfold pairScore dot2 rot; fun_prop
  have := (hc.tendsto 0).comp hω
  simpa [linScaledFreq, Function.comp_def] using this

/-- **The rate, uniform on bounded positions.** With `|i - j| ≤ T` and `s > 0` the logit is within
`Σ_p (|q_p·k_p| (T ω_p / s)² / 2 + |q_p × k_p| T |ω_p| / s)` of the NoPE logit: `O(1/s)` uniformly over every
pair of positions at distance `≤ T` (a bounded context). -/
theorem rope_scale_limit_bound {Pr : Type*} [Fintype Pr] (ω : Pr → ℝ) (q k : Pr → ℝ × ℝ) (i j T s : ℝ)
    (hT : |i - j| ≤ T) (hs : 0 < s) :
    |ropeScore (linScaledFreq ω s) q k i j - ∑ p, dot2 (q p) (k p)| ≤
      ∑ p, (|dot2 (q p) (k p)| * ((T * |ω p| / s) ^ 2 / 2) +
        |(q p).1 * (k p).2 - (q p).2 * (k p).1| * (T * |ω p| / s)) := by
  unfold ropeScore
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum (fun p _ => ?_))
  refine (pairScore_sub_dot2_le _ i j (q p) (k p)).trans ?_
  have hx : |(i - j) * linScaledFreq ω s p| ≤ T * |ω p| / s := by
    rw [linScaledFreq, abs_mul, abs_div, abs_of_pos hs, mul_div_assoc]
    exact mul_le_mul_of_nonneg_right hT (div_nonneg (abs_nonneg _) hs.le)
  have hx2 : ((i - j) * linScaledFreq ω s p) ^ 2 ≤ (T * |ω p| / s) ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hx 2
  gcongr

/-- **NoPE ⊆ RoPE at the attention output (limit).** With the keys at positions `pos j`, the linearly
scaled RoPE head's attention output (any values, any key set) tends to the NoPE head's as `s → ∞`. -/
theorem rope_scale_limit_attn {ι Pr V : Type*} [Fintype ι] [Fintype Pr] [NormedAddCommGroup V]
    [NormedSpace ℝ V] (S : Finset ι) (hS : S.Nonempty) (ω : Pr → ℝ) (i : ℝ) (pos : ι → ℝ)
    (q : Pr → ℝ × ℝ) (k : ι → Pr → ℝ × ℝ) (v : ι → V) :
    Tendsto (fun s => attn S (fun j => ropeScore (linScaledFreq ω s) q (k j) i (pos j)) v)
      atTop (𝓝 (attn S (fun j => ∑ p, dot2 (q p) (k j p)) v)) :=
  tendsto_attn S hS _ _ v (fun j => rope_scale_limit_identity ω q (k j) i (pos j))

/-! ## ALiBi ⊆ RoPE (limit) -/

/-- The ALiBi logit: content logit minus slope times distance. -/
def alibiScore (content m i j : ℝ) : ℝ := content - m * (i - j)

/-- The RoPE host for an ALiBi head: the content pairs get frequency `0` (they carry `q·k`
unrotated), and one extra pair (`none`) has frequency `θ` with the constant components
`q = (-m/θ, 0)`, `k = (0, 1)` (a bias or constant residual channel). -/
def alibiHostFreq {Pr : Type*} (θ : ℝ) : Option Pr → ℝ := fun o => o.elim θ (fun _ => 0)

def alibiHostQ {Pr : Type*} (m θ : ℝ) (q : Pr → ℝ × ℝ) : Option Pr → ℝ × ℝ :=
  fun o => o.elim (-(m / θ), 0) q

def alibiHostK {Pr : Type*} (k : Pr → ℝ × ℝ) : Option Pr → ℝ × ℝ :=
  fun o => o.elim (0, 1) k

/-- The frozen pair contributes exactly `-(m/θ) sin(θ (i - j))`. -/
theorem alibi_host_score {Pr : Type*} [Fintype Pr] (m θ : ℝ) (q k : Pr → ℝ × ℝ) (i j : ℝ) :
    ropeScore (alibiHostFreq θ) (alibiHostQ m θ q) (alibiHostK k) i j =
      ∑ p, dot2 (q p) (k p) - m / θ * sin ((i - j) * θ) := by
  rw [ropeScore, Fintype.sum_option]
  simp only [alibiHostFreq, alibiHostQ, alibiHostK, Option.elim, pairScore_eq, dot2, mul_zero,
    cos_zero, sin_zero, mul_one]
  ring_nf

/-- **ALiBi ⊆ RoPE** (limit, `θ → 0`). The host logit converges to the ALiBi logit at every pair
of positions. -/
theorem alibi_sub_rope_limit {Pr : Type*} [Fintype Pr] (m : ℝ) (q k : Pr → ℝ × ℝ) (i j : ℝ) :
    Tendsto (fun θ => ropeScore (alibiHostFreq θ) (alibiHostQ m θ q) (alibiHostK k) i j)
      (𝓝[≠] 0) (𝓝 (alibiScore (∑ p, dot2 (q p) (k p)) m i j)) := by
  simp only [alibi_host_score, alibiScore]
  refine tendsto_const_nhds.sub ?_
  by_cases hΔ : i - j = 0
  · simp [hΔ]
  -- m/θ * sin(Δ θ) = m * Δ * (sin(Δθ)/(Δθ))
  have hlim : Tendsto (fun θ : ℝ => sin ((i - j) * θ) / ((i - j) * θ)) (𝓝[≠] 0) (𝓝 1) := by
    have hmap : Tendsto (fun θ : ℝ => (i - j) * θ) (𝓝[≠] 0) (𝓝[≠] 0) := by
      refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ ?_ ?_
      · have : Tendsto (fun θ : ℝ => (i - j) * θ) (𝓝 0) (𝓝 ((i - j) * 0)) :=
          (continuous_const.mul continuous_id).tendsto 0
        simpa using this.mono_left nhdsWithin_le_nhds
      · filter_upwards [self_mem_nhdsWithin] with θ hθ
        exact mul_ne_zero hΔ hθ
    exact Real.tendsto_sin_div_nhdsNE_zero.comp hmap
  have := (hlim.const_mul (m * (i - j)))
  rw [mul_one] at this
  refine this.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with θ hθ
  field_simp

/-- **ALiBi ⊆ RoPE, the rate.** The host logit misses the ALiBi logit by at most
`|m| |i - j|³ θ² / 6`, i.e. `O(θ²)`. -/
theorem alibi_sub_rope_bound {Pr : Type*} [Fintype Pr] (m θ : ℝ) (hθ : θ ≠ 0) (q k : Pr → ℝ × ℝ)
    (i j : ℝ) :
    |ropeScore (alibiHostFreq θ) (alibiHostQ m θ q) (alibiHostK k) i j -
        alibiScore (∑ p, dot2 (q p) (k p)) m i j| ≤ |m| * |i - j| ^ 3 * θ ^ 2 / 6 := by
  rw [alibi_host_score, alibiScore]
  have h := Real.abs_sub_sin_le ((i - j) * θ)
  have e : ∑ p, dot2 (q p) (k p) - m / θ * sin ((i - j) * θ) -
      (∑ p, dot2 (q p) (k p) - m * (i - j)) = (m / θ) * ((i - j) * θ - sin ((i - j) * θ)) := by
    field_simp; ring
  rw [e, abs_mul, abs_div]
  have hθ' : 0 < |θ| := abs_pos.mpr hθ
  calc |m| / |θ| * |(i - j) * θ - sin ((i - j) * θ)|
      ≤ |m| / |θ| * (|(i - j) * θ| ^ 3 / 6) := by gcongr
    _ = |m| * |i - j| ^ 3 * θ ^ 2 / 6 := by
      rw [abs_mul, mul_pow, ← sq_abs θ]; field_simp

/-- **ALiBi ⊆ RoPE at the attention output.** With content logits `c j` and the ALiBi bias of
the key at position `pos j`, the host's attention output (any value vectors, any key set)
converges to the ALiBi head's output as `θ → 0`. -/
theorem alibi_sub_rope_attn {ι Pr V : Type*} [Fintype ι] [Fintype Pr] [NormedAddCommGroup V]
    [NormedSpace ℝ V] (S : Finset ι) (hS : S.Nonempty) (m i : ℝ) (pos : ι → ℝ)
    (q : Pr → ℝ × ℝ) (k : ι → Pr → ℝ × ℝ) (v : ι → V) :
    Tendsto (fun θ => attn S (fun j =>
        ropeScore (alibiHostFreq θ) (alibiHostQ m θ q) (alibiHostK (k j)) i (pos j)) v)
      (𝓝[≠] 0) (𝓝 (attn S (fun j => alibiScore (∑ p, dot2 (q p) (k j p)) m i (pos j)) v)) :=
  tendsto_attn S hS _ _ v (fun j => alibi_sub_rope_limit m q (k j) i (pos j))

/-- **The constant channel must sit in an unrotated pair.** A constant query component `(1, 0)`
against a fixed key `(s, 0)` (an attention-sink key or the ALiBi frozen pair's partner) in a pair
of frequency `ω` gives `s cos((i - p) ω)`, which depends on the query position unless `ω = 0`. -/
theorem const_channel_rotates (ω i p s : ℝ) :
    pairScore ω i p (1, 0) (s, 0) = s * cos ((i - p) * ω) := by
  rw [pairScore_eq]; simp [dot2]

/-- The ALiBi frozen pair is odd in `i - j`, so it reproduces the causal bias `-m (i - j)`, not a
bidirectional `-m |i - j|`. -/
theorem alibi_pair_odd (m θ i j : ℝ) :
    pairScore θ j i (-(m / θ), 0) (0, 1) = -pairScore θ i j (-(m / θ), 0) (0, 1) := by
  rw [pairScore_eq, pairScore_eq, show j - i = -(i - j) by ring, neg_mul, cos_neg, sin_neg]
  simp [dot2]

/-! ## Partial-RoPE host ⊇ full-RoPE target -/

/-- Embed a target head (`Pt` pairs) in a host head with `Pt + e` pairs: target pair `j` on host
pair `j`, the spare host pairs zero. -/
def padPairs {Pt e : ℕ} (q : Fin Pt → ℝ × ℝ) : Fin (Pt + e) → ℝ × ℝ :=
  Fin.addCases q (fun _ => 0)

/-- **Partial RoPE host ⊇ RoPE target, buffer form** (exact). If the host's first `Pt`
frequencies equal the target's (whatever the host's other `e` frequencies are), the host logit
equals the target logit for all queries, keys and positions. A partial target's pass dims are
zero-frequency pairs, so this also covers partial targets. -/
theorem partial_rope_host {Pt e : ℕ} (ωt : Fin Pt → ℝ) (ωh : Fin (Pt + e) → ℝ)
    (hω : ∀ j, ωh (Fin.castAdd e j) = ωt j) (q k : Fin Pt → ℝ × ℝ) (i j : ℝ) :
    ropeScore ωh (padPairs q) (padPairs k) i j = ropeScore ωt q k i j := by
  simp [ropeScore, Fin.sum_univ_add, padPairs, hω, pairScore, dot2, rot]

/-- The default schedule: pair `p` of a head with `P` pairs (`2P` rotated dims) and base `θ`
has frequency `θ^(-2p / 2P)`. -/
def defaultFreq (θ : ℝ) (P : ℕ) (p : Fin P) : ℝ := θ ^ (-(2 * (p : ℝ)) / (2 * P))

/-- The config-only route: base `θ_h = θ_t^(rot_h / d_t)` gives the host's pair `j` exactly the
target's frequency, for `j` below the target's pair count. -/
theorem defaultFreq_match {Pt e : ℕ} (hPt : Pt ≠ 0) (θt : ℝ) (hθ : 0 < θt) (j : Fin Pt) :
    defaultFreq (θt ^ ((2 * (Pt + e : ℕ) : ℝ) / (2 * Pt))) (Pt + e) (Fin.castAdd e j) =
      defaultFreq θt Pt j := by
  simp only [defaultFreq, Fin.val_castAdd]
  rw [← Real.rpow_mul hθ.le]
  congr 1
  have : (Pt : ℝ) ≠ 0 := by exact_mod_cast hPt
  have : ((Pt + e : ℕ) : ℝ) ≠ 0 := by positivity
  field_simp

/-- **Partial RoPE host ⊇ full RoPE target, config form** (exact): a default-RoPE host with
`rot_h = 2(Pt + e) ≥ d_t = 2 Pt` rotated dims and base `θ_t^(rot_h/d_t)` reproduces the
default-RoPE target logit exactly. -/
theorem partial_rope_host_config {Pt e : ℕ} (hPt : Pt ≠ 0) (θt : ℝ) (hθ : 0 < θt)
    (q k : Fin Pt → ℝ × ℝ) (i j : ℝ) :
    ropeScore (defaultFreq (θt ^ ((2 * (Pt + e : ℕ) : ℝ) / (2 * Pt))) (Pt + e))
        (padPairs q) (padPairs k) i j = ropeScore (defaultFreq θt Pt) q k i j :=
  partial_rope_host _ _ (defaultFreq_match hPt θt hθ) q k i j

/-! ### The host's qk-norm over a zero-padded head -/

/-- Zero-pad a width-`d` vector to width `d + e`. -/
def padVec {d e : ℕ} (x : Fin d → ℝ) : Fin (d + e) → ℝ := Fin.addCases x (fun _ => 0)

@[simp] lemma padVec_left {d e : ℕ} (x : Fin d → ℝ) (i : Fin d) :
    padVec (e := e) x (Fin.castAdd e i) = x i := by simp [padVec]

@[simp] lemma padVec_right {d e : ℕ} (x : Fin d → ℝ) (i : Fin e) :
    padVec (e := e) x (Fin.natAdd d i) = 0 := by simp [padVec]

/-- **qk-norm on a zero-padded head.** A host head of width `d + e` running RMSNorm over a
target head of width `d` zero-padded: with `eps_h = eps_t d / (d + e)` and weight
`w_h = √(d / (d + e)) w_t` on the real coordinates, the host norm equals the target norm,
padded. -/
theorem qknorm_pad {d e : ℕ} (hd : d ≠ 0) (eps : ℝ) (w x : Fin d → ℝ) :
    rmsNorm (eps * d / (d + e : ℕ)) (padVec (e := e) (fun i => √((d : ℝ) / (d + e : ℕ)) * w i))
        (padVec x) = padVec (rmsNorm eps w x) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hd
  have hde : (0 : ℝ) < (d + e : ℕ) := by exact_mod_cast (Nat.pos_of_ne_zero hd).trans_le (by omega)
  have key : √((∑ l : Fin (d + e), (padVec x l) ^ 2) / (d + e : ℕ) + eps * d / (d + e : ℕ)) =
      √((d : ℝ) / (d + e : ℕ)) * √((∑ l, x l ^ 2) / d + eps) := by
    rw [← Real.sqrt_mul (by positivity)]
    congr 1
    simp only [Fin.sum_univ_add, padVec_left, padVec_right]
    simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sum_const_zero, add_zero]
    field_simp
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · simp only [rmsNorm, padVec_left]
    rw [key]
    have : 0 < √((d : ℝ) / (d + e : ℕ)) := Real.sqrt_pos.mpr (by positivity)
    rcases (Real.sqrt_nonneg ((∑ l, x l ^ 2) / d + eps)).lt_or_eq with h | h
    · field_simp
    · rw [← h]; simp
  · simp [rmsNorm]

/-! ## Retention: parallel ≡ recurrent -/

/-- Retention, parallel form: `y_n = ∑_{m ≤ n} γ^{n-m} (q̃_n · k̃_m) v_m`, where `q̃_n = R_n q_n`,
`k̃_m = R_m k_m` are the position-rotated queries and keys (any per-position map works). -/
def retParallel {κ τ : Type*} [Fintype κ] (γ : ℝ) (q k : ℕ → κ → ℝ) (v : ℕ → τ → ℝ) (n : ℕ) :
    τ → ℝ :=
  ∑ m ∈ range (n + 1), (γ ^ (n - m) * (q n ⬝ᵥ k m)) • v m

/-- Retention's recurrent state `S_n = γ S_{n-1} + k̃_n v_nᵀ`. -/
def retState {κ τ : Type*} (γ : ℝ) (k : ℕ → κ → ℝ) (v : ℕ → τ → ℝ) : ℕ → Matrix κ τ ℝ
  | 0 => Matrix.vecMulVec (k 0) (v 0)
  | n + 1 => γ • retState γ k v n + Matrix.vecMulVec (k (n + 1)) (v (n + 1))

/-- Retention, recurrent form: `y_n = q̃_nᵀ S_n`. -/
def retRecurrent {κ τ : Type*} [Fintype κ] (γ : ℝ) (q k : ℕ → κ → ℝ) (v : ℕ → τ → ℝ) (n : ℕ) :
    τ → ℝ :=
  Matrix.vecMul (q n) (retState γ k v n)

theorem retState_eq {κ τ : Type*} (γ : ℝ) (k : ℕ → κ → ℝ) (v : ℕ → τ → ℝ) (n : ℕ) :
    retState γ k v n = ∑ m ∈ range (n + 1), γ ^ (n - m) • Matrix.vecMulVec (k m) (v m) := by
  induction n with
  | zero => simp [retState]
  | succ n ih =>
    rw [retState, ih, Finset.sum_range_succ _ (n + 1), Finset.smul_sum]
    congr 1
    · refine Finset.sum_congr rfl (fun m hm => ?_)
      rw [smul_smul, ← pow_succ']
      congr 2
      have := Finset.mem_range.mp hm
      omega
    · simp

/-- **Retention: parallel ≡ recurrent** (exact). -/
theorem retention_parallel_eq_recurrent {κ τ : Type*} [Fintype κ] (γ : ℝ) (q k : ℕ → κ → ℝ)
    (v : ℕ → τ → ℝ) (n : ℕ) : retRecurrent γ q k v n = retParallel γ q k v n := by
  rw [retRecurrent, retState_eq, Matrix.vecMul_sum, retParallel]
  refine Finset.sum_congr rfl (fun m _ => ?_)
  funext t
  simp only [Matrix.vecMul, Matrix.vecMulVec, dotProduct, Matrix.smul_apply, Matrix.of_apply,
    Pi.smul_apply, smul_eq_mul, Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl (fun l _ => ?_)
  ring

/-- The rotation rides in the state: the RoPE logit of a query rotated to `n` against a key
rotated to `m` is the same bilinear form of the unrotated vectors at relative angle `(n - m)ω`. -/
theorem rot_rides (ω n m : ℝ) (q k : ℝ × ℝ) :
    dot2 (rot (n * ω) q) (rot (m * ω) k) = dot2 q (rot ((m - n) * ω) k) := by
  simp only [dot2, rot, sub_mul, cos_sub, sin_sub]
  ring

end EP
