import EmulationProofs.Common
import EmulationProofs.Activation

/-!
# Round-2 features with no theorem before: kv_compress, hash_route, act_clip, clip_qkv

The relation's `FEATURES` table labels these walls (or free) with no proof. (`selfcond` and
`zamba_shared` are in `Round2Rec.lean`.)

* **kv_compress** (DeepSeek-V4 compressed KV pools): `pool_one` (a pool of one token is the
  token: ratio 1, which V4's code takes as "no compressor"); `kv_compress_not_plain` (no host
  whose logits are token-local reproduces the pooled weights).
* **hash_route** (DeepSeek-V4 `tid2eid`): `topk_code_select` and `code_margin` (a token-id code
  carried from the embedding selects exactly the hashed experts), `renorm_softmax_shift` (the code's
  common shift leaves a renormalised softmax gate exact); `hash_not_content` (a hash layer cannot
  select by content).
* **act_clip** (clamped SwiGLU) and **clip_qkv** (clamped q / k / v): `clampSym_id`,
  `act_clip_regime`, `act_clip_not_exact`, `clip_not_exact` (a clamped host is exact only in its
  regime); `analytic_ne_clamp`, `silu_analytic`, `swiglu_line_analytic`, `act_clip_unclamped_fails`
  (no unclamped SwiGLU FFN, any width, equals a clamped unit); `clip_qkv_plain_fails`.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## kv_compress -/

/-- **A pool of one token is the token** (exact): the softmax-gated pool over a single key is that
key. V4's code runs `max(1, compress_ratios[l])` and builds no compressor at ratio 1, so a
compressing host plays plain (windowed) attention by a config value. -/
theorem pool_one {ι V : Type*} [Fintype ι] [DecidableEq ι] [NormedAddCommGroup V]
    [NormedSpace ℝ V] (g : ι → ℝ) (k : ι → V) (j : ι) : attn {j} g k = k j := by
  simp [attn, smul_smul]

/-- The effective weight of token `j` in a two-pool compressed target (pools `{0,1}`, `{2,3}`, equal
gates, scalar keys, query `1`): its pool's softmax weight (pool logit = the pooled key) times its
share `1/2` in the pool. -/
def poolW (k : Fin 4 → ℝ) (j : Fin 4) : ℝ :=
  (if (j : ℕ) < 2 then exp ((k 0 + k 1) / 2) else exp ((k 2 + k 3) / 2)) /
    (2 * (exp ((k 0 + k 1) / 2) + exp ((k 2 + k 3) / 2)))

/-- A plain host whose logit for token `j` depends only on that token's content and position. -/
def localW (s : ℝ → Fin 4 → ℝ) (k : Fin 4 → ℝ) (j : Fin 4) : ℝ :=
  exp (s (k j) j) / ∑ l, exp (s (k l) l)

/-- **No token-local host reproduces pooled-KV attention** (the kv_compress wall, one host layer).
Changing token 1 alone changes the weight ratio between tokens 0 and 2 in the pooled target (token
0's pool key moves), but not in any host whose logits are token-local (any function of content and
position: plain attention with RoPE, ALiBi, any positional scheme, any keys). So no such host has
the target's attention weights on all inputs. Deeper hosts (an earlier layer writing pool
statistics into the stream) are not covered. -/
theorem kv_compress_not_plain : ¬ ∃ s : ℝ → Fin 4 → ℝ, ∀ k j, localW s k j = poolW k j := by
  rintro ⟨s, hs⟩
  have key : ∀ k : Fin 4 → ℝ, k 0 = 0 → k 2 = 0 →
      exp (s 0 0) * poolW k 2 = exp (s 0 2) * poolW k 0 := by
    intro k h0 h2
    have e0 := hs k 0
    have e2 := hs k 2
    simp only [localW, h0, h2] at e0 e2
    rw [← e0, ← e2]; ring
  have hA := key ![0, 0, 0, 0] rfl rfl
  have hB := key ![0, 1, 0, 0] rfl rfl
  have pA : poolW ![0, 0, 0, 0] 0 = poolW ![0, 0, 0, 0] 2 := by simp [poolW]
  have pB : poolW ![0, 1, 0, 0] 0 = exp (1 / 2) * poolW ![0, 1, 0, 0] 2 := by
    have : (0 : ℝ) < exp (1 / 2) := exp_pos _
    norm_num [poolW]
    field_simp
  have pos : ∀ k : Fin 4 → ℝ, 0 < poolW k 2 := fun k => by
    unfold poolW; split_ifs <;> positivity
  rw [pA] at hA
  have hab : exp (s 0 0) = exp (s 0 2) := mul_right_cancel₀ (pos _).ne' (by rw [hA])
  rw [pB, hab] at hB
  have h1 : exp (s 0 2) * poolW ![0, 1, 0, 0] 2 * (exp (1 / 2) - 1) = 0 := by linear_combination -hB
  have hne : exp (1 / 2 : ℝ) - 1 ≠ 0 := by
    have : (1 : ℝ) < exp (1 / 2) := Real.one_lt_exp_iff.mpr (by norm_num)
    linarith
  exact (mul_ne_zero (mul_ne_zero (exp_pos _).ne' (pos _).ne') hne) h1

/-! ## hash_route -/

/-- **A dominating code selects exactly the coded experts** (exact). If every coded expert scores
strictly above every uncoded one, any top-`|H|` selection (any tie-break) is `H`. -/
theorem topk_code_select {ι : Type*} [DecidableEq ι] (z : ι → ℝ) (H Sel : Finset ι)
    (hcard : Sel.card = H.card) (htop : ∀ a ∈ Sel, ∀ b ∉ Sel, z b ≤ z a)
    (hmargin : ∀ a ∈ H, ∀ b ∉ H, z b < z a) : Sel = H := by
  have hsub : H ⊆ Sel := by
    intro a ha
    by_contra haS
    have : ¬ Sel ⊆ H := by
      intro hSH
      have := Finset.eq_of_subset_of_card_le hSH (by omega)
      exact haS (this ▸ ha)
    obtain ⟨b, hbS, hbH⟩ := Finset.not_subset.mp this
    have h1 := htop b hbS a haS
    have h2 := hmargin a ha b hbH
    linarith
  exact (Finset.eq_of_subset_of_card_le hsub (by omega)).symm

/-- **The code margin**: router logits `w_e + λ·[e ∈ H]` with `|w_e| ≤ M` and `λ > 2M` put every
coded expert strictly above every uncoded one. The code is a `k`-hot row of extra embedding
channels that no sublayer writes (zero rows), read by the router; through the pre-norm it arrives
as `c/R` (`R` the token RMS, bounded on a bounded context), so `λ` is chosen for the smallest
`c/R`. Its fixed norm `k c²` is absorbed by the norms' ε (`const_norm_carry`, `k c² ≤ d ε`). -/
theorem code_margin {ι : Type*} [DecidableEq ι] (w : ι → ℝ) (H : Finset ι) (M lam : ℝ) (hw : ∀ e, |w e| ≤ M)
    (hl : 2 * M < lam) : ∀ a ∈ H, ∀ b ∉ H,
      w b + (if b ∈ H then lam else 0) < w a + (if a ∈ H then lam else 0) := by
  intro a ha b hb
  simp only [ha, hb, ite_true, ite_false, add_zero]
  have := abs_le.mp (hw a); have := abs_le.mp (hw b)
  linarith

/-- **The code's shift leaves a renormalised softmax gate exact**: the selected experts' logits all
move by the same `δ`, and a softmax renormalised over the selection does not see it. (A sigmoid or
`sqrtsoftplus` score renormalised over the selection does: e.g. `√softplus` at logits `(0, 1)` gives
weights 0.4208 / 0.5792, and at `(20, 21)` 0.4939 / 0.5061; so V4's own hash layers are not
reproduced this way.) -/
theorem renorm_softmax_shift {ι : Type*} (S : Finset ι) (z : ι → ℝ) (δ : ℝ) (e : ι) :
    exp (z e + δ) / ∑ e' ∈ S, exp (z e' + δ) = exp (z e) / ∑ e' ∈ S, exp (z e') := by
  simp only [exp_add, ← Finset.sum_mul]
  have := (exp_pos δ).ne'
  rw [mul_div_mul_right _ _ this]

/-- **A hash-routed layer cannot select by content** (the reverse wall, at a hash layer). Its
selection is a function of the token id; a target that routes the same token id to different
experts in two contexts is matched by no table. (V4 reads `num_hash_layers` from the config, so a
V4 host can switch hashing off: the host-side wall is a config switch.) -/
theorem hash_not_content {X ι β : Type*} (tid : X → ι) (sel : X → β) (x₁ x₂ : X)
    (htid : tid x₁ = tid x₂) (hsel : sel x₁ ≠ sel x₂) : ¬ ∃ tbl : ι → β, ∀ x, tbl (tid x) = sel x := by
  rintro ⟨tbl, h⟩
  exact hsel (by rw [← h x₁, ← h x₂, htid])

/-! ## act_clip and clip_qkv -/

/-- The symmetric clamp to `[-c, c]`. -/
def clampSym (c u : ℝ) : ℝ := max (-c) (min u c)

theorem clampSym_id {c u : ℝ} (hu : |u| ≤ c) : clampSym c u = u := by
  have := abs_le.mp hu
  simp only [clampSym, min_eq_left this.2, max_eq_right this.1]

/-- **A clamped host is exact in its regime** (act_clip): with `gate ≤ c` and `|up| ≤ c` the clamped
unit `silu(min(gate, c)) · clamp(up)` is the plain SwiGLU unit. `c` (DeepSeek-V4 `swiglu_limit`,
Step-3.5 `swiglu_limits`) is a config value, so a host sets it above every activation of a bounded
context, or to `None` where the code reads that as no clamp. -/
theorem act_clip_regime {c g u : ℝ} (hg : g ≤ c) (hu : |u| ≤ c) :
    silu (min g c) * clampSym c u = silu g * u := by
  rw [min_eq_left hg, clampSym_id hu]

/-- **…and only there**: for every limit `c > 0` the clamped unit differs from the plain one (at
`gate = c`, `up = c + 1`). -/
theorem act_clip_not_exact (c : ℝ) (hc : 0 < c) :
    silu (min c c) * clampSym c (c + 1) ≠ silu c * (c + 1) := by
  have hs : silu c ≠ 0 := by
    simp only [silu]; exact mul_ne_zero hc.ne' (sigmoid_pos c).ne'
  have : clampSym c (c + 1) = c := by
    simp only [clampSym, min_eq_right (by linarith : c ≤ c + 1), max_eq_right (by linarith : -c ≤ c)]
  rw [min_self, this]
  intro h
  have := mul_left_cancel₀ hs h
  linarith

/-- **clip_qkv, the host side**: a clamp is the identity on `[-c, c]` (a host sets `clip_qkv`
large, or to `None`, which OLMo-1 / MPT read as no clamp), and never the identity on all inputs. -/
theorem clip_not_exact (c : ℝ) : clampSym c (|c| + 1) ≠ |c| + 1 := by
  simp only [clampSym]
  intro h
  have h1 : min (|c| + 1) c ≤ c := min_le_right _ _
  have h2 : -c ≤ |c| := neg_le_abs c
  have h3 : c ≤ |c| := le_abs_self c
  rcases le_total (-c) (min (|c| + 1) c) with h4 | h4
  · rw [max_eq_right h4] at h; linarith
  · rw [max_eq_left h4] at h; linarith

/-- **No analytic function equals a scaled clamp** (κ ≠ 0, c > 0): it would equal `κ t` near `0`,
hence everywhere, but the clamp is `κ c ≠ 2κc` at `t = 2c`. -/
theorem analytic_ne_clamp (h : ℝ → ℝ) (hh : AnalyticOnNhd ℝ h Set.univ) (κ c : ℝ) (hκ : κ ≠ 0)
    (hc : 0 < c) : ¬ ∀ t, h t = κ * clampSym c t := by
  intro H
  have hev : h =ᶠ[𝓝 0] fun t => κ * t := by
    filter_upwards [Ioo_mem_nhds (by linarith : -c < 0) hc] with t ht
    rw [H, clampSym_id (abs_le.mpr ⟨ht.1.le, ht.2.le⟩)]
  have hlin : AnalyticOnNhd ℝ (fun t : ℝ => κ * t) Set.univ := fun t _ =>
    analyticAt_const.mul analyticAt_id
  have := AnalyticOnNhd.eqOn_of_preconnected_of_eventuallyEq hh hlin isPreconnected_univ
    (Set.mem_univ 0) hev (Set.mem_univ (2 * c))
  rw [H] at this
  have hcl : clampSym c (2 * c) = c := by
    simp only [clampSym, min_eq_right (by linarith : c ≤ 2 * c), max_eq_right (by linarith : -c ≤ c)]
  rw [hcl] at this
  have : κ * c = 0 := by linarith
  rcases mul_eq_zero.mp this with h1 | h1
  · exact hκ h1
  · linarith

lemma sigmoid_analytic (x : ℝ) : AnalyticAt ℝ sigmoid x := by
  have he : (fun y : ℝ => (1 + exp (-y))⁻¹) = sigmoid := by
    funext y; rw [sigmoid_def]
  rw [← he]
  refine (analyticAt_const.add (analyticAt_rexp.comp analyticAt_id.neg)).inv ?_
  have := exp_pos (-x)
  show 1 + exp (-x) ≠ 0; positivity

theorem silu_analytic (x : ℝ) : AnalyticAt ℝ silu x :=
  analyticAt_id.mul (sigmoid_analytic x)

/-- An unclamped SwiGLU FFN of any width, restricted to a line of inputs: unit `i` has gate and up
pre-activations affine in `t`. -/
def swigluLine {m : ℕ} (α a b p q : Fin m → ℝ) (t : ℝ) : ℝ :=
  ∑ i, α i * (silu (a i * t + b i) * (p i * t + q i))

theorem swiglu_line_analytic {m : ℕ} (α a b p q : Fin m → ℝ) :
    AnalyticOnNhd ℝ (swigluLine α a b p q) Set.univ := by
  intro t _
  unfold swigluLine
  refine Finset.analyticAt_fun_sum _ (fun i _ => analyticAt_const.mul ?_)
  refine AnalyticAt.mul ?_ ((analyticAt_const.mul analyticAt_id).add analyticAt_const)
  exact AnalyticAt.comp (g := silu) (silu_analytic _)
    ((analyticAt_const.mul analyticAt_id).add analyticAt_const)

/-- **An unclamped SwiGLU host cannot play a clamped unit** (act_clip wall, exact level). Along the
up-projection line of a target unit (gate fixed at `g ≤ c` with `silu g ≠ 0`), the target is
`silu(g) · clamp(t)`; no unclamped SwiGLU FFN of any width equals it. (Non-analytic hosts, e.g. a
ReLU MLP computing the clamp, are not covered.) -/
theorem act_clip_unclamped_fails {m : ℕ} (α a b p q : Fin m → ℝ) (g c : ℝ) (hc : 0 < c)
    (hg : g ≤ c) (hs : silu g ≠ 0) :
    ¬ ∀ t, swigluLine α a b p q t = silu (min g c) * clampSym c t := by
  rw [min_eq_left hg]
  exact analytic_ne_clamp _ (swiglu_line_analytic α a b p q) _ c hs hc

/-- **An unclamped host cannot play clip_qkv** (the wall, the direct construction). For a fixed
pair of keys, a plain host's logit difference is affine in the query-projection input; along a
line where the target's clamped query coordinate is `t`, the target's difference is
`κ · clamp(t)` (`κ` = the keys' difference on that coordinate, `≠ 0`). No affine function equals
it. -/
theorem clip_qkv_plain_fails (A B κ c : ℝ) (hκ : κ ≠ 0) (hc : 0 < c) :
    ¬ ∀ t, A * t + B = κ * clampSym c t :=
  analytic_ne_clamp _ (fun _ _ => (analyticAt_const.mul analyticAt_id).add analyticAt_const)
    κ c hκ hc

end EP
