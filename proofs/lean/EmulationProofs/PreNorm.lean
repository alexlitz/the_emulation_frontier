import EmulationProofs.Common

/-!
# Pre-norm failures: a shared RMSNorm rescales carried channels

Two constructions carry a vector `a` in extra residual channels next to the stream and read it
back linearly after the next pre-norm:

* **parallel ⊆ seq at 2× width:** the attention writes its output into a second channel, and the
  MLP must read the block INPUT `x` (the stream channel) through its pre-norm;
* **the CLA carry:** layer `l`'s shared KV is carried in extra channels, and layer `l + 1` must
  read it back through its pre-norm.

The target value read back (an MLP pre-activation, or a key coordinate) depends only on the
NORMED stream, so it does not change when the stream is scaled by `ρ > 0`; the carried vector
does not change either. The host's pre-norm divides by the RMS of the WHOLE residual
`ρ e + a`, which does change with `ρ` unless the carried part is collinear with the stream and
`ε = 0` (`prenorm_readout_obstruction`). So a disjoint-channel carry of anything nonzero is never
exact (`parallel_sub_seq_prenorm_fails`, `cla_carry_prenorm_fails`), and with `ε > 0` no layout
of the carry is exact (`prenorm_readout_eps_pos`).

What is quantified: ANY host width, ANY linear readout `c`, ANY norm weight `w`, ANY `ε ≥ 0`. What
is not: arbitrary nonlinear hosts whose earlier layers compute a compensating channel. The
statements refute the carry constructions, not every conceivable host.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-- **The pre-norm scale obstruction.** A host residual `ρ e + a`: the stream image `e` scaled by
`ρ`, plus a carried vector `a` that does not depend on `ρ`. If a linear readout `c` of the host's
RMSNorm (any weight `w`, any `ε ≥ 0`, any width `m`) returns the same nonzero value `t` for every
`ρ > 0`, then `|e|² (|a|² + m ε) = ⟨e, a⟩²`. By Cauchy–Schwarz this forces `ε = 0` and `a`
collinear with `e`. Proof: the squared identity is a quadratic in `ρ`; evaluating it at
`ρ = 1, 2, 3` gives its three coefficients. -/
theorem prenorm_readout_obstruction {m : ℕ} (hm : 0 < m) (eps : ℝ) (heps : 0 ≤ eps)
    (w c e a : Fin m → ℝ) (t : ℝ) (ht : t ≠ 0)
    (h : ∀ ρ : ℝ, 0 < ρ → ∑ i, c i * rmsNorm eps w (ρ • e + a) i = t) :
    (∑ i, e i ^ 2) * (∑ i, a i ^ 2 + m * eps) = (∑ i, e i * a i) ^ 2 := by
  set P := ∑ i, c i * w i * e i
  set Q := ∑ i, c i * w i * a i
  set E := ∑ i, e i ^ 2
  set A := ∑ i, a i ^ 2
  set X := ∑ i, e i * a i
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have key : ∀ ρ : ℝ, 0 < ρ →
      m * (ρ * P + Q) ^ 2 = t ^ 2 * (ρ ^ 2 * E + 2 * ρ * X + A + m * eps) := by
    intro ρ hρ
    have hρ' := h ρ hρ
    have hS : ∑ l, (ρ • e + a) l ^ 2 = ρ ^ 2 * E + 2 * ρ * X + A := by
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, E, X, A, Finset.mul_sum,
        ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl (fun i _ => by ring)
    set s := √((∑ l, (ρ • e + a) l ^ 2) / m + eps)
    have hsum : ∑ i, c i * rmsNorm eps w (ρ • e + a) i = (ρ * P + Q) / s := by
      have hi : ∀ i, c i * rmsNorm eps w (ρ • e + a) i =
          (ρ * (c i * w i * e i) + c i * w i * a i) / s := by
        intro i
        rw [show rmsNorm eps w (ρ • e + a) i = w i * (ρ • e + a) i / s from rfl]
        simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring
      rw [Finset.sum_congr rfl (fun i _ => hi i), ← Finset.sum_div, Finset.sum_add_distrib,
        ← Finset.mul_sum]
    rw [hsum] at hρ'
    have hs0 : s ≠ 0 := by
      intro h0; rw [h0, div_zero] at hρ'; exact ht hρ'.symm
    have hnn : 0 ≤ (∑ l, (ρ • e + a) l ^ 2) / m + eps := by positivity
    have heq : ρ * P + Q = t * s := by
      rw [div_eq_iff hs0] at hρ'; exact hρ'
    rw [heq, mul_pow, Real.sq_sqrt hnn, hS]
    field_simp
  have k1 := key 1 one_pos
  have k2 := key 2 two_pos
  have k3 := key 3 (by norm_num)
  have hα : m * P ^ 2 = t ^ 2 * E := by linear_combination (k1 - 2 * k2 + k3) / 2
  have hβ : m * P * Q = t ^ 2 * X := by linear_combination (-5 * k1 + 8 * k2 - 3 * k3) / 4
  have hγ : m * Q ^ 2 = t ^ 2 * (A + m * eps) := by linear_combination 3 * k1 - 3 * k2 + k3
  have h4 : t ^ 4 * (E * (A + m * eps)) = t ^ 4 * X ^ 2 := by
    linear_combination -(m * Q ^ 2) * hα - (t ^ 2 * E) * hγ + (m * P * Q + t ^ 2 * X) * hβ
  exact mul_left_cancel₀ (pow_ne_zero 4 ht) h4

lemma sum_sq_pos_of_ne_zero {m : ℕ} (u : Fin m → ℝ) (hu : u ≠ 0) : 0 < ∑ i, u i ^ 2 := by
  obtain ⟨i, hi⟩ : ∃ i, u i ≠ 0 := by
    by_contra h; exact hu (funext fun i => not_not.mp (fun hi => h ⟨i, hi⟩))
  exact Finset.sum_pos' (fun j _ => sq_nonneg (u j)) ⟨i, Finset.mem_univ _, by positivity⟩

/-- **With `ε > 0` no layout of the carry is exact.** Whatever the stream image `e ≠ 0` and the
carried vector `a` (orthogonal, collinear or anything else), no linear readout of a host RMSNorm
with `ε > 0` returns a fixed nonzero value for every scale `ρ > 0` of the stream. -/
theorem prenorm_readout_eps_pos {m : ℕ} (eps : ℝ) (heps : 0 < eps) (w c e a : Fin m → ℝ)
    (he : e ≠ 0) (t : ℝ) (ht : t ≠ 0) :
    ¬ ∀ ρ : ℝ, 0 < ρ → ∑ i, c i * rmsNorm eps w (ρ • e + a) i = t := by
  intro h
  have hm : 0 < m := by
    rcases Nat.eq_zero_or_pos m with hm | hm
    · subst hm; exact absurd (funext fun i => i.elim0) he
    · exact hm
  have hob := prenorm_readout_obstruction hm eps heps.le w c e a t ht h
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ e a
  have hE := sum_sq_pos_of_ne_zero e he
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have : 0 < (∑ i, e i ^ 2) * (m * eps) := by positivity
  nlinarith

/-- The disjoint-channel layout: the residual `[ρ u; v]` is `ρ [u; 0] + [0; v]`. -/
lemma append_smul_split {d : ℕ} (ρ : ℝ) (u v : Fin d → ℝ) :
    Fin.append (ρ • u) v = ρ • Fin.append u (0 : Fin d → ℝ) + Fin.append (0 : Fin d → ℝ) v := by
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · simp
  · simp only [Fin.append_right, Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]
    ring

/-- **Disjoint-channel carries are never exact under a pre-norm.** A host residual `[ρ u; v]`
(the stream `ρ u` in its own channels, a nonzero carried vector `v` in separate channels): no
linear readout `c` of the host RMSNorm (any weight `w`, any `ε ≥ 0`) returns a fixed nonzero
value for every `ρ > 0`. -/
theorem channel_carry_prenorm_fails {d : ℕ} (eps : ℝ) (heps : 0 ≤ eps) (w c : Fin (d + d) → ℝ)
    (u v : Fin d → ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (t : ℝ) (ht : t ≠ 0) :
    ¬ ∀ ρ : ℝ, 0 < ρ → ∑ i, c i * rmsNorm eps w (Fin.append (ρ • u) v) i = t := by
  intro h
  have hd : 0 < d + d := by
    rcases Nat.eq_zero_or_pos d with hd | hd
    · subst hd; exact absurd (funext fun i => i.elim0) hu
    · omega
  simp only [append_smul_split] at h
  have hob := prenorm_readout_obstruction hd eps heps w c _ _ t ht h
  simp only [Fin.sum_univ_add, Fin.append_left, Fin.append_right, Pi.zero_apply, mul_zero,
    zero_mul, Finset.sum_const_zero, add_zero, zero_add, ne_eq, OfNat.ofNat_ne_zero,
    not_false_eq_true, zero_pow] at hob
  have hU := sum_sq_pos_of_ne_zero u hu
  have hV := sum_sq_pos_of_ne_zero v hv
  have : 0 ≤ ((d + d : ℕ) : ℝ) * eps := by positivity
  have : 0 < (∑ i, u i ^ 2) * (∑ i, v i ^ 2 + ((d + d : ℕ) : ℝ) * eps) := by positivity
  linarith

/-- **parallel ⊆ seq at 2× width fails under a pre-norm.** In the 2×-width construction the host
residual is `[x; A(N x)]`: the stream channel holds the block input `x`, the second channel the
attention's write. The target MLP reads `W₁ N(x)`; with the target's `ε = 0` this does not change
when `x = ρ u` is scaled, and neither does the attention's output `A(N x)`. If the attention
writes anything (`v = A(N u) ≠ 0`) and some pre-activation `t` is nonzero, no linear readout of
the host MLP's pre-norm (any weight, any `ε ≥ 0`) reproduces `t` on the inputs `ρ u`, `ρ > 0`
(three of them suffice). The norm rescales the carried channel. -/
theorem parallel_sub_seq_prenorm_fails {d : ℕ} (eps : ℝ) (heps : 0 ≤ eps)
    (w c : Fin (d + d) → ℝ) (u attnOut : Fin d → ℝ) (hu : u ≠ 0) (hv : attnOut ≠ 0) (t : ℝ)
    (ht : t ≠ 0) :
    ¬ ∀ ρ : ℝ, 0 < ρ → ∑ i, c i * rmsNorm eps w (Fin.append (ρ • u) attnOut) i = t :=
  channel_carry_prenorm_fails eps heps w c u attnOut hu hv t ht

/-- **The CLA carry is not exact under a pre-norm.** The host carries layer `l`'s shared KV
`κ` (nonzero) in separate channels next to the residual stream, and layer `l + 1` reads its key
back linearly through its pre-norm. Take a target whose layer `l` writes nothing (zero output
projections), so layer `l + 1`'s residual is `ρ u` on the ray of inputs `x_l = ρ u`, while the
carried `κ = W_K N(x_l)` (target `ε = 0`) and the key coordinate `t` it should read back stay
fixed. No linear readout, norm weight or `ε ≥ 0` returns `t ≠ 0` for every `ρ > 0`. -/
theorem cla_carry_prenorm_fails {d : ℕ} (eps : ℝ) (heps : 0 ≤ eps) (w c : Fin (d + d) → ℝ)
    (u κ : Fin d → ℝ) (hu : u ≠ 0) (hκ : κ ≠ 0) (t : ℝ) (ht : t ≠ 0) :
    ¬ ∀ ρ : ℝ, 0 < ρ → ∑ i, c i * rmsNorm eps w (Fin.append (ρ • u) κ) i = t :=
  channel_carry_prenorm_fails eps heps w c u κ hu hκ t ht

end EP
