import EmulationProofs.EpsLever
import EmulationProofs.AuditRules
import EmulationProofs.Round2Rec

/-!
# fr-mixer: the mixer / recurrence walls paid (model_subsets branch `fr-mixer`, `dist/fr_mixer.md`)

* **pure_on_hybrid** (exact). A pure recurrent stack runs on a hybrid host: target layer `i` on the
  host's `i`-th recurrent layer, every other host layer the identity (`zeroed_sublayer`: its output
  projections zero). `FrmPadded t h` says `h` is `t` with identity layers inserted (an order-preserving
  embedding); `padded_run` — the host stack computes the target stack. `frmAssign ks t` places a target
  stack on a host with layer kinds `ks` (`true` = recurrent) and `assign_padded` / `assign_run`: it is a
  padding whenever the host has at least as many recurrent layers as the target has layers; its
  non-recurrent positions are the identity (`assign_off`). Every target layer has one kind, so the
  layer COUNT is the whole condition (for a mixed target the order would matter: a count is not an
  order-preserving embedding).
* **m2_nonorm_lever** (limit). The residual stream × `S` with config eps `ε S²` keeps every stream
  norm exactly (`stream_norm_residual`, from `rmsNorm_smul_eps`), and a norm fed an `O(1)` input
  against eps `ε S²` with gain `S √ε` IS the eps lever (`gated_norm_residual_lever`: equal to
  `epsLeverNorm ε S`), so it tends to the identity, within `‖w‖ ‖v‖³ / (2 S² ε)`
  (`gated_norm_residual_bound`). A norm-free Mamba-2 target's `y ⊙ silu z` is such an input.
* **mamba1_lever** (limit). Two weight-free norms fed `x / S` (Falcon-Mamba's `B`, `C`): their dot
  product times `S² ε` is `f_B f_C (B · C)` (`weightFree_lever_dot`), `f = 1/√(1+u) ∈ (0, 1]`, and
  `|f_B f_C − 1| ≤ (u_B + u_C)/2` (`lever_pair_factor`): the constant `1/(S² ε)` folds into `D` and
  `out_proj`.
* **refuted idea**: "GDN with β = 0 is linear attention" — the delta rule's one `β` gates both the
  erase and the write, so `β = 0` writes nothing (`deltaRow_beta_zero`).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## pure_on_hybrid -/

section stack

variable {X : Type*}

/-- Run a stack of layers, the head first. -/
def frmRun (l : List (X → X)) (x : X) : X := l.foldl (fun y f => f y) x

/-- `h` is `t` with identity layers inserted (an order-preserving embedding of `t` into `h`). -/
inductive FrmPadded : List (X → X) → List (X → X) → Prop
  | nil : FrmPadded [] []
  | keep (f : X → X) {t h : List (X → X)} : FrmPadded t h → FrmPadded (f :: t) (f :: h)
  | skip {t h : List (X → X)} : FrmPadded t h → FrmPadded t (id :: h)

/-- **A padded host stack computes the target stack** (exact). -/
theorem padded_run {t h : List (X → X)} (hp : FrmPadded t h) (x : X) : frmRun h x = frmRun t x := by
  induction hp generalizing x with
  | nil => rfl
  | keep f _ ih => simpa [frmRun] using ih (f x)
  | skip _ ih => simpa [frmRun] using ih x

/-- The host stack of layer kinds `ks` (`true` = a recurrent layer) playing the pure target `t`:
target layers on the recurrent layers in order, the identity everywhere else. -/
def frmAssign : List Bool → List (X → X) → List (X → X)
  | [], _ => []
  | false :: ks, t => id :: frmAssign ks t
  | true :: ks, [] => id :: frmAssign ks []
  | true :: ks, f :: t => f :: frmAssign ks t

lemma frm_padded_ids (ks : List Bool) : FrmPadded ([] : List (X → X)) (frmAssign ks []) := by
  induction ks with
  | nil => exact FrmPadded.nil
  | cons b ks ih => cases b <;> exact FrmPadded.skip ih

/-- **…it is a padding** when the host has at least as many recurrent layers as the target has
layers. -/
theorem assign_padded : ∀ (ks : List Bool) (t : List (X → X)), t.length ≤ ks.count true →
    FrmPadded t (frmAssign ks t)
  | [], [], _ => FrmPadded.nil
  | [], _ :: _, h => by simp at h
  | false :: ks, t, h => FrmPadded.skip (assign_padded ks t (by simpa using h))
  | true :: ks, [], _ => FrmPadded.skip (frm_padded_ids ks)
  | true :: ks, f :: t, h => FrmPadded.keep f (assign_padded ks t (by simp at h; omega))

/-- **pure_on_hybrid** (exact): the host stack computes the pure target stack. -/
theorem assign_run (ks : List Bool) (t : List (X → X)) (h : t.length ≤ ks.count true) (x : X) :
    frmRun (frmAssign ks t) x = frmRun t x :=
  padded_run (assign_padded ks t h) x

/-- …and every non-recurrent host layer is the identity (its attention / FFN output zeroed). -/
theorem assign_off : ∀ (ks : List Bool) (t : List (X → X)) (i : ℕ) (hi : i < (frmAssign ks t).length),
    ks[i]? = some false → (frmAssign ks t)[i] = id
  | [], _, i, hi, _ => by simp [frmAssign] at hi
  | false :: ks, t, 0, _, _ => rfl
  | false :: ks, t, i + 1, hi, h => by
      simp only [frmAssign, List.getElem_cons_succ]
      exact assign_off ks t i (by simpa [frmAssign] using hi) (by simpa using h)
  | true :: ks, _, 0, _, h => by simp at h
  | true :: ks, [], i + 1, hi, h => by
      simp only [frmAssign, List.getElem_cons_succ]
      exact assign_off ks [] i (by simpa [frmAssign] using hi) (by simpa using h)
  | true :: ks, f :: t, i + 1, hi, h => by
      simp only [frmAssign, List.getElem_cons_succ]
      exact assign_off ks t i (by simpa [frmAssign] using hi) (by simpa using h)

end stack

/-! ## m2_nonorm_lever: the residual-scaled eps lever -/

section lever

variable {d : ℕ}

/-- **Stream norms are exact under the residual scaling**: `N_{ε S²}(w, S x) = N_ε(w, x)`. -/
theorem stream_norm_residual (eps : ℝ) {S : ℝ} (hS : 0 < S) (w x : Fin d → ℝ) :
    rmsNorm (eps * S ^ 2) w (S • x) = rmsNorm eps w x := by
  rw [rmsNorm_smul_eps _ _ hS]
  congr 1
  field_simp

lemma rmsNorm_smul_weight (eps c : ℝ) (w x : Fin d → ℝ) (i : Fin d) :
    rmsNorm eps (fun l => c * w l) x i = c * rmsNorm eps w x i := by
  simp only [rmsNorm]; ring

/-- **The gated norm under the residual scaling IS the eps lever**: with eps `ε S²` and gain
`S √ε · w`, the norm of an unscaled `v` is `epsLeverNorm ε S w v`. -/
theorem gated_norm_residual_lever (eps : ℝ) {S : ℝ} (hS : 0 < S) (w v : Fin d → ℝ) :
    rmsNorm (eps * S ^ 2) (fun l => S * √eps * w l) v = epsLeverNorm eps S w v := by
  funext i
  rw [rmsNorm_smul_weight]
  unfold epsLeverNorm
  congr 1
  have h1 : (fun l => v l / S) = (1 / S) • v := by funext l; simp [div_eq_inv_mul]
  rw [h1, rmsNorm_smul_eps _ _ (by positivity)]
  congr 2
  field_simp

/-- **…so it tends to the identity** (the norm-free target), within `‖w‖ ‖v‖³ / (2 S² ε)`. -/
theorem gated_norm_residual_bound (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : 0 < S)
    (w v : Fin d → ℝ) :
    ‖rmsNorm (eps * S ^ 2) (fun l => S * √eps * w l) v - fun i => w i * v i‖ ≤
      ‖w‖ * ‖v‖ ^ 3 / (2 * S ^ 2 * eps) := by
  rw [gated_norm_residual_lever eps hS]
  exact epsLever_bound hd heps hS.ne' w v

/-! ## mamba1_lever: two weight-free norms, their dot product -/

/-- A weight-free levered norm is the lever factor times `x / (S √ε)`. -/
lemma weightFree_lever (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : S ≠ 0)
    (x : Fin d → ℝ) (i : Fin d) :
    rmsNorm eps (fun _ => 1) (fun l => x l / S) i = leverFactor eps S x * x i / (S * √eps) := by
  have h := epsLever_eq hd heps hS (fun _ => 1) x i
  unfold epsLeverNorm at h
  have hs : S * √eps ≠ 0 := mul_ne_zero hS (Real.sqrt_pos.mpr heps).ne'
  rw [eq_div_iff hs, mul_comm]
  simpa using h

/-- **The readout `C · B` of two levered weight-free norms** (Falcon-Mamba's `B`, `C`), times
`S² ε`, is `f_B f_C (B · C)`. -/
theorem weightFree_lever_dot (hd : d ≠ 0) {eps : ℝ} (heps : 0 < eps) {S : ℝ} (hS : S ≠ 0)
    (B C : Fin d → ℝ) :
    S ^ 2 * eps * ∑ i, rmsNorm eps (fun _ => 1) (fun l => B l / S) i *
        rmsNorm eps (fun _ => 1) (fun l => C l / S) i =
      leverFactor eps S B * leverFactor eps S C * ∑ i, B i * C i := by
  simp_rw [weightFree_lever hd heps hS, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  have h1 : 0 < √eps := Real.sqrt_pos.mpr heps
  have h2 : √eps ^ 2 = eps := Real.sq_sqrt heps.le
  field_simp
  rw [h2]; ring

/-- `|f_B f_C − 1| ≤ (u_B + u_C)/2`: the product of the two lever factors tends to 1. -/
theorem lever_pair_factor {eps : ℝ} (heps : 0 ≤ eps) (S : ℝ) (B C : Fin d → ℝ) :
    |leverFactor eps S B * leverFactor eps S C - 1| ≤ (leverU eps S B + leverU eps S C) / 2 := by
  have a1 := leverFactor_pos eps S heps B; have a2 := leverFactor_le_one eps S heps B
  have b1 := leverFactor_pos eps S heps C; have b2 := leverFactor_le_one eps S heps C
  have ca := one_sub_leverFactor_le eps S heps B; have cb := one_sub_leverFactor_le eps S heps C
  have hle : leverFactor eps S B * leverFactor eps S C ≤ 1 := by nlinarith
  rw [abs_of_nonpos (by linarith)]
  nlinarith

end lever

/-! ## the refuted idea: GDN with β = 0 -/

/-- **The delta rule with `β = 0` writes nothing** (so it is not GLA / linear attention): the erase
and the write share the one `β`. -/
theorem deltaRow_beta_zero {dk : ℕ} (a : ℕ → Fin dk → ℝ) (k : ℕ → Fin dk → ℝ) (v : ℕ → ℝ) :
    ∀ t, deltaRow a (fun _ => 0) k v t = 0 := by
  intro t; induction t with
  | zero => rfl
  | succ t ih => simp [deltaRow, ih]

end EP
