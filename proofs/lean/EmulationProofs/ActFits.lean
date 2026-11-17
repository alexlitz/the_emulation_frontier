import EmulationProofs.ActLimits

/-!
# Certified multi-unit activation fits

The activation-width rescue (branch `act-width-fits`, `act/data/act_width_fits.json`) fitted
target activations by `k` host units `Σᵢ αᵢ φ(βᵢ a + γᵢ)`. Their sup errors are float64 grid
maxima. This file certifies one of them as a theorem over the reals:

* **mish ≈ 6 SiLU units on `[−6, 6]`** (`"mish|SiLU|6|shift"`, `k = 6`, float error `1.68e-5`):
  with the coefficients rounded to dyadic rationals (`α, β` to `2^-24`, `γ` to `2^-40`; the
  rounded fit's float sup error is `1.80e-5`),
  `|mish a − Σᵢ αᵢ SiLU(βᵢ a + γᵢ)| ≤ 4·10⁻⁵` for every real `a ∈ [−6, 6]`
  (`actFit_mish_silu6`, `actFit_mish_silu6_explicit`).

**Method** (a verified interval-arithmetic checker, evaluated by the kernel with `decide +kernel`).
* `exp` enclosures in fixed point (scale `2^64`): Taylor polynomial with 16 terms (floor /
  ceiling rounding) plus the remainder of `Real.exp_bound'`, after the range reduction
  `eˣ = (e^{x/16})^16` (`actFit_expEnc`).
* On a grid of step `2^-8` (3072 cells), the exponentials at the grid points are carried by the
  recurrence `e^{x + βh} = eˣ·e^{βh}` with outward rounding (`actFit_stepE`).
* On each cell `[p, p + h]` the mean value theorem gives `e(a) = e(p) + e'(ξ)(a − p)`, with
  `e = mish − fit`. `e(p)` is enclosed from the point enclosures and `e'(ξ)` by interval
  arithmetic over the cell, using `SiLU' = σ(1 + x(1 − σ))`, `mish a = a·g(eᵃ)` with
  `g(t) = (t² + 2t)/(t² + 2t + 2)` (`actFit_mish_eq`), and `mish' = g(eᵃ) + a·4t(t+1)/(t²+2t+2)²`.
  The interval products are exact; only the `exp`, `σ`, `g`, `h` quotients round (outwards).
* Signed numbers are pairs of naturals (value `a.1 − a.2`): the kernel's GMP acceleration covers
  `Nat` only, and kernel `Int` arithmetic was ~10–50× slower here.
* `actFit_chunk` turns a passing chunk (`afChunkOK = true`) into the real statement; the eight
  `actFit_groupᵢ` (12 chunks of 32 cells each) are checked by `decide +kernel`, about 60 s in all.

**Not certified, and why.** Fits whose target or host is GeLU (`GeLU ≈ Σ αᵢ SiLU(βᵢ a)`, `k = 6`,
`3.3e-4` on `[−160, 160]`) need enclosures of the Gaussian CDF `Φ`, which Mathlib defines only as a
measure-theoretic `cdf` with no computable bounds; a certificate would first need `Φ` enclosures
(the alternating series of `∫₀ˣ e^{−t²/2}`, termwise integration, and Mills-ratio tails for large
`|x|`), then the same checker with `Φ`, `Φ'` in place of `σ`. The `[−160, 160]` fits need about
27× more cells, or tail arguments where the units saturate (`|βᵢ a + γᵢ|` large), plus
cancellation control for coefficients up to ~200. Neither is done here.
-/

namespace EP

/-! ## The checker (computable, kernel-evaluated) -/

/-- The fixed-point scale `2^64`. -/
def afS : ℕ := 18446744073709551616

/-- Ceiling division. -/
def afCdiv (a b : ℕ) : ℕ := Nat.div (Nat.sub (Nat.add a b) 1) b

/-- Lower Taylor sum: terms `t_{i+1} = ⌊⌊t_i Y / S⌋/(k+i+1)⌋`. -/
def afTayLo (Y : ℕ) : ℕ → ℕ → ℕ → ℕ → ℕ
  | 0, _, _, acc => acc
  | n+1, k, t, acc =>
      afTayLo Y n (Nat.add k 1) (Nat.div (Nat.div (Nat.mul t Y) afS) (Nat.add k 1)) (Nat.add acc t)

/-- Upper Taylor sum, plus the remainder `17/16 ×` the next term. -/
def afTayHi (Y : ℕ) : ℕ → ℕ → ℕ → ℕ → ℕ
  | 0, _, u, acc => Nat.add acc (afCdiv (Nat.mul u 17) 16)
  | n+1, k, u, acc =>
      afTayHi Y n (Nat.add k 1) (afCdiv (afCdiv (Nat.mul u Y) afS) (Nat.add k 1)) (Nat.add acc u)

def afSqLo (x : ℕ) : ℕ := Nat.div (Nat.mul x x) afS
def afSqHi (x : ℕ) : ℕ := afCdiv (Nat.mul x x) afS

/-- Enclosure of `S·e^{16 Y/S}`. -/
def afExpPos (Y : ℕ) : ℕ × ℕ :=
  (afSqLo (afSqLo (afSqLo (afSqLo (afTayLo Y 16 0 afS 0)))),
   afSqHi (afSqHi (afSqHi (afSqHi (afTayHi Y 16 0 afS 0)))))

/-- Signed numbers as pairs of naturals, value `a.1 − a.2` (the kernel is much faster on `ℕ`
than on `Int`). -/
abbrev AfZ := ℕ × ℕ

def afZN (a : AfZ) : AfZ := if Nat.ble a.2 a.1 then (Nat.sub a.1 a.2, 0) else (0, Nat.sub a.2 a.1)
def afZAdd (a b : AfZ) : AfZ := (Nat.add a.1 b.1, Nat.add a.2 b.2)
def afZMul (a b : AfZ) : AfZ :=
  afZN (Nat.add (Nat.mul a.1 b.1) (Nat.mul a.2 b.2), Nat.add (Nat.mul a.1 b.2) (Nat.mul a.2 b.1))
def afZNeg (a : AfZ) : AfZ := (a.2, a.1)
def afZLe (a b : AfZ) : Bool := Nat.ble (Nat.add a.1 b.2) (Nat.add b.1 a.2)
def afZNonpos (a : AfZ) : Bool := Nat.ble a.1 a.2
def afZMin (a b : AfZ) : AfZ := if afZLe a b then a else b
def afZMax (a b : AfZ) : AfZ := if afZLe a b then b else a

/-- Enclosure of `S·exp (X / 2^40)` (valid for `|X| ≤ 2^44`). -/
def afExpEnc (X : AfZ) : ℕ × ℕ :=
  if Nat.ble X.2 X.1 then afExpPos (Nat.mul (Nat.sub X.1 X.2) 1048576)
  else (Nat.div (Nat.mul afS afS) (afExpPos (Nat.mul (Nat.sub X.2 X.1) 1048576)).2,
        afCdiv (Nat.mul afS afS) (afExpPos (Nat.mul (Nat.sub X.2 X.1) 1048576)).1)

def afExpOK (X : AfZ) : Bool :=
  Nat.ble (Nat.mul X.1 1048576) (Nat.add afS (Nat.mul X.2 1048576)) &&
  Nat.ble (Nat.mul X.2 1048576) (Nat.add afS (Nat.mul X.1 1048576))

def afSigLo (t : ℕ) : ℕ := Nat.div (Nat.mul t afS) (Nat.add afS t)
def afSigHi (t : ℕ) : ℕ := afCdiv (Nat.mul t afS) (Nat.add afS t)
def afGN (t : ℕ) : ℕ := Nat.add (Nat.mul t t) (Nat.mul (Nat.mul 2 t) afS)
def afGD (t : ℕ) : ℕ := Nat.add (afGN t) (Nat.mul (Nat.mul 2 afS) afS)
def afGLo (t : ℕ) : ℕ := Nat.div (Nat.mul (afGN t) afS) (afGD t)
def afGHi (t : ℕ) : ℕ := afCdiv (Nat.mul (afGN t) afS) (afGD t)
def afHN (t : ℕ) : ℕ := Nat.mul (Nat.mul (Nat.mul 4 t) (Nat.add t afS)) (Nat.mul afS afS)
def afHLo (tl tu : ℕ) : ℕ := Nat.div (Nat.mul (afHN tl) afS) (Nat.mul (afGD tu) (afGD tu))
def afHHi (tu tl : ℕ) : ℕ := afCdiv (Nat.mul (afHN tu) afS) (Nat.mul (afGD tl) (afGD tl))

/-- Intervals of signed numbers. -/
abbrev AfI := AfZ × AfZ

/-- Interval product (exact). -/
def afMul (I J : AfI) : AfI :=
  (afZMin (afZMin (afZMul I.1 J.1) (afZMul I.1 J.2)) (afZMin (afZMul I.2 J.1) (afZMul I.2 J.2)),
   afZMax (afZMax (afZMul I.1 J.1) (afZMul I.1 J.2)) (afZMax (afZMul I.2 J.1) (afZMul I.2 J.2)))

/-- Product of a nonnegative interval `I` and an interval `J`. -/
def afMulP (I J : AfI) : AfI :=
  (if afZNonpos (afZNeg J.1) then afZMul I.1 J.1 else afZMul I.2 J.1,
   if afZNonpos (afZNeg J.2) then afZMul I.2 J.2 else afZMul I.1 J.2)

/-- Product of a number `c` and an interval `J`. -/
def afScale (c : AfZ) (J : AfI) : AfI :=
  if afZNonpos (afZNeg c) then (afZMul c J.1, afZMul c J.2) else (afZMul c J.2, afZMul c J.1)

def afAdd (I J : AfI) : AfI := (afZAdd I.1 J.1, afZAdd I.2 J.2)
def afPt (x : AfZ) : AfI := (x, x)
def afOfN (I : ℕ × ℕ) : AfI := ((I.1, 0), (I.2, 0))
def afP2 (k : ℕ) : AfZ := (Nat.pow 2 k, 0)

/-- A unit `α·SiLU(β a + γ)` with `α = A/2^24`, `β = B/2^24`, `γ = G/2^40`. -/
structure AfUnit where
  A : AfZ
  B : AfZ
  G : AfZ

/-- `2^40 · x(a)` at the grid point `a = p/2^16`. -/
def AfUnit.X (u : AfUnit) (p : AfZ) : AfZ := afZAdd (afZMul u.B p) u.G

/-- The derivative part for one unit: an enclosure of `−αβ SiLU'(x)·2^216` for every
`x ∈ [xl, xu]/2^40`, from `El.1 ≤ S e^{xl}` and `S e^{xu} ≤ Eu.2`. -/
def afDerE (u : AfUnit) (xl xu : AfZ) (El Eu : ℕ × ℕ) : AfI :=
  afScale (afZNeg (afZMul u.A u.B))
    (afMulP (afOfN (afSigLo El.1, afSigHi Eu.2))
      (afAdd (afMulP ((afS, afSigHi Eu.2), (afS, afSigLo El.1)) (xl, xu)) (afPt (afP2 104))))

/-- For one unit on the cell `[p, q]`, from the exp enclosures at both ends: an enclosure of
`−α SiLU(x_p)·2^128` and one of `−αβ SiLU'(x(ξ))·2^216` for every `ξ` in the cell. -/
def afUnitE (u : AfUnit) (p q : AfZ) (Ep Eq : ℕ × ℕ) : AfI × AfI :=
  (afScale (afZNeg u.A) (afScale (u.X p) (afOfN (afSigLo Ep.1, afSigHi Ep.2))),
   afDerE u (if afZNonpos u.B then u.X q else u.X p) (if afZNonpos u.B then u.X p else u.X q)
     (if afZNonpos u.B then Eq else Ep) (if afZNonpos u.B then Ep else Eq))

def afSumE : List AfUnit → AfZ → AfZ → List (ℕ × ℕ) → List (ℕ × ℕ) → AfI × AfI
  | u :: us, p, q, Ep :: Eps, Eq :: Eqs =>
      (afAdd (afUnitE u p q Ep Eq).1 (afSumE us p q Eps Eqs).1,
       afAdd (afUnitE u p q Ep Eq).2 (afSumE us p q Eps Eqs).2)
  | _, _, _, _, _ => (((0, 0), (0, 0)), ((0, 0), (0, 0)))

/-- The mish part: `mish(p)·2^128` and `mish'(ξ)·2^216` on the cell. -/
def afMishE (p q : AfZ) (Tp Tq : ℕ × ℕ) : AfI × AfI :=
  (afScale (afZMul p (afP2 48)) (afOfN (afGLo Tp.1, afGHi Tp.2)),
   afScale (afP2 136) (afAdd (afMulP (afOfN (afHLo Tp.1 Tq.2, afHHi Tq.2 Tp.1)) (p, q))
     (afScale (afP2 16) (afOfN (afGLo Tp.1, afGHi Tq.2)))))

/-- The state at a grid point `p/2^16`: exp enclosures of `eᵖ` and of every unit's `e^{x(p)}`. -/
structure AfSt where
  p : AfZ
  m : ℕ × ℕ
  ts : List (ℕ × ℕ)

def afStepE (t M : ℕ × ℕ) : ℕ × ℕ := (Nat.div (Nat.mul t.1 M.1) afS, afCdiv (Nat.mul t.2 M.2) afS)

def afStepAll : List (ℕ × ℕ) → List (ℕ × ℕ) → List (ℕ × ℕ)
  | t :: ts, M :: Ms => afStepE t M :: afStepAll ts Ms
  | _, _ => []

def afNext (H : ℕ) (Mm : ℕ × ℕ) (Ms : List (ℕ × ℕ)) (s : AfSt) : AfSt :=
  ⟨afZAdd s.p (H, 0), afStepE s.m Mm, afStepAll s.ts Ms⟩

/-- The cell check: `2^232·e(a) ∈ [−E, E]` on `[p, p + H]`, via `e(a) = e(p) + e'(ξ)(a − p)`. -/
def afCellOK (us : List AfUnit) (E H : ℕ) (s s' : AfSt) : Bool :=
  afZLe (afZAdd (afZMul (afAdd (afMishE s.p s'.p s.m s'.m).1 (afSumE us s.p s'.p s.ts s'.ts).1).2
      (afP2 104))
    (afZMul (afZMax (0, 0) (afAdd (afMishE s.p s'.p s.m s'.m).2 (afSumE us s.p s'.p s.ts s'.ts).2).2)
      (H, 0))) (E, 0) &&
  afZLe (0, E)
    (afZAdd (afZMul (afAdd (afMishE s.p s'.p s.m s'.m).1 (afSumE us s.p s'.p s.ts s'.ts).1).1
      (afP2 104))
    (afZMul (afZMin (0, 0) (afAdd (afMishE s.p s'.p s.m s'.m).2 (afSumE us s.p s'.p s.ts s'.ts).2).1)
      (H, 0)))

def afRun (us : List AfUnit) (E H : ℕ) (Mm : ℕ × ℕ) (Ms : List (ℕ × ℕ)) : ℕ → AfSt → Bool
  | 0, _ => true
  | n+1, s => afCellOK us E H s (afNext H Mm Ms s) && afRun us E H Mm Ms n (afNext H Mm Ms s)

/-- One chunk of `n` cells of width `H/2^16` from `a0/2^16`. -/
def afChunkOK (us : List AfUnit) (E H : ℕ) (a0 : AfZ) (n : ℕ) : Bool :=
  afExpOK (afZMul a0 (afP2 24)) && afExpOK (Nat.mul H 16777216, 0) &&
  us.all (fun u => afExpOK (u.X a0) && afExpOK (afZMul u.B (H, 0))) &&
  afRun us E H (afExpEnc (Nat.mul H 16777216, 0)) (us.map (fun u => afExpEnc (afZMul u.B (H, 0)))) n
    ⟨a0, afExpEnc (afZMul a0 (afP2 24)), us.map (fun u => afExpEnc (u.X a0))⟩

def afAllBelow (f : ℕ → Bool) : ℕ → Bool
  | 0 => true
  | n+1 => f n && afAllBelow f n

/-- The fit `"mish|SiLU|6|shift"`, `k = 6`, coefficients rounded to `A/2^24, B/2^24, G/2^40`. -/
def afFit6 : List AfUnit :=
  [⟨(0, 160505003), (0, 3020478), (20728877624, 0)⟩, ⟨(17708581, 0), (0, 16340396), (190579807737, 0)⟩,
   ⟨(172903600, 0), (2903133, 0), (0, 67410679846)⟩, ⟨(0, 10714356), (19621551, 0), (0, 988909254425)⟩,
   ⟨(8567841, 0), (0, 22279116), (244274895690, 0)⟩, ⟨(314572, 0), (0, 16954119), (4129126169491, 0)⟩]

/-- `⌊2^232 / 25000⌋`. -/
def afE6 : ℕ := 276069853871622551497390234491081018098044358886815462206500968951

/-! ## Soundness of the checker -/

section sound

open Real Finset

lemma actFit_S : (afS : ℝ) = 2 ^ 64 := by norm_num [afS]

lemma actFit_S_pos : (0 : ℝ) < afS := by rw [actFit_S]; positivity

lemma actFit_div_le (a b : ℕ) : ((Nat.div a b : ℕ) : ℝ) ≤ (a : ℝ) / b := Nat.cast_div_le

lemma actFit_cdiv_mul (a b : ℕ) (hb : 0 < b) : a ≤ afCdiv a b * b := by
  unfold afCdiv
  have h1 := Nat.div_add_mod (Nat.sub (Nat.add a b) 1) b
  have h2 := Nat.mod_lt (Nat.sub (Nat.add a b) 1) hb
  simp only [Nat.sub_eq, Nat.add_eq] at h1 h2 ⊢
  rw [Nat.mul_comm] at h1
  have : Nat.div (a + b - 1) b = (a + b - 1) / b := rfl
  rw [this]
  omega

lemma actFit_cdiv_ge (a b : ℕ) (hb : 0 < b) : (a : ℝ) / b ≤ afCdiv a b := by
  rw [div_le_iff₀ (by exact_mod_cast hb)]
  exact_mod_cast actFit_cdiv_mul a b hb

lemma actFit_cdiv_le (a b c : ℕ) (hb : 0 < b) (h : a ≤ c * b) : afCdiv a b ≤ c := by
  unfold afCdiv
  have : Nat.div (Nat.sub (Nat.add a b) 1) b = (a + b - 1) / b := rfl
  rw [this]
  refine Nat.lt_succ_iff.mp ((Nat.div_lt_iff_lt_mul hb).mpr ?_)
  rw [Nat.succ_mul]
  omega

/-! ### Taylor enclosures of `exp` on `[0, 1]` -/

lemma actFit_tayLo_le (Y : ℕ) : ∀ (n k t acc : ℕ),
    (t : ℝ) ≤ afS * ((Y : ℝ) / afS) ^ k / k.factorial →
    (afTayLo Y n k t acc : ℝ) ≤ acc + afS * ∑ i ∈ range n, ((Y : ℝ) / afS) ^ (k + i) / (k + i).factorial := by
  have hS := actFit_S_pos
  have hy : 0 ≤ (Y : ℝ) / afS := by positivity
  intro n
  induction n with
  | zero => intro k t acc _; simp [afTayLo]
  | succ n ih =>
    intro k t acc ht
    set y := (Y : ℝ) / afS
    have ht' : ((Nat.div (Nat.div (Nat.mul t Y) afS) (Nat.add k 1) : ℕ) : ℝ) ≤
        afS * y ^ (k + 1) / (k + 1).factorial := by
      refine (actFit_div_le _ _).trans ?_
      have h1 := actFit_div_le (Nat.mul t Y) afS
      calc ((Nat.div (Nat.mul t Y) afS : ℕ) : ℝ) / ((Nat.add k 1 : ℕ) : ℝ)
          ≤ ((Nat.mul t Y : ℕ) : ℝ) / afS / ((Nat.add k 1 : ℕ) : ℝ) := by gcongr
        _ = (t : ℝ) * y / (k + 1) := by
            simp only [Nat.mul_eq, Nat.add_eq, Nat.cast_mul, Nat.cast_add, Nat.cast_one, y]; ring
        _ ≤ afS * y ^ k / k.factorial * y / (k + 1) := by gcongr
        _ = afS * y ^ (k + 1) / (k + 1).factorial := by
            rw [Nat.factorial_succ]; push_cast; field_simp; ring
    have := ih (k + 1) _ (Nat.add acc t) ht'
    simp only [afTayLo]
    refine this.trans ?_
    rw [Finset.sum_range_succ', mul_add]
    simp only [Nat.add_eq, Nat.cast_add, add_zero]
    have e : ∀ i, k + 1 + i = k + (i + 1) := fun i => by ring
    simp only [e]
    have : (t : ℝ) ≤ afS * (y ^ k / k.factorial) := by rw [← mul_div_assoc]; exact ht
    linarith

lemma actFit_tayHi_ge (Y : ℕ) : ∀ (n k u acc : ℕ),
    afS * ((Y : ℝ) / afS) ^ k / k.factorial ≤ u →
    acc + afS * ∑ i ∈ range n, ((Y : ℝ) / afS) ^ (k + i) / (k + i).factorial +
      17 / 16 * (afS * ((Y : ℝ) / afS) ^ (k + n) / (k + n).factorial) ≤ (afTayHi Y n k u acc : ℝ) := by
  have hS := actFit_S_pos
  have hy : 0 ≤ (Y : ℝ) / afS := by positivity
  intro n
  induction n with
  | zero =>
    intro k u acc hu
    simp only [afTayHi, range_zero, sum_empty, mul_zero, add_zero, Nat.add_eq, Nat.cast_add,
      Nat.mul_eq]
    have := actFit_cdiv_ge (Nat.mul u 17) 16 (by norm_num)
    simp only [Nat.mul_eq, Nat.cast_mul] at this
    push_cast at this
    linarith
  | succ n ih =>
    intro k u acc hu
    set y := (Y : ℝ) / afS
    have hu' : afS * y ^ (k + 1) / (k + 1).factorial ≤
        ((afCdiv (afCdiv (Nat.mul u Y) afS) (Nat.add k 1) : ℕ) : ℝ) := by
      refine le_trans ?_ (actFit_cdiv_ge _ _ (by simp))
      have h1 := actFit_cdiv_ge (Nat.mul u Y) afS (by norm_num [afS])
      calc afS * y ^ (k + 1) / (k + 1).factorial = afS * y ^ k / k.factorial * y / (k + 1) := by
            rw [Nat.factorial_succ]; push_cast; field_simp; ring
        _ ≤ (u : ℝ) * y / (k + 1) := by gcongr
        _ = ((Nat.mul u Y : ℕ) : ℝ) / afS / ((Nat.add k 1 : ℕ) : ℝ) := by
            simp only [Nat.mul_eq, Nat.add_eq, Nat.cast_mul, Nat.cast_add, Nat.cast_one, y]; ring
        _ ≤ _ := by gcongr
    have := ih (k + 1) _ (Nat.add acc u) hu'
    simp only [afTayHi]
    refine le_trans ?_ this
    rw [Finset.sum_range_succ', mul_add]
    simp only [Nat.add_eq, Nat.cast_add, add_zero]
    have e : ∀ i, k + 1 + i = k + (i + 1) := fun i => by ring
    have e2 : k + 1 + n = k + (n + 1) := by ring
    simp only [e]
    have : afS * (y ^ k / k.factorial) ≤ (u : ℝ) := by rw [← mul_div_assoc]; exact hu
    linarith

lemma actFit_tayLo_ge_acc (Y : ℕ) : ∀ (n k t acc : ℕ), acc ≤ afTayLo Y n k t acc ∧
    (0 < n → acc + t ≤ afTayLo Y n k t acc) := by
  intro n
  induction n with
  | zero => intro k t acc; simp [afTayLo]
  | succ n ih =>
    intro k t acc
    have := (ih (Nat.add k 1) (Nat.div (Nat.div (Nat.mul t Y) afS) (Nat.add k 1)) (Nat.add acc t)).1
    simp only [afTayLo, Nat.add_eq] at this ⊢
    constructor
    · omega
    · intro _; omega

lemma actFit_sqLo (x : ℕ) (z : ℝ) (h : (x : ℝ) ≤ afS * exp z) :
    (afSqLo x : ℝ) ≤ afS * exp (2 * z) := by
  have hS := actFit_S_pos
  refine (actFit_div_le _ _).trans ?_
  rw [div_le_iff₀ hS]
  simp only [Nat.mul_eq, Nat.cast_mul]
  have hx : (0 : ℝ) ≤ x := Nat.cast_nonneg x
  calc (x : ℝ) * x ≤ (afS * exp z) * (afS * exp z) := mul_le_mul h h hx (by positivity)
    _ = afS * exp (2 * z) * afS := by rw [two_mul, exp_add]; ring

lemma actFit_sqHi (x : ℕ) (z : ℝ) (h : afS * exp z ≤ (x : ℝ)) :
    afS * exp (2 * z) ≤ (afSqHi x : ℝ) := by
  have hS := actFit_S_pos
  refine le_trans ?_ (actFit_cdiv_ge _ _ (by norm_num [afS]))
  rw [le_div_iff₀ hS]
  simp only [Nat.mul_eq, Nat.cast_mul]
  have h0 : 0 ≤ afS * exp z := by positivity
  calc afS * exp (2 * z) * afS = (afS * exp z) * (afS * exp z) := by rw [two_mul, exp_add]; ring
    _ ≤ (x : ℝ) * x := mul_le_mul h h h0 (h0.trans h)

lemma actFit_sqLo_ge (x : ℕ) (h : afS ≤ x) : afS ≤ afSqLo x := by
  unfold afSqLo
  have : Nat.div (Nat.mul x x) afS = x * x / afS := rfl
  rw [this, Nat.le_div_iff_mul_le (by norm_num [afS])]
  exact Nat.mul_le_mul h h

/-- **Exp enclosure, positive argument.** For `Y ≤ S`, `afExpPos Y` encloses `S·e^{16 Y/S}`,
and its lower end is at least `S`. -/
theorem actFit_expPos (Y : ℕ) (hY : Y ≤ afS) :
    ((afExpPos Y).1 : ℝ) ≤ afS * exp (16 * ((Y : ℝ) / afS)) ∧
    afS * exp (16 * ((Y : ℝ) / afS)) ≤ (afExpPos Y).2 ∧ afS ≤ (afExpPos Y).1 := by
  have hS := actFit_S_pos
  set y := (Y : ℝ) / afS
  have hy0 : 0 ≤ y := by positivity
  have hy1 : y ≤ 1 := by
    rw [div_le_one hS]; exact_mod_cast hY
  have hlo0 : (afTayLo Y 16 0 afS 0 : ℝ) ≤ afS * exp y := by
    have := actFit_tayLo_le Y 16 0 afS 0 (by simp)
    refine this.trans ?_
    simp only [Nat.cast_zero, zero_add]
    exact mul_le_mul_of_nonneg_left (Real.sum_le_exp_of_nonneg hy0 16) hS.le
  have hhi0 : afS * exp y ≤ (afTayHi Y 16 0 afS 0 : ℝ) := by
    have := actFit_tayHi_ge Y 16 0 afS 0 (by simp)
    refine le_trans ?_ this
    simp only [Nat.cast_zero, zero_add]
    have hb := Real.exp_bound' hy0 hy1 (n := 16) (by norm_num)
    have : afS * exp y ≤ afS * ((∑ m ∈ range 16, y ^ m / m.factorial) +
        y ^ 16 * (16 + 1) / (Nat.factorial 16 * 16)) := mul_le_mul_of_nonneg_left hb hS.le
    refine this.trans (le_of_eq ?_)
    ring
  have e16 : 16 * y = 2 * (2 * (2 * (2 * y))) := by ring
  refine ⟨?_, ?_, ?_⟩
  · rw [e16]
    exact actFit_sqLo _ _ (actFit_sqLo _ _ (actFit_sqLo _ _ (actFit_sqLo _ _ hlo0)))
  · rw [e16]
    exact actFit_sqHi _ _ (actFit_sqHi _ _ (actFit_sqHi _ _ (actFit_sqHi _ _ hhi0)))
  · have h0 : afS ≤ afTayLo Y 16 0 afS 0 := by
      have := (actFit_tayLo_ge_acc Y 16 0 afS 0).2 (by norm_num); omega
    exact_mod_cast actFit_sqLo_ge _ (actFit_sqLo_ge _ (actFit_sqLo_ge _ (actFit_sqLo_ge _ h0)))

/-! ### Signed numbers and intervals -/

/-- The value of a signed pair. -/
def afZv (a : AfZ) : ℝ := (a.1 : ℝ) - a.2

lemma actFit_ble (a b : ℕ) : Nat.ble a b = true ↔ a ≤ b := by rw [Nat.ble_eq]

lemma actFit_zN (a : AfZ) : afZv (afZN a) = afZv a := by
  unfold afZN afZv
  split_ifs with h
  · rw [actFit_ble] at h; simp only [Nat.sub_eq]; push_cast [h]; ring
  · rw [actFit_ble] at h; simp only [Nat.sub_eq]; push_cast [(not_le.mp h).le]; ring

@[simp]
lemma actFit_zAdd (a b : AfZ) : afZv (afZAdd a b) = afZv a + afZv b := by
  simp only [afZAdd, afZv, Nat.add_eq]; push_cast; ring

@[simp]
lemma actFit_zMul (a b : AfZ) : afZv (afZMul a b) = afZv a * afZv b := by
  rw [afZMul, actFit_zN]; simp only [afZv, Nat.add_eq, Nat.mul_eq]; push_cast; ring

@[simp]
lemma actFit_zNeg (a : AfZ) : afZv (afZNeg a) = - afZv a := by
  simp only [afZNeg, afZv]; ring

@[simp]
lemma actFit_zP2 (k : ℕ) : afZv (afP2 k) = 2 ^ k := by
  simp [afP2, afZv, Nat.pow_eq]

@[simp]
lemma actFit_zNat (n : ℕ) : afZv (n, 0) = n := by simp [afZv]

lemma actFit_zLe (a b : AfZ) : afZLe a b = true ↔ afZv a ≤ afZv b := by
  simp only [afZLe, actFit_ble, Nat.add_eq, afZv]
  constructor
  · intro h; have : ((a.1 + b.2 : ℕ) : ℝ) ≤ ((b.1 + a.2 : ℕ) : ℝ) := by exact_mod_cast h
    push_cast at this; linarith
  · intro h; have : ((a.1 + b.2 : ℕ) : ℝ) ≤ ((b.1 + a.2 : ℕ) : ℝ) := by push_cast; linarith
    exact_mod_cast this

lemma actFit_zNonpos (a : AfZ) : afZNonpos a = true ↔ afZv a ≤ 0 := by
  simp only [afZNonpos, actFit_ble, afZv, sub_nonpos, Nat.cast_le]

@[simp]
lemma actFit_zMin (a b : AfZ) : afZv (afZMin a b) = min (afZv a) (afZv b) := by
  unfold afZMin; split_ifs with h
  · rw [actFit_zLe] at h; rw [min_eq_left h]
  · rw [actFit_zLe] at h; rw [min_eq_right (not_le.mp h).le]

@[simp]
lemma actFit_zMax (a b : AfZ) : afZv (afZMax a b) = max (afZv a) (afZv b) := by
  unfold afZMax; split_ifs with h
  · rw [actFit_zLe] at h; rw [max_eq_right h]
  · rw [actFit_zLe] at h; rw [max_eq_left (not_le.mp h).le]

/-- `I` encloses `r`. -/
def afIn (I : AfI) (r : ℝ) : Prop := afZv I.1 ≤ r ∧ r ≤ afZv I.2

lemma actFit_in_add {I J : AfI} {r r' : ℝ} (h : afIn I r) (h' : afIn J r') :
    afIn (afAdd I J) (r + r') := by
  simp only [afIn, afAdd, actFit_zAdd] at *; constructor <;> linarith [h.1, h.2, h'.1, h'.2]

lemma actFit_in_pt (x : AfZ) : afIn (afPt x) (afZv x) := ⟨le_rfl, le_rfl⟩

lemma actFit_in_ofN {I : ℕ × ℕ} {r : ℝ} (h1 : (I.1 : ℝ) ≤ r) (h2 : r ≤ I.2) : afIn (afOfN I) r := by
  simp only [afIn, afOfN, actFit_zNat]; exact ⟨h1, h2⟩

lemma actFit_in_scale (c : AfZ) {J : AfI} {r : ℝ} (h : afIn J r) :
    afIn (afScale c J) (afZv c * r) := by
  unfold afScale
  split_ifs with hc
  · rw [actFit_zNonpos, actFit_zNeg] at hc
    have hc' : 0 ≤ afZv c := by linarith
    simp only [afIn, actFit_zMul]
    exact ⟨mul_le_mul_of_nonneg_left h.1 hc', mul_le_mul_of_nonneg_left h.2 hc'⟩
  · rw [actFit_zNonpos, actFit_zNeg] at hc
    have hc' : afZv c ≤ 0 := by linarith
    simp only [afIn, actFit_zMul]
    exact ⟨mul_le_mul_of_nonpos_left h.2 hc', mul_le_mul_of_nonpos_left h.1 hc'⟩

lemma actFit_in_mulP {I J : AfI} {r r' : ℝ} (hI0 : 0 ≤ afZv I.1) (hI : afIn I r) (hJ : afIn J r') :
    afIn (afMulP I J) (r * r') := by
  obtain ⟨h1, h2⟩ := hI
  obtain ⟨h3, h4⟩ := hJ
  have hr : 0 ≤ r := hI0.trans h1
  unfold afMulP
  constructor
  · dsimp only
    split_ifs with hc
    · rw [actFit_zNonpos, actFit_zNeg] at hc
      simp only [actFit_zMul]
      exact mul_le_mul h1 h3 (by linarith) hr
    · rw [actFit_zNonpos, actFit_zNeg] at hc
      simp only [actFit_zMul]
      nlinarith
  · dsimp only
    split_ifs with hc
    · rw [actFit_zNonpos, actFit_zNeg] at hc
      simp only [actFit_zMul]
      nlinarith
    · rw [actFit_zNonpos, actFit_zNeg] at hc
      simp only [actFit_zMul]
      nlinarith

/-! ### `exp`, `σ`, `g`, `h` enclosures -/

/-- **Exp enclosure.** If `afExpOK X` (i.e. `|X| ≤ 2^44`), `afExpEnc X` encloses
`S·exp (X/2^40)`. -/
theorem actFit_expEnc (X : AfZ) (h : afExpOK X = true) :
    ((afExpEnc X).1 : ℝ) ≤ afS * exp (afZv X / 2 ^ 40) ∧
    afS * exp (afZv X / 2 ^ 40) ≤ (afExpEnc X).2 := by
  have hS := actFit_S_pos
  simp only [afExpOK, Bool.and_eq_true, actFit_ble, Nat.mul_eq, Nat.add_eq] at h
  obtain ⟨h1, h2⟩ := h
  unfold afExpEnc
  split_ifs with hX
  · rw [actFit_ble] at hX
    have hY : (X.1 - X.2) * 1048576 ≤ afS := by
      have e : X.1 * 1048576 = (X.1 - X.2) * 1048576 + X.2 * 1048576 := by
        rw [← Nat.add_mul, Nat.sub_add_cancel hX]
      rw [e] at h1
      exact Nat.le_of_add_le_add_right h1
    have e : 16 * (((Nat.mul (Nat.sub X.1 X.2) 1048576 : ℕ) : ℝ) / afS) = afZv X / 2 ^ 40 := by
      simp only [Nat.mul_eq, Nat.sub_eq, afZv, actFit_S]; push_cast [hX]; ring
    have := actFit_expPos (Nat.mul (Nat.sub X.1 X.2) 1048576) hY
    rw [e] at this
    exact ⟨this.1, this.2.1⟩
  · rw [actFit_ble] at hX
    have hX' : X.1 ≤ X.2 := (not_le.mp hX).le
    have hY : (X.2 - X.1) * 1048576 ≤ afS := by
      have e : X.2 * 1048576 = (X.2 - X.1) * 1048576 + X.1 * 1048576 := by
        rw [← Nat.add_mul, Nat.sub_add_cancel hX']
      rw [e] at h2
      exact Nat.le_of_add_le_add_right h2
    obtain ⟨hl, hh, hpos⟩ := actFit_expPos (Nat.mul (Nat.sub X.2 X.1) 1048576) hY
    set E := afExpPos (Nat.mul (Nat.sub X.2 X.1) 1048576)
    have e : 16 * (((Nat.mul (Nat.sub X.2 X.1) 1048576 : ℕ) : ℝ) / afS) = - (afZv X / 2 ^ 40) := by
      simp only [Nat.mul_eq, Nat.sub_eq, afZv, actFit_S]; push_cast [hX']; ring
    rw [e] at hl hh
    set w := afZv X / 2 ^ 40
    have hE1 : (0 : ℝ) < E.1 := lt_of_lt_of_le hS (by exact_mod_cast hpos)
    have hew : afS * exp w * (afS * exp (-w)) = afS * afS := by
      rw [show afS * exp w * (afS * exp (-w)) = afS * afS * (exp w * exp (-w)) by ring,
        ← exp_add, add_neg_cancel, exp_zero, mul_one]
    have hpw : 0 < afS * exp (-w) := by positivity
    have hSS : ((Nat.mul afS afS : ℕ) : ℝ) = afS * afS := by rw [Nat.mul_eq, Nat.cast_mul]
    have hE2 : (0 : ℝ) < E.2 := hE1.trans_le (hl.trans hh)
    constructor
    · refine (actFit_div_le _ _).trans ?_
      rw [hSS, div_le_iff₀ hE2, ← hew]
      exact mul_le_mul_of_nonneg_left hh (by positivity)
    · refine le_trans ?_ (actFit_cdiv_ge _ _ (by exact_mod_cast hE1))
      rw [hSS, le_div_iff₀ hE1, ← hew]
      exact mul_le_mul_of_nonneg_left hl (by positivity)

lemma actFit_sigmoid_eq (x : ℝ) : sigmoid x = exp x / (1 + exp x) := by
  rw [Real.sigmoid_def, exp_neg]; field_simp; ring

lemma actFit_sigLo (T : ℕ) (x : ℝ) (h : (T : ℝ) ≤ afS * exp x) : (afSigLo T : ℝ) ≤ afS * sigmoid x := by
  have hS := actFit_S_pos
  refine (actFit_div_le _ _).trans ?_
  simp only [Nat.mul_eq, Nat.add_eq, Nat.cast_mul, Nat.cast_add]
  rw [actFit_sigmoid_eq, div_le_iff₀ (by positivity)]
  have he := exp_pos x
  rw [mul_div_assoc', div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
  nlinarith

lemma actFit_sigHi (T : ℕ) (x : ℝ) (h : afS * exp x ≤ (T : ℝ)) : afS * sigmoid x ≤ afSigHi T := by
  have hS := actFit_S_pos
  refine le_trans ?_ (actFit_cdiv_ge _ _ (by norm_num [afS]))
  simp only [Nat.mul_eq, Nat.add_eq, Nat.cast_mul, Nat.cast_add]
  rw [actFit_sigmoid_eq, le_div_iff₀ (by positivity)]
  have he := exp_pos x
  rw [mul_div_assoc', div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
  nlinarith

lemma actFit_sigHi_le (T : ℕ) : afSigHi T ≤ afS := by
  refine actFit_cdiv_le _ _ _ (by norm_num [afS]) ?_
  simp only [Nat.mul_eq, Nat.add_eq]
  nlinarith

/-- `g(t) = (t² + 2t)/(t² + 2t + 2)`, so that `mish a = a·g(eᵃ)`. -/
noncomputable def afGR (t : ℝ) : ℝ := (t ^ 2 + 2 * t) / (t ^ 2 + 2 * t + 2)

/-- `h(t) = 4t(t+1)/(t² + 2t + 2)²`, so that `d/da g(eᵃ) = h(eᵃ)`. -/
noncomputable def afHR (t : ℝ) : ℝ := 4 * t * (t + 1) / (t ^ 2 + 2 * t + 2) ^ 2

lemma actFit_gR_mono {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) : afGR a ≤ afGR b := by
  unfold afGR
  rw [div_le_div_iff₀ (by positivity) (by nlinarith)]
  nlinarith

lemma actFit_gScale (T : ℕ) :
    ((Nat.mul (afGN T) afS : ℕ) : ℝ) / (afGD T : ℝ) = afS * afGR ((T : ℝ) / afS) := by
  have hS := actFit_S_pos
  simp only [afGD, afGN, afGR, Nat.mul_eq, Nat.add_eq, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat]
  field_simp

lemma actFit_gLo (T : ℕ) (t : ℝ) (h : (T : ℝ) ≤ afS * t) : (afGLo T : ℝ) ≤ afS * afGR t := by
  have hS := actFit_S_pos
  refine (actFit_div_le _ _).trans ?_
  rw [actFit_gScale]
  refine mul_le_mul_of_nonneg_left (actFit_gR_mono (by positivity) ?_) hS.le
  rw [div_le_iff₀ hS]; linarith

lemma actFit_gHi (T : ℕ) (t : ℝ) (ht : 0 ≤ t) (h : afS * t ≤ (T : ℝ)) : afS * afGR t ≤ afGHi T := by
  have hS := actFit_S_pos
  refine le_trans ?_ (actFit_cdiv_ge _ _ ?_)
  · rw [actFit_gScale]
    refine mul_le_mul_of_nonneg_left (actFit_gR_mono ht ?_) hS.le
    rw [le_div_iff₀ hS]; linarith
  · simp only [afGD, Nat.add_eq, Nat.mul_eq]; norm_num [afS]

lemma actFit_hScale (Tl Tu : ℕ) :
    ((Nat.mul (afHN Tl) afS : ℕ) : ℝ) / ((Nat.mul (afGD Tu) (afGD Tu) : ℕ) : ℝ) =
      afS * (4 * ((Tl : ℝ) / afS) * ((Tl : ℝ) / afS + 1) /
        (((Tu : ℝ) / afS) ^ 2 + 2 * ((Tu : ℝ) / afS) + 2) ^ 2) := by
  have hS := actFit_S_pos
  simp only [afHN, afGD, afGN, Nat.mul_eq, Nat.add_eq, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat]
  field_simp

lemma actFit_hLo (Tl Tu : ℕ) (t : ℝ) (ht : 0 ≤ t) (hl : (Tl : ℝ) ≤ afS * t) (hu : afS * t ≤ (Tu : ℝ)) :
    (afHLo Tl Tu : ℝ) ≤ afS * afHR t := by
  have hS := actFit_S_pos
  refine (actFit_div_le _ _).trans ?_
  rw [actFit_hScale]
  refine mul_le_mul_of_nonneg_left ?_ hS.le
  have h1 : (Tl : ℝ) / afS ≤ t := by rw [div_le_iff₀ hS]; linarith
  have h2 : t ≤ (Tu : ℝ) / afS := by rw [le_div_iff₀ hS]; linarith
  have h0 : 0 ≤ (Tl : ℝ) / afS := by positivity
  unfold afHR
  apply div_le_div₀ (by positivity) (by nlinarith) (by positivity)
  apply pow_le_pow_left₀ (by positivity)
  nlinarith

lemma actFit_hHi (Tu Tl : ℕ) (t : ℝ) (ht : 0 ≤ t) (hl : (Tl : ℝ) ≤ afS * t) (hu : afS * t ≤ (Tu : ℝ)) :
    afS * afHR t ≤ (afHHi Tu Tl : ℝ) := by
  have hS := actFit_S_pos
  refine le_trans ?_ (actFit_cdiv_ge _ _ ?_)
  · rw [actFit_hScale]
    refine mul_le_mul_of_nonneg_left ?_ hS.le
    have h1 : (Tl : ℝ) / afS ≤ t := by rw [div_le_iff₀ hS]; linarith
    have h2 : t ≤ (Tu : ℝ) / afS := by rw [le_div_iff₀ hS]; linarith
    have h0 : 0 ≤ (Tl : ℝ) / afS := by positivity
    unfold afHR
    apply div_le_div₀ (by positivity) (by nlinarith) (by positivity)
    apply pow_le_pow_left₀ (by positivity)
    nlinarith
  · simp only [afGD, afGN, Nat.add_eq, Nat.mul_eq]
    have : 0 < 2 * afS * afS := by norm_num [afS]
    positivity

/-! ### The functions and their derivatives -/

/-- `mish a = a·g(eᵃ)`. -/
theorem actFit_mish_eq (a : ℝ) : mish a = a * afGR (exp a) := by
  have ht := exp_pos a
  have h1 : 0 < 1 + exp a := by linarith
  unfold mish softplus afGR
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq, exp_neg, exp_log h1]
  congr 1
  field_simp
  ring

/-- `SiLU'(x) = σ(x)(1 + x(1 − σ(x)))`. -/
noncomputable def afDSilu (x : ℝ) : ℝ := sigmoid x * (1 + x * (1 - sigmoid x))

lemma actFit_hasDerivAt_silu (x : ℝ) : HasDerivAt silu (afDSilu x) x := by
  have := (hasDerivAt_id x).mul (Real.hasDerivAt_sigmoid x)
  have e : silu = fun x => id x * sigmoid x := by funext y; rfl
  rw [e]; convert this using 1; simp only [afDSilu, id]; ring

/-- `mish'(a) = g(eᵃ) + a·h(eᵃ)`. -/
noncomputable def afMishD (a : ℝ) : ℝ := afGR (exp a) + a * afHR (exp a)

lemma actFit_hasDerivAt_mish (a : ℝ) : HasDerivAt mish (afMishD a) a := by
  have ht := exp_pos a
  have hN : HasDerivAt (fun a => exp a ^ 2 + 2 * exp a)
      (2 * exp a * exp a + 2 * exp a) a := by
    have := ((Real.hasDerivAt_exp a).pow 2).add ((Real.hasDerivAt_exp a).const_mul 2)
    convert this using 1; push_cast; ring
  have hD : HasDerivAt (fun a => exp a ^ 2 + 2 * exp a + 2)
      (2 * exp a * exp a + 2 * exp a) a := hN.add_const 2
  have hDne : exp a ^ 2 + 2 * exp a + 2 ≠ 0 := by positivity
  have hg : HasDerivAt (fun a => afGR (exp a)) (afHR (exp a)) a := by
    have := hN.div hD hDne
    unfold afGR afHR
    convert this using 1
    field_simp
    ring
  have e : mish = fun a => a * afGR (exp a) := funext actFit_mish_eq
  rw [e]
  have := (hasDerivAt_id' a).mul hg
  convert this using 1
  unfold afMishD
  ring

/-- `α`, `β`, `γ` of a unit, and its pre-activation `x(a) = β a + γ`. -/
noncomputable def afAl (u : AfUnit) : ℝ := afZv u.A / 2 ^ 24
noncomputable def afBe (u : AfUnit) : ℝ := afZv u.B / 2 ^ 24
noncomputable def afXR (u : AfUnit) (a : ℝ) : ℝ := afBe u * a + afZv u.G / 2 ^ 40

/-- The fit `Σ αᵢ SiLU(βᵢ a + γᵢ)`. -/
noncomputable def afFitR : List AfUnit → ℝ → ℝ
  | [], _ => 0
  | u :: us, a => afAl u * silu (afXR u a) + afFitR us a

/-- Its derivative. -/
noncomputable def afFitD : List AfUnit → ℝ → ℝ
  | [], _ => 0
  | u :: us, a => afAl u * afBe u * afDSilu (afXR u a) + afFitD us a

lemma actFit_hasDerivAt_fit (us : List AfUnit) (a : ℝ) : HasDerivAt (afFitR us) (afFitD us a) a := by
  induction us with
  | nil => exact hasDerivAt_const a (0 : ℝ) |>.congr_of_eventuallyEq (by
      filter_upwards with y; rfl)
  | cons u us ih =>
    have hx : HasDerivAt (afXR u) (afBe u) a := by
      have e : afXR u = fun x => afBe u * x + afZv u.G / 2 ^ 40 := rfl
      rw [e]; simpa using ((hasDerivAt_id' a).const_mul (afBe u)).add_const (afZv u.G / 2 ^ 40)
    have := (((actFit_hasDerivAt_silu (afXR u a)).comp a hx).const_mul (afAl u)).add ih
    have e : afFitR (u :: us) = fun a => afAl u * (silu ∘ afXR u) a + afFitR us a := by
      funext y; rfl
    rw [e]; convert this using 1; simp only [afFitD]; ring

lemma actFit_X (u : AfUnit) (p : AfZ) : afZv (u.X p) = 2 ^ 40 * afXR u (afZv p / 2 ^ 16) := by
  simp only [AfUnit.X, actFit_zAdd, actFit_zMul, afXR, afBe]; ring

/-! ### One unit, the mish part, the sum -/

/-- `t` encloses `S·exp x`. -/
def afBnd (t : ℕ × ℕ) (x : ℝ) : Prop := (t.1 : ℝ) ≤ afS * exp x ∧ afS * exp x ≤ t.2

lemma actFit_sig_in {El Eu : ℕ × ℕ} {xl xu x : ℝ} (hl : (El.1 : ℝ) ≤ afS * exp xl)
    (hu : afS * exp xu ≤ Eu.2) (h1 : xl ≤ x) (h2 : x ≤ xu) :
    afIn (afOfN (afSigLo El.1, afSigHi Eu.2)) (afS * sigmoid x) := by
  have hS := actFit_S_pos
  refine actFit_in_ofN ?_ ?_
  · exact (actFit_sigLo _ _ hl).trans (mul_le_mul_of_nonneg_left (Real.sigmoid_le h1) hS.le)
  · exact (mul_le_mul_of_nonneg_left (Real.sigmoid_le h2) hS.le).trans (actFit_sigHi _ _ hu)

lemma actFit_unit_val (u : AfUnit) (p q : AfZ) (Ep Eq : ℕ × ℕ)
    (hEp : afBnd Ep (afXR u (afZv p / 2 ^ 16))) :
    afIn (afUnitE u p q Ep Eq).1 (-(afAl u * silu (afXR u (afZv p / 2 ^ 16))) * 2 ^ 128) := by
  have hσ := actFit_sig_in hEp.1 hEp.2 le_rfl le_rfl
  have := actFit_in_scale (afZNeg u.A) (actFit_in_scale (u.X p) hσ)
  simp only [afUnitE]
  convert this using 1
  rw [actFit_zNeg, actFit_X, actFit_S]
  simp only [afAl, silu]
  ring

lemma actFit_derE (u : AfUnit) (xl xu : AfZ) (El Eu : ℕ × ℕ) (x : ℝ)
    (hl : (El.1 : ℝ) ≤ afS * exp (afZv xl / 2 ^ 40)) (hu : afS * exp (afZv xu / 2 ^ 40) ≤ Eu.2)
    (h1 : afZv xl / 2 ^ 40 ≤ x) (h2 : x ≤ afZv xu / 2 ^ 40) :
    afIn (afDerE u xl xu El Eu) (-(afAl u * afBe u * afDSilu x) * 2 ^ 216) := by
  have hS := actFit_S_pos
  have hσ := actFit_sig_in hl hu h1 h2
  have hom : afIn ((afS, afSigHi Eu.2), (afS, afSigLo El.1)) (afS * (1 - sigmoid x)) := by
    simp only [afIn, afZv] at hσ ⊢
    simp only [afOfN, Nat.cast_zero, sub_zero] at hσ
    constructor <;> linarith [hσ.1, hσ.2]
  have hom0 : 0 ≤ afZv ((afS, afSigHi Eu.2), (afS, afSigLo El.1)).1 := by
    simp only [afZv, sub_nonneg, Nat.cast_le]; exact actFit_sigHi_le _
  have hx : afIn (xl, xu) (2 ^ 40 * x) := by
    simp only [afIn]; constructor <;> [skip; skip] <;>
      [linarith [(div_le_iff₀ (by norm_num : (0:ℝ) < 2 ^ 40)).mp h1];
       linarith [(le_div_iff₀ (by norm_num : (0:ℝ) < 2 ^ 40)).mp h2]]
  have h0 : 0 ≤ afZv (afOfN (afSigLo El.1, afSigHi Eu.2)).1 := by simp [afOfN, afZv]
  have := actFit_in_scale (afZNeg (afZMul u.A u.B))
    (actFit_in_mulP h0 hσ (actFit_in_add (actFit_in_mulP hom0 hom hx) (actFit_in_pt (afP2 104))))
  unfold afDerE
  convert this using 1
  rw [actFit_zNeg, actFit_zMul, actFit_zP2, actFit_S]
  simp only [afAl, afBe, afDSilu]
  ring

lemma actFit_unit_der (u : AfUnit) (p q : AfZ) (Ep Eq : ℕ × ℕ) (_hpq : afZv p ≤ afZv q)
    (hEp : afBnd Ep (afXR u (afZv p / 2 ^ 16))) (hEq : afBnd Eq (afXR u (afZv q / 2 ^ 16)))
    (ξ : ℝ) (h1 : afZv p / 2 ^ 16 ≤ ξ) (h2 : ξ ≤ afZv q / 2 ^ 16) :
    afIn (afUnitE u p q Ep Eq).2 (-(afAl u * afBe u * afDSilu (afXR u ξ)) * 2 ^ 216) := by
  have ep : afZv (u.X p) / 2 ^ 40 = afXR u (afZv p / 2 ^ 16) := by rw [actFit_X]; ring
  have eq : afZv (u.X q) / 2 ^ 40 = afXR u (afZv q / 2 ^ 16) := by rw [actFit_X]; ring
  simp only [afUnitE]
  by_cases hB : afZNonpos u.B = true
  · simp only [hB, ↓reduceIte]
    have hB' : afBe u ≤ 0 := by
      rw [actFit_zNonpos] at hB; unfold afBe; exact div_nonpos_of_nonpos_of_nonneg hB (by norm_num)
    refine actFit_derE u _ _ _ _ _ (by rw [eq]; exact hEq.1) (by rw [ep]; exact hEp.2) ?_ ?_
    · rw [eq]; unfold afXR; nlinarith
    · rw [ep]; unfold afXR; nlinarith
  · have hB0 := hB
    simp only [Bool.not_eq_true] at hB0
    simp only [hB0, Bool.false_eq_true, ↓reduceIte]
    have hB' : 0 ≤ afBe u := by
      rw [actFit_zNonpos] at hB; unfold afBe; exact div_nonneg (by linarith) (by norm_num)
    refine actFit_derE u _ _ _ _ _ (by rw [ep]; exact hEp.1) (by rw [eq]; exact hEq.2) ?_ ?_
    · rw [ep]; unfold afXR; nlinarith
    · rw [eq]; unfold afXR; nlinarith

lemma actFit_mish_val (p q : AfZ) (Tp Tq : ℕ × ℕ) (hTp : afBnd Tp (afZv p / 2 ^ 16)) :
    afIn (afMishE p q Tp Tq).1 (mish (afZv p / 2 ^ 16) * 2 ^ 128) := by
  have hS := actFit_S_pos
  have ht := exp_pos (afZv p / 2 ^ 16)
  have hg : afIn (afOfN (afGLo Tp.1, afGHi Tp.2)) (afS * afGR (exp (afZv p / 2 ^ 16))) :=
    actFit_in_ofN (actFit_gLo _ _ hTp.1) (actFit_gHi _ _ ht.le hTp.2)
  have := actFit_in_scale (afZMul p (afP2 48)) hg
  simp only [afMishE]
  convert this using 1
  rw [actFit_mish_eq, actFit_zMul, actFit_zP2, actFit_S]
  ring

lemma actFit_mish_der (p q : AfZ) (Tp Tq : ℕ × ℕ) (hTp : afBnd Tp (afZv p / 2 ^ 16))
    (hTq : afBnd Tq (afZv q / 2 ^ 16)) (ξ : ℝ) (h1 : afZv p / 2 ^ 16 ≤ ξ) (h2 : ξ ≤ afZv q / 2 ^ 16) :
    afIn (afMishE p q Tp Tq).2 (afMishD ξ * 2 ^ 216) := by
  have hS := actFit_S_pos
  have ht := exp_pos ξ
  have hlo : (Tp.1 : ℝ) ≤ afS * exp ξ :=
    hTp.1.trans (mul_le_mul_of_nonneg_left (exp_le_exp.mpr h1) hS.le)
  have hhi : afS * exp ξ ≤ Tq.2 :=
    (mul_le_mul_of_nonneg_left (exp_le_exp.mpr h2) hS.le).trans hTq.2
  have hh : afIn (afOfN (afHLo Tp.1 Tq.2, afHHi Tq.2 Tp.1)) (afS * afHR (exp ξ)) :=
    actFit_in_ofN (actFit_hLo _ _ _ ht.le hlo hhi) (actFit_hHi _ _ _ ht.le hlo hhi)
  have hg : afIn (afOfN (afGLo Tp.1, afGHi Tq.2)) (afS * afGR (exp ξ)) :=
    actFit_in_ofN (actFit_gLo _ _ hlo) (actFit_gHi _ _ ht.le hhi)
  have hpq : afIn (p, q) (2 ^ 16 * ξ) := by
    simp only [afIn]; constructor
    · linarith [(div_le_iff₀ (by norm_num : (0:ℝ) < 2 ^ 16)).mp h1]
    · linarith [(le_div_iff₀ (by norm_num : (0:ℝ) < 2 ^ 16)).mp h2]
  have h0 : 0 ≤ afZv (afOfN (afHLo Tp.1 Tq.2, afHHi Tq.2 Tp.1)).1 := by simp [afOfN, afZv]
  have := actFit_in_scale (afP2 136) (actFit_in_add (actFit_in_mulP h0 hh hpq)
    (actFit_in_scale (afP2 16) hg))
  simp only [afMishE]
  convert this using 1
  rw [actFit_zP2, actFit_zP2, actFit_S]
  simp only [afMishD]
  ring

lemma actFit_sum (us : List AfUnit) (p q : AfZ) (hpq : afZv p ≤ afZv q) :
    ∀ (Eps Eqs : List (ℕ × ℕ)),
    List.Forall₂ (fun u E => afBnd E (afXR u (afZv p / 2 ^ 16))) us Eps →
    List.Forall₂ (fun u E => afBnd E (afXR u (afZv q / 2 ^ 16))) us Eqs →
    afIn (afSumE us p q Eps Eqs).1 (-afFitR us (afZv p / 2 ^ 16) * 2 ^ 128) ∧
    ∀ ξ, afZv p / 2 ^ 16 ≤ ξ → ξ ≤ afZv q / 2 ^ 16 →
      afIn (afSumE us p q Eps Eqs).2 (-afFitD us ξ * 2 ^ 216) := by
  induction us with
  | nil =>
    intro Eps Eqs hp hq
    cases hp; cases hq
    simp [afSumE, afFitR, afFitD, afIn, afZv]
  | cons u us ih =>
    intro Eps Eqs hp hq
    cases hp with
    | @cons _ Ep0 _ _ hu hp' =>
      cases hq with
      | @cons _ Eq0 _ _ hu' hq' =>
        obtain ⟨r1, r2⟩ := ih _ _ hp' hq'
        constructor
        · have := actFit_in_add (actFit_unit_val u p q Ep0 Eq0 hu) r1
          simp only [afSumE]; convert this using 1; simp only [afFitR]; ring
        · intro ξ h1 h2
          have := actFit_in_add (actFit_unit_der u p q Ep0 Eq0 hpq hu hu' ξ h1 h2) (r2 ξ h1 h2)
          simp only [afSumE]; convert this using 1; simp only [afFitD]; ring

/-! ### One cell, a run of cells, a chunk -/

/-- The checker's invariant at a grid point: exp enclosures of `eᵖ` and of every `e^{x(p)}`. -/
def afInv (us : List AfUnit) (s : AfSt) : Prop :=
  afBnd s.m (afZv s.p / 2 ^ 16) ∧ List.Forall₂ (fun u t => afBnd t (afXR u (afZv s.p / 2 ^ 16))) us s.ts

/-- The error of the fit. -/
noncomputable def afErr (us : List AfUnit) (a : ℝ) : ℝ := mish a - afFitR us a

lemma actFit_cell (us : List AfUnit) (E H : ℕ) (s s' : AfSt) (hH : afZv s'.p = afZv s.p + H)
    (hs : afInv us s) (hs' : afInv us s') (hc : afCellOK us E H s s' = true) (a : ℝ)
    (ha1 : afZv s.p / 2 ^ 16 ≤ a) (ha2 : a ≤ afZv s'.p / 2 ^ 16) :
    |afErr us a| ≤ E / 2 ^ 232 := by
  set P := afZv s.p / 2 ^ 16 with hPd
  set Q := afZv s'.p / 2 ^ 16 with hQd
  have hpq : afZv s.p ≤ afZv s'.p := by rw [hH]; linarith [(Nat.cast_nonneg H : (0 : ℝ) ≤ H)]
  obtain ⟨sv, sd⟩ := actFit_sum us s.p s'.p hpq s.ts s'.ts hs.2 hs'.2
  have hv := actFit_in_add (actFit_mish_val s.p s'.p s.m s'.m hs.1) sv
  have hd : ∀ ξ, P ≤ ξ → ξ ≤ Q → afIn (afAdd (afMishE s.p s'.p s.m s'.m).2
      (afSumE us s.p s'.p s.ts s'.ts).2) ((afMishD ξ - afFitD us ξ) * 2 ^ 216) := fun ξ h1 h2 => by
    have := actFit_in_add (actFit_mish_der s.p s'.p s.m s'.m hs.1 hs'.1 ξ h1 h2) (sd ξ h1 h2)
    convert this using 1; ring
  set v := afAdd (afMishE s.p s'.p s.m s'.m).1 (afSumE us s.p s'.p s.ts s'.ts).1
  set D := afAdd (afMishE s.p s'.p s.m s'.m).2 (afSumE us s.p s'.p s.ts s'.ts).2
  have hv' : afZv v.1 ≤ afErr us P * 2 ^ 128 ∧ afErr us P * 2 ^ 128 ≤ afZv v.2 := by
    have := hv; simp only [afIn, ← hPd] at this; unfold afErr; constructor <;> linarith [this.1, this.2]
  simp only [afCellOK, Bool.and_eq_true, actFit_zLe, actFit_zAdd, actFit_zMul, actFit_zMax,
    actFit_zMin, actFit_zP2, actFit_zNat] at hc
  change afZv v.2 * 2 ^ 104 + max ((0 : ℕ) : ℝ) (afZv D.2) * H ≤ E ∧
    afZv (0, E) ≤ afZv v.1 * 2 ^ 104 + min ((0 : ℕ) : ℝ) (afZv D.1) * H at hc
  have hz : afZv (0, E) = -(E : ℝ) := by simp [afZv]
  simp only [Nat.cast_zero, hz] at hc
  obtain ⟨hc1, hc2⟩ := hc
  -- the mean value theorem on `[P, a]`
  have hmvt : ∃ ξ, P ≤ ξ ∧ ξ ≤ Q ∧ afErr us a = afErr us P + (afMishD ξ - afFitD us ξ) * (a - P) := by
    rcases eq_or_lt_of_le ha1 with h | h
    · exact ⟨P, le_rfl, by linarith, by rw [← h]; ring⟩
    · have hder : ∀ x, HasDerivAt (afErr us) (afMishD x - afFitD us x) x := fun x =>
        (actFit_hasDerivAt_mish x).sub (actFit_hasDerivAt_fit us x)
      obtain ⟨ξ, hξ, he⟩ := exists_hasDerivAt_eq_slope (afErr us) (fun x => afMishD x - afFitD us x) h
        (fun x _ => (hder x).continuousAt.continuousWithinAt) (fun x _ => hder x)
      refine ⟨ξ, hξ.1.le, by linarith [hξ.2], ?_⟩
      rw [he]; field_simp; ring
  obtain ⟨ξ, hξ1, hξ2, he⟩ := hmvt
  have hD := hd ξ hξ1 hξ2
  simp only [afIn] at hD
  set dv := (afMishD ξ - afFitD us ξ) * 2 ^ 216
  set t := (a - P) * 2 ^ 16
  have ht0 : 0 ≤ t := by simp only [t]; nlinarith
  have htH : t ≤ H := by
    simp only [t, P]
    have : a ≤ (afZv s.p + H) / 2 ^ 16 := by rw [← hH]; exact ha2
    rw [le_div_iff₀ (by norm_num)] at this
    linarith
  have hEa : afErr us a * 2 ^ 232 = afErr us P * 2 ^ 128 * 2 ^ 104 + dv * t := by
    rw [he]; simp only [dv, t]; ring
  have hup : dv * t ≤ max 0 (afZv D.2) * H := by
    have h1 : dv ≤ max 0 (afZv D.2) := hD.2.trans (le_max_right _ _)
    calc dv * t ≤ max 0 (afZv D.2) * t := mul_le_mul_of_nonneg_right h1 ht0
      _ ≤ max 0 (afZv D.2) * H := mul_le_mul_of_nonneg_left htH (le_max_left _ _)
  have hlo : min 0 (afZv D.1) * H ≤ dv * t := by
    have h1 : min 0 (afZv D.1) ≤ dv := (min_le_right _ _).trans hD.1
    calc min 0 (afZv D.1) * H ≤ min 0 (afZv D.1) * t :=
          mul_le_mul_of_nonpos_left htH (min_le_left _ _)
      _ ≤ dv * t := mul_le_mul_of_nonneg_right h1 ht0
  have h2 : (0 : ℝ) < 2 ^ 232 := by positivity
  rw [abs_le, ← neg_div, div_le_iff₀ h2, le_div_iff₀ h2, hEa]
  constructor
  · nlinarith [hv'.1]
  · nlinarith [hv'.2]

lemma actFit_stepE {t M : ℕ × ℕ} {x y : ℝ} (ht : afBnd t x) (hM : afBnd M y) :
    afBnd (afStepE t M) (x + y) := by
  have hS := actFit_S_pos
  have he : afS * exp (x + y) = afS * exp x * (afS * exp y) / afS := by
    rw [exp_add]; field_simp
  constructor
  · refine (actFit_div_le _ _).trans ?_
    rw [he, Nat.mul_eq, Nat.cast_mul]
    gcongr
    · exact ht.1
    · exact hM.1
  · refine le_trans ?_ (actFit_cdiv_ge _ _ (by norm_num [afS]))
    rw [he, Nat.mul_eq, Nat.cast_mul]
    gcongr
    · exact ht.2
    · exact hM.2

lemma actFit_stepAll (P h : ℝ) : ∀ (us : List AfUnit) (ts Ms : List (ℕ × ℕ)),
    List.Forall₂ (fun u t => afBnd t (afXR u P)) us ts →
    List.Forall₂ (fun u M => afBnd M (afBe u * h)) us Ms →
    List.Forall₂ (fun u t => afBnd t (afXR u (P + h))) us (afStepAll ts Ms) := by
  intro us
  induction us with
  | nil => intro ts Ms h1 _; cases h1; simp [afStepAll]
  | cons u us ih =>
    intro ts Ms h1 h2
    cases h1 with
    | cons hu h1' =>
      cases h2 with
      | cons hM h2' =>
        simp only [afStepAll]
        refine List.Forall₂.cons ?_ (ih _ _ h1' h2')
        have : afXR u (P + h) = afXR u P + afBe u * h := by unfold afXR; ring
        rw [this]; exact actFit_stepE hu hM

lemma actFit_next_p (H : ℕ) (Mm : ℕ × ℕ) (Ms : List (ℕ × ℕ)) (s : AfSt) :
    afZv (afNext H Mm Ms s).p = afZv s.p + H := by
  simp [afNext]

lemma actFit_next_inv (us : List AfUnit) (H : ℕ) (Mm : ℕ × ℕ) (Ms : List (ℕ × ℕ))
    (hMm : afBnd Mm ((H : ℝ) / 2 ^ 16))
    (hMs : List.Forall₂ (fun u M => afBnd M (afBe u * ((H : ℝ) / 2 ^ 16))) us Ms)
    (s : AfSt) (hs : afInv us s) : afInv us (afNext H Mm Ms s) := by
  have e : afZv (afNext H Mm Ms s).p / 2 ^ 16 = afZv s.p / 2 ^ 16 + (H : ℝ) / 2 ^ 16 := by
    rw [actFit_next_p]; ring
  unfold afInv
  rw [e]
  exact ⟨actFit_stepE hs.1 hMm, actFit_stepAll _ _ us s.ts Ms hs.2 hMs⟩

lemma actFit_run (us : List AfUnit) (E H : ℕ) (Mm : ℕ × ℕ) (Ms : List (ℕ × ℕ))
    (hMm : afBnd Mm ((H : ℝ) / 2 ^ 16))
    (hMs : List.Forall₂ (fun u M => afBnd M (afBe u * ((H : ℝ) / 2 ^ 16))) us Ms) :
    ∀ (n : ℕ) (s : AfSt), afInv us s → afRun us E H Mm Ms (n + 1) s = true →
    ∀ a, afZv s.p / 2 ^ 16 ≤ a → a ≤ (afZv s.p + (n + 1) * H) / 2 ^ 16 →
      |afErr us a| ≤ E / 2 ^ 232 := by
  intro n
  induction n with
  | zero =>
    intro s hs hr a h1 h2
    simp only [afRun, Bool.and_eq_true] at hr
    refine actFit_cell us E H s _ (actFit_next_p H Mm Ms s) hs
      (actFit_next_inv us H Mm Ms hMm hMs s hs) hr.1 a h1 ?_
    rw [actFit_next_p]; simpa using h2
  | succ n ih =>
    intro s hs hr a h1 h2
    rw [afRun, Bool.and_eq_true] at hr
    have hs' := actFit_next_inv us H Mm Ms hMm hMs s hs
    by_cases ha : a ≤ (afZv s.p + H) / 2 ^ 16
    · exact actFit_cell us E H s _ (actFit_next_p H Mm Ms s) hs hs' hr.1 a h1
        (by rw [actFit_next_p]; exact ha)
    · refine ih _ hs' hr.2 a ?_ ?_
      · rw [actFit_next_p]; exact (not_le.mp ha).le
      · rw [actFit_next_p]; push_cast at h2 ⊢; linarith

lemma actFit_init (us : List AfUnit) (f : AfUnit → AfZ) (x : AfUnit → ℝ)
    (hok : ∀ u ∈ us, afExpOK (f u) = true) (hx : ∀ u ∈ us, afZv (f u) / 2 ^ 40 = x u) :
    List.Forall₂ (fun u t => afBnd t (x u)) us (us.map (fun u => afExpEnc (f u))) := by
  rw [List.forall₂_map_right_iff, List.forall₂_same]
  intro u hu
  have := actFit_expEnc (f u) (hok u hu)
  rw [hx u hu] at this
  exact this

/-- **A chunk.** If `afChunkOK us E H a0 n` (`n ≥ 1`), then on `[a0, a0 + n H]/2^16` the fit is within
`E/2^232` of mish. -/
theorem actFit_chunk (us : List AfUnit) (E H : ℕ) (a0 : AfZ) (n : ℕ)
    (h : afChunkOK us E H a0 (n + 1) = true) (a : ℝ) (h1 : afZv a0 / 2 ^ 16 ≤ a)
    (h2 : a ≤ (afZv a0 + (n + 1) * H) / 2 ^ 16) : |mish a - afFitR us a| ≤ E / 2 ^ 232 := by
  simp only [afChunkOK, Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨⟨⟨h0, hH⟩, hus⟩, hr⟩ := h
  have hMm : afBnd (afExpEnc (Nat.mul H 16777216, 0)) ((H : ℝ) / 2 ^ 16) := by
    have := actFit_expEnc _ hH
    have e : afZv (Nat.mul H 16777216, 0) / 2 ^ 40 = (H : ℝ) / 2 ^ 16 := by
      simp only [actFit_zNat, Nat.mul_eq]; push_cast; ring
    rw [e] at this; exact this
  have hMs := actFit_init us (fun u => afZMul u.B (H, 0)) (fun u => afBe u * ((H : ℝ) / 2 ^ 16))
    (fun u hu => ((hus u hu)).2) (fun u _ => by simp only [actFit_zMul, actFit_zNat, afBe]; ring)
  have hinv : afInv us ⟨a0, afExpEnc (afZMul a0 (afP2 24)), us.map (fun u => afExpEnc (u.X a0))⟩ := by
    refine ⟨?_, actFit_init us (fun u => u.X a0) (fun u => afXR u (afZv a0 / 2 ^ 16))
      (fun u hu => (hus u hu).1) (fun u _ => by rw [actFit_X]; ring)⟩
    have := actFit_expEnc _ h0
    have e : afZv (afZMul a0 (afP2 24)) / 2 ^ 40 = afZv a0 / 2 ^ 16 := by
      simp only [actFit_zMul, actFit_zP2]; ring
    rw [e] at this; exact this
  exact actFit_run us E H _ _ hMm hMs n _ hinv hr a h1 h2

lemma actFit_allBelow (f : ℕ → Bool) : ∀ n, afAllBelow f n = true → ∀ c < n, f c = true := by
  intro n
  induction n with
  | zero => intro _ c hc; omega
  | succ n ih =>
    intro h c hc
    simp only [afAllBelow, Bool.and_eq_true] at h
    rcases Nat.lt_succ_iff_lt_or_eq.mp hc with hc | hc
    · exact ih h.2 c hc
    · rw [hc]; exact h.1

end sound

/-! ## The certificate: 96 chunks of 32 cells of width `2^-8` cover `[−6, 6]` -/

/-- Group `g` of 12 chunks (`a ∈ [−6 + 3g/2, −6 + 3(g+1)/2]`). -/
def afGroup (g : ℕ) : Bool :=
  afAllBelow (fun c => afChunkOK afFit6 afE6 256
    (Nat.mul (Nat.add (Nat.mul g 12) c) 8192, 393216) 32) 12

theorem actFit_group0 : afGroup 0 = true := by decide +kernel
theorem actFit_group1 : afGroup 1 = true := by decide +kernel
theorem actFit_group2 : afGroup 2 = true := by decide +kernel
theorem actFit_group3 : afGroup 3 = true := by decide +kernel
theorem actFit_group4 : afGroup 4 = true := by decide +kernel
theorem actFit_group5 : afGroup 5 = true := by decide +kernel
theorem actFit_group6 : afGroup 6 = true := by decide +kernel
theorem actFit_group7 : afGroup 7 = true := by decide +kernel

theorem actFit_chunks : ∀ c < 96,
    afChunkOK afFit6 afE6 256 (Nat.mul c 8192, 393216) 32 = true := by
  have G : ∀ g < 8, afGroup g = true := by
    intro g hg
    interval_cases g
    exacts [actFit_group0, actFit_group1, actFit_group2, actFit_group3, actFit_group4,
      actFit_group5, actFit_group6, actFit_group7]
  intro c hc
  have := actFit_allBelow _ 12 (G (c / 12) (by omega)) (c % 12) (Nat.mod_lt _ (by norm_num))
  simp only [Nat.add_eq, Nat.mul_eq, Nat.div_add_mod'] at this
  simpa only [Nat.mul_eq] using this

/-- **mish ≈ 6 SiLU units on `[−6, 6]`, certified.** For every real `a ∈ [−6, 6]`,
`|mish a − Σᵢ αᵢ SiLU(βᵢ a + γᵢ)| ≤ 4·10⁻⁵`, for the six units of `afFit6` (the fit
`"mish|SiLU|6|shift"` of the activation-width rescue, `k = 6`, float sup error `1.68e-5`, with
`α, β` rounded to multiples of `2^-24` and `γ` to multiples of `2^-40`). -/
theorem actFit_mish_silu6 (a : ℝ) (ha : a ∈ Set.Icc (-6 : ℝ) 6) :
    |mish a - afFitR afFit6 a| ≤ 1 / 25000 := by
  obtain ⟨ha1, ha2⟩ := ha
  have h0 : 0 ≤ (a + 6) * 8 := by nlinarith
  set f := ⌊(a + 6) * 8⌋₊ with hf
  set c := min 95 f
  have hc : c < 96 := by omega
  have hcf : (c : ℝ) ≤ (a + 6) * 8 :=
    (Nat.cast_le.mpr (min_le_right _ _)).trans (Nat.floor_le h0)
  have hca : a ≤ ((c : ℝ) + 1) / 8 - 6 := by
    rcases le_total f 95 with h | h
    · have : c = f := min_eq_right h
      have := Nat.lt_floor_add_one ((a + 6) * 8)
      rw [← hf] at this
      have : (c : ℝ) = f := by exact_mod_cast ‹c = f›
      linarith
    · have : c = 95 := min_eq_left h
      rw [this]; push_cast; linarith
  have hb := actFit_chunk afFit6 afE6 256 (Nat.mul c 8192, 393216) 31 (actFit_chunks c hc) a
    (by simp only [afZv, Nat.mul_eq]; push_cast; linarith)
    (by simp only [afZv, Nat.mul_eq]; push_cast; linarith)
  refine hb.trans ?_
  norm_num [afE6]

/-- **The same, with the coefficients written out.** -/
theorem actFit_mish_silu6_explicit (a : ℝ) (ha : a ∈ Set.Icc (-6 : ℝ) 6) :
    |mish a - (-160505003 / 2 ^ 24 * silu (-3020478 / 2 ^ 24 * a + 20728877624 / 2 ^ 40) +
      17708581 / 2 ^ 24 * silu (-16340396 / 2 ^ 24 * a + 190579807737 / 2 ^ 40) +
      172903600 / 2 ^ 24 * silu (2903133 / 2 ^ 24 * a - 67410679846 / 2 ^ 40) +
      -10714356 / 2 ^ 24 * silu (19621551 / 2 ^ 24 * a - 988909254425 / 2 ^ 40) +
      8567841 / 2 ^ 24 * silu (-22279116 / 2 ^ 24 * a + 244274895690 / 2 ^ 40) +
      314572 / 2 ^ 24 * silu (-16954119 / 2 ^ 24 * a + 4129126169491 / 2 ^ 40))| ≤ 1 / 25000 := by
  have h := actFit_mish_silu6 a ha
  have e : afFitR afFit6 a = -160505003 / 2 ^ 24 * silu (-3020478 / 2 ^ 24 * a + 20728877624 / 2 ^ 40) +
      17708581 / 2 ^ 24 * silu (-16340396 / 2 ^ 24 * a + 190579807737 / 2 ^ 40) +
      172903600 / 2 ^ 24 * silu (2903133 / 2 ^ 24 * a - 67410679846 / 2 ^ 40) +
      -10714356 / 2 ^ 24 * silu (19621551 / 2 ^ 24 * a - 988909254425 / 2 ^ 40) +
      8567841 / 2 ^ 24 * silu (-22279116 / 2 ^ 24 * a + 244274895690 / 2 ^ 40) +
      314572 / 2 ^ 24 * silu (-16954119 / 2 ^ 24 * a + 4129126169491 / 2 ^ 40) := by
    simp only [afFitR, afFit6, afAl, afXR, afBe, afZv]
    push_cast
    ring_nf
  rwa [e] at h

end EP
