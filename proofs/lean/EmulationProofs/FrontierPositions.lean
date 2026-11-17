import EmulationProofs.Block
import EmulationProofs.Position

/-!
# fr-positions: positions walls of the payable frontier

Rules of branch `fr-positions` of the relation (`graph/model_graph.py`):

* **RoPE + NoPE host, target without NoPE layers** (`fp_nope_layer_host`, exact). A host whose
  NoPE blocks are zeroed (zeroed output projections: each block is the identity, whatever its
  attention computes -- Beam's code-fixed query temperature on its NoPE layers included,
  `zeroed_seqBlock_id`) is the stack of its RoPE blocks alone (`nope_mix_host_skip`); with those
  set to the target's blocks in order it is the target (`nope_mix_host_eq_target`). The relation
  charges it as a Hall matching of the target's rotating layers into the host's rotating layers.
* **fixed sinusoid table on a learned-absolute host** (`fp_sinusoid_table`, exact). A learned
  table holding the sinusoid's rows at the host's index offset, the target's embedding scale
  folded into the token table, gives the target's input embedding (`sinusoid_sub_learned_abs`).
* **sinusoidal host, sinusoidal target of another width** (`fp_sinusoid_same_table`, refutation).
  The XGLM-layout sinusoid's coordinate 1 at position 1 is `sin (exp (-c / (h - 1)))` with
  `c = log 10000` and `h` half the width; it is strictly increasing in `h`, so two widths give
  different tables (`sinusoid_width_differs`) and no per-token table `e'` absorbs the difference,
  which is 0 at position 0 (`sinusoid_shift_not_constant`).
-/

open Real

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

section Stack

variable {R : Type*}

/-- A stack of blocks, run first to last. -/
def runStack : List (R → R) → R → R
  | [], x => x
  | f :: bs, x => runStack bs (f x)

/-- A host stack: `none` is a zeroed (identity) block, `some f` a live block `f`. -/
def hostStack (hs : List (Option (R → R))) : List (R → R) := hs.map (fun o => o.getD id)

/-- **Identity blocks drop out of a stack** (exact): the host's zeroed blocks, wherever they sit,
leave the composition of its live blocks in order. -/
theorem nope_mix_host_skip : ∀ (hs : List (Option (R → R))) (x : R),
    runStack (hostStack hs) x = runStack (hs.filterMap id) x
  | [], _ => rfl
  | none :: hs, x => by
    simpa [hostStack, runStack] using nope_mix_host_skip hs x
  | some f :: hs, x => by
    simpa [hostStack, runStack] using nope_mix_host_skip hs (f x)

/-- **RoPE + NoPE host ⊇ a target without NoPE layers** (exact): the host's live (RoPE) blocks
set to the target's blocks in order, its NoPE blocks zeroed, equal the target on every input. -/
theorem nope_mix_host_eq_target (hs : List (Option (R → R))) (ts : List (R → R))
    (h : hs.filterMap id = ts) (x : R) : runStack (hostStack hs) x = runStack ts x := by
  rw [nope_mix_host_skip, h]

end Stack

section Zeroed

variable {R : Type*} [AddCommGroup R] [Module ℝ R]

/-- **A zeroed sequential block is the identity**, whatever its sublayers compute before their
output projections (`inA`, `inM`: the attention with any logits -- a position-dependent query
temperature, NoPE, any mask -- and the FFN). -/
theorem zeroed_seqBlock_id {S T : Type*} [AddCommGroup S] [Module ℝ S] [AddCommGroup T] [Module ℝ T]
    (inA : R → S) (inM : R → T) :
    seqBlock (fun x => (0 : S →ₗ[ℝ] R) (inA x)) (fun x => (0 : T →ₗ[ℝ] R) (inM x)) = id := by
  funext x; simp [seqBlock]

end Zeroed

section Sinusoid

variable {Vt D : Type*} [AddCommGroup D] [Module ℝ D]

/-- **fixed table ⊆ learned-absolute table** (exact). Target input embedding
`s_t • E (x t) + S t` (a fixed, e.g. sinusoid, table `S`); host input embedding
`s_h • E' (x t) + P (t + off)` (a learned table `P` read at the host's index offset, OPT / BioGPT
offset 2). With `E' = (s_t / s_h) • E` and `P r = S (r - off)` they are equal at every position. -/
theorem sinusoid_sub_learned_abs (E : Vt → D) (S : ℕ → D) (st sh : ℝ) (hsh : sh ≠ 0) (off : ℕ)
    (x : ℕ → Vt) (t : ℕ) :
    sh • ((st / sh) • E (x t)) + (fun r => S (r - off)) (t + off) = st • E (x t) + S t := by
  simp only [smul_smul, Nat.add_sub_cancel]
  rw [mul_div_cancel₀ st hsh]

/-- The XGLM-layout sinusoid frequency of coordinate 1 at half-width `h`: `exp (-c / (h - 1))`. -/
def sinFreq1 (c : ℝ) (h : ℕ) : ℝ := exp (-c / ((h : ℝ) - 1))

lemma sinFreq1_mem {c : ℝ} (hc : 0 < c) {h : ℕ} (hh : 2 ≤ h) :
    sinFreq1 c h ∈ Set.Icc (-(π / 2)) (π / 2) := by
  have h1 : (1 : ℝ) < h := by exact_mod_cast hh
  have hpos : 0 < (h : ℝ) - 1 := by linarith
  have hle : sinFreq1 c h ≤ 1 := by
    unfold sinFreq1
    apply Real.exp_le_one_iff.mpr
    have : 0 < c / ((h : ℝ) - 1) := div_pos hc hpos
    rw [neg_div]; linarith
  have hpi : (1 : ℝ) ≤ π / 2 := by linarith [Real.pi_gt_three]
  refine ⟨?_, le_trans hle hpi⟩
  have : 0 < sinFreq1 c h := Real.exp_pos _
  linarith [Real.pi_pos]

/-- **sinusoid tables of two widths differ** (refutation of sinusoidal ← sinusoidal across
widths): for half-widths `2 ≤ h < H` and any `c > 0` (XGLM / M2M100: `c = log 10000`),
coordinate 1 at position 1 is strictly smaller at `h`. -/
theorem sinusoid_width_differs {c : ℝ} (hc : 0 < c) {h H : ℕ} (hh : 2 ≤ h) (hH : h < H) :
    sin (1 * sinFreq1 c h) < sin (1 * sinFreq1 c H) := by
  rw [one_mul, one_mul]
  apply Real.strictMonoOn_sin (sinFreq1_mem hc hh) (sinFreq1_mem hc (by omega))
  unfold sinFreq1
  apply Real.exp_lt_exp.mpr
  have h1 : (1 : ℝ) < h := by exact_mod_cast hh
  have hhH : (h : ℝ) < H := by exact_mod_cast hH
  have hp : 0 < (h : ℝ) - 1 := by linarith
  have hP : 0 < (H : ℝ) - 1 := by linarith
  rw [neg_div, neg_div, neg_lt_neg_iff]
  exact div_lt_div_of_pos_left hc hp (by linarith)

/-- **a per-token table cannot absorb a position-dependent difference**: if `e' v + A t = e v + B t`
for every token `v` and position `t`, then `A t - B t` does not depend on `t`. A host's sinusoid
`A` and the target's `B` agree at position 0 (`sin 0 = 0`, `cos 0 = 1`) and differ at position 1
(`sinusoid_width_differs`), so no host token table `e'` repairs them. -/
theorem sinusoid_shift_not_constant {V : Type*} (e e' : V → D) (A B : ℕ → D)
    (h : ∀ v t, e' v + A t = e v + B t) (v : V) (t s : ℕ) : A t - B t = A s - B s := by
  have h1 := h v t
  have h2 := h v s
  have : A t - B t = e v - e' v := by rw [sub_eq_sub_iff_add_eq_add]; rw [add_comm (A t), h1, add_comm]
  have h3 : A s - B s = e v - e' v := by rw [sub_eq_sub_iff_add_eq_add]; rw [add_comm (A s), h2, add_comm]
  rw [this, h3]

end Sinusoid

end EP
