import EmulationProofs.Position
import EmulationProofs.Norm
import EmulationProofs.Attention
import EmulationProofs.MoE
import EmulationProofs.Activation
import EmulationProofs.Converses

/-!
# The edge-case audit's feature rules: switches, folds, routers, and the walls

The relation's `FEATURES` table (branch `edge-case-audit`, `graph/model_graph.py`) lets a host
WITH a feature play a target WITHOUT it when the feature can be switched off by a weight setting
("free" on the host side), and a host WITHOUT a feature play a target WITH it when a weight
setting reproduces it ("free" on the target side). Features the classifier does not record
(scalar multipliers, zero-centred norm weights) are treated as free both ways. This file proves
the switches and folds, refutes the rules that are not what the relation says, and proves the
walls that are tractable.

* **Switches** (`bias_off`, `out_gate_off`, `moe_shared_gate_off`, `extra_input_off`,
  `value_emb_off`, `switch_lora_off`, `router_bias_off`, `multi_rope_tables_equal`).
* **Folds** (`scalar_fold`, `attn_scale_fold`, `norm_weight_fold`, `zero_centred_norm`,
  `zero_centred_host`, `rmsNorm_smul_eps`, and the converse `stream_scalar_needs_eps`).
* **Routers** (`sigmoid_router_limit`, `softmax_router_ne_sigmoid`, `router_bias_not_bias_free`).
* **Tied embeddings** (`tied_sub_untied`).
* **Sinks** (`sink_host_factor`, `sink_host_ne_plain`: the sink host's fold is a limit, never
  exact; `bias_free_ffn_zero`: a start token's stream stays on its own channel).
* **Walls** (`biased_proj_not_bias_free`, `knorm_not_plain`, `vnorm_not_plain`).
-/

open Real Finset Filter Topology Matrix

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

/-! ## Switching a host feature off -/

/-- **A biased host plays an unbiased target** (exact, free): set the bias to `0`. Covers the
q / k / v projection biases and the output side (o_proj, MLP, LayerNorm `β`). -/
theorem bias_off {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    W *ᵥ x + (0 : Fin m → ℝ) = W *ᵥ x := add_zero _

/-- `2 σ(0) = 1`. -/
lemma two_mul_sigmoid_zero : (2 : ℝ) * sigmoid 0 = 1 := by rw [sigmoid_zero]; norm_num

/-- **An output-gated host plays an ungated target** (exact, free). The gate
`o ⊙ σ(W_g x)` with `W_g = 0` is `o / 2` on every token; doubling `o_proj` undoes it. -/
theorem out_gate_off {m n p : ℕ} (Wo : Matrix (Fin m) (Fin n) ℝ) (o : Fin n → ℝ)
    (x : Fin p → ℝ) :
    ((2 : ℝ) • Wo) *ᵥ (fun i => o i * sigmoid (((0 : Matrix (Fin n) (Fin p) ℝ) *ᵥ x) i)) =
      Wo *ᵥ o := by
  have : (fun i => o i * sigmoid (((0 : Matrix (Fin n) (Fin p) ℝ) *ᵥ x) i)) = (2⁻¹ : ℝ) • o := by
    funext i; simp [sigmoid_zero, mul_comm]
  rw [this, Matrix.smul_mulVec, Matrix.mulVec_smul, smul_smul]
  norm_num

/-- Scaling the down-projection scales the FFN. -/
lemma ffn_smul {X Θ V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {U : Type*} [Fintype U]
    (act : Θ → X → ℝ) (θ : U → Θ) (W2 : U → V) (c : ℝ) (x : X) :
    ffn act θ (fun u => c • W2 u) x = c • ffn act θ W2 x := by
  simp only [ffn, Finset.smul_sum, smul_comm c]

/-- **A host with a sigmoid-gated shared expert plays a target with an ungated one** (exact,
free). With the gate's input weights zeroed the gate is `σ(0) = 1/2` on every token; doubling the
shared down-projection undoes it (Qwen2-MoE `shared_expert_gate`). -/
theorem moe_shared_gate_off {X Θ V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    {E ew : ℕ} {Us : Type*} [Fintype Us] {n : ℕ} (act : Θ → X → ℝ) (gate : X → Fin E → ℝ)
    (θ : Fin E → Fin ew → Θ) (W2 : Fin E → Fin ew → V) (θs : Us → Θ) (W2s : Us → V)
    (xr : X) (xg : Fin n → ℝ) :
    moe act gate θ W2 xr + sigmoid ((0 : Fin n → ℝ) ⬝ᵥ xg) • ffn act θs (fun u => (2 : ℝ) • W2s u) xr
      = moe act gate θ W2 xr + ffn act θs W2s xr := by
  rw [ffn_smul, smul_smul, zero_dotProduct, mul_comm, two_mul_sigmoid_zero, one_smul]

/-- **Extra per-token inputs switched off** (exact, free): Engram memory, per-layer embeddings
(PLE), x0 / residual lambdas and smear all add a read of something the plain stack does not see
through a projection (or a scalar) that the host sets to `0`. -/
theorem extra_input_off {R S : Type*} [AddCommGroup R] [Module ℝ R] [AddCommGroup S]
    [Module ℝ S] (x : R) (u : S) : x + (0 : S →ₗ[ℝ] R) u = x := by simp

/-- **Value embeddings / value residuals switched off** (exact, free): with the mixing weight
`0`, the head's values are the plain values. -/
theorem value_emb_off {ι V : Type*} [Fintype ι] [NormedAddCommGroup V] [NormedSpace ℝ V]
    (S : Finset ι) (s : ι → ℝ) (v u : ι → V) :
    attn S s (fun j => v j + (0 : ℝ) • u j) = attn S s v := by simp

/-- **A switched-LoRA host plays a plain target** (exact, free): zero adapters leave the base
weights, whichever adapter the control tokens select. -/
theorem switch_lora_off {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (sel : Bool) (x : Fin n → ℝ) :
    W *ᵥ x + (if sel then (0 : Matrix (Fin m) (Fin n) ℝ) *ᵥ x else 0) = W *ᵥ x := by
  cases sel <;> simp

/-- A router that SELECTS experts on `l + b` and WEIGHTS them by `l` (aux-loss-free balancing). -/
def biasedRouter {E : ℕ} (sel : (Fin E → ℝ) → Finset (Fin E))
    (wt : (Fin E → ℝ) → Finset (Fin E) → Fin E → ℝ) (l b : Fin E → ℝ) : Fin E → ℝ :=
  wt l (sel (l + b))

/-- **A selection-biased router host plays a bias-free router target** (exact, free): `b = 0`. -/
theorem router_bias_off {E : ℕ} (sel : (Fin E → ℝ) → Finset (Fin E))
    (wt : (Fin E → ℝ) → Finset (Fin E) → Fin E → ℝ) (l : Fin E → ℝ) :
    biasedRouter sel wt l 0 = wt l (sel l) := by simp [biasedRouter]

/-- **A multi-table RoPE host plays a one-table target** (exact, free): every layer's table set
to the target's. -/
theorem multi_rope_tables_equal {Pr : Type*} [Fintype Pr] (ω : Pr → ℝ) (L : ℕ)
    (q k : Pr → ℝ × ℝ) (i j : ℝ) :
    ∀ l < L, ropeScore ((fun _ : ℕ => ω) l) q k i j = ropeScore ω q k i j := fun _ _ => rfl

/-! ## Scalar folds -/

/-- **Scalar multipliers fold into the adjacent matrix** (exact, free): embedding multiplier
(`α E`), residual / muP multipliers on a sublayer's output (`α W_o`), logit multiplier (`α` on
the lm_head, untied). -/
theorem scalar_fold {m n : ℕ} (α : ℝ) (W : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    α • (W *ᵥ x) = (α • W) *ᵥ x := (Matrix.smul_mulVec α W x).symm

/-- **Attention scale / `query_pre_attn_scalar` / attention multiplier** (exact, free): a scale
on the logit folds into `W_q`. -/
theorem attn_scale_fold {m n : ℕ} (α : ℝ) (Wq : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ)
    (k : Fin m → ℝ) : α * ((Wq *ᵥ x) ⬝ᵥ k) = ((α • Wq) *ᵥ x) ⬝ᵥ k := by
  rw [Matrix.smul_mulVec, smul_dotProduct, smul_eq_mul]

/-- **A scale after a norm folds into the norm weight** (exact, free): the place to put a logit
multiplier with TIED embeddings (the final norm), or an attention scale on a qk-norm host (the
q-norm weight). -/
theorem norm_weight_fold {n : ℕ} (α eps : ℝ) (w x : Fin n → ℝ) :
    α • rmsNorm eps w x = rmsNorm eps (α • w) x := by
  funext i; simp only [Pi.smul_apply, smul_eq_mul, rmsNorm]; ring

/-- **Zero-centred norm weights** (Gemma `(1 + w)`, exact reparametrisation): a standard host
plays a zero-centred target with weight `1 + w`. -/
theorem zero_centred_norm {n : ℕ} (eps : ℝ) (w x : Fin n → ℝ) :
    rmsNorm eps (fun i => 1 + w i) x = fun i => (1 + w i) * rmsNorm eps 1 x i := by
  funext i; simp only [rmsNorm, Pi.one_apply, one_mul]; ring

/-- …and a zero-centred host plays a standard target with parameter `w - 1`. -/
theorem zero_centred_host {n : ℕ} (eps : ℝ) (w x : Fin n → ℝ) :
    rmsNorm eps (fun i => 1 + (w i - 1)) x = rmsNorm eps w x := by
  congr 1; funext i; ring

/-- **A scale on the whole residual stream** (a layer scalar multiplying the stream) moves into
the next norm's epsilon: `RMSNorm_eps(λ x) = RMSNorm_{eps/λ²}(x)` for `λ > 0`. -/
theorem rmsNorm_smul_eps {n : ℕ} (eps lam : ℝ) (hl : 0 < lam) (w x : Fin n → ℝ) :
    rmsNorm eps w (lam • x) = rmsNorm (eps / lam ^ 2) w x := by
  funext i
  simp only [rmsNorm, Pi.smul_apply, smul_eq_mul]
  have e1 : (∑ l, (lam * x l) ^ 2) / n + eps = lam ^ 2 * ((∑ l, x l ^ 2) / n + eps / lam ^ 2) := by
    simp only [mul_pow, ← Finset.mul_sum]
    field_simp
  have hs : √((∑ l, (lam * x l) ^ 2) / n + eps) = lam * √((∑ l, x l ^ 2) / n + eps / lam ^ 2) := by
    rw [e1, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hl.le]
  rw [hs]
  rcases (Real.sqrt_nonneg ((∑ l, x l ^ 2) / n + eps / lam ^ 2)).lt_or_eq with h | h
  · field_simp
  · rw [← h]; simp

/-- **…so a stream scalar is NOT a free fold at a fixed epsilon** (converse). A host without the
scalar carries the stream divided by `λ`, and its norm (same `eps > 0`) misses the target's
`RMSNorm_eps(λ x)` at every coordinate where `x` and `w` are nonzero, for every `λ > 0`, `λ ≠ 1`.
The fold is exact only if the host's epsilon is `eps / λ²` (or `eps = 0`). -/
theorem stream_scalar_needs_eps {n : ℕ} (eps lam : ℝ) (he : 0 < eps) (hl : 0 < lam)
    (hl1 : lam ≠ 1) (w x : Fin n → ℝ) (i : Fin n) (hx : x i ≠ 0) (hw : w i ≠ 0) :
    rmsNorm eps w (lam • x) i ≠ rmsNorm eps w x i := by
  rw [rmsNorm_smul_eps eps lam hl]
  intro h
  have := rmsNorm_eps_inj _ _ (by positivity) he.le w x i hx hw h
  have hl2 : lam ^ 2 = 1 := by
    field_simp at this
    nlinarith
  have : lam = 1 := by nlinarith
  exact hl1 this

/-! ## Routers -/

/-- `e^s σ(z - s) = e^z / (e^{z-s} + 1)`. -/
lemma exp_mul_sigmoid_shift (z s : ℝ) : exp s * sigmoid (z - s) = exp z / (exp (z - s) + 1) := by
  rw [sigmoid_def, neg_sub]
  have h1 : exp (s - z) = exp s / exp z := by rw [exp_sub]
  have h2 : exp (z - s) = exp z / exp s := by rw [exp_sub]
  rw [h1, h2]
  have := exp_pos s; have := exp_pos z
  field_simp

/-- **A sigmoid router plays a softmax router** (limit, free): shifting every logit by `-s`,
`σ(z_e - s) / ∑_{j∈S} σ(z_j - s) → e^{z_e} / ∑_{j∈S} e^{z_j}` as `s → ∞`, over any selected set
`S` (the shift does not change the ranking, so the top-k set is the target's). -/
theorem sigmoid_router_limit {E : ℕ} (z : Fin E → ℝ) (S : Finset (Fin E)) (hS : S.Nonempty)
    (e : Fin E) :
    Tendsto (fun s => sigmoid (z e - s) / ∑ j ∈ S, sigmoid (z j - s)) atTop
      (𝓝 (exp (z e) / ∑ j ∈ S, exp (z j))) := by
  have hg : ∀ j, Tendsto (fun s => exp s * sigmoid (z j - s)) atTop (𝓝 (exp (z j))) := by
    intro j
    simp only [exp_mul_sigmoid_shift]
    have h0 : Tendsto (fun s => exp (z j - s)) atTop (𝓝 0) := by
      apply Real.tendsto_exp_atBot.comp
      exact tendsto_atBot_add_const_left _ _ tendsto_neg_atTop_atBot |>.congr (fun s => by ring)
    have h1 : Tendsto (fun s => exp (z j) * (exp (z j - s) + 1)⁻¹) atTop
        (𝓝 (exp (z j) * (0 + 1)⁻¹)) :=
      tendsto_const_nhds.mul ((h0.add_const 1).inv₀ (by norm_num))
    simpa [div_eq_mul_inv] using h1
  have hden : Tendsto (fun s => ∑ j ∈ S, exp s * sigmoid (z j - s)) atTop
      (𝓝 (∑ j ∈ S, exp (z j))) := tendsto_finsetSum _ (fun j _ => hg j)
  have hpos := sum_exp_pos hS z
  rw [div_eq_mul_inv]
  refine ((hg e).mul (hden.inv₀ hpos.ne')).congr (fun s => ?_)
  have hσ : 0 < ∑ j ∈ S, sigmoid (z j - s) := Finset.sum_pos (fun j _ => sigmoid_pos _) hS
  have := exp_pos s
  rw [← Finset.mul_sum]
  field_simp

/-- **A softmax router cannot play a sigmoid router** (impossibility, proved). Two experts, both
selected, renormalised: the sigmoid target with logits `(x, 0)` weights expert 0 by
`σ(x) / (σ(x) + σ(0))`. No softmax host with logits affine in `x` (bias allowed) gives the same
weight for every `x`: it would need `e^{a x + c} = 2 σ(x)`, and `log σ` is not affine. -/
theorem softmax_router_ne_sigmoid :
    ¬ ∃ a c : ℝ, ∀ x : ℝ,
      exp (a * x + c) / (exp (a * x + c) + 1) = sigmoid x / (sigmoid x + sigmoid 0) := by
  rintro ⟨a, c, h⟩
  have key : ∀ x, exp (a * x + c) = 2 * sigmoid x := by
    intro x
    have hx := h x
    have h1 := exp_pos (a * x + c); have h2 := sigmoid_pos x
    rw [sigmoid_zero, div_eq_div_iff (by positivity) (by positivity)] at hx
    nlinarith
  have k0 := key 0; have k1 := key 1; have km := key (-1)
  rw [sigmoid_zero, mul_zero, zero_add] at k0
  rw [sigmoid_neg] at km
  have hc : exp c = 1 := by linarith
  have hprod : exp (a * 1 + c) * exp (a * -1 + c) = 1 := by
    rw [← exp_add, show a * 1 + c + (a * -1 + c) = c + c by ring, exp_add, hc, one_mul]
  rw [k1, km] at hprod
  have hs : sigmoid 0 < sigmoid 1 := sigmoid_strictMono (by norm_num)
  rw [sigmoid_zero] at hs
  nlinarith

/-- **A bias-free router cannot play a selection-biased router** (impossibility, proved,
norm-free). Two experts, target logits `(x, 0)`, selection bias `(0, 1)`: expert 0 is selected
iff `x > 1`. A bias-free host router with logits `(a x, c x)` and index tie-break selects expert 0
iff `a x ≥ c x`; no `a, c` agree with the target at both `x = 2` and `x = 1/2`. -/
theorem router_bias_not_bias_free :
    ¬ ∃ a c : ℝ, ∀ x : ℝ, x ≠ 1 → (x + 0 > 0 + 1 ↔ a * x ≥ c * x) := by
  rintro ⟨a, c, h⟩
  have h2 := (h 2 (by norm_num)).mp (by norm_num)
  have h3 := (h (1 / 2) (by norm_num))
  have : ¬ (a * (1 / 2) ≥ c * (1 / 2)) := fun hh => by have := h3.mpr hh; norm_num at this
  apply this
  linarith

/-! ## Tied embeddings -/

/-- Logits of a head `U` (one row per token) on a final hidden state `h`. -/
def headLogits {Vb d : ℕ} (U : Fin Vb → Fin d → ℝ) (h : Fin d → ℝ) : Fin Vb → ℝ :=
  fun v => U v ⬝ᵥ h

/-- **An untied host plays a tied target** (exact, free): the host's lm_head is set to the
embedding table. -/
theorem tied_sub_untied {Vb d : ℕ} (E : Fin Vb → Fin d → ℝ) (h : Fin d → ℝ) :
    headLogits (fun v => E v) h = headLogits E h := rfl

/-! ## Sinks -/

/-- **A sink host is the plain head times a data-dependent factor**:
`sinkAttn s b v = (Z / (e^b + Z)) · attn s v`, `Z = ∑ e^{s_j}`. -/
theorem sink_host_factor {ι V : Type*} [Fintype ι] [NormedAddCommGroup V] [NormedSpace ℝ V]
    (s : ι → ℝ) (b : ℝ) (v : ι → V) :
    sinkAttn s b v = ((∑ j, exp (s j)) / (exp b + ∑ j, exp (s j))) • attn univ s v := by
  have : 0 < exp b := exp_pos b
  rcases Finset.univ.eq_empty_or_nonempty (α := ι) with hι | hι
  · simp [sinkAttn, attn, hι]
  · unfold sinkAttn attn
    rw [smul_smul]
    congr 1
    have := sum_exp_pos hι s
    field_simp

/-- **The sink fold is a limit, never exact** (refutes the relation's `exact` flag on
`sink: no -> yes`). With the target's logits and values and any finite sink logit `b`, the
sink host's output differs from the plain target's whenever the latter is nonzero
(`sink_to_plain_limit` gives the `b → -∞` limit). -/
theorem sink_host_ne_plain {ι V : Type*} [Fintype ι] [Nonempty ι] [NormedAddCommGroup V]
    [NormedSpace ℝ V] (s : ι → ℝ) (b : ℝ) (v : ι → V) (h : attn univ s v ≠ 0) :
    sinkAttn s b v ≠ attn univ s v := by
  rw [sink_host_factor]
  intro heq
  have hZ := sum_exp_pos (S := (univ : Finset ι)) univ_nonempty s
  have hb := exp_pos b
  have hne : (∑ j, exp (s j)) / (exp b + ∑ j, exp (s j)) - 1 ≠ 0 := by
    rw [div_sub_one (by positivity)]
    exact div_ne_zero (by linarith) (by positivity)
  have : ((∑ j, exp (s j)) / (exp b + ∑ j, exp (s j)) - 1) • attn univ s v = 0 := by
    rw [sub_smul, heq, one_smul, sub_self]
  exact h ((smul_eq_zero.mp this).resolve_left hne)

/-- **A start token's stream stays on its own channel.** A bias-free FFN (any activation with
`φ(0) = 0`, or any gated unit `φ(g)·u`) reading zero content writes zero, so a start token whose
embedding lives only in an extra channel that the target layers do not read keeps that
embedding at every layer (its attention, at position 0, sees only itself). This is the stream
half of the start-token sink construction; the head half is `sink_sub_plain`, with the sink
logit carried by a q bias in a zero-frequency pair (`sink_needs_zero_freq` shows the pair must
not rotate). -/
theorem bias_free_ffn_zero {Θ V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {U : Type*}
    [Fintype U] {n : ℕ} (act : Θ → (Fin n → ℝ) → ℝ) (θ : U → Θ) (W2 : U → V)
    (h0 : ∀ t, act t 0 = 0) : ffn act θ W2 0 = 0 := by
  simp [ffn, h0]

/-! ## Walls -/

lemma rmsNorm_neg {n : ℕ} (eps : ℝ) (w x : Fin n → ℝ) :
    rmsNorm eps w (-x) = -rmsNorm eps w x := by
  funext i; simp only [rmsNorm, Pi.neg_apply, neg_sq]; ring

/-- **A bias-free host cannot play a biased q / k / v projection** (impossibility, proved). A
bias-free host projection of its pre-norm output is ODD in the residual (for any weights, any
host norm weight and epsilon, and any linear interface `P` from the target residual to the host
residual), while the target's biased projection `W N(x) + b` is odd only if `b = 0`. So at `x` or
at `-x` the two differ. -/
theorem biased_proj_not_bias_free {n m k : ℕ} (eps eps' : ℝ) (w : Fin m → ℝ) (w' : Fin n → ℝ)
    (P : Matrix (Fin m) (Fin n) ℝ) (A : Matrix (Fin k) (Fin m) ℝ) (W : Matrix (Fin k) (Fin n) ℝ)
    (b : Fin k → ℝ) (hb : b ≠ 0) (x : Fin n → ℝ) :
    ¬ (A *ᵥ rmsNorm eps w (P *ᵥ x) = W *ᵥ rmsNorm eps' w' x + b ∧
       A *ᵥ rmsNorm eps w (P *ᵥ (-x)) = W *ᵥ rmsNorm eps' w' (-x) + b) := by
  rintro ⟨h1, h2⟩
  rw [Matrix.mulVec_neg, rmsNorm_neg, rmsNorm_neg, Matrix.mulVec_neg, Matrix.mulVec_neg, h1] at h2
  apply hb
  have : (2 : ℝ) • b = 0 := by
    rw [two_smul]
    have h3 : -(W *ᵥ rmsNorm eps' w' x + b) + (W *ᵥ rmsNorm eps' w' x + b) =
        -(W *ᵥ rmsNorm eps' w' x) + b + (W *ᵥ rmsNorm eps' w' x + b) := by rw [h2]
    rw [neg_add_cancel] at h3
    calc b + b = -(W *ᵥ rmsNorm eps' w' x) + b + (W *ᵥ rmsNorm eps' w' x + b) := by abel
      _ = 0 := h3.symm
  exact (smul_eq_zero.mp this).resolve_left two_ne_zero

/-- The rank-1 k-norm (or v-norm) of `x ↦ x₀`, weight `1`: the MLA normed logit. -/
lemma rmsNorm_one_dim (eps : ℝ) (x : Fin 2 → ℝ) :
    rmsNorm eps 1 (fun _ : Fin 1 => x 0) 0 = mlaNormLogit eps x := by
  simp [rmsNorm, mlaNormLogit]

/-- **A plain host cannot play a k-normed target** (impossibility, proved). A target head with
a k-norm (head dim 1, weight 1, any `eps ≥ 0`, query 1) has logit `RMSNorm(x₀)` for key input
`x`; no plain head (logit affine in the key input for a fixed query) gives the same softmax
weights on every two-key context of pre-normed inputs (the sphere of radius `√2 ρ`). This is
`mla_latent_norm_not_mha` read as a qk-norm: a per-token normalisation of a projection is not
a fixed linear map. -/
theorem knorm_not_plain (eps : ℝ) (heps : 0 ≤ eps) (ρ : ℝ) (hρ : 0 < ρ) :
    ¬ ∃ (a : Fin 2 → ℝ) (c : ℝ), ∀ x y : Fin 2 → ℝ,
      x 0 ^ 2 + x 1 ^ 2 = 2 * ρ ^ 2 → y 0 ^ 2 + y 1 ^ 2 = 2 * ρ ^ 2 →
      exp (a ⬝ᵥ x + c) / (exp (a ⬝ᵥ x + c) + exp (a ⬝ᵥ y + c)) =
        exp (rmsNorm eps 1 (fun _ : Fin 1 => x 0) 0) /
          (exp (rmsNorm eps 1 (fun _ : Fin 1 => x 0) 0) +
            exp (rmsNorm eps 1 (fun _ : Fin 1 => y 0) 0)) := by
  simp only [rmsNorm_one_dim]
  exact mla_latent_norm_not_mha eps heps ρ hρ

/-- **A plain host cannot play a v-normed target** (impossibility, proved). On a one-key context
a head outputs its value; a plain value is affine in the input, the v-normed value
`RMSNorm(x₀)` (weight-free, as Gemma-4's) is not, on the sphere of pre-normed inputs. -/
theorem vnorm_not_plain (eps : ℝ) (heps : 0 ≤ eps) (ρ : ℝ) (hρ : 0 < ρ) :
    ¬ ∃ (a : Fin 2 → ℝ) (c : ℝ), ∀ x : Fin 2 → ℝ, x 0 ^ 2 + x 1 ^ 2 = 2 * ρ ^ 2 →
      a ⬝ᵥ x + c = rmsNorm eps 1 (fun _ : Fin 1 => x 0) 0 := by
  simp only [rmsNorm_one_dim]
  exact mla_latent_norm_not_affine eps heps ρ hρ

end EP
