import EmulationProofs.Common

/-!
# Attention: KV tiling, MLA, masks, sinks, K = V, and softmax ⊄ linear

A head reads a query input `xq` and key/value inputs `xs j` (one per key `j` in a finite set
`S`); its logits are `(W_q xq) · (W_k xs j)` and its values `W_v xs j`. A softmax scale and
biases fold into `W_q` / the input, so they are omitted where they play no role.
-/

open Real Finset Filter Topology Matrix

noncomputable section

set_option linter.unusedSectionVars false

namespace EP

variable {ι : Type*} [Fintype ι] {n dh dv : ℕ}

/-- One softmax attention head. -/
def head (Wq Wk : Matrix (Fin dh) (Fin n) ℝ) (Wv : Matrix (Fin dv) (Fin n) ℝ) (S : Finset ι)
    (xq : Fin n → ℝ) (xs : ι → Fin n → ℝ) : Fin dv → ℝ :=
  attn S (fun j => (Wq *ᵥ xq) ⬝ᵥ (Wk *ᵥ xs j)) (fun j => Wv *ᵥ xs j)

/-! ## MQA ⊆ GQA ⊆ MHA (tiling) -/

/-- Grouped-query attention: query head `h` uses KV head `g h`. MQA is `G = 1`, MHA is `g = id`. -/
def gqa {H G : ℕ} (Wq : Fin H → Matrix (Fin dh) (Fin n) ℝ) (Wk : Fin G → Matrix (Fin dh) (Fin n) ℝ)
    (Wv : Fin G → Matrix (Fin dv) (Fin n) ℝ) (g : Fin H → Fin G) (S : Finset ι) (xq : Fin n → ℝ)
    (xs : ι → Fin n → ℝ) (h : Fin H) : Fin dv → ℝ :=
  head (Wq h) (Wk (g h)) (Wv (g h)) S xq xs

/-- **GQA ⊆ MHA** (exact, KV). An MHA host (`g = id`) with KV heads tiled, `W_k,h := W_k,g(h)`,
reproduces every GQA head's output for every input. -/
theorem gqa_sub_mha {H G : ℕ} (Wq : Fin H → Matrix (Fin dh) (Fin n) ℝ)
    (Wk : Fin G → Matrix (Fin dh) (Fin n) ℝ) (Wv : Fin G → Matrix (Fin dv) (Fin n) ℝ)
    (g : Fin H → Fin G) (S : Finset ι) (xq : Fin n → ℝ) (xs : ι → Fin n → ℝ) :
    gqa Wq (fun h => Wk (g h)) (fun h => Wv (g h)) id S xq xs = gqa Wq Wk Wv g S xq xs := rfl

/-- The `repeat_interleave` grouping: `H = G r` query heads, head `h` reads KV head `h / r`. -/
def groupOf {G r : ℕ} (h : Fin (G * r)) : Fin G :=
  ⟨h / r, Nat.div_lt_of_lt_mul (by simp [mul_comm])⟩

/-- **MQA ⊆ GQA** (exact, KV). A GQA host with `G` KV heads, every one a copy of the single MQA
head, reproduces MQA for any grouping. -/
theorem mqa_sub_gqa {H G : ℕ} (Wq : Fin H → Matrix (Fin dh) (Fin n) ℝ)
    (Wk : Matrix (Fin dh) (Fin n) ℝ) (Wv : Matrix (Fin dv) (Fin n) ℝ) (g : Fin H → Fin G)
    (S : Finset ι) (xq : Fin n → ℝ) (xs : ι → Fin n → ℝ) :
    gqa Wq (fun _ : Fin G => Wk) (fun _ => Wv) g S xq xs =
      gqa Wq (fun _ : Fin 1 => Wk) (fun _ => Wv) (fun _ => 0) S xq xs := rfl

/-- **MHA ⊆ MQA, the wide construction** (swing table, construction B; exact, KV width).
One shared KV head of width `H·dh` (keys) and `H·dv` (values) holds every head's key and value
in its own slice; query head `h` is zero outside slice `h`, so its logits are exactly MHA head
`h`'s, and the output projection reads slice `h` of its output. -/
theorem mha_sub_wide_mqa {H : ℕ} (Wq Wk : Fin H → Matrix (Fin dh) (Fin n) ℝ)
    (Wv : Fin H → Matrix (Fin dv) (Fin n) ℝ) (S : Finset ι) (xq : Fin n → ℝ)
    (xs : ι → Fin n → ℝ) (h : Fin H) :
    let WkW : Matrix (Fin H × Fin dh) (Fin n) ℝ := fun p c => Wk p.1 p.2 c
    let WvW : Matrix (Fin H × Fin dv) (Fin n) ℝ := fun p c => Wv p.1 p.2 c
    let WqW : Matrix (Fin H × Fin dh) (Fin n) ℝ := fun p c => if p.1 = h then Wq h p.2 c else 0
    (fun t : Fin dv => attn S (fun j => (WqW *ᵥ xq) ⬝ᵥ (WkW *ᵥ xs j)) (fun j => WvW *ᵥ xs j) (h, t))
      = head (Wq h) (Wk h) (Wv h) S xq xs := by
  intro WkW WvW WqW
  have hs : ∀ j, (WqW *ᵥ xq) ⬝ᵥ (WkW *ᵥ xs j) = (Wq h *ᵥ xq) ⬝ᵥ (Wk h *ᵥ xs j) := by
    intro j
    simp only [dotProduct, Fintype.sum_prod_type, mulVec, WqW, WkW]
    rw [Finset.sum_eq_single h]
    · simp
    · intro b _ hb; simp [hb]
    · simp
  funext t
  simp only [head, attn, hs]
  simp only [Pi.smul_apply, Finset.sum_apply]
  rfl

/-! ## MLA ⊆ MHA (norm-free) and what the latent RMSNorm breaks -/

/-- Norm-free MLA head: the keys and values are up-projections of a shared latent
`c_j = W_DKV xs j` of rank `r`. -/
def mlaHead {r : ℕ} (Wq : Matrix (Fin dh) (Fin n) ℝ) (Wdkv : Matrix (Fin r) (Fin n) ℝ)
    (Wuk : Matrix (Fin dh) (Fin r) ℝ) (Wuv : Matrix (Fin dv) (Fin r) ℝ) (S : Finset ι)
    (xq : Fin n → ℝ) (xs : ι → Fin n → ℝ) : Fin dv → ℝ :=
  attn S (fun j => (Wq *ᵥ xq) ⬝ᵥ (Wuk *ᵥ (Wdkv *ᵥ xs j))) (fun j => Wuv *ᵥ (Wdkv *ᵥ xs j))

/-- **MLA ⊆ MHA, norm-free form** (exact, KV). Materialising `W_K = W_UK W_DKV`,
`W_V = W_UV W_DKV` gives an MHA head equal to the MLA head on every input. -/
theorem mla_sub_mha {r : ℕ} (Wq : Matrix (Fin dh) (Fin n) ℝ) (Wdkv : Matrix (Fin r) (Fin n) ℝ)
    (Wuk : Matrix (Fin dh) (Fin r) ℝ) (Wuv : Matrix (Fin dv) (Fin r) ℝ) (S : Finset ι)
    (xq : Fin n → ℝ) (xs : ι → Fin n → ℝ) :
    head Wq (Wuk * Wdkv) (Wuv * Wdkv) S xq xs = mlaHead Wq Wdkv Wuk Wuv S xq xs := by
  simp [head, mlaHead, Matrix.mulVec_mulVec]

/-- Equal softmax weights on a two-key context force equal logit differences. -/
lemma logit_diff_of_weights {s₁ s₂ t₁ t₂ : ℝ}
    (h : exp s₁ / (exp s₁ + exp s₂) = exp t₁ / (exp t₁ + exp t₂)) : s₁ - s₂ = t₁ - t₂ := by
  have h1 := exp_pos s₁; have h2 := exp_pos s₂; have h3 := exp_pos t₁; have h4 := exp_pos t₂
  rw [div_eq_div_iff (by positivity) (by positivity)] at h
  have : exp (s₁ + t₂) = exp (t₁ + s₂) := by rw [exp_add, exp_add]; linarith
  have := exp_injective this
  linarith

/-- The real MLA key path with the latent RMSNorm (DeepSeek's `kv_a_layernorm`), in the
smallest case: latent rank `1`, `W_DKV = [1, 0]`, `W_UK = 1`, norm weight `1`, any `eps ≥ 0`,
query `1`. The logit of key input `x` is `x₀ / √(x₀² + eps)`. -/
def mlaNormLogit (eps : ℝ) (x : Fin 2 → ℝ) : ℝ := x 0 / √(x 0 ^ 2 / 1 + eps)

/-- **MLA with its latent RMSNorm ⊄ MHA** (counterexample, proved). For a fixed query, every
MHA head's logit is an affine function `a · x + c` of the key-side input `x` (key bias
included). On the sphere of pre-normed inputs of any radius, no affine function matches the
normed-MLA logit up to a constant shift — so no MHA head reproduces this MLA head's attention
weights, even restricted to two-key contexts of RMS-normalised inputs. -/
theorem mla_latent_norm_not_affine (eps : ℝ) (heps : 0 ≤ eps) (ρ : ℝ) (hρ : 0 < ρ) :
    ¬ ∃ (a : Fin 2 → ℝ) (c : ℝ), ∀ x : Fin 2 → ℝ, x 0 ^ 2 + x 1 ^ 2 = 2 * ρ ^ 2 →
      a ⬝ᵥ x + c = mlaNormLogit eps x := by
  rintro ⟨a, c, h⟩
  have p1 := h ![ρ, ρ] (by simp; ring)
  have p2 := h ![ρ, -ρ] (by simp; ring)
  have p3 := h ![-ρ, ρ] (by simp; ring)
  have p4 := h ![√2 * ρ, 0] (by simp [mul_pow])
  simp only [mlaNormLogit, dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero,
    Matrix.cons_val_one, div_one, neg_sq, mul_neg, neg_div, mul_zero, add_zero] at p1 p2 p3 p4
  have hu : 0 < √(ρ ^ 2 + eps) := Real.sqrt_pos.mpr (by positivity)
  have hw : 0 < √((√2 * ρ) ^ 2 + eps) := Real.sqrt_pos.mpr (by positivity)
  have ha0 : a 0 * ρ = ρ / √(ρ ^ 2 + eps) := by linarith
  have e1 : a 0 = 1 / √(ρ ^ 2 + eps) := by
    field_simp at ha0 ⊢; nlinarith [ha0]
  have hc : c = 0 := by linarith
  subst hc
  have e2 : a 0 = 1 / √((√2 * ρ) ^ 2 + eps) := by
    have : a 0 * (√2 * ρ) = √2 * ρ / √((√2 * ρ) ^ 2 + eps) := by linarith
    have h2 : 0 < √2 * ρ := by positivity
    field_simp at this ⊢; nlinarith [this]
  have key := e1.symm.trans e2
  rw [one_div, one_div, _root_.inv_inj, Real.sqrt_inj (by positivity) (by positivity), mul_pow,
    Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)] at key
  nlinarith

/-- **MLA with its latent RMSNorm ⊄ MHA, at the attention weights.** No MHA head (for a fixed
query its logit is `a · x + c`) gives the same softmax weights as the normed-MLA head on every
two-key context drawn from the sphere of pre-normed inputs. -/
theorem mla_latent_norm_not_mha (eps : ℝ) (heps : 0 ≤ eps) (ρ : ℝ) (hρ : 0 < ρ) :
    ¬ ∃ (a : Fin 2 → ℝ) (c : ℝ), ∀ x y : Fin 2 → ℝ,
      x 0 ^ 2 + x 1 ^ 2 = 2 * ρ ^ 2 → y 0 ^ 2 + y 1 ^ 2 = 2 * ρ ^ 2 →
      exp (a ⬝ᵥ x + c) / (exp (a ⬝ᵥ x + c) + exp (a ⬝ᵥ y + c)) =
        exp (mlaNormLogit eps x) / (exp (mlaNormLogit eps x) + exp (mlaNormLogit eps y)) := by
  rintro ⟨a, c, h⟩
  apply mla_latent_norm_not_affine eps heps ρ hρ
  let y0 : Fin 2 → ℝ := ![ρ, ρ]
  have hy0 : y0 0 ^ 2 + y0 1 ^ 2 = 2 * ρ ^ 2 := by simp [y0]; ring
  refine ⟨a, mlaNormLogit eps y0 - a ⬝ᵥ y0, fun x hx => ?_⟩
  have := logit_diff_of_weights (h x y0 hx hy0)
  linarith

/-- **MLA with its latent RMSNorm ⊄ MHA, over all inputs.** A logit affine in the key-side input
that agrees with a BOUNDED function (every RMSNorm output is bounded coordinatewise by `√r`) up
to a constant, on all of `ℝⁿ`, has zero slope: the head would ignore its keys. -/
theorem affine_eq_bounded_slope_zero (a : Fin n → ℝ) (c B : ℝ) (t : (Fin n → ℝ) → ℝ)
    (hB : ∀ x, |t x| ≤ B) (h : ∀ x, a ⬝ᵥ x + c = t x) : a = 0 := by
  by_contra ha
  have hpos : 0 < a ⬝ᵥ a := by
    rw [dotProduct]
    obtain ⟨i, hi⟩ := Function.ne_iff.mp ha
    exact Finset.sum_pos' (fun j _ => mul_self_nonneg _) ⟨i, by simp, by
      simpa using mul_self_pos.mpr hi⟩
  set s := (|B| + |c| + 1) / (a ⬝ᵥ a)
  have hx := h (s • a)
  have hb := hB (s • a)
  rw [← hx, dotProduct_smul, smul_eq_mul] at hb
  have : s * (a ⬝ᵥ a) = |B| + |c| + 1 := by
    simp only [s]; field_simp
  rw [this] at hb
  have := abs_le.mp hb
  have := le_abs_self B; have := neg_abs_le c
  have := abs_nonneg B
  linarith [this]

/-- Every coordinate of an RMSNorm output is bounded by `|w i| √n` (for `eps ≥ 0`). -/
theorem rmsNorm_bounded (eps : ℝ) (heps : 0 ≤ eps) (w x : Fin n → ℝ) (i : Fin n) :
    |rmsNorm eps w x i| ≤ |w i| * √n := by
  unfold rmsNorm
  rw [abs_div, abs_mul, mul_div_assoc]
  gcongr
  set D := √((∑ l, x l ^ 2) / n + eps)
  rcases (Real.sqrt_nonneg ((∑ l, x l ^ 2) / n + eps)).lt_or_eq with hD | hD
  · rw [abs_of_pos hD, div_le_iff₀ hD]
    have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
    have hsq : x i ^ 2 ≤ ∑ l, x l ^ 2 :=
      Finset.single_le_sum (f := fun l => x l ^ 2) (fun l _ => sq_nonneg _) (Finset.mem_univ i)
    rw [← Real.sqrt_sq_eq_abs, ← Real.sqrt_mul (by positivity)]
    apply Real.sqrt_le_sqrt
    calc x i ^ 2 ≤ ∑ l, x l ^ 2 := hsq
      _ = n * ((∑ l, x l ^ 2) / n) := by field_simp
      _ ≤ n * ((∑ l, x l ^ 2) / n + eps) := by nlinarith
  · simp only [D, ← hD, div_zero, abs_zero]; positivity

/-! ## Masks: local ⊆ full, sparse ⊆ full, AR ⊆ diffusion -/

/-- A full-attention host with a `0/1` mask `m`: the softmax runs over the unmasked keys
(the `{0, -∞}` additive mask). -/
def maskedAttn {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (m : ι → Prop)
    [DecidablePred m] (s : ι → ℝ) (v : ι → V) : V :=
  attn (univ.filter m) s v

/-- **sparse ⊆ full, local ⊆ full, causal ⊆ bidirectional** (exact, mask). A target attending
over ANY key set `S` (a band, a strided / sparse pattern, the causal prefix) is the full host
with the mask `m j := j ∈ S`, no weight change. -/
theorem pattern_sub_full {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (S : Finset ι)
    (s : ι → ℝ) (v : ι → V) [DecidableEq ι] : maskedAttn (· ∈ S) s v = attn S s v := by
  simp [maskedAttn]

/-- The band (local window) of width `w` at query position `i`. -/
def band {T : ℕ} (w : ℕ) (i : Fin T) : Finset (Fin T) :=
  univ.filter (fun j => j ≤ i ∧ (i : ℕ) < j + w)

/-- **local ⊆ full** (exact, mask): the window target is the full host with the band mask. -/
theorem local_sub_full {T : ℕ} {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (w : ℕ)
    (i : Fin T) (s : Fin T → ℝ) (v : Fin T → V) :
    maskedAttn (· ∈ band w i) s v = attn (band w i) s v := pattern_sub_full _ s v

/-- **The `-∞` mask is the `B → ∞` limit of a finite additive mask `-B`.** -/
theorem finite_mask_limit {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (m : ι → Prop)
    [DecidablePred m] (hm : ∃ j, m j) (s : ι → ℝ) (v : ι → V) :
    Tendsto (fun B : ℝ => attn univ (fun j => s j + B * (if m j then 0 else -1)) v) atTop
      (𝓝 (maskedAttn m s v)) := by
  have := attn_bias_limit s (fun j => if m j then 0 else -1) v
    (fun j => by split_ifs <;> norm_num) (by obtain ⟨j, hj⟩ := hm; exact ⟨j, by simp [hj]⟩)
  have hf : univ.filter (fun j => (if m j then (0 : ℝ) else -1) = 0) = univ.filter m := by
    ext j; by_cases hj : m j <;> simp [hj]
  rw [maskedAttn, ← hf]; exact this

/-- A sequence map is causal if its output at `t` depends only on inputs at positions `≤ t`. -/
def Causal {T : ℕ} {A B : Type*} (F : (Fin T → A) → (Fin T → B)) : Prop :=
  ∀ X Y t, (∀ s ≤ t, X s = Y s) → F X t = F Y t

/-- A causally masked attention layer (any per-position query, key, value and output maps, which
need not be linear) is causal. -/
theorem causal_attn_layer {T : ℕ} {A V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (score : A → A → ℝ) (val : A → V) (out : A → V → A) :
    Causal (fun X : Fin T → A => fun t =>
      out (X t) (attn (univ.filter (· ≤ t)) (fun s => score (X t) (X s)) (fun s => val (X s)))) := by
  intro X Y t h
  have ht := h t le_rfl
  have h1 : ∑ s ∈ univ.filter (· ≤ t), exp (score (Y t) (X s)) =
      ∑ s ∈ univ.filter (· ≤ t), exp (score (Y t) (Y s)) :=
    Finset.sum_congr rfl (fun s hs => by rw [h s (Finset.mem_filter.mp hs).2])
  have h2 : ∑ s ∈ univ.filter (· ≤ t), exp (score (Y t) (X s)) • val (X s) =
      ∑ s ∈ univ.filter (· ≤ t), exp (score (Y t) (Y s)) • val (Y s) :=
    Finset.sum_congr rfl (fun s hs => by rw [h s (Finset.mem_filter.mp hs).2])
  simp only [attn, ht, h1, h2]

theorem Causal.comp {T : ℕ} {A B C : Type*} {F : (Fin T → A) → (Fin T → B)}
    {G : (Fin T → B) → (Fin T → C)} (hF : Causal F) (hG : Causal G) : Causal (G ∘ F) :=
  fun X Y t h => hG (F X) (F Y) t (fun s hs => hF X Y s (fun u hu => h u (hu.trans hs)))

/-- **AR ⊆ diffusion** (exact, mask). A stack of bidirectional layers given the causal mask is
causal: position `t`'s output (the next-token logits) is a function of tokens `≤ t` only, so
left-to-right unmasking reproduces autoregressive decoding. -/
theorem causal_stack {T : ℕ} {A : Type*} (layers : List ((Fin T → A) → (Fin T → A)))
    (h : ∀ F ∈ layers, Causal F) : Causal (layers.foldr (· ∘ ·) id) := by
  induction layers with
  | nil => intro X Y t hXY; exact hXY t le_rfl
  | cons F rest ih =>
    simp only [List.foldr_cons]
    exact Causal.comp (ih (fun G hG => h G (List.mem_cons_of_mem _ hG))) (h F (by simp))

/-! ## Attention sinks -/

/-- Sink attention (gpt-oss): a per-head logit `b` in the denominator only. -/
def sinkAttn {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (s : ι → ℝ) (b : ℝ)
    (v : ι → V) : V :=
  (exp b + ∑ j, exp (s j))⁻¹ • ∑ j, exp (s j) • v j

/-- **sink ⊆ plain** (exact, KV + 1). A plain host with one extra key slot (`none`) whose logit is
`b` and whose value is `0` equals the sink head. The logit is realised with one constant channel:
query `(q, 1)`, real keys `(k j, 0)`, sink key `(0, b)`. -/
theorem sink_sub_plain {d : ℕ} {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (q : Fin d → ℝ) (k : ι → Fin d → ℝ) (b : ℝ) (v : ι → V) :
    let q' : Option (Fin d) → ℝ := fun o => o.elim 1 q
    let k' : Option ι → Option (Fin d) → ℝ := fun j => j.elim (fun o => o.elim b 0)
      (fun j o => o.elim 0 (k j))
    let v' : Option ι → V := fun j => j.elim 0 v
    attn univ (fun j => q' ⬝ᵥ k' j) v' = sinkAttn (fun j => q ⬝ᵥ k j) b v := by
  intro q' k' v'
  simp [attn, sinkAttn, Fintype.sum_option, dotProduct, q', k', v']

/-- **plain ⊆ sink** (limit, free): the sink logit `b → -∞` recovers plain softmax. -/
theorem sink_to_plain_limit {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] [Nonempty ι]
    (s : ι → ℝ) (v : ι → V) :
    Tendsto (fun b => sinkAttn s b v) atBot (𝓝 (attn univ s v)) := by
  unfold sinkAttn attn
  have hpos := sum_exp_pos (S := (univ : Finset ι)) univ_nonempty s
  have : Tendsto (fun b => exp b + ∑ j, exp (s j)) atBot (𝓝 (0 + ∑ j, exp (s j))) :=
    Real.tendsto_exp_atBot.add_const _
  rw [zero_add] at this
  exact (this.inv₀ hpos.ne').smul_const _

/-! ## Gemma K = V ⊆ separate K, V -/

/-- A head with separate key and value projections, each followed by its own fixed post-map
(`kpost`: k-norm and RoPE; `vpost`: the v-norm). -/
def kvHead {dk : ℕ} (kpost : (Fin dk → ℝ) → (Fin dk → ℝ)) (vpost : (Fin dk → ℝ) → (Fin dk → ℝ))
    (q : Fin dk → ℝ) (Wk Wv : Matrix (Fin dk) (Fin n) ℝ) (S : Finset ι) (xs : ι → Fin n → ℝ) :
    Fin dk → ℝ :=
  attn S (fun j => q ⬝ᵥ kpost (Wk *ᵥ xs j)) (fun j => vpost (Wv *ᵥ xs j))

/-- **Gemma K = V ⊆ separate K, V** (exact, free). Gemma-4's `attention_k_eq_v` uses the RAW
`k_proj` output as the value (before k-norm and RoPE, then its own v-norm). A host with separate
projections reproduces it with `W_V := W_K` (and the same v-norm). -/
theorem k_eq_v_sub_separate {dk : ℕ} (kpost vpost : (Fin dk → ℝ) → (Fin dk → ℝ))
    (q : Fin dk → ℝ) (Wk : Matrix (Fin dk) (Fin n) ℝ) (S : Finset ι) (xs : ι → Fin n → ℝ) :
    kvHead kpost vpost q Wk Wk S xs =
      attn S (fun j => q ⬝ᵥ kpost (Wk *ᵥ xs j)) (fun j => vpost (Wk *ᵥ xs j)) := rfl

/-! ## softmax ⊄ linear: no finite feature map -/

/-- **No finite feature map reproduces `exp (q k)`** (impossibility, proved). For every feature
count `m` there are no `φ, ψ : ℝ → ℝᵐ` with `exp (q k) = φ(q) · ψ(k)` for all scalars `q, k`.
Proof: the `(m+1) × (m+1)` matrix `exp (i j)` is a Vandermonde matrix in the distinct nodes
`exp i`, so it has rank `m + 1`, but a factorisation through `ℝᵐ` has rank `≤ m`. -/
theorem no_finite_feature_map (m : ℕ) :
    ¬ ∃ (φ ψ : ℝ → Fin m → ℝ), ∀ q k : ℝ, exp (q * k) = φ q ⬝ᵥ ψ k := by
  rintro ⟨φ, ψ, h⟩
  let A : Matrix (Fin (m + 1)) (Fin (m + 1)) ℝ := Matrix.vandermonde (fun i => exp i)
  let Φ : Matrix (Fin (m + 1)) (Fin m) ℝ := fun i l => φ i l
  let Ψ : Matrix (Fin m) (Fin (m + 1)) ℝ := fun l j => ψ j l
  have hA : A = Φ * Ψ := by
    ext i j
    simp only [A, Matrix.vandermonde_apply, Φ, Ψ]
    rw [← Real.exp_nat_mul, mul_comm, h]
    rfl
  have hinj : Function.Injective (fun i : Fin (m + 1) => exp (i : ℝ)) := by
    intro i j hij
    have := Real.exp_injective hij
    exact Fin.ext (by exact_mod_cast this)
  have hdet : A.det ≠ 0 := Matrix.det_vandermonde_ne_zero_iff.mpr hinj
  have hrank : A.rank = m + 1 := by
    rw [Matrix.rank_of_isUnit A ((Matrix.isUnit_iff_isUnit_det A).mpr (isUnit_iff_ne_zero.mpr hdet))]
    simp
  have : A.rank ≤ m := by
    rw [hA]
    exact (Matrix.rank_mul_le_left Φ Ψ).trans (by simpa using Matrix.rank_le_card_width Φ)
  omega

/-- The same in any head dimension `d ≥ 1` (restrict to a line). -/
theorem no_finite_feature_map_vec (d m : ℕ) (hd : 0 < d) :
    ¬ ∃ (φ ψ : (Fin d → ℝ) → Fin m → ℝ), ∀ q k, exp (q ⬝ᵥ k) = φ q ⬝ᵥ ψ k := by
  rintro ⟨φ, ψ, h⟩
  let e : Fin d := ⟨0, hd⟩
  apply no_finite_feature_map m
  refine ⟨fun q => φ (Pi.single e q), fun k => ψ (Pi.single e k), fun q k => ?_⟩
  rw [← h]
  simp

/-- **Normalised linear attention cannot reproduce softmax attention either.** If a kernel
`A(q, k) = φ(q) · ψ(k)` gives the same attention weights as softmax on every two-key context,
then `exp (q k) = φ'(q) · ψ(k)` for a rescaled finite feature map — impossible. -/
theorem linear_attn_ne_softmax (m : ℕ) (φ ψ : ℝ → Fin m → ℝ)
    (h : ∀ q k₁ k₂ : ℝ, (φ q ⬝ᵥ ψ k₁) / (φ q ⬝ᵥ ψ k₁ + φ q ⬝ᵥ ψ k₂) =
      exp (q * k₁) / (exp (q * k₁) + exp (q * k₂))) : False := by
  have h0 : ∀ q, φ q ⬝ᵥ ψ 0 ≠ 0 := by
    intro q hq
    have := h q 0 0
    rw [hq] at this
    norm_num at this
  apply no_finite_feature_map m
  refine ⟨fun q => (φ q ⬝ᵥ ψ 0)⁻¹ • φ q, ψ, fun q k => ?_⟩
  have hk := h q k 0
  simp only [mul_zero, exp_zero] at hk
  have hden : φ q ⬝ᵥ ψ k + φ q ⬝ᵥ ψ 0 ≠ 0 := by
    intro h'
    rw [h', div_zero] at hk
    have : 0 < exp (q * k) / (exp (q * k) + 1) := by positivity
    linarith
  rw [div_eq_div_iff hden (by positivity)] at hk
  rw [smul_dotProduct, smul_eq_mul]
  field_simp [h0 q]
  linarith

end EP
