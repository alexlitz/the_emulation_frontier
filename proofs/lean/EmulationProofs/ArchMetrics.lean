import Mathlib

/-!
# Architecture metrics: the generic identities behind `graph/arch_metrics.py`

The zoo's metrics are closed forms in the context length `L`. These theorems prove the identities the formulas
rely on, for every `L` (the per-model polynomials are checked symbolically against the real code in
`graph/arch_trace.py`; these are the parts that hold for every model):

* **window cap** (`cacheTokens`): a sliding-window layer with window `W ≥ 1` caches `min L (W - 1)` tokens. That
  is `L` while `L ≤ W - 1` (inside the window), the constant `W - 1` beyond it, monotone in `L`, and never more
  than a global layer's `L` (`cacheTokens_le`, `cacheTokens_inside`, `cacheTokens_beyond`, `cacheTokens_mono`);
* **per-kind sum** (`kvTotal_eq_sum`): the total KV is the sum over the layer kinds (global, sliding, MLA latent,
  K = V, DSA indexer), and it is monotone in `L` (`kvTotal_mono`);
* **causal span** (`spanSum_closed`): `∑_{i=1}^{L} min i W` is `L (L + 1) / 2` for `L ≤ W` and
  `W (W + 1) / 2 + (L - W) W` beyond, the amortised causal prefill of a window layer; it is at most the dense
  `L²` (`spanSum_le_dense`), and a decode step's span `min L W` is at most `W` (flat beyond the window);
* **MoE** (`moe_active_le_total`): top-`k` of `E` experts plus always-on weights is at most the total when `k ≤ E`.
-/

namespace EP

/-- Tokens a window-`W` layer caches after `L` tokens (the transformers sliding layer keeps `W - 1`). -/
def cacheTokens (W L : ℕ) : ℕ := min L (W - 1)

theorem cacheTokens_le (W L : ℕ) : cacheTokens W L ≤ L := min_le_left _ _

theorem cacheTokens_le_window (W L : ℕ) : cacheTokens W L ≤ W - 1 := min_le_right _ _

theorem cacheTokens_inside {W L : ℕ} (h : L ≤ W - 1) : cacheTokens W L = L := min_eq_left h

theorem cacheTokens_beyond {W L : ℕ} (h : W - 1 ≤ L) : cacheTokens W L = W - 1 := min_eq_right h

theorem cacheTokens_mono (W : ℕ) : Monotone (cacheTokens W) := fun _ _ h => min_le_min_right _ h

/-- Flat beyond the window: one more token changes nothing. -/
theorem cacheTokens_flat {W L : ℕ} (h : W - 1 ≤ L) : cacheTokens W (L + 1) = cacheTokens W L := by
  rw [cacheTokens_beyond h, cacheTokens_beyond (le_trans h (Nat.le_succ L))]

/-- Per-token elements of each kind: global, sliding (window `W`), MLA latent, K = V, DSA indexer. -/
structure KVKinds where
  global : ℕ
  sliding : ℕ
  latent : ℕ
  keqv : ℕ
  indexer : ℕ
  W : ℕ

/-- Elements cached after `L` tokens, kind by kind. -/
def kvKinds (k : KVKinds) (L : ℕ) : List ℕ :=
  [k.global * L, k.sliding * cacheTokens k.W L, k.latent * L, k.keqv * L, k.indexer * L]

/-- The total the formulas report. -/
def kvTotal (k : KVKinds) (L : ℕ) : ℕ :=
  (k.global + k.latent + k.keqv + k.indexer) * L + k.sliding * cacheTokens k.W L

theorem kvTotal_eq_sum (k : KVKinds) (L : ℕ) : kvTotal k L = (kvKinds k L).sum := by
  simp [kvTotal, kvKinds]; ring

theorem kvTotal_mono (k : KVKinds) : Monotone (kvTotal k) := by
  intro a b h
  unfold kvTotal
  exact Nat.add_le_add (Nat.mul_le_mul_left _ h) (Nat.mul_le_mul_left _ (cacheTokens_mono _ h))

/-- A model with no sliding layers grows exactly linearly. -/
theorem kvTotal_linear (k : KVKinds) (h : k.sliding = 0) (L : ℕ) :
    kvTotal k L = (k.global + k.latent + k.keqv + k.indexer) * L := by
  simp [kvTotal, h]

/-- `∑_{i=1}^{L} min i W`: the keys a causal window-`W` prefill of `L` tokens scores. -/
def spanSum (W L : ℕ) : ℕ := ∑ i ∈ Finset.range L, min (i + 1) W

theorem spanSum_inside {W L : ℕ} (h : L ≤ W) : 2 * spanSum W L = L * (L + 1) := by
  induction L with
  | zero => simp [spanSum]
  | succ n ih =>
    have hn : n ≤ W := Nat.le_of_succ_le h
    rw [spanSum, Finset.sum_range_succ, ← spanSum, mul_add, ih hn, min_eq_left h]
    ring

theorem spanSum_beyond {W L : ℕ} (h : W ≤ L) : 2 * spanSum W L = W * (W + 1) + 2 * ((L - W) * W) := by
  induction L, h using Nat.le_induction with
  | base => rw [spanSum_inside le_rfl]; simp
  | succ n hn ih =>
    rw [spanSum, Finset.sum_range_succ, ← spanSum, mul_add, ih, min_eq_right (by omega)]
    have : n + 1 - W = (n - W) + 1 := by omega
    rw [this]; ring

/-- The closed form `graph/arch_metrics._span_causal_sum` evaluates. -/
theorem spanSum_closed (W L : ℕ) :
    2 * spanSum W L = if L ≤ W then L * (L + 1) else W * (W + 1) + 2 * ((L - W) * W) := by
  split_ifs with h
  · exact spanSum_inside h
  · exact spanSum_beyond (by omega)

theorem spanSum_le_dense (W L : ℕ) : spanSum W L ≤ L * L := by
  unfold spanSum
  calc ∑ i ∈ Finset.range L, min (i + 1) W ≤ ∑ _i ∈ Finset.range L, L := by
        apply Finset.sum_le_sum; intro i hi; simp at hi; exact le_trans (min_le_left _ _) hi
    _ = L * L := by simp

/-- A decode step attends `min L W` keys: never more than the window, flat beyond it. -/
theorem decodeSpan_le (W L : ℕ) : min L W ≤ W := min_le_right _ _

theorem decodeSpan_flat {W L : ℕ} (h : W ≤ L) : min (L + 1) W = min L W := by
  rw [min_eq_right h, min_eq_right (le_trans h (Nat.le_succ L))]

/-- MoE active parameters (top-`k` of `E` experts of `per` weights, plus `shared` always-on) never exceed the
total when `k ≤ E`. -/
theorem moe_active_le_total {k E per shared : ℕ} (h : k ≤ E) : k * per + shared ≤ E * per + shared :=
  Nat.add_le_add_right (Nat.mul_le_mul_right _ h) _


end EP
