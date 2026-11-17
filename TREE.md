# The subset tree: what is a special case of what

Read `A ─▶ B` as **"A is a special case of B"** = B can be parameterized to
reproduce A exactly (B is higher/more expressive). Edge label = the **currency**
you spend to actually run the emulation:

```
  free      weight-tie / constant / zero a slot        (no cost)
  fold      one-time offline weight transform          (free at inference)
  mag       activation magnitude -> inf                (breaks norm / quant)
  kv        KV-cache bytes per token
  width     hidden units / params
  depth     extra layers
  cap       capacity condition (emulator must be wide enough)
  ∞         exact only in a limit  (the closest thing to a real gap)
```

A whole model is a tuple (positional, connectivity, kv-sharing, score-fn, ffn,
activation, norm). The model-level order is the **product** of the per-slot
orders below: model M1 ⊆ M2 iff every slot of M1 is ⊆ the matching slot of M2.
The **universal parent** is the transformer that takes the top option in every
column; every real model is a point below it.

---

## 1. Positional encoding  — two maximal families, ALiBi sits under both

```
   additive relative bias                     RoPE (rotary)
   (learnable per-distance)                  /      \
        │                                   /        \
    ┌───┴────┐                          ALiBi        NoPE
  T5-bias   ALiBi ──────mag────────────▶ (same        (θ=0)
            │        (small-angle          node)         │
            │         fixed pair)                        │
          NoPE ◀───────free───────────────────────── (θ→0 / bias→0)
```
- `ALiBi ─free▶ additive-bias`  (ALiBi = the fixed linear-ramp special case)
- `ALiBi ─mag▶ RoPE`            (the seed result: small-angle fixed pair)  ← **experiments/bloom**
- `NoPE  ─free▶ RoPE`           (θ=0 identity rotation)

## 2. Connectivity — who may attend to whom

```
     full (dense) attention
        │       ╲
     mask│        ╲ mask
        │          ╲
   local/window   causal
        ▲
        │ depth  (reverse: ⌈log_window N⌉ layers to propagate across windows)
```
- `local ─free▶ full`, `causal ─free▶ bidirectional`  (both are fixed masks)
- reverse (full from local) costs **depth**.

**Every sparse pattern is a mask ⊆ full** (strided, dilated, block-sparse,
global+local, attention-sinks — all fixed masks, free to restrict). The reverse —
recovering full reach — costs **depth = the pattern's graph diameter**
(`attn/sparse_patterns.py`): `local+strided` factorizes to **2 hops** (Sparse
Transformer), naive `local`/`block` need **O(N/W)**, and **strided *alone* never
reaches full** (residue-class gaps — it needs a distance-1 edge). Subtlety: in
*causal* decoders, **global tokens don't bridge forward**, so global/sink patterns
stay local-limited — BigBird's O(1) diameter is a bidirectional-only property.

## 3. KV sharing — how many / how compressed the K,V heads are  (`attn/`)

```
        MHA        full per-head K,V
         ▲
      kv │
        MLA        shared low-rank latent           attn/mla.py   (~8.9x)
         ▲
      kv │         (MLA generalizes GQA — TransMLA)
        GQA        grouped K,V
         ▲
      kv │
        MQA        single K,V head                  attn/mha_mqa_gqa.py
```
Every edge is an **exact** restriction downward; upward you pay back exactly the
KV memory the child saved (conserved — MQA→MHA widens the shared head to `H·d`).

**Size matters here more than anywhere else.** The chain above is the *equal-size*
view. Once the KV dimension can vary, **every** variant emulates every other given
enough capacity: the criterion is numeric — `kv_cap(B) ≥ kv_cap(A)`, where
`kv_cap = n_kv·head_dim` (MHA/GQA/MQA) or `kv_lora_rank + rope` (MLA). MLA/MQA
*shrink* `kv_cap` by design, so a 1T-param MLA model emulates **fewer** attention
configs than a wide-KV MHA one. The graph's instance-level analysis uses `kv_cap`,
not the label (`graph/model_graph.py --analyze`).

**Cross-layer KV sharing (CLA)** is the one KV edge the headline does not pay: a non-sharing host
reproduces a sharing target by carrying the shared K, V, which is exact in the norm-free toy
(`attn/cross_layer_kv.py`) but under a stock pre-norm the shared rms rescales the carried copy
(~6e-2 on stock Llama layers, `block/prenorm_carry.py`) — a floor two orders above
`ACT_FLOOR_MAX`, so `CLA_CARRY_PAYABLE = False` (headline frontier 101 → 106).

## 4. Score function — the one near-gap  (`attn/softmax_linear.py`)

```
     softmax attention
         ▲
       ∞ │   exact only as #features → ∞  (Performer et al.)
     linear / kernel attention
```
The only edge with no finite-resource exact form. Three currencies close it:
**width** (random features, err ~ 1/√m), **magnitude** (softmax→one-hot as the
score scale β→∞ = exact hard retrieval; linear attention has no temperature and
floors — same shape as ALiBi→RoPE), and **depth** (1 softmax layer ≈ 2 layers of
linear/SSM for associative-recall / induction).

**The sequence-mixer axis.** `softmax-attn ⊇ {linear-attn, SSM}` — state-space
models (Mamba, RWKV) are the recurrent/linear family, connected by the same
near-gap (SSM ⊆ softmax at ∞ features / depth) — a type order only: measured on real
checkpoints the softmax constructions that keep generation identical cost per-channel heads,
extra depth and a norm change, and the delta rule depth ~ context, so the headline does not
pay it (`attn/mixer_gap_agreement.py`, VALIDATION row 48). They have no `kv_cap` (a fixed
state, not a growing cache), so they sit in a *separate branch* the capacity graph
doesn't place; **hybrids** (Jamba, Zamba, Nemotron-H) interleave both. Survey of
the 4,384 fetched configs: **76.2% pure attention, ~4.7% SSM/hybrid** (13 SSM — Mamba,
Falcon-Mamba, RWKV; 192 hybrid — Nemotron-H, LFM2 / LFM3, Granite-MoE-hybrid, Falcon-H1, ...), the rest
non-LM or skipped. (`graph/mixer_survey.py`, output in `graph/mixer_survey.txt`)

**Local attention and SSM are DUAL restrictions of full attention** — they
compress orthogonal axes (`attn/local_vs_ssm.py`, verified on associative recall):

```
                     reach          resolution      cost
  full softmax       unbounded      full            O(N)      (the general case)
  local / window     BOUNDED (W)    full            O(W)      trades REACH
  SSM / linear       unbounded      BOUNDED (state) O(1)      trades RESOLUTION
```
Local is exact inside its window and useless beyond it; SSM/linear degrades as the
number of stored items exceeds its state capacity (interference). Both are `⊆ full`,
but **incomparable to each other** — local can't reach far, SSM can't retrieve
sharply. That's precisely why hybrids pair them (cheap bulk) with a few full layers
(the exact induction/copying neither can do) — a `reach × resolution × softmax`
decomposition of the sequence mixer.

## 5. FFN & MoE  (`moe/`, `act/`)

```
                       MoE  (conditional, E experts)
                      ╱   ╲
                 free╱     ╲ free
                    ╱       ╲
        shared-expert       plain top-k MoE
             MoE                  │
                              cap │  route→1 expert  (needs k·expert ≥ dense)
                                  ▼
                             dense FFN            ── reverse: E× params / k× FLOPs
                                  ▲
                          free    │  pin value branch = 1
                                  │
                             GLU family (SwiGLU / GeGLU / bilinear)
                                  ▲
                    width 2×  ╱   │  depth 2×
                             ╱    │  (SwiGLU: inner act forces a 2nd layer)
              bilinear-GLU ─▶     │
                                  ▼
                      squared-activation FFN   (builds products via a·b=¼[(a+b)²−(a−b)²])
```
- `vanilla ─free▶ GLU`,  `dense ─cap▶ MoE`,  `plain-MoE ─free▶ shared-expert-MoE`
- `bilinear-GLU ─width▶ squared-FFN`,  `SwiGLU ─depth▶ 2-layer squared-FFN`
- also: fine-grained (many small) experts ─free▶ few big experts (co-routing is a reparam)

**Both directions, size-based** (like KV sharing). The diagram is the equal-size view;
with size free, `dense ↔ MoE` is mutual:
- **MoE ▶ dense**: route to enough active experts — needs `ffn_active = k·expert_width ≥ dense_width`.
- **dense ▶ MoE**: hold *all* experts — needs `ffn_total = dense_width ≥ E·expert_width`
  **and ~2× depth** for the routing/argmax/gating circuit ("multiple layers + larger size").
The graph checks both `ffn_total` and `ffn_active` (`graph/model_graph.py --analyze`); the
depth surcharge is why big *dense* models rarely emulate big MoEs, but a wide-KV dense giant
still tops the frontier for the (dense-majority) zoo.

## 6. Activation  (`act/relu_swiglu.py`, `act/activations.py`)

```
              Maxout_k = max_i(w_i x + b_i)     universal piecewise-linear
                    ▲  free (k=2)
        ┌───────────┼──────────┬──────────┐
      ReLU     LeakyReLU_α     |x|      identity      ── all exact Maxout_2
        ▲             ▲
    mag │ β→∞     free │ α=0                         ReLU-network (wide)
        │              │                                   ▲  width
   Swish_β / Softplus_β / GeLU_β (smooth, β)          GeLU / SiLU / tanh (smooth)
        │
   GeLU ≈ Swish_β  (β≈1.7, cross-family approx)
```
- `ReLU ─mag▶ Swish_β` (and Softplus_β, GeLU_β): a hard op as the β→∞ limit of a
  smooth gate — **same shape as ALiBi ─mag▶ RoPE**. The limit is family-independent.
- `ReLU / LeakyReLU / |x| / identity ─free▶ Maxout` — exact, currency = pieces.
- `ReLU ─free▶ LeakyReLU_0 = ELU_0 = PReLU_0` (parameter restriction).
- `tanh ─fold▶ sigmoid`:  `tanh(x)=2σ(2x)−1`, exact affine reparam.
- `GeLU ─≈▶ Swish_β` at β≈1.7 (cross-family), and `smooth ─width▶ ReLU-net`.

## 7. Normalization  (`norm/`)

```
        general affine-norm:   γ · (A x)/‖A x‖ + β
              ╱                              ╲
        A = C (centering)                  A = I
              │                              │
         LayerNorm                        RMSNorm
```
- `β`: **free** fold into the next linear layer's bias.
- `LayerNorm ─fold▶ RMSNorm` for a whole pre-LN model: centre every residual writer
  (W ↦ PW, embeddings E ↦ EP), so the residual stays mean-zero and LN = RMS + β exactly;
  β needs a bias in every consumer (incl. an lm_head bias), and tied embeddings must be
  untied (`fold_layernorm_model`, exact on Pythia-14m / OPT-125m in fp64).
- Dropping centering per call WITHOUT that writer fold is the measured penalty
  (`norm/layernorm_rmsnorm.py`, `rms_keepbias`).

---

## 8. Small config items (the long tail)

Most architectural "knobs" are one of three kinds of edge. Detectable from config;
`block/parallel_residual.py` and `convert/` implement the starred ones.

**Free / foldable (exact, ~zero cost)** — a constant, a folded scalar, or a dropped constraint:
- attention-bias / MLP-bias / LN-bias `─fold▶` no-bias (fold into next linear) *(convert/)*
- tied `─free▶` untied embeddings (tied is a constraint on untied)
- embedding-scale (Gemma ×√d), attention-scale, router `routed_scaling_factor` — foldable scalars
- fine-grained experts `─free▶` coarse experts (co-routing is a reparam)
- RoPE base θ / scaling (linear·dynamic·yarn·llama3) — all within the RoPE family (a θ choice)

**Structural (depth / width cost)**:
- **parallel is a restriction of sequential** (drops the attn→MLP dependency for
  concurrency). Both simulate each other: `sequential ─width 2×▶ parallel` (same
  depth) and `parallel ─depth 2×▶ sequential`. Sequential is more general — it
  hosts parallel at 1:1 layers. *(block/parallel_residual.py — both exact in the norm-free toy;
  under a stock pre-norm only parallel ─depth 2×▶ sequential stays exact, see block/prenorm_carry.py)*
- local / sliding-window `─free▶` full attention (mask); reverse costs depth ⌈log_w N⌉
- vanilla `─depth▶` GLU, GLU `─depth▶` squared-FFN, dense `─depth▶` MoE (§5)

**Distinct nonlinearity (needs the matching op — approximate without it)**:
- QK-norm (Gemma-2·Qwen3·OLMo-2): an RMSNorm on Q,K inside attention — not identity-representable
- attention / final **logit soft-capping** (Gemma-2): a tanh cap on scores/logits
- partial-RoPE (`rotary_pct`<1: NeoX·Phi·StableLM) sits *between* full-RoPE and NoPE (a partial host that rotates ≥ the target's rotated dims per head reproduces full RoPE exactly: the **rot** edge)
- post-norm vs pre-norm vs sandwich (Gemma-2) placement — different function + gradient path
- learned-absolute vs relative PE — the genuine incomparability (§1)

Currency reuse: the "distinct nonlinearity" items are the same shape as the GeLU→SiLU
floor — a fixed op another architecture can only approximate, with no resource that
drives the gap to zero.

## Master edge table (revised)

`A ─▶ B` = A is a special case of B (B emulates A). Many axes are **size-based**:
A ↔ B are mutually emulable given enough of the currency (conserved either way).

| child ─▶ parent | exact? | currency | where |
|---|---|---|---|
| **Positional** | | | |
| NoPE ─▶ partial-RoPE ─▶ RoPE | exact | free (θ=0 / subspace) | `classify` |
| RoPE ─▶ partial-RoPE | exact | **rot**: host rotated dims per head ≥ the target's (every RoPE-family host: several tables / RoPE + NoPE layers need room in the host's tables, `rope_fits`); config-only (round 2, pass 5): a frequency per pair only on a longrope-capable rotary, a fixed one sets θ (and, fraction-free, its rotary fraction) only | `hf/partial_rope_host.py` |
| ALiBi ─▶ RoPE | ≈ small-angle | **mag** N; causal targets, a spare frequency-0 pair | bloom (seed) |
| ALiBi ─▶ additive-bias | exact | free | — |
| ALiBi ⟂ learned-abs | — | wall: the ramp channel is read through every pre-norm (round 2, pass 5) | `proofs/PROOFS.md` "Pre-norm rescue attempts" |
| learned-abs ─▶ RoPE / NoPE | exact | the **start-token rule**: +2 full layers, +5 hidden (+10 LayerNorm), T + 1 + erasers FFN units; only a host that can zero its frequencies | `hf/bos_constructions.py` |
| **Normalization** | | | |
| LayerNorm ─▶ RMSNorm | exact | **fold** (global reparam) | `norm/`, `convert/layernorm_to_rmsnorm` |
| **Attention — KV (size-based)** | | | |
| MQA ↔ GQA ↔ MLA ↔ MHA | exact | **kv_cap** (conserved) | `attn/`, `convert/mha_to_gqa` |
| — tied-KV ─▶ MLA target | exact | **2× kv_cap** (TransMLA) | `attn/mla` |
| **Attention — score (near-gap)** | | | |
| linear ─▶ softmax | limit | **∞** width / mag / depth-2× | `attn/softmax_linear` |
| **Attention — window** | | | |
| local ─▶ full | exact | free mask (rev: depth ⌈log_w N⌉) | `classify` |
| full ─▶ local (window in config) | exact | free: the host sets its window ≥ the context (`WINDOW_FREE_TYPES`) | `verify/levers/` |
| sink ─▶ plain (start token) | exact | a q bias, +2 / +1 head dims (a zero-frequency slot), +1 hidden | `hf/bos_constructions.py` |
| plain ─▶ sink | limit | b → −∞ (Lean `sink_host_factor`) | `attn/attention_sink` |
| **Attention — QK-norm / softcap (softened)** | | | |
| qk-norm ─▶ plain | ≈ | **fold** (penalty = CV‖q‖): measured not payable | `attn/qk_norm` |
| plain ─▶ qk-norm / sandwich / v-norm / MLA host | ≈ exact (limit) | the **eps lever**: the norm's input scaled to its linear regime (gained RMS variants, eps > 0) | `hf/eps_lever_agreement.py` |
| softcap ─▶ plain | ≈ | **fold** (penalty = logit/cap) | `attn/softcap` |
| **FFN / MoE (size-based)** | | | |
| vanilla ─▶ GLU | exact | free (rev: depth) | `act/relu_swiglu` |
| dense ↔ MoE | exact | **ffn_active / ffn_total + 2× depth** | `moe/`, graph |
| plain-MoE ─▶ shared-expert-MoE | exact | free | — |
| bilinear-GLU ─▶ squared-FFN | exact | **width 2×** | `act/` |
| SwiGLU ─▶ 2-layer squared-FFN | exact | **depth 2×** | `act/` |
| **Activation** | | | |
| ReLU ─▶ Swish_β / Softplus_β / GeLU_β | limit | **mag** β | `act/activations` |
| ReLU / LeakyReLU / \|x\| / id ─▶ Maxout | exact | free (pieces) | `act/` |
| tanh ─▶ sigmoid | exact | **fold** (affine) | `act/` |
| GeLU ─▶ SiLU | ≈ **floor 0.0139** | one unit: none; **FFN width ×k** (k = 5, measured; SiLU ◀─ GeLU ×3) | `act/`, `graph/act_width.py` |
| GeLU (erf) ↔ GeLU (tanh) | ≈ floor 3e-4 | free (measured payable both ways) | `act/gelu_swap_agreement.py` |
| smooth-act ─▶ ReLU-net | ≈ | **width** | `act/` |
| **Block structure** (size-based; sequential is more general) | | | |
| parallel ─▶ sequential | exact norm-free only (stock pre-norm: not at any width) | **width 2×** per block, (L+1)× for L blocks; ≈ by a 1-D RMS compensator (k units per carried channel) for measured families only | `block/parallel_residual`, `block/prenorm_carry` |
| sequential ─▶ parallel | exact | **depth 2×** (reverse direction) | `block/parallel_residual` |
| **Weights (size-based)** | | | |
| N:M-sparse ↔ dense | exact | **M/N× width** (FLOPs/mem, conserved) | `convert/nm_sparsity` |
| low-rank / tied / quant ─▶ full | exact | free / rounding | — |
| **Release constructions** (the first-wave rules on top of round 3) | | | |
| pure SSM stack ─▶ hybrid | exact | the attention layers zeroed (`o_proj`, `down_proj` = 0); target layer i on the host's i-th recurrent layer (Lean `FrMixer.padded_run` / `assign_run`) | `hf/mixer_rules_agreement.py` |
| norm-free Mamba-2 ─▶ gated-norm Mamba-2 | limit | residual stream × S with the config eps ε S² (the gated norm linear) | `hf/mixer_rules_agreement.py` |
| plain Mamba-1 ─▶ Falcon-Mamba / Jamba | limit | the weights-only eps lever on the dt / B / C norms | `hf/mixer_rules_agreement.py` |
| plain stream ─▶ hyper-connection host (dsv4, glm5_next, hy4, qwen4_exp) | exact | identical streams: the dynamic projections 0, the static bases summing to 1 | `hf/hc_host_off.py` |
| plain ─▶ DiffusionGemma (always-on embedding norm) | limit | stream × α, eps α² ε (payable) | `hf/selfcond_norm_agreement.py` |
| LayerNorm-β target ─▶ RMSNorm host, no lm_head bias | limit | a dominated final readout: +1 layer, +1 hidden, d (gated) / 2d FFN units | `hf/ln_beta_readout.py` |
| RMSNorm ─▶ LayerNorm (no gain written into the residual, untied host) | exact | **hidden + 1** (the mean-zero isometry), not 2× | `hf/norm_subspace.py` |
| GeLU ─▶ SiLU | ≈ | FFN width ×5 (measured on six models, `cost_act_k_measured`) | `graph/act_width.py` |
| RoPE target ─▶ RoPE + NoPE-layer host | exact | the NoPE layers zeroed; the target's layers on the host's RoPE layers, in order | `hf/fr_positions.py` |
| sinusoid table ─▶ learned-abs | exact | the table written, the start-token rule's rows | `hf/fr_positions.py` |
| sinusoid ─▶ sinusoid, learned ─▶ learned | exact | the same table only / a host table with the target's rows (soundness, transitivity) | — |
| unrenormalised softmax MoE ─▶ host with extra experts | limit | the extras sunk by a free hidden dim (payable) | `moe/router_frontier.py` |

**Three structural facts:**
1. Most axes are **size-based** — `kv_cap`, `ffn_total/active`, N:M width — so A and B
   mutually emulate given the resource; the "penalty" is a conserved currency, not a gap.
2. Two **magnitude-limit** edges (ALiBi→RoPE, ReLU→Swish) and two **fold** approximations
   (QK-norm, softcap) close to exact as a resource → limit.
3. Only **two genuine boundaries**: `learned-abs ⟂ relative` PE (incomparable), and the
   `linear→softmax` **near-gap** (exact only at ∞ features). Everything else is a trade.
```
