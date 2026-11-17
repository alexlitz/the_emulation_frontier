# the_emulation_frontier

**Modern LLM architectures are restricted subsets of more-general ones.** Almost
every "new" decoder architecture is an existing one with some axis *narrowed* to
save a resource — GQA ties KV heads, MQA ties them harder, MoE gates the FFN, ALiBi
replaces RoPE, RMSNorm drops LayerNorm's centering, sliding-window masks full
attention. Because these are *restrictions*, they can be undone: the more-general
architecture **natively emulates** the special case by setting weights to 0/1 and
choosing its shape — no retraining, no per-token translation, just the host's own
native operators.

This repo makes that claim concrete and *validated*. It classifies real Hugging
Face configs, builds the "A is emulated by B" partial order over them, and proves
each edge numerically — many against the *stock, unmodified* `transformers` module.

The seed result (a [previous post](https://alexlitzenberger.com/blog/emulating_alibi_with_rope))
is **ALiBi ⊆ RoPE**: a fixed RoPE pair in the small-angle limit *is* ALiBi. This
repo scales that idea to the whole zoo. See **[the blog post](https://alexlitzenberger.com/blog/the_emulation_frontier)** for the
writeup, the interactive graph at
[alexlitzenberger.com/blog/the_emulation_frontier/zoo.html](https://alexlitzenberger.com/blog/the_emulation_frontier/zoo.html),
and the demo of the universal host at
[hf.co/alexlitz/elephantine](https://huggingface.co/alexlitz/elephantine): one architecture, with
subfolders that reproduce SmolLM2, Pythia, Falcon-H1, a Mamba and a Qwen3 diffusion model, plus an
inflated SmolLM2 (built by `elephantine/push_hub_demo.py`).

## The pipeline

The top **10,000** most-downloaded text-generation checkpoints on the Hub →

```
4,384 unique configs  →  3,906 classified decoder LMs  →  942 arch+size shapes  →  293-model emulation frontier
```

Deduping finetunes/quantizations gives 4,384 distinct configs; 3,906 classify as
decoder LMs; deduping identical arch+size shapes leaves 942 distinct shapes; and the
**headline relation** — capacity dominance plus every architectural *type* wall that has a
currency paid (an exact edge, a floor / regime, or a limit; a wall with no currency
blocks) — leaves **293 real checkpoints** (279 autoregressive + 14 diffusion / block-causal / prefix-LM) that, sized and
weighted appropriately, emulate all 3,906. *If every type wall were free* (pure capacity + the
one-way AR ⊆ diffusion wall), the frontier would be **49** (42 + 7). (The cached configs ship
as `graph/hub_10k.json`; the snapshot is 2026-11-10.)

## The currency

Once shape is free, there is essentially never a true *function* gap between these
variants — each can emulate the other. The content of the theory is the **exchange
rate**: which resource you spend, and whether spending it destroys the property that
made the special case worth using.

| currency | meaning |
|---|---|
| **magnitude** | activation scale needed (breaks normalization / quantization) |
| **KV memory** | bytes of cache per token |
| **params / width** | how wide the emulator must be |
| **depth** | extra layers |
| **a fold pass** | a one-time offline weight transform (free at inference) |

The interesting cases are where the currency *destroys the point*: emulating MQA
with MHA gives back all the KV memory MQA saved; emulating an MoE densely costs the
`E×` params conditional-compute was avoiding. A few edges are genuine boundaries,
validated *as* boundaries: **softmax → linear attention** is a true `∞` gap, and
**GeLU ≈ SiLU** has a `0.0139` floor (never exact). And some walls have **no currency at
all** — absolute (learned) positions on a relative host, an activation with no exact fold or limit
to the host's (GeLU / mish / xIELU / squared-ReLU / situ / poly_norm on a SiLU host: every
single-op floor is above the calibrated `ACT_FLOOR_MAX`), a diffusion target on a causal host — and
those block emulation in the headline (`FRONTIER.md` lists every one present in the snapshot).

**What an activation floor is worth** ([`act/act_swap_agreement.py`](act/act_swap_agreement.py),
data `act/data/act_swap_agreement.jsonl`, `python -m act.act_swap_report`): every FFN activation of a
real checkpoint (GPT-2 / -medium, Pythia-160M / 410M / 1.4B, OPT-125M / 350M / 1.3B, SmolLM2-135M,
Qwen2.5-0.5B / 1.5B, Llama-3.2-1B, Apertus-v1.1-0.5B (xIELU), nanochat-d20 (relu²), a gpt-oss
(swigluoai) TinyStories model, Gemma-2-2B) replaced by a host activation's best single-op fit
`a·X(c·x+d)+e·x+f`, scored on the original model's own greedy 64-token continuations. A swap is
**payable** iff generation stays in de facto high agreement: **top-1 ≥ 99% and ≥ 90% of 32-token
greedy continuations token-identical**. Exact folds and β→∞ limits pass at 100% / 100% on every
model; every floor ≥ 0.009 fails on every model — the GeLU / SiLU / mish cluster (0.009–0.018)
reaches 92–99% top-1 with 0–79% of continuations identical on most models (worse than running the
original in bf16 on the SiLU models) and collapses to 2–26% on Qwen2.5-1.5B, whose pre-activations
leave [-6, 6] where the fit extrapolates; SiLU ← ReLU (0.137) 14–74%. Floors predict the
verdict but not the size of the damage, which is model-specific (GeLU ← ReLU 0.084: 78% top-1 on
GPT-2 but 24% on the 2× deeper GPT-2-medium, 72–83% across Pythia-160M / 410M / 1.4B; SiLU ← ReLU
0.137: 74% on SmolLM2-135M, 44% on Llama-3.2-1B, 14% on Qwen2.5-1.5B). Scaled fits bracket the boundary at
0.0004 (pass) – 0.001 (fail), so `ACT_FLOOR_MAX = 0.0005`; the relation is the same for any
threshold below 0.009, and every looser definition tried lands there too ("no worse than the
original in bf16", "≥ 98% top-1 and ≥ 50% identical", even teacher-forced "top-1 ≥ 95%" alone).

**What the softmax ← SSM near-gap is worth** ([`attn/mixer_gap_agreement.py`](attn/mixer_gap_agreement.py),
data `attn/data/mixer_gap_agreement.jsonl`, `python -m attn.mixer_gap_report`): the same standard,
with every recurrent mixer of a real checkpoint replaced by an explicit softmax construction —
mamba-130m (Mamba-1), RWKV-4-169m, granite-4.0-h-350m (Mamba-2 hybrid), LFM2-350M (short-conv
hybrid), Qwen3.5-0.8B (gated-DeltaNet hybrid). Softmax *can* carry them: a zero-value sink key
undoes the normaliser, a diagonal decay is a logit (`exp(A(c_t − c_s))`, ALiBi for a fixed decay),
small temperature turns `exp(ε q·k)` into `q·k`, hard-selection heads do shifts and short
convolutions, GLU units do products exactly — and the complete constructions keep 96–100% of
continuations identical. But only at a price the old edge never charged: a data-dependent decay
needs a prior layer writing its prefix sum (one logit `q_t·k_s` cannot see the tokens between
`s` and `t`), and carrying that state under a stock pre-norm needs a norm change
(`block/prenorm_carry.py`); Mamba-1 needs a head per (channel, state) pair (26,116 heads per layer
for mamba-130m, plus 256 SiLU units per channel for its softplus — its `|A|` reaches ~1e6),
RWKV-4 one per channel (769), a one-stock-layer short conv 12,288 (the widest zoo host has 128);
and the delta rule's softmax form is exact only after `T − 1` Jacobi hops — depth proportional to
context (Qwen3.5: 32 hops 92% top-1 / 54% identical, 64 hops 98.9% / 100%). What the edge's
own price buys fails: one softmax layer per mixer with a fixed decay (85% top-1, 4–12%
identical), 16–4,096 shared decay rates on mamba-130m (57–95%, ≤ 25% — shrinking with heads, exact
only at 24,576), and a per-layer distilled softmax layer of the target's width or 4× it (2–16% top-1). So the near-gap is a wall in the headline
(`MIXER_GAP_PAYABLE = False`; frontier 85 → 101); `mode="mixgap"` keeps the old free edge.
The cross-layer KV (CLA) carry has the same problem: a non-sharing host recomputes the shared K, V
through its own stock pre-norm, which rescales the carried copy (~6e-2 on stock Llama layers,
`block/prenorm_carry.py`) — two orders above `ACT_FLOOR_MAX` — so it is not payable either
(`CLA_CARRY_PAYABLE = False`; frontier 101 → 106); `mode="mixgap"` grants it free too.
A full-RoPE target on a partial-RoPE host is an exact edge when the host rotates enough dims per
head (`hf/partial_rope_host.py`): each target pair goes onto a host pair (a q / k column permutation:
half-split or interleaved rotation), its frequencies into the host's `inv_freq` buffer, the softmax
scale and YaRN's mscale into `W_q`. The graph pays it where the target's RoPE schedules fit the host's
rotated dims (`rope_fits`, dims `rope_sched` / `rope_cap`); with `classify` reading the rotary factor
nested in `rope_parameters` too (Qwen3.5 / Gemma-4) and MLA without a RoPE key as NoPE, frontier
106 → 104. Below the rotated dims there is no construction: dropping the low-frequency pairs (SmolLM2-135M
at half its rotated dims) gives 99.3% top-1 but 83% identical 32-token continuations, and 41% top-1 at
2k context.
The edge-case audit then checked every edge's conditions against the real forward passes: every target
layer needs a host layer of a kind that reaches it (full / sliding attention, own V / K = V, KV-shared,
recurrent: `LAYER_KINDS`, Hall's condition), Gemma-4's K = V layers, MLA off the KV chain (its per-token
latent norm folded to a constant fails on DeepSeek-V2-Lite: top-1 81%, `attn/mla_norm_agreement.py`),
the measured qk-norm / post-norm / q-k-v-bias / sink walls (`block/fold_agreement.py`), the forward-pass
`FEATURES` no slot carried (tied embeddings, output gates, hyper-connections, PLE, routers, DSA, ...) and
the recurrent family of SSM layers, KV heads by count, and the Lean-checked conditions (a dense host plays
only a top-1 unit-gate MoE, a sequential host no parallel block). The rot gate now covers full-RoPE hosts
too: a one-table host cannot play a target with several RoPE tables (Gemma-3's local / global theta) or
RoPE + NoPE layers (SmolLM3, Beam) unless its table holds them all (`rope_fits`, the NoPE layers'
zero-frequency slots counted). Frontier 104 → 218; the relation stays reflexive and transitive in every
mode, with exact ⊆ payable ⊆ mixgap ⊆ capacity. Round 2 read the configs the way the code runs them and
priced attention / norms, MoE / MLA and recurrent capacity (218 → 271), then (pass 5) the user's config rules
(a host sets the values its family's code reads: sliding window, rotary fraction, softcap, norm flags; RoPE
frequencies only on a longrope-capable rotary), the eps lever, the start-token rules, activation width, GeLU
erf ↔ tanh, looped hosts and the pre-norm compensators, with the swigluoai / LayerNorm-beta / ALiBi-on-learned-abs
walls: 271 → 248.

Round 3 (2026-11-15) classified from a sweep of the families' code (keys and branches the configs
carry but p5 did not read), then priced the attention, MoE, recurrent and residual-stream rules it found; the
spot checks added soundness walls (no gated target on a vanilla host at 2× depth, an up bias for a vanilla target on a
gated host, no swigluoai activation-width fit, the compensator credits off); the router score function became a
feature switch (no host lever), which made the relation transitive on the real shapes; the readout rule lets an
untied LM host play score and last-token embedding heads (fixed rows at fixed positions; pooled / L2-normed
heads stay walls); and a NoPE target (or NoPE layers) on a RoPE host is a limit (linear rope scaling → ∞):
248 → 300.

The release adds a first wave of constructions on top, each behind its own key (`R3_OFF`): pure recurrent stacks
on hybrid hosts and the norm-free Mamba-2 lever; hyper-connection hosts and DiffusionGemma's embedding norm switched
off, and the LayerNorm-β readout; the RoPE + NoPE-layer host, sinusoid tables on a learned table, and the same-table /
table-length soundness fixes; RMSNorm on a LayerNorm host at hidden + 1 and the measured GeLU ← SiLU width ×5;
unrenormalised-softmax MoE targets with the host's extra experts sunk, and the BailingMoeV3 group code (the null-expert
rule ships off): 300 → 293, still transitive on the real shapes in every mode (`python -m verify.r3.relcheck`).

## The tools

### `graph/` — the classifier and the emulation graph
- **`model_graph.py`** — the core. `classify(config)` maps a config to ~15
  architectural slots (norm / positional / KV-sharing / FFN / activation / block /
  window / mixer / …); each slot carries a partial order (e.g. `MQA → GQA → MLA →
  MHA`, `LayerNorm → RMSNorm`, `looped → untied`). `emulates(B, A)` decides whether
  B can emulate A under the **headline relation** `emul_matrix(mode="payable")`: capacity
  dominance (`kv_cap`, `q_cap`, `ffn_total/active`, depth, hidden, with the TransMLA, MoE-depth,
  block and mirror-norm penalties) **plus** every differing slot crossed by an edge that has a
  currency — exact (folds, masks, AR ⊆ diffusion), floor / regime (the activation
  floors ≤ `ACT_FLOOR_MAX` = 0.0005 — in practice none, see above — qk-norm, softcap), or limit
  (ALiBi ← RoPE, ReLU ← swish, relu² ← xIELU); the softmax ← SSM / linear / hybrid near-gap is
  measured not payable (above, `MIXER_GAP`), and so is the cross-layer KV carry
  (`CLA_CARRY_PAYABLE`). A
  hybrid / sliding-window / KV-sharing host runs a plain softmax target by zeroing its other
  layers, so only its attention (full-window, own-KV) layers count as depth (`ZERO_EDGES`). A
  slot difference with no currency blocks. `mode="capacity"` is the secondary "if every type
  wall were free" (frontier 49); `mode="exact"` is the stricter slice (320) and `mode="mixgap"`
  grants the near-gap and the CLA carry free (265; `"nomix"` is a legacy alias of the headline). A
  RoPE-family target on a RoPE-family host is rot-gated: its rotating pairs (and its NoPE layers'
  zero-frequency slots) must fit the host's inv_freq tables (`rope_fits`).
  Builds the graph, does the transitive reduction, and renders SVG / interactive HTML.
- **`reduce.py`** — CLI: does B emulate A? Reports the differing slots, the currency
  each costs, the per-axis capacity gaps, and a verdict — in both directions.
- **`edge_plan.py`** — the same plan for one edge as data (slots with their reduction and currency,
  capacity per axis with the penalties, exact or limit / regime): the zoo page's edge view and
  `elephantine/emulate.py` both read it.
- **`solve.py`** — CLI: fit target A into host B. Reports the per-slot plan, the
  capacity overhead, the FFN width↔depth dual, and the *minimal upgrade* to B if it
  doesn't fit as-is.
- **`construct.py`** — CLI: the *exhaustive* constructive search. Finds a construction if
  A embeds into B by **any** route — mixing currencies across axes, per-layer width↔depth
  trades, and the reverse / near-gap / network routes that make "incomparable" slots
  payable — or names the one genuine wall (AR↔diffusion, or capacity B can't afford).
- **`currency_curve.py`** — the frontier as a function of the currency budget
  (320 exact → **293 payable, the headline** → 265 with the softmax ← SSM near-gap and the CLA carry granted free →
  49 if every type wall were free → 1 the elephantine).
- **`frontier.py`** — computes the emulator frontier and writes
  [FRONTIER.md](FRONTIER.md) (the 293 maximal models and *why* each is maximal —
  a capacity RECORD, a type WALL with no currency that stops a bigger model, a PARETO
  trade-off, or the one-way PARADIGM wall; plus the list of walls and the secondary
  capacity-only frontier). `--top N` restricts to the N most-downloaded decoder LMs.
- **`minimal_emulator.py`** — the least upper bound: one (huge) host architecture —
  attention with a causal or full mask, a general gated-delta SSM block, MoE, and a parallel-block
  branch — sized to the per-axis maximum over **every** model (all mixers, AR + diffusion; the
  test asserts it emulates each one under `mode="capacity"`, and under the headline every model
  whose activation it carries): h 16384 (Llama-3.1-405B / Hermes-4-405B / Nemotron-Ultra; parallel
  targets such as Command-R+ run on the parallel branch at their own width, tested on a stock
  Cohere model), L 192, kv 14336, 128 query heads, 2.76M FFN units per layer (530M total), ~26.5T
  params (~4.5T active), ~4.8× the largest real model on the same proxy (Kimi-K3.1-Pro, ~5.5T). It
  carries **3,900 of the 3,906 models**: every mixer family but five is tested against its real code
  (GKA: awslabs hybrid-model-factory's Triton kernels; PseudoBrain x2: their pb27-vllm plugin under
  vLLM 0.31 in an isolated venv; 3,900 of 3,906), and a model counts only if the host also has its
  activation (`ACT_STATUS`: SiLU / SiTU / relu² / xIELU / PolyNorm / GeLU / mish / ReLU / swigluoai
  branches tested — ReLU as OPT's ungated branch, swigluoai as GPT-OSS's clip + up-shift composition) and norm. GeLU (583 models: erf or tanh-approx,
  ungated or GeGLU) and mish (4) are exact UFFN branches the loaders select from the target's
  `hidden_act` — their floors to SiLU are above the calibrated `ACT_FLOOR_MAX`, so a fit would not
  pay — checked against real GPT-2, Pythia-70m and Gemma-3-270m (fp32 casts lifted ~1e-13 rel) and a
  stock mish Llama. Open: Cyrah (code in a gated repo) and five models with unclassified recurrences
  (Solar-Open-2 x2, Ming-v3-ASR, KLM, one more). The host
  itself is `elephantine/universal.py`. **Loaded and tested:** of the 3,900 carried, **3,531** are
  models a whole-model `from_*` loader accepts (model_type + config) and a test runs against that
  type's own code — 151 model_types, 2,093 of them with a released checkpoint of their type checked
  too (`python -m elephantine.loader_coverage`; the stock `transformers` 4.57 / 5.18 families in
  `tests/test_stock_loaders.py`; no test that runs remote code is credited: those families count through
  tests against labelled transcriptions in `tests/refs/`, and `--include-remote` gives the same 3,531);
  the other 369 are carried only by priced slot constructions (remote
  code that is not run here, speculative-decoding drafts, and a few exotic transformers-5
  families: Qwen4-Exp, DeepSeek-V4, Zamba, ...).
- **`fetch_configs.py`** — fetches the top-N configs from the Hub (config.json only,
  never weights).

### `hf/` — reductions proven on real, unmodified `transformers` modules
Instantiate stock classes and reproduce a special case forward-for-forward using
only a config change and/or a state_dict remap — no code edits: **`gqa_mha.py`**
(GQA/MQA ⊆ MHA, `0.0`), **`moe_dense.py`** (dense ⊆ Mixtral MoE), **`local_full.py`**
(sliding-window ⊆ full Qwen2, `0.0`), **`nope_rope.py`** (NoPE ⊆ RoPE), **`looped_untied.py`**
(looped ⊆ untied Llama stack, `0.0`). **`real_reductions.py`** goes further: it loads
*real trained checkpoints* (gpt2, OPT-125m, Pythia-70m, Qwen2.5-0.5B, BLOOM-560m,
tiny_starcoder, OLMo-2-1B, Mamba-130m, Gemma-2-2b, DeepSeek-V2-Lite, gpt-oss-20b),
patches or remaps one real layer, and re-verifies **18 reductions** on the model's own
forward — GQA/MQA→MHA `0.0`, NoPE, SwiGLU=2-layer-square `1.8e-15`, LayerNorm fold,
ReLU=β→∞ Swish, GeLU≠SiLU floor, ALiBi=linear, parallel↔sequential, softcap, 2:4,
qk-norm (OLMo-2), linear-recurrence recurrent≡parallel (Mamba), **MLA absorption on
real DeepSeek-V2 weights** (~9× cache), and **attention sink on real gpt-oss weights +
its learned sink logits**, plus the **MTP fold on MiMo-7B's real weight index** (the aux
branch is disjoint from the base path). Only cross-layer-KV lacks a clean real-checkpoint
forward check (Hymba reuses KV at runtime, not by weight omission).

### `attn/ act/ norm/ moe/ block/` — the fp64 identity constructions
Bespoke tensor-math derivations of each reduction (ALiBi/RoPE, MLA absorption,
SwiGLU = 2-layer squared FFN, ReLU = β→∞ Swish limit, attention sinks, retention,
sparse patterns, sequential = 2 parallel blocks, …), each matching to floating-point
noise. `convert/` holds the offline weight-conversion compilers
(GeLU→SiLU, LayerNorm→RMSNorm, MHA→GQA, N:M sparsity).

### `serving/` — emulation as a serving cost: LoRA-exactness and provider price spreads
- **`lora_screen.py`** — is a derived checkpoint exactly a LoRA of its base? Range-reads
  tensors from the Hub (no downloads). `quick` triages one mid-layer; `full` reads every
  language-model tensor (row slices bound a delta's rank from below). It flags any edit to a
  non-adaptable tensor (embeddings, norms, routers, linear-attention decay/conv parameters).
- **`lora_approx.py`** / **`lora_behaviour.py`** — how much of a fine-tune's behaviour a rank-r SVD
  LoRA recovers (KL on the fine-tune's own continuations); the GPU version also removes the rank-r
  part from the fine-tune. It shows that sub-ulp updates (below one bf16 step per weight) can be
  high-rank even when the weight-space screen sees rank 1–2.
- **`embed.py`** — the constructive side of "B emulates A" for serving: embeds a Llama-family
  checkpoint (llama / mistral / qwen2 / qwen3) into a bigger host checkpoint that the host's
  STOCK model class runs exactly: wider hidden (norm gain + eps rescale), more heads / regrouped
  KV, head_dim multiples (RoPE-interleaved), wider FFN, identity layers, padded vocab (a free hidden
  dim pushes padded logits to -inf), and MoE (qwen2_moe / qwen3_moe: more / wider experts whose
  dummies the free dim keeps at exactly zero routing mass, top-k multiples via tied expert copies,
  dense layers router-steered into MoE layers). `plan()` says whether the host can, and which config values
  it needs. Proven on real Qwen2.5-0.5B→3B and Llama-3.2-1B→3.1-8B shapes.
- **`convert.py`** — exact conversion of Llama-equivalent architectures into plain Llama / Mistral
  (Granite's four multipliers folded into the weights; Phi-3's fused projections split), so they can
  be served where only Llama-family classes exist, and then embedded.
- **`arbitrage.py`** — OpenRouter vs Featherless price spreads (per workload, cache-read aware): same-repo resale, and exact
  upload into a cheaper Featherless class via `emul_matrix(exact=True)`. The computed
  spreads ship as [`serving/data/arbitrage.json`](serving/data/arbitrage.json) (caveats: ToS, fp8, context caps).

## Validation

**2,969 tests** across 121 files (the 31 real-model checks skip without the local HF cache):

```bash
python -m pytest tests/test_graph_rules.py -q           # 109: slot orders + capacity rules + the join dominates all (both relations)
python -m pytest tests/test_payable_edges.py -q         # 24: the headline's currency edges: hybrid / sliding zeroing on stock Jamba / Qwen2, regimes both ways, activation floor cells + the threshold vs the recorded real-model swaps + a live gpt2 / OPT swap, AR <= diffusion
python -m pytest tests/test_constructions.py -q         # 29: fp64 numerical identities
python -m pytest tests/test_reduction_spot_checks.py -q # 20: every reduction on 1-2 layers of REAL weights
python -m pytest tests/test_fp32.py -q                  # 15: every exact edge at fp32
python -m pytest tests/test_real_reductions.py -q       # 12: 18 reductions on real TRAINED HF checkpoints
python -m pytest tests/test_hidden_width.py -q          #  7: hidden-width + 4-axis spine + composed type edges (pos/window/mtp)
python -m pytest tests/test_edge_coverage.py -q         #  4: edge coverage of the headline: 100% no near-gap (the mixer gap is a wall) + every exact edge built
python -m pytest tests/test_mixer_gap.py -q             # 13: softmax constructions of Mamba-1/2, RWKV-4, short conv, gated delta converge in their limits; the recorded real-model swaps; the headline refuses the near-gap
python -m pytest tests/test_composed_edges.py -q        #  5: norm/block/sink/CLA/act composed with capacity
python -m pytest tests/test_hf_modules.py -q            # 13: stock transformers modules
python -m pytest tests/test_full_model_reductions.py -q # 12: whole-model reductions build token-for-token
python -m pytest tests/test_mask_reach.py -q            # 10: local ⊆ full window/depth reach
python -m pytest tests/test_mla_rank.py -q              #  8: MLA ⊆ MHA low-rank absorption
python -m pytest tests/test_linear_softmax.py -q        #  8: softmax ↝ linear feature-map floor
python -m pytest tests/test_full_emulation.py -q        #  7: a whole LlamaForCausalLM emulated to logits
python -m pytest tests/test_positional_cross.py -q      #  7: ALiBi/RoPE positional cross-matrix
python -m pytest tests/test_partial_rope_host.py -q     # 19: a full-RoPE head in GLM / Phi / GPT-NeoX / Qwen3-Next partial-RoPE hosts (fp64 exact), a whole Llama in a GLM host, real SmolLM2-135M; the rot gate in classify / dims_of / emulates
python -m pytest tests/test_edge_soundness.py -q        #  8: headline 293 + secondary 49 + 265/320 slices; transitive; edges BUILD
python -m pytest tests/test_exotic_activations.py -q    #  6: swigluoai/mish reduce; situ/xIELU don't; xIELU == HF's XIELUActivation
python -m pytest tests/test_activation_cross.py -q      #  5: 8×8 activation floor matrix (grid + true minimax)
python -m pytest tests/test_cookbook_fixes.py -q        # 12: LN→RMS whole-model fold (Pythia/OPT), dense⊆MoE zeroed router (Mixtral/Qwen2-MoE/DeepSeek-V3), carrying under real pre-norm
python -m pytest tests/test_make_elephantine.py -q      # 10: Jamba host; carries HYBRID SSM; ONE universal host spans AR + diffusion
python -m pytest tests/test_ssm_cross.py -q             #  4: SSM families one-way order; gated-delta layer carries all (state inflation)
python -m pytest tests/test_gated_delta_layer.py -q     # 11: RUNNABLE gated-delta layer == transformers' kernel; wider host LM carries every family
python -m pytest tests/test_mamba1_gated_delta.py -q    #  1: Mamba-1 is a gated-delta recurrence (real MambaMixer)
python -m pytest tests/test_universal_elephantine.py -q  # 68: ONE host class vs every family's real code incl. a parallel-block Cohere (3,900 of 3,906 mixer families tested; 3,900 carried with their activation: softmax AR, GPT-2 / Pythia / Gemma (1, 3) on the exact GeLU branches (real gpt2, pythia-70m, gemma-3-270m), a mish Llama, diffusion, Mamba-1/2/3, gated-DeltaNet/Qwen3.5, KDA, RWKV-4/5/6/7, Hymba, Phi-4-flash, PLaMo-2, Mini-K3, Apertus (xIELU), Motif-3 (GDLA differential latent attention, PolyNorm MoE, mHC), AliceAI, Granite-switch, LFM2/3, CENO, GLM-5.3, Youtu, Jet-Nemotron, GPT-S3, GKA, PseudoBrain (vLLM plugin), EdgeRock, AllGDN-Engram, ...)
python -m pytest tests/test_stock_loaders.py -q         #122: the same host vs ~80 stock transformers architectures' own code (4.57 in-process, 5.18 in one subprocess): Llama / Mistral / Qwen2/3(-MoE) / Gemma-1/2/3/4 (softcaps, PLE, KV sharing) / Phi-1/3/MoE / Bloom (ALiBi) / OPT (pre + post-LN) / Falcon / GPT-J / CodeGen / BigCode / OLMo-1/2/3 / Granite(-MoE/-4-H/-SWA) / DeepSeek-V2/V3/V3.2 / GLM-5 (DSA) / GPT-OSS (sinks, swigluoai) / Mamba hybrids / Nemotron-H / Kimi-Linear / ... fp64-lifted ~1e-15 rel, each family branch ablated; + 15 released checkpoints (gemma-2-2b, bloom-560m, opt-125m/350m, phi-1_5, OLMo-2-1B, Qwen3-0.6B, ...)
python -m pytest tests/test_gated_delta_real.py -q      #  2: our rule reproduces the REAL Qwen3-Next gated-DeltaNet module's own recurrence
python -m pytest tests/test_diffusion_real.py -q        #  2: wider host emulates a REAL diffusion model's bidirectional forward (0.0)
python -m pytest tests/test_relu_net.py -q              #  5: ReLU-net error laws + exotic folds + free-knot minimax
python -m pytest tests/test_construct.py -q             #  4: construction search finds routes / walls
python -m pytest tests/test_rope_window.py -q           #  3: local mask via RoPE-Fourier
python -m pytest tests/test_currency_curve.py -q        #  2: frontier shrinks 320→293→265→49→1 with budget (293 = headline)
python -m pytest tests/test_zoo_page.py -q              #  5: the zoo page embeds every shape, the headline frontier and every frontier model with its cover and binding
python -m pytest tests/test_fungibility.py -q           #  5: width↔depth / head / rotation carry-cost builds
python -m pytest tests/test_track.py -q                 #  4: price-tracker snapshot diffs (new / gone / flipped / moved spreads) and retention
python -m pytest tests/test_convert.py -q               #  3: Granite / Phi-3 convert exactly into stock Llama / Mistral (then embed)
python -m pytest tests/test_embed.py -q                 # 27: target checkpoint runs exactly inside a bigger stock Llama/Qwen2/Qwen3/Qwen2-MoE/Qwen3-MoE host (incl. a real Qwen2.5-0.5B in 3B shapes; dummy experts get zero routing mass; fp8 serving error not amplified; fp16-safe constants)
python -m pytest tests/test_lora_screen.py -q           #  8: LoRA-exactness screen on synthetic checkpoints (merged adapter, dtype round-trip, frozen-tensor edits)
```

**Soundness & scope (what's proven vs. accounted).** Each reduction is validated on a
**three-rung ladder**, each rung strictly stronger than the last. (1) An **fp64 tensor
identity** on random tensors (`attn/ act/ norm/ …`) — the algebra. (2) A **synthetic
real-layer forward**: [`elephantine/spot_check_reductions.py`](elephantine/spot_check_reductions.py)
rebuilds every reduction (attention sink, cross-layer-KV carry, MLA absorption,
retention rotation, qk-norm/softcap folds, ALiBi=RoPE, SwiGLU=2-layer-square,
dense⊇MoE, LayerNorm fold, parallel↔sequential, 2:4 sparsity, MTP, and the five
stock-HF edges) as a genuine 1–2 layer module forward with real `nn.Linear` weights
and requires fp64 reproduction (or, for the GeLU/SiLU floor, provably *not*) — catches
wiring bugs an identity can't (23/23). (3) A **real trained checkpoint**:
[`hf/real_reductions.py`](hf/real_reductions.py) loads twelve real checkpoints (gpt2, OPT,
Pythia, Qwen2.5, BLOOM, tiny_starcoder, OLMo-2, Mamba, Gemma-2, DeepSeek-V2-Lite,
gpt-oss-20b, MiMo-7B), patches or remaps one real component, and checks the reduction on the
trained model's own forward — e.g. expanding Qwen2.5's tied KV heads back to MHA
reproduces its logits to `0.0`, MLA absorption holds on DeepSeek-V2's real weights
(`8.5e-14`, ~9× smaller cache), and the attention-sink identity holds on gpt-oss's real
learned sink logits (`5.3e-15`), and the MTP fold is confirmed on MiMo-7B's real weight
index — **18/18**; only cross-layer-KV lacks a clean real-checkpoint forward check
(Hymba reuses KV at runtime, not by weight omission) and stays on rungs 1–2. Whole-model *compositions* are then checked
at scale:
[`elephantine/verify_edges.py`](elephantine/verify_edges.py)
samples asserted `emulates(B, A)` edges, builds `B` from `A`'s weights, and requires
token-for-token reproduction. Its `run_spine()` now builds the **entire capacity spine**
— KV-sharing, FFN width, depth, **and hidden width** (the last via the
RMSNorm-compensated construction in
[`elephantine/hidden_width.py`](elephantine/hidden_width.py), exact to `1e-13` on a real
Llama-3.2-1B) — with non-edges (a narrower host) required to fail.

**Measured edge coverage** (not asserted — [`elephantine/edge_coverage.py`](elephantine/edge_coverage.py)
classifies all `38,134` headline edges over the real graph): **100% cross no near-gap**:
every reduction is an fp64 identity, a real-module/real-checkpoint proof, or a measured
floor / regime (32.0% pure-capacity → construction-verified end-to-end; 63.1% cross type walls
whose atoms are verified on real trained models, AR targets on diffusion hosts
through the causal mask among them; 5.0% use the cross-layer-KV/MTP atoms, real-model check pending). The
softmax ← SSM/linear mixer gap (9.9% of the edges while it was granted free) is now a wall, and
**0%** put a diffusion target on a causal host.

**Every *actually-emulatable* edge, built.** The `mode="exact"` relation is precisely the
set of edges where every reduction is exact (no floor / limit / regime / gap) —
**27,243 edges**, 71.4% of the headline set. `exact_edge_coverage()` measures how many we
can *build from A's weights and run token-for-token*: **94.6% are composed and tested
end-to-end today** (the capacity spine + kv→MHA + NoPE + dense→MoE via the elephantine +
band-mask window + MTP-fold + the causal mask + a hybrid host's zeroed SSM layers), and
**100% are buildable** — the remaining ~5% (sink / norm mirror / block / cross-layer-KV) reuse
*already-validated* exact atoms and need only composition, no new reduction and no gap. The approximate edges (ALiBi limit, qk-norm/softcap folds,
and the softmax↔linear gap) are *not* in this relation — they're emulatable only up to a
floor, which is the point.

`executable_coverage()` gives the softer per-axis view over *all* headline edges: **37.4%**
already composed (capacity spine + kv-type + positional); the clean-exact type axes (with a
hybrid host's zeroed SSM layers) raise the ceiling to **65.1%**; the in-regime folds
(norm/qknorm/softcap) reach **93.5%**; the last **~6.5%** needs the harder MLA/CLA/MTP
compositions, and none is blocked by the softmax↔linear gap, which the headline no longer
claims. So the honest statement is: the capacity spine is construction-verified, the type
walls are atom-verified on real models, a growing core is *composed* end-to-end, and the
mixer gap is a measured wall, not a claimed conversion. Where
a currency factor is *calibrated, not derived* (e.g. dense→MoE routing depth), acyclicity
is a **consequence of those factors, not a structural guarantee** (`test_edge_soundness.py`
pins this). "Incomparable" means "across the gap or over a capacity floor", not an
unexamined "no construction".

**Lean proofs ([`proofs/`](proofs/README.md)).** The constructions and walls the relation uses
are proved in Lean 4 + Mathlib over exact reals (namespace `EP`, 740 theorems and lemmas on the
standard axioms only); [`proofs/PROOFS.md`](proofs/PROOFS.md) indexes them per rule, and
`proofs/lean/check.sh` builds the library, audits it and runs a negative control.
[`verify/EDGE_JUSTIFICATION.md`](verify/EDGE_JUSTIFICATION.md) maps every allowed rule to its
theorem, and [`verify/BRIDGES.md`](verify/BRIDGES.md) ties theorems to the repo code that builds them.

**[VALIDATION.md](VALIDATION.md)** is the claim → test ledger: every reduction the
graph uses, mapped to how it is validated (exact / real-module / limit / floor / gap
/ regime / edge). See also **[TREE.md](TREE.md)** for the full priced graph and
**[RELATED_WORK.md](RELATED_WORK.md)** for prior art per axis.

## Approximate conversion (`approx/`)

When the target is *not* a superset, exact emulation is impossible, but the same machinery gives a
good **initialization**: `approx/` converts a teacher checkpoint into a different architecture so
the student starts as close as possible to the teacher, then distills. Worked pair:
**google/gemma-4-26B-A4B → the openai/gpt-oss-20b architecture** (`python -m graph.reduce` says
NO: kv/q capacity and depth FAIL; GeLU and softcap are approximate). The student keeps the
teacher's tokenizer (vocab_size 262,144, a config value, as are rope_theta / sliding_window).

`python -m approx.plan` lists every mismatch as an EXACT fold or an APPROX step:

| mismatch | Gemma-4-26B-A4B | gpt-oss-20b | step |
|---|---|---|---|
| hidden | 2816 | 2880 | EXACT: zero-pad; every RMSNorm weight × √(2816/2880) |
| embed scale, tied head, `layer_scalar` | ×√h, tied, per-layer scalar on the residual | none | EXACT folds (student residual = teacher residual / cumulative scalar) |
| attn scale, sinks, k = v (global) | 1.0 after q/k norms | 1/8, sinks | EXACT (fold into q; sink = −1e4; v_proj := k_proj) |
| q/k RMSNorm (weighted), v RMSNorm | per head | none | APPROX: LS-best *affine* map of the normalized input (biases as intercepts) |
| attention width | sliding 16×256 (kv 8×256), global 16×512 (kv 2×512, 25 % rope, 75 % NoPE) | 64×64, kv 8×64 | APPROX: per kv group keep 32 rope pairs (importance = attention-weighted score variance, Hungarian assignment to the student's frequencies); V = output-weighted PCA; 4 student q-heads per teacher head (spare copies = temperature ensemble); o_proj LS refit |
| depth / layer types | 30 (25 sliding w=1024 + 5 global) | 24 alternating | APPROX: drop 6 by ablation ΔKL, DP keeps teacher-global layers on student-full slots |
| FFN | 128×704 top-8 **+ dense 2112** | 32×2880 top-4 | APPROX: 4 experts per student expert by excess co-activation; dense units replicated in every expert (top-k weights sum to 1 ⇒ dense); units kept by importance |
| router | softmax-128, top-8 renorm, per-expert scale | linear+bias, top-4 softmax | input scale/per-expert scale EXACT; group logit = weighted LS onto log-sum-exp of member logits |
| activation | gelu_tanh(g)·u | (clamp(u)+1)·clamp(g)σ(1.702g) | up bias −1 and up range EXACT inside the clamp; per-unit gate scale fit |
| sandwich post-norms | post-attn, post-FFN ×3 | none | APPROX: per-channel LS gains, then refits; **dominant error** (see below) |
| final softcap 30 | tanh | none | APPROX: lm_head scale fit by KL |

**Method.** Sequential, layer-by-layer (GPTQ-style) on one GPU, streaming the 52 GB teacher. Each
student layer gets the closed form above, then calibration refits on the *student's own* input
stream against the *teacher's* residual stream (so each layer also absorbs the dropped layers'
updates and corrects upstream drift): LS for q/k/v, o_proj, router, per-expert down_proj and
per-channel FFN gain/bias; then an optional **gradient block reconstruction** of the FFN
(Adam on router, gate/up biases and scales, down_proj). The last quarter of the calibration
sequences is held out: every LS ridge is chosen on it (closed form, from held-out Gram matrices)
and the gradient refit early-stops on it. Each finished layer is rounded to bf16 before the
student stream moves on, so later layers fit the model that is actually stored.
**Domain importance** (`approx/importance.py`, a few thousand calibration tokens): per-layer
ablation ΔKL (all 30 variants advanced together so each layer is read once), grad×activation for
every q-head / expert / expert unit / dense unit (one reverse layer-by-layer pass), routing
frequency / mass / co-activation. The conversion spends capacity by them: which 6 layers to drop,
rope-pair and V budget per head, expert grouping and which units survive, dense-vs-expert unit
trade, optional dropping of rarely used experts.

**Tiny pair** (`approx/tiny.py`: same mismatch structure, teacher trained briefly on a two-domain
Markov language; `tests/test_approx.py`, 14 tests, CPU, ~3 min). Held-out KL(teacher‖student),
nats/token (teacher ppl 8.3):

| init | KL | top-1 |
|---|---|---|
| random (gpt-oss init) | 3.45 | 0.00 |
| naive: exact folds + truncation (first L_S layers, first rope pairs/dims, consecutive experts, no dense MLP) | 3.72 | 0.03 |
| converted, closed form + LS refits | 0.96 | 0.42 |
| converted, + gradient FFN reconstruction | **0.33** | **0.76** |

Leave-one-out on the last row: layer choice by ablation ΔKL (vs first-24) 1.17; dense-MLP trade (vs
none / 64 spare units only) 0.57 / 0.53; residual-stream targets (vs local per-layer targets) 0.49;
LSE router fit (vs mean) 0.39; grad×act importance (vs activation energy) 0.36; per-unit importance
pruning (vs first units) 0.36; LS q/k/v norm linearization (vs scalar) 0.35. Attention strategy
choices barely matter here (pairs/V/kv-split/temperature all 0.31–0.33), and plain *consecutive*
expert grouping beat co-activation grouping after refits (0.27 vs 0.33; dropping 16 rarely used
experts first: 0.26).

**Domain calibration on the tiny pair** (two domains routed to disjoint expert halves; then 60
distillation steps on the calibration domain):

| calibration | KL on A (in-domain) | KL on B (out-of-domain) | after distillation A / B |
|---|---|---|---|
| general (both domains) | 0.31 | 0.24 | 0.16 / 0.13 |
| general, drop 16 experts | 0.26 | 0.19 | 0.13 / 0.10 |
| domain A | **0.16** | 6.75 | **0.08** / 6.61 |
| domain A, drop 16 experts | **0.13** | 6.84 | **0.06** / 6.88 |

Domain calibration halves the in-domain KL and is catastrophic out of domain when the other domain
uses capacity the calibration never sees -- the trade-off is real, not a free lunch.

**Real pair** (`python -m approx.real ...`; calibration 32×512 tokens per domain, last quarter of
the sequences held out for every ridge / early-stopping choice; importance from 4×512 tokens;
held-out evaluation 32×512 tokens per domain; general = wikitext-103, code = Python sources of
transformers/torch/numpy/... for calibration and of scipy/sklearn/pandas/... held out).

Per-component, on real layers (relative error of the sub-block update on held-out sequences):

| component | variant | error |
|---|---|---|
| attention, sliding layer 0 | naive truncation (first rope pairs / dims) | 0.85 |
| | importance-selected rope pairs + output-weighted V-PCA (closed form) | 0.42 |
| | + LS o_proj refit | **0.32** |
| | + temperature ensemble / split KV / importance allocation | 0.32 (no change) |
| | scalar instead of LS q/k/v-norm linearization | 0.50 |
| attention, global layers 5 / 29 | naive / closed form / + LS o_proj | 0.84 / **0.76** / 1.04 ; 0.64 / **0.53** / 0.61 |
| attention, sliding layer 15 | naive / closed form / + LS o_proj | 0.97 / 1.00 / **0.75** |
| FFN, layer 0 | output-norm surrogate alone (f ≈ a·m1 + b·m2, per channel) | 0.55 |
| | full closed form / + LS refits | 0.71 / 0.63 |
| | no unit pruning (hypothetical 4928-wide student) | 0.63 |
| | + gradient block reconstruction (300 steps) | **0.33** (consecutive groups 0.33, drop 32 experts 0.32, dense MLP only in the 64 spare units 0.37) |

End to end, held-out (32×512 tokens per domain; KL(teacher‖student) in nats/token over the full
262k vocabulary; student run with fp32 activations, weights stored bf16):

| student init | data | KL | top-1 agree | ppl student | ppl teacher | bits/byte student | bits/byte teacher |
|---|---|---|---|---|---|---|---|
| (a) random (gpt-oss init) | general | 11.08 | 0.000 | 4.6e5 | 7.16 | 4.14 | 0.63 |
| (b) naive (exact folds + truncation, no calibration) | general | 36.05 | 0.000 | 3e16 | 7.16 | 12.07 | 0.63 |
| (b) naive | code | 36.53 | 0.000 | 2e16 | 2.51 | 17.02 | 0.42 |
| (c) converted, general-calibrated | general | **4.67** | **0.299** | 718 | 7.16 | **2.09** | 0.63 |
| (c) converted, general-calibrated | code | 8.77 | 0.048 | 1.6e4 | 2.51 | 4.41 | 0.42 |
| (c) converted, code-calibrated | code | **4.81** | **0.265** | 309 | 2.51 | **2.60** | 0.42 |
| (c) converted, code-calibrated | general | 7.64 | 0.040 | 1.6e4 | 7.16 | 3.07 | 0.63 |
| (d) real gpt-oss-20b, its own tokenizer (not comparable per token) | general / code | – | – | 5998 / 2292 | – | 2.72 / 3.07 | – |

(d) is context only: on raw 512-token windows gpt-oss-20b's perplexity rises with position (≈36
for tokens < 64, ≈48 for 64–128, ≈300 beyond 256 -- past its 128-token sliding window); my
forward matches stock transformers to 1.5e-5 in fp32 on these inputs, so this is the model, not
the harness. Per token, the converted students are ~4.7 nats from the teacher: far from usable as
is, but 2.4× closer than random and a different universe from naive copying (whose overconfident
truncated logits are worse than random).

**Domain calibration on the real pair**: calibrating on code instead of general text cuts the
in-domain (code) KL from 8.77 to 4.81 and costs 2.97 nats out of domain (general 4.67 → 7.64);
symmetric for the general calibration. The same direction as the tiny pair, and large: with 16k
calibration tokens, the refits specialize hard.

**What did not work / what it took** (each found on the real pair, not the tiny one):
* *Least squares without centering* produced q_proj biases up to 120 and o_proj weights up to 3,790
  (Gemma's near-constant "massive activation" residual dims are collinear with the intercept): the
  stored bf16 student then diverged (stream error 2-6 after layer 8). Fixed by centered LS with a
  per-feature + floor ridge, and by rounding each finished layer to bf16 before the stream moves on.
* *bf16 activations*: the overfit checkpoint diverged in bf16 (calibration data: KL 1.38 in fp32 vs
  15.9 in bf16, from layer ~10). After the held-out regularization the general student is fine in
  bf16 (held-out KL 4.65 vs 4.67 in fp32), but the code student still degrades (6.69 on 8 windows vs
  4.81 in fp32) -- the same layers that break under int8. Reported numbers use fp32 activations.
* *Cascaded overfitting*: with every refit fit on all 32 calibration sequences, each layer learned
  to cancel the previous layers' *in-sample* errors: calibration KL 1.05-1.4, held-out KL 15-18
  (worse than random). Fixed by holding out the last quarter of the calibration sequences and
  choosing every ridge (closed form, from held-out Gram matrices) and the gradient refit's early
  stopping on them; held-out KL went to 4.7.
* The LS o_proj refit still *hurts* global layers out of sample (layer 5: 0.76 → 1.04); the code
  now allows the closed form as a candidate (`cv_keep_prior`, not used for the numbers above).
* Co-activation grouping: higher routing-mass recall (0.86 vs 0.64) than consecutive groups but the
  same FFN error after the gradient refit; temperature ensembles / split KV: no measurable gain.
* The lm_head scale fit hit its grid edge (0.6) in the first runs (the student wants a lower logit
  temperature than the softcapped teacher); the grid is now [0.2, 1.4] (fits 0.44 / 0.48).
* Per-column int8 storage of the frozen experts in the distiller adds ~4 % FFN error per layer
  (KL 4.7 → 7.9 at step 0); groups of 32 inputs per scale: ~1 %.

**Distillation** (`python -m approx.real distill`; full-vocabulary KL(teacher‖student) from stored
teacher final hidden states; LoRA r=16 on q/k/v/o + full training of norms, sinks, all biases and
the router; the 24×32 experts frozen and streamed from pinned host memory as group-wise int8; fp32
activations; one A5000, ~8 s/step; 120 steps × 4 × 512 tokens = 246k tokens, ~18 min; each student
distilled on its calibration domain; held-out KL on 16 windows per domain):

| run | step 0 | 30 | 60 | 90 | 120 | top-1 at 120 | other domain, step 0 → 120 |
|---|---|---|---|---|---|---|---|
| converted (general-calibrated), on general | 4.59 | 3.52 | 3.45 | 3.29 | **3.23** | 0.316 | code 8.85 → 7.53 |
| naive init, on general | 35.72 | 26.32 | 22.63 | 19.39 | 15.92 | 0.030 | code 36.39 → 21.60 |
| converted (code-calibrated), on code | 11.75 | 6.89 | 4.37 | 3.87 | **3.76** | 0.289 | general 13.68 → 6.56 |
| converted (code-calibrated), on code, 20 fragile layers kept bf16 | 7.00 | 5.87 | 4.14 | 3.51 | **3.41** | 0.320 | general 8.12 → 6.36 |

The converted init starts where naive distillation is not after 120 steps (4.6 vs 15.9 nats) and
keeps improving. The code-calibrated student's step 0 is inflated by the distiller's int8 expert
copy: 20 of its 24 converted FFNs are numerically fragile (2-1000 % output error under
group-wise int8 on a probe sequence, vs 0.7-1.3 % on the two general-student layers checked) -- a
symptom of the refits relying on large canceling terms; distillation repairs most of it within 60
steps, and keeping those 20 layers bf16 in the distiller lowers the start to 7.00 (still above the
streamed evaluator's 4.81; part is the 16- vs 32-window eval set, the rest is not resolved). After
distillation the domain
trade-off persists: on code, 3.41 (code-calibrated, bf16-kept run) vs 7.53 (general-calibrated); on
general text, 3.23 vs 6.36.

**Limitations.** One calibration size (16k tokens/domain) and one distillation budget; the
student is still ~3-5 nats/token from the teacher (the code student also needs fp32 activations); the student's
sliding window is set to the teacher's 1024 (a config value), gpt-oss-20b's own is 128; layer
types are only matched up to context 1024; LoRA/bias/norm distillation only (the experts stay
frozen); per-layer gradient refit trains only the FFN; disk: 3 student checkpoints × 40 GB + 11 GB
of cached activations under `$TEF_SCRATCH/approx_convert`, plus the 49 GB / 13 GB model downloads.

## Reproduce

Requires Python ≥ 3.10 (tested on 3.12.6), PyTorch ≥ 2.2 and `transformers` ≥ 4.57 (tested on 4.57.6; the suite fails on 4.52). `pip install -r requirements.txt`.

```bash
pip install -r requirements.txt          # torch, transformers, huggingface_hub
                                         # + system graphviz (`dot`) for SVG

# does one model emulate another? (uses the cached graph/hub_10k.json)
python -m graph.reduce Mistral-7B-v0.1 Qwen1.5-72B --verify

# fit a target into a host, with the reduction plan + minimal upgrade
python -m graph.solve DeepSeek-V2 Qwen1.5-72B

# write the emulating weights: TARGET's weights in HOST's own architecture (only HOST's config is
# downloaded), reloaded with AutoModelForCausalLM and checked against TARGET in fp32 / fp64.
# Refuses (exit 2 / 3 / 4) a non-edge, a limit / regime edge, or an exact edge outside its scope (the
# families and constructions in elephantine/emulate.py's docstring).
python -m elephantine.emulate HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M --out smol-in-360m
python -m graph.edge_plan SmolLM2-135M SmolLM2-360M     # the construction of one edge, as text

# the class graph + capacity-aware frontier over the top-10K
python -m graph.model_graph --from-json graph/hub_10k.json --no-local --analyze
python -m graph.model_graph --from-json graph/hub_10k.json --no-local --html-models graph/zoo.html

# the full zoo reference page (cards, compare, variants, metrics, recipes): ~12 min; writes zoo.html
# and zoo_pairs.json, which must be deployed next to it (Compare loads it lazily)
python -m graph.zoo_ref --out dist/zoo.html

# recompute FRONTIER.md (headline + secondary); the top-N view; the minimal universal emulator spec
python -m graph.frontier
python -m graph.frontier --top 1000
python -m graph.minimal_emulator --from-json graph/hub_10k.json
```

## License

MIT; see [LICENSE](LICENSE).

## Terminology

The weight construction is **compilation, not interpretation**: a one-time offline
transform of parameters, after which the model runs in the host architecture's
*native* operators with zero per-token overhead — the ML analogue of native
(hardware-assisted) virtualization rather than binary translation. In
function-class terms the child is a **restriction / special case** of the parent;
the map is an **embedding**, and when it holds both ways a **(bi)simulation**.
