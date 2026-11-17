# Correctness review of the core code (round2-p4)

Reviewed on `round2-p4` (3703d45). `round2-p5` has no commits beyond it at the time of writing.

Scope, in priority order:
- `graph/model_graph.py`
- `graph/frontier.py`, `graph/minimal_emulator.py`, `graph/edge_plan.py`
- `elephantine/emulate.py`
- `elephantine/universal.py`
- `graph/post_numbers.py`, `graph/check_post.py`

This is a review by reading and targeted testing. Nothing in the relation was changed.

Every confirmed finding has a reproduction in `tests/test_review_findings.py`. Each test asserts the correct behaviour and is marked `xfail(strict=True)` with the reason. All 13 cases xfail today, and each fails on its own assertion: I checked this with `--runxfail`. When a fix lands, its test XPASSes, and strict mode reports that as a failure until the marker is dropped.

## Whole-relation checks (all clean)

These checks found nothing. They are listed so that the next pass need not redo them:

| check | what | result |
|---|---|---|
| reflexive | `diag(emul_matrix(P, mode))` over the 909 points, for every mode (payable, exact, mixgap, capacity) | all True |
| transitive | `(ok @ ok) & ~ok` over the same points and modes | 0 violations |
| dedup soundness | each of the 3,901 models as its own point (no dedup), then every dedup group's members compared for identical rows and columns, in all four modes | 0 inconsistent groups: the key holds every dim the relation compares |
| `frontier.blockers` vs `emul_matrix` | every capacity-ok pair (163,584) | no failing pair with an empty blocker list; no passing pair with a blocker |
| numpy orientation | every `[:, None]` / `[None, :]` in `emul_matrix`, `kv_heads_fit`, `kind_match`, `_rec_ok` | target = rows, host = columns throughout |
| `from_*` returns | AST check of the 61 loaders in `universal.py` | every path ends in `return u` |

Two dims read by `emul_matrix` are missing from `_point_key_dims`: `mla_qswitch` and `mla_lat_free`. Only the universal host sets them (`minimal_emulator.host_point`), so real points always read 0 and the omission is harmless.

## Confirmed bugs, ranked by impact

### 1. post_numbers counts the block-causal frontier models as autoregressive

- **Where:** `graph/post_numbers.py:195` (`frontier_diffusion` counts `gen == "diffusion"` only) and `:407` (`frontier.ar = payable - nd`, labelled "gen == ar"). `model_graph.zoo_data` (`model_graph.py:4121`, `n_frontier_diff`) has the same split.
- **What's wrong:** 3 headline frontier models are SDAR, with `gen == "block"`. FRONTIER.md (`frontier.py` main, `gen != "ar"`) reports 262 AR + 9 diffusion. post_numbers would put **265 AR + 6 diffusion** into the post, so the post contradicts FRONTIER.md.
- **Repro:** `test_post_numbers_ar_split_matches_frontier_md`
- **Fix:** count `gen != "ar"`, as frontier.py does. Alternatively, report three classes: AR, block, diffusion.

### 2. Pure SSM / linear stacks count every layer as a full-attention layer

- **Where:** `graph/model_graph.py:1389`, in `_attn_counts`. A model with no per-layer list, no index keys and a type outside `HYBRID_TYPES` falls into "a softmax stack without a per-layer list" and gets `kinds = ["full"] * n`.
- **What's wrong:** this happens for `SSM_TYPES` / `LINEAR_TYPES` (`mamba`, `falcon_mamba`, `rwkv`, `rwkv5`, `rwkv7`, `gla`). Their dims then read `kFU = attn_eff = full_eff = L` and `kR = 0`: an attention-free model claims L full-attention layers and no recurrent layer. The docstring promises "0 for the kind it cannot prove".
- **Affected:** 14 snapshot models: mamba-130m / 370m / 790m / 1.4b / 2.8b (+ ArthurZ copy), the 4 Falcon-Mamba, RWKV-4-169m, RWKV-5 Eagle-7B, rwkv7-0.1B, gla-1.3B.
- **Repro:** `test_pure_ssm_layers_are_recurrent_not_full_attention` (synthetic) and `test_pure_ssm_snapshot_models_have_no_attention_layers` (real configs).
- **Impact:** recomputed with the counts moved to `kR` and `attn/full = 0`:
  - payable: 33,190 edges, frontier 271, both unchanged. A pure SSM host still has `kv_cap = 0`, and a recurrent target is still walled to its own family by `feats_ok`.
  - exact: unchanged (32,655 edges, frontier 275).
  - mixgap: 36,029 → 36,198 edges, frontier 239 unchanged.

  So it is a data error with no headline effect today. It only stays harmless because the capacity rows happen to cover for it.
- **Fix:** in `_attn_counts`, before the softmax fallback, set `kinds = ["rec"] * n` when `mt in SSM_TYPES | LINEAR_TYPES`. More generally, do the same whenever classify's mixer is not softmax and the layout cannot be read.

### 3. check_post accepts a wrong round number

- **Where:** `graph/check_post.py:164`. `compare` accepts the truncation `t` as well as the rounding `r`, and `_unit` (`:133-134`) makes a trailing zero a unit of 10 / 100 / 1000. A round number in the draft therefore passes for anything in `[x - u/2, x + u)`.
- **Examples:**

  | draft says | true value | result |
  |---|---|---|
  | "80" | 89.9 | passes |
  | "100" | 199 | passes |
  | "2,000" | 2,999 | passes |
- **Repro:** `test_check_post_approx_rejects_wrong_round_numbers`
- **Fix:** accept truncation only when the unit comes from decimals (u ≤ 1). Units that come from trailing zeros should round only.

### 4. check_post accepts a sibling model's name

- **Where:** `graph/check_post.py:146-153` (`name_ok`). The draft's tokens only need to be a subset of some alias's tokens.
- **What's wrong:** "MiMo-V2.5" passes for "MiMo-V2.5-Pro", "DeepSeek-V4" for "DeepSeek-V4-Pro", and a bare "Pro" passes too. That is exactly the ambiguity in the post's AA / OR table and the record-holder names.
- **Repro:** `test_check_post_name_rejects_a_sibling_model`
- **Fix:** require every alias token that carries a version or size (a digit, or pro / flash / mini / ...) to be present. Alternatively, match a whole normalised alias.

### 5. emulate v1 writes a wrong checkpoint for a Qwen-MoE host with a sliding window

- **Where:** `elephantine/emulate.py:110-111`. `Fam("qwen2_moe")` and `Fam("qwen3_moe")` take the default `window="none"`, so `_windows` (`:199-208`) reports every host layer as full.
- **What's wrong:** the stock Qwen2-MoE / Qwen3-MoE classes do apply `use_sliding_window` / `sliding_window` / `max_window_layers`. As a result:
  - `_place` puts full-window target layers on host layers that slide;
  - `host_config_dict` never switches the host's window off;
  - `gate()` accepts the pair and v1 builds it without refusing.
- **Error:** relative max logit error 1.42 (qwen3_moe) and 1.44 (qwen2_moe). With `use_sliding_window=False` it is about 1e-15.
- **Impact:** the released Qwen1.5-MoE / Qwen2-57B / Qwen3-MoE configs ship `use_sliding_window: false`, so it takes a config that enables it. When that happens, it fails silently: no refusal, and only `verify` would show it.
- **Repro:** `test_qwen_moe_host_with_sliding_window_is_built_exactly[qwen3_moe|qwen2_moe]`
- **Fix:** give the two families `window="layers"` and handle them like qwen2 / qwen3. Alternatively, set `use_sliding_window: False` in the host changes, or refuse.

### 6. ALiBi ← learned-abs: the "width" currency is never charged

- **Where:** `graph/model_graph.py:39`. `POS_ORDER["alibi"]` has `("learned_abs", "width", True)`, and its comment says "Currency: the reserved channel (width), bounded T". Nothing in `emul_matrix` charges extra hidden for this edge (the only width factors are the norm mirror and the sequential carry), and `edge_plan` / `reduce.axis_gaps` show none either.
- **What's wrong:** a learned-abs host of the same hidden size hosts an ALiBi target. For example, `SourAsslips/Sour` (GPT-2, h 1024) hosts `bigscience/bloom-560m` (h 1024).
- **Repro:** `test_alibi_on_learned_abs_host_needs_a_reserved_channel` (synthetic) and `test_alibi_on_learned_abs_host_snapshot_pair`.
- **Impact:**
  - Requiring hidden_h ≥ hidden_t + 1 (or + 2) removes 4 headline edges; the frontier stays at 271.
  - Dropping the edge altogether removes 49 edges; the frontier is still 271. That would be the consistent choice if the carry is a pre-norm problem (see suspicious item 1).
- **Fix:** charge the reserved channel(s) in `emul_matrix` (and in `axis_gaps`), or reclassify the edge (see suspicious item 1).

### 7. post_numbers' elephantine cache key misses a file it reads

- **Where:** `graph/post_numbers.py:163`. `ELEPH_DEPS = GRAPH_DEPS + ["elephantine/*.py"]`, but `loaded_split` → `loader_coverage._stock_tests` imports `tests/test_stock_loaders.py`. The installed transformers version is not hashed either.
- **What's wrong:** adding a stock case or a released checkpoint there changes the loaded-and-tested / priced-only numbers, but a cached run keeps returning the old ones.
- **Repro:** `test_eleph_cache_key_covers_the_stock_loader_tests`
- **Fix:** add `tests/test_stock_loaders.py`, and optionally hash `transformers.__version__` into the key.

### 8. mode="mixgap" skips the recurrent capacity check for attention-free hosts too

- **Where:** `graph/model_graph.py:3826-3828`: `if mode not in ("capacity", "mixgap"): ok &= _rec_ok(P)`. The comment's reason, "mixgap grants attention-for-recurrent anyway", holds only for a host that has attention layers.
- **What's wrong:** a pure SSM host has no attention to stand in, yet in mixgap a d_state-16 Mamba hosts a d_state-32 Mamba (payable and exact refuse the pair).
- **Repro:** `test_mixgap_pure_ssm_host_still_needs_the_recurrent_dims` (synthetic Mamba pair).
- **Impact:** 0 snapshot edges today. The 29 mixgap edges that fail `_rec_ok` all have a host with softmax layers, and the mixgap frontier stays at 239.
- **Fix:** in mixgap, apply `_rec_ok` where the host has no softmax layer (mixer `ssm` / `linear`). Once finding 2 is fixed, the rule is `k{FU,FT,LU,LT} == 0`.

### 9. universal.from_gemma caps Gemma-3's attention, which transformers never does

- **Where:** `elephantine/universal.py:1369`, `:1381`. The host's attention softcap is set from `attn_logit_softcapping` for every Gemma.
- **What's wrong:** `Gemma3Attention` (transformers 4.57) stores the value but never passes `softcap=` to the attention function. A Gemma-3 config with a non-null value gives a relative error of about 4e-2 instead of 1e-15.
- **Impact:** released Gemma-3 configs carry `attn_logit_softcapping: null`, so this is latent. The relation side is consistent: no snapshot Gemma-3 is classified `softcap=attn`.
- **Repro:** `test_from_gemma3_does_not_cap_attention_the_target_does_not_cap`
- **Fix:** set the cap only for the families whose attention passes it (Gemma-2, VaultGemma, ...).

### 10. edge_plan labels the TransMLA x2 only for GQA / MQA hosts

- **Where:** `graph/edge_plan.py:97` (`axis_notes`). The note is added only for `sb["kv"] in ("GQA", "MQA")`, while `reduce.axis_gaps` (like `emul_matrix`) charges the x2 to every non-MLA host. This is the old "penalty skips host kinds" bug, surviving in the label.
- **What's wrong:** for an MHA host the plan shows need = 2 × target with an empty note.
- **Impact:** cosmetic. MLA ← plain is a wall in every mode but capacity, so the zoo's headline edges never show it.
- **Repro:** `test_edge_plan_transmla_note_for_every_non_mla_host`
- **Fix:** use `"MLA" not in sb["kv"].split("|")`.

## Suspicious, not confirmed

1. **ALiBi ← learned-abs and the pre-norm carry.** The edge is exact in `attn/positional_cross.py`'s logit-field span check, which has no residual stream or norm. In a real model, the ramp b(j) = m j has to live in a residual channel (from the position table) that every pre-LayerNorm also sees. That is the carry measured not payable for CLA (`block/prenorm_carry.py`, ~6e-2) and for the parallel block. It is not measured here, so I can't claim it is wrong, but it is marked exact. Dropping it would remove 49 headline edges and leave the frontier unchanged.
2. **The RoPE frequency stance differs between two rules.** The rot gate (`rope_fits`, `hf/partial_rope_host.py`: "the frequency schedule is free -- it is the buffer") lets any RoPE host write its `inv_freq` buffer. That covers llama3 / YaRN schedules on fixed-rotary hosts like InternLM2 or Qwen-1. KV packing (`kv_heads_fit`) instead takes the schedule as config-bound: it needs `rope_pairs_free` (longrope) for repeated frequencies, and it walls fixed-rotary hosts of packable targets to their MHA expansion (`emul_matrix`, `rpf`). Under the buffer stance, packing would work on every host; under the config stance, the rot gate is too generous for fixed-rotary hosts. One of the two should change.
3. **`kv_heads` / `q_per_kv` read only the top-level attention shape.** `kv_heads_fit` uses them, but `_attn_variants` widens `kv_cap` / `qheads` / `head_dim` only. Examples:
   - MiMo-V2.5's SWA layers have 8 KV heads with 8 queries each; the dims say 4 × 16.
   - Gemma-4 31B's dims combine 16 KV heads (sliding) with head_dim 512 (global layers).

   Neither side is tied to a layer kind, so a host's narrower full layers can be credited with its sliding layers' head count. I found no snapshot pair where this flips an edge: `kv_cap`, the rpf wall and K = V (FT) keep the cases I tried in check.
4. **A vanilla FFN host plays a GLU target at 2x depth.** `ffn` is not a categorical slot: `FFN_RANK` only sets `depth2`, and `test_vanilla_glu_not_mutually_emulable` asserts this. An exact product act(g·x)·(u·x) from ungated layers is not obvious (it is not exact for GeLU / ReLU), and no construction is cited.
5. **The `loop` slot is never checked.** `LOOP_ORDER` says only looped ≤ untied, and "the reverse needs per-iteration conditioning". `emul_matrix` lets a looped host play an untied target whenever `param_layers` and `eff_layers` fit: 298 headline edges (hosts: IQuest-Coder-V1-40B-Loop 204, Nanbeige4.2-3B 42, qwen3loop-0.6b 32, Dory-Beam-4B 19, LFM3-1.7B-RD 1). That is right only if the host's loop count can be set to 1 by config, which nothing checks.
6. **`_normalize` is not idempotent for LFM2.** In configs without `block_ff_dim` (6 snapshot configs, e.g. onnx-community/LFM2.5-350M-ONNX), the 2/3 auto-adjust is applied again on a second pass. `dims_of(raw) != dims_of(_lm_config(raw))` for those 6. Today no path feeds a normalised config back into dims_of, so this is latent.
7. **post_numbers `funnel.non_decoder`** counts configs (27, 8 distinct types), while the draft says "18 non-decoder types". **`funnel.drafter`** is a required anchor with no phrase in the draft. `max(hosts, ...)` at `:261` raises if `ok` is ever non-transitive. `pidx` is keyed by basename, so two reps with the same basename would swap rows; there are none today.
8. **check_post `NUM` / `CELL_NUM`:** a malformed "2,0839" reads as 2,083 in a cell with no right context. This does not occur in the draft.
9. **universal.py:**
   - `_ln_from` is defined twice (`:1233`, `:2753`). The later one wins, and it does not allow a bias-free LayerNorm.
   - `_norm_from` falls back to eps 1e-6 while `_norm_any` falls back to 1e-5. Neither fallback is reached by stock modules.
10. **emulate.py:**
    - A non-integer head_dim ratio is ~1e-8 rather than exact. The cause is the fp32 `inv_freq` (documented).
    - `_place` refuses full-window targets on "global" hosts (Phi-3-mini, `sliding_window: 2047`) and on the sliding layers of "layers" hosts, even where the host could switch its window off. This is too conservative but never wrong.
11. **frontier.blockers** compares MLA by string equality (`sa["kv"] == "MLA"`) while `emul_matrix` uses branch sets. This is equivalent on real points (no branch sets) and only affects the explanation text.
