# Architecture metrics: the JSON schema the zoo consumes

Schema document version 2 (2026-11-14): adds the `arch_breakdown.json.gz` section at the end.

Two files, both generated, both read-only for the UI:

* `graph/data/arch_metrics.json`, written by `python -m graph.arch_metrics`
* `graph/data/frameworks.json`, written by `python -m graph.frameworks`

Every number is computed from a model's `config.json` (the snapshot `graph/hub_10k.json`) plus code facts
(`dims_of`, `classify`, `FEATURES` and the per-type sets in `graph/model_graph.py`). Nothing is scraped from
model cards. The schema is versioned by `schema_version`. Fields may be added within a version, but no field
is renamed or removed and no unit changes.

## Conventions (all fields)

* **Units.** Parameter counts are integers. Bytes are integers (or floats where a dtype has 0.5 bytes). FLOPs
  are integers **per token** unless the name says otherwise. Arithmetic intensity is FLOPs per byte (float).
* **`null` means unknown.** It never means zero. A field that does not apply (KV cache of a pure SSM, the
  window of a global layer) is `0` or absent from a per-kind map, never `null`.
* **FLOPs are 2 x multiply-accumulates** (one multiply plus one add = 2 FLOPs), counting only matrix products
  (linear layers, attention QK^T and AV, SSM state contractions, convolutions). Norms, activations, softmax,
  RoPE and routing top-k are not counted. This is the `torch.utils.flop_counter.FlopCounterMode`
  convention, so the trace check compares like with like.
* **Context lengths.** `L` is one of `"1k"`, `"4k"`, `"8k"`, `"32k"`, `"128k"`, `"max"` =
  1024, 4096, 8192, 32768, 131072 and the model's `max_context`. The numeric value of every L
  is in `contexts` at the top level, and per repo in `metrics.max_context`. An L above `max_context` is still
  computed (the formula is defined there). `beyond_max: true` flags it.
* **max_context** is read by arch_metrics itself (`_max_context`), not taken from the classifier's 4,096 default.
  `max_context_source` says where it comes from: `config: <key>` (the positions key; a larger YaRN-style window the
  config declares is named in brackets), `rope_scaling ...` (no positions key: original x factor), `config: <key>`
  for model_max_length / max_sequence_length / block_size / ..., `class default: ...` (the stock config class),
  `unbounded: ...` (a recurrent or window-only stack: no positional limit; `max_context_unbounded: true`, sized at
  128k, never `beyond_max`), or `default: ...` (4,096, where nothing states a limit).
* **dtypes.** `bf16` = 2 bytes per element, `fp8` = 1 byte, `int4` = 0.5 bytes. Weight bytes are
  `params x bytes`, with quantisation scales and zero points ignored. KV and state bytes are elements x bytes.
* **Batch.** `batch_B` (top level, currently 16) is the "batch B" of the memory and bandwidth sections.
  Weights are shared across the batch. KV and state scale with B.
* **Decode at position L.** The cache already holds L-1 tokens, and the new token is token L. A layer with
  window W attends to `min(L, W)` keys (itself included). Prefill of L tokens is reported **amortised per
  token**, that is, the total prefill FLOPs divided by L.
* **Two attention-FLOP conventions.** `causal` counts the keys a query really attends to:
  `sum_{i=1..L} min(i, W)` over the prefill. This is what FlashAttention-style kernels do. `dense` counts the
  full `L x L` (masked) score matrix, as HF's eager and SDPA paths compute it. That is what
  `FlopCounterMode` measures, and the trace check verifies it. The two differ only in the attention term,
  by a closed form, so verifying `dense` verifies `causal` too. The headline (`flops.*.total`) is `causal`.
* **The LM head.** Prefill counts the LM head at every position (HF's default forward and the usual
  `2 N L` convention). Decode counts it once.
* **MLA (DeepSeek-V2/V3 style).** There are two KV-cache conventions: `latent` (the compressed latent
  `kv_lora_rank` plus the shared rope key `qk_rope_head_dim` per layer, as vLLM / SGLang / FlashMLA cache it)
  and `materialized` (per-head K and V after `kv_b_proj`, as HF transformers caches it). The headline
  `kv.*` uses `latent`. `kv_materialized.*` gives the HF number, which is the one trace-verified against
  `DynamicCache`. For every non-MLA model, `kv_materialized` equals `kv`.
* **The headline KV is as allocated.** `kv.*` and `kv_per_token.*` are the bytes vLLM, SGLang, transformers and
  llama.cpp actually store, not a theoretical floor. A K = V layer (no `v_proj`) is the one case where the two
  differ. Gemma-4's `attention_k_eq_v` global layers compute V = v_norm(k_proj x) and K = RoPE(k_norm(k_proj x)),
  two different tensors, and every runtime caches both: transformers 5.18 `modeling_gemma4.py`
  (`Gemma4TextAttention.forward`, `past_key_values.update(key_states, value_states, ...)`), vLLM `gemma4.py`
  and llama.cpp `gemma4.cpp`. So the headline counts **two tensors** there. DeepSeek-V4's cache aliases values
  to keys (`self.values = self.keys`) and counts one. The theoretical minimum, one tensor for every K = V layer,
  is the secondary field `kv_minimum` (equal to `kv` for every other model, with a `note` saying which). It is
  checked against the real transformers 5.18 cache object of a tiny Gemma-4 K = V model (tests/test_kv_allocated.py).
* **Sliding windows.** A window-`W` layer caches its `W - 1` most recent tokens, which is what transformers'
  `DynamicSlidingWindowLayer` keeps (with the new token, the window is `W`). So `kv` of such a layer is
  `min(L, W - 1)` tokens, and `kv[L].in_window.sliding` is `L <= W - 1`.
* **DSA (DeepSeek sparse attention: DeepSeek-V3.2, GLM-5).** The lightning indexer scores every key
  (`index_n_heads x index_head_dim`). The attention reads the `index_topk` selected keys. The indexer's keys are
  cached (`kv_per_token.elems.dsa_indexer`, `index_head_dim` per layer) beside the MLA latent. Its weights are
  in `params`.
* **Recurrent cores.** The convention counts each recurrent layer's causal conv (`2 x channels x kernel` per
  token) and its state contraction (Mamba-1 / Mamba-2: `2 x d_inner x N`, the `C . h` readout; gated DeltaNet / KDA /
  lightning attention: `6 x heads x d_k x d_v`). Elementwise state updates are not counted. `flops.*.recurrent`
  holds this term.
* **What the transformers code executes, where it differs.** The trace check compares like with like, so these
  code facts are part of the verified formulas (`arch_metrics.code_matmul`, `trace_attn`, `trace_core`,
  `hf_alloc`). They are not in the headline numbers:
  * Llama-4 and Qwen3-VL-MoE multiply every expert on every token. The router masks the outputs.
  * Doge recomputes `dt_proj` over the whole cache at each decode step.
  * A cached conv step is elementwise. A prefill conv computes `L + K - 1` outputs.
  * Falcon (new decoder architecture) caches K and V broadcast to every query head.
  * RecurrentGemma attends over a fixed `W`-slot buffer.
  * LFM2 allocates a conv state for every layer.
  * GPT-1 (`openai-gpt`) has no KV cache.

## Provenance and validation, per field

Every metrics block carries two maps, `prov` and `valid`, keyed by **field group**:

| group | covers |
|---|---|
| `params` | `params.*` (total, embedding, non-embedding, per component) |
| `active` | `params.active*` |
| `kv` | `kv_per_token.*`, `kv.*` |
| `state` | `state.*` |
| `flops` | `flops.*` |
| `memory` | `memory.*` |
| `bandwidth` | `bandwidth.*` |

`prov[group]` is where the number comes from:

* `"config"`: read straight from config keys with the generic formula
* `"code"`: the config plus code facts (per-type sets, defaults and quirks in `model_graph.py` /
  `arch_metrics.py` that were read from the architecture's source)
* `"trace-verified"`: the formula for this model type was executed against the real HF architecture code
  and matched exactly (see `valid`)

`valid[group]` is the strongest check the formula passed for this repo's model type:

* `"symbolic-proved"`: the model type's prefill was exported with a symbolic sequence length (`torch.export`,
  one export per window regime). The FLOP and KV-cache polynomials in L read off the graph equal the closed
  forms identically, so they hold for every L (L >= 2). The decode step is trace-verified. Only `kv` and
  `flops` can carry this status. MoE routing and per-token recurrent loops are data dependent and do not
  export.
* `"trace-verified"`: equal to the real code at sampled points. For params this means instantiating this
  repo's exact config on the `meta` device. For FLOPs, KV and state it means tiny real-shaped configs of the
  model type at L = 5 and L = 13 (window 8), inside and beyond the window, prefill and decode.
* `"hub-matched"`: params equal the Hub safetensors count (graph/model_meta.json `hf_total`) exactly, or
  exactly up to named blocks the checkpoint stores beyond the LM. Those blocks are a stored copy of a tied
  head, MTP layers, router-bias buffers, old mask / rotary buffers, and a vision tower counted with a stock
  class. They are listed in `params.hub_delta_reason`.
* `"formula-only"`: no executable reference (a model type with no stock transformers class, or remote code
  that we never execute). An approximate Hub explanation also lands here: a quantised checkpoint, or towers
  we cannot count. `params.hub_delta_reason` says which.
* `"failed"`: a check ran and disagreed. Either the stock code's count differs, or the Hub count differs by
  more than 0.5% for a non-quantised checkpoint with no explanation. The UI should show the number with a
  warning. `notes` says why.

  The 105 repos that failed at the competitive evaluation are triaged in `graph/data/params_triage.json`.
  * `A-fixed` (42): a formula gap, now fixed in `arch_metrics.py`. Each fixed repo matches its Hub count
    exactly, or up to named stored blocks.
  * `A-open` (30): a formula gap that needs a bespoke builder.
  * `B` (17): the checkpoint is not the LM its config describes. Examples are partial or MTP-only uploads,
    NPU binaries, rerankers and drafters.
  * `C` (16): a config-reading gap outside this file, in `model_graph.py` or `fetch_meta.py`.

  A repo that is still `failed` carries a `notes` entry `params check, triaged (...)` with its reason.

`active` is trace-verified when params are and the model is dense or its FLOPs are trace-verified. Derived
groups (`memory`, `bandwidth`) inherit the weakest status of their inputs. For MLA models `kv` is
`formula-only` (the latent). A note gives the status of `kv_materialized`. Where only the recurrent core is
unverified, `flops` is `formula-only` and a note says the rest is trace-verified.

## `arch_metrics.json`, top level

```jsonc
{
  "schema_version": 1,
  "generated": "2026-11-13",          // date of the run
  "snapshot": "2026-11-10",           // graph/hub_10k.json fetch date (= zoo_data()["snapshot"])
  "transformers_version": "4.57.6",   // the stock code the trace checks ran against
  "contexts": {"1k": 1024, "4k": 4096, "8k": 8192, "32k": 32768, "128k": 131072},   // "max" is per repo
  "batch_B": 16,
  "dtype_bytes": {"bf16": 2, "fp8": 1, "int4": 0.5},
  "conventions": { ... },             // the strings above, machine-readable
  "coverage": {
    "repos": {                        // repos per valid status, per group (all seven groups)
      "params": {"symbolic-proved": 0, "trace-verified": 0, "hub-matched": 0, "formula-only": 0, "failed": 0},
      ...
    },
    "types": {                        // model types per status, for params / flops / kv / state
      "flops": {"symbolic-proved": 0, "trace-verified": 0, ...},
      ...
    }
  },
  "types": { "<model_type>": TypeValidation, ... },
  "shapes": [ Shape, ... ],           // index i == zoo_data()["nodes"][i]
  "repos": { "<repo id>": Metrics, ... }
}
```

### `Shape`

`shapes[i]` lines up with the zoo page's `nodes[i]`, the same `_dedup_points(frontier.load())` order.

```jsonc
{
  "rep": "Qwen/Qwen3-8B",        // the repo whose Metrics the shape shows (zoo node's members[0])
  "n_members": 12,
  "metrics": Metrics,            // == repos[rep], copied so a shape needs one lookup
  "range": {                     // min / max over the shape's members, for the headline scalars that can
    "params.total": [min, max],  // differ within a shape (vocab and context are not shape axes)
    "params.active": [min, max],
    "kv_per_token.bf16": [min, max],
    "max_context": [min, max]
  }
}
```

### `Metrics` (per repo)

```jsonc
{
  "model_type": "qwen3",
  "arch": "Qwen3ForCausalLM",
  "max_context": 40960,
  "max_context_source": "config: max_position_embeddings",
  "max_context_unbounded": false,
  "layers": {
    "total": 36,                       // decoder layers of the LM, unrolled (looped models: unrolled depth)
    "global": 36,                      // softmax attention, full window, own K/V
    "sliding": 0,                      // softmax attention with a window
    "window": null,                    // the sliding window W in tokens (null when there are no sliding layers)
    "mla": 0,                          // of global+sliding, how many are MLA
    "kv_shared": 0,                    // layers that read another layer's K/V (CLA / YOCO / Gemma-3n): no cache
    "k_eq_v": 0,                       // layers whose V is computed from their K (no v_proj)
    "recurrent": 0,                    // SSM / linear-attention / recurrent / conv mixers
    "dsa_indexer": 0                   // layers with a DeepSeek sparse-attention indexer key cache
  },
  "params": {
    "total": 8190735360,               // every LM weight, embeddings and an untied head included
    "embedding": 622329856,            // input embedding table(s)
    "lm_head": 622329856,              // 0 when tied to the input embedding
    "non_embedding": 6945079296,       // total - embedding - lm_head
    "active": 8190735360,              // per token: dense + shared experts + top-k routed experts, plus
                                       // embedding and head (MoE active <= total, dense active == total)
    "active_non_embedding": 6945079296,
    "routed_experts": 0,               // all routed-expert weights (the part MoE activates k/E of)
    "tied_embeddings": false,
    "extra_towers": null,              // non-LM params of the stock class (vision / audio towers, projectors),
                                       // counted on the meta device; null = none or not counted
    "hub_total": 8190735360,           // graph/model_meta.json hf_total (null = not published)
    "hub_delta": 0,                    // hub_total - total (null without hub_total)
    "hub_delta_reason": "exact",       // "exact"; or the exact blocks, e.g. "MTP layers + vision tower (Qwen3-VL
                                       // vision encoder class)", "tied head stored", "router bias buffers"; or an
                                       // approximate reason ("quantised checkpoint: ...", "vision / audio / other
                                       // towers ..."); or a signed delta when unexplained; null without hub_total
    "checkpoint_anomaly": null,        // the upload is not the model its config describes (params_triage.json
                                       // bucket B: NPU binaries, partial / MTP-only uploads, rerankers, drafters):
                                       // the reason. Every figure stays the architecture's and params is
                                       // formula-only. null otherwise
    "weight_bytes": {"bf16": ..., "fp8": ..., "int4": ...},          // total params
    "active_weight_bytes": {"bf16": ..., "fp8": ..., "int4": ...}    // active params
  },
  "kv_per_token": {                    // elements cached per token for ONE sequence, summed over the layers of
    "elems": {                         // each kind (a window layer's per-token cost while L <= W)
      "global": 147456,
      "sliding": 0,
      "mla_latent": 0,                 // kv_lora_rank + qk_rope_head_dim per MLA layer (latent convention)
      "k_eq_v": 0,                     // K = V layers, as allocated: K and V (Gemma-4), one (DeepSeek-V4)
      "dsa_indexer": 0,                // DSA indexer keys (index_head_dim per layer)
      "kv_shared": 0                   // always 0: sharing layers cache nothing; kept for the UI legend
    },
    "elems_total": 147456,
    "bf16": 294912,                    // bytes per token at bf16 (growing part, while every window is open)
    "fp8": 147456
  },
  "kv": {                              // total KV bytes for ONE sequence of length L, per L key
    "4k": {
      "L": 4096,
      "beyond_max": false,
      "elems": {"global": ..., "sliding": ..., "mla_latent": ..., "k_eq_v": ..., "dsa_indexer": ...},
      "in_window": {"sliding": true},  // per windowed kind: L <= W (still growing) or false (flat at W)
      "bf16": 1207959552,
      "fp8": 603979776
    },
    "...": {}
  },
  "kv_materialized": {                 // HF-cache convention (MLA as per-head K/V); == kv for non-MLA models
    "per_token_elems": 147456,
    "4k": {"bf16": ..., "fp8": ...}
  },
  "kv_minimum": {                      // the theoretical minimum: one tensor per K = V layer (no runtime does this
    "per_token_elems": 147456,         // for Gemma-4); == kv for every model without K = V layers
    "note": "equal to the headline",   // or "K = V layers counted as one tensor: the theoretical minimum; ..."
    "4k": {"bf16": ..., "fp8": ...}
  },
  "state": {                           // constant per sequence (recurrent / SSM / linear attention / conv)
    "elems": 0,                        // conv state + recurrent state, summed over the recurrent layers
    "conv_elems": 0,
    "ssm_elems": 0,
    "bf16": 0,
    "fp32": 0                          // many runtimes keep SSM state in fp32
  },
  "flops": {                           // per token
    "matmul": 16381470720,             // linear layers (MoE: active experts only) incl. LM head, no attention
    "prefill": {                       // amortised over L
      "4k": {"matmul": ..., "attn": ..., "attn_dense": ..., "recurrent": ..., "total": ...}
    },
    "decode": {                        // the step producing token L
      "4k": {"matmul": ..., "attn": ..., "recurrent": ..., "total": ...}
    }
  },
  "memory": {                          // bytes resident during a forward pass, at bf16 weights and KV
    "4k": {"weights": ..., "kv": ..., "state": ..., "batch1": ..., "batchB": ...}
  },
  "bandwidth": {                       // one decode step at batch 1, bf16 weights and KV
    "4k": {
      "weights": ...,                  // active non-embedding weights + LM head + one embedding row
      "kv": ...,                       // the KV read for the attended span (window-capped), all layers
      "state": ...,                    // recurrent state read + written
      "bytes": ...,                    // sum
      "flops": ...,                    // = flops.decode[L].total
      "intensity": 12.3,               // flops / bytes (FLOPs per byte)
      "batchB": {"bytes": ..., "flops": ..., "intensity": ...}    // B sequences, weights read once
    }
  },
  "prov":  {"params": "trace-verified", "active": "trace-verified", "kv": "trace-verified", "state": "trace-verified",
            "flops": "trace-verified", "memory": "trace-verified", "bandwidth": "trace-verified"},
  "valid": {"params": "trace-verified", "active": "trace-verified", "kv": "symbolic-proved",
            "state": "trace-verified", "flops": "symbolic-proved", "memory": "trace-verified",
            "bandwidth": "trace-verified"},
  "stock_class": true,                 // the config's class is the stock transformers code for its model type
  "notes": []                          // free-text caveats for this repo
}
```

A repo whose config the formulas cannot read has only `{"model_type": ..., "error": "..."}`.

### `TypeValidation` (per model type, keyed by the config's top-level `model_type`)

```jsonc
{
  "stock": true,                       // a stock transformers config class exists at transformers_version
  "repos": 412,                        // repos of this type in the zoo
  "params": {"status": "trace-verified",   // all checked configs equal; "failed" if any differs; else formula-only
             "checked": 380, "equal": 380, // distinct configs instantiated on the meta device
             "mismatches": [{"repo": "...", "formula": n, "trace": n}]},
  "flops":  {"status": "symbolic-proved"},
  "kv":     {"status": "trace-verified"},
  "state":  {"status": "trace-verified"},
  "symbolic": {"regimes": [            // null when the prefill does not export with a symbolic length
    {"range": [2, 6], "flops_trace": "32768*s27**2 + 13631488*s27", "kv_trace": "8192*s27"},
    {"range": [7, null], "flops_trace": "...", "kv_trace": "4096*s27 + 28672"}]},   // per window regime
  "trace_configs": ["Qwen/Qwen3-0.6B", "..."],   // the configs whose tiny versions were traced
  "method": "...",                     // how the checks ran
  "notes": []
}
```

## `frameworks.json`

Native support comes from each framework's own source registry, read as text at a pinned version
(`graph/framework_registries.py` writes `graph/data/framework_registries.json`, with the version, URLs and date).
A backdoor is a natively supported architecture that **emulates** the model under the headline relation,
`emul_matrix(P, mode="payable")`. The model's weights can be rebuilt as that host (elephantine/emulate.py) and
run on the framework. Backdoors are per **shape**, because the relation is. A repo's backdoor for framework
`fw` is `shapes[repos[repo].shape].backdoor[fw]`, used when `repos[repo].native[fw]` is not `true`.

* **Transformers-backend tier.** vLLM maps some architectures to its Transformers backend
  (`_TRANSFORMERS_SUPPORTED_MODELS` in `registry.py`), and TGI's `get_model()` falls back to the transformers
  class for a model type in transformers' causal-LM mapping (at the version TGI pins). Both run the model, but
  neither has a native implementation, so `native[fw]` stays `false` and `repos[repo].via_transformers[fw]` is
  `true`. The page shows it as its own tier, "via Transformers backend". Remote code through `auto_map` is not
  counted.
* **ExecuTorch is per model.** `native.executorch` is `true` only when the model type is registered **and** an
  export path exists for that model: its shape matches the dims of an `examples/models/<dir>` export config
  (or the `gemma4_31b` model), or holds a repo that the recipes or optimum-executorch's tests and README name
  ("X and its variants"). `repos[repo].executorch_why` says which, or why not.
* **Host ranking.** Hosts exclude embedding, reranker, reward and classifier repos. These are detected from the
  Hub pipeline tag (`graph/data/pipeline_tags.json`, written by `python -m graph.fetch_pipeline_tags`, metadata
  only) and the config's `architectures` (`repos[repo].not_generative` gives the reason). The order is: practical
  hosts first (a host over 10x the target's params carries `impractical: true` and is never the top pick unless
  nothing else exists), exact edges first, then the target's own model family (`same_family`), then the fewest
  params.

```jsonc
{
  "schema_version": 1,
  "generated": "2026-11-13",
  "snapshot": "2026-11-10",
  "frameworks": {                      // the 9 frameworks, keys exactly as below
    "vllm": {"repo": "vllm-project/vllm", "version": "v0.33.0rc3", "fetched": "2026-11-13",
             "urls": ["https://raw.githubusercontent.com/..."], "source_note": "...",
             "key": "architecture",    // registry keyed by "architecture" (class name), "model_type" or "both"
             "n_architectures": 251, "n_model_types": 0},
    // "transformers", "sglang", "llama.cpp", "mlx-lm", "tgi", "executorch", "onnx", "transformers.js"
  },
  "repos": {
    "<repo id>": {"model_type": "qwen3", "arch": "Qwen3ForCausalLM", "shape": 17,   // shape = zoo node index
                  "native": {"transformers": true, "vllm": true, "tgi": false, ...},  // null = config names neither
                  "via_transformers": {"tgi": true},      // optional: the Transformers-backend tier (not native)
                  "executorch_why": "export recipe examples/models/qwen3/config/0_6b_config.json",  // optional
                  "not_generative": "pipeline tag feature-extraction"}                     // optional: never a host
  },
  "shapes": [                          // index i == zoo_data()["nodes"][i] == arch_metrics shapes[i]
    {"rep": "<repo>", "popular": "<most-downloaded member>", "n_members": 12,
     "backdoor": {                     // only frameworks where some member is not native
       "llama.cpp": {
         "n_hosts": 21, "n_exact": 21, // every emulating shape with a native generative member (exact = mode="exact")
         "n_practical": 21,            // of those, params_x <= 10
         "hosts": [                    // the best TOP_HOSTS (3): practical, exact, own family, then by params_x
           {"host": "<repo id>",       // the host shape's most-downloaded natively supported member
            "host_arch": "LlamaForCausalLM", "host_model_type": "llama", "host_shape": 3,
            "edge": "exact",           // "exact" | "payable" (a limit / floor / regime edge: approximate)
            "exact": true,
            "width_x": 1.0,            // host hidden / target hidden
            "depth_x": 1.0,            // host unrolled depth / target unrolled depth
            "host_params": 1234,       // the host repo's total params
            "params_x": 1.3,           // host params / target params (target = the shape's popular member)
            "same_family": true,       // the host's model family is the target's (text / wrapper types merged)
            "impractical": false}      // params_x > 10
         ]
       }
     }}
  ],
  "counts": {"<framework>": {"repos": n, "native": n, "missing": n, "unknown": n,
                             "via_transformers": n,   // missing repos the Transformers-backend tier runs
                             "backdoor_exact": n,     // missing repos whose shape has an exact backdoor
                             "backdoor_payable": n,   // ... only a payable (approximate) one
                             "no_backdoor": n, "shapes_native": n, "shapes": n}}
}
```


## `arch_breakdown.json.gz` (schema 1): the detail behind every number

Written by `python -m graph.arch_breakdown` (gzip-compressed JSON, about 15 MB; 79 MB uncompressed). The same
numbers are computed in the browser by `graph/templates/arch_breakdown.js` (equal to the Python field by field on all
3,901 configs, `tests/test_arch_breakdown_js.py`); the page ships each shape's plan (`arch_breakdown.page_plan`,
gzip + base64 in `#bd-detail`) and its statuses, not the precomputed grids. This document's version: 2 (adds this
section; `arch_metrics.json` stays at `schema_version` 1, and the plan's groups gain `at`, the layer indices, and the
plan `order_exact`).

Every number is a sum over the plan's tensors: one layer of a group is a list of tensors (role, kind, in, out,
experts, elements), and the sums equal `arch_metrics`' totals exactly (`breakdown_checks`, run on every config by
`tests/test_arch_breakdown.py`).

### Conventions (in `conventions` at the top level)

* **Roles.** `embedding`, `embedding.other` (position / per-layer / n-gram tables), `lm_head` (0 when tied),
  `final_norm`, `other`, `other.linear`; `attn.{q,k,v,o,gate,q_a,q_b,kv_a,kv_b,experts,kv_proj,router,extra,bias,
  qk_norm,lora_norm,block,misc}`; `mlp.{gate,up,down,linear,receptance,bias,misc}`; `moe.{router,routed.gate,
  routed.up,routed.down,shared.gate,shared.up,shared.down,latent,shared_gate,bias}`; `rec.{linear,misc}`; `norm`.
* **FLOPs** as `arch_metrics`: 2 x MACs of matrix products per token. Decode = the step producing token L;
  prefill = an L-token prefill, `total` over the L tokens (a float: it passes 2^53) and `per_token` amortised.
  `training_approx` = 3 x the forward prefill total, an approximation. `elementwise` (not in any total): softmax 5
  FLOPs per score, RMSNorm 4 / LayerNorm 7 per element (from the norm weights), activations 4 per element (+1 GLU).
* **Weights.** fp32 / bf16 per parameter; fp8 = 1 B + an fp32 scale per 128 x 128 block; int8 = 1 B + a bf16 scale
  per output row; int4 = 4 bits + fp16 scale and 4-bit zero per 128 inputs (AWQ / GPTQ); nf4 = 4 bits + fp32 absmax
  per 64; nvfp4 = 4 bits + fp8 scale per 16 + fp32 per tensor. Embeddings, LM head, norms and routers stay bf16.
  GGUF levels Q8_0 ... Q2_K and BF16: llama.cpp's converter tensors through `llama-quantize`'s per-tensor type choice
  at the pinned commit (`graph/gguf_quant.py`, `graph/data/llama_cpp_quant.json`), plus the header.
* **KV** elements as `arch_metrics.kv` x the format: bf16 2 B, fp8 1 B, q8_0 34 B / 32, q4_0 18 B / 32.
* **Activation** = peak transient bytes of one L-token prefill (bf16, no grad, logits for the last position):
  `flash` (no L x L term) and `eager` (10 B per score: bf16 scores + their fp32 copy + the fp32 softmax; the L x L mask).
* **Serving** = bf16 weights + B x (KV + state) + the flash activation of one L-token prefill.
* **Bandwidth** = one decode step for B sequences: weights once (routed experts touched by B tokens:
  E (1 - (1 - k/E)^B), uniform routing), the head (a tied head reads the table), B embedding rows, B x KV over the
  attended span, B x state read + written.
* **Roofline** = UPPER BOUNDS: decode tokens/s <= B / max(bytes / peak bandwidth, FLOPs / peak FLOPs), prefill time >=
  max(FLOPs / peak, weights / bandwidth), from `graph/data/hardware.json` (vendor peaks, dense, sources cited).

### Top level

```jsonc
{"schema_version": 1, "generated": "2026-11-14", "snapshot": "2026-11-10",
 "contexts": {"1k": 1024, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072},  // + "max"
 "batches": [1, 8, 32], "kv_formats": {"bf16": [2, 1], "fp8": [1, 1], "q8_0": [34, 32], "q4_0": [18, 32]},
 "weight_formats": ["fp32", "bf16", "fp8", "int8", "int4", "nf4", "nvfp4"], "gguf_levels": ["Q8_0", ..., "Q2_K"],
 "roofline_contexts": ["4k", "32k", "max"], "hardware": [...], "conventions": {...},
 "hub_checks": {"gguf": {level: stats}, "gguf_all": stats, "safetensors": {method: stats}},
 "coverage": {group: {status: repos}}, "repos": {"<repo>": Breakdown}}
```

### `Breakdown` (per repo)

```jsonc
{"max_context": 40960, "hidden": 4096, "vocab": 151936, "layers": 36, "layer_order": true,
 "params": {"components": {role: n}, "active": {role: n}, "total": n, "active_total": n, "tied": false,
            "groups": [{"label": "global · FFN", "kind": "full", "n": 36, "np": 36, "window": null, "heads": 32,
                        "kv_heads": 8, "head_dim": 128, "v_head_dim": 128, "experts": null | [E, k],
                        "params": {role: n per layer}, "layer_total": n, "active_layer_total": n,
                        "kv_per_token": elems, "flops_matmul": 2 x active weights, "attn_per_key": 2 h (dqk + dv),
                        "rec_core": n}],
            "by_kind": {label: {"layers", "distinct", "params", "active"}},
            "mtp": {"layers": 1, "params": n}},          // optional: stored beside the LM, not in the total
 "per_layer": [[group index, ...] per layer],          // when layer_order
 "weights": {format: bytes}, "gguf": {"Q8_0": bytes, ..., "BF16": bytes} | null,
 "kv_per_token": {"elems", "by_kind", "bf16", "fp8", "q8_0", "q4_0"}, "state": {"elems", "bf16", "fp32"},
 "flops": {L: {"decode": {"components": {role: n}, "attn_by_kind": {label: n}, "total": n},
               "prefill": {"components": {role: per token}, "attn_by_kind": {...}, "per_token": f, "total": f,
                           "attn_total": f},
               "training_approx": f}},
 "elementwise": {L: {"softmax", "norms", "activations", "total"}},
 "kv": {L: {"L", "elems", "by_kind", "bf16", "fp8", "q8_0", "q4_0"}},
 "activation": {L: {"flash": f, "eager": f}}, "serving": {L: {"1": bytes, "8": bytes, "32": bytes}},
 "bandwidth": {L: {B: {"weights", "kv", "state", "bytes", "flops", "B", "intensity"}}},
 "roofline": {hardware id: {L: {"decode": {B: tokens/s bound}, "prefill_s": seconds bound}}},
 "valid": {group: status}, "notes": [...]}
```

### `valid` groups and statuses

Groups: `params.components`, `params.per_layer`, `flops.components`, `flops.elementwise`, `flops.training`, `kv`,
`state`, `weights.dense`, `weights.quantised`, `weights.gguf`, `activation`, `serving`, `bandwidth`, `roofline`.
Statuses, strongest first:

* `symbolic-proved`: the per-module FLOP polynomials of the exported prefill equal the components identically
  (`arch_breakdown_trace.py symbolic`, 51 types).
* `trace-verified`: per module, equal to the stock transformers code: `named_parameters` on the meta device bucketed
  (mixer / FFN / norms per layer; q / k / v / o, routed experts, router, shared experts) for this repo's exact config
  (1,507 of 1,508 distinct stock configs), and FlopCounterMode's per-module counts on tiny configs (146 configs, 83
  types, all equal).
* `hub-matched`: params equal the Hub safetensors count (inherited from `arch_metrics`).
* `size-checked`: the format's size formula matches real Hub files (`graph/arch_breakdown_hub.py`): GGUF over 2,564
  quantised files of 289 bases, median |error| 0.036%; GPTQ / AWQ / FP8 / compressed-tensors / NF4 / MLX over 446
  repos, median |error| 0.001%. Inherits the weakest of the parameter status.
* `measured`: the model type's peak prefill memory (eager) within 15% of `torch.cuda.max_memory_allocated` on a tiny
  real-shaped bf16 model (`arch_breakdown_trace.py memory`).
* `formula-only`, `approximation` (the training FLOPs), `bound` (roofline numbers), `failed`.
