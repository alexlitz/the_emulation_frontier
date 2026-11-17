# `graph/data/recipes.json`

Written by `python -m recipes.make_recipe --index`. It reads `recipes/validation.json` (written by
`python -m recipes.validate`) and `graph/data/frameworks.json` (from `python -m graph.frameworks`, taken from
the working tree or else from the `zoo-metrics` branch; without it the serving-framework command lines are
omitted). It lists, for every zoo model and each of the five run-tested frameworks, the route a recipe script
takes to run that model there, and how far that route has been validated. The zoo page reads it. The schema
is versioned by `schema_version`. Fields may be added within a version. No field is renamed or removed.

## Recipes

A **recipe** is one self-contained Python file, written by
`python -m recipes.make_recipe <target_repo> --framework F [--host H] [--route R]`. F is one of `pytorch`,
`jax`, `tinygrad`, `tensorflow`, `dynet`. The script has two stages:

* `build [--out DIR]` downloads the target from the Hub. If the route needs a host, it builds the host's
  weights with the repo's constructions. It writes a run directory (`config.json` and `model.safetensors`,
  plus the tokenizer files). A `native` route needs nothing: it downloads over HTTPS with the standard library.
  The other routes need `torch`, `transformers`, `safetensors`, `huggingface_hub` and
  `pip install ./recipes/tef_recipes`. That package vendors `elephantine/universal.py`,
  `attn/gated_delta_layer.py` and `elephantine/emulate.py` verbatim (`python -m recipes.vendor`).
* `run [--dir DIR] [--prompt TEXT | --ids 1,2,3] [--new-tokens 32] [--npz OUT.npz]` loads the run directory
  with a numpy safetensors reader that is inlined in the script. It runs greedy generation with a KV cache (and
  an SSM state cache) in framework F alone, with no `transformers` and no repository code (`torch` only for
  `pytorch`). The tokenizer is optional: the script uses `tokenizers` when it is installed, or else `--ids`.

The script's header gives both requirement lines (`# requirements-build:`, `# requirements-run:`). The
framework code is only imported for `run`, so the build stage can run in another environment (DyNet's
`run` needs CPython 3.8). tinygrad on CPU needs a C compiler. Without `clang`, set
`CC=recipes/frameworks/tinygrad/zigcc.py` (zig cc from the `ziglang` wheel).

`--framework vllm|llama.cpp|mlx-lm` writes a shell recipe instead: the `native_cli` command below. These are
not run-tested here.

### Route kinds

| `route` | what runs in framework F | built by |
|---|---|---|
| `native` | the target's own architecture: F's Llama-style decoder reference (`recipes/frameworks/F/llama.py`) on the target's published weights | download only |
| `exact-host` | a Llama-style host checkpoint whose weights make it compute the target exactly (an exact edge of the headline relation, the `elephantine/emulate.py` v1 constructions), run by the same Llama-style reference | `tef_recipes.build.build_exact` |
| `universal` | the universal host (`elephantine/universal.py`) set to the target by its `from_*` loader, exported in the `tef-ucore` format and run by F's universal-host reference (`recipes/frameworks/F/universal.py`) | `tef_recipes.build.build_universal` |
| `limit-host` | reserved: a payable (limit / floor / regime) edge. Such an edge is approximate, and v1 never builds one, so no model carries this route | none |

The route chosen is the first that applies: `native`, then the cheapest `exact-host` (by `cost.params_x`;
the built host config must also be one the Llama-style reference runs), then `universal`. `route: null`
means that no route exists (see `why`). The five frameworks have the same two references, so a model's
route is the same in all five. Only `status` differs.

The Llama-style reference runs `model_type` llama / mistral / ministral / qwen2 / qwen3, dense SwiGLU, RoPE
`default` / `linear` / `llama3` / `yarn` (partial rotary allowed), q/k/v/o and MLP biases, Qwen-3 qk-norm,
per-layer sliding windows, and a tied or untied head (`recipes/common/llama_spec.py`).

### `tef-ucore` (the universal host's run format)

`config.json` has `"format": "tef-ucore"`, the `UConfig` widths, and per-layer flags (which sub-blocks are
live, and which branch of each). `model.safetensors` holds only the tensors those flags read. The names are
the torch `state_dict` names of `UniversalElephantine`, stored as fp32. The exporter
(`tef_recipes/ucore_export.py`) drops a sub-block only when its output projection (weight and bias) is
exactly zero, which is exact. It **refuses** a model when any branch outside the core set
(`CORE_BRANCHES`) is not at its off value. So a ucore checkpoint always computes what `UniversalElephantine`
computes. The refusals seen on the validation cases are listed in `export_refusals`.

## Layout

```jsonc
{
  "schema_version": 1,
  "generated": "2026-11-13",
  "snapshot": "2026-11-10",                 // graph.model_graph.ZOO_SNAPSHOT
  "frameworks": {                           // the five run-tested frameworks
    "jax": {"package": "jax", "version": "0.12.0.1", "python": "3.12.6",
            "venv": "$TEF_RECIPES_VENVS/jax",   // where the validation found it (null: not installed)
            "references": ["recipes/frameworks/jax/llama.py", "recipes/frameworks/jax/universal.py"]}
    // "pytorch", "tinygrad", "tensorflow", "dynet"
  },
  "routes": {"native": "...", "exact-host": "...", "universal": "...", "limit-host": "..."},
  "validation": [                           // every recorded run, sorted by id
    {"id": "jax/native/HuggingFaceTB/SmolLM2-135M",
     "framework": "jax", "route": "native", "target": "HuggingFaceTB/SmolLM2-135M",
     "host": null,                          // exact-host: the host repo; universal: "universal"
     "loader": null,                        // universal: the from_* loader
     "kind": "checkpoint",                  // "checkpoint": a released model, through its generated script
                                            // "random": a tiny random-weight config (recipes/cases.py), target "random:<case>"
     "script": "recipes/scripts/jax/HuggingFaceTB__SmolLM2-135M.py",
     "max_abs_diff": 1.9e-4,                // prompt logits vs transformers fp32 (random: vs the torch universal host, fp64)
     "max_abs_diff_pos1": 4.6e-5,           // ... over positions >= 1
     "logit_scale": 24.9,                   // max |reference logit|
     "hf32_fp64_floor": 1.3e-4,             // transformers fp32 vs transformers fp64 (its own rounding error)
     "fp64_max_abs_diff": 2.7e-4,           // this run vs transformers fp64
     "greedy_tokens": 32, "greedy_identical": true, "first_divergence": null,
     "decode_max_abs_diff": 8.6e-5,         // the cached decode steps' logits vs the reference
     "pass": false,                         // strict: max_abs_diff <= 1e-4 and greedy identical
     "pass_floor": true,                    // greedy identical and fp64_max_abs_diff <= max(1e-4, 4 x hf32_fp64_floor)
     "features": ["gqa", "mt:llama", "rope:default", "tied"],   // native / exact-host: the Llama-style features run
     "run_params": 124439808,               // universal: the exported host's parameter count
     "seconds": 212.5, "date": "2026-11-13",
     "error": "..."}                        // present when the run failed
  ],
  "export_refusals": {"<case>": {"loader": "from_phimoe", "model_type": "phimoe", "refused": "<the branch>"}},
  "models": {
    "<repo id>": {
      "model_type": "llama", "shape": 17,   // shape = the zoo node index (frameworks.json repos[repo].shape)
      "loader": "from_llama",               // the universal-host loader that accepts the config, or null
      "frameworks": {
        "jax": {
          "route": "native",                // a route kind, or null
          "host": null,                     // exact-host: the host repo; universal: "universal"
          "cost": {"params_x": 1.0,         // run-time params / target params (null: not computed)
                   "width_x": 1.0, "depth_x": 1.0},
          "script": "recipes/scripts/jax/HuggingFaceTB__SmolLM2-135M.py",   // a checked-in script, or null
          "make": "python -m recipes.make_recipe HuggingFaceTB/SmolLM2-135M --framework jax",
          "status": "e2e",                  // "e2e" | "family" | "untested" (null when route is null)
          "validated_by": "jax/native/HuggingFaceTB/SmolLM2-135M",   // a validation id, or null
          "why": null                       // route null or status "untested": the reason
        }
      },
      "native_cli": {                       // from frameworks.json; not run-tested here
        "vllm": {"route": "native", "cmd": "vllm serve <repo>"},
        "llama.cpp": {"route": "backdoor", "host": "<host repo>", "edge": "exact", "params_x": 1.3,
                      "buildable": true,    // elephantine/emulate.py v1 builds this edge
                      "cmd": "python -m elephantine.emulate <repo> <host> --out DIR && ..."},
        "mlx-lm": {"route": null, "why": "..."}   // or "buildable": false with "why_not_buildable"
      }
    }
  },
  "counts": {"jax": {"models": 3901, "native": 0, "exact-host": 0, "universal": 0, "none": 0,
                     "e2e": 0, "family": 0, "untested": 0, "tested_route": 0}},
  "native_cli_source": "graph/data/frameworks.json (zoo-metrics)"
}
```

### `status`

* `e2e`: this repo went end to end through its recipe script in F (a `checkpoint` validation entry with
  `pass` or `pass_floor`).
* `family`: the same code path passed end to end in F, on another checkpoint or on a tiny random-weight
  config. For `universal` this is the same `from_*` loader. For `native` it is every Llama-style feature the
  model uses (`features`). For `exact-host` it is the host checkpoint's features plus an `exact-host` run in
  F. The model's own checkpoint was not run.
* `untested`: a route exists, but no validation covers its code path in F.

A model counts as having a **tested route** in F when its status is `e2e` or `family` (`counts.tested_route`).

### Why two pass criteria

Two fp32 implementations whose kernels sum in different orders differ by about their own rounding error.
transformers' fp32 logits are themselves `hf32_fp64_floor` away from the exact (fp64) logits. That is 1.3e-4
for SmolLM2-135M, at position 0 where its activations are massive, 1.0e-4 for GPT-2 and 2.4e-3 for
Pythia-14m. So the spec's 1e-4 between two fp32 runs can be below the noise. `pass` is that strict test.
`pass_floor` asks instead that the run be as close to the exact logits as transformers' own fp32 is, within
4x, and that the greedy continuation be identical.
