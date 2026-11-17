# local_rope_theta_unread
live 21 / placed 21 (olmo3 20, plamo3 1); direction over-credit; batch a5_positions

## 1. What the code computes
- OLMo-3:
  - Config: `tf5:models/olmo3/modular_olmo3.py:97-116` (`convert_rope_params_to_dict`) starts both layer types at `{"rope_type": "default"}` and applies a legacy `rope_scaling` ONLY to `rope_parameters["full_attention"]` (`:107-108`). Both types get the same `rope_theta` (`:113-116`).
  - Rotary: the per-layer-type rotary inits a non-default type per layer type (`:222-223`).
  - Effect: every Olmo-3 snapshot carries `rope_scaling` YaRN (factor 8, original 8192, attention_factor 1.208). The 1-in-4 full layers use the YaRN inv_freq (and cos / sin × 1.208), while the sliding layers use theta^(-2i/d). Those are two different tables of 128 rotated dims each.
- PLaMo-3 `pfnet/plamo-3-0.8b-fin-instruct/modeling_plamo.py:212-240`: the config property `rope_parameters` is built in code. Sliding layers get `{rope_theta: rope_local_theta (1e4), default}`. Full layers get `rope_theta` (1e6), as YaRN(factor = rope_scaling_factor, original = initial_context_length) when the factor is not 1. Layer types come from window_size + sliding_window_pattern (`:314`). That is two tables.
- round2-p5 `_rope_schedules` (p5:model_graph.py:1234-1287):
  - It reads per-type tables only from a `rope_parameters` dict keyed by layer type (`_rope_layer_types`), and local thetas only from `rope_local_base_freq` / `swa_rope_theta` / `compress_rope_theta` (`:1279`).
  - OLMo-3 (legacy `rope_scaling`, no `rope_parameters`) and plamo-3-0.8b-fin (no `rope_parameters`, key `rope_local_theta`) therefore get one schedule `(128,)`.
  - The two plamo-3-translate snapshots carry a two-entry `rope_parameters` and already get `(128, 128)`.
  - Not fixed in p5.

## 2. Classifier change
- reads: model_type olmo3 + `rope_scaling` (non-default type) + `layer_types` with both sliding and full; model_type plamo3 + `rope_local_theta` / `rope_theta` / `rope_scaling_factor` / `window_size`.
- new: rope_sched / rope_cap get two entries `(hd, hd)`, in `_rope_schedules`.
- detection:
```python
def detect_local_rope_theta_unread(cfg, model_type):
    lm = lm_of(cfg)
    hd = _head_dim(lm)
    lt = layer_types(lm)
    if model_type == "olmo3":
        rs = lm.get("rope_scaling")
        per = _per_type(lm.get("rope_parameters"))
        if per:
            kinds = {repr(sorted((v or {}).items())) for v in per.values()}
            return [hd, hd] if len(kinds) > 1 else None
        if (isinstance(rs, dict) and str(rs.get("rope_type") or rs.get("type") or "default") != "default"
                and any("sliding" in x for x in lt) and any("full" in x for x in lt)):
            return [hd, hd]
        return None
    if model_type == "plamo3":
        has_sliding = bool(lm.get("window_size") or any("sliding" in x for x in lt))
        loc, glob = lm.get("rope_local_theta", 10000), lm.get("rope_theta", 10000)
        if has_sliding and (loc != glob or fnum(lm.get("rope_scaling_factor"), 1.0) != 1.0
                            or len(_per_type(lm.get("rope_parameters"))) > 1):
            return [hd, hd]
    return None
```
- apply note: in `_rope_schedules`, before the `per` branch:
  - (a) for mt == "olmo3" with a non-default `rope_scaling` and no per-type `rope_parameters`, synthesise `per = {"full_attention": {**rope_scaling}, "sliding_attention": {"rope_type": "default"}}`;
  - (b) add `"rope_local_theta"` to the local-theta key tuple at `:1279`.

  In both cases `_rope_kind` should also see the YaRN entry, which it already does via `rope_scaling`.

## 3. Relation rule
- target side: two distinct tables (YaRN and default), so a host needs either two tables, or one table that holds both target schedules side by side (sum of rotated dims ≤ its cap: 256 > 128 for any 128-dim host). Most single-table hosts are walled for Olmo-3 / PLaMo-3 targets. A host with per-layer-type tables (Gemma-3, OLMo-3 itself, Beam, ...) plays them with frequencies set (rope theta / scaling are free).
- host side: an OLMo-3 / PLaMo-3 host gains a second table (cap (128, 128)): it can now play two-table targets and still plays one-table targets (`multi_rope_tables_equal`).
- kind: exact (the rot gate, rope_fits).
- cost: none (tables, not width).
- free by the config rules: theta and YaRN scaling are free. The per-type table count is code (not free).
- theorem: `multi_rope_tables_equal` (host side). The target-side wall is the OPEN "multi_rope target" row of PROOFS.md (truncation measured not payable).

## 4. Test fixtures
- flag: each -> [128, 128]
  - allenai/Olmo-3-1025-7B
  - allenai/Olmo-3-1125-32B
  - allenai/Olmo-3-7B-Think
  - allenai/Olmo-3-7B-Instruct
  - allenai/Olmo-3.1-32B-Instruct
  - moltherm1/olmo-mirror-olmo-3-1025-7b
  - pfnet/plamo-3-0.8b-fin-instruct (rope_local_theta 1e4 vs 1e6, no rope_parameters)
  - pfnet/plamo-3-translate-2b (already two tables in p5; the code fact still holds)
- not flag:
  - prithivMLmods/Olmo-3-Think-AIO-GGUF (olmo3 with an empty config: no rope_scaling)
  - google/gemma-3-1b-it (two tables, but already read through rope_local_base_freq)
  - Qwen/Qwen3-0.6B

## Verdict
Real.
- 20 Olmo-3 targets and 1 PLaMo-3 target lose their single-table hosts: fewer edges, on the order of the hosts these points reach today.
- As hosts they gain two-table targets (Gemma-3-like and Laguna-like points).
