# partial_rope_keys_unread
live 5 / placed 20 (grug_moe 7, k2_horizon 10, stablelm_epoch 1, step3p5 1, zaya1_vl 1); direction both; batch a5_positions

## 1. What the code computes
- Step-3.5 `stepfun-ai/Step-3.5-Flash/modeling_step3p5.py`:
  - `:56-59`: per layer, `config.partial_rotary_factor = partial_rotary_factors[layer_idx]` (0.5 on full layers, 1.0 on sliding).
  - `:67`: `config.rope_theta = rope_theta[layer_idx]` (5e6 full / 1e4 sliding).
  - `:125`: `apply_rotary_pos_emb` rotates the first `cos.shape[-1]` dims.
  - Two tables: 64 rotated dims (full layers) and 128 (sliding).
- K2-Horizon / Jais-3 `IFM/K2-Horizon-7B/modeling_k2_horizon.py:210,267-277`: `rope_head_dim` of head_dim rotates; the rest is split off and passes. The 375B-A23B snapshots have 64 of 128. Every other k2_horizon snapshot has rope_head_dim == head_dim (no partial).
- StableLM-Epoch `acon96/Home-3B-v3-GGUF/modeling_stablelm_epoch.py:209,235-237`: `rotary_ndims = int(head_dim * config.rope_pct)`, so rope_pct 0.25 rotates 20 of head_dim 80.
- ZAYA1-VL `ext:zaya1_vl/.../configuration_zaya1_vl.py:99,157,170`: partial_rotary_factor defaults to 0.5 and is folded into rope_parameters. The snapshot omits it (its `rope_pct` 0.5 is stored but unread). 64 of 128 rotate.
- Grug-MoE `ext:grug_moe/.../skyrl_train_models_grug_moe.py:680-700` (`_apply_half_rope`): `rotary_dim = head_dim // 2`, inv_freq = theta^(-i/32), applied only on the sliding layers (`:730-731`). 64 of 128 rotate. The long layers are NoPE: see nope_full_layers_unlisted.
- round2-p5 `_rotary_factor` (p5:model_graph.py:2387-2407) reads only rotary_pct / partial_rotary_factor (top level, nested, class default). `_rope_schedules` reads qk_rope_head_dim / rotary_dim. None of these keys or the grug constant is read, so all five give pos=rope and rope_sched (head_dim,) (verified with p5:model_graph.py classify / dims_of). Not fixed in p5.

## 2. Classifier change
- reads: `partial_rotary_factors` + list `rope_theta` (step3p5); `rope_head_dim` (k2_horizon); `rope_pct` (stablelm_epoch); class default 0.5 (zaya1_vl); type grug_moe.
- new: pos = partial_rope and rope_sched / rope_cap from the returned list, in `_rotary_factor` (scalar cases) and `_rope_schedules` (Step-3.5 per-layer tables).
- detection:
```python
def detect_partial_rope_keys_unread(cfg, model_type):
    lm = lm_of(cfg)
    hd = _head_dim(lm)
    if not hd:
        return None
    if model_type == "step3p5" and isinstance(lm.get("partial_rotary_factors"), list):
        prf = lm["partial_rotary_factors"]
        th = lm.get("rope_theta")
        th = th if isinstance(th, list) and len(th) == len(prf) else [th] * len(prf)
        tables = {}
        for f, t in zip(prf, th):
            tables[(f, t)] = int(round(float(f) * hd))
        sched = sorted(tables.values(), reverse=True)
        return sched if any(r < hd for r in sched) else None
    if model_type == "k2_horizon":
        r = num(lm.get("rope_head_dim"))
        return [r] if 0 < r < hd else None
    if model_type == "stablelm_epoch" and fnum(lm.get("rope_pct")) is not None and "rotary_pct" not in lm:
        r = int(hd * float(lm["rope_pct"]))
        return [r] if r < hd else None
    if model_type == "zaya1_vl" and lm.get("partial_rotary_factor") is None:
        rp = rope_params(lm)
        f = fnum(rp.get("partial_rotary_factor"), 0.5)
        r = int(round(f * hd))
        return [r] if r < hd else None
    if model_type == "grug_moe":
        return [hd // 2]
    return None
```
- apply note:
  - `_rotary_factor`: read `rope_pct` as an alias of `rotary_pct` for stablelm_epoch. Return `rope_head_dim / head_dim` for k2_horizon.
  - `_CLASS_DEFAULT_FALLBACK["partial_rotary_factor"]`: add `zaya1_vl: 0.5, grug_moe: 0.5`.
  - `_rope_schedules`: when `partial_rotary_factors` is a list, emit one schedule per distinct (factor, theta), need = factor × hd.
  - classify's `part` test then gives partial_rope.
  - rotation layout: the k2_horizon permutation (rotate pairs (i, i+rope_head_dim/2) after a split / interleave) acts on q and k alike, so it is a column permutation and does not change the rule.

## 3. Relation rule
- target side: these targets need fewer rotated dims than credited (64 / 20 rather than 128 / 80), so more hosts fit them through rope_fits, `partial_rope_host`. Step-3.5 gets two tables (128 + 64), which a one-table host holds only with cap ≥ 192.
- host side: these hosts' tables drive fewer dims (cap 64 / 20; Step-3.5 128 and 64). A fixed partial host is walled from targets needing more, by `partial_rope_too_few`.
- kind: exact (rot gate) both ways.
- cost: none.
- free by the config rules: rope_head_dim and rope_pct are the rotary fraction, free only on families whose code splits by a config fraction (ROT_FRAC_FREE_TYPES). None of these five is in that set, and grug's half is code-fixed.
- theorem: `partial_rope_host`, `defaultFreq_match`, `partial_rope_host_config` (target side); `partial_rope_too_few` (host side wall); `multi_rope_tables_equal` (Step-3.5 host playing a one-table target).

## 4. Test fixtures
- flag:
  - IFM/K2-Horizon-375B-A23B -> [64]
  - inception42/Jais-3-375B-A23B -> [64]
  - Zyphra/ZAYA1-VL-8B -> [64]
  - acon96/Home-3B-v3-GGUF -> [20]
  - stepfun-ai/Step-3.5-Flash -> [128, 64]
  - stepfun-ai/Step-3.5-Flash-Base-Midtrain -> [128, 64] (live, not in placed)
  - open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 -> [64]
- not flag:
  - IFM/K2-Horizon-7B, IFM/Jais-ASR (rope_head_dim == head_dim)
  - IFM/K2-Type-0.9B, inception42/Jais-3-0.9B (64 == head_dim 64)
  - stabilityai/stablelm-3b-4e1t, stabilityai/stablelm-2-1_6b (native stablelm: partial_rotary_factor, already read)

## Verdict
Real, both directions.
- About 12 points: 2 K2 375B, 2 Step-3.5, 1 Home-3B, 1 ZAYA1-VL, 7 Grug, though Grug is mostly walled by nope_layers after its own fix.
- As targets they gain hosts with fewer rotated dims: more edges, tens of hosts each.
- As hosts they lose targets needing more than 64 / 20 dims: fewer edges.
- Step-3.5 as a target loses one-table hosts.
