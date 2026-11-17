# fope_positions
live 1 / placed 1 (interns1_pro_text 1); direction over-credit; batch a5_positions

## 1. What the code computes
- `internlm/Intern-S1-Pro/modeling_rope_utils.py:134-141` (and the same block in every rope type, `:189-196,269-276,405-412,497-504,572-579`): `use_fope` if rope_scaling carries `fope_init_factor`, `fope_sep_heads` or `num_inv_freq`. Then `_compute_fope_parameters` (`:583-600`) keeps the first num_inv_freq frequencies, else only `inv_freq > 2π/max_position_embeddings` (theta 5e6, hd 128, mpe 262144: 45 of 64 kept).
- Snapshot config rope_scaling: `{rope_type: default, fope_init_factor: 0.5, fope_sep_head: true, num_inv_freq: null}`. The code tests the plural `fope_sep_heads`, but fope_init_factor alone turns FoPE on.
- `modeling_interns1_pro.py:866-897` (InternS1ProMoeTextFourierEmbedding): per-KV-head `cos_coef` / `sin_coef` buffers (n_kv × F × F: identity + xavier(gain 0.5)), `persistent=True`, so they are loaded from the checkpoint.
- `:934` (forward): `cos = cos(ω t) @ C_h`, `sin = sin(ω t) @ S_h` per KV head. The result is padded with 1 to head_dim/2 and scaled by attention_scaling.
- `:180` (`apply_rotary_pos_emb_sep`): q uses its KV group's cos / sin. The product `q cos + rotate_half(q) sin` with mixed coefficients is not a rotation, and its logit is not a function of i − j alone.
- round2-p5: no fope key is read. classify gives pos=rope, rope_sched (128,) (interns1_pro_text is in ROPE_PAIRS_FREE_TYPES). Not fixed in p5.

## 2. Classifier change
- reads: rope_scaling / rope_parameters keys `fope_init_factor`, `fope_sep_heads`, `num_inv_freq` (any non-null).
- new: FEATURES `fope` ("wall", "free"), in `_feats`. Optionally, dims rope_cap for the host side = 2 × kept frequencies (90).
- detection:
```python
def detect_fope_positions(cfg, model_type):
    lm = lm_of(cfg)
    rs = lm.get("rope_scaling") if isinstance(lm.get("rope_scaling"), dict) else {}
    rp = lm.get("rope_parameters") if isinstance(lm.get("rope_parameters"), dict) else {}
    if any((d.get(k) is not None) for d in (rs, rp) for k in ("fope_init_factor", "fope_sep_heads", "num_inv_freq")):
        return "fope"
    return None
```
- apply note:
  - In `_feats` after the nope_layers line, add `fope` when the detector's test holds, and add the FEATURES entry.
  - In `_rope_schedules`, a fope config's cap is 2 × #kept frequencies. Count `theta^(-2i/hd) > 2π/mpe`, or use num_inv_freq.

## 3. Relation rule
- target side: wall. The learned per-head cos / sin mixing gives logits that are not shift-invariant rotations, so no RoPE host forms them, and NoPE / ALiBi hosts are further off.
- host side: free.
  - Set C_h = S_h = I (buffers = weights): the kept frequencies are plain RoPE, and the host then plays a RoPE target needing ≤ 2 × kept rotated dims (90 here).
  - The dropped pairs get cos = sin = 1, the fixed map q ↦ (I + R)q on q and k alike. That gives (I+R)ᵀ(I+R) = 2I, a constant that folds into W_q.
- kind: wall (target side) / exact (host side).
- cost: none.
- free by the config rules: the fope_* keys are feature switches (not free). max_position_embeddings sets the kept count. It is a length config, not one of the listed free values; treat it as not free.
- theorem: host side `partial_rope_host`, `defaultFreq_match`, `attn_scale_fold`. Needs theorem (target side): `fope_not_rope`, i.e. a FoPE logit with C ≠ I is not a function of (i − j) for some q, k, so `rope_shift_invariant` refutes any RoPE host.

## 4. Test fixtures
- flag: internlm/Intern-S1-Pro -> fope
- not flag:
  - internlm/Intern-S1, internlm/Intern-S1-mini (the parent family: plain RoPE)
  - Qwen/Qwen3-0.6B

## Verdict
Real (over-credit), 1 point.
- Intern-S1-Pro as a target loses every RoPE host (fewer edges: its current host set).
- As a host it is unchanged, except that its rope cap drops from 128 to 90 (a few targets lost).
