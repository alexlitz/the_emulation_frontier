# rope_cap_zero_tail_table
live 20 / placed 20 (lfm3 20); direction over-wall; batch a5_positions

## 1. What the code computes
- Remote LFM3 rotary `junkdogjoe/LFM3-Lobotomized-2B/modeling_lfm3.py:94-111` (`compute_default_rope_parameters`):
  - `rope_angles = int(partial*dim)//2`
  - `inv_freq = theta^(-2i/dim)` for i < rope_angles, followed by `zeros(dim//2 - rope_angles)` (`:110`)
  - The table spans the WHOLE head (dim = head_dim = 128), with a NoPE tail of zero frequencies.
- `:127-146` (`compute_rope_parameters`, non-default types): the frequencies are computed at partial 1.0, truncated to rope_angles, and zero-padded to dim/2 (`:145`). Same layout.
- `:115-124` (forward): `emb = cat(freqs, freqs)`, so cos / sin are head_dim wide and every head dim goes through the (rotate_half) rotation, with frequency 0 on the tail. Every LFM3 / LFM3-RD / LFM3-Omni remote copy in the hub cache has the same `zeros(dim // 2 - rope_angles)` line: LiquidAI/LFM3-1.7B-Base, -MLX-5bit / 6bit, cybertruck32489/LFM3-1.7B-RD(-v2) `modeling_lfm3_rd.py`, and the rest.
- The buffer is the parameter-free inv_freq. Setting it (as NoPE <= RoPE does) gives any frequency on any of the 64 pairs. A zero-tail table is therefore a full-head table: cap = head_dim. This is exactly Gemma-4's "proportional" case.
- round2-p5 `_rope_schedules` `sched()` (p5:model_graph.py:1256-1263) gives `cap = need` except for rope_type "proportional". The LFM3 snapshots carry rope_type "default" + partial 0.5 on full_attention, so rope_cap = (128, 64): the full table's cap is understated as 64. Not fixed in p5.
- Side note:
  - Infatoshi/phonon-refiner-1.7b has no auto_map, so it runs transformers-main `models/lfm3/modeling_lfm3.py:85-109`. Its default init ignores partial_rotary_factor (full theta^(-2i/d) over all 64 pairs; the class default for full layers is rope_type "proportional", `configuration_lfm3.py:85-91`).
  - That repo really rotates 128 dims on its full table while p5 says (128, 64), a different defect. The detector does not flag it; it belongs to a rope-sched fix for tf-main lfm3 with rope_type default.

## 2. Classifier change
- reads: model_type in (lfm3, lfm3_rd, lfm3_omni) whose auto_map routes to the remote modeling_lfm3 / modeling_lfm3_rd; head_dim; the per-type rope_parameters entries present in layer_types.
- new: rope_cap = (head_dim,) per table. In `_rope_schedules`, `sched()` returns `cap = max(head, need)` for these types, as for "proportional". rope_sched (the need) is unchanged.
- detection:
```python
LFM3_TYPES = ("lfm3", "lfm3_rd", "lfm3_omni")

def detect_rope_cap_zero_tail_table(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in LFM3_TYPES or "modeling_lfm3" not in auto_map(cfg):
        return None
    hd = _head_dim(lm)
    lt = layer_types(lm)
    per = _per_type(lm.get("rope_parameters"))
    n = len([k for k in per if k in lt]) or 1
    return [hd] * n
```
- apply note: in `_rope_schedules.sched`, replace `kind == "proportional"` with `kind == "proportional" or zero_tail`, where `zero_tail = mt in ZERO_TAIL_ROPE_TYPES and "modeling_lfm3" in str(d.get("auto_map"))`. Add `ZERO_TAIL_ROPE_TYPES = {"lfm3", "lfm3_rd", "lfm3_omni"}` next to ROT_HOSTS. The auto_map test needs the outer config's auto_map for lfm3_omni (pass it through as `_alibi_dims` does, or test `_lm_unwrap`'s merged dict).

## 3. Relation rule
- target side: unchanged (need stays 64 on the full table, 128 on the sliding one).
- host side: an LFM3 host's full-attention table drives 128 rotated dims. A target schedule of up to 128 dims fits it, including (via rope_fits packing) two 64-dim schedules or a 64-dim schedule plus 64 zero-frequency NoPE slots.
- kind: exact (`partial_rope_host` with the target's pairs on host pairs, frequencies written in the buffer; zero-frequency slots `pairScore_zero_freq`).
- cost: none.
- free by the config rules: the frequencies are the free RoPE table. Partial_rotary_factor needs no change (cap does not depend on it).
- theorem: `partial_rope_host`, `defaultFreq_match`, `pairScore_zero_freq` (PROOFS.md "partial-RoPE host ⊇ full-RoPE target", "NoPE ⊆ RoPE").

## 4. Test fixtures
- flag: each -> [128, 128]
  - AJ-Gonzalez/coder-lfm3-1.7b
  - DQN-Labs/17btest-nov6-2x16k
  - LiquidAI/LFM3-1.7B-Base
  - LiquidAI/LFM3-1.7B-MLX-5bit
  - mohommed/LFM3-1.7B-Frontier-SmartAgent (no partial factor: need 128 already, cap unchanged)
  - junkdogjoe/LFM3-Lobotomized-2B (lfm3_omni wrapper)
  - cybertruck32489/LFM3-1.7B-RD (lfm3_rd)
  - mlx-community/LFM3-1.7B-OptiQ-4bit
- not flag:
  - Infatoshi/phonon-refiner-1.7b (no auto_map: transformers-main lfm3, see the side note)
  - LiquidAI/LFM2-1.2B (the parent family, conv + full RoPE)
  - google/gemma-3-1b-it

## Verdict
Real (over-wall).
- About 25 LFM3 host points gain the targets whose full-layer schedule needs 65-128 rotated dims, or 64 + NoPE slots. More edges: the two no-partial LFM3 snapshots, full-RoPE 128-dim targets on a mixed host, and so on. Bounded by LFM3's other walls (post_norm, shortconv): a few hundred host-target pairs at most.
- The phonon-refiner side note is a separate (over-credit) defect.
