# rotary_dim_ignored
live 2 / placed 9 (bailing_moe 7, llada2_moe 1, param2moe 1); direction both; batch a5_positions

## 1. What the code computes
- Ling-1T / Ring-1T run `inclusionAI/Ling-1T/modeling_bailing_moe_v2.py`.
  - `:459-460`: `partial_rotary_factor = config.partial_rotary_factor if hasattr(config, ...) else 1.0`, and `self.rope_dim = int(head_dim * partial_rotary_factor)`.
  - `:204-206`: the rotary table comes from `ROPE_INIT_FUNCTIONS[rope_type](config)`.
  - `:257-259`: `apply_rotary_pos_emb` rotates the first `cos.shape[-1]` dims.
- transformers' default init (`tf4:modeling_rope_utils.py:124-126`): `partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)`, `head_dim = getattr(config, "head_dim")`, `dim = int(head_dim * partial_rotary_factor)`. It never reads `rotary_dim`.
- `BailingMoeV2Config` has no partial_rotary_factor default, and the Ling-1T / Ring-1T configs set `rotary_dim: 64` with no partial_rotary_factor. So the code rotates the WHOLE 128-dim head: 64 pairs, theta^(-2i/128).
- round2-p5:
  - `_rope_schedules` (p5:model_graph.py:1256-1260) takes `need = rotary_dim` whenever it is set.
  - classify (`:2241`) sets pos = partial_rope when `d.get("rotary_dim")`.
  - Result: partial_rope, rope_sched (64,). Not fixed in p5.
- The other placed carriers are consistent, so their config is not on the defect:
  - Ling-flash / mini-2.0, Ring-flash / mini (-sparse): partial 0.5, no rotary_dim.
  - LLaDA2.0-mini-preview: rotary_dim 64 and partial 0.5, both 64.
  - Param2: partial 1.0.

## 2. Classifier change
- reads: model_type bailing_moe, auto_map to bailing_moe_v2, `rotary_dim`, absence of `partial_rotary_factor` (top level and nested).
- new: rope_sched / rope_cap = (head_dim,) and pos = rope, in `_rope_schedules.sched` and classify's `part` test.
- detection:
```python
def detect_rotary_dim_ignored(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "bailing_moe" or "bailing_moe_v2" not in auto_map(cfg):
        return None
    hd = _head_dim(lm)
    rd = num(lm.get("rotary_dim"))
    if rd and rd < hd and lm.get("partial_rotary_factor") is None and rope_params(lm).get("partial_rotary_factor") is None:
        return [hd]
    return None
```
- apply note: add a set `ROTARY_DIM_UNREAD_TYPES = {"bailing_moe"}` (code that builds its table through ROPE_INIT_FUNCTIONS and reads only partial_rotary_factor). For those types, `_rope_schedules.sched` ignores `rotary_dim`: need = round(f × head). classify's `pos = "partial_rope" if part or d.get("rotary_dim")` skips the `rotary_dim` clause for them. `_rotary_factor`'s "explicit rotary_dim" early return (`:2397`) must not fire for them either. llada2_moe reads only partial_rotary_factor too (`LLaDA2.0-mini-preview/modeling_llada2_moe.py`, hunk 98cd38c8dac1) and can join the set; its configs agree, so it makes no difference.

## 3. Relation rule
- target side: Ling-1T / Ring-1T need 128 rotated dims, not 64. Hosts with a 64-dim table are walled (`partial_rope_too_few`); full-RoPE 128-dim hosts are now direct (rope <= rope).
- host side: their table drives 128 dims (cap 128). bailing_moe is in ROPE_PAIRS_FREE_TYPES, so the frequencies are buffer-free. They can now play 128-dim RoPE targets they were walled from.
- kind: exact (rot gate) both ways.
- cost: none.
- free by the config rules: partial_rotary_factor would be free on this family, since the code reads it. But bailing_moe is not in ROT_FRAC_FREE_TYPES (no q_rot / q_pass split by the factor in modeling code), so no lever changes.
- theorem: `partial_rope_host`, `defaultFreq_match` (host side); `partial_rope_too_few` (the walls that move).

## 4. Test fixtures
- flag: inclusionAI/Ling-1T -> [128], inclusionAI/Ring-1T -> [128]
- not flag:
  - inclusionAI/Ling-flash-2.0, Ling-mini-2.0, Ring-flash-2.0 (partial 0.5, no rotary_dim)
  - inclusionAI/Ling-lite (V1 code, no rotary_dim)
  - inclusionAI/LLaDA2.0-mini-preview (rotary_dim agrees with partial 0.5)
  - bharatgenai/Param2-17B-A2.4B-Thinking (partial 1.0)

## Verdict
Real, both directions, on 2 points (1T MoE).
- As targets they lose 64-dim partial hosts and gain 128-dim full-RoPE hosts. Few in practice: the 1T size already limits their hosts.
- As hosts they gain 128-dim RoPE targets: more edges, potentially many, since a 1T host is wide.
- Net: more edges.
