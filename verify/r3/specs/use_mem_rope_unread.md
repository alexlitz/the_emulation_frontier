# use_mem_rope_unread
live 2 / placed 6 (zamba2 6); direction both; batch a5_positions

## 1. What the code computes
- `tf5:models/zamba2/configuration_zamba2.py:48,100`: `use_mem_rope: bool = False` (class default).
- `tf5:models/zamba2/modeling_zamba2.py`:
  - `:316-318` (`Zamba2Attention.forward`, modular `modular_zamba2.py:162-164`): `if self.config.use_mem_rope:` apply RoPE to q / k; otherwise no positional encoding at all.
  - `:1146-1151`: the rotary module is built only under use_mem_rope.
  - `:1205-1206`: position embeddings are only computed under use_mem_rope.
- Zamba2's only softmax attention is the shared transformer block. With use_mem_rope false that attention is NoPE, and the Mamba-2 layers carry order.
- Snapshot:
  - Zyphra/Zamba2-2.7B and -2.7B-instruct set `use_mem_rope: false`.
  - Zamba2-1.2B-instruct, -1.2B-Instruct-v2, -7B-Instruct and -7B-Instruct-v2 set true.
- round2-p5:
  - classify gives pos=rope for every zamba2 (rope_theta present, so rope_like; hybrid mixer). Checked: Zamba2-2.7B -> pos rope, rope_sched (80,).
  - `_nope_attention` (p5:model_graph.py:2413-2420) / NOPE_BY_CODE has zamba (v1) but not zamba2, and `use_mem_rope` is not read.
  - Not fixed in p5.

## 2. Classifier change
- reads: model_type zamba2 and `use_mem_rope` (absent = False).
- new: pos slot `nope`, in `_nope_attention`.
- detection:
```python
def detect_use_mem_rope_unread(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "zamba2" and lm.get("use_mem_rope") is not True:
        return "nope"
    return None
```
- apply note: add `or (mt == "zamba2" and d.get("use_mem_rope") is not True)` to `_nope_attention`. classify then sets pos = "nope" (`:2248`), and dims_of rope_sched falls to the NoPE path (pos not rope), so rope_sched is empty / 0.

## 3. Relation rule
- target side: a NoPE target. Any RoPE-family host with zero-frequency room plays it (`nope_sub_rope`; under p5's config rules this needs `zero_freq_room`: a pairs-free or fraction-free host, and zamba2 is in ROPE_PAIRS_FREE_TYPES). ALiBi hosts with alibi_off and learned_abs hosts (table 0) also play it. Zamba2 targets stay walled to non-Zamba hosts by `zamba_shared`, so in practice only Zamba2 hosts play them: a use_mem_rope=true Zamba2 host sets inv_freq = 0 (pairs-free).
- host side: a NoPE host cannot play a RoPE target (no rotation to set). A Zamba2-2.7B host loses the use_mem_rope=true Zamba2 targets.
- kind: exact (target side); wall (host side, NoPE host <- RoPE target).
- cost: none.
- free by the config rules: use_mem_rope is a feature switch (turning RoPE on), not free. inv_freq (the RoPE table) on a RoPE host is free.
- theorem: `nope_sub_rope`, `pairScore_zero_freq` (target side); `rope_theta_cannot_reach_nope` (why a fixed table needs pairs-free).

## 4. Test fixtures
- flag: Zyphra/Zamba2-2.7B -> nope; Zyphra/Zamba2-2.7B-instruct -> nope
- not flag:
  - Zyphra/Zamba2-1.2B-instruct, Zyphra/Zamba2-1.2B-Instruct-v2, Zyphra/Zamba2-7B-Instruct, Zyphra/Zamba2-7B-Instruct-v2 (use_mem_rope true)

## Verdict
Real, both directions, 2 points.
- Zamba2-2.7B (and -instruct) as hosts lose the RoPE Zamba2 targets (1.2B and 7B: up to 4 edges each).
- As targets they keep their Zamba2 hosts (pairs-free RoPE plays NoPE).
- Net: a few fewer edges.
