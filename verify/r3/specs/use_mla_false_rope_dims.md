# use_mla_false_rope_dims
live 1 / placed 1 (deepseek_v2 1); direction both; batch a5_positions

## 1. What the code computes
- `Isotr0py/deepseek-vl2-tiny/modeling_deepseek.py:1222-1223,1232`: the attention class is `ATTENTION_CLASSES["mla_*" if config.use_mla else "mha_*"]`, where `mha_eager` = transformers' `LlamaAttention` (imported `:38-39`).
- The repo's language_config: use_mla false, kv_lora_rank null, qk_rope_head_dim 0, qk_nope_head_dim 0, hidden 1280, 10 heads. So the layer is plain MHA with head_dim = 1280/10 = 128 and LlamaRotaryEmbedding over the full head: 64 pairs, every dim rotating. The MLA forward (hunk 0a20c905378b) is never built for this config.
- round2-p5:
  - classify correctly gives kv=MHA (no kv_lora_rank).
  - `_rope_schedules` (p5:model_graph.py:1254-1257) takes `need = qk_rope_head_dim` whenever the key is not None, so rope_sched = rope_cap = (0,) with pos=rope (checked).
  - As a target, "nothing rotates" fits any host (rope_fits returns True for an all-zero target); as a host it has no rotary capacity (a wall to every RoPE target).
  - Not fixed in p5.

## 2. Classifier change
- reads: model_type deepseek_v2 with `use_mla` false and no `kv_lora_rank`; hidden_size / num_attention_heads.
- new: rope_sched / rope_cap = (hidden // heads,), in `_rope_schedules` (ignore qk_rope_head_dim when the MLA path is off).
- detection:
```python
def detect_use_mla_false_rope_dims(cfg, model_type):
    lm = lm_of(cfg)
    if model_type != "deepseek_v2" or lm.get("use_mla") is not False or lm.get("kv_lora_rank"):
        return None
    nh = num(lm.get("num_attention_heads"))
    hd = num(lm.get("hidden_size")) // nh if nh else 0
    return [hd] if hd else None
```
- apply note:
  - In `_rope_schedules`, set `qr = d.get("qk_rope_head_dim") if d.get("kv_lora_rank") or d.get("use_mla", True) is not False else None`. The `sched` then falls back to f × head with f = 1.0.
  - The same guard belongs in classify's `qr` test (`:2238`) and in dims_of's `mla_rope` (already 0 without kv_lora_rank).

## 3. Relation rule
- target side: a full 128-dim RoPE target. Hosts need ≥ 128 rotated dims on one table (rope_fits). It no longer fits every host.
- host side: a full-RoPE 128-dim MHA host. It plays RoPE targets up to 128 rotated dims (pairs-free: deepseek_v2 is in ROPE_PAIRS_FREE_TYPES), and NoPE targets by inv_freq = 0.
- kind: exact (rot gate) both ways.
- cost: none.
- free by the config rules: theta / scaling are free; the MLA switch use_mla is a feature switch (not free).
- theorem: `partial_rope_too_few` (the target-side walls it now gets); `nope_sub_rope`, `partial_rope_host` (host side).

## 4. Test fixtures
- flag: Isotr0py/deepseek-vl2-tiny -> [128]
- not flag:
  - deepseek-ai/DeepSeek-V2-Lite, deepseek-ai/DeepSeek-V2, deepseek-ai/DeepSeek-V2.5 (real MLA, kv_lora_rank 512: qk_rope_head_dim is the shared rotary key)

## Verdict
Real, both directions, 1 point (a VL test checkpoint).
- As a target it loses hosts with < 128 rotated dims: fewer edges.
- As a host it gains every RoPE target it fits by size: more edges, tens.
- Net: more edges.
