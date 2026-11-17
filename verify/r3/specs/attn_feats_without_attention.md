# attn_feats_without_attention
live 1 / placed 1 (qwen3_5_gdn24 1); direction over-wall; batch a3_placement

## 1. What the code computes
- summerAIO/Qwen3.8-27B-Longbook27-v2/configuration_qwen35_gdn24.py:9-10, :47: `self.layer_types =
  ["linear_attention"] * num_hidden_layers` (the source model's full-attention layers converted to GDN;
  source_layer_types keeps the old topology for reference only, :20-39).
- modeling_qwen35_gdn24.py:121: "No rotary module: all 24 mixers are recurrent" -- the model builds no rotary
  embedding and no softmax attention, so the inherited Qwen3.5 attention keys (attn_output_gate true,
  rope_parameters partial_rotary_factor 0.25) are never applied.
- Same code fact (not in this finding's carriers): summerAIO/Cyber-9B-Qwen-AllGDN-Engram (allgdn, layer_types all
  linear_attention, attn_output_gate true, partial_rotary_factor 0.25).
- p5 classify (p5:model_graph.py :2252-2255): for mixer ssm / linear kv = "none" but pos stays "partial_rope" (only a
  None pos becomes "nope"); `_feats` (:1749-1755) adds out_gate from attn_output_gate whatever the mixer. p5 output for
  both: mixer ssm, pos partial_rope, feats out_gate + rf_gdn. out_gate is a target-side wall.

## 2. Classifier change
- reads: the layer schedule (layer_types / layers_block_type all recurrent) + attention-only keys
- new: for mixer in (ssm, linear): pos = "nope", qknorm = "no", softcap attn -> no, window = "full", and feats minus
  ATTN_ONLY_FEATS = {out_gate, nope_layers, dsa, kv_compress, clip_qkv, vnorm}; in classify
- detection:
```python
def detect_attn_feats_without_attention(cfg, model_type):
    lm = lm_of(cfg)
    lt = [str(x).lower() for x in (lm.get("layer_types") or lm.get("layers_block_type") or [])
          if not isinstance(x, (dict, list))]
    if not lt:
        return None
    rec = [x for x in lt if any(m in x for m in _REC_MARKS)]
    rest = [x for x in lt if x not in rec and x not in ("mlp", "moe", "ffn", "-", "e")]
    if not rec or rest:
        return None
    if any(truthy_flag(lm.get(k)) for k in _ATTN_ONLY_KEYS):
        return "no_attention"
    return None
```
  (_REC_MARKS = linear, mamba, ssm, gdn, delta, kda, rwkv, conv, recurrent, rglru, lightning; _ATTN_ONLY_KEYS =
  attn_output_gate, attention_output_gate, use_output_gate, partial_rotary_factor, rope_parameters, rope_theta,
  rope_scaling, use_qk_norm, qk_norm, qk_layernorm.)
- apply note: p5:model_graph.py `classify` (:2252): replace `if pos is None: pos = "nope"` by `pos = "nope"` and set
  qknorm = "no", window = "full", softcap = "final" if softcap == "attn" and a final cap else ("no" if "attn"); at the
  return, `feats = (_feats(...) | _rf_feats(...)) - (ATTN_ONLY_FEATS if mixer in ("ssm", "linear") else set())`.
  In the snapshot only these two configs change (every other pure-SSM point already has pos nope and no attention
  feature).

## 3. Relation rule
- target side: a pure-GDN target with no out_gate wall and pos nope: any GDN-family host with the rc dims plays it
- host side: unchanged (out_gate host side was free)
- kind: exact (removes a spurious wall)
- cost: none
- free by the config rules: n/a
- theorem: none needed (the gate is absent from the forward); recurrent rules `deltaRow_pad`, `gdn_dk_needed`

## 4. Test fixtures
- flag: summerAIO/Qwen3.8-27B-Longbook27-v2 -> no_attention; summerAIO/Cyber-9B-Qwen-AllGDN-Engram -> no_attention
- not flag: whittleOrg/Qwen3.8-Whittle-16B, autotrust/JEV-9B (Qwen3.5 hybrids with full_attention layers: the gate
  is real), LiquidAI/LFM2-350M (conv + attention), state-spaces/mamba-130m-hf (pure SSM, no attention keys),
  nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (hybrid with attention)

## Verdict
real: 2 points lose a spurious out_gate target wall: edges INTO them from ungated GDN-family hosts appear (more
edges, 2 targets). Region: recurrent-other / attention.
