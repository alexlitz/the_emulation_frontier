# gdn_out_gate_act
live 24 / placed 25 (qwen4_exp_text 24, qwen4_exp 1); direction over-credit; batch a11_recurrent_loops

## 1. What the code computes
- Qwen3-Next gated norm: `Qwen3NextRMSNormGated.activation = "silu"` fixed (tf5:models/qwen3_next/modeling_qwen3_next.py:59,64), output `RMSNorm(o) * w * silu(z)` (:73).
- Qwen4-exp: `Qwen4ExpTextRMSNormGated(..., activation)` stores the caller's activation (tf5:models/qwen4_exp/modeling_qwen4_exp.py:184) and applies `hidden_states * ACT2FN[self.activation](gate)` (:193).
- `Qwen4ExpTextGatedDeltaNet.__init__` builds it with `activation=config.output_gate_type or config.hidden_act` (modeling_qwen4_exp.py:498-499; modular_qwen4_exp.py:322-327).
- Config: `output_gate_type: str | None = None` (tf5:models/qwen4_exp/configuration_qwen4_exp.py:164); validated to {"sigmoid", "silu"} (:193-195). Default layer_types: `linear_attention` except every 4th `indexed_attention` (:174-179).
- Snapshot: all 24 qwen4_exp_text configs set `output_gate_type: "sigmoid"`, hidden_act silu. So o = RMSNorm(core) * w * sigmoid(z), not * silu(z) = z sigmoid(z).
- PeetPedro/qwen4_exp-8b-... (placed, carrier) has no output_gate_type, no hidden_act (class default silu) and no linear_* keys: its gate is silu, not flagged (it is also classified softmax; separate issue).
- p5: RECURRENT_FAMILY maps qwen4_exp / qwen4_exp_text to "gdn" (p5:model_graph.py:2027); nothing reads output_gate_type.

## 2. Classifier change
- reads: model_type qwen4_exp / qwen4_exp_text; output_gate_type, hidden_act; layer_types (has linear_attention).
- new: recurrent family variant `gdn_sig` (feature `rf_gdn_sig`), in `_recurrent_family`; `_rec_dims` GDN branch accepts it.
- detection:
```python
def detect_gdn_out_gate_act(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in ("qwen4_exp", "qwen4_exp_text"):
        return None
    lt = layer_types(lm)
    if lt and "linear_attention" not in lt:
        return None
    if not lt and not lm.get("linear_num_value_heads"):
        return None
    act = str(lm.get("output_gate_type") or lm.get("hidden_act") or "silu").lower()
    return act if act != "silu" else None
```
- apply note: `_recurrent_family` (p5:model_graph.py:2040), after `fam = RECURRENT_FAMILY.get(mt)` and before the
  olmo_hybrid line: `if fam == "gdn" and mt in ("qwen4_exp", "qwen4_exp_text") and str(d.get("output_gate_type") or
  d.get("hidden_act") or "silu").lower() == "sigmoid": return "gdn_sig"`. `_rec_dims` (:2821): `if fam in ("gdn",
  "gdn_negeig", "gdn_sig")` (same dims, RC_GDN). `_feats`' REC_EPS_CFG_TYPES (:3859) unchanged (qwen4_exp not listed).
  Update the family comment block (~:1990) with the variant.

## 3. Relation rule
- target side: a gdn_sig target needs a host whose GDN layers carry rf_gdn_sig (itself, the universal host's gated-delta
  branch if its output gate can be a plain sigmoid -- elephantine/universal.py's norm/gate stage, to be checked).
- host side: free (rf_ default): a gdn_sig host's recurrent layers are zeroed for a target without them (ZERO_EDGES).
- kind: wall between gdn and gdn_sig (both directions): sigmoid(z) vs z sigmoid(z) are different functions of the same
  z = W_z x; no weight choice of W_z makes one the other (silu is unbounded, sigmoid is in (0, 1)).
- cost: none.
- free by the config rules: output_gate_type is an activation choice = NOT free (the activation function is a feature).
- theorem: needs theorem `gated_norm_gate_act_distinct`: for no linear W_z, W_z' is sigmoid(W_z x) = silu(W_z' x) for all x
  (except W = 0 at different constants 1/2 vs 0; the doubled o_proj then repairs it only for the constant gate).

## 4. Test fixtures
- flag: CMSManhattan/JiRackDeltaNet_180b, whittleOrg/Whittle-Next-27B-A3B, Schiltmans/..-Coder-cut240,
  mlx-community/..-OptiQ-2bit, defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16, jakeatx/slimder-..., azampatti/...,
  hstolte/... -> "sigmoid" (8 of the 24; the scan flags exactly the 24 qwen4_exp_text configs, all sigmoid).
- not flag: PeetPedro/qwen4_exp-8b-... (qwen4_exp, no gate key: silu); gdubicki/Qwen3-Coder-Next-NVFP4-GB10,
  huihui-ai/Huihui-Qwen3-Coder-Next-abliterated (qwen3_next: silu fixed); Advisio/Radha-9B (qwen3_5_text);
  amd/Qwen3.6-35B-A3B-w4a16-llmcompressor (qwen3_5_moe_text); allenai/Olmo-Hybrid-7B (GDN, silu).

## Verdict
real. Fewer edges, but ~0 today: the 24 points also carry hyper_conn (wall both ways, shared only with DeepSeek-V4 /
GLM-5-Next, which have no GDN), so no GDN-to-GDN edge they have now is removed. It matters once hyper_conn gets a
rule: then the 24 stop pairing with Qwen3-Next / 3.5 GDN points (~hundreds of pairs). Region: recurrent-other.
