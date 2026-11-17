# vha_head_mix
live 1 / placed 1 (ernie5_5_moe 1); direction over-credit; batch a7_attn_exotic

## 1. What the code computes
- ext:ernie5_5_moe/gh__baidu__vLLM-ERNIE/attention.py:85 `VHA_RANK = 16`; :110-146 VHAPostMix: U, V of shape
  (num_heads, 16), `m = I + V U^T` (fold, :129-135), forward `o[t, h] <- sum_g m[h, g] o[t, g]` (:137-146): a FIXED,
  token-independent linear mix across the head axis, the same for every head_dim channel.
- Built unconditionally in both attention kinds: HCA / CSA layers :195 (`self.vha = VHAPostMix(self.n_heads)`), used
  at :229 before the grouped low-rank `wo_a` (n_groups = o_groups 8, block-diagonal over head groups) and the sigmoid
  gate `wo_b(z * sigmoid(gate))` (:230-243); DSA layers :831, used at :903-904 before `wo_b(o * sigmoid(gate))`.
  sglang/ernie55.py:184-217 the same (vha_postmix_U / _V). Neither reads use_vha_attention (true in the snapshot);
  the weights come from the checkpoint.
- Why it does not fold: in the DSA layer an elementwise data-dependent gate sigmoid(gate) sits between the mix and
  wo_b; in the HCA layer the grouped wo_a is block-diagonal over head groups and (m (x) I) mixes heads ACROSS groups,
  so wo_a (m (x) I) is no longer grouped (rank o_lora_rank per group). The mix cannot move before the softmax either
  (each head has its own attention weights). Host side: U = V = 0 gives m = I exactly.

## 2. Classifier change
- reads: model_type ernie5_5_moe (code path unconditional); use_vha_attention
- new: FEATURES key `vha` ("wall", "free"), in _feats + FEATURES
- detection:
```python
def detect_vha_head_mix(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "ernie5_5_moe" and lm.get("use_vha_attention", True) is not False:
        return True
    return None
```
- apply note: p5:model_graph.py `_feats` (after the out_gate block, ~1756): `if mt == "ernie5_5_moe" and
  d.get("use_vha_attention", True) is not False: f.add("vha")`; FEATURES: add "vha" after "out_gate".

## 3. Relation rule
- target side: a VHA target needs a VHA host (or a host whose o-path has no gate / grouping between attention and
  the output projection, where m folds into o_proj: none such carries ERNIE-5.5's other features) -- wall.
- host side: a VHA host sets U = V = 0 (m = I) and plays any non-VHA target -- free.
- kind: wall (target side); exact (host side)
- cost: none
- free by the config rules: no config value; the mix is learned weights
- theorem: needs theorem: vha_off (U = V = 0 gives m = I, exact) ; target side OPEN (same status as out_gate:
  a fixed head mix behind a data-dependent gate is not a weight of the ungated-mix host)

## 4. Test fixtures
- flag: baidu/ERNIE-5.5-Flash-Base -> True
- not flag: baidu/ERNIE-4.5-21B-A3B-PT (ernie4_5_moe, the parent family, no VHA), deepseek-ai/DeepSeek-V4-Flash
  (grouped wo_a + sinks, no head mix), deepseek-ai/DeepSeek-V3 (MLA, no mix)

## Verdict
real but no frontier change today: ERNIE-5.5 already carries hyper_conn (wall both ways) and kv_compress (wall both
ways), so it has no edge for vha to remove. Recorded so the target side stays walled if those walls are lifted.
