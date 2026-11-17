# config_default_unread
live 11 / placed 11 (llada2_moe 5, exaone_moe 3, muse_glimmer 1, muse_glimmer_text 2); direction both; batch a2_dims

## 1. What the code computes
- LLaDA2-MoE q / k norm: `inclusionAI/LLaDA2.0-mini/configuration_llada2_moe.py:19,57` default `use_qk_norm=True`
  (same lines in LLaDA2.1-mini and LLaDA2.1-flash); `inclusionAI/LLaDA2.0-mini/modeling_llada2_moe.py:448,496`
  `if self.config.use_qk_norm:` builds / applies `query_layernorm` / `key_layernorm` (RMSNorm over head_dim, before
  the partial RoPE); `yifanyu/LLaDA2.1-mini-ORL/modeling_llada2_moe.py:676,724` same. LLaDA2.0-mini-preview applies it
  unconditionally: `inclusionAI/LLaDA2.0-mini-preview/modeling_llada2_moe.py:396,431`. LLaDA2.0-mini / 2.1-mini /
  2.1-flash / 2.0-mini-preview omit the key; LLaDA2.2-mini and 2.1-mini-ORL save `use_qk_norm: true`.
- Muse-Glimmer final cap: `tf5:models/muse_glimmer/configuration_muse_glimmer.py:141` `final_logit_softcapping: float
  = 20.0`; `tf5:models/muse_glimmer/modeling_muse_glimmer.py:1188-1193` `logits = T * tanh(logits * mult / T)`
  unconditionally (output_multiplier folds into lm_head). RadixArk/Muse-Glimmer-q4-MLX omits the key; the unsloth /
  wayfinder text_config snapshots carry 20.0.
- EXAONE-MoE NoPE full layers: `tf5:models/exaone_moe/configuration_exaone_moe.py:102` `sliding_window: int = 4096`;
  `:128-129` only an explicit None zeroes the pattern; `tf5:models/exaone_moe/modeling_exaone_moe.py:150,184-185`
  RoPE only `if self.sliding_window is None or self.is_sliding`, `:202` window = sliding_window on sliding layers.
  K-EXAONE-2.0-750B-A37B omits sliding_window (58 sliding / 20 full in layer_types; its `sliding_windows` list is not
  read by 5.18), so its full layers are NoPE and its sliding layers use window 4096.
- p5 (`p5:model_graph.py`): classify's qknorm reads `d.get("use_qk_norm")` directly (llada2_moe is in
  QKNORM_HEAD_CHECKED but never reaches `_qknorm_variant` without the key) -> qknorm=no; softcap reads
  `d.get("final_logit_softcapping")` (no class default for muse_glimmer) -> softcap=no; `_normalize` derives
  no_rope_layers for exaone_moe only when `g("sliding_window")` is set -> 750B gets rope_sched (128,) without the NoPE
  slots and no nope_layers, and dims eff_window = max_context. Not fixed in p5.

## 2. Classifier change
- reads: use_qk_norm (llada2_moe), final_logit_softcapping (muse_glimmer, muse_glimmer_text), sliding_window
  (exaone_moe), each through the class default when absent
- new: `_CLASS_DEFAULT_FALLBACK` entries; classify's qknorm / softcap reads via `_cfg_value`; `_normalize`'s
  exaone_moe rule with the default window
- detection:
```python
def detect_config_default_unread(cfg, model_type):
    lm = lm_of(cfg)
    out = []
    if model_type == "llada2_moe" and "use_qk_norm" not in lm:
        out.append(["use_qk_norm", True])
    if model_type in ("muse_glimmer", "muse_glimmer_text") and "final_logit_softcapping" not in lm:
        out.append(["final_logit_softcapping", 20.0])
    if (model_type == "exaone_moe" and "sliding_window" not in lm and isinstance(lm.get("layer_types"), list)
            and any("sliding" in str(x) for x in lm["layer_types"])):
        out.append(["sliding_window", 4096])
    return out or None
```
- apply note: (1) `_CLASS_DEFAULT_FALLBACK["use_qk_norm"] = {"llada2_moe": True}` and in classify's qknorm
  expression use `_cfg_value(d, "use_qk_norm", outer)` in place of `d.get("use_qk_norm")` (llada2_moe then reaches
  `_qknorm_variant` -> "head", already in QKNORM_HEAD_CHECKED). (2) `_CLASS_DEFAULT_FALLBACK["final_logit_softcapping"]
  = {"muse_glimmer": 20.0, "muse_glimmer_text": 20.0}` and `final_cap = ... or _cfg_value(d, "final_logit_softcapping",
  outer)`. (3) In `_normalize`, the afmoe / exaone rule's condition `(g("sliding_window") or mt in ("afmoe",
  "cohere2"))` gains `or (mt == "exaone_moe" and "sliding_window" not in d)`, and that branch sets
  `u["sliding_window"] = 4096` (the class default) so dims eff_window and the window slot see it.

## 3. Relation rule
- target side: LLaDA2 x4 become qknorm=head targets (need a head-norm host: fewer in-edges); Muse-q4 becomes a
  softcap=final target (final cap: free on greedy text, `final_softcap_greedy`); the 750B gains NoPE slots in the rot
  gate (its hosts need zero-frequency room) and nope_layers (wall both ways).
- host side: LLaDA2 x4 as head-norm hosts lose plain targets except via the eps lever (payable limit); Muse-q4
  host keeps targets (softcap_free if listed); the 750B host loses rotating targets (nope_layers wall).
- kind: exact (qk-norm head <-> head), limit (eps lever host side), wall (nope_layers both ways)
- cost: none
- free by the config rules: the use_qk_norm value is a norm on/off flag (free only where the host's code reads it:
  QKNORM_SWITCH_TYPES; LLaDA2 code reads it, so llada2_moe could join QKNORM_SWITCH_TYPES); softcap values free;
  sliding_window free (a window >= context); none of the changes is a feature switch.
- theorem: `qknorm_sub_plain_regime` / `epsLever_bound`, `leverAttn_model_bound` (eps lever); `final_softcap_greedy`;
  nope_layers wall OPEN (Walls: nope_layers both)

## 4. Test fixtures
- flag: inclusionAI/LLaDA2.0-mini, LLaDA2.1-mini, LLaDA2.1-flash, LLaDA2.0-mini-preview -> [["use_qk_norm", true]];
  RadixArk/Muse-Glimmer-q4-MLX -> [["final_logit_softcapping", 20.0]]; LGAI-EXAONE/K-EXAONE-2.0-750B-A37B ->
  [["sliding_window", 4096]]
- not flag (live, but they carry the key and p5 reads it): inclusionAI/LLaDA2.2-mini, yifanyu/LLaDA2.1-mini-ORL,
  unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit, hweller1/wayfinder-glimmer-30b, LGAI-EXAONE/K-EXAONE-236B-A23B,
  furiosa-ai/K-EXAONE-236B-A23B-NVFP4A16
- scan: exactly these 6.

## Verdict
real. Both directions, ~6 models: LLaDA2 x4 lose plain hosts and gain head-norm hosts (LLaDA2 is diffusion: gen
already limits its hosts); Muse-q4 merges into its siblings' point; the 750B loses rotating hosts / targets
(fewer edges). Tens of edges.
