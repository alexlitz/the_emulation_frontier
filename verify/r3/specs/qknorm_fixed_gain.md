# qknorm_fixed_gain
live 15 / placed 15 (nanochat_gpt 7, grug_moe 7, gaggle 1); direction over-credit; batch a1_typetables

## 1. What the code computes
A weight-free per-head RMSNorm on q and k, then a constant multiplier read from the config, so the logit is `g * rms(q).rms(k) / sqrt(d)` with g != 1:
- nanochat_gpt: jkminder/d16_optimized_537m_seed5/modeling_nanochat_gpt.py:104-107 `F.rms_norm(x)` (no weight), :227 `q, k = rms_norm(q), rms_norm(k)` after RoPE, :228-230 `q *= qk_sharpen_scale; k *= qk_sharpen_scale`; config.json:41 `qk_sharpen_scale: 1.2` (all 7) -> g = 1.44
- gaggle: castorini/gaggle-.../modeling_gaggle.py:36 weight-free rms_norm, :83 `qk_scale = config.qk_scale`, :105 `q, k = q * qk_scale, k * qk_scale`; config.json:27 `qk_scale: 1.2` -> g = 1.44
- grug_moe: ext grug_moe/gh__marin-community__marin/skyrl_train_models_grug_moe.py:396 grug_rms_norm_no_weight, :728-729 on q and k, :732-733 `q *= qk_mult * (qk_mult_long_scale if is_long else 1)`; config class default qk_mult 1.5703 (:186), qk_mult_long_scale 1.0 (:187); snapshots `qk_mult: 1.75` -> g = 1.75 on every layer (long scale 1 or absent)
- same code fact, outside the carrier list: muse_glimmer(_text) tf5:models/muse_glimmer/modeling_muse_glimmer.py:322, :340 `q = rms(q) * qk_scale_factor` (default 3.87, configuration_muse_glimmer.py:144); sn38-nanochrono ext sn38-nanochrono/github__chronollm__sn38/modeling_nanochrono.py:29, :51-53 (`attention_multiplier` 1.2 on q and k, hub config :6)
- p5 maps nanochat_gpt and gaggle to qknorm "head_free" (QKNORM_VARIANT_BY_TYPE, p5:model_graph.py:1881-1882), i.e. gain 1: equal to llama4 / nanochat / arloop (gain 1). grug_moe is "no" in p5 (qknorm_by_code_unlisted).

## 2. Classifier change
- reads: model_type; qk_sharpen_scale / qk_scale / attention_multiplier (applied to q and k: g = s^2), qk_mult / qk_scale_factor (q only: g = s), with code defaults
- new: qknorm slot value "head_free_gain" (new QKNORM_VARIANTS entry) and a table `QK_GAIN_KEYS` in classify; the gain value need not be a dims field (it is a free config value on every head_free_gain host, see 3)
- detection:
```python
QK_GAIN_KEYS = {"nanochat_gpt": ("qk_sharpen_scale", None, True), "gaggle": ("qk_scale", None, True),
                "sn38-nanochrono": ("attention_multiplier", None, True),
                "grug_moe": ("qk_mult", 1.5703274004183786, False),
                "muse_glimmer": ("qk_scale_factor", 3.87, False), "muse_glimmer_text": ("qk_scale_factor", 3.87, False)}

def qk_gain(cfg, model_type):
    if model_type not in QK_GAIN_KEYS:
        return None
    key, default, both = QK_GAIN_KEYS[model_type]
    s = fnum(lm_of(cfg).get(key), default)
    if s is None:
        return None
    g = s * s if both else s
    return round(g, 4) if abs(g - 1.0) > 1e-9 else None

def detect_qknorm_fixed_gain(cfg, model_type):
    return "head_free_gain" if qk_gain(cfg, model_type) else None
```
- apply note: in p5:model_graph.py add "head_free_gain" to QKNORM_VARIANTS (:224); in `_qknorm_variant` (:1903) return "head_free_gain" when `qk_gain(d, mt)` is set (before the QKNORM_VARIANT_BY_TYPE lookup, so nanochat_gpt / gaggle with qk_sharpen_scale = null stay "head_free"); grug_moe / muse_glimmer come in through qknorm_by_code_unlisted.

## 3. Relation rule
- target side: a head_free_gain target needs a host that can put a constant c on the normalised logit: a gained per-head norm (head, head_post, head_distinct: w = c 1 commutes with RoPE), ngpt (sqk constant), or another head_free_gain host (sets its gain key to c). A gain-1 head_free host (llama4, nanochat, arloop) cannot -- unless its softmax scale is itself a config value (none of the current head_free types has one).
- host side: a head_free_gain host plays a head_free target by setting its gain key to 1 (free: a scale), and another head_free_gain target by setting c.
- QKNORM_NEST: `head_free_gain: ("head", "head_post", "head_distinct", "head_free_gain", "ngpt")`, `head_free: + ("head_free_gain",)`.
- kind: exact (nestings above); wall (head_free host for a head_free_gain target); grug_moe with qk_mult_long_scale != 1 would need per-layer gains (no snapshot has it)
- cost: none
- free by the config rules: the gain value (qk_sharpen_scale / qk_scale / qk_mult / qk_scale_factor / attention_multiplier) is a scale: free. p5's levers come from family_code.json (transformers code only); these keys are read in remote code I checked above, so list them in a `QK_GAIN_FREE_TYPES` table for the host side.
- theorem: `qkn_gain_one`, `weightFree_comm`, `rmsNorm_one_rot_comm` (PROOFS.md "qk-norm: head_free <- head / head_post / head_distinct"); needs theorem: `qkn_const_gain` (rms(q) * c = RMSNorm_w(q) with w = c 1, and c commutes with RoPE), and that a gain-1 weight-free host with a fixed 1/sqrt(d) scale cannot give logit c * (rms(q).rms(k))/sqrt(d) for c != 1 (`qkn_gain_wall`: the logit range is bounded by 1/sqrt(d) * d)

## 4. Test fixtures
- flag (every live repo): castorini/gaggle-... ; jkminder/d12_optimized_286m_seed2, _seed3; d16_optimized_537m_seed3, _seed4, _seed5; d20_optimized_897m_seed3, _seed8; open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21, -dr-doom, -2026.10.05, Grug-67B-A2B-GLM53-RLVR-SFT-2026.09.23 (and the other grug_moe in the scan) -> "head_free_gain"
- not flag: l0ulan/chinese-babylm-nanochat-d22-step10000 (transformers nanochat: weight-free, no gain, tf5:models/nanochat/modeling_nanochat.py:214-215, :236-237); lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-8bit (L2Norm, gain 1); Diamegs/PIT-1B-201912 (nanochat port without a gain); Qwen/Qwen3-0.6B
- scan: 19 = the 15 live + muse_glimmer 3 + sn38-nanochrono 1 (same code fact).

## Verdict
real. Fewer edges: the 7 nanochat_gpt + gaggle targets lose the gain-1 head_free hosts (llama4_text 1, nanochat 1, arloop*), and those hosts lose them; grug / muse / sn38 targets get the same rule. The nanochat_gpt <-> gaggle edges stay (both set their gain). Roughly 10-20 edges removed; edges to head hosts are unchanged.
