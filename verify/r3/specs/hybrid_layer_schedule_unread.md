# hybrid_layer_schedule_unread
live 5 / placed 9 (bailing_hybrid 4, <none> 1, gptx3 1, klm 1, ming_v3_asr 1, solar_open2 1); direction over-credit; batch a6_masks

The 4 bailing_hybrid Ling snapshots are placed but not live: model_type bailing_hybrid already gets `_normalize`'s
layer_group_size schedule and rf_kda_lb. The detector also flags upstage/Solar-Open2-250B, which has the same keys and
code as the live nota-ai quant.

## 1. What the code computes
Each family runs recurrent (linear / delta) mixers on most layers and softmax on a schedule p5 does not read, so
classify gives mixer=softmax with every layer FU and no rf_ family.
- gptx3 (AxiomicLabs/GPT-X3-150M): `is_full_attention_layer(i) = (i + 1) % (gdn_layers_per_gqa + 1) == 0`
  (configuration_gptx3.py:48-49). GPTX3GDN2 elsewhere (modeling_gptx3.py, GPTX3Block.__init__): per-channel decay
  `-exp(A_log) softplus(f(x) + dt_bias)`, per-channel b / w gates, the GatedDeltaNet-2 rule. 6 softmax, 18 GDN-2.
- klm (MinimaLabs/KLM-1.5): `is_attention = layer % attention_every == attention_every - 1`
  (configuration_klm.py:55-56); `Attention if is_attention else GatedDeltaNet2` (modeling_klm.py:270-272). The same
  GDN-2 rule (GatedDeltaNet2, :205) with a decay floor of -20. 6 softmax, 18 GDN-2. (Its eos document resets are
  eos_document_mask.)
- Edge0-8B-A1B-preview (no model_type, BailingMoeV3ForCausalLM): MLA iff `(i + 1) % layer_group_size == 0` or
  `i >= n // G * G`, KDA elsewhere (modeling_bailing_moe_v3.py:1720-1721). KDA lower bound `config.kda_lower_bound`
  (:1446), -5 under kda_safe_gate. G 4, 24 layers: 6 MLA, 18 KDA.
- Ming-3.0-asr-tiny (text_config without model_type; lm model_type resolves to the outer ming_v3_asr): the same
  schedule (configuration_ming_v3_asr.py:151-155; modeling_ming_v3_asr.py:1217-1221; layer_group_size default 5, :72;
  the snapshot has 4). The same KDA_lb. 6 MLA, 18 KDA.
- solar_open2: GQA iff `layer_idx in config.gqa_layers`, else `(layer_idx + 1) % gqa_interval == 0`
  (Solar-Open2-250B-Nota-NVFP4/modeling_solar_open2.py:779-783). SolarOpen2LinearAttention (KDA) elsewhere (:261) with
  `lower_bound = kda_gate_lower_bound` (-5.0, :278, :399) and `allow_neg_eigval` (:279, :401): beta = 2 sigmoid(b), in
  (0, 2), true in both snapshots. gqa_layers 0, 4, ..., 44: 12 GQA, 36 KDA.

## 2. Classifier change
- reads: gdn_layers_per_gqa (gptx3), attention_every (klm), layer_group_size + architectures BailingMoeV3ForCausalLM /
  model_type ming_v3_asr, gqa_layers / gqa_interval / kda_allow_neg_eigval (solar_open2)
- new:
  - `_normalize`: layer_types ('full_attention' / 'linear_attention') for these 5 schedules.
  - Treat Edge0 (BailingMoeV3ForCausalLM with no model_type) and ming_v3_asr as bailing_hybrid in `_normalize`'s
    layer_group_size block and in `_recurrent_family`.
  - RECURRENT_FAMILY: gptx3 -> "gdn2", klm -> "gdn2", solar_open2 -> "kda" with a new variant "kda_lb_negeig"
    (beta in (0, 2): the gdn_negeig precedent, its own family, no RF_NESTS entry).
- detection (returns [n_softmax, n_recurrent, family]; full code in verify/r3/parts/a6_masks.py):
```python
def detect_hybrid_layer_schedule_unread(cfg, model_type):
    lm = lm_of(cfg); n = layer_count(lm); a = archs(cfg)
    if model_type == "gptx3" and num(lm.get("gdn_layers_per_gqa")):   # (i + 1) % (k + 1) == 0 softmax
        ...; return [s, n - s, "gdn2"]
    if model_type == "klm":                                            # i % every == every - 1 softmax
        ...; return [s, n - s, "gdn2"]
    if model_type == "solar_open2":                                    # gqa_layers, else gqa_interval
        ...; return [s, n - s, "kda_lb_negeig" if kda_allow_neg_eigval else "kda_lb"]
    if model_type != "bailing_hybrid" and ("BailingMoeV3ForCausalLM" in a or model_type == "ming_v3_asr"):
        ...; return [s, n - s, "kda_lb" if kda_safe_gate and kda_lower_bound is not None else "kda"]
```
- apply note (p5:model_graph.py):
  - `_normalize` (~L2668, next to the bailing_hybrid block): widen the condition to `mt == "bailing_hybrid" or
    (not mt and "BailingMoeV3ForCausalLM" in archs) or mt == "ming_v3_asr"`, and add gptx3 / klm / solar_open2 blocks
    writing layer_types.
  - classify's structural fallback (~L2152) then sees 'linear_attention' entries: mixer=hybrid.
  - `_recurrent_family` (~L2040): route the bailing aliases through the "V3" branch (Edge0's arch has V3; for
    ming_v3_asr test kda_safe_gate). Add RECURRENT_FAMILY entries (~L2020) and the solar_open2 neg-eigval branch, like
    olmo_hybrid's gdn_negeig (~L2064).
  - Check QKNORM_KEY_NOT_SOFTMAX_ARCHS (~L1873) and the ming_v3_asr table entries (~L1872, ~L3708) stay consistent.

## 3. Relation rule
- target side: its recurrent layers (R) need host R layers of the same rf_ family (feats_ok rf_ rule: ("wall", "free"));
  its softmax layers need FU
- host side: its R layers play only same-family recurrent layers; softmax targets use only its softmax layers (ZERO_EDGES
  mixer softmax <- hybrid: depth = the softmax-layer count)
- kind: exact (ZERO_EDGES / kind_match); wall (rf_ family mismatch)
- cost: depth (6 / 6 / 6 / 6 / 12 softmax layers instead of 24 / 24 / 24 / 24 / 48)
- free by the config rules: none (the layer schedule is a feature switch; the KDA lower bound is code / config but the
  rf_ family is not a config lever here)
- theorem: `zeroed_sublayer` (PROOFS.md "mixer / window / crosslayer_kv: ZERO_EDGES"); the rf_ wall is the existing
  near-gap rule (not payable)

## 4. Test fixtures
- flag: AxiomicLabs/GPT-X3-150M -> [6, 18, "gdn2"]; Edge0/Edge0-8B-A1B-preview -> [6, 18, "kda_lb"]; MinimaLabs/KLM-1.5
  -> [6, 18, "gdn2"]; inclusionAI/Ming-3.0-asr-tiny -> [6, 18, "kda_lb"]; nota-ai/Solar-Open2-250B-Nota-NVFP4 and
  upstage/Solar-Open2-250B -> [12, 36, "kda_lb_negeig"]
- not flag: inclusionAI/Ling-3.1-flash, TensorFold/Ling-3.0-flash-oQ4 / -oQ5, rapid-mlx/Ling-3.0-tiny-MLX-4bit
  (bailing_hybrid: read by p5); AxiomicLabs/GPT-S3-3M (gdn_layers_per_attention: read); moonshotai/Kimi-Linear-48B-A3B-Instruct
  (linear_attn_config.full_attn_layers: read); Qwen/Qwen3-Next-80B-A3B-Instruct

## Verdict
Real, fewer edges for 6 models.
- As targets they leave the softmax-plain class: every host without their rf_ family's recurrent layers loses them.
  That is most of the plain-softmax hosts that played them as all-softmax stacks, likely hundreds of pairs each for
  the small ones.
- As hosts they keep only their softmax layers for plain targets (depth 6 / 12 instead of 24 / 48).
- Edge0 / Ming join the Ling-3.x class: a few new edges to and from the bailing_hybrid points.
Region: recurrent-other / classification.
