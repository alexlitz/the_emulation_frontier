# bias_qkv_by_code_unlisted
live 16 / placed 16 (cambrian_qwen 3, arkasr 2, llava_qwen2 2, spatiallm_qwen 2, videollama3_qwen2 2, hulumed_qwen2 1, llava-qwen2 1, meditor 1, step_audio_2 1, sundial 1); direction over-credit; batch a1_typetables

## 1. What the code computes
transformers' Qwen2Attention hard-codes q / k / v bias: tf5:models/qwen2/modeling_qwen2.py:189-191 `bias=True`, :192 o_proj `bias=False`. Each type below runs that class (or its own bias=True projections) and its config has no bias key, so p5's `_feats` (p5:model_graph.py:1793-1796: `mt in BIAS_QKV_TYPES` or a key `is True`) gives no bias_qkv:
- arkasr: Edge0/ARK-ASR-3B/modeling_arkasr.py:7, :75 `ArkasrForConditionalGeneration(Qwen2ForCausalLM)`; Edge0/Audio8-ASR-0.1B/modeling_arkasr.py:8, :92 `language_model = Qwen2ForCausalLM(config)`
- step_audio_2: stepfun-ai/Step-Audio-2-mini/configuration_step_audio_2.py:3, :81 (text_config = Qwen2Config), modeling_step_audio_2.py:8, :317 `self.model = Qwen2Model(config.text_config)`
- llava_qwen2 (FastVLM): apple/FastVLM-1.5B/llava_qwen.py:31, :2108 `LlavaQwen2Model(LlavaMetaModel, Qwen2Model)`, :2115-2120
- llava-qwen2 (nanoLLaVA): qnguyen3/nanoLLaVA/modeling_llava_qwen2.py:1046-1049 bundled Qwen2Attention q / k / v `bias=True`, o `bias=False`; :2229-2234
- hulumed_qwen2: ZJU-AI4H/Hulu-Med-7B/modeling_hulumed_qwen2.py:26, :162 `HulumedQwen2Model(HulumedMetaModel, Qwen2Model)`, :392-399
- videollama3_qwen2: DAMO-NLP-SG/VideoLLaMA3-2B/modeling_videollama3.py:28, :111, :326-332
- cambrian_qwen: ext cambrian_qwen/gh__cambrian-mllm__cambrian-s/cambrian/model/language_model/cambrian_qwen2.py:22, :43 `CambrianQwenModel(CambrianMetaModel, Qwen2Model)`, :306
- spatiallm_qwen: ext spatiallm_qwen/gh__manycore-research__SpatialLM/spatiallm/model/spatiallm_qwen.py:10-11, :37-42 `self.model = Qwen2Model(config)`
- meditor (own code): Yo-Purple/Meditor-1-2.2B-A500M-MoE/modeling_meditor.py:131-133 q / k / v `bias=True`, :134 o `bias=False`
- sundial (own code, a time-series model): thuml/sundial-base-128m/modeling_sundial.py:109-111 q / k / v `bias=True`, :112 o `bias=False`. Moot if sundial is dropped as a non-LM.

## 2. Classifier change
- reads: model_type
- new: table entries in BIAS_QKV_TYPES (feature bias_qkv); no BIAS_OUT entry (o_proj is bias-free in all of them)
- detection:
```python
BIAS_QKV_BY_CODE = {"arkasr", "cambrian_qwen", "hulumed_qwen2", "llava-qwen2", "llava_qwen2", "meditor",
                    "spatiallm_qwen", "step_audio_2", "sundial", "videollama3_qwen2"}

def detect_bias_qkv_by_code_unlisted(cfg, model_type):
    return True if model_type in BIAS_QKV_BY_CODE else None
```
- apply note: in p5:model_graph.py add the ten types to `BIAS_QKV_TYPES` (:1947). `_feats` already gates on `use_qkv_bias` / `qkv_bias` not False, which none of these configs set. Add the same ten to `QBIAS_TYPES` (:3816, the q-bias table the start-token rules read). A general alternative (resolve a wrapper whose remote class subclasses Qwen2Model to qwen2) is out of scope; the table is exact for the snapshot.

## 3. Relation rule
- target side: these 16 targets now need a host with q / k / v biases (bias_qkv target <- bias-free host is impossible)
- host side: as hosts they play bias-free targets (zero the bias), unchanged in effect
- kind: wall (target <- bias-free host); exact (biased host -> unbiased target)
- cost: none
- free by the config rules: no (attention_bias is shape-changing; here it is not even a key)
- theorem: `biased_proj_not_bias_free`, `rmsNorm_neg` (PROOFS.md "bias_qkv target <- bias-free host", PROVED IMPOSSIBLE); `bias_off` ("bias_qkv / bias_out host -> unbiased target")

## 4. Test fixtures
- flag: thuml/sundial-base-128m, Edge0/ARK-ASR-3B, Edge0/Audio8-ASR-0.1B, Yo-Purple/Meditor-1-2.2B-A500M-MoE, stepfun-ai/Step-Audio-2-mini, apple/FastVLM-0.5B, apple/FastVLM-1.5B, qnguyen3/nanoLLaVA, ZJU-AI4H/Hulu-Med-7B, DAMO-NLP-SG/VideoLLaMA3-2B, nyu-visionx/Cambrian-S-7B, manycore-research/SpatialLM-Qwen-0.5B -> true
- not flag: ZJU-AI4H/Hulu-Med-4B (Hulu-Med on a Qwen3 body: no bias); zenosai/MonkeyOCRv2-B-Parsing (Qwen3 body); Qwen/Qwen2.5-0.5B-Instruct (qwen2: already listed); Qwen/Qwen3-0.6B; liuhaotian/llava-v1.5-7b (LLaVA on Llama, bias-free)
- scan: exactly the 16 live repos.
- follow-up (not flagged, code not local): model_type "llava" with architectures LlavaQwenForCausalLM (lmms-lab/llava-onevision-qwen2-0.5b-ov / -7b-ov, lmms-lab/LLaVA-Video-7B-Qwen2) is LLaVA-NeXT's LlavaQwenModel(LlavaMetaModel, Qwen2Model) -- same fact, p5 gives them no bias_qkv; key it on the architecture once the LLaVA-NeXT source is checked.

## Verdict
real. Fewer edges: 16 targets lose every bias-free host (they keep Qwen2 / Qwen2.5 / biased hosts); typically tens of edges each, a few hundred in total, mostly Qwen2-sized hosts such as Llama / Qwen3 now walled.
