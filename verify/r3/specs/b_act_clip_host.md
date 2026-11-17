# b_act_clip_host
live 54 / placed 55 (deepseek_v41_text 24, deepseek_v4 19, glm5_next_text 7, step3p5 2, hy_v4 1, memt_deepseek_v4 1; exaone_moe 1 placed but not live); direction over-wall; batch bd

## 1. What the code computes
- tf5 DeepSeek-V4: `self.limit = config.swiglu_limit`; `gate = gate_proj(x).clamp(max=self.limit)`, `up = up_proj(x).clamp(min=-self.limit, max=self.limit)` (tf5:models/deepseek_v4/modeling_deepseek_v4.py:1022-1026; experts :1042, :1066-1067). Class default `swiglu_limit: float = 10.0` (configuration_deepseek_v4.py:166). `clamp(min=-None)` raises, so the off value is `float("inf")` (JSON `Infinity`): clamp(x, -inf, inf) = x for every finite x.
- tf5 GLM-5-Next: `gate.clamp(min=None, max=self.swiglu_limit)`, `up.clamp(min=-L, max=L)` (tf5:models/glm5_next/modeling_glm5_next.py:97-104, :119-141, :1524-1554). tf5 Hy4: same (tf5:models/hy_v4/modeling_hy_v4.py:563-585). MemT subclasses tf5 V4 (MemTensor/MemT-1.0-Flash/modeling_memt.py:34).
- V4.1 reference: `swiglu_limit: float = 0.0` default, the clamp only `if self.swiglu_limit > 0` (ext:deepseek_v41/hub__deepseek-ai__DeepSeek-V4.1-Flash/inference/model.py:70, :836-847): 0 = no clamp.
- Step-3.5 remote code: per layer `swiglu_limits[l]` / `swiglu_limits_shared[l]`, `None` when the entry is 0 or None (stepfun-ai/Step-3.5-Flash/modeling_step3p5.py:556-566); the MLP clamps only `if self.limit is not None` -- on silu(gate) and up (:222-229). Snapshot: limits 7 on layers 43-44 (routed), 16 on layer 44 (shared). tf5 Step-3.7: `limit = (... or 0) or float("inf")` (tf5:models/step3p7/modeling_step3p7.py:544, :565).
- K-EXAONE (exaone_moe): the config carries `swiglu_limits` [0 .. 7], but tf5 ExaoneMoe never reads it (no limit / clamp in tf5:models/exaone_moe/*.py, no auto_map) -- under the HF code it is an unclamped SwiGLU.
- p5 relation: `FEATURES["act_clip"] = ("wall", "wall", ...)` (p5:model_graph.py:1704-1706), set when `swiglu_limit(s)` > 0 and not gpt_oss / an "oai" activation (:1826-1828) -- exaone_moe included.

## 2. Classifier change
- reads: `swiglu_limit` / `swiglu_limits` / `swiglu_limits_shared` (> 0 anywhere) on the families whose code reads them.
- new: FEATURES["act_clip"] host side "wall" -> "free"; side fix in `_feats`: act_clip only for `ACT_CLIP_CFG_TYPES` (drops exaone_moe).
- detection:
```python
_ACT_CLIP_CFG_TYPES = {"deepseek_v4", "memt_deepseek_v4", "deepseek_v41", "deepseek_v41_text", "glm5_next",
                       "glm5_next_text", "hy_v4", "step3p5", "step3p7"}

def detect_b_act_clip_host(cfg, model_type):
    lm = lm_of(cfg)
    if model_type not in _ACT_CLIP_CFG_TYPES:
        return None
    for k in ("swiglu_limit", "swiglu_limits", "swiglu_limits_shared"):
        v = lm.get(k)
        if isinstance(v, list):
            if any((fnum(x, 0.0) or 0.0) > 0 for x in v):
                return k
        elif (fnum(v, 0.0) or 0.0) > 0:
            return k
    return None
```
- apply note: p5:model_graph.py FEATURES :1704 `("wall", "wall", ...)` -> `("wall", "free", ...)` (the why text: "a clamped host sets its limit to inf / 0 = off"); `_feats` :1826-1828 add `and mt in ACT_CLIP_CFG_TYPES`. The Z3 model reads FEATURES (S.frule), so it follows.

## 3. Relation rule
- target side: unchanged wall (`act_clip_unclamped_fails`: no analytic host equals a scaled clamp; ReLU hosts not covered).
- host side: a clamped host plays an unclamped target with the limit switched off.
- kind: exact. tf5 families: L = inf is the identity on every finite input (bitwise; also L = FLT_MAX). Step-3.5 / V4.1: L = 0 builds no clamp at all. In real arithmetic `act_clip_regime` gives exactness for any L above every activation of a bounded context, which a finite L also achieves (the same footing as DSA's index_topk >= context).
- cost: none.
- free by the config rules: free -- swiglu_limit is a cap value ("scales and caps ... softcap values" are free), shape-preserving, not a feature switch.
- theorem: `act_clip_regime` (PROOFS.md Round-2 rules, "act_clip: clamped host -> unclamped target"; "wall NOT TIGHT"); the inf / 0 setting is the identity by the code, no limit argument needed.

## 4. Test fixtures
- flag: deepseek-ai/DeepSeek-V4-Flash-0731, deepseek-ai/DeepSeek-V4.1-Flash, MemTensor/MemT-1.0-Flash, tencent/Hy4-preview-FP8, shashanth-27xA9/GLM-5.3-SuperExpert-v2-9.2GB -> "swiglu_limit" (10.0); stepfun-ai/Step-3.5-Flash, stepfun-ai/Step-3.5-Flash-Base-Midtrain -> "swiglu_limits" (7 on layers 43-44).
- not flag: HuggingFaceTB/nanowhale-100m-base (swiglu_limit 0.0); openai/gpt-oss-20b, nvidia/MiniMax-M3-NVFP4 (swigluoai clamp: the act slot's regime, not this feature); LGAI-EXAONE/K-EXAONE-2.0-750B-A37B (tf5 never reads swiglu_limits); LGAI-EXAONE/K-EXAONE-236B-A23B; zai-org/GLM-5.3 (glm_moe_dsa, no limit).

## Verdict
real (over-wall), exact, host side free. Measured (931 real shapes, host side flipped): exact +62 shape pairs, payable
+393 (1,239 target-repo edges). Per host (payable target repos): K-EXAONE-2.0-750B 1,227 (its act_clip is spurious
under tf5 -- the side fix gives it these edges anyway), Step-3.5-Flash-Base-Midtrain 103, Step-3.5-Flash 65,
Hy4-preview-FP8 2, Fx-Bio 2; the V4 / V4.1 / GLM-5.3 hosts gain nothing alone (kv_compress / hash_route /
hyper_conn). Ambiguity: if K-EXAONE's intended runtime (vLLM) clamps, the relation still reads the HF code.
