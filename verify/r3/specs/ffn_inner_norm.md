# ffn_inner_norm
live 4 / placed 4 (dory 4); direction over-credit; batch a4_norms

## 1. What the code computes
- DoryMLP (ext:dory/transformers-pr50099/modeling_dory.py:418-436): `suv_gate`, `suv_up` per intermediate channel (ones) times `suv_scale = hidden_size ** 0.5` (:428-430); `ffn_sub_norm = DoryL2Norm()` (:431); forward `down_proj(ffn_sub_norm(act_fn(gate_proj(x) * suv_gate * s) * (up_proj(x) * suv_up * s)))` (:433-436).
- suv_gate / suv_up are per-row scales of gate_proj / up_proj: fold exactly into those rows (gate: the scale is inside the activation's argument, which is gate_proj's output -- still a row scale of gate_proj). Not an issue.
- ffn_sub_norm: weight-free L2 norm (`x * rsqrt(sum x^2 + 1e-6)`, :50-61, eps fixed in code) of the SwiGLU product z over the intermediate width, before down_proj. The FFN output is down(z / |z|): its magnitude no longer follows the input's.
- Same code fact elsewhere (not in the snapshot): BitNet `ffn_sub_norm = BitNetRMSNorm(intermediate_size)` before down_proj, weighted (tf5:models/bitnet/modeling_bitnet.py:74, :77).
- Near miss: Muse-Spark's `pre_expert_norm` / `post_expert_norm` normalise the MoE latent input and each expert's OUTPUT (tfmain:models/muse_spark/modular_muse_spark.py:345, :369, :382, :391), not the intermediate activation; out of scope here.

## 2. Classifier change
- reads: model_type (code fact).
- new: FEATURES key `ffn_norm` ("wall", "wall"), set in `_feats` for FFN_NORM_TYPES = {"dory", "bitnet"} (values "l2_free" / "rms_w"; BitNet's weighted norm plays Dory's weight-free one at the same width only with eps rescaled, so keep one key and let the eps rule ride on the free eps).
- detection:
```python
def detect_ffn_inner_norm(cfg, model_type):
    if model_type == "dory":
        return "l2_free"
    if model_type == "bitnet":
        return "rms_w"
    return None
```
- apply note: p5:model_graph.py FEATURES (after "vnorm", ~1625); `_feats` (~1766): `if mt in FFN_NORM_TYPES: f.add("ffn_norm")`. Nothing replaced (ffn glu / act SiLU stay).

## 3. Relation rule
- target side: an ffn_norm target needs an ffn_norm host: a plain SwiGLU's output scales with its input (z -> 0 as x -> 0), the normed one's does not.
- host side: eps lever through down_proj: scale up_proj rows by 1/S (z / S, linear in up), then L2(z / S) = z / (S sqrt(|z|^2 / S^2 + eps)) -> z / (S sqrt(eps)); down_proj x S sqrt(eps). Eps 1e-6 > 0 fixed in code (fine: the lever keeps it). A limit, unmeasured.
- kind: target side wall; host side limit (eps lever) -- wall until measured (top-1 >= 99 %, >= 90 % identical 32-token greedy, fp32). Dory hosts are walled by ngpt anyway.
- cost: none.
- free by the config rules: none (code); suv scales are weights.
- theorem: target side: needs theorem `ffn_norm_not_plain` (one-token argument: at small inputs the plain FFN output -> 0, the normed one stays at unit-norm z direction times down -- the vnorm_not_plain shape); host side `epsLever_eq`, `epsLever_bound`, `lever_model_bound` (per-vector lever, factor absorbed into down_proj).

## 4. Test fixtures
- flag: ProCreations/Dory-Beam-4B, netprojects/nemotron-dory1.0-4b-mlx-8bit, nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8, vladimir94/dory-1.0-4b-ane-coreml -> "l2_free".
- not flag: BAAI/bge-reranker-v2.5-gemma2-lightweight (Gemma-2 MLP, no inner norm); Interchained/Muse-Spark-Open-816B-A42B-GGUF (expert in / out norms, not an intermediate norm).

## Verdict
real, but subsumed in practice: the 4 Dory points are walled both ways by ngpt_sphere_residual; ffn_norm adds no edge change on the snapshot (it matters for a future BitNet / Dory-family host). Region: MoE / FFN.
