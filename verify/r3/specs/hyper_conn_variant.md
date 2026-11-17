# hyper_conn_variant
live 53 / placed 53 (deepseek_v4 2, deepseek_v41_text 24, hy_v4 1, qwen4_exp 1, qwen4_exp_text 24, xing4_0 1); direction over-credit; batch a8_mla_dsv4

## 1. What the code computes
- dsv4 (tf5:models/deepseek_v4/modeling_deepseek_v4.py): HyperConnection :974-990 `pre = sigmoid(.) + hc_eps`, `post = 2 sigmoid(.)`, `comb = softmax(.) + hc_eps`, column norm, then `hc_sinkhorn_iters - 1` row / column passes; DecoderLayer :1176-1192 `post * f(norm(collapsed)) + comb^T @ streams` ("comb is consumed transposed ... the direction matters"); final collapse HyperHead :993-1009 `sum_i (sigmoid(.) + eps)_i x_i` (learned). Same code: MemT (subclass), ERNIE-5.5 (vLLM DeepSeek-V4 mhc / hc_head tilelang kernels, ext:ernie5_5_moe/.../model.py:18-20, :149-154).
- glm5_next (tf5:models/glm5_next/modular_glm5_next.py): :368-369 `Glm5NextTextHyperConnection(DeepseekV4HyperConnection): pass`, layer :1196 / :1205 comb^T; :372-376 HyperHead = unweighted mean of the streams (`hidden_streams.mean(dim=2)`).
- dsv4_nano (HuggingFaceTB/nanowhale-100m-base/modeling_deepseek_v4.py:417-428 `hc_post`): `einsum("bsij,bsjd->bsid", comb, residual)` = comb (not comb^T); hc_sinkhorn_iters 2 in the config.
- dsv41 (ext:deepseek_v41/hub__deepseek-ai__DeepSeek-V4.1-Flash/inference/model.py): Block.forward :970-1001 the attention's input is collapsed with the PREVIOUS FFN's pre (`x = hc_pre(x, pre_mix)`), the FFN's with this attention's pre (`hc_pre(x, attn_pre)`); hc_post :964-968 comb^T orientation (sum over the first comb axis); layer 0 starts from `make_identity_pre_mix` (one-hot stream 0, :1162-1166); the final collapse is the last FFN's Sinkhorn pre (ParallelHead, no sigmoid head).
- xing (XingChen-AGI/Xing4.0-29B-Intent/modeling_xing4_0.py): :464, :476 comb logits clamped to [mhc_h_res_clamp_min, max] (-30, 30); :473 `pre = sigmoid(.)` with no + hc_eps; :480-481 full row + column passes; :525 / :532 `matmul(comb, h)` (comb, not comb^T); :633 final collapse `hidden_states.mean(dim=2)`.
- hy4 (tf5:models/hy_v4/modular_hy_v4.py): HyperConnection :372-405 only pre / post (no comb, no Sinkhorn), `post = hc_magnitude * sigmoid(.) + hc_eps`; DecoderLayer :443-463 `streams = post * f(.) + residual` (identity carry).
- qwen4_exp (tf5:models/qwen4_exp/modular_qwen4_exp.py:531-559 GatedResidual): grouped RMSNorm over hc_count x hidden, low-rank (hc_lowrank) silu / sigmoid input mix divided by hc_count, mean over streams, inject `2 sigmoid(W_b n(h) / hc_count)`; no Sinkhorn.
- motif (Motif-Technologies/Motif-3/modeling_motif.py:162-267 MHCLayer, :1086-1101): its own mHC (sinkhorn_iters, h_post_coeff = 1 + mhc_h_post_alpha_end).
- needle / san: lanes with fixed lane offsets, 20 / 14 log-domain Sinkhorn iterations in code (see hyper_conn_key_unread).
- p5 today: `hyper_conn` is one bit (p5:model_graph.py:1757-1760), wall / wall (FEATURES :1609): any two hyper_conn models are credited to each other when the rest fits.

## 2. Classifier change
- reads: model_type; auto_map (nanowhale remote code); the hyper_conn keys (+ mhc_lanes, extras.mhc_lanes from hyper_conn_key_unread)
- new: a hyper_conn variant: FEATURES `"hyper_conn"` keeps the wall to non-HC models; a new dims / slot value `"hc_var"` in `_pass5_dims` or as `"hyper_conn:<v>"` FEATURES keys, compared like QKNORM_VARIANTS (host variant == target variant, or a nest entry)
- detection:
```python
def detect_hyper_conn_variant(cfg, model_type):
    lm = lm_of(cfg)
    if not _hc_on(lm, cfg) or model_type not in HC_VARIANT_BY_TYPE:
        return None
    if model_type == "deepseek_v4" and "modeling_deepseek_v4" in auto_map(cfg):
        return "dsv4_nano"              # nanowhale remote code: comb (not comb^T), its own Sinkhorn
    if model_type == "ernie5_5_moe" and lm.get("enable_hyper_connections") is False:
        return None
    return HC_VARIANT_BY_TYPE[model_type]   # dsv4 / glm5_next / dsv41 / xing / hy4 / qwen4_exp / motif / needle / san
```
- apply note: p5:model_graph.py: add `HC_VARIANT_BY_TYPE` (the detector's table) next to QKNORM_VARIANT_BY_TYPE (:1879) and `HC_NEST = {"glm5_next": ("dsv4",)}`; in `_feats` (:1760) replace `f.add("hyper_conn")` by `f.add("hyper_conn"); f.add("hc:" + variant)` and register each `"hc:<v>"` in FEATURES as ("wall", "wall") with an `_feat_ok` exception for HC_NEST (target hc:glm5_next on an hc:dsv4 host). The simplest equivalent: make every "hc:<v>" wall / wall and special-case the one nest pair in emul_matrix where FEATURES walls are applied.

## 3. Relation rule
- target side: a hyper_conn target with variant v needs a host with variant v; a glm5_next target is also played by a dsv4 host (its learned HyperHead with hc_fn = 0, base = logit(1/hc_mult - hc_eps) is the mean: exact up to rounding)
- host side: the same (no variant plays another except the nest pair); hc_sinkhorn_iters / hc_eps are config values inside a variant (free, shape-preserving)
- kind: wall between variants; exact (fold) for glm5_next <- dsv4. N vs T orientation (dsv4 vs dsv4_nano / xing) agree only in the Sinkhorn limit (the doubly-stochastic scaling of exp(A) is unique, and transposes with A): a LIMIT in hc_sinkhorn_iters at best, unmeasured -> wall
- cost: none
- free by the config rules: hc_sinkhorn_iters, hc_eps, hc_magnitude are free values; hc_mult / hc_count / hc_lowrank are shapes (compared as dims if wanted: equal hc_mult required, not proposed here); the variant itself is code
- theorem: needs theorem: "two hyper-connection maps with different stream-mixing codes (Sinkhorn comb vs comb^T at finite iterations, identity carry, gated residual, lagged pre) are not equal under any weights" (PROOFS.md lists hyper_conn as OPEN in "Walls of the edge-case audit"); nest: elementary (sigmoid(b) + eps = 1/n)

## 4. Test fixtures
- flag: HuggingFaceTB/nanowhale-100m, -base -> "dsv4_nano"; deepseek-ai/DeepSeek-V4-Flash, MemTensor/MemT-1.0-Flash, baidu/ERNIE-5.5-Flash-Base -> "dsv4"; deepseek-ai/DeepSeek-V4.1-Flash, rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX -> "dsv41"; XingChen-AGI/Xing4.0-29B-Intent -> "xing"; tencent/Hy4-preview-FP8 -> "hy4"; PeetPedro/qwen4_exp-8b-..., CMSManhattan/JiRackDeltaNet_180b, whittleOrg/Whittle-Next-27B-A3B -> "qwen4_exp"; local-inference-lab/GLM-5.3-Flash-NVFP4-Spark -> "glm5_next"; Motif-Technologies/Motif-3 -> "motif"; Cactus-Compute/needle2 -> "needle"; kenpeter123/x-small -> "san" (scan: all 53 live repos flagged, plus the 19 legacy V4, 7 GLM-5.3-Flash, ERNIE, MemT, Motif, needle, san that also carry hyper_conn)
- not flag: deepseek-ai/DeepSeek-V3.2; nota-ai/Solar-Open2-250B-Nota-NVFP4 (hc_rate 0: off); Advisio/Radha-9B, nvidia/Qwen3.6-35B-A3B-NVFP4 (Qwen3.5 / 3.6 without hc_count); zai-org/GLM-5.3 (glm_moe_dsa, no HC); dots-studio/dots3-note-prev

## Verdict
real (over-credit): edges between hyper_conn models of different codes disappear: chiefly V4.1 hosts -> V4 targets and V4 <-> GLM-5.3-Flash (glm5_next targets keep dsv4 hosts by the nest), V4 <-> Xing / Hy4 / Qwen4-exp where dims fit. Fewer edges, tens to ~100 pairs among the ~85 hyper_conn models; no new edges.
