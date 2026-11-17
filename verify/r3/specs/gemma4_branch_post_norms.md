# gemma4_branch_post_norms
live 17 / placed 65 (gemma4_text 64, gemma4 1); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
- `Gemma4TextDecoderLayer.__init__`: with `config.enable_moe_block` it builds `router`, `experts` and three more norms `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`, `pre_feedforward_layernorm_2` (tf5:models/gemma4/modeling_gemma4.py:1376-1382).
- forward (modeling_gemma4.py:1410-1429): `r = x`; `h1 = N_post1(mlp(N_pre(r)))`; `h2 = N_post2(experts(N_pre2(r), router(r)))`; `out = r + N_post(h1 + h2)`. Each branch gets its own output RMSNorm, then the sum is normed again (the sandwich `post_feedforward_layernorm`).
- `Gemma4RMSNorm`: `x * (ms(x) + eps)^-1/2 * w` (modeling_gemma4.py:206-215); every norm in the layer uses `config.rms_norm_eps`.
- Router (modeling_gemma4.py:1331-1352): weight-free norm, softmax, top-`top_k_experts`, renormalised, `* per_expert_scale`.
- DiffusionGemma runs the same layer with no switch (tf5:models/diffusion_gemma/modeling_diffusion_gemma.py:615-620, 646-662).
- Snapshot: all 17 live repos are 26B-A4B (hidden 2816, intermediate 2112, 128 experts x 704, top-8). Two `diffusion_gemma_text` repos run the same code; they are not placed (selfcond walls), so they are flagged but not live.
- p5 `_dense_mlp_beside_moe` (p5:model_graph.py:777) and `dims_of` (`shared = dense_inter`, ~3159) turn this into a plain always-on shared expert of width intermediate_size. `moe_why` then deals a dense target over `k*ew + shared` (one pooled FFN), which the two separately normed branches cannot form.

## 2. Classifier change
- reads: `enable_moe_block` (gemma4*), `num_experts` (diffusion_gemma*).
- new: MOE_SIG field `branch` (0 / 1) in `_moe_sig`, plus FEATURES key `moe_branch_norm` (record) in `_feats`.
- detection:
```python
def detect_gemma4_branch_post_norms(cfg, model_type):
    lm = lm_of(cfg)
    if model_type.startswith("gemma4") and lm.get("enable_moe_block") is True and num(lm.get("num_experts")) > 1:
        return "branch_norm"
    if model_type.startswith("diffusion_gemma") and num(lm.get("num_experts")) > 1:
        return "branch_norm"
    return None
```
- apply note: `_feats` (~1771, beside vnorm): add `moe_branch_norm` when `_dense_mlp_beside_moe(d)`. FEATURES gets `"moe_branch_norm": ("wall", "free", ...)`. Append `branch` to the `_moe_sig` return and to `MOE_SIG` (~950). In `moe_why` (~954): if `T["branch"] and not H["branch"]`, wall. If `H["branch"]`, the MoE-as-dense capacity is `capd = max(H["k"] * H["ew"], H["shared"])`, not the sum. If `H["branch"] and not T["branch"] and T["shared"] > 0`, wall: a shared + routed sum cannot be split over two separately normed branches. `_has_shared_expert` stays as it is (the dense branch is still always on).

## 3. Relation rule
- target side: a branch-normed target needs a branch-normed host. Then the dense branch maps to the dense branch (width >= intermediate_size) and the experts to the experts (moe_why as today). No other host forms N1(a) + N2(b) under one more norm.
- host side: a branch-normed host plays a plain (sandwich, v-norm) target on ONE branch, with the other branch zeroed (down_proj = 0 gives N(0) = 0 exactly). The double norm N_post(N_post1(y)) folds exactly: scale the branch output by 1/c with c^2 = (1 + eps)/eps and set w_post = w_t * sqrt(1 + eps). Then N(N1(y/c)) = w_t y / sqrt(ms(y) + eps), at the same eps. Dense target: dense branch width >= target width, or routed branch k*ew (zero router), not their sum.
- kind: target wall; host exact (one branch, the fold above).
- cost: host FFN capacity max(k*ew, shared) instead of k*ew + shared.
- free by the config rules: none needed (the fold uses weights only; eps unchanged). enable_moe_block is a feature switch.
- theorem: host side needs theorem `double_rmsnorm_fold`: for eps > 0 and c = sqrt((1+eps)/eps), N_eps(w sqrt(1+eps), N_eps(1, y/c)) = N_eps(w, y). The zeroed branch is `zeroed_sublayer`, the routed-only dense play is `dense_sub_moe`. Target-side wall: needs theorem (branch norms are not one norm of a sum: an OPEN wall in the "out_gate / shared_gate" style).

## 4. Test fixtures
- flag: nvidia/Gemma-4-26B-A4B-NVFP4, AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4 (gemma4 wrapper), amd/gemma-4-26B-A4B-it-w4a16-llmcompressor, mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit-REAP-14B (64 experts), PixilabAI/Blink-v0.5-26B-A4B, zentropi-ai/cope-b-a4b, Jlfn/gemma-4-26b-a4b-vq-gguf, org2ai/Wald-26B-A4B -> "branch_norm" (enable_moe_block). nvidia/diffusiongemma-26B-A4B-it-NVFP4, yifanyu/DiffusionGemma-26B-A4B-it-ORL -> "branch_norm" (same code, unconditional).
- not flag: nvidia/Gemma-4-31B-IT-NVFP4, melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell (dense Gemma-4), google/gemma-4-26B-A4B-it-assistant, google/gemma-4-E2B-it-assistant (drafters, no MoE block), Qwen/Qwen3-30B-A3B.
- scan: the 17 live gemma4_text repos and the 2 DiffusionGemma repos, nothing else.

## Verdict
real, small. The target side changes nothing today: vnorm and post_norm already limit Gemma-4 MoE targets to Gemma-family hosts, and the branch rule holds between 26B-A4B points. On the host side the 17 26B-A4B hosts lose dense targets wider than 5632 (max(8 x 704, 2112)) that the pooled 7744 admitted. In the zoo that is at most the E2B-class 6144 points, and only where their double-wide layers do not already exclude them (about 0-13 edges). The edges that remain become correct constructions (one branch plus the fold).
