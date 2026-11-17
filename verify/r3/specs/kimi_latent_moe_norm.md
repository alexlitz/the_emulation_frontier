# kimi_latent_moe_norm
live 4 / placed 4 (kimi_linear 1, ernie5_5_moe 1, muse_spark_text 2); direction over-credit; batch a10_moe_ffn

## 1. What the code computes
- Kimi-K3 (nvidia/Kimi-K3-NVFP4/modeling_kimi_linear.py): `use_latent_moe = routed_expert_hidden_size is not None`, and the experts are built at that width (:776-795). `routed_expert_down_proj` (hidden -> L), `routed_expert_up_proj` (L -> hidden), and with `latent_moe_use_norm` `routed_expert_norm = KimiRMSNorm(L, rms_norm_eps)` (:804-813). forward: `y = U(N(sum_k w_k E_k(D x)))`, while the shared experts read x at full width (:815-838). The router reads the full x (:817).
- ERNIE-5.5 (ext:ernie5_5_moe/gh__baidu__vLLM-ERNIE/moe.py): `fc1_latent_proj` (hidden -> moe_latent_size), and `latent_norm = RMSNorm(moe_latent_size, rms_norm_eps)` built unconditionally (:180). `Ernie5_5RoutedOutputTransform` applies `fc2(latent_norm(.))` after the expert combine (:91-101, 209-211), and the shared experts read the original x (:214-225). Same form as K3, with L = 2048 and hidden 4096.
- Muse-Spark (tfmain:models/muse_spark/modular_muse_spark.py:376-394): the router reads the full x. Then `pre_expert_proj` (8192 -> moe_hidden_size 4096), `pre_expert_norm = CenteredRMSNorm(4096, rms_norm_eps)` ((1 + w) gain, :293-299) ON THE LATENT INPUT, the experts at 4096, and `post_expert_proj` back to 8192. So the norm sits before the experts ("in"), not on their sum.
- p5 `_moe_sig` reads only `moe_latent_size` (p5:model_graph.py:943). So K3 and Muse get latent = 0 (experts assumed on the hidden stream), and no latent norm is represented for any of the three. The `moe_why` rank rule (`rank_ok`, `expert latent` ~973 / 990) treats a latent as a linear rank cap (Nemotron-3: fc1 / fc2 with no norm), which these are not.

## 2. Classifier change
- reads: `routed_expert_hidden_size` + `latent_moe_use_norm` (kimi_linear), `moe_latent_size` (+ `latent_moe_use_norm`, ernie5_5_moe), `moe_hidden_size` (muse_spark*).
- new: MOE_SIG field `latent_norm` ("" | "sum" | "in"), and `latent` also from routed_expert_hidden_size / moe_hidden_size, in `_moe_sig`.
- detection:
```python
def detect_kimi_latent_moe_norm(cfg, model_type):
    lm = lm_of(cfg)
    if model_type == "kimi_linear":
        w = num(lm.get("routed_expert_hidden_size"))
        if w and lm.get("latent_moe_use_norm") is True:
            return ["sum", w]
        return None
    if model_type == "ernie5_5_moe":
        w = num(lm.get("moe_latent_size"))
        if w and lm.get("latent_moe_use_norm") is not False:
            return ["sum", w]
        return None
    if model_type in ("muse_spark", "muse_spark_text"):
        return ["in", num(lm.get("moe_hidden_size")) or 4096]
    return None
```
- apply note: `_moe_sig` (~943). Replace `lat = _num(d.get("moe_latent_size"))` with `moe_latent_size or routed_expert_hidden_size (kimi_linear) or moe_hidden_size (muse_spark*)`. Append `latent_norm` (the detector's [0]) to the tuple and to `MOE_SIG` (~950). In `moe_why` add, before the `expert latent` check: if `T["latent_norm"] and T["latent_norm"] != H["latent_norm"]`, wall. If `H["latent_norm"] == "sum" and not T["latent_norm"]`, allowed only under the eps lever (`eps_h & _p5("eps_latent")` in emul_matrix ~4370, the same port as the MLA latent). If `H["latent_norm"] == "in" and not T["latent_norm"]`, allowed when `H["latent"] >= ht` (as rank_ok). Also set dims `ew` at the latent width (unchanged: the expert intermediate is moe_intermediate_size either way).

## 3. Relation rule
- target side: "sum": y = U N(sum_k w_k E_k(D x)) is a per-token normalisation of the routed sum. No host without it forms it (as a sandwich post-norm target, `post_norm` wall), so it needs a "sum" host with L_h >= L_t. That edge is exact at L_h = m L_t (m copies, `rmsNorm_rep`), and otherwise the zero-padded latent needs eps' = eps L_t / L_h (`qknorm_pad`), the MLA-latent integer-multiple rule. "in": the experts read N(P u); a plain MoE host's experts read u alone, so it needs an "in" host, with the same pad rule.
- host side: a "sum" host playing a no-latent-norm MoE target is the eps lever on a sublayer-output norm: routed sum written /S, norm weight S sqrt(eps). It is a limit (bound rate 1/S^2), and it needs L_h >= hidden_t (rank). An "in" host plays a target with hidden_t <= L_h: P is an isometric pad of the unit-RMS input, so ms(P u) = hidden_t / L_h is constant and the norm is a constant scale. That is exact if eps' = eps hidden_t / L_h is the eps the stream pad also needs, else an eps-only gap (lever).
- kind: target wall (unless a same-kind host); host limit ("sum", lever) / exact-up-to-eps ("in").
- cost: rank: hidden_t <= L_h (K3 3584, ERNIE 2048, Muse 4096).
- free by the config rules: eps is free; latent_moe_use_norm is a norm on/off flag, so it is free. A K3 host may set latent_moe_use_norm = false, which makes its latent MoE linear (rank-capped, exact). The kimi host-side rule is therefore exact through the flag, not the lever. The flag is not available to ERNIE (built unconditionally) or Muse (no flag).
- theorem: `postNorm_layer_bound` (sum, lever); `rmsNorm_rep` / `qknorm_pad` (same-kind wider latent); `lowrank_pad` (rank). Target wall needs theorem `latent_sum_norm_not_plain` (analogue of `mla_latent_norm_not_affine` for a norm on the expert sum).

## 4. Test fixtures
- flag: nvidia/Kimi-K3-NVFP4, moonshotai/Kimi-K3.1-Pro, pipenetwork/Kimi-K3-REAP80-MLX-mxfp4-q8, mlx-community/Kimi-K3-mlx-reap160-2bit -> ["sum", 3584]; baidu/ERNIE-5.5-Flash-Base -> ["sum", 2048]; Interchained/Muse-Spark-Open-816B-A42B-GGUF, staccs/Muse-Spark-Open-816B-A42B-lecore-mlx -> ["in", 4096].
- not flag: moonshotai/Kimi-Linear-48B-A3B-Instruct / -Base (kimi_linear, no latent); nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16, -Ultra-550B (moe_latent_size with no norm: a linear latent); baidu/ERNIE-4.5-21B-A3B-PT.
- scan: 7 K3 kimi_linear repos (the carrier list names only Kimi-K3-NVFP4, but K3.1-Pro, its EXL3 build and the REAP prunes take the same path by config), ERNIE-5.5, 2 Muse-Spark.

## Verdict
real. Fewer edges. The K3 / K3.1 (7), ERNIE-5.5 and Muse-Spark points lose plain-MoE hosts, keeping only same-kind latent-norm hosts (each other within a family). As hosts they keep only lever / flag edges, and the rank cap (hidden_t <= L) appears for K3 and Muse. That is about 10 points, but today only the 2 Muse-Spark points move. K3 already has attn_res (attn_res_block_size 12) and ERNIE-5.5 has hyper_conn (hc_mult 4), both wall / wall, so the change is latent for them until those walls lift.
