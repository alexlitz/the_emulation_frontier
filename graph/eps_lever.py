"""The eps lever, measured on real models: a host switches OFF a norm the target lacks.

PORTED in round 2, pass 5: graph/model_graph.emul_matrix applies it (EPS_LEVER_QKNORM / EPS_LEVER_FEATS, dims
eps_lever; the MLA / Mamba-2 / GDN / KDA multiples relaxed). The fp16 condition below is a RUN condition there (the
emulator runs in fp32 / bf16), not a property of a checkpoint's stored dtype. This module stays the measured record.

(A proposal module: graph/model_graph.py is not edited here. The rule below is what model_graph's
host-side norm walls should read, with the measurements that back it.)

Construction (hf/eps_lever_agreement.py). An RMS norm with epsilon e fed x' = 2^-a x computes
(g 2^-a / sqrt(e)) x / sqrt(1 + rho),   rho = 2^-2a ms(x) / e.
For rho below the precision of fl(ms + e) the norm is a LINEAR map, and the gain g = 2^a sqrt(e) makes it
the identity. The weights writing the norm's input choose a, per norm, so the host turns off any norm the
target lacks; every norm the target HAS keeps its exact input (the residual is untouched, host eps =
target eps -- a config value). Switched off, on the HOST side:

  qk-norm        q_proj, k_proj x 2^-a; q_norm / k_norm gain 2^a sqrt(e) (x s_t / s_h for the softmax
                 scale: Gemma-4's scaling = 1 folds head_dim^-1/2 here)
  post-norms     (sandwich hosts: Gemma-2 / -3 / -4, GLM-4-0414, AFMoE, ...) o_proj, down_proj x 2^-a;
                 post_attention_layernorm / post_feedforward_layernorm gain 2^a sqrt(e)
  v-norm         (Gemma-4, weight-free) v_proj x 2^-a; o_proj x 2^a sqrt(e) (attention is linear in V)
  MLA latent     kv_a_proj_with_mqa latent rows x 2^-a; kv_a_layernorm gain 2^a sqrt(e_latent)
                 (transformers hard-codes e_latent = 1e-6: the lever needs only e > 0, not the config E)

COST: free. Weights only (row scalings of the projections writing the norm's input, and the norm's gain);
no width, no depth, no residual scaling, no config change beyond values the host already has. The
residual-scaled split of frontier-levers (residual x S, host eps = e_t S^2) is the same lever and measures
the same (rows "host S=2^20"), except on a (1 + w)-gain host with a tied head (Gemma-2 / -3), where the
final norm would need gain 1 / S and 1 + w is formed in fp32 -- so S = 1 is the split to use.

CONDITIONS (when the rule does NOT hold):
  * eps = 0 on the norm to be switched off: x / rms(x) is scale-invariant, there is no linear regime;
  * dtype range: the scaled tensors must stay normal numbers. a reaches 32 on gemma-2b (its MLP outputs
    have ms ~ 8e3 against e = 1e-6); fp32 / bf16 (min normal 2^-126) have ~90 binades of headroom,
    fp16 (min normal 2^-14) has none -- a host RUN in fp16 cannot run the lever. This is a run-time caveat
    (run the emulator in bf16 / fp32), not a property of the stored checkpoint: an fp16-stored host loaded
    and run in bf16 / fp32 has the lever (model_graph reads no torch_dtype for it);
  * a post-only norm (OLMo-2 / -3, EXAONE-4: the post-norm is the block's ONLY norm) is not switched off
    by this: the host would be left with no pre-norm at all (frontier-levers);
  * the MLA host still needs an MQA-shaped RoPE target (one shared rotary key head): the lever removes the
    latent-norm wall only.

MEASURED (2026-11-13, hf/data/eps_lever_agreement.jsonl; act_swap standard: the target's own greedy
64-token continuations of 24 prompts; payable = top-1 >= 99% and >= 90% of 32-token continuations identical;
eager attention; a chosen per pair so that rho_max <= 2^-30 on the target's own text): MEASURED below.
Every pair is 100% / 100% in fp32 and fp64, at the rounding floor (max |dlogp| 9e-5 - 3e-3 in fp32, the
same size as eager vs sdpa on these models). In bf16 the host tracks the target's own bf16 run (both fail
the strict standard by themselves: 98.4 - 99.4% top-1, 50 - 83% identical): bf16 does not break the lever.
"""
from __future__ import annotations

# switched-off host norm attribute names (stock transformers modules)
SWITCHED_OFF = ("q_norm", "k_norm", "v_norm", "post_attention_layernorm", "post_feedforward_layernorm",
                "kv_a_layernorm")
HOST_NORM_KINDS = {"qk_norm": ("q_norm", "k_norm"), "post_norm": ("post_attention_layernorm",
                                                                 "post_feedforward_layernorm"),
                   "v_norm": ("v_norm",), "mla_latent_norm": ("kv_a_layernorm",)}
CURRENCY = "free"
# round 3, batch 3b (round3_inputs C W4-1, Lean epsLever_eq): a LIMIT, not exact. Over the reals S sqrt(eps)
# N_eps(1, x / S) = x / sqrt(1 + u) with u = ms(x) / (S^2 eps) > 0 for every x != 0, so no finite S is exact; the fp32
# match (max |dlogp| at the rounding floor) is rounding (fl(ms + e) = e once u < 2^-25), and fp64 misses by u / 2 ~ 1.5e-8
# relative at a_needed(.., RHO_EXACT). The relation already pays it as a limit (payable only: eps_h, epsLever_bound).
EXACT = False
KIND = "limit"

# pair -> {dtype: the primary construction's numbers}; bf16: host and target-bf16 control, both against the
# fp32 target (bf16 fails the strict standard on its own; the lever adds nothing to it)
MEASURED = {
    "qwen3<-smollm2": {
        "off": ("q_norm", "k_norm"), "a": 26, "ms_over_eps": 2.19e6,
        "float32": dict(top1=1.0, ex32=1.0, max_dlogp=9.73e-05),
        "float64": dict(top1=1.0, ex32=1.0, max_dlogp=2.46e-05),
        "bfloat16": dict(top1=0.9857, ex32=0.500, target_top1=0.9844, target_ex32=0.542)},
    "qwen3<-llama3.2": {
        "off": ("q_norm", "k_norm"), "a": 26, "ms_over_eps": 3.42e6,
        "float32": dict(top1=1.0, ex32=1.0, max_dlogp=8.81e-05),
        "float64": dict(top1=1.0, ex32=1.0, max_dlogp=8.25e-06),
        "bfloat16": dict(top1=0.9863, ex32=0.542, target_top1=0.9876, target_ex32=0.708)},
    "gemma2<-gemma1": {
        "off": ("post_attention_layernorm", "post_feedforward_layernorm"), "a": 32, "ms_over_eps": 8.35e9,
        "float32": dict(top1=1.0, ex32=1.0, max_dlogp=3.16e-03),
        "float64": dict(top1=1.0, ex32=1.0, max_dlogp=4.13e-04),
        "bfloat16": dict(top1=0.9896, ex32=0.708, target_top1=0.9896, target_ex32=0.708)},
    "gemma3<-gemma1": {
        "off": ("q_norm", "k_norm", "post_attention_layernorm", "post_feedforward_layernorm"), "a": 32,
        "ms_over_eps": 8.35e9,
        "float32": dict(top1=1.0, ex32=1.0, max_dlogp=2.25e-03),
        "float64": dict(top1=1.0, ex32=1.0, max_dlogp=4.40e-04),
        "bfloat16": dict(top1=0.9896, ex32=0.708, target_top1=0.9896, target_ex32=0.708)},
    "gemma4<-gemma1": {                     # transformers 5.18 (reads gemma-2b's legacy "gelu" as tanh)
        "off": ("q_norm", "k_norm", "v_norm", "post_attention_layernorm", "post_feedforward_layernorm"),
        "a": 32, "ms_over_eps": 8.35e9,
        "float32": dict(top1=1.0, ex32=1.0, max_dlogp=3.04e-03),
        "float64": dict(top1=1.0, ex32=1.0, max_dlogp=4.54e-04),
        "bfloat16": dict(top1=0.9935, ex32=0.833, target_top1=0.9883, target_ex32=0.750)},
    "mla<-gemma1": {
        "off": ("kv_a_layernorm",), "a": 28, "ms_over_eps": 4.03e7,
        "float32": dict(top1=1.0, ex32=1.0, max_dlogp=2.59e-03),
        "float64": dict(top1=1.0, ex32=1.0, max_dlogp=5.78e-04),
        "bfloat16": dict(top1=0.9870, ex32=0.667, target_top1=0.9896, target_ex32=0.708)},
}
# the input-size sweep (fp32): how small the switched-off norm's input must be. rho_max = the largest
# ms(x) / e any switched-off norm saw on the 24 prompts and continuations.
#   rho_max ~ 0.5  (2^-1):  Gemma-2 / -3 hosts pass (99.8%, 96%); Gemma-4 88%, Qwen3 hosts 29%, MLA 79%: fail
#   rho_max <= 2^-5 (3e-2): strict-payable on every pair (top-1 >= 99.93%, >= 96% identical)
#   rho_max <= 2^-10:       100% / 100% on every pair
#   rho_max <= 2^-25:       max |dlogp| at the fp32 rounding floor (fl(ms + e) = e: the norm is exactly linear)
# The input must sit ~2^-12.5 below sqrt(e) in rms to be exact, ~2^-2.5 to be payable; for gemma-2b's MLP
# outputs (ms 8.4e3, e = 1e-6) that is a = 32, values ~1e-8 -- still ~90 binades above fp32's smallest normal.
RHO_PAYABLE = 2.0 ** -5
RHO_EXACT = 2.0 ** -25


def a_needed(ms_max, eps, rho=RHO_EXACT):
    """the input scale-down exponent: 2^-2a ms_max / eps <= rho."""
    import math
    return max(0, math.ceil((math.log2(ms_max / eps) - math.log2(rho)) / 2))


def host_norm_off(kind, eps=1e-6, dtype="float32", ms_max=1e4):
    """Can a host switch off its `kind` norm (qk_norm / post_norm / v_norm / mla_latent_norm) for a target
    that lacks it?  Free when eps > 0 and the scaled tensors stay normal in `dtype`."""
    if kind not in HOST_NORM_KINDS or eps <= 0:
        return False
    a = a_needed(ms_max, eps)
    min_normal_log2 = {"float32": -126, "bfloat16": -126, "float64": -1022, "float16": -14}[dtype]
    # the scaled activations must clear the smallest normal by the activations' own dynamic range
    # (~2^-30 below their max is enough for the sum ms(x) to be unaffected)
    return -a - 30 > min_normal_log2


# What this changes in graph/model_graph.py (proposals; frontier-levers counted the effect, 218 -> 208
# alone, 194 with the window lever):
PROPOSED = {
    "QKNORM_ORDER / QKNORM_REGIME_PAYABLE": "plain target on a qk-norm host: a LIMIT, free (was: same residual "
                                             "as the regime fold). The target side (qk-norm target, plain host) "
                                             "is unchanged: block/fold_agreement.py measures it unpayable.",
    "FEATURES post_norm (host side)": "a sandwich host (pre + post) switches its post-norms off: exact, free. "
                                      "A post-only host (OLMo-2 style) does not.",
    "FEATURES vnorm (host side)": "Gemma-4's weight-free v_norm switches off (o_proj absorbs 2^a sqrt(e)): "
                                  "exact, free.",
    "KV_ORDER MLA (host side)": "the latent norm switches off: exact, free; the shared-rotary-key condition "
                                "(MQA-shaped RoPE target, or NoPE) remains.",
}
