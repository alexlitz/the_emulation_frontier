"""Round 3, batch 1: catalog instances of the reduction types the classification batch added (the new FEATURES' free /
limit sides and the gen value "prefix"); batch 2a: the output-gate host switch-offs and the qk-norm nestings; batch 2b:
the query-temperature and warped-RoPE host switch-offs; batch 2c: Doge's dynamic mask and the eos document mask; batch
2d: the head-side host switch-offs (VHA, scaled XSA, Canon, MoA, the summary memory, A.X-K2's raw q latent, clip_qkv);
batch 3a: the router forms; batch 3b: the SwiGLU clamp's host lever (act_clip) and the beta-free LayerNorm fold on an
RMSNorm host without an lm_head bias; batch 4a: the loop-boundary host switch-offs (Nanbeige4.2's skip_loop_final_norm,
LFM3-RD's step scale at 1); batch 4b: the residual-stream host switch-offs (residual gains / bias at 1 / 0, CFRD's
scalar residual gate at 1/2 folded, backout lambda 0, the n-gram memories' read 0), on the reference decoder (verify/spot/catalog/_ref.py). Each builds the host from the target's weights and runs both on the
same token ids."""
from __future__ import annotations

import math

import torch

from verify.spot.catalog import entry
from verify.spot.catalog._ref import F64, RefLM, quantized
from verify.spot.catalog.ref_slots import REF, _check, _host, _t


def _with_branch(t, seed=3, **cfg):
    """A host: the target's config with cfg on, the target's weights, and the new branch's own weights random."""
    h = _host(t, **cfg)
    tmp = RefLM(**h.c).init(seed)
    for k, v in tmp.p.items():
        if k not in h.p:
            h.p[k] = v
    return h


# ------------------------------------------------------------------ host features switched off

@entry("feature:cross_attn:host", instance=f"decoder-only -> a decoder with per-layer cross-attention to an encoder "
                                           f"memory (Parler-TTS encoder_attn), its out_proj 0 ({REF})")
def r3_cross_attn_off():
    t = _t()
    h = _with_branch(t, cross_attn=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.xo"] = torch.zeros_like(h.p[f"l{i}.xo"])
    return _check(t, h)


@entry("feature:enc_input:host", instance=f"plain token inputs -> an LM whose text ids are a bidirectional encoder's "
                                          f"projected output (Breeze-TTS), the target's tokens fed as the host's "
                                          f"per-token (audio) ids, the encoder unused ({REF})")
def r3_enc_input_off():
    t = _t()
    h = _with_branch(t, enc_input=True)
    return _check(t, h)


@entry("feature:gated_norm:host", instance=f"plain pre-norms -> low-rank sigmoid-gated pre-norms (A.X-K2 / Grug), fc2 = "
                                           f"0 (gate 1/2) and the norm weight doubled ({REF})")
def r3_gated_norm_off():
    t = _t()
    h = _with_branch(t, gated_norm=4)
    for i in range(t.n_param_layers()):
        for n in ("n1", "n2"):
            h.p[f"l{i}.{n}_g2"] = torch.zeros_like(h.p[f"l{i}.{n}_g2"])
            h.p[f"l{i}.{n}_w"] = t.p[f"l{i}.{n}_w"] * 2
    return _check(t, h)


@entry("feature:out_gate_post:host", instance=f"no output gate -> a sigmoid gate after o_proj (MM200), W_g = 0 (1/2) "
                                              f"and o_proj doubled ({REF})")
def r3_out_gate_post_off():
    t = _t()
    h = _with_branch(t, out_gate_post=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.ogp"] = torch.zeros_like(h.p[f"l{i}.ogp"])
        h.p[f"l{i}.o"] = t.p[f"l{i}.o"] * 2
    return _check(t, h)


@entry("feature:scmoe:host", instance=f"a sequential MoE layer -> LongCat's shortcut-connected MoE block with attn1 / "
                                      f"mlp0 / mlp1 zeroed (the MoE reads the first sublayer's output) ({REF})")
def r3_scmoe_off():
    t = _t(E=4, k=2, ew=16, renorm="P", sel_bias=True)
    h = _with_branch(t, scmoe=True)
    for i in range(t.n_param_layers()):
        s2 = f"l{i}s."
        for k in (s2 + "o", s2 + "m0down", s2 + "m1down"):
            h.p[k] = torch.zeros_like(h.p[k])
    return _check(t, h)


# ------------------------------------------------------------------ batch 2a: output-gate hosts switched off
# (Lean out_gate_off: the gate projection -- and its bias -- 0, sigma(0) = 1/2, o_proj doubled)

def _og_off(**cfg):
    t = _t()
    h = _with_branch(t, **cfg)
    for i in range(t.n_param_layers()):
        for k in ("ogate", "ogate_b"):
            if f"l{i}.{k}" in h.p:
                h.p[f"l{i}.{k}"] = torch.zeros_like(h.p[f"l{i}.{k}"])
        h.p[f"l{i}.o"] = t.p[f"l{i}.o"] * 2
    return _check(t, h)


@entry("feature:out_gate_head:host", instance=f"no output gate -> a head-wise sigmoid gate (one per head: LFM3, Step-3.5, "
                                              f"Spark), W_g = 0 (1/2) and o_proj doubled ({REF})")
def r3_out_gate_head_off():
    return _og_off(out_gate="head")


@entry("feature:out_gate_bias:host", instance=f"no output gate -> an elementwise sigmoid gate with a bias (Beam), W_g = 0 "
                                              f"and b = 0 (1/2), o_proj doubled ({REF})")
def r3_out_gate_bias_off():
    return _og_off(out_gate=True, og_bias=True)


@entry("feature:out_gate_head_bias:host", instance=f"no output gate -> a head-wise sigmoid gate with a bias (Ethoric), W_g = "
                                                   f"0 and b = 0 (1/2), o_proj doubled ({REF})")
def r3_out_gate_head_bias_off():
    return _og_off(out_gate="head", og_bias=True)


@entry("feature:out_gate_qlat:host", instance=f"no output gate -> a sigmoid gate of the normed q latent (ERNIE-5.5 o_gate), "
                                              f"o_gate = 0 (1/2) and wo_b doubled ({REF})")
def r3_out_gate_qlat_off():
    return _og_off(out_gate=True, og_qlat=6)


# ------------------------------------------------------------------ batch 2a: qk-norm nestings (a host weight / config)

def _qk_nest(tv, hv, write, **hcfg):
    t = _t(qknorm=tv)
    h = _host(t, qknorm=hv, **hcfg)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        for k in ("qn", "kn"):
            h.p.pop(pre + k, None)
        h.p.update(write(t, pre))
    return _check(t, h)


@entry("slot:qknorm:head_free->head_qfree", instance=f"weight-free qk-norm -> weight-free q / weighted k norm (Domyn-Edge), "
                                                      f"k gain 1 (Lean qkn_gain_one) ({REF})")
def r3_qk_free_qfree():
    return _qk_nest("head_free", "head_qfree", lambda t, pre: {pre + "kn": torch.ones(t.c["hd"], dtype=F64)})


@entry("slot:qknorm:head_free->head_free_gain", instance=f"weight-free qk-norm -> weight-free norm times a config gain "
                                                          f"(nanochat_gpt qk_sharpen_scale, Grug qk_mult), the gain key "
                                                          f"set to 1 ({REF})")
def r3_qk_free_gain():
    return _qk_nest("head_free", "head_free_gain", lambda t, pre: {}, qk_gain=1.0)


@entry("slot:qknorm:head_qfree->head", instance=f"weight-free q / weighted k norm -> per-head gained qk-norm, q gain 1, "
                                                f"k gain the target's ({REF})")
def r3_qk_qfree_head():
    return _qk_nest("head_qfree", "head", lambda t, pre: {pre + "qn": torch.ones(t.c["hd"], dtype=F64),
                                                          pre + "kn": t.p[pre + "kn"].clone()})


@entry("slot:qknorm:head_qfree->head_distinct", instance=f"weight-free q / weighted k norm -> per-head-distinct gains, q "
                                                         f"gains 1, every k head the target's gain ({REF})")
def r3_qk_qfree_distinct():
    return _qk_nest("head_qfree", "head_distinct",
                    lambda t, pre: {pre + "qn": torch.ones(t.c["H"], t.c["hd"], dtype=F64),
                                    pre + "kn": t.p[pre + "kn"].expand(t.c["KV"], t.c["hd"]).clone()})


@entry("port:eps_qknorm", standard="limit",
       instance=f"K = V attention, no norm -> DeepSeek-V4.1's weighted K = V head norm (dsv4_kv), switched off by the eps "
                f"lever: k rows x 2^-40, the gain sqrt(eps) 2^40 ({REF})")
def r3_eps_dsv4_kv():
    return _lever_k(dict(keqv=True), "dsv4_kv", lambda t: t.c["hd"])


@entry("port:eps_qknorm", "slot:qknorm:no->mla_krope", standard="limit",      # (the slot pair is observed from batch
       # 3a on: Mark-SLM-V3, now a sigmoid router, on dots3-note-prev)
       instance=f"no norm -> dots3-note's weighted norm of the rotary key dims (mla_krope), switched off by the eps lever: "
                f"the rotary k rows x 2^-40, the gain sqrt(eps) 2^40 ({REF})")
def r3_eps_mla_krope():
    return _lever_k(dict(rot=4), "mla_krope", lambda t: t.c["rot"])


def _lever_k(cfg, variant, width, s=2.0 ** -40):
    """A plain target on a host whose only q / k norm is on (part of) k: the normed k rows x s, so the norm is linear,
    its gain sqrt(eps) / s (Lean leverAttn_layer_bound, the k flag)."""
    t = _t(**cfg)
    h = _host(t, qknorm=variant)
    hd, KV, w = t.c["hd"], t.c["KV"], width(t)
    g = math.sqrt(t.c["qk_eps"]) / s
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        Wk = t.p[pre + "k"].clone().view(KV, hd, -1)
        Wk[:, :w] *= s
        h.p[pre + "k"] = Wk.reshape(KV * hd, -1)
        if cfg.get("keqv"):
            h.p[pre + "v"] = h.p[pre + "k"]
        h.p[pre + "kn"] = torch.full((w,), g, dtype=F64)
    return _check(t, h, standard="limit", tol=1e-9, note="eps lever, k x 2^-40")


# ------------------------------------------------------------------ target features a host without them reproduces

@entry("feature:fake_quant:target", instance=f"weights ternarised in the forward (BitCPM-CANN LinearQuantizer) -> a "
                                             f"plain host loaded with the quantised matrices q(W) ({REF})")
def r3_fake_quant_target():
    t = _t(fake_quant=True)
    h = _host(t, fake_quant=False)
    h.p = quantized(t.p)
    return _check(t, h)


@entry("feature:post_norm_attn:target", standard="limit",
       instance=f"a post-norm on the attention output only (Needle / IQuest-Q1) -> a full-sandwich host whose MLP post-"
                f"norm is switched off by the eps lever: down x 2^-40, the norm linear, gain sqrt(eps) 2^40 ({REF})")
def r3_post_norm_attn_target():
    s = 2.0 ** -40
    t = _t(post_norm="attn")
    h = _with_branch(t, post_norm="both")
    g = math.sqrt(t.c["eps"]) / s
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.down"] = t.p[f"l{i}.down"] * s
        h.p[f"l{i}.pf_w"] = torch.full_like(h.p[f"l{i}.pf_w"], g)
    return _check(t, h, standard="limit", tol=1e-9, note="eps lever, MLP output x 2^-40")


# ------------------------------------------------------------------ the prefix-LM mask (GEN_ORDER "prefix")
# A prefix LM (DIVE-Doc): the prompt's 10 positions see each other, the generated ones are causal. Each host takes the
# caller's mask (the mask currency, as the other GEN_ORDER entries).

@entry("slot:gen:ar->prefix", instance=f"causal LM -> a prefix-LM host given the causal mask ({REF})")
def r3_gen_ar_prefix():
    t = _t(mask="causal")
    return _check(t, _host(t, mask="prefix:10"), mask_h="causal")


@entry("slot:gen:prefix->diffusion", instance=f"prefix LM (10) -> bidirectional host given the prefix mask ({REF})")
def r3_gen_prefix_diffusion():
    t = _t(mask="prefix:10")
    return _check(t, _host(t, mask="full"), mask_h="prefix:10")


@entry("slot:gen:prefix->block", instance=f"prefix LM (10) -> block-causal host given the prefix mask ({REF})")
def r3_gen_prefix_block():
    t = _t(mask="prefix:10")
    return _check(t, _host(t, mask="block:4"), mask_h="prefix:10")


@entry("slot:gen:block->prefix", instance=f"block-causal (4) -> prefix-LM host given the block mask ({REF})")
def r3_gen_block_prefix():
    t = _t(mask="block:4")
    return _check(t, _host(t, mask="prefix:10"), mask_h="block:4")


@entry("slot:gen:diffusion->prefix", instance=f"bidirectional -> prefix-LM host given an all-ones mask ({REF})")
def r3_gen_diffusion_prefix():
    t = _t(mask="full")
    return _check(t, _host(t, mask="prefix:10"), mask_h="full")


# ------------------------------------------------------------------ batch 2b: position features switched off

@entry("feature:q_temp:host", instance=f"plain attention -> a learned per-head log-position query temperature q (s_h "
                                       f"log(pos + 1) + b_h) (Mistral-Large-4), s_h = 0 and b_h = 1 ({REF})")
def r3_q_temp_off():
    t = _t()
    h = _with_branch(t, q_temp=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.qts"] = torch.zeros_like(h.p[f"l{i}.qts"])
        h.p[f"l{i}.qtb"] = torch.ones_like(h.p[f"l{i}.qtb"])
    return _check(t, h)


@entry("feature:rope_warp:host", instance=f"plain RoPE -> a content-warped RoPE phase t w (1 + m r(t)) + s (t / (T - 1))^2 "
                                          f"with m, s = c tanh(w . mean(x)) (GPT-X2 use_symplectic_rope), the metric "
                                          f"projections 0 ({REF})")
def r3_rope_warp_off():
    t = _t()
    h = _with_branch(t, rope_warp=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.rw_m"] = torch.zeros_like(h.p[f"l{i}.rw_m"])
        h.p[f"l{i}.rw_s"] = torch.zeros_like(h.p[f"l{i}.rw_s"])
    return _check(t, h)


# ------------------------------------------------------------------ batch 2c: mask features switched off

@entry("feature:dyn_mask:host", instance=f"plain attention -> Doge's dynamic mask, a per-key logit bias "
                                         f"exp(A softplus(dt_proj(v_j))) per KV head, A = 0 (bias 1 on every key: a "
                                         f"common shift, renorm_softmax_shift; keep_window_size >= T) ({REF})")
def r3_dyn_mask_off():
    t = _t()
    h = _with_branch(t, dyn_mask=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.dmA"] = torch.zeros_like(h.p[f"l{i}.dmA"])
    return _check(t, h)


@entry("feature:doc_mask:host", instance=f"plain causal attention -> an eos document mask (same-document attention, "
                                         f"Ethoric doc_masking / KLM segments) whose eos id is a spare id no target "
                                         f"token maps to (vocab ignored): one document, the causal mask ({REF})")
def r3_doc_mask_off():
    t = _t()
    return _check(t, _host(t, doc_eos=t.c["V"]))


# ------------------------------------------------------------------ batch 2d: head-side features switched off

def _zero(h, *keys):
    for i in range(h.n_param_layers()):
        for k in keys:
            h.p[f"l{i}.{k}"] = torch.zeros_like(h.p[f"l{i}.{k}"])
    return h


@entry("feature:vha:host", instance=f"plain attention -> ERNIE-5.5's VHA head mix o[h] <- sum_g (I + V U^T)[h, g] o[g] "
                                    f"before o_proj, U = V = 0 (m = I) ({REF})")
def r3_vha_off():
    t = _t()
    return _check(t, _zero(_with_branch(t, vha=2), "vha_U", "vha_V"))


@entry("feature:xsa_scaled:host", instance=f"plain attention -> Limite's scaled XSA y - tanh(alpha_h) (y . v_hat) v_hat "
                                           f"(v_hat the token's own normalised value), alpha_h = 0 ({REF})")
def r3_xsa_scaled_off():
    t = _t()
    return _check(t, _zero(_with_branch(t, xsa="scaled"), "xsa_a"))


@entry("feature:canon:host", instance=f"plain blocks -> Canon layers (residual causal depthwise convs, kernel 3, on the "
                                      f"normed attention and MLP inputs: positions A / C), the taps 0 ({REF})")
def r3_canon_off():
    t = _t()
    return _check(t, _zero(_with_branch(t, canon=3), "cA", "cC"))


@entry("feature:moa:host", instance=f"plain MHA (2 KV heads x 1 query) -> JetMoE's MoA (E = 4 expert q / o pairs, top-2, "
                                    f"softmax over the selected logits, one shared K / V), every expert tied to the "
                                    f"target's q / o and the output bias 0, the router random (Lean tied_moe_is_dense: "
                                    f"the gates sum to 1) ({REF})")
def r3_moa_off():
    t = _t(H=2, KV=2)
    h = _with_branch(t, moa=(4, 2))
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        E_ = h.c["moa"][0]
        h.p[pre + "moa_q"] = t.p[pre + "q"][None].expand(E_, -1, -1).clone()
        h.p[pre + "moa_o"] = t.p[pre + "o"][None].expand(E_, -1, -1).clone()
        h.p[pre + "moa_b"] = torch.zeros_like(h.p[pre + "moa_b"])
    return _check(t, h)


@entry("feature:mem_attn:host", instance=f"plain attention -> a causal summary memory (haru / Bluebird: 2 learned slot "
                                         f"queries pool each 3-token chunk, tokens read earlier chunks' summaries), "
                                         f"read_o = 0 ({REF})")
def r3_mem_attn_off():
    t = _t()
    return _check(t, _zero(_with_branch(t, mem_attn=True), "mo"))


@entry("feature:mla_q_raw:host", instance=f"a q-latent query W_qb RMSNorm(W_qa x) -> A.X-K2's q_gate_proj over "
                                          f"cat(RMSNorm(W_qa x), W_qa x), the raw-latent columns 0 ({REF})")
def r3_mla_q_raw_off():
    t = _t(q_lat=12)
    return _check(t, _zero(_with_branch(t, q_raw=True), "qraw"))


@entry("feature:clip_qkv:host", instance=f"unclamped attention -> OLMo-1's clip_qkv clamp of q / k / v to [-c, c], the "
                                         f"host config lever clip_qkv None (the clamp is not run; Lean clampSym_id) "
                                         f"({REF})")
def r3_clip_qkv_off():
    t = _t()
    h = _host(t, clip_qkv=0.3)
    h.c["clip_qkv"] = None                               # host config lever: clip_qkv None, shape-preserving
    return _check(t, h)


# ------------------------------------------------------------------ batch 3a: MoE and router

_ZG = dict(E=8, k=2, ew=12, score="sigmoid", sel_bias=True, n_group=4, topk_group=2, group_score="top2sum",
           group_fill="zero", renorm="R")


def _neg_bias(t, b=-0.9):
    """Selection biases below -max(sigmoid): every biased score of a kept expert can fall under the 0-filled ones, so
    the fill decides the selection (the case round3_inputs G1 names)."""
    for i in range(t.n_param_layers()):
        t.p[f"l{i}.selb"] = torch.full_like(t.p[f"l{i}.selb"], b) + 0.3 * t.p[f"l{i}.selb"]
    return t


def _wider_experts(t, ew, **cfg):
    """A host with the target's experts zero-padded to width ew (Lean moe_wider_experts), the rest copied."""
    h = _host(t, ew=ew, **cfg)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        for k, ax in (("eg", 1), ("eu", 1), ("ed", 2)):
            W = t.p[pre + k]
            shape = list(W.shape)
            shape[ax] = ew
            out = torch.zeros(shape, dtype=F64)
            sl = [slice(None)] * 3
            sl[ax] = slice(0, W.shape[ax])
            out[tuple(sl)] = W
            h.p[pre + k] = out
    return h


@entry("moe:group:top2sum_zero", instance=f"a 0-filled top-2-sum group limit (E 8 in 4 groups, keep 2, top-2, sigmoid, "
                                          f"selection biases < -1 so the fill binds) -> the same 0-filled router with "
                                          f"the target's experts, wider expert widths zero-padded ({REF})")
def r3_group_zero_fill():
    t = _neg_bias(_t(**_ZG))
    return _check(t, _wider_experts(t, 20))


@entry("moe:group:per_group", instance=f"Intern-S1-Pro's per-group router (E 8 in 4 groups, top-1 inside each: k 4, "
                                       f"softmax, renormalised) -> the same router, wider experts zero-padded ({REF})")
def r3_group_per_group():
    t = _t(E=8, k=4, ew=12, n_group=4, topk_group=4, group_score="per_group", renorm="R")
    return _check(t, _wider_experts(t, 20))


@entry("feature:router_bias_logit:host", instance=f"a bias-free sigmoid router -> a logit-domain selection bias (top-k "
                                                  f"on l + b, Grug / Kolibri), b = 0 (Lean router_bias_off) ({REF})")
def r3_router_bias_logit_off():
    t = _t(E=4, k=2, ew=16, score="sigmoid")
    h = _with_branch(t, sel_bias=True, sel_domain="logit")
    return _check(t, _zero(h, "selb"))


@entry("feature:block_routing:host", instance=f"a per-token sigmoid router with a selection bias -> LLaDA2.2's block "
                                             f"routing (blocks of 4 tokens), the host config lever expert_capacity = "
                                             f"num_experts (every expert allowed; Lean topk_all) ({REF})")
def r3_block_routing_off():
    t = _t(E=4, k=2, ew=16, score="sigmoid", sel_bias=True)
    h = _host(t, blk=(4, 2))
    h.c["blk"] = (4, t.c["E"])                           # host config lever: expert_capacity = num_experts
    return _check(t, h)


@entry("feature:par_moe_res:host", instance=f"a sequential dense GLU block -> Arctic's block (MoE beside attention on "
                                           f"the layer input + a residual MLP on the post-attention stream): the "
                                           f"experts' down-projections 0, the residual MLP and its norm the target's "
                                           f"FFN ({REF})")
def r3_par_moe_res_off():
    t = _t()
    h = _with_branch(t, par_moe_res=True, E=4, k=2, ew=16)
    for i in range(t.n_param_layers()):
        pre, r2 = f"l{i}.", f"l{i}r."
        for k in ("up", "fgate", "down"):
            h.p[r2 + k] = t.p[pre + k].clone()
        h.p[pre + "nr_w"] = t.p[pre + "n2_w"].clone()
    return _check(t, _zero(h, "ed"))


# ------------------------------------------------------------------ batch 3b: FFN widths, activation clamps, spot fixes

@entry("feature:act_clip:host", instance=f"an unclamped SwiGLU -> DeepSeek-V4's clamped SwiGLU (gate <= L, |up| <= L), "
                                         f"the host config lever swiglu_limit = inf (clamp(x, -inf, inf) = x; Lean "
                                         f"act_clip_regime) ({REF})")
def r3_act_clip_off():
    t = _t()
    h = _host(t, act_clip=0.3)
    h.c["act_clip"] = float("inf")                       # host config lever: swiglu_limit = inf, shape-preserving
    return _check(t, h)


@entry("slot:norm:LayerNorm->RMSNorm", instance=f"a beta-FREE LayerNorm target (OLMo-1's non-parametric LayerNorm, MPT "
                                               f"no_bias: no betas, no linear biases) -> an RMSNorm host WITHOUT an "
                                               f"lm_head bias: every residual writer centred, nothing to fold (Lean "
                                               f"layerNorm_sub_rmsNorm with beta = 0) ({REF})")
def r3_ln_fold_beta_free():
    """The fold that backs every headline LayerNorm <- RMSNorm edge on a host without an lm_head bias (round 3, batch
    3b, spot_comp_removed: all of them are beta-free targets -- OLMo-1, MPT, replit): with the stream's mean 0, LN = the
    RMS norm of the same eps, and with beta = 0 no constant has to reach the logits."""
    t = _t(norm="ln", ln_bias=False, ffn="vanilla", act="gelu")
    h = _host(t, norm="rms", lm_bias=False)
    P = h.p
    cen = lambda W: W - W.mean(0, keepdim=True)
    P["emb"] = t.p["emb"] - t.p["emb"].mean(1, keepdim=True)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        P[pre + "o"], P[pre + "down"] = cen(t.p[pre + "o"]), cen(t.p[pre + "down"])
    assert not any(k.endswith("_b") for k in t.p) and "lm_b" not in h.p
    return _check(t, h)


# ------------------------------------------------------------------ batch 4a: loop boundaries switched off

@entry("feature:loop_bnorm_cfg:host", instance=f"a looped block (2 layers x 2) -> Nanbeige4.2's loop that renorms the stream "
                                                f"by the shared final norm after every pass, the host config lever "
                                                f"skip_loop_final_norm = true (the norm once at the end) ({REF})")
def r3_loop_bnorm_off():
    t = _t(loop=(2, 2), L=4)
    h = _host(t, loop_bnorm=True)
    h.c["skip_loop_final_norm"] = True                   # host config lever: a norm on/off flag, shape-preserving
    return _check(t, h)


@entry("feature:rd_controller:host", instance=f"4 untied layers -> LFM3-RD's core (positions 1-2) whose summed update "
                                             f"is rescaled by a learned step scale before the coda, the scale set to 1: "
                                             f"p + 1 (h - p) = h (scalar_fold at s = 1) ({REF})")
def r3_rd_controller_off():
    t = _t(L=4)
    h = _with_branch(t, rd_scale=(1, 3))
    h.p["rd_s"] = torch.ones_like(h.p["rd_s"])
    return _check(t, h)


# ------------------------------------------------------------------ batch 4b: residual streams and extra inputs

@entry("feature:res_gain:host", instance=f"plain residual -> per-channel residual gains before each merge (Doge "
                                         f"input_residual / post_attention_residual), g = 1: the identity setting ({REF})")
def r3_res_gain_off():
    t = _t()
    h = _with_branch(t, res_gain=True)
    for i in range(t.n_param_layers()):
        for k in (1, 2):
            h.p[f"l{i}.rg{k}"] = torch.ones_like(h.p[f"l{i}.rg{k}"])
    return _check(t, h)


@entry("feature:res_bias:host", instance=f"plain residual -> ZAYA1-VL's ResidualScaling (r + b) (.) s before each merge, "
                                         f"s = 1 and b = 0 (bias_off) ({REF})")
def r3_res_bias_off():
    t = _t()
    h = _with_branch(t, res_gain=True, res_bias=True)
    for i in range(t.n_param_layers()):
        for k in (1, 2):
            h.p[f"l{i}.rg{k}"] = torch.ones_like(h.p[f"l{i}.rg{k}"])
            h.p[f"l{i}.rb{k}"] = torch.zeros_like(h.p[f"l{i}.rb{k}"])
    return _check(t, h)


@entry("feature:res_gate:host", instance=f"ungated residual updates -> CFRD's scalar residual gate x + c sigmoid(w . u + "
                                         f"b) f(u) on both sublayers, w = 0, b = 0 (gate 1/2) and o / down x 2 / c "
                                         f"(out_gate_off, scalar_fold) ({REF})")
def r3_res_gate_off():
    t = _t()
    h = _with_branch(t, res_gate=True)
    c = h.c["rg_c"]
    for i in range(t.n_param_layers()):
        for k in (1, 2):
            h.p[f"l{i}.rgw{k}"] = torch.zeros_like(h.p[f"l{i}.rgw{k}"])
            h.p[f"l{i}.rgb{k}"] = torch.zeros_like(h.p[f"l{i}.rgb{k}"])
        h.p[f"l{i}.o"] = t.p[f"l{i}.o"] * (2 / c)
        h.p[f"l{i}.down"] = t.p[f"l{i}.down"] * (2 / c)
    return _check(t, h)


@entry("feature:backout:host", instance=f"4 layers -> the backout path x_L - lambda x_m before the final norm (Gaggle / "
                                        f"nanochat_gpt), lambda = 0 (extra_input_off) ({REF})")
def r3_backout_off():
    t = _t(L=4)
    h = _with_branch(t, backout=True)
    h.p["bo"] = torch.zeros_like(h.p["bo"])
    return _check(t, h)


@entry("feature:engram_qwen4:host", "feature:engram_allgdn:host", "feature:engram_needle:host",
       instance=f"no n-gram memory -> a gated n-gram memory with a causal depthwise conv on every layer, sigmoid(<rms(x), "
                f"rms(K e)> / sqrt(d)) V e + conv(.) (AllGDN Engram, Qwen4-exp's PLE, needle's Engram), V = 0: the gate "
                f"times 0 and the conv of 0 add 0 (extra_input_off) ({REF})")
def r3_engram_gate_off():
    t = _t()
    h = _with_branch(t, engram_gate=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.ge_v"] = torch.zeros_like(h.p[f"l{i}.ge_v"])
    return _check(t, h)


@entry("feature:engram_crowd:host", instance=f"no n-gram memory -> Crowd's bucket table of (previous, current) token added "
                                             f"to the residual, the table 0 (extra_input_off) ({REF})")
def r3_engram_crowd_off():
    t = _t()
    h = _with_branch(t, engram=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.eng"] = torch.zeros_like(h.p[f"l{i}.eng"])
    return _check(t, h)


# ------------------------------------------------------------------ round 3, nonlm_heads: the readout rule
# (graph.model_graph HEAD_ORDER: the output read as fixed coordinates at fixed positions, like decoding reads logits).
# The target is the reference body with its head in place of the lm_head: Nf(x) @ W^T (+ b), W of the head's width;
# the host is an UNTIED reference LM with the target's body and an lm_head whose first rows are W (its other rows the
# target's own vocabulary rows), read at those rows.

def _head_target(t, W, b=None):
    s = _host(t)
    s.p["lm"] = W
    if b is not None:
        s.p["lm_b"] = b
    return s


def _readout_host(t, W, b=None):
    h = _host(t)
    h.p["lm"] = torch.cat([W, t.p["lm"][W.shape[0]:]], 0)
    if b is not None:
        h.p["lm_b"] = torch.cat([b, torch.zeros(t.c["V"] - W.shape[0], dtype=F64)])
    return h


@entry("slot:head:score->lm", instance=f"a score head Linear(d, 1) after the final norm, under the final softcap (the "
                                       f"BGE rerankers' LayerWiseHead / CostWiseHead at the default exit L) -> an untied "
                                       f"LM, lm_head row 0 := the score head, read at that row ({REF})")
def r3_head_score_lm():
    from verify.spot.catalog import compare
    t = _t(softcap_final=30.0)
    g = torch.Generator().manual_seed(5)
    W = torch.randn(1, t.c["d"], dtype=F64, generator=g)
    tgt, h = _head_target(t, W), _readout_host(t, W)
    return compare(lambda x: tgt(x), lambda x: h(x)[..., :1], t.c["V"], new_tokens=0,
                   note="the score at every position (the caller reads the last)")


@entry("slot:head:embedding->lm", instance=f"a last-token embedding without L2: a Dense(d, 6) with a bias on the final-"
                                           f"normed last token -> an untied LM with an lm_head bias (bias_out), lm_head "
                                           f"rows 0..5 := the Dense, read at the last position ({REF})")
def r3_head_embedding_lm():
    from verify.spot.catalog import compare
    t = _t()
    g = torch.Generator().manual_seed(6)
    W, b = torch.randn(6, t.c["d"], dtype=F64, generator=g), torch.randn(6, dtype=F64, generator=g)
    tgt, h = _head_target(t, W, b), _readout_host(t, W, b)
    return compare(lambda x: tgt(x)[:, -1:], lambda x: h(x)[:, -1:, :6], t.c["V"], new_tokens=0,
                   note="the last token's embedding")


@entry("feature:exit_heads:host", instance=f"a plain LM -> a host with an exit head after layer 1 (norm(x after 1 "
                                           f"layer) through its own head, output beside the final logits), read at the "
                                           f"final logits only ({REF})")
def r3_exit_heads_off():
    from verify.spot.catalog import compare
    t = _t(L=3)
    h = _host(t)
    ex = _host(t, L=1)                                   # the first layer and the exit's own norm / head
    g = torch.Generator().manual_seed(7)
    ex.p["lm"] = torch.randn(1, t.c["d"], dtype=F64, generator=g)
    ex.p["nf_w"] = torch.rand(t.c["d"], dtype=F64, generator=g) + 0.5
    host = lambda x: torch.cat([h(x), ex(x)], -1)       # the host's outputs: the final logits, then the exit score
    V = t.c["V"]
    return compare(lambda x: t(x), lambda x: host(x)[..., :V], V)


# ------------------------------------------------------------------ round 3 (nope_rope_limit): the linear-scaling lever

_NOPE_KW = dict(vocab_size=61, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, max_position_embeddings=128, rms_norm_eps=1e-5,
                tie_word_embeddings=False, attention_bias=False, hidden_act="silu", pad_token_id=0, bos_token_id=1,
                eos_token_id=2)


def _nope_pair(factor, seed=0):
    """A stock NoPE decoder (SmolLM3, no_rope_layers all 0: every layer skips the rotary, modeling_smollm3.py:147,
    :171) and a stock Llama host with the SAME weights whose only change is its config rope_scaling {"rope_type":
    "linear", "factor": factor} (inv_freq / factor, modeling_rope_utils.py:166-175), both fp64 eager."""
    from transformers import LlamaConfig, LlamaForCausalLM, SmolLM3Config, SmolLM3ForCausalLM
    tc = SmolLM3Config(no_rope_layers=[0] * _NOPE_KW["num_hidden_layers"], use_sliding_window=False, **_NOPE_KW)
    tc._attn_implementation = "eager"
    torch.manual_seed(seed)
    tgt = SmolLM3ForCausalLM(tc).to(F64).eval()
    with torch.no_grad():
        for n, p in tgt.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0, 0.5)
    hc = LlamaConfig(rope_scaling={"rope_type": "linear", "factor": factor}, **_NOPE_KW)
    hc._attn_implementation = "eager"
    host = LlamaForCausalLM(hc).to(F64).eval()
    missing, unexpected = host.load_state_dict(tgt.state_dict(), strict=False)
    assert not unexpected and all("rotary" in k for k in missing), (missing, unexpected)
    return tgt, host


@entry("lever:nope_rope_limit", standard="limit",
       instance="NoPE target (stock SmolLM3, no_rope_layers all 0) -> stock Llama host, the target's weights, config "
                "rope_scaling linear, factor 1e12 (every angle t w_p / s -> 0; Lean rope_scale_limit_identity); the "
                "error falls with s (factor 1 rotates: the control; 1e6 in the note) to the rounding floor -- the eager "
                "softmax runs in fp32, so from ~1e11 on the angle perturbation is below its rounding and the logits match")
def r3_nope_rope_limit():
    from verify.spot.catalog import compare, hf_fn
    errs = {}
    for s in (1.0, 1e6):
        tgt, host = _nope_pair(s)
        ids = torch.randint(0, _NOPE_KW["vocab_size"], (1, 24), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            errs[s] = (tgt(ids).logits - host(ids).logits).abs().max().item()
    tgt, host = _nope_pair(1e12)
    return compare(hf_fn(tgt), hf_fn(host), _NOPE_KW["vocab_size"], standard="limit", tol=1e-9,
                   note=f"linear factor 1e12; max |dlogit| at factor 1 {errs[1.0]:.2e} (rotating), 1e6 {errs[1e6]:.2e}")


# ------------------------------------------------------------------ negative controls: the branch is live
# (a construction that only "works" because the branch does nothing would prove nothing)

def branch_is_live():
    """Each new branch, left at its random weights, changes the logits."""
    out = {}
    t = _t()
    for name, cfg in (("cross_attn", dict(cross_attn=True)), ("gated_norm", dict(gated_norm=4)),
                      ("out_gate_post", dict(out_gate_post=True)), ("post_norm", dict(post_norm="attn")),
                      ("fake_quant", dict(fake_quant=True)), ("out_gate_head", dict(out_gate="head")),
                      ("out_gate_bias", dict(out_gate=True, og_bias=True)),
                      ("out_gate_head_bias", dict(out_gate="head", og_bias=True)),
                      ("out_gate_qlat", dict(out_gate=True, og_qlat=6)),
                      ("q_temp", dict(q_temp=True)), ("rope_warp", dict(rope_warp=True)),
                      ("dyn_mask", dict(dyn_mask=True))):
        h = _with_branch(t, **cfg)
        ids = torch.randint(0, t.c["V"], (1, 10), generator=torch.Generator().manual_seed(1))
        out[name] = (t(ids) - h(ids)).abs().max().item()
    te = _t(enc_input=True)
    ids = torch.randint(0, te.c["V"], (1, 10), generator=torch.Generator().manual_seed(1))
    ids_t = ids.clone()
    ids_t[0, 2:5] = te.c["V"] + torch.tensor([1, 4, 7])                  # three text positions
    ids_u = ids_t.clone()
    ids_u[0, 4] = te.c["V"] + 9                                          # a LATER text id changes
    out["enc_input"] = (te(ids_t)[0, 2] - te(ids_u)[0, 2]).abs().max().item()   # position 2 reads it (bidirectional)
    ts = _t(E=4, k=2, ew=16, renorm="P", sel_bias=True)
    out["scmoe"] = (ts(ids) - _with_branch(ts, scmoe=True)(ids)).abs().max().item()
    # batch 2a: the k gain of head_qfree and the config gain of head_free_gain are live
    tq = _t(qknorm="head_qfree")
    out["head_qfree"] = (tq(ids) - _host(tq, qknorm="head_free")(ids)).abs().max().item()
    tg = _t(qknorm="head_free")
    out["head_free_gain"] = (tg(ids) - _host(tg, qknorm="head_free_gain", qk_gain=1.44)(ids)).abs().max().item()
    tk = _t(keqv=True)
    out["dsv4_kv"] = (tk(ids) - _with_branch(tk, qknorm="dsv4_kv")(ids)).abs().max().item()
    tr = _t(rot=4)
    out["mla_krope"] = (tr(ids) - _with_branch(tr, qknorm="mla_krope")(ids)).abs().max().item()
    tp = _t(mask="prefix:10")
    out["prefix"] = (tp(ids, mask="prefix:10") - tp(ids, mask="causal")).abs().max().item()
    # batch 2b: the warped phase reads LATER tokens (an earlier position's logits move when a later token changes)
    tw = _with_branch(t, rope_warp=True)
    ids2 = ids.clone()
    ids2[0, -1] = (ids[0, -1] + 1) % t.c["V"]
    out["rope_warp_noncausal"] = (tw(ids)[0, 2] - tw(ids2)[0, 2]).abs().max().item()
    # batch 2c: an eos id that occurs in the input splits the documents (the mask is live)
    out["doc_mask"] = (t(ids) - _host(t, doc_eos=int(ids[0, 3]))(ids)).abs().max().item()
    # batch 2d: the head-side branches at random weights, the unit XSA, the MoA's untied experts, the raw q latent, a
    # clamp the activations reach
    for name, cfg in (("vha", dict(vha=2)), ("xsa_scaled", dict(xsa="scaled")), ("xsa_unit", dict(xsa="unit")),
                      ("canon", dict(canon=3)), ("mem_attn", dict(mem_attn=True))):
        out[name] = (t(ids) - _with_branch(t, **cfg)(ids)).abs().max().item()
    tm = _t(H=2, KV=2)
    out["moa"] = (tm(ids) - _with_branch(tm, moa=(4, 2))(ids)).abs().max().item()
    hm = _with_branch(tm, moa=(4, 2))                    # tied experts, but the output bias left on
    for i in range(tm.n_param_layers()):
        pre = f"l{i}."
        hm.p[pre + "moa_q"] = tm.p[pre + "q"][None].expand(4, -1, -1).clone()
        hm.p[pre + "moa_o"] = tm.p[pre + "o"][None].expand(4, -1, -1).clone()
    out["moa_bias"] = (tm(ids) - hm(ids)).abs().max().item()
    tl = _t(q_lat=12)
    out["mla_q_raw"] = (tl(ids) - _with_branch(tl, q_raw=True)(ids)).abs().max().item()
    out["clip_qkv"] = (t(ids) - _host(t, clip_qkv=0.3)(ids)).abs().max().item()
    # batch 3a: the 0 fill binds (vs -inf at the same weights), the per-group router is no plain top-k, a nonzero
    # logit-domain bias is no score-domain one, a capacity below E drops experts, the Arctic MoE branch adds
    tz = _neg_bias(_t(**_ZG))
    out["group_fill_zero"] = (tz(ids) - _host(tz, group_fill="-inf")(ids)).abs().max().item()
    tg = _t(E=8, k=4, ew=12, n_group=4, topk_group=4, group_score="per_group", renorm="R")
    out["per_group"] = (tg(ids) - _host(tg, n_group=0, topk_group=0)(ids)).abs().max().item()
    tb = _t(E=4, k=2, ew=16, score="sigmoid", sel_bias=True)
    out["router_bias_logit"] = (tb(ids) - _host(tb, sel_domain="logit")(ids)).abs().max().item()
    out["block_routing"] = (tb(ids) - _host(tb, blk=(4, 2))(ids)).abs().max().item()
    out["par_moe_res"] = (t(ids) - _with_branch(t, par_moe_res=True, E=4, k=2, ew=16)(ids)).abs().max().item()
    # batch 3b: a SwiGLU clamp the activations reach; the LayerNorm centring (the beta-free fold needs the centred writers)
    out["act_clip"] = (t(ids) - _host(t, act_clip=0.3)(ids)).abs().max().item()
    tn = _t(norm="ln", ln_bias=False, ffn="vanilla", act="gelu")
    out["ln_centre"] = (tn(ids) - _host(tn, norm="rms")(ids)).abs().max().item()
    # batch 4a: the between-loop norm (flag not set) and a learned step scale move the logits
    tl4 = _t(loop=(2, 2), L=4)
    out["loop_bnorm"] = (tl4(ids) - _host(tl4, loop_bnorm=True)(ids)).abs().max().item()
    t4 = _t(L=4)
    out["rd_controller"] = (t4(ids) - _with_branch(t4, rd_scale=(1, 3))(ids)).abs().max().item()
    # batch 4b: the residual gains / bias / gate, the backout path and the gated n-gram memory at their random weights
    for name, cfg in (("res_gain", dict(res_gain=True)), ("res_bias", dict(res_gain=True, res_bias=True)),
                      ("res_gate", dict(res_gate=True)), ("engram_gate", dict(engram_gate=True))):
        out[name] = (t(ids) - _with_branch(t, **cfg)(ids)).abs().max().item()
    out["backout"] = (t4(ids) - _with_branch(t4, backout=True)(ids)).abs().max().item()
    hb = _with_branch(t, res_gain=True, res_bias=True)        # the bias alone (gains 1) is live
    for i in range(t.n_param_layers()):
        for k in (1, 2):
            hb.p[f"l{i}.rg{k}"] = torch.ones_like(hb.p[f"l{i}.rg{k}"])
    out["res_bias_only"] = (t(ids) - hb(ids)).abs().max().item()
    hg = _with_branch(t, res_gate=True)                       # the gate at w = 0 but o / down unscaled: 0.35 x f
    for i in range(t.n_param_layers()):
        for k in (1, 2):
            hg.p[f"l{i}.rgw{k}"] = torch.zeros_like(hg.p[f"l{i}.rgw{k}"])
            hg.p[f"l{i}.rgb{k}"] = torch.zeros_like(hg.p[f"l{i}.rgb{k}"])
    out["res_gate_unscaled"] = (t(ids) - hg(ids)).abs().max().item()
    return out
