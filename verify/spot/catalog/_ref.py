"""RefLM: a small reference decoder (fp64, pure torch) with a switch for every slot value the relation names, for the
reduction types no stock transformers class instantiates on both sides. A block is a stock pre-norm block:

  x = emb[ids] (+ pos_table[t])
  sequential:  x += Attn(N1(x));  x += FFN(N2(x))          parallel:  x += Attn(N1(x)) + FFN(N1(x))
  logits = cap_f(lm_head(Nf(x)) + lm_bias)

Attention: q / k / v (+ biases), a qk-norm variant, RoPE on the first `rot` dims of each head (rotate_half pairs
(i, i + rot / 2), the inv_freq table explicit and settable), ALiBi slopes, an additive bias table, a softcap,
per-head sinks (a null key with a learned logit), a window, a caller-supplied mask (causal / full / block), an output
gate, K = V; FFN: none (round 3), vanilla / GLU with any activation, or MoE (softmax / sigmoid / sqrtsoftplus scores, renorm or not,
a selection-only bias, group-limited top-k by max or top-2-sum, a shared expert with an optional sigmoid gate, an
expert-input latent); layers may be recurrent (a gated linear recurrence: the host's non-attention layers), share the
previous layer's K / V (CLA), or loop (a block of b layers K times). Extra host branches (value embeddings, a per-layer
embedding input, an n-gram memory, self-conditioning, a switched LoRA) are added outputs a weight setting turns off.
Round 3, batch 2d adds the head-side branches: a fixed head mix (VHA), the self-value projection removal (XSA), Canon
convs, routed attention experts (MoA), a summary memory, a q-latent query (+ a raw-latent term) and the clip_qkv clamp.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def act_fn(name):
    if name == "relu":
        return F.relu
    if name == "relu2":
        return lambda x: F.relu(x) ** 2
    if name == "silu":
        return F.silu
    if name == "gelu":
        return lambda x: F.gelu(x)
    if name == "gelu_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "mish":
        return F.mish
    if name == "xielu":                           # Apertus xIELU, alpha_p = alpha_n = 0.8, beta = 0.5
        ap, an, b, eps = 0.8, 0.8, 0.5, -1e-6
        return lambda x: torch.where(x > 0, ap * x * x + b * x, (torch.expm1(torch.clamp(x, max=eps)) - x) * an + b * x)
    raise KeyError(name)


def swigluoai(g, u, limit=7.0, alpha=1.702):
    g = g.clamp(max=limit)
    u = u.clamp(-limit, limit)
    return (u + 1) * g * torch.sigmoid(alpha * g)


def alibi_slopes(H):
    p = 2 ** math.floor(math.log2(H))
    base = [2 ** (-8 * (i + 1) / p) for i in range(p)]
    extra = [2 ** (-4 * (2 * i + 1) / p) for i in range(H - p)]
    return torch.tensor(base + extra, dtype=F64)


def t5_bucket(rel, n=8, maxd=32):
    """T5's causal relative-position bucket of i - j >= 0."""
    r = rel.clamp(min=0)
    exact = n // 2
    big = exact + (torch.log(r.clamp(min=1).to(F64) / exact) / math.log(maxd / exact) * (n - exact)).long()
    return torch.where(r < exact, r, big.clamp(max=n - 1))


DEFAULT = dict(V=61, d=32, L=2, H=4, KV=2, hd=8, F=48, norm="rms", eps=1e-5, pos="rope", rot=None, theta=10000.0,
               Tmax=64, ffn="glu", act="silu", block="seq", window=None, qknorm="no", qk_eps=1e-6, softcap_attn=None,
               softcap_final=None, sink=False, qkv_bias=False, out_bias=False, ln_bias=True, lm_bias=False, tied=False,
               mask="causal", block_size=4, loop=None, mtp=False, kinds=None, cla=None, out_gate=False, keqv=False,
               E=0, k=1, ew=None, renorm="R", score="softmax", sel_bias=False, n_group=0, topk_group=0,
               group_score="max", shared=0, shared_gate=False, latent=0, dense_layers=(), route_scale=1.0,
               scale=None, value_emb=False, ple=False, engram=False, selfcond=False, switch_lora=False,
               dsa_topk=None, emb_scale=1.0, ebias=False,
               # round 3, batch 1: post-sublayer norms ("attn": the attention output only, "both": a sandwich), a
               # per-layer cross-attention to a fixed encoder memory, a low-rank sigmoid gate on the pre-norms (its
               # rank), an output gate after o_proj, LongCat's shortcut-connected MoE block, ternary weights
               # quantised in the forward, an encoder-fed text input (text ids V .. V + n_text)
               post_norm="no", cross_attn=False, gated_norm=0, out_gate_post=False, scmoe=False,
               fake_quant=False, enc_input=False, n_text=13, n_mem=6,
               # round 3, batch 2a: the output gate's forms (out_gate True = one gate per channel, "head" = one per
               # head broadcast over head_dim; a bias inside the gate; its activation; a gate read from a normed
               # q latent of rank og_qlat, ERNIE-5.5) and the qk-norm config gain (head_free_gain: rms(q) * qk_gain)
               og_bias=False, og_act="sigmoid", og_qlat=0, qk_gain=1.0,
               # round 3, batch 2b: a learned per-head log-position query temperature q (s_h log(pos + 1) + b_h)
               # (Mistral-Large-4), and a content-warped RoPE phase whose deltas c tanh(w . z) read the mean of the
               # WHOLE sequence's attention input z (GPT-X2 use_symplectic_rope: non-causal), radial base rw_base
               q_temp=False, rope_warp=False, rw_c=0.5, rw_base=16.0,
               # round 3, batch 2c: Doge's dynamic mask (a per-key logit bias exp(A softplus(dt_proj(v_j))) per KV head,
               # modeling_doge.py:303-358; keep_window_size >= T, so no top-k), and an eos document mask (the token id
               # doc_eos closes a document: same-document attention, Ethoric doc_masking / KLM segments)
               dyn_mask=False, doc_eos=None,
               # round 3, batch 2d: ERNIE-5.5's VHA head mix (its rank; 0 = off), XSA (the self-value projection
               # removal: "unit" = factor 1, "scaled" = tanh(alpha_h) per head), Canon causal depthwise convs (their
               # kernel; positions A / C), JetMoE's MoA ((E, k): routed expert q / o pairs over one shared K / V), a
               # causal summary memory (mem_attn: chunk mem_chunk, mem_slots learned slot queries), a q-latent query
               # (q_lat: its rank; q_raw adds A.X-K2's un-normed latent term) and OLMo-1's clip_qkv clamp
               vha=0, xsa=None, xsa_eps=1e-6, canon=0, moa=None, mem_attn=False, mem_chunk=3, mem_slots=2,
               q_lat=0, q_raw=False, clip_qkv=None,
               # round 3, batch 3a: the router forms -- the dropped groups' fill ("-inf", or "zero": DeepSeek-V3's first-
               # release modeling_deepseek.py / A.X-K2 fill the biased scores with 0.0), group_score "per_group"
               # (Intern-S1-Pro: top-(k / G) inside every group), the selection bias's domain ("score": top-k on
               # score + b; "logit": on logit + b, Grug / Kolibri), LLaDA2.2's block routing ((block size, capacity):
               # per block only the capacity experts with the best block-max selection score), and Arctic's parallel
               # MoE + residual MLP block (out = h + res_mlp(n_r h) + moe(n2 x), h = x + attn(n1 x))
               group_fill="-inf", sel_domain="score", blk=None, par_moe_res=False,
               # round 3, batch 3b: DeepSeek-V4's SwiGLU clamp (swiglu_limit L: gate.clamp(max=L) before the activation,
               # up.clamp(-L, L); None = no clamp, inf = the host config lever's off value)
               act_clip=None,
               # round 3, batch 4a: the loop boundaries -- the shared final norm ("nf") after every loop pass but the
               # last (Nanbeige4.2 / qwen3loop num_loops > 1) unless skip_loop_final_norm (its host config lever), and
               # LFM3-RD's step scale (rd_scale = (a, b): after order position b - 1 the stream is x_a + s (x - x_a), x_a
               # the stream entering position a, s the learned rd_s)
               loop_bnorm=False, skip_loop_final_norm=False, rd_scale=None,
               # round 3, batch 4b: the residual-stream branches -- a per-channel gain (res_gain: Doge input_residual /
               # post_attention_residual) and bias (res_bias: ZAYA1-VL ResidualScaling (r + b) s) on the residual before
               # each merge; CFRD's data-dependent scalar residual gate x + rg_c sigmoid(w . u + b) f(u), u the normed
               # sublayer input (res_gate); the backout path x_L - lambda x_m before the final norm, m = half the
               # unrolled depth (backout: Gaggle / nanochat_gpt); a gated n-gram memory with a causal depthwise conv,
               # sigmoid(<rms(x), rms(K e)> / sqrt(d)) V e + conv(.) on the bigram row e (engram_gate: AllGDN / Qwen4-exp
               # / needle)
               res_gain=False, res_bias=False, res_gate=False, rg_c=0.7, backout=False, engram_gate=False,
               # fr-resnorm: hyper-connections (hc = (n streams, variant)) in the three stock forms -- "dsv4"
               # (DeepSeek-V4 / GLM-5-Next: pre = sigmoid + hc_eps, post = 2 sigmoid, comb = Sinkhorn(softmax +
               # hc_eps) with hc_eps in every normaliser, In @ comb; "glm5_next" reads comb^T and a mean head), "hy4"
               # (Hy4: no comb, post = hc_mag sigmoid + hc_eps) and "qwen4" (Qwen4-exp's gated residual: mean_i of
               # sigmoid-gated per-stream (1 + w) RMSNorms as the sublayer input, injection 2 sigmoid(.) / n); the
               # dynamic projections read the RMS-normed flattened streams. selfcond_norm: DiffusionGemma's always-on
               # weight-free RMSNorm of the input embedding (eps = the config eps, the zero self-conditioning signal)
               hc=None, hc_eps=1e-6, hc_iters=20, hc_mag=2.0, hc_low=4, selfcond_norm=False)


class RefLM:
    def __init__(self, **cfg):
        c = dict(DEFAULT)
        c.update(cfg)
        c["rot"] = c["hd"] if c["rot"] is None and c["pos"] == "rope" else (c["rot"] or 0)
        if c["ew"] is None:
            c["ew"] = c["F"]
        self.c = c
        self.p = {}
        self.inv_freq = 1.0 / (c["theta"] ** (torch.arange(0, c["rot"], 2, dtype=F64) / max(c["rot"], 1)))
        self.slopes = alibi_slopes(c["H"])

    # ------------------------------------------------------------------ parameters
    def init(self, seed=0, s=0.25):
        g = torch.Generator().manual_seed(seed)
        c, p = self.c, self.p
        r = lambda *shape, sc=s: torch.randn(*shape, generator=g, dtype=F64) * sc
        d, H, KV, hd, V = c["d"], c["H"], c["KV"], c["hd"], c["V"]
        p["emb"] = r(V, d, sc=1.0)
        p["lm"] = p["emb"] if c["tied"] else r(V, d, sc=0.5)
        if c["lm_bias"]:
            p["lm_b"] = r(V)
        if c["pos"] == "learned_abs":
            p["pos"] = r(c["Tmax"], d, sc=0.5)
        if c["pos"] in ("additive_bias",):
            p["abias"] = r(H, c["Tmax"], c["Tmax"], sc=0.5)
        if c["pos"] == "t5":
            p["t5"] = r(H, 8, sc=0.5)
        p["nf_w"] = 1 + r(d, sc=0.1)
        if c["norm"] == "ln" and c["ln_bias"]:
            p["nf_b"] = r(d, sc=0.1)
        for i in range(self.n_param_layers()):
            pre = f"l{i}."
            for n in ("n1", "n2"):
                p[pre + n + "_w"] = 1 + r(d, sc=0.1)
                if c["norm"] == "ln" and c["ln_bias"]:
                    p[pre + n + "_b"] = r(d, sc=0.1)
            kind = self.kind(i)
            if kind == "rec":
                p[pre + "rin"] = r(2 * d, d)
                p[pre + "rdec"] = r(d, d)
                p[pre + "rout"] = r(d, d)
            else:
                if c["q_lat"]:                            # q = W_qb RMSNorm(W_qa x) (+ W_raw W_qa x: A.X-K2)
                    p[pre + "qa"], p[pre + "qa_w"] = r(c["q_lat"], d), 1 + r(c["q_lat"], sc=0.1)
                    p[pre + "q"] = r(H * hd, c["q_lat"])
                    if c["q_raw"]:
                        p[pre + "qraw"] = r(H * hd, c["q_lat"])
                else:
                    p[pre + "q"] = r(H * hd, d)
                p[pre + "k"] = r(KV * hd, d)
                p[pre + "v"] = p[pre + "k"] if c["keqv"] else r(KV * hd, d)
                p[pre + "o"] = r(d, H * hd)
                if c["qkv_bias"]:
                    p[pre + "qb"], p[pre + "kb"] = r(H * hd), r(KV * hd)
                    p[pre + "vb"] = p[pre + "kb"] if c["keqv"] else r(KV * hd)
                if c["out_bias"]:
                    p[pre + "ob"] = r(d)
                if c["sink"]:
                    p[pre + "sink"] = r(H, sc=1.0)
                if c["out_gate"]:
                    rows = H if c["out_gate"] == "head" else H * hd
                    if c["og_qlat"]:
                        p[pre + "oqa"], p[pre + "oqa_w"] = r(c["og_qlat"], d), 1 + r(c["og_qlat"], sc=0.1)
                    p[pre + "ogate"] = r(rows, c["og_qlat"] or d)
                    if c["og_bias"]:
                        p[pre + "ogate_b"] = r(rows, sc=0.5)
                self._init_qknorm(pre, r)
            self._init_ffn(i, pre, r)
            self._init_r3(i, pre, r)
            if c["value_emb"]:
                p[pre + "ve"] = r(V, KV * hd)
            if c["ple"]:
                p[pre + "ple_e"] = r(V, 8)
                p[pre + "ple_w"] = r(d, 8)
            if c["engram"]:
                p[pre + "eng"] = r(V * V, d, sc=0.1) if V <= 61 else None
            if c["switch_lora"]:
                p[pre + "lora_a"] = r(4, d)
                p[pre + "lora_b"] = r(H * hd, 4)
        if c["cross_attn"]:
            p["mem"] = r(c["n_mem"], d, sc=1.0)          # the encoder states (fixed)
        if c["enc_input"]:
            p["temb"], p["enc_w"], p["enc_proj"] = r(c["n_text"], d, sc=1.0), r(d, d), r(d, d)
        if c["mtp"]:
            p["mtp"] = r(d, 2 * d)
        if c["selfcond"]:
            p["sc"] = r(d, V, sc=0.05)
        if c["rd_scale"]:
            p["rd_s"] = 1 + r(1, sc=0.5)
        if c["backout"]:
            p["bo"] = 0.3 + r(1, sc=0.2)
        if c["hc"]:
            n, var = c["hc"]
            nd = n * d
            sites = [f"l{i}.hc{k}" for i in range(self.n_param_layers()) for k in (1, 2)]
            for s_ in sites:
                if var == "qwen4":
                    p[s_ + "_nw"] = r(nd, sc=0.1)                  # the per-stream (1 + w) norm
                    p[s_ + "_down"], p[s_ + "_up"] = r(c["hc_low"], nd, sc=1.0), r(nd, c["hc_low"], sc=1.0)
                    p[s_ + "_inj"] = r(n, nd, sc=1.0)
                else:
                    m = (2 + n) * n if var in ("dsv4", "glm5_next") else 2 * n
                    p[s_ + "_fn"], p[s_ + "_base"] = r(m, nd, sc=1.0), r(m, sc=1.0)
                    p[s_ + "_scale"] = r(3 if var in ("dsv4", "glm5_next") else 2, sc=1.0)
            if var == "qwen4":
                p["hch_nw"] = r(nd, sc=0.1)
                p["hch_down"], p["hch_up"] = r(c["hc_low"], nd, sc=1.0), r(nd, c["hc_low"], sc=1.0)
            elif var in ("dsv4", "hy4"):
                p["hch_fn"], p["hch_base"], p["hch_scale"] = r(n, nd, sc=1.0), r(n, sc=1.0), r(1, sc=1.0)
        return self

    def _init_r3(self, i, pre, r):
        c, p, d, H, KV, hd = self.c, self.p, self.c["d"], self.c["H"], self.c["KV"], self.c["hd"]
        for k in (1, 2):                                 # round 3, batch 4b: the residual merges
            if c["res_gain"]:
                p[pre + f"rg{k}"] = 1 + r(d, sc=0.3)
            if c["res_bias"]:
                p[pre + f"rb{k}"] = r(d, sc=0.3)
            if c["res_gate"]:
                p[pre + f"rgw{k}"], p[pre + f"rgb{k}"] = r(d), r(1, sc=0.5)
        if c["engram_gate"]:
            V = c["V"]
            p[pre + "ge_e"] = r(V * V, 6, sc=1.0)
            p[pre + "ge_k"], p[pre + "ge_v"], p[pre + "ge_conv"] = r(d, 6), r(d, 6), r(d, 3)
        if c["post_norm"] in ("attn", "both"):
            p[pre + "pa_w"] = 1 + r(d, sc=0.1)
        if c["post_norm"] == "both":
            p[pre + "pf_w"] = 1 + r(d, sc=0.1)
        if c["cross_attn"]:
            p[pre + "nc_w"] = 1 + r(d, sc=0.1)
            p[pre + "xq"], p[pre + "xk"], p[pre + "xv"] = r(H * hd, d), r(KV * hd, d), r(KV * hd, d)
            p[pre + "xo"] = r(d, H * hd)
        if c["gated_norm"]:
            for n in ("n1", "n2"):
                p[pre + n + "_g1"], p[pre + n + "_g2"] = r(c["gated_norm"], d), r(d, c["gated_norm"])
        if c["out_gate_post"]:
            p[pre + "ogp"] = r(d, d)
        if c["q_temp"]:
            p[pre + "qts"], p[pre + "qtb"] = r(H, sc=0.5), 1 + r(H, sc=0.2)
        if c["rope_warp"]:
            p[pre + "rw_m"], p[pre + "rw_s"] = r(d, sc=1.0), r(d, sc=1.0)
        if c["dyn_mask"]:
            p[pre + "dmA"], p[pre + "dmdt"] = r(c["KV"], sc=1.0), r(c["KV"], c["KV"] * c["hd"], sc=0.5)
        if c["vha"]:                                     # VHAPostMix U, V (num_heads x rank)
            p[pre + "vha_U"], p[pre + "vha_V"] = r(H, c["vha"], sc=0.5), r(H, c["vha"], sc=0.5)
        if c["xsa"] == "scaled":                         # Limite xsa_alpha, one per head
            p[pre + "xsa_a"] = r(H, sc=1.0)
        if c["canon"]:                                   # depthwise causal conv taps at A (attention in) / C (MLP in)
            p[pre + "cA"], p[pre + "cC"] = r(d, c["canon"], sc=0.3), r(d, c["canon"], sc=0.3)
        if c["moa"]:                                     # E expert q / o pairs (KV heads each), router, output bias
            E_ = c["moa"][0]
            p[pre + "moa_q"], p[pre + "moa_o"] = r(E_, KV * hd, d), r(E_, d, KV * hd)
            p[pre + "moa_r"], p[pre + "moa_b"] = r(E_, d, sc=1.0), r(d)
        if c["mem_attn"]:                                # slot queries, pool key / memory k / v / q, read_o, gain
            S_ = c["mem_slots"]
            p[pre + "ms"] = r(S_, d, sc=1.0)
            for k_ in ("mpk", "mk", "mv", "mq"):
                p[pre + k_] = r(d, d)
            p[pre + "mo"], p[pre + "mg"] = r(d, d), r(1, sc=1.0)
        if c["scmoe"]:                                   # the second sublayer pair and the dense MLP beside the MoE
            s2 = f"l{i}s."
            p[s2 + "q"], p[s2 + "k"], p[s2 + "v"] = r(H * hd, d), r(KV * hd, d), r(KV * hd, d)
            p[s2 + "o"] = r(d, H * hd)
            p[s2 + "n1_w"], p[s2 + "n2_w"] = 1 + r(d, sc=0.1), 1 + r(d, sc=0.1)
            for m in ("m0", "m1"):
                p[s2 + m + "up"], p[s2 + m + "fgate"], p[s2 + m + "down"] = r(c["F"], d), r(c["F"], d), r(d, c["F"])
        if c["par_moe_res"]:                             # round 3, batch 3a: Arctic's residual MLP and its norm
            r2 = f"l{i}r."
            p[pre + "nr_w"] = 1 + r(d, sc=0.1)
            p[r2 + "up"], p[r2 + "fgate"], p[r2 + "down"] = r(c["F"], d), r(c["F"], d), r(d, c["F"])

    def _init_qknorm(self, pre, r):
        c, p, H, hd = self.c, self.p, self.c["H"], self.c["hd"]
        v = c["qknorm"]
        for t in ("qn", "kn"):
            if v in ("head_qfree", "dsv4_kv") and t == "kn":   # Domyn-Edge / V4.1: only the k norm has a gain
                p[pre + t] = 1 + r(hd, sc=0.2)
            if v == "mla_krope" and t == "kn":               # dots3-note: a gain over the rotary key dims
                p[pre + t] = 1 + r(c["rot"], sc=0.2)
            if v in ("head", "head_post"):
                p[pre + t] = 1 + r(hd, sc=0.2)
            elif v == "head_distinct":
                p[pre + t] = 1 + r(H if t == "qn" else c["KV"], hd, sc=0.2)
            elif v == "proj":
                p[pre + t] = 1 + r((H if t == "qn" else c["KV"]) * hd, sc=0.2)
            elif v in ("head_ln", "head_ln_bias"):
                p[pre + t] = 1 + r(H if t == "qn" else c["KV"], hd, sc=0.2)
                if v == "head_ln_bias":
                    p[pre + t + "_b"] = r(H if t == "qn" else c["KV"], hd, sc=0.2)
            elif v == "ln_shared_bias":
                p[pre + t] = 1 + r(hd, sc=0.2)
                p[pre + t + "_b"] = r(hd, sc=0.2)

    def _init_ffn(self, i, pre, r):
        c, p, d = self.c, self.p, self.c["d"]
        if c["E"] and i not in c["dense_layers"]:
            E, ew = c["E"], c["ew"]
            ind = c["latent"] or d
            p[pre + "router"] = r(E, d, sc=1.0)
            if c["sel_bias"]:
                p[pre + "selb"] = r(E, sc=0.5)
            if c["latent"]:
                p[pre + "lat_in"] = r(c["latent"], d)
                p[pre + "lat_out"] = r(d, c["latent"])
            p[pre + "eg"], p[pre + "eu"], p[pre + "ed"] = r(E, ew, ind), r(E, ew, ind), r(E, ind, ew)
            if c["ebias"]:
                p[pre + "eub"] = r(E, ew)
            if c["shared"]:
                p[pre + "sg"], p[pre + "su"], p[pre + "sd"] = r(c["shared"], d), r(c["shared"], d), r(d, c["shared"])
                if c["shared_gate"]:
                    p[pre + "sgate"] = r(d)
            return
        if c["ffn"] == "none":                           # (round 3, mamba_ffn_none: a block with no FFN)
            return
        Fw = c["F"]
        p[pre + "up"] = r(Fw, d)
        if c["ffn"] == "glu":
            p[pre + "fgate"] = r(Fw, d)
        p[pre + "down"] = r(d, Fw)
        if c["out_bias"]:
            p[pre + "upb"], p[pre + "downb"] = r(Fw), r(d)

    def n_param_layers(self):
        return self.c["loop"][0] if self.c["loop"] else self.c["L"]

    def kind(self, i):
        k = self.c["kinds"]
        return k[i] if k else "attn"

    def clone(self):
        o = RefLM(**copy.deepcopy(self.c))
        o.p = {k: v.clone() if v is not None else None for k, v in self.p.items()}
        o.inv_freq, o.slopes = self.inv_freq.clone(), self.slopes.clone()
        return o

    def zeros_like_cfg(self, **cfg):
        """A host with this config whose every parameter is zero (norm gains too); the construction writes it."""
        h = RefLM(**cfg).init()
        h.p = {k: (torch.zeros_like(v) if v is not None else None) for k, v in h.p.items()}
        return h

    # ------------------------------------------------------------------ forward
    def norm(self, x, pre):
        c, p = self.c, self.p
        if c["norm"] == "ln":
            x = x - x.mean(-1, keepdim=True)
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c["eps"]) * p[pre + "_w"]
        if pre + "_b" in p:
            y = y + p[pre + "_b"]
        if c["gated_norm"] and pre + "_g1" in p:           # A.X-K2 / Grug gated pre-norm: y sigmoid(fc2 silu(fc1 y))
            y = y * torch.sigmoid(F.silu(y @ p[pre + "_g1"].T) @ p[pre + "_g2"].T)
        return y

    def rope(self, x, pos, ang=None):
        rot = self.c["rot"]
        if rot == 0:
            return x
        if ang is None:
            ang = pos[:, None].to(F64) * self.inv_freq[None, :]         # (T, rot / 2)
            cos, sin = torch.cos(ang).repeat(1, 2), torch.sin(ang).repeat(1, 2)
        else:                                                           # (B, T, rot / 2): per sequence, all heads
            cos, sin = torch.cos(ang).repeat(1, 1, 2)[:, None], torch.sin(ang).repeat(1, 1, 2)[:, None]
        xr, xp = x[..., :rot], x[..., rot:]
        h = rot // 2
        rh = torch.cat([-xr[..., h:], xr[..., :h]], -1)
        return torch.cat([xr * cos + rh * sin, xp], -1)

    def qknorm(self, x, pre, t, n):
        c, p, v = self.c, self.p, self.c["qknorm"]
        hd, e = c["hd"], c["qk_eps"]
        rms = lambda y: y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + e)
        if v == "no" or v == "head_post":
            return x
        if v == "head":
            return rms(x) * p[pre + t]
        if v == "head_distinct":
            return rms(x) * p[pre + t][None, :, None, :] if x.dim() == 4 else rms(x) * p[pre + t]
        if v == "head_free":
            return rms(x)
        if v == "dsv4_kv":                                  # V4.1: only the K = V head, weighted
            return x if t == "qn" else rms(x) * p[pre + t]
        if v == "mla_krope":                                # dots3-note: only the rotary key dims, weighted
            if t == "qn":
                return x
            r_ = c["rot"]
            return torch.cat([rms(x[..., :r_]) * p[pre + t], x[..., r_:]], -1)
        if v == "head_qfree":                               # weight-free q, weighted k (per head, before RoPE)
            return rms(x) if t == "qn" else rms(x) * p[pre + t]
        if v == "head_free_gain":                           # weight-free, the logit times a config constant
            return rms(x) * c["qk_gain"] if t == "qn" else rms(x)
        if v == "proj":
            B, h, T, D = x.shape
            y = x.transpose(1, 2).reshape(B, T, h * D)
            y = rms(y) * p[pre + t]
            return y.reshape(B, T, h, D).transpose(1, 2)
        if v in ("head_ln", "head_ln_bias", "ln_shared_bias"):
            y = rms(x - x.mean(-1, keepdim=True))
            if v == "ln_shared_bias":
                return y * p[pre + t] + p[pre + t + "_b"]
            y = y * p[pre + t][None, :, None, :]
            if v == "head_ln_bias":
                y = y + p[pre + t + "_b"][None, :, None, :]
            return y
        raise KeyError(v)

    def mask(self, T, mask=None):
        c = self.c
        mode = mask or c["mask"]
        i = torch.arange(T)[:, None]
        j = torch.arange(T)[None, :]
        if mode == "causal":
            m = j <= i
        elif mode == "full":
            m = torch.ones(T, T, dtype=torch.bool)
        elif mode.startswith("prefix"):                 # a prefix LM: the first P positions see each other
            P = int(mode.split(":")[1]) if ":" in mode else 10
            m = (j <= i) | (j < P)
        elif mode.startswith("block"):
            b = int(mode.split(":")[1]) if ":" in mode else c["block_size"]
            m = (j // b) <= (i // b)
        else:
            raise KeyError(mode)
        return m

    def attn(self, x, li, pre, ids, kv_cache, mask=None):
        c, p = self.c, self.p
        B, T, _ = x.shape
        H, KV, hd = c["H"], c["KV"], c["hd"]
        pos = torch.arange(T)
        ang = None
        if c["rope_warp"] and pre + "rw_m" in p:          # GPT-X2's metric-warped phase (modeling_gptx2.py:706-744)
            z = x.mean(1)                                 # the whole sequence (non-causal), per batch row
            mass = c["rw_c"] * torch.tanh(z @ p[pre + "rw_m"])
            spin = c["rw_c"] * torch.tanh(z @ p[pre + "rw_s"])
            t_ = pos.to(F64)
            base = t_[:, None] * self.inv_freq[None, :]
            radial = 1.0 / (1.0 + t_ / c["rw_base"])
            drag = (t_ / max(T - 1, 1)) ** 2
            ang = (base[None] * (1 + mass[:, None, None] * radial[None, :, None])
                   + spin[:, None, None] * drag[None, :, None])
        if c["moa"] and pre + "moa_q" in p:
            return self.moa_attn(x, pre, mask)
        if pre + "qa" in p:                               # the q-latent query (MLA q_lora path)
            z = x @ p[pre + "qa"].T
            zn = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + c["eps"]) * p[pre + "qa_w"]
            q = zn @ p[pre + "q"].T
            if pre + "qraw" in p:                         # A.X-K2 q_gate_proj(cat(q_a_layernorm(z), z)), modular_axk2.py:327-331
                q = q + z @ p[pre + "qraw"].T
        else:
            q = x @ p[pre + "q"].T
        q = q + (p[pre + "qb"] if pre + "qb" in p else 0)
        if c["clip_qkv"] is not None:                     # OLMo-1 query_states.clamp_(-c, c), modeling_olmo.py:245-248
            q = q.clamp(-c["clip_qkv"], c["clip_qkv"])
        q = q.view(B, T, H, hd).transpose(1, 2)
        share = c["cla"] and li in c["cla"]
        if share:
            k, v = kv_cache["kv"]
        else:
            k = x @ p[pre + "k"].T + (p[pre + "kb"] if pre + "kb" in p else 0)
            v = x @ p[pre + "v"].T + (p[pre + "vb"] if pre + "vb" in p else 0)
            if c["value_emb"]:
                v = v + p[pre + "ve"][ids]
            if c["clip_qkv"] is not None:                 # (the key / value clamp, the same lines)
                k, v = k.clamp(-c["clip_qkv"], c["clip_qkv"]), v.clamp(-c["clip_qkv"], c["clip_qkv"])
            k = k.view(B, T, KV, hd).transpose(1, 2)
            v = v.view(B, T, KV, hd).transpose(1, 2)
            k = self.qknorm(k, pre, "kn", KV)
            if c["qknorm"] == "dsv4_kv":                    # DeepSeek-V4.1: the one K = V head normed (weighted)
                v = k
            if c["pos"] in ("rope",):
                k = self.rope(k, pos, ang)
            if c["qknorm"] == "head_post":
                k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + c["qk_eps"]) * p[pre + "kn"]
            kv_cache["kv"] = (k, v)
        q = self.qknorm(q, pre, "qn", H)
        if c["pos"] in ("rope",):
            q = self.rope(q, pos, ang)
        if c["q_temp"] and pre + "qts" in p:              # Mistral-Large-4 softmax scaling (modular_mistral_large4.py:296-309)
            q = q * (p[pre + "qts"][:, None] * torch.log(pos.to(F64) + 1)[None, :] + p[pre + "qtb"][:, None])[None, ..., None]
        if c["qknorm"] == "head_post":
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + c["qk_eps"]) * p[pre + "qn"]
        g = H // KV
        dt = None
        if c["dyn_mask"] and pre + "dmA" in p:          # Doge: dt_states = exp(A softplus(dt_proj(values))) per key
            vflat = v.transpose(1, 2).reshape(B, T, KV * hd)
            dt = torch.exp(p[pre + "dmA"][None, None, :] * F.softplus(vflat @ p[pre + "dmdt"].T))   # (B, T, KV)
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        sc = c["scale"] if c["scale"] is not None else 1 / math.sqrt(hd)
        s = (q @ k.transpose(-1, -2)) * sc
        i, j = pos[:, None], pos[None, :]
        if c["pos"] == "alibi":
            s = s + self.slopes[None, :, None, None] * (j - i).to(F64)[None, None]
        if c["pos"] == "additive_bias":
            s = s + p["abias"][None, :, :T, :T]
        if c["pos"] == "t5":
            s = s + p["t5"][:, t5_bucket(i - j)][None]
        if c["softcap_attn"]:
            s = c["softcap_attn"] * torch.tanh(s / c["softcap_attn"])
        m = self.mask(T, mask)
        w = c["window"][li] if isinstance(c["window"], (list, tuple)) else c["window"]
        if w:
            m = m & (j > i - w)
        if c["dsa_topk"] is not None:                # DSA: the index_topk best keys per query (indexer = the scores)
            kk = min(c["dsa_topk"], T)
            thr = s.masked_fill(~m, -1e300).topk(kk, -1).values[..., -1:]
            m = m & (s >= thr)
        if dt is not None:                                # the per-key bias, repeated to the query heads (:315)
            s = s + dt.transpose(1, 2).repeat_interleave(g, 1)[:, :, None, :]
        if c["doc_eos"] is not None:                      # document ids: cumsum(ids == eos) - is_eos
            is_eos = (ids == c["doc_eos"]).long()
            doc = is_eos.cumsum(-1) - is_eos
            m = m & (doc[:, None, :, None] == doc[:, None, None, :])
        s = s.masked_fill(~m, float("-inf"))
        if c["sink"]:
            sk = p[pre + "sink"][None, :, None, None].expand(B, H, T, 1)
            a = torch.softmax(torch.cat([s, sk], -1), -1)[..., :T]
        else:
            a = torch.softmax(s, -1)
        o = a @ v                                         # (B, H, T, hd)
        if c["xsa"]:                                      # XSA: y - f (y . v_hat) v_hat, v_i the token's own value of
            #                                               the head's KV group (modeling_gptx2.py:106-111; Limite
            #                                               modeling_limite.py:367-395, f = tanh(xsa_alpha_h))
            vh = v / v.norm(dim=-1, keepdim=True).clamp(min=c["xsa_eps"])
            f = torch.tanh(p[pre + "xsa_a"])[None, :, None, None] if c["xsa"] == "scaled" else 1.0
            o = o - f * (o * vh).sum(-1, keepdim=True) * vh
        o = o.transpose(1, 2)                             # (B, T, H, hd)
        if c["vha"]:                                      # ERNIE-5.5 VHAPostMix o[t, h] <- sum_g m[h, g] o[t, g],
            #                                               m = I + V U^T (vLLM-ERNIE attention.py:110-146, :229)
            m_ = torch.eye(H, dtype=F64) + p[pre + "vha_V"] @ p[pre + "vha_U"].T
            o = torch.einsum("hg,btgd->bthd", m_, o)
        o = o.reshape(B, T, H * hd)
        if c["out_gate"]:
            gin = x
            if c["og_qlat"]:                                # ERNIE-5.5: the gate reads the normed q latent
                z = x @ p[pre + "oqa"].T
                gin = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + c["eps"]) * p[pre + "oqa_w"]
            z = gin @ p[pre + "ogate"].T + (p[pre + "ogate_b"] if pre + "ogate_b" in p else 0)
            g_ = {"sigmoid": torch.sigmoid, "softplus": F.softplus, "silu": F.silu}[c["og_act"]](z)
            if c["out_gate"] == "head":                     # one gate per head, broadcast over its head_dim channels
                g_ = g_.repeat_interleave(hd, -1)
            o = o * g_
        if c["switch_lora"]:
            o = o + (x @ p[pre + "lora_a"].T) @ p[pre + "lora_b"].T
        out = o @ p[pre + "o"].T + (p[pre + "ob"] if pre + "ob" in p else 0)
        if c["out_gate_post"] and pre + "ogp" in p:     # MM200: o_proj(attn) * sigmoid(W_g x)
            out = out * torch.sigmoid(x @ p[pre + "ogp"].T)
        if c["mem_attn"] and pre + "mo" in p:
            out = out + self.memory(x, pre)
        return out

    def canon_conv(self, y, key):
        """A Canon layer y + conv(y): a causal depthwise Conv1d (per-channel taps w[:, j] on y[t - j], left zero pad), no
        bias, residual (dhara modeling_dhara_ar.py:46-100, Ethoric modeling_ethoric.py:38-74, positions A / C)."""
        if not self.c["canon"] or key not in self.p:
            return y
        w = self.p[key]
        out = y.clone()
        for j in range(w.shape[1]):
            sh = torch.cat([torch.zeros_like(y[:, :j]), y[:, :y.shape[1] - j]], 1) if j else y
            out = out + sh * w[:, j]
        return out

    def memory(self, x, pre):
        """A causal summary memory (haru model.py:315-445 CausalSummaryMemory, Bluebird fabric_runtime.py:209-262):
        mem_slots learned queries softmax-pool each complete mem_chunk-token chunk into summary keys / values, a token
        attends to the summaries of EARLIER chunks only (0 where none), read_o times sigmoid(memory_gain) (:543)."""
        c, p = self.c, self.p
        B, T, d = x.shape
        C, S_ = c["mem_chunk"], c["mem_slots"]
        nc = T // C
        if nc == 0:
            return torch.zeros_like(x)
        xc = x[:, :nc * C].reshape(B, nc, C, d)
        w = torch.softmax(torch.einsum("sd,bncd->bnsc", p[pre + "ms"], xc @ p[pre + "mpk"].T) / math.sqrt(d), -1)
        summ = torch.einsum("bnsc,bncd->bnsd", w, xc).reshape(B, nc * S_, d)
        km, vm = summ @ p[pre + "mk"].T, summ @ p[pre + "mv"].T
        qm = x @ p[pre + "mq"].T
        s = qm @ km.transpose(-1, -2) / math.sqrt(d)                       # (B, T, nc S)
        chunk_of_t = torch.arange(T) // C
        chunk_of_s = torch.arange(nc * S_) // S_
        allowed = chunk_of_s[None, :] < chunk_of_t[:, None]
        s = s.masked_fill(~allowed, float("-inf"))
        a = torch.softmax(s, -1)
        a = torch.where(allowed.any(-1)[:, None], a, torch.zeros_like(a))  # no earlier chunk: no memory read
        a = torch.nan_to_num(a)
        return (a @ vm) @ p[pre + "mo"].T * torch.sigmoid(p[pre + "mg"])

    def moa_attn(self, x, pre, mask=None):
        """JetMoE MoA (modular_jetmoe.py:212-373): top-k of E expert query projections, each a block of KV query heads
        on the ONE shared K / V (kv_proj), RoPE, softmax attention, out = sum_j gate_j o_{e_j}(attn_j) + bias, the
        gates softmax over the k selected router logits (JetMoeTopKGating :106-149)."""
        c, p = self.c, self.p
        B, T, _ = x.shape
        KV, hd = c["KV"], c["hd"]
        E_, kk = c["moa"]
        pos = torch.arange(T)
        k = (x @ p[pre + "k"].T).view(B, T, KV, hd).transpose(1, 2)
        v = (x @ p[pre + "v"].T).view(B, T, KV, hd).transpose(1, 2)
        if c["pos"] == "rope":
            k = self.rope(k, pos)
        logits = x @ p[pre + "moa_r"].T                                    # (B, T, E)
        top, idx = logits.topk(kk, -1)
        gate = torch.zeros_like(logits).scatter(-1, idx, torch.softmax(top, -1))
        m = self.mask(T, mask)
        out = p[pre + "moa_b"].expand(B, T, -1).clone()
        for e in range(E_):
            q = (x @ p[pre + "moa_q"][e].T).view(B, T, KV, hd).transpose(1, 2)
            if c["pos"] == "rope":
                q = self.rope(q, pos)
            s = (q @ k.transpose(-1, -2)) * (c["scale"] if c["scale"] is not None else 1 / math.sqrt(hd))
            a = torch.softmax(s.masked_fill(~m, float("-inf")), -1)
            o = (a @ v).transpose(1, 2).reshape(B, T, KV * hd) @ p[pre + "moa_o"][e].T
            out = out + gate[..., e:e + 1] * o
        return out

    def xattn(self, x, pre):
        """Cross-attention to the encoder memory (Parler-TTS encoder_attn): no mask, no positions."""
        c, p = self.c, self.p
        B, T, _ = x.shape
        H, KV, hd = c["H"], c["KV"], c["hd"]
        mem = p["mem"][None].expand(B, -1, -1)
        q = (x @ p[pre + "xq"].T).view(B, T, H, hd).transpose(1, 2)
        k = (mem @ p[pre + "xk"].T).view(B, -1, KV, hd).transpose(1, 2).repeat_interleave(H // KV, 1)
        v = (mem @ p[pre + "xv"].T).view(B, -1, KV, hd).transpose(1, 2).repeat_interleave(H // KV, 1)
        a = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(hd), -1)
        return (a @ v).transpose(1, 2).reshape(B, T, H * hd) @ p[pre + "xo"].T

    def dense(self, x, pre):
        p = self.p
        return (act_fn(self.c["act"])(x @ p[pre + "fgate"].T) * (x @ p[pre + "up"].T)) @ p[pre + "down"].T

    def rec(self, x, pre):
        """A gated linear recurrence h_t = sigmoid(W_a x_t) * h_{t-1} + W_b x_t, out W_o h (the host's recurrent
        layers; written as the identity by zeroing W_o)."""
        p = self.p
        a, b = (x @ p[pre + "rin"].T).chunk(2, -1)
        dec = torch.sigmoid(x @ p[pre + "rdec"].T)
        h = torch.zeros_like(x[:, 0])
        out = []
        for t in range(x.shape[1]):
            h = dec[:, t] * h + b[:, t]
            out.append(h)
        return torch.stack(out, 1) @ p[pre + "rout"].T

    def ffn(self, x, i, pre):
        c, p = self.c, self.p
        if c["E"] and i not in c["dense_layers"]:
            return self.moe(x, pre)
        if c["ffn"] == "none":                           # (round 3, mamba_ffn_none: the sublayer adds nothing)
            return torch.zeros_like(x)
        h = x @ p[pre + "up"].T + (p[pre + "upb"] if pre + "upb" in p else 0)
        if c["ffn"] == "glu":
            gte = x @ p[pre + "fgate"].T + (p[pre + "fgateb"] if pre + "fgateb" in p else 0)
            if c["act_clip"] is not None:                # (DeepSeek-V4 modeling_deepseek_v4.py:1022-1026)
                gte, h = gte.clamp(max=c["act_clip"]), h.clamp(min=-c["act_clip"], max=c["act_clip"])
            h = swigluoai(gte, h) if c["act"] == "swigluoai" else act_fn(c["act"])(gte) * h
        else:
            h = act_fn(c["act"])(h)
        return h @ p[pre + "down"].T + (p[pre + "downb"] if pre + "downb" in p else 0)

    def moe(self, x, pre):
        c, p = self.c, self.p
        E, k = c["E"], c["k"]
        logits = x @ p[pre + "router"].T
        if c["score"] == "softmax":
            sc = torch.softmax(logits, -1)
        elif c["score"] == "sigmoid":
            sc = torch.sigmoid(logits)
        else:
            sc = torch.sqrt(F.softplus(logits))
        sel = (logits if c["sel_domain"] == "logit" else sc) + (p[pre + "selb"] if pre + "selb" in p else 0)
        idx = None
        if c["n_group"] and c["group_score"] == "per_group":        # top-(k / G) inside every group, none dropped
            G = c["n_group"]
            gs = sel.view(*sel.shape[:-1], G, E // G)
            off = (torch.arange(G) * (E // G)).view(G, 1)
            idx = (gs.topk(k // G, -1).indices + off).flatten(-2)
        elif c["n_group"]:
            G = c["n_group"]
            gs = sel.view(*sel.shape[:-1], G, E // G)
            gscore = gs.max(-1).values if c["group_score"] == "max" else gs.topk(2, -1).values.sum(-1)
            keep = torch.zeros_like(gscore).scatter(-1, gscore.topk(c["topk_group"], -1).indices, 1.0)
            fill = 0.0 if c["group_fill"] == "zero" else float("-inf")
            sel = (gs.masked_fill(keep[..., None] == 0, fill)).view(sel.shape)
        if c["blk"]:                     # block routing: per block of bs tokens the cap experts with the best block max
            bs, cap = c["blk"]
            B, T, _ = sel.shape
            allow = torch.zeros_like(sel, dtype=torch.bool)
            for s in range(0, T, bs):    # (a last partial block is its own block: the code asserts T % bs == 0)
                bmax = sel[:, s:s + bs].max(1).values
                allow[:, s:s + bs] = torch.zeros_like(bmax, dtype=torch.bool).scatter(
                    -1, bmax.topk(cap, -1).indices, True)[:, None]
            sel = sel.masked_fill(~allow, float("-inf"))
        if idx is None:
            idx = sel.topk(k, -1).indices
        w = sc.gather(-1, idx)
        if c["renorm"] == "R":
            w = w / w.sum(-1, keepdim=True)
        w = w * c["route_scale"]
        xin = x @ p[pre + "lat_in"].T if c["latent"] else x
        out = torch.zeros_like(xin)
        for e in range(E):
            ge, ue = xin @ p[pre + "eg"][e].T, xin @ p[pre + "eu"][e].T + (p[pre + "eub"][e] if pre + "eub" in p else 0)
            if c["act_clip"] is not None:                # (the experts' clamp, modeling_deepseek_v4.py:1066-1067)
                ge, ue = ge.clamp(max=c["act_clip"]), ue.clamp(min=-c["act_clip"], max=c["act_clip"])
            he = act_fn(c["act"])(ge) * ue
            ye = he @ p[pre + "ed"][e].T
            we = (w * (idx == e)).sum(-1, keepdim=True)
            out = out + we * ye
        if c["latent"]:
            out = out @ p[pre + "lat_out"].T
        if c["shared"]:
            hs = act_fn(c["act"])(x @ p[pre + "sg"].T) * (x @ p[pre + "su"].T)
            ys = hs @ p[pre + "sd"].T
            if c["shared_gate"]:
                ys = ys * torch.sigmoid(x @ p[pre + "sgate"])[..., None]
            out = out + ys
        return out

    def layer(self, x, li, pi, ids, cache, mask):
        c, p = self.c, self.p
        pre = f"l{pi}."
        if c["ple"]:
            x = x + p[pre + "ple_e"][ids] @ p[pre + "ple_w"].T
        if c["engram"] and p.get(pre + "eng") is not None:
            prev = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], 1)
            x = x + p[pre + "eng"][prev * c["V"] + ids]
        if c["engram_gate"]:                             # AllGDN Engram (modeling_allgdn.py:110-126), Qwen4-exp PLE
            prev = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], 1)
            e = p[pre + "ge_e"][prev * c["V"] + ids]
            rms = lambda z: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)
            g = torch.sigmoid((rms(x) * rms(e @ p[pre + "ge_k"].T)).sum(-1, keepdim=True) / math.sqrt(c["d"]))
            v = g * (e @ p[pre + "ge_v"].T)
            w, out = p[pre + "ge_conv"], v.clone()
            for j in range(w.shape[1]):
                out = out + (torch.cat([torch.zeros_like(v[:, :j]), v[:, :v.shape[1] - j]], 1) if j else v) * w[:, j]
            x = x + out
        mix = (lambda y: self.rec(y, pre)) if self.kind(pi) == "rec" else (
            lambda y: self.attn(y, li, pre, ids, cache, mask))
        if c["scmoe"]:                                   # LongCat ScMoE (modeling_longcat.py:514-579)
            s2 = f"l{pi}s."
            x1 = x + mix(self.norm(x, pre + "n1"))
            h = self.norm(x1, pre + "n2")
            m = self.ffn(h, pi, pre)
            x2 = x1 + self.dense(h, s2 + "m0")
            x3 = x2 + self.attn(self.norm(x2, s2 + "n1"), li, s2, ids, {}, mask)
            return x3 + self.dense(self.norm(x3, s2 + "n2"), s2 + "m1") + m
        if c["block"] == "parallel":
            y = self.norm(x, pre + "n1")
            return x + mix(y) + self.ffn(y, pi, pre)
        if c["par_moe_res"]:                             # Arctic (modeling_arctic.py:1083-1110)
            h = x + mix(self.norm(x, pre + "n1"))
            return h + self.dense(self.norm(h, pre + "nr"), f"l{pi}r.") + self.ffn(self.norm(x, pre + "n2"), pi, pre)
        u1 = self.canon_conv(self.norm(x, pre + "n1"), pre + "cA")
        a = mix(u1)
        if c["post_norm"] in ("attn", "both"):
            a = self.norm(a, pre + "pa")
        x = self._merge(x, a, u1, pre, 1)
        if c["cross_attn"]:
            x = x + self.xattn(self.norm(x, pre + "nc"), pre)
        u2 = self.canon_conv(self.norm(x, pre + "n2"), pre + "cC")
        f = self.ffn(u2, pi, pre)
        if c["post_norm"] == "both":
            f = self.norm(f, pre + "pf")
        return self._merge(x, f, u2, pre, 2)

    def _merge(self, x, y, u, pre, k):
        """The residual merge x + y; round 3, batch 4b: the residual's gain / bias ((x + b) s, Doge / ZAYA) and CFRD's
        scalar gate on the update (rg_c sigmoid(w . u + b) y)."""
        c, p = self.c, self.p
        if c["res_bias"]:
            x = x + p[pre + f"rb{k}"]
        if c["res_gain"]:
            x = x * p[pre + f"rg{k}"]
        if c["res_gate"]:
            y = c["rg_c"] * torch.sigmoid(u @ p[pre + f"rgw{k}"] + p[pre + f"rgb{k}"])[..., None] * y
        return x + y

    @torch.no_grad()
    def __call__(self, ids, mask=None, prev_logits=None):
        c = self.c
        if c["fake_quant"]:                             # BitCPM LinearQuantizer: every projection ternary per row
            saved, self.p = self.p, quantized(self.p)
            try:
                return self._call(ids, mask, prev_logits)
            finally:
                self.p = saved
        return self._call(ids, mask, prev_logits)

    def _call(self, ids, mask=None, prev_logits=None):
        c, p = self.c, self.p
        if c["enc_input"]:                              # Breeze: text ids (>= V) are a bidirectional encoder's output
            V = c["V"]
            txt = ids >= V
            x = p["emb"][ids.clamp(max=V - 1)] * c["emb_scale"]
            if txt.any():
                e = p["temb"][(ids - V).clamp(min=0)] * txt[..., None]
                pool = e.sum(1, keepdim=True) / txt.sum(1, keepdim=True).clamp(min=1)[..., None]
                enc = torch.tanh((e + pool) @ p["enc_w"].T) @ p["enc_proj"].T
                x = torch.where(txt[..., None], enc, x)
        else:
            x = p["emb"][ids] * c["emb_scale"]
        T = ids.shape[1]
        if c["pos"] == "learned_abs":
            x = x + p["pos"][:T]
        if c["selfcond"] and prev_logits is not None:
            x = x + torch.softmax(prev_logits, -1) @ p["sc"].T
        if c["selfcond_norm"]:                           # DiffusionGemmaSelfConditioning.post_norm (with_scale False)
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c["eps"])
        if c["hc"]:
            return self._hc_call(x, ids, mask)
        cache = {}
        if c["loop"]:
            b, K = c["loop"]
            order = [i for _ in range(K) for i in range(b)]
        else:
            order = list(range(c["L"]))
        rd, bn = c["rd_scale"], c["loop"] and c["loop_bnorm"] and not c["skip_loop_final_norm"]
        x_a = x_bo = None
        for li, pi in enumerate(order):
            if rd and li == rd[0]:
                x_a = x
            x = self.layer(x, li, pi, ids, cache, mask)
            if c["backout"] and li == len(order) // 2 - 1:
                x_bo = x                                       # Gaggle / nanochat_gpt: x_backout at the mid layer
            if rd and li == rd[1] - 1:
                x = x_a + p["rd_s"] * (x - x_a)              # LFM3-RD: prelude_out + step_scale (h_core - prelude_out)
            if bn and (li + 1) % c["loop"][0] == 0 and li + 1 < len(order):
                x = self.norm(x, "nf")                         # Nanbeige4.2: self.norm after every loop pass
        if c["backout"]:
            x = x - p["bo"] * x_bo                             # x_L - lambda x_m before the final norm
        y = self.norm(x, "nf")
        out = y @ p["lm"].T + (p["lm_b"] if "lm_b" in p else 0)
        if c["softcap_final"]:
            out = c["softcap_final"] * torch.tanh(out / c["softcap_final"])
        return out

    # ------------------------------------------------------------------ fr-resnorm: hyper-connections
    def _hc_mix(self, s, site):
        """(sublayer input, merge) of one HC site on streams s [B, T, n, d] (the stock forward of each variant)."""
        c, p = self.c, self.p
        n, var = c["hc"]
        B, T, _, d = s.shape
        flat = s.reshape(B, T, n * d)
        rms = lambda z, eps: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)
        if var == "qwen4":                               # Qwen4ExpTextGatedResidual (modeling_qwen4_exp.py:1002-1031)
            sn = (rms(s, c["eps"]) * (1 + p[site + "_nw"].view(n, d))).reshape(B, T, n * d)
            g = torch.sigmoid(F.silu(sn @ p[site + "_down"].T / n) @ p[site + "_up"].T).view(B, T, n, d)
            u = (g * sn.view(B, T, n, d)).mean(2)
            if site + "_inj" not in p:
                return u, None
            return u, ("inj", 2 * torch.sigmoid(sn @ p[site + "_inj"].T / n))
        mixes = rms(flat, c["eps"]) @ p[site + "_fn"].T
        b, sc = p[site + "_base"], p[site + "_scale"]
        e = c["hc_eps"]
        pre = torch.sigmoid(mixes[..., :n] * sc[0] + b[:n]) + e
        u = (pre[..., None] * s).sum(2)
        if var == "hy4":                                 # HYV4HyperConnection: no comb
            post = c["hc_mag"] * torch.sigmoid(mixes[..., n:2 * n] * sc[1] + b[n:2 * n]) + e
            return u, ("post", post, None)
        post = 2 * torch.sigmoid(mixes[..., n:2 * n] * sc[1] + b[n:2 * n])
        comb = torch.softmax((mixes[..., 2 * n:] * sc[2] + b[2 * n:]).view(B, T, n, n), -1) + e
        comb = comb / (comb.sum(-2, keepdim=True) + e)
        for _ in range(c["hc_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + e)
            comb = comb / (comb.sum(-2, keepdim=True) + e)
        if var == "glm5_next":
            comb = comb.transpose(-1, -2)                # Glm5NextTextDecoderLayer: comb^T @ In
        return u, ("post", post, comb)

    @staticmethod
    def _hc_merge(s, y, how):
        if how[0] == "inj":
            return s + y[..., None, :] * how[1][..., None]
        _, post, comb = how
        base = s if comb is None else torch.matmul(comb, s)
        return post[..., None] * y[..., None, :] + base

    def _hc_call(self, x, ids, mask):
        """A plain sequential pre-norm stack with n hyper-connection streams (the embedding copied to every stream);
        the dsv4 / hy4 sublayers keep their own pre-norm on the collapsed input, Qwen4-exp's normed mix IS its input."""
        c, p = self.c, self.p
        n, var = c["hc"]
        s = x[:, :, None, :].expand(-1, -1, n, -1).clone()
        cache = {}
        for li in range(c["L"]):
            pre = f"l{li}."
            for k in (1, 2):
                u, how = self._hc_mix(s, pre + f"hc{k}")
                if var != "qwen4":
                    u = self.norm(u, pre + f"n{k}")
                y = self.attn(u, li, pre, ids, cache, mask) if k == 1 else self.ffn(u, li, pre)
                s = self._hc_merge(s, y, how)
        if var == "qwen4":                               # the final mixer is the final norm
            y, _ = self._hc_mix(s, "hch")
        else:
            if var == "glm5_next":
                h = s.mean(2)                            # Glm5NextTextHyperHead: an unweighted mean
            else:
                B, T = s.shape[:2]
                flat = s.reshape(B, T, -1)
                fl = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + c["eps"])
                pre_h = torch.sigmoid(fl @ p["hch_fn"].T * p["hch_scale"] + p["hch_base"]) + c["hc_eps"]
                h = (pre_h[..., None] * s).sum(2)
            y = self.norm(h, "nf")
        out = y @ p["lm"].T + (p["lm_b"] if "lm_b" in p else 0)
        if c["softcap_final"]:
            out = c["softcap_final"] * torch.tanh(out / c["softcap_final"])
        return out


QUANT_KEYS = ("q", "k", "v", "o", "up", "fgate", "down")


def ternary(W):
    """SteTernaryQuantizer (modeling_minicpm.py:129-140): s = 1 / mean |W_row|, W_q = clamp(round(W s), -1, 1) / s."""
    sc = 1.0 / W.abs().mean(-1, keepdim=True).clamp(min=1e-5)
    return (W * sc).round().clamp(-1, 1) / sc


def quantized(p):
    return {k: (ternary(v) if v is not None and k.split(".")[-1] in QUANT_KEYS else v) for k, v in p.items()}


def fn(model, mask=None):
    return lambda ids: model(ids, mask=mask)
