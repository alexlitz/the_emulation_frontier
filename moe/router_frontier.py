"""Router / MoE constructions of branch fr-moe: null experts, the sigmoid exp regime, and sunk extras.

All three read a CONSTANT CHANNEL: the host has e >= 1 more hidden dims than the target; its embedding writes c into
the first of them for every token, no sublayer writes there (o_proj / down_proj rows zero), and every reader of the
normed stream has a zero column there except the router rows below. Lean const_norm_carry (PreNormRescue.lean): a
host RMSNorm with weight sqrt(d / (d + e)) w on the stream and eps' = (d eps - c^2) / (d + e) returns the target's
RMSNorm_eps(x) on the stream exactly, and wz c / R on the channel, R = sqrt(d / (d + e)) sqrt(ms(x) + eps) a positive
multiple of the target's RMS. A router row with entry rho there adds rho wz c / R to that expert's logit. Every
norm of hidden width shares the config eps, so the host config sets eps'; a narrower norm reading a projection of
the stream (OLMoE's q / k norm) keeps its target eps by scaling the projection rows by s = sqrt(eps' / eps)
(N_eps'(s y) = N_eps(y): the norm is scale-free up to its eps).

  null experts   (moe_null_experts)  k_h = m k_t + r. Each target expert has m whole copies (the target's router row
                 and selection bias); r NULL experts have zero weights, a selection bias above every copy's and the
                 router entry -M on the channel, so the host's top-k_h is the copies plus the nulls (Lean
                 moeR_null_topk) and the nulls' score sigma(-M c / R) is 0 in fp32: the renormalised gates are the
                 target's (moeR_null_renorm: the host is the target times A / (A + B), B the nulls' score mass).
  exp regime     (moe_softmax_exp_regime)  a renormalised SOFTMAX target on a renormalised SIGMOID host: every copy's
                 row also reads -rho on the channel, so its score is sigma(z - s), s = rho c / R one shift per token;
                 sigma is monotone (the target's selection) and sigma(z - s) = e^(z - s) / (1 + e^(z - s)), so the
                 renormalised gates are softmax_S(z) up to e^(z_max - s). REFUTED on real checkpoints (granite MoE:
                 the massive-activation first token sits outside the shift window): no rule. The host code adds 1e-20 to the gate sum (Glm4MoeTopkRouter), so s is held in a window per layer:
                 rho_l is calibrated on prompts DISJOINT from the scored ones (act_swap_agreement.STORY_PROMPTS).
  sunk extras    (moe_plain_extras_sink)  an unrenormalised softmax target with E_h > m E_t: the extras' rows read -M
                 on the channel, exp(-M c / R) / Z_h = 0 in fp32 (Lean moeR_dup_plain_extras_bound: relative error
                 Z_D / (m Z), a limit reached at the fp32 underflow).

Pairs (stock transformers 4.57 classes; the host is written from the target's tensors and the host config only):
  exp:granite30<-glm4     ibm-granite/granite-3.0-1b-a400m-base  softmax R, top-8 of 32 -> Glm4MoeForCausalLM, h + 1
  exp:granite31<-glm4     ibm-granite/granite-3.1-1b-a400m-base  "
  null:glm05<-glm4        PrimeIntellect/GLM-0.5B  sigmoid + bias, top-4 of 8 -> top-10 of 18 (m = 2, r = 2), h + 1
  null:glm28<-glm4        imdatta0/tiny_glm4_moe_2.8B_0.7B  top-8 of 64 -> top-10 of 66 (m = 1, r = 2), h + 1
  null+exp:granite30<-glm4  granite-3.0-1b top-8 -> Glm4Moe top-10 of 34 (m = 1, r = 2) in the exp regime
  plain:qwen15<-qwen2moe  Qwen/Qwen1.5-MoE-A2.7B  softmax P, 60 experts -> Qwen2MoeForCausalLM, 64 experts, h + 1
  plain:olmoe<-olmoe      allenai/OLMoE-1B-7B-0924  softmax P, 64 experts -> OlmoeForCausalLM, 72 experts, h + 32
                          (OLMoE's head_dim is hidden / heads: 16 heads of 130, RoPE pairs (i, i + 65), inv_freq buffer)

Measured with the act/act_swap_agreement.py standard (hf/eps_lever_agreement.py's reference / score): the target's own
greedy 64-token continuations of 24 prompts, fp32; payable = top-1 >= 99% AND >= 90% identical 32-token continuations.

  python -m moe.router_frontier --ref            # the fp64 tiny-model references (stock classes, random weights)
  python -m moe.router_frontier exp:granite30<-glm4 [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch

F64, F32 = torch.float64, torch.float32
OUT = "moe/data/router_frontier.jsonl"
M_SINK = 1e9                 # the -M entry of null / extra rows (x c / R: below -1e5 for any stream rms < 1e2 c M)
U_TARGET = -23.0             # the exp regime's target top host logit z_max - s (sigma = e^u to 1e-10, 1e-20 << sum)


# --- the constant channel (Lean const_norm_carry) ------------------------------------------------------------------

def carry(d, e, eps, c=None):
    """(c, eps', stream weight factor) of a constant channel c in e extra dims over a d-dim stream of norm eps."""
    c = math.sqrt(d * eps / 2) if c is None else c
    assert c * c <= d * eps
    return c, (d * eps - c * c) / (d + e), math.sqrt(d / (d + e))


@dataclass
class Plan:
    m: int = 1                    # copies of each target expert
    r: int = 0                    # null experts (a top-k remainder k_h = m k_t + r)
    x: int = 0                    # extra experts sunk below the copies (no selection)
    e: int = 1                    # extra hidden dims (the first holds the constant)
    exp_rho: list | None = None   # per-layer channel entry of every copy's row (the exp regime), else None
    M: float = M_SINK
    null_bias: float = 10.0
    c: float | None = None
    wz: float = 1.0               # the host norms' weight on the channel


# --- target views (one interface for the families the host writers read) -------------------------------------------

class View:
    """A target's tensors by role: emb, lm_head (with the target's output scale folded), final norm, and per layer the
    attention and the MoE block. get(name) reads the state dict lazily (a model or safetensors files)."""

    def __init__(self, get, cfg):
        self.get, self.cfg = get, cfg

    def t(self, n):
        return self.get(n).to(F64)


class GraniteView(View):
    kind, score, renorm = "granitemoe", "softmax", "R"

    def dims(self):
        c = self.cfg
        hd = c.hidden_size // c.num_attention_heads
        return dict(d=c.hidden_size, L=c.num_hidden_layers, H=c.num_attention_heads, KV=c.num_key_value_heads, hd=hd,
                    E=c.num_local_experts, k=c.num_experts_per_tok, f=c.intermediate_size, eps=c.rms_norm_eps,
                    theta=c.rope_theta, prf=1.0, vocab=c.vocab_size, scale=c.attention_multiplier, rsf=1.0,
                    attn_bias=False, dense_first=0, shared=0, dense_f=0)

    def emb(self):
        return self.t("model.embed_tokens.weight") * self.cfg.embedding_multiplier

    def lm_head(self):
        n = "model.embed_tokens.weight" if self.cfg.tie_word_embeddings else "lm_head.weight"
        return self.t(n) / self.cfg.logits_scaling

    def final_norm(self):
        return self.t("model.norm.weight")

    def layer(self, i):
        p, rm = f"model.layers.{i}.", self.cfg.residual_multiplier
        f = self.cfg.intermediate_size
        inp, outp = self.t(p + "block_sparse_moe.input_linear.weight"), self.t(p + "block_sparse_moe.output_linear.weight")
        return dict(ln1=self.t(p + "input_layernorm.weight"), ln2=self.t(p + "post_attention_layernorm.weight"),
                    q=self.t(p + "self_attn.q_proj.weight"), k=self.t(p + "self_attn.k_proj.weight"),
                    v=self.t(p + "self_attn.v_proj.weight"), o=self.t(p + "self_attn.o_proj.weight") * rm,
                    router=self.t(p + "block_sparse_moe.router.layer.weight"), bias=None,
                    experts=[(inp[e, :f], inp[e, f:], outp[e] * rm) for e in range(inp.shape[0])], shared=None,
                    dense=None)


class Glm4View(View):
    kind, score, renorm = "glm4_moe", "sigmoid", "R"

    def dims(self):
        c = self.cfg
        hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        return dict(d=c.hidden_size, L=c.num_hidden_layers, H=c.num_attention_heads, KV=c.num_key_value_heads, hd=hd,
                    E=c.n_routed_experts, k=c.num_experts_per_tok, f=c.moe_intermediate_size, eps=c.rms_norm_eps,
                    theta=c.rope_theta, prf=c.partial_rotary_factor, vocab=c.vocab_size, scale=hd ** -0.5,
                    rsf=c.routed_scaling_factor, attn_bias=c.attention_bias, dense_first=c.first_k_dense_replace,
                    shared=c.n_shared_experts, dense_f=c.intermediate_size, norm_topk_prob=c.norm_topk_prob)

    def emb(self):
        return self.t("model.embed_tokens.weight")

    def lm_head(self):
        return self.t("model.embed_tokens.weight" if self.cfg.tie_word_embeddings else "lm_head.weight")

    def final_norm(self):
        return self.t("model.norm.weight")

    def layer(self, i):
        p = f"model.layers.{i}."
        out = dict(ln1=self.t(p + "input_layernorm.weight"), ln2=self.t(p + "post_attention_layernorm.weight"))
        for n in "qkvo":
            out[n] = self.t(p + f"self_attn.{n}_proj.weight")
            if n != "o" and self.cfg.attention_bias:
                out[n + "b"] = self.t(p + f"self_attn.{n}_proj.bias")
        mlp = lambda q: tuple(self.t(f"{q}.{w}_proj.weight") for w in ("gate", "up", "down"))
        if i < self.cfg.first_k_dense_replace:
            out.update(router=None, bias=None, experts=None, shared=None, dense=mlp(p + "mlp"))
        else:
            out.update(router=self.t(p + "mlp.gate.weight"), bias=self.t(p + "mlp.gate.e_score_correction_bias"),
                       experts=[mlp(p + f"mlp.experts.{e}") for e in range(self.cfg.n_routed_experts)],
                       shared=mlp(p + "mlp.shared_experts"), dense=None)
        return out


class Dots1View(Glm4View):
    """dots.llm1: the DeepSeek-V3 router (sigmoid + e_score_correction_bias, norm_topk_prob, routed_scaling_factor) and
    a Qwen3 attention (per-head q / k RMSNorm, full RoPE): a Glm4Moe host with use_qk_norm."""
    kind = "dots1"

    def dims(self):
        c = self.cfg
        c.partial_rotary_factor = getattr(c, "partial_rotary_factor", 1.0)
        return dict(super().dims(), qk_norm=True)

    def layer(self, i):
        out = super().layer(i)
        p = f"model.layers.{i}.self_attn."
        out.update(qn=self.t(p + "q_norm.weight"), kn=self.t(p + "k_norm.weight"))
        return out


VIEWS = {"granitemoe": GraniteView, "glm4_moe": Glm4View, "dots1": Dots1View}


# --- the Glm4Moe host ------------------------------------------------------------------------------------------------

def _empty(cls, hc, dtype, device="cpu"):
    """A stock model built from its config (its __init__ computes the rotary buffers), parameters and persistent
    buffers zeroed (no weight init)."""
    from transformers.modeling_utils import no_init_weights
    hc._attn_implementation = "eager"
    with no_init_weights(), torch.device(device):
        h = cls(hc)
    h = h.to(dtype)
    with torch.no_grad():
        for v in h.state_dict().values():
            v.zero_()
    return h.eval()


def _pad(dst, src, rows=None, cols=None):
    """dst[:a, :b] = src (zero elsewhere is the caller's: dst starts at 0)."""
    if src.dim() == 1:
        dst[: src.shape[0]] = src
    else:
        dst[: src.shape[0], : src.shape[1]] = src


def glm4_host_config(D, plan):
    from transformers import Glm4MoeConfig
    k_h = plan.m * D["k"] + plan.r
    E_h = plan.m * D["E"] + plan.r + plan.x
    c, eps_h, _ = carry(D["d"], plan.e, D["eps"], plan.c)
    return Glm4MoeConfig(
        vocab_size=D["vocab"], hidden_size=D["d"] + plan.e, intermediate_size=max(D["dense_f"], 1),
        num_hidden_layers=D["L"], num_attention_heads=D["H"], num_key_value_heads=D["KV"], head_dim=D["hd"],
        partial_rotary_factor=D["prf"], rope_theta=D["theta"], max_position_embeddings=4096, rms_norm_eps=eps_h,
        n_routed_experts=E_h, num_experts_per_tok=k_h, moe_intermediate_size=D["f"], n_shared_experts=max(D["shared"], 1),
        n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=D["rsf"], first_k_dense_replace=D["dense_first"],
        use_qk_norm=D.get("qk_norm", False), attention_bias=D["attn_bias"], tie_word_embeddings=False, hidden_act="silu")


@torch.no_grad()
def glm4_host(view, plan, dtype=F32, device="cpu"):
    """The Glm4Moe host of a granitemoe / glm4_moe target under plan (module docstring)."""
    from transformers import Glm4MoeForCausalLM
    D = view.dims()
    if view.score == "softmax":
        assert plan.exp_rho is not None, "a softmax target needs the exp regime"
    h = _empty(Glm4MoeForCausalLM, glm4_host_config(D, plan), dtype, device)
    sd = h.state_dict()
    d, e = D["d"], plan.e
    c, _, wf = carry(d, e, D["eps"], plan.c)

    def norm(dst, w):
        dst[:d] = w * wf
        dst[d:] = plan.wz

    sd["model.embed_tokens.weight"][:, :d] = view.emb()
    sd["model.embed_tokens.weight"][:, d] = c
    sd["lm_head.weight"][:, :d] = view.lm_head()
    norm(sd["model.norm.weight"], view.final_norm())
    qscale = D["scale"] / D["hd"] ** -0.5
    hc_eps = h.config.rms_norm_eps
    k_t, E_t = D["k"], D["E"]
    for i in range(D["L"]):
        p, T = f"model.layers.{i}.", view.layer(i)
        norm(sd[p + "input_layernorm.weight"], T["ln1"])
        norm(sd[p + "post_attention_layernorm.weight"], T["ln2"])
        # a per-head q / k norm keeps the target eps by the row scale rs (N_eps'(rs y) = N_eps(y)); the softmax-scale
        # fold then goes into q_norm's weight instead of the q rows
        qk = "qn" in T
        rs = math.sqrt(hc_eps / D["eps"]) if qk else 1.0
        sc = {"q": rs * (1.0 if qk else qscale), "k": rs, "v": 1.0}
        for n in "qkv":
            _pad(sd[p + f"self_attn.{n}_proj.weight"], T[n] * sc[n])
            if n + "b" in T:
                sd[p + f"self_attn.{n}_proj.bias"].copy_(T[n + "b"] * sc[n])
        if qk:
            sd[p + "self_attn.q_norm.weight"].copy_(T["qn"] * qscale)
            sd[p + "self_attn.k_norm.weight"].copy_(T["kn"])
        _pad(sd[p + "self_attn.o_proj.weight"], T["o"])
        if T["dense"] is not None:
            for w, src in zip(("gate", "up", "down"), T["dense"]):
                _pad(sd[p + f"mlp.{w}_proj.weight"], src)
            continue
        W, b = sd[p + "mlp.gate.weight"], sd[p + "mlp.gate.e_score_correction_bias"]
        tb = T["bias"] if T["bias"] is not None else torch.zeros(E_t, dtype=F64)
        rho = plan.exp_rho[i] if plan.exp_rho is not None else 0.0
        for cp in range(plan.m):
            sl = slice(cp * E_t, (cp + 1) * E_t)
            W[sl, :d] = T["router"]
            W[sl, d] = -rho / plan.wz
            b[sl] = tb
            for et, (g, u, dn) in enumerate(T["experts"]):
                q = f"{p}mlp.experts.{cp * E_t + et}."
                _pad(sd[q + "gate_proj.weight"], g)
                _pad(sd[q + "up_proj.weight"], u)
                _pad(sd[q + "down_proj.weight"], dn)
        n0 = plan.m * E_t
        W[n0:n0 + plan.r, d] = -plan.M / plan.wz           # nulls: zero weights, always selected, score 0
        b[n0:n0 + plan.r] = float(tb.max()) + plan.null_bias
        W[n0 + plan.r:, d] = -plan.M / plan.wz             # extras: below every copy
        b[n0 + plan.r:] = float(tb.min()) - plan.null_bias
        if T["shared"] is not None:
            for w, src in zip(("gate", "up", "down"), T["shared"]):
                _pad(sd[p + f"mlp.shared_experts.{w}_proj.weight"], src)
    # (the rotary buffer is the target's: the same head_dim, partial factor and theta, computed by the same init)
    return h


# --- unrenormalised softmax: same-family hosts with sunk extras ----------------------------------------------------

def _head_layout(d, e, H):
    """(hd, hdh, target head dim -> host head dim map, host inv_freq writer) of a host whose head_dim is hidden / heads
    (OLMoE, Qwen2-MoE: the code checks it), so e = 2 x heads extra dims put 2 zero dims in each head: RoPE pairs
    (i, i + hd / 2) -> (i, i + hdh / 2), the host's inv_freq buffer holding the target's frequencies then 0."""
    hd = d // H
    assert e % H == 0 and (e // H) % 2 == 0
    hdh = (d + e) // H
    half, gap = hd // 2, (hdh - hd) // 2
    hmap = torch.tensor([j if j < half else j + gap for j in range(hd)])

    def rows(nh):
        return torch.cat([i * hdh + hmap for i in range(nh)])

    def write_rope(module, theta):
        inv_t = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.int64).to(F32) / hd))
        inv_h = torch.zeros(hdh // 2, dtype=module.inv_freq.dtype)
        inv_h[:half] = inv_t
        module.inv_freq.copy_(inv_h)                    # the buffer setting (hf/nope_rope.py, hf/partial_rope_host.py)
        module.original_inv_freq = module.inv_freq.clone()
    return hd, hdh, rows, write_rope


@torch.no_grad()
def qwen2moe_host(get, cfg, x, e=32, dtype=F32, device="cpu", M=M_SINK):
    """A Qwen2-MoE target (norm_topk_prob False) on a Qwen2MoeForCausalLM with x more experts and e = 2 x heads more
    hidden dims (_head_layout): the constant channel, the extras' router rows -M on it, the softmax scale hd^-1/2 ->
    hdh^-1/2 folded into the q rows and q bias; the shared expert and its sigmoid gate copied (gate column 0 on the
    channel)."""
    from transformers import Qwen2MoeForCausalLM
    d, E, H, KV = cfg.hidden_size, cfg.num_experts, cfg.num_attention_heads, cfg.num_key_value_heads
    hd, hdh, rows, write_rope = _head_layout(d, e, H)
    c, eps_h, wf = carry(d, e, cfg.rms_norm_eps)
    hc = type(cfg).from_dict({**cfg.to_dict(), "hidden_size": d + e, "num_experts": E + x, "rms_norm_eps": eps_h,
                              "head_dim": hdh, "tie_word_embeddings": False})
    h = _empty(Qwen2MoeForCausalLM, hc, dtype, device)
    sd = h.state_dict()
    rq, rkv = rows(H), rows(KV)
    sm = math.sqrt(hdh / hd)
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        g = lambda n: get(p + n).to(F64)
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            sd[p + n][:d] = g(n) * wf
            sd[p + n][d:] = 1.0
        for n, r, sc in (("q", rq, sm), ("k", rkv, 1.0), ("v", rkv, 1.0)):
            sd[p + f"self_attn.{n}_proj.weight"][r, :d] = (g(f"self_attn.{n}_proj.weight") * sc).to(dtype)
            if p + f"self_attn.{n}_proj.bias" in sd:
                sd[p + f"self_attn.{n}_proj.bias"][r] = (g(f"self_attn.{n}_proj.bias") * sc).to(dtype)
        sd[p + "self_attn.o_proj.weight"][:d, rq] = (g("self_attn.o_proj.weight")).to(dtype)
        sd[p + "mlp.gate.weight"][:E, :d] = g("mlp.gate.weight")
        sd[p + "mlp.gate.weight"][E:, d] = -M
        for ex in range(E):
            for w in ("gate_proj", "up_proj", "down_proj"):
                _pad(sd[p + f"mlp.experts.{ex}.{w}.weight"], g(f"mlp.experts.{ex}.{w}.weight"))
        for w in ("gate_proj", "up_proj", "down_proj"):
            _pad(sd[p + f"mlp.shared_expert.{w}.weight"], g(f"mlp.shared_expert.{w}.weight"))
        _pad(sd[p + "mlp.shared_expert_gate.weight"], g("mlp.shared_expert_gate.weight"))
    emb = get("model.embed_tokens.weight").to(F64)
    sd["model.embed_tokens.weight"][:, :d] = emb
    sd["model.embed_tokens.weight"][:, d] = c
    sd["lm_head.weight"][:, :d] = emb if cfg.tie_word_embeddings else get("lm_head.weight").to(F64)
    sd["model.norm.weight"][:d] = get("model.norm.weight").to(F64) * wf
    sd["model.norm.weight"][d:] = 1.0
    for mod in h.modules():
        if type(mod).__name__.endswith("RotaryEmbedding"):
            write_rope(mod, cfg.rope_theta)
    return h


@torch.no_grad()
def olmoe_host(get, cfg, x, e=32, dtype=F32, device="cpu", M=M_SINK):
    """An OLMoE target on an OlmoeForCausalLM with x more experts and e more hidden dims (head_dim = hidden / heads, so
    e = 2 x heads: each head gets 2 zero dims, pairs (i, i + hd_t / 2) -> (i, i + hd_h / 2), inv_freq written), the
    q / k norms over the padded projections kept at the target eps by the row scale s, the softmax scale folded into
    q_norm's weight."""
    from transformers import OlmoeForCausalLM
    d, E, H, KV = cfg.hidden_size, cfg.num_experts, cfg.num_attention_heads, cfg.num_key_value_heads
    hd, hdh, rows, write_rope = _head_layout(d, e, H)
    eps = cfg.rms_norm_eps
    c, eps_h, wf = carry(d, e, eps)
    hc = type(cfg).from_dict({**cfg.to_dict(), "hidden_size": d + e, "num_experts": E + x, "rms_norm_eps": eps_h,
                              "tie_word_embeddings": False})
    h = _empty(OlmoeForCausalLM, hc, dtype, device)
    sd = h.state_dict()
    rq, rkv = rows(H), rows(KV)
    # the q / k norms (over all H hdh / KV hdh projected dims): s^2 = eps_h (n_h / n_t) / eps keeps their target eps
    s_q = math.sqrt(eps_h * (H * hdh) / (H * hd) / eps)
    s_k = math.sqrt(eps_h * (KV * hdh) / (KV * hd) / eps)
    sm = math.sqrt(hdh / hd)                       # softmax scale hd^-1/2 -> hdh^-1/2, folded into q_norm
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        g = lambda n: get(p + n).to(F64)
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            sd[p + n][:d] = g(n) * wf
            sd[p + n][d:] = 1.0
        sd[p + "self_attn.q_proj.weight"][rq, :d] = (g("self_attn.q_proj.weight") * s_q).to(dtype)
        sd[p + "self_attn.k_proj.weight"][rkv, :d] = (g("self_attn.k_proj.weight") * s_k).to(dtype)
        sd[p + "self_attn.v_proj.weight"][rkv, :d] = (g("self_attn.v_proj.weight")).to(dtype)
        sd[p + "self_attn.o_proj.weight"][:d, rq] = (g("self_attn.o_proj.weight")).to(dtype)
        sd[p + "self_attn.q_norm.weight"][rq] = (g("self_attn.q_norm.weight") * math.sqrt(H * hd / (H * hdh)) * sm).to(dtype)
        sd[p + "self_attn.k_norm.weight"][rkv] = (g("self_attn.k_norm.weight") * math.sqrt(KV * hd / (KV * hdh))).to(dtype)
        sd[p + "mlp.gate.weight"][:E, :d] = g("mlp.gate.weight")
        sd[p + "mlp.gate.weight"][E:, d] = -M
        for ex in range(E):
            for w in ("gate_proj", "up_proj", "down_proj"):
                _pad(sd[p + f"mlp.experts.{ex}.{w}.weight"], g(f"mlp.experts.{ex}.{w}.weight"))
    sd["model.embed_tokens.weight"][:, :d] = get("model.embed_tokens.weight").to(F64)
    sd["model.embed_tokens.weight"][:, d] = c
    sd["lm_head.weight"][:, :d] = get("lm_head.weight").to(F64)
    sd["model.norm.weight"][:d] = get("model.norm.weight").to(F64) * wf
    sd["model.norm.weight"][d:] = 1.0
    for mod in h.modules():
        if type(mod).__name__.endswith("RotaryEmbedding"):
            write_rope(mod, cfg.rope_theta)
    return h


@torch.no_grad()
def dsv3_host(get, cfg, m, r, e=1, dtype=F32, device="cpu", M=M_SINK, null_bias=10.0):
    """A DeepSeek-V3-class target (sigmoid + e_score_correction_bias, MLA) on a DeepseekV3ForCausalLM with top-(m k + r)
    of (m E + r) experts and e more hidden dims: m whole copies of each routed expert (router row and bias copied), r
    null experts (zero weights, bias max + null_bias, router -M on the constant channel). The MLA latent norms
    (q_a_layernorm, kv_a_layernorm) are built with the class default eps 1e-6, not the config's, so the config eps'
    leaves them unchanged; every hidden-width norm takes the const_norm_carry weight."""
    from transformers import DeepseekV3ForCausalLM
    d, E, k = cfg.hidden_size, cfg.n_routed_experts, cfg.num_experts_per_tok
    c, eps_h, wf = carry(d, e, cfg.rms_norm_eps)
    hc = type(cfg).from_dict({**cfg.to_dict(), "hidden_size": d + e, "n_routed_experts": m * E + r,
                              "num_experts_per_tok": m * k + r, "rms_norm_eps": eps_h, "tie_word_embeddings": False})
    h = _empty(DeepseekV3ForCausalLM, hc, dtype, device)
    sd = h.state_dict()
    for key, v in sd.items():
        if ".mlp.experts." in key:
            i = int(key.split(".mlp.experts.")[1].split(".")[0])
            if i >= m * E:
                continue                                       # a null expert: zero weights
            src = get(key.replace(f".mlp.experts.{i}.", f".mlp.experts.{i % E}."))
        elif key.endswith("mlp.gate.weight") or key.endswith("e_score_correction_bias"):
            src = get(key)
            for cp in range(m):
                if src.dim() == 2:
                    v[cp * E:(cp + 1) * E, :d] = src.to(dtype)
                else:
                    v[cp * E:(cp + 1) * E] = src.to(dtype)
            if src.dim() == 2:
                v[m * E:, d] = -M
            else:
                v[m * E:] = float(src.max()) + null_bias
            continue
        else:
            src = get(key)
        src = src.to(F64)
        if key.endswith("input_layernorm.weight") or key.endswith("post_attention_layernorm.weight") \
                or key == "model.norm.weight":
            v[:d] = src * wf
            v[d:] = 1.0
        else:
            _pad(v, src)
    sd["model.embed_tokens.weight"][:, d] = c
    return h


class TiktokenTok:
    """A tiktoken tokenizer from a checkpoint's tiktoken.model DATA file (the pattern and special-token layout read off
    its tokenization_moonshot.py, which is not executed): tok(text, return_tensors="pt").input_ids, decode."""

    PAT = "|".join([r"""[\p{Han}]+""",
                    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                    r"""\p{N}{1,3}""", r""" ?[^\s\p{L}\p{N}]+[\r\n]*""", r"""\s*[\r\n]+""", r"""\s+(?!\S)""",
                    r"""\s+"""])

    def __init__(self, path, bos_id=None):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        ranks = load_tiktoken_bpe(path)
        self.enc = tiktoken.Encoding(name="moonshot", pat_str=self.PAT, mergeable_ranks=ranks, special_tokens={})
        self.bos = bos_id

    def __call__(self, text, return_tensors="pt"):
        ids = ([self.bos] if self.bos is not None else []) + self.enc.encode(text)

        class _B:
            input_ids = torch.tensor([ids])
        return _B()

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids if int(i) < self.enc.n_vocab])


# --- the exp regime's calibration ------------------------------------------------------------------------------------

@torch.no_grad()
def calibrate_rho(target, view, tok, prompts, plan_e=1, c=None, wz=1.0, U=U_TARGET, device="cpu"):
    """Per MoE layer the channel entry rho_l that centres the host's top copy logit z_max - s (s = rho c' q, q = 1 /
    sqrt(ms + eps) the token's inverse RMS at the router's norm) on U over the calibration prompts (minimax)."""
    D = view.dims()
    d = D["d"]
    c, _, wf = carry(d, plan_e, D["eps"], c)
    cq = c / wf                          # the copy rows' entry -rho / wz times the channel wz c / (wf sqrt(ms + eps))
    stats = {}
    hooks = []
    layers = target.model.layers
    for i, L in enumerate(layers):
        mod = getattr(L, "block_sparse_moe", None) or getattr(L, "mlp", None)
        router = getattr(getattr(mod, "router", None), "layer", None) or getattr(mod, "gate", None)
        if router is None:
            continue

        def pre(m, args, i=i):
            x = args[0].to(F64)
            stats.setdefault(i, []).append(("x", x.reshape(-1, x.shape[-1])))
        hooks.append(L.post_attention_layernorm.register_forward_pre_hook(pre))

        def rout(m, args, out, i=i):
            stats.setdefault(i, []).append(("z", out.detach().to(F64).reshape(-1, out.shape[-1])))
        hooks.append(router.register_forward_hook(rout))
    for ptxt in prompts:
        ids = tok(ptxt, return_tensors="pt").input_ids.to(device)
        target(ids)
    for hk in hooks:
        hk.remove()
    rho = [0.0] * len(layers)
    for i, rows in stats.items():
        xs = torch.cat([v for k, v in rows if k == "x"])
        zs = torch.cat([v for k, v in rows if k == "z"])
        q = 1.0 / torch.sqrt(xs.pow(2).mean(-1) + D["eps"])
        zmax = zs.max(-1).values
        # minimise max_t |zmax_t - rho cq q_t - U| over rho > 0 (the extremes bind), over the BULK of the tokens: a
        # massive-activation token (q below a tenth of the median: its rms ~100x, the attention-sink position) cannot
        # share the window, and its MoE write is small next to its own residual -- it runs in the sigmoid regime
        bulk = q >= 0.1 * q.median()
        grid = torch.logspace(-3, 8, 4000, dtype=F64)
        dev = (zmax[None, bulk] - grid[:, None] * cq * q[None, bulk] - U).abs().max(-1).values
        rho[i] = float(grid[int(dev.argmin())])
        stats[i] = dict(q_min=float(q.min()), q_max=float(q.max()), zmax=float(zmax.max()),
                        dev=float(dev.min()), outliers=int((~bulk).sum()), tokens=int(q.numel()))
    return rho, {i: v for i, v in stats.items() if isinstance(v, dict)}


# --- fp64 references (stock classes, random weights) -------------------------------------------------------------

def _rand(model, seed, std=0.08):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std + (1.0 if n.endswith("norm.weight") else 0.0))
        for n, b in model.named_buffers():
            if "e_score_correction_bias" in n:
                b.copy_(torch.randn(b.shape, generator=g, dtype=b.dtype) * 0.05)
    return model


def _tiny(mt, seed=0, **kw):
    from transformers import AutoConfig, AutoModelForCausalLM
    base = dict(vocab_size=97, hidden_size=48, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=False)
    base.update(kw)
    c = AutoConfig.for_model(mt, **base)
    c._attn_implementation = "eager"
    m = AutoModelForCausalLM.from_config(c).to(F64).eval()
    return _rand(m, seed), c


def _sd_get(model):
    sd = model.state_dict()
    return lambda n: sd[n]


@torch.no_grad()
def _compare(t, h, vocab, n=8, T=24, new=16, seed=0):
    from tests.test_universal_elephantine import _lift_fp32       # the stock fp32 casts kept in fp64
    with _lift_fp32():
        return _compare64(t, h, vocab, n, T, new, seed)


def _compare64(t, h, vocab, n, T, new, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n, T), generator=g)
    lt, lh = t(ids).logits, h(ids).logits
    rel = float((lt - lh).abs().max() / lt.abs().max())
    same = 0
    for r in range(n):
        a = t.generate(ids[r:r + 1, :8], max_new_tokens=new, do_sample=False, pad_token_id=0)
        b = h.generate(ids[r:r + 1, :8], max_new_tokens=new, do_sample=False, pad_token_id=0)
        same += bool(torch.equal(a, b))
    top1 = float((lt.argmax(-1) == lh.argmax(-1)).double().mean())
    return dict(rel=rel, max_abs=float((lt - lh).abs().max()), top1=top1, identical=same, n=n)


def ref_null(m=2, r=2, seed=0, M=M_SINK):
    """glm4_moe (sigmoid, bias, top-2 of 4) -> top-(2m + r) of (4m + r), h + 1: exact at the fp64 underflow."""
    t, tc = _tiny("glm4_moe", seed, n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=24,
                  first_k_dense_replace=1, head_dim=12, n_shared_experts=1, norm_topk_prob=True, n_group=1,
                  topk_group=1, routed_scaling_factor=1.7, partial_rotary_factor=0.5, attention_bias=True,
                  rms_norm_eps=1e-5)
    h = glm4_host(Glm4View(_sd_get(t), tc), Plan(m=m, r=r, M=M), dtype=F64)
    return _compare(t, h, tc.vocab_size)


def ref_null_dots1(m=2, r=1, seed=0, M=M_SINK):
    """dots1 (sigmoid, bias, per-head q / k norms, top-1 of 4) -> a Glm4Moe host with use_qk_norm, top-(m + r)."""
    t, tc = _tiny("dots1", seed, n_routed_experts=4, num_experts_per_tok=1, moe_intermediate_size=24,
                  first_k_dense_replace=1, n_shared_experts=1, norm_topk_prob=True, routed_scaling_factor=2.5,
                  rms_norm_eps=1e-5)
    h = glm4_host(Dots1View(_sd_get(t), tc), Plan(m=m, r=r, M=M), dtype=F64)
    return _compare(t, h, tc.vocab_size)


def ref_null_dsv3(m=1, r=2, seed=0, M=M_SINK):
    """deepseek_v3 (MLA, sigmoid + bias, top-3 of 8, 2 shared, 1 dense layer) -> DeepseekV3 top-(3m + r), h + 1."""
    t, tc = _tiny("deepseek_v3", seed, n_routed_experts=8, num_experts_per_tok=3, moe_intermediate_size=24,
                  first_k_dense_replace=1, n_shared_experts=2, norm_topk_prob=True, n_group=1, topk_group=1,
                  routed_scaling_factor=2.4, kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=8,
                  qk_nope_head_dim=8, v_head_dim=8, num_key_value_heads=4, rms_norm_eps=1e-5)
    h = dsv3_host(_sd_get(t), tc, m, r, dtype=F64, M=M)
    return _compare(t, h, tc.vocab_size)


def ref_exp(shift_scale, seed=0):
    """granitemoe (softmax R, top-2 of 4) -> Glm4Moe (sigmoid R), h + 1, every copy row -rho on the channel: the
    error falls as e^(z_max - s) until Glm4MoeTopkRouter's + 1e-20 binds (fp64 floor ~1e-10)."""
    t, tc = _tiny("granitemoe", seed, num_local_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-6,
                  embedding_multiplier=3.0, residual_multiplier=0.5, attention_multiplier=0.2, logits_scaling=2.0)
    v = GraniteView(_sd_get(t), tc)
    h = glm4_host(v, Plan(exp_rho=[shift_scale] * tc.num_hidden_layers), dtype=F64)
    return _compare(t, h, tc.vocab_size)


def ref_plain_qwen2moe(x=3, seed=0, M=M_SINK):
    t, tc = _tiny("qwen2_moe", seed, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=24,
                  shared_expert_intermediate_size=32, norm_topk_prob=False, rms_norm_eps=1e-6)
    h = qwen2moe_host(_sd_get(t), tc, x, e=8, dtype=F64, M=M)
    return _compare(t, h, tc.vocab_size)


def ref_plain_olmoe(x=3, seed=0, M=M_SINK):
    t, tc = _tiny("olmoe", seed, num_experts=4, num_experts_per_tok=2, norm_topk_prob=False, rms_norm_eps=1e-5,
                  num_key_value_heads=4)
    h = olmoe_host(_sd_get(t), tc, x, e=8, dtype=F64, M=M)
    return _compare(t, h, tc.vocab_size)


def references():
    rows = []
    for M in (0.0, 1e1, 1e3, 1e9):
        rows.append(("null m=2 r=2", M, ref_null(2, 2, M=M)))
    for M in (1e9,):
        rows.append(("null m=1 r=3", M, ref_null(1, 3, M=M)))
    for M in (0.0, 1e3, 1e9):
        rows.append(("null dots1->glm4 m=2 r=1", M, ref_null_dots1(2, 1, M=M)))
    for s in (0.0, 1.0, 10.0, 100.0, 1e3, 1e4):
        rows.append(("exp regime rho", s, ref_exp(s)))
    for M in (0.0, 1e1, 1e3, 1e9):
        rows.append(("plain extras qwen2_moe x=3", M, ref_plain_qwen2moe(3, M=M)))
        rows.append(("plain extras olmoe x=3", M, ref_plain_olmoe(3, M=M)))
    for name, par, r in rows:
        print(f"{name:30s} {par:>8g}  rel {r['rel']:.2e}  max|d| {r['max_abs']:.2e}  top-1 {r['top1']:.3f}  "
              f"identical {r['identical']}/{r['n']}", flush=True)
    return rows


# --- real checkpoints -------------------------------------------------------------------------------------------

PAIRS = {
    "exp:granite30<-glm4": ("ibm-granite/granite-3.0-1b-a400m-base", "glm4", dict(m=1, r=0)),
    "exp:granite31<-glm4": ("ibm-granite/granite-3.1-1b-a400m-base", "glm4", dict(m=1, r=0)),
    "null+exp:granite30<-glm4": ("ibm-granite/granite-3.0-1b-a400m-base", "glm4", dict(m=1, r=2)),
    "null:dots1dev<-glm4": ("Goekdeniz-Guelmez/Dots1-Dev", "glm4", dict(m=2, r=1)),
    "null:glm28<-glm4": ("imdatta0/tiny_glm4_moe_2.8B_0.7B", "glm4", dict(m=1, r=2)),
    "plain:qwen15<-qwen2moe": ("Qwen/Qwen1.5-MoE-A2.7B", "qwen2moe", dict(x=4, e=32)),
    "plain:olmoe<-olmoe": ("allenai/OLMoE-1B-7B-0924", "olmoe", dict(x=8, e=32)),
    # a TRAINED sigmoid + selection-bias target: Moonlight top-6 of 64 -> DeepseekV3 top-8 of 66 (m 1, r 2), h + 1
    "null:moonlight<-dsv3": ("moonshotai/Moonlight-16B-A3B", "dsv3", dict(m=1, r=2)),
}


# a checkpoint shipped without a tokenizer: the one its vocabulary is (dots.llm1's 152064 = Qwen2.5's)
TOKENIZER = {"Goekdeniz-Guelmez/Dots1-Dev": "Qwen/Qwen2.5-0.5B"}


def lazy_get(repo):
    """name -> tensor, read from the snapshot's safetensors files on demand (no model held)."""
    import glob
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    root = snapshot_download(repo, local_files_only=True)
    where = {}
    for f in glob.glob(os.path.join(root, "*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                where[k] = f
    handles = {}

    def get(n):
        f = where[n]
        if f not in handles:
            handles[f] = safe_open(f, "pt")
        return handles[f].get_tensor(n)
    return get


def run(pair, device="cpu", n_prompts=24, sink=None):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from act.act_swap_agreement import STORY_PROMPTS
    from hf.eps_lever_agreement import reference, score
    repo, host, kw = PAIRS[pair]
    torch.manual_seed(0)
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(repo, local_files_only=True)
    try:
        if repo == "moonshotai/Moonlight-16B-A3B":       # a tiktoken tokenizer in remote code: its data file only
            from huggingface_hub import snapshot_download
            tok = TiktokenTok(os.path.join(snapshot_download(repo, local_files_only=True), "tiktoken.model"),
                              bos_id=None)        # (its __call__ adds no BOS: no build_inputs override)
        else:
            tok = AutoTokenizer.from_pretrained(TOKENIZER.get(repo, repo), local_files_only=True)
    except ValueError:                       # a tokenizer_config written by transformers 5 (TokenizersBackend)
        from huggingface_hub import snapshot_download
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(snapshot_download(repo, local_files_only=True),
                                                                  "tokenizer.json"))
    tgt = AutoModelForCausalLM.from_pretrained(repo, dtype=F32, attn_implementation="eager",
                                               local_files_only=True).to(device).eval()
    ref = reference(tgt, tok, device, n_prompts)
    print(f"{pair}: reference done ({time.time() - t0:.0f} s); first continuation: "
          f"{tok.decode(ref['cont'][0][0][:24])!r}", flush=True)
    meta = dict(pair=pair, target=repo, host=host, **kw)
    if host == "glm4":
        view = VIEWS[cfg.model_type](_sd_get(tgt), cfg)
        rho = None
        if view.score == "softmax":
            rho, st = calibrate_rho(tgt, view, tok, STORY_PROMPTS, device=device)
            meta["rho"] = rho
            meta["calib"] = {str(i): v for i, v in st.items()}
        h = glm4_host(view, Plan(exp_rho=rho, **kw), F32, device)
        del tgt
    else:
        del tgt
        import gc
        gc.collect()
        if host == "dsv3":
            h = dsv3_host(lazy_get(repo), cfg, kw["m"], kw["r"], dtype=F32, device=device)
        else:
            build = qwen2moe_host if host == "qwen2moe" else olmoe_host
            h = build(lazy_get(repo), cfg, kw["x"], e=kw["e"], dtype=F32, device=device)
    r = score(h, ref, device)
    r.update(meta, payable=bool(r["top1"] >= 0.99 and r["ex32"] >= 0.90), secs=round(time.time() - t0),
             transformers=__import__("transformers").__version__, n_prompts=n_prompts)
    print(f"{pair}: top1 {r['top1']:.4f} ex32 {r['ex32']:.2f} ex64 {r['ex64']:.2f} kl {r['kl']:.2e} "
          f"max|dlogp| {r['max_dlogp']:.2e} payable {r['payable']}", flush=True)
    if sink:
        sink(r)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="?", choices=sorted(PAIRS))
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.ref:
        references()
    if a.pair:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)

        def sink(r):
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
        run(a.pair, a.device, a.n_prompts, sink)
