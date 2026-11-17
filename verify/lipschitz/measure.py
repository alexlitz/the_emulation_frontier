"""Coverage metric M12: the whole-model limit theorems, evaluated with MEASURED constants.

The Lean library (proofs/lean: ModelLimit.lean, LimitKit.lean,
LimitInstances.lean, LimitInstances2.lean, EpsLever.lean) proves, for a stack of layers f_i:

    host logits within  Lh * errBound(L, eps)  <=  Lh * (prod_i L_i) * (sum_i eps_i)      (model_error_bound(_head),
                                                                                         errBound_le_prod_sum)
    identical greedy text once 2 * Lh * (prod L) * (sum eps) < delta                   (greedy_of_bound,
                                                                                         greedy_of_layer_bounds(_on))

with delta the target's top-1 margin along its own greedy trajectory. Every per-layer error eps_i is an explicit
function of one host parameter (1/beta, theta^2, 1/c^2, e^-B, e^b, 1/S^2, e^{-beta delta_r}), so each limit
construction has an explicit threshold. The Lipschitz constants L_i, Lh, the value bounds (M, R, |pre|, router
margins) and delta are HYPOTHESES of those theorems. This script measures them on real checkpoints and evaluates
every threshold, then measures the host error the construction actually makes (teacher-forced on the target's
own greedy continuation: the quantity the theorem bounds).

Norms. The state is the residual of every position; dist = max over positions of the per-position l2 norm (the
Pi sup norm of Lean's `Fin T -> V`, with V = R^d in l2). Logits: sup norm over the vocabulary (`pick_eq_of_margin`).
  * a layer's constant L_i = sup over unit perturbations (every position's l2 norm <= 1) of the output change at the
    worst position: for query t this is the (inf,2)->2 norm of the row block J_t = d y_t / d x. We measure, at real
    activations, both the l2 spectral norm of J_t ("spec", power iteration with autograd JVP / VJP) and the mixed
    (inf,2) norm ("mix", the same iteration with a per-position normalisation: a monotone ascent, a lower bound).
    FFN maps act per position, so for them spec = mix = the per-token Jacobian's spectral norm.
  * the head's constant Lh (final norm + unembedding, last position) from l2 to sup-norm logits:
    max_v || J_norm(x)^T w_v ||_2, computed in closed form for RMSNorm / LayerNorm (checked against autograd).
LOCAL constants. The theorems need Lipschitz constants valid on the whole region the host's states can reach
(every state within the error radius of the target trajectory). A Jacobian norm at a real activation is the local
Lipschitz constant there, so it LOWER-BOUNDS the constant the theorem needs: every threshold below is a lower
bound on the true theorem threshold (the theorem demands at least this much).

Layers. The stack is taken at block granularity (one decoder layer = one f_i; the composition of the attention and
FFN residual maps). An edit inside the FFN changes the block by eps_ffn; an edit inside attention by at most
L_ffn * eps_attn (the FFN residual map's constant), except for parallel blocks (Pythia), where it is eps_attn.

Rules evaluated (eps_i = kappa_i * g(param); threshold g* = delta / (2 Lh prod(max(L_i, 1)) sum kappa_i), the
"sharp" form uses errBound instead of the product):
  relu_silu      ReLU target (OPT-125m) on a SiLU host, silu(beta x)/beta:   g = 1/beta,  kappa = T sum_n ||W_down[:, n]||
                 (`relu_silu_greedy`; Lean's units range over the whole state: T positions x neurons)
  act_swap       any activation swap within eta (`act_swap_greedy`): g = eta, same kappa (non-gated MLPs)
  alibi_rope     ALiBi target (BLOOM-560m) on a RoPE host at angle theta (`alibi_rope_greedy`):
                 g = theta^2, kappa = L_ffn (2/3) mbar T^3 M_i   (valid while mbar T^3 theta^2 <= 3)
  softcap_host   plain target on an attention-softcap host, cap c (`cap_greedy`): g = 1/c^2,
                 kappa = L_ffn (4/3) R_i^3 M_i   (valid while 2 R^3 <= 3 c^2)
  mask           hard mask from an additive -B (`mask_greedy`): g = e^-B, kappa = L_ffn 2 M_i T e^{2 R_i}
  sink           plain target on a sink host, sink logit b (`sink_greedy`): g = e^b, kappa = L_ffn 2 M_i e^{R_i}
  soft_router    top-1 MoE target (Huihui-MoE-0.8B-2E) on a dense soft-router host (`soft_router_greedy`):
                 g = e^{-beta delta_r}, kappa = 2 M_i (E - 1)
  lever_qk       qk-norm switched off on the host (Qwen3 host <- Llama target; `leverAttn_greedy`): g = 1/S^2 (S = 2^a),
                 kappa = L_ffn sum_h O_h Vb (2 C_B Q K (Q^2 + K^2) + Vb^2 / 2) / eps
  lever_post     post-norms switched off (sandwich host <- plain target; `postNorm_greedy`): g = 1/S^2,
                 kappa = sqrt(D) (L_ffn R_attn^3 + R_ffn^3) / (2 eps)  (R: sup norm of the sublayer output; Lean's
                 bound is in the sup norm, sqrt(D) converts it to l2)
M_i = sum over heads of max_j ||W_O,h v_h(x_j)||_2 (Lean's lemmas are one head; heads add), R_i = max |score|.

  python -m verify.lipschitz.measure smollm2-135m --device cuda:0
  python -m verify.lipschitz.measure all --device cuda:0
  python -m verify.lipschitz.measure --report          # writes the tables of LIPSCHITZ.md from out/*.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from act.act_swap_agreement import N_NEW, PROMPTS, greedy

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "pythia-160m": "EleutherAI/pythia-160m",
    "gpt2": "openai-community/gpt2",
    "gemma-2-2b": "google/gemma-2-2b",
    "olmo-2-1b": "allenai/OLMo-2-0425-1B",
    "opt-125m": "facebook/opt-125m",                       # the ReLU target
    "bloom-560m": "bigscience/bloom-560m",                 # the ALiBi target
    "huihui-moe-0.8b-2e": "huihui-ai/Huihui-MoE-0.8B-2E",  # the top-1 MoE target
}

# the settings the constructions use (and where they come from)
CONSTRUCTION = {
    "relu_silu": dict(beta=3417.2, src="act/data/act_swap_agreement.jsonl: the SiLU-host fit of OPT-125m's ReLU, "
                                        "a SiLU(c x + d) with c = 3417 (top-1 100%, 64/64 identical)"),
    "alibi_rope": dict(theta=(1e-2, 6.25e-4), src="attn/positional_cross.py: theta 1e-2 ... 6.25e-4 (synthetic); "
                                                  "no real-model ALiBi-on-RoPE host is built"),
    "softcap_host": dict(c_over_zmax=5.0, src="attn/regime_agreement.py, tests/test_regime_agreement.py: every plain "
                                              "model passes for c >= 5 max|z|"),
    "mask": dict(B=float(torch.finfo(torch.float32).max), src="HF / elephantine: masked logits at finfo.min or -inf "
                                                              "(exp underflows to exactly 0)"),
    "sink": dict(b=float("-inf"), src="elephantine/universal.py: null_logit = -inf"),
    "soft_router": dict(beta=1e7, src="moe/dense_host_prenorm.py (branch moe-loop-rescue): select-unit sharpness "
                                      "beta = 1e7 (soft zone |t| <~ 1e-5); top-1 100%, 64/64 identical on Huihui"),
    "lever_qk": dict(rho=2.0 ** -30, src="graph/eps_lever.py (branch measure-batch2): a = ceil((log2(ms_max/eps) + 30)/2)"
                                         "; a = 26 for SmolLM2 and Llama-3.2 (100% / 100%, max|dlogp| ~9e-5 fp32)"),
    "lever_post": dict(rho=2.0 ** -30, src="graph/eps_lever.py (branch measure-batch2): same rule; a = 32 on gemma-2b"),
}

# --- families -------------------------------------------------------------------------------------------------
FAM = {
    "llama": dict(layers="model.layers", attn="self_attn", mlp="mlp", final="model.norm", down="mlp.down_proj"),
    "qwen2": dict(layers="model.layers", attn="self_attn", mlp="mlp", final="model.norm", down="mlp.down_proj"),
    "olmo2": dict(layers="model.layers", attn="self_attn", mlp="mlp", final="model.norm", down="mlp.down_proj"),
    "gemma2": dict(layers="model.layers", attn="self_attn", mlp="mlp", final="model.norm", down="mlp.down_proj"),
    "qwen3_moe": dict(layers="model.layers", attn="self_attn", mlp="mlp", final="model.norm", down=None),
    "gpt2": dict(layers="transformer.h", attn="attn", mlp="mlp", final="transformer.ln_f", down="mlp.c_proj"),
    "gpt_neox": dict(layers="gpt_neox.layers", attn="attention", mlp="mlp", final="gpt_neox.final_layer_norm",
                     down="mlp.dense_4h_to_h"),
    "opt": dict(layers="model.decoder.layers", attn="self_attn", mlp="fc2", final="model.decoder.final_layer_norm",
                down="fc2"),
    "bloom": dict(layers="transformer.h", attn="self_attention", mlp="mlp", final="transformer.ln_f",
                  down="mlp.dense_4h_to_h"),
}
LLAMA_LIKE = ("llama", "qwen2", "olmo2")          # host simulations (manual eager attention) are built for these
ACT_NAMES = ("act_fn", "act", "activation_fn", "gelu_impl")
OPROJ = ("o_proj", "c_proj", "dense", "out_proj")


def _get(obj, path):
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def _rmax(a, b):
    return max(float(a), float(b))


# --- global state: recording and host variants ---------------------------------------------------------------
S = types.SimpleNamespace(rec=None, host=None, zero=None, idx={}, fam=None, cfg=None)


def _new_rec():
    return dict(zmax=0.0, M=0.0, q_inf=0.0, k_inf=0.0, v_inf=0.0, q_ms=0.0, k_ms=0.0, ya_inf=0.0, ya_l2=0.0,
                ya_ms=0.0, yf_inf=0.0, yf_l2=0.0, yf_ms=0.0, pre_max=0.0, up_max=0.0, router_margin_min=float("inf"),
                router_margin=[], M_exp=0.0, O_h=[], M_h=[])


def _head_out(module, v):
    """per-head outputs W_O,h v_h: v (B, H, T, hd) (already one per query head) -> (B, H, T, d)."""
    for n in OPROJ:
        o = getattr(module, n, None)
        if o is not None:
            break
    W = o.weight
    H, hd = v.shape[1], v.shape[-1]
    if type(o).__name__ == "Conv1D":            # gpt2: y = x W, W (H hd, d)
        Wh = W.reshape(H, hd, -1)
    else:                                       # nn.Linear: W (d, H hd)
        Wh = W.t().reshape(H, hd, -1)
    return torch.einsum("bhtk,hkd->bhtd", v, Wh), Wh


def _record_attn(module, q, k, v, scaling, softcap, alibi_rel=None):
    """q (B, H, T, hd), k / v (B, Hkv, T, hd), all after RoPE: score and value bounds of one layer."""
    r = S.rec[S.idx[id(module)]]
    H, Hkv = q.shape[1], k.shape[1]
    kk = k.repeat_interleave(H // Hkv, 1)
    vv = v.repeat_interleave(H // Hkv, 1)
    z = torch.matmul(q, kk.transpose(-1, -2)) * scaling
    if alibi_rel is not None:
        z = z + alibi_rel
    T = z.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=z.device).tril()
    r["zmax"] = _rmax(r["zmax"], z.abs().masked_fill(~causal, 0).max())
    ho, Wh = _head_out(module, vv)
    Mh = ho.norm(dim=-1).amax(dim=(0, 2))                       # (H,)
    r["M_h"] = Mh.tolist() if not r["M_h"] else [max(a, b) for a, b in zip(r["M_h"], Mh.tolist())]
    r["M"] = sum(r["M_h"])
    if not r["O_h"]:                                            # ||W_O,h|| from l_inf(hd) to l2(d): <= min(sqrt(hd) ||W||_2, sum of rows)
        sv = torch.linalg.matrix_norm(Wh.double(), ord=2)
        rows = Wh.double().norm(dim=-1).sum(-1)
        r["O_h"] = torch.minimum(sv * math.sqrt(Wh.shape[1]), rows).tolist()


def make_eager(orig, fam):
    def fn(module, query, key, value, attention_mask, *args, **kw):
        if S.rec is None and S.host is None:
            return orig(module, query, key, value, attention_mask, *args, **kw)
        if fam == "gpt2":
            scaling = (value.shape[-1] ** -0.5 if module.scale_attn_weights else 1.0)
            if module.scale_attn_by_inverse_layer_idx:
                scaling /= float(module.layer_idx + 1)
        else:
            scaling = kw.get("scaling", None)
            if scaling is None:
                scaling = module.head_dim ** -0.5
        softcap = kw.get("softcap", None)
        if S.rec is not None:
            with torch.no_grad():
                _record_attn(module, query, key, value, scaling, softcap)
        if S.host is None:
            return orig(module, query, key, value, attention_mask, *args, **kw)
        return _host_attention(module, query, key, value, scaling, softcap)
    return fn


def _host_attention(module, q, k, v, scaling, softcap):
    """eager causal attention with the host edits of S.host (llama-like families)."""
    h = S.host
    if "qk_S" in h:          # the eps lever, exact identity: S sqrt(e) N_e(x / S) = x / sqrt(1 + ||x||^2 / (d S^2 e))
        Sv, e = h["qk_S"], h["eps"]
        q = q / torch.sqrt(1 + q.pow(2).sum(-1, keepdim=True) / (q.shape[-1] * Sv ** 2 * e))
        k = k / torch.sqrt(1 + k.pow(2).sum(-1, keepdim=True) / (k.shape[-1] * Sv ** 2 * e))
    H, Hkv = q.shape[1], k.shape[1]
    k = k.repeat_interleave(H // Hkv, 1)
    v = v.repeat_interleave(H // Hkv, 1)
    s = torch.matmul(q, k.transpose(-1, -2)) * scaling
    if softcap is not None:
        s = softcap * torch.tanh(s / softcap)
    if "cap" in h:
        s = h["cap"] * torch.tanh(s / h["cap"])
    T = s.shape[-1]
    allowed = torch.ones(T, T, dtype=torch.bool, device=s.device).tril()
    if "mask_B" in h:        # full attention, -B on the keys outside the (causal) mask
        s = s - h["mask_B"] * (~allowed).to(s.dtype)
    else:
        s = s.masked_fill(~allowed, float("-inf"))
    s = s.float()
    if "sink_b" in h:        # one extra key with logit b and value 0
        s = torch.cat([s, torch.full_like(s[..., :1], h["sink_b"])], -1)
        p = torch.softmax(s, -1)[..., :-1]
    else:
        p = torch.softmax(s, -1)
    p = p.to(q.dtype)
    return torch.matmul(p, v).transpose(1, 2).contiguous(), p


def install_attention(model, fam):
    import importlib
    mod = importlib.import_module(type(_get(model, FAM[fam]["layers"])[0]).__module__)
    if fam == "bloom":
        return
    if not getattr(mod, "_lip_patched", False):
        orig = mod.eager_attention_forward
        mod.eager_attention_forward = make_eager(orig, fam)
        mod._lip_patched = True


# --- per-layer hooks: inputs (for the Jacobians), sublayer outputs, activations, zeroing -----------------------
class Hooks:
    def __init__(self, model, fam):
        self.model, self.fam = model, fam
        self.layers = list(_get(model, FAM[fam]["layers"]))
        self.capture = None            # dict i -> (args, kwargs) when capturing layer inputs
        self.hs = []
        for i, L in enumerate(self.layers):
            a = _get(L, FAM[fam]["attn"])
            m = _get(L, FAM[fam]["mlp"])
            S.idx[id(a)] = i
            self.hs.append(L.register_forward_pre_hook(self._pre(i), with_kwargs=True))
            self.hs.append(a.register_forward_hook(self._attn_out(i), with_kwargs=True))
            self.hs.append(m.register_forward_hook(self._mlp_out(i), with_kwargs=True))
            for n, sub in L.named_modules():
                if n.split(".")[-1] in ACT_NAMES:
                    self.hs.append(sub.register_forward_pre_hook(self._act(i)))
                if n.split(".")[-1] == "up_proj":
                    self.hs.append(sub.register_forward_hook(self._up(i)))
                if fam in LLAMA_LIKE and n.split(".")[-1] in ("q_proj", "k_proj", "v_proj"):
                    self.hs.append(sub.register_forward_hook(self._qkv(i, n.split(".")[-1][0], a)))
            if fam == "bloom":
                self.hs.append(a.register_forward_hook(self._bloom_attn(i), with_kwargs=True))

    def remove(self):
        for h in self.hs:
            h.remove()

    def _pre(self, i):
        def f(mod, args, kwargs):
            if self.capture is not None:
                kw = {k: v for k, v in kwargs.items()}
                self.capture[i] = (tuple(args), kw)
        return f

    @staticmethod
    def _resid(args, kwargs):
        return args[1] if len(args) > 1 else kwargs["residual"]

    def _attn_out(self, i):
        def f(mod, args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            if self.fam == "bloom":
                y = o - self._resid(args, kwargs)
            else:
                y = o
            if S.rec is not None:
                with torch.no_grad():
                    r = S.rec[i]
                    r["ya_inf"] = _rmax(r["ya_inf"], y.abs().max())
                    r["ya_l2"] = _rmax(r["ya_l2"], y.norm(dim=-1).max())
                    r["ya_ms"] = _rmax(r["ya_ms"], y.pow(2).mean(-1).max())
            if S.zero == ("attn", i) or (S.zero == ("attn", "all")):
                z = self._resid(args, kwargs) if self.fam == "bloom" else torch.zeros_like(o)
                return (z,) + tuple(out[1:]) if isinstance(out, tuple) else z
            if S.host is not None and "post_S" in S.host and self.fam in LLAMA_LIKE:
                o = _post_lever(o)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return f

    def _mlp_out(self, i):
        def f(mod, args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            y = o - self._resid(args, kwargs) if self.fam == "bloom" else o
            if S.rec is not None:
                with torch.no_grad():
                    r = S.rec[i]
                    r["yf_inf"] = _rmax(r["yf_inf"], y.abs().max())
                    r["yf_l2"] = _rmax(r["yf_l2"], y.norm(dim=-1).max())
                    r["yf_ms"] = _rmax(r["yf_ms"], y.pow(2).mean(-1).max())
            if S.zero == ("mlp", i):
                z = self._resid(args, kwargs) if self.fam == "bloom" else torch.zeros_like(o)
                return (z,) + tuple(out[1:]) if isinstance(out, tuple) else z
            if S.host is not None and "post_S" in S.host and self.fam in LLAMA_LIKE:
                o = _post_lever(o)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return f

    def _act(self, i):
        def f(mod, args):
            if S.rec is not None:
                S.rec[i]["pre_max"] = _rmax(S.rec[i]["pre_max"], args[0].detach().abs().max())
        return f

    def _up(self, i):
        def f(mod, args, out):
            if S.rec is not None:
                S.rec[i]["up_max"] = _rmax(S.rec[i]["up_max"], out.detach().abs().max())
        return f

    def _qkv(self, i, which, attn):
        hd = getattr(attn, "head_dim", None)

        def f(mod, args, out):
            if S.rec is None:
                return
            with torch.no_grad():
                y = out.reshape(*out.shape[:-1], -1, hd)            # (B, T, heads, hd), pre-RoPE
                r = S.rec[i]
                r[which + "_inf"] = _rmax(r[which + "_inf"], y.abs().max())
                if which in "qk":
                    r[which + "_ms"] = _rmax(r[which + "_ms"], y.pow(2).mean(-1).max())
        return f

    def _bloom_attn(self, i):
        def f(mod, args, kwargs, out):
            if S.rec is None:
                return
            with torch.no_grad():
                hs = args[0] if args else kwargs["hidden_states"]
                alibi = kwargs.get("alibi", args[2] if len(args) > 2 else None)
                q, k, v = mod._reshape(mod.query_key_value(hs))          # (B, H, T, hd)
                B, H, T, _ = q.shape
                slopes = alibi.reshape(B, H, -1)[0, :, 1] if alibi.shape[-1] > 1 else None
                pos = torch.arange(T, device=q.device, dtype=q.dtype)
                rel = slopes[:, None, None] * (pos[None, None, :] - pos[None, :, None]) if slopes is not None else None
                _record_attn(mod, q, k, v, mod.inv_norm_factor, None, alibi_rel=rel)
        return f


def _post_lever(y):
    Sv, e = S.host["post_S"], S.host["eps"]
    return y / torch.sqrt(1 + y.pow(2).sum(-1, keepdim=True) / (y.shape[-1] * Sv ** 2 * e))


# --- MoE (qwen3_moe, top-1): a functional block, exact for the target, with the soft-router host -----------------
def moe_forward(self, hidden_states):
    B, T, D = hidden_states.shape
    h = hidden_states.reshape(-1, D)
    z = self.gate(h)                                                    # (N, E)
    Y = torch.stack([e(h) for e in self.experts], 1)                    # (N, E, D): every expert (dense)
    i = S.idx_moe[id(self)]
    if S.rec is not None:
        with torch.no_grad():
            zs = z.float().sort(-1, descending=True).values
            mg = (zs[:, 0] - zs[:, 1])
            r = S.rec[i]
            r["router_margin_min"] = min(r["router_margin_min"], float(mg.min()))
            r["router_margin"].append(mg.cpu())
            r["M_exp"] = _rmax(r["M_exp"], Y.norm(dim=-1).max())
    if S.host is not None and "router_beta" in S.host:
        w = torch.softmax(S.host["router_beta"] * z.double(), -1).to(h.dtype)
    else:                                                               # top-1, renormalised: gate exactly 1
        assert self.top_k == 1 and self.norm_topk_prob
        w = F.one_hot(z.argmax(-1), z.shape[-1]).to(h.dtype)
    out = (w[..., None] * Y).sum(1).reshape(B, T, D)
    return out, z


# --- loading -----------------------------------------------------------------------------------------------------
def load(key, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = MODELS.get(key, key)
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, attn_implementation="eager").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    fam = cfg.model_type
    assert fam in FAM, fam
    if fam == "gemma2":                      # the final cap is strictly increasing: never moves the argmax
        cfg.final_logit_softcapping = None   # (final_softcap_greedy); margins and Lh are taken before it
    nbytes = sum(p.numel() * 4 for p in model.parameters())
    on_gpu = False
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(torch.device(device))[0]
        if nbytes + 3e9 < free:                 # the GPUs are shared: leave room for the Jacobian passes
            model.to(device)
            on_gpu = True
    S.fam, S.cfg = fam, cfg
    S.idx, S.idx_moe = {}, {}
    install_attention(model, fam)
    if fam == "qwen3_moe":
        for i, L in enumerate(_get(model, FAM[fam]["layers"])):
            L.mlp.forward = types.MethodType(moe_forward, L.mlp)
            S.idx_moe[id(L.mlp)] = i
    return tok, model, fam, on_gpu


def model_device(model):
    return next(model.parameters()).device


# --- 1. the target's greedy trajectory and its margins -----------------------------------------------------------
@torch.no_grad()
def references(model, tok, prompts):
    dev = model_device(model)
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        cont = greedy(model, ids)
        seq = torch.cat([ids, cont], 1)
        out.append((seq.cpu(), ids.shape[1]))
    return out


@torch.no_grad()
def trajectory_logits(model, seq, P):
    """logits at the 64 positions that predict the continuation (P-1 ... T-2), fp32, on the model's device."""
    return model(seq.to(model_device(model))).logits[0, P - 1:-1].float()


def margins(logits, toks):
    """delta at each step: the trajectory token's logit minus the best other (negative if it is not the argmax)."""
    a = logits.gather(1, toks[:, None])[:, 0]
    other = logits.scatter(1, toks[:, None], float("-inf")).max(1).values
    return a - other


# --- 2. the head's constant Lh (l2 state -> sup-norm logits), closed form ----------------------------------------
def final_norm_info(model, fam):
    nm = _get(model, FAM[fam]["final"])
    cls = type(nm).__name__
    if isinstance(nm, torch.nn.LayerNorm) or "LayerNorm" in cls and hasattr(nm, "bias") and nm.bias is not None:
        return nm, "ln", nm.weight.detach().float(), float(nm.eps)
    eps = float(getattr(nm, "variance_epsilon", getattr(nm, "eps", 1e-6)))
    w = nm.weight.detach().float()
    if fam == "gemma2":
        w = 1.0 + w
    return nm, "rms", w, eps


@torch.no_grad()
def head_constants(W, kind, g, eps, X, chunk=64):
    """Lh at each row of X (N, d): max_v || J_norm(x)^T w_v ||_2. W (V, d) on the compute device."""
    d = X.shape[1]
    A = W * g[None, :]                                   # rows a_v = g * w_v
    a2 = A.pow(2).sum(1)                                 # ||a_v||^2
    s1 = A.sum(1)                                        # 1 . a_v
    out = []
    for c in range(0, X.shape[0], chunk):
        x = X[c:c + chunk].to(A)
        if kind == "rms":
            r2 = x.pow(2).mean(1) + eps                  # r^2
            b = x @ A.t()                                # (n, V): a_v . x
            xx = x.pow(2).sum(1)
            v = (a2[None] / r2[:, None] - 2 * b.pow(2) / (d * r2[:, None] ** 2)
                 + xx[:, None] * b.pow(2) / (d * d * r2[:, None] ** 3))
        else:
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1)
            sig2 = var + eps
            xh = (x - mu) / sig2.sqrt()[:, None]
            kap = xh.pow(2).sum(1) / d
            p = xh @ A.t()
            v = (a2[None] - s1[None].pow(2) / d - p.pow(2) * (2 - kap[:, None]) / d) / sig2[:, None]
        out.append(v.clamp_min(0).max(1).values.sqrt().cpu())
    return torch.cat(out)


def head_constant_autograd(norm_mod, W, x):
    """the same quantity by autograd, for one row x (d,): for the test."""
    J = torch.autograd.functional.jacobian(lambda y: norm_mod(y[None])[0], x)
    return (W @ J).norm(dim=1).max()


# --- 3. capture: one full forward per prompt, recording everything ----------------------------------------------
@torch.no_grad()
def capture(model, hooks, seq, keep_inputs):
    hooks.capture = {} if keep_inputs else None
    fn = []

    nm = _get(model, FAM[hooks.fam]["final"])
    h = nm.register_forward_pre_hook(lambda m, a: fn.append(a[0].detach()))
    out = model(seq.to(model_device(model)), use_cache=False)
    h.remove()
    caps = hooks.capture
    hooks.capture = None
    return out.logits[0].float(), fn[0][0], caps


# --- 4. Jacobian norms by power iteration (autograd JVP / VJP) ----------------------------------------------------
def _call(layer, X, cap):
    args, kw = cap
    kw = dict(kw)
    B = X.shape[0]
    if "alibi" in kw and kw["alibi"] is not None and B > 1:
        kw["alibi"] = kw["alibi"].repeat(B, 1, 1)
    for k in ("use_cache", "output_attentions"):
        if k in kw:
            kw[k] = False
    if args:
        a = list(args[1:])
        if S.fam == "bloom" and len(a) >= 1 and B > 1 and torch.is_tensor(a[0]) and a[0].dim() == 3:
            a[0] = a[0].repeat(B, 1, 1)
        out = layer(X, *a, **kw)
    else:
        kw["hidden_states"] = X
        out = layer(**kw)
    return out[0] if isinstance(out, tuple) else out


def _to(cap, dev):
    args, kw = cap

    def mv(v):
        if torch.is_tensor(v):
            return v.to(dev)
        if isinstance(v, tuple):
            return tuple(mv(u) for u in v)
        return v
    return tuple(mv(a) for a in args), {k: mv(v) for k, v in kw.items()}


def _layer_fn(layer, cap, zero):
    def f(X):
        S.zero = zero
        try:
            return _call(layer, X, cap)
        finally:
            S.zero = None
    return f


def per_token_norms(f, X, pos, n_l2=20, n_mix=10, seed=0):
    """Row-block norms of the Jacobian of f (B, T, d) -> (B, T, d) at X (1, T, d), for query positions `pos`:
    spec[t] = ||J_t||_2 (l2 power iteration), mix[t] = a lower bound on the (inf,2)->2 norm of J_t
    (power iteration with per-position normalisation, started from the l2 vector)."""
    B, T, D = len(pos), X.shape[1], X.shape[2]
    Xb = X.expand(B, T, D).contiguous()
    tp = torch.tensor(pos, device=X.device)
    ar = torch.arange(B, device=X.device)
    causal = (torch.arange(T, device=X.device)[None, :] <= tp[:, None]).to(X.dtype)[..., None]
    out, vjp_fn = torch.func.vjp(f, Xb)

    def J(d):
        return torch.func.jvp(f, (Xb,), (d,))[1][ar, tp]

    def JT(u):
        c = torch.zeros_like(out)
        c[ar, tp] = u
        return vjp_fn(c)[0] * causal

    gen = torch.Generator(device=X.device).manual_seed(seed)
    d = torch.randn(B, T, D, device=X.device, generator=gen, dtype=X.dtype) * causal
    d = d / d.flatten(1).norm(dim=1)[:, None, None]
    prev = None
    for _ in range(n_l2):
        u = J(d)
        prev = u.norm(dim=-1)
        g = JT(u)
        d = g / g.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None]
    spec = J(d).norm(dim=-1)
    conv = float(((spec - prev).abs() / spec.clamp_min(1e-30)).max())
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-30) * causal
    d = d * (d.norm(dim=-1, keepdim=True) > 0)
    best = J(d).norm(dim=-1)
    for _ in range(n_mix):
        u = J(d)
        best = torch.maximum(best, u.norm(dim=-1))
        g = JT(u)
        d = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        d = d * (g.norm(dim=-1, keepdim=True) > 0)
    best = torch.maximum(best, J(d).norm(dim=-1))
    return spec.detach().cpu(), torch.maximum(best, spec).detach().cpu(), conv


def per_position_norms(f, X, n_it=25, seed=0):
    """For a map acting per position (the FFN residual map): the spectral norm of every token's Jacobian at once."""
    out, vjp_fn = torch.func.vjp(f, X)
    gen = torch.Generator(device=X.device).manual_seed(seed)
    d = torch.randn(X.shape, device=X.device, generator=gen, dtype=X.dtype)
    d = d / d.norm(dim=-1, keepdim=True)
    prev = None
    for _ in range(n_it):
        u = torch.func.jvp(f, (X,), (d,))[1]
        prev = u.norm(dim=-1)
        g = vjp_fn(u)[0]
        d = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    s = torch.func.jvp(f, (X,), (d,))[1].norm(dim=-1)
    conv = float(((s - prev).abs() / s.clamp_min(1e-30)).max())
    return s[0].detach().cpu(), conv


def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(max=float(x.max()), p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), n=int(x.size))


def _oom_retry(fn, *a, tries=6, **kw):
    for t in range(tries):
        try:
            return fn(*a, **kw)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            time.sleep(10 * (t + 1))
    return fn(*a, **kw)


def jacobians(model, hooks, caps_by_prompt, dev, stride=4, chunk=16, n_l2=20, n_mix=10, log=print):
    """per layer: FFN map (every token), attention map and block (query positions at `stride`), spec + mix."""
    fam = hooks.fam
    nL = len(hooks.layers)
    res = []
    for i, layer in enumerate(hooks.layers):
        t0 = time.time()
        home = next(layer.parameters()).device
        try:
            out = _layer_norms(layer, i, caps_by_prompt, dev, stride, chunk, n_l2, n_mix)
        except torch.OutOfMemoryError:            # the GPUs are shared: a layer that gets no room runs on the CPU
            layer.to("cpu")
            torch.cuda.empty_cache()
            log(f"  layer {i + 1}/{nL}: out of GPU memory, on the CPU")
            out = _layer_norms(layer, i, caps_by_prompt, "cpu", stride, chunk, n_l2, n_mix)
        if next(layer.parameters()).device != home:
            layer.to(home)
            torch.cuda.empty_cache()
        res.append(out)
        log(f"  layer {i + 1}/{nL}: L_ffn {res[-1]['ffn']['max']:.3g}  L_attn {res[-1]['attn_mix']['max']:.3g} "
            f"(spec {res[-1]['attn_spec']['max']:.3g})  L_block {res[-1]['block_mix']['max']:.3g} "
            f"(spec {res[-1]['block_spec']['max']:.3g})  [{time.time() - t0:.0f}s]")
    return res


def _layer_norms(layer, i, caps_by_prompt, dev, stride, chunk, n_l2, n_mix):
    """one layer's FFN, attention and block norms on `dev` (the layer is moved there)."""
    if next(layer.parameters()).device != torch.device(dev):
        layer.to(dev)
    ffn, attn_s, attn_m, blk_s, blk_m, convs = [], [], [], [], [], []
    for caps in caps_by_prompt:
        cap = _to(caps[i], dev)
        X = (cap[0][0] if cap[0] else cap[1]["hidden_states"]).detach().to(dev)
        T = X.shape[1]
        s, c = _oom_retry(per_position_norms, _layer_fn(layer, cap, ("attn", i)), X, n_it=n_l2 + 5)
        ffn.append(s)
        convs.append(c)
        pos = sorted(set(list(range(0, T, stride)) + [T - 1]))
        for zero, acc_s, acc_m in ((("mlp", i), attn_s, attn_m), (None, blk_s, blk_m)):
            for c0 in range(0, len(pos), chunk):
                sp, mx, c = _oom_retry(per_token_norms, _layer_fn(layer, cap, zero), X, pos[c0:c0 + chunk],
                                       n_l2=n_l2, n_mix=n_mix)
                acc_s.append(sp)
                acc_m.append(mx)
                convs.append(c)
    cat = lambda xs: torch.cat(xs).numpy()
    return dict(ffn=_pct(cat(ffn)), attn_spec=_pct(cat(attn_s)), attn_mix=_pct(cat(attn_m)),
                block_spec=_pct(cat(blk_s)), block_mix=_pct(cat(blk_m)), conv_max=max(convs))


# --- 5. host simulations: the error each construction actually makes -------------------------------------------
@torch.no_grad()
def observed(model, refs, tgt_logits, setup, variants, log=print):
    """for each variant: max sup-norm logit error at the target's 64 trajectory positions (teacher-forced), top-1
    agreement (100% <=> identical greedy text), and the fraction of prompts identical."""
    out = []
    for v in variants:
        undo = setup(v)
        err, agree, n, ident = 0.0, 0, 0, 0
        try:
            for (seq, P), lt in zip(refs, tgt_logits):
                lh = trajectory_logits(model, seq, P)
                lt = lt.to(lh.device)
                err = max(err, float((lh - lt).abs().max()))
                toks = seq[0, P:].to(lh.device)
                ok = (lh.argmax(-1) == toks)
                agree += int(ok.sum())
                n += ok.numel()
                ident += int(bool(ok.all()))
        finally:
            undo()
        out.append(dict(param=v, max_err=err, top1=agree / n, identical=ident / len(refs)))
        log(f"    {v!r:>14}: max|dlogit| {err:.3e}  top1 {agree / n:.4f}  identical {ident}/{len(refs)}")
    return out


def _host_setter(key):
    def setup(v):
        S.host = {key: v, "eps": S.eps}
        def undo():
            S.host = None
        return undo
    return setup


def opt_silu_setter(model):
    layers = list(_get(model, FAM["opt"]["layers"]))
    orig = [L.activation_fn for L in layers]

    class SiluB(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):
            return F.silu(self.b * x.double()).to(x.dtype) / self.b

    def setup(beta):
        for L in layers:
            L.activation_fn = SiluB(beta)

        def undo():
            for L, a in zip(layers, orig):
                L.activation_fn = a
        return undo
    return setup


def bloom_rope_setter(model):
    tr = model.transformer
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor

    def setup(theta):
        def host_alibi(attention_mask, num_heads, dtype):
            a = build_alibi_tensor(attention_mask, num_heads, torch.float64)       # (B H, 1, T): slope * j
            B = attention_mask.shape[0]
            T = attention_mask.shape[1]
            m = a.reshape(B, num_heads, T)[:, :, 1].reshape(B * num_heads, 1, 1)    # slopes
            pos = torch.arange(T, dtype=torch.float64, device=a.device)
            dd = pos[None, None, :] - pos[None, :, None]                            # j - t
            return ((m / theta) * torch.sin(theta * dd)).to(dtype)                  # -N sin(theta (t - j)), N = m / theta
        tr.build_alibi_tensor = host_alibi

        def undo():
            del tr.build_alibi_tensor
        return undo
    return setup


# --- 6. theorem evaluation -----------------------------------------------------------------------------------------
def _lse(xs):
    xs = [x for x in xs if x != float("-inf")]
    if not xs:
        return float("-inf")
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def log_err_bound(L, log_kappa):
    """log of Lean's errBound(L, kappa) = sum_i kappa_i prod_{j>i} L_j  (E0 = 0, E_{n+1} = L_n E_n + kappa_n)."""
    lL = [math.log(l) for l in L]
    tail = [sum(lL[i + 1:]) for i in range(len(L))]
    return _lse([k + t for k, t in zip(log_kappa, tail)])


def err_bound(L, eps):
    """Lean's errBound, directly (for the test)."""
    E = 0.0
    for l, e in zip(L, eps):
        E = l * E + e
    return E


def log_threshold(Lh, L, log_kappa, delta):
    """log of the largest g with 2 Lh (prod max(L, 1)) (sum kappa) g < delta (the product form of the stated
    theorems) and with 2 Lh errBound(L, kappa g) < delta (Lean's model_error_bound, sharper). Logs: the products
    overflow floats."""
    lp = math.log(Lh) + sum(math.log(max(l, 1.0)) for l in L) + _lse(log_kappa)
    ls = math.log(Lh) + log_err_bound(L, log_kappa)
    ld = math.log(delta) - math.log(2) if delta > 0 else float("-inf")
    return dict(logg_prod=ld - lp, logg_sharp=ld - ls, log_bound_per_g_prod=lp, log_bound_per_g_sharp=ls)


def a_needed(ms_max, eps, rho=2.0 ** -30):
    return max(0, math.ceil((math.log2(ms_max / eps) - math.log2(rho)) / 2))


LN2 = math.log(2)
# each rule's parameter <-> log g (eps_i = kappa_i g)
TO_PARAM = {"relu_silu": lambda lg, r: math.exp(-lg), "act_swap": lambda lg, r: math.exp(lg),
            "alibi_rope": lambda lg, r: math.exp(lg / 2), "softcap_host": lambda lg, r: math.exp(-lg / 2),
            "mask": lambda lg, r: -lg, "sink": lambda lg, r: lg,
            "soft_router": lambda lg, r: -lg / r["router_margin_min"] if r["router_margin_min"] > 0 else float("inf"),
            "lever_qk": lambda lg, r: -lg / (2 * LN2), "lever_post": lambda lg, r: -lg / (2 * LN2)}
LOG_G = {"relu_silu": lambda b, r: -math.log(b), "act_swap": lambda e, r: math.log(e),
         "alibi_rope": lambda th, r: 2 * math.log(th), "softcap_host": lambda c, r: -2 * math.log(c),
         "mask": lambda B, r: -B, "sink": lambda b, r: b,
         "soft_router": lambda b, r: -b * r["router_margin_min"],
         "lever_qk": lambda a, r: -2 * a * LN2, "lever_post": lambda a, r: -2 * a * LN2}
SMALLER_IS_SAFER = ("alibi_rope", "sink", "act_swap")


def _ln(x):
    return math.log(x) if x > 0 else float("-inf")


def evaluate(res):
    """every applicable rule: log kappa_i, the threshold (product and sharp), the construction's setting, satisfied?"""
    fam = res["family"]
    lay = res["layers"]
    L = [l["jac"]["block_mix"]["max"] for l in lay]
    Lf = [max(l["jac"]["ffn"]["max"], 1.0) if not res["parallel"] else 1.0 for l in lay]
    Lh = res["Lh"]["max"]
    dmin = res["margins"]["min_over_prompts"]
    dmed = res["margins"]["median_prompt_min"]
    T = res["T_max"]
    rules = {}

    def add(name, log_kappa, setting, extra=None):
        r = dict(extra or {})
        th = log_threshold(Lh, L, log_kappa, dmin)
        thm = log_threshold(Lh, L, log_kappa, dmed)
        r.update(log10_kappa_sum=_lse(log_kappa) / math.log(10),
                 log10_prod_L=sum(math.log10(max(l, 1.0)) for l in L), Lh=Lh, delta_min=dmin,
                 delta_median_prompt=dmed, log_bound_per_g_prod=th["log_bound_per_g_prod"],
                 log_bound_per_g_sharp=th["log_bound_per_g_sharp"],
                 param_star_prod=TO_PARAM[name](th["logg_prod"], r),
                 param_star_sharp=TO_PARAM[name](th["logg_sharp"], r),
                 param_star_prod_median_prompt=TO_PARAM[name](thm["logg_prod"], r), setting=setting)
        rules[name] = r
        return r

    W = res.get("down_col_sum")
    if W is not None and res.get("gated") is False:
        lk = [math.log(T * w) for w in W]
        if fam == "opt":
            add("relu_silu", lk, CONSTRUCTION["relu_silu"]["beta"])
        add("act_swap", lk, None)
    M = [l["rec"]["M"] for l in lay]
    R = [l["rec"]["zmax"] for l in lay]
    if fam == "bloom":
        mbar = res["alibi_slope_max"]
        lk = [_ln(lf * (2 / 3) * mbar * T ** 3 * m) for lf, m in zip(Lf, M)]
        add("alibi_rope", lk, CONSTRUCTION["alibi_rope"]["theta"],
            extra=dict(theta_max_valid=math.sqrt(3 / (mbar * T ** 3)), mbar=mbar))
    if fam != "bloom" and not res.get("has_attn_cap"):
        zmax = max(R)
        lk = [_ln(lf * (4 / 3) * m) + 3 * _ln(r) for lf, r, m in zip(Lf, R, M)]
        add("softcap_host", lk, CONSTRUCTION["softcap_host"]["c_over_zmax"] * zmax,
            extra=dict(c_min_valid=math.sqrt(2 * zmax ** 3 / 3), zmax=zmax))
        lk = [_ln(lf * 2 * m * T) + 2 * r for lf, r, m in zip(Lf, R, M)]
        add("mask", lk, CONSTRUCTION["mask"]["B"], extra=dict(zmax=zmax))
        lk = [_ln(lf * 2 * m) + r for lf, r, m in zip(Lf, R, M)]
        add("sink", lk, CONSTRUCTION["sink"]["b"], extra=dict(zmax=zmax))
    if fam == "qwen3_moe":
        dr = min(l["rec"]["router_margin_min"] for l in lay)
        E = res["n_experts"]
        lk = [_ln(2 * l["rec"]["M_exp"] * (E - 1)) for l in lay]
        add("soft_router", lk, CONSTRUCTION["soft_router"]["beta"], extra=dict(router_margin_min=dr))
    if fam in LLAMA_LIKE and fam != "olmo2":
        eps = res["eps"]
        hd = res["head_dim"]
        CB = math.sqrt(2) * hd * hd ** -0.5      # |q^T R k| / sqrt(hd) <= sqrt(2) hd ||q||_inf ||k||_inf / sqrt(hd)
        lk, need = [], 0.0
        for lf, l in zip(Lf, lay):
            r = l["rec"]
            Q, K, Vb = r["q_inf"], r["k_inf"], r["v_inf"]
            per_head = [o * Vb * (2 * CB * Q * K * (Q * Q + K * K) + Vb * Vb / 2) / eps for o in r["O_h"]]
            lk.append(_ln(lf * sum(per_head)))
            need = max(need, CB * Q * K * (Q * Q + K * K) / eps)
        msq = max(max(l["rec"]["q_ms"], l["rec"]["k_ms"]) for l in lay)
        add("lever_qk", lk, a_needed(msq, eps),
            extra=dict(a_min_region=0.5 * math.log2(need), ms_over_eps=msq / eps, C_B=CB))
        D = res["d_model"]
        lk = [_ln(math.sqrt(D) * (lf * l["rec"]["ya_inf"] ** 3 + l["rec"]["yf_inf"] ** 3) / (2 * eps))
              for lf, l in zip(Lf, lay)]
        msy = max(max(l["rec"]["ya_ms"], l["rec"]["yf_ms"]) for l in lay)
        add("lever_post", lk, a_needed(msy, eps), extra=dict(ms_over_eps=msy / eps))
    for name, r in rules.items():
        s = r["setting"]
        if s is None:
            r["satisfied"] = None
        elif name == "alibi_rope":
            r["satisfied"] = bool(max(s) < r["param_star_prod"])
            r["satisfied_best_setting"] = bool(min(s) < r["param_star_prod"])
        elif name in SMALLER_IS_SAFER:
            r["satisfied"] = bool(s < r["param_star_prod"])
        else:
            r["satisfied"] = bool(s > r["param_star_prod"])
    return rules


# --- 7. the run ------------------------------------------------------------------------------------------------------
def run(key, device="cuda:0", n_prompts=24, jac_prompts=8, stride=4, chunk=16, n_l2=20, n_mix=10, sweeps=True,
        out_dir=OUT, log=None):
    log = log or (lambda *a: print(*a, file=sys.stderr, flush=True))
    t0 = time.time()
    tok, model, fam, on_gpu = load(key, device)
    log(f"{key}: {fam}, on {'GPU' if on_gpu else 'CPU (layers offloaded)'}")
    refs = references(model, tok, PROMPTS[:n_prompts])
    log(f"  references {time.time() - t0:.0f}s")
    return measure(model, fam, refs, key, device=device, jac_prompts=jac_prompts, stride=stride, chunk=chunk,
                   n_l2=n_l2, n_mix=n_mix, sweeps=sweeps, out_dir=out_dir, log=log, t0=t0)


def measure(model, fam, refs, key, device="cuda:0", jac_prompts=8, stride=4, chunk=16, n_l2=20, n_mix=10,
            sweeps=True, out_dir=OUT, log=None, t0=None):
    """everything after the references: constants, margins, thresholds, host sweeps; writes out_dir/key.json."""
    log = log or (lambda *a: print(*a, file=sys.stderr, flush=True))
    t0 = t0 or time.time()
    dev = device if device.startswith("cuda") else "cpu"
    cfg = model.config
    hooks = Hooks(model, fam)

    S.rec = {i: _new_rec() for i in range(len(hooks.layers))}
    nm, kind, g, eps_f = final_norm_info(model, fam)
    W = model.get_output_embeddings().weight.detach()
    heads, deltas, caps_by_prompt, tgt_logits, mism = [], [], [], [], 0
    for j, (seq, P) in enumerate(refs):
        lg, xin, caps = capture(model, hooks, seq, keep_inputs=j < jac_prompts)
        if caps is not None:
            caps_by_prompt.append({i: _to(c, "cpu") for i, c in caps.items()})
        tl = lg[P - 1:-1]
        toks = seq[0, P:].to(tl.device)
        mg = margins(tl, toks)
        mism += int((mg <= 0).sum())
        deltas.append(mg.cpu())
        tgt_logits.append(tl.cpu())
        heads.append(xin[P - 1:-1].float())
    rec = S.rec
    S.rec = None
    try:
        Wd = W.to(dev).float()
        Lh_tok = _oom_retry(head_constants, Wd, kind, g.to(dev), eps_f, torch.cat(heads).to(dev))
    except torch.OutOfMemoryError:                # a shared GPU without room: the closed form on the CPU
        Wd = None
        torch.cuda.empty_cache()
        Lh_tok = head_constants(W.cpu().float(), kind, g.cpu(), eps_f, torch.cat(heads).cpu())
    del Wd
    torch.cuda.empty_cache()
    dm = torch.stack(deltas)                                   # (prompts, 64)
    per_prompt_min = dm.min(1).values
    log(f"  margins: min {float(per_prompt_min.min()):.3g}, median prompt-min {float(per_prompt_min.median()):.3g}; "
        f"Lh max {float(Lh_tok.max()):.3g}")

    jac = jacobians(model, hooks, caps_by_prompt, dev, stride=stride, chunk=chunk, n_l2=n_l2, n_mix=n_mix, log=log)

    # weights: down-projection column norms (Lean's ||w_u||) for non-gated MLPs
    gated = any(n.endswith("up_proj") for n, _ in hooks.layers[0].named_modules())
    down = FAM[fam]["down"]
    col = None
    if down is not None and fam != "qwen3_moe":
        col = []
        for L in hooks.layers:
            m = _get(L, down)
            Wt = m.weight.detach().float()
            col.append(float((Wt.norm(dim=1) if type(m).__name__ == "Conv1D" else Wt.norm(dim=0)).sum()))
    layers = []
    for i in range(len(hooks.layers)):
        r = dict(rec[i])
        rm = r.pop("router_margin")
        if rm:
            rmv = torch.cat(rm).numpy()
            r["router_margin_frac_below"] = {str(t): float((rmv < t).mean()) for t in (1e-5, 1e-3, 1e-2)}
        if r["router_margin_min"] == float("inf"):
            r["router_margin_min"] = None
        layers.append(dict(i=i, rec=r, jac=jac[i]))
    T_max = max(int(s.shape[1]) for s, _ in refs)
    res = dict(key=key, model=MODELS.get(key, key), family=fam, n_layers=len(hooks.layers),
               d_model=int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0))),
               head_dim=int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
               if hasattr(cfg, "num_attention_heads") else None,
               eps=float(getattr(cfg, "rms_norm_eps", None) or getattr(cfg, "layer_norm_eps", None)
                         or getattr(cfg, "layer_norm_epsilon", None) or 1e-5),
               parallel=bool(getattr(cfg, "use_parallel_residual", False)), gated=gated,
               has_attn_cap=getattr(cfg, "attn_logit_softcapping", None) is not None,
               n_experts=getattr(cfg, "num_experts", None), T_max=T_max, n_prompts=len(refs),
               jac_prompts=len(caps_by_prompt), stride=stride, n_l2=n_l2, n_mix=n_mix,
               margins=dict(min_over_prompts=float(per_prompt_min.min()),
                            median_prompt_min=float(per_prompt_min.median()),
                            per_prompt_min=per_prompt_min.tolist(),
                            all=_pct(dm.flatten().numpy()) | dict(
                                p1=float(np.percentile(dm.numpy(), 1)), p5=float(np.percentile(dm.numpy(), 5)),
                                min=float(dm.min())),
                            frac_below={str(t): float((dm < t).float().mean()) for t in (0.01, 0.1, 0.5, 1.0)},
                            nonpositive=mism),
               Lh=_pct(Lh_tok.numpy()), down_col_sum=col, layers=layers,
               construction={k: v["src"] for k, v in CONSTRUCTION.items()})
    if fam == "bloom":
        from transformers.models.bloom.modeling_bloom import build_alibi_tensor
        a = build_alibi_tensor(torch.ones(1, 2), cfg.n_head, torch.float32)
        res["alibi_slope_max"] = float(a[:, 0, 1].abs().max())
    prodL = float(np.prod([max(l["jac"]["block_mix"]["max"], 1) for l in layers]))
    prodS = float(np.prod([max(l["jac"]["block_spec"]["max"], 1) for l in layers]))
    prodSub = float(np.prod([max(l["jac"]["attn_mix"]["max"], 1) * max(l["jac"]["ffn"]["max"], 1) for l in layers]))
    res["products"] = dict(block_mix=prodL, block_spec=prodS, sublayers_mix=prodSub)
    log(f"  prod L_block {prodL:.3g} (spec {prodS:.3g}); prod over sublayers {prodSub:.3g}")

    res["rules"] = evaluate(res)
    S.eps = res["eps"]

    # observed: the host error at a sweep of each rule's parameter
    obs = {}
    if sweeps:
        dm_min = res["margins"]["min_over_prompts"]
        if fam in LLAMA_LIKE:
            z = max(l["rec"]["zmax"] for l in layers)
            log("  softcap host (c):")
            obs["softcap_host"] = observed(model, refs, tgt_logits, _host_setter("cap"),
                                           [round(m * z, 3) for m in (1, 2, 5, 10, 100, 1000)], log)
            log("  mask (-B outside the causal mask):")
            obs["mask"] = observed(model, refs, tgt_logits, _host_setter("mask_B"), [5.0, 10.0, 20.0, 40.0, 80.0], log)
            log("  sink (b):")
            obs["sink"] = observed(model, refs, tgt_logits, _host_setter("sink_b"), [0.0, -5.0, -10.0, -20.0, -40.0],
                                   log)
            if fam != "olmo2":
                log("  qk lever (S = 2^a):")
                obs["lever_qk"] = observed(model, refs, tgt_logits, lambda a: _host_setter("qk_S")(2.0 ** a),
                                           [0, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, res["rules"]["lever_qk"]["setting"]], log)
                log("  post-norm lever (S = 2^a):")
                obs["lever_post"] = observed(model, refs, tgt_logits, lambda a: _host_setter("post_S")(2.0 ** a),
                                             [0, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, res["rules"]["lever_post"]["setting"]], log)
        if fam == "opt":
            log("  ReLU -> silu(beta x)/beta:")
            obs["relu_silu"] = observed(model, refs, tgt_logits, opt_silu_setter(model),
                                        [1.0, 10.0, 100.0, 1e3, 3417.2, 1e4, 1e5, 1e7], log)
        if fam == "bloom":
            log("  ALiBi -> RoPE pair at angle theta:")
            obs["alibi_rope"] = observed(model, refs, tgt_logits, bloom_rope_setter(model),
                                         [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 6.25e-4, 1e-4], log)
        if fam == "qwen3_moe":
            log("  soft router (beta):")
            obs["soft_router"] = observed(model, refs, tgt_logits, _host_setter("router_beta"),
                                          [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e7], log)
        for name, rows in obs.items():
            r = res["rules"].get(name)
            for row in rows:
                if r is None:
                    continue
                lg = LOG_G[name](row["param"], r)
                for form in ("prod", "sharp"):
                    lb = r[f"log_bound_per_g_{form}"] + lg
                    row[f"log10_bound_{form}"] = lb / math.log(10)
                    row[f"log10_looseness_{form}"] = ((lb - math.log(row["max_err"])) / math.log(10)
                                                      if row["max_err"] > 0 else None)
                row["observation_certifies"] = bool(2 * row["max_err"] < dm_min)
    res["observed"] = obs
    res["seconds"] = time.time() - t0
    hooks.remove()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{key}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"{key}: done in {res['seconds']:.0f}s")
    return res


# --- 8. the report (the tables of LIPSCHITZ.md) -----------------------------------------------------------------------
def _fmt(x, p=3):
    if x is None:
        return "-"
    if isinstance(x, (tuple, list)):
        return " ... ".join(_fmt(v, p) for v in x)
    if isinstance(x, bool):
        return "yes" if x else "no"
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    if abs(x) >= 1e5 or (0 < abs(x) < 1e-3):
        return f"{x:.{p - 1}e}"
    return f"{x:.{p}g}"


PARAM = {"relu_silu": ("beta >", "beta"), "act_swap": ("eta <", "eta"), "alibi_rope": ("theta <", "theta"),
         "softcap_host": ("c >", "c"), "mask": ("B >", "B"), "sink": ("b <", "b"), "soft_router": ("beta >", "beta"),
         "lever_qk": ("a >", "a (S = 2^a)"), "lever_post": ("a >", "a (S = 2^a)")}


def _empirical(name, rows, dmin):
    """the loosest setting in the sweep that already gives identical text on every prompt, and the max error at the
    construction's setting (or the closest sweep point)."""
    if not rows:
        return None, None
    good = [r for r in rows if r["identical"] == 1.0]
    order = max if name in SMALLER_IS_SAFER else min
    best = order(good, key=lambda r: r["param"]) if good else None
    if best is not None and best["max_err"] == 0:          # prefer a point where fp32 still sees the error
        nz = [r for r in good if r["max_err"] > 0]
        best = order(nz, key=lambda r: r["param"]) if nz else best
    return best, rows


def report(out_dir=OUT):
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json") and f != "summary.json")
    R = [json.load(open(os.path.join(out_dir, f))) for f in files]
    order = list(MODELS)
    R.sort(key=lambda r: order.index(r["key"]) if r["key"] in order else 99)
    lines = []
    lines.append("### Measured constants (local: Jacobians at real activations)\n")
    lines.append("| model | n | Lh max | L_block max (mix / spec) | prod L_block | prod sublayers | L_ffn max | "
                 "L_attn max (mix) | max \\|pre\\| | max \\|z\\| | M max | delta min | delta median-prompt min | "
                 "delta p1 | steps < 0.1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        lay = r["layers"]
        lines.append("| {} | {} | {} | {} / {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["key"], r["n_layers"], _fmt(r["Lh"]["max"]),
            _fmt(max(l["jac"]["block_mix"]["max"] for l in lay)), _fmt(max(l["jac"]["block_spec"]["max"] for l in lay)),
            _fmt(r["products"]["block_mix"]), _fmt(r["products"]["sublayers_mix"]),
            _fmt(max(l["jac"]["ffn"]["max"] for l in lay)), _fmt(max(l["jac"]["attn_mix"]["max"] for l in lay)),
            _fmt(max(l["rec"]["pre_max"] for l in lay)), _fmt(max(l["rec"]["zmax"] for l in lay)),
            _fmt(max(l["rec"]["M"] for l in lay)), _fmt(r["margins"]["min_over_prompts"]),
            _fmt(r["margins"]["median_prompt_min"]), _fmt(r["margins"]["all"]["p1"]),
            f"{100 * r['margins']['frac_below']['0.1']:.1f}%"))
    lines.append("")
    lines.append("### Limit rule -> theorem threshold -> construction setting\n")
    lines.append("| limit rule | model | theorem threshold (prod form, worst prompt) | sharp form (errBound) | "
                 "median prompt | construction setting | satisfied? | loosest swept setting with identical text | "
                 "observed max \\|dlogit\\| there | bound / observed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        dmin = r["margins"]["min_over_prompts"]
        for name, ru in r["rules"].items():
            if name == "act_swap":
                continue
            op, pn = PARAM[name]
            best, rows = _empirical(name, r.get("observed", {}).get(name), dmin)
            emp = f"{pn} = {_fmt(best['param'])}" if best else ("-" if rows is None else "none in sweep")
            errs = _fmt(best["max_err"]) if best else "-"
            if best and best.get("log10_looseness_prod") is not None:
                loose = f"10^{best['log10_looseness_prod']:.0f} (sharp 10^{best['log10_looseness_sharp']:.0f})"
            else:
                loose = "-" if not best else "error 0 in fp32"
            lines.append(f"| {name} | {r['key']} | {op} {_fmt(ru['param_star_prod'])} | {op} "
                         f"{_fmt(ru['param_star_sharp'])} | {op} {_fmt(ru['param_star_prod_median_prompt'])} | "
                         f"{_fmt(ru['setting'])} | {_fmt(ru['satisfied'])} | {emp} | {errs} | {loose} |")
    lines.append("")
    txt = "\n".join(lines)
    summ = {r["key"]: dict(rules=r["rules"], products=r["products"], Lh=r["Lh"], margins={
        k: r["margins"][k] for k in ("min_over_prompts", "median_prompt_min", "frac_below", "nonpositive")})
        for r in R}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--jac-prompts", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    keys = list(MODELS) if a.models == ["all"] else a.models
    for k in keys:
        run(k, device=a.device, n_prompts=a.n_prompts, jac_prompts=a.jac_prompts, stride=a.stride, chunk=a.chunk,
            sweeps=not a.no_sweeps)
    if a.report:
        print(report())


if __name__ == "__main__":
    main()
