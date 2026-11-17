"""Export a UniversalElephantine (universal.py) in the `tef-ucore` run format: a config.json of per-layer flags
and scalars plus the tensors those flags read (torch state_dict names, fp32). Each framework's
recipes/frameworks/<fw>/universal.py runs it.

The export is EXACT or it refuses. A sub-block (attention / SSM / routed experts / shared FFN) is dropped only
when its output projection's weight and bias are exactly zero, so its output is exactly zero. Every branch
outside the core set (CORE_BRANCHES) must be at its off value, or export() raises Unsupported and names the
branch. A ucore checkpoint therefore computes what the torch module computes.
"""
from __future__ import annotations

import math

import numpy as np
import torch

FORMAT, VERSION = "tef-ucore", 1

CORE_BRANCHES = """
model      token embedding, learned absolute positions, embedding norm, final norm, LM head (+ bias), final softcap
norm       RMSNorm / LayerNorm (centre, bias), eps, identity
layer      sequential / parallel blocks (seq, par), post-sublayer norms (post_coef), residual coefficients,
           output scale, stream norms (sn_in / sn_out), a split FFN (Gemma-4), per-layer embeddings (PLE)
attention  q/k/v/o (+ biases), normalized latent features (MLA), RoPE (any inv_freq, 0 = NoPE pair), scale,
           qk-norm (per head / whole projection, RMS / LayerNorm, before / after the rotary, narrow width),
           v-norm, q/k/v clip, query temperature, logit softcap, ALiBi, sliding window + sink positions,
           learned sinks (null logit), sigmoid / softplus output gate, RMS-normed output, the cross-layer KV
           register (a layer reads the K / V of an earlier layer of its type)
SSM        in_proj, causal conv, silu / linear act, normalized + sigmoid knob features, q/k l2 norms, the
           gated-delta recurrence with the decay / delta / write knobs (A1-A3, mem, cg, cg0, cw[0..5], RG-LRU
           reset), dt floor, D skip, channel mix, output gate (silu / sigmoid / gelu-tanh / identity / const),
           grouped RMS norm (norm-gate / gate-norm / plain)
FFN        routed top-k MoE (softmax / sigmoid scores, selection bias, selection on logits, router softcap,
           group-limited top-k, renorm (+ eps), route scale, per-expert scale, expert biases) and a shared /
           dense path (+ biases, sigmoid gate); activations silu (+ clip, up-shift), relu, relu^2, gelu / geglu
           (erf or tanh), mish, xIELU
""".strip()


class Unsupported(Exception):
    pass


def _f(t):
    return float(t.detach().reshape(-1)[0]) if torch.is_tensor(t) else float(t)


def _nz(t):
    return bool(torch.is_tensor(t) and t.numel() and bool((t != 0).any()))


def _fin(t):
    v = _f(t)
    return None if math.isinf(v) else v


class _Out:
    def __init__(self):
        self.tensors = {}

    def put(self, name, t):
        self.tensors[name] = t.detach().to(torch.float64).cpu().numpy().astype(np.float32)

    def lin(self, name, lin, bias=True):
        self.put(name + ".weight", lin.weight)
        has = bias and getattr(lin, "bias", None) is not None and _nz(lin.bias)
        if has:
            self.put(name + ".bias", lin.bias)
        return bool(has)


def _refuse(where, what):
    raise Unsupported(f"{where}: {what} is not in the tef-ucore core set")


def _check_off(where, cond, what):
    if cond:
        _refuse(where, what)


def _norm(o: _Out, name, n):
    _check_off(name, _f(n.gate_c) != 0, "a gated norm (gate_c)")
    _check_off(name, _f(n.pre_c) != 0, "a two-stage norm (pre_c)")
    if _f(n.ident) != 0:
        return {"ident": True}
    o.put(name + ".weight", n.weight)
    bias = _nz(n.bias)
    if bias:
        o.put(name + ".bias", n.bias)
    return {"ident": False, "eps": _f(n.eps), "centre": _f(n.centre), "bias": bias}


def _live(lin):
    return _nz(lin.weight) or (getattr(lin, "bias", None) is not None and _nz(lin.bias))


def _attn(o: _Out, p, a, c):
    w = p.rstrip(".")
    _check_off(w, _f(a.n_gate_c) != 0, "a gated latent norm")
    _check_off(w, any(_nz(t) for t in list(a.la.values()) + list(a.lb.values())), "switched LoRA")
    _check_off(w, _f(a.kv_c) != 0, "the memory key/value writer")
    _check_off(w, not math.isinf(_f(a.sel_cap)), "the score-selected memory mask")
    _check_off(w, a.dsa_on and (not math.isinf(_f(a.dsa_topk)) or _f(a.dsa_share) != 0), "the DSA indexer")
    _check_off(w, _nz(a.hg_c), "the per-head read gate")
    _check_off(w, _f(a.dl_c) != 0, "differential heads")
    _check_off(w, not torch.equal(a.o_hmix, torch.eye(a.H, dtype=a.o_hmix.dtype)), "the output head mix")
    s = {"Dn": a.Dn}
    if a.Dn:
        o.put(p + "pn.weight", a.pn.weight)
        o.put(p + "n_avg", a.n_avg)
        o.put(p + "n_w", a.n_w)
        s["n_eps"] = _f(a.n_eps)
    s["bias"] = {n: o.lin(p + n + "_proj", getattr(a, n + "_proj")) for n in ("q", "k", "v", "o")}
    s["v_norm"] = _f(a.v_c) != 0
    if s["v_norm"]:
        o.put(p + "v_nw", a.v_nw)
        s["v_eps"] = _f(a.v_eps)
    s["q_coef"], s["k_coef"] = a.q_coef.tolist(), a.k_coef.tolist()
    s["qk_full"], s["qk_centre"], s["qk_post"] = _f(a.qk_full) != 0, [bool(v != 0) for v in a.qk_centre], _f(a.qk_post) != 0
    s["qk_eps"], s["qk_width"] = a.qk_eps.tolist(), _f(a.qk_width)
    if s["q_coef"][1] != 0 or s["k_coef"][1] != 0:
        for n in ("q_nw", "k_nw", "q_nb", "k_nb"):
            o.put(p + n, getattr(a, n))
    s["clip"] = _fin(a.qkv_clip)
    s["rope"] = _nz(a.inv_freq)
    if s["rope"]:
        o.put(p + "inv_freq", a.inv_freq)
    s["scale"] = _f(a.scale)
    s["cap"] = _fin(a.cap)
    s["alibi"] = _nz(a.alibi)
    if s["alibi"]:
        o.put(p + "alibi", a.alibi)
    s["window"] = _fin(a.window)
    s["n_sink"] = _f(a.n_sink)
    nl = a.null_logit
    s["null"] = bool(torch.isfinite(nl).any())
    if s["null"]:
        o.put(p + "null_logit", torch.where(torch.isfinite(nl), nl, torch.full_like(nl, -1e30)))
        s["null_off"] = [not bool(v) for v in torch.isfinite(nl)]
    s["o_coef"] = a.o_coef.tolist()
    if s["o_coef"][1] != 0:
        o.put(p + "o_avg", a.o_avg)
        o.put(p + "o_w", a.o_w)
        s["o_eps"] = _f(a.o_eps)
    s["gate_coef"] = a.gate_coef.tolist()
    s["gate_sp"] = _f(a.gate_sp)
    if s["gate_coef"][1] != 0 or s["gate_sp"] != 0:
        o.lin(p + "g_proj", a.g_proj)
    s["qtemp"] = a.qtemp.tolist() if _f(a.qtemp[0]) != 0 else None   # q *= 1 + beta log1p(floor(t / M))
    return s


def _ssm(o: _Out, p, m, c):
    w = p.rstrip(".")
    _check_off(w, m.Dp or m.Dh2, "product / second-stage sigmoid knob features (RWKV-6)")
    _check_off(w, _nz(m.A4), "the Mamba-3 decay term")
    _check_off(w, _nz(m.A5), "the RWKV-6 decay term")
    _check_off(w, _nz(m.c_rm) or _nz(m.c_ka), "the RWKV-7 generalized removal")
    _check_off(w, _nz(m.c_er), "the read-key erase gate")
    _check_off(w, _nz(m.cw_prev), "the previous-token write")
    _check_off(w, _nz(m.rot_c) or _nz(m.rot_k), "the cumulative q/k rotation")
    _check_off(w, _nz(m.ridge_c), "the ridge readout")
    _check_off(w, _nz(m.c_pre), "the pre-update read")
    _check_off(w, m.dcW > 0, "the dynamic conv")
    _check_off(w, any(_f(t) != 0 for t in (m.v_write, m.v_read, m.y_write, m.y_read)), "the value / output registers")
    _check_off(w, _nz(m.out_coef[3]), "the ratio output branch")
    s = {"D": m.D, "K": m.K, "H": m.H, "dk": m.dk, "dv": m.dv, "Dn": m.Dn, "Dh": m.Dh,
         "l2_eps": _f(m.l2_eps), "l2_eps_out": _f(m.l2_eps_out), "dt_min": _f(m.dt_min), "norm_eps": _f(m.norm_eps),
         "q_coef": m.q_coef.tolist(), "k_coef": m.k_coef.tolist()}
    for n in ("in_proj", "pq", "pk", "pv", "pz", "p1", "p2", "pb", "pe", "pi", "out_proj"):
        o.lin(p + n, getattr(m, n))
    if m.Dn:
        o.lin(p + "pn", m.pn)
        o.put(p + "n_avg", m.n_avg)
        o.put(p + "n_w", m.n_w)
        s["n_eps"] = _f(m.n_eps)
    if m.Dh:
        o.lin(p + "ph", m.ph)
    for n in ("conv_w", "conv_b", "act_silu", "act_lin", "A1", "A2", "A3", "mem", "cg", "cg0", "cw", "reset",
              "Dskip", "mix", "gate", "out_coef", "avg", "norm_w", "norm_b"):
        o.put(p + n, getattr(m, n))
    return s


def _act(f):
    _check_off("ffn", _f(f.c_situ) != 0, "SiTU-GLU")
    _check_off("ffn", _f(f.c_poly) != 0, "PolyNorm-GLU")
    lim = f.clip_lim.tolist()
    xi = None
    if _f(f.c_xielu) != 0:                               # xIELU: [a_p, a_n, beta, eps] (softplus taken here)
        sp = torch.nn.functional.softplus
        beta, eps = f.xielu_be.tolist()
        xi = [float(sp(f.xielu_a[0])), beta + float(sp(f.xielu_a[1])), beta, eps]
    return {"xielu": xi, "relu2": _f(f.c_relu2) != 0, "relu": _f(f.c_relu) != 0, "gelu": _f(f.c_gelu) != 0,
            "geglu": _f(f.c_geglu) != 0, "gelu_tanh": _f(f.gelu_tanh) != 0, "mish": _f(f.c_mish) != 0,
            "c_clip": _f(f.c_clip), "clip": [None if math.isinf(v) else v for v in lim], "up_shift": _f(f.up_shift)}


def _ffn(o: _Out, p, f, c):
    w = p.rstrip(".")
    _check_off(w, _f(f.smix_eps) > 0, "SparseMixer routing")
    _check_off(w, _f(f.r_c) != 0, "the post-routing norm")
    _check_off(w, any(_nz(t) for t in list(f.la.values()) + list(f.lb.values())), "switched LoRA")
    s = {"act": _act(f), "routed": None, "shared": None}
    if _nz(f.w_down) or _nz(f.b_down):
        r = {"E": f.w_gate.shape[0], "top_k": f.top_k, "score_c": _f(f.score_c), "sel_logit": _f(f.sel_logit) != 0,
             "router_cap": _fin(f.router_cap), "grp": f.grp.tolist(), "renorm": _f(f.renorm),
             "renorm_eps": _f(f.renorm_eps), "route_scale": _f(f.route_scale)}
        r["router_bias"] = o.lin(p + "router", f.router)
        for n in ("sel_bias", "exp_scale", "w_gate", "w_up", "w_down", "b_gate", "b_up", "b_down"):
            o.put(p + n, getattr(f, n))
        s["routed"] = r
    if _live(f.s_down):
        sh = {"s_coef": f.s_coef.tolist()}
        ungated = s["act"]["relu2"] or s["act"]["relu"] or s["act"]["gelu"]
        if not ungated:
            sh["gate_bias"] = o.lin(p + "s_gate", f.s_gate)
        sh["up_bias"] = o.lin(p + "s_up", f.s_up)
        sh["down_bias"] = o.lin(p + "s_down", f.s_down)
        if sh["s_coef"][1] != 0:
            o.put(p + "s_router.weight", f.s_router.weight)
        s["shared"] = sh
    return s


def export(host):
    """(config dict, {name: fp32 numpy}) of a UniversalElephantine, or Unsupported."""
    c = host.cfg
    for k, why in (("n_prefix", "learned prefix tokens"), ("lora_adapters", "switched LoRA adapters"),
                   ("attnres_block", "Block AttnRes"),
                   ("hc_mult", "hyper-connections"), ("mem_attn", "a memory attention")):
        if getattr(c, k):
            _refuse("model", why)
    if c.attnres_full:
        _refuse("model", "Full AttnRes")
    o = _Out()
    cfg = {"format": FORMAT, "version": VERSION, "vocab": c.vocab, "d": c.d, "L": c.L, "heads": c.heads,
           "head_dim": c.head_dim, "max_pos": c.max_pos}
    o.put("embed.weight", host.embed.weight)
    if c.max_pos:
        o.put("pos", host.pos)
    cfg["emb_norm"] = _norm(o, "emb_norm", host.emb_norm) if _f(host.emb_norm_c) != 0 else None
    cfg["norm"] = _norm(o, "norm", host.norm)
    cfg["lm_head_bias"] = o.lin("lm_head", host.lm_head)
    cfg["final_cap"] = _fin(host.final_cap)
    cfg["ple"] = None
    if c.ple_dim:                                        # per-layer embeddings: pli = rms_l(P x0) w + E[ids]
        o.put("ple_E", host.ple_E)
        o.put("ple_P.weight", host.ple_P.weight)
        o.put("ple_w", host.ple_w)
        cfg["ple"] = {"dim": c.ple_dim, "eps": _f(host.ple_eps)}
    layers = []
    for i, ul in enumerate(host.layers):
        p = f"layers.{i}."
        w = f"layer {i}"
        # registers carry state between layers even from a sub-block whose own output is zero
        kr, kw = _f(ul.attn.kv_read), _f(ul.attn.kv_write)
        _check_off(w, kr not in (0.0, 1.0) or (kr and kw), "a mixed KV register read")
        _check_off(w, _f(ul.attn.dsa_share) != 0, "a shared DSA mask")
        _check_off(w, any(_f(t) != 0 for t in (ul.ssm.v_write, ul.ssm.v_read, ul.ssm.y_write, ul.ssm.y_read)),
                   "the SSM value / output registers")
        _check_off(w, ul.engram is not None and _f(ul.engram.c) != 0, "the n-gram memory (Engram)")
        _check_off(w, _f(ul.res_write) != 0 or _f(ul.res_mix) != 0, "the residual snapshot")
        L = {"seq": _f(ul.seq), "par": _f(ul.par), "res_c": ul.res_c.tolist(), "out_scale": _f(ul.out_scale),
             "post_coef": ul.post_coef.tolist()}
        L["sn_in"] = _norm(o, p + "sn_in", ul.sn_in) if ul.sn_in is not None and _f(ul.sn_c[0]) != 0 else None
        L["sn_out"] = _norm(o, p + "sn_out", ul.sn_out) if ul.sn_out is not None and _f(ul.sn_c[1]) != 0 else None
        L["attn"] = _attn(o, p + "attn.", ul.attn, c) if (_live(ul.attn.o_proj) or kw) else None
        if L["attn"] is not None:                          # the KV register (one slot per layer type)
            L["attn"].update(kv_read=bool(kr), kv_write=bool(kw), kv_slot=int(_f(ul.attn.kv_slot)))
        _check_off(w, kr and L["attn"] is None, "a KV-register read in a dropped attention")
        L["ssm"] = _ssm(o, p + "ssm.", ul.ssm, c) if _live(ul.ssm.out_proj) else None
        L["ffn"] = _ffn(o, p + "ffn.", ul.ffn, c)
        if L["ffn"]["routed"] is None and L["ffn"]["shared"] is None:
            L["ffn"] = None
        L["ffn_split"] = _f(ul.ffn_split) != 0 and L["ffn"] is not None   # Gemma-4: own norms per path
        if L["ffn_split"]:
            for nm in ("norm_f2", "norm_r", "post_f1", "post_f2"):
                L[nm] = _norm(o, p + nm, getattr(ul, nm))
        L["ple"] = None
        if _f(ul.ple_c) != 0:
            _check_off(w, not c.ple_dim, "a per-layer embedding input without the table")
            o.put(p + "ple_gate.weight", ul.ple_gate.weight)
            o.put(p + "ple_proj.weight", ul.ple_proj.weight)
            L["ple"] = {"c": _f(ul.ple_c), "tanh": _f(ul.ple_tanh) != 0,
                        "post": _norm(o, p + "ple_post", ul.ple_post)}
        for key, nm, post, j in (("attn", "norm_a", "post_a", 0), ("ssm", "norm_s", "post_s", 1),
                                 ("ffn", "norm_f", "post_f", 2)):
            if L[key] is not None:
                L[nm] = _norm(o, p + nm, getattr(ul, nm))
                pc = L["post_coef"][j]
                L[post] = _norm(o, p + post, getattr(ul, post)) if pc[1] != 0 else None
        layers.append(L)
    cfg["layers"] = layers
    cfg["n_params"] = int(sum(v.size for v in o.tensors.values()))
    return cfg, o.tensors


def save(out_dir, host, extra=None):
    """Write out_dir/config.json + model.safetensors; returns the config."""
    import json
    import os
    from safetensors.numpy import save_file
    cfg, tensors = export(host)
    if extra:
        cfg.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    return cfg
