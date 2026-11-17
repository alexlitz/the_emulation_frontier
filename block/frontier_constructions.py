"""Frontier study constructions (dist/frontier_study.md section 3): host settings that pay walls the round-3 relation
keeps, each proved on the reference decoder in fp64 and measured on real checkpoints in fp32.

  softplus_gate_off   an output-gate host whose gate is softplus(W_g x [+ b]) (Laguna, K2-Horizon) plays an ungated
                      target: W_g = 0, b = 0 gives the constant softplus(0) = ln 2 on every channel, folded into o_proj
                      (o_proj / ln 2). Exact. The same holds per head (out_gate_head_softplus).
  bias_silu_gate_off  a silu gate WITH a bias: W_g = 0, b = 1 gives silu(1) > 0, folded. Exact. (Without a bias silu(0)
                      = 0 and no setting gives a nonzero constant: the relation's wall stands.)
  head_sub_elem       an elementwise gate host plays a head-wise gate target of the same activation: the host's gate
                      rows for every channel of head h are the target's row h (and bias). Exact
                      (FEAT_NEST_PENDING out_gate_head_sub_elem).
  zero_bias           a gate host with a bias plays the bias-free gate target: b = 0. Exact (out_gate_zero_bias).
  tied_host_untied    a TIED host plays an UNTIED target (PROOFS.md lists it OPEN: "under a pre-norm the carry blocks
                      it"). The carry is avoided, not carried: the host's table row for token v is
                      [e_v, u_v, p_v, c] (target embedding, target lm_head row, a pad making every row's norm R, a
                      constant c). At the first layer the residual is the row itself, so its pre-norm divides by the
                      SAME constant for every token. One extra first layer (attention off: o_proj = 0) has a GLU FFN whose
                      gate rows read only the constant channel (a constant gate silu(1)) and whose up rows are the linear
                      map [e, u, p, c] -> [-e, e - u, -p, -c]: after it the residual is [0, e_v, 0, 0]. The target's
                      layers run in the second block (norm gains x sqrt(d / d_h), eps x d / d_h: the zero-pad rule), the
                      final norm's output is [0, y, 0, 0], and the tied head gives row_v . [0, y, 0, 0] = u_v . y.
                      Exact. Charges: hidden >= 2 d + 2, one extra layer, that layer's FFN >= 2 d + 2 units, a gated (or
                      MoE: zero router with tied experts / a shared expert) host, no output-side norm. Necessary width:
                      the host table must hold E and U, rank 2 d for generic weights, so 2 d is a lower bound.

Run:  python -m block.frontier_constructions          # fp64 reference checks + the real-checkpoint measurements
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from verify.spot.catalog import compare
from verify.spot.catalog._ref import F64, RefLM
from verify.spot.catalog.ref_slots import _check, _host, _t

LN2 = math.log(2.0)
DEVICE = __import__("os").environ.get("FS_DEVICE", "cpu")      # real runs: "cuda" (with CUDA_VISIBLE_DEVICES=1) or cpu


def hf_fn(model):
    """ids -> logits computed on DEVICE, returned on the CPU."""
    model.to(DEVICE)
    return lambda x: model(input_ids=x.to(DEVICE)).logits.cpu()


# ------------------------------------------------------------------ reference decoder (fp64)

def _gate_host(t, **cfg):
    """The target's weights in a host with the given output gate, its gate weights random (then written)."""
    h = _host(t, **cfg)
    tmp = RefLM(**h.c).init(5)
    for k, v in tmp.p.items():
        if k not in h.p:
            h.p[k] = v
    return h


def ref_softplus_gate_off(head=False, bias=False):
    t = _t()
    h = _gate_host(t, out_gate="head" if head else True, og_act="softplus", og_bias=bias)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.ogate"] = torch.zeros_like(h.p[f"l{i}.ogate"])
        if bias:
            h.p[f"l{i}.ogate_b"] = torch.zeros_like(h.p[f"l{i}.ogate_b"])
        h.p[f"l{i}.o"] = t.p[f"l{i}.o"] / LN2
    return _check(t, h)


def ref_bias_silu_gate_off(head=False):
    t = _t()
    h = _gate_host(t, out_gate="head" if head else True, og_act="silu", og_bias=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.ogate"] = torch.zeros_like(h.p[f"l{i}.ogate"])
        h.p[f"l{i}.ogate_b"] = torch.ones_like(h.p[f"l{i}.ogate_b"])
        h.p[f"l{i}.o"] = t.p[f"l{i}.o"] / F.silu(torch.tensor(1.0, dtype=F64))
    return _check(t, h)


def ref_head_sub_elem(act="sigmoid", bias=False):
    t = _t(out_gate="head", og_act=act, og_bias=bias)
    h = _host(t, out_gate=True)
    hd = t.c["hd"]
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.ogate"] = t.p[f"l{i}.ogate"].repeat_interleave(hd, 0)
        if bias:
            h.p[f"l{i}.ogate_b"] = t.p[f"l{i}.ogate_b"].repeat_interleave(hd, 0)
    return _check(t, h)


def ref_zero_bias(head=False, act="sigmoid"):
    t = _t(out_gate="head" if head else True, og_act=act)
    h = _host(t, og_bias=True)
    rows = t.c["H"] if head else t.c["H"] * t.c["hd"]
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.ogate_b"] = torch.zeros(rows, dtype=F64)
    return _check(t, h)


def ref_negative_silu_no_bias():
    """Negative control: a bias-free silu gate with W_g = 0 outputs 0 (silu(0) = 0), the attention is lost."""
    t = _t()
    h = _gate_host(t, out_gate=True, og_act="silu")
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.ogate"] = torch.zeros_like(h.p[f"l{i}.ogate"])
    return _check(t, h)


def _tied_rows(E, U, c=1.0, margin=1.0):
    """Host table rows [e_v, u_v, p_v, c] with one norm R for every row."""
    n2 = (E * E).sum(1) + (U * U).sum(1) + c * c
    R2 = n2.max() + margin
    p = torch.sqrt(R2 - n2)
    return torch.cat([E, U, p[:, None], torch.full_like(p[:, None], c)], 1), R2


def ref_tied_host(seed=0, **cfg):
    """An untied RMSNorm / SwiGLU / RoPE target on a tied host of width 2 d + 2 with one extra first layer."""
    t = _t(seed=seed, **cfg)
    c = t.c
    d, L, F_ = c["d"], c["L"], c["F"]
    dh = 2 * d + 2
    h = RefLM(**dict(c, d=dh, L=L + 1, F=max(F_, dh), tied=True, eps=c["eps"] * d / dh)).init(1)
    p = {k: torch.zeros_like(v) for k, v in h.p.items()}
    rows, R2 = _tied_rows(t.p["emb"], t.p["lm"])
    p["emb"] = rows
    p["lm"] = p["emb"]
    s = 1.0 / math.sqrt(R2.item() / dh + h.c["eps"])        # the first pre-norm's constant scale (gain 1)
    blk = slice(d, 2 * d)
    # layer 0: the eraser
    p["l0.n1_w"] = torch.ones(dh, dtype=F64)
    p["l0.n2_w"] = torch.ones(dh, dtype=F64)
    B = torch.zeros(dh, dh, dtype=F64)
    B[:d, :d] = -torch.eye(d, dtype=F64)                     # block 1: -e
    B[d:2 * d, :d] = torch.eye(d, dtype=F64)                 # block 2: e - u
    B[d:2 * d, d:2 * d] = -torch.eye(d, dtype=F64)
    B[2 * d, 2 * d] = -1.0                                   # pad, const: erased
    B[2 * d + 1, 2 * d + 1] = -1.0
    cval = rows[0, 2 * d + 1].item()
    p["l0.fgate"][:dh, 2 * d + 1] = 1.0 / (cval * s)         # gate pre-activation 1 on every token
    p["l0.up"][:dh] = B / s
    p["l0.down"][:, :dh] = torch.eye(dh, dtype=F64) / F.silu(torch.tensor(1.0, dtype=F64))
    # layers 1..L: the target's, in block 2
    sc = math.sqrt(d / dh)
    for i in range(L):
        a, b = f"l{i}.", f"l{i + 1}."
        for n in ("n1_w", "n2_w"):
            p[b + n][blk] = t.p[a + n] * sc
        for k in ("q", "k", "v"):
            p[b + k][:, blk] = t.p[a + k]
        p[b + "o"][blk] = t.p[a + "o"]
        p[b + "fgate"][:F_, blk] = t.p[a + "fgate"]
        p[b + "up"][:F_, blk] = t.p[a + "up"]
        p[b + "down"][blk, :F_] = t.p[a + "down"]
    p["nf_w"][blk] = t.p["nf_w"] * sc
    h.p = p
    return _check(t, h)


def ref_tied_host_negative():
    """Negative control: the same host without the eraser layer (layer 0 a plain zero layer) -- the residual keeps
    u_v, p_v and c, every pre-norm sees them, the logits differ."""
    t = _t()
    r = ref_tied_host()
    del r
    c = t.c
    d = c["d"]
    dh = 2 * d + 2
    h = RefLM(**dict(c, d=dh, L=c["L"], tied=True, eps=c["eps"] * d / dh)).init(1)
    p = {k: torch.zeros_like(v) for k, v in h.p.items()}
    rows, _ = _tied_rows(t.p["emb"], t.p["lm"])
    p["emb"] = rows
    p["lm"] = rows
    blk = slice(d, 2 * d)
    sc = math.sqrt(d / dh)
    for i in range(c["L"]):
        a = f"l{i}."
        for n in ("n1_w", "n2_w"):
            p[a + n][blk] = t.p[a + n] * sc
        for k in ("q", "k", "v"):
            p[a + k][:, blk] = t.p[a + k]
        p[a + "o"][blk] = t.p[a + "o"]
        p[a + "fgate"][:, blk] = t.p[a + "fgate"]
        p[a + "up"][:, blk] = t.p[a + "up"]
        p[a + "down"][blk] = t.p[a + "down"]
    p["nf_w"][blk] = t.p["nf_w"] * sc
    h.p = p
    return _check(t, h)


REF_CHECKS = {
    "softplus_gate_off (elementwise)": lambda: ref_softplus_gate_off(),
    "softplus_gate_off (head-wise)": lambda: ref_softplus_gate_off(head=True),
    "softplus_gate_off (head-wise, bias)": lambda: ref_softplus_gate_off(head=True, bias=True),
    "bias_silu_gate_off (elementwise)": lambda: ref_bias_silu_gate_off(),
    "bias_silu_gate_off (head-wise)": lambda: ref_bias_silu_gate_off(head=True),
    "head_sub_elem (sigmoid)": lambda: ref_head_sub_elem(),
    "head_sub_elem (softplus, bias)": lambda: ref_head_sub_elem("softplus", True),
    "zero_bias (elementwise)": lambda: ref_zero_bias(),
    "zero_bias (head-wise, softplus)": lambda: ref_zero_bias(True, "softplus"),
    "tied_host_untied (d 32, L 2)": lambda: ref_tied_host(),
    "tied_host_untied (d 24, L 3, GQA 6/2)": lambda: ref_tied_host(seed=2, d=24, L=3, H=6, KV=2, hd=4, F=40),
}
REF_NEGATIVE = {
    "silu gate, no bias, W_g = 0": ref_negative_silu_no_bias,
    "tied host without the eraser layer": ref_tied_host_negative,
}


# ------------------------------------------------------------------ real checkpoints (fp32)

def _load(name, dtype=torch.float32):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, local_files_only=True, trust_remote_code=False,
                                             attn_implementation="eager")
    return m.eval()


class GatedO(torch.nn.Module):
    """o_proj(attn * act(W_g x + b)), x the attention's input (stashed by a pre-hook): the host's gated attention.
    rows = heads (one gate per head, broadcast over head_dim) or heads x head_dim (one per channel)."""

    def __init__(self, o_proj, hidden, heads, head_dim, act, per_head, W=None, b=None):
        super().__init__()
        self.o, self.act, self.per_head, self.hd = o_proj, act, per_head, head_dim
        rows = heads if per_head else heads * head_dim
        dt = o_proj.weight.dtype
        self.W = torch.nn.Parameter(W if W is not None else torch.zeros(rows, hidden, dtype=dt), requires_grad=False)
        self.b = torch.nn.Parameter(b if b is not None else torch.zeros(rows, dtype=dt), requires_grad=False)
        self.x = None

    def forward(self, a):
        z = self.x @ self.W.T + self.b
        g = {"sigmoid": torch.sigmoid, "softplus": F.softplus, "silu": F.silu}[self.act](z)
        if self.per_head:
            g = g.repeat_interleave(self.hd, -1)
        return self.o(a * g)


def add_gates(model, act, per_head, W_fn=None, b_fn=None, o_scale=1.0):
    """Wrap every attention's o_proj with a gate; W_fn / b_fn(layer index) give the gate weights (default 0)."""
    cfg = model.config
    H = cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    for i, layer in enumerate(model.model.layers):
        at = layer.self_attn
        g = GatedO(at.o_proj, cfg.hidden_size, H, hd, act, per_head, W_fn(i) if W_fn else None,
                   b_fn(i) if b_fn else None)
        with torch.no_grad():
            at.o_proj.weight.mul_(o_scale)
        at.o_proj = g

        def hook(mod, args, kwargs, g=g):
            g.x = kwargs.get("hidden_states", args[0] if args else None)
        at.register_forward_pre_hook(hook, with_kwargs=True)
    return model


def real_gate(name, kind):
    """Target: the real checkpoint (ungated, or with a random gate for the nest checks); host: the same weights in the
    gated host form the construction writes."""
    t = _load(name)
    h = _load(name)
    cfg = t.config
    H = cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    gen = torch.Generator().manual_seed(0)
    rnd = lambda *s: torch.randn(*s, generator=gen) * 0.05
    if kind == "softplus_elem":
        add_gates(h, "softplus", False, o_scale=1 / LN2)
    elif kind == "softplus_head":
        add_gates(h, "softplus", True, o_scale=1 / LN2)
    elif kind == "bias_silu_head":
        add_gates(h, "silu", True, b_fn=lambda i: torch.ones(H), o_scale=1 / F.silu(torch.tensor(1.0)).item())
    elif kind in ("head_sub_elem", "zero_bias"):
        Ws = [rnd(H, cfg.hidden_size) for _ in t.model.layers]
        if kind == "head_sub_elem":
            add_gates(t, "sigmoid", True, W_fn=lambda i: Ws[i])
            add_gates(h, "sigmoid", False, W_fn=lambda i: Ws[i].repeat_interleave(hd, 0))
        else:
            add_gates(t, "sigmoid", True, W_fn=lambda i: Ws[i])
            add_gates(h, "sigmoid", True, W_fn=lambda i: Ws[i], b_fn=lambda i: torch.zeros(H))
    return compare(hf_fn(t), hf_fn(h), min(cfg.vocab_size, 32000), standard="limit", tol=1e-4,
                   note=f"{name} {kind} fp32")


def tied_host_model(t):
    """The tied host of an untied Llama-family checkpoint t (see tied_host_untied above), as a stock LlamaForCausalLM."""
    from transformers import LlamaConfig, LlamaForCausalLM
    c = t.config
    d, L = c.hidden_size, c.num_hidden_layers
    H, KV = c.num_attention_heads, c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or d // H
    dh = 2 * d + 2
    Fw = max(c.intermediate_size, dh)
    hc = LlamaConfig(vocab_size=c.vocab_size, hidden_size=dh, intermediate_size=Fw, num_hidden_layers=L + 1,
                     num_attention_heads=H, num_key_value_heads=KV, head_dim=hd, rms_norm_eps=c.rms_norm_eps * d / dh,
                     max_position_embeddings=c.max_position_embeddings, rope_theta=getattr(c, "rope_theta", 10000.0),
                     rope_scaling=getattr(c, "rope_scaling", None), tie_word_embeddings=True,
                     attention_bias=False, mlp_bias=False, hidden_act="silu")
    hc._attn_implementation = "eager"
    with torch.device("meta"):
        h = LlamaForCausalLM(hc)
    h = h.to_empty(device="cpu").to(torch.float64).eval()
    for prm in h.parameters():
        prm.data.zero_()
    if hasattr(h.model, "rotary_emb"):
        h.model.rotary_emb = type(t.model.rotary_emb)(config=hc).to(torch.float64)
    E = t.model.embed_tokens.weight.double()
    U = t.lm_head.weight.double()
    rows, R2 = _tied_rows(E, U, c=1.0)
    with torch.no_grad():
        h.model.embed_tokens.weight.copy_(rows)
        h.lm_head.weight = h.model.embed_tokens.weight
        eps_h = hc.rms_norm_eps
        s = 1.0 / math.sqrt(R2.item() / dh + eps_h)
        l0 = h.model.layers[0]
        l0.input_layernorm.weight.fill_(1.0)
        l0.post_attention_layernorm.weight.fill_(1.0)
        B = torch.zeros(dh, dh, dtype=F64)
        B[:d, :d] = -torch.eye(d, dtype=F64)
        B[d:2 * d, :d] = torch.eye(d, dtype=F64)
        B[d:2 * d, d:2 * d] = -torch.eye(d, dtype=F64)
        B[2 * d, 2 * d] = -1.0
        B[2 * d + 1, 2 * d + 1] = -1.0
        l0.mlp.gate_proj.weight[:dh, 2 * d + 1] = 1.0 / (1.0 * s)
        l0.mlp.up_proj.weight[:dh] = B / s
        l0.mlp.down_proj.weight[:, :dh] = torch.eye(dh, dtype=F64) / F.silu(torch.tensor(1.0, dtype=F64))
        sc = math.sqrt(d / dh)
        blk = slice(d, 2 * d)
        for i, lt in enumerate(t.model.layers):
            lh = h.model.layers[i + 1]
            lh.input_layernorm.weight[blk] = lt.input_layernorm.weight.double() * sc
            lh.post_attention_layernorm.weight[blk] = lt.post_attention_layernorm.weight.double() * sc
            for k in ("q_proj", "k_proj", "v_proj"):
                getattr(lh.self_attn, k).weight[:, blk] = getattr(lt.self_attn, k).weight.double()
            lh.self_attn.o_proj.weight[blk] = lt.self_attn.o_proj.weight.double()
            Ft = lt.mlp.gate_proj.weight.shape[0]
            lh.mlp.gate_proj.weight[:Ft, blk] = lt.mlp.gate_proj.weight.double()
            lh.mlp.up_proj.weight[:Ft, blk] = lt.mlp.up_proj.weight.double()
            lh.mlp.down_proj.weight[blk, :Ft] = lt.mlp.down_proj.weight.double()
        h.model.norm.weight[blk] = t.model.norm.weight.double() * sc
    return h


def real_tied(name, dtype=torch.float32):
    t = _load(name)
    assert not t.config.tie_word_embeddings, f"{name} is tied"
    h = tied_host_model(t).to(dtype)
    return compare(hf_fn(t), hf_fn(h), min(t.config.vocab_size, 32000), standard="limit", tol=1e-4,
                   note=f"{name} tied host fp32")


REAL_GATE = [("Qwen/Qwen3-0.6B", k) for k in ("softplus_elem", "softplus_head", "bias_silu_head", "head_sub_elem",
                                               "zero_bias")] + \
            [("HuggingFaceTB/SmolLM2-135M", k) for k in ("softplus_elem", "softplus_head", "bias_silu_head",
                                                          "head_sub_elem", "zero_bias")]
REAL_TIED = ["amd/AMD-Llama-135m", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]


def _fmt(r):
    return (f"max |d logits| {r.max_abs:.2e} (rel {r.rel:.1e}), top-1 {100 * r.top1:.2f} %, identical 32-token "
            f"continuations {r.identical}/{r.n}")


def main():
    import json
    import os
    import sys
    path = "dist/frontier_study/constructions.json"
    out = json.load(open(path)) if os.path.exists(path) else {"ref": {}, "negative": {}, "real": {}}
    save = lambda: json.dump(out, open(path, "w"), indent=1)
    if "--real-only" not in sys.argv:
        for k, f in REF_CHECKS.items():
            out["ref"][k] = _fmt(f())
            print("ref ", k, out["ref"][k], flush=True)
        for k, f in REF_NEGATIVE.items():
            out["negative"][k] = _fmt(f())
            print("neg ", k, out["negative"][k], flush=True)
        save()
    if "--ref-only" in sys.argv:
        return
    torch.set_num_threads(4)
    jobs = [(f"{n} {k}", lambda n=n, k=k: real_gate(n, k)) for n, k in REAL_GATE]
    jobs += [(f"{n} tied_host", lambda n=n: real_tied(n)) for n in REAL_TIED]
    for tag, job in jobs:                              # incremental: a finished pair is kept
        if tag in out["real"]:
            continue
        out["real"][tag] = _fmt(job())
        print("real", tag, out["real"][tag], flush=True)
        save()


if __name__ == "__main__":
    main()
