"""dense SwiGLU FFN  subset  MoE, on the REAL transformers MoE blocks, via a zeroed router.

Construction. Zero the router's input weights. Every token then sees the SAME router logits
(all 0), so the SAME expert set fires for every token with the SAME weights. Deal the dense
FFN's d_ff hidden units into those picked experts (expert e gets a slice of the gate/up rows and
the matching down columns; leftover units are zero rows -> SiLU(0)*0 = 0), plus any shared
expert, and divide each slot's down_proj by the constant weight the block multiplies it by.
Capacity: d_ff <= k * e_w (+ e_s for a shared expert) -- ACTIVE capacity.

Two things the one-line version ("W2 x k") gets wrong, both handled here by PROBING the real
module at construction time rather than assuming:

  1. WHICH experts fire is a tie-break. topk over E equal scores is implementation-defined. On
     CPU torch 2.x, Mixtral (E=8, k=2) picks experts [6, 5], not [0, 1]: dealing into 0..k-1
     silently drops the whole FFN. probe_router() records which experts the block actually
     calls (forward hooks) and asserts every token was routed identically.
  2. The CONSTANT weight depends on the router family:
       Mixtral          softmax, renormalised top-k     -> 1/k        (down x k)
       Qwen2-MoE        softmax, norm_topk_prob=False   -> 1/E        (down x E)
                        (+ shared expert behind sigmoid(shared_expert_gate x) = 1/2 -> x 2)
       DeepSeek-V3      sigmoid scores = 1/2, group-limited top-k, then
                        norm_topk_prob=True             -> rsf / k    (down x k / rsf)
                        norm_topk_prob=False            -> rsf / 2    (down x 2 / rsf)
                        (rsf = routed_scaling_factor; shared experts are ungated -> x 1)
     (HF's DeepseekV3MoE computes its router in fp32 and accumulates in the router dtype, so
     that block only runs in fp32.) probe_router() measures each slot's weight from the block's own output, so the scale is
     whatever the module really applies.

A tie-robust variant (mode="replicate") puts the WHOLE dense FFN into every expert, so any
tie-break gives the same answer; it needs d_ff <= e_w (one expert's width), not k * e_w.

Run:  python -m moe.zero_router
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- expert accessors (Mixtral: w1 gate / w3 up / w2 down; Qwen2-MoE, DeepSeek: *_proj) ---

def _proj(mlp):
    if hasattr(mlp, "w1"):
        return mlp.w1, mlp.w3, mlp.w2
    return mlp.gate_proj, mlp.up_proj, mlp.down_proj


def _router_weight(block):
    g = block.gate
    return g.weight


def _shared(block):
    for name in ("shared_expert", "shared_experts"):
        if hasattr(block, name):
            return getattr(block, name)
    return None


def _zero_shared_gate(block):
    if getattr(block, "shared_expert_gate", None) is not None:
        with torch.no_grad():
            block.shared_expert_gate.weight.zero_()


@torch.no_grad()
def probe_router(block, d, dtype, T=7, seed=123):
    """With the router zeroed: which experts does this block call, and with what constant weight?
    Returns (picked expert ids in call order, {slot: weight}) where slot is an expert id or
    'shared'. Weights are measured from the block's own output (one slot active at a time)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, d, dtype=dtype, generator=g)
    calls = {}
    hooks = [e.register_forward_hook(lambda m, i, o, j=j: calls.__setitem__(j, calls.get(j, 0) + i[0].shape[0]))
             for j, e in enumerate(block.experts)]
    sh = _shared(block)
    saved = {j: [p.weight.clone() for p in _proj(e)] for j, e in enumerate(block.experts)}
    if sh is not None:
        saved["shared"] = [p.weight.clone() for p in _proj(sh)]
    try:
        # probe weights: every expert = the same random SwiGLU, so each slot's output is w * f(x)
        f_w = [torch.randn(p.weight.shape, dtype=dtype, generator=g) for p in _proj(block.experts[0])]
        for e in block.experts:
            for p, w in zip(_proj(e), f_w):
                p.weight.copy_(w)
        _ = block(x)
        picked = [j for j in range(len(block.experts)) if calls.get(j, 0)]
        assert all(calls[j] == T for j in picked), f"tokens routed differently: {calls}"
        f_x = F.linear(F.silu(F.linear(x, f_w[0])) * F.linear(x, f_w[1]), f_w[2])

        def out_with_only(slot):
            for j, e in enumerate(block.experts):
                _proj(e)[2].weight.copy_(f_w[2] if j == slot else torch.zeros_like(f_w[2]))
            if sh is not None:
                gp, up, dn = _proj(sh)
                if slot == "shared":   # shared expert shapes differ; give it its own probe map
                    for p in (gp, up, dn):
                        p.weight.copy_(torch.randn(p.weight.shape, dtype=dtype, generator=g))
                else:
                    dn.weight.zero_()
            o = block(x)
            return o[0] if isinstance(o, tuple) else o

        weights = {}
        for j in picked:
            o = out_with_only(j)
            weights[j] = float((o * f_x).sum() / (f_x * f_x).sum())
        if sh is not None:
            o = out_with_only("shared")
            gp, up, dn = _proj(sh)
            s_x = dn(F.silu(gp(x)) * up(x))
            weights["shared"] = float((o * s_x).sum() / (s_x * s_x).sum())
    finally:
        for h in hooks:
            h.remove()
        for j, e in enumerate(block.experts):
            for p, w in zip(_proj(e), saved[j]):
                p.weight.copy_(w)
        if sh is not None:
            for p, w in zip(_proj(sh), saved["shared"]):
                p.weight.copy_(w)
    return picked, weights


@torch.no_grad()
def deal_dense_into_moe(block, Wg, Wu, Wd, mode="deal", use_shared=True, picked=None):
    """Write the dense SwiGLU  y = Wd (SiLU(Wg x) * Wu x)  (Wg, Wu: [m, d]; Wd: [d, m]) into a real
    MoE block through a zeroed router. mode='deal' slices the m units over the picked experts
    (+ shared); mode='replicate' copies the whole FFN into every expert (tie-robust, m <= e_w).
    `picked` overrides the probe (to show what a wrong assumption costs). Returns the probe."""
    dtype, d, m = Wg.dtype, Wg.shape[1], Wg.shape[0]
    _router_weight(block).zero_()
    _zero_shared_gate(block)
    if hasattr(block.gate, "e_score_correction_bias"):
        block.gate.e_score_correction_bias.zero_()
    probed, weights = probe_router(block, d, dtype)
    picked = probed if picked is None else list(picked)
    sh = _shared(block) if use_shared else None
    for e in block.experts:                                 # start from all-zero experts
        for p in _proj(e):
            p.weight.zero_()
    if _shared(block) is not None:
        for p in _proj(_shared(block)):
            p.weight.zero_()

    def write(mlp, lo, hi, scale):
        gp, up, dn = _proj(mlp)
        n = hi - lo
        gp.weight[:n] = Wg[lo:hi]
        up.weight[:n] = Wu[lo:hi]
        dn.weight[:, :n] = Wd[:, lo:hi] * scale

    if mode == "replicate":
        assert m <= _proj(block.experts[0])[0].weight.shape[0], "replicate needs d_ff <= e_w"
        tot = sum(weights[j] for j in probed)
        for e in block.experts:
            write(e, 0, m, 1.0 / tot)
        return probed, weights
    slots = [(block.experts[j], weights.get(j, weights[probed[0]])) for j in picked]
    if sh is not None:
        slots.append((sh, weights["shared"]))
    lo = 0
    for mlp, w in slots:
        if lo >= m:
            break
        hi = min(m, lo + _proj(mlp)[0].weight.shape[0])
        write(mlp, lo, hi, 1.0 / w)
        lo = hi
    assert lo >= m, f"capacity: d_ff={m} > what the {len(slots)} active slots hold"
    return probed, weights


# --- real blocks from tiny configs ---------------------------------------------------------

def make_block(family, dtype=torch.float64, **kw):
    torch.manual_seed(0)
    if family == "mixtral":
        from transformers import MixtralConfig
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock as B
        cfg = MixtralConfig(hidden_size=32, intermediate_size=24, num_local_experts=8,
                            num_experts_per_tok=2, **kw)
    elif family == "qwen2_moe":
        from transformers import Qwen2MoeConfig
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock as B
        cfg = Qwen2MoeConfig(hidden_size=32, moe_intermediate_size=24, shared_expert_intermediate_size=16,
                             num_experts=8, num_experts_per_tok=2, **{"norm_topk_prob": False, **kw})
    elif family == "deepseek_v3":
        from transformers import DeepseekV3Config
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE as B
        cfg = DeepseekV3Config(hidden_size=32, moe_intermediate_size=24, n_routed_experts=16,
                               num_experts_per_tok=4, n_group=4, topk_group=2, n_shared_experts=1,
                               **{"routed_scaling_factor": 2.5, "norm_topk_prob": True, **kw})
    else:
        raise ValueError(family)
    return B(cfg).to(dtype).eval(), cfg


def check(family, dtype=torch.float64, d_ff=None, mode="deal", use_shared=True, picked=None, T=33, **kw):
    """Build the dense FFN, deal it into a real block, compare outputs. Returns a dict."""
    block, cfg = make_block(family, dtype, **kw)
    d = cfg.hidden_size
    ew = _proj(block.experts[0])[0].weight.shape[0]
    sh = _shared(block)
    es = _proj(sh)[0].weight.shape[0] if (sh is not None and use_shared) else 0
    k = cfg.num_experts_per_tok
    m = d_ff or (ew if mode == "replicate" else k * ew + es)
    g = torch.Generator().manual_seed(7)
    Wg, Wu, Wd = (torch.randn(s, dtype=dtype, generator=g) / s[1] ** 0.5 for s in ((m, d), (m, d), (d, m)))
    probed, weights = deal_dense_into_moe(block, Wg, Wu, Wd, mode, use_shared, picked)
    x = torch.randn(2, T, d, dtype=dtype, generator=g)
    with torch.no_grad():
        dense = F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)
        out = block(x)
        out = out[0] if isinstance(out, tuple) else out
    return dict(family=family, dtype=str(dtype).split(".")[-1], mode=mode, d_ff=m, k=k, e_w=ew, e_s=es,
                picked=probed, weights={k_: round(v, 6) for k_, v in weights.items()},
                err=(out - dense).abs().max().item(), scale=dense.abs().max().item())


def main():
    rows = []
    for dtype in (torch.float64, torch.float32):
        rows.append(check("mixtral", dtype))
        rows.append(check("mixtral", dtype, mode="replicate"))
        rows.append(check("qwen2_moe", dtype))                         # norm_topk_prob=False
        rows.append(check("qwen2_moe", dtype, norm_topk_prob=True))
        if dtype == torch.float32:   # HF DeepseekV3MoE hard-casts routing to fp32 (index_add_
            rows.append(check("deepseek_v3", dtype))                   # fails on an fp64 block)
            rows.append(check("deepseek_v3", dtype, norm_topk_prob=False))
    rows.append(check("mixtral", torch.float64, picked=[0, 1]))        # the naive assumption
    for r in rows:
        print(f"{r['family']:12s} {r['dtype']:8s} {r['mode']:9s} d_ff={r['d_ff']:3d} k={r['k']} "
              f"picked={r['picked']} w={r['weights']}  max|d|={r['err']:.2e} (|y|max {r['scale']:.1f})")
    print("last row: dealing into experts 0..k-1 (the naive reading) instead of the probed set.")


if __name__ == "__main__":
    main()
