"""Recurrent capacity (graph/model_graph.py _rc_pair_ok, rules RC_GE / RC_M2 / RC_GDN / RC_KDA / RC_EQ) and Zamba2's
shared blocks: stock Mamba / Mamba2 / Qwen3-Next / RWKV where transformers has the class, a reference module for KDA
(no stock class) and for the shared-block rule."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from verify.spot.catalog import compare, entry, hf_fn
from verify.spot.catalog._ref import F64, RefLM, fn
from verify.spot.catalog.ffn_moe import _cmp, _pad_into
from verify.spot.catalog.ref_slots import REF, _check

V = 61


def _stock(mt, seed=0, noise=0.08, **kw):
    from transformers import AutoConfig, AutoModelForCausalLM
    from elephantine.emulate import warnings_off
    c = AutoConfig.for_model(mt, vocab_size=V, **kw)
    c._attn_implementation = "eager"
    with warnings_off():
        torch.manual_seed(seed)
        m = AutoModelForCausalLM.from_config(c).to(F64).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * noise)
    return m


def _zero(mt, **kw):
    h = _stock(mt, seed=1, **kw)
    with torch.no_grad():
        for p in h.parameters():
            p.zero_()
    return h


def _assert_edge(traw_model, hraw_model, exact=True):
    """The pair (configs of the two stock models) must be an edge of the headline relation."""
    from graph.edge_plan import edge_plan
    from graph.model_graph import classify, dims_of
    d = lambda m: {**m.config.to_dict(), "architectures": [type(m).__name__]}
    A = {"slots": classify(d(traw_model)), "dims": dims_of(d(traw_model))}
    B = {"slots": classify(d(hraw_model)), "dims": dims_of(d(hraw_model))}
    p = edge_plan(A, B)
    assert p["ok"], f"not an edge of the headline relation: {[r['slot'] for r in p['slots']]}"


_M1 = dict(hidden_size=32, state_size=8, expand=2, time_step_rank=4, conv_kernel=4, num_hidden_layers=2,
           tie_word_embeddings=False)


@entry("recurrent:ge", instance="mamba (d_inner 64, d_state 8, dt rank 4, conv 4) -> mamba (d_inner 96, d_state 16, "
                                "dt rank 8, conv 6): channels / states / rank / taps zero-padded")
def rc_mamba1_ge():
    t = _stock("mamba", **_M1)
    h = _zero("mamba", **{**_M1, "expand": 3, "state_size": 16, "time_step_rank": 8, "conv_kernel": 6})
    _assert_edge(t, h)
    ts, hs = t.state_dict(), h.state_dict()
    di, N, R, K = 64, 8, 4, 4
    Dh, Nh, Rh, Kh = 96, 16, 8, 6
    with torch.no_grad():
        for k, v in ts.items():
            if ".mixer." not in k:
                hs[k].copy_(v)
                continue
            n = k.split(".mixer.")[1]
            H = hs[k]
            if n == "in_proj.weight":                       # [x (di); z (di)]
                H[:di], H[Dh:Dh + di] = v[:di], v[di:]
            elif n == "conv1d.weight":                      # causal: the target's taps on the newest positions
                H[:di, :, Kh - K:] = v
            elif n == "x_proj.weight":                      # [dt rank; B (N); C (N)]
                H[:R, :di], H[Rh:Rh + N, :di], H[Rh + Nh:Rh + Nh + N, :di] = v[:R], v[R:R + N], v[R + N:]
            elif n == "dt_proj.weight":
                H[:di, :R] = v
            elif n == "A_log":
                H[:di, :N] = v
            else:                                           # conv bias, dt bias, D, out_proj
                _pad_into(H, v)
    return _cmp(t, h)


_M2 = dict(hidden_size=32, state_size=8, expand=2, num_heads=8, head_dim=8, n_groups=1, conv_kernel=4,
           num_hidden_layers=2, tie_word_embeddings=False)


@entry("recurrent:m2", instance="mamba2 (8 heads x 8, one group, gated norm over the full width) -> mamba2 (16 heads "
                                "x 8, d_state 16): every channel twice (the rms over two copies is the rms of one), "
                                "out_proj reads the first copy")
def rc_mamba2_copies():
    t = _stock("mamba2", **_M2)
    h = _zero("mamba2", **{**_M2, "expand": 4, "num_heads": 16, "state_size": 16})
    _assert_edge(t, h)
    ts, hs = t.state_dict(), h.state_dict()
    I, N, nh = 64, 8, 8
    Ih, Nh, nhh = 128, 16, 16
    with torch.no_grad():
        for k, v in ts.items():
            if ".mixer." not in k:
                hs[k].copy_(v)
                continue
            n = k.split(".mixer.")[1]
            H = hs[k]
            if n == "in_proj.weight":                       # [z (I); x (I); B (N); C (N); dt (nh)]
                z, x, B, C, dt = v.split([I, I, N, N, nh])
                H[:I], H[I:2 * I] = z, z
                H[Ih:Ih + I], H[Ih + I:2 * Ih] = x, x
                H[2 * Ih:2 * Ih + N] = B
                H[2 * Ih + Nh:2 * Ih + Nh + N] = C
                H[2 * Ih + 2 * Nh:2 * Ih + 2 * Nh + nh], H[2 * Ih + 2 * Nh + nh:] = dt, dt
            elif n in ("conv1d.weight", "conv1d.bias"):     # channels [x (I); B (N); C (N)]
                x, B, C = v.split([I, N, N])
                H[:I], H[I:Ih] = x, x
                H[Ih:Ih + N] = B
                H[Ih + Nh:Ih + Nh + N] = C
            elif n in ("dt_bias", "A_log", "D"):
                H[:nh], H[nh:] = v, v
            elif n == "norm.weight":
                H[:I], H[I:] = v, v
            elif n == "out_proj.weight":
                H[:, :I] = v
    return _cmp(t, h)


_QN = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
           head_dim=8, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
           linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
           num_experts=0, mlp_only_layers=[0, 1], tie_word_embeddings=False, max_position_embeddings=128)


def _torch_path_qwen3next():
    """The fla / causal-conv1d kernels are GPU-only: build the stock module on its torch fallbacks."""
    import transformers.models.qwen3_next.modeling_qwen3_next as q3
    for name in ("FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule"):
        setattr(q3, name, None)
    q3.is_fast_path_available = False


@entry("recurrent:gdn", instance="qwen3_next GDN (2 key heads x 8, 4 value heads x 8) -> qwen3_next (value head dim "
                                 "16: each value channel twice, the per-head gated RMSNorm sees the same rms; key "
                                 "dim equal)")
def rc_gdn_value_multiple():
    _torch_path_qwen3next()
    t = _stock("qwen3_next", **_QN)
    h = _zero("qwen3_next", **{**_QN, "linear_value_head_dim": 16})
    _assert_edge(t, h)
    ts, hs = t.state_dict(), h.state_dict()
    nk, nv, dk, dv = 2, 4, 8, 8
    g = nv // nk
    with torch.no_grad():
        for k, v in ts.items():
            H = hs[k]
            if ".linear_attn." not in k:
                H.copy_(v)
                continue
            n = k.split(".linear_attn.")[1]
            if n == "in_proj_qkvz.weight":                  # per key group: [q dk; k dk; v g dv; z g dv]
                for kg in range(nk):
                    src = v.view(nk, 2 * dk + 2 * g * dv, -1)[kg]
                    dst = H.view(nk, 2 * dk + 2 * g * 2 * dv, -1)[kg]
                    q_, k_, vv, zz = src.split([dk, dk, g * dv, g * dv])
                    dst[:dk], dst[dk:2 * dk] = q_, k_
                    for j in range(g):
                        for c in range(2):
                            dst[2 * dk + j * 2 * dv + c * dv: 2 * dk + j * 2 * dv + (c + 1) * dv] = vv[j * dv:(j + 1) * dv]
                            o = 2 * dk + g * 2 * dv
                            dst[o + j * 2 * dv + c * dv: o + j * 2 * dv + (c + 1) * dv] = zz[j * dv:(j + 1) * dv]
            elif n == "conv1d.weight":                      # [q (nk dk); k (nk dk); v (nv dv)]
                qk = 2 * nk * dk
                H[:qk] = v[:qk]
                for hv in range(nv):
                    for c in range(2):
                        H[qk + hv * 2 * dv + c * dv: qk + hv * 2 * dv + (c + 1) * dv] = v[qk + hv * dv: qk + (hv + 1) * dv]
            elif n == "norm.weight":
                H[:dv], H[dv:] = v, v
            elif n == "out_proj.weight":
                for hv in range(nv):
                    H[:, hv * 2 * dv: hv * 2 * dv + dv] = v[:, hv * dv:(hv + 1) * dv]
            else:                                           # in_proj_ba, dt_bias, A_log: per value head
                H.copy_(v)
    from tests.test_universal_elephantine import _lift_fp32
    with _lift_fp32():
        return compare(hf_fn(t), hf_fn(h), V, tol=1e-9)


_RW = dict(hidden_size=32, num_hidden_layers=2, attention_hidden_size=32, intermediate_size=64,
           tie_word_embeddings=False)


@entry("recurrent:eq", instance="rwkv (2 layers x 32) -> rwkv with the same recurrent dims (RC_EQ: every recurrent dim "
                                "equal, so the construction is the copy)")
def rc_rwkv_eq():
    t = _stock("rwkv", **_RW)
    h = _zero("rwkv", **_RW)
    _assert_edge(t, h)
    ts, hs = t.state_dict(), h.state_dict()
    with torch.no_grad():
        for k, v in ts.items():
            hs[k].copy_(v)
    return _cmp(t, h)


# ------------------------------------------------------------------ KDA (no stock class): reference

class KDA:
    """A KDA-style layer: q, k l2-normed (q x 1 / sqrt(hd)), a per-channel decay alpha = sigmoid(f_b f_a x) from a
    low-rank gate, a delta rule S <- diag(alpha) S (I - beta k k^T) + beta k v^T, a short causal conv on q / k / v."""

    def __init__(self, d, H, hd, rank, conv, seed=0):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g, dtype=F64) * 0.3
        self.H, self.hd, self.conv = H, hd, conv
        self.W = {"q": r(H * hd, d), "k": r(H * hd, d), "v": r(H * hd, d), "fa": r(rank, d), "fb": r(H * hd, rank),
                  "b": r(H, d), "o": r(d, H * hd), "cq": r(H * hd, conv), "ck": r(H * hd, conv), "cv": r(H * hd, conv)}

    def zero(self):
        self.W = {k: torch.zeros_like(v) for k, v in self.W.items()}
        return self

    def _conv(self, x, w):                                 # causal depthwise, taps w[:, -1] on the current token
        T = x.shape[1]
        xp = F.pad(x, (0, 0, w.shape[1] - 1, 0))
        return sum(xp[:, i:i + T] * w[:, i] for i in range(w.shape[1]))

    def __call__(self, x):
        W, H, hd = self.W, self.H, self.hd
        B, T, _ = x.shape
        q = F.silu(self._conv(x @ W["q"].T, W["cq"])).view(B, T, H, hd)
        k = F.silu(self._conv(x @ W["k"].T, W["ck"])).view(B, T, H, hd)
        v = F.silu(self._conv(x @ W["v"].T, W["cv"])).view(B, T, H, hd)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) / hd ** 0.5
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        alpha = torch.sigmoid((x @ W["fa"].T) @ W["fb"].T).view(B, T, H, hd)
        beta = torch.sigmoid(x @ W["b"].T)
        S = torch.zeros(B, H, hd, hd, dtype=F64)
        out = []
        for t in range(T):
            S = alpha[:, t][..., None] * S
            kt, vt, bt = k[:, t], v[:, t], beta[:, t][..., None, None]
            S = S - bt * kt[..., :, None] * (kt[..., None, :] @ S) + bt * kt[..., :, None] * vt[..., None, :]
            out.append((q[:, t][..., None, :] @ S).squeeze(-2))
        return torch.stack(out, 1).reshape(B, T, H * hd) @ W["o"].T


class KdaLM:
    def __init__(self, d=32, L=2, **kw):
        g = torch.Generator().manual_seed(7)
        self.emb = torch.randn(V, d, generator=g, dtype=F64)
        self.lm = torch.randn(V, d, generator=g, dtype=F64) * 0.5
        self.layers = [KDA(d, seed=i, **kw) for i in range(L)]

    def __call__(self, ids):
        x = self.emb[ids]
        for l in self.layers:
            x = x + l(x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt())
        return x @ self.lm.T


@entry("recurrent:kda", instance=f"KDA (2 heads x 8, gate rank 4, conv 3) -> KDA (3 heads x 8, gate rank 8, conv 4): "
                                 f"heads / rank / taps zero-padded, head_dim equal ({REF}: no stock KDA class)")
def rc_kda():
    t = KdaLM(H=2, hd=8, rank=4, conv=3)
    h = KdaLM(H=3, hd=8, rank=8, conv=4)
    h.emb, h.lm = t.emb.clone(), t.lm.clone()
    for lt, lh in zip(t.layers, h.layers):
        lh.zero()
        Wt, Wh = lt.W, lh.W
        n = 2 * 8
        for key in ("q", "k", "v", "b"):
            Wh[key][: Wt[key].shape[0]] = Wt[key]
        Wh["fa"][:4] = Wt["fa"]
        Wh["fb"][:n, :4] = Wt["fb"]
        Wh["o"][:, :n] = Wt["o"]
        for key in ("cq", "ck", "cv"):
            Wh[key][:n, 1:] = Wt[key]                  # the oldest tap zero
        # an all-zero head: q = k = 0 (l2 norm of 0 is 0), its state stays 0, o reads nothing from it
    return compare(t, h, V, tol=1e-9)


# ------------------------------------------------------------------ Zamba2's shared blocks: reference

@entry("recurrent:zamba_shared", instance=f"one shared block at 4 positions -> 2 shared blocks alternating (a multiple: "
                                          f"each host block a copy of the target's) ({REF})")
def rc_zamba_shared():
    t = RefLM(loop=(1, 4), L=4).init(0)
    h = RefLM(**{**t.c, "loop": (2, 2)}).init(1)
    for k, v in t.p.items():
        if k.startswith("l0."):
            h.p[k] = v.clone()
            h.p["l1." + k[3:]] = v.clone()
        else:
            h.p[k] = v.clone()
    return _check(t, h)
