"""Capacity growth inside elephantine: a LARGER UniversalElephantine configuration emulates a SMALLER one exactly.

`grow(small, big_cfg)` writes the weights of a UniversalElephantine with a larger UConfig from a smaller one's, so
that big(ids) == small(ids) (fp64 rounding). Every capacity axis the relation compares is a zero-padding of the
small weights -- no other configuration change -- which is what makes capacity-only ordering config-only inside
elephantine:

  hidden d          readers (q / k / v / g, in_proj, router, experts, shared FFN, lm_head) pad input columns,
                    writers (embedding, o_proj, out_proj, expert / shared down) pad output rows; every UNorm over d
                    takes eps_B = eps_A d_A / d_B and gamma_B = [gamma_A sqrt(d_A / d_B), 0] (elephantine/
                    hidden_width.py), so rms_B(x) = sqrt(d_A / d_B) rms_A(x) on a stream whose tail is 0
  depth L           the small layers at increasing positions `layer_pos`; every other layer is the identity
                    (all three sub-block coefficients post_coef = 0)
  heads H           extra heads have zero q / k / v / o (their output is 0), sink logit -inf, null key 0
  head_dim hd       the small head's halves at [0, hd_A/2) and [hd_B/2, hd_B/2 + hd_A/2) (the rotary pairs
                    (i, i + hd/2) kept, inv_freq per pair copied, the extra pairs zero); the attention scale and the
                    qk-norm averaging width (qk_width) keep the small values
  KV / q capacity   = H * hd in this host (its attention is MHA with GQA targets tiled): the two axes above
  shared_ff, expert_ff   zero-padded gate / up rows and down columns (act(0, 0) = 0 for the gated branches)
  experts E         extra experts zero, never selected (selection bias -inf); exact for a sigmoid router, or a
                    softmax router renormalised over the top k whose selection order is that of the scores
                    (no selection bias, or selection on the raw logits)
  recurrent state   the SSM block's heads, dk, dv (zero-padded q / k / v rows: the state's extra rows / columns stay
                    0; the output norm's averaging matrix `avg` is a weight, mapped onto the small channels) and its
                    in_proj width ssm_in (the [u, conv(u)] knob input remapped)

Raises NotExact for a branch whose growth is not a weight setting here (see the messages), and NotImplementedError
for a branch this construction does not map (it checks that every unmapped parameter is at its default).
"""
from __future__ import annotations

import dataclasses
import math

import torch

from elephantine.universal import UAttention, UConfig, UFFN, ULayer, UNorm, USSM, UniversalElephantine, F64

CAPACITY = ("d", "L", "heads", "head_dim", "ssm_in", "ssm_heads", "dk", "dv", "experts", "expert_ff", "shared_ff")


class NotExact(ValueError):
    pass


class _Maps:
    def __init__(self, a: UConfig, b: UConfig):
        ar = torch.arange
        self.r = a.d / b.d
        hs, hb = a.head_dim, b.head_dim
        assert hs % 2 == 0 and hb % 2 == 0
        j = ar(hs)
        jm = torch.where(j < hs // 2, j, hb // 2 + j - hs // 2)
        head = lambda n_a, w_a, w_b, m: (ar(n_a)[:, None] * w_b + m[None, :]).reshape(-1)
        self.sp = {
            None: None, "d": ar(a.d), "H": ar(a.heads), "hd": jm, "Hhd": head(a.heads, hs, hb, jm), "P": ar(hs // 2),
            "E": ar(a.experts), "f": ar(a.expert_ff), "fs": ar(a.shared_ff), "D": ar(a.ssm_in),
            "R": torch.cat((ar(a.ssm_in), b.ssm_in + ar(a.ssm_in))), "SH": ar(a.ssm_heads), "dk": ar(a.dk),
            "dv": ar(a.dv), "Sk": head(a.ssm_heads, a.dk, b.dk, ar(a.dk)), "C": head(a.ssm_heads, a.dv, b.dv, ar(a.dv)),
        }

    @torch.no_grad()
    def put(self, dst, src, spaces):
        if not spaces:
            assert dst.shape == src.shape, (dst.shape, src.shape)
            dst.copy_(src)
            return
        idx = [self.sp[s] if s is not None else torch.arange(n) for s, n in zip(spaces, src.shape)]
        for i, n in zip(idx, src.shape):
            assert len(i) == n, (spaces, src.shape)
        dst[torch.meshgrid(*idx, indexing="ij")] = src


# per module type: parameter / buffer name -> its spaces (() = same shape, copied)
_S = ()
ATTN = {**{f"{p}.weight": ("Hhd", "d") for p in "qkvg" for p in [p + "_proj"]},
        **{f"{p}_proj.bias": ("Hhd",) for p in "qkvg"},
        "o_proj.weight": ("d", "Hhd"), "o_proj.bias": ("d",),
        **{n: ("H", "hd") for n in ("q_nw", "k_nw", "q_nb", "k_nb", "v_nw", "hg_w")},
        "inv_freq": ("P",), "o_hmix": ("H", "H"), "o_avg": ("Hhd", "Hhd"), "o_w": ("Hhd",),
        "null_logit": ("H",), "null_key": ("H", "hd"), "alibi": ("H",), "hg_c": ("H",), "hg_b": ("H",),
        **{n: _S for n in ("q_coef", "k_coef", "gate_coef", "scale", "o_coef", "kv_read", "kv_write", "qk_centre",
                           "qk_post", "qk_full", "v_c", "gate_sp", "gate_silu", "qk_eps", "o_eps", "window", "n_sink", "cap",
                           "qkv_clip", "qk_width", "v_eps", "kv_slot", "qtemp")}}
SSM = {"in_proj.weight": ("D", "d"), "in_proj.bias": ("D",), "conv_w": ("D", None), "conv_b": ("D",),
       "act_silu": ("D",), "act_lin": ("D",),
       **{f"{p}.weight": ("Sk", "R") for p in ("pq", "pk", "p1", "p2")}, **{f"{p}.bias": ("Sk",) for p in ("pq", "pk", "p1", "p2")},
       **{f"{p}.weight": ("C", "R") for p in ("pv", "pz")}, **{f"{p}.bias": ("C",) for p in ("pv", "pz")},
       **{f"{p}.weight": ("SH", "R") for p in ("pb", "pe", "pi")}, **{f"{p}.bias": ("SH",) for p in ("pb", "pe", "pi")},
       **{n: ("SH", "dk") for n in ("q_nw", "k_nw", "A1", "A2", "A3", "mem")},
       **{n: ("SH",) for n in ("cg", "cg0", "reset")},
       "cw": (None, "SH"), "Dskip": ("SH", "dv"), "gate": (None, "C"), "avg": ("C", "C"), "norm_w": ("C",),
       "norm_b": ("C",), "mix": ("C", "C"), "out_coef": (None, "C"), "out_proj.weight": ("d", "C"), "out_proj.bias": ("d",),
       **{n: _S for n in ("q_coef", "k_coef", "v_write", "v_read", "y_write", "y_read", "n_eps", "l2_eps", "kap_eps",
                          "ridge_iters", "l2_eps_out", "norm_eps", "dt_min")}}
FFN = {"router.weight": ("E", "d"), "router.bias": ("E",), "sel_bias": ("E",), "exp_scale": ("E",),
       "w_gate": ("E", "f", "d"), "w_up": ("E", "f", "d"), "w_down": ("E", "d", "f"), "b_gate": ("E", "f"),
       "b_up": ("E", "f"), "b_down": ("E", "d"), "poly_w": ("E", None), "poly_b": ("E", None),
       "s_gate.weight": ("fs", "d"), "s_gate.bias": ("fs",), "s_up.weight": ("fs", "d"), "s_up.bias": ("fs",),
       "s_down.weight": ("d", "fs"), "s_down.bias": ("d",), "s_router.weight": (None, "d"),
       **{n: _S for n in ("renorm", "score_c", "sel_logit", "route_scale", "c_situ", "c_clip", "c_relu2", "c_relu",
                          "up_shift", "c_gelu", "c_geglu", "gelu_tanh", "c_mish", "c_xielu", "xielu_a", "c_poly",
                          "poly_sig", "s_poly_w", "s_poly_b", "s_coef", "grp", "renorm_eps", "score_sp", "hash_tab",
                          "router_cap", "smix_eps", "situ_beta", "clip_lim", "clip_post", "xielu_be", "poly_cfg")}}
NORM = {"weight": ("d",), "bias": ("d",), "pre_w": ("d",), "g_up.weight": (None, "d"), "g_down.weight": ("d", None),
        **{n: _S for n in ("centre", "gate_c", "pre_c", "ident", "eps", "pre_eps")}}
LAYER = {n: _S for n in ("sn_c", "post_coef", "ffn_split", "res_write", "res_mix", "seq", "par", "out_scale", "res_c")}
TOP = {"embed.weight": (None, "d"), "lm_head.weight": (None, "d"), "lm_head.bias": _S,
       **{n: _S for n in ("final_cap", "head_act", "pool", "pool_pad", "l2_out", "block_size", "emb_norm_c")}}
TABLES = {UAttention: ATTN, USSM: SSM, UFFN: FFN, UNorm: NORM, ULayer: LAYER, UniversalElephantine: TOP}


def _own(m):
    """A module's own tensors, by the names the tables use (child Linear / ParameterDict tensors as 'child.name'),
    excluding the tensors of child modules that have a table of their own."""
    out = dict(m.named_parameters(recurse=False))
    out.update(m.named_buffers(recurse=False))
    for cn, ch in m.named_children():
        if type(ch) in TABLES or isinstance(ch, torch.nn.ModuleList):
            continue
        for n, t in list(ch.named_parameters()) + list(ch.named_buffers()):
            out[f"{cn}.{n}"] = t
    return out


def _pairs(small, fresh, big, layer_pos):
    """(small module, fresh small module, big module) triples, the small layers at big positions layer_pos."""
    def walk(a, f, b):
        yield a, f, b
        for (n, ca), (_, cf) in zip(a.named_children(), f.named_children()):
            if n == "layers":
                for i, l in enumerate(ca):
                    yield from walk(l, cf[i], b.layers[layer_pos[i]])
            elif type(ca) in TABLES:
                yield from walk(ca, cf, getattr(b, n))
            elif any(type(x) in TABLES for x in ca.modules()):
                raise NotImplementedError(f"module {n} ({type(ca).__name__}) is not grown")
    yield from walk(small, fresh, big)


@torch.no_grad()
def grow(small: UniversalElephantine, big_cfg: UConfig, layer_pos=None) -> UniversalElephantine:
    a, b = small.cfg, big_cfg
    diff = {f.name for f in dataclasses.fields(UConfig) if getattr(a, f.name) != getattr(b, f.name)} - set(CAPACITY)
    if diff:
        raise NotImplementedError(f"non-capacity config fields differ: {sorted(diff)}")
    for k in CAPACITY:
        if getattr(b, k) < getattr(a, k):
            raise ValueError(f"{k}: the big config is smaller ({getattr(b, k)} < {getattr(a, k)})")
    layer_pos = list(range(a.L)) if layer_pos is None else list(layer_pos)
    assert len(layer_pos) == a.L and layer_pos == sorted(set(layer_pos)) and layer_pos[-1] < b.L
    if any(getattr(a, k) for k in ("attn_norm_feats", "ssm_norm_feats", "ssm_sig_feats", "ssm_prod_feats",
                                   "ssm_sig2_feats", "lora_adapters", "ple_dim", "hc_mult", "attnres_block",
                                   "mem_attn", "n_prefix", "max_pos", "head_ff", "head_out", "patch_len", "flow_out",
                                   "engram_slots", "engram_hc", "cmp_pool", "dsa_heads", "ssm_dconv")) or a.attnres_full:
        raise NotImplementedError("a branch with its own width (latent norms, LoRA, PLE, hyper-connections, AttnRes, "
                                  "memory, prefix, readouts, Engram, pools, DSA, dynamic conv) is not grown here")
    M = _Maps(a, b)
    fresh = UniversalElephantine(a)
    big = UniversalElephantine(b)
    for ms, mf, mb in _pairs(small, fresh, big, layer_pos):
        tab = TABLES[type(ms)]
        S, F_, B_ = _own(ms), _own(mf), _own(mb)
        for n, t in S.items():
            if n not in tab:
                if not torch.equal(t, F_[n]):
                    raise NotImplementedError(f"{type(ms).__name__}.{n} is set but not grown by this construction")
                continue
            if tab[n]:
                B_[n].zero_()                            # the padding is zero (the big model's own init is not)
            M.put(B_[n], t.detach(), tab[n])
        if isinstance(ms, UNorm):                        # the wider RMSNorm: eps and gamma rescaled
            if ms.centre != 0 and b.d > a.d:
                raise NotExact("UNorm centre (LayerNorm): x - mean(x) averages over the FULL width d_B, so a zero "
                               "tail moves the mean (and writes -mean into the tail); UNorm has no averaging-width knob")
            for n in ("weight", "pre_w"):
                B_[n][: a.d] *= math.sqrt(M.r)
            B_["eps"].mul_(M.r); B_["pre_eps"].mul_(M.r)
        if isinstance(ms, UAttention):
            if ms.qk_full != 0 and b.heads > a.heads:
                raise NotExact("UAttention qk_full: the whole-projection qk-norm width is qk_width * H (t.shape[-2]), "
                               "so extra heads change it")
            if b.heads > a.heads:
                B_["null_logit"][a.heads:] = float("-inf")
        if isinstance(ms, UFFN) and b.experts > a.experts:
            if int(ms.grp[0]) > 1:
                raise NotExact("group-limited routing: extra experts change the groups")
            softmax = ms.score_c != 1 and ms.score_sp == 0
            if softmax and not (ms.renorm == 1 and ms.renorm_eps == 0 and (ms.sel_logit != 0 or not ms.sel_bias.any())):
                raise NotExact("a softmax router over more experts scales every score by Z_A / Z_B: exact only "
                               "renormalised over the top k with a selection order that scale keeps")
            B_["sel_bias"][a.experts:] = float("-inf")
    for j in set(range(b.L)) - set(layer_pos):           # extra depth: identity layers
        big.layers[j].post_coef.zero_()
    return big.eval()
