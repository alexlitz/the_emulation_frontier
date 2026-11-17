"""A larger elephantine emulates a smaller one along every capacity axis, by weights alone (elephantine/grow.py).

The small model is a mixed-feature UniversalElephantine with random weights on every branch it runs: a hybrid stack
(an attention-only layer, a gated-delta-only layer, a layer with both and a post-normed attention output), per-head
weighted qk-norm, an elementwise sigmoid output gate, partial RoPE (one NoPE pair), an attention logit softcap,
scalar per-head sinks AND learned sink keys (null_key), a sigmoid-routed MoE with a selection bias, renormalised top-2,
a sigmoid-gated shared expert, biases everywhere, a final logit softcap. Each capacity axis is grown on its own and
all together; the logits agree to fp64 rounding. A misplaced axis fails (test_misplaced_head_dim_fails), and the one
host limitation found -- LayerNorm (UNorm centre) over a wider stream -- is a strict xfail.

Run:  pytest tests/test_elephantine_grow.py -q
"""
import dataclasses

import pytest
import torch

from elephantine.grow import NotExact, grow
from elephantine.universal import F64, UConfig, UniversalElephantine

SMALL = UConfig(vocab=50, d=16, L=3, heads=4, head_dim=8, ssm_in=12, conv_k=3, ssm_heads=2, dk=4, dv=4,
                experts=4, expert_ff=6, top_k=2, shared_ff=10, sink_key=True)
GROW = {"hidden": dict(d=24), "depth": dict(L=5), "shared_ff": dict(shared_ff=17), "expert_ff": dict(expert_ff=9),
        "heads": dict(heads=6), "head_dim": dict(head_dim=12), "experts": dict(experts=7),
        "ssm_heads": dict(ssm_heads=3), "dk": dict(dk=6), "dv": dict(dv=7), "ssm_in": dict(ssm_in=15)}
ALL = {k: v for g in GROW.values() for k, v in g.items()}
LAYER_POS = {5: [0, 2, 4]}                     # the small layers inside the deeper host (identity layers between)


def _r(g, *shape, s=0.3):
    return torch.randn(*shape, dtype=F64, generator=g) * s


@torch.no_grad()
def small_model(cfg=SMALL, seed=0, layernorm=False):
    g = torch.Generator().manual_seed(seed)
    u = UniversalElephantine(cfg).eval()
    d, H, hd = cfg.d, cfg.heads, cfg.head_dim
    u.embed.weight.copy_(_r(g, cfg.vocab, d, s=1.0))
    u.lm_head.weight.copy_(_r(g, cfg.vocab, d)); u.lm_head.bias.copy_(_r(g, cfg.vocab))
    u.final_cap.fill_(8.0)
    norms = [u.norm] + [n for l in u.layers for n in (l.norm_a, l.norm_s, l.norm_f, l.post_a)]
    for n in norms:
        n.weight.copy_(1 + _r(g, d, s=0.2)); n.eps.fill_(1e-5)
        n.bias.copy_(_r(g, d, s=0.1))
        if layernorm:
            n.centre.fill_(1.0)
    for i, l in enumerate(u.layers):
        a, s, f = l.attn, l.ssm, l.ffn
        # hybrid: layer 0 attention only, layer 1 gated-delta only, layer 2 both (attention post-normed)
        l.post_coef.copy_(torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=F64))
        if i == 0:
            l.post_coef[1] = 0.0
        if i == 1:
            l.post_coef[0] = 0.0
        if i == 2:
            l.post_coef[0] = torch.tensor([0.3, 1.0], dtype=F64)
        # attention
        for p in (a.q_proj, a.k_proj, a.v_proj, a.g_proj):
            p.weight.copy_(_r(g, H * hd, d)); p.bias.copy_(_r(g, H * hd, s=0.1))
        a.o_proj.weight.copy_(_r(g, d, H * hd)); a.o_proj.bias.copy_(_r(g, d, s=0.1))
        a.q_coef.copy_(torch.tensor([0.0, 1.0])); a.k_coef.copy_(torch.tensor([0.0, 1.0]))
        a.q_nw.copy_(1 + _r(g, H, hd, s=0.2)); a.k_nw.copy_(1 + _r(g, H, hd, s=0.2)); a.qk_eps.fill_(1e-5)
        a.gate_coef.copy_(torch.tensor([0.2, 1.0]))
        inv = 1.0 / (100.0 ** (torch.arange(hd // 2, dtype=F64) / (hd // 2)))
        inv[-1] = 0.0                                   # a NoPE pair: partial rotary
        a.inv_freq.copy_(inv)
        a.cap.fill_(5.0)
        a.null_logit.copy_(_r(g, H, s=1.0)); a.null_logit[0] = float("-inf")
        a.null_key.copy_(_r(g, H, hd, s=1.0))
        # gated delta (Qwen3-Next style knobs): silu conv, l2 q / k, decay exp(-A softplus(p1)), delta write
        s.in_proj.weight.copy_(_r(g, cfg.ssm_in, d)); s.in_proj.bias.copy_(_r(g, cfg.ssm_in, s=0.1))
        s.conv_w.copy_(_r(g, cfg.ssm_in, cfg.conv_k)); s.conv_b.copy_(_r(g, cfg.ssm_in, s=0.1))
        s.act_silu.fill_(1.0)
        R = 2 * cfg.ssm_in
        for p, n in ((s.pq, cfg.ssm_heads * cfg.dk), (s.pk, cfg.ssm_heads * cfg.dk), (s.p1, cfg.ssm_heads * cfg.dk),
                     (s.pv, cfg.ssm_heads * cfg.dv), (s.pz, cfg.ssm_heads * cfg.dv), (s.pb, cfg.ssm_heads)):
            p.weight.copy_(_r(g, n, R)); p.bias.copy_(_r(g, n, s=0.1))
        s.q_coef.copy_(torch.tensor([0.0, 1.0])); s.k_coef.copy_(torch.tensor([0.0, 1.0]))
        s.A1.copy_(0.2 + _r(g, cfg.ssm_heads, cfg.dk, s=0.1).abs())
        s.cg.fill_(1.0); s.cw[0].fill_(1.0)
        s.Dskip.copy_(_r(g, cfg.ssm_heads, cfg.dv))
        s.gate[0].fill_(1.0); s.out_coef[0].fill_(1.0)
        C = cfg.ssm_heads * cfg.dv
        s.avg.copy_(torch.block_diag(*[torch.full((cfg.dv, cfg.dv), 1.0 / cfg.dv, dtype=F64)] * cfg.ssm_heads))
        s.norm_w.copy_(1 + _r(g, C, s=0.2))
        s.out_proj.weight.copy_(_r(g, d, C)); s.out_proj.bias.copy_(_r(g, d, s=0.1))
        # MoE: sigmoid router + selection bias, renormalised top-2, a sigmoid-gated shared expert
        E, ef, fs = cfg.experts, cfg.expert_ff, cfg.shared_ff
        f.router.weight.copy_(_r(g, E, d, s=1.0)); f.router.bias.copy_(_r(g, E, s=0.2))
        f.score_c.fill_(1.0); f.renorm.fill_(1.0); f.sel_bias.copy_(_r(g, E, s=0.2)); f.route_scale.fill_(1.5)
        f.w_gate.copy_(_r(g, E, ef, d)); f.w_up.copy_(_r(g, E, ef, d)); f.w_down.copy_(_r(g, E, d, ef))
        f.b_gate.copy_(_r(g, E, ef, s=0.1)); f.b_up.copy_(_r(g, E, ef, s=0.1)); f.b_down.copy_(_r(g, E, d, s=0.1))
        f.s_gate.weight.copy_(_r(g, fs, d)); f.s_up.weight.copy_(_r(g, fs, d)); f.s_down.weight.copy_(_r(g, d, fs))
        f.s_gate.bias.copy_(_r(g, fs, s=0.1)); f.s_up.bias.copy_(_r(g, fs, s=0.1)); f.s_down.bias.copy_(_r(g, d, s=0.1))
        f.s_router.weight.copy_(_r(g, 1, d)); f.s_coef.copy_(torch.tensor([0.0, 1.0]))
    return u


def _ids(seed=1):
    return torch.randint(0, SMALL.vocab, (2, 11), generator=torch.Generator().manual_seed(seed))


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


def _check(small, over):
    big_cfg = dataclasses.replace(small.cfg, **over)
    big = grow(small, big_cfg, LAYER_POS.get(big_cfg.L))
    x = _ids()
    with torch.no_grad():
        a, b = small(x), big(x)
    return _rel(a, b), big


@pytest.fixture(scope="module")
def small():
    return small_model()


def test_small_model_runs_every_branch(small):
    """Each branch binds in the small model: switching it off moves the logits."""
    x = _ids()
    with torch.no_grad():
        ref = small(x)
        for flip in (lambda u: u.layers[2].attn.null_key.zero_(), lambda u: u.layers[0].attn.null_logit.fill_(float("-inf")),
                     lambda u: u.layers[1].ssm.out_proj.weight.zero_(), lambda u: u.layers[2].ffn.sel_bias.zero_(),
                     lambda u: u.layers[0].attn.inv_freq[:1].zero_(), lambda u: u.layers[2].ffn.s_coef.fill_(0.0)):
            u = small_model()
            flip(u)
            assert _rel(ref, u(x)) > 1e-4


@pytest.mark.parametrize("axis", sorted(GROW))
def test_grow_one_axis(small, axis):
    err, big = _check(small, GROW[axis])
    for k, v in GROW[axis].items():
        assert getattr(big.cfg, k) == v > getattr(small.cfg, k)
    assert err < 1e-12, (axis, err)


def test_grow_every_axis_at_once(small):
    err, _ = _check(small, ALL)
    assert err < 1e-12, err


@torch.no_grad()
def test_misplaced_head_dim_fails(small):
    """The rotary pairs bind: the small head's halves placed contiguously at [0, hd_A) (not at [0, hd_A/2) and
    [hd_B/2, hd_B/2 + hd_A/2)) break the rotation."""
    big_cfg = dataclasses.replace(SMALL, head_dim=12)
    big = grow(small, big_cfg)
    hs, hb, H = 8, 12, SMALL.heads
    for l, ls in zip(big.layers, small.layers):
        a, s_ = l.attn, ls.attn
        for n in ("q_proj", "k_proj", "v_proj", "g_proj"):
            w = getattr(a, n).weight.view(H, hb, -1); w.zero_(); w[:, :hs] = getattr(s_, n).weight.view(H, hs, -1)
            bb = getattr(a, n).bias.view(H, hb); bb.zero_(); bb[:, :hs] = getattr(s_, n).bias.view(H, hs)
        for n in ("q_nw", "k_nw", "null_key"):
            getattr(a, n)[:, :hs] = getattr(s_, n)
        o = a.o_proj.weight.view(-1, H, hb); o.zero_(); o[:, :, :hs] = s_.o_proj.weight.view(-1, H, hs)
    x = _ids()
    assert _rel(small(x), big(x)) > 1e-3


@pytest.mark.xfail(strict=True, raises=NotExact,
                   reason="host limitation: UNorm's centre (LayerNorm) subtracts the mean over the FULL width d_B "
                          "(x - x.mean(-1)), so a zero-padded stream tail moves the mean and receives -mean; UNorm has "
                          "no averaging-width knob (UAttention's qk-norm has qk_width). A LayerNorm small model grows "
                          "along every axis but the hidden width (the centred-writer fold would be needed)")
def test_grow_hidden_layernorm():
    small = small_model(layernorm=True)
    err, _ = _check(small, dict(d=24))
    assert err < 1e-12


def test_grow_layernorm_other_axes():
    small = small_model(layernorm=True)
    over = {k: v for k, v in ALL.items() if k != "d"}
    err, _ = _check(small, over)
    assert err < 1e-12, err
