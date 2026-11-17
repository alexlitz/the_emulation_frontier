"""Layer 2 of the verification plan, second batch: the Lean theorems added since the first 118
(namespace EP, proofs/lean), each checked against
the repo function that BUILDS that construction. The first batch is tests/test_lean_bridges.py.

Every test is named after its theorem (with the theorem name in its docstring) and checks the
equation (exact), the convergence (limit) or the inequality (bound) the theorem states, on the repo's
own function, over Hypothesis-drawn inputs in float64. verify/BRIDGES.md maps theorem -> function ->
test -> status. Tests named `test_condition_*` show the equation FAILS when one of the theorem's
hypotheses is dropped; tests named `test_mismatch_*` pin a repo function that builds something other
than the theorem's construction (each is explained in BRIDGES.md).

Tolerances. U = 2^-53 is the float64 unit roundoff. A result assembled from n floating-point
operations on terms of magnitude <= S is within about n U S of the exact value, so an exact identity is
checked at `fp(S, n)` = n U (1 + S) with n a generous count for that computation (stated at each use).
Bounds get the same slack on the right-hand side. Where `transformers` itself rounds to float32 the
slack is float32's, 2^-24, and the test says so.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

F64 = torch.float64
U = 2.0 ** -53                      # float64 unit roundoff
U32 = 2.0 ** -24                    # float32 unit roundoff (where transformers casts to float32)


def fp(scale, n=1e3):
    """n float64 roundings of terms of size <= scale."""
    return n * U * (1.0 + float(scale))


SEED = st.integers(0, 2 ** 31 - 1)
FAST = settings(max_examples=2000, deadline=None)                 # pure numeric
MID = settings(max_examples=300, deadline=None)                   # small torch computations
SLOW = settings(max_examples=40, deadline=None,                   # stock transformers modules / models
                suppress_health_check=[HealthCheck.too_slow])
FLOAT = lambda lo, hi: st.floats(lo, hi, allow_nan=False, allow_infinity=False)


def rng(seed):
    return np.random.default_rng(seed)


def gen(seed):
    return torch.Generator().manual_seed(seed)


# =====================================================================================================
# Capacity.lean
# =====================================================================================================

import types as _types                                    # noqa: E402

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from elephantine import emulate as em                     # noqa: E402
from elephantine import hidden_width as hw                # noqa: E402
from elephantine import universal as uni                  # noqa: E402


def _w1_model(mt, kw, seed):
    """A tiny stock transformers model in float64 with random weights (norm gains and biases too)."""
    cfg = AutoConfig.for_model(mt, **kw)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.add_(torch.randn_like(p) * (0.3 if p.dim() == 1 else 0.08))
    return cfg, m


def _w1_f64_norms(model):
    """transformers' RMSNorm computes in float32 even in a float64 model; run it in float64 so the
    comparison sees the construction, not that cast (as elephantine/hidden_width.verify does)."""
    for _, mod in model.named_modules():
        if type(mod).__name__.endswith("RMSNorm"):
            def fwd(self, x, eps=mod.variance_epsilon):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * self.weight
            mod.forward = _types.MethodType(fwd, mod)
    return model


def _w1_emulate(tmt, tkw, hmt, hkw, seed):
    """Run elephantine/emulate's own pipeline in memory: plan_construction -> read_target -> build_host,
    then load the host state dict into the host's stock class. Returns (target, host, construction)."""
    tc, T = _w1_model(tmt, tkw, seed)
    traw = tc.to_dict()
    if hmt == tmt:
        hraw = {**traw, **hkw}
    else:
        hraw = AutoConfig.for_model(hmt, **{**{k: traw[k] for k in W1_SHAPE_KEYS if k in traw}, **hkw}).to_dict()
    con = em.plan_construction(traw, hraw)
    R = em.read_target(T, em.resolved(traw), con.tfam)
    hdict = em.host_config_dict(hraw, con)
    sd = em.build_host(R, em.resolved(traw), hdict, con, log=lambda *a: None)
    hc = em.resolved(hdict)
    hc._attn_implementation = "eager"
    Hm = AutoModelForCausalLM.from_config(hc).to(F64).eval()
    missing, unexpected = Hm.load_state_dict({k: v.to(F64) for k, v in sd.items()}, strict=False)
    assert not unexpected and all("rotary" in k or "lm_head" in k for k in missing), (missing, unexpected)
    return T, Hm, con


W1_HEAVY = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])   # model builds


W1_SHAPE_KEYS = ("vocab_size", "max_position_embeddings", "rms_norm_eps", "rope_theta", "tie_word_embeddings")
W1_SMALL = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False,
                rms_norm_eps=1e-2)            # a large eps, so a wrong padded eps is visible (1e-5 would hide it)


def _w1_residuals(model, ids):
    """(after-attention residual, after-block residual) of every decoder layer, and the logits."""
    caps = {}
    hs = []
    for i, L in enumerate(model.model.layers):
        hs.append(L.post_attention_layernorm.register_forward_pre_hook(
            lambda m, a, i=i: caps.__setitem__(("attn", i), a[0])))
        hs.append(L.register_forward_hook(
            lambda m, a, o, i=i: caps.__setitem__(("block", i), o[0] if isinstance(o, tuple) else o)))
    with torch.no_grad():
        logits = model(ids).logits
    for h in hs:
        h.remove()
    return caps, logits


@W1_HEAVY
@given(SEED, st.sampled_from([8, 16, 40]))
def test_hidden_pad_stack(seed, extra):
    """EP.hidden_pad_stack, on elephantine/emulate.build_host (hidden width d -> D = d + e, nothing else
    changed): the host residual is the zero-padded target residual [h_l; 0] after every layer, the host
    eps is eps d/D, and the logits (the head reading the final norm) are the target's."""
    d = W1_SMALL["hidden_size"]
    T, Hm, con = _w1_emulate("llama", W1_SMALL, "llama", dict(hidden_size=d + extra), seed)
    # (emulate v3 computes the ratio in another order: equal to the float rounding)
    assert con.config_changes["rms_norm_eps"] == pytest.approx(W1_SMALL["rms_norm_eps"] * d / (d + extra), rel=1e-12)
    _w1_f64_norms(T), _w1_f64_norms(Hm)
    ids = torch.randint(0, 64, (2, 9), generator=gen(seed))
    ct, lt = _w1_residuals(T, ids)
    ch, lh = _w1_residuals(Hm, ids)
    for key, t in ct.items():
        S = t.abs().max().item()
        # a few hundred float64 roundings per layer (projections over <= 48 terms, softmax, norm)
        assert (ch[key][..., :d] - t).abs().max().item() <= fp(S, 2e3), key
        assert ch[key][..., d:].abs().max().item() == 0.0, key          # the pad stays exactly 0
    assert (lh - lt).abs().max().item() <= fp(lt.abs().max().item(), 2e3)


@W1_HEAVY
@given(SEED, st.integers(1, 3), st.sampled_from([4, 16, 32]))
def test_hidden_pad_blocks(seed, L, extra):
    """EP.hidden_pad_blocks, on elephantine/hidden_width.widen (the 2L-sublayer stack of L attention +
    FFN pre-norm blocks): after EVERY sublayer (attention and FFN) the host residual is [h; 0]."""
    cfg, A = _w1_model("llama", {**W1_SMALL, "num_hidden_layers": L}, seed)
    B = hw.widen(A, cfg.hidden_size + extra)
    B.config._attn_implementation = "eager"       # torch's CPU sdpa is not float64-exact (~1e-8 here)
    assert B.config.rms_norm_eps == cfg.rms_norm_eps * cfg.hidden_size / (cfg.hidden_size + extra)
    _w1_f64_norms(A), _w1_f64_norms(B)
    ids = torch.randint(0, 64, (2, 7), generator=gen(seed))
    ca, la = _w1_residuals(A, ids)
    cb, lb = _w1_residuals(B, ids)
    d = cfg.hidden_size
    assert len(ca) == 2 * L
    for key, t in ca.items():
        assert (cb[key][..., :d] - t).abs().max().item() <= fp(t.abs().max().item(), 2e3), key
        assert cb[key][..., d:].abs().max().item() == 0.0, key
    assert (lb - la).abs().max().item() <= fp(la.abs().max().item(), 2e3)


@MID
@given(SEED, st.integers(2, 40), st.integers(1, 40), FLOAT(1e-3, 1.0))
def test_condition_hidden_pad_needs_eps(seed, d, e, eps):
    """EP.hidden_pad_needs_eps (the converse): a host RMSNorm over the padded width reproduces the target's
    norm only with eps' = eps d/(d+e). hidden_width.widen_config sets exactly that; keeping the target's eps
    (what a host that cannot change its eps does) misses on every input."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    ca = LlamaConfig(hidden_size=d, head_dim=2, num_attention_heads=1, num_key_value_heads=1, rms_norm_eps=eps)
    cb = hw.widen_config(ca, d + e)
    assert cb.rms_norm_eps == eps * d / (d + e)
    g = rng(seed)
    x = torch.tensor(g.standard_normal((3, d)), dtype=F64)
    w = torch.tensor(g.standard_normal(d), dtype=F64)
    xp = torch.cat([x, torch.zeros(3, e, dtype=F64)], -1)
    wp = torch.cat([w * math.sqrt(d / (d + e)), torch.zeros(e, dtype=F64)])
    tn = LlamaRMSNorm(d, eps).to(F64)
    _w1_f64_norms(tn)
    with torch.no_grad():
        tn.weight.copy_(w)
        target = tn(x)
        for eps_h, ok in ((cb.rms_norm_eps, True), (eps, False)):
            hn = LlamaRMSNorm(d + e, eps_h).to(F64)
            _w1_f64_norms(hn)
            hn.weight.copy_(wp)
            err = (hn(xp)[..., :d] - target).abs().max().item()
            if ok:
                assert err <= fp(target.abs().max().item(), 100)
            else:
                # |w_i x_i| |1/sqrt(ms d/D + eps) sqrt(D/d) - 1/sqrt(ms + eps)|, bounded below on these draws
                ms = x.pow(2).mean(-1)
                gap = ((w * x).abs() * ((ms * d / (d + e) + eps).rsqrt() * math.sqrt(d / (d + e))
                                        - (ms + eps).rsqrt()).abs()[:, None]).max().item()
                assert err >= 0.5 * gap > fp(target.abs().max().item(), 100)


def _w1_unorm(n, eps, w, b, centre):
    m = uni.UNorm(n)
    with torch.no_grad():
        m.eps.fill_(eps)
        m.weight.copy_(w)
        m.bias.copy_(b)
        m.centre.fill_(centre)
    return m


@MID
@given(SEED, st.integers(2, 24), st.integers(1, 24), FLOAT(0.0, 1.0), FLOAT(0.1, 10))
def test_layerNorm_pad_center(seed, d, e, eps, scale):
    """EP.layerNorm_pad_center, on elephantine/universal.UNorm (centre = 1 is the repo's LayerNorm): the host
    LayerNorm (weight sqrt(d/(d+e)) gamma, bias [beta; 0], eps d/(d+e)) on the padded CENTRED vector
    [P x; 0] is the padded target LayerNorm. (NO CODE builds the padded LayerNorm host; see BRIDGES.md.)"""
    g = rng(seed)
    x = torch.tensor(g.standard_normal((4, d)) * scale, dtype=F64)
    gam, beta = (torch.tensor(g.standard_normal(d), dtype=F64) for _ in range(2))
    z = torch.zeros(e, dtype=F64)
    tgt = _w1_unorm(d, eps, gam, beta, 1.0)
    host = _w1_unorm(d + e, eps * d / (d + e), torch.cat([gam * math.sqrt(d / (d + e)), z]),
                     torch.cat([beta, z]), 1.0)
    with torch.no_grad():
        xc = x - x.mean(-1, keepdim=True)
        out = host(torch.cat([xc, torch.zeros(4, e, dtype=F64)], -1))
        ref = tgt(x)
    S = ref.abs().max().item()
    assert (out[..., :d] - ref).abs().max().item() <= fp(S, 200)
    assert (out[..., d:]).abs().max().item() == 0.0


@MID
@given(SEED, st.integers(2, 24), st.integers(1, 24))
def test_condition_hidden_pad_layerNorm_needs_centring(seed, d, e):
    """EP.hidden_pad_stack_layerNorm's centring hypothesis: zero-padding an UNCENTRED residual into a host
    LayerNorm (UNorm centre = 1) shifts the mean, so the padded LayerNorm misses the target's."""
    g = rng(seed)
    x = torch.tensor(g.standard_normal((4, d)) + 3.0, dtype=F64)          # a clearly nonzero mean
    gam = torch.tensor(1.0 + 0.1 * g.standard_normal(d), dtype=F64)
    z = torch.zeros(e, dtype=F64)
    tgt = _w1_unorm(d, 1e-5, gam, torch.zeros(d, dtype=F64), 1.0)
    host = _w1_unorm(d + e, 1e-5 * d / (d + e), torch.cat([gam * math.sqrt(d / (d + e)), z]),
                     torch.zeros(d + e, dtype=F64), 1.0)
    with torch.no_grad():
        err = (host(torch.cat([x, torch.zeros(4, e, dtype=F64)], -1))[..., :d] - tgt(x)).abs().max().item()
    assert err > 1e-2


def test_hidden_pad_stack_layerNorm_is_refused_by_inflate():
    """EP.hidden_pad_stack_layerNorm: the repo's only hidden-width inflater for LayerNorm-capable hosts,
    elephantine/push_hub_demo.inflate, REFUSES a LayerNorm (centre != 0) target rather than centring the
    writers (NO CODE builds the centred padded LayerNorm host)."""
    from elephantine import push_hub_demo as phd
    u = uni.UniversalElephantine(uni.UConfig(vocab=8, d=4, L=1))
    with torch.no_grad():
        u.norm.centre.fill_(1.0)
    with pytest.raises(NotImplementedError):
        phd.inflate(u, d=6)


def _w1_ffn_host(seed, extra):
    T, Hm, con = _w1_emulate("llama", W1_SMALL, "llama", dict(intermediate_size=W1_SMALL["intermediate_size"] + extra),
                             seed)
    return T.model.layers[0].mlp, Hm.model.layers[0].mlp


def _w1_mlp_eq(tm, hm, seed, n=200):
    x = torch.tensor(rng(seed).standard_normal((5, W1_SMALL["hidden_size"])), dtype=F64)
    with torch.no_grad():
        a, b = tm(x), hm(x)
    # one GLU FFN: <= 2 x 80-term dot products per unit, then an 80-term down sum
    return (a - b).abs().max().item() <= fp(a.abs().max().item(), n)


@W1_HEAVY
@given(SEED, st.integers(1, 40))
def test_ffn_extra_units(seed, extra):
    """EP.ffn_extra_units, on elephantine/emulate.build_host (host intermediate_size > target's): the extra
    units' down-projection columns are 0, and the host FFN equals the target FFN."""
    tm, hm = _w1_ffn_host(seed, extra)
    It = W1_SMALL["intermediate_size"]
    assert hm.down_proj.weight[:, It:].abs().max().item() == 0.0
    assert _w1_mlp_eq(tm, hm, seed)


@W1_HEAVY
@given(SEED, st.integers(1, 40))
def test_ffn_extra_zero_input(seed, extra):
    """EP.ffn_extra_zero_input, on emulate.build_host: the extra units' input rows are 0, so they output
    act(0) = SiLU(0) * 0 = 0 and their down-projection is arbitrary -- randomised here, still exact."""
    tm, hm = _w1_ffn_host(seed, extra)
    It = W1_SMALL["intermediate_size"]
    assert hm.gate_proj.weight[It:].abs().max().item() == 0.0 and hm.up_proj.weight[It:].abs().max().item() == 0.0
    with torch.no_grad():
        hm.down_proj.weight[:, It:] = torch.randn(hm.down_proj.weight[:, It:].shape, dtype=F64, generator=gen(seed))
    assert _w1_mlp_eq(tm, hm, seed)


@W1_HEAVY
@given(SEED, st.integers(1, 40))
def test_glu_zero_up_unit(seed, extra):
    """EP.glu_zero_up_unit, on emulate.build_host's extra SwiGLU units: with the up row 0 the unit is
    SiLU(g.x) * 0 = 0 whatever the gate row -- gate rows and down columns randomised, still exact."""
    tm, hm = _w1_ffn_host(seed, extra)
    It = W1_SMALL["intermediate_size"]
    g = gen(seed)
    with torch.no_grad():
        hm.gate_proj.weight[It:] = torch.randn(hm.gate_proj.weight[It:].shape, dtype=F64, generator=g)
        hm.down_proj.weight[:, It:] = torch.randn(hm.down_proj.weight[:, It:].shape, dtype=F64, generator=g)
    assert _w1_mlp_eq(tm, hm, seed)


@W1_HEAVY
@given(SEED, st.integers(1, 40))
def test_glu_zero_unit(seed, extra):
    """EP.glu_zero_unit, on emulate.build_host as built (both input rows 0): every extra unit's activation
    SiLU(0) * 0 is exactly 0."""
    _, hm = _w1_ffn_host(seed, extra)
    It = W1_SMALL["intermediate_size"]
    x = torch.tensor(rng(seed).standard_normal((5, W1_SMALL["hidden_size"])), dtype=F64)
    with torch.no_grad():
        units = hm.act_fn(hm.gate_proj(x)) * hm.up_proj(x)
    assert units[:, It:].abs().max().item() == 0.0


@SLOW
@given(SEED, st.sampled_from(["c_relu", "c_relu2", "c_gelu", "c_xielu"]), st.integers(2, 8))
def test_plain_zero_unit(seed, act, d):
    """EP.plain_zero_unit, on elephantine/universal.UFFN's ungated activations (the shared / dense path):
    a zero-input ungated unit outputs the constant phi(0). ReLU, relu^2 and GeLU have phi(0) = 0; xIELU
    does not (phi(0) = alpha_n expm1(eps) != 0), so only a zero down-projection makes its extra unit
    vanish -- which is what universal's zero-initialised extra units (push_hub_demo.inflate) have."""
    c = uni.UConfig(vocab=4, d=d, L=1, shared_ff=3)
    f = uni.UFFN(c)
    with torch.no_grad():
        getattr(f, act).fill_(1.0)
        if act == "c_xielu":
            f.xielu_a.copy_(torch.tensor(rng(seed).standard_normal(2), dtype=F64))
    x = torch.tensor(rng(seed).standard_normal((3, d)), dtype=F64)
    with torch.no_grad():
        units = f._act(f.s_gate(x), f.s_up(x))                 # s_up = 0: every unit has zero input
        phi0 = f._act(torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64)).item()
    assert torch.equal(units, torch.full_like(units, phi0))
    assert (phi0 == 0.0) == (act != "c_xielu")
    with torch.no_grad():
        assert f(x[None]).abs().max().item() == 0.0                 # zero down-projection: the FFN writes 0


def _w1_block_out(block, x):
    with torch.no_grad():
        o = block(x[None])
    return (o[0] if isinstance(o, tuple) else o)[0]


@W1_HEAVY
@given(SEED, st.booleans(), st.integers(1, 16))
def test_moe_wider_experts(seed, norm, extra):
    """EP.moe_wider_experts, on serving/embed.embed_state (Qwen3-MoE target -> Qwen3-MoE host with wider
    experts, same experts / top-k / hidden): every expert's extra units have zero down columns, and the
    host MoE block equals the target's for the target's own (live, input-dependent) router."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from serving.embed import embed_state, plan
    base = dict(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8, intermediate_size=48,
                moe_intermediate_size=16, num_experts=8, num_experts_per_tok=2, num_hidden_layers=1,
                vocab_size=50, norm_topk_prob=norm, max_position_embeddings=64, tie_word_embeddings=False)
    tcfg = Qwen3MoeConfig(**base)
    torch.manual_seed(seed)
    tgt = Qwen3MoeForCausalLM(tcfg).double().eval()
    hcfg = Qwen3MoeConfig(**{**base, "moe_intermediate_size": 16 + extra})
    p = plan(tcfg, hcfg)
    assert p["ok"] and not p["free"], p["why"]
    for k, v in p["config"].items():
        if v is not None:
            setattr(hcfg, k, v)
    host = Qwen3MoeForCausalLM(hcfg).double().eval()
    host.load_state_dict(embed_state(tgt.state_dict(), tcfg, hcfg), strict=False)
    tb, hb = tgt.model.layers[0].mlp, host.model.layers[0].mlp
    for e in hb.experts:
        assert e.down_proj.weight[:, 16:].abs().max().item() == 0.0
    x = torch.tensor(rng(seed).standard_normal((9, 32)), dtype=F64)
    a, b = _w1_block_out(tb, x), _w1_block_out(hb, x)
    # identical router logits and float32 routing weights on both sides; the expert sums differ only by
    # float64 rounding over <= 32 extra zero terms
    assert (a - b).abs().max().item() <= fp(a.abs().max().item(), 500)


@W1_HEAVY
@given(SEED, st.sampled_from([9, 12, 16]))
def test_moe_extra_experts(seed, E_h):
    """EP.moe_extra_experts, on serving/embed.embed_state (Qwen3-MoE target with E = 8 experts -> host with
    E_h > 8, norm_topk_prob = True): the extra experts are zero AND gated 0 on every token (their router rows
    read -M on the free dim p, so they rank below every real expert and the renormalised top-k never picks
    them), and the host block equals the target's. Checked on host inputs [x; p; 0] for any p > 0, which is
    all the free dim ever carries (1 / rms after the post-attention norm). (norm_topk_prob = False is
    exact only by float32 underflow: MoERouting's moeR_dup_plain_extras_ne.)"""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from serving.embed import embed_state, plan
    base = dict(num_attention_heads=4, num_key_value_heads=2, head_dim=8, intermediate_size=48,
                moe_intermediate_size=16, num_experts_per_tok=2, num_hidden_layers=1, vocab_size=50,
                norm_topk_prob=True, max_position_embeddings=64, tie_word_embeddings=False, rms_norm_eps=1e-6)
    tcfg = Qwen3MoeConfig(hidden_size=32, num_experts=8, **base)
    torch.manual_seed(seed)
    tgt = Qwen3MoeForCausalLM(tcfg).double().eval()
    hcfg = Qwen3MoeConfig(hidden_size=48, num_experts=E_h, **base)
    p = plan(tcfg, hcfg)
    assert p["ok"] and p["free"], p["why"]
    for k, v in p["config"].items():
        if v is not None:
            setattr(hcfg, k, v)
    host = Qwen3MoeForCausalLM(hcfg).double().eval()
    host.load_state_dict(embed_state(tgt.state_dict(), tcfg, hcfg), strict=False)
    tb, hb = tgt.model.layers[0].mlp, host.model.layers[0].mlp
    g = rng(seed)
    x = torch.tensor(g.standard_normal((9, 32)), dtype=F64)
    pv = torch.tensor(g.uniform(0.05, 20.0, (9, 1)), dtype=F64)
    xh = torch.cat([x, pv, torch.zeros(9, 48 - 33, dtype=F64)], -1)
    with torch.no_grad():
        top = hb.gate(xh).topk(2, -1).indices
    assert int(top.max()) < 8                                       # the extras are never selected
    for e in range(8, E_h):
        assert all(w.weight.abs().max().item() == 0.0 for w in (hb.experts[e].gate_proj, hb.experts[e].up_proj,
                                                                 hb.experts[e].down_proj))
    a, b = _w1_block_out(tb, x), _w1_block_out(hb, xh)
    # router logits agree up to float64 rounding of a 48- vs 32-term sum; transformers rounds the routing
    # weights to float32, so a one-ulp float32 difference in a gate is possible: slack 4 U32 |out|
    assert (a - b[:, :32]).abs().max().item() <= 4 * U32 * a.abs().max().item() + fp(a.abs().max().item(), 500)
    assert b[:, 32:].abs().max().item() == 0.0


W1_Q2 = dict(hidden_size=48, num_attention_heads=6, num_key_value_heads=6, num_hidden_layers=2,
             num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[])


def _w1_q2_host(seed, moe_ff, shared_ff):
    T, Hm, con = _w1_emulate("llama", W1_SMALL, "qwen2_moe",
                             dict(W1_Q2, moe_intermediate_size=moe_ff, shared_expert_intermediate_size=shared_ff),
                             seed)
    return T, Hm, con


def _w1_pad(x, D):
    return torch.cat([x, torch.zeros(*x.shape[:-1], D - x.shape[-1], dtype=F64)], -1)


@W1_HEAVY
@given(SEED)
def test_moe_sub_moe_shared(seed):
    """EP.moe_sub_moe_shared, on elephantine/emulate.build_host's 'tied' mode for a Qwen2-MoE host (the
    dense target in the routed experts, the shared expert unused): the shared expert's down-projection is 0,
    so the host block is the routed MoE WHATEVER the shared gate and the shared gate / up weights -- all
    three randomised here -- and equals the target FFN."""
    T, Hm, con = _w1_q2_host(seed, moe_ff=64, shared_ff=16)
    assert list(con.ffn_mode.values()) == ["tied", "tied"]           # (emulate v3: host layer -> mode)
    tb, hb = T.model.layers[0].mlp, Hm.model.layers[0].mlp
    assert hb.shared_expert.down_proj.weight.abs().max().item() == 0.0
    g = gen(seed)
    with torch.no_grad():
        for w in (hb.shared_expert.gate_proj.weight, hb.shared_expert.up_proj.weight, hb.shared_expert_gate.weight):
            w.copy_(torch.randn(w.shape, dtype=F64, generator=g))
    x = torch.tensor(rng(seed).standard_normal((7, 32)), dtype=F64)
    with torch.no_grad():
        a = tb(x)
    b = _w1_block_out(hb, _w1_pad(x, 48))
    # routing weights 1/E * (E/top_k) from a zero router: exact in float32 for E = 4, k = 2
    assert (a - b[:, :32]).abs().max().item() <= fp(a.abs().max().item(), 500)
    assert b[:, 32:].abs().max().item() == 0.0


@W1_HEAVY
@given(SEED)
def test_moe_sub_moe_shared_gated(seed):
    """EP.moe_sub_moe_shared_gated, on emulate.build_host's 'tied' mode as built: the sigmoid shared gate's
    input weights are 0 (gate sigma(0) = 1/2 on every token) and the shared down-projection is 0, so the
    shared term is exactly 0 and the host block is the routed MoE (= the target FFN)."""
    T, Hm, _ = _w1_q2_host(seed, moe_ff=64, shared_ff=16)
    tb, hb = T.model.layers[0].mlp, Hm.model.layers[0].mlp
    x = torch.tensor(rng(seed).standard_normal((7, 32)), dtype=F64)
    xh = _w1_pad(x, 48)
    with torch.no_grad():
        assert torch.equal(torch.sigmoid(hb.shared_expert_gate(xh)), torch.full((7, 1), 0.5, dtype=F64))
        assert hb.shared_expert(xh).abs().max().item() == 0.0
        a = tb(x)
    b = _w1_block_out(hb, xh)
    assert (a - b[:, :32]).abs().max().item() <= fp(a.abs().max().item(), 500)


@W1_HEAVY
@given(SEED)
def test_shared_gate_half(seed):
    """EP.shared_gate_half, on emulate.build_host's 'shared' mode (the dense target in a Qwen2-MoE host's
    shared expert, routed experts zeroed): the zeroed shared_expert_gate gives sigma(0) = 1/2 exactly
    (never 0), and the shared down-projection doubled (x 2) undoes it: the block equals the target FFN."""
    T, Hm, con = _w1_q2_host(seed, moe_ff=16, shared_ff=64)
    assert list(con.ffn_mode.values()) == ["shared", "shared"]       # (emulate v3: host layer -> mode)
    tb, hb = T.model.layers[0].mlp, Hm.model.layers[0].mlp
    x = torch.tensor(rng(seed).standard_normal((7, 32)), dtype=F64)
    xh = _w1_pad(x, 48)
    with torch.no_grad():
        gate = torch.sigmoid(hb.shared_expert_gate(xh))
        a = tb(x)
    assert torch.equal(gate, torch.full_like(gate, 0.5)) and bool((gate > 0).all())
    b = _w1_block_out(hb, xh)
    assert (a - b[:, :32]).abs().max().item() <= fp(a.abs().max().item(), 500)


# =====================================================================================================
# CapacityDepth.lean
# =====================================================================================================

from block import parallel_residual as w1_pr              # noqa: E402
from moe import dense_emulates_moe as w1_dem              # noqa: E402


@FAST
@given(SEED, st.integers(2, 6), st.integers(1, 5), FLOAT(1e-3, 2.0), FLOAT(1.0, 4.0))
def test_moe_two_layer_margin(seed, E, ew, delta, slack):
    """EP.moe_two_layer_margin, on moe/dense_emulates_moe.dense_two_layer_margin (helper): on a router
    margin delta and with lam delta >= every loser's pre-activation, the select FFN + masked expert FFN
    give exactly the winner's ReLU FFN, and the select channels are erased exactly (relu(s) = s)."""
    g = rng(seed)
    z = g.standard_normal(E) * 3
    estar = int(z.argmax())
    z[np.arange(E) != estar] = np.minimum(z[np.arange(E) != estar], z[estar] - delta)   # enforce the margin
    P = g.standard_normal((E, ew)) * 3
    V = g.standard_normal((E, ew, 4))
    lam = slack * max(P.max(), 0.0) / delta
    zt, Pt, Vt = (torch.tensor(a, dtype=F64) for a in (z, P, V))
    out, rest = w1_dem.dense_two_layer_margin(zt, Pt, Vt, lam)
    ref = torch.einsum("j,jd->d", F.relu(Pt[estar]), Vt[estar])
    # s_e is a sum of E terms; lam s_e - p_ej >= (slack - 1) p_ej >= 0 is decided with ~E roundings of
    # margin, and the winner's pre-activations pass through exactly (s = 0 exactly for the winner)
    assert (out - ref).abs().max().item() <= fp(ref.abs().max().item() + float(np.abs(V).max() * np.abs(P).max()), 50)
    assert rest.abs().max().item() == 0.0


@FAST
@given(SEED, st.integers(2, 6))
def test_condition_moe_two_layer_margin_needs_margin(seed, E):
    """EP.moe_two_layer_margin's margin hypothesis: at a router tie no loser is masked (s = 0 for both tied
    experts) and the two-layer host returns the SUM of the tied experts, not a top-1 pick."""
    g = rng(seed)
    z = torch.tensor(g.standard_normal(E), dtype=F64)
    z[1] = z[0] = z.max() + 1.0
    P = torch.tensor(np.abs(g.standard_normal((E, 3))) + 0.5, dtype=F64)
    V = torch.tensor(g.standard_normal((E, 3, 4)), dtype=F64)
    out, _ = w1_dem.dense_two_layer_margin(z, P, V, lam=1e6)
    both = torch.einsum("ej,ejd->d", F.relu(P[:2]), V[:2])
    assert (out - both).abs().max().item() <= fp(both.abs().max().item(), 50)
    assert min((out - torch.einsum("j,jd->d", P[i], V[i])).abs().max().item() for i in (0, 1)) > 1e-6


@W1_HEAVY
@given(SEED, st.integers(1, 2), st.integers(0, 2), st.integers(1, 3))
def test_untied_sub_looped_periodic(seed, b, extra, K):
    """EP.untied_sub_looped_periodic: a looped host of b' = b + extra layers -- the target's period-b block
    plus identity layers (elephantine/hidden_width.build_caps's extra depth: o_proj and down_proj zeroed) --
    looped K times equals the untied K b-layer target whose layer t is block layer t mod b (the periodic
    stack hf/looped_untied.build ties), on every input. (extra = 0 is looped_untied's own check.)"""
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaModel
    cfg = LlamaConfig(hidden_size=32, num_hidden_layers=b, num_attention_heads=4, num_key_value_heads=4,
                      intermediate_size=48, vocab_size=16, max_position_embeddings=64, rms_norm_eps=1e-6,
                      attn_implementation="eager")
    torch.manual_seed(seed)
    A = LlamaForCausalLM(cfg).to(F64).eval()
    host = hw.build_caps(A, layers=b + extra).eval()                 # the looped block: b' = b + extra layers
    deep = LlamaModel(LlamaConfig(**{**cfg.to_dict(), "num_hidden_layers": K * b})).to(F64).eval()
    host.config._attn_implementation = deep.config._attn_implementation = "eager"
    for t in range(K * b):                                          # the periodic untied target
        deep.layers[t].load_state_dict(A.model.layers[t % b].state_dict())
    _w1_f64_norms(host), _w1_f64_norms(deep)
    seq = 6
    x = torch.randn(1, seq, 32, dtype=F64, generator=gen(seed))
    pos = torch.arange(seq)[None]
    pe = deep.rotary_emb(x, pos)
    mask = torch.triu(torch.full((seq, seq), torch.finfo(F64).min, dtype=F64), 1).view(1, 1, seq, seq)

    def run(layers, h, times):
        for _ in range(times):
            for L in layers:
                o = L(h, position_embeddings=pe, attention_mask=mask)
                h = o[0] if isinstance(o, tuple) else o
        return h

    with torch.no_grad():
        looped = run(host.model.layers, x, K)
        untied = run(deep.layers, x, 1)
    # identity layers add exactly 0 (zero o_proj / down_proj); the rest is the same computation
    assert (looped - untied).abs().max().item() <= fp(untied.abs().max().item(), 100)


@FAST
@given(FLOAT(-1e6, 1e6))
def test_relu_identity_pair(t):
    """EP.relu_identity_pair: relu t - relu(-t) = t, exactly, as parallel_stack_in_seq_2x's ReLU identity
    units compute it (torch F.relu)."""
    x = torch.tensor([t], dtype=F64)
    assert (F.relu(x) - F.relu(-x)).item() == t


@FAST
@given(FLOAT(-50, 50))
def test_silu_identity_pair(t):
    """EP.silu_identity_pair: silu t - silu(-t) = t (sigma t + sigma(-t)) = t, with torch's F.silu (the
    identity units of parallel_stack_in_seq_2x with phi = SiLU): a few roundings of |t|."""
    x = torch.tensor([t], dtype=F64)
    assert abs((F.silu(x) - F.silu(-x)).item() - t) <= fp(abs(t), 8)


@MID
@given(SEED, st.integers(1, 5), st.sampled_from(["relu", "silu"]))
def test_parallel_sub_seq_2x(seed, L, act):
    """EP.parallel_sub_seq_2x, on block/parallel_residual.parallel_stack_in_seq_2x (helper): L parallel
    blocks run in L sequential blocks of width 2d, norm-free; after every block l the host residual is
    (target_l, 0), with 2d identity units phi(v) - phi(-v) = v (ReLU exact, SiLU up to rounding) moving the
    attention write into the stream."""
    g = gen(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    T, d, m = 5, 8, 12
    x = r(T, d)
    As, Ms = [], []
    for _ in range(L):
        n1, n2, Wa, W1, W2 = r(d), r(d), r(d, d) / 3, r(d, m) / 3, r(m, d) / 3
        As.append(lambda z, n1=n1, Wa=Wa: w1_pr._rms(z, n1) @ Wa)                  # arbitrary maps of c0
        Ms.append(lambda z, n2=n2, W1=W1, W2=W2: F.silu(w1_pr._rms(z, n2) @ W1) @ W2)
    phi = F.relu if act == "relu" else F.silu
    host = w1_pr.parallel_stack_in_seq_2x(x, As, Ms, phi)
    t = x
    for l, (A, M) in enumerate(zip(As, Ms)):
        t = w1_pr.parallel_stack(t, [A], [M])
        c0, c1 = host[l]
        S = t.abs().max().item()
        # SiLU identity units: each block adds a few roundings of |c1| <= S; ReLU: exactly c1
        assert (c0 - t).abs().max().item() <= fp(S, 200 * (l + 1)), l
        if act == "relu":
            assert c1.abs().max().item() == 0.0
        else:
            assert c1.abs().max().item() <= fp(S, 20)


@MID
@given(SEED, st.integers(2, 5))
def test_parallel_residual_width_claim(seed, L):
    """Was MISMATCH (W1-1) against EP.parallel_sub_seq_2x: block/parallel_residual and block/prenorm_carry
    said a norm-free L-block parallel model needs (L+1) d sequential width. Their docstrings now say 2d at
    every L, norm-free, with 2d extra MLP units per block (asserted). The theorem's 2d host (helper
    parallel_stack_in_seq_2x) is exact for every L >= 2, at half the (L+1)d of parallel_stack_in_seq_wide
    for L = 3."""
    from block import prenorm_carry
    assert "Width 2d is enough for L = 1" not in w1_pr.parallel_stack_in_seq_wide.__doc__
    assert "an L-block parallel\n  model needs (L+1) d" not in w1_pr.__doc__
    assert "needs (L+1) d. The MLP readout" not in prenorm_carry.__doc__
    for doc in (w1_pr.__doc__, prenorm_carry.__doc__):
        assert "2d extra MLP units per block" in doc and "parallel_sub_seq_2x" in doc
    g = gen(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)
    x = r(4, 6)
    As = [lambda z, W=r(6, 6) / 3: torch.tanh(z @ W) for _ in range(L)]
    Ms = [lambda z, W=r(6, 6) / 3: F.silu(z @ W) for _ in range(L)]
    target = w1_pr.parallel_stack(x, As, Ms)
    c0, c1 = w1_pr.parallel_stack_in_seq_2x(x, As, Ms, F.relu)[-1]
    assert torch.cat([c0, c1], -1).shape[-1] == 2 * x.shape[-1]
    assert (c0 - target).abs().max().item() <= fp(target.abs().max().item(), 200 * L) and c1.abs().max() == 0


# ===== AuditRules.lean =====
#
# Most of the AuditRules switches and folds are built by elephantine/emulate.py (a target's weights
# rewritten into a stock host checkpoint) or are weight settings of the universal host
# (elephantine/universal.py). The emulate tests build tiny random models in memory, run the repo's own
# plan_construction / read_target / build_host, load the state dict into the host's stock class, and
# compare logits in float64 with the stock fp32 casts lifted (tests.test_universal_elephantine._lift_fp32),
# so the remaining gap is float64 rounding only.

from attn import attention_sink as asink         # noqa: E402
from elephantine import emulate as em             # noqa: E402
from elephantine import universal as uni          # noqa: E402

UNI = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
EMU = settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.too_slow])
_E_SMALL = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
_E_BIG = dict(vocab_size=64, hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=6,
              num_key_value_heads=3, head_dim=8, max_position_embeddings=64)
_GRANITE = dict(embedding_multiplier=3.0, residual_multiplier=0.3, attention_multiplier=0.11, logits_scaling=2.5)


def _lift():
    from tests.test_universal_elephantine import _lift_fp32
    return _lift_fp32()


def _hf_cfg(mt, kw):
    from transformers import AutoConfig, AutoModelForCausalLM
    c = AutoConfig.for_model(mt, **kw)
    c.architectures = [AutoModelForCausalLM._model_mapping[type(c)].__name__]
    return c


def _rand_model(mt, kw, seed, scale=0.08):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(_hf_cfg(mt, kw), attn_implementation="eager").to(F64).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * scale)
    return m


def _emulate(tmt, tkw, hmt, hkw, seed):
    """elephantine/emulate.py's construction, in memory: (target model, the plan, the host state dict,
    the host model with that state dict, max |logit difference|, max |target logit|)."""
    from transformers import AutoModelForCausalLM
    tm = _rand_model(tmt, tkw, seed)
    traw, hraw = tm.config.to_dict(), _hf_cfg(hmt, hkw).to_dict()
    con = em.plan_construction(traw, hraw)
    tcr = em.resolved(traw)
    T = em.read_target(tm, tcr, con.tfam)
    hdict = em.host_config_dict(hraw, con)
    sd = em.build_host(T, tcr, hdict, con, log=lambda *a: None)
    hm = AutoModelForCausalLM.from_config(em.resolved(hdict), attn_implementation="eager").to(F64).eval()
    info = hm.load_state_dict({k: v.to(F64) for k, v in sd.items()}, strict=False)
    assert not info.unexpected_keys and set(info.missing_keys) <= {"lm_head.weight"}, info
    ids = torch.randint(0, tm.config.vocab_size, (2, 7), generator=gen(seed))
    with torch.no_grad(), _lift():
        a, b = tm(ids).logits, hm(ids).logits
    return dict(tm=tm, con=con, hdict=hdict, sd=sd, hm=hm, err=(a - b).abs().max().item(), scale=a.abs().max().item())


# Every emulate comparison: two stacks of ~10^3-10^4 float64 operations per logit on O(1) values, so a
# few 1e-13 is a generous rounding allowance (measured: ~1e-15).
EMU_TOL = 1e-12


@EMU
@given(SEED)
def test_bias_off(seed):
    """EP.bias_off: W x + 0 = W x. emulate.build_host writes a bias-free target into a host with q / k / v /
    o and MLP biases (Llama attention_bias = mlp_bias = True) by leaving every host bias at 0: the host
    biases are all exactly 0 and the logits equal the target's."""
    r = _emulate("llama", _E_SMALL, "llama", {**_E_BIG, "attention_bias": True, "mlp_bias": True}, seed)
    biases = {k: v for k, v in r["sd"].items() if k.endswith(".bias")}
    assert biases and all(float(v.abs().max()) == 0.0 for v in biases.values())
    assert r["err"] <= EMU_TOL


def _uattn(seed, d=12, H=2, hd=4, T=6):
    c = uni.UConfig(vocab=8, d=d, L=1, heads=H, head_dim=hd)
    a = uni.UAttention(c)
    g = gen(seed)
    with torch.no_grad():
        for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, a.g_proj):
            lin.weight.copy_(torch.randn(lin.weight.shape, dtype=F64, generator=g) * 0.5)
            lin.bias.copy_(torch.randn(lin.bias.shape, dtype=F64, generator=g) * 0.1)
    x = torch.randn(1, T, d, dtype=F64, generator=g)
    return a, x


@UNI
@given(SEED)
def test_out_gate_off(seed):
    """EP.out_gate_off: (2 W_o)(o ⊙ σ(0·x)) = W_o o. The universal host's output gate
    (UAttention.gate_coef = [0, 1]: o * sigmoid(g_proj x)) with g_proj = 0 and o_proj doubled equals the
    ungated head (gate_coef = [1, 0]) with the original o_proj. A weight setting of UAttention: the plain
    loaders switch the gate off with gate_coef instead, so no loader writes this one."""
    a, x = _uattn(seed)
    with torch.no_grad():
        plain = a(x, causal=True)
        a.gate_coef.copy_(torch.tensor([0.0, 1.0]))
        a.g_proj.weight.zero_(); a.g_proj.bias.zero_()
        a.o_proj.weight.mul_(2.0)
        o_bias = a.o_proj.bias.clone()
        a.o_proj.bias.zero_()                       # the theorem has no o bias: o_proj(o) = W_o o + b_o
        gated = a(x, causal=True) + o_bias
    # T·d·H·hd ~ 300 products of O(1) terms; σ(0) = 1/2 and ×2 are exact in binary
    assert (gated - plain).abs().max().item() <= fp(plain.abs().max(), 1e3)


@EMU
@given(SEED)
def test_moe_shared_gate_off(seed):
    """EP.moe_shared_gate_off (and EP.ffn_smul): moe + σ(0·x) · ffn(2 W2) = moe + ffn(W2). emulate.build_host
    puts a dense target FFN into a Qwen2-MoE host's shared expert: shared_expert_gate.weight = 0 (σ = 1/2
    on every token) and the shared down-projection doubled; the routed experts write 0. Logits equal the
    target's."""
    r = _emulate("llama", _E_SMALL, "qwen2_moe", {**_E_BIG, "moe_intermediate_size": 16,
                                                   "shared_expert_intermediate_size": 96,
                                                   "num_experts": 2, "num_experts_per_tok": 1}, seed)
    assert set(r["con"].ffn_mode.values()) == {"shared"}              # (emulate v3: host layer -> mode)
    sd, T = r["sd"], r["tm"]
    d = T.config.hidden_size
    for j, i in enumerate(r["con"].layer_pos):
        p = f"model.layers.{i}.mlp."
        assert float(sd[p + "shared_expert_gate.weight"].abs().max()) == 0.0
        tdown = T.model.layers[j].mlp.down_proj.weight.detach()
        hdown = sd[p + "shared_expert.down_proj.weight"].to(F64)
        assert torch.equal(hdown[:d, : tdown.shape[1]], 2.0 * tdown)          # x 2 is exact in binary
    assert r["err"] <= EMU_TOL


def _ucfg(**kw):
    return uni.UConfig(**{**dict(vocab=16, d=8, L=2, heads=2, head_dim=4, shared_ff=8), **kw})


_LIVE = ("embed.", "lm_head.", "attn.q_proj.", "attn.k_proj.", "attn.v_proj.", "attn.o_proj.", "ffn.s_gate.",
         "ffn.s_up.", "ffn.s_down.", "ple_", "engram.table", "engram.key", "engram.val", "engram.conv_w")


def _randomise(u, seed, scale=0.3):
    """Random weights on the plain transformer path and the extra-input branches (the other universal
    branches stay at their off settings)."""
    g = gen(seed)
    with torch.no_grad():
        for n, p in u.named_parameters():
            if p.dim() >= 2 and any(t in n for t in _LIVE):
                p.copy_(torch.randn(p.shape, dtype=F64, generator=g) * scale)
    return u


@UNI
@given(SEED)
def test_extra_input_off(seed):
    """EP.extra_input_off: x + 0(u) = x. The universal host's extra per-token inputs (per-layer embeddings,
    ULayer.ple_*; the hashed n-gram memory, UEngram) with their read-out projection at 0 (ple_proj = 0,
    UEngram.val = 0) leave the stack equal to the same weights with the branches off (ple_c = 0, c = 0)."""
    cfg = _ucfg(ple_dim=3, engram_slots=5, engram_dim=2)
    u = _randomise(uni.UniversalElephantine(cfg), seed)
    ids = torch.randint(0, cfg.vocab, (1, 6), generator=gen(seed))
    with torch.no_grad():
        for L in u.layers:
            L.ple_c.fill_(0.0); L.engram.c.fill_(0.0)
        off = u(ids)
        for L in u.layers:
            L.ple_c.fill_(1.0); L.engram.c.fill_(1.0)
            L.ple_proj.weight.zero_(); L.engram.val.weight.zero_()
        on = u(ids)
    # the switched-on branches add exact zeros (0 * finite, then x + 0): bitwise equal
    assert torch.equal(on, off)


@UNI
@given(SEED)
def test_switch_lora_off(seed):
    """EP.switch_lora_off: W x + (if sel then 0 x else 0) = W x. The universal host's switched LoRA
    (lora_adapters, la / lb, the adapter picked per token by the switch head sw_score / sw_val) with zero
    adapters equals the base weights, whichever adapters the random switch head selects."""
    cfg = _ucfg(lora_adapters=2, lora_rank=2)
    u = _randomise(uni.UniversalElephantine(cfg), seed)
    ids = torch.randint(0, cfg.vocab, (1, 6), generator=gen(seed))
    g = gen(seed + 1)
    with torch.no_grad():
        for p in [*[m.la[n] for L in u.layers for m in (L.attn, L.ffn) for n in m.la],
                  *[m.lb[n] for L in u.layers for m in (L.attn, L.ffn) for n in m.lb]]:
            p.zero_()
        u.sw_val.zero_()                              # adapter 0 everywhere: the base path
        base = u(ids)
        u.sw_score.copy_(torch.randn(cfg.vocab, dtype=F64, generator=g))
        u.sw_val.copy_(torch.randint(0, 3, (cfg.vocab,), generator=g).to(F64))
        switched = u(ids)
    assert torch.equal(switched, base)


def _uffn(seed, E=5, k=2, d=6, f=3):
    c = _ucfg(d=d, experts=E, expert_ff=f, top_k=k)
    m = uni.UFFN(c)
    g = gen(seed)
    with torch.no_grad():
        for p in (m.router.weight, m.w_gate, m.w_up, m.w_down):
            p.copy_(torch.randn(p.shape, dtype=F64, generator=g))
        m.renorm.fill_(1.0)
        m.s_down.weight.zero_()                       # the routed part only
    x = torch.randn(1, 4, d, dtype=F64, generator=g)
    return m, x


def _routed_ref(m, x, weights_fn, sel_scores):
    """sum over the selected set of w_e Y_e, the selection on sel_scores (the theorem's wt l (sel l))."""
    Y = torch.einsum("edf,btef->bted", m.w_down,
                     F.silu(torch.einsum("efd,btd->btef", m.w_gate, x)) * torch.einsum("efd,btd->btef", m.w_up, x))
    topi = sel_scores.topk(m.top_k, -1).indices
    w = weights_fn(topi)
    return (Y.gather(2, topi[..., None].expand(*topi.shape, x.shape[-1])) * w[..., None]).sum(2)


@UNI
@given(SEED)
def test_router_bias_off(seed):
    """EP.router_bias_off: biasedRouter sel wt l 0 = wt l (sel l). The universal host's selection bias
    (UFFN.sel_bias, added to the scores for the top-k SELECTION only: e_score_correction_bias) at 0 selects
    on the plain scores; the output equals the plain renormalised softmax top-k MoE computed directly."""
    m, x = _uffn(seed)
    with torch.no_grad():
        m.sel_bias.zero_()
        out = m(x)
        p = m.router(x).softmax(-1)
        ref = _routed_ref(m, x, lambda ti: (lambda v: v / v.sum(-1, keepdim=True))(p.gather(-1, ti)), p)
    assert (out - ref).abs().max().item() <= fp(ref.abs().max(), 1e3)


@SLOW
@given(SEED)
def test_multi_rope_tables_equal(seed):
    """EP.multi_rope_tables_equal: every layer's table set to the target's gives the target's RoPE logit at
    every layer. The universal host carries one inv_freq table PER LAYER (UAttention.inv_freq);
    universal.from_llama loads a one-table Llama target by writing the target's inv_freq into every layer,
    and the host's logits equal the target's."""
    tm = _rand_model("llama", {**_E_SMALL, "num_hidden_layers": 3, "rope_theta": 500.0 + seed % 1000}, seed)
    u = uni.from_llama(tm)
    want = tm.model.rotary_emb.inv_freq.to(F64)
    for L in u.layers:
        assert torch.equal(L.attn.inv_freq.detach(), want)
    ids = torch.randint(0, 64, (1, 7), generator=gen(seed))
    with torch.no_grad(), _lift():
        a, b = tm(ids).logits, u(ids)
    assert (a - b).abs().max().item() <= EMU_TOL


@EMU
@given(SEED)
def test_scalar_fold(seed):
    """EP.scalar_fold: α (W x) = (α W) x. emulate.build_host folds Granite's embedding multiplier into the
    embedding table, its residual multiplier into o_proj / down_proj, and 1/logits_scaling into the
    lm_head of a stock Llama host (no multipliers); the logits equal the target's. Also
    universal.from_granite (_head_from emb_scale / logit_scale, _fold_residual)."""
    r = _emulate("granite", {**_E_SMALL, **_GRANITE}, "llama", _E_BIG, seed)
    tm, sd, con = r["tm"], r["sd"], r["con"]
    d, V, a = tm.config.hidden_size, tm.config.vocab_size, con.a      # a: host stream / target stream
    E = tm.model.embed_tokens.weight.detach()
    assert torch.allclose(sd["model.embed_tokens.weight"].to(F64)[:, :d], 3.0 * a * E, rtol=0,
                          atol=fp(3 * a * E.abs().max(), 4))
    o_t = tm.model.layers[0].self_attn.o_proj.weight.detach()
    o_h = sd[f"model.layers.{con.layer_pos[0]}.self_attn.o_proj.weight"].to(F64)[:d]
    # (emulate v3: qmap target head -> (host head, block); no KV packing here, block 0)
    assert all(blk == 0 for _, blk in con.qmap.values())
    cols = [con.qmap[h][0] * 8 + e for h in range(4) for e in range(8)]
    assert torch.allclose(o_h[:, cols], 0.3 * a * o_t, rtol=0, atol=fp(o_t.abs().max(), 4))
    lm_t = tm.lm_head.weight.detach()
    assert torch.allclose(sd["lm_head.weight"].to(F64)[:, :d], lm_t / 2.5, rtol=0, atol=fp(lm_t.abs().max(), 4))
    assert r["err"] <= EMU_TOL
    u = uni.from_granite(tm)
    ids = torch.randint(0, V, (1, 7), generator=gen(seed))
    with torch.no_grad(), _lift():
        assert (tm(ids).logits - u(ids)).abs().max().item() <= EMU_TOL


@EMU
@given(SEED)
def test_attn_scale_fold(seed):
    """EP.attn_scale_fold: α ((W_q x)·k) = ((α W_q) x)·k. emulate.build_host folds the target's attention
    scale (Granite's attention_multiplier, 0.11) over the host's own 1/sqrt(head_dim) into the host's q rows:
    host W_q = (s_t / s_h) W_q. The host logits equal the target's."""
    r = _emulate("granite", {**_E_SMALL, **_GRANITE}, "llama", _E_BIG, seed)
    tm, sd, con = r["tm"], r["sd"], r["con"]
    alpha = 0.11 / (1 / math.sqrt(8))
    wq_t = tm.model.layers[0].self_attn.q_proj.weight.detach().view(4, 8, -1)
    wq_h = sd[f"model.layers.{con.layer_pos[0]}.self_attn.q_proj.weight"].to(F64).view(6, 8, -1)
    for h, (j, blk) in con.qmap.items():                # (emulate v3: (host head, block))
        assert blk == 0
        assert torch.allclose(wq_h[j, :, : wq_t.shape[-1]], alpha * wq_t[h], rtol=0, atol=fp(wq_t.abs().max(), 4))
    assert r["err"] <= EMU_TOL


@EMU
@given(SEED)
def test_norm_weight_fold(seed):
    """EP.norm_weight_fold: α · RMSNorm(w, x) = RMSNorm(α w, x). A tied Llama target on a TIED Granite host
    (logits_scaling 8, embedding_multiplier 12, residual multiplier 0.22): the host's lm_head is its
    embedding, so emulate.build_host puts the logit and embedding scales into the final norm weight
    (kappa gam c_h L_h / a). The stock host norm with that weight is α times the norm with the target's,
    and the logits equal the target's."""
    hk = dict(embedding_multiplier=12.0, residual_multiplier=0.22, attention_multiplier=0.015625,
              logits_scaling=8.0, tie_word_embeddings=True)
    r = _emulate("llama", {**_E_SMALL, "tie_word_embeddings": True}, "granite", {**_E_BIG, **hk}, seed)
    tm, hm = r["tm"], r["hm"]
    d, D = 32, 48
    a, gam = 0.22, math.sqrt(d / D)
    alpha = 1.0 * 8.0 * gam * 12.0 / a / gam            # kappa c_h L_h / a, kappa = 1: the fold beyond gam
    w_t = tm.model.norm.weight.detach()
    w_h = hm.model.norm.weight.detach()[:d]
    assert torch.allclose(w_h, alpha * gam * w_t, rtol=16 * U, atol=0)
    x = torch.randn(3, D, dtype=F64, generator=gen(seed))
    x[:, d:] = 0
    with _lift():
        ref = hm.model.norm.__class__(D, eps=hm.config.rms_norm_eps).to(F64)
        with torch.no_grad():
            ref.weight.zero_(); ref.weight[:d] = gam * w_t
        lhs, rhs = alpha * ref(x), hm.model.norm(x)
    assert (lhs - rhs).abs().max().item() <= fp(rhs.abs().max(), 16)
    assert r["err"] <= EMU_TOL * 10


@EMU
@given(SEED)
def test_zero_centred_norm(seed):
    """EP.zero_centred_norm: RMSNorm(1 + w) x = (1 + w) ⊙ RMSNorm(1, x). A Gemma target ((1 + w) norms) on a
    standard Llama host: emulate.build_host stores the EFFECTIVE weight 1 + w (times the padding factor
    sqrt(d/D)) in the host norm; the logits equal the target's."""
    r = _emulate("gemma", {**_E_SMALL, "hidden_act": "silu"}, "llama", _E_BIG, seed)
    tm, sd = r["tm"], r["sd"]
    d, gam = 32, math.sqrt(32 / 48)
    w_t = tm.model.layers[0].input_layernorm.weight.detach()
    w_h = sd[f"model.layers.{r['con'].layer_pos[0]}.input_layernorm.weight"].to(F64)
    assert torch.allclose(w_h[:d], (1 + w_t) * gam, rtol=0, atol=fp(2, 4))
    assert r["err"] <= EMU_TOL


@EMU
@given(SEED)
def test_zero_centred_host(seed):
    """EP.zero_centred_host: RMSNorm(1 + (w - 1)) = RMSNorm(w). A standard Llama target on a Gemma host
    ((1 + w) norms): emulate.build_host stores w - 1 (stored = effective - offset, `norm_vec`); the
    padded tail is stored as -1, i.e. effective 0. The logits equal the target's."""
    r = _emulate("llama", _E_SMALL, "gemma", {**_E_BIG, "hidden_act": "silu"}, seed)
    tm, sd = r["tm"], r["sd"]
    d, gam = 32, math.sqrt(32 / 48)
    w_t = tm.model.layers[0].input_layernorm.weight.detach()
    w_h = sd[f"model.layers.{r['con'].layer_pos[0]}.input_layernorm.weight"].to(F64)
    assert torch.allclose(1 + w_h[:d], w_t * gam, rtol=0, atol=fp(2, 4))
    assert torch.equal(w_h[d:], torch.full((48 - d,), -1.0, dtype=F64))
    assert r["err"] <= EMU_TOL


@FAST
@given(SEED, st.integers(1, 12), FLOAT(1e-8, 1e-1), FLOAT(0.05, 20.0))
def test_rmsNorm_smul_eps(seed, n, eps, lam):
    """EP.rmsNorm_smul_eps: RMSNorm_eps(λ x) = RMSNorm_{eps/λ²}(x), λ > 0, on the universal host's UNorm."""
    x = torch.as_tensor(rng(seed).standard_normal(n))
    w = torch.as_tensor(rng(seed + 1).standard_normal(n))
    a, b = uni.UNorm(n), uni.UNorm(n)
    with torch.no_grad():
        a.weight.copy_(w); b.weight.copy_(w)
        a.eps.fill_(eps); b.eps.fill_(eps / lam ** 2)
        lhs, rhs = a(lam * x), b(x)
    # each side: a mean of n squares, a sqrt and a product; the eps / λ² division is one more rounding
    assert (lhs - rhs).abs().max().item() <= fp(rhs.abs().max(), 8 * n + 16)


@EMU
@given(SEED)
def test_rmsNorm_smul_eps_in_emulate(seed):
    """EP.rmsNorm_smul_eps, as emulate uses it: a Granite host with residual multiplier r_h carries the
    target's stream times a = r_h (zero-padded), so plan_construction sets the host's
    rms_norm_eps = a² eps d/D (the stream scalar moved into epsilon, with the padding's d/D). Logits equal."""
    hk = dict(embedding_multiplier=12.0, residual_multiplier=0.22, attention_multiplier=0.015625,
              logits_scaling=8.0)
    r = _emulate("llama", _E_SMALL, "granite", {**_E_BIG, **hk}, seed)
    eps_t = r["tm"].config.rms_norm_eps
    assert r["con"].a == pytest.approx(0.22, rel=1e-15)
    assert r["hdict"]["rms_norm_eps"] == pytest.approx(0.22 ** 2 * eps_t * 32 / 48, rel=1e-15)
    assert r["err"] <= EMU_TOL * 10


@FAST
@given(SEED, st.integers(2, 64), FLOAT(0.05, 0.95), st.booleans())
def test_mismatch_approx_layer_scalar_keeps_eps(seed, n, C, up):
    """EP.rmsNorm_smul_eps / EP.stream_scalar_needs_eps vs approx/plan.py's "layer_scalar ... EXACT".
    The Gemma-4 -> gpt-oss converter (approx/convert.Converter.run) carries the student residual as the
    teacher residual divided by the cumulative layer scalar C_l and calls the fold exact because "all
    sub-blocks start with a scale-invariant RMSNorm". RMSNorm with eps > 0 is not scale invariant, and the
    student keeps the teacher's eps (approx/real.py: s.eps = tc.eps). approx.arch.rmsnorm(h / C, w, eps)
    misses rmsnorm(h, w, eps) by the factor sqrt((ms + eps) / (ms + eps C²)); the fold is exact only at
    eps / C² (asserted too). Fixed (W2-1): approx.convert.eps_plan gives each layer its eps / C² through a
    constant padded channel (asserted below on one padded dim); tests/test_approx.py checks the converter."""
    from approx.arch import rmsnorm as arms
    C = C if up else 1.0 / C
    eps = 1e-6
    h = torch.as_tensor(rng(seed).standard_normal(n)) * 1e-2            # rms ~ 1e-2: eps / ms ~ 1e-2
    w = torch.as_tensor(rng(seed + 1).standard_normal(n))
    ms = h.pow(2).mean().item()
    with _lift():                                       # rmsnorm casts to fp32; keep it in float64
        teacher = arms(h, w, eps)
        student = arms(h / C, w, eps)
        fixed = arms(h / C, w, eps / C ** 2)
    gap = (student - teacher).abs().max().item()
    want = teacher.abs().max().item() * abs(math.sqrt((ms + eps) / (ms + eps * C ** 2)) - 1)
    assert gap == pytest.approx(want, rel=1e-6, abs=fp(teacher.abs().max(), 8 * n))
    assert gap > 100 * fp(teacher.abs().max(), 8 * n)
    assert (fixed - teacher).abs().max().item() <= fp(teacher.abs().max(), 8 * n + 16)
    from approx.convert import eps_plan
    eps_s, (v0, v) = eps_plan(eps, n, n + 1, [1.0, C])
    hp = torch.cat([h / C, torch.tensor([v], dtype=F64)])
    with _lift():
        chan = arms(hp, torch.cat([w * math.sqrt(n / (n + 1)), torch.zeros(1, dtype=F64)]), eps_s)[:n]
    assert (chan - teacher).abs().max().item() <= fp(teacher.abs().max(), 8 * n + 32)


@FAST
@given(SEED, st.integers(2, 48), st.integers(1, 48))
def test_mismatch_approx_hidden_pad_keeps_eps(seed, d, e):
    """EP.hidden_pad_needs_eps (Capacity.lean) vs approx/plan.py's "hidden ... EXACT": the converter
    zero-pads the residual from d to D = d + e and scales every RMSNorm weight by sqrt(d/D) but keeps the
    teacher's eps (approx/real.py: s.eps = tc.eps). Over the padded width that is
    w x / sqrt(ms + eps D/d), not w x / sqrt(ms + eps); the theorem forces eps' = eps d/D. Fixed (W2-1):
    approx.convert.eps_plan returns eps d/D when no layer scalar exceeds 1 (asserted)."""
    from approx.arch import rmsnorm as arms
    eps, D = 1e-6, d + e
    h = torch.as_tensor(rng(seed).standard_normal(d)) * 1e-2
    w = torch.as_tensor(rng(seed + 1).standard_normal(d))
    pad = lambda t: torch.cat([t, torch.zeros(e, dtype=F64)])
    with _lift():
        teacher = arms(h, w, eps)
        student = arms(pad(h), pad(w * math.sqrt(d / D)), eps)[:d]
        fixed = arms(pad(h), pad(w * math.sqrt(d / D)), eps * d / D)[:d]
    assert (student - teacher).abs().max().item() > 100 * fp(teacher.abs().max(), 8 * D)
    assert (fixed - teacher).abs().max().item() <= fp(teacher.abs().max(), 8 * D + 16)
    from approx.convert import eps_plan
    assert eps_plan(eps, d, D, [1.0])[0] == pytest.approx(eps * d / D, rel=1e-15)


@UNI
@given(SEED, FLOAT(4.0, 30.0))
def test_sigmoid_router_limit(seed, s):
    """EP.sigmoid_router_limit: σ(z_e - s) / Σ_{j∈S} σ(z_j - s) -> e^{z_e} / Σ_{j∈S} e^{z_j} as s -> ∞. On
    the universal host's router (UFFN): sigmoid scores (score_c = 1), renormalised, with every logit shifted
    by -s through the router bias, against the softmax-renormalised target (score_c = 0) with the same
    weights. σ is monotone, so the top-k set is the target's; each weight is within p_e r (r = e^{max z - s})
    of the softmax weight, so the output is within r Σ p_e |Y_e| <= r max |Y|. A weight setting of UFFN
    (no loader maps a softmax target onto the sigmoid branch: the host carries both)."""
    m, x = _uffn(seed)
    with torch.no_grad():
        logits = m.router(x)
        target = m(x)
        m.score_c.fill_(1.0)
        m.router.bias.fill_(-s)
        host = m(x)
        m.router.bias.fill_(-(s + 4.0))
        host_far = m(x)
        Y = torch.einsum("edf,btef->bted", m.w_down, F.silu(torch.einsum("efd,btd->btef", m.w_gate, x))
                         * torch.einsum("efd,btd->btef", m.w_up, x))
    r = math.exp(logits.max().item() - s)
    bound = r * Y.abs().max().item()
    err = (host - target).abs().max().item()
    assert err <= bound + fp(Y.abs().max(), 1e3)
    err_far = (host_far - target).abs().max().item()
    assert err_far <= math.exp(-4.0) * bound + fp(Y.abs().max(), 1e3)


@EMU
@given(SEED)
def test_tied_sub_untied(seed):
    """EP.tied_sub_untied: headLogits E h = headLogits E h with the host's lm_head := E. A tied Llama target
    on an untied Llama host: emulate.build_host writes the target's embedding table into the host's
    lm_head (zero-padded); the logits equal the target's."""
    r = _emulate("llama", {**_E_SMALL, "tie_word_embeddings": True}, "llama",
                 {**_E_BIG, "tie_word_embeddings": False}, seed)
    E = r["tm"].model.embed_tokens.weight.detach()
    lm = r["sd"]["lm_head.weight"].to(F64)
    assert torch.equal(lm[:, :32], E) and float(lm[:, 32:].abs().max()) == 0.0
    assert r["err"] <= EMU_TOL


@FAST
@given(SEED, st.integers(1, 40), FLOAT(-20.0, 20.0))
def test_sink_host_factor(seed, N, b):
    """EP.sink_host_factor: sinkAttn s b v = (Z / (e^b + Z)) · attn s v, Z = Σ e^{s_j}, on
    attn/attention_sink.sink_attention (gpt-oss's off-by-one softmax) and its null-key form
    emulate_with_null_token."""
    r_ = rng(seed)
    s = torch.as_tensor(r_.standard_normal(N) * 2)
    V = torch.as_tensor(r_.standard_normal((N, 3)))
    Z = s.exp().sum().item()
    plain = torch.softmax(s, 0) @ V
    rhs = Z / (math.exp(b) + Z) * plain
    for lhs in (asink.sink_attention(s, V, b), asink.emulate_with_null_token(s, V, b)):
        assert (lhs - rhs).abs().max().item() <= fp(V.abs().max(), 8 * N + 16)


@FAST
@given(SEED, st.integers(1, 40), FLOAT(-30.0, 5.0))
def test_mismatch_sink_fold_is_a_limit(seed, N, b):
    """EP.sink_host_ne_plain (via EP.sink_host_factor) vs the repo's "exact" sink fold. graph/model_graph's
    SINK_ORDER flags a plain target on a sink host as an exact "fold" (exact = True); approx/plan.py lists
    "sinks: per-head sink logit ... EXACT, sink = -1e4"; attn/attention_sink.py calls b -> -inf "free". At
    every finite b the sink host is the plain head times Z / (e^b + Z) < 1, so it differs from the plain
    target by exactly e^b / (e^b + Z) |o|. It equals it in floating point only once e^b underflows against
    Z (b = -1e4 here), i.e. as a limit, not as a weight setting over the reals."""
    r_ = rng(seed)
    s = torch.as_tensor(r_.standard_normal(N) * 2)
    V = torch.as_tensor(r_.standard_normal((N, 3)))
    plain = torch.softmax(s, 0) @ V
    Z = s.exp().sum().item()
    gap = (asink.sink_attention(s, V, b) - plain).abs().max().item()
    want = math.exp(b) / (math.exp(b) + Z) * plain.abs().max().item()
    assert gap == pytest.approx(want, rel=1e-9, abs=fp(V.abs().max(), 8 * N + 16))
    assert (asink.sink_attention(s, V, -1e4) - plain).abs().max().item() <= fp(V.abs().max(), 8 * N + 16)
    assume(want > 1e3 * fp(V.abs().max(), 8 * N + 16))
    assert gap > 0


# ===== PosBias.lean =====

def _plain_attn_ref(a, x, bias=None):
    """softmax(q·k scale + bias) v through o_proj, causal, from UAttention's own projections (NoPE)."""
    B, T, _ = x.shape
    q = a.q_proj(x).view(B, T, a.H, a.hd)
    k = a.k_proj(x).view(B, T, a.H, a.hd)
    v = a.v_proj(x).view(B, T, a.H, a.hd)
    s = torch.einsum("bthd,bshd->bhts", q, k) * a.scale
    if bias is not None:
        s = s + bias
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    o = torch.einsum("bhts,bshd->bthd", s.softmax(-1), v).reshape(B, T, -1)
    return a.o_proj(o)


@UNI
@given(SEED)
def test_nope_sub_alibi(seed):
    """EP.nope_sub_alibi: alibiScore c 0 i j = c. The universal host's ALiBi branch (UAttention.alibi) at
    slope 0 is plain NoPE attention."""
    a, x = _uattn(seed)
    with torch.no_grad():
        a.alibi.zero_()
        out, ref = a(x, causal=True), _plain_attn_ref(a, x)
    assert (out - ref).abs().max().item() <= fp(ref.abs().max(), 1e3)


@UNI
@given(SEED)
def test_nope_sub_additive_bias(seed):
    """EP.nope_sub_additive_bias: biasAttn S c 0 i v = attn S c v. The zero bias table, on the universal
    host's only additive-bias branch (UAttention.alibi, the ramp table) set to 0, against attention with an
    explicit zero table."""
    a, x = _uattn(seed)
    with torch.no_grad():
        a.alibi.zero_()
        out, ref = a(x, causal=True), _plain_attn_ref(a, x, torch.zeros(1, a.H, x.shape[1], x.shape[1], dtype=F64))
    assert (out - ref).abs().max().item() <= fp(ref.abs().max(), 1e3)


@UNI
@given(SEED, st.lists(FLOAT(-2.0, 2.0), min_size=2, max_size=2))
def test_alibi_sub_additive_bias(seed, slopes):
    """EP.alibi_sub_additive_bias: the additive-bias head with table B i j = -m (i - j) is the ALiBi head.
    The universal host's ALiBi branch (UAttention.alibi, s += alibi_h (s_pos - t_pos); universal.from_bloom
    loads BLOOM's slopes into it) against attention with the explicit ramp table."""
    a, x = _uattn(seed)
    T = x.shape[1]
    m = torch.tensor(slopes, dtype=F64)
    i, j = torch.arange(T)[:, None].to(F64), torch.arange(T)[None].to(F64)
    B = -(m[:, None, None] * (i - j))[None]
    with torch.no_grad():
        a.alibi.copy_(m)
        out, ref = a(x, causal=True), _plain_attn_ref(a, x, B)
    assert (out - ref).abs().max().item() <= fp(ref.abs().max() * (1 + 2 * T), 1e3)


@UNI
@given(SEED)
def test_nope_sub_learned_abs(seed):
    """EP.nope_sub_learned_abs: e(X t) + 0 = e(X t). The universal host's learned-absolute table
    (UniversalElephantine.pos, max_pos > 0) at 0 gives the same model as the host without a table."""
    T = 6
    u0 = _randomise(uni.UniversalElephantine(_ucfg()), seed)
    u1 = uni.UniversalElephantine(_ucfg(max_pos=T + 2))
    sd = u0.state_dict()
    sd["pos"] = torch.zeros(T + 2, 8, dtype=F64)
    u1.load_state_dict(sd)
    ids = torch.randint(0, 16, (1, T), generator=gen(seed))
    with torch.no_grad():
        assert torch.equal(u1(ids), u0(ids))


# ===== Converses.lean (condition tests not already in tests/test_lean_bridges.py) =====

@FAST
@given(SEED, st.integers(2, 32), FLOAT(1.1, 10.0))
def test_condition_qknorm_out_of_regime(seed, n, ratio):
    """EP.qknorm_out_of_regime: if a token pair's RMS product differs from the folded constants and the
    folded plain logit is nonzero, the plain host's logit is not the qk-norm logit (attn/qk_norm.rmsnorm).
    The fold divides by rq rk of one token pair; a second key whose rms is `ratio` times larger gets a
    plain logit `ratio` times the qk-norm logit (up to eps)."""
    from attn.qk_norm import rmsnorm
    q, k = (torch.as_tensor(rng(seed).standard_normal((2, n))) * 3).unbind(0)
    assume(abs(float(rmsnorm(q) @ rmsnorm(k))) > 1e-3)
    rq, rk = (q.pow(2).mean() + 1e-6).sqrt(), (k.pow(2).mean() + 1e-6).sqrt()
    k2 = ratio * k                                     # same direction, different rms
    folded = (q / rq) @ (k2 / rk)
    true = rmsnorm(q) @ rmsnorm(k2)
    assert abs(float(folded - true)) > 0.05 * abs(float(true))


# =====================================================================================================
# ===== ActLimits.lean =====
# =====================================================================================================
import json as _json                                # noqa: E402
import os as _os                                    # noqa: E402
import types as _types                              # noqa: E402

from act import act_swap_agreement as asa          # noqa: E402
from act import act_width_fit as awf               # noqa: E402
from act import activation_cross as ac             # noqa: E402
from act import activations as acts                # noqa: E402
from act import exotic_activations as ea           # noqa: E402
from elephantine import universal as _uni          # noqa: E402

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def _xielu_params():
    """alpha_p > 0, alpha_n >= 0, beta, eps <= 0: the hypotheses of EP.relu2_xielu_bound."""
    return st.tuples(FLOAT(0.05, 3.0), FLOAT(0.0, 3.0), FLOAT(-3.0, 3.0), FLOAT(-1e-3, 0.0))


@FAST
@given(_xielu_params(), FLOAT(-20, 20), FLOAT(1.0, 1e5))
def test_relu2_xielu_bound(prm, x, c):
    """EP.relu2_xielu_bound: for alpha_p > 0, alpha_n >= 0, eps <= 0, c >= 1,
    |xIELU(c x)/(alpha_p c^2) - relu(x)^2| <= (alpha_n + (|beta| + alpha_n)|x|)/(alpha_p c),
    on the repo's xIELU (act/exotic_activations.xielu, torch; act/activation_cross.xielu, numpy: the
    function act_swap_agreement folds as  a X(c x + d) + e x + f  with a = 1/(alpha_p c^2), d = e = f = 0)."""
    ap, an, be, eps = prm
    host_t = ea.xielu(torch.tensor([c * x], dtype=F64), ap, an, be, eps).item() / (ap * c * c)
    host_n = float(ac.xielu(np.array([c * x]), ap, an, be, eps)[0]) / (ap * c * c)
    bound = (an + (abs(be) + an) * abs(x)) / (ap * c)
    # xIELU(c x) is a sum of <= 4 terms of size <= (ap + an + |be|)(c x)^2 + an; after / (ap c^2):
    # magnitude S <= (1 + (an + |be|)/ap)(x^2 + |x| + 1); 20 roundings
    S = (1 + (an + abs(be)) / ap) * (x * x + abs(x) + 1)
    for host in (host_t, host_n):
        assert abs(host - max(x, 0.0) ** 2) <= bound + fp(S, 20)


@MID
@given(_xielu_params(), FLOAT(-10, 10))
def test_relu2_sub_xielu(prm, x):
    """EP.relu2_sub_xielu: xIELU(c x)/(alpha_p c^2) -> relu(x)^2 as c -> inf (the act_swap fold
    a = 1/(alpha_p c^2) of act/exotic_activations.xielu): the error is under the 1/c bound at every c of
    a geometric sequence, and below 1e-6 relative at c = 1e8."""
    ap, an, be, eps = prm
    errs = []
    for c in 10.0 ** np.arange(0, 9):
        h = ea.xielu(torch.tensor([c * x], dtype=F64), ap, an, be, eps).item() / (ap * c * c)
        errs.append(abs(h - max(x, 0.0) ** 2))
        S = (1 + (an + abs(be)) / ap) * (x * x + abs(x) + 1)
        assert errs[-1] <= (an + (abs(be) + an) * abs(x)) / (ap * c) + fp(S, 20)
    assert errs[-1] <= 1e-6 * (1 + x * x)


@FAST
@given(FLOAT(-50, 50), FLOAT(1e-3, 1e6))
def test_relu_mish_bound(x, beta):
    """EP.relu_mish_bound: |mish(beta x)/beta - relu x| <= 1/beta for every x, on the repo's mish
    (act/act_swap_agreement.TACTS["mish"] = F.mish, what the act swap installs; act/activation_cross
    ACTS["mish"], numpy)."""
    t = asa.TACTS["mish"](torch.tensor([beta * x], dtype=F64)).item() / beta
    n = float(ac.ACTS["mish"](np.array([beta * x]))[0]) / beta
    for h in (t, n):
        # mish(beta x) / beta: |x| times a tanh(softplus) factor in [-0.31, 1], ~10 roundings
        assert abs(h - max(x, 0.0)) <= 1.0 / beta + fp(abs(x), 10)


@MID
@given(FLOAT(-20, 20))
def test_relu_sub_mish(x):
    """EP.relu_sub_mish: mish(beta x)/beta -> relu x as beta -> inf, uniformly (<= 1/beta at each beta of
    a geometric sequence; below 1e-9 at beta = 1e9)."""
    for beta in 10.0 ** np.arange(0, 10):
        h = asa.TACTS["mish"](torch.tensor([beta * x], dtype=F64)).item() / beta
        assert abs(h - max(x, 0.0)) <= 1.0 / beta + fp(abs(x), 10)
    assert abs(h - max(x, 0.0)) <= 1e-9 + fp(abs(x), 10)


def _oai_host():
    """The universal host's swigluoai unit (UFFN._act) exactly as elephantine/universal.from_gpt_oss
    configures it: gate rows x alpha = 1.702 (so the clip is at [alpha 7, 7]), up_shift = 1, w_down / alpha.
    Returns unit(gate, up) = the gpt-oss unit  (clip(up, -7, 7) + 1) g' sigmoid(1.702 g'), g' = min(gate, 7)."""
    z1 = torch.zeros(1, dtype=F64)
    alpha, lim = 1.702, ea.SWIGLUOAI_LIMIT
    stub = _types.SimpleNamespace(c_relu2=z1, c_relu=z1, c_gelu=z1, c_geglu=z1, gelu_tanh=z1, c_mish=z1,
                                  c_xielu=z1, c_poly=z1, c_clip=torch.ones(1, dtype=F64),
                                  clip_lim=torch.tensor([alpha * lim, lim], dtype=F64),
                                  up_shift=torch.ones(1, dtype=F64), c_situ=z1,
                                  clip_shared=torch.ones(1, dtype=F64),   # (UFFN default: clip the shared path too)
                                  clip_post=torch.full((2,), float("inf"), dtype=F64))   # (no post-act clamp)

    def unit(gate, up):
        g = torch.as_tensor(gate, dtype=F64) * alpha
        return _uni.UFFN._act(stub, g, torch.as_tensor(up, dtype=F64)) / alpha
    return unit


def _swigluoai_clamped(gate, up):
    """EP.swigluoai (Activation.lean), the gpt-oss unit, written out."""
    g = min(gate, 7.0)
    return (max(min(up, 7.0), -7.0) + 1) * g / (1 + math.exp(-1.702 * g))


@FAST
@given(FLOAT(-30, 30), FLOAT(-30, 30))
def test_swigluoai_host_unit_is_the_lean_unit(g, u):
    """(Support for the swigluoai rows) the universal host's swigluoai unit (UFFN._act with
    from_gpt_oss's clip_lim / up_shift / 1.702 fold) is EP.swigluoai: (clip(up,-7,7)+1) g' sigma(1.702 g')."""
    h = _oai_host()(torch.tensor([g], dtype=F64), torch.tensor([u], dtype=F64)).item()
    # the 1.702 fold and unfold, clamp, sigmoid (torch's float64 sigmoid is ~1e2 ulps off near 12): 1e3 roundings
    # + the blended clip g + c_clip (clamp(g) - g) of UFFN._act, which loses U (|g| + |u|)
    assert abs(h - _swigluoai_clamped(g, u)) <= fp(64, 1e3) + fp(8 * (2 * abs(g) + abs(u)), 10)


@FAST
@given(FLOAT(1e-3, 30), FLOAT(1.0, 1e6))
def test_swigluoai_scaled_gate_vanishes(z, beta):
    """EP.swigluoai_scaled_gate_vanishes: scaling the gate inside swigluoai's clamp gives
    swigluoai(beta z, 0)/beta = K/beta -> 0 (z > 0, beta z >= 7), K = 7 sigma(1.702*7), not relu z = z.
    Checked on the universal host's gpt-oss unit (from_gpt_oss settings)."""
    assume(beta * z >= 7.0)
    h = _oai_host()(torch.tensor([beta * z], dtype=F64), torch.zeros(1)).item() / beta
    # UFFN._act clips by g + c_clip (clamp(g) - g): that loses U |g| when the gate is far past the clamp
    assert abs(h - ea.SWIGLUOAI_K / beta) <= (fp(8, 1e3) + fp(1.702 * beta * z, 10)) / beta
    assert h <= 7.0 / beta


@FAST
@given(FLOAT(1e-3, 50), FLOAT(-1, 1), FLOAT(1.0, 1e6))
def test_relu_sub_swigluoai(a, s, c):
    """EP.relu_sub_swigluoai: gate c z, up a z - 1 (up bias -1), down 1/(a K) on the clamped swigluoai
    unit; for |a z| <= 6 the unit is relu z EXACTLY once c z >= 7 (z > 0), 0 at z = 0, and for z < 0 it
    is z * c z sigma(1.702 c z) / K, which tends to 0 as c -> inf (bounded here by |z| c|z| e^{-1.702 c|z|}/K).
    Weights from act/exotic_activations.relu_swigluoai_rows (helper); unit = the universal host's
    swigluoai (UFFN._act as from_gpt_oss sets it)."""
    z = s * 6.0 / a                                     # |a z| <= 6
    gs, us, ub, ds = ea.relu_swigluoai_rows(a, c)
    h = _oai_host()(torch.tensor([gs * z], dtype=F64), torch.tensor([us * z + ub], dtype=F64)).item() * ds
    # unit value <= 7 * 7 before the down scale 1/(a K); 1e3 roundings (as above)
    tol = (fp(49.0, 1e3) + fp(8 * (2 * gs * abs(z) + abs(z * a)), 10)) / (a * ea.SWIGLUOAI_K) + fp(abs(z), 1e3)
    if z >= 7.0 / c:
        assert abs(h - z) <= tol
    elif z <= 0:
        cz = c * abs(z)
        assert abs(h) <= abs(z) * cz * math.exp(-1.702 * cz) / ea.SWIGLUOAI_K + tol
    else:
        assert 0 <= h <= z + tol                          # 0 < c z < 7: the gate branch is below K


@FAST
@given(FLOAT(1.0, 3.0), FLOAT(8.0, 50.0), FLOAT(1e3, 1e6))
def test_condition_relu_sub_swigluoai_needs_bounded_inputs(a, az, c):
    """EP.relu_sub_swigluoai needs |a z| <= 6 (the up clamp): with a z > 8 the up branch is clipped at 7
    and the unit gives 8/(a) instead of z (z = az/a)."""
    z = az / a
    gs, us, ub, ds = ea.relu_swigluoai_rows(a, c)
    h = _oai_host()(torch.tensor([gs * z], dtype=F64), torch.tensor([us * z + ub], dtype=F64)).item() * ds
    assert abs(h - 8.0 / a) <= 1e-9 * (1 + z)
    assert z - h >= (az - 8.0) / a - 1e-9 * (1 + z)


def _fits_cache():
    return _json.load(open(_os.path.join(_ROOT, "act", "data", "act_fits_cache.json")))


@FAST
@given(FLOAT(-6.0, 6.0))
def test_relu_on_swigluoai_fit_takes_the_up_bias_route(x):
    """Was MISMATCH W3-1 (EP.swigluoai_scaled_gate_vanishes vs the repo's ReLU <- swigluoai fit): the cached
    fit was the scaled-gate route a X(c x + d), c ~ 1767, a ~ 1/c, which the clamped gpt-oss unit sends to
    a K ~ 0.004 for every x > 0. Now act/activation_cross.fit_host takes EP.relu_sub_swigluoai (gate c x,
    up a x - 1, down 1/(a K), |a x| <= 6), cached in act/data/act_fits_cache.json 'ReLU[]|swigluoai' and
    installed by act/act_swap_agreement.FitAct; on the real clamped unit (_oai_host) it is relu x within
    the recorded floor."""
    fl, p = _fits_cache()["ReLU[]|swigluoai"]
    assert p["route"] == "relu_sub_swigluoai" and p["up_bias"] == -1.0 and abs(p["a"]) * p["bound"] <= 6.0
    assert abs(p["down"] - 1.0 / (p["a"] * ea.SWIGLUOAI_K)) <= 1e-12
    real = _oai_host()(torch.tensor([p["c"] * x], dtype=F64),
                       torch.tensor([p["a"] * x + p["up_bias"]], dtype=F64)).item() * p["down"]
    tol = fp(49.0 * p["down"], 1e3) + fp(8 * (2 * p["c"] * abs(x) + abs(x)) * p["down"], 10)
    assert abs(real - max(x, 0.0)) <= fl + tol
    mine = asa.FitAct("swigluoai", p)(torch.tensor([x], dtype=F64)).item()
    assert abs(mine - max(x, 0.0)) <= fl + 1e-12
    assert 0 < fl <= 2e-5                                   # the limit at c = 1e5: M / c, M ~ 1.75


# =====================================================================================================
# ===== ActWidth.lean =====
# =====================================================================================================

_SWISH_HOSTS = ("SiLU", "GeLU", "swigluoai")


@FAST
@given(FLOAT(-200, 200))
def test_silu_odd_pair(t):
    """EP.silu_odd_pair: silu t - silu(-t) = t, on act/act_width_fit.TA["SiLU"] (the host unit of the
    width fits; why a linear term costs exactly 2 swish units). GeLU and the unclamped swigluoai satisfy
    the same identity (Phi(t) + Phi(-t) = 1, sigma(s) + sigma(-s) = 1), as ActWidth's docstring says."""
    for h in _SWISH_HOSTS:
        X = awf.TA[h]
        v = (X(torch.tensor([t], dtype=F64)) - X(torch.tensor([-t], dtype=F64))).item()
        assert abs(v - t) <= fp(abs(t), 10)


def _width_fits():
    return _json.load(open(_os.path.join(_ROOT, "act", "data", "act_width_fits.json")))


_WFITS = None


def _noshift_fits(host):
    """[(key, k, err, params)] of every stored shift-free fit on `host` (act/data/act_width_fits.json)."""
    global _WFITS
    if _WFITS is None:
        _WFITS = _width_fits()
    out = []
    for key, curve in _WFITS.items():
        t, h, R, fam = key.split("|")
        if h == host and fam == "noshift":
            for k, err, p in curve:
                if p is not None:
                    out.append((key, k, err, p))
    return out


def _F(host, p, a):
    """The k-unit fit sum_i alpha_i X(beta_i a + gamma_i) via act/act_width_fit.units."""
    x = torch.as_tensor(np.atleast_1d(a), dtype=F64)
    return (awf.units(awf.TA[host], x, p["beta"], p["gamma"]) @ torch.tensor(p["alpha"], dtype=F64)).numpy()


@MID
@given(st.sampled_from(_SWISH_HOSTS), st.integers(0, 10 ** 6), FLOAT(-160, 160))
def test_shiftfree_odd_part(host, idx, a):
    """EP.shiftfree_odd_part: with phi t - phi(-t) = t, a shift-free fit F(a) = sum_i alpha_i phi(beta_i a)
    has F a - F(-a) = (sum_i alpha_i beta_i) a. Checked on the repo's stored shift-free k-unit fits
    (act/act_width_fit, act/data/act_width_fits.json, every target, hosts SiLU / GeLU / swigluoai)."""
    fits = _noshift_fits(host)
    key, k, err, p = fits[idx % len(fits)]
    assert all(g == 0 for g in p["gamma"])
    lhs = float(_F(host, p, a)[0] - _F(host, p, -a)[0])
    rhs = sum(al * be for al, be in zip(p["alpha"], p["beta"])) * a
    S = sum(abs(al) * (abs(be * a) + 1) for al, be in zip(p["alpha"], p["beta"]))
    assert abs(lhs - rhs) <= fp(S, 50)                     # 2k unit evaluations and sums


def _obstruction_lhs(Y, x0, x1):
    Yp = lambda v: float(Y(torch.tensor([v], dtype=F64))[0])
    return abs(x1 * (Yp(x0) - Yp(-x0)) - x0 * (Yp(x1) - Yp(-x1)))


@MID
@given(st.sampled_from(_SWISH_HOSTS), st.integers(0, 10 ** 6), FLOAT(0, 1), FLOAT(0, 1))
def test_odd_obstruction(host, idx, s0, s1):
    """EP.odd_obstruction: any F with linear odd part (every shift-free swish fit) within eps of Y at
    +-x0, +-x1 has |x1 (Y x0 - Y(-x0)) - x0 (Y x1 - Y(-x1))| <= 2 (x0 + x1) eps. On every stored
    shift-free fit of act/data/act_width_fits.json (hosts SiLU / GeLU / swigluoai), eps the fit's
    actual error at the four points."""
    fits = _noshift_fits(host)
    key, k, err, p = fits[idx % len(fits)]
    t = key.split("|")[0]
    R = float(key.split("|")[2])
    x0, x1 = s0 * R, s1 * R
    Y = awf.TA[t]
    pts = np.array([x0, -x0, x1, -x1])
    Yv = Y(torch.tensor(pts, dtype=F64)).numpy()
    eps = float(np.abs(_F(host, p, pts) - Yv).max())
    S = sum(abs(al) * (abs(be) * R + 1) for al, be in zip(p["alpha"], p["beta"])) + float(np.abs(Yv).max())
    assert _obstruction_lhs(Y, x0, x1) <= 2 * (x0 + x1) * eps + fp(S * R, 100)


@MID
@given(st.sampled_from(["mish", "situ", "relu2", "xielu", "GeLU", "gelu_tanh", "swigluoai", "ReLU"]),
       st.sampled_from([6.0, 160.0]), st.integers(1, 8), FLOAT(0, 1), FLOAT(0, 1))
def test_shiftfree_silu_obstruction(target, R, k, s0, s1):
    """EP.shiftfree_silu_obstruction: every bias-free k-unit SiLU fit sum alpha_i silu(beta_i a) within eps
    of Y at +-x0, +-x1 has |x1 (Y x0 - Y(-x0)) - x0 (Y x1 - Y(-x1))| <= 2 (x0 + x1) eps. On the repo's
    stored 'target|SiLU|R|noshift' fits (act/act_width_fit), eps measured at the four points."""
    curve = _width_fits_cached().get(f"{target}|SiLU|{R:g}|noshift")
    assume(curve is not None and len(curve) >= k and curve[k - 1][2] is not None)
    _, err, p = curve[k - 1]
    x0, x1 = s0 * R, s1 * R
    Y = awf.TA[target]
    pts = np.array([x0, -x0, x1, -x1])
    Yv = Y(torch.tensor(pts, dtype=F64)).numpy()
    eps = float(np.abs(_F("SiLU", p, pts) - Yv).max())
    S = sum(abs(al) * (abs(be) * R + 1) for al, be in zip(p["alpha"], p["beta"])) + float(np.abs(Yv).max())
    assert _obstruction_lhs(Y, x0, x1) <= 2 * (x0 + x1) * eps + fp(S * R, 100)


def _width_fits_cached():
    global _WFITS
    if _WFITS is None:
        _WFITS = _width_fits()
    return _WFITS


def test_shiftfree_silu_obstruction_floors_the_stored_mish_fits():
    """EP.shiftfree_silu_obstruction, read as a lower bound: on the grid act/act_width_fit evaluates, the
    stored error of every shift-free SiLU-host mish fit (k = 1..8, R = 6 and 160) is at least
    max over grid pairs |x1 (Y x0 - Y(-x0)) - x0 (Y x1 - Y(-x1))| / (2 (x0 + x1)). That is the floor the
    graph/act_width docstring quotes (0.10 on [-160, 160]): no width helps without gate shifts."""
    fits = _width_fits_cached()
    for R in (6.0, 160.0):
        x = awf.grid(R, dense=4).numpy()
        xp = np.unique(np.abs(x))
        xp = xp[xp > 0]
        Y = awf.TA["mish"]
        odd = (Y(torch.tensor(xp, dtype=F64)) - Y(torch.tensor(-xp, dtype=F64))).numpy()
        sub = xp[:: max(1, len(xp) // 400)]
        osub = odd[:: max(1, len(xp) // 400)]
        lb = np.abs(sub[None, :] * osub[:, None] - sub[:, None] * osub[None, :]) / (2 * (sub[:, None] + sub[None, :]))
        floor = float(lb.max())
        for k, err, p in fits[f"mish|SiLU|{R:g}|noshift"]:
            # the stored error is a max over that grid (which contains +-sub), so it dominates the floor
            assert err >= floor - 1e-12, (R, k, err, floor)
        assert floor > (0.05 if R == 160.0 else 0.02)


# =====================================================================================================
# ===== ActFits.lean =====
# =====================================================================================================

_LEAN_FIT6 = dict(   # EP.actFit_mish_silu6_explicit: alpha, beta in units of 2^-24, gamma of 2^-40
    alpha=[-160505003, 17708581, 172903600, -10714356, 8567841, 314572],
    beta=[-3020478, -16340396, 2903133, 19621551, -22279116, -16954119],
    gamma=[20728877624, 190579807737, -67410679846, -988909254425, 244274895690, 4129126169491])


def _stored_fit6():
    k, err, p = _width_fits_cached()["mish|SiLU|6|shift"][5]
    assert k == 6
    return err, p


def _mish_fit_err(p, a):
    a = torch.as_tensor(np.atleast_1d(a), dtype=F64)
    F_ = awf.units(awf.TA["SiLU"], a, p["beta"], p["gamma"]) @ torch.tensor(p["alpha"], dtype=F64)
    return (F_ - awf.TA["mish"](a)).abs()


def test_actFit_mish_silu6_explicit():
    """EP.actFit_mish_silu6_explicit: the certified coefficients are the repo's stored fit
    'mish|SiLU|6|shift', k = 6 (act/act_width_fit -> act/data/act_width_fits.json, branch act-width-fits)
    rounded to 2^-24 (alpha, beta) and 2^-40 (gamma): every integer matches. Both the stored fit and the
    rounded one are within 4e-5 of mish on a 4x-dense grid of [-6, 6] (act_width_fit.evaluate)."""
    err, p = _stored_fit6()
    assert [round(v * 2 ** 24) for v in p["alpha"]] == _LEAN_FIT6["alpha"]
    assert [round(v * 2 ** 24) for v in p["beta"]] == _LEAN_FIT6["beta"]
    assert [round(v * 2 ** 40) for v in p["gamma"]] == _LEAN_FIT6["gamma"]
    pr = dict(alpha=[v / 2 ** 24 for v in _LEAN_FIT6["alpha"]], beta=[v / 2 ** 24 for v in _LEAN_FIT6["beta"]],
              gamma=[v / 2 ** 40 for v in _LEAN_FIT6["gamma"]])
    e_stored = awf.evaluate(awf.TA["SiLU"], awf.TA["mish"], 6.0, p)
    e_round = awf.evaluate(awf.TA["SiLU"], awf.TA["mish"], 6.0, pr)
    assert abs(e_stored - err) <= 1e-12 and e_stored <= 4e-5
    assert e_round <= 4e-5
    assert abs(e_round - 1.80e-5) <= 1e-7                 # the rounded fit's float error PROOFS.md quotes


@FAST
@given(FLOAT(-6, 6))
def test_actFit_mish_silu6(a):
    """EP.actFit_mish_silu6: for every real a in [-6, 6], |mish a - sum_{i<=6} alpha_i SiLU(beta_i a + gamma_i)|
    <= 4e-5, with the stored k = 6 fit (and its dyadic rounding) evaluated by act/act_width_fit.units.
    Terms: |alpha_i| <= 11, |units| <= 11, so the sum has magnitude <= 6 * 121; 20 roundings."""
    err, p = _stored_fit6()
    pr = dict(alpha=[v / 2 ** 24 for v in _LEAN_FIT6["alpha"]], beta=[v / 2 ** 24 for v in _LEAN_FIT6["beta"]],
              gamma=[v / 2 ** 40 for v in _LEAN_FIT6["gamma"]])
    for q in (p, pr):
        assert float(_mish_fit_err(q, a)[0]) <= 4e-5 + fp(6 * 121, 20)


# =====================================================================================================
# ===== LimitInstances.lean, ModelLimit.lean, LimitKit.lean =====
# =====================================================================================================

@FAST
@given(FLOAT(-50, 50), FLOAT(1e-3, 1e6))
def test_relu_gelu_bound(x, beta):
    """EP.relu_gelu_bound: |GeLU(beta x)/beta - ReLU x| = |x| P(Z >= beta|x|) <= 1/beta for every x, on
    act/activations.gelu_b (= x Phi(beta x)) and the act swap's GeLU (TACTS["GeLU"] = F.gelu, erf)."""
    a = acts.gelu_b(torch.tensor([x], dtype=F64), beta).item()
    b = asa.TACTS["GeLU"](torch.tensor([beta * x], dtype=F64)).item() / beta
    for h in (a, b):
        assert abs(h - max(x, 0.0)) <= 1.0 / beta + fp(abs(x), 10)


# --- the activation-swap host inside a model: act/act_swap_agreement.act_slots / install / FitAct -----

def _opt(seed):
    from transformers import OPTConfig, OPTForCausalLM
    torch.manual_seed(seed)
    cfg = OPTConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2, ffn_dim=24, num_attention_heads=2,
                    max_position_embeddings=32, word_embed_proj_dim=16, activation_function="relu",
                    do_layer_norm_before=True, dropout=0.0, attention_dropout=0.0)
    m = OPTForCausalLM(cfg).to(F64).eval()
    for prm in m.parameters():                          # random, nonzero biases / LN affine
        with torch.no_grad():
            prm.add_(0.3 * torch.randn(prm.shape, dtype=F64, generator=gen(seed + prm.numel())))
    m.config._attn_implementation = "eager"
    return m


HF = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@HF
@given(SEED, st.sampled_from(["SiLU", "mish", "GeLU"]), FLOAT(0.5, 1e4))
def test_silu_layer_bound(seed, host, beta):
    """EP.silu_layer_bound (and EP.actLayer_dist with eta = 1/beta, EP.relu_mish_bound / relu_gelu_bound
    for the other two hosts): a stock OPT decoder layer (ReLU MLP) with its activation swapped by
    act/act_swap_agreement (act_slots + install + FitAct(host, (1/beta, beta, 0, 0, 0)), i.e. SiLU(beta .)/beta)
    is within (sum_u ||w_u||)/beta of the ReLU layer, at every token of a random state (w_u = fc2 columns,
    l2 norms)."""
    m = _opt(seed)
    slots = asa.act_slots(m)
    assert len(slots) == 2
    _, set_fns = asa.install(slots)
    layer = m.model.decoder.layers[0]
    x = torch.randn(1, 7, 16, dtype=F64, generator=gen(seed)) * 3
    with torch.no_grad():
        set_fns(None)
        yt = layer(x)[0]
        set_fns([asa.FitAct(host, (1.0 / beta, beta, 0.0, 0.0, 0.0))] * 2)
        yh = layer(x)[0]
        set_fns(None)
    wsum = layer.fc2.weight.norm(dim=0).sum().item()
    S = yt.abs().max().item() + 1
    assert ((yh - yt).norm(dim=-1).max().item() <= wsum / beta + fp(S, 1e3))


@HF
@given(SEED)
def test_actLayer_dist(seed):
    """EP.actLayer_dist: activations within eta at this state's pre-activations => layers within
    eta sum_u ||w_u||. Stock OPT layer, target activation swapped (act_swap_agreement.install) for the
    repo's GeLU <- SiLU single-op fit (act/data/act_fits_cache.json 'GELUActivation[]|SiLU', FitAct) vs the
    exact GeLU; eta measured at the layer's own fc1 pre-activations (captured by a hook)."""
    m = _opt(seed)
    _, set_fns = asa.install(asa.act_slots(m))
    _, prm = _fits_cache()["GELUActivation[]|SiLU"]
    layer = m.model.decoder.layers[1]
    x = torch.randn(1, 6, 16, dtype=F64, generator=gen(seed)) * 2
    pre = {}
    hk = layer.fc1.register_forward_hook(lambda mod, i, o: pre.__setitem__("z", o.detach()))
    with torch.no_grad():
        set_fns([asa.TACTS["GeLU"]] * 2)
        yt = layer(x)[0]
        fit = asa.FitAct("SiLU", tuple(prm))
        set_fns([fit] * 2)
        yh = layer(x)[0]
    hk.remove()
    z = pre["z"]
    eta = (fit(z) - asa.TACTS["GeLU"](z)).abs().max().item()
    wsum = layer.fc2.weight.norm(dim=0).sum().item()
    assert (yh - yt).norm(dim=-1).max().item() <= eta * wsum + fp(yt.abs().max().item() + 1, 1e3)


# --- toy residual MLP stacks (EP.actLayer: x -> A x + sum_u sigma(a_u . x + b_u) w_u) with the repo's swap
# module FitAct as the host activation. Norms: l2 on states, sup on logits; head Lh = max row l2 norm.

def _stack(seed, n, d, m, s):
    r = rng(seed)
    T = lambda a: torch.tensor(a, dtype=F64)
    return [(T(np.eye(d) + s * r.standard_normal((d, d)) / math.sqrt(d)),
             T(s * r.standard_normal((m, d)) / math.sqrt(d)), T(s * r.standard_normal(m)),
             T(s * r.standard_normal((d, m)) / math.sqrt(m))) for _ in range(n)]


def _run(layers, act, x, record=None):
    for A, P, b, W in layers:
        z = x @ P.T + b
        if record is not None:
            record.append(float(z.abs().max()))
        x = x @ A.T + act(z) @ W.T
    return x


def _opn(M):
    return float(torch.linalg.matrix_norm(M, ord=2))


def _lips(layers, lip_act):
    """Per-layer Lipschitz constants (>= 1, as the theorems ask): ||A|| + Lip(sigma) ||W|| ||P||."""
    return [max(1.0, _opn(A) + lip_act * _opn(W) * _opn(P)) for A, P, b, W in layers]


def _wsum(layers):
    return sum(float(W.norm(dim=0).sum()) for A, P, b, W in layers)


def _amax(layers):
    return max(float(P.norm(dim=1).max()) for A, P, b, W in layers)


def _relu_swap(host, beta):
    return asa.FitAct(host, (1.0 / beta, beta, 0.0, 0.0, 0.0))


def _xielu_swap(c):
    return asa.FitAct("xielu", (1.0 / (0.8 * c * c), c, 0.0, 0.0, 0.0))  # TACTS xielu: alpha_p = 0.8


_XI = dict(ap=0.8, an=0.8, be=0.5)                      # TACTS["xielu"]'s parameters (default init)


def _model_bound_check(seed, host, beta, n=3, d=6, m=10, s=0.6):
    layers = _stack(seed, n, d, m, s)
    Wh = torch.tensor(rng(seed + 1).standard_normal((5, d)), dtype=F64)
    Lh = float(Wh.norm(dim=1).max())
    x = torch.tensor(rng(seed + 2).standard_normal(d) * 3, dtype=F64)
    yt = Wh @ _run(layers, torch.relu, x)
    yh = Wh @ _run(layers, _relu_swap(host, beta), x)
    bound = Lh * math.prod(_lips(layers, 1.0)) * _wsum(layers) / beta
    return float((yh - yt).abs().max()), bound, float(yt.abs().max())


@MID
@given(SEED, FLOAT(0.1, 1e6))
def test_relu_silu_model_bound(seed, beta):
    """EP.relu_silu_model_bound (via EP.model_error_bound_head, EP.errBound_le_prod_sum, EP.silu_layer_bound):
    a stack of n layers A_i + ReLU-MLP_i with the activation swapped for act_swap_agreement.FitAct
    ("SiLU", (1/beta, beta, 0, 0, 0)) has logits within Lh (prod L_i)(sum_i sum_u ||w_iu||)/beta."""
    err, bound, S = _model_bound_check(seed, "SiLU", beta)
    assert err <= bound + fp(S + 10, 1e3)


@MID
@given(SEED, FLOAT(0.1, 1e6))
def test_relu_mish_model_bound(seed, beta):
    """EP.relu_mish_model_bound: as relu_silu_model_bound with FitAct("mish", (1/beta, beta, 0, 0, 0))."""
    err, bound, S = _model_bound_check(seed, "mish", beta)
    assert err <= bound + fp(S + 10, 1e3)


@MID
@given(SEED, FLOAT(0.1, 1e6))
def test_relu_gelu_model_bound(seed, beta):
    """EP.relu_gelu_model_bound: as relu_silu_model_bound with FitAct("GeLU", (1/beta, beta, 0, 0, 0))."""
    err, bound, S = _model_bound_check(seed, "GeLU", beta)
    assert err <= bound + fp(S + 10, 1e3)


@MID
@given(SEED, st.sampled_from(["SiLU", "mish", "GeLU"]))
def test_relu_silu_model_limit(seed, host):
    """EP.relu_silu_model_limit, EP.relu_mish_model_limit, EP.relu_gelu_model_limit (instances of
    EP.act_swap_model_limit / EP.model_limit_pointwise): along beta = 10^k the FitAct host stack's logits
    stay under the 1/beta bound and reach the ReLU target's to 1e-6 relative at beta = 1e8."""
    errs = []
    for beta in 10.0 ** np.arange(0, 9):
        err, bound, S = _model_bound_check(seed, host, beta)
        assert err <= bound + fp(S + 10, 1e3)
        errs.append(err)
    assert errs[-1] <= 1e-6 * (1 + S)


_ETA = {}


def _swish_gelu_eta():
    if "g" not in _ETA:
        host, tgt = acts._act_fn("swish", beta=1.702), acts.gelu
        g = torch.linspace(-30, 30, 60001, dtype=F64)
        _ETA["g"] = float((host(g) - tgt(g)).abs().max()) + 2.2 * 1e-3
    return _ETA["g"]


@MID
@given(SEED, FLOAT(1.0, 1e5))
def test_act_swap_model_bound(seed, scale):
    """EP.act_swap_model_bound: a host activation within eta of the target's EVERYWHERE puts the logits within
    Lh (prod L_i)(sum sum ||w||) eta. Target GeLU (act/activations.gelu), host the repo's classic GeLU <-
    SiLU swap swish_b(., 1.702) (act/activations._act_fn("swish", 1.702), run_gpt2's variant); eta = the
    global sup |gelu - swish_1.702| (both tend to relu at +-inf), from a grid on [-30, 30] plus the
    derivative slack (|d/dx| <= 2.2, spacing 1e-3). Pre-activations scaled by `scale` exercise the tails."""
    host, tgt = acts._act_fn("swish", beta=1.702), acts.gelu
    eta = _swish_gelu_eta()
    layers = _stack(seed, 3, 6, 10, 0.6)
    layers = [(A, P * scale, b * scale, W / scale) for A, P, b, W in layers]
    Wh = torch.tensor(rng(seed + 1).standard_normal((5, 6)), dtype=F64)
    Lh = float(Wh.norm(dim=1).max())
    x = torch.tensor(rng(seed + 2).standard_normal(6) * 3, dtype=F64)
    yt, yh = Wh @ _run(layers, tgt, x), Wh @ _run(layers, host, x)
    lip_gelu = 1.13                                       # max |gelu'| = 1.1289...
    bound = Lh * math.prod(_lips(layers, lip_gelu)) * _wsum(layers) * eta
    assert float((yh - yt).abs().max()) <= bound + fp(float(yt.abs().max()) + 10, 1e4)


def _region_check(seed, sigT, lipT, sigH_of, eta_of, R_extra=1.0, n=3, d=6, m=10, s=0.5):
    """act_swap_model_bound_on / relu2_xielu_model_bound: returns (err, bound, S) or None if the region
    hypothesis (every state within B of the target trajectory in the pre-activation box [-R, R]) fails."""
    layers = _stack(seed, n, d, m, s)
    Wh = torch.tensor(rng(seed + 1).standard_normal((5, d)), dtype=F64)
    Lh = float(Wh.norm(dim=1).max())
    x = torch.tensor(rng(seed + 2).standard_normal(d) * 2, dtype=F64)
    rec = []
    yt = Wh @ _run(layers, sigT, x, rec)
    R = max(rec) + R_extra
    eta = eta_of(R)
    L = _lips(layers, lipT(R))
    B = math.prod(L) * _wsum(layers) * eta
    if max(rec) + _amax(layers) * B > R:                  # hK: the B-neighbourhood stays in the boxes
        return None
    yh = Wh @ _run(layers, sigH_of(R), x)
    return float((yh - yt).abs().max()), Lh * B, float(yt.abs().max())


@MID
@given(SEED)
def test_act_swap_model_bound_on(seed):
    """EP.act_swap_model_bound_on (with EP.model_bound_on_prod): activations within eta only on [-R, R]. Target
    GeLU (TACTS), host the repo's single-op GeLU <- SiLU fit FitAct("SiLU", act_fits_cache 'GELUActivation[]|SiLU'),
    which has a linear term (no global eta). eta = grid sup on [-R, R] + derivative slack; R = the target
    trajectory's max |pre-activation| + 3, and the test checks that every state within
    B = (prod L)(sum sum ||w||) eta stays in the box (||a_u|| B <= 1). Logits within Lh B."""
    _, prm = _fits_cache()["GELUActivation[]|SiLU"]
    fit = asa.FitAct("SiLU", tuple(prm))

    def eta_of(R):
        g = torch.linspace(-R, R, int(2000 * R) + 1, dtype=F64)
        return float((fit(g) - asa.TACTS["GeLU"](g)).abs().max()) + 2.5 * (R / (1000 * R))
    out = _region_check(seed, asa.TACTS["GeLU"], lambda R: 1.13, lambda R: fit, eta_of, R_extra=3.0, s=0.4)
    assume(out is not None)
    err, bound, S = out
    assert err <= bound + fp(S + 10, 1e4)


@MID
@given(SEED, FLOAT(1.0, 1e6))
def test_relu2_xielu_model_bound(seed, c):
    """EP.relu2_xielu_model_bound (via EP.xielu_host_close): relu^2-MLP stack, host FitAct("xielu",
    (1/(alpha_p c^2), c, 0, 0, 0)) (TACTS xielu: alpha_p = alpha_n = 0.8, beta = 0.5, eps = -1e-6); with every
    pre-activation within R along the B-neighbourhood of the target trajectory, logits within
    Lh (prod L_i)(sum sum ||w||)(alpha_n + (|beta| + alpha_n) R)/(alpha_p c). Target layers are
    (||A|| + 2R ||W|| ||P||)-Lipschitz on the box."""
    eta_of = lambda R: (_XI["an"] + (abs(_XI["be"]) + _XI["an"]) * R) / (_XI["ap"] * c)
    out = _region_check(seed, lambda z: torch.relu(z) ** 2, lambda R: 2 * R, lambda R: _xielu_swap(c), eta_of,
                        s=0.4)
    assume(out is not None)
    err, bound, S = out
    assert err <= bound + fp(S + 10, 1e4)


@MID
@given(SEED)
def test_relu2_xielu_model_limit(seed):
    """EP.relu2_xielu_model_limit: along c = 10^k the xIELU host stack's error stays under the bound and
    goes to 0 (1e-4 relative at c = 1e8; the bound is O(1/c))."""
    errs = []
    for c in 10.0 ** np.arange(2, 9):
        eta_of = lambda R, c=c: (_XI["an"] + (abs(_XI["be"]) + _XI["an"]) * R) / (_XI["ap"] * c)
        out = _region_check(seed, lambda z: torch.relu(z) ** 2, lambda R: 2 * R, lambda R, c=c: _xielu_swap(c),
                            eta_of, s=0.4)
        if out is None:
            continue
        err, bound, S = out
        assert err <= bound + fp(S + 10, 1e4)
        errs.append((err, S))
    assume(errs)
    assert errs[-1][0] <= 1e-4 * (1 + errs[-1][1])


# --- greedy generation: act/act_swap_agreement.greedy (the repo's greedy routine) on a toy LM --------

class _ToyLM(torch.nn.Module):
    """prefix -> embed -> stack -> head. greedy() first tries the KV-cache path; this model refuses it
    (TypeError), so greedy falls back to full recompute, reading logits[:, -1]."""

    def __init__(self, E, layers, act, Wh, cap=None):
        super().__init__()
        self.E, self.layers, self.act, self.Wh, self.cap = E, layers, act, Wh, cap

    def state(self, ids):
        e = self.E[ids[0]]
        return e[-1] + 0.5 * torch.tanh(e.sum(0))

    def logits_of(self, ids):
        z = self.Wh @ _run(self.layers, self.act, self.state(ids))
        return z if self.cap is None else self.cap(z)

    def forward(self, ids, use_cache=None, **kw):
        if use_cache:
            raise TypeError("no KV cache")
        z = self.logits_of(ids)
        out = torch.zeros(1, ids.shape[1], z.shape[0], dtype=F64)
        out[0, -1] = z
        return _types.SimpleNamespace(logits=out)


def _toy(seed, d=6, m=10, n=3, V=7, s=0.6):
    r = rng(seed)
    E = torch.tensor(r.standard_normal((V, d)), dtype=F64)
    Wh = torch.tensor(r.standard_normal((V, d)) * 2, dtype=F64)
    return E, _stack(seed + 5, n, d, m, s), Wh


def _trajectory(model, ids, n_new):
    """The target's greedy tokens and its margin delta (top1 - top2) at every step."""
    toks = asa.greedy(model, ids, n_new)
    seq = torch.cat([ids, toks], 1)
    margins = []
    for t in range(ids.shape[1], seq.shape[1]):
        z = model.logits_of(seq[:, :t]).sort(descending=True).values
        margins.append(float(z[0] - z[1]))
    return toks, min(margins)


N_NEW = 6


@MID
@given(SEED, st.sampled_from(["SiLU", "mish", "GeLU"]))
def test_relu_silu_greedy(seed, host):
    """EP.relu_silu_greedy, EP.relu_mish_greedy, EP.relu_gelu_greedy (via EP.greedy_of_bound,
    EP.act_swap_greedy): with the target's greedy margin delta along its own trajectory, every
    beta > beta* = 2 Lh (prod L_i)(sum sum ||w||)/delta makes the FitAct(host, (1/beta, beta, ...)) LM generate
    exactly the ReLU LM's tokens (act_swap_agreement.greedy on both)."""
    E, layers, Wh = _toy(seed)
    ids = torch.tensor([[seed % 7, (seed // 7) % 7]])
    tgt = _ToyLM(E, layers, torch.relu, Wh)
    toks, delta = _trajectory(tgt, ids, N_NEW)
    assume(delta > 1e-6)
    Lh = float(Wh.norm(dim=1).max())
    beta_star = 2 * Lh * math.prod(_lips(layers, 1.0)) * _wsum(layers) / delta
    beta = 1.01 * beta_star
    hostlm = _ToyLM(E, layers, _relu_swap(host, beta), Wh)
    assert torch.equal(asa.greedy(hostlm, ids, N_NEW), toks)


@MID
@given(SEED)
def test_act_swap_greedy_on(seed):
    """EP.act_swap_greedy_on (and EP.greedy_of_layer_bounds_on): the region version for the repo's GeLU <- SiLU
    single-op fit. When 2 Lh B < delta (B = (prod L)(sum sum ||w||) eta on the box the B-neighbourhood of the
    target trajectory stays in), the fit-host LM generates the GeLU LM's tokens. The fit's eta (~0.012) does
    not shrink, so the MLP down-projections are scaled by 0.02 to bring the threshold within the margin."""
    _, prm = _fits_cache()["GELUActivation[]|SiLU"]
    fit = asa.FitAct("SiLU", tuple(prm))
    E, layers, Wh = _toy(seed, s=0.6)
    layers = [(A, P, b, W * 0.02) for A, P, b, W in layers]   # small MLP writes: 2 Lh B < delta is reachable
    ids = torch.tensor([[seed % 7, (seed // 7) % 7]])
    tgt = _ToyLM(E, layers, asa.TACTS["GeLU"], Wh)
    toks, delta = _trajectory(tgt, ids, N_NEW)
    # region: max |pre| over every state of the trajectory (all prefixes)
    seq = torch.cat([ids, toks], 1)
    rec = []
    for t in range(ids.shape[1], seq.shape[1]):
        _run(layers, asa.TACTS["GeLU"], tgt.state(seq[:, :t]), rec)
    R = max(rec) + 0.5
    g = torch.linspace(-R, R, 20001, dtype=F64)
    eta = float((fit(g) - asa.TACTS["GeLU"](g)).abs().max()) + 2.5 * (2 * R / 20000)
    B = math.prod(_lips(layers, 1.13)) * _wsum(layers) * eta
    Lh = float(Wh.norm(dim=1).max())
    assume(max(rec) + _amax(layers) * B <= R and 2 * Lh * B < delta)
    assert torch.equal(asa.greedy(_ToyLM(E, layers, fit, Wh), ids, N_NEW), toks)


@MID
@given(SEED)
def test_relu2_xielu_greedy(seed):
    """EP.relu2_xielu_greedy: for c >= 1 with c > c* = 2 Lh (prod L)(sum sum ||w||)(alpha_n + (|beta| + alpha_n) R)
    /(alpha_p delta), and the region condition, the FitAct("xielu", (1/(alpha_p c^2), c, 0, 0, 0)) LM
    generates the relu^2 LM's tokens."""
    E, layers, Wh = _toy(seed, s=0.4)
    ids = torch.tensor([[seed % 7, (seed // 7) % 7]])
    relu2 = lambda z: torch.relu(z) ** 2
    tgt = _ToyLM(E, layers, relu2, Wh)
    toks, delta = _trajectory(tgt, ids, N_NEW)
    assume(delta > 1e-6)
    seq = torch.cat([ids, toks], 1)
    rec = []
    for t in range(ids.shape[1], seq.shape[1]):
        _run(layers, relu2, tgt.state(seq[:, :t]), rec)
    R = max(rec) + 1.0
    PL, ws, Lh = math.prod(_lips(layers, 2 * R)), _wsum(layers), float(Wh.norm(dim=1).max())
    K = _XI["an"] + (abs(_XI["be"]) + _XI["an"]) * R
    c_star = 2 * Lh * PL * ws * K / (_XI["ap"] * delta)
    c_reg = PL * ws * K * _amax(layers) / _XI["ap"]       # makes ||a|| B <= 1: the B-ball stays in the box
    c = max(1.0, 1.01 * c_star, 1.01 * c_reg)
    assert torch.equal(asa.greedy(_ToyLM(E, layers, _xielu_swap(c), Wh), ids, N_NEW), toks)


@MID
@given(SEED, FLOAT(0.0, 0.499))
def test_greedy_preserved(seed, frac):
    """EP.greedy_preserved, EP.greedy_of_bound, EP.pick_eq_of_margin on the repo's greedy routine
    (act/act_swap_agreement.greedy, argmax pick): host logits within < delta/2 of the target's at every
    prefix of the target's own trajectory (here: target + a bounded perturbation of size frac * delta,
    frac < 1/2) generate exactly the target's tokens."""
    E, layers, Wh = _toy(seed)
    ids = torch.tensor([[seed % 7]])
    tgt = _ToyLM(E, layers, torch.relu, Wh)
    toks, delta = _trajectory(tgt, ids, N_NEW)
    assume(delta > 1e-9)
    noise = torch.tensor(rng(seed + 9).uniform(-1, 1, 7), dtype=F64) * frac * delta
    host = _ToyLM(E, layers, torch.relu, Wh, cap=lambda z: z + noise)
    assert torch.equal(asa.greedy(host, ids, N_NEW), toks)


from attn import softcap as _sc                   # noqa: E402
from attn.qk_norm import attend as _attend        # noqa: E402


@MID
@given(SEED, FLOAT(0.0, 0.499), FLOAT(2.0, 100.0))
def test_final_softcap_greedy(seed, frac, c):
    """EP.final_softcap_greedy: host logits = softcap c tanh(z/c) (attn/softcap.softcap, the expression of
    UniversalElephantine.final_cap) of logits within < delta/2 of the target's generate the target's tokens,
    at every c > 0 (the cap is strictly increasing). Float caveat: tanh rounds to exactly +-1 once
    |z|/c > ~19, which would create ties; the logits here stay below 15 c."""
    E, layers, Wh = _toy(seed)
    ids = torch.tensor([[seed % 7]])
    tgt = _ToyLM(E, layers, torch.relu, Wh)
    toks, delta = _trajectory(tgt, ids, N_NEW)
    assume(delta > 1e-9)
    seq = torch.cat([ids, toks], 1)
    zmax = max(float(tgt.logits_of(seq[:, :t]).abs().max()) for t in range(1, seq.shape[1] + 1))
    assume(zmax + delta < 15 * c)
    noise = torch.tensor(rng(seed + 9).uniform(-1, 1, 7), dtype=F64) * frac * delta
    host = _ToyLM(E, layers, torch.relu, Wh, cap=lambda z: _sc.softcap(z + noise, c))
    assert torch.equal(asa.greedy(host, ids, N_NEW), toks)


@MID
@given(SEED, st.integers(1, 12), FLOAT(0.0, 2.0), FLOAT(0.1, 10.0))
def test_attn_perturb(seed, T, eta, M):
    """EP.attn_perturb: logits moved by <= eta on the key set and values of norm <= M move the attention output
    by <= (e^{2 eta} - 1) M. On attn/qk_norm.attend (causal softmax), every query row."""
    r = rng(seed)
    s = torch.tensor(r.standard_normal((T, T)) * 3, dtype=F64)
    t = s + torch.tensor(r.uniform(-1, 1, (T, T)), dtype=F64) * eta
    v = torch.tensor(r.standard_normal((T, 4)), dtype=F64)
    v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-300) * torch.tensor(r.uniform(0, M, (T, 1)), dtype=F64)
    d = (_attend(s) @ v - _attend(t) @ v).norm(dim=1)
    assert float(d.max()) <= (math.exp(2 * eta) - 1) * M + fp(M, 100 * T)


# --- ALiBi on the RoPE frozen-pair host: attn/positional_cross.alibi_rope_host + rope_score -----------

from attn import positional_cross as _pc          # noqa: E402


def _alibi_layer(x, q, k, W, m, theta=None):
    """One causal head over T positions, residual included: target logits q.k - m (i - j); host logits from
    the frozen-pair RoPE host (alibi_rope_host + rope_score) at frequency theta. q, k: constant (P, 2)
    query / key pairs (EP.alibiLayer allows any maps; constant ones keep the target layer
    (1 + ||W||)-Lipschitz); values val(x) = W tanh(x), bounded by ||W|| sqrt(d)."""
    T = x.shape[0]
    s = np.empty((T, T))
    if theta is not None:
        fr, qh, kh = _pc.alibi_rope_host(q, k, m, theta)
    for i in range(T):
        for j in range(T):
            s[i, j] = (_pc.rope_score(fr, qh, kh, i, j) if theta is not None
                       else float(np.sum(q * k)) - m * (i - j))
    a = _attend(torch.tensor(s, dtype=F64))
    return x + a @ (torch.tanh(x) @ W.T)


def _alibi_stack(seed, n, T, d, P=2):
    r = rng(seed)
    return [(r.standard_normal((P, 2)), r.standard_normal((P, 2)),
             torch.tensor(r.standard_normal((d, d)) * 0.5 / math.sqrt(d), dtype=F64), float(r.uniform(-1, 1)))
            for _ in range(n)]


@MID
@given(SEED, st.integers(1, 6), FLOAT(1e-4, 0.3), FLOAT(-2, 2))
def test_rope_alibi_layer_bound(seed, T, theta, m):
    """EP.rope_alibi_layer_bound: one causal ALiBi layer vs its frozen-pair RoPE host
    (attn/positional_cross.alibi_rope_host + rope_score, softmax by attn/qk_norm.attend), values bounded by M:
    within (e^{2 eta} - 1) M, eta = |m| T^3 theta^2 / 6, at every position (sup over positions of l2)."""
    d = 4
    r = rng(seed)
    q, k = r.standard_normal((2, 2)), r.standard_normal((2, 2))
    W = torch.tensor(r.standard_normal((d, d)), dtype=F64)
    x = torch.tensor(r.standard_normal((T, d)) * 2, dtype=F64)
    M = _opn(W) * math.sqrt(d)
    yt, yh = _alibi_layer(x, q, k, W, m), _alibi_layer(x, q, k, W, m, theta)
    eta = abs(m) * T ** 3 * theta ** 2 / 6
    # host logit: (m/theta) sin(theta (i-j)) carries |m|/theta * U relative rounding through cos/sin
    slack = fp(M, 100) + 2 * M * (abs(m) / theta + 10) * 100 * U
    assert float((yh - yt).norm(dim=1).max()) <= (math.exp(2 * eta) - 1) * M + slack


def _alibi_model(seed, n, T, d, theta, x):
    stk = _alibi_stack(seed, n, T, d)
    yt, yh = x, x
    for q, k, W, m in stk:
        yt = _alibi_layer(yt, q, k, W, m)
        yh = _alibi_layer(yh, q, k, W, m, theta)
    L = math.prod(1 + _opn(W) for q, k, W, m in stk)
    mbar = max(abs(m) for q, k, W, m in stk)
    Mbar = max(_opn(W) * math.sqrt(d) for q, k, W, m in stk)
    return yt, yh, L, mbar, Mbar


@MID
@given(SEED, st.integers(1, 3), st.integers(1, 5), FLOAT(1e-4, 0.2))
def test_alibi_rope_model_bound(seed, n, T, theta):
    """EP.alibi_rope_model_bound: n causal ALiBi layers over T positions on the frozen-pair RoPE host; with
    m_bar T^3 theta^2 <= 3 the head's logits (Wh x_{T-1}, Lh = max row norm) are within
    Lh (prod L_i) n (2/3) m_bar T^3 theta^2 M_bar."""
    d = 4
    x = torch.tensor(rng(seed + 3).standard_normal((T, d)), dtype=F64)
    yt, yh, L, mbar, Mbar = _alibi_model(seed, n, T, d, theta, x)
    assume(mbar * T ** 3 * theta ** 2 <= 3)
    Wh = torch.tensor(rng(seed + 4).standard_normal((5, d)), dtype=F64)
    Lh = float(Wh.norm(dim=1).max())
    err = float((Wh @ (yh[-1] - yt[-1])).abs().max())
    bound = Lh * L * n * (2 / 3) * mbar * T ** 3 * theta ** 2 * Mbar
    assert err <= bound + Lh * L * n * (fp(Mbar, 100) + 2 * Mbar * (mbar / theta + 10) * 100 * U)


@MID
@given(SEED, st.integers(1, 3), st.integers(1, 5))
def test_alibi_rope_model_limit(seed, n, T):
    """EP.alibi_rope_model_limit: as theta -> 0 the RoPE host stack converges to the ALiBi stack (theta =
    10^-k, k = 1..5: under the O(theta^2) bound, and the error at theta = 1e-5 is below 1e-8)."""
    d = 4
    x = torch.tensor(rng(seed + 3).standard_normal((T, d)), dtype=F64)
    for theta in 10.0 ** -np.arange(1, 6):
        yt, yh, L, mbar, Mbar = _alibi_model(seed, n, T, d, theta, x)
        if mbar * T ** 3 * theta ** 2 <= 3:
            err = float((yh - yt).norm(dim=1).max())
            assert err <= L * n * (2 / 3) * mbar * T ** 3 * theta ** 2 * Mbar + L * n * (
                fp(Mbar, 100) + 2 * Mbar * (mbar / theta + 10) * 100 * U)
    assert err <= 1e-8


class _AlibiLM(torch.nn.Module):
    """Toy LM over the last T tokens (front-padded with token 0): ALiBi stack (target, theta None) or its
    frozen-pair RoPE host, head Wh on the last position."""

    def __init__(self, E, stk, Wh, T, theta=None):
        super().__init__()
        self.E, self.stk, self.Wh, self.T, self.theta = E, stk, Wh, T, theta

    def logits_of(self, ids):
        t = ids[0, -self.T:].tolist()
        t = [0] * (self.T - len(t)) + t
        x = self.E[t]
        for q, k, W, m in self.stk:
            x = _alibi_layer(x, q, k, W, m, self.theta)
        return self.Wh @ x[-1]

    def forward(self, ids, use_cache=None, **kw):
        if use_cache:
            raise TypeError("no KV cache")
        z = self.logits_of(ids)
        out = torch.zeros(1, ids.shape[1], z.shape[0], dtype=F64)
        out[0, -1] = z
        return _types.SimpleNamespace(logits=out)


@MID
@given(SEED, st.integers(1, 2), st.integers(2, 4))
def test_alibi_rope_greedy(seed, n, T):
    """EP.alibi_rope_greedy: with the ALiBi LM's greedy margin delta along its trajectory, every theta with
    m_bar T^3 theta^2 <= 3 and 4 Lh (prod L) n m_bar T^3 M_bar theta^2 < 3 delta makes the frozen-pair RoPE
    host LM generate the same tokens (act_swap_agreement.greedy)."""
    d, V = 4, 6
    r = rng(seed)
    E = torch.tensor(r.standard_normal((V, d)), dtype=F64)
    Wh = torch.tensor(r.standard_normal((V, d)) * 3, dtype=F64)
    stk = _alibi_stack(seed, n, T, d)
    ids = torch.tensor([[seed % V]])
    tgt = _AlibiLM(E, stk, Wh, T)
    toks = asa.greedy(tgt, ids, 4)
    seq = torch.cat([ids, toks], 1)
    delta = min(float(z[0] - z[1]) for z in (tgt.logits_of(seq[:, :t]).sort(descending=True).values
                                             for t in range(1, seq.shape[1])))
    assume(delta > 1e-3)
    L = math.prod(1 + _opn(W) for q, k, W, m in stk)
    mbar = max(abs(m) for q, k, W, m in stk)
    Mbar = max(_opn(W) * math.sqrt(d) for q, k, W, m in stk)
    Lh = float(Wh.norm(dim=1).max())
    th2 = 0.99 * min(3 / (mbar * T ** 3), 3 * delta / (4 * Lh * L * n * mbar * T ** 3 * Mbar))
    host = _AlibiLM(E, stk, Wh, T, theta=math.sqrt(th2))
    assert torch.equal(asa.greedy(host, ids, 4), toks)


# =====================================================================================================
# ===== EpsLever.lean =====
# =====================================================================================================
#
# The construction: hf/eps_lever_agreement.host_weights (branch measure-batch2, copied here with
# graph/eps_lever.py). A host norm the target lacks reads its input x 2^-a (q_proj / k_proj / o_proj /
# down_proj rows x 2^-a) and its gain is 2^a sqrt(E): S = 2^a, eps = E, w = 1 in EP.epsLeverNorm.
# transformers' RMSNorm modules compute in float32 even in a float64 model, so every check on a stock
# module carries float32 slack (U32), and the test says so.

from graph import eps_lever as gel                # noqa: E402
from hf import eps_lever_agreement as ela         # noqa: E402
from attn.qk_norm import rmsnorm as _rms64        # noqa: E402


MODEL = settings(max_examples=8, deadline=None,                   # whole stock models, several hosts each
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])


def _lever_target(kind, seed, eps, L=2):
    """A tiny stock target with O(1) weights: Llama (for a Qwen3 qk-norm host) or Gemma (for a Gemma-2
    post-norm host). Random init is std 0.02, which leaves every switched-off norm input far below sqrt(eps)
    at every a; the weights are redrawn at scale 1/sqrt(fan_in) (q, k x 3) so the lever's error is visible."""
    torch.manual_seed(seed)
    common = dict(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=L, num_attention_heads=4,
                  head_dim=8, max_position_embeddings=64, rms_norm_eps=eps, attn_implementation="eager")
    if kind == "qwen3":
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(num_key_value_heads=2, tie_word_embeddings=False, **common)
        m = LlamaForCausalLM(cfg)
    else:
        from transformers import GemmaConfig, GemmaForCausalLM
        cfg = GemmaConfig(num_key_value_heads=1, hidden_activation="gelu_pytorch_tanh", **common)
        m = GemmaForCausalLM(cfg)
    m = m.double().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2:
                p.normal_(0, 1 / math.sqrt(p.shape[1]))
                if n.endswith(("q_proj.weight", "k_proj.weight")):
                    p.mul_(3)
            else:
                p.uniform_(0.5, 1.5) if kind == "qwen3" else p.uniform_(-0.5, 0.5)
    return cfg, m


def _lever_host(kind, cfg, tgt, a):
    """The host the repo writes: hf/eps_lever_agreement.host_config + host_weights + write_host, float64."""
    t = ela.tdims(cfg)
    W = ela.host_weights(kind, ela.tfamily(cfg), t, tgt.state_dict(), a)
    host = ela._build(kind, ela.host_config(kind, t, t["eps"]), F64, "cpu")
    ela.write_host(host, W, F64)
    return host, W, t


def _lever_factor(x, eps, S):
    """EP.leverFactor: 1 / sqrt(1 + ||x||_2^2 / (d S^2 eps)), per row."""
    d = x.shape[-1]
    return 1.0 / torch.sqrt(1.0 + x.pow(2).sum(-1, keepdim=True) / (d * S * S * eps))


@MID
@given(SEED, st.integers(0, 12), FLOAT(1e-3, 0.5), st.sampled_from([4, 8, 16]))
def test_epsLever_eq(seed, a, eps, hd):
    """EP.epsLever_eq: S sqrt(eps) N_eps(w, x/S) = w (.) x / sqrt(1 + ||x||^2/(d S^2 eps)). Checked on the stock
    Qwen3RMSNorm module with the gain hf/eps_lever_agreement.host_weights writes into q_norm (2^a sqrt(E)), fed
    the query x 2^-a that host_weights' q_proj (W_q x 2^-a) produces. The module rounds to float32: slack
    8 U32 |x| (the fp32 cast of x, the mean square, rsqrt, and the products)."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    S = 2.0 ** a
    g = torch.full((hd,), S * math.sqrt(eps), dtype=F64)        # = host_weights' g_off (s_t / s_h = 1)
    norm = Qwen3RMSNorm(hd, eps=eps).double()
    with torch.no_grad():
        norm.weight.copy_(g)
        x = torch.randn(6, hd, dtype=F64, generator=gen(seed)) * float(rng(seed).uniform(0.01, 30))
        got = norm(x / S)
    want = _lever_factor(x, eps, S) * x
    assert torch.all((got - want).abs() <= 8 * U32 * x.abs().max(-1, keepdim=True).values + 1e-300)


@SLOW
@given(SEED, st.integers(0, 10))
def test_epsLever_eq_host_weights(seed, a):
    """EP.epsLever_eq, the weights: hf/eps_lever_agreement.host_weights writes q_proj = W_q 2^-a, k_proj =
    W_k 2^-a and q_norm = k_norm = 2^a sqrt(E) (Qwen3 host, Llama target, s_t = s_h), i.e. exactly the levered
    norm of EP.epsLeverNorm with S = 2^a, eps = E, w = 1 -- bit for bit (powers of two)."""
    cfg, tgt = _lever_target("qwen3", seed, 0.05, L=1)
    t = ela.tdims(cfg)
    sd = tgt.state_dict()
    W = ela.host_weights("qwen3", "llama", t, sd, a)
    p = "model.layers.0.self_attn."
    assert torch.equal(W[p + "q_proj.weight"], sd[p + "q_proj.weight"] * 2.0 ** -a)
    assert torch.equal(W[p + "k_proj.weight"], sd[p + "k_proj.weight"] * 2.0 ** -a)
    g = 2.0 ** a * math.sqrt(t["eps"])
    assert torch.all(W[p + "q_norm.weight"] == g) and torch.all(W[p + "k_norm.weight"] == g)
    assert torch.equal(W[p + "v_proj.weight"], sd[p + "v_proj.weight"])        # bv = false: v untouched


@FAST
@given(SEED, st.integers(1, 64), st.integers(-4, 16), FLOAT(1e-6, 1.0))
def test_epsLever_coord_bound(seed, d, a, eps):
    """EP.epsLever_coord_bound: |S sqrt(eps) N_eps(w, x/S)_i - w_i x_i| <= |w_i x_i| u/2, u = ||x||^2/(d S^2 eps).
    The levered norm as the repo computes it in float64 (attn/qk_norm.rmsnorm, gain S sqrt(eps) w, input x/S);
    slack fp(|w x|, 20)."""
    r = rng(seed)
    S = 2.0 ** a
    x = torch.from_numpy(r.standard_normal(d) * 10 ** r.uniform(-3, 2))
    w = torch.from_numpy(r.uniform(-2, 2, d))
    got = S * math.sqrt(eps) * w * _rms64(x / S, eps)
    u = x.pow(2).sum() / (d * S * S * eps)
    assert torch.all((got - w * x).abs() <= (w * x).abs() * u / 2 + fp((w * x).abs().max(), 20))


@FAST
@given(SEED, st.integers(1, 64), st.integers(-4, 16), FLOAT(1e-6, 1.0))
def test_epsLever_bound(seed, d, a, eps):
    """EP.epsLever_bound: ||S sqrt(eps) N_eps(w, x/S) - w (.) x||_inf <= ||w||_inf ||x||_inf^3 / (2 S^2 eps)
    (attn/qk_norm.rmsnorm in float64; slack fp(|w x|, 20))."""
    r = rng(seed)
    S = 2.0 ** a
    x = torch.from_numpy(r.standard_normal(d) * 10 ** r.uniform(-3, 2))
    w = torch.from_numpy(r.uniform(-2, 2, d))
    got = S * math.sqrt(eps) * w * _rms64(x / S, eps)
    rhs = w.abs().max() * x.abs().max() ** 3 / (2 * S * S * eps)
    assert (got - w * x).abs().max() <= rhs + fp((w * x).abs().max(), 20)


@FAST
@given(SEED, st.integers(1, 32), FLOAT(1e-6, 1.0), FLOAT(1e-3, 1e3))
def test_epsLever_uniform(seed, d, eps, ms_max):
    """EP.epsLever_tendsto, EP.epsLever_uniform: the lever tends to w (.) x as S -> inf, uniformly on balls only.
    graph/eps_lever.a_needed(ms_max, eps, rho) is the repo's choice of S = 2^a for a ball {ms(x) <= ms_max}:
    every x in the ball then has u <= rho, so the relative error is <= rho/2 uniformly (and -> 0 as rho -> 0).
    Not uniform on R^n: an x outside the ball (ms = 4^k ms_max) at the same a is off by more than the ball's
    bound (relative error 1 - 1/sqrt(1 + u) with u = 4^k rho)."""
    r = rng(seed)
    for rho in (2.0 ** -5, 2.0 ** -12, 2.0 ** -20):
        a = gel.a_needed(ms_max, eps, rho)
        S = 2.0 ** a
        x = torch.from_numpy(r.standard_normal(d))
        x = x * math.sqrt(ms_max / x.pow(2).mean().item()) * r.uniform(0, 1)       # in the ball
        got = S * math.sqrt(eps) * _rms64(x / S, eps)
        assert x.pow(2).mean().item() / (S * S * eps) <= rho * (1 + 8 * U)
        assert (got - x).abs().max() <= x.abs().max() * rho / 2 + fp(x.abs().max(), 20)
    big = x * 2.0 ** 12                                                              # far outside the ball
    u = big.pow(2).mean().item() / (S * S * eps)
    assume(u > 1.0)
    rel = ((S * math.sqrt(eps) * _rms64(big / S, eps) - big).abs().max() / big.abs().max()).item()
    assert rel > rho / 2 and rel >= 1 - 1 / math.sqrt(1 + u) - 1e-9


def test_mismatch_eps_lever_exact_flag():
    """W4-1 vs EP.epsLever_eq (round 3, batch 3b: the label corrected -- was a recorded MISMATCH, graph/eps_lever.py set
    EXACT = True, "exact to the kernel's rounding, not a limit"): the module labels the lever a LIMIT. Over the reals
    S sqrt(eps) N_eps(1, x/S) = x / sqrt(1 + u), and u > 0 for every x != 0, so it is never exact at a finite S. It
    is exact only through float32 rounding (fl(ms + e) = e once u < 2^-25). At the module's own a_needed(..,
    RHO_EXACT) a float64 host misses by u/2 ~ 2^-26 relative, ~10^8 float64 roundings -- the miss the label now
    states."""
    assert gel.EXACT is False and gel.KIND == "limit" and gel.RHO_EXACT == 2.0 ** -25
    assert "LIMIT" in gel.PROPOSED["QKNORM_ORDER / QKNORM_REGIME_PAYABLE"]
    eps, ms_max = 1e-6, 8.4e3                         # gemma-2b's MLP outputs (the module's own example)
    a = gel.a_needed(ms_max, eps)
    S = 2.0 ** a
    x = torch.full((16,), math.sqrt(ms_max), dtype=F64)
    got = S * math.sqrt(eps) * _rms64(x / S, eps)
    rel = ((got - x).abs().max() / x.abs().max()).item()
    u = ms_max / (S * S * eps)
    assert abs(rel - (1 - 1 / math.sqrt(1 + u))) <= 1e-15 and rel > 1e6 * U
    assert gel.host_norm_off("qk_norm", eps, "float64")       # free in fp64 -- as a limit, which the label says


def _attn_out(model, layer, ids):
    """layer `layer`'s self_attn output, and its q / k / v projection outputs (forward hooks)."""
    cap = {}
    at = model.model.layers[layer].self_attn
    hs = [at.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o[0]))]
    for n in ("q_proj", "k_proj", "v_proj"):
        hs.append(getattr(at, n).register_forward_hook(lambda m, i, o, n=n: cap.__setitem__(n, o)))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    return cap


@SLOW
@given(SEED, st.integers(8, 20), st.integers(4, 16))
def test_leverAttn_layer_bound(seed, extra, T):
    """EP.leverAttn_layer_bound (qk-norm switched off, bq = bk = true, bv = false): the Qwen3 host layer that
    hf/eps_lever_agreement.host_weights writes from a Llama target is within
        sum_h O_h V_h (2 C_B Q_h K_h (Q_h^2 + K_h^2) + V_h^2/2) / (eps S^2)
    of the target layer (sup norm over tokens and coordinates), once C_B Q_h K_h (Q_h^2 + K_h^2) <= eps S^2.
    Per head h: Q_h, K_h, V_h = the sup norms of the (unlevered) q, k, v vectors over the tokens, O_h = the
    inf->inf norm of o_proj's columns for head h; C_B = scale * head_dim bounds the RoPE dot product in the
    sup norm (|<Rq, Rk>| <= ||q||_2 ||k||_2 <= hd ||q|| ||k||). One head per Lean statement, summed over the
    heads. Both layers see the same input (same embedding and input norm). S = 2^a with a past the threshold
    by `extra`/2 binades; slack: the stock modules' float32 rounding (below)."""
    eps = 0.05
    cfg, tgt = _lever_target("qwen3", seed, eps, L=1)
    ids = torch.randint(0, 50, (1, T), generator=gen(seed))
    ct = _attn_out(tgt, 0, ids)
    H, KV, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q, k, v = ct["q_proj"][0].view(T, H, hd), ct["k_proj"][0].view(T, KV, hd), ct["v_proj"][0].view(T, KV, hd)
    Wo = tgt.model.layers[0].self_attn.o_proj.weight.detach().view(-1, H, hd)
    CB = hd ** -0.5 * hd
    Q = q.abs().amax((0, 2))
    Kh = k.abs().amax((0, 2)).repeat_interleave(H // KV)
    Vh = v.abs().amax((0, 2)).repeat_interleave(H // KV)
    Oh = Wo.abs().sum(-1).amax(0)
    need = (CB * Q * Kh * (Q ** 2 + Kh ** 2)).max().item() / eps
    a = max(0, math.ceil(math.log2(need) / 2)) + extra // 2
    S = 2.0 ** a
    assert (CB * Q * Kh * (Q ** 2 + Kh ** 2) <= eps * S * S).all()
    host, _, _ = _lever_host("qwen3", cfg, tgt, a)
    ch = _attn_out(host, 0, ids)
    bound = (Oh * Vh * (2 * CB * Q * Kh * (Q ** 2 + Kh ** 2) + Vh ** 2 / 2)).sum().item() / (eps * S * S)
    diff = (ch["o"] - ct["o"]).abs().max().item()
    # float32 in the stock modules (the q / k norms' outputs, eager attention's softmax): each logit moves by
    # <= 8 U32 C_B Q K and each weight by <= 8 U32, so the head output by <= (16 U32 C_B Q K + 8 U32) O V
    slack = (Oh * Vh * (16 * U32 * CB * Q * Kh + 8 * U32)).sum().item()
    assert diff <= bound + slack


@pytest.mark.parametrize("kind", ["qwen3", "gemma2"])
@MODEL
@given(seed=SEED)
def test_leverAttn_model_bound_and_postNorm_model_bound(kind, seed):
    """EP.leverAttn_model_bound, EP.leverAttn_model_limit (kind qwen3: qk-norm switched off) and
    EP.postNorm_model_bound, EP.postNorm_model_limit (kind gemma2: sandwich post-norms switched off), with the
    generic EP.lever_model_bound / EP.lever_model_limit: the logits are within C / S^2, C = Lh (prod L)(sum C_i),
    and tend to the target's as S -> inf. The Lipschitz constants are hypotheses, not computed, so the test
    checks the bound's form on the repo's whole host (hf/eps_lever_agreement, Llama -> Qwen3, Gemma -> Gemma-2,
    float64): the logit gap times S^2 is the same constant at a = 8 and a = 10 (the 1/S^2 rate, within 10%:
    the next term is O(1/S^4)), and at a = 22 (u < 2^-25: the stock float32 norms are linear) the gap is at
    the float32 floor, <= 64 U32 max|logit|. That floor gap is the rounding noise allowed in the ratio."""
    cfg, tgt = _lever_target(kind, seed, 0.05)
    ids = torch.randint(0, 50, (1, 12), generator=gen(seed))
    with torch.no_grad():
        zt = tgt(ids).logits
        gaps = {}
        for a in (8, 10, 22):
            host, _, _ = _lever_host(kind, cfg, tgt, a)
            gaps[a] = (host(ids).logits - zt).abs().max().item()
    noise = gaps[22]
    assert noise <= 64 * U32 * zt.abs().max().item()
    assume(gaps[10] > 100 * noise)
    assert abs(gaps[10] * 4 ** 10 - gaps[8] * 4 ** 8) <= 0.1 * gaps[8] * 4 ** 8 + 2 * noise * 4 ** 10


@pytest.mark.parametrize("kind", ["qwen3", "gemma2"])
@MODEL
@given(seed=SEED, a=st.integers(10, 24))
def test_leverAttn_greedy_and_postNorm_greedy(kind, seed, a):
    """EP.leverAttn_greedy, EP.postNorm_greedy, EP.lever_greedy: once the host's logits are within less than
    half the target's greedy margin along the target's own trajectory, the host generates exactly the target's
    tokens. Checked on the repo's host (hf/eps_lever_agreement) with act/act_swap_agreement.greedy: whenever the
    measured gap is below margin/2 at every step of the target's 16-token greedy continuation, the two
    continuations are identical."""
    from act.act_swap_agreement import greedy
    cfg, tgt = _lever_target(kind, seed, 0.05)
    host, _, _ = _lever_host(kind, cfg, tgt, a)
    ids = torch.randint(0, 50, (1, 6), generator=gen(seed))
    ct = greedy(tgt, ids, 16)
    seq = torch.cat([ids, ct], 1)
    with torch.no_grad():
        zt, zh = tgt(seq).logits[0, 5:-1], host(seq).logits[0, 5:-1]
    top2 = zt.topk(2, -1).values
    margin = (top2[:, 0] - top2[:, 1]).min().item()
    gap = (zh - zt).abs().max().item()
    assume(gap < margin / 2)
    assert torch.equal(greedy(host, ids, 16), ct)


@SLOW
@given(SEED, st.integers(2, 14), st.integers(2, 12))
def test_postNorm_layer_bound(seed, a, T):
    """EP.postNorm_layer_bound: x + S sqrt(eps) N_eps(1, y/S) is within R^3/(2 eps S^2) of x + y when every
    token's sublayer output has ||y||_inf <= R. On the Gemma-2 host hf/eps_lever_agreement.host_weights writes
    from a Gemma target: o_proj = W_o 2^-a (the sublayer writes y/S) and post_attention_layernorm gain
    (1 + w) = 2^a sqrt(E) (asserted), applied through the stock Gemma2RMSNorm to the target's own attention
    output y. Gemma2RMSNorm computes in float32 (1 + w included): slack 16 U32 R."""
    eps = 0.05
    cfg, tgt = _lever_target("gemma2", seed, eps, L=1)
    t = ela.tdims(cfg)
    W = ela.host_weights("gemma2", "gemma", t, tgt.state_dict(), a)
    p = "model.layers.0."
    S = 2.0 ** a
    assert torch.equal(W[p + "self_attn.o_proj.weight"], tgt.state_dict()[p + "self_attn.o_proj.weight"] / S)
    assert torch.allclose(W[p + "post_attention_layernorm.weight"] + 1, torch.full((32,), S * math.sqrt(eps),
                                                                                 dtype=F64), rtol=4 * U, atol=0)
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
    pn = Gemma2RMSNorm(32, eps=eps).double()
    with torch.no_grad():
        pn.weight.copy_(W[p + "post_attention_layernorm.weight"])
        y = _attn_out(tgt, 0, torch.randint(0, 50, (1, T), generator=gen(seed)))["o"][0]
        got = pn(y / S)
    R = y.abs().max(-1, keepdim=True).values
    assert torch.all((got - y).abs() <= R ** 3 / (2 * eps * S * S) + 16 * U32 * R)


# =====================================================================================================
# ===== LimitInstances2.lean =====
# =====================================================================================================

from hf.local_full import band_mask as _band      # noqa: E402
from attn.qk_norm import attend as _attend        # noqa: E402
from attn.softcap import softcap as _softcap      # noqa: E402
from attn.attention_sink import sink_attention as _sink_attention   # noqa: E402


@MID
@given(SEED, st.integers(2, 16), st.integers(1, 6), FLOAT(0.1, 4))
def test_attn_extra_keys_bound(seed, T, W, sc):
    """EP.attn_extra_keys_bound: attention over S with logits t equal to s on S' (subset of S) is within
    2 M (sum_{S \\ S'} e^t) / (sum_{S'} e^s) of attention over S' with logits s (values bounded by M).
    S = the causal set of attn/qk_norm.attend, S' = the band of hf/local_full.band_mask; t on S \\ S' arbitrary.
    Sup norm per query; slack fp(M, 10 T)."""
    g = gen(seed)
    allow = _band(T, W)[0, 0] == 0
    s = torch.randn(T, T, dtype=F64, generator=g) * sc
    t = torch.where(allow, s, torch.randn(T, T, dtype=F64, generator=g) * sc)
    v = torch.randn(T, 3, dtype=F64, generator=g)
    M = v.abs().max().item()
    host = _attend(t) @ v
    targ = torch.softmax(s.masked_fill(~allow, float("-inf")), -1) @ v
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    extra = torch.where(causal & ~allow, t.exp(), torch.zeros_like(t)).sum(-1)
    Z = torch.where(allow, s.exp(), torch.zeros_like(s)).sum(-1)
    assert torch.all((host - targ).abs().max(-1).values <= 2 * M * extra / Z + fp(M, 10 * T))


@MID
@given(SEED, st.integers(2, 16), st.integers(1, 6), FLOAT(0.1, 3), FLOAT(0, 40))
def test_mask_layer_bound(seed, T, W, R, B):
    """EP.mask_layer_bound: full attention over all T keys with -B added outside the mask (the host logits
    EP.maskHostLogits) is within 2 M T e^{2R - B} of the masked target, logits |s| <= R, values <= M. The mask is
    hf/local_full.band_mask with its -finfo.max entries clamped to -B (band_mask itself is the B = finfo.max
    end point). Sup norm per query; slack fp(M, 10 T)."""
    g = gen(seed)
    s = (2 * torch.rand(T, T, dtype=F64, generator=g) - 1) * R
    v = torch.randn(T, 3, dtype=F64, generator=g)
    M = v.abs().max().item()
    m = _band(T, W)[0, 0]
    host = torch.softmax(s + m.clamp(min=-B), -1) @ v
    targ = torch.softmax(s + m, -1) @ v
    Rm = s.abs().max().item()
    assert torch.all((host - targ).abs().max(-1).values <= 2 * M * T * math.exp(2 * Rm - B) + fp(M, 10 * T))


@MID
@given(SEED, st.integers(1, 32), FLOAT(0.1, 4), FLOAT(-40, 3))
def test_sink_layer_bound(seed, N, R, b):
    """EP.sink_layer_bound: the sink host (extra key, logit b, value 0) is within 2 M e^{b + R} of plain
    attention, logits |s| <= R, values <= M (attn/attention_sink.sink_attention; slack fp(M, 10 N))."""
    g = gen(seed)
    s = (2 * torch.rand(N, dtype=F64, generator=g) - 1) * R
    v = torch.randn(N, 4, dtype=F64, generator=g)
    M = v.abs().max().item()
    gap = (_sink_attention(s, v, b) - torch.softmax(s, 0) @ v).abs().max().item()
    assert gap <= 2 * M * math.exp(b + s.abs().max().item()) + fp(M, 10 * N)


@MID
@given(SEED, st.integers(2, 16), FLOAT(0.05, 5), FLOAT(0.5, 100))
def test_cap_layer_bound(seed, T, R, c):
    """EP.cap_layer_bound: attention over softcapped logits c tanh(s/c) is within (e^{2R^3/(3c^2)} - 1) M of
    attention over s, logits |s| <= R, values <= M (attn/softcap.softcap, attn/qk_norm.attend; also
    EP.cap_const_le's (4/3) R^3 M / c^2 once 2R^3 <= 3c^2). Slack fp(M, 10 T)."""
    g = gen(seed)
    s = (2 * torch.rand(T, T, dtype=F64, generator=g) - 1) * R
    v = torch.randn(T, 3, dtype=F64, generator=g)
    M = v.abs().max().item()
    gap = (_attend(_softcap(s, c)) @ v - _attend(s) @ v).abs().max().item()
    Rm = s.abs().max().item()
    assert gap <= (math.exp(2 * Rm ** 3 / (3 * c * c)) - 1) * M + fp(M, 10 * T)
    if 2 * Rm ** 3 <= 3 * c * c:
        assert gap <= 4 / 3 * Rm ** 3 * M / (c * c) + fp(M, 10 * T)


@MID
@given(SEED, st.integers(2, 8), FLOAT(0, 200))
def test_soft_router_layer_bound(seed, E, beta):
    """EP.soft_router_layer_bound: the soft router sum_e softmax(beta z)_e Y_e (moe/dense_emulates_moe.
    dense_softrouter) is within 2 M (E - 1) e^{-beta delta_r} of the top-1 output Y_{e*} (moe_top1), per token,
    delta_r = that token's router margin z_(1) - z_(2), M = max_e ||Y_e(x)||_inf at that token (the theorem
    applied to the expert outputs frozen at x, which are bounded by M for every input). Slack fp(M, 10 E)."""
    from moe import dense_emulates_moe as dem
    g = gen(seed)
    d, h, Tn = 6, 5, 9
    x = torch.randn(Tn, d, dtype=F64, generator=g)
    W1, W2 = torch.randn(E, d, h, dtype=F64, generator=g), torch.randn(E, h, d, dtype=F64, generator=g)
    Wg = torch.randn(d, E, dtype=F64, generator=g)
    soft, top = dem.dense_softrouter(x, W1, W2, Wg, beta), dem.moe_top1(x, W1, W2, Wg)
    z = (x @ Wg).sort(-1, descending=True).values
    dr = z[:, 0] - z[:, 1]
    M = dem.experts_all(x, W1, W2).abs().amax((1, 2))
    gap = (soft - top).abs().max(-1).values
    assert torch.all(gap <= 2 * M * (E - 1) * torch.exp(-beta * dr) + fp(M.max(), 10 * E))


@MODEL
@given(SEED, st.integers(2, 6), st.integers(8, 14))
def test_mask_model_bound(seed, W, seq):
    """EP.mask_model_bound, EP.mask_model_limit: a whole stack whose every layer's hard mask is replaced by -B
    is within Lh (prod L) n 2 M T e^{2R - B} of the masked stack, and tends to it as B -> inf. On the repo's
    two-layer stock Qwen2Model of hf/local_full (full-attention config, band mask clamped to -B; the target is
    the sliding-window config with the same weights). The Lipschitz constants are hypotheses, so the test
    checks the e^{-B} form: gap(B) e^B converges (from below: the leaked mass is L e^{-B} / (Z + L e^{-B}))
    to a finite constant, so gap(10) e^10 <= 1.5 gap(4) e^4 (sound while L e^{-4} <= Z / 2, true here), the
    gaps decrease, and at B = 60 the gap is at the float32 floor of eager attention's softmax (16 U32 max|out|,
    also the slack on the scaled gaps)."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    from hf import local_full
    torch.manual_seed(seed)
    full = Qwen2Model(local_full._cfg(False, W)).double().eval()
    win = Qwen2Model(local_full._cfg(True, W)).double().eval()
    win.load_state_dict(full.state_dict())
    x = torch.randn(1, seq, 64, dtype=F64, generator=gen(seed))
    m = local_full.band_mask(seq, W)
    with torch.no_grad():
        ref = win(inputs_embeds=x).last_hidden_state
        gap = {B: (full(inputs_embeds=x, attention_mask=m.clamp(min=-B)).last_hidden_state - ref).abs().max().item()
               for B in (4, 8, 10, 60)}
    noise = 16 * U32 * ref.abs().max().item()
    assert gap[10] * math.exp(10) <= 1.5 * gap[4] * math.exp(4) + noise * math.exp(10)
    assert gap[10] <= gap[8] + noise and gap[8] <= gap[4] + noise
    assert gap[60] <= noise


import types                                      # noqa: E402

from moe import dense_host_prenorm as dhp        # noqa: E402


def _host_norm_read(h, s, delta, eps, w):
    """moe/dense_host_prenorm.HostNorm over [h; delta s] (its algebraic form of the padded host norm with
    eps' = eps d/(d+e), weight w sqrt(d/(d+e))): the stream read and the carried read x 1/delta
    (the carried channel over the norm's own rms, side.r, as HostMoE reads it)."""
    side = dhp.Side(s.shape[-1])
    side.c = delta * s
    norm = dhp.HostNorm(types.SimpleNamespace(weight=w, variance_epsilon=eps), side)
    out = norm(h)
    return out, (side.c / side.r) / delta





@MID
@given(SEED, st.integers(1, 32), st.integers(1, 8), FLOAT(1e-4, 1.0), st.integers(1, 12))
def test_vanishing_carry_rate(seed, d, e, eps, k):
    """EP.vanishing_carry_rate: with the carried vector written at scale delta, the host norm's stream read and
    its carried read x 1/delta are within relative error delta^2 ||s||^2 / (2 d eps) of the target's
    RMSNorm_eps(x) and of s / rms_t(x). On moe/dense_host_prenorm.HostNorm (the norm of the dense-host-for-MoE
    construction, gamma = delta). HostNorm casts the stream to float32: slack 8 U32 of the read."""
    r = rng(seed)
    delta = 10.0 ** (-k / 2)
    h = torch.from_numpy(r.standard_normal((5, d)) * 10 ** r.uniform(-2, 1))
    s = torch.from_numpy(r.standard_normal((5, e)))
    w = torch.from_numpy(r.uniform(0.5, 1.5, d))
    out, carried = _host_norm_read(h, s, delta, eps, w)
    rt = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    want, want_c = w * h / rt, s / rt
    rel = delta ** 2 * s.pow(2).sum(-1, keepdim=True) / (2 * d * eps)
    assert torch.all((out - want).abs() <= want.abs() * rel + 8 * U32 * want.abs().max(-1, keepdim=True).values)
    assert torch.all((carried - want_c).abs() <= want_c.abs() * rel + 8 * U32 * want_c.abs())


@MID
@given(SEED, st.integers(1, 16), st.integers(1, 6), FLOAT(1e-4, 1.0))
def test_scaled_read_layer_bound(seed, d, e, eps):
    """EP.scaled_read_layer_bound: a consumer K-Lipschitz in its read, read bounded by M, |lambda - 1| <= rho:
    within K M rho. The read is moe/dense_host_prenorm.HostNorm's stream read, lambda = rms_t / rms_host its
    per-token factor (rho = delta^2 ||s||^2 / (2 d eps), EP.vanishing_carry_rate), and the consumer is the
    compensator form G x y = x + A y (K = ||A||_inf->inf). Slack 8 U32 K M (HostNorm is float32)."""
    r = rng(seed)
    delta = 0.3
    h = torch.from_numpy(r.standard_normal((5, d)))
    s = torch.from_numpy(r.standard_normal((5, e)))
    w = torch.from_numpy(r.uniform(0.5, 1.5, d))
    A = torch.from_numpy(r.standard_normal((d, d)))
    out, _ = _host_norm_read(h, s, delta, eps, w)
    read = w * h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    K, M = A.abs().sum(-1).max().item(), read.abs().max(-1, keepdim=True).values
    rho = delta ** 2 * s.pow(2).sum(-1, keepdim=True) / (2 * d * eps)
    gap = ((h + out @ A.T) - (h + read @ A.T)).abs().max(-1, keepdim=True).values
    assert torch.all(gap <= K * M * rho + 8 * U32 * K * M)


# =====================================================================================================
# ===== PreNormRescue.lean =====
# =====================================================================================================


@MID
@given(SEED, st.integers(1, 32), st.integers(1, 8), FLOAT(1e-4, 1.0))
def test_vanishing_carry_limit(seed, d, e, eps):
    """EP.vanishing_carry_limit: as delta -> 0 the host norm over [x; delta s] tends to the target's RMSNorm
    on the stream, and the carried read x 1/delta to s / rms_t(x). On moe/dense_host_prenorm.HostNorm: the
    errors shrink with delta (by EP.vanishing_carry_rate, ~delta^2) and at the construction's own scale
    GAMMA = 1e-8 both reads are the limit to float32 rounding (HostNorm casts to float32: 8 U32)."""
    r = rng(seed)
    h = torch.from_numpy(r.standard_normal((5, d)) * 10 ** r.uniform(-2, 1))
    s = torch.from_numpy(r.standard_normal((5, e)) * 10 ** r.uniform(-1, 1))
    w = torch.from_numpy(r.uniform(0.5, 1.5, d))
    rt = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    want, want_c = w * h / rt, s / rt
    errs = []
    for delta in (1.0, 1e-1, 1e-2, dhp.GAMMA):
        out, carried = _host_norm_read(h, s, delta, eps, w)
        errs.append(((out - want).abs().max().item(), (carried - want_c).abs().max().item()))
    for (a0, c0), (a1, c1) in zip(errs, errs[1:]):
        assert a1 <= a0 + 8 * U32 * want.abs().max().item() and c1 <= c0 + 8 * U32 * want_c.abs().max().item()
    assert errs[-1][0] <= 8 * U32 * want.abs().max().item() and errs[-1][1] <= 8 * U32 * want_c.abs().max().item()


# =====================================================================================================
# ===== RandomFeatures.lean =====
# =====================================================================================================
#
# attn/linear_softmax.favor_features: phi(x) = exp(W x - ||x||^2/2) / sqrt(m), W ~ N(0, I). The estimator
# EP.rfEst is phi(q) . phi(k). Expectations over the Gaussian rows are computed by Gauss-Hermite quadrature
# (probabilists' weights, 60 nodes per dimension) applied to the repo's feature map at the nodes: for
# e^{a w} with |a| <= 4 the quadrature error is below a^{2n}/(2^n n! (2n)!) ~ 1e-100, so the only error is
# float64 summation (relative 1e-12 is generous).

from attn import linear_softmax as lsm           # noqa: E402

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(60)
_GH_W = _GH_W / math.sqrt(2 * math.pi)


def _grid(d):
    """Gauss-Hermite product grid over R^d: nodes (n^d, d), weights (n^d,)."""
    X = np.stack(np.meshgrid(*[_GH_X] * d, indexing="ij"), -1).reshape(-1, d)
    Wt = np.prod(np.stack(np.meshgrid(*[_GH_W] * d, indexing="ij"), -1).reshape(-1, d), -1)
    return X, Wt


@FAST
@given(SEED, st.integers(1, 2))
def test_rf_unbiased(seed, d):
    """EP.rf_unbiased (with EP.gauss_exp_integral): E[phi(q).phi(k)] = exp(q.k) for the repo's positive features
    (attn/linear_softmax.favor_features), each feature row's expectation by Gauss-Hermite quadrature; |q|,|k|
    entries <= 1. Relative tolerance 1e-12 (float64 summation of 3600 terms)."""
    r = rng(seed)
    q, k = r.uniform(-1, 1, d), r.uniform(-1, 1, d)
    nodes, wts = _grid(d)
    phi = lsm.favor_features(np.stack([q, k]), nodes)              # (2, m), m = number of nodes, /sqrt(m)
    Y = phi[0] * phi[1] * nodes.shape[0]                           # one feature's product, per node
    want = math.exp(q @ k)
    assert abs((wts * Y).sum() - want) <= 1e-12 * want


@MID
@given(SEED, st.integers(1, 2))
def test_rf_mse_and_variance(seed, m):
    """EP.rf_mse, EP.rf_variance: E[(rfEst - e^{q.k})^2] = Var(rfEst) = (e^{|q|^2 + |k|^2 + 4 q.k} - e^{2 q.k}) / m
    for m random features (attn/linear_softmax.favor_features with an m-row W), d = 1, by product quadrature
    over the m Gaussian rows (60^m nodes). Relative tolerance 1e-10."""
    r = rng(seed)
    q, k = r.uniform(-1, 1, 1), r.uniform(-1, 1, 1)
    X = np.stack([q, k])
    nodes, wts = _grid(m)                                          # each node = one draw of the m rows
    est = np.array([(lambda p: p[0] @ p[1])(lsm.favor_features(X, row[:, None])) for row in nodes])
    e = math.exp(q @ k)
    C = math.exp(q @ q + k @ k + 4 * (q @ k)) - math.exp(2 * (q @ k))
    mean = (wts * est).sum()
    mse = (wts * (est - e) ** 2).sum()
    var = (wts * est ** 2).sum() - mean ** 2
    assert abs(mse - C / m) <= 1e-10 * (C / m + e * e)
    assert abs(var - C / m) <= 1e-10 * (C / m + e * e)
    assert abs(C - math.exp(2 * (q @ k)) * (math.exp(((q + k) @ (q + k))) - 1)) <= 1e-12 * (C + e * e)


@MID
@given(SEED, st.integers(1, 4), st.integers(1, 64), FLOAT(0.05, 3.0))
def test_rf_chebyshev(seed, d, m, delta):
    """EP.rf_chebyshev: P(|rfEst - e^{q.k}| >= delta) <= C(q,k) / (m delta^2). Monte Carlo over N = 20000 draws
    of the m rows (attn/linear_softmax.favor_features); the empirical frequency may exceed the probability by
    sqrt(ln(1e9) / (2N)) = 0.032 with probability <= 1e-9 (Hoeffding), so that is the slack."""
    r = rng(seed)
    q, k = r.uniform(-1, 1, d) / math.sqrt(d), r.uniform(-1, 1, d) / math.sqrt(d)
    N = 20000
    Wrows = r.standard_normal((N * m, d))
    phi = lsm.favor_features(np.stack([q, k]), Wrows) * math.sqrt(N * m)     # per-row features, unnormalised
    est = (phi[0] * phi[1]).reshape(N, m).mean(-1)
    e = math.exp(q @ k)
    C = math.exp(q @ q + k @ k + 4 * (q @ k)) - math.exp(2 * (q @ k))
    freq = float((np.abs(est - e) >= delta).mean())
    assert freq <= C / (m * delta ** 2) + math.sqrt(math.log(1e9) / (2 * N))


# =====================================================================================================
# KVPack.lean
# =====================================================================================================

from verify.levers import constructions as lev   # noqa: E402  (copied from branch frontier-levers)


@MID
@given(SEED, st.sampled_from([2, 4, 6, 8, 12, 16]), st.integers(1, 24))
def test_kvPack_rope_attn(seed, dh, T):
    """EP.kvPack_rope_attn (and EP.kvPack_rope_score through it): two RoPE KV heads packed into ONE host
    KV head of width 2 dh, each target head in its own slice, the host frequencies equal to the target's
    on every slice (inv_freq = [inv; inv]), queries zero outside their slice, o_proj reading the value
    slice. The host's attention output equals every target query head's, at every position.
    Built by verify/levers/constructions.kv_pack (branch frontier-levers). Each output is a softmax
    average of values of size ~3 over <= 24 keys after O(dh) products, cos / sin evaluated at angles <= T:
    fp(3, 2e3)."""
    assert lev.kv_pack(dh=dh, T=T, seed=seed % 2 ** 31) <= fp(3, 2e3)


@MID
@given(SEED, st.sampled_from([2, 4, 8, 16]), st.integers(1, 24))
def test_kvPack_nope(seed, dh, T):
    """EP.kvPack_nope: NoPE targets (zero inv_freq on the packed slices): one host KV head of width 2 dh
    reproduces both KV heads' query heads for every input. kv_pack(nope=True). Same rounding as above."""
    assert lev.kv_pack(dh=dh, T=T, seed=seed % 2 ** 31, nope=True) <= fp(3, 2e3)


@FAST
@given(SEED, FLOAT(-3, 3), st.integers(-5, 5), st.integers(-40, 40), st.integers(-40, 40))
def test_kvPack_pair_iff(seed, w, n, i, j):
    """EP.kvPack_pair_iff (the 'if' half, on the repo's rotary): a pair at frequency w' = w + 2 pi n gives
    the same logit as at w for every query / key at all integer positions, through
    verify/levers/constructions.rope (the half-split rotary kv_pack uses). Angles up to |40 (w + 2 pi n)|
    ~ 1.4e3 rad: cos / sin of such arguments are accurate to ~|angle| U each: fp(1.4e3, 64)."""
    q, k = torch.tensor(rng(seed).standard_normal((2, 1, 2)))
    def logit(f):
        inv = torch.tensor([f], dtype=F64)
        return (lev.rope(q, torch.tensor([float(i)], dtype=F64), inv)
                * lev.rope(k, torch.tensor([float(j)], dtype=F64), inv)).sum().item()
    S = float(q.abs().max() * k.abs().max())
    assert abs(logit(w + 2 * math.pi * n) - logit(w)) <= fp(1.4e3, 64) * (1 + S)


@MID
@given(SEED, st.sampled_from([4, 8, 16]), st.integers(0, 1), st.integers(3, 16))
def test_kvPack_rope_iff(seed, dh, which_pair, T):
    """EP.kvPack_rope_iff (both halves on kv_pack): with the target's frequencies repeated on the host's
    slices the pack is exact (test_kvPack_rope_attn); change the host frequency of ONE slice pair (pair
    `which_pair` of block b, moved by 0.5 rad, not a multiple of 2 pi) and the packed head is wrong by
    far more than rounding, for generic weights."""
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, dtype=F64) / dh))
    inv_h = torch.cat([inv, inv])
    inv_h[dh // 2 + which_pair] += 0.5
    assert lev.kv_pack(dh=dh, T=T, seed=seed % 2 ** 31, inv_host=inv_h) > 1e-6


@MID
@given(SEED, st.sampled_from([4, 8, 16]), FLOAT(2.0, 1e6), st.integers(3, 16))
def test_kvPack_default_not_exact(seed, dh, theta_h, T):
    """EP.kvPack_default_not_exact / EP.kvPack_default_fails (impossibility; not counted): a host whose
    frequencies are the config-only default schedule theta_h^(-2p / 2dh) over its 2 dh-wide head (any base
    theta_h > 1) cannot pack two default-RoPE target heads: kv_pack with that table is far off."""
    inv_h = 1.0 / (theta_h ** (torch.arange(0, 2 * dh, 2, dtype=F64) / (2 * dh)))
    assert lev.kv_pack(dh=dh, T=T, seed=seed % 2 ** 31, inv_host=inv_h) > 1e-6


def test_condition_kvPack_emulate_refuses_packing():
    """The repo's builder `elephantine/emulate.head_maps` (v1) tiles KV heads and REFUSES packing, so the
    KV-packing construction exists only in verify/levers (the graph's kv_heads_fit counts packing)."""
    from elephantine.emulate import Refused, head_maps
    with pytest.raises(Refused):
        head_maps(8, 4, 8, 2)               # 4 target KV heads of 2 queries on 2 host KV heads


# =====================================================================================================
# MLARope.lean (+ MLAHost.lean)
# =====================================================================================================

def _deepseek_lite(seed):
    from tests.test_stock_loaders import CASES, _build
    cfg_name, model_name, _, kw, opt = CASES["deepseek_v2-lite"]
    torch.manual_seed(seed)
    return _build(cfg_name, model_name, kw, opt)


@SLOW
@given(st.integers(0, 2 ** 20))
def test_mla_rope_sub_mha_partial(seed):
    """EP.mla_rope_sub_mha_partial: the MHA head with pass pairs (frequency 0) for the content part and
    the MLA's rotating pairs, query [W_qC; W_qR], key [W_UK W_DKV; W_KR] (the RoPE key SHARED by all
    heads), value W_UV W_DKV, equals the MLA head. Built by `elephantine/universal._mla_from` (via
    `from_deepseek`, stock DeepseekV2ForCausalLM, q_lora_rank = None). Checked: (1) the layout -- the
    host's inv_freq is the target's on the first qk_rope/2 pairs and 0 on every other pair; every head's
    rope key rows are the same rows of kv_a_proj (one shared W_KR) and read only x; the content key and
    value rows read only the latent; (2) the logits equal the stock model's. The code also carries the
    latent RMSNorm (on the host's normed-latent input branch), which the norm-free theorem leaves out.
    Logits: fp64 forward of ~100 ops per layer on O(1) activations, 3 layers: rel fp(1, 1e4)."""
    import elephantine.universal as uni
    from tests.test_universal_elephantine import _ids, _lift_fp32
    t = _deepseek_lite(seed)
    tc = t.config
    host = uni.from_deepseek(t)
    H, nope, rope, vd, kl, ql, hd = uni._mla_dims(tc)
    d, r2, h2 = tc.hidden_size, rope // 2, hd // 2
    rope_host = list(range(r2)) + [h2 + i for i in range(r2)]
    for ul, tl in zip(host.layers, t.model.layers):
        a = ul.attn
        assert torch.equal(a.inv_freq[:r2], t.model.rotary_emb.inv_freq.to(F64))
        assert torch.all(a.inv_freq[r2:] == 0)                                   # pass pairs
        K = a.k_proj.weight.view(H, hd, -1)
        for h in range(H):
            assert torch.equal(K[h, rope_host], K[0, rope_host])                 # one shared rope key
        assert torch.all(K[:, rope_host, d:] == 0)                               # ... reading x only
        nope_rows = [i for i in range(hd) if i not in rope_host][:nope]
        assert torch.all(K[:, nope_rows, :d] == 0)                               # content key: the latent
        assert torch.all(a.v_proj.weight[:, :d] == 0)                            # value: the latent
    x = _ids(tc.vocab_size, T=12)
    with torch.no_grad(), _lift_fp32():
        ref = t(x).logits
    with torch.no_grad():
        out = host(x)
    assert (ref - out).abs().max().item() <= fp(1, 1e4) * ref.abs().max().item()


def test_mla_host_sphere_full_rank_no_code():
    """EP.mla_host_sphere_full_rank: NO CODE. No repo function builds a GQA / MQA / MHA target on an MLA
    host (the only exact case: target eps = 0, an isometric down-projection of rank >= the input width).
    Pinned so the BRIDGES row is revisited if one appears: universal has no `to_mla` / MLA-host builder."""
    import elephantine.universal as uni
    assert not [n for n in dir(uni) if "mla" in n.lower() and ("host" in n.lower() or n.startswith("to_"))]


# =====================================================================================================
# MoERouting.lean -- expert duplication in serving/embed.py
# =====================================================================================================

from serving import embed as emb                 # noqa: E402


def _moe_pair(seed, r, extras, norm):
    """A stock Qwen3-MoE target (8 experts, top-2) and serving/embed's host state with top-k 2r and
    r * 8 + extras experts (r copies of every expert, `extras` dummy rows reading -M on the free dim)."""
    from tests.test_embed import MOE_H, MOE_T, moe_cfg, moe_model
    tcfg, tgt = moe_model("qwen3_moe", seed=seed, norm_topk_prob=norm, rms_norm_eps=1e-10, **MOE_T)
    hcfg, _ = moe_cfg("qwen3_moe", **{**MOE_H, "num_experts": r * MOE_T["num_experts"] + extras,
                                      "num_experts_per_tok": r * MOE_T["num_experts_per_tok"],
                                      "norm_topk_prob": norm})
    p = emb.plan(tcfg, hcfg)
    assert p["ok"], p["why"]
    for k_, v in p["config"].items():
        if v is not None:
            setattr(hcfg, k_, v)
    sd = emb.embed_state(tgt.state_dict(), tcfg, hcfg)
    return tcfg, tgt, hcfg, sd, p


def _experts(sd, pre, E, h):
    """Per-expert SwiGLU maps (as float64 closures on inputs of width h) from a state dict."""
    out = []
    for e in range(E):
        g, u, dn = (sd[f"{pre}experts.{e}.{n}_proj.weight"].to(F64) for n in ("gate", "up", "down"))
        out.append(lambda x, g=g, u=u, dn=dn: (F.silu(x[..., :g.shape[1]] @ g.T) * (x[..., :g.shape[1]] @ u.T)) @ dn.T)
    return out


def _renorm(z, S, Y):
    w = torch.exp(z[S])
    return (w[:, None] * torch.stack([Y[e] for e in S.tolist()])).sum(0) / w.sum()


def _plain(z, S, Y):
    w = torch.exp(z[S])
    return (w[:, None] * torch.stack([Y[e] for e in S.tolist()])).sum(0) / torch.exp(z).sum()


def _host_view(seed, r, extras, norm, rho):
    """Target and host router logits, top-k sets and expert outputs at one random normed token x
    (target width) and host token [x; rho; 0] (rho = the free dim's normed value, 1 / rms_B in the model)."""
    tcfg, tgt, hcfg, sd, p = _moe_pair(seed, r, extras, norm)
    tsd = {k_: v.to(F64) for k_, v in tgt.state_dict().items()}
    hA, hB, EA, EB = tcfg.hidden_size, hcfg.hidden_size, tcfg.num_experts, hcfg.num_experts
    L = 0
    pre = f"model.layers.{L}.mlp."
    x = torch.tensor(rng(seed).standard_normal(hA))
    xh = torch.zeros(hB, dtype=F64)
    xh[:hA], xh[hA] = x, rho
    z, zh = tsd[pre + "gate.weight"] @ x, sd[pre + "gate.weight"].to(F64) @ xh
    Yt = [f(x) for f in _experts(tsd, pre, EA, hA)]
    Yh = [f(xh)[:hA] for f in _experts(sd, pre, EB, hB)]
    assert all(f(xh)[hA:].abs().max() == 0 for f in _experts(sd, pre, EB, hB))
    return dict(z=z, zh=zh, Yt=Yt, Yh=Yh, kA=tcfg.num_experts_per_tok, kB=hcfg.num_experts_per_tok, EA=EA, EB=EB,
                r=r, p=p)


def _strict_topk(z, k):
    v, S = torch.topk(z, k)
    rest = torch.ones_like(z, dtype=torch.bool)
    rest[S] = False
    return S, (v.min() - z[rest].max()).item() if rest.any() else float("inf")


@MID
@given(st.integers(0, 2 ** 20), st.integers(1, 3), st.integers(0, 4), st.booleans(), FLOAT(-3, 1))
def test_moeR_copies_topk(seed, r, extras, norm, log_rho):
    """EP.moeR_copies_topk: the target's top-k set strict and every extra scoring below it, so EVERY host
    top-(k r) set is exactly the copies of the target's set. serving/embed.embed_state writes r copies of
    each router row (copy c of expert e at index c E_A + e) and -M rows for the dummies; checked with
    torch.topk (the stock router's selection) on the host logits."""
    v = _host_view(seed, r, extras, norm, 10.0 ** log_rho)
    S, margin = _strict_topk(v["z"], v["kA"])
    assume(margin > 1e-9)                               # strict (the theorem's StrictTopOn)
    Sh, _ = _strict_topk(v["zh"], v["kB"])
    copies = {c * v["EA"] + e for e in S.tolist() for c in range(r)}
    assert set(Sh.tolist()) == copies
    assert v["zh"][r * v["EA"]:].max().item() < v["z"][S].min().item() if extras else True


@MID
@given(st.integers(0, 2 ** 20), st.integers(1, 3), st.integers(0, 4), FLOAT(-3, 1))
def test_moeR_dup_renorm(seed, r, extras, log_rho):
    """EP.moeR_dup_renorm: renormalised gates (norm_topk_prob = True): the host's renormalised output over
    its top-(k r) set equals the target's over its top-k, whatever the extras, given copies whose outputs
    sum to r Y_e (here whole copies, zero-padded: each copy's output IS Y_e, checked). serving/embed host.
    Sums of <= 6 terms of size ~|Y| after the softmax ratio: fp(|Y|, 200)."""
    v = _host_view(seed, r, extras, True, 10.0 ** log_rho)
    S, margin = _strict_topk(v["z"], v["kA"])
    assume(margin > 1e-9)
    Sh, _ = _strict_topk(v["zh"], v["kB"])
    for e in range(v["EA"]):
        for c in range(r):
            assert torch.equal(v["Yh"][c * v["EA"] + e], v["Yt"][e])             # hY with whole copies
    tgt, host = _renorm(v["z"], S, v["Yt"]), _renorm(v["zh"], Sh, v["Yh"])
    s = max(float(y.abs().max()) for y in v["Yt"])
    assert (tgt - host).abs().max().item() <= fp(s, 200)


@MID
@given(st.integers(0, 2 ** 20), st.integers(1, 3), st.integers(1, 4), FLOAT(-9, -4))
def test_moeR_dup_plain_factor(seed, r, extras, log_rho):
    """EP.moeR_dup_plain_factor and EP.moeR_dup_plain_extras_bound: softmax over ALL host experts, no
    renormalisation (norm_topk_prob = False). The host output is the target's times r Z / (r Z + Z_D)
    (Z the target's softmax mass, Z_D the dummies'), so within relative Z_D / (r Z). serving/embed's dummy
    rows read -1e8 on the free dim; at rho = 1e-9..1e-4 their logits are -0.1..-1e4, so Z_D runs from O(1)
    to exp underflow, and the factor is checked as an identity, not by underflow. fp(|Y|, 400)."""
    v = _host_view(seed, r, extras, False, 10.0 ** log_rho)
    S, margin = _strict_topk(v["z"], v["kA"])
    assume(margin > 1e-9)
    Sh, _ = _strict_topk(v["zh"], v["kB"])
    assume(set(Sh.tolist()) == {c * v["EA"] + e for e in S.tolist() for c in range(r)})   # dummies below S
    Z = torch.exp(v["z"]).sum().item()
    ZD = torch.exp(v["zh"][r * v["EA"]:]).sum().item()
    tgt, host = _plain(v["z"], S, v["Yt"]), _plain(v["zh"], Sh, v["Yh"])
    s = max(float(y.abs().max()) for y in v["Yt"])
    assert (host - r * Z / (r * Z + ZD) * tgt).abs().max().item() <= fp(s, 400)
    assert (host - tgt).abs().max().item() <= ZD / (r * Z) * tgt.abs().max().item() + fp(s, 400)


@MID
@given(st.integers(0, 2 ** 20), st.integers(1, 3), FLOAT(-3, 1))
def test_moeR_dup_plain(seed, r, log_rho):
    """EP.moeR_dup_plain: not renormalised and NO extras (E_h = r E_t): exact. serving/embed host with
    r * 8 experts. fp(|Y|, 400)."""
    v = _host_view(seed, r, 0, False, 10.0 ** log_rho)
    S, margin = _strict_topk(v["z"], v["kA"])
    assume(margin > 1e-9)
    Sh, _ = _strict_topk(v["zh"], v["kB"])
    tgt, host = _plain(v["z"], S, v["Yt"]), _plain(v["zh"], Sh, v["Yh"])
    s = max(float(y.abs().max()) for y in v["Yt"])
    assert (tgt - host).abs().max().item() <= fp(s, 400)


def _stock_block(cfg, sd, L):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    b = Qwen3MoeSparseMoeBlock(cfg).to(F64).eval()
    pre = f"model.layers.{L}.mlp."
    b.load_state_dict({k_[len(pre):]: v.to(F64) for k_, v in sd.items() if k_.startswith(pre)})
    return b


@SLOW
@given(st.integers(0, 2 ** 20), st.integers(1, 3), st.integers(0, 4), st.booleans())
def test_moeR_dup_on_stock_block(seed, r, extras, norm):
    """EP.moeR_dup_renorm / EP.moeR_dup_plain on the STOCK Qwen3MoeSparseMoeBlock: the host block loaded
    with serving/embed's layer reproduces the target block on [x; rho; 0] tokens (rho = 0.5). transformers
    routes in float32 (softmax with dtype=float, weights cast back), so the slack is float32's: 64 U32
    relative to the output scale. Tokens whose target top-k margin is below float32 resolution are
    dropped (a float32 near-tie can split the copies, see the tie MISMATCH)."""
    tcfg, tgt, hcfg, sd, p = _moe_pair(seed, r, extras, norm)
    hA, hB = tcfg.hidden_size, hcfg.hidden_size
    tb, hb = tgt.model.layers[0].mlp, _stock_block(hcfg, sd, 0)
    x = torch.tensor(rng(seed).standard_normal((1, 16, hA)))
    xh = torch.zeros(1, 16, hB, dtype=F64)
    xh[..., :hA], xh[..., hA] = x, 0.5
    probs = torch.softmax(tb.gate(x).float(), -1)[0]
    vals = probs.sort(-1, descending=True).values
    keep = (vals[:, tcfg.num_experts_per_tok - 1] - vals[:, tcfg.num_experts_per_tok]) > 1e-5
    with torch.no_grad():
        yt, yh = tb(x)[0][0], hb(xh)[0][0]
    assert yh[:, hA:].abs().max().item() == 0
    s = yt.abs().max().item()
    assert (yt - yh[:, :hA])[keep].abs().max().item() <= 64 * U32 * (1 + s)


@SLOW
@given(st.integers(0, 2 ** 20), st.booleans())
def test_moe_dup_refuses_tied_router_rows(seed, norm):
    """Was MISMATCH (W5-1), EP.moeR_tie_split vs serving/embed's 'k_B = r k_A is exact'. EP.moeR_copies_topk
    needs the target's top-k to be STRICT. At a tie across the target's boundary the host's top-(k r) may
    take copies of more than k experts and then equals no target tie-break (EP.moeR_tie_split). Instance:
    a Qwen3-MoE target whose layer-0 router is zero (every expert tied -- e.g. a dense FFN router-steered
    into the MoE, as serving/embed and moe/zero_router themselves build), embedded with r = 2. embed_state
    now refuses it (ValueError naming the tied rows); with ties="warn" it embeds it, warns, and the stock
    host block differs from the stock target block by O(|Y|) -- the split the refusal guards against."""
    tcfg, tgt, hcfg, sd, p = _moe_pair(seed, 2, 0, norm)
    with torch.no_grad():
        tgt.model.layers[0].mlp.gate.weight.zero_()
    assert emb.plan(tcfg, hcfg)["ok"]                   # a weight condition: plan() sees only configs
    assert emb.tied_router_rows(tgt.model.layers[0].mlp.gate.weight) == [list(range(tcfg.num_experts))]
    with pytest.raises(ValueError, match="not strict"):
        emb.embed_state(tgt.state_dict(), tcfg, hcfg)
    with pytest.warns(UserWarning, match="not strict"):
        sd = emb.embed_state(tgt.state_dict(), tcfg, hcfg, ties="warn")
    hA, hB = tcfg.hidden_size, hcfg.hidden_size
    tb, hb = tgt.model.layers[0].mlp, _stock_block(hcfg, sd, 0)
    x = torch.tensor(rng(seed).standard_normal((1, 8, hA)))
    xh = torch.zeros(1, 8, hB, dtype=F64)
    xh[..., :hA] = x
    with torch.no_grad():
        yt, yh = tb(x)[0][0], hb(xh)[0][0][:, :hA]
        _, St = torch.topk(torch.softmax(tb.gate(x).float(), -1), tcfg.num_experts_per_tok, -1)
        _, Sh = torch.topk(torch.softmax(hb.gate(xh).float(), -1), hcfg.num_experts_per_tok, -1)
    host_experts = set((Sh[0, 0] % tcfg.num_experts).tolist())
    assert host_experts != set(St[0, 0].tolist())                 # the host took more than k experts
    assert (yt - yh).abs().max().item() > 1e-3 * yt.abs().max().item()


def test_moe_dup_accepts_distinct_router_rows():
    """The strict-top-k guard does not fire on an ordinary (random, distinct-row) router, nor at r = 1."""
    tcfg, tgt, hcfg, sd, p = _moe_pair(0, 2, 0, False)
    assert emb.tied_router_rows(tgt.model.layers[0].mlp.gate.weight) == []
    tcfg1, tgt1, hcfg1, _, _ = _moe_pair(0, 1, 4, False)
    with torch.no_grad():
        tgt1.model.layers[0].mlp.gate.weight.zero_()
    emb.embed_state(tgt1.state_dict(), tcfg1, hcfg1)              # r = 1: no copies, no split

def test_moeR_dup_needs_divisibility_and_experts_enforced():
    """EP.moeR_dup_needs_divisibility, EP.moeR_dup_needs_experts, EP.moeR_renorm_host_needs_const_mass /
    EP.moeR_plain_host_needs_const_mass (necessity, not counted): serving/embed.plan refuses k_B not a
    multiple of k_A, E_B < (k_B / k_A) E_A, and a norm_topk_prob mismatch."""
    from tests.test_embed import MOE_H, MOE_T, moe_cfg
    tcfg, _ = moe_cfg("qwen3_moe", **MOE_T)
    for kw, why in (({"num_experts_per_tok": 3}, "top-k"), ({"num_experts": 12, "num_experts_per_tok": 4}, "experts"),
                    ({"norm_topk_prob": True}, "norm_topk_prob")):
        hcfg, _ = moe_cfg("qwen3_moe", **{**MOE_H, "num_experts": 16, **kw})
        p = emb.plan(tcfg, hcfg)
        assert not p["ok"] and any(why in w for w in p["why"]), p["why"]


from moe import zero_router as zr                # noqa: E402


@SLOW
@given(FLOAT(0.25, 8.0), st.booleans())
def test_moeR_scaling_fold(s_h, norm):
    """EP.moeR_scaling_fold (renormalised) and EP.moeR_scaling_fold_plain (norm_topk_prob = False):
    routed_scaling_factor s_h folds into the down-projections. moe/zero_router.deal_dense_into_moe on a
    stock DeepseekV3MoE (sigmoid gates, group-limited top-k) probes each slot's gate -- s_h / k
    renormalised, s_h / 2 not -- and divides the down-projection by it, so the block reproduces the dense
    target (scale 1) for every s_h: the probed weight is exactly linear in s_h. HF runs this block only
    in float32: slack 1e3 U32 relative."""
    r = zr.check("deepseek_v3", torch.float32, routed_scaling_factor=s_h, norm_topk_prob=norm)
    k = r["k"]
    w = [v for key, v in r["weights"].items() if key != "shared"]
    want = s_h / k if norm else s_h / 2
    assert all(abs(x - want) <= 1e-5 * want + 1e-6 for x in w), (w, want)   # weights rounded to 1e-6
    assert r["err"] <= 1e3 * U32 * (1 + r["scale"])


# =====================================================================================================
# LearnedAbs.lean -- hf/bos_constructions.py (copied from branch bos-token)
# =====================================================================================================

from hf import bos_constructions as bos          # noqa: E402


def _rand_spec(seed, V, d, T, H):
    g = torch.Generator().manual_seed(seed)
    return dict(E=torch.randn(V, d, dtype=F64, generator=g), Ptab=torch.randn(T + 1, d, dtype=F64, generator=g),
                bos=0, H=H)


@SLOW
@given(st.integers(0, 2 ** 20), st.integers(2, 40), st.sampled_from([(8, 2), (12, 3), (16, 4)]),
       st.sampled_from(["relu", "silu", "gelu_new"]), st.sampled_from([0.0, None]))
def test_learned_abs_two_block(seed, T, dH, act, L):
    """EP.learned_abs_two_block: a stock LlamaForCausalLM with NoPE (inv_freq = 0), width d + 5, pre-norm
    RMSNorm and bias-free MLPs, whose two blocks are written by hf/bos_constructions.write_bos_prefix, ends
    with residual [e(x_t) + P_t; 0] at EVERY position t = 0..T of a sequence with the BOS at position 0
    only, for any token table e and position table P (prefix_check: one sequence spans all T + 1
    positions). L = 0 is the Lean's zero-logit head (u_t = 1/(t+1)); L = None the code's default log T.
    The MLP units are ReLU (the Lean's), SiLU or GELU (the code solves the knot system exactly for each).
    Rounding: the table is a sum of T + 1 ramps whose terms grow like (c_beta T)^2 |P| (the docstring's
    T^2 cancellation): fp(scale (40 (T+1))^2, 1e2)."""
    d, H = dH
    r = bos.prefix_check(_rand_spec(seed, 23, d, T, H), act=act, n_seq=2, T_seq=T + 1, L=L, seed=seed)
    assert r["max_err"] <= fp(r["scale"] * (40 * (T + 1)) ** 2, 1e2), r


def _rand_gpt2(seed, n_layer, T):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=41, n_embd=16, n_layer=n_layer, n_head=4, n_positions=T + 1, bos_token_id=40,
                     eos_token_id=40, attn_implementation="sdpa")
    m = GPT2LMHeadModel(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.add_(0.2 * torch.randn_like(p) if "ln" in n else 0.05 * torch.randn_like(p))
    return m


@SLOW
@given(st.integers(0, 2 ** 20), st.integers(1, 3), st.integers(4, 24))
def test_learned_abs_model(seed, n_layer, T):
    """EP.learned_abs_model: the two-block prefix followed by the padded copies of the target's pre-norm
    sublayers reproduces the learned-absolute target end to end. hf/bos_constructions.learned_abs_host on
    a random GPT-2 (learned wpe over T + 1 positions, LayerNorm folded to RMSNorm by writer centring, the
    padded eps' = eps d / (d + 5) and weight sqrt(d / (d + 5)) w): logits equal on every BOS-first input
    of length T + 1. Rounding: the prefix's T^2 cancellation (as above) times the downstream layers'
    Lipschitz growth, on logits of size ~1: fp((40 (T+1))^2, 1e3)."""
    tgt = _rand_gpt2(seed, n_layer, T)
    host = bos.learned_abs_host(bos.gpt2_spec(tgt))
    ids = torch.randint(0, 40, (2, T + 1), generator=torch.Generator().manual_seed(seed))
    ids[:, 0] = 40
    with torch.no_grad(), bos.exact_norms():
        lt, lh = tgt(ids).logits, host(ids).logits
    assert (lt - lh).abs().max().item() <= fp((40 * (T + 1)) ** 2, 1e3) * (1 + lt.abs().max().item())


@SLOW
@given(st.integers(0, 2 ** 20), st.integers(1, 3), st.integers(4, 24))
def test_padded_prenorm_sublayer(seed, n_layer, T):
    """EP.padded_prenorm_sublayer / EP.padded_prenorm_stack: each padded host sublayer (norm weight
    sqrt(d/(d+5)) w, eps d/(d+5), reads and writes the first d channels) maps the padded target residual
    to the padded next residual. Checked layer by layer on learned_abs_host's GPT-2 copy: host residual
    after layer l + 2 = [P h_l; 0], where h_l is the target's residual after block l and P the centring
    (the LayerNorm -> RMSNorm writer fold). Same rounding as test_learned_abs_model."""
    tgt = _rand_gpt2(seed, n_layer, T)
    host = bos.learned_abs_host(bos.gpt2_spec(tgt))
    ids = torch.randint(0, 40, (1, T + 1), generator=torch.Generator().manual_seed(seed))
    ids[:, 0] = 40
    th, hh = {}, {}
    hooks = [b.register_forward_hook(lambda m, a, o, i=i: th.__setitem__(i, o[0] if isinstance(o, tuple) else o))
             for i, b in enumerate(tgt.transformer.h)]
    hooks += [b.register_forward_hook(lambda m, a, o, i=i: hh.__setitem__(i, o[0] if isinstance(o, tuple) else o))
              for i, b in enumerate(host.model.layers)]
    with torch.no_grad(), bos.exact_norms():
        tgt(ids), host(ids)
    for h in hooks:
        h.remove()
    d = tgt.config.n_embd
    for i in range(n_layer):
        want = th[i][0] - th[i][0].mean(-1, keepdim=True)
        got = hh[i + 2][0]
        tol = fp((40 * (T + 1)) ** 2, 1e3) * (1 + want.abs().max().item())
        assert (got[:, :d] - want).abs().max().item() <= tol
        assert got[:, d:].abs().max().item() <= tol


from attn import attention_sink as asink         # noqa: E402


@FAST
@given(SEED, st.integers(0, 2000), st.integers(1, 8))
def test_sink_uniform_position(seed, t, dv):
    """EP.sink_uniform_position: zero logits on the t + 1 visible keys, sink logit 0, the same value c on
    every key: the output is (t+1)/(t+2) c. attn/attention_sink.sink_attention. A matmul summing t + 1
    terms (accumulated in order, up to (t+1) U each) and one division: fp(|c|, 4 (t+2))."""
    c = torch.tensor(rng(seed).standard_normal(dv))
    out = asink.sink_attention(torch.zeros(t + 1, dtype=F64), c[None].expand(t + 1, dv), 0.0)
    assert (out - (t + 1) / (t + 2) * c).abs().max().item() <= fp(c.abs().max().item(), 4 * (t + 2))


@SLOW
@given(st.integers(0, 2 ** 20), st.integers(0, 40), st.integers(2, 12))
def test_no_learned_abs_without_marker(seed, tok, T):
    """EP.PresConst.attention / EP.PresConst.stack / EP.no_learned_abs_without_marker (impossibility, not
    counted): a stock LlamaModel with RoPE (position-dependent logits), causal mask, pre-norms and MLPs
    sends a constant token sequence to a constant sequence of hidden states, so no such host writes a
    position table without a marker. Softmax over equal values: rounding only, fp(|h|, 1e3)."""
    from transformers import LlamaConfig, LlamaModel
    torch.manual_seed(seed)
    m = LlamaModel(LlamaConfig(vocab_size=41, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa")).to(F64).eval()
    with torch.no_grad(), bos.exact_norms():           # the stock RMSNorm formula at fp64 (it casts to fp32)
        h = m(torch.full((1, T), tok)).last_hidden_state[0]
    assert (h - h[0]).abs().max().item() <= fp(h.abs().max().item(), 1e3)


# =====================================================================================================
# RopeRank.lean -- impossibility; the repo's gate
# =====================================================================================================

from hf import partial_rope_host as prh          # noqa: E402


@MID
@given(st.sampled_from([8, 16, 32]), st.integers(1, 3))
def test_partial_rope_too_few_enforced(d, short):
    """EP.partial_rope_too_few / EP.partial_rope_too_few_default (impossibility, not counted): a host with
    fewer rotating pairs than the target cannot play it. hf/partial_rope_host.column_map refuses
    rot_h < rot_t (it raises instead of building a lossy layout)."""
    t = prh.Target(**{**dict(H=1, KV=1, d=d, dv=d, rot=d, wq=None, wk=None, wv=None, wo=None,
                             inv_freq=torch.ones(d // 2, dtype=F64), scale=1.0, m=1.0, bq=None, bk=None, bv=None,
                             bo=None, q_norm=None, k_norm=None, eps=0.0)})
    h = prh.Host(H=1, KV=1, hd=2 * d, rot=d - 2 * short, interleaved=False, scale=1.0)
    with pytest.raises(ValueError, match="rotates"):
        prh.column_map(t, h)
