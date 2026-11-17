"""Layer 2 of the verification plan, third batch: the 29 Lean theorems that batch 2 (tests/test_lean_bridges_2.py)
marked NO CODE, each checked against the repo code written to build its construction. verify/BRIDGES.md maps
theorem -> function -> test -> status (the "Batch 3" section).

Same conventions as the first two batches: every test is named after its theorem and checks the theorem's
equation (exact), convergence (limit) or inequality (bound) on the repo's function, over Hypothesis-drawn
inputs in float64. `test_condition_*` tests show the equation fails when one of the theorem's hypotheses is
dropped.

Tolerances. U = 2^-53 is the float64 unit roundoff; `fp(S, n)` = n U (1 + S) allows n roundings of terms of
size <= S. Where `transformers` rounds to float32 (router logits, DeepSeek's RMSNorm) the slack is float32's,
U32 = 2^-24, and the test says so.

Whole-model limit families (LimitInstances2.lean). The theorems take per-layer Lipschitz constants L_i, a head
constant Lh, value / read bounds and a region as hypotheses. The tests build the multi-layer host and target
(block/limit_stacks.py) on a tiny random model, MEASURE those constants (block/limit_stacks.lipschitz: the
largest inf->inf operator norm of the layer Jacobian over the target and host trajectories and random points
near them; Lh, M and R are read off the weights and the trajectories), evaluate the theorem's bound formula
with them, and check the measured whole-model error is below it. The measured L_i is a sample maximum, not a
certified Lipschitz constant, so this checks the bound's formula on real numbers, not the hypotheses.

Run:  python -m pytest tests/test_lean_bridges_3.py -q
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np                                        # noqa: E402
import pytest                                             # noqa: E402
import torch                                              # noqa: E402
import torch.nn.functional as F                           # noqa: E402
from hypothesis import HealthCheck, assume, given, settings   # noqa: E402
from hypothesis import strategies as st                   # noqa: E402

F64 = torch.float64
U = 2.0 ** -53
U32 = 2.0 ** -24


def fp(scale, n=1e3):
    """n float64 roundings of terms of size <= scale."""
    return n * U * (1.0 + float(scale))


SEED = st.integers(0, 2 ** 31 - 1)
FAST = settings(max_examples=500, deadline=None)
MID = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
SLOW = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
MODEL = settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
FLOAT = lambda lo, hi: st.floats(lo, hi, allow_nan=False, allow_infinity=False)   # noqa: E731


def rng(seed):
    return np.random.default_rng(seed)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def _cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "config.json"), str)
    except Exception:
        return False


# =====================================================================================================
# Capacity.lean: LayerNorm hidden padding
# =====================================================================================================

from norm import layernorm_pad as lnp                     # noqa: E402


@FAST
@given(SEED, st.integers(1, 24), st.integers(0, 24), st.one_of(st.just(0.0), FLOAT(1e-10, 1.0)), FLOAT(-3, 3))
def test_layerNorm_pad_center(seed, d, e, eps, logscale):
    """EP.layerNorm_pad_center: the host LayerNorm (weight [sqrt(d/D) gamma; 0], bias [beta; 0], eps d/D) on the
    padded centred vector [P x; 0] is [LayerNorm_eps(x); 0]. Host side through torch's own F.layer_norm with
    lnp.pad_ln_params; slack fp(|out|, 100 d). eps = 0 needs d > 1 (a constant x is 0/0)."""
    assume(d > 1 or eps > 0)
    r = rng(seed)
    x = torch.from_numpy(r.standard_normal(d) * 10 ** logscale)
    g, b = torch.from_numpy(r.standard_normal(d)), torch.from_numpy(r.standard_normal(d))
    w2, b2, eps2 = lnp.pad_ln_params(g, b, eps, e)
    host = F.layer_norm(lnp.pad(lnp.center(x), e), (d + e,), w2, b2, eps2)
    want = lnp.pad(F.layer_norm(x, (d,), g, b, eps), e)
    S = want.abs().max().item() + 1
    assert torch.allclose(host, want, rtol=0, atol=fp(S, 100 * d))
    assert torch.equal(host[d:], torch.zeros(e, dtype=F64))
    assert torch.allclose(lnp.host_layer_norm(x, g, b, eps, e), want, rtol=0, atol=fp(S, 100 * d))


@FAST
@given(SEED, st.integers(2, 16), st.integers(1, 16))
def test_condition_layerNorm_pad_needs_centring(seed, d, e):
    """EP.layerNorm_pad_center's centring: the same host norm on the UNcentred padded vector [x; 0] is not the
    padded LayerNorm (the pad channels shift the mean) once x has a nonzero mean."""
    r = rng(seed)
    x = torch.from_numpy(r.standard_normal(d)) + 3.0
    g, b = torch.ones(d, dtype=F64), torch.zeros(d, dtype=F64)
    w2, b2, eps2 = lnp.pad_ln_params(g, b, 1e-5, e)
    host = F.layer_norm(lnp.pad(x, e), (d + e,), w2, b2, eps2)
    assert (host[:d] - F.layer_norm(x, (d,), g, b, 1e-5)).abs().max() > 1e-3


def _tiny(mt, seed, **kw):
    from transformers import AutoConfig, AutoModelForCausalLM
    base = dict(gpt_neox=dict(hidden_size=32, num_attention_heads=4, intermediate_size=48, num_hidden_layers=2,
                              vocab_size=64, max_position_embeddings=64, rotary_pct=0.5),
                opt=dict(hidden_size=32, num_attention_heads=4, ffn_dim=48, num_hidden_layers=2, vocab_size=64,
                         max_position_embeddings=64, word_embed_proj_dim=32))[mt]
    base.update(kw)
    cfg = AutoConfig.for_model(mt, **base)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.add_(torch.randn_like(p) * (0.3 if p.dim() == 1 else 0.1))
    return m


@MODEL
@given(SEED, st.sampled_from(["gpt_neox", "opt"]), st.integers(1, 3), st.booleans())
def test_hidden_pad_stack_layerNorm(seed, mt, k, parallel):
    """EP.hidden_pad_stack_layerNorm, on norm/layernorm_pad.pad_layernorm_model: a stock GPT-NeoX / OPT target of
    width d and its stock host of width d + k * head_dim (centred writers, padded LayerNorms with eps d/D, extra
    zero heads). Both halves of the theorem: the host residual is [P h_l; 0] after every layer (HF's
    hidden_states, all but the last, which HF returns after the final norm), and the host's final LayerNorm (so
    the logits) is the padded target's. Slack fp(S, 1e4): a few hundred roundings per layer."""
    kw = dict(use_parallel_residual=parallel) if mt == "gpt_neox" else {}
    m = _tiny(mt, seed, **kw)
    d = m.config.hidden_size
    e = k * (d // m.config.num_attention_heads)
    h = lnp.pad_layernorm_model(m, e)
    assert h.config.hidden_size == d + e and type(h) is type(m)
    ids = torch.randint(0, 64, (2, 12), generator=gen(seed))
    with torch.no_grad():
        a, b = m(ids, output_hidden_states=True), h(ids, output_hidden_states=True)
    S = max(x.abs().max().item() for x in a.hidden_states) + a.logits.abs().max().item()
    for ht, hh in zip(a.hidden_states[:-1], b.hidden_states[:-1]):
        assert (hh - lnp.pad(lnp.center(ht), e)).abs().max().item() <= fp(S, 1e4)
    assert (a.logits - b.logits).abs().max().item() <= fp(S, 1e4)


@pytest.mark.parametrize("name", ["EleutherAI/pythia-14m", "facebook/opt-125m"])
def test_hidden_pad_stack_layerNorm_real_checkpoint(name):
    """EP.hidden_pad_stack_layerNorm end to end on the stock checkpoints (local HF cache only): pythia-14m
    (d = 128 -> 160, 4 -> 5 heads, parallel residual, untied) and OPT-125m (768 -> 832, 12 -> 13 heads, tied
    embeddings untied by the host). float64 logits agree to 1e-9 (measured ~1e-13)."""
    if not _cached(name):
        pytest.skip(f"{name} not in the local HF cache")
    r = lnp.pad_check(name, T=16, n=1)
    assert r["D"] > r["d"] and r["err"] < 1e-9, r


def test_condition_hidden_pad_stack_layerNorm_needs_centred_writers():
    """EP.hidden_pad_stack_layerNorm's centring of the writers: the same host with UNcentred writers and embedding
    ([W; 0] instead of [P W; 0]) does not reproduce the target."""
    m = _tiny("gpt_neox", 0)
    h = lnp.pad_layernorm_model(m, 8)
    sd = {k: v.clone() for k, v in h.state_dict().items()}
    d = m.config.hidden_size
    sd["gpt_neox.embed_in.weight"][:, :d] = m.state_dict()["gpt_neox.embed_in.weight"]
    h.load_state_dict(sd)
    ids = torch.randint(0, 64, (1, 10), generator=gen(0))
    with torch.no_grad():
        assert (m(ids).logits - h(ids).logits).abs().max().item() > 1e-3


# =====================================================================================================
# CapacityDepth.lean: looped hosts
# =====================================================================================================

from block import looped_counter as lc                    # noqa: E402


@FAST
@given(FLOAT(-1e6, 1e6))
def test_looped_width2_neg(x):
    """EP.looped_width2_neg: block/looped_counter.width2_block (p + (R p - p), R the quarter rotation) looped
    twice maps (x, 0) to (-x, 0): EP.negTarget's x -> -x, which no width-1 loop computes. The update is
    computed as written (R p - p, then added), so the result is within 2 roundings of |x|."""
    p = lc.loop(lc.width2_block(), torch.tensor([x, 0.0], dtype=F64), 2)
    assert abs(p[0].item() + x) <= 4 * U * abs(x) and abs(p[1].item()) <= 4 * U * abs(x)
    nt = lc.neg_target(torch.tensor(x, dtype=F64)).item()
    assert abs(nt + x) <= 4 * U * abs(x)


@MID
@given(SEED, st.integers(1, 8), st.integers(1, 8), st.integers(1, 8), FLOAT(0.0, 10.0))
def test_looped_counter_two(seed, d, m, m2, slack):
    """EP.looped_counter_two: one residual ReLU-MLP block (block/looped_counter.counter_block, units of both
    target layers masked by -B c and -B (o - c), one counter unit relu(o)) looped twice on [x; 0; 1] gives
    (f2(f1 x), 2, 1) exactly, for any B with a2_k . x <= B and a_j . f1(x) <= B (here B = that max + slack).
    The host's units are the target's units shifted by 0 or -B, so the slack is fp(S, 50 (m + m2 + d))."""
    g = gen(seed)
    a, v = torch.randn(m, d, dtype=F64, generator=g), torch.randn(m, d, dtype=F64, generator=g)
    a2, v2 = torch.randn(m2, d, dtype=F64, generator=g), torch.randn(m2, d, dtype=F64, generator=g)
    x = torch.randn(7, d, dtype=F64, generator=g) * 3
    f1 = x + torch.relu(x @ a.T) @ v
    want = f1 + torch.relu(f1 @ a2.T) @ v2
    assert torch.allclose(lc.two_layer_target(a, v, a2, v2)(x), want, rtol=0, atol=0)
    B = max((x @ a2.T).max().item(), (f1 @ a.T).max().item()) + slack
    blk = lc.counter_block(a, v, a2, v2, B)
    assert blk.W_in.shape == (m + m2 + 1, d + 2) and torch.count_nonzero(blk.b_in) == 0
    s = lc.loop(blk, lc.embed_counter(x), 2)
    S = want.abs().max().item() + abs(B) + f1.abs().max().item()
    assert (s[:, :d] - want).abs().max().item() <= fp(S * (1 + v.abs().max().item() + v2.abs().max().item()),
                                                      50 * (m + m2 + d))
    assert torch.equal(s[:, d], torch.full((7,), 2.0, dtype=F64)) and torch.equal(s[:, d + 1], torch.ones(7, dtype=F64))


def test_condition_looped_counter_two_needs_bounded_inputs():
    """EP.looped_counter_two's bound a2_k . x <= B: with B below it, the layer-2 units fire in the first
    iteration and the loop is not f2 o f1."""
    g = gen(1)
    d, m = 4, 5
    a, v, a2, v2 = (torch.randn(m, d, dtype=F64, generator=g) for _ in range(4))
    x = torch.randn(9, d, dtype=F64, generator=g) * 3
    want = lc.two_layer_target(a, v, a2, v2)(x)
    B = 0.5 * (x @ a2.T).max().item()
    assume_pos = (x @ a2.T).max().item() > 0
    assert assume_pos
    s = lc.loop(lc.counter_block(a, v, a2, v2, B), lc.embed_counter(x), 2)
    assert (s[:, :d] - want).abs().max().item() > 1e-6


# =====================================================================================================
# PosBias.lean: additive-bias and learned-absolute hosts (norm-free)
# =====================================================================================================

from attn import posbias_hosts as pbh                     # noqa: E402


def _t5_attention(seed, decoder, H=3, dkv=4, dm=10, nb=8, maxd=20):
    from transformers import T5Config
    from transformers.models.t5.modeling_t5 import T5Attention
    cfg = T5Config(d_model=dm, d_kv=dkv, num_heads=H, relative_attention_num_buckets=nb,
                   relative_attention_max_distance=maxd, is_decoder=decoder, dropout_rate=0.0)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    at = T5Attention(cfg, has_relative_attention_bias=True, layer_idx=0).to(F64).eval()
    with torch.no_grad():
        for p in at.parameters():
            p.normal_(0, 0.5)
    return at


@MID
@given(SEED, st.integers(1, 40), st.booleans())
def test_t5_sub_additive_bias(seed, T, decoder):
    """EP.t5_sub_additive_bias: a T5 relative-bias head is the additive-bias head with the table
    B[i, j] = b(bucket(j - i)). attn/posbias_hosts.t5_bias_table + bias_attention, with the stock T5Attention's
    q/k/v/o and relative_attention_bias weights, against T5Attention's own forward (bidirectional buckets for
    the encoder; causal buckets and a causal mask for the decoder). Bucketing also checked against
    T5Attention._relative_position_bucket on every offset. T5Attention takes its softmax in float32, so the
    slack is 16 U32 ||o||_inf max|v| (plus fp(S, 100 T))."""
    at = _t5_attention(seed, decoder)
    H, dkv = at.n_heads, at.key_value_proj_dim
    rel = torch.arange(-T, T + 1)
    assert torch.equal(pbh.t5_bucket(rel, not decoder, at.relative_attention_num_buckets,
                                     at.relative_attention_max_distance),
                       at._relative_position_bucket(rel, not decoder, at.relative_attention_num_buckets,
                                                    at.relative_attention_max_distance))
    x = torch.randn(1, T, at.d_model, dtype=F64, generator=gen(seed))
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool)) if decoder else None
    mask = None if keep is None else torch.zeros(1, 1, T, T, dtype=F64).masked_fill(~keep, torch.finfo(F64).min)
    with torch.no_grad():
        ref = at(x, mask=mask, cache_position=torch.arange(T))[0]
        B = pbh.t5_bias_table(T, at.relative_attention_bias.weight, not decoder, at.relative_attention_num_buckets,
                              at.relative_attention_max_distance)
        sp = lambda W: (x @ W.weight.T).view(1, T, H, dkv).transpose(1, 2)      # noqa: E731
        o = pbh.bias_attention(sp(at.q), sp(at.k), sp(at.v), B, keep, 1.0)
        out = at.o(o.transpose(1, 2).reshape(1, T, H * dkv))
        vmax = sp(at.v).abs().max().item()
    K = at.o.weight.abs().sum(-1).max().item()            # inf->inf norm of o
    assert (out - ref).abs().max().item() <= 16 * U32 * K * vmax + fp(ref.abs().max().item() + 1, 100 * T)


def _head(seed, d, dh, dv=3):
    g = gen(seed)
    return (torch.randn(dh, d, dtype=F64, generator=g), torch.randn(dh, d, dtype=F64, generator=g),
            torch.randn(dv, d, dtype=F64, generator=g))


@MID
@given(SEED, st.integers(1, 24), st.integers(1, 6), st.integers(1, 5), st.integers(0, 7), st.booleans())
def test_alibi_sub_learned_abs_causal(seed, T, d, dh, h, causal):
    """EP.alibi_sub_learned_abs_causal: attn/posbias_hosts.alibi_learned_abs_host (position table [0; 1; t], q
    reads the 1, k reads m t / scale) gives, on the residual [x; 0] + P_t, exactly the ALiBi head's output
    softmax(q.k scale - m (i - j)) v, for a causal or a full key set. The slope m is head h's of an 8-head
    model from transformers' BLOOM build_alibi_tensor. The host logit is the ALiBi logit + m i (a per-query
    constant), so the slack covers |m| T: fp(S + m T, 100 T)."""
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor
    m = build_alibi_tensor(torch.ones(1, 2), 8, F64)[h, 0, 1].item()
    Wq, Wk, Wv = _head(seed, d, dh)
    x = torch.randn(T, d, dtype=F64, generator=gen(seed + 1))
    scale = 1 / math.sqrt(dh)
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool)) if causal else None
    c = (x @ Wq.T) @ (x @ Wk.T).T * scale
    s = pbh.alibi_scores(c, m)
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    want = torch.softmax(s, -1) @ (x @ Wv.T)
    P, Wq2, Wk2, Wv2 = pbh.alibi_learned_abs_host(Wq, Wk, Wv, m, scale, T)
    host = pbh.learned_abs_head(x, P, Wq2, Wk2, Wv2, keep, scale)
    S = want.abs().max().item() + c.abs().max().item() + m * T
    assert (host - want).abs().max().item() <= fp(S, 100 * T) * (1 + (x @ Wv.T).abs().max().item())


@MID
@given(SEED, st.integers(1, 16), st.integers(1, 6), st.integers(1, 5), st.sampled_from(["random", "alibi2", "t5"]))
def test_bias_sub_learned_abs(seed, T, d, dh, kind):
    """EP.bias_sub_learned_abs: attn/posbias_hosts.onehot_learned_abs_host (one-hot position channels, q reads
    e_i, k reads B^T e_j) gives the additive-bias head softmax(q.k scale + B[i, j]) v for ANY table B: random,
    bidirectional ALiBi -m|i - j|, or a stock T5Attention's bucketed table (head 0). Slack fp(S, 100 T)."""
    r = rng(seed)
    if kind == "random":
        B = torch.from_numpy(r.standard_normal((T, T)) * 3)
    elif kind == "alibi2":
        i = torch.arange(T, dtype=F64)
        B = -0.5 * (i[:, None] - i[None, :]).abs()
    else:
        at = _t5_attention(seed, False)
        with torch.no_grad():
            B = pbh.t5_bias_table(T, at.relative_attention_bias.weight, True, at.relative_attention_num_buckets,
                                  at.relative_attention_max_distance)[0]
    Wq, Wk, Wv = _head(seed, d, dh)
    x = torch.randn(T, d, dtype=F64, generator=gen(seed + 1))
    scale = 1 / math.sqrt(dh)
    want = pbh.bias_attention(x @ Wq.T, x @ Wk.T, x @ Wv.T, B, None, scale)
    P, Wq2, Wk2, Wv2 = pbh.onehot_learned_abs_host(Wq, Wk, Wv, B, scale)
    host = pbh.learned_abs_head(x, P, Wq2, Wk2, Wv2, None, scale)
    S = want.abs().max().item() + B.abs().max().item() + ((x @ Wq.T) @ (x @ Wk.T).T).abs().max().item()
    assert (host - want).abs().max().item() <= fp(S, 100 * T) * (1 + (x @ Wv.T).abs().max().item())


# =====================================================================================================
# AuditRules.lean: value embeddings off, a start token's stream on its own channel
# =====================================================================================================

from attn import value_emb as vemb                        # noqa: E402
from block import marker_stream as mks                    # noqa: E402

_LLAMA = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False,
              rms_norm_eps=1e-5)


def _llama(seed, **kw):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(**{**_LLAMA, **kw})
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    m = LlamaForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return m


@SLOW
@given(SEED, FLOAT(-5, 5), st.integers(0, 1))
def test_value_emb_off(seed, lam, layer):
    """EP.value_emb_off: attn/value_emb.add_value_mix puts v_j + lam u_j (value embeddings / a value residual)
    into a stock LlamaAttention's values; value_emb_off sets lam = 0, and the model's logits are then the plain
    model's, bit for bit (v + 0 * u = v exactly). With lam != 0 they differ (the switch is the only change)."""
    assume(abs(lam) > 1e-3)
    m = _llama(seed)
    ids = torch.randint(0, 64, (2, 9), generator=gen(seed))
    with torch.no_grad():
        plain = m(ids).logits
        at = m.model.layers[layer].self_attn
        vemb.add_value_mix(at, lam)
        vemb.set_mix(at, torch.randn(2, 9, at.v_proj.out_features, dtype=F64, generator=gen(seed + 1)))
        mixed = m(ids).logits
        vemb.value_emb_off(at)
        off = m(ids).logits
    assert torch.equal(off, plain)
    assert (mixed - plain).abs().max().item() > 1e-9


@FAST
@given(SEED, st.sampled_from(["silu", "gelu", "relu", "gelu_pytorch_tanh", "relu2", "mish"]), st.integers(1, 3))
def test_bias_free_ffn_zero(seed, act, n):
    """EP.bias_free_ffn_zero: a bias-free FFN whose activation has phi(0) = 0 (or a gated unit phi(g) u) writes
    exactly 0 on a zero input. On stock transformers LlamaMLP (gated) with random weights and six activations;
    block/marker_stream.is_bias_free confirms no projection has a bias."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    torch.manual_seed(seed)
    mlp = LlamaMLP(LlamaConfig(hidden_size=16, intermediate_size=24, hidden_act=act)).to(F64)
    assert mks.is_bias_free(mlp)
    with torch.no_grad():
        assert torch.equal(mlp(torch.zeros(n, 5, 16, dtype=F64)), torch.zeros(n, 5, 16, dtype=F64))


@SLOW
@given(SEED, st.integers(1, 16), FLOAT(0.1, 10))
def test_bias_free_ffn_zero_marker_stream(seed, e, kappa):
    """EP.bias_free_ffn_zero's consequence (the stream half of the start-token sink): on the hidden-padded host
    block/marker_stream.marker_host (elephantine/hidden_width.widen, width d + e) a start token embedded as
    kappa e_d (a pad channel no projection reads) keeps exactly that residual at every layer at position 0:
    its normed content is 0 on the read channels, its value is 0 (bias-free v_proj), it attends only to
    itself, and the bias-free MLP writes 0."""
    m = _llama(seed)
    host, mid = mks.marker_host(m, e, kappa)
    d = m.config.hidden_size
    ids = torch.randint(1, 64, (2, 8), generator=gen(seed))
    ids[:, 0] = mid
    want = torch.zeros(d + e, dtype=F64)
    want[d] = kappa
    with torch.no_grad():
        hs = host(ids, output_hidden_states=True).hidden_states
    for h in hs[:-1]:                                     # the last one is after the final norm
        assert torch.equal(h[:, 0], want.expand(2, -1))


def test_condition_bias_free_ffn_zero_needs_bias_free():
    """EP.bias_free_ffn_zero's hypothesis: a biased FFN (stock GPTNeoXMLP) writes its bias path on zero input."""
    from transformers import GPTNeoXConfig
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
    torch.manual_seed(0)
    mlp = GPTNeoXMLP(GPTNeoXConfig(hidden_size=16, intermediate_size=24, num_attention_heads=4)).to(F64)
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_()
        assert not mks.is_bias_free(mlp) and mlp(torch.zeros(1, 16, dtype=F64)).abs().max().item() > 1e-3


# =====================================================================================================
# PreNormRescue.lean: the exact pre-norm rescue (constant-norm channels, the strong-margin MoE select)
# =====================================================================================================

from norm import const_norm as cnm                        # noqa: E402
from moe import exact_select_prenorm as esp               # noqa: E402


@FAST
@given(SEED, st.integers(1, 32), st.integers(1, 12), FLOAT(1e-6, 1.0), FLOAT(0.0, 1.0), FLOAT(-2, 2))
def test_const_norm_carry(seed, d, e, eps, frac, logscale):
    """EP.const_norm_carry: extra channels z of fixed squared norm S <= d eps. torch's own F.rms_norm over [x; z]
    with norm/const_norm.const_norm_host's weight [sqrt(d/(d+e)) w; wz] and eps' = (d eps - S)/(d + e) returns
    RMSNorm_eps(x) on the stream and wz z / R on the extra channels (R = const_norm.carried_factor). S = frac d eps.
    Slack fp(S, 100 (d + e))."""
    r = rng(seed)
    x = torch.from_numpy(r.standard_normal(d) * 10 ** logscale)
    w, wz = torch.from_numpy(r.uniform(-2, 2, d)), torch.from_numpy(r.uniform(-2, 2, e))
    S = frac * d * eps
    z = torch.from_numpy(r.standard_normal(e))
    z = z * math.sqrt(S) / z.norm() if S > 0 else torch.zeros(e, dtype=F64)
    W, eps2 = cnm.const_norm_host(w, eps, wz, S)
    assert eps2 >= 0
    out = F.rms_norm(torch.cat([x, z]), (d + e,), W, eps2)
    want = F.rms_norm(x, (d,), w, eps)
    want_z = wz * z / cnm.carried_factor(x, eps, e)
    sc = max(want.abs().max().item(), want_z.abs().max().item()) + 1
    assert (out[:d] - want).abs().max().item() <= fp(sc, 100 * (d + e))
    assert (out[d:] - want_z).abs().max().item() <= fp(sc, 100 * (d + e))


def test_condition_const_norm_carry_needs_fixed_norm():
    """EP.const_norm_carry's fixed S: the host norm set for S misreads the stream when ||z||^2 is not S
    (the content-dependent case, EP.prenorm_readout_obstruction)."""
    d, e, eps = 8, 3, 0.1
    w = torch.ones(d, dtype=F64)
    W, eps2 = cnm.const_norm_host(w, eps, torch.ones(e, dtype=F64), 0.5)
    x = torch.linspace(-1, 1, d, dtype=F64)
    out = F.rms_norm(torch.cat([x, torch.full((e,), 0.6, dtype=F64)]), (d + e,), W, eps2)
    assert (out[:d] - F.rms_norm(x, (d,), w, eps)).abs().max().item() > 1e-3


@FAST
@given(SEED, st.integers(1, 64), FLOAT(-3, 3))
def test_rmsNorm_zero_sq_sum(seed, m, logscale):
    """EP.rmsNorm_zero_sq_sum: with eps = 0 and weight 1 the normed vector's squared norm is m, so the bias-free
    GLU norm/const_norm.glu_const_writer (2m SiLU units silu(u) u - silu(-u) u = u^2 in a stock LlamaMLP) writes
    the exact constant m into its channel and 0 elsewhere, for every h != 0. Slack fp(m, 100 m)."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    h = torch.from_numpy(rng(seed).standard_normal((4, m)) * 10 ** logscale)
    u = F.rms_norm(h, (m,), None, 0.0)
    assert (u.pow(2).sum(-1) - m).abs().max().item() <= fp(m, 10 * m)
    mlp = cnm.glu_const_writer(LlamaMLP(LlamaConfig(hidden_size=m, intermediate_size=2 * m + 3,
                                                    hidden_act="silu")).to(F64), channel=m // 2)
    with torch.no_grad():
        out = mlp(u)
    want = torch.zeros(4, m, dtype=F64)
    want[:, m // 2] = m
    assert (out - want).abs().max().item() <= fp(m, 100 * m)


def _select_problem(seed, d, E, eps, n=6):
    r = rng(seed)
    T = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))   # noqa: E731
    ew = 3
    Wr, P, V = T(r.standard_normal((E, d))), T(r.standard_normal((E, ew, d))), T(r.standard_normal((E, ew, d)))
    w, x = T(r.uniform(0.5, 1.5, d)), T(r.standard_normal((n, d)) * 2)
    return Wr, P, V, w, x


def _select_params(x, eps, w, Wr, P, V, E, d):
    """A delta within the budget S <= d eps, Lam giving a strong margin on every token, lam above every
    loser pre-activation times R / delta (the three conditions of EP.moe_select_prenorm_exact)."""
    _, z, p = esp.target_moe(x, eps, w, Wr, P, V)
    gaps = (z[:, :, None] - z[:, None, :]).abs() + torch.eye(E, dtype=F64) * 1e300
    gamma = gaps.min().item()
    delta = 0.9 * math.sqrt(2 * d * eps / max(E * (E - 1), 1))
    Lam = 1.01 * delta / gamma
    R = cnm.carried_factor(x, eps, E * E + 1)
    win = z.argmax(-1)
    loser = torch.ones_like(p, dtype=torch.bool)
    loser[torch.arange(x.shape[0]), win] = False
    lam = 1.1 * max(((p * R[:, :, None]).masked_fill(~loser, 0.0)).max().item(), 0.0) / delta + 1.0
    return gamma, delta, Lam, lam


@MID
@given(SEED, st.integers(1, 12), st.integers(1, 5), FLOAT(1e-5, 1.0))
def test_moe_select_prenorm_exact(seed, d, E, eps):
    """EP.moe_select_prenorm_exact, on moe/exact_select_prenorm.build_exact_select_host: residual [x; g; kappa],
    one RMSNorm with eps' = (d eps - S)/(d + E^2 + 1), a select ReLU FFN (clamped pair units, pad -> 0) and an
    expert ReLU FFN (masked experts, erase units, pad -> sqrt S). All three conclusions: both norm reads are the
    target's N(x); the expert layer's output is the winner's FFN, so the host stream is x + MoE(x); and the
    extra channels are restored ([0; sqrt S]). Slack: Lam amplifies the stream read's rounding into the pair
    features, so fp(S (1 + Lam + lam), 100 (d E ew)) on the outputs."""
    Wr, P, V, w, x = _select_problem(seed, d, E, eps)
    gamma, delta, Lam, lam = _select_params(x, eps, w, Wr, P, V, E, d)
    assume(gamma > 1e-6 and Lam < 1e8)
    host = esp.build_exact_select_host(eps, w, Wr, P, V, delta, Lam, lam)
    assert host.eps2 >= 0
    tr = {}
    r2 = host(x, tr)
    out, z, p = esp.target_moe(x, eps, w, Wr, P, V)
    assert torch.all(esp.strong_margin(z, delta, Lam))
    N = esp.rms_norm(x, w, eps)
    sN = N.abs().max().item() + 1
    assert (tr["n0"][:, :d] - N).abs().max().item() <= fp(sN, 100 * (d + E * E))
    assert (tr["n1"][:, :d] - N).abs().max().item() <= fp(sN, 100 * (d + E * E))
    S = (x + out).abs().max().item() + p.abs().max().item() * (1 + V.abs().max().item())
    tol = fp(S * (1 + Lam + lam) * (1 + V.abs().sum((0, 1)).max().item()), 100 * d * E * 3)
    assert (r2[:, :d] - (x + out)).abs().max().item() <= tol
    assert r2[:, d:-1].abs().max().item() <= fp(delta * (1 + Lam * z.abs().max().item()), 100 * d)
    assert abs(r2[:, -1] - math.sqrt(host.S)).max().item() <= fp(math.sqrt(host.S), 10)


@FAST
@given(SEED, st.integers(1, 8), FLOAT(1e-3, 10), FLOAT(1.001, 100.0))
def test_pairSel_sq_sum(seed, E, delta, boost):
    """EP.pairSel_sq_sum: on a strong margin (every pairwise gap Lam |z_e - z_e'| >= delta) the clamped pair
    features moe/exact_select_prenorm.pair_sel take only the values 0 and delta, and sum_{e,e'} g^2 =
    E(E-1)/2 delta^2 whatever z. Lam = boost delta / (the smallest gap). Each feature rounds at the scale of its
    pre-activation t_max = Lam (max z - min z): slack fp(t_max, 10) per feature, fp(delta t_max E^2, 10 E^2) on the sum."""
    z = torch.from_numpy(rng(seed).standard_normal(E) * 3)
    gaps = (z[:, None] - z[None, :]).abs() + torch.eye(E, dtype=F64) * 1e300
    assume(gaps.min().item() > 1e-9)
    Lam = boost * delta / gaps.min().item()
    assert bool(esp.strong_margin(z, delta, Lam))
    g = esp.pair_sel(z, delta, Lam)
    tmax = Lam * (z.max() - z.min()).item()              # the largest |pre-activation|: relu(t) - relu(t - delta)
    assert (g.pow(2).sum() - E * (E - 1) / 2 * delta ** 2).abs().item() <= fp(delta * tmax * E * E, 10 * E * E)
    assert torch.all((g.abs() <= fp(tmax, 10)) | ((g - delta).abs() <= fp(tmax, 10)))


def test_condition_moe_select_needs_strong_margin():
    """EP.moe_select_prenorm_exact's StrongMargin: with a gap below delta / Lam a pair feature is strictly between
    0 and delta, the select vector's squared norm is not S, and the host's first norm read after the select
    (the expert layer's input) misreads the stream."""
    d, E, eps = 6, 3, 0.5
    Wr, P, V, w, x = _select_problem(3, d, E, eps, n=1)
    _, z, _ = esp.target_moe(x, eps, w, Wr, P, V)
    gamma = ((z[:, :, None] - z[:, None, :]).abs() + torch.eye(E, dtype=F64) * 1e300).min().item()
    delta = 0.9 * math.sqrt(2 * d * eps / (E * (E - 1)))
    Lam = 0.5 * delta / gamma                         # half the needed slope: one pair is unsaturated
    host = esp.build_exact_select_host(eps, w, Wr, P, V, delta, Lam, 1.0)
    tr = {}
    host(x, tr)
    assert (tr["n1"][:, :d] - esp.rms_norm(x, w, eps)).abs().max().item() > 1e-6


# =====================================================================================================
# MLAHost.lean: a GQA target's value path on a stock MLA host (the sphere, full latent rank)
# =====================================================================================================

from attn import mla_host as mlh                          # noqa: E402


def _ds_config(**kw):
    from transformers import DeepseekV3Config
    base = dict(vocab_size=64, hidden_size=16, intermediate_size=24, moe_intermediate_size=8, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8,
                num_experts_per_tok=2, n_group=4, topk_group=2, kv_lora_rank=20, q_lora_rank=None,
                qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=6, first_k_dense_replace=0,
                max_position_embeddings=64, rope_scaling=None, rope_interleave=False)
    base.update(kw)
    cfg = DeepseekV3Config(**base)
    cfg._attn_implementation = "eager"
    return cfg


@SLOW
@given(SEED, FLOAT(0.1, 10), FLOAT(0.25, 4), st.integers(16, 24))
def test_mla_host_sphere_full_rank(seed, R, alpha, r):
    """EP.mla_host_sphere_full_rank, on attn/mla_host.write_mla_value_host: a stock LlamaAttention target (GQA,
    4 heads / 2 kv heads, width 16) and a stock DeepseekV3Attention host with latent rank r >= 16, its
    down-projection a scaled isometry (W^T W = alpha I) and M = (D/alpha) A W^T diag(1/w). On single-token
    inputs on the sphere |x| = R the host's output (o_proj of its MLA value, through the latent RMSNorm) is the
    target's A x. DeepSeek's RMSNorm and eager softmax run in float32: slack 16 U32 (|A x| + ||M|| |W x|/D)."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention, DeepseekV3RotaryEmbedding
    n = 16
    lc = LlamaConfig(hidden_size=n, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                     max_position_embeddings=64)
    lc._attn_implementation = "eager"
    torch.manual_seed(seed)
    tgt = LlamaAttention(lc, 0).to(F64).eval()
    hc = _ds_config(kv_lora_rank=r)
    host = DeepseekV3Attention(hc, 0).to(F64).eval()
    x = torch.randn(3, 1, n, dtype=F64, generator=gen(seed))
    x = R * x / x.norm(dim=-1, keepdim=True)
    w = torch.from_numpy(rng(seed).uniform(0.5, 2, r) * rng(seed + 1).choice([-1, 1], r))
    with torch.no_grad():
        A = mlh.single_token_matrix(tgt)
        W, M = mlh.write_mla_value_host(host, A, R, w, alpha, seed)
        assert torch.allclose(W.T @ W, alpha * torch.eye(n, dtype=F64), atol=1e-12)
        pe_t = LlamaRotaryEmbedding(lc)(x, torch.zeros(1, 1, dtype=torch.long))
        yt = tgt(x, pe_t, None)[0]
        pe_h = DeepseekV3RotaryEmbedding(hc)(x, torch.zeros(1, 1, dtype=torch.long))
        yh = host(x, pe_h, None)[0]
    want = x @ A.T
    assert (yt - want).abs().max().item() <= fp(want.abs().max().item() + 1, 100 * n)
    D = math.sqrt(alpha * R ** 2 / r + host.kv_a_layernorm.variance_epsilon)
    sc = want.abs().max().item() + M.abs().sum(-1).max().item() * (x @ W.T).abs().max().item() / D * w.abs().max().item()
    assert (yh - want).abs().max().item() <= 16 * U32 * sc


def test_condition_mla_host_needs_the_sphere():
    """EP.mla_host_sphere_full_rank's sphere (EP.mla_value_ray): off the sphere radius R the same host is wrong,
    since along a ray it is degree 0 (2x gives the same output) and the target degree 1."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention, DeepseekV3RotaryEmbedding
    lc = LlamaConfig(hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=4)
    torch.manual_seed(0)
    tgt = LlamaAttention(lc, 0).to(F64).eval()
    hc = _ds_config()
    host = DeepseekV3Attention(hc, 0).to(F64).eval()
    x = torch.randn(1, 1, 16, dtype=F64, generator=gen(0))
    x = 2.0 * x / x.norm()
    with torch.no_grad():
        A = mlh.single_token_matrix(tgt)
        mlh.write_mla_value_host(host, A, 1.0, torch.ones(20, dtype=F64))
        pe = DeepseekV3RotaryEmbedding(hc)(x, torch.zeros(1, 1, dtype=torch.long))
        assert (host(x, pe, None)[0] - x @ A.T).abs().max().item() > 1e-2


# =====================================================================================================
# MoERouting.lean: group-limited routing under expert duplication (stock DeepSeek-V3 MoE)
# =====================================================================================================

from moe import group_dup as gdup                         # noqa: E402


def _ds_moe(seed, norm, **kw):
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
    cfg = _ds_config(norm_topk_prob=norm, routed_scaling_factor=2.5, **kw)
    torch.manual_seed(seed)
    m = DeepseekV3MoE(cfg).float().eval()      # its forward accumulates in the router's float32 (fails in float64)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.5)
        m.gate.e_score_correction_bias.uniform_(0, 0.5)   # >= 0: HF masks dropped groups with score 0
    return m


@SLOW
@given(SEED, st.integers(1, 3), st.booleans(), st.sampled_from([(8, 4, 2, 2), (8, 2, 1, 3), (12, 3, 2, 2)]))
def test_moeR_group_copies(seed, m, norm, shape):
    """EP.moeR_group_copies, on moe/group_dup.group_dup_moe: a stock DeepseekV3MoE target (E experts, n_group G,
    topk_group tg, top-k) and its stock host with m copies per expert, copy r of group g its own host group
    (n_group G m, topk_group tg m, top-k k m). On every token whose target selection is strict at both levels
    (group scores and expert scores; random weights: all of them), the host's selected set is exactly the copies of
    the target's (router probed through gate.get_topk_indices), and the MoE outputs agree (renormalised gates:
    whole copies; plain gates: down_proj / m). The stock DeepseekV3MoE only runs in float32 (its accumulator takes
    the router's float32 dtype), so the slack is 32 U32 x the output scale."""
    E, G, tg, k = shape
    tgt = _ds_moe(seed, norm, n_routed_experts=E, n_group=G, topk_group=tg, num_experts_per_tok=k)
    host = gdup.group_dup_moe(tgt, m)
    x = torch.randn(1, 10, 16, dtype=F64, generator=gen(seed)).float()
    with torch.no_grad():
        logits = F.linear(x.view(-1, 16).float(), tgt.gate.weight.float())
        st_ = tgt.gate.get_topk_indices(logits.sigmoid())
        sh = host.gate.get_topk_indices(F.linear(x.view(-1, 16).float(), host.gate.weight.float()).sigmoid())
        yt, yh = tgt(x), host(x)
    for t in range(10):
        want = {gdup.dup_index(e, r, E, G, m) for e in st_[t].tolist() for r in range(m)}
        assert set(sh[t].tolist()) == want
    assert (yh - yt).abs().max().item() <= 32 * U32 * (yt.abs().max().item() + 1)


# =====================================================================================================
# LimitInstances2.lean: whole-model limit families on tiny random stacks (block/limit_stacks.py)
# =====================================================================================================
#
# Every check measures the theorem's constants on the built host and target: L_i (block/limit_stacks.lipschitz on
# the target layer, at the states of BOTH trajectories and random points near them), Lh = ||W_h||_inf->inf, the value
# / read bound Mb, the region constant (R, delta_r, rho), and the greedy margin delta (the target's smallest
# top1 - top2 logit gap over the generated steps), then evaluates the theorem's formula. Slack fp(S, 1e4) on the
# bound (float64 stacks of a few hundred operations).

from block import limit_stacks as ls                      # noqa: E402

VOCAB = 7


def _lhead(seed, d):
    Wh = torch.randn(VOCAB, d, dtype=F64, generator=gen(seed + 11)) * 2
    return (lambda x: (x[-1] if x.dim() == 2 else x) @ Wh.T), ls.opnorm_inf(Wh)


def _Ls(target_layers, *trajs):
    """Measured L_i: target layer i at the inputs of layer i along every trajectory given."""
    return [ls.lipschitz(f, [xs[i] for xs in trajs]) for i, f in enumerate(target_layers)]


def _attn_case(seed, n=2, T=5, d=4, W=2, Mb=1.0):
    stk = ls.AttnStack(n, T, d, W=W, Mb=Mb, qk_scale=1.5, seed=seed)
    emb = ls.SeqEmbed(VOCAB, T, d, seed=seed)
    head, Lh = _lhead(seed, d)
    prompt = rng(seed).integers(0, VOCAB, 3).tolist()
    return stk, emb, head, Lh, prompt


def _attn_bound(seed, kind, p, per_layer):
    """(err, bound, S) for one input: per_layer(stk, R) is the theorem's per-layer error."""
    stk, emb, head, Lh, prompt = _attn_case(seed)
    x = emb(prompt)
    xt, xh = ls.run(stk.layers("target"), x), ls.run(stk.layers(kind, p), x)
    R = max(stk.logit_max(xt), stk.logit_max(xh))
    Ls_ = _Ls(stk.layers("target"), xt, xh)
    err = (head(xh[-1]) - head(xt[-1])).abs().max().item()
    return err, ls.model_bound(Lh, Ls_, per_layer(stk, R)), head(xt[-1]).abs().max().item()


def _attn_greedy(seed, kind, threshold_param, k=4):
    """Target greedy run; measured constants over every step; the host at the theorem's threshold parameter.
    Returns (target tokens, host tokens) or None when the host leaves the measured region."""
    stk, emb, head, Lh, prompt = _attn_case(seed)
    tt, trt, lt = ls.greedy(stk.layers("target"), head, emb, prompt, k)
    delta = min(ls.margin(z) for z in lt)
    R = 1.05 * max(stk.logit_max(xs) for xs in trt)
    Ls_ = [ls.lipschitz(f, [xs[i] for xs in trt]) for i, f in enumerate(stk.layers("target"))]
    p = threshold_param(stk, Lh, math.prod(Ls_), stk.n, delta, R)
    th, trh, _ = ls.greedy(stk.layers(kind, p), head, emb, prompt, k)
    if max(stk.logit_max(xs) for xs in trh) > R:
        return None
    return tt, th


@SLOW
@given(SEED)
def test_mask_greedy(seed):
    """EP.mask_greedy: with B above B* = 2R + log(4 Lh (prod L) n Mb T / delta), the additive-mask host stack
    (block/limit_stacks.AttnStack.mask_host_layer: hf/local_full.band_mask clamped at -B, all T keys) generates
    exactly the hard-masked target's greedy tokens. Constants measured on the target's generation (R x 1.05);
    the host's states are checked to stay in that logit box."""
    def B_of(stk, Lh, PL, n, delta, R):
        return 2 * R + math.log(4 * Lh * PL * n * stk.Mb * stk.T / delta) + 0.5
    out = _attn_greedy(seed, "mask", B_of)
    assume(out is not None)
    assert out[0] == out[1]


@SLOW
@given(SEED, FLOAT(-30, 3))
def test_sink_model_bound(seed, b):
    """EP.sink_model_bound: the sink host stack (every layer an extra key with logit b and value 0,
    attn/attention_sink.sink_attention) has head logits within Lh (prod L_i) n 2 Mb e^{b + R} of the plain target
    stack, constants measured."""
    err, bound, S = _attn_bound(seed, "sink", b, lambda stk, R: 2 * stk.Mb * math.exp(b + R))
    assert err <= bound + fp(S, 1e4)


@SLOW
@given(SEED)
def test_sink_greedy(seed):
    """EP.sink_greedy: with b below b* = log(delta / (4 Lh (prod L) n Mb)) - R the sink host stack generates exactly
    the plain target's greedy tokens."""
    def b_of(stk, Lh, PL, n, delta, R):
        return math.log(delta / (4 * Lh * PL * n * stk.Mb)) - R - 0.5
    out = _attn_greedy(seed, "sink", b_of)
    assume(out is not None)
    assert out[0] == out[1]


@SLOW
@given(SEED)
def test_sink_model_limit(seed):
    """EP.sink_model_limit: along b = 0, then -R, -R - 5, ..., -R - 40 (R the logit bound; the rate is e^{b + R}) the sink host stack's states tend to the target's (the gaps
    shrink, the last within float64 noise: the host takes a different route through the sink formula, so the floor
    is fp(S, 1e5)), for several inputs at once (uniform on the set)."""
    stk, emb, head, Lh, _ = _attn_case(seed)
    xs = [emb(rng(seed + j).integers(0, VOCAB, 3).tolist()) for j in range(3)]
    tgt = [ls.run(stk.layers("target"), x)[-1] for x in xs]
    R = max(stk.logit_max(ls.run(stk.layers("target"), x)) for x in xs)
    gaps = []
    for b in [R] + list(range(0, -45, -5)):
        b = b - R                                         # b = 0, then -R - k: the rate is e^{b + R}
        gaps.append(max((ls.run(stk.layers("sink", float(b)), x)[-1] - t).abs().max().item() for x, t in zip(xs, tgt)))
    S = max(t.abs().max().item() for t in tgt)
    assert all(g1 <= g0 + fp(S, 1e5) for g0, g1 in zip(gaps, gaps[1:]))
    assert gaps[-1] <= fp(S, 1e5) and gaps[0] > 1e-6


@SLOW
@given(SEED, FLOAT(1.0, 1e4))
def test_cap_model_bound(seed, c):
    """EP.cap_model_bound: the attention-softcap host stack (c tanh(s/c) in every layer, attn/softcap.softcap) has
    head logits within Lh (prod L_i) n (4/3) R^3 Mb / c^2 of the plain stack when 2 R^3 <= 3 c^2, and within the
    exp form Lh (prod L_i) n (e^{2R^3/(3c^2)} - 1) Mb (EP.cap_layer_bound) always."""
    err, bound_exp, S = _attn_bound(seed, "cap", c,
                                    lambda stk, R: (math.exp(min(2 * R ** 3 / (3 * c * c), 700)) - 1) * stk.Mb)
    assert err <= bound_exp + fp(S, 1e4)
    err, bound, S = _attn_bound(seed, "cap", c, lambda stk, R: 4 / 3 * R ** 3 * stk.Mb / c ** 2
                                if 2 * R ** 3 <= 3 * c * c else math.inf)
    assert err <= bound + fp(S, 1e4)


@SLOW
@given(SEED)
def test_cap_greedy(seed):
    """EP.cap_greedy: with c above c* = sqrt(8 Lh (prod L) n R^3 Mb / (3 delta)) (and 2R^3 <= 3c^2) the softcap host
    stack generates exactly the plain target's greedy tokens."""
    def c_of(stk, Lh, PL, n, delta, R):
        return 1.1 * max(math.sqrt(8 * Lh * PL * n * R ** 3 * stk.Mb / (3 * delta)), math.sqrt(2 * R ** 3 / 3))
    out = _attn_greedy(seed, "cap", c_of)
    assume(out is not None)
    assert out[0] == out[1]


@SLOW
@given(SEED)
def test_cap_model_limit(seed):
    """EP.cap_model_limit: along c = 1, 10, ..., 1e7 the softcap host stack's states tend to the target's (the last
    within float64 noise, fp(S, 1e5): tanh rounds differently from the identity) at the 1/c^2 rate (gap x c^2
    does not grow from c = 1e2 to 1e3), on several inputs."""
    stk, emb, head, Lh, _ = _attn_case(seed)
    xs = [emb(rng(seed + j).integers(0, VOCAB, 3).tolist()) for j in range(3)]
    tgt = [ls.run(stk.layers("target"), x)[-1] for x in xs]
    S = max(t.abs().max().item() for t in tgt)
    gaps = {k: max((ls.run(stk.layers("cap", 10.0 ** k), x)[-1] - t).abs().max().item() for x, t in zip(xs, tgt))
            for k in range(0, 8)}
    assert gaps[7] <= fp(S, 1e5) and gaps[0] > 1e-6
    assert gaps[3] * 1e6 <= 2 * gaps[2] * 1e4 + fp(S, 1e5) * 1e6


def _router_case(seed, n=2, d=4, E=4):
    stk = ls.RouterStack(n, d, E, seed=seed)
    emb = ls.BagEmbed(VOCAB, d, seed=seed)
    head, Lh = _lhead(seed, d)
    return stk, emb, head, Lh, rng(seed).integers(0, VOCAB, 3).tolist()


@SLOW
@given(SEED, FLOAT(0.0, 200.0))
def test_soft_router_model_bound(seed, beta):
    """EP.soft_router_model_bound: the dense soft-router stack (moe/dense_emulates_moe.dense_softrouter in every
    layer) has head logits within Lh (prod L_i) n 2 Mb (E - 1) e^{-beta delta_r} of the top-1 stack (moe_top1),
    delta_r the smallest router margin and Mb the largest expert output over both trajectories."""
    stk, emb, head, Lh, prompt = _router_case(seed)
    x = emb(prompt)
    xt, xh = ls.run(stk.layers("target"), x), ls.run(stk.layers("host", beta), x)
    dr = min(stk.margin_min(xt), stk.margin_min(xh))
    Mb = max(stk.expert_max(xt), stk.expert_max(xh))
    Ls_ = _Ls(stk.layers("target"), xt, xh)
    err = (head(xh[-1]) - head(xt[-1])).abs().max().item()
    bound = ls.model_bound(Lh, Ls_, 2 * Mb * (stk.E - 1) * math.exp(-beta * dr))
    assert err <= bound + fp(head(xt[-1]).abs().max().item(), 1e4)


@SLOW
@given(SEED)
def test_soft_router_greedy(seed):
    """EP.soft_router_greedy: with beta above beta* = log(4 Lh (prod L) n Mb (E - 1) / delta) / delta_r the soft-router
    stack generates exactly the top-1 target's greedy tokens (constants measured on the target's generation; the
    host's margins and expert outputs are checked to stay in that region)."""
    stk, emb, head, Lh, prompt = _router_case(seed)
    tt, trt, lt = ls.greedy(stk.layers("target"), head, emb, prompt, 4)
    delta = min(ls.margin(z) for z in lt)
    dr = 0.95 * min(stk.margin_min(xs) for xs in trt)
    Mb = 1.05 * max(stk.expert_max(xs) for xs in trt)
    assume(dr > 1e-3)
    PL = math.prod(ls.lipschitz(f, [xs[i] for xs in trt]) for i, f in enumerate(stk.layers("target")))
    beta = math.log(max(4 * Lh * PL * stk.n * Mb * (stk.E - 1) / delta, 1.0)) / dr + 1.0
    th, trh, _ = ls.greedy(stk.layers("host", beta), head, emb, prompt, 4)
    assume(min(stk.margin_min(xs) for xs in trh) >= dr and max(stk.expert_max(xs) for xs in trh) <= Mb)
    assert th == tt


@SLOW
@given(SEED)
def test_soft_router_model_limit(seed):
    """EP.soft_router_model_limit: along beta = 0, 25, ..., 400 the soft-router stack tends to the top-1 stack (the
    last within float64 noise), on several inputs at once whose trajectories keep a router margin."""
    stk, emb, head, Lh, _ = _router_case(seed)
    xs = [emb(rng(seed + j).integers(0, VOCAB, 3).tolist()) for j in range(3)]
    tr = [ls.run(stk.layers("target"), x) for x in xs]
    assume(min(stk.margin_min(t) for t in tr) > 0.2)
    S = max(t[-1].abs().max().item() for t in tr)
    gaps = [max((ls.run(stk.layers("host", float(beta)), x)[-1] - t[-1]).abs().max().item() for x, t in zip(xs, tr))
            for beta in range(0, 425, 25)]
    assert gaps[-1] <= fp(S, 1e3) and gaps[0] > 1e-6


def _read_case(seed, n=3, d=5):
    stk = ls.ReadStack(n, d, seed=seed)
    emb = ls.BagEmbed(VOCAB, d, seed=seed)
    head, Lh = _lhead(seed, d)
    return stk, emb, head, Lh, rng(seed).integers(0, VOCAB, 3).tolist()


@SLOW
@given(SEED, FLOAT(1e-4, 1.0))
def test_scaled_read_model_bound(seed, delta):
    """EP.scaled_read_model_bound: every layer's read reaches its consumer x + A_i y through the padded host norm
    over [x; delta s_i] (block/limit_stacks.ReadStack.host_read: F.rms_norm with eps d/(d+e), the algebra of
    moe/dense_host_prenorm.HostNorm), so scaled by lambda with |lambda - 1| <= rho = delta^2 |s|^2 / (2 d eps):
    head logits within Lh (prod L_i) n K Mb rho, K = max ||A_i||_inf, Mb = max|w_i| sqrt(d)."""
    stk, emb, head, Lh, prompt = _read_case(seed)
    x = emb(prompt)
    xt, xh = ls.run(stk.layers("target"), x), ls.run(stk.layers("host", delta), x)
    Ls_ = _Ls(stk.layers("target"), xt, xh)
    err = (head(xh[-1]) - head(xt[-1])).abs().max().item()
    bound = ls.model_bound(Lh, Ls_, stk.K() * stk.read_max() * stk.rho(delta))
    assert err <= bound + fp(head(xt[-1]).abs().max().item(), 1e4)


@SLOW
@given(SEED)
def test_scaled_read_greedy(seed):
    """EP.scaled_read_greedy: with rho below rho* = delta / (2 Lh (prod L) n K Mb), i.e. a carry scale delta with
    delta^2 max|s|^2 / (2 d eps) < rho*, the scaled-read host stack generates exactly the target's greedy tokens."""
    stk, emb, head, Lh, prompt = _read_case(seed)
    tt, trt, lt = ls.greedy(stk.layers("target"), head, emb, prompt, 4)
    marg = min(ls.margin(z) for z in lt)
    PL = math.prod(ls.lipschitz(f, [xs[i] for xs in trt]) for i, f in enumerate(stk.layers("target")))
    rho_star = marg / (2 * Lh * PL * stk.n * stk.K() * stk.read_max())
    dl = 0.9 * math.sqrt(rho_star / stk.rho(1.0))
    assert stk.rho(dl) < rho_star
    th, _, _ = ls.greedy(stk.layers("host", dl), head, emb, prompt, 4)
    assert th == tt


@SLOW
@given(SEED)
def test_scaled_read_model_limit(seed):
    """EP.scaled_read_model_limit: along delta = 1, 0.1, ..., 1e-8 the scaled-read host stack tends to the target
    (rho ~ delta^2: gap / delta^2 does not grow from 1e-2 to 1e-3), the last within float64 noise, on several
    inputs at once."""
    stk, emb, head, Lh, _ = _read_case(seed)
    xs = [emb(rng(seed + j).integers(0, VOCAB, 3).tolist()) for j in range(3)]
    tgt = [ls.run(stk.layers("target"), x)[-1] for x in xs]
    S = max(t.abs().max().item() for t in tgt)
    gaps = {k: max((ls.run(stk.layers("host", 10.0 ** -k), x)[-1] - t).abs().max().item() for x, t in zip(xs, tgt))
            for k in range(0, 9)}
    assert gaps[8] <= fp(S, 1e3) and gaps[0] > 1e-8
    assert gaps[3] * 1e6 <= 2 * gaps[2] * 1e4 + fp(S, 1e3) * 1e6


@SLOW
@given(SEED, st.integers(1, 8))
def test_compensator_model_bound(seed, deg):
    """EP.compensator_model_bound: every layer's magnitude read u_i(x) = U_i x reaches the residual as
    u_i(x) t(x)/kappa (t the normed constant pad of a host norm over [x; kappa]) times a degree-`deg` polynomial
    fit g(t) of kappa/t on the pad range (block/limit_stacks.CompensatorStack.fit; rho its sup relative error on
    the range): head logits within Lh (prod L_i) n Mb rho, Mb = max |u_i(x)| over both trajectories."""
    stk = ls.CompensatorStack(3, 4, seed=seed)
    emb = ls.BagEmbed(VOCAB, 4, seed=seed)
    head, Lh = _lhead(seed, 4)
    x = emb(rng(seed).integers(0, VOCAB, 3).tolist())
    xt = ls.run(stk.layers("target"), x)
    ts = [stk.t(s).item() for s in xt[:-1]]
    tmin, tmax = 0.8 * min(ts), 1.25 * max(ts)
    stk.fit(deg, tmin, tmax)
    xh = ls.run(stk.layers("host"), x)
    th = [stk.t(s).item() for s in xh[:-1]]
    assume(tmin <= min(th) and max(th) <= tmax)          # the region hypothesis hK
    Mb = max(stk.u(i, s).abs().max().item() for traj in (xt, xh) for i, s in enumerate(traj[:-1]))
    Ls_ = _Ls(stk.layers("target"), xt, xh)
    err = (head(xh[-1]) - head(xt[-1])).abs().max().item()
    bound = ls.model_bound(Lh, Ls_, Mb * stk.rho)
    assert err <= bound + fp(head(xt[-1]).abs().max().item(), 1e4)
