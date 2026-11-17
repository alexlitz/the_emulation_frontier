"""The universal host's residual-stream / input / FFN / norm / loop branches (elephantine/branches_stream.py), each
run against the target family's own code: a stock transformers class (4.57 in process, 5.18 in a subprocess), or a
REFERENCE transcription of its remote code (read as text, never run; each names the file, its hash and the lines it
transcribes). Every test also shows that switching the new branch off moves the output.

Run:  pytest tests/test_eleph_gaps_stream.py -q
"""
import math

import pytest
import torch

from tests.test_universal_elephantine import _tf5_run

F64 = torch.float64

LIFT = '''
import contextlib
import torch.nn.functional as tF
@contextlib.contextmanager
def lift():
    T, zl, sm = torch.Tensor, torch.zeros_like, tF.softmax
    of, ot, oty = T.float, T.to, T.type
    def to(self, *a, **k):
        a = tuple(F64 if v is torch.float32 else v for v in a)
        if k.get("dtype") is torch.float32:
            k["dtype"] = F64
        return ot(self, *a, **k)
    T.float, T.to = (lambda self: ot(self, F64)), to
    T.type = lambda self, *a, **k: oty(self, *(F64 if v is torch.float32 else v for v in a), **k)
    torch.zeros_like = lambda v, **k: zl(v, **({**k, "dtype": F64} if k.get("dtype") is torch.float32 else k))
    tF.softmax = lambda v, dim=None, dtype=None, **k: sm(v, dim=dim, dtype=F64 if dtype is torch.float32 else dtype, **k)
    try:
        yield
    finally:
        T.float, T.to, T.type, torch.zeros_like, tF.softmax = of, ot, oty, zl, sm
def rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()
'''


# --- Qwen4-Exp: GatedResidual hyper-connections (hc_qwen4_exp) + the n-gram PLE (engram_qwen4) -------------------

@pytest.mark.parametrize("gate", ["sigmoid", "silu"])
def test_same_host_carries_qwen4_exp_gated_residual_ngram_ple(gate):
    """Qwen4-Exp (transformers 5.18 models/qwen4_exp, the stock class, in a subprocess) vs from_qwen4_exp: three
    GatedResidual streams (the only norms: no sub-block input norm, no final norm), the n-gram PLE on layer 2 (EOS-reset
    segments: the ids contain EOS), gated-DeltaNet with a sigmoid (Qwen3.8-Flash-Next's output_gate_type) or silu
    output gate, gated partial-RoPE attention (its QSA indexer with a budget covering the context: dense), the sparse
    MoE + gated shared expert. fp64 with the reference's fp32 casts lifted; then each new branch switched off (the
    PLE, the stream injection weights, the gated collapse) moves the logits."""
    gap, scale = _tf5_run(LIFT + f'''
from transformers.models.qwen4_exp import modeling_qwen4_exp as q, configuration_qwen4_exp as qc
from elephantine.universal import from_qwen4_exp
cfg = qc.Qwen4ExpTextConfig(vocab_size=97, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
    linear_value_head_dim=8, linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
    shared_expert_intermediate_size=24, hc_count=3, hc_lowrank=8, ple_layer_ids=[2], ple_embed_dim=32,
    ple_conv_kernel_size=3, ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=50,
    make_ngram_vocab_size_divisible_by=8, eos_token_id=5, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
    indexer_budget=16, indexer_compress_ratio=4, output_gate_type="{gate}", tie_word_embeddings=False,
    rope_parameters={{"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [1, 1, 0], "mrope_interleaved": True}},
    attn_implementation="eager", experts_implementation="eager")
t = q.Qwen4ExpForCausalLM(cfg).to(F64).eval()
for l in t.model.layers:
    g = getattr(l, "linear_attn", None)
    if g is not None:
        g.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
        g.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
        g.causal_conv1d_fn = None
rand_(t)
with torch.no_grad():
    for l in t.model.layers:
        if l.ple is not None:
            l.ple.conv1d.weight.normal_(0, 0.3)                  # (initialised to zero)
x = ids(97, T=12)
x[0, 4] = 5; x[1, 2] = 5; x[1, 9] = 5                          # EOS: the n-gram segments restart
assert t.model.layers[1].ple is not None and t.model.layers[3].layer_type == "indexed_attention"
with torch.no_grad():
    u = from_qwen4_exp(t)
    ours = u(x)
    with lift():
        ref = t(x, use_cache=False).logits
    assert ref.dtype == F64
    moved = []
    for name, off in (("ple", lambda h: h.layers[1].engram_hc.c.zero_()),
                      ("inject", lambda h: [l.hc_a.inject.weight.zero_() for l in h.layers]),
                      ("mix", lambda h: h.hc_final.up.weight.zero_())):
        h = from_qwen4_exp(t); off(h)
        moved.append(rel(ref, h(x)))
    assert min(moved) > 1e-3, moved
report(ref, ours)
''')
    assert gap < 1e-12 * max(scale, 1), (gap, scale)


# --- Zamba2: the shared transformer block over concat(x, embedding) (zamba_shared) --------------------------------

def _lift():
    from tests.test_universal_elephantine import _lift_fp32
    return _lift_fp32()


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


def _rand_(model, seed=0, s=0.05):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * s)
    return model


@pytest.mark.parametrize("adapters,rope", [(True, True), (False, False)])
@torch.no_grad()
def test_same_host_carries_zamba2_shared_block(adapters, rope):
    """Zamba2 (the stock transformers Zamba2ForCausalLM, its torch Mamba-2 path) vs from_zamba2: Mamba-2 layers, two
    shared transformer blocks cycled over the hybrid layers, each reading concat(x, embedding) through a 2d-wide
    RMSNorm, attention (scale (hd / 2)^-1/2, optional RoPE) -> RMSNorm -> GeGLU MLP with no inner residual, then the
    hybrid layer's own linear into the Mamba-2 layer's input (UCatBlock); per-invocation LoRA adapters on q / k / v and
    gate_up fold into the untied copies (the k / v adapters are 2d wide: MHA there, GQA without them). One SSD chunk
    (chunk_size >= T): transformers 4.57's naive multi-chunk Zamba2 path sums its inter-chunk decay over the wrong
    index (modeling_zamba2.py:882 .sum(dim=2); Mamba-2 / Bamba transpose first, modeling_mamba2.py:619-620), so
    there it is not the SSD recurrence the CUDA kernels compute. fp64 with the fp32 casts lifted; the block switched off moves the logits."""
    from transformers import Zamba2Config, Zamba2ForCausalLM
    from elephantine.universal import from_zamba2
    cfg = Zamba2Config(vocab_size=97, hidden_size=32, num_hidden_layers=6,
                       layers_block_type=["mamba", "hybrid", "mamba", "hybrid", "hybrid", "mamba"],
                       mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, n_mamba_heads=8, mamba_ngroups=2, chunk_size=16,
                       intermediate_size=48, num_attention_heads=4, num_key_value_heads=4 if adapters else 2, num_mem_blocks=2,
                       use_shared_attention_adapter=adapters, adapter_rank=4, use_mem_rope=rope, hidden_act="gelu",
                       attn_implementation="eager")
    t = _rand_(Zamba2ForCausalLM(cfg).to(F64).eval())
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    u = from_zamba2(t)
    ours = u(x)
    with _lift():
        ref = t(x, use_cache=False).logits
    assert ref.dtype == F64 and _rel(ref, ours) < 1e-12, _rel(ref, ours)
    for ul in u.layers:
        if ul.cat is not None:
            ul.cat.c.zero_()
    assert _rel(ref, u(x)) > 1e-3


# --- nanochat ports: value embeddings, smear, x0 / resid lambdas, backout (value_emb, backout) ---------------------
# REFERENCE, not the remote code: a transcription (read as text, never run) of
#   jkminder/d16_optimized_537m_seed5 modeling_nanochat_gpt.py (snapshot 34be31b8, md5 6cdb858d8765ac1d68301117bcd420bb):
#     rms_norm l. 104-107, apply_rotary_emb l. 110-119, compute_rotary_cos_sin l. 122-131, sliding_window_sdpa
#     l. 151-181, NanochatGPTAttention.forward l. 208-246 (the reference path), NanochatGPTMLP l. 262-269,
#     NanochatGPTBlock l. 272-281, NanochatGPTModel.__init__ l. 333-348 / _smear l. 359-365 / forward l. 450-497,
#     NanochatGPTForCausalLM.forward l. 544-548 (the logit softcap);
#   castorini/gaggle-nanochat-pretrained-climbmix-20260924 modeling_gaggle.py (snapshot be5162e6, md5
#     be7c1bded679987aa391d8a565264281; configuration_gaggle.py md5 0768c2316c5ae34ae6b0d7e865e400a1): the same
#     blocks under its own names (GaggleAttention l. 84-121, GaggleMLP l. 124-135, GaggleModel.__init__ l. 158-172,
#     forward l. 281-318: resid / x0 lambdas unconditional, backout at n_layer // 2, VE on layers i % 2 == (n - 1) % 2
#     (configuration_gaggle.py has_value_embedding l. 87-89), the logit softcap l. 352-355;
#   chronollm/sn38 modeling_nanochrono.py (~/.cache/nocode/sn38-nanochrono, md5 d30ac9f4ab35e3f9245bfdb5ac326e33):
#     _norm / _rotate l. 12-19, NanochronoAttention l. 22-82 (aux value tables, g_proj gate x 3, qk_scale =
#     attention_multiplier), NanochronoMLP l. 85-92, NanochronoModel l. 107-208 (residual_scales / skip_scales,
#     mix_gate / mix_scale smear, tap_scale at tap_layer: the backout), the softcap l. 244-246.
# The module names follow each file so the loader reads them as it reads the real checkpoints. The short windows are
# set on the attention modules (as the files' models do from their window list) small enough to bind at T = 12.

import torch.nn as _nn
import torch.nn.functional as _F


def _nc_rms(x):
    return _F.rms_norm(x, (x.size(-1),))


def _nc_rot(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


def _nc_sdpa(q, k, v, window):
    Tq, Tk = q.size(2), k.size(2)
    gqa = q.size(1) != k.size(1)
    if (window < 0 or window >= Tq) and Tq == Tk:
        return _F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    row = (Tk - Tq) + torch.arange(Tq).unsqueeze(1)
    col = torch.arange(Tk).unsqueeze(0)
    mask = col <= row
    if 0 <= window < Tk:
        mask = mask & ((row - col) <= window)
    return _F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)


class _NcAttn(_nn.Module):
    def __init__(self, d, H, nkv, gc, ve, scale, fl):
        super().__init__()
        hd = d // H
        gag, sn = fl == "gaggle", fl == "sn38"
        self.head_dim = hd
        if sn:
            self.num_heads, self.num_kv_heads, self.gate_dim = H, nkv, gc
        else:
            self.n_head, self.n_kv_head, self.ve_gate_channels = H, nkv, gc
        self.H_, self.nkv_, self.gc_ = H, nkv, gc
        names = ("c_q", "c_k", "c_v", "c_proj") if gag else ("q_proj", "k_proj", "v_proj", "o_proj")
        for n_, o in zip(names, (H * hd, nkv * hd, nkv * hd, d)):
            setattr(self, n_, _nn.Linear(d, o, bias=False))
        self.names = names
        self.gname = "g_proj" if sn else "ve_gate"
        setattr(self, self.gname, _nn.Linear(gc, nkv, bias=False) if ve else None)
        setattr(self, "qk_sharpen_scale" if fl == "nanochat_gpt" else "qk_scale", scale)
        self.scale_ = scale
        self.window = None

    def forward(self, x, ve, cos, sin):
        B, T, C = x.size()
        q, k, v, o = (getattr(self, n_) for n_ in self.names)
        q = q(x).view(B, T, self.H_, self.head_dim)
        k = k(x).view(B, T, self.nkv_, self.head_dim)
        v = v(x).view(B, T, self.nkv_, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.nkv_, self.head_dim)
            gate = 3 * torch.sigmoid(getattr(self, self.gname)(x[..., :self.gc_]))
            v = v + gate.unsqueeze(-1) * ve
        q, k = _nc_rot(q, cos, sin), _nc_rot(k, cos, sin)
        q, k = _nc_rms(q), _nc_rms(k)
        if self.scale_ is not None:
            q, k = q * self.scale_, k * self.scale_
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = _nc_sdpa(q, k, v, self.window).transpose(1, 2).contiguous().view(B, T, -1)
        return o(y)


class _NcMLP(_nn.Module):
    def __init__(self, d, ff, fl):
        super().__init__()
        self.names = ("c_fc", "c_proj") if fl == "gaggle" else ("up_proj", "down_proj")
        setattr(self, self.names[0], _nn.Linear(d, ff, bias=False))
        setattr(self, self.names[1], _nn.Linear(ff, d, bias=False))

    def forward(self, x):
        up, dn = (getattr(self, n_) for n_ in self.names)
        return dn(_F.relu(up(x)).square())


class _NcBlock(_nn.Module):
    def __init__(self, a, mlp, fl):
        super().__init__()
        self.gag = fl == "gaggle"
        setattr(self, "attn" if self.gag else "self_attn", a)
        self.mlp = mlp

    def forward(self, x, ve, cos, sin):
        a = self.attn if self.gag else self.self_attn
        x = x + a(_nc_rms(x), ve, cos, sin)
        return x + self.mlp(_nc_rms(x))


class _NcModel(_nn.Module):
    def __init__(self, cfg, fl):
        super().__init__()
        d, H, nkv, L, V = cfg.d, cfg.H, cfg.nkv, cfg.L, cfg.vocab
        hd = d // H
        gag, sn = fl == "gaggle", fl == "sn38"
        ve_layers = [i for i in range(L) if i % 2 == (L - 1) % 2] if gag else cfg.ve_layers
        blocks = [_NcBlock(_NcAttn(d, H, nkv, cfg.gc, i in ve_layers, cfg.scale, fl), _NcMLP(d, 4 * d, fl), fl)
                  for i in range(L)]
        if gag:
            self.transformer = _nn.ModuleDict({"wte": _nn.Embedding(V, d), "h": _nn.ModuleList(blocks)})
        else:
            self.embed_tokens, self.layers = _nn.Embedding(V, d), _nn.ModuleList(blocks)
        self.gag, self.fl = gag, fl
        setattr(self, "windows" if sn else "window_sizes", cfg.windows)
        for b, w in zip(blocks, cfg.windows):
            (b.attn if gag else b.self_attn).window = w
        P = lambda v: _nn.Parameter(v)
        names = (("residual_scales", "skip_scales", "mix_gate", "mix_scale", "tap_scale", "aux_embeds") if sn else
                 ("resid_lambdas", "x0_lambdas", "smear_gate", "smear_lambda", "backout_lambda", "value_embeds"))
        self.names = names
        setattr(self, names[0], P(torch.ones(L)))
        setattr(self, names[1], P(torch.zeros(L)))
        setattr(self, names[2], _nn.Linear(cfg.smear, 1, bias=False))
        setattr(self, names[3], P(torch.zeros(1)))
        setattr(self, names[4], P((0.2 if gag else 0.0) * torch.ones(1)))
        setattr(self, names[5], _nn.ModuleDict({str(i): _nn.Embedding(V, nkv * hd) for i in ve_layers}))
        self.cfg_ = cfg

    def forward(self, ids):
        cfg, (B, T) = self.cfg_, ids.shape
        hd = cfg.d // cfg.H
        emb = self.transformer["wte"] if self.gag else self.embed_tokens
        blocks = self.transformer["h"] if self.gag else self.layers
        x = _nc_rms(emb(ids))
        ch = cfg.smear
        rl, xl, sg, sl, bl, vt = (getattr(self, n_) for n_ in self.names)
        gate = sl.to(x.dtype) * torch.sigmoid(sg(x[:, 1:, :ch]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        inv_freq = 1.0 / (cfg.theta ** (torch.arange(0, hd, 2, dtype=x.dtype) / hd))
        fr = torch.outer(torch.arange(T, dtype=x.dtype), inv_freq)
        cos, sin = fr.cos()[None, :, None, :], fr.sin()[None, :, None, :]
        x0 = x
        bo = cfg.L // 2 if self.gag else cfg.backout_layer
        x_backout = None
        for i, blk in enumerate(blocks):
            x = rl[i] * x + xl[i] * x0
            ve = vt[str(i)](ids) if str(i) in vt else None
            x = blk(x, ve, cos, sin)
            if i == bo:
                x_backout = x
        if x_backout is not None:
            x = x - bl * x_backout
        return _nc_rms(x)


class _NcLM(_nn.Module):
    def __init__(self, cfg, fl):
        super().__init__()
        from types import SimpleNamespace as NS
        self.model = _NcModel(cfg, fl)
        self.lm_head = _nn.Linear(cfg.d, cfg.vocab, bias=False)
        if fl == "sn38":
            self.config = NS(final_logit_softcapping=cfg.cap, rope_theta=cfg.theta, tap_layer=cfg.backout_layer)
        else:
            self.config = NS(logit_softcap=cfg.cap, rope_theta=cfg.theta, n_layer=cfg.L, backout_layer=cfg.backout_layer)

    def forward(self, ids):
        logits = self.lm_head(self.model(ids))
        cap = self.config.final_logit_softcapping if self.model.fl == "sn38" else self.config.logit_softcap
        return cap * torch.tanh(logits / cap)


@pytest.mark.parametrize("fl", ["nanochat_gpt", "gaggle", "sn38"])
@torch.no_grad()
def test_same_host_carries_nanochat_value_embeddings_backout(fl):
    """nanochat_gpt (jkminder d*_optimized), Gaggle and sn38-nanochrono vs from_nanochat, each against the REFERENCE
    transcription above: the smear, x <- r_i x + s_i x0, GQA with nanochat's -theta rotary, weight-free q / k norms after the rotary
    x the sharpening scale, left windows (3 on the short layers: they bind at T = 12), gated value embeddings on the VE
    layers, relu^2 MLPs, the backout before the final norm, the 15-softcap. fp64 (F.rms_norm's eps = the dtype's
    epsilon). Each new branch switched off -- the value embeddings, the smear, the x0 mix, the backout -- moves the
    logits."""
    from types import SimpleNamespace as NS
    from elephantine.universal import from_nanochat
    cfg = NS(d=32, H=4, nkv=2, L=6, vocab=97, gc=4, smear=6, scale=1.2, theta=100000.0, cap=15.0,
             backout_layer=3, ve_layers=[1, 3, 5], windows=[3, 3, 3, 64, 3, 64])
    t = _rand_(_NcLM(cfg, fl).to(F64).eval(), s=0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    ref = t(x)
    u = from_nanochat(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    moved = {}
    for name, off in (("ve", lambda h: [l.attn.ve.c.zero_() for l in h.layers if l.attn.ve is not None]),
                      ("smear", lambda h: h.smear.lam.zero_()),
                      ("x0", lambda h: [l.x0mix.s.zero_() for l in h.layers]),
                      ("backout", lambda h: h.bo_c.zero_())):
        h = from_nanochat(t)
        off(h)
        moved[name] = _rel(ref, h(x))
    assert min(moved.values()) > 1e-4, moved


# --- ARLoop / PreNormLoop: shared value embeddings, looped Block AttnRes with a source window ---------------------
# REFERENCE, not the remote code: a transcription (read as text, never run) of osunlp/ARLoop-1.2B modeling_arloop.py
# (snapshot a29f310a, md5 9469943eb1480ae16099847c19d668c5; ARLoop-0.6B byte-identical): RMSNorm l. 15-24,
# Attention.__init__ / forward l. 27-75 (ve_gate on layer % 2 == (physical_layers - 1) % 2, gate 2 sigmoid(x[:32]),
# adjacent-pair rotary at positions / rope_condense_ratio, weight-free F.rms_norm q / k after it), MLP l. 78-85,
# ARLoopBlock l. 88-93, _route l. 127-145, forward l. 182-209 (loop_count passes over the physical layers, blocks of
# attnres_block_size // 2 effective layers, the tied head); and osunlp/PreNormLoop-0.6B modeling_baseline.py (snapshot
# e4ad59d0, md5 d3f841e847e7fded2381220b75c9bcb1): the same blocks (norm_1 / norm_2) on a plain pre-norm residual,
# forward l. 163-177. The autocast / bf16 casts (numerics, not the function) are left out: fp64 throughout.

class _ArNorm(_nn.Module):
    def __init__(self, n, eps):
        super().__init__()
        self.weight, self.eps = _nn.Parameter(torch.ones(n)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class _ArAttn(_nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.c = c
        dim, kv = c.hidden_size, c.num_key_value_heads * c.head_dim
        self.c_q, self.c_k = _nn.Linear(dim, dim, bias=False), _nn.Linear(dim, kv, bias=False)
        self.c_v, self.c_proj = _nn.Linear(dim, kv, bias=False), _nn.Linear(dim, dim, bias=False)
        self.ve_gate = (_nn.Linear(32, c.num_key_value_heads, bias=False)
                        if layer % 2 == (c.physical_layers - 1) % 2 else None)

    def forward(self, x, positions, value_embedding):
        c = self.c
        B, T, _ = x.shape
        q = self.c_q(x).view(B, T, c.num_attention_heads, c.head_dim)
        k = self.c_k(x).view(B, T, c.num_key_value_heads, c.head_dim)
        v = self.c_v(x).view_as(k)
        if self.ve_gate is not None:
            v = v + (2 * torch.sigmoid(self.ve_gate(x[..., :32]))).unsqueeze(-1) * value_embedding.view_as(v)
        freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=x.dtype) / c.head_dim))
        ang = positions.to(x.dtype).unsqueeze(-1) / c.rope_condense_ratio * freq
        cos, sin = ang.cos().unsqueeze(2), ang.sin().unsqueeze(2)

        def rotate(t):
            p = t.unflatten(-1, (-1, 2))
            return torch.stack((p[..., 0] * cos - p[..., 1] * sin, p[..., 1] * cos + p[..., 0] * sin), -1).flatten(-2)
        q, k = rotate(q), rotate(k)
        if c.qk_norm:
            q, k = _F.rms_norm(q, (c.head_dim,)), _F.rms_norm(k, (c.head_dim,))
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = _F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=c.num_attention_heads != c.num_key_value_heads)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, c.hidden_size))


class _ArMLP(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = _nn.Linear(c.hidden_size, 2 * c.intermediate_size, bias=False)
        self.proj = _nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        g, v = self.fc(x).chunk(2, dim=-1)
        return self.proj(_F.silu(g) * v)


class _ArBlock(_nn.Module):
    def __init__(self, c, layer, ar):
        super().__init__()
        n1, n2 = ("attn_norm", "mlp_norm") if ar else ("norm_1", "norm_2")
        setattr(self, n1, _ArNorm(c.hidden_size, c.rms_norm_eps))
        self.attn = _ArAttn(c, layer)
        setattr(self, n2, _ArNorm(c.hidden_size, c.rms_norm_eps))
        self.mlp = _ArMLP(c)


class _ArLoop(_nn.Module):
    def __init__(self, c, ar):
        super().__init__()
        self.config, self.ar = c, ar
        self.transformer = _nn.ModuleDict({
            "wte": _nn.Embedding(c.vocab_size, c.hidden_size),
            "h": _nn.ModuleList(_ArBlock(c, i, ar) for i in range(c.physical_layers)),
            "ln_f": _ArNorm(c.hidden_size, c.rms_norm_eps)})
        self.shared_value_embedding = _nn.Embedding(c.vocab_size, c.num_key_value_heads * c.head_dim)
        if ar:
            self.attnres_queries = _nn.Parameter(torch.zeros(2 * c.num_hidden_layers, c.hidden_size))
            self.attnres_norms = _nn.ModuleList(_ArNorm(c.hidden_size, c.rms_norm_eps)
                                                for _ in range(2 * c.num_hidden_layers))
            self.final_attnres_query = _nn.Parameter(torch.zeros(c.hidden_size))
            self.final_attnres_norm = _ArNorm(c.hidden_size, c.rms_norm_eps)

    def _route(self, query, norm, blocks, partial, output_norm=None):
        values = blocks + [partial]
        if len(values) > self.config.sw_blocks + 1:
            values = [values[0], *values[-self.config.sw_blocks:]]
        stacked = torch.stack(values, dim=-2)
        scores = (stacked * (query * norm.weight)).sum(-1)
        scores = scores * torch.rsqrt(stacked.square().mean(-1) + norm.eps)
        w = scores.softmax(dim=-1)
        y = torch.zeros_like(partial)
        for i, value in enumerate(values):
            y = y + w[..., i, None] * value
        if output_norm is not None:
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + norm.eps) * output_norm.weight
        return y

    def forward(self, ids):
        c, tr = self.config, self.transformer
        pos = torch.arange(ids.shape[1])[None, :]
        ve = self.shared_value_embedding(ids)
        if not self.ar:
            h = tr.wte(ids) * c.embedding_scale
            for loop in range(c.loop_count):
                for block in tr.h:
                    h = h + block.attn(block.norm_1(h), pos, ve)
                    h = h + block.mlp(block.norm_2(h))
            return _F.linear(tr.ln_f(h), tr.wte.weight)
        emb = tr.wte(ids) * c.embedding_scale
        blocks, partial = [emb], emb
        for loop in range(c.loop_count):
            for i, block in enumerate(tr.h):
                e = loop * c.physical_layers + i
                h = self._route(self.attnres_queries[2 * e], self.attnres_norms[2 * e], blocks, partial, block.attn_norm)
                if e > 0 and e % (c.attnres_block_size // 2) == 0:
                    blocks.append(partial)
                    partial = None
                a = block.attn(h, pos, ve)
                partial = a if partial is None else partial + a
                h = self._route(self.attnres_queries[2 * e + 1], self.attnres_norms[2 * e + 1], blocks, partial,
                                block.mlp_norm)
                partial = partial + block.mlp(h)
        h = self._route(self.final_attnres_query, self.final_attnres_norm, blocks, partial)
        return _F.linear(tr.ln_f(h), tr.wte.weight)


@pytest.mark.parametrize("ar", [True, False], ids=["arloop", "prenormloop"])
@torch.no_grad()
def test_same_host_carries_arloop_shared_value_embeddings(ar):
    """ARLoop / PreNormLoop vs from_arloop against the REFERENCE above: 3 physical layers x 2 loops unrolled, the shared
    value embedding on the VE layers (gate 2 sigmoid(W x[:32])), adjacent-pair RoPE with a condense ratio, weight-free
    q / k norms after the rotary, GQA, SwiGLU, a tied scaled embedding; ARLoop's Block AttnRes (2-layer blocks, the
    embedding + last 2 sources window -- binding here, 4 sources by the end -- the first partial starting at the
    embedding). fp64. Switching off the value embeddings, the source window or the embedding-start moves the logits."""
    from types import SimpleNamespace as NS
    from elephantine.universal import from_arloop
    c = NS(vocab_size=97, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, physical_layers=3,
           loop_count=2, num_hidden_layers=6, intermediate_size=48, rope_theta=50000.0, rope_condense_ratio=2.0,
           qk_norm=True, rms_norm_eps=1e-5, embedding_scale=1.5, logit_softcap=None, attnres_block_size=4,
           sw_blocks=2)
    t = _rand_(_ArLoop(c, ar).to(F64).eval(), s=0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    ref = t(x)
    u = from_arloop(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    offs = [("ve", lambda h: [l.attn.ve.c.zero_() for l in h.layers if l.attn.ve.c != 0])]
    if ar:
        offs += [("window", lambda h: [l.ar_sw.zero_() for l in h.layers]),
                 ("emb_partial", lambda h: setattr(h.cfg, "attnres_emb_partial", False))]
    moved = {}
    for name, off in offs:
        h = from_arloop(t)
        off(h)
        moved[name] = _rel(ref, h(x))
    assert min(moved.values()) > 1e-4, moved


# --- K2-Horizon / Jais-3: the grouped RMSNorm (norm_groups) ---------------------------------------------------------
# REFERENCE, not the remote code: IFM/K2-Horizon-7B modeling_k2_horizon.py (snapshot 1ad5223c, md5
# 0e542c838e22328e545c1262d37ac6f3), read as text. K2HorizonRMSNorm (l. 613-636) is transcribed below; the rest of the
# dense model is the stock Llama: K2HorizonAttention (l. 197-326) with attention_gate_func None, query_key_norm False
# and rope_head_dim = head_dim (every grouped snapshot config but Jais-3-MoVA) is LlamaAttention (q / k / v / o, the
# rotate_half rotary l. 52-83, eager softmax l. 110-133), K2HorizonMLP (l. 515-528) the SwiGLU, K2HorizonDecoderLayer
# (l. 642-730) / K2HorizonModel (l. 823-909) the pre-norm Llama layer / model with input_layernorm,
# post_attention_layernorm and the final norm all K2HorizonRMSNorm(hidden, layernorm_num_groups). So the reference is
# the stock LlamaForCausalLM with those three norms replaced by the transcription.

class _K2RMSNorm(_nn.Module):
    def __init__(self, hidden_size, n_groups, eps=1e-6):
        super().__init__()
        self.n_groups, self.hidden_size = n_groups, hidden_size
        assert hidden_size % n_groups == 0
        self.weight = _nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(*hidden_states.shape[:-1], self.n_groups, -1)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:-2], -1)
        return self.weight * hidden_states


@pytest.mark.parametrize("groups", [2, 4])
@torch.no_grad()
def test_same_host_carries_k2_horizon_grouped_rmsnorm(groups):
    """K2-Horizon dense vs from_k2_horizon: the REFERENCE above (a stock Llama body with K2HorizonRMSNorm at every
    norm, layernorm_num_groups 2 / 4). fp64. The host's UNorm groups set back to 1 (a full-width RMSNorm) moves the
    logits."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from elephantine.universal import from_k2_horizon
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=1e7, attn_implementation="eager")
    t = LlamaForCausalLM(cfg).to(F64).eval()
    for layer in t.model.layers:
        layer.input_layernorm = _K2RMSNorm(32, groups, 1e-6).to(F64)
        layer.post_attention_layernorm = _K2RMSNorm(32, groups, 1e-6).to(F64)
    t.model.norm = _K2RMSNorm(32, groups, 1e-6).to(F64)
    t.config.layernorm_num_groups = groups
    _rand_(t, s=0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x).logits
    u = from_k2_horizon(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    for n_ in [u.norm] + [n for ul in u.layers for n in (ul.norm_a, ul.norm_f)]:
        n_.groups.fill_(1.0)
    assert _rel(ref, u(x)) > 1e-4


# --- LongCat-Flash: the shortcut-connected MoE (scmoe) and zero-computation experts --------------------------------

@torch.no_grad()
def test_same_host_carries_longcat_flash_scmoe():
    """LongCat-Flash (the stock transformers LongcatFlashForCausalLM) vs from_longcat_flash: each decoder layer as two
    host layers, the routed MoE reading the first sublayer's FFN input and joining after the second (ULayer.moe_defer),
    identity (zero-computation) experts (UFFN.exp_ident), softmax scores selected with a correction bias, no renorm,
    x routed_scaling_factor; MLA with the LoRA scales folded, interleaved rotary. fp64, the router's fp32 casts lifted.
    Undeferring the routed output (added at once) or making the identity experts ordinary (zero) experts moves the
    logits."""
    from transformers import LongcatFlashConfig, LongcatFlashForCausalLM
    from elephantine.universal import from_longcat_flash
    cfg = LongcatFlashConfig(vocab_size=97, hidden_size=32, num_layers=2, num_attention_heads=4, ffn_hidden_size=48,
                             q_lora_rank=16, kv_lora_rank=12, qk_nope_head_dim=8, qk_rope_head_dim=4, head_dim=4,
                             v_head_dim=8, moe_topk=3, n_routed_experts=4, zero_expert_num=3,
                             expert_ffn_hidden_size=16, routed_scaling_factor=2.0, rope_theta=10000.0,
                             attn_implementation="eager")
    t = _rand_(LongcatFlashForCausalLM(cfg).to(F64).eval(), s=0.3)
    for layer in t.model.layers:
        layer.mlp.router.e_score_correction_bias.normal_(0, 0.1)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x).logits
    u = from_longcat_flash(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    h = from_longcat_flash(t)
    for ul in h.layers:
        ul.moe_defer.zero_()                                 # the routed output joins at once
    assert _rel(ref, h(x)) > 1e-4
    h = from_longcat_flash(t)
    for ul in h.layers:
        ul.ffn.exp_ident.zero_()
    assert _rel(ref, h(x)) > 1e-4


# --- Ouro: the loop-boundary norm (loop_bnorm) and the per-token early exit (early_exit) ---------------------------
# REFERENCE, not the remote code: ByteDance/Ouro-1.4B-Thinking modeling_ouro.py (snapshot 3aaa2224, md5
# 7129d9d071a1e064a8a1b07da411b662), read as text. OuroAttention (l. 280-367, eager l. 250-277), OuroMLP (l. 65-79),
# OuroRMSNorm (l. 370-387) and OuroRotaryEmbedding (l. 467-538) are line for line Llama's, so the stock Llama modules
# stand for them; transcribed below: OuroDecoderLayer.forward (l. 410-446: the sandwich norms input_layernorm_2 /
# post_attention_layernorm_2 on the sub-block outputs), OuroModel.forward's loop (l. 632-666: total_ut_steps passes,
# h <- norm(h) after each, early_exit_gate(h) collected) and OuroForCausalLM.forward without labels (l. 764-781 the
# exit pdf, l. 822-850 exit_at_step / exit_threshold, l. 852-854 the last step otherwise).

class _OuroLayer(_nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
        self.self_attn, self.mlp = LlamaAttention(cfg, i), LlamaMLP(cfg)
        for n_ in ("input_layernorm", "input_layernorm_2", "post_attention_layernorm", "post_attention_layernorm_2"):
            setattr(self, n_, LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps))

    def forward(self, h, pe, mask):
        r = h
        h, _ = self.self_attn(self.input_layernorm(h), position_embeddings=pe, attention_mask=mask)
        h = r + self.input_layernorm_2(h)
        r = h
        return r + self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(h)))


class _OuroModel(_nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
        self.embed_tokens = _nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = _nn.ModuleList(_OuroLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(cfg)
        self.early_exit_gate = _nn.Linear(cfg.hidden_size, 1)
        self.total_ut_steps = cfg.total_ut_steps

    def forward(self, ids):
        h = self.embed_tokens(ids)
        T = ids.shape[1]
        pe = self.rotary_emb(h, torch.arange(T)[None])
        mask = torch.full((T, T), float("-inf"), dtype=h.dtype).triu(1)[None, None]
        hs, gates = [], []
        for _ in range(self.total_ut_steps):
            for layer in self.layers:
                h = layer(h, pe, mask)
            h = self.norm(h)
            hs.append(h)
            gates.append(self.early_exit_gate(h))
        return h, hs, gates


class _OuroLM(_nn.Module):
    def __init__(self, cfg, step=None, threshold=None):
        super().__init__()
        self.config, self.model = cfg, _OuroModel(cfg)
        self.lm_head = _nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.early_exit_step, self.early_exit_threshold = step, threshold

    def forward(self, ids):
        last, hs, gates = self.model(ids)
        pdf, rem = [], torch.ones_like(gates[0].squeeze(-1))
        for i, g in enumerate(gates):
            lam = torch.sigmoid(g.squeeze(-1))
            if i < len(gates) - 1:
                pdf.append(lam * rem)
                rem = rem * (1.0 - lam)
            else:
                pdf.append(rem)
        pdf = torch.stack(pdf, dim=2)
        if self.early_exit_step is not None and 0 <= self.early_exit_step < len(hs):
            return self.lm_head(hs[self.early_exit_step]), None
        if self.early_exit_threshold is not None:
            mask = torch.cumsum(pdf, dim=2) >= self.early_exit_threshold
            steps = torch.argmax(mask.float().to(pdf.dtype), dim=2)
            steps[~mask.any(dim=2)] = pdf.shape[2] - 1
            st = torch.stack(hs, dim=2)
            hsel = torch.gather(st, 2, steps[..., None, None].expand(-1, -1, 1, st.size(-1))).squeeze(2)
            return self.lm_head(hsel), steps
        return self.lm_head(last), None


@pytest.mark.parametrize("step,threshold", [(None, 0.6), (None, 1.0), (1, None), (None, None)],
                         ids=["threshold-0.6", "threshold-1", "exit_step-1", "last"])
@torch.no_grad()
def test_same_host_carries_ouro_loop_norm_early_exit(step, threshold):
    """Ouro vs from_ouro against the REFERENCE above: 2 shared layers x 3 UT steps unrolled, sandwich norms, the
    loop-boundary norm after every pass, the early-exit gate pdf and the per-token threshold exit (0.6: tokens exit at
    different steps; 1.0, the snapshot's: the last step bar rounding), a fixed exit step, or the last step. fp64 with
    the fp32 casts lifted. Switching the boundary norms off moves the logits, and so does always exiting last when the
    threshold binds."""
    from transformers import LlamaConfig
    from elephantine.universal import from_ouro
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=1e6, attn_implementation="eager")
    cfg.total_ut_steps = 3
    t = _rand_(_OuroLM(cfg, step, threshold).to(F64).eval(), s=0.3)
    with torch.no_grad():
        t.model.early_exit_gate.weight.normal_(0, 1.0)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref, steps = t(x)
    if threshold == 0.6:
        assert len(set(steps.flatten().tolist())) > 1           # the exit binds: tokens leave at different steps
    u = from_ouro(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    h = from_ouro(t)
    for ul in h.layers:
        ul.sn_c.zero_()
    assert _rel(ref, h(x)) > 1e-4
    if threshold == 0.6:
        h = from_ouro(t)
        h.early_exit.threshold.fill_(float("inf"))
        assert _rel(ref, h(x)) > 1e-4


# --- HRM-Text: two-state hierarchical loops (hrm_two_state) and the per-stack final norm (loop_bnorm) --------------

def test_same_host_carries_hrm_text_two_state_loops():
    """HRM-Text (transformers 5.18 models/hrm_text, the stock HrmTextForCausalLM, in a subprocess) vs from_hrm_text:
    2 layers per stack, H_cycles 2 x (L_cycles 2 + 1) stack invocations unrolled onto 12 host layers; the two states
    z_H / z_L in the register (UTwoState), each stack's weight-free final norm as its last layer's stream norm, the
    gated MHA, a scaled embedding. fp64 with the fp32 casts lifted. Switching the two-state branch off (a plain
    residual stream through the same layers) and the stack-end norms off each move the logits."""
    gap, scale = _tf5_run(LIFT + '''
from transformers.models.hrm_text import modeling_hrm_text as q, configuration_hrm_text as qc
from elephantine.universal import from_hrm_text
cfg = qc.HrmTextConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
    head_dim=8, H_cycles=2, L_cycles=2, embedding_scale=3.0, rms_norm_eps=1e-6, rope_theta=10000.0,
    attn_implementation="eager")
t = rand_(q.HrmTextForCausalLM(cfg).to(F64).eval())
x = ids(97, T=12)
with torch.no_grad():
    u = from_hrm_text(t)
    ours = u(x)
    with lift():
        ref = t(x, use_cache=False).logits
    assert ref.dtype == F64
    h = from_hrm_text(t)
    for l in h.layers:
        l.zstate.on.zero_(); l.zstate.put.zero_()
    m1 = rel(ref, h(x))
    h = from_hrm_text(t)
    for l in h.layers:
        l.sn_c.zero_()
    m2 = rel(ref, h(x))
    assert min(m1, m2) > 1e-3, (m1, m2)
report(ref, ours)
''')
    assert gap < 1e-12 * max(scale, 1), (gap, scale)


# --- Nanbeige4.2: the looped Llama body with a norm after every pass (loop_bnorm_cfg) ------------------------------
# REFERENCE, not the remote code: Nanbeige/Nanbeige4.2-3B modeling_nanbeige.py (snapshot b82e54bd, md5
# b98ae004ddad4dc001384915bcc16f46; configuration_nanbeige.py md5 bdaa233e467be058c6f8fc8765999593), read as text.
# With the snapshot's flags (no n-gram embeddings, hyper-connections, double-loop split, loop_share_kv, depth attention
# or qk_layernorm) NanbeigeDecoderLayer.forward (l. 1880-1931) is LlamaDecoderLayer's, NanbeigeAttention (l. 1117-1327,
# eager) LlamaAttention's with its own default rotary (l. 1044-1071), NanbeigeMLP (l. 1102-1114) / NanbeigeRMSNorm
# (l. 480-496) Llama's; transcribed: NanbeigeModel.forward's loop (l. 2311-2407): num_loops passes over the same
# layers, `if not skip_loop_final_norm: hidden_states = self.norm(hidden_states)` after each, else once at the end.

class _NanbeigeLM(_nn.Module):
    def __init__(self, cfg, loops, skip):
        super().__init__()
        from transformers import LlamaForCausalLM
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        base = LlamaForCausalLM(cfg)
        self.model, self.lm_head = base.model, base.lm_head
        for layer in self.model.layers:
            layer.self_attn.rotary_emb = LlamaRotaryEmbedding(cfg)
        self.config = cfg
        cfg.num_loops, cfg.skip_loop_final_norm = loops, skip

    def forward(self, ids):
        m, cfg = self.model, self.config
        h = m.embed_tokens(ids)
        T = ids.shape[1]
        mask = torch.full((T, T), float("-inf"), dtype=h.dtype).triu(1)[None, None]
        pe = m.layers[0].self_attn.rotary_emb(h, torch.arange(T)[None])
        for _ in range(cfg.num_loops):
            for layer in m.layers:
                h = layer(h, attention_mask=mask, position_embeddings=pe)
                h = h[0] if isinstance(h, tuple) else h
            if not cfg.skip_loop_final_norm:
                h = m.norm(h)
        if cfg.skip_loop_final_norm:
            h = m.norm(h)
        return self.lm_head(h)


@pytest.mark.parametrize("skip", [False, True])
@torch.no_grad()
def test_same_host_carries_nanbeige_loop_norm(skip):
    """Nanbeige4.2 vs from_nanbeige against the REFERENCE above: 3 layers x 2 loops unrolled, the final norm after each
    pass (skip_loop_final_norm false, the snapshot's) or once. fp64, the fp32 casts lifted. With the pass norms the
    host's stream norms switched off move the logits."""
    from transformers import LlamaConfig
    from elephantine.universal import from_nanbeige
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-5, rope_theta=7e7, attn_implementation="eager")
    t = _rand_(_NanbeigeLM(cfg, 2, skip).to(F64).eval(), s=0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x)
    u = from_nanbeige(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    if not skip:
        for ul in u.layers:
            ul.sn_c.zero_()
        assert _rel(ref, u(x)) > 1e-4


# --- nanowhale: DeepSeek-V4 hyper-connections read comb X (hc_dsv4_nano) -------------------------------------------
# REFERENCE, not the remote code: HuggingFaceTB/nanowhale-100m modeling_deepseek_v4.py (snapshot 6a3bed5f, md5
# 069132b2a6d22455d89935d97c13dc87), read as text: DeepseekV4RMSNorm l. 39-50, precompute_freqs_cis l. 53-60,
# apply_rotary_emb l. 63-79, hc_split_sinkhorn l. 85-123, DeepseekV4Attention l. 130-252 (SDPA path: no sink, no
# window), DeepseekV4Expert / Gate / MoE l. 255-359, DeepseekV4Block l. 362-461 (hc_post: comb X, l. 417-428),
# DeepseekV4Model l. 497-620 (the HC head l. 532-546), DeepseekV4ForCausalLM.forward l. 644-668.

class _NwNorm(_nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, _nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def _nw_freqs(dim, seqlen, base):
    f = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=F64) / dim))
    f = torch.outer(torch.arange(seqlen, dtype=F64), f)
    return torch.stack([f.cos(), f.sin()], dim=0)


def _nw_rot(x, cs):
    cos, sin = cs[0], cs[1]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    while cos.ndim < x1.ndim:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


def _nw_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    pre_raw, post_raw = mixes[..., :hc_mult], mixes[..., hc_mult:2 * hc_mult]
    comb_raw = mixes[..., 2 * hc_mult:].reshape(*mixes.shape[:-1], hc_mult, hc_mult)
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb = comb_raw * hc_scale[2] + hc_base[2 * hc_mult:].reshape(hc_mult, hc_mult)
    comb = _F.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class _NwAttn(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config, self.num_heads, self.head_dim = c, c.num_attention_heads, c.head_dim
        self.qk_rope_head_dim, self.q_lora_rank = c.qk_rope_head_dim, c.q_lora_rank
        self.o_groups, self.o_lora_rank, self.scaling = c.o_groups, c.o_lora_rank, c.head_dim ** -0.5
        self.wq_a = _nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.q_norm = _NwNorm(c.q_lora_rank, c.rms_norm_eps)
        self.wq_b = _nn.Linear(c.q_lora_rank, c.num_attention_heads * c.head_dim, bias=False)
        self.wkv = _nn.Linear(c.hidden_size, c.head_dim, bias=False)
        self.kv_norm = _NwNorm(c.head_dim, c.rms_norm_eps)
        self.wo_a = _nn.Linear(c.num_attention_heads * c.head_dim // c.o_groups, c.o_groups * c.o_lora_rank, bias=False)
        self.wo_b = _nn.Linear(c.o_groups * c.o_lora_rank, c.hidden_size, bias=False)
        self.attn_sink = _nn.Parameter(torch.zeros(c.num_attention_heads))

    def forward(self, x, mask, fc):
        B, S, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
        kv = self.kv_norm(self.wkv(x)).unsqueeze(1)
        r = self.qk_rope_head_dim
        q = torch.cat([q[..., :-r], _nw_rot(q[..., -r:], fc)], dim=-1)
        kv = torch.cat([kv[..., :-r], _nw_rot(kv[..., -r:], fc)], dim=-1)
        kve = kv.expand(-1, self.num_heads, -1, -1)
        o = _F.scaled_dot_product_attention(q, kve, kve, attn_mask=mask, is_causal=False, scale=self.scaling)
        cos, sin = fc[0][None, None], -fc[1][None, None]
        out = o[..., -r:]
        d = out.shape[-1] // 2
        o1, o2 = out[..., :d], out[..., d:]
        o = torch.cat([o[..., :-r], torch.cat([o1 * cos + o2 * sin, o1 * (-sin) + o2 * cos], dim=-1)], dim=-1)
        o = o.transpose(1, 2).reshape(B, S, self.o_groups, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)).flatten(2)
        return self.wo_b(o)


class _NwExpert(_nn.Module):
    def __init__(self, h, i, lim=0.0):
        super().__init__()
        self.w1, self.w2, self.w3 = _nn.Linear(h, i, bias=False), _nn.Linear(i, h, bias=False), _nn.Linear(h, i, bias=False)
        self.swiglu_limit = lim

    def forward(self, x):
        g, u = self.w1(x), self.w3(x)
        if self.swiglu_limit > 0:
            u, g = u.clamp(-self.swiglu_limit, self.swiglu_limit), g.clamp(max=self.swiglu_limit)
        return self.w2(_F.silu(g) * u)


class _NwGate(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.topk, self.scoring_func, self.route_scale = c.num_experts_per_tok, c.scoring_func, c.routed_scaling_factor
        self.weight = _nn.Parameter(torch.zeros(c.n_routed_experts, c.hidden_size))
        self.bias = _nn.Parameter(torch.zeros(c.n_routed_experts))

    def forward(self, x):
        s = _F.linear(x, self.weight)
        s = {"softmax": lambda t: t.softmax(-1), "sigmoid": torch.sigmoid,
             "sqrtsoftplus": lambda t: _F.softplus(t).sqrt()}[self.scoring_func](s)
        orig = s
        idx = (s + self.bias).topk(self.topk, dim=-1)[1]
        w = orig.gather(1, idx)
        if self.scoring_func != "softmax":
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return w * self.route_scale, idx


class _NwMoE(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.hidden_size, self.n_routed_experts = c.hidden_size, c.n_routed_experts
        self.gate = _NwGate(c)
        self.experts = _nn.ModuleList(_NwExpert(c.hidden_size, c.moe_intermediate_size, c.swiglu_limit)
                                      for _ in range(c.n_routed_experts))
        self.shared_expert = _NwExpert(c.hidden_size, c.moe_intermediate_size)

    def forward(self, x):
        shape, xf = x.shape, x.view(-1, self.hidden_size)
        w, idx = self.gate(xf)
        y = torch.zeros_like(xf)
        for i in range(self.n_routed_experts):
            r, top = torch.where(idx == i)
            if r.numel():
                y[r] += w[r, top].unsqueeze(-1) * self.experts[i](xf[r])
        return (y + self.shared_expert(xf)).view(shape)


class _NwBlock(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.hc_mult, self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters = c.hc_mult, c.rms_norm_eps, c.hc_eps, c.hc_sinkhorn_iters
        self.attn, self.ffn = _NwAttn(c), _NwMoE(c)
        self.attn_norm, self.ffn_norm = _NwNorm(c.hidden_size, c.rms_norm_eps), _NwNorm(c.hidden_size, c.rms_norm_eps)
        mix, hcd = (2 + c.hc_mult) * c.hc_mult, c.hc_mult * c.hidden_size
        self.hc_attn_fn, self.hc_ffn_fn = _nn.Parameter(torch.zeros(mix, hcd)), _nn.Parameter(torch.zeros(mix, hcd))
        self.hc_attn_base, self.hc_ffn_base = _nn.Parameter(torch.zeros(mix)), _nn.Parameter(torch.zeros(mix))
        self.hc_attn_scale, self.hc_ffn_scale = _nn.Parameter(torch.ones(3)), _nn.Parameter(torch.ones(3))

    def hc_pre(self, x, fn, sc, base):
        xf = x.flatten(2)
        mixes = _F.linear(xf, fn) * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        pre, post, comb = _nw_sinkhorn(mixes, sc, base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        return (pre.unsqueeze(-1) * x).sum(dim=2), post, comb

    def hc_post(self, x, residual, post, comb):
        return post.unsqueeze(-1) * x.unsqueeze(2) + torch.einsum("bsij,bsjd->bsid", comb, residual)

    def forward(self, x, mask, fc):
        r = x
        y, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_post(self.attn(self.attn_norm(y), mask, fc), r, post, comb)
        r = x
        y, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        return self.hc_post(self.ffn(self.ffn_norm(y)), r, post, comb)


class _NwModel(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embed_tokens = _nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = _nn.ModuleList(_NwBlock(c) for _ in range(c.num_hidden_layers))
        self.norm = _NwNorm(c.hidden_size, c.rms_norm_eps)
        hcd = c.hc_mult * c.hidden_size
        self.hc_head_fn = _nn.Parameter(torch.zeros(c.hc_mult, hcd))
        self.hc_head_base, self.hc_head_scale = _nn.Parameter(torch.zeros(c.hc_mult)), _nn.Parameter(torch.ones(1))

    def hc_head(self, x):
        xf = x.flatten(2)
        mixes = _F.linear(xf, self.hc_head_fn) * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        return (pre.unsqueeze(-1) * x).sum(dim=2)

    def forward(self, ids):
        c = self.config
        e = self.embed_tokens(ids)
        S = ids.shape[1]
        fc = _nw_freqs(c.qk_rope_head_dim, S, c.rope_theta)
        mask = torch.triu(torch.full((S, S), float("-inf"), dtype=e.dtype), diagonal=1)[None, None]
        h = e.unsqueeze(2).expand(-1, -1, c.hc_mult, -1).contiguous()
        for layer in self.layers:
            h = layer(h, mask, fc)
        return self.norm(self.hc_head(h))


class _NwLM(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config, self.model = c, _NwModel(c)
        self.lm_head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, ids):
        return self.lm_head(self.model(ids))


@torch.no_grad()
def test_same_host_carries_nanowhale_comb_hyper_connections():
    """nanowhale vs from_nanowhale against the REFERENCE above (hc_mult 3, 2 Sinkhorn iterations, the comb X stream
    update, the HC head, K = V MQA with trailing -theta RoPE and output de-rotation, the grouped low-rank output, a
    sqrt-softplus MoE + shared expert). fp64. The transposed (DeepSeek-V4) comb moves the logits."""
    from types import SimpleNamespace as NS
    from elephantine.universal import from_nanowhale
    c = NS(vocab_size=97, hidden_size=32, num_hidden_layers=3, num_attention_heads=4, head_dim=12, qk_rope_head_dim=4,
           q_lora_rank=16, o_groups=2, o_lora_rank=8, rms_norm_eps=1e-6, rope_theta=10000.0, hc_mult=3, hc_eps=1e-6,
           hc_sinkhorn_iters=2, n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
           scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0)
    t = _rand_(_NwLM(c).to(F64).eval(), s=0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    ref = t(x)
    u = from_nanowhale(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    for ul in u.layers:                                     # read comb^T X instead: DeepSeek-V4's orientation
        for hc in (ul.hc_a, ul.hc_f):
            hc.forward = hc._uhc.forward
    assert _rel(ref, u(x)) > 1e-4


# --- Xing4.0: clamped, max-shifted Sinkhorn hyper-connections read comb X (hc_xing) --------------------------------
# REFERENCE, not the remote code: XingChen-AGI/Xing4.0-29B-Intent modeling_xing4_0.py (snapshot 9bfc59b7, md5
# 48b1e75371fd51ac32dcd2e89647879b; configuration_xing4_0.py md5 7636aea15a6e8e965a362319bdf90db2), read as text.
# Xing4_0Attention (l. 333-448), Xing4_0MLP (l. 123-136), Xing4_0TopkRouter / Xing4_0MoE (l. 139-211: sigmoid scores
# selected on scores + bias, renorm + 1e-20, x routed_scaling_factor, + shared experts; n_group 1) and Xing4_0RMSNorm
# (l. 31-48) are DeepSeek-V3's (the stock DeepseekV3Attention / MLP / MoE / RMSNorm / rotary stand for them);
# transcribed: Xing4_0UnweightedRMSNorm (l. 51-57), Xing4_0HyperConnection (l. 451-483), Xing4_0DecoderLayer (l.
# 486-535: comb @ hidden_states) and Xing4_0Model.forward's stream expansion / mean collapse / norm (l. 625-638).

class _XingHC(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps = c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps
        self.input_norm = _NcFreeNorm(c.rms_norm_eps)
        mix = (2 + c.hc_mult) * c.hc_mult
        self.hc_fn = _nn.Parameter(torch.zeros(mix, c.hc_mult * c.hidden_size))
        self.hc_base, self.hc_scale = _nn.Parameter(torch.zeros(mix)), _nn.Parameter(torch.ones(3))
        self.mhc_h_res_clamp_max, self.mhc_h_res_clamp_min = c.mhc_h_res_clamp_max, c.mhc_h_res_clamp_min

    def forward(self, hs):
        hc = self.hc_mult
        flat = self.input_norm(hs.flatten(start_dim=2))
        pre_w, post_w, comb_w = _F.linear(flat, self.hc_fn).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.hc_base.split([hc, hc, hc * hc])
        s0, s1, s2 = self.hc_scale.unbind(0)
        pre, post = torch.sigmoid(pre_w * s0 + pre_b), 2 * torch.sigmoid(post_w * s1 + post_b)
        lg = comb_w.view(*comb_w.shape[:-1], hc, hc) * s2 + comb_b.view(hc, hc)
        lg = torch.clamp(lg, min=self.mhc_h_res_clamp_min, max=self.mhc_h_res_clamp_max)
        comb = torch.exp(lg - lg.amax(dim=-1, keepdim=True))
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return post, comb, (pre.unsqueeze(-1) * hs).sum(dim=2)


class _NcFreeNorm(_nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)


class _XingLayer(_nn.Module):
    def __init__(self, c, i):
        super().__init__()
        from transformers.models.deepseek_v3 import modeling_deepseek_v3 as dv
        self.self_attn = dv.DeepseekV3Attention(c, i)
        self.mlp = dv.DeepseekV3MoE(c) if i >= c.first_k_dense_replace else dv.DeepseekV3MLP(c)
        self.input_layernorm = dv.DeepseekV3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = dv.DeepseekV3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.attn_hc, self.ffn_hc = _XingHC(c), _XingHC(c)

    def forward(self, h, mask, pe):
        post, comb, col = self.attn_hc(h)
        a, _ = self.self_attn(hidden_states=self.input_layernorm(col), position_embeddings=pe, attention_mask=mask)
        h = post.unsqueeze(-1) * a.unsqueeze(-2) + torch.matmul(comb, h)
        post, comb, col = self.ffn_hc(h)
        return post.unsqueeze(-1) * self.mlp(self.post_attention_layernorm(col)).unsqueeze(-2) + torch.matmul(comb, h)


class _XingLM(_nn.Module):
    def __init__(self, c):
        super().__init__()
        from transformers.models.deepseek_v3 import modeling_deepseek_v3 as dv
        self.config = c
        self.model = _nn.Module()
        self.model.embed_tokens = _nn.Embedding(c.vocab_size, c.hidden_size)
        self.model.layers = _nn.ModuleList(_XingLayer(c, i) for i in range(c.num_hidden_layers))
        self.model.norm = dv.DeepseekV3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.model.rotary_emb = dv.DeepseekV3RotaryEmbedding(c)
        self.lm_head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, ids):
        m, T = self.model, ids.shape[1]
        e = m.embed_tokens(ids)
        h = e.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pe = m.rotary_emb(e, torch.arange(T)[None])
        mask = torch.full((T, T), float("-inf"), dtype=e.dtype).triu(1)[None, None]
        for layer in m.layers:
            h = layer(h, mask, pe)
        return self.lm_head(m.norm(h.mean(dim=2)))


@torch.no_grad()
def test_same_host_carries_xing_hyper_connections():
    """Xing4.0 vs from_xing against the REFERENCE above: 3 layers (1 dense, 2 MoE), hc_mult 3, the clamped, max-shifted
    Sinkhorn (8 passes; a +-2 clamp that binds here) read comb X, the mean collapse; MLA with interleaved rotary.
    fp64, the fp32 casts lifted. DeepSeek-V4's comb^T orientation (or no clamp) moves the logits."""
    from transformers import DeepseekV3Config
    from elephantine.universal import from_xing
    c = DeepseekV3Config(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=4,
                         num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1, n_group=1, topk_group=1,
                         kv_lora_rank=12, q_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
                         routed_scaling_factor=2.0, norm_topk_prob=True, rope_interleave=True, rms_norm_eps=1e-6,
                         attn_implementation="eager")
    c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps, c.mhc_h_res_clamp_min, c.mhc_h_res_clamp_max = 3, 8, 1e-6, -2.0, 2.0
    t = _XingLM(c).to(F64).eval()
    torch.manual_seed(0)
    for p in t.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    for layer in t.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.normal_(0, 0.1)
        for hc in (layer.attn_hc, layer.ffn_hc):
            hc.hc_scale.fill_(5.0)                              # comb logits beyond the +-2 clamp
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x)
    u = from_xing(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    h = from_xing(t)
    for ul in h.layers:
        for hc in (ul.hc_a, ul.hc_f):
            hc.lim.copy_(torch.tensor([-1e9, 1e9]))
    assert _rel(ref, h(x)) > 1e-4


# --- HY-V4: pre / post-only hyper-connections (hc_hy4) ------------------------------------------------------------

def test_same_host_carries_hy_v4_nocomb_hyper_connections():
    """HY-V4 (transformers 5.18 models/hy_v4, the stock HYV4ForCausalLM, in a subprocess) vs from_hy_v4: 3 hyper-
    connection streams with pre / post maps only (post magnitude 1.5) and the HyperHead collapse; gated MLA with sinks,
    its DSA indexer with index_topk 64 >= T (dense); 1 dense + 2 sparse FFNs, the group-limited (2 groups, keep 1)
    sigmoid router, the routed experts clamped at swiglu_limit 0.3 (binding) and the shared expert not. fp64 with the
    fp32 casts lifted. Mixing the streams (a uniform comb in place of the identity carry) moves the logits; so does
    clamping the shared expert too."""
    gap, scale = _tf5_run(LIFT + '''
from transformers.models.hy_v4 import modeling_hy_v4 as q, configuration_hy_v4 as qc
from elephantine.universal import from_hy_v4
cfg = qc.HYV4Config(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
    num_hidden_layers=3, num_attention_heads=4, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
    n_group=2, topk_group=1, q_lora_rank=16, kv_lora_rank=12, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
    index_topk=64, index_head_dim=8, index_n_heads=2, hc_mult=3, hc_magnitude=1.5, learnable_sink_init=0.3,
    swiglu_limit=0.3, rms_norm_eps=1e-6, routed_scaling_factor=2.0, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    attn_implementation="eager", experts_implementation="eager")
t = q.HYV4ForCausalLM(cfg).to(F64).eval()
torch.manual_seed(0)
with torch.no_grad():
    for n_, p in t.named_parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    for l in t.model.layers:
        if hasattr(l.mlp, "gate"):
            l.mlp.gate.e_score_correction_bias.normal_(0, 0.1)
x = ids(97, T=12)
with torch.no_grad():
    u = from_hy_v4(t)
    ours = u(x)
    with lift():
        ref = t(x, use_cache=False).logits
    assert ref.dtype == F64
    h = from_hy_v4(t)
    for l in h.layers:
        l.ffn.clip_shared.fill_(1.0)
    m1 = rel(ref, h(x))
    h = from_hy_v4(t)
    mixed = lambda f0: lambda X, reg=None: (lambda r: (r[0], torch.full_like(r[1], 1.0 / r[1].shape[-1]), r[2]))(f0(X, reg))
    for l in h.layers:                                         # the streams mixed (uniformly) instead of carried
        for hc in (l.hc_a, l.hc_f):
            hc.forward = mixed(hc.forward)
    m2 = rel(ref, h(x))
    assert min(m1, m2) > 1e-4, (m1, m2)
report(ref, ours)
''')
    assert gap < 1e-12 * max(scale, 1), (gap, scale)


# --- QwenDLM-ORL: the self-conditioning input path (selfcond, selfcond_norm) -------------------------------------
# REFERENCE, not the remote code: yifanyu/QwenDLM-4B-ORL modeling_sdar.py (snapshot 80cb0fe1, md5
# ad3471c61d943bf849a6c7d535751a9d), read as text: SDARSelfConditioning (l. 601-641, without the training-time
# statistics), _build_self_conditioning_signal (l. 1488-1510: softmax(lm_head(h)) @ E on the selected positions) and the
# signal replay forward (l. 1657-1668: inputs_embeds = self_conditioning(embed(ids), signal)). The SDAR body is Qwen3's
# under the caller's block mask (as in tests/test_host_gaps.py's SDAR test: the stock Qwen3 given SDAR's block mask).

class _SdarSelfCond(_nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        self.config = cfg
        f = cfg.self_conditioning_intermediate_size
        self.pre_norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_norm_eps = cfg.rms_norm_eps
        self.post_norm_scale = float(cfg.self_conditioning_post_norm_scale)
        self.gate_proj = _nn.Linear(cfg.hidden_size, f, bias=False)
        self.up_proj = _nn.Linear(cfg.hidden_size, f, bias=False)
        self.down_proj = _nn.Linear(f, cfg.hidden_size, bias=False)

    def forward(self, inputs_embeds, signal):
        signal = self.pre_norm(signal) * float(self.config.self_conditioning_prenorm_scale)
        residual = self.down_proj(_F.silu(self.gate_proj(signal)) * self.up_proj(signal))
        combined = inputs_embeds + residual * float(self.config.self_conditioning_reparam_out_scale)
        return _F.rms_norm(combined, (combined.size(-1),), eps=self.post_norm_eps) * self.post_norm_scale


@torch.no_grad()
def test_same_host_carries_qwendlm_self_conditioning():
    """QwenDLM-ORL vs from_sdar_selfcond against the REFERENCE above (QwenDLM's scales: prenorm 0.125, out 0.125, post
    0.02; block size 4): the first pass and its soft-embedding signal (selfcond_signal vs
    _build_self_conditioning_signal), then the conditioned pass forward(ids, selfcond=s). fp64, the fp32 casts lifted.
    The signal changes the output, and without the post-norm the host would not match."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from elephantine.universal import from_sdar_selfcond, selfcond_signal
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=5e6, attn_implementation="sdpa")
    cfg.self_conditioning_intermediate_size, cfg.self_conditioning_prenorm_scale = 40, 0.125
    cfg.self_conditioning_reparam_out_scale, cfg.self_conditioning_post_norm_scale = 0.125, 0.02
    t = Qwen3ForCausalLM(cfg).to(F64).eval()
    t.self_conditioning = _SdarSelfCond(cfg).to(F64)
    _rand_(t, s=0.3)
    T, b = 12, 4
    x = torch.randint(0, 97, (2, T), generator=torch.Generator().manual_seed(0))
    blk = torch.arange(T) // b
    am = (blk[None, :] <= blk[:, None])[None, None].expand(2, 1, T, T)
    noisy = torch.zeros(2, T, dtype=torch.bool)
    noisy[:, 8:] = True                                     # the last block is the noisy one
    with _lift():
        h1 = t.model(x, attention_mask=am).last_hidden_state
        sig = (t.lm_head(h1).softmax(-1) @ t.model.embed_tokens.weight) * noisy[..., None].to(F64)
        ref = t(inputs_embeds=t.self_conditioning(t.model.embed_tokens(x), sig), attention_mask=am).logits
        plain = t(x, attention_mask=am).logits
    u = from_sdar_selfcond(t, block_size=b)
    assert _rel(plain, u(x)) < 1e-12                        # no signal: plain SDAR
    s = selfcond_signal(u, x, noisy)
    assert _rel(sig[noisy], s[noisy]) < 1e-12
    assert _rel(ref, u(x, selfcond=s)) < 1e-12, _rel(ref, u(x, selfcond=s))
    assert _rel(ref, plain) > 1e-3
    u.selfcond.post_c.zero_()
    assert _rel(ref, u(x, selfcond=s)) > 1e-3


# --- Dory: nGPT sphere residuals, the FFN hidden norm, loop input re-injection (ngpt, ffn_norm, loop_inject, loop_bnorm)
# REFERENCE, not the remote code: Nemotron-Labs Dory, transformers PR 50099 models/dory/modeling_dory.py
# (~/.cache/nocode/dory/transformers-pr50099, md5 ad49a4276dfa5c5ba20602fd35ac0bda), read as text: DoryL2Norm l. 49-62,
# the rotary (l. 177-236, per layer type), DoryAttention l. 316-416 (unit q / k, q x sqk^2 after the rotary, the
# s_gate^2 sigmoid gate, attn_sub_norm before o_proj), DoryMLP l. 419-436, DoryDecoderLayer l. 439-482, DoryLMHead
# l. 485-494, DoryModel.forward's sections l. 595-636 (input layers, L2(h) + L2(h_input) at every loop start, output
# layers, the final L2 norm). The KV-cache classes (prefill-irrelevant) are left out.

class _DoryL2(_nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + self.eps)


def _dory_rot(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


class _DoryAttn(_nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.c, self.head_dim = c, c.head_dim
        self.num_key_value_groups = c.num_attention_heads // c.num_key_value_heads
        self.scaling = c.head_dim ** -0.5
        H, kv, hd, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.hidden_size
        self.q_proj, self.k_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(d, kv * hd, bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, kv * hd, bias=False), _nn.Linear(H * hd, d, bias=False)
        self.sliding_window = c.sliding_window if c.layer_types[i].endswith("sliding_attention") else None
        self.q_norm, self.k_norm, self.attn_sub_norm = _DoryL2(), _DoryL2(), _DoryL2()
        self.gate_proj = _nn.Linear(d, H * hd, bias=False)
        self.sqk = _nn.Parameter(torch.full((H * hd,), c.initializer_range))
        self.s_gate = _nn.Parameter(torch.full((H * hd,), c.initializer_range))
        self.sqk_scale = hd ** 0.5 / c.initializer_range
        self.s_gate_scale = d ** 0.25 / c.initializer_range

    def forward(self, h, cos, sin):
        B, T, _ = h.shape
        shp = (B, T, -1, self.head_dim)
        q, k, v = self.q_proj(h).view(shp), self.k_proj(h).view(shp), self.v_proj(h).view(shp)
        g = self.gate_proj(h)
        q, k, v = self.q_norm(q).transpose(1, 2), self.k_norm(k).transpose(1, 2), v.transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = q * cos + _dory_rot(q) * sin, k * cos + _dory_rot(k) * sin
        q = q * (self.sqk * self.sqk_scale).view(-1, 1, self.head_dim).square()
        k = k.repeat_interleave(self.num_key_value_groups, 1)
        v = v.repeat_interleave(self.num_key_value_groups, 1)
        s = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        tt, ss = torch.arange(T)[:, None], torch.arange(T)[None]
        ok = ss <= tt
        if self.sliding_window is not None:
            ok = ok & (tt - ss < self.sliding_window)
        o = (s.masked_fill(~ok, float("-inf")).softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(g * (self.s_gate * self.s_gate_scale).square())
        return self.o_proj(self.attn_sub_norm(o))


class _DoryMLP(_nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f = c.hidden_size, c.intermediate_size
        self.gate_proj, self.up_proj = _nn.Linear(d, f, bias=False), _nn.Linear(d, f, bias=False)
        self.down_proj = _nn.Linear(f, d, bias=False)
        self.suv_gate, self.suv_up = _nn.Parameter(torch.ones(f)), _nn.Parameter(torch.ones(f))
        self.suv_scale, self.ffn_sub_norm = d ** 0.5, _DoryL2()

    def forward(self, h):
        g = self.gate_proj(h) * (self.suv_gate * self.suv_scale)
        u = self.up_proj(h) * (self.suv_up * self.suv_scale)
        return self.down_proj(self.ffn_sub_norm(_F.silu(g) * u))


class _DoryLayer(_nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.self_attn, self.mlp = _DoryAttn(c, i), _DoryMLP(c)
        self.input_layernorm, self.post_attention_layernorm = _DoryL2(), _DoryL2()
        self.pre_feedforward_layernorm, self.post_feedforward_layernorm = _DoryL2(), _DoryL2()
        self.layer_type = c.layer_types[i]
        self.lambda_1 = _nn.Parameter(torch.full((c.hidden_size,), c.initializer_range))
        self.lambda_2 = _nn.Parameter(torch.full((c.hidden_size,), c.initializer_range))
        self.lambda_scale = c.ngpt_alpha_init / c.initializer_range

    def forward(self, h, cos, sin):
        h = self.input_layernorm(h)
        a = self.post_attention_layernorm(self.self_attn(h, cos, sin))
        h = h + (self.lambda_1 * self.lambda_scale).abs() * (a - h)
        h = self.pre_feedforward_layernorm(h)
        f = self.post_feedforward_layernorm(self.mlp(h))
        return h + (self.lambda_2 * self.lambda_scale).abs() * (f - h)


class _DoryLM(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.model = _nn.Module()
        self.model.embed_tokens = _nn.Embedding(c.vocab_size, c.hidden_size)
        self.model.layers = _nn.ModuleList(_DoryLayer(c, i) for i in range(c.num_hidden_layers))
        self.model.norm = _DoryL2()
        self.model.rotary_emb = _nn.Module()
        for lt, th in c.rope_thetas.items():
            inv = 1.0 / (th ** (torch.arange(0, c.head_dim, 2, dtype=F64) / c.head_dim))
            self.model.rotary_emb.register_buffer(f"{lt}_inv_freq", inv)
        self.lm_head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.lm_head.logit_scale = _nn.Parameter(torch.full((c.vocab_size,), c.initializer_range))
        self.lm_head.logit_scale_factor = 1 / c.initializer_range

    def forward(self, ids):
        c, m = self.config, self.model
        T = ids.shape[1]
        h = m.embed_tokens(ids)

        def cs(lt):
            f = torch.arange(T, dtype=F64)[None, :, None] * getattr(m.rotary_emb, f"{lt}_inv_freq")
            e = torch.cat((f, f), -1)
            return e.cos(), e.sin()
        run = lambda layer, h: layer(h, *cs(layer.layer_type))
        a, b = c.n_input_layers, c.n_input_layers + c.n_recurrent_layers
        for layer in m.layers[:a]:
            h = run(layer, h)
        hin = m.norm(h)
        for _ in range(c.n_recurrent_loops):
            h = m.norm(h) + hin
            for layer in m.layers[a:b]:
                h = run(layer, h)
        for layer in m.layers[c.num_hidden_layers - c.n_output_layers:]:
            h = run(layer, h)
        h = m.norm(h)
        return self.lm_head(h) * (self.lm_head.logit_scale * self.lm_head.logit_scale_factor)


@torch.no_grad()
def test_same_host_carries_dory_ngpt_loops():
    """Dory vs from_dory against the REFERENCE above: 1 input + 2 recurrent (x 2 loops) + 1 output layers (each two
    host layers), unit-sphere streams and updates, unit q / k with sqk^2, the s_gate^2 gate and the gated-output norm,
    the suv-scaled MLP with its hidden norm, sliding (window 4, binding) / full layers with their own rope thetas, the
    loop re-injection, the logit scale. fp64. Switching off each new branch moves the logits: the nGPT update (plain
    residuals), the FFN hidden norm, the gated-output norm (both only through their eps: each feeds an L2 norm), the
    loop re-injection."""
    from types import SimpleNamespace as NS
    from elephantine.universal import from_dory
    c = NS(vocab_size=97, hidden_size=32, intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
           head_dim=8, num_hidden_layers=4, n_input_layers=1, n_recurrent_layers=2, n_output_layers=1,
           n_recurrent_loops=2, sliding_window=4, initializer_range=0.0177, ngpt_alpha_init=0.05, hidden_act="silu",
           layer_types=["sliding_attention", "recurrent_sliding_attention", "recurrent_full_attention",
                        "full_attention"],
           rope_thetas={"sliding_attention": 1e4, "full_attention": 1e7, "recurrent_sliding_attention": 1e4,
                        "recurrent_full_attention": 1e7})
    t = _DoryLM(c).to(F64).eval()
    torch.manual_seed(0)
    for n_, p in t.named_parameters():                      # around the nGPT scales (s ~ initializer_range)
        if n_.endswith(("sqk", "s_gate", "lambda_1", "lambda_2", "logit_scale")):
            p.copy_(c.initializer_range * (1 + 0.5 * torch.randn_like(p)))
        elif n_.endswith(("suv_gate", "suv_up")):
            p.copy_(1 + 0.3 * torch.randn_like(p))
        else:
            p.copy_(0.3 * torch.randn_like(p))
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    ref = t(x)
    u = from_dory(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    moved = {}
    for name, off in (("ngpt", lambda h: [(l.rgain.gx.fill_(1.0), l.rgain.go.fill_(1.0)) for l in h.layers]),
                      ("ffn_norm", lambda h: [setattr(l.ffn, "s_hid", None) for l in h.layers]),
                      ("o_late", lambda h: [setattr(l.attn, "o_late", None) for l in h.layers]),
                      ("loop_inject", lambda h: [l.loopinj.inject.zero_() for l in h.layers])):
        h = from_dory(t)
        off(h)
        moved[name] = _rel(ref, h(x))
    # (the two inner norms sit right before an L2-normalised output, which is scale-free but for its eps: switched off
    # they move the logits by ~1e-8 -- still 4 orders of magnitude above the fp64 agreement)
    assert moved["ngpt"] > 1e-2 and moved["loop_inject"] > 1e-3, moved
    assert min(moved["ffn_norm"], moved["o_late"]) > 1e-10, moved


# --- A.X-K2 / Grug-MoE: the low-rank gated pre-norm (gated_norm) -------------------------------------------------
# A.X-K2: the stock transformers 5.18 AXK2GatedRMSNorm (models/axk2/modeling_axk2.py: AXK2GateMLP, AXK2GatedRMSNorm),
# run in a subprocess. Grug-MoE: a REFERENCE (read as text, never run) of marin-community/marin
# skyrl_train_models_grug_moe.py (md5 ba45ddae0a6ed95cfa9c2d813305b276): GrugMoeRMSNorm l. 382-393 and GrugMoeGatedNorm
# l. 402-410 (rank GRUG_GATED_NORM_RANK = 128, l. 54), applied after the embedding norm, both pre-norms and the final
# norm (l. 843-859, l. 887-896). Block level: neither whole model is carried (A.X-K2's raw-latent MLA query, Grug's
# XSA are attention features).

def test_gated_norm_blocks_axk2_stock():
    """A.X-K2's gated RMSNorm (the stock class, rank 16) vs UNorm's gate branch (gated_norm_from) at fp64 with the fp32
    casts lifted; the gate switched off (gate_c 0) moves the output."""
    gap, scale = _tf5_run(LIFT + '''
from transformers.models.axk2 import modeling_axk2 as q, configuration_axk2 as qc
from elephantine.universal import UNorm, gated_norm_from
cfg = qc.AXK2Config(hidden_size=32, gated_norm_rank=16, rms_norm_eps=1e-5, vocab_size=97, pad_token_id=0,
                    bos_token_id=1, eos_token_id=2)
m = q.AXK2GatedRMSNorm(cfg).to(F64)
torch.manual_seed(0)
with torch.no_grad():
    for p in m.parameters():
        p.copy_(torch.randn_like(p) * 0.5)
x = torch.randn(2, 12, 32, dtype=F64)
with torch.no_grad():
    u = UNorm(32, 24)
    gated_norm_from(u, m)
    ours = u(x)
    with lift():
        ref = m(x)
    u.gate_c.zero_()
    assert rel(ref, u(x)) > 1e-2
report(ref, ours)
''')
    assert gap < 1e-12 * max(scale, 1), (gap, scale)


class _GrugRMSNorm(_nn.Module):
    def __init__(self, n, eps):
        super().__init__()
        self.weight, self.eps = _nn.Parameter(torch.ones(n)), eps

    def forward(self, h):
        return h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class _GrugGatedNorm(_nn.Module):
    def __init__(self, n, rank=128):
        super().__init__()
        self.down_proj, self.up_proj = _nn.Linear(n, rank, bias=False), _nn.Linear(rank, n, bias=False)

    def forward(self, h):
        return h * torch.sigmoid(self.up_proj(_F.silu(self.down_proj(h))))


@torch.no_grad()
def test_gated_norm_blocks_grug_reference():
    """Grug-MoE's RMSNorm + GrugMoeGatedNorm (rank 128) vs UNorm's gate branch, at a layer pre-norm and at the gated
    embedding-norm site (UniversalElephantine.emb_norm built gated when norm_gate_rank > 0); fp64. Off moves it."""
    from elephantine.universal import UConfig, UNorm, UniversalElephantine, gated_norm_from
    torch.manual_seed(0)
    n, g = _GrugRMSNorm(64, 1e-6).to(F64), _GrugGatedNorm(64).to(F64)
    for p in list(n.parameters()) + list(g.parameters()):
        p.copy_(torch.randn_like(p) * 0.3)
    x = torch.randn(2, 12, 64, dtype=F64)
    ref = g(n(x))
    u = UNorm(64, 128)
    gated_norm_from(u, n, g)
    assert _rel(ref, u(x)) < 1e-12
    h = UniversalElephantine(UConfig(vocab=97, d=64, L=1, norm_gate_rank=128))
    assert h.emb_norm.g_up.out_features == 128
    gated_norm_from(h.emb_norm, n, g)
    assert _rel(ref, h.emb_norm(x)) < 1e-12
    u.gate_c.zero_()
    assert _rel(ref, u(x)) > 1e-2


# --- Doge: per-channel residual gains (res_gain) -----------------------------------------------------------------

@torch.no_grad()
def test_same_host_carries_doge_residual_gains():
    """Doge (the stock transformers DogeForCausalLM) vs from_doge: per-head qk-norms, SwiGLU, tied embeddings and the
    per-channel residual gains (URGain gx). Its dynamic-mask attention is an attention feature: neutral here (A = 0: the
    key bias exp(0) = 1 is a per-row constant; keep_window_size >= T). fp64, the fp32 casts lifted. Resetting the gains
    to 1 (the plain residual) moves the logits."""
    from transformers import DogeConfig, DogeForCausalLM
    # (this branch's URGain loader; elephantine.universal.from_doge is eleph-gaps-attn's, with the dynamic mask)
    from elephantine.branches_stream import from_doge
    cfg = DogeConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8, keep_window_size=64, is_moe=False, rms_norm_eps=1e-6,
                     tie_word_embeddings=True, attn_implementation="eager")
    t = _rand_(DogeForCausalLM(cfg).to(F64).eval(), s=0.3)
    for layer in t.model.layers:
        layer.self_attn.A.zero_()
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x).logits
    u = from_doge(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    for ul in u.layers:
        ul.rgain.gx.fill_(1.0)
    assert _rel(ref, u(x)) > 1e-3


# --- ZAYA1: ResidualScaling merges (res_gain, res_bias) ------------------------------------------------------------
# REFERENCE, not the remote code: Zyphra/transformers modeling_zaya1_vl.py (~/.cache/nocode/zaya1_vl, md5
# 74187bfe955b1596a125a95f32ceb26c), read as text: ResidualScaling l. 1512-1531 and the fused-add flow of the decoder
# layers (l. 1430-1445: residual, hidden = res_scale(residual, hidden); residual = residual + hidden; norm -> sublayer)
# and the final merge before the final norm. The sub-layers here are the host's own attention / FFN blocks (ZAYA's
# CCA attention and its MoE are not carried): the test checks the merge algebra end to end through them.

class _ZayaResScale(_nn.Module):
    def __init__(self, d, layer_n):
        super().__init__()
        self.not_first_layer = layer_n != 0
        self.hidden_states_scale, self.hidden_states_bias = _nn.Parameter(torch.ones(d)), _nn.Parameter(torch.zeros(d))
        if self.not_first_layer:
            self.residual_scale, self.residual_bias = _nn.Parameter(torch.ones(d)), _nn.Parameter(torch.zeros(d))

    def forward(self, residual, hidden_states):
        hidden_states = (hidden_states + self.hidden_states_bias.expand(1, 1, -1)) * self.hidden_states_scale.expand(1, 1, -1)
        if self.not_first_layer and residual is not None:
            residual = (residual + self.residual_bias.expand(1, 1, -1)) * self.residual_scale.expand(1, 1, -1)
        return residual, hidden_states


@torch.no_grad()
def test_zaya_residual_scaling_merges():
    """ZAYA1's ResidualScaling merges (REFERENCE above) through 3 layers of host attention / FFN sub-blocks vs the same
    host with zaya_residual_scaling_into (URGain gx / go / gb, the first affine folded into the token table); fp64. The
    plain residual (gains 1, biases 0) moves the logits."""
    import copy
    from elephantine.universal import UConfig, UniversalElephantine, zaya_residual_scaling_into
    torch.manual_seed(0)
    c = UConfig(vocab=97, d=32, L=3, heads=4, head_dim=8, shared_ff=48)
    base = UniversalElephantine(c)
    for n_, p in base.named_parameters():
        if n_.split(".")[-2:-1] and any(k in n_ for k in ("q_proj", "k_proj", "v_proj", "o_proj", "s_gate", "s_up",
                                                           "s_down", "embed", "lm_head")):
            p.copy_(torch.randn_like(p) * 0.3)
    for ul in base.layers:
        ul.attn.inv_freq.copy_(1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=F64) / 8)))
    scalings = _nn.ModuleList(_ZayaResScale(32, k) for k in range(2 * c.L + 1)).to(F64)
    for p in scalings.parameters():
        p.add_(torch.randn_like(p) * 0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    h, residual, k = base.embed(x), None, 0
    for ul in base.layers:
        for norm, sub in ((ul.norm_a, lambda t: ul.attn(t, True, {})), (ul.norm_f, lambda t: ul.ffn(t, {}))):
            residual, h = scalings[k](residual, h)
            residual = h if residual is None else residual + h
            h, k = sub(norm(residual)), k + 1
    residual, h = scalings[k](residual, h)
    ref = base.lm_head(base.norm(residual + h))
    u = copy.deepcopy(base)
    zaya_residual_scaling_into(u, list(scalings))
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    assert _rel(ref, base(x)) > 1e-3


# --- CFRD (haru): data-dependent scalar residual gates (res_gate) -------------------------------------------------
# REFERENCE, not the remote code: alice-noa-chan/haru_1.1 model.py (snapshot b834a95b, md5
# 77bc183bcfe9fa08c266497cf988e94e), read as text: RMSNorm l. 137-147, build_rope_cache l. 150-161, apply_rope
# l. 164-182 (interleaved even / odd pairs), FullCausalAttention l. 269-312, SwiGLU l. 448-457, BindingBlock l. 460-491
# (x + c sigmoid(attn_gate(u)) attn(u), the same for the SwiGLU, c = 1 / sqrt 2). Block level: CFRD's FoldedCell
# reads a score-selected summary memory (mem_attn, an attention feature) and is not carried.

class _CfNorm(_nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, _nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def _cf_rope(x, cos, sin):
    out = torch.empty_like(x)
    ev, od = x[..., 0::2], x[..., 1::2]
    out[..., 0::2] = ev * cos - od * sin
    out[..., 1::2] = ev * sin + od * cos
    return out


class _CfAttn(_nn.Module):
    def __init__(self, d, H, kv):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = H, kv, d // H
        self.q_proj, self.k_proj = _nn.Linear(d, d, bias=False), _nn.Linear(d, kv * (d // H), bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, kv * (d // H), bias=False), _nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        c, s = cos[:T].view(1, 1, T, -1), sin[:T].view(1, 1, T, -1)
        q, k = _cf_rope(q, c, s), _cf_rope(k, c, s)
        y = _F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_head != self.n_kv_head)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class _CfSwiGLU(_nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1, self.w3, self.w2 = _nn.Linear(d, f, bias=False), _nn.Linear(d, f, bias=False), _nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(_F.silu(self.w1(x)) * self.w3(x))


class _CfBinding(_nn.Module):
    def __init__(self, d, H, kv, f):
        super().__init__()
        self.attn_norm, self.ffn_norm = _CfNorm(d), _CfNorm(d)
        self.attention, self.ffn = _CfAttn(d, H, kv), _CfSwiGLU(d, f)
        self.attn_gate, self.ffn_gate = _nn.Linear(d, 1, bias=True), _nn.Linear(d, 1, bias=True)
        self.residual_scale = 1.0 / math.sqrt(2.0)

    def forward(self, x, cos, sin):
        u = self.attn_norm(x)
        x = x + self.residual_scale * torch.sigmoid(self.attn_gate(u)) * self.attention(u, cos, sin)
        v = self.ffn_norm(x)
        return x + self.residual_scale * torch.sigmoid(self.ffn_gate(v)) * self.ffn(v)


@torch.no_grad()
def test_cfrd_binding_block_residual_gates():
    """CFRD's BindingBlock (REFERENCE above) vs one host layer (cfrd_binding_into: URGain's data-dependent gate) at
    fp64; the gates switched off (g_on 0: the plain c-scaled residual) move the output."""
    from elephantine.universal import UConfig, UniversalElephantine, cfrd_binding_into
    d, H, kv, f, theta = 32, 4, 2, 48, 10000.0
    torch.manual_seed(0)
    blk = _CfBinding(d, H, kv, f).to(F64)
    for p in blk.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    hd = d // H
    fr = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    ang = torch.outer(torch.arange(12, dtype=F64), fr)
    x = torch.randn(2, 12, d, dtype=F64)
    ref = blk(x, ang.cos(), ang.sin())
    u = UniversalElephantine(UConfig(vocab=7, d=d, L=1, heads=H, head_dim=hd, shared_ff=f, res_gain=True))
    ul = u.layers[0]
    cfrd_binding_into(ul, blk, theta)
    assert _rel(ref, ul(x, True, {})) < 1e-12, _rel(ref, ul(x, True, {}))
    ul.rgain.g_on.zero_()
    assert _rel(ref, ul(x, True, {})) > 1e-3


# --- Muse-Spark: the coupled per-channel residual gate (res_gate_ms) ----------------------------------------------
# REFERENCE, not the remote code: transformers main models/muse_spark/modular_muse_spark.py
# (~/.cache/nocode/muse_spark/transformers-main, md5 5d28048c7fbcfd892acc63218c95e738), read as text:
# MuseSparkResidualGate l. 451-462 (beta = sigmoid(g / T), alpha = sqrt(max(sigmoid(-g / T) (1 + beta), 1e-3)),
# out = alpha r + beta h), used at both residual adds. Static per channel: URGain's gx = alpha, go = beta. Block level
# (its o_norm attention output norm and latent MoE with per-expert output norms are not carried).

@torch.no_grad()
def test_muse_spark_residual_gate_block():
    """MuseSparkResidualGate (REFERENCE above; gates spanning the 1e-3 clamp) vs URGain.mix loaded by
    muse_residual_gate_into (gx = alpha, go = beta) at fp64; the plain residual moves it."""
    from types import SimpleNamespace as NS
    from elephantine.branches_stream import URGain
    from elephantine.universal import muse_residual_gate_into
    torch.manual_seed(0)
    gate, T = torch.randn(32, dtype=F64) * 4, 0.7
    r, h = torch.randn(2, 12, 32, dtype=F64), torch.randn(2, 12, 32, dtype=F64)
    w = gate / T
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    assert (torch.sigmoid(-w) * (1.0 + beta) < 1e-3).any()     # the clamp binds on some channels
    ref = alpha * r + beta * h
    g = URGain(NS(d=32))
    muse_residual_gate_into(g, 0, gate, T)
    assert _rel(ref, g.mix(0, r, h)) < 1e-14
    g.gx.fill_(1.0); g.go.fill_(1.0)
    assert _rel(ref, g.mix(0, r, h)) > 1e-2


# --- Dhara-AR: Canon layers at A / B / C / D (canon) -------------------------------------------------------------
# REFERENCE, not the remote code: codelion/dhara-250m modeling_dhara_ar.py (snapshot 5f6da498, md5
# 8cf37a7c2ba3f57d473ae98123553fad), read as text: CanonLayer l. 46-115 (causal left pad, depthwise conv, optional SiLU,
# residual), RMSNorm l. 118-129, the default rotary l. 132-245 (no YaRN), DharaARAttention l. 258-402 (Canon-B on q / k
# / v, the rotary, then q / k RMSNorm; the eager path), DharaARMLP l. 405-437 (Canon-D on the SwiGLU hidden),
# DharaARDecoderLayer l. 440-515 (Canon-A / -C after the pre-norms), DharaARModel / ForCausalLM forward l. 550-665
# (the causal mask, the final norm, the logit softcap).

class _DhCanon(_nn.Module):
    def __init__(self, n, k, res=True, act=False, bias=False):
        super().__init__()
        self.kernel_size, self.use_residual, self.use_activation, self.hidden_size = k, res, act, n
        self.conv = _nn.Conv1d(n, n, k, padding=k - 1, groups=n, bias=bias)

    def forward(self, h):
        T = h.shape[1]
        x = _F.pad(h.transpose(1, 2), (self.kernel_size - 1, 0))
        out = _F.conv1d(x, self.conv.weight, self.conv.bias, padding=0, groups=self.hidden_size)[:, :, :T]
        if self.use_activation:
            out = _F.silu(out)
        out = out.transpose(1, 2)
        return h + out if self.use_residual else out


class _DhNorm(_nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight, self.eps = _nn.Parameter(torch.ones(n)), eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class _DhRope(_nn.Module):
    def __init__(self, dim, theta):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=F64) / dim)))
        self._yarn_attn_factor = 1.0

    def forward(self, T):
        f = torch.outer(torch.arange(T, dtype=F64), self.inv_freq)
        e = torch.cat((f, f), -1)
        return e.cos()[None, None], e.sin()[None, None]


def _dh_rot(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), -1)


class _DhAttn(_nn.Module):
    def __init__(self, c):
        super().__init__()
        d, H, kv = c.hidden_size, c.num_attention_heads, c.num_key_value_heads
        hd = d // H
        self.num_heads, self.num_kv_heads, self.head_dim, self.num_kv_groups = H, kv, hd, H // kv
        self.q_proj, self.k_proj = _nn.Linear(d, H * hd, bias=False), _nn.Linear(d, kv * hd, bias=False)
        self.v_proj, self.o_proj = _nn.Linear(d, kv * hd, bias=False), _nn.Linear(H * hd, d, bias=False)
        b = "B" in c.canon_set
        self.canon_b_q = _DhCanon(H * hd, c.canon_kernel) if b else None
        self.canon_b_k = _DhCanon(kv * hd, c.canon_kernel) if b else None
        self.canon_b_v = _DhCanon(kv * hd, c.canon_kernel) if b else None
        self.q_norm, self.k_norm = _DhNorm(hd, c.rms_norm_eps), _DhNorm(hd, c.rms_norm_eps)
        self.rotary_emb = _DhRope(hd, c.rope_theta)

    def forward(self, h):
        B, T, _ = h.shape
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        if self.canon_b_q is not None:
            q, k, v = self.canon_b_q(q), self.canon_b_k(k), self.canon_b_v(v)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(T)
        q, k = q * cos + _dh_rot(q) * sin, k * cos + _dh_rot(k) * sin
        q, k = self.q_norm(q), self.k_norm(k)
        k, v = k.repeat_interleave(self.num_kv_groups, 1), v.repeat_interleave(self.num_kv_groups, 1)
        w = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        w = w + torch.triu(torch.full((T, T), float("-inf"), dtype=h.dtype), diagonal=1)
        o = torch.matmul(w.softmax(-1), v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(o)


class _DhMLP(_nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f = c.hidden_size, c.intermediate_size
        self.gate_proj, self.up_proj = _nn.Linear(d, f, bias=False), _nn.Linear(d, f, bias=False)
        self.down_proj = _nn.Linear(f, d, bias=False)
        self.canon_d = _DhCanon(f, c.canon_kernel) if "D" in c.canon_set else None

    def forward(self, x):
        h = _F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.canon_d(h) if self.canon_d is not None else h)


class _DhLayer(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = _DhNorm(c.hidden_size), _DhNorm(c.hidden_size)
        self.canon_a = _DhCanon(c.hidden_size, c.canon_kernel) if "A" in c.canon_set else None
        self.canon_c = _DhCanon(c.hidden_size, c.canon_kernel) if "C" in c.canon_set else None
        self.self_attn, self.mlp = _DhAttn(c), _DhMLP(c)

    def forward(self, x):
        h = self.input_layernorm(x)
        x = x + self.self_attn(self.canon_a(h) if self.canon_a is not None else h)
        h = self.post_attention_layernorm(x)
        return x + self.mlp(self.canon_c(h) if self.canon_c is not None else h)


class _DhLM(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.model = _nn.Module()
        self.model.embed_tokens = _nn.Embedding(c.vocab_size, c.hidden_size)
        self.model.layers = _nn.ModuleList(_DhLayer(c) for _ in range(c.num_hidden_layers))
        self.model.norm = _DhNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, ids):
        x = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            x = layer(x)
        lg = self.lm_head(self.model.norm(x))
        cap = self.config.logit_softcap
        return cap * torch.tanh(lg / cap)


@pytest.mark.parametrize("sites", ["ABCD", "AC"])
@torch.no_grad()
def test_same_host_carries_dhara_canon_layers(sites):
    """Dhara-AR vs from_dhara against the REFERENCE above: Canon convs (kernel 4, residual) at every position (the
    snapshot's ABCD) or A / C only, GQA, post-rotary qk-norms, the logit softcap, tied embeddings. fp64. The Canon convs
    switched off move the logits."""
    from types import SimpleNamespace as NS
    from elephantine.universal import from_dhara
    c = NS(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
           num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=8e6, canon_set=sites, canon_kernel=4, logit_softcap=30.0,
           use_logit_softcap=True)
    t = _rand_(_DhLM(c).to(F64).eval(), s=0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    ref = t(x)
    u = from_dhara(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    for m_ in u.modules():
        if type(m_).__name__ == "UCanon":
            m_.c.zero_()
    assert _rel(ref, u(x)) > 1e-3


# --- Crowd-v1: a hashed bigram embedding at the input (engram_crowd) ----------------------------------------------
# REFERENCE, not the remote code: Vxtzq/Crowd-v1 modeling_crowd.py (md5 b1337c25ab928086c3c9f863824ccd71; every
# snapshot's copy is the same blob), read as text: rotate_half l. 16-18, CrowdAttention l. 21-57, CrowdMLP l. 60-68,
# CrowdBlock l. 71-82, CrowdEngram l. 85-97 (prev * 1000003 + ids mod buckets, prev 0 at position 0),
# CrowdForCausalLM.__init__ / forward l. 113-181 (wte + engram, the cached rotary, the final LayerNorm, the tied head).

class _CrAttn(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        d = c.hidden_size
        self.q = _nn.Linear(d, self.num_heads * self.head_dim, bias=False)
        self.k = _nn.Linear(d, self.num_kv_heads * self.head_dim, bias=False)
        self.v = _nn.Linear(d, self.num_kv_heads * self.head_dim, bias=False)
        self.o = _nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos[:, :, :T] + _dh_rot(q) * sin[:, :, :T]
        k = k * cos[:, :, :T] + _dh_rot(k) * sin[:, :, :T]
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        a = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        a = a.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, T, -1))


class _CrBlock(_nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f = c.hidden_size, c.intermediate_size
        self.ln1, self.ln2 = _nn.LayerNorm(d), _nn.LayerNorm(d)
        self.attn = _CrAttn(c)
        self.mlp = _nn.Module()
        self.mlp.w1, self.mlp.w2, self.mlp.w3 = _nn.Linear(d, f, bias=False), _nn.Linear(d, f, bias=False), _nn.Linear(f, d, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        return x + self.mlp.w3(_F.silu(self.mlp.w1(h)) * self.mlp.w2(h))


class _CrLM(_nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.wte = _nn.Embedding(c.vocab_size, c.hidden_size)
        self.engram = _nn.Module()
        self.engram.table = _nn.Embedding(c.engram_buckets, c.hidden_size)
        self.blocks = _nn.ModuleList(_CrBlock(c) for _ in range(c.num_hidden_layers))
        self.ln = _nn.LayerNorm(c.hidden_size)
        self.head = _nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=F64) / c.head_dim))
        f = torch.outer(torch.arange(64, dtype=F64), inv)
        f = torch.cat((f, f), dim=-1)
        self.cos_cached, self.sin_cached = f.cos()[None, None], f.sin()[None, None]

    def forward(self, ids):
        prev = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        h = (prev * 1000003 + ids) % self.engram.table.num_embeddings
        x = self.wte(ids) + self.engram.table(h)
        for b in self.blocks:
            x = b(x, self.cos_cached, self.sin_cached)
        return self.head(self.ln(x))


@torch.no_grad()
def test_same_host_carries_crowd_bigram_engram():
    """Crowd-v1 vs from_crowd against the REFERENCE above (61 buckets: collisions bind), fp64. Without the bigram
    table the logits move."""
    from types import SimpleNamespace as NS
    from elephantine.universal import from_crowd
    c = NS(vocab_size=97, hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
           head_dim=8, intermediate_size=48, rope_theta=10000.0, engram_buckets=61)
    t = _rand_(_CrLM(c).to(F64).eval(), s=0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    ref = t(x)
    u = from_crowd(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    u.ngram_in = None
    assert _rel(ref, u(x)) > 1e-3


# --- LLaMA-MoE: the MLP router (router_mlp) -----------------------------------------------------------------------
# REFERENCE, not the remote code: llama-moe/LLaMA-MoE-v1-3_0B-2_16 modeling_llama_moe_hf.py (snapshot 5429d300, md5
# 4723dc9c357870f6aba3f9b2e890ef7c), read as text: TopKBalancedNoisyGate l. 393-540 (eval path: logits = gate_network(x)
# with gate_network "mlp" = Linear -> Tanh -> Linear, l. 440-455; the top k + 1, the first k, softmax over them),
# LinearGLUExperts l. 543-675 (per-expert SwiGLU of size_experts[i]), UniversalCalculator l. 692-767 (outputs x score x
# score_scale_factor, summed), BaseMoELayer.forward l. 808-822, LlamaMoEDecoderLayer l. 976-1070 (the Llama layer with
# mlp = the MoE layer; its .hidden_states). The attention / norms / rotary are its copy of Llama's (the stock Llama
# modules stand for them); the MoE layer below returns the hidden states the decoder layer reads.

class _LmGate(_nn.Module):
    def __init__(self, d, E, k):
        super().__init__()
        self.num_experts, self.num_selects, self.gate_network_type, self.use_softmax = E, k, "mlp", True
        self.gate_network = _nn.Sequential(_nn.Linear(d, E, bias=False), _nn.Tanh(), _nn.Linear(E, E, bias=False))

    def forward(self, x):
        logits = self.gate_network(x)
        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)
        return top_indices[:, :self.num_selects], top_logits[:, :self.num_selects].softmax(1)


class _LmExperts(_nn.Module):
    def __init__(self, d, sizes):
        super().__init__()
        self.num_experts = len(sizes)
        self.weight_gate = _nn.ParameterList(_nn.Parameter(torch.zeros(n, d)) for n in sizes)
        self.weight_up = _nn.ParameterList(_nn.Parameter(torch.zeros(n, d)) for n in sizes)
        self.weight_down = _nn.ParameterList(_nn.Parameter(torch.zeros(d, n)) for n in sizes)

    def forward(self, x, i):
        return _F.linear(_F.silu(_F.linear(x, self.weight_gate[i])) * _F.linear(x, self.weight_up[i]), self.weight_down[i])


class _LmMoE(_nn.Module):
    def __init__(self, d, sizes, k, scale):
        super().__init__()
        self.input_size = d
        self.gate = _LmGate(d, len(sizes), k)
        self.calculator = _nn.Module()
        self.calculator.experts = _LmExperts(d, sizes)
        self.calculator.multiply_gate_scores, self.calculator.score_scale_factor = True, scale

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, self.input_size)
        idx, sc = self.gate(x)
        n, k = idx.shape
        idx, sc = idx.flatten(), sc.flatten()
        rows = torch.arange(n).repeat_interleave(k)
        y = torch.zeros_like(x)
        for e in range(self.calculator.experts.num_experts):
            sel = idx == e
            if sel.any():
                out = self.calculator.experts(x[rows[sel]], e) * (sc[sel] * self.calculator.score_scale_factor)[:, None]
                y = y.index_add(0, rows[sel], out)
        return y.reshape(shp)


@torch.no_grad()
def test_same_host_carries_llama_moe_mlp_router():
    """LLaMA-MoE vs from_llama_moe: the stock Llama body with the REFERENCE MoE layers above (6 experts of unequal
    sizes, top-2, the tanh MLP router, score scale 8); fp64 with the fp32 casts lifted. A linear router in place of the
    MLP router (W1 alone) moves the logits."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from elephantine.universal import from_llama_moe
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, rms_norm_eps=1e-5, attn_implementation="eager")
    t = LlamaForCausalLM(cfg).to(F64).eval()
    for layer in t.model.layers:
        layer.mlp = _LmMoE(32, [8, 8, 6, 8, 10, 8], 2, 8.0).to(F64)
    _rand_(t, s=0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x).logits
    u = from_llama_moe(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    for ul in u.layers:
        ul.ffn.router.weight.copy_(ul.ffn.router_mlp.w1.weight)
        ul.ffn.router_mlp = None
    assert _rel(ref, u(x)) > 1e-3


# --- Snowflake Arctic: the dense residual MLP beside the parallel MoE (par_moe_res) -------------------------------
# REFERENCE, not the remote code: Snowflake/snowflake-arctic-instruct modeling_arctic.py (snapshot be318cae, md5
# 5a35a1f1503f556df5bbe31f7b49a813), read as text: ArcticMLP l. 866-914 (w1 / w3 / w2 SwiGLU; the residual MLP is
# hidden_size wide), ArcticMoE._moe_foreward l. 961-1003 (softmax, top-k, renorm), ArcticDecoderLayer l. 1020-1112
# (h = x + attn(n_1 x); out = h + residual_mlp(n_r h) + moe(post_attention_layernorm(x))). Its attention is the
# Mixtral attention (the stock Llama attention stands for it, no window).

class _ArcMLP(_nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1, self.w2, self.w3 = _nn.Linear(d, f, bias=False), _nn.Linear(f, d, bias=False), _nn.Linear(d, f, bias=False)

    def forward(self, x):
        return self.w2(_F.silu(self.w1(x)) * self.w3(x))


class _ArcMoE(_nn.Module):
    def __init__(self, d, f, E, k, is_moe):
        super().__init__()
        self.num_experts, self.top_k, self.is_moe_layer = E, k, is_moe
        if is_moe:
            self.gate = _nn.Linear(d, E, bias=False)
            self.experts = _nn.ModuleList(_ArcMLP(d, f) for _ in range(E))
        else:
            self.mlp = _ArcMLP(d, f)

    def forward(self, h):
        if not self.is_moe_layer:
            return self.mlp(h)
        B, T, D = h.shape
        h = h.view(-1, D)
        w = _F.softmax(self.gate(h), dim=1)
        w, sel = torch.topk(w, self.top_k, dim=-1)
        if self.top_k > 1:
            w = w / w.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(h)
        for e, ex in enumerate(self.experts):
            tok, kk = torch.where(sel == e)
            if tok.numel():
                out = out.index_add(0, tok, ex(h[tok]) * w[tok, kk, None])
        return out.view(B, T, D)


class _ArcLayer(_nn.Module):
    def __init__(self, cfg, i, par):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm
        d = cfg.hidden_size
        self.self_attn = LlamaAttention(cfg, i)
        self.block_sparse_moe = _ArcMoE(d, cfg.intermediate_size, cfg.num_local_experts, cfg.num_experts_per_tok, i > 0)
        self.input_layernorm, self.post_attention_layernorm = LlamaRMSNorm(d, cfg.rms_norm_eps), LlamaRMSNorm(d, cfg.rms_norm_eps)
        self.parallel_attn_mlp_res = par and self.block_sparse_moe.is_moe_layer
        if self.parallel_attn_mlp_res:
            self.residual_layernorm, self.residual_mlp = LlamaRMSNorm(d, cfg.rms_norm_eps), _ArcMLP(d, d)

    def forward(self, x, pe, mask):
        a, _ = self.self_attn(self.input_layernorm(x), position_embeddings=pe, attention_mask=mask)
        h = x + a
        if self.parallel_attn_mlp_res:
            rr = h + self.residual_mlp(self.residual_layernorm(h))
            return rr + self.block_sparse_moe(self.post_attention_layernorm(x))
        return h + self.block_sparse_moe(self.post_attention_layernorm(h))


@pytest.mark.parametrize("par", [True, False])
@torch.no_grad()
def test_same_host_carries_arctic_parallel_residual_moe(par):
    """Arctic vs from_arctic against the REFERENCE above (1 dense + 2 MoE layers, 4 experts top-2; with and without
    parallel_attn_mlp_res); fp64, the fp32 casts lifted. The experts reading the post-attention stream (moe_par off)
    move the logits."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from elephantine.universal import from_arctic
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=24, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, rms_norm_eps=1e-5, attn_implementation="eager")
    cfg.num_local_experts, cfg.num_experts_per_tok = 4, 2
    t = LlamaForCausalLM(cfg).to(F64).eval()
    t.model.layers = _nn.ModuleList(_ArcLayer(cfg, i, par) for i in range(3)).to(F64)
    _rand_(t, s=0.3)

    def run(ids):
        m = t.model
        h = m.embed_tokens(ids)
        T = ids.shape[1]
        pe = m.rotary_emb(h, torch.arange(T)[None])
        mask = torch.full((T, T), float("-inf"), dtype=h.dtype).triu(1)[None, None]
        for layer in m.layers:
            h = layer(h, pe, mask)
        return t.lm_head(m.norm(h))
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = run(x)
    u = from_arctic(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    if par:
        for ul in u.layers:
            ul.moe_par.zero_()
        assert _rel(ref, u(x)) > 1e-3


# --- HunYuan-A13B: expert-capacity drops (moe_capacity) -----------------------------------------------------------
# REFERENCE, not the remote code: tencent/Hunyuan-A13B-Instruct modeling_hunyuan.py (snapshot 290ddb9a, md5
# 957ffd147b1d2b010102fa7e0ca5a38d), read as text: topkgating l. 74-140 (softmax, capacity max(k, k S // E), top-k,
# renormalised router probabilities, rank-major priority, dispatch / combine masks) and HunYuanMoE.forward l. 484-510
# (shared MLP + the dispatched experts). The rest of the remote model is the stock HunYuan-MoE (its MoE minus the
# capacity: transformers 4.57 modeling_hunyuan_v1_moe.py), so the reference is the stock class with each MoE's
# forward replaced by the transcription.

def _hy_topkgating(logits, topk):
    gates = _F.softmax(logits, dim=1)
    cap = max(topk, topk * gates.shape[0] // gates.shape[1])
    E = int(gates.shape[1])
    _, eidx = torch.topk(gates, topk)
    emask = _F.one_hot(eidx, E)
    gates_s = torch.clamp(torch.matmul(emask.to(gates.dtype), gates.unsqueeze(-1)).sum(dim=1), min=torch.finfo(gates.dtype).eps)
    probs = gates / gates_s
    eidx = torch.transpose(eidx, 0, 1).reshape(-1)
    emask = _F.one_hot(eidx, E).to(torch.int32)
    prio = torch.cumsum(emask, dim=0) * emask - 1
    prio = torch.transpose(prio.reshape((topk, -1, E)), 0, 1).max(dim=1)[0]
    valid = torch.logical_and(prio >= 0, prio < cap)
    prio = torch.masked_fill(prio, ~valid, 0)
    dispatch = _F.one_hot(prio.long(), cap).to(torch.bool)
    dispatch = torch.masked_fill(dispatch, ~valid.unsqueeze(-1).expand(-1, -1, cap), 0)
    combine = torch.einsum("...te,...tec->...tec", probs, dispatch)
    return combine, dispatch


def _hy_moe_forward(moe):
    def fwd(h):
        B, T, D = h.shape
        mlp = moe.shared_mlp(h)
        combine, dispatch = _hy_topkgating(moe.gate.wg(h.reshape(-1, D)), moe.top_k)
        x = h.reshape(-1, D)
        din = torch.einsum("sec,sm->ecm", dispatch.to(h.dtype), x)
        out = torch.cat([ex(ch) for ch, ex in zip(din.chunk(moe.num_experts, dim=0), moe.experts)], dim=0)
        return mlp + torch.einsum("sec,ecm->sm", combine.to(h.dtype), out).reshape(B, T, D)
    return fwd


@torch.no_grad()
def test_same_host_carries_hunyuan_a13b_capacity_drops():
    """HunYuan-A13B vs from_hunyuan_capacity: the stock HunYuan-MoE body with the REFERENCE capacity router above
    (8 experts, top-2, 2 x 12 tokens: capacity 6, binding); fp64, the fp32 casts lifted. Without the drops (cap_drop 0)
    the logits move."""
    from transformers import HunYuanMoEV1Config, HunYuanMoEV1ForCausalLM
    from elephantine.branches_stream import capacity_keep
    from elephantine.universal import from_hunyuan_capacity
    cfg = HunYuanMoEV1Config(vocab_size=97, hidden_size=32, intermediate_size=24, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=8, moe_topk=2,
                             rms_norm_eps=1e-5, attn_implementation="eager")
    t = _rand_(HunYuanMoEV1ForCausalLM(cfg).to(F64).eval(), s=0.3)
    for layer in t.model.layers:
        layer.mlp.forward = _hy_moe_forward(layer.mlp)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x).logits
    u = from_hunyuan_capacity(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    for ul in u.layers:
        ul.ffn.cap_drop.zero_()
    assert _rel(ref, u(x)) > 1e-4
    ix = torch.zeros(2, 12, 2, dtype=torch.long)                # every token to experts 0, 1: capacity 6 of 24 binds
    ix[..., 1] = 1
    assert int(capacity_keep(ix, 8).sum()) == 12


# --- LLaDA2.2: block routing (block_routing) -----------------------------------------------------------------------
# REFERENCE, not the remote code: inclusionAI/LLaDA2.2-mini modeling_llada2_moe.py (snapshot e4f77d25, md5
# 34d10d1107c3d6ef00745121c6c426ca), read as text: LLaDA2MoeMLP l. 204-217, LLaDA2MoeGate l. 220-298 (sigmoid scores,
# selection on scores + expert_bias through block_routing l. 244-275, gathered unbiased scores renormalised, x
# routed_scaling_factor), LLaDA2MoeSparseMoeBlock l. 301-434 (moe_infer + the shared experts). Block level (no host
# loader for the LLaDA2 body).

class _LlMLP(_nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.gate_proj, self.up_proj = _nn.Linear(d, f, bias=False), _nn.Linear(d, f, bias=False)
        self.down_proj = _nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down_proj(_F.silu(self.gate_proj(x)) * self.up_proj(x))


class _LlGate(_nn.Module):
    def __init__(self, d, E, k, block, cap, scale):
        super().__init__()
        self.top_k, self.num_experts, self.block_size, self.expert_capacity = k, E, block, cap
        self.weight = _nn.Parameter(torch.zeros(E, d))
        self.routed_scaling_factor = scale
        self.register_buffer("expert_bias", torch.zeros(E))

    def block_routing(self, s):
        n = s.shape[0]
        nb = n // self.block_size
        bs = s.view(nb, self.block_size, self.num_experts).max(dim=1).values
        _, bi = torch.topk(bs, k=self.expert_capacity, dim=-1)
        allowed = torch.zeros(nb, self.num_experts, dtype=torch.bool).scatter_(1, bi, True)
        allowed = allowed.unsqueeze(1).expand(-1, self.block_size, -1).reshape(n, self.num_experts)
        return torch.topk(s.masked_fill(~allowed, -torch.inf), k=self.top_k, dim=-1)[1]

    def forward(self, h):
        h = h.view(-1, h.shape[-1])
        scores = torch.sigmoid(_F.linear(h, self.weight))
        idx = self.block_routing(scores + self.expert_bias)
        w = torch.gather(scores, dim=1, index=idx)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else w
        return idx, w * self.routed_scaling_factor


class _LlMoE(_nn.Module):
    def __init__(self, d, f, E, k, block, cap, scale):
        super().__init__()
        self.experts = _nn.ModuleList(_LlMLP(d, f) for _ in range(E))
        self.gate = _LlGate(d, E, k, block, cap, scale)
        self.shared_experts = _LlMLP(d, f)

    def forward(self, h):
        B, T, D = h.shape
        idx, w = self.gate(h)
        x = h.view(-1, D)
        y = torch.zeros_like(x)
        for e, ex in enumerate(self.experts):
            tok, kk = torch.where(idx == e)
            if tok.numel():
                y = y.index_add(0, tok, ex(x[tok]) * w[tok, kk, None])
        return y.view(B, T, D) + self.shared_experts(h)


@torch.no_grad()
def test_llada22_block_routing_moe():
    """LLaDA2.2's block-routed MoE (REFERENCE above: 8 experts, top-2, blocks of 4 tokens keeping 3 experts each) vs
    UFFN with blk_route (llada2_block_moe_into) at fp64; per-token routing (blk_route off) moves it."""
    from elephantine.universal import UConfig, UFFN, llada2_block_moe_into
    torch.manual_seed(0)
    moe = _LlMoE(32, 16, 8, 2, 4, 3, 2.5).to(F64)
    for p in moe.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    moe.gate.expert_bias.normal_(0, 0.1)
    x = torch.randn(2, 12, 32, dtype=F64)
    ref = moe(x)
    f = UFFN(UConfig(vocab=1, d=32, L=1, experts=8, expert_ff=16, top_k=2, shared_ff=16))
    llada2_block_moe_into(f, moe)
    assert _rel(ref, f(x)) < 1e-12, _rel(ref, f(x))
    f.blk_route.zero_()
    assert _rel(ref, f(x)) > 1e-3


# --- Llama-4: input-gated routed experts (router_in_gate) ----------------------------------------------------------

@pytest.mark.parametrize("k,step", [(1, 1), (2, 2)])
@torch.no_grad()
def test_same_host_carries_llama4_input_gated_experts(k, step):
    """Llama-4 text (the stock transformers Llama4ForCausalLM) vs from_llama4: experts reading sigmoid(top logit) x
    (UFFN.in_gate) beside a shared expert (every layer MoE, or interleaved with dense layers), the interleaved rotary
    with post-rotary L2 qk-norms, NoPE layers with the query temperature (floor((t + 1) / floor_scale): qtemp_off), the
    chunk (64) covering T. fp64, the fp32 casts lifted. Output-weighted experts (in_gate off) move the logits."""
    from transformers import Llama4TextConfig, Llama4ForCausalLM
    from elephantine.universal import from_llama4
    cfg = Llama4TextConfig(vocab_size=97, hidden_size=32, intermediate_size=16, intermediate_size_mlp=40,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                           num_local_experts=4, num_experts_per_tok=k, interleave_moe_layer_step=step,
                           no_rope_layers=[1, 1, 0, 1], use_qk_norm=True, attn_temperature_tuning=True, floor_scale=4,
                           attn_scale=0.3, attention_chunk_size=64, rms_norm_eps=1e-5, rope_theta=500000.0,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2, attn_implementation="eager")
    t = Llama4ForCausalLM(cfg).to(F64).eval()
    torch.manual_seed(0)
    for p in t.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x).logits
    u = from_llama4(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    for ul in u.layers:
        ul.ffn.in_gate.zero_()
    assert _rel(ref, u(x)) > 1e-3


# --- ERNIE-5.5: summed dual-sigmoid routing (sigmoid2) and the multimax logits (multimax) --------------------------
# REFERENCE, not the remote code: baidu vLLM-ERNIE (~/.cache/nocode/ernie5_5_moe/gh__baidu__vLLM-ERNIE), read as text:
# moe.py (md5 846aea04420b887858430a372f67bdb0) l. 66-88 ernie5_5_routing and l. 214-224 the split-feature scores
# sigmoid(l[:, :E]) + sigmoid(l[:, E:]); model.py (md5 3489ba7b98413c389664ebb63a73a65f) l. 84-101 the multimax.
# Block level (ERNIE-5.5's attention features and its latent MoE are not carried); the experts here are plain SwiGLU
# stand-ins so the routing is what is compared.

@torch.no_grad()
def test_ernie55_dual_sigmoid_router_block():
    """ERNIE-5.5's routing (REFERENCE above) over 8 SwiGLU experts, top-3, vs UFFN with router_dual
    (ernie55_router_into) at fp64; a single sigmoid router moves it."""
    from elephantine.universal import UConfig, UniversalElephantine, ernie55_router_into
    torch.manual_seed(0)
    d, E, f_, k = 32, 8, 16, 3
    gw, bias, scale = torch.randn(2 * E, d, dtype=F64), torch.randn(E, dtype=F64) * 0.1, 1 + 0.2 * torch.rand(E, dtype=F64)
    Wg, Wu, Wd = (torch.randn(E, f_, d, dtype=F64) * 0.3, torch.randn(E, f_, d, dtype=F64) * 0.3,
                  torch.randn(E, d, f_, dtype=F64) * 0.3)
    x = torch.randn(2, 12, d, dtype=F64)
    xs = x.reshape(-1, d)
    lg = xs @ gw.T
    scores = lg[:, :E].sigmoid() + lg[:, E:].sigmoid()
    ids = torch.topk(scores + bias, k, dim=-1, sorted=False).indices
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-20) * scale[ids]
    ref = torch.zeros_like(xs)
    for j in range(k):
        e = ids[:, j]
        h = _F.silu(torch.einsum("nfd,nd->nf", Wg[e], xs)) * torch.einsum("nfd,nd->nf", Wu[e], xs)
        ref = ref + w[:, j, None] * torch.einsum("ndf,nf->nd", Wd[e], h)
    f = UniversalElephantine(UConfig(vocab=1, d=d, L=1, experts=E, expert_ff=f_, top_k=k, router_dual=True)).layers[0].ffn
    f.w_gate.copy_(Wg); f.w_up.copy_(Wu); f.w_down.copy_(Wd)
    ernie55_router_into(f, gw, bias, scale)
    assert _rel(ref.view_as(x), f(x)) < 1e-12, _rel(ref.view_as(x), f(x))
    f.router2 = None
    assert _rel(ref.view_as(x), f(x)) > 1e-3


@torch.no_grad()
def test_ernie55_multimax_logits():
    """ERNIE-5.5's multimax (REFERENCE above; ranges / slopes spanning the logits) as the host readout's UMultimax
    (UConfig multimax) at fp64; t = 0 is the plain head."""
    from elephantine.universal import UConfig, UniversalElephantine
    torch.manual_seed(0)
    u = UniversalElephantine(UConfig(vocab=97, d=32, L=1, multimax=True))
    u.embed.weight.normal_(0, 1.0); u.lm_head.weight.normal_(0, 0.5)
    r, t = torch.tensor([-1.0, 1.0, -2.0, 2.0], dtype=F64), torch.tensor([0.5, -0.3, 0.1, -0.05], dtype=F64)
    u.multimax.r.copy_(r); u.multimax.t.copy_(t)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    l = u.lm_head(u.norm(u.embed(x)))
    ref = l + t[0] * (r[0] - l).relu() + t[1] * (l - r[1]).relu() + t[2] * (r[2] - l).relu().square() + \
        t[3] * (l - r[3]).relu().square()
    assert _rel(ref, u(x)) < 1e-14
    u.multimax.t.zero_()
    assert _rel(ref, u(x)) > 1e-2


@torch.no_grad()
def test_same_host_carries_zamba_v1_shared_block():
    """Zamba v1 (the stock transformers ZambaForCausalLM, its torch slow path) vs from_zamba: multi-head (2) Mamba-1
    layers and the single shared transformer block over concat(x, embedding) on the hybrid layers (UCatBlock). fp64, the
    fp32 casts lifted. The shared block switched off moves the logits."""
    from transformers import ZambaConfig, ZambaForCausalLM
    from elephantine.universal import from_zamba
    cfg = ZambaConfig(vocab_size=97, hidden_size=32, num_hidden_layers=5, attn_layer_period=2, attn_layer_offset=1,
                      mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, n_mamba_heads=2, mamba_dt_rank=4,
                      intermediate_size=48, num_attention_heads=4, num_key_value_heads=4, use_mamba_kernels=False,
                      attn_implementation="eager")
    t = _rand_(ZambaForCausalLM(cfg).to(F64).eval(), s=0.1)
    assert any(hasattr(l, "shared_transf") for l in t.model.layers)
    x = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    with _lift():
        ref = t(x, use_cache=False).logits
    u = from_zamba(t)
    assert _rel(ref, u(x)) < 1e-12, _rel(ref, u(x))
    for ul in u.layers:
        if ul.cat is not None:
            ul.cat.c.zero_()
    assert _rel(ref, u(x)) > 1e-3


# --- Muse-Spark: the latent MoE with per-expert output norms (expert_norm) --------------------------------------
# REFERENCE, not the remote code: transformers main models/muse_spark/modular_muse_spark.py (md5
# 5d28048c7fbcfd892acc63218c95e738), read as text: MuseSparkTextCenteredRMSNorm l. 292-298, MuseSparkTextInputScaled-
# RMSNorm l. 300-305 (+ the _norm of modeling_muse_spark.py l. 133-134 / 153-154), MuseSparkTopkRouter l. 315-337,
# MuseSparkExperts l. 340-374, MuseSparkMoE l. 377-397. Block level (its attention output norm is not carried).

class _MsNorm(_nn.Module):
    def __init__(self, n, eps, scaled_first):
        super().__init__()
        self.weight, self.eps, self.first = _nn.Parameter(torch.zeros(n)), eps, scaled_first

    def forward(self, x):
        nrm = lambda t: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)
        return nrm(x * (1.0 + self.weight)) if self.first else nrm(x) * (1.0 + self.weight)


class _MsMoE(_nn.Module):
    def __init__(self, d, m, f, E, k, mult, scale, eps=1e-6):
        super().__init__()
        self.gate = _nn.Module()
        self.gate.weight = _nn.Parameter(torch.zeros(E, d))
        self.gate.register_buffer("e_score_correction_bias", torch.zeros(E))
        self.gate.output_multiplier, self.gate.routed_scaling_factor, self.gate.norm_topk_prob_eps = mult, scale, 1e-15
        self.top_k = k
        self.pre_expert_proj, self.post_expert_proj = _nn.Linear(d, m, bias=False), _nn.Linear(m, d, bias=False)
        self.pre_expert_norm = _MsNorm(m, eps, False)
        self.experts = _nn.Module()
        self.experts.gate_up_proj = _nn.Parameter(torch.zeros(E, 2 * f, m))
        self.experts.down_proj = _nn.Parameter(torch.zeros(E, m, f))
        self.experts.post_expert_norm = _MsNorm(m, eps, True)

    def forward(self, h):
        B, T, D = h.shape
        x = h.view(-1, D)
        g = self.gate
        logits = _F.linear(x, g.weight) * g.output_multiplier
        w = torch.sigmoid(logits)
        idx = torch.topk(w + g.e_score_correction_bias, k=self.top_k, dim=-1, sorted=False)[1]
        tw = w.gather(1, idx)
        tw = tw / (tw.sum(dim=-1, keepdim=True) + g.norm_topk_prob_eps) * g.routed_scaling_factor
        z = self.pre_expert_norm(self.pre_expert_proj(x))
        out = torch.zeros_like(z)
        ex = self.experts
        for e in range(ex.gate_up_proj.shape[0]):
            tok, kk = torch.where(idx == e)
            if tok.numel():
                gt, up = _F.linear(z[tok], ex.gate_up_proj[e]).chunk(2, dim=-1)
                y = ex.post_expert_norm(_F.linear(_F.silu(gt) * up, ex.down_proj[e]))
                out = out.index_add(0, tok, y * tw[tok, kk, None])
        return self.post_expert_proj(out).view(B, T, D)


@torch.no_grad()
def test_muse_spark_latent_moe_expert_norms():
    """Muse-Spark's latent MoE (REFERENCE above: hidden 32, latent 12, 6 experts top-2) vs UFFN with latent_moe
    (muse_spark_moe_into) at fp64; without the per-expert output norms the output moves."""
    from elephantine.universal import UConfig, UniversalElephantine, muse_spark_moe_into
    torch.manual_seed(0)
    moe = _MsMoE(32, 12, 16, 6, 2, 0.7, 2.0).to(F64)
    for p in moe.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    moe.gate.e_score_correction_bias.normal_(0, 0.1)
    x = torch.randn(2, 12, 32, dtype=F64)
    ref = moe(x)
    f = UniversalElephantine(UConfig(vocab=1, d=32, L=1, experts=6, expert_ff=16, top_k=2,
                                     latent_moe=12)).layers[0].ffn
    muse_spark_moe_into(f, moe)
    assert _rel(ref, f(x)) < 1e-12, _rel(ref, f(x))
    f.e_post = None
    assert _rel(ref, f(x)) > 1e-3
