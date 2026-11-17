"""Qwen3.5-family remote-code models (Agnes, AllGDN-Engram, EdgeRock) on the universal host, WITHOUT running remote code.

Each test mirrors a remote-gated test in tests/test_universal_elephantine.py, but the target is a REFERENCE
transcription of the remote modelling file (read as text, never run) in tests/refs/{agnes,allgdn,edgerock}.py;
each module's header names the repo, snapshot, sha256 and transcribed lines. AllGDN and EdgeRock subclass the
stock transformers 5.18 Qwen3.5 classes exactly as the remote files do, so everything runs inside the TF5
subprocess (`_tf5_run`). Agnes is a self-contained rename of Qwen3.5; its reference is a full fp64 transcription.

Bounds: Agnes reaches fp64 round-off (1e-12). AllGDN / EdgeRock whole models keep the stock Qwen3.5 fp32 casts
(RMSNorm, gated norm, the chunked delta rule and the rotary table compute in fp32), so the whole-model gap sits at
~1e-7 and is bounded at 1e-5 relative, as in the old tests; the block test with those casts lifted reaches ~1e-16.

Run:  pytest tests/test_notremote_qwen35x.py -q -n 0
"""
from tests.test_universal_elephantine import _tf5_run

REL = '''
rel = lambda a, b: (a - b).abs().max().item() / max(a.abs().max().item(), 1)
'''


def test_same_host_carries_agnes_parallel_ffn_notremote():
    """Agnes-3.0-Flash: a renamed Qwen3.5 (agnes_delta / agnes_global layer types, delta_attn / global_attn) with a
    parallel SwiGLU summed into every layer's MLP; the host folds it into the dense FFN's hidden channels.
    Reference: tests/refs/agnes.py (fp64-clean). Measured gap ~1e-15 relative. Switched off on the host (the
    parallel channels' down-projection zeroed) the logits move by ~0.68 relative, and the reference with
    `mlp.parallel_ffn = None` agrees with that switched-off host to ~1e-15."""
    gap, scale = _tf5_run(REL + '''
from tests.refs import agnes as am
from elephantine.universal import from_qwen35
cfg = am.AgnesTextConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=6,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48,
    parallel_ffn_intermediate_size=16,
    layer_types=[am.LAYER_DELTA, am.LAYER_DELTA, am.LAYER_DELTA, am.LAYER_GLOBAL])
torch.manual_seed(0)
t = rand_(am.AgnesForCausalLM(cfg).to(F64).eval())
assert t.model.layers[0].mlp.parallel_ffn is not None
x = ids(97)
with torch.no_grad():
    ref = t(x, use_cache=False)
    u = from_qwen35(t)
    ours = u(x)
    for l in u.layers:                            # the parallel FFN's channels (48: 64 of the folded FFN)
        l.ffn.s_down.weight[:, 48:] = 0
    off = u(x)
    assert rel(ref, off) > 1e-2, rel(ref, off)
    for l in t.model.layers:
        l.mlp.parallel_ffn = None
    assert rel(t(x), off) < 1e-12, rel(t(x), off)
    report(ref, ours)
''')
    assert gap < 1e-12 * max(scale, 1), (gap, scale)


def test_same_host_carries_allgdn_engram_notremote():
    """AllGDN-Engram: an all-gated-DeltaNet Qwen3.5 with Engram hashed n-gram memories injected before chosen
    layers (UEngram on the host). Reference: tests/refs/allgdn.py over the stock 5.18 Qwen3.5 body. Bound 1e-5
    relative (measured ~6e-7): the stock body's fp32 casts (see the module docstring). The Engram switched off
    on the host (c = 0) moves the logits by ~0.31 relative; the reference with its Engram value projections
    zeroed (its output is then exactly 0) agrees with that host to ~5e-7 (same fp32 floor)."""
    gap, scale = _tf5_run(REL + '''
from tests.refs import allgdn as ag
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
from elephantine.universal import from_qwen35
cfg = ag.AllGDNConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48,
    layer_types=["linear_attention"] * 4)
cfg.engram = {"layers": [1, 3], "orders": [2, 3], "heads": 2, "slots": 16, "dim": 8}
torch.manual_seed(0)
t = ag.AllGDNForCausalLM(cfg).to(F64).eval()
for l in t.model.layers:                          # CPU torch paths (fla kernels are GPU Triton)
    g = l.linear_attn
    g.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
    g.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
    g.causal_conv1d_fn = None
rand_(t)
assert set(t.engrams) == {"1", "3"}
x = ids(97, T=12)
x[0, 3:6] = x[0, 7:10] = torch.tensor([5, 6, 7])                 # a repeated trigram hits the same slots
with torch.no_grad():
    ref = t(x, use_cache=False).logits
    u = from_qwen35(t)
    ours = u(x)
    for l in u.layers:                            # the n-gram memory really matters
        if l.engram is not None:
            l.engram.c.zero_()
    off = u(x)
    assert rel(ref, off) > 1e-3, rel(ref, off)
    for e in t.engrams.values():
        e.val.weight.zero_()
    assert rel(t(x, use_cache=False).logits, off) < 1e-5
    report(ref, ours)
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


EDGE_CFG = '''
from tests.refs import edgerock as em
from elephantine.universal import from_qwen35
def edge_cfg(layer_types, swa_layers, **kw):
    cfg = em.EdgeRockConfig(layer_types=layer_types, **kw)
    cfg.fixed_memory_revision = em.INDEPENDENT_MEMORY_REVISION
    cfg.fixed_memory_swa_layers, cfg.fixed_memory_swa_window = swa_layers, 4
    cfg.independent_memory = {"admission_rank": 8, "capacity": 3, "max_commit_chunk": 2, "writer_intermediate_size": 24}
    return cfg
'''


def test_same_host_carries_edgerock_fixed_memory_notremote():
    """EdgeRock: Qwen3.5 gated-DeltaNet + gated attention whose attention layers are sliding-window and carry an
    independent fixed-capacity memory (SwiGLU key / value writer, learned admission score, readable one chunk
    later while in the top `capacity`, null key, per-head tanh-sigmoid read gate). Reference:
    tests/refs/edgerock.py over the stock 5.18 Qwen3.5 classes. Bound 1e-5 relative (measured ~3e-7): the stock
    fp32 casts. Switched off on the host, with the reference switched off the same way where it can be:
      selection (host sel_cap = inf, as the old test)       moves ~0.80; capacity 3 -> 1000 on both: agree ~2e-7,
                                                            and the reference itself moves (capacity bites);
      the memory read (host o_proj = 0; ref read_scale = 0) moves ~1.05; agree ~3e-7;
      the read gate (host hg_c = 0: ungated, as the old test) moves ~0.38;
      the window (host window = inf; ref swa_window = 1e6)  moves ~0.65; agree ~4e-7."""
    gap, scale = _tf5_run(REL + EDGE_CFG + '''
cfg = edge_cfg(["linear_attention", "full_attention", "linear_attention", "full_attention"], [1, 3],
    vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48)
torch.manual_seed(0)
t = em.EdgeRockForCausalLM(cfg).to(F64).eval()
rand_(t)
with torch.no_grad():                             # open the read gates (they init at tanh(0) = 0)
    for l in t.model.layers:
        if hasattr(l, "memory"):
            l.memory.read_scale.fill_(1.0)
assert sum(hasattr(l, "memory") for l in t.model.layers) == 2
assert [type(l.self_attn).__name__ for l in t.model.layers if hasattr(l, "self_attn")] == ["Qwen3_5FixedWindowAttention"] * 2
mems = [t.model.layers[i].memory for i in (1, 3)]
x = ids(97, T=14)
R = lambda: t(x, use_cache=False).logits
with torch.no_grad():
    ref = R()
    u = from_qwen35(t)
    ours = u(x)
    for l in u.layers:                            # selection really bites: capacity 3 < keys before the chunk
        if l.mem is not None:
            l.mem.sel_cap.fill_(float("inf"))
    assert rel(ref, u(x)) > 1e-3
    u = from_qwen35(t)
    for i in (1, 3):
        u.layers[i].mem.sel_cap.fill_(1000.0)
    for m in mems:
        m.capacity = 1000
    r_big = R()
    assert rel(ref, r_big) > 1e-3, rel(ref, r_big)
    assert rel(r_big, u(x)) < 1e-5, rel(r_big, u(x))
    for m in mems:
        m.capacity = 3
    u = from_qwen35(t)
    for i in (1, 3):                              # the memory really matters
        u.layers[i].mem.o_proj.weight.zero_()
    off = u(x)
    assert rel(ref, off) > 1e-3
    for m in mems:
        m.read_scale.zero_()
    assert rel(R(), off) < 1e-5, rel(R(), off)
    for m in mems:
        m.read_scale.fill_(1.0)
    u = from_qwen35(t)
    for l in u.layers:                            # so does its read gate
        if l.mem is not None:
            l.mem.hg_c.zero_()
    assert rel(ref, u(x)) > 1e-3
    u = from_qwen35(t)
    for l in u.layers:                            # and the window
        l.attn.window.fill_(float("inf"))
    off = u(x)
    assert rel(ref, off) > 1e-3
    for i in (1, 3):
        t.model.layers[i].self_attn.swa_window = 10 ** 6
    assert rel(R(), off) < 1e-5, rel(R(), off)
    report(ref, ours)
''')
    assert gap < 1e-5 * max(scale, 1), (gap, scale)


def test_edgerock_memory_and_engram_blocks_fp64_notremote():
    """Block level, with the stock Qwen3.5 RMSNorm's fp32 cast lifted to fp64 (a patch of the library class in
    the subprocess, as the old test did) and the rotary table built in fp64: EdgeRock's memory read
    (tests/refs/edgerock.py) and AllGDN's Engram (tests/refs/allgdn.py, whose EngramRMS is already lifted)
    agree with the host's mem_attn / UEngram blocks to ~1e-16 relative (bound 1e-13)."""
    gap, scale = _tf5_run(EDGE_CFG + '''
from tests.refs import allgdn as ag
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
q.Qwen3_5RMSNorm.forward = lambda self, x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1 + self.weight)
kw = dict(vocab_size=97, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=8,
    linear_conv_kernel_dim=4, intermediate_size=48)
cfg = edge_cfg(["linear_attention", "full_attention"], [1], **kw)
torch.manual_seed(0)
t = rand_(em.EdgeRockForCausalLM(cfg).to(F64).eval())
mem = t.model.layers[1].memory
with torch.no_grad():
    mem.read_scale.fill_(1.0)
torch.manual_seed(5)
h = torch.randn(2, 14, 64, dtype=F64)
with torch.no_grad():
    fr = torch.arange(14, dtype=F64)[:, None] * t.model.rotary_emb.inv_freq.double()[None]
    emb = torch.cat((fr, fr), -1)[None].expand(2, -1, -1)
    ref = mem(h, (emb.cos(), emb.sin()))
    ul = from_qwen35(t).layers[1]
    gaps = [(ref - ul.mem(ul.norm_m(h), True, {})).abs().max().item()]
    scales = [ref.abs().max().item()]
cfg = ag.AllGDNConfig(layer_types=["linear_attention"] * 2, **kw)
cfg.engram = {"layers": [1], "orders": [2, 3], "heads": 2, "slots": 16, "dim": 8}
torch.manual_seed(0)
t = rand_(ag.AllGDNForCausalLM(cfg).to(F64).eval())
x = ids(97, T=14)
with torch.no_grad():
    ref = t.engrams["1"](h, x)
    gaps.append((ref - from_qwen35(t).layers[1].engram(h, x)).abs().max().item())
    scales.append(ref.abs().max().item())
print("GAP", max(g / max(s, 1) for g, s in zip(gaps, scales)), "SCALE", 1.0)
''')
    assert gap < 1e-13, gap
