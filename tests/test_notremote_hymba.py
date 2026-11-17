"""Hymba on the universal host, against a labelled transcription of its remote code (tests/refs/hymba.py, read
as text, never executed): the remote-code-free counterparts of
tests/test_universal_elephantine.py::test_same_host_carries_hymba_parallel_kv_sharing_meta_tokens and
::test_same_host_carries_hymba_window_with_meta_token_sinks.

Run:  pytest tests/test_notremote_hymba.py -q
"""
import torch

from tests.refs import hymba as ref
from tests.test_universal_elephantine import _ids, _rand_

F64 = torch.float64


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


def _hymba(**over):
    kw = dict(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=5, num_attention_heads=4,
              num_key_value_heads=2, num_experts=1, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2,
              mamba_dt_rank=8, mamba_inner_layernorms=True, kv_reuse_group=[[1, 2], [3, 4]],
              global_attn_idx=[0], sliding_window=64, num_memory_tokens=6, memory_tokens_interspersed_every=0,
              attn_implementation_new="sdpa", rope=True, rope_type="default", kq_norm="none",
              attn_hidden_size=-1, kq_head_dim=-1, v_head_dim=32, mlp_hidden_act="silu", seq_length=64,
              tie_word_embeddings=False)
    kw.update(over)
    torch.manual_seed(0)
    return _rand_(ref.HymbaForCausalLM(ref.HymbaConfig(**kw)).to(F64).eval())


def _moved(t, ids, r, edit):
    from elephantine.universal import from_hymba
    h = from_hymba(t)
    edit(h)
    return _rel(r, h(ids))


@torch.no_grad()
def test_same_host_carries_hymba_parallel_kv_sharing_meta_tokens_notremote():
    """Hymba (SDPA path: plain causal over the meta tokens + the sequence) vs from_hymba at fp64: attention and a
    Mamba-1 SSM in parallel on one input, each output RMS-normed and averaged into one out_proj; RMS-normed
    dt / B / C; cross-layer KV sharing (layers 2 and 4 read 1's and 3's post-RoPE k / v); 6 learned meta tokens.
    The reference lifts the remote code's fp32 islands, so the bound is fp64 rounding. Each branch switched off
    on the host -- the KV register, the meta tokens, the attention half, the SSM half -- moves the logits."""
    from elephantine.universal import from_hymba
    t = _hymba()
    assert [l.mamba.reuse_kv for l in t.model.layers] == [False, False, True, False, True]
    ids = _ids(97)
    r = t(ids)
    d = _rel(r, from_hymba(t)(ids))
    assert d < 1e-12, d

    def no_reuse(h):
        for l in h.layers:
            l.attn.kv_read.zero_()
    assert _moved(t, ids, r, no_reuse) > 1e-3                                        # KV sharing
    assert _moved(t, ids, r, lambda h: h.prefix.zero_()) > 1e-3                      # meta tokens
    for part in ("attn", "ssm"):                                                     # both parallel halves
        def off(h, part=part):
            for l in h.layers:
                (l.attn.o_proj if part == "attn" else l.ssm.out_proj).weight.zero_()
        assert _moved(t, ids, r, off) > 1e-3, part


@torch.no_grad()
def test_same_host_carries_hymba_window_with_meta_token_sinks_notremote():
    """Hymba's flex-attention block mask on non-global layers: causal AND (t - s <= window OR s is a meta token).
    Window 3 over 6 meta tokens + 12 tokens, so both the window and the sinks bind; layer 0 is global. fp64 vs
    from_hymba; dropping the window or the sinks on the host moves the logits."""
    from elephantine.universal import from_hymba
    t = _hymba(attn_implementation_new="flex", sliding_window=3, global_attn_idx=[0], seq_length=12)
    ids = _ids(97, T=12)
    r = t(ids)
    host = from_hymba(t)
    assert any(l.attn.window < 12 for l in host.layers) and host.layers[1].attn.n_sink == 6
    d = _rel(r, host(ids))
    assert d < 1e-12, d

    def no_window(h):
        for l in h.layers:
            l.attn.window.fill_(float("inf"))

    def no_sinks(h):
        for l in h.layers:
            l.attn.n_sink.fill_(0.0)
    assert _moved(t, ids, r, no_window) > 1e-3
    assert _moved(t, ids, r, no_sinks) > 1e-3
