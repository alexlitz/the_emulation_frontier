"""RWKV-5 / -6 / -7 time-mix blocks on the host's SSM block, against labelled REFERENCE transcriptions of the
remote modelling code (tests/refs/rwkv.py: read as text, never run; no transformers release has these classes).
The remote-code versions of these tests (tests/test_universal_elephantine.py test_ssm_block_carries_rwkv5 /
_rwkv6_ddlerp / _rwkv7_with_v_first) stay remote-gated. fp64 throughout; each family branch switched off on
both sides still agrees, and moves the output.

Run:  pytest tests/test_notremote_rwkv.py -q
"""
import torch

from tests.refs import rwkv as R

F64 = torch.float64


def _init_(module, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return module


def _x(d, T=10, seed=3):
    torch.manual_seed(seed)
    return torch.randn(2, T, d, dtype=F64)


def _rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1.0)


@torch.no_grad()
def test_ssm_block_carries_rwkv5_notremote():
    from elephantine.universal import rwkv5_block
    cfg = R.Rwkv5Config(hidden_size=32, head_size=8, num_hidden_layers=1, attention_hidden_size=32)
    m = _init_(R.Rwkv5SelfAttention(cfg, 0).to(F64).eval())
    x = _x(32)
    ref = m(x)[0]
    assert _rel(ref, rwkv5_block(m)(x)) < 1e-12, _rel(ref, rwkv5_block(m)(x))
    m.time_faaaa.zero_()                                                  # the current-token bonus u off
    off = m(x)[0]
    assert _rel(ref, off) > 1e-2
    assert _rel(off, rwkv5_block(m)(x)) < 1e-12


@torch.no_grad()
def test_ssm_block_carries_rwkv6_ddlerp_notremote():
    from elephantine.universal import rwkv6_block
    cfg = R.Rwkv6Config(hidden_size=32, attention_hidden_size=32, head_size=8, num_hidden_layers=1)
    m = _init_(R.Rwkv6SelfAttention(cfg, 0).to(F64).eval())
    x = _x(32)
    ref = m(x)[0]
    assert _rel(ref, rwkv6_block(m)(x)) < 1e-12, _rel(ref, rwkv6_block(m)(x))
    for p in ("time_maa_w2", "time_decay_w2"):                            # data-dependent lerp, then decay, off
        getattr(m, p).zero_()
        off = m(x)[0]
        assert _rel(ref, off) > 1e-2, p
        assert _rel(off, rwkv6_block(m)(x)) < 1e-12, p
        ref = off


@torch.no_grad()
def test_ssm_block_carries_rwkv7_with_v_first_notremote():
    from types import SimpleNamespace
    from elephantine.universal import rwkv7_block
    cfg = SimpleNamespace(hidden_size=32, head_dim=8, num_heads=4, decay_low_rank_dim=8, a_low_rank_dim=8,
                          gate_low_rank_dim=8, v_low_rank_dim=8, intermediate_size=64, norm_eps=1e-5,
                          norm_bias=True, wkv_implementation="eager", wkv_state_dtype="float64")
    a0, a1 = (_init_(R.Rwkv7Attention(cfg, i).to(F64).eval(), seed=i) for i in (0, 1))
    x0, x1 = _x(32, seed=3), _x(32, seed=4)
    y0, v_first = a0(x0)
    y1, _ = a1(x1, v_first)
    reg = {}
    b0, b1 = rwkv7_block(a0), rwkv7_block(a1)
    h0, h1 = b0(x0, reg), b1(x1, reg)
    assert _rel(y0, h0) < 1e-12, _rel(y0, h0)
    assert _rel(y1, h1) < 1e-12, _rel(y1, h1)                             # v_first via the register
    # v_first mixing off: reference v_gate = sigmoid(-1e300) = 0 exactly; host: layer 1 stops reading the register
    a1.v0.fill_(-1e300)
    off, _ = a1(x1, v_first)
    assert _rel(y1, off) > 1e-2
    b1.v_read.zero_()
    reg = {}
    b0(x0, reg)
    assert _rel(off, b1(x1, reg)) < 1e-12
