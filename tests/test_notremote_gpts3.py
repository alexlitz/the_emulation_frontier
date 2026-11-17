"""GPT-S3 (AxiomicLabs/GPT-S3-3M) without running its remote code: the universal host against tests/refs/gpts3.py,
a labelled fp64 REFERENCE transcription of configuration_gpts3.py + modeling_gpts3.py (read as text, never run).

Run:  pytest tests/test_notremote_gpts3.py -q
"""
import torch
import torch.nn as nn

from tests.refs import gpts3 as R

F64 = torch.float64


def _rand_(model, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 1:
                p.add_(torch.randn_like(p) * 0.05)
    return model


def _ids(vocab, T=12, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (2, T))


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1)


class _Const(nn.Module):
    """A projection replaced by a constant logit: sigmoid(40) == 1.0 exactly in fp64 (the gate switched off)."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.full((*x.shape[:-1], self.n), 40.0, dtype=x.dtype)


def test_same_host_carries_gpts3_gdn2_full_attnres_notremote():
    """GPT-S3: GDN-2 (erase gate on the read key, sigmoid write gate on v, per-channel decay, separate q / k / v
    short convs) + gated GQA (offset qk-norm, interleaved partial RoPE) under Full Attention Residuals (every
    sub-block output a source). Same config / init as the remote-gated test; the reference is fp64 throughout,
    so the bound is 1e-12 relative. The host loader builds its RoPE inverse frequencies at fp32 (to match the
    remote's fp32 table); as the old test's fp64 floor did, they are reset to the fp64 table here."""
    import copy
    from elephantine.universal import from_gpts3
    cfg = R.GPTS3Config(vocab_size=97, hidden_size=32, num_hidden_layers=4, gdn_layers_per_attention=1,
                        gdn_head_dim=16, num_attention_heads=2, num_key_value_heads=1, head_dim=16, rotary_dim=8,
                        intermediate_size=48, max_position_embeddings=128)
    torch.manual_seed(0)
    t = _rand_(R.GPTS3ForCausalLM(cfg).to(F64).eval())
    with torch.no_grad():                                        # AttnRes queries start at 0 (uniform mix)
        for mod in t.modules():
            if isinstance(mod, R.AttnResAggregator):
                mod.query.normal_(0, 0.5)
    assert [b.is_attention for b in t.transformer["h"]] == [False, True, False, True]
    x = _ids(97, T=12)
    inv64 = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2, dtype=F64) / cfg.rotary_dim))

    def host_of(model):
        h = from_gpts3(model)
        with torch.no_grad():
            for l in h.layers:
                l.attn.inv_freq[: cfg.rotary_dim // 2] = inv64
        return h

    with torch.no_grad():
        ref = t(x).logits
        host = host_of(t)
        out = host(x)
        assert _rel(ref, out) < 1e-12, _rel(ref, out)

        # erase gate (b on the read key): off on the host (c_er = 0: erase along k) == the reference with b = 1
        h_off = host_of(t)
        for l in h_off.layers:
            l.ssm.c_er.zero_()
        y_off = h_off(x)
        assert _rel(ref, y_off) > 1e-3, _rel(ref, y_off)
        t_off = copy.deepcopy(t)
        for b in t_off.transformer["h"]:
            if not b.is_attention:
                b.attn.kernel.b_proj = _Const(cfg.hidden_size)
        assert _rel(t_off(x).logits, y_off) < 1e-12, _rel(t_off(x).logits, y_off)

        # write gate (sigmoid w on v): off on the host (v read straight from the conv, not from w * v)
        # == the reference with w = 1
        h_off = host_of(t)
        C = cfg.hidden_size
        for l in h_off.layers:
            s = l.ssm
            s.pv.weight.zero_()
            s.pv.weight[:, s.D + 2 * C: s.D + 3 * C] = torch.eye(C, dtype=F64)
        y_off = h_off(x)
        assert _rel(ref, y_off) > 1e-3, _rel(ref, y_off)
        t_off = copy.deepcopy(t)
        for b in t_off.transformer["h"]:
            if not b.is_attention:
                b.attn.kernel.w_proj = _Const(cfg.hidden_size)
        assert _rel(t_off(x).logits, y_off) < 1e-12, _rel(t_off(x).logits, y_off)

        # Full AttnRes: the learned depth queries matter (zeroed on both sides: uniform mix, still agreeing),
        # and the host with its attnres_full branch off (a plain residual stream) is a different model
        h_q0, t_q0 = host_of(t), copy.deepcopy(t)
        for mod in t_q0.modules():
            if isinstance(mod, R.AttnResAggregator):
                mod.query.zero_()
        for l in h_q0.layers:
            l.ar_qa.zero_(); l.ar_qf.zero_()
        h_q0.ar_q_out.zero_()
        y_q0 = h_q0(x)
        assert _rel(ref, y_q0) > 1e-3, _rel(ref, y_q0)
        assert _rel(t_q0(x).logits, y_q0) < 1e-12, _rel(t_q0(x).logits, y_q0)
        h_off = host_of(t)
        h_off.cfg.attnres_full = False
        assert _rel(ref, h_off(x)) > 1e-3, _rel(ref, h_off(x))
