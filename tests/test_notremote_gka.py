"""Gated KalmaNet (awslabs hybrid-model-factory) without running hybrid-model-factory's code: the host against
the REFERENCE transcription in tests/refs/gka.py (read from the source tree as text; file, commit, sha256 and
line ranges in its header), on CPU at fp64.

The reference has two transcribed evaluation paths: the token-by-token decode recurrence (torch_decoding_one_step
and its inference kernel) and the chunked prefill (gka_chebyshev_gla: the chunked Chebyshev kernel and chunked
simple GLA). They replace the old tests' Triton prefill-vs-decode-kernel comparison.

Run:  pytest tests/test_notremote_gka.py -q
"""
import pytest
import torch

from tests.refs import gka as R

F64 = torch.float64


@pytest.fixture(autouse=True)
def _one_thread():
    """The references are loops of tiny fp64 ops: one thread is ~100x faster than an oversubscribed pool."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1.0)


def _block():
    torch.manual_seed(0)
    layer = R.GatedKalmaNet(hidden_size=32, head_dim=16, num_q_heads=4, num_k_heads=2, num_v_heads=2, layer_idx=0)
    with torch.no_grad():
        for n, p in layer.named_parameters():
            if n not in ("A_log", "dt_bias"):
                p.copy_(torch.randn_like(p) * 0.3 + (1.0 if n.endswith("norm.weight") else 0.0))
    torch.manual_seed(3)
    return layer.double().eval(), torch.randn(2, 80, 32, dtype=F64)


def test_ssm_block_carries_gated_kalmanet_notremote():
    """Gated KalmaNet (amazon/GKA-primed-HQwen3-8B-Instruct's mixer) vs the host's ridge-readout branch: a
    key-covariance state H_t and a fixed 30-step Chebyshev solve of (H_t + 0.02 ||H_t||_F I) x = q_t feeding
    the GLA query. Two 64-token chunks (T = 80) exercise the chunk-boundary state passing.

    Bound: 1e-12 relative (gap / max(|ref|, 1)). The fixed-step Chebyshev solve is a fixed polynomial in H_t
    (no data-dependent stopping), so the only amplification is its conditioning, at most
    (1 + 0.02) / 0.02 ~ 51 on a well-scaled H_t; measured gaps are ~5e-14."""
    from elephantine.universal import gka_block
    layer, x = _block()
    with torch.no_grad():
        rec = layer(x, mode="recurrent")[0]
        chk = layer(x, mode="chunk")[0]
        assert _rel(rec, chk) < 1e-12, _rel(rec, chk)           # decode recurrence == chunked prefill
        host = gka_block(layer)
        ours = host(x)
        assert _rel(rec, ours) < 1e-12, _rel(rec, ours)
        assert _rel(chk, ours) < 1e-12, _rel(chk, ours)
        # ridge readout off: the host moves, and the reference without the solve (num_iter = 0 skips it in
        # gka_chebyshev_gla / torch_decoding_one_step) agrees with the switched-off host
        host.ridge_c.zero_()
        off = host(x)
        assert _rel(rec, off) > 1e-1, _rel(rec, off)
        layer.num_iter = 0
        assert _rel(layer(x, mode="recurrent")[0], off) < 1e-12
        assert _rel(layer(x, mode="chunk")[0], off) < 1e-12


def test_same_host_carries_gka_hybrid_qwen3_notremote():
    """GKA-primed HQwen3 (model_type hybrid_qwen3): the REFERENCE HybridQwen3ForCausalLM with the
    hybrid_override_pattern '*-GKA-GKA-*' (Qwen3 attention layers and Gated KalmaNet layers, the HF
    checkpoint's gka_config), whole-model logits vs the host at fp64; bound 1e-12 relative as for the block."""
    from elephantine.universal import from_hybrid_qwen3
    gka = dict(bp_lambda=True, chunk_size=64, conv_size=4, gla_rescale=True, norm_eps=1e-6, num_iter=30,
               ridge_strength=0.02, solver_type="chebyshev", use_alpha_connection=True, use_beta_gate=True,
               use_forgetting_gate=True, use_forgetting_gate_kk=True, use_gate=True, use_v_conv=True)
    cfg = R.HybridQwen3Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                              rope_parameters={"rope_type": "default", "rope_theta": 5000000.0}, gka_config=gka,
                              hybrid_override_pattern="*-GKA-GKA-*")
    torch.manual_seed(0)
    t = R.HybridQwen3ForCausalLM(cfg).eval()
    assert [type(l).__name__ for l in t.model.layers] == ["Qwen3DecoderLayer", "Qwen3GKADecoderLayer",
                                                          "Qwen3GKADecoderLayer", "Qwen3DecoderLayer"]
    with torch.no_grad():
        for n, p in t.named_parameters():
            if not n.endswith(("A_log", "dt_bias")):
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if n.endswith("norm.weight") else 0.0))
        t.double()
        torch.manual_seed(1)
        x = torch.randint(0, 97, (2, 80))
        ref = t(x).logits
        assert _rel(ref, t(x, mode="chunk").logits) < 1e-12
        host = from_hybrid_qwen3(t)
        ours = host(x)
        assert _rel(ref, ours) < 1e-12, _rel(ref, ours)
        for ul, tl in zip(host.layers, t.model.layers):              # the ridge readout matters in the model
            if hasattr(tl, "gka"):
                ul.ssm.ridge_c.zero_()
        off = host(x)
        assert _rel(ref, off) > 1e-3, _rel(ref, off)
        for tl in t.model.layers:
            if hasattr(tl, "gka"):
                tl.gka.num_iter = 0
        assert _rel(t(x).logits, off) < 1e-12
