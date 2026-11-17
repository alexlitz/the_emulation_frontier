"""PLaMo-2 on the universal host, against a labelled transcription of its remote code (tests/refs/plamo2.py,
read as text, never executed): the remote-code-free counterpart of
tests/test_universal_elephantine.py::test_same_host_carries_plamo2.

Run:  pytest tests/test_notremote_plamo2.py -q
"""
import torch

from tests.refs import plamo2 as ref
from tests.test_universal_elephantine import _ids, _rand_

F64 = torch.float64


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


@torch.no_grad()
def test_same_host_carries_plamo2_notremote():
    """PLaMo-2 vs from_plamo2 at fp64: Mamba-2 layers with RMS-normed B / C / dt (the normalized-feature branch)
    alternating with per-head qk-normed GQA attention, sandwich norms with offset weights, a vocab-padded head.
    Layer 1 slides (window 3 < T = 10, local RoPE base) and layer 3 is full attention. The reference lifts the
    remote code's fp32 islands (RMS norms, SSM state, RoPE tables), so the bound is fp64 rounding. Switching
    each family branch off on the host (post norms, qk norms, the window) or in the reference (the B / C / dt
    norms) moves the logits."""
    from elephantine.universal import from_plamo2
    cfg = ref.Plamo2Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
                           hidden_size_per_head=16, attention_window_size=3, full_attention_idx=[3],
                           rope_theta=10000, rope_local_theta=500, mamba_d_state=16, mamba_d_conv=4,
                           mamba_num_heads=4, mamba_step=2, intermediate_size=96, vocab_size=97,
                           tie_word_embeddings=False)
    torch.manual_seed(0)
    t = _rand_(ref.Plamo2ForCausalLM(cfg).to(F64).eval())
    L = t.model.layers.layers
    assert [l.is_mamba for l in L] == [True, False, True, False] and [L[1].mixer.full_attn, L[3].mixer.full_attn] == [False, True]
    ids = _ids(97)
    r = t(ids)
    u = from_plamo2(t)
    d = _rel(r, u(ids)[..., : r.shape[-1]])                              # the host's embed is vocab-padded
    assert d < 1e-12, d

    def moved(edit):
        h = from_plamo2(t)
        for ul, tl in zip(h.layers, L):
            edit(ul, tl)
        return _rel(r, h(ids)[..., : r.shape[-1]])

    off = torch.tensor([1.0, 0.0], dtype=F64)
    assert moved(lambda ul, tl: ul.post_coef.copy_(off.expand(3, 2))) > 1e-3            # sandwich post norms
    assert moved(lambda ul, tl: tl.is_mamba or (ul.attn.q_coef.copy_(off), ul.attn.k_coef.copy_(off))) > 1e-3
    assert moved(lambda ul, tl: ul.attn.window.fill_(float("inf"))) > 1e-4               # the window binds
    for l in L:
        if l.is_mamba:
            l.mixer.bcdt_norm = False
    assert _rel(t(ids), u(ids)[..., : r.shape[-1]]) > 1e-3                                # B / C / dt norms
