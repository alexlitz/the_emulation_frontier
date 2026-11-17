"""Mamba-3 MIMO (arianraje/qwen3-4b-mamba3-hybrid's Qwen3Mamba3Mixer) on the host's SSM block, against a labelled
REFERENCE transcription of the remote modelling code (tests/refs/mamba3.py: read as text, never run; neither
transformers nor mamba_ssm 2.3.2 ships Mamba-3). The remote-code version (test_ssm_block_carries_mamba3_mimo in
tests/test_universal_elephantine.py) stays remote-gated. fp64; the data-dependent RoPE angles and the B / C
biases switched off on both sides still agree, and move the output.

Run:  pytest tests/test_notremote_mamba3.py -q
"""
import torch

from tests.refs import mamba3 as M

F64 = torch.float64


def _rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1.0)


@torch.no_grad()
def test_ssm_block_carries_mamba3_mimo_notremote():
    from elephantine.universal import mamba3_block
    cfg = M.Qwen3Mamba3Config(hidden_size=32, mamba_n_heads=4, mamba_d_head=8, mamba_d_state=16,
                              mamba_n_groups=2, mamba_mimo_rank=2, mamba_expand=1, mamba_chunk_size=16,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                              num_hidden_layers=2, layer_types=["linear_attention", "full_attention"])
    m = M.Qwen3Mamba3Mixer(cfg, 0).to(F64).eval()
    torch.manual_seed(0)
    for p in m.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    assert m.num_rope_angles > 0 and m.mimo_rank > 1
    torch.manual_seed(3)
    x = torch.randn(2, 10, 32, dtype=F64)
    ref = m(x)
    assert _rel(ref, mamba3_block(m)(x)) < 1e-12, _rel(ref, mamba3_block(m)(x))
    ang = slice(sum(m.split_sizes[:-1]), None)
    for name, off in (("rope angles", lambda: m.in_proj.weight[ang].zero_()),
                      ("B / C biases", lambda: (m.B_bias.zero_(), m.C_bias.zero_()))):
        off()
        out = m(x)
        assert _rel(ref, out) > 1e-2, name
        assert _rel(out, mamba3_block(m)(x)) < 1e-12, name
        ref = out
