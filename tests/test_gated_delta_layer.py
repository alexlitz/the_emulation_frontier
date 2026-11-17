"""The gated-delta layer is a RUNNABLE module, and it IS the real library recurrence.

Closes the last construction-only gap: the 420 non-Mamba linear recurrences in the zoo
(gated-DeltaNet / RWKV-7 / MiniMax-lightning / LFM2 / RG-LRU). `attn/ssm_cross.py` proved
the general gated-delta rule subsumes every family as recurrence *math*; here the same rule
runs as an `nn.Module` inside an end-to-end LM, and:

  * it reproduces transformers' OWN shipped gated-delta kernels (`torch_recurrent_gated_delta_rule`
    and `torch_chunk_gated_delta_rule`, the Qwen3-Next reference forms) to within their own
    sequential-vs-chunked fp32 gap -- so it is the real operator, not a look-alike;
  * a WIDER host layer reproduces a NARROWER guest layer of EVERY family token-for-token
    (state-inflation, run as a module forward), and a wider host LM reproduces a guest LM.
"""
import pytest

torch = pytest.importorskip("torch")

from attn.gated_delta_layer import (FAMILIES, GatedDeltaLayer, GatedDeltaLM, gd_recurrence,
                                     inflate, inflate_lm)

F64 = torch.float64


def test_general_rule_is_the_real_transformers_gated_delta_kernel():
    # gd_recurrence == transformers' torch_recurrent_gated_delta_rule AND torch_chunk_gated_delta_rule
    # (the Qwen3-Next reference forms), to within transformers' own seq-vs-chunk fp32 gap.
    q3 = pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")
    torch.manual_seed(0)
    B, H, T, dk, dv = 1, 3, 10, 16, 16
    q = torch.randn(B, T, H, dk, dtype=F64); k = torch.randn(B, T, H, dk, dtype=F64)
    v = torch.randn(B, T, H, dv, dtype=F64)
    g = -torch.rand(B, T, H, dtype=F64); beta = torch.rand(B, T, H, dtype=F64)
    scale = 1 / dk ** 0.5
    qn, kn = q3.l2norm(q, dim=-1, eps=1e-6), q3.l2norm(k, dim=-1, eps=1e-6)   # the kernel l2-norms q,k
    a = g.exp()[..., None].expand(B, T, H, dk).permute(0, 2, 1, 3).contiguous()  # scalar decay -> per-chan
    mine = gd_recurrence((qn * scale).transpose(1, 2), kn.transpose(1, 2), v.transpose(1, 2),
                         a, beta.transpose(1, 2), beta.transpose(1, 2)).transpose(1, 2)
    rseq, _ = q3.torch_recurrent_gated_delta_rule(q, k, v, g, beta, None, False, use_qk_l2norm_in_kernel=True)
    rchk, _ = q3.torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=4, initial_state=None,
                                              output_final_state=False, use_qk_l2norm_in_kernel=True)
    lib_gap = (rseq - rchk).abs().max().item()                       # transformers' own two impls differ this much
    d_seq = (rseq - mine).abs().max().item()
    d_chk = (rchk - mine).abs().max().item()
    assert d_seq <= max(lib_gap, 1e-6) and d_chk <= max(lib_gap, 1e-6), (d_seq, d_chk, lib_gap)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_wider_host_layer_carries_every_family_exactly(family):
    # state-inflation as a MODULE forward: a host with a wider per-head state reproduces the
    # guest layer of this family token-for-token (extra state dims stay 0 -> inert).
    torch.manual_seed(0)
    guest = GatedDeltaLayer(16, 4, 6, family).eval()
    host = inflate(guest, head_dim=11).eval()
    x = torch.randn(2, 9, 16, dtype=F64)
    with torch.no_grad():
        d = (guest(x) - host(x)).abs().max().item()
    assert d < 1e-12, f"{family}: wider host must reproduce the guest layer (max|Δ|={d:.2e})"


@pytest.mark.parametrize("family", ["gated_deltanet", "gla", "deltanet"])
def test_wider_host_LM_reproduces_guest_LM_forward(family):
    # end-to-end: a wider host LM (bigger per-head state + wider FFN, same d_model) runs the
    # guest LM's compressed recurrence token-for-token -- the elephantine recipe for the SSM island.
    torch.manual_seed(1)
    guest = GatedDeltaLM(40, 16, 3, 4, 6, family, d_ff=32).eval()
    host = inflate_lm(guest, head_dim=10, d_ff=48).eval()
    ids = torch.randint(0, 40, (2, 9))
    with torch.no_grad():
        d = (guest(ids) - host(ids)).abs().max().item()
    assert d < 1e-12, f"{family}: wider host LM must reproduce the guest LM (max|Δ logits|={d:.2e})"


def test_gated_deltanet_is_the_top_that_carries_the_rest():
    # the general rule (gated_deltanet knobs) folds the whole family lattice: run one host LM's
    # forward and confirm it is non-trivial, so the carry above is load-bearing, not vacuous.
    torch.manual_seed(2)
    m = GatedDeltaLM(40, 16, 2, 4, 6, "gated_deltanet", d_ff=32).eval()
    ids = torch.randint(0, 40, (1, 8))
    with torch.no_grad():
        base = m(ids).clone()
        for b in m.blocks:                                           # kill the recurrence -> logits move
            b["mix"].o_proj.weight.zero_()
        changed = (base - m(ids)).abs().max().item()
    assert changed > 1e-3, "the gated-delta recurrence must be load-bearing"
