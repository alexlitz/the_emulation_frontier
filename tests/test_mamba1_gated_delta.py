"""Mamba-1 ⊆ gated-delta, checked against transformers' REAL `MambaMixer`.

Mamba-1's selective scan is a gated-delta recurrence with one head per channel c:
key dim = ssm_state_size, value dim = 1, per-key decay a = exp(Δ_c·A_c), write strength
w = Δ_c, k = B_t, q = C_t, and no delta-remove (gamma = 0). The conv, gate and D skip sit
outside the recurrence. So a host whose SSM layer is the general gated-delta rule also carries
the Mamba family, and one SSM primitive covers every autoregressive recurrence in the zoo.

Run:  pytest tests/test_mamba1_gated_delta.py -q
"""
import torch
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer

from attn.gated_delta_layer import gd_recurrence


def _mixer(seed=0):
    torch.manual_seed(seed)
    cfg = MambaConfig(hidden_size=16, state_size=8, expand=2, conv_kernel=4, time_step_rank=4,
                      use_mambapy=False)
    m = MambaMixer(cfg, layer_idx=0).double().eval()
    with torch.no_grad():                    # non-trivial A, D, dt bias
        m.A_log.copy_(torch.randn_like(m.A_log) * 0.5)
        m.D.copy_(torch.randn_like(m.D))
        m.dt_proj.bias.copy_(torch.randn_like(m.dt_proj.bias))
    return m


@torch.no_grad()
def mamba1_via_gated_delta(m, x):
    """The Mamba-1 mixer forward with the selective scan replaced by `gd_recurrence`."""
    T = x.shape[1]
    hs, gate = m.in_proj(x).transpose(1, 2).chunk(2, dim=1)               # (B, D, T)
    hs = m.act(m.conv1d(hs)[..., :T])
    dt, Bm, Cm = torch.split(m.x_proj(hs.transpose(1, 2)),
                             [m.time_step_rank, m.ssm_state_size, m.ssm_state_size], dim=-1)
    delta = torch.nn.functional.softplus(m.dt_proj(dt)).transpose(1, 2)   # (B, D, T)
    # the real slow_forward rounds A, B and the scan input to fp32 even on fp64 inputs;
    # mirror those casts so the comparison isolates the recurrence itself
    A = -torch.exp(m.A_log.float())                                       # (D, N)
    Dn = m.intermediate_size
    a = torch.exp(A[None, :, None, :] * delta[..., None])                 # (B, D, T, N) per-key decay
    k = Bm.float().double()[:, None].expand(-1, Dn, -1, -1)               # (B, D, T, N)
    q = Cm[:, None].expand(-1, Dn, -1, -1)
    v = hs.float().double()[..., None]                                    # (B, D, T, 1)
    y = gd_recurrence(q, k, v, a, torch.zeros_like(delta), delta)[..., 0]  # (B, D, T)
    y = (y + hs * m.D[None, :, None]) * m.act(gate)
    return m.out_proj(y.transpose(1, 2))


def test_gated_delta_reproduces_real_mamba1_mixer():
    m = _mixer()
    torch.manual_seed(1)
    x = torch.randn(2, 12, 16, dtype=torch.float64)
    ref = m.slow_forward(x)
    ours = mamba1_via_gated_delta(m, x)
    assert (ref - ours).abs().max().item() < 1e-15
