"""Nemotron-Flash's two SSM mixers, checked against the LIBRARY modules its remote files copy
(no Hub code is executed; the remote files were only read and diffed as text).

Source repo nvidia/Nemotron-Flash-1B, snapshot da87f707f678b5936e67fdcfe598f76ebcc5fdbc:
  delta_net.py  sha256 56404909ba965ed62c2dab032b86139cc28bdcde2c539a7453dd934df3ddf07f
  mamba2.py     sha256 dcb3e1bfa0696e716bd10e2d56b0cc93f5b4b964892352add14bef6c2bf0d064
Library files they were diffed against:
  flash_linear_attention-0.4.0/fla/layers/delta_net.py  sha256 f503aedf...f3df5
  mamba_ssm-2.3.2.post1/mamba_ssm/modules/mamba2.py      sha256 605e4439...b08707
(the per-test docstrings give the diff result).
"""
import os
import sys

import pytest
import torch

from tests.test_universal_elephantine import _fla_layers, _upstream_kernels, _x

GPU_TOL = 1e-5          # fp32 Triton kernels on the reference side; relative to max|ref|


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1e-30)


def _randomize(layer, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return layer


# --------------------------------------------------------------------------------------------
# DeltaNet
# --------------------------------------------------------------------------------------------

def _deltanet(L, **kw):
    args = dict(hidden_size=32, num_heads=2, use_short_conv=True, conv_size=4, layer_idx=0)
    args.update(kw)
    return L.DeltaNet(**args).cuda().eval()


def test_ssm_block_carries_nemotron_flash_deltanet_notremote():
    """Nemotron-Flash's delta_net.py vs fla 0.4.0's fla/layers/delta_net.py (`diff` of the two
    texts). Every difference is off this forward (attention_mask=None, no cache, T=10 <= 64 so
    both take mode='fused_recurrent', default conv_bias=False):
      * imports (remote adds typing/transformers, drops `warnings` and fla.layers.utils); the
        remote appends its own `Cache` class (lines 319-471), unused without a cache;
      * __init__: extra unused `config=None` arg; `self.silu = nn.SiLU()` (no parameters);
        ShortConvolution built without `bias=conv_bias` -- ShortConvolution's own default is
        bias=False, the same as conv_bias's default; `raise UserWarning` instead of
        `warnings.warn` when use_short_conv=False (only reached in that config, see below);
        an xavier init pass (all parameters are overwritten here);
      * forward: the library's varlen unpad/pad on attention_mask vs the remote's conv `mask=` and
        beta*mask (both no-ops with mask None); the remote passes `seq_idx=position_ids` (None) to
        ShortConvolution, the library `cu_seqlens=None`; F.silu vs nn.SiLU; `beta.sigmoid()` split
        in two lines; `q.new_ones(...)` vs `ones_like(q[..., 0])` (use_beta=False only);
        `offset=q.shape[1]` vs `q_len` (cache only).
    So the forward used is identical and the library class runs as the reference, with the old
    test's constructor args. Real Triton kernels in fp32 on one GPU -> 1e-5 relative.

    Switched off: the short conv (use_short_conv=False; the remote file can't even build that
    config, it raises, so this checks the host's no-conv path against the library) and the
    negative-eigenvalue beta (allow_neg_eigval=True, beta x2). Each variant moves the output
    > 1e-3 relative from the base layer with the same shared weights, and the host follows it."""
    from elephantine.universal import deltanet_block
    L = _fla_layers()
    x = _x(32)
    xg = x.float().cuda()

    def run(layer):
        with torch.no_grad():
            ref = layer(xg)[0].double().cpu()
            ours = deltanet_block(layer)(x)
        return ref, ours

    base = _randomize(_deltanet(L))
    ref, ours = run(base)
    gap = _rel(ref, ours)
    assert gap < GPU_TOL, gap

    for kw in (dict(use_short_conv=False), dict(allow_neg_eigval=True)):
        var = _deltanet(L, **kw)
        missing, _ = var.load_state_dict(base.state_dict(), strict=False)   # same weights (minus the convs)
        assert not missing, missing
        vref, vours = run(var)
        assert _rel(vref, vours) < GPU_TOL, (kw, _rel(vref, vours))
        assert _rel(ref, vref) > 1e-3, (kw, _rel(ref, vref))                # the branch matters
        assert _rel(ref, vours) > 1e-3, (kw, _rel(ref, vours))


# --------------------------------------------------------------------------------------------
# Mamba-2
# --------------------------------------------------------------------------------------------

def _mamba2_module():
    """mamba_ssm 2.3.2's modules/mamba2.py from its source tarball (the `_upstream_kernels` stubs
    for the CUDA extensions; the conv routed to the upstream causal_conv1d_ref, as the old test
    did for the remote copy)."""
    import importlib, importlib.machinery as im, types
    _upstream_kernels()
    import causal_conv1d.causal_conv1d_interface as ci
    sys.modules["causal_conv1d"].causal_conv1d_fn = ci.causal_conv1d_ref
    sys.modules["causal_conv1d"].causal_conv1d_update = ci.causal_conv1d_update_ref
    for name in ("mamba_ssm.modules", "mamba_ssm.distributed"):        # package stubs, as for .ops
        if name not in sys.modules:
            path = os.path.join(sys.modules["mamba_ssm"].__path__[0], name.split(".")[1])
            mod = types.ModuleType(name); mod.__path__ = [path]
            mod.__spec__ = im.ModuleSpec(name, None, is_package=True); mod.__spec__.submodule_search_locations = [path]
            sys.modules[name] = mod
    m2 = importlib.import_module("mamba_ssm.modules.mamba2")
    m2.causal_conv1d_fn = ci.causal_conv1d_ref
    return m2


def test_ssm_block_carries_nemotron_flash_mamba2_notremote():
    """Nemotron-Flash's mamba2.py vs mamba_ssm 2.3.2.post1's mamba_ssm/modules/mamba2.py (`diff`
    of the two texts). Differences, none on this forward (use_mem_eff_path=False,
    inference_params=None, D_has_hdim=False, ngroups=1, the conv through causal_conv1d_fn):
      * constructor: the remote reads d_model / d_state / d_conv / expand / headdim from a config
        (hidden_size, mamba_d_state, mamba_d_conv, mamba_expand, mamba2_headdim) -- passed here
        directly with the old test's values -- defaults use_mem_eff_path=False (passed
        explicitly), asserts nheads % 8 == 0 (holds: 8 heads), drops PyTorchModelHubMixin;
      * forward: extra attention_mask / past_key_value args, returns (out, past_key_value)
        instead of out; `seq_idx` not passed to causal_conv1d_fn (None here); the scan gets
        `D=self.D` instead of the D_has_hdim rearrange (same with D_has_hdim=False); the
        causal_conv1d_fn-is-None fallback slices `[:, -(self.dconv - 1):]` (a typo'd attribute
        and a different slice) -- not taken, the conv goes through causal_conv1d_fn as in the
        old test; `y_full, z_full` aliases before the norm;
      * step(): a multi-token decode path (inference only).
    So the library class runs as the reference with the old test's arguments: real Triton SSD
    scan and gated RMSNorm in fp32 on one GPU (TF32 off) -> 1e-5 relative.

    Switched off: the D skip (D = 0 on both sides) and the norm's z gate (the library norm called
    with z=None; on the host the gate made the constant 1). Each moves the output > 1e-3 and the
    host with the same switch agrees with the library."""
    from elephantine.universal import mamba2_block, mamba_ssm_mamba2_shim
    if not torch.cuda.is_available():
        pytest.skip("Triton kernels need a GPU")
    os.environ["TRITON_F32_DEFAULT"] = "ieee"   # mamba_ssm's tl.dot defaults to TF32 (~1e-3) on Ampere
    m2 = _mamba2_module()
    mx = m2.Mamba2(d_model=32, d_state=16, d_conv=4, expand=2, headdim=8, ngroups=1, rmsnorm=True,
                   norm_before_gate=False, use_mem_eff_path=False, chunk_size=16, layer_idx=0).cuda().eval()
    _randomize(mx)
    x = _x(32)
    xg = x.float().cuda()

    def ref_run():
        with torch.no_grad():
            return mx(xg).double().cpu()

    with torch.no_grad():
        ref = ref_run()
        host = mamba2_block(mamba_ssm_mamba2_shim(mx))
        gap = _rel(ref, host(x))
        assert gap < GPU_TOL, gap

        # D skip off
        D0 = mx.D.detach().clone()
        mx.D.zero_()
        ref_d = ref_run()
        host_d = mamba2_block(mamba_ssm_mamba2_shim(mx))
        assert _rel(ref_d, host_d(x)) < GPU_TOL, _rel(ref_d, host_d(x))
        assert _rel(ref, ref_d) > 1e-3 and _rel(ref, host_d(x)) > 1e-3
        mx.D.copy_(D0)

        # z gate of the gated RMSNorm off
        norm_fwd = mx.norm.forward
        mx.norm.forward = lambda y, z=None: norm_fwd(y, None)
        try:
            ref_g = ref_run()
        finally:
            del mx.norm.forward
        host.gate.zero_(); host.gate[4] = 1                                 # constant gate 1
        assert _rel(ref_g, host(x)) < GPU_TOL, _rel(ref_g, host(x))
        assert _rel(ref, ref_g) > 1e-3 and _rel(ref, host(x)) > 1e-3
