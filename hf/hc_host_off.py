"""Hyper-connection hosts switched OFF: a host with n mixed residual streams plays a plain (one-stream) target exactly.

The construction (Lean EP.hc_identical_streams / EP.hc_stack, proofs/lean/EmulationProofs/ResNorm.lean):
every stream carries the SAME vector, the target's residual. The embedding already enters as n copies (every stock
HC model expands it that way). With the dynamic projection zero (fn / hc_fn / input_mix_weight_* / block_inject_weight
= 0) every mixing weight is its static base, and the bases are set so that

  pre   sum_i pre_i = 1      the sublayer reads exactly the stream                (sigmoid(b) = 1/n, minus hc_eps)
  post  post_i = 1           every stream gets the sublayer output once           (sigmoid(b) = 1/2 at magnitude 2)
  comb  rows (or columns, for comb^T) sum to 1  identical streams stay identical  (constant logits -> 1/n)
  head  sum_i pre_i = 1      the head reads the stream                            (DeepSeek-V4 / Hy4; GLM-5-Next: mean)

so after every layer all n streams equal the target's residual. Per stock variant (transformers 5.18 code):

  dsv4 / glm5_next  pre = sigmoid + hc_eps, post = 2 sigmoid, comb = Sinkhorn(softmax + hc_eps) with hc_eps in every
                    normaliser: identical entries stay identical, but the rows sum to n c / (n c + hc_eps) < 1, so
                    the streams shrink by a fixed factor per layer unless hc_eps = 0 -- a shape-preserving config
                    value (free). With hc_eps = 0 the Sinkhorn of a constant matrix is exactly 1/n everywhere.
  hy4               pre = sigmoid + hc_eps, post = hc_magnitude sigmoid + hc_eps, no comb (the identity): exact at any
                    hc_eps < 1/n (sigmoid(b) = 1/n - hc_eps, sigmoid(b') = (1 - hc_eps) / hc_magnitude).
  qwen4_exp         the sublayer input is mean_i(sigmoid(up(silu(down(N(s))))) * N_group(s_i)) with N_group the
                    per-stream (1 + w) RMSNorm: up = 0 gives 1/2 N(x) (1 + w), so (1 + w) = 2 gamma_target; the
                    injection 2 sigmoid(block_inject(.) / n) = 1 at block_inject = 0; the final mixer is the final norm.

This module checks the identity on the stock modules themselves (fp64, random target data): streams in, streams out,
max |host - target|. Run under transformers 5.18:

  python -m hf.hc_host_off            # (re-execs under ~/.cache/upstream_refs/transformers-5.18.0)
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

F64 = torch.float64
TF5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")


def logit(p):
    return math.log(p / (1 - p))


def set_static(mod, n, pre=None, post=None, comb=True, hc_eps=0.0, magnitude=2.0):
    """DeepSeek-V4 / GLM-5-Next / Hy4 HyperConnection: fn = 0 and the bases of the construction."""
    with torch.no_grad():
        mod.fn.zero_()
        base = torch.zeros_like(mod.base)
        base[:n] = logit(1.0 / n - hc_eps)                       # pre_i = 1/n
        base[n:2 * n] = logit((1.0 - (hc_eps if magnitude != 2.0 or not comb else 0.0)) / magnitude)
        if comb:
            base[2 * n:] = 0.0                                    # constant comb logits
        mod.base.copy_(base)
        mod.scale.fill_(1.0)


def check_dsv4_like(kind, n=4, d=16, T=5, seed=0):
    torch.manual_seed(seed)
    if kind == "dsv4":
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection as HC, \
            DeepseekV4HyperHead as Head
        from transformers import DeepseekV4Config as C
    else:
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection as HC, \
            Glm5NextTextHyperHead as Head
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig as C
    out = {}
    for hc_eps in (0.0, 1e-6):
        cfg = C(hidden_size=d, hc_mult=n, hc_eps=hc_eps, hc_sinkhorn_iters=20, rms_norm_eps=1e-6)
        hc = HC(cfg).to(F64)
        with torch.no_grad():
            hc.fn.copy_(torch.randn_like(hc.fn))
            hc.base.copy_(torch.randn_like(hc.base))
            hc.scale.copy_(torch.randn_like(hc.scale))
        with torch.no_grad():
            hc.fn.zero_()
            base = torch.zeros_like(hc.base)
            base[:n] = logit(1.0 / n - hc_eps)
            base[n:2 * n] = 0.0                                   # post = 2 sigmoid(0) = 1
            hc.base.copy_(base)
            hc.scale.fill_(1.0)
        x = torch.randn(2, T, d, dtype=F64)
        f = torch.randn(2, T, d, dtype=F64)                       # a sublayer output
        s = x.unsqueeze(2).expand(-1, -1, n, -1).contiguous()
        post, comb, collapsed = hc(s)
        if kind == "dsv4":                                         # DeepseekV4DecoderLayer: In @ comb + post * f
            new = post.unsqueeze(-1) * f.unsqueeze(-2) + torch.matmul(comb, s)
        else:                                                      # Glm5NextTextDecoderLayer: comb^T @ In
            new = post.unsqueeze(-1) * f.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), s)
        err_in = (collapsed - x).abs().max().item()
        err_out = (new - (x + f).unsqueeze(2)).abs().max().item()
        head = Head(cfg).to(F64) if kind == "dsv4" else Head()
        if kind == "dsv4":
            with torch.no_grad():
                head.hc_fn.zero_()
                head.hc_base.fill_(logit(1.0 / n - hc_eps))
                head.hc_scale.fill_(1.0)
        err_head = (head(new) - (x + f)).abs().max().item()
        out[f"hc_eps={hc_eps:g}"] = dict(collapse=err_in, layer=err_out, head=err_head,
                                         comb_rowsum_min=float(comb.sum(-1).min()), comb_colsum_min=float(comb.sum(-2).min()))
    return out


def check_hy4(n=4, d=16, T=5, seed=0):
    torch.manual_seed(seed)
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4HyperConnection as HC, HYV4HyperHead as Head
    from transformers import HYV4Config as C
    out = {}
    for hc_eps, mag in ((1e-6, 2.0), (0.0, 2.0), (1e-6, 1.5)):
        cfg = C(hidden_size=d, hc_mult=n, hc_eps=hc_eps, hc_magnitude=mag, rms_norm_eps=1e-6)
        hc = HC(cfg).to(F64)
        with torch.no_grad():
            hc.fn.zero_()
            base = torch.zeros_like(hc.base)
            base[:n] = logit(1.0 / n - hc_eps)
            base[n:] = logit((1.0 - hc_eps) / mag)
            hc.base.copy_(base)
            hc.scale.fill_(1.0)
        x = torch.randn(2, T, d, dtype=F64)
        f = torch.randn(2, T, d, dtype=F64)
        s = x.unsqueeze(2).expand(-1, -1, n, -1).contiguous()
        post, collapsed = hc(s)
        new = post.unsqueeze(-1) * f.unsqueeze(-2) + s              # HYV4DecoderLayer
        head = Head(cfg).to(F64)
        with torch.no_grad():
            head.hc_fn.zero_()
            head.hc_base.fill_(logit(1.0 / n - hc_eps))
            head.hc_scale.fill_(1.0)
        out[f"hc_eps={hc_eps:g},mag={mag:g}"] = dict(collapse=(collapsed - x).abs().max().item(),
                                                     layer=(new - (x + f).unsqueeze(2)).abs().max().item(),
                                                     head=(head(new) - (x + f)).abs().max().item())
    return out


def check_qwen4(n=4, d=16, T=5, seed=0, eps=1e-6):
    torch.manual_seed(seed)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual as GR
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig as C
    cfg = C(hidden_size=d, hc_count=n, hc_lowrank=8, rms_norm_eps=eps)
    gr, mixer = GR(cfg).to(F64), GR(cfg, use_combine=False).to(F64)
    gamma = 1 + 0.1 * torch.randn(d, dtype=F64)                 # the target's pre-norm gain
    for m in (gr, mixer):
        with torch.no_grad():
            m.input_mix_weight_up.weight.zero_()
            m.input_mix_weight_down.weight.copy_(torch.randn_like(m.input_mix_weight_down.weight))
            if m.block_inject_weight is not None:
                m.block_inject_weight.weight.zero_()
            m.hc_norm.weight.copy_((2 * gamma - 1).repeat(n))    # (1 + w) = 2 gamma: the 1/2 gate undone
    x = torch.randn(2, T, d, dtype=F64) * 3
    f = torch.randn(2, T, d, dtype=F64)
    s = x.repeat(1, 1, n)
    mixed, hyper, inj = gr(s)
    N = lambda z: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * gamma
    new = hyper + (f.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)   # Qwen4ExpTextDecoderLayer
    return dict(collapse=(mixed - N(x)).abs().max().item(),
                layer=(new - (x + f).repeat(1, 1, n)).abs().max().item(),
                head=(mixer(new) - N(x + f)).abs().max().item())


def main():
    if not os.environ.get("HC_TF5"):
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prelude = ("import sys, types\n"
                   "m = types.ModuleType('transformers.dependency_versions_check'); m.dep_version_check = lambda *a, **k: None\n"
                   "sys.modules['transformers.dependency_versions_check'] = m\n"
                   "sys.modules['fla'] = None\n"
                   f"sys.argv = {['hc_host_off'] + sys.argv[1:]!r}\n"
                   "import runpy; runpy.run_module('hf.hc_host_off', run_name='__main__')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, root]), HC_TF5="1")
        sys.exit(subprocess.run([sys.executable, "-c", prelude], env=env).returncode)
    import transformers
    res = dict(transformers=transformers.__version__, dsv4=check_dsv4_like("dsv4"),
               glm5_next=check_dsv4_like("glm5_next"), hy4=check_hy4(), qwen4_exp=check_qwen4())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
