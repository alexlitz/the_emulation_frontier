"""The exact qk-norm nestings (graph/model_graph.py QKNORM_NEST, round 2 pass 2) on the stock modules, fp64:
a gained per-head RMSNorm with gain 1 is the weight-free one, before or after the rotary (the rms is
rotation-invariant), and a per-head-distinct gain / a LayerNorm bias reduce to the shared / bias-free forms.
These are the module-level constructions behind the exact qk-norm edges (elephantine/edge_coverage.py
MODULE_LEVEL)."""
import torch
from transformers.models.chameleon.modeling_chameleon import ChameleonLayerNorm
from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1RMSNorm
from transformers.models.llama4.modeling_llama4 import Llama4TextL2Norm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb
from transformers.models.stablelm.modeling_stablelm import StableLmLayerNormPerHead

F64 = torch.float64
H, T, D, EPS = 4, 7, 16, 1e-6
TOL = 1e-6          # the stock RMSNorms compute in fp32 (hidden_states.to(torch.float32)): fp32 noise


def _rot(T, D):
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=F64) / D))
    ang = torch.arange(T, dtype=F64)[:, None] * inv[None]
    emb = torch.cat((ang, ang), -1)
    return emb.cos()[None], emb.sin()[None]


def test_weight_free_after_rope_is_gained_before_rope_at_gain_one():
    torch.manual_seed(0)
    q, k = torch.randn(1, H, T, D, dtype=F64), torch.randn(1, H, T, D, dtype=F64)
    cos, sin = _rot(T, D)
    l2 = Llama4TextL2Norm(EPS)                                         # Llama-4: weight-free, after the rotary
    tq, tk = apply_rotary_pos_emb(q, k, cos, sin)
    tq, tk = l2(tq), l2(tk)
    qn = Qwen3RMSNorm(D, eps=EPS).to(F64)                              # Qwen3: gained, before the rotary
    qn.weight.data.fill_(1.0)
    hq, hk = apply_rotary_pos_emb(qn(q), qn(k), cos, sin)
    assert (tq - hq).abs().max() < TOL and (tk - hk).abs().max() < TOL
    hy = HunYuanDenseV1RMSNorm(D, eps=EPS).to(F64)                     # HunYuan: gained, after the rotary
    hy.weight.data.fill_(1.0)
    pq, pk = apply_rotary_pos_emb(q, k, cos, sin)
    assert (hy(pq) - tq).abs().max() < TOL and (hy(pk) - tk).abs().max() < TOL
    # with a gain that is not constant on the rotated pairs, before != after (no nesting head <-> head_post)
    qn.weight.data = torch.rand(D, dtype=F64) + 0.5
    hy.weight.data = qn.weight.data.clone()
    a, _ = apply_rotary_pos_emb(qn(q), qn(k), cos, sin)
    b = hy(apply_rotary_pos_emb(q, k, cos, sin)[0])
    assert (a - b).abs().max() > 1e-3


def test_distinct_gains_and_bias_reduce():
    torch.manual_seed(1)
    x = torch.randn(1, T, H, D, dtype=F64)
    # head <- head_distinct: per-head gains all equal to the shared gain
    g = torch.rand(D, dtype=F64) + 0.5
    shared = Qwen3RMSNorm(D, eps=EPS).to(F64)
    shared.weight.data = g.clone()
    dist = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g.expand(H, D)
    assert (shared(x) - dist).abs().max() < TOL
    # head_ln <- head_ln_bias: StableLM's per-head LayerNorm (no bias) is Chameleon's with a zero bias
    st = StableLmLayerNormPerHead(D, H, eps=1e-5).to(F64)
    for n in st.norms:
        n.weight.data = torch.rand(D, dtype=F64) + 0.5
    ch = ChameleonLayerNorm((H, D), eps=1e-5).to(F64)
    ch.weight.data = torch.stack([n.weight.data for n in st.norms])
    ch.bias.data.zero_()
    xs = x.transpose(1, 2)                                            # StableLM normalises [B, H, T, D]
    assert (st(xs).transpose(1, 2) - ch(x)).abs().max() < TOL
