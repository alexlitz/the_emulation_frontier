"""The softmax <- SSM / linear / hybrid near-gap, measured (attn/mixer_gap_agreement.py).

1. the constructions on synthetic tensors (fp64, CPU): each softmax construction converges to its
   recurrence in its limit knob, and the budget-matched variants do not;
2. the recorded real-model swaps (attn/data/mixer_gap_agreement.jsonl): the controls are exact, the
   passing constructions pass the payable standard, the edge's own price fails it, and
   graph/model_graph.py MIXER_GAP matches the rows;
3. the headline refuses the near-gap and no zoo host could pay any passing construction's price.

Run:  python -m pytest tests/test_mixer_gap.py -q
"""
import math
import os

import pytest
import torch
import torch.nn.functional as F

import attn.mixer_gap_agreement as M

F64 = torch.float64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def g(seed):
    return torch.Generator().manual_seed(seed)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


# --- 1. constructions ------------------------------------------------------------------------
def _mamba_inputs(T=24, d=6, N=4, seed=0):
    r = lambda *s: torch.randn(*s, generator=g(seed + len(s)), dtype=F64)
    x, B, C = r(T, d), r(T, N), torch.randn(T, N, generator=g(seed + 9), dtype=F64)
    dt = F.softplus(torch.randn(T, d, generator=g(seed + 7), dtype=F64) - 1)
    A = -torch.exp(torch.randn(d, N, generator=g(seed + 5), dtype=F64))
    return x, dt, A, B, C


def test_glu_product_and_prefix_sum_and_shift():
    x = torch.randn(50, generator=g(1), dtype=F64) * 4
    y = torch.randn(50, generator=g(2), dtype=F64)
    assert torch.allclose(M.glu_product(x, y), x * y, atol=1e-12)        # x y = silu(x) y - silu(-x) y
    v = torch.rand(30, 3, generator=g(3), dtype=F64)
    errs = [rel(M.prefix_sum_layer(v, L), v.cumsum(0)) for L in (4.0, 12.0, 30.0)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-11               # leak (t+1) e^{-L}
    sh = M.shift_heads(v, 3, 60.0)
    assert torch.allclose(sh[2][2:], v[:-2], atol=1e-12) and sh[2][:2].abs().max() < 1e-12


def test_mamba_sink_construction_converges_and_fixed_decay_does_not():
    x, dt, A, B, C = _mamba_inputs()
    ex = M.mamba_scan_exact(x, dt, A, B, C)
    errs = [rel(M.mamba_scan_sink(x, dt, A, B, C, L=L), ex) for L in (2.0, 8.0, 30.0)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-10              # the normaliser undone by a sink
    assert rel(M.mamba_scan_sink(x, dt, A, B, C, L=30.0, Lp=30.0), ex) < 1e-9   # prefix sum by a sink head
    fixed = M.mamba_scan_sink(x, dt, A, B, C, L=30.0, decay="fixed", dt_mean=dt.mean(0))
    assert rel(fixed, ex) > 1e-2                                         # one layer: data-independent decay
    d, N = A.shape
    lab = torch.arange(d * N)
    assert rel(M.mamba_scan_sink(x, dt, A, B, C, L=30.0, groups=(lab, d * N)), ex) < 1e-10
    lab2 = M.kmeans_groups((A[None] * dt[:, :, None]).reshape(len(dt), -1).T, 3)
    assert rel(M.mamba_scan_sink(x, dt, A, B, C, L=30.0, groups=(lab2, 3)), ex) > 1e-2   # 3 shared rates


def test_mamba2_small_temperature_converges():
    T, H, P, N = 20, 3, 4, 5
    x = torch.randn(T, H, P, generator=g(1), dtype=F64)
    B = torch.randn(T, H, N, generator=g(2), dtype=F64) * 0.3
    C = torch.randn(T, H, N, generator=g(3), dtype=F64) * 0.3
    dt = F.softplus(torch.randn(T, H, generator=g(4), dtype=F64))
    A = -torch.exp(torch.randn(H, generator=g(5), dtype=F64))
    ex = M.mamba2_scan_exact(x, dt, A, B, C)
    e2 = [rel(M.mamba2_scan_smallT(x, dt, A, B, C, eps=e, order=2), ex) for e in (0.3, 0.1, 0.03)]
    e4 = [rel(M.mamba2_scan_smallT(x, dt, A, B, C, eps=e, order=4), ex) for e in (0.3, 0.1, 0.03)]
    assert e2[0] > e2[1] > e2[2] and e4[0] > e4[1] > e4[2]              # O(eps^2) / O(eps^4)
    assert e4[0] < e2[0] / 10 and e4[2] < 1e-7


def test_rwkv4_is_one_alibi_softmax_head_per_channel():
    T, Cd = 40, 5
    k = torch.randn(T, Cd, generator=g(1), dtype=F64)
    v = torch.randn(T, Cd, generator=g(2), dtype=F64)
    w, u = torch.rand(Cd, generator=g(3), dtype=F64) + 0.1, torch.randn(Cd, generator=g(4), dtype=F64)
    ex = M.rwkv4_wkv_exact(w, u, k, v)
    # the sequential HF recurrence (rwkv_linear_attention_cpu) equals the closed form
    num, den, out = torch.zeros(Cd, dtype=F64), torch.zeros(Cd, dtype=F64), []
    for t in range(T):
        out.append((num + torch.exp(u + k[t]) * v[t]) / (den + torch.exp(u + k[t])))
        num, den = torch.exp(-w) * num + torch.exp(k[t]) * v[t], torch.exp(-w) * den + torch.exp(k[t])
    assert torch.allclose(ex, torch.stack(out), atol=1e-12)
    assert rel(M.rwkv4_wkv_softmax(w, u, k, v, m=64), ex) < 1e-12        # exact for T <= m
    assert rel(M.rwkv4_wkv_softmax(w, u, k, v, m=16), ex) > 1e-4         # aliases at D = 16, 32
    assert rel(M.rwkv4_wkv_softmax(w, u, k, v, m=0), ex) > 1e-2          # no diagonal bonus


def test_lfm2_one_layer_stencil_converges():
    T, Cd, K = 16, 4, 3
    Bm, Cm, xx = (torch.randn(T, Cd, generator=g(i), dtype=F64) for i in (1, 2, 3))
    w = torch.randn(Cd, 1, K, generator=g(4), dtype=F64)
    ex = Cm * M.causal_conv(Bm * xx, w, None)
    e = [rel(M.lfm2_onelayer(Bm, Cm, xx, w, 60.0, 40.0, k / 3, k / 3, order=4), ex) for k in (0.3, 0.1, 0.03)]
    assert e[0] > e[1] > e[2] and e[2] < 1e-7
    assert rel(Cm * M.causal_conv(Bm * xx, w, None, beta=60.0), ex) < 1e-10   # shift heads, carried B x


def test_gated_delta_needs_depth_proportional_to_context():
    T, H, dk = 24, 2, 8
    q = M._l2(torch.randn(T, H, dk, generator=g(1), dtype=F64)) / math.sqrt(dk)
    k = M._l2(torch.randn(T, H, dk, generator=g(2), dtype=F64))
    v = torch.randn(T, H, dk, generator=g(3), dtype=F64)
    beta = torch.sigmoid(torch.randn(T, H, generator=g(4), dtype=F64))
    gg = -F.softplus(torch.randn(T, H, generator=g(5), dtype=F64)) * 0.2
    ex = M.gdn_exact(q, k, v, beta, gg)
    errs = {p: rel(M.gdn_neumann(q, k, v, beta, gg, p), ex) for p in (0, 2, 8, T - 2, T - 1)}
    assert errs[0] > 1e-2 and errs[2] > 1e-3 and errs[T - 2] > 0         # every truncation is off ...
    assert errs[T - 1] < 1e-12                                           # ... exact only at p = T - 1
    soft = M.gdn_neumann(q, k, v, beta, gg, T - 1, soft=(0.03, 40.0, 4))
    assert rel(soft, ex) < 1e-6                                          # each hop as softmax heads


# --- 2. the recorded real-model swaps ---------------------------------------------------------
@pytest.fixture(scope="module")
def rows():
    from attn.mixer_gap_report import load, whole
    return {(r["family"], r["variant"]): r for r in whole(load(os.path.join(ROOT, "attn/data/mixer_gap_agreement.jsonl")))}


def strict(r):
    return r["top1"] >= 0.99 and r["ex32"] >= 0.90


@pytest.mark.parametrize("fam", ["mamba", "rwkv4", "granite", "lfm2", "qwen35"])
def test_controls_are_exact(rows, fam):
    r = rows[(fam, "exact")]
    assert r["top1"] == 1.0 and r["ex32"] == 1.0 and r["kl"] < 1e-8 and r["n_prompts"] == 24


def test_recorded_constructions_and_budget_variants(rows):
    from graph.model_graph import MIXER_GAP
    fam_of = {"mamba1": "mamba", "rwkv4": "rwkv4", "mamba2": "granite", "shortconv": "lfm2", "gdn": "qwen35"}
    for key, spec in MIXER_GAP.items():
        for var in (spec["best"], spec.get("alt")):
            if var:
                r = rows[(fam_of[key], var)]
                assert strict(r), (key, var, r["top1"], r["ex32"])      # the priced construction passes ...
    # ... and what the edge's own price allows does not
    for fam, var in [("mamba", "fixed-decay"), ("mamba", "heads4096"), ("rwkv4", "softmax-m0"),
                     ("granite", "fixed-decay"), ("lfm2", "conv4"), ("qwen35", "neumann-p16"),
                     ("mamba", "distill-h12-s600"), ("mamba", "distill-h48-s600")]:
        assert not strict(rows[(fam, var)]), (fam, var)
    # gated DeltaNet: every truncated Jacobi depth that is shorter than the context fails
    for p in (0, 1, 2, 4, 8, 16, 32):
        assert not strict(rows[("qwen35", f"neumann-p{p}")]), p


# --- 3. the relation ---------------------------------------------------------------------------
def test_headline_refuses_the_mixer_gap_and_no_host_pays_the_price():
    import numpy as np
    from graph.currency_curve import points
    from graph.model_graph import MIXER_GAP, MIXER_GAP_PAYABLE, emul_matrix, payable_reaches, slot_reaches
    assert not MIXER_GAP_PAYABLE
    for t in ("ssm", "linear", "hybrid"):
        assert slot_reaches("mixer", t, "softmax")                       # the TYPE order (plans, lattice)
        assert not payable_reaches("mixer", t, "softmax")                # ... is not a headline edge
    P = points()
    mix = np.array([p["slots"]["mixer"] for p in P])
    soft = mix == "softmax"
    ok, okg = emul_matrix(P, mode="payable"), emul_matrix(P, mode="mixgap")
    assert not ok[np.ix_(~soft, soft)].any()                             # no softmax host for a recurrent target
    assert okg[np.ix_(~soft, soft)].any()                                # the old relation granted them
    assert (emul_matrix(P, mode="nomix") == ok).all()                    # legacy alias
    # every passing construction's price exceeds every zoo host: heads beyond the widest softmax host,
    # or a non-stock group norm, or depth ~ context
    max_heads = max(p["dims"]["qheads"] for p in P if p["slots"]["mixer"] == "softmax")
    for key, spec in MIXER_GAP.items():
        assert spec["heads"] > max_heads or spec["groupnorm"] or spec["depth"] == "T", key
