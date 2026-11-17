"""approx/: approximate Gemma-4 -> gpt-oss conversion, on a tiny pair with the same mismatch
structure (approx/tiny.py). CPU, ~3-4 min the first time (the tiny teacher is trained once and
cached under ~/.cache/approx_tiny), ~1-2 min after."""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from approx.arch import (SCfg, layer_view, random_student, rmsnorm, s_embed, s_layer, s_logits, student_forward,
                         student_rope, swiglu_oai, gelu_tanh, t_embed, t_logits, teacher_forward, to_hf_student)
from approx.convert import Converter, Opts, group_experts, layer_map, naive_opts
from approx.evaluate import Student, compare, save_student
from approx.importance import Teacher, attribution, layer_ablation, stream
from approx.tiny import tcfg_to_hf, tiny_scfg, tiny_teacher, tiny_tokens, trained_tiny_teacher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TF5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")


@pytest.fixture(scope="module", autouse=True)
def _grad_enabled():
    """elephantine/spot_check_reductions.py turns autograd off at import (torch.set_grad_enabled(False));
    in a full-suite run tests/test_reduction_spot_checks.py imports it at collection. The attribution /
    refit / distill code here needs gradients."""
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(prev)


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(0)
    cfg, w = trained_tiny_teacher()
    T = Teacher(cfg, w)
    cal = tiny_tokens(48, 48, seed=1)
    ev = tiny_tokens(24, 48, seed=2)
    hs = stream(T, cal)
    return dict(T=T, cfg=cfg, w=w, cal=cal, ev=ev, hs=hs, dkl=layer_ablation(T, cal), imp=attribution(T, cal, hs=hs),
                hev=stream(T, ev)[-1])


def convert(p, o, cal=None, hs=None, imp="default", dkl="default"):
    sc = tiny_scfg()
    conv = Converter(p["T"], sc, replace(o, verbose=False), imp=p["imp"] if imp == "default" else imp,
                     dkl=p["dkl"] if dkl == "default" else dkl)
    layers = {}
    glob, info = conv.run(p["cal"] if cal is None else cal, hs=p["hs"] if hs is None else hs,
                          sink=lambda i, lw: layers.__setitem__(i, lw))
    sw = dict(glob)
    for i, lw in layers.items():
        sw.update({f"layers.{i}.{k}": v for k, v in lw.items()})
    return sc, sw, info, glob, layers


# ----------------------------------------------------------------------------- the reference forwards

def test_student_forward_matches_stock_gpt_oss():
    from transformers import GptOssConfig, GptOssForCausalLM
    s = tiny_scfg()
    s.rope_scaling = {"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0,
                      "original_max_position_embeddings": 16, "truncate": False}
    w = random_student(s, std=0.3, dtype=torch.float64)
    for k in w:
        if k.endswith("bias") or k.endswith("sinks"):
            w[k] = torch.randn_like(w[k]) * 0.3
    c = GptOssConfig(**{k: v for k, v in s.hf_config().items() if k not in ("architectures", "model_type")})
    c._attn_implementation = "eager"
    m = GptOssForCausalLM(c).double().eval()
    m.load_state_dict(to_hf_student(w), strict=True)
    ids = tiny_tokens(2, 40)
    with torch.no_grad():
        hs = m.model(ids, output_hidden_states=True).hidden_states
        rope = student_rope(s, 40, "cpu")
        for li in range(s.L - 1):            # the last recorded state is post-norm
            lw = layer_view(w, li)
            lw["_type"] = s.layer_types[li]
            assert (s_layer(lw, s, hs[li], rope) - hs[li + 1]).abs().max() < 1e-10
        assert (m(ids).logits - student_forward(w, s, ids)).abs().max() < 1e-5


TF5_BODY = r'''
import sys, types, torch
m = types.ModuleType("transformers.dependency_versions_check"); m.dep_version_check = lambda *a, **k: None
sys.modules["transformers.dependency_versions_check"] = m
sys.modules["fla"] = None
from transformers.models.gemma4 import modeling_gemma4 as gm, configuration_gemma4 as gc
from approx.tiny import tiny_teacher, tcfg_to_hf, tiny_tokens
from approx.arch import t_layer, layer_view, t_embed
cfg, w = tiny_teacher(dtype=torch.float64)
hc = gc.Gemma4TextConfig(**tcfg_to_hf(cfg)); hc._attn_implementation = "eager"; hc._experts_implementation = "eager"
m = gm.Gemma4ForCausalLM(hc).double().eval()
m.load_state_dict({"model." + k: v for k, v in w.items()}, strict=False)
ids = tiny_tokens(2, 40)
gap = 0.0
with torch.no_grad():
    hs = m.model(ids, output_hidden_states=True).hidden_states
    gap = max(gap, (hs[0] - t_embed(w, cfg, ids)).abs().max().item())
    for li in range(cfg.L - 1):
        gap = max(gap, ((hs[li + 1] - t_layer(layer_view(w, li), cfg, li, hs[li])).abs().max() / hs[li + 1].abs().max()).item())
print("GAP", gap)
'''


def test_teacher_forward_matches_stock_gemma4():
    """per layer vs transformers 5.18's Gemma4TextDecoderLayer (in a subprocess: 4.57 is installed).
    ~1e-6 not 1e-12: HF's eager attention / router softmax run in fp32 even for fp64 models."""
    if not os.path.isdir(TF5):
        pytest.skip("transformers 5.18 not unpacked")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, ROOT]))
    r = subprocess.run([sys.executable, "-c", TF5_BODY], env=env, capture_output=True, text=True, timeout=600)
    m = re.search(r"GAP (\S+)", r.stdout)
    assert m, r.stdout[-2000:] + r.stderr[-3000:]
    assert float(m.group(1)) < 1e-5


# ----------------------------------------------------------------------------- the exact parts are exact

def test_exact_embedding_pad_final_norm_and_lm_head(pair):
    """no layers: student logits == teacher's pre-softcap logits (pad 60->64, sqrt(h) scale folded,
    RMSNorm weight x sqrt(60/64), untied lm_head)."""
    p = pair
    cfg = p["cfg"]
    sc, sw, info, glob, _ = convert(p, replace(Opts(), store_dtype="float32"))
    ids = p["ev"][:4]
    hT = t_embed(p["w"], cfg, ids)
    hS = s_embed(glob, ids)
    assert (hS[..., :cfg.h] - hT).abs().max() == 0 and hS[..., cfg.h + 1:].abs().max() == 0
    # the first padded dim is the eps channel (approx.convert.eps_plan), at the first kept layer's C
    assert (hS[..., cfg.h] - info["eps_v"][info["pi"][0]]).abs().max() == 0
    assert info["eps_s"] == pytest.approx(cfg.eps * cfg.h / (sc.h * max(c * c for c in info["C"])))
    zT = t_logits(p["w"], cfg, hT, softcap=False)
    g = dict(glob)
    g["lm_head.weight"] = g["lm_head.weight"] / info.get("lm_scale", 1.0)
    zS = s_logits(g, sc, hS)
    assert (zT - zS).abs().max() < 1e-4 * zT.abs().max()


def test_exact_input_norm_fold_and_residual_scale(pair):
    """student input RMSNorm on (padded residual / C) == teacher input_layernorm on the residual, at the
    REAL eps (Lean rmsNorm_smul_eps, hidden_pad_needs_eps): the student eps is eps h_t / (h_s max C^2) and
    the eps channel carries v_l^2 = h_t eps / C_l^2 - h_s eps_s, so every layer gets its eps / C_l^2. With
    the teacher's eps kept and no channel (the old converter) it misses by sqrt((ms + eps)/(ms + eps C^2))."""
    p = pair
    cfg = p["cfg"]
    sc, sw, info, glob, layers = convert(p, naive_opts(store_dtype="float32"))   # exact up to fp32, not bf16 storage
    for i in range(len(info["pi"])):
        l = info["pi"][i]
        h = p["hs"][l][:2].double() * 1e-3                 # small residual: eps / ms ~ 1, the eps term matters
        C = info["C"][l]
        hp = F.pad(h, (0, sc.h - cfg.h)) / C
        hp[..., cfg.h] = info["eps_v"][l]
        wS = layers[i]["input_layernorm.weight"].double()
        wT = p["w"][f"layers.{l}.input_layernorm.weight"].double()
        assert wS[cfg.h:].abs().max() == 0                  # the channel never reaches a linear map
        s = rmsnorm(hp, wS, info["eps_s"]).double()
        t = rmsnorm(h, wT, cfg.eps).double()
        assert (s[..., :cfg.h] - t).abs().max() < 1e-5 * t.abs().max()
        if abs(C - 1) > 1e-2:                               # the old fold (teacher eps, no channel) misses
            old = rmsnorm(F.pad(h, (0, sc.h - cfg.h)) / C, wS, cfg.eps * cfg.h / sc.h).double()
            assert (old[..., :cfg.h] - t).abs().max() > 1e-3 * t.abs().max()


def test_eps_plan():
    """approx.convert.eps_plan: per-stream eps / C^2 through the channel; the plain pad fold at C = 1."""
    from approx.convert import eps_plan
    e, v = eps_plan(1e-6, 60, 64, [1.0, 1.0])
    assert e == pytest.approx(1e-6 * 60 / 64) and v == [0.0, 0.0]
    C = [1.0, 0.5, 2.0, 0.25]
    e, v = eps_plan(1e-6, 60, 64, C)
    for c, vl in zip(C, v):
        assert (64 * e + vl * vl) / 60 == pytest.approx(1e-6 / (c * c), rel=1e-12)
    assert eps_plan(1e-6, 64, 64, C) == (1e-6, None)

def test_exact_up_bias_and_range_fold():
    """(clamp(k u - 1) + 1) * gate / k == u * gate while |k u| <= 6: the +1 and the up clamp cost
    nothing inside the range."""
    g = torch.randn(1000) * 2
    u = torch.randn(1000) * 30
    k = 6.0 / u.abs().max()
    got = swiglu_oai(g.clamp(max=7), k * u - 1) / k
    ref = (g.clamp(max=7) * torch.sigmoid(1.702 * g.clamp(max=7))) * u
    assert (got - ref).abs().max() < 1e-4 * ref.abs().max()


def test_exact_router_fold_singleton_groups():
    """with one teacher expert per student expert the folded router reproduces the teacher's
    logit ordering exactly (input scale / sqrt(h) / scale vector are exact folds)."""
    from approx.arch import t_router
    from approx.convert import ffn_closed
    cfg, w = tiny_teacher()
    lw = layer_view(w, 0)
    sc = SCfg(vocab=256, h=64, L=1, nq=16, nkv=2, hd=8, E=cfg.E, topk=cfg.topk, I=8, window=16)
    st = {"a_out": torch.ones(cfg.h), "b_out": torch.ones(cfg.h)}
    fw = ffn_closed(lw, cfg, sc, [[e] for e in range(cfg.E)], [[(e, u) for u in range(8)] for e in range(cfg.E)],
                    [], st, 1.0, replace(Opts(), act_fit=False))
    n = rmsnorm(torch.randn(50, cfg.h), None, cfg.eps)
    lt, _, _, ti = t_router(lw, cfg, n)
    ls = F.pad(n, (0, 4)) @ fw["mlp.router.weight"].T + fw["mlp.router.bias"]
    assert (lt - ls).abs().max() < 1e-4 * lt.abs().max()


def test_exact_sinks_off():
    from approx.arch import attend
    q, k, v = torch.randn(2, 4, 10, 8), torch.randn(2, 2, 10, 8), torch.randn(2, 2, 10, 8)
    a = attend(q, k, v, 8 ** -0.5, 4)
    b = attend(q, k, v, 8 ** -0.5, 4, sinks=torch.full((4,), -1e4))
    assert (a - b).abs().max() < 1e-6


# ----------------------------------------------------------------------------- planning pieces

def test_layer_map_respects_types():
    cfg = tiny_teacher()[0]
    sc = tiny_scfg()
    dkl = torch.rand(cfg.L)
    pi = layer_map(cfg, sc, dkl)
    assert len(pi) == sc.L and pi == sorted(pi)
    for i, l in enumerate(pi):
        if cfg.layer_types[l] == "full_attention":
            assert sc.layer_types[i] == "full_attention"
    # cheapest feasible drop set: no other type-compatible choice drops less dKL
    import itertools
    best = min(sum(dkl[d] for d in drop) for drop in itertools.combinations(range(cfg.L), cfg.L - sc.L)
               if _feasible([l for l in range(cfg.L) if l not in drop], cfg, sc))
    assert abs(sum(dkl[l] for l in range(cfg.L) if l not in pi) - best) < 1e-6


def _feasible(keep, cfg, sc):
    return all(not (cfg.layer_types[l] == "full_attention" and sc.layer_types[i] != "full_attention") for i, l in enumerate(keep))


def test_grouping_is_a_partition():
    C = torch.rand(64, 64)
    C = C @ C.T
    for mode in ("modularity", "coact", "consecutive"):
        g = group_experts(C, C.diagonal(), 16, mode, drop=[3, 5, 7, 9])
        flat = sorted(e for gs in g for e in gs)
        assert flat == sorted(set(range(64)) - {3, 5, 7, 9}) and len(g) == 16 and max(map(len, g)) <= 4


# ----------------------------------------------------------------------------- end to end

def test_converted_beats_naive_and_random(pair):
    p = pair
    sc, sw, info, _, _ = convert(p, Opts())
    r_conv = compare(p["T"], Student(sc, sw), p["ev"], hT=p["hev"])
    sc2, sw2, _, _, _ = convert(p, naive_opts(), imp=None, dkl=None)
    r_naive = compare(p["T"], Student(sc2, sw2), p["ev"], hT=p["hev"])
    r_rand = compare(p["T"], Student(sc, random_student(sc)), p["ev"], hT=p["hev"])
    print("conv", r_conv, "\nnaive", r_naive, "\nrandom", r_rand)
    assert r_conv["kl"] < 0.6 * r_naive["kl"]
    assert r_conv["kl"] < 0.6 * r_rand["kl"]
    assert r_conv["top1"] > r_naive["top1"] + 0.1
    assert len(info["pi"]) == sc.L


def test_grad_refit_never_worse_on_holdout(pair):
    """the gradient block reconstruction keeps its best held-out step, so per layer it can only
    lower the held-out FFN error it starts from; end to end it should help."""
    p = pair
    sc, sw, info, _, _ = convert(p, replace(Opts(), grad_steps=40, grad_lr=1e-2))
    for x in info["layers"]:
        assert x["grad_refit"]["after"] <= x["grad_refit"]["before"] + 1e-9
    r = compare(p["T"], Student(sc, sw), p["ev"], hT=p["hev"])
    sc0, sw0, _, _, _ = convert(p, Opts())
    r0 = compare(p["T"], Student(sc0, sw0), p["ev"], hT=p["hev"])
    print("grad refit", r["kl"], "LS only", r0["kl"])
    assert r["kl"] < 1.05 * r0["kl"]


def test_saved_student_loads_in_stock_gpt_oss(pair, tmp_path):
    from transformers import GptOssConfig, GptOssForCausalLM
    p = pair
    sc, sw, info, glob, layers = convert(p, naive_opts(), imp=None, dkl=None)
    save_student(str(tmp_path), sc, glob, layers, dtype=torch.float32)
    m = GptOssForCausalLM.from_pretrained(str(tmp_path), torch_dtype=torch.float32).eval()
    m.config._attn_implementation = "eager"
    ids = p["ev"][:2]
    with torch.no_grad():
        ref = m(ids).logits
    ours = student_forward(sw, sc, ids)
    assert (ref - ours).abs().max() < 1e-3 * ref.abs().max()


def test_domain_calibration_preserves_domain(pair):
    """two synthetic domains route to disjoint halves of the experts: calibrating on domain A keeps
    A's routing (mass recall) and fidelity better on A than calibrating on B does."""
    p = pair
    T = p["T"]
    res = {}
    evA = tiny_tokens(24, 48, "A", seed=11)
    hA = stream(T, evA)[-1]
    for dom in ("A", "B"):
        cal = tiny_tokens(48, 48, dom, seed=21)
        hs = stream(T, cal)
        imp = attribution(T, cal, hs=hs)
        dkl = layer_ablation(T, cal)
        sc, sw, info, _, _ = convert(p, replace(Opts(), drop_experts=16), cal=cal, hs=hs, imp=imp, dkl=dkl)
        res[dom] = compare(T, Student(sc, sw), evA, hT=hA)
    print(res)
    assert res["A"]["kl"] < res["B"]["kl"]


def test_distill_reduces_kl(pair):
    from approx.distill import PipeStudent, TeacherHead, distill
    p = pair
    sc, sw, _, _, _ = convert(p, naive_opts(), imp=None, dkl=None)
    torch.manual_seed(0)
    model = PipeStudent(Student(sc, sw), devices=("cpu",), r=4, r_exp=2, dtype=torch.float32, T=48, log=lambda *a: None)
    th = TeacherHead(p["cfg"], {"embed_tokens.weight": p["w"]["embed_tokens.weight"], "norm.weight": p["w"]["norm.weight"]}, "cpu")
    curve = distill(model, th, p["cal"], p["hs"][-1], p["ev"][:8], p["hev"][:8], steps=30, bs=8, lr=3e-3, lr_other=3e-3,
                    eval_every=30, log=lambda *a: None)
    assert curve[-1]["kl"] < 0.9 * curve[0]["kl"]
