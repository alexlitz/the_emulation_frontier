"""Llama-style decoder in DyNet 2.1 (Python 3.8; no transformers): RMSNorm, RoPE (default / linear / llama3 /
yarn, partial), GQA, SwiGLU, optional biases, Qwen-3 qk-norm, per-layer sliding windows, tied / untied head;
greedy generation with a KV cache. fp32.

DyNet builds a fresh computation graph per token, so tokens run one at a time (a prompt is fed token by
token: exactly causal attention). The weights are DyNet parameters; the KV cache holds the computed keys and
values as numpy arrays and is fed back as input tensors; a sliding window keeps only the last W keys."""
import os

import numpy as np

import dynet_config

dynet_config.set(mem=int(os.environ.get("DYNET_MEM", "4096")), random_seed=1)
import dynet as dy  # noqa: E402

from recipes.common.runtime import Generator, load_run_dir  # tef:inline
from recipes.common.llama_spec import llama_spec, llama_weights  # tef:inline


def _split(d):
    """A divisor of d near sqrt(d). DyNet's sum_elems accumulates in one fp32 running sum; over a wide
    massive activation (position 0 of Qwen2.5 / SmolLM2) that rounding grows with d, so the RMS mean
    reduces in two stages (Qwen2.5-0.5B: 1.9e-4 -> 5.7e-5 from fp64)."""
    c = max(int(d ** 0.5), 1)
    while d % c:
        c -= 1
    return c


def _sum_sq_split(x, n):
    if n <= 64:
        return dy.sum_elems(dy.square(x))
    c = _split(n)
    k = n // c
    return dy.esum([dy.sum_elems(dy.square(dy.pick_range(x, i * k, (i + 1) * k))) for i in range(c)])


def _rms(x, w, eps, n):
    return dy.cmult(x, w) * dy.pow(_sum_sq_split(x, n) * (1.0 / n) + eps, dy.scalarInput(-0.5))


def _rot_half(x, r):
    h = r // 2
    return dy.concatenate([-dy.pick_range(x, h, r), dy.pick_range(x, 0, h)])


class LlamaModel(Generator):
    def __init__(self, run_dir):
        cfg, sd = load_run_dir(run_dir)
        self.s = s = llama_spec(cfg)
        W = llama_weights(s, sd)
        self.pc = pc = dy.ParameterCollection()
        P = lambda a: pc.parameters_from_numpy(np.asarray(a, np.float32))
        self.E = pc.lookup_parameters_from_numpy(np.asarray(W["embed"], np.float32))
        self.norm, self.head = P(W["norm"]), P(W["head"])
        self.layers = [{k: P(v) for k, v in l.items()} for l in W["layers"]]

    def new_cache(self, max_len=None):
        return [([], []) for _ in range(self.s["L"])]       # per layer: keys, values (each (KV, hd) numpy)

    def _rope(self, x, cos, sin):
        r, hd = self.s["rot"], self.s["hd"]
        xr = dy.pick_range(x, 0, r)
        out = dy.cmult(xr, cos) + dy.cmult(_rot_half(xr, r), sin)
        return out if r == hd else dy.concatenate([out, dy.pick_range(x, r, hd)])

    def _token(self, tid, cache, pos):
        s = self.s
        H, KV, hd, d = s["H"], s["KV"], s["hd"], s["d"]
        dy.renew_cg()
        ang = np.float32(pos) * s["inv_freq"]
        emb = np.concatenate((ang, ang)).astype(np.float32)
        cos = dy.inputTensor((np.cos(emb) * np.float32(s["rope_att"])).astype(np.float32))
        sin = dy.inputTensor((np.sin(emb) * np.float32(s["rope_att"])).astype(np.float32))
        x = dy.lookup(self.E, tid)
        g = H // KV
        for i, l in enumerate(self.layers):
            h = _rms(x, l["ln1"], s["eps"], d)
            q, k, v = l["q"] * h, l["k"] * h, l["v"] * h
            if "bq" in l:
                q, k, v = q + l["bq"], k + l["bk"], v + l["bv"]
            ks, vs = [], []
            for j in range(KV):
                kj = dy.pick_range(k, j * hd, (j + 1) * hd)
                if s["qk_norm"]:
                    kj = _rms(kj, l["kn"], s["eps"], hd)
                ks.append(self._rope(kj, cos, sin))
                vs.append(dy.pick_range(v, j * hd, (j + 1) * hd))
            knew = np.stack([e.npvalue().reshape(hd) for e in ks])
            vnew = np.stack([e.npvalue().reshape(hd) for e in vs])
            Kc, Vc = cache[i]
            Kc.append(knew)
            Vc.append(vnew)
            w = s["windows"][i]
            lo = 0 if w is None else max(0, len(Kc) - w)          # keys s > t - W
            Kall, Vall = np.stack(Kc[lo:], 2), np.stack(Vc[lo:], 2)   # (KV, hd, S)
            heads = []
            for hh in range(H):
                qh = dy.pick_range(q, hh * hd, (hh + 1) * hd)
                if s["qk_norm"]:
                    qh = _rms(qh, l["qn"], s["eps"], hd)
                qh = self._rope(qh, cos, sin)
                Km = dy.inputTensor(Kall[hh // g])
                Vm = dy.inputTensor(Vall[hh // g])
                p = dy.softmax(dy.transpose(Km) * qh * s["scale"])
                heads.append(Vm * p)
            o = l["o"] * dy.concatenate(heads)
            if "bo" in l:
                o = o + l["bo"]
            x = x + o
            h = _rms(x, l["ln2"], s["eps"], d)
            gt, up = l["g"] * h, l["u"] * h
            if "bg" in l:
                gt, up = gt + l["bg"], up + l["bu"]
            f = l["dn"] * dy.cmult(dy.cmult(gt, dy.logistic(gt)), up)
            if "bd" in l:
                f = f + l["bd"]
            x = x + f
        return (self.head * _rms(x, self.norm, s["eps"], d)).npvalue()

    def step(self, ids, cache, pos):
        return np.stack([self._token(t, cache, pos + j) for j, t in enumerate(ids)])


def load_model(run_dir):
    return LlamaModel(run_dir)
