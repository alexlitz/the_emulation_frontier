"""Llama-style decoder in tinygrad (no transformers): RMSNorm, RoPE (default / linear / llama3 / yarn,
partial), GQA, SwiGLU, optional biases, Qwen-3 qk-norm, per-layer sliding windows, tied / untied head;
greedy generation with a KV cache. fp32.

The KV cache is preallocated to the full generation length and every position-dependent input (RoPE table,
mask, cache write) is passed as DATA, so every decode step has the same shapes and reuses the compiled
kernels. Long reductions run in two stages (_split). CPU needs a C compiler: clang, or CC pointing at
recipes/frameworks/tinygrad/zigcc.py."""
import os

import numpy as np

os.environ.setdefault("DEV", "CPU")
from tinygrad import Tensor, dtypes  # noqa: E402

from recipes.common.runtime import Generator, load_run_dir  # tef:inline
from recipes.common.llama_spec import llama_spec, llama_weights  # tef:inline


def _split(d):
    """A divisor of d near sqrt(d). tinygrad's CPU kernels accumulate a reduction in one fp32 running sum, so
    over d = 896 or 4864 terms of a massive activation (position 0 of Qwen2.5 / SmolLM2) the rounding grows
    with d: matmuls and the RMSNorm mean reduce in two stages instead (Qwen2.5-0.5B: 6e-4 -> 3e-5)."""
    c = max(int(d ** 0.5), 1)
    while d % c:
        c -= 1
    return c


def _rms(x, w, eps):
    d = x.shape[-1]
    c = _split(d)
    ss = (x * x).reshape(*x.shape[:-1], c, d // c).sum(-1).sum(-1, keepdim=True)
    return x * (ss / d).add(eps).rsqrt() * w


def _mm(x, w):
    """x @ w.T with the K reduction in two stages (pairwise-style fp32 accumulation)."""
    K = x.shape[-1]
    c = _split(K)
    xs = x.reshape(*x.shape[:-1], 1, c, K // c)
    ws = w.reshape(w.shape[0], c, K // c)
    return (xs * ws).sum(-1).sum(-1)


def _rot_half(x):
    h = x.shape[-1] // 2
    return (-x[..., h:]).cat(x[..., :h], dim=-1)


def _T(a):
    return Tensor(np.ascontiguousarray(a, dtype=np.float32), dtype=dtypes.float32)


class LlamaModel(Generator):
    def __init__(self, run_dir):
        cfg, sd = load_run_dir(run_dir)
        self.s = s = llama_spec(cfg)
        W = llama_weights(s, sd)
        self.embed, self.norm, self.head = _T(W["embed"]).realize(), _T(W["norm"]).realize(), _T(W["head"]).realize()
        self.layers = [{k: _T(v).realize() for k, v in l.items()} for l in W["layers"]]

    def new_cache(self, max_len=None):
        s = self.s
        n = int(max_len or 1)
        z = lambda: Tensor.zeros(s["KV"], n, s["hd"], dtype=dtypes.float32).contiguous().realize()
        return {"n": n, "kv": [(z(), z()) for _ in range(s["L"])]}

    def _tables(self, pos, T, n):
        s = self.s
        p = np.arange(pos, pos + T, dtype=np.float32)
        ang = p[:, None] * s["inv_freq"][None]
        emb = np.concatenate((ang, ang), -1)
        cos = (np.cos(emb) * np.float32(s["rope_att"])).astype(np.float32)
        sin = (np.sin(emb) * np.float32(s["rope_att"])).astype(np.float32)
        tq = np.arange(pos, pos + T)[:, None]
        tk = np.arange(n)[None]
        place = (tk.T == tq.T).astype(np.float32)                 # (n, T): cache row <- new token
        masks = {}
        for w in set(s["windows"]):
            ok = tk <= tq
            if w is not None:
                ok &= tk > tq - w
            masks[w] = _T(np.where(ok, 0.0, -np.inf).astype(np.float32))
        return _T(cos), _T(sin), _T(place), masks

    def _rope(self, x, cos, sin):
        r = self.s["rot"]
        xr = x[..., :r]
        out = xr * cos + _rot_half(xr) * sin
        return out if r == x.shape[-1] else out.cat(x[..., r:], dim=-1)

    def step(self, ids, cache, pos):
        s = self.s
        H, KV, hd = s["H"], s["KV"], s["hd"]
        T, n = len(ids), cache["n"]
        if pos + T > n:
            raise ValueError("cache too short")
        cos, sin, place, masks = self._tables(pos, T, n)
        x = self.embed[Tensor(np.array(ids, dtype=np.int32))]
        g = H // KV
        for i, l in enumerate(self.layers):
            h = _rms(x, l["ln1"], s["eps"])
            q, k, v = _mm(h, l["q"]), _mm(h, l["k"]), _mm(h, l["v"])
            if "bq" in l:
                q, k, v = q + l["bq"], k + l["bk"], v + l["bv"]
            q = q.reshape(T, H, hd).permute(1, 0, 2)
            k = k.reshape(T, KV, hd).permute(1, 0, 2)
            v = v.reshape(T, KV, hd).permute(1, 0, 2)
            if s["qk_norm"]:
                q, k = _rms(q, l["qn"], s["eps"]), _rms(k, l["kn"], s["eps"])
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            K, V = cache["kv"][i]
            K = (K + place @ k).realize()                         # rows pos..pos+T-1 were zero
            V = (V + place @ v).realize()
            cache["kv"][i] = (K, V)
            Kr = K.reshape(KV, 1, n, hd).expand(KV, g, n, hd).reshape(H, n, hd)
            Vr = V.reshape(KV, 1, n, hd).expand(KV, g, n, hd).reshape(H, n, hd)
            sc = (q @ Kr.permute(0, 2, 1)) * s["scale"] + masks[s["windows"][i]]
            o = (sc.softmax(-1) @ Vr).permute(1, 0, 2).reshape(T, H * hd)
            o = _mm(o, l["o"])
            if "bo" in l:
                o = o + l["bo"]
            x = x + o
            h = _rms(x, l["ln2"], s["eps"])
            gt, up = _mm(h, l["g"]), _mm(h, l["u"])
            if "bg" in l:
                gt, up = gt + l["bg"], up + l["bu"]
            f = _mm(gt.silu() * up, l["dn"])
            if "bd" in l:
                f = f + l["bd"]
            x = (x + f).realize()
        return _mm(_rms(x, self.norm, s["eps"]), self.head).numpy()


def load_model(run_dir):
    return LlamaModel(run_dir)
