"""Llama-style decoder in plain PyTorch (no transformers): RMSNorm, RoPE (default / linear / llama3 / yarn,
partial), GQA, SwiGLU, optional q/k/v/o and MLP biases, Qwen-3 qk-norm, per-layer sliding windows,
tied / untied head; greedy generation with a KV cache. fp32."""
import numpy as np
import torch

from recipes.common.runtime import Generator, load_run_dir  # tef:inline
from recipes.common.llama_spec import llama_spec, llama_weights  # tef:inline

F32 = torch.float32


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


class LlamaModel(Generator):
    def __init__(self, run_dir):
        cfg, sd = load_run_dir(run_dir)
        self.s = s = llama_spec(cfg)
        W = llama_weights(s, sd)
        self.embed, self.norm, self.head = _t(W["embed"]), _t(W["norm"]), _t(W["head"])
        self.layers = [{k: _t(v) for k, v in l.items()} for l in W["layers"]]
        self.inv = _t(s["inv_freq"])
        torch.set_grad_enabled(False)

    def new_cache(self, max_len=None):
        return [None] * self.s["L"]

    def _rope(self, x, pos):                          # x: (h, T, hd); positions pos .. pos+T-1
        s, T = self.s, x.shape[1]
        ang = torch.arange(pos, pos + T, dtype=F32)[:, None] * self.inv[None]
        emb = torch.cat((ang, ang), -1)
        cos, sin = emb.cos() * s["rope_att"], emb.sin() * s["rope_att"]
        r = s["rot"]
        xr, xp = x[..., :r], x[..., r:]
        return torch.cat((xr * cos + _rot_half(xr) * sin, xp), -1)

    def step(self, ids, cache, pos):
        s = self.s
        H, KV, hd = s["H"], s["KV"], s["hd"]
        x = self.embed[torch.tensor(ids)]
        T = x.shape[0]
        for i, l in enumerate(self.layers):
            h = _rms(x, l["ln1"], s["eps"])
            q, k, v = h @ l["q"].T, h @ l["k"].T, h @ l["v"].T
            if "bq" in l:
                q, k, v = q + l["bq"], k + l["bk"], v + l["bv"]
            q = q.view(T, H, hd).transpose(0, 1)
            k = k.view(T, KV, hd).transpose(0, 1)
            v = v.view(T, KV, hd).transpose(0, 1)
            if s["qk_norm"]:
                q, k = _rms(q, l["qn"], s["eps"]), _rms(k, l["kn"], s["eps"])
            q, k = self._rope(q, pos), self._rope(k, pos)
            if cache[i] is not None:
                k = torch.cat((cache[i][0], k), 1)
                v = torch.cat((cache[i][1], v), 1)
            cache[i] = (k, v)
            S = k.shape[1]
            g = H // KV
            k, v = k.repeat_interleave(g, 0), v.repeat_interleave(g, 0)
            sc = (q @ k.transpose(1, 2)) * s["scale"]
            tq = torch.arange(pos, pos + T)[:, None]
            tk = torch.arange(S)[None]
            ok = tk <= tq
            if s["windows"][i] is not None:
                ok = ok & (tk > tq - s["windows"][i])
            sc = sc.masked_fill(~ok, float("-inf"))
            o = (sc.softmax(-1) @ v).transpose(0, 1).reshape(T, H * hd)
            o = o @ l["o"].T
            if "bo" in l:
                o = o + l["bo"]
            x = x + o
            h = _rms(x, l["ln2"], s["eps"])
            gt, up = h @ l["g"].T, h @ l["u"].T
            if "bg" in l:
                gt, up = gt + l["bg"], up + l["bu"]
            f = (torch.nn.functional.silu(gt) * up) @ l["dn"].T
            if "bd" in l:
                f = f + l["bd"]
            x = x + f
        return (_rms(x, self.norm, s["eps"]) @ self.head.T).numpy()


def load_model(run_dir):
    return LlamaModel(run_dir)
