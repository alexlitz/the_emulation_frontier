"""Llama-style decoder in JAX (jax.numpy, no flax / transformers): RMSNorm, RoPE (default / linear / llama3 /
yarn, partial), GQA, SwiGLU, optional biases, Qwen-3 qk-norm, per-layer sliding windows, tied / untied head;
greedy generation with a KV cache. fp32 (highest matmul precision)."""
import numpy as np

import jax
import jax.numpy as jnp

from recipes.common.runtime import Generator, load_run_dir  # tef:inline
from recipes.common.llama_spec import llama_spec, llama_weights  # tef:inline

jax.config.update("jax_default_matmul_precision", "highest")


def _rms(x, w, eps):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w


def _rot_half(x):
    h = x.shape[-1] // 2
    return jnp.concatenate((-x[..., h:], x[..., :h]), -1)


class LlamaModel(Generator):
    def __init__(self, run_dir):
        cfg, sd = load_run_dir(run_dir)
        self.s = s = llama_spec(cfg)
        W = llama_weights(s, sd)
        a = lambda t: jnp.asarray(t, jnp.float32)
        self.embed, self.norm, self.head = a(W["embed"]), a(W["norm"]), a(W["head"])
        self.layers = [{k: a(v) for k, v in l.items()} for l in W["layers"]]
        self.inv = a(s["inv_freq"])

    def new_cache(self, max_len=None):
        return [None] * self.s["L"]

    def _rope(self, x, pos):                          # x: (h, T, hd)
        s, T = self.s, x.shape[1]
        ang = jnp.arange(pos, pos + T, dtype=jnp.float32)[:, None] * self.inv[None]
        emb = jnp.concatenate((ang, ang), -1)
        cos, sin = jnp.cos(emb) * s["rope_att"], jnp.sin(emb) * s["rope_att"]
        r = s["rot"]
        xr, xp = x[..., :r], x[..., r:]
        return jnp.concatenate((xr * cos + _rot_half(xr) * sin, xp), -1)

    def step(self, ids, cache, pos):
        s = self.s
        H, KV, hd = s["H"], s["KV"], s["hd"]
        x = self.embed[jnp.asarray(ids)]
        T = x.shape[0]
        for i, l in enumerate(self.layers):
            h = _rms(x, l["ln1"], s["eps"])
            q, k, v = h @ l["q"].T, h @ l["k"].T, h @ l["v"].T
            if "bq" in l:
                q, k, v = q + l["bq"], k + l["bk"], v + l["bv"]
            q = q.reshape(T, H, hd).transpose(1, 0, 2)
            k = k.reshape(T, KV, hd).transpose(1, 0, 2)
            v = v.reshape(T, KV, hd).transpose(1, 0, 2)
            if s["qk_norm"]:
                q, k = _rms(q, l["qn"], s["eps"]), _rms(k, l["kn"], s["eps"])
            q, k = self._rope(q, pos), self._rope(k, pos)
            if cache[i] is not None:
                k = jnp.concatenate((cache[i][0], k), 1)
                v = jnp.concatenate((cache[i][1], v), 1)
            cache[i] = (k, v)
            S = k.shape[1]
            g = H // KV
            k, v = jnp.repeat(k, g, 0), jnp.repeat(v, g, 0)
            sc = (q @ k.transpose(0, 2, 1)) * s["scale"]
            tq = jnp.arange(pos, pos + T)[:, None]
            tk = jnp.arange(S)[None]
            ok = tk <= tq
            if s["windows"][i] is not None:
                ok = ok & (tk > tq - s["windows"][i])
            sc = jnp.where(ok, sc, -jnp.inf)
            o = (jax.nn.softmax(sc, -1) @ v).transpose(1, 0, 2).reshape(T, H * hd)
            o = o @ l["o"].T
            if "bo" in l:
                o = o + l["bo"]
            x = x + o
            h = _rms(x, l["ln2"], s["eps"])
            gt, up = h @ l["g"].T, h @ l["u"].T
            if "bg" in l:
                gt, up = gt + l["bg"], up + l["bu"]
            f = (jax.nn.silu(gt) * up) @ l["dn"].T
            if "bd" in l:
                f = f + l["bd"]
            x = x + f
        return np.asarray(_rms(x, self.norm, s["eps"]) @ self.head.T)


def load_model(run_dir):
    return LlamaModel(run_dir)
