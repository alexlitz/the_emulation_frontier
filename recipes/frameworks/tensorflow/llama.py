"""Llama-style decoder in TensorFlow 2 (eager tf ops, no Keras / transformers): RMSNorm, RoPE (default /
linear / llama3 / yarn, partial), GQA, SwiGLU, optional biases, Qwen-3 qk-norm, per-layer sliding windows,
tied / untied head; greedy generation with a KV cache. fp32."""
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

from recipes.common.runtime import Generator, load_run_dir  # tef:inline
from recipes.common.llama_spec import llama_spec, llama_weights  # tef:inline

tf.config.experimental.enable_op_determinism()


def _rms(x, w, eps):
    return x * tf.math.rsqrt(tf.reduce_mean(x * x, -1, keepdims=True) + eps) * w


def _rot_half(x):
    h = x.shape[-1] // 2
    return tf.concat((-x[..., h:], x[..., :h]), -1)


def _mm(x, w):                                        # x @ w.T
    return tf.linalg.matmul(x, w, transpose_b=True)


class LlamaModel(Generator):
    def __init__(self, run_dir):
        cfg, sd = load_run_dir(run_dir)
        self.s = s = llama_spec(cfg)
        W = llama_weights(s, sd)
        a = lambda t: tf.constant(np.asarray(t, np.float32))
        self.embed, self.norm, self.head = a(W["embed"]), a(W["norm"]), a(W["head"])
        self.layers = [{k: a(v) for k, v in l.items()} for l in W["layers"]]
        self.inv = a(s["inv_freq"])

    def new_cache(self, max_len=None):
        return [None] * self.s["L"]

    def _rope(self, x, pos):                          # x: (h, T, hd)
        s, T = self.s, x.shape[1]
        ang = tf.range(pos, pos + T, dtype=tf.float32)[:, None] * self.inv[None]
        emb = tf.concat((ang, ang), -1)
        cos, sin = tf.cos(emb) * s["rope_att"], tf.sin(emb) * s["rope_att"]
        r = s["rot"]
        xr, xp = x[..., :r], x[..., r:]
        return tf.concat((xr * cos + _rot_half(xr) * sin, xp), -1)

    def step(self, ids, cache, pos):
        s = self.s
        H, KV, hd = s["H"], s["KV"], s["hd"]
        x = tf.gather(self.embed, tf.constant(ids, tf.int32))
        T = int(x.shape[0])
        for i, l in enumerate(self.layers):
            h = _rms(x, l["ln1"], s["eps"])
            q, k, v = _mm(h, l["q"]), _mm(h, l["k"]), _mm(h, l["v"])
            if "bq" in l:
                q, k, v = q + l["bq"], k + l["bk"], v + l["bv"]
            q = tf.transpose(tf.reshape(q, (T, H, hd)), (1, 0, 2))
            k = tf.transpose(tf.reshape(k, (T, KV, hd)), (1, 0, 2))
            v = tf.transpose(tf.reshape(v, (T, KV, hd)), (1, 0, 2))
            if s["qk_norm"]:
                q, k = _rms(q, l["qn"], s["eps"]), _rms(k, l["kn"], s["eps"])
            q, k = self._rope(q, pos), self._rope(k, pos)
            if cache[i] is not None:
                k = tf.concat((cache[i][0], k), 1)
                v = tf.concat((cache[i][1], v), 1)
            cache[i] = (k, v)
            S = int(k.shape[1])
            g = H // KV
            k, v = tf.repeat(k, g, 0), tf.repeat(v, g, 0)
            sc = tf.linalg.matmul(q, k, transpose_b=True) * s["scale"]
            tq = tf.range(pos, pos + T)[:, None]
            tk = tf.range(S)[None]
            ok = tk <= tq
            if s["windows"][i] is not None:
                ok = ok & (tk > tq - s["windows"][i])
            sc = tf.where(ok[None], sc, tf.constant(-np.inf, tf.float32))
            o = tf.reshape(tf.transpose(tf.linalg.matmul(tf.nn.softmax(sc, -1), v), (1, 0, 2)), (T, H * hd))
            o = _mm(o, l["o"])
            if "bo" in l:
                o = o + l["bo"]
            x = x + o
            h = _rms(x, l["ln2"], s["eps"])
            gt, up = _mm(h, l["g"]), _mm(h, l["u"])
            if "bg" in l:
                gt, up = gt + l["bg"], up + l["bu"]
            f = _mm(tf.nn.silu(gt) * up, l["dn"])
            if "bd" in l:
                f = f + l["bd"]
            x = x + f
        return _mm(_rms(x, self.norm, s["eps"]), self.head).numpy()


def load_model(run_dir):
    return LlamaModel(run_dir)
