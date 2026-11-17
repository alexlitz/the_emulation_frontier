"""The universal host (tef-ucore export) in TensorFlow 2: recipes/common/ucore_forward.py over eager tf ops.
fp32."""
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

from recipes.common.ucore_forward import UCoreForward  # tef:inline


def _clip(x, lo, hi):
    if lo is not None:
        x = tf.maximum(x, lo)
    if hi is not None:
        x = tf.minimum(x, hi)
    return x


def _topk(x, k):
    r = tf.math.top_k(x, k)
    return r.values, r.indices


class TFOps(object):
    linear = staticmethod(lambda x, w: tf.linalg.matmul(x, w, transpose_b=True))
    mean_sq = staticmethod(lambda x: tf.reduce_mean(x * x, -1, keepdims=True))
    asarray = staticmethod(lambda a: tf.constant(a))
    numpy = staticmethod(lambda t: t.numpy())
    zeros = staticmethod(lambda shape: tf.zeros(shape, tf.float32))
    cat = staticmethod(lambda xs, axis: tf.concat(list(xs), axis))
    stack = staticmethod(lambda xs, axis: tf.stack(list(xs), axis))
    reshape = staticmethod(lambda x, shape: tf.reshape(x, [int(d) for d in shape]))
    transpose = staticmethod(lambda x, perm: tf.transpose(x, perm))
    sum = staticmethod(lambda x, axis, keepdims=False: tf.reduce_sum(x, axis, keepdims=keepdims))
    mean = staticmethod(lambda x, axis, keepdims=False: tf.reduce_mean(x, axis, keepdims=keepdims))
    max = staticmethod(lambda x, axis: tf.reduce_max(x, axis))
    exp, tanh, sqrt, relu = staticmethod(tf.exp), staticmethod(tf.tanh), staticmethod(tf.sqrt), staticmethod(tf.nn.relu)
    sigmoid, silu, softplus = staticmethod(tf.sigmoid), staticmethod(tf.nn.silu), staticmethod(tf.nn.softplus)
    rsqrt = staticmethod(tf.math.rsqrt)
    gelu = staticmethod(lambda x, tanh: tf.nn.gelu(x, approximate=bool(tanh)))
    mish = staticmethod(lambda x: x * tf.tanh(tf.nn.softplus(x)))
    clip = staticmethod(_clip)
    softmax = staticmethod(lambda x, axis: tf.nn.softmax(x, axis))
    einsum = staticmethod(lambda spec, a, b: tf.einsum(spec, a, b))
    take = staticmethod(lambda table, ids: tf.gather(table, tf.constant(ids)))
    topk = staticmethod(_topk)
    one_hot = staticmethod(lambda idx, n: tf.one_hot(idx, n, dtype=tf.float32))
    take_along_last = staticmethod(lambda x, idx: tf.gather(x, idx, batch_dims=1))
    where_pos = staticmethod(lambda c, a, b, thr=0.5: tf.where(c > thr, a, b))
    realize = staticmethod(lambda x: x)


class UCoreModel(UCoreForward):
    B = TFOps


def load_model(run_dir):
    return UCoreModel(run_dir)
