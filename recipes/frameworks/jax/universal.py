"""The universal host (tef-ucore export) in JAX: recipes/common/ucore_forward.py over jax.numpy. fp32."""
import numpy as np

import jax
import jax.numpy as jnp

from recipes.common.ucore_forward import UCoreForward  # tef:inline

jax.config.update("jax_default_matmul_precision", "highest")


class JaxOps(object):
    linear = staticmethod(lambda x, w: x @ w.T)
    mean_sq = staticmethod(lambda x: jnp.mean(x * x, -1, keepdims=True))
    asarray = staticmethod(lambda a: jnp.asarray(a))
    numpy = staticmethod(lambda t: np.asarray(t))
    zeros = staticmethod(lambda shape: jnp.zeros(shape, jnp.float32))
    cat = staticmethod(lambda xs, axis: jnp.concatenate(xs, axis))
    stack = staticmethod(lambda xs, axis: jnp.stack(xs, axis))
    reshape = staticmethod(lambda x, shape: jnp.reshape(x, tuple(shape)))
    transpose = staticmethod(lambda x, perm: jnp.transpose(x, perm))
    sum = staticmethod(lambda x, axis, keepdims=False: jnp.sum(x, axis, keepdims=keepdims))
    mean = staticmethod(lambda x, axis, keepdims=False: jnp.mean(x, axis, keepdims=keepdims))
    max = staticmethod(lambda x, axis: jnp.max(x, axis))
    exp, tanh, sqrt, relu = staticmethod(jnp.exp), staticmethod(jnp.tanh), staticmethod(jnp.sqrt), staticmethod(jax.nn.relu)
    sigmoid, silu, softplus = staticmethod(jax.nn.sigmoid), staticmethod(jax.nn.silu), staticmethod(jax.nn.softplus)
    rsqrt = staticmethod(jax.lax.rsqrt)
    gelu = staticmethod(lambda x, tanh: jax.nn.gelu(x, approximate=bool(tanh)))
    mish = staticmethod(lambda x: x * jnp.tanh(jax.nn.softplus(x)))
    clip = staticmethod(lambda x, lo, hi: jnp.clip(x, lo, hi))
    softmax = staticmethod(lambda x, axis: jax.nn.softmax(x, axis))
    einsum = staticmethod(lambda spec, a, b: jnp.einsum(spec, a, b, precision=jax.lax.Precision.HIGHEST))
    take = staticmethod(lambda table, ids: table[jnp.asarray(ids)])
    topk = staticmethod(lambda x, k: jax.lax.top_k(x, k))
    one_hot = staticmethod(lambda idx, n: jax.nn.one_hot(idx, n, dtype=jnp.float32))
    take_along_last = staticmethod(lambda x, idx: jnp.take_along_axis(x, idx, -1))
    where_pos = staticmethod(lambda c, a, b, thr=0.5: jnp.where(c > thr, a, b))
    realize = staticmethod(lambda x: x)


class UCoreModel(UCoreForward):
    B = JaxOps


def load_model(run_dir):
    return UCoreModel(run_dir)
