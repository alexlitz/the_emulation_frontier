"""The universal host (tef-ucore export) in tinygrad: recipes/common/ucore_forward.py over tinygrad Tensors.
fp32. CPU needs a C compiler: clang, or CC pointing at recipes/frameworks/tinygrad/zigcc.py."""
import os

import numpy as np

os.environ.setdefault("DEV", "CPU")
from tinygrad import Tensor, dtypes  # noqa: E402

from recipes.common.ucore_forward import UCoreForward  # tef:inline


def _asarray(a):
    a = np.ascontiguousarray(a)
    if a.dtype.kind in "iu":
        return Tensor(a.astype(np.int32), dtype=dtypes.int32)
    return Tensor(a.astype(np.float32), dtype=dtypes.float32)


def _clip(x, lo, hi):
    if lo is not None:
        x = x.maximum(lo)
    if hi is not None:
        x = x.minimum(hi)
    return x


def _split(d):
    """A divisor of d near sqrt(d): long fp32 reductions run in two stages (see llama.py)."""
    c = max(int(d ** 0.5), 1)
    while d % c:
        c -= 1
    return c


def _linear(x, w):
    K = x.shape[-1]
    c = _split(K)
    return (x.reshape(*x.shape[:-1], 1, c, K // c) * w.reshape(w.shape[0], c, K // c)).sum(-1).sum(-1)


def _mean_sq(x):
    d = x.shape[-1]
    c = _split(d)
    return (x * x).reshape(*x.shape[:-1], c, d // c).sum(-1).sum(-1, keepdim=True) / d


class TinyOps(object):
    linear = staticmethod(_linear)
    mean_sq = staticmethod(_mean_sq)
    asarray = staticmethod(_asarray)
    numpy = staticmethod(lambda t: t.numpy())
    zeros = staticmethod(lambda shape: Tensor.zeros(*shape, dtype=dtypes.float32).contiguous())
    cat = staticmethod(lambda xs, axis: xs[0].cat(*xs[1:], dim=axis))
    stack = staticmethod(lambda xs, axis: Tensor.stack(*xs, dim=axis))
    reshape = staticmethod(lambda x, shape: x.reshape(tuple(int(d) for d in shape)))
    transpose = staticmethod(lambda x, perm: x.permute(perm))
    sum = staticmethod(lambda x, axis, keepdims=False: x.sum(axis, keepdim=keepdims))
    mean = staticmethod(lambda x, axis, keepdims=False: x.mean(axis, keepdim=keepdims))
    max = staticmethod(lambda x, axis: x.max(axis))
    exp = staticmethod(lambda x: x.exp())
    tanh = staticmethod(lambda x: x.tanh())
    sqrt = staticmethod(lambda x: x.sqrt())
    relu = staticmethod(lambda x: x.relu())
    sigmoid = staticmethod(lambda x: x.sigmoid())
    silu = staticmethod(lambda x: x.silu())
    softplus = staticmethod(lambda x: x.softplus())
    rsqrt = staticmethod(lambda x: x.rsqrt())
    gelu = staticmethod(lambda x, tanh: x.gelu(approximate="tanh" if tanh else "none"))
    mish = staticmethod(lambda x: x.mish())
    clip = staticmethod(_clip)
    softmax = staticmethod(lambda x, axis: x.softmax(axis))
    einsum = staticmethod(lambda spec, a, b: Tensor.einsum(spec, a, b))
    take = staticmethod(lambda table, ids: table[_asarray(np.asarray(ids, np.int32))])
    topk = staticmethod(lambda x, k: x.topk(k, -1))
    one_hot = staticmethod(lambda idx, n: idx.one_hot(n).cast(dtypes.float32))
    take_along_last = staticmethod(lambda x, idx: x.gather(-1, idx))
    where_pos = staticmethod(lambda c, a, b, thr=0.5: (c > thr).where(a, b))
    realize = staticmethod(lambda x: x.realize())


class UCoreModel(UCoreForward):
    B = TinyOps


def load_model(run_dir):
    return UCoreModel(run_dir)
