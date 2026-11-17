"""The reductions at MODEL precision (fp32), not just fp64.

Reuses the CLI's dtype-parametrized demos (graph.reduce.DEMOS). Every exact identity
must hold at fp32 (model precision) to ~fp32 epsilon, and stay bit-exact at fp64 (so
the suite still certifies exactness -- fp32 alone can't tell an exact identity from a
1e-6 approximation). The one limit (ReLU = beta->inf Swish) is checked to be an
approximation, not exact, in both precisions.

Run:  pytest tests/test_fp32.py
"""
import pytest

torch = pytest.importorskip("torch")

from graph.reduce import DEMOS

EXACT = ["norm", "kv", "ffn", "window", "block", "sink", "loop"]


@pytest.mark.parametrize("name", EXACT)
def test_reduction_holds_at_fp32(name):
    # fp32 machine-eps ~1.2e-7; the two computation paths round differently, so an
    # exact identity lands ~1e-7..1e-5. Several (kv/ffn/block/loop) stay bit-exact.
    assert DEMOS[name](torch.float32) < 1e-4


@pytest.mark.parametrize("name", EXACT)
def test_reduction_exact_at_fp64(name):
    # fp64 certifies EXACTNESS: a true identity is ~1e-14, distinguishable from any
    # real approximation. This is why fp64 stays the default for the identity proofs.
    assert DEMOS[name](torch.float64) < 1e-8


def test_relu_limit_is_approx_not_exact():
    # ReLU = beta->inf Swish is a LIMIT: at beta=256 it is still ~1e-3, and identical
    # in fp32 and fp64 (a math gap, not fp rounding) -- so it must NOT read as exact.
    e64, e32 = DEMOS["act"](torch.float64), DEMOS["act"](torch.float32)
    assert 1e-6 < e64 < 1e-2
    assert abs(e64 - e32) < 1e-5      # the gap is arithmetic, not precision
