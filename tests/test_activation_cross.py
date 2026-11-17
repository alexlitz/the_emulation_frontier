"""The activation cross-emulation structure over EVERY activation in the zoo, analytically
(single-op floors): ReLU is the bottom; {SiLU,GeLU,mish,swigluoai} is the swish cluster
(swigluoai == SiLU exactly); {relu2, xIELU} is a separate quadratic cluster, incomparable
to swish — which is *why* xIELU doesn't fold.

Run:  python -m pytest tests/test_activation_cross.py -q
"""
from act.activation_cross import ACTS, floor


def test_relu_is_the_bottom():
    # every smooth swish emulates ReLU near-exactly (the beta->inf limit)
    for h in ("SiLU", "GeLU", "mish", "swigluoai"):
        assert floor(ACTS[h], ACTS["ReLU"]) < 0.02, h
    # ...but ReLU is a poor single-op host for a smooth curve -> needs a network
    assert floor(ACTS["ReLU"], ACTS["SiLU"]) > 0.1


def test_swish_cluster_is_interchangeable():
    assert floor(ACTS["SiLU"], ACTS["GeLU"]) < 0.05
    assert floor(ACTS["GeLU"], ACTS["SiLU"]) < 0.05
    assert floor(ACTS["mish"], ACTS["SiLU"]) < 0.15


def test_swigluoai_is_silu_exactly():
    # swigluoai(x) = SiLU(1.702x)/1.702 -> a scaled SiLU, exact both ways
    assert floor(ACTS["swigluoai"], ACTS["SiLU"]) < 0.02
    assert floor(ACTS["SiLU"], ACTS["swigluoai"]) < 0.02


def test_quadratic_family_with_the_real_xielu():
    # xIELU here is the real HF XIELUActivation at its default init (act/activation_cross.py).
    # xIELU hosts relu2 near-exactly (its negative branch is ~linear far out)...
    assert floor(ACTS["xielu"], ACTS["relu2"]) < 0.01
    # ...but relu2 does NOT host xIELU exactly: the nonlinear negative branch leaves 0.166.
    assert 0.15 < floor(ACTS["relu2"], ACTS["xielu"]) < 0.18
    # on the coarse grid floor() the swish hosts look far from xIELU (0.4-1.3)...
    assert floor(ACTS["xielu"], ACTS["SiLU"]) > 0.3
    assert floor(ACTS["SiLU"], ACTS["xielu"]) > 0.3


def test_minimax_floor_is_much_lower_than_the_grid():
    # ...but floor()'s (c,d) grid is coarse and its (a,e,f) are least squares. The true
    # single-op minimax (floor_minimax) is much lower in several cells -- the quadratic /
    # swish "wall" is mostly a grid artifact.
    from act.activation_cross import floor_minimax
    assert floor_minimax(ACTS["SiLU"], ACTS["xielu"]) < 0.15            # grid: 0.71
    assert floor_minimax(ACTS["SiLU"], ACTS["relu2"]) < 0.25            # grid: 0.69
    assert floor_minimax(ACTS["swigluoai"], ACTS["SiLU"]) < 1e-4        # exact fold (grid: 0.006)
    assert floor_minimax(ACTS["xielu"], ACTS["SiLU"]) > 0.3             # this direction stays far
