"""Which exotic activations reduce to SiLU (and are wired into ACT_ORDER) and which
genuinely resist. swigluoai folds exactly (wired); mish approximates with a bounded floor (above the
calibrated ACT_FLOOR_MAX, so not wired);
situ and xIELU do not reduce."""
import pytest

torch = pytest.importorskip("torch")


def test_swigluoai_folds_into_silu_exact():
    from act.exotic_activations import swigluoai_folds_into_silu
    assert swigluoai_folds_into_silu() < 1e-12          # g*sigmoid(1.702g)=SiLU(1.702g)/1.702


def test_situ_reaches_silu_only_in_the_limit():
    from act.exotic_activations import situ_limit_to_silu
    assert situ_limit_to_silu(4.0) > 1.0                # Kimi-K3's trained beta=4: far from SiLU
    assert situ_limit_to_silu(256.0) < 1e-2             # only converges as beta -> inf


def test_mish_silu_is_a_bounded_floor():
    from act.exotic_activations import mish_silu_floor
    assert 0.05 < mish_silu_floor() < 0.2               # a real bounded approx (like GeLU's 0.014)


def test_xielu_does_not_reduce():
    from act.exotic_activations import xielu_positive_vs_silu
    assert xielu_positive_vs_silu() > 1.0               # quadratic branch diverges from SiLU


def test_xielu_is_the_real_hf_function():
    # our xIELU == transformers' XIELUActivation (default init) to fp64 rounding; the bf16
    # default parameters round alpha_p / alpha_n to 0.8004 / 0.8009.
    pytest.importorskip("transformers")
    from act.exotic_activations import xielu_matches_hf
    assert xielu_matches_hf(torch.float64) < 1e-12
    assert xielu_matches_hf(torch.float32) < 1e-4


def test_act_order_wires_only_the_exact_fold():
    # swigluoai (exact fold) reduces to SiLU; mish (floor 0.009), situ and xIELU (floor 0.135) are
    # floors above the calibrated ACT_FLOOR_MAX (act/act_swap_agreement.py: a SiLU-cluster swap is
    # not high-agreement on real models), so walls
    from graph.model_graph import ACT_ORDER, REACH
    assert "SiLU" in REACH["act"].get("swigluoai", set())
    assert "SiLU" not in REACH["act"].get("mish", {"mish"})
    assert "SiLU" not in REACH["act"].get("situ", {"situ"})     # incomparable
    assert not any(h == "SiLU" for h, _, _ in ACT_ORDER.get("xielu", []))
