"""The capacity trades the frontier relies on are only counted if a microcosm build confirms
them. Key guard: FFN/MoE width<->depth is exact ONLY with input-carrying (a width cost), and
the naive residual-stack split diverges — so 'wide FFN = free deep stack' must NOT be assumed.

Run:  python -m pytest tests/test_fungibility.py -q
"""
from graph.fungibility_probe import (ffn_subset_to_depth, ffn_width_to_depth,
                                     heads_width_to_depth, moe_experts_to_depth,
                                     residual_rotation)


def test_ffn_width_to_depth_needs_input_carrying():
    naive, carried, width_cost = ffn_width_to_depth()
    assert naive > 0.1, f"naive split should diverge, got {naive}"   # reads accumulated residual
    assert carried < 1e-9, f"input-carried split must be exact, got {carried}"
    assert width_cost > 0                                            # the trade costs reserved width


def test_moe_experts_to_depth_needs_input_carrying():
    naive, carried = moe_experts_to_depth()
    assert naive > 0.1, f"naive expert-stack should diverge, got {naive}"
    assert carried < 1e-9, f"input-carried expert-stack must be exact, got {carried}"


def test_heads_to_depth_carry_is_the_projection_dim():
    # spreading heads over depth is exact with input-carrying, and a head reads only its
    # d_head-wide projection -> carrying one head (3*dh) beats carrying the whole input (d)
    # for a subset (< H/3 heads). Low-projection ops are depth-fungible cheaply.
    naive, carried, carry_full, carry_per_head = heads_width_to_depth()
    assert naive > 0.1, f"naive head-stack should diverge, got {naive}"
    assert carried < 1e-9, f"input-carried head-stack must be exact, got {carried}"
    assert carry_per_head < carry_full                       # a head's read is cheaper than full x


def test_ffn_subset_carry_is_k_scalars():
    # spreading k of H FFN units costs only k carried channels (each unit's read is one scalar),
    # not a flat 2x width -- exact.
    err, k = ffn_subset_to_depth(d=8, H=32, k=4)
    assert err < 1e-9, f"subset spread must be exact, got {err}"
    assert k == 4                                            # carry = the number of spread units


def test_residual_rotation_cancels_and_costs_attention_rank():
    # the residual is tied (only added to); a rotation R@x is f(x)=(R-I)@x (the -I cancels the
    # old residual), realized by a self-attention layer of rank(R-I) -- exact for linear attn.
    err, rank = residual_rotation(d=8)
    assert err < 1e-9, f"rotation must be exact (linear attn), got {err}"
    assert 1 <= rank <= 8                                    # cost = rank(R-I) head-dims, one layer
