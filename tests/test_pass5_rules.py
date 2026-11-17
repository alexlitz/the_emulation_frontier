"""Round 2, pass 5: the config rules, the rescues and the refutations, ported into graph/model_graph.emul_matrix.

One test per port, on small synthetic configs read by classify / dims_of (or a point edited in one slot / dim), each
also checking the port's leave-one-out switch (model_graph.P5_OFF) where the switch is what decides the pair."""
import copy
import contextlib

import numpy as np
import pytest

from graph import model_graph as G


def _llama(**kw):
    c = dict(model_type="llama", architectures=["LlamaForCausalLM"], hidden_size=256, num_hidden_layers=4,
             num_attention_heads=4, num_key_value_heads=4, intermediate_size=512, vocab_size=1000,
             max_position_embeddings=2048, rms_norm_eps=1e-5, hidden_act="silu", tie_word_embeddings=False,
             rope_theta=10000.0)
    c.update(kw)
    return c


def _pt(cfg, name="x"):
    sl = G.classify(cfg)
    assert sl, cfg
    return {"slots": sl, "dims": G.dims_of(cfg), "members": [name]}


def _edit(p, slots=None, **dims):
    q = copy.deepcopy(p)
    q["slots"].update(slots or {})
    q["dims"].update(dims)
    return q


def E(host, target, mode="payable"):
    return G.emulates(host, target, mode=mode)


@contextlib.contextmanager
def off(*ports):
    G.P5_OFF.update(ports)
    try:
        yield
    finally:
        G.P5_OFF.difference_update(ports)


@contextlib.contextmanager
def r3_off(*specs):
    """Round-3 keys switched off for the block (model_graph.R3_OFF): the pass-5 rule a round-3 key replaced."""
    added = [s for s in specs if s not in G.R3_OFF]
    for s in added:
        G.R3_OFF.add(s)
    try:
        yield
    finally:
        for s in added:
            G.R3_OFF.discard(s)


BIG = dict(hidden_size=1024, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=8, intermediate_size=4096)


# --- A1: the window lever ------------------------------------------------------------------------------------------

def test_window_lever_sliding_host_plays_a_full_target():
    t = _pt(_llama(), "llama")
    h = _pt(_llama(model_type="mistral", architectures=["MistralForCausalLM"], sliding_window=512, **BIG), "mistral")
    assert h["slots"]["window"] == "local" and h["dims"]["win_free"] == 1
    assert E(h, t) and E(h, t, "exact")                       # free config: exact
    with off("window"):
        assert not E(h, t)                                     # pass 4: only the host's full layers count
    # a family whose code is not checked keeps the pass-4 rule
    r = _edit(h, win_free=0)
    assert not E(r, t)


def test_window_free_target_on_a_fixed_window_host_is_played_full():
    """The transitivity-forced rule: a window-free target stands for its full-window self on a fixed-window host."""
    t = _pt(_llama(model_type="mistral", architectures=["MistralForCausalLM"], sliding_window=512), "mistral")
    h = _edit(_pt(_llama(model_type="mistral", architectures=["MistralForCausalLM"], sliding_window=512, **BIG)),
              win_free=0)
    assert not E(h, t)
    assert E(_edit(h, win_free=1), t)


# --- A2: partial rotary fraction, softcap values, norm on / off flags --------------------------------------------

def test_fraction_free_host_raises_its_rotary_fraction():
    t = _pt(_llama(hidden_size=512), "llama")                   # full RoPE, 128 rotated dims per head
    h = _pt(_llama(model_type="phi3", architectures=["Phi3ForCausalLM"], partial_rotary_factor=0.5, **BIG), "phi3")
    assert h["slots"]["pos"] == "partial_rope" and h["dims"]["rot_frac_free"] == 1
    assert E(h, t)
    with off("partial"):
        assert not E(h, t)                                      # its table drives 64 of its 128 dims


def test_softcap_value_free_host_drops_its_cap_exactly():
    t = _pt(_llama(), "llama")
    gem = _edit(t, slots={"softcap": "attn"}, softcap_free=1, hidden=2048)
    assert E(gem, t, "exact")
    assert not E(_edit(gem, softcap_free=0), t, "exact")        # the large-c setting is a regime (payable only)
    assert E(_edit(gem, softcap_free=0), t, "payable")


def test_qk_norm_config_flag_switches_off_exactly():
    t = _pt(_llama(), "llama")
    h = _edit(t, slots={"qknorm": "head_ln"}, qk_switch=1, hidden=2048)
    assert E(h, t, "exact")
    assert not E(_edit(h, qk_switch=0), t, "payable")           # a LayerNorm qk-norm: no eps lever either


# --- A3: the rot gate under config-only rules ---------------------------------------------------------------------

def test_fixed_rotary_host_plays_only_fixed_targets():
    t = _pt(_llama(), "llama")
    assert t["dims"]["rope_pairs_free"] == 1
    fixed = _edit(_pt(_llama(**BIG)), rope_pairs_free=0)
    assert not E(fixed, t)                                     # a pairs-free target: any schedule, which theta cannot form
    with off("rot_config"):
        assert E(fixed, t)                                     # pass 4: the buffer was taken as settable
    tf = _edit(t, rope_pairs_free=0)                           # a fixed, default-schedule target of the same dims
    assert E(fixed, tf)
    assert not E(fixed, _edit(tf, rope_kind="yarn@x"))         # a scaled schedule the host's code does not compute


def test_fixed_full_rope_host_has_no_zero_frequency_slot():
    nope = _edit(_pt(_llama()), slots={"pos": "nope"}, rope_sched=(), rope_cap=())
    fixed = _edit(_pt(_llama(**BIG)), rope_pairs_free=0)
    # was `assert not E(fixed, nope)` (payable) -- round 3, nope_rope_limit: a fixed host whose rotary reads linear rope
    # scaling (rope_lin, the Llama code: ROPE_INIT_FUNCTIONS["linear"]) plays a NoPE target as a LIMIT (factor -> inf,
    # Lean rope_scale_limit_identity): payable, still never exact; a host that reads no linear scaling stays a wall
    assert not E(fixed, nope, "exact")                         # Lean rope_theta_cannot_reach_nope
    assert fixed["dims"]["rope_lin"] == 1 and E(fixed, nope)
    assert not E(_edit(fixed, rope_lin=0), nope)
    G.R3_OFF.add("nope_rope_limit")
    try:
        off_fixed = _edit(_pt(_llama(**BIG)), rope_pairs_free=0)
        assert off_fixed["dims"]["rope_lin"] == 0 and not E(off_fixed, nope)
    finally:
        G.R3_OFF.discard("nope_rope_limit")
    assert E(_edit(fixed, rope_pairs_free=1), nope)
    assert E(_edit(fixed, rope_pairs_free=1), nope, "exact")   # the buffer write (longrope): exact


# --- B1: the eps lever -----------------------------------------------------------------------------------------------

def test_eps_lever_qk_norm_host_plays_a_plain_target():
    t = _pt(_llama(), "llama")
    h = _pt(_llama(model_type="qwen3", architectures=["Qwen3ForCausalLM"], head_dim=64, **BIG), "qwen3")
    assert h["slots"]["qknorm"] == "head" and h["dims"]["eps_lever"] == 1
    assert E(h, t) and not E(h, t, "exact")                    # payable, a limit
    with off("eps_qknorm"):
        assert not E(h, t)
    assert not E(t, _edit(t, slots={"qknorm": "head"}))         # the target side stays a wall


def test_eps_lever_sandwich_post_norm_and_v_norm():
    t = _pt(_llama(), "llama")
    h = _edit(_pt(_llama(**BIG)), slots={"feats": "post_norm+vnorm"}, eps_lever=1)
    assert E(h, t)
    with off("eps_feats"):
        assert not E(h, t)
    assert not E(_edit(h, slots={"feats": "post_only"}), t)     # a post-only block is not switched off


def test_mla_host_plays_an_mqa_or_nope_target():
    mqa = _pt(_llama(num_key_value_heads=1, hidden_size=256, num_attention_heads=4), "mqa")
    mla = _pt(dict(model_type="deepseek_v2", architectures=["DeepseekV2ForCausalLM"], hidden_size=2048,
                   num_hidden_layers=8, num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512,
                   q_lora_rank=None, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
                   intermediate_size=8192, vocab_size=1000, max_position_embeddings=2048, rms_norm_eps=1e-6,
                   hidden_act="silu", rope_theta=10000.0, n_routed_experts=None), "dsv2")
    assert mla["slots"]["kv"] == "MLA"
    assert E(mla, mqa)
    with off("eps_mla_plain"):
        assert not E(mla, mqa)
    gqa = _pt(_llama(num_key_value_heads=2), "gqa")             # two rotating KV heads: 2 x 64 > the 64-dim key
    assert not E(mla, gqa)


# --- B2: the start-token rules ---------------------------------------------------------------------------------------

def _gpt2(**kw):
    c = dict(model_type="gpt2", architectures=["GPT2LMHeadModel"], n_embd=256, n_layer=4, n_head=4, n_positions=1024,
             vocab_size=1000, activation_function="gelu_new", layer_norm_epsilon=1e-5, tie_word_embeddings=True)
    c.update(kw)
    return c


def test_learned_abs_target_on_a_rope_host_pays_the_start_token_charges():
    t = _pt(_gpt2(), "gpt2")
    assert t["slots"]["pos"] == "learned_abs" and t["dims"]["pos_table"] == 1024
    neox = dict(model_type="gpt_neox", architectures=["GPTNeoXForCausalLM"], num_attention_heads=4,
                intermediate_size=2048, vocab_size=1000, max_position_embeddings=2048, rotary_pct=0.25,
                use_parallel_residual=False, hidden_act="gelu_new", layer_norm_eps=1e-5, tie_word_embeddings=False)
    ok = _pt(dict(neox, hidden_size=266, num_hidden_layers=6), "neox")     # LayerNorm host: +10 hidden, +2 layers
    assert ok["slots"]["norm"] == "LayerNorm" and ok["slots"]["block"] == "sequential"
    assert E(ok, t) and E(ok, t, "exact")
    assert not E(_edit(ok, hidden=265), t)
    assert not E(_edit(ok, eff_layers=5, param_layers=5, kFU_eff=5, kFU_param=5, attn_eff=5, attn_param=5,
                       full_eff=5, full_param=5), t)
    assert not E(_edit(ok, ffn_total=1024 + 9, ffn_active=1024 + 9), t)   # T + 1 + 10 erasers (vanilla host)
    with off("bos_labs"):
        assert not E(ok, t)


def test_sink_target_on_a_sink_free_host_with_a_q_bias():
    t = _pt(_llama(use_attention_sink=True), "sink")
    assert t["slots"]["sink"] == "yes"
    qwen = _pt(_llama(model_type="qwen2", architectures=["Qwen2ForCausalLM"], hidden_size=264, num_attention_heads=4,
                      head_dim=66, intermediate_size=512), "qwen2")
    assert qwen["dims"]["q_bias"] == 1
    assert E(qwen, t) and E(qwen, t, "exact")
    assert not E(_edit(qwen, q_bias=0), t)                     # a k bias alone cancels in the softmax
    assert not E(_edit(qwen, head_dim=65), t)                  # a whole zero-frequency pair
    assert not E(_edit(qwen, hidden=256), t)                   # + the marker's channel
    with off("bos_sink"):
        assert not E(qwen, t)


def test_plain_target_on_a_sink_host_is_a_limit():
    t = _pt(_llama(), "plain")
    h = _pt(_llama(use_attention_sink=True, **BIG), "sink")
    assert E(h, t, "payable") and not E(h, t, "exact")


# --- B3 / B4 / B7: activations -----------------------------------------------------------------------------------------

def test_activation_width_fit():
    t = _pt(_llama(), "silu")                                  # SiLU target, ffn 512
    h = _edit(_pt(_llama(**BIG)), slots={"act": "GeLU"})        # a GeLU host: SiLU <- GeLU at k = 3
    assert E(h, t) and not E(h, t, "exact")
    assert not E(_edit(h, ffn_total=3 * 512 - 1, ffn_active=3 * 512 - 1), t)
    assert E(_edit(h, ffn_total=3 * 512, ffn_active=3 * 512), t)
    with off("act_width"):
        assert not E(h, t)


def test_gelu_erf_tanh_both_ways_approximate():
    t = _edit(_pt(_llama()), slots={"act": "GeLU"})
    h = _edit(_pt(_llama(**BIG)), slots={"act": "GeLU_tanh"})
    assert E(h, t) and not E(h, t, "exact")
    assert E(_edit(h, slots={"act": "GeLU"}), _edit(t, slots={"act": "GeLU_tanh"}))


def test_swigluoai_target_needs_a_host_up_bias():
    t = _edit(_pt(_llama()), slots={"act": "swigluoai"})
    h = _pt(_llama(**BIG))
    assert not E(h, t)
    assert E(_edit(h, ffn_bias=1), t)
    with off("oai_bias"):
        assert E(h, t)


# --- B5: looped hosts -----------------------------------------------------------------------------------------------

def test_untied_target_on_a_looped_host():
    t = _pt(_llama(), "untied")
    h = _edit(_pt(_llama(**BIG)), slots={"loop": "looped"})
    assert not E(h, t)                                         # any looped host but a plain one: a wall
    nb = _edit(h, loop_plain=1, loop_norm=1)
    assert E(nb, t) and not E(nb, t, "exact")
    d = t["dims"]["hidden"]
    assert not E(_edit(nb, ffn_active=512 + 2 * d + 2 + 1, ffn_total=512 + 2 * d + 2 + 1), t)  # + 2d + 2 + 2
    assert E(_edit(nb, ffn_active=512 + 2 * d + 4, ffn_total=512 + 2 * d + 4), t)
    with off("loop"):
        assert E(h, t)


# --- B6: pre-norm rescues and the LayerNorm fold -----------------------------------------------------------------------

def test_layernorm_beta_target_on_an_rms_host_needs_an_lm_head_bias():
    t = _edit(_pt(_llama()), slots={"norm": "LayerNorm", "feats": "bias_out"}, ln_beta=1)
    h = _edit(_pt(_llama(**BIG)), slots={"feats": "bias_out"})
    assert not E(h, t)
    assert E(_edit(h, lm_head_bias=1), t) and E(_edit(h, lm_head_bias=1), t, "exact")
    with off("ln_beta"):
        assert E(h, t)


def test_compensator_credits_a_measured_family_only():
    t = _edit(_pt(_llama()), slots={"norm": "LayerNorm", "feats": "bias_out"}, ln_beta=1, rms_ok=1)
    h = _edit(_pt(_llama(**BIG)), slots={"feats": "bias_out"})
    # round 3, batch 3b (spot_comp_removed): the compensator credits nothing -- its construction is measured only by
    # the scripts of verify/prenorm_rescue/ (no host construction builds it); was E(h, t). The pass-5 rule is kept behind the key:
    assert not E(h, t) and not G.p5_port_live("comp_ln")
    with r3_off("spot_comp_removed"):
        assert E(h, t) and not E(h, t, "exact")                # GLU host: k = 8 units, +2 hidden, +1 layer
        assert not E(h, _edit(t, rms_ok=0))
        # the transitivity-forced closure: a credited target that hosts an unmeasured one loses the credit
        u = _edit(t, rms_ok=0, hidden=128)
        ok = G.emul_matrix([u, t, h])
        assert ok[0, 1] and not ok[1, 2] and not ok[0, 2]


def test_parallel_target_on_a_sequential_host_by_the_compensator():
    t = _edit(_pt(_llama()), slots={"block": "parallel"}, rms_ok=1)
    h = _pt(_llama(hidden_size=1024, num_hidden_layers=4, intermediate_size=512 + 8 * 256))
    # round 3, batch 3b (spot_comp_removed): no compensator credit (was E(h, t)); the pass-5 rule behind the key, and
    # with w1_par_units the host also pays the 2 d identity-pair units of parallel_sub_seq_2x
    assert not E(h, t) and not E(_edit(h, ffn_total=10 ** 6, ffn_active=10 ** 6), t)
    with r3_off("spot_comp_removed", "w1_par_units"):
        assert E(h, t)                                         # 2d + 1 hidden, k d units per layer
        assert not E(_edit(h, ffn_total=512 + 8 * 256 - 1, ffn_active=512 + 8 * 256 - 1), t)
        assert not E(h, _edit(t, rms_ok=0))
    with r3_off("spot_comp_removed"):
        assert not E(h, t)
        assert E(_edit(h, ffn_total=512 + 10 * 256, ffn_active=512 + 10 * 256), t)     # + 2 d_target units


# --- the relation stays reflexive and transitive on a mixed sample -------------------------------------------------

def test_pass5_sample_is_a_preorder():
    pts = [_pt(_llama(), "a"), _pt(_llama(**BIG), "b"), _pt(_gpt2(), "c"),
           _pt(_llama(model_type="qwen3", architectures=["Qwen3ForCausalLM"], head_dim=64, **BIG), "d"),
           _pt(_llama(model_type="mistral", architectures=["MistralForCausalLM"], sliding_window=512, **BIG), "e"),
           _pt(_llama(use_attention_sink=True), "f")]
    for mode in ("exact", "payable", "mixgap", "capacity"):
        ok = G.emul_matrix(pts, mode=mode)
        assert np.diag(ok).all()
        S = ok.astype(int)
        assert not (((S @ S) > 0) & ~ok).any(), mode
