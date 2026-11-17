"""Validate the GRAPH rules: the per-slot partial orders, the emulation relation
(reflexive / transitive / size-aware), the capacity math, and the classifier.

Run:  pytest tests/test_graph_rules.py
"""
import itertools

import pytest

from graph.model_graph import (ACT_ORDER, BLOCK_ORDER, FFN_ORDER, KV_ORDER, LOOP_ORDER,
                               NORM_ORDER, POS_ORDER, QKNORM_ORDER, SLOTS, SOFTCAP_ORDER,
                               WINDOW_ORDER, _lm_config, classify, dims_of, edge_cost, emulates,
                               reach)

ORDERS = {"pos": POS_ORDER, "kv": KV_ORDER, "ffn": FFN_ORDER, "norm": NORM_ORDER,
          "act": ACT_ORDER, "block": BLOCK_ORDER, "window": WINDOW_ORDER,
          "qknorm": QKNORM_ORDER, "softcap": SOFTCAP_ORDER, "loop": LOOP_ORDER}


# --- per-slot partial orders are valid DAGs with correct transitive closure ---

# slots whose order has intentional 2-cycles: LayerNorm <-> RMSNorm (fold / mirror), the qk-norm and
# softcap regime edges (the same residual both ways); the activation relation is the thresholded
# one-hop floor table (tests/test_payable_edges.py), not a partial order. pos: partial_rope <= rope
# (free) and rope <= partial_rope (rot-gated: a host that rotates enough dims per head,
# hf/partial_rope_host.py).
CYCLIC = {"norm", "qknorm", "softcap", "act", "pos"}


@pytest.mark.parametrize("name", [n for n in ORDERS if n not in CYCLIC])
def test_order_is_acyclic(name):
    R = reach(ORDERS[name])
    for a in R:
        for b in R[a]:
            if a != b:
                assert a not in R[b], f"cycle {a}<->{b} in {name}"


@pytest.mark.parametrize("name", list(ORDERS))
def test_reach_is_transitive_and_reflexive(name):
    R = reach(ORDERS[name])
    for a in R:
        assert a in R[a]
        for b in R[a]:
            for c in R[b]:
                assert c in R[a], f"transitivity broken {a}->{b}->{c} in {name}"


# --- emulation relation: a synthetic model builder ---------------------------

def mk(norm="RMSNorm", pos="rope", kv="MHA", ffn="glu", act="SiLU",
       block="sequential", window="full", qknorm="no", softcap="no",
       mixer="softmax", sink="no", crosslayer_kv="no", mtp="no", gen="ar", loop="untied",
       **dims):
    base = dict(hidden=1024, layers=12, heads=16, inter=4096, experts=0, kv_cap=1024,
                q_cap=1024, qheads=16, head_dim=64, ffn_active=4096, ffn_total=4096,
                param_layers=12, eff_layers=12, vocab=32000)
    base.update(dims)
    # unless overridden, the two depth dims track `layers` (plain, non-looped stack)
    if "param_layers" not in dims:
        base["param_layers"] = base["layers"]
    if "eff_layers" not in dims:
        base["eff_layers"] = base["layers"]
    if "head_dim" not in dims:                     # Q/K head_dim = q_cap / qheads
        base["head_dim"] = base["q_cap"] // max(base["qheads"], 1)
    # ZERO_EDGES layer counts: every layer is own-KV full attention unless said otherwise (a pure
    # SSM / linear stack has none)
    for k, ref in (("attn_eff", "eff_layers"), ("attn_param", "param_layers"),
                   ("full_eff", "eff_layers"), ("full_param", "param_layers")):
        if k not in dims:
            base[k] = 0 if mixer in ("ssm", "linear") else base[ref]
    return {"name": "m", "slots": dict(norm=norm, pos=pos, kv=kv, ffn=ffn, act=act,
            block=block, window=window, qknorm=qknorm, softcap=softcap, mixer=mixer,
            sink=sink, crosslayer_kv=crosslayer_kv, mtp=mtp, gen=gen, loop=loop), "dims": base}


def test_emulates_reflexive():
    for a in [mk(), mk(kv="MLA", kv_cap=512), mk(ffn="moe", experts=8, ffn_total=32768),
              mk(norm="LayerNorm", pos="learned_abs", act="GeLU")]:
        assert emulates(a, a)


def test_emulates_transitive():
    A = mk(kv="MQA", kv_cap=256, ffn="glu", hidden=512, layers=8, ffn_total=2048, ffn_active=2048)
    B = mk(kv="GQA", kv_cap=512, ffn="glu", hidden=1024, layers=12, ffn_total=4096, ffn_active=4096)
    C = mk(kv="MHA", kv_cap=2048, ffn="moe", experts=8, hidden=2048, layers=24,
           ffn_total=32768, ffn_active=8192)
    assert emulates(B, A) and emulates(C, B) and emulates(C, A)


# --- attention is SIZE-based (kv_cap), not categorical ------------------------

def test_bigger_kv_cap_emulates_smaller_any_variant():
    # MQA / GQA / MHA: size-based. MLA is off the chain under the headline (its per-token latent norm:
    # attn/mla_norm_agreement.py) -- see test_mla_is_a_wall_both_ways
    mha, gqa, mqa = mk(kv="MHA", kv_cap=2048), mk(kv="GQA", kv_cap=512), mk(kv="MQA", kv_cap=128)
    assert emulates(mha, gqa) and emulates(gqa, mqa) and emulates(mha, mqa)
    assert not emulates(gqa, mha)      # small kv_cap cannot emulate big
    mla = mk(kv="MLA", kv_cap=512)
    assert not emulates(mha, mla) and not emulates(mla, mk(kv="MHA", kv_cap=256))
    assert emulates(mha, mla, mode="capacity")          # the capacity view still sizes it


def test_kv_heads_are_compared_by_count():
    """A target group needs ceil(g_t / g_h) host KV heads (its K, V replicated) -- unless a WIDER host head packs
    several target KV heads side by side (round 2, pass 2: test_kv_head_packing). Softmax normalises per QUERY
    head, so packing is exact without RoPE; what stops it for a RoPE target is frequency distinctness (each block
    needs its own copy of the target's frequencies). History: SmolLM2-135M (9 q, 3 kv, head_dim 64; kv_cap 192) was
    NOT hosted by Qwen2.5-3B (16 q, 2 kv, head_dim 128) before the packing construction; now each Qwen2.5-3B KV
    head holds two SmolLM2 KV heads (2 x 64 <= 128, 8 query heads; Qwen2's rotary takes a longrope table)."""
    import json
    from graph.model_graph import classify, dims_of, emulates
    cfg = {r["id"]: r["config"] for r in json.load(open("graph/hub_10k.json"))
           if r["id"] in ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-3B")}
    pt = lambda c: {"slots": classify(c), "dims": dims_of(c)}
    smol, qwen = pt(cfg["HuggingFaceTB/SmolLM2-135M"]), pt(cfg["Qwen/Qwen2.5-3B"])
    assert (smol["dims"]["kv_heads"], smol["dims"]["q_per_kv"]) == (3, 3)
    assert (qwen["dims"]["kv_heads"], qwen["dims"]["q_per_kv"]) == (2, 8)
    assert qwen["dims"]["kv_cap"] >= smol["dims"]["kv_cap"] and qwen["dims"]["head_dim"] >= smol["dims"]["head_dim"]
    assert emulates(qwen, smol, mode="capacity")                  # 2 host KV heads x 2 packed >= 3 (was: not)
    # synthetic (equal head_dim: no packing): MHA hosts any grouping with enough heads; a GQA host needs
    # ceil(g_t / g_h) heads per group
    assert emulates(mk(kv="MHA", kv_heads=16, q_per_kv=1), mk(kv="GQA", kv_heads=4, q_per_kv=4))
    assert not emulates(mk(kv="MHA", kv_heads=15, q_per_kv=1), mk(kv="GQA", kv_heads=4, q_per_kv=4))
    assert emulates(mk(kv="GQA", kv_heads=8, q_per_kv=2), mk(kv="GQA", kv_heads=4, q_per_kv=4))
    assert not emulates(mk(kv="GQA", kv_heads=4, q_per_kv=2), mk(kv="GQA", kv_heads=4, q_per_kv=4))


def test_mla_is_a_wall_both_ways():
    """KV_ORDER: a plain host folds the MLA latent norm to a constant at best (measured not payable on
    DeepSeek-V2-Lite); an MLA host cannot switch its latent norm off and has one shared rotary key head.
    MLA <- MLA needs latents at least as wide and the query latent norm in both or neither."""
    t = mk(kv="MLA", kv_cap=576, kv_latent=512, q_latent=1536, mla_rope=64)
    assert emulates(mk(kv="MLA", kv_cap=576, kv_latent=512, q_latent=1536, mla_rope=64, layers=24), t)
    assert not emulates(mk(kv="MLA", kv_cap=576, kv_latent=512, q_latent=0, mla_rope=64, layers=24), t)
    assert not emulates(mk(kv="MLA", kv_cap=576, kv_latent=256, q_latent=1536, mla_rope=64, layers=24), t)
    assert not emulates(mk(kv="MHA", kv_cap=99999, layers=24), t)
    assert not emulates(mk(kv="MLA", kv_cap=99999, kv_latent=99999, layers=24), mk(kv="GQA", kv_cap=128))
    # round 2, pass 3: the latent norms' eps is hard-coded (1e-6), so a wider host latent is exact only as an integer
    # multiple of the target's (the latent stored m times); the nope and v head widths compared on their own
    big = dict(kv="MLA", kv_cap=1088, q_latent=1536, mla_rope=64, layers=24)
    assert emulates(mk(kv_latent=1024, **big), t) and not emulates(mk(kv_latent=768, **big), t)
    assert not emulates(mk(kv="MLA", kv_cap=576, kv_latent=512, q_latent=2048, mla_rope=64, layers=24), t)
    tw = mk(kv="MLA", kv_cap=576, kv_latent=512, q_latent=1536, mla_rope=64, mla_nope=128, mla_v=128)
    assert emulates(mk(kv="MLA", kv_cap=576, kv_latent=512, q_latent=1536, mla_rope=64, mla_nope=192, mla_v=256,
                       layers=24), tw)
    assert not emulates(mk(kv="MLA", kv_cap=576, kv_latent=512, q_latent=1536, mla_rope=64, mla_nope=128, mla_v=64,
                           layers=24), tw)


def test_mla_small_kv_cannot_emulate_larger_gqa():
    # MLA is "fancy" but its tiny kv_cap gates it -- by design
    big_mla = mk(kv="MLA", kv_cap=576, hidden=7168, layers=61)
    gqa = mk(kv="GQA", kv_cap=1024, hidden=4096, layers=32)
    assert not emulates(big_mla, gqa)  # kv_cap 576 < 1024


# --- FFN is size-based both directions, with a depth cost for dense->MoE ------

def test_moe_emulates_dense_with_active_width():
    dense = mk(ffn="glu", ffn_total=4096, ffn_active=4096)
    moe = mk(ffn="moe", experts=8, ffn_total=32768, ffn_active=4096)
    assert emulates(moe, dense)        # route to k experts, active width >= dense


def test_dense_needs_double_depth_and_all_experts_for_moe():
    # (a top-1 MoE with a unit gate: the hard select is exact, Lean dense_hard_select; top-k > 1 or a gate
    # times the router score is not priced)
    moe = mk(ffn="moe", experts=8, layers=12, ffn_total=32768, ffn_active=8192, moe_unit_top1=1)
    shallow = mk(ffn="glu", layers=12, ffn_total=40000, ffn_active=40000)
    deep = mk(ffn="glu", layers=24, ffn_total=40000, ffn_active=40000)
    assert not emulates(shallow, moe)  # enough width but not 2x depth
    assert emulates(deep, moe)         # 2x depth + holds all experts
    assert not emulates(deep, mk(ffn="moe", experts=8, layers=12, ffn_total=32768, ffn_active=8192))   # top-k


def test_dense_cannot_emulate_moe_without_total_capacity():
    moe = mk(ffn="moe", experts=8, layers=12, ffn_total=32768, ffn_active=8192, moe_unit_top1=1)
    deep_narrow = mk(ffn="glu", layers=24, ffn_total=16000, ffn_active=16000)
    assert not emulates(deep_narrow, moe)   # 2x depth but can't hold all experts


def test_vanilla_glu_not_mutually_emulable():
    # C1 regression: GLU >= vanilla (free), but vanilla needs depth for the gate. Round 3, batch 3b: GLU >= vanilla
    # only with an up bias (the up branch held at 1: spot_vanilla_glu_bias; was: free on any GLU host), and a vanilla
    # host plays no GLU target at any depth (spot_glu_vanilla: the 2x-depth route carries act(g) across a pre-norm; was:
    # the 24-layer vanilla emulated it)
    from graph import model_graph as G
    vanilla = mk(ffn="vanilla", layers=12)
    glu = mk(ffn="glu", layers=12)
    assert emulates(mk(ffn="glu", layers=12, ffn_bias=1, bias_up=1), vanilla)   # GLU with an up bias emulates vanilla
    assert not emulates(glu, vanilla)                                          # ... a bias-free one does not
    assert not emulates(vanilla, glu)                                          # vanilla: no gate
    assert not emulates(mk(ffn="vanilla", layers=24), glu)                     # ... not even at 2x depth
    for key, host, target in (("spot_vanilla_glu_bias", glu, vanilla), ("spot_glu_vanilla", mk(ffn="vanilla", layers=24), glu)):
        G.R3_OFF.add(key)
        try:
            assert emulates(host, target)                                      # the pass-4 rule behind the key
        finally:
            G.R3_OFF.discard(key)


def test_nontrivial_mutual_subset_exists_in_principle():
    # distinct KV variants at IDENTICAL capacity mutually emulate (preorder, not
    # antisymmetric). Among real models this never happens (sizes never match exactly).
    mha = mk(kv="MHA")
    gqa = mk(kv="GQA")   # same dims, different KV variant
    assert emulates(mha, gqa) and emulates(gqa, mha)     # mutual
    assert not (emulates(mha, mk(kv="GQA", hidden=1025))  # any size mismatch breaks it
                and emulates(mk(kv="GQA", hidden=1025), mha))


def test_tied_kv_needs_extra_capacity_to_emulate_mla():
    # M4 / TransMLA: GQA at EQUAL kv_cap cannot emulate MLA; needs the penalty factor
    # (the capacity view: under the headline MLA <-> GQA is a wall, test_mla_is_a_wall_both_ways)
    mla = mk(kv="MLA", kv_cap=512)
    gqa_equal = mk(kv="GQA", kv_cap=512)
    gqa_big = mk(kv="GQA", kv_cap=1024)
    assert not emulates(gqa_equal, mla, mode="capacity")     # equal cache is not enough
    assert emulates(gqa_big, mla, mode="capacity")           # 2x cache clears the penalty
    assert emulates(mk(kv="MLA", kv_cap=512), mla)   # MLA-to-MLA at equal cache is fine


# --- depth is a HARD constraint: width cannot substitute for it --------------

def test_depth_not_substitutable_by_width():
    deep = mk(layers=32, hidden=1024, kv_cap=1024, ffn_total=4096, ffn_active=4096)
    wide_shallow = mk(layers=16, hidden=99999, kv_cap=99999,
                      ffn_total=999999, ffn_active=999999)
    assert not emulates(wide_shallow, deep)   # fewer layers -> cannot emulate


# --- looped / recurrent-depth: a restriction; params <-> serial-compute trade ----

def test_untied_unrolls_a_looped_stack():
    # looped = block of 2 layers x recurrence 8 -> eff depth 16 from 2 unique layers.
    looped = mk(loop="looped", layers=2, param_layers=2, eff_layers=16)
    untied16 = mk(loop="untied", layers=16, param_layers=16, eff_layers=16)
    untied8 = mk(loop="untied", layers=8, param_layers=8, eff_layers=8)
    assert emulates(untied16, looped)      # deep enough to unroll all 16 steps
    assert not emulates(untied8, looped)   # too shallow: can't unroll 16 forward steps


def test_looped_cannot_fake_extra_distinct_layers():
    # a looped model has FEW unique layers; without per-iteration conditioning it
    # cannot emulate an untied model needing more distinct layer-functions.
    looped = mk(loop="looped", layers=2, param_layers=2, eff_layers=32)
    untied = mk(loop="untied", layers=8, param_layers=8, eff_layers=8)
    assert not emulates(looped, untied)    # 2 unique layers < 8 distinct layer-functions
    # ...and round 2, pass 5: even a looped stack whose block is deep enough cannot, unless its later iterations can
    # be silenced (a plain looped host, Nanbeige4.2: the counter construction, +2 channels, 2d + 2 units)
    fat_loop = mk(loop="looped", layers=8, param_layers=8, eff_layers=64)
    assert not emulates(fat_loop, untied)
    d = fat_loop["dims"]
    plain = dict(fat_loop, dims=dict(d, loop_plain=1, hidden=d["hidden"] + 2,
                                     ffn_total=d["ffn_total"] + 2 * d["hidden"] + 2,
                                     ffn_active=d["ffn_active"] + 2 * d["hidden"] + 2))
    assert emulates(plain, untied)


def test_classify_and_dims_looped_recurrent():
    cfg = dict(model_type="huginn_raven", hidden_size=1024, num_attention_heads=16,
               rms_norm_eps=1e-5, hidden_act="silu", num_hidden_layers=8,
               n_layers_in_recurrent_block=4, mean_recurrence=8, rope_theta=1e4,
               intermediate_size=4096)
    c = classify(cfg)
    assert c["loop"] == "looped"
    d = dims_of(cfg)
    assert d["param_layers"] == 4          # unique layers = the recurrent block
    assert d["eff_layers"] == 4 * 8        # forward depth = block x recurrence
    # a plain stack has loop == "untied" and the two depths coincide
    plain = dims_of(CONFIGS["llama"])
    assert plain["param_layers"] == plain["eff_layers"] == 32
    assert classify(CONFIGS["llama"])["loop"] == "untied"


# --- norm: RMSNorm >= LayerNorm (fold), directional --------------------------

def test_norm_order_layernorm_reduces_to_rmsnorm():
    # LayerNorm folds INTO RMSNorm (whole-model fold, free); the reverse is the [h; -h] mirror at 2x
    # hidden (tests/test_composed_edges.py), which the headline charges as width.
    assert edge_cost(NORM_ORDER, "LayerNorm", "RMSNorm") == (["fold"], True)
    assert edge_cost(NORM_ORDER, "RMSNorm", "LayerNorm") == (["width"], True)
    assert emulates(mk(norm="RMSNorm"), mk(norm="LayerNorm"))
    assert not emulates(mk(norm="LayerNorm"), mk(norm="RMSNorm"))
    assert emulates(mk(norm="LayerNorm", hidden=2048), mk(norm="RMSNorm"))
    assert emulates(mk(norm="LayerNorm"), mk(norm="RMSNorm"), mode="capacity")


# --- positional: absolute vs relative is incomparable ------------------------

def test_learned_abs_and_rope_incomparable_in_order():
    # absolute and relative PE are incomparable (neither reduces to the other): a wall with no
    # currency, so the headline blocks both ways; only the secondary (capacity) ignores it.
    assert edge_cost(POS_ORDER, "rope", "learned_abs") is None
    assert edge_cost(POS_ORDER, "learned_abs", "rope") is None
    assert not emulates(mk(pos="rope"), mk(pos="learned_abs"))
    assert not emulates(mk(pos="learned_abs"), mk(pos="rope"))
    assert emulates(mk(pos="rope"), mk(pos="learned_abs"), mode="capacity")


def test_alibi_reduces_to_rope():
    assert edge_cost(POS_ORDER, "alibi", "rope") is not None    # a fixed RoPE pair IS ALiBi
    assert edge_cost(POS_ORDER, "rope", "alibi") is None
    # the frozen pair needs a SPARE head-dim pair at inv_freq = 0 and a causal target (Lean alibi_pair_odd)
    assert emulates(mk(pos="rope", head_dim=66), mk(pos="alibi", head_dim=64))
    assert not emulates(mk(pos="rope", head_dim=64), mk(pos="alibi", head_dim=64))       # no spare pair
    assert not emulates(mk(pos="rope", head_dim=66, gen="diffusion"), mk(pos="alibi", head_dim=64, gen="diffusion"))
    assert not emulates(mk(pos="alibi"), mk(pos="rope"))


# --- activation: SiLU emulates ReLU (limit); GeLU / a ReLU host are floors above the calibrated
# threshold (act/act_swap_agreement.py: not high-agreement on real models) -------------

def test_relu_gelu_reduce_to_silu():
    assert edge_cost(ACT_ORDER, "ReLU", "SiLU") == (["mag"], False)     # beta -> inf limit
    assert edge_cost(ACT_ORDER, "GeLU", "SiLU") is None                 # minimax 0.012: a wall
    assert edge_cost(ACT_ORDER, "SiLU", "ReLU") is None                 # a ReLU host, minimax 0.137
    assert emulates(mk(act="SiLU"), mk(act="ReLU"))
    assert not emulates(mk(act="ReLU"), mk(act="SiLU"))
    assert not emulates(mk(act="SiLU"), mk(act="GeLU"))
    assert not emulates(mk(act="SiLU"), mk(act="ReLU"), mode="exact")


# --- small config slots ------------------------------------------------------

def test_block_sequential_more_general_both_directions():
    # parallel is a restriction of sequential (drop the attn->MLP dependency). Both
    # simulate each other, but sequential is more general: it emulates parallel at
    # 2x WIDTH (same depth); parallel emulates sequential only at 2x DEPTH.
    par = mk(block="parallel", hidden=1024, layers=12)
    seq = mk(block="sequential", hidden=1024, layers=12)
    # sequential emulates parallel -> width (same depth)
    assert not emulates(seq, par)                                        # equal width: no
    # 2x width: the norm-free construction (round 3, batch 3b: 2 d width + 2 d MLP units per block, Lean
    # parallel_sub_seq_2x -- was "(L+1) d, parallel_sub_seq_wide"); under a stock pre-norm the carry fails (~1e-2,
    # block/prenorm_carry.py): not payable (PAR_CARRY_PAYABLE), kept by mixgap at ffn + 2 d units (w1_par_units; was
    # the equal FFN, 4096)
    wide = dict(block="sequential", hidden=2048, layers=12, ffn_total=4096 + 2048, ffn_active=4096 + 2048)
    assert not emulates(mk(**wide), par)
    assert emulates(mk(**wide), par, mode="mixgap")
    assert not emulates(mk(block="sequential", hidden=2048, layers=12), par, mode="mixgap")
    # parallel emulates sequential -> depth
    assert not emulates(par, seq)                                        # equal depth: no
    assert emulates(mk(block="parallel", hidden=1024, layers=24), seq)   # 2x depth: yes


def test_local_reduces_to_full():
    assert edge_cost(WINDOW_ORDER, "local", "full") is not None   # full masks -> local
    assert edge_cost(WINDOW_ORDER, "full", "local") is None
    assert emulates(mk(window="full"), mk(window="local"))
    assert emulates(mk(window="local"), mk(window="full"))        # payable (depth) at equal size


def test_qknorm_softcap_order_and_payable():
    # qk-norm / softcap are REGIME edges with the same residual both ways (tests/test_payable_edges.py):
    # approximate in both directions, refused by the exact slice.
    # (round 2, pass 2: qk-norm split into variants, softcap into attn / final / no; the old yes is "head" / "attn")
    for order, v, cur in ((QKNORM_ORDER, "head", "approx"), (SOFTCAP_ORDER, "attn", "cap-large")):
        assert edge_cost(order, v, "no") == (["approx"], False)
        assert edge_cost(order, "no", v) == ([cur], False)
    assert not emulates(mk(qknorm="head"), mk(qknorm="no"), mode="exact")
    # MEASURED (block/fold_agreement.py, attn/regime_agreement.py): the qk-norm regime is NOT payable either way
    # (Qwen3-0.6B / OLMo-2-1B with the norms folded to constants: top-1 15% / 27%)
    assert not emulates(mk(qknorm="no"), mk(qknorm="head"))
    assert not emulates(mk(qknorm="head"), mk(qknorm="no"))
    assert emulates(mk(qknorm="no"), mk(qknorm="head"), mode="capacity")
    # softcap (REGIME_PAYABLE, attn/regime_agreement.py): a capped host plays a plain target (c set large); a plain
    # host does not play an attention-capped target (Gemma-2-9B 75% identical). History: before round 2 pass 2 the
    # softcap regime was payable both ways (Gemma-2-2B uncapped: 99.7% / 92%).
    assert emulates(mk(softcap="attn"), mk(softcap="no"))
    assert not emulates(mk(softcap="no"), mk(softcap="attn"))


def test_softcap_split():
    """attn / final / none (round 2, pass 2; REGIME_PAYABLE): final <-> none payable (the final cap is monotone:
    argmax-exact; a final-capped host sets c large), a capped host plays any target, an attention-capped target
    needs an attention-capped host; the exact slice refuses every regime crossing."""
    no, fin, att = mk(softcap="no"), mk(softcap="final"), mk(softcap="attn")
    assert emulates(no, fin) and emulates(fin, no)
    assert emulates(att, fin) and emulates(att, no)
    assert not emulates(fin, att) and not emulates(no, att)
    assert not emulates(no, fin, mode="exact") and not emulates(att, no, mode="exact")
    assert emulates(no, att, mode="capacity")


def test_qknorm_variants():
    """The qk-norm variants are different functions (round 2, pass 2: QKNORM_VARIANTS): a target needs the
    same variant, or one that contains it by a gain / bias setting (QKNORM_NEST); a qk-norm host never plays a
    plain target (the regime, measured not payable) and vice versa."""
    from graph.model_graph import QKNORM_NEST, QKNORM_VARIANTS
    for v in QKNORM_VARIANTS:
        assert emulates(mk(qknorm=v), mk(qknorm=v))
        assert not emulates(mk(qknorm=v), mk(qknorm="no")) and not emulates(mk(qknorm="no"), mk(qknorm=v))
    assert not emulates(mk(qknorm="head"), mk(qknorm="proj")) and not emulates(mk(qknorm="proj"), mk(qknorm="head"))
    assert not emulates(mk(qknorm="head"), mk(qknorm="head_post"))         # Hunyuan: the gain after RoPE
    assert not emulates(mk(qknorm="head_post"), mk(qknorm="head"))
    assert not emulates(mk(qknorm="head"), mk(qknorm="head_ln"))           # RMS vs LayerNorm
    # gain 1 on a gained per-head norm is the weight-free one (the rms is rotation-invariant): exact
    assert emulates(mk(qknorm="head"), mk(qknorm="head_free"), mode="exact")
    assert emulates(mk(qknorm="head_post"), mk(qknorm="head_free"), mode="exact")
    assert not emulates(mk(qknorm="head_free"), mk(qknorm="head"))
    assert QKNORM_NEST["head"] == ("head_distinct",)


def test_qknorm_variant_classification():
    """Variants read from the code of each family (hub snapshot configs)."""
    import json
    want = {"Qwen/Qwen3-0.6B": "head", "allenai/OLMo-2-0425-1B": "proj", "allenai/OLMoE-1B-7B-0924": "proj",
            # (d16: was "head_free" (round 3, batch 2a: qknorm_fixed_gain reads its qk_sharpen_scale 1.2 on q and k,
            # a logit gain 1.44 -- head_free_gain))
            "tencent/Hy-MT2-7B": "head_post", "jkminder/d16_optimized_537m_seed5": "head_free_gain",
            "adept/persimmon-8b-base": "ln_shared_bias", "alpindale/c4ai-command-r-plus-GPTQ": "head_ln",
            "pfnet/plamo-2-1b": "head_distinct", "MiniMaxAI/MiniMax-M2.5": "proj", "sarvamai/sarvam-105b": "no",
            "google/gemma-2-9b": "no", "tencent/Hy3": "head"}
    cfg = {r["id"]: r["config"] for r in json.load(open("graph/hub_10k.json")) if r["id"] in want}
    for name, v in want.items():
        assert classify(cfg[name])["qknorm"] == v, name
    sc = {r["id"]: r["config"] for r in json.load(open("graph/hub_10k.json"))
          if r["id"] in ("google/gemma-2-9b", "EssentialAI/rnj-1-instruct", "Qwen/Qwen3-0.6B")}
    assert classify(sc["google/gemma-2-9b"])["softcap"] == "attn"
    assert classify(sc["EssentialAI/rnj-1-instruct"])["softcap"] == "final"
    assert classify(sc["Qwen/Qwen3-0.6B"])["softcap"] == "no"


def test_post_only_split_from_sandwich():
    """post-only (no pre-norm: OLMo-2 / 3, EXAONE-4) and sandwich (pre + post: Gemma-2 / 3 / 4) are different
    features (round 2, pass 2), each a wall both ways."""
    import json
    cfg = {r["id"]: r["config"] for r in json.load(open("graph/hub_10k.json"))
           if r["id"] in ("allenai/OLMo-2-0425-1B", "google/gemma-2-9b", "LGAI-EXAONE/EXAONE-4.0-32B")}
    f = lambda n: set(classify(cfg[n])["feats"].split("+"))
    assert "post_only" in f("allenai/OLMo-2-0425-1B") and "post_norm" not in f("allenai/OLMo-2-0425-1B")
    assert "post_only" in f("LGAI-EXAONE/EXAONE-4.0-32B")
    assert "post_norm" in f("google/gemma-2-9b") and "post_only" not in f("google/gemma-2-9b")
    def fm(feats):
        m = mk()
        m["slots"]["feats"] = feats
        return m
    assert not emulates(fm("post_norm"), fm("post_only"))
    assert not emulates(fm("post_only"), fm("post_norm"))
    assert not emulates(fm("post_only"), fm(""))
    assert emulates(fm("post_only"), fm("post_only"))


def test_alibi_slope_schemes():
    """ALiBi slopes are computed in code (round 2, pass 2): BLOOM / MPT / Baichuan share one slope set per head
    count (to fp32 rounding), nested in the head count; Falcon's are bf16-rounded; NoPE <- ALiBi only where a
    config value switches the bias off (Falcon alibi, the remote MPT's alibi_bias_max)."""
    from graph.model_graph import alibi_nests, alibi_slopes
    assert alibi_nests((16, 8, False), (32, 8, False)) and alibi_nests((40, 8, False), (48, 8, False))
    assert not alibi_nests((32, 8, False), (16, 8, False))
    assert not alibi_nests((32, 8, True), (32, 8, False)) and not alibi_nests((32, 8, False), (32, 8, True))
    assert alibi_nests((32, 8, True), (64, 8, True))                      # bf16 rounding keeps the nesting
    assert alibi_nests((8, 8, True), (8, 8, False))                       # n <= 8: powers of two, bf16-exact
    # the transformers functions agree with alibi_slopes (BLOOM's set, MPT's reorder, Falcon's bf16)
    import torch
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor
    from transformers.models.falcon.modeling_falcon import build_alibi_tensor as falcon_alibi
    from transformers.models.mpt.modeling_mpt import build_mpt_alibi_tensor
    for n in (12, 32, 40, 112):
        bl = sorted(build_alibi_tensor(torch.ones(1, 2), n, torch.float32)[:, 0, 1].tolist())
        a = build_mpt_alibi_tensor(n, 2)
        mp = sorted((a[:, 0, 1] - a[:, 0, 0]).tolist())
        fa = sorted(falcon_alibi(torch.ones(1, 2), n, torch.float32)[:, 0, 1].tolist())
        for ref, bf in ((bl, False), (mp, False), (fa, True)):
            assert all(abs(x - y) <= 1e-6 * y for x, y in zip(ref, alibi_slopes(n, 8, bf)))
    al = lambda n, bf=0, off=0, **k: mk(pos="alibi", alibi_heads=n, alibi_max=8, alibi_bf16=bf, alibi_off=off, **k)
    assert emulates(al(32), al(16)) and not emulates(al(16), al(32))
    assert not emulates(al(32, bf=1, off=1), al(32)) and not emulates(al(32), al(32, bf=1, off=1))
    assert not emulates(al(32), mk(pos="nope"))                           # BLOOM: slopes fixed in code
    assert emulates(al(32, bf=1, off=1), mk(pos="nope"))                  # Falcon: alibi=False
    assert not emulates(al(32), al(16, off=1))                            # transitivity: a switchable target
    assert emulates(al(32, off=1), al(16, off=1))                         #   needs a switchable host


def test_kv_head_packing():
    """One host KV head of head_dim D holds m target KV heads of head_dim d side by side when m d <= D
    (round 2, pass 2; verify/levers/constructions.py C: 4e-16), each block with >= 1 of its query heads. With
    RoPE the host table must REPEAT the target's frequencies once per block (Lean KVPack.kvPack_rope_iff; the
    default theta schedule never does, kvPack_default_not_exact): only a host whose rotary takes a longrope config
    (per-pair factors: rope_pairs_free), and rope_fits on m x the target's rotated dims. NoPE targets: free."""
    from graph.model_graph import kv_hosts_needed
    assert kv_hosts_needed(4, 1, 2, 1) == 4 and kv_hosts_needed(4, 1, 2, 2) == 2     # MHA on GQA, 2 per head
    assert kv_hosts_needed(4, 4, 2, 1) == 8 == kv_hosts_needed(4, 4, 2, 2)          # q = 1 is replication
    assert kv_hosts_needed(5, 2, 3, 3) == 4                                          # mixed runs: 10 queries, 3 / head
    t = mk(kv="MHA", kv_heads=16, q_per_kv=1, qheads=16, head_dim=64, kv_cap=1024, q_cap=1024, rope_sched=(64,))
    h = lambda hd, kv="GQA", pos="rope", **k: mk(kv=kv, pos=pos, kv_heads=8, q_per_kv=2, qheads=16, head_dim=hd,
                                                 kv_cap=8 * hd, q_cap=16 * hd, rope_sched=(hd,), **k)
    assert emulates(h(128, rope_pairs_free=1), t) and not emulates(h(96, rope_pairs_free=1), t)   # 2 x 64 <= 128
    assert not emulates(h(128), t)                                       # default schedule: frequencies distinct
    assert emulates(h(128), t, mode="capacity")
    nope_t = mk(pos="nope", kv="MHA", kv_heads=16, q_per_kv=1, qheads=16, head_dim=64, kv_cap=1024, q_cap=1024)
    assert emulates(h(128, rope_pairs_free=1), nope_t)                   # zero frequencies, as unpacked
    assert not emulates(h(128), nope_t)        # round 2, pass 5: a fixed full rotary has no zero-frequency slot
    assert not emulates(h(128, rope_pairs_free=1, rope_cap=(64,)), t)   # the table holds one block's pairs
    assert not emulates(h(128, kv="MLA", rope_pairs_free=1), t)          # MLA hosts never pack


# --- EVERY categorical reduction the graph uses is exercised through emulates() ---
# (grep-verified gap: these edges fed the frontier but no test asserted them.)

@pytest.mark.parametrize("slot,special,general", [
    # partial_rope <-> rope is a 2-cycle (rope <= partial_rope is rot-gated): tests/test_partial_rope_host.py
    ("pos", "alibi", "rope"),                # fixed RoPE pair (mag)
    ("pos", "nope", "learned_abs"),            # no-pos <= absolute (zero the embeddings)
    ("pos", "nope", "alibi"),                  # no-pos <= alibi (zero the slopes; emul_matrix gates it on alibi_off)
    ("pos", "nope", "partial_rope"),           # no-pos <= partial rope (zero the freqs)
    ("pos", "alibi", "additive_bias"),         # alibi is a specific additive bias
    ("pos", "t5_relbias", "additive_bias"),    # T5 relative bias is an additive bias
    ("act", "ReLU", "SiLU"),                   # beta -> inf limit; a ReLU host's 0.137 floor is a wall
    ("window", "local", "full"),               # mask
    ("mixer", "ssm", "softmax"),               # near-gap
    ("mixer", "linear", "softmax"),            # near-gap
    ("mixer", "hybrid", "softmax"),            # near-gap
    ("crosslayer_kv", "yes", "no"),            # untie shared KV (weight restriction)
    ("gen", "ar", "diffusion"),                # causal mask (AR <= diffusion), one-way
])
def test_slot_order_special_reduces_to_general(slot, special, general):
    # The per-slot TYPE order still exists (it drives reduce/construct's per-slot plans):
    # `special` reduces INTO `general`, not the reverse (the one-way edges; the 2-cycles -- norm,
    # qk-norm, softcap, the activation floors -- are tested on their own above).
    order = dict(SLOTS)[slot]
    assert edge_cost(order, special, general) is not None, f"{special} should reduce to {general}"
    assert edge_cost(order, general, special) is None, f"{general} should NOT reduce to {special}"


def test_emulates_is_payable_and_capacity_is_the_secondary():
    # the HEADLINE emulates(): a wall is crossed only where it has a currency, in its direction;
    # mode="capacity" (every type wall free) makes these mutual at equal size. gen stays one-way.
    for slot, a, b in [("act", "ReLU", "SiLU"), ("pos", "alibi", "rope")]:
        A, B = mk(**{slot: a}, head_dim=64), mk(**{slot: b}, head_dim=66)
        assert emulates(B, A) and not emulates(A, B), f"{slot}: payable one way only"
        assert emulates(B, A, mode="capacity") and emulates(mk(**{slot: b}, head_dim=64), A, mode="capacity")
    # the softmax <- SSM near-gap is measured NOT payable (attn/mixer_gap_agreement.py): a wall in the
    # headline, one-way under mode="mixgap" (the old free near-gap), free under capacity
    A, B = mk(mixer="ssm"), mk(mixer="softmax")
    assert not emulates(B, A) and not emulates(A, B)
    assert emulates(B, A, mode="mixgap") and not emulates(A, B, mode="mixgap")
    assert emulates(B, A, mode="capacity") and emulates(A, B, mode="capacity")
    assert not emulates(mk(gen="ar"), mk(gen="diffusion"), mode="capacity")   # one-way even if free
    assert emulates(mk(gen="diffusion"), mk(gen="ar"), mode="capacity")


def test_auxiliary_slots_emulate_both_directions():
    # mtp (extra predict heads) is AUXILIARY: free to add or drop either way, so it never differentiates
    # two models (excluded from the dedup signature so an aux head never splits a class).
    A, B = mk(mtp="no"), mk(mtp="yes")
    assert emulates(B, A) and emulates(A, B)


def test_learned_sinks_are_one_way():
    # a sink host drops its sinks exactly (-inf); a plain host has no query-independent logit to form one
    # (attn/attention_sink.py's null token needs a constant query component), and dropping the sinks is
    # measured not payable (gpt-oss-tinystories: top-1 97.5%, 33% of continuations identical)
    A, B = mk(sink="no"), mk(sink="yes")
    assert emulates(B, A) and not emulates(A, B)


def test_gen_paradigm_is_one_way():
    # AR <= diffusion: a diffusion host applies the causal mask (attn/ar_diffusion.py, exact);
    # a causal host can never reproduce a bidirectional diffusion forward
    ar, diff = mk(gen="ar"), mk(gen="diffusion")
    assert not emulates(ar, diff)
    assert emulates(diff, ar)


def test_qheads_richness_gate():
    # query-head COUNT is a hard richness axis: can't fake H softmaxes with fewer -- but only at
    # EQUAL head_dim. (More heads at the SAME q_cap means NARROWER heads, which can't reproduce the
    # wider attention patterns -- that is the head_dim axis below.) So scale q_cap with qheads to
    # hold head_dim fixed; then more heads emulate fewer, not the reverse.
    few = mk(qheads=8, q_cap=1024)                 # head_dim 128
    many = mk(qheads=16, q_cap=2048)               # head_dim 128, twice the heads
    assert emulates(many, few)
    assert not emulates(few, many)


def test_head_dim_gate_narrow_cannot_emulate_wide():
    # Q/K head_dim is a hard axis: a wide head's attention pattern (score rank <= head_dim) can't
    # be reproduced by narrower heads (softmax blocks combining), so a smaller-head_dim model can't
    # emulate a wider-head one even with >= q_cap. (The V budget stays fungible, via kv_cap.)
    wide = mk(qheads=16, q_cap=2048)               # head_dim 128
    narrow = mk(qheads=16, q_cap=1024)             # head_dim 64, same head count
    assert emulates(wide, narrow)                  # wide heads emulate narrow
    assert not emulates(narrow, wide)              # narrow heads can't fake the wide pattern


def test_solver_fits_and_lists_the_plan():
    from graph.solve import solve
    A = mk(window="local", kv="GQA", kv_cap=1024, hidden=1024)      # target
    B = mk(window="full", kv="MHA", kv_cap=8192, hidden=2048)       # roomy host
    r = solve(A, B)
    assert r["feasible"]
    assert ("window", "local->full", "mask") in r["plan"]


def test_solver_reports_incomparable_obstruction():
    from graph.solve import solve
    r = solve(mk(pos="learned_abs"), mk(pos="rope"))               # absolute vs relative PE
    assert not r["feasible"] and not r["via_depth"]
    assert any(slot == "pos" for slot, _, _ in r["obstructions"])


def test_solver_finds_the_depth_dual_for_moe_into_dense():
    from graph.solve import solve
    # a MoE the dense host can't hold in WIDTH but can across DEPTH (E experts as layers)
    A = mk(ffn="moe", experts=8, ffn_total=131072, ffn_active=32768,
           layers=32, eff_layers=32, param_layers=32)
    B = mk(ffn="glu", ffn_total=24576, ffn_active=24576, hidden=8192,
           layers=80, eff_layers=80, param_layers=80)
    r = solve(A, B)
    assert not r["feasible"]              # the width model can't hold all experts
    assert r["via_depth"]                 # ...but the depth dual (8 masked layers) clears it
    assert r["ffn_dual"]["depth"]["ok"] and not r["ffn_dual"]["width"]["ok"]


def test_solver_minimal_upgrade_when_it_cannot_fit():
    from graph.solve import solve
    A = mk(kv_cap=8192, q_cap=8192, hidden=4096)
    B = mk(kv_cap=1024, q_cap=1024, hidden=1024)                    # too small everywhere
    r = solve(A, B)
    assert not r["feasible"]
    assert r["upgrade"]["kv_cap"] == 8192 - 1024                    # exact deficit reported


def test_vocab_axis_is_opt_in():
    # vocab/tokenizer capacity is OFF by default (reductions live in hidden space); a big
    # body with a small vocab emulates a small body with a huge vocab. Turn it ON and the
    # 256k-vocab target needs an emulator with >= its embedding/head rows.
    import graph.model_graph as G
    big_body = mk(vocab=32000, hidden=2048, layers=24, kv_cap=2048, q_cap=2048,
                  ffn_total=8192, ffn_active=8192)
    big_vocab = mk(vocab=256000, hidden=512, layers=8, kv_cap=512, q_cap=512,
                   ffn_total=2048, ffn_active=2048)
    assert G.USE_VOCAB is False
    assert emulates(big_body, big_vocab)          # off: vocab ignored
    G.USE_VOCAB = True
    try:
        assert not emulates(big_body, big_vocab)  # on: 32k < 256k embedding rows
        assert emulates(mk(vocab=256000, hidden=2048, layers=24, kv_cap=2048, q_cap=2048,
                           ffn_total=8192, ffn_active=8192), big_vocab)   # enough vocab: yes
    finally:
        G.USE_VOCAB = False


# --- classifier on known real configs ----------------------------------------

CONFIGS = {
    "llama": dict(model_type="llama", hidden_size=4096, num_attention_heads=32,
                  num_key_value_heads=8, rope_theta=1e4, rms_norm_eps=1e-5,
                  hidden_act="silu", intermediate_size=11008, num_hidden_layers=32),
    "gpt2": dict(model_type="gpt2", n_embd=768, n_head=12, activation_function="gelu_new",
                 layer_norm_epsilon=1e-5, n_layer=12),
    "bloom": dict(model_type="bloom", hidden_size=1024, n_head=16,
                  layer_norm_epsilon=1e-5, num_hidden_layers=24),
    "deepseek_v2": dict(model_type="deepseek_v2", hidden_size=2048, num_attention_heads=16,
                        num_key_value_heads=16, kv_lora_rank=512, qk_rope_head_dim=64,
                        qk_nope_head_dim=128, v_head_dim=128,
                        rope_theta=1e4, rms_norm_eps=1e-6, hidden_act="silu",
                        n_routed_experts=64, n_shared_experts=2, moe_intermediate_size=1408,
                        num_experts_per_tok=6, num_hidden_layers=27),
    "mixtral": dict(model_type="mixtral", hidden_size=4096, num_attention_heads=32,
                    num_key_value_heads=8, rope_theta=1e6, rms_norm_eps=1e-5, hidden_act="silu",
                    num_local_experts=8, num_experts_per_tok=2, intermediate_size=14336,
                    num_hidden_layers=32),
    "pythia": dict(model_type="gpt_neox", hidden_size=768, num_attention_heads=12,
                   rotary_pct=0.25, layer_norm_eps=1e-5, hidden_act="gelu",
                   use_parallel_residual=True, num_hidden_layers=12, intermediate_size=3072),
    "gemma2": dict(model_type="gemma2", hidden_size=2304, num_attention_heads=8,
                   num_key_value_heads=4, head_dim=256, rope_theta=1e4, rms_norm_eps=1e-6,
                   hidden_act="gelu_pytorch_tanh", sliding_window=4096,
                   attn_logit_softcapping=50.0, intermediate_size=9216, num_hidden_layers=26),
}
# (round 3, nonlm_heads: + the readout / input slots, an LM reading tokens: head lm, input tokens)
_D = dict(block="sequential", window="full", qknorm="no", softcap="no",   # small-config defaults
          mixer="softmax", sink="no", crosslayer_kv="no", mtp="no", gen="ar", loop="untied", head="lm", input="tokens")
_D2 = dict(mixer="softmax", sink="no", crosslayer_kv="no", mtp="no", gen="ar", loop="untied", head="lm",
           input="tokens")
_B = "bias_out+bias_qkv"          # FEATURES: biases fixed in code (GPT-2 / Bloom / GPT-NeoX)
EXPECT = {
    "llama": dict(norm="RMSNorm", pos="rope", kv="GQA", ffn="glu", act="SiLU", feats="", **_D),
    # GPT2Config keeps tie_word_embeddings=True; no activation_function key = the class default gelu_new (tanh)
    "gpt2": dict(norm="LayerNorm", pos="learned_abs", kv="MHA", ffn="vanilla", act="GeLU_tanh",
                 feats=_B + "+tied_emb", **_D),
    # Bloom: bloom_gelu_forward is the tanh form (no key)
    "bloom": dict(norm="LayerNorm", pos="alibi", kv="MHA", ffn="vanilla", act="GeLU_tanh", feats=_B + "+tied_emb", **_D),
    "deepseek_v2": dict(norm="RMSNorm", pos="rope", kv="MLA", ffn="moe_shared", act="SiLU", feats="", **_D),
    "mixtral": dict(norm="RMSNorm", pos="rope", kv="GQA", ffn="moe", act="SiLU", feats="", **_D),
    # small-config detection:
    "pythia": dict(norm="LayerNorm", pos="partial_rope", kv="MHA", ffn="vanilla", act="GeLU",
                   block="parallel", window="full", qknorm="no", softcap="no", feats=_B, **_D2),
    # Gemma-2: no q / k norm (query_pre_attn_scalar is only the softmax scale), sandwich post-norms; its MLP reads
    # hidden_activation (class default gelu_pytorch_tanh), not the config's hidden_act
    "gemma2": dict(norm="RMSNorm", pos="rope", kv="GQA", ffn="glu", act="GeLU_tanh",
                   block="sequential", window="local", qknorm="no", softcap="attn",
                   feats="post_norm+tied_emb", **_D2),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_classify(name):
    assert classify(CONFIGS[name]) == EXPECT[name]


# --- dims_of capacity math ---------------------------------------------------

def test_dims_mla_kv_cap():
    d = dims_of(CONFIGS["deepseek_v2"])
    assert d["kv_cap"] == 512 + 64                       # kv_lora_rank + rope
    # M7: shared expert (n_shared=2) counts toward active + total FFN width
    assert d["ffn_total"] == 64 * 1408 + 2 * 1408        # E*ew + shared
    assert d["ffn_active"] == 6 * 1408 + 2 * 1408        # k*ew + shared


def test_dims_mla_q_cap_includes_rope():
    # M3: MLA query = n_heads * (qk_nope + qk_rope) = 16*(128+64) = 3072, NOT hidden=2048
    d = dims_of(CONFIGS["deepseek_v2"])
    assert d["q_cap"] == 16 * (128 + 64)


def test_dims_gpt2_ffn_fallback():
    d = dims_of(CONFIGS["gpt2"])
    assert d["ffn_total"] == 4 * 768                     # n_inner null -> 4*hidden
    assert d["kv_cap"] == 768                            # MHA: n_heads * head_dim


def test_dims_mixtral_expert_capacity():
    d = dims_of(CONFIGS["mixtral"])
    assert d["ffn_total"] == 8 * 14336
    assert d["ffn_active"] == 2 * 14336
    assert d["kv_cap"] == 8 * (4096 // 32)               # GQA: n_kv * head_dim


def test_dims_longcat_expert_width():
    # LongCat: ffn_hidden_size is the dense width; experts use expert_ffn_hidden_size, top-k is moe_topk
    d = dims_of({"hidden_size": 6144, "num_layers": 28, "num_attention_heads": 64,
                 "ffn_hidden_size": 12288, "expert_ffn_hidden_size": 2048,
                 "n_routed_experts": 512, "moe_topk": 12})
    assert d["ffn_total"] == 512 * 2048
    assert d["ffn_active"] == 12 * 2048


def test_dims_doge_single_neuron_experts():
    # Doge CDMoE: dense MLP + 16384 width-1 experts, not 16384 copies of the dense MLP
    d = dims_of({"model_type": "doge", "hidden_size": 512, "num_hidden_layers": 16,
                 "num_attention_heads": 4, "intermediate_size": 1024, "num_experts": 16384,
                 "num_cdmoe_experts": 16348, "num_experts_per_tok": 64})
    assert d["ffn_total"] == 16384 + 1024
    assert d["ffn_active"] == 64 + 1024


def test_dims_nemotron_nas_reads_block_configs():
    # DeciLM / Nemotron-NAS: GQA via n_heads_in_group, no-op blocks, per-block FFN widths
    blk = lambda g, mult: {"attention": {"n_heads_in_group": g, "no_op": g is None},
                           "ffn": {"ffn_mult": mult, "no_op": mult is None}}
    d = dims_of({"model_type": "nemotron-nas", "hidden_size": 768, "num_attention_heads": 12,
                 "num_hidden_layers": 4, "num_key_value_heads": None,
                 "block_configs": [blk(4, 2.0), blk(None, 1.0), blk(None, None), blk(6, 3.0)]})
    assert d["kv_cap"] == (12 // 4) * 64                      # widest per-block cache, not MHA
    assert d["ffn_total"] == 1536                             # 2*3.0*768/3 = 1536 (multiple of 256)
    assert d["ffn_units"] == 1024 + 512 + 1536                # summed over live blocks
    assert d["eff_layers"] == 3                               # the fully no-op block is skipped


def test_dims_unused_moe_fields_are_dense():
    # explicit off switches (Doge is_moe=false, use_moe=false) and a Nemotron-H layer list with
    # 'mlp' but no 'moe' layer: the expert counts are template defaults, the model is dense
    base = {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8,
            "intermediate_size": 1024, "num_experts": 16384, "num_experts_per_tok": 64}
    for flag in ({"model_type": "doge", "is_moe": False}, {"model_type": "x", "use_moe": False}):
        d = dims_of({**base, **flag})
        assert d["experts"] == 0 and d["ffn_total"] == d["ffn_active"] == 1024
    nh = {"model_type": "nemotron_h", "hidden_size": 3136, "num_attention_heads": 40,
          "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12544,
          "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 7688,
          "moe_shared_expert_intermediate_size": 7688, "n_shared_experts": 1, "mamba_num_heads": 96,
          "layers_block_type": ["linear_attention", "mlp", "full_attention", "mlp"]}
    assert classify(nh)["ffn"] == "vanilla" and dims_of(nh)["ffn_total"] == 12544   # ungated relu2 MLP
    moe = {**nh, "layers_block_type": ["mamba", "moe", "attention", "moe"]}
    assert classify(moe)["ffn"] == "moe_shared"
    # Nemotron-H's shared expert is moe_shared_expert_intermediate_size wide (n_shared_experts=1 is a count)
    assert dims_of({**moe, "moe_shared_expert_intermediate_size": 2 * 7688})["ffn_total"] == 8 * 7688 + 2 * 7688


def test_dims_nemotron_h_puzzle_per_block_moe():
    # Nemotron-H-Puzzle sets expert width / top-k / expert count PER MoE block; the top-level
    # intermediate_size (21504) is the dense template, not an expert width (was priced 512 x 21504)
    d = dims_of({"model_type": "nemotron_h_puzzle", "hidden_size": 4096, "num_attention_heads": 32,
                 "num_key_value_heads": 2, "head_dim": 128, "intermediate_size": 21504,
                 "n_routed_experts": 512, "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 5376,
                 "mamba_num_heads": 128, "layers_block_type": ["mamba", "moe", "attention", "moe"],
                 "block_configs": [{"block_type": "mamba"},
                                   {"block_type": "moe", "moe_intermediate_size": 1280, "num_experts_per_tok": 4},
                                   {"block_type": "attention"},
                                   {"block_type": "moe", "moe_intermediate_size": 2688, "num_experts_per_tok": 22,
                                    "n_routed_experts": 240}]})
    assert d["ffn_total"] == max(512 * 1280, 240 * 2688) + 5376
    assert d["ffn_active"] == 22 * 2688 + 5376
    assert d["ffn_units"] == (512 * 1280 + 5376) + (240 * 2688 + 5376)


def test_dims_widest_attention_layer():
    # heterogeneous stacks: the host must hold the widest layer's Q/K rank and cache
    g4 = dims_of({"model_type": "gemma4_text", "hidden_size": 5376, "num_hidden_layers": 60,
                  "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 256,
                  "global_head_dim": 512, "num_global_key_value_heads": 4, "intermediate_size": 21504})
    assert g4["head_dim"] == 512 and g4["q_cap"] == 32 * 512 and g4["kv_cap"] == 16 * 256
    plc = dims_of({"model_type": "gemma4_text", "hidden_size": 3840, "num_hidden_layers": 48,
                   "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
                   "per_layer_config": {"05": {"head_dim": 512, "num_key_value_heads": 1}}})
    assert plc["head_dim"] == 512 and plc["kv_cap"] == 8 * 256
    # dots3: sliding-window layers are MLA with a WIDER latent (1024) and qk head (192+64)
    dots = dims_of({"model_type": "dots3_note", "hidden_size": 5120, "num_hidden_layers": 46,
                    "num_attention_heads": 128, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
                    "qk_rope_head_dim": 64, "swa_num_attention_heads": 64, "swa_kv_lora_rank": 1024,
                    "swa_qk_nope_head_dim": 192, "swa_qk_rope_head_dim": 64})
    assert dots["kv_cap"] == 1024 + 64 and dots["head_dim"] == 256
    assert dots["q_cap"] == 128 * 192 and dots["qheads"] == 128


def test_classify_hybrid_from_full_attention_index():
    # no per-layer list, but full-attention layers given by index / period -> hybrid, not pure SSM
    lin = {"model_type": "new_gdn", "hidden_size": 1024, "num_hidden_layers": 8,
           "num_attention_heads": 8, "rms_norm_eps": 1e-6, "linear_num_value_heads": 16}
    assert classify(lin)["mixer"] == "ssm"
    assert classify({**lin, "full_attention_layers": [3, 7]})["mixer"] == "hybrid"
    assert classify({**lin, "full_attention_interval": 4})["mixer"] == "hybrid"
    assert classify({**lin, "mla_layers": [4, 8]})["mixer"] == "hybrid"            # Mini-K3
    assert classify({**lin, "full_attention_idx": [0, 2]})["mixer"] == "hybrid"     # PLaMo-2
    # softmax layers under family-specific names (Qwen3.8 sparse attention, Agnes global)
    for full in ("qwen_sparse_attention", "agnes_global_attention", "a"):   # 'a': Nemotron-Flash code
        assert classify({**lin, "layer_types": ["linear_attention", full]})["mixer"] == "hybrid"
    assert classify({**lin, "layer_types": ["agnes_delta_attention"] * 8})["mixer"] == "ssm"
    # an explicit all-linear layer list wins over a leftover interval field
    assert classify({**lin, "full_attention_interval": 4,
                     "layer_types": ["linear_attention"] * 8})["mixer"] == "ssm"


def test_classify_nemotron_relu2_norm_and_ungated_mlp():
    # Nemotron-H keeps the FFN activation under mlp_hidden_act (relu2) with no hidden_act, uses
    # NemotronHRMSNorm despite its layer_norm_epsilon key, and its dense MLP is up -> relu2 -> down
    nh = {"model_type": "nemotron_h", "hidden_size": 4096, "num_attention_heads": 32,
          "num_key_value_heads": 8, "intermediate_size": 21504, "layer_norm_epsilon": 1e-5,
          "mlp_hidden_act": "relu2", "mamba_hidden_act": "silu", "mamba_num_heads": 128,
          "hybrid_override_pattern": "M-M*-M-"}
    c = classify(nh)
    assert (c["act"], c["norm"], c["ffn"], c["mixer"]) == ("relu2", "RMSNorm", "vanilla", "hybrid")
    moe = classify({**nh, "n_routed_experts": 128, "num_experts_per_tok": 6,
                    "moe_shared_expert_intermediate_size": 3712, "hybrid_override_pattern": "MEM*EME"})
    assert (moe["act"], moe["ffn"]) == ("relu2", "moe_shared")
    # Nemotron-4 / Minitron: NemotronLayerNorm1P (key norm_eps) and an ungated relu2 MLP
    n4 = classify({"model_type": "nemotron", "hidden_size": 3072, "num_attention_heads": 24,
                   "num_key_value_heads": 8, "hidden_act": "relu2", "norm_eps": 1e-5,
                   "rope_theta": 1e4, "partial_rotary_factor": 0.5, "num_hidden_layers": 32})
    assert (n4["norm"], n4["ffn"], n4["act"]) == ("LayerNorm", "vanilla", "relu2")


def test_classify_hybrids_from_recurrent_layer_names():
    base = {"hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16,
            "num_key_value_heads": 4, "rms_norm_eps": 1e-5, "rope_theta": 1e6}
    # LFM2 / LFM3 gated short conv, Jet-Nemotron JetBlock, GKA: recurrent layers + softmax layers
    for mt, lt in (("lfm3", ["conv", "full_attention", "conv", "sliding_attention"]),
                   ("lfm2_moe", ["conv", "conv", "full_attention", "conv"]),
                   ("jet_nemotron", ["jet", "attn", "jet", "swa"]),
                   ("hybrid_qwen3", ["*", "GKA", "GKA", "*"])):
        c = classify({**base, "model_type": mt, "layer_types": lt})
        assert c["mixer"] == "hybrid" and c["kv"] == "GQA", mt
    assert classify({**base, "model_type": "lfm2"})["mixer"] == "hybrid"          # by type
    assert classify({**base, "model_type": "new", "layer_types": ["conv"] * 4})["mixer"] == "ssm"
    # softmax layer names that are NOT recurrent stay softmax
    for lt in (["sliding_attention", "full_attention"], ["deepseek_sparse_attention"] * 2,
               ["sliding_attention", "mlp"]):
        assert classify({**base, "model_type": "new", "layer_types": lt})["mixer"] == "softmax"
    # no layer list: Youtu linear_attention_pattern (0 = full), GPT-S3 GDN-per-softmax period,
    # PseudoBrain pb_hybrid_layers (a strict subset converted to linear attention)
    lin = {**base, "model_type": "new", "linear_num_value_heads": 16}
    assert classify({**lin, "linear_attention_pattern": [1, 1, 1, 0]})["mixer"] == "hybrid"
    assert classify({**lin, "linear_attention_pattern": [1, 1, 1, 1]})["mixer"] == "ssm"
    assert classify({**base, "model_type": "gpts3", "gdn_layers_per_attention": 3})["mixer"] == "hybrid"
    assert classify({**lin, "model_type": "qwen3", "pb_hybrid_layers": [3]})["mixer"] == "hybrid"


def test_dims_looped_depth_schemas():
    base = {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5, "rope_theta": 1e4, "intermediate_size": 5632}
    cases = [  # (extra config, param_layers, eff_layers)
        ({"model_type": "arloop", "num_hidden_layers": 72, "physical_layers": 24, "loop_count": 3}, 24, 72),
        ({"model_type": "kotodama", "num_layers": 72, "loop_start": 19, "loop_len": 20, "loop_times": 2}, 52, 72),
        ({"model_type": "qwen3loop", "num_hidden_layers": 28, "loop_middle_layers": 14, "num_loops": 2}, 28, 42),
        ({"model_type": "nanbeige", "num_hidden_layers": 22, "num_loops": 2}, 22, 44),
        ({"model_type": "iquestloopcoder", "num_hidden_layers": 80, "loop_num": 2}, 80, 160),
        # round 3, batch 4a (loop_depth_unread): one motif PHASE per recurrent step -- at recurrent_loops = num_motifs
        # every layer runs once (was 104 = n + motifs x size x (T - 1), and looped), +motif_size per extra step;
        # Ouro's total_ut_steps; CFRD's cells x recurrences (+ the binding block); HRM's two stacks
        # (hrm_hierarchical_loop); the double loop split (loop_split_depth_miscount: was 42)
        ({"model_type": "lfm3_rd", "num_hidden_layers": 24, "num_prelude_layers": 2, "num_motifs": 5,
          "motif_size": 4, "num_coda_layers": 2, "recurrent_loops": 7}, 24, 32),
        ({"model_type": "ouro", "num_hidden_layers": 24, "total_ut_steps": 4}, 24, 96),
        ({"model_type": "cfrd", "physical_cells": 3, "recurrences": 6, "use_binding_block": True}, 4, 7),
        ({"model_type": "hrm_text", "num_hidden_layers": 16, "H_cycles": 2, "L_cycles": 3}, 32, 128),
        ({"model_type": "qwen3loop", "num_hidden_layers": 28, "loop_middle_layers": 14, "num_loops": 2,
          "enable_double_loop_split": True}, 28, 56),
    ]
    for extra, pl, el in cases:
        cfg = {**base, **extra}
        d = dims_of(cfg)
        assert (d["param_layers"], d["eff_layers"]) == (pl, el), extra["model_type"]
        assert classify(cfg)["loop"] == "looped", extra["model_type"]
    rd = {**base, "model_type": "lfm3_rd", "num_hidden_layers": 24, "num_motifs": 5, "motif_size": 4,
          "recurrent_loops": 5}
    assert (dims_of(rd)["eff_layers"], classify(rd)["loop"]) == (24, "untied")       # every layer once: untied
    from graph import model_graph as mg
    mg.R3_OFF.add("loop_depth_unread")
    try:
        assert (dims_of(rd)["eff_layers"], classify(rd)["loop"]) == (104, "looped")  # the p5 reading, behind its key
    finally:
        mg.R3_OFF.discard("loop_depth_unread")
    # Dory: attention and MLP sub-layers listed separately (36 blocks), 40 entries looped twice
    dory = dims_of({**base, "model_type": "dory", "num_hidden_layers": 72, "n_input_layers": 16,
                    "n_recurrent_layers": 40, "n_recurrent_loops": 2, "n_output_layers": 16,
                    "layer_types": ["sliding_attention", "mlp"] * 36})
    assert (dory["layers"], dory["param_layers"], dory["eff_layers"]) == (36, 36, 56)
    # a loop count of 1 is not a loop
    assert classify({**base, "model_type": "x", "num_hidden_layers": 8, "num_loops": 1})["loop"] == "untied"


def test_lm_config_unwraps_modified_backbones():
    # Looped-Qwen: hidden/depth at the top, attention + GDN shape under qwen_config
    qc = {"model_type": "qwen3_5_text", "hidden_size": 2048, "num_hidden_layers": 24,
          "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256, "rms_norm_eps": 1e-6,
          "intermediate_size": 6144, "linear_num_value_heads": 16, "full_attention_interval": 4}
    lq = {"model_type": "looped_gdn", "hidden_size": 2048, "num_hidden_layers": 24,
          "num_groups": 6, "loops_per_group": 4, "qwen_config": qc}
    c, d = classify(lq), dims_of(lq)
    assert c is not None and c["mixer"] == "hybrid" and c["loop"] == "looped"
    assert (d["param_layers"], d["eff_layers"], d["kv_cap"]) == (24, 96, 2 * 256)
    # MemT: the backbone sits under backbone_configs -> text_config; the wrapper names the model
    memt = {"model_type": "memt_qwen3_5_moe", "memt": {"gdn_version": "stable_gdn2"},
            "backbone_configs": {"model_type": "qwen3_5_moe", "text_config": qc}}
    assert _lm_config(memt)["model_type"] == "memt_qwen3_5_moe" and classify(memt)["mixer"] == "hybrid"


def test_dims_moe_and_kv_key_variants():
    # Step3 (MFA): moe_num_experts / moe_top_k / share_expert_dim, one shared KV head of 256
    step3 = {"model_type": "step3_text", "hidden_size": 7168, "intermediate_size": 18432,
             "num_hidden_layers": 61, "num_attention_heads": 64, "num_attention_groups": 1,
             "head_dim": 256, "moe_num_experts": 48, "moe_top_k": 3, "moe_intermediate_size": 5120,
             "share_expert_dim": 5120, "rope_theta": 5e5}
    c, d = classify(step3), dims_of(step3)
    assert (c["kv"], c["ffn"]) == ("MQA", "moe_shared")
    assert (d["kv_cap"], d["ffn_total"], d["ffn_active"]) == (256, 48 * 5120 + 5120, 3 * 5120 + 5120)
    # Step-3.5: GQA groups, and the sliding layers' wider shape under attention_other_setting
    s35 = dims_of({**step3, "hidden_size": 4096, "num_attention_groups": 8, "head_dim": 128,
                   "attention_other_setting": {"num_attention_heads": 96, "num_attention_groups": 8,
                                               "head_dim": 128}})
    assert (s35["kv_cap"], s35["q_cap"], s35["qheads"]) == (1024, 96 * 128, 96)
    # ERNIE-4.5: moe_k and moe_num_shared_experts (a count of expert-width MLPs)
    ernie = dims_of({"model_type": "ernie4_5_moe", "hidden_size": 2560, "num_hidden_layers": 28,
                     "num_attention_heads": 20, "num_key_value_heads": 4, "intermediate_size": 12288,
                     "moe_num_experts": 64, "moe_k": 6, "moe_intermediate_size": 1536,
                     "moe_num_shared_experts": 2})
    assert (ernie["ffn_total"], ernie["ffn_active"]) == (66 * 1536, 8 * 1536)
    # gpt-oss-Puzzle: expert counts per block, sinks, per-block sliding windows
    gp = {"model_type": "gpt_oss_puzzle", "hidden_size": 2880, "intermediate_size": 2880,
          "num_hidden_layers": 4, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
          "num_experts_per_tok": 4, "rms_norm_eps": 1e-5, "rope_theta": 1.5e5,
          "block_configs": [{"num_local_experts": 128, "sliding_window": 128},
                            {"num_local_experts": 128, "sliding_window": None},
                            {"num_local_experts": 64, "sliding_window": 128},
                            {"num_local_experts": 64, "sliding_window": None}]}
    c, d = classify(gp), dims_of(gp)
    assert (c["ffn"], c["sink"], c["window"]) == ("moe", "yes", "local")
    assert (d["ffn_total"], d["ffn_active"], d["ffn_units"]) == (128 * 2880, 4 * 2880, 384 * 2880)
    # Meditor: num_routed_experts / routed_intermediate_size / shared_intermediate_size
    # (Granite-4.0-H uses the same shared_intermediate_size for its always-on shared MLP)
    med = {"model_type": "meditor", "hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 14,
           "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "num_routed_experts": 22,
           "num_experts_per_tok": 1, "routed_intermediate_size": 1216, "shared_intermediate_size": 4864}
    assert classify(med)["ffn"] == "moe_shared" and dims_of(med)["ffn_total"] == 22 * 1216 + 4864
    # DeciLM: per-layer KV head counts -> GQA at the widest layer
    deci = {"model_type": "deci", "hidden_size": 4096, "num_hidden_layers": 4, "num_attention_heads": 32,
            "num_key_value_heads_per_layer": [4, 2, 1, 4], "rms_norm_eps": 1e-5, "rope_theta": 1e4}
    assert classify(deci)["kv"] == "GQA" and dims_of(deci)["kv_cap"] == 4 * 128


def test_stock_dense_class_ignores_unread_fields():
    # NSFW-flash: a StableLmForCausalLM config carrying num_local_experts / head_dim / rms_norm_eps
    # that StableLM's code never reads -> dense, head_dim = hidden // heads, LayerNorm
    cfg = {"architectures": ["StableLmForCausalLM"], "model_type": "stablelm", "hidden_size": 2560,
           "num_attention_heads": 32, "num_key_value_heads": 32, "head_dim": 256, "intermediate_size": 6912,
           "num_hidden_layers": 25, "num_local_experts": 8, "num_experts_per_tok": 2,
           "layer_norm_eps": 1e-5, "rms_norm_eps": 1e-6, "partial_rotary_factor": 0.25, "rope_theta": 1e4}
    c, d = classify(cfg), dims_of(cfg)
    assert (c["ffn"], c["norm"]) == ("glu", "LayerNorm")
    assert (d["kv_cap"], d["q_cap"], d["ffn_total"], d["experts"]) == (2560, 2560, 6912, 0)
    # the same fields under remote code (auto_map) are the model's own: still read
    assert dims_of({**cfg, "auto_map": {"AutoModelForCausalLM": "m.M"}})["experts"] == 8


def test_classify_cross_layer_kv_source_lists():
    base = {"model_type": "x", "hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8,
            "rms_norm_eps": 1e-5, "rope_theta": 1e4}
    assert classify({**base, "kv_source_layers": [0, 1, 2, 3]})["crosslayer_kv"] == "no"
    assert classify({**base, "kv_source_layers": [0, 1, 1, 3]})["crosslayer_kv"] == "yes"   # Mistral-Large-4
    assert classify({**base, "kv_source_layer_ids": [2]})["crosslayer_kv"] == "yes"        # DeepSeek-V4.1


# --- 2026-11-11 audit fixes (notes/blogpost_revisions.md section 13: A2, A8, C30, D32) ---

def test_classify_gemma_hidden_activation_is_gelu():
    # Gemma-family configs carry the FFN activation as `hidden_activation` (GeGLU, gelu_pytorch_tanh);
    # gemma3 / gemma4 configs have no hidden_act at all (were labelled SiLU)
    base = {"hidden_size": 2560, "num_hidden_layers": 34, "num_attention_heads": 8,
            "num_key_value_heads": 4, "head_dim": 256, "rms_norm_eps": 1e-6, "rope_theta": 1e6}
    assert classify({**base, "model_type": "gemma3_text",
                     "hidden_activation": "gelu_pytorch_tanh"})["act"] == "GeLU_tanh"
    wrapped = {"model_type": "gemma4", "text_config": {**base, "model_type": "gemma4_text",
                                                       "hidden_activation": "gelu_pytorch_tanh"}}
    assert classify(wrapped)["act"] == "GeLU_tanh"                   # nested text_config
    # Gemma 1: GemmaMLP applies ACT2FN[config.hidden_act] -- a "gelu" hidden_act is the exact erf GeLU whatever
    # hidden_activation says (round 2); no key = GemmaConfig's default gelu_pytorch_tanh
    assert classify({**base, "model_type": "gemma", "hidden_act": "gelu",
                     "hidden_activation": "gelu_pytorch_tanh"})["act"] == "GeLU"
    assert classify({**base, "model_type": "gemma"})["act"] == "GeLU_tanh"
    assert classify({**base, "model_type": "gemma3_text"})["act"] == "GeLU_tanh"   # Gemma3TextConfig default
    # erf and tanh GeLU are distinct values (4.7e-4 apart on [-6, 6]); Nemotron-H's mlp_hidden_act still wins
    # over hidden_activation
    assert classify({**base, "model_type": "x", "hidden_act": "gelu"})["act"] == "GeLU"
    for a in ("gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
        assert classify({**base, "model_type": "x", "hidden_act": a})["act"] == "GeLU_tanh", a
    # quick_gelu = x sigmoid(1.702 x) is a SiLU fold, and Phi-3-small's gegelu is quick_gelu(g) * (u + 1)
    # with a clamp, i.e. swigluoai (modeling_phi3_small.py); neither is the tanh / erf GeLU
    assert classify({**base, "model_type": "x", "hidden_act": "quick_gelu"})["act"] == "SiLU"
    assert classify({**base, "model_type": "x", "hidden_act": "gegelu"})["act"] == "swigluoai"
    assert classify({**base, "model_type": "x", "hidden_activation": "silu",
                     "mlp_hidden_act": "relu2"})["act"] == "relu2"


def test_classify_qwen_v1_is_rmsnorm():
    # Qwen v1 (QWenLMHeadModel, remote modeling_qwen.py): ln_1 / ln_2 / ln_f are its RMSNorm class,
    # built with eps=config.layer_norm_epsilon -- the key name is not the norm type
    cfg = {"architectures": ["QWenLMHeadModel"], "auto_map": {"AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel"},
           "model_type": "qwen", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "intermediate_size": 22016, "kv_channels": 128, "layer_norm_epsilon": 1e-6,
           "rotary_emb_base": 10000, "rotary_pct": 1.0, "max_position_embeddings": 32768}
    c = classify(cfg)
    assert (c["norm"], c["pos"], c["act"], c["ffn"]) == ("RMSNorm", "rope", "SiLU", "glu")
    # a genuine layer_norm_epsilon LayerNorm model is untouched
    assert classify({**cfg, "model_type": "gpt_neox", "architectures": ["GPTNeoXForCausalLM"]})["norm"] == "LayerNorm"


def test_dedup_key_follows_context_and_vocab_flags():
    # two models identical except max_context / vocab: one point with the flags off (the headline
    # key is unchanged), two points when the flag that compares that axis is on
    import graph.model_graph as mg
    cfg = {"model_type": "llama", "hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8,
           "intermediate_size": 2048, "rms_norm_eps": 1e-5, "rope_theta": 1e4, "vocab_size": 32000,
           "max_position_embeddings": 4096}
    mk = lambda n, **kw: {"name": n, "slots": classify({**cfg, **kw}), "dims": dims_of({**cfg, **kw})}
    ms = [mk("a"), mk("b", max_position_embeddings=131072), mk("c", vocab_size=128256)]
    try:
        assert len(mg._dedup_points(ms)) == 1
        mg.USE_CONTEXT = True
        P = mg._dedup_points(ms)
        assert len(P) == 2 and sorted(p["dims"]["max_context"] for p in P) == [4096, 131072]
        mg.USE_CONTEXT, mg.USE_VOCAB = False, True
        assert len(mg._dedup_points(ms)) == 2
    finally:
        mg.USE_CONTEXT = mg.USE_VOCAB = False


def test_ffn_units_count_only_ffn_sublayers():
    # Nemotron-H lists mamba / attention / moe as SEPARATE entries: only the moe ('E') / mlp ('-')
    # entries carry an FFN (was len(list) * ffn_total, pricing every Mamba entry as a MoE layer)
    nh = {"model_type": "nemotron_h", "hidden_size": 1024, "num_attention_heads": 8, "num_key_value_heads": 2,
          "head_dim": 128, "intermediate_size": 3072, "n_routed_experts": 8, "num_experts_per_tok": 2,
          "moe_intermediate_size": 512, "moe_shared_expert_intermediate_size": 1024, "n_shared_experts": 1,
          "mamba_num_heads": 32, "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe", "mamba"]}
    d = dims_of(nh)
    assert d["ffn_total"] == 8 * 512 + 1024 and d["eff_layers"] == 6
    assert d["ffn_units"] == 2 * (8 * 512 + 1024)
    pat = {k: v for k, v in nh.items() if k != "layers_block_type"}
    pat.update(hybrid_override_pattern="M-M*EM", num_hidden_layers=6)
    assert dims_of(pat)["ffn_units"] == 1 * (8 * 512 + 1024) + 1 * 3072        # one MoE, one dense MLP
    # DeepSeek-style first_k_dense_replace: the first k layers are a dense MLP, not the expert bank
    ds = {"model_type": "deepseek_v3", "hidden_size": 1024, "num_hidden_layers": 6, "num_attention_heads": 8,
          "intermediate_size": 4096, "n_routed_experts": 16, "num_experts_per_tok": 2,
          "moe_intermediate_size": 256, "n_shared_experts": 1, "first_k_dense_replace": 2,
          "rms_norm_eps": 1e-6, "rope_theta": 1e4}
    assert dims_of(ds)["ffn_units"] == 4 * (16 * 256 + 256) + 2 * 4096


def test_act_status_covers_every_activation_and_flags_open_ones():
    # every activation classify can emit has an ACT_STATUS row (xIELU / poly_norm: host branches since 2026-11-11)
    from graph.minimal_emulator import ACT_STATUS, NORM_STATUS, act_name, coverage
    for a in ("silu", "gelu", "relu", "relu2", "situ", "xielu", "poly_norm", "mish", "swigluoai"):
        assert a in ACT_STATUS
    assert {k for k, v in ACT_STATUS.items() if v[0] == "tested"} >= {"silu", "situ", "relu2"}
    assert set(NORM_STATUS) >= {"RMSNorm", "LayerNorm"}
    base = {"model_type": "llama", "hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8,
            "intermediate_size": 2048, "rms_norm_eps": 1e-5, "rope_theta": 1e4}

    def mk(**kw):
        c = {**base, **kw}
        return {"name": "m", "cfg": c, "slots": classify(c), "dims": dims_of(c)}

    for raw, want in (("relu2", "relu2"), ("relu", "relu"), ("silu", "silu"), ("xielu", "xielu")):
        m = mk(hidden_act=raw)
        assert act_name(m["cfg"], m["slots"]) == want
    # (carried, native): SiLU tested; ReLU tested -- the ungated UFFN c_relu branch (OPT; before it landed,
    # priced by the beta -> inf limit to SiLU: carried, not native); GeLU / mish tested -- exact UFFN branches
    # (c_gelu / c_geglu / c_mish; the GeLU <- SiLU floor 0.012 is above the calibrated ACT_FLOOR_MAX, so
    # without them they would be open); xIELU tested
    from graph.minimal_emulator import act_status
    assert coverage([mk()]) == (1, 1)
    assert coverage([mk(hidden_act="relu")]) == ((1, 1) if ACT_STATUS["relu"][0] == "tested" else (1, 0))
    for a in ("gelu", "gelu_new", "gelu_pytorch_tanh", "mish"):
        assert coverage([mk(hidden_act=a)]) == (1, 1), a
    assert act_status("gelu")[0] == act_status("mish")[0] == "tested"
    assert coverage([mk(hidden_act="xielu")]) == ((0, 0) if ACT_STATUS["xielu"][0] == "open" else (1, 1))


def test_join_is_per_axis_max_and_dominates_every_model():
    # the elephantine join over EVERY model (all mixers, AR + diffusion): each axis is a real model's
    # value (x the join's MLA / parallel penalty), and the host emulates every point under
    # emul_matrix(mode="capacity") and (minus open activations) the headline mode="payable", also with the opt-in context / vocab axes on
    import numpy as np
    import graph.model_graph as mg
    from graph.minimal_emulator import JOIN_AXES, host_point, join, load
    models = load()
    j = join(models)
    for k in JOIN_AXES:
        holder = next(m for m in models if m["name"] == j["who"][k])
        v = holder["dims"][k]
        if v is None:                                     # (round 3: an unknown context -> the join is unbounded there)
            assert k in ("max_context", "eff_window") and j[k] == mg.CTX_UNBOUNDED, k
            continue
        if k in ("eff_layers", "param_layers"):           # one untied depth >= both depth axes
            assert j["layers"] == max(m["dims"][a] for m in models for a in ("eff_layers", "param_layers")) >= v
        else:
            assert j[k] in (v, 2 * v), k
    assert j["ffn_units"] == j["layers"] * j["ffn_total"]
    # the host has a parallel-block branch: no 2x width for parallel targets (Command-R+ h 12288)
    assert j["hidden"] == max(m["dims"]["hidden"] for m in models) == 16384
    try:
        for flags in ((False, False), (True, False), (False, True)):
            mg.USE_CONTEXT, mg.USE_VOCAB = flags
            P = mg._dedup_points(models)
            ok = mg.emul_matrix(P + [host_point(j, "ar"), host_point(j, "diffusion")], mode="capacity")
            n = len(P)
            col = np.array([n + (p["slots"]["gen"] != "ar") for p in P])
            # block-causal targets (SDAR: gen "block"). Round 2 pass 1 walled them both ways; pass 4 (2026-11-13):
            # SDAR's code runs non-causal attention under a caller-supplied mask, so block and diffusion are one class
            # under the mask currency (GEN_ORDER) and the diffusion host reaches them in the relation. The universal
            # host runs them through its block_size / caller-supplied mask (minimal_emulator FAMILY_STATUS
            # 'block-causal', tested: tests/test_host_gaps.py) -- the AR host still plays none of them
            blk = np.array([p["slots"]["gen"] == "block" for p in P])
            from graph.minimal_emulator import FAMILY_STATUS
            assert FAMILY_STATUS["block-causal"][0] == "tested"
            assert blk.any() and not ok[np.where(blk)[0], n].any() and ok[np.where(blk)[0], n + 1].all()
            assert ok[np.arange(n), col][~blk].all(), (flags, [P[i]["members"][0] for i in np.where(~ok[np.arange(n), col] & ~blk)[0]][:5])
            # (round 3, nonlm_kept / nonlm_heads: the non-next-token models -- score-head rerankers, sentence embeddings,
            # GRM's value head, sundial's patch input / flow head -- ARE carried above by the relation's head / input
            # branch sets (host_point HOST_HEAD / HOST_INPUT), and the universal host's readout / patch / flow branches
            # carry them (minimal_emulator HEAD_STATUS / INPUT_STATUS, tested: tests/test_nonlm_host.py))
            nonlm = np.array([p["slots"].get("head", "lm") != "lm" or p["slots"].get("input", "tokens") != "tokens"
                              for p in P])
            from graph.minimal_emulator import HEAD_STATUS, INPUT_STATUS
            assert nonlm.any() and {P[i]["slots"].get("head", "lm") for i in np.where(nonlm)[0]} <= {
                k for k, v in HEAD_STATUS.items() if v[0] == "tested"}
            assert {P[i]["slots"].get("input", "tokens") for i in np.where(nonlm)[0]} <= {
                k for k, v in INPUT_STATUS.items() if v[0] == "tested"}
            # the HEADLINE relation: the host's branch sets carry every point except an activation
            # with no host branch and no floor edge to one (poly_norm, Motif-3: ACT_STATUS 'open');
            # one AR host carries the AR points too (AR <= diffusion: the diffusion host also would)
            okp = mg.emul_matrix(P + [host_point(j, "ar"), host_point(j, "diffusion")], mode="payable")
            # (GeLU / mish are host branches -- c_gelu / c_geglu / c_mish -- so not open although no
            # SiLU floor edge reaches them within the calibrated ACT_FLOOR_MAX)
            from graph.minimal_emulator import ACT_SLOT, ACT_STATUS, act_status
            open_acts = {"poly_norm"} | {ACT_SLOT[a] for a in ACT_STATUS if act_status(a)[0] == "open"}
            assert not open_acts & {"GeLU", "mish"}
            # ... and a forward-pass feature the host has no branch for (model_graph.FEATURES not in HOST_FEATS:
            # nanochat's value embeddings)
            from graph.minimal_emulator import HOST_FEATS
            # (round 3, batch 1: also a grouped RMSNorm, dims norm_groups > 1 -- UNorm has no per-group statistic; the
            # embedding norm and OPT's factorized embedding ARE host branches: host_point emb_norm / emb_fold)
            # (round 3, batch 2a: also a qk-norm variant the host's HOST_QKNORM branches do not reach -- the config gain of
            # head_free_gain needs theorem qkn_const_gain, Dory's ngpt and dots3-note's mla_krope have no branch -- and
            # IQuest-Q1's key sink; these were 5 points carried before the batch read their norms)
            # (round 3, batch 2d: also Step-3's normed query latent, dims q_lat_norm > 0 -- hosted only by an equal latent;
            # the host has none: 1 point carried before the batch read share_q_dim)
            from graph.minimal_emulator import HOST_QKNORM
            uncarried = lambda p: (bool(set(filter(None, p["slots"].get("feats", "").split("+"))) - HOST_FEATS
                                        - ({f for f, r in mg.FEATURES.items() if r[0] == "free"} - mg.MOE_PRICED_FEATS))
                                   or p["dims"].get("norm_groups", 1) > 1
                                   or not mg.payable_reaches("qknorm", p["slots"]["qknorm"], HOST_QKNORM)
                                   or p["slots"]["sink"] == "key"
                                   or p["dims"].get("q_lat_norm", 0) > 0)
            miss = {P[i]["slots"]["act"] for i in np.where(~okp[np.arange(n), col] & ~blk)[0] if not uncarried(P[i])}
            assert miss <= open_acts, (flags, miss)
            assert okp[[i for i in range(n) if P[i]["slots"]["gen"] == "ar" and P[i]["slots"]["act"] not in open_acts
                        and not uncarried(P[i])], n + 1].all()
    finally:
        mg.USE_CONTEXT = mg.USE_VOCAB = False


# --- round 2 (2026-11-11): labels and FFN pricing checked against each model's modeling code ---

_BASE = {"hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8, "intermediate_size": 2048}


def test_fused_glu_intermediate_is_halved_qwen_v1_and_molmo():
    # Qwen v1: w1, w2 = Linear(h, intermediate_size // 2) (modeling_qwen.py) -> Qwen-72B GLU 24576, not 49152
    q72 = {"architectures": ["QWenLMHeadModel"], "model_type": "qwen", "hidden_size": 8192,
           "num_hidden_layers": 80, "num_attention_heads": 64, "intermediate_size": 49152,
           "layer_norm_epsilon": 1e-6, "rotary_pct": 1.0, "kv_channels": 128}
    d = dims_of(q72)
    assert (d["ffn_total"], d["ffn_active"], d["ffn_units"]) == (24576, 24576, 80 * 24576)
    # Molmo: ff_proj (intermediate_size wide) -> SwiGLU chunk(2): the GLU is half (output_multiplier 0.5)
    mo = {"model_type": "molmo", "hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64,
          "num_key_value_heads": 8, "intermediate_size": 59136, "layer_norm_type": "rms", "rope_theta": 1e6}
    assert dims_of(mo)["ffn_total"] == 29568 and classify(mo)["norm"] == "RMSNorm"
    # Phi-3-small: up_proj = Linear(h, 2 * ff_intermediate_size), gegelu splits it -> width 14336
    p3s = {"model_type": "phi3small", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "num_key_value_heads": 8, "ff_intermediate_size": 14336, "hidden_act": "gegelu",
           "layer_norm_epsilon": 1e-5, "rope_embedding_base": 1e6}
    assert dims_of(p3s)["ffn_total"] == 14336


@pytest.mark.parametrize("mt,extra,norm", [
    ("mamba", {"layer_norm_epsilon": 1e-5}, "RMSNorm"),          # MambaRMSNorm
    ("falcon_mamba", {"layer_norm_epsilon": 1e-5}, "RMSNorm"),   # FalconMambaRMSNorm
    ("exaone", {"layer_norm_epsilon": 1e-5}, "RMSNorm"),         # ExaoneRMSNorm
    ("olmo", {"rms_norm_eps": 1e-5}, "LayerNorm"),               # OlmoLayerNorm (non-affine F.layer_norm)
    ("orion", {"rms_norm_eps": 1e-5}, "LayerNorm"),              # nn.LayerNorm(eps=rms_norm_eps)
    ("cohere", {"rms_norm_eps": 1e-5}, "LayerNorm"),             # CohereLayerNorm (mean-centred)
    ("persimmon", {"rms_norm_eps": 1e-6}, "LayerNorm"),          # nn.LayerNorm
    ("starcoder2", {"norm_epsilon": 1e-5}, "LayerNorm"),         # nn.LayerNorm
    ("stablelm_epoch", {"norm_eps": 1e-5}, "LayerNorm"),
    ("pharia-v1", {}, "LayerNorm"),
    ("rwkv7", {"norm_eps": 1e-5}, "LayerNorm"),                  # fla RWKV7 nn.LayerNorm
    ("nanochat_gpt", {}, "RMSNorm"),                             # F.rms_norm
])
def test_norm_label_follows_the_modeling_code(mt, extra, norm):
    cfg = {**_BASE, "model_type": mt, "rope_theta": 1e4, **extra}
    if mt in ("mamba", "falcon_mamba", "rwkv7"):
        cfg["ssm_state_size" if "mamba" in mt else "head_dim"] = 16
    assert classify(cfg)["norm"] == norm


def test_ffn_gating_block_and_activation_follow_the_code():
    c = lambda mt, **kw: classify({**_BASE, "model_type": mt, "rope_theta": 1e4, **kw})
    # ungated up -> act -> down (fc1/fc2, c_fc/c_proj, fc_in/fc_out, key/value)
    for mt in ("phi", "progen", "starcoder2", "gpt_neox_japanese", "pharia-v1", "nanochat_gpt"):
        assert c(mt, hidden_act="gelu")["ffn"] == "vanilla", mt
    # attention and MLP read the same normed input and are summed (parallel); Persimmon is sequential
    for mt in ("phi", "codegen", "progen", "cohere", "cohere2"):
        assert c(mt)["block"] == "parallel", mt
    assert c("persimmon", hidden_act="relu2")["block"] == "sequential"
    # squared-ReLU FFNs whose config carries no activation key
    from graph.minimal_emulator import act_name
    for mt in ("nanochat_gpt", "rwkv", "rwkv5"):
        cfg = {**_BASE, "model_type": mt, "rope_theta": 1e4}
        s = classify(cfg)
        assert s["act"] == "relu2" and act_name(cfg, s) == "relu2", mt
    # rotary_pct = partial_rotary_factor = 0 (NoPE GPT-NeoX): no RoPE at all
    assert c("gpt_neox_nope", rotary_pct=0.0, partial_rotary_factor=0.0)["pos"] == "nope"
    assert c("gpt_neox", rotary_pct=0.25)["pos"] == "partial_rope"


def test_gemma4_moe_block_beside_dense_mlp_and_top_k_experts():
    # Gemma-4 26B-A4B (modeling_gemma4.py Gemma4TextDecoderLayer): every layer runs its dense MLP
    # (intermediate_size 2112) AND 128 experts of 704 routed top_k_experts = 8, summed
    g4 = {"model_type": "gemma4_text", "hidden_size": 2816, "num_hidden_layers": 30, "num_attention_heads": 16,
          "num_key_value_heads": 8, "head_dim": 256, "intermediate_size": 2112, "moe_intermediate_size": 704,
          "num_experts": 128, "top_k_experts": 8, "enable_moe_block": True,
          "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6, "rope_theta": 1e6}
    d = dims_of(g4)
    assert (d["ffn_active"], d["ffn_total"]) == (8 * 704 + 2112, 128 * 704 + 2112) == (7744, 92224)
    assert d["ffn_units"] == 30 * 92224 and classify(g4)["ffn"] == "moe_shared"
    # Gemma-4 E-models: the num_kv_shared_layers KV-shared layers get a 2x-wide MLP (use_double_wide_mlp)
    e2b = {**{k: v for k, v in g4.items() if k not in ("num_experts", "top_k_experts", "enable_moe_block",
                                                       "moe_intermediate_size")},
           "num_hidden_layers": 35, "intermediate_size": 6144, "use_double_wide_mlp": True, "num_kv_shared_layers": 20}
    d = dims_of(e2b)
    assert (d["ffn_total"], d["ffn_units"]) == (12288, 15 * 6144 + 20 * 12288)


def test_alternate_top_k_and_shared_count_keys():
    moe = {**_BASE, "num_hidden_layers": 6, "rms_norm_eps": 1e-6, "rope_theta": 1e4,
           "num_experts": 64, "moe_intermediate_size": 256}
    # Kimi-Linear / K3: num_experts_per_token; num_shared_experts is a COUNT of moe_intermediate_size MLPs
    d = dims_of({**moe, "model_type": "kimi_linear", "num_experts_per_token": 16, "num_shared_experts": 2})
    assert (d["ffn_active"], d["ffn_total"]) == (16 * 256 + 2 * 256, 64 * 256 + 2 * 256)
    assert classify({**moe, "model_type": "kimi_linear", "num_shared_experts": 2})["ffn"] == "moe_shared"
    assert classify({**moe, "model_type": "x", "num_shared_experts": 0})["ffn"] == "moe"
    # Motif-3: experts_top_k; n_dense_first_layers leading dense MLPs
    d = dims_of({**moe, "model_type": "Motif", "experts_top_k": 8, "n_dense_first_layers": 2})
    assert d["ffn_active"] == 8 * 256 and d["ffn_units"] == 4 * (64 * 256) + 2 * 2048
    # AFMoE / LFM2-MoE: num_dense_layers
    assert dims_of({**moe, "model_type": "afmoe", "num_dense_layers": 1})["ffn_units"] == 5 * 64 * 256 + 2048
    # LLaMA-MoE: the Llama FFN split into experts (size_experts per layer), num_selects routed
    lm = {**_BASE, "model_type": "llama_moe", "rms_norm_eps": 1e-6, "rope_theta": 1e4, "intermediate_size": 11008,
          "num_experts": 16, "num_selects": 2, "size_experts": [[688] * 16] * 4}
    d = dims_of(lm)
    assert (d["ffn_total"], d["ffn_active"], d["ffn_units"]) == (11008, 2 * 688, 4 * 11008)
