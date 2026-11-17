"""The currency edges the HEADLINE relation (emul_matrix(mode="payable")) adds, each backed here by
a construction on stock modules or a measured floor, and the walls it keeps.

  ZERO_EDGES  a hybrid host (stock Jamba) zeroes its Mamba layers and runs a plain softmax target
              on its attention layers alone; a sliding-window host (stock Qwen2 layer_types) zeroes
              its sliding layers and runs a full-attention target on its full layers. Exact; the
              currency is depth (only the host's layers of the target's kind count).
  regime      qk-norm and softcap: the residual is the same expression in both directions.
  act         the thresholded single-op minimax floors (ACT_FLOORS / ACT_FLOOR_MAX), re-measured on
              the cells the threshold decides; the threshold checked against the recorded real-model
              swaps (act/data/act_swap_agreement.jsonl) and one live swap on gpt2 / OPT-125m.
  pos         ALiBi <- learned-abs exact on the field level (attn/positional_cross.py).
  gen         AR <= diffusion one-way (attn/ar_diffusion.py).

Run:  pytest tests/test_payable_edges.py -q
"""
import math

import pytest

torch = pytest.importorskip("torch")
F64 = torch.float64


def _rand_(model, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def _ids(vocab, T=12, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (2, T))


def _zero_layer(layer, mixer_out):
    """An identity layer: zero the mixer's output projection and the FFN's down projection (the
    residual passes through, exact)."""
    with torch.no_grad():
        mixer_out.weight.zero_()
        if mixer_out.bias is not None:
            mixer_out.bias.zero_()
        layer_ffn = getattr(layer, "feed_forward", None) or layer.mlp
        layer_ffn.down_proj.weight.zero_()


def _copy_llama_layer(dst, src):
    with torch.no_grad():
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(dst.self_attn, n).weight.copy_(getattr(src.self_attn, n).weight)
        ffn = getattr(dst, "feed_forward", None) or dst.mlp
        for n in ("gate_proj", "up_proj", "down_proj"):
            getattr(ffn, n).weight.copy_(getattr(src.mlp, n).weight)
        dst.input_layernorm.weight.copy_(src.input_layernorm.weight)
        post = getattr(dst, "pre_ff_layernorm", None) or dst.post_attention_layernorm
        post.weight.copy_(src.post_attention_layernorm.weight)


@torch.no_grad()
def test_hybrid_host_zeroes_its_ssm_layers_for_a_softmax_target():
    # mixer softmax <- hybrid: a stock JambaForCausalLM (Mamba layers 0, 2; attention layers 1, 3)
    # reproduces a 2-layer softmax target on its 2 attention layers with the Mamba layers zeroed.
    # Jamba attention carries no positional encoding, so the target is a NoPE Llama (the NoPE edge:
    # inv_freq = 0). The depth currency is real: the 4-layer host hosts only a 2-layer target.
    from transformers import JambaConfig, JambaForCausalLM, LlamaConfig, LlamaForCausalLM
    V, d = 97, 32
    tc = LlamaConfig(vocab_size=V, hidden_size=d, intermediate_size=48, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6,
                     attn_implementation="eager", tie_word_embeddings=False)
    t = _rand_(LlamaForCausalLM(tc).to(F64).eval())
    t.model.rotary_emb.inv_freq.zero_()                       # NoPE target
    hc = JambaConfig(vocab_size=V, hidden_size=d, intermediate_size=48, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, num_experts=1,
                     attn_layer_period=2, attn_layer_offset=1, use_mamba_kernels=False,
                     mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, attn_implementation="eager",
                     tie_word_embeddings=False)
    h = _rand_(JambaForCausalLM(hc).to(F64).eval(), seed=1)
    with torch.no_grad():
        h.model.embed_tokens.weight.copy_(t.model.embed_tokens.weight)
        h.model.final_layernorm.weight.copy_(t.model.norm.weight)
        h.lm_head.weight.copy_(t.lm_head.weight)
    for i, layer in enumerate(h.model.layers):
        if i % 2 == 0:
            _zero_layer(layer, layer.mamba.out_proj)          # the SSM layers become identities
        else:
            _copy_llama_layer(layer, t.model.layers[i // 2])
    ids = _ids(V)
    err = (t(ids).logits - h(ids).logits).abs().max().item()
    assert err < 1e-6, err                                    # fp32 RMSNorm inside both stock modules
    # the SSM layers really are SSM: un-zeroed, they change the output
    h2 = _rand_(JambaForCausalLM(hc).to(F64).eval(), seed=1)
    h2.load_state_dict(h.state_dict())
    with torch.no_grad():
        h2.model.layers[0].mamba.out_proj.weight.normal_(0, 0.5)
    assert (t(ids).logits - h2(ids).logits).abs().max().item() > 1e-3


@torch.no_grad()
def test_sliding_host_zeroes_its_sliding_layers_for_a_full_target():
    # window full <- local: a stock Qwen2 host with layer_types [sliding, full, sliding, full] runs a
    # 2-layer full-attention Qwen2 target on its two full layers, the sliding ones zeroed.
    from transformers import Qwen2Config, Qwen2ForCausalLM
    V, d, T = 97, 32, 16
    kw = dict(vocab_size=V, hidden_size=d, intermediate_size=48, num_attention_heads=4,
              num_key_value_heads=2, rms_norm_eps=1e-6, attn_implementation="eager",
              tie_word_embeddings=False)
    t = _rand_(Qwen2ForCausalLM(Qwen2Config(num_hidden_layers=2, **kw)).to(F64).eval())
    hc = Qwen2Config(num_hidden_layers=4, use_sliding_window=True, sliding_window=4, max_window_layers=0,
                     layer_types=["sliding_attention", "full_attention"] * 2, **kw)
    h = _rand_(Qwen2ForCausalLM(hc).to(F64).eval(), seed=1)
    with torch.no_grad():
        h.model.embed_tokens.weight.copy_(t.model.embed_tokens.weight)
        h.model.norm.weight.copy_(t.model.norm.weight)
        h.lm_head.weight.copy_(t.lm_head.weight)
        for i, layer in enumerate(h.model.layers):
            if i % 2 == 0:
                _zero_layer(layer, layer.self_attn.o_proj)
            else:
                src = t.model.layers[i // 2]
                for n in ("q_proj", "k_proj", "v_proj"):
                    getattr(layer.self_attn, n).weight.copy_(getattr(src.self_attn, n).weight)
                    getattr(layer.self_attn, n).bias.copy_(getattr(src.self_attn, n).bias)
                _copy_llama_layer(layer, src)
    ids = _ids(V, T)
    assert (t(ids).logits - h(ids).logits).abs().max().item() < 1e-9
    # the window is real: the same target layers placed on the SLIDING layers do not reproduce it
    h.model.layers[0].load_state_dict(h.model.layers[1].state_dict())
    h.model.layers[2].load_state_dict(h.model.layers[3].state_dict())
    _zero_layer(h.model.layers[1], h.model.layers[1].self_attn.o_proj)
    _zero_layer(h.model.layers[3], h.model.layers[3].self_attn.o_proj)
    assert (t(ids).logits - h(ids).logits).abs().max().item() > 1e-4


def test_qknorm_regime_is_the_same_residual_both_ways():
    # plain host for a qk-norm target folds 1/rms(q) into 1/c_q; a qk-norm host for a plain target
    # sets its gain to c_q. Both residuals vanish when ||q|| is constant across tokens and grow with
    # its variation (CV) -- the same regime currency in both directions.
    from attn.qk_norm import attend, make_qk, rmsnorm
    fwd, rev = [], []
    for cv in (0.0, 0.1, 0.4):
        q, k = make_qk(48, 64, cv, seed=0)
        cq, ck = q.pow(2).mean(-1).sqrt().mean(), k.pow(2).mean(-1).sqrt().mean()
        fwd.append((attend(rmsnorm(q, 0) @ rmsnorm(k, 0).T) - attend((q / cq) @ (k / ck).T)).abs().max().item())
        rev.append((attend(q @ k.T) - attend((cq * rmsnorm(q, 0)) @ (ck * rmsnorm(k, 0)).T)).abs().max().item())
    assert fwd[0] < 1e-12 and rev[0] < 1e-12                  # constant norm: both exact
    assert fwd[0] < fwd[1] < fwd[2] and rev[0] < rev[1] < rev[2]


def test_softcap_regime_is_the_same_residual_both_ways():
    from attn.qk_norm import attend
    from attn.softcap import softcap
    g = torch.Generator().manual_seed(0)
    s = torch.randn(32, 32, dtype=F64, generator=g)
    for ratio, small in ((0.05, True), (4.0, False)):
        x = s / s.abs().max() * ratio                          # max|logit| / cap, cap = 1
        plain_host = (attend(softcap(x, 1.0)) - attend(x)).abs().max().item()     # host plain, target capped
        cap_host = (attend(x) - attend(softcap(x, 1.0))).abs().max().item()       # host capped, target plain
        assert plain_host == cap_host
        assert (plain_host < 1e-3) == small


# cells the calibrated ACT_FLOOR_MAX threshold decides (host, target, admitted?): the limits and folds
# (0.000) are in, every floor -- the swish cluster at 0.009 - 0.014 included -- is out
_ACT_CELLS = [("SiLU", "xielu", False), ("relu2", "xielu", False), ("ReLU", "SiLU", False),
              ("situ", "SiLU", False), ("GeLU", "SiLU", False), ("SiLU", "GeLU", False),
              ("SiLU", "mish", False), ("mish", "xielu", False), ("situ", "ReLU", False),
              ("GeLU", "situ", False), ("SiLU", "relu2", False), ("xielu", "relu2", True),
              ("SiLU", "ReLU", True), ("swigluoai", "SiLU", True), ("SiLU", "swigluoai", True)]


@pytest.mark.parametrize("host,target,admitted", _ACT_CELLS)
def test_activation_floor_cells_match_the_recorded_table(host, target, admitted):
    from act.activation_cross import ACTS, floor_minimax
    from graph.model_graph import ACT_FLOOR_MAX, ACT_FLOORS, REACH
    fl = floor_minimax(ACTS[host], ACTS[target])
    assert abs(fl - ACT_FLOORS[target][host]) < 5e-3, (fl, ACT_FLOORS[target][host])
    assert (fl <= ACT_FLOOR_MAX) == admitted
    assert (host in REACH["act"].get(target, {target})) == admitted


def test_activation_relation_is_the_one_hop_table():
    # REACH["act"] is exactly the thresholded table, not its closure; at the calibrated threshold
    # only exact folds and beta -> inf limits remain, every floor cell is a wall
    from graph.model_graph import ACT_EXACT, ACT_FLOOR_MAX, ACT_FLOORS, ACT_LIMIT, REACH
    for tgt, row in ACT_FLOORS.items():
        assert REACH["act"].get(tgt, {tgt}) - {tgt, "act_top"} == {h for h, fl in row.items() if fl <= ACT_FLOOR_MAX}
        for h in REACH["act"].get(tgt, {tgt}) - {tgt, "act_top"}:
            # (round 2, pass 5: the GeLU erf <-> tanh floors, 0.0003, measured payable both ways)
            assert (tgt, h) in ACT_EXACT | ACT_LIMIT | {("GeLU", "GeLU_tanh"), ("GeLU_tanh", "GeLU")}
    assert "ReLU" not in REACH["act"]["SiLU"] and "SiLU" not in REACH["act"]["GeLU"]
    assert REACH["act"].get("situ", {"situ"}) == {"situ"}                     # nothing within the threshold
    assert REACH["act"].get("poly_norm", {"poly_norm"}) == {"poly_norm"}   # no measured cell at all


_SWAP = "act/data/act_swap_agreement.jsonl"


def test_threshold_is_calibrated_on_the_recorded_swaps():
    # the recorded real-model swaps (act/act_swap_agreement.py): every swap at floor <= ACT_FLOOR_MAX
    # is high-agreement (top-1 >= 99% and >= 90% of 32-token greedy continuations identical) and the
    # smallest failing floor is above it -- on every model, real hosts and the scaled-fit dose points
    import os
    from act.act_swap_report import DEFS, load, swaps, threshold
    from graph.model_graph import ACT_FLOOR_MAX, ACT_FLOORS
    if not os.path.exists(_SWAP):
        pytest.skip("no recorded swaps")
    sw = swaps(load(_SWAP))
    assert len({r["model"] for r, _ in sw}) >= 10
    passing, first_fail = threshold(sw, DEFS["strict"])
    assert passing <= ACT_FLOOR_MAX < first_fail, (passing, ACT_FLOOR_MAX, first_fail)
    # no real cell of the floor table lies between the threshold and the first failure, and the
    # first real (non-dose) floor cell fails on every model it was measured on
    # (round 2, pass 5: except the GeLU erf <-> tanh cells, 0.0003, measured payable on six models: gelu_swap)
    cells = sorted({v for t, row in ACT_FLOORS.items() for h, v in row.items() if v > 0 and (t, h) not in {("GeLU", "GeLU_tanh"), ("GeLU_tanh", "GeLU")}})
    assert cells[0] >= first_fail
    real = [(r, c) for r, c in sw if "*" not in r["host"] and r["floor"] > ACT_FLOOR_MAX]
    assert real and not any(DEFS["strict"](r, c) for r, c in real)


def test_live_swap_gpt2_and_opt():
    # live (CPU, 4 prompts): GeLU <- SiLU (floor 0.012) on real gpt2 disagrees; ReLU <- SiLU (the
    # beta -> inf limit) on real OPT-125m is token-identical
    from act.act_swap_agreement import run
    try:
        g = run("gpt2", "cpu", hosts=("SiLU",), n_prompts=4, only=["SiLU"])
        o = run("facebook/opt-125m", "cpu", hosts=("SiLU",), n_prompts=4, only=["SiLU"])
    except OSError:
        pytest.skip("gpt2 / opt-125m not cached")
    assert g[0]["top1"] < 0.99 and g[0]["floor"] > 0.01
    assert o[0]["top1"] == 1.0 and o[0]["ex64"] == 1.0


def test_new_edges_through_the_headline_relation():
    from graph.model_graph import emulates
    from tests.test_graph_rules import mk
    # hybrid host for a softmax target: only its attention layers count
    hyb = mk(mixer="hybrid", layers=48, attn_eff=12, attn_param=12, full_eff=12, full_param=12)
    assert emulates(hyb, mk(layers=12)) and not emulates(hyb, mk(layers=13))
    assert emulates(hyb, mk(layers=48), mode="capacity")       # the secondary ignores the wall
    # sliding host for a full target: only its full layers; for a local target: every layer
    loc = mk(window="local", layers=30, full_eff=5, full_param=5)
    assert emulates(loc, mk(layers=5)) and not emulates(loc, mk(layers=6))
    assert emulates(loc, mk(window="local", layers=30, full_eff=5, full_param=5))
    # ... but a local target's FULL layers still need host full layers (per-layer kinds, model_graph.KIND_REACH)
    assert not emulates(loc, mk(window="local", layers=30, full_eff=6, full_param=6))
    # AR <= diffusion, one-way
    assert emulates(mk(gen="diffusion"), mk(gen="ar")) and not emulates(mk(gen="ar"), mk(gen="diffusion"))
    # RMSNorm target on a LayerNorm host: the [h; -h] mirror, 2x hidden
    assert not emulates(mk(norm="LayerNorm"), mk(norm="RMSNorm"))
    assert emulates(mk(norm="LayerNorm", hidden=2048), mk(norm="RMSNorm"))
    # ALiBi <- learned-abs (bounded table); learned-abs <- RoPE stays a wall
    assert not emulates(mk(pos="learned_abs"), mk(pos="alibi"))   # round 2, pass 5: a wall (the pre-norm reads j / R_j)
    assert not emulates(mk(pos="rope"), mk(pos="learned_abs"))
    # softcap regime (REGIME_PAYABLE, attn/regime_agreement.py): a capped host for a plain target pays, a plain
    # host for an attention-capped target does not (Gemma-2-9B); final <-> none both ways. (Before round 2 pass 2:
    # yes <-> no both ways.) The qk-norm regime is measured NOT payable (a wall), every variant
    assert emulates(mk(softcap="attn"), mk(softcap="no")) and not emulates(mk(softcap="no"), mk(softcap="attn"))
    assert emulates(mk(softcap="final"), mk(softcap="no")) and emulates(mk(softcap="no"), mk(softcap="final"))
    assert not emulates(mk(qknorm="head"), mk(qknorm="no")) and not emulates(mk(qknorm="no"), mk(qknorm="head"))
    # activations: relu2 <- xIELU and ReLU <- SiLU (limits) pay; GeLU <- SiLU (floor 0.012, not
    # high-agreement) and xIELU <- SiLU (0.135) are walls
    assert emulates(mk(act="xielu"), mk(act="relu2")) and emulates(mk(act="SiLU"), mk(act="ReLU"))
    assert not emulates(mk(act="SiLU"), mk(act="GeLU")) and not emulates(mk(act="SiLU"), mk(act="xielu"))
    assert emulates(mk(act="SiLU"), mk(act="GeLU"), mode="capacity")     # the secondary ignores the wall
    # exact slice: limits / regimes refused, zeroing / mask / mirror kept
    assert not emulates(mk(act="SiLU"), mk(act="ReLU"), mode="exact")
    assert emulates(hyb, mk(layers=12), mode="exact")
    assert emulates(mk(gen="diffusion"), mk(gen="ar"), mode="exact")


def test_ar_is_a_masked_diffusion_forward():
    from attn.ar_diffusion import demo
    assert demo() == 0.0
