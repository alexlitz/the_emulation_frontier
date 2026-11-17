"""The start-marker (BOS) constructions of hf/bos_constructions.py, on stock modules.

  learned-abs <- NoPE / RoPE(inv_freq = 0) host:  two extra blocks and five extra channels reproduce
      e(x_t) + P_t exactly when the input starts with the marker (Lean `learned_abs_two_block`); the
      knot system is exact with ReLU, SiLU and GELU units at a finite scale; GPT-2 and OPT-125m end to
      end in a stock LlamaForCausalLM.
  sink <- q-bias host with a spare zero-frequency pair:  a host-only marker supplies the sink key
      (Lean `sink_sub_plain`); gpt-oss-tinystories end to end with stock Qwen2Attention modules.

Real-checkpoint checks skip if the checkpoint is not in the local HF cache.
Run:  pytest tests/test_bos_constructions.py -q
"""
import math

import pytest

torch = pytest.importorskip("torch")

from hf.bos_constructions import (F64, SINK_REPO, _act, _prompts, _zero_rope, agreement, exact_norms,  # noqa: E402
                                  gpt2_spec, kink_gates, learned_abs_host, load_sink_target, marker_weights,
                                  opt_spec, prefix_check, sink_host, solve_units, write_bos_prefix)


def _cached(repo):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:                                   # noqa: BLE001
        return False


# --- the knot system ----------------------------------------------------------------------------------

@pytest.mark.parametrize("act", ["relu", "silu", "gelu_new"])
@pytest.mark.parametrize("orient", ["below", "above"])
def test_knot_system_exact_at_finite_scale(act, orient):
    """H c = Y over the T + 1 knots is solvable for a ROUGH table with ReLU, SiLU and GELU units alike
    (exact at the knots, not a limit)."""
    T = 255
    u = marker_weights(T, math.log(T))
    Y = torch.randn(T + 1, 3, dtype=F64, generator=torch.Generator().manual_seed(0))
    a, g = kink_gates(u, 1.0, orient=orient)
    c, res = solve_units(_act(act), a, g, u, torch.ones_like(u), torch.ones_like(u), Y)
    H = _act(act)(u[:, None] * a[None] + g[None])
    mag = (H.abs() @ c.abs()).max().item()             # the size of the sums the MLP forms
    assert res < 1e-14 * mag and res < 1e-8, (res, mag)


def test_zero_logit_knots_are_the_lean_ones():
    """L = 0: the marker's weight is 1 / (t + 1), the Lean head's 1 - t / (t + 1)."""
    t = torch.arange(10, dtype=F64)
    assert torch.allclose(marker_weights(9, 0.0), 1 / (t + 1))


# --- the two-block prefix on a default (bias-free) stock Llama ------------------------------------------

def _rand_spec(V=61, d=16, T=47, H=4, bos=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(E=torch.randn(V, d, dtype=F64, generator=g), Ptab=torch.randn(T + 1, d, dtype=F64, generator=g),
                bos=bos, H=H)


@pytest.mark.parametrize("act", ["silu", "gelu_new", "relu"])
def test_prefix_exact_on_default_llama(act):
    """Residual after the two extra blocks = [e(x_t) + P_t; 0] at every position (fp64), on a stock
    LlamaForCausalLM with no biases and the given MLP activation (SiLU: the stock default)."""
    r = prefix_check(_rand_spec(), act=act, n_seq=3, T_seq=20)
    assert r["max_err"] < 1e-10 * r["scale"], r


def test_prefix_needs_the_marker():
    """Negative control: the same host on a sequence WITHOUT the marker at position 0 is far off (no
    construction can do it: `no_learned_abs_without_marker`)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    s = _rand_spec()
    V, d = s["E"].shape
    T = s["Ptab"].shape[0] - 1
    host = LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=d + 5, intermediate_size=T + 6,
                                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                                        head_dim=d // 4, attn_implementation="sdpa")).to(F64).eval()
    _zero_rope(host)
    write_bos_prefix(host, s["E"], s["Ptab"], s["bos"])
    out = {}
    host.model.layers[1].register_forward_hook(lambda m, a, o: out.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    ids = torch.arange(1, 21)[None]                                    # no marker anywhere
    with torch.no_grad(), exact_norms():
        host.model(ids)
    want = s["E"][ids[0]] + s["Ptab"][:20]
    assert (out["h"][0, :, :d] - want).abs().max() > 1e-2


# --- whole learned-abs models ---------------------------------------------------------------------------

def _rand_gpt2(seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=97, n_embd=32, n_layer=2, n_head=4, n_positions=64, bos_token_id=96,
                     eos_token_id=96, attn_implementation="sdpa")
    m = GPT2LMHeadModel(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "ln" in n:                                                  # non-trivial LayerNorm affine
                p.add_(0.2 * torch.randn_like(p))
    return m


def test_learned_abs_random_gpt2_exact():
    """A random GPT-2 (LayerNorm with random affine, gelu_new, learned positions) in a stock Llama host:
    logits equal to fp64 rounding on inputs that start with the BOS, over the whole context."""
    tgt = _rand_gpt2()
    host = learned_abs_host(gpt2_spec(tgt))
    ids = torch.randint(0, 96, (2, 64), generator=torch.Generator().manual_seed(1))
    ids[:, 0] = 96
    with torch.no_grad(), exact_norms():
        d = (tgt(ids).logits - host(ids).logits).abs().max().item()
    assert d < 1e-9, d


def test_learned_abs_logit_bias_is_the_final_beta():
    """Without the lm_head bias (the final LayerNorm's beta as a logit shift) the host is off: the one
    non-config attribute of the GPT-2 build is the LayerNorm fold's, not the marker's."""
    tgt = _rand_gpt2()
    host = learned_abs_host(gpt2_spec(tgt), logit_bias=False)
    ids = torch.randint(0, 96, (1, 20), generator=torch.Generator().manual_seed(2))
    ids[:, 0] = 96
    with torch.no_grad(), exact_norms():
        d = (tgt(ids).logits - host(ids).logits).abs().max().item()
    assert d > 1e-3


@pytest.mark.skipif(not _cached("gpt2"), reason="gpt2 not in the HF cache")
def test_learned_abs_gpt2_real():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tgt = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True, dtype=F64,
                                               attn_implementation="sdpa").eval()
    host = learned_abs_host(gpt2_spec(tgt))
    with exact_norms():
        r = agreement(tgt, host, _prompts(tok, bos=50256, n=3), n_new=12, bos=50256)
    assert r["max_logit_diff"] < 1e-6 and r["top1"] == 1.0 and r["ex64"] == 1.0, r


@pytest.mark.skipif(not _cached("facebook/opt-125m"), reason="opt-125m not in the HF cache")
def test_learned_abs_opt_real():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/opt-125m", local_files_only=True)
    tgt = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", local_files_only=True, dtype=F64,
                                               attn_implementation="sdpa").eval()
    host = learned_abs_host(opt_spec(tgt))
    assert host.config.hidden_act == "relu" and host.config.hidden_size == 768 + 5
    with exact_norms():
        r = agreement(tgt, host, _prompts(tok, n=3), n_new=12, bos=2)
    assert r["max_logit_diff"] < 1e-6 and r["top1"] == 1.0 and r["ex64"] == 1.0, r


# --- sinks ----------------------------------------------------------------------------------------------

def _rand_gptoss(seed=0):
    from transformers import GptOssConfig, GptOssForCausalLM
    torch.manual_seed(seed)
    cfg = GptOssConfig(vocab_size=101, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       intermediate_size=16, num_local_experts=4, num_experts_per_tok=2, num_hidden_layers=2,
                       layer_types=["sliding_attention", "full_attention"], sliding_window=64, rope_theta=1e4,
                       rope_scaling=None, max_position_embeddings=256, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    m = GptOssForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if "norm" not in n else 0.1) + (1.0 if "norm" in n else 0.0))
            if n.endswith("sinks"):
                p.copy_(2.0 * torch.randn_like(p))                         # large sinks: a real test
    return m


def _sink_diff(tgt, host, T=40, seed=3):
    V = tgt.config.vocab_size
    x = torch.randint(0, V, (1, T), generator=torch.Generator().manual_seed(seed))
    hx = torch.cat([torch.tensor([[host.marker_id]]), x], 1)
    with torch.no_grad(), exact_norms():
        lt = tgt(x).logits
        lh = host(hx, position_ids=torch.arange(T + 1)[None] - 1).logits[:, 1:, :V]
    return (lt - lh).abs().max().item()


def test_sink_random_gptoss_exact():
    tgt = _rand_gptoss()
    host = sink_host(tgt)
    a = host.model.layers[0].self_attn
    assert type(a).__name__ == "Qwen2Attention" and a.o_proj.bias is None and not hasattr(a, "sinks")
    assert a.head_dim == tgt.config.head_dim + 2
    assert _sink_diff(tgt, host) < 1e-11


def test_sink_needs_the_q_bias():
    """Negative control: drop the constant spare query (the q bias) and the sink logit is 0 instead of
    s_h, so the host is off."""
    tgt = _rand_gptoss()
    host = sink_host(tgt)
    hd = tgt.config.head_dim
    with torch.no_grad():
        for lh in host.model.layers:
            b = lh.self_attn.q_proj.bias.view(tgt.config.num_attention_heads, hd + 2)
            b[:, hd // 2] = 0.0
    assert _sink_diff(tgt, host) > 1e-3


def test_sink_needs_the_zero_frequency_pair():
    """Negative control: give the spare pair a nonzero frequency and the sink logit depends on the query
    position (`sink_needs_zero_freq`)."""
    from hf.partial_rope_host import set_inv_freq
    tgt = _rand_gptoss()
    host = sink_host(tgt)
    inv = host.model.rotary_emb.inv_freq.clone()
    inv[tgt.config.head_dim // 2] = 0.3
    set_inv_freq(host.model.rotary_emb, inv)
    assert _sink_diff(tgt, host) > 1e-3


@pytest.mark.skipif(not _cached(SINK_REPO), reason=f"{SINK_REPO} not in the HF cache")
def test_sink_gptoss_tinystories_real():
    tok, tgt = load_sink_target()
    host = sink_host(tgt)
    with exact_norms():
        r = agreement(tgt, host, _prompts(tok, story=True, n=3), n_new=16,
                      prefix=torch.tensor([[host.marker_id]]), V=tgt.config.vocab_size)
    assert r["max_logit_diff"] < 1e-10 and r["top1"] == 1.0 and r["ex64"] == 1.0, r
