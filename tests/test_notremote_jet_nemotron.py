"""Jet-Nemotron without its remote code: the universal host against the labelled REFERENCE transcription
tests/refs/jet_nemotron.py (jet-ai/Jet-Nemotron-2B read as text, fla pieces from library sources)."""
import torch

from tests.test_universal_elephantine import F64, _ids, _rand_


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1)


def _cfg():
    from tests.refs.jet_nemotron import JetNemotronConfig
    cfg = JetNemotronConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, layer_types=["jet", "attn", "swa", "jet"],
        max_position_embeddings=128, rope_theta=10000.0, tie_word_embeddings=True,
        efficient_attention_config={"jet": {"mode": "chunk", "expand_v": 2, "num_heads": 2, "head_dim": 8, "norm_eps": 1e-5,
                                            "conv_size": 4, "dconv_generator_reduction": 8, "dconv_implementation": "naive"},
                                    "swa": {"window_size": 4}})
    cfg._attn_implementation = "sdpa"
    return cfg


@torch.no_grad()
def test_same_host_carries_jet_nemotron_notremote():
    """Jet-Nemotron (dconv_implementation="naive" = its torch dynamic conv): full attention, SWA (window 4 < T = 12,
    so it binds) and JetBlocks (dynamic per-token conv on v -> the SSM block's dconv branch, gated delta rule in the
    fused-recurrent branch it takes at q_len <= 64, gated RMSNorm). Reference: tests/refs/jet_nemotron.py at fp64
    (the remote's fp32 casts lifted), called like the old test with a 2-D all-ones attention mask (so the JetBlocks
    take the unpadded varlen path). Same config as test_same_host_carries_jet_nemotron. Bound: 1e-12 relative.
    Switched off on the host: the dynamic kernel (dc_w2 weight zeroed: only the static bias kernel remains) and the
    sliding window (window = inf); each moves the output by > 1e-3 relative and agrees with the reference with the
    same feature switched off (w2 weight zeroed / sliding_window None)."""
    from elephantine.universal import from_jet_nemotron
    from tests.refs.jet_nemotron import JetBlock, JetNemotronAttention, JetNemotronForCausalLM
    cfg = _cfg()
    torch.manual_seed(0)
    t = _rand_(JetNemotronForCausalLM(cfg).to(F64).eval())
    kinds = [type(l.self_attn) for l in t.model.layers]
    assert kinds == [JetBlock, JetNemotronAttention, JetNemotronAttention, JetBlock]
    assert [l.self_attn.sliding_window for l in t.model.layers[1:3]] == [None, 4]
    x = _ids(97, T=12)
    run = lambda: t(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
    ref = run()
    host = from_jet_nemotron(t)
    gap = _rel(ref, host(x))
    assert gap < 1e-12, gap

    # the dynamic kernel really matters, and the reference without it agrees with the host without it
    w2 = [l.self_attn.dynamic_conv1d.kernel_generator.w2.weight for l in t.model.layers if isinstance(l.self_attn, JetBlock)]
    saved = [w.clone() for w in w2]
    for l in host.layers:
        l.ssm.dc_w2.weight.zero_()
    off = host(x)
    assert _rel(ref, off) > 1e-3, _rel(ref, off)
    for w in w2:
        w.zero_()
    assert _rel(run(), off) < 1e-12, _rel(run(), off)
    for w, s in zip(w2, saved):
        w.copy_(s)

    # the sliding window really matters (fresh host), likewise against the reference with it off
    host = from_jet_nemotron(t)
    assert _rel(ref, host(x)) < 1e-12
    host.layers[2].attn.window.fill_(float("inf"))
    off = host(x)
    assert _rel(ref, off) > 1e-3, _rel(ref, off)
    t.model.layers[2].self_attn.sliding_window = None
    assert _rel(run(), off) < 1e-12, _rel(run(), off)
