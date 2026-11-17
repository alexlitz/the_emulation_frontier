"""dQwen3.5 (diffusion Qwen3.5) on the universal host, against a labelled reference transcription of its
remote modelling code (tests/refs/dqwen35.py, read as text, never run) built on the stock transformers 5.18
Qwen3_5ForCausalLM, in the TF5 subprocess.

Run:  pytest tests/test_notremote_dqwen35.py -q
"""
from tests.test_eleph_gaps_stream import LIFT
from tests.test_universal_elephantine import _tf5_run


def test_same_host_carries_dqwen35_diffusion_notremote():
    """dQwen3.5: Qwen3.5 with its full-attention layers made bidirectional (the gated-delta layers stay
    causal) and a BOS / logit shift. On the host (from_qwen35): the same weights run with causal=False,
    BOS prepended, last position dropped. fp64 with the fp32 casts lifted. Each branch switched off moves
    the logits: a causal host, a host without the BOS shift, and a reference with causal attention layers."""
    gap, scale = _tf5_run(LIFT + '''
import tests.refs.dqwen35 as dq
dq.BOS_TOKEN_ID = 5                                   # fit the tiny test vocab
from elephantine.universal import from_qwen35
kw = dict(vocab_size=97, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=8, linear_conv_kernel_dim=4, intermediate_size=48)
def build(**k):
    cfg = dq.DiffuQwen35Config(**kw)
    cfg._attn_implementation = "sdpa"                 # the remote seeds with sdpa: it honours is_causal=False
    t = dq.DiffuQwen35(cfg, **k).to(F64).eval()
    for l in t.model.layers:                          # CPU torch paths (fla kernels are GPU Triton)
        g = getattr(l, "linear_attn", None)
        if g is not None:
            g.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
            g.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
            g.causal_conv1d_fn = None
    return rand_(t)
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
t = build()
assert {l.block_type for l in t.model.layers} == {"linear_attention", "full_attention"}
x = ids(97)
with torch.no_grad():
    with lift():
        ref = t(x)
        ref_causal = build(bidirectional=False)(x)    # the remote's is_causal flip switched off
    assert ref.dtype == F64
    u = from_qwen35(t)
    bos = torch.full((x.shape[0], 1), 5)
    ours = u(torch.cat([bos, x], 1), causal=False)[:, :-1]
    assert rel(ref, u(torch.cat([bos, x], 1), causal=True)[:, :-1]) > 1e-3     # bidirectional attention matters
    assert rel(ref, u(x, causal=False)) > 1e-3                                  # the BOS shift matters
    assert rel(ref, ref_causal) > 1e-3
    report(ref, ours)
''')
    assert gap < 1e-12 * max(scale, 1), (gap, scale)
