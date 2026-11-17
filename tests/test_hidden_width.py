"""Hidden-width emulation is constructive and exact on a real model.

The one capacity axis `verify_edges.py` couldn't build ("needs norm compensation").
`elephantine/hidden_width.py` builds it: a wider-hidden host reproduces the narrower
model token-for-token, exact in fp64 (the native residual is only the model's own
fp32-internal RMSNorm). Skips if the checkpoint isn't cached.

Run:  pytest tests/test_hidden_width.py -q
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import harness as H
from elephantine.hidden_width import verify, widen


def test_hidden_width_is_constructive_and_exact():
    try:
        H.load("meta-llama/Llama-3.2-1B", dtype=torch.float64, device="cpu")
    except Exception as e:
        pytest.skip(f"Llama-3.2-1B unavailable: {type(e).__name__}")
    r = verify("meta-llama/Llama-3.2-1B", add=256)
    assert r["dB"] > r["dA"]
    assert r["exact"] < 1e-9, f"construction not exact in fp64: {r['exact']:.2e}"
    assert r["native"] < 5e-5, f"native forward drifted too far: {r['native']:.2e}"
    assert r["tokens_identical"], "greedy token stream diverged"


def test_hidden_width_odd_padding_still_exact():
    # a non-round widening (not a multiple of head_dim) must still be exact
    try:
        A, _, _ = H.load("meta-llama/Llama-3.2-1B", dtype=torch.float64, device="cpu")
    except Exception as e:
        pytest.skip(f"Llama-3.2-1B unavailable: {type(e).__name__}")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    B = widen(A, A.config.hidden_size + 100)
    ids = torch.randint(0, 2000, (1, 12), generator=torch.Generator().manual_seed(1))
    na, nb = H.modules_of_type(A, LlamaRMSNorm), H.modules_of_type(B, LlamaRMSNorm)
    from elephantine.hidden_width import _rms_f64
    with H.patched_forwards(na, _rms_f64), H.patched_forwards(nb, _rms_f64):
        d = (A(ids).logits.double() - B(ids).logits.double()).abs().max().item()
    assert d < 1e-9, f"odd widening not exact: {d:.2e}"


def test_multi_axis_conversion_on_real_model():
    # convert a REAL trained model across ALL FOUR capacity axes at once (MHA + wider FFN
    # + deeper + wider hidden) and reproduce it token-for-token, exact in fp64.
    try:
        A, _, _ = H.load("meta-llama/Llama-3.2-1B", dtype=torch.float64, device="cpu")
    except Exception as e:
        pytest.skip(f"Llama-3.2-1B unavailable: {type(e).__name__}")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from elephantine.hidden_width import _rms_f64, convert
    ca = A.config
    B = convert(A, kv=ca.num_attention_heads, inter=ca.intermediate_size + 512,
                layers=ca.num_hidden_layers + 2, hidden=ca.hidden_size + 256)
    ids = torch.randint(0, 2000, (2, 16), generator=torch.Generator().manual_seed(0))
    na, nb = H.modules_of_type(A, LlamaRMSNorm), H.modules_of_type(B, LlamaRMSNorm)
    with H.patched_forwards(na, _rms_f64), H.patched_forwards(nb, _rms_f64):
        la, lb = A(ids).logits.double(), B(ids).logits.double()
    assert (la.argmax(-1) == lb.argmax(-1)).all(), "4-axis conversion not token-for-token"
    assert (la - lb).abs().max().item() < 1e-9, "4-axis conversion not exact in fp64"


def test_full_capacity_spine_constructs_and_rejects():
    # the whole spine (KV + FFN + depth + hidden) built from A's weights, on synthetic
    # Llama models (no download): every B>=A edge reproduces A token-for-token, every
    # narrower-hidden host is rejected.
    from elephantine.verify_edges import run_spine
    r = run_spine(24, seed=0)
    assert r["pos_ok"] == r["pos_n"] > 0, f"spine sufficiency {r['pos_ok']}/{r['pos_n']}"
    assert r["neg_reject"] == r["neg_n"] > 0, f"spine necessity {r['neg_reject']}/{r['neg_n']}"


def test_composed_type_plus_capacity_edge():
    # a TYPE-crossing edge built end-to-end: target = MQA + NoPE + narrow; host = MHA +
    # RoPE(zeroed) + wider FFN + deeper + wider hidden -> reproduces the target
    # token-for-token. Composes the kv-type + positional + capacity axes in one build.
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from elephantine.hidden_width import _rms_f64, build_caps, widen
    from elephantine.verify_edges import cfg
    torch.manual_seed(0)

    def zero_rope(m):
        re = m.model.rotary_emb
        re.inv_freq = torch.zeros_like(re.inv_freq)
        if hasattr(re, "original_inv_freq"):
            re.original_inv_freq = re.inv_freq

    from transformers import LlamaForCausalLM
    A = LlamaForCausalLM(cfg(4, 1, 64, 2)).to(torch.float64).eval()
    zero_rope(A)                                             # target is NoPE (position-invariant)
    B = widen(build_caps(A, kv=4, inter=96, layers=3), 4 * 16 + 16)   # MHA + wider + deeper + wider hidden
    zero_rope(B)                                            # RoPE host reproduces the NoPE target
    ids = torch.randint(0, 64, (2, 8))
    na, nb = H.modules_of_type(A, LlamaRMSNorm), H.modules_of_type(B, LlamaRMSNorm)
    with H.patched_forwards(na, _rms_f64), H.patched_forwards(nb, _rms_f64):
        la, lb = A(ids).logits.double(), B(ids).logits.double()
    assert (la.argmax(-1) == lb.argmax(-1)).all(), "composed kv+pos+capacity edge not token-for-token"


def _band_mask(T, W):
    i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    return torch.where((j <= i) & (j > i - W), 0.0, float("-inf")).to(torch.float64)[None, None]


def test_composed_window_plus_capacity_edge():
    # window axis: a full-attention host + band mask reproduces a local (sliding-window)
    # target, composed with kv + capacity, token-for-token.
    from transformers import LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from elephantine.hidden_width import _rms_f64, build_caps, widen
    from elephantine.verify_edges import cfg
    torch.manual_seed(0)
    A = LlamaForCausalLM(cfg(4, 1, 64, 2)).to(torch.float64).eval()
    B = widen(build_caps(A, kv=4, inter=96, layers=3), 4 * 16 + 16)
    ids, m = torch.randint(0, 64, (2, 8)), _band_mask(8, 4)
    na, nb = H.modules_of_type(A, LlamaRMSNorm), H.modules_of_type(B, LlamaRMSNorm)
    with H.patched_forwards(na, _rms_f64), H.patched_forwards(nb, _rms_f64):
        la = A(ids, attention_mask=m).logits.double()
        lb = B(ids, attention_mask=m).logits.double()
    assert (la.argmax(-1) == lb.argmax(-1)).all(), "composed window+kv+capacity edge not token-for-token"


def test_composed_mtp_plus_capacity_edge():
    # MTP axis: the target carries auxiliary prediction heads; the host reproduces the BASE
    # next-token forward (aux heads fold away), composed with kv + capacity.
    from transformers import LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from elephantine.hidden_width import _rms_f64, build_caps, widen
    from elephantine.verify_edges import cfg
    torch.manual_seed(0)
    base = LlamaForCausalLM(cfg(4, 1, 64, 2)).to(torch.float64).eval()
    aux = torch.nn.Linear(base.config.hidden_size, base.config.vocab_size, bias=False).to(torch.float64)

    def target(ids):
        o = base(ids, output_hidden_states=True)
        _ = aux(o.hidden_states[-1])                        # aux head runs; the base next-token ignores it
        return o.logits.double()
    B = widen(build_caps(base, kv=4, inter=96, layers=3), 4 * 16 + 16)
    ids = torch.randint(0, 64, (2, 8))
    na, nb = H.modules_of_type(base, LlamaRMSNorm), H.modules_of_type(B, LlamaRMSNorm)
    with H.patched_forwards(na, _rms_f64), H.patched_forwards(nb, _rms_f64):
        la, lb = target(ids), B(ids).logits.double()
    assert (la.argmax(-1) == lb.argmax(-1)).all(), "composed mtp+kv+capacity edge not token-for-token"
