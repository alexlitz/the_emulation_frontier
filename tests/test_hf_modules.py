"""Emulation reductions proven against REAL, UNMODIFIED HuggingFace modules.

Unlike tests/test_constructions.py (which re-derives each identity with bespoke
fp64 tensor math), these instantiate stock `transformers` classes and show that a
general architecture reproduces a special case forward-for-forward using ONLY a
config change and/or a state_dict remap -- no code edits to the modules.

Run:  pytest tests/test_hf_modules.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


# --- GQA / MQA is a special case of MHA (LlamaModel, tiled K/V weights) -------

@pytest.mark.parametrize("n_kv", [1, 2, 4])   # MQA (1) and GQA (2, 4)
def test_mha_emulates_gqa_via_tiled_kv(n_kv):
    from hf.gqa_mha import build
    assert build(n_heads=8, n_kv=n_kv) < 1e-9    # bit-exact: it comes out 0.0


# --- dense FFN is a special case of MoE (MixtralSparseMoeBlock, tied experts) --

def test_moe_emulates_dense_top1_exact():
    from hf.moe_dense import build
    assert build(k=1) < 1e-12                    # single expert: exact


@pytest.mark.parametrize("k", [2, 4])
def test_moe_emulates_dense_topk(k):
    from hf.moe_dense import build
    # exact up to HF's hardcoded float32 router softmax (renormalized top-k gates)
    assert build(k=k) < 1e-6


# --- looped / recurrent-depth is a special case of an untied stack ------------

@pytest.mark.parametrize("b,K", [(1, 6), (2, 4), (3, 3)])
def test_untied_stack_emulates_looped(b, K):
    from hf.looped_untied import build
    assert build(b=b, K=K) < 1e-9                # tied deep LlamaModel == block looped K times


# --- local (sliding-window) attention is a special case of full attention -----

@pytest.mark.parametrize("W", [2, 4, 8])
def test_full_attention_emulates_local_via_band_mask(W):
    from hf.local_full import build
    eq, diff = build(W=W)
    assert eq < 1e-9      # unmodified full Qwen2Model + band mask == sliding-window model
    assert diff > 1e-3    # and it is NOT trivially equal (plain-full genuinely differs)


# --- NoPE is a special case of RoPE (LlamaModel, inv_freq -> 0) ---------------

def test_nope_is_rope_with_zero_frequency():
    from hf.nope_rope import build
    rope_change, nope_change = build()
    assert rope_change > 1e-5      # RoPE encodes relative distance (stretch changes output)
    assert nope_change < 1e-12     # inv_freq=0 -> position-invariant == NoPE
