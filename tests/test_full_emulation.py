"""FULL-MODEL emulation, end-to-end in the test suite.

A whole `LlamaForCausalLM` reproduces a DIFFERENT, more special `LlamaForCausalLM`
(GQA + narrow FFN) all the way to the LM-head logits, by composing two exact
reductions in the weights -- KV-head tiling (GQA subset MHA) and FFN zero-padding
(narrow subset wide) -- across every decoder layer. The `elephantine` build does
this at frontier scale on a hybrid; this is the lightweight, asserted version that
proves a real, unmodified `transformers` model does another model's WHOLE forward
by weight-setting alone.

Run:  pytest tests/test_full_emulation.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.parametrize("n_kv", [1, 2, 4])   # MHA host emulates MQA (1) and GQA (2, 4)
def test_full_causal_lm_emulation_fp64(n_kv):
    from hf.full_emulation import build
    assert build(n_kv=n_kv, dtype=torch.float64) < 1e-10   # bit-exact: comes out 0.0


@pytest.mark.parametrize("n_kv", [1, 2, 4])
def test_full_causal_lm_emulation_fp32(n_kv):
    from hf.full_emulation import build
    assert build(n_kv=n_kv, dtype=torch.float32) < 1e-4    # fp32 = model precision


def test_wide_host_is_not_a_trivial_copy():
    # the host is genuinely more general (MHA + 4x-wider FFN), not an identity copy,
    # and still reproduces the target exactly.
    from hf.full_emulation import build
    assert build(n_kv=2, inter_t=64, inter_h=256, dtype=torch.float64) < 1e-10
