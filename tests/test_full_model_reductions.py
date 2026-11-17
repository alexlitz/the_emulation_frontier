"""A whole bunch of subset relations, verified on FULL models (embeddings → decoder
layers → LM-head logits), not just single ops.

The `make_elephantine` builder composes several reductions at once across a whole
`LlamaForCausalLM` — KV-head carrying (GQA/MQA ⊆ MHA), the dense-SwiGLU ⊆ tied-MoE
collapse, NoPE ⊆ RoPE, narrow ⊆ wide FFN (zero-padded), and identity SSM layers — so a
matrix over its target shape and its inflate/efficient modes checks all of them together,
token-for-token. The per-op proofs on stock modules live in tests/test_hf_modules.py;
the whole-model composition is tests/test_full_emulation.py.

Run:  pytest tests/test_full_model_reductions.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _llama(dtype, hidden=32, layers=3, heads=4, kv=2, inter=64, vocab=128):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=kv, intermediate_size=inter, vocab_size=vocab,
                      max_position_embeddings=64, rms_norm_eps=1e-5, hidden_act="silu",
                      attn_implementation="eager", tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).to(dtype).eval()


# each row is a whole-model emulation composing several reductions at once
@pytest.mark.parametrize("kv,experts,ffn_mult,depth_mult,efficient,label", [
    (4,  4, 1, 2, False, "MHA target, minimal hybrid"),
    (2,  8, 1, 2, False, "GQA target"),
    (1,  4, 1, 2, False, "MQA target"),
    (2,  8, 4, 3, False, "GQA + inflated (4x wider FFN, 3x depth)"),
    (2,  8, 1, 2, True,  "GQA + efficient (top_k=1)"),
    (1, 16, 8, 4, True,  "MQA + inflated + efficient (73x, 1 expert)"),
])
def test_elephantine_emulates_full_model(kv, experts, ffn_mult, depth_mult, efficient, label):
    from elephantine.make_elephantine import emulate
    _, _, d = emulate(_llama(torch.float64, kv=kv), experts, ffn_mult, depth_mult, efficient)
    assert d < 1e-9, f"{label}: max|Δ logits| = {d}"        # token-for-token to fp64 noise


def test_efficient_matches_full_and_routes_to_one_expert():
    import copy
    from elephantine.make_elephantine import emulate
    tgt = _llama(torch.float64, kv=2)
    _, cf, df = emulate(copy.deepcopy(tgt), num_experts=8, efficient=False)
    _, ce, de = emulate(copy.deepcopy(tgt), num_experts=8, efficient=True)
    assert cf.num_experts_per_tok == 8 and ce.num_experts_per_tok == 1
    assert df < 1e-9 and de < 1e-9                          # both exact; efficient computes 1/8 the experts


@pytest.mark.parametrize("hidden,layers,heads,kv,inter", [
    (48, 4, 6, 3, 96),      # bigger, GQA
    (64, 2, 8, 8, 176),     # MHA, wide FFN
])
def test_elephantine_across_shapes(hidden, layers, heads, kv, inter):
    from elephantine.make_elephantine import emulate
    tgt = _llama(torch.float64, hidden=hidden, layers=layers, heads=heads, kv=kv, inter=inter)
    _, _, d = emulate(tgt, num_experts=4)
    assert d < 1e-9


@pytest.mark.parametrize("experts,ffn_mult,depth_mult,efficient", [
    (4, 1, 2, False),    # minimal
    (8, 1, 2, True),     # efficient (top_k=1)
    (8, 4, 3, False),    # inflated
])
def test_greedy_generation_is_token_for_token(experts, ffn_mult, depth_mult, efficient):
    """Not just matching logits: the actual GREEDY-decoded token stream is identical."""
    import copy
    from elephantine.make_elephantine import emulate, nope_ize
    tgt = _llama(torch.float64, hidden=48, heads=6, kv=2, inter=96, vocab=200)
    nope_ize(tgt)                                          # target the hybrid reproduces
    prompt = torch.randint(0, 200, (1, 6))
    gk = dict(do_sample=False, max_new_tokens=24, use_cache=True,
              attention_mask=torch.ones_like(prompt), pad_token_id=0)
    ref = tgt.generate(prompt, **gk)
    eleph, _, _ = emulate(copy.deepcopy(tgt), experts, ffn_mult, depth_mult, efficient)
    assert torch.equal(ref, eleph.generate(prompt, **gk))  # same tokens, every step
