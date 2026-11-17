"""The general elephantine builder: an arbitrary Llama-family decoder LM is reproduced
token-for-token by a Jamba elephantine that embeds its (smaller) tensors into the bigger
hybrid architecture. Covers MHA / GQA / MQA targets, AND a HYBRID (Jamba) target whose
Mamba-SSM recurrences are CARRIED (not set to identity) -- the '50 Mamba-family' claim."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import harness as H


def _tiny_target(dtype, layers=3, kv=2):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=32, num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=kv, intermediate_size=64, vocab_size=128,
                      max_position_embeddings=64, rms_norm_eps=1e-5, hidden_act="silu",
                      attn_implementation="eager", tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).to(dtype).eval()


@pytest.mark.parametrize("kv", [4, 2, 1])   # MHA (4), GQA (2), MQA (1) targets
def test_elephantine_emulates_arbitrary_target_fp64(kv):
    from elephantine.make_elephantine import emulate
    _, cfg, d = emulate(_tiny_target(torch.float64, kv=kv), num_experts=4)
    assert cfg.layers_block_type.count("attention") == 3        # one attn layer per target layer
    assert d < 1e-9                                             # bit-exact


def test_elephantine_emulates_arbitrary_target_fp32():
    from elephantine.make_elephantine import emulate
    _, _, d = emulate(_tiny_target(torch.float32), num_experts=4)
    assert d < 1e-4                                             # fp32 = model precision


def _tiny_hybrid_target(dtype, num_experts=4):
    # a HYBRID (Jamba) target: interleaved Mamba-SSM + attention + MoE layers
    from transformers import JambaConfig
    from transformers.models.jamba.modeling_jamba import JambaForCausalLM
    torch.manual_seed(0)
    cfg = JambaConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, intermediate_size=48, vocab_size=64,
                      attn_layer_period=2, attn_layer_offset=1, expert_layer_period=2,
                      expert_layer_offset=1, num_experts=num_experts, num_experts_per_tok=num_experts,
                      use_mamba_kernels=False, attn_implementation="eager", tie_word_embeddings=False)
    return JambaForCausalLM(cfg).to(dtype).eval()


@pytest.mark.parametrize("ffn_mult", [1, 2])
def test_elephantine_carries_a_hybrid_target(ffn_mult):
    # the SSM layers CARRY the target's Mamba recurrence (not identity); a wider-FFN host
    # reproduces the smaller hybrid token-for-token.
    from elephantine.make_elephantine import emulate_hybrid
    tgt = _tiny_hybrid_target(torch.float64)
    _, _, d = emulate_hybrid(tgt, ffn_mult=ffn_mult)
    assert d < 1e-9, f"hybrid emulation max|Δ| = {d:.2e}"


def test_diffusion_elephantine_emulates_a_bidirectional_forward():
    # a DIFFUSION LM's forward is a bidirectional transformer; a capacity-widened full-
    # attention host reproduces it token-for-token (the gen wall is about the decode loop,
    # not the forward). Uses a Llama target run with a full (non-causal) attention mask.
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from elephantine.hidden_width import _rms_f64, convert
    tgt = _tiny_target(torch.float64, kv=2)
    host = convert(tgt, kv=tgt.config.num_attention_heads, inter=tgt.config.intermediate_size + 32,
                   layers=tgt.config.num_hidden_layers + 1, hidden=tgt.config.hidden_size + 16)
    ids = torch.randint(0, 128, (1, 8))
    full = torch.zeros(1, 1, 8, 8, dtype=torch.float64)                       # bidirectional mask
    # sanity: the forward really IS bidirectional under a full mask
    base = tgt(ids, attention_mask=full).logits.double()
    moved = ids.clone(); moved[0, 5] = (moved[0, 5] + 1) % 128
    assert (base - tgt(moved, attention_mask=full).logits.double())[0, 0].abs().max() > 1e-3
    # and the host reproduces the bidirectional forward token-for-token
    na, nb = H.modules_of_type(tgt, LlamaRMSNorm), H.modules_of_type(host, LlamaRMSNorm)
    with H.patched_forwards(na, _rms_f64), H.patched_forwards(nb, _rms_f64):
        la = tgt(ids, attention_mask=full).logits.double()
        lb = host(ids, attention_mask=full).logits.double()
    # token-for-token on this tiny (ill-conditioned) synthetic model; the construction is
    # exact on a real model (test_multi_axis_conversion_on_real_model: 1.1e-13 on Llama-3.2-1B).
    assert (la.argmax(-1) == lb.argmax(-1)).all() and (la - lb).abs().max() < 1e-6


@pytest.mark.parametrize("causal", [True, False])
def test_one_universal_host_spans_ar_and_diffusion(causal):
    # a SINGLE full-attention host reproduces the target under a CAUSAL mask (AR) and a FULL
    # mask (diffusion) -- one architecture across both paradigms; only the mask switches.
    from elephantine.make_elephantine import emulate_universal
    tgt = _tiny_target(torch.float64, kv=2)
    _, d = emulate_universal(tgt, causal=causal, kv=tgt.config.num_attention_heads,
                             inter=tgt.config.intermediate_size + 32,
                             hidden=tgt.config.hidden_size + 16)
    assert d < 1e-6, f"universal host (causal={causal}) max|Δ| = {d:.2e}"


def test_elephantine_actually_uses_the_carried_ssm():
    # sanity: the target's Mamba layers do real work, so carrying them matters (a
    # softmax-only emulation with identity SSM would NOT reproduce a hybrid target).
    tgt = _tiny_hybrid_target(torch.float64)
    ids = torch.randint(0, 64, (2, 8))
    base = tgt(ids).logits.double()
    mamba = next(l.mamba for l in tgt.model.layers if hasattr(l, "mamba"))
    saved = mamba.out_proj.weight.data.clone()
    mamba.out_proj.weight.data.zero_()                          # kill one SSM layer
    changed = (base - tgt(ids).logits.double()).abs().max().item()
    mamba.out_proj.weight.data.copy_(saved)
    assert changed > 1e-6, "the Mamba recurrence must be load-bearing for the carry to matter"


@pytest.mark.parametrize("argv,want", [([], False), (["--trust-remote-code"], True)])
def test_cli_runs_no_remote_code_by_default(argv, want, monkeypatch):
    """The command line loads the target with trust_remote_code=False unless --trust-remote-code is given."""
    from elephantine import make_elephantine as M
    seen = {}

    class Stop(Exception):
        pass

    def fake(repo, **kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(M.AutoModelForCausalLM, "from_pretrained", fake)
    with pytest.raises(Stop):
        M.main(["some/repo"] + argv)
    assert seen["trust_remote_code"] is want
