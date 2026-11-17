"""Granite-Switch against a labelled transcription of its source (tests/refs/granite_switch.py), not its own code.

The remote-code test (test_universal_elephantine.py::test_same_host_carries_granite_switch_alora) imports the
granite-switch package; this one runs the same configuration through the reference, read from that source as text.
"""
import torch

from tests.refs.granite_switch import GraniteSwitchForCausalLM, granite_switch_config
from tests.test_universal_elephantine import _lift_fp32, _rand_

F64 = torch.float64


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


@torch.no_grad()
def test_same_host_carries_granite_switch_alora_notremote():
    """Granite-Switch (single switch, LoRA on qkv / o / shared-MLP in and out): a switch head picks an adapter PER
    TOKEN from the control tokens seen so far, so an aLoRA adapter turns on mid-sequence; control tokens are
    exchanged for substitute tokens before embedding. from_granite_switch vs the reference at fp64 with the fp32
    casts (RMSNorm, rotary table, softmax) lifted: <= 1e-12 relative. Switching off the adapters (B = 0), the
    switch (every token on the base) or the token exchange each moves the logits."""
    from elephantine.universal import from_granite_switch
    cfg = granite_switch_config(vocab_size=97, hidden_size=64, intermediate_size=96, shared_intermediate_size=96,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, num_adapters=2,
        adapter_token_ids=[90, 91], adapter_substitute_token_ids=[5, 6], adapter_ranks=[4, 4],
        lora_target_modules=["qkv_proj", "o_proj", "shared_input_linear", "shared_output_linear"],
        embedding_multiplier=12.0, residual_multiplier=0.22, logits_scaling=10.0, attention_multiplier=1 / 16,
        rope_theta=1e7)
    t = _rand_(GraniteSwitchForCausalLM(cfg).to(F64).eval())
    for n, p in t.named_parameters():                               # non-zero adapters (B inits to 0)
        if "lora_B" in n:
            p.copy_(torch.randn_like(p) * 0.2)
    x = torch.randint(0, 88, (2, 10), generator=torch.Generator().manual_seed(0))
    x[0, 4], x[1, 6] = 90, 91                                       # adapter 1 from t=4, adapter 2 from t=6
    with _lift_fp32():
        ref = t(x)
    ai = t.model._last_adapter_indices
    assert ai[0].tolist() == [0] * 4 + [1] * 6 and ai[1].tolist() == [0] * 6 + [2] * 4
    assert ref.dtype == F64
    host = from_granite_switch(t)
    assert _rel(ref, host(x)) < 1e-12, _rel(ref, host(x))
    h = from_granite_switch(t)                                      # the adapters really change the output
    for l in h.layers:
        for P in list(l.attn.lb.values()) + list(l.ffn.lb.values()):
            P.zero_()
    assert _rel(ref, h(x)) > 1e-3
    h = from_granite_switch(t)                                      # the switch: no adapter ever selected
    h.sw_val.zero_()
    assert _rel(ref, h(x)) > 1e-3
    h = from_granite_switch(t)                                      # the token exchange: control rows kept
    for c in cfg.adapter_token_ids:
        h.embed.weight[c] = t.model.embed_tokens.weight[c] * cfg.embedding_multiplier
    assert _rel(ref, h(x)) > 1e-3
