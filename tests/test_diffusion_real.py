"""The elephantine emulates a REAL diffusion model, bit-for-bit.

Loads nvidia/Nemotron-Labs-Diffusion-3B (a real masked-diffusion LM), confirms its forward
is genuinely BIDIRECTIONAL, and shows a wider (MHA) host reproduces that bidirectional
forward EXACTLY (0.0) -- so the tie-and-zero capacity recipe carries a diffusion target,
not just an autoregressive one. Skips if the checkpoint isn't cached (7.8 GB, custom code).

Run:  pytest tests/test_diffusion_real.py -q
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

MODEL = "nvidia/Nemotron-Labs-Diffusion-3B"


def _load():
    from transformers import AutoConfig, AutoModel
    try:
        cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, local_files_only=True,
                                          trust_remote_code=True).eval()
    except Exception as e:
        pytest.skip(f"{MODEL} not cached / unloadable: {type(e).__name__}")
    return cfg, model.encoder


def test_real_diffusion_encoder_is_bidirectional():
    cfg, enc = _load()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        h0 = enc(input_ids=ids).last_hidden_state.double()
        ids2 = ids.clone(); ids2[0, 7] = (ids2[0, 7] + 1) % cfg.vocab_size
        moved = (h0 - enc(input_ids=ids2).last_hidden_state.double())[0, 0].abs().max().item()
    assert moved > 1e-2, "a diffusion encoder must be bidirectional (pos 0 sees a future token)"


def test_wider_host_reproduces_real_diffusion_forward_exactly():
    # widen GQA -> MHA IN PLACE (no re-instantiation, so the real bidirectional forward +
    # YaRN rope stay intact) and require the wider host to reproduce the forward bit-for-bit.
    cfg, enc = _load()
    hd, kvA, HH = cfg.head_dim, cfg.num_key_value_heads, cfg.num_attention_heads
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    full = torch.zeros(1, 1, 12, 12, dtype=torch.float32)                 # bidirectional mask
    with torch.no_grad():
        ref = enc(input_ids=ids, attention_mask=full).last_hidden_state.clone()
        rep = HH // kvA
        for layer in enc.layers:                                         # GQA kv -> MHA, in place
            a = layer.self_attn
            for pr in ("k_proj", "v_proj"):
                w = getattr(a, pr).weight.view(kvA, hd, -1)
                new = torch.nn.Linear(w.shape[-1], HH * hd, bias=False).to(torch.float32)
                new.weight.copy_(w[torch.arange(HH) // rep].reshape(HH * hd, -1))
                setattr(a, pr, new)
            if hasattr(a, "num_key_value_groups"):
                a.num_key_value_groups = 1
            if hasattr(a, "config"):
                a.config.num_key_value_heads = HH
        wide = enc(input_ids=ids, attention_mask=full).last_hidden_state
    assert (ref - wide).abs().max().item() < 1e-9, "wider host must reproduce the diffusion forward exactly"
