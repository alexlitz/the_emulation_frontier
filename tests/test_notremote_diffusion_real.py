"""The elephantine emulates a REAL diffusion model -- without running its remote code.

tests/test_diffusion_real.py loads nvidia/Nemotron-Labs-Diffusion-3B through trust_remote_code. These tests
run the same REAL checkpoint weights through a labelled reference transcription of its encoder
(tests/refs/nemotron_diffusion.py: the stock transformers 5.18 Ministral3Model with the remote's diffusion
attention branch transcribed, read as text), in the TF5 subprocess. fp32, as the originals (the checkpoint
is bf16; the widening is an exact re-indexing, so it reproduces the forward to rounding of the same ops).
Skips if the checkpoint isn't cached.

Run:  pytest tests/test_notremote_diffusion_real.py -q
"""
import json
import os
import re
import subprocess
import sys

import pytest

from tests.test_universal_elephantine import TF5, TF5_PRELUDE

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Nemotron-Labs-Diffusion-3B/snapshots/"
                          "0d51902da1f8869f83413ce642fab402fa5641e0/model.safetensors")


def _run(body):
    if not os.path.isdir(TF5):
        pytest.skip("transformers 5.18 not unpacked (python -m elephantine.fetch_upstream_refs)")
    if not os.path.exists(SNAP):
        pytest.skip("Nemotron-Labs-Diffusion-3B weights not cached")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, root]))
    r = subprocess.run([sys.executable, "-c", TF5_PRELUDE + body], env=env, capture_output=True, text=True, timeout=1800)
    m = re.search(r"RESULT (.*)", r.stdout)
    assert m, r.stdout[-2000:] + r.stderr[-4000:]
    return json.loads(m.group(1))


def test_real_diffusion_encoder_is_bidirectional_notremote():
    """Position 0 sees a future token (7) in the diffusion encoder; with the diffusion branch switched off
    (the AR path, diffusion_lm False) it does not."""
    r = _run('''
import json
import tests.refs.nemotron_diffusion as nd
cfg, enc = nd.encoder()
torch.manual_seed(0)
x = torch.randint(0, cfg.vocab_size, (1, 10))
x2 = x.clone(); x2[0, 7] = (x2[0, 7] + 1) % cfg.vocab_size
def moved():
    with torch.no_grad():
        return (enc(input_ids=x).last_hidden_state.double() - enc(input_ids=x2).last_hidden_state.double())[0, 0].abs().max().item()
bi = moved()
for l in enc.layers:
    l.self_attn.diffusion_lm = False
print("RESULT", json.dumps({"bi": bi, "ar": moved()}))
''')
    assert r["bi"] > 1e-2, "a diffusion encoder must be bidirectional (pos 0 sees a future token)"
    assert r["ar"] == 0.0, r                         # the branch switched off: pos 0 is blind to the future


def test_wider_host_reproduces_real_diffusion_forward_exactly_notremote():
    """Widen GQA -> MHA in place (the bidirectional forward and the YaRN / Llama-4 query scale stay intact) and
    require the wider encoder to reproduce the forward; the causal (AR) forward of the same weights differs."""
    r = _run('''
import json
import tests.refs.nemotron_diffusion as nd
cfg, enc = nd.encoder()
hd, kvA, HH = cfg.head_dim, cfg.num_key_value_heads, cfg.num_attention_heads
torch.manual_seed(0)
x = torch.randint(0, cfg.vocab_size, (1, 12))
with torch.no_grad():
    ref = enc(input_ids=x).last_hidden_state.clone()
    for l in enc.layers:
        l.self_attn.diffusion_lm = False
    ar = enc(input_ids=x).last_hidden_state.clone()
    for l in enc.layers:
        l.self_attn.diffusion_lm = True
    rep = HH // kvA
    for layer in enc.layers:                                         # GQA kv -> MHA, in place
        a = layer.self_attn
        for pr in ("k_proj", "v_proj"):
            w = getattr(a, pr).weight.view(kvA, hd, -1)
            new = torch.nn.Linear(w.shape[-1], HH * hd, bias=False).to(torch.float32)
            new.weight.copy_(w[torch.arange(HH) // rep].reshape(HH * hd, -1))
            setattr(a, pr, new)
        a.num_key_value_groups = 1
    wide = enc(input_ids=x).last_hidden_state
print("RESULT", json.dumps({"wide": (ref - wide).abs().max().item(), "ar": (ref - ar).abs().max().item(),
                            "scale": ref.abs().max().item()}))
''')
    assert r["wide"] < 1e-9, r                       # the wider host reproduces the diffusion forward
    assert r["ar"] > 1e-2 * r["scale"], r            # the bidirectional branch matters on the real weights
