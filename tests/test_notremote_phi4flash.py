"""Phi-4-mini-flash (SambaY) without running remote code: the universal host (from_phi4flash) against the
REFERENCE transcription in tests/refs/phi4flash.py (microsoft/Phi-4-mini-flash-reasoning modelling code read as
text), at fp64.

Run:  pytest tests/test_notremote_phi4flash.py -q
"""
import copy
import math

import torch

from tests.refs import phi4flash as P
from tests.test_universal_elephantine import _ids, _rand_

F64 = torch.float64


def _rel(ref, ours):
    return (ref - ours).abs().max().item() / max(ref.abs().max().item(), 1.0)


def _model():
    cfg = P.Phi4FlashConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                            num_attention_heads=4, num_key_value_heads=2, sliding_window=4, mb_per_layer=2,
                            mamba_d_state=8, mamba_d_conv=4, mamba_expand=2)
    t = P.Phi4FlashForCausalLM(cfg).to(F64).eval()
    for l in t.model.layers:
        if l.use_mamba:
            l.attn.use_fast_path = False
    return _rand_(t)


@torch.no_grad()
def test_same_host_carries_phi4flash_sambay_notremote():
    """from_phi4flash equals the REFERENCE SambaY (Mamba, GMU, sliding-window and full differential attention, YOCO
    cross-attention) to <= 1e-12 relative at fp64. Switched-off branches, each moving the output and (where the
    reference can express it) matching the reference with the same branch off:
      GMU gate      host ssm.gate zeroed on the GMU layers == reference with the GMU in_proj zeroed (silu(0) * y = 0)
      diff lambda   host o_hmix's -lambda entries zeroed == reference with lambda_q1 . lambda_k1 = log(1 - lambda_init),
                    lambda_q2 = lambda_k2 = 0 (lambda_full = 0)
      sliding win   host window = inf on the SWA layers == reference with config.sliding_window[i] = None
      SSM register  host y_read = 0 on the GMU layers (margin only)."""
    from elephantine.universal import from_phi4flash
    t = _model()
    kinds = [("gmu" if l.attn.yoco_cross else "mamba") if l.use_mamba else ("xattn" if l.attn.yoco_cross else "attn")
             for l in t.model.layers]
    assert {"gmu", "mamba", "xattn", "attn"} <= set(kinds), kinds
    x = _ids(97, T=10)
    ref = t(x, use_cache=False).logits
    gap = _rel(ref, from_phi4flash(t)(x))
    assert gap <= 1e-12, gap

    L = t.config.num_hidden_layers
    gmu = [i for i, k in enumerate(kinds) if k == "gmu"]
    att = [i for i, k in enumerate(kinds) if k in ("attn", "xattn")]
    swa = [i for i in att if t.model.layers[i].attn.config.sliding_window is not None
           and t.model.layers[i].attn.config.sliding_window[i] is not None]
    assert swa and gmu

    # GMU gate off
    h = from_phi4flash(t)
    for i in gmu:
        h.layers[i].ssm.gate.zero_()
    off = h(x)
    t2 = copy.deepcopy(t)
    for i in gmu:
        t2.model.layers[i].attn.in_proj.weight.zero_()
    assert _rel(ref, off) > 1e-3, _rel(ref, off)
    assert _rel(t2(x, use_cache=False).logits, off) <= 1e-12

    # differential lambda off (lambda_full = 0)
    h = from_phi4flash(t)
    for i in att:
        h.layers[i].attn.o_hmix[1::2].zero_()
    off = h(x)
    t2 = copy.deepcopy(t)
    for i in att:
        ia = t2.model.layers[i].attn.inner_cross_attn
        for p in (ia.lambda_q1, ia.lambda_k1, ia.lambda_q2, ia.lambda_k2):
            p.zero_()
        ia.lambda_q1[0] = math.log(1 - ia.lambda_init); ia.lambda_k1[0] = 1.0
    assert _rel(ref, off) > 1e-3, _rel(ref, off)
    assert _rel(t2(x, use_cache=False).logits, off) <= 1e-12

    # sliding window off
    h = from_phi4flash(t)
    for i in swa:
        h.layers[i].attn.window.fill_(float("inf"))
    off = h(x)
    t2 = copy.deepcopy(t)
    for i in swa:
        t2.model.layers[i].attn.config.sliding_window[i] = None
    assert _rel(ref, off) > 1e-3, _rel(ref, off)
    assert _rel(t2(x, use_cache=False).logits, off) <= 1e-12

    # GMU reads the layer-L/2 SSM output register
    assert kinds[L // 2] == "mamba" and t.model.layers[L // 2].attn.yoco_kv
    h = from_phi4flash(t)
    for i in gmu:
        h.layers[i].ssm.y_read.zero_()
    off = h(x)
    assert _rel(ref, off) > 1e-3, _rel(ref, off)
