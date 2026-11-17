"""PseudoBrain (pb27-vllm plugin) without running the plugin or vLLM: the universal host against the labelled
REFERENCE transcription in tests/refs/pseudobrain.py (the plugin's model code plus the vLLM layer semantics, fp64).

Run:  pytest tests/test_notremote_pseudobrain.py -q
"""
import pytest
import torch

from tests.refs.pseudobrain import PB27ForCausalLM
from tests.test_universal_elephantine import _pb27_checkpoint

IDS = [[5, 17, 3, 88, 42, 9, 61, 2, 33, 70, 12, 7, 50, 91, 4, 26, 38, 19, 77, 64, 1, 2, 3, 4],
       [9, 9, 1, 40, 23, 80, 6, 55, 13, 2, 71, 44, 30, 8]]


def _gap(ref, host):
    """max over sequences of max|ref - host| / max(|ref|, 1)."""
    with torch.no_grad():
        return max(((r := ref(torch.tensor(s))) - host(torch.tensor([s]))[0]).abs().max().item() / max(r.abs().max().item(), 1.0)
                   for s in IDS)


@pytest.mark.parametrize("moe", [False, True], ids=["PB27Qwen3", "PB27Qwen3Moe"])
def test_same_host_carries_pseudobrain_vllm_plugin_notremote(tmp_path, moe):
    """PseudoBrain-Qwen3-32B (PB27Qwen3ForCausalLM) / -Qwen3-Coder-30B-A3B (PB27Qwen3MoeForCausalLM) on the same tiny
    random checkpoint as the vLLM test: Qwen3 (-MoE) whose every attention layer has a W = pb_attention_window = 8
    sliding window (no sinks: the plugin warns pb_attention_sinks is not served), plus on pb_hybrid_layers [1, 3] a
    parallel Qwen3.5 gated-DeltaNet on the same normed input summed with attention (host: seq = 0). Host vs the
    fp64 reference <= 1e-12 relative; the window and the gdn branch each switched off on the host move the logits,
    and the reference with the same branch off (window = inf / no gdn) agrees with the switched-off host."""
    from elephantine.universal import from_pseudobrain
    cfg, sd = _pb27_checkpoint(str(tmp_path), moe)
    ref = PB27ForCausalLM.from_state_dict(cfg, sd)
    assert [l.self_attn.sliding_window for l in ref.model.layers] == [8] * 4
    assert [hasattr(l, "gdn") for l in ref.model.layers] == [False, True, False, True]
    host = from_pseudobrain(cfg, sd)
    gap = _gap(ref, host)
    print(f"host vs reference: {gap:.2e}")
    assert gap <= 1e-12, gap

    for l in host.layers:                                     # the window (on all layers) is exercised
        l.attn.window.fill_(float("inf"))
    moved = _gap(ref, host)
    ref_nw = PB27ForCausalLM.from_state_dict(cfg, sd, window=float("inf"))
    gap_nw = _gap(ref_nw, host)
    print(f"window off: host moves {moved:.2e}; reference(window = inf) vs host {gap_nw:.2e}")
    assert moved > 1e-2 and gap_nw <= 1e-12, (moved, gap_nw)

    host = from_pseudobrain(cfg, sd)                          # the gdn branch (pb_hybrid_layers) is exercised
    for i in cfg["pb_hybrid_layers"]:
        host.layers[i].post_coef.data[1].zero_()
    moved = _gap(ref, host)
    ref_ng = PB27ForCausalLM.from_state_dict(cfg, sd, gdn_on=False)
    gap_ng = _gap(ref_ng, host)
    print(f"gdn off: host moves {moved:.2e}; reference(no gdn) vs host {gap_ng:.2e}")
    assert moved > 1e-3 and gap_ng <= 1e-12, (moved, gap_ng)
