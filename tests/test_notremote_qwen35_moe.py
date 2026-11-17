"""Qwen3.5-MoE and MemT-1.0 on the universal host, without running remote code.

test_same_host_carries_qwen35_moe_and_memt (tests/test_universal_elephantine.py) ran MemT's own modelling
file; it is remote-gated. Here Qwen3.5-MoE is the stock transformers 5.18 class and MemT is the labelled
transcription tests/refs/memt.py (built on the same stock modules the remote file subclasses).

Run:  pytest tests/test_notremote_qwen35_moe.py -q
"""
from tests.test_eleph_gaps_stream import LIFT
from tests.test_universal_elephantine import TF5_Q35MOE, _tf5_run


def test_same_host_carries_qwen35_moe_and_memt_notremote():
    """Qwen3.5-MoE (stock 5.18 Qwen3_5MoeForCausalLM, 3-D expert tensors) and MemT-1.0-Nano (the reference
    transcription: Qwen3.5-MoE whose attention layers add a session-memory read) vs from_qwen35, fp64 with the
    fp32 casts lifted. MemT's plain LM forward (no memory state) is Qwen3.5-MoE: the host carries it and not
    the memory. Shown: the memory read is real (a nonzero state moves the reference's logits), and the
    Qwen3.5 attention output gate and MoE routing, switched off on the host, each move the logits."""
    gap, scale = _tf5_run(LIFT + TF5_Q35MOE + '''
import json
from elephantine.universal import from_qwen35
from tests.refs import memt as R
gaps = []
x = ids(97, T=12)
# 1) stock Qwen3.5-MoE
cfg = qc.Qwen3_5MoeTextConfig(**qkw, experts_implementation="eager", attn_implementation="eager")
torch.manual_seed(0)
t = rand_(cpu_paths(qm.Qwen3_5MoeForCausalLM(cfg).to(F64).eval()))
with torch.no_grad():
    with lift():
        ref = t(x, use_cache=False).logits
    assert ref.dtype == F64
    u = from_qwen35(t)
    gaps.append(rel(ref, u(x)))
    for l in u.layers:                                    # top-2 of 6 experts: routing really selects
        l.ffn.top_k = 6
    assert rel(ref, u(x)) > 1e-4
# 2) MemT-1.0-Nano: its memt block (config.json "memt") on a tiny Qwen3.5-MoE text config
memt = {"memory_update_mode": "token", "gdn_version": "stable_gdn2", "memory_use_short_conv": False,
        "memory_head_mode": "mha", "memory_read_scale": "k_inv_sqrt", "linear_attention_state_policy": "reset_per_chunk",
        "chunk_tokens": 2048, "tail_policy": "commit", "output_gate_type": "silu", "stable_gdn2_sqrt_erase_gate": True,
        "backbone_gdn_override": False, "num_memory_heads": 4, "num_kv_heads": 4, "output_gate_bias": False,
        "output_gate_softcap_enabled": True, "output_gate_softcap_limit": 10.0}
text = qc.Qwen3_5MoeTextConfig(**qkw, rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                               "partial_rotary_factor": 0.25, "mrope_interleaved": True, "mrope_section": [1, 1, 0]},
                               attn_output_gate=True, experts_implementation="eager", attn_implementation="eager")
torch.manual_seed(0)
t = rand_(cpu_paths(R.ForCausalLM(text, memt).to(F64).eval()))
assert all(hasattr(l.self_attn, "memt") for l in t.model.language_model.layers if l.layer_type == "full_attention")
with torch.no_grad():
    with lift():
        ref = t(x)
        B, H, hd = x.shape[0], text.num_attention_heads, text.head_dim
        state = {i: torch.randn(B, H, hd, hd, dtype=F64) for i, l in enumerate(t.model.language_model.layers)
                 if l.layer_type == "full_attention"}
        assert rel(ref, t(x, memory_state=state)) > 1e-4   # the session memory read is a real branch
    assert ref.dtype == F64
    u = from_qwen35(t)
    gaps.append(rel(ref, u(x)))
    h = from_qwen35(t)
    for l in h.layers:                                    # the sigmoid output gate on the attention layers
        l.attn.gate_coef.copy_(torch.tensor([1.0, 0.0]))
    assert rel(ref, h(x)) > 1e-4
print("GAPS", gaps)
print("GAP", max(gaps), "SCALE", 1.0)
''')
    assert gap < 1e-12, gap
