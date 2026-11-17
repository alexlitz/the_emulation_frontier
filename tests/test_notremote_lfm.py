"""The LFM family on the universal host WITHOUT running remote code.

LFM2-MoE is the stock transformers 5.18 class (subprocess). LFM3 and LFM3-RD exist only as remote code
(LiquidAI/LFM3-1.7B-Base, cybertruck32489/LFM3-1.7B-RD): their references are labelled transcriptions read as
text (tests/refs/lfm3.py, tests/refs/lfm3_rd.py), compared with from_lfm at fp64.

Run:  pytest tests/test_notremote_lfm.py -q
"""
import torch

from tests.test_eleph_gaps_stream import LIFT
from tests.test_universal_elephantine import _ids, _rand_, _tf5_run

F64 = torch.float64


def _rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


def test_same_host_carries_lfm2_moe_notremote():
    """LFM2-MoE (the stock transformers 5.18 Lfm2MoeForCausalLM): LFM2 short convs + attention and a sigmoid top-k MoE
    with expert-bias selection, (sum + 1e-6) renormalisation and a routed scale. fp64 with the fp32 casts lifted;
    the expert bias switched off moves the logits."""
    gap, scale = _tf5_run(LIFT + '''
from transformers.models.lfm2_moe import modeling_lfm2_moe as lm, configuration_lfm2_moe as lc
from elephantine.universal import from_lfm
cfg = lc.Lfm2MoeConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, conv_L_cache=3, num_dense_layers=1,
    num_experts=6, num_experts_per_tok=2, use_expert_bias=True, norm_topk_prob=True, routed_scaling_factor=1.5,
    layer_types=["conv", "full_attention", "conv", "full_attention"], experts_implementation="eager")
torch.manual_seed(0)
t = rand_(lm.Lfm2MoeForCausalLM(cfg).to(F64).eval())
with torch.no_grad():
    for l in t.model.layers:
        if hasattr(l.feed_forward, "expert_bias"):
            l.feed_forward.expert_bias.copy_(torch.randn(6, dtype=F64) * 0.3)
x = ids(97, T=12)
with torch.no_grad():
    with lift():
        ref = t(x, use_cache=False).logits
    assert ref.dtype == F64
    ours = from_lfm(t)(x)
    h = from_lfm(t)
    for l in h.layers:                                    # selection without the expert bias
        l.ffn.sel_bias.zero_()
    assert rel(ref, h(x)) > 1e-4, rel(ref, h(x))
report(ref, ours)
''')
    assert gap < 1e-12 * max(scale, 1), (gap, scale)


_LFM3_KW = dict(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=6, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, conv_L_cache=7, conv_bias=False, sliding_window=4,
                use_sandwich_norm=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                layer_types=["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"],
                rope_parameters={"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1e6, "rope_type": "default"},
                                 "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})


def _norms_off_one_(t, s=0.3):
    """Push the RMSNorm weights away from 1 (rand_ adds only 0.05), so the sandwich norms bite."""
    for n, p in t.named_parameters():
        if "norm" in n:
            p.add_(torch.randn_like(p) * s)
    return t


def _lfm3_branches(t, x, ref, from_lfm, hd):
    """rel. change of the host's logits with each LFM3-specific branch switched off."""
    moved = {}
    h = from_lfm(t)
    for l in h.layers:
        l.attn.window.fill_(float("inf"))
    moved["window"] = _rel(ref, h(x))
    h = from_lfm(t)
    for l in h.layers:
        l.attn.gate_coef.copy_(torch.tensor([1.0, 0.0]))
    moved["gate"] = _rel(ref, h(x))
    h = from_lfm(t)
    for l in h.layers:
        l.post_coef.copy_(torch.tensor([1.0, 0.0]).expand_as(l.post_coef))
    moved["sandwich"] = _rel(ref, h(x))
    rot = t.model.rotary_emb
    whole = 1.0 / (1e6 ** (torch.arange(0, hd, 2, dtype=F64) / hd))  # the full progression, no zero tail
    assert (rot.full_attention_inv_freq[hd // 4:] == 0).all()
    assert (rot.full_attention_inv_freq[: hd // 4] == whole[: hd // 4]).all()
    types = t.config.layer_types
    h = from_lfm(t)
    for i, l in enumerate(h.layers):
        if types[i] == "full_attention":
            l.attn.inv_freq[: hd // 2] = whole                   # every pair rotates
    moved["rope_tail"] = _rel(ref, h(x))
    h = from_lfm(t)
    for i, l in enumerate(h.layers):
        if types[i] == "sliding_attention":
            l.attn.inv_freq[: hd // 2] = whole                   # one theta for both layer types
    moved["rope_theta"] = _rel(ref, h(x))
    return moved


@torch.no_grad()
def test_same_host_carries_lfm3_notremote():
    """LFM3 vs from_lfm against the REFERENCE tests/refs/lfm3.py (a transcription of LiquidAI/LFM3-1.7B-Base's
    modeling_lfm3.py, read as text): width-7 gated short convs, per-head qk-norm, a per-head sigmoid output gate,
    full layers with zero-tail partial RoPE (theta 1e6) and sliding layers (window 4 < T = 12, theta 1e4), sandwich
    norms, tied embeddings. fp64 throughout (the reference lifts the remote code's fp32 casts): 1e-12 relative.
    Each LFM3-specific branch switched off moves the logits: the sliding window, the attention gate, the sandwich
    norms, the partial-RoPE zero tail, the per-type RoPE theta."""
    from tests.refs.lfm3 import Lfm3Config, Lfm3ForCausalLM
    from elephantine.universal import from_lfm
    torch.manual_seed(0)
    t = _norms_off_one_(_rand_(Lfm3ForCausalLM(Lfm3Config(**_LFM3_KW)).to(F64).eval()))
    assert t.lm_head.weight is t.model.embed_tokens.weight
    x = _ids(97, T=12)
    ref = t(x)
    d = _rel(ref, from_lfm(t)(x))
    assert d < 1e-12, d
    moved = _lfm3_branches(t, x, ref, from_lfm, hd=8)
    assert min(moved.values()) > 1e-4, moved


@torch.no_grad()
def test_same_host_carries_lfm3_rd_unrolled_notremote():
    """LFM3-RD (looped, recurrent-depth LFM3) vs from_lfm against the REFERENCE tests/refs/lfm3_rd.py (a transcription
    of cybertruck32489/LFM3-1.7B-RD's modeling_lfm3_rd.py, read as text) at its shipped manual setting with
    recurrent_loops == num_motifs: prelude, 2 motifs whose attention / conv projections are one shared core + per-layer
    LoRA (grouped for the conv in_proj; merged as at inference), the rescale p + step_scale (x - p) around the core, coda.
    fp64 throughout: 1e-12 relative. Switched off, each moves the logits: the rescale (res_mix), the per-layer LoRA
    (the host built from the shared core alone), and the LFM3 branches (window, head gate, sandwich norms, RoPE)."""
    import copy
    from tests.refs.lfm3_rd import Lfm3RDConfig, Lfm3RDForCausalLM
    from elephantine.universal import from_lfm
    kw = dict(_LFM3_KW, num_hidden_layers=12, num_prelude_layers=2, num_motifs=2, motif_size=4, num_coda_layers=2,
              recurrent_loops=2, rank_attn=4, rank_conv=4, rank_kv=4, conv_lora_groups=3, reasoning_mode="manual",
              layer_types=["conv", "conv"] + ["full_attention", "conv", "sliding_attention", "conv"] * 2
                          + ["full_attention", "conv"])
    torch.manual_seed(0)
    t = Lfm3RDForCausalLM(Lfm3RDConfig(**kw)).to(F64).eval()
    for n, p in t.named_parameters():                            # the shared core / LoRA factors start as torch.empty
        if n.startswith("model.core."):
            p.copy_(torch.randn_like(p) * 0.2)
    _norms_off_one_(_rand_(t))
    t.model.rd_controller.step_scales.fill_(0.7)                 # make the rescale matter
    ph = t.model.core.phases
    assert ph[0].blocks[0].self_attn.q_proj.shared_weight is ph[1].blocks[0].self_attn.q_proj.shared_weight
    assert ph[0].blocks[1].conv.in_proj.shared_weight is ph[1].blocks[1].conv.in_proj.shared_weight
    x = _ids(97, T=12)
    ref = t(x)
    d = _rel(ref, from_lfm(t)(x))
    assert d < 1e-12, d
    moved = _lfm3_branches(t, x, ref, from_lfm, hd=8)
    h = from_lfm(t)
    for l in h.layers:
        l.res_mix.zero_()
    moved["rescale"] = _rel(ref, h(x))
    t0 = copy.deepcopy(t)
    for n, p in t0.named_parameters():
        if n.endswith((".B", ".A")) and n.startswith("model.core.phases."):
            p.zero_()
    moved["lora"] = _rel(ref, from_lfm(t0)(x))
    assert min(moved.values()) > 1e-4, moved
