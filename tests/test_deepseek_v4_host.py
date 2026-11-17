"""DeepSeek-V4 on the universal host: from_deepseek_v4 vs the STOCK transformers 5.18 DeepseekV4ForCausalLM
(a subprocess: tests/test_universal_elephantine.py's _tf5_run), a small synthetic config with random weights in
fp64. The config makes every V4 feature bind: sliding layers (window 6 < T), CSA layers (overlapping pools of 2,
the lightning indexer's top-3 of up to 10 pools) and an HCA layer (pools of 4), hash-routed and learned-routed
MoE layers (sqrt(softplus) scores, a selection bias), a SwiGLU limit small enough to clamp, mHC streams with the
final HyperHead, learned sinks, a grouped low-rank output projection.

Run:  pytest tests/test_deepseek_v4_host.py -q
"""
import pytest

from tests.test_universal_elephantine import _tf5_run

DSV4 = '''
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mv, configuration_deepseek_v4 as cv
from elephantine.universal import from_deepseek_v4
cfg = cv.DeepseekV4Config(vocab_size=97, hidden_size=32, moe_intermediate_size=24, num_hidden_layers=5,
    num_attention_heads=4, num_key_value_heads=1, head_dim=16, q_lora_rank=12, partial_rotary_factor=0.5,
    num_experts_per_tok=2, n_routed_experts=6, n_shared_experts=1, scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5, rope_theta=10000.0, compress_rope_theta=160000.0,
    layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention",
                 "compressed_sparse_attention", "sliding_attention"],
    compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
    mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe", "moe"], hc_mult=3, hc_sinkhorn_iters=6,
    sliding_window=6, o_groups=2, o_lora_rank=10, index_n_heads=2, index_head_dim=12, index_topk=3,
    swiglu_limit=SWIGLU, rms_norm_eps=1e-6, attn_implementation="eager", experts_implementation="eager")
t = mv.DeepseekV4ForCausalLM(cfg).to(F64).eval()
rand_(t, seed=SEED)
torch.manual_seed(SEED + 1)
with torch.no_grad():
    for l in t.model.layers:
        g = l.mlp.gate
        if hasattr(g, "tid2eid"):                       # a frozen token -> experts table (distinct picks)
            g.tid2eid.copy_(torch.stack([torch.randperm(6)[:2] for _ in range(97)]))
        else:
            g.e_score_correction_bias.copy_(torch.randn(6, dtype=F64) * 0.05)
if LIFT:                                                 # lift the reference's fp32 casts to fp64
    import types as _t
    torch.Tensor.float = lambda self, *a, **k: self.to(F64)          # every .float() -> fp64 (int ids too)
    px = _t.ModuleType("torch_f64")
    px.__dict__.update({k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
    px.float32 = F64
    mv.torch = px
x = ids(97, T=TLEN, seed=SEED)
with torch.no_grad():
    ref = t(x).logits
    u = from_deepseek_v4(t)
    ours = u(x)
    if CHECK:                                            # the features under test actually bind
        a = [l.attn for l in u.layers]
        assert a[1].cmp.cfg[1] == 1 and torch.isfinite(a[1].cmp.cfg[3]) and a[2].cmp.cfg[1] == 0
        assert u.layers[0].ffn.hash_tab.numel() and not u.layers[2].ffn.hash_tab.numel()
        g = t.model.layers[3].mlp.shared_experts
        h = torch.randn(4, 32, dtype=F64)
        assert (g.gate_proj(h) > SWIGLU).any()           # the clamp bites
report(ref, ours)
'''


@pytest.mark.parametrize("seed,T,lift", [(0, 20, False), (1, 23, True)])
def test_same_host_carries_deepseek_v4(seed, T, lift):
    """from_deepseek_v4 vs the stock DeepseekV4ForCausalLM (transformers 5.18), as shipped and with its fp32 casts
    lifted."""
    body = (DSV4.replace("SWIGLU", "0.1").replace("SEED", str(seed)).replace("TLEN", str(T))
            .replace("CHECK", "True").replace("LIFT", str(lift)))
    gap, scale = _tf5_run(body)
    # as shipped the stock class rounds the rotary (cos / sin and the rotated slice), the norms, the mHC weights,
    # the compressor's pool softmax and the indexer scores to fp32 (~2e-7 rel); with those casts lifted to fp64
    # (torch.Tensor.float and the module's torch.float32 -> fp64) the host agrees to fp64 rounding
    assert gap < (1e-12 if lift else 1e-5) * max(scale, 1), (gap, scale)
