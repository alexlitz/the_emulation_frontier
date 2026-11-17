"""A target checkpoint embedded into a bigger host checkpoint runs in the host's stock transformers
class and reproduces the target's logits: wider hidden, more heads with regrouped KV, a doubled
head_dim (RoPE-interleaved), wider FFN, extra identity layers, and a padded vocab whose extra
logits are pushed to ~-inf. Llama, Qwen2 (q/k/v biases), Qwen3 (q/k-norm), llama -> qwen2. MoE
hosts (Qwen3-MoE, Qwen2-MoE): more / wider experts whose dummies get exactly zero routing mass, top-k
multiples via tied expert copies, dense layers router-steered into MoE layers.

transformers' RMSNorm computes in fp32 even in a float64 model, so "exact" here means the fp32
floor (~1e-7 in log-prob); a wrong eps or norm gain shows up at >= 1e-5.

Run:  python -m pytest tests/test_embed.py -q
"""
EXACT = 1e-6
import pytest
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config,
                          Qwen3ForCausalLM)

from serving.embed import embed_state, plan

CLS = {"llama": (LlamaConfig, LlamaForCausalLM), "qwen2": (Qwen2Config, Qwen2ForCausalLM),
       "qwen3": (Qwen3Config, Qwen3ForCausalLM)}


def make(fam, h, H, kv, hd, ff, L, V, tied=False, theta=10000.0, eps=1e-5, scaling=None, seed=0):
    C, M = CLS[fam]
    cfg = C(hidden_size=h, num_attention_heads=H, num_key_value_heads=kv, head_dim=hd, intermediate_size=ff,
            num_hidden_layers=L, vocab_size=V, tie_word_embeddings=tied, rope_theta=theta, rms_norm_eps=eps,
            max_position_embeddings=128, rope_scaling=scaling, attn_implementation="eager")
    cfg.head_dim = hd                                   # Qwen2Config takes it as an attribute
    torch.manual_seed(seed)
    m = M(cfg).double().eval()
    with torch.no_grad():                               # random norms / biases too, not just ones / zeros
        for n, p in m.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.3 * torch.randn_like(p) if "norm" in n else 0.1 * torch.randn_like(p))
    return cfg, m


def host_for(tcfg, fam, **kw):
    C, M = CLS[fam]
    p = None
    base = dict(hidden_size=48, num_attention_heads=8, num_key_value_heads=4, head_dim=16, intermediate_size=80,
                num_hidden_layers=3, vocab_size=tcfg.vocab_size, tie_word_embeddings=False,
                max_position_embeddings=128, attn_implementation="eager")
    base.update(kw)
    hcfg = C(**base)
    hcfg.head_dim = base["head_dim"]
    p = plan(tcfg, hcfg)
    assert p["ok"], p["why"]
    for k, v in p["config"].items():                    # the host config values plan() asks for
        if v is not None:
            setattr(hcfg, k, v)
    return hcfg, M, p


def run(fam_t, fam_h, tol, t_kw=None, h_kw=None):
    tcfg, tgt = make(fam_t, **{**dict(h=32, H=4, kv=2, hd=8, ff=48, L=2, V=50), **(t_kw or {})})
    hcfg, M, p = host_for(tcfg, fam_h, **(h_kw or {}))
    host = M(hcfg).double().eval()
    sd = embed_state({k: v for k, v in tgt.state_dict().items()}, tcfg, hcfg)
    missing, unexpected = host.load_state_dict(sd, strict=False)
    allowed = ("rotary",) + (("lm_head",) if hcfg.tie_word_embeddings else ())
    assert not unexpected and all(any(t in m for t in allowed) for m in missing), (missing, unexpected)
    ids = torch.randint(0, tcfg.vocab_size, (2, 24))
    with torch.no_grad():
        lt, lh = tgt(ids).logits, host(ids).logits
    V = tcfg.vocab_size
    lpt, lph = lt.log_softmax(-1), lh.log_softmax(-1)
    err = (lpt - lph[..., :V]).abs().max().item()       # compare log-probs: tied hosts shift logits uniformly
    leak = lph[..., V:].exp().sum(-1).max().item() if lh.shape[-1] > V else 0.0
    assert err < tol and leak < 1e-12, (err, leak, p)
    return err


def test_llama_into_every_axis_bigger_is_exact():
    run("llama", "llama", EXACT)


def test_llama3_rope_scaling_and_theta_carry_over():
    sc = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 16}
    run("llama", "llama", EXACT, t_kw=dict(theta=500000.0, scaling=sc))


def test_gqa_regroup_when_target_groups_are_larger():
    run("llama", "llama", EXACT, t_kw=dict(H=4, kv=1))          # g_A = 4 queries/KV into g_B = 2


def test_padded_vocab_puts_no_mass_on_non_tokens():
    run("llama", "llama", EXACT, h_kw=dict(vocab_size=64))
    run("llama", "llama", EXACT, t_kw=dict(tied=True), h_kw=dict(vocab_size=64, tie_word_embeddings=True))


def test_qwen2_biases_and_llama_into_qwen2():
    run("qwen2", "qwen2", EXACT)
    run("llama", "qwen2", EXACT)                                 # host biases set to zero


def test_qwen3_qk_norm():
    run("qwen3", "qwen3", EXACT, h_kw=dict(hidden_size=64))     # hd 8->16 and h 32->64: same ratio, exact
    # ratios differ: q/k-norm eps is off by ~eps relative to mean(q^2), which this tiny random model
    # makes ~1e-2 (real checkpoints: ~1, so ~1e-6)
    run("qwen3", "qwen3", 3e-4)


def test_plan_rejects_what_the_host_code_cannot_do():
    tcfg, _ = make("llama", 32, 4, 2, 8, 48, 2, 50)
    C = LlamaConfig
    bad = {"head_dim not a multiple": dict(head_dim=12, num_attention_heads=8),
           "kv groups": dict(num_key_value_heads=1),
           "vocab pad needs a free hidden dim": dict(hidden_size=32, vocab_size=64),
           "tie": dict(tie_word_embeddings=True)}
    for why, kw in bad.items():
        base = dict(hidden_size=48, num_attention_heads=8, num_key_value_heads=4, head_dim=16,
                    intermediate_size=80, num_hidden_layers=3, vocab_size=50)
        base.update(kw)
        p = plan(tcfg, C(**base))
        assert not p["ok"] and any(why.split()[0] in w for w in p["why"]), (why, p["why"])


def test_tolerance_catches_a_wrong_eps():
    # control: the same embedding with the target's eps left on the host misses by >> EXACT
    tcfg, tgt = make("llama", 32, 4, 2, 8, 48, 2, 50, eps=1e-2)
    hcfg, M, _ = host_for(tcfg, "llama")
    sd = embed_state(tgt.state_dict(), tcfg, hcfg)
    hcfg.rms_norm_eps = tcfg.rms_norm_eps
    host = M(hcfg).double().eval()
    host.load_state_dict(sd, strict=False)
    ids = torch.randint(0, 50, (2, 24))
    with torch.no_grad():
        err = (tgt(ids).logits.log_softmax(-1) - host(ids).logits.log_softmax(-1)).abs().max().item()
    assert err > 100 * EXACT, err


def test_real_qwen25_05b_runs_inside_a_qwen25_3b_shaped_host():
    # real weights: hidden 896 -> 2048, 14 -> 16 heads, head_dim 64 -> 128 (RoPE-interleaved),
    # 24 -> 36 layers, FFN 4864 -> 11008, both tied. The 3.1B stock Qwen2 model reproduces the 0.5B.
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoConfig, AutoModelForCausalLM
    t, h = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"
    if not isinstance(try_to_load_from_cache(t, "model.safetensors"), str):
        pytest.skip("Qwen2.5-0.5B-Instruct weights not cached")
    tcfg, hcfg = AutoConfig.from_pretrained(t), AutoConfig.from_pretrained(h)
    p = plan(tcfg, hcfg)
    assert p["ok"] and p["exact"] == "exact", p
    for k, v in p["config"].items():
        setattr(hcfg, k, v)
    tgt = AutoModelForCausalLM.from_pretrained(t, dtype=torch.float32).eval()
    host = AutoModelForCausalLM.from_config(hcfg, dtype=torch.float32).eval()
    host.load_state_dict(embed_state(tgt.state_dict(), tcfg, hcfg), strict=False)
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(t)("A bigger model can emulate a smaller one by weight-setting alone: "
                                           "pad the hidden width, regroup the heads, and add identity layers.",
                                           return_tensors="pt").input_ids
    with torch.no_grad():
        a, b = tgt(ids).logits.log_softmax(-1), host(ids).logits.log_softmax(-1)
    assert (a - b).abs().max().item() < 1e-3                 # fp32 matmuls of different shapes
    assert (a.argmax(-1) == b.argmax(-1)).all()


def _fp8(model):
    """Per-tensor e4m3 weight quantization of every Linear but lm_head (how fp8 serving treats them)."""
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and n != "lm_head":
                w = m.weight
                s = w.abs().max().clamp_min(1e-30) / 448.0
                m.weight.copy_((w / s).to(torch.float8_e4m3fn).to(w.dtype) * s)
    return model


def test_fp8_serving_error_is_not_amplified_by_the_embedding():
    # Featherless serves fp8. The embedded host (padding, regrouped heads, sqrt(k) q-scale, vocab trick)
    # must lose no more to fp8 than the target itself does.
    import copy
    tcfg, tgt = make("llama", 32, 4, 2, 8, 48, 2, 50)
    hcfg, M, _ = host_for(tcfg, "llama", vocab_size=64)
    host = M(hcfg).double().eval()
    host.load_state_dict(embed_state(tgt.state_dict(), tcfg, hcfg), strict=False)
    ids = torch.randint(0, 50, (4, 32))
    with torch.no_grad():
        ref = tgt(ids).logits.log_softmax(-1)
        t8 = _fp8(copy.deepcopy(tgt))(ids).logits.log_softmax(-1)
        h8 = _fp8(host)(ids).logits.log_softmax(-1)
    kl = lambda lp: (ref.exp() * (ref - lp[..., :50])).sum(-1).mean().item()
    assert h8[..., 50:].exp().sum(-1).max().item() < 1e-9          # suppression survives fp8
    assert kl(h8) < 3 * kl(t8), (kl(h8), kl(t8))


# ---------------------------------------------------------------- MoE (qwen3_moe / qwen2_moe hosts)
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

MOE_CLS = {"qwen3_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM), "qwen2_moe": (Qwen2MoeConfig, Qwen2MoeForCausalLM)}
MOE_T = dict(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8, intermediate_size=48,
             moe_intermediate_size=16, num_experts=8, num_experts_per_tok=2, num_hidden_layers=2, vocab_size=50,
             shared_expert_intermediate_size=24)
MOE_H = dict(hidden_size=48, num_attention_heads=8, num_key_value_heads=4, head_dim=8, intermediate_size=80,
             moe_intermediate_size=24, num_experts=12, num_experts_per_tok=2, num_hidden_layers=3, vocab_size=50,
             shared_expert_intermediate_size=32)
Q2_H = {**MOE_H, "hidden_size": 64}                     # Qwen2-MoE needs heads x head_dim = hidden (8 x 8)


def moe_cfg(fam, **kw):
    C, M = MOE_CLS[fam] if fam in MOE_CLS else CLS[fam]
    kw = {"max_position_embeddings": 128, "attn_implementation": "eager", "tie_word_embeddings": False, **kw}
    if fam != "qwen2_moe":
        kw.pop("shared_expert_intermediate_size", None)
    if fam == "qwen2_moe":
        kw.pop("head_dim", None)
    cfg = C(**kw)
    if "head_dim" in kw:
        cfg.head_dim = kw["head_dim"]
    return cfg, M


def moe_model(fam, seed=0, router_std=1.0, **kw):
    cfg, M = moe_cfg(fam, **kw)
    torch.manual_seed(seed)
    m = M(cfg).double().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.3 * torch.randn_like(p) if "norm" in n else 0.1 * torch.randn_like(p))
            elif n.endswith("mlp.gate.weight"):          # peaked, input-dependent routing
                p.copy_(router_std * torch.randn_like(p))
    return cfg, m


def _fp8_experts(model):
    """fp8 serving: per-tensor e4m3 on every Linear but lm_head and the router."""
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and n != "lm_head" and not n.endswith("mlp.gate"):
                w = m.weight
                s = w.abs().max().clamp_min(1e-30) / 448.0
                m.weight.copy_((w / s).to(torch.float8_e4m3fn).to(w.dtype) * s)
    return model


def compare(tcfg, tgt, fam_h, tol=EXACT, **hkw):
    """Embed tgt into a stock fam_h host, check log-probs and that suppressed experts get exactly 0
    routing probability (fp32 softmax, as the host computes it)."""
    hcfg, M = moe_cfg(fam_h, **hkw)
    p = plan(tcfg, hcfg)
    assert p["ok"], p["why"]
    for k, v in p["config"].items():
        if v is not None:
            setattr(hcfg, k, v)
    host = M(hcfg).double().eval()
    missing, unexpected = host.load_state_dict(embed_state(tgt.state_dict(), tcfg, hcfg), strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    used = {j: (r[1] * tcfg.num_experts if r[0] == "moe" else r[1]) for j, r in p["layers"] if r[0] != "dense"}
    probs = {}
    for j, layer in enumerate(host.model.layers):
        if hasattr(layer.mlp, "experts"):
            layer.mlp.register_forward_hook(
                lambda mod, inp, out, j=j: probs.__setitem__(j, torch.softmax(out[1].float(), -1)))
    ids = torch.randint(0, tcfg.vocab_size, (2, 24))
    with torch.no_grad():
        lt, lh = tgt(ids).logits.log_softmax(-1), host(ids).logits.log_softmax(-1)
    V = tcfg.vocab_size
    err = (lt - lh[..., :V]).abs().max().item()
    leak = lh[..., V:].exp().sum(-1).max().item() if lh.shape[-1] > V else 0.0
    assert err < tol and leak < 1e-12, (err, leak, p["exact"], p["layers"])
    for j, n in used.items():
        assert probs[j][:, n:].sum().item() == 0.0, (j, probs[j][:, n:].max())
    return err, p, host, ids


@pytest.mark.parametrize("norm", [False, True])
def test_qwen3_moe_into_more_and_wider_experts(norm):
    # hidden 32->48, heads 4->8, experts 8->12 (4 dummies get zero mass), expert width 16->24,
    # layers 2->3. eps is tiny so qwen3's q/k-norm eps (which the free dim puts off) is below fp32
    tcfg, tgt = moe_model("qwen3_moe", norm_topk_prob=norm, rms_norm_eps=1e-10, **MOE_T)
    _, p, _, _ = compare(tcfg, tgt, "qwen3_moe", norm_topk_prob=norm, **MOE_H)
    assert p["free"] and p["layers"] == [(0, ("moe", 1)), (1, ("moe", 1))]


def test_qwen3_moe_same_expert_count_needs_no_free_dim():
    # E_B = E_A and h_B/h_A = hd_B/hd_A: no dummies, no free dim, exact at an ordinary eps
    tcfg, tgt = moe_model("qwen3_moe", **MOE_T)
    _, p, _, _ = compare(tcfg, tgt, "qwen3_moe", **{**MOE_H, "hidden_size": 64, "num_attention_heads": 4,
                                                    "num_key_value_heads": 2, "head_dim": 16, "num_experts": 8})
    assert not p["free"] and p["exact"] == "exact"


@pytest.mark.parametrize("norm", [False, True])
def test_top_k_multiple_via_tied_expert_copies(norm):
    # k 2 -> 4: every expert stored twice (tied probabilities); 8 -> 16 (no dummies), 8 -> 20 (4 dummies)
    tcfg, tgt = moe_model("qwen3_moe", norm_topk_prob=norm, rms_norm_eps=1e-10, **MOE_T)
    for E in (16, 20):
        _, p, _, _ = compare(tcfg, tgt, "qwen3_moe", **{**MOE_H, "num_experts": E, "num_experts_per_tok": 4,
                                                        "norm_topk_prob": norm})
        assert p["layers"][0][1] == ("moe", 2) and p["free"] == (E == 20)


def test_dense_layers_in_moe_models_map_in_order():
    # target: layer 0 dense (mlp_only_layers), 1-2 MoE. Host: decoder_sparse_step 2 -> layers 1, 3, 5
    # MoE and 0, 2, 4 dense. Target 0 -> host 0, 1 -> 1, 2 -> 3; host 2, 4, 5 are identities.
    tcfg, tgt = moe_model("qwen3_moe", rms_norm_eps=1e-10, **{**MOE_T, "num_hidden_layers": 3, "mlp_only_layers": [0]})
    _, p, _, _ = compare(tcfg, tgt, "qwen3_moe", **{**MOE_H, "num_hidden_layers": 6, "decoder_sparse_step": 2})
    assert p["layers"] == [(0, ("dense",)), (1, ("moe", 1)), (3, ("moe", 1))]
    # all-MoE host: the dense target layer is router-steered onto 2 experts (48 = 2 x 24)
    _, p, _, _ = compare(tcfg, tgt, "qwen3_moe", **MOE_H)
    assert p["layers"][0] == (0, ("steer", 2))


@pytest.mark.parametrize("norm,k,fm,m", [(False, 2, 24, 2), (True, 2, 24, 2), (False, 8, 12, 4), (True, 1, 48, 1)])
def test_dense_qwen3_into_moe_host_by_router_steering(norm, k, fm, m):
    # dense FFN 48 split over m experts of width fm, each weighted exactly 1/m: m = 1 with k = 1
    # (weight exactly 1), m = 4 of k = 8 (the other 4 picks have zero probability and zero weights)
    tcfg, tgt = make("qwen3", 32, 4, 2, 8, 48, 2, 50, eps=1e-10)
    _, p, _, _ = compare(tcfg, tgt, "qwen3_moe", **{**MOE_H, "num_experts_per_tok": k, "norm_topk_prob": norm,
                                                    "moe_intermediate_size": fm})
    assert all(r == ("steer", m) for _, r in p["layers"]), p["layers"]


def test_steering_over_three_experts_rounds_only_the_routing_weight():
    # 48 = 3 x 16 with k = 3: m = 3 is not a power of 2, so the weights 1/3 round (plan says 'eps');
    # here they round in fp32 only (in bf16 serving: 2^-9 relative)
    tcfg, tgt = make("qwen3", 32, 4, 2, 8, 48, 2, 50, eps=1e-10)
    _, p, _, _ = compare(tcfg, tgt, "qwen3_moe", **{**MOE_H, "num_experts_per_tok": 3, "moe_intermediate_size": 16})
    assert p["layers"][0][1] == ("steer", 3) and p["exact"] == "eps"


@pytest.mark.parametrize("norm", [False, True])
def test_qwen2_moe_shared_expert_and_padded_vocab(norm):
    # shared expert 24 -> 32 with its sigmoid gate copied; q/k/v biases; vocab 50 -> 64 on the same
    # free dim the dummy experts use
    tcfg, tgt = moe_model("qwen2_moe", norm_topk_prob=norm, **MOE_T)
    _, p, _, _ = compare(tcfg, tgt, "qwen2_moe", **{**Q2_H, "norm_topk_prob": norm})
    assert p["exact"] == "exact"
    compare(tcfg, tgt, "qwen2_moe", **{**Q2_H, "norm_topk_prob": norm, "vocab_size": 64})


def test_llama_into_qwen2_moe_by_router_steering():
    tcfg, tgt = make("llama", 32, 4, 2, 8, 48, 2, 50)
    _, p, _, _ = compare(tcfg, tgt, "qwen2_moe", **Q2_H)   # host shared expert and q/k/v biases zero
    assert p["layers"][0][1] == ("steer", 2) and p["exact"] == "exact"


def test_moe_fp8_error_is_not_amplified_and_dummies_stay_at_zero():
    # fp8 serving quantizes the experts (not the router): the embedded host must lose no more to fp8
    # than the target, and dummy experts must still get exactly zero mass
    import copy
    tcfg, tgt = moe_model("qwen2_moe", **MOE_T)
    _, p, host, ids = compare(tcfg, tgt, "qwen2_moe", **Q2_H)
    probs = {}
    for j, layer in enumerate(host.model.layers):
        layer.mlp.register_forward_hook(lambda mod, inp, out, j=j: probs.__setitem__(j, torch.softmax(out[1].float(), -1)))
    with torch.no_grad():
        ref = tgt(ids).logits.log_softmax(-1)
        t8 = _fp8_experts(copy.deepcopy(tgt))(ids).logits.log_softmax(-1)
        h8 = _fp8_experts(host)(ids).logits.log_softmax(-1)
    kl = lambda lp: (ref.exp() * (ref - lp[..., :50])).sum(-1).mean().item()
    assert kl(h8) < 3 * kl(t8), (kl(h8), kl(t8))
    assert all(probs[j][:, tcfg.num_experts:].sum().item() == 0.0 for j, _ in p["layers"])


def test_plan_rejects_what_moe_routing_cannot_do():
    tcfg, _ = moe_cfg("qwen3_moe", **MOE_T)
    bad = {"top-k": dict(num_experts_per_tok=3),                         # 3 is not a multiple of 2
           "experts": dict(num_experts=6),                               # fewer experts
           "experts:": dict(num_experts=12, num_experts_per_tok=4),      # k x2 needs 16 expert slots
           "norm_topk_prob": dict(norm_topk_prob=True),
           "moe_intermediate": dict(moe_intermediate_size=8),
           "dummy": dict(hidden_size=32, num_attention_heads=4, num_key_value_heads=2),   # no free dim
           "layers": dict(num_hidden_layers=3, decoder_sparse_step=2)}  # 1 MoE host layer for 2
    for why, kw in bad.items():
        p = plan(tcfg, moe_cfg("qwen3_moe", **{**MOE_H, **kw})[0])
        assert not p["ok"] and any(why.rstrip(":") in w for w in p["why"]), (why, p["why"])
    assert "MoE target needs an MoE host" in plan(tcfg, moe_cfg("qwen3", **MOE_H)[0])["why"]
    q2, _ = moe_cfg("qwen2_moe", **MOE_T)
    p = plan(q2, moe_cfg("qwen2_moe", **{**Q2_H, "shared_expert_intermediate_size": 8})[0])
    assert any("shared" in w for w in p["why"]), p["why"]
    assert not plan(q2, moe_cfg("qwen3_moe", **MOE_H)[0])["ok"]           # qwen2 attention into q/k-norm
    # a dense FFN wider than k_B experts can hold
    d, _ = moe_cfg("qwen3", **{**MOE_T, "intermediate_size": 100})
    assert any("layers" in w for w in plan(d, moe_cfg("qwen3_moe", **MOE_H)[0])["why"])


def test_fp16_safe_constants():
    # Featherless serves < 5B models in fp16: the vocab-trick gain must not overflow (inf * 0 = NaN)
    tcfg, tgt = make("llama", 32, 4, 2, 8, 48, 2, 50)
    hcfg, M, _ = host_for(tcfg, "llama", vocab_size=64)
    sd = embed_state(tgt.state_dict(), tcfg, hcfg, fp16=True)
    assert all(torch.isfinite(v.half()).all() for v in sd.values())
    host = M(hcfg).double().eval()
    host.load_state_dict(sd, strict=False)
    ids = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        lp = host(ids).logits.log_softmax(-1)
        ref = tgt(ids).logits.log_softmax(-1)
    assert (lp[..., :50] - ref).abs().max().item() < EXACT and lp[..., 50:].max().item() < -100
