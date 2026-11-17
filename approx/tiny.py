"""A tiny random teacher/student pair with the same mismatch structure as
Gemma-4-26B-A4B -> gpt-oss-20b, for fast CPU tests:

                     teacher (Gemma-4-like)            student (gpt-oss-like)
  hidden             60                                 64 (pad)
  depth              10 (s s s s G s s s s G)           8 alternating s/f  (drop 2)
  attn sliding       4 q x 32, kv 2 x 32, rope full     16 q x 8, kv 2 x 8, sinks, one rope theta
  attn global        4 q x 64, kv 1 x 64, k==v, 1/4 rope
  FFN                64 experts x 8, top-4 + dense 24   16 experts x 34, top-2 (4 x 8 + 2 spare)
  act / norms        GeLU-tanh, sandwich, layer_scalar  swiglu-oai (clamped), pre-norm only
  logits             sqrt(h) embed scale, softcap 30    plain

`domains=True` plants two synthetic "domains": tokens [0, V/2) carry direction uA and route to
experts [0, E/2), tokens [V/2, V) carry uB and route to the other half.
"""
from __future__ import annotations

import torch

from approx.arch import SCfg, TCfg

V = 256


def tiny_tcfg():
    lt = ["sliding_attention"] * 4 + ["full_attention"] + ["sliding_attention"] * 4 + ["full_attention"]
    return TCfg(vocab=V, h=60, layer_types=lt, nq=4, hd=32, nkv=2, hd_g=64, nkv_g=1, k_eq_v=True,
                rope={"sliding_attention": {"theta": 10000.0, "type": "default", "partial": 1.0},
                      "full_attention": {"theta": 1e6, "type": "proportional", "partial": 0.25}},
                window=16, E=64, Ie=8, topk=4, Id=24, softcap=30.0, eps=1e-6,
                embed_scale=torch.tensor(60 ** 0.5).to(torch.float32).item(), dtype=torch.float32)


def tiny_scfg():
    return SCfg(vocab=V, h=64, L=8, nq=16, nkv=2, hd=8, E=16, topk=2, I=34, window=16, eps=1e-6,
                rope_theta=10000.0, rope_scaling=None)


def tiny_teacher(seed=0, domains=True, dtype=torch.float32):
    cfg = tiny_tcfg()
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s, std=1.0: torch.randn(*s, generator=g) * std
    h, w = cfg.h, {}
    uA, uB = torch.zeros(h), torch.zeros(h)
    uA[0], uB[1] = 1.0, 1.0
    emb = rn(V, h, std=0.15)
    if domains:
        emb[: V // 2] += 0.6 * uA
        emb[V // 2:] += 0.6 * uB
    w["embed_tokens.weight"] = emb
    w["norm.weight"] = 1 + 0.2 * rn(h)
    for li, lt in enumerate(cfg.layer_types):
        hd, nkv, keqv = cfg.head(li)
        p = f"layers.{li}."
        w[p + "self_attn.q_proj.weight"] = rn(cfg.nq * hd, h, std=h ** -0.5)
        w[p + "self_attn.k_proj.weight"] = rn(nkv * hd, h, std=h ** -0.5)
        if not keqv:   # values with a decaying spectrum (real heads are far from full rank)
            Wv = rn(nkv * hd, h, std=h ** -0.5).view(nkv, hd, h)
            spec = torch.exp(-torch.arange(hd) / (hd / 5))
            w[p + "self_attn.v_proj.weight"] = torch.stack([torch.linalg.qr(rn(hd, hd))[0] @ (spec[:, None] * Wv[i])
                                                            for i in range(nkv)]).reshape(nkv * hd, h) * 2.0
        decay = torch.exp(-torch.arange(hd // 2) / (hd / 6)).repeat(2)     # a few dominant rope pairs
        w[p + "self_attn.q_norm.weight"] = (0.3 + 1.2 * decay) * (1 + 0.2 * rn(hd))   # logit std ~ 3
        w[p + "self_attn.k_norm.weight"] = (0.3 + 1.2 * decay) * (1 + 0.2 * rn(hd))
        w[p + "self_attn.o_proj.weight"] = rn(h, cfg.nq * hd, std=(cfg.nq * hd) ** -0.5)
        for n in ["input_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm_1",
                  "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"]:
            w[p + n + ".weight"] = 0.6 + 0.4 * rn(h).abs()
        for n in ["post_attention_layernorm", "post_feedforward_layernorm"]:   # each layer a perturbation of
            w[p + n + ".weight"] = 0.25 * (0.6 + 0.4 * rn(h).abs())              # the residual stream, as in LMs
        w[p + "layer_scalar"] = torch.tensor([0.5 + 0.5 * torch.rand(1, generator=g).item()])
        w[p + "mlp.gate_proj.weight"] = rn(cfg.Id, h, std=1.5 * h ** -0.5)
        w[p + "mlp.up_proj.weight"] = rn(cfg.Id, h, std=h ** -0.5)
        w[p + "mlp.down_proj.weight"] = rn(h, cfg.Id, std=cfg.Id ** -0.5)
        proj = rn(cfg.E, h, std=2.0)
        if domains:
            proj[: cfg.E // 2] += 12.0 * uA
            proj[cfg.E // 2:] += 12.0 * uB
        w[p + "router.proj.weight"] = proj
        w[p + "router.scale"] = 1 + 0.1 * rn(h)
        w[p + "router.per_expert_scale"] = 0.6 + 0.8 * torch.rand(cfg.E, generator=g)
        w[p + "experts.gate_up_proj"] = rn(cfg.E, 2 * cfg.Ie, h, std=1.5 * h ** -0.5)
        w[p + "experts.down_proj"] = rn(cfg.E, h, cfg.Ie, std=cfg.Ie ** -0.5)
    return cfg, {k: v.to(dtype) for k, v in w.items()}


def _chain(seed=123):
    """sparse Markov chain per domain: each token has 4 successors in its own half of the vocab."""
    g = torch.Generator().manual_seed(seed)
    succ = torch.zeros(V, 4, dtype=torch.long)
    for t in range(V):
        lo = 0 if t < V // 2 else V // 2
        succ[t] = lo + torch.randperm(V // 2, generator=g)[:4]
    return succ


_SUCC = _chain()
_PROB = torch.tensor([0.55, 0.25, 0.12, 0.08])


def tiny_tokens(n, T, domain=None, seed=1):
    """Token batches from a sparse Markov chain (+ copying from 1-7 tokens back w.p. 0.15).
    domain None: each sequence starts in a random half; 'A' / 'B': the chain stays in its half."""
    g = torch.Generator().manual_seed(seed)
    if domain is None:
        start = torch.randint(0, V, (n,), generator=g)
    else:
        lo = 0 if domain == "A" else V // 2
        start = torch.randint(lo, lo + V // 2, (n,), generator=g)
    x = torch.zeros(n, T, dtype=torch.long)
    x[:, 0] = start
    for t in range(1, T):
        nxt = _SUCC[x[:, t - 1], torch.multinomial(_PROB.expand(n, 4), 1, generator=g)[:, 0]]
        sh = torch.randint(1, 8, (n,), generator=g)
        cp = (torch.rand(n, generator=g) < 0.15) & (t - sh >= 0)
        x[:, t] = torch.where(cp, x[torch.arange(n), (t - sh).clamp(min=0)], nxt)
    return x


def tcfg_to_hf(cfg: TCfg):
    """Gemma4TextConfig kwargs (transformers 5.x) for a TCfg."""
    return dict(vocab_size=cfg.vocab, hidden_size=cfg.h, num_hidden_layers=cfg.L, layer_types=cfg.layer_types,
                num_attention_heads=cfg.nq, head_dim=cfg.hd, num_key_value_heads=cfg.nkv,
                global_head_dim=cfg.hd_g, num_global_key_value_heads=cfg.nkv_g, attention_k_eq_v=cfg.k_eq_v,
                rope_parameters={t: ({"rope_type": r["type"], "rope_theta": r["theta"]} |
                                     ({"partial_rotary_factor": r["partial"]} if r["type"] == "proportional" else {}))
                                 for t, r in cfg.rope.items()},
                sliding_window=cfg.window, num_experts=cfg.E, moe_intermediate_size=cfg.Ie,
                top_k_experts=cfg.topk, intermediate_size=cfg.Id, enable_moe_block=True,
                final_logit_softcapping=cfg.softcap, rms_norm_eps=cfg.eps, hidden_activation="gelu_pytorch_tanh",
                tie_word_embeddings=True, hidden_size_per_layer_input=0, num_kv_shared_layers=0,
                attention_bias=False, use_bidirectional_attention=None, pad_token_id=0)


def trained_tiny_teacher(steps=400, seed=0, cache=True):
    """tiny_teacher() trained briefly (next-token NLL on tiny_tokens with both domains) so it is an
    actual LM: its attention learns the copy structure and its experts specialize. Cached under
    ~/.cache/approx_tiny/."""
    import os
    from approx.arch import teacher_forward
    path = os.path.expanduser(f"~/.cache/approx_tiny/teacher_markov_s{seed}_n{steps}.pt")
    cfg, w = tiny_teacher(seed)
    if cache and os.path.exists(path):
        return cfg, torch.load(path)
    params = {k: v.clone().requires_grad_(True) for k, v in w.items() if not k.endswith("layer_scalar")}
    fixed = {k: v for k, v in w.items() if k.endswith("layer_scalar")}
    opt = torch.optim.Adam(params.values(), lr=3e-3)
    g = torch.Generator().manual_seed(seed + 7)
    for it in range(steps):
        dom = ["A", "B", None][it % 3]
        ids = tiny_tokens(16, 48, dom, seed=1000 + it)
        z = teacher_forward({**params, **fixed}, cfg, ids)
        loss = torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = {k: v.detach() for k, v in {**params, **fixed}.items()}
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(w, path)
    return cfg, w
