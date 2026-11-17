"""RMSNorm target on a LayerNorm host at hidden d + 1 (not 2 d): the mean-zero isometry.

The relation charged a LayerNorm host 2x hidden for an RMSNorm target (NORM_WIDTH_FACTOR: the mirror [h; -h], Lean
rmsNorm_sub_layerNorm_mirror). The mirror is one way to make the host's centring a no-op; any linear isometry U
(d_h x d, U^T U = I) whose columns are orthogonal to the ones vector (1^T U = 0) does the same, and one exists as soon
as d_h >= d + 1 (the mean-zero subspace of R^{d_h} has dimension d_h - 1). Every residual writer of the host writes
U (the target's write) -- embedding rows, o_proj, down_proj -- so the host residual is z = U h at every layer. Then
mean(z) = 0 and |z| = |h|, so

    LayerNorm_{d_h}(z; gamma 1, beta 0, eps_h) = U h / sqrt(|h|^2 / d_h + eps_h) = sqrt(d_h / d) U RMSNorm_d(h; 1, eps_t)

with eps_h = eps_t d / d_h (config; or scale the residual by s, s^2 = d_h eps_h / (d eps_t), the eps of the host
unchanged). Each reader folds the target norm gain and U^T: W_h = sqrt(d / d_h) W_t diag(gamma_t) U^T (q / k / v,
gate / up, lm_head). Exact. Lean: NormSubspace.rmsNorm_sub_layerNorm_subspace (proofs/lean).

What it does NOT cover (the relation keeps the 2x mirror there): a norm whose gained output is written INTO the residual
(sandwich post-norms, post-only, post-LN, nGPT, an embedding norm) -- the gain would have to commute with U, which needs
every row of U on one target channel, i.e. the mirror -- and a TIED host (its lm_head is the embedding, so the final
norm's gain cannot fold into it).

U here: the Helmert-type column set on the first d + 1 host channels, U[i, j] = delta_ij - a (i < d), U[d, j] = -b,
a = (1 - 1 / sqrt(d + 1)) / d, b = 1 / sqrt(d + 1); the other host channels are 0.

Measured on stock StableLmForCausalLM hosts (LayerNorm, SwiGLU, RoPE with partial_rotary_factor 1, head_dim =
hidden / heads, so the host has H_t + 1 heads: d_h = d + head_dim, the extra head and channels zero) written from
Llama-family RMSNorm targets, at the act/act_swap_agreement.py standard (top-1 >= 99%, >= 90% identical 32-token
greedy continuations of 24 prompts):

  python -m hf.norm_subspace --check                       # fp64: functional d + 1 check and a random Llama -> StableLM
  python -m hf.norm_subspace HuggingFaceTB/SmolLM2-135M amd/AMD-Llama-135m --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "norm_subspace_agreement.jsonl")


def subspace_basis(d, dh, dtype=torch.float64):
    """U (dh x d): orthonormal columns, each orthogonal to the ones vector (needs dh >= d + 1)."""
    assert dh >= d + 1
    a = (1 - 1 / math.sqrt(d + 1)) / d
    b = 1 / math.sqrt(d + 1)
    U = torch.zeros(dh, d, dtype=torch.float64)
    U[:d] = torch.eye(d, dtype=torch.float64) - a
    U[d] = -b
    return U.to(dtype)


def check_functional(d=24, T=10, seed=0):
    """fp64: LayerNorm_{d+1}(U h) = sqrt((d+1)/d) U RMSNorm_d(h) with eps_h = eps_t d / (d + 1)."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(T, d, generator=g, dtype=torch.float64) * 3
    eps_t = 1e-2
    U = subspace_basis(d, d + 1)
    orth = (U.T @ U - torch.eye(d, dtype=torch.float64)).abs().max().item()
    mean0 = (U.sum(0)).abs().max().item()
    rms = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps_t)
    ln = torch.nn.functional.layer_norm(h @ U.T, (d + 1,), eps=eps_t * d / (d + 1))
    err = (ln - math.sqrt((d + 1) / d) * rms @ U.T).abs().max().item()
    return dict(orth=orth, mean0=mean0, err=err)


def target_dims(cfg):
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return dict(V=cfg.vocab_size, D=cfg.hidden_size, I=cfg.intermediate_size, L=cfg.num_hidden_layers,
                H=cfg.num_attention_heads, KV=cfg.num_key_value_heads, hd=hd, eps=cfg.rms_norm_eps,
                theta=getattr(cfg, "rope_theta", 10000.0), maxpos=cfg.max_position_embeddings,
                act=cfg.hidden_act, bias=bool(getattr(cfg, "attention_bias", cfg.model_type == "qwen2")))


def host_config(t, extra_heads=1):
    """A StableLM config with H_t + extra_heads heads of the target's head_dim (MHA: each target KV head copied)."""
    from transformers import StableLmConfig
    H = t["H"] + extra_heads
    dh = H * t["hd"]
    return StableLmConfig(vocab_size=t["V"], hidden_size=dh, intermediate_size=t["I"], num_hidden_layers=t["L"],
                          num_attention_heads=H, num_key_value_heads=H, hidden_act=t["act"],
                          max_position_embeddings=t["maxpos"], rope_theta=t["theta"], partial_rotary_factor=1.0,
                          layer_norm_eps=t["eps"] * t["D"] / dh, use_qkv_bias=t["bias"], qk_layernorm=False,
                          use_parallel_residual=False, hidden_dropout=0.0, attention_dropout=0.0,
                          tie_word_embeddings=False)


@torch.no_grad()
def write_host(target, host):
    """Write the StableLM host's weights from the Llama-family target (fp64 arithmetic, cast on copy)."""
    t = target_dims(target.config)
    D, hd, H, KV = t["D"], t["hd"], t["H"], t["KV"]
    dh = host.config.hidden_size
    Hh = host.config.num_attention_heads
    g_t = H // KV
    U = subspace_basis(D, dh)
    c = math.sqrt(D / dh)
    f = lambda x: x.detach().to(torch.float64).cpu()
    put = lambda p, x: p.copy_(x.to(p.dtype).to(p.device))
    tm, hm = target.model, host.model
    put(hm.embed_tokens.weight, f(tm.embed_tokens.weight) @ U.T)            # rows U e_v
    lm = target.lm_head.weight if target.lm_head is not None else tm.embed_tokens.weight
    put(host.lm_head.weight, c * (f(lm) * f(tm.norm.weight)[None, :]) @ U.T)
    put(hm.norm.weight, torch.ones(dh, dtype=torch.float64))
    put(hm.norm.bias, torch.zeros(dh, dtype=torch.float64))
    for lt, lh in zip(tm.layers, hm.layers):
        g1, g2 = f(lt.input_layernorm.weight), f(lt.post_attention_layernorm.weight)
        read = lambda W, gam: c * (f(W) * gam[None, :]) @ U.T              # W_t diag(gamma) U^T sqrt(d / d_h)
        at, ah = lt.self_attn, lh.self_attn
        q, k, v = read(at.q_proj.weight, g1), read(at.k_proj.weight, g1), read(at.v_proj.weight, g1)
        Q = torch.zeros(Hh * hd, dh, dtype=torch.float64)
        K, Vv = torch.zeros_like(Q), torch.zeros_like(Q)
        O = torch.zeros(dh, Hh * hd, dtype=torch.float64)
        o = U @ f(at.o_proj.weight)                                         # writes U (target write)
        qb = torch.zeros(Hh * hd, dtype=torch.float64)
        kb, vb = torch.zeros_like(qb), torch.zeros_like(qb)
        for j in range(H):
            kv = j // g_t
            sl, sk = slice(j * hd, (j + 1) * hd), slice(kv * hd, (kv + 1) * hd)
            Q[sl], K[sl], Vv[sl], O[:, sl] = q[sl], k[sk], v[sk], o[:, sl]
            if t["bias"]:
                qb[sl], kb[sl], vb[sl] = f(at.q_proj.bias)[sl], f(at.k_proj.bias)[sk], f(at.v_proj.bias)[sk]
        put(ah.q_proj.weight, Q)
        put(ah.k_proj.weight, K)
        put(ah.v_proj.weight, Vv)
        put(ah.o_proj.weight, O)
        if t["bias"]:
            put(ah.q_proj.bias, qb)
            put(ah.k_proj.bias, kb)
            put(ah.v_proj.bias, vb)
        mt, mh = lt.mlp, lh.mlp
        put(mh.gate_proj.weight, read(mt.gate_proj.weight, g2))
        put(mh.up_proj.weight, read(mt.up_proj.weight, g2))
        put(mh.down_proj.weight, U @ f(mt.down_proj.weight))
        for ln in (lh.input_layernorm, lh.post_attention_layernorm):
            put(ln.weight, torch.ones(dh, dtype=torch.float64))
            put(ln.bias, torch.zeros(dh, dtype=torch.float64))
    return host


def build_host(target, dtype, device):
    from transformers import StableLmForCausalLM
    cfg = host_config(target_dims(target.config))
    cfg._attn_implementation = "eager"
    host = StableLmForCausalLM(cfg).to(dtype).eval()
    write_host(target, host)
    return host.to(device)


def check_random_llama(seed=0):
    """fp64: a random small Llama (GQA, tied) and its StableLM host agree to rounding. (LlamaRMSNorm upcasts to fp32
    internally, a 1e-7 rounding floor in an fp64 run; the reference runs it in fp64.)"""
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama import modeling_llama as ml

    def fwd64(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    old = ml.LlamaRMSNorm.forward
    ml.LlamaRMSNorm.forward = fwd64
    try:
        return _check_random_llama(seed)
    finally:
        ml.LlamaRMSNorm.forward = old


def _check_random_llama(seed):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=97, hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=6,
                      num_key_value_heads=2, max_position_embeddings=64, rms_norm_eps=1e-5, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    t = LlamaForCausalLM(cfg).to(torch.float64).eval()
    for n, p in t.named_parameters():                       # non-trivial norm gains
        if "norm" in n:
            p.data = 1 + 0.3 * torch.randn_like(p)
    h = build_host(t, torch.float64, "cpu")
    ids = torch.randint(0, 97, (2, 20))
    with torch.no_grad():
        a, b = t(ids).logits, h(ids).logits
    return dict(max_abs=(a - b).abs().max().item(), scale=a.abs().max().item(), host_hidden=h.config.hidden_size,
                target_hidden=48)


def measure(repo, device="cuda:0", n_prompts=24):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf.eps_lever_agreement import reference, score
    tok = AutoTokenizer.from_pretrained(repo)
    tgt = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager").eval()
    tdev = tgt.to(device)
    ref = reference(tdev, tok, device, n_prompts)
    host = build_host(tdev, torch.float32, device)
    r = score(host, ref, device)
    t = target_dims(tgt.config)
    r.update(model=repo, host="StableLmForCausalLM", target_hidden=t["D"], host_hidden=host.config.hidden_size,
             mirror_hidden=2 * t["D"], dtype="float32", n_prompts=n_prompts,
             payable=bool(r["top1"] >= 0.99 and r["ex32"] >= 0.90))
    del host, tgt, tdev
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.check:
        print("functional d + 1:", check_functional())
        print("random Llama -> StableLM (fp64):", check_random_llama())
    for repo in a.repos:
        r = measure(repo, a.device)
        print(json.dumps(r), flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
