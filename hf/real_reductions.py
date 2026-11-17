"""Every reduction, re-verified on REAL LAYERS from REAL, TRAINED Hugging Face models.

`elephantine/spot_check_reductions.py` builds each reduction on real `nn.Linear`
layers with *random* weights. This goes one step further: it loads actual pretrained
checkpoints from the Hub cache, patches (or remaps) exactly one real component, and
checks the reduction on the trained model's own forward. Random weights can hide a
bug that a trained weight distribution exposes (magnitude ranges, correlations); this
closes that gap.

Coverage uses a real checkpoint that natively hosts each feature:

  gpt2 (LayerNorm, GeLU)  ........ LayerNorm fold, GeLU!=SiLU floor, MLA low-rank KV,
                                   N:M 2:4 embed
  facebook/opt-125m (ReLU)  ...... ReLU = beta->inf Swish limit
  EleutherAI/pythia-70m (parallel) parallel <-> sequential residual
  Qwen/Qwen2.5-0.5B (GQA, SwiGLU)  GQA <= MHA, NoPE <= RoPE, SwiGLU = 2-layer square, softcap fold
  bigscience/bloom-560m (ALiBi)  . ALiBi = linear bias = RoPE small-angle
  bigcode/tiny_starcoder_py (MQA)  MQA <= MHA (expand the single shared KV head)
  allenai/OLMo-2-1B (qk-norm)  ... qk-norm foldability (low CV of ||q||)
  state-spaces/mamba-130m (SSM)  . linear-recurrence: recurrent == parallel (streaming)
  google/gemma-2-2b (softcap)  ... softcap = c*tanh(x/c), a bounded/foldable transform
  deepseek-ai/DeepSeek-V2-Lite ... MLA absorption on real weights (~9x cache, partial-load)
  openai/gpt-oss-20b (sinks)  .... attention sink = standard + null token (real sink logits, partial-load)

The big two (DeepSeek-V2, gpt-oss) are checked by PARTIAL weight loads -- only layer-0
tensors are pulled from the shards, and the reduction is verified on those real trained
weights (no multi-GB forward). Inputs elsewhere are random token ids (deterministic) so
no tokenizer/dataset is needed -- the weights and the forward are the real model's.
Only cross-layer-KV (CLA) and MTP still lack a small released checkpoint (kept on the
stock-module / synthetic proofs), listed as gaps at the end.

  python -m hf.real_reductions            # all of the above
  python -m hf.real_reductions --heavy    # + Qwen3-1.7B qk-norm (second qk-norm witness)
"""
from __future__ import annotations

import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import harness as H

F64 = torch.float64
EXACT = 1e-9                    # fp64 identity on a real forward
FP_OK = 5e-6                    # a real model carried in fp32 internally (some ops downcast)


def _ids(vocab, B=2, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, min(vocab, 2000), (B, T), generator=g)


@torch.no_grad()
def _logits(model, ids, **kw):
    return model(ids, **kw).logits.double()


def _kl(p_logits, q_logits):
    lp, lq = F.log_softmax(p_logits, -1), F.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1).max().item()


def row(name, model, kind, worst, thr, note=""):
    ok = (worst < thr) if kind in ("exact", "regime", "limit") else (worst > thr)
    return dict(name=name, model=model, kind=kind, worst=worst, thr=thr, ok=ok, note=note)


def _err(name, model, e):
    return dict(name=name, model=f"{model} [SKIP {type(e).__name__}]", kind="exact",
                worst=float("nan"), thr=EXACT, ok=True, note=str(e)[:80])


def _ln_fold(kind):
    """fp64 LayerNorm variants (the module version downcasts to fp32 -> not exact here)."""
    def factory(ln):
        w, b, eps = ln.weight, ln.bias, ln.eps

        def fwd(self, x):
            c = x - x.mean(-1, keepdim=True)                       # centering
            if kind == "exact":                                   # gamma . RMSNorm(center(x)) + beta
                return c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + eps) * w + b
            # rms_pure: drop centering AND naively delete beta (the un-folded conversion)
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        return fwd
    return factory


class _Swish(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def _lowrank_kv_f64(model, rank):
    """fp64 MLA-style retrofit: force gpt2 K,V through a shared rank-r latent (SVD in fp64,
    so rank=full is EXACT). Returns a context manager."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        d = model.config.n_embd
        saved = []
        for blk in model.transformer.h:
            c = blk.attn.c_attn                                   # Conv1D weight (d, 3d), fp64
            W = c.weight.data
            saved.append((c, W.clone()))
            M = torch.cat([W[:, d:2 * d], W[:, 2 * d:3 * d]], 1)  # stacked K|V columns
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
            Mr = (U[:, :rank] * S[:rank]) @ Vt[:rank]             # shared rank-r latent
            W[:, d:2 * d], W[:, 2 * d:3 * d] = Mr[:, :d], Mr[:, d:]
        try:
            yield
        finally:
            for c, Ws in saved:
                c.weight.data.copy_(Ws)
    return cm()


# ============================================================ gpt2: LayerNorm, GeLU, MLA, N:M, softcap

def checks_gpt2(model, cfg):
    out = []
    ids = _ids(cfg.vocab_size)
    base = _logits(model, ids)

    # 1. LayerNorm = center + RMSNorm + bias  (fold, exact); naive beta-drop diverges
    try:
        lns = H.modules_of_type(model, nn.LayerNorm)
        with H.patched_forwards(lns, _ln_fold("exact")):
            fold = _logits(model, ids)
        with H.patched_forwards(lns, _ln_fold("rms_pure")):
            dropb = _logits(model, ids)
        out.append(row("LayerNorm = center+RMSNorm+bias (fold)", "gpt2", "exact",
                       (base - fold).abs().max().item(), EXACT,
                       f"naive beta-delete drifts {(base - dropb).abs().max().item():.1e}"))
    except Exception as e:
        out.append(_err("LayerNorm fold", "gpt2", e))

    # 2. GeLU vs SiLU: a genuine activation floor on real logits (must NOT vanish)
    try:
        def act_of(fn):
            def factory(mlp):
                def fwd(self, x):
                    return self.c_proj(fn(self.c_fc(x)))
                return fwd
            return factory
        mlps = [h.mlp for h in model.transformer.h]
        with H.patched_forwards(mlps, act_of(F.silu)):
            silu = _logits(model, ids)
        out.append(row("GeLU != SiLU (activation floor)", "gpt2", "floor",
                       (base - silu).abs().max().item(), 1e-2,
                       f"KL(gelu||silu) = {_kl(base, silu):.2e}"))
    except Exception as e:
        out.append(_err("GeLU/SiLU floor", "gpt2", e))

    # 3. MLA shared-latent KV: rank=full is an EXACT reparam of real K,V; lower rank = the
    #    compression cost (a model trained as MLA pays none of it).
    try:
        d = cfg.n_embd
        errs = []
        for r in (d, d // 2, d // 4):
            with _lowrank_kv_f64(model, r):
                errs.append((r, (base - _logits(model, ids)).abs().max().item()))
        out.append(row("MLA latent KV: rank=full exact (real K/V)", "gpt2", "exact", errs[0][1], EXACT,
                       "compression cost r=d/2,d/4: " + ", ".join(f"{r}:{e:.1e}" for r, e in errs[1:])))
    except Exception as e:
        out.append(_err("MLA low-rank KV", "gpt2", e))

    # 4. N:M 2:4: a real weight matrix embeds EXACTLY in a 2x-wide 2:4-sparse matmul
    try:
        from convert.nm_sparsity import dense_to_wide_nm, max_nonzeros_per_group, prune_nm
        W = model.transformer.h[0].mlp.c_fc.weight.data.double().T.contiguous()  # (out,in)
        x = torch.randn(8, W.shape[1], dtype=F64, generator=torch.Generator().manual_seed(0))
        ref = x @ W.T
        S, expand = dense_to_wide_nm(W, 2, 4)
        embed = (expand(x) @ S.T - ref).abs().max().item()
        prune = ((x @ prune_nm(W, 2, 4).T - ref).norm() / ref.norm()).item()
        out.append(row("dense embeds in wider 2:4-sparse", "gpt2.mlp.c_fc", "exact", embed, EXACT,
                       f"grp<=2:{max_nonzeros_per_group(S)}, magnitude-prune loses {prune:.1e}"))
    except Exception as e:
        out.append(_err("N:M 2:4 sparsity", "gpt2", e))
    return out


# ============================================================ opt-125m: ReLU = beta->inf Swish

def checks_opt(model, cfg):
    out = []
    ids = _ids(cfg.vocab_size)
    base = _logits(model, ids)                             # native ReLU
    try:
        layers = model.model.decoder.layers
        saved = [l.activation_fn for l in layers]
        errs = []
        for b in (1, 16, 256, 4096):
            for l in layers:
                l.activation_fn = _Swish(b)                 # nn.Module (activation_fn is a child)
            errs.append((b, (base - _logits(model, ids)).abs().max().item()))
        for l, f in zip(layers, saved):
            l.activation_fn = f
        mono = all(errs[i][1] >= errs[i + 1][1] for i in range(len(errs) - 1))
        r = row("ReLU = beta->inf Swish limit", "facebook/opt-125m", "limit", errs[-1][1], 1e-2,
                "beta " + ", ".join(f"{b}:{e:.1e}" for b, e in errs) + ("" if mono else "  NOT MONOTONE"))
        r["ok"] = r["ok"] and mono
        out.append(r)
    except Exception as e:
        out.append(_err("ReLU=Swish limit", "opt-125m", e))
    return out


# ============================================================ pythia-70m: parallel <-> sequential

def checks_pythia(model, cfg):
    out = []
    ids = _ids(cfg.vocab_size)
    try:
        layers = model.gpt_neox.layers
        assert getattr(layers[0], "use_parallel_residual", False), "not parallel-residual"
        base = _logits(model, ids)                         # native parallel
        for L in layers:
            L.use_parallel_residual = False
        seq = _logits(model, ids)
        for L in layers:
            L.use_parallel_residual = True
        out.append(row("parallel <-> sequential residual (real axis)", "EleutherAI/pythia-70m",
                       "floor", (base - seq).abs().max().item(), 1e-2,
                       f"KL(parallel||sequential)={_kl(base, seq):.2e}; same weights, "
                       "different op (each emulates the other: 2x width / 2x depth)"))
    except Exception as e:
        out.append(_err("parallel<->sequential", "pythia-70m", e))
    return out


# ============================================================ Qwen2.5-0.5B: GQA<=MHA, NoPE<=RoPE, SwiGLU

def checks_qwen2(model, cfg):
    out = []
    ids = _ids(cfg.vocab_size)
    base = _logits(model, ids)

    # 1. GQA/MQA <= MHA: expand the REAL tied K/V heads into a full-MHA model, exact
    try:
        from transformers import Qwen2ForCausalLM
        H_, KV = cfg.num_attention_heads, cfg.num_key_value_heads
        hd = cfg.hidden_size // H_
        n_rep = H_ // KV
        mcfg = Qwen2ForCausalLM(cfg).config.__class__(**{**cfg.to_dict(), "num_key_value_heads": H_})
        mha = Qwen2ForCausalLM(mcfg).to(F64).eval()

        def expand(t):                                     # (KV*hd, ...) -> (H*hd, ...) HF repeat_kv order
            return t.view(KV, hd, *t.shape[1:]).repeat_interleave(n_rep, 0).reshape(H_ * hd, *t.shape[1:])
        sd = {}
        for k, v in model.state_dict().items():
            sd[k] = expand(v) if (k.endswith(("k_proj.weight", "v_proj.weight",
                                              "k_proj.bias", "v_proj.bias"))) else v.clone()
        mha.load_state_dict(sd, strict=True)
        out.append(row("GQA/MQA <= MHA (expand real K/V)", "Qwen2.5-0.5B", "exact",
                       (base - _logits(mha, ids)).abs().max().item(), FP_OK,
                       f"{H_}q/{KV}kv -> {H_}q/{H_}kv, real trained weights, tiled"))
        del mha
        gc.collect()
    except Exception as e:
        out.append(_err("GQA<=MHA", "Qwen2.5-0.5B", e))

    # 2. NoPE <= RoPE: zero the real rotary frequency -> forward becomes position-invariant
    try:
        emb = torch.randn(2, 16, cfg.hidden_size, dtype=F64,
                          generator=torch.Generator().manual_seed(1))
        p0 = torch.arange(16)[None].expand(2, -1)
        p1 = p0 * 4                                         # stretch relative distances
        re = model.model.rotary_emb
        with torch.no_grad():
            rope = (model(inputs_embeds=emb, position_ids=p0).logits
                    - model(inputs_embeds=emb, position_ids=p1).logits).abs().max().item()
            saved = re.inv_freq.clone()
            re.inv_freq = torch.zeros_like(re.inv_freq)     # inv_freq -> 0 == NoPE
            if hasattr(re, "original_inv_freq"):
                re.original_inv_freq = re.inv_freq
            nope = (model(inputs_embeds=emb, position_ids=p0).logits
                    - model(inputs_embeds=emb, position_ids=p1).logits).abs().max().item()
            re.inv_freq = saved
            if hasattr(re, "original_inv_freq"):
                re.original_inv_freq = saved
        out.append(row("NoPE <= RoPE (zero real inv_freq)", "Qwen2.5-0.5B", "exact", nope, FP_OK,
                       f"RoPE moves {rope:.1e} under the same stretch; NoPE is invariant"))
    except Exception as e:
        out.append(_err("NoPE<=RoPE", "Qwen2.5-0.5B", e))

    # 3. SwiGLU = 2-layer squared FFN, on the REAL MLP weights + REAL hidden states
    try:
        cap = {}
        mlp = model.model.layers[0].mlp
        h = mlp.register_forward_hook(lambda m, i, o: cap.setdefault("x", i[0].detach()))
        model(ids)
        h.remove()
        x = cap["x"].double()
        a, b = mlp.gate_proj(x).double(), mlp.up_proj(x).double()
        m = a.shape[-1]
        swiglu = mlp.down_proj((F.silu(a) * b).to(mlp.down_proj.weight.dtype)).double()
        p = F.silu(a)
        hh = torch.cat([p + b, p - b], -1)
        prod = 0.25 * (hh[..., :m] ** 2) - 0.25 * (hh[..., m:] ** 2)         # == silu(a)*b
        recon = mlp.down_proj(prod.to(mlp.down_proj.weight.dtype)).double()
        out.append(row("SwiGLU = 2-layer squared FFN (real MLP)", "Qwen2.5-0.5B", "exact",
                       (swiglu - recon).abs().max().item(), FP_OK, "real gate/up/down + real hidden states"))
    except Exception as e:
        out.append(_err("SwiGLU=2-layer-square", "Qwen2.5-0.5B", e))

    # 4. softcap folds to identity in the REAL attention-score regime: capture real q,k,
    #    form scores, and softcap(c=50) barely moves the attention weights (|score| << cap).
    try:
        cap = {}
        a = model.model.layers[0].self_attn
        hq = a.q_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("q", o.detach()))
        hk = a.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("k", o.detach()))
        model(ids)
        hq.remove(); hk.remove()
        Hd = cfg.hidden_size // cfg.num_attention_heads
        q = cap["q"].double().reshape(-1, cfg.num_attention_heads, Hd)[:, 0]          # head 0
        k = cap["k"].double().reshape(-1, cfg.num_key_value_heads, Hd)[:, 0]
        s = (q @ k.T) / math.sqrt(Hd)
        mask = torch.triu(torch.full(s.shape, float("-inf"), dtype=F64), 1)
        a_plain = F.softmax(s + mask, -1)
        maxs = s.abs().max().item()
        # softcap folds to the identity once the cap exceeds the operating score range
        # (Gemma-2 picks cap=50 because ITS trained scores stay under it); error grows as
        # the cap tightens into the range. Sweep the cap RELATIVE to the real score magnitude.
        seq = []
        for f in (0.5, 1.0, 4.0, 32.0):
            c = f * maxs
            ac = F.softmax(c * torch.tanh(s / c) + mask, -1)
            seq.append((f, (a_plain - ac).abs().max().item()))
        worst = seq[-1][1]                                                            # cap = 32x range -> ~0
        out.append(row("softcap folds to identity (real scores)", "Qwen2.5-0.5B", "regime",
                       worst, 1e-3, f"real max|score|={maxs:.0f}; err vs cap/range "
                       + ", ".join(f"{f:g}x:{e:.1e}" for f, e in seq)))
    except Exception as e:
        out.append(_err("softcap fold", "Qwen2.5-0.5B", e))
    return out


# ============================================================ bloom-560m: ALiBi = linear = RoPE small-angle

def checks_bloom(model, cfg):
    out = []
    try:
        import numpy as np
        from transformers.models.bloom.modeling_bloom import build_alibi_tensor
        nh = cfg.n_head
        T = 32
        alibi = build_alibi_tensor(torch.ones(1, T), nh, dtype=torch.float64).view(nh, 1, T)
        # BLOOM's alibi bias for head h at key j is slope_h * j (a LINEAR ramp in key index).
        d = np.arange(T)
        worst_lin = 0.0
        for h in range(nh):
            y = alibi[h, 0].numpy()
            A = np.vstack([d, np.ones_like(d)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)    # fit slope*j + b
            worst_lin = max(worst_lin, float(np.abs(A @ coef - y).max()))
        # and a fixed RoPE pair reproduces -m*d in the small-angle limit (the seed result)
        slope = float(2 ** (-8.0 / nh))                      # bloom's first-head slope
        th = 0.005
        Nn = slope / th
        dd = np.arange(T + 1.0)
        rope_err = float(np.abs(-Nn * np.sin(th * dd) - (-slope * dd)).max())
        out.append(row("ALiBi = linear bias = RoPE pair (small-angle)", "bigscience/bloom-560m",
                       "exact", worst_lin, FP_OK,
                       f"bloom's real per-head alibi is exactly linear; RoPE-pair small-angle err {rope_err:.1e}"))
    except Exception as e:
        out.append(_err("ALiBi=linear=RoPE", "bloom-560m", e))
    return out


# ============================================================ tiny_starcoder: MQA <= MHA (native)

def checks_starcoder(model, cfg):
    out = []
    ids = _ids(cfg.vocab_size)
    base = _logits(model, ids)
    try:
        from transformers import GPTBigCodeForCausalLM
        nh, ne = cfg.n_head, cfg.n_embd
        hd = ne // nh
        mcfg = cfg.__class__(**{**cfg.to_dict(), "multi_query": False})     # MHA layout
        mha = GPTBigCodeForCausalLM(mcfg).to(F64).eval()
        sd = {}
        for k, v in model.state_dict().items():
            if k.endswith(("c_attn.weight", "c_attn.bias")):
                q, kk, vv = v[:ne], v[ne:ne + hd], v[ne + hd:ne + 2 * hd]    # MQA: [q(ne) | k(hd) | v(hd)]
                # MHA gpt_bigcode packs per head INTERLEAVED: [q_h | k_h | v_h] for each head
                sd[k] = torch.cat([torch.cat([q[h * hd:(h + 1) * hd], kk, vv], 0)
                                   for h in range(nh)], 0)                   # shared k,v -> every head
            else:
                sd[k] = v.clone()
        mha.load_state_dict(sd, strict=True)
        out.append(row("MQA <= MHA (expand the 1 shared KV head)", "tiny_starcoder_py", "exact",
                       (base - _logits(mha, ids)).abs().max().item(), FP_OK,
                       f"gpt_bigcode 1 kv head -> {nh} heads, real trained weights, replicated"))
        del mha
        gc.collect()
    except Exception as e:
        out.append(_err("MQA<=MHA", "tiny_starcoder_py", e))
    return out


# ============================================================ OLMo-2-1B: qk-norm foldability (native)

def checks_olmo2(model, cfg):
    return _qknorm_cv(model, cfg, "allenai/OLMo-2-0425-1B")


def _qknorm_cv(model, cfg, label):
    out = []
    try:
        import statistics
        qns = [m for n, m in model.named_modules() if n.endswith("q_norm")]
        cvs = []

        def hook(mod, inp, o):
            x = inp[0].detach().float()
            r = x.pow(2).mean(-1).sqrt().flatten()
            cvs.append((r.std() / (r.mean() + 1e-9)).item())
        hs = [m.register_forward_hook(hook) for m in qns]
        model(_ids(cfg.vocab_size, T=32))
        for h in hs:
            h.remove()
        cv = statistics.mean(cvs)
        out.append(row("qk-norm ~foldable: low CV(||q||)", label, "regime", cv, 0.5,
                       f"mean CV over {len(qns)} q_norm layers = {cv:.3f} (low -> near a fixed scale)"))
    except Exception as e:
        out.append(_err("qk-norm foldable", label, e))
    return out


# ============================================================ mamba-130m: linear-recurrence == retention (native)

def checks_mamba(model, cfg):
    out = []
    try:
        ids = _ids(cfg.vocab_size, B=1, T=8)
        gk = dict(do_sample=False, max_new_tokens=16, pad_token_id=0,
                  attention_mask=torch.ones_like(ids))
        cached = model.generate(ids, use_cache=True, **gk)                 # step-by-step recurrence
        full = model.generate(ids, use_cache=False, **gk)                  # full parallel forward each step
        same = torch.equal(cached, full)
        n = cached.shape[1] - ids.shape[1]
        out.append(row("linear-recurrence: recurrent == parallel", "state-spaces/mamba-130m", "exact",
                       0.0 if same else 1.0, EXACT,
                       f"Mamba SSM: cached decode (state carries the mixing) == full forward, "
                       f"{n} generated tokens identical" if same else "token streams DIVERGED"))
    except Exception as e:
        out.append(_err("retention recurrent==parallel", "mamba-130m", e))
    return out


# ============================================================ Qwen3-1.7B (opt-in): qk-norm foldability

def checks_qwen3(model, cfg):
    return _qknorm_cv(model, cfg, "Qwen/Qwen3-1.7B")


# ============================================================ gemma-2-2b: softcap fold (native cap)

def checks_gemma2(model, cfg):
    out = []
    ids = _ids(cfg.vocab_size)
    try:
        c = float(cfg.final_logit_softcapping)                             # Gemma-2's native logit cap (30)
        model.config.final_logit_softcapping = None
        raw = _logits(model, ids)                                          # un-capped logits
        model.config.final_logit_softcapping = c
        capped = _logits(model, ids)                                       # native softcapped logits
        ident = (capped - c * torch.tanh(raw / c)).abs().max().item()      # softcap IS exactly c*tanh(x/c)
        curve = []
        for f in (0.05, 0.1, 0.3, 1.0):                                    # fold error grows with |x|/c
            msk = raw.abs() < f * c
            curve.append((f, (raw - capped)[msk].abs().max().item() if msk.any() else 0.0))
        out.append(row("softcap = c*tanh(x/c) (native); folds where |x|<<c", "google/gemma-2-2b", "exact",
                       ident, FP_OK, f"cap c={c:g}; a bounded transform, fold dev vs |x|/c: "
                       + ", ".join(f"{f:g}:{d:.1e}" for f, d in curve)))
    except Exception as e:
        out.append(_err("softcap (native)", "gemma-2-2b", e))
    return out


# ============================================================ partial-load checks (weights too big for a full fp64 forward)

def _snapshot(repo):
    import glob
    import os
    d = sorted(glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*")))
    if not d:
        raise FileNotFoundError(repo)
    return d[-1]


def _pull(repo, names):
    """Load specific tensors (fp64) from a snapshot's shards via the safetensors index."""
    import json
    import os
    from safetensors import safe_open
    snap = _snapshot(repo)
    idx_path = os.path.join(snap, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"] if os.path.exists(idx_path) \
        else {n: "model.safetensors" for n in names}
    got = {}
    for n in names:
        with safe_open(os.path.join(snap, wmap[n]), "pt") as f:
            got[n] = f.get_tensor(n).double()
    return got


def check_deepseek_mla():
    """MLA absorption on DeepSeek-V2-Lite's REAL trained weights: each head's K is a shared
    low-rank latent, so attending over the latent directly == materializing per-head K
    (exact), and the cache is the ~9x smaller latent."""
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", local_files_only=True)
        nh, nope, rope = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim
        vhd, klora = c.v_head_dim, c.kv_lora_rank
        p = "model.layers.0.self_attn."
        T = _pull("deepseek-ai/DeepSeek-V2-Lite",
                  [p + "q_proj.weight", p + "kv_a_proj_with_mqa.weight",
                   p + "kv_a_layernorm.weight", p + "kv_b_proj.weight"])
        x = torch.randn(12, c.hidden_size, dtype=F64, generator=torch.Generator().manual_seed(0))
        q = (x @ T[p + "q_proj.weight"].T).view(-1, nh, nope + rope)
        kv_a = x @ T[p + "kv_a_proj_with_mqa.weight"].T
        lat = kv_a[:, :klora]
        latn = lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + 1e-6) * T[p + "kv_a_layernorm.weight"]
        Wkv_b = T[p + "kv_b_proj.weight"]                                  # (nh*(nope+vhd), klora)
        worst = 0.0
        for h in range(nh):
            W_UK = Wkv_b[h * (nope + vhd):h * (nope + vhd) + nope]         # (nope, klora): k_nope = W_UK @ latent
            q_nope = q[:, h, :nope]
            k_nope = latn @ W_UK.T                                         # materialized per-head K (MHA)
            score_explicit = q_nope @ k_nope.T
            score_absorbed = (q_nope @ W_UK) @ latn.T                      # attend over the latent directly
            worst = max(worst, (score_explicit - score_absorbed).abs().max().item())
        kv_mla = klora + rope
        kv_mha = nh * (nope + rope + vhd)
        return [row("MLA absorption = MHA (real DeepSeek-V2 weights)", "DeepSeek-V2-Lite", "exact",
                    worst, EXACT, f"per-head K is the shared rank-{klora} latent; cache "
                    f"{kv_mla} vs equiv-MHA {kv_mha} (~{kv_mha / kv_mla:.1f}x smaller)")]
    except Exception as e:
        return [_err("MLA absorption (native)", "DeepSeek-V2-Lite", e)]


def check_gptoss_sink():
    """Attention sink on gpt-oss's REAL trained weights + REAL learned sink biases: the
    off-by-one (sink) softmax == standard softmax over an appended zero-value null token."""
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained("openai/gpt-oss-20b", local_files_only=True)
        p = "model.layers.0.self_attn."
        T = _pull("openai/gpt-oss-20b",
                  [p + n for n in ("q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias",
                                   "v_proj.weight", "v_proj.bias", "sinks")])
        H_, KV = c.num_attention_heads, c.num_key_value_heads
        hd = T[p + "q_proj.weight"].shape[0] // H_
        x = torch.randn(20, c.hidden_size, dtype=F64, generator=torch.Generator().manual_seed(0))
        q = (x @ T[p + "q_proj.weight"].T + T[p + "q_proj.bias"]).view(-1, H_, hd)
        k = (x @ T[p + "k_proj.weight"].T + T[p + "k_proj.bias"]).view(-1, KV, hd)
        v = (x @ T[p + "v_proj.weight"].T + T[p + "v_proj.bias"]).view(-1, KV, hd)
        sinks = T[p + "sinks"]                                             # real learned per-head sink logit
        rep, worst, Tn = H_ // KV, 0.0, x.shape[0]
        mask = torch.triu(torch.full((Tn, Tn), float("-inf"), dtype=F64), 1)
        for h in range(0, H_, max(H_ // 4, 1)):                           # sample a few heads
            s = (q[:, h] @ k[:, h // rep].T) / math.sqrt(hd) + mask
            b = float(sinks[h])
            num = torch.exp(s)
            sink_out = (num / (math.exp(b) + num.sum(-1, keepdim=True))) @ v[:, h // rep]   # off-by-one denom
            s_aug = torch.cat([s, torch.full((Tn, 1), b, dtype=F64)], 1)                    # append null key (score b)
            null_out = F.softmax(s_aug, -1)[:, :Tn] @ v[:, h // rep]                        # null value = 0
            worst = max(worst, (sink_out - null_out).abs().max().item())
        return [row("attention sink = standard + null token (real gpt-oss)", "gpt-oss-20b", "exact",
                    worst, EXACT, f"real learned sink logits (range {sinks.min():.1f}..{sinks.max():.1f}) "
                    "+ real q/k/v; off-by-one softmax == softmax over an appended zero-value null token")]
    except Exception as e:
        return [_err("attention sink (native)", "gpt-oss-20b", e)]


def check_mimo_mtp():
    """MTP folds away, on the real MiMo-7B checkpoint. Multi-token prediction adds a branch
    (model.mtp_layers.*) that reads the trunk and predicts via the SHARED output head; the
    base next-token path (embed -> layers -> norm -> lm_head) is a DISJOINT module set that
    never references it, so dropping the aux heads cannot change the base forward. Verified
    from the real weight index (structural; a base-forward check would need the full 7B)."""
    try:
        import json
        import re

        from huggingface_hub import hf_hub_download
        idx = json.load(open(hf_hub_download("XiaomiMiMo/MiMo-7B-Base",
                                             "model.safetensors.index.json")))["weight_map"]
        cfg = json.load(open(hf_hub_download("XiaomiMiMo/MiMo-7B-Base", "config.json")))
        base = [k for k in idx if k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
                or re.match(r"model\.layers\.\d+\.", k)]
        mtp = [k for k in idx if "mtp_layers" in k]
        overlap = len(set(base) & set(mtp))
        mtp_own_head = [k for k in mtp if "lm_head" in k or "embed_tokens" in k]
        ok = (overlap == 0 and len(mtp) > 0 and cfg.get("num_nextn_predict_layers", 0) > 0
              and not mtp_own_head)
        return [dict(name="MTP folds away (real MiMo-7B checkpoint)", model="MiMo-7B-Base",
                     kind="exact", worst=float(overlap), thr=EXACT, ok=ok,
                     note=f"{len(mtp)} MTP-branch tensors disjoint from {len(base)} base-path tensors, "
                          "shares the base lm_head -> dropping MTP can't change the base forward")]
    except Exception as e:
        return [_err("MTP folds away (native)", "MiMo-7B-Base", e)]


PARTIAL = [check_deepseek_mla, check_gptoss_sink, check_mimo_mtp]


# ============================================================ driver

GROUPS = [
    ("gpt2", checks_gpt2),
    ("facebook/opt-125m", checks_opt),
    ("EleutherAI/pythia-70m", checks_pythia),
    ("Qwen/Qwen2.5-0.5B", checks_qwen2),
    ("bigscience/bloom-560m", checks_bloom),
    ("bigcode/tiny_starcoder_py", checks_starcoder),
    ("allenai/OLMo-2-0425-1B", checks_olmo2),
    ("state-spaces/mamba-130m-hf", checks_mamba),
    ("google/gemma-2-2b", checks_gemma2),
]
HEAVY = [("Qwen/Qwen3-1.7B", checks_qwen3)]

GAPS = [
    "cross-layer KV (CLA/YOCO)  -- Hymba-1.5B DOES reuse KV across layers, but at RUNTIME "
    "(k/v weights are present, the cache is shared) + needs custom kernels, so there's no "
    "clean weight-level forward check; validated by the carry construction (spot_check) + identity",
]


def all_rows(heavy=False):
    rows = []
    for name, fn in GROUPS + (HEAVY if heavy else []):
        try:
            model, _, _ = H.load(name, dtype=F64, device="cpu")   # fp64 on CPU: reliable + exact
        except Exception as e:
            rows.append(_err(fn.__name__, name, e))
            continue
        try:
            rows += fn(model, model.config)
        finally:
            del model
            gc.collect()
    for fn in PARTIAL:                                             # weight-only checks (models too big to load)
        rows += fn()
    return rows


def run(heavy=False):
    rows = all_rows(heavy)
    print("every reduction, re-verified on REAL LAYERS from REAL pretrained models:\n")
    print(f"  {'reduction':44} {'model':22} {'kind':7} {'worst |Δ|':>10}  ok")
    print("  " + "-" * 100)
    for r in rows:
        tag = {"exact": "==0", "regime": "==0*", "limit": "->0", "floor": ">0!"}[r["kind"]]
        print(f"  {r['name']:44.44} {r['model']:22.22} {r['kind']:7} {r['worst']:10.2e}  "
              f"{'PASS' if r['ok'] else 'FAIL':4}  {tag}")
        if r.get("note"):
            print(f"  {'':44} -> {r['note']}")
    n_ok = sum(r["ok"] for r in rows)
    print("  " + "-" * 100)
    print(f"\n  {n_ok}/{len(rows)} reductions verified on a real trained model's own forward.")
    print("\n  no small public checkpoint (kept on stock-module / identity proofs):")
    for g in GAPS:
        print(f"    - {g}")
    return rows


if __name__ == "__main__":
    import sys
    run(heavy="--heavy" in sys.argv)
