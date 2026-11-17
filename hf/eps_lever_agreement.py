"""The eps lever on real models: a host switches OFF a norm the target lacks, by weights alone.

An RMS norm with epsilon e computes  g x / sqrt(ms(x) + e).  Feed it x' = 2^-a x and it computes
g 2^-a x / sqrt(2^-2a ms(x) + e) = (g 2^-a / sqrt(e)) x / sqrt(1 + rho),  rho = 2^-2a ms(x) / e.
Once rho is below the precision of the sum (rho < 2^-25 in fp32: fl(ms + e) = e) the norm IS the
linear map x -> (g 2^-a / sqrt(e)) x, and the gain g = 2^a sqrt(e) (times whatever scale the target
needs there) makes it the identity. The weights that WRITE the norm's input choose a, so a host
switches off any norm the target lacks -- per norm, independently of its shared config epsilon:

  * qk-norm (Qwen3, Gemma-3 / -4, OLMo-2 style, per head before RoPE): q_proj, k_proj x 2^-a,
    q_norm / k_norm gain 2^a sqrt(e) (x the softmax-scale ratio s_t / s_h);
  * sandwich post-norms (Gemma-2 / -3 / -4 post_attention_layernorm / post_feedforward_layernorm):
    o_proj, down_proj x 2^-a, post-norm gain 2^a sqrt(e);
  * weight-free v-norm (Gemma-4): v_proj x 2^-a, o_proj x 2^a sqrt(e) (attention is linear in V);
  * MLA latent norm (kv_a_layernorm, DeepSeek-V2 / V3): the latent rows of kv_a_proj_with_mqa x 2^-a,
    kv_a_layernorm gain 2^a sqrt(e).

The residual is NOT touched (S_res = 1, host eps = target eps): every norm the target HAS keeps its exact
input. frontier-levers' synthetic check (verify/levers/eps_lever.py) also scaled the residual by S = 2^20
with host eps = e_t S^2; that is the same construction with a different split and is run here as a variant
(--s-log2 20). The only requirements are eps > 0 on the switched-off norm and exponent range: the scaled
activations must stay normal (fp32 / bf16: 2^-126; fp16 has 2^-14 and fails).

Measured with the act/act_swap_agreement.py standard: the ORIGINAL target's own greedy 64-token
continuations of the same 24 prompts; payable = top-1 >= 99% AND >= 90% of 32-token continuations
identical. Also: max |delta log p| teacher-forced, and rho_max (the largest ms(x) / eps any switched-off
norm saw, measured by hooks on the host).

Pairs (stock transformers modules; the host's weights are all written from the target's):
  qwen3<-smollm2      HuggingFaceTB/SmolLM2-135M on a Qwen3ForCausalLM            (qk-norm off)
  qwen3<-llama3.2     meta-llama/Llama-3.2-1B on a Qwen3ForCausalLM               (qk-norm off)
  gemma2<-gemma1      unsloth/gemma-2b (= google/gemma-2b) on a Gemma2ForCausalLM (post-norms off)
  gemma3<-gemma1      gemma-2b on a Gemma3ForCausalLM                       (qk-norm + post-norms off)
  gemma4<-gemma1      gemma-2b on a Gemma4ForCausalLM (transformers 5.18)  (qk + v + post-norms off)
  mla<-gemma1         gemma-2b on a DeepseekV2ForCausalLM (MLA)                   (latent norm off)
Host config values that are not the lever are set to the target's (activation, theta, head_dim, eps;
softcaps None, query_pre_attn_scalar = head_dim, sliding_window >= context). The MLA host needs an MQA
target (its rotary key is one head shared by every query head): gemma-2b is MQA, with RoPE.

  python -m hf.eps_lever_agreement qwen3<-smollm2 --device cuda:0
  python -m hf.eps_lever_agreement gemma4<-gemma1 --device cuda:0 --host-device cuda:1    # runs under tf5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from act.act_swap_agreement import N_NEW, PROMPTS, cont_logp, greedy

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ATTN = "eager"
PRIMARY = -30            # the recorded construction: a chosen so that rho_max <= 2^-30 on the target's own text
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eps_lever_agreement.jsonl")
PAIRS = {
    "qwen3<-smollm2": ("HuggingFaceTB/SmolLM2-135M", "qwen3"),
    "qwen3<-llama3.2": ("meta-llama/Llama-3.2-1B", "qwen3"),
    "gemma2<-gemma1": ("unsloth/gemma-2b", "gemma2"),
    "gemma3<-gemma1": ("unsloth/gemma-2b", "gemma3"),
    "gemma4<-gemma1": ("unsloth/gemma-2b", "gemma4"),
    "mla<-gemma1": ("unsloth/gemma-2b", "mla"),
}
# which host norms the lever switches off, per host kind (attribute-name suffixes)
OFF = {
    "qwen3": ("q_norm", "k_norm"),
    "gemma2": ("post_attention_layernorm", "post_feedforward_layernorm"),
    "gemma3": ("q_norm", "k_norm", "post_attention_layernorm", "post_feedforward_layernorm"),
    "gemma4": ("q_norm", "k_norm", "v_norm", "post_attention_layernorm", "post_feedforward_layernorm"),
    "mla": ("kv_a_layernorm",),
}
ONE_PLUS_W = {"gemma", "gemma2", "gemma3"}      # norm gain is (1 + w)
EMBED_SCALE = {"gemma", "gemma2", "gemma3", "gemma4"}


def strict(r):
    return r["top1"] >= 0.99 and r["ex32"] >= 0.90


# --- the target, read off a stock module ----------------------------------------------------------
def tfamily(cfg):
    return "gemma" if cfg.model_type == "gemma" else "llama"


def tdims(cfg):
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return dict(V=cfg.vocab_size, D=cfg.hidden_size, I=cfg.intermediate_size, L=cfg.num_hidden_layers,
                H=cfg.num_attention_heads, KV=cfg.num_key_value_heads, hd=hd, eps=cfg.rms_norm_eps,
                theta=getattr(cfg, "rope_theta", 10000.0), rope_scaling=getattr(cfg, "rope_scaling", None),
                act=getattr(cfg, "hidden_activation", None) or cfg.hidden_act,
                tie=bool(getattr(cfg, "tie_word_embeddings", True)), maxpos=cfg.max_position_embeddings)


def host_config(kind, t, E):
    """A stock host config with the target's shapes; E = the host's (shared) norm epsilon."""
    common = dict(vocab_size=t["V"], hidden_size=t["D"], intermediate_size=t["I"], num_hidden_layers=t["L"],
                  num_attention_heads=t["H"], max_position_embeddings=t["maxpos"], rms_norm_eps=E,
                  attention_bias=False, bos_token_id=None, eos_token_id=None, pad_token_id=None,
                  attn_implementation=ATTN)
    win = max(t["maxpos"], 8192)
    if kind == "qwen3":
        from transformers import Qwen3Config
        return Qwen3Config(num_key_value_heads=t["KV"], head_dim=t["hd"], hidden_act=t["act"],
                           rope_theta=t["theta"], rope_scaling=t["rope_scaling"], tie_word_embeddings=t["tie"],
                           use_sliding_window=False, **common)
    if kind == "gemma2":
        from transformers import Gemma2Config
        return Gemma2Config(num_key_value_heads=t["KV"], head_dim=t["hd"], hidden_activation=t["act"],
                            rope_theta=t["theta"], query_pre_attn_scalar=t["hd"], attn_logit_softcapping=None,
                            final_logit_softcapping=None, sliding_window=win, tie_word_embeddings=True, **common)
    if kind == "gemma3":
        from transformers import Gemma3TextConfig
        return Gemma3TextConfig(num_key_value_heads=t["KV"], head_dim=t["hd"], hidden_activation=t["act"],
                                rope_theta=t["theta"], rope_local_base_freq=t["theta"], rope_scaling=None,
                                query_pre_attn_scalar=t["hd"], attn_logit_softcapping=None,
                                final_logit_softcapping=None, sliding_window=win, tie_word_embeddings=True,
                                **common)
    if kind == "gemma4":
        from transformers import Gemma4TextConfig
        rope = {"rope_type": "default", "rope_theta": t["theta"]}
        common.pop("attn_implementation")
        c = Gemma4TextConfig(num_key_value_heads=t["KV"], head_dim=t["hd"], global_head_dim=t["hd"],
                             hidden_activation=t["act"], rope_parameters={"sliding_attention": dict(rope),
                                                                          "full_attention": dict(rope)},
                             layer_types=["sliding_attention"] * (t["L"] - 1) + ["full_attention"],
                             sliding_window=win, final_logit_softcapping=None, hidden_size_per_layer_input=0,
                             attention_k_eq_v=False, num_kv_shared_layers=0, enable_moe_block=False,
                             use_double_wide_mlp=False, tie_word_embeddings=True, **common)
        c._attn_implementation = ATTN
        return c
    if kind == "mla":
        from transformers import DeepseekV2Config
        assert t["KV"] == 1, "an MLA host shares ONE rotary key head: the target must be MQA"
        return DeepseekV2Config(num_key_value_heads=t["H"], hidden_act=t["act"], rope_theta=t["theta"],
                                rope_scaling=None, tie_word_embeddings=False, q_lora_rank=None,
                                kv_lora_rank=t["hd"], qk_nope_head_dim=0, qk_rope_head_dim=t["hd"],
                                v_head_dim=t["hd"], first_k_dense_replace=t["L"], **common)
    raise ValueError(kind)


def _causal_lm(kind, cfg):
    import transformers as tr
    return {"qwen3": tr.Qwen3ForCausalLM, "gemma2": tr.Gemma2ForCausalLM, "gemma3": tr.Gemma3ForCausalLM,
            "gemma4": getattr(tr, "Gemma4ForCausalLM", None), "mla": tr.DeepseekV2ForCausalLM}[kind](cfg)


def _build(kind, cfg, dtype, device):
    """the host, initialised directly in `dtype` on `device` (an fp32 init + cast would need both copies)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            return _causal_lm(kind, cfg).eval()
    finally:
        torch.set_default_dtype(old)


def _interleave_perm(d):
    """half-split rotary layout (i, i + d/2) -> interleaved (2i, 2i + 1): new row 2i <- old i."""
    p = torch.empty(d, dtype=torch.long)
    p[0::2] = torch.arange(d // 2)
    p[1::2] = torch.arange(d // 2) + d // 2
    return p


def host_weights(kind, tfam, t, sd, a, s_log2=0):
    """{host parameter name: tensor} (float64, CPU) for the construction with input scale-down 2^-a and
    residual scale S = 2^s_log2 (host eps = e_t S^2)."""
    S = 2.0 ** s_log2
    E = t["eps"] * S * S
    down = 2.0 ** -a
    g_off = 2.0 ** a * math.sqrt(E)                 # gain that undoes 2^-a / sqrt(E)
    hd, H = t["hd"], t["H"]

    def T(n):
        return sd[n].double()

    def tgain(n):                                    # the target norm's effective gain
        return T(n) + 1.0 if tfam == "gemma" else T(n)

    def hgain(g):                                    # host parameter for an effective gain g
        return g - 1.0 if kind in ONE_PLUS_W else g

    emb_t = T("model.embed_tokens.weight")
    sc_t = math.sqrt(t["D"]) if tfam in EMBED_SCALE else 1.0
    sc_h = math.sqrt(t["D"]) if kind in EMBED_SCALE else 1.0
    out = {}
    c = S * sc_t / sc_h                               # host embedding = c * target embedding
    out["model.embed_tokens.weight"] = emb_t * c
    lm_t = emb_t if t["tie"] else T("lm_head.weight")
    tied_h = kind in ("gemma2", "gemma3", "gemma4") or (kind == "qwen3" and t["tie"])
    if tied_h:
        assert t["tie"]
        out["model.norm.weight"] = hgain(tgain("model.norm.weight") / c)
        out["lm_head.weight"] = out["model.embed_tokens.weight"]
    else:
        out["model.norm.weight"] = hgain(tgain("model.norm.weight"))
        out["lm_head.weight"] = lm_t
    post = kind in ("gemma2", "gemma3", "gemma4")
    for i in range(t["L"]):
        p, q = f"model.layers.{i}.", f"model.layers.{i}."
        out[q + "input_layernorm.weight"] = hgain(tgain(p + "input_layernorm.weight"))
        pre_mlp = "pre_feedforward_layernorm" if post else "post_attention_layernorm"
        out[q + pre_mlp + ".weight"] = hgain(tgain(p + "post_attention_layernorm.weight"))
        for m in ("gate_proj", "up_proj"):
            out[q + f"mlp.{m}.weight"] = T(p + f"mlp.{m}.weight")
        Wq, Wk, Wv, Wo = (T(p + f"self_attn.{m}_proj.weight") for m in "qkvo")
        Wd = T(p + "mlp.down_proj.weight")
        s_t = hd ** -0.5
        s_h = 1.0 if kind == "gemma4" else hd ** -0.5   # the host softmax scale (query_pre_attn_scalar = hd)
        if post:
            out[q + "post_attention_layernorm.weight"] = hgain(torch.full((t["D"],), S * g_off, dtype=torch.float64))
            out[q + "post_feedforward_layernorm.weight"] = hgain(torch.full((t["D"],), S * g_off,
                                                                           dtype=torch.float64))
            o_sc, d_sc = down, down
        else:
            o_sc, d_sc = S, S
        out[q + "mlp.down_proj.weight"] = Wd * d_sc
        if kind == "mla":
            perm = _interleave_perm(hd)
            Wq_h = Wq.view(H, hd, -1)[:, perm].reshape(H * hd, -1) * (s_t / s_h)
            out[q + "self_attn.q_proj.weight"] = Wq_h
            Wk_h = Wk[perm]
            # kv_a_proj_with_mqa rows: [latent (= v, scaled down) ; rotary key]
            out[q + "self_attn.kv_a_proj_with_mqa.weight"] = torch.cat([Wv * down, Wk_h], 0)
            eps_lat = 1e-6                               # DeepseekV2RMSNorm(kv_lora_rank): eps hard-coded
            out[q + "self_attn.kv_a_layernorm.weight"] = torch.full((hd,), 2.0 ** a * math.sqrt(eps_lat),
                                                                    dtype=torch.float64)
            out[q + "self_attn.kv_b_proj.weight"] = torch.eye(hd, dtype=torch.float64).repeat(H, 1)
            out[q + "self_attn.o_proj.weight"] = Wo * o_sc
            continue
        qk_norm = kind in ("qwen3", "gemma3", "gemma4")
        if qk_norm:
            out[q + "self_attn.q_proj.weight"] = Wq * down
            out[q + "self_attn.k_proj.weight"] = Wk * down
            out[q + "self_attn.q_norm.weight"] = hgain(torch.full((hd,), g_off * s_t / s_h, dtype=torch.float64))
            out[q + "self_attn.k_norm.weight"] = hgain(torch.full((hd,), g_off, dtype=torch.float64))
        else:
            out[q + "self_attn.q_proj.weight"] = Wq * (s_t / s_h)
            out[q + "self_attn.k_proj.weight"] = Wk
        if kind == "gemma4":                              # weight-free v_norm
            out[q + "self_attn.v_proj.weight"] = Wv * down
            o_sc = o_sc * g_off
        else:
            out[q + "self_attn.v_proj.weight"] = Wv
        out[q + "self_attn.o_proj.weight"] = Wo * o_sc
    return out


def write_host(host, W, dtype):
    names = dict(host.named_parameters(remove_duplicate=False))
    missing = [n for n in names if n not in W]
    assert not missing, f"no rule for host parameters {missing[:5]}"
    with torch.no_grad():
        for n, p in names.items():
            if p.dtype == dtype:
                p.data.copy_(W[n].to(dtype))         # in place: no second copy of the host on the GPU
            else:
                p.data = W[n].to(dtype).to(p.device)
    if names.get("lm_head.weight") is not None and getattr(host.config, "tie_word_embeddings", False):
        host.lm_head.weight = host.model.embed_tokens.weight


def off_norms(host, kind):
    return [(n, m) for n, m in host.named_modules() if n.split(".")[-1] in OFF[kind]]


class RhoProbe:
    """max over switched-off norms and tokens of ms(x) / eps (the norm's distance from linear)."""

    def __init__(self, mods):
        self.rho, self.hooks = 0.0, []
        for _, m in mods:
            eps = getattr(m, "variance_epsilon", getattr(m, "eps", None))
            self.hooks.append(m.register_forward_pre_hook(lambda mod, args, eps=eps: self._see(args[0], eps)))

    def _see(self, x, eps):
        self.rho = max(self.rho, (x.detach().double().pow(2).mean(-1).max() / eps).item())

    def remove(self):
        for h in self.hooks:
            h.remove()


# --- scoring (the act_swap standard) ----------------------------------------------------------------
@torch.no_grad()
def reference(model, tok, device, n_prompts=24):
    enc = [tok(p, return_tensors="pt").input_ids for p in PROMPTS[:n_prompts]]
    cont, seqs, lps = [], [], []
    for ids in enc:
        c = greedy(model, ids.to(device)).cpu()
        s = torch.cat([ids, c], 1)
        cont.append(c)
        seqs.append(s)
        lps.append(cont_logp(model, s.to(device), ids.shape[1]))
    return dict(enc=enc, cont=cont, seq=seqs, lp=lps)


@torch.no_grad()
def score(model, ref, device):
    top1 = kl = 0.0
    n = ex32 = ex64 = 0
    dlp, div = 0.0, []
    for ids, s, lq, rc in zip(ref["enc"], ref["seq"], ref["lp"], ref["cont"]):
        lp = cont_logp(model, s.to(device), ids.shape[1])
        kl += (lq.exp() * (lq - lp)).sum().item()
        top1 += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
        dlp = max(dlp, (lp - lq).abs().max().item())
        n += lp.shape[0]
        g = greedy(model, ids.to(device)).cpu()
        same = (g[0] == rc[0]).numpy()
        ex32 += bool(same[:32].all())
        ex64 += bool(same.all())
        div.append(int(np.argmin(same)) if not same.all() else N_NEW)
    m = len(ref["enc"])
    return dict(top1=top1 / n, kl=kl / n, ex32=ex32 / m, ex64=ex64 / m, div=float(np.mean(div)), max_dlogp=dlp)


def _retry(fn, tries=30):
    for k in range(tries):
        try:
            return fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("  OOM, retrying in 60 s", file=sys.stderr, flush=True)
            time.sleep(60)
    return fn()


def load_target(repo, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, attn_implementation=ATTN).eval()
    return tok, model


@torch.no_grad()
def target_ms_max(model, ref, device, kind):
    """max ms of every tensor the lever will feed a switched-off norm (q / k per head, v per head,
    o_proj / down_proj outputs, the MLA latent = v), on the target's own reference sequences."""
    hd = model.config.head_dim if getattr(model.config, "head_dim", None) else \
        model.config.hidden_size // model.config.num_attention_heads
    best = {"q": 0.0, "k": 0.0, "v": 0.0, "o": 0.0, "down": 0.0}
    hooks = []
    for n, m in model.named_modules():
        leaf = n.split(".")[-1]
        key = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "down_proj": "down"}.get(leaf)
        if key is None:
            continue
        per_head = key in ("q", "k", "v")

        def hook(mod, args, out, key=key, per_head=per_head):
            y = out.detach().double()
            if per_head:
                y = y.reshape(*y.shape[:-1], -1, hd)
            best[key] = max(best[key], y.pow(2).mean(-1).max().item())
        hooks.append(m.register_forward_hook(hook))
    for s in ref["seq"]:
        model(s.to(device))
    for h in hooks:
        h.remove()
    return best


def a_for(ms, eps, log2_rho):
    """smallest integer a with 2^-2a ms / eps <= 2^log2_rho."""
    return max(0, math.ceil((math.log2(ms / eps) - log2_rho) / 2))


def run(pair, device="cuda:0", host_device=None, n_prompts=24, sweep=(10, 0, -5, -10, -15, -20, -25, -30, -40),
        primary=PRIMARY, dtypes=("float32", "float64", "bfloat16"), s_variant=20, sink=None):
    repo, kind = PAIRS[pair]
    host_device = host_device or device
    t0 = time.time()
    rows, meta = [], {}

    def emit(r):
        r = dict(pair=pair, target=repo, host=kind, switched_off=list(OFF[kind]), attn=ATTN, **meta, **r)
        r["payable"] = strict(r)
        rows.append(r)
        if sink:
            sink(r)
        print(f"  {r['variant']:>24} {r['dtype']:>8} a={r.get('a')!s:>3} rho_max {r.get('rho_max', 0):.2e}  "
              f"top1 {r['top1']:.4f} ex32 {r['ex32']:.2f} ex64 {r['ex64']:.2f} max|dlogp| {r['max_dlogp']:.2e}"
              f"  {'PASS' if r['payable'] else 'fail'}  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    for dname in [d for d in dtypes if d != "bfloat16"]:
        dt = getattr(torch, dname)
        tok, tgt = load_target(repo, dt)
        t = tdims(tgt.config)
        tfam = tfamily(tgt.config)
        meta["act"] = t["act"]                      # (tf5 reads gemma-2b's legacy "gelu" as tanh, 4.57 as erf)
        sd = {n: p.detach().cpu() for n, p in tgt.state_dict().items()}
        tgt = _retry(lambda: tgt.to(device))
        ref = reference(tgt, tok, device, n_prompts)
        ms = target_ms_max(tgt, ref, device, kind)
        # the inputs each switched-off norm will see, at a = 0
        feeds = {"qwen3": ("q", "k"), "gemma2": ("o", "down"), "gemma3": ("q", "k", "o", "down"),
                 "gemma4": ("q", "k", "v", "o", "down"), "mla": ("v",)}[kind]
        eps_off = 1e-6 if kind == "mla" else t["eps"]
        ms_max = max(ms[f] for f in feeds)
        print(f"{pair} {dname}: target ms max {ms} -> ms/eps {ms_max / eps_off:.3g}", file=sys.stderr, flush=True)
        if dname == "float32" and "bfloat16" in dtypes:          # target bf16 control vs the fp32 reference
            for n, p in tgt.named_parameters():
                p.data = sd[n].to(torch.bfloat16).to(device)
            emit(dict(variant="target_bf16_control", dtype="bfloat16", **score(tgt, ref, device)))
            for n, p in tgt.named_parameters():
                p.data = sd[n].to(device)
        if host_device == device:
            tgt.cpu()
            torch.cuda.empty_cache()
        cfg = host_config(kind, t, t["eps"])
        host = _retry(lambda: _build(kind, cfg, dt, host_device))
        mods = off_norms(host, kind)
        assert mods, "no switched-off norm found on the host"
        sweep_here = sweep if dname == "float32" else (primary,)
        for k in sweep_here:
            a = a_for(ms_max, eps_off, k)
            write_host(host, host_weights(kind, tfam, t, sd, a), dt)
            probe = RhoProbe(mods)
            r = score(host, ref, host_device)
            probe.remove()
            emit(dict(variant=f"host log2rho<={k}", dtype=dname, a=a, rho_max=probe.rho, ms_over_eps=ms_max / eps_off,
                      n_off=len(mods), **r))
            if k == primary and dname == "float32" and "bfloat16" in dtypes:
                write_host(host, host_weights(kind, tfam, t, sd, a), torch.float32)
                for n, p in host.named_parameters():
                    p.data = p.data.to(torch.bfloat16)
                probe = RhoProbe(mods)
                r = score(host, ref, host_device)
                probe.remove()
                emit(dict(variant=f"host log2rho<={k}", dtype="bfloat16", a=a, rho_max=probe.rho,
                          ms_over_eps=ms_max / eps_off, n_off=len(mods), **r))
        if dname == "float32" and s_variant and kind not in ONE_PLUS_W:
            # (not for a (1 + w) host: its tied head needs a final-norm gain 1 / S, and 1 + w with
            #  w = 1 / S - 1 is computed in fp32 (Gemma2RMSNorm), which loses the gain -- S = 1 is the split)
            # frontier-levers' split: residual x S, host eps = e_t S^2 (the same lever)
            a = a_for(ms_max, eps_off, primary)
            aS = a if kind == "mla" else max(0, a - s_variant)   # the MLA latent eps is hard-coded, not E
            cfgS = host_config(kind, t, t["eps"] * 4.0 ** s_variant)
            hostS = _retry(lambda: _build(kind, cfgS, dt, host_device))
            modsS = off_norms(hostS, kind)
            write_host(hostS, host_weights(kind, tfam, t, sd, aS, s_variant), dt)
            probe = RhoProbe(modsS)
            r = score(hostS, ref, host_device)
            probe.remove()
            emit(dict(variant=f"host S=2^{s_variant} log2rho<={primary}", dtype=dname, a=aS,
                      s_log2=s_variant, rho_max=probe.rho, ms_over_eps=ms_max / eps_off, n_off=len(modsS), **r))
            del hostS
        del host, tgt, sd, ref
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=sorted(PAIRS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host-device")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--dtypes", default="float32,float64,bfloat16")
    ap.add_argument("--sweep", default="10,0,-5,-10,-15,-20,-25,-30,-40")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if PAIRS[a.pair][1] == "gemma4" and not os.environ.get("EPS_LEVER_TF5"):
        import subprocess
        tf5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prelude = ("import sys, types\n"
                   "m = types.ModuleType('transformers.dependency_versions_check'); m.dep_version_check = lambda *a, **k: None\n"
                   "sys.modules['transformers.dependency_versions_check'] = m\n"
                   "sys.modules['fla'] = None\n"
                   f"sys.argv = {['eps_lever'] + sys.argv[1:]!r}\n"
                   "import runpy; runpy.run_module('hf.eps_lever_agreement', run_name='__main__')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([tf5, root]), EPS_LEVER_TF5="1")
        sys.exit(subprocess.run([sys.executable, "-c", prelude], env=env).returncode)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def sink(r):
        if os.environ.get("EPS_LEVER_TF5"):
            import transformers
            r["transformers"] = transformers.__version__
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.pair, a.device, a.host_device, a.n_prompts, sweep=tuple(int(x) for x in a.sweep.split(",")),
        dtypes=tuple(a.dtypes.split(",")), sink=sink)


if __name__ == "__main__":
    main()
