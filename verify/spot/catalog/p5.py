"""Round 2, pass 5: the config levers, the eps lever per norm kind, the start-token rules, the activation width,
the looped and dense hosts (graph/model_graph.py P5_PORTS), built with the repo's own pass-5 construction modules
(hf/eps_lever_agreement.py, hf/bos_constructions.py, hf/looped_host_prenorm.py, moe/dense_host_prenorm.py,
act/data/act_width_fits.json) and, for the config levers, elephantine/emulate.py after the lever is applied."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch

from verify.spot.catalog import Result, compare, entry, hf_fn, prompts
from verify.spot.catalog._emu import BIG, SMALL, V, emulate_pair, model_from, raw
from verify.spot.catalog._ref import F64, RefLM, fn
from verify.spot.catalog.ref_slots import REF, _check, _host, _t

ROOT = Path(__file__).resolve().parents[3]


def _lift():
    from tests.test_universal_elephantine import _lift_fp32
    return _lift_fp32()


def _edge_ok(traw, hraw):
    from graph.edge_plan import edge_plan
    from graph.model_graph import classify, dims_of
    p = edge_plan({"slots": classify(traw), "dims": dims_of(traw)}, {"slots": classify(hraw), "dims": dims_of(hraw)})
    assert p["ok"], "not an edge of the headline relation"
    return p


def _lever_emu(tmt, tkw, hmt, hkw, lever, tol=1e-9, note=""):
    """The pair (host's own config) is an edge; the host's config lever is applied; emulate.py builds the rest."""
    traw, hraw = raw(tmt, {**SMALL, **tkw}), raw(hmt, {**BIG, **hkw})
    _edge_ok(traw, hraw)
    h2 = {**hraw, **lever}
    target, host, con, _ = emulate_pair(traw, h2)
    with _lift():
        return compare(hf_fn(target), hf_fn(host), V, tol=tol, note=note or f"lever {lever}")


# ------------------------------------------------------------------ config levers

@entry("port:window", instance="llama (full layers) -> mistral (sliding window 4 on every layer): the window lever "
                               "sets the host's window >= the context, so its sliding layers are full ones")
def p5_window_lever():
    return _lever_emu("llama", {}, "mistral", dict(sliding_window=4), dict(sliding_window=None))


@entry("port:partial", instance="llama (full RoPE) -> phi3 (partial_rotary_factor 0.5): the rotary-fraction lever "
                                "raises the host's factor to 1, then emulate.py")
def p5_partial_lever():
    return _lever_emu("llama", {}, "phi3", dict(partial_rotary_factor=0.5, pad_token_id=None),
                      dict(partial_rotary_factor=1.0))


@entry("port:softcap", instance="gemma2 without caps -> gemma2 with attention / final caps 50 / 30: the caps set to "
                                "null in the host's config (exact)")
def p5_softcap_lever():
    LT = ["sliding_attention", "full_attention", "sliding_attention"]
    return _lever_emu("gemma2", dict(sliding_window=4, layer_types=LT[:2], query_pre_attn_scalar=16,
                                     attn_logit_softcapping=None, final_logit_softcapping=None),
                      "gemma2", dict(sliding_window=4, layer_types=LT, query_pre_attn_scalar=24,
                                     attn_logit_softcapping=50.0, final_logit_softcapping=30.0),
                      dict(attn_logit_softcapping=None, final_logit_softcapping=None))


@entry("port:rot_config", instance="llama (head_dim 16) -> llama (head_dim 24): a fixed-rotary host sets only "
                                   "rope_theta, theta_h = theta_t^(24 / 16) (emulate.py)")
def p5_rot_config():
    return _lever_emu("llama", {}, "llama", dict(head_dim=24), {}, tol=1e-6,
                      note="inv_freq at fp32 init: ~1e-7 relative")


def _stock(mt, seed=0, noise=0.08, **kw):
    from transformers import AutoConfig, AutoModelForCausalLM
    from elephantine.emulate import warnings_off
    c = AutoConfig.for_model(mt, vocab_size=V, **kw)
    c._attn_implementation = "eager"
    with warnings_off():
        torch.manual_seed(seed)
        m = AutoModelForCausalLM.from_config(c).to(F64).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * noise)
    return m


@entry("port:qk_switch", "slot:qknorm:no->head_ln",
       instance="cohere without qk-norm -> cohere with use_qk_norm (per-head LayerNorm): the flag switched off")
def p5_qk_switch():
    kw = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=128)
    t = _stock("cohere", use_qk_norm=False, **kw)
    h = _stock("cohere", seed=1, use_qk_norm=True, **kw)
    d = lambda m: {**m.config.to_dict(), "architectures": [type(m).__name__]}
    _edge_ok(d(t), d(h))
    from transformers import AutoModelForCausalLM
    hc = copy.deepcopy(h.config)
    hc.use_qk_norm = False                               # the lever
    h2 = AutoModelForCausalLM.from_config(hc).to(F64).eval()
    h2.load_state_dict(t.state_dict())
    with _lift():
        return compare(hf_fn(t), hf_fn(h2), V)


# ------------------------------------------------------------------ the eps lever (hf/eps_lever_agreement.py)

def _el_target(fam, seed=0):
    torch.manual_seed(seed)
    kw = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
              head_dim=16, max_position_embeddings=64, rope_theta=10000.0, rms_norm_eps=1e-6,
              attn_implementation="eager", tie_word_embeddings=True)
    if fam == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        m = LlamaForCausalLM(LlamaConfig(num_key_value_heads=2, **kw))
    else:
        from transformers import GemmaConfig, GemmaForCausalLM
        m = GemmaForCausalLM(GemmaConfig(num_key_value_heads=1, hidden_act="gelu", **kw))
    m = m.to(F64).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5) if fam == "llama" else p.uniform_(-0.5, 0.5)
            elif "embed" in n:
                p.normal_(0, 1.0)
            else:
                p.normal_(0, 0.3)
    return m


def _el(kind, fam, a=30):
    from hf import eps_lever_agreement as el
    tgt = _el_target(fam)
    t = el.tdims(tgt.config)
    sd = {n: p.detach() for n, p in tgt.state_dict().items()}
    host = el._causal_lm(kind, el.host_config(kind, t, t["eps"])).to(F64).eval()
    el.write_host(host, el.host_weights(kind, el.tfamily(tgt.config), t, sd, a), F64)
    with _lift():
        return compare(hf_fn(tgt), hf_fn(host), V, standard="limit", tol=1e-9,
                       note=f"eps lever, input x 2^-{a}")


@entry("port:eps_qknorm", "slot:qknorm:no->head", standard="limit",
       instance="llama -> qwen3 (per-head qk-norm): q / k rows x 2^-30, the norm linear, gain 2^30 sqrt(eps)")
def p5_eps_qknorm_qwen3():
    return _el("qwen3", "llama")


@entry("port:eps_feats", standard="limit",
       instance="gemma -> gemma2 (sandwich post-norms): the post-norm inputs x 2^-30, the norms linear")
def p5_eps_feats_gemma2():
    return _el("gemma2", "gemma")


@entry("port:eps_feats", "port:eps_qknorm", standard="limit",
       instance="gemma -> gemma3 (sandwich post-norms and qk-norm), both switched off by the eps lever")
def p5_eps_feats_gemma3():
    return _el("gemma3", "gemma")


@entry("port:eps_mla_plain", "slot:kv:MQA->MLA", standard="limit",
       instance="gemma MQA -> deepseek_v2 MLA: V in the latent (its kv_a_layernorm linear by the eps lever), K the "
                "shared rotary key")
def p5_eps_mla_mqa():
    return _el("mla", "gemma")


def _qk_lever(variant, s=2.0 ** -40):
    """A plain target on a host with qk-norm `variant`: q / k rows x s, so the norm reads s x with s^2 ms << eps
    and is linear; its gain sqrt(eps) / s undoes it (reference decoder)."""
    t = _t()
    h = _host(t, qknorm=variant)
    e, H, KV, hd = t.c["qk_eps"], t.c["H"], t.c["KV"], t.c["hd"]
    g = math.sqrt(e) / s
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        h.p[pre + "q"], h.p[pre + "k"] = t.p[pre + "q"] * s, t.p[pre + "k"] * s
        shape = {"head": (hd,), "head_post": (hd,), "proj": None}[variant]
        h.p[pre + "qn"] = torch.full(shape or (H * hd,), g, dtype=F64)
        h.p[pre + "kn"] = torch.full(shape or (KV * hd,), g, dtype=F64)
    return _check(t, h, standard="limit", tol=1e-9, note="eps lever, q / k x 2^-40")


@entry("slot:qknorm:no->head_post", "port:eps_qknorm", standard="limit",
       instance=f"no qk-norm -> per-head qk-norm after RoPE (Hunyuan), switched off by the eps lever ({REF})")
def p5_eps_qk_head_post():
    return _qk_lever("head_post")


@entry("slot:qknorm:no->proj", "port:eps_qknorm", standard="limit",
       instance=f"no qk-norm -> whole-projection qk-norm (OLMo-2 style), switched off by the eps lever ({REF})")
def p5_eps_qk_proj():
    return _qk_lever("proj")


_DS2 = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, first_k_dense_replace=0,
            kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
            n_shared_experts=1, topk_method="greedy", max_position_embeddings=128, tie_word_embeddings=False)


@entry("port:eps_latent", "capacity:mla_latent", standard="limit",
       instance="deepseek_v2 MLA (kv latent 16) -> deepseek_v2 (latent 24, not a multiple): zero-padded, the latent "
                "input x sqrt(24 / 16) so the hard-coded eps reads as the target's, the gain / sqrt(24 / 16)")
def p5_eps_latent():
    t = _stock("deepseek_v2", **_DS2)
    h = _stock("deepseek_v2", seed=1, **{**_DS2, "kv_lora_rank": 24})
    for p in h.parameters():
        p.data.zero_()
    ts, hs = t.state_dict(), h.state_dict()
    kl, klh = 16, 24
    s = math.sqrt(klh / kl)
    with torch.no_grad():
        for k, v in ts.items():
            if "kv_a_proj_with_mqa" in k:
                hs[k][:kl], hs[k][klh:] = v[:kl] * s, v[kl:]
            elif "kv_a_layernorm" in k:
                hs[k][:kl] = v / s
            elif "kv_b_proj" in k:
                hs[k][:, :kl] = v
            else:
                hs[k].copy_(v)
    with _lift():
        return compare(hf_fn(t), hf_fn(h), V, standard="limit", tol=1e-9)


@entry("port:eps_rec", "recurrent:m2", standard="limit",
       instance="mamba2 (8 heads x 8, gated norm over 64) -> mamba2 (12 heads x 8): zero-padded channels, the shared "
                "eps x 64 / 96 and the residual stream x sqrt(64 / 96) so the stream norms read the same")
def p5_eps_mamba2():
    M2 = dict(hidden_size=32, state_size=8, expand=2, num_heads=8, head_dim=8, n_groups=1, conv_kernel=4,
              num_hidden_layers=2, tie_word_embeddings=False, layer_norm_epsilon=1e-5)
    t = _stock("mamba2", **M2)
    r = 64 / 96
    h = _stock("mamba2", seed=1, **{**M2, "expand": 3, "num_heads": 12, "layer_norm_epsilon": 1e-5 * r})
    for p in h.parameters():
        p.data.zero_()
    _edge_ok({**t.config.to_dict(), "architectures": ["Mamba2ForCausalLM"]},
             {**_stock("mamba2", seed=2, **{**M2, "expand": 3, "num_heads": 12}).config.to_dict(),
              "architectures": ["Mamba2ForCausalLM"]})
    a = math.sqrt(r)
    ts, hs = t.state_dict(), h.state_dict()
    I, N, nh, Ih, nhh = 64, 8, 8, 96, 12
    with torch.no_grad():
        for k, v in ts.items():
            H = hs[k]
            if "embeddings" in k:
                H.copy_(v * a)
            elif ".mixer." not in k:
                H.copy_(v)                                   # norms: scale-invariant up to eps, eps scaled by a^2
            else:
                n = k.split(".mixer.")[1]
                if n == "in_proj.weight":
                    z, x, B, C, dt = v.split([I, I, N, N, nh])
                    H[:I], H[Ih:Ih + I] = z, x
                    H[2 * Ih:2 * Ih + N], H[2 * Ih + N:2 * Ih + 2 * N] = B, C
                    H[2 * Ih + 2 * N:2 * Ih + 2 * N + nh] = dt
                elif n in ("conv1d.weight", "conv1d.bias"):
                    x, B, C = v.split([I, N, N])
                    H[:I], H[Ih:Ih + N], H[Ih + N:Ih + 2 * N] = x, B, C
                elif n in ("dt_bias", "A_log", "D"):
                    H[:nh] = v
                elif n == "norm.weight":
                    H[:I] = v * math.sqrt(r)
                elif n == "out_proj.weight":
                    H[:, :I] = v * a
    with _lift():
        return compare(hf_fn(t), hf_fn(h), V, standard="limit", tol=1e-9)


# ------------------------------------------------------------------ plain targets on MLA hosts, KV heads packed

@torch.no_grad()
def _mla_packed(KVt, H=4):
    """llama (H query heads on KVt KV heads of head_dim 8, full RoPE) -> deepseek_v2 MLA (H heads, nope dims 0, the
    ONE shared rotary key of 8 KVt dims holding every target KV head's rotating key side by side on a longrope table
    that repeats the target's frequencies per block; the latent (8 KVt) carrying the values, its kv_a_layernorm
    linear by the eps lever; kv_b_proj handing each host head its target KV head's values)."""
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, LlamaConfig, LlamaForCausalLM
    from hf.eps_lever_agreement import _interleave_perm
    hd, d, L = 8, 32, 2
    tc = LlamaConfig(vocab_size=V, hidden_size=d, intermediate_size=48, num_hidden_layers=L, num_attention_heads=H,
                     num_key_value_heads=KVt, head_dim=hd, max_position_embeddings=128, tie_word_embeddings=False)
    tc._attn_implementation = "eager"
    torch.manual_seed(0)
    t = LlamaForCausalLM(tc).to(F64).eval()
    for p in t.parameters():
        p.add_(torch.randn_like(p) * 0.08)
    dr = hd * KVt
    half = hd // 2
    inv_t = 1.0 / (tc.rope_theta ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    # host interleaved pair j = b half + i holds target pair i of block b: inv_freq_h[j] = inv_t[i]
    inv_h0 = 1.0 / (tc.rope_theta ** (torch.arange(0, dr, 2, dtype=F64) / dr))
    fac = [float(inv_h0[j] / inv_t[j % half]) for j in range(dr // 2)]
    rs = {"rope_type": "longrope", "short_factor": fac, "long_factor": fac, "factor": 1.0, "attention_factor": 1.0,
          "original_max_position_embeddings": 128}
    hc = DeepseekV2Config(vocab_size=V, hidden_size=d, intermediate_size=48, num_hidden_layers=L,
                          num_attention_heads=H, num_key_value_heads=H, kv_lora_rank=dr, q_lora_rank=None,
                          qk_nope_head_dim=0, qk_rope_head_dim=dr, v_head_dim=hd, first_k_dense_replace=L,
                          rope_theta=tc.rope_theta, rope_scaling=rs, max_position_embeddings=128,
                          rms_norm_eps=tc.rms_norm_eps, tie_word_embeddings=False, n_routed_experts=2, num_experts_per_tok=1,
                          moe_intermediate_size=48)
    hc._attn_implementation = "eager"
    h = DeepseekV2ForCausalLM(hc).to(F64).eval()
    for p in h.parameters():
        p.zero_()
    _edge_ok({**tc.to_dict(), "architectures": ["LlamaForCausalLM"], "model_type": "llama"},
             {**hc.to_dict(), "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2"})
    perm = _interleave_perm(hd)                            # half-split (i, i + hd/2) -> interleaved (2i, 2i + 1)
    g = tc.num_attention_heads // KVt
    a = 2.0 ** -30
    fold = math.sqrt(dr / hd)                              # host scale 1 / sqrt(dr) vs the target's 1 / sqrt(hd)
    for k in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        h.state_dict()[k].copy_(t.state_dict()[k])
    for Lt, Lh in zip(t.model.layers, h.model.layers):
        at, ah = Lt.self_attn, Lh.self_attn
        Lh.input_layernorm.weight.copy_(Lt.input_layernorm.weight)
        Lh.post_attention_layernorm.weight.copy_(Lt.post_attention_layernorm.weight)
        for m in ("gate_proj", "up_proj", "down_proj"):
            getattr(Lh.mlp, m).weight.copy_(getattr(Lt.mlp, m).weight)
        Wk, Wv = at.k_proj.weight.view(KVt, hd, d), at.v_proj.weight.view(KVt, hd, d)
        A = ah.kv_a_proj_with_mqa.weight                     # [latent (dr); rotary key (dr)]
        for c in range(KVt):
            A[c * hd:(c + 1) * hd] = Wv[c] * a               # the values, scaled down: the latent norm is linear
            A[dr + c * hd: dr + (c + 1) * hd] = Wk[c][perm]  # block c of the shared rotary key, interleaved
        ah.kv_a_layernorm.weight.fill_(math.sqrt(1e-6) / a)
        B = ah.kv_b_proj.weight.view(H, hd, dr)              # head j's values = its target KV head's
        for j in range(H):
            c = j // g
            B[j, :, c * hd:(c + 1) * hd] = torch.eye(hd, dtype=F64)
        Q = ah.q_proj.weight.view(H, dr, d)
        Wq = at.q_proj.weight.view(H, hd, d)
        for j in range(H):
            c = j // g
            Q[j, c * hd:(c + 1) * hd] = Wq[j][perm] * fold
        ah.o_proj.weight.copy_(at.o_proj.weight)
    with _lift():
        return compare(hf_fn(t), hf_fn(h), V, standard="limit", tol=1e-6,
                       note="eps lever on the latent norm; longrope factors fp32 (~1e-7)")


@entry("slot:kv:MHA->MLA", "port:eps_mla_plain", standard="limit",
       instance="llama MHA (4 heads x 8) -> deepseek_v2 MLA: 4 rotating keys packed in the shared rotary key (32 dims, "
                "a longrope table), values through the latent (eps lever)")
def p5_mla_packed_mha():
    return _mla_packed(4)


@entry("slot:kv:GQA->MLA", "port:eps_mla_plain", standard="limit",
       instance="llama GQA (4 q / 2 kv x 8) -> deepseek_v2 MLA: 2 rotating keys packed in the shared rotary key")
def p5_mla_packed_gqa():
    return _mla_packed(2)


# ------------------------------------------------------------------ the start-token rules (hf/bos_constructions.py)

def _rand_gpt2(seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=97, n_embd=32, n_layer=2, n_head=4, n_positions=64, bos_token_id=96,
                     eos_token_id=96, attn_implementation="sdpa")
    m = GPT2LMHeadModel(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "ln" in n:
                p.add_(0.2 * torch.randn_like(p))
    return m


def _bos_ids(n=8, T=10, bos=96, V_=96):
    ids = prompts(V_, n=n, T=T)
    for x in ids:
        x[:, 0] = bos
    return ids


@entry("port:bos_labs", "slot:pos:learned_abs->rope", "slot:pos:learned_abs->nope", "slot:pos:learned_abs->partial_rope",
       instance="gpt2 (learned positions, LayerNorm) -> llama with inv_freq 0, +2 layers, +5 channels: the start-token "
                "rule (hf/bos_constructions.learned_abs_host). A RoPE host at inv_freq 0 is the NoPE host, and a "
                "partial-RoPE host at inv_freq 0 on its rotated dims computes the same function")
def p5_bos_learned_abs():
    from hf.bos_constructions import exact_norms, gpt2_spec, learned_abs_host
    tgt = _rand_gpt2()
    host = learned_abs_host(gpt2_spec(tgt))
    with exact_norms():
        return compare(hf_fn(tgt), hf_fn(host), 97, ids=_bos_ids(), tol=1e-9, note="inputs start with the BOS")


def _rand_gptoss(seed=0):
    from transformers import GptOssConfig, GptOssForCausalLM
    torch.manual_seed(seed)
    cfg = GptOssConfig(vocab_size=101, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       intermediate_size=16, num_local_experts=4, num_experts_per_tok=2, num_hidden_layers=2,
                       layer_types=["sliding_attention", "full_attention"], sliding_window=64, rope_theta=1e4,
                       rope_scaling=None, max_position_embeddings=256, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    m = GptOssForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if "norm" not in n else 0.1) + (1.0 if "norm" in n else 0.0))
            if n.endswith("sinks"):
                p.copy_(2.0 * torch.randn_like(p))
    return m


@entry("port:bos_sink", instance="gpt-oss (learned sinks) -> sink-free q-bias attention (Qwen2Attention, head_dim + 2 "
                                 "with a zero-frequency spare pair): a host-only marker supplies the sink key "
                                 "(hf/bos_constructions.sink_host)")
def p5_bos_sink():
    from hf.bos_constructions import exact_norms, sink_host
    tgt = _rand_gptoss()
    host = sink_host(tgt)
    Vt = tgt.config.vocab_size

    def hfn(x):
        T = x.shape[1]
        hx = torch.cat([torch.tensor([[host.marker_id]]), x], 1)
        return host(hx, position_ids=torch.arange(T + 1)[None] - 1).logits[:, 1:, :Vt]
    with exact_norms():
        return compare(hf_fn(tgt), hfn, Vt, tol=1e-9, note="the marker is the host's only")


# ------------------------------------------------------------------ activation width and GeLU erf <-> tanh

FITS = None


def _fit(target, host, k, shift=False):
    global FITS
    if FITS is None:
        FITS = json.loads((ROOT / "act/data/act_width_fits.json").read_text())
    rows = FITS[f"{target}|{host}|160|{'shift' if shift else 'noshift'}"]
    row = next(r for r in rows if r[0] == k)
    return row[1], row[2]


def _act_width(t_act, h_act, fit_t, fit_h, k, ffn="vanilla"):
    """Each target unit Y(a) b -> k host units sum_i alpha_i X(beta_i a) b (gate rows x beta_i, down columns x
    alpha_i), FFN width x k (graph/act_width.py). A swigluoai host's (u + 1) is held at the target's b by an up bias
    -1 (gpt-oss has one)."""
    err, p = _fit(fit_t, fit_h, k)
    t = _t(ffn=ffn, act=t_act)
    F_ = t.c["F"]
    hcfg = dict(t.c, act=h_act, F=F_ * k, ffn="glu" if h_act == "swigluoai" else ffn,
                out_bias=h_act == "swigluoai")
    h = RefLM(**hcfg).init(1)
    for key, v in t.p.items():
        if key.split(".")[-1] not in ("up", "fgate", "down", "upb", "downb", "ob"):
            h.p[key] = v.clone() if v is not None else None
    al, be = torch.tensor(p["alpha"], dtype=F64), torch.tensor(p["beta"], dtype=F64)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        a_rows = t.p[pre + ("fgate" if ffn == "glu" else "up")]
        b_rows = t.p[pre + "up"] if ffn == "glu" else None
        G = torch.cat([a_rows * be[j] for j in range(k)], 0)
        D = torch.cat([t.p[pre + "down"] * al[j] for j in range(k)], 1)
        if h.c["ffn"] == "glu":
            h.p[pre + "fgate"] = G
            if b_rows is not None:
                h.p[pre + "up"] = torch.cat([b_rows] * k, 0)
            else:
                h.p[pre + "up"] = torch.zeros(F_ * k, t.c["d"], dtype=F64)
            if h.c["out_bias"]:
                h.p[pre + "upb"] = torch.full((F_ * k,), -1.0 if b_rows is not None else 0.0, dtype=F64)
                h.p[pre + "downb"] = torch.zeros(t.c["d"], dtype=F64)
                h.p[pre + "ob"] = torch.zeros(t.c["d"], dtype=F64)
        else:
            h.p[pre + "up"] = G
        h.p[pre + "down"] = D
    return _check(t, h, standard="approx", note=f"k = {k}, fit error {err:.1e} on [-160, 160]")


@entry("port:act_width", "slot:act:GeLU->SiLU", standard="approx",
       instance=f"GeLU MLP -> SiLU MLP at FFN x 6 (6 scaled SiLU units per GeLU unit) ({REF})")
def p5_act_width_gelu_silu():
    return _act_width("gelu", "silu", "GeLU", "SiLU", 6)


@entry("slot:act:GeLU_tanh->SiLU", "port:act_width", standard="approx",
       instance=f"tanh-GeLU MLP -> SiLU MLP at FFN x 6 ({REF})")
def p5_act_width_gelutanh_silu():
    return _act_width("gelu_tanh", "silu", "gelu_tanh", "SiLU", 6)


@entry("slot:act:SiLU->GeLU", "port:act_width", standard="approx",
       instance=f"SiLU MLP -> GeLU (erf) MLP at FFN x 3 ({REF})")
def p5_act_width_silu_gelu():
    return _act_width("silu", "gelu", "SiLU", "GeLU", 3)


@entry("slot:act:SiLU->GeLU_tanh", "port:act_width", standard="approx",
       instance=f"SiLU MLP -> tanh-GeLU MLP at FFN x 3 ({REF})")
def p5_act_width_silu_gelutanh():
    return _act_width("silu", "gelu_tanh", "SiLU", "gelu_tanh", 3)


@entry("slot:act:GeLU->swigluoai", "port:act_width", standard="approx",
       instance=f"GeGLU -> swigluoai GLU (gate <= 7, up bias -1 for the (u + 1)) at FFN x 6 ({REF})",
       finding="graph/act_width's k-unit fits for a swigluoai host (act/act_width_fit.py TA['swigluoai'] = "
               "z sigmoid(1.702 z)) ignore gpt-oss's gate clamp min(g, 7) (swiglu_limit): the k = 6 GeLU fit's betas "
               "reach 3.8, so the clamp binds from |a| ~ 1.8; with the clamp the fit's error on [-6, 6] is 0.54, not "
               "4.8e-4. Every act-width cell with a swigluoai host is affected (7 headline edges GeLU <- swigluoai)")
def p5_act_width_gelu_oai():
    return _act_width("gelu", "swigluoai", "GeLU", "swigluoai", 6, ffn="glu")


def _gelu_swap(t_act, h_act):
    t = _t(ffn="vanilla", act=t_act)
    return _check(t, _host(t, act=h_act), standard="approx", note="raw swap, floor 3e-4")


@entry("slot:act:GeLU->GeLU_tanh", "port:gelu", standard="approx",
       instance=f"GeLU (erf) MLP -> tanh-GeLU MLP, the same weights ({REF})")
def p5_gelu_erf_tanh():
    return _gelu_swap("gelu", "gelu_tanh")


@entry("slot:act:GeLU_tanh->GeLU", "port:gelu", standard="approx",
       instance=f"tanh-GeLU MLP -> GeLU (erf) MLP, the same weights ({REF})")
def p5_gelu_tanh_erf():
    return _gelu_swap("gelu_tanh", "gelu")


# ------------------------------------------------------------------ LayerNorm without a beta

@entry("port:ln_beta", "slot:norm:LayerNorm->RMSNorm",
       instance=f"LayerNorm without a beta (Cohere / OLMo-1) -> RMSNorm host without an lm_head bias: writers "
                f"centred, nothing to fold ({REF})")
def p5_ln_no_beta():
    t = _t(norm="ln", ffn="vanilla", act="gelu", qkv_bias=True, out_bias=True, ln_bias=False)
    h = _host(t, norm="rms")
    cen = lambda W: W - W.mean(0, keepdim=True)
    h.p["emb"] = t.p["emb"] - t.p["emb"].mean(1, keepdim=True)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        h.p[pre + "o"], h.p[pre + "ob"] = cen(t.p[pre + "o"]), t.p[pre + "ob"] - t.p[pre + "ob"].mean()
        h.p[pre + "down"], h.p[pre + "downb"] = cen(t.p[pre + "down"]), t.p[pre + "downb"] - t.p[pre + "downb"].mean()
    return _check(t, h)


# ------------------------------------------------------------------ looped and dense hosts

@entry("port:loop", "slot:loop:untied->looped", standard="limit",
       instance="llama (3 untied layers) -> a looped host running its block K = 2 times, +2 channels (counter, start "
                "flag), under the start-token rule (hf/looped_host_prenorm.py, variant 'host')")
def p5_looped_host():
    import hf.looped_host_prenorm as LH
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=64, max_position_embeddings=64, initializer_range=0.2,
                      rms_norm_eps=1e-5, bos_token_id=1)
    m = LlamaForCausalLM(cfg).eval()
    net = LH.Llama(m)
    ids = [torch.cat([torch.tensor([[1]]), x + 2], 1) for x in prompts(62, T=9)]   # the start token only first
    tf = lambda x: net.forward(x)[None]
    hf_ = lambda x: net.forward(x, K=2, v=LH.VARIANTS["host"], bos_id=1)[None]
    r = compare(lambda x: tf(x)[:, 1:], lambda x: hf_(x)[:, 1:], 64, ids=ids, standard="limit", tol=1e-4,
                new_tokens=0, note="every position but the start token; fp32 module")
    # greedy continuations: the target's and the host's own, from the start token on
    same = 0
    for x in ids:
        a, b = x, x
        for _ in range(32):
            a = torch.cat([a, tf(a)[0, -1].argmax().view(1, 1)], 1)
            b = torch.cat([b, hf_(b)[0, -1].argmax().view(1, 1)], 1)
        same += torch.equal(a, b)
    r.identical, r.n = same, len(ids)
    return r


@entry("moe:dense_host_top1", "slot:ffn:moe->glu", standard="limit",
       instance="mixtral (4 experts, top-1) -> a dense host: per MoE layer a select FFN writing gamma-scaled masks "
                "into 4 carried channels and an expert FFN of masked units + erase units (moe/dense_host_prenorm.py "
                "'top1', simulated on the target's own modules)")
def p5_dense_host_top1():
    import moe.dense_host_prenorm as D
    from transformers import MixtralConfig, MixtralForCausalLM
    torch.manual_seed(0)
    cfg = MixtralConfig(hidden_size=48, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=4, num_local_experts=4, num_experts_per_tok=1, vocab_size=64,
                        max_position_embeddings=64, initializer_range=0.3, rms_norm_eps=1e-5)
    t = MixtralForCausalLM(cfg).eval()
    h = copy.deepcopy(t)
    D.Installed(h, D.moe_blocks(h), "top1", 1)
    return compare(hf_fn(t), hf_fn(h), 64, standard="limit", tol=1e-5, note="fp32 modules; sharp constants")


# ------------------------------------------------------------------ vanilla targets on gated hosts

@entry("slot:ffn:vanilla->moe", "slot:ffn:vanilla->moe_shared", "moe:host_for_dense",
       instance=f"vanilla SiLU MLP -> MoE host with expert up biases (gpt-oss style): every expert the FFN with its "
                f"up branch held at 1 by the bias, router 0 ({REF})")
def p5_vanilla_on_moe_bias():
    t = _t(ffn="vanilla", act="silu")
    E, k = 4, 2
    h = RefLM(**{**t.c, "E": E, "k": k, "ew": t.c["F"], "ebias": True, "shared": 8}).init(1)
    for key, v in t.p.items():
        if key.split(".")[-1] not in ("up", "down"):
            h.p[key] = v.clone()
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        h.p[pre + "router"].zero_()
        for e in range(E):
            h.p[pre + "eg"][e] = t.p[pre + "up"]
            h.p[pre + "eu"][e].zero_()
            h.p[pre + "eub"][e].fill_(1.0)
            h.p[pre + "ed"][e] = t.p[pre + "down"]
        h.p[pre + "sd"].zero_()
    return _check(t, h)


@entry("slot:ffn:vanilla->glu", instance=f"vanilla SiLU MLP -> bias-free SwiGLU host: the up branch read off a "
                                         f"constant residual channel (the only bias-free way to a constant) ({REF})",
       finding="the vanilla <- GLU step is FREE in FFN_ORDER, but the construction (EP.glu_ungated: 'up weights 0, "
               "bias 1') needs an up-projection bias; a bias-free host can only read a constant residual channel, "
               "which the pre-norm divides by the token's rms. 366 headline edges (and 134 vanilla -> MoE ones) use a "
               "host without an FFN bias (dims ffn_bias = 0)")
def p5_vanilla_on_glu_nobias():
    t = _t(ffn="vanilla", act="silu")
    d = t.c["d"]
    h = RefLM(**{**t.c, "ffn": "glu", "d": d + 1}).init(1)
    h.p = {k_: torch.zeros_like(v) for k_, v in h.p.items()}
    h.p["emb"][:, :d], h.p["emb"][:, d] = t.p["emb"], 1.0          # the constant channel
    h.p["lm"][:, :d] = t.p["lm"]
    h.p["nf_w"][:d] = t.p["nf_w"] * math.sqrt(d / (d + 1))
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        for n in ("n1_w", "n2_w"):
            h.p[pre + n][:d] = t.p[pre + n] * math.sqrt(d / (d + 1))
            h.p[pre + n][d] = 1.0
        for key in ("q", "k", "v"):
            h.p[pre + key][:, :d] = t.p[pre + key]
        h.p[pre + "o"][:d] = t.p[pre + "o"]
        h.p[pre + "fgate"][:, :d] = t.p[pre + "up"]
        h.p[pre + "up"][:, d] = 1.0                                   # u = the normed constant channel
        h.p[pre + "down"][:d] = t.p[pre + "down"]
    return _check(t, h, note="the normed constant is 1 / rms(x), not 1")
