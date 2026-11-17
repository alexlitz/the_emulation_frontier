# Vendored from the_emulation_frontier/elephantine/branches_attn.py by `python -m recipes.vendor`. Do not edit: edit elephantine/branches_attn.py.
"""branches_attn.py -- loaders for the round-3 ATTENTION features the universal host carries as branches.

The branches themselves live in elephantine/universal.py (UAttention.add_branches, each OFF by default and built
only when its UConfig flag is set, so a host without them keeps its state dict); this module holds the loaders
and setters that switch them on for a target, each checked in tests/test_attn_gaps.py against the target's own
code (a stock transformers class) or a labelled REFERENCE transcription of its remote code (read as text, never
run):

  feature (model_graph)  branch                                  loader / setter
  xsa / xsa_scaled       UAttention.xsa_c, xsa_eps               from_gptx2 (GPT-X2.5, GPT-S2), set_xsa (Limite)
  out_gate_post          UAttention.pg_proj, pg_c                from_mm200 (Ying-K1.5)
  dyn_mask               UAttention.dm_proj / dm_A / dm_keep     from_doge (+ ULayer.rgain: res_gain)
  swa_trunc              UAttention.trunc (mask_ext)             set_kv_trunc (Meditor-1 / -2 sliding layers)
  (chunked attention)    UAttention.chunk, qtemp_off (mask_ext)  branches_stream.from_llama4 (Llama-4 text)
  attn_out_sat           UAttention.sat (_dros)                  from_qwen3_dros, set_dros (Qwen3-D-ROS)
  doc_mask               UniversalElephantine.doc_eos            set_doc_mask (Ethoric, KLM: the attention part)
  moa                    UAttention.hr_proj / hr_k               from_jetmoe (JetMoE)
  onorm                  UAttention.o_coef / o_avg (existing)    set_onorm (Muse-Spark's weight-free per-head o_norm)
  vha                    UAttention.o_hmix (existing)            set_vha (ERNIE-5.5's VHAPostMix)
"""
from __future__ import annotations

__all__ = ["set_xsa", "from_gptx2", "from_mm200", "from_doge", "set_kv_trunc", "set_doc_mask", "set_dros",
           "from_qwen3_dros", "from_jetmoe", "set_onorm", "set_vha"]

import torch
import torch.nn.functional as F
from torch import nn

from tef_recipes.universal import (F64, UAttention, _W, _B, _attn_any, _cp, _head_from, _mha_from, _mlp_any,
                                   _norm_any, _qkv, _rope_of, _set_act, _shell, _sliding, _tile_kv, from_llama)


def _layers_attn(u):
    return [ul.attn for ul in u.layers]


# --- XSA ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def set_xsa(a: UAttention, strength=1.0, clamp=1e-6, add=0.0):
    """Exclusive self attention on one attention: o_h <- o_h - c_h (o_h . v_h) / (max(|v_h|^2, clamp) + add) v_h.
    strength: a scalar or (H,) (a KV group's heads share its value). GPT-X2.5 clamp 1e-6; GPT-S2 / Limite
    F.normalize(v, eps) is clamp eps^2; Grug-MoE (o . v) / (|v|^2 + 1e-6) is clamp 0, add 1e-6."""
    a.add_branches(xsa=True)
    a.xsa_c.copy_(torch.as_tensor(strength, dtype=F64).expand(a.H))
    a.xsa_eps.copy_(torch.tensor([clamp, add], dtype=F64))
    return a


@torch.no_grad()
def from_gptx2(target):
    """GPT-X2.5 (model_type gptx2, modeling_gptx2.py) and GPT-S2 (gpts2, modeling_gpts2.py; class GPTX3ForCausalLM):
    a pre-RMSNorm GQA stack, interleaved full RoPE (pairs (2i, 2i + 1)), SwiGLU w_gate / w_up / w_down, the token's
    own value projected out of each head's output (xsa_projection), tied or untied lm_head. GPT-X2's symplectic-RoPE
    variant (use_symplectic_rope: rope_warp, a content-warped phase) is a different file and is not this loader."""
    tc, tr = target.config, target.transformer
    assert not getattr(tc, "use_symplectic_rope", False), "GPT-X2 CIx (symplectic RoPE): rope_warp, not carried"
    H, Hk, hd, d = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.hidden_size
    blocks = list(tr["h"])
    u = _shell(tc, d=d, L=len(blocks), H=H, hd=hd, ff=tc.intermediate_size, xsa=True)
    _head_from(u, tr["wte"], tr["ln_f"], target.lm_head)
    inv = 1.0 / (tc.rope_theta ** (torch.arange(0, hd, 2, dtype=F64) / hd))
    clamp = 1e-6 if tc.model_type == "gptx2" else 1e-24          # gpts2: F.normalize(v, eps=1e-12)
    for ul, b in zip(u.layers, blocks):
        _norm_any(ul.norm_a, b.ln_1)
        _norm_any(ul.norm_f, b.ln_2)
        at = b.attn
        _attn_any(ul.attn, _W(at.q_proj), _W(at.k_proj), _W(at.v_proj), _W(at.o_proj), n_kv=Hk, scale=hd ** -0.5,
                  inv_freq=inv, layout="interleave")
        set_xsa(ul.attn, 1.0 if tc.xsa_projection else 0.0, clamp=clamp)
        f, ml = ul.ffn, b.mlp
        _cp(f.s_gate.weight, ml.w_gate.weight); _cp(f.s_up.weight, ml.w_up.weight); _cp(f.s_down.weight, ml.w_down.weight)
    return u


# --- MM200 (Ying-K1.5): the output gate after o_proj ---------------------------------------------------------

@torch.no_grad()
def from_mm200(target):
    """MM200 (Karasumikura/Ying-K1.5, modeling_mm200.py; the HF wrapper mm200_hf.py MM200ForCausalLM holds it as
    .model): zero-centred RMSNorms ((1 + w) x / rms), GQA with per-head RMS qk-norm BEFORE a partial rotate-half
    RoPE on the first rot_dim dims, 3:1 local (window W: 0 <= t - s <= W) / global layers, the sigmoid output gate
    AFTER o_proj (out_gate_post), SwiGLU, an untied or tied lm_head (MTP heads are training-only)."""
    m = getattr(target, "model", target)
    cfg = m.cfg
    H, Hk, hd, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    zc = cfg.use_zero_centered_norm
    blocks = list(m.blocks)

    class _TC:                                           # _shell reads vocab_size only
        vocab_size = cfg.vocab_size
    u = _shell(_TC, d=d, L=len(blocks), H=H, hd=hd, ff=cfg.intermediate_size, post_gate=cfg.use_gated_attention)
    _head_from(u, m.embed, m.norm, m.lm_head, norm_kind="rms1p" if zc else "rms")
    rot = max(2, int(hd * cfg.partial_rope_frac) // 2 * 2)
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2, dtype=F64) / rot))
    for ul, b in zip(u.layers, blocks):
        _norm_any(ul.norm_a, b.attn_norm, "rms1p" if zc else "rms")
        _norm_any(ul.norm_f, b.ffn_norm, "rms1p" if zc else "rms")
        at = b.attn
        _attn_any(ul.attn, _W(at.q_proj), _W(at.k_proj), _W(at.v_proj), _W(at.o_proj), n_kv=Hk,
                  scale=hd ** -0.5 * cfg.yarn_mscale, inv_freq=inv, layout="half",
                  window=cfg.sliding_window + 1 if at.is_local else None,
                  q_norm=getattr(at, "q_norm", None), k_norm=getattr(at, "k_norm", None))
        if cfg.use_gated_attention:
            _cp(ul.attn.pg_proj.weight, at.gate_proj.weight)
            ul.attn.pg_c.copy_(torch.tensor([0.0, 1.0]))
        f, ml = ul.ffn, b.ffn
        _cp(f.s_gate.weight, ml.gate.weight); _cp(f.s_up.weight, ml.up.weight); _cp(f.s_down.weight, ml.down.weight)
    return u


# --- Doge: the dynamic mask and the per-channel residual gains ------------------------------------------------

@torch.no_grad()
def from_doge(target):
    """Doge (DogeForCausalLM: transformers' stock class -- per-KV-head A / dt_proj, q / k RMS norms -- or the remote
    modeling_doge.py of SmallDoge/Doge-320M -- per-query-head A / dt_proj, no qk-norm): the dynamic mask (a per-key
    logit bias exp(A softplus(dt_proj(v))) and, past keep_window_size keys, only the top-biased keys visible), the
    per-channel residual gains h = g1 (.) x + attn, h' = g2 (.) h + mlp (input_residual / post_attention_residual),
    a dense SwiGLU (is_moe: CDMoE is not carried)."""
    tc, m = target.config, target.model
    assert not tc.is_moe, "Doge CDMoE (is_moe): not carried"
    H, Hk = tc.num_attention_heads, tc.num_key_value_heads
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // H
    u = _shell(tc, d=tc.hidden_size, L=tc.num_hidden_layers, H=H, hd=hd, ff=tc.intermediate_size, dyn_mask=True,
               res_gain=True)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    inv, att = _rope_of(m.rotary_emb)
    rep = H // Hk
    for ul, tl in zip(u.layers, m.layers):
        a, at = ul.attn, tl.self_attn
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        _attn_any(a, *_qkv(at), n_kv=Hk, scale=at.scaling, inv_freq=inv, att_scale=att,
                  window=getattr(tc, "sliding_window", None), q_norm=getattr(at, "q_norm", None),
                  k_norm=getattr(at, "k_norm", None))
        nA = at.A.numel()                               # Hk (stock) or H (Doge-320M's remote code)
        src = torch.arange(H) // (H // nA)              # host head -> its A / dt_proj row
        Wd = _W(at.dt_proj)                              # (nA, Hk * hd), read off the KV heads' values
        a.dm_proj.weight.zero_()
        for j in range(Hk):                              # the host's v tiles KV head j onto heads j*rep .. ; read
            c0 = j * rep * hd                            # it from the first of them
            a.dm_proj.weight[:, c0:c0 + hd] = Wd[src, j * hd:(j + 1) * hd]
        if at.dt_proj.bias is not None:
            a.dm_proj.bias.copy_(_B(at.dt_proj)[src])
        a.dm_A.copy_(at.A.detach().to(F64)[src])
        a.dm_c.fill_(1.0)
        a.dm_keep.fill_(float(tc.keep_window_size))
        ul.rgain.gx[0] = tl.input_residual.detach().to(F64)          # (URGain: x <- g_x x + f per sub-block)
        ul.rgain.gx[1] = tl.post_attention_residual.detach().to(F64)
        _mlp_any(ul.ffn, tl.mlp, tc.hidden_act)
    return u


# --- masks: Meditor's truncated K / V, Llama-4 chunks, documents ---------------------------------------------

@torch.no_grad()
def set_kv_trunc(a: UAttention, window):
    """Meditor-1 / -2 sliding layers (modeling_meditor2.py:171-188, modeling_meditor.py:164-185): in a prefill of
    T > W tokens K / V keep their last W rows and SDPA runs is_causal on the (T, W) scores, which is TOP-LEFT
    aligned: query t sees keys T - W .. T - W + t (later queries' keys included). T <= W: plain causal."""
    a.add_branches(mask_ext=True)
    a.trunc.fill_(float(window))
    return a


@torch.no_grad()
def set_doc_mask(u, eos_id):
    """Same-document attention (Ethoric doc_masking: attention.py dense_mask, modeling_ethoric.py document_ids;
    KLM segments): doc_t = #eos strictly before t (an eos closes its own document); query t sees key s only when
    doc_s = doc_t. The host's attention part of the feature (the conv / recurrent / n-gram resets are not)."""
    if not hasattr(u, "doc_eos"):
        u.cfg.doc_mask = True
        u.register_buffer("doc_eos", torch.tensor(-1.0, dtype=F64))
    u.doc_eos.fill_(float(eos_id))
    return u


# --- D-ROS: attention-output saturation ----------------------------------------------------------------------

_CURVES = {"tanh": 0, "sqrtsoftplus": 1, "hard": 2}
_POLS = {"positive_only": 0, "symmetric": 1}


def _band(b):
    b = b or {}
    return [float(b.get("threshold", 26.0)), float(b.get("knee", 6.0)), float(b.get("alpha_max", 0.35)),
            float(_CURVES.get(b.get("curve_type", "tanh"), 3)), float(_POLS.get(b.get("polarity", "positive_only"), 2))]


def _layer_gain(cfg, i):
    """Qwen3DROSAttention._compute_layer_gain (modeling_qwen3_eq.py:35-73)."""
    import math
    if not cfg.get("enabled", True):
        return 1.0
    eq = cfg.get("layer_eq", {})
    if not eq.get("enabled", False):
        return 1.0
    if "layer_gains_db" in eq:
        g, db = eq["layer_gains_db"], 0.0
        if isinstance(g, (list, tuple)):
            db = float(g[i]) if i < len(g) else 0.0
        elif isinstance(g, dict):
            db = float(g.get(str(i), g.get(i, 0.0)))
        return float(10.0 ** (db / 20.0))
    lo, mid, hi = eq.get("low_gain_db", 0.0), eq.get("mid_gain_db", 0.0), eq.get("high_gain_db", 0.0)
    c1, c2 = (float(v) for v in eq.get("crossover_layers", [10, 20]))
    l_ = float(i)
    if l_ <= c1:
        db = lo + (mid - lo) * 0.5 * (1.0 - math.cos(math.pi * l_ / max(1.0, c1)))
    elif l_ <= c2:
        db = mid + (hi - mid) * 0.5 * (1.0 - math.cos(math.pi * (l_ - c1) / max(1.0, c2 - c1)))
    else:
        db = hi
    return float(10.0 ** (db / 20.0))


@torch.no_grad()
def set_dros(a: UAttention, cfg, i, L):
    """Layer i of L under a d_ros_config (Qwen3-D-ROS modeling_qwen3_eq.py:22-186): the saturation on the middle
    layers int(L r0) <= i <= int(L r1) (multiband or the single band fallback; makeup; dry / wet), and the layer EQ
    gain on every layer."""
    a.add_branches(out_sat=True)
    r0, r1 = cfg.get("layer_ratio", (0.60, 0.90))
    on = cfg.get("enabled", True) and int(L * r0) <= i <= int(L * r1)
    mb = cfg.get("multiband", {})
    multi = bool(mb.get("enabled", False))
    a.sat.cfg.copy_(torch.tensor([float(on), float(cfg.get("dry_wet", 0.70)), float(cfg.get("makeup_gain", 1.0)),
                                  _layer_gain(cfg, i), float(multi), float(mb.get("crossover_kernel_low", 9)),
                                  float(mb.get("crossover_kernel_mid", 3))], dtype=F64))
    if multi:
        bd = mb.get("bands", {})
        rows = [_band(bd.get(k, {})) for k in ("low", "mid", "high")]
    else:
        rows = [_band(cfg.get("single_band_fallback", cfg))] * 3
    a.sat.bands.copy_(torch.tensor(rows, dtype=F64))
    return a


@torch.no_grad()
def from_qwen3_dros(target):
    """Qwen3-D-ROS (Qwen3EQForCausalLM, modeling_qwen3_eq.py): a stock Qwen3 whose attention outputs pass the D-ROS
    console (config.d_ros_config): from_llama + set_dros on every layer."""
    u = from_llama(target)
    cfg = getattr(target.config, "d_ros_config", None) or {}
    for i, a in enumerate(_layers_attn(u)):
        set_dros(a, cfg, i, len(u.layers))
    return u


# --- JetMoE: the mixture of attention heads --------------------------------------------------------------------

@torch.no_grad()
def from_jetmoe(target):
    """JetMoE (JetMoeForCausalLM): MoA attention -- E experts, each a (q, o) pair over one shared GQA K / V
    (kv_proj), top-k routed per token with a softmax over the k logits; the host gives expert e its own Hk heads
    (e Hk + j reads KV head j), scales them by the token's gate (0 if unrouted: hr_proj / hr_k) and sums every
    expert's o_proj, + the shared output bias -- and the MoE FFN (gated SiLU experts, the same top-k softmax, its
    output bias on the zero shared path)."""
    tc, m = target.config, target.model
    E, k, Hk, hd, d = tc.num_local_experts, tc.num_experts_per_tok, tc.num_key_value_heads, tc.kv_channels, tc.hidden_size
    ff = tc.intermediate_size
    u = _shell(tc, d=d, L=tc.num_hidden_layers, H=E * Hk, hd=hd, ff=1, experts=E, expert_ff=ff, top_k=k,
               head_route=E)
    _head_from(u, m.embed_tokens, m.norm, target.lm_head)
    for ul, tl in zip(u.layers, m.layers):
        a, at = ul.attn, tl.self_attention
        _norm_any(ul.norm_a, tl.input_layernorm)
        _norm_any(ul.norm_f, tl.post_attention_layernorm)
        ex = at.experts
        wq = ex.input_linear.weight.detach().to(F64).reshape(E * Hk * hd, d)       # (E, Hk hd, d)
        wk, wv = _W(at.kv_proj).chunk(2, 0)
        wo = ex.output_linear.weight.detach().to(F64).permute(1, 0, 2).reshape(d, E * Hk * hd)   # (E, d, Hk hd)
        rp = lambda w: w.view(Hk, hd, d).repeat(E, 1, 1).reshape(E * Hk * hd, d)  # slot e, head j <- KV head j
        _mha_from(a, wq, rp(wk), rp(wv), None, None, None, wo, ex.bias.detach(), hd ** -0.5,
                  inv_freq=at.rotary_emb.inv_freq)
        _cp(a.hr_proj.weight, ex.router.layer.weight)
        a.hr_k.fill_(float(k))
        f, mo = ul.ffn, tl.mlp
        wi = mo.input_linear.weight.detach().to(F64)                                # (E, 2 ff, d): [gate | up]
        _cp(f.w_gate, wi[:, :ff]); _cp(f.w_up, wi[:, ff:]); _cp(f.w_down, mo.output_linear.weight)
        _cp(f.router.weight, mo.router.layer.weight)
        f.renorm.fill_(1.0)                                                         # softmax over the top-k logits
        _cp(f.s_down.bias, mo.bias)
        _set_act(f, tc.activation_function, gated=True)
    return u


# --- output-side head maps on existing branches: Muse-Spark's o_norm, ERNIE-5.5's VHA --------------------------

@torch.no_grad()
def set_onorm(a: UAttention, eps):
    """A weight-free RMSNorm of each head's attention output before the gate and o_proj (Muse-Spark o_norm,
    modular_muse_spark.py): the host's output norm o_coef = [0, 1] with a block-diagonal per-head average o_avg."""
    H, hd = a.H, a.hd
    a.o_avg.copy_(torch.block_diag(*[torch.full((hd, hd), 1.0 / hd, dtype=F64)] * H))
    a.o_w.fill_(1.0)
    a.o_eps.fill_(float(eps))
    a.o_coef.copy_(torch.tensor([0.0, 1.0]))
    return a


@torch.no_grad()
def set_vha(a: UAttention, U, V):
    """ERNIE-5.5's VHAPostMix o[h] <- sum_g (I + V U^T)[h, g] o[g] (vLLM-ERNIE attention.py:110-146): the host's
    head mix o_hmix (o_new[g] = sum_h o[h] M[h, g]) is (I + V U^T)^T."""
    M = torch.eye(a.H, dtype=F64) + V.to(F64) @ U.to(F64).T
    a.o_hmix.copy_(M.T)
    return a
