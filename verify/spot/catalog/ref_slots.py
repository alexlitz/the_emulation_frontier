"""Slot transitions, zero edges and layer kinds on the reference decoder (verify/spot/catalog/_ref.py): the target
and the host are RefLMs that differ in the slot; the host's weights are written from the target's."""
from __future__ import annotations

import math

import torch

from verify.spot.catalog import compare, entry
from verify.spot.catalog._ref import F64, RefLM, fn, t5_bucket

REF = "synthetic, reference module"


def _t(seed=0, **cfg):
    return RefLM(**cfg).init(seed)


def _check(t, h, standard="exact", tol=1e-9, mask_t=None, mask_h=None, note="", new_tokens=32):
    return compare(fn(t, mask_t), fn(h, mask_h), t.c["V"], standard=standard, tol=tol, note=note,
                   new_tokens=new_tokens)


def _host(t, **cfg):
    """A host RefLM whose config is the target's with cfg changed, its parameters the target's (copied)."""
    c = dict(t.c)
    c.update(cfg)
    h = RefLM(**c)
    h.p = {k: (v.clone() if v is not None else None) for k, v in t.p.items()}
    return h


def _perm_qk(h, perm):
    """Permute the head-dim columns of q / k (rows of the projections, per head) by perm (target dim -> host dim)."""
    c = h.c
    for i in range(h.n_param_layers()):
        for key, n in (("q", c["H"]), ("k", c["KV"])):
            W = h.p[f"l{i}.{key}"]
            hd_t = len(perm)
            Wt = W.view(n, hd_t, -1)
            out = torch.zeros(n, c["hd"], W.shape[1], dtype=F64)
            out[:, perm] = Wt
            h.p[f"l{i}.{key}"] = out.reshape(n * c["hd"], -1)
            bk = f"l{i}.{key}b"
            if bk in h.p:
                b = torch.zeros(n, c["hd"], dtype=F64)
                b[:, perm] = h.p[bk].view(n, hd_t)
                h.p[bk] = b.reshape(-1)


def _widen_heads(h, t, perm, fold):
    """Host head_dim > target's: q / k rows placed by perm (q rows x fold for the scale), v / o on the first dims."""
    c, tc = h.c, t.c
    for i in range(h.n_param_layers()):
        pre = f"l{i}."
        for key, n in (("q", c["H"]), ("k", c["KV"]), ("v", c["KV"])):
            W = t.p[pre + key].view(n, tc["hd"], -1)
            out = torch.zeros(n, c["hd"], W.shape[-1], dtype=F64)
            out[:, perm if key != "v" else list(range(tc["hd"]))] = W * (fold if key == "q" else 1.0)
            h.p[pre + key] = out.reshape(n * c["hd"], -1)
            if pre + key + "b" in t.p:
                b = torch.zeros(n, c["hd"], dtype=F64)
                b[:, perm if key != "v" else list(range(tc["hd"]))] = t.p[pre + key + "b"].view(n, tc["hd"]) * (
                    fold if key == "q" else 1.0)
                h.p[pre + key + "b"] = b.reshape(-1)
        O = t.p[pre + "o"].view(-1, c["H"], tc["hd"])
        out = torch.zeros(O.shape[0], c["H"], c["hd"], dtype=F64)
        out[:, :, : tc["hd"]] = O
        h.p[pre + "o"] = out.reshape(O.shape[0], -1)


# ------------------------------------------------------------------ positions

@entry("slot:pos:nope->rope", instance=f"NoPE -> RoPE with inv_freq = 0 ({REF})")
def ap_pos_nope_rope():
    t = _t(pos="nope")
    h = _host(t, pos="rope", rot=t.c["hd"])
    h.inv_freq = torch.zeros(t.c["hd"] // 2, dtype=F64)
    return _check(t, h)


@entry("slot:pos:nope->partial_rope", instance=f"NoPE -> partial RoPE (rot hd / 2) with inv_freq = 0 ({REF})")
def ap_pos_nope_partial():
    t = _t(pos="nope")
    h = _host(t, pos="rope", rot=t.c["hd"] // 2)
    h.inv_freq = torch.zeros(t.c["hd"] // 4, dtype=F64)
    return _check(t, h)


@entry("slot:pos:nope->learned_abs", instance=f"NoPE -> learned absolute positions, table 0 ({REF})")
def ap_pos_nope_labs():
    t = _t(pos="nope")
    h = _host(t, pos="learned_abs")
    h.p["pos"] = torch.zeros(t.c["Tmax"], t.c["d"], dtype=F64)
    return _check(t, h)


@entry("slot:pos:nope->alibi", instance=f"NoPE -> ALiBi with the bias switched off (slopes 0) ({REF})")
def ap_pos_nope_alibi():
    t = _t(pos="nope")
    h = _host(t, pos="alibi")
    h.slopes = torch.zeros_like(h.slopes)
    return _check(t, h)


@entry("slot:pos:partial_rope->rope", instance=f"partial RoPE (rot 4 of 8) -> full RoPE: pass dims on zero-frequency "
                                              f"pairs of a settable table ({REF})")
def ap_pos_partial_full():
    t = _t(pos="rope", rot=4)
    hd, rot = t.c["hd"], 4
    h = _host(t, pos="rope", rot=hd)
    half_t, half_h, npass = rot // 2, hd // 2, (hd - rot) // 2
    perm = [e if e < half_t else half_h + e - half_t for e in range(rot)]
    perm += [half_t + q if q < npass else half_h + half_t + q - npass for q in range(hd - rot)]
    _perm_qk(h, perm)
    h.inv_freq = torch.cat([t.inv_freq, torch.zeros(half_h - half_t, dtype=F64)])
    return _check(t, h)


@entry("slot:pos:rope->partial_rope", "capacity:rot_gate", instance=f"RoPE (hd 8) -> partial RoPE (hd 16, rot 8): the "
                                                                    f"target's pairs on the host's rotated dims ({REF})")
def ap_pos_rope_partial():
    t = _t(pos="rope")
    h = _host(t, pos="rope", hd=16, rot=8)
    _widen_heads(h, t, list(range(8)), math.sqrt(16 / 8))
    h.inv_freq = t.inv_freq.clone()
    return _check(t, h)


@entry("capacity:rot_gate", "slot:pos:rope->partial_rope",
       instance="llama (RoPE) -> glm (partial RoPE host, stock): hf/partial_rope_host.glm_from_llama")
def ap_rot_gate_glm():
    from hf.partial_rope_host import _llama_target, glm_from_llama
    from verify.spot.catalog import hf_fn
    import elephantine.emulate as E
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=61, hidden_size=48, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=128, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    t = LlamaForCausalLM(cfg).to(F64).eval()
    with torch.no_grad():
        for p in t.parameters():
            p.add_(torch.randn_like(p) * 0.08)
    h = glm_from_llama(t, factor=0.5, hd=32).eval()
    with E._lift():
        return compare(hf_fn(t), hf_fn(h), 61, tol=1e-9)


@entry("slot:pos:alibi->additive_bias", instance=f"ALiBi -> an additive per-head bias table m_h (j - i) ({REF})")
def ap_pos_alibi_additive():
    t = _t(pos="alibi")
    h = _host(t, pos="additive_bias")
    T = t.c["Tmax"]
    i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    h.p["abias"] = t.slopes[:, None, None] * (j - i).to(F64)[None]
    return _check(t, h)


@entry("slot:pos:t5_relbias->additive_bias", instance=f"T5 bucketed relative bias -> an additive bias table ({REF})")
def ap_pos_t5_additive():
    t = _t(pos="t5")
    h = _host(t, pos="additive_bias")
    T = t.c["Tmax"]
    i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    h.p["abias"] = t.p["t5"][:, t5_bucket(i - j)]
    return _check(t, h)


def _alibi_rope(partial, omega=1e-6):
    """ALiBi target -> RoPE host (head_dim + 2): the target's dims on zero-frequency pairs, one spare pair at a small
    angle omega read through constant q / k components (the q / k biases): s c sin(omega (j - i)) -> m (j - i)."""
    t = _t(pos="alibi", qkv_bias=True)
    hd = t.c["hd"]
    H, KV = t.c["H"], t.c["KV"]
    hdh = hd + 2
    if partial:                                          # rot = 2: the spare pair is the rotated one
        h = _host(t, pos="rope", hd=hdh, rot=2)
        perm = [2 + e for e in range(hd)]
        sa, sb = 0, 1
        h.inv_freq = torch.tensor([omega], dtype=F64)
    else:
        h = _host(t, pos="rope", hd=hdh, rot=hdh)
        half_t, half_h = hd // 2, hdh // 2
        perm = [e if e < half_t else half_h + e - half_t for e in range(hd)]
        sa, sb = half_t, half_h + half_t
        h.inv_freq = torch.cat([torch.zeros(half_t, dtype=F64), torch.tensor([omega], dtype=F64)])
    sc_t, sc_h = 1 / math.sqrt(hd), 1 / math.sqrt(hdh)
    _widen_heads(h, t, perm, sc_t / sc_h)
    for i in range(h.n_param_layers()):
        pre = f"l{i}."
        qb = h.p[pre + "qb"].view(H, hdh)
        kb = h.p[pre + "kb"].view(KV, hdh)
        kb[:, sa] = 1.0
        g = H // KV
        for hh in range(H):
            qb[hh, sb] = t.slopes[hh] / (sc_h * omega)
        h.p[pre + "qb"], h.p[pre + "kb"] = qb.reshape(-1), kb.reshape(-1)
    return _check(t, h, standard="limit", tol=1e-7, note=f"small-angle limit, omega = {omega:g}")


@entry("slot:pos:alibi->rope", instance=f"ALiBi -> RoPE (head_dim + 2, q / k biases): the small-angle limit ({REF})",
       standard="limit")
def ap_pos_alibi_rope():
    return _alibi_rope(False)


@entry("slot:pos:alibi->partial_rope", instance=f"ALiBi -> partial RoPE (one rotated pair, q / k biases): the "
                                                f"small-angle limit ({REF})", standard="limit")
def ap_pos_alibi_partial():
    return _alibi_rope(True)


@entry("slot:pos:alibi->learned_abs", instance=f"ALiBi -> learned-abs (pre-LN, q / k biases): a key channel j read by "
                                              f"a constant query channel ({REF})",
       finding="the position table enters the residual stream, so a pre-norm's mean and variance see it: the ramp "
               "is not reproduced (round 2 pass 5 removes this POS_ORDER edge)")
def ap_pos_alibi_labs():
    """The relation's construction (attn/positional_cross.py: a per-position table b(j) on a reserved key channel read
    by a constant query channel), in a pre-LayerNorm block: the table enters the residual stream, so the norm's mean
    and variance see it. Expected to FAIL under a pre-norm (round 2 pass 5 removes the edge)."""
    t = _t(pos="alibi", norm="ln", qkv_bias=True)
    d, H, KV, hd = t.c["d"], t.c["H"], t.c["KV"], t.c["hd"]
    dh = d + KV
    h = RefLM(**{**t.c, "pos": "learned_abs", "d": dh, "hd": hd + 1}).init()
    h.p = {k: torch.zeros_like(v) for k, v in h.p.items()}
    pad = lambda W, rows, cols: torch.nn.functional.pad(W, (0, cols - W.shape[1], 0, rows - W.shape[0]))
    h.p["emb"][:, :d] = t.p["emb"]
    h.p["lm"][:, :d] = t.p["lm"]
    h.p["nf_w"][:d], h.p["nf_b"][:d] = t.p["nf_w"], t.p["nf_b"]
    T = t.c["Tmax"]
    for c_ in range(KV):
        h.p["pos"][:, d + c_] = torch.arange(T, dtype=F64)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        for n in ("n1", "n2"):
            h.p[pre + n + "_w"][:d], h.p[pre + n + "_b"][:d] = t.p[pre + n + "_w"], t.p[pre + n + "_b"]
            h.p[pre + n + "_w"][d:] = 1.0
        for key, n in (("q", H), ("k", KV), ("v", KV)):
            W = t.p[pre + key].view(n, hd, d)
            Wn = torch.zeros(n, hd + 1, dh, dtype=F64)
            Wn[:, :hd, :d] = W * (math.sqrt((hd + 1) / hd) if key == "q" else 1)
            b = torch.zeros(n, hd + 1, dtype=F64)
            b[:, :hd] = t.p[pre + key + "b"].view(n, hd) * (math.sqrt((hd + 1) / hd) if key == "q" else 1)
            if key == "k":
                for c_ in range(n):
                    Wn[c_, hd, d + c_] = 1.0                 # the key's extra dim reads the position channel
            if key == "q":
                for hh in range(H):
                    b[hh, hd] = t.slopes[hh] * math.sqrt(hd + 1)    # constant query: slope x j (x the scale)
            h.p[pre + key], h.p[pre + key + "b"] = Wn.reshape(n * (hd + 1), dh), b.reshape(-1)
        O = t.p[pre + "o"].view(d, H, hd)
        On = torch.zeros(dh, H, hd + 1, dtype=F64)
        On[:d, :, :hd] = O
        h.p[pre + "o"] = On.reshape(dh, -1)
        h.p[pre + "up"][:, :d] = t.p[pre + "up"]
        h.p[pre + "fgate"][:, :d] = t.p[pre + "fgate"]
        h.p[pre + "down"][:d] = t.p[pre + "down"]
    return _check(t, h, note="position channel inside the pre-norm")


# ------------------------------------------------------------------ norms

def _ln_fold(lm_bias):
    """LayerNorm target -> RMSNorm host: every writer into the residual stream centred (so the stream has mean 0 and
    LN = RMSNorm), each LN beta folded into its consumer's bias -- the final one into an lm_head bias."""
    t = _t(norm="ln", ffn="vanilla", act="gelu", qkv_bias=True, out_bias=True)
    h = _host(t, norm="rms", lm_bias=lm_bias)
    P = h.p
    cen = lambda W: W - W.mean(0, keepdim=True)
    P["emb"] = t.p["emb"] - t.p["emb"].mean(1, keepdim=True)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        P[pre + "o"], P[pre + "ob"] = cen(t.p[pre + "o"]), t.p[pre + "ob"] - t.p[pre + "ob"].mean()
        P[pre + "down"], P[pre + "downb"] = cen(t.p[pre + "down"]), t.p[pre + "downb"] - t.p[pre + "downb"].mean()
        b1, b2 = P.pop(pre + "n1_b"), P.pop(pre + "n2_b")
        for key in ("q", "k", "v"):
            P[pre + key + "b"] = t.p[pre + key + "b"] + t.p[pre + key] @ b1
        P[pre + "upb"] = t.p[pre + "upb"] + t.p[pre + "up"] @ b2
    bf = P.pop("nf_b")
    if lm_bias:
        P["lm_b"] = t.p["lm"] @ bf
    if t.c["tied"]:
        raise ValueError
    return _check(t, h, note="" if lm_bias else "host without an lm_head bias: the final beta is dropped")


@entry("slot:norm:LayerNorm->RMSNorm", instance=f"LayerNorm (biased, GeLU MLP) -> RMSNorm host with an lm_head bias "
                                               f"(Phi-style): writers centred, betas folded ({REF})")
def ap_norm_ln_fold():
    return _ln_fold(True)


@entry("slot:norm:LayerNorm->RMSNorm", instance=f"LayerNorm -> RMSNorm host WITHOUT an lm_head bias ({REF})",
       finding="the final LayerNorm's beta has no lm_head bias to fold into: exact only on hosts with one (round 2 "
               "pass 5: the LN fold without an lm_head bias is refuted as exact)")
def ap_norm_ln_fold_no_lm_bias():
    """The same fold on a host WITHOUT an lm_head bias (most RMSNorm hosts): the final LayerNorm's beta has nowhere to
    go. Not registered as a catalog instance: it is the finding behind the round 2 pass 5 'LN fold without an lm_head
    bias is refuted' rule."""
    return _ln_fold(False)


@entry("slot:norm:RMSNorm->LayerNorm", "capacity:norm_width",
       instance=f"RMSNorm -> LayerNorm at 2x width: the residual mirrored [h; -h] ({REF})")
def ap_norm_mirror():
    t = _t(norm="rms")
    d = t.c["d"]
    h = RefLM(**{**t.c, "norm": "ln", "d": 2 * d}).init()
    h.p = {k: torch.zeros_like(v) for k, v in h.p.items()}
    P = h.p
    mir = lambda W: torch.cat([W, -W], 0)
    P["emb"] = torch.cat([t.p["emb"], -t.p["emb"]], 1)
    P["lm"][:, :d] = t.p["lm"]
    P["nf_w"] = torch.cat([t.p["nf_w"], t.p["nf_w"]])
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        for n in ("n1", "n2"):
            P[pre + n + "_w"] = torch.cat([t.p[pre + n + "_w"]] * 2)
        for key in ("q", "k", "v", "up", "fgate"):
            P[pre + key][:, :d] = t.p[pre + key]
        P[pre + "o"] = mir(t.p[pre + "o"])
        P[pre + "down"] = mir(t.p[pre + "down"])
    return _check(t, h)


# ------------------------------------------------------------------ qk-norm nestings

def _qk(tv, hv, mapper):
    t = _t(qknorm=tv)
    h = _host(t, qknorm=hv)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        for k_ in ("qn", "kn"):
            n = t.c["H"] if k_ == "qn" else t.c["KV"]
            for key in list(h.p):
                if key.startswith(pre + k_):
                    del h.p[key]
            h.p.update({pre + k_ + s: v for s, v in mapper(t.p.get(pre + k_), t.p.get(pre + k_ + "_b"), n,
                                                           t.c["hd"]).items()})
    return _check(t, h)


@entry("slot:qknorm:head->head_distinct", instance=f"per-head qk RMSNorm (one gain) -> per-head gains, all equal ({REF})")
def ap_qk_head_distinct():
    return _qk("head", "head_distinct", lambda w, b, n, hd: {"": w.expand(n, hd).clone()})


@entry("slot:qknorm:head_free->head", instance=f"weight-free qk-norm -> gained per-head qk-norm, gain 1 ({REF})")
def ap_qk_free_head():
    return _qk("head_free", "head", lambda w, b, n, hd: {"": torch.ones(hd, dtype=F64)})


@entry("slot:qknorm:head_free->head_distinct", instance=f"weight-free qk-norm -> per-head gains 1 ({REF})")
def ap_qk_free_distinct():
    return _qk("head_free", "head_distinct", lambda w, b, n, hd: {"": torch.ones(n, hd, dtype=F64)})


@entry("slot:qknorm:head_free->head_post", instance=f"weight-free qk-norm (before RoPE) -> gained qk-norm after RoPE, "
                                                     f"gain 1: the rms is rotation-invariant ({REF})")
def ap_qk_free_post():
    return _qk("head_free", "head_post", lambda w, b, n, hd: {"": torch.ones(hd, dtype=F64)})


@entry("slot:qknorm:head_ln->head_ln_bias", instance=f"per-head LayerNorm qk-norm -> with biases, bias 0 ({REF})")
def ap_qk_ln_bias():
    return _qk("head_ln", "head_ln_bias", lambda w, b, n, hd: {"": w.clone(), "_b": torch.zeros(n, hd, dtype=F64)})


@entry("slot:qknorm:ln_shared_bias->head_ln_bias", instance=f"shared-gain LayerNorm qk-norm -> per-head gains and biases, "
                                                             f"repeated ({REF})")
def ap_qk_shared_ln():
    return _qk("ln_shared_bias", "head_ln_bias", lambda w, b, n, hd: {"": w.expand(n, hd).clone(),
                                                                       "_b": b.expand(n, hd).clone()})


# ------------------------------------------------------------------ softcaps (regime edges: high agreement)

@entry("slot:softcap:final->no", instance=f"final-logit cap 10 -> no cap: argmax unchanged ({REF})", standard="approx")
def ap_cap_final_no():
    t = _t(softcap_final=10.0)
    return _check(t, _host(t, softcap_final=None), standard="approx")


@entry("slot:softcap:no->final", instance=f"no cap -> final-logit cap set large (1e4) ({REF})", standard="approx")
def ap_cap_no_final():
    t = _t()
    return _check(t, _host(t, softcap_final=1e4), standard="approx")


@entry("slot:softcap:no->attn", instance=f"no cap -> attention cap set large (1e4) ({REF})", standard="approx")
def ap_cap_no_attn():
    t = _t()
    return _check(t, _host(t, softcap_attn=1e4), standard="approx")


@entry("slot:softcap:final->attn", instance=f"final cap 30 -> attention cap 1e4 + the same final cap ({REF})",
       standard="approx")
def ap_cap_final_attn():
    t = _t(softcap_final=30.0)
    return _check(t, _host(t, softcap_attn=1e4), standard="approx")


# ------------------------------------------------------------------ sinks, masks, MTP, loops, blocks, mixers

@entry("slot:sink:no->yes", instance=f"plain attention -> learned per-head sinks at logit -1e4 (exp underflows to 0) "
                                     f"({REF})")
def ap_sink_add():
    t = _t()
    h = _host(t, sink=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.sink"] = torch.full((t.c["H"],), -1e4, dtype=F64)
    return _check(t, h)


@entry("slot:gen:ar->diffusion", instance=f"causal LM -> bidirectional (diffusion) host given the causal mask ({REF})")
def ap_gen_ar_diffusion():
    t = _t(mask="causal")
    return _check(t, _host(t, mask="full"), mask_h="causal")


@entry("slot:gen:ar->block", instance=f"causal LM -> block-causal host given block size 1 ({REF})")
def ap_gen_ar_block():
    t = _t(mask="causal")
    return _check(t, _host(t, mask="block:4"), mask_h="block:1")


@entry("slot:gen:block->diffusion", instance=f"block-causal (4) -> bidirectional host given the block mask ({REF})")
def ap_gen_block_diffusion():
    t = _t(mask="block:4")
    return _check(t, _host(t, mask="full"), mask_h="block:4")


@entry("slot:gen:diffusion->block", instance=f"bidirectional -> block-causal host given an all-ones mask ({REF})")
def ap_gen_diffusion_block():
    t = _t(mask="full")
    return _check(t, _host(t, mask="block:4"), mask_h="full")


@entry("slot:mtp:no->yes", instance=f"no MTP head -> a host with an MTP head (the main logits do not read it) ({REF})")
def ap_mtp_add():
    t = _t()
    h = _host(t, mtp=True)
    h.p["mtp"] = torch.randn(t.c["d"], 2 * t.c["d"], dtype=F64)
    return _check(t, h)


@entry("slot:mtp:yes->no", instance=f"an MTP head -> no MTP head (dropped) ({REF})")
def ap_mtp_drop():
    t = _t(mtp=True)
    h = _host(t, mtp=False)
    h.p.pop("mtp")
    return _check(t, h)


@entry("slot:loop:looped->untied", instance=f"a looped block (2 layers x 2) -> 4 untied layers (unrolled) ({REF})")
def ap_loop_unroll():
    t = _t(loop=(2, 2), L=4)
    h = _host(t, loop=None, L=4)
    for i in range(4):
        for k in [k for k in t.p if k.startswith(f"l{i % 2}.")]:
            h.p[f"l{i}." + k.split(".", 1)[1]] = t.p[k].clone()
    return _check(t, h)


@entry("slot:loop:untied->looped", instance=f"2 untied layers -> a looped host (block of 2) with its loop count set to "
                                            f"1 ({REF}; see the finding: a fixed loop count cannot)")
def ap_loop_host():
    t = _t(L=2)
    h = _host(t, loop=(2, 1))
    return _check(t, h)


def _par_host(t):
    """Sequential target -> parallel host at 2x depth: host layer 2i the attention (FFN zero), 2i + 1 the FFN
    (attention output zero), each reading its one norm with the target's gains for that sublayer."""
    L = t.c["L"]
    h = RefLM(**{**t.c, "block": "parallel", "L": 2 * L}).init()
    h.p = {k: torch.zeros_like(v) for k, v in h.p.items()}
    for k in ("emb", "lm", "nf_w"):
        h.p[k] = t.p[k].clone()
    for i in range(L):
        a, f = f"l{2 * i}.", f"l{2 * i + 1}."
        for key in ("q", "k", "v", "o"):
            h.p[a + key] = t.p[f"l{i}." + key].clone()
        h.p[a + "n1_w"] = t.p[f"l{i}.n1_w"].clone()
        h.p[a + "n2_w"] = torch.ones_like(h.p[a + "n2_w"])
        for key in ("up", "fgate", "down"):
            h.p[f + key] = t.p[f"l{i}." + key].clone()
        h.p[f + "n1_w"] = t.p[f"l{i}.n2_w"].clone()
        h.p[f + "n2_w"] = torch.ones_like(h.p[f + "n2_w"])
        h.p[f + "q"] = torch.zeros_like(h.p[f + "q"])            # attention output 0 (o = 0)
    return h


@entry("slot:block:sequential->parallel", "capacity:moe_depth",
       instance=f"sequential block -> parallel host at 2x depth (attention and FFN in alternate layers) ({REF})")
def ap_block_seq_par():
    t = _t()
    return _check(t, _par_host(t))


@entry("slot:mixer:softmax->hybrid", "zero:mixer:softmax<-hybrid",
       instance=f"softmax (2 layers) -> hybrid host (attn, rec, attn, rec): recurrent layers zeroed ({REF})")
def ap_mixer_hybrid():
    t = _t(L=2)
    h = RefLM(**{**t.c, "L": 4, "kinds": ["attn", "rec", "attn", "rec"]}).init(1)
    for k in ("emb", "lm", "nf_w"):
        h.p[k] = t.p[k].clone()
    for i in range(2):
        for k in [k for k in t.p if k.startswith(f"l{i}.")]:
            h.p[f"l{2 * i}." + k.split(".", 1)[1]] = t.p[k].clone()
        r = f"l{2 * i + 1}."
        h.p[r + "rout"].zero_()
        h.p[r + "down"].zero_()
    return _check(t, h)


@entry("zero:crosslayer_kv:no<-yes", "slot:crosslayer_kv:no->yes", instance=f"own-KV target (2 layers) -> CLA host (layers 1, 3 reuse the previous "
                                              f"layer's K / V): the sharing layers zeroed ({REF})")
def ap_zero_cla():
    t = _t(L=2)
    h = RefLM(**{**t.c, "L": 4, "cla": [1, 3]}).init(1)
    for k in ("emb", "lm", "nf_w"):
        h.p[k] = t.p[k].clone()
    for i in range(2):
        for k in [k for k in t.p if k.startswith(f"l{i}.")]:
            h.p[f"l{2 * i}." + k.split(".", 1)[1]] = t.p[k].clone()
        r = f"l{2 * i + 1}."
        h.p[r + "o"].zero_()
        h.p[r + "down"].zero_()
    return _check(t, h)


@entry("capacity:layer_kind:S<-S", instance=f"CLA target (layer 1 reads layer 0's K / V) -> a deeper CLA host "
                                            f"(layers 1, 3 share), the extra layers identity ({REF})")
def ap_kind_cla():
    t = _t(L=2, cla=[1])
    h = RefLM(**{**t.c, "L": 4, "cla": [1, 3]}).init(1)
    for k in ("emb", "lm", "nf_w"):
        h.p[k] = t.p[k].clone()
    for i in range(2):
        for k in [k for k in t.p if k.startswith(f"l{i}.")]:
            h.p[f"l{i}." + k.split(".", 1)[1]] = t.p[k].clone()
    for r in ("l2.", "l3."):
        h.p[r + "o"].zero_()
        h.p[r + "down"].zero_()
    return _check(t, h)


@entry("capacity:layer_kind:R<-R", instance=f"hybrid target (attn, rec) -> hybrid host (attn, rec, attn, rec), recurrent "
                                            f"layer on a recurrent layer, the extra ones identity ({REF})")
def ap_kind_rec():
    t = _t(L=2, kinds=["attn", "rec"])
    h = RefLM(**{**t.c, "L": 4, "kinds": ["attn", "rec", "attn", "rec"]}).init(1)
    for k in ("emb", "lm", "nf_w"):
        h.p[k] = t.p[k].clone()
    for i in range(2):
        for k in [k for k in t.p if k.startswith(f"l{i}.")]:
            h.p[f"l{i}." + k.split(".", 1)[1]] = t.p[k].clone()
    h.p["l2.o"].zero_()
    h.p["l2.down"].zero_()
    h.p["l3.rout"].zero_()
    h.p["l3.down"].zero_()
    return _check(t, h)
