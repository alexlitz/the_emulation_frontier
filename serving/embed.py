"""Embed a target checkpoint into a BIGGER host checkpoint of a served architecture, so the host's
stock serving code (transformers / vLLM, no code edits) computes the target's logits.

This is the constructive side of "B emulates A" for serving: a provider that runs a fixed model class
(e.g. a Featherless class) can then serve A at B's price. Llama-family dense decoders (llama,
mistral, qwen2, qwen3) and their MoE versions (qwen2_moe, qwen3_moe) are handled. A's shapes need
only be <= B's:

  hidden    zero-pad. RMSNorm averages over all h_B dims, so the norm weights are scaled by
            sqrt(h_A/h_B) and eps_B = eps_A h_A/h_B: rms_B = sqrt(h_A/h_B) rms_A exactly.
  heads     each target KV group is copied into ceil(g_A/g_B) host KV groups (g = queries per KV
            head), and its query heads fill their slots. Unused heads get q = 0 and zero o_proj columns.
  head_dim  hd_B = k hd_A. Target dim i of the rotate-half layout goes to host dim k i (first half)
            or hd_B/2 + k j (second half). Host frequency theta^(-2 k j/hd_B) = theta^(-2j/hd_A),
            so RoPE (and llama3 frequency scaling, a function of frequency) agrees exactly. The
            1/sqrt(hd) softmax scale is folded into q (or into the qwen3 q/k-norm weights).
  ffn       zero-pad: SiLU(0) * 0 = 0, and the down_proj columns are zero.
  layers    extra layers are identities (zero o_proj / down_proj). They may sit anywhere: target
            layers map in order onto compatible host layers (MoE hosts with dense layers).
  vocab     V_B > V_A: padded rows would give logit 0, leaking mass onto non-tokens. One free hidden
            dim p carries a constant c on every real token (c^2 + h_B eps_B = h_A eps_A, so every
            RMSNorm stays exact), and the padded lm_head rows read -M along p: logits ~ -inf. Needs
            h_B > h_A. With tied embeddings the real tokens' logits all shift by the same constant,
            which softmax ignores.

MoE (routing: softmax over ALL E_B router logits in fp32, top-k, optional renormalization):

  experts   E_A <= E_B and moe_intermediate f_A <= f_B (zero-padded). Dummy experts (beyond the
            real ones) are zero AND must never receive routing mass, else they would dilute the
            softmax (norm_topk_prob=False) or the renormalized top-k: their router rows read -M on
            the free dim p (post_attention_layernorm gain 1/c there, so p enters the router as
            1/rms; the experts' and shared expert's p columns are zero), giving exp(-M/rms) = 0
            exactly in fp32. Real router rows are copied (zero on p), so softmax over E_B equals the
            target's softmax over E_A: exact for norm_topk_prob both ways. Needs h_B > h_A.
  top-k     k_B = r k_A (r integer) is exact IF THE TARGET'S TOP-k IS STRICT (Lean moeR_copies_topk,
            moeR_dup_renorm): every real expert is stored r times with identical router rows, the r
            copies are tied in probability, so when every selected expert beats every unselected one
            the host's top-r k_A set is exactly the copies of the target's top-k_A, each with weight
            p/r (renormalization divides by the same total). At a TIE across the target's top-k
            boundary the host's top-(r k_A) can take copies of more than k_A distinct experts, a
            mixture no target tie-break gives (moeR_tie_split). Tied router rows make that happen on
            every token that reaches the boundary, and they are not exotic: a zero router (router
            steering, moe/zero_router) ties every expert. So for r > 1 (and k_A < E_A) embed_state
            refuses a target layer whose router has two identical rows (ties="warn" embeds it anyway,
            with a warning). Distinct rows can still tie (or nearly, below the serving dtype's
            resolution) on particular inputs; those tokens are not exact. Any other k_B is not: with k_B < k_A real mass is dropped; with
            k_B > k_A and not a multiple, some pick beyond the target's set must be a real expert with
            positive weight (dummies have prob 0, so they rank below every real expert and are never
            picked while real ones remain) or an unequal share of a replicated one, and renormalizing
            cannot remove a positive-weight term. norm_topk_prob must match: renormalizing divides by
            the picked mass S_k, which a linear expert cannot undo or supply.
  shared    qwen2_moe's sigmoid-gated shared expert is zero-padded (f_s,A <= f_s,B); a qwen2_moe
            target needs a qwen2_moe host. Hosts' unused shared experts are zero.
  dense     dense target layer -> host MoE layer ("router steering"): its FFN of width f is split
  -> MoE    into m = ceil(f/f_B) (rounded up to a power of 2 if that still fits k_B) slices held by
            m experts whose router rows are all zero (equal logits); every other expert reads -M on p,
            so softmax gives exactly 1/m to each of the m and 0 to the rest (whatever the
            renormalization: the m weights already sum to 1). Their down_proj is scaled by m. Exact
            for power-of-2 m; otherwise the 1/m rounding of the routing weight (2^-9 in bf16) remains.
  layers    decoder_sparse_step / mlp_only_layers give the dense/MoE pattern; a dense target layer
            can use a dense or an MoE host layer, an MoE one only an MoE host layer.

The M magnitude sits in the router weights (vLLM's Qwen-MoE and Qwen's FP8 checkpoints keep the
router unquantized; a per-row quantizer keeps a one-entry row exact anyway). The p activation itself is
O(1/rms), so dynamic activation quantization of the expert inputs is unaffected. The 1e8 magnitudes
overflow fp16 (max 65504; Featherless serves models under 5B in fp16): embed_state(..., fp16=True)
uses fp16-safe magnitudes instead (suppression ~ c 1e4/rms: padded-token log-probs <= -600 on
AMD-Llama-135m inside Qwen1.5-1.8B shapes).

The host CONFIG must also match on non-shape values (rope_theta / rope_scaling, the eps above,
sliding window); plan() lists them. qwen3's q/k-norms share eps with the layer norms, so when
hd_B/hd_A != h_B/h_A (or with the free dim) they are exact only up to eps (~1e-6 relative).

Featherless MoE classes (2026-10-19, `python -m serving.arbitrage classes moe`; uploads pay 4x the
class minimum, blended 3:1 $/Mtok). One hit: qwen3moe-235b (4.85, 1 listed) -> qwen3-235b (3.19),
34% cheaper, identical shapes, needs the target's rope_theta 5e6 (host 1e6). Nothing else:
qwen3moe-30b (0.82) is already the cheapest Qwen3-MoE class; 235B (94 layers) doesn't fit
Coder-480B (62); Coder-480B (h 6144, 160 experts) fits nothing; and dense Qwen3 uploads (8 KV heads)
fit only Coder-480B (the others have 4), which costs more than every dense Qwen3 class.
qwen3moe-80b is Qwen3-Next and qwen3_5moe is Qwen3.5 (gated-delta hybrids, not handled); no Featherless
class is qwen2_moe; qwen3vlmoe-30b's text tower (0.95) would fit qwen3moe-30b only for text-only use
(not handled: Qwen3-VL's MRoPE).
"""
from __future__ import annotations

import math
import warnings

import torch

FAMILY = {"llama": "llama", "mistral": "llama", "qwen2": "qwen2", "qwen3": "qwen3",
          "qwen2_moe": "qwen2", "qwen3_moe": "qwen3"}          # attention family
MOE = ("qwen2_moe", "qwen3_moe")
VOCAB_C2 = 0.5          # c^2 = VOCAB_C2 * eps_A * h_A when the free dim is used; any split with
                        # c^2 + h_B eps_B = h_A eps_A keeps every RMSNorm exact, and eps_B > 0 keeps
                        # qwen3's q/k-norm finite on the all-zero unused heads
# Featherless (and most fp8 serving) quantizes the linear weights but keeps norms, embeddings and
# usually lm_head in bf16; still, the magnitudes go where quantization can't reach them:
VOCAB_GAIN = 1e8        # untied host: final-norm gain on the free dim; padded lm_head rows read -1 there
VOCAB_ROW_TIED = 1e4    # tied host: the padded EMBEDDING rows (= head rows, unquantized) read -1e4; the
                        # gain stays 1 so the real tokens' common logit shift c^2/rms stays negligible
MOE_ROW = 1e8           # suppressed router rows read -MOE_ROW on p; p enters the router as 1/rms_B, so
                        # their logits are <= -1e4 for any rms_B <= 1e4: exp() is exactly 0 in fp32


def _g(cfg, k, default=None):
    v = cfg.get(k, default) if isinstance(cfg, dict) else getattr(cfg, k, default)
    return default if v is None else v


def dims(cfg):
    h, H = _g(cfg, "hidden_size"), _g(cfg, "num_attention_heads")
    mt = _g(cfg, "model_type", "llama")
    L = _g(cfg, "num_hidden_layers")
    E = _g(cfg, "num_experts", 0) if mt in MOE else 0
    only, step = set(_g(cfg, "mlp_only_layers", []) or []), _g(cfg, "decoder_sparse_step", 1)
    sparse = tuple(E > 0 and i not in only and (i + 1) % step == 0 for i in range(L))
    return dict(h=h, H=H, kv=_g(cfg, "num_key_value_heads", H), hd=_g(cfg, "head_dim", h // H),
                ff=_g(cfg, "intermediate_size"), L=L, V=_g(cfg, "vocab_size"),
                tied=bool(_g(cfg, "tie_word_embeddings", False)), eps=_g(cfg, "rms_norm_eps", 1e-6),
                fam=FAMILY.get(mt), mt=mt,
                bias=bool(_g(cfg, "attention_bias", _g(cfg, "qkv_bias", mt in ("qwen2", "qwen2_moe")))),
                E=E, k=_g(cfg, "num_experts_per_tok", 0) if E else 0, fm=_g(cfg, "moe_intermediate_size", 0) if E else 0,
                norm=bool(_g(cfg, "norm_topk_prob", False)),
                fs=_g(cfg, "shared_expert_intermediate_size", 0) if mt == "qwen2_moe" else 0, sparse=sparse)


def resolved(cfg, keys=("rope_theta", "rope_scaling", "sliding_window", "rms_norm_eps")):
    """Effective values of non-shape config keys, with transformers' defaults for whatever a config
    dict leaves out (old Llama-2 configs omit rope_theta; Qwen2 keeps sliding_window but disables it)."""
    d = cfg if isinstance(cfg, dict) else cfg.to_dict()
    try:
        from transformers import AutoConfig
        c = AutoConfig.for_model(d.get("model_type", "llama"),
                                 **{k: v for k, v in d.items() if k not in ("model_type", "transformers_version")})
    except Exception:
        c = None
    out = {k: (getattr(c, k, None) if c is not None else d.get(k)) for k in keys}
    if c is not None and getattr(c, "rope_parameters", None) and out["rope_theta"] is None:   # transformers 5
        out["rope_theta"] = c.rope_parameters.get("rope_theta")
    if d.get("model_type") in ("qwen2", "qwen3") + MOE and not d.get("use_sliding_window", False):
        out["sliding_window"] = None
    rs = out["rope_scaling"]
    if isinstance(rs, dict) and rs.get("rope_type", rs.get("type")) in (None, "default"):
        out["rope_scaling"] = None
    return out


def _pow2(n):
    return n > 0 and n & (n - 1) == 0


def _steer_m(ff, b):
    """Experts a dense FFN of width ff is split over inside one of b's MoE layers (None: doesn't fit)."""
    need, cap = math.ceil(ff / b["fm"]), min(b["k"], b["E"])
    if need > cap:
        return None
    m = 1 << (need - 1).bit_length()
    return m if m <= cap else need


def _layers(a, b):
    """Order-preserving map target layer -> host layer, plus per-target-layer routing info:
    ('dense',) | ('steer', m) | ('moe', r). Extra host layers become identities."""
    rep = b["k"] // a["k"] if a["E"] and b["E"] and b["k"] % a["k"] == 0 else None

    def kind(i, j):
        if a["sparse"][i]:
            return ("moe", rep) if b["sparse"][j] and rep else None
        if not b["sparse"][j]:
            return ("dense",) if a["ff"] <= b["ff"] else None
        m = _steer_m(a["ff"], b)
        return ("steer", m) if m else None

    out, j = [], 0
    for i in range(a["L"]):
        while j < b["L"] and kind(i, j) is None:
            j += 1
        if j == b["L"]:
            return None
        out.append((j, kind(i, j)))
        j += 1
    return out


def plan(tcfg, hcfg):
    """{'ok', 'why': [...], 'exact': 'exact' | 'eps' | 'approx', 'config': {...host config values to set},
    'layers': [(host layer, route) per target layer], 'free': uses the free hidden dim}."""
    a, b = dims(tcfg), dims(hcfg)
    why = []
    if a["fam"] is None or b["fam"] is None:
        why.append("family not handled")
    elif a["fam"] != b["fam"] and not (a["fam"] == "llama" and b["fam"] == "qwen2"):
        why.append(f"{a['fam']} into {b['fam']}")       # llama -> qwen2: zero q/k/v biases
    for k in ("h", "H", "L", "V"):
        if b[k] < a[k]:
            why.append(f"{k} {a[k]} > {b[k]}")
    if b["hd"] % a["hd"]:
        why.append(f"head_dim {b['hd']} not a multiple of {a['hd']}")
    gA, gB = a["H"] // a["kv"], b["H"] // b["kv"]
    if a["kv"] * math.ceil(gA / gB) > b["kv"]:
        why.append(f"kv groups: need {a['kv'] * math.ceil(gA / gB)} > {b['kv']}")
    if math.ceil(gA / gB) * gB * a["kv"] > b["H"]:
        why.append("query slots")
    if b["V"] > a["V"] and b["h"] == a["h"]:
        why.append("vocab pad needs a free hidden dim")
    if b["tied"] and not a["tied"]:
        why.append("host ties lm_head to the embeddings, target doesn't")
    if a["E"]:                                          # MoE target: global routing conditions
        if not b["E"]:
            why.append("MoE target needs an MoE host")
        else:
            if b["k"] % a["k"]:
                why.append(f"top-k {b['k']} not a multiple of {a['k']}")
            elif b["E"] < (b["k"] // a["k"]) * a["E"]:
                why.append(f"experts: need {(b['k'] // a['k']) * a['E']} > {b['E']}")
            if a["norm"] != b["norm"]:
                why.append(f"norm_topk_prob {a['norm']} vs {b['norm']}")
            if a["fm"] > b["fm"]:
                why.append(f"moe_intermediate {a['fm']} > {b['fm']}")
            if a["fs"] > b["fs"]:
                why.append(f"shared expert {a['fs']} > {b['fs']}")
    lm = None if why else _layers(a, b)
    if not why and lm is None:
        why.append(f"layers: no in-order map of target's {sum(a['sparse'])} MoE / {a['L'] - sum(a['sparse'])} "
                   f"dense layers onto host's {sum(b['sparse'])} / {b['L'] - sum(b['sparse'])} (FFN / expert width)")
    vocab_trick = b["V"] > a["V"]
    routes = [r for _, r in lm or []]
    suppress = any((r[0] == "moe" and b["E"] > r[1] * a["E"]) or (r[0] == "steer" and b["E"] > r[1]) for r in routes)
    if suppress and b["h"] == a["h"]:
        why.append("dummy experts need a free hidden dim")
    free = vocab_trick or suppress
    eps_b = a["eps"] * a["h"] * ((1 - VOCAB_C2) if free else 1.0) / b["h"]
    cfg = {"rms_norm_eps": eps_b, "rope_theta": _g(tcfg, "rope_theta"), "rope_scaling": _g(tcfg, "rope_scaling"),
           "sliding_window": _g(tcfg, "sliding_window"), "max_position_embeddings": _g(tcfg, "max_position_embeddings")}
    exact = "exact"
    if a["fam"] == "qwen3" and (free or b["hd"] * a["h"] != a["hd"] * b["h"]):
        exact = "eps"
    if any(r[0] in ("steer", "moe") and not _pow2(r[1]) for r in routes):
        exact = "eps"           # 1/m (or p/r) routing weights round in the serving dtype
    if vocab_trick and b["tied"]:
        exact = "approx"        # suppression ~ 1e4 c / rms: strong for ordinary activations, not unbounded
    # config values that must differ from the host's own (if a provider forces class defaults, the
    # eps rescale alone costs ~eps relative error, far below fp8 serving noise; rope/window are not free)
    rt, rh = resolved(tcfg), resolved(hcfg)
    rt["rms_norm_eps"] = eps_b
    cfg.update({k: rt[k] for k in ("rope_theta", "rope_scaling", "sliding_window")})
    diff = {k: (rt[k], rh[k]) for k in rt if rt[k] != rh[k]}
    return {"ok": not why, "why": why, "exact": exact, "config": cfg, "config_diff": diff,
            "needs_config": sorted(k for k in diff if k != "rms_norm_eps"), "layers": lm, "free": free}


def _hmap(hdA, hdB):
    k, half = hdB // hdA, hdA // 2
    return torch.tensor([i * k for i in range(half)] + [hdB // 2 + j * k for j in range(half)])


FP16_SAFE = 1e4        # VOCAB_GAIN and MOE_ROW when the host will be served in fp16


def tied_router_rows(W):
    """Groups (lists of expert indices, size >= 2) of identical rows of a router weight W [E, h]."""
    groups = {}
    for e, row in enumerate(W.detach().cpu()):
        groups.setdefault(tuple(row.tolist()), []).append(e)
    return [g for g in groups.values() if len(g) > 1]


def embed_state(tsd, tcfg, hcfg, dtype=None, fp16=False, ties="refuse"):
    """Host state dict (HF names) whose forward under hcfg (with plan()'s config values) equals the
    target's. tsd: target state dict. fp16=True keeps every constant inside fp16 range.
    ties: what to do when a duplicated ('moe', r > 1) layer's target router has identical rows, which
    breaks the strict top-k that k_B = r k_A needs (see the module docstring): "refuse" (ValueError)
    or "warn" (embed anyway)."""
    gain, moe_row = (FP16_SAFE, FP16_SAFE) if fp16 else (VOCAB_GAIN, MOE_ROW)
    p = plan(tcfg, hcfg)
    if not p["ok"]:
        raise ValueError(p["why"])
    a, b = dims(tcfg), dims(hcfg)
    dt = dtype or next(iter(tsd.values())).dtype
    z = lambda *s: torch.zeros(*s, dtype=dt)
    hA, hB, hdA, hdB = a["h"], b["h"], a["hd"], b["hd"]
    k = hdB // hdA
    mu = _hmap(hdA, hdB)
    vocab_trick = b["V"] > a["V"]
    eps_b = p["config"]["rms_norm_eps"]
    c = math.sqrt(VOCAB_C2 * a["eps"] * hA) if p["free"] else 0.0
    nscale = math.sqrt(hA / hB)                         # rms_B = sqrt(hA/hB) rms_A (eps_B chosen to match)

    def norm_w(w):
        out = z(hB)
        out[:hA] = w * nscale                           # x / rms_B = x / (nscale rms_A): scale the gain back
        return out

    def mlp(pre, f, w=None, rows=None, scale=1.0):
        """Host MLP (width f) at prefix pre; w: target MLP getter, rows: target intermediate slice."""
        g, u, dn = z(f, hB), z(f, hB), z(hB, f)
        if w is not None:
            tg, tu, td = w("gate_proj.weight"), w("up_proj.weight"), w("down_proj.weight")
            rows = rows if rows is not None else slice(0, tg.shape[0])
            n = len(range(*rows.indices(tg.shape[0])))
            g[:n, :hA], u[:n, :hA], dn[:hA, :n] = tg[rows], tu[rows], td[:, rows] * scale
        return {pre + "gate_proj.weight": g, pre + "up_proj.weight": u, pre + "down_proj.weight": dn}

    # head assignment: target head -> host head, target kv head -> host kv heads
    gA, gB = a["H"] // a["kv"], b["H"] // b["kv"]
    head_of, kv_of, nxt = {}, {}, 0
    for kv in range(a["kv"]):
        heads = list(range(kv * gA, (kv + 1) * gA))
        for chunk in range(math.ceil(gA / gB)):
            grp = nxt
            nxt += 1
            kv_of.setdefault(kv, []).append(grp)
            for s, th in enumerate(heads[chunk * gB:(chunk + 1) * gB]):
                head_of[th] = grp * gB + s
    qk = b["fam"] == "qwen3"
    qscale = 1.0 if qk else math.sqrt(k)                # 1/sqrt(hd_B) -> 1/sqrt(hd_A)
    src_of = {j: (i, r) for i, (j, r) in enumerate(p["layers"])}

    out = {}
    E = z(b["V"], hB)
    E[:a["V"], :hA] = tsd["model.embed_tokens.weight"]
    if p["free"]:
        E[:a["V"], hA] = c
    out["model.embed_tokens.weight"] = E
    for L in range(b["L"]):
        pre = f"model.layers.{L}."
        i, route = src_of.get(L, (None, None))
        src = f"model.layers.{i}." if i is not None else None
        q, kk, v, o = z(b["H"] * hdB, hB), z(b["kv"] * hdB, hB), z(b["kv"] * hdB, hB), z(hB, b["H"] * hdB)
        qb, kb, vb = z(b["H"] * hdB), z(b["kv"] * hdB), z(b["kv"] * hdB)
        n1, n2 = z(hB), z(hB)
        n1[:] = 1.0
        n2[:] = 1.0
        T = (lambda n: tsd[src + n]) if src is not None else None
        if src is not None:
            tq, tk, tv, to = T("self_attn.q_proj.weight"), T("self_attn.k_proj.weight"), T("self_attn.v_proj.weight"), T("self_attn.o_proj.weight")
            has_b = src + "self_attn.q_proj.bias" in tsd
            for th, hh in head_of.items():
                q[hh * hdB + mu, :hA] = tq[th * hdA:(th + 1) * hdA] * qscale
                o[:hA, hh * hdB + mu] = to[:, th * hdA:(th + 1) * hdA]
                if has_b:
                    qb[hh * hdB + mu] = T("self_attn.q_proj.bias")[th * hdA:(th + 1) * hdA] * qscale
            for tkv, groups in kv_of.items():
                for grp in groups:
                    kk[grp * hdB + mu, :hA] = tk[tkv * hdA:(tkv + 1) * hdA]
                    v[grp * hdB + mu, :hA] = tv[tkv * hdA:(tkv + 1) * hdA]
                    if has_b:
                        kb[grp * hdB + mu] = T("self_attn.k_proj.bias")[tkv * hdA:(tkv + 1) * hdA]
                        vb[grp * hdB + mu] = T("self_attn.v_proj.bias")[tkv * hdA:(tkv + 1) * hdA]
            n1, n2 = norm_w(T("input_layernorm.weight")), norm_w(T("post_attention_layernorm.weight"))
            if qk:
                # host rms over hd_B dims = sqrt(hd_A/hd_B) x target's: normalized q, k each come out
                # sqrt(k) larger, and 1/sqrt(hd_B) takes one sqrt(k) back, so k's gain absorbs the other
                qn, kn = z(hdB), z(hdB)
                qn[mu] = T("self_attn.q_norm.weight")
                kn[mu] = T("self_attn.k_norm.weight") / math.sqrt(k)
                out[pre + "self_attn.q_norm.weight"], out[pre + "self_attn.k_norm.weight"] = qn, kn
        elif qk:
            out[pre + "self_attn.q_norm.weight"] = torch.ones(hdB, dtype=dt)
            out[pre + "self_attn.k_norm.weight"] = torch.ones(hdB, dtype=dt)
        tw = (lambda base: (lambda n: tsd[src + base + n])) if src is not None else None
        if not b["sparse"][L]:
            out.update(mlp(pre + "mlp.", b["ff"], tw("mlp.") if src is not None else None))
        else:
            R = z(b["E"], hB)                           # router; experts default to zero (unused layer)
            used = 0
            ex = {e: mlp(pre + f"mlp.experts.{e}.", b["fm"]) for e in range(b["E"])}
            if route is not None and route[0] == "moe":
                r = route[1]
                tR = T("mlp.gate.weight")
                tg = tied_router_rows(tR) if r > 1 and a["k"] < a["E"] else []
                if tg:
                    msg = (f"target {src}router: rows {tg} are identical, so the top-{a['k']} is not "
                           f"strict and the {r}x duplication (k_B = {r} k_A) is not exact at the ties")
                    if ties == "refuse":
                        raise ValueError(msg)
                    warnings.warn(msg)
                for cp in range(r):                    # r tied copies of every real expert
                    for e in range(a["E"]):
                        R[cp * a["E"] + e, :hA] = tR[e]
                        ex[cp * a["E"] + e] = mlp(pre + f"mlp.experts.{cp * a['E'] + e}.", b["fm"],
                                                  tw(f"mlp.experts.{e}."))
                used = r * a["E"]
            elif route is not None and route[0] == "steer":
                m = route[1]
                s = math.ceil(a["ff"] / m)
                for e in range(m):                      # equal (zero) router logits: weight 1/m each
                    ex[e] = mlp(pre + f"mlp.experts.{e}.", b["fm"], tw("mlp."),
                                rows=slice(e * s, min((e + 1) * s, a["ff"])), scale=float(m))
                used = m
            if route is not None and used < b["E"]:
                R[used:, hA] = -moe_row                 # dummies: softmax prob exactly 0
                n2[hA] = 1.0 / c                        # p enters the router as 1/rms_B
            for e in range(b["E"]):
                out.update(ex[e])
            out[pre + "mlp.gate.weight"] = R
            if b["mt"] == "qwen2_moe":
                has_sh = route is not None and route[0] == "moe" and a["fs"] > 0
                out.update(mlp(pre + "mlp.shared_expert.", b["fs"], tw("mlp.shared_expert.") if has_sh else None))
                sg = z(1, hB)
                if has_sh:
                    sg[:, :hA] = T("mlp.shared_expert_gate.weight")
                out[pre + "mlp.shared_expert_gate.weight"] = sg
        out.update({pre + "self_attn.q_proj.weight": q, pre + "self_attn.k_proj.weight": kk,
                    pre + "self_attn.v_proj.weight": v, pre + "self_attn.o_proj.weight": o,
                    pre + "input_layernorm.weight": n1, pre + "post_attention_layernorm.weight": n2})
        if b["bias"]:
            out.update({pre + "self_attn.q_proj.bias": qb, pre + "self_attn.k_proj.bias": kb,
                        pre + "self_attn.v_proj.bias": vb})
    fn = norm_w(tsd["model.norm.weight"])
    head = z(b["V"], hB)
    head[:a["V"], :hA] = tsd["lm_head.weight"] if "lm_head.weight" in tsd else tsd["model.embed_tokens.weight"]
    if vocab_trick:
        if b["tied"]:                                   # tied host: the head IS the embedding table
            E[a["V"]:, hA] = -VOCAB_ROW_TIED
            fn[hA] = 1.0
            head = E
        else:
            fn[hA] = gain
            head[a["V"]:, hA] = -1.0
    out["model.norm.weight"] = fn
    if not b["tied"]:
        out["lm_head.weight"] = head
    assert eps_b > 0
    return out
