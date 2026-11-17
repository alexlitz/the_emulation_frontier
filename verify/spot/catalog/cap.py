"""Capacity, KV-head and layer-kind reductions, built by elephantine/emulate.py (in memory, its relation gate on)
or, for KV-head packing which emulate.py does not build, by a packing writer below."""
from __future__ import annotations

import math

import torch

from verify.spot.catalog import EXACT_REL, F64, compare, entry, hf_fn
from verify.spot.catalog._emu import BIG, SMALL, V, emu_check, model_from, raw

LT3 = ["sliding_attention", "full_attention", "sliding_attention"]


@entry("capacity:hidden", "capacity:eff_layers", "capacity:param_layers", "capacity:ffn_total", "capacity:ffn_active",
       "capacity:kv_cap", "capacity:q_cap", "capacity:qheads", "capacity:head_dim", "capacity:layer_kind:FU<-FU",
       instance="llama (d 64, 2 layers, 4q/2kv x 16, FFN 96) -> llama (d 96, 3 layers, 6q/3kv x 32, FFN 160)")
def cap_llama_every_axis():
    return emu_check("llama", {}, "llama", {})


@entry("slot:kv:MQA->GQA", instance="llama MQA (4q/1kv) -> llama GQA (6q/3kv)")
def cap_kv_mqa_gqa():
    return emu_check("llama", dict(num_key_value_heads=1), "llama", {})


@entry("slot:kv:GQA->MHA", instance="llama GQA (4q/2kv) -> llama MHA (6q/6kv)")
def cap_kv_gqa_mha():
    return emu_check("llama", {}, "llama", dict(num_key_value_heads=6))


@entry("slot:kv:MQA->MHA", instance="llama MQA (4q/1kv) -> llama MHA (6q/6kv)")
def cap_kv_mqa_mha():
    return emu_check("llama", dict(num_key_value_heads=1), "llama", dict(num_key_value_heads=6))


@entry("slot:kv:MHA->GQA", instance="llama MHA (3q/3kv) -> llama GQA (6q/3kv): one query per host group used")
def cap_kv_mha_gqa():
    return emu_check("llama", dict(num_attention_heads=3, num_key_value_heads=3, hidden_size=48), "llama", {})


@entry("slot:kv:MHA->MQA", instance="llama MHA (1q/1kv) -> llama MQA (6q/1kv)")
def cap_kv_mha_mqa():
    return emu_check("llama", dict(num_attention_heads=1, num_key_value_heads=1, hidden_size=16), "llama",
                     dict(num_key_value_heads=1))


@entry("capacity:layer_kind:LU<-LU", "capacity:layer_kind:FU<-FU",
       instance="gemma2 (sliding, full) -> gemma2 (sliding, full, sliding): each layer on a host layer of its kind")
def cap_kind_gemma2():
    return emu_check("gemma2", dict(sliding_window=4, layer_types=LT3[:2], query_pre_attn_scalar=16),
                     "gemma2", dict(sliding_window=4, layer_types=LT3, query_pre_attn_scalar=24))


@entry("capacity:layer_kind:LU<-FU", "slot:window:local->full",
       instance="mistral (window 4) -> qwen2 (full layers): the host's window set to the target's")
def cap_kind_local_on_full():
    return emu_check("mistral", dict(sliding_window=4), "qwen2", dict(use_sliding_window=False))


@entry("zero:window:full<-local", "slot:window:full->local",
       instance="qwen2 (2 full layers) -> qwen2 (sliding, full, full): full target layers only on host full layers")
def cap_zero_window():
    return emu_check("qwen2", {}, "qwen2", dict(use_sliding_window=True, sliding_window=4,
                                                 layer_types=["sliding_attention", "full_attention", "full_attention"]))


# --- K = V layers (Gemma-4 attention_k_eq_v: no stock class in this transformers) -------------------------------
# A K = V layer computes V with the key projection. As a target: a llama whose v_proj IS its k_proj (the same
# function); a host with its own v_proj plays it by copying k_proj into v_proj (KIND_REACH U -> T).

def _keqv(m, layers=None):
    with torch.no_grad():
        for i, L in enumerate(m.model.layers):
            if layers is None or i in layers:
                L.self_attn.v_proj.weight.copy_(L.self_attn.k_proj.weight)
                if L.self_attn.v_proj.bias is not None:
                    L.self_attn.v_proj.bias.copy_(L.self_attn.k_proj.bias)
    return m


@torch.no_grad()
def _kind_check(t_types, h_types, t_keqv, h_keqv):
    """A K = V target (layers t_keqv) on a host whose layers h_keqv are K = V too: emulate.py's construction with
    V written from the target's (tied) key projection; a host K = V layer reads only a K = V target layer."""
    from verify.spot.catalog._emu import emulate_pair
    import elephantine.emulate as E
    tkw = dict(SMALL, sliding_window=4, layer_types=t_types, use_sliding_window=True)
    hkw = dict(BIG, sliding_window=4, layer_types=h_types, use_sliding_window=True, num_hidden_layers=len(h_types))
    target, host, con, _ = emulate_pair(raw("qwen2", tkw), raw("qwen2", hkw))
    _keqv(target, t_keqv)
    # re-run the construction on the K = V target (its v_proj now the key projection)
    T = E.read_target(target, E.resolved(raw("qwen2", tkw)), con.tfam)
    hdict = E.host_config_dict(raw("qwen2", hkw), con)
    sd = E.build_host(T, E.resolved(raw("qwen2", tkw)), hdict, con, log=lambda *a: None)
    host.load_state_dict({k: v.to(F64) for k, v in sd.items()}, strict=False)
    host.tie_weights()
    for j in h_keqv:
        L = host.model.layers[j].self_attn
        assert torch.equal(L.v_proj.weight, L.k_proj.weight) and torch.equal(L.v_proj.bias, L.k_proj.bias), \
            f"host layer {j} is K = V but holds an own-V target layer"
    with E._lift():
        return compare(hf_fn(target), hf_fn(host), V, note=f"layer_pos {con.layer_pos}")


@entry("capacity:layer_kind:FT<-FU", "capacity:layer_kind:LT<-LU",
       instance="qwen2 K=V target (sliding, full) -> qwen2 own-V host (reference: K = V as tied k / v projections)")
def cap_kind_keqv_on_own_v():
    return _kind_check(LT3[:2], LT3, {0, 1}, set())


@entry("capacity:layer_kind:FT<-FT", "capacity:layer_kind:LT<-LT", "capacity:layer_kind:LT<-FT",
       "capacity:layer_kind:LT<-FU",
       instance="qwen2 K=V (sliding, sliding) -> qwen2 K=V host (full, sliding, full; reference: tied k / v)")
def cap_kind_keqv_on_keqv():
    # target K = V on both (sliding) layers; host layers 0 (full) and 1 (sliding) are K = V: LT <- FT, LT <- LT
    return _kind_check(["sliding_attention", "sliding_attention"], ["full_attention", "sliding_attention",
                                                                     "full_attention"], {0, 1}, {0, 1})


# --- KV-head packing (graph.model_graph.kv_heads_fit; emulate.py refuses it) ------------------------------------

def _edge(traw, hraw):
    from graph.edge_plan import edge_plan
    from graph.model_graph import classify, dims_of
    A = {"slots": classify(traw), "dims": dims_of(traw)}
    B = {"slots": classify(hraw), "dims": dims_of(hraw)}
    p = edge_plan(A, B)
    assert p["ok"], "the synthetic pair is not an edge of the headline relation"
    return p


@torch.no_grad()
def _pack(t, hcfg_kw, nope=False):
    """llama target t (H_t query heads on KV_t heads of head_dim d) -> llama host with KV_h heads of head_dim
    D = m d: m target KV heads side by side in one host head (zero-padded), each target query head on its own host
    query head reading its block (the 1/sqrt(D) scale folded into q), o_proj reading that block. RoPE: the host's
    longrope table repeats the target's frequencies once per block (rotate_half layout: block b's halves at
    b d / 2 and D / 2 + b d / 2)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    tc = t.config
    Ht, KVt, d = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    D, Hh, KVh = hcfg_kw["head_dim"], hcfg_kw["num_attention_heads"], hcfg_kw["num_key_value_heads"]
    m = D // d
    gt, gh = Ht // KVt, Hh // KVh
    assert KVt <= m * KVh and gh >= m * gt
    base = tc.rope_theta
    half_t, half_h = d // 2, D // 2
    # host pair j holds target pair i of some block: longrope factor so that inv_freq_h[j] = inv_freq_t[i]
    fac = []
    for j in range(half_h):
        i = j % half_t
        fac.append(base ** (2 * i / d) / base ** (2 * j / D))
    hkw = dict(hcfg_kw, rope_theta=base, rope_scaling={"rope_type": "longrope", "short_factor": fac,
                                                        "long_factor": fac, "factor": 1.0, "attention_factor": 1.0,
                                                        "original_max_position_embeddings": tc.max_position_embeddings})
    hc = LlamaConfig(**hkw)
    hc._attn_implementation = "eager"
    h = LlamaForCausalLM(hc).to(F64).eval()
    for p in h.parameters():
        p.zero_()
    perm = lambda b: [b * half_t + e if e < half_t else half_h + b * half_t + (e - half_t) for e in range(d)]
    fold = math.sqrt(D / d)                                    # host scale 1/sqrt(D) vs target 1/sqrt(d)
    h.model.embed_tokens.weight[:, : tc.hidden_size] = t.model.embed_tokens.weight
    h.lm_head.weight[:, : tc.hidden_size] = t.lm_head.weight
    dd, DD = tc.hidden_size, hc.hidden_size
    for Lt, Lh in zip(t.model.layers, h.model.layers):
        at, ah = Lt.self_attn, Lh.self_attn
        for c in range(KVt):
            hk, b = c // m, c % m
            cols = [hk * D + p for p in perm(b)]
            ah.k_proj.weight[cols, :dd] = at.k_proj.weight[c * d:(c + 1) * d]
            ah.v_proj.weight[cols, :dd] = at.v_proj.weight[c * d:(c + 1) * d]
            for r in range(gt):
                jt = c * gt + r
                jh = hk * gh + b * gt + r                      # a distinct host query head in host group hk
                rows = [jh * D + p for p in perm(b)]
                ah.q_proj.weight[rows, :dd] = at.q_proj.weight[jt * d:(jt + 1) * d] * fold
                ah.o_proj.weight[:dd, rows] = at.o_proj.weight[:, jt * d:(jt + 1) * d]
        g = math.sqrt(dd / DD)
        Lh.input_layernorm.weight[:dd] = Lt.input_layernorm.weight * g
        Lh.post_attention_layernorm.weight[:dd] = Lt.post_attention_layernorm.weight * g
        It = tc.intermediate_size
        Lh.mlp.gate_proj.weight[:It, :dd] = Lt.mlp.gate_proj.weight
        Lh.mlp.up_proj.weight[:It, :dd] = Lt.mlp.up_proj.weight
        Lh.mlp.down_proj.weight[:dd, :It] = Lt.mlp.down_proj.weight
    h.model.norm.weight[:dd] = t.model.norm.weight * math.sqrt(dd / DD)
    h.config.rms_norm_eps = tc.rms_norm_eps * dd / DD
    for mod in h.modules():
        if hasattr(mod, "variance_epsilon"):
            mod.variance_epsilon = tc.rms_norm_eps * dd / DD
    if nope:                                                   # position-free target and host: inv_freq = 0
        for mm in (t, h):
            mm.model.rotary_emb.inv_freq.zero_()
            mm.model.rotary_emb.original_inv_freq = mm.model.rotary_emb.inv_freq
    return h


def _pack_check(nope):
    from transformers import LlamaConfig
    t = model_from(raw("llama", dict(SMALL, tie_word_embeddings=False, num_attention_heads=4, num_key_value_heads=2,
                                     head_dim=8, hidden_size=32)))
    hkw = dict(BIG, num_attention_heads=8, num_key_value_heads=1, head_dim=16, hidden_size=48,
               tie_word_embeddings=False)
    hraw = raw("llama", dict(hkw, rope_scaling={"rope_type": "longrope", "short_factor": [1.0] * 8,
                                                "long_factor": [1.0] * 8, "factor": 1.0,
                                                "original_max_position_embeddings": 256}))
    _edge(raw("llama", dict(SMALL, num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_size=32)), hraw)
    h = _pack(t, hkw, nope=nope)
    import elephantine.emulate as E
    with E._lift():
        return compare(hf_fn(t), hf_fn(h), V, tol=1e-9 if nope else 1e-6,
                       note="" if nope else "the longrope factors are fp32: inv_freq agrees to ~1e-7")


@entry("capacity:kv_pack:rope", "slot:kv:GQA->MQA",
       instance="llama GQA (4q/2kv x 8) -> llama MQA (8q/1kv x 16, longrope table): 2 KV heads packed in one")
def cap_kv_pack_rope():
    return _pack_check(nope=False)


@entry("capacity:kv_pack:nope", instance="llama GQA (4q/2kv x 8, inv_freq 0 = NoPE) -> llama MQA (8q/1kv x 16, NoPE)")
def cap_kv_pack_nope():
    return _pack_check(nope=True)
