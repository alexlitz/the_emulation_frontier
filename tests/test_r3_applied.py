"""Round 3: the applied specs (graph/model_graph.py) against their detectors (verify/r3/detectors.py, the oracle).

For every applied spec (verify/r3/specs/<name>.md) a value function maps model_graph's classification of a snapshot
config (classify / _feats / dims_of / skipped) to the detector's value. For each fixture `flag` repo the value equals
the detector's (verify/r3/fixtures/*.json; a `true` fixture: any truthy value); for each `not_flag` repo the spec
changes nothing (the value with the spec on equals the value with it in R3_OFF); and switching the spec off brings the
old value back for at least one flag repo (the change is gated by its spec name, for the leave-one-out tables).
Configs come from graph/hub_10k.json through iter_json_configs (which carries the repo id, `repo_keyed` fixtures).

Batch 1 (classification) adds its specs here; later batches append theirs to APPLIED.

Run:  python -m pytest tests/test_r3_applied.py -q
"""
import contextlib
import functools
import importlib.util
import json
import os
import sys

import pytest

from graph import model_graph as mg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "verify", "r3")
sys.path.insert(0, R3)
import r3util  # noqa: E402

_spec = importlib.util.spec_from_file_location("r3_detectors", os.path.join(R3, "detectors.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

FIXTURES = {}
for _fn in sorted(os.listdir(os.path.join(R3, "fixtures"))):
    if _fn.endswith(".json"):
        FIXTURES.update(json.load(open(os.path.join(R3, "fixtures", _fn))))

UNPLACED = "<unplaced>"          # classify gives no point (the value cannot be read off a slot)


@functools.lru_cache(maxsize=None)
def _snap():
    return dict(mg.iter_json_configs(mg.ZOO_CONFIGS))


@contextlib.contextmanager
def off(*specs):
    """Switch round-3 specs off for the block (R3_OFF), restoring the set after."""
    added = [s for s in specs if s not in mg.R3_OFF]
    for s in added:
        mg.R3_OFF.add(s)
    try:
        yield
    finally:
        for s in added:
            mg.R3_OFF.discard(s)


def _lm(cfg):
    return mg._lm_config(cfg)


def _feats(cfg):
    d = _lm(cfg)
    return set(mg._feats(d, cfg if cfg is not d else None))


def _cls(cfg):
    """classify, with the skip-only specs off so a skipped model (sundial, the rerankers) still shows its slots."""
    with off("not_a_causal_lm", "remote_code_not_config_family"):
        return mg.classify(cfg)


def _dims(cfg):
    with off("not_a_causal_lm", "remote_code_not_config_family"):
        return mg.dims_of(cfg) if mg.classify(cfg) else None


def _slot(cfg, s):
    c = _cls(cfg)
    return c[s] if c else UNPLACED


# --- per spec: (cfg) -> the value the detector computes -----------------------------------------------------------
def v_bias_qkv(cfg):
    return "bias_qkv" in _feats(cfg)


def v_bias_qkv_table_wrong(cfg):
    # the detector flags a type whose code has no q / k / v bias although p5 listed it: no bias_qkv, no gate bias
    return "bias_qkv" not in _feats(cfg) and not mg._ffn_bias(_lm(cfg))


def v_bias_out(cfg):
    return "bias_out" in _feats(cfg)


def v_nanochat(cfg):
    # (batch 2a: qknorm_fixed_gain refines a ported norm with a config gain != 1 -- gaggle qk_scale 1.2 -- to
    # head_free_gain; this spec's own value is the gain-free reading)
    with off("qknorm_fixed_gain"):
        c = _cls(cfg)
    return f"ffn={c['ffn']},act={c['act']},qknorm={c['qknorm']}" if c else UNPLACED


def v_tied(cfg):
    d = _lm(cfg)
    return mg._tied(d, cfg if cfg is not d else None)


def v_config_default(cfg):
    lm, raw = _lm(cfg), r3util.lm_of(cfg)
    mt = str(lm.get("model_type") or "").lower()
    c = _cls(cfg)
    out = []
    if mt == "llada2_moe" and "use_qk_norm" not in raw and c and c["qknorm"] != "no":
        out.append(["use_qk_norm", True])
    if mt in ("muse_glimmer", "muse_glimmer_text") and "final_logit_softcapping" not in raw and c and c["softcap"] == "final":
        out.append(["final_logit_softcapping", 20.0])
    if mt == "exaone_moe" and "sliding_window" not in raw and lm.get("sliding_window"):
        out.append(["sliding_window", lm["sliding_window"]])
    return out or None


def v_kv_heads(cfg):
    dm = _dims(cfg)
    return dm["kv_heads"] if dm and dm["kv_heads"] != dm["heads"] else None


def v_head_dim(cfg):
    dm = _dims(cfg)
    return dm["head_dim"] if dm else UNPLACED


def v_layers(cfg):
    dm = _dims(cfg)
    return dm["layers"] if dm else UNPLACED


def v_mm200(cfg):
    dm, f = _dims(cfg), _feats(cfg)
    return [max(dm["rope_sched"]), "out_gate_post" if "out_gate_post" in f else ""] if dm else UNPLACED


def v_not_lm(cfg):
    # (round 3, nonlm_kept -- a design decision: the three stay placed with a wall-both-ways feature, score_head /
    # ts_patch, read back as the detector's kind; with nonlm_kept off batch 1's not_lm skip gives the kind)
    # (round 3, nonlm_heads: the head / input slots instead of the features -- a score head is the detector's
    # score_head, a flow head on patches its continuous_io)
    if mg.skipped(cfg) == "not_lm":
        return mg._not_lm(cfg, _lm(cfg))
    c = mg.classify(cfg)
    if c and c.get("head") == "score":
        return "score_head"
    if c and c.get("input") == "patches":
        return "continuous_io"
    inv = {v: k for k, v in mg.NONLM_FEAT.items()}
    f = _feats(cfg) if c else set()
    return next((inv[x] for x in sorted(f) if x in inv), None)


def v_model_type(cfg):
    return str(_lm(cfg).get("model_type") or "").lower() or None


def v_skip(cfg):
    return "skip" if mg.skipped(cfg) == "non_decoder" else None


def v_audio_qwen2(cfg):
    c = _cls(cfg)
    if not c:
        return UNPLACED
    return "qwen2" if c["pos"] == "rope" and "bias_qkv" in c["feats"].split("+") else None


def v_breeze(cfg):
    if str(cfg.get("model_type") or "").lower() != "breeze":
        return None
    mt = v_model_type(cfg)
    return mt if mt != "breeze" else None


def feat(name):
    return lambda cfg: name in _feats(cfg)


def v_emb_rank(cfg):
    return mg._r3_dims(_lm(cfg))["emb_rank"] or None


def v_emb_norm(cfg):
    return mg._r3_dims(_lm(cfg))["emb_norm"] or None


def v_norm_groups(cfg):
    g = mg._r3_dims(_lm(cfg))["norm_groups"]
    return g if g > 1 else None


def v_placement(cfg):
    return "+".join(sorted(_feats(cfg) & {"post_norm", "post_only", "post_ln", "post_norm_attn"})) or None


def v_multimax(cfg):
    return str(r3util.get(cfg, "multimax")) if "multimax" in _feats(cfg) else None


def v_dsv4(cfg):
    f = _feats(cfg)
    return "+".join(x for x in ("kv_compress", "hash_route") if x in f) or None


# --- batch 2a (qk-norm and output gates) ---
def _og(cfg):
    """The output-gate FEATURES key (model_graph._out_gate_key: out_gate[_head][_bias][_act] / out_gate_qlat), or None."""
    ks = [x for x in _feats(cfg) if x.startswith("out_gate") and x != "out_gate_post"]
    assert len(ks) <= 1, ks
    return ks[0] if ks else None


def v_og_kind(cfg):
    # the out_gate_by_code_unlisted / out_gate_key_unread value: granularity + "_bias", or "elem_qlat"
    k = _og(cfg)
    if not k:
        return None
    if k == "out_gate_qlat":
        return "elem_qlat"
    return ("head" if "_head" in k else "elem") + ("_bias" if "_bias" in k else "")


def v_og_head(cfg):
    k = _og(cfg)
    return "head" if k and "_head" in k else None


def v_og_act(cfg):
    k = _og(cfg)
    return next((a for a in ("softplus", "silu") if k and k.endswith("_" + a)), None)


# --- batch 2b (positions) ---
_GATE_KIND = {"qwen": "logn", "ministral3": "llama4_beta", "mistral4": "llama4_beta",
              "nemotron_labs_diffusion": "llama4_beta", "llama4": "nope_temp", "llama4_text": "nope_temp"}


def v_q_temp(cfg):
    # the ungated forms are FEATURES keys; the gated ones the dims pos_gate (PLAN decision 3), this spec's gates alone
    f = _feats(cfg)
    if "q_temp" in f:
        return "learned_logpos"
    if "q_temp_nope" in f:
        return "ssmax"
    with off("dynamic_ntk_in_context"):
        d = _dims(cfg)
    pg = d and d["pos_gate"]
    return f"{_GATE_KIND.get(str(_lm(cfg).get('model_type') or '').lower(), '?')}@{pg}" if pg else None


def v_dyn_ntk(cfg):
    with off("position_query_temperature"):
        d = _dims(cfg)
    return f"dyn_ntk@{d['pos_gate']}" if d and d["pos_gate"] else None


def _dim(key, pos_only=False):
    def v(cfg):
        d = _dims(cfg)
        if d is None:
            return UNPLACED
        x = d[key]
        return tuple(r for r in x if r > 0) if pos_only else x
    return v


def v_partial_rope(cfg):
    # the rotating tables only (Grug's NoPE long layers are nope_full_layers_unlisted's -64 entry)
    with off("nope_full_layers_unlisted"):
        return _dim("rope_sched", pos_only=True)(cfg)


def v_alibi_scale(cfg):
    d = _dims(cfg)
    sc = d.get("alibi_scale", 0.0) if d else 0.0
    return sc if sc and sc != 1.0 else None


def v_no_rope_layers(cfg):
    return tuple(int(bool(x)) for x in _lm(cfg)["no_rope_layers"]) if "nope_layers" in _feats(cfg) else None


# batch 2c
def featv(name):
    return lambda cfg: name if name in _feats(cfg) else None


def _binds(dm):
    """A read window binds below the context: a read one smaller, or an unknown context (None: round 3,
    max_context_unknown -- no default to compare against)."""
    w, c = dm["eff_window"], dm["max_context"]
    return w is not None and (c is None or w < c)


def v_sliding_layout(cfg):
    """[n_full, n_local, visible keys per sliding layer] (_attn_counts kinds, dims_of eff_window); None when no
    sliding layer binds."""
    dm = _dims(cfg)
    if dm is None:
        return UNPLACED
    d = _lm(cfg)
    kc = mg._attn_counts(d, mg._layer_count(d))[4]
    nf, nl = kc["FU"] + kc["FT"], kc["LU"] + kc["LT"]
    return [nf, nl, dm["eff_window"]] if nl and _binds(dm) else None


def v_chunked(cfg):
    dm = _dims(cfg)
    if dm is None:
        return UNPLACED
    k = sum(1 for x in (_lm(cfg).get("no_rope_layers") or []) if x)
    return [k, dm["eff_window"]] if k and _binds(dm) else None


def v_window_sink(cfg):
    c = _cls(cfg)
    return [c["window"], c["sink"]] if c else UNPLACED


# batch 2d
def v_kv_shapes(cfg):
    """The distinct (kv heads, q per kv, head dim) of the per-layer shapes (kv_shapes; None for one shape)."""
    d = _dims(cfg)
    if d is None:
        return UNPLACED
    s = sorted({tuple(x[:3]) for x in d["kv_shapes"]})
    return [list(x) for x in s] if len(s) > 1 else None


def v_v_head(cfg):
    d = _dims(cfg)
    if d is None:
        return UNPLACED
    return [d["head_dim"], d["v_head"]] if d["v_head"] != d["head_dim"] else None


def v_o_groups(cfg):
    d = _dims(cfg)
    if d is None:
        return UNPLACED
    return f"g{d['o_groups']}r{d['o_rank']}" if d["o_groups"] else None


def v_xsa(cfg):
    f = _feats(cfg)
    return "unit" if "xsa" in f else ("scaled" if "xsa_scaled" in f else None)


def v_mem(cfg):
    if "mem_attn" not in _feats(cfg):
        return None
    return "topk_kv" if str(_lm(cfg).get("model_type") or "").lower() == "edgerock" else "summary"


def v_mova(cfg):
    lm = _lm(cfg)
    return [lm["mova_num_experts"], lm["mova_num_experts_per_tok"]] if "mova" in _feats(cfg) else None


def v_moa(cfg):
    d = _dims(cfg)
    if d is None:
        return UNPLACED
    return f"E{d['moa_e']}k{d['moa_k']}" if "moa" in _feats(cfg) and d["moa_e"] else None


def v_shared_mla(cfg):
    d, c = _dims(cfg), _cls(cfg)
    sm = mg._shared_mla(_lm(cfg))
    if not (d and c and sm):
        return None
    return {"kv": c["kv"], "kv_latent": d["kv_latent"], "q_latent": d["q_latent"], "mla_rope": d["mla_rope"],
            "mla_nope": d["mla_nope"], "mla_v": d["mla_v"], "mla_layers": len(sm["layers"])}


def v_ernie_mla(cfg):
    d = _dims(cfg)
    if d is None:
        return UNPLACED
    if str(_lm(cfg).get("model_type") or "").lower() != "ernie5_5_moe":
        return None
    return {"heads": d["heads"], "q_latent": d["q_latent"], "kv_latent": d["kv_latent"], "mla_nope": d["mla_nope"],
            "mla_rope": d["mla_rope"], "mla_v": d["mla_v"]}


def v_kv_compress(cfg):
    f = _feats(cfg)
    return "dsv41" if "kv_compress_v41" in f else ("dsv4" if "kv_compress" in f else None)


def v_cyrah(cfg):
    c = _cls(cfg)
    return f"kv={c['kv']},pos={c['pos']},window={c['window']},qknorm={c['qknorm']}" if c else UNPLACED


def v_clip_host(cfg):
    # b_clip_qkv_host is a host-side config lever (clip_qkv None): the feature's host rule
    return "clip_qkv" if "clip_qkv" in _feats(cfg) and mg.FEATURES["clip_qkv"][1] == "free" else None


# batch 3a
def _sigv(cfg, f):
    """f of the moe_sig fields (graph.model_graph._moe_fields) of a MoE point, None for a dense one."""
    d = _dims(cfg)
    if d is None:
        return UNPLACED
    s = mg._moe_fields(d["moe_sig"]) if d.get("moe_sig") else None
    return f(s) if s and s["E"] else None


def v_renorm(cfg):
    # the detector's 'key:' prefix: the renorm read from a config key (renorm_cap P | R)
    return _sigv(cfg, lambda s: ("key:" if tuple(s["renorm_cap"]) == ("P", "R") else "") + s["renorm"])


def v_score(cfg):
    # (the closing step's score_not_free makes every score_cap the own score; router_score_unread's detector value is
    # the settable set the code reads, so it is read with score_not_free off)
    with off("score_not_free"):
        return _sigv(cfg, lambda s: f"{s['score']}/{'|'.join(s['score_cap'])}")


def v_router_bias(cfg):
    f = _feats(cfg)
    return "logit" if "router_bias_logit" in f else ("score" if "router_bias" in f else None)


def v_router_dense(cfg):
    c = _cls(cfg)
    if not c:
        return UNPLACED
    if c["ffn"].startswith("moe"):
        return None
    f = _feats(cfg)
    return None if any(x.startswith("router_") or x in ("shared_gate", "hash_route") for x in f) else "dense"


# batch 3b
def v_clip_list(cfg):
    """act_clip_per_layer_list: the largest Ling-3 per-layer limit, where act_clip is set."""
    d = _lm(cfg)
    lim = max(mg._num(d.get("expert_swiglu_limit_list")), mg._num(d.get("share_expert_swiglu_limit_list")))
    return (lim or None) if "act_clip" in _feats(cfg) else None


def v_act_clip_host(cfg):
    # b_act_clip_host is a host-side config lever (swiglu_limit = inf / 0): the key whose limit is read, where the
    # feature is set and its host side free
    if "act_clip" not in _feats(cfg) or mg.FEATURES["act_clip"][1] != "free":
        return None
    d = _lm(cfg)
    return next((k for k in ("swiglu_limit", "swiglu_limits", "swiglu_limits_shared") if mg._num(d.get(k)) > 0), None)


def _ffn_widths(cfg):
    d = _dims(cfg)
    if d is None:
        return None
    s = mg._moe_fields(d["moe_sig"]) if d.get("moe_sig") else {}
    return {"inter": d["inter"], "ew": s.get("ew", 0) if s.get("E") else 0,
            "dense_inter": s.get("dense_w", 0) if s.get("E") else 0}


def v_ffn_width(cfg):
    """ffn_width_keys_unread: the first width (dense_inter, inter, ew) that the spec moves, with its new value."""
    on = _ffn_widths(cfg)
    if on is None:
        return UNPLACED
    with off("ffn_width_keys_unread"):
        was = _ffn_widths(cfg)
    return next(([k, on[k]] for k in ("dense_inter", "inter", "ew") if on[k] != was[k]), None)


# batch 4a (recurrent mixers and loops)
def _fam(cfg):
    """The recurrent family of a config's recurrent layers (None for a softmax-only stack)."""
    c = _cls(cfg)
    return mg._recurrent_family(_lm(cfg), c["mixer"]) if c else UNPLACED


def _rc(keys):
    def v(cfg):
        d = _dims(cfg)
        return UNPLACED if d is None else [d.get(k, 0) for k in keys]
    return v


def v_heads(cfg):
    d = _dims(cfg)
    return UNPLACED if d is None else d["heads"]


def v_no_attention(cfg):
    """no_attention: an all-recurrent stack (mixer ssm / linear) with attention keys, read as position-free with no
    attention-only feature."""
    c = _cls(cfg)
    if not c or c["mixer"] not in ("ssm", "linear"):
        return None
    raw = r3util.lm_of(cfg)
    if not any(D.truthy_flag(raw.get(k)) for k in D._ATTN_ONLY_KEYS):
        return None
    fs = set(filter(None, c["feats"].split("+")))
    clean = c["pos"] == "nope" and not (fs & mg.ATTN_ONLY_FEATS) and not any(f.startswith("out_gate") for f in fs)
    return "no_attention" if clean else None


def v_schedule(cfg):
    """[n_softmax, n_recurrent, family]: the own-KV and recurrent layer kinds (weight-distinct) and the family."""
    d = _dims(cfg)
    if d is None:
        return UNPLACED
    return [d.get("attn_param", 0), d.get("kR_param", 0), _fam(cfg)]


def v_loop_bnorm(cfg):
    fs, mt = _feats(cfg), str(_lm(cfg).get("model_type") or "").lower()
    if "loop_bnorm_cfg" in fs:
        return "final_norm_cfg"
    if "loop_bnorm" in fs:
        return {"ouro": "final_norm", "hrm_text": "stack_norm", "dory": "l2"}[mt]
    return None


def v_loop_inject(cfg):
    mt = str(_lm(cfg).get("model_type") or "").lower()
    if "loop_inject" not in _feats(cfg):
        return None
    return {"dory": "add", "lfm3_rd": "feedback"}.get(mt, "concat_adapter")


def v_loop_kv_mix(cfg):
    return (mg._num(_lm(cfg).get("loop_window_size")) or 64) if "loop_kv_mix" in _feats(cfg) else None


def v_early_exit(cfg):
    mt = str(_lm(cfg).get("model_type") or "").lower()
    return ("act" if mt == "lfm3_rd" else "threshold") if "early_exit" in _feats(cfg) else None


# batch 4b (residual streams and extra inputs)
def v_hc_variant(cfg):
    return next((f[3:] for f in sorted(_feats(cfg)) if f.startswith("hc_")), None)


def v_mhc_lanes(cfg):
    return mg._mhc_lanes(_lm(cfg)) if "hyper_conn" in _feats(cfg) else None


def v_res_gain(cfg):
    fs = _feats(cfg)
    return "gain_bias" if "res_bias" in fs else "gain" if "res_gain" in fs else "gate" if "res_gate_ms" in fs else None


def v_engram_kind(cfg):
    return next((f for f in sorted(_feats(cfg)) if f in mg.ENGRAM_KINDS and f != "engram_qwen4"), None)


def v_ve_kind(cfg):
    k = mg._ve_kind(_lm(cfg), _feats(cfg))
    return k if k not in ("", "ve") else None              # ("ve": the key-path families, unchanged)


def v_attn_res_sig(cfg):
    s = mg._attn_res_sig(_lm(cfg), _feats(cfg))
    return [int(x) for x in s.split("/")] if s and s != "-1/-1" else None


def v_cfrd_bias(cfg):
    fs = _feats(cfg)
    return "bias_qkv+bias_out" if {"bias_qkv", "bias_out"} <= fs else None


APPLIED = {
    "bias_qkv_by_code_unlisted": v_bias_qkv,
    "bias_qkv_table_wrong": v_bias_qkv_table_wrong,
    "bias_out_by_code_unlisted": v_bias_out,
    "parallel_by_code_unlisted": lambda cfg: _slot(cfg, "block"),
    "nanochat_port_unlisted": v_nanochat,
    "tie_default_unread": v_tied,
    "config_default_unread": v_config_default,
    "n_kv_heads_key_unread": v_kv_heads,
    "head_dim_key_unread": v_head_dim,
    "switch_cache_layers_counted": v_layers,
    "ungated_ffn_type_unlisted": lambda cfg: _slot(cfg, "ffn"),
    "norm_fallback_rmsnorm_wrong": lambda cfg: _slot(cfg, "norm"),
    "mm200_core_keys_unread": v_mm200,
    "not_a_causal_lm": v_not_lm,
    "no_model_type_longcat": v_model_type,
    "remote_code_not_config_family": v_skip,
    "audio_key_read_as_lm": v_audio_qwen2,
    "breeze_backbone_config_unread": v_breeze,
    "text_encoder_input_embeds": feat("enc_input"),
    "cross_attention_decoder_unflagged": feat("cross_attn"),
    "fake_quant_forward": feat("fake_quant"),
    "factorized_embedding_proj": v_emb_rank,
    "bidirectional_flag_true": lambda cfg: _slot(cfg, "gen"),
    "prefix_lm_mask": lambda cfg: _slot(cfg, "gen"),
    "embedding_norm": v_emb_norm,
    "norm_placement_keys_unread": v_placement,
    "grouped_rmsnorm": v_norm_groups,
    "gated_prenorm_lowrank": feat("gated_norm"),
    "multimax_logits": v_multimax,
    "dsv4_layer_types_schema": v_dsv4,
    # batch 2a
    "out_gate_by_code_unlisted": v_og_kind,
    "out_gate_headwise": v_og_head,
    "out_gate_activation": v_og_act,
    "out_gate_key_unread": v_og_kind,
    "qknorm_by_code_unlisted": lambda cfg: _slot(cfg, "qknorm"),
    "qknorm_always_remote_lacks": lambda cfg: _slot(cfg, "qknorm"),
    "qknorm_fixed_gain": lambda cfg: _slot(cfg, "qknorm"),
    "glm5_next_mla_no_head_qknorm": lambda cfg: _slot(cfg, "qknorm"),
    "dsv41_q_head_norm_absent": lambda cfg: _slot(cfg, "qknorm"),
    "mla_k_rope_norm": lambda cfg: _slot(cfg, "qknorm"),
    "attn_output_head_norm": lambda cfg: "head_free" if "onorm" in _feats(cfg) else None,
    "sink_key_query_dependent": lambda cfg: _slot(cfg, "sink"),
    # batch 2b
    "position_query_temperature": v_q_temp,
    "dynamic_ntk_in_context": v_dyn_ntk,
    "local_rope_theta_unread": _dim("rope_sched"),
    "rope_cap_zero_tail_table": _dim("rope_cap"),
    "partial_rope_keys_unread": v_partial_rope,
    "rotary_dim_ignored": _dim("rope_sched"),
    "use_mla_false_rope_dims": _dim("rope_sched"),
    "falcon_alibi_slope_scale": v_alibi_scale,
    "mpt_remote_alibi_default_false": lambda cfg: _slot(cfg, "pos"),
    "use_mem_rope_unread": lambda cfg: _slot(cfg, "pos"),
    "nope_layer_keys_unread": v_no_rope_layers,
    "nope_full_layers_unlisted": v_no_rope_layers,
    "rope_embeddings_flag_unread": lambda cfg: _slot(cfg, "pos"),
    "fope_positions": lambda cfg: "fope" if "fope" in _feats(cfg) else None,
    "content_warped_rope": lambda cfg: "rope_warp" if "rope_warp" in _feats(cfg) else None,
    # batch 2c (blocksparse_mask: a no-op by a design decision -- Phi-3-small is dense, as DSA;
    # sliding_window_backend_dependent / layer_types_all_full_window_ignored: no change, handled in effect by p5)
    "sliding_window_keys_unread": v_sliding_layout,
    "chunked_attention_unread": v_chunked,
    "swa_window_plus_one": _dim("eff_window"),
    "swa_prefill_key_truncation": featv("swa_trunc"),
    "doge_dynamic_mask": featv("dyn_mask"),
    "eos_document_mask": featv("doc_mask"),
    "code_ignores_window_and_sink": v_window_sink,
    "attn_output_saturation": featv("attn_out_sat"),
    # batch 2d (b_kv_compress_host: the wall is kept, no code change; d_transitivity_kv_heads: a relation fix, applied
    # in batch 2b under d_transitivity_rot_gate -- test_rot_gate_cap_above_sched_is_its_class)
    "per_layer_head_shapes": v_kv_shapes,
    "v_head_dim_narrow": v_v_head,
    "k_eq_v_by_code": lambda cfg: mg._k_eq_v(_lm(cfg)),
    "grouped_lowrank_oproj": v_o_groups,
    "vha_head_mix": featv("vha"),
    "diff_attn": featv("diff_attn"),
    "xsa_self_value_projection": v_xsa,
    "canon_conv_layers": lambda cfg: (_dims(cfg) or {}).get("canon") or None,
    "cca_conv_qk_attention": featv("cca"),
    "moa_routed_attention_heads": v_moa,
    "memory_attention": v_mem,
    "mova_routed_values": v_mova,
    "mla_query_raw_latent_skip": featv("mla_q_raw"),
    "step3_mfa_query_latent_norm": lambda cfg: (_dims(cfg) or {}).get("q_lat_norm") or None,
    "hybrid_mla_layers_unread": v_shared_mla,
    "ernie_mla_layer_dims_unread": v_ernie_mla,
    "kv_compress_variant": v_kv_compress,
    "cyrah_attn_impl_unread": v_cyrah,
    "b_clip_qkv_host": v_clip_host,
    # batch 3a (b_hash_route_host: the wall is kept, its mlp_layer_types reading came with dsv4_layer_types_schema;
    # b_hash_route_target_learned_host: the rule recorded, no snapshot target qualifies; spot_phimoe_dup: a relation
    # fix, test_spot_phimoe_dup)
    "renorm_by_code_unlisted": lambda cfg: v_renorm(cfg),
    "router_score_unread": lambda cfg: v_score(cfg),
    "router_bias_unread": lambda cfg: v_router_bias(cfg),
    "group_mask_zero_fill": lambda cfg: _sigv(cfg, lambda s: s["group_cap"] if s["group_cap"] == "top2sum_zero" else None),
    "router_group_balanced": lambda cfg: _sigv(cfg, lambda s: s["group_cap"] if s["group_cap"] == "per_group" else None),
    "dual_sigmoid_router_sum": lambda cfg: _sigv(cfg, lambda s: s["score"] if s["score"] == "sigmoid2" else None),
    "hash_route_by_code": feat("hash_route"),
    "mlp_router": lambda cfg: "tanh_mlp" if "router_mlp" in _feats(cfg) else None,
    "moe_capacity_drop": feat("moe_capacity"),
    "block_routing": feat("block_routing"),
    "gemma4_branch_post_norms": lambda cfg: _sigv(cfg, lambda s: "branch_norm" if s["branch"] else None),
    "kimi_latent_moe_norm": lambda cfg: _sigv(cfg, lambda s: [s["latent_norm"].rstrip("?"), s["latent"]]
                                              if s["latent_norm"] else None),
    "expert_output_norm": lambda cfg: "input_scaled_rms" if "expert_norm" in _feats(cfg) else None,
    "identity_experts": lambda cfg: mg._zero_experts(_lm(cfg)) or None,
    "shared_gate_undetected": feat("shared_gate"),
    "shared_expert_code_fixed_unread": lambda cfg: _sigv(cfg, lambda s: s["shared"] or None),
    "dense_first_key_unread": lambda cfg: _sigv(cfg, lambda s: s["n_dense"] or None),
    "hierarchical_moe_unread": lambda cfg: _sigv(cfg, lambda s: [*map(int, s["score"][4:].split("x")), s["ew"]]
                                                 if s["score"].startswith("hier") else None),
    "arctic_dense_residual_parallel_moe": lambda cfg: _sigv(cfg, lambda s: s["par_res"] or None),
    "router_feats_on_dense_model": lambda cfg: v_router_dense(cfg),
    # batch 3b (act_clip_order: a non-issue, no change; the spot-check / notes relation fixes: test_batch3b_*)
    "act_clip_per_layer_list": v_clip_list,
    "b_act_clip_host": v_act_clip_host,
    "ffn_width_keys_unread": v_ffn_width,
    "hadamard_mlp_unlisted": lambda cfg: _slot(cfg, "ffn"),
    "ffn_inner_norm": lambda cfg: ({"dory": "l2_free", "bitnet": "rms_w"}.get(str(_lm(cfg).get("model_type")).lower())
                                   if "ffn_norm" in _feats(cfg) else None),
    # batch 4a (d_transitivity_depth / d_transitivity_mixer_cla: relation fixes, test_batch4a_relation_fixes)
    "gdn_out_gate_act": lambda cfg: "sigmoid" if _fam(cfg) == "gdn_sig" else None,
    "kda_classified_as_gdn": _fam,
    "fla_num_heads_unplaced": v_heads,
    "gla_rc_dims_unread": _rc(("rc_h", "rc_k", "rc_v", "rc_kh", "rc_conv", "rc_rank")),
    "jet_dims_unread": _rc(("rc_v", "rc_rank")),
    "cyrah_shortconv_is_lfm2": _fam,
    "attn_feats_without_attention": v_no_attention,
    "hybrid_layer_schedule_unread": v_schedule,
    "loop_depth_unread": _rc(("param_layers", "eff_layers")),
    "loop_split_depth_miscount": _rc(("param_layers", "eff_layers")),
    "loop_boundary_norm": v_loop_bnorm,
    "loop_input_injection": v_loop_inject,
    "loop_kv_share_gated_mix": v_loop_kv_mix,
    "hrm_hierarchical_loop": _rc(("param_layers", "eff_layers")),
    "per_token_early_exit": v_early_exit,
    "lfm3rd_recurrent_controller": feat("rd_controller"),
    # batch 4b (b_selfcond_norm_escape: the wall is kept, no code change -- every snapshot selfcond_norm host is tied)
    "hyper_conn_variant": v_hc_variant,
    "hyper_conn_key_unread": v_mhc_lanes,
    "ceno_megatron_stream_renorm": featv("post_ln"),
    "residual_gain_vector": v_res_gain,
    "backout_residual": feat("backout"),
    "ngpt_sphere_residual": featv("ngpt"),
    "qwen4_ple_reads_ngrams": featv("engram_qwen4"),
    "ngram_memory_unread": v_engram_kind,
    "value_emb_keys_unread": v_ve_kind,
    "attn_res_keys_unread": v_attn_res_sig,
    "cfrd_film_norm_shift": v_cfrd_bias,
    "cfrd_sigmoid_residual_gate": featv("res_gate"),
    "bolmo_byte_patch_hierarchy": featv("byte_patch"),
}
# flag repos the spec's value cannot be read for: unplaced either way (no positional / head keys), so classify gives no
# slot (their _feats / dims-free values are still checked where the value function does not need classify)
UNPLACED_OK = {"parler-tts/parler_tts_mini_v0.1",
               "allenai/MolmoE-1B-0924"}     # (batch 3a, renorm_by_code_unlisted: a carrier the relation does not place)


def _want(fx, repo):
    cfg = _snap()[repo]
    return getattr(D, fx["detector"])(cfg, r3util.model_type_of(cfg))


def _norm(v):
    return tuple(_norm(x) for x in v) if isinstance(v, (list, tuple)) else v


@pytest.mark.parametrize("spec", sorted(APPLIED))
def test_flag_repos_get_the_detector_value(spec):
    fx, val = FIXTURES[spec], APPLIED[spec]
    checked = 0
    for repo, want in fx["flag"].items():
        if repo not in _snap():
            continue
        got = val(_snap()[repo])
        if got == UNPLACED:
            assert repo in UNPLACED_OK, f"{spec}: {repo} unplaced"
            continue
        det = _want(fx, repo)
        assert det, f"{spec}: the detector no longer flags {repo}"
        if want is True:
            assert got, f"{spec}: {repo} gives {got!r}, want a truthy value"
        else:
            assert _norm(got) == _norm(want), f"{spec}: {repo} gives {got!r}, want {want!r}"
        checked += 1
    assert checked, f"{spec}: no flag repo checked"


@pytest.mark.parametrize("spec", sorted(APPLIED))
def test_not_flag_repos_unchanged(spec):
    fx, val = FIXTURES[spec], APPLIED[spec]
    for repo in fx["not_flag"]:
        if repo not in _snap():
            continue
        cfg = _snap()[repo]
        on = val(cfg)
        with off(spec):
            was = val(cfg)
        assert _norm(on) == _norm(was), f"{spec}: not-flag {repo} changed {was!r} -> {on!r}"


@pytest.mark.parametrize("spec", sorted(APPLIED))
def test_spec_is_gated(spec):
    fx, val = FIXTURES[spec], APPLIED[spec]
    moved = []
    for repo in fx["flag"]:
        if repo not in _snap():
            continue
        cfg = _snap()[repo]
        on = val(cfg)
        with off(spec):
            was = val(cfg)
        if _norm(on) != _norm(was):
            moved.append(repo)
    assert moved, f"{spec}: R3_OFF does not bring any old value back"


def test_r3_tables_follow_r3_off():
    # the tables' own storage is the live content, so CPython's set / dict fast paths (set(x), plain & x, {**x})
    # see the same thing as `in`
    assert "glm_superexpert" in mg.SKIP_TYPES and {"glm_superexpert"} & mg.SKIP_TYPES
    assert "glm_superexpert" in set(mg.SKIP_TYPES) and "glm_superexpert" in frozenset(mg.SKIP_TYPES)
    assert "gpt_neox_japanese" not in set(mg.BIAS_QKV_TYPES)
    assert {**mg.NORM_BY_CODE}.get("parler_tts_decoder") == "LayerNorm"
    with off("remote_code_not_config_family", "bias_qkv_table_wrong", "norm_fallback_rmsnorm_wrong"):
        assert "glm_superexpert" not in mg.SKIP_TYPES and not ({"glm_superexpert"} & mg.SKIP_TYPES)
        assert "glm_superexpert" not in frozenset(mg.SKIP_TYPES)
        assert "gpt_neox_japanese" in set(mg.BIAS_QKV_TYPES)
        assert "parler_tts_decoder" not in dict(mg.NORM_BY_CODE)
    assert "glm_superexpert" in frozenset(mg.SKIP_TYPES)
    # a rebinding of R3_OFF to a plain set is honoured at the next call
    saved = mg.R3_OFF
    try:
        mg.R3_OFF = {"remote_code_not_config_family"}
        assert "glm_superexpert" not in mg.SKIP_TYPES
    finally:
        mg.R3_OFF = saved
    assert "glm_superexpert" in mg.SKIP_TYPES


def test_batch1_pair_rules():
    """The batch-1 pair rules on the snapshot points (emul_matrix payable): an embedding-norm target is walled from a
    plain host (no emb_fold_untied yet), a grouped-RMSNorm target from a full-width host, a Parler decoder (cross_attn)
    from decoder-only hosts, the factorized OPT-350m keeps only factorized hosts."""
    from graph.currency_curve import points
    P = points()
    ok = mg.emul_matrix(P, mode="payable")
    idx = {m: i for i, p in enumerate(P) for m in p["members"]}

    def hosts(name):
        i = idx[name]
        return {P[j]["members"][0] for j in range(len(P)) if ok[i, j] and j != i}

    for name, why in (("bigscience/bloom-560m", "emb_norm"), ("IFM/K2-Horizon-7B", "norm_groups"),
                      ("parler-tts/parler-tts-mini-v1", "feats")):
        i = idx[name]
        for j in range(len(P)):
            if ok[i, j] and j != i:
                pt, ph = P[i], P[j]
                if why == "emb_norm":
                    assert ph["dims"].get("emb_norm") == pt["dims"]["emb_norm"], (name, ph["members"][0])
                elif why == "norm_groups":
                    assert ph["dims"].get("norm_groups", 1) >= pt["dims"]["norm_groups"], (name, ph["members"][0])
                else:
                    assert "cross_attn" in ph["slots"]["feats"].split("+"), (name, ph["members"][0])
    assert hosts("facebook/opt-350m") <= {P[j]["members"][0] for j in range(len(P)) if P[j]["dims"].get("emb_rank")}


def test_catalog_r3_branches_are_live():
    """Negative control for the round-3 reduction-catalog instances (verify/spot/catalog/r3.py): each new reference-
    decoder branch, at random weights, changes the logits, so the switch-off constructions are not vacuous (the
    encoder-fed text input reads LATER text ids at an earlier position: it is bidirectional)."""
    from verify.spot.catalog.r3 import branch_is_live
    live = branch_is_live()
    assert all(v > 1e-3 for v in live.values()), live


def test_batch2a_pair_rules():
    """The batch-2a rules on the snapshot points (emul_matrix payable): an output-gated target is hosted only by a host
    with the same gate key (granularity, bias, activation: every FEAT_NEST candidate waits for its theorem), a
    softplus / silu gate host plays no ungated target, IQuest-Q1's key sink and Muse-Spark's o-norm are walled both ways
    from points without them, and a head_free_gain / ngpt / dsv4_kv qk-norm target needs a host of its variant."""
    import numpy as np
    from graph.currency_curve import points
    P = points()
    ok = mg.emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, False)
    fs = [set(filter(None, p["slots"]["feats"].split("+"))) for p in P]
    og = [next((x for x in f if x.startswith("out_gate") and x != "out_gate_post"), None) for f in fs]
    assert {k for k in og if k} >= {"out_gate", "out_gate_head", "out_gate_head_softplus", "out_gate_softplus"}
    for i, j in zip(*np.nonzero(ok)):
        name = (P[i]["members"][0], P[j]["members"][0])
        if og[i]:
            assert og[j] == og[i], (name, og[i], og[j])
        if og[j] and og[j].endswith(("_softplus", "_silu")):
            assert og[i] == og[j], name
        for slot, v in (("sink", "key"), ("qknorm", "head_free_gain"), ("qknorm", "ngpt"), ("qknorm", "dsv4_kv")):
            if P[i]["slots"][slot] == v:
                assert P[j]["slots"][slot] == v, (slot, v, name)
        if P[j]["slots"]["sink"] == "key":
            assert P[i]["slots"]["sink"] == "key", name
        assert ("onorm" in fs[i]) == ("onorm" in fs[j]), name
    # the new nestings (slot level: on the snapshot the other walls of those points decide), and the walls pending
    # qkn_const_gain
    for a, b in (("head_free", "head_free_gain"), ("head_free", "head_qfree"), ("head_qfree", "head"),
                 ("head_qfree", "head_distinct")):
        assert mg.slot_reaches("qknorm", a, b, strict=True) and mg.payable_reaches("qknorm", a, b), (a, b)
    for a, b in (("head_free_gain", "head"), ("head_free_gain", "head_free"), ("head_free", "ngpt"),
                 ("dsv4_kv", "dsv4"), ("dsv4", "dsv4_kv"), ("mla_krope", "no"), ("head_qfree", "head_post")):
        assert not mg.payable_reaches("qknorm", a, b), (a, b)


def test_feat_nest(monkeypatch):
    """FEAT_NEST (PLAN decision 1): a target feature is carried by a host with a listed parent (an exact entry in every
    mode, a limit one outside the exact slice), and the parent the host spends on it is not switched off; empty today
    (every candidate waits for its theorem: FEAT_NEST_PENDING)."""
    assert not dict(mg.FEAT_NEST) and mg.FEAT_NEST_PENDING
    t, h = "out_gate_head_softplus", "out_gate_softplus"         # the parent's host side is a wall
    assert not mg.feats_ok(t, h) and not mg.feats_ok("out_gate_head", "out_gate")
    monkeypatch.setattr(mg, "FEAT_NEST", {t: ((h, "exact"),), "out_gate_head": (("out_gate", "limit"),)})
    assert mg.feats_ok(t, h) and mg.feats_ok(t, h, strict=True)
    assert mg.feats_ok("out_gate_head", "out_gate") and not mg.feats_ok("out_gate_head", "out_gate", strict=True)
    assert not mg.feats_ok("", h)                                 # a parent not spent is still switched off (a wall)
    assert mg.feats_ok("out_gate_head", "*out_gate")              # the universal host's branches nest too


def test_batch2b_pair_rules():
    """The batch-2b rules on the snapshot points (emul_matrix payable): a q_temp / q_temp_nope / fope / rope_warp target
    is hosted only by a host with the same feature, a q_temp_nope / fope host plays no target without it (walls: no
    switch-off theorem), a scaled-slope (Falcon) ALiBi target only by ALiBi hosts whose slopes include its scaled set,
    and a sinusoidal (Parler-TTS) target only by sinusoidal hosts."""
    import numpy as np
    from graph.currency_curve import points
    P = points()
    ok = mg.emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, False)
    fs = [set(filter(None, p["slots"]["feats"].split("+"))) for p in P]
    seen = set().union(*fs)
    assert {"q_temp", "q_temp_nope", "fope", "rope_warp"} <= seen
    sig = lambda d: (d["alibi_heads"], d.get("alibi_max", 8) or 8, bool(d["alibi_bf16"]), d["alibi_scale"] or 1.0)
    for i, j in zip(*np.nonzero(ok)):
        name = (P[i]["members"][0], P[j]["members"][0])
        for x in ("q_temp", "q_temp_nope", "fope", "rope_warp"):
            if x in fs[i]:
                assert x in fs[j], (x, name)
        for x in ("q_temp_nope", "fope"):
            if x in fs[j]:
                # (fr-positions, fp_nope_layer_host: Beam's NoPE layers, where the temperature runs, are zeroed for a
                # target without NoPE layers)
                assert x in fs[i] or (x == "q_temp_nope" and "nope_layers" not in fs[i]
                                      and mg._r3("fp_nope_layer_host")), (x, name)
        if P[i]["slots"]["pos"] == "sinusoidal":
            # (fr-positions, fp_sinusoid_table: a learned table holds the sinusoid; the start-token rule plays it as one)
            assert P[j]["slots"]["pos"] in ("sinusoidal", "learned_abs", "rope", "partial_rope", "nope"), name
        ti, tj = P[i]["dims"], P[j]["dims"]
        if P[i]["slots"]["pos"] == P[j]["slots"]["pos"] == "alibi" and ti.get("alibi_scale", 1.0) != 1.0:
            assert mg.alibi_nests(sig(ti), sig(tj)), name
    # the scaled Falcon slopes are no BLOOM slope set (2 heads, hd 16: {2^-6, 2^-10}), but BLOOM's 2-head set
    # {2^-4, 2^-8} is the scale-1/4 set of >= 4 heads
    assert not mg.alibi_nests((2, 8, True, 0.25), (16, 8, False, 1.0))
    assert mg.alibi_nests((2, 8, False, 1.0), (4, 8, True, 0.25))
    assert not mg.alibi_nests((2, 8, False, 1.0), (3, 8, True, 0.25))


def test_context_gate_caps_both_sides():
    """Design decision 3 (as for DSA): under USE_CONTEXT a context-gated point is credited only up to its
    gate, as a target and as a host; the gate changes nothing with USE_CONTEXT off (the headline)."""
    import numpy as np
    from verify.z3_relation import _plain_point
    a = _plain_point(max_context=32768, eff_window=8192, pos_gate=0)          # (the windows equal: no depth charge)
    q = _plain_point(max_context=32768, eff_window=8192, pos_gate=8192)       # Qwen-7B: seq_length 8192
    b = _plain_point(max_context=8192, eff_window=8192, pos_gate=0)
    P = [a, q, b]
    assert mg.emul_matrix(P, mode="payable").all()                           # USE_CONTEXT off
    saved = mg.USE_CONTEXT
    try:
        mg.USE_CONTEXT = True
        ok = mg.emul_matrix(P, mode="payable")
        assert ok[1, 2] and ok[2, 1]                  # the gated target on an 8k host, and the 8k target on it
        assert not ok[0, 1] and not ok[0, 2]          # a 32k target is hosted by neither
        assert ok[1, 0] and ok[2, 0]
        S = ok.astype(np.int32)
        assert not (((S @ S) > 0) & ~ok).any()
        with off("position_query_temperature", "dynamic_ntk_in_context"):
            ok0 = mg.emul_matrix(P, mode="payable")
        assert ok0[0, 1] and not ok0[1, 2]
    finally:
        mg.USE_CONTEXT = saved


def test_rot_gate_cap_above_sched_is_its_class():
    """d_transitivity_rot_gate: the synthetic triple (verify/r3/specs/d_transitivity_rot_gate.md) -- A full RoPE (4,),
    B fixed with sched (2,) and cap (4,) (a proportional table), C fixed partial (2,): A <= B and, with the fix, not
    B <= C (B stands for its cap), so no violation; with the fix off the violation is back."""
    from verify.z3_relation import _plain_point
    base = dict(head_dim=4, kv_heads=1, q_per_kv=1, qheads=1, kv="MHA", kv_cap=4, q_cap=4)
    A = _plain_point(pos="rope", rope_sched=(4,), rope_cap=(4,), **base)
    B = _plain_point(pos="partial_rope", rope_sched=(2,), rope_cap=(4,), **base)
    C = _plain_point(pos="partial_rope", rope_sched=(2,), rope_cap=(2,), **base)
    for mode in ("exact", "payable", "mixgap"):
        ok = mg.emul_matrix([A, B, C], mode=mode)
        assert ok[0, 1] and not ok[0, 2] and not ok[1, 2], mode
        assert ok.diagonal().all()
        with off("d_transitivity_rot_gate"):
            ok = mg.emul_matrix([A, B, C], mode=mode)
        assert ok[0, 1] and ok[1, 2] and not ok[0, 2], mode


def test_batch2c_pair_rules():
    """The batch-2c rules on the snapshot points (emul_matrix payable): a swa_trunc / attn_out_sat point is walled both
    ways from points without it, a dyn_mask / doc_mask target is hosted only by a host with the same feature (their
    host sides are free: catalog feature:dyn_mask:host / feature:doc_mask:host); Phi-3-small's block-sparse layers stay
    dense (blocksparse_mask: a design decision, as DSA -- no layer kind, no feature), Llama-4's chunked layers stay
    FU (only eff_window moves), and cost_wise_gemma is not window-free (the conservative choice)."""
    import numpy as np
    from graph.currency_curve import points
    P = points()
    ok = mg.emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, False)
    fs = [set(filter(None, p["slots"]["feats"].split("+"))) for p in P]
    assert {"swa_trunc", "attn_out_sat", "dyn_mask", "doc_mask"} <= set().union(*fs)
    for i, j in zip(*np.nonzero(ok)):
        name = (P[i]["members"][0], P[j]["members"][0])
        for x in ("swa_trunc", "attn_out_sat", "dyn_mask", "doc_mask"):
            if x in fs[i]:
                assert x in fs[j], (x, name)
        for x in ("swa_trunc", "attn_out_sat"):
            if x in fs[j]:
                assert x in fs[i], (x, name)
    snap = _snap()
    d = _lm(snap["microsoft/Phi-3-small-128k-instruct"])
    kc = mg._attn_counts(d, mg._layer_count(d))[4]
    assert kc["FU"] == 32 and not kc["LU"] and _cls(snap["microsoft/Phi-3-small-128k-instruct"])["window"] == "full"
    d = _lm(snap["lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-8bit"])
    assert mg._attn_counts(d, mg._layer_count(d))[4]["FU"] == 48
    assert "cost_wise_gemma" not in mg.WINDOW_FREE_TYPES


def _point(repo):
    """The snapshot point (classes deduped) whose members include repo."""
    from graph.currency_curve import points
    return next(p for p in points() if repo in p["members"])


def test_batch2d_pair_rules():
    """The batch-2d feature rules on the snapshot points (emul_matrix payable): xsa / diff_attn / cca / mova / mla_qmix /
    kv_compress_v41 are walled both ways from points without them (V4.1's compressor from V4's too); a vha / xsa_scaled /
    canon / moa / mem_attn / mla_q_raw target needs a host with the feature (their host sides are free: catalog
    feature:<key>:host); a clip_qkv host now plays unclamped targets (b_clip_qkv_host: the host config lever clip_qkv
    None), and every pair of Canon / MoA / q-latent-norm points satisfies its dims rule."""
    import numpy as np
    from graph.currency_curve import points
    P = points()
    ok = mg.emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, False)
    fs = [set(filter(None, p["slots"]["feats"].split("+"))) for p in P]
    both = ("xsa", "diff_attn", "cca", "mova", "mla_qmix", "kv_compress_v41", "kv_compress")
    tgt = ("vha", "xsa_scaled", "canon", "moa", "mem_attn", "mla_q_raw")
    assert {"xsa", "kv_compress_v41", "canon", "mem_attn", "clip_qkv"} <= set().union(*fs)
    for i, j in zip(*np.nonzero(ok)):
        name = (P[i]["members"][0], P[j]["members"][0])
        for x in both:
            assert (x in fs[i]) == (x in fs[j]), (x, name)
        for x in tgt:
            if x in fs[i]:
                assert x in fs[j], (x, name)
        di, dj = P[i]["dims"], P[j]["dims"]
        assert di["q_lat_norm"] == dj["q_lat_norm"], name
        if di["canon"] and dj["canon"]:
            assert mg._canon_covers(di["canon"], dj["canon"]), name
        if di["moa_e"] and dj["moa_e"]:
            assert (di["moa_e"], di["moa_k"]) == (dj["moa_e"], dj["moa_k"]), name
    clip = np.array(["clip_qkv" in f for f in fs])
    assert (ok & ~clip[:, None] & clip[None, :]).any()            # clipped hosts, unclipped targets
    assert not (ok & clip[:, None] & ~clip[None, :]).any()        # the target side stays walled
    with off("b_clip_qkv_host"):
        assert mg.FEATURES["clip_qkv"][1] == "wall"


def test_kv_shapes_counted_and_v_width():
    """per_layer_head_shapes: a host offers each attention shape only on its own layers -- DeciLM-7B has its 4-KV-head
    shape on 7 of 32 layers, so it no longer hosts a uniform 4 x 8 stack of 32 layers (it did when every target shape
    only needed SOME host shape); v_head_dim_narrow: MiMo-V2.5's V heads are 128 wide under 192-wide q / k, so a target
    whose V is 192 wide no longer fits, and the packing factor reads the V width."""
    import copy
    host = _point("Deci/DeciLM-7B-instruct")
    assert host["dims"]["kv_shapes"] and host["dims"]["kv_shape_n"] == (11, 14, 7)
    tgt = copy.deepcopy(host)
    tgt["dims"].update(kv_shapes=(), kv_shape_n=(), kv_heads=4, q_per_kv=8)
    assert not mg.kv_heads_fit([tgt, host])[0, 1]
    assert mg.kv_heads_fit([tgt, host], mode="capacity")[0, 1]   # (capacity: the uncounted fit)
    with off("per_layer_head_shapes"):
        assert mg.kv_heads_fit([tgt, host])[0, 1]
    mimo = _point("XiaomiMiMo/MiMo-V2.5")
    assert {s[3] for s in mimo["dims"]["kv_shapes"]} == {128} and mimo["dims"]["v_head"] == 128
    wide = copy.deepcopy(mimo)
    wide["dims"]["kv_shapes"] = tuple(s[:3] + (192,) for s in mimo["dims"]["kv_shapes"])
    assert not mg.kv_heads_fit([wide, mimo])[0, 1] and mg.kv_heads_fit([mimo, wide])[0, 1]
    with off("v_head_dim_narrow"):
        assert mg.kv_heads_fit([wide, mimo])[0, 1]
    # Domyn-Edge: its 256-wide full layers and 64-wide RoPE layers (head_sizes), the rot gate over the 64 (rope_dim)
    dom = _point("domyn/Domyn-Edge-Base")["dims"]
    assert dom["head_dim"] == 256 and dom["kv_shapes"] == ((8, 4, 64, 64), (8, 4, 256, 256))
    assert dom["kv_shape_n"] == (24, 12) and dom["rope_sched"] == (64, -256)


def test_grouped_oproj_rule():
    """grouped_lowrank_oproj (_r3_masks_2d): a grouped host fits a target's o-map group by group -- DeepSeek-V4-Pro
    (16 groups of 8 heads, rank 1024) plays V4-Flash (8 groups of 8, 1024); a dense host plays any grouped target; a
    grouped host plays a dense target only inside one group with the rank (8 heads in a group of 8, rank 1024 >= min(8
    x 512, hidden 1024)), not 16 heads, nor 8 heads in a group of 4)."""
    import numpy as np
    mk = lambda g, r, H, hd, hid: {"dims": dict(o_groups=g, o_rank=r, qheads=H, head_dim=hd)}
    P = [mk(8, 1024, 64, 512, 4096), mk(16, 1024, 128, 512, 7168), mk(0, 0, 8, 512, 4096), mk(0, 0, 16, 512, 4096),
         mk(2, 64, 8, 64, 512)]
    hidden = np.array([4096, 7168, 1024, 4096, 512])
    m = mg._r3_masks_2d(P, hidden)["o_proj groups"]
    assert m[0, 1] and m[1, 2] and m[2, 0] and not m[3, 0] and not m[0, 4]
    assert not m[2, 4]                                            # 8 heads in a 4-head group: no
    assert all(m[i, i] for i in range(len(P)))


# --- batch 3a: MoE and router ----------------------------------------------------------------------------------------
def _msig(E=8, k=2, ew=16, shared=0, renorm="R", renorm_cap=("R",), score="softmax", score_cap=("softmax",), ng=0,
          tg=0, gcap="", n_moe=4, n_dense=0, dense_w=0, latent=0, bias=0, **r3):
    """A moe_sig tuple in MOE_SIG order, the round-3 fields by name (MOE_SIG_R3)."""
    f = dict(zip(mg.MOE_SIG_R3, mg.MOE_SIG_R3_DENSE))
    f.update(r3)
    return (E, k, ew, shared, renorm, tuple(renorm_cap), score, tuple(score_cap), ng, tg, gcap, n_moe, n_dense, dense_w,
            latent, bias) + tuple(f[x] for x in mg.MOE_SIG_R3)


def _why(t, h, ht=64, hh=64, ft=128, fh=128):
    return mg.moe_why((t, ht, ft), (h, hh, fh))


def test_spot_phimoe_dup():
    """spot_phimoe_dup (verify/spot/SPOT.md, catalog fm_phimoe_dup2): a SparseMixer target takes no m > 1 copies (its
    second pick masks out only the first expert's index); m = 1 (a wider host) stays."""
    sm = dict(renorm="sparsemixer", renorm_cap=("sparsemixer",), score="sparsemixer", score_cap=("sparsemixer",))
    t, h2, h1 = _msig(E=4, k=2, **sm), _msig(E=8, k=4, **sm), _msig(E=4, k=2, ew=32, **sm)
    assert _why(t, h2) and _why(t, h1) is None
    with off("spot_phimoe_dup"):
        assert _why(t, h2) is None


def test_batch3a_moe_rules():
    """The batch-3a routing rules on synthetic signatures (moe_why): 0-filled / per-group group codes only with their own
    code and the target's experts, a block-routed target only on the same block routing (a block host plays per-token
    targets: the capacity lever), identity slots H = m T, latent-norm sites (LATENT_NORM_PLAYS: the Kimi-K3 flag plays
    norm-free targets, ERNIE / Muse do not), branch post-norms both ways, Arctic's host only for dense layers."""
    sg = dict(score="sigmoid", score_cap=("sigmoid",), bias=1)
    z = lambda **kw: _msig(ng=4, tg=2, gcap="top2sum_zero", **{**sg, **kw})
    inf = lambda **kw: _msig(ng=4, tg=2, gcap="top2sum", **{**sg, **kw})
    assert _why(z(), z(ew=32)) is None                                      # zero <- zero, the same experts
    assert _why(z(), z(E=16)) and _why(z(), inf()) and _why(inf(), z())     # extras / either fill across: walls
    assert _why(inf(), inf(E=16)) is None                                   # (-inf: moeR_group_copies with extras)
    pg = lambda **kw: _msig(**{**dict(E=8, k=4, ng=4, tg=4, gcap="per_group"), **kw})
    assert _why(pg(), pg(ew=32)) is None and _why(pg(), pg(E=16)) and _why(pg(), _msig(E=8, k=4))
    assert _why(_msig(E=8, k=4), pg()) is None                              # router_n_groups = 1 (host lever)
    b = lambda **kw: _msig(**{**sg, "blk": 32, "blk_cap": 4, **kw})
    assert _why(b(), b(ew=32)) is None and _why(b(), b(blk_cap=6)) and _why(b(), _msig(**sg))
    assert _why(_msig(**sg), b()) is None                                   # expert_capacity = E (host lever)
    li = lambda n, **kw: _msig(ident=n, renorm="P", renorm_cap=("P",), **kw)
    assert _why(li(4), li(4, ew=32)) is None and _why(li(4), li(2)) and _why(li(0), li(4)) and _why(li(4), li(0))
    assert _why(li(2, E=4, k=2), li(4, E=8, k=4)) is None                   # m = 2 copies of each slot
    for tn, hn, ok in (("", "sum?", True), ("sum", "sum?", True), ("sum?", "sum", False), ("", "sum", False),
                       ("", "in", False), ("in", "in", True), ("sum", "", False), ("sum?", "sum?", True)):
        assert (_why(_msig(latent=32, latent_norm=tn), _msig(latent=32, latent_norm=hn)) is None) == ok, (tn, hn)
    br = _msig(branch=1, shared=32)
    assert _why(br, br) is None and _why(br, _msig(shared=32)) and _why(_msig(), br)
    dense = _msig(E=0, k=0, ew=0, renorm="", renorm_cap=(), score="", score_cap=(), n_moe=0, n_dense=4, dense_w=48)
    assert _why(dense, br) and _why(dense, _msig(shared=32)) is None      # (k e_w + s = 64 >= 48)
    ar = _msig(par_res=64)
    assert _why(dense, ar) is None and _why(_msig(), ar)                    # dense layers on the residual MLP only
    assert _why(_msig(E=0, k=0, ew=0, renorm="", renorm_cap=(), score="", score_cap=(), n_moe=0, n_dense=4,
                      dense_w=96), ar)                                      # wider than the residual MLP
    hier = _msig(E=15, k=1, renorm="R", score="hier3x5", score_cap=("hier3x5",), shared=64)
    assert _why(hier, _msig(E=15, k=1)) and _why(_msig(E=4, k=1), hier) and _why(hier, hier) is None
    assert _why(hier, dense, hh=4096, fh=1 << 20)                           # no dense host for a two-level argmax


def test_batch3a_pair_rules():
    """The batch-3a feature rules on the snapshot points (emul_matrix payable): router_mlp / moe_capacity / expert_norm /
    moe_branch_norm walled both ways, a router_bias_logit / block_routing / par_moe_res target only on a host with the
    feature, and every MoE pair keeps the routing forms (moe_sig): the 0-fill / per-group group codes, identity slots,
    latent-norm sites, the branch flag, the two-level router."""
    import numpy as np
    from graph.currency_curve import points
    P = points()
    ok = mg.emul_matrix(P, mode="payable")
    np.fill_diagonal(ok, False)
    fs = [set(filter(None, p["slots"]["feats"].split("+"))) for p in P]
    assert {"router_mlp", "moe_capacity", "expert_norm", "moe_branch_norm", "router_bias_logit", "block_routing",
            "par_moe_res"} <= set().union(*fs)
    S = [mg._moe_fields(p["dims"]["moe_sig"]) if p["dims"].get("moe_sig") else None for p in P]
    for i, j in zip(*np.nonzero(ok)):
        name = (P[i]["members"][0], P[j]["members"][0])
        for x in ("router_mlp", "moe_capacity", "expert_norm", "moe_branch_norm"):
            assert (x in fs[i]) == (x in fs[j]), (x, name)
        for x in ("router_bias_logit", "block_routing", "par_moe_res"):
            if x in fs[i]:
                assert x in fs[j], (x, name)
        T, H = S[i], S[j]
        if not (T and H and T["E"] and H["E"]):
            continue
        assert T["branch"] == H["branch"], name
        assert H["latent_norm"] in mg.LATENT_NORM_PLAYS[T["latent_norm"]], name
        if T["ident"] or H["ident"]:
            assert H["ident"] == (H["k"] // T["k"]) * T["ident"], name
        if T["n_group"] and T["group_cap"] in ("top2sum_zero", "per_group"):
            assert H["group_cap"] == T["group_cap"] and H["E"] == T["E"], name
        if T["score"].startswith("hier") or T["score"] == "sigmoid2":
            assert H["score"] == T["score"], name


# ---- batch 3b: FFN widths, activation clamps and the spot-check / notes relation fixes ------------------------------

@functools.lru_cache(maxsize=None)
def _payable():
    from graph.currency_curve import points
    P = points()
    return P, mg.emul_matrix(P, mode="payable")


def test_batch3b_pair_rules():
    """The batch-3b rules on the snapshot points (emul_matrix payable): no gated / MoE target on a vanilla host
    (spot_glu_vanilla); a vanilla target on a gated host and an up-biased target only on an up-biased host
    (spot_vanilla_glu_bias); HadamardMLPs only on themselves at equal n; ReLU <- swigluoai only with the host's up bias
    (w3_relu_oai_bias); a swigluoai host plays only SiLU / ReLU / swigluoai targets (spot_act_width_oai: no act-width
    fit); a LayerNorm-beta target on an RMSNorm host only with an lm_head bias (spot_comp_removed); a Dory target only
    on a Dory host (ffn_norm); and act_clip's host side is used (a clamped host plays an unclamped target)."""
    import numpy as np
    P, ok = _payable()
    ok = ok.copy()
    np.fill_diagonal(ok, False)
    sl = lambda s: np.array([p["slots"][s] for p in P])
    dm = lambda k: np.array([p["dims"].get(k, 0) for p in P])
    ffn, act, norm = sl("ffn"), sl("act"), sl("norm")
    bu, hn = dm("bias_up") > 0, dm("hada_n")
    fs = [set(filter(None, p["slots"]["feats"].split("+"))) for p in P]
    assert (hn > 0).sum() >= 2 and "ffn_norm" in set().union(*fs)
    ti, hj = np.nonzero(ok)
    assert not ((ffn[hj] == "vanilla") & (ffn[ti] != "vanilla")).any()
    assert not ((ffn[ti] == "vanilla") & (ffn[hj] != "vanilla") & ~bu[hj]).any()
    assert not (bu[ti] & ~bu[hj]).any()
    assert (hn[ti] == hn[hj]).all()
    assert not ((act[ti] == "ReLU") & (act[hj] == "swigluoai") & ~bu[hj]).any()
    assert np.isin(act[ti][act[hj] == "swigluoai"], ("SiLU", "ReLU", "swigluoai")).all()
    lnb = (norm == "LayerNorm") & (dm("ln_beta") > 0)
    rms = np.array(["LayerNorm" not in mg.branches(x) for x in norm])
    # (fr-resnorm, fr_ln_beta_readout: or a dominated final readout -- an untied host with FFN-input and output biases
    # and one more hidden channel; tests/test_fr_resnorm.py)
    ro = (dm("ffn_bias") > 0) & np.array([("bias_out" in f) and ("tied_emb" not in f) for f in fs])
    assert not (lnb[ti] & rms[hj] & ~(dm("lm_head_bias") > 0)[hj] & ~ro[hj]).any()
    with off("fr_ln_beta_readout"):
        ok0 = mg.emul_matrix(P, mode="payable")
    np.fill_diagonal(ok0, False)
    t0, h0 = np.nonzero(ok0)
    assert not (lnb[t0] & rms[h0] & ~(dm("lm_head_bias") > 0)[h0]).any()
    for i, j in zip(ti, hj):
        if "ffn_norm" in fs[i]:
            assert "ffn_norm" in fs[j]
    clip = np.array(["act_clip" in f for f in fs])
    assert (clip[hj] & ~clip[ti]).any()                      # b_act_clip_host: the host config lever is used
    assert not (clip[ti] & ~clip[hj]).any()


def test_batch3b_rules_follow_their_keys():
    """Each batch-3b relation key switches its rule (R3_OFF): the act-width fit on a swigluoai host, the compensator
    ports, the up-bias / gated-vanilla walls, the ReLU <- swigluoai bias, the parallel units, the hadamard wall."""
    assert mg.act_width_k("GeLU", "swigluoai", True) == 0
    with off("spot_act_width_oai"):
        assert mg.act_width_k("GeLU", "swigluoai", True) > 0
    assert mg.act_width_k("GeLU", "SiLU|swigluoai", True) == mg.act_width_k("GeLU", "SiLU", True) > 0
    assert not mg.p5_port_live("comp_ln") and not mg.p5_port_live("comp_par")
    with off("spot_comp_removed"):
        assert mg.p5_port_live("comp_ln") and mg.p5_port_live("comp_par")
    import numpy as np
    from tests.test_pass5_rules import _edit, _llama, _pt, BIG
    tgt_v = _edit(_pt(_llama()), slots={"ffn": "vanilla"}, ffn_bias=0, bias_up=0)
    host_g = _edit(_pt(_llama(**BIG)), ffn_bias=0, bias_up=0)
    host_gb = _edit(host_g, ffn_bias=1, bias_up=1)
    tgt_g = _pt(_llama())
    host_v = _edit(_pt(_llama(**BIG)), slots={"ffn": "vanilla"}, ffn_bias=0, bias_up=0)
    em = lambda t, h: bool(mg.emul_matrix([t, h], mode="payable")[0, 1])
    assert not em(tgt_v, host_g) and em(tgt_v, host_gb)
    with off("spot_vanilla_glu_bias"):
        assert em(tgt_v, host_g)
    assert not em(tgt_g, host_v)
    with off("spot_glu_vanilla"):
        assert em(tgt_g, host_v)                          # (the old 2x-depth credit: 4 target layers, 8 host layers)
    tr = _edit(_pt(_llama()), slots={"act": "ReLU"})
    ho = _edit(_pt(_llama(**BIG)), slots={"act": "swigluoai"}, ffn_bias=0, bias_up=0)
    assert not em(tr, ho) and em(tr, _edit(ho, ffn_bias=1, bias_up=1))
    with off("w3_relu_oai_bias"):
        assert em(tr, ho)
    par = _edit(_pt(_llama()), slots={"block": "parallel"})
    seq = _pt(_llama(**BIG))
    cap = lambda: mg.emul_matrix([par, _edit(seq, ffn_total=par["dims"]["ffn_total"] + 1,
                                             ffn_active=par["dims"]["ffn_active"] + 1)], mode="capacity")[0, 1]
    assert not cap()                                      # w1_par_units: + 2 d_target units per block
    with off("w1_par_units"):
        assert cap()
    a, b = _edit(_pt(_llama()), hada_n=512), _edit(_pt(_llama(**BIG)), hada_n=1024)
    assert not em(a, b) and not em(_edit(a, hada_n=0), b) and not em(a, _edit(b, hada_n=0))
    with off("hadamard_mlp_unlisted"):
        assert em(a, b)


def test_spot_glu_vanilla_wall_costs_and_keeps():
    """spot_glu_vanilla walls every gated / MoE target on a vanilla host; the 2x depth stays for the dense GLU host
    of a top-1 MoE target (moe/dense_host_prenorm.py) -- some payable MoE <- dense GLU edge remains."""
    import numpy as np
    P, ok = _payable()
    ffn = np.array([p["slots"]["ffn"] for p in P])
    moe_t = np.isin(ffn, ("moe", "moe_shared"))
    off_ = ~np.eye(len(P), dtype=bool)
    assert (ok & off_ & moe_t[:, None] & (ffn == "glu")[None, :]).any()


def test_comp_removed_costs_no_edge():
    """spot_comp_removed (the measurement): switching the compensator credits off loses no payable edge on the snapshot
    -- every LayerNorm target on an RMSNorm host without an lm_head bias is beta-free (OLMo-1, MPT, replit), the exact
    fold of catalog r3_ln_fold_beta_free, and no parallel <= sequential pair used the parallel compensator."""
    import numpy as np
    P, _ = _payable()
    with off("fr_ln_beta_readout"):                      # (fr-resnorm's readout credits LN-beta edges of its own)
        ok = mg.emul_matrix(P, mode="payable")
        with off("spot_comp_removed"):
            ok_was = mg.emul_matrix(P, mode="payable")
    assert (ok == ok_was).all()
    norm = np.array([p["slots"]["norm"] for p in P])
    lnb = np.array([bool(p["dims"].get("ln_beta")) for p in P])
    lmb = np.array([bool(p["dims"].get("lm_head_bias")) for p in P])
    rms = np.array(["LayerNorm" not in mg.branches(x) for x in norm])
    e = ok & ~np.eye(len(P), dtype=bool) & (norm == "LayerNorm")[:, None] & rms[None, :] & ~lmb[None, :]
    assert e.any() and not (e & lnb[:, None]).any()


def test_lfm2_normalize_is_idempotent():
    """round3_inputs E: _normalize(_normalize(d)) == _normalize(d) for every LFM2-family snapshot config (the
    block_auto_adjust_ff_dim width no longer shrinks on a second pass: lfm2_normalize_idempotent)."""
    n = 0
    for repo, cfg in _snap().items():
        d = _lm(cfg)
        if not str(d.get("model_type") or "").lower().startswith("lfm"):
            continue
        n += 1
        once = mg._normalize(d)
        assert mg._normalize(once) == once, repo
    assert n >= 50
    thox = _lm(_snap()["Thox-ai/ThoxFlux-350M"])
    assert mg._normalize(mg._normalize(thox))["intermediate_size"] == 4608
    with off("lfm2_normalize_idempotent"):
        thox = _lm(_snap()["Thox-ai/ThoxFlux-350M"])                 # (one pass: 6656 -> 4608)
        assert thox["intermediate_size"] == 4608 and mg._normalize(thox)["intermediate_size"] == 3072


def test_batch4a_pair_rules():
    """The batch-4a rules on the snapshot points (emul_matrix payable): the loop-boundary walls (a target with
    loop_bnorm / loop_bnorm_cfg / loop_inject / loop_kv_mix / hrm_two_state / early_exit / rd_controller only on a host
    with it; a host with a two-sided wall only for a target with it) and the two host sides used (Nanbeige4.2's
    skip_loop_final_norm lever, LFM3-RD's step scale at 1); the new recurrent families only on their own family; Ouro /
    HRM looped at their unrolled depth, LFM3-RD and CFRD haru_2 untied."""
    import numpy as np
    P, ok = _payable()
    ok = ok.copy()
    np.fill_diagonal(ok, False)
    fs = [set(filter(None, p["slots"]["feats"].split("+"))) for p in P]
    ti, hj = np.nonzero(ok)
    has = lambda f: np.array([f in x for x in fs])
    for f in ("loop_bnorm", "loop_bnorm_cfg", "loop_inject", "loop_kv_mix", "hrm_two_state", "early_exit",
              "rd_controller", "rf_gdn_sig", "rf_kda_lb_negeig", "rf_deltanet"):
        h = has(f)
        assert h.any(), f
        assert not (h[ti] & ~h[hj]).any(), f                 # target side: a wall
        if mg.FEATURES.get(f, ("wall", "free"))[1] == "wall":
            assert not (~h[ti] & h[hj]).any(), f             # host side: a wall
    assert (~has("loop_bnorm_cfg")[ti] & has("loop_bnorm_cfg")[hj]).any()      # the host config lever is used
    assert (~has("rd_controller")[ti] & has("rd_controller")[hj]).any()        # the step scale at 1 is used
    gdn = has("rf_gdn")
    assert not (gdn[ti] & has("rf_gdn_sig")[hj] & ~gdn[hj]).any()
    pt = {r: _point(r) for r in ("ByteDance/Ouro-1.4B", "sapientinc/HRM-Text-1B", "cybertruck32489/LFM3-1.7B-RD",
                                 "alice-noa-chan/haru_2", "Lordnyx/qwen3loop-0.6b-sft-deep-supervision-v1")}
    assert [pt["ByteDance/Ouro-1.4B"]["slots"]["loop"], pt["ByteDance/Ouro-1.4B"]["dims"]["eff_layers"]] == ["looped", 96]
    assert pt["sapientinc/HRM-Text-1B"]["dims"]["eff_layers"] == 128
    assert pt["cybertruck32489/LFM3-1.7B-RD"]["slots"]["loop"] == pt["alice-noa-chan/haru_2"]["slots"]["loop"] == "untied"
    assert pt["Lordnyx/qwen3loop-0.6b-sft-deep-supervision-v1"]["dims"]["eff_layers"] == 56
    with off("loop_depth_unread"):
        assert mg.classify(_snap()["cybertruck32489/LFM3-1.7B-RD"])["loop"] == "looped"


def test_batch4a_relation_fixes():
    """d_transitivity_depth: every point's zero-edge counts are its own-KV kind sums (the "built" wf clause of
    verify/z3_relation.py); d_transitivity_mixer_cla: in mixgap a KV-sharing softmax target reaches a hybrid KV-sharing
    host through the CLA carry + the ZERO_EDGES (the synthetic triple of the spec), not in payable, and not with the key
    off."""
    pytest.importorskip("z3")
    from graph.currency_curve import points
    from verify import z3_relation as Z
    own = ("FU", "FT", "LU", "LT")
    for p in points():
        d = p["dims"]
        for w in ("eff", "param"):
            assert d.get(f"attn_{w}", 0) == sum(d.get(f"k{k}_{w}", 0) for k in own), p["members"][0]
            assert d.get(f"full_{w}", 0) == d.get(f"kFU_{w}", 0) + d.get(f"kFT_{w}", 0), p["members"][0]
    pt = Z._plain_point
    A = pt(hidden=8, crosslayer_kv="yes", attn_eff=1, attn_param=1)
    B = pt(hidden=8)
    C = pt(hidden=8, mixer="hybrid", crosslayer_kv="yes", feats="rf_mamba2_nonorm", eff_layers=2, param_layers=2,
           attn_eff=1, attn_param=1, kR_eff=1, kR_param=1, rc_rule=mg.RC_M2)
    for i, x in enumerate((A, B, C)):
        x["members"] = ["ABC"[i]]
    R = mg.emul_matrix([A, B, C], mode="mixgap")
    assert R[0, 1] and R[1, 2] and R[0, 2] and R[0, 0]
    assert not mg.emul_matrix([A, B, C], mode="payable")[0, 1]
    with off("d_transitivity_mixer_cla"):
        assert not mg.emul_matrix([A, B, C], mode="mixgap")[0, 2]


def test_batch4b_pair_rules():
    """The batch-4b rules on the snapshot points (emul_matrix payable): every new residual-stream / extra-input feature
    is a target-side wall (hc_* variants: walled both ways, the glm5_next <- dsv4 nest pending its theorem; CENO post_ln,
    res_gate_ms, ngpt, byte_patch, engram_klm: both ways), the residual-gain host side used (Doge), the signatures equal on every edge between points that both carry one, and no CFRD
    host for a target with independent biases."""
    import numpy as np
    P, ok = _payable()
    ok = ok.copy()
    np.fill_diagonal(ok, False)
    fs = [set(filter(None, p["slots"]["feats"].split("+"))) for p in P]
    ti, hj = np.nonzero(ok)
    has = lambda f: np.array([f in x for x in fs])
    new = ([f"hc_{v}" for v in mg.HC_VARIANTS] + list(mg.ENGRAM_KINDS)
           + ["res_gain", "res_bias", "res_gate_ms", "backout", "ngpt", "res_gate", "byte_patch"])
    seen = set()
    for f in new:
        h = has(f)
        if not h.any():
            continue
        seen.add(f)
        assert not (h[ti] & ~h[hj]).any(), f                 # target side: a wall
        # (fr-resnorm, fr_hc_host_off: a host of a read variant switches its streams off -- graph/model_graph.py
        # HC_HOST_OFF_VARIANTS; the other variants keep the host-side wall)
        if mg.FEATURES[f][1] == "wall" and f not in {f"hc_{v}" for v in mg.HC_HOST_OFF_VARIANTS}:
            assert not (~h[ti] & h[hj]).any(), f             # host side: a wall
    assert seen >= {"hc_dsv4", "hc_dsv41", "hc_glm5_next", "hc_qwen4_exp", "hc_needle", "engram_qwen4", "engram_needle",
                    "res_gain", "res_gate_ms", "backout", "ngpt", "res_gate", "byte_patch"}, seen
    assert not (has("hc_glm5_next")[ti] & has("hc_dsv4")[hj]).any()           # (FEAT_NEST_PENDING hc_head_mean)
    # the host switch-off is used (Doge hosts; no snapshot backout / CFRD host has a target without its feature that the
    # rest of the relation lets it play)
    assert (~has("res_gain")[ti] & has("res_gain")[hj]).any()
    ceno = np.array([any(m.startswith("CladeTeam/CENO") for m in p["members"]) for p in P])
    assert ceno.any() and all("post_ln" in fs[i] for i in np.where(ceno)[0])
    pl = has("post_ln")
    assert not (pl[ti] ^ pl[hj]).any()
    for k in ("attn_res_sig", "engram_sig", "ve_kind"):
        v = [str(p["dims"].get(k, "") or "") for p in P]
        assert any(v), k
        for i, j in zip(ti, hj):
            if v[i] and v[j]:
                assert v[i] == v[j], (k, P[i]["members"][0], P[j]["members"][0])
    bs = np.array([p["dims"].get("bias_shift", 0) > 0 for p in P])
    fb = has("bias_qkv") | has("bias_out")
    assert bs.any() and not (bs[hj] & ~bs[ti] & fb[ti]).any()


def test_batch4b_rules_follow_their_keys():
    """Each batch-4b pair rule switches with its key (R3_OFF): the AttnRes partition, the n-gram geometry, the
    value-embedding kind, the norm-shift host wall; the universal host carries GLM-5-Next / Motif / DeepSeek-V4 / V4.1
    hyper-connections and AllGDN's and V4.1's Engram."""
    from tests.test_pass5_rules import _edit, _llama, _pt, BIG
    from graph.minimal_emulator import HOST_FEATS
    t0, h0 = _pt(_llama()), _pt(_llama(**BIG))
    em = lambda t, h: bool(mg.emul_matrix([t, h], mode="payable")[0, 1])

    def pair(feat, key, a, b):
        t = _edit(t0, slots={"feats": feat}, **{key: a})
        h = _edit(h0, slots={"feats": feat}, **{key: b})
        return t, h
    for spec, feat, key, a, b in (("attn_res_keys_unread", "attn_res", "attn_res_sig", "1/0", "2/0"),
                                  ("ngram_memory_unread", "engram_needle", "engram_sig", "(1,)", "(2,)"),
                                  ("value_emb_keys_unread", "value_emb", "ve_kind", "ve_shared", "ve_layer")):
        t, h = pair(feat, key, a, b)
        assert em(*pair(feat, key, a, a)) and not em(t, h), spec
        with off(spec, "qwen4_ple_reads_ngrams"):
            assert em(t, h), spec
    tb = _edit(t0, slots={"feats": "bias_qkv"})
    hb = _edit(h0, slots={"feats": "bias_qkv+bias_out"}, bias_shift=1)
    assert not em(tb, hb) and em(_edit(tb, bias_shift=1), hb)
    with off("cfrd_film_norm_shift"):
        assert em(tb, hb)
    assert {"hyper_conn", "hc_glm5_next", "hc_motif", "engram_allgdn"} <= set(HOST_FEATS)
    # (the release: from_deepseek_v4 / from_deepseek_v41 run V4's and V4.1's hyper-connections and V4.1's Engram against
    # a transcription of the official inference code -- tests/test_deepseek_v4_host.py, test_deepseek_v41_host.py)
    assert {"hc_dsv4", "hc_dsv41", "engram"} <= set(HOST_FEATS)
    with off("ngram_memory_unread", "hyper_conn_variant"):
        assert "engram" in set(HOST_FEATS) and "engram_allgdn" not in set(HOST_FEATS)
        assert not {"hc_dsv4", "hc_dsv41"} & set(HOST_FEATS)


# --- round 3: max_context_unknown (the max context read from the config and code; unknown instead of 4096) ----------

@pytest.mark.parametrize("repo,value,source", [
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", 1048576, "alt_key"),   # model_max_length: KimiLinearConfig's attribute_map
    ("baichuan-inc/Baichuan2-13B-Chat", 4096, "alt_key"),
    ("huggyllama/llama-65b", 2048, "alt_key"),                                  # max_sequence_length
    ("RWKV/rwkv-4-169m-pile", 1024, "alt_key"),                                 # context_length (the CUDA Tmax)
    ("castorini/gaggle-nanochat-pretrained-climbmix-20260924", 2048, "alt_key"),  # sequence_len
    ("root-sys-n/Mima-Pico-15m", 512, "alt_key"),                               # block_size (modeling_mima.py:140)
    ("tiiuae/falcon-7b", 2048, "class_default"),                                # FalconConfig
    ("anandkaman/gemma3_12b_deedexp", 131072, "class_default"),                 # Gemma3TextConfig
    ("openai-community/gpt2", 1024, "pos_key"),
])
def test_max_context_read(repo, value, source):
    cfg = _snap()[repo]
    d = _lm(cfg)
    assert mg._max_context(d, cfg) == (value, source)
    assert _dims(cfg)["max_context"] == value


def test_max_context_rope_extension():
    """No position key: original_max_position_embeddings x factor for the window-extending rope types, never llama3
    (its factor shapes the frequencies); a factor <= 1 extends nothing (arch_metrics' rule, the zoo side)."""
    rs = lambda t, f: {"model_type": "nosuchtype", "rope_scaling": {"rope_type": t, "factor": f,
                                                                    "original_max_position_embeddings": 32768}}
    assert mg._max_context(rs("yarn", 4)) == (131072, "rope_scaling")
    assert mg._max_context(rs("longrope", 2.0)) == (65536, "rope_scaling")
    assert mg._max_context(rs("llama3", 32)) == (None, None)
    assert mg._max_context(rs("yarn", 1)) == (None, None)
    assert mg._max_context({**rs("yarn", 4), "max_position_embeddings": 40960}) == (40960, "pos_key")


def test_max_context_unknown_and_unbounded():
    """No source: unknown (None), not 4096 -- BLOOM (ALiBi attention, no length key, no class default); an attention-free
    / window-local recurrent point with no limit in its code: CTX_UNBOUNDED (Mamba; RecurrentGemma, whose local layers
    keep their attention_window_size reach). The gating: with the spec off the old 4096 comes back."""
    S = _snap()
    bloom, mamba, rg = (_dims(S[r]) for r in ("bigscience/bloom-560m", "state-spaces/mamba-130m-hf",
                                              "google/recurrentgemma-2b"))
    assert bloom["max_context"] is None and bloom["eff_window"] is None
    assert mamba["max_context"] == mamba["eff_window"] == mg.CTX_UNBOUNDED
    assert rg["max_context"] == mg.CTX_UNBOUNDED and rg["eff_window"] == 2048
    with off("max_context_unknown"):
        for r in ("bigscience/bloom-560m", "state-spaces/mamba-130m-hf", "moonshotai/Kimi-Linear-48B-A3B-Instruct"):
            assert _dims(S[r])["max_context"] == 4096, r
        assert _dims(S["google/recurrentgemma-2b"])["eff_window"] == 4096
    # no snapshot point keeps a silent default: no source -> unknown or unbounded
    for n, c in S.items():
        d = _lm(c)
        if isinstance(d, dict) and mg._max_context(d, c)[0] is None and _dims(c):
            assert _dims(c)["max_context"] in (None, mg.CTX_UNBOUNDED), n


def test_context_axis_unknown_and_unbounded():
    """Under USE_CONTEXT: an unknown host supplies no read target, an unknown target needs an unknown or unbounded host,
    an unbounded host supplies anything; unknown compares as its own value (reflexive, transitive). With USE_CONTEXT off
    (the headline) the axis changes nothing; with the spec off the codes are not read (the old >=)."""
    import numpy as np
    from verify.z3_relation import _plain_point
    k8 = _plain_point(max_context=8192, eff_window=8192)
    k32 = _plain_point(max_context=32768, eff_window=32768)
    u1 = _plain_point(max_context=None, eff_window=None)
    u2 = _plain_point(max_context=None, eff_window=None, hidden=128)
    ub = _plain_point(max_context=mg.CTX_UNBOUNDED, eff_window=mg.CTX_UNBOUNDED, hidden=128)
    P = [k8, k32, u1, u2, ub]
    assert mg.emul_matrix(P, mode="payable")[:, [3, 4]].all()               # USE_CONTEXT off: the wide hosts host all
    saved = mg.USE_CONTEXT
    try:
        mg.USE_CONTEXT = True
        ok = mg.emul_matrix(P, mode="payable")
        assert np.diag(ok).all()
        assert ok[0, 1] and not ok[1, 0]                    # read values compare
        assert not ok[0, 2] and not ok[0, 3]                # an unknown host supplies no read target
        assert not ok[2, 1] and not ok[3, 1]                # an unknown target is not supplied by a read host
        assert ok[2, 3]                                     # ... and is by an unknown one (its own value)
        assert ok[:, 4].all() and not ok[4, :4].any()       # unbounded supplies anything, needs an unbounded host
        S = ok.astype(np.int32)
        assert not (((S @ S) > 0) & ~ok).any()
        with off("max_context_unknown"):
            P0 = [k8, k32, _plain_point(max_context=4096, eff_window=4096)]
            ok0 = mg.emul_matrix(P0, mode="payable")
        assert ok0[2, 0] and ok0[0, 1] and not ok0[1, 0]
    finally:
        mg.USE_CONTEXT = saved


def test_start_token_rule_needs_a_table_length():
    """The learned-abs start-token rule charges T + 1 = the table's length: an unknown one is not charged, not credited
    (bos_rules; emul_matrix: pos_table 0)."""
    from graph.bos_rules import bos_learned_abs_charges, bos_learned_abs_ok
    gpt2 = dict(hidden=768, eff_layers=12, param_layers=12, layers=12, ffn_active=3072, ffn_total=3072,
                max_context=None, head_dim=64, q_cap=768, kv_cap=768)
    llama = dict(norm="RMSNorm", pos="rope", act="SiLU", ffn="glu", mixer="softmax", window="full", sink="no")
    host = dict(hidden=1 << 14, eff_layers=64, param_layers=64, ffn_active=1 << 24, full_eff=64, head_dim=64)
    assert not bos_learned_abs_ok(gpt2, host, llama)
    assert bos_learned_abs_ok({**gpt2, "max_context": 1024}, host, llama)
    with pytest.raises(ValueError):
        bos_learned_abs_charges(gpt2, llama)


# --- round 3: the classifier gaps the zoo's params triage and bespoke builders found (dist/r3_from_zoo.md, branch
# zoo-errors, sections 2-3; the rest of its items were already fixed by the batches' specs) ------------------------

def _zoo_v(repo, f):
    cfg = _snap()[repo]
    c = _cls(cfg)
    return f(c, _dims(cfg) if c else None)


ZOO_FIXES = [
    # (key, repo, value function (classify, dims) -> value, value with the key on, value with it off)
    ("zoo_gemma3moe_post_norm", "tardellirs/gemma3moe-16x4-270m-starter",
     lambda c, d: "post_norm" in c["feats"].split("+"), True, False),
    ("zoo_kanana2_qknorm", "kakaocorp/kanana-2-1.3b-instruct", lambda c, d: c["qknorm"], "head", "no"),
    ("zoo_apertus1p5_ungated", "swiss-ai/Apertus-v1.5-8B-base", lambda c, d: (c["ffn"], c["act"], d["inter"]),
     ("vanilla", "xielu", 21504), ("glu", "xielu", 21504)),
    ("zoo_phi3small_rope", "microsoft/Phi-3-small-8k-instruct",
     lambda c, d: c and (c["pos"], c["norm"], c["feats"]), ("rope", "LayerNorm", "bias_out+bias_qkv+tied_emb"), None),
    ("zoo_phi3small_biases", "microsoft/Phi-3-small-8k-instruct",
     lambda c, d: (d["q_bias"], d["ffn_bias"], d["bias_up"]), (1, 1, 1), (0, 0, 0)),
    ("zoo_phi3small_biases", "microsoft/Phi-3-small-128k-instruct",
     lambda c, d: (d["q_bias"], d["ffn_bias"], d["bias_up"]), (1, 1, 1), (0, 0, 0)),
    ("zoo_deep_topk_legacy", "Pacific-i64/TR-MOE-306",
     lambda c, d: (d["moe_sig"][:4], d["ffn_active"]), ((4, 2, 64, 3840), 2 * 64 + 3840), ((4, 1, 64, 3840), 64 + 3840)),
]
# the same value functions on repos the keys must not touch (the base families and a top_k-carrying non-deep config)
ZOO_NOT_FLAG = [
    ("zoo_gemma3moe_post_norm", "google/gemma-3-270m", lambda c, d: "post_norm" in c["feats"].split("+")),
    ("zoo_kanana2_qknorm", "Qwen/Qwen3-0.6B", lambda c, d: c["qknorm"]),
    ("zoo_apertus1p5_ungated", "swiss-ai/Apertus-8B-Instruct-2509", lambda c, d: c["ffn"]),
    ("zoo_phi3small_rope", "microsoft/Phi-3.5-mini-instruct", lambda c, d: c["pos"]),
    ("zoo_phi3small_biases", "microsoft/Phi-3.5-mini-instruct", lambda c, d: (d["q_bias"], d["ffn_bias"])),
    ("zoo_deep_topk_legacy", "AETHORIA-AI/TR-HASH-MOE-500M-HF", lambda c, d: d["moe_sig"][:4]),
    ("zoo_deep_topk_legacy", "moonshotai/Kimi-K3.1-Pro", lambda c, d: d["moe_sig"][:4]),
]


@pytest.mark.parametrize("key,repo,f,on,was", ZOO_FIXES, ids=[f"{k}:{r}" for k, r, *_ in ZOO_FIXES])
def test_zoo_classifier_fix(key, repo, f, on, was):
    """The value on the named repo (the zoo's code reading), and the old value with the key in R3_OFF (the gating)."""
    assert _zoo_v(repo, f) == on
    with off(key):
        assert _zoo_v(repo, f) == was


@pytest.mark.parametrize("key,repo,f", ZOO_NOT_FLAG, ids=[f"{k}:{r}" for k, r, _ in ZOO_NOT_FLAG])
def test_zoo_classifier_fix_not_flag(key, repo, f):
    v = _zoo_v(repo, f)
    with off(key):
        assert _zoo_v(repo, f) == v


# --- round 3, nonlm_kept (a design decision on not_a_causal_lm) --------------------------------------------------
NONLM = {"BAAI/bge-reranker-v2-minicpm-layerwise": "score_head",
         "BAAI/bge-reranker-v2.5-gemma2-lightweight": "score_head",
         "thuml/sundial-base-128m": "ts_patch"}


@pytest.mark.parametrize("repo", sorted(NONLM))
def test_nonlm_kept_places_them(repo):
    """The two score-head rerankers and sundial stay placed (nonlm_kept): with the head / input slots (nonlm_heads) as
    head score / flow + input patches, and with nonlm_heads off as the WIP's score_head / ts_patch features (walls both
    ways against plain LMs, carried by HOST_FEATS); the universal host carries them (HEAD_STATUS / INPUT_STATUS,
    tests/test_nonlm_host.py). With nonlm_kept
    in R3_OFF batch 1's not_lm skip comes back."""
    from graph.minimal_emulator import HOST_FEATS, is_carried, readout_kind
    cfg, feat = _snap()[repo], NONLM[repo]
    c = mg.classify(cfg)
    assert mg.skipped(cfg) is None and c and feat not in c["feats"].split("+")
    assert (c["head"], c["input"]) == (("score", "tokens") if feat == "score_head" else ("flow", "patches"))
    m = {"slots": c, "cfg": cfg, "name": repo}
    assert readout_kind(m)[::2] == (c["head"], c["input"]) and is_carried(m)
    with off("nonlm_heads"):
        c = mg.classify(cfg)
        assert (c["head"], c["input"]) == ("lm", "tokens") and feat in c["feats"].split("+")
        rest = "+".join(x for x in c["feats"].split("+") if x != feat)
        assert not mg.feats_ok(c["feats"], rest) and not mg.feats_ok(rest, c["feats"])     # a wall both ways
        assert feat in HOST_FEATS and mg.feats_ok(c["feats"], "*" + "+".join(sorted(HOST_FEATS)))
    with off("nonlm_kept"):
        assert mg.skipped(cfg) == "not_lm" and mg.classify(cfg) is None
        assert feat not in HOST_FEATS


def test_nonlm_kept_sundial_norm_and_biases():
    """Sundial, placed: its affine nn.LayerNorm (norm_fallback_rmsnorm_wrong: LayerNorm + the beta, bias_out) and the
    q / k / v biases of its own attention (bias_qkv_by_code_unlisted), batch 1's facts kept for the record, now live."""
    c = mg.classify(_snap()["thuml/sundial-base-128m"])
    assert c["norm"] == "LayerNorm" and {"bias_out", "bias_qkv"} <= set(c["feats"].split("+"))


def test_nonlm_kept_cost_wise_gemma_window():
    """The cost-wise reranker's code uses the window (the imported Gemma2DecoderLayer bands the even layers at
    sliding_window 4096): local window slot, and not window-free (a remote-code lever)."""
    c = mg.classify(_snap()["BAAI/bge-reranker-v2.5-gemma2-lightweight"])
    assert c["window"] == "local" and "cost_wise_gemma" not in mg.WINDOW_FREE_TYPES


# --- round 3, nonlm_heads (dist/r3_nonlm.md; a design decision on readouts) ------------------------------------------
# the head / input slots and the readout signature of every non-next-token model of the snapshot
HEADS = {**{r: ("embedding", "tokens", "last+l2") for r in (
    "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B",
    "boboliu/Qwen3-Embedding-4B-W4A16-G128", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "mlx-community/Qwen3-Embedding-4B-4bit-DWQ", "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
    "mlx-community/Qwen3-Embedding-8B-mxfp8", "jinaai/jina-code-embeddings-0.5b", "jinaai/jina-code-embeddings-1.5b",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "Alibaba-NLP/gte-Qwen2-7B-instruct", "TIGER-Lab/VLM2Vec-Full")},
    "NovaSearch/stella_en_1.5B_v5": ("embedding", "tokens", "mean"),
    "sdadas/stella-pl-retrieval-8k": ("embedding", "tokens", "mean"),
    "MongoDB/voyage-4-nano": ("embedding", "tokens", "mean+l2"),
    "BAAI/bge-reranker-v2-minicpm-layerwise": ("score", "tokens", ""),
    "BAAI/bge-reranker-v2.5-gemma2-lightweight": ("score", "tokens", ""),
    "Ray2333/GRM-Gemma-2B-sftreg": ("value", "tokens", ""),
    "thuml/sundial-base-128m": ("flow", "patches", "")}


def _pt(repo, **slots):
    cfg = _snap()[repo]
    return {"slots": {**mg.classify(cfg), **slots}, "dims": mg.dims_of(cfg), "members": [repo]}


@pytest.mark.parametrize("repo", sorted(HEADS))
def test_nonlm_heads_slots(repo):
    """head / input / readout_sig per model (the readout table of elephantine.loader_coverage READOUTS / readout_of); with
    nonlm_heads off every model is lm / tokens again."""
    cfg = _snap()[repo]
    c, dm = mg.classify(cfg), mg.dims_of(cfg)
    assert (c["head"], c["input"], dm["readout_sig"]) == HEADS[repo]
    assert "tied_emb" not in c["feats"].split("+")          # a non-LM head has no output table to tie
    with off("nonlm_heads"):
        c = mg.classify(cfg)
        assert (c["head"], c["input"], mg.dims_of(cfg)["readout_sig"]) == ("lm", "tokens", "")


def test_nonlm_heads_no_other_model_changes():
    """Only the HEADS models get a non-LM head; dims of the rerankers / sundial / GRM read off the code."""
    odd = [r for r, cfg in _snap().items() if isinstance(cfg, dict) and r not in HEADS and mg.classify(cfg)
           and (mg.classify(cfg)["head"], mg.classify(cfg)["input"]) != ("lm", "tokens")]
    assert not odd, odd[:5]
    d = lambda r: mg.dims_of(_snap()[r])
    assert (d("BAAI/bge-reranker-v2-minicpm-layerwise")["exits"],
            d("BAAI/bge-reranker-v2.5-gemma2-lightweight")["exits"]) == (33, 35)
    assert (d("Ray2333/GRM-Gemma-2B-sftreg")["readout_ff"], d("Ray2333/GRM-Gemma-2B-sftreg")["readout_bias"]) == (1024, 1)
    s = d("thuml/sundial-base-128m")
    assert (s["flow_sig"], s["patch_sig"], s["readout_w"]) == ("720/768/3", "16/3072", 720)
    assert (d("MongoDB/voyage-4-nano")["readout_w"], d("NovaSearch/stella_en_1.5B_v5")["readout_bias"]) == (2048, 1)


def _without(p, feat):
    return dict(p, slots={**p["slots"], "feats": "+".join(x for x in p["slots"]["feats"].split("+") if x and x != feat)})


def _with(p, feat):
    return dict(p, slots={**p["slots"], "feats": "+".join(sorted(set(filter(None, p["slots"]["feats"].split("+")))
                                                                 | {feat}))})


def test_nonlm_heads_readout_rule():
    """The readout rule (every mode but capacity): a score head is EXACT on an untied LM host (an lm_head row), not on a
    tied one; an embedding that pools by the mean or L2-normalises is a wall against LMs both ways, a last-token
    embedding without L2 is a fixed read (exact; none in the snapshot: synthetic); value / flow heads and the patch
    input are walls both ways; a head bias needs the host's bias_out."""
    import numpy as np
    sc, emb, val, flow = (_pt(r) for r in ("BAAI/bge-reranker-v2-minicpm-layerwise", "Qwen/Qwen3-Embedding-0.6B",
                                            "Ray2333/GRM-Gemma-2B-sftreg", "thuml/sundial-base-128m"))
    host = _pt("Qwen/Qwen-14B")                          # untied: plays the reranker (the leave-one-out's edge)
    assert "tied_emb" not in host["slots"]["feats"].split("+")
    qw = _pt("Qwen/Qwen3-0.6B")                          # the embedding's own body, as an LM (tied)
    assert "tied_emb" in qw["slots"]["feats"].split("+")
    fixed = dict(emb, dims={**emb["dims"], "readout_sig": "last"})               # a last-token embedding without L2
    fixed_b = dict(fixed, dims={**fixed["dims"], "readout_bias": 1})
    qw_u = _without(qw, "tied_emb")
    P = [sc, emb, val, flow, host, _with(host, "tied_emb"), qw, qw_u, fixed, fixed_b, _with(qw_u, "bias_out")]
    for mode in ("exact", "payable", "mixgap"):
        ok = mg.emul_matrix(P, mode=mode)
        assert ok[0, 4] and not ok[0, 5], mode           # score <- untied LM, not <- tied
        assert not ok[4, 0] and not ok[6, 0]             # a reranker plays no LM
        assert not ok[1, 6] and not ok[1, 7] and not ok[6, 1] and not ok[7, 1], mode    # last+l2: walls both ways
        assert ok[8, 7] and not ok[8, 6] and not ok[7, 8], mode      # a fixed read on the untied body; not tied
        assert not ok[9, 7] and ok[9, 10], mode          # a biased head needs the host's bias_out
        for k in (2, 3):                                 # value / flow: walls both ways against every LM
            assert not ok[k, [4, 5, 6, 7]].any() and not ok[[4, 5, 6, 7], k].any(), (mode, k)
        assert np.diag(ok).all()
    with off("nonlm_heads"):
        P2 = [_pt("BAAI/bge-reranker-v2-minicpm-layerwise"), _pt("Qwen/Qwen-14B")]
        assert not mg.emul_matrix(P2, mode="payable")[0, 1]          # (the WIP feature: a wall both ways)


def test_nonlm_heads_join_carries_them():
    """The universal host's head / input branch sets (dist/r3_nonlm.md section 5) play every non-LM point."""
    import numpy as np
    from graph.minimal_emulator import HOST_HEAD, HOST_INPUT, host_point, join, load
    allm = load()
    models = [m for m in allm if m["name"] in HEADS]
    assert {m["name"] for m in models} == set(HEADS)
    j = join(allm)
    P = [{"slots": m["slots"], "dims": m["dims"], "members": [m["name"]]} for m in models]
    ok = mg.emul_matrix(P + [host_point(j, "ar"), host_point(j, "diffusion")], mode="payable")
    n = len(P)
    col = np.array([n + (p["slots"]["gen"] != "ar") for p in P])
    assert ok[np.arange(n), col].all(), [P[i]["members"][0] for i in np.where(~ok[np.arange(n), col])[0]]
    assert set(mg.branches(HOST_HEAD)) == {"lm", "score", "embedding", "value", "flow"}
    assert set(mg.branches(HOST_INPUT)) == {"tokens", "patches"}


def test_nonlm_heads_exit_heads_bind_nobody():
    """exit_heads (target side a wall, host side free): a saved non-default cutoff_layers; no snapshot config has one
    (the released default call reads one exit, at L)."""
    cfg = dict(_snap()["BAAI/bge-reranker-v2-minicpm-layerwise"])
    assert "exit_heads" not in mg.classify(cfg)["feats"].split("+")
    assert mg.FEATURES["exit_heads"][:2] == ("wall", "free")
    assert "exit_heads" in mg.classify({**cfg, "cutoff_layers": [8, 40]})["feats"].split("+")
    assert "exit_heads" not in mg.classify({**cfg, "cutoff_layers": [40]})["feats"].split("+")
    assert not [r for r, c in _snap().items() if isinstance(c, dict) and mg.classify(c)
                and "exit_heads" in mg.classify(c)["feats"].split("+")]


def test_bidir_stella_voyage_already_bidirectional():
    """bidir_stella_voyage (dist/r3_nonlm.md section 3: 'classified gen = ar'): a NON-ISSUE on round 3 -- batch 1's
    bidirectional_flag_true already reads voyage's use_bidirectional_attention true and the stella modeling_qwen.py
    Qwen2Model (forward is_causal=False by default) as gen diffusion; with that spec off they are ar again."""
    for r in ("NovaSearch/stella_en_1.5B_v5", "sdadas/stella-pl-retrieval-8k", "MongoDB/voyage-4-nano"):
        assert mg.classify(_snap()[r])["gen"] == "diffusion", r
        with off("bidirectional_flag_true"):
            assert mg.classify(_snap()[r])["gen"] == "ar", r


# ---------------------------------------------------------------- round 3 (nope_rope_limit): the linear-scaling lever

def _rope_pt(pos, sched, cap, hd=4, **dims):
    from verify.z3_relation import _plain_point
    return _plain_point(pos=pos, rope_sched=sched, rope_cap=cap, head_dim=hd, kv_cap=hd, q_cap=hd, **dims)


def test_nope_rope_limit_nope_target():
    """A NoPE target on a fixed full-RoPE host (no zero-frequency room: rope_theta cannot reach inv_freq = 0) is a
    LIMIT edge where the host's rotary reads linear scaling (rope_lin): payable and mixgap, never exact; without
    rope_lin, or with the spec off, a wall."""
    T = _rope_pt("nope", (), ())
    H = _rope_pt("rope", (4,), (4,), rope_lin=1)
    H0 = _rope_pt("rope", (4,), (4,), rope_lin=0)
    for mode, want in (("exact", False), ("payable", True), ("mixgap", True)):
        ok = mg.emul_matrix([T, H, H0], mode=mode)
        assert ok[0, 1] == want and not ok[0, 2], mode
        assert ok.diagonal().all()
    with off("nope_rope_limit"):
        assert not mg.emul_matrix([T, H, H0], mode="payable")[0, 1]
    # a partial-RoPE host with pass dims holds a narrow NoPE target exactly already (zero_freq_room): no lever needed
    P = _rope_pt("partial_rope", (2,), (2,), hd=4)
    assert mg.emul_matrix([_rope_pt("nope", (), (), hd=2), P], mode="exact")[0, 1]
    # ALiBi still needs its exact zero dims and small-angle pair: the lever is for NoPE only
    A = _rope_pt("alibi", (), (), alibi_heads=1, alibi_max=8)
    assert not mg.emul_matrix([A, H], mode="payable")[0, 1]


def test_nope_rope_limit_mixed_target():
    """A mixed target (one rotating schedule + NoPE layers: rope_sched (2, -4)) needs a host table of its own for the
    NoPE layers: a per-table linear host (rope_lin 2) with two tables plays it (one table to infinity, the rotating
    schedule on the other by fixed_rot_fits), payable only; a one-table linear host cannot (scaling its table also
    stops the rotating layers), nor a two-table host whose one factor reaches every table (rope_lin 1)."""
    M = _rope_pt("rope", (2, -4), (2, -4))
    H2 = _rope_pt("rope", (2, 2), (2, 2), rope_lin=2)
    H2one = _rope_pt("rope", (2, 2), (2, 2), rope_lin=1)
    H1 = _rope_pt("rope", (4,), (4,), rope_lin=2)
    for mode, want in (("exact", False), ("payable", True), ("mixgap", True)):
        ok = mg.emul_matrix([M, H2, H2one, H1], mode=mode)
        assert ok[0, 1] == want and not ok[0, 2] and not ok[0, 3], mode
    with off("nope_rope_limit"):
        assert not mg.emul_matrix([M, H2], mode="payable")[0, 1]
    # the predicate itself: the class conditions of the rot gate (a fixed host plays no pairs-free target)
    assert mg.nope_rope_limit_rot((2, -4), 4, "", False, False, (2, 2), 4, False, False, "", 2)
    assert not mg.nope_rope_limit_rot((2, -4), 4, "", True, False, (2, 2), 4, False, False, "", 2)
    assert not mg.nope_rope_limit_rot((2, -4), 4, "", False, False, (2,), 4, False, False, "", 2)
    assert not mg.nope_rope_limit_rot((2,), 4, "", False, False, (2, 2), 4, False, False, "", 2)
    assert mg.nope_rope_limit_slot("nope", 1, True) and not mg.nope_rope_limit_slot("nope", 0, True)
    assert not mg.nope_rope_limit_slot("alibi", 1, True) and not mg.nope_rope_limit_slot("nope", 1, False)


@pytest.mark.parametrize("repo,want", [
    ("OrionStarAI/Orion-14B-Chat", 1),          # remote modeling_orion.py: rope_scaling "linear" (one table)
    ("openbmb/MiniCPM4-8B", 1),                 # remote modeling_minicpm.py: linear / dynamic / longrope
    ("dots-studio/dots3-note-prev", 2),         # tf-main: one table per layer type, ROPE_INIT_FUNCTIONS each
    ("meta-llama/Llama-3.2-1B", 1),             # ROPE_INIT_FUNCTIONS (pairs-free: linear too)
    ("google/recurrentgemma-2b", 0),            # raises on any rope type but "default"
    ("EleutherAI/gpt-j-6b", 0),                 # a fixed sin / cos table, no scaling
])
def test_nope_rope_limit_rope_lin(repo, want):
    """rope_lin (dims_of) per family; 0 everywhere with the spec off."""
    cfg = _snap().get(repo)
    if cfg is None:
        pytest.skip(f"{repo} not in the snapshot")
    assert mg.dims_of(cfg)["rope_lin"] == want
    with off("nope_rope_limit"):
        assert mg.dims_of(cfg)["rope_lin"] == 0


# --- round 3 (kind_order): layer kinds matched IN ORDER (graph/model_graph.py kind_match / _order_mask) -------------
def _kpt(seq, **counts):
    """A synthetic point for kind_match: per-kind eff / param counts and the dims kseq (runs "FU*1,R*1")."""
    d = {f"k{k}_{w}": 0 for k in mg.LAYER_KINDS for w in ("eff", "param")}
    for k, c in counts.items():
        d[f"k{k}_eff"] = d[f"k{k}_param"] = c
    d["kseq"] = seq
    return {"slots": {}, "dims": d}


def test_kind_order_refuses_a_reordered_host():
    """Negative control: target [attention, recurrent] on host [recurrent, attention] passes the count (Hall) and is
    refused in order; a same-order host, and a host with the target's order inside a longer stack, are accepted.
    With the spec off the count grants the reordered pair (the old relation)."""
    AR, RA = _kpt("FU*1,R*1", FU=1, R=1), _kpt("R*1,FU*1", FU=1, R=1)
    ARAR, RRA = _kpt("FU*1,R*1,FU*1,R*1", FU=2, R=2), _kpt("R*2,FU*1", FU=1, R=2)
    P = [AR, RA, ARAR, RRA]
    ok = mg.kind_match(P)
    assert not ok[0, 1] and not ok[1, 0]                 # the reorder: refused both ways
    assert ok[0, 2] and ok[1, 2]                         # [A, R] and [R, A] both embed in [A, R, A, R]
    assert ok[1, 3] and not ok[0, 3]                     # [R, A] in [R, R, A]; [A, R] not
    assert ok.diagonal().all()
    assert mg.kind_order_ok((("FU", 1), ("R", 1)), (("FU", 1), ("R", 1)))
    assert not mg.kind_order_ok((("FU", 1), ("R", 1)), (("R", 1), ("FU", 1)))
    with off("kind_order"):
        ok0 = mg.kind_match(P)
    assert ok0[0, 1] and ok0[1, 0] and ok0[0, 3]


def test_kind_order_variants():
    """The 2x depth takes each target layer twice in order; the start-token marker layers go before the first target
    layer; a full layer plays a sliding one (KIND_REACH); a parallel attention + recurrent layer (X|R) plays either
    kind or another parallel layer, never two sequential layers; greedy earliest fit is the order embedding."""
    t, h = (("FU", 1), ("R", 1)), (("FU", 1), ("R", 1), ("FU", 1), ("R", 1))
    assert mg.kind_order_ok(t, h) and not mg.kind_order_ok(t, h, factor=2)
    assert mg.kind_order_ok(t, (("FU", 2), ("R", 2)), factor=2)
    assert not mg.kind_order_ok((("LU", 2),), (("LU", 2), ("FU", 2)), add=2)    # the markers need full layers first
    assert mg.kind_order_ok((("LU", 2),), (("FU", 2), ("LU", 2)), add=2)
    assert mg.kind_order_ok((("LU", 1), ("FU", 1)), (("FU", 2),))
    assert not mg.kind_order_ok((("FU", 1), ("LU", 1)), (("LU", 1), ("FU", 1)))
    assert mg.kind_order_ok((("FU", 1), ("LU", 1)), (("LU", 1), ("FU", 1)), hfull=True)   # the window lever
    par = (("FU|R", 2),)
    assert mg.kind_order_ok((("FU", 1), ("R", 1)), par) and mg.kind_order_ok(par, par)
    assert not mg.kind_order_ok(par, (("FU", 2), ("R", 2)))     # a sequential pair is not a parallel layer
    assert not mg.kind_order_ok((("FU", 1), ("R", 1)), (("FU|R", 1),))
    # greedy = an exhaustive order-embedding search on small random sequences
    import itertools
    import random
    R = mg._TOK_REACH_BASE
    rng = random.Random(0)
    toks = ("FU", "LU", "R", "FU|R")

    def brute(tl, hl):
        for pos in itertools.combinations(range(len(hl)), len(tl)):
            if all(a in R[hl[p]] for a, p in zip(tl, pos)):
                return True
        return False
    for _ in range(400):
        tl = [rng.choice(toks) for _ in range(rng.randint(1, 4))]
        hl = [rng.choice(toks) for _ in range(rng.randint(1, 6))]
        runs = lambda xs: tuple((x, 1) for x in xs)
        assert mg.kind_order_ok(runs(tl), runs(hl)) == brute(tl, hl), (tl, hl)


@pytest.mark.parametrize("repo,want", [
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "R*3,FU*1," * 6 + "R*2,FU*1"),   # 1-based full_attn_layers
    ("LiquidAI/LFM2-350M", "R*2,FU*1,R*2,FU*1,R*2,FU*1,R*1,FU*1,R*1,FU*1,R*1,FU*1,R*1"),
    ("Qwen/Qwen3-Coder-Next", ",".join(["R*3,FU*1"] * 12)),                       # (i + 1) % interval == 0
    ("AxiomicLabs/GPT-S3-3M", "R*3,FU*1,R*3,FU*1"),
    ("ibm-ai-platform/Bamba-9B-v1", "R*9,FU*1,R*8,FU*1,R*8,FU*1,R*4"),
    ("ai21labs/AI21-Jamba-Reasoning-3B", "R*7,FU*1,R*13,FU*1,R*6"),               # i % period == offset
    ("tiiuae/Falcon-H1-0.5B-Base", "FU|R*36"),                                    # parallel attention + SSM
    ("defnoteatin/PseudoBrain-Qwen3.5-4B-BF16", ",".join(["R*3,FU|R*1"] * 8)),
    ("google/gemma-3-1b-it", ",".join(["LU*5,FU*1"] * 4) + ",LU*2"),
    ("meta-llama/Llama-3.2-1B", ""),                                              # one plain kind: the counts
])
def test_kind_order_kseq(repo, want):
    """dims kseq, read in each family's code order; the index-list layouts (Kimi / LFM2 / Bamba / Jamba / Qwen3-Next /
    GPT-S3) were laid out attention-first before (the counts unchanged), and kseq is "" with the spec off."""
    cfg = _snap().get(repo)
    if cfg is None:
        pytest.skip(f"{repo} not in the snapshot")
    d = mg.dims_of(cfg)
    assert d["kseq"] == want
    with off("kind_order"):
        d0 = mg.dims_of(cfg)
    assert d0["kseq"] == ""
    assert all(d[f"k{k}_{w}"] == d0[f"k{k}_{w}"] for k in mg.LAYER_KINDS for w in ("eff", "param"))


def test_kind_order_real_pairs():
    """On the snapshot: granite-4.0-h-350m (R10 A R2 A R3 A R9 A R4) is refused by granite-4.0-h-tiny (R5 A R9 A R9 A
    R9 A R4: no order fits, though the counts do), Jamba2-3B by Jamba-v0.1; both granted with the spec off; every
    point hosts itself."""
    from graph.currency_curve import points
    pairs = [("ibm-granite/granite-4.0-h-350m", "ibm-granite/granite-4.0-h-tiny"),
             ("ai21labs/AI21-Jamba2-3B", "ai21labs/Jamba-v0.1")]
    for spec_off in (False, True):
        with (off("kind_order") if spec_off else contextlib.nullcontext()):
            P = points()
            ok = mg.emul_matrix(P, mode="payable")
        idx = {m: i for i, p in enumerate(P) for m in p["members"]}
        for t, h in pairs:
            if t not in idx or h not in idx:
                pytest.skip(f"{t} / {h} not in the snapshot")
            assert ok[idx[t], idx[h]] == spec_off, (t, h, spec_off)
        assert ok.diagonal().all()


# --- round 3 (mamba_ffn_none, mamba1_lever): FFN-less Mamba stacks, the Mamba-1 dt / B / C norm lever ---------------
@pytest.mark.parametrize("repo,want,was", [
    ("state-spaces/mamba-130m-hf", "none", "vanilla"),      # MambaBlock: norm + mixer, no MLP
    ("tiiuae/falcon-mamba-7b", "none", "glu"),               # FalconMambaBlock: likewise
    ("RWKV/rwkv-4-169m-pile", None, None),                   # RwkvBlock's feed_forward (channel mix): keeps its FFN
    ("fla-hub/gla-1.3B-100B", None, None),                   # fla GLABlock: an MLP after the mixer
])
def test_mamba_ffn_none_classify(repo, want, was):
    cfg = _snap().get(repo)
    if cfg is None:
        pytest.skip(f"{repo} not in the snapshot")
    ffn, d = _slot(cfg, "ffn"), _dims(cfg)
    with off("mamba_ffn_none"):
        ffn0, d0 = _slot(cfg, "ffn"), _dims(cfg)
    if want is None:
        assert ffn != "none" and ffn == ffn0 and d["ffn_total"] == d0["ffn_total"] > 0
        return
    assert ffn == want and d["ffn_total"] == d["ffn_active"] == 0
    assert ffn0 == was and d0["ffn_total"] > 0           # the old slot / width (the mixer's d_inner)


def test_mamba_ffn_none_order():
    """FFN none is hosted by every FFN kind (zero the MLP's down_proj, exact) and hosts only itself."""
    for h in ("none", "vanilla", "glu", "moe", "moe_shared"):
        assert mg.slot_reaches("ffn", "none", h, strict=True), h
    for t in ("vanilla", "glu", "moe"):
        assert not mg.slot_reaches("ffn", t, "none"), t
    assert mg.FFN_RANK["none"] < mg.FFN_RANK["vanilla"]


def test_mamba1_lever_pairs():
    """mamba-130m on falcon-mamba-7b: a payable LIMIT edge (refused exactly); refused with either key off (the FFN-kind
    wall without mamba_ffn_none, the family wall without mamba1_lever). The measurement meets the standard."""
    from graph.currency_curve import points
    t, h = "state-spaces/mamba-130m-hf", "tiiuae/falcon-mamba-7b"
    got = {}
    for keys in ((), ("mamba_ffn_none",), ("mamba1_lever",)):
        with off(*keys):
            P = points()
            idx = {m: i for i, p in enumerate(P) for m in p["members"]}
            if t not in idx or h not in idx:
                pytest.skip("not in the snapshot")
            got[keys] = (mg.emul_matrix(P, mode="payable")[idx[t], idx[h]],
                         mg.emul_matrix(P, mode="exact")[idx[t], idx[h]])
    assert got[()] == (True, False)
    assert not got[("mamba_ffn_none",)][0] and not got[("mamba1_lever",)][0]
    rows = [json.loads(x) for x in open(os.path.join(ROOT, "hf", "data", "mixer_rules_agreement.jsonl"))]
    m1 = [r for r in rows if r["rule"].startswith("mamba1_lever")]
    assert {r["target"] for r in m1} >= {"state-spaces/mamba-130m-hf", "state-spaces/mamba-370m-hf"}
    assert all(r["dtype"] == "float32" and r["top1"] >= 0.99 and r["ex32"] >= 0.9 and r["payable"] for r in m1)


# --- round 3, the closing step: the router score is a feature switch; the grouped o-map is reflexive -----------------

def test_score_not_free_real_shapes_transitive():
    """The 9 real triples (small softmax MoE targets <= K2-Horizon-375B with scoring_func set to softmax <= Ling-1T,
    sigmoid only) are gone with score_not_free: 0 violations in payable / mixgap, and they come back with it off."""
    import numpy as np
    from graph.currency_curve import points
    for mode in ("payable", "mixgap"):
        ok = mg.emul_matrix(points(), mode=mode)
        S = ok.astype(np.int32)
        assert not (((S @ S) > 0) & ~ok).any(), mode
    with off("score_not_free"):
        ok = mg.emul_matrix(points(), mode="payable")
        S = ok.astype(np.int32)
        assert (((S @ S) > 0) & ~ok).sum() > 0


def test_score_not_free_caps_are_the_own_score():
    """With score_not_free every MoE point's score_cap is its own score."""
    caps = []
    for repo, cfg in _snap().items():
        if isinstance(cfg, dict) and mg.classify(cfg):
            sig = mg.dims_of(cfg).get("moe_sig")
            if sig and mg._moe_fields(sig)["score"]:      # a routed MoE (a dense point's sig has no score)
                T = mg._moe_fields(sig)
                caps.append(T["score_cap"] == (T["score"],))
    assert caps and all(caps)


def test_o_groups_reflexive():
    """A grouped low-rank o-map with qheads not a multiple of o_groups hosts itself (the identical map); with the key
    off the old refusal comes back."""
    import numpy as np
    base = {"hidden": 64, "qheads": 3, "head_dim": 8}
    P = [{"slots": {}, "dims": {**base, "o_groups": 2, "o_rank": 1}},
         {"slots": {}, "dims": {**base, "o_groups": 2, "o_rank": 4}}]
    m = mg._r3_masks_2d(P, np.array([64, 64]))["o_proj groups"]
    assert m[0, 0] and m[1, 1]
    with off("o_groups_reflexive"):
        m = mg._r3_masks_2d(P, np.array([64, 64]))["o_proj groups"]
        assert not m[0, 0]
