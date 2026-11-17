"""Standalone config helpers for the round-3 detectors (verify/r3/detectors.py).

They copy the few graph/model_graph.py helpers the detectors need (_num, _lm_unwrap, _layer_count) so the detectors do
not import model_graph, which another pass is editing. Keep them in step with model_graph by hand: the copies follow
graph/model_graph.py at the round-2 pass-4 commit (3703d45).

Detector convention: detect_<finding>(cfg, model_type) -> value (a fixture marked repo_keyed passes cfg with
"_repo_id" set: for a code fact no config shows)
  cfg         the raw snapshot config dict (graph/hub_10k.json "config"; may be a multimodal wrapper)
  model_type  model_type_of(cfg): the lower-cased model_type of the unwrapped text LM (the one classify() keys on)
  value       a falsy value (None / False / 0 / "") when the finding does not apply, else the new slot value,
              FEATURES key, dims value, ... the spec names.
"""

LM_NEST_KEYS = ("text_config", "llm_config", "llm_cfg", "language_config", "thinker_config",
                "language_model", "text_model", "decoder")
LM_WRAP_KEYS = ("qwen_config", "backbone_configs")
LM_DIM_KEYS = ("hidden_size", "n_embd", "n_embed", "d_model", "hidden_dim", "model_dim", "dim", "embed_dim")
LM_LAYER_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "num_transformer_layers", "n_layers")
LM_LAYER_LIST_KEYS = ("layers_block_type", "attn_layers", "layer_types")
LOOP_COUNT_KEYS = ("loop_count", "num_loops", "loop_num", "loop_times", "n_recurrent_loops", "recurrent_loops",
                   "loops_per_group")


def num(v):
    """model_graph._num: a scalar int; a per-layer list gives its max; anything else 0."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, tuple)):
        nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return int(max(nums)) if nums else 0
    return int(v) if isinstance(v, (int, float)) else 0


def fnum(v, default=None):
    """A float value, or default."""
    if isinstance(v, bool):
        return default
    return float(v) if isinstance(v, (int, float)) else default


def layer_count(d):
    for k in LM_LAYER_KEYS:
        if d.get(k):
            return num(d[k])
    for k in LM_LAYER_LIST_KEYS:
        v = d.get(k)
        if isinstance(v, (list, tuple)) and v:
            return len(v)
    return 0


def has_lm_dims(d):
    return isinstance(d, dict) and any(d.get(k) for k in LM_DIM_KEYS) and layer_count(d) > 0


def lm_of(d):
    """model_graph._lm_unwrap: the sub-config holding the text LM (flat configs returned as-is)."""
    if isinstance(d, dict) and not (d.get("num_attention_heads") or d.get("n_head")):
        for k in LM_WRAP_KEYS:
            sub = d.get(k)
            cand = lm_of(sub) if isinstance(sub, dict) else None
            if cand is not None and has_lm_dims(cand):
                extra = {kk: d[kk] for kk in LOOP_COUNT_KEYS if kk in d}
                return {**cand, **extra, "model_type": d.get("model_type") or cand.get("model_type")}
    if not isinstance(d, dict) or has_lm_dims(d):
        return d if isinstance(d, dict) else {}
    for k in LM_NEST_KEYS:
        sub = d.get(k)
        if isinstance(sub, dict):
            cand = lm_of(sub)
            if has_lm_dims(cand):
                if not cand.get("model_type") and d.get("model_type"):
                    cand = {**cand, "model_type": d["model_type"]}
                return cand
    return d


def model_type_of(cfg):
    """The detectors' model_type argument: the unwrapped LM's model_type, lower-cased ("" if none)."""
    lm = lm_of(cfg or {})
    return str(lm.get("model_type") or "").lower()


def outer_type(cfg):
    return str((cfg or {}).get("model_type") or "").lower()


def archs(cfg):
    """The architectures list of the outer config and of the LM config (strings)."""
    out = []
    for d in (cfg or {}, lm_of(cfg or {})):
        a = d.get("architectures")
        if isinstance(a, list):
            out += [str(x) for x in a]
    return out


def auto_map(cfg):
    """The outer + LM auto_map values joined (remote-code module paths), '' if none."""
    parts = []
    for d in (cfg or {}, lm_of(cfg or {})):
        am = d.get("auto_map")
        if isinstance(am, dict):
            parts += [str(v) for v in am.values()]
    return " ".join(parts)


def rope_params(lm):
    """rope_parameters / rope_scaling as a dict ({} if absent or not a dict)."""
    r = lm.get("rope_parameters") or lm.get("rope_scaling") or {}
    return r if isinstance(r, dict) else {}


def get(cfg, key, default=None):
    """A key from the LM config, else from the outer config."""
    lm = lm_of(cfg or {})
    if key in lm:
        return lm[key]
    return (cfg or {}).get(key, default)


def truthy_flag(v):
    """A config switch the code tests with `if x` or string compare: True, 1, "true", "yes", a non-empty string
    other than "false" / "none" / "0"."""
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "none", "no", "0", "off")
    return bool(v)


def layer_types(lm):
    lt = lm.get("layer_types")
    return [str(x) for x in lt] if isinstance(lt, list) else []
