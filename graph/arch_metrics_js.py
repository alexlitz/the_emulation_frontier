"""The data tables of the in-browser port of graph/arch_metrics.py (graph/templates/arch_metrics.js).

The JS port computes a config's metrics with no Python and no transformers at runtime. Everything it would otherwise
read from Python is generated here, from code, never typed in:

  mg    graph/model_graph.py's per-type sets and tables (the classifier facts the formulas read)
  am    graph/arch_metrics.py's tables (HEAD_DIM_FROM_HIDDEN, LAYOUT, the contexts, dtypes and batch)
  tf    the installed transformers' config classes: per model type the config class; per class its __init__
        signature (parameter defaults), attribute_map, sub-configs, properties, class attributes and its __init__
        body as a statement program (_program: the JS replays it, interpreting every non-trivial statement from its
        AST, with the self._methods and properties it calls); the class-default values model_graph._class_default
        reads (getattr on a default instance);
        and the stock model classes (lower-cased name -> name, config class, module) that arch_metrics.stock_arch
        tests

Run:  python -m graph.arch_metrics_js          # writes graph/data/arch_metrics_js_tables.json
"""
from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import re
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "arch_metrics_js_tables.json")

# model_graph names the port reads (sets are written as sorted lists)
MG_NAMES = (
    "GATED", "VANILLA", "LN_FAMILIES", "LEARNED_ABS", "SINUSOIDAL_TYPES", "SSM_TYPES", "LINEAR_TYPES", "HYBRID_TYPES",
    "NEMOTRON_H_TYPES", "RMS_LNEPS_TYPES", "RMS_LNEPS_ARCHS", "NORM_BY_CODE", "RAW_ACT_BY_CODE", "UNGATED_BY_CODE",
    "PARALLEL_BY_CODE", "FUSED_GLU_HALF", "_CLASS_DEFAULT_FALLBACK", "ACT_KEY_BY_CODE", "GELU_FORM_BY_CODE",
    "_REC_LAYER_NAMES", "SKIP_TYPES", "DRAFTER_KEYS", "LOOPED_TYPES", "_LOOP_COUNT_KEYS", "_DENSE_ARCHS",
    "_TOPK_ALT_KEYS", "NOPE_BY_CODE", "QKNORM_ALWAYS", "QKNORM_TYPES", "QKNORM_KEY_NOT_SOFTMAX",
    "QKNORM_KEY_NOT_SOFTMAX_ARCHS", "QKNORM_BY_CODE", "QKNORM_VARIANT_BY_TYPE", "QKNORM_HEAD_CHECKED", "OUT_GATE_TYPES",
    "SHARED_GATE_TYPES", "BIAS_QKV_TYPES", "BIAS_OUT_TYPES", "O_BIAS_FROM_ATTN_BIAS", "POST_NORM_TYPES",
    "POST_ONLY_TYPES", "VNORM_TYPES", "OUT_GATE_BY_CODE", "_SABI_SHARED_MLA", "CODE_SLOT_OVERRIDES", "_KV_STD_KEYS",
    "QK_GAIN_KEYS", "K_EQ_V_BY_CODE", "LONGCAT_ARCHS", "TIED_BY_CODE", "FLA_ARCHS", "INCLUSIVE_WINDOW_TYPES",
    "HADAMARD_FFN_TYPES", "FFN_NONE_TYPES", "ATTN_ONLY_FEATS", "DSV4_SCHEMA_TYPES", "RECURRENT_FAMILY", "_LOCAL_MARKS", "_FFN_ENTRIES", "_LM_NEST_KEYS",
    "_LM_WRAP_KEYS", "_LM_DIM_KEYS", "_LM_LAYER_KEYS", "_LM_LAYER_LIST_KEYS", "LAYER_KINDS", "MOE_SIG", "USE_ACT_CODE",
    "ZOO_SNAPSHOT")
AM_NAMES = ("HEAD_DIM_FROM_HIDDEN", "LAYOUT", "NO_CACHE_TYPES", "DENSE_EXPERT_CODE", "LONGCAT_ARCHS", "NANOCHAT_TYPES",
            "DSV4_TYPES", "CONTEXTS", "DTYPE_BYTES", "BIAS_OUT_NORM_BETA",
            "BATCH_B", "SCHEMA_VERSION")
# the keys model_graph._class_default / _cfg_value reads on the paths the metrics reach
CDEF_KEYS = ("tie_word_embeddings", "mamba_rms_norm", "mamba_norm_before_gate", "parallel_attn",
             "use_parallel_residual", "multi_query", "block_auto_adjust_ff_dim", "block_ffn_dim_multiplier",
             "block_multiple_of", "no_rope_layer_interval", "hidden_act", "activation_function", "hidden_activation",
             "activation", "norm_topk_prob", "rotary_pct", "partial_rotary_factor", "attn_logit_softcapping",
             "adapter_rank")
NONJSON = "__nonjson__"
REQUIRED = "__required__"


def _plain(v):
    """A JSON value of a Python constant (sets / frozensets / tuples -> sorted or plain lists)."""
    if isinstance(v, (set, frozenset)):
        return sorted(_plain(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def _json_ok(v):
    try:
        json.dumps(v, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


_SKIP_FIELDS = {"ctx", "type_comment", "kind", "type_params"}
AST_TYPES = []                                          # node type names; a node is [type index, *fields]
_AST_FIELDS = {}


def _encode(node):
    """A compact JSON form of a Python AST for the JS interpreter: a node is [index into AST_TYPES, its fields in
    ast._fields order (minus _SKIP_FIELDS)]; a list stays a list; a constant that JSON cannot hold is
    {"__repr__": repr}. tf.ast_types / tf.ast_fields give the names back."""
    if isinstance(node, ast.AST):
        t = type(node).__name__
        if t not in _AST_FIELDS:
            _AST_FIELDS[t] = [f for f in node._fields if f not in _SKIP_FIELDS]
            AST_TYPES.append(t)
        out = [AST_TYPES.index(t)]
        for f in _AST_FIELDS[t]:
            v = getattr(node, f, None)
            if isinstance(node, ast.Constant) and f == "value":
                out.append(v if _json_ok(v) else {"__repr__": repr(v)})
            else:
                out.append(_encode(v))
        return out
    if isinstance(node, list):
        return [_encode(x) for x in node]
    return node if _json_ok(node) else {"__repr__": repr(node)}


def _function(fn_obj):
    """{"args": [...], "defaults": [...], "body": [...]} of a function's source, or None."""
    try:
        src = textwrap.dedent(inspect.getsource(fn_obj))
        fd = ast.parse(src).body[0]
    except (OSError, TypeError, SyntaxError, IndexError):
        return None
    if not isinstance(fd, ast.FunctionDef):
        return None
    a = fd.args
    return {"args": [x.arg for x in a.posonlyargs + a.args], "defaults": _encode(a.defaults),
            "vararg": a.vararg.arg if a.vararg else None, "kwarg": a.kwarg.arg if a.kwarg else None,
            "body": _encode(fd.body)}


def _base_attr(name):
    from transformers import PretrainedConfig
    f = getattr(PretrainedConfig, name, None)
    return getattr(f, "__func__", f)


def _methods(cls, prog_texts):
    """The methods the __init__ statements call on self (self._name(...)), recursively, and the property getters of
    the class (both interpreted by the JS)."""
    from transformers import PretrainedConfig
    names, out, todo = set(), {}, list(prog_texts)
    while todo:
        for n in re.findall(r"self\.(\w+)\(", todo.pop()):
            if n in names or n == "__init__":
                continue
            names.add(n)
            f = getattr(cls, n, None)
            f = getattr(f, "__func__", f)
            if f is None or f is _base_attr(n):
                continue
            enc = _function(f)
            if enc is not None:
                raw_attr = next((vars(k)[n] for k in cls.__mro__ if n in vars(k)), None)
                enc["static"] = isinstance(raw_attr, staticmethod)
                out[n] = enc
                try:
                    todo.append(textwrap.dedent(inspect.getsource(f)))
                except (OSError, TypeError):
                    pass
    props = {}
    for klass in cls.__mro__:
        if klass is PretrainedConfig or klass is object:
            break
        for k, v in vars(klass).items():
            if isinstance(v, property) and k not in props and v.fget is not None:
                enc = _function(v.fget)
                if enc is not None:
                    props[k] = enc
    return out, props


def _program(cls):
    """The top-level statements of cls.__init__, for the JS emulation: ["a", attr, local] for `self.attr = local`,
    ["S", [[keyword, value], ...]] for the super().__init__ call (keyword None = **kwargs; value ["n", local],
    ["c", constant] or ["e", AST]), ["x", source] for anything else (stored as its AST in tf.stmts_ast, which the
    JS interprets). None when the source cannot be read."""
    try:
        src = textwrap.dedent(inspect.getsource(cls.__init__))
        fn = ast.parse(src).body[0]
    except (OSError, TypeError, SyntaxError, IndexError):
        return None
    prog = []
    for st in fn.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            continue                                        # the docstring
        t = st.targets[0] if isinstance(st, ast.Assign) and len(st.targets) == 1 else None
        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
                and isinstance(st.value, ast.Name)):
            prog.append(["a", t.attr, st.value.id])
            continue
        c = st.value if isinstance(st, ast.Expr) else None
        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "__init__"
                and isinstance(c.func.value, ast.Call) and isinstance(c.func.value.func, ast.Name)
                and c.func.value.func.id == "super" and not c.args):
            kws = []
            for kw in c.keywords:
                v = kw.value
                if isinstance(v, ast.Name):
                    kws.append([kw.arg, ["n", v.id]])
                elif isinstance(v, ast.Constant) and _json_ok(v.value):
                    kws.append([kw.arg, ["c", v.value]])
                else:
                    kws.append([kw.arg, ["e", _encode(v)]])
            prog.append(["S", kws])
            continue
        prog.append(["x", ast.unparse(st)])
    return prog


def _class_info(cls):
    """One config class: its __init__ signature defaults, attribute_map, sub-configs, the statement program
    (_program), its properties (rw / ro: a setattr on a read-only one fails) and JSON-able class attributes."""
    from transformers import PretrainedConfig
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    params = {}
    for n, p in sig.parameters.items():
        if n == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if p.default is inspect.Parameter.empty:
            params[n] = REQUIRED
        else:
            params[n] = p.default if _json_ok(p.default) else NONJSON
    subs = {}
    sc_map = getattr(cls, "sub_configs", {})
    for k, sc in (sc_map.items() if isinstance(sc_map, dict) else ()):
        if isinstance(sc, type) and issubclass(sc, PretrainedConfig):
            subs[k] = sc.__name__
        else:
            subs[k] = None                                  # AutoConfig: the sub-config's own model_type decides
    props, attrs = {}, {}
    for klass in reversed(cls.__mro__):
        if klass is object or klass is PretrainedConfig:   # (PretrainedConfig's own are emulated in the JS)
            continue
        for k, v in vars(klass).items():
            if isinstance(v, property):
                props[k] = "rw" if v.fset is not None else "ro"
            elif (not k.startswith("__") and not callable(v) and not isinstance(v, (classmethod, staticmethod, dict, list))
                  and _json_ok(v) and k not in ("sub_configs", "attribute_map", "base_config_key")):
                attrs[k] = v
    prog = _program(cls)
    methods, getters = _methods(cls, [st[1] for st in prog or [] if st[0] == "x"])
    return {"mt": getattr(cls, "model_type", "") or "", "params": params,
            "amap": dict(getattr(cls, "attribute_map", {}) or {}), "subs": subs, "prog": prog,
            "props": props, "attrs": attrs, "methods": methods, "getters": getters}


MOD_PREFIX = "transformers.models."


def _keep_classes(mapping, classes):
    """The config classes the tables carry: the model types with a causal-LM, image-text-to-text or vision-to-text
    head (the LMs a config lookup is for), their sub-configs, and the classes / model types their __init__ builds by
    name. A config of any other stock type falls back to the raw path, as a config the stock class rejects does."""
    from transformers.models.auto import modeling_auto as ma
    types = set(ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES) | set(ma.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES) | set(
        ma.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
    todo = [mapping[t] for t in types if t in mapping]
    keep = set()
    while todo:
        c = todo.pop()
        if c in keep or c not in classes or classes[c] is None:
            continue
        keep.add(c)
        info = classes[c]
        todo += [x for x in (info.get("subs") or {}).values() if x]
        for st in info.get("prog") or []:
            if st[0] != "x":
                continue
            todo += re.findall(r"\b([A-Z]\w*Config)\(", st[1])
            for t in re.findall(r"CONFIG_MAPPING\['(\w+)'\]|'model_type', '(\w+)'", st[1]):
                for x in t:
                    if x in mapping:
                        todo.append(mapping[x])
    return keep


def _causal_lm_types():
    from transformers.models.auto import modeling_auto as ma
    return set(ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)


def tf_tables():
    """The transformers facts: model types, config classes, class defaults, stock model classes."""
    import transformers
    from transformers import CONFIG_MAPPING, PretrainedConfig, PreTrainedModel
    lg = logging.getLogger("transformers")
    lvl = lg.level
    lg.setLevel(logging.ERROR)
    AST_TYPES.clear()
    _AST_FIELDS.clear()
    try:
        mapping, classes, cdef = {}, {}, {}

        def add(cls):
            if cls.__name__ in classes:
                return
            classes[cls.__name__] = _class_info(cls)
            sc_map = getattr(cls, "sub_configs", {})
            for sc in (sc_map.values() if isinstance(sc_map, dict) else ()):
                if isinstance(sc, type) and issubclass(sc, PretrainedConfig):
                    add(sc)

        for mt in CONFIG_MAPPING.keys():
            try:
                cls = CONFIG_MAPPING[mt]
            except Exception:                               # an optional dependency the mapping cannot import
                continue
            mapping[mt] = cls.__name__
            add(cls)
            try:
                inst = cls()
            except Exception:
                cdef[mt] = None                             # _class_config: None (no class defaults)
                continue
            vals = {}
            for k in CDEF_KEYS:
                if hasattr(inst, k):
                    v = getattr(inst, k)
                    vals[k] = v if _json_ok(v) else NONJSON
            cdef[mt] = vals
        keep = _keep_classes(mapping, classes)
        classes = {k: v for k, v in classes.items() if k in keep}
        stmts, sid = [], {}
        for info in classes.values():                     # the statement ASTs, each stored once
            for st in (info or {}).get("prog") or []:
                if st[0] == "x":
                    if st[1] not in sid:
                        sid[st[1]] = len(stmts)
                        stmts.append(st[1])
                    st[1] = sid[st[1]]
                elif st[0] == "a" and st[1] == st[2]:
                    del st[2]
        archs = {}
        for name in dir(transformers):
            try:
                obj = getattr(transformers, name)
            except Exception:
                continue
            if not (isinstance(obj, type) and issubclass(obj, PreTrainedModel)):
                continue
            cc = getattr(obj, "config_class", None)
            ccn = cc.__name__ if isinstance(cc, type) else None
            if ccn is not None and ccn not in keep:
                continue                                    # a model of a type the port reads raw anyway
            mod = str(getattr(obj, "__module__", ""))
            archs[name.lower()] = [ccn, "~" + mod[len(MOD_PREFIX):] if mod.startswith(MOD_PREFIX) else mod]
        try:
            import torch
            dtypes = sorted(n for n in dir(torch) if isinstance(getattr(torch, n, None), torch.dtype))
        except Exception:          # pragma: no cover
            dtypes = []
        stmts_ast = [_encode(ast.parse(t).body[0]) for t in stmts]
        return {"version": transformers.__version__, "mapping": mapping, "classes": classes,
                "stmts_ast": stmts_ast, "ast_types": AST_TYPES, "ast_fields": _AST_FIELDS,
                "causal_lm_types": sorted(_causal_lm_types()),
                "cdef": cdef, "archs": archs, "torch_dtypes": dtypes}
    finally:
        lg.setLevel(lvl)


def build(out=OUT):
    from graph import arch_metrics as AM
    from graph import model_graph as mg
    data = {"mg": {n: _plain(getattr(mg, n)) for n in MG_NAMES},
            "am": {n: _plain(getattr(AM, n)) for n in AM_NAMES},
            "tf": tf_tables()}
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return data


if __name__ == "__main__":
    d = build()
    s = os.path.getsize(OUT)
    print(f"wrote {OUT}: {s / 1e3:.0f} kB, {len(d['tf']['mapping'])} model types, {len(d['tf']['classes'])} config "
          f"classes, {len(d['tf']['archs'])} model classes")
