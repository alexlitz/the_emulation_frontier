/* The in-browser port of graph/arch_metrics.py: a config.json -> params / KV / state / FLOPs / memory / bandwidth,
   the same record as Python's arch_metrics.metrics(raw, hub_total), with no Python and no transformers at runtime.

   It ports, function by function, what metrics() reaches: arch_metrics.plan() and its builders, the parts of
   graph/model_graph.py they call (classify / dims_of / _attn_counts / _moe_layer_mask / ...), and the transformers
   4.57 config-class behaviour arch_metrics.resolve() relies on (stock_arch, from_dict, get_text_config, to_dict).
   Every table it needs (model_graph's per-type sets, transformers' config-class signatures and defaults) comes from
   graph/data/arch_metrics_js_tables.json, written by `python -m graph.arch_metrics_js`.

   Python semantics are kept where they change a number: truthiness, int() truncation, // and % flooring, round()
   to even, exactly rounded int / int division (BigInt where products pass 2^53), dict insertion order.
   tests/test_arch_metrics_js.py runs it under node on every snapshot config against the Python.

   API:  ArchMetrics.metrics(raw, hubTotal = null, tables)   ArchMetrics.plan(raw, tables)
         ArchMetrics.setTables(tables)                        (tables then optional) */
(function (root) {
  "use strict";
  let T = null;                       // the tables
  let MG = null, AMT = null, TF = null;
  const SETS = {};                    // model_graph set name -> Set

  // the compact ASTs of the tables ([type index, *fields]) -> {_: type, field: value} nodes, once
  function decodeAst(x, types, fields) {
    if (Array.isArray(x)) {
      if (x.length && typeof x[0] === "number") {
        const t = types[x[0]], o = { _: t };
        fields[t].forEach((f, i) => { o[f] = decodeAst(x[i + 1], types, fields); });
        return o;
      }
      return x.map(y => decodeAst(y, types, fields));
    }
    return x;
  }
  function decodeTables(tf) {
    if (tf.__decoded) return;
    const dec = x => decodeAst(x, tf.ast_types, tf.ast_fields);
    tf.stmts_ast = tf.stmts_ast.map(dec);
    for (const info of Object.values(tf.classes)) {
      if (!info) continue;
      for (const grp of [info.methods || {}, info.getters || {}]) for (const f of Object.values(grp)) { f.defaults = dec(f.defaults); f.body = dec(f.body); }
      for (const st of info.prog || []) if (st[0] === "S") for (const kw of st[1]) if (kw[1][0] === "e") kw[1][1] = dec(kw[1][1]);
    }
    Object.defineProperty(tf, "__decoded", { value: true });
  }
  function setTables(t) {
    if (!t || t === T) return;
    decodeTables(t.tf);
    T = t; MG = t.mg; AMT = t.am; TF = t.tf;
    for (const [k, v] of Object.entries(MG)) if (Array.isArray(v)) SETS[k] = new Set(v);
    for (const [k, v] of Object.entries(AMT)) if (Array.isArray(v)) SETS["am." + k] = new Set(v);
  }
  const S = name => SETS[name];

  // ------------------------------------------------------------------------------------------ Python semantics
  class PyError extends Error {
    constructor(type, msg) { super(type + ": " + msg); this.pyType = type; this.pyMsg = msg; }
  }
  const hasOwn = (o, k) => o !== null && typeof o === "object" && !Array.isArray(o) && Object.prototype.hasOwnProperty.call(o, k);
  const isBool = v => typeof v === "boolean";
  const isNumber = v => typeof v === "number";
  const isNumLike = v => isNumber(v) || isBool(v);                     // isinstance(v, (int, float))
  const isIntLike = v => isBool(v) || (isNumber(v) && Number.isInteger(v));   // isinstance(v, int)
  const isDict = v => v !== null && typeof v === "object" && !Array.isArray(v);
  const isList = Array.isArray;
  const isStr = v => typeof v === "string";
  function truthy(v) {
    if (v === null || v === undefined) return false;
    if (isBool(v)) return v;
    if (isNumber(v)) return v !== 0;
    if (isStr(v) || isList(v)) return v.length > 0;
    if (isDict(v)) return Object.keys(v).length > 0;
    return true;
  }
  // d.get(k, default): the value when the key is present (None included), else the default
  function get(d, k, def = null) {
    if (!isDict(d)) throw new PyError("AttributeError", `'${pyType(d)}' object has no attribute 'get'`);
    return Object.prototype.hasOwnProperty.call(d, k) ? d[k] : def;
  }
  const getOr = (d, k) => (isDict(d) ? (Object.prototype.hasOwnProperty.call(d, k) ? d[k] : null) : null);
  function por(...xs) {                                                  // a or b or ...
    for (let i = 0; i < xs.length - 1; i++) if (truthy(xs[i])) return xs[i];
    return xs[xs.length - 1];
  }
  function pyType(v) {
    if (v === null || v === undefined) return "NoneType";
    if (isBool(v)) return "bool";
    if (isNumber(v)) return Number.isInteger(v) ? "int" : "float";
    if (isStr(v)) return "str";
    if (isList(v)) return "list";
    return "dict";
  }
  function pyRepr(v) {
    if (isStr(v)) return "'" + v.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "'";
    return pyStr(v);
  }
  function floatRepr(x) {
    if (Number.isNaN(x)) return "nan";
    if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
    if (Number.isInteger(x) && Math.abs(x) < 1e16) return x.toFixed(1);
    let s = String(x);
    if (s.includes("e")) {
      let [m, e] = s.split("e");
      const sign = e[0] === "-" ? "-" : "+";
      e = e.replace(/^[+-]/, "");
      if (e.length < 2) e = "0" + e;
      return m + "e" + sign + e;
    }
    const ax = Math.abs(x);
    if (ax !== 0 && ax < 1e-4) {
      const ex = x.toExponential();
      let [m, e] = ex.split("e");
      const sign = e[0] === "-" ? "-" : "+";
      e = e.replace(/^[+-]/, "");
      if (e.length < 2) e = "0" + e;
      return m + "e" + sign + e;
    }
    return s;
  }
  function pyStr(v) {
    if (v === null || v === undefined) return "None";
    if (isBool(v)) return v ? "True" : "False";
    if (isNumber(v)) return Number.isInteger(v) ? String(v) : floatRepr(v);
    if (isStr(v)) return v;
    if (isList(v)) return "[" + v.map(pyRepr).join(", ") + "]";
    return "{" + Object.entries(v).map(([k, x]) => pyRepr(k) + ": " + pyRepr(x)).join(", ") + "}";
  }
  const lower = v => pyStr(v).toLowerCase();
  function pyInt(v) {                                                    // int(v)
    if (isBool(v)) return v ? 1 : 0;
    if (isNumber(v)) {
      if (!Number.isFinite(v)) throw new PyError(Number.isNaN(v) ? "ValueError" : "OverflowError", "cannot convert float to integer");
      return Math.trunc(v) || 0;
    }
    if (isStr(v)) {
      const s = v.trim().replace(/_/g, "");
      if (/^[+-]?\d+$/.test(s) && !/^[+-]?_|__/.test(v.trim())) return parseInt(s, 10);
      throw new PyError("ValueError", `invalid literal for int() with base 10: ${pyRepr(v)}`);
    }
    throw new PyError("TypeError", `int() argument must be a string, a bytes-like object or a real number, not '${pyType(v)}'`);
  }
  function pyFloat(v) {
    if (isBool(v)) return v ? 1 : 0;
    if (isNumber(v)) return v;
    if (isStr(v)) {
      const s = v.trim().toLowerCase();
      if (/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/.test(s)) return parseFloat(s);
      if (/^[+-]?(inf|infinity)$/.test(s)) return s[0] === "-" ? -Infinity : Infinity;
      if (/^[+-]?nan$/.test(s)) return NaN;
      throw new PyError("ValueError", `could not convert string to float: ${pyRepr(v)}`);
    }
    throw new PyError("TypeError", `float() argument must be a string or a real number, not '${pyType(v)}'`);
  }
  const numVal = v => (isBool(v) ? (v ? 1 : 0) : v);
  function _num(v) {                                                     // model_graph._num
    if (isList(v)) {
      const nums = v.filter(isNumLike).map(numVal);
      return nums.length ? Math.trunc(Math.max(...nums)) : 0;
    }
    return isNumLike(v) ? Math.trunc(numVal(v)) : 0;
  }
  function floordiv(a, b) {
    a = numVal(a); b = numVal(b);
    if (b === 0) throw new PyError("ZeroDivisionError", "integer division or modulo by zero");
    if (Number.isInteger(a) && Number.isInteger(b)) {
      if (Math.abs(a) < 2 ** 52 && Math.abs(b) < 2 ** 52) {
        const q = Math.floor(a / b);
        // exactness check: q * b <= a < (q + 1) * b for b > 0
        const r = a - q * b;
        if (b > 0 ? (r >= 0 && r < b) : (r <= 0 && r > b)) return q;
      }
      const A = BigInt(a), B = BigInt(b);
      let q = A / B;
      if ((A % B !== 0n) && ((A < 0n) !== (B < 0n))) q -= 1n;
      return Number(q);
    }
    return Math.floor(a / b);
  }
  function pymod(a, b) {
    a = numVal(a); b = numVal(b);
    if (b === 0) throw new PyError("ZeroDivisionError", "integer division or modulo by zero");
    const r = a % b;
    return (r !== 0 && (r < 0) !== (b < 0)) ? r + b : (r === 0 ? 0 : r);
  }
  function pyRound(x) {                                                  // round(x): half to even
    if (Number.isInteger(x)) return x;
    const f = Math.floor(x), diff = x - f;
    if (diff > 0.5) return f + 1;
    if (diff < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1;
  }
  // exact decimal digits of a finite double: [sign, digits BigInt, exp10] with value = digits * 10^exp10
  function exactDecimal(x) {
    const buf = new DataView(new ArrayBuffer(8));
    buf.setFloat64(0, x);
    const hi = buf.getUint32(0), lo = buf.getUint32(4);
    const sign = hi >>> 31 ? -1 : 1;
    const ex = (hi >>> 20) & 0x7ff;
    let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
    let e2;
    if (ex === 0) e2 = -1074; else { mant |= 1n << 52n; e2 = ex - 1075; }
    if (e2 >= 0) return [sign, mant << BigInt(e2), 0];
    // mant * 2^e2 = mant * 5^-e2 / 10^-e2
    return [sign, mant * 5n ** BigInt(-e2), e2];
  }
  function pyRoundN(x, n) {                                              // round(float, n), n >= 0
    if (!Number.isFinite(x) || x === 0) return x;
    const [sign, digits, e10] = exactDecimal(x);
    // value = digits * 10^e10 ; want q = round_half_even(value * 10^n)
    const shift = -(e10 + n);
    let q;
    if (shift <= 0) q = digits * 10n ** BigInt(-shift);
    else {
      const p = 10n ** BigInt(shift);
      q = digits / p;
      const r = digits % p, twice = 2n * r;
      if (twice > p || (twice === p && (q % 2n === 1n))) q += 1n;
    }
    const s = q.toString();
    let str;
    if (n === 0) str = s;
    else {
      const padded = s.padStart(n + 1, "0");
      str = padded.slice(0, padded.length - n) + "." + padded.slice(padded.length - n);
    }
    const v = parseFloat(str);
    return sign < 0 ? -v : v;
  }
  // exactly rounded a / b for BigInt a, b > 0 (Python int / int)
  function divExact(a, b) {
    if (b === 0n) throw new PyError("ZeroDivisionError", "division by zero");
    let neg = false;
    if (a < 0n) { a = -a; neg = !neg; }
    if (b < 0n) { b = -b; neg = !neg; }
    if (a === 0n) return 0;
    const bl = x => x.toString(2).length;
    // scale so the integer quotient has 55..56 bits
    let k = 55 - (bl(a) - bl(b));
    let A = a, B = b;
    if (k > 0) A = a << BigInt(k); else if (k < 0) B = b << BigInt(-k);
    let q = A / B, r = A % B;
    // q has 55 or 56 bits; round to 53 bits, half to even with the sticky remainder
    const qb = bl(q);
    const drop = qb - 53;
    const mask = (1n << BigInt(drop)) - 1n;
    let low = q & mask, top = q >> BigInt(drop);
    const half = 1n << BigInt(drop - 1);
    if (low > half || (low === half && (r !== 0n || (top & 1n) === 1n))) top += 1n;
    else if (low === half && r === 0n && (top & 1n) === 0n) { /* tie to even: keep */ }
    // value = top * 2^(drop - k)
    let v = Number(top) * Math.pow(2, drop - k);
    return neg ? -v : v;
  }
  const big = v => BigInt(v);
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  const pyMin = (...xs) => xs.reduce((a, b) => (b < a ? b : a));
  const pyMax = (...xs) => xs.reduce((a, b) => (b > a ? b : a));
  function pyEq(a, b) {                                                  // a == b for JSON values
    if (isNumLike(a) && isNumLike(b)) return numVal(a) === numVal(b);
    if (a === null || b === null) return a === b;
    if (isStr(a) || isStr(b)) return a === b;
    if (isList(a) && isList(b)) return a.length === b.length && a.every((x, i) => pyEq(x, b[i]));
    if (isDict(a) && isDict(b)) {
      const ka = Object.keys(a), kb = Object.keys(b);
      return ka.length === kb.length && ka.every(k => hasOwn(b, k) && pyEq(a[k], b[k]));
    }
    return false;
  }
  const inTuple = (v, xs) => xs.some(x => pyEq(v, x));
  function deepcopy(v) { return v === null || typeof v !== "object" ? v : JSON.parse(JSON.stringify(v)); }
  // json.dumps(g, sort_keys=True, default=str) as an identity key
  function sortedKey(v) {
    if (isList(v)) return "[" + v.map(sortedKey).join(",") + "]";
    if (isDict(v)) return "{" + Object.keys(v).sort().map(k => JSON.stringify(k) + ":" + sortedKey(v[k])).join(",") + "}";
    if (v instanceof Set) return sortedKey([...v].sort());
    return JSON.stringify(v === undefined ? null : v);
  }

  // ------------------------------------------------------------------------------------------ model_graph.py
  function _classConfigVals(mt) {                                        // _class_config(mt) -> getattr values
    if (!mt || !hasOwn(TF.cdef, mt)) return null;
    return TF.cdef[mt];
  }
  function _class_default(mt, key) {
    mt = lower(mt || "");
    const cfg = mt ? _classConfigVals(mt) : null;
    if (cfg !== null && cfg !== undefined && hasOwn(cfg, key)) return cfg[key];
    const fb = MG._CLASS_DEFAULT_FALLBACK[key] || {};
    return hasOwn(fb, mt) ? fb[mt] : null;
  }
  function _cfg_value(d, key, outer = null) {
    if (hasOwn(d, key) && d[key] !== null) return d[key];
    if (outer && hasOwn(outer, key) && outer[key] !== null) return outer[key];
    let v = _class_default(getOr(d, "model_type"), key);
    if (v === null && outer) v = _class_default(getOr(outer, "model_type"), key);
    return v;
  }
  function raw_act(d) {
    const mt = lower(por(getOr(d, "model_type"), ""));
    let a;
    if (hasOwn(MG.ACT_KEY_BY_CODE, mt)) {
      const k = MG.ACT_KEY_BY_CODE[mt];
      a = por(getOr(d, k), MG.USE_ACT_CODE ? _class_default(mt, k) : null);
    } else {
      a = por(getOr(d, "mlp_hidden_act"), getOr(d, "hidden_activation"), getOr(d, "hidden_act"),
        getOr(d, "activation_function"), mt === "falcon" ? getOr(d, "activation") : null);
      if (!truthy(a) && MG.USE_ACT_CODE) {
        let nx = "";
        for (const k of ["hidden_act", "activation_function", "hidden_activation", "activation"]) {
          if (isStr(_class_default(mt, k))) { nx = _class_default(mt, k); break; }
        }
        a = por(hasOwn(MG.RAW_ACT_BY_CODE, mt) ? MG.RAW_ACT_BY_CODE[mt] : null, nx);
      }
    }
    return pyStr(por(a, "")).toLowerCase();
  }
  function _is_drafter(outer, d) {
    const mts = [lower(por(getOr(outer, "model_type"), "")), lower(por(getOr(d, "model_type"), ""))];
    return MG.DRAFTER_KEYS.some(k => hasOwn(outer, k) || hasOwn(d, k)) || mts.some(m => m.endsWith("_assistant"));
  }
  function skipped(c) {
    const d = _lm_config(c);
    if (!isDict(d)) return null;
    const mts = [lower(por(getOr(c, "model_type"), "")), lower(por(getOr(d, "model_type"), ""))];
    if (mts.some(m => S("SKIP_TYPES").has(m))) return "non_decoder";
    if (_is_drafter(c, d)) return "drafter";
    return null;
  }
  function _looped(d) {
    if (S("LOOPED_TYPES").has(getOr(d, "model_type"))) return true;
    if (MG._LOOP_COUNT_KEYS.some(k => _num(getOr(d, k)) > 1)) return true;
    const mt = lower(por(getOr(d, "model_type"), ""));
    if ((mt === "ouro" && por(_num(getOr(d, "total_ut_steps")), 4) > 1) || (mt === "cfrd" && _num(getOr(d, "physical_cells")))) return true;
    if (mt === "hrm_text") return true;
    return truthy(por(getOr(d, "mean_recurrence"), getOr(d, "num_recurrence"), getOr(d, "recurrence"),
      getOr(d, "n_layers_in_recurrent_block"), getOr(d, "tie_layers"), getOr(d, "share_layer_weights"),
      getOr(d, "reuse_layers"), getOr(d, "layer_sharing")));
  }
  function _loop_depth(d, n_layers) {
    const g = k => _num(getOr(d, k));
    const mt = lower(por(getOr(d, "model_type"), ""));
    if (mt === "hrm_text") {
      const H = hasOwn(d, "H_cycles") ? g("H_cycles") : 2;
      const L = hasOwn(d, "L_cycles") ? g("L_cycles") : 3;
      const P = por(g("num_layers_per_stack"), n_layers, 16);
      if (H && P) return [2 * P, P * H * (L + 1)];
    }
    if (getOr(d, "enable_double_loop_split") === true && n_layers) {
      const m = por(g("loop_middle_layers"), floordiv(n_layers, 2));
      if (m && !pymod(n_layers, m)) return [n_layers, n_layers + m * (floordiv(n_layers + m, m) - 1)];
    }
    if (mt === "cfrd" && g("physical_cells")) {
      const cells = por(g("physical_cells"), 2);
      const rec = por(g("inference_recurrences"), g("recurrences"), 6);
      const bind = getOr(d, "use_binding_block") === true ? 1 : 0;
      return [pyMin(cells, rec) + bind, rec + bind];
    }
    if (g("physical_layers")) return [g("physical_layers"), pyMax(n_layers, g("physical_layers") * pyMax(1, g("loop_count")))];
    if (g("loop_len") && g("loop_times")) return [pyMax(1, n_layers - g("loop_len") * (g("loop_times") - 1)), n_layers];
    for (const [span, rep] of [["n_recurrent_layers", "n_recurrent_loops"], ["loop_middle_layers", "num_loops"]]) {
      if (g(span) && g(rep)) return [n_layers, n_layers + g(span) * (g(rep) - 1)];
    }
    if (g("num_motifs") && g("motif_size") && g("recurrent_loops"))
      return [n_layers, n_layers + g("motif_size") * pyMax(0, g("recurrent_loops") - g("num_motifs"))];
    let rec = 1;
    for (const k of MG._LOOP_COUNT_KEYS) if (g(k) > 1) { rec = g(k); break; }
    if (mt === "ouro") rec = por(g("total_ut_steps"), 4);
    return [n_layers, n_layers * rec];
  }
  // model_graph._loop_layers: (n_layers, param_layers, eff_layers)
  function _loop_layers(d) {
    const g = (...keys) => {
      for (const k of keys) { const v = getOr(d, k); if (!(v === null || (isList(v) && v.length === 0) || v === "")) return _num(v); }
      return 0;
    };
    let n_layers = por(g("num_hidden_layers", "n_layer", "num_layers", "num_transformer_layers", "n_layers"), _layer_count(d));
    let param_layers, eff_layers;
    if (_looped(d)) {
      const rec = por(g("mean_recurrence", "num_recurrence", "recurrence"), 1);
      const blk = g("n_layers_in_recurrent_block");
      if (getOr(d, "model_type") === "albert") {
        param_layers = pyMax(1, g("num_hidden_groups")) * pyMax(1, g("inner_group_num"));
        eff_layers = por(n_layers, param_layers);
      } else if (blk) { param_layers = blk; eff_layers = blk * rec; }
      else if (!g("mean_recurrence", "num_recurrence", "recurrence")) {
        [param_layers, eff_layers] = _loop_depth(d, n_layers);
        if (!n_layers && lower(por(getOr(d, "model_type"), "")) === "cfrd") n_layers = param_layers;
      } else { param_layers = n_layers; eff_layers = n_layers * rec; }
    } else param_layers = eff_layers = n_layers;
    return [n_layers, param_layers, eff_layers];
  }
  // model_graph._auto_map_str: the auto_map class paths of the LM config and its wrapper, joined
  function _auto_map_str(d, outer = null) {
    const out = [];
    for (const x of [isDict(outer) ? outer : {}, d]) {
      const am = getOr(x, "auto_map");
      if (isDict(am)) for (const v of Object.values(am)) out.push(pyStr(v));
    }
    return out.join(" ");
  }
  function _int_list(v) {                                                // model_graph._int_list
    if (isStr(v)) { try { v = JSON.parse(v); } catch (e) { return []; } }
    return isList(v) ? v.filter(x => isNumber(x)).map(x => Math.trunc(x)) : [];
  }
  const _archsOf = (d, outer) => new Set([...(isList(getOr(d, "architectures")) ? d.architectures : []),
    ...(isList(getOr(outer || {}, "architectures")) ? outer.architectures : [])]);
  function _bailing_v3_alias(d) {
    const mt = lower(por(getOr(d, "model_type"), ""));
    return mt === "ming_v3_asr" || (!mt && (por(getOr(d, "architectures"), [])).includes("BailingMoeV3ForCausalLM"));
  }
  function _shared_mla(d) {
    const sm = getOr(d, "shared_mla");
    if (!isDict(sm) || get(sm, "enabled", true) === false) return null;
    const blk = Object.assign({}, MG._SABI_SHARED_MLA, sm);
    return (por(getOr(blk, "layers"), [])).some(i => isIntLike(i) && !isBool(i)) ? blk : null;
  }
  function _code_slot_override(d, outer = null) {
    const am = _auto_map_str(d, outer);
    const cr = getOr(d, "compress_ratios");
    for (const [path, slots] of Object.entries(MG.CODE_SLOT_OVERRIDES)) {
      if (am.includes(path) && !(isList(cr) && cr.some(x => _num(x)))) return slots;
    }
    return {};
  }
  function _nkv_alias(d) {
    if (MG._KV_STD_KEYS.some(k => { const v = getOr(d, k); return v !== null && v !== false; })) return 0;
    return _num(por(getOr(d, "n_kv_heads"), getOr(d, "n_kv_head")));
  }
  function _hier_moe(d) {
    if (lower(por(getOr(d, "model_type"), "")) !== "meditor2") return null;
    const D = _num(get(d, "num_macro_domains", 3)), Ss = _num(get(d, "num_sub_experts_per_domain", 5));
    return D * Ss > 1 ? [D, Ss, _num(get(d, "sub_intermediate_size", 768))] : null;
  }
  function _llama4_shared(d) {
    return ["llama4", "llama4_text"].includes(lower(por(getOr(d, "model_type"), ""))) && _num(getOr(d, "num_local_experts")) > 1;
  }
  function _moa(d) {
    if (lower(por(getOr(d, "model_type"), "")) !== "jetmoe") return null;
    const e = _num(get(d, "num_local_experts", get(d, "moe_num_experts", 8)));
    const k = _num(get(d, "num_experts_per_tok", get(d, "moe_top_k", 2)));
    return (e > 1 && k >= 1) ? [e, k] : null;
  }
  function _emb_rank(d) {
    if (lower(por(getOr(d, "model_type"), "")) !== "opt") return 0;
    const wpd = _num(getOr(d, "word_embed_proj_dim")), h = _num(getOr(d, "hidden_size"));
    return (wpd && h && wpd !== h) ? wpd : 0;
  }
  function _qk_gain(d, mt) {
    if (!hasOwn(MG.QK_GAIN_KEYS, mt)) return null;
    const [key, def, both] = MG.QK_GAIN_KEYS[mt];
    const v = getOr(d, key);
    const s = (isNumber(v)) ? v : def;
    if (s === null || s === undefined) return null;
    const gg = both ? s * s : s;
    return Math.abs(gg - 1.0) > 1e-9 ? gg : null;
  }
  function _ffn_width_by_code(d, mt, hidden) {
    if (!hidden) return 0;
    if (mt === "rwkv5") return Math.trunc(Math.floor((hidden * 3.5) / 32) * 32);
    if (mt === "ivme" && isNumLike(getOr(d, "ffn_mult")) && hasOwn(d, "ffn_mult")) {
      const w = pyInt(hidden * numVal(d.ffn_mult) * 2 / 3);
      return floordiv(w + 7, 8) * 8;
    }
    if (["gla", "delta_net", "hgrn_bit"].includes(mt)) {
      const hr = isNumLike(getOr(d, "hidden_ratio")) ? numVal(d.hidden_ratio) : 4;
      const w = pyInt(hidden * (hr || 4) * 2 / 3);
      return 256 * floordiv(w + 255, 256);
    }
    return 0;
  }
  // model_graph._sliding_layout (round 3, sliding_window_keys_unread)
  function _sliding_layout(d, mt) {
    const g = k => getOr(d, k);
    const n = _layer_count(d);
    if (!n) return {};
    const sl = kinds => kinds.map(k => (k === "local" ? "sliding_attention" : (k === "full" ? "full_attention" : k)));
    if (mt === "cost_wise_gemma" && _num(g("sliding_window"))) return { layer_types: sl(_range(n).map(i => (i % 2 === 0 ? "local" : "full"))) };
    if (mt === "needle") {
      const ex = g("extras");
      const gl = new Set(_int_list(isDict(ex) ? getOr(ex, "global_attention_layers") : null));
      return gl.size ? { layer_types: sl(_range(n).map(i => (gl.has(i) ? "full" : "local"))) } : {};
    }
    if (mt === "iquest_q1" && get(d, "use_hybrid_layers", true) !== false && get(d, "use_sliding_window", true) !== false) {
      const first = [...(por(g("first_layers_types"), ["full_attention"]))];
      const block = [...(por(g("hybrid_layers_types_block"), ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]))];
      const last = [...(por(g("last_layers_types"), ["full_attention", "full_attention", "full_attention"]))];
      let lt = first.slice();
      const nb = por(_num(g("num_hybrid_layers_block")), 21);
      for (let i = 0; i < nb; i++) lt = lt.concat(block);
      lt = lt.concat(last);
      return lt.length === n ? { layer_types: lt.map(pyStr) } : {};
    }
    if (mt === "motif" && truthy(g("use_sliding_window")) && _num(g("sliding_window"))) {
      const pat = pyStr(por(g("sliding_window_pattern"), "interleave")), per = por(_num(g("sliding_window_period")), 2);
      if (pat !== "all" && pat !== "interleave") return {};
      return { layer_types: sl(_range(n).map(i => ((pat === "all" || pymod(i, per) !== 0) ? "local" : "full"))), sliding_window: _num(g("sliding_window")) + 1 };
    }
    if (mt === "meditor" || mt === "meditor2") {
      const ge = por(_num(g("global_attn_every_n_layers")), 6);
      return { layer_types: sl(_range(n).map(i => (pymod(i + 1, ge) === 0 ? "full" : "local"))), sliding_window: por(_num(g("sliding_window_size")), 2048) };
    }
    if (mt === "gaggle") {
      const pat = pyStr(por(g("window_pattern"), "SSSL")).toUpperCase();
      if (!pat || [...pat].some(c => c !== "S" && c !== "L")) return {};
      const long_w = por(_num(g("sequence_len")), 2048);
      const kinds = _range(n).map(i => (pat[i % pat.length] === "L" ? "full" : "local"));
      kinds[kinds.length - 1] = "full";
      return kinds.includes("local") ? { layer_types: sl(kinds), sliding_window: -floordiv(-floordiv(long_w, 4), 128) * 128 + 1 } : {};
    }
    if (mt === "jet_nemotron") {
      const eac = g("efficient_attention_config");
      const swa = isDict(eac) ? getOr(eac, "swa") : null;
      const w = isDict(swa) ? _num(getOr(swa, "window_size")) : 0;
      return (w && (por(g("layer_types"), [])).map(pyStr).includes("swa")) ? { sliding_window: w } : {};
    }
    if (mt === "grug_moe") {
      const ge = por(_num(g("global_every")), 4);
      return { layer_types: sl(_range(n).map(i => ((pymod(i + 1, ge) === 0 || i === n - 1) ? "full" : "local"))), sliding_window: por(_num(g("sliding_window")), 2048) };
    }
    if (mt === "plamo3") {
      const pp = por(_num(g("sliding_window_pattern")), 8);
      return { layer_types: sl(_range(n).map(i => (pymod(i + 1, pp) === 0 ? "full" : "local"))), sliding_window: por(_num(g("window_size")), 2048) };
    }
    if (mt === "edgerock") {
      const swa = new Set(_int_list(g("fixed_memory_swa_layers"))), lt = g("layer_types");
      if (!swa.size || !isList(lt) || lt.length !== n) return {};
      return { layer_types: lt.map((x, i) => ((swa.has(i) && x !== "linear_attention") ? "sliding_attention" : x)), sliding_window: por(_num(g("fixed_memory_swa_window")), 1024) };
    }
    if (mt === "ethoric") {
      const wl = new Set(_int_list(g("attn_window_layers"))), w = _num(g("attn_window"));
      return (wl.size && w) ? { layer_types: sl(_range(n).map(i => (wl.has(i) ? "local" : "full"))), sliding_window: w } : {};
    }
    if (mt === "fabric" && pyStr(por(g("architecture"), "fabric")) !== "full")
      return { layer_types: Array(n).fill("sliding_attention"), sliding_window: por(_num(g("local_attention_window")), 2048) };
    return {};
  }
  function _stock_arch_mg(d) {                                           // model_graph._stock_arch
    const a = getOr(d, "architectures");
    return isList(a) && a.length && !truthy(getOr(d, "auto_map")) ? a[0] : null;
  }
  function _moe_disabled(d) {
    if (getOr(d, "is_moe") === false || getOr(d, "use_moe") === false || S("_DENSE_ARCHS").has(_stock_arch_mg(d))) return true;
    const Lh = _num(getOr(d, "num_hidden_layers"));
    if (Lh && _dense_first(d) >= Lh) return true;
    const lbt = getOr(d, "layers_block_type"), pat = getOr(d, "hybrid_override_pattern");
    if (isList(lbt) && lbt.length) {
      const s = new Set(lbt.map(lower));
      return s.has("mlp") && !s.has("moe");
    }
    if (isStr(pat) && pat) return pat.includes("-") && !pat.includes("E");
    return false;
  }
  function _dense_mlp_beside_moe(d) {
    const mt = pyStr(por(getOr(d, "model_type"), ""));
    return truthy(getOr(d, "enable_moe_block")) || (mt.startsWith("diffusion_gemma") && truthy(getOr(d, "num_experts")));
  }
  function _dense_first(d) {
    return por(_num(getOr(d, "first_k_dense_replace")), _num(getOr(d, "num_dense_layers")), _num(getOr(d, "n_dense_first_layers")),
      lower(por(getOr(d, "model_type"), "")) === "kohaku" ? _num(get(d, "first_k_dense", 1)) : 0);
  }
  const _range = n => Array.from({ length: Math.max(0, n) }, (_, i) => i);
  function _moe_layer_mask(d, n_layers) {
    const L = pyInt(por(n_layers, 0));
    if (L <= 0) return null;
    const mlt = getOr(d, "mlp_layer_types");
    if (isList(mlt) && mlt.length === L) return mlt.map(x => lower(x) !== "dense");
    const mlf = getOr(d, "moe_layer_freq");
    if (isList(mlf) && mlf.length === L) return mlf.map(x => truthy(_num(x)));
    const en = getOr(d, "moe_layers_enum");
    if (truthy(en)) {
      const items = isStr(en) ? en.split(",") : (isList(en) ? en : null);
      if (items === null) throw new PyError("TypeError", `'${pyType(en)}' object is not iterable`);
      const idx = new Set();
      for (const x of items) { const s = pyStr(x).trim(); if (/^[0-9]+$/.test(s)) idx.add(pyInt(isStr(x) ? x : x)); }
      if (idx.size) return _range(L).map(i => idx.has(i));
    }
    const mt = lower(por(getOr(d, "model_type"), ""));
    if (mt.startsWith("llama4")) {
      const ml = getOr(d, "moe_layers");
      if (isList(ml) && ml.length) { const s = new Set(ml.map(numVal)); return _range(L).map(i => s.has(i)); }
      const step = pyMax(1, por(_num(getOr(d, "interleave_moe_layer_step")), 1));
      const s = new Set(); for (let i = step - 1; i < L; i += step) s.add(i);
      return _range(L).map(i => s.has(i));
    }
    if (truthy(getOr(d, "mlp_only_layers")) || _num(getOr(d, "decoder_sparse_step")) > 1) {
      const only = new Set((por(getOr(d, "mlp_only_layers"), [])).filter(isIntLike).map(numVal));
      const step = pyMax(1, por(_num(getOr(d, "decoder_sparse_step")), 1));
      return _range(L).map(i => !only.has(i) && pymod(i + 1, step) === 0);
    }
    if (getOr(d, "moe_layer_start_index") !== null || _num(getOr(d, "moe_layer_interval")) > 1) {
      const start = _num(getOr(d, "moe_layer_start_index"));
      let end = getOr(d, "moe_layer_end_index");
      end = (end === null || _num(end) < 0) ? L - 1 : _num(end);
      const iv = pyMax(1, por(_num(getOr(d, "moe_layer_interval")), 1));
      return _range(L).map(i => pymod(i + 1, iv) === 0 && start <= i && i <= end);
    }
    if (mt === "jamba" || getOr(d, "expert_layer_period") !== null) {
      const per = pyMax(1, por(_num(getOr(d, "expert_layer_period")), 2));
      const off = getOr(d, "expert_layer_offset") !== null ? _num(getOr(d, "expert_layer_offset")) : 1;
      return _range(L).map(i => pymod(i, per) === off);
    }
    const k = _dense_first(d);
    let freq = !isList(getOr(d, "moe_layer_freq")) ? _num(getOr(d, "moe_layer_freq")) : 0;
    const step = mt === "motif" ? _num(getOr(d, "interleave_moe_layer_step")) : 0;
    if (k || freq > 1 || step > 1) {
      freq = pyMax(freq, step, 1);
      return _range(L).map(i => i >= k && pymod(i, freq) === 0);
    }
    return null;
  }
  function _rec_layer(x) {
    const s = lower(x);
    if (!MG._REC_LAYER_NAMES.some(r => s.includes(r))) return false;
    const softmax_name = (s.includes("attention") || s.includes("attn")) &&
      !["linear", "delta", "mamba", "ssm", "gdn", "kda"].some(r => s.includes(r));
    return !softmax_name;
  }
  function _kind(x) {
    const s = lower(x);
    if (S("_FFN_ENTRIES").has(s)) return "ffn";
    if (s === "*") return "full";
    if (s === "m" || s === "recurrent" || _rec_layer(s)) return "rec";
    if (MG._LOCAL_MARKS.some(m => s.includes(m))) return "local";
    if (["t", "a", "h", "hybrid"].includes(s) || ["attention", "attn", "full", "global"].some(m => s.includes(m))) return "full";
    return null;
  }
  function _all_attention_list(d) {
    if (["falcon_h1", "hymba"].includes(lower(por(getOr(d, "model_type"), ""))) || truthy(getOr(d, "pb_hybrid_layers"))) return false;
    for (const key of ["layers_block_type", "layer_types", "hybrid_override_pattern"]) {
      const v = getOr(d, key);
      const seq = isStr(v) ? [...v] : v;
      if (isList(seq) && seq.length && seq.every(isStr)) {
        const kinds = seq.map(_kind);
        return kinds.every(k => ["full", "local", "ffn"].includes(k)) && kinds.some(k => k === "full" || k === "local");
      }
    }
    return false;
  }
  function _block_experts(d) {
    const bc = getOr(d, "block_configs");
    if (!isList(bc)) return [];
    return bc.filter(b => isDict(b) && _num(getOr(b, "num_local_experts"))).map(b => [_num(getOr(b, "num_local_experts")), getOr(b, "sliding_window")]);
  }
  const _k_eq_v = d => truthy(getOr(d, "attention_k_eq_v")) || S("K_EQ_V_BY_CODE").has(lower(por(getOr(d, "model_type"), "")));
  const _k_eq_v_all = d => S("VNORM_TYPES").has(lower(por(getOr(d, "model_type"), "")));
  function _is_alibi(d) {
    const mt = lower(por(getOr(d, "model_type"), ""));
    return truthy(getOr(d, "alibi")) || truthy(getOr(por(getOr(d, "attn_config"), {}), "alibi")) || mt === "bloom" ||
      (mt === "baichuan" && !hasOwn(d, "max_position_embeddings"));
  }
  function _nope_attention(d, mt) {
    const use_rope_off = getOr(d, "use_rope") === false && mt !== "arkasr";
    return S("NOPE_BY_CODE").has(mt) || lower(por(getOr(d, "position_embedding_type"), "")) === "nope" ||
      getOr(d, "mla_use_nope") === true || use_rope_off || getOr(d, "attn_use_rope") === false ||
      (mt === "olmo_hybrid" && !truthy(getOr(d, "rope_theta")) && !truthy(getOr(por(getOr(d, "rope_parameters"), {}), "rope_theta"))) ||
      (mt === "zamba2" && getOr(d, "use_mem_rope") !== true);
  }
  function _has_shared_expert(d) {
    if (_llama4_shared(d)) return true;
    if (hasOwn(d, "n_shared_experts") && d.n_shared_experts !== null && _num(d.n_shared_experts) === 0) return _dense_mlp_beside_moe(d);
    return truthy(por(getOr(d, "n_shared_experts"), getOr(d, "shared_expert_intermediate_size"), getOr(d, "share_expert_dim"),
      getOr(d, "moe_num_shared_experts"), getOr(d, "moe_shared_expert_intermediate_size"), getOr(d, "shared_intermediate_size"))) ||
      _num(getOr(d, "num_shared_experts")) > 0 || _num(getOr(d, "num_shared_expert")) > 0 || truthy(getOr(d, "use_mixed_mlp_moe")) ||
      _dense_mlp_beside_moe(d);
  }
  const _moe_like = d => truthy(por(getOr(d, "num_experts"), getOr(d, "moe_num_experts"), getOr(d, "n_routed_experts")));
  function _layer_count(d) {
    for (const k of MG._LM_LAYER_KEYS) if (truthy(getOr(d, k))) return _num(d[k]);
    for (const k of MG._LM_LAYER_LIST_KEYS) { const v = getOr(d, k); if (isList(v) && v.length) return v.length; }
    return 0;
  }
  const _has_lm_dims = d => isDict(d) && MG._LM_DIM_KEYS.some(k => truthy(getOr(d, k))) && _layer_count(d) > 0;
  function _make_divisible(v, divisor) {
    const new_v = pyMax(divisor, floordiv(pyInt(v + divisor / 2), divisor) * divisor);
    return new_v < 0.9 * v ? new_v + divisor : new_v;
  }
  function _lm_unwrap(d) {
    if (isDict(d) && !truthy(por(getOr(d, "num_attention_heads"), getOr(d, "n_head")))) {
      for (const k of MG._LM_WRAP_KEYS) {
        const sub = getOr(d, k);
        const cand = isDict(sub) ? _lm_unwrap(sub) : null;
        if (cand !== null && _has_lm_dims(cand)) {
          const extra = {};
          for (const kk of MG._LOOP_COUNT_KEYS) if (hasOwn(d, kk)) extra[kk] = d[kk];
          return Object.assign({}, cand, extra, { model_type: por(getOr(d, "model_type"), getOr(cand, "model_type")) });
        }
      }
    }
    if (!isDict(d) || _has_lm_dims(d)) return d;
    for (const k of MG._LM_NEST_KEYS) {
      const sub = getOr(d, k);
      if (isDict(sub)) {
        let cand = _lm_unwrap(sub);
        if (_has_lm_dims(cand)) {
          if (!truthy(getOr(cand, "model_type")) && truthy(getOr(d, "model_type"))) cand = Object.assign({}, cand, { model_type: d.model_type });
          return cand;
        }
      }
    }
    return d;
  }
  function _normalize(d) {
    if (!isDict(d)) return d;
    const mt = lower(por(getOr(d, "model_type"), ""));
    const u = {};
    const g = k => getOr(d, k);
    if (mt === "gpt_neo") {
      u.num_attention_heads = g("num_heads");
      const al = g("attention_layers");
      if (isList(al) && al.length) {
        u.layer_types = al.map(x => pyStr(x) === "local" ? "sliding_attention" : "full_attention");
        if (al.some(x => pyStr(x) === "local")) u.sliding_window = por(g("window_size"), 256);
      }
    } else if (mt === "mpt") {
      const ac = por(g("attn_config"), {});
      const nh = g("n_heads");
      const at = pyStr(por(getOr(ac, "attn_type"), "multihead_attention"));
      Object.assign(u, {
        num_attention_heads: nh, alibi: get(ac, "alibi", !truthy(g("auto_map"))),
        num_key_value_heads: at.startsWith("multiquery") ? 1 : at.startsWith("grouped") ? por(getOr(ac, "kv_n_heads"), nh) : nh,
        intermediate_size: pyInt(numVal(por(g("expansion_ratio"), 4)) * numVal(por(g("d_model"), 0))),
        max_position_embeddings: g("max_seq_len"),
      });
      if (truthy(getOr(ac, "clip_qkv"))) u.clip_qkv = ac.clip_qkv;
      if (truthy(getOr(ac, "qk_ln"))) u.qk_layernorm = true;
    } else if (mt === "xglm") {
      u.num_attention_heads = g("attention_heads");
    } else if (mt === "llada" || mt === "hf_olmo") {
      const nh = g("n_heads");
      const hid = por(g("mlp_hidden_size"), pyInt(numVal(por(g("mlp_ratio"), 4)) * numVal(por(g("d_model"), 0))));
      const act = lower(por(g("activation_type"), "swiglu"));
      Object.assign(u, {
        num_attention_heads: nh,
        num_key_value_heads: por(g("n_kv_heads"), truthy(g("multi_query_attention")) ? 1 : nh),
        intermediate_size: act === "swiglu" ? floordiv(hid, 2) : hid,
        hidden_act: (act === "swiglu" || act === "silu") ? "silu" : act,
        max_position_embeddings: g("max_sequence_length"),
        tie_word_embeddings: truthy(g("weight_tying")),
      });
      if (truthy(g("rope")) && !truthy(g("rope_theta"))) u.rope_theta = 10000.0;
      if (lower(por(g("layer_norm_type"), "")) === "rms" && !hasOwn(d, "rms_norm_eps")) u.rms_norm_eps = 1e-5;
      if (truthy(g("include_bias"))) Object.assign(u, { attention_bias: true, mlp_bias: true });
      if (truthy(g("include_qkv_bias"))) u.qkv_bias = true;
      if (truthy(g("attention_layer_norm"))) u.qk_norm = true;
    } else if (mt === "openelm") {
      const dm = por(g("model_dim"), 0), div = por(g("ffn_dim_divisor"), 256);
      let mults = g("ffn_multipliers");
      const n = por(g("num_transformer_layers"), 0);
      mults = isList(mults) ? mults : Array(n).fill(por(mults, 4.0));
      Object.assign(u, {
        num_attention_heads: g("num_query_heads"), num_key_value_heads: g("num_kv_heads"),
        intermediate_size: mults.map(m => pyInt(_make_divisible(numVal(m) * dm, div))),
        tie_word_embeddings: truthy(g("share_input_output_layers")),
        rope_theta: por(g("rope_freq_constant"), 10000), max_position_embeddings: g("max_context_length"),
        hidden_act: g("activation_fn_name"),
      });
      if (pyStr(por(g("normalization_layer_name"), "")).startsWith("rms")) u.rms_norm_eps = por(g("rms_norm_eps"), 1e-6);
      if (truthy(g("normalize_qk_projections"))) u.qk_norm = true;
    } else if (mt === "internlm" && isDict(g("rotary"))) {
      u.rope_theta = get(g("rotary"), "base", 10000);
      if (truthy(getOr(g("rotary"), "type"))) u.rope_scaling = { type: g("rotary").type };
    } else if (mt === "chatglm") {
      if (truthy(g("multi_query_attention")) && truthy(g("multi_query_group_num"))) u.num_key_value_heads = g("multi_query_group_num");
      if (truthy(g("kv_channels"))) u.head_dim = g("kv_channels");
    } else if (mt === "gpt2" && pyStr(por(g("attention_head_type"), "")) === "multiquery") {
      Object.assign(u, { num_key_value_heads: 1, multi_query: true });
    } else if (mt === "falcon") {
      const nh = g("num_attention_heads");
      let nkv;
      if (truthy(g("new_decoder_architecture"))) nkv = por(g("num_kv_heads"), nh);
      else if (truthy(_cfg_value(d, "multi_query"))) nkv = 1;
      else nkv = nh;
      Object.assign(u, { num_key_value_heads: nkv, multi_query: pyEq(nkv, 1) });
    } else if (mt === "bamba" && truthy(g("attn_rotary_emb"))) {
      u.rotary_dim = g("attn_rotary_emb");
    }
    if (g("head_dim") === null) {
      const hid = _num(g("hidden_size")), nh0 = _num(g("num_attention_heads"));
      let hd0 = 0;
      if (nh0) {
        if (["nemotron_h", "nemotron_h_puzzle", "ceno"].includes(mt)) {
          if (g("attention_head_dim") !== null) hd0 = _num(g("attention_head_dim"));
          else if (mt !== "ceno" && !hasOwn(d, "attention_head_dim")) hd0 = 128;
        } else if (mt === "zamba" || mt === "zamba2") hd0 = por(_num(g("attention_head_dim")), floordiv(2 * hid, nh0));
        else if (mt === "jetmoe") hd0 = por(_num(g("kv_channels")), 128);
        else if (mt === "plamo2" || mt === "plamo2vl") hd0 = por(_num(g("hidden_size_per_head")), 128);
      }
      if (hd0 && hd0 !== floordiv(hid, nh0)) u.head_dim = hd0;
    }
    if (mt === "granite_switch" && _num(g("num_adapters")) > 0) {
      const n_sw = _num(g("num_hidden_layers"));
      if (n_sw > 2) {
        u.num_hidden_layers = n_sw - 2;
        if (isList(g("layer_types")) && g("layer_types").length === n_sw) u.layer_types = g("layer_types").slice(2);
      }
    }
    if ((mt === "lfm2" || mt === "lfm2_moe") && truthy(_cfg_value(d, "block_auto_adjust_ff_dim"))) {
      let inter = por(g("block_ff_dim"), g("intermediate_size"));
      if (isIntLike(inter) && truthy(inter)) {
        inter = pyInt(2 * numVal(inter) / 3);
        const mult = hasOwn(d, "block_ffn_dim_multiplier") ? g("block_ffn_dim_multiplier") : _class_default(mt, "block_ffn_dim_multiplier");
        if (mult !== null) {
          const mo = por(g("block_multiple_of"), _class_default(mt, "block_multiple_of"), 256);
          inter = pyInt(numVal(mult) * inter);
          inter = mo * floordiv(inter + mo - 1, mo);
        }
        u.intermediate_size = inter;
        u.block_auto_adjust_ff_dim = false;
      }
    }
    if ((mt === "llama4" || mt === "llama4_text") && !truthy(g("no_rope_layers"))) {
      const n = _num(g("num_hidden_layers"));
      const iv = por(g("no_rope_layer_interval"), _class_default("llama4_text", "no_rope_layer_interval"), 4);
      if (n) u.no_rope_layers = _range(n).map(i => (pymod(i + 1, iv) !== 0 ? 1 : 0));
    }
    let lt = hasOwn(u, "layer_types") ? u.layer_types : g("layer_types");
    const exm_default = mt === "exaone_moe" && !hasOwn(d, "sliding_window") && isList(lt) && lt.some(x => pyStr(x).includes("sliding"));
    if (exm_default) u.sliding_window = 4096;
    const kol = mt === "kolibri1";
    if ((["afmoe", "cohere2", "exaone4", "exaone4_5", "exaone_moe"].includes(mt) || kol) && isList(lt) && !truthy(g("no_rope_layers")) &&
      lt.some(x => pyStr(x).includes("sliding")) &&
      (truthy(g("sliding_window")) || mt === "afmoe" || mt === "cohere2" || exm_default || (kol && !hasOwn(d, "sliding_window")))) {
      u.no_rope_layers = lt.map(x => (pyStr(x).includes("sliding") ? 1 : 0));
    }
    if (!truthy(g("no_rope_layers"))) {
      const n = _num(g("num_hidden_layers"));
      if (["muse_glimmer", "muse_glimmer_text", "muse_spark", "muse_spark_text"].includes(mt)) {
        const lrt = g("layer_rope_theta");
        if (isList(lrt) && lrt.length) u.no_rope_layers = lrt.map(x => (truthy(x) ? 1 : 0));
        else if (n) u.no_rope_layers = _range(n).map(i => (pymod(n - 1 - i, 4) === 0 ? 0 : 1));
      } else if (mt === "domynedge" && isList(g("position_layers")) && g("position_layers").length) {
        u.no_rope_layers = g("position_layers").map(x => (pyStr(x) === "rope" ? 1 : 0));
      }
    }
    if (mt === "grug_moe" && !truthy(g("no_rope_layers"))) {
      const n = _num(g("num_hidden_layers"));
      if (n) u.no_rope_layers = _range(n).map(i => ((pymod(i + 1, 4) === 0 || i === n - 1) ? 0 : 1));
    }
    if (!truthy(g("num_attention_heads")) && _num(g("num_heads")) && [...(por(g("architectures"), []))].some(a => S("FLA_ARCHS").has(a))) {
      u.num_attention_heads = g("num_heads");
      u.num_key_value_heads = por(g("num_kv_heads"), g("num_heads"));
      if (g("intermediate_size") === null && g("architectures").includes("TransformerForCausalLM")) {
        const hr = isNumLike(g("hidden_ratio")) && hasOwn(d, "hidden_ratio") ? numVal(g("hidden_ratio")) : 4;
        u.intermediate_size = 256 * -floordiv(-pyInt(_num(g("hidden_size")) * (hr || 4) * 2 / 3), 256);
      }
    }
    {
      const n = _num(g("num_hidden_layers"));
      let sched = null;
      if (mt === "gptx3" && _num(g("gdn_layers_per_gqa"))) {
        const k = _num(g("gdn_layers_per_gqa"));
        sched = _range(n).map(i => pymod(i + 1, k + 1) === 0);
      } else if (mt === "klm") {
        const e = por(_num(g("attention_every")), 4);
        sched = _range(n).map(i => pymod(i, e) === e - 1);
      } else if (mt === "solar_open2") {
        const gl = g("gqa_layers");
        if (isList(gl)) { const st = new Set(gl.map(_num)); sched = _range(n).map(i => st.has(i)); }
        else if (_num(g("gqa_interval"))) sched = _range(n).map(i => pymod(i + 1, _num(g("gqa_interval"))) === 0);
      }
      if (sched && sched.length && !sched.every(x => x) && !isList(g("layer_types")))
        u.layer_types = sched.map(a => (a ? "full_attention" : "linear_attention"));
    }
    if ((mt === "bailing_hybrid" || _bailing_v3_alias(d)) && (truthy(g("layer_group_size")) || mt === "ming_v3_asr")) {
      const n = _num(g("num_hidden_layers")), G = por(_num(g("layer_group_size")), 5);
      u.layer_types = _range(n).map(i => ((pymod(i + 1, G) === 0 || i >= floordiv(n, G) * G) ? "full_attention" : "linear_attention"));
    }
    if (mt === "minicpm_sala" && isList(g("mixer_types"))) {
      u.layer_types = g("mixer_types").map(x => (pyStr(x).includes("lightning") ? "linear_attention" : "full_attention"));
    }
    Object.assign(u, _sliding_layout(d, mt));
    if (S("INCLUSIVE_WINDOW_TYPES").has(mt) && _num(g("sliding_window")) && isList(g("layer_types")) &&
      g("layer_types").some(x => pyStr(x).includes("sliding"))) u.sliding_window = _num(g("sliding_window")) + 1;
    if (getOr(_code_slot_override(d), "window") === "full") Object.assign(u, { sliding_window: null, use_sliding_window: false });
    let drop = [];
    if (mt === "needle" && _num(g("qk_head_dim")) && g("head_dim") === null) u.head_dim = _num(g("qk_head_dim"));
    if (mt === "cyrah" && lower(por(g("attn_impl"), "mla")) === "gqa") {
      drop = ["kv_lora_rank", "q_lora_rank", "index_topk", "index_n_heads", "index_head_dim"];
      const hd_c = _num(g("qk_nope_head_dim")) + _num(g("qk_rope_head_dim"));
      if (hd_c) {
        u.head_dim = hd_c;
        if (_num(g("rotary_dim")) >= hd_c) drop.push("rotary_dim");
      }
      const lt_c = hasOwn(u, "layer_types") ? u.layer_types : g("layer_types");
      if (!(isList(lt_c) && lt_c.some(x => pyStr(x) === "sliding_attention"))) Object.assign(u, { sliding_window: null, use_sliding_window: false });
    }
    const sm = _shared_mla(d);
    if (sm) {
      Object.assign(u, { kv_lora_rank: _num(sm.kv_lora_rank), qk_rope_head_dim: _num(sm.qk_rope_head_dim),
        qk_nope_head_dim: _num(sm.qk_nope_head_dim), v_head_dim: _num(sm.v_head_dim) });
      drop.push("q_lora_rank");
    }
    const sw0 = hasOwn(u, "sliding_window") ? u.sliding_window : g("sliding_window");
    const sw = _num(sw0), mpe = _num(por(hasOwn(u, "max_position_embeddings") ? u.max_position_embeddings : null, g("max_position_embeddings")));
    if (sw && mpe && sw >= mpe && !isList(sw0)) {
      Object.assign(u, { sliding_window: null, use_sliding_window: false });
      lt = hasOwn(u, "layer_types") ? u.layer_types : g("layer_types");
      if (isList(lt)) u.layer_types = lt.map(x => (pyStr(x) === "sliding_attention" ? "full_attention" : x));
    }
    const uu = {};
    for (const [k, v] of Object.entries(u)) if (v !== null && v !== undefined || k === "sliding_window") uu[k] = v === undefined ? null : v;
    if (drop.length) {
      const out = Object.assign({}, d, uu);
      for (const k of drop) delete out[k];
      return out;
    }
    return Object.keys(uu).length ? Object.assign({}, d, uu) : d;
  }
  function _lm_config(d) {
    if (isDict(d) && lower(por(getOr(d, "model_type"), "")) === "breeze" &&
      lower(por(getOr(d, "backbone_model_type"), "breeze")) !== "breeze" && _has_lm_dims(getOr(d, "backbone_config"))) {
      const bb = d.backbone_config;
      return _normalize(Object.assign({}, bb, { model_type: lower(por(getOr(bb, "model_type"), d.backbone_model_type)),
        tie_word_embeddings: get(d, "tie_word_embeddings", false) }));
    }
    let u = _lm_unwrap(d);
    if (isDict(u) && isDict(d) && !truthy(getOr(u, "model_type")) && !truthy(getOr(d, "model_type")) &&
      [..._archsOf(d, u)].some(a => S("LONGCAT_ARCHS").has(a))) u = Object.assign({}, u, { model_type: "longcat_flash" });
    return _normalize(u);
  }

  function _kda_lower_bound(d) {
    const mt = lower(por(getOr(d, "model_type"), ""));
    const la = isDict(getOr(d, "linear_attn_config")) ? d.linear_attn_config : {};
    if (getOr(la, "gate_lower_bound") !== null) return la.gate_lower_bound;
    for (const k of ["linear_lower_bound", "kda_gate_lower_bound"]) if (getOr(d, k) !== null) return d[k];
    if (truthy(getOr(d, "kda_safe_gate"))) return get(d, "kda_lower_bound", -5.0);
    if (mt.startsWith("glm5_next") && !hasOwn(d, "linear_lower_bound") && truthy(get(la, "safe_gate", true))) return -5.0;
    return null;
  }
  const _kda_variant = d => (_kda_lower_bound(d) !== null ? "kda_lb" : "kda");
  function _recurrent_family(d, mixer) {
    if (mixer === "softmax") return null;
    const mt = lower(por(getOr(d, "model_type"), ""));
    if (mt === "bailing_hybrid") {
      const arch = (por(getOr(d, "architectures"), [])).join(" ");
      return arch.includes("V3") ? _kda_variant(d) : "bailing_lightning";
    }
    if (_bailing_v3_alias(d)) return _kda_variant(d);
    if (mt === "minicpm_sala") return "sala_lightning";
    if (mt === "cyrah") return (truthy(getOr(d, "conv_dual_gate")) && !truthy(getOr(d, "conv_bias"))) ? "shortconv" : "other_cyrah";
    let fam = hasOwn(MG.RECURRENT_FAMILY, mt) ? MG.RECURRENT_FAMILY[mt] : null;
    if (fam === null) {
      if (truthy(por(getOr(d, "pb_hybrid_layers"), getOr(d, "linear_num_value_heads"), getOr(d, "gdn_layers_per_attention")))) fam = "gdn";
      else if (truthy(por(getOr(d, "mamba_num_heads"), getOr(d, "mamba_head_dim"), getOr(d, "mamba_n_heads")))) fam = "mamba2";
      else if (truthy(por(getOr(d, "mamba_expand"), getOr(d, "mamba_d_state"), getOr(d, "ssm_state_size")))) fam = "mamba1";
      else return "other_" + (mt || "unknown");
    }
    if (fam === "mamba2" && mt === "falcon_h1") {
      if (_cfg_value(d, "mamba_rms_norm") === false) return "mamba2_nonorm";
      if (truthy(_cfg_value(d, "mamba_norm_before_gate"))) return "mamba2_nbg";
    }
    if (fam === "gdn" && (mt === "qwen4_exp" || mt === "qwen4_exp_text") &&
      lower(por(getOr(d, "output_gate_type"), getOr(d, "hidden_act"), "silu")) === "sigmoid") return "gdn_sig";
    if (mt === "olmo_hybrid" && get(d, "linear_allow_neg_eigval", true) !== false) return "gdn_negeig";
    if (fam === "kda") {
      if (mt === "solar_open2" && get(d, "kda_gate_lower_bound", -5.0) !== null) return getOr(d, "kda_allow_neg_eigval") === true ? "kda_lb_negeig" : "kda_lb";
      return _kda_variant(d);
    }
    return fam;
  }
  function _tied(d, outer = null) {
    outer = outer || {};
    const mt = lower(por(getOr(d, "model_type"), ""));
    if (S("TIED_BY_CODE").has(mt)) return true;
    if (_emb_rank(d)) return false;
    for (const k of ["tie_word_embeddings", "tie_embedding", "weight_tying"]) {
      const v = hasOwn(d, k) ? d[k] : getOr(outer, k);
      if (v !== null) return truthy(v);
    }
    const archs = _archsOf(d, outer);
    if (!mt && archs.has("MiniCPMForCausalLM")) return truthy(_class_default("minicpm", "tie_word_embeddings"));
    if (mt === "minicpm" && !archs.has("MiniCPMForCausalLM")) return false;
    return truthy(_cfg_value(d, "tie_word_embeddings", outer));
  }
  function _qknorm_variant(d, mt) {
    const nt = lower(por(getOr(d, "qk_norm_type"), ""));
    if (nt === "per_layer") return "proj";
    if (mt === "bolmo") return truthy(getOr(d, "use_head_qk_norm")) ? "head" : "proj";
    if (["molmo", "hf_olmo", "olmo"].includes(mt)) {
      const lnt = lower(por(getOr(d, "layer_norm_type"), ""));
      return lnt === "rms" ? "proj" : "proj_ln";
    }
    if (_qk_gain(d, mt)) return "head_free_gain";
    if (_shared_mla(d)) return "q_head";
    if (hasOwn(MG.QKNORM_VARIANT_BY_TYPE, mt)) return MG.QKNORM_VARIANT_BY_TYPE[mt];
    if (nt === "per_head" || S("QKNORM_HEAD_CHECKED").has(mt)) return "head";
    return "unverified:" + (mt || "unknown");
  }
  function _archSet(d) {                                                 // set(d.get("architectures") or ())
    const a = por(getOr(d, "architectures"), []);
    if (isStr(a)) return new Set([...a]);
    return new Set(isList(a) ? a : Object.keys(a));
  }

  // classify(): the slots the metrics read (norm, pos, ffn, mixer, block, qknorm, and the features
  // bias_qkv / bias_out / out_gate / shared_gate / post_norm / post_only / dsa), or null where Python returns None
  function classify(raw) {
    const outer = raw;
    const d = _lm_config(raw);
    if (!isDict(d)) return null;
    const mt = lower(por(getOr(d, "model_type"), ""));
    if (skipped(outer)) return null;
    let mixer;
    const atl = getOr(d, "attn_type_list");
    if (S("SSM_TYPES").has(mt)) mixer = "ssm";
    else if (S("LINEAR_TYPES").has(mt)) mixer = "linear";
    else if (S("HYBRID_TYPES").has(mt) && !(isList(atl) && atl.length && atl.every(x => pyEq(x, 1)))) mixer = "hybrid";
    else if (["mamba_num_heads", "ssm_state_size", "mamba_d_state", "mamba_expand", "mamba_head_dim", "linear_num_value_heads",
      "linear_attention_pattern", "gdn_layers_per_attention"].some(k => hasOwn(d, k)) ||
      ["layers_block_type", "layer_types"].some(k => isList(getOr(d, k)) && d[k].some(_rec_layer))) {
      const src = por(getOr(d, "layers_block_type"), getOr(d, "layer_types"), getOr(d, "hybrid_override_pattern"), []);
      const lbt = (isStr(src) ? [...src] : src).map(lower);
      let has_full = lbt.some(x => ["attention", "full_attention", "attn", "*", "t", "a"].includes(x) || x.includes("full_attn") ||
        ((x.includes("attention") || x.includes("attn")) && !["linear", "delta", "mamba", "ssm", "gdn"].some(r => x.includes(r))));
      if (!lbt.length) {
        const lap = getOr(d, "linear_attention_pattern");
        const n_l = _layer_count(d);
        const hyb = Object.entries(d).filter(([k, v]) => k.endsWith("hybrid_layers") && isList(v)).map(([, v]) => v);
        has_full = truthy(por(getOr(d, "full_attention_layers"), getOr(d, "full_attention_interval"), getOr(d, "full_attention_idx"),
          getOr(d, "mla_layers"), getOr(d, "gdn_layers_per_attention"))) || (isList(lap) && lap.some(x => pyEq(x, 0))) ||
          hyb.some(v => 0 < v.length && v.length < n_l);
      }
      mixer = has_full ? "hybrid" : "ssm";
    } else mixer = "softmax";
    if (mixer === "hybrid" && _all_attention_list(d)) mixer = "softmax";
    const nh = _num(por(getOr(d, "num_attention_heads"), getOr(d, "n_head"))) || null;
    if (nh === null && mixer === "softmax") return null;
    const attn_cfg = por(getOr(d, "attn_config"), {});
    const rope_like = truthy(por(getOr(d, "rope_theta"), getOr(d, "rope_parameters"), getOr(d, "rope_scaling"), getOr(d, "rotary_dim"),
      getOr(d, "rotary_pct"), getOr(d, "partial_rotary_factor"), getOr(d, "swa_rope_theta"), getOr(attn_cfg, "rope_theta"))) ||
      getOr(d, "position_embedding_type") === "rotary" || truthy(getOr(d, "rope_embedding_base"));
    const n_exp_like = truthy(por(getOr(d, "num_experts"), getOr(d, "n_routed_experts"), getOr(d, "num_local_experts"),
      getOr(d, "moe_num_experts"), getOr(por(getOr(d, "ffn_config"), {}), "moe_num_experts"), getOr(d, "num_routed_experts"))) ||
      _block_experts(d).length > 0;
    const modern = rope_like || n_exp_like || S("GATED").has(mt);
    let norm;
    const archs = _archSet(d);
    if (_stock_arch_mg(d) === "StableLmForCausalLM") norm = "LayerNorm";
    else if (hasOwn(MG.NORM_BY_CODE, mt)) norm = MG.NORM_BY_CODE[mt];
    else if (mt === "molmo" && truthy(getOr(d, "layer_norm_type"))) norm = d.layer_norm_type === "rms" ? "RMSNorm" : "LayerNorm";
    else if (hasOwn(d, "rms_norm_eps") || S("NEMOTRON_H_TYPES").has(mt) || (mt === "chatglm" && truthy(getOr(d, "rmsnorm")))) norm = "RMSNorm";
    else if (S("RMS_LNEPS_TYPES").has(mt) || [...archs].some(a => S("RMS_LNEPS_ARCHS").has(a))) norm = "RMSNorm";
    else if (hasOwn(d, "layer_norm_epsilon") || hasOwn(d, "layer_norm_eps") || S("LN_FAMILIES").has(mt)) norm = "LayerNorm";
    else norm = (S("GATED").has(mt) || modern) ? "RMSNorm" : null;
    let pos;
    if (_is_alibi(d)) pos = "alibi";
    else if (mt === "parler_tts_decoder" && !_truthyFlag(get(d, "rope_embeddings", false))) pos = "sinusoidal";
    else if (S("SINUSOIDAL_TYPES").has(mt)) pos = "sinusoidal";
    else if (S("LEARNED_ABS").has(mt)) pos = "learned_abs";
    else if (mt === "mpt" && !truthy(getOr(d, "alibi")) && get(d, "learned_pos_emb", true) !== false) pos = "learned_abs";
    else if (rope_like || S("GATED").has(mt) || mt === "gpt_neox" || mt === "falcon" || modern) pos = "rope";
    else pos = null;
    if (mixer !== "ssm" && mixer !== "linear" && _nope_attention(d, mt)) pos = "nope";
    if (mixer === "ssm" || mixer === "linear") { if (pos === null) pos = "nope"; }
    // FFN
    const ffn_cfg = por(getOr(d, "ffn_config"), {});
    const be = _block_experts(d);
    let n_exp = por(getOr(d, "num_experts"), getOr(d, "n_routed_experts"), getOr(d, "num_local_experts"), getOr(d, "moe_num_experts"),
      getOr(ffn_cfg, "moe_num_experts"), getOr(d, "num_routed_experts"), be.length ? pyMax(...be.map(x => x[0])) : null);
    const hier = _hier_moe(d);
    if (hier) n_exp = hier[0] * hier[1];
    if (_moe_disabled(d) || _num(n_exp) <= 1) n_exp = null;
    const shared = _has_shared_expert(d);
    let ffn;
    if (truthy(n_exp)) ffn = shared ? "moe_shared" : "moe";
    else if (S("HADAMARD_FFN_TYPES").has(mt)) ffn = "hadamard";
    else if (S("FFN_NONE_TYPES").has(mt)) ffn = "none";
    else if (S("VANILLA").has(mt) || S("NEMOTRON_H_TYPES").has(mt) || S("UNGATED_BY_CODE").has(mt) || getOr(d, "mlp_use_swiglu") === false) ffn = "vanilla";
    else ffn = "glu";
    // activation: only whether classify gives one
    const a = raw_act(d);
    const actNone = a === "" && !hasOwn(MG.GELU_FORM_BY_CODE, mt) && !(S("GATED").has(mt) || mixer !== "softmax" || modern) &&
      !(mt === "gpt_oss" || mt === "gpt_oss_puzzle");
    // block
    let par;
    if (mt === "falcon") par = truthy(getOr(d, "new_decoder_architecture")) || truthy(_cfg_value(d, "parallel_attn"));
    else par = truthy(_cfg_value(d, "use_parallel_residual")) || truthy(getOr(d, "parallel_attn")) || truthy(getOr(attn_cfg, "parallel_attn")) ||
      S("PARALLEL_BY_CODE").has(mt);
    const block = par ? "parallel" : "sequential";
    // qk-norm
    let uqn = getOr(d, "use_qk_norm");
    if (uqn === null) { const fb = MG._CLASS_DEFAULT_FALLBACK.use_qk_norm || {}; uqn = hasOwn(fb, mt) ? fb[mt] : null; }
    let qknorm = (truthy(uqn) || truthy(getOr(d, "qk_layernorm")) || getOr(d, "qk_norm") === true ||
      truthy(getOr(d, "use_head_qk_norm")) || truthy(getOr(d, "attention_layer_norm")) || truthy(getOr(d, "qkv_layernorm")) ||
      truthy(getOr(d, "qk_norm_type")) || S("QKNORM_TYPES").has(mt)) ? "yes" : "no";
    if (getOr(d, "use_qk_norm") === false || getOr(d, "qk_norm") === false || getOr(d, "query_key_norm") === false)
      qknorm = !S("QKNORM_ALWAYS").has(mt) ? "no" : qknorm;
    if (S("QKNORM_KEY_NOT_SOFTMAX").has(mt) || [...archs].some(x => S("QKNORM_KEY_NOT_SOFTMAX_ARCHS").has(x))) qknorm = "no";
    if (S("QKNORM_BY_CODE").has(mt) && qknorm === "no" && getOr(d, "qk_norm") !== false) qknorm = "yes";
    if (mt === "ethoric" && lower(por(getOr(d, "qk_norm"), "")) === "per_head") qknorm = "yes";
    if (mt === "doge" && _auto_map_str(d, outer).includes("modeling_doge")) qknorm = "no";
    if (qknorm === "yes") qknorm = _qknorm_variant(d, mt);
    if (mixer !== "softmax") {
      norm = norm || "RMSNorm";
      pos = pos || (mixer === "ssm" || mixer === "linear" ? "nope" : "rope");
    }
    const no_attn = mixer === "ssm" || mixer === "linear";
    if (no_attn) { pos = "nope"; qknorm = "no"; }
    if (norm === null || pos === null || actNone) return null;
    let feats = _feats(d, outer !== d ? outer : null);
    if (no_attn) feats = new Set([...feats].filter(f => !S("ATTN_ONLY_FEATS").has(f) && !f.startsWith("out_gate")));
    return { norm, pos, ffn, mixer, block, qknorm, feats };
  }
  // model_graph._out_gate_key (round 3: out_gate[_head][_bias][_softplus|_silu] / out_gate_qlat, or null)
  function _out_gate_key(d, mt) {
    const g = k => getOr(d, k);
    let on = truthy(g("attn_output_gate")) || truthy(g("attention_output_gate")) || truthy(g("mla_use_output_gate")) ||
      truthy(g("headwise_attn_output_gate")) || truthy(g("use_head_wise_attn_gate")) || g("gating") === true ||
      (truthy(g("use_output_gate")) && mt !== "gla") || truthy(g("attn_use_output_gate")) || truthy(g("gate_attn_act_mode")) ||
      truthy(g("attention_gate_func")) || truthy(g("use_gqa_gate")) || truthy(g("gated_attention")) ||
      truthy(g("gated_attention_proj_granularity_type")) || S("OUT_GATE_TYPES").has(mt);
    let head = false, bias = false, act = "";
    let ag = lower(por(g("attn_gate"), ""));
    if (mt === "ethoric") {
      if (ag === "headwise" || ag === "elementwise") { on = true; head = ag === "headwise"; bias = true; }
    } else if (mt === "hrm_text") { on = true; bias = g("attention_bias") === true; }
    else if (hasOwn(MG.OUT_GATE_BY_CODE, mt)) {
      const k = MG.OUT_GATE_BY_CODE[mt];
      on = true; head = k.startsWith("head"); bias = k.endsWith("_bias");
    }
    const gt = get(d, "gating", true);
    if (mt === "laguna" || mt === "step3p7") {
      on = on && gt !== false;
      head = gt === true || lower(gt) === "per-head";
    } else if (mt === "youtu_vl_moe") head = lower(get(d, "mla_output_gate_type", "head_wise")) === "head_wise";
    else if (lower(por(g("gated_attention_proj_granularity_type"), "")) === "head_wise") head = true;
    else if ((mt === "spark2_5" || mt === "motif") && g("headwise_attn_output_gate") === true) head = true;
    else if (mt === "step3p5" && g("use_head_wise_attn_gate") === true) head = true;
    else if (mt === "ethoric" || mt === "slm_tavenel") {
      ag = ag || "headwise";
      if (ag === "headwise" || ag === "elementwise") { on = true; head = ag === "headwise"; }
    }
    if (mt === "motif" && truthy(g("elementwise_attn_output_gate"))) on = true;
    if (mt === "laguna") { on = get(d, "gating", true) !== false; act = on ? "softplus" : ""; }
    else if (mt === "k2_horizon") { const fn = lower(por(g("attention_gate_func"), "")); act = (fn === "softplus" || fn === "silu") ? fn : ""; }
    else if (mt === "spark2_5" && g("headwise_attn_output_gate") === true) act = lower(por(g("gate_attn_act_mode"), "")) === "silu" ? "silu" : "";
    if (mt === "ernie5_5_moe" && get(d, "use_gated_attn", true) !== false) return "out_gate_qlat";
    if (!on) return null;
    return "out_gate" + (head ? "_head" : "") + (bias ? "_bias" : "") + (act ? "_" + act : "");
  }
  // model_graph._truthy: a config switch the code tests with `if x` (or a string compare)
  function _truthyFlag(v) {
    if (isStr(v)) return !["", "false", "none", "no", "0", "off"].includes(v.trim().toLowerCase());
    return truthy(v);
  }
  function _dsv4_schema(d, mt) {
    const n = _num(getOr(d, "num_hidden_layers"));
    const lt = getOr(d, "layer_types"), mlt = getOr(d, "mlp_layer_types"), cr = getOr(d, "compress_ratios");
    let comp, hashed;
    if (isList(lt) && lt.length) comp = lt.slice(0, n || undefined).some(x => pyStr(x).includes("compressed"));
    else if (cr !== null) comp = isList(cr) && cr.some(x => _num(x));
    else comp = n > 0;
    if (isList(mlt) && mlt.length) hashed = mlt.slice(0, n || undefined).some(x => pyStr(x) === "hash_moe");
    else if (getOr(d, "num_hash_layers") !== null) hashed = _num(d.num_hash_layers) > 0;
    else hashed = n > 0;
    return [comp, hashed];
  }
  // the FEATURES the metrics read
  function _feats(d, outer = null) {
    const mt = lower(por(getOr(d, "model_type"), ""));
    const f = new Set();
    const g = k => getOr(d, k);
    const og = _out_gate_key(d, mt);
    if (og) f.add(og);
    if (_moe_like(d) || truthy(g("num_local_experts")) || truthy(g("n_routed_experts"))) {
      if (S("SHARED_GATE_TYPES").has(mt) && (truthy(g("shared_expert_intermediate_size")) || mt === "qwen3_next")) f.add("shared_gate");
    }
    if ((S("BIAS_QKV_TYPES").has(mt) && get(d, "use_qkv_bias", get(d, "qkv_bias", true)) !== false) ||
      ["attention_bias", "use_qkv_bias", "qkv_bias", "add_qkv_bias", "bias"].some(k => g(k) === true) ||
      (mt === "falcon" && g("bias") === true) || (mt === "mpt" && g("no_bias") === false)) f.add("bias_qkv");
    if (S("BIAS_OUT_TYPES").has(mt) || ["mlp_bias", "use_bias", "add_bias_linear", "enable_bias"].some(k => g(k) === true) ||
      (mt === "mpt" && g("no_bias") === false) || (S("O_BIAS_FROM_ATTN_BIAS").has(mt) && g("attention_bias") === true) ||
      (["internlm", "internlm2", "internlm3"].includes(mt) && g("bias") === true) ||
      (mt === "rwkv7" && get(d, "norm_bias", true) !== false)) f.add("bias_out");
    if (_moa(d)) f.add("bias_out");
    if (mt === "mm200" && truthy(get(d, "use_gated_attention", true))) f.add("out_gate_post");
    if (_shared_mla(d)) f.add("bias_qkv");
    const archs = (por(g("architectures"), getOr(outer || {}, "architectures"), [])).map(pyStr);
    const post_ln_mm = (mt === "minimax" && !archs.some(a => a.includes("MiniMaxM2"))) ||
      ((mt === "minimax_text_01" || mt === "minimax_m1") && _truthyFlag(g("postnorm")));
    if (S("POST_ONLY_TYPES").has(mt) || (mt === "molmo" && g("norm_after") === true) ||
      (mt === "bolmo" && pyStr(por(g("block_type"), "reordered_norm")) === "reordered_norm")) f.add("post_only");
    else if (!post_ln_mm && (S("POST_NORM_TYPES").has(mt) || g("use_post_norm") === true || g("use_sandwich_norm") === true ||
      g("post_norm") === true || g("postnorm") === true)) f.add("post_norm");
    const cr = g("compress_ratios");
    let no_compressor = isList(cr) && cr.length && !cr.some(x => _num(x));
    if (S("DSV4_SCHEMA_TYPES").has(mt)) no_compressor = !_dsv4_schema(d, mt)[0];
    const sp = isDict(g("sparse_config")) ? g("sparse_config") : {};
    if ((truthy(por(g("index_topk"), g("index_n_heads"), g("indexer_budget"), g("index_top_k"))) &&
      get(d, "enable_dsa", true) !== false && !no_compressor) ||
      truthy(getOr(por(g("sparse_attention_config"), {}), "use_sparse_attention")) || truthy(g("moba_topk")) ||
      (mt === "minicpm_sala" && truthy(getOr(sp, "topk")))) f.add("dsa");
    return f;
  }

  // dims_of(): the fields the metrics read
  function dimsOf(raw) {
    const d = _lm_config(raw);
    const g = (...keys) => {
      for (const k of keys) {
        const v = getOr(d, k);
        if (!(v === null || (isList(v) && v.length === 0) || v === "")) return _num(v);
      }
      return 0;
    };
    const hidden = g("hidden_size", "n_embd", "n_embed", "d_model", "hidden_dim", "model_dim", "dim", "embed_dim");
    const n_heads = g("num_attention_heads", "n_head");
    let n_exp = _moe_disabled(d) ? 0 : por(g("num_experts", "n_routed_experts", "num_local_experts", "num_routed_experts", "moe_num_experts"),
      _num(getOr(por(getOr(d, "ffn_config"), {}), "moe_num_experts")));
    const blk_exp = _block_experts(d).map(x => x[0]);
    if (!n_exp && blk_exp.length) n_exp = pyMax(...blk_exp);
    const hier = _hier_moe(d);
    if (hier) n_exp = hier[0] * hier[1];
    if (n_exp <= 1) n_exp = 0;
    let k = por(g("num_experts_per_tok", "moe_topk", "moe_top_k", "moe_k", ...MG._TOPK_ALT_KEYS),
      _num(getOr(por(getOr(d, "ffn_config"), {}), "moe_top_k")));
    const mt_ = lower(por(getOr(d, "model_type"), ""));
    if (!k && mt_ === "deep") k = g("top_k");
    if (hier) k = 1;
    let dense_inter = por(g("intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size", "ff_intermediate_size", "ffn_hidden"),
      _ffn_width_by_code(d, mt_, hidden), hidden ? 4 * hidden : 0);
    if (S("HADAMARD_FFN_TYPES").has(mt_) && hidden) dense_inter = 2 ** (pyInt(hidden) - 1).toString(2).length;
    const dmt = getOr(d, "model_type");
    if (S("FUSED_GLU_HALF").has(dmt) && !(dmt === "molmo" && _moe_like(d))) dense_inter = floordiv(dense_inter, 2);
    let ew_r = 0, sh_r = 0;
    if (n_exp && truthy(getOr(d, "num_cdmoe_experts"))) { ew_r = 1; sh_r = dense_inter; }
    else if (n_exp && isList(getOr(d, "size_experts")) && d.size_experts.length) {
      const per_layer = d.size_experts.map(row => (isList(row) ? row.map(_num) : [_num(row)]));
      ew_r = pyMax(...per_layer.map(r => pyMax(...r)));
    } else if (n_exp) {
      let ew = por(g("moe_intermediate_size", "moe_ffn_hidden_size", "expert_ffn_hidden_size", "expert_hidden_dim", "ffn_hidden_size",
        "routed_intermediate_size", "expert_intermediate_size"), dense_inter);
      if (mt_ === "zaya1_vl" && truthy(getOr(d, "gated_linear_unit")) && g("ffn_hidden_size")) ew = floordiv(g("ffn_hidden_size"), 2);
      if (hier) ew = hier[2];
      if (getOr(d, "model_type") === "deep") ew = pyMax(1, floordiv(dense_inter, n_exp));
      let shared = por(g("moe_shared_expert_intermediate_size"), g("n_shared_experts") * ew, g("shared_expert_intermediate_size"),
        g("shared_intermediate_size"), g("share_expert_dim"), g("moe_num_shared_experts") * ew, g("num_shared_experts") * ew,
        g("num_shared_expert") * dense_inter);
      if (!_has_shared_expert(d)) shared = 0;
      else if (_llama4_shared(d)) shared = g("intermediate_size");
      if (_dense_mlp_beside_moe(d)) shared = dense_inter;
      const bc = getOr(d, "block_configs");
      if (isList(bc) && bc.some(b => isDict(b) && getOr(b, "block_type") === "moe")) {
        const per = bc.filter(b => isDict(b) && getOr(b, "block_type") === "moe").map(b => [
          por(_num(getOr(b, "n_routed_experts")), n_exp), por(_num(getOr(b, "num_experts_per_tok")), k, 1),
          por(_num(getOr(b, "moe_intermediate_size")), ew)]);
        k = pyMax(...per.map(x => x[1]));
        ew = pyMax(...per.map(x => x[2]));
      }
      ew_r = ew; sh_r = shared;
    }
    if (S("FFN_NONE_TYPES").has(mt_)) dense_inter = 0;
    let [n_layers, param_layers, eff_layers] = _loop_layers(d);
    const lt = getOr(d, "layer_types");
    if (isList(lt) && lt.some(x => pyEq(x, "mlp")) && !lt.some(_rec_layer) && lt.length === n_layers) {
      const frac = lt.filter(x => pyStr(x) !== "mlp").length / lt.length;
      n_layers = pyMax(1, pyRound(n_layers * frac));
      param_layers = pyMax(1, pyRound(param_layers * frac));
      eff_layers = pyMax(1, pyRound(eff_layers * frac));
    }
    const bc = getOr(d, "block_configs");
    if (getOr(d, "model_type") === "nemotron-nas" && isList(bc) && bc.length) {
      const live = bc.filter(b => !(truthy(getOr(get(b, "attention", {}), "no_op")) && truthy(getOr(get(b, "ffn", {}), "no_op")))).length;
      param_layers = eff_layers = por(live, n_layers);
    }
    const mpe = g("max_position_embeddings", "max_seq_len", "n_positions", "seq_length");
    const max_context = mpe ? pyMin(pyInt(mpe), 1 << 20) : 4096;
    const moe_sig = n_exp ? [pyInt(n_exp), pyInt(por(k, 1)), pyInt(ew_r), pyInt(sh_r)] : [0, 0, 0, 0];
    return {
      hidden, layers: n_layers, heads: n_heads, inter: por(dense_inter, g("moe_intermediate_size")), experts: n_exp,
      param_layers, eff_layers, vocab: g("vocab_size", "n_vocab", "padded_vocab_size"), max_context, moe_sig,
    };
  }

  // _attn_counts(d, n): the per-kind counts kc
  function _attn_counts(d, n) {
    const mt = lower(por(getOr(d, "model_type"), ""));
    const g = k => _num(getOr(d, k));
    const sw_on = truthy(por(getOr(d, "sliding_window"), getOr(por(getOr(d, "attn_config"), {}), "sliding_window"))) &&
      truthy(get(d, "use_sliding_window", true));
    let kinds = null;
    const pat = getOr(d, "hybrid_override_pattern");
    const bc = getOr(d, "block_configs");
    if (mt === "nemotron-nas" && isList(bc) && bc.length) {
      kinds = [];
      for (const b of bc) {
        const a = por(getOr(b, "attention"), {}), f = por(getOr(b, "ffn"), {});
        if (truthy(getOr(a, "no_op")) && truthy(getOr(f, "no_op"))) continue;
        kinds.push((truthy(getOr(a, "no_op")) || truthy(getOr(a, "replace_with_linear"))) ? "noattn" : (truthy(getOr(a, "window_length")) ? "local" : "full"));
      }
    }
    if (mt === "hymba" && n) {
      const gi = new Set((por(getOr(d, "global_attn_idx"), [])).map(numVal));
      kinds = _range(n).map(i => ((!sw_on || gi.has(i)) ? "full" : "local"));
    }
    if (!(kinds && kinds.length)) {
      for (const key of ["layers_block_type", "layer_types", "layer_type"]) {
        const v = getOr(d, key);
        if (isList(v) && v.length && v.every(isStr)) { kinds = v.map(_kind); break; }
      }
    }
    if (isStr(pat) && pat && (kinds === null || S("NEMOTRON_H_TYPES").has(mt))) kinds = [...pat].map(_kind);
    if (kinds === null) {
      for (const [key, soft] of [["attn_type_list", 1], ["linear_attention_pattern", 0]]) {
        const v = getOr(d, key);
        if (isList(v) && v.length) kinds = v.map(x => (pyEq(x, soft) ? "full" : "rec"));
      }
      const hlp = getOr(d, "hybrid_layer_pattern");
      if (kinds === null && isList(hlp) && hlp.length && hlp.every(x => pyEq(x, 0) || pyEq(x, 1))) kinds = hlp.map(x => (pyEq(x, 0) ? "full" : "local"));
    }
    if (kinds === null) {
      const bt = por(getOr(d, "_block_types"), getOr(d, "block_types"));
      if (isList(bt) && bt.length && n) {
        kinds = _range(n).map(i => _kind(bt[i % bt.length]));
        kinds = kinds.map(k => (k === "full" && truthy(getOr(d, "attention_window_size")) ? "local" : k));
      }
    }
    if (kinds === null && n) {
      let idx = null;
      const lac = getOr(d, "linear_attn_config");
      if (isDict(lac) && isList(getOr(lac, "full_attn_layers"))) idx = lac.full_attn_layers.length;
      for (const key of ["full_attn_idxs", "attn_layer_indices", "full_attention_layers"]) if (idx === null && isList(getOr(d, key))) idx = d[key].length;
      if (idx === null && g("attn_layer_period")) {
        const per = g("attn_layer_period"), off = g("attn_layer_offset");
        idx = _range(n).filter(i => pymod(i, per) === off).length;
      }
      if (idx === null && g("full_attention_interval") && !["plamo2", "plamo2vl"].includes(mt)) idx = floordiv(n, g("full_attention_interval"));
      if (idx === null && g("gdn_layers_per_attention")) idx = floordiv(n, g("gdn_layers_per_attention") + 1);
      if (idx !== null) kinds = Array(Math.max(0, idx)).fill("full").concat(Array(Math.max(0, n - idx)).fill("rec"));
      else if (["plamo2", "plamo2vl"].includes(mt) && g("mamba_step")) {
        const st = g("mamba_step");
        const fi = new Set((por(getOr(d, "full_attention_idx"), [])).map(numVal));
        kinds = _range(n).map(i => (pymod(i, st) === floordiv(st, 2) ? (fi.has(i) ? "full" : "local") : "rec"));
      } else if (mt === "falcon_h1" || truthy(getOr(d, "pb_hybrid_layers"))) kinds = Array(n).fill(sw_on ? "local" : "full");
      else if (S("HYBRID_TYPES").has(mt) || mt === "phi4flash") kinds = Array(n).fill("rec");
      else if (S("SSM_TYPES").has(mt) || S("LINEAR_TYPES").has(mt)) kinds = Array(n).fill("rec");
    }
    if (kinds === null) {
      if (!sw_on) kinds = Array(Math.max(0, n)).fill("full");
      else if (mt === "gemma2") kinds = _range(n).map(i => (i % 2 === 0 ? "local" : "full"));
      else if (g("sliding_window_pattern") || mt === "gemma3" || mt === "gemma3_text") {
        const p = por(g("sliding_window_pattern"), 6);
        kinds = _range(n).map(i => (pymod(i + 1, p) === 0 ? "full" : "local"));
      } else if (["qwen2", "qwen3", "qwen2_moe", "qwen3_moe"].includes(mt) && getOr(d, "max_window_layers") !== null)
        kinds = _range(n).map(i => (i >= g("max_window_layers") ? "local" : "full"));
      else kinds = Array(Math.max(0, n)).fill("local");
    }
    let own = Array(kinds.length).fill(true);
    const nsh = g("num_kv_shared_layers");
    if (nsh) own = kinds.map((_, i) => i < kinds.length - nsh);
    const ksl = getOr(d, "kv_source_layers");
    if (isList(ksl) && ksl.length === kinds.length) own = ksl.map((x, i) => x === null || pyEq(x, i));
    if (truthy(getOr(d, "use_cla")) && _num(getOr(d, "cla_share_factor")) > 1) own = own.map((o, i) => o && pymod(i, _num(d.cla_share_factor)) === 0);
    if (isList(getOr(d, "kv_reuse_group"))) {
      for (const grp of d.kv_reuse_group) for (const j of (isList(grp) ? grp.slice(1) : [])) if (isIntLike(j) && numVal(j) < own.length) own[numVal(j)] = false;
    }
    if (isList(getOr(d, "kv_source_layer_ids")) && d.kv_source_layer_ids.length) {
      const lt = getOr(d, "layer_types");
      if (isList(lt) && lt.length === kinds.length) own = own.map((o, i) => o && !pyStr(lt[i]).includes("shared"));
      else { const src = new Set(d.kv_source_layer_ids.map(numVal)); own = own.map((o, i) => o && src.has(i)); }
    }
    const sep_ffn = kinds.includes("ffn");
    const keqv = _k_eq_v(d), keqv_all = _k_eq_v_all(d);
    const kc = { FU: 0, FT: 0, LU: 0, LT: 0, S: 0, R: 0 };
    kinds.forEach((k, i) => {
      if (k === "rec") { kc.R += 1; return; }
      if (k !== "full" && k !== "local") return;
      if (sep_ffn) {
        const nxt = kinds.slice(i + 1).find(kk => kk === "full" || kk === "local" || kk === "ffn");
        if (nxt !== "ffn") return;
      }
      if (!own[i]) { kc.S += 1; return; }
      const tied = (keqv && k === "full") || keqv_all;
      kc[(k === "full" ? "F" : "L") + (tied ? "T" : "U")] += 1;
    });
    if (mt === "falcon_h1" || mt === "hymba") kc.R = pyMax(kc.R, kinds.length);
    else if (isList(getOr(d, "pb_hybrid_layers"))) kc.R = pyMax(kc.R, d.pb_hybrid_layers.length);
    return kc;
  }

  // ------------------------------------------------------------------------------------------ transformers: stock_arch / resolve
  function _stock_config_class(mt) {
    return (mt !== null && mt !== undefined && truthy(mt) && hasOwn(TF.mapping, mt)) ? TF.mapping[mt] : null;
  }
  function stock_arch(raw) {
    const mt = getOr(raw, "model_type");
    if (_stock_config_class(mt) === null) return false;
    const arch = (por(getOr(raw, "architectures"), [null]))[0];
    if (!isStr(arch)) return true;
    if (mt === "mpt" && pyStr(get(por(getOr(raw, "attn_config"), {}), "attn_type", "")).startsWith("grouped")) return false;
    const st = getOr(raw, "text_config");
    if (isDict(st) && truthy(getOr(st, "model_type")) && _stock_config_class(st.model_type) === null) return false;
    const rec = hasOwn(TF.archs, arch.toLowerCase()) ? TF.archs[arch.toLowerCase()] : null;
    if (rec === null) return false;
    const [cfgName, m0] = rec;
    const mod = m0.startsWith("~") ? "transformers.models." + m0.slice(1) : m0;
    return cfgName === null || cfgName === _stock_config_class(mt) || mod.includes(mt);
  }
  // resolve() and the config-class emulation live in RESOLVE (filled below)
  let RESOLVE = null;
  const DEBUG = { on: false, log: [] };
  function resolve(raw, opts) {
    if (opts && opts.resolved) return opts.resolved;
    const mt = getOr(raw, "model_type");
    const cls = stock_arch(raw) ? _stock_config_class(mt) : null;
    if (cls !== null && RESOLVE) {
      try {
        const r = RESOLVE(raw, mt, cls);
        if (r) return [r, true];
      } catch (e) {
        if (!(e instanceof PyError)) throw e;              // a config the stock class rejects: read it raw
        if (DEBUG.on) DEBUG.log.push(e.message);
      }
    }
    return [_lm_config(raw), false];
  }

  // ------------------------------------------------------------------------------------------ arch_metrics.py: the plan
  const HEAD_DIM_FROM_HIDDEN = () => S("am.HEAD_DIM_FROM_HIDDEN");
  function _g(d, keys, def = 0) {
    for (const k of keys) {
      const v = getOr(d, k);
      if (isBool(v)) continue;
      if (isNumber(v)) return v;
    }
    return def;
  }
  function _norm_params(norm, width, bias = null) {
    if (norm === "none") return 0;
    if (bias === null) bias = norm === "LayerNorm";
    return width * (bias ? 2 : 1);
  }
  function _layer_kinds(d, raw, n) {
    const lt = getOr(d, "layer_types");
    const mt = lower(por(getOr(d, "model_type"), ""));
    let W = por(_g(d, ["sliding_window", "attention_chunk_size", "attention_window_size"]), null);
    if (mt === "gpt_neo") W = _g(d, ["window_size"], 256);
    let kinds = null;
    if (isList(lt) && lt.length === n && lt.every(isStr)) {
      kinds = [];
      for (const x of lt) {
        const s = x.toLowerCase();
        if (["full_attention", "attention", "global", "full", "global_attention"].includes(s)) kinds.push("full");
        else if (s.includes("sliding") || s.includes("chunk") || s === "local") kinds.push("sliding");
        else if (["linear_attention", "mamba", "conv", "recurrent", "ssm"].includes(s) || s.includes("linear") || s.includes("mamba")) kinds.push("rec");
        else kinds.push("full");
      }
    }
    if (kinds === null) {
      const kc = _attn_counts(getOr(raw, "model_type") === mt ? raw : d, n);
      const full = kc.FU + kc.FT, loc = kc.LU + kc.LT;
      kinds = [].concat(Array(full).fill("full"), Array(loc).fill("sliding"), Array(kc.R).fill("rec"), Array(kc.S).fill("shared"));
      kinds = kinds.concat(Array(Math.max(0, n - kinds.length)).fill("full"));
      kinds = kinds.slice(0, Math.max(0, n));
    }
    if (kinds.includes("sliding") && !truthy(W)) kinds = kinds.map(k => (k === "sliding" ? "full" : k));
    if (getOr(d, "use_sliding_window") === false && ["qwen2", "qwen2_moe", "qwen3", "qwen3_moe"].includes(mt)) kinds = kinds.map(k => (k === "sliding" ? "full" : k));
    return [kinds, (truthy(W) && kinds.includes("sliding")) ? pyInt(W) : null];
  }

  // max context (port of arch_metrics._max_context: the limit read by arch_metrics itself, with its provenance)
  const CTX_KEYS = ["max_position_embeddings", "max_seq_len", "n_positions", "seq_length"];
  const ROPE_EXTENDING = ["yarn", "deepseek_yarn", "dynamic", "linear", "longrope"];
  const CTX_KEYS_ALT = ["model_max_length", "max_sequence_length", "sequence_len", "context_length", "max_seq_length",
    "block_size", "n_ctx"];
  const _ctx_num = v => (isNumber(v) && v > 0 ? Math.trunc(v) : 0);
  const _commas = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  function _max_context(p, d, raw, cl) {
    let lm = _lm_config(raw);
    lm = isDict(lm) ? lm : {};
    let val = null, src = null;
    for (const k of CTX_KEYS) {
      const v = _ctx_num(getOr(lm, k));
      if (v) { val = Math.min(v, 1 << 20); src = `config: ${k}`; break; }
    }
    let rs = null;
    for (const c of [lm, d, raw]) {
      for (const k of ["rope_scaling", "rope_parameters"]) {
        const x = getOr(c, k);
        if (isDict(x) && getOr(x, "factor") !== null) { rs = x; break; }
      }
      if (rs) break;
    }
    let ext = null;
    if (rs && ROPE_EXTENDING.includes(String(por(getOr(rs, "rope_type"), getOr(rs, "type"), "")))) {
      const orig = _ctx_num(getOr(rs, "original_max_position_embeddings")) || _ctx_num(getOr(lm, "original_max_position_embeddings"));
      const fac = getOr(rs, "factor");
      if (orig && isNumber(fac) && fac > 1)
        ext = [Math.min(Math.trunc(orig * fac), 1 << 20),
          `rope_scaling ${por(getOr(rs, "rope_type"), getOr(rs, "type"))}: original_max_position_embeddings ${_commas(orig)} x factor ${String(fac)}`];
    }
    if (val !== null) {
      if (ext && ext[0] > val) src += ` (${ext[1]} = ${_commas(ext[0])} declared)`;
      return [val, src, false];
    }
    if (ext) return [ext[0], ext[1], false];
    let alt = null;
    for (const [c, where] of [[lm, "config"], [raw, "config (top level)"]]) {
      for (const k of CTX_KEYS_ALT) { const v = _ctx_num(getOr(c, k)); if (v) { alt = [Math.min(v, 1 << 20), `${where}: ${k}`]; break; } }
      if (alt) break;
    }
    const full = p.groups.some(g => g.attn !== null && g.window === null && g.kind !== "none");
    if (!full && cl.pos !== "learned_abs")
      return [AMT.CONTEXTS["128k"], "unbounded: a recurrent / window-only stack has no positional limit (sized at 128k)"
        + (alt ? `; trained context ${_commas(alt[0])} (${alt[1]})` : ""), true];
    if (alt) return [alt[0], alt[1], false];
    if (p.stock) {
      const v = _ctx_num(getOr(d, "max_position_embeddings"));
      if (v) return [Math.min(v, 1 << 20), `class default: max_position_embeddings of the stock ${p.mt} config`, false];
    }
    return [4096, "default: no context limit in the config or the code's defaults (4,096 assumed)", false];
  }
  function plan(raw, opts) {
    const [d, stock] = resolve(raw, opts);
    const cl = classify(raw) || {};
    const clget = (k, def) => (hasOwn(cl, k) ? cl[k] : def);
    const dims = dimsOf(raw);
    const mt = lower(por(getOr(d, "model_type"), getOr(raw, "model_type"), ""));
    const feats = cl.feats || new Set();
    const notes = [];
    const hidden = pyInt(_g(d, ["hidden_size", "n_embd", "d_model", "dim"], dims.hidden));
    const vocab = pyInt(_g(d, ["vocab_size", "n_vocab", "padded_vocab_size"], dims.vocab));
    const nh = pyInt(_g(d, ["num_attention_heads", "n_head", "num_heads"], dims.heads));
    let hd = pyInt(por(_g(d, ["head_dim", "attention_head_dim"]), nh ? floordiv(hidden, nh) : 0));
    if (HEAD_DIM_FROM_HIDDEN().has(mt)) hd = nh ? floordiv(hidden, nh) : 0;
    if (mt === "zamba" || mt === "zamba2") hd = pyInt(por(_g(d, ["attention_head_dim"]), hd));
    const nkv = pyInt(por(_g(d, ["num_key_value_heads", "num_kv_heads", "n_head_kv"]), truthy(getOr(d, "multi_query")) ? 1 : nh));
    const n = pyInt(_g(d, ["num_hidden_layers", "n_layer", "num_layers"], dims.layers));
    let norm = clget("norm", "RMSNorm");
    if (mt === "olmo") norm = "none";
    const tied = stock ? truthy(get(d, "tie_word_embeddings", true)) : _tied(d, raw);
    const p = { mt, arch: (por(getOr(raw, "architectures"), [null]))[0], stock, hidden, vocab, tied, notes, max_context: dims.max_context };
    const emb = vocab * hidden;
    let pos_rows = 0;
    if (clget("pos", null) === "learned_abs") {
      pos_rows = pyInt(_g(d, ["max_position_embeddings", "n_positions", "n_ctx"]));
      if (mt === "opt" || mt === "biogpt") pos_rows += 2;
    }
    p.embedding = emb + pos_rows * hidden;
    p.lm_head_W = vocab * hidden;
    p.lm_head = tied ? 0 : vocab * hidden;
    p.final_norm = _norm_params(norm, hidden, norm === "LayerNorm" ? true : null);
    p.extra = 0;
    p.extra_W = 0;
    const qkv_bias = feats.has("bias_qkv");
    const out_bias = feats.has("bias_out") && !S("am.BIAS_OUT_NORM_BETA").has(mt);
    let o_bias = out_bias;
    if (S("O_BIAS_FROM_ATTN_BIAS").has(mt)) o_bias = truthy(getOr(d, "attention_bias"));
    if (["qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl", "qwen2_vl_text", "qwen2_5_vl_text", "seed_oss", "glm", "glm4"].includes(mt))
      o_bias = mt === "seed_oss" ? truthy(get(d, "attention_out_bias", false)) : false;
    const mlp_bias = truthy(getOr(d, "mlp_bias")) || (out_bias && !S("O_BIAS_FROM_ATTN_BIAS").has(mt));
    const qkn = clget("qknorm", "no");
    let qk_norm = 0;
    if (["head", "head_post", "head_distinct"].includes(qkn)) qk_norm = 2 * hd;
    else if (qkn === "proj") qk_norm = nh * hd + nkv * hd;
    else if (qkn === "head_ln") qk_norm = 2 * (nh * hd + nkv * hd);
    else if (qkn === "head_free") qk_norm = 0;
    const og = [...feats].filter(f => f.startsWith("out_gate"));
    const attn = { nh, nkv, dqk: hd, dv: hd, qkv_bias, o_bias, qk_norm, out_gate: og.length > 0,
      gate_headwise: truthy(getOr(d, "headwise_attn_output_gate")) || truthy(getOr(d, "use_head_wise_attn_gate")) ||
        og.some(f => f.startsWith("out_gate_head")) };
    let mla = null;
    if (truthy(getOr(d, "kv_lora_rank"))) {
      const rope = pyInt(_g(d, ["qk_rope_head_dim"]));
      const nope = pyInt(_g(d, ["qk_nope_head_dim"]));
      const vdim = pyInt(por(_g(d, ["v_head_dim"]), nope));
      mla = { q_lora: pyInt(por(_g(d, ["q_lora_rank"]), 0)), kv_lora: pyInt(d.kv_lora_rank), rope, nope, v: vdim };
      Object.assign(attn, { dqk: nope + rope, dv: vdim, nkv: nh });
    }
    const sig = { E: dims.moe_sig[0], k: dims.moe_sig[1], ew: dims.moe_sig[2], shared: dims.moe_sig[3] };
    const ffn_kind = clget("ffn", "glu");
    let dense_inter = pyInt(por(dims.inter, 0));
    if (stock && isIntLike(getOr(d, "intermediate_size")) && !["mamba", "falcon_mamba", "moshi"].includes(mt)) dense_inter = pyInt(d.intermediate_size);
    if (mt === "lfm2" && truthy(getOr(d, "block_auto_adjust_ff_dim"))) {
      dense_inter = pyInt(2 * dense_inter / 3);
      if (getOr(d, "block_ffn_dim_multiplier") !== null) {
        dense_inter = pyInt(numVal(d.block_ffn_dim_multiplier) * dense_inter);
        const m_ = pyInt(por(getOr(d, "block_multiple_of"), 256));
        dense_inter = m_ * floordiv(dense_inter + m_ - 1, m_);
      }
    }
    let glu = ["glu", "moe", "moe_shared"].includes(ffn_kind);
    if (["arcee", "nemotron", "apertus", "starcoder2", "phi", "gpt2", "gpt_bigcode", "gpt_neox", "gptj", "codegen", "bloom", "falcon", "opt",
      "gpt_neo", "mpt", "xglm", "ctrl", "biogpt", "openai-gpt", "persimmon", "gpt_neox_japanese", "stablelm"].includes(mt) && ffn_kind === "vanilla") glu = false;
    if (ffn_kind === "vanilla") glu = false;
    const dense_ffn = { glu, inter: dense_inter, bias: mlp_bias };
    let moe = null;
    if (sig.E) {
      moe = { E: sig.E, k: sig.k, ew: sig.ew, shared: sig.shared,
        shared_gate: feats.has("shared_gate") || (S("SHARED_GATE_TYPES").has(mt) && sig.shared > 0),
        router_bias: ["ernie4_5_moe", "gpt_oss"].includes(mt), expert_bias: mt === "gpt_oss", glu: true,
        out_bias: mt === "jetmoe", dense_compute: S("am.DENSE_EXPERT_CODE").has(mt) && stock };
      if (stock) {
        let ew = getOr(d, "moe_intermediate_size");
        if (!isIntLike(ew) || isBool(ew)) ew = getOr(d, "intermediate_size");
        if (isIntLike(ew) && mt !== "doge") moe.ew = numVal(ew);
      }
    }
    if (moe === null && stock && ["mixtral", "phimoe", "minimax"].includes(mt) && pyEq(getOr(d, "num_local_experts"), 1)) {
      moe = { E: 1, k: 1, ew: pyInt(d.intermediate_size), shared: 0, shared_gate: false, router_bias: false, expert_bias: false,
        glu: true, out_bias: false, dense_compute: false };
    }
    const [kinds, W] = _layer_kinds(d, raw, n);
    const lt_ = getOr(d, "layer_types");
    p.order_exact = new Set(kinds).size <= 1 || (isList(lt_) && lt_.length === n && lt_.every(isStr));
    let moe_mask = moe ? _moe_layer_mask(d, n) : null;
    if (moe && moe_mask === null) moe_mask = Array(n).fill(true);
    const keqv = _k_eq_v(d) || _k_eq_v_all(d);
    const groups = new Map();
    kinds.forEach((k, i) => {
      const is_moe = !!(moe && moe_mask && moe_mask[i]);
      const key = k + "|" + is_moe;
      if (!groups.has(key)) groups.set(key, [k, is_moe, 0, []]);
      groups.get(key)[2] += 1;
      groups.get(key)[3].push(i);
    });
    let G = [];
    for (const [k, is_moe, cnt, at] of groups.values()) {
      const g = { n: cnt, np: cnt, at, kind: k, window: k === "sliding" ? W : null, attn: null, mla: null, rec: null, ffn: null, moe: null,
        norm: 0, k_eq_v: !!(keqv && (k === "full" || _k_eq_v_all(d))) };
      if (k === "full" || k === "sliding" || k === "shared") { g.attn = Object.assign({}, attn); g.mla = mla; }
      if (k === "rec") g.rec = { family: _recurrent_family(d, clget("mixer", "hybrid")) };
      if (is_moe) g.moe = moe; else g.ffn = dense_ffn;
      let nn_ = 2;
      if (feats.has("post_norm")) nn_ = 4;
      if (feats.has("post_only")) nn_ = 2;
      if (clget("block", null) === "parallel") {
        nn_ = 1;
        if (mt === "falcon" && truthy(getOr(d, "new_decoder_architecture")) && pyEq(_g(d, ["num_ln_in_parallel_attn"], 2), 2)) nn_ = 2;
        if (mt === "gpt_neox" || mt === "bitsy_parallel") nn_ = 2;
      }
      g.norm = nn_ * _norm_params(norm, hidden);
      G.push(g);
    }
    p.groups = G;
    p.layers = n;
    if (feats.has("dsa") && truthy(_g(d, ["index_head_dim"])) && truthy(_g(d, ["index_n_heads"]))) {
      const ih = pyInt(_g(d, ["index_n_heads"])), idim = pyInt(_g(d, ["index_head_dim"]));
      const ql = pyInt(por(_g(d, ["q_lora_rank"]), hidden));
      p.dsa = { heads: ih, dim: idim, topk: pyInt(_g(d, ["index_topk"], 2048)) };
      p.dsa_dim = idim;
      const add = a => Object.assign({}, a, { extra_W: (a.extra_W || 0) + ql * ih * idim + hidden * idim + hidden * ih,
        extra_b: (a.extra_b || 0) + 2 * idim });
      const it = getOr(d, "indexer_types");
      if (isList(it) && it.length === n && mt === "glm_moe_dsa") {
        // GLM-5.x indexer_types: only the "full" layers hold the indexer weights and cache its keys
        const layers = [];
        kinds.forEach((k, i) => {
          const is_moe = !!(moe && moe_mask && moe_mask[i]);
          const t = G.find(g => g.kind === k && (g.moe !== null) === is_moe);
          if (t === undefined) throw new PyError("StopIteration", "");
          const g = {};
          for (const [kk, v] of Object.entries(t)) if (kk !== "n" && kk !== "np") g[kk] = v;
          if (g.attn !== null) {
            g.dsa_index = it[i] === "full";
            if (g.dsa_index) g.attn = add(g.attn);
          }
          layers.push(g);
        });
        G = _per_layer_groups(layers);
        p.groups = G;
      } else {
        for (const g of G) if (g.attn !== null) g.attn = add(g.attn);
      }
    }
    _type_fixups(p, d, raw, cl, dims);
    if (mt.startsWith("gemma4")) _gemma4(p, d);
    if (AMT.LONGCAT_ARCHS.includes(p.arch)) _longcat(p, d);
    if (mt === "nemotron-nas" && isList(getOr(d, "block_configs"))) _nemotron_nas(p, d);
    if (mt === "openelm" && isList(getOr(d, "num_query_heads"))) _openelm(p, d);
    const kvl = getOr(d, "num_key_value_heads_per_layer");
    if (isList(kvl) && kvl.length === p.layers && p.groups.length === 1 && p.groups[0].attn) {
      const tmpl = p.groups[0];
      const layers = [];
      for (const x of kvl) {
        const g = {};
        for (const [k, v] of Object.entries(tmpl)) if (k !== "n" && k !== "np") g[k] = v;
        g.attn = Object.assign({}, tmpl.attn, { nkv: pyInt(x) });
        layers.push(g);
      }
      p.groups = _per_layer_groups(layers);
    }
    if (dims.param_layers && dims.eff_layers && dims.param_layers < dims.eff_layers && !stock) {
      // scale only by distinct / planned, and only when the plan holds more layers than there are weight sets
      const L = por(sum(p.groups.map(g => g.n)), p.layers);
      if (L > dims.param_layers) {
        const f = dims.param_layers / L;
        for (const g of p.groups) g.np = g.n ? pyMax(1, pyRound(g.n * f)) : 0;
      }
      p.notes.push(`looped: ${dims.eff_layers} unrolled layers over ${dims.param_layers} distinct ones`);
    }
    _recurrent(p, d, raw, cl);
    _bespoke(p, d, raw, cl);
    [p.max_context, p.max_context_source, p.max_context_unbounded] = _max_context(p, d, raw, cl);
    return p;
  }

  // recurrent mixers
  function req(d, k) {
    if (!hasOwn(d, k)) throw new PyError("KeyError", pyRepr(k));
    return d[k];
  }
  const getd = (d, k, def) => get(d, k, def);
  function _mamba1(d, h, mt) {
    let di, N, K, R, bias, cbias, extra;
    if (mt === "mamba" || mt === "falcon_mamba") {
      di = pyInt(req(d, "intermediate_size")); N = pyInt(req(d, "state_size")); K = pyInt(req(d, "conv_kernel"));
      R = getOr(d, "time_step_rank");
      R = (R === null || R === "auto") ? Math.ceil(h / 16) : pyInt(R);
      bias = truthy(getOr(d, "use_bias")); cbias = truthy(get(d, "use_conv_bias", true));
      extra = 0;
    } else {
      di = pyInt(numVal(getd(d, "mamba_expand", 2)) * h);
      N = pyInt(getd(d, "mamba_d_state", 16)); K = pyInt(getd(d, "mamba_d_conv", 4));
      R = getOr(d, "mamba_dt_rank");
      R = (R === null || R === "auto") ? Math.ceil(h / 16) : pyInt(R);
      bias = truthy(getOr(d, "mamba_proj_bias")); cbias = truthy(get(d, "mamba_conv_bias", true));
      extra = mt === "jamba" ? (R + 2 * N) : 0;
    }
    const W = h * 2 * di + di * (R + 2 * N) + R * di + di * h;
    const b = (bias ? 2 * di + h : 0) + di * K + (cbias ? di : 0) + di + di * N + di + extra;
    const W_raw = mt === "zamba" ? di * (R + 2 * N) + R * di : 0;
    return { family: "mamba1", W, b, conv_state: di * K, ssm_state: di * N, core_flops: 2 * di * K + 2 * di * N, conv_ch: di, conv_k: K,
      W_raw, readout: 2 * di * N };
  }
  function _mamba2(d, h, mt) {
    let H, P, G, I, cbias, pbias, obias, norm;
    if (mt === "zamba2") {
      H = pyInt(req(d, "n_mamba_heads")); P = pyInt(req(d, "mamba_headdim")); G = pyInt(getd(d, "mamba_ngroups", 1));
      I = pyInt(numVal(getd(d, "mamba_expand", 2)) * h);
      cbias = truthy(get(d, "use_conv_bias", true)); pbias = truthy(getOr(d, "add_bias_linear"));
      obias = pbias; norm = true;
    } else if (mt === "mamba2") {
      H = pyInt(req(d, "num_heads")); P = pyInt(req(d, "head_dim")); G = pyInt(getd(d, "n_groups", 1));
      I = pyInt(numVal(getd(d, "expand", 2)) * h);
      cbias = truthy(get(d, "use_conv_bias", true)); pbias = truthy(getOr(d, "use_bias"));
      obias = pbias; norm = true;
    } else {
      H = pyInt(req(d, "mamba_n_heads")); P = pyInt(req(d, "mamba_d_head")); G = pyInt(getd(d, "mamba_n_groups", 1));
      I = pyInt(por(getOr(d, "mamba_d_ssm"), numVal(getd(d, "mamba_expand", 2)) * h));
      cbias = truthy(get(d, "mamba_conv_bias", true)); pbias = truthy(getOr(d, "mamba_proj_bias"));
      obias = mt === "falcon_h1" ? truthy(getOr(d, "projectors_bias")) : pbias;
      norm = mt === "falcon_h1" ? truthy(get(d, "mamba_rms_norm", true)) : true;
    }
    const N = pyInt(mt === "mamba2" ? getd(d, "state_size", 128) : getd(d, "mamba_d_state", 128));
    const K = pyInt(mt === "mamba2" ? getd(d, "conv_kernel", 4) : getd(d, "mamba_d_conv", 4));
    const conv = I + 2 * G * N;
    const proj = 2 * I + 2 * G * N + H;
    const W = h * proj + I * h;
    const b = (pbias ? proj : 0) + conv * K + (cbias ? conv : 0) + 3 * H + (norm ? I : 0) + (obias ? h : 0);
    return { family: "mamba2", W, b, conv_state: conv * K, ssm_state: H * P * N, core_flops: 2 * conv * K + 2 * H * P * N, conv_ch: conv, conv_k: K };
  }
  function _gdn(d, h, mt) {
    const Hk = pyInt(req(d, "linear_num_key_heads")), Hv = pyInt(req(d, "linear_num_value_heads"));
    const dk = pyInt(req(d, "linear_key_head_dim")), dv = pyInt(req(d, "linear_value_head_dim")), K = pyInt(getd(d, "linear_conv_kernel_dim", 4));
    const kd = Hk * dk, vd = Hv * dv;
    const conv = 2 * kd + vd;
    const W = h * (2 * kd + 2 * vd) + h * 2 * Hv + vd * h;
    const b = conv * K + 2 * Hv + dv;
    return { family: "gdn", W, b, conv_state: conv * K, ssm_state: Hv * dk * dv, core_flops: 2 * conv * K + 6 * Hv * dk * dv, conv_ch: conv, conv_k: K };
  }
  function _kda(d, h, mt) {
    const lac = isDict(getOr(d, "linear_attn_config")) ? d.linear_attn_config : {};
    const dk = pyInt(por(getOr(lac, "head_dim"), getOr(d, "linear_head_dim"), 128));
    const H = pyInt(por(getOr(lac, "num_heads"), getOr(d, "linear_num_heads"), 32));
    const K = pyInt(por(getOr(lac, "short_conv_kernel_size"), getOr(d, "linear_conv_kernel_dim"), 4));
    const q = H * dk;
    const conv = 3 * q;
    const W = 3 * h * q + (h * dk + dk * q) + h * H + (h * dk + dk * q) + q * h;
    const b = conv * K + H + q + dk;
    return { family: "kda", W, b, conv_state: conv * K, ssm_state: H * dk * dk, core_flops: 2 * conv * K + 6 * H * dk * dk, conv_ch: conv, conv_k: K };
  }
  function _shortconv(d, h, mt) {
    const K = pyInt(getd(d, "conv_L_cache", 3));
    const bias = truthy(getOr(d, "conv_bias"));
    const W = h * 3 * h + h * h;
    const b = h * K + (bias ? (h + 3 * h + h) : 0);
    return { family: "shortconv", W, b, conv_state: h * K, ssm_state: 0, core_flops: 2 * h * K, conv_ch: h, conv_k: K, W_raw: 0, readout: 0 };
  }
  function _rglru(d, h, mt) {
    const lru = pyInt(req(d, "lru_width")), nh = pyInt(req(d, "num_attention_heads")), K = pyInt(getd(d, "conv1d_width", 4));
    const bw = floordiv(lru, nh);
    const W = 2 * h * lru + lru * h + 2 * nh * bw * bw;
    const b = 2 * lru + h + lru * K + lru + lru + 2 * lru;
    return { family: "rglru", W, b, conv_state: lru * (K - 1), ssm_state: lru, core_flops: 2 * lru * K, conv_ch: lru, conv_k: K,
      W_raw: 2 * nh * bw * bw, readout: 0 };
  }
  function _rwkv4(d, h, mt) {
    const ah = pyInt(por(getOr(d, "attention_hidden_size"), h));
    const W = 3 * h * ah + ah * h;
    const b = 3 * h + 2 * ah;
    return { family: "rwkv4", W, b, conv_state: 0, ssm_state: 2 * h + 3 * ah, core_flops: 0 };
  }
  function _lightning(d, h, mt) {
    const nh = pyInt(req(d, "num_attention_heads"));
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(h, nh)));
    const W = h * 3 * nh * hd + nh * hd * h + h * nh * hd;
    const b = nh * hd;
    return { family: "lightning", W, b, conv_state: 0, ssm_state: nh * hd * hd, core_flops: 6 * nh * hd * hd };
  }
  function _bailing_lightning(d, h, mt) {
    const dd = Object.assign({}, d);
    if (truthy(getOr(d, "num_kv_heads_for_linear_attn"))) dd.num_attention_heads = d.num_kv_heads_for_linear_attn;
    const rec = _lightning(dd, h, mt);
    const nh = pyInt(req(dd, "num_attention_heads"));
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(h, nh)));
    if (truthy(getOr(d, "use_qk_norm"))) rec.b += 2 * hd;
    rec.family = "bailing_lightning";
    return rec;
  }
  function _gla(d, h, mt) {
    const nh = pyInt(por(getOr(d, "num_heads"), getOr(d, "num_attention_heads")));
    const nkv = pyInt(por(getOr(d, "num_kv_heads"), nh));
    const kd = pyRound(h * pyFloat(get(d, "expand_k", 0.5))), vd = pyRound(h * pyFloat(get(d, "expand_v", 1)));
    const grp = floordiv(nh, nkv);
    const kg = floordiv(kd, grp), vg = floordiv(vd, grp);
    const r = pyInt(por(getOr(d, "gate_low_rank_dim"), 16));
    const K = pyInt(por(getOr(d, "conv_size"), 4));
    const W = h * kd + h * kg + h * vg + (truthy(get(d, "use_output_gate", true)) ? h * vd : 0) + vd * h + h * r + r * kg;
    const conv = truthy(getOr(d, "use_short_conv")) ? (kd + kg + vg) : 0;
    const b = kg + (truthy(get(d, "elementwise_affine", true)) ? floordiv(vd, nh) : 0) + conv * K + (truthy(getOr(d, "conv_bias")) ? conv : 0);
    const hk = floordiv(kd, nh), hv = floordiv(vd, nh);
    return { family: "gla", W, b, conv_state: conv * K, ssm_state: nh * hk * hv, core_flops: 2 * conv * K + 4 * nh * hk * hv, conv_ch: conv, conv_k: K };
  }
  const _rwkv7_vdim = (d, h) => (!isList(getOr(d, "value_dim")) ? pyInt(por(getOr(d, "value_dim"), h)) : pyInt(d.value_dim[0]));
  function _deltanet(d, h, mt) {                                       // arch_metrics._deltanet (fla DeltaNet)
    const nh = pyInt(por(getOr(d, "num_heads"), getOr(d, "num_attention_heads")));
    const kd = pyInt(h * pyFloat(get(d, "expand_k", 1))), vd = pyInt(h * pyFloat(get(d, "expand_v", 1)));
    const K = pyInt(por(getOr(d, "conv_size"), 4));
    const W = 2 * h * kd + h * vd + (truthy(get(d, "use_beta", true)) ? h * nh : 0) + (truthy(getOr(d, "use_gate")) ? h * vd : 0) + vd * h;
    const conv = truthy(get(d, "use_short_conv", true)) ? 2 * kd + vd : 0;
    const b = conv * K + (truthy(getOr(d, "conv_bias")) ? conv : 0) + floordiv(vd, nh);
    const hk = floordiv(kd, nh), hv = floordiv(vd, nh);
    return { family: "deltanet", W, b, conv_state: conv * K, ssm_state: nh * hk * hv, core_flops: 2 * conv * K + 6 * nh * hk * hv, conv_ch: conv, conv_k: K };
  }
  function _rwkv7(d, h, mt) {
    const hd = pyInt(por(getOr(d, "head_dim"), 64));
    const nh = floordiv(h, hd);
    const v = _rwkv7_vdim(d, h);
    const Dw = pyInt(por(getOr(d, "decay_low_rank_dim"), 64)), Da = pyInt(por(getOr(d, "a_low_rank_dim"), 64)), Dg = pyInt(por(getOr(d, "gate_low_rank_dim"), 128));
    const W = 2 * h * h + h * v + v * h + (h * Dw + Dw * h) + (h * Da + Da * h) + (h * Dg + Dg * v);
    const b = 6 * h + 2 * h + nh * hd + h + h + 2 * v;
    return { family: "rwkv7", W, b, conv_state: 0, ssm_state: nh * hd * floordiv(v, nh) + h, core_flops: 6 * nh * hd * floordiv(v, nh) };
  }
  function _rwkv7_layers(p, d, g) {
    const h = p.hidden;
    const I = pyInt(por(getOr(d, "intermediate_size"), pyInt(h * pyFloat(por(getOr(d, "hidden_ratio"), 4)))));
    const unit = h * (truthy(get(d, "norm_bias", true)) ? 2 : 1);
    const v = _rwkv7_vdim(d, h);
    const Dv = pyInt(por(getOr(d, "v_low_rank_dim"), 16));
    const n = g.n;
    Object.assign(g, { n: 1, np: 1, at: [0], ffn: { glu: false, inter: I, bias: false }, norm: 2 * unit + h });
    const rest = Object.assign({}, g, { n: n - 1, np: n - 1, at: _range(n).slice(1), rec: Object.assign({}, g.rec, { W: g.rec.W + h * Dv + Dv * v, b: g.rec.b + v }) });
    p.groups = [g].concat(n > 1 ? [rest] : []);
    p.order_exact = true;
    if (truthy(get(d, "norm_first", true))) p.extra += unit;
    p.final_norm = unit;
  }
  function _zamba(p, d, rec) {
    const mt = p.mt, h = p.hidden;
    const n = p.layers;
    const lbt = por(getOr(d, "layers_block_type"), []);
    const n_h = lbt.filter(x => x === "hybrid").length;
    const M = mt === "zamba2" ? pyInt(por(getOr(d, "num_mem_blocks"), 1)) : 1;
    const nh = pyInt(req(d, "num_attention_heads")), nkv = pyInt(por(getOr(d, "num_key_value_heads"), d.num_attention_heads));
    const hd = pyInt(por(getOr(d, "attention_head_dim"), floordiv(2 * h, nh)));
    const inter = pyInt(req(d, "intermediate_size"));
    const a = { nh, nkv, dqk: hd, dv: hd, qkv_bias: false, o_bias: false, qk_norm: 0, out_gate: false,
      in_dim: pyInt(por(getOr(d, "attention_hidden_size"), 2 * h)) };
    const hyb = [];
    lbt.forEach((x, i) => { if (x === "hybrid") hyb.push(i); });
    const rec_g = { n, np: n, at: _range(n), kind: "rec", window: null, attn: null, mla: null, rec: Object.assign({}, rec), ffn: null, moe: null, norm: h, k_eq_v: false };
    const shared = { n: n_h, np: M, at: hyb, kind: "full", window: null, attn: a, mla: null, rec: null, ffn: { glu: true, inter, bias: false },
      moe: null, norm: 2 * h + h, k_eq_v: false };
    let glue_W = h * h;
    if (mt === "zamba2") {
      const r = pyInt(por(getOr(d, "adapter_rank"), 128));
      if (truthy(getOr(d, "use_shared_attention_adapter"))) glue_W += 3 * (a.in_dim * r + r * nh * hd);
      if (truthy(get(d, "use_shared_mlp_adapter", true))) glue_W += h * r + r * 2 * inter;
    }
    const glue = { n: n_h, np: n_h, at: hyb, kind: "none", window: null, attn: null, mla: null,
      rec: { family: "glue", W: glue_W, b: 0, conv_state: 0, ssm_state: 0, core_flops: 0 }, ffn: null, moe: null, norm: 0, k_eq_v: false };
    p.groups = [rec_g, shared, glue];
    p.order_exact = true;
  }
  function _nemotron_h(p, d, raw) {
    const h = p.hidden;
    const pat = por(getOr(raw, "hybrid_override_pattern"), getOr(d, "hybrid_override_pattern"));
    const bc = isList(getOr(d, "block_configs")) ? d.block_configs : null;
    let blocks;
    if (isStr(pat) && pat) {
      const m = { M: "mamba", "*": "attention", "-": "mlp", E: "moe" };
      blocks = [...pat].map(ch => (hasOwn(m, ch) ? m[ch] : "mlp"));
    } else {
      const lbt = por(getOr(d, "layers_block_type"), getOr(d, "layer_types"), []);
      const mp = { linear_attention: "mamba", full_attention: "attention" };
      blocks = lbt.map(x => (hasOwn(mp, pyStr(x)) ? mp[pyStr(x)] : pyStr(x)));
    }
    const per_block = (bc || Array(blocks.length).fill({})).map(b => (isDict(b) ? Object.assign({}, b) : {}));
    const dd = Object.assign({}, d);
    const setdefault = (k, v) => { if (!hasOwn(dd, k)) dd[k] = v; };
    setdefault("mamba_n_heads", getOr(d, "mamba_num_heads"));
    setdefault("mamba_d_head", getOr(d, "mamba_head_dim"));
    setdefault("mamba_n_groups", get(d, "n_groups", get(d, "mamba_n_groups", 1)));
    setdefault("mamba_d_state", get(d, "ssm_state_size", get(d, "mamba_d_state", 128)));
    setdefault("mamba_d_conv", get(d, "conv_kernel", get(d, "mamba_d_conv", 4)));
    dd.mamba_d_ssm = pyInt(dd.mamba_n_heads) * pyInt(dd.mamba_d_head);
    dd.mamba_conv_bias = get(d, "use_conv_bias", true);
    dd.mamba_proj_bias = get(d, "use_bias", false);
    const rec = _mamba2(dd, h, "nemotron_h");
    const nh = pyInt(req(d, "num_attention_heads"));
    const hd = pyInt(por(getOr(d, "head_dim"), getOr(d, "attention_head_dim"), floordiv(h, nh)));
    const nkv = pyInt(por(getOr(d, "num_key_value_heads"), nh));
    const attn = { nh, nkv, dqk: hd, dv: hd, qkv_bias: truthy(getOr(d, "attention_bias")), o_bias: false, qk_norm: 0, out_gate: false };
    const glu = getOr(d, "mlp_use_swiglu") === true;
    const mlp_b = truthy(getOr(d, "mlp_bias"));
    const E = pyInt(por(getOr(d, "n_routed_experts"), getOr(d, "num_experts"), 0));
    let shared_w = pyInt(por(getOr(d, "moe_shared_expert_intermediate_size"), 0));
    if (pyEq(getOr(d, "n_shared_experts"), 0)) shared_w = 0;
    const Gm = new Map();
    blocks.forEach((b, i) => {
      const cfg = i < per_block.length ? per_block[i] : {};
      let key;
      if (b === "mamba") key = ["rec"];
      else if (b === "attention") key = ["attn"];
      else if (b === "moe") key = ["moe", pyInt(por(getOr(cfg, "moe_intermediate_size"), getOr(d, "moe_intermediate_size"))),
        pyInt(por(getOr(cfg, "num_experts_per_tok"), getOr(d, "num_experts_per_tok"), getOr(d, "moe_top_k"), 1)),
        pyInt(por(getOr(cfg, "n_routed_experts"), getOr(cfg, "num_experts"), E))];
      else key = ["mlp", pyInt(por(getOr(cfg, "intermediate_size"), getOr(d, "intermediate_size")))];
      const ks = JSON.stringify(key);
      if (!Gm.has(ks)) Gm.set(ks, [key, 0, []]);
      Gm.get(ks)[1] += 1;
      Gm.get(ks)[2].push(i);
    });
    const groups = [];
    for (const [key, cnt, at] of Gm.values()) {
      const g = { n: cnt, np: cnt, at, kind: "none", window: null, attn: null, mla: null, rec: null, ffn: null, moe: null, norm: h, k_eq_v: false };
      if (key[0] === "rec") Object.assign(g, { kind: "rec", rec: Object.assign({}, rec) });
      else if (key[0] === "attn") Object.assign(g, { kind: "full", attn: Object.assign({}, attn) });
      else if (key[0] === "moe") g.moe = { E: key[3], k: key[2], ew: key[1], shared: shared_w, shared_gate: false, router_bias: false, expert_bias: false,
        glu, shared_glu: glu, latent: pyInt(por(getOr(d, "moe_latent_size"), 0)) || null, out_bias: false };
      else g.ffn = { glu, inter: key[1], bias: mlp_b };
      groups.push(g);
    }
    p.groups = groups;
    p.layers = blocks.length;
    p.order_exact = true;
  }
  const REC_BUILDERS = { mamba1: _mamba1, mamba1_bcdt: _mamba1, mamba1_bcdtw: _mamba1, mamba1_mh: _mamba1, mamba2: _mamba2,
    mamba2_nonorm: _mamba2, mamba2_nbg: _mamba2, gdn: _gdn, gdn_negeig: _gdn, gdn_sig: _gdn, kda: _kda, kda_lb_negeig: _kda, deltanet: _deltanet, shortconv: _shortconv, rglru: _rglru,
    rwkv4: _rwkv4, lightning: _lightning, bailing_lightning: _bailing_lightning, gla: _gla, rwkv7: _rwkv7 };
  function _recurrent(p, d, raw, cl) {
    const mt = p.mt, h = p.hidden;
    const mixer = hasOwn(cl, "mixer") ? cl.mixer : "softmax";
    const fam = mixer !== "softmax" ? _recurrent_family(d, mixer) : null;
    if (fam === null) return;
    if (["nemotron_h", "nemotron_h_puzzle", "ceno"].includes(mt)) { _nemotron_h(p, d, raw); return; }
    const build = hasOwn(REC_BUILDERS, fam) ? REC_BUILDERS[fam] : null;
    let rec = null;
    if (build !== null) {
      try { rec = build(d, h, mt); } catch (e) {
        if (e instanceof PyError && ["KeyError", "TypeError", "ValueError"].includes(e.pyType)) {
          p.notes.push(`recurrent family ${fam}: config keys missing (${e.pyMsg}); recurrent params unknown`);
        } else throw e;
      }
    }
    if (rec === null) {
      rec = { family: fam, W: 0, b: 0, conv_state: 0, ssm_state: 0, core_flops: 0 };
      p.unknown = p.unknown || new Set();
      p.unknown.add("rec");
    }
    if (mt === "zamba" || mt === "zamba2") { _zamba(p, d, rec); return; }
    const pure = cl.mixer === "ssm" || cl.mixer === "linear";
    if (pure) {
      const n = p.layers;
      const norm = h * (cl.norm === "LayerNorm" ? 2 : 1);
      const g = { n, np: n, at: _range(n), kind: "rec", window: null, attn: null, mla: null, rec: Object.assign({}, rec), ffn: null, moe: null, norm, k_eq_v: false };
      p.groups = [g];
      p.order_exact = true;
      if (mt === "rwkv") {
        const I = pyInt(por(getOr(d, "intermediate_size"), 4 * h));
        g.ffn = { glu: false, inter: I, bias: false, rwkv: true };
        g.norm = 2 * 2 * h;
        p.extra += 2 * h;
        p.final_norm = 2 * h;
      } else if (fam === "gla" || fam === "gdn" || fam === "deltanet") {
        // a recurrent mixer in every layer, each with its own SwiGLU MLP and two RMSNorms (fla GLABlock, AllGDN,
        // Qwen3.5-GDN24)
        let I = getOr(d, "intermediate_size");
        if (!isIntLike(I) || isBool(I) || !truthy(I)) {
          const hr = por(getOr(d, "hidden_ratio"), 4);
          I = pyInt(h * numVal(hr) * 2 / 3);
          I = 256 * floordiv(I + 255, 256);
        }
        g.ffn = { glu: true, inter: pyInt(I), bias: false };
        g.norm = 2 * norm;
      } else if (fam === "rwkv7") _rwkv7_layers(p, d, g);
      return;
    }
    for (const g of p.groups) if (g.kind === "rec" || mt === "falcon_h1") g.rec = Object.assign({}, rec);
  }

  function _gemma4(p, d) {
    const h = p.hidden;
    const n = pyInt(req(d, "num_hidden_layers"));
    let lt = por(getOr(d, "layer_types"), _range(n).map(i => (pymod(i + 1, 6) ? "sliding_attention" : "full_attention")));
    lt = [...lt].slice(0, n);
    if (lt.length && lt[lt.length - 1] !== "full_attention") lt[lt.length - 1] = "full_attention";
    const nh = pyInt(req(d, "num_attention_heads"));
    const hd = pyInt(por(getOr(d, "head_dim"), 256));
    const ghd = pyInt(por(getOr(d, "global_head_dim"), 512));
    const keqv = truthy(getOr(d, "attention_k_eq_v"));
    const nkv = pyInt(por(getOr(d, "num_key_value_heads"), nh));
    let gkv = keqv ? getOr(d, "num_global_key_value_heads") : null;
    gkv = truthy(gkv) ? pyInt(gkv) : nkv;
    const nsh = pyInt(por(getOr(d, "num_kv_shared_layers"), 0));
    const inter = pyInt(req(d, "intermediate_size"));
    const wide = truthy(getOr(d, "use_double_wide_mlp"));
    let hp = get(d, "hidden_size_per_layer_input", 256);
    hp = truthy(hp) ? pyInt(hp) : 0;
    const bias = truthy(getOr(d, "attention_bias"));
    const W = pyInt(por(getOr(d, "sliding_window"), 512));
    let moe = null;
    if (truthy(getOr(d, "enable_moe_block"))) {
      const E = pyInt(req(d, "num_experts")), k = pyInt(por(getOr(d, "top_k_experts"), 1)), ew = pyInt(req(d, "moe_intermediate_size"));
      moe = { E, k, ew, shared: inter, shared_gate: false, router_bias: false, expert_bias: false, glu: true, out_bias: false, gemma4: true };
    }
    const plc = isDict(getOr(d, "per_layer_config")) ? d.per_layer_config : {};
    const Gm = new Map();
    lt.forEach((t, i) => {
      const full = t === "full_attention";
      const shared = i >= n - nsh;
      const o = por(getOr(plc, String(i).padStart(2, "0")), getOr(plc, String(i)), {});
      const dh = pyInt(por(getOr(o, "head_dim"), full ? ghd : hd));
      const kv = pyInt(por(getOr(o, "num_key_value_heads"), full ? gkv : nkv));
      const key = [full, shared, dh, kv].join("|");
      if (!Gm.has(key)) Gm.set(key, [full, shared, dh, kv, 0, []]);
      Gm.get(key)[4] += 1;
      Gm.get(key)[5].push(i);
    });
    const groups = [];
    for (const [full, shared, dh, kv, cnt, at] of Gm.values()) {
      const tied = keqv && full;
      const a = { nh, nkv: kv, dqk: dh, dv: dh, qkv_bias: bias, o_bias: bias, out_gate: false, qk_norm: dh + (shared ? 0 : dh) };
      if (shared) a.no_kv_proj = true;
      const g = { n: cnt, np: cnt, at, kind: shared ? "shared" : (full ? "full" : "sliding"), window: full ? null : W, attn: a, mla: null, rec: null,
        ffn: moe ? null : { glu: true, inter: inter * ((wide && shared) ? 2 : 1), bias: false }, moe: moe ? Object.assign({}, moe) : null,
        norm: 4 * h, k_eq_v: tied };
      if (moe) g.norm += 3 * h + h + moe.E;
      if (hp) { g.norm += h; a.extra_W = 2 * h * hp; }
      groups.push(g);
    }
    p.groups = groups;
    p.layers = n;
    p.order_exact = true;
    if (hp) {
      const vple = pyInt(por(getOr(d, "vocab_size_per_layer_input"), p.vocab));
      p.embedding += vple * n * hp;
      p.extra_W += h * n * hp;
      p.extra += hp;
    }
  }
  function _longcat(p, d) {
    const h = p.hidden, V = p.vocab;
    const n = pyInt(por(getOr(d, "num_layers"), getOr(d, "num_hidden_layers")));
    const nh = pyInt(req(d, "num_attention_heads"));
    const rope = pyInt(req(d, "qk_rope_head_dim")), nope = pyInt(req(d, "qk_nope_head_dim"));
    const vd = pyInt(por(getOr(d, "v_head_dim"), nope));
    const mla = { q_lora: pyInt(por(getOr(d, "q_lora_rank"), 0)), kv_lora: pyInt(req(d, "kv_lora_rank")), rope, nope, v: vd };
    const attn = { nh, nkv: nh, dqk: nope + rope, dv: vd, qkv_bias: false, o_bias: false, qk_norm: 0, out_gate: false };
    const fi = pyInt(req(d, "ffn_hidden_size")), E = pyInt(req(d, "n_routed_experts")), Z = pyInt(por(getOr(d, "zero_expert_num"), 0));
    const moe = { E, k: pyInt(por(getOr(d, "moe_topk"), getOr(d, "num_experts_per_tok"), 1)), ew: pyInt(req(d, "expert_ffn_hidden_size")),
      shared: 0, shared_gate: false, router_bias: false, expert_bias: false, glu: true, out_bias: false, zero: Z };
    const dense = { glu: true, inter: fi, bias: false };
    const a0 = Object.assign({}, attn);
    if (truthy(getOr(d, "index_n_heads"))) {
      const ih = pyInt(d.index_n_heads), idim = pyInt(req(d, "index_head_dim"));
      const ql = por(mla.q_lora, h);
      Object.assign(a0, { extra_W: ql * ih * idim + h * idim + h * ih, extra_b: idim });
      p.dsa = { heads: ih, dim: idim, topk: pyInt(por(getOr(d, "index_topk"), 2048)) };
      p.dsa_dim = idim;
    }
    const blank = { window: null, rec: null, k_eq_v: false, at: _range(n) };
    const A = Object.assign({}, blank, { n, np: n, kind: "full", attn: a0, mla, ffn: dense, moe: null, norm: 2 * h });
    const B = Object.assign({}, blank, { n, np: n, kind: "full", attn: Object.assign({}, attn), mla, ffn: null, moe, norm: h });
    const C = Object.assign({}, blank, { n, np: n, kind: "none", attn: null, mla: null, ffn: Object.assign({}, dense), moe: null, norm: h });
    if (truthy(getOr(d, "index_n_heads"))) { A.dsa_index = true; B.dsa_index = false; }
    p.groups = [A, B, C];
    p.layers = n;
    p.order_exact = true;
    p.final_norm = h;
    if (truthy(getOr(d, "oe_split_num"))) {
      const N = pyInt(d.oe_split_num) * (pyInt(por(getOr(d, "oe_neighbor_num"), 2)) - 1);
      const dim = floordiv(h, N);
      const m = pyFloat(por(getOr(d, "oe_vocab_size_ratio"), 0)) * V;
      p.embedding += sum(_range(N).map(i => pyInt(m + 2 * i + 1) * dim));
      p.extra_W += N * dim * h;
      p.longcat_oe = sum(_range(N).map(i => pyInt(m + 2 * i + 1) * dim + dim * h));
    }
    p.notes.push(`LongCat: two MLA attentions, two dense MLPs and one MoE per layer (${n} layers); active ` +
      `params count all ${moe.k} selected experts as real (zero experts, ${Z} of ${E + Z}, are ` +
      "identities, so this is the upper bound)");
  }
  function _dsv4_layers(d, n) {
    const lt = getOr(d, "layer_types");
    let ratios;
    if (isList(lt) && isDict(getOr(d, "compress_rates"))) {
      const cr = d.compress_rates;
      ratios = lt.slice(0, n).map(t => (t === "sliding_attention" ? 0 : pyInt(get(cr, t, 0))));
    } else ratios = [...pyIterList(por(getOr(d, "compress_ratios"), Array(n).fill(0)))].slice(0, n).map(pyInt);
    while (ratios.length < n) ratios.push(0);
    const mlt = getOr(d, "mlp_layer_types");
    let hashl;
    if (isList(mlt)) hashl = mlt.slice(0, n).map(t => t === "hash_moe");
    else hashl = _range(n).map(i => i < pyInt(por(getOr(d, "num_hash_layers"), 0)));
    while (hashl.length < n) hashl.push(false);
    return [ratios, hashl];
  }
  function pyIterList(v) {
    if (isList(v)) return v;
    if (isStr(v)) return [...v];
    if (isDict(v)) return Object.keys(v);
    throw new PyError("TypeError", `'${pyType(v)}' object is not iterable`);
  }
  const numSet = v => new Set(pyIterList(por(v, [])).map(numVal));
  function _deepseek_v4(p, d, raw) {
    const mt = p.mt;
    const h = p.hidden, V = p.vocab;
    const n = p.layers;
    const nh = pyInt(req(d, "num_attention_heads")), hd = pyInt(req(d, "head_dim")), ql = pyInt(req(d, "q_lora_rank"));
    const og = pyInt(req(d, "o_groups")), ol = pyInt(req(d, "o_lora_rank"));
    const hc = pyInt(por(getOr(d, "hc_mult"), 4));
    const mix = (2 + hc) * hc;
    const ih = pyInt(por(getOr(d, "index_n_heads"), 0)), idim = pyInt(por(getOr(d, "index_head_dim"), 0));
    const v41 = mt.startsWith("deepseek_v41");
    const [ratios, hashl] = _dsv4_layers(d, n);
    const kvsrc = numSet(getOr(d, "kv_source_layer_ids")), idxsrc = numSet(getOr(d, "index_source_layer_ids")), eng = numSet(getOr(d, "engram_layer_ids"));
    const memt = isDict(getOr(d, "memt")) ? d.memt : null;
    const mlay = numSet(getOr(memt || {}, "layers"));
    const W0 = h * ql + ql * nh * hd + h * hd + nh * hd * ol + og * ol * h;
    const b0 = ql + hd + nh;
    const comp = (r, dd) => { const cd = r === 4 ? 2 * dd : dd; return [2 * cd * h, r * cd + dd]; };
    const per = (i, g) => {
      if (g.attn === null) return g;
      let W = W0 + 2 * mix * hc * h, b = b0 + 2 * (mix + 3);
      const r = ratios[i];
      let has_idx = false;
      if (!v41) {
        if (r) { const [w, bb] = comp(r, hd); W += w; b += bb; }
        if (r === 4 && ih) { const [w, bb] = comp(4, idim); W += ql * ih * idim + h * ih + w; b += bb; has_idx = true; }
      } else {
        if (kvsrc.has(i)) { W += hd * h + (r > 1 ? hd * h : 0) + idim * hd; b += hd + idim; }
        if (idxsrc.has(i)) { W += ql * ih * idim + h * ih; has_idx = true; }
        if (eng.has(i)) W += 2 * hc * h + (hc + 1) * h * (pyInt(req(d, "engram_max_ngram_size")) - 1) * pyInt(req(d, "engram_n_heads")) * pyInt(req(d, "engram_head_dim"));
      }
      if (memt && mlay.has(i)) {
        const Hm = pyInt(req(memt, "num_memory_heads")), dm = pyInt(req(memt, "head_dim"));
        const qr = pyInt(req(memt, "q_rank")), kr = pyInt(req(memt, "kv_rank")), Gq = pyInt(req(memt, "kv_groups")), gr = pyInt(req(memt, "gate_rank"));
        const nkvm = pyInt(por(getOr(memt, "num_kv_heads"), Hm));
        W += h * (qr + 2 * Gq * kr + 4 * gr) + Hm * dm * qr + 2 * nkvm * dm * kr + 4 * Hm * dm * gr;
        b += 2 * Gq * kr + 2 * Hm * dm + Hm + dm + qr;
      }
      g = Object.assign({}, g, { attn: Object.assign({}, g.attn, { W_override: W, b_override: b }) });
      if (truthy(p.dsa_dim)) g.dsa_index = has_idx;
      if (g.moe !== null) g.moe = Object.assign({}, g.moe, { router_bias: !hashl[i] && !v41 });
      return g;
    };
    _regroup_layers(p, d, raw, per);
    if (!v41) { p.extra_W += hc * hc * h; p.extra += hc + 1; }
    if (v41 && isList(getOr(d, "engram_num_embeddings"))) p.embedding += sum(d.engram_num_embeddings.map(pyInt)) * pyInt(req(d, "engram_head_dim"));
    p.dsv4_hash_layers = hashl.filter(Boolean).length;
  }
  function _regroup_layers(p, d, raw, fn) {
    const n = p.layers;
    const [kinds] = _layer_kinds(d, raw, n);
    const has_moe = p.groups.some(g => g.moe !== null);
    let mask;
    if (has_moe) { const m = _moe_layer_mask(d, n); mask = truthy(m) ? m : Array(n).fill(true); }
    else mask = Array(n).fill(false);
    const layers = [];
    for (let i = 0; i < kinds.length; i++) {
      const k = kinds[i];
      if (i >= mask.length) throw new PyError("IndexError", "list index out of range");
      const t = p.groups.find(g => g.kind === k && (g.moe !== null) === !!mask[i]);
      if (t === undefined) return false;
      const g = {};
      for (const [kk, v] of Object.entries(t)) if (kk !== "n" && kk !== "np") g[kk] = v;
      layers.push(fn(i, g));
    }
    p.groups = _per_layer_groups(layers);
    return true;
  }
  function _remote_fixups(p, d, raw, cl, dims) {
    const mt = p.mt, h = p.hidden, V = p.vocab, n = p.layers;
    const eg = isDict(getOr(d, "engram")) ? d.engram : null;
    if (mt === "allgdn" && truthy(eg)) {
      const O = pyLen(por(getOr(eg, "orders"), [2, 3])), Hh = pyInt(por(getOr(eg, "heads"), 4)), Sl = pyInt(por(getOr(eg, "slots"), 65536)),
        D = pyInt(por(getOr(eg, "dim"), 128));
      const L = pyLen(por(getOr(eg, "layers"), []));
      p.embedding += L * O * Hh * Sl * D;
      p.extra_W += L * 2 * O * Hh * D * h;
      p.extra += L * (2 * h + 4 * h);
    }
    if (AMT.NANOCHAT_TYPES.includes(mt)) {
      let ve;
      if (mt === "gaggle") ve = _range(n).filter(i => pymod(i, 2) === pymod(n - 1, 2));
      else { ve = mt === "sn38-nanochrono" ? getOr(d, "aux_layers") : getOr(d, "value_embedding_layers"); ve = isList(ve) ? ve : []; }
      const ga = p.groups.find(g => g.attn !== null);
      const nh = ga ? ga.attn.nh : 0;
      const G = pyInt(por(getOr(d, "ve_gate_channels"), getOr(d, "aux_gate_dim"), 12));
      const C = pyInt(por(getOr(d, "smear_gate_channels"), getOr(d, "mix_gate_dim"), getOr(d, "smear_channels"), 24));
      p.final_norm = 0;
      for (const g of p.groups) {
        g.norm = 0;
        if (g.attn !== null) g.attn = Object.assign({}, g.attn, { qk_norm: 0 });
        if ((mt === "gaggle" || mt === "sn38-nanochrono") && g.ffn !== null) g.ffn = Object.assign({}, g.ffn, { glu: false });
      }
      p.embedding += ve.length * V * h;
      p.extra += ve.length * nh * G + 2 * n + C + 2;
    }
    const hpl = getOr(d, "num_attention_heads_per_layer");
    if (mt === "laguna" && isList(hpl) && hpl.length === n) {
      _regroup_layers(p, d, raw, (i, g) => {
        if (g.attn !== null) g.attn = Object.assign({}, g.attn, { nh: pyInt(hpl[i]), out_gate: truthy(get(d, "gating", true)), gate_headwise: truthy(get(d, "gating", true)) });
        return g;
      });
    }
    const pb = getOr(d, "pb_hybrid_layers");
    if (isList(pb) && pb.length && truthy(getOr(d, "linear_num_key_heads"))) {
      const rec = _gdn(d, h, mt);
      const pbs = new Set(pb.filter(isIntLike).map(x => pyInt(x)));
      _regroup_layers(p, d, raw, (i, g) => (pbs.has(i) ? Object.assign({}, g, { rec: Object.assign({}, rec) }) : g));
    }
    if (AMT.DSV4_TYPES.includes(mt) && truthy(getOr(d, "o_groups")) && truthy(getOr(d, "q_lora_rank"))) _deepseek_v4(p, d, raw);
    if (mt === "arctic" && truthy(getOr(d, "use_residual"))) {
      for (const g of p.groups) if (g.moe !== null) { g.moe = Object.assign({}, g.moe, { shared: g.moe.shared + h }); g.norm += h; }
    }
    if (mt === "hrm_text") {
      const per = pyInt(por(getOr(d, "num_layers_per_stack"), getOr(d, "num_hidden_layers")));
      const Hc = pyInt(por(getOr(d, "H_cycles"), 1)), Lc = pyInt(por(getOr(d, "L_cycles"), 1));
      p.final_norm = 0;
      for (const g of p.groups) {
        g.norm = 0;
        g.np = floordiv(2 * per * g.n, pyMax(1, sum(p.groups.map(x => x.n))));
        g.n = floordiv(g.np, 2) * Hc * (Lc + 1);
        if (g.attn !== null) g.attn = Object.assign({}, g.attn, { out_gate: true, gate_headwise: false, qk_norm: 0 });
      }
      p.layers = per * Hc * (Lc + 1);
      p.extra += h;
      p.notes.push(`HRM: two stacks of ${per} layers; per token the unrolled depth is ${p.layers} (${Hc} H cycles x (${Lc} L cycles + 1))`);
    }
    if (mt === "iquestloopcoder") {
      for (const g of p.groups) if (g.attn !== null) {
        const a = g.attn;
        g.attn = Object.assign({}, a, { extra_b: (a.extra_b || 0) + a.nh * a.dqk + a.nh });
      }
    }
  }
  // bespoke families (port of graph/arch_metrics.py's bespoke section; same numbers and notes)
  // ---- bespoke builders, group 1 (port of graph/arch_metrics.py's bespoke families, run by _bespoke); called at
  // the very end of plan(), like the Python. moe_params counts no router when m.no_router.
  // (hub_components / hub_blocks have no JS counterpart: hub_reason runs in the Python build only.)
  const is_token_routed = (p, d, raw) => p.mt === "deep" && getOr(d, "mlp_type") === "token_routed";
  function _token_routed(p, d, raw, cl) {
    const h = p.hidden, V = p.vocab;
    const E = pyInt(por(getOr(d, "num_experts"), 4));
    const k = pyInt(por(getOr(d, "num_experts_per_tok"), getOr(d, "top_k"), 2));
    const ew = floordiv(pyInt(getOr(d, "intermediate_size")), E);
    const legacy = truthy(getOr(d, "use_shared_routed_gates"));
    let n_moe = 0;
    for (const g of p.groups) {
      if (g.moe !== null) {
        g.moe = Object.assign({}, g.moe, { E, k, ew, no_router: true,
          shared: truthy(get(d, "shared_expert", true)) ? pyInt(por(getOr(d, "shared_intermediate_size"), 0)) : 0 });
        if (legacy) g.norm += 2;
        n_moe += g.n;
      }
    }
    const rows = legacy ? 1 : k + (truthy(getOr(d, "persist_primary_route")) ? 1 : 0);
    const blocks = { "token-to-expert route tables": n_moe * rows * V };
    if (legacy) blocks["0-d scalars (the Hub API counts none)"] = -2 * n_moe;
    p.hub_blocks = Object.assign({}, p.hub_blocks || {}, blocks);
    p.notes.push(`token-id routed MoE: top-${k} of ${E} experts by a fixed token table, no router`);
  }
  const is_meditor = (p, d, raw) => p.mt === "meditor" && truthy(getOr(d, "num_routed_experts"));
  function _meditor(p, d, raw, cl) {
    const every = pyInt(por(getOr(d, "global_attn_every_n_layers"), 0));
    const W = pyInt(por(getOr(d, "sliding_window_size"), 0));
    _regroup_layers(p, d, raw, (i, g) => {
      if (g.attn !== null) {
        g.attn = Object.assign({}, g.attn, { qkv_bias: true, o_bias: false });
        if (every && W && pymod(i + 1, every) !== 0) { g.kind = "sliding"; g.window = W; }
      }
      if (g.moe !== null) g.norm += 1;
      return g;
    });
  }
  const is_xing_hc = (p, d, raw) => p.mt === "xing4_0" && truthy(getOr(d, "hc_mult"));
  function _xing_hc(p, d, raw, cl) {
    const h = p.hidden;
    const hc = pyInt(d.hc_mult);
    const mix = (2 + hc) * hc;
    const n = sum(p.groups.map(g => g.np));
    p.extra_W += n * 2 * mix * hc * h;
    p.extra += n * 2 * (mix + 3);
    p.notes.push(`hyper-connections: ${hc} residual streams, attn_hc and ffn_hc per layer`);
  }
  // ---- bespoke builders, group 2 (port of graph/arch_metrics.py's bespoke families): GPT-S3 / GPT-X3, KLM, Cyrah ----
  // Python p["stored_blocks"] -> p.stored_blocks (Python-only consumer: hub_components)
  function _gdn2_rec(h, inner, H, D, K, f_rank = null, g_rank = null) {
    const f = !truthy(f_rank) ? h * inner : h * f_rank + f_rank * inner;
    const g = !truthy(g_rank) ? h * inner : h * g_rank + g_rank * inner;
    const W = 3 * h * inner + f + 2 * h * inner + g + inner * h;
    const conv = 3 * inner;
    const b = conv * K + H + inner + inner + D;
    return { family: "gdn2", W, b, conv_state: conv * (K - 1), ssm_state: H * D * D,
      core_flops: 2 * conv * K + 6 * H * D * D, conv_ch: conv, conv_k: K };
  }
  function _drop_unknown_rec(p) {
    if (p.unknown && p.unknown.has("rec")) p.unknown.delete("rec");
  }
  function is_gpt_s3x3(p, d, raw) {
    const mt = getOr(raw, "model_type");
    return (mt === "gpts3_hybrid" || mt === "gptx3") && truthy(getOr(raw, "gdn_head_dim"));
  }
  function _gpt_s3x3(p, d, raw, cl) {
    const mt = getOr(raw, "model_type");
    const h = p.hidden, n = p.layers;
    const s3 = mt === "gpts3_hybrid";
    const per = pyInt(por(s3 ? getOr(d, "gdn_layers_per_attention") : getOr(d, "gdn_layers_per_gqa"), 3));
    const D = pyInt(req(d, "gdn_head_dim"));
    const K = pyInt(por(s3 ? getOr(d, "gdn_conv_size") : getOr(d, "conv_size"), 4));
    const nh = pyInt(req(d, "num_attention_heads")), nkv = pyInt(req(d, "num_key_value_heads"));
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(h, nh)));
    const I = pyInt(req(d, "intermediate_size"));
    const rec = _gdn2_rec(h, h, floordiv(h, D), D, K, s3 ? null : D, s3 ? null : D);
    const attn = { nh, nkv, dqk: hd, dv: hd, qkv_bias: false, o_bias: false, qk_norm: 2 * hd, out_gate: true, gate_headwise: false };
    const norm = 2 * h + (s3 ? 4 * h : 0);
    const layers = [];
    for (let i = 0; i < n; i++) {
      const g = _blank(pymod(i + 1, per + 1) === 0 ? "full" : "rec");
      if (g.kind === "full") g.attn = Object.assign({}, attn);
      else g.rec = Object.assign({}, rec);
      Object.assign(g, { ffn: { glu: true, inter: I, bias: false }, norm });
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
    p.final_norm = h;
    if (s3) p.extra += 2 * h;
    _drop_unknown_rec(p);
  }
  function is_klm(p, d, raw) {
    return getOr(raw, "model_type") === "klm" && truthy(getOr(raw, "gdn_heads")) && truthy(getOr(raw, "attention_every"));
  }
  function _klm(p, d, raw, cl) {
    const h = p.hidden, n = p.layers;
    const every = pyInt(req(d, "attention_every"));
    const H = pyInt(req(d, "gdn_heads")), D = pyInt(req(d, "gdn_head_dim"));
    const nh = pyInt(req(d, "num_attention_heads")), nkv = pyInt(req(d, "num_key_value_heads"));
    const hd = floordiv(h, nh);
    const I = pyInt(req(d, "intermediate_size"));
    const rec = _gdn2_rec(h, H * D, H, D, 4, D, D);
    const attn = { nh, nkv, dqk: hd, dv: hd, qkv_bias: false, o_bias: false, qk_norm: 2 * hd, out_gate: true, gate_headwise: false, extra_b: 2 };
    const layers = [];
    for (let i = 0; i < n; i++) {
      const g = _blank(pymod(i, every) === every - 1 ? "full" : "rec");
      if (g.kind === "full") g.attn = Object.assign({}, attn);
      else g.rec = Object.assign({}, rec);
      Object.assign(g, { ffn: { glu: true, inter: I, bias: false }, norm: 2 * h });
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
    const nb = pyInt(req(d, "n_bigrams")), nt = pyInt(req(d, "n_trigrams")), nd = pyInt(req(d, "ngram_dim"));
    p.embedding += (nb + 1 + nt + 1) * nd;
    p.extra_W += 2 * nd * h;
    p.extra += 2 * (2 * n + 1) * h;
    p.final_norm = h;
    p.tied = true; p.lm_head = 0;
    p.stored_blocks = Object.assign({}, p.stored_blocks || {}, { "n-gram key buffers": nb + nt });
    _drop_unknown_rec(p);
  }
  function is_cyrah(p, d, raw) {
    return getOr(raw, "model_type") === "cyrah" && isList(getOr(raw, "layer_types"));
  }
  function _cyrah_codecs(d) {
    const ch = pyInt(por(getOr(d, "codec_hidden"), 64));
    const cap = pyInt(por(getOr(d, "codec_max_channels"), 192));
    const L = pyInt(por(getOr(d, "codec_layers"), 4));
    const gn = c => 2 * c;
    const conv = (i, o, k) => i * o * k + o;
    function conv_codec(cin, kk, nd, rvq) {
      const k3 = 3 ** nd, k4 = 4 ** nd;
      const res = c => 2 * gn(c) + 2 * conv(c, c, k3);
      let s = conv(cin, ch, kk);
      let cur = ch;
      for (let j = 0; j < L; j++) { const nxt = Math.min(cur * 2, cap); s += conv(cur, nxt, k4) + res(nxt); cur = nxt; }
      s += conv(cur, ch, k3) + gn(ch);
      s += gn(ch) + conv(ch, cur, k3);
      for (let j = 0; j < L; j++) { const nxt = Math.max(ch, floordiv(cur, 2)); s += res(cur) + conv(cur, nxt, k4); cur = nxt; }
      s += conv(cur, cin, kk);
      return s + rvq;
    }
    const audio = conv_codec(1, 7, 1, pyInt(por(getOr(d, "audio_codebooks"), 8)) * pyInt(por(getOr(d, "audio_codebook_size"), 1024)) * ch);
    const image = conv_codec(3, 9, 2, pyInt(por(getOr(d, "image_codebook_size"), 8192)) * ch);
    const signal = conv_codec(pyInt(por(getOr(d, "signal_channels"), 8)), 7, 1, pyInt(por(getOr(d, "signal_codebook_size"), 1024)) * ch);
    const c2 = 2 * ch;
    const C = pyInt(por(getOr(d, "gaussian_channels"), 14));
    const T = Math.max(1, pyInt(por(getOr(d, "gaussian_tokens"), 256)));
    const per = Math.max(1, -floordiv(-Math.max(1, pyInt(por(getOr(d, "gaussian_points"), 262144))), T));
    const lin = (i, o) => i * o + o;
    const cross = 2 * gn(c2) + lin(c2, c2) + lin(c2, 2 * c2) + lin(c2, c2) + lin(c2, 2 * c2) + lin(2 * c2, c2);
    const gauss = lin(C, c2) + lin(c2, c2) + 2 * cross + pyInt(por(getOr(d, "gaussian_codebook_size"), 8192)) * c2
      + lin(c2, c2) + lin(c2, per * C);
    return audio + image + gauss + signal;
  }
  function _cyrah(p, d, raw, cl) {
    const h = p.hidden;
    const lt = raw.layer_types;
    const n = lt.length;
    const nh = pyInt(req(d, "num_attention_heads"));
    const nkv = Math.max(1, pyInt(por(getOr(d, "num_key_value_heads"), 1)));
    const dqk = pyInt(por(getOr(d, "qk_nope_head_dim"), 0)) + pyInt(por(getOr(d, "qk_rope_head_dim"), 0));
    const dv = pyInt(por(getOr(d, "v_head_dim"), dqk));
    const gate = get(d, "use_output_gate", true) !== false;
    const I = pyInt(req(d, "intermediate_size"));
    const sc = _shortconv(d, h, p.mt);
    const attn = { nh, nkv, dqk, dv, qkv_bias: false, o_bias: false, qk_norm: 2 * dqk, out_gate: gate, gate_headwise: false, extra_b: gate ? nh * dv : 0 };
    const W = getOr(d, "sliding_window");
    const layers = [];
    for (const t of lt) {
      let g;
      if (t === "conv") { g = _blank("rec"); g.rec = Object.assign({}, sc); }
      else {
        g = _blank(t === "sliding_attention" ? "sliding" : "full");
        g.attn = Object.assign({}, attn);
        if (t === "sliding_attention" && truthy(W)) g.window = pyInt(W);
      }
      Object.assign(g, { ffn: { glu: true, inter: I, bias: false }, norm: 2 * h });
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
    p.layers = n;
    const M = pyInt(por(getOr(d, "num_modalities"), 95));
    p.embedding += M * h;
    p.extra_W += M * h;
    p.extra += h;
    p.final_norm = h;
    p.tied = true; p.lm_head = 0;
    if (truthy(getOr(d, "use_builtin_codecs")))
      p.stored_blocks = Object.assign({}, p.stored_blocks || {}, { "built-in codecs (audio / image / 3-D / signal)": _cyrah_codecs(d) });
    _drop_unknown_rec(p);
  }
  // ---------------------------------------------------------------- bespoke builders, group 3 (port of bespoke_g3.py)
  // Dory, ZAYA1-VL, Hybrid-Qwen3 GKA, Jet-Nemotron; called at the very end of plan(), as apply(p, d, raw, cl)
  const _blank_group = kind => ({ kind, window: null, attn: null, mla: null, rec: null, ffn: null, moe: null, norm: 0, k_eq_v: false });

  function is_dory(p, d, raw) {
    const lt = getOr(d, "layer_types");
    return p.mt === "dory" && isList(lt) && lt.length === pyInt(por(getOr(d, "num_hidden_layers"), 0))
      && truthy(getOr(d, "n_recurrent_layers"));
  }
  function _dory(p, d, raw, cl) {
    const h = p.hidden, V = p.vocab;
    const lt = d.layer_types;
    const N = lt.length;
    const n_in = pyInt(por(getOr(d, "n_input_layers"), 0)), n_rec = pyInt(por(getOr(d, "n_recurrent_layers"), 0));
    const loops = pyInt(por(getOr(d, "n_recurrent_loops"), 1));
    const I = pyInt(req(d, "intermediate_size"));
    const W = pyInt(por(getOr(d, "sliding_window"), 0)) || null;
    const tmpl = p.groups.find(g => g.attn !== null).attn;
    const attn = Object.assign({}, tmpl, { qkv_bias: false, o_bias: false, qk_norm: 0, out_gate: true, gate_headwise: false,
      extra_b: 2 * tmpl.nh * tmpl.dqk });
    const G = new Map();
    for (let i = 0; i < N; i++) {
      const t = lt[i];
      const m = (n_in <= i && i < n_in + n_rec) ? loops : 1;
      const kind = t === "mlp" ? "none" : ((pyStr(t).includes("sliding") && W) ? "sliding" : "full");
      const key = kind + "|" + m;
      if (!G.has(key)) {
        const g = _blank_group(kind);
        if (kind === "none") {
          g.ffn = { glu: true, inter: I, bias: false };
          g.norm = h + 2 * I;
        } else {
          g.attn = Object.assign({}, attn);
          g.norm = h;
          g.window = kind === "sliding" ? W : null;
        }
        g.n = 0; g.np = 0;
        G.set(key, g);
      }
      G.get(key).np += 1;
      G.get(key).n += m;
    }
    p.groups = [...G.values()];
    p.layers = N + (loops - 1) * n_rec;
    p.final_norm = 0;
    p.extra += V;
    for (let i = p.notes.length - 1; i >= 0; i--) if (p.notes[i].startsWith("looped:")) p.notes.splice(i, 1);
    p.notes.push(`Dory: ${N} single-sublayer layers (attention or MLP), ${n_rec} of them looped ${loops}x; no norms (nGPT)`);
  }

  function is_zaya1_vl(p, d, raw) {
    return p.mt === "zaya1_vl" && truthy(getOr(d, "cca")) && truthy(getOr(d, "num_experts"));
  }
  function _qwen25_vision(v) {
    const hv = pyInt(por(getOr(v, "hidden_size"), 3584));
    const depth = pyInt(por(getOr(v, "depth"), 32));
    const Iv = pyInt(por(getOr(v, "intermediate_size"), 3420));
    const inch = pyInt(por(getOr(v, "in_channels"), 3));
    const ps = pyInt(por(getOr(v, "patch_size"), 14));
    const tp = pyInt(por(getOr(v, "temporal_patch_size"), 2));
    const sm = pyInt(por(getOr(v, "spatial_merge_size"), 2));
    const out = pyInt(por(getOr(v, "out_hidden_size"), 3584));
    const blk = 2 * hv + (3 * hv * hv + 3 * hv) + (hv * hv + hv) + 2 * (Iv * hv + Iv) + (hv * Iv + hv);
    const mh = hv * sm * sm;
    return hv * inch * tp * ps * ps + depth * blk + hv + (mh * mh + mh) + (out * mh + out);
  }
  function _zaya1_vl(p, d, raw, cl) {
    const h = p.hidden;
    const nh = pyInt(req(d, "num_attention_heads")), nkv = pyInt(req(d, "num_key_value_heads")), hd = pyInt(req(d, "head_dim"));
    const E = pyInt(req(d, "num_experts")), k = pyInt(por(getOr(d, "moe_router_topk"), getOr(d, "num_experts_per_tok"), 1));
    const I = floordiv(pyInt(por(getOr(d, "ffn_hidden_size"), 2 * pyInt(por(getOr(d, "moe_intermediate_size"), 2048)))), 2);
    const R = pyInt(por(getOr(d, "zaya_mlp_expansion"), getOr(d, "router_hidden_size"), 256));
    const K0 = pyInt(por(getOr(d, "cca_time0"), 2)), K1 = pyInt(por(getOr(d, "cca_time1"), 2));
    const C = (nh + nkv) * hd;
    const conv_W = C * K0 + C * floordiv(C, nh + nkv) * K1;
    const router_W = h * R + 2 * R * R + R * (E + 1);
    const a0 = { nh, nkv, dqk: hd, dv: hd, qkv_bias: false, o_bias: false, qk_norm: 0, out_gate: false, gate_headwise: false,
      extra_W: conv_W + router_W - h * (E + 1), extra_b: 2 * C + nkv + R + 2 * R + R };
    const moe = { E, k, ew: I, shared: 0, shared_gate: false, router_bias: false, expert_bias: false, glu: true,
      out_bias: false, dense_compute: false, zero: 1 };
    const n = p.layers;
    const layers = [];
    for (let i = 0; i < n; i++) {
      const g = _blank_group("full");
      const a = Object.assign({}, a0);
      if (i > 0) a.extra_b += R;
      g.attn = a;
      g.moe = Object.assign({}, moe);
      g.norm = 2 * h + 2 * h + 4 * h + (i > 0 ? 2 * h : 0);
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
    p.extra += 4 * h;
    p.final_norm = h;
    const ra = pyInt(por(getOr(d, "vision_lora_rank_attn"), 0)), rm = pyInt(por(getOr(d, "vision_lora_rank_mlp"), 0));
    if (!p.stored_blocks) p.stored_blocks = {};
    const blocks = p.stored_blocks;
    if (truthy(getOr(d, "vision_lora")) && (ra || rm)) {
      const la = ra * (h + nh * hd) + ra * (h + nkv * hd) + 2 * ra * (h + floordiv(nkv * hd, 2)) + ra * (nh * hd + h);
      const lm = E * (rm * (h + 2 * I) + rm * (I + h));
      blocks["vision LoRA adapters"] = n * (la + lm);
    }
    if (isDict(getOr(raw, "vision_config"))) blocks["vision tower (Qwen2.5-VL vision encoder)"] = _qwen25_vision(raw.vision_config);
    p.notes.push("ZAYA1: CCA attention (conv-mixed q / k, delayed half-values), an EDA-routed MoE with a skip expert");
  }

  function is_hybrid_gka(p, d, raw) {
    const lt = getOr(d, "layer_types");
    return p.mt === "hybrid_qwen3" && isDict(getOr(d, "gka_config")) && isList(lt) && lt.includes("GKA");
  }
  function _hybrid_gka(p, d, raw, cl) {
    const h = p.hidden;
    const gk = d.gka_config;
    const hd = pyInt(por(getOr(gk, "head_dim"), getOr(d, "head_dim"), 128));
    const Hq = pyInt(por(getOr(gk, "num_q_heads"), req(d, "num_attention_heads")));
    const Hk = pyInt(por(getOr(gk, "num_k_heads"), Hq));
    const Hv = pyInt(por(getOr(gk, "num_v_heads"), Hq));
    const E = pyMax(Hq, Hk, Hv);
    const K = pyInt(por(getOr(gk, "conv_size"), 4));
    const it = pyInt(por(getOr(gk, "num_iter"), 30));
    const qd = Hq * hd, kd = Hk * hd, vd = Hv * hd, ed = E * hd;
    const beta = truthy(get(gk, "use_beta_gate", true)), alpha = truthy(get(gk, "use_alpha_connection", true));
    const vconv = truthy(get(gk, "use_v_conv", true)), gate = truthy(get(gk, "use_gate", true));
    const W = h * (qd + kd + vd) + h * E + (beta ? h * E : 0) + (alpha ? h * E : 0) + (gate ? h * ed : 0) + ed * h;
    const conv = qd + kd + (vconv ? vd : 0);
    const b = (beta ? E : 0) + (alpha ? E : 0) + 2 * E + conv * K + hd;
    const rec = { family: "gka", W, b, conv_state: conv * K, ssm_state: 2 * E * hd * hd,
      core_flops: 2 * conv * K + E * hd * hd * (2 * it + 6), conv_ch: conv, conv_k: K };
    const att = p.groups.find(g => g.attn !== null);
    const layers = [];
    for (const t of d.layer_types) {
      let g = {};
      for (const [kk, v] of Object.entries(att)) if (kk !== "n" && kk !== "np") g[kk] = v;
      if (t === "GKA") g = Object.assign({}, g, { kind: "rec", attn: null, mla: null, rec: Object.assign({}, rec), window: null });
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
    if (p.unknown) p.unknown.delete("rec");
  }

  function is_jet_nemotron(p, d, raw) {
    const lt = getOr(d, "layer_types");
    const ea = getOr(d, "efficient_attention_config");
    return p.mt === "jet_nemotron" && isList(lt) && isDict(ea) && isDict(getOr(ea, "jet"));
  }
  function _jet_nemotron(p, d, raw, cl) {
    const h = p.hidden;
    const ea = d.efficient_attention_config;
    const jc = ea.jet;
    const nh = pyInt(por(getOr(jc, "num_heads"), 6)), hk = pyInt(por(getOr(jc, "head_dim"), 256));
    const ev = pyFloat(por(getOr(jc, "expand_v"), 2.0));
    const K = pyInt(por(getOr(jc, "conv_size"), 4));
    const red = getOr(jc, "dconv_generator_reduction");
    const kd = nh * hk;
    const vd = pyInt(kd * ev);
    const hv = pyInt(hk * ev);
    const Gh = red === null ? vd : floordiv(vd, pyInt(red));
    const W = 2 * h * kd + h * vd + 2 * h * nh + h * Gh + Gh * vd * K + h * vd + vd * h;
    const b = 2 * nh + vd * K + hv;
    const rec = { family: "jet", W, b, conv_state: vd * K, ssm_state: nh * hk * hv,
      core_flops: 2 * vd * K + 6 * nh * hk * hv, conv_ch: vd, conv_k: K };
    const swa = isDict(getOr(ea, "swa")) ? ea.swa : {};
    const Wn = pyInt(por(getOr(swa, "window_size"), 0)) || null;
    const att = p.groups.find(g => g.attn !== null);
    const layers = [];
    for (const t of d.layer_types) {
      let g = {};
      for (const [kk, v] of Object.entries(att)) if (kk !== "n" && kk !== "np") g[kk] = v;
      if (t === "jet") g = Object.assign({}, g, { kind: "rec", attn: null, mla: null, rec: Object.assign({}, rec), window: null });
      else if (t === "swa" && Wn) g = Object.assign({}, g, { kind: "sliding", window: Wn });
      else g = Object.assign({}, g, { kind: "full", window: null });
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
    if (p.unknown) p.unknown.delete("rec");
  }
  // ---- bespoke builders, group 4 (port of graph/arch_metrics.py's bespoke families; same numbers and notes) ----
  // attn_params' plain (non-MLA, non-moa) case returns [W, W - (a.W_inactive || 0), other], as the Python does

  // SliderGPT (CrystaLLM-pi, C-Bone-UCL/CrystaLLM-pi _models/Slider_model.py lines 84-112, 456-457)
  const is_slider_gpt = (p, d, raw) => (por(getOr(raw, "architectures"), [])).includes("SliderGPT")
    && truthy(getOr(d, "slider_on")) && truthy(getOr(d, "slider_n_variables"));
  function _slider_gpt(p, d, raw, cl) {
    const h = p.hidden;
    const nh = pyInt(por(getOr(d, "n_head"), getOr(d, "num_attention_heads")));
    const hd = floordiv(h, nh);
    const V = pyInt(d.slider_n_variables);
    const H = pyInt(por(getOr(d, "slider_n_hidden"), 64));
    const share = pyInt(por(getOr(d, "slider_n_heads_sharing_slider"), 1));
    const kv = 2 * hd * floordiv(nh, share);
    const per = (V * kv + V * kv) + (kv * V * H + V * H) + (H * V * kv + V * kv) + 1;
    p.extra += p.layers * per;
    p.notes.push("SliderGPT: a condition encoder in every block (run once per sequence, not per token)");
  }

  // LFM3-RD (cybertruck32489/LFM3-1.7B-RD modeling_lfm3_rd.py lines 77-225, 349-354, 410-445, 515-538, 619-637,
  // 713-742, 899-913, 1102-1124)
  const is_lfm3_rd = (p, d, raw) => p.mt === "lfm3_rd" && truthy(getOr(d, "num_motifs")) && truthy(getOr(d, "motif_size"))
    && isList(getOr(d, "layer_types"));
  function _lfm3_rd(p, d, raw, cl) {
    const h = p.hidden;
    const lt = d.layer_types;
    const n = lt.length;
    const nh = pyInt(d.num_attention_heads);
    const nkv = pyInt(por(getOr(d, "num_key_value_heads"), nh));
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(h, nh)));
    const q = nh * hd, kvd = nkv * hd;
    const K = pyInt(por(getOr(d, "conv_L_cache"), 3));
    const cb = truthy(getOr(d, "conv_bias"));
    const inter = p.groups.find(g => g.ffn !== null).ffn.inter;
    const W = getOr(d, "sliding_window");
    const pre = pyInt(por(getOr(d, "num_prelude_layers"), 0)), M = pyInt(d.num_motifs), S = pyInt(d.motif_size);
    const ra = pyInt(d.rank_attn), rkv = pyInt(d.rank_kv), rc = pyInt(d.rank_conv);
    const G = pyInt(por(getOr(d, "conv_lora_groups"), 3));
    const nn = truthy(getOr(d, "use_sandwich_norm")) ? 4 * h : 2 * h;
    const ffn = () => ({ glu: true, inter, bias: false });
    const convB = h * K + (cb ? 3 * h + h : 0);
    const conv_rec = () => ({ family: "shortconv", W: 3 * h * h + h * h, b: convB, conv_state: h * K, ssm_state: 0,
      core_flops: 2 * h * K, conv_ch: h, conv_k: K, W_raw: 0, readout: 0 });
    const kind = t => (t !== "full_attention" && t !== "sliding_attention") ? "rec"
      : (t === "sliding_attention" && truthy(W) ? "sliding" : "full");
    const attn = gate => ({ nh, nkv, dqk: hd, dv: hd, qkv_bias: false, o_bias: false, qk_norm: gate ? 2 * hd : 0,
      out_gate: gate, gate_headwise: gate });
    const blank = k => ({ kind: k, window: k === "sliding" ? pyInt(W) : null, attn: null, mla: null, rec: null,
      ffn: null, moe: null, norm: 0, k_eq_v: false });
    const layers = [];
    const idx = _range(pre).concat(_range(n - (pre + M * S)).map(i => pre + M * S + i));
    for (const i of idx) {
      const k = kind(lt[i]);
      const g = blank(k);
      if (k === "rec") g.rec = conv_rec(); else g.attn = attn(true);
      g.ffn = ffn(); g.norm = nn;
      layers.push(g);
    }
    const out = _per_layer_groups(layers);
    for (let pos = 0; pos < S; pos++) {
      const k = kind(lt[pre + pos]);
      const mix = blank(k);
      let lora, own_W, own_b;
      if (k === "rec") {
        mix.rec = Object.assign(conv_rec(), { b: 0 });
        lora = G * (h * rc + rc * h) + (h * rc + rc * h);
        own_W = 0; own_b = convB;
      } else {
        mix.attn = attn(false);
        lora = (q * ra + ra * h) + 2 * (kvd * rkv + rkv * h) + (h * ra + ra * q);
        own_W = h * nh; own_b = 2 * hd;
      }
      Object.assign(mix, { n: M, np: 1 });
      const own = blank("none");
      Object.assign(own, { n: M, np: M, ffn: ffn(), norm: nn,
        rec: { family: "glue", W: own_W, b: own_b + lora, conv_state: 0, ssm_state: 0, core_flops: 0 } });
      out.push(mix, own);
    }
    p.groups = out;
    p.layers = n;
    const R = 128, ri = 64, rr = 64;
    p.extra += (4 * h + 2 * h * R + (4 * R + R) + (R + 1) + 32 + 2 * ri * h + 6 + 2 * M * rr * h + 32 + (h + 1));
    p.notes.push(`LFM3-RD: a ${M} x ${S}-layer core on a shared weight bank with per-layer LoRA; `
      + `${pyInt(por(getOr(d, "recurrent_loops"), M))} recurrent steps by default`);
  }

  // Domyn-Edge (domyn/Domyn-Edge-Base modeling_domynedge.py lines 203, 211-224, 282-285; configuration lines 122-124)
  const is_domynedge = (p, d, raw) => { const hs = getOr(d, "head_sizes"); return p.mt === "domynedge" && isList(hs) && hs.length === p.layers; };
  function _domynedge(p, d, raw, cl) {
    const hs = d.head_sizes;
    const h = p.hidden;
    _regroup_layers(p, d, raw, (i, g) => {
      if (g.attn !== null) { const hd = pyInt(hs[i]); g.attn = Object.assign({}, g.attn, { dqk: hd, dv: hd, qk_norm: hd }); }
      g.norm = 4 * h;
      return g;
    });
  }

  // K2-Horizon MoVA (inception42/Jais-3-MoVA-36B-A4B modeling_k2_horizon.py lines 351-410, 428-443, 542, 646-647)
  const is_k2_horizon_mova = (p, d, raw) => p.mt === "k2_horizon" && pyInt(por(getOr(d, "mova_num_experts"), 0)) > 0;
  function _k2_horizon_mova(p, d, raw, cl) {
    const h = p.hidden;
    const Ev = pyInt(d.mova_num_experts), kv_ = pyInt(por(getOr(d, "mova_num_experts_per_tok"), 1));
    const rb = truthy(getOr(d, "moe_gate_bias"));
    for (const g of p.groups) {
      if (g.moe === null) continue;
      g.moe = Object.assign({}, g.moe, { router_bias: rb });
      const a = g.attn;
      if (a === null) continue;
      const Wv = h * a.nkv * a.dv;
      g.attn = Object.assign({}, a, { extra_W: (a.extra_W || 0) + (Ev - 1) * Wv + h * Ev,
        W_inactive: (a.W_inactive || 0) + (Ev - kv_) * Wv, extra_b: (a.extra_b || 0) + (rb ? Ev : 0) });
    }
    p.notes.push(`MoVA: ${Ev} routed value experts (top ${kv_}) replace v_proj on the sparse layers`);
  }
  // ---- bespoke builders, group 5 (port of graph/arch_metrics.py's bespoke families): Qwen4-Exp, Step3, SDAR self-conditioning
  const is_qwen4_exp = (p, d, raw) => (p.mt === "qwen4_exp_text" || p.mt === "qwen4_exp") && getOr(d, "indexer_n_heads") !== null;
  function _is_prime_q4(v) {
    if (v < 2) return false;
    if (v % 2 === 0) return v === 2;
    for (let k = 3; k * k <= v; k += 2) if (v % k === 0) return false;
    return true;
  }
  function _ngram_rows(base, heads, layer_index, div) {
    let total = 0, prime = base - 1, found = 0;
    const first = layer_index * heads;
    while (found < first + heads) {
      prime += 1;
      while (!_is_prime_q4(prime)) prime += 1;
      if (found >= first) total += prime;
      found += 1;
    }
    return -floordiv(-total, div) * div;
  }
  function _qwen4_exp(p, d, raw, cl) {
    const h = p.hidden, n = p.layers;
    const hc = pyInt(por(getOr(d, "hc_count"), 4)), r = pyInt(por(getOr(d, "hc_lowrank"), 320));
    const H = hc * h;
    const gr_W = H * r + r * H + H * hc;
    const ih = pyInt(por(getOr(d, "indexer_n_heads"), 0)), ikv = pyInt(por(getOr(d, "indexer_kv_heads"), 0)),
      idim = pyInt(por(getOr(d, "indexer_head_dim"), 0));
    for (const g of p.groups) {
      g.norm = 2 * H;
      if (g.attn !== null) {
        const a = g.attn;
        g.attn = Object.assign({}, a, { out_gate: true, gate_headwise: false,
          extra_W: (a.extra_W || 0) + 2 * gr_W + h * (ih + ikv) * idim, extra_b: (a.extra_b || 0) + 2 * idim });
      } else if (g.rec !== null) {
        g.rec = Object.assign({}, g.rec, { W: g.rec.W + 2 * gr_W });
      }
    }
    p.final_norm = 0;
    p.extra += H;
    p.extra_W += 2 * H * r;
    const raw_ids = por(getOr(d, "ple_layer_ids"), []);
    const ids = [...new Set((isList(raw_ids) ? raw_ids : []).filter(isIntLike).map(x => pyInt(x)).filter(x => x >= 1 && x <= n))]
      .sort((a, b) => a - b);
    if (ids.length) {
      const E = pyInt(por(getOr(d, "ple_embed_dim"), h));
      const K = pyInt(por(getOr(d, "ple_conv_kernel_size"), 4));
      const ng = pyInt(por(getOr(d, "ngram_size"), 3));
      const heads = (ng - 1) * pyInt(por(getOr(d, "heads_per_ngram"), 8));
      const base = pyInt(por(getOr(d, "ngram_vocab_size_base"), 20000000));
      const div = pyInt(por(getOr(d, "make_ngram_vocab_size_divisible_by"), 128));
      const L = ids.length;
      p.extra_W += L * (E * H + E * h);
      p.extra += L * (3 * H + H * K);
      let rows;
      if (getOr(d, "num_embeddings_per_layer_input") !== null) rows = L * base * heads;
      else { rows = 0; for (let k = 0; k < L; k++) rows += _ngram_rows(base, heads, k, div); }
      p.embedding += rows * floordiv(E, heads);
      p.stored_blocks = Object.assign({}, p.stored_blocks || {}, { "n-gram index buffers (int64)": L * (ng + 2 * heads) });
    }
    p.notes.push(`Qwen4-Exp: ${hc} hyper-connection streams (low rank ${r}), QSA indexer and PLE n-gram table counted; attention over the indexer's selected tokens not modelled`);
  }
  const is_step3 = (p, d, raw) => ["step3_text", "step3_vl", "step3v"].includes(p.mt) && truthy(getOr(d, "share_q_dim"));
  function _step3_vision(raw, h) {
    const v = getOr(raw, "vision_config");
    if (!isDict(v)) return 0;
    const hv = pyInt(req(v, "hidden_size")), I = pyInt(req(v, "intermediate_size")), Lv = pyInt(req(v, "num_hidden_layers"));
    const P = pyInt(por(getOr(v, "patch_size"), 14)), img = pyInt(por(getOr(v, "image_size"), 728)),
      C = pyInt(por(getOr(v, "num_channels"), 3));
    const out = pyInt(por(getOr(v, "output_hidden_size"), hv));
    const side = floordiv(img, P);
    const emb = hv + C * P * P * hv + hv + (side * side + 1) * hv;
    const lay = 4 * hv + 3 * hv * hv + 3 * hv + hv * hv + hv + hv * I + I + I * hv + hv;
    const proj = hv * out * 4 + out + out * 2 * out * 9 + 2 * out + 2 * out * h + (truthy(getOr(raw, "projector_bias")) ? h : 0);
    return emb + Lv * lay + proj;
  }
  function _step3(p, d, raw, cl) {
    const h = p.hidden;
    const nh = pyInt(req(d, "num_attention_heads"));
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(h, nh)));
    const sq = pyInt(req(d, "share_q_dim"));
    const W = h * sq + sq * nh * hd + 2 * h * hd + nh * hd * h;
    for (const g of p.groups) if (g.attn !== null) g.attn = Object.assign({}, g.attn, { nkv: 1, W_override: W, b_override: sq });
    const vt = _step3_vision(raw, h);
    if (vt) p.stored_blocks = Object.assign({}, p.stored_blocks || {}, { "vision tower (Step3 StepCLIP + projector, from config)": vt });
  }
  const is_sdar_selfcond = (p, d, raw) => p.mt === "sdar" && Object.keys(d).some(k => k.startsWith("self_conditioning"));
  function _sdar_selfcond(p, d, raw, cl) {
    const h = p.hidden;
    const sci = getOr(d, "self_conditioning_intermediate_size");
    const I = pyInt(truthy(sci) ? sci : req(d, "intermediate_size"));
    p.extra_W += 3 * h * I;
    p.extra += h;
  }
  // ---- bespoke builders, group 6 (port of graph/arch_metrics.py's bespoke families; same numbers, same notes)
  const _blank_g = (kind, n) => ({ n, np: n, kind, window: null, attn: null, mla: null, rec: null, ffn: null, moe: null, norm: 0, k_eq_v: false });
  const _dtRank = (R, h) => ((R === null || R === "auto") ? Math.ceil(h / 16) : pyInt(R));

  const is_sambay = (p, d, raw) => p.mt === "phi4flash" || p.arch === "Phi4FlashForCausalLM";
  function _sambay(p, d, raw, cl) {
    const h = p.hidden, n = pyInt(req(d, "num_hidden_layers"));
    const nh = pyInt(req(d, "num_attention_heads"));
    const hd = floordiv(h, nh);
    const nkv = pyInt(por(getOr(d, "num_key_value_heads"), nh));
    const di = pyInt(por(getOr(d, "mamba_expand"), 2)) * h;
    const N = pyInt(por(getOr(d, "mamba_d_state"), 16)), K = pyInt(por(getOr(d, "mamba_d_conv"), 4));
    const R = _dtRank(getOr(d, "mamba_dt_rank"), h);
    let W = getOr(d, "sliding_window");
    W = (isNumber(W) && Number.isInteger(W)) ? pyInt(W) : 2047;
    const I = pyInt(req(d, "intermediate_size"));
    const ln = 2 * h;
    const mamba = { family: "mamba1", W: h * 2 * di + di * (R + 2 * N) + R * di + di * h, b: di * K + di + di + di * N + di,
      conv_state: di * K, ssm_state: di * N, core_flops: 2 * di * K + 2 * di * N, conv_ch: di, conv_k: K };
    const gmu = { family: "gmu", W: h * di + di * h, b: 0, conv_state: 0, ssm_state: 0, core_flops: 0 };
    const att = { nh, nkv, dqk: hd, dv: hd, qkv_bias: true, o_bias: true, qk_norm: 0, out_gate: false, gate_headwise: false, extra_b: 4 * hd + 2 * hd };
    const ffn = { glu: true, inter: I, bias: false };
    const half = floordiv(n, 2);
    const G = new Map();
    for (let i = 0; i < n; i++) {
      let key;
      if (i < half + 2) key = i % 2 === 0 ? "mamba" : (i < half ? "sliding" : "full");
      else key = i % 2 === 0 ? "gmu" : "cross";
      G.set(key, (G.get(key) || 0) + 1);
    }
    const groups = [];
    for (const [key, cnt] of G) {
      const g = _blank_g("rec", cnt);
      g.ffn = Object.assign({}, ffn); g.norm = 2 * ln;
      if (key === "mamba") g.rec = Object.assign({}, mamba);
      else if (key === "gmu") g.rec = Object.assign({}, gmu);
      else if (key === "cross") { g.kind = "shared"; g.attn = Object.assign({}, att, { no_kv_proj: true }); }
      else { g.kind = key; g.attn = Object.assign({}, att); g.window = key === "sliding" ? W : null; }
      groups.push(g);
    }
    p.groups = groups;
    p.layers = n;
    p.final_norm = ln;
    delete p.unknown;
    p.notes.push(`SambaY: layers ${half + 2}+ reuse layer ${half + 1}'s KV (cross attention) and layer ${half}'s SSM output (GMU)`);
  }

  const is_hymba = (p, d, raw) => p.mt === "hymba" || p.arch === "HymbaForCausalLM";
  function _hymba(p, d, raw, cl) {
    const h = p.hidden, n = pyInt(req(d, "num_hidden_layers"));
    const nh = pyInt(req(d, "num_attention_heads"));
    const nkv = pyInt(por(getOr(d, "num_key_value_heads"), nh));
    const hd = floordiv(h, nh);
    const ex = pyInt(por(getOr(d, "mamba_expand"), 2));
    const di = ex * h;
    const N = pyInt(por(getOr(d, "mamba_d_state"), 16)), K = pyInt(por(getOr(d, "mamba_d_conv"), 4));
    const R = _dtRank(getOr(d, "mamba_dt_rank"), h);
    const I = pyInt(req(d, "intermediate_size"));
    let W = getOr(d, "sliding_window");
    W = truthy(W) ? pyInt(W) : null;
    const glob = new Set(por(getOr(d, "global_attn_idx"), []).map(x => pyInt(x)));
    const groupsCfg = por(getOr(d, "kv_reuse_group"), []);
    const cons = new Set();
    for (const grp of groupsCfg) for (const x of grp.slice(1)) cons.add(pyInt(x));
    const prod_win = new Map();
    for (const grp of groupsCfg) for (const x of grp.slice(1)) prod_win.set(pyInt(x), !glob.has(pyInt(grp[0])));
    const rec = { family: "mamba1", W: h * 2 * di + di * (R + 2 * N) + R * di,
      b: di * K + (truthy(get(d, "mamba_conv_bias", true)) ? di : 0) + di + di * N + di + 2 * di
        + (truthy(get(d, "mamba_inner_layernorms", true)) ? (R + 2 * N) : 0),
      conv_state: di * K, ssm_state: di * N, core_flops: 2 * di * K + 2 * di * N, conv_ch: di, conv_k: K };
    const att = { nh, nkv, dqk: hd, dv: floordiv(di, nh), qkv_bias: false, o_bias: false, qk_norm: 0, out_gate: false, gate_headwise: false };
    const G = new Map();
    for (let i = 0; i < n; i++) {
      let key;
      if (cons.has(i)) key = ["shared", W !== null && W !== 0 && (prod_win.has(i) ? prod_win.get(i) : true)];
      else key = [(glob.has(i) || !truthy(W)) ? "full" : "sliding", false];
      const ks = key[0] + "|" + key[1];
      if (!G.has(ks)) G.set(ks, [key, []]);
      G.get(ks)[1].push(i);
    }
    const groups = [];
    for (const [[kind, win], at] of G.values()) {
      const g = _blank_g(kind, at.length);
      g.at = at;
      g.attn = kind === "shared" ? Object.assign({}, att, { no_kv_proj: true }) : Object.assign({}, att);
      g.window = (kind === "sliding" || win) ? W : null;
      g.rec = Object.assign({}, rec);
      g.ffn = { glu: true, inter: I, bias: false };
      g.norm = 2 * h;
      groups.push(g);
    }
    p.groups = groups;
    p.layers = n;
    p.order_exact = true;
    p.final_norm = h;
    const M = pyInt(por(getOr(d, "num_memory_tokens"), 0));
    p.extra += M * h;
    delete p.unknown;
    if (M) p.notes.push(`Hymba: ${M} learned memory tokens are prepended to every prompt (not in the KV counts)`);
  }

  const is_sundial = (p, d, raw) => p.mt === "sundial" || p.arch === "SundialForPrediction";
  function _sundial(p, d, raw, cl) {
    const h = p.hidden;
    const nh = pyInt(req(d, "num_attention_heads"));
    const I = pyInt(req(d, "intermediate_size"));
    const P = pyInt(por(getOr(d, "input_token_len"), 16));
    const ol = por(getOr(d, "output_token_lens"), [720]);
    const C = pyInt(ol[ol.length - 1]);
    const D = pyInt(por(getOr(d, "flow_loss_depth"), 3));
    const w = h;
    for (const g of p.groups) {
      if (g.attn !== null) g.attn = Object.assign({}, g.attn, { nh, nkv: nh, dqk: floordiv(h, nh), dv: floordiv(h, nh), qkv_bias: true, o_bias: false, qk_norm: 0 });
      g.ffn = { glu: true, inter: I, bias: false };
      g.norm = 2 * 2 * h;
    }
    p.final_norm = 2 * h;
    p.vocab = 0; p.embedding = 0; p.tied = false;
    p.extra_W += 2 * P * I + I * h + 2 * P * h;
    p.extra += I + h + h;
    const head_W = 256 * w + w * w + h * w + C * w + D * (2 * w * w + w * 3 * w) + w * C + w * 2 * w;
    const head_b = 2 * w + w + w + D * (2 * w + 2 * w + 3 * w) + C + 2 * w;
    p.lm_head_W = head_W;
    p.lm_head = head_W + head_b;
    p.notes.push("Sundial (time series): patch embedding instead of a token table, the TimeFlow head as the head (one sampling step)");
  }

  const is_ethoric = (p, d, raw) => p.mt === "ethoric" || p.arch === "EthoricForCausalLM";
  function _ethoric(p, d, raw, cl) {
    const h = p.hidden;
    const n = pyInt(truthy(getOr(d, "n_layer")) ? d.n_layer : req(d, "num_hidden_layers"));
    const hd = pyInt(req(d, "head_dim"));
    const gate = pyStr(por(getOr(d, "attn_gate"), ""));
    const can = isDict(getOr(d, "canon")) ? d.canon : {};
    const npos = truthy(getOr(can, "enabled")) ? por(getOr(can, "positions"), []).filter(x => x === "A" || x === "C").length : 0;
    const k = pyInt(por(getOr(can, "kernel"), 4));
    const win = pyInt(por(getOr(d, "attn_window"), 0));
    const wl = new Set(por(getOr(d, "attn_window_layers"), []).map(x => pyInt(x)));
    const I = pyInt(truthy(getOr(d, "ffn_hidden")) ? d.ffn_hidden : req(d, "intermediate_size"));
    const qkn = pyStr(getOr(d, "qk_norm")) === "per_head" ? 2 * hd : 0;
    const per = (i, g) => {
      const a = Object.assign({}, g.attn, { qk_norm: qkn });
      if (gate === "headwise") Object.assign(a, { out_gate: true, gate_headwise: true, extra_b: (a.extra_b || 0) + a.nh });
      else if (gate === "elementwise") Object.assign(a, { out_gate: true, gate_headwise: false, extra_b: (a.extra_b || 0) + a.nh * a.dqk });
      g = Object.assign({}, g, { attn: a, norm: 4 * h, ffn: Object.assign({}, g.ffn, { inter: I }) });
      if (npos) g.rec = { family: "glue", W: 0, b: npos * h * k, conv_state: npos * h * (k - 1), ssm_state: 0, core_flops: npos * 2 * h * k };
      if (win && wl.has(i)) Object.assign(g, { kind: "sliding", window: win });
      return g;
    };
    const tmpl = {};
    for (const [kk, v] of Object.entries(p.groups[0])) if (kk !== "n" && kk !== "np") tmpl[kk] = v;
    p.groups = _per_layer_groups(_range(n).map(i => per(i, Object.assign({}, tmpl))));
  }
  // ---- bespoke builders, group 7 (port of graph/arch_metrics.py's bespoke families)
  function is_arloop(p, d, raw) {
    return (p.mt === "arloop" || p.mt === "arloop_baseline") && isIntLike(getOr(d, "physical_layers"))
      && p.groups.length === 1 && p.groups[0].attn !== null;
  }
  function _arloop(p, d, raw, cl) {
    const h = p.hidden, V = p.vocab;
    const P = pyInt(d.physical_layers);
    const t = p.groups[0];
    const nkv = t.attn.nkv, hd = t.attn.dqk;
    const loop = pyMax(1, floordiv(t.n, pyMax(1, t.np)));
    const G = _range(P).filter(i => pymod(i, 2) === pymod(P - 1, 2)).length;
    const gat = _range(t.n).filter(j => pymod(pymod(j, P), 2) === pymod(P - 1, 2));
    const pla = _range(t.n).filter(j => pymod(pymod(j, P), 2) !== pymod(P - 1, 2));
    const gated = Object.assign({}, t, { n: G * loop, np: G,
      attn: Object.assign({}, t.attn, { extra_W: (t.attn.extra_W || 0) + 32 * nkv }), at: gat });
    const plain = Object.assign({}, t, { n: (P - G) * loop, np: P - G, at: pla });
    p.groups = [gated, plain].filter(g => truthy(g.np));
    p.embedding += V * nkv * hd;
    if (p.mt === "arloop") {
      const N = pyInt(por(getOr(d, "num_hidden_layers"), P * loop));
      p.extra += 2 * N * h + 2 * N * h + 2 * h;
    }
    p.notes.push(`value embedding (V x ${nkv * hd}) gated into V on ${G} of ${P} physical layers`);
  }
  function is_gptx2(p, d, raw) {
    return p.mt === "gptx2" && truthy(getOr(d, "use_symplectic_rope"));
  }
  function _gptx2(p, d, raw, cl) {
    const h = p.hidden, n = p.layers;
    const clock = k => 2 * k + (h * 2 * k + 2 * k) + (2 * k * 2 * k + 2 * k) + h * k + (2 * k * 2 + 2);
    const kg = pyInt(por(getOr(d, "symplectic_global_k_dim"), 32));
    const kl = pyInt(por(getOr(d, "symplectic_local_k_dim"), 16));
    p.extra += clock(kg) + n * clock(kl);
    p.notes.push(`Hamiltonian clocks: one global (k ${kg}) and one per layer (k ${kl}), on the sequence mean`);
  }

  // ---------------------------------------------------------------- C-bucket corrections, group 1 (port of c_g1.py)
  // Inferno, BananaMind-2-MoE, PIT, Ivme, Kohaku (TIPO), MemT-Qwen3.5-MoE, Meditor-2, Nexus; idempotent (they SET values)
  function _set_tied(p, tied) {
    p.tied = truthy(tied);
    p.lm_head = p.tied ? 0 : p.lm_head_W;
  }
  const _archs = raw => por(getOr(raw, "architectures"), []);
  // ---- CFRD (haru, haru_1.1, haru_2): port of arch_metrics._cfrd
  const is_cfrd = (p, d, raw) => p.mt === "cfrd" && isIntLike(getOr(d, "physical_cells")) && !isBool(d.physical_cells);
  function _cfrd(p, d, raw, cl) {
    const h = p.hidden, V = p.vocab;
    const nh = pyInt(req(d, "num_attention_heads")), nkv = pyInt(por(getOr(d, "num_key_value_heads"), d.num_attention_heads));
    const hd = floordiv(h, nh);
    const ff = pyInt(req(d, "intermediate_size"));
    const m = pyInt(req(d, "memory_dim")), Sl = pyInt(req(d, "summary_slots")), mh = pyInt(req(d, "memory_heads"));
    const R = pyInt(por(getOr(d, "inference_recurrences"), req(d, "recurrences")));
    const Rp = pyInt(req(d, "recurrences"));
    const C = pyInt(req(d, "physical_cells"));
    const ctx = pyInt(por(getOr(d, "context_length"), req(d, "max_position_embeddings"))), ch = pyInt(req(d, "chunk_size"));
    const local = pyStr(por(getOr(d, "cell_attention"), "local")) === "local";
    const att = { nh, nkv, dqk: hd, dv: hd, qkv_bias: false, o_bias: false, qk_norm: 0, out_gate: false, gate_headwise: false,
      extra_W: 2 * h, extra_b: 2 };
    const cell_att = Object.assign({}, att);
    if (local) { cell_att.extra_W += 3 * h * m + 2 * m * m + m * h; cell_att.extra_b += Sl * m + m + mh; }
    const ffn = { glu: true, inter: ff, bias: false };
    const base = () => ({ mla: null, rec: null, moe: null, ffn, norm: 2 * h, k_eq_v: false });
    const groups = [Object.assign(base(), { n: R, np: C, at: _range(R), kind: local ? "sliding" : "full", window: local ? ch : null, attn: cell_att })];
    const bind = truthy(getOr(d, "use_binding_block"));
    if (bind) groups.push(Object.assign(base(), { n: 1, np: 1, at: [R], kind: "full", window: null, attn: Object.assign({}, att) }));
    p.groups = groups;
    p.layers = R + (bind ? 1 : 0);
    p.order_exact = true;
    _set_tied(p, true);
    const surf = truthy(getOr(d, "use_surface_features"));
    const F = surf ? pyInt(por(getOr(d, "surface_feature_dim"), 0)) : 0;
    p.extra_W += F * h;
    p.extra += 4 * Rp * h + (local ? Rp : 0) + (surf ? 1 : 0);
    p.notes.push(`CFRD: ${C} folded cells over ${R} depths` + (local ? ", chunk-local attention + summary memory" : ", full attention") +
      (bind ? ", a final binding block" : ""));
  }
  const is_inferno = (p, d, raw) => p.mt === "inferno" && _archs(raw).includes("InfernoForCausalLM");
  function _inferno(p, d, raw, cl) { _set_tied(p, true); }
  const is_bananamind2_moe = (p, d, raw) => p.mt === "bananamind2_moe" && truthy(getOr(d, "expert_intermediate_size"));
  function _bananamind2_moe(p, d, raw, cl) {
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(p.hidden, pyInt(req(d, "num_attention_heads")))));
    const E = pyInt(req(d, "num_experts")), k = pyInt(por(getOr(d, "num_experts_per_token"), 1)), ew = pyInt(req(d, "expert_intermediate_size"));
    for (const g of p.groups) {
      if (g.attn !== null) g.attn = Object.assign({}, g.attn, { qk_norm: 2 * hd });
      if (g.moe !== null) g.moe = Object.assign({}, g.moe, { E, k, ew, shared: 0, shared_gate: false });
    }
    _set_tied(p, get(d, "tie_word_embeddings", true));
  }
  const is_pit = (p, d, raw) => p.mt === "pit" && _archs(raw).includes("PITForCausalLM");
  function _pit(p, d, raw, cl) {
    const h = p.hidden;
    p.final_norm = 0;
    for (const g of p.groups) {
      g.norm = 0;
      if (g.attn !== null) g.attn = Object.assign({}, g.attn, { qk_norm: 0 });
      if (g.ffn !== null) g.ffn = Object.assign({}, g.ffn, { glu: false, inter: 4 * h, bias: false });
    }
    _set_tied(p, true);
  }
  const is_ivme = (p, d, raw) => p.mt === "ivme" && getOr(d, "ffn_mult") !== null;
  function _ivme(p, d, raw, cl) {
    const hd_ = pyInt(por(getOr(d, "hidden_dim"), p.hidden));
    const inter = pyInt(hd_ * Number(d.ffn_mult) * 2 / 3);
    for (const g of p.groups) if (g.ffn !== null) g.ffn = Object.assign({}, g.ffn, { glu: true, inter });
    _set_tied(p, get(d, "tie_word_embeddings", true));
  }
  const is_kohaku_moe = (p, d, raw) => p.mt === "kohaku" && isIntLike(getOr(d, "first_k_dense")) && truthy(getOr(d, "n_routed_experts"))
    && p.groups.length > 0 && p.groups[0].attn !== null;
  function _kohaku_moe(p, d, raw, cl) {
    const n = p.layers;
    const kd = pyInt(d.first_k_dense);
    const E = pyInt(d.n_routed_experts), k = pyInt(por(getOr(d, "num_experts_per_tok"), 1)), ew = pyInt(req(d, "moe_intermediate_size"));
    const tmpl = p.groups.find(g => g.attn !== null);
    const base = {};
    for (const [kk, v] of Object.entries(tmpl)) if (kk !== "n" && kk !== "np") base[kk] = v;
    const moe = { E, k, ew, shared: ew * pyInt(por(getOr(d, "n_shared_experts"), 0)), shared_gate: false, router_bias: false,
      expert_bias: false, glu: true, out_bias: false, dense_compute: false };
    const layers = [];
    for (let i = 0; i < n; i++) {
      const g = Object.assign({}, base);
      if (i < kd) Object.assign(g, { moe: null, ffn: { glu: true, inter: pyInt(req(d, "intermediate_size")), bias: false } });
      else Object.assign(g, { moe: Object.assign({}, moe), ffn: null });
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
  }
  const is_memt_qwen35_moe = (p, d, raw) => p.mt === "memt_qwen3_5_moe" && isDict(getOr(raw, "memt"));
  function _memt_qwen35_moe(p, d, raw, cl) {
    const h = p.hidden;
    const t = por(getOr(por(getOr(raw, "backbone_configs"), {}), "text_config"), d);
    const nh = pyInt(req(t, "num_attention_heads")), hd = pyInt(req(t, "head_dim"));
    const V = nh * hd;
    const gb = truthy(getOr(por(getOr(raw, "memt"), {}), "output_gate_bias"));
    const memW = h * (5 * V + 2 * hd) + 2 * hd * V;
    const memb = (gb ? V : 0) + V + nh + hd;
    for (const g of p.groups) {
      if (g.attn !== null) g.attn = Object.assign({}, g.attn, { qk_norm: 2 * hd, extra_W: memW, extra_b: memb });
      if (g.moe !== null && truthy(g.moe.shared)) g.moe = Object.assign({}, g.moe, { shared_gate: true });
    }
  }
  const is_meditor2 = (p, d, raw) => p.mt === "meditor2" && truthy(getOr(d, "num_macro_domains")) && truthy(getOr(d, "num_sub_experts_per_domain"));
  function _meditor2(p, d, raw, cl) {
    const D = pyInt(d.num_macro_domains), S = pyInt(d.num_sub_experts_per_domain);
    const every = pyInt(por(getOr(d, "global_attn_every_n_layers"), 0)), W = pyInt(por(getOr(d, "sliding_window_size"), 0));
    const moe = { E: D * S, k: 1, ew: pyInt(req(d, "sub_intermediate_size")), shared: pyInt(req(d, "shared_intermediate_size")),
      shared_gate: false, router_bias: false, expert_bias: false, glu: true, out_bias: false, dense_compute: false, zero: D };
    const tmpl = p.groups.find(g => g.attn !== null);
    const base = {};
    for (const [kk, v] of Object.entries(tmpl)) if (kk !== "n" && kk !== "np") base[kk] = v;
    const layers = [];
    for (let i = 0; i < p.layers; i++) {
      const g = Object.assign({}, base, { ffn: null, moe: Object.assign({}, moe), norm: 2 * p.hidden + 1 });
      const glob = !(every && W) || pymod(i + 1, every) === 0;
      if (glob) { g.kind = "full"; g.window = null; } else { g.kind = "sliding"; g.window = W; }
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
    _set_tied(p, get(d, "tie_word_embeddings", true));
  }
  const is_nexus = (p, d, raw) => p.mt === "nexus" && _archs(raw).includes("NexusForCausalLM");
  function _nexus(p, d, raw, cl) {
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(p.hidden, pyInt(req(d, "num_attention_heads")))));
    for (const g of p.groups) if (g.attn !== null) g.attn = Object.assign({}, g.attn, { qk_norm: 2 * hd });
    _set_tied(p, get(raw, "tie_word_embeddings", true));
  }
  // ---------------------------------------------------------------- C-bucket corrections, group 2 (port of c_g2.py)
  // families graph/model_graph.py misreads: each SETS what the code builds (idempotent), at the end of plan()
  function _c2_set_tied(p) { p.tied = true; p.lm_head = 0; }
  function _c2_set_qk_norm(p, width) {
    for (const g of p.groups) if (g.attn !== null) g.attn = Object.assign({}, g.attn, { qk_norm: width });
  }
  function _c2_hd(p) { const ga = p.groups.find(g => g.attn !== null); return ga ? ga.attn.dqk : 0; }

  const is_kanana2_tiny = (p, d, raw) => p.mt === "kanana2_tiny";
  function _kanana2_tiny(p, d, raw, cl) {
    if (getOr(d, "tie_word_embeddings") !== false) _c2_set_tied(p);
    _c2_set_qk_norm(p, 2 * _c2_hd(p));
  }
  const is_phi3small = (p, d, raw) => p.mt === "phi3small" && getOr(raw, "tie_word_embeddings") === null;
  function _phi3small(p, d, raw, cl) {
    const h = p.hidden;
    _c2_set_tied(p);
    p.final_norm = 2 * h;
    for (const g of p.groups) {
      if (g.attn !== null) g.attn = Object.assign({}, g.attn, { qkv_bias: true, o_bias: true });
      if (g.ffn !== null) g.ffn = Object.assign({}, g.ffn, { bias: true });
      g.norm = 2 * 2 * h;
    }
  }
  function is_minicpm_remote_tied(p, d, raw) {
    const am = por(getOr(raw, "auto_map"), {});
    return p.arch === "MiniCPMForCausalLM" && getOr(raw, "tie_word_embeddings") === null
      && pyStr(por(getOr(am, "AutoConfig"), "")).includes("configuration_minicpm");
  }
  function _minicpm_remote_tied(p, d, raw, cl) { _c2_set_tied(p); }
  const is_mima = (p, d, raw) => p.mt === "mima";
  function _mima(p, d, raw, cl) {
    _c2_set_tied(p);
    const I = pyInt(por(getOr(d, "ffn_hidden"), 1024));
    for (const g of p.groups) if (g.ffn !== null) g.ffn = Object.assign({}, g.ffn, { inter: I, glu: true });
  }
  const is_apertus1p5 = (p, d, raw) => (p.mt === "apertus1p5_text" || p.mt === "apertus1p5") && pyStr(getOr(d, "hidden_act")) === "xielu";
  function _apertus1p5(p, d, raw, cl) {
    const h = p.hidden;
    const Vo = pyInt(por(getOr(d, "output_vocab_size"), p.vocab));
    p.lm_head_W = Vo * h;
    p.tied = truthy(get(d, "tie_word_embeddings", false));
    p.lm_head = p.tied ? 0 : Vo * h;
    const hd = _c2_hd(p);
    if (truthy(get(d, "qk_norm", true))) _c2_set_qk_norm(p, 2 * hd);
    for (const g of p.groups) if (g.ffn !== null) { g.ffn = Object.assign({}, g.ffn, { glu: false }); g.norm = 2 * h + 2; }
  }
  const is_gemma3_moe = (p, d, raw) => p.mt === "gemma3_moe";
  function _gemma3_moe(p, d, raw, cl) {
    const h = p.hidden;
    if (truthy(get(d, "tie_word_embeddings", true))) _c2_set_tied(p);
    _c2_set_qk_norm(p, 2 * _c2_hd(p));
    for (const g of p.groups) g.norm = 4 * h;
  }
  const is_bitsy = (p, d, raw) => p.arch === "BitsyForCausalLM";
  function _bitsy(p, d, raw, cl) {
    if (getOr(raw, "tie_word_embeddings") !== false) _c2_set_tied(p);
    _c2_set_qk_norm(p, 2 * _c2_hd(p));
  }
  const is_crowd = (p, d, raw) => p.mt === "crowd" && por(getOr(raw, "architectures"), []).includes("CrowdForCausalLM");
  function _crowd(p, d, raw, cl) {
    for (const g of p.groups) {
      if (g.attn !== null) g.attn = Object.assign({}, g.attn, { qkv_bias: false, o_bias: false });
      if (g.ffn !== null) g.ffn = Object.assign({}, g.ffn, { bias: false });
    }
    if (!p.crowd_engram) { p.crowd_engram = true; p.embedding += pyInt(por(getOr(d, "engram_buckets"), 0)) * p.hidden; }
  }
  const is_nae1 = (p, d, raw) => p.mt === "nae1" && truthy(getOr(d, "n_kv_heads"));
  function _nae1(p, d, raw, cl) {
    const nkv = pyInt(d.n_kv_heads);
    for (const g of p.groups) if (g.attn !== null) g.attn = Object.assign({}, g.attn, { nkv });
  }
  // ---- extension builders (port of graph/arch_metrics.py's bespoke families: _bailing_v3 and the rest of its extension builders; stored_blocks are Python-only: the JS has no hub_reason)
  const _dropUnknownRec = p => { if (p.unknown) { p.unknown.delete("rec"); if (!p.unknown.size) delete p.unknown; } };
  const is_bailing_v3 = (p, d, raw) => p.mt === "bailing_hybrid" && por(getOr(raw, "architectures"), []).includes("BailingMoeV3ForCausalLM")
    && truthy(getOr(d, "layer_group_size"));
  function _bailing_v3(p, d, raw, cl) {
    const h = p.hidden;
    const H = pyInt(req(d, "num_attention_heads"));
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(h, H))), K = pyInt(por(getOr(d, "short_conv_kernel_size"), 4));
    const q = H * hd;
    const lora = !truthy(getOr(d, "no_kda_lora"));
    const fg = lora ? (h * hd + hd * q) : h * q;
    const W = 3 * h * q + 2 * fg + h * H + q * h;
    const b = 3 * q * K + H + q + hd;
    const rec = { family: "kda", W, b, conv_state: 3 * q * K, ssm_state: H * hd * hd, core_flops: 2 * 3 * q * K + 6 * H * hd * hd, conv_ch: 3 * q, conv_k: K };
    const headwise = String(por(getOr(d, "gated_attention_proj_granularity_type"), "")) === "head_wise";
    for (const g of p.groups) {
      if (g.rec !== null) g.rec = Object.assign({}, rec);
      if (g.attn !== null && truthy(g.mla) && !g.attn.mla_gate) {
        const a = g.attn;
        g.attn = Object.assign({}, a, { mla_gate: true, gate_headwise: headwise, extra_W: (a.extra_W || 0) + h * a.nh * (headwise ? 1 : a.dv) });
      }
    }
    _dropUnknownRec(p);
  }
  const is_llada_moe = (p, d, raw) => p.mt === "llada" && truthy(getOr(d, "expert_intermediate_size")) && truthy(getOr(d, "num_experts"));
  function _llada_moe(p, d, raw, cl) {
    const ew = pyInt(d.expert_intermediate_size);
    for (const g of p.groups) {
      if (g.moe !== null) g.moe = Object.assign({}, g.moe, { ew });
      if (g.attn !== null && truthy(getOr(d, "qk_layernorm"))) g.attn = Object.assign({}, g.attn, { qk_norm: 2 * g.attn.dqk });
    }
  }
  const is_fuyu = (p, d, raw) => getOr(raw, "model_type") === "fuyu" && por(getOr(raw, "architectures"), []).includes("FuyuForCausalLM");
  function _fuyu(p, d, raw, cl) {
    for (const g of p.groups) if (g.ffn !== null) g.ffn = Object.assign({}, g.ffn, { glu: false });
  }
  const is_dynin = (p, d, raw) => p.mt === "dynin_omni" && isIntLike(getOr(d, "mlp_hidden_size"));
  function _dynin(p, d, raw, cl) {
    const I = pyInt(d.mlp_hidden_size);
    for (const g of p.groups) if (g.ffn !== null) g.ffn = Object.assign({}, g.ffn, { inter: I });
  }
  const is_mamba2_stock = (p, d, raw) => p.mt === "mamba2" && truthy(p.stock);
  function _mamba2_stock(p, d, raw, cl) {
    const h = p.hidden;
    for (const g of p.groups) g.norm = h;
    p.final_norm = h;
  }
  const is_mobius = (p, d, raw) => (p.mt === "interns2_mobius_text" || p.mt === "interns2_mobius") && isIntLike(getOr(d, "num_blocks"))
    && truthy(getOr(d, "num_experts"));
  function _mobius(p, d, raw, cl) {
    const h = p.hidden, n = p.layers;
    const Bk = pyInt(d.num_blocks);
    let bank = null;
    for (const g of p.groups) {
      const m = g.moe;
      if (m === null) continue;
      bank = bank || Object.assign({}, m, { shared: 0, shared_gate: false });
      g.moe = null;
      g.ffn = { glu: true, inter: pyInt(por(m.shared, getOr(d, "shared_expert_intermediate_size"), 0)), bias: false };
      g.norm = 2 * h;
      if (g.attn !== null) { const a = g.attn; g.attn = Object.assign({}, a, { qk_norm: 2 * a.dqk, extra_W: (a.extra_W || 0) + h }); }
      else if (g.rec !== null) g.rec = Object.assign({}, g.rec, { W: g.rec.W + h });
    }
    if (bank === null) return;
    p.groups.push({ n, np: Bk, kind: "none", window: null, attn: null, mla: null, rec: null, ffn: null, moe: bank, norm: 0, k_eq_v: false });
    p.notes.push(`routed experts shared across layers: ${Bk} banks of ${bank.E} experts, layer i uses bank i mod ${Bk}`);
  }
  const is_inkling = (p, d, raw) => (p.mt === "inkling_mm_model" || p.mt === "inkling") && isList(getOr(d, "local_layer_ids"))
    && truthy(getOr(d, "n_routed_experts"));
  function _inkling(p, d, raw, cl) {
    const h = p.hidden, n = p.layers;
    const K = pyInt(por(getOr(d, "sconv_kernel_size"), getOr(d, "conv_kernel_size"), 4));
    const loc = new Set(d.local_layer_ids.filter(isIntLike).map(x => pyInt(x)));
    const W = pyInt(por(getOr(d, "sliding_window_size"), 0));
    const dr = pyInt(por(getOr(d, "d_rel"), 16));
    const E = pyInt(d.n_routed_experts), k = pyInt(d.num_experts_per_tok), ew = pyInt(d.intermediate_size);
    const S = pyInt(por(getOr(d, "n_shared_experts"), 0));
    const dense_n = pyInt(por(getOr(d, "dense_mlp_idx"), 0)), dense_w = pyInt(por(getOr(d, "dense_intermediate_size"), ew));
    const attn = (local, extent) => {
      const nh = local ? pyInt(por(getOr(d, "swa_num_attention_heads"), d.num_attention_heads)) : pyInt(d.num_attention_heads);
      const nkv = local ? pyInt(por(getOr(d, "swa_num_key_value_heads"), d.num_key_value_heads)) : pyInt(d.num_key_value_heads);
      const hd = local ? pyInt(por(getOr(d, "swa_head_dim"), d.head_dim)) : pyInt(d.head_dim);
      return { nh, nkv, dqk: hd, dv: hd, qkv_bias: false, o_bias: false, qk_norm: 2 * hd, out_gate: false, gate_headwise: false,
        extra_W: h * nh * dr, extra_b: 2 * nkv * hd * K + dr * extent };
    };
    const layer = (i, local, extent, moe) => {
      const g = { kind: local ? "sliding" : "full", window: local ? W : null, attn: attn(local, extent), mla: null, rec: null, ffn: null,
        moe: null, norm: 2 * h + 2 * h * K, k_eq_v: false };
      if (moe) {
        g.moe = { E, k, ew, shared: S * ew, shared_gate: false, router_bias: truthy(get(d, "use_gate_bias", true)), expert_bias: false,
          glu: true, out_bias: false, dense_compute: false };
        g.attn.extra_W += S * h;
        g.norm += 1;
      } else {
        g.ffn = { glu: true, inter: dense_w, bias: false };
        g.norm += 1;
      }
      return g;
    };
    const ext = pyInt(por(getOr(d, "rel_extent"), 1024));
    p.groups = _per_layer_groups(_range(n).map(i => layer(i, loc.has(i), loc.has(i) ? W : ext, i >= dense_n)));
    p.extra += h;
    p.final_norm = h;
    p.tied = false; p.lm_head = p.lm_head_W;
  }
  const BESPOKE = [[is_token_routed, _token_routed], [is_cfrd, _cfrd], [is_meditor, _meditor], [is_xing_hc, _xing_hc], [is_gpt_s3x3, _gpt_s3x3], [is_klm, _klm], [is_cyrah, _cyrah], [is_dory, _dory], [is_zaya1_vl, _zaya1_vl], [is_hybrid_gka, _hybrid_gka], [is_jet_nemotron, _jet_nemotron], [is_slider_gpt, _slider_gpt], [is_lfm3_rd, _lfm3_rd], [is_domynedge, _domynedge], [is_k2_horizon_mova, _k2_horizon_mova], [is_qwen4_exp, _qwen4_exp], [is_step3, _step3], [is_sdar_selfcond, _sdar_selfcond], [is_sambay, _sambay], [is_hymba, _hymba], [is_sundial, _sundial], [is_ethoric, _ethoric], [is_arloop, _arloop], [is_gptx2, _gptx2], [is_inferno, _inferno], [is_bananamind2_moe, _bananamind2_moe], [is_pit, _pit], [is_ivme, _ivme], [is_kohaku_moe, _kohaku_moe], [is_memt_qwen35_moe, _memt_qwen35_moe], [is_meditor2, _meditor2], [is_nexus, _nexus], [is_kanana2_tiny, _kanana2_tiny], [is_phi3small, _phi3small], [is_minicpm_remote_tied, _minicpm_remote_tied], [is_mima, _mima], [is_apertus1p5, _apertus1p5], [is_gemma3_moe, _gemma3_moe], [is_bitsy, _bitsy], [is_crowd, _crowd], [is_nae1, _nae1], [is_bailing_v3, _bailing_v3], [is_llada_moe, _llada_moe], [is_fuyu, _fuyu], [is_dynin, _dynin], [is_mamba2_stock, _mamba2_stock], [is_mobius, _mobius], [is_inkling, _inkling]];
  function _bespoke(p, d, raw, cl) {
    for (const [test, fn] of BESPOKE) if (test(p, d, raw)) { fn(p, d, raw, cl); return; }
  }
  function _per_layer_groups(layers) {
    const out = new Map();
    layers.forEach((g0, i) => {                      // `at`: the layer indices (graph/arch_breakdown.py), not the identity
      const g = {};
      for (const [k, v] of Object.entries(g0)) if (k !== "at") g[k] = v;
      const key = sortedKey(g);
      if (out.has(key)) { out.get(key).n += 1; out.get(key).np += 1; out.get(key).at.push(i); }
      else out.set(key, Object.assign({}, g, { n: 1, np: 1, at: [i] }));
    });
    return [...out.values()];
  }
  const _blank = (kind = "full") => ({ kind, window: null, attn: null, mla: null, rec: null, ffn: null, moe: null, norm: 0, k_eq_v: false });
  function _nemotron_nas(p, d) {
    const h = p.hidden;
    const nh = pyInt(req(d, "num_attention_heads"));
    const hd = pyInt(por(getOr(d, "head_dim"), floordiv(h, nh)));
    const layers = [];
    for (const b of por(getOr(d, "block_configs"), [])) {
      const a = por(getOr(b, "attention"), {}), f = por(getOr(b, "ffn"), {});
      const g = _blank();
      if (truthy(getOr(a, "no_op"))) g.kind = "none";
      else if (truthy(getOr(a, "replace_with_linear"))) {
        g.kind = "none";
        g.rec = { family: "glue", W: h * h, b: 0, conv_state: 0, ssm_state: 0, core_flops: 0 };
        g.norm += h;
      } else {
        g.attn = { nh, nkv: floordiv(nh, pyInt(por(getOr(a, "n_heads_in_group"), 1))), dqk: hd, dv: hd, qkv_bias: truthy(getOr(d, "attention_bias")),
          o_bias: truthy(getOr(d, "attention_bias")), qk_norm: 0, out_gate: false };
        if (truthy(getOr(a, "window_length"))) Object.assign(g, { kind: "sliding", window: pyInt(a.window_length) });
        g.norm += h;
      }
      if (truthy(getOr(f, "no_op"))) { /* no FFN */ }
      else if (truthy(getOr(f, "replace_with_linear"))) { g.ffn = { glu: false, inter: 0, bias: false, linear: true }; g.norm += h; }
      else {
        let inter = pyInt(2 * pyFloat(por(getOr(f, "ffn_mult"), 4)) * h / 3);
        inter = -floordiv(-inter, 256) * 256;
        g.ffn = { glu: true, inter, bias: truthy(getOr(d, "mlp_bias")) };
        g.norm += h;
      }
      layers.push(g);
    }
    if (layers.length) { p.groups = _per_layer_groups(layers); p.layers = layers.length; p.order_exact = true; }
  }
  function _openelm(p, d) {
    const h = p.hidden;
    const hd = pyInt(req(d, "head_dim"));
    const nq = req(d, "num_query_heads"), nkv = req(d, "num_kv_heads"), fm = req(d, "ffn_multipliers");
    const div = pyInt(por(getOr(d, "ffn_dim_divisor"), 256));
    const glu = get(d, "ffn_with_glu", true);
    const layers = [];
    for (let i = 0; i < pyInt(req(d, "num_transformer_layers")); i++) {
      const g = _blank();
      g.attn = { nh: pyInt(nq[i]), nkv: pyInt(nkv[i]), dqk: hd, dv: hd, qkv_bias: false, o_bias: false,
        qk_norm: truthy(getOr(d, "normalize_qk_projections")) ? 2 * hd : 0, out_gate: false };
      g.ffn = { glu: truthy(glu), inter: _make_divisible(pyFloat(fm[i]) * h, div), bias: false };
      g.norm = 2 * h;
      layers.push(g);
    }
    p.groups = _per_layer_groups(layers);
    p.layers = layers.length;
    p.order_exact = true;
  }
  const _layout_value = (v, d) => (isStr(v) ? truthy(getOr(d, v)) : v);
  function _type_fixups(p, d, raw, cl, dims) {
    const mt = p.mt, h = p.hidden;
    const lay = hasOwn(AMT.LAYOUT, mt) ? AMT.LAYOUT[mt] : null;
    const norm = hasOwn(cl, "norm") ? cl.norm : "RMSNorm";
    let unit = 0;
    if (lay) {
      const nb = _layout_value(hasOwn(lay, "nb") ? lay.nb : norm === "LayerNorm", d);
      unit = h * (nb ? 2 : 1);
      if (lay.final_norm === false) p.final_norm = 0; else p.final_norm = unit;
      if (_layout_value(hasOwn(lay, "head_bias") ? lay.head_bias : false, d)) p.extra += p.vocab;
      if (truthy(lay.emb_norm)) p.extra += 2 * h;
      if (mt === "opt") {
        const wp = pyInt(por(getOr(d, "word_embed_proj_dim"), h));
        if (wp !== h) {
          p.embedding = p.vocab * wp + (p.embedding - p.vocab * h);
          p.lm_head_W = p.vocab * wp;
          p.lm_head = p.tied ? 0 : p.vocab * wp;
          p.extra_W += 2 * wp * h;
        }
        if (!truthy(get(d, "do_layer_norm_before", true)) || truthy(getOr(d, "_remove_final_layer_norm"))) p.final_norm = 0;
      }
      if (truthy(lay.last_dense_bias)) p.extra += h;
    }
    if (mt === "moshi") {
      const nq = pyInt(por(getOr(d, "num_codebooks"), 8));
      const av = pyInt(por(getOr(d, "audio_vocab_size"), 2048));
      p.embedding = (p.vocab + 1) * h + 2 * nq * (av + 1) * h;
      p.notes.push("depth decoder (per-step codebook generation) not counted");
    }
    for (const g of p.groups) {
      const a = g.attn;
      let f = g.ffn;
      if (mt === "moshi" && f !== null) { g.ffn = f = Object.assign({}, f, { inter: floordiv(pyInt(get(d, "ffn_dim", 4 * h)), 2) }); }
      if (mt === "recurrent_gemma" && f !== null) { g.ffn = f = Object.assign({}, f, { inter: floordiv(pyInt(req(d, "intermediate_size")), 2) }); }
      if (lay && a !== null) {
        a.qkv_bias = _layout_value(hasOwn(lay, "qkv") ? lay.qkv : a.qkv_bias, d);
        a.o_bias = _layout_value(hasOwn(lay, "o") ? lay.o : a.o_bias, d);
      }
      if (lay && f !== null) f.bias = _layout_value(hasOwn(lay, "mlp") ? lay.mlp : f.bias, d);
      if (lay) {
        let nl = hasOwn(lay, "norms") ? lay.norms : 2;
        if (mt === "stablelm") nl = truthy(getOr(d, "use_parallel_residual")) ? 1 : 2;
        if (mt === "falcon") {
          if (!truthy(get(d, "parallel_attn", true))) nl = 2;
          else nl = pyEq(por(getOr(d, "num_ln_in_parallel_attn"), truthy(getOr(d, "new_decoder_architecture")) ? 2 : 1), 2) ? 2 : 1;
        }
        g.norm = nl * unit;
      }
      if (a === null) continue;
      const hd = a.dqk;
      if (mt === "falcon") {
        if (truthy(getOr(d, "new_decoder_architecture"))) { p.falcon_new_arch = true; a.nkv = pyInt(por(getOr(d, "num_kv_heads"), a.nh)); }
        else if (truthy(getOr(d, "multi_query"))) a.nkv = 1;
        else a.nkv = a.nh;
        if (f !== null) f.inter = pyInt(por(getOr(d, "ffn_hidden_size"), 4 * h));
      }
      if (mt === "mpt") { a.nkv = a.nh; if (f !== null) f.inter = 4 * h; }
      if (mt === "ctrl" && f !== null) f.inter = pyInt(por(getOr(d, "dff"), 4 * h));
      if (mt === "gpt_neox_japanese" && f !== null) f.inter = pyInt(h * numVal(get(d, "intermediate_multiple_size", 4)));
      if (mt === "persimmon" && truthy(get(d, "qk_layernorm", true))) a.qk_norm = 4 * hd;
      if (mt === "stablelm") a.qk_norm = truthy(getOr(d, "qk_layernorm")) ? (a.nh + a.nkv) * hd : 0;
      if (mt === "cohere") a.qk_norm = truthy(getOr(d, "use_qk_norm")) ? (a.nh + a.nkv) * hd : 0;
      if (mt === "cohere2") a.qk_norm = 0;
      if (mt === "apertus") g.norm += 2;
      if (mt === "gpt_oss") a.sinks = a.nh;
      if (mt === "doge") { a.extra_W = a.nkv * hd * a.nkv; a.extra_b = a.nkv; g.norm += 2 * h; }
      if ((mt === "llama4" || mt === "llama4_text") && g.moe !== null) g.moe = Object.assign({}, g.moe, { shared: pyInt(getOr(d, "intermediate_size")) });
      if ((mt === "llama4" || mt === "llama4_text") && f !== null) g.ffn = Object.assign({}, f, { inter: pyInt(por(getOr(d, "intermediate_size_mlp"), f.inter)) });
      if (mt === "jetmoe") {
        const E = pyInt(req(d, "num_local_experts")), k = pyInt(req(d, "num_experts_per_tok"));
        const D = a.nkv * pyInt(por(getOr(d, "kv_channels"), hd));
        Object.assign(a, { moa: { E, k, D }, dqk: pyInt(por(getOr(d, "kv_channels"), hd)), dv: pyInt(por(getOr(d, "kv_channels"), hd)), nh: k * a.nkv });
      }
    }
    _remote_fixups(p, d, raw, cl, dims);
  }

  // ------------------------------------------------------------------------------------------ evaluation
  function attn_params(a, mla, hidden, k_eq_v = false) {
    if (a.W_override !== undefined && a.W_override !== null) return [a.W_override, a.W_override, hasOwn(a, "b_override") ? a.b_override : 0];
    const nh = a.nh, nkv = a.nkv, dqk = a.dqk, dv = a.dv;
    const extra_W = a.extra_W || 0;
    let other = (a.extra_b || 0) + (a.sinks || 0);
    if (mla) {
      let W = 0;
      if (mla.q_lora) { W += hidden * mla.q_lora + mla.q_lora * nh * dqk; other += mla.q_lora; }
      else W += hidden * nh * dqk;
      W += hidden * (mla.kv_lora + mla.rope) + mla.kv_lora * nh * (mla.nope + dv);
      other += mla.kv_lora;
      W += nh * dv * hidden + extra_W;
      other += a.o_bias ? hidden : 0;
      return [W, W, other];
    }
    const moa = a.moa;
    if (moa) {
      const E = moa.E, k = moa.k, D = moa.D;
      const W = E * 2 * hidden * D + 2 * hidden * D + hidden * E;
      const Wa = k * 2 * hidden * D + 2 * hidden * D + hidden * E;
      return [W, Wa, other + hidden];
    }
    const xin = a.in_dim || hidden;
    let Wq = xin * nh * dqk, Wk = xin * nkv * dqk, Wv = k_eq_v ? 0 : xin * nkv * dv;
    if (a.no_kv_proj) Wk = Wv = 0;
    const Wo = nh * dv * hidden;
    const Wg = a.out_gate ? (a.gate_headwise ? hidden * nh : hidden * nh * dv) : 0;
    const W = Wq + Wk + Wv + Wo + Wg + extra_W;
    other += a.qk_norm;
    if (a.qkv_bias) other += nh * dqk + (a.no_kv_proj ? 0 : nkv * dqk + (k_eq_v ? 0 : nkv * dv));
    if (a.o_bias) other += hidden;
    return [W, W - (a.W_inactive || 0), other];
  }
  function ffn_params(f, hidden) {
    if (f === null) return [0, 0];
    if (f.linear) return [hidden * hidden, 0];
    if (f.rwkv) return [hidden * f.inter + hidden * hidden + f.inter * hidden, 2 * hidden];
    const m = f.glu ? 3 : 2;
    const W = m * hidden * f.inter;
    let other = f.bias ? ((f.glu ? 2 : 1) * f.inter + hidden) : 0;
    if (f.glu && f.bias) other = 2 * f.inter + hidden;
    return [W, other];
  }
  function moe_params(m, hidden) {
    if (m === null) return [0, 0, 0, 0];
    const xin = m.latent || hidden;
    const per = ((hasOwn(m, "glu") ? m.glu : true) ? 3 : 2) * xin * m.ew;
    const routed = m.E * per;
    const sglu = hasOwn(m, "shared_glu") ? m.shared_glu : (hasOwn(m, "glu") ? m.glu : true);
    let shared = (sglu ? 3 : 2) * hidden * m.shared;
    if (m.latent) shared += 2 * hidden * m.latent;
    const router = m.no_router ? 0 : hidden * (m.E + (m.zero || 0));
    const gate = m.shared_gate ? hidden : 0;
    let other = m.router_bias ? m.E : 0;
    if (m.expert_bias) other += m.E * (2 * m.ew + hidden);
    if (m.out_bias) other += hidden;
    const total = routed + shared + router + gate;
    const active = m.k * per + shared + router + gate;
    return [total, active, other, routed];
  }
  function layer_params(g, hidden) {
    const out = { attn_W: 0, attn_Wa: 0, attn_b: 0, ffn_W: 0, ffn_Wa: 0, ffn_b: 0, routed: 0, rec_W: 0, rec_b: 0, norm: g.norm };
    if (g.attn !== null) [out.attn_W, out.attn_Wa, out.attn_b] = attn_params(g.attn, g.mla, hidden, g.k_eq_v);
    if (g.ffn !== null) { const [W, b] = ffn_params(g.ffn, hidden); Object.assign(out, { ffn_W: W, ffn_Wa: W, ffn_b: b }); }
    if (g.moe !== null) { const [t, a, b, r] = moe_params(g.moe, hidden); Object.assign(out, { ffn_W: t, ffn_Wa: a, ffn_b: b, routed: r }); }
    if (g.rec !== null) { out.rec_W = g.rec.W || 0; out.rec_b = g.rec.b || 0; }
    return out;
  }
  function params(p) {
    const h = p.hidden;
    let tot = p.embedding + p.lm_head + p.final_norm + p.extra + p.extra_W;
    let routed = 0, inactive = 0;
    for (const g of p.groups) {
      const lp = layer_params(g, h);
      const per = lp.attn_W + lp.attn_b + lp.ffn_W + lp.ffn_b + lp.rec_W + lp.rec_b + lp.norm;
      tot += g.np * per;
      routed += g.np * lp.routed;
      inactive += g.np * (lp.ffn_W - lp.ffn_Wa + lp.attn_W - lp.attn_Wa);
    }
    const non_emb = tot - p.embedding - p.lm_head;
    return { total: pyInt(tot), embedding: pyInt(p.embedding), lm_head: pyInt(p.lm_head), non_embedding: pyInt(non_emb),
      active: pyInt(tot - inactive), active_non_embedding: pyInt(non_emb - inactive), routed_experts: pyInt(routed), tied_embeddings: !!p.tied };
  }
  function matmul_weights(p) {
    const h = p.hidden;
    let s = 0;
    for (const g of p.groups) { const lp = layer_params(g, h); s += g.n * (lp.attn_Wa + lp.ffn_Wa + lp.rec_W); }
    return s;
  }
  function _span_causal_sum(L, W) {                                       // BigInt
    const Lb = big(L);
    if (W === null || W === undefined || L <= W) return Lb * (Lb + 1n) / 2n;
    const Wb = big(W);
    return Wb * (Wb + 1n) / 2n + (Lb - Wb) * Wb;
  }
  // attention FLOPs: Python accumulates exact ints (decode) or floats from exact int / int divisions (prefill)
  function attn_flops(p, L, mode) {
    let tot = 0;
    for (const g of p.groups) {
      const a = g.attn;
      if (a === null) continue;
      const per_key = 2 * a.nh * (a.dqk + a.dv);
      const W = g.window;
      const dsa = g.kind !== "shared" ? p.dsa : null;
      if (dsa && mode !== "prefill_dense") {
        const idx_key = get(g, "dsa_index", true) ? 2 * dsa.heads * dsa.dim : 0;
        if (mode === "decode") tot += g.n * (per_key * pyMin(L, dsa.topk) + idx_key * L);
        else tot += divExact(big(g.n) * (big(per_key) * _span_causal_sum(L, dsa.topk) + big(idx_key) * _span_causal_sum(L, null)), big(L));
        continue;
      }
      if (mode === "decode") {
        const span = W ? pyMin(L, W) : L;
        tot += g.n * per_key * span;
      } else if (mode === "prefill_dense") tot += g.n * per_key * L;
      else tot += divExact(big(g.n) * big(per_key) * _span_causal_sum(L, W), big(L));
    }
    return tot;
  }
  function _one_kv_tensor_hf(p, g) { return !!g.k_eq_v && S("VNORM_TYPES").has(lower(p.mt)); }
  const _kv_tensors = (p, g) => (_one_kv_tensor_hf(p, g) ? 1 : 2);
  function kv_per_token(p) {
    const e = { global: 0, sliding: 0, mla_latent: 0, k_eq_v: 0, dsa_indexer: 0, kv_shared: 0 };
    const hf = { global: 0, sliding: 0 };
    let mn = 0;
    for (const g of p.groups) {
      const a = g.attn;
      if (a === null || g.kind === "shared") continue;
      let per_hf;
      if (g.mla) {
        e.mla_latent += g.n * (g.mla.kv_lora + g.mla.rope);
        mn += g.n * (g.mla.kv_lora + g.mla.rope);
        per_hf = a.nh * (a.dqk + a.dv);
      } else {
        let per;
        if (g.k_eq_v) {
          per = a.nkv * a.dqk * _kv_tensors(p, g);
          e.k_eq_v += g.n * per;
          mn += g.n * a.nkv * a.dqk;
        } else {
          per = a.nkv * (a.dqk + a.dv);
          e[g.kind === "full" ? "global" : "sliding"] += g.n * per;
          mn += g.n * per;
        }
        per_hf = per;
      }
      hf[g.kind === "full" ? "global" : "sliding"] += g.n * per_hf;
    }
    return [e, hf, mn];
  }
  function kv_at(p, L, convention = "headline") {
    const out = { global: 0, sliding: 0, mla_latent: 0, k_eq_v: 0, dsa_indexer: 0 };
    for (const g of p.groups) {
      const a = g.attn;
      if (a === null || g.kind === "shared") continue;
      const W = g.window;
      const toks = W ? pyMin(L, W - 1) : L;
      if (convention === "hf") {
        const per = g.mla ? a.nh * (a.dqk + a.dv) : (g.k_eq_v ? a.nkv * a.dqk * _kv_tensors(p, g) : a.nkv * (a.dqk + a.dv));
        out[!W ? "global" : "sliding"] += g.n * per * toks;
        continue;
      }
      if (g.mla) out.mla_latent += g.n * (g.mla.kv_lora + g.mla.rope) * toks;
      else if (g.k_eq_v) out.k_eq_v += g.n * a.nkv * a.dqk * toks * (convention === "minimum" ? 1 : _kv_tensors(p, g));
      else out[!W ? "global" : "sliding"] += g.n * a.nkv * (a.dqk + a.dv) * toks;
    }
    return out;
  }
  function state_elems(p) {
    let conv = 0, ssm = 0;
    for (const g of p.groups) { const r = g.rec; if (r) { conv += g.n * (r.conv_state || 0); ssm += g.n * (r.ssm_state || 0); } }
    return [conv, ssm];
  }
  function rec_flops(p) {
    let s = 0;
    for (const g of p.groups) { const r = g.rec; if (r) s += g.n * (r.core_flops || 0); }
    return s;
  }
  function flops(p, L, mode) {
    const mm = 2 * (matmul_weights(p) + p.lm_head_W + p.extra_W);
    const at = attn_flops(p, L, mode);
    const rc = rec_flops(p);
    return { matmul: pyInt(mm), attn: pyInt(pyRound(at)), recurrent: pyInt(rc), total: pyInt(pyRound(mm + at + rc)) };
  }
  function _kind_counts(p) {
    const c = { total: pyInt(p.layers), global: 0, sliding: 0, window: null, mla: 0, kv_shared: 0, k_eq_v: 0, recurrent: 0, dsa_indexer: 0 };
    for (const g of p.groups) {
      if (g.kind === "none") continue;
      if (g.attn !== null) {
        if (g.kind === "shared") c.kv_shared += g.n;
        else if (g.kind === "full") c.global += g.n;
        else { c.sliding += g.n; c.window = g.window; }
        if (g.mla) c.mla += g.n;
        if (g.k_eq_v) c.k_eq_v += g.n;
        if (truthy(p.dsa_dim) && g.kind !== "shared" && truthy(get(g, "dsa_index", true))) c.dsa_indexer += g.n;
      }
      if (g.rec !== null && g.rec.family !== "glue") c.recurrent += g.n;
    }
    return c;
  }
  function _kv_headline(p, L, convention = "headline") {
    const k = kv_at(p, L, convention);
    if (truthy(p.dsa_dim)) {
      const n = sum(p.groups.filter(g => g.attn !== null && g.kind !== "shared" && truthy(get(g, "dsa_index", true))).map(g => g.n));
      k.dsa_indexer = n * p.dsa_dim * L;
    }
    return k;
  }
  function _bytes(elems, dt) {
    const b = elems * AMT.DTYPE_BYTES[dt];
    return Number.isInteger(b) ? b : b;
  }
  function contexts_for(p) {
    const ctx = Object.assign({}, AMT.CONTEXTS);
    ctx.max = pyInt(p.max_context);
    return ctx;
  }
  const DT = () => Object.keys(AMT.DTYPE_BYTES);

  function metrics(raw, hub_total = null, tables = null, opts = null) {
    if (tables) setTables(tables);
    if (!T) throw new Error("ArchMetrics: tables not set");
    const batch_B = AMT.BATCH_B;
    const p = plan(raw, opts);
    const prm = params(p);
    const h = p.hidden;
    const out = { model_type: p.mt, arch: p.arch, max_context: pyInt(p.max_context), max_context_source: p.max_context_source ?? null, max_context_unbounded: !!p.max_context_unbounded, layers: _kind_counts(p), notes: [...p.notes] };
    prm.extra_towers = null;
    prm.hub_total = hub_total;
    prm.hub_delta = truthy(hub_total) ? hub_total - prm.total : null;
    prm.hub_delta_reason = null;
    prm.weight_bytes = {}; prm.active_weight_bytes = {};
    for (const dt of DT()) { prm.weight_bytes[dt] = _bytes(prm.total, dt); prm.active_weight_bytes[dt] = _bytes(prm.active, dt); }
    out.params = prm;
    let [e, hf, mn] = kv_per_token(p);
    if (truthy(p.dsa_dim)) { e.dsa_indexer = p.dsa_dim * out.layers.dsa_indexer; mn += e.dsa_indexer; }
    const tot = sum(Object.values(e));
    out.kv_per_token = { elems: e, elems_total: tot, bf16: _bytes(tot, "bf16"), fp8: _bytes(tot, "fp8") };
    const keqv_two = p.groups.some(g => g.k_eq_v && g.attn !== null && !_one_kv_tensor_hf(p, g));
    const [conv, ssm] = state_elems(p);
    const st = conv + ssm;
    out.state = { elems: st, conv_elems: conv, ssm_elems: ssm, bf16: 2 * st, fp32: 4 * st };
    const ctx = contexts_for(p);
    out.kv = {}; out.kv_materialized = { per_token_elems: sum(Object.values(hf)) };
    out.kv_minimum = { per_token_elems: mn, note: keqv_two ? "K = V layers counted as one tensor: the theoretical minimum; transformers, vLLM and llama.cpp allocate K " +
      "and V separately (the headline)" : "equal to the headline" };
    out.flops = { matmul: flops(p, 1, "decode").matmul, prefill: {}, decode: {} };
    out.memory = {}; out.bandwidth = {};
    const weights_bf16 = 2 * prm.total;
    const read_params = prm.active_non_embedding + p.lm_head_W + h;
    for (const [key, L] of Object.entries(ctx)) {
      const k = _kv_headline(p, L);
      const ke = sum(Object.values(k));
      const win = {};
      for (const g of p.groups) if (truthy(g.window)) win.sliding = L <= g.window - 1;
      out.kv[key] = { L, beyond_max: L > p.max_context && !p.max_context_unbounded, elems: k, in_window: win, bf16: _bytes(ke, "bf16"), fp8: _bytes(ke, "fp8") };
      const kh = sum(Object.values(kv_at(p, L, "hf")));
      out.kv_materialized[key] = { bf16: _bytes(kh, "bf16"), fp8: _bytes(kh, "fp8") };
      const km = sum(Object.values(_kv_headline(p, L, "minimum")));
      out.kv_minimum[key] = { bf16: _bytes(km, "bf16"), fp8: _bytes(km, "fp8") };
      const fp = flops(p, L, "prefill");
      const fpd = flops(p, L, "prefill_dense");
      out.flops.prefill[key] = { matmul: fp.matmul, attn: fp.attn, attn_dense: fpd.attn, recurrent: fp.recurrent, total: fp.total };
      const fd = flops(p, L, "decode");
      out.flops.decode[key] = fd;
      const kvb = 2 * ke;
      out.memory[key] = { weights: weights_bf16, kv: kvb, state: 2 * st, batch1: weights_bf16 + kvb + 2 * st, batchB: weights_bf16 + batch_B * (kvb + 2 * st) };
      const kv_read = 2 * sum(Object.values(_kv_headline(p, pyMax(L - 1, 0))));
      const wr = 2 * read_params;
      const b1 = wr + kv_read + 2 * 2 * st;
      const bB = wr + batch_B * (kv_read + 2 * 2 * st);
      out.bandwidth[key] = { weights: wr, kv: kv_read, state: 2 * 2 * st, bytes: b1, flops: fd.total,
        intensity: b1 ? pyRoundN(fd.total / b1, 4) : null,
        batchB: { bytes: bB, flops: batch_B * fd.total, intensity: bB ? pyRoundN(batch_B * fd.total / bB, 4) : null } };
    }
    if (keqv_two) out.notes.push("K = V layers: the KV counts K and V as two tensors, as transformers, vLLM and llama.cpp " +
      "allocate them; kv_minimum gives the one-tensor theoretical minimum");
    if (p.unknown && p.unknown.has("rec")) {
      const s2 = {};
      for (const k of Object.keys(out.state)) s2[k] = null;
      out.state = s2;
      out.notes.push("recurrent family not modelled: state and recurrent FLOPs unknown");
    }
    return out;
  }

  // ------------------------------------------------------------------------------------------ transformers 4.57 config classes
  // arch_metrics.resolve() runs `cls.from_dict(raw)`, `get_text_config(decoder=True)` and `to_dict()`. The emulation
  // replays each class's __init__ as the generator recorded it (tables tf.classes[*].prog): `self.x = local`
  // assignments, the super().__init__ call (PretrainedConfig.__init__ below) and every other statement through the
  // small Python interpreter further down (their ASTs in tf.stmts_ast). A statement the interpreter does not
  // support is skipped (UNHANDLED counts them).
  class Cfg {
    constructor(name) { this.cls = name; this.info = TF.classes[name]; this.d = {}; }
  }
  const isCfg = v => v instanceof Cfg;
  const UNHANDLED = new Map();
  const GEN_DEFAULTS = { max_length: 20, min_length: 0, do_sample: false, early_stopping: false, num_beams: 1, temperature: 1.0,
    top_k: 50, top_p: 1.0, typical_p: 1.0, repetition_penalty: 1.0, length_penalty: 1.0, no_repeat_ngram_size: 0,
    encoder_no_repeat_ngram_size: 0, bad_words_ids: null, num_return_sequences: 1, output_scores: false, return_dict_in_generate: false,
    forced_bos_token_id: null, forced_eos_token_id: null, remove_invalid_values: false, exponential_decay_length_penalty: null,
    suppress_tokens: null, begin_suppress_tokens: null, num_beam_groups: 1, diversity_penalty: 0.0 };
  const BASE_PARAMS = { output_hidden_states: false, output_attentions: false, return_dict: true, torchscript: false, dtype: null,
    pruned_heads: null, tie_word_embeddings: true, chunk_size_feed_forward: 0, is_encoder_decoder: false, is_decoder: false,
    cross_attention_hidden_size: null, add_cross_attention: false, tie_encoder_decoder: false, architectures: null, finetuning_task: null,
    id2label: null, label2id: null, num_labels: null, task_specific_params: null, problem_type: null, tokenizer_class: null, prefix: null,
    bos_token_id: null, pad_token_id: null, eos_token_id: null, sep_token_id: null, decoder_start_token_id: null };
  const amapKey = (c, k) => (c.info && hasOwn(c.info.amap, k) ? c.info.amap[k] : k);
  // PretrainedConfig's own properties (4.57)
  const BASE_PROPS = { name_or_path: "rw", output_attentions: "rw", use_return_dict: "ro", num_labels: "rw", _attn_implementation: "rw",
    torch_dtype: "rw" };
  function cfgHas(c, k) {
    k = amapKey(c, k);
    return hasOwn(c.d, k) || (c.info && (hasOwn(c.info.props || {}, k) || hasOwn(c.info.attrs || {}, k))) || k === "model_type" ||
      hasOwn(BASE_PROPS, k);
  }
  function cfgGet(c, k, def) {
    const k2 = amapKey(c, k);
    if (hasOwn(c.d, k2)) return c.d[k2];
    if (k2 === "model_type") return c.info.mt;
    if (k2 === "output_attentions") return hasOwn(c.d, "_output_attentions") ? c.d._output_attentions : false;
    if (k2 === "num_labels") return isDict(c.d.id2label) ? Object.keys(c.d.id2label).length : 2;
    if (k2 === "name_or_path") return c.d._name_or_path;
    if (k2 === "use_return_dict") return truthy(c.d.return_dict) && !truthy(c.d.torchscript);
    if (k2 === "torch_dtype") return hasOwn(c.d, "dtype") ? c.d.dtype : null;
    if (k2 === "_attn_implementation") return hasOwn(c.d, "_attn_implementation_internal") ? c.d._attn_implementation_internal : null;
    const gt = c.info && c.info.getters && c.info.getters[k2];
    if (gt) return callFunction(gt, [c], null, c);
    if (c.info && hasOwn(c.info.attrs || {}, k2)) return c.info.attrs[k2];
    if (arguments.length > 2) return def;
    throw new PyError("AttributeError", `'${c.cls}' object has no attribute '${k}'`);
  }
  function cfgSet(c, k, v) {
    k = amapKey(c, k);
    const pr = c.info && c.info.props && hasOwn(c.info.props, k) ? c.info.props[k] : (hasOwn(BASE_PROPS, k) ? BASE_PROPS[k] : null);
    if (pr === "ro") throw new PyError("AttributeError", `property '${k}' of '${c.cls}' object has no setter`);
    if (k === "output_attentions") { c.d._output_attentions = v; return; }
    if (k === "torch_dtype") { c.d.dtype = v; return; }
    if (k === "_attn_implementation") { c.d._attn_implementation_internal = v; return; }
    if (k === "name_or_path") { c.d._name_or_path = pyStr(v); return; }
    if (k === "num_labels") {
      if (!isDict(c.d.id2label) || Object.keys(c.d.id2label).length !== v) {
        c.d.id2label = {}; c.d.label2id = {};
        for (let i = 0; i < pyInt(v); i++) { c.d.id2label[i] = `LABEL_${i}`; c.d.label2id[`LABEL_${i}`] = i; }
      }
      return;
    }
    c.d[k] = v;
  }
  // PretrainedConfig.__init__(**kw)
  function baseInit(c, kw) {
    const P = Object.assign({}, BASE_PARAMS);
    const K = {};
    for (const [k, v] of Object.entries(kw)) { if (hasOwn(BASE_PARAMS, k)) P[k] = v; else K[k] = v; }
    if (P.label2id !== null && !isDict(P.label2id)) throw new PyError("ValueError", "Argument label2id should be a dictionary.");
    if (P.id2label !== null && !isDict(P.id2label)) throw new PyError("ValueError", "Argument id2label should be a dictionary.");
    if (P.problem_type !== null && !["regression", "single_label_classification", "multi_label_classification"].includes(P.problem_type))
      throw new PyError("ValueError", `The config parameter \`problem_type\` was not understood: received ${pyStr(P.problem_type)}`);
    let dtype = P.dtype;
    const td = hasOwn(K, "torch_dtype") ? K.torch_dtype : null;
    delete K.torch_dtype;
    if (td !== null) dtype = dtype !== null ? dtype : td;
    if (dtype !== null && isStr(dtype)) {
      if (!(TF.torch_dtypes || []).includes(dtype)) throw new PyError("AttributeError", `module 'torch' has no attribute '${dtype}'`);
    }
    for (const k of ["return_dict", "output_hidden_states", "torchscript"]) cfgSet(c, k, P[k]);
    c.d.dtype = dtype;
    c.d._output_attentions = P.output_attentions;
    c.d.pruned_heads = P.pruned_heads !== null ? P.pruned_heads : {};
    for (const k of ["tie_word_embeddings", "chunk_size_feed_forward", "is_encoder_decoder", "is_decoder", "cross_attention_hidden_size",
      "add_cross_attention", "tie_encoder_decoder", "architectures", "finetuning_task", "id2label", "label2id", "task_specific_params",
      "problem_type"]) cfgSet(c, k, P[k]);
    if (c.d.id2label === null || c.d.id2label === undefined) {
      const n = P.num_labels !== null ? pyInt(P.num_labels) : 2;
      c.d.id2label = {}; c.d.label2id = {};
      for (let i = 0; i < n; i++) { c.d.id2label[i] = `LABEL_${i}`; c.d.label2id[`LABEL_${i}`] = i; }
    } else {
      const o = {};
      for (const [k, v] of Object.entries(c.d.id2label)) o[pyInt(k)] = v;
      c.d.id2label = o;
    }
    for (const k of ["tokenizer_class", "prefix", "bos_token_id", "pad_token_id", "eos_token_id", "sep_token_id", "decoder_start_token_id"]) cfgSet(c, k, P[k]);
    for (const [k, v] of Object.entries(GEN_DEFAULTS)) { cfgSet(c, k, hasOwn(K, k) ? K[k] : v); delete K[k]; }
    c.d._name_or_path = pyStr(hasOwn(K, "name_or_path") ? K.name_or_path : ""); delete K.name_or_path;
    c.d._commit_hash = hasOwn(K, "_commit_hash") ? K._commit_hash : null; delete K._commit_hash;
    c.d._attn_implementation_internal = hasOwn(K, "attn_implementation") ? K.attn_implementation : null; delete K.attn_implementation;
    c.d.transformers_version = hasOwn(K, "transformers_version") ? K.transformers_version : null; delete K.transformers_version;
    for (const [k, v] of Object.entries(K)) cfgSet(c, k, v);
    c.d.tf_legacy_loss = hasOwn(K, "tf_legacy_loss") ? K.tf_legacy_loss : false;
    c.d.use_bfloat16 = hasOwn(K, "use_bfloat16") ? K.use_bfloat16 : false;
  }
  // the super().__init__(...) statement: explicit keywords from the locals, **kwargs from K
  function superCall(c, kws, L, K, X) {
    const kw = {};
    for (const [name, val] of kws) {
      let v;
      if (val[0] === "n") v = name === null ? null : (val[1] === "kwargs" ? K : (hasOwn(L, val[1]) ? L[val[1]] : null));
      else if (val[0] === "c") v = val[1];
      else v = EXPR(val[1], X);
      if (name === null) {
        const src = val[0] === "n" && val[1] === "kwargs" ? K : v;
        for (const [k, x] of Object.entries(src || {})) {
          if (hasOwn(kw, k)) throw new PyError("TypeError", `got multiple values for keyword argument '${k}'`);
          kw[k] = x;
        }
      } else {
        if (hasOwn(kw, name)) throw new PyError("TypeError", `got multiple values for keyword argument '${name}'`);
        kw[name] = v;
      }
    }
    baseInit(c, kw);
  }
  // cls(**kwargs)
  function instantiate(name, kwargs) {
    if (!name || !hasOwn(TF.classes, name) || !TF.classes[name]) throw new PyError("KeyError", pyRepr(name));
    const c = new Cfg(name);
    const info = c.info;
    const L = {}, K = {};
    for (const [p, def] of Object.entries(info.params)) L[p] = def === "__required__" ? undefined : deepcopy(def);
    for (const [k, v] of Object.entries(kwargs || {})) { if (hasOwn(info.params, k)) L[k] = v; else K[k] = v; }
    for (const [p, v] of Object.entries(L)) if (v === undefined) throw new PyError("TypeError", `${name}.__init__() missing 1 required positional argument: '${p}'`);
    const X = { c, L, K, info, get: (k, ...d) => cfgGet(c, k, ...d), set: (k, v) => cfgSet(c, k, v), has: k => cfgHas(c, k) };
    const prog = info.prog || (Object.keys(info.params).map(p => ["a", p, p]).concat([["S", [[null, ["n", "kwargs"]]]]]));
    for (const st of prog) {
      if (st[0] === "a") { const loc = st.length > 2 ? st[2] : st[1]; cfgSet(c, st[1], hasOwn(L, loc) ? L[loc] : null); }
      else if (st[0] === "S") superCall(c, st[1], L, K, X);
      else {
        try { interpret(st[1], X); } catch (e) {
          if (!(e instanceof Unsupported)) throw e;
          const key = `statement ${st[1]} (${c.cls}): ${e.message}`;
          UNHANDLED.set(key, (UNHANDLED.get(key) || 0) + 1);
        }
      }
    }
    return c;
  }
  function toDict(c) {
    const out = {};
    for (const [k, v] of Object.entries(c.d)) {
      if (isCfg(v)) { const sub = toDict(v); delete sub.transformers_version; out[k] = sub; }
      else out[k] = deepcopy(v);
    }
    out.model_type = c.info.mt;
    out.transformers_version = TF.version;
    const strip = o => {                                             // _remove_keys_not_serialized
      if (hasOwn(o, "quantization_config") || (o === out && cfgHas(c, "quantization_config"))) delete o._pre_quantization_dtype;
      delete o._auto_class;
      if (hasOwn(o, "_output_attentions")) { o.output_attentions = o._output_attentions; delete o._output_attentions; }
      for (const k of ["_commit_hash", "_attn_implementation_internal", "base_model_tp_plan", "base_model_pp_plan"]) delete o[k];
      for (const v of Object.values(o)) if (isDict(v)) strip(v);
    };
    strip(out);
    return out;
  }
  function getTextConfig(c) {
    const valid = ["decoder", "generator", "text_config"].filter(n => cfgHas(c, n) && cfgGet(c, n, null) !== null);
    if (valid.length > 1) throw new PyError("ValueError", `Multiple valid text configs were found in the model config: ${pyStr(valid)}`);
    if (valid.length === 1) return cfgGet(c, valid[0]);
    if (truthy(cfgGet(c, "is_encoder_decoder", false))) {
      // a legacy flat encoder-decoder config: drop the encoder_* keys, decoder_* -> the plain names
      const t = new Cfg(c.cls);
      t.d = deepcopyCfg(c.d);
      const mapped = new Set(Object.values(c.info.amap || {}));
      for (const key of Object.keys(toDict(c))) {
        if (key.startsWith("encoder") && !mapped.has(key)) {
          if (!hasOwn(t.d, key)) throw new PyError("AttributeError", key);
          delete t.d[key];
        }
        if (key.startsWith("decoder")) {
          let nk = key === "decoder_layers" ? "num_hidden_layers" : key === "decoder_attention_heads" ? "num_attention_heads" : key.slice(8);
          if (hasOwn(c.info.amap || {}, nk)) nk = c.info.amap[nk];
          const v = cfgGet(t, key);
          if (!hasOwn(t.d, key)) throw new PyError("AttributeError", key);
          delete t.d[key];
          cfgSet(t, nk, v);
        }
      }
      return t;
    }
    return c;
  }
  function deepcopyCfg(d) {
    const o = {};
    for (const [k, v] of Object.entries(d)) o[k] = isCfg(v) ? Object.assign(new Cfg(v.cls), { d: deepcopyCfg(v.d) }) : deepcopy(v);
    return o;
  }
  // PretrainedConfig.__repr__ -> to_json_string -> to_diff_dict: the parts that can raise
  let BASE_DEFAULTS = null;
  function baseDefaults() {
    if (!BASE_DEFAULTS) {
      const c = new Cfg("__base__");
      c.info = { mt: "", amap: {}, params: {}, prog: [["S", [[null, ["n", "kwargs"]]]]], props: {}, attrs: {}, subs: {} };
      baseInit(c, {});
      BASE_DEFAULTS = toDict(c);
    }
    return BASE_DEFAULTS;
  }
  function recursiveDiff(a, b, obj) {
    const dflt = obj ? toDict(instantiate(obj.cls, {})) : {};
    if (!isDict(a)) throw new PyError("AttributeError", `'${pyType(a)}' object has no attribute 'items'`);
    for (const [k, v] of Object.entries(a)) {
      const ov = obj ? cfgGet(obj, k, null) : null;
      if (isCfg(ov) && hasOwn(b, k) && isDict(b[k])) recursiveDiff(v, b[k], ov);
      else if (hasOwn(b, k)) {
        if (!hasOwn(dflt, k)) throw new PyError("KeyError", pyRepr(k));
      }
    }
  }
  function reprCheck(c) {
    const cd = toDict(c);
    const base = baseDefaults();
    const nodef = truthy(c.info.attrs && c.info.attrs.has_no_defaults_at_init);
    const ccd = nodef ? {} : toDict(instantiate(c.cls, {}));
    const subs = c.info.subs || {};
    for (const [k, v] of Object.entries(cd)) {
      if ((isCfg(cfgGet(c, k, null)) && hasOwn(ccd, k) && isDict(ccd[k])) || hasOwn(subs, k)) recursiveDiff(v, base, cfgGet(c, k, null));
    }
  }
  // arch_metrics.resolve, stock branch
  function resolveStock(raw, mt, clsName) {
    const kwargs = deepcopy(raw);
    kwargs.attn_implementation = null;
    const hc = instantiate(clsName, kwargs);
    if (cfgHas(hc, "pruned_heads")) {
      const ph = cfgGet(hc, "pruned_heads");
      if (!isDict(ph)) throw new PyError("AttributeError", `'${pyType(ph)}' object has no attribute 'items'`);
      const o = {};
      for (const [k, v] of Object.entries(ph)) o[pyInt(k)] = v;
      cfgSet(hc, "pruned_heads", o);
    }
    reprCheck(hc);                                     // from_dict logs f"Model config {config}": __repr__ runs
    const text = getTextConfig(hc);
    if (!isCfg(text)) throw new PyError("AttributeError", `'${pyType(text)}' object has no attribute 'to_dict'`);
    const d = toDict(text);
    if (!hasOwn(d, "model_type")) d.model_type = text.info.mt || mt;
    if (text !== hc) d._outer_tie_word_embeddings = cfgGet(hc, "tie_word_embeddings");
    return d;
  }

  // Python helpers for the statements
  function pyCmp(a, b) {                                                 // a < b, a == b with Python's TypeError
    if (isNumLike(a) && isNumLike(b)) return numVal(a) - numVal(b);
    if (isStr(a) && isStr(b)) return a < b ? -1 : a > b ? 1 : 0;
    throw new PyError("TypeError", `'<' not supported between instances of '${pyType(a)}' and '${pyType(b)}'`);
  }
  function pyLen(v) {
    if (isStr(v) || isList(v)) return v.length;
    if (isDict(v)) return Object.keys(v).length;
    throw new PyError("TypeError", `object of type '${pyType(v)}' has no len()`);
  }
  function pyIter(v) {
    if (isList(v)) return v;
    if (isStr(v)) return [...v];
    if (isDict(v)) return Object.keys(v);
    throw new PyError("TypeError", `'${pyType(v)}' object is not iterable`);
  }
  const pyIn = (x, v) => (isStr(v) ? (isStr(x) ? v.includes(x) : (() => { throw new PyError("TypeError", "'in <string>' requires string as left operand"); })())
    : isDict(v) ? hasOwn(v, isStr(x) ? x : pyStr(x)) : pyIter(v).some(y => pyEq(x, y)));
  function dget(v, k, def = null) {                                      // v.get(k, def) on a Python dict
    if (!isDict(v)) throw new PyError("AttributeError", `'${pyType(v)}' object has no attribute 'get'`);
    return hasOwn(v, k) ? v[k] : def;
  }
  function pyAdd(a, b) {
    if (isNumLike(a) && isNumLike(b)) return numVal(a) + numVal(b);
    if (isStr(a) && isStr(b)) return a + b;
    if (isList(a) && isList(b)) return a.concat(b);
    throw new PyError("TypeError", `unsupported operand type(s) for +: '${pyType(a)}' and '${pyType(b)}'`);
  }
  function pyMul(a, b) {
    if (isNumLike(a) && isNumLike(b)) return numVal(a) * numVal(b);
    if (isList(a) && isIntLike(b)) { let o = []; for (let i = 0; i < numVal(b); i++) o = o.concat(a); return o; }
    if (isStr(a) && isIntLike(b)) return a.repeat(Math.max(0, numVal(b)));
    throw new PyError("TypeError", `unsupported operand type(s) for *: '${pyType(a)}' and '${pyType(b)}'`);
  }
  function pyFloordiv(a, b) {
    if (!(isNumLike(a) && isNumLike(b))) throw new PyError("TypeError", `unsupported operand type(s) for //: '${pyType(a)}' and '${pyType(b)}'`);
    return floordiv(a, b);
  }
  function pyMod(a, b) {
    if (!(isNumLike(a) && isNumLike(b))) throw new PyError("TypeError", `unsupported operand type(s) for %: '${pyType(a)}' and '${pyType(b)}'`);
    return pymod(a, b);
  }
  function pyDiv(a, b) {
    if (!(isNumLike(a) && isNumLike(b))) throw new PyError("TypeError", `unsupported operand type(s) for /: '${pyType(a)}' and '${pyType(b)}'`);
    if (numVal(b) === 0) throw new PyError("ZeroDivisionError", "division by zero");
    return numVal(a) / numVal(b);
  }
  function pyRange(...a) {
    a = a.map(x => { if (!isIntLike(x)) throw new PyError("TypeError", `'${pyType(x)}' object cannot be interpreted as an integer`); return numVal(x); });
    let [start, stop, step] = a.length === 1 ? [0, a[0], 1] : [a[0], a[1], a.length > 2 ? a[2] : 1];
    if (step === 0) throw new PyError("ValueError", "range() arg 3 must not be zero");
    const o = [];
    if (step > 0) for (let i = start; i < stop; i += step) o.push(i); else for (let i = start; i > stop; i += step) o.push(i);
    return o;
  }
  const isFloatish = v => isNumber(v) && !Number.isInteger(v);          // isinstance(v, float), as JSON can tell
  // transformers.configuration_utils.layer_type_validation
  function layer_type_validation(lt, n = null) {
    const allowed = ["full_attention", "sliding_attention", "chunked_attention", "linear_attention"];
    if (!pyIter(lt).every(x => allowed.includes(x))) throw new PyError("ValueError", `The \`layer_types\` entries must be in ${pyStr(allowed)}`);
    if (n !== null && !pyEq(n, pyLen(lt))) throw new PyError("ValueError", `\`num_hidden_layers\` (${pyStr(n)}) must be equal to the number of layer types (${pyLen(lt)})`);
  }
  // transformers.modeling_rope_utils.rope_config_validation: the statements that can raise
  function rope_config_validation(X, ignore = null) {
    const rs = X.get("rope_scaling", null);
    if (rs === null) return;
    const rope_type = dget(rs, "rope_type", dget(rs, "type", "default"));
    const check = (req, opt) => {
      let received = new Set(Object.keys(rs));
      const required = new Set(req);
      if (received.has("type")) { received.delete("type"); required.add("rope_type"); }
      if (ignore) for (const k of ignore) received.delete(k);
      const missing = [...required].filter(k => !received.has(k));
      if (missing.length) throw new PyError("KeyError", `"Missing required keys in \`rope_scaling\` for 'rope_type'='${pyStr(rope_type)}': ${missing}"`);
    };
    const rt = isStr(rope_type) ? rope_type : null;
    if (rt === "default") check(["rope_type"]);
    else if (rt === "linear" || rt === "dynamic") check(["rope_type", "factor"]);
    else if (rt === "yarn") {
      check(["rope_type", "factor"]);
      const bf = dget(rs, "beta_fast"), bs = dget(rs, "beta_slow");
      pyCmp(por(bf, 32), por(bs, 1));
      const omp = dget(rs, "original_max_position_embeddings");
      if (omp !== null) pyDiv(X.get("max_position_embeddings"), omp);
    } else if (rt === "longrope") {
      check(["rope_type", "short_factor", "long_factor"]);
      const prf = X.get("partial_rotary_factor", 1.0);
      const dflt = pyFloordiv(X.get("hidden_size"), X.get("num_attention_heads"));
      const hd = X.has("head_dim") ? X.get("head_dim") : dflt;
      const dim = pyInt(pyMul(hd, prf));
      for (const k of ["short_factor", "long_factor"]) {
        const f = dget(rs, k);
        if (!isList(f)) pyIter(f);
        if (pyLen(f) !== floordiv(dim, 2)) { /* a warning */ }
      }
    } else if (rt === "llama3") {
      check(["rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"]);
      pyCmp(rs.high_freq_factor, rs.low_freq_factor);
      pyCmp(rs.original_max_position_embeddings, X.get("max_position_embeddings"));
    }
  }
  function mapCls(mt) {
    if (!isStr(mt) || !hasOwn(TF.mapping, mt)) throw new PyError("KeyError", pyRepr(mt));
    return TF.mapping[mt];
  }
  function EXPR(node, X) {                                               // keyword values of super().__init__ that are expressions
    return evalExpr(node, { vars: X.L, self: X.c, K: X.K });
  }
  RESOLVE = (raw, mt, cls) => resolveStock(raw, mt, cls);

  // ------------------------------------------------------------------------------------------ a small Python interpreter
  // for the config __init__ statements and the self._methods / properties they call: tf.stmts_ast holds each
  // statement's AST (graph/arch_metrics_js.py _encode). It covers the subset these classes use; anything else raises
  // Unsupported, and the statement is skipped (UNHANDLED counts it).
  class Unsupported extends Error {}
  class ReturnSignal { constructor(v) { this.v = v; } }
  const PYTYPE = name => ({ __pytype: name });
  const TYPES = { dict: PYTYPE("dict"), list: PYTYPE("list"), int: PYTYPE("int"), float: PYTYPE("float"), str: PYTYPE("str"),
    bool: PYTYPE("bool"), tuple: PYTYPE("tuple"), set: PYTYPE("set"), PretrainedConfig: PYTYPE("PretrainedConfig"),
    Sequence: PYTYPE("Sequence") };
  const EXC = ["ValueError", "TypeError", "KeyError", "AttributeError", "IndexError", "NotImplementedError", "RuntimeError",
    "AssertionError", "ZeroDivisionError", "Exception", "ImportError", "OverflowError", "FutureWarning", "UserWarning", "DeprecationWarning"];
  function isInstance(v, t) {
    if (isList(t)) return t.some(x => isInstance(v, x));
    if (t && t.__pycls) return isCfg(v) && (v.cls === t.__pycls || (TF.classes[v.cls] && false));
    const n = t && t.__pytype;
    if (n === "dict") return isDict(v) && !isCfg(v) && !v.__pycls && !v.__pytype;
    if (n === "list") return isList(v);
    if (n === "tuple") return false;
    if (n === "int") return isIntLike(v);
    if (n === "float") return isFloatish(v);
    if (n === "str") return isStr(v);
    if (n === "bool") return isBool(v);
    if (n === "set") return v instanceof Set;
    if (n === "Sequence") return isList(v) || isStr(v);
    if (n === "PretrainedConfig") return isCfg(v);
    throw new Unsupported("isinstance " + JSON.stringify(t));
  }
  const NOOP_OBJ = new Proxy({}, { get: () => () => null });
  const MATH = { ceil: a => Math.ceil(numVal(a[0])), floor: a => Math.floor(numVal(a[0])), sqrt: a => Math.sqrt(numVal(a[0])),
    log: a => Math.log(numVal(a[0])), log2: a => Math.log2(numVal(a[0])), prod: a => pyIter(a[0]).reduce((x, y) => pyMul(x, y), 1),
    isqrt: a => Math.floor(Math.sqrt(numVal(a[0]))) };
  function builtin(name) {
    switch (name) {
      case "len": return (a) => pyLen(a[0]);
      case "int": return (a) => (a.length ? pyInt(a[0]) : 0);
      case "float": return (a) => (a.length ? pyFloat(a[0]) : 0);
      case "bool": return (a) => truthy(a[0]);
      case "str": return (a) => (a.length ? pyStr(a[0]) : "");
      case "list": return (a) => (a.length ? [...pyIter(a[0])] : []);
      case "tuple": return (a) => (a.length ? [...pyIter(a[0])] : []);
      case "set": return (a) => new Set(a.length ? pyIter(a[0]) : []);
      case "dict": return (a, kw) => Object.assign({}, a.length ? a[0] : {}, kw || {});
      case "range": return (a) => pyRange(...a);
      case "isinstance": return (a) => isInstance(a[0], a[1]);
      case "hasattr": return (a) => (isCfg(a[0]) ? cfgHas(a[0], a[1]) : isDict(a[0]) ? false : false);
      case "getattr": return (a) => (isCfg(a[0]) ? (a.length > 2 ? cfgGet(a[0], a[1], a[2]) : cfgGet(a[0], a[1])) :
        (() => { if (a.length > 2) return a[2]; throw new PyError("AttributeError", `no attribute '${a[1]}'`); })());
      case "setattr": return (a) => { if (!isCfg(a[0])) throw new Unsupported("setattr"); cfgSet(a[0], a[1], a[2]); return null; };
      case "any": return (a) => pyIter(a[0]).some(truthy);
      case "all": return (a) => pyIter(a[0]).every(truthy);
      case "sum": return (a) => pyIter(a[0]).reduce((x, y) => pyAdd(x, y), a.length > 1 ? a[1] : 0);
      case "min": return (a) => { const xs = a.length === 1 ? pyIter(a[0]) : a; return xs.reduce((x, y) => (pyCmp(y, x) < 0 ? y : x)); };
      case "max": return (a) => { const xs = a.length === 1 ? pyIter(a[0]) : a; return xs.reduce((x, y) => (pyCmp(y, x) > 0 ? y : x)); };
      case "abs": return (a) => Math.abs(numVal(a[0]));
      case "round": return (a) => (a.length > 1 && a[1] !== null ? pyRoundN(numVal(a[0]), numVal(a[1])) : pyRound(numVal(a[0])));
      case "sorted": return (a) => [...pyIter(a[0])].sort((x, y) => pyCmp(x, y));
      case "enumerate": return (a) => pyIter(a[0]).map((x, i) => [i + (a.length > 1 ? numVal(a[1]) : 0), x]);
      case "zip": return (a) => { const ls = a.map(pyIter); const n = Math.min(...ls.map(l => l.length)); return _range(n).map(i => ls.map(l => l[i])); };
      case "reversed": return (a) => [...pyIter(a[0])].reverse();
      case "print": return () => null;
      case "super": throw new Unsupported("super");
    }
    return undefined;
  }
  function lookupName(name, env) {
    if (hasOwn(env.vars, name)) return env.vars[name];
    if (name === "self") return env.self;
    if (name === "kwargs" && env.K) return env.K;
    if (name === "None") return null;
    if (name === "True") return true;
    if (name === "False") return false;
    if (hasOwn(TYPES, name)) return TYPES[name];
    const b = builtin(name);
    if (b) return { __pyfn: b };
    if (EXC.includes(name)) return { __pyexc: name };
    if (name === "math") return { __pymod: MATH };
    if (name === "logger" || name === "warnings" || name === "logging") return { __pynoop: true };
    if (name === "CONFIG_MAPPING") return { __pycfgmap: true };
    if (name === "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES") return Object.fromEntries((TF.causal_lm_types || []).map(t => [t, t]));
    if (name === "AutoConfig") return { __pyautocfg: true };
    if (name === "rope_config_validation") return { __pyfn: (a, kw) => { rope_config_validation(wrapX(a[0]), (kw && kw.ignore_keys) ? [...kw.ignore_keys] : (a[1] ? [...a[1]] : null)); return null; } };
    if (name === "layer_type_validation") return { __pyfn: a => { layer_type_validation(a[0], a.length > 1 ? a[1] : null); return null; } };
    if (hasOwn(TF.classes, name) && TF.classes[name]) return { __pycls: name };
    throw new Unsupported("name " + name);
  }
  const wrapX = c => ({ c, info: c.info, get: (k, ...d) => cfgGet(c, k, ...d), set: (k, v) => cfgSet(c, k, v), has: k => cfgHas(c, k) });
  function getAttr(obj, name) {
    if (isCfg(obj)) {
      const m = obj.info && obj.info.methods && obj.info.methods[name];
      if (m) return { __pyfn: (a, kw) => callFunction(m, m.static ? a : [obj, ...a], kw, obj) };
      if (name === "sub_configs") return Object.fromEntries(Object.entries(obj.info.subs || {}).map(([k, v]) => [k, v ? { __pycls: v } : { __pyautocfg: true }]));
      if (!cfgHas(obj, name) && !(obj.info.getters && obj.info.getters[name])) throw new Unsupported("config attribute " + name);
      return cfgGet(obj, name);
    }
    if (obj && obj.__pymod) { if (hasOwn(obj.__pymod, name)) return { __pyfn: obj.__pymod[name] }; throw new Unsupported("module attr " + name); }
    if (obj && obj.__pynoop) return { __pyfn: () => null };
    if (obj && obj.__pyautocfg) {
      if (name === "for_model") return { __pyfn: (a, kw) => instantiate(mapCls(a[0]), kw || {}) };
      throw new Unsupported("AutoConfig." + name);
    }
    if (isDict(obj)) {
      const methods = {
        get: a => dget(obj, a[0], a.length > 1 ? a[1] : null),
        pop: a => { if (hasOwn(obj, a[0])) { const v = obj[a[0]]; delete obj[a[0]]; return v; } if (a.length > 1) return a[1]; throw new PyError("KeyError", pyRepr(a[0])); },
        items: () => Object.entries(obj), keys: () => Object.keys(obj), values: () => Object.values(obj),
        update: (a, kw) => { Object.assign(obj, a.length ? a[0] : {}, kw || {}); return null; },
        setdefault: a => { if (!hasOwn(obj, a[0])) obj[a[0]] = a.length > 1 ? a[1] : null; return obj[a[0]]; },
        copy: () => Object.assign({}, obj),
      };
      if (hasOwn(methods, name)) return { __pyfn: methods[name] };
      throw new PyError("AttributeError", `'dict' object has no attribute '${name}'`);
    }
    if (isList(obj)) {
      const methods = { append: a => { obj.push(a[0]); return null; }, extend: a => { obj.push(...pyIter(a[0])); return null; },
        copy: () => [...obj], index: a => { const i = obj.findIndex(x => pyEq(x, a[0])); if (i < 0) throw new PyError("ValueError", "not in list"); return i; },
        count: a => obj.filter(x => pyEq(x, a[0])).length, pop: a => (a.length ? obj.splice(numVal(a[0]), 1)[0] : obj.pop()),
        insert: a => { obj.splice(numVal(a[0]), 0, a[1]); return null; } };
      if (hasOwn(methods, name)) return { __pyfn: methods[name] };
      throw new PyError("AttributeError", `'list' object has no attribute '${name}'`);
    }
    if (isStr(obj)) {
      const methods = { startswith: a => (isList(a[0]) ? a[0].some(p => obj.startsWith(p)) : obj.startsWith(a[0])),
        endswith: a => (isList(a[0]) ? a[0].some(p => obj.endsWith(p)) : obj.endsWith(a[0])), lower: () => obj.toLowerCase(),
        upper: () => obj.toUpperCase(), strip: () => obj.trim(), split: a => (a.length && a[0] !== null ? obj.split(a[0]) : obj.trim().split(/\s+/)),
        replace: a => obj.split(a[0]).join(a[1]), join: a => pyIter(a[0]).join(obj), format: () => obj };
      if (hasOwn(methods, name)) return { __pyfn: methods[name] };
    }
    if (obj instanceof Set) {
      const methods = { add: a => { obj.add(a[0]); return null; } };
      if (hasOwn(methods, name)) return { __pyfn: methods[name] };
    }
    throw new Unsupported("attribute " + name + " of " + pyType(obj));
  }
  function setAttrTarget(obj, name, v) {
    if (isCfg(obj)) { cfgSet(obj, name, v); return; }
    throw new Unsupported("setattr on " + pyType(obj));
  }
  function callValue(fn, args, kw) {
    if (fn && fn.__pyfn) return fn.__pyfn(args, kw);
    if (fn && fn.__pycls) { if (args.length) throw new Unsupported("positional config args"); return instantiate(fn.__pycls, kw || {}); }
    if (fn && fn.__pyexc) return { __pyexcval: fn.__pyexc, msg: args.length ? pyStr(args[0]) : "" };
    if (fn && fn.__pytype) return builtin(fn.__pytype)(args, kw);
    throw new Unsupported("call of " + pyType(fn));
  }
  function callFunction(f, args, kw, selfObj) {
    const env = { vars: {}, self: selfObj, K: null };
    const names = f.args;
    const nd = (f.defaults || []).length;
    names.forEach((n, i) => {
      if (i < args.length) env.vars[n] = args[i];
      else if (kw && hasOwn(kw, n)) env.vars[n] = kw[n];
      else if (i >= names.length - nd) env.vars[n] = evalExpr(f.defaults[i - (names.length - nd)], env);
      else throw new PyError("TypeError", `missing argument '${n}'`);
    });
    if (f.kwarg) env.vars[f.kwarg] = Object.assign({}, kw || {});
    if (f.vararg) env.vars[f.vararg] = args.slice(names.length);
    try { execBlock(f.body, env); } catch (e) { if (e instanceof ReturnSignal) return e.v; throw e; }
    return null;
  }
  function binop(op, a, b) {
    switch (op) {
      case "Add": return pyAdd(a, b);
      case "Sub": if (a instanceof Set && b instanceof Set) return new Set([...a].filter(x => !b.has(x)));
        if (!(isNumLike(a) && isNumLike(b))) throw new PyError("TypeError", "unsupported operand type(s) for -"); return numVal(a) - numVal(b);
      case "Mult": return isList(b) || isStr(b) ? pyMul(b, a) : pyMul(a, b);
      case "Div": return pyDiv(a, b);
      case "FloorDiv": return pyFloordiv(a, b);
      case "Mod": if (isStr(a)) throw new Unsupported("% format"); return pyMod(a, b);
      case "Pow": if (!(isNumLike(a) && isNumLike(b))) throw new PyError("TypeError", "unsupported operand type(s) for **"); return Math.pow(numVal(a), numVal(b));
      case "BitOr": if (a instanceof Set) return new Set([...a, ...b]); if (isDict(a) && isDict(b)) return Object.assign({}, a, b); return numVal(a) | numVal(b);
      case "BitAnd": if (a instanceof Set) return new Set([...a].filter(x => b.has(x))); return numVal(a) & numVal(b);
    }
    throw new Unsupported("binop " + op);
  }
  function compare(op, a, b) {
    switch (op) {
      case "Eq": return pyEq(a, b);
      case "NotEq": return !pyEq(a, b);
      case "Lt": return pyCmp(a, b) < 0;
      case "LtE": return pyCmp(a, b) <= 0;
      case "Gt": return pyCmp(a, b) > 0;
      case "GtE": return pyCmp(a, b) >= 0;
      case "Is": return a === b || (a === null && b === null);
      case "IsNot": return !(a === b || (a === null && b === null));
      case "In": return b instanceof Set ? [...b].some(x => pyEq(x, a)) : pyIn(a, b);
      case "NotIn": return !(b instanceof Set ? [...b].some(x => pyEq(x, a)) : pyIn(a, b));
    }
    throw new Unsupported("compare " + op);
  }
  function subscript(obj, key) {
    if (isList(obj) || isStr(obj)) {
      if (key && key.__slice) {
        const n = obj.length;
        let [lo, hi, st] = key.__slice;
        st = st === null ? 1 : st;
        if (st !== 1) throw new Unsupported("slice step");
        lo = lo === null ? 0 : (lo < 0 ? Math.max(0, n + lo) : Math.min(lo, n));
        hi = hi === null ? n : (hi < 0 ? Math.max(0, n + hi) : Math.min(hi, n));
        return obj.slice(lo, Math.max(lo, hi));
      }
      if (!isIntLike(key)) throw new PyError("TypeError", "indices must be integers");
      let i = numVal(key);
      if (i < 0) i += obj.length;
      if (i < 0 || i >= obj.length) throw new PyError("IndexError", "list index out of range");
      return obj[i];
    }
    if (obj && obj.__pycfgmap) return { __pycls: mapCls(key) };
    if (isDict(obj)) {
      const k = isStr(key) ? key : pyStr(key);
      if (!hasOwn(obj, k)) throw new PyError("KeyError", pyRepr(key));
      return obj[k];
    }
    throw new PyError("TypeError", `'${pyType(obj)}' object is not subscriptable`);
  }
  function comprehension(node, env, build) {
    const out = [];
    const gens = node.generators;
    const loop = (gi, scope) => {
      if (gi === gens.length) { out.push(build(scope)); return; }
      const g = gens[gi];
      for (const item of pyIter(evalExpr(g.iter, scope))) {
        const sc = { vars: Object.assign(Object.create(null), scope.vars), self: scope.self, K: scope.K };
        assignTarget(g.target, item, sc);
        if (g.ifs.every(c => truthy(evalExpr(c, sc)))) loop(gi + 1, sc);
      }
    };
    loop(0, env);
    return out;
  }
  function evalExpr(n, env) {
    switch (n._) {
      case "Constant": return n.value && n.value.__repr__ !== undefined ? (() => { throw new Unsupported("constant " + n.value.__repr__); })() : n.value;
      case "Name": return lookupName(n.id, env);
      case "Attribute": return getAttr(evalExpr(n.value, env), n.attr);
      case "Subscript": {
        const obj = evalExpr(n.value, env);
        const s = n.slice;
        const key = s._ === "Slice" ? { __slice: [s.lower ? numVal(evalExpr(s.lower, env)) : null, s.upper ? numVal(evalExpr(s.upper, env)) : null,
          s.step ? numVal(evalExpr(s.step, env)) : null] } : evalExpr(s, env);
        return subscript(obj, key);
      }
      case "Call": {
        const fn = evalExpr(n.func, env);
        const args = [];
        for (const a of n.args) { if (a._ === "Starred") args.push(...pyIter(evalExpr(a.value, env))); else args.push(evalExpr(a, env)); }
        let kw = null;
        for (const k of n.keywords) {
          kw = kw || {};
          if (k.arg === null) Object.assign(kw, evalExpr(k.value, env)); else kw[k.arg] = evalExpr(k.value, env);
        }
        return callValue(fn, args, kw);
      }
      case "BinOp": return binop(n.op._, evalExpr(n.left, env), evalExpr(n.right, env));
      case "UnaryOp": {
        const v = evalExpr(n.operand, env);
        if (n.op._ === "Not") return !truthy(v);
        if (n.op._ === "USub") { if (!isNumLike(v)) throw new PyError("TypeError", "bad operand type for unary -"); return -numVal(v); }
        if (n.op._ === "UAdd") return numVal(v);
        throw new Unsupported("unary " + n.op._);
      }
      case "BoolOp": {
        let v = null;
        for (const x of n.values) {
          v = evalExpr(x, env);
          if (n.op._ === "And" ? !truthy(v) : truthy(v)) return v;
        }
        return v;
      }
      case "Compare": {
        let left = evalExpr(n.left, env);
        for (let i = 0; i < n.ops.length; i++) {
          const right = evalExpr(n.comparators[i], env);
          if (!compare(n.ops[i]._, left, right)) return false;
          left = right;
        }
        return true;
      }
      case "IfExp": return truthy(evalExpr(n.test, env)) ? evalExpr(n.body, env) : evalExpr(n.orelse, env);
      case "List": case "Tuple": {
        const out = [];
        for (const e of n.elts) { if (e._ === "Starred") out.push(...pyIter(evalExpr(e.value, env))); else out.push(evalExpr(e, env)); }
        return out;
      }
      case "Set": return new Set(n.elts.map(e => evalExpr(e, env)));
      case "Dict": {
        const o = {};
        n.keys.forEach((k, i) => { if (k === null) Object.assign(o, evalExpr(n.values[i], env)); else o[pyStr(evalExpr(k, env))] = evalExpr(n.values[i], env); });
        return o;
      }
      case "ListComp": case "GeneratorExp": return comprehension(n, env, sc => evalExpr(n.elt, sc));
      case "SetComp": return new Set(comprehension(n, env, sc => evalExpr(n.elt, sc)));
      case "DictComp": { const o = {}; for (const [k, v] of comprehension(n, env, sc => [evalExpr(n.key, sc), evalExpr(n.value, sc)])) o[pyStr(k)] = v; return o; }
      case "JoinedStr": return n.values.map(v => (v._ === "Constant" ? v.value : (() => { try { return pyStr(evalExpr(v.value, env)); } catch (e) { return "?"; } })())).join("");
      case "FormattedValue": return pyStr(evalExpr(n.value, env));
      case "NamedExpr": { const v = evalExpr(n.value, env); assignTarget(n.target, v, env); return v; }
      case "Lambda": return { __pyfn: (a) => { const sc = { vars: Object.assign(Object.create(null), env.vars), self: env.self, K: env.K };
        n.args.args.forEach((x, i) => { sc.vars[x.arg] = a[i]; }); return evalExpr(n.body, sc); } };
    }
    throw new Unsupported("expr " + n._);
  }
  function assignTarget(t, v, env) {
    if (t._ === "Name") { env.vars[t.id] = v; return; }
    if (t._ === "Attribute") { setAttrTarget(evalExpr(t.value, env), t.attr, v); return; }
    if (t._ === "Subscript") {
      const obj = evalExpr(t.value, env), key = evalExpr(t.slice, env);
      if (isList(obj)) { let i = numVal(key); if (i < 0) i += obj.length; if (i < 0 || i >= obj.length) throw new PyError("IndexError", "list assignment index out of range"); obj[i] = v; return; }
      if (isDict(obj)) { obj[isStr(key) ? key : pyStr(key)] = v; return; }
      throw new PyError("TypeError", "object does not support item assignment");
    }
    if (t._ === "Tuple" || t._ === "List") {
      const items = pyIter(v);
      if (items.length !== t.elts.length) throw new PyError("ValueError", "unpack mismatch");
      t.elts.forEach((e, i) => assignTarget(e, items[i], env));
      return;
    }
    throw new Unsupported("target " + t._);
  }
  function execBlock(body, env) { for (const st of body) execStmt(st, env); }
  function execStmt(st, env) {
    switch (st._) {
      case "Expr": evalExpr(st.value, env); return;
      case "Assign": { const v = evalExpr(st.value, env); for (const t of st.targets) assignTarget(t, v, env); return; }
      case "AnnAssign": if (st.value) assignTarget(st.target, evalExpr(st.value, env), env); return;
      case "AugAssign": {
        const cur = evalExpr(st.target, env);
        const v = binop(st.op._, cur, evalExpr(st.value, env));
        if (isList(cur) && st.op._ === "Add") { cur.splice(0, cur.length, ...v); return; }
        assignTarget(st.target, v, env);
        return;
      }
      case "If": if (truthy(evalExpr(st.test, env))) execBlock(st.body, env); else execBlock(st.orelse, env); return;
      case "For": {
        for (const item of pyIter(evalExpr(st.iter, env))) { assignTarget(st.target, item, env); execBlock(st.body, env); }
        execBlock(st.orelse || [], env);
        return;
      }
      case "Raise": {
        const e = st.exc ? evalExpr(st.exc, env) : null;
        if (e && e.__pyexcval) throw new PyError(e.__pyexcval, e.msg);
        if (e && e.__pyexc) throw new PyError(e.__pyexc, "");
        throw new PyError("RuntimeError", "raise");
      }
      case "Assert": if (!truthy(evalExpr(st.test, env))) throw new PyError("AssertionError", st.msg ? pyStr(evalExpr(st.msg, env)) : ""); return;
      case "Return": throw new ReturnSignal(st.value ? evalExpr(st.value, env) : null);
      case "Pass": return;
      case "FunctionDef": return;                                        // decorated property definitions inside __init__
      case "Delete": for (const t of st.targets) { if (t._ === "Subscript") { const o = evalExpr(t.value, env); delete o[pyStr(evalExpr(t.slice, env))]; } } return;
    }
    throw new Unsupported("stmt " + st._);
  }
  // run one __init__ statement with the interpreter: locals L, kwargs K, self c
  function interpret(idx, X) {
    const node = TF.stmts_ast[idx];
    if (!node) throw new Unsupported("no ast");
    const env = { vars: X.L, self: X.c, K: X.K };
    execStmt(node, env);
  }

  // JSON.parse for a config.json as Python writes it: bare NaN / Infinity / -Infinity (outside strings) are numbers
  function parseConfig(text) {
    const SENT = "\u0000pyfloat:";
    let out = "", i = 0, inStr = false;
    while (i < text.length) {
      const c = text[i];
      if (inStr) {
        out += c;
        if (c === "\\") { out += text[i + 1] || ""; i += 2; continue; }
        if (c === '"') inStr = false;
        i += 1;
        continue;
      }
      if (c === '"') { inStr = true; out += c; i += 1; continue; }
      let m = null;
      for (const tok of ["-Infinity", "Infinity", "NaN"]) if (text.startsWith(tok, i)) { m = tok; break; }
      if (m) { out += JSON.stringify(SENT + m); i += m.length; continue; }
      out += c; i += 1;
    }
    return JSON.parse(out, (k, v) => (typeof v === "string" && v.startsWith(SENT)) ?
      ({ "-Infinity": -Infinity, Infinity: Infinity, NaN: NaN })[v.slice(SENT.length)] : v);
  }

  const API = {
    setTables, metrics, parseConfig, plan: (raw, tables, opts) => { if (tables) setTables(tables); return plan(raw, opts); },
    classify: (raw, tables) => { if (tables) setTables(tables); return classify(raw); },
    dimsOf: (raw, tables) => { if (tables) setTables(tables); return dimsOf(raw); },
    resolve: (raw, tables) => { if (tables) setTables(tables); return resolve(raw); },
    stock_arch: (raw, tables) => { if (tables) setTables(tables); return stock_arch(raw); },
    _internal: { pyRound, pyRoundN, divExact, floordiv, pymod, pyInt, truthy, PyError, _kind, _num,
      setResolver: f => { RESOLVE = f; }, debug: DEBUG, unhandled: () => [...UNHANDLED.entries()] },
  };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.ArchMetrics = API;
})(typeof window !== "undefined" ? window : globalThis);
