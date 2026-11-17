// The in-browser port of graph/arch_breakdown.py: the detailed breakdown of a plan (graph/templates/arch_metrics.js
// plan(), or the plan the page ships per shape). Same numbers as the Python, field by field (tests/test_arch_breakdown_js.py
// runs both on every snapshot config): integers exactly (BigInt where a product passes 2^53), floats from the same
// operations in the same order (Python int / int is exactly rounded: ArchMetrics._internal.divExact).
// Tables: setTables({am: arch_metrics_js tables, quant: graph/data/llama_cpp_quant.json, hardware: [...],
// calib: graph/data/gguf_calibration.json}); needs ArchMetrics and GgufQuant loaded first.
"use strict";
(function (root) {
  const AMX = (typeof module !== "undefined" && module.exports) ? require("./arch_metrics.js") : root.ArchMetrics;
  const GQ = (typeof module !== "undefined" && module.exports) ? require("./gguf_quant.js") : root.GgufQuant;
  const I = AMX._internal;
  let T = null;
  function setTables(t) {
    T = t;
    if (t.am) AMX.setTables(t.am);
    if (t.quant) GQ.setTables(t.quant);
  }
  const CONTEXTS = { "1k": 1024, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072 };
  const BATCHES = [1, 8, 32];
  const ROOFLINE_L = ["4k", "32k", "max"];
  const KV_FORMATS = { bf16: [2, 1], fp8: [1, 1], q8_0: [34, 32], q4_0: [18, 32] };
  const WEIGHT_FORMATS = ["fp32", "bf16", "fp8", "int8", "int4", "nf4", "nvfp4"];
  const GGUF_LEVELS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];
  const big = BigInt;
  const has = (o, k) => o !== null && o !== undefined && Object.prototype.hasOwnProperty.call(o, k);
  const get = (o, k, d) => (has(o, k) ? o[k] : d);
  const fdiv = (a, b) => Math.floor(a / b);                       // // on exact integers (< 2^53, b > 0)
  const ceil = (a, b) => -fdiv(-a, b);
  const isNum = v => typeof v === "number";
  const truthy = v => !(v === null || v === undefined || v === false || v === 0 || v === "" ||
    (Array.isArray(v) && v.length === 0) || (typeof v === "object" && !Array.isArray(v) && Object.keys(v).length === 0));
  const sum = a => a.reduce((x, y) => x + y, 0);
  const add = (d, k, v) => { d[k] = (has(d, k) ? d[k] : 0) + v; };

  // ------------------------------------------------------------------------------------------ tensors per group
  function _t(role, kind, n_in, n_out, experts = 1, elems = null, active = null) {
    const e = elems !== null ? elems : n_in * n_out * experts;
    return { role, kind, in: n_in, out: n_out, experts, elems: e, active: active === null ? e : active };
  }
  const _lump = (role, elems, h) => _t(role, "lump", h, h ? fdiv(elems, h) : 0, 1, elems);
  const _vec = (role, elems) => _t(role, "vector", null, null, 1, elems);

  function attn_tensors(a, mla, h, k_eq_v = false) {
    const out = [];
    if (a.W_override !== undefined && a.W_override !== null) {
      out.push(_lump("attn.block", a.W_override, h));
      const b = get(a, "b_override", 0);
      if (b) out.push(_vec("attn.misc", b));
      return out;
    }
    const nh = a.nh, nkv = a.nkv, dqk = a.dqk, dv = a.dv;
    const extra_W = get(a, "extra_W", 0) || 0, other = (get(a, "extra_b", 0) || 0) + (get(a, "sinks", 0) || 0);
    if (mla) {
      if (mla.q_lora) {
        out.push(_t("attn.q_a", "linear", h, mla.q_lora));
        out.push(_t("attn.q_b", "linear", mla.q_lora, nh * dqk));
        out.push(_vec("attn.lora_norm", mla.q_lora));
      } else out.push(_t("attn.q", "linear", h, nh * dqk));
      out.push(_t("attn.kv_a", "linear", h, mla.kv_lora + mla.rope));
      out.push(_t("attn.kv_b", "linear", mla.kv_lora, nh * (mla.nope + dv)));
      out.push(_vec("attn.lora_norm", mla.kv_lora));
      out.push(_t("attn.o", "linear", nh * dv, h));
      if (extra_W) out.push(_lump("attn.extra", extra_W, h));
      if (a.o_bias) out.push(_vec("attn.bias", h));
      if (other) out.push(_vec("attn.misc", other));
      return out;
    }
    const moa = get(a, "moa", null);
    if (moa) {
      const E = moa.E, k = moa.k, D = moa.D;
      out.push(_t("attn.experts", "linear", h, 2 * D, E, E * 2 * h * D, k * 2 * h * D));
      out.push(_t("attn.kv_proj", "linear", h, 2 * D));
      out.push(_t("attn.router", "linear", h, E));
      out.push(_vec("attn.bias", h));
      if (other) out.push(_vec("attn.misc", other));
      return out;
    }
    const xin = get(a, "in_dim", null) || h;
    out.push(_t("attn.q", "linear", xin, nh * dqk));
    if (!get(a, "no_kv_proj", false)) {
      out.push(_t("attn.k", "linear", xin, nkv * dqk));
      if (!k_eq_v) out.push(_t("attn.v", "linear", xin, nkv * dv));
    }
    out.push(_t("attn.o", "linear", nh * dv, h));
    if (get(a, "out_gate", false)) out.push(_t("attn.gate", "linear", h, get(a, "gate_headwise", false) ? nh : nh * dv));
    if (extra_W) { const e = _lump("attn.extra", extra_W, h); e.active = extra_W - get(a, "W_inactive", 0); out.push(e); }
    if (a.qk_norm) out.push(_vec("attn.qk_norm", a.qk_norm));
    let b = 0;
    if (a.qkv_bias) b += nh * dqk + (get(a, "no_kv_proj", false) ? 0 : nkv * dqk + (k_eq_v ? 0 : nkv * dv));
    if (a.o_bias) b += h;
    if (b) out.push(_vec("attn.bias", b));
    if (other) out.push(_vec("attn.misc", other));
    return out;
  }
  function ffn_tensors(f, h) {
    if (f === null || f === undefined) return [];
    if (get(f, "linear", false)) return [_t("mlp.linear", "linear", h, h)];
    if (get(f, "rwkv", false)) return [_t("mlp.up", "linear", h, f.inter), _t("mlp.receptance", "linear", h, h),
      _t("mlp.down", "linear", f.inter, h), _vec("mlp.misc", 2 * h)];
    const out = [];
    if (f.glu) out.push(_t("mlp.gate", "linear", h, f.inter));
    out.push(_t("mlp.up", "linear", h, f.inter), _t("mlp.down", "linear", f.inter, h));
    if (f.bias) out.push(_vec("mlp.bias", f.glu ? (2 * f.inter + h) : (f.inter + h)));
    return out;
  }
  function moe_tensors(m, h) {
    if (m === null || m === undefined) return [];
    const xin = get(m, "latent", null) || h;
    const E = m.E, k = m.k, ew = m.ew;
    const glu = get(m, "glu", true);
    const out = [];
    const parts = glu ? [["gate", xin, ew], ["up", xin, ew], ["down", ew, xin]] : [["up", xin, ew], ["down", ew, xin]];
    for (const [nm, a, b] of parts) out.push(_t("moe.routed." + nm, "linear", a, b, E, E * a * b, k * a * b));
    const s = m.shared;
    if (s) {
      const sglu = get(m, "shared_glu", glu);
      const sp = sglu ? [["gate", h, s], ["up", h, s], ["down", s, h]] : [["up", h, s], ["down", s, h]];
      for (const [nm, a, b] of sp) out.push(_t("moe.shared." + nm, "linear", a, b));
    }
    if (get(m, "latent", null)) out.push(_t("moe.latent", "linear", h, m.latent), _t("moe.latent", "linear", m.latent, h));
    if (!get(m, "no_router", false)) out.push(_t("moe.router", "linear", h, E + (get(m, "zero", 0) || 0)));
    if (m.shared_gate) out.push(_t("moe.shared_gate", "linear", h, 1));
    const b = (m.router_bias ? E : 0) + (m.expert_bias ? E * (2 * ew + h) : 0) + (get(m, "out_bias", false) ? h : 0);
    if (b) out.push(_vec("moe.bias", b));
    return out;
  }
  function group_tensors(g, h) {
    let out = [];
    if (g.attn !== null) out = out.concat(attn_tensors(g.attn, g.mla, h, g.k_eq_v));
    if (g.ffn !== null) out = out.concat(ffn_tensors(g.ffn, h));
    if (g.moe !== null) out = out.concat(moe_tensors(g.moe, h));
    if (g.rec !== null) {
      if (get(g.rec, "W", 0)) out.push(_lump("rec.linear", g.rec.W, h));
      if (get(g.rec, "b", 0)) out.push(_vec("rec.misc", g.rec.b));
    }
    if (g.norm) out.push(_vec("norm", g.norm));
    return out;
  }
  function global_tensors(p) {
    const h = p.hidden, V = p.vocab;
    const emb = p.embedding;
    const tok_cols = (V && p.lm_head_W && p.lm_head_W % V === 0) ? fdiv(p.lm_head_W, V) : h;
    const tok = Math.min(emb, V * tok_cols);
    const out = [_t("embedding", "embed", tok_cols, V, 1, tok)];
    if (emb - tok) out.push(_t("embedding.other", "embed", null, null, 1, emb - tok));
    if (p.lm_head) out.push(_t("lm_head", "linear", tok_cols, tok_cols ? fdiv(p.lm_head, tok_cols) : 0, 1, p.lm_head));
    if (p.final_norm) out.push(_vec("final_norm", p.final_norm));
    if (p.extra) out.push(_vec("other", p.extra));
    if (p.extra_W) out.push(_lump("other.linear", p.extra_W, h));
    return out;
  }

  // ------------------------------------------------------------------------------------------ labels and order
  function group_label(g) {
    const k = g.kind;
    let mix;
    if (g.attn !== null) {
      mix = ({ full: "global", sliding: "sliding", shared: "KV-shared" })[k] || k;
      if (g.mla) mix += " MLA";
      if (g.k_eq_v) mix += " K=V";
      if (get(g, "dsa_index", null) === false) mix += " (no indexer)";
      if (g.rec !== null && get(g.rec, "family", null) !== "glue") mix += " + " + String(get(g.rec, "family", null));
    } else if (g.rec !== null) mix = get(g.rec, "family", null) === "glue" ? "glue" : String(get(g.rec, "family", null));
    else mix = "no mixer";
    const ffn = g.moe !== null ? "MoE" : (g.ffn !== null ? "FFN" : "");
    return mix + (ffn ? " · " + ffn : "");
  }
  function true_kinds(d, n) {
    const _kind = I._kind, _num = I._num;
    const g = k => _num(get(d, k, null));
    const mt = String(get(d, "model_type", "") || "").toLowerCase();
    const isList = Array.isArray;
    for (const key of ["layers_block_type", "layer_type"]) {
      const v = get(d, key, null);
      if (isList(v) && v.length === n && v.every(x => typeof x === "string")) return v.map(_kind);
    }
    for (const [key, soft] of [["attn_type_list", 1], ["linear_attention_pattern", 0]]) {
      const v = get(d, key, null);
      if (isList(v) && v.length === n) return v.map(x => (x === soft ? "full" : "rec"));
    }
    const hlp = get(d, "hybrid_layer_pattern", null);
    if (isList(hlp) && hlp.length === n && hlp.every(x => x === 0 || x === 1)) return hlp.map(x => (x === 0 ? "full" : "local"));
    const bt = get(d, "_block_types", null) || get(d, "block_types", null);
    if (isList(bt) && bt.length && n) {
      const kinds = Array.from({ length: n }, (_, i) => _kind(bt[i % bt.length]));
      return kinds.map(k => (k === "full" && truthy(get(d, "attention_window_size", null)) ? "local" : k));
    }
    const lac = get(d, "linear_attn_config", null);
    if (lac && typeof lac === "object" && !isList(lac) && isList(get(lac, "full_attn_layers", null))) {
      const idx = new Set(lac.full_attn_layers.filter(i => Number.isInteger(i) && typeof i === "number").map(i => i - 1));
      return Array.from({ length: n }, (_, i) => (idx.has(i) ? "full" : "rec"));
    }
    for (const key of ["full_attn_idxs", "attn_layer_indices", "full_attention_layers"]) {
      if (isList(get(d, key, null))) {
        const idx = new Set(d[key].filter(i => Number.isInteger(i)));
        return Array.from({ length: n }, (_, i) => (idx.has(i) ? "full" : "rec"));
      }
    }
    if (g("attn_layer_period")) {
      const per = g("attn_layer_period"), off = g("attn_layer_offset");
      return Array.from({ length: n }, (_, i) => (I.pymod(i, per) === off ? "full" : "rec"));
    }
    if (mt === "gemma2") return Array.from({ length: n }, (_, i) => (i % 2 === 0 ? "local" : "full"));
    if (g("sliding_window_pattern")) {
      const q = g("sliding_window_pattern");
      return Array.from({ length: n }, (_, i) => (I.pymod(i + 1, q) === 0 ? "full" : "local"));
    }
    if (["qwen2", "qwen3", "qwen2_moe", "qwen3_moe"].includes(mt) && get(d, "max_window_layers", null) !== null) {
      return Array.from({ length: n }, (_, i) => (i >= g("max_window_layers") ? "local" : "full"));
    }
    return null;
  }
  const sortedJSON = o => JSON.stringify(o, (k, v) => (v && typeof v === "object" && !Array.isArray(v)) ?
    Object.keys(v).sort().reduce((a, kk) => { a[kk] = v[kk]; return a; }, {}) : v);
  function fix_order(p, d) {
    if (get(p, "order_exact", true) || d === null || d === undefined) return p;
    const n = p.layers;
    const tk = true_kinds(d, n);
    const rows = layer_order(Object.assign({}, p, { order_exact: true }));
    if (tk === null || rows === null || tk.length !== n || rows.some(r => r.length !== 1)) return p;
    const G = p.groups;
    const att = ["full", "sliding"].filter(k => G.some(g => g.kind === k));
    const vocab = { rec: "rec", full: att.includes("full") ? "full" : (att.length ? att[0] : null),
      local: att.includes("sliding") ? "sliding" : (att.length ? att[0] : null) };
    const want = tk.map((k, i) => [has(vocab, k) ? vocab[k] : null, G[rows[i][0]].moe !== null]);
    const key = g => [g.kind, g.moe !== null];
    const ks = x => JSON.stringify(x);
    const keys = G.map(g => ks(key(g)));
    if (new Set(keys).size !== keys.length) return p;
    const cw = {};
    for (const w of want) add(cw, ks(w), 1);
    const cg = {};
    for (const g of G) cg[ks(key(g))] = g.n;
    const same = Object.keys(cw).length === Object.keys(cg).length && Object.keys(cw).every(k => cg[k] === cw[k]);
    if (same) {
      const G2 = G.map(g => Object.assign({}, g, { at: want.map((w, i) => [w, i]).filter(([w]) => ks(w) === ks(key(g))).map(([, i]) => i) }));
      return Object.assign({}, p, { groups: G2, order_exact: true });
    }
    const MIX = ["kind", "window", "attn", "mla", "rec", "k_eq_v", "dsa_index"], FF = ["ffn", "moe"];
    const sig = (g, kk) => sortedJSON(kk.reduce((o, k) => { o[k] = has(g, k) ? g[k] : null; return o; }, {}));
    const mix = {}, ff = {};
    for (const g of G) {
      const fk = String(g.moe !== null);
      if (!has(mix, g.kind)) mix[g.kind] = sig(g, MIX);
      if (!has(ff, fk)) ff[fk] = sig(g, FF);
      if (g.np !== g.n || mix[g.kind] !== sig(g, MIX) || ff[fk] !== sig(g, FF) || g.norm !== G[0].norm) return p;
    }
    const tm = {}, tf = {};
    for (const g of G) { tm[g.kind] = g; tf[String(g.moe !== null)] = g; }
    if (want.some(([k, m]) => !has(tm, k) || !has(tf, String(m)))) return p;
    const G2 = [], seen = new Set();
    for (const w of want) {
      const s = ks(w);
      if (seen.has(s)) continue;
      seen.add(s);
      const at = want.map((x, i) => [x, i]).filter(([x]) => ks(x) === s).map(([, i]) => i);
      G2.push(Object.assign({}, tm[w[0]], { ffn: tf[String(w[1])].ffn, moe: tf[String(w[1])].moe, n: at.length, np: at.length, at }));
    }
    return Object.assign({}, p, { groups: G2, order_exact: true, repaired: true });
  }
  function checkpoint_quant(raw) {
    const q = get(raw, "quantization_config", null) || get(get(raw, "text_config", null) || {}, "quantization_config", null);
    return (q && typeof q === "object" && !Array.isArray(q) && truthy(get(q, "quant_method", null))) ? String(q.quant_method).toLowerCase() : null;
  }
  function gnum(d, keys) {                                   // arch_metrics._g: the first numeric (non-bool) value
    for (const k of keys) { const v = get(d, k, null); if (typeof v === "number") return v; }
    return 0;
  }
  function n_mtp(raw, p) {
    const tc = get(raw, "text_config", null);
    const t = (tc && typeof tc === "object" && !Array.isArray(tc)) ? tc : raw;
    let n = gnum(t, ["num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers"]);
    if (!n && p.mt === "qwen3_next") n = 1;
    return I.pyInt(n || 0);
  }
  function plan_ordered(raw) {
    let p = AMX.plan(raw);
    if (!get(p, "order_exact", true)) p = fix_order(p, AMX.resolve(raw)[0]);
    p.ckpt_quant = checkpoint_quant(raw);
    p.n_mtp = n_mtp(raw, p);
    p.norm_kind = get(AMX.classify(raw) || {}, "norm", "RMSNorm");
    return p;
  }
  function layer_order(p) {
    if (!get(p, "order_exact", true)) return null;
    const n = p.layers;
    const rows = Array.from({ length: n }, () => []);
    for (let gi = 0; gi < p.groups.length; gi++) {
      const g = p.groups[gi], at = get(g, "at", null);
      if (!Array.isArray(at) || at.length !== g.n || at.some(i => !Number.isInteger(i) || i < 0 || i >= n)) return null;
      for (const i of at) rows[i].push(gi);
    }
    if (rows.some(r => !r.length)) return null;
    return rows;
  }

  // ------------------------------------------------------------------------------------------ parameters
  function pyMaxBy(arr, keyf) {                               // Python max(key=...): the first maximal element
    let best = null, bk = null;
    for (const x of arr) {
      const k = keyf(x);
      if (best === null || cmpTuple(k, bk) > 0) { best = x; bk = k; }
    }
    return best;
  }
  function cmpTuple(a, b) {
    for (let i = 0; i < a.length; i++) { const x = +a[i], y = +b[i]; if (x !== y) return x < y ? -1 : 1; }
    return 0;
  }
  const mainLayer = p => pyMaxBy(p.groups, g => [g.moe !== null, g.attn !== null && g.kind === "full", get(g, "dsa_index", true)]);
  function params_breakdown(p) {
    const h = p.hidden;
    const comp = {}, act = {};
    for (const t of global_tensors(p)) { add(comp, t.role, t.elems); add(act, t.role, t.active); }
    const groups = [], by_kind = {};
    for (const g of p.groups) {
      const ts = group_tensors(g, h);
      const per = {}, per_act = {};
      for (const t of ts) { add(per, t.role, t.elems); add(per_act, t.role, t.active); }
      for (const [r, v] of Object.entries(per)) add(comp, r, g.np * v);
      for (const [r, v] of Object.entries(per_act)) add(act, r, g.np * v);
      const lt = sum(Object.values(per));
      const lab = group_label(g);
      const a = g.attn;
      const mm = sum(ts.filter(t => t.kind === "linear" || t.kind === "lump").map(t => 2 * t.active));
      const pa = sum(Object.values(per_act));
      groups.push({ label: lab, kind: g.kind, n: g.n, np: g.np, window: g.window,
        heads: a ? a.nh : 0, kv_heads: a ? (!g.mla ? a.nkv : 0) : 0, head_dim: a ? a.dqk : 0, v_head_dim: a ? a.dv : 0,
        experts: g.moe !== null ? [g.moe.E, g.moe.k] : null, params: per, layer_total: lt, active_layer_total: pa,
        kv_per_token: kv_group_per_token(p, g), flops_matmul: mm, attn_per_key: a ? 2 * a.nh * (a.dqk + a.dv) : 0,
        rec_core: g.rec ? get(g.rec, "core_flops", 0) || 0 : 0 });
      if (!has(by_kind, lab)) by_kind[lab] = { layers: 0, distinct: 0, params: 0, active: 0 };
      const bk = by_kind[lab];
      bk.layers += g.n; bk.distinct += g.np; bk.params += g.np * lt; bk.active += g.np * pa;
    }
    const out = { components: comp, active: act, total: sum(Object.values(comp)), active_total: sum(Object.values(act)),
      groups, by_kind, tied: !!p.tied };
    const nm = get(p, "n_mtp", 0) || 0;
    if (nm && p.groups.length) {
      const L1 = sum(group_tensors(mainLayer(p), h).map(t => t.elems));
      out.mtp = { layers: nm, params: nm * (L1 + 2 * h * h + 3 * h) };
    }
    return out;
  }

  // ------------------------------------------------------------------------------------------ FLOPs
  function span_sum(L, W) {                                  // BigInt sum_{i=1..L} min(i, W)
    const Lb = big(L);
    if (W === null || W === undefined || L <= W) return Lb * (Lb + 1n) / 2n;
    const Wb = big(W);
    return Wb * (Wb + 1n) / 2n + (Lb - Wb) * Wb;
  }
  function _attn_group(p, g, L, mode) {                      // decode: Numbers; prefill: BigInt totals
    const a = g.attn;
    if (a === null) return mode === "decode" ? [0, 0, 0] : [0n, 0n, 0n];
    const W = g.window;
    const sc = 2 * a.nh * a.dqk, av = 2 * a.nh * a.dv;
    const dsa = g.kind !== "shared" ? get(p, "dsa", null) : null;
    if (dsa) {
      const idx = get(g, "dsa_index", true) ? 2 * dsa.heads * dsa.dim : 0;
      if (mode === "decode") { const s = Math.min(L, dsa.topk); return [sc * s, av * s, idx * L]; }
      const S = span_sum(L, dsa.topk);
      return [big(sc) * S, big(av) * S, big(idx) * span_sum(L, null)];
    }
    if (mode === "decode") { const s = W ? Math.min(L, W) : L; return [sc * s, av * s, 0]; }
    const S = span_sum(L, W);
    return [big(sc) * S, big(av) * S, 0n];
  }
  function matmul_components(p) {
    const h = p.hidden, out = {};
    for (const g of p.groups) for (const t of group_tensors(g, h)) if (t.kind === "linear" || t.kind === "lump") add(out, t.role, 2 * g.n * t.active);
    out.lm_head = 2 * p.lm_head_W;
    if (p.extra_W) out["other.linear"] = 2 * p.extra_W;
    return out;
  }
  const rec_core = p => sum(p.groups.filter(g => g.rec).map(g => g.n * (get(g.rec, "core_flops", 0) || 0)));
  function flops_at(p, L) {
    const mm = matmul_components(p);
    const mm_tot = sum(Object.values(mm));
    const core = rec_core(p);
    const dec = Object.assign({}, mm);
    const kinds_dec = {}, kinds_pre = {};
    const att_d = [0, 0, 0], att_p = [0n, 0n, 0n];
    for (const g of p.groups) {
      if (g.attn === null) continue;
      const lab = group_label(g);
      const d3 = _attn_group(p, g, L, "decode"), p3 = _attn_group(p, g, L, "prefill");
      for (let j = 0; j < 3; j++) { att_d[j] += g.n * d3[j]; att_p[j] += big(g.n) * p3[j]; }
      add(kinds_dec, lab, g.n * sum(d3));
      kinds_pre[lab] = (has(kinds_pre, lab) ? kinds_pre[lab] : 0n) + big(g.n) * (p3[0] + p3[1] + p3[2]);
    }
    dec["attn.scores"] = att_d[0]; dec["attn.av"] = att_d[1];
    if (att_d[2]) dec["attn.indexer"] = att_d[2];
    if (core) dec["rec.core"] = core;
    const dec_total = mm_tot + sum(att_d) + core;
    const pre_attn_total = att_p[0] + att_p[1] + att_p[2];
    const Lb = big(L);
    const total_L = Lb * (big(mm_tot) + big(core)) + pre_attn_total;
    const pre = Object.assign({}, mm);
    pre["attn.scores"] = I.divExact(att_p[0], Lb);
    pre["attn.av"] = I.divExact(att_p[1], Lb);
    if (att_p[2] !== 0n) pre["attn.indexer"] = I.divExact(att_p[2], Lb);
    if (core) pre["rec.core"] = core;
    const kp = {};
    for (const [k, v] of Object.entries(kinds_pre)) kp[k] = I.divExact(v, Lb);
    return { decode: { components: dec, attn_by_kind: kinds_dec, total: dec_total },
      prefill: { components: pre, attn_by_kind: kp, per_token: I.divExact(total_L, Lb), total: Number(total_L),
        attn_total: Number(pre_attn_total) },
      training_approx: Number(3n * total_L) };
  }
  const NORM_FLOPS = { RMSNorm: 4, LayerNorm: 7 };
  function elementwise_at(p, L, norm_kind) {
    const per_elem = has(NORM_FLOPS, norm_kind) ? NORM_FLOPS[norm_kind] : 4;
    const div = norm_kind === "LayerNorm" ? 2 : 1;
    let sm = 0, norms = 0, act = 0;
    for (const g of p.groups) {
      const a = g.attn;
      if (a !== null) {
        const W = g.window, dsa = g.kind !== "shared" ? get(p, "dsa", null) : null;
        const span = dsa ? Math.min(L, dsa.topk) : (W ? Math.min(L, W) : L);
        sm += g.n * 5 * a.nh * span;
      }
      norms += g.n * per_elem * fdiv(g.norm, div);
      const f = g.ffn;
      if (f !== null && !get(f, "linear", false)) act += g.n * (get(f, "glu", false) ? 5 : 4) * f.inter;
      const m = g.moe;
      if (m !== null) act += g.n * (get(m, "glu", true) ? 5 : 4) * (m.k * m.ew + m.shared);
    }
    norms += per_elem * fdiv(p.final_norm, div);
    return { softmax: sm, norms, activations: act, total: sm + norms + act };
  }

  // ------------------------------------------------------------------------------------------ KV and state
  const vnorm = () => new Set((T.am && T.am.mg && T.am.mg.VNORM_TYPES) || []);
  const _kv_tensors = (p, g) => ((g.k_eq_v && vnorm().has(String(p.mt).toLowerCase())) ? 1 : 2);
  function kv_group_per_token(p, g) {
    const a = g.attn;
    if (a === null || g.kind === "shared") return 0;
    let e;
    if (g.mla) e = g.mla.kv_lora + g.mla.rope;
    else if (g.k_eq_v) e = a.nkv * a.dqk * _kv_tensors(p, g);
    else e = a.nkv * (a.dqk + a.dv);
    if (truthy(get(p, "dsa_dim", null)) && get(g, "dsa_index", true)) e += p.dsa_dim;
    return e;
  }
  function _kv_group_at(p, g, L) {
    const e = kv_group_per_token(p, g);
    if (!e) return 0;
    const W = g.window, toks = W ? Math.min(L, W - 1) : L;
    if (truthy(get(p, "dsa_dim", null)) && get(g, "dsa_index", true)) return g.n * ((e - p.dsa_dim) * toks + p.dsa_dim * L);
    return g.n * e * toks;
  }
  const kv_elems = (p, L) => sum(p.groups.map(g => _kv_group_at(p, g, L)));
  function kv_bytes(elems, fmt) { const [b, per] = KV_FORMATS[fmt]; return per === 1 ? elems * b : elems * b / per; }
  function kv_by_kind(p, L) {
    const out = {};
    for (const g of p.groups) { if (!kv_group_per_token(p, g)) continue; add(out, group_label(g), _kv_group_at(p, g, L)); }
    return out;
  }
  const state_elems = p => sum(p.groups.filter(g => g.rec).map(g => g.n * ((get(g.rec, "conv_state", 0) || 0) + (get(g.rec, "ssm_state", 0) || 0))));

  // ------------------------------------------------------------------------------------------ weight formats
  const EXCL = ["embedding", "embedding.other", "lm_head", "moe.router", "attn.router"];
  const _is_quant_target = t => (t.kind === "linear" || t.kind === "lump") && !EXCL.includes(t.role) && !t.role.startsWith("other");
  function tensor_format_bytes(t, fmt) {
    const e = t.elems;
    if (fmt === "fp32") return 4 * e;
    if (fmt === "bf16" || !_is_quant_target(t)) return 2 * e;
    const n_in = t.in || 1, X = t.experts;
    const n_out = (n_in && X) ? fdiv(e, n_in * X) : 0;
    if (fmt === "fp8") return e + 4 * X * ceil(n_in, 128) * ceil(n_out, 128);
    if (fmt === "int8") return e + 2 * X * n_out;
    if (fmt === "int4") return fdiv(e, 2) + (e % 2) + fdiv(X * ceil(n_in, 128) * n_out * 5, 2);
    if (fmt === "nf4") return fdiv(e, 2) + (e % 2) + 4 * ceil(e, 64);
    if (fmt === "nvfp4") return fdiv(e, 2) + (e % 2) + ceil(e, 16) + 4 * X;
    throw new Error("weight format " + fmt);
  }
  function all_tensors(p) {
    const out = global_tensors(p).map(t => [t, 1]);
    for (const g of p.groups) for (const t of group_tensors(g, p.hidden)) out.push([t, g.np]);
    return out;
  }
  function weight_bytes(p, fmt, tensors) {
    tensors = tensors || all_tensors(p);
    const out = { total: 0, routed: 0, embedding: 0, lm_head: 0 };
    for (const [t, c] of tensors) {
      const b = c * tensor_format_bytes(t, fmt);
      out.total += b;
      if (t.role.startsWith("moe.routed") || t.role === "attn.experts") out.routed += b;
      if (t.role === "embedding" || t.role === "embedding.other") out.embedding += b;
      if (t.role === "lm_head") out.lm_head += b;
    }
    return out;
  }
  function routed_fraction(p) {
    for (const g of p.groups) {
      if (g.moe !== null && g.moe.E > 1) return [g.moe.E, g.moe.k];
      const a = g.attn;
      if (a !== null && get(a, "moa", null)) return [a.moa.E, a.moa.k];
    }
    return null;
  }
  const expert_coverage = (E, k, B) => (B === 1 ? k / E : 1 - Math.pow(1 - k / E, B));

  // ------------------------------------------------------------------------------------------ GGUF
  const LLAMA_ARCH = { llama: "llama", mistral: "llama", mixtral: "llama", internlm3: "llama", qwen2: "qwen2", qwen3: "qwen3",
    qwen3_moe: "qwen3moe", qwen2_moe: "qwen2moe", qwen3_next: "qwen3next", gemma: "gemma", gemma2: "gemma2",
    gemma3: "gemma3", gemma3_text: "gemma3", phi3: "phi3", phi: "phi2", phimoe: "phimoe",
    deepseek_v2: "deepseek2", deepseek_v3: "deepseek2", deepseek: "deepseek", olmo: "olmo", olmo2: "olmo2",
    olmoe: "olmoe", olmo3: "olmo2", granite: "granite", granitemoe: "granitemoe",
    granitemoehybrid: "granitehybrid", falcon: "falcon", falcon_h1: "falcon-h1", falcon_mamba: "mamba",
    mamba: "mamba", internlm2: "internlm2", glm4: "glm4", chatglm: "chatglm", glm4_moe: "glm4moe",
    exaone: "exaone", exaone4: "exaone4", cohere: "command-r", cohere2: "cohere2", starcoder2: "starcoder2",
    gpt_bigcode: "starcoder", gpt_oss: "gpt-oss", smollm3: "smollm3", stablelm: "stablelm", gpt2: "gpt2",
    gpt_neox: "gptneox", bloom: "bloom", lfm2: "lfm2", hunyuan_v1_dense: "hunyuan-dense", ernie4_5: "ernie4_5",
    ernie4_5_moe: "ernie4_5-moe", seed_oss: "seed_oss", apertus: "apertus", nemotron_h: "nemotron_h",
    nemotron: "nemotron", minimax_m2: "minimax-m2", bailing_moe: "bailingmoe", arcee: "arcee",
    baichuan: "baichuan", minicpm: "minicpm", jamba: "jamba", dots1: "dots1", "nemotron-nas": "deci",
    mistral3: "llama", llama4: "llama4" };
  const GGUF_FUSED_QKV = new Set(["gpt_neox", "falcon", "bloom", "persimmon", "chatglm", "gpt2", "gpt_bigcode", "qwen", "jais",
    "mpt", "dbrx", "phi3", "openelm", "plamo2", "bailing_moe"]);
  const GGUF_FUSED_GATE_UP = new Set(["phi3", "chatglm"]);
  const GGUF_MTP_ARCHS = new Set(["Glm4MoeForCausalLM", "Glm4vMoeForConditionalGeneration", "Glm4MoeLiteForCausalLM", "GlmMoeDsaForCausalLM",
    "Glm5NextForConditionalGeneration", "GlmOcrForConditionalGeneration", "Qwen3NextForCausalLM",
    "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5TextModel", "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM", "DeepseekV32ForCausalLM",
    "DeepseekV4ForCausalLM", "MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM", "Step3p5ForCausalLM",
    "Step3p7ForConditionalGeneration", "ExaoneMoEForCausalLM", "ExaoneMoeForCausalLM",
    "Exaone4_5_ForConditionalGeneration", "BailingMoeV2ForCausalLM", "MellumForCausalLM", "Cohere2MoeForCausalLM"]);
  const GGUF_NAME = { "attn.q": "attn_q", "attn.k": "attn_k", "attn.v": "attn_v", "attn.o": "attn_output",
    "attn.gate": "attn_gate", "attn.q_a": "attn_q_a", "attn.q_b": "attn_q_b", "attn.kv_a": "attn_kv_a_mqa",
    "attn.extra": "attn_extra", "attn.experts": "attn_exps", "attn.kv_proj": "attn_kv_proj",
    "attn.router": "attn_router", "attn.block": "attn_block", "mlp.gate": "ffn_gate", "mlp.up": "ffn_up",
    "mlp.down": "ffn_down", "mlp.linear": "ffn_up", "mlp.receptance": "channel_mix_receptance",
    "moe.routed.gate": "ffn_gate_exps", "moe.routed.up": "ffn_up_exps", "moe.routed.down": "ffn_down_exps",
    "moe.shared.gate": "ffn_gate_shexp", "moe.shared.up": "ffn_up_shexp", "moe.shared.down": "ffn_down_shexp",
    "moe.latent": "ffn_latent", "moe.router": "ffn_gate_inp", "moe.shared_gate": "ffn_gate_inp_shexp",
    "rec.linear": "ssm_in" };
  const GGUF_VEC = { "attn.bias": "attn_qkv.bias", "attn.qk_norm": "attn_q_norm.weight", "attn.lora_norm": "attn_q_a_norm.weight",
    "attn.misc": "attn_sinks.weight", "mlp.bias": "ffn_up.bias", "mlp.misc": "channel_mix_lerp.weight",
    "moe.bias": "exp_probs_b.bias", "rec.misc": "ssm_d.weight", norm: "attn_norm.weight" };
  const gguf_mtp_layers = p => (GGUF_MTP_ARCHS.has(p.arch) ? (get(p, "n_mtp", 0) || 0) : 0);
  function _layer_slots(g) {
    const at = get(g, "at", null);
    if (Array.isArray(at) && at.length >= g.np) return at.slice(0, g.np);
    return Array.from({ length: g.np }, (_, i) => i);
  }
  function gguf_tensors(p) {
    const mt = String(p.mt).toLowerCase(), h = p.hidden, out = [];
    const lin = (name, n_in, n_out, X = 1) => out.push({ name, shape: [n_in, n_out].concat(X > 1 ? [X] : []) });
    const lump = (name, e, base) => {
      if (h && fdiv(e, h)) lin(base + ".weight", h, fdiv(e, h));
      if (h && e % h) out.push({ name: base + "_rest.weight", shape: [e % h] });
    };
    for (const t of global_tensors(p)) {
      const r = t.role;
      if (r === "embedding") lin("token_embd.weight", t.in, t.out);
      else if (r === "lm_head") lin("output.weight", t.in, t.out);
      else if (r === "embedding.other") lump(r, t.elems, (mt.startsWith("gemma3n") || mt.startsWith("gemma4")) ? "per_layer_token_embd" : "position_embd");
      else if (r === "other.linear") lump("output_extra", t.elems, "output_extra");
      else if (r === "final_norm") out.push({ name: "output_norm.weight", shape: [t.elems] });
      else out.push({ name: "output_misc.bias", shape: [t.elems] });
    }
    const slots = p.groups.map(g => [g, _layer_slots(g)]);
    const nmtp = gguf_mtp_layers(p);
    if (nmtp) {
      slots.push([mainLayer(p), Array.from({ length: nmtp }, (_, i) => p.layers + i)]);
      for (let i = p.layers; i < p.layers + nmtp; i++) {
        lin(`blk.${i}.nextn.eh_proj.weight`, 2 * h, h);
        for (const n_ of ["enorm", "hnorm", "shared_head_norm"]) out.push({ name: `blk.${i}.nextn.${n_}.weight`, shape: [h] });
      }
    }
    for (const [g, at] of slots) {
      const ts = group_tensors(g, h);
      for (const i of at) {
        const pre = `blk.${i}.`;
        const qkv = GGUF_FUSED_QKV.has(mt) && ["attn.q", "attn.k"].every(r => ts.some(t => t.role === r));
        const gu = GGUF_FUSED_GATE_UP.has(mt) && ts.some(t => t.role === "mlp.gate");
        const fused = new Map();
        for (const t of ts) {
          const r = t.role;
          if (t.kind === "vector") { out.push({ name: pre + (has(GGUF_VEC, r) ? GGUF_VEC[r] : "misc.weight"), shape: [t.elems] }); continue; }
          if (qkv && ["attn.q", "attn.k", "attn.v"].includes(r)) { if (!fused.has("attn_qkv")) fused.set("attn_qkv", [t.in, 0]); fused.get("attn_qkv")[1] += t.out; continue; }
          if (gu && ["mlp.gate", "mlp.up"].includes(r)) { if (!fused.has("ffn_up")) fused.set("ffn_up", [t.in, 0]); fused.get("ffn_up")[1] += t.out; continue; }
          if (r === "attn.kv_b" && g.mla) {
            const m = g.mla, nh = g.attn.nh;
            lin(pre + "attn_k_b.weight", m.nope, m.kv_lora, nh);
            lin(pre + "attn_v_b.weight", m.kv_lora, m.v, nh);
            continue;
          }
          if (t.kind === "lump") { lump(r, t.elems, pre + (has(GGUF_NAME, r) ? GGUF_NAME[r] : "misc")); continue; }
          if (r === "moe.shared_gate") { out.push({ name: pre + "ffn_gate_inp_shexp.weight", shape: [t.in] }); continue; }
          let nm = has(GGUF_NAME, r) ? GGUF_NAME[r] : "misc";
          if (r === "moe.latent") nm = t.in === h ? "ffn_latent_in" : "ffn_latent_out";
          lin(pre + nm + ".weight", t.in, t.out, t.experts);
        }
        for (const [nm, [a, b]] of fused) lin(pre + nm + ".weight", a, b);
      }
    }
    return out;
  }
  function gguf_header(p, tensors) {
    const c = (T && T.calib) || { bytes_per_vocab: 28.0, fixed: 4096 };
    let infos = 0;
    for (const t of tensors) infos += 8 + t.name.length + 4 + 8 * t.shape.length + 4 + 8;
    return Math.trunc(c.fixed + infos + c.bytes_per_vocab * p.vocab);
  }
  function gguf_sizes(p, levels) {
    levels = levels || GGUF_LEVELS;
    const mt = String(p.mt).toLowerCase(), arch = has(LLAMA_ARCH, mt) ? LLAMA_ARCH[mt] : mt;
    if (get(p, "head_stored", false) && p.tied) p = Object.assign({}, p, { tied: false, lm_head: p.lm_head_W });
    let ts = gguf_tensors(p);
    const src = GQ.sourceTypes(ts, arch);
    ts = ts.map((t, i) => Object.assign({}, t, { src: src[i] }));
    let kept = 0;
    if (get(p, "ckpt_quant", null) === "mxfp4" || mt === "gpt_oss") {
      const align = T.quant.alignment;
      const nel = s => s.reduce((a, b) => a * b, 1);
      const ex = ts.filter(t => t.name.includes("_exps."));
      ts = ts.filter(t => !t.name.includes("_exps."));
      for (const t of ex) kept += Math.ceil(GQ.tensorBytes("MXFP4", nel(t.shape), t.shape[0]) / align) * align;
    }
    const a0 = (p.groups.find(g => g.attn !== null) || {}).attn || null;
    const gm = p.groups.find(g => g.moe !== null);
    const ctx = { n_expert: gm ? gm.moe.E : 0, n_layer: p.layers + gguf_mtp_layers(p) };
    if (a0) { ctx.n_head = a0.nh; ctx.n_head_kv = a0.nkv; }
    const head = gguf_header(p, ts) + kept;
    const out = {};
    for (const lv of levels) {
      const ftype = (lv === "BF16" || lv === "F16") ? null : lv;
      let b;
      if (ftype === null) {
        const tts = ts.map(t => Object.assign({}, t, { src: (lv === "F16" && t.src === "BF16") ? "F16" : t.src }));
        b = GQ.breakdown(tts, null, arch, head, ctx);
      } else b = GQ.breakdown(ts, ftype, arch, head, ctx);
      out[lv] = b.total;
    }
    return out;
  }

  // ------------------------------------------------------------------------------------------ activations
  function activation_bytes(p, L, attn = "flash", b = 1, a_bytes = 2) {
    const B = big(b), Lb = big(L), A = big(a_bytes), h = big(p.hidden);
    const base = 2n * B * Lb * h * A;
    let peak = 0n;
    const mx = (x, y) => (x > y ? x : y);
    for (const g of p.groups) {
      let at = 0n;
      const a = g.attn;
      if (a !== null) {
        const q = B * Lb * big(a.nh) * big(a.dqk);
        const kv = B * Lb * big(!g.mla ? a.nkv : a.nh) * big(a.dqk + a.dv);
        const o = B * Lb * big(a.nh) * big(a.dv);
        at = (2n * q + kv + o + B * Lb * h) * A;
        if (attn === "eager") at += B * big(a.nh) * Lb * Lb * 10n;
      }
      let ff = 0n;
      const f = g.ffn;
      if (f !== null && !get(f, "linear", false) && !get(f, "rwkv", false)) ff = big(f.glu ? 3 : 2) * B * Lb * big(f.inter) * A;
      const m = g.moe;
      if (m !== null) {
        const w = big(m.k * m.ew + m.shared);
        ff = mx(ff, 3n * B * Lb * w * A + B * Lb * big(m.E) * 4n + 8n * B * Lb * big(m.k) * big(m.E));
      }
      let rc = 0n;
      const r = g.rec;
      if (r !== null && get(r, "W", 0)) rc = 2n * B * Lb * big(p.hidden ? fdiv(r.W, p.hidden) : 0) * A;
      peak = mx(peak, mx(at, mx(ff, rc)));
    }
    const mask = attn === "eager" ? B * Lb * Lb * A : 0n;
    let rope = 0n;
    const g0 = p.groups.find(g => g.attn !== null);
    if (g0) rope = 2n * B * Lb * big(g0.attn.dqk) * A;
    const logits = p.hidden ? big(fdiv(b * p.lm_head_W, p.hidden) * 4) : 0n;
    return Number(base + peak + mask + rope + logits);
  }

  // ------------------------------------------------------------------------------------------ hardware and the record
  function hw_peak(hw, fmt) {
    const fl = hw.flops || {};
    let t = null;
    if (fmt === "fp8") t = has(fl, "fp8_dense") ? fl.fp8_dense : null;
    if (t === null || t === undefined) t = (fl.bf16_dense || fl.fp16_dense);
    return [hw.mem_bw_gbps * 1e9, (t || 0) * 1e12];
  }
  function decode_bound(bw, hw, fmt = "bf16") {
    const [bps, fps] = hw_peak(hw, fmt);
    let t = bw.bytes / bps;
    if (fps) t = Math.max(t, bw.flops / fps);
    return t ? bw.B / t : null;
  }
  function prefill_bound(total_flops, wbytes, hw, fmt = "bf16") {
    const [bps, fps] = hw_peak(hw, fmt);
    let t = wbytes / bps;
    if (fps) t = Math.max(t, total_flops / fps);
    return t;
  }
  function contexts_for(p) { const c = Object.assign({}, CONTEXTS); c.max = I.pyInt(p.max_context); return c; }
  function decode_read(p, L, B, wb, emb_row, kvfmt = "bf16", state_bytes = 2) {
    const ek = routed_fraction(p);
    const routed = wb.routed;
    const routed_read = (ek && routed) ? routed * expert_coverage(ek[0], ek[1], B) : routed;
    let head = p.lm_head ? wb.lm_head : (p.tied ? wb.embedding : 0);
    if (p.tied && wb.embedding) head = p.embedding ? p.lm_head_W * (wb.embedding / p.embedding) : 0;
    const w = wb.total - wb.embedding - wb.lm_head - routed + routed_read + head + B * emb_row;
    const kv = B * kv_bytes(kv_elems(p, Math.max(L - 1, 0)), kvfmt);
    const st = B * 2 * state_bytes * state_elems(p);
    return { weights: w, kv, state: st, bytes: w + kv + st };
  }
  function breakdown(p, hw_list) {
    if (!T) throw new Error("ArchBreakdown: tables not set");
    const h = p.hidden;
    const norm_kind = get(p, "norm_kind", null) || "RMSNorm";
    const prm = params_breakdown(p);
    const order = layer_order(p);
    const tensors = all_tensors(p);
    const wb = {};
    for (const f of WEIGHT_FORMATS) wb[f] = weight_bytes(p, f, tensors);
    const ctx = contexts_for(p);
    const st = state_elems(p);
    const out = { max_context: I.pyInt(p.max_context), hidden: h, vocab: p.vocab, layers: p.layers, layer_order: order !== null, params: prm };
    if (order !== null) out.per_layer = order;
    out.weights = {};
    for (const f of WEIGHT_FORMATS) out.weights[f] = wb[f].total;
    try { out.gguf = gguf_sizes(p, GGUF_LEVELS.concat(["BF16"])); } catch (e) {
      out.gguf = null;
      out.notes = (out.notes || []).concat([`GGUF sizes unavailable: ${e.message}`.slice(0, 200)]);
    }
    const kpt = kv_elems(p, 1);
    out.kv_per_token = { elems: kpt, by_kind: kv_by_kind(p, 1) };
    for (const f of Object.keys(KV_FORMATS)) out.kv_per_token[f] = kv_bytes(kpt, f);
    out.state = { elems: st, bf16: 2 * st, fp32: 4 * st };
    out.flops = {}; out.elementwise = {}; out.kv = {}; out.activation = {}; out.serving = {}; out.bandwidth = {};
    const emb_row = 2 * ((p.vocab && p.lm_head_W) ? fdiv(p.lm_head_W, p.vocab) : h);
    for (const [key, L] of Object.entries(ctx)) {
      out.flops[key] = flops_at(p, L);
      out.elementwise[key] = elementwise_at(p, L, norm_kind);
      const ke = kv_elems(p, L);
      const kr = { L, elems: ke, by_kind: kv_by_kind(p, L) };
      for (const f of Object.keys(KV_FORMATS)) kr[f] = kv_bytes(ke, f);
      out.kv[key] = kr;
      const act_f = activation_bytes(p, L, "flash");
      out.activation[key] = { flash: act_f, eager: activation_bytes(p, L, "eager") };
      const sv = {};
      for (const B of BATCHES) sv[String(B)] = wb.bf16.total + B * (2 * ke + 2 * st) + act_f;
      out.serving[key] = sv;
      const dfl = out.flops[key].decode.total;
      const bw = {};
      for (const B of BATCHES) {
        const r = decode_read(p, L, B, wb.bf16, emb_row);
        r.flops = B * dfl; r.B = B;
        r.intensity = r.bytes ? r.flops / r.bytes : null;
        bw[String(B)] = r;
      }
      out.bandwidth[key] = bw;
    }
    const hws = hw_list !== undefined && hw_list !== null ? hw_list : (T.hardware || []);
    if (hws.length) {
      const rf = {};
      for (const hw of hws) {
        const d = {};
        for (const key of ROOFLINE_L) {
          const dec = {};
          for (const B of BATCHES) dec[String(B)] = decode_bound(out.bandwidth[key][String(B)], hw);
          d[key] = { decode: dec, prefill_s: prefill_bound(out.flops[key].prefill.total, wb.bf16.total, hw) };
        }
        rf[hw.id] = d;
      }
      out.roofline = rf;
    }
    return out;
  }

  const API = { setTables, breakdown, plan_ordered, fix_order, layer_order, params_breakdown, flops_at, elementwise_at, kv_elems, kv_bytes,
    weight_bytes, gguf_sizes, gguf_tensors, activation_bytes, decode_read, decode_bound, prefill_bound, hw_peak,
    contexts_for, state_elems, group_label, expert_coverage, routed_fraction, kv_group_per_token,
    CONTEXTS, BATCHES, KV_FORMATS, WEIGHT_FORMATS, GGUF_LEVELS, ROOFLINE_L };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.ArchBreakdown = API;
})(typeof window !== "undefined" ? window : globalThis);
