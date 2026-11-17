// The zoo page's Variants view (graph/zoo_variants.py builds the data block #zoo-variants).
// Which variants of attention (and of the other components) the snapshot's models use, and which models use them:
// a taxonomy with live counts, numeric dimension histograms, faceted filters, a models list (families -> shapes ->
// repos), a per-variant detail (explanation, first on the Hub, frameworks, the relation's rules, code evidence),
// a co-occurrence matrix and an adoption timeline. Every count is computed here from the embedded memberships.
// The page calls ZooVariants.init(api) once; the hooks are render / hashParts / fromHash / keepShape.
window.ZooVariants = (() => {
"use strict";
const SRC = document.getElementById("zoo-variants");
let X = null;
try { X = SRC ? JSON.parse(SRC.textContent) : null; } catch (e) { X = null; }
if (!X || !X.variants) return null;

function init(api) {
  const { N, el, sv, fmt, fmtParams, $ } = api;
  const n = X.n;
  // ---- the models, node by node, members in order (graph/zoo_variants.py model_facts)
  const mNode = new Int32Array(n), mRepo = new Array(n);
  { let k = 0; N.forEach((d, i) => d.members.forEach(r => { mNode[k] = i; mRepo[k] = r; k++; })); }
  const firstOfNode = new Int32Array(N.length); { let k = 0; N.forEach((d, i) => { firstOfNode[i] = k; k += d.members.length; }); }
  const M = X.models, TYPES = X.types, FWN = X.fw_names || [];
  const mParams = Array.from({ length: n }, (_, k) => M.p[k] || N[mNode[k]].params || 0);
  const unb = s => { const b = Uint8Array.from(atob(s), c => c.charCodeAt(0)), o = new Uint8Array(n); for (let i = 0; i < n; i++) o[i] = (b[i >> 3] >> (7 - (i & 7))) & 1; return o; };
  const noCode = unb(M.nc);
  const VS = X.variants, VBY = new Map(VS.map(v => [v.id, v]));
  VS.forEach(v => { v.M = unb(v.m); v.U = v.um ? unb(v.um) : null; v.kids = []; });
  VS.forEach(v => { if (v.p && VBY.has(v.p)) VBY.get(v.p).kids.push(v); });
  const AX = X.axes, COMPS = X.comps;
  const FS = X.facets, FBY = new Map(FS.map(f => [f.id, f]));
  const KLABEL = { any: "any layer", g: "global layers", s: "sliding layers", x: "KV-shared layers" };
  const variantsOfAxis = a => VS.filter(v => v.a === a || v.also === a);
  const FINDS = X.findings || {};

  // ---- numeric facets: per model value (number, "?" unknown, null not applicable), per kind for pk facets
  FS.forEach(f => {
    const vals = [];
    if (f.pk) for (const k in f.v) for (const x of f.v[k]) { if (typeof x === "number") vals.push(x); }
    else for (const x of f.v) if (typeof x === "number") vals.push(x);
    const uniq = [...new Set(vals)].sort((a, b) => a - b);
    f.kinds = f.pk ? Object.keys(f.v) : [];
    if (uniq.length <= 28 || (f.disc && uniq.length <= 96)) f.bins = uniq.map(x => ({ lo: x, hi: x, ho: 0 }));
    else if (f.log) {
      const lo = Math.floor(Math.log2(Math.max(uniq[0], 1e-9))), hi = Math.floor(Math.log2(uniq[uniq.length - 1]));
      f.bins = []; for (let e = lo; e <= hi; e++) f.bins.push({ lo: 2 ** e, hi: 2 ** (e + 1), ho: 1 });
    } else {
      const a = uniq[0], b = uniq[uniq.length - 1], raw = (b - a) / 16, p = 10 ** Math.floor(Math.log10(raw || 1));
      const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw) || raw;
      f.bins = []; for (let x = Math.floor(a / step) * step; x <= b; x += step) f.bins.push({ lo: +x.toPrecision(12), hi: +(x + step).toPrecision(12), ho: 1 });
    }
  });
  const valsOf = (f, k, kind) => {                       // the model's values under a kind selection
    if (!f.pk) { const x = f.v[k]; return x == null ? null : [x]; }
    const ks = kind === "any" ? f.kinds : [kind], out = [];
    for (const q of ks) { const x = f.v[q] ? f.v[q][k] : null; if (x != null) out.push(x); }
    return out.length ? out : null;
  };
  const inR = (x, F) => typeof x === "number" && x >= F.lo && (F.ho ? x < F.hi : x <= F.hi);

  // ---- state
  const DEF = { comp: "attention", pane: "models", vsel: "", inc: [], nf: {}, fam: "", from: "", to: "", pmin: 0, pmax: 0,
    fwk: "", front: false, q: "", sort: "dl", dir: -1, carry: false, ca: "attention", cb: "attention", kids: false,
    ta: "", tm: "month", all: false };
  const S = JSON.parse(JSON.stringify(DEF));
  let R = new Uint8Array(n), REX = {}, dirty = true;

  function catMask() {                                   // categorical + common filters (not the numeric ones)
    const o = new Uint8Array(n).fill(1);
    for (const id of S.inc) { const v = VBY.get(id); if (!v) continue; for (let k = 0; k < n; k++) o[k] &= v.M[k]; }
    const ti = S.fam ? TYPES.indexOf(S.fam) : -1, fi = S.fwk ? FWN.indexOf(S.fwk) : -1, q = S.q.toLowerCase();
    for (let k = 0; k < n; k++) {
      if (!o[k]) continue;
      if (S.fam && M.mt[k] !== ti) { o[k] = 0; continue; }
      const dt = M.date[k];
      if (S.from && !(dt && dt.slice(0, 7) >= S.from)) { o[k] = 0; continue; }
      if (S.to && !(dt && dt.slice(0, 7) <= S.to)) { o[k] = 0; continue; }
      if (S.pmin && !(mParams[k] >= S.pmin)) { o[k] = 0; continue; }
      if (S.pmax && !(mParams[k] <= S.pmax)) { o[k] = 0; continue; }
      if (fi >= 0 && X.fw[k][fi] !== "1") { o[k] = 0; continue; }
      if (S.front && !(N[mNode[k]].fr > 0)) { o[k] = 0; continue; }
      if (q && !mRepo[k].toLowerCase().includes(q)) { o[k] = 0; continue; }
    }
    return o;
  }
  function numMask(fid) {
    const F = S.nf[fid], f = FBY.get(fid), o = new Uint8Array(n);
    if (!f || !F) return o.fill(1);
    for (let k = 0; k < n; k++) { const vs = valsOf(f, k, F.k); if (vs && vs.some(x => inR(x, F))) o[k] = 1; }
    return o;
  }
  function compute() {
    if (!dirty) return;
    const C = catMask(), NM = {};
    for (const fid in S.nf) NM[fid] = numMask(fid);
    const and = skip => { const o = C.slice(); for (const fid in NM) if (fid !== skip) { const m = NM[fid]; for (let k = 0; k < n; k++) o[k] &= m[k]; } return o; };
    R = and(null); REX = {};
    for (const fid in NM) REX[fid] = and(fid);
    dirty = false;
  }
  const count = (A, B) => { let c = 0; for (let k = 0; k < n; k++) c += A[k] & B[k]; return c; };
  const total = A => { let c = 0; for (let k = 0; k < n; k++) c += A[k]; return c; };
  const shapesIn = A => { const s = new Set(); for (let k = 0; k < n; k++) if (A[k]) s.add(mNode[k]); return s; };
  const filtersOn = () => S.inc.length || Object.keys(S.nf).length || S.fam || S.from || S.to || S.pmin || S.pmax || S.fwk || S.front || S.q;
  function keepShape(i) {                                 // the Models table's filter, when the reader carries it over
    if (!S.carry || !filtersOn()) return true;
    compute();
    const a = firstOfNode[i];
    for (let j = 0; j < N[i].members.length; j++) if (R[a + j]) return true;
    return false;
  }
  function changed(opt = {}) { dirty = true; render(); if (!opt.nohash) api.writeHash(); if (S.carry) api.renderModels(); }

  // ---- the view's skeleton
  const root = $("view-variants");
  root.replaceChildren();
  const head = el("div", { class: "view-head" }, el("h2", { text: "Variants" }), el("span", { class: "sub", id: "zv-count" }));
  const ctabs = el("div", { class: "zv-ctabs", role: "tablist", "aria-label": "Component" });
  for (const c of COMPS) ctabs.append(el("button", { type: "button", role: "tab", "data-c": c.id, text: c.t,
    onclick: () => { S.comp = c.id; S.ca = S.cb = c.id; S.ta = ""; changed(); } }));
  const famSel = el("select", { id: "zv-fam", "aria-label": "Family (model type)" });
  const fromSel = el("select", { id: "zv-from", "aria-label": "Released from" }), toSel = el("select", { id: "zv-to", "aria-label": "Released until" });
  const pminSel = el("select", { id: "zv-pmin", "aria-label": "Min params" }), pmaxSel = el("select", { id: "zv-pmax", "aria-label": "Max params" });
  const fwSel = el("select", { id: "zv-fw", "aria-label": "Native framework" });
  const frontChk = el("input", { type: "checkbox", id: "zv-front" });
  const qIn = el("input", { type: "search", id: "zv-q", placeholder: "repo contains", autocomplete: "off", spellcheck: "false" });
  const months = [...new Set(M.date.filter(Boolean).map(d => d.slice(0, 7)))].sort();
  fromSel.append(el("option", { value: "", text: "any" }), ...months.map(m => el("option", { value: m, text: m })));
  toSel.append(el("option", { value: "", text: "any" }), ...months.map(m => el("option", { value: m, text: m })));
  const PSTEPS = [1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11, 3e11, 1e12];
  pminSel.append(el("option", { value: "0", text: "any" }), ...PSTEPS.map(p => el("option", { value: String(p), text: "≥ " + fmtParams(p) })));
  pmaxSel.append(el("option", { value: "0", text: "any" }), ...PSTEPS.map(p => el("option", { value: String(p), text: "≤ " + fmtParams(p) })));
  famSel.addEventListener("change", () => { S.fam = famSel.value; changed(); });
  fromSel.addEventListener("change", () => { S.from = fromSel.value; changed(); });
  toSel.addEventListener("change", () => { S.to = toSel.value; changed(); });
  pminSel.addEventListener("change", () => { S.pmin = +pminSel.value; changed(); });
  pmaxSel.addEventListener("change", () => { S.pmax = +pmaxSel.value; changed(); });
  fwSel.addEventListener("change", () => { S.fwk = fwSel.value; changed(); });
  frontChk.addEventListener("change", () => { S.front = frontChk.checked; changed(); });
  let qt = 0; qIn.addEventListener("input", () => { clearTimeout(qt); qt = setTimeout(() => { S.q = qIn.value.trim(); changed(); }, 120); });
  const common = el("div", { class: "zv-common" },
    el("label", {}, "Family ", famSel), el("label", {}, "Released ", fromSel, " – ", toSel),
    el("label", {}, "Size ", pminSel, " ", pmaxSel), el("label", {}, "Native in ", fwSel),
    el("label", { class: "chip" }, frontChk, "Frontier only"), qIn);
  const carryChk = el("input", { type: "checkbox", id: "zv-carry" });
  carryChk.addEventListener("change", () => { S.carry = carryChk.checked; changed(); api.renderModels(); });
  const actions = el("div", { class: "zv-actions" },
    el("button", { class: "btn small", type: "button", id: "zv-clear", text: "Clear filters", onclick: () => { const keep = { comp: S.comp, pane: S.pane, vsel: S.vsel, carry: S.carry, sort: S.sort, dir: S.dir };
      Object.assign(S, JSON.parse(JSON.stringify(DEF)), keep); changed(); } }),
    el("button", { class: "btn small", type: "button", id: "zv-csv", text: "CSV", title: "The matching models as CSV", onclick: () => download("csv") }),
    el("button", { class: "btn small", type: "button", id: "zv-json", text: "JSON", title: "The matching models as JSON", onclick: () => download("json") }),
    el("label", { class: "chip", title: "Apply these filters to the Models table too" }, carryChk, "Apply to the Models table"),
    el("button", { class: "btn small", type: "button", id: "zv-to-models", text: "Open in Models", onclick: () => { S.carry = true; dirty = true; api.setView("models"); api.writeHash(); } }));
  const chips = el("div", { class: "zv-chips", id: "zv-chips", "aria-live": "polite" });
  const tax = el("div", { class: "zv-tax", id: "zv-tax" });
  const dims = el("div", { class: "zv-dims", id: "zv-dims" });
  const ptabs = el("div", { class: "zv-ptabs", role: "tablist", "aria-label": "Pane" });
  for (const [p, t] of [["models", "Models"], ["detail", "Variant"], ["cooc", "Co-occurrence"], ["timeline", "Timeline"]])
    ptabs.append(el("button", { type: "button", role: "tab", "data-p": p, text: t, onclick: () => { S.pane = p; render(); api.writeHash(); } }));
  const pane = el("div", { class: "zv-pane", id: "zv-pane" });
  const tip = el("div", { class: "zv-tip", hidden: "" });
  root.append(head, el("div", { class: "zv-top" }, ctabs, common, actions), chips,
    el("div", { class: "zv-grid" }, el("div", { class: "zv-left" }, el("h3", { text: "Variants" }), tax, el("h3", { text: "Dimensions" }), dims),
      el("div", { class: "zv-right" }, ptabs, pane)), tip);
  const showTip = (e, text) => { tip.textContent = text; tip.hidden = false; const r = root.getBoundingClientRect();
    tip.style.left = Math.min(e.clientX - r.left + 12, r.width - 260) + "px"; tip.style.top = (e.clientY - r.top + 14) + "px"; };
  const hideTip = () => { tip.hidden = true; };

  function fillSelects() {
    const C = (() => { const s = S.fam; S.fam = ""; const c = catMask(); S.fam = s; return c; })();
    const cnt = new Map(); for (let k = 0; k < n; k++) if (C[k]) cnt.set(M.mt[k], (cnt.get(M.mt[k]) || 0) + 1);
    const opts = [el("option", { value: "", text: "any" })];
    [...TYPES.keys()].filter(t => cnt.get(t) || TYPES[t] === S.fam).sort((a, b) => (cnt.get(b) || 0) - (cnt.get(a) || 0) || (TYPES[a] < TYPES[b] ? -1 : 1))
      .forEach(t => opts.push(el("option", { value: TYPES[t], text: `${TYPES[t]} (${fmt(cnt.get(t) || 0)})` })));
    famSel.replaceChildren(...opts); famSel.value = S.fam;
    const F2 = (() => { const s = S.fwk; S.fwk = ""; const c = catMask(); S.fwk = s; return c; })();
    fwSel.replaceChildren(el("option", { value: "", text: "any" }), ...FWN.map((f, i) => { let c = 0; for (let k = 0; k < n; k++) if (F2[k] && X.fw[k][i] === "1") c++;
      return el("option", { value: f, text: `${f} (${fmt(c)})` }); }));
    fwSel.value = S.fwk; fromSel.value = S.from; toSel.value = S.to; pminSel.value = String(S.pmin); pmaxSel.value = String(S.pmax);
    frontChk.checked = S.front; carryChk.checked = S.carry; if (document.activeElement !== qIn) qIn.value = S.q;
  }

  // ---- chips (the active conditions)
  function renderChips() {
    const out = [];
    const chip = (label, off, cls) => el("span", { class: "chip on " + (cls || "") }, label,
      el("button", { type: "button", class: "x", "aria-label": "Remove " + label, text: "×", onclick: off }));
    for (const id of S.inc) { const v = VBY.get(id); if (v) out.push(chip(v.t, () => { S.inc = S.inc.filter(x => x !== id); changed(); }, "cat")); }
    for (const fid in S.nf) { const F = S.nf[fid], f = FBY.get(fid); if (!f) continue;
      out.push(chip(`${f.t}${f.pk ? " (" + KLABEL[F.k] + ")" : ""} ${rangeText(F)}`, () => { delete S.nf[fid]; changed(); }, "num")); }
    if (S.fam) out.push(chip("family " + S.fam, () => { S.fam = ""; changed(); }));
    if (S.from || S.to) out.push(chip(`released ${S.from || "…"} – ${S.to || "…"}`, () => { S.from = S.to = ""; changed(); }));
    if (S.pmin || S.pmax) out.push(chip(`size ${S.pmin ? "≥ " + fmtParams(S.pmin) : ""} ${S.pmax ? "≤ " + fmtParams(S.pmax) : ""}`, () => { S.pmin = S.pmax = 0; changed(); }));
    if (S.fwk) out.push(chip("native in " + S.fwk, () => { S.fwk = ""; changed(); }));
    if (S.front) out.push(chip("frontier only", () => { S.front = false; changed(); }));
    if (S.q) out.push(chip(`repo contains "${S.q}"`, () => { S.q = ""; changed(); }));
    chips.replaceChildren(...(out.length ? out : [el("span", { class: "muted", text: "No filter: every model of the snapshot. Tick variants, click histogram bars or pick a family / date / size to combine conditions." })]));
  }
  const num = x => Math.abs(x) >= 1e4 ? fmtParams(x) : String(+x.toPrecision(4));
  function rangeText(F) {
    if (F.lo === F.hi && !F.ho) return "= " + num(F.lo);
    if (F.lo == null || F.lo === -Infinity) return "≤ " + num(F.hi);
    if (F.hi == null || F.hi === Infinity) return "≥ " + num(F.lo);
    return `${num(F.lo)} – ${F.ho ? "<" : ""}${num(F.hi)}`;
  }

  // ---- the taxonomy: axes of the component, variants with live counts
  function renderTax() {
    const c = S.comp, axes = AX.filter(a => a.c === c);
    const rtot = total(R);
    const rows = [];
    for (const a of axes) {
      const vs = variantsOfAxis(a.id);
      const used = vs.filter(v => v.n > 0), unused = vs.filter(v => v.n === 0);
      const sec = el("section", { class: "zv-axis", "data-axis": a.id }, el("h4", { title: a.d, text: a.t }));
      const top = used.filter(v => !v.p || !VBY.has(v.p));
      const line = (v, depth) => {
        const live = count(R, v.M), u = v.U ? count(R, v.U) : 0, on = S.inc.includes(v.id);
        const inp = el("input", { type: "checkbox", "aria-label": "Filter on " + v.t }); inp.checked = on;
        inp.addEventListener("change", () => { if (inp.checked) S.inc.push(v.id); else S.inc = S.inc.filter(x => x !== v.id); changed(); });
        const w = rtot ? Math.round(100 * live / rtot) : 0;
        const row = el("div", { class: "zv-v" + (depth ? " kid" : "") + (live ? "" : " zero") + (S.vsel === v.id ? " sel" : ""), "data-v": v.id },
          inp,
          el("button", { type: "button", class: "zv-name", text: v.t, title: v.d, onclick: () => { S.vsel = v.id; S.pane = "detail"; render(); api.writeHash(); } }),
          el("span", { class: "zv-bar", "aria-hidden": "true" }, el("i", { style: `width:${Math.max(w, live ? 2 : 0)}%` })),
          el("span", { class: "zv-n", title: `${fmt(live)} of the ${fmt(rtot)} matching models (${fmt(v.n)} in the snapshot, ${fmt(v.s)} shapes)` },
            fmt(live), live !== v.n ? el("small", { text: " / " + fmt(v.n) }) : null),
          u ? el("span", { class: "zv-u", title: `${fmt(u)} matching models whose type has no modelling code to check: unknown, not absent`, text: "? " + fmt(u) }) : null);
        sec.append(row);
        if (v.kids.length) for (const k of v.kids.filter(k => k.n > 0)) line(k, 1);
      };
      for (const v of top) line(v, 0);
      const zero = unused.filter(v => !v.p || !VBY.get(v.p) || VBY.get(v.p).n > 0);
      if (zero.length) sec.append(el("p", { class: "zv-unused", text: "Not in the snapshot: " + zero.map(v => v.t).join(", ") }));
      rows.push(sec);
    }
    tax.replaceChildren(...rows);
  }

  // ---- dimensions: histograms of the component's numeric facets (counts given every OTHER filter)
  function renderDims() {
    const fs = FS.filter(f => f.c === S.comp);
    const out = [];
    for (const f of fs) {
      const F = S.nf[f.id], kind = F ? F.k : (f.kindSel || "any"), base = REX[f.id] || R;
      const counts = f.bins.map(() => 0); let unk = 0, na = 0, anyv = 0;
      for (let k = 0; k < n; k++) {
        if (!base[k]) continue;
        const vs = valsOf(f, k, kind);
        if (!vs) { na++; continue; }
        let hit = false; const seen = new Set();
        for (const x of vs) { if (x === "?") { continue; } const b = f.bins.findIndex(B => inR(x, B)); if (b >= 0 && !seen.has(b)) { seen.add(b); counts[b]++; hit = true; } }
        if (!hit && vs.includes("?")) unk++;
        if (hit) anyv++;
      }
      const max = Math.max(1, ...counts);
      const sec = el("section", { class: "zv-dim", "data-f": f.id });
      const kindSel = f.pk ? el("select", { class: "zv-kind", "aria-label": f.t + ": layer kind" },
        ...["any", ...f.kinds].map(k => el("option", { value: k, text: KLABEL[k] }))) : null;
      if (kindSel) { kindSel.value = kind; kindSel.addEventListener("change", () => { f.kindSel = kindSel.value; if (S.nf[f.id]) { S.nf[f.id].k = kindSel.value; changed(); } else render(); }); }
      const lo = el("input", { type: "number", class: "zv-lo", placeholder: "min", step: "any", "aria-label": f.t + " min", value: F && F.lo != null && isFinite(F.lo) ? F.lo : "" });
      const hi = el("input", { type: "number", class: "zv-hi", placeholder: "max", step: "any", "aria-label": f.t + " max", value: F && F.hi != null && isFinite(F.hi) ? F.hi : "" });
      const apply = () => { const a = lo.value === "" ? -Infinity : +lo.value, b = hi.value === "" ? Infinity : +hi.value;
        if (a === -Infinity && b === Infinity) delete S.nf[f.id]; else S.nf[f.id] = { k: kind, lo: a, hi: b, ho: 0 }; changed(); };
      lo.addEventListener("change", apply); hi.addEventListener("change", apply);
      sec.append(el("div", { class: "zv-dimh" }, el("b", { text: f.t }), f.unit ? el("span", { class: "muted", text: " " + f.unit }) : null,
        kindSel, el("span", { class: "zv-rng" }, lo, "–", hi), F ? el("button", { type: "button", class: "btn small", text: "clear", onclick: () => { delete S.nf[f.id]; changed(); } }) : null));
      const W = 360, H = 64, bw = W / Math.max(f.bins.length + (unk ? 1 : 0), 1);
      // tick labels: evenly spaced for range bins; for one-value bars the most frequent values that do not collide
      const labelled = new Set(), room = 34 / bw;
      if (f.bins.length && f.bins[0].lo === f.bins[0].hi && f.bins.length > 8) {
        const order = f.bins.map((_, b) => b).sort((a, b) => counts[b] - counts[a] || a - b);
        for (const b of order) { if (!counts[b] || labelled.size >= 8) break; if ([...labelled].every(c => Math.abs(c - b) >= room)) labelled.add(b); }
      } else { const every = Math.max(1, Math.ceil(room)); f.bins.forEach((_, b) => { if (b % every === 0) labelled.add(b); }); }
      const g = sv("svg", { viewBox: `0 0 ${W} ${H + 16}`, class: "zv-hist", role: "img", "aria-label": `${f.t}: models per value` });
      f.bins.forEach((B, b) => {
        const h = counts[b] ? Math.max(2, Math.round((H - 4) * counts[b] / max)) : 0;
        const on = F && B.lo >= F.lo && (F.ho ? B.hi <= F.hi : B.hi <= F.hi);
        const grp = sv("g", { class: "bin" + (on ? " on" : "") + (counts[b] ? "" : " empty"), "data-lo": B.lo, "data-hi": B.hi, "data-n": counts[b], tabindex: counts[b] ? "0" : "-1" });
        grp.append(sv("rect", { x: b * bw, y: 0, width: bw, height: H, class: "hit" }));
        if (h) grp.append(sv("rect", { x: b * bw + 1, y: H - h, width: Math.max(bw - 2, 1), height: h, rx: Math.min(2, bw / 4), class: "zv-b" }));
        const label = B.lo === B.hi ? num(B.lo) : `${num(B.lo)}–${num(B.hi)}`;
        const pick = e => { if (!counts[b] && !on) return;
          const cur = S.nf[f.id];
          if (e.shiftKey && cur) S.nf[f.id] = { k: kind, lo: Math.min(cur.lo, B.lo), hi: Math.max(cur.hi, B.hi), ho: B.ho || cur.ho };
          else if (cur && cur.lo === B.lo && cur.hi === B.hi) delete S.nf[f.id];
          else S.nf[f.id] = { k: kind, lo: B.lo, hi: B.hi, ho: B.ho };
          changed(); };
        grp.addEventListener("click", pick);
        grp.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(e); } });
        grp.addEventListener("mousemove", e => showTip(e, `${f.t} ${label}: ${fmt(counts[b])} models`));
        grp.addEventListener("mouseleave", hideTip);
        g.append(grp);
        if (labelled.has(b)) { const t = sv("text", { x: Math.min(Math.max(b * bw + bw / 2, 12), W - 12), y: H + 12, "text-anchor": "middle", class: "tick" }); t.textContent = num(B.lo); g.append(t); }
      });
      if (unk) {
        const b = f.bins.length, h = Math.max(2, Math.round((H - 4) * unk / max));
        const grp = sv("g", { class: "bin unk" }); grp.append(sv("rect", { x: b * bw + 1, y: H - h, width: Math.max(bw - 2, 1), height: h, rx: 2, class: "zv-b" }));
        grp.addEventListener("mousemove", e => showTip(e, `${f.t} unknown: ${fmt(unk)} models (the classifier gives no value)`)); grp.addEventListener("mouseleave", hideTip);
        const t = sv("text", { x: b * bw + bw / 2, y: H + 12, "text-anchor": "middle", class: "tick" }); t.textContent = "?"; g.append(grp, t);
      }
      sec.append(g, el("p", { class: "zv-dimf muted", text: `${fmt(anyv)} with a value${unk ? `, ${fmt(unk)} unknown` : ""}${na ? `, ${fmt(na)} n/a` : ""}${f.pk && kind === "any" ? " · a model whose layer kinds differ counts in each of their bars" : ""}` }));
      out.push(sec);
    }
    dims.replaceChildren(...(out.length ? out : [el("p", { class: "muted", text: "No numeric dimension for this component." })]));
  }

  // ---- the models list: families -> shapes -> repos
  const open = new Set();
  function familyTree(A) {
    const fams = new Map();
    for (let k = 0; k < n; k++) {
      if (!A[k]) continue;
      const t = TYPES[M.mt[k]], i = mNode[k];
      if (!fams.has(t)) fams.set(t, new Map());
      const sh = fams.get(t); if (!sh.has(i)) sh.set(i, []); sh.get(i).push(k);
    }
    const key = ks => S.sort === "dl" ? ks.reduce((s, k) => s + (M.dl[k] || 0), 0)
      : S.sort === "date" ? (ks.map(k => M.date[k]).filter(Boolean).sort()[0] || "")
      : Math.max(...ks.map(k => N[mNode[k]].params || 0));
    const keyRepo = k => S.sort === "dl" ? M.dl[k] || 0 : S.sort === "date" ? M.date[k] || "" : mParams[k];
    const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0) * S.dir;
    const out = [];
    for (const [t, sh] of fams) {
      const shapes = [...sh].map(([i, ks]) => ({ i, ks: ks.sort((a, b) => cmp(keyRepo(a), keyRepo(b)) || a - b), key: key(ks) }));
      shapes.sort((a, b) => cmp(a.key, b.key) || a.i - b.i);
      const all = shapes.flatMap(s => s.ks);
      out.push({ t, shapes, key: key(all), n: all.length });
    }
    out.sort((a, b) => cmp(a.key, b.key) || b.n - a.n);
    return out;
  }
  function cardLink(k, text) {
    const i = mNode[k], r = mRepo[k];
    return el("a", { href: "#m=" + encodeURIComponent(r) + "&v=variants", class: "zv-card", "data-i": i, "data-repo": r, text: text || r,
      onclick: e => { e.preventDefault(); api.select(i, { repo: r }); } });
  }
  function modelsList(A, opt = {}) {
    const fams = familyTree(A);
    const wrap = el("div", { class: "zv-list" });
    const sortBar = el("div", { class: "zv-sort" }, el("span", { class: "muted", text: "Sort by " }),
      ...[["dl", "downloads"], ["date", "date"], ["size", "size"]].map(([k, t]) => el("button", { type: "button", class: "btn small" + (S.sort === k ? " on" : ""), "data-sort": k,
        text: t + (S.sort === k ? (S.dir < 0 ? " ↓" : " ↑") : ""), onclick: () => { if (S.sort === k) S.dir = -S.dir; else { S.sort = k; S.dir = -1; } render(); api.writeHash(); } })));
    wrap.append(sortBar);
    const lim = S.all ? fams.length : Math.min(fams.length, opt.lim || 40);
    const tbl = el("table", { class: "zv-tbl" }, el("thead", {}, el("tr", {}, el("th", { text: "Family / shape / repo" }), el("th", { class: "num", text: "Models" }),
      el("th", { class: "num", text: "Downloads" }), el("th", { text: "First on Hub" }), el("th", { class: "num", text: "Params" }))));
    const body = el("tbody");
    const agg = ks => ({ dl: ks.reduce((s, k) => s + (M.dl[k] || 0), 0), date: ks.map(k => M.date[k]).filter(Boolean).sort()[0] || "", p: Math.max(...ks.map(k => N[mNode[k]].params || 0)) });
    for (const fm of fams.slice(0, lim)) {
      const ks = fm.shapes.flatMap(s => s.ks), a = agg(ks), fo = open.has("f:" + fm.t);
      const tr = el("tr", { class: "fam", "data-fam": fm.t }, el("td", {}, el("button", { type: "button", class: "zv-tog", "aria-expanded": String(fo), text: (fo ? "▾ " : "▸ ") + fm.t,
        onclick: () => { fo ? open.delete("f:" + fm.t) : open.add("f:" + fm.t); render(); } }), el("span", { class: "muted", text: ` · ${fm.shapes.length} shape${fm.shapes.length > 1 ? "s" : ""}` })),
        el("td", { class: "num", text: fmt(ks.length) }), el("td", { class: "num", text: fmtDl(a.dl) }), el("td", { text: a.date }), el("td", { class: "num", text: a.p ? fmtParams(a.p) : "" }));
      body.append(tr);
      const shapes = fo ? fm.shapes : fm.shapes.slice(0, 3);
      for (const sh of shapes) {
        const so = open.has("s:" + sh.i), b = agg(sh.ks);
        body.append(el("tr", { class: "shape", "data-i": sh.i }, el("td", {}, el("button", { type: "button", class: "zv-tog", "aria-expanded": String(so), "aria-label": "Show the repos", text: so ? "▾" : "▸",
          onclick: () => { so ? open.delete("s:" + sh.i) : open.add("s:" + sh.i); render(); } }), " ", cardLink(sh.ks[0], N[sh.i].name),
          sh.ks.length > 1 ? el("span", { class: "muted", text: ` · ${sh.ks.length} repos` }) : null),
          el("td", { class: "num", text: fmt(sh.ks.length) }), el("td", { class: "num", text: fmtDl(b.dl) }), el("td", { text: b.date }), el("td", { class: "num", text: b.p ? fmtParams(b.p) : "" })));
        if (so) for (const k of sh.ks) body.append(el("tr", { class: "repo" }, el("td", {}, cardLink(k)), el("td", {}), el("td", { class: "num", text: fmtDl(M.dl[k] || 0) }),
          el("td", { text: M.date[k] || "" }), el("td", { class: "num", text: mParams[k] ? fmtParams(mParams[k]) : "" })));
      }
      if (!fo && fm.shapes.length > 3) body.append(el("tr", { class: "more" }, el("td", { colspan: "5" }, el("button", { type: "button", class: "btn small", text: `+ ${fm.shapes.length - 3} more shapes`,
        onclick: () => { open.add("f:" + fm.t); render(); } }))));
    }
    tbl.append(body);
    wrap.append(tbl);
    if (lim < fams.length) wrap.append(el("button", { class: "btn small", type: "button", text: `Show all ${fmt(fams.length)} families`, onclick: () => { S.all = true; render(); } }));
    if (!fams.length) wrap.append(el("p", { class: "muted", text: "No model matches every condition." }));
    return wrap;
  }
  const fmtDl = x => x >= 1e6 ? (x / 1e6).toFixed(1).replace(/\.0$/, "") + "M" : x >= 1e3 ? Math.round(x / 1e3) + "K" : String(x);

  // ---- the detail of one variant
  function diagram(kind) {
    const [t, k] = kind.split(":");
    if (t === "mask") {
      const T = 12, c = 7, g = sv("svg", { viewBox: `0 0 ${T * c + 2} ${T * c + 2}`, class: "zv-dg", role: "img", "aria-label": "Attention mask: rows are queries, columns keys" });
      const on = (i, j) => ({ causal: j <= i, sliding: j <= i && i - j < 4, interleave: j <= i, chunked: j <= i && Math.floor(i / 4) === Math.floor(j / 4),
        blocksparse: j <= i && (Math.floor(i / 2) - Math.floor(j / 2) < 2 || Math.floor(j / 2) % 3 === 0), topk: j <= i && ((i * 7 + j * 3) % 5 < 2 || i === j),
        pooled: j <= i && j % 3 === 0, prefix: j <= i || j < 5, bidir: true, block: Math.floor(j / 4) <= Math.floor(i / 4), sinktok: j <= i && (i - j < 4 || j < 2) }[k]);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) g.append(sv("rect", { x: 1 + j * c, y: 1 + i * c, width: c - 1, height: c - 1, rx: 1, class: on(i, j) ? "m1" : "m0" }));
      return g;
    }
    if (t === "heads") {
      const q = 8, kv = { mha: 8, gqa: 2, mqa: 1, mla: 1 }[k], W = 200, g = sv("svg", { viewBox: `0 0 ${W} 64`, class: "zv-dg", role: "img", "aria-label": `${q} query heads reading ${kv} key / value head${kv > 1 ? "s" : ""}` });
      const qx = i => 10 + i * (W - 20) / (q - 1), kx = j => kv === 1 ? W / 2 : 10 + j * (W - 20) / (kv - 1);
      for (let i = 0; i < q; i++) { const j = Math.floor(i / (q / kv)); g.append(sv("line", { x1: qx(i), y1: 14, x2: kx(j), y2: 48, class: "ln" })); }
      for (let i = 0; i < q; i++) g.append(sv("circle", { cx: qx(i), cy: 10, r: 5, class: "q" }));
      for (let j = 0; j < kv; j++) g.append(k === "mla" ? sv("rect", { x: kx(j) - 16, y: 46, width: 32, height: 10, rx: 3, class: "kv" }) : sv("circle", { cx: kx(j), cy: 52, r: 6, class: "kv" }));
      return g;
    }
    if (t === "pos") {
      const W = 200, g = sv("svg", { viewBox: `0 0 ${W} 40`, class: "zv-dg", role: "img", "aria-label": k === "partial" ? "RoPE on part of the head" : "RoPE on the whole head" });
      const d = 16, part = k === "partial" ? 6 : 16;
      for (let i = 0; i < d; i++) g.append(sv("rect", { x: 4 + i * 12, y: 12, width: 10, height: 16, rx: 2, class: i < part ? "m1" : "m0" }));
      return g;
    }
    return null;
  }
  function relationBlock(v) {
    const r = v.rel; if (!r) return null;
    const box = el("div", { class: "zv-rel" });
    const badge = (t, cls) => el("span", { class: "badge " + cls, text: t });
    if (r.k === "slot" && r.v) {
      const of = r.of || [];
      const tg = r.t.length ? r.t.map(([u, c, ex]) => `${u} (${c || "same"}${ex ? ", exact" : ""})`).join(", ") : "none";
      const hs = r.h.length ? r.h.map(([u, c, ex]) => `${u} (${c || "same"}${ex ? ", exact" : ""})`).join(", ") : "none";
      box.append(el("p", {}, badge(r.t.length ? (r.t.every(x => x[2]) ? "free / exact" : "payable") : "wall", r.t.length ? "ok" : "wall"),
        ` As a target: played by hosts whose ${r.slot} is `, el("b", { text: tg }), `${of.length ? ` (of the snapshot's other values: ${of.join(", ")})` : ""}.`));
      box.append(el("p", {}, badge(r.h.length ? (r.h.every(x => x[2]) ? "free / exact" : "payable") : "wall", r.h.length ? "ok" : "wall"),
        ` As a host: plays targets whose ${r.slot} is `, el("b", { text: hs }), "."));
      if (r.zero && r.zero.length) box.append(el("p", { class: "muted", text: "Zeroing edges: plays " + r.zero.map(([t, nt]) => `${t} targets by ${nt}`).join("; ") + "." }));
      box.append(el("p", { class: "muted", text: `The type rule of slot ${r.slot} (model_graph.SLOTS, payable_reaches); capacity (heads, widths, depth) is checked separately.` }));
    } else if (r.k === "slot") box.append(el("p", { class: "muted", text: r.note }));
    else if (r.k === "feat") {
      const T = { free: "ok", wall: "wall", limit: "warn" };
      box.append(el("p", {}, badge(r.t, T[r.t] || ""), ` As a target: a host without ${r.feat} — ${r.t}.`),
        el("p", {}, badge(r.h, T[r.h] || ""), ` As a host: a target without it — ${r.h}.`));
      if (r.why) box.append(el("p", { class: "muted", text: r.why }));
      if (r.absent) box.append(el("p", { class: "muted", text: `This variant is the ABSENCE of ${r.feat}: the rules above are for models that have it.` }));
    } else if (r.k === "kind") {
      box.append(el("p", {}, badge("payable", "ok"), " As a target: an own-V host plays a K = V layer exactly (copy k_proj into v_proj)."),
        el("p", {}, badge("wall", "wall"), " As a host: a K = V layer cannot play a target layer whose V differs from its K."),
        el("p", { class: "muted", text: "Layer-kind matching (model_graph.KIND_REACH): FU plays FU, FT, LU, LT; FT plays FT, LT." }));
    } else if (r.k === "finding") {
      const f = Object.values(FINDS).find(x => x.key === r.key);
      box.append(el("p", {}, badge("not modelled", "warn"), ` The classifier does not read this variant, so the relation does not price it (sweep finding ${r.key}${f ? ", " + f.dir : ""}).`));
    } else box.append(el("p", {}, badge(r.k === "dims" ? "capacity" : "not read", r.k === "dims" ? "ok" : ""), " " + r.note));
    return box;
  }
  function renderDetail() {
    const v = VBY.get(S.vsel) || VS.find(x => x.c === S.comp && x.n > 0);
    if (!v) return el("p", { class: "muted", text: "Pick a variant on the left." });
    const A = v.M, live = count(R, A), axis = AX.find(a => a.id === v.a);
    const out = el("div", { class: "zv-detail", "data-v": v.id });
    const on = S.inc.includes(v.id);
    out.append(el("div", { class: "zv-dh" }, el("h3", { text: v.t }), el("span", { class: "muted", text: `${(COMPS.find(c => c.id === v.c) || {}).t} · ${axis ? axis.t : v.a}` }),
      el("button", { type: "button", class: "btn small", id: "zv-dfilter", text: on ? "Remove from the filter" : "Filter on this variant",
        onclick: () => { if (on) S.inc = S.inc.filter(x => x !== v.id); else S.inc.push(v.id); changed(); } })));
    const fams = new Set(); for (let k = 0; k < n; k++) if (A[k]) fams.add(M.mt[k]);
    out.append(el("dl", { class: "zv-nums" },
      ...[["models", fmt(v.n)], ["shapes", fmt(v.s)], ["families", fmt(fams.size)], ["unknown", fmt(v.u)], ["matching now", fmt(live)]].map(([a, b]) => el("div", {}, el("dt", { text: a }), el("dd", { text: b })))));
    const dg = v.dg ? diagram(v.dg) : null;
    out.append(el("div", { class: "zv-expl" }, el("p", { text: v.d }), dg, v.f ? el("pre", { class: "zv-f", text: v.f }) : null));
    out.append(el("p", {}, el("b", { text: "First on the Hub: " }), v.first ? el("span", {}, v.first[0] + " · ", (() => { const k = mRepo.indexOf(v.first[1]); return k >= 0 ? cardLink(k) : v.first[1]; })()) : "no dated repo"),
      el("p", { class: "muted" }, "Membership: " + v.src + (v.code ? ". Detection needs the type's modelling code: models of a type without code are unknown, not absent." : ".")));
    if (v.kids.length) out.append(el("h4", { text: "Sub-variants" }), el("ul", { class: "zv-kids" }, ...v.kids.map(k => el("li", {},
      el("button", { type: "button", class: "zv-name", text: k.t, onclick: () => { S.vsel = k.id; render(); api.writeHash(); } }), ` ${fmt(k.n)} models · ${fmt(k.s)} shapes`))));
    if (v.p && VBY.get(v.p)) out.append(el("p", {}, "Part of ", el("button", { type: "button", class: "zv-name", text: VBY.get(v.p).t, onclick: () => { S.vsel = v.p; render(); api.writeHash(); } }), "."));
    out.append(el("h4", { text: "The relation" }), relationBlock(v) || el("p", { class: "muted", text: "No rule." }));
    // frameworks
    if (v.fw && Object.keys(v.fw).length) {
      const t = el("table", { class: "zv-fw" }, el("thead", {}, el("tr", {}, el("th", { text: "Framework" }), el("th", { class: "num", text: "Native" }), el("th", { text: "" }), el("th", { class: "num", text: "via Transformers" }))));
      const b = el("tbody");
      for (const f of FWN) { const c = v.fw[f]; if (!c) continue; const pct = Math.round(100 * c[0] / c[2]);
        b.append(el("tr", { "data-fw": f }, el("td", { text: f }), el("td", { class: "num", text: `${fmt(c[0])} / ${fmt(c[2])}` }),
          el("td", {}, el("span", { class: "zv-bar" }, el("i", { style: `width:${pct}%` })), " " + pct + "%"), el("td", { class: "num", text: c[1] ? fmt(c[1]) : "" }))); }
      t.append(b);
      out.append(el("h4", { text: "Native framework support (frameworks.json, per member model)" }), t);
    }
    // code evidence
    out.append(el("h4", { text: "Code evidence" }));
    if (v.ev && v.ev.length) {
      for (const e of v.ev) {
        const lines = el("ol", { class: "zv-lines" }, ...e.lines.map(([ln, txt]) => el("li", {}, e.url ? el("a", { href: e.url + "#L" + ln, target: "_blank", rel: "noopener", text: "L" + ln }) : "L" + ln, el("code", { text: " " + txt }))));
        out.append(el("div", { class: "zv-ev" }, el("p", {}, el("b", { text: e.type }), el("span", { class: "muted", text: ` (${fmt(e.n)} member models) · ` }),
          e.url ? el("a", { href: e.url, target: "_blank", rel: "noopener", text: e.file }) : e.file), lines));
      }
      if (X.evidence) out.append(el("p", { class: "muted", text: "Lines found by a text search of " + X.evidence.source + " for the variant's pattern; the member types come from the sweep inventory." }));
    } else out.append(el("p", { class: "muted", text: "No modelling file of the member types matched the variant's pattern (or the evidence file was not generated)." }));
    // findings
    const fl = (v.find || []).map(k => FINDS[k]).filter(Boolean);
    if (fl.length) {
      out.append(el("h4", { text: "Sweep findings (where code and classifier disagree)" }));
      for (const f of fl) out.append(el("div", { class: "zv-find" }, el("p", {}, el("b", { text: f.key }), el("span", { class: "muted", text: ` · ${f.dir} · ${fmt(f.n)} live models · types ${f.types.slice(0, 8).join(", ")}` })),
        el("p", { text: f.note }), f.ex.length ? el("p", { class: "muted", text: "e.g. " + f.ex.join(", ") }) : null));
    }
    // most frequent companions
    const comp = VS.filter(w => w.id !== v.id && w.n > 0 && w.n < n && w.id !== v.p && w.p !== v.id && w.c !== v.c || (w.id !== v.id && w.n > 0 && w.n < n && w.a !== v.a && w.id !== v.p && w.p !== v.id))
      .map(w => [w, count(A, w.M)]).filter(([w, c]) => c > 0).sort((a, b) => b[1] - a[1] || a[0].n - b[0].n).slice(0, 12);
    if (comp.length) out.append(el("h4", { text: "Most often with (other axes; models using both)" }), el("ul", { class: "zv-kids zv-with" },
      ...comp.map(([w, c]) => el("li", {}, el("button", { type: "button", class: "zv-name", text: w.t, onclick: () => { S.vsel = w.id; render(); api.writeHash(); } }), ` ${fmt(c)} of ${fmt(v.n)} (${Math.round(100 * c / v.n)}%)`))));
    const AR = new Uint8Array(n); for (let k = 0; k < n; k++) AR[k] = A[k] & R[k];
    out.append(el("h4", { text: `Models using it${filtersOn() ? " (within the filters)" : ""}: ${fmt(total(AR))}` }), modelsList(AR, { lim: 25 }));
    return out;
  }

  // ---- co-occurrence
  function renderCooc() {
    const pick = c => VS.filter(v => v.c === c || (v.also && AX.find(a => a.id === v.also && a.c === c))).filter(v => (S.kids || !v.p) && count(R, v.M) > 0)
      .sort((a, b) => count(R, b.M) - count(R, a.M)).slice(0, 28);
    const rows = pick(S.ca), cols = pick(S.cb);
    const selA = el("select", { id: "zv-ca", "aria-label": "Rows" }, ...COMPS.map(c => el("option", { value: c.id, text: c.t }))), selB = el("select", { id: "zv-cb", "aria-label": "Columns" }, ...COMPS.map(c => el("option", { value: c.id, text: c.t })));
    selA.value = S.ca; selB.value = S.cb;
    selA.addEventListener("change", () => { S.ca = selA.value; render(); api.writeHash(); }); selB.addEventListener("change", () => { S.cb = selB.value; render(); api.writeHash(); });
    const kids = el("input", { type: "checkbox" }); kids.checked = S.kids; kids.addEventListener("change", () => { S.kids = kids.checked; render(); });
    const wrap = el("div", { class: "zv-cooc" }, el("div", { class: "zv-ctl" }, el("label", {}, "Rows ", selA), el("label", {}, "Columns ", selB), el("label", { class: "chip" }, kids, "sub-variants")),
      el("p", { class: "muted", text: "Models (within the filters) using both the row and the column variant. Shade: the share of the row's models. Click a cell to filter on both." }));
    const t = el("table", { class: "zv-mx" });
    t.append(el("thead", {}, el("tr", {}, el("th", {}), ...cols.map(c => el("th", { class: "vh" }, el("span", { text: c.t }))))));
    const b = el("tbody");
    const RC = rows.map(r => { const a = new Uint8Array(n); for (let k = 0; k < n; k++) a[k] = r.M[k] & R[k]; return a; });
    rows.forEach((r, ri) => {
      const rn = total(RC[ri]), tr = el("tr", {}, el("th", { text: r.t, title: `${fmt(rn)} models` }));
      for (const c of cols) {
        const x = count(RC[ri], c.M), s = rn ? x / rn : 0;
        const td = el("td", { class: "cell" + (x ? "" : " z"), "data-r": r.id, "data-c": c.id, tabindex: x ? "0" : "-1", style: x ? `--s:${(0.12 + 0.88 * s).toFixed(3)}` : "" , text: x ? fmt(x) : "" });
        td.addEventListener("mousemove", e => showTip(e, `${r.t} + ${c.t}: ${fmt(x)} models (${Math.round(100 * s)}% of ${r.t})`)); td.addEventListener("mouseleave", hideTip);
        const go = () => { if (!x) return; for (const id of [r.id, c.id]) if (!S.inc.includes(id)) S.inc.push(id); S.pane = "models"; changed(); };
        td.addEventListener("click", go); td.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
        tr.append(td);
      }
      b.append(tr);
    });
    t.append(b);
    wrap.append(el("div", { class: "zv-mxw" }, t));
    return wrap;
  }

  // ---- timeline: stacked counts per month of the axis' variants
  const PAL = [1, 2, 3, 4, 5, 6, 7].map(i => `var(--zv-c${i})`);
  function renderTimeline() {
    const axes = AX.filter(a => a.c === S.comp && variantsOfAxis(a.id).filter(v => !v.p && v.n).length);
    if (!axes.length) return el("p", { class: "muted", text: "No variant here." });
    const ax = axes.find(a => a.id === S.ta) || axes[0];
    let vs = variantsOfAxis(ax.id).filter(v => !v.p && count(R, v.M) > 0).sort((a, b) => b.n - a.n);
    const other = vs.length > 7 ? vs.slice(6) : [];
    if (other.length) vs = vs.slice(0, 6);
    const series = [...vs.map((v, i) => ({ t: v.t, id: v.id, M: v.M, c: PAL[i] })), ...(other.length ? [{ t: `Other (${other.length})`, id: "_other", M: (() => { const a = new Uint8Array(n); for (const v of other) for (let k = 0; k < n; k++) a[k] |= v.M[k]; return a; })(), c: "var(--zv-other)" }] : [])];
    const ms = months.slice(); const mi = new Map(ms.map((m, i) => [m, i]));
    const cnt = series.map(() => new Array(ms.length).fill(0)); let undated = 0;
    for (let k = 0; k < n; k++) { if (!R[k]) continue; const d = M.date[k]; if (!d) { undated++; continue; } const m = mi.get(d.slice(0, 7));
      series.forEach((s, j) => { if (s.M[k]) cnt[j][m]++; }); }
    if (S.tm === "cum") cnt.forEach(a => { for (let i = 1; i < a.length; i++) a[i] += a[i - 1]; });
    const tot = ms.map((_, i) => cnt.reduce((s, a) => s + a[i], 0)), max = Math.max(1, ...tot);
    const selA = el("select", { id: "zv-ta", "aria-label": "Axis" }, ...axes.map(a => el("option", { value: a.id, text: a.t }))); selA.value = ax.id;
    selA.addEventListener("change", () => { S.ta = selA.value; render(); api.writeHash(); });
    const selM = el("select", { id: "zv-tm", "aria-label": "Mode" }, el("option", { value: "month", text: "new models per month" }), el("option", { value: "cum", text: "cumulative" })); selM.value = S.tm;
    selM.addEventListener("change", () => { S.tm = selM.value; render(); api.writeHash(); });
    const W = 900, H = 260, L = 40, B = 22, bw = (W - L) / ms.length;
    const g = sv("svg", { viewBox: `0 0 ${W} ${H + B}`, class: "zv-tl", role: "img", "aria-label": `${ax.t}: models first on the Hub per month, stacked by variant` });
    const raw = max / 4, p10 = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 2.5, 5, 10].map(m => m * p10).find(s => s >= raw);
    for (let v = 0; v <= max; v += step) { const y = H - (H - 8) * v / max; g.append(sv("line", { x1: L, x2: W, y1: y, y2: y, class: "grid" }));
      const t = sv("text", { x: L - 6, y: y + 4, "text-anchor": "end", class: "tick" }); t.textContent = fmt(v); g.append(t); }
    ms.forEach((m, i) => {
      let y = H;
      series.forEach((s, j) => { const v = cnt[j][i]; if (!v) return; const h = (H - 8) * v / max; y -= h;
        const r = sv("rect", { x: L + i * bw + 0.5, y: y + (h > 3 ? 1 : 0), width: Math.max(bw - 1, 0.5), height: Math.max(h - (h > 3 ? 1 : 0), 0.5), fill: s.c, class: "seg", "data-s": s.id, "data-m": m });
        r.addEventListener("mousemove", e => showTip(e, `${m} · ${s.t}: ${fmt(v)}${S.tm === "cum" ? " (cumulative)" : ""}`)); r.addEventListener("mouseleave", hideTip);
        g.append(r); });
      if (m.endsWith("-01")) { const t = sv("text", { x: L + i * bw, y: H + 15, class: "tick" }); t.textContent = m.slice(0, 4); g.append(t); }
    });
    const legend = el("div", { class: "zv-legend" }, ...series.map(s => el("span", {}, el("i", { style: `background:${s.c}` }), s.t)));
    return el("div", { class: "zv-tlw" }, el("div", { class: "zv-ctl" }, el("label", {}, "Axis ", selA), el("label", {}, "Show ", selM)), legend, g,
      el("p", { class: "muted", text: `By the repo's creation date on the Hub (models within the filters${undated ? `; ${fmt(undated)} without a date are left out` : ""}). A model counts once per variant of the axis it uses.` }));
  }

  // ---- render
  function render() {
    if (root.hidden) return;
    compute();
    const rt = total(R), rs = shapesIn(R).size;
    $("zv-count").textContent = filtersOn() ? `${fmt(rt)} of ${fmt(n)} models · ${fmt(rs)} of ${fmt(N.length)} shapes match` : `${fmt(n)} models · ${fmt(N.length)} shapes`;
    ctabs.querySelectorAll("button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.c === S.comp)));
    ptabs.querySelectorAll("button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.p === S.pane)));
    fillSelects(); renderChips(); renderTax(); renderDims();
    const body = S.pane === "detail" ? renderDetail() : S.pane === "cooc" ? renderCooc() : S.pane === "timeline" ? renderTimeline() : el("div", {},
      el("p", { class: "muted", text: `${fmt(rt)} models in ${fmt(rs)} shapes. Families by model type; open a family for all its shapes, a shape for its repos; a name opens its card.` }), modelsList(R));
    pane.replaceChildren(body);
  }

  // ---- download
  function download(kind) {
    compute();
    const rows = [];
    const facetIds = FS.map(f => f.id);
    for (let k = 0; k < n; k++) {
      if (!R[k]) continue;
      const r = { repo: mRepo[k], shape: N[mNode[k]].name, model_type: TYPES[M.mt[k]], first_on_hub: M.date[k], downloads: M.dl[k], params: mParams[k],
        variants: VS.filter(v => v.M[k]).map(v => v.id).join(" "), unknown: VS.filter(v => v.U && v.U[k]).map(v => v.id).join(" ") };
      for (const fid of facetIds) { const f = FBY.get(fid); if (f.pk) { for (const q of f.kinds) { const x = f.v[q][k]; r[fid + "_" + q] = x == null ? "" : x; } }
        else { const x = f.v[k]; r[fid] = x == null ? "" : x; } }
      rows.push(r);
    }
    let blob;
    if (kind === "json") blob = new Blob([JSON.stringify({ filters: hashParts("variants").join("&"), n: rows.length, models: rows }, null, 1)], { type: "application/json" });
    else {
      const cols = rows.length ? Object.keys(rows[0]) : ["repo"], q = v => { v = String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
      blob = new Blob([[cols.join(","), ...rows.map(r => cols.map(c => q(r[c])).join(","))].join("\n") + "\n"], { type: "text/csv" });
    }
    const a = el("a", { href: URL.createObjectURL(blob), download: "zoo-variants." + kind }); document.body.append(a); a.click(); a.remove();
  }

  // ---- permalink
  const E = encodeURIComponent;
  function hashParts(view) {
    const p = [];
    const filt = () => {
      if (S.inc.length) p.push("zi=" + S.inc.map(E).join(","));
      const nf = Object.entries(S.nf).map(([id, F]) => [id, F.k, isFinite(F.lo) ? F.lo : "", isFinite(F.hi) ? F.hi : "", F.ho ? 1 : 0].join("~"));
      if (nf.length) p.push("zn=" + nf.map(E).join(","));
      if (S.fam) p.push("zf=" + E(S.fam));
      if (S.from) p.push("zfrom=" + S.from);
      if (S.to) p.push("zto=" + S.to);
      if (S.pmin) p.push("zpmin=" + S.pmin);
      if (S.pmax) p.push("zpmax=" + S.pmax);
      if (S.fwk) p.push("zw=" + E(S.fwk));
      if (S.front) p.push("zfr=1");
      if (S.q) p.push("zq=" + E(S.q));
    };
    if (view === "variants") {
      if (S.comp !== DEF.comp) p.push("zc=" + S.comp);
      if (S.pane !== DEF.pane) p.push("zp=" + S.pane);
      if (S.vsel) p.push("zv=" + E(S.vsel));
      filt();
      if (S.sort !== DEF.sort || S.dir !== DEF.dir) p.push("zs=" + S.sort + (S.dir > 0 ? "" : "-"));
      if (S.carry) p.push("zm=1");
      if (S.pane === "cooc" && (S.ca !== S.comp || S.cb !== S.comp)) p.push("zca=" + S.ca, "zcb=" + S.cb);
      if (S.pane === "timeline" && S.ta) p.push("zta=" + S.ta);
      if (S.pane === "timeline" && S.tm !== "month") p.push("ztm=" + S.tm);
    } else if (view === "models" && S.carry && filtersOn()) { filt(); p.push("zm=1"); }
    return p;
  }
  function fromHash(P, dec) {
    if (!Object.keys(P).some(k => /^z[a-z]/.test(k))) { if (P.v === "variants") { Object.assign(S, JSON.parse(JSON.stringify(DEF))); dirty = true; } return; }
    Object.assign(S, JSON.parse(JSON.stringify(DEF)));
    if (P.zc && COMPS.some(c => c.id === P.zc)) S.comp = S.ca = S.cb = P.zc;
    if (P.zp && ["models", "detail", "cooc", "timeline"].includes(P.zp)) S.pane = P.zp;
    if (P.zv) { const v = dec(P.zv); if (VBY.has(v)) S.vsel = v; }
    if (P.zi) S.inc = P.zi.split(",").map(dec).filter(id => VBY.has(id));
    if (P.zn) for (const s of P.zn.split(",").map(dec)) { const [id, k, lo, hi, ho] = s.split("~"); const f = FBY.get(id); if (!f) continue;
      S.nf[id] = { k: f.pk && (k === "any" || f.kinds.includes(k)) ? k : "any", lo: lo === "" ? -Infinity : +lo, hi: hi === "" ? Infinity : +hi, ho: ho === "1" ? 1 : 0 }; }
    if (P.zf) S.fam = dec(P.zf);
    if (P.zfrom) S.from = P.zfrom;
    if (P.zto) S.to = P.zto;
    if (P.zpmin) S.pmin = +P.zpmin || 0;
    if (P.zpmax) S.pmax = +P.zpmax || 0;
    if (P.zw && FWN.includes(dec(P.zw))) S.fwk = dec(P.zw);
    if (P.zfr) S.front = true;
    if (P.zq) S.q = dec(P.zq);
    if (P.zs) { S.dir = P.zs.endsWith("-") ? -1 : 1; S.sort = P.zs.replace(/-$/, ""); if (!["dl", "date", "size"].includes(S.sort)) { S.sort = "dl"; S.dir = -1; } }
    if (P.zm) S.carry = true;
    if (P.zca && COMPS.some(c => c.id === P.zca)) S.ca = P.zca;
    if (P.zcb && COMPS.some(c => c.id === P.zcb)) S.cb = P.zcb;
    if (P.zta) S.ta = P.zta;
    if (P.ztm === "cum") S.tm = "cum";
    dirty = true;
  }
  // the Models table shows the carried filter as a chip
  function modelsNote() {
    const box = $("m-zv");
    if (!box) return;
    if (S.carry && filtersOn()) {
      compute();
      box.replaceChildren(el("span", { class: "chip on" }, `Variants filter: ${fmt(total(R))} models`, el("button", { type: "button", class: "x", "aria-label": "Stop applying the Variants filter", text: "×",
        onclick: () => { S.carry = false; dirty = true; api.renderModels(); api.writeHash(); } })),
        el("button", { type: "button", class: "btn small", text: "Edit in Variants", onclick: () => { api.setView("variants"); api.writeHash(); } }));
    } else box.replaceChildren();
  }
  return { render, hashParts, fromHash, keepShape, modelsNote, state: S, data: X, R: () => { compute(); return R; } };
}
return { init };
})();
