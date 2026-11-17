// Node runner for tests/test_arch_breakdown_js.py: reads a JSON file of cases, runs the JS port of graph/arch_breakdown.py
// on each, writes the results. Usage: node tests/arch_breakdown_js_runner.js <tables.json> <cases.json> <out.json>
// A case is {name, raw} (the JS plans the raw config: the lookup path) or {name, plan} (a plan as the page ships it).
"use strict";
const fs = require("fs");
const path = require("path");
const AM = require(path.join(__dirname, "..", "graph", "templates", "arch_metrics.js"));
const BD = require(path.join(__dirname, "..", "graph", "templates", "arch_breakdown.js"));
const [tablesPath, casesPath, outPath] = process.argv.slice(2);
BD.setTables(JSON.parse(fs.readFileSync(tablesPath, "utf8")));
const cases = AM.parseConfig(fs.readFileSync(casesPath, "utf8"));
const out = {};
for (const c of cases) {
  try {
    const p = c.plan ? c.plan : BD.plan_ordered(c.raw);
    if (c.head_stored) p.head_stored = true;
    out[c.name] = { ok: BD.breakdown(p) };
  } catch (e) {
    out[c.name] = { error: String(e && e.message || e), stack: String(e && e.stack || "").split("\n").slice(0, 4).join(" | ") };
  }
}
fs.writeFileSync(outPath, JSON.stringify(out));
