// Node runner for tests/test_arch_metrics_js.py: reads a JSON file of cases, runs the JS port on each, writes the
// results. Usage: node tests/arch_metrics_js_runner.js <tables.json> <cases.json> <out.json>
// A case is {name, raw, hub, resolved?}; `resolved` = [d, stock] injects Python's resolve() output (diagnostics only).
"use strict";
const fs = require("fs");
const path = require("path");
const AM = require(path.join(__dirname, "..", "graph", "templates", "arch_metrics.js"));
const [tablesPath, casesPath, outPath] = process.argv.slice(2);
AM.setTables(JSON.parse(fs.readFileSync(tablesPath, "utf8")));
const cases = AM.parseConfig(fs.readFileSync(casesPath, "utf8"));
const out = {};
for (const c of cases) {
  try {
    out[c.name] = { ok: AM.metrics(c.raw, c.hub === undefined ? null : c.hub, null, c.resolved ? { resolved: c.resolved } : null) };
  } catch (e) {
    out[c.name] = { error: String(e && e.message || e), stack: String(e && e.stack || "").split("\n").slice(0, 4).join(" | ") };
  }
}
fs.writeFileSync(outPath, JSON.stringify(out));
