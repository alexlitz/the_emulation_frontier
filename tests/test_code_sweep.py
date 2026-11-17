"""The code-diff sweep (verify/sweep) covers the snapshot: every model type has an inventory entry, every hunk a
classification. A new architecture in graph/hub_10k.json fails here until verify/sweep/sweep.py has inventoried it
and its hunks are classified (rules.py or a reviews/*.jsonl line), then build_report.py is rerun."""
import json
import os

import pytest

from graph import model_graph as MG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SW = os.path.join(ROOT, "verify", "sweep")
LABELS = {"MAPPED", "IRRELEVANT", "UNMAPPED"}


@pytest.fixture(scope="module")
def inventory():
    return json.load(open(os.path.join(SW, "inventory.json")))


@pytest.fixture(scope="module")
def hunks():
    return [json.loads(l) for l in open(os.path.join(SW, "hunks.jsonl")) if l.strip()]


def snapshot_types():
    out = {}
    for m in json.load(open(os.path.join(ROOT, "graph", "hub_10k.json"))):
        cfg = m.get("config") or {}
        outer = str(cfg.get("model_type") or "") or "<none>"
        lm = MG._lm_unwrap(cfg)
        inner = str(lm.get("model_type") or "") if isinstance(lm, dict) else ""
        for t in (outer, inner or outer):
            out.setdefault(t, m["id"])
    return out


def test_every_snapshot_type_inventoried(inventory):
    types = inventory["types"]
    missing = {t: m for t, m in snapshot_types().items() if t not in types}
    assert not missing, f"model types with no sweep inventory entry (rerun verify/sweep/sweep.py): {missing}"


def test_inventory_entries_resolved(inventory):
    for t, e in inventory["types"].items():
        assert e["code"] or e["no_code"], t
        for c in e["code"]:
            assert c["variant"].split(":")[0] in ("tf5", "tf4", "hub", "tfmain", "ext"), (t, c)
            if c["variant"] in inventory["var_models"] and not c["variant"].startswith("tf4"):
                assert "how" in c, (t, c["variant"], "variant never swept")


def test_every_hunk_classified(hunks):
    assert hunks
    seen = set()
    bad = []
    for h in hunks:
        assert h["id"] not in seen, h["id"]
        seen.add(h["id"])
        if h.get("label") not in LABELS or not h.get("reason"):
            bad.append(h["id"])
        elif h["label"] == "UNMAPPED" and not (h.get("finding") and h.get("direction")):
            bad.append(h["id"])
    assert not bad, f"{len(bad)} hunks without a classification, e.g. {bad[:10]}"


def test_every_swept_variant_has_models(inventory, hunks):
    variants = {o.split(" ", 1)[0] for h in hunks for o in h["occ"]}
    assert variants <= set(inventory["var_models"]), variants - set(inventory["var_models"])


def test_findings_ranked(hunks):
    fl = json.load(open(os.path.join(SW, "findings.json")))
    keys = {f["key"] for f in fl}
    assert keys == {h["finding"] for h in hunks if h.get("label") == "UNMAPPED"}
    order = [(-f["live_models"], -f["placed_models"], -f["models"]) for f in fl]
    assert order == sorted(order)


def test_nocode_hunt_recorded(inventory):
    """Every type the no-code hunt covers (verify/sweep/nocode_sources.json) has its sources in the inventory, and a
    type still without code says what was tried."""
    nc = json.load(open(os.path.join(SW, "nocode_sources.json")))
    for t, e in nc.items():
        e = nc[e["same_as"]] if e.get("same_as") else e
        assert e["status"] in ("code", "wrapper", "paper-only", "missing"), t
        if t not in inventory["types"]:
            continue
        got = {c["variant"] for c in inventory["types"][t]["code"] if c.get("nocode_hunt")}
        assert got == {sv["variant"] for sv in e.get("sweep", [])}, t
        if e["status"] in ("missing", "paper-only"):
            assert e.get("tried"), t
    for t, e in inventory["types"].items():
        if e["no_code"]:
            assert t in nc, f"{t} has no code and no no-code hunt entry (nocode_sources.json)"
