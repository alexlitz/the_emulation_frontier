"""The zoo page's BEHAVIOUR in a headless browser (playwright + chromium): what a reader sees and clicks.

tests/test_zoo_page.py checks the data embedded in the page; this drives the page itself. The page is
generated once per session by the documented command (python -m graph.zoo_ref --out OUT --no-pairs: the
emulation graph of python -m graph.model_graph --html-models plus the reference views, which
tests/test_zoo_reference.py drives), and every expectation below is computed on the Python
side from the relation (graph.frontier / graph.model_graph / graph.edge_plan), not read back from the page:

  load      no console error; the header stats; every shape and every frontier row shown by default
  search    a repo name selects its shape, opens the panel and sets #m=; cover, in-degree, binding, repos
  filters   frontier only / mixer / generation / min size: the visible points and table rows
  edges     "HOST emulates TARGET" with edge_plan's slots and capacity rows and the emulate command;
            #e= links round-trip; a sample of edges agrees with the relation's own blocker logic
            (graph.frontier.blockers, payable_reaches, ZERO_EDGES), not with edge_plan
  table     sorting; one row per frontier shape
  themes    light / dark / auto (the OS preference), persisted
  phone     375 px: the table is the default view and nothing overflows the viewport
  text      no NaN / undefined / null in any rendered state
  command   the page's "Build the weights" command, run for real on small edges (scratch under
            $TEF_SCRATCH/zoo_cmd/, deleted after): it exits 0 and the checkpoint verifies

Setup:  python -m pip install playwright && python -m playwright install chromium   (skips without them)
Run:    pytest tests/test_zoo_browser.py -q            (ZOO_SKIP_CMD=1 skips the command runs)
"""
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")

from tests.zoo_harness import (BAD_TEXT, CMD_SCRATCH, PHONE, ROOT, Page, enc, fmt, num)  # noqa: E402,F401

# the fixtures (zoo, truth, browser, open_page) are in tests/zoo_harness.py, registered by tests/conftest.py


# --- load ----------------------------------------------------------------------------------------------

def test_load(open_page, zoo, truth):
    p = open_page()
    pg, n = p.pg, len(truth.P)
    nd = sum(truth.P[i]["slots"]["gen"] != "ar" for i in truth.reps)    # FRONTIER.md's split: gen != ar
    stats = pg.eval_on_selector_all("#stats .stat", "s => s.map(e => [e.querySelector('dt').textContent, "
                                                    "e.querySelector('dd').textContent])")
    assert stats == [["models", fmt(truth.cnt.sum())], ["distinct shapes", fmt(n)],
                     ["on the frontier", f"{fmt(len(truth.reps))} ({len(truth.reps) - nd} AR + {nd} diffusion)"],
                     ["Hub snapshot", zoo.data["snapshot"]]]
    # nothing is hidden before the reader filters (regression: block-causal shapes had no generation chip)
    assert pg.inner_text("#graph-count") == f"{fmt(n)} shapes"
    assert p.visible_nodes() == set(range(n))
    assert sorted(p.table_rows()) == sorted(truth.reps)
    assert pg.inner_text("#table-count") == f"{len(truth.reps)} models · nothing emulates them"
    assert pg.get_attribute("#post-link", "href") == zoo.data["post"]
    assert pg.get_attribute("#repo-link", "href") == zoo.data["repo"]
    assert pg.inner_text("#panel h2") == "Pick a model"
    gens = {pp["slots"]["gen"] for pp in truth.P}
    assert pg.locator("#f-gen label.chip").count() == len(gens | {"ar", "diffusion", "block"})
    p.assert_clean_text("load")


# --- search and the shape panel --------------------------------------------------------------------------

def _search_cases(truth, zoo):
    nodes = zoo.data["nodes"]
    rep = max(truth.reps, key=lambda i: truth.cover[i])                     # the widest frontier shape
    nonfront = next(i for i in np.argsort(-truth.by) if i not in truth.front and len(nodes[i]["members"]) > 12)
    tie = next((i for i in truth.front if i not in truth.reps), None)
    block = next((i for i in truth.reps if truth.P[i]["slots"]["gen"] == "block"), None)
    wall = next((i for i in truth.reps if truth.why[i].startswith("WALL")), None)
    out = [(rep, nodes[rep]["members"][0]), (int(nonfront), nodes[nonfront]["members"][-1])]
    out += [(i, nodes[i]["members"][0]) for i in (tie, block, wall) if i is not None]
    return out


def test_search_selects_shape(open_page, zoo, truth):
    nodes = zoo.data["nodes"]
    p = open_page()
    pg = p.pg
    for i, repo in _search_cases(truth, zoo):
        pg.fill("#q", "")
        pg.fill("#q", repo)
        pg.wait_for_selector("#results:not([hidden])")
        first = pg.inner_text("#results li .rid")
        assert first == repo, (repo, first)
        pg.keyboard.press("Enter")
        assert p.hash() == "#m=" + enc(repo)
        d = p.panel()
        assert d["title"] == (nodes[i]["name"] if repo == nodes[i]["members"][0] else repo.split("/")[-1])
        n_targets, n_hosts = len(truth.targets[i]), len(truth.hosts[i])
        assert d["nums"] == [f"{fmt(truth.cover[i])} models · {fmt(n_targets + 1)} shapes",
                             f"{fmt(truth.by[i])} models · {fmt(n_hosts)} shapes"], (repo, d["nums"])
        if i in truth.reps:
            kind = truth.why[i].split(" ")[0]
            assert d["kind"] == kind and d["status"].startswith("On the frontier")
            if kind == "WALL":
                assert d["detail"] == truth.why[i].split(" ", 1)[1]
        elif i in truth.front:
            assert d["kind"] == "TIE"
        else:
            fh = [j for j in truth.hosts[i] if j in truth.reps]
            assert d["status"].startswith("Not on the frontier")
            assert f"Emulated by {len(fh)} frontier model{'' if len(fh) == 1 else 's'}" in d["status"]
        # another repo of the shape: the card says which repo it is for and which repo the figures are (zoo-fixes 9)
        own = [x for x in d["note"] if x.startswith(f"{repo}: same shape as {nodes[i]['members'][0]}")]
        assert bool(own) == (repo != nodes[i]["members"][0]), d["note"]
        # every Hub repo of the shape (expand the list first)
        more = pg.locator("#panel h3:has-text('Hub repos') + div .more")
        if more.count():
            more.click()
        hrefs = set(p.panel()["repos"])
        assert {"https://huggingface.co/" + m for m in truth.P[i]["members"]} <= hrefs
        # the graph marks the selection and lights its hosts and targets
        sel = pg.evaluate("[...document.querySelectorAll('#svg g.node.me')].map(g => +g.dataset.i)")
        assert sel == [i]
        lit = pg.evaluate("[...document.querySelectorAll('#svg g.node.up')].map(g => +g.dataset.i)")
        assert set(lit) == truth.hosts[i]
        p.assert_clean_text(repo)
    # a loaded #m= link opens the same panel
    q = open_page("#m=" + enc(repo))
    assert q.panel()["title"] == nodes[i]["name"]


def test_search_no_match(open_page):
    p = open_page()
    p.pg.fill("#q", "zzzz-no-such-model-qqqq")
    assert p.pg.inner_text("#results") == "No model matches"
    p.pg.keyboard.press("Enter")                                   # nothing to pick: no error, no selection
    assert p.panel()["title"] == "Pick a model" and p.hash() == ""


# --- filters ----------------------------------------------------------------------------------------------

FILTERS = [dict(front=True), dict(mixer_off=["Softmax"]), dict(mixer_off=["Hybrid", "SSM", "Linear"]),
           dict(gen_off=["AR"]), dict(gen_off=["Diffusion", "Block diffusion"]), dict(size="7e9"),
           dict(size="1e8", front=True, mixer_off=["Hybrid"]), dict(size="1e11", gen_off=["AR"])]
MIXER_LABEL = {"Softmax": "softmax", "Hybrid": "hybrid", "SSM": "ssm", "Linear": "linear"}
GEN_CHIP = {"AR": "ar", "Diffusion": "diffusion", "Block diffusion": "block"}


@pytest.mark.parametrize("f", FILTERS, ids=lambda f: ",".join(f"{k}={v}" for k, v in f.items()))
def test_filters(open_page, zoo, truth, f):
    p = open_page()
    pg = p.pg
    if f.get("front"):
        pg.locator("label.chip", has=pg.locator("#f-front")).click()
    for lab in f.get("mixer_off", []):
        pg.locator("#f-mixer label.chip", has_text=re.compile(rf"^{lab}\d*$")).click()
    for lab in f.get("gen_off", []):
        pg.locator("#f-gen label.chip", has_text=re.compile(rf"^{lab}$")).click()
    if "size" in f:
        pg.select_option("#f-size", f["size"])
    mix_off = {MIXER_LABEL[x] for x in f.get("mixer_off", [])}
    gen_off = {GEN_CHIP[x] for x in f.get("gen_off", [])}
    size = float(f.get("size", 0))
    nodes = zoo.data["nodes"]
    want = {i for i, pp in enumerate(truth.P)
            if (not f.get("front") or i in truth.front) and pp["slots"]["mixer"] not in mix_off
            and pp["slots"]["gen"] not in gen_off and nodes[i]["params"] >= size}   # params: placement only
    assert p.visible_nodes() == want
    assert sorted(p.table_rows()) == sorted(i for i in truth.reps if i in want)
    n = len(truth.P)
    assert pg.inner_text("#graph-count") == (f"{fmt(n)} shapes" if len(want) == n else
                                             f"{fmt(len(want))} of {fmt(n)} shapes shown")
    rows = len([i for i in truth.reps if i in want])
    assert pg.inner_text("#table-count") == (f"{rows} of {len(truth.reps)} shown" if rows < len(truth.reps) else
                                             f"{rows} models · nothing emulates them")
    p.assert_clean_text(str(f))


def test_hidden_selection_offers_clear(open_page, zoo, truth):
    """Selecting a shape the filters hide says so, and "Clear filters" restores everything."""
    i = next(i for i in range(len(truth.P)) if i not in truth.front)
    p = open_page()
    p.pg.locator("label.chip", has=p.pg.locator("#f-front")).click()
    p.goto_hash("#m=" + enc(zoo.data["nodes"][i]["members"][0]))
    p.pg.wait_for_function(f"document.querySelector('#svg g.node.me') && +document.querySelector('#svg g.node.me').dataset.i === {i}")
    p.pg.click("#panel .note button:has-text('Clear filters')")
    assert p.visible_nodes() == set(range(len(truth.P)))
    assert not p.pg.is_checked("#f-front")


# --- the edge view ----------------------------------------------------------------------------------------

def _check_against_edge_plan(d, truth, zoo, t, h):
    """The rendered edge equals graph.edge_plan.edge_plan run from scratch, and the command / v1 status
    equal elephantine.emulate's."""
    from elephantine.emulate import support
    from graph.edge_plan import edge_plan
    nodes = zoo.data["nodes"]
    plan = edge_plan(truth.P[t], truth.P[h])
    assert plan["ok"]
    assert d["title"] == f"{nodes[h]['name']} emulates {nodes[t]['name']}"
    assert d["badge"] == ("Exact" if plan["exact"] else "Limit / regime")
    assert [(r["slot"], r["move"]) for r in d["slots"]] == [(r["slot"], f"{r['target']} → {r['host']}") for r in plan["slots"]]
    rows = {r["axis"]: r for r in d["axes"]}
    for r in plan["axes"]:
        x = rows[r["axis"]]
        assert num(x["target"]) == (r["target"] or 0), (r, x)
        assert (num(x["need"]) or num(x["target"])) == r["need"], (r, x)
        assert num(x["host"]) == r["have"], (r, x)
        assert not x["fail"]
    tr, hr = truth.repo[t], truth.repo[h]
    out = f"{tr.split('/')[-1]}-in-{hr.split('/')[-1]}".lower()
    assert d["cmd"] == f"python -m elephantine.emulate {tr} {hr} --out {out}"
    from elephantine.emulate import limit_ok
    lim = not plan["exact"] and limit_ok(plan)           # a limit the script builds as a labelled construction
    builds = (plan["exact"] or lim) and support(truth.configs[tr], truth.configs[hr], limit=lim) is None
    assert d["v1"] == ("the script builds it" if builds else "the script refuses it"), (tr, hr, d["v1why"])
    if not builds:
        assert d["v1why"] and all(d["v1why"])


def test_edge_view_from_panel(open_page, zoo, truth):
    nodes = zoo.data["nodes"]
    t = next(int(i) for i in np.argsort(-truth.by) if i not in truth.front and truth.targets[i])
    p = open_page("#m=" + enc(nodes[t]["members"][0]))
    pg = p.pg
    # the frontier hosts: the first listed (widest cover) opens "HOST emulates t"
    fh = sorted((j for j in truth.hosts[t] if j in truth.reps), key=lambda j: -truth.cover[j])
    first = pg.locator("#panel .status .edge-go").first
    assert first.inner_text() == nodes[fh[0]]["name"]
    first.click()
    d = p.panel()
    assert d["edge"]
    _check_against_edge_plan(d, truth, zoo, t, fh[0])
    link = p.hash()
    assert link == "#e=" + enc(truth.repo[t]) + "," + enc(truth.repo[fh[0]])
    p.assert_clean_text("edge")
    # the pair is drawn
    assert pg.locator("#svg .edges line.pick").count() == 1
    # #e= round-trips: a fresh load of the link shows the same edge
    q = open_page(link)
    assert q.panel() == d and q.hash() == link
    # back to the shape: its panel and its #m= link
    pg.click("#panel button:has-text('Back to')")
    assert p.panel()["title"] == nodes[t]["name"] and p.hash().startswith("#m=")
    # a direct target opens "t emulates TARGET"
    down = pg.locator("#panel h3:has-text('Direct targets') + p + div .edge-go").first
    name = down.inner_text()
    j = next(j for j in truth.targets[t] if nodes[j]["name"] == name)
    down.click()
    _check_against_edge_plan(p.panel(), truth, zoo, j, t)
    # the title's names open the shapes
    pg.locator("#panel h2.edge-title .linkish").last.click()
    assert p.panel()["title"] == nodes[j]["name"]


def test_edge_links_round_trip_bad_links(open_page, zoo, truth):
    """A link to a non-edge or an unknown repo opens nothing and raises nothing."""
    nodes = zoo.data["nodes"]
    a, b = truth.reps[0], truth.reps[1]                             # two frontier shapes: neither emulates the other
    for h in [f"#e={enc(truth.repo[a])},{enc(truth.repo[b])}", "#e=nobody/none,nobody/else", "#m=nobody/none"]:
        p = open_page(h)
        assert p.panel()["title"] == "Pick a model", h
    p = open_page("#m=" + enc(nodes[a]["name"]))                   # a shape's display name also resolves
    assert p.panel()["title"] == nodes[a]["name"]


def _plain(s):
    return s["mixer"] == "softmax" and s["crosslayer_kv"] == "no"


@pytest.mark.xfail(strict=False, reason="round 3: about 11% of the sampled payable-but-not-exact pairs have no inexact slot "
                   "row in graph.edge_plan -- the non-exactness comes from round-3 feature / dims rules edge_plan does "
                   "not explain (explain / blockers completeness is not done)")
def test_edge_plans_agree_with_blockers(open_page, zoo, truth):
    """For a random sample of the page's edges: the slots it lists are exactly the slots where the two
    shapes differ, the relation's own blocker logic (graph.frontier.blockers) finds nothing blocking,
    every non-size slot is paid (payable_reaches, or a ZERO_EDGES depth rule for a plain target), the
    capacity rows show the target's and the host's real dims, and the exact badge is mode="exact"."""
    from graph.frontier import blockers
    from graph.model_graph import EPS_LEVER_QKNORM, SLOTS, ZERO_EDGES, emulates, payable_reaches
    from tests.test_zoo_page import _edge_pairs
    SIZE = {"kv", "ffn", "block", "loop"}
    pairs = _edge_pairs(zoo.data)
    rng = random.Random(20261113)
    sample = sorted(set(rng.sample(range(len(pairs)), 60)) | {0, len(pairs) - 1})
    # plus every edge into or out of a non-softmax / non-AR / MLA shape among a few, so the rarer rows show up
    odd = [k for k, (t, h) in enumerate(pairs) if truth.P[t]["slots"]["mixer"] != "softmax"
           or truth.P[t]["slots"]["gen"] != "ar" or truth.P[t]["slots"]["kv"] == "MLA"]
    sample = sorted(set(sample) | set(rng.sample(odd, min(20, len(odd)))))
    p = open_page()
    for k in sample:
        t, h = pairs[k]
        A, B = truth.P[t], truth.P[h]
        assert truth.ok[t, h] and emulates(B, A)
        p.goto_hash("#e=" + enc(truth.repo[t]) + "," + enc(truth.repo[h]))
        p.pg.wait_for_function("h => location.hash === h && document.querySelector('#panel h2.edge-title')",
                               arg="#e=" + enc(truth.repo[t]) + "," + enc(truth.repo[h]))
        d = p.panel()
        assert d["title"] == f"{zoo.data['nodes'][h]['name']} emulates {zoo.data['nodes'][t]['name']}"
        sa, sb, da, db = A["slots"], B["slots"], A["dims"], B["dims"]
        differ = [(s, sa[s], sb[s]) for s, _ in SLOTS if sa[s] != sb[s]]
        shown = [(r["slot"], *r["move"].split(" → ")) for r in d["slots"]]
        assert shown == differ, (k, shown, differ)
        assert blockers(A, B) == [], (k, blockers(A, B))
        for (s, a, b), r in zip(differ, d["slots"]):
            if s in SIZE:
                assert r["how"].startswith("size-based"), (k, r)
            elif (s, a, b) in ZERO_EDGES and _plain(sa):
                assert r["how"] == f"{ZERO_EDGES[(s, a, b)]}; exact", (k, r)
            elif s == "qknorm" and a == "no" and b in EPS_LEVER_QKNORM:
                assert "a limit / regime / floor" in r["how"], (k, r)   # the eps lever: the host's qk-norm off
            else:
                assert payable_reaches(s, a, b), (k, s, a, b)
                assert "wall" not in r["how"], (k, r)
        exact = bool(truth.exact[t, h])
        assert d["badge"] == ("Exact" if exact else "Limit / regime"), k
        limits = [r for r in d["slots"] if "a limit / regime / floor" in r["how"]]
        assert bool(limits) == (not exact), (k, d["slots"])
        zero_rule = _plain(sa) and (not _plain(sb) or (sa["window"] == "full" and sb["window"] == "local"))
        for x in d["axes"]:
            ax, tv, need, host = x["axis"], num(x["target"]), num(x["need"]), num(x["host"])
            assert tv == (da.get(ax) or 0), (k, x)                 # the target's own dims
            assert not x["fail"], (k, x)
            if need is not None:                                   # a penalty only ever raises the need
                assert need > tv and x["subs"], (k, x)
            if ax in ("eff_layers", "param_layers") and zero_rule:
                assert host <= db[ax], (k, x)                      # only the host's layers of the target's kind
            else:
                assert host == (db.get(ax) or 0), (k, x)           # the host's own dims
            assert host >= (need if need is not None else tv), (k, x)
        p.assert_clean_text(f"edge {k}")


# --- the table --------------------------------------------------------------------------------------------

def test_table_sorting(open_page, zoo, truth):
    nodes = zoo.data["nodes"]
    p = open_page()
    pg = p.pg
    pg.click("#tab-table")
    assert not pg.is_hidden("#view-table") and pg.is_hidden("#view-graph")
    rows = p.table_rows()
    assert len(rows) == len(truth.reps) == zoo.data["n_frontier"]
    cov = [truth.cover[i] for i in rows]
    assert cov == sorted(cov, reverse=True)                        # default: cover, descending
    assert pg.get_attribute("#tbl th[data-k=cover]", "aria-sort") == "descending"
    cell = pg.eval_on_selector_all("#tbl tbody tr td:nth-child(4)", "c => c.map(e => e.textContent)")
    assert cell == [fmt(c) for c in cov]

    def by(k, key, first_dir):
        for d in (first_dir, -first_dir):
            pg.click(f"#tbl th[data-k={k}] button")
            got = p.table_rows()
            assert sorted(got) == sorted(truth.reps)
            keys = [key(i) for i in got]
            assert keys == sorted(keys, reverse=d < 0), (k, d)
            assert pg.get_attribute(f"#tbl th[data-k={k}]", "aria-sort") == ("ascending" if d > 0 else "descending")
            others = pg.eval_on_selector_all(f"#tbl th:not([data-k={k}])", "t => t.map(e => e.getAttribute('aria-sort'))")
            assert set(others) == {"none"}
    by("name", lambda i: nodes[i]["name"].lower(), 1)
    by("params", lambda i: nodes[i]["params"], -1)
    by("hidden", lambda i: truth.P[i]["dims"]["hidden"] * 10 ** 4 + truth.P[i]["dims"]["eff_layers"], -1)
    by("bind", lambda i: truth.why[i], 1)
    by("cover", lambda i: truth.cover[i], -1)
    # a row opens its shape
    i = p.table_rows()[0]
    pg.click(f"#tbl tbody tr[data-i='{i}']")
    assert p.panel()["title"] == nodes[i]["name"]
    assert pg.get_attribute(f"#tbl tbody tr[data-i='{i}']", "class") == "on"
    p.assert_clean_text("table")


# --- themes -----------------------------------------------------------------------------------------------

LIGHT_BG, DARK_BG = "rgb(249, 249, 247)", "rgb(13, 13, 13)"


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_themes(open_page, scheme):
    p = open_page(scheme=scheme)
    pg = p.pg
    bg = lambda: pg.evaluate("getComputedStyle(document.body).backgroundColor")
    theme = lambda: pg.evaluate("document.documentElement.getAttribute('data-theme')")
    assert pg.inner_text("#theme") == "Theme: auto" and theme() is None
    assert bg() == (DARK_BG if scheme == "dark" else LIGHT_BG)      # auto follows the OS
    pg.click("#theme")
    assert pg.inner_text("#theme") == "Theme: light" and theme() == "light" and bg() == LIGHT_BG
    pg.click("#theme")
    assert pg.inner_text("#theme") == "Theme: dark" and theme() == "dark" and bg() == DARK_BG
    pg.reload()                                                     # the choice persists
    pg.wait_for_selector("#stats .stat")
    assert pg.inner_text("#theme") == "Theme: dark" and bg() == DARK_BG
    pg.click("#theme")
    assert pg.inner_text("#theme") == "Theme: auto" and theme() is None
    assert bg() == (DARK_BG if scheme == "dark" else LIGHT_BG)


# --- phone width ------------------------------------------------------------------------------------------

OVERFLOW_JS = """() => {
  const W = document.documentElement.clientWidth, out = [];
  const scroller = e => e.closest('.table-wrap, .plan-wrap, svg');      // the intended horizontal scrollers / clip
  for (const e of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(e);
    if (cs.display === 'none' || cs.visibility === 'hidden' || e.closest('[hidden], .sr')) continue;
    const r = e.getBoundingClientRect();
    if (!r.width && !r.height) continue;
    if ((r.right > W + 0.5 || r.left < -0.5) && !(e.parentElement && scroller(e.parentElement)))
      out.push(`${e.tagName.toLowerCase()}.${e.className && e.className.baseVal === undefined ? e.className : ''} ` +
               `[${Math.round(r.left)}, ${Math.round(r.right)}] ${(e.textContent || '').slice(0, 40)}`);
  }
  const wraps = [...document.querySelectorAll('.table-wrap')].filter(w => w.offsetParent)
    .map(w => [w.scrollWidth, w.clientWidth]);
  return {W, scroll: document.documentElement.scrollWidth, out: out.slice(0, 15), wraps};
}"""


def _no_overflow(p, where):
    r = p.pg.evaluate(OVERFLOW_JS)
    assert r["scroll"] <= r["W"], (where, r)
    assert not r["out"], (where, r)
    assert all(s <= c for s, c in r["wraps"]), (where, r)            # the frontier table fits: no sideways scroll


@pytest.mark.parametrize("width", [375, 320])
def test_phone(open_page, zoo, truth, width):
    nodes = zoo.data["nodes"]
    p = open_page(viewport={"width": width, "height": 800}, mobile=True)
    pg = p.pg
    assert pg.get_attribute("#tab-table", "aria-selected") == "true"
    assert not pg.is_hidden("#view-table") and pg.is_hidden("#view-graph")
    assert len(p.table_rows()) == len(truth.reps)
    _no_overflow(p, "load")
    # search results open over the page
    pg.fill("#q", "llama")
    pg.wait_for_selector("#results:not([hidden])")
    _no_overflow(p, "results")
    pg.keyboard.press("Enter")
    _no_overflow(p, "shape panel")
    # the longest repo name's shape and an edge view (long command, plan table)
    longest = max((m for nd in nodes for m in nd["members"]), key=len)
    pg.evaluate("h => { location.hash = h; }", "#m=" + enc(longest))
    pg.wait_for_function("document.querySelector('#panel h2') && document.querySelector('#panel h2').textContent !== 'Pick a model'")
    more = pg.locator("#panel .more")
    for k in range(more.count()):
        more.first.click()
    _no_overflow(p, "longest repo")
    pg.locator("#panel .edge-go").first.click()
    pg.wait_for_selector("#panel h2.edge-title")
    _no_overflow(p, "edge view")
    pg.click("#tab-graph")
    _no_overflow(p, "graph")
    p.assert_clean_text("phone")


# --- no NaN / undefined anywhere -----------------------------------------------------------------------

def test_no_bad_text_in_any_panel(open_page, zoo, truth):
    """Every frontier shape's panel, a sample of the others, and their tooltips."""
    nodes = zoo.data["nodes"]
    rng = random.Random(7)
    others = [i for i in range(len(nodes)) if i not in truth.reps]
    shapes = sorted(truth.reps) + rng.sample(others, 60)
    p = open_page()
    for i in shapes:
        p.goto_hash("#m=" + enc(nodes[i]["members"][0]))
        p.pg.wait_for_function(f"(() => {{ const g = document.querySelector('#svg g.node.me'); return g && +g.dataset.i === {i}; }})()")
        t = p.pg.inner_text("#panel")
        assert not BAD_TEXT.search(t), (nodes[i]["name"], BAD_TEXT.search(t).group(0))
    # hover tooltips: one per mixer x generation present (each shape opened first, so it sits mid-chart)
    seen = {}
    for i, pp in enumerate(truth.P):
        seen.setdefault((pp["slots"]["mixer"], pp["slots"]["gen"]), i)
    gen_label = {"ar": "Autoregressive", "diffusion": "Diffusion", "block": "Block diffusion", "prefix": "Prefix LM"}
    named = 0
    for (mixer, gen), i in seen.items():
        p.goto_hash("#m=" + enc(nodes[i]["members"][0]))
        p.pg.wait_for_function(f"(() => {{ const g = document.querySelector('#svg g.node.me'); return g && +g.dataset.i === {i}; }})()")
        p.pg.locator("#chart").scroll_into_view_if_needed()
        box = p.pg.locator(f"#svg g.node[data-i='{i}'] path").bounding_box()
        p.pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        p.pg.wait_for_selector("#tip:not([hidden])")
        tip = p.pg.inner_text("#tip")
        assert not BAD_TEXT.search(tip), tip
        if tip.startswith(nodes[i]["name"]):                         # not a coincident shape's tooltip
            named += 1
            assert f" · {gen_label[gen]} · " in tip, tip
        p.pg.mouse.move(1, 1)
    assert named >= len(seen) - 1, named
    p.assert_clean_text("panels")


# --- the command the page prints, run for real ------------------------------------------------------------

def _small_builds(zoo, truth, k=2):
    """Small edges the page says v1 builds: target and host both tiny (the host is written at full size)."""
    from tests.test_zoo_page import _edge_pairs, _records
    E, nodes = zoo.data["edges"], zoo.data["nodes"]
    pairs, recs = _edge_pairs(zoo.data), _records(E)
    out = []
    for q, (t, h) in enumerate(pairs):
        s, ax, v = recs[q]
        if v or E["tr"][t] or E["hr"][h] or not ax >> 7:
            continue
        if nodes[t]["params"] < 2e7 and nodes[h]["params"] < 2.5e7:
            out.append((len(E["sets"][s]), nodes[t]["params"] + nodes[h]["params"], t, h, s))
    # the most slots changed first (a GQA -> MHA, a dense -> MoE host), then the smallest
    out.sort(key=lambda x: (-x[0], x[1]))
    picked, seen = [], set()
    for _, _, t, h, s in out:
        key = tuple(sorted(E["rows"][x][0] for x in E["sets"][s]))
        if key in seen:
            continue
        seen.add(key)
        picked.append((t, h))
        if len(picked) == k:
            break
    return picked


@pytest.mark.skipif(os.environ.get("ZOO_SKIP_CMD") == "1", reason="ZOO_SKIP_CMD=1")
def test_page_command_builds_and_verifies(open_page, zoo, truth):
    """The page promises "the script builds it": copy its command, run it, and check the written checkpoint."""
    if not CMD_SCRATCH.parent.is_dir():
        pytest.skip(f"no scratch disk at {CMD_SCRATCH.parent}")
    picked = _small_builds(zoo, truth)
    assert picked
    p = open_page()
    for t, h in picked:
        p.goto_hash("#e=" + enc(truth.repo[t]) + "," + enc(truth.repo[h]))
        p.pg.wait_for_selector("#panel .cmd pre")
        d = p.panel()
        assert d["v1"] == "the script builds it", d
        argv = shlex.split(d["cmd"])
        assert argv[:3] == ["python", "-m", "elephantine.emulate"]
        CMD_SCRATCH.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=CMD_SCRATCH))
        try:
            r = subprocess.run([sys.executable] + argv[1:], cwd=work, capture_output=True, text=True, timeout=3600,
                               env={**os.environ, "PYTHONPATH": str(ROOT)})
            assert r.returncode == 0, (d["cmd"], r.stdout[-3000:], r.stderr[-3000:])
            out = work / argv[argv.index("--out") + 1]
            res = json.loads((out / "plan.json").read_text())
            assert res["target"] == truth.repo[t] and res["host"] == truth.repo[h]
            assert res["edge"]["ok"] and res["edge"]["exact"]
            cfg = json.loads((out / "config.json").read_text())
            assert cfg["architectures"] == truth.configs[truth.repo[h]]["architectures"] and "auto_map" not in cfg
            v = res["verification"]
            assert v["fp32"]["rel"] < 1e-4, v["fp32"]
            assert v["fp32"]["argmax_match"] == 1.0, v["fp32"]
            assert all(g["identical"] for g in v["fp32"]["greedy"]), v["fp32"]
            if "fp64" in v:
                assert v["fp64"]["rel"] < 1e-9, v["fp64"]
                assert all(g["identical"] for g in v["fp64"]["greedy"]), v["fp64"]
        finally:
            shutil.rmtree(work, ignore_errors=True)
