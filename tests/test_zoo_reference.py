"""The zoo page's REFERENCE views in a headless browser: spec cards, block diagrams, the models table, compare,
the timeline, glossary popovers, permalinks and keys. The page is the session page of tests/test_zoo_browser.py
(python -m graph.zoo_ref --out OUT --no-pairs; ZOO_HTML=path reuses one), and every expectation is computed on
the Python side from the snapshot and the classifier (graph.zoo_ref.spec_of, graph/model_meta.json, the relation),
not read back from the page:

  data      the embedded block is spec_of() of each shape's most-downloaded repo, the Hub dates and counts
  card      the spec rows of a sample of shapes (dense, GQA, MLA + MoE, sliding, sandwich, hybrid, SSM), the
            "code" marks exactly on the fields a classifier code table set, no NaN / undefined / null
  diagram   the block diagram's norms sit where the placement says, the stack shows the depth, the head says
            tied; the layer strip has one cell per layer of each kind
  models    one row per shape; family / feature / date filters equal the Python filter; sorting; #v=models&...
            permalinks round-trip; arrow keys, Enter and c
  compare   #c= links; the highlighted rows are the rows whose values differ; "A emulates B" is the relation;
            the blocker reasons (lazy zoo_pairs.json over http) are graph.frontier.blockers
  timeline  each feature's first shape and date
  glossary  a term's popover, Escape, "Open in the glossary" and its #g= link
  passes    the metrics / frameworks sections (synthetic files in the documented schema) render their numbers
  fallback  a page without the reference block hides its tabs and still works
  phone     the card, the models table and compare fit a 375 px screen

Run:  pytest tests/test_zoo_reference.py -q -n 2
"""
import functools
import http.server
import json
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")

from tests.test_zoo_browser import _no_overflow  # noqa: E402
from tests.zoo_harness import BAD_TEXT, PHONE, Page, enc, fmt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ref(zoo):
    m = re.search(r'<script id="zoo-ref" type="application/json">(.*?)</script>', zoo.html, re.S)
    assert m, "reference block missing"
    return json.loads(m.group(1))


def node(zoo, repo):
    return next(i for i, nd in enumerate(zoo.data["nodes"]) if repo in nd["members"])


def cases(zoo, ref, truth):
    """Shapes that exercise every branch of the card: by property, the most-downloaded such shape."""
    S, P = ref["shapes"], truth.P
    pick = {}
    tests = {
        "gqa": lambda i: P[i]["slots"]["kv"] == "GQA" and P[i]["slots"]["mixer"] == "softmax" and P[i]["slots"]["window"] == "full",
        "mla_moe": lambda i: P[i]["slots"]["kv"] == "MLA" and P[i]["slots"]["ffn"].startswith("moe"),
        "sliding": lambda i: P[i]["slots"]["window"] == "local" and "sched" in S[i] and "S" in S[i]["sched"],
        "sandwich": lambda i: "post_norm" in P[i]["slots"]["feats"],
        "post_only": lambda i: "post_only" in P[i]["slots"]["feats"],
        "parallel": lambda i: P[i]["slots"]["block"] == "parallel",
        "hybrid": lambda i: P[i]["slots"]["mixer"] == "hybrid" and "sched" in S[i],
        "ssm": lambda i: P[i]["slots"]["mixer"] == "ssm",
        "learned": lambda i: P[i]["slots"]["pos"] == "learned_abs",
        "tied": lambda i: "tied_emb" in P[i]["slots"]["feats"],
        "attn_kinds": lambda i: "akinds" in S[i],              # Gemma-4: sliding and global shapes differ
    }
    for k, f in tests.items():
        hit = [i for i in range(len(P)) if f(i)]
        if hit:
            pick[k] = hit[0]                                 # nodes are in snapshot (download) order
    return pick


# --- data ------------------------------------------------------------------------------------------------

def test_ref_block_is_the_classifier(zoo, ref, truth):
    from graph.zoo_ref import spec_of
    meta = json.load(open(ROOT / "graph" / "model_meta.json"))
    nodes = zoo.data["nodes"]
    assert len(ref["shapes"]) == len(nodes)
    for i, nd in enumerate(nodes):
        want = spec_of(truth.configs[nd["members"][0]])
        got = {k: v for k, v in ref["shapes"][i].items() if k != "var"}
        assert got == json.loads(json.dumps(want)), nd["name"]
        assert ref["mdate"][i] == [((meta.get(m) or {}).get("created_at") or "")[:10] for m in nd["members"]]
        assert ref["mpar"][i] == [int((meta.get(m) or {}).get("hf_total") or 0) for m in nd["members"]]
    # every code mark names a field the card shows
    assert set().union(*map(set, (s["code"] for s in ref["shapes"]))) <= set(CODE_ROW)
    # the schedule strips agree with the classifier's per-kind layer counts
    for s in ref["shapes"]:
        if "sched" in s and not re.search("[FEPQ]", s["sched"]):
            kc = s["kc"]
            assert s["sched"].count("R") == kc.get("R", 0), s["mt"]
            assert s["sched"].count("G") + s["sched"].count("S") == sum(kc.get(k, 0) for k in ("FU", "FT", "LU", "LT", "S"))


# --- the card ---------------------------------------------------------------------------------------------

CARD_JS = """() => {
  const P = document.getElementById('panel'), rows = {};
  P.querySelectorAll('.spec-body > dl.spec').forEach(dl => {
    const dts = dl.querySelectorAll(':scope > dt');
    dts.forEach(dt => { const dd = dt.nextElementSibling;
      rows[dt.textContent] = {v: [...dd.childNodes].filter(n => !(n.classList && n.classList.contains('fc'))).map(n => n.textContent).join(''),
                              code: !!dd.querySelector('.fc'), gl: !!dd.querySelector('.gl')}; });
  });
  const kn = [...P.querySelectorAll('dl.keynums > div')].map(d => [d.querySelector('dt').textContent, d.querySelector('dd').firstChild.textContent]);
  return {rows, kn, title: P.querySelector('h2').textContent};
}"""
# card row label -> the spec_of "code" field it marks
CODE_ROW = {"norm": "Norm", "place": "Placement", "qknorm": "qk-norm", "act": "Activation", "ffn": "FFN", "block": "Block",
            "sink": "Sinks", "pos": "Position", "bias_qkv": "q / k / v bias", "bias_out": "Output biases", "tied": "Embeddings",
            "out_gate": "Output gate", "keqv": "K = V", "router": "Router", "sched": "Window", "vnorm": "V norm"}
PLACE = lambda f: "sandwich" if "post_norm" in f else "post-only" if "post_only" in f else "post-LN" if "post_ln" in f else "pre"


def expected_rows(s, slots):
    f = set(filter(None, slots["feats"].split("+")))
    out = {"Model type": s["mt"] + (f" (in {s['omt']})" if s.get("omt") else ""), "Norm": slots["norm"],
           "Placement": PLACE(f), "Block": slots["block"], "Activation": slots["act"],
           "Embeddings": "tied (lm_head = Eᵀ)" if "tied_emb" in f else "untied"}
    if s.get("d"):
        out["d_model"] = fmt(s["d"])
    if s.get("vocab"):
        out["Vocab"] = fmt(s["vocab"])
    if slots["kv"] != "none":
        out["Attention"] = slots["kv"]
        if s.get("akinds"):                                  # per layer kind (Gemma-4): one row per kind
            for a in s["akinds"]:
                lab = {"global": "Global", "sliding": "Sliding", "kv-shared": "KV-shared"}[a["k"]]
                out[f"{lab} layers ({fmt(a['n'])})"] = " · ".join(filter(None, [
                    f"{fmt(a['h'])} query · {fmt(a['kvh'])} KV · hd {fmt(a['hd'])}" + (f" (v {fmt(a['vd'])})" if a.get("vd") else ""),
                    f"window {fmt(a['win'])}" if a.get("win") else "full", "K = V" if a.get("keqv") else "",
                    "reads an earlier layer's K / V" if a["k"] == "kv-shared" else ""]))
        else:
            out["Heads"] = f"{fmt(s['h'])} query · {fmt(s.get('kvh') or s['h'])} KV"
        out["q / k / v bias"] = "yes" if "bias_qkv" in f else "no"
        out["Window"] = "full" if slots["window"] == "full" else "sliding" + (f" {fmt(s['win'])}" if s.get("win") else "")
    if s.get("E"):
        out["Experts"] = f"{fmt(s['E'])} routed, top-{fmt(s['k'])}" + (f" + {fmt(s['nsh']) + ' ' if s.get('nsh') else ''}shared" if s.get("shw") else "")
    elif s.get("ffn") and not s.get("ffn_widths"):
        out["FFN width"] = fmt(s["ffn"])
    if s.get("theta"):
        out["RoPE theta"] = None                            # present (format checked by the page's own rules)
    return out


def test_cards(open_page, zoo, ref, truth):
    nodes = zoo.data["nodes"]
    p = open_page()
    seen = 0
    for k, i in cases(zoo, ref, truth).items():
        p.goto_hash("#m=" + enc(nodes[i]["members"][0]))
        p.pg.wait_for_function(f"(() => {{ const g = document.querySelector('#svg g.node.me'); return g && +g.dataset.i === {i}; }})()")
        got = p.pg.evaluate(CARD_JS)
        s, slots = ref["shapes"][i], truth.P[i]["slots"]
        for label, v in expected_rows(s, slots).items():
            assert label in got["rows"], (k, label, list(got["rows"]))
            if v is not None:
                assert got["rows"][label]["v"] == v, (k, label, got["rows"][label], v)
        # a code mark exactly where a code table set the value (for the rows the card shows)
        for field, label in CODE_ROW.items():
            if label in got["rows"]:
                assert got["rows"][label]["code"] == (field in s["code"]), (k, label, s["code"])
        kn = dict(got["kn"])
        assert kn["d_model"] == fmt(s["d"]) and kn["Layers"] == fmt(s.get("Le") or s["L"])
        first = min(filter(None, ref["mdate"][i]), default="")
        assert kn["First on Hub"] == (first[:7] or "–")
        t = p.pg.inner_text("#panel")
        assert not BAD_TEXT.search(t), (k, BAD_TEXT.search(t).group(0))
        seen += 1
    assert seen >= 8


def test_card_code_links_and_lineage(open_page, zoo, ref):
    """The card links the transformers modelling file of the type and walks its modular parents."""
    T = ref["types"]
    i = next(i for i, s in enumerate(ref["shapes"]) if T.get(s["mt"], {}).get("parent") and T[s["mt"]].get("files"))
    s = ref["shapes"][i]
    p = open_page("#m=" + enc(zoo.data["nodes"][i]["members"][0]))
    hrefs = p.pg.eval_on_selector_all("#panel .codelinks a", "a => a.map(x => x.href)")
    for f in T[s["mt"]]["files"]:
        assert ref["tf_url"] + f in hrefs
    lin = p.pg.eval_on_selector_all("#panel .lineage .linkish", "b => b.map(x => x.textContent)")
    assert lin[:2] == [s["mt"], T[s["mt"]]["parent"]]
    # a lineage name filters the models table by that type
    p.pg.locator("#panel .lineage .linkish").nth(1).click()
    assert "v=models" in p.hash() and "fam=" + enc(T[s["mt"]]["parent"]) in p.hash()


# --- the diagram ------------------------------------------------------------------------------------------

DIAG_JS = """() => {
  const D = document.querySelector('#panel .diagram svg'), S = document.querySelector('#panel .sched svg');
  const texts = [...D.querySelectorAll('text')].map(t => t.textContent);
  const cells = S ? [...S.querySelectorAll('rect')].map(r => [r.getAttribute('class'), +r.getAttribute('y')]) : [];
  return {norms: D.querySelectorAll('rect.dg-norm').length, attn: D.querySelectorAll('rect.dg-attn').length,
          rec: D.querySelectorAll('rect.dg-rec').length, ffn: D.querySelectorAll('rect.dg-ffn').length,
          adds: D.querySelectorAll('circle.dg-add').length, texts, cells};
}"""


def test_diagrams(open_page, zoo, ref, truth):
    nodes = zoo.data["nodes"]
    p = open_page()
    for k, i in cases(zoo, ref, truth).items():
        p.goto_hash("#m=" + enc(nodes[i]["members"][0]))
        p.pg.wait_for_function(f"(() => {{ const g = document.querySelector('#svg g.node.me'); return g && +g.dataset.i === {i}; }})()")
        g = p.pg.evaluate(DIAG_JS)
        s, sl = ref["shapes"][i], truth.P[i]["slots"]
        f = set(filter(None, sl["feats"].split("+")))
        place, par = PLACE(f), sl["block"] == "parallel"
        halves = 1 if par else 2
        per = {"pre": 1, "sandwich": 2, "post-only": 1, "post-LN": 1}[place]
        assert g["norms"] == halves * per + (0 if place == "post-LN" else 1), (k, g["norms"], place, par)
        assert g["adds"] == halves, k
        assert any("(output)" in t for t in g["texts"]) == (place in ("sandwich", "post-only") and not par), k
        assert f"× {s.get('Le') or s['L']}" in g["texts"], k
        assert any(t.startswith("LM head") and ("tied" in t) == ("tied_emb" in f) for t in g["texts"]), k
        if sl["mixer"] == "softmax":
            assert g["attn"] == 1 and g["rec"] == 0, k
        elif sl["mixer"] in ("ssm", "linear"):
            assert g["rec"] >= 1 and g["attn"] == 0, k
        assert g["ffn"] == 1, k
        if s.get("sched"):                                      # one strip cell per layer of each kind
            row0 = Counter(c for c, y in g["cells"] if y == 0)
            want = Counter("sk-" + ch for ch in s["sched"])
            assert row0 == want, (k, row0, want)
        if s.get("moe_mask"):
            ys = sorted({y for _, y in g["cells"]})
            row1 = Counter(c for c, y in g["cells"] if y == ys[1])
            assert row1 == Counter("sk-" + ch for ch in s["moe_mask"]), k


# --- search -----------------------------------------------------------------------------------------------

def test_search_shows_the_spec(open_page, zoo, ref):
    i = 0
    repo = zoo.data["nodes"][i]["members"][0]
    p = open_page()
    p.pg.fill("#q", repo)
    p.pg.wait_for_selector("#results:not([hidden])")
    line = p.pg.inner_text("#results li .rshape")
    s = ref["shapes"][i]
    assert f"{s['mt']} · {fmt(s['Le'])} layers × {fmt(s['d'])}" in line


# --- the models table -----------------------------------------------------------------------------------

def _mrows(p):
    return p.pg.evaluate("[...document.querySelectorAll('#mtbl tbody tr')].map(tr => +tr.dataset.i)")


def _has(truth, ref, name):
    P = truth.P
    f = lambda i: set(filter(None, P[i]["slots"]["feats"].split("+")))
    return {"MoE": lambda i: P[i]["slots"]["ffn"].startswith("moe"), "MLA": lambda i: P[i]["slots"]["kv"] == "MLA",
            "sliding window": lambda i: P[i]["slots"]["window"] == "local",
            "sandwich norm": lambda i: "post_norm" in f(i), "qk-norm": lambda i: P[i]["slots"]["qknorm"] != "no",
            "tied embeddings": lambda i: "tied_emb" in f(i)}[name]


def test_models_table(open_page, zoo, ref, truth):
    n = len(truth.P)
    p = open_page("#v=models")
    pg = p.pg
    assert pg.get_attribute("#tab-models", "aria-selected") == "true" and not pg.is_hidden("#view-models")
    assert pg.inner_text("#models-count") == f"{fmt(n)} shapes · {fmt(zoo.data['n_models'])} repos"
    rows = _mrows(p)
    assert len(rows) == min(200, n)
    cov = [truth.cover[i] for i in rows]
    assert cov == sorted(cov, reverse=True)                     # default: models emulated, descending
    pg.click("#m-more button")
    assert sorted(_mrows(p)) == list(range(n))
    # a family filter
    fam = Counter(s["mt"] for s in ref["shapes"]).most_common(2)[1][0]
    pg.select_option("#m-fam", fam)
    want = {i for i, s in enumerate(ref["shapes"]) if s["mt"] == fam or s.get("omt") == fam}
    assert set(_mrows(p)) == want
    assert "fam=" + enc(fam) in p.hash()
    pg.select_option("#m-fam", "")
    # two "has" filters: AND
    for h in ("MoE", "MLA"):
        pg.select_option("#m-has", h)
    want = {i for i in range(n) if _has(truth, ref, "MoE")(i) and _has(truth, ref, "MLA")(i)}
    assert set(_mrows(p)) == want and want
    assert "has=MoE,MLA" in p.hash()
    # sort by d_model, both directions
    for d in (-1, 1):
        pg.click("#mtbl th[data-k=d] button")
        got = [ref["shapes"][i]["d"] for i in _mrows(p)]
        assert got == sorted(got, reverse=d < 0), d
    link = p.hash()
    q = open_page(link)                                         # the permalink round-trips
    assert _mrows(q) == _mrows(p)
    assert q.pg.input_value("#m-has") == "" and q.pg.inner_text("#m-has-chips").count("×") == 2
    # the CSV is the rows shown
    with pg.expect_download() as dl:
        pg.click("#m-csv")
    lines = Path(dl.value.path()).read_text().splitlines()
    assert len(lines) == len(_mrows(p)) + 1 and lines[0].startswith("repo,Model,")
    assert [ln.split(",")[0] for ln in lines[1:]] == [zoo.data["nodes"][i]["members"][0] for i in _mrows(p)]
    # a text filter on a repo name
    pg.click("#m-has-chips button")
    pg.click("#m-has-chips button")
    repo = zoo.data["nodes"][5]["members"][0]
    pg.fill("#mq", repo)
    pg.wait_for_function("document.querySelectorAll('#mtbl tbody tr').length < 50")
    assert 5 in _mrows(p)
    p.assert_clean_text("models")


def test_models_keys_and_from_filter(open_page, zoo, ref, truth):
    p = open_page()
    pg = p.pg
    pg.keyboard.press("3")                                      # 1-6 switch the views
    assert pg.get_attribute("#tab-models", "aria-selected") == "true" and "v=models" in p.hash()
    years = sorted({d[:4] for ds in ref["mdate"] for d in ds if d})
    pg.select_option("#m-from", years[-1])
    if pg.locator("#m-more button").count():
        pg.click("#m-more button")
    want = {i for i, ds in enumerate(ref["mdate"]) if min(filter(None, ds), default="") >= years[-1]}
    assert set(_mrows(p)) == want
    pg.select_option("#m-from", "")
    first = pg.locator("#mtbl tbody tr").first
    first.focus()
    pg.keyboard.press("ArrowDown")
    i2 = pg.evaluate("+document.activeElement.dataset.i")
    assert i2 == _mrows(p)[1]
    pg.keyboard.press("c")                                      # c adds the row to the comparison
    pg.keyboard.press("Enter")                                  # Enter opens its card (the drawer)
    assert p.panel()["title"] == zoo.data["nodes"][i2]["name"]
    assert pg.evaluate("document.body.classList.contains('wide') && !document.body.classList.contains('nosel')")
    pg.keyboard.press("Escape")                                 # Escape closes the drawer
    assert p.panel()["title"] == "Pick a model"
    pg.keyboard.press("4")
    assert pg.eval_on_selector_all("#cmptbl thead th .linkish", "b => b.map(x => x.textContent)") == [zoo.data["nodes"][i2]["name"]]


# --- compare --------------------------------------------------------------------------------------------

def _compare_case(zoo, truth):
    """Three popular shapes: a pair where one emulates the other and not back, and a third."""
    for a in range(60):
        for b in range(a + 1, 60):
            if truth.ok[a, b] != truth.ok[b, a]:
                c = next(c for c in range(60) if c not in (a, b))
                return [a, b, c]
    return [0, 1, 2]


def test_compare(open_page, zoo, ref, truth):
    nodes = zoo.data["nodes"]
    ids = _compare_case(zoo, truth)
    link = "#v=compare&c=" + ",".join(enc(nodes[i]["members"][0]) for i in ids)
    p = open_page(link)
    pg = p.pg
    assert p.hash() == link
    assert pg.eval_on_selector_all("#cmptbl thead th .linkish", "b => b.map(x => x.textContent)") == [nodes[i]["name"] for i in ids]
    assert pg.locator("#cmptbl .diagram svg").count() == 3
    rows = pg.evaluate("""[...document.querySelectorAll('#cmptbl tbody tr[data-k]')].map(tr => ({k: tr.dataset.k, diff: tr.classList.contains('diff'),
        cells: [...tr.cells].slice(1).map(c => { const x = c.cloneNode(true); x.querySelectorAll('.fc').forEach(e => e.remove()); return x.textContent; })}))""")
    assert rows
    for r in rows:
        assert r["diff"] == (len(set(r["cells"])) > 1), r
    by = {r["k"]: r for r in rows}
    d = [ref["shapes"][i]["d"] for i in ids]
    assert by["d"]["diff"] == (len(set(d)) > 1) and by["d"]["cells"] == [fmt(x) for x in d]
    # "A emulates B" is the relation
    pairs = pg.evaluate("""[...document.querySelectorAll('#cmp-emu .pair[data-host]')].map(e => [+e.dataset.host, +e.dataset.target,
        e.querySelector('div').textContent])""")
    assert len(pairs) == 6
    for h, t, txt in pairs:
        yes = bool(truth.ok[t, h])                              # ok[i, j]: j emulates i
        assert txt == f"{nodes[h]['name']}{' emulates ' if yes else ' does not emulate '}{nodes[t]['name']}"
    # a stored construction opens the edge view
    btn = pg.locator("#cmp-emu .pair button:has-text('Open the construction')")
    if btn.count():
        btn.first.click()
        assert p.panel()["edge"]
    # the smallest common host
    common = [j for j in range(len(nodes)) if all(j == i or truth.ok[i, j] for i in ids)]
    t = pg.inner_text("#cmp-emu .pair:last-child")
    if common:
        j = min(common, key=lambda j: nodes[j]["params"])
        assert nodes[j]["name"] in t
    else:
        assert t == "No single shape emulates all of them."
    p.assert_clean_text("compare")


def test_compare_from_card_and_cap(open_page, zoo):
    nodes = zoo.data["nodes"]
    p = open_page("#m=" + enc(nodes[0]["members"][0]))
    p.pg.click("#add-cmp")
    assert p.pg.get_attribute("#tab-compare", "aria-selected") == "true"
    assert p.hash().endswith("v=compare&c=" + enc(nodes[0]["members"][0]))
    for i in range(1, 6):                                       # four at most: the oldest drops
        p.pg.fill("#cq", nodes[i]["members"][0])
        p.pg.keyboard.press("Enter")
    names = p.pg.eval_on_selector_all("#cmptbl thead th .linkish", "b => b.map(x => x.textContent)")
    assert names == [nodes[i]["name"] for i in range(2, 6)]


@pytest.fixture(scope="module")
def served(zoo, truth, tmp_path_factory):
    """The page and its lazy pair file over http (fetch() refuses file:// pages)."""
    from graph.model_graph import emul_matrix
    from graph.zoo_ref import SIDE_PAIRS, pair_blockers
    d = tmp_path_factory.mktemp("served")
    (d / "zoo.html").write_text(zoo.html)
    pb = pair_blockers(truth.P, truth.ok, emul_matrix(truth.P, mode="capacity"))
    (d / SIDE_PAIRS).write_text(json.dumps(pb))
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    h.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/zoo.html", pb
    srv.shutdown()


def test_compare_blockers_over_http(browser, zoo, truth, served):
    from graph.frontier import blockers
    from graph.model_graph import emul_matrix
    url, _ = served
    nodes = zoo.data["nodes"]
    cap = emul_matrix(truth.P, mode="capacity")
    x = cap & ~truth.ok
    np.fill_diagonal(x, False)
    t, h = map(int, next(zip(*np.where(x))))                    # h dominates t on capacity, does not emulate it
    p = Page(browser, type("Z", (), {"url": url})(), f"#v=compare&c={enc(nodes[h]['members'][0])},{enc(nodes[t]['members'][0])}")
    try:
        sel = f"#cmp-emu .pair[data-host='{h}'][data-target='{t}'] li"
        p.pg.wait_for_function(f"document.querySelector(\"{sel}\") && !document.querySelector(\"{sel}\").textContent.startsWith('loading')")
        got = p.pg.eval_on_selector_all(sel, "l => l.map(x => x.textContent)")
        assert got == blockers(truth.P[t], truth.P[h])
    finally:
        p.close()


# --- timeline -------------------------------------------------------------------------------------------

def test_timeline(open_page, zoo, ref, truth):
    nodes = zoo.data["nodes"]
    p = open_page("#v=timeline")
    first = lambda i: min(filter(None, ref["mdate"][i]), default="")
    for attr in ("MoE", "MLA", "sliding window", "qk-norm", "tied embeddings"):
        has = [i for i in range(len(nodes)) if first(i) and _has(truth, ref, attr)(i)]
        i0 = min(has, key=lambda i: (first(i), i))
        row = p.pg.locator(f"#tl g.row[data-attr='{attr}']")
        assert row.locator("circle").count() == len(has), attr
        lab = row.locator("text.first").first.text_content()
        assert lab == f"{nodes[i0]['name']} ({first(i0)[:7]})", (attr, lab)
    p.pg.locator("#tl g.row[data-attr='MLA'] circle.first").dispatch_event("click")
    assert p.panel()["title"] != "Pick a model"


# --- glossary -------------------------------------------------------------------------------------------

def test_glossary_popover(open_page, zoo, ref):
    p = open_page("#m=" + enc(zoo.data["nodes"][0]["members"][0]))
    pg = p.pg
    term = pg.locator("#panel .spec-body .gl").first
    key = term.get_attribute("data-g")
    term.click()
    assert pg.is_visible("#pop")
    assert pg.inner_text("#pop b") == ref["glossary"][key]["t"]
    pg.keyboard.press("Escape")
    assert pg.is_hidden("#pop")
    term.click()
    pg.click("#pop a")
    assert pg.get_attribute("#tab-glossary", "aria-selected") == "true"
    assert p.hash().endswith("v=glossary&g=" + enc(key))
    assert pg.locator("#gloss section.hit h4").inner_text() == ref["glossary"][key]["t"]
    assert pg.locator("#gloss section").count() == len(ref["glossary"])
    q = open_page("#v=glossary&g=" + enc(key))
    assert q.pg.locator("#gloss section.hit").count() == 1


# --- the optional passes ----------------------------------------------------------------------------------

def _fmt_bytes(b):
    for v, s in ((2 ** 40, "TiB"), (2 ** 30, "GiB"), (2 ** 20, "MiB"), (2 ** 10, "KiB")):
        if b >= v:
            x = b / v
            return (f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}") + " " + s
    return f"{round(b):,} B"


def synthetic_passes(nodes):
    """TEST-ONLY files in the documented schema (graph/arch_metrics_schema.md) with made-up numbers, to drive
    the metrics and frameworks sections; the shipped page reads the real graph/data/*.json."""
    Ls = {"1k": 1024, "4k": 4096, "8k": 8192, "32k": 32768, "128k": 131072}
    shapes = []
    for k, nd in enumerate(nodes[:3]):
        per = (k + 1) * 1000
        kv = {L: {"L": v, "beyond_max": v > 32768, "elems": {"global": per * v, "sliding": per * min(v, 4096)},
                  "in_window": {"sliding": v <= 4096}, "bf16": 2 * per * (v + min(v, 4096)), "fp8": per * (v + min(v, 4096))}
              for L, v in {**Ls, "max": 32768}.items()}
        m = {"model_type": "x", "max_context": 32768, "layers": {"total": 4, "global": 2, "sliding": 2, "window": 4096},
             "params": {"total": 10 ** 9 * (k + 1), "active": 10 ** 8 * (k + 1), "embedding": 1, "lm_head": 0},
             "kv_per_token": {"elems": {"global": per, "sliding": per}, "elems_total": 2 * per, "bf16": 4 * per, "fp8": 2 * per},
             "kv": kv, "state": {"elems": 0, "bf16": 0}, "flops": {"matmul": 1, "prefill": {L: {"total": 1e9 * v} for L, v in Ls.items()},
                                                                      "decode": {L: {"total": 2e9, "attn": 1e8} for L in Ls}},
             "memory": {L: {"batch1": 3 * 2 ** 30, "batchB": 9 * 2 ** 30} for L in Ls},
             "bandwidth": {L: {"bytes": 2 ** 31, "intensity": 1.5} for L in Ls},
             "prov": {}, "valid": {"params": "trace-verified", "kv": "symbolic-proved", "flops": "formula-only", "memory": "trace-verified",
                                   "bandwidth": "failed", "state": "formula-only", "active": "trace-verified"}, "notes": ["synthetic"]}
        shapes.append({"rep": nd["members"][0], "n_members": len(nd["members"]), "metrics": m, "range": {}})
    am = {"schema_version": 1, "contexts": Ls, "batch_B": 16, "shapes": shapes + [None] * (len(nodes) - 3), "repos": {}, "types": {}}
    names = ["transformers", "vllm", "llama.cpp"]
    fw = {"schema_version": 1, "frameworks": {f: {"repo": f, "version": "v1"} for f in names},
          "repos": {nodes[0]["members"][0]: {"native": {"transformers": True, "vllm": False, "llama.cpp": None}}},
          "shapes": [{"rep": nodes[0]["members"][0], "backdoor": {"vllm": {"n_hosts": 4, "n_exact": 3, "hosts": [
              {"host": nodes[1]["members"][0], "edge": "exact", "params_x": 1.5, "width_x": 1.0, "depth_x": 2.0}]}}}]
          + [{"rep": nd["members"][0]} for nd in nodes[1:]]}
    r0, r1 = nodes[0]["members"][0], nodes[1]["members"][0]
    rc = {"schema_version": 1, "frameworks": {"jax": {"package": "jax", "version": "0.8.1"}, "pytorch": {"package": "torch", "version": "2.9"},
                                             "dynet": {"package": "dynet", "version": "2.1"}},
          "validation": [{"id": "jax/native/" + r0, "framework": "jax", "route": "native", "target": r0, "kind": "checkpoint",
                          "max_abs_diff": 3.1e-5, "logit_scale": 21.3, "greedy_tokens": 32, "greedy_identical": True, "pass": True, "date": "2026-11-13"}],
          "models": {r0: {"shape": 0, "loader": "from_llama", "frameworks": {
              "jax": {"route": "native", "host": None, "cost": {"params_x": 1.0, "width_x": 1.0, "depth_x": 1.0},
                      "script": "recipes/scripts/jax/" + r0.replace("/", "__") + ".py",
                      "make": f"python -m recipes.make_recipe {r0} --framework jax", "status": "e2e", "validated_by": "jax/native/" + r0},
              "pytorch": {"route": "exact-host", "host": r1, "cost": {"params_x": 1.25, "width_x": 1.0, "depth_x": 1.0}, "script": None,
                          "make": f"python -m recipes.make_recipe {r0} --framework pytorch --host {r1}", "status": "family", "validated_by": None},
              "dynet": {"route": None, "why": "no universal loader for this config"}},
              "native_cli": {"vllm": {"route": "native", "cmd": f"vllm serve {r0}"}}}}}
    return am, fw, rc


@pytest.fixture(scope="module")
def with_passes(zoo, ref, tmp_path_factory):
    from graph.zoo_ref import compact_frameworks, compact_metrics, compact_recipes
    nodes = zoo.data["nodes"]
    am, fw, rc = synthetic_passes(nodes)
    node_of = {m: i for i, nd in enumerate(nodes) for m in nd["members"]}
    r = dict(ref, metrics=compact_metrics(am, nodes), frameworks=compact_frameworks(fw, nodes, node_of),
             recipes=compact_recipes(rc, nodes, node_of))
    blob = json.dumps(r, separators=(",", ":")).replace("</", "<\\/")
    html = re.sub(r'(<script id="zoo-ref" type="application/json">)(.*?)(</script>)', lambda m: m.group(1) + blob + m.group(3), zoo.html, flags=re.S)
    path = tmp_path_factory.mktemp("passes") / "zoo.html"
    path.write_text(html)
    return type("Z", (), {"url": path.as_uri()})(), am, fw, rc


def test_metrics_and_frameworks_sections(browser, zoo, with_passes):
    z, am, fw, rc = with_passes
    nodes = zoo.data["nodes"]
    p = Page(browser, z, "#m=" + enc(nodes[0]["members"][0]))
    try:
        pg = p.pg
        m = am["shapes"][0]["metrics"]
        cost = lambda: pg.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#cost dl.spec dt')].map(dt =>
            [dt.textContent, [dt.nextElementSibling.firstChild.textContent, (dt.nextElementSibling.querySelector('.vb') || {}).textContent || '']]))""")
        c = cost()                                              # the default context: 4k
        assert c["KV cache (bf16)"][0].startswith(_fmt_bytes(m["kv"]["4k"]["bf16"])) and c["KV cache (bf16)"][1] == "symbolic-proved"
        assert c["Memory, batch 1"] == [_fmt_bytes(3 * 2 ** 30), "trace-verified"]
        assert c["Decode reads"][1] == "check failed"
        assert "inside the window" in c["KV cache (bf16)"][0]
        pg.locator("#cost input[type=range]").fill("4")         # 128k: beyond the max, the sliding layers flat
        c = cost()
        assert c["KV cache (bf16)"][0].startswith(_fmt_bytes(m["kv"]["128k"]["bf16"]))
        assert "flat at the window" in c["KV cache (bf16)"][0]
        assert pg.inner_text("#cost .ctxrow").endswith("beyond the model's max")
        assert pg.locator("#cost svg circle").count() == 6
        # params: the metrics pass's total and active
        kn = dict(pg.evaluate("[...document.querySelectorAll('dl.keynums > div')].map(d => [d.querySelector('dt').textContent, d.querySelector('dd').textContent])"))
        assert kn["Params"] == "1B100M active incl. emb. · 100M excl."   # both conventions (zoo-fixes 8)
        fwr = pg.evaluate("[...document.querySelectorAll('#panel tr[data-fw]')].map(tr => [tr.dataset.fw, tr.cells[1].textContent])")
        fwr = dict(fwr)
        assert fwr["transformers"] == "native" and fwr["llama.cpp"] == "unknown"
        assert fwr["vllm"].startswith("no; via " + nodes[1]["name"] + " (exact, 1.50× params, 2.00× depth)") and "4 hosts, 3 exact" in fwr["vllm"]
        # "Run it": the recipes, with the repo and host names put back into the commands
        r0, r1 = nodes[0]["members"][0], nodes[1]["members"][0]
        rr = dict(pg.evaluate("[...document.querySelectorAll('#recipes tr[data-rf]')].map(tr => [tr.dataset.rf, tr.cells[1].innerText])"))
        assert set(rr) == {"jax", "pytorch", "dynet", "vllm"}
        assert rr["jax"].startswith("native") and "run end to end" in rr["jax"] and f"python -m recipes.make_recipe {r0} --framework jax" in rr["jax"]
        assert "this checkpoint: max |Δlogit| 3.1e-5, 32 greedy tokens identical" in rr["jax"]
        assert rr["pytorch"].startswith("exact host " + nodes[1]["name"]) and "1.25× params" in rr["pytorch"] and "tested on its code path" in rr["pytorch"]
        assert f"--host {r1}" in rr["pytorch"]
        assert rr["dynet"].startswith("no route") and "no universal loader" in rr["dynet"]
        assert f"vllm serve {r0}" in rr["vllm"]
        href = pg.get_attribute("#recipes a[download]", "href")
        assert href.endswith("/main/recipes/scripts/jax/" + r0.replace("/", "__") + ".py") and href.startswith("https://raw.githubusercontent.com/")
        p.assert_clean_text("passes")
        # the models table grows the columns, and the "runs on" filter
        p.goto_hash("#v=models&has=" + enc("runs natively on transformers"))
        pg.wait_for_function("document.querySelectorAll('#mtbl tbody tr').length === 1")
        assert pg.locator("#mtbl th[data-k=kvt]").count() == 1 and pg.locator("#mtbl th[data-k=fw]").count() == 1
    finally:
        p.close()


def test_page_without_the_reference(browser, zoo, tmp_path):
    """The emulation-only page (graph.model_graph --html-models leaves the placeholder): the reference tabs hide."""
    html = re.sub(r'(<script id="zoo-ref" type="application/json">)(.*?)(</script>)', r"\1__ZOO_REF__\3", zoo.html, flags=re.S)
    path = tmp_path / "zoo.html"
    path.write_text(html)
    p = Page(browser, type("Z", (), {"url": path.as_uri()})(), "#m=" + enc(zoo.data["nodes"][0]["members"][0]))
    try:
        assert p.pg.is_hidden("#tab-models") and p.pg.is_hidden("#tab-compare")
        assert p.panel()["title"] == zoo.data["nodes"][0]["name"]
        assert p.pg.locator("#panel h3:text-is('Shape')").count() == 1
        p.pg.keyboard.press("3")                                # no reference: the key does nothing
        assert p.pg.get_attribute("#tab-graph", "aria-selected") == "true"
    finally:
        p.close()


# --- phone and themes -----------------------------------------------------------------------------------

@pytest.mark.parametrize("hash_", ["#m={r}", "#v=models", "#v=compare&c={r},{r2}", "#v=timeline", "#v=glossary"])
def test_phone_reference(open_page, zoo, hash_):
    nodes = zoo.data["nodes"]
    h = hash_.format(r=enc(nodes[0]["members"][0]), r2=enc(nodes[1]["members"][0]))
    p = open_page(h, viewport=PHONE, mobile=True)
    _no_overflow(p, h)
    p.assert_clean_text(h)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_reference_views_render_in_both_themes(open_page, zoo, scheme):
    nodes = zoo.data["nodes"]
    for h in ("#v=models", "#v=timeline", "#v=compare&c=" + enc(nodes[0]["members"][0]) + "," + enc(nodes[3]["members"][0])):
        p = open_page(h, scheme=scheme)
        assert p.pg.locator("#view-" + h[3:].split("&")[0] + " svg, #view-" + h[3:].split("&")[0] + " tbody tr").count() > 0
        p.assert_clean_text(h)
