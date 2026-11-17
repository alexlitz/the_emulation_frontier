"""The zoo page (--html-models) embeds the headline numbers: every shape, the frontier size from
emul_matrix(mode="payable"), and every frontier model with FRONTIER.md's cover and binding.

Run:  pytest tests/test_zoo_page.py
"""
import json
import re

import numpy as np
import pytest

from graph.frontier import frontier, load, reps_of
from graph.model_graph import _dedup_points, emul_matrix, export_html_points


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    path = tmp_path_factory.mktemp("zoo") / "zoo.html"
    export_html_points(load(), str(path))
    html = path.read_text()
    m = re.search(r'<script id="zoo-data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "data block missing"
    return html, json.loads(m.group(1))


def test_counts(page):
    _, data = page
    P = _dedup_points(load())
    ok = emul_matrix(P, mode="payable")
    n_front = len(reps_of(ok, P))
    assert data["n_shapes"] == len(P) == len(data["nodes"])
    assert data["n_models"] == sum(len(p["members"]) for p in P) == len(load())
    assert data["n_frontier"] == n_front == sum(nd["fr"] == 1 for nd in data["nodes"])
    # the relation bitset is the headline matrix itself (diagonal dropped)
    o = ok.copy()
    np.fill_diagonal(o, False)
    bits = np.frombuffer(__import__("base64").b64decode(data["emu"]), np.uint8)
    rows = np.unpackbits(bits.reshape(len(P), data["emu_row"]), axis=1)[:, :len(P)].astype(bool)
    assert (rows == o.T).all()


def test_frontier_split_matches_frontier_md(page):
    """The header's "(AR + diffusion)" split counts every non-AR frontier shape as diffusion, as FRONTIER.md
    does (regression: block-causal SDAR shapes were counted as AR)."""
    _, data = page
    rows, _, _, _ = frontier()
    assert data["n_frontier_diff"] == sum(r[2]["gen"] != "ar" for r in rows)
    gen = data["slot_keys"].index("gen")
    assert data["n_frontier_diff"] == sum(nd["fr"] == 1 and nd["slots"][gen] != "ar" for nd in data["nodes"])


def test_every_frontier_model_present(page):
    _, data = page
    rows, _, _, _ = frontier()
    reps = {nd["name"]: nd for nd in data["nodes"] if nd["fr"] == 1}
    assert len(rows) == len(reps)
    for cover, name, _, _, why in rows:
        assert name in reps, name
        assert reps[name]["cover"] == cover
        assert reps[name]["bind"] == why


def test_page_is_self_contained(page):
    html, data = page
    assert "__ZOO_DATA__" not in html
    assert "<script src=" not in html                     # no external scripts
    assert data["post"].startswith("https://alexlitzenberger.com/blog/the_emulation_frontier")
    assert all(0 <= a < data["n_shapes"] and 0 <= b < data["n_shapes"] for a, b in data["hasse"])


# --- the edge view -------------------------------------------------------------------------------------

def _edge_pairs(data):
    """The pairs zoo_edges stored, in the page's order: covering pairs, then (shape, frontier host)."""
    import base64
    n, R = data["n_shapes"], data["emu_row"]
    bits = np.frombuffer(base64.b64decode(data["emu"]), np.uint8)
    rows = np.unpackbits(bits.reshape(n, R), axis=1)[:, :n].astype(bool)      # rows[j, i]: j emulates i
    hasse = [tuple(x) for x in data["hasse"]]
    hs = set(hasse)
    reps = sorted(i for i, nd in enumerate(data["nodes"]) if nd["fr"] == 1)
    return hasse + [(i, j) for i in range(n) for j in reps if i != j and rows[j, i] and (i, j) not in hs]


def _records(E):
    import base64
    import struct
    b = base64.b64decode(E["eb"])
    return [struct.unpack("<HBH", b[5 * k:5 * k + 5]) for k in range(len(b) // 5)]


def test_edge_view_matches_edge_plan(page):
    """Every stored edge decodes; a sample of them equals graph.edge_plan.edge_plan run from scratch (the
    relation's verdicts recomputed by emulates(), not read from the matrix)."""
    from graph.edge_plan import edge_plan
    _, data = page
    E, nodes, DK = data["edges"], data["nodes"], data["dim_keys"]
    pairs, recs = _edge_pairs(data), _records(data["edges"])
    assert len(pairs) == len(recs) == E["n"] > len(data["hasse"])
    P = _dedup_points(load())
    dim = lambda i, k: nodes[i]["dims"][DK.index(k)]
    for k in list(range(0, len(pairs), 97)) + [0, len(pairs) - 1]:
        t, h = pairs[k]
        s, ax, v = recs[k]
        plan = edge_plan(P[t], P[h])
        assert plan["ok"] and bool(ax >> 7) == plan["exact"], (k, plan)
        rows = [E["rows"][r] for r in E["sets"][s]]
        assert [(r[0], r[1], r[2], r[3], r[4]) for r in rows] == \
               [(r["slot"], r["target"], r["host"], r["kind"], r["currency"]) for r in plan["slots"]]
        dev = {x[0]: x for x in E["axsets"][ax & 127]}
        for r in plan["axes"]:
            x = dev.get(E["axes"].index(r["axis"]))
            tv = dim(t, r["axis"])
            need, have = ((x[1], x[2]) if x and len(x) == 5 else
                          (x[1] * tv, dim(h, E["have_keys"][x[2]])) if x else (tv, dim(h, r["axis"])))
            assert (need, have) == (r["need"], r["have"]), (k, r)
        if not plan["exact"]:                   # refused, or a limit the script builds (LIMIT_BUILT): labelled
            from elephantine.emulate import limit_ok
            assert E["support"][v].startswith("the edge needs a limit") or limit_ok(plan), (k, E["support"][v])


def test_edge_view_v1_status(page):
    """The page's build status is elephantine.emulate.support for the command repos, and the page carries the
    edge view (clickable neighbours, the command). The families added after v1 show up among the edges it
    builds: LayerNorm and learned-position targets, ERNIE MoE hosts, MoE targets."""
    from elephantine.emulate import side_support, support
    from graph.model_graph import ZOO_CONFIGS, iter_json_configs
    html, data = page
    E, nodes = data["edges"], data["nodes"]
    configs = dict(iter_json_configs(ZOO_CONFIGS))
    pairs, recs = _edge_pairs(data), _records(E)
    repo = lambda i: nodes[i]["members"][nodes[i].get("ci", 0)]
    builds = [k for k, (t, h) in enumerate(pairs) if not recs[k][2] and not E["tr"][t] and not E["hr"][h]]
    assert len(builds) > 3000
    for k in builds[::50]:
        t, h = pairs[k]
        # (a non-exact edge it builds is a labelled limit, LIMIT_BUILT: the eps lever, ReLU <- GeLU)
        assert support(configs[repo(t)], configs[repo(h)], limit=not recs[k][1] >> 7) is None
    mt = lambda i: configs[repo(i)].get("model_type")
    targets, hosts = {mt(pairs[k][0]) for k in builds}, {mt(pairs[k][1]) for k in builds}
    assert {"gpt_neox", "gpt2", "mixtral"} <= targets and "ernie4_5_moe" in hosts, (targets, hosts)
    # emulate v3: KV-head packing, MLA, post-norm-only, Mamba-1 / LFM2 and the families added appear among them
    assert {"olmo2", "mamba", "gpt_neo", "lfm2"} & targets and {"phimoe", "seed_oss", "glm4_moe"} & hosts, (targets, hosts)
    assert any("MLA" in s for s in E["support"]) and any("post-norm" in s for s in E["support"])
    for i in range(0, len(nodes), 41):
        rs = side_support(configs[repo(i)], "target")
        assert E["support"][E["tr"][i]] == ("\n".join(rs) if rs else "")
    for needle in ("function renderEdge", "python -m elephantine.emulate", "edge-go", "#e="):
        assert needle in html
