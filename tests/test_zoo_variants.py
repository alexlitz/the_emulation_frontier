"""The zoo page's Variants view (graph/zoo_variants.py + graph/templates/zoo_variants.js).

Data (the embedded #zoo-variants block, from the session page python -m graph.zoo_ref --out OUT --no-pairs):
  membership   every slot / feature variant equals classify() on the snapshot, model by model; code-dependent
               variants differ only by UNKNOWN on models whose type has no modelling code
  spot         Gemma-4 global layers K = V; gpt-oss sinks + sliding + MoE; DeepSeek-V3 MLA; Qwen3 per-head qk-norm;
               OLMo-2 whole-projection qk-norm; Gemma-2 attention softcap; Jamba a Mamba hybrid; Falcon-7B MQA
  counts       n / s / u are the bitsets' counts; exclusive slot axes sum to the models; parents hold their children
  facets       every numeric facet equals spec_of / classify read directly; sweep-finding variants hold the finding's
               live examples that are in the snapshot
Browser (playwright + chromium, skipped without them):
  render, component tabs, live counts after combined filters (variants + numeric ranges + date) equal a direct
  query, histogram bars, permalinks round-trip, card links, the Models table carry-over, CSV, co-occurrence and
  timeline.

Run:  pytest tests/test_zoo_variants.py -q -n 2
"""
import base64
import json
import re
from collections import Counter

import pytest

from graph import model_graph as mg
from graph import zoo_ref
from graph import zoo_variants as zv


def unbits(s, n):
    b = base64.b64decode(s)
    return [bool((b[i >> 3] >> (7 - (i & 7))) & 1) for i in range(n)]


@pytest.fixture(scope="module")
def V(zoo):
    m = re.search(r'<script id="zoo-variants" type="application/json">(.*?)</script>', zoo.html, re.S)
    assert m, "variants block missing"
    X = json.loads(m.group(1))
    n = X["n"]
    names = [r for nd in zoo.data["nodes"] for r in nd["members"]]
    node = [i for i, nd in enumerate(zoo.data["nodes"]) for _ in nd["members"]]
    assert len(names) == n == zoo.data["n_models"]
    mem = {v["id"]: unbits(v["m"], n) for v in X["variants"]}
    unk = {v["id"]: unbits(v["um"], n) if v.get("um") else [False] * n for v in X["variants"]}
    by = {v["id"]: v for v in X["variants"]}
    return X, names, node, mem, unk, by


@pytest.fixture(scope="module")
def direct(V):
    """classify / dims_of / spec_of of every repo, read directly (not through graph.zoo_variants)."""
    X, names, *_ = V
    configs = dict(mg.iter_json_configs(mg.ZOO_CONFIGS))
    out = []
    for r in names:
        cfg = configs[r]
        out.append({"slots": mg.classify(cfg), "spec": zoo_ref.spec_of(cfg), "cfg": cfg})
    return out


# --- membership = the classifier ---------------------------------------------------------------------------------

def test_slot_and_feature_variants_are_the_classifier(V, direct):
    X, names, node, mem, unk, by = V
    nc = unbits(X["models"]["nc"], X["n"])
    checked = 0
    for v in X["variants"]:
        src = v["src"]
        m = re.match(r"classifier: classify\(\) slot (\w+) = (\S+)$", src)
        f = re.match(r"classifier: classify\(\) feature (\w+)( absent)?$", src)
        if not (m or f):
            continue
        if m:
            truth = [d["slots"][m.group(1)] == m.group(2) for d in direct]
        else:
            has = [f.group(1) in set(d["slots"]["feats"].split("+")) for d in direct]
            truth = [not h for h in has] if f.group(2) else has
        yes, u = mem[v["id"]], unk[v["id"]]
        for k in range(X["n"]):
            if u[k]:                                   # unknown only where the classifier says no and there is no code
                assert v.get("code") and not truth[k] and nc[k], (v["id"], names[k])
            else:
                assert yes[k] == truth[k], (v["id"], names[k], yes[k], truth[k])
        checked += 1
    assert checked >= 60


def test_qknorm_flavours_and_recurrent_families_are_the_classifier(V, direct):
    X, names, node, mem, unk, by = V
    for flav in mg.QKNORM_VARIANTS:
        truth = [d["slots"]["qknorm"] == flav for d in direct]
        assert [a and not b for a, b in zip(mem["qkn_" + flav], unk["qkn_" + flav])] == truth, flav
    rf = [d["spec"].get("rf") for d in direct]
    for gid, _, _, _, fams in zv._RF_GROUPS:
        assert mem["rf_" + gid] == [x in fams for x in rf], gid
        for fam in fams if len(fams) > 1 else ():
            assert mem[f"rf_{gid}:{fam}"] == [x == fam for x in rf], fam
    assert sum(unk["rf_gdn"]) == sum(1 for x in rf if x and x.startswith("other_"))   # unknown family: unknown, not absent


def test_dynamic_variants_are_the_classifier(V, direct):
    X, names, node, mem, unk, by = V
    for v in X["variants"]:
        kind, _, val = v["id"].partition(":")
        if kind == "act":
            assert mem[v["id"]] == [d["slots"]["act"] == val for d in direct]
        elif kind == "rope_scale":
            assert mem[v["id"]] == [d["spec"].get("scaling") == val for d in direct]
        elif kind == "score":
            assert mem[v["id"]] == [d["spec"].get("score") == val for d in direct]


# --- spot checks on known models ---------------------------------------------------------------------------------

SPOT = [
    ("nvidia/Gemma-4-31B-IT-NVFP4", ["keqv_global", "swa_mix", "akinds", "vnorm", "qkn_head"], ["keqv_all", "mla"]),
    ("openai/gpt-oss-20b", ["sink_learned", "swa", "swa_mix", "moe_any", "moe", "act:swigluoai"], ["mla", "qknorm"]),
    ("deepseek-ai/DeepSeek-V3", ["mla", "mla_kv_norm", "mla_rope", "moe_shared", "router_bias", "router_group", "mtp",
                                 "score:sigmoid"], ["gqa", "swa"]),
    ("Qwen/Qwen3-8B", ["gqa", "qkn_head", "rope"], ["qkn_proj", "sink_learned"]),
    ("allenai/OLMo-2-1124-7B-Instruct", ["qkn_proj", "post_only"], ["qkn_head", "sandwich"]),
    ("google/gemma-2-9b", ["softcap_attn", "softcap_final", "sandwich", "swa_mix"], ["scale_qpas"]),   # qpas = head_dim
    ("google/gemma-2-27b-it", ["softcap_attn", "scale_qpas"], []),                                         # 144 vs 128
    ("ai21labs/Jamba-v0.1", ["hybrid", "rf_mamba1", "rf_mamba1:mamba1_bcdtw", "nope", "moe"], ["pure_rec"]),
    ("tiiuae/falcon-7b", ["mqa", "parallel", "rope"], ["gqa", "mha"]),
    ("meta-llama/Llama-3.1-8B", ["gqa", "rope", "rope_scale:llama3", "causal_full", "rmsnorm", "glu", "act:SiLU"], ["swa", "qknorm"]),
]


@pytest.mark.parametrize("repo,has,hasnt", SPOT, ids=[s[0] for s in SPOT])
def test_known_models(V, repo, has, hasnt):
    X, names, node, mem, unk, by = V
    k = names.index(repo)
    for v in has:
        assert mem[v][k], (repo, v)
    for v in hasnt:
        assert not mem[v][k], (repo, v)


# --- counts ------------------------------------------------------------------------------------------------------

def test_counts_are_consistent(V):
    X, names, node, mem, unk, by = V
    n = X["n"]
    for v in X["variants"]:
        yes = [k for k in range(n) if mem[v["id"]][k]]
        assert v["n"] == len(yes) and v["u"] == sum(unk[v["id"]]), v["id"]
        assert v["s"] == len({node[k] for k in yes}), v["id"]
        assert not any(a and b for a, b in zip(mem[v["id"]], unk[v["id"]])), v["id"]
        if v.get("p"):
            assert all(mem[v["p"]][k] for k in yes), (v["id"], "outside its parent")
        if v["first"]:
            d = min(X["models"]["date"][k] for k in yes if X["models"]["date"][k])
            assert v["first"][0] == d and mem[v["id"]][names.index(v["first"][1])]
    total = lambda *ids: sum(sum(mem[i]) for i in ids)
    assert total("mha", "gqa", "mqa", "mla", "kv_none") == n
    assert total("rmsnorm", "layernorm") == n
    assert total("rope", "partial_rope", "nope", "alibi", "learned_abs", "sinusoidal", "t5_bias") == n
    assert total("sequential", "parallel") == n
    assert total("tied", "untied") == n
    assert total(*[v["id"] for v in X["variants"] if v["id"].startswith("act:")]) == n
    assert total("glu", "vanilla", "moe_any", "hadamard", "ffn_none") == n and total("moe", "moe_shared") == total("moe_any")
    assert total("hybrid", "pure_rec") + sum(1 for k in range(n) if not mem["hybrid"][k] and not mem["pure_rec"][k]) == n
    for parent in ("swa", "out_gate", "cla", "rf_mamba1", "rf_mamba2", "rf_gdn", "rf_kda", "rf_rwkv", "rf_lightning"):
        kids = [v["id"] for v in X["variants"] if v.get("p") == parent]
        assert total(*kids) == total(parent), parent           # the children partition the parent
    dsa_kids = [v["id"] for v in X["variants"] if v.get("p") == "dsa"]
    assert [any(mem[c][k] for c in dsa_kids) for k in range(n)] == mem["dsa"]
    q = [v["id"] for v in X["variants"] if v.get("p") == "qknorm"]
    assert total(*q) == total("qknorm")
    # the attention component's "used" variants: the report of the view
    assert by["gqa"]["n"] > by["mha"]["n"] > by["mla"]["n"] > 0


def test_finding_variants_hold_the_live_examples(V):
    X, names, node, mem, unk, by = V
    findings = json.load(open(zv.FINDINGS))
    fk = {f["key"]: f for f in findings}
    for v in X["variants"]:
        if not v["src"].startswith("sweep finding"):
            continue
        key = v["src"].split()[2]
        ex = [r for r in fk[key]["example_live"] if r in names]
        assert ex, (v["id"], key)
        for r in ex:
            assert mem[v["id"]][names.index(r)], (v["id"], r)


def test_facets_are_the_classifier(V, direct):
    X, names, node, mem, unk, by = V
    F = {f["id"]: f for f in X["facets"]}
    for k, d in enumerate(direct):
        sp, s = d["spec"], d["slots"]
        assert F["d"]["v"][k] == (sp.get("d") or "?")
        assert F["vocab"]["v"][k] == (sp.get("vocab") or "?")
        assert F["ctx"]["v"][k] == (sp.get("ctx") or "?")
        assert F["E"]["v"][k] == ((sp.get("E") or "?") if s["ffn"].startswith("moe") else None)
        assert F["k"]["v"][k] == ((sp.get("k") or "?") if s["ffn"].startswith("moe") else None)
        assert F["win"]["v"][k] == ((sp.get("win") or "?") if s["window"] == "local" else None)
        assert F["mla_kv"]["v"][k] == (((sp.get("mla") or {}).get("kv") or "?") if s["kv"] == "MLA" else None)
        if s["kv"] == "none":
            assert all(F["hd"]["v"][q][k] is None for q in F["hd"]["v"])
        elif not sp.get("akinds"):                        # one attention shape: the card's head dim on every kind
            vals = {F["hd"]["v"][q][k] for q in F["hd"]["v"]} - {None}
            assert vals == {sp.get("hd") or "?"}, names[k]
        else:                                             # per-kind shapes from the metrics plan
            want = {zv.ATTN_KINDS[e["k"]]: e["hd"] for e in reversed(sp["akinds"])}
            for q, hd in want.items():
                assert F["hd"]["v"][q][k] == hd, (names[k], q)
    g = names.index("nvidia/Gemma-4-31B-IT-NVFP4")
    assert F["hd"]["v"]["g"][g] == 512 and F["hd"]["v"]["s"][g] == 256      # per kind, not merged


# --- the browser -------------------------------------------------------------------------------------------------

pw = pytest.importorskip("playwright.sync_api")


def _count(p):
    t = p.pg.inner_text("#zv-count")
    m = re.match(r"([\d,]+) of [\d,]+ models · ([\d,]+) of", t) or re.match(r"([\d,]+) models · ([\d,]+) shapes", t)
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


def _expected(V, direct, pred):
    X, names, node, *_ = V
    ks = [k for k in range(X["n"]) if pred(k)]
    return len(ks), len({node[k] for k in ks})


def test_view_renders(open_page, V):
    X, names, node, mem, unk, by = V
    p = open_page("#v=variants")
    pg = p.pg
    pg.wait_for_selector("#zv-tax .zv-v")
    assert _count(p) == (X["n"], len({*node}))
    gqa = pg.inner_text('#zv-tax .zv-v[data-v="gqa"] .zv-n')
    assert gqa.replace(",", "") == str(by["gqa"]["n"])
    for c in X["comps"]:                                   # every component tab renders variants and no error
        pg.click(f'.zv-ctabs button[data-c="{c["id"]}"]')
        assert pg.locator("#zv-tax .zv-v").count() > 0, c["id"]
        assert f"zc={c['id']}" in p.hash() or c["id"] == "attention"
    for pane in ("detail", "cooc", "timeline", "models"):
        pg.click(f'.zv-ptabs button[data-p="{pane}"]')
        assert pg.locator("#zv-pane > *").count() > 0
    p.assert_clean_text("variants")


def test_combined_filters_equal_a_direct_query(open_page, V, direct):
    X, names, node, mem, unk, by = V
    hd = {f["id"]: f for f in X["facets"]}["hd"]["v"]
    dates = X["models"]["date"]
    p = open_page("#v=variants")
    pg = p.pg
    pg.wait_for_selector("#zv-tax .zv-v")
    pg.check('#zv-tax .zv-v[data-v="gqa"] input')
    pg.check('#zv-tax .zv-v[data-v="swa"] input')
    assert _count(p) == _expected(V, direct, lambda k: mem["gqa"][k] and mem["swa"][k])
    pg.check('#zv-tax .zv-v[data-v="qkn_head"] input')
    pg.click('.zv-dim[data-f="hd"] .bin[data-lo="128"][data-hi="128"]')      # head_dim = 128 on any layer
    pg.select_option("#zv-from", "2025-01")
    want = _expected(V, direct, lambda k: mem["gqa"][k] and mem["swa"][k] and mem["qkn_head"][k]
                     and any(hd[q][k] == 128 for q in hd) and dates[k] >= "2025-01")
    assert _count(p) == want and want[0] > 0
    # the live counts on the other facets are counts within the filter
    sink = int(pg.inner_text('#zv-tax .zv-v[data-v="rope"] .zv-n').split("/")[0].replace(",", "").strip())
    assert sink == _expected(V, direct, lambda k: mem["gqa"][k] and mem["swa"][k] and mem["qkn_head"][k]
                             and any(hd[q][k] == 128 for q in hd) and dates[k] >= "2025-01" and mem["rope"][k])[0]
    # the permalink carries everything; reloading it gives the same set
    h = p.hash()
    assert "zi=gqa,swa,qkn_head" in h and "zn=" in h and "zfrom=2025-01" in h
    q = open_page(h)
    q.pg.wait_for_selector("#zv-tax .zv-v")
    assert _count(q) == want
    assert q.pg.locator("#zv-chips .chip.on").count() == 5


def test_per_kind_and_range_filters(open_page, V, direct):
    X, names, node, mem, unk, by = V
    F = {f["id"]: f for f in X["facets"]}
    hd, ratio = F["hd"]["v"], F["ratio"]["v"]
    # global layers' head dim 512 (Gemma-4), and a GQA ratio >= 8 through the min box
    p = open_page("#v=variants&zn=" + "hd~g~512~512~0")
    p.pg.wait_for_selector("#zv-tax .zv-v")
    assert _count(p) == _expected(V, direct, lambda k: hd.get("g", [None] * X["n"])[k] == 512)
    p2 = open_page("#v=variants&zn=" + "hd~s~512~512~0")
    p2.pg.wait_for_selector("#zv-tax .zv-v")
    assert _count(p2) == _expected(V, direct, lambda k: hd.get("s", [None] * X["n"])[k] == 512)
    p3 = open_page("#v=variants")
    p3.pg.wait_for_selector(".zv-dim[data-f=ratio]")
    p3.pg.fill('.zv-dim[data-f="ratio"] .zv-lo', "8")
    p3.pg.dispatch_event('.zv-dim[data-f="ratio"] .zv-lo', "change")
    want = _expected(V, direct, lambda k: any(isinstance(ratio[q][k], (int, float)) and ratio[q][k] >= 8 for q in ratio))
    assert _count(p3) == want
    # each histogram bar's count is the facet value's count given the other filters (none here: all models)
    bars = p3.pg.eval_on_selector_all('.zv-dim[data-f="kvh"] .bin:not(.unk)', "b => b.map(g => [+g.dataset.lo, +g.dataset.hi, +g.dataset.n])")
    kvh = F["kvh"]["v"]
    sel = [k for k in range(X["n"]) if any(isinstance(ratio[q][k], (int, float)) and ratio[q][k] >= 8 for q in ratio)]
    inbin = lambda x, lo, hi: isinstance(x, (int, float)) and (x == lo if lo == hi else lo <= x < hi)
    assert bars
    for lo, hi, cnt in bars:                              # a model counts once in a bar, whichever kinds fall in it
        assert cnt == sum(1 for k in sel if any(inbin(kvh[q][k], lo, hi) for q in kvh)), (lo, hi)


def test_card_links_and_detail(open_page, V):
    X, names, node, mem, unk, by = V
    p = open_page("#v=variants&zp=detail&zv=mla")
    pg = p.pg
    pg.wait_for_selector(".zv-detail")
    assert pg.inner_text(".zv-detail h3") == "MLA"
    assert by["mla"]["first"][1] in pg.inner_text(".zv-detail")
    assert pg.locator(".zv-ev a").count() > 0                      # code evidence links
    assert pg.locator('.zv-fw tr[data-fw="vllm"]').count() == 1
    link = pg.locator(".zv-detail a.zv-card").first
    repo = link.get_attribute("data-repo")
    link.click()
    pg.wait_for_function("() => !document.body.classList.contains('nosel')")
    assert "m=" + re.sub(r"/", "%2F", repo) in p.hash()
    assert "v=variants" in p.hash()
    i = int(link.get_attribute("data-i"))
    assert node[names.index(repo)] == i


def test_models_table_carries_the_filter(open_page, V, zoo):
    X, names, node, mem, unk, by = V
    p = open_page("#v=models&zi=mla,dsa&zm=1&q=")
    pg = p.pg
    pg.wait_for_selector("#mtbl tbody tr")
    want = sorted({node[k] for k in range(X["n"]) if mem["mla"][k] and mem["dsa"][k]})
    pg.click("#m-more button") if pg.locator("#m-more button").count() else None
    rows = sorted(int(x) for x in pg.eval_on_selector_all("#mtbl tbody tr", "t => t.map(r => r.dataset.i)"))
    assert rows == want
    assert "Variants filter" in pg.inner_text("#m-zv")


def test_csv_and_cooc_and_timeline(open_page, V, tmp_path):
    X, names, node, mem, unk, by = V
    p = open_page("#v=variants&zi=sink_learned")
    pg = p.pg
    pg.wait_for_selector("#zv-tax .zv-v")
    with pg.expect_download() as d:
        pg.click("#zv-csv")
    path = tmp_path / "v.csv"
    d.value.save_as(path)
    lines = path.read_text().strip().split("\n")
    assert len(lines) - 1 == by["sink_learned"]["n"]
    assert lines[0].startswith("repo,shape,model_type")
    # co-occurrence: a cell is the count of models with both; clicking it filters on both
    pg.click('.zv-ptabs button[data-p="cooc"]')
    cell = pg.locator('.zv-mx td.cell[data-r="sink_learned"][data-c="swa"]')
    both = sum(1 for k in range(X["n"]) if mem["sink_learned"][k] and mem["swa"][k])
    assert int(cell.inner_text().replace(",", "")) == both
    cell.click()
    assert _count(p)[0] == both
    # timeline: the stacked segments of a month sum per variant to the dated matching models
    pg.click('.zv-ptabs button[data-p="timeline"]')
    segs = pg.eval_on_selector_all(".zv-tl rect.seg", "r => r.length")
    assert segs > 0
