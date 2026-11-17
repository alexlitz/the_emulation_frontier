"""The zoo page's detail panel (graph/templates/arch_breakdown.js, shipped by graph/zoo_ref.py): browser checks on the
session page (tests/zoo_harness.py). The numbers shown are graph/arch_breakdown.py's, the controls recompute them,
the expandable tables and context curves render, the Look up view shows the same panel, nothing prints a bad value,
and the page stays usable on a phone.

Run:  pytest tests/test_zoo_detail.py
"""
import json

import pytest

from tests.test_zoo_fixes import _fmt_bytes_page, _serve
from tests.zoo_harness import enc

SUM_JS = """() => Object.fromEntries([...document.querySelectorAll('#detail dl.dsum > dt')].map(dt =>
  [dt.textContent, [dt.nextElementSibling.firstChild.textContent, (dt.nextElementSibling.querySelector('.vb') || {}).textContent || '']]))"""


def _node(zoo, repo):
    return next(i for i, nd in enumerate(zoo.data["nodes"]) if repo in nd["members"])


def _open(open_page, zoo, repo, **kw):
    p = open_page("#m=" + enc(repo), **kw)
    if p.pg.locator("#detail").count() == 0:
        pytest.skip("the page was built without the breakdown (graph/data/arch_breakdown.json.gz)")
    p.pg.wait_for_selector("#detail .detail")
    return p


def _plan(zoo, repo):
    from graph import arch_breakdown as BD
    from graph.model_graph import ZOO_CONFIGS, iter_json_configs
    raw = dict(iter_json_configs(ZOO_CONFIGS))[zoo.data["nodes"][_node(zoo, repo)]["members"][0]]
    return BD.plan_ordered(raw), raw


def _unfold(p, key):
    p.pg.evaluate("k => { const d = document.querySelector(`#detail details[data-fold=${k}]`); d.open = true; }", key)
    p.pg.wait_for_function("k => document.querySelector(`#detail details[data-fold=${k}]`).children.length > 1", arg=key)


def _select(p, label, value):
    p.pg.locator(f"#detail label.dctl:has(span:text-is('{label}')) select").select_option(value)


def test_detail_numbers_are_the_breakdown(open_page, zoo):
    from graph import arch_breakdown as BD
    repo = "Qwen/Qwen3-8B"
    p = _open(open_page, zoo, repo)
    pl, raw = _plan(zoo, repo)
    s = p.pg.evaluate(SUM_JS)
    assert s["Weights"][0] == _fmt_bytes_page(p, BD.weight_bytes(pl, "bf16")["total"])
    assert s["Weights"][1] == "trace-verified"
    kv = BD.kv_bytes(BD.kv_elems(pl, 4096), "bf16")
    assert s["KV cache (4,096 tokens × 1)"] == [_fmt_bytes_page(p, kv), "symbolic-proved"]
    assert s["Decode FLOPs / token"][1] == "symbolic-proved"
    assert s["Training FLOPs (≈ 3 × forward)"][1] == "approximation"
    assert any(k.startswith("Decode bound on") for k in s) and s["Time to first token"][1] == "bound"
    # GGUF level, q4_0 KV, batch 8: recomputed
    _select(p, "Weights", "Q4_K_M")
    _select(p, "KV cache", "q4_0")
    p.pg.locator("#detail label.dctl:has(span:text-is('Batch')) input").fill("3")        # B_STEPS[3] = 8
    s = p.pg.evaluate(SUM_JS)
    assert s["Weights"] == [_fmt_bytes_page(p, BD.gguf_sizes(pl, ("Q4_K_M",))["Q4_K_M"]), "Hub-size-checked"]
    assert s["KV cache (4,096 tokens × 8)"][0] == _fmt_bytes_page(p, 8 * BD.kv_bytes(BD.kv_elems(pl, 4096), "q4_0"))
    p.assert_clean_text("detail")


def test_detail_folds_and_layers(open_page, zoo):
    from graph import arch_breakdown as BD
    repo = "google/gemma-3-1b-it"
    p = _open(open_page, zoo, repo)
    pl, raw = _plan(zoo, repo)
    for k in ("params", "flops", "memory", "layers", "roofline"):
        _unfold(p, k)
    rows = p.pg.evaluate("[...document.querySelectorAll('#detail table.dlayers tbody tr')].map(tr => [...tr.cells].map(c => c.textContent))")
    assert len(rows) == pl["layers"]
    kinds = [r[1] for r in rows]
    order = BD.layer_order(pl)
    assert kinds == [" + ".join(BD.group_label(pl["groups"][gi]) for gi in gis) for gis in order]
    assert any(r[2] == "512" for r in rows)                              # the sliding layers' window
    tot = p.pg.evaluate("[...document.querySelector('#detail details[data-fold=params] table').tBodies[0].rows].pop().cells[1].textContent")
    assert tot == f"{BD.params_breakdown(pl)['total']:,}"
    hw = p.pg.evaluate("[...document.querySelectorAll('#detail details[data-fold=roofline] tbody tr')].length")
    assert hw == len(BD.hardware()["hardware"])
    # the curves: three charts; a sliding-window model marks its window
    assert p.pg.locator("#detail .dchart svg").count() == 3
    assert "window 512" in p.pg.inner_text("#detail .dcurves")
    # the context slider moves the KV
    p.pg.locator("#detail label.dctl:has(span:text-is('Context')) input").fill("0")
    s = p.pg.evaluate(SUM_JS)
    assert s["KV cache (256 tokens × 1)"][0] == _fmt_bytes_page(p, BD.kv_bytes(BD.kv_elems(pl, 256), "bf16"))
    p.assert_clean_text("folds")


def test_detail_on_moe_and_hybrid(open_page, zoo):
    for repo in ("Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-Next-80B-A3B-Instruct", "deepseek-ai/DeepSeek-R1"):
        try:
            _node(zoo, repo)
        except StopIteration:
            continue
        p = _open(open_page, zoo, repo)
        _unfold(p, "params")
        t = p.pg.inner_text("#detail details[data-fold=params]")
        assert "moe.routed" in t and "moe.router" in t
        p.assert_clean_text(repo)


def test_lookup_shows_the_detail(open_page, zoo, truth):
    from graph import arch_breakdown as BD
    raw = truth.configs["Qwen/Qwen3-0.6B"]
    p = open_page()
    if p.pg.locator("#tab-lookup").is_hidden():
        pytest.skip("the page was built without the JS port")
    _serve(p, {"x/qwen3-small": (200, json.dumps(raw))})
    p.goto_hash("#v=lookup&id=x%2Fqwen3-small")
    p.pg.wait_for_selector("#lk-out .detail dl.dsum")
    s = dict(p.pg.evaluate("[...document.querySelectorAll('#lk-out dl.dsum > dt')].map(dt => [dt.textContent, dt.nextElementSibling.firstChild.textContent])"))
    pl = BD.plan_ordered(json.loads(json.dumps(raw)))
    assert s["Weights"] == _fmt_bytes_page(p, BD.weight_bytes(pl, "bf16")["total"])
    assert p.pg.locator("#lk-out .detail .vb", has_text="formula only").count() >= 3


@pytest.mark.parametrize("width", [375])
def test_detail_on_a_phone(open_page, zoo, width):
    p = _open(open_page, zoo, "Qwen/Qwen3-8B", viewport={"width": width, "height": 800}, mobile=True)
    over = p.pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert over <= 1
    p.assert_clean_text("phone")


def test_page_weight(zoo):
    """The detail adds the two JS modules, their tables and the compressed plans: well under 1 MB."""
    import re
    m = re.search(r'<script id="bd-detail" type="application/octet-stream">(.*?)</script>', zoo.html, re.S)
    if not m or not m.group(1).strip():
        pytest.skip("built without the breakdown")
    extra = sum(len(x) for x in re.findall(r'<script id="(?:bd-detail|bd-tables|gq-js|bd-js)"[^>]*>(.*?)</script>', zoo.html, re.S))
    assert extra < 600_000, extra
