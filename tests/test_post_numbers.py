"""graph/check_post.py on a small fixture draft with known numbers: it must catch a stale number, report
a missing anchor, accept a correct one (exact, approximate, a table cell, a model name), list the stray
numbers no key covers, and preview fixes without ever writing the draft."""
import json
import os

import pytest

from graph.check_post import check, main, preview, render

DRAFT = """---
topic: fixture
draft_version: 3
---

Why it is interesting:
It collapses 3,933 models; one ~22T spec emulates all of them.

Blog post (draft):

I found that they can all be natively emulated by no more than 106 of them, and a stray 41 here.

| currency budget | frontier | what folds in |
|---|:--:|---|
| exact only | $126$ | nothing |
| $+$ a bigger model | $1$ | the elephantine |

| frontier model | shape | emulates |
|---|---|---:|
| **Qwen-72B** | RoPE·MHA·SwiGLU, $h8192\\ L80\\ \\mathrm{kv}8192$ | $\\mathbf{2{,}083}$ |

Fixed-decay layer: $85\\%$ top-1.

Blog post (published):
no more than 999 of them
"""

NUMBERS = {
    "funnel.models": {"value": 3933, "kind": "int", "source": "fixture", "phrase": "", "category": "funnel",
                      "anchors": [{"re": r"collapses <N> models"}]},
    "frontier.headline": {"value": 104, "kind": "int", "source": "fixture", "phrase": "", "category": "curve",
                          "anchors": [{"re": r"no more than <N> of them"}]},
    "currency.exact": {"value": 126, "kind": "int", "source": "fixture", "phrase": "", "category": "curve",
                       "anchors": [{"table": r"\| currency budget \| frontier \|",
                                    "row": {"col": 0, "re": "exact only"}, "col": 1}]},
    "currency.elephantine": {"value": 1, "kind": "int", "source": "fixture", "phrase": "", "category": "curve",
                             "anchors": [{"table": r"\| currency budget \| frontier \|",
                                          "row": {"col": 0, "re": "a bigger model"}, "col": 1}]},
    "eleph.total_T": {"value": 22.07, "kind": "approx", "source": "fixture", "phrase": "", "category": "eleph",
                      "anchors": [{"re": r"one ~(\d+(?:\.\d+)?)T spec"}]},
    "top.1.name": {"value": "Qwen-72B", "kind": "name", "aliases": ["Qwen-72B", "Qwen-72B-Chat"],
                   "source": "fixture", "phrase": "", "category": "top",
                   "anchors": [{"table": r"\| frontier model \| shape \|", "row": 0, "col": 0}]},
    "top.1.cover": {"value": 2071, "kind": "int", "source": "fixture", "phrase": "", "category": "top",
                    "anchors": [{"table": r"\| frontier model \| shape \|", "row": 0, "col": 2}]},
    "top.1.kv": {"value": 8192, "kind": "int", "source": "fixture", "phrase": "", "category": "top",
                 "anchors": [{"table": r"\| frontier model \| shape \|", "row": 0, "col": 1,
                              "re": r"\\mathrm\{kv\}(\d+)"}]},
    "mixer.fixed_decay_top1": {"value": 85.35, "kind": "approx", "source": "fixture", "phrase": "",
                               "category": "mixer", "anchors": [{"re": r"layer: <N>\\%\$ top-1"}]},
    "axes.vocab": {"value": 144, "kind": "int", "source": "fixture", "phrase": "", "category": "axes",
                   "anchors": [{"re": r"grows from \$\d+\$ to <N>\$ as wide-vocab"}]},
}


@pytest.fixture
def res():
    return check(DRAFT, NUMBERS)


def test_catches_a_stale_number(res):
    bad = {r["key"]: r for r in res["mismatches"]}
    assert set(bad) == {"frontier.headline", "top.1.cover"}
    assert bad["frontier.headline"]["draft_value"] == 106 and bad["frontier.headline"]["value"] == 104
    assert bad["top.1.cover"]["draft"] == "2{,}083"


def test_reports_a_missing_anchor(res):
    assert [r["key"] for r in res["missing"]] == ["axes.vocab"]


def test_accepts_correct_numbers(res):
    ok = {r["key"] for r in res["ok"]}
    assert {"funnel.models", "currency.exact", "currency.elephantine", "eleph.total_T", "top.1.name",
            "top.1.kv", "mixer.fixed_decay_top1"} <= ok


def test_ignores_the_published_part_and_front_matter(res):
    assert all(r["draft_value"] != 999 for r in res["mismatches"] + res["ok"])
    assert "3" not in {u["number"] for u in res["uncovered"]}          # draft_version: 3


def test_lists_uncovered_numbers(res):
    assert [u["number"] for u in res["uncovered"]] == ["41"]


def test_approx_precision():
    e = {"kind": "approx", "value": 0.00955}
    from graph.check_post import compare
    assert compare(e, "0.010")[0] and compare(e, "0.009")[0] and not compare(e, "0.011")[0]
    assert compare({"kind": "approx", "value": 2546}, "2{,}550")[0]
    assert not compare({"kind": "approx", "value": 2546}, "2{,}083")[0]
    assert render({"kind": "approx", "value": 3.77}, "3.8") == "3.8"
    assert render({"kind": "int", "value": 2071}, "2{,}083") == "2{,}071"


def test_fix_preview_never_writes(tmp_path, capsys, res):
    draft = tmp_path / "draft.md"
    draft.write_text(DRAFT, encoding="utf-8")
    nums = tmp_path / "n.json"
    nums.write_text(json.dumps({"meta": {}, "numbers": NUMBERS}), encoding="utf-8")
    before = os.stat(draft).st_mtime_ns
    assert main([str(draft), str(nums), "--fix-preview"]) == 1
    out = capsys.readouterr().out
    assert draft.read_text(encoding="utf-8") == DRAFT and os.stat(draft).st_mtime_ns == before
    assert "-I found that they can all be natively emulated by no more than 106" in out
    assert "+I found that they can all be natively emulated by no more than 104" in out
    assert "\\mathbf{2{,}071}" in preview(DRAFT, res["replacements"])


def test_exit_zero_when_consistent(tmp_path):
    fixed = dict(NUMBERS)
    fixed["frontier.headline"] = dict(NUMBERS["frontier.headline"], value=106)
    fixed["top.1.cover"] = dict(NUMBERS["top.1.cover"], value=2083)
    draft = tmp_path / "draft.md"
    draft.write_text(DRAFT, encoding="utf-8")
    nums = tmp_path / "n.json"
    nums.write_text(json.dumps({"meta": {}, "numbers": fixed}), encoding="utf-8")
    assert main([str(draft), str(nums)]) == 0
    assert main([str(draft), str(nums), "--strict"]) == 1               # axes.vocab's anchor is missing


def test_post_numbers_anchors_compile():
    """every anchor graph.post_numbers writes is a valid pattern (checked on the last run's output)"""
    import re
    from graph.check_post import compile_re
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dist", "post_numbers.json")
    if not os.path.exists(path):
        pytest.skip("run python -m graph.post_numbers first")
    for key, e in json.load(open(path))["numbers"].items():
        for a in e["anchors"]:
            for p in (a.get("re"), a.get("table")):
                if p:
                    compile_re(p)
            if isinstance(a.get("row"), dict):
                re.compile(a["row"]["re"])


def test_act_cross_numbers_handles_an_activation_outside_the_table():
    """act.activation_cross.ACTS has GeLU_tanh, which the post's 8 x 8 table has no row or column for: its cells
    are recorded without an anchor (they crashed build() with "'GeLU_tanh' is not in list"); the table's own
    cells keep their anchor, at the right column."""
    from act.activation_cross import ACTS
    from graph.check_post import check
    from graph.post_numbers import ACT_TABLE_NAMES, Numbers, act_cross_numbers
    names = list(ACTS)
    assert "GeLU_tanh" in names and "GeLU_tanh" not in ACT_TABLE_NAMES
    x = {Y: {X: 0.0 if X == Y else 0.25 for X in names} for Y in names}
    N = Numbers()
    act_cross_numbers(N, x)
    assert len(N.d) == len(names) * (len(names) - 1)
    assert N.d["actx.GeLU<-GeLU_tanh"]["anchors"] == [] and N.d["actx.GeLU_tanh<-SiLU"]["anchors"] == []
    (a,) = N.d["actx.GeLU<-SiLU"]["anchors"]
    assert a["col"] == 1 + list(ACT_TABLE_NAMES).index("SiLU")
    draft = "| target | ReLU | relu² | SiLU |\n|---|---|---|---|\n| **GeLU** | $0.1$ | $0.2$ | $0.25$ |\n"
    keys = ("actx.GeLU<-SiLU", "actx.GeLU<-GeLU_tanh")
    res = check(draft, {k: N.d[k] for k in keys})
    assert [r["key"] for r in res["ok"]] == ["actx.GeLU<-SiLU"] and not res["mismatches"] and not res["missing"]
