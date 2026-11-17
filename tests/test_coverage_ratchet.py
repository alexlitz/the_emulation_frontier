"""Coverage can only go up: every metric of verify/coverage_report.py that is available now must be at or above
its value in verify/coverage_baseline.json, and M1, M10, M15 must be at 100 % once their source is
merged (read from disk). M3 and M6 (planned hard gates whose gaps need post-release Lean theorems / constructions:
verify/coverage_report.DEFERRED_GATES names them) are ratchets whose uncovered items may only shrink. A metric whose source is missing, or read from an unmerged branch for a hard gate,
is skipped with the reason.

Raise the baseline after an improvement:  python -m verify.coverage_report --update-baseline

Run:  pytest tests/test_coverage_ratchet.py
"""
import json
import os

import pytest

from verify import coverage_report as C

IDS = [m[0] for m in C.METRICS]


@pytest.fixture(scope="module")
def current():
    # never generates dist/post_numbers.json here: M14 is skipped when it is missing
    return C.compute(compute_numbers=False, log=lambda *a: None)


@pytest.fixture(scope="module")
def baseline():
    if not os.path.exists(C.BASELINE):
        pytest.skip("no verify/coverage_baseline.json")
    with open(C.BASELINE) as f:
        return json.load(f)["metrics"]


@pytest.mark.parametrize("mid", IDS)
def test_not_below_baseline(mid, current, baseline):
    cur = current[mid]
    if cur["status"] != "ok":
        pytest.skip(f"{mid}: {cur['status']}")
    base = baseline.get(mid)
    if base is None:
        pytest.skip(f"{mid}: no baseline value yet (run --update-baseline)")
    assert C.ge(cur, base), (f"{mid} {cur['name']} fell to {cur['numerator']} / {cur['denominator']} from the "
                             f"baseline {base['numerator']} / {base['denominator']}: {cur['detail']}; "
                             f"not covered: {cur.get('items', [])[:10]}")


@pytest.mark.parametrize("mid", list(C.HARD_GATED))
def test_hard_gate_at_100(mid, current):
    cur = current[mid]
    if cur["status"] != "ok":
        pytest.skip(f"{mid}: {cur['status']}")
    if not cur["merged"]:
        pytest.skip(f"{mid}: source read from {cur['source']}, gated at 100 % once merged")
    assert cur["numerator"] == cur["denominator"], (
        f"{mid} {cur['name']}: {cur['numerator']} / {cur['denominator']}; not covered: {cur.get('items', [])[:20]}")


@pytest.mark.parametrize("mid", sorted(C.DEFERRED_GATES))
def test_deferred_gate_gaps_only_shrink(mid, current):
    """The deferred hard gates (M3, M6): every uncovered item is one recorded in DEFERRED_GATES (no new gap)."""
    cur = current[mid]
    if cur["status"] != "ok":
        pytest.skip(f"{mid}: {cur['status']}")
    new = sorted(set(cur.get("items") or []) - set(C.DEFERRED_GATES[mid]))
    assert not new, f"{mid} {cur['name']}: {cur['numerator']} / {cur['denominator']}; new gaps: {new}"
    assert cur["denominator"] - cur["numerator"] == len(cur.get("items") or []), "items list the whole gap"


def test_baseline_names_known_metrics():
    if not os.path.exists(C.BASELINE):
        pytest.skip("no verify/coverage_baseline.json")
    with open(C.BASELINE) as f:
        base = json.load(f)["metrics"]
    assert set(base) <= set(IDS)
    for mid, b in base.items():
        assert 0 <= b["numerator"] <= b["denominator"] or b["denominator"] == 0, mid


def test_missing_source_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(C.SOURCES, "proofs", [("path", "verify/__no_such_file__.md"),
                                              ("git", "__no_such_branch__", "verify/PROOFS.md")])
    r = C.compute(only={"M4"}, log=lambda *a: None)["M4"]
    assert r["status"].startswith("n/a (source missing:") and r["numerator"] is None
