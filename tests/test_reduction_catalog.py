"""The reduction catalog: every type of reduction the headline relation uses has a concrete instance that is built
(a host written from a target) and verified end to end (verify/spot/catalog; the types: verify/spot/reduction_types.py,
generated from graph/model_graph.py).

  exact rules   fp64 max |d logits| at rounding level and identical greedy continuations
  limit rules   the rounding floor of the limit and identical greedy continuations
  approx rules  top-1 >= 99% and >= 90% identical 32-token continuations

Real-checkpoint entries are marked slow and skipped when the checkpoint is not cached; every type also has a
synthetic entry that always runs (test_every_reduction_type_has_a_catalog_test).

  pytest tests/test_reduction_catalog.py -q              # everything
  pytest tests/test_reduction_catalog.py -q -m "not slow"
"""
import pytest
import torch

from verify.spot.catalog import CATALOG, UNINSTANTIATED, WALLS, cached, load_all

load_all()
ENTRIES = {}
for _t, _es in CATALOG.items():
    for _e in _es:
        ENTRIES.setdefault(_e.check.__name__, _e)


def _param(name, e):
    marks = [pytest.mark.slow] if e.slow else []
    return pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize("name", [_param(n, e) for n, e in sorted(ENTRIES.items())])
def test_catalog_entry(name):
    e = ENTRIES[name]
    missing = [r for r in e.needs if not cached(r)]
    if missing:
        pytest.skip(f"not cached: {missing}")
    torch.manual_seed(0)
    r = e.check()
    if e.finding:
        assert not r.passed, f"{name}: the finding no longer reproduces ({r.line()}): {e.finding}"
        pytest.xfail(f"FINDING: {e.finding} ({r.line()})")
    assert r.passed, f"{name} ({', '.join(e.types)}; {e.instance}): {r.line()}"


def test_every_reduction_type_has_a_catalog_test():
    """Recompute the reduction types from the relation's code: each one the headline uses needs a catalog entry, and
    one that is not skipped for want of a checkpoint (a synthetic instance)."""
    from verify.spot.reduction_types import types
    T = types()
    used = sorted(t for t, v in T.items() if v["headline"])
    missing = [t for t in used if t not in CATALOG and t not in UNINSTANTIATED and t not in WALLS]
    for t, name in WALLS.items():                       # a refutation port: its evidence is a failing construction
        assert name in ENTRIES and ENTRIES[name].finding, (t, name)
    assert not missing, f"reduction types with no catalog entry: {missing}"
    stale = [t for t in UNINSTANTIATED if t in CATALOG or t not in used]
    assert not stale, f"UNINSTANTIATED lists types that have an entry or are no longer used: {stale}"
    if UNINSTANTIATED:
        import warnings
        warnings.warn(f"FINDINGS -- reduction types with no end-to-end instance: {sorted(UNINSTANTIATED)}")
    used = [t for t in used if t not in UNINSTANTIATED and t not in WALLS]
    no_fast = [t for t in used if all(e.slow or e.needs for e in CATALOG[t])]
    assert not no_fast, f"reduction types whose only entries need a cached checkpoint: {no_fast}"
    # a type whose only instances are findings (the build does not reproduce the target) is reported, not hidden
    only_findings = [t for t in used if all(e.finding for e in CATALOG[t])]
    if only_findings:
        import warnings
        warnings.warn(f"reduction types instantiated only as FINDINGS (not reproduced): {only_findings}")
