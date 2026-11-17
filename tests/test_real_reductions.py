"""Reductions verified on REAL LAYERS from REAL, TRAINED Hugging Face checkpoints.

Loads small cached models (gpt2, opt-125m, pythia-70m, Qwen2.5-0.5B, bloom-560m),
patches / remaps ONE real component, and checks the reduction on the trained model's
own forward (see hf/real_reductions.py). Each group SKIPS cleanly if its checkpoint
is not in the local Hub cache, so this is a no-op in a bare CI and a full check on a
machine with the models.

Run:  pytest tests/test_real_reductions.py -q
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import harness as H
from hf.real_reductions import GROUPS, PARTIAL


def _assert_rows(rows, name):
    assert rows, f"{name}: no checks ran"
    for r in rows:
        if "SKIP" in r["model"]:                            # a single check couldn't run its module
            continue
        exact_like = r["kind"] in ("exact", "regime", "limit")
        cond = (r["worst"] < r["thr"]) if exact_like else (r["worst"] > r["thr"])
        assert cond and r["ok"], (
            f"{r['name']} on {name}: worst |Δ| = {r['worst']:.2e} "
            f"(thr {r['thr']:.0e}, kind {r['kind']})")


@pytest.mark.parametrize("name,fn", GROUPS, ids=[g[0].split("/")[-1] for g in GROUPS])
def test_reduction_on_real_trained_model(name, fn):
    try:
        model, _, _ = H.load(name, dtype=torch.float64, device="cpu")
    except Exception as e:                                   # not cached / offline -> skip, don't fail
        pytest.skip(f"{name} unavailable: {type(e).__name__}")
    _assert_rows(fn(model, model.config), name)


@pytest.mark.parametrize("fn", PARTIAL, ids=[f.__name__ for f in PARTIAL])
def test_reduction_on_partial_loaded_weights(fn):
    rows = fn()
    if rows and "SKIP" in rows[0]["model"]:                  # weights not cached -> skip
        pytest.skip(f"{rows[0]['model']}")
    _assert_rows(rows, fn.__name__)
