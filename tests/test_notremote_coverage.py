"""elephantine/loader_coverage's "loaded and tested" count credits no test that runs remote code by default:
the rows of tests/conftest.py's REMOTE_CODE_TESTS count only with include_remote=True (--include-remote).

Run:  pytest tests/test_notremote_coverage.py -q
"""
import elephantine.loader_coverage as L
from tests.conftest import REMOTE_CODE_TESTS

_LLAMA = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "num_hidden_layers": 2,
          "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100,
          "hidden_act": "silu", "rms_norm_eps": 1e-6, "max_position_embeddings": 128}


def test_remote_code_tests_is_the_conftest_set():
    assert L.remote_code_tests() == set().union(*REMOTE_CODE_TESTS.values())


def test_default_rows_never_credit_a_remote_code_test():
    remote = L.remote_code_tests()
    full = L.all_tests(include_remote=True)
    assert any(t[0] in remote for rows in full.values() for t in rows)      # there are such rows to drop
    for mt, rows in L.all_tests().items():
        assert not [t for t in rows if t[0] in remote], mt
    assert not [t for rows in L._ALL_TESTS().values() for t in rows if t[0] in remote]


def test_a_model_tested_only_by_remote_code_is_not_counted(monkeypatch):
    """A model_type whose only test row is a remote-code test: untested by default, tested with include_remote."""
    remote_test = sorted(L.remote_code_tests())[0]
    monkeypatch.setattr(L, "TESTS", {"llama": [(remote_test, "remote", "from_llama")]})
    monkeypatch.setattr(L, "READOUT_TESTS", {})
    monkeypatch.setattr(L, "_stock_tests", lambda: {})
    monkeypatch.setattr(L, "_CACHE", {})
    loader, tested, _, _ = L.classify_model(_LLAMA)
    assert loader == "from_llama" and not tested
    assert L.classify_model(_LLAMA, include_remote=True)[1]
    rows, _ = L.inventory([{"name": "x/llama", "cfg": _LLAMA}])
    assert rows["llama"]["loader"] == 1 and rows["llama"]["tested"] == 0
    rows, _ = L.inventory([{"name": "x/llama", "cfg": _LLAMA}], include_remote=True)
    assert rows["llama"]["tested"] == 1
