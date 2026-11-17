"""elephantine.load: one call from a Hub id to the universal host (skips if the checkpoint isn't cached)."""
import pytest
import torch

REPO = "HuggingFaceTB/SmolLM2-135M"


def _cached(repo):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cached(REPO), reason=f"{REPO} not in the local HF cache")
def test_load_picks_loader_and_matches():
    from elephantine.load import load, loader_name
    assert loader_name(REPO, local_files_only=True) == "from_llama"
    host, original = load(REPO, local_files_only=True)
    ids = torch.randint(0, 1000, (2, 12))
    with torch.no_grad():
        a, b = host(ids), original(ids).logits
    a = a.logits if hasattr(a, "logits") else a
    assert (a.argmax(-1) == b.argmax(-1)).all()
    assert (a.double() - b.double()).abs().max() < 1e-3


def test_unknown_model_type_raises():
    from elephantine.loader_coverage import loader_for
    name, why = loader_for({"model_type": "no_such_arch"})
    assert name is None and why


def test_cli_arguments(capsys):
    """python -m elephantine.load: --help and a missing repo are argparse's (exit 0 / 2), not a traceback."""
    from elephantine.load import main
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "repo" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2


def test_cli_unloadable_repo_is_a_message(tmp_path, capsys, monkeypatch):
    """A missing / gated repo (transformers raises OSError) ends with a one-line message and exit 1; a directory
    whose config no loader takes, with a short message and exit 2."""
    import transformers
    from elephantine.load import main
    assert main([str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and len(err) < 400, err

    def gated(*a, **k):
        raise OSError("You are trying to access a gated repo.\n401 Client Error")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", gated)
    assert main(["some/gated-repo"]) == 1
    err = capsys.readouterr().err
    assert "not found, gated or private" in err and "HF_TOKEN" in err and len(err.splitlines()) == 1


@pytest.mark.skipif(not _cached(REPO), reason=f"{REPO} not in the local HF cache")
def test_cli_compares_with_the_original(capsys, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from elephantine.load import main
    assert main([REPO, "Hello", "--dtype", "float32"]) == 0
    assert "via from_llama" in capsys.readouterr().out

