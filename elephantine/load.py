"""One call from a Hub model to the universal host carrying it.

    from elephantine.load import load
    u, original = load("HuggingFaceTB/SmolLM2-135M")
    logits = u(input_ids)            # the same model, run by the one universal architecture

The right `from_*` loader is picked from the model's config (elephantine/loader_coverage.py, the
same table that counts loader coverage). Families whose code ships only as Hub remote code have no
loader here and raise with the reason. Computes in float64 by default, the precision the
constructions are checked at; pass dtype=torch.float32 for the original's own precision.

    python -m elephantine.load HuggingFaceTB/SmolLM2-135M "The capital of France is" [--dtype float32]
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch                                                            # noqa: E402

from elephantine import universal as U                                 # noqa: E402
from elephantine.loader_coverage import loader_for                      # noqa: E402


def loader_name(repo_id: str, **kw) -> str:
    """The from_* loader for repo_id's config; ValueError (with the reasons) if none accepts it."""
    from transformers import AutoConfig
    from graph.model_graph import _lm_config
    cfg = _lm_config(AutoConfig.from_pretrained(repo_id, **kw).to_dict())
    name, why = loader_for(cfg)
    if name is None:
        raise ValueError(f"{repo_id}: no universal-host loader ({why})")
    return name


def load(repo_id: str, dtype=torch.float64, **kw):
    """(host, original): the original model loaded with transformers, and the universal host set to
    it by the matching from_* loader. kw go to from_pretrained (revision=, token=, ...)."""
    return _load(repo_id, loader_name(repo_id, **kw), dtype, **kw)


def _load(repo_id, name, dtype, **kw):
    from transformers import AutoModelForCausalLM
    original = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype, **kw).eval()
    host = getattr(U, name)(original).eval()
    return host, original


def main(argv=None):
    ap = argparse.ArgumentParser(description="Load a Hub model into the universal host and compare it with the "
                                             "original on one prompt.")
    ap.add_argument("repo", help="Hub repo id or local directory of a decoder LM (stock transformers class)")
    ap.add_argument("text", nargs="?", default="The capital of France is", help="the prompt")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    try:
        name = loader_name(args.repo)
        host, original = _load(args.repo, name, getattr(torch, args.dtype))
        tok = AutoTokenizer.from_pretrained(args.repo)
    except ValueError as e:                     # no loader accepts the config, or transformers knows no such type
        msg = str(e).splitlines()[0]
        print(msg if len(msg) <= 300 else msg[:300] + " ...", file=sys.stderr)
        return 2
    except OSError as e:                        # not found, gated or private, or no network
        print(f"{args.repo}: could not be loaded -- not found, gated or private (for a gated repo set HF_TOKEN and "
              f"accept its licence on the Hub): {str(e).splitlines()[0]}", file=sys.stderr)
        return 1
    ids = tok(args.text, return_tensors="pt").input_ids
    with torch.no_grad():
        a, b = host(ids), original(ids).logits
        a = a.logits if hasattr(a, "logits") else a
    print(f"{args.repo} via {name}: max |delta logits| vs the original = "
          f"{(a.double() - b.double()).abs().max().item():.2e}; "
          f"argmax agreement {(a.argmax(-1) == b.argmax(-1)).float().mean().item():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
