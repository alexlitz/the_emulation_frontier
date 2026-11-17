"""The reduction catalog: one concrete, end-to-end verified instance of every reduction type the headline relation
uses (verify/spot/reduction_types.py lists the types from the relation's code).

Each entry BUILDS a host from a target and runs both: a target model (a real small checkpoint when the rule has a
cached one, else a small stock-transformers config with random weights) and the host the rule's construction
writes, compared on the same token ids --

  exact   fp64 max |d logits| at rounding level (rel <= EXACT_REL) and identical greedy continuations;
  limit   the rounding floor of the limit (rel <= the entry's tolerance) and identical greedy continuations;
  approx  the high-agreement standard: top-1 >= 99% and >= 90% of the 32-token greedy continuations identical.

  CATALOG[type id] = Entry(check, instance, real, standard)
  python -m verify.spot.catalog            # run every entry, print the table
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import os

torch.set_num_threads(int(os.environ.get("SPOT_THREADS", "4")))      # the machine is shared
F64 = torch.float64
EXACT_REL = 1e-9
N_PROMPTS, PROMPT_LEN, NEW_TOKENS = 8, 10, 32


@dataclass
class Entry:
    types: tuple
    check: object                 # () -> Result
    instance: str                 # "target -> host"
    real: bool = False
    standard: str = "exact"       # exact | limit | approx
    slow: bool = False            # a real checkpoint (pytest marker "slow", skipped when not cached)
    needs: tuple = ()             # Hub repos that must be cached
    finding: str = ""             # the relation claims the type, but this build does NOT reproduce the target (a
    #                               finding: the test expects the failure and fails if it starts passing)


@dataclass
class Result:
    max_abs: float
    rel: float
    top1: float
    identical: int
    n: int
    standard: str = "exact"
    tol: float = EXACT_REL
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def passed(self):
        if self.standard == "approx":
            return self.top1 >= 0.99 and self.identical >= 0.9 * self.n
        return self.rel <= self.tol and self.identical == self.n

    def line(self):
        return (f"max|d| {self.max_abs:.2e} rel {self.rel:.1e} top-1 {self.top1:.1%} identical "
                f"{self.identical}/{self.n}" + (f" ({self.note})" if self.note else ""))


CATALOG: dict = {}
# reduction types the relation uses that have NO end-to-end instance: each one is a finding (the relation credits an
# edge no construction in the repo builds). type id -> why.
UNINSTANTIATED: dict = {}
# round 3, batch 3b: the three findings it listed are walls now, so the headline no longer uses them --
#   slot:ffn:glu->vanilla  a GLU target on a vanilla host (no construction under a pre-norm: spot_glu_vanilla);
#   port:comp_ln / port:comp_par  the 1-D RMS compensators, whose construction is measured only by the
#     scripts of verify/prenorm_rescue/: spot_comp_removed (graph.model_graph.P5_REMOVED_BY); the LayerNorm <- RMSNorm edges on
#     hosts without an lm_head bias are all beta-free targets, backed by catalog r3_ln_fold_beta_free
# pass-5 ports that REMOVE an edge (a refutation, no reduction to build): type id -> the catalog entry whose
# construction fails, as the port says
WALLS: dict = {"port:alibi_labs": "ap_pos_alibi_labs",
               # round 3, batch 3a (spot_phimoe_dup): moe_why refuses SparseMixer targets with m > 1 copies; the failing
               # duplication build is the evidence
               "moe:dup:sparsemixer:m": "fm_phimoe_dup2"}


def entry(*types, instance, real=False, standard="exact", slow=False, needs=(), finding=""):
    def deco(fn):
        e = Entry(tuple(types), fn, instance, real, standard, slow, tuple(needs), finding)
        for t in types:
            CATALOG.setdefault(t, []).append(e)
        return fn
    return deco


def prompts(vocab, n=N_PROMPTS, T=PROMPT_LEN, seed=0, lo=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(lo, vocab, (1, T), generator=g) for _ in range(n)]


@torch.no_grad()
def greedy(fn, ids, new_tokens=NEW_TOKENS):
    """Greedy continuation of ids with fn: ids (1, T) -> logits (1, T, V) (no KV cache: the forward is the model's)."""
    x = ids
    for _ in range(new_tokens):
        nxt = fn(x)[0, -1].argmax()
        x = torch.cat([x, nxt.view(1, 1)], 1)
    return x[0, ids.shape[1]:].tolist()


@torch.no_grad()
def compare(tfn, hfn, vocab=None, standard="exact", tol=EXACT_REL, ids=None, new_tokens=NEW_TOKENS, note="", seed=0,
            n=N_PROMPTS, T=PROMPT_LEN):
    """Target vs host on the same token ids: logits over the prompts, and greedy continuations."""
    ids = ids if ids is not None else prompts(vocab, n=n, T=T, seed=seed)
    mx = sc = 0.0
    agree = tot = 0
    same = 0
    for x in ids:
        a, b = tfn(x).to(F64), hfn(x).to(F64)
        mx = max(mx, (a - b).abs().max().item())
        sc = max(sc, a.abs().max().item())
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        tot += a.shape[1]
        if new_tokens:
            same += greedy(tfn, x, new_tokens) == greedy(hfn, x, new_tokens)
    return Result(mx, mx / max(sc, 1e-300), agree / max(tot, 1), same if new_tokens else len(ids), len(ids),
                  standard, tol, note)


def hf_fn(model):
    """ids -> logits of a transformers causal LM."""
    return lambda x: model(input_ids=x).logits


def tiny(model_type, seed=0, noise=0.08, dtype=F64, **kw):
    """A small stock-transformers causal LM of model_type with random weights (fp64, eager attention)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model(model_type, **kw)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(cfg).to(dtype).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * noise)
    return m


def cached(repo):
    from pathlib import Path
    from huggingface_hub import constants
    d = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots"
    return d.is_dir() and any(f.suffix in (".safetensors", ".bin") for s in d.iterdir() for f in s.iterdir())


def load_all():
    """Import every catalog module (each registers its entries)."""
    import importlib
    import pkgutil
    import verify.spot.catalog as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"verify.spot.catalog.{m.name}")
    return CATALOG
