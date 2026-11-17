"""Corpora for calibration, distillation and evaluation.

  general  wikitext-103 (raw): train split for calibration / distillation, test split held out
  code     Python sources from installed packages: calibration / distillation from one set of
           packages, held-out from *different* packages (no shared or vendored files)

Sequences are fixed-length token windows ([BOS] + T-1 tokens) of the teacher's tokenizer; each
batch also returns its UTF-8 byte count so perplexities across tokenizers can be compared as
bits per byte.
"""
from __future__ import annotations

import glob
import os
import random

import torch

CODE_TRAIN = ["transformers", "torch", "numpy", "datasets", "accelerate"]
CODE_HELDOUT = ["scipy", "sklearn", "pandas", "sympy", "networkx"]


def _site():
    import transformers
    return os.path.dirname(os.path.dirname(transformers.__file__))


def code_texts(split, max_files=4000, seed=0):
    pk = CODE_TRAIN if split == "train" else CODE_HELDOUT
    files = []
    for p in pk:
        files += sorted(glob.glob(os.path.join(_site(), p, "**", "*.py"), recursive=True))
    random.Random(seed).shuffle(files)
    out = []
    for f in files[:max_files]:
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if len(t) > 2000:
            out.append(t)
    return out


def general_texts(split, max_docs=20000, seed=0):
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test" if split == "heldout" else "train")
    docs, cur = [], []
    for row in d:                                    # rebuild articles from the line records
        t = row["text"]
        if t.startswith(" = ") and not t.startswith(" = = ") and cur:
            docs.append("".join(cur))
            cur = []
        cur.append(t)
        if len(docs) >= max_docs:
            break
    if cur:
        docs.append("".join(cur))
    docs = [x for x in docs if len(x) > 2000]
    random.Random(seed).shuffle(docs)
    return docs


def texts(domain, split, **kw):
    return code_texts(split, **kw) if domain == "code" else general_texts(split, **kw)


def windows(tok, docs, n, T, bos=None, seed=0):
    """n token windows of length T (one per document while they last, then more), plus the
    number of UTF-8 bytes the predicted tokens (positions 1..T-1) cover."""
    rng = random.Random(seed)
    ids, nbytes = [], 0
    i = 0
    while len(ids) < n and i < 50 * n:
        d = docs[i % len(docs)]
        i += 1
        t = tok(d, add_special_tokens=False)["input_ids"]
        need = T - (1 if bos is not None else 0)
        if len(t) < need:
            continue
        s = rng.randrange(0, len(t) - need + 1)
        w = t[s:s + need]
        if bos is not None:
            w = [bos] + w
        ids.append(w)
        nbytes += len(tok.decode(w[1:]).encode("utf-8"))
    return torch.tensor(ids), nbytes


def load_tok(name_or_path):
    """AutoTokenizer, or the raw tokenizer.json when the installed transformers (4.57) does not know
    the tokenizer class of a 5.x checkpoint (Gemma-4)."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except Exception:
        return PreTrainedTokenizerFast(tokenizer_file=os.path.join(name_or_path, "tokenizer.json"))
