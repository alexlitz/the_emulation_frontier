"""Read slices of safetensors checkpoints without downloading them, and measure the rank of a
weight delta against the bf16 storage floor.

A repo is either a Hub id (read by HTTP range requests) or a local directory of *.safetensors
files (used by the tests). Tensor names are canonicalized, so `model.language_model.layers.3...`
and `model.layers.3...` compare as the same tensor.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct

import torch

DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
NBYTES = {"BF16": 2, "F16": 2, "F32": 4}
PREFIX = re.compile(r"^(model\.language_model\.|language_model\.model\.|language_model\.|model\.|transformer\.)")


def canon(name):
    return PREFIX.sub("", name)


class Repo:
    """Index of one checkpoint: canonical name -> (file, dtype, shape, byte offsets)."""

    def __init__(self, repo):
        self.repo, self.local = repo, os.path.isdir(repo)
        self.idx, self.header = {}, {}
        for fname, header, hlen in self._headers():
            self.header[fname] = hlen
            for name, ti in header.items():
                if name != "__metadata__":
                    self.idx[canon(name)] = (fname, ti["dtype"], list(ti["shape"]), ti["data_offsets"])

    def _headers(self):
        if self.local:
            for fname in sorted(f for f in os.listdir(self.repo) if f.endswith(".safetensors")):
                with open(os.path.join(self.repo, fname), "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    yield fname, json.loads(f.read(n)), 8 + n
            return
        from huggingface_hub import get_safetensors_metadata
        for fname, fmd in get_safetensors_metadata(self.repo).files_metadata.items():
            hdr = {n: {"dtype": t.dtype, "shape": t.shape, "data_offsets": t.data_offsets}
                   for n, t in fmd.tensors.items()}
            yield fname, hdr, None                     # header length fetched lazily

    def _hlen(self, fname):
        if self.header[fname] is None:
            self.header[fname] = 8 + struct.unpack("<Q", self._bytes(fname, 0, 8))[0]
        return self.header[fname]

    def _bytes(self, fname, start, n):
        if self.local:
            with open(os.path.join(self.repo, fname), "rb") as f:
                f.seek(start)
                return f.read(n)
        import requests
        from huggingface_hub import get_token, hf_hub_url
        tok = get_token()
        h = {"Range": f"bytes={start}-{start + n - 1}", **({"Authorization": f"Bearer {tok}"} if tok else {})}
        for attempt in range(4):
            try:
                r = requests.get(hf_hub_url(self.repo, fname), headers=h, allow_redirects=True, timeout=300)
                r.raise_for_status()
                if len(r.content) == n:
                    return r.content
            except Exception:
                if attempt == 3:
                    raise
        raise IOError(f"short read {self.repo}/{fname}")

    def shape(self, name):
        return self.idx[name][2]

    def read(self, name, rows=None, lead=None):
        """The whole tensor, or its first `rows` rows. For a 3-D tensor (fused experts) `lead`
        picks the expert index and the result is 2-D."""
        fname, dtype, shape, (start, end) = self.idx[name]
        nb = NBYTES[dtype]
        if lead is not None:
            inner = math.prod(shape[1:]) * nb
            start, shape = start + lead * inner, shape[1:]
            end = start + inner
        if rows is not None and len(shape) == 2 and rows < shape[0]:
            end, shape = start + rows * shape[1] * nb, [rows, shape[1]]
        raw = self._bytes(fname, self._hlen(fname) + start, end - start)
        return torch.frombuffer(bytearray(raw), dtype=DT[dtype]).reshape(shape)


def _ulp(w):
    return 2.0 ** (torch.floor(torch.log2(w.abs().clamp_min(1e-30))) - 7)


def bf16_noise(w):
    """Expected squared error of storing `w` in bf16 (uniform rounding, 8-bit mantissa)."""
    return (_ulp(w) ** 2 / 12).sum().item()


def delta_rank(b, d, ranks=(1, 2, 4, 8, 16, 32, 64)):
    """Rank of the edit d - b as a LoRA would need it, separating signal from bf16 rounding.

    signal = singular values above the largest one rounding noise alone would produce (a spectral
    bound for independent noise with the per-element bf16 rounding variances of both sides).
    Returns (r, rel, subulp): r = the smallest rank in `ranks` that covers the signal AND leaves a
    residual within 2x the bf16 floor (None if no listed rank does, 0 if the tensors are equal or
    the delta has no signal above rounding); rel = ||d - b|| / ||b||; subulp = the whole delta sits
    below the rounding floor (an update smaller than bf16 resolves per weight: a full-rank remainder
    there is indistinguishable from rounding, so the rank is a lower bound)."""
    b, d = b.double(), d.double()
    delta = d - b
    tot = (delta ** 2).sum().item()
    rel = math.sqrt(tot) / max(b.norm().item(), 1e-30)
    if tot == 0:
        return 0, 0.0, False
    floor = 4 * bf16_noise(d)                          # (2x the rms floor)^2, in energy
    m, n = delta.shape
    var = (_ulp(d) ** 2 + _ulp(b) ** 2) / 12           # both sides were rounded independently
    subulp = tot <= floor
    if subulp:   # an update below one bf16 step is stored as random 1-ulp flips (P ~ |u|/ulp), so the
        var = torch.maximum(var, delta ** 2)            # noise grows with the update: bound it by the delta itself
    # spectral norm of independent noise with uneven variances (outlier rows/columns of LLM weights
    # round coarser): <= max row norm + max column norm of the standard deviations (Bandeira-van Handel)
    edge = 1.1 * (var.sum(1).max().sqrt() + var.sum(0).max().sqrt()).item()
    s = torch.linalg.svdvals(delta)
    signal = int((s > edge).sum())
    tail = (s ** 2).flip(0).cumsum(0).flip(0)
    if signal == 0 and subulp:
        return 0, rel, True
    for r in ranks:
        if r >= signal and (r >= len(s) or tail[r].item() <= floor):
            return (r if r < min(m, n) else None), rel, subulp
    return None, rel, subulp
