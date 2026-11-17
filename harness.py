"""Shared harness for architecture-subset / emulation-penalty experiments.

Pattern (mirrors the ALiBi->RoPE BLOOM experiment): load a native model, patch
exactly one component at runtime with a specialization of a more expressive
parent architecture, and measure how far the patched logits drift from native.

The thesis these experiments test: when shape (widths, dims, #layers) is allowed
to vary, one architecture can almost always emulate another *exactly*. The
"emulation penalty" is never function error -- it is the resource currency you
must spend (KV memory, width, depth, magnitude, params) and whether spending it
destroys the property that made the emulated child worth using.

Metrics reported everywhere:
  top1_agreement  fraction of positions where argmax logit matches native
  kl_mean/kl_max  KL(native || patched) over the vocab, per position
  max_logit_delta max abs difference of any logit vs native
"""
from __future__ import annotations

import contextlib
import types
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(name, dtype=torch.float32, device=None, local_files_only=True, trust_remote_code=False):
    tok = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, local_files_only=local_files_only, trust_remote_code=trust_remote_code
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tok, device


# --- eval data -------------------------------------------------------------

_FALLBACK_TEXTS = [
    "The theory of relativity describes gravity as the curvature of spacetime, "
    "a geometric reformulation that replaces Newton's action at a distance with "
    "the dynamics of the metric tensor sourced by the stress-energy of matter.",
    "In compiler design, register allocation is often modeled as graph coloring, "
    "where interference between simultaneously live values forms edges and the "
    "number of physical registers bounds the chromatic number of a valid solution.",
    "Mitochondria are frequently described as the powerhouse of the cell because "
    "they synthesize adenosine triphosphate through oxidative phosphorylation "
    "across the inner membrane, coupling electron transport to a proton gradient.",
    "A hypervisor that runs guest code on the host's native instruction set can "
    "execute most instructions directly and only trap on privileged operations, "
    "which is far cheaper than interpreting every instruction through translation.",
] * 16


def eval_batches(tok, device, max_length=128, num_samples=64, min_tokens=16):
    """Return a list of (1, T) input_id tensors from wikitext, or a fallback."""
    texts = None
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 200]
    except Exception as e:  # offline / missing dataset -> deterministic fallback
        print(f"[harness] dataset load failed ({type(e).__name__}); using fallback texts")
        texts = list(_FALLBACK_TEXTS)

    batches = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        if ids.shape[1] >= min_tokens:
            batches.append(ids.to(device))
        if len(batches) >= num_samples:
            break
    return batches


# --- forward pass + metrics ------------------------------------------------

@torch.no_grad()
def logits_for(model, batches):
    outs = []
    for ids in batches:
        outs.append(model(ids).logits.float().cpu())
    return outs


def compare(base_logits, new_logits):
    top1 = tot = 0
    kls = []
    max_delta = 0.0
    for a, b in zip(base_logits, new_logits):
        a, b = a[0], b[0]  # (T, V)
        top1 += (a.argmax(-1) == b.argmax(-1)).sum().item()
        tot += a.shape[0]
        la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
        kls.append((la.exp() * (la - lb)).sum(-1))
        max_delta = max(max_delta, (a - b).abs().max().item())
    kl = torch.cat(kls)
    return dict(
        top1_agreement=top1 / tot,
        kl_mean=kl.mean().item(),
        kl_max=kl.max().item(),
        max_logit_delta=max_delta,
    )


def print_row(label, metrics, width=22):
    print(
        f"  {label:<{width}} "
        f"top1={metrics['top1_agreement']:.4f}  "
        f"KL_mean={metrics['kl_mean']:.3e}  "
        f"KL_max={metrics['kl_max']:.3e}  "
        f"max_dlogit={metrics['max_logit_delta']:.3e}"
    )


# --- runtime patching ------------------------------------------------------

@contextlib.contextmanager
def patched_forwards(modules: Iterable[torch.nn.Module], make_forward: Callable):
    """Temporarily replace .forward on each module with make_forward(module).

    make_forward(module) -> fn(self, *args, **kwargs). Restored on exit.
    """
    saved = []
    for m in modules:
        saved.append((m, m.forward))
        m.forward = types.MethodType(make_forward(m), m)
    try:
        yield
    finally:
        for m, f in saved:
            m.forward = f


def modules_of_type(model, cls_or_name):
    """Collect submodules whose class is `cls` or whose class name == str."""
    out = []
    for _, m in model.named_modules():
        if isinstance(cls_or_name, str):
            if type(m).__name__ == cls_or_name:
                out.append(m)
        elif isinstance(m, cls_or_name):
            out.append(m)
    return out
