"""Build (and optionally upload) the Hub demo of the ONE universal elephantine architecture.

The repo (`alexlitz/elephantine`, private) holds remote code, `configuration_elephantine.py` +
`modeling_elephantine.py` (a self-contained, verbatim copy of elephantine/universal.py's
architecture), and one subfolder per SETTING: the same config class and the same model class, with a
different `config.json` (host sizes, default mask) and `model.safetensors` (weights and branch
coefficients). Each setting is produced by an existing `from_*` loader from a real checkpoint, then
reloaded through `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and measured
against the original model's own class: max |delta logits| vs the original in fp32 and in fp64 (its
fp32 casts lifted, `tests.test_universal_elephantine._lift_fp32`), and greedy-continuation agreement.

Storage: every tensor is written in the narrowest dtype that holds it EXACTLY (bf16 for a bf16
checkpoint's copied weights, fp16 for an fp16 one, fp32 / fp64 for folded or derived values), and
tensors still at the architecture's default are left out (the model class recreates them), so a
reload is bit-identical to the in-memory host. The model computes in fp64.

Run (from the repo root):
  python -m elephantine.push_hub_demo --vendor                 # refresh the vendored copy only
  python -m elephantine.push_hub_demo --out DIR                # build + measure every setting
  python -m elephantine.push_hub_demo --out DIR --only smollm2-135m pythia-160m
  python -m elephantine.push_hub_demo --out DIR --push         # build, measure, then upload (private)
  python -m elephantine.push_hub_demo --out DIR --push --skip-build   # upload an existing build
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import elephantine.universal as U                                     # noqa: E402
from elephantine.universal import UNorm, UniversalElephantine         # noqa: E402

F64 = torch.float64
HUB = Path(__file__).resolve().parent / "hub"
REMOTE_FILES = ("configuration_elephantine.py", "modeling_elephantine.py")
REPO_ID = "alexlitz/elephantine"
DEFAULT_OUT = Path(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier"))) / "elephantine-hub-demo"
GITHUB = "https://github.com/alexlitz/the_emulation_frontier"
BLOG = "https://alexlitzenberger.com/blog/the_emulation_frontier"
AUTO_MAP = {"AutoConfig": "configuration_elephantine.ElephantineConfig",
            "AutoModelForCausalLM": "modeling_elephantine.ElephantineForCausalLM"}

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Once upon a time, there was a little"]
NEW_TOKENS = 16           # greedy continuation length (autoregressive settings)
DIFF_MASKS = 8            # masked positions decoded one at a time (diffusion setting)


# --- the settings: one architecture, many weight settings -----------------------------------------

@dataclasses.dataclass
class Setting:
    name: str
    repo: str
    loader: str
    family: str                   # one line for the model card
    causal: bool = True
    inflate: dict | None = None   # wider / deeper host: UConfig overrides (d, heads, shared_ff, L)
    diffusion: bool = False
    note: str = ""


SETTINGS = {s.name: s for s in [
    Setting("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", "from_llama",
            "Llama: softmax attention, RoPE, GQA (tiled to MHA), RMSNorm, SwiGLU"),
    Setting("smollm2-135m-inflated", "HuggingFaceTB/SmolLM2-135M", "from_llama",
            "the same SmolLM2-135M inside a wider and deeper host (extra units zeroed)",
            inflate=dict(d=768, heads=12, shared_ff=2048, L=40)),
    Setting("pythia-160m", "EleutherAI/pythia-160m", "from_gpt_neox",
            "GPT-NeoX: LayerNorm, parallel attention + MLP block, partial RoPE (25% of each head), "
            "ungated exact (erf) GeLU MLP, biases"),
    Setting("falcon-h1-tiny-90m", "tiiuae/Falcon-H1-Tiny-90M-Base", "from_falcon_h1",
            "Falcon-H1 hybrid: a Mamba-2 SSM (through the gated-delta recurrence) and GQA attention in "
            "parallel in every layer, then SwiGLU; muP multipliers folded into the weights"),
    Setting("mamba-130m-mic", "huggingface-mic/mamba-130m-hf-mic", "from_mamba",
            "Mamba-1 (selective SSM through the gated-delta recurrence): the code path of "
            "state-spaces/mamba-130m-hf",
            note="Random-weight test fixture (seed 17) published by huggingface-mic: the real "
                 "MambaForCausalLM class and mamba-130m-hf's tokenizer, shrunk to d 64 and 2 layers. "
                 "It is not a trained model; its continuations are not meaningful text. The trained "
                 "mamba-130m-hf needs a ~33B-parameter host (the SSM branch's knob projections are "
                 "dense: (2 x ssm_in) x (ssm_heads x dk) = 6144 x 24576 per projection per layer), "
                 "~264 GB in fp64, so it is not part of this demo."),
    Setting("qwen3-0.6b-diffusion-mdlm", "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "from_llama",
            "masked-diffusion LM (MDLM, from Qwen3-0.6B): bidirectional attention, per-head qk-norm; "
            "runs under the full mask (causal=False)",
            causal=False, diffusion=True),
]}


# --- vendoring: the remote module is a verbatim copy of universal.py's architecture ----------------

def _region(src: str, tag: str):
    start, end = f"# >>> vendored: {tag}\n", f"# <<< vendored: {tag}\n"
    i, j = src.index(start) + len(start), src.index(end)
    return i, j


def vendored_sources() -> dict[str, str]:
    """The exact text each vendored region must hold."""
    from attn.gated_delta_layer import gd_recurrence
    uni = (ROOT / "elephantine" / "universal.py").read_text()
    a = uni.index("F64 = torch.float64\n")
    b = uni.index("# --- loaders: each family is a weight setting")
    return {"attn/gated_delta_layer.py gd_recurrence": inspect.getsource(gd_recurrence),
            "elephantine/universal.py": uni[a:b].rstrip("\n") + "\n"}


def vendor(check: bool = False) -> bool:
    """Write (or with check=True only compare) the vendored regions of hub/modeling_elephantine.py.
    Returns True if the file already matched."""
    path = HUB / "modeling_elephantine.py"
    src = path.read_text()
    new = src
    for tag, text in vendored_sources().items():
        i, j = _region(new, tag)
        new = new[:i] + text + new[j:]
    same = new == src
    if not check and not same:
        path.write_text(new)
    return same


def remote_module():
    """Import hub/modeling_elephantine.py as a plain module (no transformers dynamic-module cache)."""
    sys.path.insert(0, str(HUB))
    try:
        spec = importlib.util.spec_from_file_location("hub_modeling_elephantine", HUB / "modeling_elephantine.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod                     # (dataclasses look their module up)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(HUB))
    return mod


# --- loading the originals -------------------------------------------------------------------------

_A2D = {}


def _snapshot(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo)


def load_original(s: Setting, dtype):
    """The original model through its OWN class, on CPU, eager attention."""
    from transformers import AutoModelForCausalLM
    path = _snapshot(s.repo)
    if s.diffusion:                       # its remote code (modeling_qwen3.py: A2DQwen3LMHeadModel), run as is
        if s.repo not in _A2D:            # (imported directly: its `if __name__ == "__main__"` block imports
            spec = importlib.util.spec_from_file_location("a2d_qwen3_remote", os.path.join(path, "modeling_qwen3.py"))
            mod = importlib.util.module_from_spec(spec)     # dllm, which transformers' import check rejects)
            spec.loader.exec_module(mod)
            _A2D[s.repo] = mod.A2DQwen3LMHeadModel
        return _A2D[s.repo].from_pretrained(path, dtype=dtype, attn_implementation="eager").eval(), path
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager").eval(), path


def _revision(path):
    return Path(path).name if "snapshots" in str(path) else None


# --- inflation: the same target in a wider / deeper host, extra units zeroed ------------------------

@torch.no_grad()
def inflate(u: UniversalElephantine, *, d=None, heads=None, shared_ff=None, L=None) -> UniversalElephantine:
    """Embed host `u` in a bigger host of the same architecture. Every tensor is copied into the
    leading block of its bigger counterpart; the rest keeps the architecture's default (zero
    projections, so extra heads / FFN units / layers add exactly 0). The target's layers are spread
    over the deeper stack; the layers between them are left at their default (all-zero output
    projections: identity layers). A wider residual stream d needs RMSNorm compensation, the same
    weight setting elephantine/hidden_width.py proves exact: the extra coordinates stay 0, so over
    d' dims  rms'(x) = sqrt(d / d') rms(x)  iff  eps' = eps d / d',  and  w' = w sqrt(d / d').
    LayerNorm (mean-centred) targets are refused for d' > d: their mean is not carried that way."""
    c = u.cfg
    big_c = dataclasses.replace(c, d=d or c.d, heads=heads or c.heads, shared_ff=shared_ff or c.shared_ff,
                                L=L or c.L)
    assert big_c.d >= c.d and big_c.heads >= c.heads and big_c.shared_ff >= c.shared_ff and big_c.L >= c.L
    if c.attnres_block or c.attnres_full or c.hc_mult or c.ple_dim:
        raise NotImplementedError("inflate: depth-mixing residuals / PLE targets are not inflated here")
    big = UniversalElephantine(big_c)
    big.embed.weight.zero_()                                    # (nn.Embedding's own init is random)
    pos =[i * big_c.L // c.L for i in range(c.L)]                # distinct, increasing
    sb = big.state_dict()
    for k, v in u.state_dict().items():
        kb = re.sub(r"^layers\.(\d+)\.", lambda m: f"layers.{pos[int(m.group(1))]}.", k)
        dst = sb[kb]
        if dst.shape == v.shape:
            dst.copy_(v)
        else:
            assert dst.dim() == v.dim() and all(a >= b for a, b in zip(dst.shape, v.shape)), (k, dst.shape, v.shape)
            dst[tuple(slice(0, n) for n in v.shape)].copy_(v)
    if big_c.d != c.d:
        r = c.d / big_c.d
        for name, m in big.named_modules():
            if not isinstance(m, UNorm) or m.ident != 0:
                continue
            if m.centre != 0 or m.pre_c != 0 or m.gate_c != 0:
                raise NotImplementedError(f"inflate: {name} is not a plain RMSNorm; d cannot grow")
            m.eps.mul_(r)
            m.weight[: c.d].mul_(math.sqrt(r))
            m.weight[c.d:].zero_()
    return big


# --- building a setting ----------------------------------------------------------------------------

def build_host(s: Setting, target=None):
    """(host, base) for setting s; base is the un-inflated host (same object when not inflated)."""
    if target is None:
        target, _ = load_original(s, F64)
    base = getattr(U, s.loader)(target).eval()
    host = inflate(base, **s.inflate).eval() if s.inflate else base
    return host, base


def _narrow(t):
    """The narrowest dtype that holds t exactly (bf16 / fp16 / fp32), else t as is."""
    if not t.is_floating_point():
        return t
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        n = t.to(dt)
        if torch.equal(n.to(t.dtype), t):
            return n
    return t


def _pattern(k):
    return re.sub(r"^layers\.\d+\.", "layers.*.", k)


def _fmt(v):
    v = v.reshape(-1).tolist()
    v = [x if math.isfinite(x) else str(x) for x in v]
    return v[0] if len(v) == 1 else v


@torch.no_grad()
def setting_state(host: UniversalElephantine):
    """(stored, branches): the tensors that differ from the architecture's defaults (narrowed), and a
    readable summary of the small branch coefficients / buffers among them."""
    ref = UniversalElephantine(host.cfg)                     # the defaults
    rsd = ref.state_dict()
    stored, small = {}, {}
    for k, v in host.state_dict().items():
        dv = rsd[k]
        if v.shape == dv.shape and torch.equal(v, dv):
            continue
        stored[k] = _narrow(v.detach()).contiguous().clone()
        if v.numel() <= 16 and not k.startswith(("embed", "lm_head")):
            small.setdefault(_pattern(k), []).append(v.detach().clone())
    del ref, rsd
    branches = {}
    for p, vals in sorted(small.items()):
        same = all(x.shape == vals[0].shape and torch.equal(x, vals[0]) for x in vals)
        n = f" ({len(vals)} of {host.cfg.L} layers)" if p.startswith("layers.*") and len(vals) != host.cfg.L else ""
        branches[p + n] = _fmt(vals[0]) if same else "varies by layer"
    return stored, branches


def save_setting(s: Setting, host, folder: Path, tokenizer_path=None, revision=None):
    """config.json + model.safetensors + the remote code (+ the source tokenizer) in `folder`."""
    from safetensors.torch import save_file
    sys.path.insert(0, str(HUB))
    try:
        from configuration_elephantine import ElephantineConfig
    finally:
        sys.path.remove(str(HUB))
    folder.mkdir(parents=True, exist_ok=True)
    stored, branches = setting_state(host)
    host_dict = dataclasses.asdict(host.cfg)
    host_dict["engram_orders"] = list(host_dict["engram_orders"])
    cfg = ElephantineConfig(host=host_dict, causal=s.causal, branches=branches,
                            stored_tensors=sorted({_pattern(k) for k in stored}),
                            source={"repo": s.repo, "revision": revision, "loader": f"elephantine.universal.{s.loader}",
                                    "inflate": s.inflate},
                            architectures=["ElephantineForCausalLM"], auto_map=AUTO_MAP)
    cfg.save_pretrained(folder)
    save_file({"model." + k: v for k, v in stored.items()}, str(folder / "model.safetensors"), metadata={"format": "pt"})
    for f in REMOTE_FILES:
        shutil.copy(HUB / f, folder / f)
    if tokenizer_path is not None:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(tokenizer_path).save_pretrained(folder)
    counts = {}
    for v in stored.values():
        counts[str(v.dtype).replace("torch.", "")] = counts.get(str(v.dtype).replace("torch.", ""), 0) + v.numel()
    return {"stored_params": sum(v.numel() for v in stored.values()),
            "host_params": sum(v.numel() for v in host.state_dict().values()),
            "stored_dtypes": counts, "bytes": (folder / "model.safetensors").stat().st_size}


def load_setting(folder):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(folder), trust_remote_code=True).eval()


# --- measuring ---------------------------------------------------------------------------------------

def _lift():
    from tests.test_universal_elephantine import _lift_fp32
    return _lift_fp32()


@torch.no_grad()
def mdlm_greedy(logits_fn, prompt, mask_id, n):
    """MDLM decoding, greedy: append n mask tokens; each step fills the masked position whose argmax
    probability is highest with that argmax token. Returns the n filled tokens (in position order)."""
    x = torch.cat([prompt, torch.full((1, n), mask_id, dtype=prompt.dtype)], 1)
    for _ in range(n):
        p = logits_fn(x)[0].double().softmax(-1)
        conf, tok = p.max(-1)
        conf = conf.masked_fill(x[0] != mask_id, -1.0)
        i = int(conf.argmax())
        x[0, i] = tok[i]
    return x[0, prompt.shape[1]:].tolist()


def _greedy(model_or_fn, ids, s, mask_id=None, host=False):
    if s.diffusion:
        fn = (lambda x: model_or_fn(input_ids=x).logits)
        return mdlm_greedy(fn, ids, mask_id, DIFF_MASKS)
    kw = dict(max_new_tokens=NEW_TOKENS, do_sample=False, num_beams=1, pad_token_id=0)
    if host:
        kw["use_cache"] = False
    out = model_or_fn.generate(ids, attention_mask=torch.ones_like(ids), **kw)
    return out[0, ids.shape[1]:].tolist()


def _agree(a, b):
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"identical": a == b, "first_divergence": None if a == b else n,
            "token_match": sum(x == y for x, y in zip(a, b)) / max(len(a), 1)}


@torch.no_grad()
def measure(s: Setting, host_in_memory, base, reloaded, tok, fp64_ref=True):
    """Max |delta logits| of the reloaded setting vs the original (fp32; fp64 with its casts lifted),
    vs the in-memory host (0.0 expected), and greedy agreement."""
    res = {"prompts": PROMPTS}
    sd_a, sd_b = host_in_memory.state_dict(), reloaded.model.state_dict()
    res["reload_bit_identical_state"] = set(sd_a) == set(sd_b) and all(
        sd_a[k].dtype == sd_b[k].dtype and torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    mask_id = tok.mask_token_id if s.diffusion else None
    inputs = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        if s.diffusion:
            ids = torch.cat([ids, torch.full((1, DIFF_MASKS), mask_id)], 1)
        inputs.append(ids)
    host_logits = [reloaded(input_ids=x).logits for x in inputs]
    res["vs_in_memory_host"] = max((h - host_in_memory(x, causal=s.causal)).abs().max().item()
                                   for h, x in zip(host_logits, inputs))
    if base is not host_in_memory:
        res["vs_uninflated_host"] = max((h - base(x, causal=s.causal)).abs().max().item()
                                        for h, x in zip(host_logits, inputs))
    t32, _ = load_original(s, torch.float32)
    ref32 = [t32(input_ids=x).logits.double() for x in inputs]
    res["logit_scale"] = max(r.abs().max().item() for r in ref32)
    res["max_abs_diff_fp32"] = max((h - r).abs().max().item() for h, r in zip(host_logits, ref32))
    res["argmax_match_fp32"] = sum((h.argmax(-1) == r.argmax(-1)).sum().item() for h, r in zip(host_logits, ref32)) / \
        sum(x.numel() for x in inputs)
    g32 = [_greedy(t32, x[:, :-DIFF_MASKS] if s.diffusion else x, s, mask_id) for x in inputs]
    del t32, ref32
    gh = [_greedy(reloaded, x[:, :-DIFF_MASKS] if s.diffusion else x, s, mask_id, host=True) for x in inputs]
    res["greedy_host"] = gh
    res["greedy_vs_fp32"] = [_agree(a, b) for a, b in zip(gh, g32)]
    if fp64_ref:
        t64, _ = load_original(s, F64)
        with _lift():
            ref64 = [t64(input_ids=x).logits for x in inputs]
            g64 = [_greedy(t64, x[:, :-DIFF_MASKS] if s.diffusion else x, s, mask_id) for x in inputs]
        res["max_abs_diff_fp64"] = max((h - r).abs().max().item() for h, r in zip(host_logits, ref64))
        res["greedy_vs_fp64"] = [_agree(a, b) for a, b in zip(gh, g64)]
        del t64, ref64
    res["greedy_text"] = [tok.decode(g) for g in gh]
    return res


def build_setting(s: Setting, out: Path, fp64_ref=True, log=print):
    """Build, save, reload through AutoModelForCausalLM, measure. Writes results.json + README.md."""
    from transformers import AutoTokenizer
    t0 = time.time()
    target, path = load_original(s, F64)
    target_params = sum(p.numel() for p in target.parameters())
    host, base = build_host(s, target)
    del target
    folder = out / s.name
    if folder.exists():
        shutil.rmtree(folder)
    info = save_setting(s, host, folder, tokenizer_path=path, revision=_revision(path))
    log(f"[{s.name}] built + saved in {time.time() - t0:.0f}s: {info['bytes'] / 1e6:.0f} MB")
    reloaded = load_setting(folder)
    tok = AutoTokenizer.from_pretrained(folder)
    res = measure(s, host, base, reloaded, tok, fp64_ref=fp64_ref)
    res.update(info, name=s.name, repo=s.repo, target_params=target_params, revision=_revision(path), loader=s.loader, causal=s.causal,
               inflate=s.inflate, host=dataclasses.asdict(host.cfg), seconds=time.time() - t0)
    res["host"]["engram_orders"] = list(res["host"]["engram_orders"])
    (folder / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (folder / "README.md").write_text(setting_card(s, res, standalone=True))
    log(f"[{s.name}] fp32 {res['max_abs_diff_fp32']:.2e}  fp64 {res.get('max_abs_diff_fp64', float('nan')):.2e}  "
        f"reload exact {res['reload_bit_identical_state']} / {res['vs_in_memory_host']:.1e}  "
        f"greedy {[g['identical'] for g in res['greedy_vs_fp32']]}  ({time.time() - t0:.0f}s)")
    return res


# --- the model card --------------------------------------------------------------------------------

def _e(x):
    return "n/a" if x is None else ("0" if x == 0 else f"{x:.1e}")


def _greedy_line(gs):
    n = sum(g["identical"] for g in gs)
    return f"{n} / {len(gs)} identical"


def setting_card(s: Setting, r: dict, standalone=False) -> str:
    h = r["host"]
    dt = ", ".join(f"{k} {v / 1e6:.1f}M" for k, v in sorted(r["stored_dtypes"].items()))
    head = "#" if standalone else "###"
    lines = [f"{head} `{s.name}`", ""]
    if standalone:
        lines += [f"A setting of the universal elephantine architecture; see the repo's main model card "
                  f"(`README.md` at the root). Source: [`{s.repo}`](https://huggingface.co/{s.repo}).", ""]
    lines += [f"- **Source:** [`{s.repo}`](https://huggingface.co/{s.repo})"
              + (f" @ `{r['revision'][:10]}`" if r.get("revision") else "")
              + f", loaded with `elephantine.universal.{s.loader}`"
              + (f", then inflated ({', '.join(f'{k} {v}' for k, v in s.inflate.items())})" if s.inflate else "") + ".",
              f"- **Family:** {s.family}.",
              f"- **Host:** d {h['d']}, L {h['L']}, {h['heads']} heads x {h['head_dim']}, FFN {h['shared_ff']}, "
              f"SSM in {h['ssm_in']} / {h['ssm_heads']} heads (dk {h['dk']}, dv {h['dv']}), vocab {h['vocab']}; "
              f"{r['host_params'] / 1e6:.0f}M host parameters, {r['stored_params'] / 1e6:.0f}M stored "
              f"(non-default), {r['bytes'] / 1e6:.0f} MB on disk ({dt}).",
              f"- **Mask:** {'causal' if s.causal else 'full (bidirectional): the config sets causal=false'}."]
    if s.note:
        lines.append(f"- **Note:** {s.note}")
    lines += ["", "| measured (3 prompts) | value |", "|---|---|",
              f"| max \\|Δ logits\\| vs the original in fp32 | {_e(r['max_abs_diff_fp32'])} "
              f"(max \\|logit\\| {r['logit_scale']:.1f}) |"]
    if "max_abs_diff_fp64" in r:
        lines.append(f"| max \\|Δ logits\\| vs the original in fp64 (its fp32 casts lifted) | {_e(r['max_abs_diff_fp64'])} |")
    if "vs_uninflated_host" in r:
        lines.append(f"| max \\|Δ logits\\| vs the un-inflated host | {_e(r['vs_uninflated_host'])} |")
    lines.append(f"| reload through AutoModel vs the in-memory host | {_e(r['vs_in_memory_host'])} "
                 f"(state {'bit-identical' if r['reload_bit_identical_state'] else 'DIFFERS'}) |")
    lines.append(f"| argmax agreement with the fp32 original (all positions) | {100 * r['argmax_match_fp32']:.1f}% |")
    what = f"MDLM greedy fill of {DIFF_MASKS} masks" if s.diffusion else f"greedy {NEW_TOKENS} new tokens"
    lines.append(f"| {what}, vs the fp32 original | {_greedy_line(r['greedy_vs_fp32'])} |")
    if "greedy_vs_fp64" in r:
        lines.append(f"| {what}, vs the fp64 original | {_greedy_line(r['greedy_vs_fp64'])} |")
    lines += ["", "```python", "from transformers import AutoModelForCausalLM, AutoTokenizer",
              f'm = AutoModelForCausalLM.from_pretrained("{REPO_ID}", subfolder="{s.name}", trust_remote_code=True)',
              f'tok = AutoTokenizer.from_pretrained("{REPO_ID}", subfolder="{s.name}")']
    if s.diffusion:
        lines += ['ids = tok("The capital of France is", return_tensors="pt").input_ids',
                  "masked = torch.cat([ids, torch.full((1, 8), tok.mask_token_id)], 1)  # import torch",
                  "logits = m(input_ids=masked).logits      # full mask (config.causal = False)"]
    else:
        lines += ['ids = tok("The capital of France is", return_tensors="pt").input_ids',
                  "out = m.generate(ids, max_new_tokens=16, do_sample=False)",
                  "print(tok.decode(out[0]))"]
    lines += ["```", ""]
    if not s.diffusion and not s.note:
        lines += [f"Greedy continuations from this setting: "
                  + "; ".join(f"`{p}` → `{t!r}`"[:200] for p, t in zip(PROMPTS, r["greedy_text"])), ""]
    if r.get("branches_note"):
        lines += [r["branches_note"], ""]
    return "\n".join(lines)


def model_card(results: list[dict], coverage: dict) -> str:
    rows = []
    for r in results:
        s = SETTINGS[r["name"]]
        rows.append(f"| `{r['name']}` | {s.repo} | {_e(r['max_abs_diff_fp32'])} | {_e(r.get('max_abs_diff_fp64'))} | "
                    f"{_greedy_line(r['greedy_vs_fp32'])} | {r['bytes'] / 1e9:.2f} GB |")
    total = sum(r["bytes"] for r in results)
    c = coverage
    fh = next((f"a {r['host_params'] / 1e9:.2f}B-parameter host for a {r['target_params'] / 1e6:.0f}M-parameter model"
               for r in results if r["name"] == "falcon-h1-tiny-90m"), "not built for this card")
    body = f"""---
license: mit
library_name: transformers
tags:
- emulation
- custom_code
---

# elephantine: one architecture, many models

This repo holds **one** transformers architecture, `ElephantineForCausalLM` (remote code:
`configuration_elephantine.py`, `modeling_elephantine.py`), and several **settings** of it. Each
subfolder holds a `config.json` and a `model.safetensors`. Every subfolder uses the same config class and the same model
class. Only the sizes, the weights and the branch coefficients differ. Each setting reproduces a
different public model through that model's own forward pass.

This is a small demo of the idea in [The Emulation Frontier]({BLOG}) (code:
[{GITHUB.split('github.com/')[1]}]({GITHUB})). It replaces an earlier August 2026 demo, a stock
JambaForCausalLM reproducing one tiny NoPE-Llama.

## What native emulation is

Model A *natively emulates* model B when some setting of A's weights makes A's forward pass compute
B's, token for token, with no fine-tuning and no approximation beyond floating-point rounding. The
"elephantine" is the architecture that emulates everything in a zoo of public decoder LMs. It takes the union of what
the models do, and each operation a family may or may not use is a **branch with a coefficient**,
off by default. Every layer is

    x += Attn(norm_a(x))     softmax attention: RoPE (settable frequencies), causal OR full mask, ...
    x += SSM(norm_s(x))      short conv -> general gated-delta recurrence -> norm / gate -> out
    x += FFN(norm_f(x))      routed MoE + a gated shared expert; SiLU / GeLU / ReLU / ... branches

and the SSM recurrence is the general gated-delta rule
`S_t = Diag(a_t) S_(t-1) - gamma_t (S_(t-1) k_t) k_t^T + w_t v_t k_t^T,  y_t = S_t q_t`, whose
knobs cover Mamba-1/2/3, gated DeltaNet, RWKV-4..7, RG-LRU, LFM2's gated conv, and more. A model is
loaded by zeroing the branches it doesn't use and copying its weights into the ones it does
(`elephantine.universal.from_*` in the code repo). `modeling_elephantine.py` is a verbatim copy of
that architecture (`elephantine/universal.py`, pure torch; no fla / mamba_ssm needed). It computes
in float64.

## Settings in this repo

| subfolder | original | max \\|Δ logits\\| vs original, fp32 | ... fp64 (casts lifted) | greedy continuations, 3 prompts | size |
|---|---|---|---|---|---|
{chr(10).join(rows)}

"fp32" compares against the original model run by its own transformers class in float32 (its normal
precision); the gap is the original's own float32 rounding. "fp64" runs the same original code in
float64 with its internal float32 casts lifted (`_lift_fp32` in the code repo's tests). The
remaining gap is float64 rounding, which is the precision of the construction itself. All numbers are
max |Δ| over every position and vocabulary entry of 3 prompts, measured after reloading each
subfolder through `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. The reload
is bit-identical to the host built in memory.

**Storage.** Each tensor is stored in the narrowest dtype that holds it exactly. A bf16
checkpoint's copied weights are stored as bf16, Pythia's as fp16, and folded or derived values as fp32 or fp64. Tensors still at
the architecture's default (the unused branches) are not stored, and the model class recreates
them at load. The model widens everything to float64. Total upload: {total / 1e9:.2f} GB.

**No KV cache.** `generate()` works, but every step re-runs the whole prefix (a demo, not a server).
Padding masks are not supported. Use unpadded, equal-length batches.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("{REPO_ID}", subfolder="smollm2-135m", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("{REPO_ID}", subfolder="smollm2-135m")
out = m.generate(tok("The capital of France is", return_tensors="pt").input_ids, max_new_tokens=16, do_sample=False)
print(tok.decode(out[0]))
```

Each subfolder's section below has its own snippet. `m.config.host` holds the host sizes;
`m.config.branches` summarises the branch coefficients the setting switches on (the weights file
is authoritative).

{chr(10).join(setting_card(SETTINGS[r['name']], r) for r in results)}

## Inflation: any size, the active subset

`smollm2-135m-inflated` is SmolLM2-135M inside a host that is wider (d 576 → 768, 9 → 12 heads,
FFN 1536 → 2048) and deeper (30 → 40 layers). The extra heads, FFN units and residual coordinates
are zeroed. The 10 extra layers are left at the architecture's default, with all-zero output projections, so each one adds
exactly 0. The wider residual stream needs one weight setting on each RMSNorm,
`eps' = eps·d/d'` and `w' = w·sqrt(d/d')`. The output matches the un-inflated host to float64 rounding (table
above). The same move inflates a target to any size. A bigger host runs a smaller model on the
active subset of its units, and the rest stay zero. `push_hub_demo.inflate` in the code repo does
this for heads, FFN width, depth, and (for RMSNorm targets) hidden width.

## How this relates to the ~{c['total']}T elephantine

The writeup's elephantine is this same architecture **sized to carry the whole zoo**. It takes the
per-axis maximum over all {c['models']:,} classified decoder LMs: hidden {c['h']:,}, {c['L']} layers,
kv/token {c['kv']:,}, {c['qheads']} query heads, ~{c['ffn'] / 1e6:.2f}M FFN units per layer, ~{c['total']}T
parameters in total and ~{c['active']}T active. **This demo is not that model.** Each setting here is
the same architecture at the size its one target needs (one target per subfolder), plus one
inflated example. The ~{c['total']}T model is a size, not a checkpoint. Nothing here was trained.

Coverage, from the code repo (`python -m graph.minimal_emulator`, `FRONTIER.md`): the one universal
host carries **{c['carried']:,} of {c['models']:,}** classified models. The {c['models'] - c['carried']:,} not carried are
in mixer families with no tested branch (Cyrah, whose code is in a gated repo, and unclassified recurrences). Of the carried
models, **{c['tested']:,}** have a whole-model `from_*` loader that accepts their model_type and config. For each of these a test runs
the loaded host end to end against that type's own transformers code, or a labelled transcription of its remote code. The other
{c['carried'] - c['tested']:,} are carried by priced slot constructions only.

Two things this demo found that the coverage counts don't show. A *trained* SSM is expensive in the
universal host: the SSM branch's knob projections are dense, so the trained
`state-spaces/mamba-130m-hf` needs a ~33B-parameter host (~264 GB in fp64) and `LiquidAI/LFM2-350M`
~13B. That is why the Mamba-1 setting here is a random-weight fixture of mamba-130m's code path,
and the trained SSM setting is Falcon-H1-Tiny-90M (Mamba-2: {fh}).
RWKV-4 is carried block by block, but universal.py has no whole-model RWKV loader, so it is not a
setting here.

## Rebuilding

From the code repo: `python -m elephantine.push_hub_demo --out DIR [--push]` rebuilds every
setting from the original checkpoints, measures it, and writes this card.
`tests/test_hub_demo.py` checks the vendored copy against `elephantine/universal.py` (source and
forward), rebuilds two settings, reloads them through `AutoModelForCausalLM` with
`trust_remote_code`, and checks them against the originals.

## License

MIT (the code and this card). The weights in each subfolder are re-encodings of the original
checkpoints and keep those models' terms: SmolLM2-135M, Pythia-160m and the dLLM Qwen3-0.6B
diffusion model are Apache-2.0; Falcon-H1-Tiny is under the Falcon-LLM license
(https://falconllm.tii.ae/falcon-terms-and-conditions.html); the mamba-130m-hf-mic fixture's card
declares no license.
"""
    return body


def coverage_numbers() -> dict:
    """The headline numbers, read from the code (graph.minimal_emulator) rather than typed in."""
    import contextlib
    import io
    buf = io.StringIO()
    from graph import minimal_emulator as me
    argv, sys.argv = sys.argv, [sys.argv[0]]          # me.main() parses sys.argv; give it none of ours
    try:
        with contextlib.redirect_stdout(buf):
            me.main() if hasattr(me, "main") else None
    finally:
        sys.argv = argv
    txt = buf.getvalue()
    if not txt:
        raise RuntimeError("graph.minimal_emulator printed nothing; run `python -m graph.minimal_emulator`")
    return parse_coverage(txt)


def parse_coverage(txt: str) -> dict:
    g = lambda pat: re.search(pat, txt).group(1)
    return {"carried": int(g(r"carried by the one universal host: (\d+) of")),
            "models": int(g(r"carried by the one universal host: \d+ of (\d+)")),
            "tested": int(g(r"loaded and tested[^:]*: (\d+) of")),
            "h": int(g(r"hidden\s+h =\s+(\d+)")), "L": int(g(r"layers\s+L =\s+(\d+)")),
            "kv": int(g(r"kv/token\s+kv =\s+(\d+)")), "qheads": int(g(r"query heads\s+=\s+(\d+)")),
            "ffn": int(g(r"ffn/layer\s+=\s+(\d+)")),
            "total": g(r"~([\d.]+)T params total"), "active": g(r"~([\d.]+)T active")}


# --- upload ---------------------------------------------------------------------------------------

def push(out: Path, repo_id: str = REPO_ID):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        info = api.model_info(repo_id)
        if not info.private:
            raise RuntimeError(f"{repo_id} exists and is PUBLIC; refusing to upload (visibility is not changed here)")
    except Exception as e:                                          # not found -> create it, private
        from huggingface_hub.utils import RepositoryNotFoundError
        if not isinstance(e, RepositoryNotFoundError):
            raise
        api.create_repo(repo_id, private=True, repo_type="model")
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model",
                      commit_message="Rebuild the universal elephantine demo (push_hub_demo.py)",
                      ignore_patterns=["*/results.json"])
    return f"https://huggingface.co/{repo_id}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", default=None, choices=list(SETTINGS))
    ap.add_argument("--vendor", action="store_true", help="refresh the vendored copy and exit")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="reuse the results already in --out")
    ap.add_argument("--no-fp64", action="store_true", help="skip the fp64 (casts lifted) reference")
    ap.add_argument("--coverage", type=Path, default=None,
                    help="a saved `python -m graph.minimal_emulator` output (default: run it)")
    ap.add_argument("--repo", default=REPO_ID)
    a = ap.parse_args(argv)
    changed = not vendor()
    print(f"vendored copy {'updated' if changed else 'up to date'}: {HUB / 'modeling_elephantine.py'}")
    if a.vendor:
        return
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    a.out.mkdir(parents=True, exist_ok=True)
    names = a.only or list(SETTINGS)
    if not a.skip_build:
        for n in names:
            build_setting(SETTINGS[n], a.out, fp64_ref=not a.no_fp64)
    results = [json.loads((a.out / n / "results.json").read_text()) for n in SETTINGS if (a.out / n / "results.json").exists()]
    cov = parse_coverage(a.coverage.read_text()) if a.coverage else coverage_numbers()
    for f in REMOTE_FILES:
        shutil.copy(HUB / f, a.out / f)
    (a.out / "README.md").write_text(model_card(results, cov))
    (a.out / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    print(f"model card: {a.out / 'README.md'}  ({len(results)} settings)")
    if a.push:
        print("uploaded:", push(a.out, a.repo))


if __name__ == "__main__":
    main()
