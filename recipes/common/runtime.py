"""The framework-free part of every recipe: Hub download, a numpy safetensors reader / writer, the
tokenizer, the CLI. Pure stdlib + numpy (Python 3.8+), so it runs in every framework's venv, DyNet's 3.8
included. make_recipe inlines this file into each generated script (the repository imports are dropped).
"""
import argparse
import glob
import json
import os
import struct
import sys
import urllib.error
import urllib.request

import numpy as np

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# --- the Hub ------------------------------------------------------------------------------------------


def _cached(repo, fname, revision="main"):
    """A file already in the local Hugging Face cache (no huggingface_hub needed), or None."""
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    base = os.path.join(root, "models--" + repo.replace("/", "--"))
    ref = os.path.join(base, "refs", revision)
    snaps = []
    if os.path.isfile(ref):
        snaps.append(os.path.join(base, "snapshots", open(ref).read().strip()))
    snaps += sorted(glob.glob(os.path.join(base, "snapshots", "*")))
    for s in snaps:
        p = os.path.join(s, fname)
        if os.path.isfile(p):
            return p
    return None


def hub_file(repo, fname, cache_dir, revision="main", optional=False):
    """Path of repo/fname: a local directory, the HF cache, or downloaded over HTTPS into cache_dir."""
    if os.path.isdir(repo):
        p = os.path.join(repo, fname)
        if os.path.isfile(p):
            return p
        if optional:
            return None
        raise FileNotFoundError(p)
    hit = _cached(repo, fname, revision)
    if hit:
        return hit
    dst = os.path.join(cache_dir, repo.replace("/", "--"), fname)
    if os.path.isfile(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = "%s/%s/resolve/%s/%s" % (HF_ENDPOINT, repo, revision, fname)
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(req) as r, open(dst + ".part", "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
    except urllib.error.HTTPError as e:
        if os.path.exists(dst + ".part"):
            os.remove(dst + ".part")
        if optional and e.code == 404:
            return None
        if e.code in (401, 403, 404):
            raise SystemExit("%s: %s not found, or the repo is gated or private (HTTP %d). For a gated repo set "
                             "HF_TOKEN and accept its licence on the Hub." % (repo, fname, e.code))
        raise SystemExit("%s: downloading %s failed (HTTP %d %s)" % (repo, fname, e.code, e.reason))
    except urllib.error.URLError as e:
        if os.path.exists(dst + ".part"):
            os.remove(dst + ".part")
        raise SystemExit("%s: cannot reach %s (%s)" % (repo, HF_ENDPOINT, e.reason))
    os.replace(dst + ".part", dst)
    return dst


def hub_model_files(repo, cache_dir, revision="main"):
    """(config.json path, [safetensors paths]) of a Hub repo or a local directory."""
    cfg = hub_file(repo, "config.json", cache_dir, revision)
    idx = hub_file(repo, "model.safetensors.index.json", cache_dir, revision, optional=True)
    if idx:
        shards = sorted(set(json.load(open(idx))["weight_map"].values()))
        return cfg, [hub_file(repo, s, cache_dir, revision) for s in shards]
    return cfg, [hub_file(repo, "model.safetensors", cache_dir, revision)]


# --- safetensors in numpy ---------------------------------------------------------------------------

_DT = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
       "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def read_safetensors(path, names=None, dtype=np.float32):
    """{name: np.ndarray} (floats cast to dtype; bf16 widened exactly). names: only these tensors."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        for k, v in header.items():
            if k == "__metadata__" or (names is not None and k not in names):
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            shape = tuple(v["shape"])
            if v["dtype"] == "BF16":
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw, dtype=_DT[v["dtype"]]).reshape(shape)
            if arr.dtype.kind == "f" and dtype is not None:
                arr = arr.astype(dtype)
            out[k] = np.array(arr)
    return out


def read_safetensors_all(paths, dtype=np.float32):
    out = {}
    for p in paths:
        out.update(read_safetensors(p, dtype=dtype))
    return out


def write_safetensors(path, tensors, metadata=None):
    """Write {name: np.ndarray} (F64 / F32 / F16 / ints) as a safetensors file."""
    rev = {np.dtype(v): k for k, v in _DT.items()}
    header, blobs, off = {}, [], 0
    for k in sorted(tensors):
        a = np.ascontiguousarray(tensors[k])
        b = a.tobytes()
        header[k] = {"dtype": rev[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def load_run_dir(run_dir, dtype=np.float32):
    """(config dict, {name: np.ndarray}) of a run directory or a downloaded HF checkpoint directory."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    idx = os.path.join(run_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        shards = sorted(set(json.load(open(idx))["weight_map"].values()))
        paths = [os.path.join(run_dir, s) for s in shards]
    else:
        paths = [os.path.join(run_dir, "model.safetensors")]
    return cfg, read_safetensors_all(paths, dtype)


# --- tokenizer ------------------------------------------------------------------------------------


def get_tokenizer(run_dir):
    """A `tokenizers` Tokenizer from run_dir/tokenizer.json, or None (then pass --ids)."""
    p = os.path.join(run_dir, "tokenizer.json")
    if not os.path.isfile(p):
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    return Tokenizer.from_file(p)


def encode(tok, text):
    return tok.encode(text).ids


def decode(tok, ids):
    return tok.decode(list(ids)) if tok is not None else " ".join(map(str, ids))


# --- generation (every framework's model has new_cache() and step(ids, cache, pos) -> (T, vocab) logits) ---


class Generator(object):
    """Greedy generation with a KV / recurrent-state cache: one prefill step over the prompt, then one
    step per new token. Subclasses implement new_cache() and step(ids, cache, pos) -> numpy (T, vocab)."""

    def prefill_logits(self, ids):
        return np.asarray(self.step(list(ids), self.new_cache(len(ids)), 0), np.float32)

    def generate(self, ids, n):
        cache = self.new_cache(len(ids) + n)
        logits = np.asarray(self.step(list(ids), cache, 0))[-1]
        pos, new, steps = len(ids), [], []
        for _ in range(n):
            steps.append(np.asarray(logits, np.float32))
            t = int(np.argmax(logits))
            new.append(t)
            if len(new) == n:
                break
            logits = np.asarray(self.step([t], cache, pos))[-1]
            pos += 1
        return new, steps


# --- the CLI shared by every script ----------------------------------------------------------------


def run_cli(load_model, framework, argv=None, build=None, defaults=None):
    """`build` (download / construct the run directory) and `run` (greedy generation in the framework).
    load_model(run_dir) -> model with .generate(ids, n) -> (new ids, [logits per step]) and
    .prefill_logits(ids) -> (T, vocab) numpy."""
    defaults = defaults or {}
    ap = argparse.ArgumentParser(description="Emulation-frontier recipe: %s via %s" %
                                 (defaults.get("target", "?"), framework))
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="download the target (and build its host) into DIR")
    b.add_argument("--out", default=defaults.get("out", "./recipe-run"))
    b.add_argument("--target", default=defaults.get("target"))
    b.add_argument("--host", default=defaults.get("host"))
    b.add_argument("--cache", default=os.path.expanduser("~/.cache/tef-recipes"))
    r = sub.add_parser("run", help="greedy generation from DIR in " + framework)
    r.add_argument("--dir", default=defaults.get("out", "./recipe-run"))
    r.add_argument("--prompt", default="The capital of France is")
    r.add_argument("--ids", default=None, help="comma-separated prompt token ids (instead of --prompt)")
    r.add_argument("--new-tokens", type=int, default=32)
    r.add_argument("--npz", "--json", dest="npz", default=None,
                   help="write ids / logits to this .npz file (validation; --json is the old name)")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        if build is None:
            raise SystemExit("this script has no build stage")
        build(args)
        return 0
    if args.cmd != "run":
        ap.print_help()
        return 1
    if not os.path.isfile(os.path.join(args.dir, "config.json")):
        raise SystemExit("%s is not a run directory (no config.json): run `build --out %s` first"
                         % (args.dir, args.dir))
    ids = None
    if args.ids:
        try:
            ids = [int(t) for t in args.ids.split(",")]
        except ValueError:
            ap.error("--ids takes comma-separated integers, e.g. --ids 1,2,3")
    model = load_model(args.dir)
    tok = get_tokenizer(args.dir)
    if ids is None:
        if tok is None:
            raise SystemExit("no tokenizer (pip install tokenizers, or pass --ids)")
        ids = encode(tok, args.prompt)
    pre = model.prefill_logits(ids)
    new, steps = model.generate(ids, args.new_tokens)
    print(decode(tok, ids) + "|" + decode(tok, new))
    if args.npz:
        np.savez(args.npz, ids=np.array(ids), new=np.array(new), prefill=np.asarray(pre, np.float32),
                 steps=np.stack([np.asarray(s, np.float32) for s in steps]) if steps else np.zeros((0,)))
    return 0
