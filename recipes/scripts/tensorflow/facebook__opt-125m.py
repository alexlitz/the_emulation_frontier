#!/usr/bin/env python3
"""Emulation-frontier recipe: run facebook/opt-125m in tensorflow.

Route: universal host via elephantine.universal.from_opt.
Written by `python -m recipes.make_recipe facebook/opt-125m --framework tensorflow` (https://github.com/alexlitz/the_emulation_frontier).

    python facebook__opt-125m-tensorflow.py build [--out DIR]      # stage 1: download + build the host weights
    python facebook__opt-125m-tensorflow.py run [--dir DIR] [--prompt TEXT | --ids 1,2,3] [--new-tokens 32]
"""
# requirements-build: torch transformers safetensors huggingface_hub tef-recipes  (pip install ./recipes/tef_recipes)
# requirements-run: tensorflow-cpu numpy  (optional: tokenizers)

import os
import sys

GITHUB = 'https://github.com/alexlitz/the_emulation_frontier'
DEFAULTS = {'target': 'facebook/opt-125m', 'framework': 'tensorflow', 'route': 'universal', 'host': 'universal', 'loader': 'from_opt', 'out': './facebook__opt-125m-tensorflow'}

# ---- recipes/common/runtime.py ----------------------------------------
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

def build(args):
    """Download the target (and build its host) into args.out."""
    route = DEFAULTS["route"]
    target = args.target or DEFAULTS["target"]
    out = args.out
    os.makedirs(out, exist_ok=True)
    if route == "native":
        cfg, shards = hub_model_files(target, args.cache)
        for p in [cfg] + shards + [hub_file(target, f, args.cache, optional=True)
                                   for f in ("model.safetensors.index.json", "tokenizer.json")]:
            if p:
                dst = os.path.join(out, os.path.basename(p))
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.realpath(p), dst)
    else:
        try:
            from tef_recipes import build as tb
        except ImportError:
            raise SystemExit("the build stage needs tef_recipes: pip install "
                             "'git+%s#subdirectory=recipes/tef_recipes' (with torch, transformers)" % GITHUB)
        if route == "exact-host":
            tb.build_exact(target, args.host or DEFAULTS["host"], out)
        else:
            tb.build_universal(target, DEFAULTS["loader"], out)
    print("built", out, "(%s route); now: python %s run --dir %s" % (route, sys.argv[0], out))

_RUN = __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'run'
if _RUN:  # the framework code is only imported for the run stage (the build stage may run elsewhere)
    # ---- recipes/common/ucore_forward.py ----------------------------------------
    """The universal host's forward over a small array backend, for the frameworks whose arrays are numpy-like
    (JAX, TensorFlow, tinygrad). It is recipes/frameworks/pytorch/universal.py (checked against the torch
    UniversalElephantine on every stock-loader case) with torch calls replaced by backend calls `B.*`.

    Every position-dependent input (RoPE angles, masks, cache writes) is computed in numpy and passed as data,
    and the KV cache is preallocated to the generation length, so every decode step has the same shapes.
    Inlined into the JAX / TensorFlow / tinygrad recipes."""
    import math

    import numpy as np



    class UCoreForward(Generator):
        """B: a backend (see recipes/frameworks/<fw>/universal.py) with linear (x @ w.T), mean_sq (the mean of x^2
        over the last axis, kept), asarray, numpy, zeros, cat, stack, reshape, transpose, sum, mean, max, exp, tanh,
        sigmoid, silu, softplus, gelu, mish, relu, sqrt, rsqrt, clip, softmax, einsum, take, topk, one_hot,
        take_along_last, where_pos, realize."""
        B = None

        def __init__(self, run_dir):
            cfg, sd = load_run_dir(run_dir)
            if cfg.get("format") != "tef-ucore":
                raise ValueError("not a tef-ucore run directory")
            self.c = cfg
            self.w = {k: self.B.asarray(np.ascontiguousarray(v, dtype=np.float32)) for k, v in sd.items()}
            self.npw = {k: v for k, v in sd.items() if k.endswith("inv_freq")}

        # --- pieces ---------------------------------------------------------------------------------------
        def _lin(self, name, x):
            y = self.B.linear(x, self.w[name + ".weight"])
            b = self.w.get(name + ".bias")
            return y if b is None else y + b

        def _rms(self, x, eps):
            B = self.B
            return x * B.rsqrt(B.mean_sq(x) + eps)

        def _norm(self, spec, name, x):
            B = self.B
            if spec["ident"]:
                return x
            if spec["centre"] != 0:
                x = x - spec["centre"] * B.mean(x, -1, True)
            y = self._rms(x, spec["eps"]) * self.w[name + ".weight"]
            b = self.w.get(name + ".bias")
            return y if b is None else y + b

        def _qk_norm(self, s, p, t, i):
            B = self.B
            coef = s["q_coef"] if i == 0 else s["k_coef"]
            if coef[1] == 0:
                return coef[0] * t
            dims = (-2, -1) if s["qk_full"] else (-1,)
            red = lambda v: B.sum(B.sum(v, -1, True), -2, True) if s["qk_full"] else B.sum(v, -1, True)
            n = t - red(t) / float(t.shape[-1] * (t.shape[-2] if s["qk_full"] else 1)) if s["qk_centre"][i] else t
            width = s["qk_width"] * (t.shape[-2] if s["qk_full"] else 1)
            nw = self.w[p + ("q_nw" if i == 0 else "k_nw")]
            n = n * B.rsqrt(red(n * n) / width + s["qk_eps"][i]) * nw
            if s["qk_centre"][i]:
                n = n + self.w[p + ("q_nb" if i == 0 else "k_nb")]
            return coef[0] * t + coef[1] * n

        @staticmethod
        def _rot_half(B, x):
            h = x.shape[-1] // 2
            return B.cat((-x[..., h:], x[..., :h]), -1)

        def _attn(self, s, p, x, cache, pos, tab):
            B, c, w = self.B, self.c, self.w
            T, H, hd = x.shape[0], c["heads"], c["head_dim"]
            n_max = tab["n"]
            if s["Dn"]:
                n = B.linear(x, w[p + "pn.weight"])
                n = n * B.rsqrt((n * n) @ w[p + "n_avg"] + s["n_eps"]) * w[p + "n_w"]
                x = B.cat((x, n), -1)
            q = B.reshape(self._lin(p + "q_proj", x), (T, H, hd))
            k = B.reshape(self._lin(p + "k_proj", x), (T, H, hd))
            v = B.reshape(self._lin(p + "v_proj", x), (T, H, hd))
            if s["v_norm"]:
                v = v * B.rsqrt(B.sum(v * v, -1, True) / s["qk_width"] + s["v_eps"]) * w[p + "v_nw"]
            if s["rope"]:
                inv = self.npw[p + "inv_freq"]
                ang = np.arange(pos, pos + T, dtype=np.float32)[:, None] * inv[None]
                cos = B.asarray(np.concatenate((np.cos(ang), np.cos(ang)), -1)[:, None].astype(np.float32))
                sin = B.asarray(np.concatenate((np.sin(ang), np.sin(ang)), -1)[:, None].astype(np.float32))
                rope = lambda t: t * cos + self._rot_half(B, t) * sin
            else:
                rope = lambda t: t
            if s["qk_post"]:
                q, k = rope(q), rope(k)
            q, k = self._qk_norm(s, p, q, 0), self._qk_norm(s, p, k, 1)
            if s["clip"] is not None:
                cl = s["clip"]
                q, k, v = B.clip(q, -cl, cl), B.clip(k, -cl, cl), B.clip(v, -cl, cl)
            if not s["qk_post"]:
                q, k = rope(q), rope(k)
            if s.get("qtemp"):
                beta, M = s["qtemp"]
                t_ = np.arange(pos, pos + T, dtype=np.float64)
                q = q * B.asarray((1 + beta * np.log1p(np.floor(t_ / M))).astype(np.float32)[:, None, None])
            reg = tab["reg"]
            if s.get("kv_read"):                                         # the KV register: an earlier layer's cache
                K, V = reg[s["kv_slot"]]
            else:
                if "k" not in cache:
                    cache["k"] = B.zeros((n_max, H, hd))
                    cache["v"] = B.zeros((n_max, H, hd))
                place = tab["place"]                                     # (n, T) one-hot: cache row <- token
                K = cache["k"] + B.einsum("st,thd->shd", place, k)
                V = cache["v"] + B.einsum("st,thd->shd", place, v)
                cache["k"], cache["v"] = K, V = B.realize(K), B.realize(V)
                if s.get("kv_write"):
                    reg[s["kv_slot"]] = (K, V)
            sc = B.einsum("thd,shd->hts", q, K) * s["scale"]
            if s["cap"] is not None:
                sc = s["cap"] * B.tanh(sc / s["cap"])
            tt = np.arange(pos, pos + T)[:, None]
            ss = np.arange(n_max)[None]
            if s["alibi"]:
                sc = sc + B.reshape(w[p + "alibi"], (H, 1, 1)) * B.asarray((ss - tt).astype(np.float32))
            win = math.inf if s["window"] is None else s["window"]
            ok = (((tt - ss) <= win) | (ss < s["n_sink"])) & (ss <= tt)
            sc = sc + B.asarray(np.where(ok, 0.0, -np.inf).astype(np.float32))
            if s["null"]:
                nl = B.reshape(w[p + "null_logit"], (H, 1, 1)) + B.zeros((H, T, 1))
                sc = B.cat((sc, nl), -1)
                pr = B.softmax(sc, -1)[..., :n_max]
            else:
                pr = B.softmax(sc, -1)
            o = B.reshape(B.einsum("hts,shd->thd", pr, V), (T, H * hd))
            if s["o_coef"][1] != 0:
                oc = s["o_coef"]
                o = oc[0] * o + oc[1] * o * B.rsqrt((o * o) @ w[p + "o_avg"] + s["o_eps"]) * w[p + "o_w"]
            g = s["gate_coef"]
            if s.get("gate_sp"):
                gx = self._lin(p + "g_proj", x)
                o = o * (g[0] + g[1] * B.sigmoid(gx) + s["gate_sp"] * B.softplus(gx))
            elif g[1] != 0:
                o = o * (g[0] + g[1] * B.sigmoid(self._lin(p + "g_proj", x)))
            elif g[0] != 1:
                o = o * g[0]
            return self._lin(p + "o_proj", o)

        def _ssm(self, s, p, x, cache, pos):
            B, w = self.B, self.w
            T = x.shape[0]
            H, dk, dv, K = s["H"], s["dk"], s["dv"], s["K"]
            u = self._lin(p + "in_proj", x)                                   # (T, D)
            if "conv" not in cache:
                cache["conv"] = B.zeros((K - 1, s["D"])) if K > 1 else None
                cache["S"] = B.zeros((H, dv, dk))
            up = B.cat((cache["conv"], u), 0) if K > 1 else u
            if K > 1:
                cache["conv"] = B.realize(up[up.shape[0] - (K - 1):])
            cw = w[p + "conv_w"]
            cv = w[p + "conv_b"]
            for j in range(K):
                cv = cv + cw[:, j] * up[j:j + T]
            cv = w[p + "act_silu"] * B.silu(cv) + w[p + "act_lin"] * cv
            r = B.cat((u, cv), -1)
            if s["Dn"]:
                n = self._lin(p + "pn", r)
                r = B.cat((r, n * B.rsqrt((n * n) @ w[p + "n_avg"] + s["n_eps"]) * w[p + "n_w"]), -1)
            if s["Dh"]:
                r = B.cat((r, B.sigmoid(self._lin(p + "ph", r))), -1)
            q = B.reshape(self._lin(p + "pq", r), (T, H, dk))
            k = B.reshape(self._lin(p + "pk", r), (T, H, dk))
            if s["l2_eps_out"] != 0:
                l2 = lambda t: t / (B.sqrt(B.sum(t * t, -1, True) + s["l2_eps"]) + s["l2_eps_out"])
            else:
                l2 = lambda t: t * B.rsqrt(B.sum(t * t, -1, True) + s["l2_eps"])
            q = s["q_coef"][0] * q + s["q_coef"][1] * l2(q)
            k = s["k_coef"][0] * k + s["k_coef"][1] * l2(k)
            v = B.reshape(self._lin(p + "pv", r), (T, H, dv))
            sp = B.softplus(B.reshape(self._lin(p + "p1", r), (T, H, dk)))
            if s["dt_min"] > 0:
                sp = B.clip(sp, s["dt_min"], None)
            loga = -(w[p + "A1"] * sp + w[p + "A2"] * B.sigmoid(B.reshape(self._lin(p + "p2", r), (T, H, dk))) + w[p + "A3"])
            dt = sp[..., 0]                                                   # (T, H)
            a = w[p + "mem"] * B.exp(loga)                                    # (T, H, dk)
            sb, si = B.sigmoid(self._lin(p + "pb", r)), B.sigmoid(self._lin(p + "pi", r))
            gamma = w[p + "cg"] * sb + w[p + "cg0"]
            a0 = a[..., 0]
            lru = B.sqrt(B.clip(1 - a0 * a0, 0, None))
            first = B.asarray((np.arange(pos, pos + T) == 0).astype(np.float32)[:, None])
            lru = lru + w[p + "reset"] * first * (1 - lru)
            c6 = w[p + "cw"]
            wr = (c6[0] * dt + c6[1] * sb + c6[2] * B.exp(self._lin(p + "pe", r)) + c6[3] * lru * si + c6[4]
                  + c6[5] * dt * sb)
            S = cache["S"]
            ys = []
            for t in range(T):
                kt = B.reshape(k[t], (H, 1, dk))
                S = S * B.reshape(a[t], (H, 1, dk))
                Sk = B.sum(S * kt, -1, True)                                  # (H, dv, 1)
                S = S - B.reshape(gamma[t], (H, 1, 1)) * Sk * kt
                S = S + B.reshape(wr[t], (H, 1, 1)) * B.reshape(v[t], (H, dv, 1)) * kt
                S = B.realize(S)
                ys.append(B.sum(S * B.reshape(q[t], (H, 1, dk)), -1))
            cache["S"] = S
            y = B.stack(ys, 0) + w[p + "Dskip"] * v                           # (T, H, dv)
            y = B.reshape(y, (T, H * dv)) @ w[p + "mix"]
            z = self._lin(p + "pz", r)
            g = w[p + "gate"]
            G = g[0] * B.silu(z) + g[1] * B.sigmoid(z) + g[2] * B.gelu(z, True) + g[3] * z + g[4]
            nrm = lambda t: t * B.rsqrt((t * t) @ w[p + "avg"] + s["norm_eps"]) * w[p + "norm_w"] + w[p + "norm_b"]
            oc = w[p + "out_coef"]
            out = oc[0] * nrm(y) * G + oc[1] * nrm(y * G) + oc[2] * y * G
            return self._lin(p + "out_proj", out)

        def _act(self, a, g, u):
            B = self.B
            if a["relu2"]:
                r = B.relu(u)
                return r * r
            if a["relu"]:
                return B.relu(u)
            if a["gelu"] or a["geglu"]:
                return B.gelu(u, a["gelu_tanh"]) if a["gelu"] else B.gelu(g, a["gelu_tanh"]) * u
            if a["mish"]:
                return B.mish(g) * u
            if a.get("xielu"):
                ap, an, beta, eps = a["xielu"]
                neg = (B.exp(B.clip(u, None, eps)) - 1 - u) * an + beta * u
                return B.where_pos(u, ap * u * u + beta * u, neg, 0.0)
            if a["c_clip"] != 0:
                lg, lu = a["clip"]
                if lg is not None:
                    g = g + a["c_clip"] * (B.clip(g, None, lg) - g)
                if lu is not None:
                    u = u + a["c_clip"] * (B.clip(u, -lu, lu) - u)
            u = u + a["up_shift"]
            return B.silu(g) * u

        def _ffn(self, s, p, x, xe=None, xg=None, parts=False):
            """routed(xe) (router on xg) + shared(x) * gate; parts: (routed, gated shared) (Gemma-4's split FFN)."""
            B, w, a = self.B, self.w, s["act"]
            xs, xe, xg = x, (x if xe is None else xe), (x if xg is None else xg)
            out = routed = shared = None
            r = s["routed"]
            if r is not None:
                E = r["E"]
                x = xe
                logits = self._lin(p + "router", xg)                         # (T, E)
                if r["router_cap"] is not None:
                    logits = r["router_cap"] * B.tanh(logits / r["router_cap"])
                sc_ = r["score_c"]
                sm = B.softmax(logits, -1)
                pr = sm if sc_ == 0 else (1 - sc_) * sm + sc_ * B.sigmoid(logits)
                sel = (logits if r["sel_logit"] else pr) + w[p + "sel_bias"]
                G, kg = int(r["grp"][0]), int(r["grp"][1])
                if G > 1 and kg < G:
                    T = x.shape[0]
                    sg = B.reshape(sel, (T, G, E // G))
                    if r["grp"][2] != 0:
                        gs = B.max(sg, -1)
                    else:
                        gs = B.sum(B.topk(sg, 2)[0], -1)
                    gi = B.topk(gs, kg)[1]                                    # (T, kg)
                    keep = B.sum(B.one_hot(gi, G), -2)                        # (T, G)
                    keep = B.reshape(B.reshape(keep, (T, G, 1)) + B.zeros((T, G, E // G)), (T, E))
                    sel = B.where_pos(keep, sel, B.zeros((T, E)) + float(r["grp"][3]))
                topv_sel, topi = B.topk(sel, r["top_k"])
                topv = B.take_along_last(pr, topi)
                topv = (topv + r["renorm"] * (topv / (B.sum(topv, -1, True) + r["renorm_eps"]) - topv)) * r["route_scale"]
                oh = B.one_hot(topi, E)                                       # (T, k, E)
                topv = topv * B.sum(oh * w[p + "exp_scale"], -1)
                wts = B.sum(oh * B.reshape(topv, tuple(topv.shape) + (1,)), -2)      # (T, E), zero off the top-k
                h = self._act(a, B.einsum("efd,td->tef", w[p + "w_gate"], x) + w[p + "b_gate"],
                              B.einsum("efd,td->tef", w[p + "w_up"], x) + w[p + "b_up"])
                eo = B.einsum("edf,tef->ted", w[p + "w_down"], h) + w[p + "b_down"]
                out = routed = B.einsum("ted,te->td", eo, wts)
            x = xs
            sh = s["shared"]
            if sh is not None:
                u = self._lin(p + "s_up", x)
                g = self._lin(p + "s_gate", x) if (p + "s_gate.weight") in w else u
                y = self._lin(p + "s_down", self._act(a, g, u))
                c = sh["s_coef"]
                if c[1] != 0:
                    y = y * (c[0] + c[1] * B.sigmoid(B.linear(x, w[p + "s_router.weight"])))
                elif c[0] != 1:
                    y = y * c[0]
                shared = y
                out = y if out is None else out + y
            return (routed, shared) if parts else out

        def _layer(self, i, L, x, cache, pos, tab):
            p = "layers.%d." % i

            def post(j, o, nm):
                pc = L["post_coef"][j]
                r = pc[0] * o if pc[0] != 1 else o
                return r + pc[1] * self._norm(L[nm], p + nm, o) if pc[1] != 0 else r
            if L["sn_in"] is not None:
                x = self._norm(L["sn_in"], p + "sn_in", x)
            a = s = None
            if L["attn"] is not None:
                a = post(0, self._attn(L["attn"], p + "attn.", self._norm(L["norm_a"], p + "norm_a", x), cache, pos, tab),
                         "post_a")
            if L["ssm"] is not None:
                xs = x if a is None else x + L["seq"] * a
                s = post(1, self._ssm(L["ssm"], p + "ssm.", self._norm(L["norm_s"], p + "norm_s", xs), cache, pos), "post_s")
            x_in = x
            x = x if L["res_c"][0] == 1 else L["res_c"][0] * x
            if a is not None:
                x = x + a
            if s is not None:
                x = x + s
            xf = x_in if L["par"] != 0 else x
            if L["res_c"][1] != 1:
                x = L["res_c"][1] * x
            B = self.B
            if L["ffn"] is not None and L.get("ffn_split"):
                nf = lambda nm: self._norm(L[nm], p + nm, xf)
                r_, s_ = self._ffn(L["ffn"], p + "ffn.", nf("norm_f"), xe=nf("norm_f2"), xg=nf("norm_r"), parts=True)
                y = None
                for part, nm in ((s_, "post_f1"), (r_, "post_f2")):
                    if part is not None:
                        yy = self._norm(L[nm], p + nm, part)
                        y = yy if y is None else y + yy
                x = x + post(2, y, "post_f")
            elif L["ffn"] is not None:
                x = x + post(2, self._ffn(L["ffn"], p + "ffn.", self._norm(L["norm_f"], p + "norm_f", xf)), "post_f")
            if L.get("ple"):                                                  # per-layer embedding input
                g = B.gelu(B.linear(x, self.w[p + "ple_gate.weight"]), L["ple"]["tanh"])
                y = B.linear(g * tab["pli"][:, i], self.w[p + "ple_proj.weight"])
                x = x + L["ple"]["c"] * self._norm(L["ple"]["post"], p + "ple_post", y)
            if L["sn_out"] is not None:
                x = self._norm(L["sn_out"], p + "sn_out", x)
            x = x * L["out_scale"] if L["out_scale"] != 1 else x
            return self.B.realize(x)

        # --- the model --------------------------------------------------------------------------------------
        def new_cache(self, max_len=None):
            return {"n": int(max_len or 1), "layers": [{} for _ in range(self.c["L"])]}

        def step(self, ids, cache, pos):
            B, c, w = self.B, self.c, self.w
            T, n = len(ids), cache["n"]
            if pos + T > n:
                raise ValueError("cache too short")
            place = (np.arange(n)[:, None] == np.arange(pos, pos + T)[None]).astype(np.float32)
            tab = {"n": n, "place": B.asarray(place), "reg": {}, "pli": None}
            x = B.take(w["embed.weight"], np.asarray(ids, np.int32))
            if c["max_pos"]:
                x = x + w["pos"][pos:pos + T]
            if c["emb_norm"] is not None:
                x = self._norm(c["emb_norm"], "emb_norm", x)
            if c.get("ple"):                                                  # pli = rms(P x0) w + E[ids], per layer
                dp = c["ple"]["dim"]
                pp = B.reshape(B.linear(x, w["ple_P.weight"]), (T, c["L"], dp))
                pe = B.reshape(B.take(w["ple_E"], np.asarray(ids, np.int32)), (T, c["L"], dp))
                tab["pli"] = self._rms(pp, c["ple"]["eps"]) * w["ple_w"] + pe
            for i, L in enumerate(c["layers"]):
                x = self._layer(i, L, x, cache["layers"][i], pos, tab)
            out = self._lin("lm_head", self._norm(c["norm"], "norm", x))
            if c["final_cap"] is not None:
                out = c["final_cap"] * B.tanh(out / c["final_cap"])
            return B.numpy(out)
    # ---- recipes/frameworks/tensorflow/universal.py ----------------------------------------
    """The universal host (tef-ucore export) in TensorFlow 2: recipes/common/ucore_forward.py over eager tf ops.
    fp32."""
    import os

    import numpy as np

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # noqa: E402



    def _clip(x, lo, hi):
        if lo is not None:
            x = tf.maximum(x, lo)
        if hi is not None:
            x = tf.minimum(x, hi)
        return x


    def _topk(x, k):
        r = tf.math.top_k(x, k)
        return r.values, r.indices


    class TFOps(object):
        linear = staticmethod(lambda x, w: tf.linalg.matmul(x, w, transpose_b=True))
        mean_sq = staticmethod(lambda x: tf.reduce_mean(x * x, -1, keepdims=True))
        asarray = staticmethod(lambda a: tf.constant(a))
        numpy = staticmethod(lambda t: t.numpy())
        zeros = staticmethod(lambda shape: tf.zeros(shape, tf.float32))
        cat = staticmethod(lambda xs, axis: tf.concat(list(xs), axis))
        stack = staticmethod(lambda xs, axis: tf.stack(list(xs), axis))
        reshape = staticmethod(lambda x, shape: tf.reshape(x, [int(d) for d in shape]))
        transpose = staticmethod(lambda x, perm: tf.transpose(x, perm))
        sum = staticmethod(lambda x, axis, keepdims=False: tf.reduce_sum(x, axis, keepdims=keepdims))
        mean = staticmethod(lambda x, axis, keepdims=False: tf.reduce_mean(x, axis, keepdims=keepdims))
        max = staticmethod(lambda x, axis: tf.reduce_max(x, axis))
        exp, tanh, sqrt, relu = staticmethod(tf.exp), staticmethod(tf.tanh), staticmethod(tf.sqrt), staticmethod(tf.nn.relu)
        sigmoid, silu, softplus = staticmethod(tf.sigmoid), staticmethod(tf.nn.silu), staticmethod(tf.nn.softplus)
        rsqrt = staticmethod(tf.math.rsqrt)
        gelu = staticmethod(lambda x, tanh: tf.nn.gelu(x, approximate=bool(tanh)))
        mish = staticmethod(lambda x: x * tf.tanh(tf.nn.softplus(x)))
        clip = staticmethod(_clip)
        softmax = staticmethod(lambda x, axis: tf.nn.softmax(x, axis))
        einsum = staticmethod(lambda spec, a, b: tf.einsum(spec, a, b))
        take = staticmethod(lambda table, ids: tf.gather(table, tf.constant(ids)))
        topk = staticmethod(_topk)
        one_hot = staticmethod(lambda idx, n: tf.one_hot(idx, n, dtype=tf.float32))
        take_along_last = staticmethod(lambda x, idx: tf.gather(x, idx, batch_dims=1))
        where_pos = staticmethod(lambda c, a, b, thr=0.5: tf.where(c > thr, a, b))
        realize = staticmethod(lambda x: x)


    class UCoreModel(UCoreForward):
        B = TFOps


    def load_model(run_dir):
        return UCoreModel(run_dir)


if __name__ == '__main__':
    sys.exit(run_cli(lambda d: load_model(d), 'tensorflow', build=build, defaults=DEFAULTS))
