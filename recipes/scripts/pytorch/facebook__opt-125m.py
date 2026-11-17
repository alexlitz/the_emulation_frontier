#!/usr/bin/env python3
"""Emulation-frontier recipe: run facebook/opt-125m in pytorch.

Route: universal host via elephantine.universal.from_opt.
Written by `python -m recipes.make_recipe facebook/opt-125m --framework pytorch` (https://github.com/alexlitz/the_emulation_frontier).

    python facebook__opt-125m-pytorch.py build [--out DIR]      # stage 1: download + build the host weights
    python facebook__opt-125m-pytorch.py run [--dir DIR] [--prompt TEXT | --ids 1,2,3] [--new-tokens 32]
"""
# requirements-build: torch transformers safetensors huggingface_hub tef-recipes  (pip install ./recipes/tef_recipes)
# requirements-run: torch numpy  (optional: tokenizers)

import os
import sys

GITHUB = 'https://github.com/alexlitz/the_emulation_frontier'
DEFAULTS = {'target': 'facebook/opt-125m', 'framework': 'pytorch', 'route': 'universal', 'host': 'universal', 'loader': 'from_opt', 'out': './facebook__opt-125m-pytorch'}

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
    # ---- recipes/frameworks/pytorch/universal.py ----------------------------------------
    """The universal host (elephantine/universal.py) in plain PyTorch, run from its tef-ucore export (no
    transformers, no repo code): every core branch of UniversalElephantine (tef_recipes.ucore_export.CORE_BRANCHES)
    for one sequence, with a KV cache for attention and a conv + recurrent-state cache for the SSM. fp32."""
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F


    F32 = torch.float32
    NEG = float("-inf")


    def _rms(x, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


    def _rot_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), -1)


    class UCoreModel(Generator):
        def __init__(self, run_dir):
            cfg, sd = load_run_dir(run_dir)
            if cfg.get("format") != "tef-ucore":
                raise ValueError("not a tef-ucore run directory")
            self.c = cfg
            self.w = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in sd.items()}
            torch.set_grad_enabled(False)

        # --- pieces ---------------------------------------------------------------------------------------
        def _lin(self, name, x):
            y = x @ self.w[name + ".weight"].T
            b = self.w.get(name + ".bias")
            return y if b is None else y + b

        def _norm(self, spec, name, x):
            if spec["ident"]:
                return x
            if spec["centre"] != 0:
                x = x - spec["centre"] * x.mean(-1, keepdim=True)
            y = _rms(x, spec["eps"]) * self.w[name + ".weight"]
            b = self.w.get(name + ".bias")
            return y if b is None else y + b

        def _qk_norm(self, s, p, t, i):
            coef = s["q_coef"] if i == 0 else s["k_coef"]
            if coef[1] == 0:
                return coef[0] * t
            dims = (-2, -1) if s["qk_full"] else (-1,)
            n = t - t.mean(dims, keepdim=True) if s["qk_centre"][i] else t
            width = s["qk_width"] * (t.shape[-2] if s["qk_full"] else 1)
            nw = self.w[p + ("q_nw" if i == 0 else "k_nw")]
            n = n * torch.rsqrt(n.pow(2).sum(dims, keepdim=True) / width + s["qk_eps"][i]) * nw
            if s["qk_centre"][i]:
                n = n + self.w[p + ("q_nb" if i == 0 else "k_nb")]
            return coef[0] * t + coef[1] * n

        def _attn(self, s, p, x, cache, pos, reg):
            c, w = self.c, self.w
            T, H, hd = x.shape[0], c["heads"], c["head_dim"]
            if s["Dn"]:
                n = x @ w[p + "pn.weight"].T
                n = n * torch.rsqrt(n.pow(2) @ w[p + "n_avg"] + s["n_eps"]) * w[p + "n_w"]
                x = torch.cat((x, n), -1)
            q = self._lin(p + "q_proj", x).view(T, H, hd)
            k = self._lin(p + "k_proj", x).view(T, H, hd)
            v = self._lin(p + "v_proj", x).view(T, H, hd)
            if s["v_norm"]:
                v = v * torch.rsqrt(v.pow(2).sum(-1, keepdim=True) / s["qk_width"] + s["v_eps"]) * w[p + "v_nw"]
            if s["rope"]:
                ang = torch.arange(pos, pos + T, dtype=F32)[:, None] * w[p + "inv_freq"][None]
                cos = torch.cat((ang.cos(), ang.cos()), -1)[:, None]
                sin = torch.cat((ang.sin(), ang.sin()), -1)[:, None]
                rope = lambda t: t * cos + _rot_half(t) * sin
            else:
                rope = lambda t: t
            if s["qk_post"]:
                q, k = rope(q), rope(k)
            q, k = self._qk_norm(s, p, q, 0), self._qk_norm(s, p, k, 1)
            if s["clip"] is not None:
                cl = s["clip"]
                q, k, v = q.clamp(-cl, cl), k.clamp(-cl, cl), v.clamp(-cl, cl)
            if not s["qk_post"]:
                q, k = rope(q), rope(k)
            if s.get("qtemp"):
                beta, M = s["qtemp"]
                t_ = torch.arange(pos, pos + T, dtype=F32)
                q = q * (1 + beta * torch.log1p(torch.floor(t_ / M)))[:, None, None]
            if s.get("kv_read"):                              # the KV register: an earlier layer's whole cache
                k, v = reg[s["kv_slot"]]
            else:
                if cache.get("k") is not None:
                    k = torch.cat((cache["k"], k), 0)
                    v = torch.cat((cache["v"], v), 0)
                cache["k"], cache["v"] = k, v
                if s.get("kv_write"):
                    reg[s["kv_slot"]] = (k, v)
            S = k.shape[0]
            sc = torch.einsum("thd,shd->hts", q, k) * s["scale"]
            if s["cap"] is not None:
                sc = s["cap"] * torch.tanh(sc / s["cap"])
            tt = torch.arange(pos, pos + T)[:, None]
            ss = torch.arange(S)[None]
            if s["alibi"]:
                sc = sc + w[p + "alibi"][:, None, None] * (ss - tt).to(F32)
            win = math.inf if s["window"] is None else s["window"]
            allowed = ((tt - ss) <= win) | (ss < s["n_sink"])
            allowed = allowed & (ss <= tt)
            sc = sc.masked_fill(~allowed, NEG)
            if s["null"]:
                sc = torch.cat((sc, w[p + "null_logit"][:, None, None].expand(H, T, 1)), -1)
            o = torch.einsum("hts,shd->thd", sc.softmax(-1)[..., :S], v).reshape(T, H * hd)
            if s["o_coef"][1] != 0:
                oc = s["o_coef"]
                o = oc[0] * o + oc[1] * o * torch.rsqrt(o.pow(2) @ w[p + "o_avg"] + s["o_eps"]) * w[p + "o_w"]
            g = s["gate_coef"]
            if s.get("gate_sp"):
                gx = self._lin(p + "g_proj", x)
                o = o * (g[0] + g[1] * torch.sigmoid(gx) + s["gate_sp"] * F.softplus(gx))
            elif g[1] != 0:
                o = o * (g[0] + g[1] * torch.sigmoid(self._lin(p + "g_proj", x)))
            elif g[0] != 1:
                o = o * g[0]
            return self._lin(p + "o_proj", o)

        def _ssm(self, s, p, x, cache, pos):
            w = self.w
            T = x.shape[0]
            H, dk, dv, K = s["H"], s["dk"], s["dv"], s["K"]
            u = self._lin(p + "in_proj", x)                                   # (T, D)
            hist = cache.get("conv")
            if hist is None:
                hist = torch.zeros(K - 1, s["D"], dtype=F32)
            up = torch.cat((hist, u), 0)
            cache["conv"] = up[up.shape[0] - (K - 1):] if K > 1 else hist
            cw = w[p + "conv_w"]
            cv = sum(cw[:, j] * up[j:j + T] for j in range(K)) + w[p + "conv_b"]
            cv = w[p + "act_silu"] * F.silu(cv) + w[p + "act_lin"] * cv
            r = torch.cat((u, cv), -1)
            if s["Dn"]:
                n = self._lin(p + "pn", r)
                r = torch.cat((r, n * torch.rsqrt(n.pow(2) @ w[p + "n_avg"] + s["n_eps"]) * w[p + "n_w"]), -1)
            if s["Dh"]:
                r = torch.cat((r, torch.sigmoid(self._lin(p + "ph", r))), -1)
            q = self._lin(p + "pq", r).view(T, H, dk)
            k = self._lin(p + "pk", r).view(T, H, dk)
            if s["l2_eps_out"] != 0:
                l2 = lambda t: t / ((t.pow(2).sum(-1, keepdim=True) + s["l2_eps"]).sqrt() + s["l2_eps_out"])
            else:
                l2 = lambda t: t * torch.rsqrt(t.pow(2).sum(-1, keepdim=True) + s["l2_eps"])
            q = s["q_coef"][0] * q + s["q_coef"][1] * l2(q)
            k = s["k_coef"][0] * k + s["k_coef"][1] * l2(k)
            v = self._lin(p + "pv", r).view(T, H, dv)
            sp = F.softplus(self._lin(p + "p1", r).view(T, H, dk))
            if s["dt_min"] > 0:
                sp = sp.clamp(min=s["dt_min"])
            loga = -(w[p + "A1"] * sp + w[p + "A2"] * torch.sigmoid(self._lin(p + "p2", r).view(T, H, dk)) + w[p + "A3"])
            dt = sp[..., 0]                                                   # (T, H)
            a = w[p + "mem"] * loga.exp()                                     # (T, H, dk)
            sb, si = torch.sigmoid(self._lin(p + "pb", r)), torch.sigmoid(self._lin(p + "pi", r))
            gamma = w[p + "cg"] * sb + w[p + "cg0"]
            lru = torch.sqrt((1 - a[..., 0] ** 2).clamp_min(0))
            first = (torch.arange(pos, pos + T) == 0).to(F32)[:, None]
            lru = lru + w[p + "reset"] * first * (1 - lru)
            c6 = w[p + "cw"]
            wr = (c6[0] * dt + c6[1] * sb + c6[2] * self._lin(p + "pe", r).exp() + c6[3] * lru * si + c6[4]
                  + c6[5] * dt * sb)
            S = cache.get("S")
            if S is None:
                S = torch.zeros(H, dv, dk, dtype=F32)
            ys = []
            for t in range(T):
                S = S * a[t][:, None, :]
                Sk = (S * k[t][:, None, :]).sum(-1)
                S = S - gamma[t][:, None, None] * Sk[..., None] * k[t][:, None, :]
                S = S + wr[t][:, None, None] * v[t][:, :, None] * k[t][:, None, :]
                ys.append((S * q[t][:, None, :]).sum(-1))
            cache["S"] = S
            y = torch.stack(ys, 0) + w[p + "Dskip"] * v                       # (T, H, dv)
            y = y.reshape(T, H * dv) @ w[p + "mix"]
            z = self._lin(p + "pz", r)
            g = w[p + "gate"]
            G = g[0] * F.silu(z) + g[1] * torch.sigmoid(z) + g[2] * F.gelu(z, approximate="tanh") + g[3] * z + g[4]
            nrm = lambda t: t * torch.rsqrt(t.pow(2) @ w[p + "avg"] + s["norm_eps"]) * w[p + "norm_w"] + w[p + "norm_b"]
            oc = w[p + "out_coef"]
            out = oc[0] * nrm(y) * G + oc[1] * nrm(y * G) + oc[2] * y * G
            return self._lin(p + "out_proj", out)

        @staticmethod
        def _act(a, g, u):
            if a["relu2"]:
                return F.relu(u) ** 2
            if a["relu"]:
                return F.relu(u)
            if a["gelu"] or a["geglu"]:
                ge = lambda t: F.gelu(t, approximate="tanh" if a["gelu_tanh"] else "none")
                return ge(u) if a["gelu"] else ge(g) * u
            if a["mish"]:
                return F.mish(g) * u
            if a.get("xielu"):
                ap, an, beta, eps = a["xielu"]
                return torch.where(u > 0, ap * u * u + beta * u, (torch.expm1(u.clamp(max=eps)) - u) * an + beta * u)
            if a["c_clip"] != 0:
                lg, lu = a["clip"]
                if lg is not None:
                    g = g + a["c_clip"] * (g.clamp(max=lg) - g)
                if lu is not None:
                    u = u + a["c_clip"] * (u.clamp(-lu, lu) - u)
            u = u + a["up_shift"]
            return F.silu(g) * u

        def _ffn(self, s, p, x, xe=None, xg=None, parts=False):
            """routed(xe) (router on xg) + shared(x) * gate; parts: (routed, gated shared) (Gemma-4's split FFN)."""
            w, a = self.w, s["act"]
            xs, xe, xg = x, (x if xe is None else xe), (x if xg is None else xg)
            out = routed = 0
            r = s["routed"]
            if r is not None:
                x = xe
                logits = self._lin(p + "router", xg)                         # (T, E)
                if r["router_cap"] is not None:
                    logits = r["router_cap"] * torch.tanh(logits / r["router_cap"])
                sc_ = r["score_c"]
                pr = logits.softmax(-1) if sc_ == 0 else (1 - sc_) * logits.softmax(-1) + sc_ * torch.sigmoid(logits)
                sel = (logits if r["sel_logit"] else pr) + w[p + "sel_bias"]
                G, kg = int(r["grp"][0]), int(r["grp"][1])
                if G > 1 and kg < G:
                    sg = sel.view(sel.shape[0], G, -1)
                    gs = sg.max(-1).values if r["grp"][2] != 0 else sg.topk(2, -1).values.sum(-1)
                    keep = torch.zeros_like(gs).scatter_(-1, gs.topk(kg, -1).indices, 1.0)
                    sel = sel.masked_fill(keep.repeat_interleave(sel.shape[-1] // G, -1) == 0, float(r["grp"][3]))
                topi = sel.topk(r["top_k"], -1).indices
                topv = pr.gather(-1, topi)
                topv = (topv + r["renorm"] * (topv / (topv.sum(-1, keepdim=True) + r["renorm_eps"]) - topv)) * r["route_scale"]
                topv = topv * w[p + "exp_scale"][topi]
                wts = torch.zeros_like(pr).scatter_(-1, topi, topv)          # (T, E), zero off the top-k
                h = self._act(a, torch.einsum("efd,td->tef", w[p + "w_gate"], x) + w[p + "b_gate"],
                              torch.einsum("efd,td->tef", w[p + "w_up"], x) + w[p + "b_up"])
                eo = torch.einsum("edf,tef->ted", w[p + "w_down"], h) + w[p + "b_down"]
                out = routed = (eo * wts[..., None]).sum(1)
            x = xs
            sh = s["shared"]
            shared = 0
            if sh is not None:
                u = self._lin(p + "s_up", x)
                g = self._lin(p + "s_gate", x) if (p + "s_gate.weight") in w else u
                y = self._lin(p + "s_down", self._act(a, g, u))
                c = sh["s_coef"]
                sgt = c[0] + (c[1] * torch.sigmoid(x @ w[p + "s_router.weight"].T) if c[1] != 0 else 0)
                shared = y * sgt
                out = out + shared
            return (routed, shared) if parts else out

        def _layer(self, i, L, x, cache, pos, reg, pli):
            p = "layers.%d." % i

            def post(j, o, nm):
                pc = L["post_coef"][j]
                return pc[0] * o + (pc[1] * self._norm(L[nm], p + nm, o) if pc[1] != 0 else 0)
            if L["sn_in"] is not None:
                x = self._norm(L["sn_in"], p + "sn_in", x)
            a = s = 0
            if L["attn"] is not None:
                a = post(0, self._attn(L["attn"], p + "attn.", self._norm(L["norm_a"], p + "norm_a", x), cache, pos, reg),
                         "post_a")
            if L["ssm"] is not None:
                s = post(1, self._ssm(L["ssm"], p + "ssm.", self._norm(L["norm_s"], p + "norm_s", x + L["seq"] * a),
                                      cache, pos), "post_s")
            x_in = x
            x = (x if L["res_c"][0] == 1 else L["res_c"][0] * x) + a + s
            xf = x_in if L["par"] != 0 else x
            if L["res_c"][1] != 1:
                x = L["res_c"][1] * x
            if L["ffn"] is not None and L.get("ffn_split"):
                nf = lambda nm: self._norm(L[nm], p + nm, xf)
                r_, s_ = self._ffn(L["ffn"], p + "ffn.", nf("norm_f"), xe=nf("norm_f2"), xg=nf("norm_r"), parts=True)
                x = x + post(2, self._norm(L["post_f1"], p + "post_f1", s_) + self._norm(L["post_f2"], p + "post_f2", r_),
                             "post_f")
            elif L["ffn"] is not None:
                x = x + post(2, self._ffn(L["ffn"], p + "ffn.", self._norm(L["norm_f"], p + "norm_f", xf)), "post_f")
            if L.get("ple"):                                  # per-layer embedding input
                g = F.gelu(x @ self.w[p + "ple_gate.weight"].T, approximate="tanh" if L["ple"]["tanh"] else "none")
                y = (g * pli[:, i]) @ self.w[p + "ple_proj.weight"].T
                x = x + L["ple"]["c"] * self._norm(L["ple"]["post"], p + "ple_post", y)
            if L["sn_out"] is not None:
                x = self._norm(L["sn_out"], p + "sn_out", x)
            return x * L["out_scale"] if L["out_scale"] != 1 else x

        # --- the model --------------------------------------------------------------------------------------
        def new_cache(self, max_len=None):
            return [{} for _ in range(self.c["L"])]

        def step(self, ids, cache, pos):
            c, w = self.c, self.w
            x = w["embed.weight"][torch.tensor(ids)]
            if c["max_pos"]:
                x = x + w["pos"][pos:pos + len(ids)]
            if c["emb_norm"] is not None:
                x = self._norm(c["emb_norm"], "emb_norm", x)
            pli = None
            if c.get("ple"):                                  # pli = rms(P x0) w + E[ids], per layer
                dp, T = c["ple"]["dim"], x.shape[0]
                pp = (x @ w["ple_P.weight"].T).view(T, c["L"], dp)
                pli = _rms(pp, c["ple"]["eps"]) * w["ple_w"] + w["ple_E"][torch.tensor(ids)].view(T, c["L"], dp)
            reg = {}
            for i, L in enumerate(c["layers"]):
                x = self._layer(i, L, x, cache[i], pos, reg, pli)
            out = self._lin("lm_head", self._norm(c["norm"], "norm", x))
            if c["final_cap"] is not None:
                out = c["final_cap"] * torch.tanh(out / c["final_cap"])
            return out.numpy()


    def load_model(run_dir):
        return UCoreModel(run_dir)


if __name__ == '__main__':
    sys.exit(run_cli(lambda d: load_model(d), 'pytorch', build=build, defaults=DEFAULTS))
