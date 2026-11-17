#!/usr/bin/env python3
"""Emulation-frontier recipe: run openai-community/gpt2 in dynet.

Route: universal host via elephantine.universal.from_gpt2.
Written by `python -m recipes.make_recipe openai-community/gpt2 --framework dynet` (https://github.com/alexlitz/the_emulation_frontier).

    python openai-community__gpt2-dynet.py build [--out DIR]      # stage 1: download + build the host weights
    python openai-community__gpt2-dynet.py run [--dir DIR] [--prompt TEXT | --ids 1,2,3] [--new-tokens 32]
"""
# requirements-build: torch transformers safetensors huggingface_hub tef-recipes  (pip install ./recipes/tef_recipes)
# requirements-run: dyNET==2.1.2 numpy<1.25  on CPython 3.8  (optional: tokenizers<0.21)

import os
import sys

GITHUB = 'https://github.com/alexlitz/the_emulation_frontier'
DEFAULTS = {'target': 'openai-community/gpt2', 'framework': 'dynet', 'route': 'universal', 'host': 'universal', 'loader': 'from_gpt2', 'out': './openai-community__gpt2-dynet'}

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
    # ---- recipes/frameworks/dynet/universal.py ----------------------------------------
    """The universal host (tef-ucore export) in DyNet 2.1 (Python 3.8): every core branch of UniversalElephantine,
    one token at a time (DyNet builds a graph per token; a prompt is fed token by token, which is exactly causal).
    fp32. The weights are DyNet parameters; the caches (attention keys / values, the SSM conv history and
    recurrent state) hold computed values as numpy arrays and are fed back as inputs. The routed-expert top-k
    and the attention mask choose WHICH keys / experts enter the graph (numpy decides, DyNet computes)."""
    import math
    import os

    import numpy as np

    import dynet_config

    dynet_config.set(mem=int(os.environ.get("DYNET_MEM", "4096")), random_seed=1)
    import dynet as dy  # noqa: E402



    def _const(a):
        return dy.inputTensor(np.asarray(a, np.float32))


    def _rsqrt(v):
        return dy.cdiv(_const(np.ones(v.dim()[0][0], np.float32)), dy.sqrt(v)) if v.dim()[0][0] > 1 else \
            dy.pow(v, dy.scalarInput(-0.5))


    def _softplus(x):
        return dy.rectify(x) + dy.log(dy.exp(-dy.abs(x)) + 1.0)


    def _gelu(x, tanh):
        if tanh:
            return 0.5 * dy.cmult(x, 1.0 + dy.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * dy.cube(x))))
        return 0.5 * dy.cmult(x, 1.0 + dy.erf(x * (1.0 / math.sqrt(2))))


    def _silu(x):
        return dy.cmult(x, dy.logistic(x))


    def _clip(x, lo, hi):
        n = x.dim()[0][0]
        if lo is not None:
            x = dy.bmax(x, _const(np.full(n, lo, np.float32)))
        if hi is not None:
            x = dy.bmin(x, _const(np.full(n, hi, np.float32)))
        return x


    def _sum_sq(x):
        return dy.sum_elems(dy.square(x))


    def _split(d):
        """A divisor of d near sqrt(d). DyNet's sum_elems accumulates in one fp32 running sum; over a wide
        massive activation (position 0 of Qwen2.5 / SmolLM2) that rounding grows with d, so the RMS mean
        reduces in two stages (Qwen2.5-0.5B: 1.9e-4 -> 5.7e-5 from fp64)."""
        c = max(int(d ** 0.5), 1)
        while d % c:
            c -= 1
        return c


    def _sum_sq_split(x, n):
        if n <= 64:
            return dy.sum_elems(dy.square(x))
        c = _split(n)
        k = n // c
        return dy.esum([dy.sum_elems(dy.square(dy.pick_range(x, i * k, (i + 1) * k))) for i in range(c)])


    class UCoreModel(Generator):
        def __init__(self, run_dir):
            cfg, sd = load_run_dir(run_dir)
            if cfg.get("format") != "tef-ucore":
                raise ValueError("not a tef-ucore run directory")
            self.c = cfg
            self.np = sd
            self.pc = pc = dy.ParameterCollection()
            self.P = {}
            for k, v in sd.items():
                if k == "embed.weight":
                    self.E = pc.lookup_parameters_from_numpy(np.asarray(v, np.float32))
                elif k == "pos":
                    self.POS = pc.lookup_parameters_from_numpy(np.asarray(v, np.float32))
                elif k == "ple_E":
                    self.PLE = pc.lookup_parameters_from_numpy(np.asarray(v, np.float32))
                elif v.ndim == 3:                                    # routed experts: one matrix per expert
                    for e in range(v.shape[0]):
                        self.P["%s.%d" % (k, e)] = pc.parameters_from_numpy(np.asarray(v[e], np.float32))
                elif v.ndim == 2 and k.endswith(("b_gate", "b_up", "b_down")):
                    for e in range(v.shape[0]):
                        self.P["%s.%d" % (k, e)] = pc.parameters_from_numpy(np.asarray(v[e], np.float32))
                else:
                    self.P[k] = pc.parameters_from_numpy(np.asarray(v, np.float32))

        def w(self, k):
            return dy.parameter(self.P[k])

        def has(self, k):
            return k in self.P

        # --- pieces ---------------------------------------------------------------------------------------
        def _lin(self, name, x):
            y = self.w(name + ".weight") * x
            return y + self.w(name + ".bias") if self.has(name + ".bias") else y

        def _rms(self, x, eps):
            n = x.dim()[0][0]
            return x * dy.pow(_sum_sq_split(x, n) * (1.0 / n) + eps, dy.scalarInput(-0.5))

        def _norm(self, spec, name, x):
            if spec["ident"]:
                return x
            n = x.dim()[0][0]
            if spec["centre"] != 0:
                x = x - _const(np.ones(n, np.float32)) * (dy.sum_elems(x) * (spec["centre"] / n))
            y = dy.cmult(self._rms(x, spec["eps"]), self.w(name + ".weight"))
            return y + self.w(name + ".bias") if self.has(name + ".bias") else y

        def _grouped(self, x, avg, eps):
            """x * rsqrt(x^2 @ avg + eps) (an RMS norm whose grouping is the matrix avg)."""
            return dy.cmult(x, _rsqrt(dy.transpose(self.w(avg)) * dy.square(x) + eps))

        def _qk_norm(self, s, p, heads, i):
            """heads: list of H (hd,) expressions -> the [plain, normed] mix."""
            coef = s["q_coef"] if i == 0 else s["k_coef"]
            if coef[1] == 0:
                return [h * coef[0] if coef[0] != 1 else h for h in heads]
            H, hd = len(heads), heads[0].dim()[0][0]
            nw, nb = self.np[p + ("q_nw" if i == 0 else "k_nw")], self.np[p + ("q_nb" if i == 0 else "k_nb")]
            cen = s["qk_centre"][i]
            if s["qk_full"]:
                t = dy.concatenate(heads)
                m = t - _const(np.ones(H * hd, np.float32)) * (dy.sum_elems(t) * (1.0 / (H * hd))) if cen else t
                r = dy.pow(_sum_sq(m) * (1.0 / (s["qk_width"] * H)) + s["qk_eps"][i], dy.scalarInput(-0.5))
                n = dy.cmult(m * r, _const(nw.reshape(-1)))
                if cen:
                    n = n + _const(nb.reshape(-1))
                out = coef[0] * t + coef[1] * n
                return [dy.pick_range(out, h * hd, (h + 1) * hd) for h in range(H)]
            res = []
            for h, t in enumerate(heads):
                m = t - _const(np.ones(hd, np.float32)) * (dy.sum_elems(t) * (1.0 / hd)) if cen else t
                r = dy.pow(_sum_sq(m) * (1.0 / s["qk_width"]) + s["qk_eps"][i], dy.scalarInput(-0.5))
                n = dy.cmult(m * r, _const(nw[h]))
                if cen:
                    n = n + _const(nb[h])
                res.append(coef[0] * t + coef[1] * n)
            return res

        def _attn(self, s, p, x, cache, pos, reg):
            c = self.c
            H, hd = c["heads"], c["head_dim"]
            if s["Dn"]:
                n = self.w(p + "pn.weight") * x
                n = dy.cmult(self._grouped(n, p + "n_avg", s["n_eps"]), self.w(p + "n_w"))
                x = dy.concatenate([x, n])
            q, k, v = self._lin(p + "q_proj", x), self._lin(p + "k_proj", x), self._lin(p + "v_proj", x)
            sl = lambda t, h: dy.pick_range(t, h * hd, (h + 1) * hd)
            qs, ks, vs = [sl(q, h) for h in range(H)], [sl(k, h) for h in range(H)], [sl(v, h) for h in range(H)]
            if s["v_norm"]:
                vnw = self.np[p + "v_nw"]
                vs = [dy.cmult(t * dy.pow(_sum_sq(t) * (1.0 / s["qk_width"]) + s["v_eps"], dy.scalarInput(-0.5)),
                               _const(vnw[h])) for h, t in enumerate(vs)]
            if s["rope"]:
                ang = np.float32(pos) * self.np[p + "inv_freq"]
                cos, sin = _const(np.concatenate((np.cos(ang), np.cos(ang)))), _const(np.concatenate((np.sin(ang), np.sin(ang))))
                half = hd // 2

                def rope(t):
                    rh = dy.concatenate([-dy.pick_range(t, half, hd), dy.pick_range(t, 0, half)])
                    return dy.cmult(t, cos) + dy.cmult(rh, sin)
            else:
                rope = lambda t: t
            if s["qk_post"]:
                qs, ks = [rope(t) for t in qs], [rope(t) for t in ks]
            qs, ks = self._qk_norm(s, p, qs, 0), self._qk_norm(s, p, ks, 1)
            if s["clip"] is not None:
                cl = s["clip"]
                qs, ks, vs = ([_clip(t, -cl, cl) for t in ts] for ts in (qs, ks, vs))
            if not s["qk_post"]:
                qs, ks = [rope(t) for t in qs], [rope(t) for t in ks]
            if s.get("qtemp"):
                beta, M = s["qtemp"]
                qs = [t * float(1 + beta * np.log1p(np.floor(pos / M))) for t in qs]
            kn = np.stack([t.npvalue().reshape(hd) for t in ks])         # (H, hd)
            vn = np.stack([t.npvalue().reshape(hd) for t in vs])
            if s.get("kv_read"):                             # the KV register: an earlier layer's cache
                Kc, Vc = reg[s["kv_slot"]]
            else:
                Kc, Vc = cache.setdefault("k", []), cache.setdefault("v", [])
                Kc.append(kn)
                Vc.append(vn)
                if s.get("kv_write"):
                    reg[s["kv_slot"]] = (Kc, Vc)
            S = len(Kc)
            ss = np.arange(S)
            win = math.inf if s["window"] is None else s["window"]
            keep = np.nonzero(((pos - ss) <= win) | (ss < s["n_sink"]))[0]      # (ss <= pos always: the cache)
            Kall, Vall = np.stack(Kc, 2)[:, :, keep], np.stack(Vc, 2)[:, :, keep]   # (H, hd, S')
            rel = (keep - pos).astype(np.float32)
            heads = []
            for h in range(H):
                sc = dy.transpose(_const(Kall[h])) * qs[h] * s["scale"]
                if s["cap"] is not None:
                    sc = s["cap"] * dy.tanh(sc * (1.0 / s["cap"]))
                if s["alibi"]:
                    sc = sc + _const(rel) * dy.pick(self.w(p + "alibi"), h)
                if s["null"]:
                    pr = dy.softmax(dy.concatenate([sc, dy.pick(self.w(p + "null_logit"), h)]))
                    pr = dy.pick_range(pr, 0, len(keep))
                else:
                    pr = dy.softmax(sc)
                heads.append(_const(Vall[h]) * pr)
            o = dy.concatenate(heads)
            if s["o_coef"][1] != 0:
                oc = s["o_coef"]
                o = oc[0] * o + oc[1] * dy.cmult(self._grouped(o, p + "o_avg", s["o_eps"]), self.w(p + "o_w"))
            g = s["gate_coef"]
            if s.get("gate_sp"):
                gx = self._lin(p + "g_proj", x)
                o = dy.cmult(o, g[0] + g[1] * dy.logistic(gx) + s["gate_sp"] * _softplus(gx))
            elif g[1] != 0:
                o = dy.cmult(o, g[0] + g[1] * dy.logistic(self._lin(p + "g_proj", x)))
            elif g[0] != 1:
                o = o * g[0]
            return self._lin(p + "o_proj", o)

        def _ssm(self, s, p, x, cache, pos):
            H, dk, dv, K, D = s["H"], s["dk"], s["dv"], s["K"], s["D"]
            u = self._lin(p + "in_proj", x)                                  # (D,)
            hist = cache.get("conv")
            if hist is None:
                hist = np.zeros((K - 1, D), np.float32)
            cw = self.np[p + "conv_w"]
            cv = self.w(p + "conv_b") + dy.cmult(_const(cw[:, K - 1]), u)
            for j in range(K - 1):
                cv = cv + _const(cw[:, j] * hist[j])
            if K > 1:
                cache["conv"] = np.concatenate((hist, u.npvalue().reshape(1, D)), 0)[1:]
            cv = dy.cmult(self.w(p + "act_silu"), _silu(cv)) + dy.cmult(self.w(p + "act_lin"), cv)
            r = dy.concatenate([u, cv])
            if s["Dn"]:
                n = self._lin(p + "pn", r)
                r = dy.concatenate([r, dy.cmult(self._grouped(n, p + "n_avg", s["n_eps"]), self.w(p + "n_w"))])
            if s["Dh"]:
                r = dy.concatenate([r, dy.logistic(self._lin(p + "ph", r))])
            q, k, v = self._lin(p + "pq", r), self._lin(p + "pk", r), self._lin(p + "pv", r)
            hs = lambda t, n, h: dy.pick_range(t, h * n, (h + 1) * n)

            def l2(t):
                if s["l2_eps_out"] != 0:
                    return t * dy.pow(dy.sqrt(_sum_sq(t) + s["l2_eps"]) + s["l2_eps_out"], dy.scalarInput(-1.0))
                return t * dy.pow(_sum_sq(t) + s["l2_eps"], dy.scalarInput(-0.5))
            qc, kc = s["q_coef"], s["k_coef"]
            qh = [qc[0] * hs(q, dk, h) + qc[1] * l2(hs(q, dk, h)) for h in range(H)]
            kh = [kc[0] * hs(k, dk, h) + kc[1] * l2(hs(k, dk, h)) for h in range(H)]
            vh = [hs(v, dv, h) for h in range(H)]
            sp = _softplus(self._lin(p + "p1", r))                            # (H dk,)
            if s["dt_min"] > 0:
                sp = _clip(sp, s["dt_min"], None)
            A = lambda n: _const(self.np[p + n].reshape(-1))
            loga = -(dy.cmult(A("A1"), sp) + dy.cmult(A("A2"), dy.logistic(self._lin(p + "p2", r))) + A("A3"))
            a = dy.cmult(A("mem"), dy.exp(loga))
            sb, si = dy.logistic(self._lin(p + "pb", r)), dy.logistic(self._lin(p + "pi", r))
            gamma = dy.cmult(self.w(p + "cg"), sb) + self.w(p + "cg0")
            pe = dy.exp(self._lin(p + "pe", r))
            c6, reset = self.np[p + "cw"], self.np[p + "reset"]
            Snp = cache.get("S")
            if Snp is None:
                Snp = np.zeros((H, dv, dk), np.float32)
            ones = _const(np.ones((dv, 1), np.float32))
            ys, Snew = [], []
            for h in range(H):
                dt = dy.pick(sp, h * dk)
                a_h = hs(a, dk, h)
                lru = dy.sqrt(dy.rectify(1.0 - dy.square(dy.pick(a_h, 0))))
                if pos == 0 and reset[h] != 0:
                    lru = lru + float(reset[h]) * (1.0 - lru)
                sb_h = dy.pick(sb, h)
                cf = [float(v) for v in c6[:, h]]
                w_h = (cf[0] * dt + cf[1] * sb_h + cf[2] * dy.pick(pe, h) + cf[3] * dy.cmult(lru, dy.pick(si, h))
                       + cf[4] + cf[5] * dy.cmult(dt, sb_h))
                k_t = dy.reshape(kh[h], (1, dk))                              # (1, dk) (transpose fails at dk = 1)
                S1 = dy.cmult(_const(Snp[h]), ones * dy.reshape(a_h, (1, dk)))
                Sk = S1 * kh[h]                                               # (dv,)
                S2 = S1 - (Sk * dy.pick(gamma, h)) * k_t
                S3 = S2 + (vh[h] * w_h) * k_t
                ys.append(S3 * qh[h])
                Snew.append(S3.npvalue().reshape(dv, dk))
            cache["S"] = np.stack(Snew).astype(np.float32)
            y = dy.concatenate(ys) + dy.cmult(_const(self.np[p + "Dskip"].reshape(-1)), v)
            y = dy.transpose(self.w(p + "mix")) * y
            z = self._lin(p + "pz", r)
            g = self.np[p + "gate"]
            G = (dy.cmult(_const(g[0]), _silu(z)) + dy.cmult(_const(g[1]), dy.logistic(z))
                 + dy.cmult(_const(g[2]), _gelu(z, True)) + dy.cmult(_const(g[3]), z) + _const(g[4]))
            nrm = lambda t: dy.cmult(self._grouped(t, p + "avg", s["norm_eps"]), self.w(p + "norm_w")) + self.w(p + "norm_b")
            oc = self.np[p + "out_coef"]
            out = (dy.cmult(_const(oc[0]), dy.cmult(nrm(y), G)) + dy.cmult(_const(oc[1]), nrm(dy.cmult(y, G)))
                   + dy.cmult(_const(oc[2]), dy.cmult(y, G)))
            return self._lin(p + "out_proj", out)

        @staticmethod
        def _act(a, g, u):
            if a["relu2"]:
                return dy.square(dy.rectify(u))
            if a["relu"]:
                return dy.rectify(u)
            if a["gelu"] or a["geglu"]:
                return _gelu(u, a["gelu_tanh"]) if a["gelu"] else dy.cmult(_gelu(g, a["gelu_tanh"]), u)
            if a["mish"]:
                return dy.cmult(dy.cmult(g, dy.tanh(_softplus(g))), u)
            if a.get("xielu"):
                ap, an, beta, eps = a["xielu"]
                pos_ = (u.npvalue().reshape(-1) > 0).astype(np.float32)     # numpy picks the branch per unit
                neg = (dy.exp(_clip(u, None, eps)) - 1.0 - u) * an + u * beta
                return dy.cmult(_const(pos_), dy.square(u) * ap + u * beta) + dy.cmult(_const(1 - pos_), neg)
            if a["c_clip"] != 0:
                lg, lu = a["clip"]
                if lg is not None:
                    g = g + a["c_clip"] * (_clip(g, None, lg) - g)
                if lu is not None:
                    u = u + a["c_clip"] * (_clip(u, -lu, lu) - u)
            if a["up_shift"] != 0:
                u = u + a["up_shift"]
            return dy.cmult(_silu(g), u)

        def _ffn(self, s, p, x, xe=None, xg=None, parts=False):
            """routed(xe) (router on xg) + shared(x) * gate; parts: (routed, gated shared) (Gemma-4's split FFN)."""
            a = s["act"]
            xs, xe, xg = x, (x if xe is None else xe), (x if xg is None else xg)
            out = routed = shared = None
            r = s["routed"]
            if r is not None:
                E = r["E"]
                x = xe
                logits = self._lin(p + "router", xg)
                if r["router_cap"] is not None:
                    logits = r["router_cap"] * dy.tanh(logits * (1.0 / r["router_cap"]))
                sc_ = r["score_c"]
                pr = dy.softmax(logits) if sc_ == 0 else (1 - sc_) * dy.softmax(logits) + sc_ * dy.logistic(logits)
                sel = ((logits if r["sel_logit"] else pr) + self.w(p + "sel_bias")).npvalue().reshape(E)
                G, kg = int(r["grp"][0]), int(r["grp"][1])
                if G > 1 and kg < G:
                    sg = sel.reshape(G, E // G)
                    gs = sg.max(-1) if r["grp"][2] != 0 else np.sort(sg, -1)[:, -2:].sum(-1)
                    keep = np.zeros(G, bool)
                    keep[np.argsort(-gs, kind="stable")[:kg]] = True
                    sel = np.where(np.repeat(keep, E // G), sel, r["grp"][3])
                top = np.argsort(-sel, kind="stable")[:r["top_k"]]
                topv = [dy.pick(pr, int(e)) for e in top]
                if r["renorm"] != 0:
                    tot = dy.esum(topv) + r["renorm_eps"]
                    topv = [t + r["renorm"] * (dy.cdiv(t, tot) - t) for t in topv]
                es = self.np[p + "exp_scale"]
                terms = []
                for e, t in zip(top, topv):
                    e = int(e)
                    g = self.w("%sw_gate.%d" % (p, e)) * x + self.w("%sb_gate.%d" % (p, e))
                    u = self.w("%sw_up.%d" % (p, e)) * x + self.w("%sb_up.%d" % (p, e))
                    eo = self.w("%sw_down.%d" % (p, e)) * self._act(a, g, u) + self.w("%sb_down.%d" % (p, e))
                    terms.append(eo * (t * (r["route_scale"] * float(es[e]))))
                out = routed = dy.esum(terms)
            x = xs
            sh = s["shared"]
            if sh is not None:
                u = self._lin(p + "s_up", x)
                g = self._lin(p + "s_gate", x) if self.has(p + "s_gate.weight") else u
                y = self._lin(p + "s_down", self._act(a, g, u))
                c = sh["s_coef"]
                if c[1] != 0:
                    y = y * (c[0] + c[1] * dy.logistic(self.w(p + "s_router.weight") * x))
                elif c[0] != 1:
                    y = y * c[0]
                shared = y
                out = y if out is None else out + y
            return (routed, shared) if parts else out

        def _layer(self, i, L, x, cache, pos, reg, pli):
            p = "layers.%d." % i

            def post(j, o, nm):
                pc = L["post_coef"][j]
                r = o * pc[0] if pc[0] != 1 else o
                return r + pc[1] * self._norm(L[nm], p + nm, o) if pc[1] != 0 else r
            if L["sn_in"] is not None:
                x = self._norm(L["sn_in"], p + "sn_in", x)
            a = s = None
            if L["attn"] is not None:
                a = post(0, self._attn(L["attn"], p + "attn.", self._norm(L["norm_a"], p + "norm_a", x), cache, pos, reg),
                         "post_a")
            if L["ssm"] is not None:
                xs = x if a is None else x + a * L["seq"]
                s = post(1, self._ssm(L["ssm"], p + "ssm.", self._norm(L["norm_s"], p + "norm_s", xs), cache, pos), "post_s")
            x_in = x
            x = x if L["res_c"][0] == 1 else x * L["res_c"][0]
            if a is not None:
                x = x + a
            if s is not None:
                x = x + s
            xf = x_in if L["par"] != 0 else x
            if L["res_c"][1] != 1:
                x = x * L["res_c"][1]
            if L["ffn"] is not None and L.get("ffn_split"):
                nf = lambda nm: self._norm(L[nm], p + nm, xf)
                r_, s_ = self._ffn(L["ffn"], p + "ffn.", nf("norm_f"), xe=nf("norm_f2"), xg=nf("norm_r"), parts=True)
                ys = [self._norm(L[nm], p + nm, part) for part, nm in ((s_, "post_f1"), (r_, "post_f2")) if part is not None]
                x = x + post(2, dy.esum(ys), "post_f")
            elif L["ffn"] is not None:
                x = x + post(2, self._ffn(L["ffn"], p + "ffn.", self._norm(L["norm_f"], p + "norm_f", xf)), "post_f")
            if L.get("ple"):                                  # per-layer embedding input
                g = _gelu(self.w(p + "ple_gate.weight") * x, L["ple"]["tanh"])
                y = self.w(p + "ple_proj.weight") * dy.cmult(g, pli[i])
                x = x + self._norm(L["ple"]["post"], p + "ple_post", y) * L["ple"]["c"]
            if L["sn_out"] is not None:
                x = self._norm(L["sn_out"], p + "sn_out", x)
            return x * L["out_scale"] if L["out_scale"] != 1 else x

        # --- the model --------------------------------------------------------------------------------------
        def new_cache(self, max_len=None):
            return [{} for _ in range(self.c["L"])]

        def _token(self, tid, cache, pos):
            c = self.c
            dy.renew_cg()
            x = dy.lookup(self.E, tid)
            if c["max_pos"]:
                x = x + dy.lookup(self.POS, pos)
            if c["emb_norm"] is not None:
                x = self._norm(c["emb_norm"], "emb_norm", x)
            pli = None
            if c.get("ple"):                                  # pli = rms(P x0) w + E[ids], per layer
                dp = c["ple"]["dim"]
                pp, pe = self.w("ple_P.weight") * x, dy.lookup(self.PLE, tid)
                pli = [dy.cmult(self._rms(dy.pick_range(pp, l * dp, (l + 1) * dp), c["ple"]["eps"]), self.w("ple_w"))
                       + dy.pick_range(pe, l * dp, (l + 1) * dp) for l in range(c["L"])]
            reg = {}
            for i, L in enumerate(c["layers"]):
                x = self._layer(i, L, x, cache[i], pos, reg, pli)
            out = self._lin("lm_head", self._norm(c["norm"], "norm", x))
            if c["final_cap"] is not None:
                out = c["final_cap"] * dy.tanh(out * (1.0 / c["final_cap"]))
            return out.npvalue()

        def step(self, ids, cache, pos):
            return np.stack([self._token(t, cache, pos + j) for j, t in enumerate(ids)])


    def load_model(run_dir):
        return UCoreModel(run_dir)


if __name__ == '__main__':
    sys.exit(run_cli(lambda d: load_model(d), 'dynet', build=build, defaults=DEFAULTS))
