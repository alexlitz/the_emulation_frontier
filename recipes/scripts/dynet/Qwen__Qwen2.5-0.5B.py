#!/usr/bin/env python3
"""Emulation-frontier recipe: run Qwen/Qwen2.5-0.5B in dynet.

Route: native: the target's own Llama-style weights.
Written by `python -m recipes.make_recipe Qwen/Qwen2.5-0.5B --framework dynet` (https://github.com/alexlitz/the_emulation_frontier).

    python Qwen__Qwen2.5-0.5B-dynet.py build [--out DIR]      # stage 1: download
    python Qwen__Qwen2.5-0.5B-dynet.py run [--dir DIR] [--prompt TEXT | --ids 1,2,3] [--new-tokens 32]
"""
# requirements-build: none (downloads over HTTPS with the standard library)
# requirements-run: dyNET==2.1.2 numpy<1.25  on CPython 3.8  (optional: tokenizers<0.21)

import os
import sys

GITHUB = 'https://github.com/alexlitz/the_emulation_frontier'
DEFAULTS = {'target': 'Qwen/Qwen2.5-0.5B', 'framework': 'dynet', 'route': 'native', 'host': None, 'loader': None, 'out': './Qwen__Qwen2.5-0.5B-dynet'}

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
    # ---- recipes/common/llama_spec.py ----------------------------------------
    """The Llama-style decoder as a plain spec read off a Hub config.json (no transformers): the shape, per-layer
    sliding windows, biases, qk-norm and the RoPE table, with HF transformers' own conventions (fp32 inv_freq,
    `kv > q - W` windows, Qwen's max_window_layers). Every framework's llama.py runs this spec. Inlined into recipes.

    Supported (anything else raises Unsupported with the reason):
      model_type  llama, mistral, ministral, qwen2, qwen3
      rope        default, linear, llama3, yarn; partial_rotary_factor
      FFN         dense SwiGLU (hidden_act silu); optional mlp_bias (llama)
    """
    import math

    import numpy as np

    LLAMA_TYPES = ("llama", "mistral", "ministral", "qwen2", "qwen3")
    ROPE_TYPES = ("default", "linear", "llama3", "yarn")


    class Unsupported(Exception):
        pass


    def _rope_params(cfg):
        rp = cfg.get("rope_parameters")
        rs = cfg.get("rope_scaling")
        if isinstance(rp, dict) and "rope_theta" in rp:          # transformers 5 layout
            if any(isinstance(v, dict) for v in rp.values()):
                raise Unsupported("per-layer-type rope_parameters")
            return float(rp["rope_theta"]), dict(rp)
        theta = float(cfg.get("rope_theta", 10000.0) or 10000.0)
        return theta, dict(rs or {})


    def rope_table(cfg, hd):
        """(inv_freq fp32 (rot/2,), attention factor, rot_dim)."""
        theta, rs = _rope_params(cfg)
        typ = rs.get("rope_type", rs.get("type", "default")) or "default"
        if typ not in ROPE_TYPES:
            raise Unsupported("rope type %r" % typ)
        prf = float(cfg.get("partial_rotary_factor", rs.get("partial_rotary_factor", 1.0)) or 1.0)
        dim = int(hd * prf)
        f32 = np.float32
        ar = np.arange(0, dim, 2).astype(f32) / f32(dim)
        pos_freqs = (f32(theta) ** ar).astype(f32)
        inv = (f32(1.0) / pos_freqs).astype(f32)
        att = 1.0
        if typ == "linear":
            inv = (inv / f32(rs["factor"])).astype(f32)
        elif typ == "llama3":
            factor = rs["factor"]
            lo, hi = rs["low_freq_factor"], rs["high_freq_factor"]
            old = rs["original_max_position_embeddings"]
            lo_w, hi_w = old / lo, old / hi
            wl = (2 * math.pi / inv).astype(f32)
            il = np.where(wl > lo_w, inv / f32(factor), inv).astype(f32)
            sm = ((old / wl - lo) / (hi - lo)).astype(f32)
            smoothed = ((1 - sm) * il / f32(factor) + sm * il).astype(f32)
            med = ~(wl < hi_w) & ~(wl > lo_w)
            inv = np.where(med, smoothed, il).astype(f32)
        elif typ == "yarn":
            factor = rs["factor"]
            att = rs.get("attention_factor")
            ms, msa = rs.get("mscale"), rs.get("mscale_all_dim")
            omax = rs.get("original_max_position_embeddings") or cfg.get("max_position_embeddings")

            def gm(s, m=1):
                return 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0
            if att is None:
                att = float(gm(factor, ms) / gm(factor, msa)) if (ms and msa) else gm(factor)
            bf, bs = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1

            def corr(nr):
                return (dim * math.log(omax / (nr * 2 * math.pi))) / (2 * math.log(theta))
            lo, hi = corr(bf), corr(bs)
            if rs.get("truncate", True):
                lo, hi = math.floor(lo), math.ceil(hi)
            lo, hi = max(lo, 0), min(hi, dim - 1)
            if lo == hi:
                hi += 0.001
            ramp = np.clip((np.arange(dim // 2, dtype=f32) - f32(lo)) / f32(hi - lo), 0, 1).astype(f32)
            ext = (f32(1) - ramp).astype(f32)
            interp = (f32(1.0) / (f32(factor) * pos_freqs)).astype(f32)
            inv = (interp * (1 - ext) + inv * ext).astype(f32)
        return inv, float(att), dim


    def llama_spec(cfg):
        """dict spec of a Llama-style config.json; Unsupported if it is not one this reference runs."""
        mt = cfg.get("model_type")
        if mt not in LLAMA_TYPES:
            raise Unsupported("model_type %r is not a Llama-style decoder" % mt)
        if cfg.get("quantization_config"):
            raise Unsupported("quantized checkpoint")
        for k in ("num_local_experts", "num_experts", "moe_intermediate_size"):
            if cfg.get(k):
                raise Unsupported("MoE (%s)" % k)
        act = cfg.get("hidden_act", "silu")
        if act not in ("silu", "swish"):
            raise Unsupported("hidden_act %r" % act)
        d, L, H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
        KV = cfg.get("num_key_value_heads") or H
        hd = cfg.get("head_dim") or d // H
        inv, att, rot = rope_table(cfg, hd)
        # windows (None = full attention), HF semantics: key s visible to query t iff t - W < s <= t
        win = [None] * L
        lt = cfg.get("layer_types")
        if mt in ("qwen2", "qwen3"):
            sw = cfg.get("sliding_window") if cfg.get("use_sliding_window", False) else None
            if lt is None:
                mwl = cfg.get("max_window_layers", 28)
                lt = ["sliding_attention" if sw is not None and i >= mwl else "full_attention" for i in range(L)]
            win = [sw if t == "sliding_attention" else None for t in lt]
        elif mt == "mistral":
            win = [cfg.get("sliding_window")] * L
        elif mt == "ministral" or lt is not None:
            sw = cfg.get("sliding_window")
            lt = lt or ["sliding_attention" if sw else "full_attention"] * L
            win = [sw if t == "sliding_attention" else None for t in lt]
        if mt == "qwen2":
            qkv_b, o_b = True, False
        elif mt in ("llama", "qwen3"):
            qkv_b = o_b = bool(cfg.get("attention_bias", False))
        else:
            qkv_b = o_b = False
        return {
            "model_type": mt, "vocab": cfg["vocab_size"], "d": d, "L": L, "H": H, "KV": KV, "hd": hd,
            "ff": cfg["intermediate_size"], "eps": float(cfg.get("rms_norm_eps", 1e-6)),
            "tie": bool(cfg.get("tie_word_embeddings", False)),      # the llama / mistral / qwen config default
            "inv_freq": inv, "rope_att": att, "rot": rot, "windows": win,
            "qkv_bias": qkv_b, "o_bias": o_b, "mlp_bias": bool(cfg.get("mlp_bias", False)) and mt == "llama",
            "qk_norm": mt == "qwen3", "scale": hd ** -0.5,
            "eos": cfg.get("eos_token_id"),
        }


    def llama_weights(spec, sd):
        """The tensors of a Llama-style checkpoint under short per-layer names (numpy):
        {"embed", "norm", "head", "layers": [{"q","k","v","o","bq","bk","bv","bo","qn","kn","ln1","ln2","g","u","dn", ...}]}"""
        pre = "model."
        if pre + "embed_tokens.weight" not in sd:
            pre = ""
        W = {"embed": sd[pre + "embed_tokens.weight"], "norm": sd[pre + "norm.weight"]}
        W["head"] = W["embed"] if spec["tie"] else sd["lm_head.weight"]
        layers = []
        for i in range(spec["L"]):
            p = "%slayers.%d." % (pre, i)
            a, m = p + "self_attn.", p + "mlp."
            l = {"q": sd[a + "q_proj.weight"], "k": sd[a + "k_proj.weight"], "v": sd[a + "v_proj.weight"],
                 "o": sd[a + "o_proj.weight"], "ln1": sd[p + "input_layernorm.weight"],
                 "ln2": sd[p + "post_attention_layernorm.weight"],
                 "g": sd[m + "gate_proj.weight"], "u": sd[m + "up_proj.weight"], "dn": sd[m + "down_proj.weight"]}
            for n, k in (("bq", "q_proj"), ("bk", "k_proj"), ("bv", "v_proj"), ("bo", "o_proj")):
                if a + k + ".bias" in sd:
                    l[n] = sd[a + k + ".bias"]
            for n, k in (("bg", "gate_proj"), ("bu", "up_proj"), ("bd", "down_proj")):
                if m + k + ".bias" in sd:
                    l[n] = sd[m + k + ".bias"]
            if spec["qk_norm"]:
                l["qn"], l["kn"] = sd[a + "q_norm.weight"], sd[a + "k_norm.weight"]
            layers.append(l)
        W["layers"] = layers
        return W
    # ---- recipes/frameworks/dynet/llama.py ----------------------------------------
    """Llama-style decoder in DyNet 2.1 (Python 3.8; no transformers): RMSNorm, RoPE (default / linear / llama3 /
    yarn, partial), GQA, SwiGLU, optional biases, Qwen-3 qk-norm, per-layer sliding windows, tied / untied head;
    greedy generation with a KV cache. fp32.

    DyNet builds a fresh computation graph per token, so tokens run one at a time (a prompt is fed token by
    token: exactly causal attention). The weights are DyNet parameters; the KV cache holds the computed keys and
    values as numpy arrays and is fed back as input tensors; a sliding window keeps only the last W keys."""
    import os

    import numpy as np

    import dynet_config

    dynet_config.set(mem=int(os.environ.get("DYNET_MEM", "4096")), random_seed=1)
    import dynet as dy  # noqa: E402



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


    def _rms(x, w, eps, n):
        return dy.cmult(x, w) * dy.pow(_sum_sq_split(x, n) * (1.0 / n) + eps, dy.scalarInput(-0.5))


    def _rot_half(x, r):
        h = r // 2
        return dy.concatenate([-dy.pick_range(x, h, r), dy.pick_range(x, 0, h)])


    class LlamaModel(Generator):
        def __init__(self, run_dir):
            cfg, sd = load_run_dir(run_dir)
            self.s = s = llama_spec(cfg)
            W = llama_weights(s, sd)
            self.pc = pc = dy.ParameterCollection()
            P = lambda a: pc.parameters_from_numpy(np.asarray(a, np.float32))
            self.E = pc.lookup_parameters_from_numpy(np.asarray(W["embed"], np.float32))
            self.norm, self.head = P(W["norm"]), P(W["head"])
            self.layers = [{k: P(v) for k, v in l.items()} for l in W["layers"]]

        def new_cache(self, max_len=None):
            return [([], []) for _ in range(self.s["L"])]       # per layer: keys, values (each (KV, hd) numpy)

        def _rope(self, x, cos, sin):
            r, hd = self.s["rot"], self.s["hd"]
            xr = dy.pick_range(x, 0, r)
            out = dy.cmult(xr, cos) + dy.cmult(_rot_half(xr, r), sin)
            return out if r == hd else dy.concatenate([out, dy.pick_range(x, r, hd)])

        def _token(self, tid, cache, pos):
            s = self.s
            H, KV, hd, d = s["H"], s["KV"], s["hd"], s["d"]
            dy.renew_cg()
            ang = np.float32(pos) * s["inv_freq"]
            emb = np.concatenate((ang, ang)).astype(np.float32)
            cos = dy.inputTensor((np.cos(emb) * np.float32(s["rope_att"])).astype(np.float32))
            sin = dy.inputTensor((np.sin(emb) * np.float32(s["rope_att"])).astype(np.float32))
            x = dy.lookup(self.E, tid)
            g = H // KV
            for i, l in enumerate(self.layers):
                h = _rms(x, l["ln1"], s["eps"], d)
                q, k, v = l["q"] * h, l["k"] * h, l["v"] * h
                if "bq" in l:
                    q, k, v = q + l["bq"], k + l["bk"], v + l["bv"]
                ks, vs = [], []
                for j in range(KV):
                    kj = dy.pick_range(k, j * hd, (j + 1) * hd)
                    if s["qk_norm"]:
                        kj = _rms(kj, l["kn"], s["eps"], hd)
                    ks.append(self._rope(kj, cos, sin))
                    vs.append(dy.pick_range(v, j * hd, (j + 1) * hd))
                knew = np.stack([e.npvalue().reshape(hd) for e in ks])
                vnew = np.stack([e.npvalue().reshape(hd) for e in vs])
                Kc, Vc = cache[i]
                Kc.append(knew)
                Vc.append(vnew)
                w = s["windows"][i]
                lo = 0 if w is None else max(0, len(Kc) - w)          # keys s > t - W
                Kall, Vall = np.stack(Kc[lo:], 2), np.stack(Vc[lo:], 2)   # (KV, hd, S)
                heads = []
                for hh in range(H):
                    qh = dy.pick_range(q, hh * hd, (hh + 1) * hd)
                    if s["qk_norm"]:
                        qh = _rms(qh, l["qn"], s["eps"], hd)
                    qh = self._rope(qh, cos, sin)
                    Km = dy.inputTensor(Kall[hh // g])
                    Vm = dy.inputTensor(Vall[hh // g])
                    p = dy.softmax(dy.transpose(Km) * qh * s["scale"])
                    heads.append(Vm * p)
                o = l["o"] * dy.concatenate(heads)
                if "bo" in l:
                    o = o + l["bo"]
                x = x + o
                h = _rms(x, l["ln2"], s["eps"], d)
                gt, up = l["g"] * h, l["u"] * h
                if "bg" in l:
                    gt, up = gt + l["bg"], up + l["bu"]
                f = l["dn"] * dy.cmult(dy.cmult(gt, dy.logistic(gt)), up)
                if "bd" in l:
                    f = f + l["bd"]
                x = x + f
            return (self.head * _rms(x, self.norm, s["eps"], d)).npvalue()

        def step(self, ids, cache, pos):
            return np.stack([self._token(t, cache, pos + j) for j, t in enumerate(ids)])


    def load_model(run_dir):
        return LlamaModel(run_dir)


if __name__ == '__main__':
    sys.exit(run_cli(lambda d: load_model(d), 'dynet', build=build, defaults=DEFAULTS))
