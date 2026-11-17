"""An exact model of llama.cpp's GGUF file sizes: which ggml type llama-quantize gives every tensor for an ftype,
which type convert_hf_to_gguf.py writes it in for a --outtype bf16 / f16 conversion, and how many bytes the
file then takes (tensor data padded to the GGUF alignment, plus the header).

Everything that is a number or a list in llama.cpp (block sizes and block bytes, the ftype -> default type
switch, the "may this tensor be quantised" substring rules, the incompatible-shape fallback switch, the tensor
names the converter forces to F32, the alignment) is PARSED from the llama.cpp sources at a pinned commit and
written to graph/data/llama_cpp_quant.json; the branching logic of llama_tensor_get_type_impl is ported by hand
below, one comment per branch with its source line range, and the JSON records a sha256 of every function the
port mirrors, so a re-extraction at a newer commit shows which ports went stale.

  python -m graph.gguf_quant                 # fetch the pinned sources to $TEF_SCRATCH/llamacpp_src (if absent),
                                             # extract, write graph/data/llama_cpp_quant.json
  python -m graph.gguf_quant --latest        # re-pin to the current master first

Sources (raw text from raw.githubusercontent.com/ggml-org/llama.cpp/<sha>/...), read and never executed:
  ggml/src/ggml-common.h   block structs and their static_assert sizes, QK_K, K_SCALE_SIZE, ...
  ggml/src/ggml.c          type_traits[]: blck_size, type_size, is_quantized
  ggml/include/ggml.h      the ggml_type enum ids
  ggml/include/gguf.h      GGUF_DEFAULT_ALIGNMENT
  ggml/src/gguf.cpp        the writer (header / KV / tensor-info layout)
  include/llama.h          the llama_ftype enum ids
  src/llama-quant.cpp      tensor categories, tensor_allows_quantization, tensor_type_fallback,
                           llama_tensor_get_type_impl, llama_ftype_get_default_type, the counters
  src/llama-model-loader.h weight_name_comparer (the order llama-quantize visits tensors in, which the
                           i_attention_wv / i_ffn_down counters depend on)
  src/llama-arch.cpp       arch names and tensor name strings
  src/llama-hparams.cpp    n_gqa(), n_layer()
  src/llama-model.cpp, src/models/*.cpp, src/models/models.h
                           which archs / layer counts load as LLM_TYPE_70B (the attn_v bump)
  convert_hf_to_gguf.py, conversion/*.py, gguf-py/gguf/constants.py, gguf-py/gguf/quants.py
                           which tensors the converter writes as F32 regardless of --outtype

Tensor inventory format (the input of every function): a list of dicts
  {"name": "blk.3.attn_v.weight", "shape": [ne0, ne1, ...] in GGML order (ne0 = row length = the input dim of a
   linear), "src": "BF16" | "F16" | "F32" (the type in the unquantised GGUF; source_types() computes it)}
"""
import argparse
import ast
import datetime
import glob
import hashlib
import json
import math
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "llama_cpp_quant.json")
SRC_DIR = os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "llamacpp_src")
REPO = "ggml-org/llama.cpp"
PINNED = "a956a852c9c8bc1fb07206364838ddef299178c0"
RAW = "https://raw.githubusercontent.com/" + REPO + "/{sha}/{path}"

FILES = [
    "ggml/src/ggml-common.h", "ggml/src/ggml.c", "ggml/include/ggml.h", "ggml/include/gguf.h", "ggml/src/gguf.cpp",
    "include/llama.h", "src/llama-quant.cpp", "src/llama-model-loader.h", "src/llama-arch.cpp",
    "src/llama-hparams.cpp", "src/llama-model.cpp", "src/models/models.h",
    "convert_hf_to_gguf.py", "gguf-py/gguf/constants.py", "gguf-py/gguf/quants.py",
]
DIRS = [("src/models/", ".cpp"), ("conversion/", ".py")]      # every file of these, listed from the git tree

TABLES = None


# ----------------------------------------------------------------------------------------------------------------
# fetching
# ----------------------------------------------------------------------------------------------------------------

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "gguf_quant"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def latest_commit():
    return json.loads(_get("https://api.github.com/repos/%s/commits/master" % REPO))["sha"]


def fetch(src_dir=SRC_DIR, sha=PINNED):
    """Download the sources at `sha` as text into src_dir (files already present are kept)."""
    os.makedirs(src_dir, exist_ok=True)
    tree_path = os.path.join(src_dir, "tree.json")
    if not os.path.exists(tree_path):
        with open(tree_path, "wb") as f:
            f.write(_get("https://api.github.com/repos/%s/git/trees/%s?recursive=1" % (REPO, sha)))
    with open(tree_path) as f:
        tree = json.load(f)
    paths = list(FILES)
    for d, ext in DIRS:
        paths += sorted(e["path"] for e in tree["tree"]
                        if e["type"] == "blob" and e["path"].startswith(d) and e["path"].endswith(ext)
                        and "/" not in e["path"][len(d):])
    for p in paths:
        out = os.path.join(src_dir, p)
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        data = _get(RAW.format(sha=sha, path=p))
        with open(out, "wb") as f:
            f.write(data)
    with open(os.path.join(src_dir, "COMMIT"), "w") as f:
        f.write(sha + "\n")
    return paths


# ----------------------------------------------------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------------------------------------------------

class _Src:
    def __init__(self, src_dir, path):
        self.path = path
        with open(os.path.join(src_dir, path), encoding="utf-8") as f:
            self.text = f.read()
        self.lines = self.text.split("\n")

    def line_of(self, pattern, start=0):
        """1-based line of the first line (from `start`, 1-based exclusive) matching the regex."""
        rx = re.compile(pattern)
        for i in range(start, len(self.lines)):
            if rx.search(self.lines[i]):
                return i + 1
        raise ValueError("%s: no line matches %r" % (self.path, pattern))

    def c_block(self, pattern):
        """(first line, last line, text) of the brace block whose header line matches `pattern`."""
        first = self.line_of(pattern)
        depth, seen, i = 0, False, first - 1
        while i < len(self.lines):
            line = re.sub(r'"(\\.|[^"\\])*"', '""', self.lines[i])
            line = re.sub(r"'(\\.|[^'\\])*'", "''", line)
            line = line.split("//")[0]
            for c in line:
                if c == "{":
                    depth += 1
                    seen = True
                elif c == "}":
                    depth -= 1
            if seen and depth == 0:
                return first, i + 1, "\n".join(self.lines[first - 1:i + 1])
            i += 1
        raise ValueError("%s: unbalanced block at %d" % (self.path, first))

    def py_block(self, pattern):
        """(first line, last line, text) of the indented Python block (def / class) whose header matches."""
        first = self.line_of(pattern)
        indent = len(self.lines[first - 1]) - len(self.lines[first - 1].lstrip())
        last = first
        for i in range(first, len(self.lines)):
            s = self.lines[i]
            if s.strip() and (len(s) - len(s.lstrip())) <= indent and not s.lstrip().startswith(")"):
                break
            if s.strip():
                last = i + 1
        return first, last, "\n".join(self.lines[first - 1:last])

    def cite(self, a, b=None):
        return "%s:%d" % (self.path, a) if b is None or b == a else "%s:%d-%d" % (self.path, a, b)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


C_SIZES = {"ggml_half": 2, "ggml_fp16_t": 2, "ggml_bf16_t": 2, "ggml_fp8_e4m3_t": 1, "float": 4, "double": 8,
           "int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4,
           "int64_t": 8, "uint64_t": 8}


def _c_int(expr, defs, sizes, depth=0):
    """Evaluate an integer C constant expression (macros, sizeof, + - * / and parentheses)."""
    if depth > 20:
        raise ValueError("macro recursion in %r" % expr)
    s = re.sub(r"sizeof\s*\(\s*([A-Za-z_]\w*)\s*\)", lambda m: "(%d)" % sizes[m.group(1)], expr)

    def ident(m):
        name = m.group(0)
        if name not in defs:
            raise ValueError("unknown identifier %s in %r" % (name, expr))
        return "(%d)" % _c_int(defs[name], defs, sizes, depth + 1)
    s = re.sub(r"\b[A-Za-z_]\w*\b", ident, s)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                q = abs(a) // abs(b)              # C integer division truncates
                return q if (a >= 0) == (b >= 0) else -q
        raise ValueError("cannot evaluate %r" % expr)
    return ev(ast.parse(s.strip(), mode="eval"))


def _c_defines(text):
    defs = {}
    for m in re.finditer(r"^\s*#define\s+([A-Za-z_]\w*)[ \t]+([^\n]*)$", text, re.M):
        v = m.group(2).split("//")[0].split("/*")[0].strip()
        if v and re.fullmatch(r"[\w\s()+\-*/]+", v):
            defs.setdefault(m.group(1), v)
    return defs


# ----------------------------------------------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------------------------------------------

def _extract_types(src_dir, cites):
    common = _Src(src_dir, "ggml/src/ggml-common.h")
    ggml_c = _Src(src_dir, "ggml/src/ggml.c")
    ggml_h = _Src(src_dir, "ggml/include/ggml.h")
    defs = _c_defines(common.text)
    sizes = dict(C_SIZES)
    asserts = {}
    for i, line in enumerate(common.lines):
        m = re.match(r"\s*static_assert\s*\(\s*sizeof\s*\(\s*(block_\w+)\s*\)\s*==\s*(.*),\s*\"[^\"]*\"\s*\)\s*;", line)
        if m:
            sizes[m.group(1)] = _c_int(m.group(2), defs, sizes)
            asserts[m.group(1)] = i + 1
    ids = {}
    for i, line in enumerate(ggml_h.lines):
        m = re.match(r"\s*GGML_TYPE_(\w+)\s*=\s*(\d+)\s*,", line)
        if m and m.group(1) != "COUNT":
            ids[m.group(1)] = int(m.group(2))
    types = {}
    ta, tb, traits = ggml_c.c_block(r"type_traits\[GGML_TYPE_COUNT\] = \{")
    for m in re.finditer(r"^\s*\[GGML_TYPE_(\w+)\]\s*=\s*\{(.*?)\n\s*\},", traits, re.S | re.M):
        name, body = m.group(1), m.group(2)
        line = ta + traits.count("\n", 0, m.start())
        blck = _c_int(re.search(r"\.blck_size\s*=\s*([^,\n]+),", body).group(1), defs, sizes)
        tsz_expr = re.search(r"\.type_size\s*=\s*([^,\n]+),", body).group(1).strip()
        tsz = _c_int(tsz_expr, defs, sizes)
        quant = re.search(r"\.is_quantized\s*=\s*(true|false)", body).group(1) == "true"
        blk = re.fullmatch(r"sizeof\s*\(\s*(block_\w+)\s*\)", tsz_expr)
        t = {"id": ids[name], "block": blck, "bytes": tsz, "bpw": 8.0 * tsz / blck, "quantized": quant,
             "cite": [ggml_c.cite(line)]}
        if blk:
            t["cite"].append(common.cite(asserts[blk.group(1)]))
        types[name] = t
    cites["types"] = [ggml_c.cite(ggml_c.line_of(r"type_traits\[GGML_TYPE_COUNT\]"))]
    return types


def _extract_quant(src_dir, types, arch_names, tensor_names, cites, hashes):
    q = _Src(src_dir, "src/llama-quant.cpp")
    out = {}

    # ftype -> default type
    a, b, body = q.c_block(r"^ggml_type llama_ftype_get_default_type\(")
    ftypes, pending = {}, []
    for line in body.split("\n"):
        for tok in re.finditer(r"case LLAMA_FTYPE_(?:MOSTLY_|ALL_)?(\w+):|return GGML_TYPE_(\w+);", line):
            if tok.group(1):
                pending.append(tok.group(1))
            elif tok.group(2) != "COUNT":
                for f in pending:
                    ftypes[f] = tok.group(2)
                pending = []
            else:
                pending = []
    out["ftypes"] = ftypes
    cites["ftypes"] = q.cite(a, b)
    hashes["llama_ftype_get_default_type"] = _sha(body)

    # tensor_allows_quantization
    a, b, body = q.c_block(r"^static bool tensor_allows_quantization\(")
    rules, arch = [], None
    for k, raw in enumerate(body.split("\n")[1:-1]):
        line_no = a + 1 + k
        s = raw.split("//")[0].strip()
        if not s:
            continue
        m = re.fullmatch(r"if \(arch == LLM_ARCH_(\w+)\) \{", s)
        if m:
            arch = arch_names[m.group(1)]
            continue
        if s == "}":
            arch = None
            continue
        if s in ("if (params->only_copy)       return false;", "if (params->only_copy) return false;",
                 "const std::string name = ggml_get_name(tensor);", "return quantize;"):
            continue
        if re.fullmatch(r"if \(ggml_n_dims\(tensor\) < 2\) return false;", s):
            rules.append({"op": "min_dims", "n": 2, "arch": arch, "line": line_no})
            continue
        if s == 'bool quantize = name.rfind("weight") == name.size() - 6; // ends with \'weight\'?' or \
                s == 'bool quantize = name.rfind("weight") == name.size() - 6;':
            rules.append({"op": "ends_weight", "arch": arch, "line": line_no})
            continue
        if s == 'quantize &= params->quantize_output_tensor || name != "output.weight";':
            continue                                        # default params: quantize_output_tensor = true
        m = (re.fullmatch(r'quantize &= name\.find\("([^"]+)"\)\s*== std::string::npos;', s))
        if m:
            rules.append({"op": "not_contains", "s": m.group(1), "arch": arch, "line": line_no})
            continue
        m = (re.fullmatch(r'quantize &= name\.rfind\("([^"]+)", 0\)\s*(?:!= 0|== std::string::npos);', s))
        if m:
            rules.append({"op": "not_prefix", "s": m.group(1), "arch": arch, "line": line_no})
            continue
        m = re.fullmatch(r'quantize &= name\.compare\(0, (\d+), "([^"]+)"\) != 0;', s)
        if m:
            if int(m.group(1)) != len(m.group(2)):
                raise ValueError("compare length mismatch at %d" % line_no)
            rules.append({"op": "not_prefix", "s": m.group(2), "arch": arch, "line": line_no})
            continue
        m = re.fullmatch(r'quantize &= name != LLM_TN\(arch\)\(LLM_TENSOR_(\w+),\s*"(\w+)"\);', s)
        if m:
            rules.append({"op": "not_equal", "s": tensor_names[m.group(1)] + "." + m.group(2), "arch": arch,
                          "line": line_no})
            continue
        raise ValueError("unrecognised rule in tensor_allows_quantization, line %d: %s" % (line_no, s))
    out["allow_rules"] = rules
    cites["allow_rules"] = q.cite(a, b)
    hashes["tensor_allows_quantization"] = _sha(body)

    # tensor_type_fallback
    a, b, body = q.c_block(r"^static ggml_type tensor_type_fallback\(")
    sw = body[body.index("switch (target_type)"):body.index("default:")]
    fallback, pending = {}, []
    for tok in re.finditer(r"case GGML_TYPE_(\w+):|return_type = GGML_TYPE_(\w+);", sw):
        if tok.group(1):
            pending.append(tok.group(1))
        else:
            for t in pending:
                fallback[t] = tok.group(2)
            pending = []
    out["fallback"] = fallback
    cites["fallback"] = q.cite(a, b)
    hashes["tensor_type_fallback"] = _sha(body)

    for key, pat in [("tensor_name_match_token_embd", r"^static bool tensor_name_match_token_embd\("),
                     ("tensor_name_match_output_weight", r"^static bool tensor_name_match_output_weight\("),
                     ("tensor_get_category", r"^static tensor_category tensor_get_category\("),
                     ("tensor_is_expert", r"^static bool tensor_is_expert\("),
                     ("category_is_attn_v", r"^static bool category_is_attn_v\("),
                     ("llama_tensor_get_type_impl", r"^static ggml_type llama_tensor_get_type_impl\("),
                     ("llama_tensor_get_type", r"^static ggml_type llama_tensor_get_type\("),
                     ("init_quantize_state_counters", r"^static void init_quantize_state_counters\("),
                     ("llama_model_quantize_default_params",
                      r"^llama_model_quantize_params llama_model_quantize_default_params\(")]:
        a, b, body = q.c_block(pat)
        cites[key] = q.cite(a, b)
        hashes[key] = _sha(body)
    a, b, body = q.c_block(r"^static void llama_model_quantize_impl\(")
    cites["llama_model_quantize_impl"] = q.cite(a, b)
    ln = q.line_of(r"auto use_more_bits = ")
    cites["use_more_bits"] = q.cite(ln, ln + 2)
    for key, pat in [("preliminary_loop", r"preliminary iteration over all weights"),
                     ("dry_run_size", r"new_size = ggml_nrows\(tensor\) \* ggml_row_size"),
                     ("padding", r"zeros\(fout, GGML_PAD\(new_size, align\) - new_size\)"),
                     ("align", r"const size_t align = GGUF_DEFAULT_ALIGNMENT;")]:
        cites[key] = q.cite(q.line_of(pat))
    return out


def _extract_70b(src_dir, arch_names, cites, hashes):
    """Which archs load as LLM_TYPE_70B, at which n_layer() (src/models/*.cpp load_arch_hparams)."""
    model_cpp = _Src(src_dir, "src/llama-model.cpp")
    models_h = _Src(src_dir, "src/models/models.h")
    a, b, body = model_cpp.c_block(r"^static llama_model \* llama_model_mapping\(")
    cls_archs, pending = {}, []
    for tok in re.finditer(r"case LLM_ARCH_(\w+):|return new llama_model_(\w+)\(", body):
        if tok.group(1):
            pending.append(tok.group(1))
        else:
            cls_archs.setdefault(tok.group(2), []).extend(pending)
            pending = []
    parents = {m.group(1): m.group(2) for m in
               re.finditer(r"struct llama_model_(\w+)\s*:\s*public llama_model_(\w+)\s*\{", models_h.text)}
    rules = []
    for path in sorted(glob.glob(os.path.join(src_dir, "src/models/*.cpp"))):
        src = _Src(src_dir, os.path.relpath(path, src_dir))
        if "LLM_TYPE_70B" not in src.text:
            continue
        a, b, body = src.c_block(r"^void llama_model_(\w+)::load_arch_hparams\(")
        cls = re.match(r"void llama_model_(\w+)::", body).group(1)
        for k, line in enumerate(body.split("\n")):
            if "LLM_TYPE_70B" not in line:
                continue
            m = re.match(r"\s*case (\d+): type = (.*); break;", line)
            if not m:
                raise ValueError("unrecognised 70B line %s:%d" % (src.path, a + k))
            before = body[:body.find(line)]
            sw = re.findall(r"switch \((.*?)\) \{", before)[-1]
            if sw != "hparams.n_layer()":
                raise ValueError("70B switch on %s at %s:%d" % (sw, src.path, a + k))
            expr = m.group(2).strip()
            if expr == "LLM_TYPE_70B":
                cond = None
            elif expr == "hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B":
                cond = "gqa"
            else:
                raise ValueError("unrecognised 70B condition %r" % expr)
            not_n_expert = 8 if "if (hparams.n_expert == 8) {" in before else None
            classes = [cls] + [c for c, p in parents.items() if p == cls and
                               "llama_model_%s::load_arch_hparams" % c not in "".join(
                                   open(f).read() for f in glob.glob(os.path.join(src_dir, "src/models/*.cpp")))]
            for c in classes:
                for ar in cls_archs.get(c, []):
                    rules.append({"arch": arch_names[ar], "n_layer": int(m.group(1)), "cond": cond,
                                  "not_n_expert": not_n_expert, "cite": src.cite(a + k)})
            hashes["70b:" + src.path] = _sha(body)
    cites["type_70b"] = model_cpp.cite(model_cpp.line_of(r"^static llama_model \* llama_model_mapping\("))
    return sorted(rules, key=lambda r: (r["arch"], r["n_layer"]))


# The text-model converter classes whose tensor_force_quant forces extra tensors to F32 (mmproj / audio / TTS
# classes are out of scope: they write a separate mmproj GGUF). Each rule: all given conditions hold; a list is
# any-of. "templates" are gguf-py TENSOR_NAMES keys matched as <name>.weight. extract() checks every literal
# against the cited method body.
CONVERTER_ARCH_F32 = {
    "glm5-next": {"cls": "Glm5NextModel", "file": "conversion/glm.py", "rules": [
        {"prefix": ["blk.", "output_hc"],
         "contains": ["hc_attn_", "hc_ffn_", "indexer_compressor_", "ssm_a", "ssm_dt", "exp_probs_b"]}]},
    "limite": {"cls": "LimiteModel", "file": "conversion/limite.py", "rules": [
        {"templates": ["ATTN_GATE", "ATTN_VE_GATE", "MUDD_DOWN", "MUDD_UP"]}]},
    "metis16": {"cls": "Metis16Model", "file": "conversion/metis.py", "rules": [
        {"prefix": ["cont_", "depth_mem_entry_bias", "depth_mem_stream_gate"]},
        {"suffix": ["ffn_exp_embd.weight", "ffn_gate_inp.weight", "hc_attn_mix.weight", "hc_attn_read.weight",
                    "hc_attn_write.weight", "hc_ffn_mix.weight", "hc_ffn_read.weight", "hc_ffn_write.weight"]}]},
    "minimax-m3": {"cls": "MiniMaxM3Model", "file": "conversion/minimax.py", "rules": [
        {"contains": [".indexer."]}]},
    "vllm-decision2": {"cls": "VllmDecision2Model", "file": "conversion/vllm_decision2.py", "rules": [
        {"prefix": ["decision."]}]},
    "hy_v4": {"cls": "HYV4Model", "file": "conversion/hy_v4.py", "rules": [
        {"suffix": ["hc_attn_fn.weight", "hc_ffn_fn.weight", "output_hc_fn.weight"]},
        {"equal": ["output.weight"], "flag": "lm_head_fp32"}]},     # hparams enable_lm_head_fp32
}
# Not modelled (their F32 / BF16 / Q8_0 choice depends on the checkpoint's own dtypes, not on the tensor name):
CONVERTER_UNMODELLED = {"deepseek4": "DeepseekV4Model.tensor_force_quant (conversion/deepseek.py)",
                        "deepseek41": "DeepseekV41Model.tensor_force_quant (conversion/deepseek41.py)"}


def _template_regex(t):
    return "^" + re.escape(t).replace(r"\{bid\}", "[0-9]+").replace(r"\{xid\}", "[0-9]+") + r"\.weight$"


def _extract_converter(src_dir, cites, hashes):
    base = _Src(src_dir, "conversion/base.py")
    const = _Src(src_dir, "gguf-py/gguf/constants.py")
    a = const.line_of(r"^TENSOR_NAMES: dict\[MODEL_TENSOR, str\] = \{")
    b = const.line_of(r"^\}", a)
    tn_body = "\n".join(const.lines[a - 1:b])
    names = {m.group(1): m.group(2) for m in re.finditer(r"MODEL_TENSOR\.(\w+):\s*\"([^\"]+)\"", tn_body)}
    cites["tensor_names_py"] = const.cite(a, b)
    a, b, body = base.py_block(r"^    def _tensor_requires_f32\(")
    must = ['return n_dims <= 1 or new_name.endswith("_norm.weight") or (',
            'or new_name[-7:] not in (".weight", ".lora_a", ".lora_b")']
    for s in must:
        if s not in body:
            raise ValueError("conversion/base.py _tensor_requires_f32 changed: %r missing" % s)
    keys = re.findall(r"gguf\.MODEL_TENSOR\.(\w+)", body)
    out = {"requires_f32": [{"key": k, "template": names[k], "regex": _template_regex(names[k])} for k in keys],
           "weight_suffixes": [".weight", ".lora_a", ".lora_b"], "norm_suffix": "_norm.weight"}
    cites["requires_f32"] = base.cite(a, b)
    hashes["converter:_tensor_requires_f32"] = _sha(body)
    a, b, body = base.py_block(r"^    def prepare_tensors\(self\)")
    for s in ['if n_dims <= 1 or new_name.endswith("_norm.weight"):',
              "if data_qtype is False and self._tensor_requires_f32(new_name, bid, n_dims):",
              "elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:", "elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:",
              "except gguf.QuantError as e:"]:
        if s not in body:
            raise ValueError("conversion/base.py prepare_tensors changed: %r missing" % s)
    cites["prepare_tensors"] = base.cite(a, b)
    hashes["converter:prepare_tensors"] = _sha(body)
    a, b, body = base.py_block(r"^    def tensor_force_quant\(self, name: str")
    cites["tensor_force_quant"] = base.cite(a, b)
    hashes["converter:tensor_force_quant"] = _sha(body)
    arch = {}
    for ar, spec in CONVERTER_ARCH_F32.items():
        src = _Src(src_dir, spec["file"])
        c0 = src.line_of(r"^class %s\(" % spec["cls"])
        a, b, body = _py_block_at(src, src.line_of(r"^    def tensor_force_quant\(", c0))
        rules = []
        for r in spec["rules"]:
            rr = {}
            for k in ("prefix", "contains", "suffix", "equal"):
                if k in r:
                    for s in r[k]:
                        if '"%s"' % s not in body:
                            raise ValueError("%s: literal %r not in %s" % (spec["file"], s, src.cite(a, b)))
                    rr[k] = r[k]
            if "templates" in r:
                for k in r["templates"]:
                    if "MODEL_TENSOR.%s" % k not in body:
                        raise ValueError("%s: %s not in %s" % (spec["file"], k, src.cite(a, b)))
                rr["regex"] = [_template_regex(names[k]) for k in r["templates"]]
            if "flag" in r:
                rr["flag"] = r["flag"]
            rules.append(rr)
        arch[ar] = {"rules": rules, "cite": src.cite(a, b)}
        hashes["converter:%s" % spec["cls"]] = _sha(body)
    out["arch_f32"] = arch
    out["unmodelled"] = CONVERTER_UNMODELLED
    qs = _Src(src_dir, "gguf-py/gguf/quants.py")
    cites["quant_error"] = qs.cite(qs.line_of(r"raise QuantError"))
    return out


def _py_block_at(src, first):
    indent = len(src.lines[first - 1]) - len(src.lines[first - 1].lstrip())
    last = first
    for i in range(first, len(src.lines)):
        s = src.lines[i]
        if s.strip() and (len(s) - len(s.lstrip())) <= indent and not s.lstrip().startswith(")"):
            break
        if s.strip():
            last = i + 1
    return first, last, "\n".join(src.lines[first - 1:last])


def extract(src_dir=SRC_DIR):
    """Parse the pinned sources in src_dir into the tables (the content of graph/data/llama_cpp_quant.json)."""
    with open(os.path.join(src_dir, "COMMIT")) as f:
        sha = f.read().strip()
    cites, hashes = {}, {}
    arch_cpp = _Src(src_dir, "src/llama-arch.cpp")
    arch_names = {m.group(1): m.group(2) for m in
                  re.finditer(r"\{\s*LLM_ARCH_(\w+),\s*\"([^\"]+)\"\s*\}", arch_cpp.text)}
    tensor_names = {m.group(1): m.group(2) for m in
                    re.finditer(r"\{\s*LLM_TENSOR_(\w+),\s*\"([^\"]+)\"\s*\}", arch_cpp.text)}
    cites["arch_names"] = arch_cpp.cite(arch_cpp.line_of(r"LLM_ARCH_NAMES"))
    types = _extract_types(src_dir, cites)
    quant = _extract_quant(src_dir, types, arch_names, tensor_names, cites, hashes)
    gguf_h = _Src(src_dir, "ggml/include/gguf.h")
    align_line = gguf_h.line_of(r"#define GGUF_DEFAULT_ALIGNMENT")
    alignment = int(re.search(r"GGUF_DEFAULT_ALIGNMENT\s+(\d+)", gguf_h.lines[align_line - 1]).group(1))
    cites["alignment"] = gguf_h.cite(align_line)
    gguf_cpp = _Src(src_dir, "ggml/src/gguf.cpp")
    for key, pat in [("gguf_write_string", r"^    void write\(const std::string & val\) \{"),
                     ("gguf_write_kv", r"^    void write\(const struct gguf_kv & kv\) \{"),
                     ("gguf_write_tensor_meta", r"^    void write_tensor_meta\("),
                     ("gguf_write_out", r"^static void gguf_write_out\(")]:
        a, b, body = gguf_cpp.c_block(pat)
        cites[key] = gguf_cpp.cite(a, b)
        hashes[key] = _sha(body)
    loader = _Src(src_dir, "src/llama-model-loader.h")
    a, b, body = loader.c_block(r"struct weight_name_comparer \{")
    cites["weight_order"] = loader.cite(a, b)
    hashes["weight_name_comparer"] = _sha(body)
    hp = _Src(src_dir, "src/llama-hparams.cpp")
    for key, pat in [("n_gqa", r"^uint32_t llama_hparams::n_gqa\("), ("n_layer", r"^uint32_t llama_hparams::n_layer\(")]:
        a, b, body = hp.c_block(pat)
        cites[key] = hp.cite(a, b)
        hashes[key] = _sha(body)
    type_70b = _extract_70b(src_dir, arch_names, cites, hashes)
    conv = _extract_converter(src_dir, cites, hashes)
    llama_h = _Src(src_dir, "include/llama.h")
    ftype_ids = {}
    for line in llama_h.lines:
        m = re.match(r"\s*LLAMA_FTYPE_(?:MOSTLY_|ALL_)?(\w+)\s*=\s*(\d+)\s*,", line)
        if m:
            ftype_ids[m.group(1)] = int(m.group(2))
    files = sorted(set(FILES) | {"conversion/base.py"} | {c.split(":")[0] for c in _flat_cites(cites)})
    return {
        "commit": sha,
        "repo": REPO,
        "urls": [RAW.format(sha=sha, path=p) for p in files],
        "types": types,
        "ftypes": quant["ftypes"],
        "ftype_ids": {k: v for k, v in ftype_ids.items() if k in quant["ftypes"]},
        "alignment": alignment,
        "allow_rules": quant["allow_rules"],
        "fallback": quant["fallback"],
        "type_70b": type_70b,
        "converter": conv,
        "citations": cites,
        "source_sha256": hashes,
    }


def _flat_cites(c):
    if isinstance(c, str):
        return [c]
    if isinstance(c, list):
        return [y for x in c for y in _flat_cites(x)]
    if isinstance(c, dict):
        return [y for x in c.values() for y in _flat_cites(x)]
    return []


def write(tables, path=DATA):
    tables = dict(tables)
    tables["fetched"] = datetime.date.today().isoformat()
    keys = ["commit", "fetched"] + [k for k in tables if k not in ("commit", "fetched")]
    with open(path, "w") as f:
        json.dump({k: tables[k] for k in keys}, f, indent=1)
        f.write("\n")


def tables():
    global TABLES
    if TABLES is None:
        with open(DATA) as f:
            TABLES = json.load(f)
    return TABLES


# ----------------------------------------------------------------------------------------------------------------
# small helpers shared by the rules
# ----------------------------------------------------------------------------------------------------------------

def _n_dims(shape):
    """ggml_n_dims: the number of dims ignoring trailing 1s (at least 1)."""
    for i in range(len(shape) - 1, 0, -1):
        if shape[i] > 1:
            return i + 1
    return 1


def _n_elements(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _blk_layer(name):
    """sscanf(name, "blk.%d.", &il): the layer, or None when the conversion fails."""
    m = re.match(r"blk\.[ \t\n\r\f\v]*([+-]?[0-9]+)", name)
    return int(m.group(1)) if m else None


def _weight_order_key(name):
    # src/llama-model-loader.h:52-63 weight_name_comparer: by the blk.%d. layer (-1 when absent), then by name
    il = _blk_layer(name)
    return (-1 if il is None else il, name.encode())


# tensor categories, src/llama-quant.cpp:121-163 tensor_get_category
def tensor_category(name):
    if name == "output.weight":                                                     # :111-113, :122
        return "OUTPUT"
    if name in ("token_embd.weight", "per_layer_token_embd.weight"):                # :106-109, :125
        return "TOKEN_EMBD"
    if "attn_qkv.weight" in name or "attn_qkv_gate.weight" in name:                 # :129
        return "ATTENTION_QKV"
    if "attn_kv_b.weight" in name:                                                  # :132
        return "ATTENTION_KV_B"
    if "attn_v.weight" in name:                                                     # :135
        return "ATTENTION_V"
    if "attn_k.weight" in name:                                                     # :138
        return "ATTENTION_K"
    if "attn_q.weight" in name:                                                     # :141
        return "ATTENTION_Q"
    if "attn_output.weight" in name:                                                # :144
        return "ATTENTION_OUTPUT"
    if "ffn_up" in name:                                                            # :147
        return "FFN_UP"
    if "ffn_gate" in name:                                                          # :150
        return "FFN_GATE"
    if "ffn_latent" in name:                                                        # :153
        return "FFN_LATENT"
    if "attn_gate.weight" in name:                                                  # :156
        return "ATTENTION_GATE"
    if "ffn_down" in name:                                                          # :159
        return "FFN_DOWN"
    return "OTHER"


def _is_expert(name):
    # :166-169 tensor_is_expert
    return "_exps." in name or "_chexps." in name


def _is_attn_v(cat):
    # :172-176 category_is_attn_v
    return cat in ("ATTENTION_V", "ATTENTION_QKV", "ATTENTION_KV_B")


def allows_quantization(name, shape, arch, T=None):
    """src/llama-quant.cpp:302-444 tensor_allows_quantization, with the default params (no --only-copy,
    quantize_output_tensor = true). The substring rules come from the JSON (allow_rules)."""
    T = T or tables()
    quantize = True
    for r in T["allow_rules"]:
        if r["arch"] is not None and r["arch"] != arch:
            continue
        op = r["op"]
        if op == "min_dims":
            if _n_dims(shape) < r["n"]:
                return False
        elif op == "ends_weight":
            # name.rfind("weight") == name.size() - 6: true for a name ending in "weight", and (size_t
            # wrap-around: size - 6 == npos == rfind's miss) for any 5-character name
            quantize = name.endswith("weight") or len(name.encode()) == 5
        elif op == "not_contains":
            quantize = quantize and r["s"] not in name
        elif op == "not_prefix":
            quantize = quantize and not name.startswith(r["s"])
        elif op == "not_equal":
            quantize = quantize and name != r["s"]
        else:
            raise ValueError("unknown rule op %s" % op)
    return quantize


def _bpw_lt(T, a, b):
    ta, tb = T["types"][a], T["types"][b]
    return ta["bytes"] * tb["block"] < tb["bytes"] * ta["block"]


class _QS:
    """quantize_state_impl (src/llama-quant.cpp:182-215) plus the hparams the rules read."""
    def __init__(self, n_expert, n_gqa, n_layer, is_70b, has_imatrix):
        self.n_expert = n_expert
        self.n_gqa = n_gqa
        self.is_70b = is_70b
        self.has_imatrix = has_imatrix
        self.n_attention_wv = 0
        self.n_ffn_down = self.n_ffn_gate = self.n_ffn_up = n_layer
        self.i_attention_wv = self.i_ffn_down = self.i_ffn_gate = self.i_ffn_up = 0
        self.n_fallback = 0
        self.has_tied_embeddings = True


def _use_more_bits(i_layer, n_layers):
    # src/llama-quant.cpp:513-515 (C int division; the % operand is >= 0 whenever it is evaluated)
    return i_layer < n_layers // 8 or i_layer >= 7 * n_layers // 8 or (i_layer - n_layers // 8) % 3 == 2


def _get_type_impl(T, qs, new_type, name, ne0, ftype, cat, arch):
    """src/llama-quant.cpp:507-783 llama_tensor_get_type_impl, default params (no --output-tensor-type /
    --token-embedding-type)."""
    n_expert = max(1, qs.n_expert)                                                  # :516
    n_expert_h = qs.n_expert
    gqa = qs.n_gqa

    def layer_info(i_layer, n_layer):                                               # :517-531
        if n_expert > 1:
            i_layer = _blk_layer(name)
            if i_layer is None:
                raise ValueError("Failed to determine layer for tensor %s" % name)
            if i_layer < 0 or i_layer >= n_layer:
                raise ValueError("Bad layer %d for tensor %s. Must be in [0, %d)" % (i_layer, name, n_layer))
        return i_layer, n_layer

    # :534-547 glm5-next: attn_q_a / attn_q_b / nextn.eh_proj never below Q8_0
    if arch == "glm5-next" and ("attn_q_a" in name or "attn_q_b" in name or "nextn.eh_proj" in name):
        if new_type not in ("F32", "BF16", "F16"):
            return "Q8_0"
    # :552-560 muse-spark: ffn_latent / attn_gate bumped to Q8_0 when below its bpw
    if arch == "muse-spark" and cat in ("FFN_LATENT", "ATTENTION_GATE"):
        if T["types"][new_type]["quantized"] and _bpw_lt(T, new_type, "Q8_0"):
            new_type = "Q8_0"

    if cat == "OUTPUT" or (qs.has_tied_embeddings and cat == "TOKEN_EMBD"):          # :564-585
        qk_k = T["types"][new_type]["block"]
        if ftype == "MXFP4_MOE":                                                    # :571
            new_type = "Q8_0"
        elif arch == "falcon" or ne0 % qk_k != 0:                                   # :574
            new_type = "Q8_0"
        elif ftype in ("IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ2_S", "IQ2_M", "IQ1_M"):  # :577-580
            new_type = "Q5_K"
        elif new_type != "Q8_0":                                                    # :582
            new_type = "Q6_K"
    elif ftype == "MXFP4_MOE":                                                      # :586-593
        new_type = "MXFP4" if _is_expert(name) else "Q8_0"
    elif cat == "TOKEN_EMBD":                                                       # :594-611
        if ftype in ("IQ2_XXS", "IQ2_XS", "IQ1_S", "IQ1_M"):
            new_type = "Q2_K"
        elif ftype in ("IQ2_S", "IQ2_M"):
            new_type = "IQ3_S"
        elif ftype == "IQ3_XXS":
            new_type = "IQ3_S"
        elif ftype in ("TQ1_0", "TQ2_0", "Q2_0"):
            new_type = "Q4_K"
    elif ftype in ("IQ2_XXS", "IQ2_XS", "IQ1_S", "IQ2_S", "IQ2_M", "IQ1_M"):          # :612-635
        if _is_attn_v(cat):                                                         # :614-618
            if gqa >= 4 or n_expert_h >= 4:
                new_type = "Q4_K"
            else:
                new_type = "IQ3_S" if ftype in ("IQ2_S", "IQ2_M") else "Q2_K"
            qs.i_attention_wv += 1
        elif n_expert_h == 8 and cat == "ATTENTION_K":                              # :619-621
            new_type = "Q4_K"
        elif cat == "FFN_DOWN":                                                     # :622-627
            if qs.i_ffn_down < qs.n_ffn_down // 8:
                new_type = "IQ3_S" if ftype in ("IQ2_S", "IQ2_M") else "Q2_K"
            qs.i_ffn_down += 1
        elif cat == "ATTENTION_OUTPUT":                                             # :628-635
            if n_expert_h == 8:
                new_type = "Q5_K"
            elif ftype in ("IQ1_S", "IQ1_M"):
                new_type = "IQ2_XXS"
            elif ftype in ("IQ2_S", "IQ2_M"):
                new_type = "IQ3_S"
    elif _is_attn_v(cat):                                                           # :636-673
        if ftype == "Q2_K":                                                         # :637-639
            new_type = "Q4_K" if gqa >= 4 else "Q3_K"
        elif ftype == "Q2_K_S" and gqa >= 4:                                        # :640-642
            new_type = "Q4_K"
        elif ftype == "IQ3_XXS":                                                    # :643-645
            new_type = "Q4_K" if gqa >= 4 else ("IQ3_S" if not qs.has_imatrix else "IQ3_XXS")
        elif ftype in ("IQ3_XS", "IQ3_S") and gqa >= 4:                             # :646-648
            new_type = "Q4_K"
        elif ftype == "IQ3_M":                                                      # :649-651
            new_type = "Q4_K"
        elif ftype == "Q3_K_M":                                                     # :652-654
            new_type = "Q5_K" if qs.i_attention_wv < 2 else "Q4_K"
        elif ftype == "Q3_K_L":                                                     # :655
            new_type = "Q5_K"
        elif ftype in ("IQ4_NL", "IQ4_XS") and gqa >= 4:                            # :656-658
            new_type = "Q5_K"
        elif ftype in ("Q4_K_M", "Q5_K_M") and _use_more_bits(qs.i_attention_wv, qs.n_attention_wv):  # :659-660
            new_type = "Q6_K"
        elif ftype == "Q4_K_S" and qs.i_attention_wv < 4:                           # :661
            new_type = "Q5_K"
        if qs.is_70b:                                                               # :662-667
            if new_type in ("Q3_K", "Q4_K"):
                new_type = "Q5_K"
        if n_expert_h == 8:                                                         # :668-672
            new_type = "Q8_0"
        qs.i_attention_wv += 1                                                      # :673
    elif cat == "ATTENTION_K":                                                      # :674-685
        if n_expert_h == 8:
            new_type = "Q8_0"
        elif ftype == "IQ3_XS":
            new_type = "IQ3_XXS"
        elif ftype == "IQ3_XXS":
            new_type = "IQ2_S"
    elif cat == "ATTENTION_Q":                                                      # :686-692
        if ftype == "IQ3_XS":
            new_type = "IQ3_XXS"
        elif ftype == "IQ3_XXS":
            new_type = "IQ2_S"
    elif cat == "FFN_DOWN":                                                         # :693-737
        i_layer, n_layer = layer_info(qs.i_ffn_down, qs.n_ffn_down)
        if ftype == "Q2_K":                                                         # :696
            new_type = "Q3_K"
        elif ftype == "Q2_K_S":                                                     # :697-699
            if i_layer < n_layer // 8:
                new_type = "Q4_K"
        elif ftype == "IQ3_XXS" and not qs.has_imatrix:                             # :700-702
            new_type = "Q4_K" if i_layer < n_layer // 8 else "Q3_K"
        elif ftype == "Q3_K_M":                                                     # :703-707
            new_type = ("Q5_K" if i_layer < n_layer // 16 else
                        "Q4_K" if arch != "falcon" or _use_more_bits(i_layer, n_layer) else "Q3_K")
        elif ftype == "IQ3_M" and (i_layer < n_layer // 8 or
                                   (n_expert_h == 8 and _use_more_bits(i_layer, n_layer))):  # :708-711
            new_type = "Q4_K"
        elif ftype == "Q3_K_L":                                                     # :712-714
            new_type = "Q4_K" if arch == "falcon" else "Q5_K"
        elif ftype == "Q4_K_M":                                                     # :715-722
            if arch == "falcon":
                new_type = ("Q6_K" if i_layer < n_layer // 16 else
                            "Q5_K" if _use_more_bits(i_layer, n_layer) else "Q4_K")
            elif _use_more_bits(i_layer, n_layer):
                new_type = "Q6_K"
        elif i_layer < n_layer // 8 and ftype in ("IQ4_NL", "IQ4_XS") and not qs.has_imatrix:  # :723-725
            new_type = "Q5_K"
        elif ftype == "Q5_K_M" and _use_more_bits(i_layer, n_layer):                # :726
            new_type = "Q6_K"
        elif ftype == "Q4_K_S" and arch != "falcon" and i_layer < n_layer // 8:     # :727-729
            new_type = "Q5_K"
        elif ftype in ("Q4_0", "Q5_0") and qs.has_imatrix and i_layer < n_layer // 8:  # :730-736
            new_type = "Q4_1" if ftype == "Q4_0" else "Q5_1"
        qs.i_ffn_down += 1                                                          # :737
    elif cat == "ATTENTION_OUTPUT":                                                 # :738-757
        if arch != "falcon":
            if n_expert_h == 8:                                                     # :740-746
                if ftype in ("Q2_K", "IQ3_XS", "IQ3_XXS", "Q3_K_S", "Q3_K_M", "IQ4_NL", "Q4_K_S", "Q4_K_M",
                             "IQ3_S", "IQ3_M", "IQ4_XS"):
                    new_type = "Q5_K"
            else:                                                                   # :747-753
                if ftype == "Q2_K":
                    new_type = "Q3_K"
                elif ftype == "IQ3_XXS":
                    new_type = "IQ3_S"
                elif ftype == "Q3_K_M":
                    new_type = "Q4_K"
                elif ftype == "Q3_K_L":
                    new_type = "Q5_K"
                elif ftype == "IQ3_M":
                    new_type = "Q4_K"
        else:                                                                       # :754-756
            if ftype == "Q3_K_L":
                new_type = "Q4_K"
    elif cat == "ATTENTION_QKV":                                                    # :758-764
        # unreachable: category_is_attn_v(ATTENTION_QKV) is true, so :636 takes every qkv tensor first
        if ftype in ("Q3_K_M", "Q3_K_L", "IQ3_M"):
            new_type = "Q4_K"
        elif ftype == "Q4_K_M":
            new_type = "Q5_K"
        elif ftype == "Q5_K_M":
            new_type = "Q6_K"
    elif cat == "FFN_GATE":                                                         # :765-772
        i_layer, n_layer = layer_info(qs.i_ffn_gate, qs.n_ffn_gate)
        if ftype == "IQ3_XS" and (i_layer >= n_layer // 8 and i_layer < 7 * n_layer // 8):
            new_type = "IQ3_XXS"
        qs.i_ffn_gate += 1
    elif cat == "FFN_UP":                                                           # :773-780
        i_layer, n_layer = layer_info(qs.i_ffn_up, qs.n_ffn_up)
        if ftype == "IQ3_XS" and (i_layer >= n_layer // 8 and i_layer < 7 * n_layer // 8):
            new_type = "IQ3_XXS"
        qs.i_ffn_up += 1
    return new_type


def _fallback(T, qs, ne0, target):
    """src/llama-quant.cpp:451-504 tensor_type_fallback: a type whose block does not divide ne0."""
    qk_k = T["types"][target]["block"]
    ret = target
    if ne0 % qk_k != 0:                                                             # :457
        qs.n_fallback += 1
        if target in T["fallback"]:                                                 # :462-479 (JSON)
            ret = T["fallback"][target]
        elif qk_k <= 32:                                                            # :480-486
            ret = target
        else:                                                                       # :487-488
            raise ValueError("no tensor type fallback is defined for type %s" % target)
        if ne0 % T["types"][ret]["block"] != 0:                                     # :490-500
            ret = "F16"
    return ret


def is_70b(arch, n_layer, n_head=None, n_head_kv=None, n_expert=0, n_layer_nextn=0, T=None):
    """Whether llama.cpp loads the model as LLM_TYPE_70B (src/models/*.cpp load_arch_hparams, JSON type_70b).
    The llama rule needs n_head != n_head_kv; with heads unknown it is taken as GQA (the 70B case)."""
    T = T or tables()
    n = n_layer - n_layer_nextn                                                     # llama_hparams::n_layer()
    for r in T["type_70b"]:
        if r["arch"] != arch or r["n_layer"] != n:
            continue
        if r["not_n_expert"] is not None and n_expert == r["not_n_expert"]:
            continue
        if r["cond"] == "gqa" and n_head is not None and n_head_kv is not None and n_head == n_head_kv:
            continue
        return True
    return False


def tensor_types(tensors, ftype, arch, n_expert=0, n_gqa=1, n_layer=None, n_head=None, n_head_kv=None,
                 n_layer_nextn=0, model_70b=None, has_imatrix=False, T=None, _stats=None):
    """The ggml type llama-quantize (default params, no imatrix unless has_imatrix, no --pure / overrides) gives
    each tensor, in input order. ftype None = the unquantised source file (each tensor's src).

    n_layer: block_count (hparams.n_layer_all); default max blk index + 1. n_gqa: n_head / n_head_kv of layer 0,
    or give n_head and n_head_kv. model_70b: override the LLM_TYPE_70B detection."""
    T = T or tables()
    srcs = [t.get("src") for t in tensors]
    if any(s is None for s in srcs):
        st = source_types(tensors, arch, T=T)
        srcs = [s if s is not None else st[i] for i, s in enumerate(srcs)]
    if ftype is None:
        return srcs
    if ftype not in T["ftypes"]:
        raise ValueError("invalid output file type %s" % ftype)
    default_type = T["ftypes"][ftype]                                               # :1025
    if n_layer is None:
        n_layer = 1 + max([_blk_layer(t["name"]) for t in tensors if _blk_layer(t["name"]) is not None] or [-1])
    if n_head is not None and n_head_kv is not None:
        n_gqa = 0 if n_head_kv == 0 else n_head // n_head_kv                         # llama_hparams::n_gqa
    if model_70b is None:
        model_70b = is_70b(arch, n_layer, n_head, n_head_kv, n_expert, n_layer_nextn, T)
    qs = _QS(n_expert, n_gqa, n_layer, model_70b, has_imatrix)
    names = [t["name"] for t in tensors]
    cats = [tensor_category(n) for n in names]
    # :996-1010 init_quantize_state_counters
    for c in cats:
        if _is_attn_v(c):
            qs.n_attention_wv += 1
        if c == "OUTPUT":
            qs.has_tied_embeddings = False
    order = sorted(range(len(tensors)), key=lambda i: _weight_order_key(names[i]))   # weights_map order
    out = [None] * len(tensors)
    quantized_default = T["types"][default_type]["quantized"]
    for i in order:                                                                 # :1180-1215
        t = tensors[i]
        shape = t["shape"]
        if not allows_quantization(names[i], shape, arch, T):                       # :787-789, :1190-1196
            out[i] = srcs[i]
            continue
        new_type = default_type                                                     # :811
        if quantized_default:                                                       # :814-839
            new_type = _get_type_impl(T, qs, new_type, names[i], shape[0], ftype, cats[i], arch)
            new_type = _fallback(T, qs, shape[0], new_type)
        out[i] = new_type
    if _stats is not None:
        _stats["n_fallback"] = qs.n_fallback
    return out


# ----------------------------------------------------------------------------------------------------------------
# convert_hf_to_gguf.py
# ----------------------------------------------------------------------------------------------------------------

_RX = {}


def _rx(pattern):
    """The JSON regexes ("^...$", ASCII) compiled so that Python matches exactly what JS RegExp does."""
    if pattern not in _RX:
        _RX[pattern] = re.compile(pattern[:-1] + r"\Z" if pattern.endswith("$") else pattern, re.ASCII)
    return _RX[pattern]


def _arch_force_f32(T, name, arch, flags):
    spec = T["converter"]["arch_f32"].get(arch)
    if not spec:
        return False
    for r in spec["rules"]:
        ok = True
        if "prefix" in r:
            ok = ok and any(name.startswith(s) for s in r["prefix"])
        if "contains" in r:
            ok = ok and any(s in name for s in r["contains"])
        if "suffix" in r:
            ok = ok and any(name.endswith(s) for s in r["suffix"])
        if "equal" in r:
            ok = ok and name in r["equal"]
        if "regex" in r:
            ok = ok and any(_rx(x).search(name) for x in r["regex"])
        if "flag" in r:
            ok = ok and bool(flags.get(r["flag"]))
        if ok:
            return True
    return False


def source_types(tensors, arch, outtype="bf16", flags=None, T=None):
    """The type convert_hf_to_gguf.py writes each tensor in for --outtype bf16 / f16 / f32 / q8_0 (not auto:
    auto keeps FP8 checkpoints' weights as F8_E4M3). n_dims is len(shape), as the converter sees the torch
    tensor. conversion/base.py prepare_tensors + _tensor_requires_f32, plus the text-model tensor_force_quant
    overrides (CONVERTER_ARCH_F32). flags: {"lm_head_fp32": bool} (hy_v4's enable_lm_head_fp32)."""
    T = T or tables()
    flags = flags or {}
    C = T["converter"]
    base = {"bf16": "BF16", "f16": "F16", "f32": "F32", "q8_0": "Q8_0"}[outtype]
    out = []
    for t in tensors:
        name, shape = t["name"], t["shape"]
        n_dims = len(shape)
        if _arch_force_f32(T, name, arch, flags):                                   # tensor_force_quant
            out.append("F32")
            continue
        if n_dims <= 1 or name.endswith(C["norm_suffix"]):                          # 1-D and *_norm.weight
            out.append("F32")
            continue
        if any(_rx(r["regex"]).search(name) for r in C["requires_f32"]) or \
                name[-7:] not in C["weight_suffixes"]:                              # _tensor_requires_f32
            out.append("F32")
            continue
        q = base
        if q == "Q8_0" and shape[0] % T["types"]["Q8_0"]["block"] != 0:             # QuantError -> F16
            q = "F16"
        out.append(q)
    return out


# ----------------------------------------------------------------------------------------------------------------
# sizes
# ----------------------------------------------------------------------------------------------------------------

def tensor_bytes(type_, n_elements, ne0, T=None):
    """ggml_nrows * ggml_row_size(type, ne0) (what llama-quantize writes for the tensor, before padding)."""
    T = T or tables()
    t = T["types"][type_]
    if ne0 <= 0 or n_elements % ne0 != 0:
        raise ValueError("n_elements %d is not a multiple of ne0 %d" % (n_elements, ne0))
    if ne0 % t["block"] != 0:
        raise ValueError("ne0 %d is not a multiple of the %s block %d" % (ne0, type_, t["block"]))
    return (n_elements // ne0) * (t["bytes"] * (ne0 // t["block"]))


def _pad(n, align):
    return ((n + align - 1) // align) * align


def breakdown(tensors, ftype, arch, header_bytes=0, **ctx):
    """Bytes of the GGUF: per tensor, per type and per kind (tensor category, "_exps" for experts)."""
    T = ctx.pop("T", None) or tables()
    stats = {}
    types = tensor_types(tensors, ftype, arch, T=T, _stats=stats, **ctx)
    align = T["alignment"]
    by_type, by_kind, rows = {}, {}, []
    data = 0
    for t, ty in zip(tensors, types):
        n = _n_elements(t["shape"])
        b = _pad(tensor_bytes(ty, n, t["shape"][0], T), align)
        data += b
        kind = tensor_category(t["name"]).lower() + ("_exps" if _is_expert(t["name"]) else "")
        for d, k in ((by_type, ty), (by_kind, kind)):
            e = d.setdefault(k, {"tensors": 0, "elements": 0, "bytes": 0})
            e["tensors"] += 1
            e["elements"] += n
            e["bytes"] += b
        rows.append({"name": t["name"], "type": ty, "bytes": b})
    return {"ftype": ftype, "header": header_bytes, "data": data, "total": header_bytes + data,
            "n_fallback": stats.get("n_fallback", 0), "by_type": by_type, "by_kind": by_kind, "tensors": rows}


def gguf_bytes(tensors, ftype, arch, header_bytes=0, **ctx):
    """header + the tensor data, each tensor padded to the alignment (src/llama-quant.cpp:1437)."""
    return breakdown(tensors, ftype, arch, header_bytes, **ctx)["total"]


# ----------------------------------------------------------------------------------------------------------------
# the header
# ----------------------------------------------------------------------------------------------------------------

# GGUF value type sizes (ggml/src/gguf.cpp writer): scalars as is, a string is u64 length + bytes, an array is
# u32 element type + u64 count + elements; every KV is key string + u32 type + value.
_SCALAR = {"u8": 1, "i8": 1, "bool": 1, "u16": 2, "i16": 2, "u32": 4, "i32": 4, "f32": 4, "u64": 8, "i64": 8,
           "f64": 8}


def _kv_bytes(key, kind, value):
    n = 8 + len(key.encode()) + 4
    if kind == "str":
        return n + 8 + value
    if kind.startswith("arr:"):
        el = kind[4:]
        count, per = value
        return n + 4 + 8 + (count * (8 + per) if el == "str" else count * _SCALAR[el])
    return n + _SCALAR[kind]


def header_kvs(vocab_size, arch="llama", tokenizer="gpt2", avg_token_bytes=7.0, n_merges=None,
               avg_merge_bytes=None, chat_template_bytes=0, name_bytes=16, moe=False, extra_kv=None):
    """The KV list of a typical converted + quantised decoder LM, as (key, kind, value) with value = string length
    (estimated), a scalar placeholder, or (count, average string length) for arrays."""
    a = arch
    kvs = [("general.architecture", "str", len(arch.encode())), ("general.type", "str", 5),
           ("general.name", "str", name_bytes), ("general.basename", "str", name_bytes),
           ("general.size_label", "str", 3), ("general.file_type", "u32", 0),
           ("general.quantization_version", "u32", 0),
           (a + ".block_count", "u32", 0), (a + ".context_length", "u32", 0), (a + ".embedding_length", "u32", 0),
           (a + ".feed_forward_length", "u32", 0), (a + ".attention.head_count", "u32", 0),
           (a + ".attention.head_count_kv", "u32", 0), (a + ".rope.freq_base", "f32", 0),
           (a + ".attention.layer_norm_rms_epsilon", "f32", 0), (a + ".attention.key_length", "u32", 0),
           (a + ".attention.value_length", "u32", 0), (a + ".vocab_size", "u32", 0),
           (a + ".rope.dimension_count", "u32", 0)]
    if moe:
        kvs += [(a + ".expert_count", "u32", 0), (a + ".expert_used_count", "u32", 0)]
    kvs += [("tokenizer.ggml.model", "str", len(tokenizer.encode())), ("tokenizer.ggml.pre", "str", 8),
            ("tokenizer.ggml.tokens", "arr:str", (vocab_size, avg_token_bytes)),
            ("tokenizer.ggml.token_type", "arr:i32", (vocab_size, 0))]
    if tokenizer == "gpt2":
        nm = vocab_size - 256 if n_merges is None else n_merges
        per = avg_token_bytes + 1 if avg_merge_bytes is None else avg_merge_bytes
        kvs.append(("tokenizer.ggml.merges", "arr:str", (nm, per)))
    else:
        kvs.append(("tokenizer.ggml.scores", "arr:f32", (vocab_size, 0)))
    kvs += [("tokenizer.ggml.bos_token_id", "u32", 0), ("tokenizer.ggml.eos_token_id", "u32", 0),
            ("tokenizer.ggml.padding_token_id", "u32", 0), ("tokenizer.ggml.add_bos_token", "bool", 0)]
    if chat_template_bytes:
        kvs.append(("tokenizer.chat_template", "str", chat_template_bytes))
    kvs += list(extra_kv or [])
    return kvs


def gguf_header_bytes(vocab_size, n_tensors=None, tensors=None, avg_name_bytes=24, arch="llama", tokenizer="gpt2",
                      avg_token_bytes=7.0, n_merges=None, avg_merge_bytes=None, chat_template_bytes=0,
                      name_bytes=16, moe=False, extra_kv=None, alignment=None, T=None):
    """A first-principles estimate of the GGUF metadata size (ggml/src/gguf.cpp gguf_write_out): magic + version
    + n_tensors + n_kv (24 bytes), the KVs (header_kvs), the tensor infos (name string + u32 n_dims + u64 per dim
    + u32 type + u64 offset), padded to the alignment.

    Exact: the fixed header, the key strings, scalar values, the token_type (i32) / scores (f32) arrays, and the
    tensor infos when `tensors` is given. Estimated (to calibrate on real files): avg_token_bytes (tokens are
    stored in the tokenizer's own form, e.g. GPT-2 byte-to-unicode, where a non-ASCII byte takes 2 UTF-8 bytes),
    n_merges (default vocab_size - 256) and avg_merge_bytes (default avg_token_bytes + 1: a merge "a b" is its
    result plus a space), chat_template_bytes, name_bytes (general.name), tokenizer.ggml.pre (8 bytes), and
    which optional KVs exist. Without `tensors`, each tensor info is taken as avg_name_bytes of name and 2 dims."""
    T = T or tables()
    align = T["alignment"] if alignment is None else alignment
    kvs = header_kvs(vocab_size, arch, tokenizer, avg_token_bytes, n_merges, avg_merge_bytes,
                     chat_template_bytes, name_bytes, moe, extra_kv)
    total = 4 + 4 + 8 + 8
    for k, kind, v in kvs:
        total += _kv_bytes(k, kind, v)
    if tensors is not None:
        for t in tensors:
            total += 8 + len(t["name"].encode()) + 4 + 8 * _n_dims(t["shape"]) + 4 + 8
    else:
        total += (n_tensors or 0) * (8 + avg_name_bytes + 4 + 8 * 2 + 4 + 8)
    total = int(math.ceil(total))
    return _pad(total, align)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=SRC_DIR)
    ap.add_argument("--latest", action="store_true", help="pin to the current master instead of PINNED")
    args = ap.parse_args()
    sha = latest_commit() if args.latest else PINNED
    commit_file = os.path.join(args.src, "COMMIT")
    if os.path.exists(commit_file) and open(commit_file).read().strip() != sha:
        raise SystemExit("%s holds another commit; move it away first" % args.src)
    fetch(args.src, sha)                         # downloads only the files not already there
    t = extract(args.src)
    write(t)
    print("wrote %s (commit %s, %d types, %d ftypes, %d allow rules)" % (
        DATA, t["commit"], len(t["types"]), len(t["ftypes"]), len(t["allow_rules"])))


if __name__ == "__main__":
    main()
