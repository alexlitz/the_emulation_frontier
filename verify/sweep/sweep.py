"""Code-diff sweep: every model_type in the snapshot -> its forward-pass code -> its parent -> the diff hunks.

    python verify/sweep/fetch_remote.py             # the snapshot's Hub remote code, as text
    python verify/sweep/fetch_old_transformers.py   # transformers 4.28 - 4.55 model sources, as text (reference pool)
    python verify/sweep/sweep.py                    # -> inventory_raw.json, hunks_raw.jsonl (not committed)
    python verify/sweep/coverage.py                 # -> coverage.json / coverage.md
    python verify/sweep/build_report.py             # rules + reviews/*.jsonl -> hunks.jsonl, inventory.json,
                                                    #    findings.json, SWEEP.md (exits 1 on an unclassified hunk)
    python verify/sweep/make_batches.py             # unclassified hunks -> batches/ for hand review

Code is READ AS TEXT (ast.parse), never imported or run: the transformers 5.18 tree under
~/.cache/upstream_refs/transformers-5.18.0, the installed transformers 4.57 tree, and the Hub remote code fetched by
fetch_remote.py.

Parent resolution
  * a transformers type with modular_<type>.py: the modular file IS the diff. Each class's bases are resolved through
    its imports to the parent class in the parent's (generated, complete) modeling file; every member the class
    defines is a hunk against the parent's member (or a new member / new class / new function).
  * a transformers type without a modular file, and every Hub remote-code file: each class is matched to the most
    similar class of the same role in the reference pool (normalised source, token similarity), and diffed against
    it member by member; module functions are diffed against the same-named function of the matched module.
  * config classes: one hunk per default that differs from the parent config class, plus the config's own logic.

Normalisation before diffing: ast.unparse (comments, formatting and quoting gone), docstrings dropped, the family
prefix of class names (Qwen3Next, Llama, ...) replaced by PFX. A unified diff (1 line of context) is cut into its @@
hunks; a hunk's id is the hash of (role, member, its +/- lines), so the same edit in many files is one hunk.
"""
import ast
import collections
import difflib
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

T5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0/transformers")
T4 = None
for _p in sys.path + glob.glob(os.path.expanduser("~/.pyenv/versions/3.12.6/lib/python3.12/site-packages")):
    if os.path.isdir(os.path.join(_p, "transformers", "models", "llama")) and "upstream_refs" not in _p:
        T4 = os.path.join(_p, "transformers")
        break
REMOTE_INDEX = os.path.expanduser("~/.cache/code_sweep/remote_index.json")
HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")

TF_OLD = os.path.expanduser("~/.cache/upstream_refs/tf_old")     # older releases (text only, fetched by
OLD_VERSIONS = sorted((d for d in os.listdir(TF_OLD) if os.path.isdir(os.path.join(TF_OLD, d))),  # get_old_tf.py)
                      key=lambda v: tuple(int(x) for x in v.split(".")), reverse=True) if os.path.isdir(TF_OLD) else []
# the no-code hunt (nocode_sources.json): code for snapshot types with no transformers module and no auto_map, fetched
# as text under NOCODE (Hub sibling repos, GitHub repos, inference engines) and the transformers main branch (modules
# added after 5.18)
NOCODE = os.environ.get("NOCODE_CACHE", os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "nocode"))
TMAIN = os.path.join(NOCODE, "_repos", "transformers-main", "src", "transformers")
NOCODE_SOURCES = os.path.join(HERE, "nocode_sources.json")
TAG = os.environ.get("SWEEP_TAG", "")     # a partial run (sweep.py <variant substrings>) writes *_raw.*<TAG>
ROOTS = {"tf5": T5, "tf4": T4, "hub": HF_HUB,
         **{"tf" + v: os.path.join(TF_OLD, v, "transformers") for v in OLD_VERSIONS},
         "tfmain": TMAIN, "ext": NOCODE}      # (tfmain before ext: TMAIN lies under NOCODE)


def rel(path):
    for k, r in ROOTS.items():
        if r and path.startswith(r + "/"):
            return f"{k}:{path[len(r) + 1:]}"
    return path


def short_hub(path):
    """models--org--name/snapshots/<sha>/f.py -> org/name/f.py"""
    m = re.search(r"models--([^/]+)--([^/]+)/snapshots/[^/]+/(.+)$", path)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else path


# ----------------------------------------------------------------------------------------------- transformers maps
def _literal_assign(path, name):
    tree = ast.parse(open(path).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            v = n.value
            if isinstance(v, ast.Call):           # OrderedDict([...])
                v = v.args[0]
            return dict(ast.literal_eval(v))
    return {}


def tf_maps(root):
    """model_type -> module dir (transformers' own model_type_to_module_name), read as text."""
    am = os.path.join(root, "models/auto/auto_mappings.py")
    ca = os.path.join(root, "models/auto/configuration_auto.py")
    src = am if os.path.exists(am) else ca
    names = _literal_assign(src, "CONFIG_MAPPING_NAMES")
    special = _literal_assign(src, "SPECIAL_MODEL_TYPE_TO_MODULE_NAME")
    if src != ca:
        special.update(_literal_assign(ca, "SPECIAL_MODEL_TYPE_TO_MODULE_NAME") or {})
        # configuration_auto.py may .update() it with a literal dict
        tree = ast.parse(open(ca).read())
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update"
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "SPECIAL_MODEL_TYPE_TO_MODULE_NAME"):
                try:
                    special.update(ast.literal_eval(n.args[0]))
                except Exception:
                    pass
    out = {}
    for mt in names:
        mod = special.get(mt, mt).replace("-", "_")
        for cand in (mod, "deprecated/" + mod):
            if os.path.isdir(os.path.join(root, "models", cand)):
                out[mt] = cand
                break
    return out, names


# ------------------------------------------------------------------------------------------------ normalisation
SUFFIXES = ("PreTrainedModel", "ForCausalLM", "ForConditionalGeneration", "LMHeadModel", "DecoderLayer", "Attention",
            "SdpaAttention", "FlashAttention2", "MLP", "RMSNorm", "LayerNorm", "RotaryEmbedding", "Model", "Config",
            "SparseMoeBlock", "MoE", "Experts", "Block", "ForSequenceClassification", "ForTokenClassification",
            "TextModel", "TextConfig")


def strip_docstrings(node):
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n.body = b[1:] or [ast.Pass()]
    return node


def prefixes_of(class_names):
    c = collections.Counter()
    for n in class_names:
        for s in SUFFIXES:
            if n.endswith(s) and len(n) > len(s):
                c[n[: -len(s)]] += 1
                break
    return [p for p, k in c.items() if k >= 2 and len(p) >= 2] or [p for p, _ in c.most_common(1)]


def prefix_re(prefs):
    if not prefs:
        return None
    alt = "|".join(re.escape(p) for p in sorted(set(prefs), key=len, reverse=True))
    return re.compile(r"\b(?:" + alt + r")(?=[A-Z_])")


def norm_text(node, pre):
    import copy
    n = strip_docstrings(copy.deepcopy(node))
    try:
        s = ast.unparse(n)
    except Exception:
        s = ""
    if pre is not None:
        s = pre.sub("PFX", s)
    return s


# --------------------------------------------------------------------------------------------- module parsing
class Mod:
    """A parsed source file: classes {name: ClassDef}, functions, imports {local name: (module path, name)}."""
    _cache = {}

    def __init__(self, path):
        self.path = path
        src = open(path, encoding="utf-8", errors="replace").read()
        try:
            self.tree = ast.parse(src)
        except SyntaxError:
            self.tree = ast.Module(body=[], type_ignores=[])
        self.classes, self.funcs, self.imports, self.assigns = {}, {}, {}, {}
        for n in self.tree.body:
            if isinstance(n, ast.ClassDef):
                self.classes[n.name] = n
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[n.name] = n
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    self.imports[a.asname or a.name] = ("." * n.level + (n.module or ""), a.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name):
                        self.assigns[t.id] = n
        self.pre = prefix_re(prefixes_of(list(self.classes)))

    @classmethod
    def get(cls, path):
        if path not in cls._cache:
            cls._cache[path] = Mod(path)
        return cls._cache[path]

    def resolve(self, modname):
        """A relative import's file path."""
        if not modname.startswith("."):
            if modname.startswith("transformers."):
                base = T5 if self.path.startswith(T5) else TMAIN if self.path.startswith(TMAIN) else T4
                p = os.path.join(base, *modname.split(".")[1:]) + ".py"
                return p if os.path.exists(p) else None
            if self.path.startswith(NOCODE + "/") and not self.path.startswith(TMAIN + "/"):
                # no-code hunt copies flatten a package (fla.models.modeling_utils -> modeling_utils.py beside it)
                p = os.path.join(os.path.dirname(self.path), modname.split(".")[-1] + ".py")
                return p if os.path.exists(p) else None
            return None
        lvl = len(modname) - len(modname.lstrip("."))
        d = os.path.dirname(self.path)
        for _ in range(lvl - 1):
            d = os.path.dirname(d)
        p = os.path.join(d, *modname.lstrip(".").split(".")) + ".py"
        return p if os.path.exists(p) else None


def members(cls_node):
    """{member name: node} for a class body (methods and class attributes)."""
    out = {}
    for n in cls_node.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = n
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out[n.target.id] = n
    return out


def base_names(cls_node):
    out = []
    for b in cls_node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(ast.unparse(b))
        elif isinstance(b, ast.Subscript):
            out.append(ast.unparse(b.value))
    return out


# ---------------------------------------------------------------------------------------------------- roles
VISION = re.compile(r"Vision|Visual|ViT|Vit[A-Z]|Siglip|SigLip|Clip|CLIP|Image|Patch|Audio|Speech|Whisper|Projector|"
                    r"Resampler|Connector|Perceiver|QFormer|Qformer|Video|Pixel|Navit|NaViT|Mel|Wav|Conformer|Encoder"
                    r"(?!.*Decoder)|VQ|Vq|Tokenizer|Processor|Ocr|OCR|Pooler|Mmproj|MultiModal|Multimodal|Merger")
ROLE_RULES = [
    ("output", re.compile(r"Output(WithPast)?$|OutputWith|Outputs?$")),
    ("cache", re.compile(r"Cache")),
    ("pretrained", re.compile(r"PreTrainedModel$|PretrainedModel$")),
    ("config", re.compile(r"Config$")),
    ("other_head", re.compile(r"For(Sequence|Token)Classification|ForQuestionAnswering|ForMaskedLM|DoubleHeads|"
                              r"ForMultipleChoice|ForPreTraining|RewardModel|ValueHead|ForScore")),
    ("lm_head", re.compile(r"ForCausalLM|LMHeadModel|ForConditionalGeneration|LMHead|ForDiffusion|ForMaskedDiffusion"
                           r"|ForDLM|LmHead")),
    ("rotary", re.compile(r"Rotary|RoPE|Rope|Alibi|ALiBi|Positional|PositionEmbedding|PosEmb|Sinusoid")),
    ("norm", re.compile(r"Norm")),
    ("mixer", re.compile(r"Mamba|Mixer|SSM|Ssm|DeltaNet|GatedDelta|Delta|ShortConv|Conv|RWKV|Rwkv|Lightning|Linear"
                         r"Attention|LinearAttn|Recurrent|RGLRU|Rglru|Hyena|Retention|GLA|Kda|KDA|TimeMix|ChannelMix|"
                         r"Lru|LRU")),
    ("attention", re.compile(r"Attention|Attn|Indexer|Compressor")),
    ("moe", re.compile(r"Expert|MoE|Moe|Router|Gate|TopK|Topk|Gating")),
    ("mlp", re.compile(r"MLP|Mlp|FeedForward|FFN|Ffn|FF$|SwiGLU|GLU")),
    ("decoder_layer", re.compile(r"DecoderLayer|Layer$|Block$|Layers$|HybridLayer|Decoder$")),
    ("embedding", re.compile(r"Embedding|Embed")),
    ("model", re.compile(r"Model$|Transformer$|TextModel$|LM$")),
]
ROOT_MODULES = {"llama", "gpt2"}    # the two reference implementations every other family is diffed toward
GROUP = {"attention": "mix", "mixer": "mix", "mlp": "ffn", "moe": "ffn"}


def role_of(name):
    if VISION.search(name) and not re.search(r"Text|Language|LM|Decoder|ForCausalLM", name):
        return "vision"
    for r, rx in ROLE_RULES:
        if rx.search(name):
            return r
    return "other"


# ----------------------------------------------------------------------------------------------- diffing
TOK = re.compile(r"\w+|[^\w\s]")


def sim(a, b):
    ta, tb = TOK.findall(a), TOK.findall(b)
    if not ta or not tb:
        return 0.0
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    return sm.ratio()


def quick(a, b):
    sa, sb = set(TOK.findall(a)), set(TOK.findall(b))
    return len(sa & sb) / max(1, len(sa | sb))


def split_hunks(parent_txt, child_txt):
    d = list(difflib.unified_diff(parent_txt.splitlines(), child_txt.splitlines(), lineterm="", n=1))[2:]
    hunks, cur = [], None
    for line in d:
        if line.startswith("@@"):
            cur = [line]
            hunks.append(cur)
        elif cur is not None:
            cur.append(line)
    return ["\n".join(h) for h in hunks]


def hid(*parts):
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:12]


def change_lines(h):
    return "\n".join(l.strip() for l in h.splitlines()[1:] if l[:1] in "+-")


# ------------------------------------------------------------------------------------------------ config defaults
def config_defaults(cls_node):
    """{key: default} of a config class: tf5 dataclass attributes, or the __init__ keyword defaults."""
    out = {}
    for n in cls_node.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            out[n.target.id] = _lit(n.value)
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ("model_type", "keys_to_ignore_at_inference", "base_model_tp_plan",
                                   "base_model_pp_plan", "sub_configs", "attribute_map", "base_config_key",
                                   "default_theta", "base_model_ep_plan"):
                continue
            out[n.targets[0].id] = _lit(n.value)
        elif isinstance(n, ast.FunctionDef) and n.name == "__init__":
            a = n.args
            pos = a.args[len(a.args) - len(a.defaults):]
            for arg, dv in list(zip(pos, a.defaults)) + list(zip(a.kwonlyargs, a.kw_defaults)):
                if dv is not None:
                    out[arg.arg] = _lit(dv)
    return out


def _lit(v):
    try:
        x = ast.literal_eval(v)
        json.dumps(x)
        return x
    except Exception:
        return "<expr " + ast.unparse(v)[:80] + ">"


def config_attribute_map(cls_node):
    for n in cls_node.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "attribute_map" for t in n.targets):
            try:
                return ast.literal_eval(n.value)
            except Exception:
                return {}
    return {}


TRIVIAL_INIT = re.compile(r"^self\.(\w+) = (\w+)$")


def config_logic(cls_node):
    """The config class's statements beyond `self.k = k` and the super() call (its code-side logic)."""
    out = []
    for n in cls_node.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "__post_init__", "validate_architecture",
                                                         "validate", "_rope_scaling_validation"):
            for s in n.body:
                t = ast.unparse(strip_docstrings(s))
                m = TRIVIAL_INIT.match(t)
                if m and m.group(1) == m.group(2):
                    continue
                if "super().__init__(" in t or "super().__post_init__(" in t:
                    continue
                out.append(f"[{n.name}] " + t.replace("\n", " ; "))
        elif isinstance(n, ast.FunctionDef) and n.name not in ("__init__",):
            out.append(f"[{n.name}] " + ast.unparse(strip_docstrings(n)).replace("\n", " ; ")[:2000])
    return out


# ------------------------------------------------------------------------------------------- class resolution
def find_class(mod, name, depth=0):
    """(Mod, ClassDef) for a class name visible in mod (defined there or imported from a resolvable module)."""
    if depth > 8:
        return None
    if name in mod.classes:
        return mod, mod.classes[name]
    if name in mod.imports:
        src, orig = mod.imports[name]
        p = mod.resolve(src)
        if p:
            return find_class(Mod.get(p), orig, depth + 1)
    return None


def find_member(mod, cls, member, depth=0):
    """The member as the class resolves it: own body, else the bases (left to right, depth first)."""
    if depth > 12:
        return None
    m = members(cls).get(member)
    if m is not None:
        return mod, cls, m
    for b in base_names(cls):
        r = find_class(mod, b)
        if r:
            got = find_member(r[0], r[1], member, depth + 1)
            if got:
                return got
    return None


MODEL_DIR = re.compile(r"/models/(?:deprecated/)?([^/]+)/")


def module_of(path):
    m = MODEL_DIR.search(path)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------------------------- the pool
FORWARD_ROLES = ("attention", "mixer", "mlp", "moe", "decoder_layer", "model", "norm", "rotary", "embedding",
                 "lm_head", "other")
PREFERRED = ["llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "gpt_neox", "mamba", "mamba2",
             "mixtral", "qwen2_moe", "qwen3_moe", "deepseek_v2", "deepseek_v3", "phi3", "gpt2", "falcon", "bloom",
             "opt", "gptj", "phi", "qwen3_next", "granite", "olmo2", "cohere", "starcoder2", "gpt_bigcode"]


class Pool:
    """Reference classes / functions of transformers 5.18 and 4.57 decoder modules, deduplicated by normalised text
    (a generated copy keeps the PREFERRED module's name)."""

    def __init__(self, roots_only_tf5=None):
        self.cls = collections.defaultdict(list)    # group -> [(text, tokens, ref, path, name)]
        self.funcs = collections.defaultdict(list)  # name -> [(text, ref)]
        self.configs = {}                           # module ref -> (path, ClassDef)
        seen = {}
        files = []
        for tag, root in [("tf5", T5), ("tf4", T4)] + [("tf" + v, ROOTS["tf" + v]) for v in OLD_VERSIONS]:
            for p in sorted(glob.glob(os.path.join(root, "models", "*", "modeling_*.py"))
                            + glob.glob(os.path.join(root, "models", "deprecated", "*", "modeling_*.py"))):
                if "_tf_" in p or "_flax_" in p or os.path.basename(p).startswith(("modeling_tf", "modeling_flax")):
                    continue
                files.append((tag, p))
        rank = {m: i for i, m in enumerate(PREFERRED)}
        order = {t: i for i, t in enumerate(["tf5", "tf4"] + ["tf" + v for v in OLD_VERSIONS])}
        files.sort(key=lambda tp: (rank.get(module_of(tp[1]), 999), order[tp[0]], tp[1]))
        for tag, p in files:
            mod = Mod(p) if tag.startswith("tf4.") else Mod.get(p)    # old releases: not cached (memory)
            if not any(role_of(c) == "lm_head" for c in mod.classes):
                continue
            mname = module_of(p)
            for name, node in mod.classes.items():
                r = role_of(name)
                if r not in FORWARD_ROLES:
                    continue
                t = norm_text(node, mod.pre)
                if t in seen:
                    continue
                ref = f"{tag}:{mname}.{name}"
                seen[t] = ref
                self.cls[GROUP.get(r, r)].append((t, set(TOK.findall(t)), ref, p, name))
            for name, node in mod.funcs.items():
                t = norm_text(node, mod.pre)
                if ("f", t) not in seen:
                    seen[("f", t)] = 1
                    self.funcs[name].append((t, f"{tag}:{mname}.{name}", p))

    @classmethod
    def cached(cls):
        """The pool, pickled under ~/.cache/code_sweep (rebuilt when the reference trees change)."""
        import pickle
        key = hashlib.sha1(repr([T5, T4] + OLD_VERSIONS + [PREFERRED, sorted(FORWARD_ROLES)]).encode()).hexdigest()[:10]
        path = os.path.expanduser(f"~/.cache/code_sweep/pool-{key}.pkl")
        if os.path.exists(path):
            return pickle.load(open(path, "rb"))
        pool = cls()
        pickle.dump(pool, open(path, "wb"))
        return pool

    def best(self, role, text, exclude=()):
        g = GROUP.get(role, role)
        toks = set(TOK.findall(text))
        cands = sorted(((len(toks & c[1]) / max(1, len(toks | c[1])), c) for c in self.cls.get(g, ())
                        if module_of(c[3]) not in exclude), key=lambda x: -x[0])[:6]
        best = None
        for _, c in cands:
            s = sim(text, c[0])
            if best is None or s > best[0]:
                best = (s, c)
        return best


# ------------------------------------------------------------------------------------------------ hunks
WHOLE_CLASS_ROLES = ("vision", "output", "cache")


class Sweep:
    def __init__(self):
        self.hunks = {}          # id -> record
        self.pool = None

    def emit(self, var, path, cname, role, member, kind, pref, ptxt, ctxt):
        occ = dict(variant=var, file=short_hub(rel(path)), cls=cname, member=member, parent=pref)
        if ptxt is not None and ptxt == ctxt:
            return 0
        if ptxt is None:
            body = "\n".join("+" + l for l in ctxt.splitlines())
            parts = [("@@ new @@\n" + body, change_lines("@@\n" + body))]
        else:
            parts = [(h, change_lines(h)) for h in split_hunks(ptxt, ctxt)]
        n = 0
        for h, ch in parts:
            if not ch:
                continue
            i = hid(role, member if kind != "class" else "", ch)
            rec = self.hunks.setdefault(i, dict(id=i, role=role, member=member, kind=kind, text=h, occ=[]))
            rec["occ"].append(occ)
            n += 1
        return n

    # -- one class against one parent class, member by member ------------------------------------------------
    def diff_class(self, var, mod, cls, pmod, pcls, kind="override", pref=None, own_only=True):
        role = role_of(cls.name)
        if role in WHOLE_CLASS_ROLES:
            return self.emit(var, mod.path, cls.name, role, "<class>", "class", None, None,
                             f"class {cls.name}  # {role} class, not diffed")
        n = 0
        for mname, node in members(cls).items():
            ctxt = norm_text(node, mod.pre)
            ptxt, pr = None, pref
            if pcls is not None:
                got = find_member(pmod, pcls, mname)
                if got:
                    ptxt = norm_text(got[2], got[0].pre)
                    pr = f"{short_hub(rel(got[0].path))}::{got[1].name}.{mname}"
            n += self.emit(var, mod.path, cls.name, role, mname, kind if (ptxt is not None or kind in ("root", "new_class")) else "new_member",
                           pr or "-", ptxt, ctxt)
        return n

    def diff_func(self, var, mod, node, pmods):
        ctxt = norm_text(node, mod.pre)
        for pm in pmods:
            if node.name in pm.funcs:
                ptxt = norm_text(pm.funcs[node.name], pm.pre)
                return self.emit(var, mod.path, "<module>", "function", node.name, "override",
                                 f"{short_hub(rel(pm.path))}::{node.name}", ptxt, ctxt)
        # nearest same-named function anywhere in the pool
        cands = self.pool.funcs.get(node.name, [])
        if cands:
            t, ref, p = max(cands, key=lambda c: sim(ctxt, c[0]))
            return self.emit(var, mod.path, "<module>", "function", node.name, "override", ref, t, ctxt)
        return self.emit(var, mod.path, "<module>", "function", node.name, "new_function", "-", None, ctxt)

    def same_suffix(self, pmod, cname, cpre):
        """The class of pmod playing cname's part (same name after the family prefix)."""
        if pmod is None:
            return None
        t = cpre.sub("PFX", cname) if cpre else cname
        for n in pmod.classes:
            if (pmod.pre.sub("PFX", n) if pmod.pre else n) == t:
                return pmod.classes[n]
        return None

    # -- a config class against the parent config ---------------------------------------------------------------
    def diff_config(self, var, mod, cls, pmod, pcls, models_missing):
        n = 0
        cd, pd = config_defaults(cls), (config_defaults(pcls) if pcls is not None else {})
        pref = f"{short_hub(rel(pmod.path))}::{pcls.name}" if pcls is not None else "-"
        for k in sorted(cd):
            if k in pd and json.dumps(pd[k], sort_keys=True) == json.dumps(cd[k], sort_keys=True):
                continue
            txt = (f"{k}: default {json.dumps(cd[k])[:300]}"
                   + (f" (parent {json.dumps(pd[k])[:300]})" if k in pd else " (new key)"))
            miss = models_missing(k)
            body = f"@@ config default @@\n-{k} = {json.dumps(pd.get(k, '<absent>'))[:300]}\n+{k} = {json.dumps(cd[k])[:300]}"
            i = hid("config", k, json.dumps(cd[k], sort_keys=True), json.dumps(pd.get(k, "<absent>"), sort_keys=True),
                    var)
            rec = self.hunks.setdefault(i, dict(id=i, role="config", member=k, kind="config_default", text=body,
                                                summary=txt, occ=[]))
            rec["occ"].append(dict(variant=var, file=short_hub(rel(mod.path)), cls=cls.name, member=k, parent=pref,
                                   configs_without_key=miss))
            n += 1
        am, pam = config_attribute_map(cls), (config_attribute_map(pcls) if pcls is not None else {})
        for k in sorted(set(am) | set(pam)):
            if am.get(k) != pam.get(k):
                n += self.emit(var, mod.path, cls.name, "config", "attribute_map", "config_rename", pref,
                               f"{k} -> {pam.get(k)}", f"{k} -> {am.get(k)}")
        cl, pl = config_logic(cls), (config_logic(pcls) if pcls is not None else [])
        pre = mod.pre
        cl = [pre.sub("PFX", s) if pre else s for s in cl]
        pl = [pmod.pre.sub("PFX", s) if (pcls is not None and pmod.pre) else s for s in pl]
        n += self.emit(var, mod.path, cls.name, "config", "<logic>", "config_logic", pref,
                       "\n".join(pl) if pcls is not None else None, "\n".join(cl)) if (cl or pl) else 0
        return n

    # -- a transformers module ---------------------------------------------------------------------------------
    def config_class(self, mdir, prefer=None):
        """(Mod, ClassDef) of the module's main config class (prefer: the class name CONFIG_MAPPING_NAMES gives)."""
        best = None
        for p in sorted(glob.glob(os.path.join(mdir, "configuration_*.py"))):
            m = Mod.get(p)
            for n, c in m.classes.items():
                if role_of(n) != "config":
                    continue
                if prefer and n == prefer:
                    return m, c
                if best is None or (m.pre and m.pre.sub("PFX", n) == "PFXConfig"):
                    best = (m, c)
        return best

    def sweep_modular(self, var, mdir, modular, mt_config, missing):
        modm = Mod.get(modular)
        gens = [Mod.get(p) for p in sorted(glob.glob(os.path.join(mdir, "modeling_*.py")))]
        resolved, votes = {}, collections.Counter()
        for n, c in modm.classes.items():
            for b in base_names(c):
                r = find_class(modm, b)
                if r and r[0].path != modm.path and module_of(r[0].path) != module_of(modm.path):
                    resolved[n] = r
                    if role_of(n) in FORWARD_ROLES and r[0].path.endswith(".py") and "/modeling_" in r[0].path:
                        votes[module_of(r[0].path)] += 3 if role_of(n) in ("attention", "decoder_layer") else 1
                    break
                if r and r[0].path == modm.path:
                    resolved[n] = r
                    break
        rank = {m: i for i, m in enumerate(PREFERRED)}
        primary = max(votes, key=lambda m: (votes[m], -rank.get(m, 999))) if votes else None
        pmod = None
        if primary:
            root = T5 if modular.startswith(T5) else TMAIN if modular.startswith(TMAIN) else T4
            cand = glob.glob(os.path.join(root, "models", primary, f"modeling_{primary}.py")) or \
                glob.glob(os.path.join(root, "models", "*", primary, f"modeling_{primary}.py"))
            pmod = Mod.get(cand[0]) if cand else None
        for n, c in modm.classes.items():
            if role_of(n) == "config":
                continue
            r = resolved.get(n)
            if r and r[0].path == modm.path and role_of(r[1].name) != role_of(n):
                r = None              # an in-file base of another role (the PreTrainedModel): diff by role instead
            if r:
                self.diff_class(var, modm, c, r[0], r[1])
                if pmod is not None and r[0].path != pmod.path and r[0].path != modm.path \
                        and role_of(n) in FORWARD_ROLES:
                    P = self.same_suffix(pmod, n, modm.pre)
                    if P is not None:
                        self.diff_class(var, r[0], r[1], pmod, P, kind="rebase")
            else:
                P = self.same_suffix(pmod, n, modm.pre)
                self.diff_class(var, modm, c, pmod, P, kind="override" if P is not None else "new_class")
        for g in gens:
            for n, c in g.classes.items():
                if n in modm.classes or role_of(n) == "config":
                    continue
                P = self.same_suffix(pmod, n, g.pre)
                self.diff_class(var, g, c, pmod, P, kind="dependency" if P is not None else "new_class")
        for f in modm.funcs.values():
            self.diff_func(var, modm, f, [pmod] if pmod else [])
        for g in gens:
            for n, f in g.funcs.items():
                if n not in modm.funcs:
                    self.diff_func(var, g, f, [pmod] if pmod else [])
        # the config, against the primary parent's
        cc = None
        for n, c in modm.classes.items():
            if role_of(n) == "config" and (mt_config is None or n == mt_config):
                cc = (modm, c)
                break
        if cc is None:
            cc = self.config_class(mdir, mt_config)
        if cc:
            pc = None
            for b in base_names(cc[1]):
                r = find_class(cc[0], b)
                if r and role_of(r[1].name) == "config" and r[1].name not in ("PretrainedConfig", "PreTrainedConfig"):
                    pc = r
                    break
            if pc is None and primary:
                pc = self.config_class(os.path.dirname(pmod.path)) if pmod else None
            self.diff_config(var, cc[0], cc[1], pc[0] if pc else None, pc[1] if pc else None, missing)
        return primary

    def sweep_similar(self, var, mods, exclude, missing):
        """Non-modular code: each class against its resolvable base, else the most similar pool class."""
        votes = collections.Counter()
        cfgs = []
        for mod in mods:
            for n, c in mod.classes.items():
                role = role_of(n)
                if role == "config":
                    cfgs.append((mod, c))
                    continue
                if role in WHOLE_CLASS_ROLES:
                    self.diff_class(var, mod, c, None, None)
                    continue
                r = None
                for b in base_names(c):
                    rr = find_class(mod, b)
                    if rr and (("/models/" in rr[0].path and rr[0].path != mod.path) or rr[0].path.startswith(HF_HUB)
                               and rr[0].path != mod.path or (rr[0].path == mod.path and role_of(rr[1].name) == role)):
                        r = rr
                        break
                if r:
                    self.diff_class(var, mod, c, r[0], r[1])
                    if r[0].path != mod.path:
                        votes[r[0].path] += 1
                    continue
                if role in ("pretrained", "other_head"):
                    self.diff_class(var, mod, c, None, None, kind="new_class")
                    continue
                best = self.pool.best(role, norm_text(c, mod.pre), exclude)
                if best and best[0] >= 0.35:
                    pm = Mod.get(best[1][3])
                    self.diff_class(var, mod, c, pm, pm.classes[best[1][4]])
                    votes[best[1][3]] += 1 if role != "other" else 0
                else:
                    self.diff_class(var, mod, c, None, None, kind="new_class")
        rank = {m: i for i, m in enumerate(PREFERRED)}
        ppath = max(votes, key=lambda p: (votes[p], -rank.get(module_of(p), 999))) if votes else None
        pmods = [Mod.get(ppath)] if ppath else []
        for mod in mods:
            for f in mod.funcs.values():
                self.diff_func(var, mod, f, pmods)
        pcfg = self.config_class(os.path.dirname(ppath)) if ppath and "/models/" in ppath else None
        for mod, c in cfgs:
            pc = None
            for b in base_names(c):
                r = find_class(mod, b)
                if r and r[1].name not in ("PretrainedConfig", "PreTrainedConfig"):
                    pc = r
                    break
            pc = pc or pcfg
            self.diff_config(var, mod, c, pc[0] if pc else None, pc[1] if pc else None, missing)
        return module_of(ppath) if ppath and "/models/" in ppath else (short_hub(ppath) if ppath else None)

    def sweep_against(self, var, mods, pmods, missing):
        """Code whose parent is a known file set (nocode_sources.json parent_files: the previous release of the same
        reference code, e.g. DeepSeek-V4.1's inference/model.py against V4's): each class / function against the
        same-named one of the parent files, else (a new class) the most similar pool class of its role."""
        pc = {n: (m, c) for m in pmods for n, c in m.classes.items()}
        for mod in mods:
            for n, c in mod.classes.items():
                role = role_of(n)
                if role in WHOLE_CLASS_ROLES:
                    self.diff_class(var, mod, c, None, None)
                elif n in pc:
                    if role == "config":
                        self.diff_config(var, mod, c, pc[n][0], pc[n][1], missing)
                    else:
                        self.diff_class(var, mod, c, pc[n][0], pc[n][1])
                else:
                    self.diff_class(var, mod, c, None, None, kind="new_class")
            for f in mod.funcs.values():
                if any(f.name in pm.funcs for pm in pmods):
                    self.diff_func(var, mod, f, pmods)
                else:
                    self.emit(var, mod.path, "<module>", "function", f.name, "new_function", "-", None,
                              norm_text(f, mod.pre))
        return short_hub(rel(pmods[0].path)) if pmods else None


# ------------------------------------------------------------------------------------------- reachability
MODEL_AUTO_KEYS = re.compile(r"^AutoModel")


def entry_classes(cfgs):
    """The model classes the configs' auto_map names (AutoModel*; tokenizers / processors are not models)."""
    sys.path.insert(0, HERE)
    from fetch_remote import auto_maps
    out = set()
    for c in cfgs:
        for a in auto_maps(c):
            for k, v in a.items():
                if not MODEL_AUTO_KEYS.match(k):
                    continue
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, str):
                        out.add(x.rsplit(".", 1)[-1])
    return out


def reach(mods, roots):
    """What the entry classes can run: (names, members). Names are classes / functions / module assignments reached
    by reference; members are (class, member) pairs of reached classes that a reached body can call: __init__,
    forward, __call__, Flax setup and class attributes always, any other method only when some reached code accesses an
    attribute of that name (self.f(...), super().f(...), module.f). Vision / audio classes are not entered."""
    defs = {}
    for m in mods:
        for n, c in m.classes.items():
            defs.setdefault(n, c)
        for n, f in m.funcs.items():
            defs.setdefault(n, f)
        for n, a in m.assigns.items():
            defs.setdefault(n, a)
    names, members_r, attrs = set(), set(), set()
    todo = []

    def add_name(n):
        if n in names or n not in defs:
            return
        node = defs[n]
        if isinstance(node, ast.ClassDef) and role_of(n) == "vision":
            return
        names.add(n)
        if isinstance(node, ast.ClassDef):
            for b in base_names(node):
                add_name(b.split(".")[-1])
            for mn, mnode in members(node).items():
                if (mn in ("__init__", "forward", "__call__", "__post_init__", "setup") or mn in attrs
                        or not isinstance(mnode, (ast.FunctionDef, ast.AsyncFunctionDef))):
                    add_member(n, mn, mnode)
        else:
            todo.append(node)

    def add_member(cn, mn, mnode):
        if (cn, mn) not in members_r:
            members_r.add((cn, mn))
            todo.append(mnode)

    for r in roots:
        add_name(r)
    while todo:
        node = todo.pop()
        new_attrs = set()
        for x in ast.walk(node):
            if isinstance(x, ast.Name):
                add_name(x.id)
            elif isinstance(x, ast.Constant) and isinstance(x.value, str) and x.value in defs:
                add_name(x.value)
            elif isinstance(x, ast.Attribute) and x.attr not in attrs:
                attrs.add(x.attr)
                new_attrs.add(x.attr)
                add_name(x.attr) if x.attr in defs and isinstance(defs[x.attr], (ast.FunctionDef,)) else None
        if new_attrs:
            for cn in list(names):
                node2 = defs.get(cn)
                if isinstance(node2, ast.ClassDef):
                    for mn, mnode in members(node2).items():
                        if mn in new_attrs:
                            add_member(cn, mn, mnode)
    return names, members_r, attrs, set(defs)

# ---------------------------------------------------------------------------------------------------- main
def snapshot():
    from graph import model_graph as MG
    models = json.load(open(os.path.join(ROOT, "graph/hub_10k.json")))
    rows = []
    for m in models:
        cfg = m.get("config") or {}
        outer = str(cfg.get("model_type") or "") or "<none>"
        lm = MG._lm_unwrap(cfg)
        inner = str(lm.get("model_type") or "") if isinstance(lm, dict) else ""
        try:
            placed = MG.classify(cfg) is not None
        except Exception:
            placed = False
        rows.append(dict(id=m["id"], outer=outer, inner=inner or outer, placed=placed, cfg=cfg,
                         lm=lm if isinstance(lm, dict) else {}))
    return rows


def remote_variants(rows):
    """model id -> dict(variant, paths, missing) for every model whose auto_map names code."""
    sys.path.insert(0, HERE)
    from fetch_remote import auto_maps
    idx = json.load(open(REMOTE_INDEX))
    out = {}
    for r in rows:
        files = set()
        for a in auto_maps(r["cfg"]):
            for v in a.values():
                for x in (v if isinstance(v, list) else [v]):
                    if not isinstance(x, str):
                        continue
                    repo = r["id"]
                    if "--" in x:
                        repo, x = x.split("--", 1)
                    files.add((repo, x.rsplit(".", 1)[0].replace(".", "/") + ".py"))
        if not files:
            continue
        repos = {rp for rp, _ in files}
        have = sorted({p for rp in repos for p in idx.get(rp, {}).values() if p})
        missing = sorted(f"{rp}/{f}" for rp, f in files if not idx.get(rp, {}).get(f))
        h = hashlib.sha1()
        for p in have:
            h.update(open(p, "rb").read())
        out[r["id"]] = dict(variant=("hub:" + h.hexdigest()[:10]) if have else None, paths=have, missing=missing)
    return out


def main():
    rows = snapshot()
    t5map, t5names = tf_maps(T5)
    t4map, t4names = tf_maps(T4)
    rv = remote_variants(rows)
    print(len(rows), "models;", len(rv), "with remote code", flush=True)

    types = collections.defaultdict(lambda: dict(models=set(), outer=0, inner=0, placed=0))
    for r in rows:
        for t, k in ((r["outer"], "outer"), (r["inner"], "inner")):
            e = types[t]
            if r["id"] not in e["models"]:
                e["models"].add(r["id"])
                e["placed"] += r["placed"]
            e[k] += 1

    var_models = collections.defaultdict(set)
    var_info = {}
    for t in types:
        if t in t5map:
            v = "tf5:" + t5map[t]
        elif t in t4map:
            v = "tf4:" + t4map[t]
        else:
            continue
        var_models[v] |= types[t]["models"]
        var_info[v] = dict(root=T5 if v.startswith("tf5") else T4, module=v.split(":", 1)[1])
    for mid, x in rv.items():
        if x["variant"]:
            var_models[x["variant"]].add(mid)
            var_info.setdefault(x["variant"], dict(paths=x["paths"]))
    byid = {r["id"]: r for r in rows}
    # the no-code hunt: code found elsewhere for types with no module and no auto_map
    nc = json.load(open(NOCODE_SOURCES)) if os.path.exists(NOCODE_SOURCES) else {}
    tmmap, tmnames = tf_maps(TMAIN) if os.path.isdir(TMAIN) else ({}, {})
    nc_code = collections.defaultdict(list)       # type -> inventory code entries
    for t, e in sorted(nc.items()):
        if t not in types:
            continue
        e = nc[e["same_as"]] if e.get("same_as") else e      # an outer / text-config pair: one entry
        for sv in e.get("sweep", []):
            v = sv["variant"]
            var_models[v] |= types[t]["models"]
            if v.startswith(("tfmain:", "tf5:")):
                var_info.setdefault(v, dict(root=TMAIN if v.startswith("tfmain:") else T5, module=v.split(":", 1)[1]))
            elif v.startswith("ext:"):
                var_info.setdefault(v, dict(paths=[os.path.join(NOCODE, f) for f in sv["files"]],
                                            parent_paths=[os.path.join(NOCODE, f) for f in sv.get("parent_files", [])],
                                            entry=sv.get("entry", []), manual=sv.get("manual", False)))
            nc_code[t].append(dict(variant=v, source=sv.get("source", ""), wrapper=sv.get("wrapper", False)))

    sw = Sweep()
    sw.pool = Pool.cached()
    print("pool:", {k: len(v) for k, v in sw.pool.cls.items()}, flush=True)
    var_parent, var_reach = {}, {}
    only = sys.argv[1:]
    for v in sorted(var_info):
        if only and not any(o in v for o in only):
            continue
        info = var_info[v]
        mids = var_models[v]

        def missing(k, mids=mids):
            return sum(1 for m in mids if k not in byid[m]["cfg"] and k not in byid[m]["lm"])

        if v.startswith("tf"):
            tmap, names = ((t5map, t5names) if v.startswith("tf5:") else (tmmap, tmnames) if v.startswith("tfmain:")
                           else (t4map, t4names))
            mdir = os.path.join(info["root"], "models", info["module"])
            mods = sorted(glob.glob(os.path.join(mdir, "modular_*.py")))
            mt_cfg = next((names[t] for t in sorted(types) if tmap.get(t) == info["module"] and t in names), None)
            if info["module"] in ROOT_MODULES:
                files = sorted(glob.glob(os.path.join(mdir, "modeling_*.py")))
                for p in files:
                    m = Mod.get(p)
                    for n, c in m.classes.items():
                        sw.diff_class(v, m, c, None, None, kind="root")
                    for f in m.funcs.values():
                        sw.emit(v, p, "<module>", "function", f.name, "root", "-", None, norm_text(f, m.pre))
                cc = sw.config_class(mdir)
                if cc:
                    sw.diff_config(v, cc[0], cc[1], None, None, missing)
                var_parent[v] = dict(parent=None, how="root (the reference implementation)", files=[rel(p) for p in files])
            elif mods:
                files = mods + sorted(glob.glob(os.path.join(mdir, "modeling_*.py")))
                var_parent[v] = dict(parent=sw.sweep_modular(v, mdir, mods[0], mt_cfg, missing), how="modular",
                                     files=[rel(p) for p in files])
            else:
                files = sorted(glob.glob(os.path.join(mdir, "modeling_*.py"))
                               + glob.glob(os.path.join(mdir, "configuration_*.py")))
                files = [p for p in files if "modeling_tf_" not in p and "modeling_flax_" not in p]
                var_parent[v] = dict(parent=sw.sweep_similar(v, [Mod.get(p) for p in files], {info["module"]}, missing),
                                     how="similarity", files=[rel(p) for p in files])
        elif v.startswith("ext:") and info.get("manual"):
            var_parent[v] = dict(parent=None, how="manual (non-Python or engine code: hunks written by hand in "
                                 "nocode_manual.jsonl)", files=[rel(p) for p in info["paths"]])
        elif v.startswith("ext:") and info.get("parent_paths"):
            mods = [Mod.get(p) for p in info["paths"]]
            var_parent[v] = dict(parent=sw.sweep_against(v, mods, [Mod.get(p) for p in info["parent_paths"]], missing),
                                 how="against the named parent files", files=[rel(p) for p in info["paths"]],
                                 parent_files=[rel(p) for p in info["parent_paths"]])
        else:
            mods = [Mod.get(p) for p in info["paths"]]
            var_parent[v] = dict(parent=sw.sweep_similar(v, mods, set(), missing), how="similarity",
                                 files=[short_hub(rel(p)) for p in info["paths"]])
        # what the causal-LM entry reaches (a class no entry reaches never runs in the text forward)
        if v.startswith("tf"):
            rmods = [Mod.get(rp) for rp in sorted(glob.glob(os.path.join(mdir, "modeling_*.py")))
                     + sorted(glob.glob(os.path.join(mdir, "modular_*.py")))]
            roots = {n for m in rmods for n in m.classes if role_of(n) in ("lm_head", "model")}
        elif info.get("manual"):
            rmods, roots = [], set()
        else:
            rmods = [Mod.get(rp) for rp in info["paths"]]
            roots = set(info["entry"]) if info.get("entry") else entry_classes([byid[m]["cfg"] for m in mids])
            if not any(r in m.classes for m in rmods for r in roots):
                roots = {n for m in rmods for n in m.classes if role_of(n) in ("lm_head", "model")}
        var_reach[v] = reach(rmods, roots)
        var_parent[v]["entry"] = sorted(r for r in roots if any(r in m.classes for m in rmods))
        print(v, len(mids), var_parent[v]["parent"], len(sw.hunks), flush=True)
    for x in (json.loads(l) for l in open(os.path.join(HERE, "nocode_manual.jsonl")) if l.strip()) \
            if os.path.exists(os.path.join(HERE, "nocode_manual.jsonl")) else ():
        # hand-written hunks for code the sweep cannot parse (llama.cpp C++, engine kernels): the hunk text is a
        # transcription of the changed lines, with the file and line range it comes from
        i = hid(x["role"], x["member"], change_lines(x["text"]))
        h = sw.hunks.setdefault(i, dict(id=i, role=x["role"], member=x["member"], kind="manual", text=x["text"],
                                        occ=[]))
        h["occ"].append(dict(variant=x["variant"], file=x["file"], cls=x["cls"], member=x["member"],
                             parent=x.get("parent", "-"), reached=True))

    inv = {}
    for t in sorted(types):
        e = types[t]
        code = []
        if t in t5map:
            code.append(dict(variant="tf5:" + t5map[t], **var_parent.get("tf5:" + t5map[t], {})))
        if t in t4map:
            mdir = os.path.join(T4, "models", t4map[t])
            extra = {k: x for k, x in var_parent.get("tf4:" + t4map[t], {}).items() if k != "files"}
            code.append(dict(variant="tf4:" + t4map[t],
                             files=[rel(p) for p in sorted(glob.glob(os.path.join(mdir, "modeling_*.py")))],
                             note=("installed version, read; the 5.18 code is the one diffed" if t in t5map else ""),
                             **extra))
        rem = collections.Counter(rv[m]["variant"] for m in e["models"] if m in rv and rv[m]["variant"])
        for v, n in rem.most_common():
            code.append(dict(variant=v, models=n, **var_parent.get(v, {})))
        for c in nc_code.get(t, []):
            how = dict(var_parent.get(c["variant"], {}))
            if c["wrapper"]:
                how = dict(parent=how.get("parent"), how=f"wrapper: the inner LM's code ({c['source']})")
            code.append(dict(variant=c["variant"], nocode_hunt=c["source"], **how))
        miss = sorted({f for m in e["models"] if m in rv for f in rv[m]["missing"]})
        inv[t] = dict(models=len(e["models"]), as_outer=e["outer"], as_lm=e["inner"], placed=e["placed"],
                      code=code, remote_missing=miss, no_code=not code, examples=sorted(e["models"])[:3])
        if t in nc:
            x = nc[nc[t]["same_as"]] if nc[t].get("same_as") else nc[t]
            inv[t]["nocode_hunt"] = {k: x[k] for k in ("status", "found", "parent", "tried") if k in x}
    json.dump(dict(var_models={v: sorted(m) for v, m in var_models.items()}, types=inv),
              open(os.path.join(HERE, "inventory_raw.json" + TAG + ".tmp"), "w"), indent=1, sort_keys=True)

    with open(os.path.join(HERE, "hunks_raw.jsonl" + TAG + ".tmp"), "w") as f:
        for i, h in sorted(sw.hunks.items()):
            ms = set()
            for o in h["occ"]:
                ms |= var_models[o["variant"]]
            for o in h["occ"]:
                if h["kind"] == "manual":
                    continue                      # transcribed by hand from code on the forward path
                RN, RM, RA, RD = var_reach.get(o["variant"], (set(), set(), set(), set()))
                if o["cls"] not in RD and o["cls"] != "<module>":
                    # a class of another module (a rebase onto the parent's tree, an imported base): its member
                    # runs if reached code calls it by name (or it is a constructor / forward / attribute)
                    reached = (o["member"] in ("__init__", "forward", "__call__", "setup", "<class>") or o["member"] in RA
                               or not h["text"].lstrip("@ new\n+-").startswith(("def ", "@")))
                else:
                    reached = ((o["cls"] == "<module>" and o["member"] in RN) or (o["cls"], o["member"]) in RM
                               or (o["member"] == "<class>" and o["cls"] in RN))
                o["reached"] = bool(h["role"] == "config" or reached)
            h["models"] = len(ms)
            h["placed_models"] = sum(1 for m in ms if byid[m]["placed"])
            h["types"] = sorted({byid[m]["inner"] for m in ms} | {byid[m]["outer"] for m in ms})
            f.write(json.dumps(h, sort_keys=True) + "\n")
    for n in ("inventory_raw.json", "hunks_raw.jsonl"):
        os.replace(os.path.join(HERE, n + TAG + ".tmp"), os.path.join(HERE, n + TAG))
    print(len(inv), "types,", len(sw.hunks), "hunks")


if __name__ == "__main__":
    main()
