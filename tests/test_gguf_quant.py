"""graph/gguf_quant.py (llama.cpp GGUF quantisation sizes) and its JS port graph/templates/gguf_quant.js.

(a) the type table extracted from the pinned ggml sources gives the published bits per weight;
(b) hand-checked llama-quantize rule cases (use_more_bits layers, tied embeddings, the incompatible-shape
    fallback, MoE / GQA / 70B / falcon branches, the visiting order the counters depend on, converter F32 rules);
(c) the JS port equals the Python on randomised inventories (hypothesis), for every ftype, errors included;
(d) graph/data/llama_cpp_quant.json is what extract() gives on the pinned sources, when they are in /tmp.

Run:  pytest tests/test_gguf_quant.py -n 4 -q
"""
import json
import os
import random
import shutil
import subprocess

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from graph import gguf_quant as g

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(HERE, "graph", "templates", "gguf_quant.js")
NODE = shutil.which("node")
T = g.tables()
FTYPES = list(T["ftypes"])


# ---------------------------------------------------------------------------------------------------------------
# inventories
# ---------------------------------------------------------------------------------------------------------------

def llama(n_layer=32, d=4096, ff=14336, kv=1024, vocab=128256, tied=False, src="BF16"):
    ts = [{"name": "token_embd.weight", "shape": [d, vocab]}, {"name": "output_norm.weight", "shape": [d]}]
    if not tied:
        ts.append({"name": "output.weight", "shape": [d, vocab]})
    for i in range(n_layer):
        p = "blk.%d." % i
        ts += [{"name": p + "attn_norm.weight", "shape": [d]}, {"name": p + "attn_q.weight", "shape": [d, d]},
               {"name": p + "attn_k.weight", "shape": [d, kv]}, {"name": p + "attn_v.weight", "shape": [d, kv]},
               {"name": p + "attn_output.weight", "shape": [d, d]}, {"name": p + "ffn_norm.weight", "shape": [d]},
               {"name": p + "ffn_gate.weight", "shape": [d, ff]}, {"name": p + "ffn_up.weight", "shape": [d, ff]},
               {"name": p + "ffn_down.weight", "shape": [ff, d]}]
    for t, s in zip(ts, g.source_types(ts, "llama", "bf16" if src == "BF16" else "f16")):
        t["src"] = s
    return ts


def mixtral(n_layer=32, d=4096, ff=14336, kv=1024, n_expert=8):
    ts = [{"name": "token_embd.weight", "shape": [d, 32000]}, {"name": "output.weight", "shape": [d, 32000]},
          {"name": "output_norm.weight", "shape": [d]}]
    for i in range(n_layer):
        p = "blk.%d." % i
        ts += [{"name": p + "attn_norm.weight", "shape": [d]}, {"name": p + "attn_q.weight", "shape": [d, d]},
               {"name": p + "attn_k.weight", "shape": [d, kv]}, {"name": p + "attn_v.weight", "shape": [d, kv]},
               {"name": p + "attn_output.weight", "shape": [d, d]}, {"name": p + "ffn_norm.weight", "shape": [d]},
               {"name": p + "ffn_gate_inp.weight", "shape": [d, n_expert]},
               {"name": p + "ffn_gate_exps.weight", "shape": [d, ff, n_expert]},
               {"name": p + "ffn_up_exps.weight", "shape": [d, ff, n_expert]},
               {"name": p + "ffn_down_exps.weight", "shape": [ff, d, n_expert]}]
    for t, s in zip(ts, g.source_types(ts, "llama")):
        t["src"] = s
    return ts


def types_of(ts, ftype, arch="llama", **kw):
    return dict(zip([t["name"] for t in ts], g.tensor_types(ts, ftype, arch, **kw)))


def umb(i, n):
    """use_more_bits written out independently of the port."""
    return i < n // 8 or i >= (7 * n) // 8 or (i - n // 8) % 3 == 2


# ---------------------------------------------------------------------------------------------------------------
# (a) the type table
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ty,bpw", [("Q4_K", 4.5), ("Q6_K", 6.5625), ("Q8_0", 8.5), ("Q2_K", 2.625),
                                    ("Q3_K", 3.4375), ("Q5_K", 5.5), ("Q4_0", 4.5), ("Q4_1", 5.0), ("Q5_0", 5.5),
                                    ("Q5_1", 6.0), ("IQ4_NL", 4.5), ("IQ4_XS", 4.25), ("F16", 16.0),
                                    ("BF16", 16.0), ("F32", 32.0)])
def test_published_bpw(ty, bpw):
    assert T["types"][ty]["bpw"] == bpw
    assert T["types"][ty]["bpw"] == 8 * T["types"][ty]["bytes"] / T["types"][ty]["block"]


def test_table_shape():
    blocks = {"Q4_K": (256, 144), "Q6_K": (256, 210), "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110),
              "Q5_K": (256, 176), "Q4_0": (32, 18), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136)}
    for ty, (blk, by) in blocks.items():
        assert (T["types"][ty]["block"], T["types"][ty]["bytes"]) == (blk, by)
    assert T["alignment"] == 32
    for ft, ty in {"Q8_0": "Q8_0", "Q6_K": "Q6_K", "Q5_K_M": "Q5_K", "Q4_K_M": "Q4_K", "Q3_K_M": "Q3_K",
                   "Q2_K": "Q2_K", "Q4_0": "Q4_0", "BF16": "BF16", "F16": "F16"}.items():
        assert T["ftypes"][ft] == ty
    assert T["fallback"]["Q4_K"] == "Q5_0" and T["fallback"]["Q6_K"] == "Q8_0" and T["fallback"]["Q2_K"] == "Q4_0"
    assert g.tensor_bytes("Q4_K", 4096 * 4096, 4096) == 4096 * 16 * 144
    with pytest.raises(ValueError):
        g.tensor_bytes("Q4_K", 100 * 10, 100)


# ---------------------------------------------------------------------------------------------------------------
# (b) rule cases
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ftype", ["Q4_K_M", "Q5_K_M"])
def test_k_m_use_more_bits(ftype):
    ts = llama()
    ty = types_of(ts, ftype, n_gqa=4)
    base = T["ftypes"][ftype]
    assert ty["output.weight"] == "Q6_K"
    assert ty["token_embd.weight"] == base
    assert ty["output_norm.weight"] == "F32"
    for i in range(32):
        want = "Q6_K" if umb(i, 32) else base
        assert ty["blk.%d.attn_v.weight" % i] == want
        assert ty["blk.%d.ffn_down.weight" % i] == want
        for k in ("attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"):
            assert ty["blk.%d.%s.weight" % (i, k)] == base
        assert ty["blk.%d.attn_norm.weight" % i] == "F32"
    bumped = sorted(i for i in range(32) if umb(i, 32))
    assert bumped == [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31]


def test_tied_embeddings():
    ty = types_of(llama(tied=True), "Q4_K_M")
    assert ty["token_embd.weight"] == "Q6_K"
    assert types_of(llama(tied=True), "Q8_0")["token_embd.weight"] == "Q8_0"
    assert types_of(llama(tied=True), "Q4_0")["token_embd.weight"] == "Q6_K"
    assert types_of(llama(tied=False), "Q4_0")["token_embd.weight"] == "Q4_0"


def test_fallback_rows():
    ts = llama(n_layer=2, d=896, ff=4864, kv=128, vocab=151936)       # Qwen2-0.5B-like: 896 % 256 = 128
    for ftype, want in [("Q4_K_M", "Q5_0"), ("Q5_K_M", "Q5_1"), ("Q6_K", "Q8_0"), ("Q2_K", "Q4_0"),
                        ("Q3_K_M", "Q4_0"), ("Q8_0", "Q8_0"), ("Q4_0", "Q4_0")]:
        ty = types_of(ts, ftype)
        assert ty["blk.0.attn_q.weight"] == want, ftype
        assert ty["blk.1.ffn_up.weight"] == want, ftype
    ty = types_of(ts, "Q4_K_M")
    assert ty["output.weight"] == "Q8_0"                               # nx % qk_k != 0 -> Q8_0
    # 4864 % 256 == 0, so ffn_down keeps the K rules: of 2 layers only layer 1 (>= 7*2/8) uses more bits
    assert [ty["blk.%d.ffn_down.weight" % i] for i in range(2)] == ["Q4_K", "Q6_K"]
    odd = [{"name": "blk.0.attn_q.weight", "shape": [100, 64], "src": "BF16"}]
    for ftype in ("Q4_K_M", "Q4_0", "Q8_0", "Q6_K", "IQ4_XS"):
        assert g.tensor_types(odd, ftype, "llama") == ["F16"]
    st_ = {}
    g.tensor_types(ts, "Q4_K_M", "llama", _stats=st_)
    assert st_["n_fallback"] > 0


def test_q3_k_m_and_order():
    ts = llama(n_layer=12)
    random.Random(1).shuffle(ts)
    ty = types_of(ts, "Q3_K_M")
    assert [ty["blk.%d.attn_v.weight" % i] for i in range(12)] == ["Q5_K", "Q5_K"] + ["Q4_K"] * 10
    assert all(ty["blk.%d.attn_output.weight" % i] == "Q4_K" for i in range(12))
    assert [ty["blk.%d.ffn_down.weight" % i] for i in range(12)] == ["Q4_K"] * 12      # 12 // 16 == 0
    ty32 = types_of(llama(), "Q3_K_M")
    assert [ty32["blk.%d.ffn_down.weight" % i] for i in range(4)] == ["Q5_K", "Q5_K", "Q4_K", "Q4_K"]
    assert ty32["blk.5.attn_q.weight"] == "Q3_K"
    fal = types_of(llama(), "Q3_K_M", "falcon")
    assert [fal["blk.%d.ffn_down.weight" % i] for i in range(32)] == \
        ["Q5_K" if i < 2 else "Q4_K" if umb(i, 32) else "Q3_K" for i in range(32)]
    assert fal["output.weight"] == "Q8_0"


def test_q2_k_gqa():
    ts = llama(n_layer=4)
    ty = types_of(ts, "Q2_K", n_gqa=1)
    assert ty["blk.0.attn_v.weight"] == "Q3_K" and ty["blk.0.ffn_down.weight"] == "Q3_K"
    assert ty["blk.0.attn_output.weight"] == "Q3_K" and ty["blk.0.attn_q.weight"] == "Q2_K"
    assert ty["output.weight"] == "Q6_K"
    assert types_of(ts, "Q2_K", n_head=32, n_head_kv=8)["blk.0.attn_v.weight"] == "Q4_K"


def test_moe_8_experts():
    ts = mixtral()
    ty = types_of(ts, "Q4_K_M", n_expert=8, n_gqa=4)
    assert ty["blk.3.attn_v.weight"] == "Q8_0" and ty["blk.3.attn_k.weight"] == "Q8_0"
    assert ty["blk.3.attn_output.weight"] == "Q5_K"
    assert ty["blk.3.ffn_gate_inp.weight"] == "F32"
    assert [ty["blk.%d.ffn_down_exps.weight" % i] for i in range(32)] == \
        ["Q6_K" if umb(i, 32) else "Q4_K" for i in range(32)]
    assert types_of(ts, "Q4_0", n_expert=8)["blk.0.attn_v.weight"] == "Q8_0"
    bad = ts + [{"name": "blk.40.ffn_down_exps.weight", "shape": [14336, 4096, 8], "src": "BF16"}]
    with pytest.raises(ValueError):
        g.tensor_types(bad, "Q4_K_M", "llama", n_expert=8, n_layer=32)


def test_70b():
    ts = llama(n_layer=80, d=256, ff=512, kv=64, vocab=1000)
    assert g.is_70b("llama", 80, 64, 8) and not g.is_70b("llama", 80, 64, 64) and not g.is_70b("llama", 80, 64, 8, 8)
    assert g.is_70b("qwen2", 80) and not g.is_70b("qwen2", 64)
    ty = types_of(ts, "Q3_K_M", n_head=64, n_head_kv=8)
    assert all(ty["blk.%d.attn_v.weight" % i] == "Q5_K" for i in range(80))
    ty = types_of(ts, "Q3_K_M", n_head=64, n_head_kv=64)
    assert ty["blk.5.attn_v.weight"] == "Q4_K"


def test_unquantized_and_source():
    ts = llama(n_layer=2)
    for ftype in ("BF16", "F16"):
        ty = types_of(ts, ftype)
        assert ty["blk.0.attn_q.weight"] == ftype and ty["output_norm.weight"] == "F32"
    assert g.tensor_types(ts, None, "llama") == [t["src"] for t in ts]
    inv = [{"name": "blk.0.ffn_gate_inp.weight", "shape": [64, 8]}, {"name": "blk.0.attn_q.bias", "shape": [64]},
           {"name": "blk.0.attn_norm.weight", "shape": [64]}, {"name": "blk.0.ssm_conv1d.weight", "shape": [4, 64]},
           {"name": "blk.0.attn_q.weight", "shape": [64, 64]}, {"name": "rope_freqs.weight", "shape": [32]},
           {"name": "blk.0.exp_probs_b.bias", "shape": [8, 1]}, {"name": "position_embd.weight", "shape": [64, 512]},
           {"name": "blk.0.hc_attn_fn.weight", "shape": [64, 4]}]
    assert g.source_types(inv, "llama") == ["F32", "F32", "F32", "F32", "BF16", "F32", "F32", "F32", "BF16"]
    assert g.source_types(inv, "llama", "f16")[4] == "F16"
    assert g.source_types(inv, "hy_v4")[8] == "F32"
    assert g.source_types([{"name": "output.weight", "shape": [64, 10]}], "hy_v4", flags={"lm_head_fp32": True}) == ["F32"]
    assert g.source_types([{"name": "blk.0.x.weight", "shape": [100, 3]}], "llama", "q8_0") == ["F16"]
    # quantize keeps a 2-D F32 the rules exclude, and quantizes a 2-D F32 they do not
    ty = g.tensor_types([{"name": "blk.0.ssm_conv1d.weight", "shape": [4, 64], "src": "F32"},
                         {"name": "blk.0.indexer.proj.weight", "shape": [256, 64], "src": "F32"}], "Q4_K_M", "deepseek2")
    assert ty == ["F32", "Q4_K"]


def test_sizes_and_header():
    ts = llama()
    b = g.breakdown(ts, "Q4_K_M", "llama", n_gqa=4)
    assert b["data"] == sum(r["bytes"] for r in b["tensors"]) == sum(v["bytes"] for v in b["by_type"].values())
    assert all(r["bytes"] % 32 == 0 for r in b["tensors"])
    # Llama-3-8B: 8.03e9 params; Q4_K_M ~4.89 bpw on the data
    assert 4.85e9 < b["data"] < 4.95e9
    assert g.gguf_bytes(ts, "Q4_K_M", "llama", 1000, n_gqa=4) == b["data"] + 1000
    h = g.gguf_header_bytes(128256, tensors=ts)
    assert h % 32 == 0 and 2e6 < h < 1e7
    h2 = g.gguf_header_bytes(128256, n_tensors=len(ts))
    assert abs(h2 - h) < 5000


# ---------------------------------------------------------------------------------------------------------------
# (c) Python == JS
# ---------------------------------------------------------------------------------------------------------------

RUNNER = r"""
const Q = require(process.argv[1]);
const fs = require("fs");
Q.setTables(JSON.parse(fs.readFileSync(process.argv[2], "utf8")));
const rl = require("readline").createInterface({ input: process.stdin });
const call = (c) => {
  switch (c.fn) {
    case "tensorTypes": return Q.tensorTypes(c.tensors, c.ftype, c.arch, c.opts);
    case "breakdown": return Q.breakdown(c.tensors, c.ftype, c.arch, c.header, c.opts);
    case "sourceTypes": return Q.sourceTypes(c.tensors, c.arch, c.outtype, c.flags);
    case "headerBytes": return Q.headerBytes(c.vocab, c.opts);
    case "tensorBytes": return Q.tensorBytes(c.type, c.n, c.ne0);
  }
  throw new Error("unknown fn " + c.fn);
};
rl.on("line", (line) => {
  const out = JSON.parse(line).map(c => { try { return { ok: call(c) }; } catch (e) { return { err: String(e.message) }; } });
  process.stdout.write(JSON.stringify(out) + "\n");
});
"""


@pytest.fixture(scope="module")
def node():
    if NODE is None:
        pytest.skip("node is not installed")
    p = subprocess.Popen([NODE, "-e", RUNNER, JS, g.DATA], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True)
    yield p
    p.stdin.close()
    p.wait(timeout=10)


def _js(node, calls):
    node.stdin.write(json.dumps(calls) + "\n")
    node.stdin.flush()
    return json.loads(node.stdout.readline())


def _py(c):
    try:
        if c["fn"] == "tensorTypes":
            r = g.tensor_types(c["tensors"], c["ftype"], c["arch"], **c["opts"])
        elif c["fn"] == "breakdown":
            r = g.breakdown(c["tensors"], c["ftype"], c["arch"], c["header"], **c["opts"])
        elif c["fn"] == "sourceTypes":
            r = g.source_types(c["tensors"], c["arch"], c["outtype"], c["flags"])
        elif c["fn"] == "headerBytes":
            o = dict(c["opts"])
            r = g.gguf_header_bytes(c["vocab"], **o)
        elif c["fn"] == "tensorBytes":
            r = g.tensor_bytes(c["type"], c["n"], c["ne0"])
        return {"ok": json.loads(json.dumps(r))}
    except (ValueError, KeyError, TypeError) as e:
        return {"err": str(e)}


def _check(node, calls):
    js = _js(node, calls)
    for c, a, b in zip(calls, [_py(c) for c in calls], js):
        assert ("ok" in a) == ("ok" in b), (c["fn"], c.get("ftype"), c.get("arch"), a, b)
        if "ok" in a:
            assert a["ok"] == b["ok"], (c["fn"], c.get("ftype"), c.get("arch"))


ARCHS = ["llama", "falcon", "qwen2", "qwen3moe", "deepseek2", "glm5-next", "muse-spark", "limite", "metis16",
         "kimi-k31", "xing4_0", "vllm-decision2", "hy_v4", "minimax-m3", "jais2", "olmo", "deci", "llama-embed"]
LAYER_SUFFIXES = [
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
    "attn_qkv_gate.weight", "attn_kv_b.weight", "attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight",
    "attn_k_b.weight", "attn_v_b.weight", "attn_gate.weight", "attn_ve_gate.weight", "ffn_norm.weight",
    "ffn_up.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up_exps.weight", "ffn_gate_exps.weight",
    "ffn_down_exps.weight", "ffn_down_chexps.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_latent.weight", "ffn_gate_up_exps.weight",
    "ssm_conv1d.weight", "ssm_in.weight", "ssm_out.weight", "ssm_a", "ssm_dt.bias", "indexer.proj.weight",
    "indexer.attn_k.weight", "indexer.k_proj.weight", "hc_attn_fn.weight", "hc_attn_mix.weight", "mudd_up.weight",
    "time_mix_w1.weight", "attn_q.bias", "exp_probs_b.bias", "nextn.eh_proj.weight", "attn_rel_b.weight",
    "shortconv.conv.weight", "altup_router.weight", "attn_q.lora_a", "ffn_exp_embd.weight"]
GLOBAL_NAMES = ["token_embd.weight", "output.weight", "output_norm.weight", "per_layer_token_embd.weight",
                "position_embd.weight", "token_types.weight", "decision.head.weight", "cont_gate.weight",
                "depth_mem_entry_bias", "output_hc_fn.weight", "rope_freqs.weight", "abcde", "blk.x.ffn_down.weight",
                "blk.99.ffn_down_exps.weight", "blk.2.weight", "v.position_embd.weight", "a.conv1d.0.weight"]
NE0 = [1, 16, 32, 64, 96, 100, 128, 192, 256, 512, 768, 896, 1000, 1024, 2048, 4096]
NE = [1, 2, 3, 8, 64, 128]

shape_st = st.tuples(st.sampled_from(NE0), st.lists(st.sampled_from(NE), max_size=3)).map(lambda x: [x[0]] + x[1])
src_st = st.sampled_from(["BF16", "F16", "F32", None])


@st.composite
def inventory(draw):
    n_layer = draw(st.sampled_from([0, 1, 2, 3, 5, 8, 9, 16, 17, 33, 68, 80]))
    suffixes = draw(st.lists(st.sampled_from(LAYER_SUFFIXES), min_size=1, max_size=10, unique=True))
    shapes = {s: draw(shape_st) for s in suffixes}
    srcs = {s: draw(src_st) for s in suffixes}
    drop = draw(st.floats(0, 0.3))
    rnd = random.Random(draw(st.integers(0, 2 ** 31)))
    ts = []
    for name in draw(st.lists(st.sampled_from(GLOBAL_NAMES), max_size=6, unique=True)):
        ts.append({"name": name, "shape": draw(shape_st), "src": draw(src_st)})
    for i in range(n_layer):
        for s in suffixes:
            if rnd.random() >= drop:
                ts.append({"name": "blk.%d.%s" % (i, s), "shape": shapes[s], "src": srcs[s]})
    rnd.shuffle(ts)
    for t in ts:
        if t["src"] is None:
            del t["src"]
    opts = {}
    if draw(st.booleans()):
        opts["n_expert"] = draw(st.sampled_from([0, 1, 4, 8, 16]))
    if draw(st.booleans()):
        opts["n_gqa"] = draw(st.sampled_from([0, 1, 2, 4, 8]))
    if draw(st.booleans()):
        opts["n_head"] = draw(st.sampled_from([8, 32, 64]))
        opts["n_head_kv"] = draw(st.sampled_from([0, 1, 8, 32, 64]))
    if draw(st.booleans()):
        opts["has_imatrix"] = draw(st.booleans())
    if draw(st.integers(0, 4)) == 0:
        opts["n_layer"] = draw(st.integers(0, 90))
    if draw(st.integers(0, 4)) == 0:
        opts["model_70b"] = draw(st.booleans())
    if draw(st.integers(0, 4)) == 0:
        opts["n_layer_nextn"] = draw(st.integers(0, 2))
    return draw(st.sampled_from(ARCHS)), ts, opts


CHUNKS = [FTYPES[i::4] for i in range(4)]


@pytest.mark.parametrize("chunk", range(4))
@settings(max_examples=150, deadline=None, suppress_health_check=list(HealthCheck))
@given(case=inventory(), header=st.integers(0, 10 ** 7))
def test_js_equals_python(node, chunk, case, header):
    arch, ts, opts = case
    calls = []
    for ftype in CHUNKS[chunk] + ([None] if chunk == 0 else []):
        calls.append({"fn": "tensorTypes", "tensors": ts, "ftype": ftype, "arch": arch, "opts": opts})
        calls.append({"fn": "breakdown", "tensors": ts, "ftype": ftype, "arch": arch, "header": header,
                      "opts": opts})
    _check(node, calls)


@settings(max_examples=150, deadline=None, suppress_health_check=list(HealthCheck))
@given(case=inventory(), outtype=st.sampled_from(["bf16", "f16", "f32", "q8_0"]), flag=st.booleans(),
       vocab=st.integers(1, 300000), tok=st.floats(1, 20), merges=st.one_of(st.none(), st.integers(0, 400000)),
       tmpl=st.integers(0, 20000), tokenizer=st.sampled_from(["gpt2", "llama"]), moe=st.booleans(),
       use_tensors=st.booleans())
def test_js_equals_python_source_header(node, case, outtype, flag, vocab, tok, merges, tmpl, tokenizer, moe,
                                         use_tensors):
    arch, ts, _ = case
    o = {"avg_token_bytes": tok, "n_merges": merges, "chat_template_bytes": tmpl, "tokenizer": tokenizer,
         "moe": moe, "arch": arch}
    if use_tensors:
        o["tensors"] = ts
    else:
        o["n_tensors"] = len(ts)
    calls = [{"fn": "sourceTypes", "tensors": ts, "arch": arch, "outtype": outtype, "flags": {"lm_head_fp32": flag}},
             {"fn": "headerBytes", "vocab": vocab, "opts": o}]
    for t in ts[:5]:
        n = 1
        for d in t["shape"]:
            n *= d
        for ty in ("Q4_K", "Q8_0", "BF16", "IQ4_NL"):
            calls.append({"fn": "tensorBytes", "type": ty, "n": n, "ne0": t["shape"][0]})
    _check(node, calls)


def test_js_known_cases(node):
    calls = []
    for ts, arch, opts in [(llama(), "llama", {"n_gqa": 4}), (llama(tied=True), "llama", {}),
                           (mixtral(), "llama", {"n_expert": 8}), (llama(n_layer=80, d=256, ff=512, kv=64,
                                                                          vocab=1000), "llama",
                                                                    {"n_head": 64, "n_head_kv": 8}),
                           (llama(), "falcon", {})]:
        for ftype in FTYPES + [None]:
            calls.append({"fn": "breakdown", "tensors": ts, "ftype": ftype, "arch": arch, "header": 123,
                          "opts": opts})
    _check(node, calls)


# ---------------------------------------------------------------------------------------------------------------
# (d) the JSON is current
# ---------------------------------------------------------------------------------------------------------------

def test_json_current():
    commit_file = os.path.join(g.SRC_DIR, "COMMIT")
    if not os.path.exists(commit_file):
        pytest.skip("pinned llama.cpp sources not in %s (python -m graph.gguf_quant fetches them)" % g.SRC_DIR)
    with open(commit_file) as f:
        sha = f.read().strip()
    if sha != T["commit"]:
        pytest.skip("%s holds commit %s, the JSON %s" % (g.SRC_DIR, sha, T["commit"]))
    fresh = json.loads(json.dumps(g.extract(g.SRC_DIR)))
    stored = {k: v for k, v in T.items() if k != "fetched"}
    for k in sorted(set(fresh) | set(stored)):
        assert fresh.get(k) == stored.get(k), "graph/data/llama_cpp_quant.json[%r] is stale: rerun " \
                                              "python -m graph.gguf_quant" % k
    assert T["commit"] == g.PINNED
