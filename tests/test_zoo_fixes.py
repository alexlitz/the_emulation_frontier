"""The zoo page fixes from the competitive evaluation (dist/zoo_competitive_eval.md section 6): per-kind attention
shapes on the card, both active-param conventions, the reader's own repo in compare and "Run it", the compact
header, and the paste-any-HF-id lookup. Data checks run on the Python side; the page checks drive the session page
of tests/zoo_harness.py in a headless browser.

Run:  pytest tests/test_zoo_fixes.py -q -n 2
"""
import copy

from graph.zoo_ref import attn_kinds

G4_31B = {"model_type": "gemma4_text", "architectures": ["Gemma4ForCausalLM"], "hidden_size": 5376,
          "num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 256,
          "global_head_dim": 512, "num_global_key_value_heads": 4, "attention_k_eq_v": True,
          "intermediate_size": 21504, "sliding_window": 1024, "rms_norm_eps": 1e-6, "vocab_size": 262144,
          "hidden_activation": "gelu_pytorch_tanh", "max_position_embeddings": 262144,
          "hidden_size_per_layer_input": 0, "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10}


def test_gemma4_attention_kinds():
    """Gemma-4-31B: sliding layers 16 KV heads x 256 with the window, global layers 4 KV heads x 512, full, K = V
    -- shown per kind, not mixed into one '32/16 hd 512'."""
    ak = attn_kinds(copy.deepcopy(G4_31B))
    assert ak == [{"k": "global", "n": 10, "h": 32, "kvh": 4, "hd": 512, "keqv": True},
                  {"k": "sliding", "n": 50, "h": 32, "kvh": 16, "hd": 256, "win": 1024}]


def test_one_shape_has_no_kinds():
    llama = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 4096,
             "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336,
             "vocab_size": 128256, "rms_norm_eps": 1e-5, "max_position_embeddings": 131072}
    assert attn_kinds(llama) is None
    gemma2 = {"model_type": "gemma2", "architectures": ["Gemma2ForCausalLM"], "hidden_size": 3584,
              "num_hidden_layers": 42, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
              "intermediate_size": 14336, "vocab_size": 256000, "sliding_window": 4096, "rms_norm_eps": 1e-6,
              "max_position_embeddings": 8192}
    assert attn_kinds(gemma2) is None                  # sliding and global differ only in the window


# --- the page (headless browser, the session page of tests/zoo_harness.py) ------------------------------------

import json  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("playwright.sync_api")

from tests.zoo_harness import DESKTOP, PHONE, enc  # noqa: E402

ROWS_JS = """() => { const rows = {}; document.querySelectorAll('#panel .spec-body > dl.spec > dt').forEach(dt => {
  const dd = dt.nextElementSibling; rows[dt.textContent] = [...dd.childNodes].filter(n => !(n.classList && n.classList.contains('fc')))
  .map(n => n.textContent).join(''); }); return rows; }"""


def _node(zoo, repo):
    return next(i for i, nd in enumerate(zoo.data["nodes"]) if repo in nd["members"])


def _open_card(p, zoo, repo):
    i = _node(zoo, repo)
    p.goto_hash("#m=" + enc(repo))
    p.pg.wait_for_function(f"(() => {{ const g = document.querySelector('#svg g.node.me'); return g && +g.dataset.i === {i}; }})()")
    return i


def test_card_attention_kinds_page(open_page, zoo):
    p = open_page()
    _open_card(p, zoo, "nvidia/Gemma-4-31B-IT-NVFP4")
    rows = p.pg.evaluate(ROWS_JS)
    assert rows["Global layers (10)"] == "32 query · 4 KV · hd 512 · full · K = V", rows
    assert rows["Sliding layers (50)"] == "32 query · 16 KV · hd 256 · window 1,024", rows
    assert "Heads" not in rows and "Head dim" not in rows
    kn = p.pg.eval_on_selector_all("#panel dl.keynums > div", "ds => ds.map(d => d.textContent)")
    assert any(t.startswith("Heads32/4 · 32/16") for t in kn), kn


def test_active_params_both_conventions(open_page, zoo):
    p = open_page()
    _open_card(p, zoo, "openai/gpt-oss-20b")
    rows = p.pg.evaluate(ROWS_JS)
    assert rows["Active params (incl. embeddings)"].startswith("4.2B"), rows      # 4.19B: embedding and head in
    assert rows["Active params (excl. embeddings, vendor convention)"].startswith("3.6B"), rows   # OpenAI's 3.6B
    p.goto_hash("#v=models")
    p.pg.wait_for_selector("#mtbl tbody tr")
    assert p.pg.locator("#mtbl tbody td .sub:has-text('excl. emb.')").count() > 0


def test_compare_keeps_the_users_repos(open_page, zoo):
    nodes = zoo.data["nodes"]
    picks = ["meta-llama/Llama-3.1-8B", "google/gemma-2-9b", "Qwen/Qwen3-30B-A3B"]
    link = "#v=compare&c=" + ",".join(enc(r) for r in picks)
    p = open_page(link)
    p.pg.wait_for_selector("#cmptbl thead th .linkish")
    assert p.hash() == link                                         # the permalink keeps the exact repos
    heads = p.pg.eval_on_selector_all("#cmptbl thead th .linkish", "b => b.map(x => x.textContent)")
    want = [nodes[_node(zoo, r)]["name"] if nodes[_node(zoo, r)]["members"][0] == r else r.split("/")[-1] for r in picks]
    assert heads == want, heads
    notes = p.pg.eval_on_selector_all("#cmptbl thead th", "ths => ths.slice(1).map(t => (t.querySelector('.repnote') || {}).textContent || '')")
    for r, nt in zip(picks, notes):
        rep0 = nodes[_node(zoo, r)]["members"][0]
        assert (rep0 in nt) == (rep0 != r), (r, nt)


def test_run_it_uses_the_selected_repo(open_page, zoo):
    repo = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    i = _node(zoo, repo)
    rep0 = zoo.data["nodes"][i]["members"][0]
    p = open_page()
    _open_card(p, zoo, repo)
    cmds = p.pg.eval_on_selector_all("#recipes pre", "ps => ps.map(x => x.textContent)")
    assert cmds and any(repo in c for c in cmds)
    if rep0 != repo:
        assert not any(rep0 in c for c in cmds), cmds
        assert p.pg.locator("#recipes p.repnote").count() == 1
        assert p.pg.locator("#panel p.repnote").count() >= 1


@pytest.mark.parametrize("vp,mobile", [(DESKTOP, False), (PHONE, True)])
def test_first_screen_has_search_and_view(open_page, vp, mobile):
    p = open_page(viewport=vp, mobile=mobile)
    box = lambda s: p.pg.evaluate(f"(() => {{ const b = document.querySelector('{s}').getBoundingClientRect(); return [b.top, b.bottom]; }})()")
    q = box("#q")
    assert q[1] < 0.45 * vp["height"], q                            # the search box near the top
    view = "#view-graph" if not mobile else "#view-table"
    assert box(view)[0] < 0.75 * vp["height"], box(view)            # the active view starts on the first screen
    assert p.pg.locator("#about").is_hidden()
    p.pg.click("#about-btn")
    assert p.pg.locator("#about .lede").first.is_visible()


# --- the any-HF-id lookup (zoo-fixes 11): config.json served by a route, so no network is needed --------------

LOOKUP_JS = """() => { const rows = {}; document.querySelectorAll('#lk-out dl.spec > dt').forEach(dt => {
  rows[dt.textContent] = dt.nextElementSibling.firstChild.textContent; }); return {rows,
  status: document.getElementById('lk-status').textContent,
  near: [...document.querySelectorAll('#lk-out ul.lk-near li')].map(li => [+li.dataset.i, li.dataset.exact === '1']),
  fw: Object.fromEntries([...document.querySelectorAll('#lk-fw tbody tr')].map(tr => [tr.dataset.fw, tr.cells[1].firstChild.textContent]))}; }"""


def _fmt_bytes_page(p, b):
    """The page's fmtBytes (graph/templates/zoo.html), in Python."""
    for v, s in ((2 ** 40, "TiB"), (2 ** 30, "GiB"), (2 ** 20, "MiB"), (2 ** 10, "KiB")):
        if b >= v:
            x = b / v
            return p.pg.evaluate("([x, d]) => x.toFixed(d)", [x, 0 if x >= 100 else 1 if x >= 10 else 2]) + " " + s
    return f"{round(b):,} B"


def _serve(p, routes):
    def handle(route):
        url = route.request.url
        for key, (status, body) in routes.items():
            if f"/{key}/resolve/main/config.json" in url:
                return route.fulfill(status=status, body=body, headers={"Access-Control-Allow-Origin": "*",
                                                                         "Content-Type": "application/json"})
        return route.fulfill(status=404, body="not found", headers={"Access-Control-Allow-Origin": "*"})
    p.pg.route("https://huggingface.co/**", handle)


def test_lookup_any_hf_id(open_page, zoo, truth):
    from graph import arch_metrics as AM
    src = "nvidia/Gemma-4-31B-IT-NVFP4"
    raw = truth.configs[src]
    p = open_page()
    if p.pg.locator("#tab-lookup").is_hidden():
        pytest.skip("the page was built without the JS port (graph/templates/arch_metrics.js)")
    _serve(p, {"some-org/gemma4-copy": (200, json.dumps(raw)), "gated-org/locked": (401, "{}")})
    p.goto_hash("#v=lookup&id=" + enc("some-org/gemma4-copy"))
    p.pg.wait_for_selector("#lk-fw tbody tr")
    got = p.pg.evaluate(LOOKUP_JS)
    m = AM.metrics(json.loads(json.dumps(raw)))
    assert got["rows"]["KV per token (bf16)"] == _fmt_bytes_page(p, m["kv_per_token"]["bf16"])
    assert got["rows"]["Weights (bf16)"] == _fmt_bytes_page(p, m["params"]["weight_bytes"]["bf16"])
    assert got["rows"]["KV cache (bf16)"] == _fmt_bytes_page(p, m["kv"]["4k"]["bf16"])     # the 4k default
    i = _node(zoo, src)
    assert got["near"][0] == [i, True], got["near"]                 # the nearest zoo shape is its own
    assert got["fw"]["vllm"] == "native" and got["fw"]["executorch"] == "native"   # gemma4_31b export recipe
    assert p.hash() == "#v=lookup&id=" + enc("some-org/gemma4-copy")
    # a gated repo and a missing one say so
    p.pg.fill("#lk-id", "https://huggingface.co/gated-org/locked")
    p.pg.click("#lk-go")
    p.pg.wait_for_function("() => /gated or private/.test(document.getElementById('lk-status').textContent)")
    p.pg.fill("#lk-id", "nobody/nothing")
    p.pg.click("#lk-go")
    p.pg.wait_for_function("() => /no config.json/.test(document.getElementById('lk-status').textContent)")
    # the browser itself logs the 401 / 404 responses; the page handled them (no exception, a message shown)
    p.errors[:] = [e for e in p.errors if "Failed to load resource" not in e]


def test_lookup_transformers_tier(open_page, zoo, truth):
    """A Phi-2 config: vLLM shows the Transformers-backend tier, as in the zoo's own framework table."""
    raw = truth.configs["microsoft/phi-2"]
    p = open_page()
    if p.pg.locator("#tab-lookup").is_hidden():
        pytest.skip("the page was built without the JS port")
    _serve(p, {"x/phi2": (200, json.dumps(raw))})
    p.goto_hash("#v=lookup&id=x%2Fphi2")
    p.pg.wait_for_selector("#lk-fw tbody tr")
    got = p.pg.evaluate(LOOKUP_JS)
    assert got["fw"]["vllm"] == "via Transformers backend", got["fw"]
