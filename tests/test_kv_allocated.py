"""The headline KV is what the runtimes ALLOCATE: Gemma-4's attention_k_eq_v layers cache K and V as two tensors.

transformers' Gemma4TextAttention (5.18 modeling_gemma4.py) computes V = v_norm(k_proj x) and K = RoPE(k_norm(k_proj x))
and passes both to past_key_values.update; vLLM gemma4.py and llama.cpp gemma4.cpp also store both. The one-tensor
figure is kept as `kv_minimum`. DeepSeek-V4's cache aliases V to K (one tensor in both conventions).

The real-cache check builds a tiny Gemma-4 K = V model with the stock transformers 5.18 code (unpacked locally,
the library only: no Hub code) and counts the elements its cache object holds after a prefill and a decode step.
"""
import copy
import json
import os
import subprocess
import sys

import pytest

from graph import arch_metrics as AM

# a tiny Gemma-4 text config with the 31B's structure: 5 sliding + 1 full layer, K = V on the full layers, a
# different KV-head count and head dim on the global layers
G4 = {"model_type": "gemma4_text", "architectures": ["Gemma4ForCausalLM"], "vocab_size": 97, "hidden_size": 64,
      "intermediate_size": 96, "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 2,
      "head_dim": 16, "global_head_dim": 32, "num_global_key_value_heads": 1, "attention_k_eq_v": True,
      "sliding_window": 8, "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
      "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "rms_norm_eps": 1e-6,
      "hidden_activation": "gelu_pytorch_tanh", "max_position_embeddings": 128, "tie_word_embeddings": True}

# the 31B's own numbers (the eval reference, dist/zoo_eval_data.json): 901,120 B per token, 3.28 GiB at 32k
G4_31B = {"model_type": "gemma4_text", "architectures": ["Gemma4ForCausalLM"], "hidden_size": 5376,
          "num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 256,
          "global_head_dim": 512, "num_global_key_value_heads": 4, "attention_k_eq_v": True,
          "intermediate_size": 21504, "sliding_window": 1024, "rms_norm_eps": 1e-6, "vocab_size": 262144,
          "hidden_activation": "gelu_pytorch_tanh", "max_position_embeddings": 262144,
          "hidden_size_per_layer_input": 0, "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10}


def test_gemma4_headline_counts_k_and_v():
    m = AM.metrics(copy.deepcopy(G4_31B))
    assert m["kv_per_token"]["bf16"] == 901_120                       # 50 x 16 x 512 x 2 + 10 x 4 x 512 x 2 x 2
    assert m["kv"]["32k"]["bf16"] == 3_522_396_160                    # 3.28 GiB, the eval reference
    assert m["kv_materialized"]["32k"]["bf16"] == m["kv"]["32k"]["bf16"]
    assert m["kv_minimum"]["32k"]["bf16"] == 2_180_218_880           # the old one-tensor headline (2.03 GiB)
    assert m["kv_minimum"]["per_token_elems"] == 430_080
    assert "theoretical minimum" in m["kv_minimum"]["note"]
    assert m["memory"]["32k"]["kv"] == m["kv"]["32k"]["bf16"]         # memory and bandwidth read the headline


def test_minimum_equals_headline_without_k_eq_v():
    cfg = dict(copy.deepcopy(G4_31B), attention_k_eq_v=False)
    m = AM.metrics(cfg)
    for L in ("1k", "32k", "max"):
        assert m["kv_minimum"][L]["bf16"] == m["kv"][L]["bf16"]
    assert m["kv_minimum"]["note"] == "equal to the headline"


def test_snapshot_gemma4_repos_allocated():
    """Every Gemma-4 K = V repo of the built metrics file: headline = kv_materialized > kv_minimum."""
    if not os.path.exists(AM.OUT):
        pytest.skip("graph/data/arch_metrics.json not built")
    with open(AM.OUT) as f:
        repos = json.load(f)["repos"]
    n = 0
    for name, r in repos.items():
        if "error" in r or not r["layers"].get("k_eq_v") or not str(r["model_type"]).startswith("gemma4"):
            continue
        n += 1
        for L in ("1k", "32k", "max"):
            assert r["kv"][L]["bf16"] == r["kv_materialized"][L]["bf16"], name
            assert r["kv_minimum"][L]["bf16"] < r["kv"][L]["bf16"], name
    assert n >= 40


BODY = '''
import json
from transformers import Gemma4TextConfig, Gemma4ForCausalLM
from graph.arch_trace import cache_elems
raw = json.loads(RAW)
cfg = Gemma4TextConfig(**{k: v for k, v in raw.items() if k not in ("model_type", "architectures")})
cfg._attn_implementation = "eager"
torch.manual_seed(0)
m = Gemma4ForCausalLM(cfg).eval()
out = {}
with torch.no_grad():
    for L in (5, 13):
        x = torch.randint(0, raw["vocab_size"], (1, L))
        o = m(input_ids=x, use_cache=True)
        pre = cache_elems(o)["kv"]
        o2 = m(input_ids=x[:, -1:], past_key_values=o.past_key_values, use_cache=True)
        out[L] = {"prefill": pre, "decode": cache_elems(o2)["kv"]}
print("RESULT " + json.dumps(out))
'''


def test_gemma4_kv_against_the_real_cache():
    """The tiny K = V model's cache object (transformers 5.18 DynamicCache + sliding layers) holds exactly the
    headline (two tensors), at L = 5 (inside the window 8) and L = 13 (beyond), after prefill and after a decode."""
    from tests.test_universal_elephantine import TF5, TF5_PRELUDE
    if not os.path.isdir(TF5):
        pytest.skip("transformers 5.18 not unpacked (python -m elephantine.fetch_upstream_refs)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, root]))
    body = TF5_PRELUDE + f"RAW = {json.dumps(json.dumps(G4))}\n" + BODY
    r = subprocess.run([sys.executable, "-c", body], env=env, capture_output=True, text=True, timeout=900, cwd=root)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, r.stdout[-2000:] + r.stderr[-4000:]
    got = json.loads(line[7:])
    p = AM.plan(copy.deepcopy(G4))
    for L in (5, 13):
        for phase, n in (("prefill", L), ("decode", L + 1)):
            head = sum(AM.kv_at(p, n).values())
            assert got[str(L)][phase] == head, (L, phase, got, head)
            assert got[str(L)][phase] == sum(AM.kv_at(p, n, "hf").values())
            assert got[str(L)][phase] > sum(AM.kv_at(p, n, "minimum").values())
