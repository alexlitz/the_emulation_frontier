"""Kimi-Linear's KDA (Kimi Delta Attention) layers: params, recurrent state and FLOPs (arch_metrics._kda).

The formula is transcribed from the modelling code read as text (Hub modeling_kimi.py KimiDeltaAttention, and the
transformers 5.18 port). Checks: the Hub safetensors count of both snapshot repos matches exactly up to the named
router-bias buffers, and the transformers 5.18 class (local library code, the snapshot config, meta device)
holds the same parameters.
"""
import json
import os
import subprocess
import sys

import pytest

from graph import arch_metrics as AM
from graph.model_graph import iter_json_configs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "moonshotai/Kimi-Linear-48B-A3B-Base")
HUB_TOTAL = 49_122_681_728            # graph/model_meta.json hf_total (safetensors), both repos


@pytest.fixture(scope="module")
def cfgs():
    want = set(REPOS)
    return {k: v for k, v in iter_json_configs(os.path.join(HERE, "graph", "hub_10k.json")) if k in want}


def test_kda_layer_params(cfgs):
    p = AM.plan(cfgs[REPOS[0]])
    rec = [g for g in p["groups"] if g["kind"] == "rec"]
    assert len(rec) == 1 and rec[0]["n"] == 20 and rec[0]["rec"]["family"] == "kda"
    r = rec[0]["rec"]
    # h = 2304, H = 32, dk = 128 (q = 4096), K = 4: q / k / v / o 4 x 2304 x 4096, the two low-rank gates
    # 2 x (2304 x 128 + 128 x 4096), b_proj 2304 x 32; conv 3 x 4096 x 4, A_log 32, dt_bias 4096, o_norm 128
    assert r["W"] == 4 * 2304 * 4096 + 2 * (2304 * 128 + 128 * 4096) + 2304 * 32 == 39_460_864
    assert r["b"] == 3 * 4096 * 4 + 32 + 4096 + 128 == 53_408
    assert 20 * (r["W"] + r["b"]) == 790_285_440
    assert r["conv_state"] == 3 * 4096 * 4 and r["ssm_state"] == 32 * 128 * 128
    assert "rec" not in p.get("unknown", set())


def test_kimi_linear_matches_the_hub(cfgs):
    for name in REPOS:
        raw = cfgs[name]
        p = AM.plan(raw)
        m = AM.metrics(raw, HUB_TOTAL, p=p)
        assert m["params"]["total"] == 49_122_675_072
        # the only difference: e_score_correction_bias, one 256-entry buffer per MoE layer (26 of them)
        assert HUB_TOTAL - m["params"]["total"] == 26 * 256
        assert AM.hub_reason(p, raw, m["params"]["total"], HUB_TOTAL, name) == ("hub-matched", "router bias buffers")
        assert m["params"]["active"] == 3_484_453_248              # +790M KDA over the old 2.69B
        assert m["state"]["elems"] == 20 * (3 * 4096 * 4 + 32 * 128 * 128)
        assert m["flops"]["matmul"] == 4_633_100_288 + 2 * 20 * 39_460_864
        assert not any("not modelled" in n for n in m["notes"])


BODY = '''
import json
from transformers.models.kimi_linear.configuration_kimi_linear import KimiLinearConfig
from transformers.models.kimi_linear.modeling_kimi_linear import KimiLinearForCausalLM
raw = json.loads(RAW)
cfg = KimiLinearConfig(**{k: v for k, v in raw.items() if k not in ("model_type", "architectures", "auto_map")})
with torch.device("meta"):
    m = KimiLinearForCausalLM(cfg)
print("RESULT " + json.dumps({"params": sum(p.numel() for p in m.parameters()),
                              "buffers": sum(b.numel() for n, b in m.named_buffers() if "e_score" in n)}))
'''


def test_kimi_linear_against_the_stock_class(cfgs):
    from tests.test_universal_elephantine import TF5, TF5_PRELUDE
    if not os.path.isdir(TF5):
        pytest.skip("transformers 5.18 not unpacked (python -m elephantine.fetch_upstream_refs)")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, HERE]))
    body = TF5_PRELUDE + f"RAW = {json.dumps(json.dumps(cfgs[REPOS[0]]))}\n" + BODY
    r = subprocess.run([sys.executable, "-c", body], env=env, capture_output=True, text=True, timeout=900, cwd=HERE)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, r.stdout[-2000:] + r.stderr[-4000:]
    got = json.loads(line[7:])
    assert got["params"] == AM.metrics(cfgs[REPOS[0]])["params"]["total"]
    assert got["params"] + got["buffers"] == HUB_TOTAL
