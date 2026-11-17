"""graph/model_graph.py REGIME_PAYABLE against the measured rows (attn/data/regime_agreement.jsonl, written by
attn/regime_agreement.py): the act-swap standard, payable = top-1 >= 99% and >= 90% of 32-token greedy
continuations identical (fp32 rows; the @bf16 rows are controls)."""
import json

from graph.model_graph import QKNORM_VARIANTS, REGIME_PAYABLE

ROWS = [json.loads(line) for line in open("attn/data/regime_agreement.jsonl")]
PASS = lambda r: r["top1"] >= 0.99 and r["ex32"] >= 0.90


def rows(variant):
    return [r for r in ROWS if r["variant"].split(":")[0] == variant and "@" not in r["variant"]]


def test_attention_capped_target_on_a_plain_host_is_refused():
    # Gemma-2-9B fails without its attention cap (its scores reach 66 > c = 50); the 2B models pass narrowly
    nocap = {r["model"]: PASS(r) for r in rows("nocap_attn")}
    assert nocap["google/gemma-2-9b"] is False and nocap["google/gemma-2-2b"] is True
    assert REGIME_PAYABLE[("softcap", "attn", "no")] is False and REGIME_PAYABLE[("softcap", "attn", "final")] is False


def test_final_cap_dropped_is_argmax_exact():
    assert all(PASS(r) and r["top1"] == 1.0 for r in rows("nocap_final"))
    assert REGIME_PAYABLE[("softcap", "final", "no")] is True


def test_capped_host_for_a_plain_target_pays_with_a_large_cap():
    plain = {r["model"] for r in rows("cap_attn")}
    assert len(plain) == 6
    for m in plain:                                   # some cap c passes, and c >= 5 x the target's max |z| does
        caps = sorted((float(r["variant"].split(":")[1]), PASS(r)) for r in rows("cap_attn") if r["model"] == m)
        zmax = next(r["ranges"]["zmax"] for r in ROWS if r["model"] == m and r["variant"] == "self")
        assert any(ok for _, ok in caps), m
        assert all(ok for c, ok in caps if c >= 5 * zmax), m
    for c in ("15", "30"):                            # a final cap of 15 / 30: 100% / 100% on all six
        assert all(PASS(r) for r in rows("cap_final") if r["variant"].endswith(":" + c))
    assert REGIME_PAYABLE[("softcap", "no", "attn")] and REGIME_PAYABLE[("softcap", "no", "final")]
    assert REGIME_PAYABLE[("softcap", "final", "attn")]


def test_qknorm_regime_fails_both_ways():
    qk = rows("qkfold") + rows("qkfold_story") + rows("qkhost_head") + rows("qkhost_joint")
    assert qk and not any(PASS(r) for r in qk)
    assert all(not REGIME_PAYABLE[("qknorm", v, "no")] and not REGIME_PAYABLE[("qknorm", "no", v)]
               for v in QKNORM_VARIANTS)
