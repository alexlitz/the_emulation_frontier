"""elephantine/emulate.py in memory: the relation gate (graph.edge_plan -- the pair must be an exact edge), the
construction plan and the host state dict, for small stock configs with random target weights."""
from __future__ import annotations

import torch

from verify.spot.catalog import EXACT_REL, F64, compare, hf_fn

V = 97
SMALL = dict(vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, max_position_embeddings=256, tie_word_embeddings=True,
             bos_token_id=1, eos_token_id=2, pad_token_id=None)
BIG = dict(vocab_size=V, hidden_size=96, intermediate_size=160, num_hidden_layers=3, num_attention_heads=6,
           num_key_value_heads=3, head_dim=32, max_position_embeddings=256, tie_word_embeddings=False,
           bos_token_id=1, eos_token_id=2, pad_token_id=None)


def raw(mt, kw):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model(mt, **kw)
    d = cfg.to_dict()
    d["architectures"] = [AutoModelForCausalLM._model_mapping[type(cfg)].__name__]
    d["model_type"] = mt
    return d


def model_from(d, seed=0, noise=0.08):
    from transformers import AutoModelForCausalLM
    from elephantine.emulate import resolved, warnings_off
    with warnings_off():
        torch.manual_seed(seed)
        m = AutoModelForCausalLM.from_config(resolved(d), attn_implementation="eager").to(F64).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * noise)
    return m


@torch.no_grad()
def emulate_pair(traw, hraw, seed=0):
    """(target model, host model, construction) -- emulate.gate refuses a pair that is not an exact edge."""
    import elephantine.emulate as E
    from transformers import AutoModelForCausalLM
    plan, con = E.gate(traw, hraw)
    tc = E.resolved(traw)
    target = model_from(traw, seed)
    T = E.read_target(target, tc, con.tfam)
    hdict = E.host_config_dict(hraw, con)
    sd = E.build_host(T, tc, hdict, con, log=lambda *a: None)
    with E.warnings_off():
        host = AutoModelForCausalLM.from_config(E.resolved(hdict), attn_implementation="eager").to(F64).eval()
    missing, unexpected = host.load_state_dict({k: v.to(F64) for k, v in sd.items()}, strict=False)
    assert not unexpected and all(k == "lm_head.weight" for k in missing), (missing, unexpected)
    host.tie_weights()
    return target, host, con, plan


def emu_check(tmt, tkw, hmt, hkw, tol=EXACT_REL, note="", seed=0, base_t=SMALL, base_h=BIG):
    import elephantine.emulate as E
    traw, hraw = raw(tmt, {**base_t, **tkw}), raw(hmt, {**base_h, **hkw})
    target, host, con, plan = emulate_pair(traw, hraw, seed)
    with E._lift():
        r = compare(hf_fn(target), hf_fn(host), V, tol=tol, note=note, seed=seed)
    r.extra["plan_slots"] = [f"{s['slot']} {s['target']}->{s['host']}" for s in plan["slots"]]
    return r
