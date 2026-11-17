"""At one (prompt, position) of a measured pair: per MoE layer the target's router margin between its k-th and (k+1)-th
expert and whether the host selected the same experts -- a top-k boundary near-tie flipped by fp32 rounding (the
host's sums run over a padded width) shows as a tiny margin at the one layer that differs.

  python -m moe.router_gap "plain:qwen15<-qwen2moe" 2 59
"""
from __future__ import annotations

import gc
import json
import sys

import torch

from moe import router_frontier as rf


def _router_logits(model, seq):
    out = {}
    hooks = []
    for i, L in enumerate(model.model.layers):
        gate = getattr(L.mlp, "gate", None)
        if gate is None:
            continue
        hooks.append(gate.register_forward_hook(lambda m, a, o, i=i: out.__setitem__(i, o.detach().double())))
    with torch.no_grad():
        model(seq)
    for h in hooks:
        h.remove()
    return out


def main(pair, prompt, pos):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from act.act_swap_agreement import PROMPTS, greedy
    repo, host, kw = rf.PAIRS[pair]
    cfg = AutoConfig.from_pretrained(repo, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    tgt = AutoModelForCausalLM.from_pretrained(repo, dtype=rf.F32, attn_implementation="eager",
                                               local_files_only=True).eval()
    ids = tok(PROMPTS[prompt], return_tensors="pt").input_ids
    with torch.no_grad():
        seq = torch.cat([ids, greedy(tgt, ids).cpu()], 1)
    P = ids.shape[1]
    t_log = _router_logits(tgt, seq)
    del tgt
    gc.collect()
    build = rf.qwen2moe_host if host == "qwen2moe" else rf.olmoe_host
    h = build(rf.lazy_get(repo), cfg, kw["x"], e=kw["e"], dtype=rf.F32)
    h_log = _router_logits(h, seq)
    k, E = cfg.num_experts_per_tok, cfg.num_experts
    row = P - 1 + pos                                  # the token predicting continuation position pos
    flips = []
    for t_pos in range(row + 1):
        res = []
        for i in sorted(t_log):
            zt, zh = t_log[i][t_pos], h_log[i][t_pos][:E]
            st = set(zt.topk(k).indices.tolist())
            sh = set(h_log[i][t_pos].topk(k).indices.tolist())
            srt = zt.sort(descending=True).values
            res.append(dict(layer=i, margin=float(srt[k - 1] - srt[k]), same=st == sh,
                            max_dlogit=float((zt - zh).abs().max())))
        flips += [dict(r, token_row=t_pos) for r in res if not r["same"]]
        if t_pos == row:
            print(json.dumps(dict(at_row=row, min_margin=min(r["margin"] for r in res),
                                  max_router_dlogit=max(r["max_dlogit"] for r in res))))
    print(json.dumps(dict(pair=pair, prompt=prompt, pos=pos, selection_flips=flips)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
