"""Where a measured pair's max |delta log p| sits: the token, its target log-prob and rank, and the target's top-2 gap
at that position (is the 0.50 of plain:qwen15<-qwen2moe a near-tie, a tail token, or a construction error?).

  python -m moe.near_tie "plain:qwen15<-qwen2moe"
"""
from __future__ import annotations

import json
import sys

import torch

from moe import router_frontier as rf


def main(pair, n_prompts=24):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from hf.eps_lever_agreement import cont_logp, reference
    repo, host, kw = rf.PAIRS[pair]
    cfg = AutoConfig.from_pretrained(repo, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    tgt = AutoModelForCausalLM.from_pretrained(repo, dtype=rf.F32, attn_implementation="eager",
                                               local_files_only=True).eval()
    ref = reference(tgt, tok, "cpu", n_prompts)
    del tgt
    build = rf.qwen2moe_host if host == "qwen2moe" else rf.olmoe_host
    h = build(rf.lazy_get(repo), cfg, kw["x"], e=kw["e"], dtype=rf.F32)
    rows = []
    with torch.no_grad():
        for k, (ids, s, lq) in enumerate(zip(ref["enc"], ref["seq"], ref["lp"])):
            lp = cont_logp(h, s, ids.shape[1])
            d = (lp - lq).abs()
            pos, v = divmod(int(d.argmax()), d.shape[1])
            top2 = lq[pos].topk(2)
            rank = int((lq[pos] > lq[pos, v]).sum())
            # the same max restricted to tokens within 10 nats of the top (the ones greedy / top-1 can see)
            near = lq >= lq.max(-1, keepdim=True).values - 10
            rows.append(dict(prompt=k, max_dlogp=float(d.max()), pos=pos, token=v, rank=rank,
                             lq=float(lq[pos, v]), lp=float(lp[pos, v]), top2_gap=float(top2.values[0] - top2.values[1]),
                             max_dlogp_within_10nats=float(d[near].max()),
                             argmax_same=bool((lp.argmax(-1) == lq.argmax(-1)).all())))
            print(json.dumps(rows[-1]), flush=True)
    w = max(rows, key=lambda r: r["max_dlogp"])
    print("WORST", json.dumps(w))
    return rows


if __name__ == "__main__":
    main(sys.argv[1])
