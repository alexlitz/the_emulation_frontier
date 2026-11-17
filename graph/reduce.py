"""CLI: test the emulation reduction between two models.

  python -m graph.reduce Qwen1.5-72B Llama-3.1-405B
  python -m graph.reduce mistralai/Mistral-7B-v0.1 microsoft/phi-2 --verify
  python -m graph.reduce deepseek-ai/DeepSeek-V2 Qwen1.5-72B --dtype fp64

Reports, for each direction, whether B can emulate A under the HEADLINE relation (capacity +
every type wall that has a currency; a slot with no currency is a WALL): the differing
architectural slots and the currency each costs, the per-axis capacity gaps (with the TransMLA /
MoE-depth / block / mirror-norm penalties and the zeroing-edge depth), and a verdict -- plus the
secondary "if every type wall were free" (capacity-only) verdict. With --verify it also runs the numerical
identity for each differing slot, in the chosen precision (fp32 by default -- model
precision), so you can watch the reduction hold.

Models resolve by exact id or case-insensitive substring against the cached config
set (graph/hub_10k.json), falling back to a live AutoConfig fetch for anything else.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from graph.model_graph import (FFN_RANK, MLA_PENALTY, MOE_DEPTH_FACTOR, NORM_WIDTH_FACTOR, SLOT_DEFAULT, SLOTS,
                               ZERO_EDGES, classify, dims_of, edge_cost, emulates, iter_json_configs,
                               slot_str)

DTYPES = {"fp32": torch.float32, "fp64": torch.float64, "bf16": torch.bfloat16}


# --- model resolution --------------------------------------------------------

def load_index(path):
    return dict(iter_json_configs(path))


def resolve(query, index):
    if query in index:
        return query, index[query]
    q = query.lower()
    hits = sorted((m for m in index if q in m.lower()), key=lambda m: (len(m), m))
    if hits:
        return hits[0], index[hits[0]]
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(query, trust_remote_code=False).to_dict()  # never run Hub code
    return query, cfg


# --- the size-aware axis report (mirrors emulates(), but explains itself) -----

def axis_gaps(A, B):
    """per-axis (need, have, ok) for 'B emulates A', with the same penalties."""
    da, db, sa, sb = A["dims"], B["dims"], A["slots"], B["slots"]
    kv_pen = MLA_PENALTY if (sa["kv"] == "MLA" and "MLA" not in str(sb["kv"]).split("|")) else 1
    lf = MOE_DEPTH_FACTOR if FFN_RANK[sb["ffn"]] < FFN_RANK[sa["ffn"]] else 1
    wf = 1
    if sa["block"] != sb["block"]:
        if sb["block"] == "sequential":
            wf = 2
        else:
            lf = max(lf, MOE_DEPTH_FACTOR)
    if sa["norm"] == "RMSNorm" and sb["norm"] == "LayerNorm":
        wf *= NORM_WIDTH_FACTOR                       # the [h; -h] mirror
    # ZERO_EDGES: a plain softmax target on a hybrid / sliding / KV-sharing host uses only the host's
    # layers of its kind (full-window ones for a full target)
    plain = lambda x: x["mixer"] == "softmax" and x["crosslayer_kv"] == "no"
    usable, usable_p = db["eff_layers"], db["param_layers"]
    if plain(sa) and (not plain(sb) or (sa["window"] == "full" and sb["window"] == "local")):
        kind = "full" if sa["window"] == "full" else "attn"
        usable, usable_p = db.get(f"{kind}_eff", usable), db.get(f"{kind}_param", usable_p)
    rows = [
        ("kv_cap", da["kv_cap"] * kv_pen, db["kv_cap"]),
        ("q_cap", da["q_cap"], db["q_cap"]),
        ("qheads", da["qheads"], db["qheads"]),
        ("ffn_total", da["ffn_total"], db["ffn_total"]),
        ("ffn_active", da["ffn_active"], db["ffn_active"]),
        ("eff_layers", lf * da["eff_layers"], usable),
        ("param_layers", da["param_layers"], usable_p),
        ("hidden", wf * da["hidden"], db["hidden"]),
    ]
    return [(n, need, have, have >= need) for n, need, have in rows]


def cat_diffs(A, B):
    """categorical slot differences and the currency to climb A_slot -> B_slot."""
    out = []
    for slot, order in SLOTS:
        a, b = (A["slots"].get(slot, SLOT_DEFAULT.get(slot)), B["slots"].get(slot, SLOT_DEFAULT.get(slot)))   # (round 3)
        if a == b:
            continue
        cost = edge_cost(order, a, b)                 # None if a can't reach b
        out.append((slot, a, b, cost))
    return out


# --- numerical demos: run each differing slot's identity in the chosen dtype --

def _r(dt, *s):
    return torch.randn(*s, dtype=dt, generator=torch.Generator().manual_seed(0))


def demo_norm(dt):                                     # LayerNorm = center + RMSNorm + bias
    T, d = 32, 48
    x, w, b = _r(dt, T, d), _r(dt, d), _r(dt, d)
    ref = F.layer_norm(x, (d,), w, b, 1e-5)
    c = x - x.mean(-1, keepdim=True)
    got = c * torch.rsqrt((c ** 2).mean(-1, keepdim=True) + 1e-5) * w + b
    return (ref - got).abs().max().item()


def demo_kv(dt):                                       # wide shared KV == MHA
    T, H, hd = 16, 4, 12
    d = H * hd
    X, Wq, Wk, Wv, Wo = (_r(dt, *s) for s in [(T, d), (d, d), (d, d), (d, d), (d, d)])
    sc = 1 / math.sqrt(hd)
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=dt), 1)
    q = (X @ Wq).view(T, H, hd); k = (X @ Wk).view(T, H, hd); v = (X @ Wv).view(T, H, hd)
    o = torch.stack([F.softmax((q[:, h] @ k[:, h].T) * sc + mask, -1) @ v[:, h] for h in range(H)], 1)
    mha = o.reshape(T, d) @ Wo
    Kb, Vb, sel = X @ Wk, X @ Wv, []
    for h in range(H):
        qb = torch.zeros(T, d, dtype=dt); qb[:, h * hd:(h + 1) * hd] = X @ Wq[:, h * hd:(h + 1) * hd]
        sel.append((F.softmax((qb @ Kb.T) * sc + mask, -1) @ Vb)[:, h * hd:(h + 1) * hd])
    return (mha - torch.cat(sel, -1) @ Wo).abs().max().item()


def demo_ffn(dt):                                      # dense embeds in MoE (tie experts)
    T, d, ew = 16, 16, 24
    x, W1, W2 = _r(dt, T, d), _r(dt, d, ew), _r(dt, ew, d)
    dense = F.silu(x @ W1) @ W2
    moe = sum((1 / 2) * (F.silu(x @ W1) @ (2 * W2)) for _ in range(1))  # k=1 tied
    return (dense - moe).abs().max().item()


def demo_act(dt):                                      # ReLU = beta-limit of Swish
    x = torch.linspace(-6, 6, 2001, dtype=dt)
    return (x * torch.sigmoid(256 * x) - F.relu(x)).abs().max().item()


def demo_window(dt):                                   # local = full + band mask
    T, d, W = 16, 8, 4
    q, k, v = _r(dt, T, d), _r(dt, T, d), _r(dt, T, d)
    i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    band = torch.where((j <= i) & (j > i - W), 0.0, float("-inf")).to(dt)
    a = F.softmax(q @ k.T / math.sqrt(d) + band, -1) @ v
    ref = torch.stack([F.softmax(q[t] @ k[max(0, t - W + 1):t + 1].T / math.sqrt(d), -1)
                       @ v[max(0, t - W + 1):t + 1] for t in range(T)])
    return (a - ref).abs().max().item()


def demo_block(dt):                                    # sequential == two parallel blocks
    T, d, m = 8, 64, 128
    x, n1, n2, Wa, W1, W2 = (_r(dt, *s) for s in [(T, d), (d,), (d,), (d, d), (d, m), (m, d)])
    rms = lambda z, w: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    A = lambda z: rms(z, n1) @ Wa
    M = lambda z: F.silu(rms(z, n2) @ W1) @ W2
    seq = x + A(x) + M(x + A(x))
    b1 = x + A(x); b2 = b1 + M(b1)
    return (seq - b2).abs().max().item()


def demo_sink(dt):                                     # sink == standard + null token
    s, V = _r(dt, 32), _r(dt, 32, 16)
    b = 2.0
    ysink = (torch.exp(s) @ V) / (torch.exp(s).sum() + math.exp(b))
    se = torch.cat([s, torch.tensor([b], dtype=dt)]); Vn = torch.cat([V, torch.zeros(1, 16, dtype=dt)])
    return (ysink - F.softmax(se, 0) @ Vn).abs().max().item()


def demo_loop(dt):                                     # looped block == unrolled tied stack
    T, d = 8, 16
    x, W = _r(dt, T, d), _r(dt, d, d)
    step = lambda h: h + torch.tanh(h @ W)
    looped = x
    for _ in range(6):
        looped = step(looped)
    untied = x
    for _ in range(6):                                 # 6 tied layers == 6 loop iters
        untied = step(untied)
    return (looped - untied).abs().max().item()


DEMOS = {"norm": demo_norm, "kv": demo_kv, "ffn": demo_ffn, "act": demo_act,
         "window": demo_window, "block": demo_block, "sink": demo_sink, "loop": demo_loop}


# --- driver ------------------------------------------------------------------

def describe(name, m):
    d = m["dims"]
    return (f"{name}\n    {slot_str(m['slots'])}\n    "
            f"h{d['hidden']} L{d['layers']} kv{d['kv_cap']} q{d['q_cap']} "
            f"heads{d['qheads']} ffnT{d['ffn_total']} ffnA{d['ffn_active']}")


SIZE_SLOTS = {"kv", "ffn", "block", "loop"}   # handled by capacity, not categorically


def direction(A, B, an, bn, verify, dt):
    print(f"\n=== does  {bn}  emulate  {an} ? ===")
    diffs = cat_diffs(A, B)
    for s, a, b, c in diffs:
        if s in SIZE_SLOTS:
            print(f"  slot {s}: {a} vs {b}   size-based (decided by the capacity rows below)")
        elif (s, a, b) in ZERO_EDGES and A["slots"]["mixer"] == "softmax" and A["slots"]["crosslayer_kv"] == "no":
            print(f"  slot {s}: {a} <- {b}   currency: {ZERO_EDGES[(s, a, b)]} (exact; eff_layers row below)")
        elif c is None:
            print(f"  slot {s}: {a} ≠ {b}   WALL (no currency: blocks under the headline)")
        else:
            cur, exact = c
            uniq = ",".join(dict.fromkeys(cur))
            print(f"  slot {s}: {a} -> {b}   currency: {uniq}{' (approx)' if not exact else ''}")
    gaps = axis_gaps(A, B)
    for n, need, have, ok in gaps:
        if need != have:
            print(f"  {n:13} need {need:>9}  have {have:>9}  {'ok' if ok else 'FAIL'}")
    ok = emulates(B, A)
    cap = emulates(B, A, mode="capacity")
    print(f"  => {'YES' if ok else 'NO'}: {bn} {'can' if ok else 'cannot'} emulate {an}"
          f"   [if every type wall were free: {'YES' if cap else 'NO'}]")
    if verify:
        slots = {s for s, _, _, c in diffs if c}
        for s in [x for x in DEMOS if x in slots]:
            err = DEMOS[s](dt)
            print(f"     verify[{s}] identity in {dt}: max|Δ| = {err:.2e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_a")
    ap.add_argument("model_b")
    ap.add_argument("--from-json", default="graph/hub_10k.json")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--verify", action="store_true",
                    help="run the numerical identity for each differing slot")
    args = ap.parse_args()
    index = load_index(args.from_json)
    an, ca = resolve(args.model_a, index)
    bn, cb = resolve(args.model_b, index)
    A = {"slots": classify(ca), "dims": dims_of(ca)}
    B = {"slots": classify(cb), "dims": dims_of(cb)}
    if A["slots"] is None or B["slots"] is None:
        raise SystemExit("one of the models did not classify as a decoder LM")
    dt = DTYPES[args.dtype]
    print(describe("A = " + an, A))
    print(describe("B = " + bn, B))
    a2b = direction(A, B, an, bn, args.verify, dt)
    b2a = direction(B, A, bn, an, args.verify, dt)
    print("\n=== verdict ===")
    if a2b and b2a:
        print(f"  {an} ≡ {bn}  (mutual: identical capacity + comparable slots)")
    elif a2b:
        print(f"  {an} ⊆ {bn}  ({bn} is the more general)")
    elif b2a:
        print(f"  {bn} ⊆ {an}  ({an} is the more general)")
    else:
        print(f"  {an} and {bn} are INCOMPARABLE (neither emulates the other)")


if __name__ == "__main__":
    main()
