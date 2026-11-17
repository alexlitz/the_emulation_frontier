"""Attention subset: MHA  vs  MQA / GQA.

Two directions, both EXACT when shape is allowed to vary -- the penalty is
KV-cache memory, and it is *conserved* (whichever direction you emulate, you
pay back exactly the memory the other side saved).

  downward  MHA -> GQA(n_kv) -> MQA:  tie K,V projection blocks within each
            group. This is a strict restriction of a *trained* MHA, so the
            agreement drop measures the expressivity the child cannot hold.
            KV cache shrinks by n_head/n_kv -- that is the child's whole point.

  upward    MQA -> MHA (your construction, verified numerically on real gpt2
            weights): widen the single shared KV head to H*head_dim (concat all
            heads' key blocks), give query head h a query that is W_q^(h)x in
            block h and zero elsewhere so q_h . K_shared selects head h's score,
            share the wide V, and let W_o keep only the diagonal (h,h) blocks.
            Reproduces MHA EXACTLY. Cost: the shared KV head is now H*head_dim,
            i.e. the KV cache is back to MHA size -- MQA's saving is exactly
            cancelled. No expressivity gap; a conserved memory exchange.

Run:
  python -m attn.mha_mqa_gqa --model gpt2
"""
from __future__ import annotations

import argparse
import contextlib
import math

import torch
import torch.nn.functional as F

import harness as H


# --------------------------------------------------------------------------
# downward: collapse a trained MHA into GQA/MQA by group-averaging K,V blocks
# --------------------------------------------------------------------------

def _gpt2_kv_slices(model):
    d = model.config.hidden_size
    H_ = model.config.num_attention_heads
    hd = d // H_
    return d, H_, hd


@contextlib.contextmanager
def collapsed_kv(model, n_kv):
    """Tie K and V projections into n_kv groups across all gpt2 layers.

    Heads in a group share the group-averaged K,V projection -> a GQA-shaped
    function (n_kv=1 is MQA). Restores original weights on exit.
    """
    d, n_head, hd = _gpt2_kv_slices(model)
    assert n_head % n_kv == 0, f"n_head={n_head} not divisible by n_kv={n_kv}"
    hpg = n_head // n_kv  # heads per group
    saved = []
    for blk in model.transformer.h:
        cattn = blk.attn.c_attn
        Wsaved, bsaved = cattn.weight.data.clone(), cattn.bias.data.clone()
        saved.append((cattn, Wsaved, bsaved))
        W, b = cattn.weight.data, cattn.bias.data  # W: (d, 3d); K=[:,d:2d] V=[:,2d:3d]
        for base in (d, 2 * d):  # K block start, then V block start
            Wk = W[:, base:base + d].view(d, n_head, hd)
            bk = b[base:base + d].view(n_head, hd)
            for g in range(n_kv):
                sl = slice(g * hpg, (g + 1) * hpg)
                Wk[:, sl, :] = Wk[:, sl, :].mean(dim=1, keepdim=True)
                bk[sl, :] = bk[sl, :].mean(dim=0, keepdim=True)
    try:
        yield
    finally:
        for cattn, Ws, bs in saved:
            cattn.weight.data.copy_(Ws)
            cattn.bias.data.copy_(bs)


def run_downward(model, batches, base):
    d, n_head, hd = _gpt2_kv_slices(model)
    print(f"downward  (MHA -> GQA -> MQA), n_head={n_head}, head_dim={hd}")
    H.print_row(f"native (n_kv={n_head})", H.compare(base, base))
    n_kv = n_head
    while n_kv >= 1:
        if n_head % n_kv == 0 and n_kv != n_head:
            with collapsed_kv(model, n_kv):
                logits = H.logits_for(model, batches)
            m = H.compare(base, logits)
            ratio = n_head / n_kv
            label = f"n_kv={n_kv} (KV /{ratio:g})" + ("  [MQA]" if n_kv == 1 else "")
            H.print_row(label, m)
        n_kv //= 2


# --------------------------------------------------------------------------
# upward: MQA (single wide shared KV head) reproduces MHA exactly
# --------------------------------------------------------------------------

def _causal(T, device):
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


def verify_mqa_to_mha_exact(model, T=32, seed=0):
    d, n_head, hd = _gpt2_kv_slices(model)
    dev = next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(T, d, generator=g).to(dev, torch.float64)   # fp64 -> truly exact

    attn = model.transformer.h[0].attn
    W = attn.c_attn.weight.data.double()  # (d, 3d)
    b = attn.c_attn.bias.data.double()    # (3d,)
    Wq, Wk, Wv = W[:, :d], W[:, d:2 * d], W[:, 2 * d:]
    bq, bk, bv = b[:d], b[d:2 * d], b[2 * d:]
    Wo = attn.c_proj.weight.data.double()  # (d, d)
    bo = attn.c_proj.bias.data.double()
    mask = _causal(T, dev)
    scale = 1.0 / math.sqrt(hd)

    # --- standard MHA: H independent KV heads ---
    q = (X @ Wq + bq).view(T, n_head, hd)
    k = (X @ Wk + bk).view(T, n_head, hd)
    v = (X @ Wv + bv).view(T, n_head, hd)
    out = torch.empty(T, n_head, hd, device=dev, dtype=torch.float64)
    for h in range(n_head):
        a = F.softmax((q[:, h] @ k[:, h].T) * scale + mask, dim=-1)
        out[:, h] = a @ v[:, h]
    mha_out = out.reshape(T, d) @ Wo + bo

    # --- MQA: ONE shared KV head of width m = H*hd (= d here) ---
    m = n_head * hd
    K_big = X @ Wk + bk           # (T, m) shared across all query heads
    V_big = X @ Wv + bv           # (T, m) shared
    sel_blocks = []
    for h in range(n_head):
        q_big = torch.zeros(T, m, device=dev, dtype=torch.float64)
        q_big[:, h * hd:(h + 1) * hd] = X @ Wq[:, h * hd:(h + 1) * hd] + bq[h * hd:(h + 1) * hd]
        a = F.softmax((q_big @ K_big.T) * scale + mask, dim=-1)  # == head-h scores
        o_big = a @ V_big                                        # (T, m)
        sel_blocks.append(o_big[:, h * hd:(h + 1) * hd])         # W_o keeps diagonal block
    mqa_out = torch.cat(sel_blocks, dim=-1) @ Wo + bo

    err = (mha_out - mqa_out).abs().max().item()
    kv_mha = 2 * n_head * hd
    kv_mqa_compact = 2 * hd
    kv_mqa_wide = 2 * m
    print("upward  (MQA -> MHA, your wide-shared-KV construction, gpt2 layer 0)")
    print(f"  max |MHA - MQA_wide| = {err:.3e}   (exact reproduction)")
    print(f"  KV/token: MHA={kv_mha}  compact-MQA={kv_mqa_compact}  "
          f"wide-MQA(emulating MHA)={kv_mqa_wide}")
    print(f"  exchange rate: emulation blows MQA's KV cache by "
          f"{kv_mqa_wide // kv_mqa_compact}x = n_head -> exactly cancels MQA's saving\n")
    return err


def run(model_name="gpt2", max_length=128, num_samples=64):
    assert "gpt" in model_name, "this scaffold implements the fused-QKV surgery for gpt2"
    model, tok, device = H.load(model_name, dtype=torch.float32)
    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    print(f"{model_name}: {len(batches)} contexts, "
          f"~{sum(b.shape[1] for b in batches)} positions\n")
    verify_mqa_to_mha_exact(model)
    run_downward(model, batches, base)
    print("\nreading: upward is exact (fp noise) at fixed function, paid in KV memory. "
          "downward degradation = expressivity a trained MHA loses when squeezed into "
          "fewer KV heads (this is a *retrofit* penalty; a model trained as GQA avoids it).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=64)
    args = ap.parse_args()
    run(args.model, args.max_length, args.num_samples)
