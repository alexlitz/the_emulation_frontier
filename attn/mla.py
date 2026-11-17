"""MLA (multi-head latent attention)  vs  MHA.

MLA is MHA whose per-head K and V are LOW-RANK: every head reads keys/values
from one shared latent c = x W_DKV (dim kv_lora_rank), up-projected per head.
So MLA is a structured restriction of MHA -> MLA subset of MHA, EXACT. The
"absorption" identity is why the cache stays small:

    q_C . k_C^h = q_C . (W_UK^h c) = (W_UK^h^T q_C) . c          (score)
    sum_j a_j v^h_j = (A c) W_UV^h                               (output)

i.e. you attend over the shared latent c directly; per-head K,V are never
materialized in the cache. Currency: KV memory (MLA caches c + rope key; a full
MHA caches per-head K,V). Making MLA into full MHA means latent rank = full ->
compression gone. Conserved, like MQA<->MHA.

Hierarchy: MQA subset GQA subset MLA subset MHA  (MLA generalizes GQA: TransMLA).

Part 1 verifies the absorption identity at DeepSeek-V2-Lite's real config dims.
Part 2 measures the reverse cost on a real trained MHA (gpt2): force K,V through
a shared rank-r latent and sweep r.

Run:
  python -m attn.mla                 # identity at real MLA dims (synthetic weights)
  python -m attn.mla --gpt2          # + low-rank KV retrofit on gpt2
"""
from __future__ import annotations

import argparse
import contextlib
import math

import torch
import torch.nn.functional as F

import harness as H

# DeepSeek-V2-Lite MLA dims
CFG = dict(hidden=2048, n_head=16, qk_nope=128, qk_rope=64, v_head=128, kv_lora=512)


def verify_absorption(seed=0):
    c = CFG
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, dtype=torch.float64, generator=g)
    T = 24
    x = rand(T, c["hidden"])

    W_DKV = rand(c["hidden"], c["kv_lora"])                 # down-proj to latent
    W_UK = rand(c["kv_lora"], c["n_head"] * c["qk_nope"])   # latent -> per-head nope key
    W_UV = rand(c["kv_lora"], c["n_head"] * c["v_head"])    # latent -> per-head value
    W_QC = rand(c["hidden"], c["n_head"] * c["qk_nope"])    # query (nope part)

    latent = x @ W_DKV                                      # (T, kv_lora)  <-- the whole cache
    scale = 1.0 / math.sqrt(c["qk_nope"])
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=torch.float64), 1)

    max_score_err = max_out_err = 0.0
    for h in range(c["n_head"]):
        UKh = W_UK[:, h * c["qk_nope"]:(h + 1) * c["qk_nope"]]   # (kv_lora, qk_nope)
        UVh = W_UV[:, h * c["v_head"]:(h + 1) * c["v_head"]]     # (kv_lora, v_head)
        qC = x @ W_QC[:, h * c["qk_nope"]:(h + 1) * c["qk_nope"]]  # (T, qk_nope)

        # explicit MHA: materialize per-head key/value from the latent
        kC = latent @ UKh                                    # (T, qk_nope)
        v = latent @ UVh                                     # (T, v_head)
        score_explicit = (qC @ kC.T) * scale
        A = F.softmax(score_explicit + mask, dim=-1)
        out_explicit = A @ v                                 # (T, v_head)

        # absorbed MLA: fold up-projections; attend over the latent directly
        q_abs = qC @ UKh.T                                   # (T, kv_lora)
        score_abs = (q_abs @ latent.T) * scale               # == score_explicit
        out_abs = (A @ latent) @ UVh                         # == out_explicit

        max_score_err = max(max_score_err, (score_explicit - score_abs).abs().max().item())
        max_out_err = max(max_out_err, (out_explicit - out_abs).abs().max().item())

    kv_mla = c["kv_lora"] + c["qk_rope"]
    kv_mha = c["n_head"] * (c["qk_nope"] + c["qk_rope"] + c["v_head"])
    print("MLA absorption identity at DeepSeek-V2-Lite dims (synthetic weights):")
    print(f"  max score error = {max_score_err:.3e}   max output error = {max_out_err:.3e}   (exact)")
    print(f"  KV/token: MLA={kv_mla}  equiv-MHA={kv_mha}  compression ~{kv_mha / kv_mla:.1f}x")
    print("  -> MLA subset MHA: per-head K,V are a shared low-rank latent; exact reparam\n")


# --- reverse cost on a real trained MHA: low-rank shared-latent KV (gpt2) ------

@contextlib.contextmanager
def lowrank_kv(model, rank):
    """Force K,V through a shared rank-r latent (MLA-style) in every gpt2 layer."""
    d = model.config.hidden_size
    saved = []
    for blk in model.transformer.h:
        cattn = blk.attn.c_attn
        saved.append((cattn, cattn.weight.data.clone()))
        W = cattn.weight.data
        M = torch.cat([W[:, d:2 * d], W[:, 2 * d:3 * d]], dim=1).float()  # (d, 2d) stacked K|V
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        Mr = (U[:, :rank] * S[:rank]) @ Vt[:rank]                          # shared rank-r latent
        W[:, d:2 * d] = Mr[:, :d].to(W.dtype)
        W[:, 2 * d:3 * d] = Mr[:, d:].to(W.dtype)
    try:
        yield
    finally:
        for cattn, Ws in saved:
            cattn.weight.data.copy_(Ws)


def run_gpt2(max_length=128, num_samples=32):
    model, tok, device = H.load("gpt2", dtype=torch.float32)
    d = model.config.hidden_size
    full = 2 * d  # K+V columns cached in vanilla MHA
    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    print(f"low-rank KV retrofit on gpt2 (full K+V dim = {full}):")
    H.print_row(f"native (rank {full})", H.compare(base, base))
    for r in [512, 256, 128, 64, 32]:
        with lowrank_kv(model, r):
            logits = H.logits_for(model, batches)
        H.print_row(f"rank={r} (KV /{full / r:g})", H.compare(base, logits))
    print("reading: the low-rank identity is exact by construction (part 1); this is "
          "the *retrofit* cost of imposing it on a model trained full-rank. A model "
          "trained as MLA pays none of this -- it just gets the ~9x smaller cache.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt2", action="store_true", help="also run the gpt2 retrofit sweep")
    args = ap.parse_args()
    verify_absorption()
    if args.gpt2:
        run_gpt2()
