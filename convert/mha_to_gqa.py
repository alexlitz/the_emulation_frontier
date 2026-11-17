"""Compile an MHA model into a GQA model (weight-level conversion, retrofit).

Ties the K,V projections into n_kv groups by averaging each group's per-head
blocks -> a GQA-shaped model with an n_head/n_kv smaller KV cache. This is a
LOSSY retrofit (a model trained as GQA avoids the loss); the script reports the
degradation and the cache saving so you can pick n_kv.

Run:
  python -m convert.mha_to_gqa --model gpt2 --n-kv 4
  python -m convert.mha_to_gqa --model gpt2 --n-kv 1 --save out/gpt2-mqa
"""
from __future__ import annotations

import argparse

import torch

import harness as H


def convert_gpt2_kv(model, n_kv):
    d = model.config.hidden_size
    n_head = model.config.num_attention_heads
    hd = d // n_head
    assert n_head % n_kv == 0, f"n_head={n_head} not divisible by n_kv={n_kv}"
    hpg = n_head // n_kv
    for blk in model.transformer.h:
        W, b = blk.attn.c_attn.weight.data, blk.attn.c_attn.bias.data   # (d,3d),(3d,)
        for base in (d, 2 * d):                      # K block, then V block
            Wk = W[:, base:base + d].view(d, n_head, hd)
            bk = b[base:base + d].view(n_head, hd)
            for g in range(n_kv):
                sl = slice(g * hpg, (g + 1) * hpg)
                Wk[:, sl, :] = Wk[:, sl, :].mean(1, keepdim=True)
                bk[sl, :] = bk[sl, :].mean(0, keepdim=True)


def run(model_name="gpt2", n_kv=4, save=None, max_length=128, num_samples=48):
    assert "gpt2" in model_name, "this converter implements gpt2's fused Conv1D QKV"
    model, tok, device = H.load(model_name, dtype=torch.float32)
    n_head = model.config.num_attention_heads
    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    print(f"compile {model_name}: MHA (n_head={n_head}) -> GQA (n_kv={n_kv}), "
          f"KV cache /{n_head / n_kv:g}")
    H.print_row(f"native (n_kv={n_head})", H.compare(base, base))
    convert_gpt2_kv(model, n_kv)
    new = H.logits_for(model, batches)
    H.print_row(f"converted n_kv={n_kv}", H.compare(base, new))
    print("reading: this is a retrofit -- averaging trained heads loses information. The")
    print("cache shrinks by n_head/n_kv; a model TRAINED as GQA pays none of this drift.")
    if save:
        model.save_pretrained(save)
        tok.save_pretrained(save)
        print(f"saved converted model to {save}  (note: still stores {n_head} KV heads; "
              "tie them structurally for the real memory win)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-kv", type=int, default=4)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    run(args.model, args.n_kv, args.save)
