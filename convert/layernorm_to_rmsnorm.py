"""Compile a LayerNorm model into an RMSNorm model (weight-level conversion).

Two differences between LN and RMSNorm:
  * bias beta:  LN(x) = gamma*norm(x) + beta.  beta is EXACTLY foldable into the
    consuming linear layer's bias:  W(gamma*norm + beta) = W(gamma*norm) + W beta
    (the final norm's beta into an lm_head bias).
  * centering:  LN centers (x-mean); RMSNorm does not. The centring folds too: post-compose
    every residual WRITER (embedding, attention out, MLP out) with P = I - 11^T/d, so the
    residual stays mean-zero and RMSNorm(P x) = LayerNorm's normalisation (Lean
    layerNorm_sub_rmsNorm; norm/layernorm_rmsnorm.fold_layernorm_model).

convert_pythia now runs that exact fold (round 2, pass 5: verify/BRIDGES.md MISMATCH 1 -- it used to fold
beta only and drop the centring, a different, lossy function it called "the one irreducible penalty").
convert_pythia_beta_only keeps the old lossy conversion for the comparison.

Run:
  python -m convert.layernorm_to_rmsnorm --model EleutherAI/pythia-160m
"""
from __future__ import annotations

import argparse
import types

import torch

import harness as H


def _rmsify(ln):
    w, eps = ln.weight, ln.eps

    def fwd(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        return y.to(x.dtype)

    ln.forward = types.MethodType(fwd, ln)


def _fold_beta_into(consumer, beta):
    add = consumer.weight.data @ beta                    # nn.Linear: out += W beta
    if consumer.bias is None:
        consumer.bias = torch.nn.Parameter(add)
    else:
        consumer.bias.data += add


def convert_pythia(model):
    """In place, exactly: the LayerNorm -> RMSNorm fold with the centring folded into every writer (and an
    lm_head bias for the final norm's beta): norm/layernorm_rmsnorm.fold_layernorm_model."""
    from norm.layernorm_rmsnorm import fold_layernorm_model
    assert hasattr(model, "gpt_neox"), "convert_pythia takes a GPTNeoXForCausalLM"
    return fold_layernorm_model(model)


def convert_pythia_beta_only(model):
    """The old conversion: beta folded into the consumers, the centring DROPPED (a different, lossy function)."""
    for blk in model.gpt_neox.layers:
        _fold_beta_into(blk.attention.query_key_value, blk.input_layernorm.bias.data.clone())
        _rmsify(blk.input_layernorm)
        _fold_beta_into(blk.mlp.dense_h_to_4h, blk.post_attention_layernorm.bias.data.clone())
        _rmsify(blk.post_attention_layernorm)
    _fold_beta_into(model.embed_out, model.gpt_neox.final_layer_norm.bias.data.clone())
    _rmsify(model.gpt_neox.final_layer_norm)


def run(model_name="EleutherAI/pythia-160m", max_length=128, num_samples=48, save=None):
    assert "pythia" in model_name or "neox" in model_name.lower(), \
        "this converter implements GPT-NeoX (Pythia) nn.Linear consumers"
    model, tok, device = H.load(model_name, dtype=torch.float32)
    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    print(f"compile {model_name}: LayerNorm -> RMSNorm (fold beta and the centring)")
    H.print_row("native LayerNorm", H.compare(base, base))
    convert_pythia(model)
    new = H.logits_for(model, batches)
    H.print_row("converted RMSNorm", H.compare(base, new))
    print("reading: exact up to rounding (Lean layerNorm_sub_rmsNorm); convert_pythia_beta_only, which drops")
    print("the centring, is the lossy variant.")
    if save:
        model.save_pretrained(save)
        tok.save_pretrained(save)
        print(f"saved to {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    run(args.model, save=args.save)
