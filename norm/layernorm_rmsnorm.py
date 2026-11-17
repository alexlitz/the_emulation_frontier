"""Normalization subset: LayerNorm  vs  RMSNorm.

Per-call identity:   LayerNorm(x) = gamma (.) RMSNorm(P x) + beta,
where P = I - (1/d) 11^T is mean-centering and rms(Px) = std(x) (biased, as torch computes it).

Whole-model FOLD (fold_layernorm_model below, exact on the stock Pythia / OPT checkpoints):
  * The centering P can't be folded into the norm call itself, but it CAN be folded into the
    model. If every residual WRITER emits mean-zero vectors, the residual stream lives in 1^perp,
    so P x = x at every norm and LayerNorm(x) = gamma (.) RMSNorm(x) + beta exactly (same eps).
    Writers: W -> P W, b -> P b for every residual-writing linear (attn out / dense / out_proj,
    MLP fc2 / dense_4h_to_h). Embeddings: E -> E P (centre each row), and learned position
    embeddings the same way.
  * gamma stays as the RMSNorm gain.
  * beta has to go into the NEXT linear's bias: W(r + beta) + b = W r + (W beta + b). A
    "constant residual channel" can NOT carry it under RMSNorm: every channel comes out
    divided by the token's rms, so nothing after an RMSNorm is constant. So every consumer
    of a normalised input needs a bias. Pythia / OPT have biases on qkv / q,k,v / fc1, but NOT
    on the LM head (Pythia embed_out, OPT lm_head). There the final norm's beta becomes a
    per-vocab logit shift W_U beta_f, which changes the softmax. The fold adds an lm_head bias
    for it (a module change; also measured without it).
  * Tied embeddings break the fold: the input embedding must be centred (E P) while the
    LM head must stay E. OPT ties them, so the fold UNTIES them (lm_head keeps the original E).
  * A residual reader that is NOT behind a norm would see P x instead of x. In pre-LN Pythia /
    OPT every residual reader is a norm, so the fold is exact there.

The per-call variants below are patched onto every nn.LayerNorm with no fold. They measure what
dropping the centering costs WITHOUT the writer fold:

  native        real LayerNorm                                 (baseline)
  exact         center -> RMSNorm -> affine   == native to fp noise (sanity)
  rms_keepbias  drop centering, keep gamma & beta   (the cost of skipping the writer fold)
  rms_pure      drop centering & delete beta = literal RMSNorm module, nothing folded
  affine_only   gamma*x + beta, no normalization          (control)

Run:
  python -m norm.layernorm_rmsnorm --model EleutherAI/pythia-160m
  python -m norm.layernorm_rmsnorm --model gpt2
  python -m norm.layernorm_rmsnorm --fold                    # the whole-model fold, fp64 + fp32
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import harness as H


def _rms(x, eps):
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def make_variant(kind):
    def factory(module: nn.LayerNorm):
        w, b, eps = module.weight, module.bias, module.eps

        def fwd(self, x):
            xf = x.float()
            if kind == "exact":
                c = xf - xf.mean(-1, keepdim=True)
                y = c * _rms(c, eps) * w.float() + b.float()
            elif kind == "rms_keepbias":
                y = xf * _rms(xf, eps) * w.float() + b.float()
            elif kind == "rms_pure":
                y = xf * _rms(xf, eps) * w.float()
            elif kind == "affine_only":
                y = xf * w.float() + b.float()
            else:
                raise ValueError(kind)
            return y.to(x.dtype)

        return fwd

    return factory


def run(model_name, max_length=128, num_samples=64):
    model, tok, device = H.load(model_name, dtype=torch.float32)
    lns = H.modules_of_type(model, nn.LayerNorm)
    print(f"{model_name}: {len(lns)} LayerNorm modules, hidden={model.config.hidden_size}")

    batches = H.eval_batches(tok, device, max_length=max_length, num_samples=num_samples)
    base = H.logits_for(model, batches)
    print(f"eval: {len(batches)} contexts, ~{sum(b.shape[1] for b in batches)} positions\n")

    H.print_row("native (self-check)", H.compare(base, base))
    for kind in ["exact", "rms_keepbias", "rms_pure", "affine_only"]:
        with H.patched_forwards(lns, make_variant(kind)):
            logits = H.logits_for(model, batches)
        H.print_row(kind, H.compare(base, logits))

    print(
        "\nreading: 'exact' matches native to fp noise (centering-fold is exact). "
        "'rms_keepbias' = the ONLY irreducible penalty (dropping centering on a "
        "LayerNorm-trained model). 'rms_pure' naively DELETES beta instead of "
        "folding it downstream, so its extra damage is a missing weight-fold pass, "
        "not a cost of RMSNorm. 'affine_only' = normalization's raw contribution."
    )


# --- the whole-model fold ------------------------------------------------------------------

def _family(model):
    """(embeddings to centre, [(norm, [consumer linears])], [residual writer linears],
    final norm, lm_head attribute name) for the LayerNorm families the fold supports."""
    t = model.config.model_type
    if t == "gpt_neox":
        g = model.gpt_neox
        norms = [(L.input_layernorm, [L.attention.query_key_value]) for L in g.layers]
        norms += [(L.post_attention_layernorm, [L.mlp.dense_h_to_4h]) for L in g.layers]
        writers = [L.attention.dense for L in g.layers] + [L.mlp.dense_4h_to_h for L in g.layers]
        return [g.embed_in], norms, writers, g.final_layer_norm, "embed_out"
    if t == "opt":
        dec = model.model.decoder
        assert model.config.do_layer_norm_before and dec.project_in is None, "pre-LN OPT without projections"
        norms = [(L.self_attn_layer_norm, [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj])
                 for L in dec.layers]
        norms += [(L.final_layer_norm, [L.fc1]) for L in dec.layers]
        writers = [L.self_attn.out_proj for L in dec.layers] + [L.fc2 for L in dec.layers]
        return [dec.embed_tokens, dec.embed_positions], norms, writers, dec.final_layer_norm, "lm_head"
    raise NotImplementedError(t)


def _swap(root, old, new):
    for _, mod in root.named_modules():
        for cname, child in mod.named_children():
            if child is old:
                setattr(mod, cname, new)
                return
    raise KeyError("module not found")


def _rmsnorm(ln):
    r = nn.RMSNorm(ln.normalized_shape, eps=ln.eps, elementwise_affine=True,
                   device=ln.weight.device, dtype=ln.weight.dtype)
    with torch.no_grad():
        r.weight.copy_(ln.weight)
    return r


@torch.no_grad()
def fold_layernorm_model(model, lm_head_bias=True):
    """In place: turn a pre-LN LayerNorm model (GPT-NeoX / OPT) into an RMSNorm model computing
    the same function. Writers -> P W, embeddings -> E P, every nn.LayerNorm -> nn.RMSNorm(gamma,
    eps), beta -> consumer biases (+ an lm_head bias for the final norm unless lm_head_bias=False).
    Unties tied embeddings. Returns a dict of what was changed."""
    embs, norms, writers, final_ln, head_name = _family(model)
    head = getattr(model, head_name)
    tied = head.weight.data_ptr() == embs[0].weight.data_ptr()
    if tied:                                            # the head must keep the UNcentred E
        new_head = nn.Linear(head.in_features, head.out_features, bias=False,
                             device=head.weight.device, dtype=head.weight.dtype)
        new_head.weight.copy_(head.weight)
        setattr(model, head_name, new_head)
        model.config.tie_word_embeddings = False
        head = new_head
    for e in embs:
        e.weight.sub_(e.weight.mean(-1, keepdim=True))  # rows of E -> E P
    for w in writers:
        w.weight.sub_(w.weight.mean(0, keepdim=True))   # W -> P W (centre each output column)
        if w.bias is not None:
            w.bias.sub_(w.bias.mean())
    for ln, consumers in norms:
        for c in consumers:
            c.bias.add_(c.weight @ ln.bias)
        _swap(model, ln, _rmsnorm(ln))
    if lm_head_bias:
        nh = nn.Linear(head.in_features, head.out_features, bias=True,
                       device=head.weight.device, dtype=head.weight.dtype)
        nh.weight.copy_(head.weight)
        nh.bias.copy_(head.weight @ final_ln.bias)
        setattr(model, head_name, nh)
    _swap(model, final_ln, _rmsnorm(final_ln))
    left = H.modules_of_type(model, nn.LayerNorm)
    assert not left, f"{len(left)} LayerNorms left"
    return dict(untied=tied, norms=len(norms) + 1, writers=len(writers), embeddings=len(embs))


def fold_check(name="EleutherAI/pythia-14m", dtype=torch.float64, lm_head_bias=True, T=64, n=4, seed=0):
    """max|logits(fold) - logits(original)| on random token ids (no tokenizer / dataset needed),
    plus the max |log_softmax| difference. Loads the checkpoint from the local HF cache only."""
    from transformers import AutoModelForCausalLM
    m0 = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, local_files_only=True).eval()
    m1 = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, local_files_only=True).eval()
    info = fold_layernorm_model(m1, lm_head_bias=lm_head_bias)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, m0.config.vocab_size, (n, T), generator=g)
    with torch.no_grad():
        a, b = m0(ids).logits.double(), m1(ids).logits.double()
    return dict(**info, logit_err=(a - b).abs().max().item(), logit_scale=a.abs().max().item(),
                logprob_err=(a.log_softmax(-1) - b.log_softmax(-1)).abs().max().item())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--fold", action="store_true", help="whole-model LayerNorm -> RMSNorm fold")
    args = ap.parse_args()
    if args.fold:
        for name in ("EleutherAI/pythia-14m", "facebook/opt-125m"):
            for dt in (torch.float64, torch.float32):
                for hb in (True, False):
                    r = fold_check(name, dt, lm_head_bias=hb)
                    print(f"{name:22s} {str(dt)[6:]:8s} lm_head bias={hb!s:5s} untied={r['untied']!s:5s} "
                          f"max|dlogit| = {r['logit_err']:.2e} (|logit|max {r['logit_scale']:.1f})  "
                          f"max|dlogprob| = {r['logprob_err']:.2e}")
    else:
        run(args.model, args.max_length, args.num_samples)
