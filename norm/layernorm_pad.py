"""Hidden-width padding of a LayerNorm model (Lean EP.layerNorm_pad_center, EP.hidden_pad_stack_layerNorm).

A pre-LN target of width d runs exactly on a host of width D = d + e whose residual is the CENTRED
target residual padded with zeros, [P h; 0] (P = I - 11^T/d). Zero padding alone is not enough under
LayerNorm: the pad channels change both the mean and the variance. With the residual centred the host
LayerNorm is the padded RMSNorm of the centred vector, so the host norm takes

    weight [sqrt(d/D) gamma; 0],   bias [beta; 0],   eps' = eps d / D,

and every residual writer is post-composed with P (as in norm/layernorm_rmsnorm.fold_layernorm_model).
Readers (q/k/v, fc1) read the first d channels; the head reads the first d channels of the final norm.

`pad_layernorm_model(model, e)` builds that host from a stock GPT-NeoX or OPT model as another stock
model of the same class with hidden size d + e. The head dimension is kept, so e must be a multiple of
it: the host gets e / head_dim extra heads with zero q/k/v rows and zero output columns. Tied
embeddings are untied (the input embedding is centred, the head keeps the original rows).

Run:  python -m norm.layernorm_pad
"""
from __future__ import annotations

import math

import torch


# --- the norm-level identity ---------------------------------------------------------------------

def center(x):
    return x - x.mean(-1, keepdim=True)


def pad(x, e):
    return torch.cat([x, x.new_zeros(*x.shape[:-1], e)], -1)


def pad_ln_params(gamma, beta, eps, e):
    """The host LayerNorm (weight, bias, eps) for a target LayerNorm (gamma, beta, eps) of width d
    on a residual padded by e zeros (EP.layerNorm_pad_center)."""
    d = gamma.shape[-1]
    s = math.sqrt(d / (d + e))
    return pad(s * gamma, e), pad(beta, e), eps * d / (d + e)


def layer_norm(x, w, b, eps):
    """torch's LayerNorm (biased variance) written out in the caller's dtype."""
    mu = x.mean(-1, keepdim=True)
    var = (x - mu).pow(2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * w + b


def host_layer_norm(x, gamma, beta, eps, e):
    """The host LayerNorm of EP.layerNorm_pad_center applied to the padded centred vector [P x; 0]."""
    w, b, eps2 = pad_ln_params(gamma, beta, eps, e)
    return layer_norm(pad(center(x), e), w, b, eps2)


# --- the whole-model host --------------------------------------------------------------------------

def _rows(W, D):                     # zero rows below
    out = W.new_zeros(D, *W.shape[1:])
    out[: W.shape[0]] = W
    return out


def _cols(W, D):                     # zero columns to the right (a reader of the first d channels)
    out = W.new_zeros(W.shape[0], D)
    out[:, : W.shape[1]] = W
    return out


def _writer(W, b, D):
    """A residual writer post-composed with P and padded: rows [P W; 0], bias [P b; 0]."""
    W = W - W.mean(0, keepdim=True)
    return _rows(W, D), (None if b is None else pad(b - b.mean(), D - b.shape[0]))


def _ln(sd, key, eps, e, out):
    w, b, _ = pad_ln_params(sd[key + ".weight"], sd[key + ".bias"], eps, e)
    out[key + ".weight"], out[key + ".bias"] = w, b


def _host_config(model, e):
    cfg = model.config.to_dict()
    t = cfg["model_type"]
    d, H = cfg["hidden_size"], cfg["num_attention_heads"]
    hd = d // H
    if e % hd:
        raise ValueError(f"e = {e} must be a multiple of the head dim {hd} (the host keeps the head dim)")
    cfg.update(hidden_size=d + e, num_attention_heads=H + e // hd, tie_word_embeddings=False)
    if t == "gpt_neox":
        cfg["layer_norm_eps"] = model.config.layer_norm_eps * d / (d + e)
    elif t == "opt":
        if not cfg.get("do_layer_norm_before", True) or cfg.get("word_embed_proj_dim", d) != d:
            raise NotImplementedError("pre-LN OPT without projections only")
        cfg["word_embed_proj_dim"] = d + e
    else:
        raise NotImplementedError(t)
    return type(model.config).from_dict(cfg)


@torch.no_grad()
def pad_layernorm_model(model, e):
    """A stock model of the same class, hidden size d + e, computing the same logits as `model` (a
    pre-LN GPT-NeoX or OPT causal LM): the host of EP.hidden_pad_stack_layerNorm. Returns the host."""
    from transformers import AutoModelForCausalLM
    t = model.config.model_type
    d = model.config.hidden_size
    D = d + e
    hcfg = _host_config(model, e)
    hcfg._attn_implementation = model.config._attn_implementation
    host = AutoModelForCausalLM.from_config(hcfg).to(model.dtype).eval()
    sd = model.state_dict()
    out = {}
    if t == "gpt_neox":
        eps = model.config.layer_norm_eps
        H, hs = model.config.num_attention_heads, d // model.config.num_attention_heads
        Hh = hcfg.num_attention_heads
        E = sd["gpt_neox.embed_in.weight"]
        out["gpt_neox.embed_in.weight"] = pad(center(E), e)
        for i in range(model.config.num_hidden_layers):
            p = f"gpt_neox.layers.{i}."
            _ln(sd, p + "input_layernorm", eps, e, out)
            _ln(sd, p + "post_attention_layernorm", eps, e, out)
            W = sd[p + "attention.query_key_value.weight"].view(H, 3, hs, d)
            Wh = W.new_zeros(Hh, 3, hs, D)
            Wh[:H, :, :, :d] = W
            out[p + "attention.query_key_value.weight"] = Wh.view(3 * D, D)
            b = sd.get(p + "attention.query_key_value.bias")
            if b is not None:
                bh = b.new_zeros(Hh, 3, hs)
                bh[:H] = b.view(H, 3, hs)
                out[p + "attention.query_key_value.bias"] = bh.view(-1)
            W, b = _writer(sd[p + "attention.dense.weight"], sd.get(p + "attention.dense.bias"), D)
            out[p + "attention.dense.weight"] = _cols(W, D)
            if b is not None:
                out[p + "attention.dense.bias"] = b
            out[p + "mlp.dense_h_to_4h.weight"] = _cols(sd[p + "mlp.dense_h_to_4h.weight"], D)
            out[p + "mlp.dense_h_to_4h.bias"] = sd[p + "mlp.dense_h_to_4h.bias"]
            W, b = _writer(sd[p + "mlp.dense_4h_to_h.weight"], sd[p + "mlp.dense_4h_to_h.bias"], D)
            out[p + "mlp.dense_4h_to_h.weight"], out[p + "mlp.dense_4h_to_h.bias"] = W, b
        _ln(sd, "gpt_neox.final_layer_norm", eps, e, out)
        out["embed_out.weight"] = _cols(sd["embed_out.weight"], D)
    else:                                                    # opt
        pre = "model.decoder."
        E = sd[pre + "embed_tokens.weight"]
        out[pre + "embed_tokens.weight"] = pad(center(E), e)
        out[pre + "embed_positions.weight"] = pad(center(sd[pre + "embed_positions.weight"]), e)
        head = sd.get("lm_head.weight", E)
        out["lm_head.weight"] = _cols(head, D)
        lns = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
        eps = lns[0].eps
        assert all(m.eps == eps for m in lns)
        for i in range(model.config.num_hidden_layers):
            p = f"{pre}layers.{i}."
            _ln(sd, p + "self_attn_layer_norm", eps, e, out)
            _ln(sd, p + "final_layer_norm", eps, e, out)
            for q in ("q_proj", "k_proj", "v_proj"):
                out[p + f"self_attn.{q}.weight"] = _cols(_rows(sd[p + f"self_attn.{q}.weight"], D), D)
                out[p + f"self_attn.{q}.bias"] = pad(sd[p + f"self_attn.{q}.bias"], e)
            W, b = _writer(sd[p + "self_attn.out_proj.weight"], sd[p + "self_attn.out_proj.bias"], D)
            out[p + "self_attn.out_proj.weight"], out[p + "self_attn.out_proj.bias"] = _cols(W, D), b
            out[p + "fc1.weight"], out[p + "fc1.bias"] = _cols(sd[p + "fc1.weight"], D), sd[p + "fc1.bias"]
            W, b = _writer(sd[p + "fc2.weight"], sd[p + "fc2.bias"], D)
            out[p + "fc2.weight"], out[p + "fc2.bias"] = W, b
        _ln(sd, pre + "final_layer_norm", eps, e, out)
        for m in host.modules():                              # OPT's LayerNorm eps is not a config field
            if isinstance(m, torch.nn.LayerNorm):
                m.eps = eps * d / D
    missing, unexpected = host.load_state_dict(out, strict=False)
    missing = [k for k in missing if "rotary" not in k and "masked_bias" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    return host


@torch.no_grad()
def pad_check(name="EleutherAI/pythia-14m", e=None, dtype=torch.float64, T=32, n=2, seed=0):
    """max |logits(host) - logits(target)| on random ids, for the padded LayerNorm host of a cached
    checkpoint, plus the same for a host that zero-pads WITHOUT centring (control)."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, local_files_only=True,
                                             attn_implementation="eager").eval()
    hd = m.config.hidden_size // m.config.num_attention_heads
    e = hd if e is None else e
    h = pad_layernorm_model(m, e)
    ids = torch.randint(0, m.config.vocab_size, (n, T), generator=torch.Generator().manual_seed(seed))
    a, b = m(ids).logits, h(ids).logits
    return dict(d=m.config.hidden_size, D=h.config.hidden_size, err=(a - b).abs().max().item(),
                scale=a.abs().max().item())


if __name__ == "__main__":
    for nm in ("EleutherAI/pythia-14m", "facebook/opt-125m"):
        print(nm, pad_check(nm))
