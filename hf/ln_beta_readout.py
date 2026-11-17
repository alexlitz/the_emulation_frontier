"""The LayerNorm-beta fold WITHOUT an lm_head bias, on real checkpoints: a stock LlamaForCausalLM host (RMSNorm,
attention_bias, mlp_bias; untied, no lm_head bias) plays a LayerNorm-with-beta StableLM target. Port ln_beta_readout of
graph/model_graph.py; Lean dominated_readout_limit, pad_ratio_limit, pos_scale_le_iff (ResNorm.lean); the fp64
synthetic instance is catalog fr_ln_beta_readout.

Construction, host hidden D = d + 1, depth L + 1:
  layers 0 .. L-1  the exact LayerNorm -> RMSNorm fold: every residual writer centred (embedding rows, o_proj and
                   down_proj columns: P W), every LN gamma folded into the reading weights and every LN beta into the
                   readers' biases (q / k / v biases from the input LN, gate / up biases from the post-attention LN), the
                   zero-padded extra channel handled by the hidden-pad rule (host eps = eps d / D, norm gains
                   sqrt(d / D)); StableLM's 25% partial RoPE by permuting the head dims into Llama's rotate_half pairs and
                   writing the host inv_freq (the target's frequencies on the first rot / 2 pairs, 0 on the rest);
  layer L          the readout: attention off (o_proj 0); the MLP's gate is the constant SiLU(c0) (gate weights 0, gate
                   bias c0), its up-projection the identity on the d channels, so it writes M (gamma_f x_hat + beta_f)
                   -- the target's final LayerNorm output, scaled -- into the stream and M kappa into the extra
                   channel (down bias);
  head             final norm gain 1, lm_head = (kappa / sqrt(D)) [U; 0]: logits = (U . y) kappa / sqrt(|y|^2 + kappa^2)
                   up to the stream's P x / M -- a positive multiple of the target's logits -> them as kappa -> oo.

Scored with the act_swap standard (hf/eps_lever_agreement.py: the target's own greedy 64-token continuations of 24
prompts; payable = top-1 >= 99% and >= 90% identical 32-token continuations), fp32, eager. Also the control "drop the
final beta" (the readout layer removed, the final norm folded): what the rule pays for.

  python -m hf.ln_beta_readout stabilityai/stable-code-3b --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ln_beta_readout.jsonl")
TARGETS = ("stabilityai/stable-code-3b", "stabilityai/stablelm-2-1_6b")
C0 = 20.0


def host_config(tc, extra_layer=True):
    from transformers import LlamaConfig
    d = tc.hidden_size
    D = d + 1
    hd = d // tc.num_attention_heads
    c = LlamaConfig(vocab_size=tc.vocab_size, hidden_size=D, intermediate_size=max(tc.intermediate_size, d),
                    num_hidden_layers=tc.num_hidden_layers + int(extra_layer), num_attention_heads=tc.num_attention_heads,
                    num_key_value_heads=tc.num_key_value_heads, head_dim=hd, hidden_act="silu",
                    max_position_embeddings=tc.max_position_embeddings, rms_norm_eps=tc.layer_norm_eps * d / D,
                    rope_theta=tc.rope_theta, attention_bias=True, mlp_bias=True, tie_word_embeddings=False,
                    bos_token_id=None, eos_token_id=None, pad_token_id=None)
    c._attn_implementation = "eager"
    return c


def head_perm(hd, rot):
    """target head dim -> host head dim: the target's rotate_half pairs (i, i + rot/2) onto Llama's (i, i + hd/2)."""
    r2, h2 = rot // 2, hd // 2
    perm = list(range(r2)) + [h2 + i for i in range(r2)]
    rest = [j for j in range(hd) if j not in set(perm)]
    return torch.tensor(perm + rest)


def host_weights(tc, sd, M, kappa, readout=True):
    T = lambda n: sd[n].double()
    d, L, H, KV = tc.hidden_size, tc.num_hidden_layers, tc.num_attention_heads, tc.num_key_value_heads
    hd = d // H
    rot = int(hd * tc.partial_rotary_factor)
    D = d + 1
    pad = math.sqrt(d / D)
    perm = head_perm(hd, rot)
    cen = lambda W: W - W.mean(0, keepdim=True)
    padr = lambda W: torch.cat([W, torch.zeros(1, *W.shape[1:], dtype=torch.float64)], 0)
    padc = lambda W: torch.cat([W, torch.zeros(W.shape[0], 1, dtype=torch.float64)], 1)
    padv = lambda v: torch.cat([v, torch.zeros(1, dtype=torch.float64)])

    def permute_rows(W, n):                           # q / k rows per head, target dim -> host dim
        Wt = W.view(n, hd, *W.shape[1:])
        out = torch.zeros_like(Wt)
        out[:, perm] = Wt
        return out.reshape(W.shape)
    out = {}
    E = T("model.embed_tokens.weight")
    out["model.embed_tokens.weight"] = padc(E - E.mean(1, keepdim=True))
    I_h = max(tc.intermediate_size, d)
    for i in range(L):
        p = f"model.layers.{i}."
        g1, b1 = T(p + "input_layernorm.weight"), T(p + "input_layernorm.bias")
        g2, b2 = T(p + "post_attention_layernorm.weight"), T(p + "post_attention_layernorm.bias")
        out[p + "input_layernorm.weight"] = torch.full((D,), pad, dtype=torch.float64)
        out[p + "post_attention_layernorm.weight"] = torch.full((D,), pad, dtype=torch.float64)
        a = p + "self_attn."
        for m, n in (("q", H), ("k", KV), ("v", KV)):
            W = T(a + f"{m}_proj.weight")
            bias = T(a + f"{m}_proj.bias") if (a + f"{m}_proj.bias") in sd else torch.zeros(W.shape[0], dtype=torch.float64)
            Wf, bf = W * g1, bias + W @ b1
            if m != "v":
                Wf, bf = permute_rows(Wf, n), permute_rows(bf, n)
            out[a + f"{m}_proj.weight"], out[a + f"{m}_proj.bias"] = padc(Wf), bf
        Wo = T(a + "o_proj.weight")
        out[a + "o_proj.weight"] = padr(cen(Wo))
        bo = T(a + "o_proj.bias") if (a + "o_proj.bias") in sd else torch.zeros(d, dtype=torch.float64)
        out[a + "o_proj.bias"] = padv(bo - bo.mean())
        mm = p + "mlp."
        for m in ("gate", "up"):
            W = T(mm + f"{m}_proj.weight")
            Wf = torch.zeros(I_h, D, dtype=torch.float64)
            Wf[:W.shape[0], :d] = W * g2
            bf = torch.zeros(I_h, dtype=torch.float64)
            bf[:W.shape[0]] = W @ b2
            out[mm + f"{m}_proj.weight"], out[mm + f"{m}_proj.bias"] = Wf, bf
        Wd = T(mm + "down_proj.weight")
        Wdf = torch.zeros(D, I_h, dtype=torch.float64)
        Wdf[:d, :Wd.shape[1]] = cen(Wd)
        out[mm + "down_proj.weight"], out[mm + "down_proj.bias"] = Wdf, torch.zeros(D, dtype=torch.float64)
    gf, bfin = T("model.norm.weight"), T("model.norm.bias")
    U = T("lm_head.weight")
    if readout:
        p = f"model.layers.{L}."
        out[p + "input_layernorm.weight"] = torch.full((D,), pad, dtype=torch.float64)
        out[p + "post_attention_layernorm.weight"] = torch.full((D,), pad, dtype=torch.float64)
        a = p + "self_attn."
        for m, rows in (("q", H * hd), ("k", KV * hd), ("v", KV * hd)):
            out[a + f"{m}_proj.weight"] = torch.zeros(rows, D, dtype=torch.float64)
            out[a + f"{m}_proj.bias"] = torch.zeros(rows, dtype=torch.float64)
        out[a + "o_proj.weight"] = torch.zeros(D, H * hd, dtype=torch.float64)
        out[a + "o_proj.bias"] = torch.zeros(D, dtype=torch.float64)
        s = C0 / (1 + math.exp(-C0))                   # SiLU(c0)
        mm = p + "mlp."
        out[mm + "gate_proj.weight"] = torch.zeros(I_h, D, dtype=torch.float64)
        out[mm + "gate_proj.bias"] = torch.full((I_h,), C0, dtype=torch.float64)
        up = torch.zeros(I_h, D, dtype=torch.float64)
        up[:d, :d] = torch.eye(d, dtype=torch.float64)
        out[mm + "up_proj.weight"], out[mm + "up_proj.bias"] = up, torch.zeros(I_h, dtype=torch.float64)
        dn = torch.zeros(D, I_h, dtype=torch.float64)
        dn[:d, :d] = torch.diag(gf) * (M / s)
        out[mm + "down_proj.weight"] = dn
        out[mm + "down_proj.bias"] = torch.cat([bfin * M, torch.tensor([M * kappa], dtype=torch.float64)])
        out["model.norm.weight"] = torch.ones(D, dtype=torch.float64)
        out["lm_head.weight"] = padc(U) * (kappa / math.sqrt(D))
    else:                                              # control: the final beta dropped
        out["model.norm.weight"] = padv(gf * pad)
        out["lm_head.weight"] = padc(U)
    return out, rot, perm


def build_host(tc, sd, M, kappa, readout=True):
    from transformers import LlamaForCausalLM
    cfg = host_config(tc, readout)
    host = LlamaForCausalLM(cfg).eval()
    W, rot, perm = host_weights(tc, sd, M, kappa, readout)
    names = dict(host.named_parameters())
    missing = [k for k in names if k not in W]
    assert not missing, missing[:5]
    with torch.no_grad():
        for k, prm in names.items():
            prm.data = W[k].float()
        hd = tc.hidden_size // tc.num_attention_heads
        inv = torch.zeros(hd // 2, dtype=torch.float32)
        inv[:rot // 2] = 1.0 / (tc.rope_theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot)).float()
        host.model.rotary_emb.inv_freq = inv            # (a buffer write: the target's frequencies, 0 elsewhere)
        host.model.rotary_emb.original_inv_freq = inv.clone()
    return host


def run(repo, device, n_prompts, sink, M=2.0 ** 24, kappa=2.0 ** 14):
    from hf.eps_lever_agreement import load_target, reference, score, strict
    t0 = time.time()
    tok, tgt = load_target(repo, torch.float32)
    tc = tgt.config
    sd = {k: v.detach().cpu() for k, v in tgt.state_dict().items()}
    ref = reference(tgt.to(device), tok, device, n_prompts)
    del tgt
    for variant, ro in (("readout", True), ("drop_final_beta", False)):
        host = build_host(tc, sd, M, kappa, ro).to(device)
        r = score(host, ref, device)
        import transformers
        r = dict(target=repo, host="llama(attention_bias, mlp_bias)", variant=variant, M=M, kappa=kappa, c0=C0,
                 dtype="float32", transformers=transformers.__version__, n_prompts=n_prompts, **r)
        r["payable"] = strict(r)
        sink(r)
        print(f"{repo} {variant}: top1 {r['top1']:.4f} ex32 {r['ex32']:.2f} ex64 {r['ex64']:.2f} "
              f"max|dlogp| {r['max_dlogp']:.2e} kl {r['kl']:.2e} {'PASS' if r['payable'] else 'fail'} "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        del host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=TARGETS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.target, a.device, a.n_prompts, sink)


if __name__ == "__main__":
    main()
