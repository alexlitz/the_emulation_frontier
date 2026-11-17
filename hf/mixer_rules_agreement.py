"""The fr-mixer rules on real checkpoints: a recurrent target played by a host of another recurrent layout, stock
transformers classes only (no remote code), the host's weights all written from the target's.

Rules (graph/model_graph.py, branch fr-mixer; dist/fr_mixer.md; Lean FrMixer.lean):

  pure_on_hybrid   a PURE recurrent stack on a HYBRID host: target layer i on the host's i-th recurrent layer, every
                   other host layer zeroed (attention o_proj = 0, FFN down_proj = 0: the identity). EXACT
                   (Lean padded_run / assign_run); measured here as a check, fp64 max |dlogit| at rounding.
  m2_nonorm_lever  a Mamba-2 target WITHOUT the gated output norm (Falcon-H1 mamba_rms_norm = false: out_proj(y silu z))
                   on a host WITH it (RMSNorm(y silu z)). The norm's input is a scan / silu output, so no weight scales
                   it down; the CONFIG eps does: residual stream x S (embedding_multiplier x S, every output projection
                   x S), host rms_norm_eps = e_t S^2 (Lean rmsNorm_smul_eps: every stream norm is exactly the
                   target's), and the gated norm, fed an O(1) input against eps e_t S^2, is linear (epsLever_eq): gain
                   S sqrt(e_t) makes it the identity. A LIMIT: rho = ms(y silu z) / (e_t S^2) -> 0; at the fp32
                   rounding floor once rho < 2^-25.
  mamba1_lever     a plain Mamba-1 target on a host whose B / C / dt pass RMS norms (Falcon-Mamba: weight-free, eps
                   mixer_rms_eps; Jamba: weighted, eps rms_norm_eps). The norms' inputs ARE linear maps' outputs
                   (x_proj rows), so the weights-only eps lever applies: x_proj rows x 2^-a, the norm then linear;
                   Jamba's gains 2^a sqrt(e); Falcon-Mamba's weight-free constant 2^-a / sqrt(e) folds into dt_proj
                   (x 2^a sqrt(e)), D (x c) and out_proj (x 1 / c), c = 2^-2a / e (B and C both scaled). A LIMIT.

Standard (act/act_swap_agreement.py, as hf/eps_lever_agreement.py): the ORIGINAL target's greedy 64-token
continuations of 24 prompts; payable = top-1 >= 99% AND >= 90% of 32-token continuations identical; eager / torch
paths (no mamba_ssm kernels installed), TF32 off.

  CUDA_VISIBLE_DEVICES=1 python -m hf.mixer_rules_agreement nonorm:tiny-r-90m --device cuda:0 --dtype float32
  python -m hf.mixer_rules_agreement all --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import torch

import hf.eps_lever_agreement as _ela
from act.act_swap_agreement import N_NEW
from hf.eps_lever_agreement import reference, score, strict


@torch.no_grad()
def greedy_recompute(model, ids, n_new=N_NEW):
    """Greedy continuation by full recompute (no cache). The hybrid classes (FalconH1, Bamba, Jamba) return NO cache
    unless one is built by the caller, and act_swap's cached greedy then feeds the next token as a fresh 1-token
    sequence; recomputing is the same function for every model (mamba / mamba2 included)."""
    seq = ids
    for _ in range(n_new):
        seq = torch.cat([seq, model(seq, use_cache=False).logits[:, -1:].argmax(-1)], 1)
    return seq[:, ids.shape[1]:]


_ela.greedy = greedy_recompute          # reference() / score() look greedy up in that module

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mixer_rules_agreement.jsonl")
RHO_LOG2 = -30          # the recorded construction: S (or a) chosen so that rho_max <= 2^-30 on the target's own text

PAIRS = {
    # m2_nonorm_lever: Falcon-H1 targets with mamba_rms_norm = false, on a FalconH1 host with the gated norm on
    "nonorm:tiny-90m": dict(rule="m2_nonorm_lever", target="tiiuae/Falcon-H1-Tiny-90M-Base", host="falcon_h1"),
    "nonorm:tiny-r-90m": dict(rule="m2_nonorm_lever", target="tiiuae/Falcon-H1-Tiny-R-90M", host="falcon_h1"),
    "nonorm:0.5b": dict(rule="m2_nonorm_lever", target="tiiuae/Falcon-H1-0.5B-Base", host="falcon_h1"),
    # pure_on_hybrid: a pure Mamba-2 stack on Bamba (attention layers interleaved, zeroed)
    "pure:mamba2-130m@bamba": dict(rule="pure_on_hybrid", target="AntonV/mamba2-130m-hf", host="bamba"),
    # mamba1_lever (+ pure_on_hybrid on Jamba): a plain Mamba-1 stack on hosts with B / C / dt norms
    "mamba1:130m@falcon_mamba": dict(rule="mamba1_lever", target="state-spaces/mamba-130m-hf", host="falcon_mamba"),
    "mamba1:370m@falcon_mamba": dict(rule="mamba1_lever", target="state-spaces/mamba-370m-hf", host="falcon_mamba"),
    "mamba1:130m@jamba": dict(rule="mamba1_lever+pure_on_hybrid", target="state-spaces/mamba-130m-hf", host="jamba"),
}


def load_target(repo, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    cfg = AutoConfig.from_pretrained(repo)
    if hasattr(cfg, "use_mamba_kernels"):
        cfg.use_mamba_kernels = False
    if getattr(cfg, "residual_in_fp32", False) and dtype == torch.float64:
        cfg.residual_in_fp32 = False          # (a downcast of the fp64 residual: precision only, both runs equal)
    model = AutoModelForCausalLM.from_pretrained(repo, config=cfg, dtype=dtype, attn_implementation="eager").eval()
    return tok, model


@torch.no_grad()
def max_ms(model, ref, device, module_filter, which="input"):
    """The largest mean square (over the last dim) any matching module saw on the reference sequences."""
    best = [0.0]

    def hook(_m, args, out):
        x = args[0] if which == "input" else out
        best[0] = max(best[0], x.float().pow(2).mean(-1).max().item())
    hs = [m.register_forward_hook(hook) for n, m in model.named_modules() if module_filter(n, m)]
    for s in ref["seq"]:
        model(s.to(device))
    for h in hs:
        h.remove()
    return best[0]


def pow2_at_least(x):
    return 2.0 ** math.ceil(math.log2(max(x, 1e-300)))


# --- m2_nonorm_lever ----------------------------------------------------------------------------------------------
def host_nonorm(target, S):
    """A FalconH1 host WITH the gated norm playing the norm-free target: residual x S, eps e S^2, gain S sqrt(e)."""
    from transformers import FalconH1Config, FalconH1ForCausalLM
    tc = target.config
    d = tc.to_dict()
    e = tc.rms_norm_eps
    d.update(mamba_rms_norm=True, mamba_norm_before_gate=False, rms_norm_eps=e * S * S,
             embedding_multiplier=tc.embedding_multiplier * S, use_mamba_kernels=False)
    for k in ("architectures", "transformers_version", "_name_or_path", "model_type"):
        d.pop(k, None)
    hc = FalconH1Config(**d)
    host = FalconH1ForCausalLM(hc).to(next(target.parameters()).dtype).eval()
    sd = {k: v.clone() for k, v in target.state_dict().items()}
    for k in list(sd):
        if any(k.endswith(f"{p}.{w}") for p in ("self_attn.o_proj", "feed_forward.down_proj", "mamba.out_proj")
               for w in ("weight", "bias")):
            sd[k] = sd[k] * S
    hsd = host.state_dict()
    for k in hsd:
        if k.endswith("mamba.norm.weight"):
            sd[k] = torch.full_like(hsd[k], S * math.sqrt(e))
    missing, unexpected = host.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.endswith("lm_head.weight") for k in missing), (missing, unexpected)
    host.tie_weights()
    return host


# --- pure_on_hybrid: Mamba-2 stack on Bamba -------------------------------------------------------------------------
def host_bamba(target, period=3, heads=12, kv=4):
    """BambaForCausalLM with an attention layer after every `period` Mamba layers; target Mamba-2 layer i on the host's
    i-th Mamba layer; every attention layer's o_proj and every FFN's down_proj zero."""
    from transformers import BambaConfig, BambaForCausalLM
    tc = target.config
    n = tc.num_hidden_layers
    kinds = []
    for i in range(n):
        kinds.append("mamba")
        if i % period == period - 1:
            kinds.append("attention")
    hc = BambaConfig(vocab_size=tc.vocab_size, tie_word_embeddings=tc.tie_word_embeddings, hidden_size=tc.hidden_size,
                     intermediate_size=256, num_hidden_layers=len(kinds), num_attention_heads=heads,
                     num_key_value_heads=kv, rms_norm_eps=tc.layer_norm_epsilon,
                     attn_layer_indices=[j for j, k in enumerate(kinds) if k == "attention"],
                     mamba_n_heads=tc.num_heads, mamba_d_head=tc.head_dim, mamba_n_groups=tc.n_groups,
                     mamba_d_state=tc.state_size, mamba_d_conv=tc.conv_kernel, mamba_expand=tc.expand,
                     mamba_chunk_size=tc.chunk_size, mamba_conv_bias=tc.use_conv_bias, mamba_proj_bias=tc.use_bias,
                     pad_token_id=tc.pad_token_id, bos_token_id=tc.bos_token_id, eos_token_id=tc.eos_token_id,
                     use_mamba_kernels=False)
    host = BambaForCausalLM(hc).to(next(target.parameters()).dtype).eval()
    tsd = target.state_dict()
    sd = {k: v.clone() for k, v in host.state_dict().items()}
    sd["model.embed_tokens.weight"] = tsd["backbone.embeddings.weight"].clone()
    sd["model.final_layernorm.weight"] = tsd["backbone.norm_f.weight"].clone()
    sd["lm_head.weight"] = (sd["model.embed_tokens.weight"] if tc.tie_word_embeddings
                            else tsd["lm_head.weight"].clone())      # (tied: the shared tensor must not be overwritten)
    i = 0
    for j, k in enumerate(kinds):
        p = f"model.layers.{j}."
        sd[p + "feed_forward.down_proj.weight"].zero_()
        if k == "attention":
            sd[p + "self_attn.o_proj.weight"].zero_()
            continue
        sd[p + "input_layernorm.weight"] = tsd[f"backbone.layers.{i}.norm.weight"].clone()
        for name in ("in_proj.weight", "in_proj.bias", "conv1d.weight", "conv1d.bias", "dt_bias", "A_log", "D",
                     "norm.weight", "out_proj.weight", "out_proj.bias"):
            tk = f"backbone.layers.{i}.mixer.{name}"
            if tk in tsd:
                assert sd[p + "mamba." + name].shape == tsd[tk].shape, name
                sd[p + "mamba." + name] = tsd[tk].clone()
        i += 1
    assert i == n
    host.load_state_dict(sd, strict=True)
    if tc.tie_word_embeddings:
        host.tie_weights()
    else:
        host.lm_head.weight.data.copy_(target.lm_head.weight.data)
    return host


# --- mamba1_lever: plain Mamba-1 on Falcon-Mamba / Jamba -------------------------------------------------------------
def _mamba1_layer_weights(tsd, i, a, e, weight_free):
    """The levered Mamba-1 mixer weights of target layer i: x_proj's dt / B / C rows x 2^-a; with a weight-free norm
    (Falcon-Mamba) dt_proj x 2^a sqrt(e), D x c and out_proj x 1 / c, c = 2^-2a / e; with gains (Jamba) the gains
    2^a sqrt(e) (returned separately)."""
    p = f"backbone.layers.{i}.mixer."
    w = {k[len(p):]: v.clone() for k, v in tsd.items() if k.startswith(p)}
    s = 2.0 ** -a
    w["x_proj.weight"] = w["x_proj.weight"] * s
    if weight_free:
        g = 1.0 / (s / math.sqrt(e))                      # the norm's constant s / sqrt(e), undone
        c = (s / math.sqrt(e)) ** 2
        w["dt_proj.weight"] = w["dt_proj.weight"] * g
        w["D"] = w["D"] * c
        w["out_proj.weight"] = w["out_proj.weight"] / c
        if "out_proj.bias" in w:
            w["out_proj.bias"] = w["out_proj.bias"]       # (the bias is added after out_proj's weight: unchanged)
    return w, 1.0 / s * math.sqrt(e)


def host_falcon_mamba(target, a):
    from transformers import FalconMambaConfig, FalconMambaForCausalLM
    tc = target.config
    d = tc.to_dict()
    for k in ("architectures", "transformers_version", "_name_or_path", "model_type"):
        d.pop(k, None)
    d.update(use_mamba_kernels=False, mixer_rms_eps=1e-6)
    hc = FalconMambaConfig(**d)
    host = FalconMambaForCausalLM(hc).to(next(target.parameters()).dtype).eval()
    tsd = target.state_dict()
    sd = {k: v.clone() for k, v in tsd.items()}
    for i in range(tc.num_hidden_layers):
        w, _ = _mamba1_layer_weights(tsd, i, a, hc.mixer_rms_eps, weight_free=True)
        for k, v in w.items():
            sd[f"backbone.layers.{i}.mixer.{k}"] = v
    host.load_state_dict(sd, strict=True)
    host.tie_weights()
    return host


def host_jamba(target, a, period=3, heads=12, kv=4):
    """JambaForCausalLM (dense MLPs, attention after every `period` Mamba layers), the Mamba-1 target on its Mamba
    layers (B / C / dt norms levered: gains 2^a sqrt(e)), attention o_proj and every FFN down_proj zero."""
    from transformers import JambaConfig, JambaForCausalLM
    tc = target.config
    n = tc.num_hidden_layers
    kinds = []
    for i in range(n):
        kinds.append("mamba")
        if i % period == period - 1:
            kinds.append("attention")
    # Jamba places attention by period / offset: layer j is attention iff j % P == off; with one attention layer per
    # period + 1 layers, P = period + 1 and off = period
    P = period + 1
    assert all((k == "attention") == (j % P == period) for j, k in enumerate(kinds))
    hc = JambaConfig(vocab_size=tc.vocab_size, tie_word_embeddings=tc.tie_word_embeddings, hidden_size=tc.hidden_size,
                     intermediate_size=256, num_hidden_layers=len(kinds), num_attention_heads=heads,
                     num_key_value_heads=kv, rms_norm_eps=tc.layer_norm_epsilon, num_experts=1, num_experts_per_tok=1,
                     expert_layer_period=1, expert_layer_offset=0, attn_layer_period=P, attn_layer_offset=period,
                     use_mamba_kernels=False, mamba_d_state=tc.state_size, mamba_d_conv=tc.conv_kernel,
                     mamba_expand=tc.expand, mamba_dt_rank=tc.time_step_rank, mamba_conv_bias=tc.use_conv_bias,
                     mamba_proj_bias=tc.use_bias, pad_token_id=tc.pad_token_id, bos_token_id=tc.bos_token_id,
                     eos_token_id=tc.eos_token_id)
    host = JambaForCausalLM(hc).to(next(target.parameters()).dtype).eval()
    tsd = target.state_dict()
    sd = {k: v.clone() for k, v in host.state_dict().items()}
    sd["model.embed_tokens.weight"] = tsd["backbone.embeddings.weight"].clone()
    sd["model.final_layernorm.weight"] = tsd["backbone.norm_f.weight"].clone()
    sd["lm_head.weight"] = (sd["model.embed_tokens.weight"] if tc.tie_word_embeddings
                            else tsd["lm_head.weight"].clone())      # (tied: the shared tensor must not be overwritten)
    i = 0
    for j, k in enumerate(kinds):
        p = f"model.layers.{j}."
        sd[p + "feed_forward.down_proj.weight"].zero_()
        if k == "attention":
            sd[p + "self_attn.o_proj.weight"].zero_()
            continue
        sd[p + "input_layernorm.weight"] = tsd[f"backbone.layers.{i}.norm.weight"].clone()
        w, gain = _mamba1_layer_weights(tsd, i, a, hc.rms_norm_eps, weight_free=False)
        for name, v in w.items():
            assert sd[p + "mamba." + name].shape == v.shape, name
            sd[p + "mamba." + name] = v
        for nm in ("dt_layernorm", "b_layernorm", "c_layernorm"):
            sd[p + f"mamba.{nm}.weight"] = torch.full_like(sd[p + f"mamba.{nm}.weight"], gain)
        i += 1
    host.load_state_dict(sd, strict=True)
    if tc.tie_word_embeddings:
        host.tie_weights()
    return host


# --- runs ---------------------------------------------------------------------------------------------------------
def _x_proj_ms(target, ref, device):
    """The largest mean square of each of x_proj's dt / B / C slices (the levered norms' inputs)."""
    tc = target.config
    r, N = tc.time_step_rank, tc.state_size
    best = [0.0]

    def hook(_m, _a, out):
        for sl in (out[..., :r], out[..., r:r + N], out[..., r + N:]):
            best[0] = max(best[0], sl.float().pow(2).mean(-1).max().item())
    hs = [m.register_forward_hook(hook) for n, m in target.named_modules() if n.endswith("mixer.x_proj")]
    with torch.no_grad():
        for s in ref["seq"]:
            target(s.to(device))
    for h in hs:
        h.remove()
    return best[0]


def run(name, device="cuda:0", dtype=torch.float32, n_prompts=24, sweep=None):
    spec = PAIRS[name]
    tok, target = load_target(spec["target"], dtype)
    target.to(device)
    ref = reference(target, tok, device, n_prompts)
    rows = []
    if spec["rule"] == "m2_nonorm_lever":
        ms = max_ms(target, ref, device, lambda n, m: n.endswith("mamba.out_proj"))
        e = target.config.rms_norm_eps
        S_rec = math.sqrt(pow2_at_least(ms / e / 2.0 ** RHO_LOG2))
        grid = [S_rec] if sweep is None else [S_rec * 2.0 ** k for k in sweep]
        for S in grid:
            host = host_nonorm(target, S).to(device)
            r = score(host, ref, device)
            r.update(S_log2=math.log2(S), rho_max=ms / (e * S * S), ms_max=ms)
            rows.append(r)
            del host
    elif spec["rule"] == "pure_on_hybrid":
        host = host_bamba(target).to(device)
        rows.append(score(host, ref, device))
    else:                                                     # mamba1_lever (+ pure_on_hybrid on Jamba)
        ms = _x_proj_ms(target, ref, device)
        e = 1e-6 if spec["host"] == "falcon_mamba" else target.config.layer_norm_epsilon
        a_rec = math.ceil(0.5 * math.log2(ms / e / 2.0 ** RHO_LOG2))
        grid = [a_rec] if sweep is None else [a_rec + k for k in sweep]
        for a in grid:
            host = (host_falcon_mamba if spec["host"] == "falcon_mamba" else host_jamba)(target, a).to(device)
            r = score(host, ref, device)
            r.update(a=a, rho_max=ms * 2.0 ** (-2 * a) / e, ms_max=ms)
            rows.append(r)
            del host
    out = []
    for r in rows:
        r.update(pair=name, rule=spec["rule"], target=spec["target"], host=spec["host"], dtype=str(dtype).split(".")[-1],
                 n_prompts=n_prompts, payable=strict(r), time=time.strftime("%Y-%m-%d %H:%M"))
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--sweep", default=None, help="comma-separated log2 offsets of S (or a) around the recorded one")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    names = list(PAIRS) if a.pairs == ["all"] else a.pairs
    sweep = [int(x) for x in a.sweep.split(",")] if a.sweep else None
    for name in names:
        for r in run(name, a.device, getattr(torch, a.dtype), a.n_prompts, sweep):
            print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
