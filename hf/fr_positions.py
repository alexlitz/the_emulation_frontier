"""fr-positions: the positions-wall constructions of branch fr-positions, measured on real checkpoints.

Stock transformers code plus config only (no buffer writes, no code changes), weights written into the host:

  nope_host   a RoPE + NoPE host (SmolLM3ForCausalLM, no_rope_layers: a NoPE layer every 4th) plays an all-RoPE
              target on its RoPE layers, its NoPE layers zeroed (o_proj = 0, down_proj = 0: identity blocks; Lean
              nope_mix_host_skip). graph rule fp_nope_layer_host.
  sinusoid    a fixed sinusoid-table target (XGLM / InCoder, XGLMSinusoidalPositionalEmbedding) on a learned-abs host
              (BioGptForCausalLM: learned table, offset 2, scale_embedding, pre-LN, GeLU): the table's rows copied into
              the learned table (Lean sinusoid_sub_learned_abs). graph rule fp_sinusoid_table.

Both are exact constructions; the agreement is measured at the standard (fp32: top-1 >= 99%, >= 90% identical 32-token
greedy continuations; hf/bos_constructions.agreement) and in fp64.

Run:  python -m hf.fr_positions [--only nope,sinusoid] [--out hf/data/fr_positions.jsonl]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math

import torch

from hf.bos_constructions import _prompts, agreement, exact_norms

F64 = torch.float64
NOPE_TARGETS = ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B")
SIN_TARGETS = ("facebook/xglm-564M", "facebook/incoder-1B")


def _load(repo, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True, dtype=dtype, attn_implementation="eager")
    return tok, m.eval()


@torch.no_grad()
def nope_host(tgt, every=4):
    """SmolLM3 host for a Llama / Qwen2 target: L_t RoPE layers + one NoPE layer per (every - 1) RoPE layers."""
    from transformers import SmolLM3Config, SmolLM3ForCausalLM
    c = tgt.config
    L = c.num_hidden_layers
    n = L + -(-L // (every - 1))                                 # enough layers for L RoPE ones
    nrl = [int((i + 1) % every != 0) for i in range(n)]          # 1 = RoPE (SmolLM3: use_rope = no_rope_layers[i])
    while sum(nrl) > L and nrl[-1] == 1:                         # trim trailing RoPE layers beyond L
        nrl.pop()
    n = len(nrl)
    bias = bool(getattr(c, "attention_bias", False) or c.model_type == "qwen2")
    hc = SmolLM3Config(vocab_size=c.vocab_size, hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
                       num_hidden_layers=n, num_attention_heads=c.num_attention_heads,
                       num_key_value_heads=c.num_key_value_heads, hidden_act=c.hidden_act,
                       max_position_embeddings=c.max_position_embeddings, rms_norm_eps=c.rms_norm_eps,
                       rope_theta=c.rope_theta, rope_scaling=getattr(c, "rope_scaling", None),
                       tie_word_embeddings=c.tie_word_embeddings, attention_bias=bias, use_sliding_window=False,
                       no_rope_layers=nrl, layer_types=["full_attention"] * n,
                       bos_token_id=c.bos_token_id, eos_token_id=c.eos_token_id, pad_token_id=c.pad_token_id)
    hc._attn_implementation = "eager"
    host = SmolLM3ForCausalLM(hc).to(next(tgt.parameters()).dtype).eval()
    sd = {k: torch.zeros_like(v) for k, v in host.state_dict().items()}
    t = tgt.state_dict()
    rope_layers = [i for i, r in enumerate(nrl) if r]
    assert len(rope_layers) == L
    for k, v in t.items():
        if k.startswith("model.layers."):
            parts = k.split(".")
            parts[2] = str(rope_layers[int(parts[2])])
            sd[".".join(parts)] = v.clone()
        elif k in sd:
            sd[k] = v.clone()
    for i, r in enumerate(nrl):                                  # NoPE layers: identity blocks, norms at 1
        if not r:
            for nm in ("input_layernorm", "post_attention_layernorm"):
                sd[f"model.layers.{i}.{nm}.weight"] = torch.ones_like(sd[f"model.layers.{i}.{nm}.weight"])
    if "lm_head.weight" in sd and "lm_head.weight" not in t:
        sd["lm_head.weight"] = t["model.embed_tokens.weight"].clone()
    host.load_state_dict(sd)
    host.desc = (f"SmolLM3ForCausalLM({n} layers, NoPE layers {[i for i, r in enumerate(nrl) if not r]} zeroed, "
                 f"attention_bias {bias})")
    return host


@torch.no_grad()
def sinusoid_host(tgt):
    """BioGPT host (learned positions, offset 2) for an XGLM-layout target: same widths, the sinusoid rows copied."""
    from transformers import BioGptConfig, BioGptForCausalLM
    c = tgt.config
    hc = BioGptConfig(vocab_size=c.vocab_size, hidden_size=c.d_model, num_hidden_layers=c.num_layers,
                      num_attention_heads=c.attention_heads, intermediate_size=c.ffn_dim, hidden_act=c.activation_function,
                      max_position_embeddings=c.max_position_embeddings, layer_norm_eps=1e-5,
                      scale_embedding=c.scale_embedding, pad_token_id=c.pad_token_id, bos_token_id=c.bos_token_id,
                      eos_token_id=c.eos_token_id, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                      activation_dropout=0.0, layerdrop=0.0, tie_word_embeddings=True)
    hc._attn_implementation = "eager"
    host = BioGptForCausalLM(hc).to(next(tgt.parameters()).dtype).eval()
    tm, hm = tgt.model, host.biogpt
    hm.embed_tokens.weight.copy_(tm.embed_tokens.weight)
    S = tm.embed_positions.weights                                  # rows p + 2 (offset 2, as BioGPT's table)
    rows = hm.embed_positions.weight.shape[0]
    hm.embed_positions.weight.copy_(S[:rows].to(hm.embed_positions.weight.dtype))
    for tl, hl in zip(tm.layers, hm.layers):
        for a in ("q_proj", "k_proj", "v_proj", "out_proj"):
            getattr(hl.self_attn, a).load_state_dict(getattr(tl.self_attn, a).state_dict())
        hl.self_attn_layer_norm.load_state_dict(tl.self_attn_layer_norm.state_dict())
        hl.final_layer_norm.load_state_dict(tl.final_layer_norm.state_dict())
        hl.fc1.load_state_dict(tl.fc1.state_dict())
        hl.fc2.load_state_dict(tl.fc2.state_dict())
        hl.self_attn_layer_norm.eps = tl.self_attn_layer_norm.eps
        hl.final_layer_norm.eps = tl.final_layer_norm.eps
    hm.layer_norm.load_state_dict(tm.layer_norm.state_dict())
    hm.layer_norm.eps = tm.layer_norm.eps
    host.output_projection.weight = hm.embed_tokens.weight          # tied, as XGLM's lm_head
    host.desc = (f"BioGptForCausalLM(hidden {hc.hidden_size}, {hc.num_hidden_layers} layers, learned table "
                 f"{rows} rows = the sinusoid's)")
    return host


def measure(kind, repo, dtypes=(F64, torch.float32), n_prompts=24, n_new=32):
    tok, tgt = _load(repo)
    host = nope_host(tgt) if kind == "nope" else sinusoid_host(tgt)
    prompts = _prompts(tok, n=n_prompts)
    V = tgt.config.vocab_size
    rows = []
    for dt in dtypes:
        t, h = tgt.to(dt), host.to(dt)
        with exact_norms() if dt == F64 else contextlib.nullcontext():
            r = agreement(t, h, prompts, n_new=n_new, V=V)
        rows.append(dict(edge={"nope": "rope target <- RoPE + NoPE host (NoPE layers zeroed)",
                               "sinusoid": "sinusoidal <- learned_abs (table copied)"}[kind],
                         target=repo, host=host.desc, dtype=str(dt).replace("torch.", ""), **r))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="nope,sinusoid")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for kind, repos in (("nope", NOPE_TARGETS), ("sinusoid", SIN_TARGETS)):
        if kind in a.only.split(","):
            for r in repos:
                # (XGLM's legacy 4-D mask builder overflows in fp64 under transformers 4.57: fp32 only, the standard)
                rows += measure(kind, r, n_prompts=a.n_prompts,
                                dtypes=(F64, torch.float32) if kind == "nope" else (torch.float32,))
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()


# --- fp64 references for the refuted items ---------------------------------------------------------------------------

def _fit_residual(target, freqs, T):
    """Relative least-squares residual of a logit sequence target[Delta] (Delta = 0..T-1) by any host head whose pairs
    rotate at `freqs` (each pair's logit is a cos(w D) + b sin(w D); a zero frequency is the constant)."""
    D = torch.arange(T, dtype=F64)
    cols = [torch.ones(T, dtype=F64)] + [f(w * D) for w in freqs for f in (torch.cos, torch.sin) if w != 0]
    A = torch.stack(cols, 1)
    x = torch.linalg.lstsq(A, target[:, None]).solution
    return ((A @ x)[:, 0] - target).norm().item() / target.norm().item()


def refutations(T=2048, seed=0):
    """Each refuted positions item on a small fp64 reference: the best host of the item's kind leaves a residual."""
    g = torch.Generator().manual_seed(seed)
    D = torch.arange(T, dtype=F64)
    P = 4
    wA = 100.0 ** (-torch.arange(P, dtype=F64) / P)
    wB = wA * torch.tensor([1.0, 0.7, 0.5, 0.3], dtype=F64)        # a second table (another theta / scaling)
    c = torch.randn(P, 2, generator=g, dtype=F64)
    logit = lambda w: sum(c[p, 0] * torch.cos(w[p] * D) + c[p, 1] * torch.sin(w[p] * D) for p in range(P))
    out = {}
    # two RoPE tables (Olmo-3 / Mellum2 / PLaMo-3) on a one-table host of the same P pairs: the table serving layer A
    # (it must hold A's P distinct frequencies, Lean partial_rope_too_few) cannot serve layer B
    out["two_tables_on_one"] = _fit_residual(logit(wB), wA.tolist(), T)
    # RoPE + NoPE target (K-EXAONE / Command-R7B) on a one-table host: the table holding the RoPE layer's frequencies
    # leaves the NoPE layer's content (a constant per pair) rotating
    qk = torch.randn(P, 2, generator=g, dtype=F64)
    nope = sum(qk[p, 0] * torch.cos(wA[p] * D * 0) for p in range(P)) + 0 * D
    rot_cos = lambda w: sum(qk[p, 0] * torch.cos(w[p] * D) + qk[p, 1] * torch.sin(w[p] * D) for p in range(P))
    out["nope_layer_on_rotating_table"] = (rot_cos(wA) - nope).norm().item() / nope.norm().item()
    # a pairs-free target class member (longrope factors: every frequency below 1) on a fixed host (default
    # theta^(-2i/d), any theta): the host's pair 0 is at frequency 1 for every theta (Lean rope_theta_cannot_reach_nope),
    # so P - 1 host pairs are left for P distinct target frequencies (Lean partial_rope_too_few); best theta on a grid
    wr = 0.5 * wA
    best = min(_fit_residual(logit(wr), (th ** (-torch.arange(P, dtype=F64) / P)).tolist(), T)
               for th in torch.logspace(1, 6, 200, dtype=F64))
    out["pairs_free_class_on_fixed_host"] = best
    # ALiBi (Baichuan-13B / PhoGPT / MPT) on a RoPE host of the target's head width: the content uses every pair at
    # frequency 0 (a content pair at w != 0 rotates, Lean content_pair_rotates), so no pair is left for the ramp
    m = 0.05
    out["alibi_no_spare_pair"] = _fit_residual(-m * D, [], T)
    return out


def xglm_spec(model):
    """An XGLM-layout target (fixed sinusoid table) as the start-token rule's table spec (hf/bos_constructions
    gpt2_spec / opt_spec): the scaled token table, the sinusoid rows read from position 0 (offset 2), the fairseq
    pre-LN layers. The sinusoid is then played as a learned table (graph rule fp_sinusoid_table)."""
    m, c = model.model, model.config
    d = c.d_model
    layers = []
    for b in m.layers:
        a = b.self_attn
        layers.append(dict(ln1=(b.self_attn_layer_norm.weight, b.self_attn_layer_norm.bias),
                           wq=a.q_proj.weight, bq=a.q_proj.bias, wk=a.k_proj.weight, bk=a.k_proj.bias,
                           wv=a.v_proj.weight, bv=a.v_proj.bias, wo=a.out_proj.weight, bo=a.out_proj.bias,
                           ln2=(b.final_layer_norm.weight, b.final_layer_norm.bias),
                           w1=b.fc1.weight, b1=b.fc1.bias, w2=b.fc2.weight, b2=b.fc2.bias))
    scale = math.sqrt(d) if c.scale_embedding else 1.0
    off = m.embed_positions.offset
    return dict(E=m.embed_tokens.weight * scale, Ptab=m.embed_positions.weights[off:off + c.max_position_embeddings],
                head=model.lm_head.weight, bos=c.bos_token_id, eps=m.layers[0].self_attn_layer_norm.eps,
                act=c.activation_function, H=c.attention_heads, inter=c.ffn_dim,
                lnf=(m.layer_norm.weight, m.layer_norm.bias), layers=layers, name="xglm")
