"""Two walls that fall once the input begins with ONE fixed start token (a BOS-like marker).

The marker is the same token for every input. Both constructions are exact (Lean:
proofs/lean, proofs/PROOFS.md rows `learned_abs_two_block` and `sink_sub_plain`),
and both are built here in stock `transformers` modules.

1. LEARNED-ABSOLUTE POSITIONS ON A POSITION-FREE HOST (`learned_abs_host`)
   Target: GPT-2 or OPT (a learned table P_t added to the token embedding). The target reads the
   marker as its own BOS at position 0, and the marker appears nowhere else. Host: a stock
   LlamaForCausalLM with RoPE's inv_freq set to 0 (NoPE), width d + 5, and two extra blocks in front
   of the target's layers. The five extra channels are s, b, kappa, u, v:
     embedding  [e(x); s(x), b(x), 1, 0, 0]. b flags the marker. s(x) pads every row to the same
                squared norm C, so block 1's pre-norm divides every token by the same rms.
     block 1    attention, head 0: the query reads kappa and the key reads b. The marker's key gets
                logit L, every other key 0, and the value is b. So u_t = e^L / (e^L + t), the
                marker's weight. (L = 0 is the Lean construction's zero-logit head, u = 1/(t+1).
                The default L = log T spaces the knots no closer than 1/(4T) instead of 1/T^2.)
                The MLP writes v_t = sqrt(1 - u_t^2), so after block 1 the residual's squared norm
                is C + 1 at every position, for every token.
     block 2    attention writes nothing. The MLP reads at a constant rms. It writes the table P_t
                into the content channels and erases the five extra channels.
   The result is [e(x_t) + P_t; 0] at every position. The target's layers then run on the
   zero-padded residual, with the norm weight scaled by sqrt(d / (d + 5)) and eps' = eps d / (d + 5).

   The MLPs need only T + 1 units, plus 5 erasers in block 2, with ReLU, SiLU, GELU and the other
   ramp-like activations. Every
   quantity the extra MLPs read is a function of the position alone: u_t, and the constant kappa
   channel divided by an rms that depends only on t. So an MLP restricted to those channels computes
   sum_k c_k h_k(t), where h_k(t) is unit k's value at position t. Writing an arbitrary table Y_t at
   the T + 1 positions is the linear system H c = Y, with H[t, k] = h_k(t). The units are kinks
   act(beta_k (u - m_k)), with m_k between consecutive knots and beta_k = c_beta / gap_k, plus a
   constant unit and a ramp. As beta -> inf, H tends to the ReLU kink matrix, which is triangular and
   invertible. So H is invertible at a finite beta for ReLU, SiLU, GELU and swigluoai alike, and
   Y = H c holds EXACTLY at the knots. This is not a limit: SiLU's distance from ReLU is absorbed
   into c. The erasers are exact linear units. In a GLU host, act(const) * up(y) is linear in y. In a
   vanilla host with an act of the form x F(x) where F(x) + F(-x) = 1 (SiLU, GELU, gelu_new,
   swigluoai), act(z) - act(-z) = z.
     The units are SHARED across the d content coordinates (one down-projection column per unit), so
   block 2 needs T + 1 + 5 units, not the Lean construction's 5 + d (T + 3). The Lean writes each
   coordinate's interpolant separately. Its kinks relu(u - t_i) are identical across coordinates,
   so sharing them is sound in the proof's own terms.

   GPT-2 / OPT on Llama: LayerNorm -> RMSNorm is the exact writer-centring fold
   (norm/layernorm_rmsnorm.py). Every residual writer is centred, and so are the embedding rows and
   the table P. LN's beta goes into the next linear's bias (Llama attention_bias=True / mlp_bias=True).
   The ungated MLP is the gated one with up = 0 and up-bias 1, and the target's own activation is
   used (hidden_act gelu_new / relu: a stock ACT2FN entry). One module attribute is not
   config-reachable: the final norm's beta becomes the per-vocab logit shift W_U beta_f. A
   constant channel cannot carry it past the final RMSNorm (`prenorm_readout_obstruction`), so the
   stock bias-free lm_head gets a bias Parameter (the LayerNorm fold's cost, as in
   norm/layernorm_rmsnorm.py, not the marker's). The two-block prefix itself is bias-free and runs on
   a default stock Llama (SiLU, no biases): `prefix_check`.

   Without a marker this is impossible at any depth (`no_learned_abs_without_marker`): a constant
   sequence stays constant.

2. ATTENTION SINKS ON A SINK-FREE HOST (`sink_host`)
   Target: gpt-oss-tinystories (learned per-head sink logit s_h: off-by-one softmax). Host: the same
   residual stack at width d + 1, with every attention replaced by a stock Qwen2Attention (q / k / v
   biases, no o bias, no sinks). Its head_dim is the target's + 2, and that spare pair's RoPE
   frequency is 0. The marker is a host-only token prepended at position 0 (an extra vocabulary row;
   logits are read on the target's vocabulary). It sees only itself, so its residual is a FIXED
   vector at every layer, the same for every input. Per head:
     query   q-bias puts the constant s_h / scale on the spare dim of every query;
     key     the marker's key is 1 on the spare dim and 0 on every content dim. A reserved residual
             channel sigma, nonzero only at the marker, carries it. The k_proj column on sigma cancels
             the marker's fixed W_k n_0 + b_k, layer by layer. Real tokens have sigma = 0 and a zero
             spare key. So the marker's logit is s_h for every query, and the real logits are
             untouched;
     value   0 on the content dims for the marker (the same sigma-column cancellation). The spare value
             dim is 1 for EVERY key, marker included, so it reads 1 at every query, and the o_proj
             column on it is the target's o-bias. A Qwen2 host has no o bias, and this pays it.
   That is exactly `sink_sub_plain`: query (q, 1), sink key (0, s), real keys (k, 0), sink value 0.
   It needs a q bias. A k bias alone puts the constant on every key, and softmax cancels it. A reserved
   query channel is not constant after RMSNorm (`prenorm_readout_obstruction`). It also needs a
   zero-frequency pair: in a rotating pair the sink logit depends on the query position
   (`sink_needs_zero_freq`). Limits: a sliding-window layer drops the marker once the query is more
   than `window` tokens past it (a window-plus-sink mask is not a stock mask), so it is exact up to the
   window (512 here). And a target that itself reads a BOS needs the marker as an extra token BEFORE
   that BOS, because one key cannot be both the sink and the BOS.
   The host's MoE / norms here are gpt-oss's own (padded to d + 1). MoE -> dense is a separate edge.

PRECISION. The learned-abs table is exact in real arithmetic, but a stock MLP forms it as a sum of
ramps over T + 1 knots spaced >= 1/(4T) apart. The terms grow like T^2 |dP| (~4e6 for GPT-2's wpe), so
the sums cancel. fp64 keeps the table to ~1e-10. fp32 keeps it to ~5e-3 over the first 128 positions
and ~1.6e-2 over all 1,024. No choice of ramps avoids the T^2: every unit with an unbounded activation
is live on one side of its kink. The kinks are therefore oriented to be live at the LATER positions
(kink_gates orient="below"), which keeps the early positions, the ones every token attends to, the
most accurate. The sink construction has no such sums: it is at rounding level in both precisions.

Measured (`python -m hf.bos_constructions`, rows in hf/data/bos_constructions.jsonl): real text, the
act/act_swap_agreement prompts (24, plus the BOS for GPT-2; the story prompts for gpt-oss), each target's
own greedy 64-token continuation. The fp64 runs use the stock RMSNorm formula at fp64 (`exact_norms`:
the stock forward casts to fp32).
  GPT-2 in Llama(773, 14 layers, gelu_new)    fp64 max|dlogit| 4.5e-9, top-1 100%, greedy 100% / 100%
                                              fp32 max|dlogit| 0.84,   top-1 100%, greedy 100% / 100%
  OPT-125m in Llama(773, 14 layers, relu)     fp64 7.9e-11, 100%, 100% / 100%
                                              fp32 6.8e-2, top-1 99.93%, greedy 100% / 100%
  gpt-oss-tinystories, Qwen2Attention host    fp64 2.7e-14, fp32 1.4e-5, 100%, 100% / 100%
  (OPT generated its BOS again in one continuation, which is scored up to that token.)

Run:  python -m hf.bos_constructions [--device cuda:0] [--only gpt2,opt,sink,prefix]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time

import torch

F64 = torch.float64
ATTN = "sdpa"          # the eager Llama / Qwen2 paths round the softmax to fp32; sdpa keeps fp64


# --- numerics: run the stock RMSNorms at the input dtype ---------------------------------------------

@contextlib.contextmanager
def exact_norms():
    """LlamaRMSNorm / GptOssRMSNorm cast to fp32 inside forward. For the fp64 runs they compute the
    same formula at the input dtype (a no-op in fp32)."""
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    def fwd(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    saved = {c: c.forward for c in (LlamaRMSNorm, GptOssRMSNorm)}
    for c in saved:
        c.forward = fwd
    try:
        yield
    finally:
        for c, f in saved.items():
            c.forward = f


def _act(name):
    from transformers.activations import ACT2FN
    return ACT2FN[name]


# --- knot interpolation with any activation ----------------------------------------------------------

def marker_weights(T, L):
    """u_t = e^L / (e^L + t), t = 0..T: the marker's attention weight with logit L (others 0)."""
    t = torch.arange(T + 1, dtype=F64)
    return math.exp(L) / (math.exp(L) + t)


def kink_gates(u, r_ref, c_beta=40.0, orient="below"):
    """Gate coefficients (on u, on kappa) of T + 1 units whose matrix over the knots u is invertible:
    a constant unit, a ramp that is positive at every knot, and a kink between each further pair of
    neighbouring knots, of steepness c_beta / gap. r_ref ~ the rms the gate is read at.

    orient="below": kink k is act(beta (m_k - u)), live at the knots BELOW m_k, i.e. the LATER
    positions (u_t decreases with t). Then position t's sum over units runs over the kinks of earlier
    positions only, so the early positions (which every later token attends to) carry the smallest
    fp32 sums. "above" is the Lean's relu(u - t_i). Both span the same functions."""
    v, _ = torch.sort(u)
    span = (v[-1] - v[0]).item()
    if orient == "above":
        a = [0.0, r_ref / span]
        g = [r_ref, -(v[0].item() - span) * r_ref / span]
        js = range(2, len(v))
    else:
        a = [0.0, -r_ref / span]
        g = [r_ref, (v[-1].item() + span) * r_ref / span]
        js = range(1, len(v) - 1)
    s = 1.0 if orient == "above" else -1.0
    for j in js:
        gap = (v[j] - v[j - 1]).item()
        beta, m = c_beta / gap, 0.5 * (v[j] + v[j - 1]).item()
        a.append(s * beta * r_ref)
        g.append(-s * beta * m * r_ref)
    return torch.tensor(a, dtype=F64), torch.tensor(g, dtype=F64)


def solve_units(act, a, g, u, r, up, Y):
    """Down-projection columns c (K x out) with sum_k act((a_k u_t + g_k) / r_t) up_t c_k = Y_t at
    every knot. Returns (c, max residual)."""
    z = (u[:, None] * a[None, :] + g[None, :]) / r[:, None]
    H = act(z) * up[:, None]
    c = torch.linalg.solve(H, Y)
    return c, (H @ c - Y).abs().max().item()


# --- the two-block prefix ----------------------------------------------------------------------------

@torch.no_grad()
def write_bos_prefix(host, E, Ptab, bos, L=None, c_beta=40.0):
    """Write the embedding and layers 0 and 1 of a Llama-architecture host of width d + 5 so that its
    residual after layer 1 is [E[x_t] + Ptab[t]; 0] for every input with `bos` at position 0 only.
    E: (V, d) token rows, Ptab: (T + 1, d). Bias-free: any biases of layers 0 / 1 are zeroed.
    Returns diagnostics (knot gap, solve residuals)."""
    cfg = host.config
    D = cfg.hidden_size
    V, d = E.shape
    assert D == d + 5, (D, d)
    T = Ptab.shape[0] - 1
    eps = cfg.rms_norm_eps
    act = _act(cfg.hidden_act)
    inter = cfg.intermediate_size
    assert inter >= T + 6, f"intermediate_size {inter} < T + 6 = {T + 6}"
    if L is None:
        L = math.log(max(T, 1))
    S, B, K, U, Vc = d, d + 1, d + 2, d + 3, d + 4            # the extra channels s, b, kappa, u, v

    # embedding: every row padded to squared norm C
    E = E.to(F64)
    bflag = torch.zeros(V, dtype=F64)
    bflag[bos] = 1.0
    n2 = E.pow(2).sum(1) + bflag ** 2 + 1.0
    C = n2.max().item() + 1.0
    emb = torch.zeros(V, D, dtype=F64)
    emb[:, :d] = E
    emb[:, S] = (C - n2).sqrt()
    emb[:, B] = bflag
    emb[:, K] = 1.0
    host.model.embed_tokens.weight.copy_(emb)

    for layer in host.model.layers[:2]:
        for p in layer.parameters():
            p.zero_()
        layer.input_layernorm.weight.fill_(1.0)
        layer.post_attention_layernorm.weight.fill_(1.0)

    r0 = math.sqrt(C / D + eps)
    u = marker_weights(T, L)
    r1 = ((C + u ** 2) / D + eps).sqrt()
    r2 = math.sqrt((C + 1.0) / D + eps)

    # block 1 attention, head 0: logit L on the marker's key, value b
    at = host.model.layers[0].self_attn
    at.q_proj.weight[0, K] = 1.0
    at.k_proj.weight[0, B] = L * r0 * r0 / at.scaling
    at.v_proj.weight[0, B] = r0
    at.o_proj.weight[U, 0] = 1.0

    # block 1 MLP: v_t = sqrt(1 - u_t^2) on channel v
    m1 = host.model.layers[0].mlp
    a, g = kink_gates(u, float(r1.mean()), c_beta)
    up1 = float(r1.mean()) / r1
    c1, res1 = solve_units(act, a, g, u, r1, up1, (1 - u ** 2).clamp(min=0).sqrt()[:, None])
    n = T + 1
    m1.gate_proj.weight[:n, U] = a
    m1.gate_proj.weight[:n, K] = g
    m1.up_proj.weight[:n, K] = float(r1.mean())
    m1.down_proj.weight[Vc, :n] = c1[:, 0]

    # block 2 MLP: the table on the content channels, and the five erasers
    m2 = host.model.layers[1].mlp
    r2v = torch.full_like(u, r2)
    a, g = kink_gates(u, r2, c_beta)
    c2, res2 = solve_units(act, a, g, u, r2v, torch.ones_like(u), Ptab.to(F64))
    m2.gate_proj.weight[:n, U] = a
    m2.gate_proj.weight[:n, K] = g
    m2.up_proj.weight[:n, K] = r2
    m2.down_proj.weight[:d, :n] = c2.T
    a1 = act(torch.tensor(1.0, dtype=F64)).item()               # gate reads kappa: act(1), a constant
    for j in range(5):
        k = n + j
        m2.gate_proj.weight[k, K] = r2
        m2.up_proj.weight[k, d + j] = r2
        m2.down_proj.weight[d + j, k] = -1.0 / a1
    v = u.sort().values
    return dict(T=T, L=L, C=C, min_gap=(v[1:] - v[:-1]).min().item(), solve_res=max(res1, res2))


def _zero_rope(host):
    from hf.partial_rope_host import set_inv_freq
    set_inv_freq(host.model.rotary_emb, torch.zeros_like(host.model.rotary_emb.inv_freq))


# --- learned-abs targets (GPT-2, OPT) read off the stock modules --------------------------------------

def _center(W, dim=0):
    """Centre along `dim` (the residual coordinate): P W with P = I - 11^T / d."""
    return W - W.mean(dim, keepdim=True)


def gpt2_spec(model):
    tr = model.transformer
    c = model.config
    d = c.n_embd
    layers = []
    for b in tr.h:
        W = b.attn.c_attn.weight.detach().T.to(F64)                # (3d, d), Conv1D -> Linear
        bb = b.attn.c_attn.bias.detach().to(F64)
        layers.append(dict(ln1=(b.ln_1.weight, b.ln_1.bias), wq=W[:d], wk=W[d:2 * d], wv=W[2 * d:],
                           bq=bb[:d], bk=bb[d:2 * d], bv=bb[2 * d:],
                           wo=b.attn.c_proj.weight.T, bo=b.attn.c_proj.bias,
                           ln2=(b.ln_2.weight, b.ln_2.bias), w1=b.mlp.c_fc.weight.T, b1=b.mlp.c_fc.bias,
                           w2=b.mlp.c_proj.weight.T, b2=b.mlp.c_proj.bias))
    return dict(E=tr.wte.weight, Ptab=tr.wpe.weight, head=model.lm_head.weight, bos=c.bos_token_id,
                eps=c.layer_norm_epsilon, act=c.activation_function, H=c.n_head, inter=c.n_inner or 4 * d,
                lnf=(tr.ln_f.weight, tr.ln_f.bias), layers=layers, name="gpt2")


def opt_spec(model):
    dec = model.model.decoder
    c = model.config
    assert dec.project_in is None and dec.project_out is None and c.do_layer_norm_before
    layers = []
    for b in dec.layers:
        a = b.self_attn
        layers.append(dict(ln1=(b.self_attn_layer_norm.weight, b.self_attn_layer_norm.bias),
                           wq=a.q_proj.weight, bq=a.q_proj.bias, wk=a.k_proj.weight, bk=a.k_proj.bias,
                           wv=a.v_proj.weight, bv=a.v_proj.bias, wo=a.out_proj.weight, bo=a.out_proj.bias,
                           ln2=(b.final_layer_norm.weight, b.final_layer_norm.bias),
                           w1=b.fc1.weight, b1=b.fc1.bias, w2=b.fc2.weight, b2=b.fc2.bias))
    off = dec.embed_positions.offset                                # position t reads row t + 2
    eps = dec.layers[0].self_attn_layer_norm.eps
    return dict(E=dec.embed_tokens.weight, Ptab=dec.embed_positions.weight[off:], head=model.lm_head.weight,
                bos=c.bos_token_id, eps=eps, act=c.activation_function, H=c.num_attention_heads,
                inter=c.ffn_dim, lnf=(dec.final_layer_norm.weight, dec.final_layer_norm.bias), layers=layers,
                name="opt")


@torch.no_grad()
def learned_abs_host(spec, L=None, c_beta=40.0, logit_bias=True):
    """A stock LlamaForCausalLM (NoPE via inv_freq = 0, width d + 5, L_t + 2 layers) reproducing the
    learned-abs target `spec` (gpt2_spec / opt_spec) on inputs that start with its BOS. fp64."""
    from transformers import LlamaConfig, LlamaForCausalLM
    f = lambda x: x.detach().to(F64)
    E, Ptab, head = f(spec["E"]), f(spec["Ptab"]), f(spec["head"])
    V, d = E.shape
    T = Ptab.shape[0] - 1
    D = d + 5
    H = spec["H"]
    hd = d // H
    cfg = LlamaConfig(vocab_size=V, hidden_size=D, intermediate_size=max(spec["inter"], T + 6),
                      num_hidden_layers=len(spec["layers"]) + 2, num_attention_heads=H, num_key_value_heads=H,
                      head_dim=hd, hidden_act=spec["act"], attention_bias=True, mlp_bias=True,
                      rms_norm_eps=spec["eps"] * d / D, max_position_embeddings=T + 1, tie_word_embeddings=False,
                      bos_token_id=spec["bos"], eos_token_id=None, pad_token_id=None, attn_implementation=ATTN)
    host = LlamaForCausalLM(cfg).to(F64).eval()
    _zero_rope(host)
    diag = write_bos_prefix(host, _center(E, 1), _center(Ptab, 1), spec["bos"], L=L, c_beta=c_beta)
    k = math.sqrt(d / D)
    for lh, s in zip(host.model.layers[2:], spec["layers"]):
        for p in lh.parameters():
            p.zero_()
        g1, b1n = f(s["ln1"][0]), f(s["ln1"][1])
        g2, b2n = f(s["ln2"][0]), f(s["ln2"][1])
        lh.input_layernorm.weight[:d] = g1 * k
        lh.post_attention_layernorm.weight[:d] = g2 * k
        at = lh.self_attn
        for lin, w, b in ((at.q_proj, "wq", "bq"), (at.k_proj, "wk", "bk"), (at.v_proj, "wv", "bv")):
            lin.weight[:, :d] = f(s[w])
            lin.bias.copy_(f(s[w]) @ b1n + f(s[b]))                  # LN beta folds into the bias
        at.o_proj.weight[:d] = _center(f(s["wo"]))
        at.o_proj.bias[:d] = _center(f(s["bo"]))
        m = lh.mlp
        m.gate_proj.weight[:, :d] = f(s["w1"])
        m.gate_proj.bias.copy_(f(s["w1"]) @ b2n + f(s["b1"]))
        m.up_proj.bias.fill_(1.0)                                    # ungated: up = 1
        m.down_proj.weight[:d] = _center(f(s["w2"]))
        m.down_proj.bias[:d] = _center(f(s["b2"]))
    gf, bf = f(spec["lnf"][0]), f(spec["lnf"][1])
    host.model.norm.weight.zero_()
    host.model.norm.weight[:d] = gf * k
    host.lm_head.weight.zero_()
    host.lm_head.weight[:, :d] = head
    if logit_bias:                         # the final LN's beta: a per-vocab logit shift (see docstring)
        host.lm_head.bias = torch.nn.Parameter(head @ bf)
    host.bos_diag = diag
    return host


@torch.no_grad()
def prefix_check(spec, act="silu", n_seq=4, T_seq=48, L=None, seed=0, device="cpu", dtype=F64):
    """The two-block prefix alone on a DEFAULT stock Llama (bias-free, `act` MLPs, NoPE): max |residual
    after layer 1 - [E x_t + P_t; 0]| over random token sequences that start with the BOS; the first
    sequence spans all T + 1 positions. Built in fp64, run at `dtype`."""
    from transformers import LlamaConfig, LlamaForCausalLM
    E, Ptab = spec["E"].detach().to(F64), spec["Ptab"].detach().to(F64)
    V, d = E.shape
    T = Ptab.shape[0] - 1
    cfg = LlamaConfig(vocab_size=V, hidden_size=d + 5, intermediate_size=T + 6, num_hidden_layers=2,
                      num_attention_heads=spec["H"], num_key_value_heads=spec["H"], head_dim=d // spec["H"],
                      hidden_act=act, rms_norm_eps=1e-5, max_position_embeddings=T + 1,
                      attn_implementation=ATTN)
    assert not cfg.attention_bias and not cfg.mlp_bias
    host = LlamaForCausalLM(cfg).to(F64).eval()
    _zero_rope(host)
    diag = write_bos_prefix(host, E, Ptab, spec["bos"], L=L)
    host = host.to(device=device, dtype=dtype)
    gen = torch.Generator().manual_seed(seed)
    err = err128 = 0.0
    out = {}
    hook = host.model.layers[1].register_forward_hook(        # the residual after block 2 (pre final norm)
        lambda m, a, o: out.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    with exact_norms():
        for i in range(n_seq):
            n = T + 1 if i == 0 else T_seq                         # one sequence spans every position
            ids = torch.randint(0, V, (1, n), generator=gen)
            ids[ids == spec["bos"]] = (spec["bos"] + 1) % V
            ids[0, 0] = spec["bos"]
            host.model(ids.to(device))
            hs = out["h"][0].cpu().double()
            want = torch.zeros(n, d + 5, dtype=F64)
            want[:, :d] = E[ids[0]] + Ptab[:n]
            e = (hs - want).abs().max(1).values
            err = max(err, e.max().item())
            err128 = max(err128, e[:128].max().item())
    hook.remove()
    return dict(act=act, dtype=str(dtype).replace("torch.", ""), max_err=err, max_err_first128=err128,
                scale=(E.abs().max() + Ptab.abs().max()).item(), **diag)


# --- sinks: gpt-oss on a sink-free Qwen2-attention host -----------------------------------------------

SINK_REPO = "AlexKoff88/gpt-oss-tinystories"
# the checkpoint's config is transformers-5 (rope_parameters); under 4.x the same RoPE is these keys
SINK_LOAD_KW = dict(rope_theta=1e4, rope_scaling=None)


@torch.no_grad()
def sink_host(target, sigma0=1.0):
    """GptOssForCausalLM at width d + 1, vocab V + 1 (the marker = id V), every attention a stock
    Qwen2Attention with head_dim + 2 (spare pair at frequency 0) and q / k / v biases, no sinks.
    Reproduces `target` (fp64) on [marker] + x, logits read on [:V] from position 1 on."""
    from transformers import GptOssConfig, GptOssForCausalLM, Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding

    from hf.partial_rope_host import Host, place, set_inv_freq, target_from
    tc = target.config
    d, V, H, KV, hd = tc.hidden_size, tc.vocab_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    D, hdh = d + 1, hd + 2
    SP_QK, SP_V = hd // 2, hd                  # host dim hd/2 pairs with hd+1 (pair hd/2: the spare pair)
    hc = GptOssConfig(**{**tc.to_dict(), "hidden_size": D, "vocab_size": V + 1, "tie_word_embeddings": False,
                         "rms_norm_eps": tc.rms_norm_eps * d / D})
    hc._attn_implementation = "eager"
    host = GptOssForCausalLM(hc).to(F64).eval()
    qc = Qwen2Config(hidden_size=D, num_attention_heads=H, num_key_value_heads=KV, head_dim=hdh,
                     num_hidden_layers=tc.num_hidden_layers, layer_types=tc.layer_types,
                     sliding_window=tc.sliding_window, use_sliding_window=True, max_window_layers=0,
                     rope_theta=tc.rope_theta, max_position_embeddings=tc.max_position_embeddings,
                     attn_implementation=ATTN)
    h = Host(H=H, KV=KV, hd=hdh, rot=hdh, interleaved=False, scale=hdh ** -0.5)
    k = math.sqrt(d / D)
    f = lambda x: x.detach().to(F64)
    for i, (lt, lh) in enumerate(zip(target.model.layers, host.model.layers)):
        for p in lh.parameters():
            p.zero_()
        lh.input_layernorm.weight[:d] = f(lt.input_layernorm.weight) * k
        lh.input_layernorm.weight[d] = 1.0
        lh.post_attention_layernorm.weight[:d] = f(lt.post_attention_layernorm.weight) * k
        r, e = lh.mlp.router, lh.mlp.experts
        r.weight[:, :d] = f(lt.mlp.router.weight)
        r.bias.copy_(f(lt.mlp.router.bias))
        e.gate_up_proj[:, :d] = f(lt.mlp.experts.gate_up_proj)
        e.gate_up_proj_bias.copy_(f(lt.mlp.experts.gate_up_proj_bias))
        e.down_proj[:, :, :d] = f(lt.mlp.experts.down_proj)
        e.down_proj_bias[:, :d] = f(lt.mlp.experts.down_proj_bias)
        t = target_from(lt.self_attn, target.model.rotary_emb, tc)
        t = type(t)(**{kk: (f(vv) if torch.is_tensor(vv) else vv) for kk, vv in t.__dict__.items()})
        P = place(t, h)
        assert P["inv_freq"][SP_QK].item() == 0.0
        a = Qwen2Attention(qc, i).to(F64).eval()
        a.q_proj.weight.zero_()
        a.q_proj.weight[:, :d] = P["wq"]
        a.q_proj.bias.copy_(P["bq"])
        a.k_proj.weight.zero_()
        a.k_proj.weight[:, :d] = P["wk"]
        a.k_proj.bias.copy_(P["bk"])
        a.v_proj.weight.zero_()
        a.v_proj.weight[:, :d] = P["wv"]
        a.v_proj.bias.copy_(P["bv"])
        a.o_proj.weight.zero_()
        a.o_proj.weight[:d] = P["wo"]
        sinks = f(lt.self_attn.sinks)
        for q in range(H):                     # constant spare query: s_h / scale
            a.q_proj.bias[q * hdh + SP_QK] = sinks[q] / a.scaling
        for g in range(KV):                    # spare value dim = 1 for every key
            a.v_proj.bias[g * hdh + SP_V] = 1.0
        a.o_proj.weight[:d, SP_V] = f(lt.self_attn.o_proj.bias)    # head 0's spare value dim -> o bias
        lh.self_attn = a
        inv = P["inv_freq"]
    rot = Qwen2RotaryEmbedding(qc).to(F64)
    set_inv_freq(rot, inv.to(F64))
    host.model.rotary_emb = rot
    host.model.norm.weight.zero_()
    host.model.norm.weight[:d] = f(target.model.norm.weight) * k
    host.model.embed_tokens.weight.zero_()
    host.model.embed_tokens.weight[:V, :d] = f(target.model.embed_tokens.weight)
    host.model.embed_tokens.weight[V, d] = sigma0
    host.lm_head.weight.zero_()
    host.lm_head.weight[:V, :d] = f(target.lm_head.weight)
    # the marker's key / value, layer by layer: its residual is fixed (it sees only itself)
    mark = torch.tensor([[V]])
    kd = torch.zeros(KV * hdh, dtype=F64)
    vd = torch.zeros(KV * hdh, dtype=F64)
    for g in range(KV):
        kd[g * hdh + SP_QK] = 1.0
        vd[g * hdh + SP_V] = 1.0
    with exact_norms():
        for i, lh in enumerate(host.model.layers):
            x0 = host.model(mark, output_hidden_states=True).hidden_states[i][0, 0]
            n0 = lh.input_layernorm(x0)
            a = lh.self_attn
            for lin, want in ((a.k_proj, kd), (a.v_proj, vd)):
                lin.weight[:, d] = 0.0
                lin.weight[:, d] = (want - lin(n0)) / n0[d]
    host.marker_id = V
    return host


# --- agreement on real text --------------------------------------------------------------------------

@torch.no_grad()
def _greedy(model, ids, n_new, prefix=None, V=None):
    """Greedy continuation with the KV cache. With a host-only prefix the host's position ids start at
    -len(prefix), so every real token has the target's position and RoPE's fp32 cos / sin tables are
    the same numbers in both models. (The marker's spare pair is unrotated and its content key is 0,
    so its own position does not matter.)"""
    off = 0 if prefix is None else prefix.shape[1]
    x = ids if prefix is None else torch.cat([prefix, ids], 1)
    n = x.shape[1]
    out = model(x, position_ids=torch.arange(n, device=x.device)[None] - off, use_cache=True)
    past, nxt, toks = out.past_key_values, out.logits[:, -1, :V].argmax(-1), []
    for _ in range(n_new):
        toks.append(nxt)
        if len(toks) == n_new:
            break
        out = model(nxt[:, None], position_ids=torch.tensor([[n - off]], device=x.device),
                    past_key_values=past, use_cache=True)
        n += 1
        past, nxt = out.past_key_values, out.logits[:, -1, :V].argmax(-1)
    return torch.stack(toks, 1)


@torch.no_grad()
def agreement(target, host, prompt_ids, n_new=64, prefix=None, V=None, bos=None):
    """The agreement harness on real text (as act/act_swap_agreement.py): on the TARGET's own greedy
    n_new-token continuation of each prompt, the max |logit diff| over every position, teacher-forced
    top-1 agreement on the continuation, and greedy identity of the host's own 32 / n_new tokens.
    prefix: tokens the host alone reads first (the sink marker); V: the target vocabulary. bos: the
    learned-abs marker, which the construction assumes at position 0 only; a continuation in which the
    target generates it again is scored up to that token (counted in bos_repeats)."""
    dev = next(host.parameters()).device
    off = 0 if prefix is None else prefix.shape[1]
    pre = None if prefix is None else prefix.to(dev)
    md = mlp = 0.0
    top1 = n = ex32 = exn = rebos = k = 0
    for ids in prompt_ids:
        ids = ids.to(dev)
        ref = _greedy(target, ids, n_new)
        if bos is not None and (ref[0] == bos).any():       # the marker must not recur: score up to it
            rebos += 1
            ref = ref[:, : int((ref[0] == bos).nonzero()[0, 0])]
        if ref.shape[1] == 0:
            continue
        seq = torch.cat([ids, ref], 1)
        lt = target(seq).logits[0].double()
        hseq = seq if pre is None else torch.cat([pre, seq], 1)
        hpos = torch.arange(hseq.shape[1], device=dev)[None] - off
        lh = host(hseq, position_ids=hpos).logits[0, off:, :V].double()
        md = max(md, (lt - lh).abs().max().item())
        mlp = max(mlp, (lt.log_softmax(-1) - lh.log_softmax(-1)).abs().max().item())
        P = ids.shape[1]
        top1 += (lt[P - 1:-1].argmax(-1) == lh[P - 1:-1].argmax(-1)).sum().item()
        n += ref.shape[1]
        same = (_greedy(host, ids, ref.shape[1], pre, V)[0] == ref[0])
        ex32 += bool(same[:32].all())
        exn += bool(same.all())
        k += 1
    return dict(max_logit_diff=md, max_logprob_diff=mlp, top1=top1 / n, ex32=ex32 / k, ex64=exn / k,
                n_prompts=k, bos_repeats=rebos)


def _prompts(tok, story=False, bos=None, n=24):
    from act.act_swap_agreement import PROMPTS, STORY_PROMPTS
    out = []
    for p in (STORY_PROMPTS if story else PROMPTS)[:n]:
        ids = tok(p, return_tensors="pt").input_ids
        if bos is not None and ids[0, 0].item() != bos:
            ids = torch.cat([torch.tensor([[bos]]), ids], 1)
        out.append(ids)
    return out


def measure_learned_abs(repo, dtypes=(F64, torch.float32), device="cpu", n_prompts=24, n_new=64):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    tgt = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True, dtype=F64,
                                               attn_implementation=ATTN).eval()
    spec = opt_spec(tgt) if tgt.config.model_type == "opt" else gpt2_spec(tgt)
    host = learned_abs_host(spec)
    prompts = _prompts(tok, bos=spec["bos"], n=n_prompts)
    rows = []
    for dt in dtypes:
        t, hm = tgt.to(device=device, dtype=dt), host.to(device=device, dtype=dt)
        with exact_norms() if dt == F64 else contextlib.nullcontext():
            r = agreement(t, hm, prompts, n_new=n_new, bos=spec["bos"])
        rows.append(dict(edge="learned_abs <- NoPE/RoPE(inv_freq=0) + BOS", target=repo,
                         host=f"LlamaForCausalLM(hidden {host.config.hidden_size}, "
                              f"{host.config.num_hidden_layers} layers, {host.config.hidden_act}, inv_freq=0)",
                         dtype=str(dt).replace("torch.", ""), **r, **host.bos_diag))
    return rows


def load_sink_target(dtype=F64):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SINK_REPO, local_files_only=True)
    tgt = AutoModelForCausalLM.from_pretrained(SINK_REPO, local_files_only=True, dtype=dtype,
                                               attn_implementation="eager", **SINK_LOAD_KW).eval()
    return tok, tgt


def measure_sink(dtypes=(F64, torch.float32), device="cpu", n_prompts=24, n_new=64):
    tok, tgt = load_sink_target()
    host = sink_host(tgt)
    prompts = _prompts(tok, story=True, n=n_prompts)
    V = tgt.config.vocab_size
    rows = []
    for dt in dtypes:
        t, hm = tgt.to(device=device, dtype=dt), host.to(device=device, dtype=dt)
        with exact_norms() if dt == F64 else contextlib.nullcontext():
            r = agreement(t, hm, prompts, n_new=n_new, prefix=torch.tensor([[host.marker_id]]), V=V)
        rows.append(dict(edge="sink <- q-bias host + spare zero-freq pair + marker", target=SINK_REPO,
                         host=f"GptOss stack at hidden {host.config.hidden_size} with stock Qwen2Attention "
                              f"(head_dim {tgt.config.head_dim + 2}, q/k/v bias, no sinks)",
                         dtype=str(dt).replace("torch.", ""), **r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="prefix,gpt2,opt,sink")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--out", default=None, help="append JSON rows here")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    only = a.only.split(",")
    rows = []
    t0 = time.time()
    if "prefix" in only:
        from transformers import AutoModelForCausalLM
        tgt = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True, dtype=F64)
        for act in ("silu", "gelu_new", "relu"):
            for dt in (F64, torch.float32):
                r = prefix_check(gpt2_spec(tgt), act=act, device=a.device, dtype=dt)
                rows.append(dict(edge="two-block prefix (GPT-2 wte / wpe), stock bias-free Llama", **r))
                print(rows[-1], file=sys.stderr, flush=True)
    for name, repo in (("gpt2", "gpt2"), ("opt", "facebook/opt-125m")):
        if name in only:
            for r in measure_learned_abs(repo, device=a.device, n_prompts=a.n_prompts):
                rows.append(r)
                print(r, file=sys.stderr, flush=True)
    if "sink" in only:
        for r in measure_sink(device=a.device, n_prompts=a.n_prompts):
            rows.append(r)
            print(r, file=sys.stderr, flush=True)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)
    if a.out:
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    main()
