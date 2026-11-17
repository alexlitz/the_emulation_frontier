"""DiffusionGemma's always-on embedding norm switched off, on real checkpoints (port sc_norm_off of graph/model_graph.py;
Lean selfcond_norm_row, rmsNorm_smul_eps_sq in ResNorm.lean; the fp64 instances are catalog fr_selfcond_norm_*).

The host is the stock transformers 5.18 DiffusionGemma DECODER (DiffusionGemmaDecoderModel's modules: the scaled
embedding, DiffusionGemmaSelfConditioning with its weight-free post_norm, the decoder layers with their dense MLP +
MoE branches and branch norms, the final norm, the tied lm_head), run as an AR host under the caller's causal mask
(the AR <= diffusion mask currency) with an empty encoder prefix and the zero self-conditioning signal (the first
denoising step). The driver below is the decoder's own forward minus the read-only encoder cache.

The rule under test: the whole residual stream scaled by alpha (= 2^-12), host rms_norm_eps = alpha^2 eps (config), the
embedding rows lambda_v e_v with lambda_v = alpha^2 sqrt(eps) / sqrt(1 - alpha^2 ms(sqrt(d) e_v)), so the always-on
embedding norm returns alpha x_0 exactly; the tied lm_head then multiplies logit v by lambda_v / (alpha^2 sqrt(eps)) =
1 / sqrt(1 - alpha^2 ms) -> 1 (a limit), the final-norm gain divided by alpha^2 sqrt(eps).

The rest is the measured eps lever of hf/eps_lever_agreement.py (gemma4<-gemma1), applied at the host eps: the q / k
norms, the weight-free v norm, the sandwich post-norms and the MoE layer's dense-branch norm switched off (writers
2^-a, gains 2^a sqrt(eps_h)); the experts' down projections 0 (post_feedforward_layernorm_2 of 0 is exactly 0); the
forced full-attention last layer (K = V: v_proj absent) packs the target's k and v into one 512-wide key head (q zero
on the v slots, o_proj reading only the v slots, the inv_freq buffer the target's frequencies on the k pairs and 0 on
the v pairs).

Targets: gemma-2b (unsloth/gemma-2b) and codegemma-2b (MQA, RoPE: every layer's KV is one head). fp32, eager, the
act_swap standard.

  python -m hf.selfcond_norm_agreement unsloth/gemma-2b       # (re-execs under transformers 5.18)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

TF5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "selfcond_norm_agreement.jsonl")
TARGETS = ("unsloth/gemma-2b", "google/codegemma-2b")
ALPHA = 2.0 ** -12
LEV = 45                                   # the eps lever's input scale 2^-LEV (rho ~ 2^-90 ms / eps_h)


class DecoderAR(nn.Module):
    """ids -> logits through the stock DiffusionGemma decoder modules, causal mask, zero self-conditioning signal."""

    def __init__(self, dec, lm_head, softcap):
        super().__init__()
        self.dec, self.lm_head, self.softcap = dec, lm_head, softcap

    def forward(self, ids, use_cache=False, past_key_values=None, **kw):
        if use_cache or past_key_values is not None:
            raise RuntimeError("no cache: greedy recomputes")
        dec = self.dec
        h = dec.embed_tokens(ids)
        h = dec.self_conditioning(h, torch.zeros_like(h))
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device)[None]
        mask = torch.full((T, T), torch.finfo(h.dtype).min, dtype=h.dtype, device=ids.device).triu(1)[None, None]
        pe = {lt: dec.rotary_emb(h, pos, lt) for lt in dec.unique_layer_types}
        for i, layer in enumerate(dec.layers):
            lt = dec.text_config.layer_types[i]
            h = layer(h, position_embeddings=pe[lt], attention_mask=mask, position_ids=pos, past_key_values=None)
        logits = self.lm_head(dec.norm(h))
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap

        class O:
            pass
        o = O()
        o.logits = logits
        return o


def build(tc, sd, dtype=torch.float32):
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaDecoderModel
    d, L, H, hd, V = tc.hidden_size, tc.num_hidden_layers, tc.num_attention_heads, tc.head_dim, tc.vocab_size
    assert tc.num_key_value_heads == 1, "MQA targets (one KV head: the packed K = V last layer)"
    eps_t = tc.rms_norm_eps
    eps_h = ALPHA ** 2 * eps_t
    theta = getattr(tc, "rope_theta", None) or (tc.rope_parameters or {}).get("rope_theta", 10000.0)
    rope = {"rope_type": "default", "rope_theta": theta}
    text = dict(vocab_size=V, hidden_size=d, intermediate_size=tc.intermediate_size, num_hidden_layers=L,
                num_attention_heads=H, num_key_value_heads=1, head_dim=hd, hidden_activation=tc.hidden_act,
                max_position_embeddings=tc.max_position_embeddings, rms_norm_eps=eps_h, tie_word_embeddings=True,
                sliding_window=1 << 16, layer_types=["sliding_attention"] * (L - 1) + ["full_attention"],
                rope_parameters={"sliding_attention": dict(rope), "full_attention": dict(rope)}, num_experts=2,
                top_k_experts=1, moe_intermediate_size=8, final_logit_softcapping=None, global_head_dim=2 * hd,
                pad_token_id=None, bos_token_id=None, eos_token_id=None)
    cfg = DiffusionGemmaConfig(text_config=text)
    cfg.text_config._attn_implementation = "eager"
    cfg._attn_implementation = "eager"
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        dec = DiffusionGemmaDecoderModel(cfg).eval()
    finally:
        torch.set_default_dtype(old)
    lm = nn.Linear(d, V, bias=False)
    T = lambda n: sd[n].double()
    g1 = lambda n: T(n) + 1.0                          # gemma-1 (1 + w) gains
    down, up = 2.0 ** -LEV, 2.0 ** LEV * math.sqrt(eps_h)
    W = {}
    E = T("model.embed_tokens.weight")
    x0 = E * math.sqrt(d)
    ms = x0.pow(2).mean(-1, keepdim=True)
    assert (ALPHA ** 2 * ms < 1).all()
    lam = ALPHA ** 2 * math.sqrt(eps_t) / torch.sqrt(1 - ALPHA ** 2 * ms)
    W["embed_tokens.weight"] = E * lam
    W["norm.weight"] = g1("model.norm.weight") / (ALPHA ** 2 * math.sqrt(eps_t))
    sc = "self_conditioning."
    for n, p in dec.named_parameters():
        if n.startswith(sc):
            W[n] = torch.ones(p.shape, dtype=torch.float64) if n.endswith("pre_norm.weight") else \
                torch.zeros(p.shape, dtype=torch.float64)
    sc_t = hd ** -0.5                                   # the target's softmax scale (host: 1.0)
    perm_k = torch.cat([torch.arange(hd // 2), 2 * hd // 2 + torch.arange(hd // 2)])     # k dims -> rotating pairs
    perm_v = torch.cat([hd // 2 + torch.arange(hd // 2), 3 * hd // 2 + torch.arange(hd // 2)])  # v dims -> frozen
    for i in range(L):
        p, q = f"model.layers.{i}.", f"layers.{i}."
        full = i == L - 1
        Wq, Wk, Wv, Wo = (T(p + f"self_attn.{m}_proj.weight") for m in "qkvo")
        a = q + "self_attn."
        if not full:
            W[a + "q_proj.weight"], W[a + "k_proj.weight"], W[a + "v_proj.weight"] = Wq * down, Wk * down, Wv * down
            W[a + "q_norm.weight"] = torch.full((hd,), up * sc_t, dtype=torch.float64)
            W[a + "k_norm.weight"] = torch.full((hd,), up, dtype=torch.float64)
            W[a + "o_proj.weight"] = Wo * up * down     # v_norm(v 2^-a) ~ v 2^-a / sqrt(eps_h); output x 2^-a again
        else:
            D2 = 2 * hd
            qh = torch.zeros(H, D2, d, dtype=torch.float64)
            qh[:, perm_k] = Wq.view(H, hd, d)
            W[a + "q_proj.weight"] = qh.reshape(H * D2, d) * down
            kh = torch.zeros(D2, d, dtype=torch.float64)
            kh[perm_k], kh[perm_v] = Wk, Wv
            W[a + "k_proj.weight"] = kh * down
            W[a + "q_norm.weight"] = torch.full((D2,), up * sc_t, dtype=torch.float64)
            W[a + "k_norm.weight"] = torch.full((D2,), up, dtype=torch.float64)
            oh = torch.zeros(d, H, D2, dtype=torch.float64)
            oh[:, :, perm_v] = Wo.view(d, H, hd)
            W[a + "o_proj.weight"] = oh.reshape(d, H * D2) * up * down
        W[q + "post_attention_layernorm.weight"] = torch.full((d,), up * ALPHA, dtype=torch.float64)
        W[q + "input_layernorm.weight"] = g1(p + "input_layernorm.weight")
        W[q + "pre_feedforward_layernorm.weight"] = g1(p + "post_attention_layernorm.weight")
        for m in ("gate", "up"):
            W[q + f"mlp.{m}_proj.weight"] = T(p + f"mlp.{m}_proj.weight")
        W[q + "mlp.down_proj.weight"] = T(p + "mlp.down_proj.weight") * down
        W[q + "post_feedforward_layernorm_1.weight"] = torch.full((d,), up * down, dtype=torch.float64)
        W[q + "post_feedforward_layernorm.weight"] = torch.full((d,), up * ALPHA, dtype=torch.float64)
    for n, prm in dec.named_parameters():
        if n not in W:
            leaf = n.split(".")[-1]
            if ".experts." in n or n.endswith("router.proj.weight"):
                W[n] = torch.zeros(prm.shape, dtype=torch.float64)
            elif n.endswith(("router.scale", "router.per_expert_scale")) or "layernorm" in n:
                W[n] = torch.ones(prm.shape, dtype=torch.float64)
            else:
                raise KeyError(n)
    with torch.no_grad():
        for n, prm in dec.named_parameters():
            prm.data = W[n].to(dtype)
        lm.weight = dec.embed_tokens.weight                # tied
        inv_t = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        full_inv = torch.zeros(hd, dtype=torch.float64)     # 512-wide head: 256 pairs, k pairs rotate, v pairs frozen
        full_inv[:hd // 2] = inv_t
        dec.rotary_emb.full_attention_inv_freq = full_inv.to(torch.float32)
        dec.rotary_emb.full_attention_original_inv_freq = full_inv.to(torch.float32).clone()
    return DecoderAR(dec, lm, None).eval()


def run(repo, device, n_prompts, sink):
    from hf.eps_lever_agreement import load_target, reference, score, strict
    t0 = time.time()
    tok, tgt = load_target(repo, torch.float32)
    tc = tgt.config
    sd = {k: v.detach().cpu() for k, v in tgt.state_dict().items()}
    ref = reference(tgt.to(device), tok, device, n_prompts)
    del tgt
    host = build(tc, sd).to(device)
    r = score(host, ref, device)
    import transformers
    r = dict(target=repo, host="diffusion_gemma decoder (AR, causal mask)", alpha=ALPHA, lever=LEV, dtype="float32",
             transformers=transformers.__version__, n_prompts=n_prompts, **r)
    r["payable"] = strict(r)
    sink(r)
    print(f"{repo}: top1 {r['top1']:.4f} ex32 {r['ex32']:.2f} ex64 {r['ex64']:.2f} max|dlogp| {r['max_dlogp']:.2e} "
          f"kl {r['kl']:.2e} {'PASS' if r['payable'] else 'fail'} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=TARGETS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if not os.environ.get("SC_TF5"):
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prelude = ("import sys, types\n"
                   "m = types.ModuleType('transformers.dependency_versions_check'); m.dep_version_check = lambda *a, **k: None\n"
                   "sys.modules['transformers.dependency_versions_check'] = m\n"
                   "sys.modules['fla'] = None\n"
                   f"sys.argv = {['selfcond_norm_agreement'] + sys.argv[1:]!r}\n"
                   "import runpy; runpy.run_module('hf.selfcond_norm_agreement', run_name='__main__')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, root]), SC_TF5="1")
        sys.exit(subprocess.run([sys.executable, "-c", prelude], env=env).returncode)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.target, a.device, a.n_prompts, sink)


if __name__ == "__main__":
    main()
