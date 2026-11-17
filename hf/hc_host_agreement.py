"""Hyper-connections switched OFF on real checkpoints: a stock HC host plays a plain target (hf/hc_host_off.py has the
module-level identity; Lean EP.hc_identical_streams / EP.hc_stack).

Pairs (stock transformers 5.18 modules; every host weight written from the target's):

  hy4<-gemma-2b        unsloth/gemma-2b on a HYV4ForCausalLM (Hy4-preview's family), hc_mult = 4
  hy4<-codegemma-2b    google/codegemma-2b on the same host class
  qwen4_exp<-qwen3-0.6b  Qwen/Qwen3-0.6B on a Qwen4ExpForCausalLM (Qwen3.8-Flash-Next's family), hc_count = 4
  qwen4_exp<-qwen3-1.7b  Qwen/Qwen3-1.7B on the same host class

The HC part (the rule under test): every stream carries the target's residual -- fn / hc_fn = 0 (static mixes), pre
bases sigmoid(b) = 1/n - hc_eps, post bases sigmoid(b) = (1 - hc_eps) / hc_magnitude, the HyperHead's base 1/n; on the
Qwen4-exp host input_mix_weight_up = 0 and block_inject_weight = 0 (gates 1/2 and 1) with the per-stream (1 + w) norm
gain 2 gamma. The rest of each host is the measured construction of hf/eps_lever_agreement.py, its own (already
payable) levers:

  hy4: the MLA host (the target must be MQA: gemma-2b's one KV head is MLA's shared rotary key; the latent = v with
       kv_a_layernorm switched off by the eps lever), q_lora_rank = hidden with q_a_layernorm switched off by the
       lever, the DSA indexer keeping every key (index_topk >= context), the sinks at -1e4 (exp underflows to 0), the
       output gate 0 (sigmoid 1/2, o_proj x 2), every MLP layer dense;
  qwen4_exp: every layer an indexed-attention layer whose QSA budget keeps every key, the output gate 0 (o_proj x 2),
       the dense MLP in the shared expert (its gate 0: x 2) and the routed experts' down_proj 0, the qk-norm (1 + w).

Host config values that are not the lever are the target's (activation, theta, head_dim, eps), as in
hf/eps_lever_agreement.py. Scored with the act_swap standard (payable = top-1 >= 99% and >= 90% identical 32-token
continuations; fp32, eager).

  python -m hf.hc_host_agreement hy4<-gemma-2b --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

TF5 = os.path.expanduser("~/.cache/upstream_refs/transformers-5.18.0")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hc_host_agreement.jsonl")
PAIRS = {
    "hy4<-gemma-2b": ("unsloth/gemma-2b", "hy4"),
    "hy4<-codegemma-2b": ("google/codegemma-2b", "hy4"),
    "qwen4_exp<-qwen3-0.6b": ("Qwen/Qwen3-0.6B", "qwen4_exp"),
    "qwen4_exp<-qwen3-1.7b": ("Qwen/Qwen3-1.7B", "qwen4_exp"),
}
N_HC = 4
LEVER_A = 30                     # the eps lever's input scale 2^-a on the switched-off MLA norms (rho ~ 2^-60 ms / eps)


def logit(p):
    return math.log(p / (1 - p))


def tdims(cfg):
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    rp = getattr(cfg, "rope_parameters", None) or {}
    return dict(V=cfg.vocab_size, D=cfg.hidden_size, I=cfg.intermediate_size, L=cfg.num_hidden_layers,
                H=cfg.num_attention_heads, KV=cfg.num_key_value_heads, hd=hd, eps=cfg.rms_norm_eps,
                theta=rp.get("rope_theta", getattr(cfg, "rope_theta", 10000.0)),
                act=getattr(cfg, "hidden_activation", None) or cfg.hidden_act,
                tie=bool(getattr(cfg, "tie_word_embeddings", True)), maxpos=cfg.max_position_embeddings,
                mt=cfg.model_type)


def host_config(kind, t, hc_eps):
    rope = {"rope_type": "default", "rope_theta": t["theta"]}
    if kind == "hy4":
        from transformers import HYV4Config
        assert t["KV"] == 1, "an MLA host shares ONE rotary key head: the target must be MQA"
        c = HYV4Config(vocab_size=t["V"], hidden_size=t["D"], intermediate_size=t["I"], moe_intermediate_size=8,
                       num_hidden_layers=t["L"], num_attention_heads=t["H"], num_key_value_heads=t["H"],
                       hidden_act=t["act"], max_position_embeddings=max(t["maxpos"], 4096), rms_norm_eps=t["eps"],
                       tie_word_embeddings=False, attention_bias=False, n_routed_experts=4, n_shared_experts=1,
                       num_experts_per_tok=1, q_lora_rank=t["D"], kv_lora_rank=t["hd"], qk_nope_head_dim=0,
                       qk_rope_head_dim=t["hd"], v_head_dim=t["hd"], mlp_layer_types=["dense"] * t["L"],
                       layer_types=["indexed_attention"] * t["L"], index_topk=1 << 16, index_head_dim=t["hd"],
                       index_n_heads=1, indexer_types=["full"] * t["L"], hc_mult=N_HC, hc_magnitude=2.0,
                       hc_eps=hc_eps, swiglu_limit=1e30, rope_parameters=rope, pad_token_id=None,
                       bos_token_id=None, eos_token_id=None)
    else:
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        half = t["hd"] // 2
        sec = [half - 2 * (half // 3), half // 3, half // 3]
        rope.update(mrope_section=sec, partial_rotary_factor=1.0)
        c = Qwen4ExpTextConfig(vocab_size=t["V"], hidden_size=t["D"], num_hidden_layers=t["L"],
                               num_attention_heads=t["H"], num_key_value_heads=t["KV"], head_dim=t["hd"],
                               hidden_act=t["act"], max_position_embeddings=max(t["maxpos"], 4096),
                               rms_norm_eps=t["eps"], tie_word_embeddings=t["tie"], attention_bias=False,
                               moe_intermediate_size=8, shared_expert_intermediate_size=t["I"],
                               num_experts_per_tok=1, num_experts=2, layer_types=["indexed_attention"] * t["L"],
                               hc_count=N_HC, hc_lowrank=8, ple_layer_ids=[], indexer_n_heads=1, indexer_kv_heads=1,
                               indexer_head_dim=t["hd"], indexer_budget=1 << 16, indexer_compress_ratio=4,
                               rope_parameters=rope, mtp_num_hidden_layers=0, pad_token_id=None, bos_token_id=None,
                               eos_token_id=None)
    c._attn_implementation = "eager"
    return c


def build(kind, cfg, dtype):
    import transformers as tr
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        cls = tr.HYV4ForCausalLM if kind == "hy4" else tr.Qwen4ExpForCausalLM
        if kind == "qwen4_exp":
            from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM as cls
        return cls(cfg).eval()
    finally:
        torch.set_default_dtype(old)


def host_weights(kind, t, sd, host_names, hc_eps):
    """{host parameter: float64 tensor}."""
    T = lambda n: sd[n].double()
    gemma = t["mt"] == "gemma"
    tgain = (lambda n: T(n) + 1.0) if gemma else T
    D, L, H, hd, n = t["D"], t["L"], t["H"], t["hd"], N_HC
    out = {}
    emb = T("model.embed_tokens.weight")
    sc = math.sqrt(D) if gemma else 1.0
    out["model.embed_tokens.weight"] = emb * sc
    lm = emb if t["tie"] else T("lm_head.weight")
    down = 2.0 ** -LEVER_A
    for name in host_names:                                   # HC parameters: static mixes (the rule)
        leaf = name.split(".")[-1]
        if name.endswith(("attn_hc.fn", "ffn_hc.fn")) or leaf == "hc_fn":
            out[name] = torch.zeros(host_names[name], dtype=torch.float64)
        elif name.endswith(("attn_hc.base", "ffn_hc.base")):
            b = torch.empty(2 * n, dtype=torch.float64)
            b[:n] = logit(1.0 / n - hc_eps)
            b[n:] = logit((1.0 - hc_eps) / 2.0)
            out[name] = b
        elif name.endswith(("attn_hc.scale", "ffn_hc.scale")) or leaf == "hc_scale":
            out[name] = torch.ones(host_names[name], dtype=torch.float64)
        elif leaf == "hc_base":
            out[name] = torch.full(host_names[name], logit(1.0 / n - hc_eps), dtype=torch.float64)
    if kind == "qwen4_exp":
        def hcn(g):                                          # per-stream (1 + w) gain = 2 g
            return (2.0 * g - 1.0).repeat(n)
        out["model.hyper_connection_mixer.hc_norm.weight"] = hcn(tgain("model.norm.weight"))
        out["model.hyper_connection_mixer.input_mix_weight_down.weight"] = torch.zeros(
            host_names["model.hyper_connection_mixer.input_mix_weight_down.weight"], dtype=torch.float64)
        out["model.hyper_connection_mixer.input_mix_weight_up.weight"] = torch.zeros(
            host_names["model.hyper_connection_mixer.input_mix_weight_up.weight"], dtype=torch.float64)
    else:
        out["model.norm.weight"] = tgain("model.norm.weight")
    out["lm_head.weight"] = lm
    for i in range(L):
        p = f"model.layers.{i}."
        Wq, Wk, Wv, Wo = (T(p + f"self_attn.{m}_proj.weight") for m in "qkvo")
        Wg, Wu, Wd = (T(p + f"mlp.{m}_proj.weight") for m in ("gate", "up", "down"))
        g1, g2 = tgain(p + "input_layernorm.weight"), tgain(p + "post_attention_layernorm.weight")
        if kind == "hy4":
            out[p + "input_layernorm.weight"] = g1
            out[p + "post_attention_layernorm.weight"] = g2
            a = p + "self_attn."
            out[a + "q_a_proj.weight"] = torch.eye(D, dtype=torch.float64) * down
            out[a + "q_a_layernorm.weight"] = torch.full((D,), 2.0 ** LEVER_A * math.sqrt(1e-6), dtype=torch.float64)
            out[a + "q_b_proj.weight"] = Wq                     # (scale 1/sqrt(hd) both sides)
            out[a + "kv_a_proj_with_mqa.weight"] = torch.cat([Wv * down, Wk], 0)
            out[a + "kv_a_layernorm.weight"] = torch.full((hd,), 2.0 ** LEVER_A * math.sqrt(1e-6),
                                                           dtype=torch.float64)
            out[a + "kv_b_proj.weight"] = torch.eye(hd, dtype=torch.float64).repeat(H, 1)
            out[a + "o_proj.weight"] = Wo * 2.0                 # the output gate's sigmoid(0) = 1/2
            out[a + "gate_proj.weight"] = torch.zeros(host_names[a + "gate_proj.weight"], dtype=torch.float64)
            out[a + "sinks"] = torch.full((H,), -1e4, dtype=torch.float64)
            ix = a + "indexer."
            for leaf in ("wq_b.weight", "wk.weight", "weights_proj.weight"):
                out[ix + leaf] = torch.zeros(host_names[ix + leaf], dtype=torch.float64)
            out[ix + "k_norm.weight"] = torch.ones(host_names[ix + "k_norm.weight"], dtype=torch.float64)
            out[ix + "k_norm.bias"] = torch.zeros(host_names[ix + "k_norm.bias"], dtype=torch.float64)
            out[p + "mlp.gate_proj.weight"], out[p + "mlp.up_proj.weight"] = Wg, Wu
            out[p + "mlp.down_proj.weight"] = Wd
        else:
            for nm, g in (("attn_hyper_connection", g1), ("mlp_hyper_connection", g2)):
                q = p + nm + "."
                out[q + "hc_norm.weight"] = (2.0 * g - 1.0).repeat(n)
                for leaf in ("input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight"):
                    out[q + leaf] = torch.zeros(host_names[q + leaf], dtype=torch.float64)
            a = p + "self_attn."
            Wq_h = torch.cat([Wq.view(H, hd, D), torch.zeros(H, hd, D, dtype=torch.float64)], 1).reshape(2 * H * hd, D)
            out[a + "q_proj.weight"] = Wq_h                     # [q_h ; gate_h = 0] per head
            out[a + "k_proj.weight"], out[a + "v_proj.weight"] = Wk, Wv
            out[a + "o_proj.weight"] = Wo * 2.0
            out[a + "q_norm.weight"] = T(a + "q_norm.weight") - 1.0
            out[a + "k_norm.weight"] = T(a + "k_norm.weight") - 1.0
            ix = a + "indexer."
            out[ix + "index_qk_proj.weight"] = torch.zeros(host_names[ix + "index_qk_proj.weight"], dtype=torch.float64)
            for leaf in ("q_layernorm.weight", "k_layernorm.weight"):
                out[ix + leaf] = torch.zeros(host_names[ix + leaf], dtype=torch.float64)
            m = p + "mlp."
            out[m + "gate.weight"] = torch.zeros(host_names[m + "gate.weight"], dtype=torch.float64)
            out[m + "experts.gate_up_proj"] = torch.zeros(host_names[m + "experts.gate_up_proj"], dtype=torch.float64)
            out[m + "experts.down_proj"] = torch.zeros(host_names[m + "experts.down_proj"], dtype=torch.float64)
            out[m + "shared_expert.gate_proj.weight"], out[m + "shared_expert.up_proj.weight"] = Wg, Wu
            out[m + "shared_expert.down_proj.weight"] = Wd * 2.0  # sigmoid(0) = 1/2 shared gate
            out[m + "shared_expert_gate.weight"] = torch.zeros(host_names[m + "shared_expert_gate.weight"],
                                                                dtype=torch.float64)
    return out


def write_host(host, W, dtype):
    names = dict(host.named_parameters(remove_duplicate=False))
    missing = [k for k in names if k not in W]
    assert not missing, f"no rule for host parameters {missing[:8]}"
    with torch.no_grad():
        for k, p in names.items():
            p.data = W[k].to(dtype)


def run(pair, device, n_prompts, sink, hc_eps_list=(1e-6, 0.0)):
    from hf.eps_lever_agreement import load_target, reference, score, strict
    repo, kind = PAIRS[pair]
    t0 = time.time()
    tok, tgt = load_target(repo, torch.float32)
    t = tdims(tgt.config)
    sd = {k: v.detach().cpu() for k, v in tgt.state_dict().items()}
    tgt = tgt.to(device)
    ref = reference(tgt, tok, device, n_prompts)
    del tgt
    rows = []
    for hc_eps in hc_eps_list:
        cfg = host_config(kind, t, hc_eps)
        host = build(kind, cfg, torch.float32).to(device)
        shapes = {k: tuple(v.shape) for k, v in host.named_parameters(remove_duplicate=False)}
        write_host(host, host_weights(kind, t, sd, shapes, hc_eps), torch.float32)
        r = score(host, ref, device)
        import transformers
        r = dict(pair=pair, target=repo, host=kind, hc_mult=N_HC, hc_eps=hc_eps, lever_a=LEVER_A, dtype="float32",
                 transformers=transformers.__version__, n_prompts=n_prompts, **r)
        r["payable"] = strict(r)
        rows.append(r)
        sink(r)
        print(f"{pair} hc_eps={hc_eps:g}: top1 {r['top1']:.4f} ex32 {r['ex32']:.2f} ex64 {r['ex64']:.2f} "
              f"max|dlogp| {r['max_dlogp']:.2e} kl {r['kl']:.2e} {'PASS' if r['payable'] else 'fail'} "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        del host
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=sorted(PAIRS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if not os.environ.get("HC_TF5"):
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prelude = ("import sys, types\n"
                   "m = types.ModuleType('transformers.dependency_versions_check'); m.dep_version_check = lambda *a, **k: None\n"
                   "sys.modules['transformers.dependency_versions_check'] = m\n"
                   "sys.modules['fla'] = None\n"
                   f"sys.argv = {['hc_host_agreement'] + sys.argv[1:]!r}\n"
                   "import runpy; runpy.run_module('hf.hc_host_agreement', run_name='__main__')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([TF5, root]), HC_TF5="1")
        sys.exit(subprocess.run([sys.executable, "-c", prelude], env=env).returncode)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.pair, a.device, a.n_prompts, sink)


if __name__ == "__main__":
    main()
