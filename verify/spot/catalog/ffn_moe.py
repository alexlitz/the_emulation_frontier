"""FFN kinds, activations, MoE routing and host-side feature switch-offs: stock MoE classes (mixtral, qwen2_moe,
phimoe, deepseek_v2 / v3) where they exist, elephantine/emulate.py for the dense -> MoE hosts, the reference decoder
(verify/spot/catalog/_ref.py) for the rest."""
from __future__ import annotations

import math

import torch

from verify.spot.catalog import compare, entry, hf_fn
from verify.spot.catalog._emu import emu_check
from verify.spot.catalog._ref import F64, RefLM, fn
from verify.spot.catalog.ref_slots import REF, _check, _host, _t

V = 61


# ------------------------------------------------------------------ stock MoE helpers

def _stock(mt, seed=0, noise=0.08, **kw):
    from transformers import AutoConfig, AutoModelForCausalLM
    base = dict(vocab_size=V, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False)
    base.update(kw)
    c = AutoConfig.for_model(mt, **base)
    c._attn_implementation = "eager"
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(c).to(F64).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * noise)
    return m


def _zero_host(mt, **kw):
    h = _stock(mt, seed=1, **kw)
    with torch.no_grad():
        for p in h.parameters():
            p.zero_()
    return h


def _pad_into(dst, src):
    """dst[:src.shape] = src (zero-padded)."""
    sl = tuple(slice(0, s) for s in src.shape)
    dst[sl] = src


@torch.no_grad()
def _moe_dup(t, h, m, expert_key, router_key, names=("gate_proj", "up_proj", "down_proj"), down_scale=1.0,
             extra_bias=None):
    """Copy every shared tensor; each target expert e on host experts c E_t + e (c < m copies), the router rows copied
    the same way, the widths zero-padded; host experts beyond m E_t get a selection bias extra_bias (if given)."""
    ts, hs = t.state_dict(), h.state_dict()
    E_t = sum(1 for k in ts if k.startswith("model.layers.0.") and f"{expert_key}." in k and k.endswith(
        f".{names[0]}.weight") and "shared" not in k)
    for k, v in ts.items():
        if f".{expert_key}." in k or k.endswith(router_key + ".weight") or "e_score_correction_bias" in k:
            continue
        if k in hs:
            _pad_into(hs[k], v)
    for k, v in ts.items():
        if k.endswith(router_key + ".weight"):
            for c in range(m):
                hs[k][c * E_t:(c + 1) * E_t] = v
        if "e_score_correction_bias" in k:
            for c in range(m):
                hs[k][c * E_t:(c + 1) * E_t] = v
            if extra_bias is not None:
                hs[k][m * E_t:] = extra_bias
        if f".{expert_key}." in k:
            pre, rest = k.split(f".{expert_key}.")
            e, name = rest.split(".", 1)
            for c in range(m):
                hk = f"{pre}.{expert_key}.{c * E_t + int(e)}.{name}"
                _pad_into(hs[hk], v * (down_scale if name.startswith(names[2]) else 1.0))


def _cmp(t, h, standard="exact", tol=1e-9, note=""):
    from tests.test_universal_elephantine import _lift_fp32
    with _lift_fp32():
        return compare(hf_fn(t), hf_fn(h), V, standard=standard, tol=tol, note=note)


@entry("moe:dup:R:m", instance="mixtral E 4, top-2 (renormalised) -> mixtral E 8, top-4: each expert twice")
def fm_mixtral_dup2():
    t = _stock("mixtral", num_local_experts=4, num_experts_per_tok=2)
    h = _zero_host("mixtral", num_local_experts=8, num_experts_per_tok=4)
    _moe_dup(t, h, 2, "experts", "block_sparse_moe.gate", ("w1", "w3", "w2"))
    return _cmp(t, h)


@entry("moe:dup:R:1", instance="mixtral E 4, top-2, expert width 48 -> mixtral E 4, top-2, width 64 (zero-padded)")
def fm_mixtral_wider():
    t = _stock("mixtral", num_local_experts=4, num_experts_per_tok=2)
    h = _zero_host("mixtral", num_local_experts=4, num_experts_per_tok=2, intermediate_size=64)
    _moe_dup(t, h, 1, "experts", "block_sparse_moe.gate", ("w1", "w3", "w2"))
    return _cmp(t, h)


_QM = dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=24,
           norm_topk_prob=False)


@entry("moe:dup:P:m", instance="qwen2_moe E 4, top-2, softmax not renormalised -> E 8 (exactly 2 x), top-4")
def fm_qwen2moe_P_dup2():
    t = _stock("qwen2_moe", **_QM)
    h = _zero_host("qwen2_moe", **{**_QM, "num_experts": 8, "num_experts_per_tok": 4})
    _moe_dup(t, h, 2, "experts", "mlp.gate")
    return _cmp(t, h)


@entry("moe:dup:P:1", "moe:shared", "feature:shared_gate:host",
       instance="qwen2_moe (P, expert 16, shared 24) -> qwen2_moe (expert 24, shared 40): widths zero-padded, the "
                "shared expert's sigmoid gate copied")
def fm_qwen2moe_P_wider():
    t = _stock("qwen2_moe", **_QM)
    h = _zero_host("qwen2_moe", **{**_QM, "moe_intermediate_size": 24, "shared_expert_intermediate_size": 40})
    _moe_dup(t, h, 1, "experts", "mlp.gate")
    return _cmp(t, h)


@entry("slot:ffn:moe->moe_shared", instance="mixtral (E 4, top-2) -> qwen2_moe (E 4, top-2, renormalised, a gated "
                                            "shared expert set to 0)")
def fm_moe_on_shared_host():
    t = _stock("mixtral", num_local_experts=4, num_experts_per_tok=2)
    h = _zero_host("qwen2_moe", **{**_QM, "norm_topk_prob": True, "moe_intermediate_size": 48},
                   rope_theta=t.config.rope_theta, rms_norm_eps=t.config.rms_norm_eps)
    ts, hs = t.state_dict(), h.state_dict()
    with torch.no_grad():
        for k, v in ts.items():
            hk = (k.replace("block_sparse_moe.gate", "mlp.gate").replace("block_sparse_moe.experts", "mlp.experts")
                  .replace(".w1.", ".gate_proj.").replace(".w3.", ".up_proj.").replace(".w2.", ".down_proj."))
            _pad_into(hs[hk], v)
    return _cmp(t, h)


@entry("moe:dup:sparsemixer:m", instance="phimoe E 4, top-2 (SparseMixer) -> phimoe E 8, top-4: each expert twice",
       finding="SparseMixer is a top-2 in code whatever num_experts_per_tok says, and its second pick masks out only "
               "the first expert's index: a duplicate of the top-1 expert wins the second slot. moe_why let a "
               "sparsemixer target take m > 1 copies (no zoo pair used it: latent); round 3, batch 3a requires m = 1 "
               "(spot_phimoe_dup), and this failing build is that wall's evidence (catalog WALLS)")
def fm_phimoe_dup2():
    t = _stock("phimoe", num_local_experts=4, num_experts_per_tok=2)
    h = _zero_host("phimoe", num_local_experts=8, num_experts_per_tok=4)
    _moe_dup(t, h, 2, "experts", "block_sparse_moe.gate", ("w1", "w3", "w2"))
    return _cmp(t, h)


@entry("moe:dup:sparsemixer:1", instance="phimoe E 4, top-2, width 48 -> phimoe E 4, top-2, width 64")
def fm_phimoe_wider():
    t = _stock("phimoe", num_local_experts=4, num_experts_per_tok=2)
    h = _zero_host("phimoe", num_local_experts=4, num_experts_per_tok=2, intermediate_size=64)
    _moe_dup(t, h, 1, "experts", "block_sparse_moe.gate", ("w1", "w3", "w2"))
    return _cmp(t, h)


_DS3 = dict(n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, n_group=2, topk_group=1,
            first_k_dense_replace=1, kv_lora_rank=16, q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=8,
            v_head_dim=8, n_shared_experts=1, norm_topk_prob=True, routed_scaling_factor=2.5, num_key_value_heads=4)


@entry("moe:group:top2sum", "moe:extra_experts", "moe:dense_layers", "moe:score", "feature:router_sigmoid:target",
       "feature:router_sigmoid:host", "feature:router_group:target", "feature:router_group:host",
       instance="deepseek_v3 (E 4 in 2 groups, top-2, sigmoid, top-2-sum groups, 1 dense layer) -> deepseek_v3 (E 6 "
                "in 2 groups: each group's extra expert at selection bias -1e4; wider experts and dense layer)")
def fm_dsv3_groups_extra():
    t = _stock("deepseek_v3", **_DS3)
    h = _zero_host("deepseek_v3", **{**_DS3, "n_routed_experts": 6, "moe_intermediate_size": 24,
                                     "intermediate_size": 64})
    ts, hs = t.state_dict(), h.state_dict()
    with torch.no_grad():
        for k, v in ts.items():
            if ".mlp.experts." in k or k.endswith("mlp.gate.weight") or "e_score_correction_bias" in k:
                continue
            _pad_into(hs[k], v)
        # target groups {0, 1}, {2, 3} -> host groups {0, 1, 2}, {3, 4, 5}: expert e of group g at 3 g + e % 2
        mp = {e: 3 * (e // 2) + e % 2 for e in range(4)}
        for k, v in ts.items():
            if k.endswith("mlp.gate.weight"):
                for e, he in mp.items():
                    hs[k][he] = v[e]
            if "e_score_correction_bias" in k:
                hs[k].fill_(-1e4)
                for e, he in mp.items():
                    hs[k][he] = v[e]
            if ".mlp.experts." in k:
                pre, rest = k.split(".mlp.experts.")
                e, name = rest.split(".", 1)
                _pad_into(hs[f"{pre}.mlp.experts.{mp[int(e)]}.{name}"], v)
    return _cmp(t, h)


_DS2 = dict(n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, n_group=2, topk_group=1,
            first_k_dense_replace=0, kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=8, qk_nope_head_dim=8,
            v_head_dim=8, n_shared_experts=1, topk_method="group_limited_greedy", num_key_value_heads=4)


@entry("moe:group:max", instance="deepseek_v2 (E 4 in 2 groups, group-limited greedy = max) -> deepseek_v2, wider "
                                 "experts (24)")
def fm_dsv2_group_max():
    t = _stock("deepseek_v2", **_DS2)
    h = _zero_host("deepseek_v2", **{**_DS2, "moe_intermediate_size": 24})
    _moe_dup(t, h, 1, "experts", "mlp.gate")
    return _cmp(t, h)


@entry("capacity:mla_latent", instance="deepseek_v2 MLA (kv latent 16) -> deepseek_v2 (kv latent 32 = 2 copies: the "
                                       "latent norm's rms over two copies is the rms of one, same eps)")
def fm_mla_latent_multiple():
    t = _stock("deepseek_v2", **_DS2)
    h = _zero_host("deepseek_v2", **{**_DS2, "kv_lora_rank": 32})
    ts, hs = t.state_dict(), h.state_dict()
    r, kl = _DS2["qk_rope_head_dim"], 16
    with torch.no_grad():
        for k, v in ts.items():
            if "kv_a_proj_with_mqa" in k:
                hs[k][:kl], hs[k][kl:2 * kl], hs[k][2 * kl:] = v[:kl], v[:kl], v[kl:]
            elif "kv_a_layernorm" in k:
                hs[k][:kl], hs[k][kl:] = v, v
            elif "kv_b_proj" in k:
                hs[k][:, :kl] = v
            else:
                _pad_into(hs[k], v)
    return _cmp(t, h)


# ------------------------------------------------------------------ dense <-> MoE through emulate.py

@entry("slot:ffn:glu->moe", "moe:host_for_dense",
       instance="llama (dense SwiGLU) -> mixtral (4 experts, top-2): every expert = the dense FFN, router 0 "
                "(emulate.py)")
def fm_dense_on_mixtral():
    return emu_check("llama", {}, "mixtral", dict(intermediate_size=112, num_local_experts=4, num_experts_per_tok=2,
                                                  sliding_window=None))


@entry("slot:ffn:glu->moe_shared", "feature:shared_gate:host", "moe:host_for_dense", "feature:bias_qkv:host",
       instance="llama -> qwen2_moe: the dense FFN in the shared expert (sigmoid gate 0 -> x 2), routed experts 0, "
                "q / k / v biases 0 (emulate.py)")
def fm_dense_on_qwen2moe_shared():
    return emu_check("llama", {}, "qwen2_moe", dict(head_dim=16, moe_intermediate_size=48,
                                                    shared_expert_intermediate_size=128, num_experts=4,
                                                    num_experts_per_tok=2))


@entry("feature:bias_out:host", "feature:tied_emb:target",
       instance="llama (tied, no biases) -> llama with attention and MLP biases (set 0), untied head (emulate.py)")
def fm_bias_out_host():
    return emu_check("llama", {}, "llama", dict(attention_bias=True, mlp_bias=True))


@entry("feature:bias_qkv:host", instance="llama -> qwen2 (q / k / v biases set 0) (emulate.py)")
def fm_bias_qkv_host():
    return emu_check("llama", {}, "qwen2", {})


# ------------------------------------------------------------------ activations (limits / regimes)

def _act_scale(t_act, h_act, beta, ffn="vanilla", tol=1e-7):
    """target act(x) <- host act_h(beta x) / beta: the gate (vanilla: up) rows x beta, down / beta."""
    t = _t(ffn=ffn, act=t_act)
    h = _host(t, act=h_act)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        key = "fgate" if ffn == "glu" else "up"
        h.p[pre + key] = t.p[pre + key] * beta
        h.p[pre + "down"] = t.p[pre + "down"] / beta
    return _check(t, h, standard="limit", tol=tol, note=f"beta = {beta:g}")


@entry("slot:act:ReLU->SiLU", instance=f"ReLU MLP -> SiLU MLP: SiLU(beta x) / beta, beta = 1e9 ({REF})",
       standard="limit")
def fm_act_relu_silu():
    return _act_scale("relu", "silu", 1e9)


@entry("slot:act:ReLU->GeLU", instance=f"ReLU MLP -> GeLU (erf) MLP, beta = 1e9 ({REF})", standard="limit")
def fm_act_relu_gelu():
    return _act_scale("relu", "gelu", 1e9)


@entry("slot:act:ReLU->GeLU_tanh", instance=f"ReLU MLP -> GeLU (tanh) MLP, beta = 1e9 ({REF})", standard="limit")
def fm_act_relu_gelu_tanh():
    return _act_scale("relu", "gelu_tanh", 1e9)


@entry("slot:act:ReLU->mish", instance=f"ReLU MLP -> mish MLP, beta = 1e9 ({REF})", standard="limit")
def fm_act_relu_mish():
    return _act_scale("relu", "mish", 1e9)


@entry("slot:act:ReLU->swigluoai", instance=f"ReLU MLP -> swigluoai GLU (gpt-oss: gate <= 7, |up| <= 7, gate and up "
                                            f"biases): gate held at 7 by its bias, the up clamp's lower kink at the "
                                            f"target's kink, slope s = 1e-3 ({REF})", standard="limit")
def fm_act_relu_swigluoai():
    """swigluoai(g, u) = (clamp(u, -7, 7) + 1) g sigmoid(1.702 g), g = min(g, 7). With g = 7 (a gate bias) the gate
    factor is c = 7 sigmoid(11.9); u = s x - 7 makes clamp(u) + 7 = s relu(x) while s x <= 14: the target's relu(x)
    is ((swigluoai - (-6) c) / (s c)), the constant into the down bias. Exact on |x| <= 14 / s (the clamp regime),
    not the beta -> inf limit ACT_LIMIT names (the gate's clamp at 7 forbids scaling the gate up)."""
    t = _t(ffn="vanilla", act="relu", out_bias=True)
    h = _host(t, ffn="glu", act="swigluoai", out_bias=True)
    s, c = 1e-3, 7.0 / (1 + math.exp(-1.702 * 7.0))
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        h.p[pre + "fgate"] = torch.zeros_like(t.p[pre + "up"])
        h.p[pre + "fgateb"] = torch.full((t.c["F"],), 7.0, dtype=F64)
        h.p[pre + "up"] = t.p[pre + "up"] * s
        h.p[pre + "upb"] = t.p[pre + "upb"] * s - 7.0
        h.p[pre + "down"] = t.p[pre + "down"] / (s * c)
        h.p[pre + "downb"] = t.p[pre + "downb"] + t.p[pre + "down"].sum(1) * (6.0 / s)
    return _check(t, h, standard="limit", tol=1e-9, note="exact inside the clamp regime |x| <= 14 / s")


@entry("slot:act:SiLU->swigluoai", "port:oai_bias", instance=f"SiLU GLU -> swigluoai GLU: gate / 1.702, up scaled by 1e-3 with an up "
                                            f"bias -1 inside the clamps ({REF})", standard="limit")
def fm_act_silu_swigluoai():
    t = _t(ffn="glu", act="silu")
    h = _host(t, act="swigluoai", out_bias=True)
    eps = 1e-3
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        h.p[pre + "fgate"] = t.p[pre + "fgate"] / 1.702
        h.p[pre + "up"] = t.p[pre + "up"] * eps
        h.p[pre + "upb"] = -torch.ones(t.c["F"], dtype=F64)
        h.p[pre + "down"] = t.p[pre + "down"] * 1.702 / eps
        h.p[pre + "ob"], h.p[pre + "downb"] = torch.zeros(t.c["d"], dtype=F64), torch.zeros(t.c["d"], dtype=F64)
    return _check(t, h, standard="limit", tol=1e-9, note="exact while |gate| < 11.9 (the clamp regime)")


@entry("slot:act:swigluoai->SiLU", instance=f"swigluoai GLU -> SiLU GLU with an up bias +1: gate x 1.702, exact while "
                                            f"the target's clamps do not bind ({REF})", standard="limit")
def fm_act_swigluoai_silu():
    t = _t(ffn="glu", act="swigluoai")
    h = _host(t, act="silu", out_bias=True)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        h.p[pre + "fgate"] = t.p[pre + "fgate"] * 1.702
        h.p[pre + "upb"] = torch.ones(t.c["F"], dtype=F64)
        h.p[pre + "down"] = t.p[pre + "down"] / 1.702
        h.p[pre + "ob"], h.p[pre + "downb"] = torch.zeros(t.c["d"], dtype=F64), torch.zeros(t.c["d"], dtype=F64)
    return _check(t, h, standard="limit", tol=1e-9, note="the clamp regime")


@entry("slot:act:relu2->xielu", instance=f"relu^2 MLP -> xIELU MLP: xIELU(c x) / (alpha_p c^2), c = 1e8 ({REF})",
       standard="limit")
def fm_act_relu2_xielu():
    t = _t(ffn="vanilla", act="relu2")
    h = _host(t, act="xielu")
    c = 1e8
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        h.p[pre + "up"] = t.p[pre + "up"] * c
        h.p[pre + "down"] = t.p[pre + "down"] / (0.8 * c * c)
    return _check(t, h, standard="limit", tol=1e-6, note=f"c = {c:g}")


# ------------------------------------------------------------------ FFN kinds on the reference decoder

@entry("slot:ffn:vanilla->glu", instance=f"vanilla GeLU MLP -> GeGLU host with an up bias: the up path held at 1 ({REF})")
def fm_ffn_vanilla_glu():
    t = _t(ffn="vanilla", act="gelu")
    h = _host(t, ffn="glu", out_bias=True)
    for i in range(t.n_param_layers()):
        pre = f"l{i}."
        h.p[pre + "fgate"] = t.p[pre + "up"].clone()
        h.p[pre + "up"] = torch.zeros_like(t.p[pre + "up"])
        h.p[pre + "upb"] = torch.ones(t.c["F"], dtype=F64)
        h.p[pre + "ob"], h.p[pre + "downb"] = torch.zeros(t.c["d"], dtype=F64), torch.zeros(t.c["d"], dtype=F64)
    return _check(t, h)


@entry("slot:ffn:none->vanilla", instance=f"a block with no FFN (the pure Mamba stacks) -> a vanilla GeLU MLP host, "
                                         f"its down_proj 0 (round 3, mamba_ffn_none; Lean zeroed_sublayer) ({REF})")
def fm_ffn_none_vanilla():
    t = _t(ffn="none")
    h = _host(t, ffn="vanilla", act="gelu")
    tmp = RefLM(**h.c).init(3)                     # the host MLP's own (random) weights, then its output zeroed
    for k, v in tmp.p.items():
        if k not in h.p:
            h.p[k] = v
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.down"] = torch.zeros_like(h.p[f"l{i}.down"])
    return _check(t, h)


def _ffn_none_on(**cfg):
    """A no-FFN target on a host whose FFN (its own random weights) writes 0: down_proj / every expert's and the shared
    expert's down projections 0 (Lean zeroed_sublayer)."""
    t = _t(ffn="none")
    h = _host(t, **cfg)
    tmp = RefLM(**h.c).init(3)
    for k, v in tmp.p.items():
        if k not in h.p:
            h.p[k] = v
    for i in range(t.n_param_layers()):
        for key in ("down", "ed", "sd"):
            if f"l{i}.{key}" in h.p:
                h.p[f"l{i}.{key}"] = torch.zeros_like(h.p[f"l{i}.{key}"])
    return _check(t, h)


# (release, with fr-mixer's pure_on_hybrid: a pure Mamba stack now also sits on GLU / MoE hybrid hosts)
@entry("slot:ffn:none->glu", instance=f"a block with no FFN -> a SwiGLU MLP host, its down_proj 0 (Lean zeroed_sublayer) "
                                     f"({REF})")
def fm_ffn_none_glu():
    return _ffn_none_on(ffn="glu", act="silu")


@entry("slot:ffn:none->moe", instance=f"a block with no FFN -> a top-2 MoE host, every expert's down projection 0 "
                                     f"(Lean zeroed_sublayer) ({REF})")
def fm_ffn_none_moe():
    return _ffn_none_on(ffn="glu", act="silu", E=4, k=2, ew=16, F=48)


# ------------------------------------------------------------------ MoE on the reference decoder

_RM = dict(E=4, k=2, ew=16, F=48)


@entry("moe:latent", instance=f"MoE with an expert-input latent 12 -> latent 16 (zero-padded) ({REF})")
def fm_moe_latent():
    t = _t(**_RM, latent=12)
    h = RefLM(**{**t.c, "latent": 16}).init()
    h.p = {k: torch.zeros_like(v) for k, v in h.p.items()}
    for k, v in t.p.items():
        if k.endswith(("eg", "eu")):
            h.p[k][:, :, :12] = v
        elif k.endswith("ed"):
            h.p[k][:, :12] = v
        elif k.endswith("lat_in"):
            h.p[k][:12] = v
        elif k.endswith("lat_out"):
            h.p[k][:, :12] = v
        else:
            h.p[k] = v.clone()
    return _check(t, h)


@entry("feature:router_sqrtsoftplus:target", "feature:router_sqrtsoftplus:host",
       instance=f"sqrt(softplus) router (DeepSeek-V4) -> a host whose score is a config value, set to the same; a "
                f"softmax target on that host likewise ({REF})")
def fm_router_sqrtsoftplus():
    t = _t(**_RM, score="sqrtsoftplus")
    r1 = _check(t, _host(t))
    t2 = _t(**_RM, score="softmax")
    r2 = _check(t2, _host(t2, sel_bias=False))
    return r1 if not r1.passed else r2


@entry("feature:router_bias:host", instance=f"sigmoid router without a selection bias -> a host with one, set 0 ({REF})")
def fm_router_bias_host():
    t = _t(**_RM, score="sigmoid")
    h = _host(t, sel_bias=True)
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.selb"] = torch.zeros(4, dtype=F64)
    return _check(t, h)


@entry("moe:dense_layers", instance=f"a MoE target whose layer 0 is dense -> an all-MoE host: layer 0's experts all "
                                    f"hold the dense FFN, router 0 (k experts at weight 1 / k) ({REF})")
def fm_moe_as_dense():
    t = _t(**{**_RM, "ew": 48}, dense_layers=(0,))
    h = RefLM(**{**t.c, "dense_layers": ()}).init()
    h.p = {k: torch.zeros_like(v) for k, v in h.p.items()}
    for k, v in t.p.items():
        if k.startswith("l0.") and k.split(".")[1] in ("up", "fgate", "down"):
            continue
        h.p[k] = v.clone()
    for e in range(4):
        h.p["l0.eg"][e] = t.p["l0.fgate"]
        h.p["l0.eu"][e] = t.p["l0.up"]
        h.p["l0.ed"][e] = t.p["l0.down"]
    return _check(t, h)


# ------------------------------------------------------------------ host branches switched off

def _off(flag, zero_keys, note):
    t = _t()
    h = _host(t, **{flag: True})
    tmp = RefLM(**h.c).init(3)
    for k, v in tmp.p.items():
        if k not in h.p:
            h.p[k] = v
    for k in list(h.p):
        if any(k.endswith(z) for z in zero_keys) and h.p[k] is not None:
            h.p[k] = torch.zeros_like(h.p[k])
    return h, t


@entry("feature:out_gate:host", instance=f"no output gate -> a sigmoid output gate with W_g = 0 (0.5, folded into o) "
                                         f"({REF})")
def fm_out_gate_off():
    h, t = _off("out_gate", ("ogate",), "")
    for i in range(t.n_param_layers()):
        h.p[f"l{i}.o"] = t.p[f"l{i}.o"] * 2
    return _check(t, h)


@entry("feature:value_emb:host", instance=f"-> value embeddings, table 0 ({REF})")
def fm_value_emb_off():
    h, t = _off("value_emb", ("ve",), "")
    return _check(t, h)


@entry("feature:ple:host", instance=f"-> per-layer embedding inputs, projections 0 ({REF})")
def fm_ple_off():
    h, t = _off("ple", ("ple_w",), "")
    return _check(t, h)


@entry("feature:engram:host", instance=f"-> a hashed n-gram memory added to the residual, table 0 ({REF})")
def fm_engram_off():
    h, t = _off("engram", ("eng",), "")
    return _check(t, h)


@entry("feature:switch_lora:host", instance=f"-> a switched LoRA adapter, B = 0 ({REF})")
def fm_switch_lora_off():
    h, t = _off("switch_lora", ("lora_b",), "")
    return _check(t, h)


@entry("feature:selfcond:host", instance=f"-> a self-conditioning input (previous logits), projection 0 ({REF})")
def fm_selfcond_off():
    h, t = _off("selfcond", ("sc",), "")
    prev = torch.randn(1, 10, t.c["V"], dtype=F64)
    return compare(fn(t), lambda ids: h(ids, prev_logits=torch.randn(1, ids.shape[1], t.c["V"], dtype=F64)), V)


@entry("feature:dsa:target", "feature:dsa:host",
       instance=f"DSA (index_topk 64 >= the context) <-> dense attention, both ways ({REF}; DSA is dense attention "
                f"while the context fits index_topk)")
def fm_dsa():
    t = _t(dsa_topk=64)
    r1 = _check(t, _host(t, dsa_topk=None))
    t2 = _t()
    r2 = _check(t2, _host(t2, dsa_topk=64))
    return r1 if not r1.passed else r2


# ------------------------------------------------------------------ fr-moe: constant-channel router constructions

def _rf_cmp(t, h, standard="limit", tol=1e-12):
    from tests.test_universal_elephantine import _lift_fp32
    with _lift_fp32():
        return compare(hf_fn(t), hf_fn(h), t.config.vocab_size, standard=standard, tol=tol)


@entry("moe:null_experts", instance="glm4_moe (sigmoid + selection bias, top-2 of 4) -> Glm4Moe top-6 of 10, h + 1: "
                                    "2 copies of each expert + 2 null experts (bias +10, router -1e9 on the constant "
                                    "channel; moe/router_frontier.py)", standard="limit")
def fm_null_experts():
    from moe import router_frontier as rf
    t, tc = rf._tiny("glm4_moe", 0, n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=24,
                     first_k_dense_replace=1, head_dim=12, n_shared_experts=1, norm_topk_prob=True, n_group=1,
                     topk_group=1, routed_scaling_factor=1.7, partial_rotary_factor=0.5, attention_bias=True,
                     rms_norm_eps=1e-5)
    h = rf.glm4_host(rf.Glm4View(rf._sd_get(t), tc), rf.Plan(m=2, r=2), dtype=F64)
    return _rf_cmp(t, h)


@entry("moe:null_experts", instance="dots1 (sigmoid + bias, per-head q / k norms, top-1 of 4) -> Glm4Moe with "
                                    "use_qk_norm, top-3 of 9 (2 copies + 1 null), h + 1, q / k rows x sqrt(eps' / eps)",
       standard="limit")
def fm_null_experts_qknorm():
    from moe import router_frontier as rf
    t, tc = rf._tiny("dots1", 0, n_routed_experts=4, num_experts_per_tok=1, moe_intermediate_size=24,
                     first_k_dense_replace=1, n_shared_experts=1, norm_topk_prob=True, routed_scaling_factor=2.5,
                     rms_norm_eps=1e-5)
    h = rf.glm4_host(rf.Dots1View(rf._sd_get(t), tc), rf.Plan(m=2, r=1), dtype=F64)
    return _rf_cmp(t, h)


@entry("moe:null_experts", instance="the null experts with M = 0 (router row 0 on the channel): sigma(0) = 1/2 enters "
                                    "the renormalised sum -- the construction needs the sunk score",
       standard="limit", finding="without the -M entry the nulls take gate mass (moeR_null_renorm: the factor "
                                 "m A / (m A + B), B = r / 2): the host is not the target")
def fm_null_experts_unsunk():
    from moe import router_frontier as rf
    t, tc = rf._tiny("glm4_moe", 0, n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=24,
                     first_k_dense_replace=1, head_dim=12, n_shared_experts=1, norm_topk_prob=True, n_group=1,
                     topk_group=1, routed_scaling_factor=1.7, partial_rotary_factor=0.5, attention_bias=True,
                     rms_norm_eps=1e-5)
    h = rf.glm4_host(rf.Glm4View(rf._sd_get(t), tc), rf.Plan(m=2, r=2, M=0.0), dtype=F64)
    return _rf_cmp(t, h)


@entry("moe:plain_extras_sink", instance="qwen2_moe (softmax, norm_topk_prob False, E 4, a gated shared expert) -> "
                                         "qwen2_moe E 7, h + 8 (2 zero dims per head, inv_freq written): the 3 extras "
                                         "read -1e9 on the constant channel", standard="limit")
def fm_plain_extras_sink_qwen2moe():
    from moe import router_frontier as rf
    t, tc = rf._tiny("qwen2_moe", 0, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=24,
                     shared_expert_intermediate_size=32, norm_topk_prob=False, rms_norm_eps=1e-6)
    return _rf_cmp(t, rf.qwen2moe_host(rf._sd_get(t), tc, 3, e=8, dtype=F64))


@entry("moe:plain_extras_sink", instance="olmoe (softmax, norm_topk_prob False, full-width q / k norms, E 4) -> olmoe "
                                         "E 7, h + 8: q / k rows x s keep the norms' eps", standard="limit")
def fm_plain_extras_sink_olmoe():
    from moe import router_frontier as rf
    t, tc = rf._tiny("olmoe", 0, num_experts=4, num_experts_per_tok=2, norm_topk_prob=False, rms_norm_eps=1e-5,
                     num_key_value_heads=4)
    return _rf_cmp(t, rf.olmoe_host(rf._sd_get(t), tc, 3, e=8, dtype=F64))
